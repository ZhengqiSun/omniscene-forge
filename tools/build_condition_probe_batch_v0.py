#!/usr/bin/env python3
"""Build the frozen R4-val condition-probe batch (kind=condition_probe_batch_v1).

Selects N canonical-val windows (sha256 deterministic, per-match round-robin,
sidecar-covered only), assigns each a different-match partner from the selected
set, and materialises the weight-independent tensors (xt/timestep/target/
cond_chunk/cam_chunk/text_context) plus records and pickled MapMemorySample
lists. Condition tokens are NOT precomputed: the trainer probe builds them at
probe time through the current encoder/projectors (contract v1 rationale).

Sigma is stratified over the LOW band [0, 0.947) (window i gets
(i+0.5)/N * 0.947), matching decision D7' (probe stays below the expert
boundary). Noise is per-window deterministic from sha256(seed|window_id).
"""

from __future__ import annotations
from runtime_paths import source_path

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

TRAINER_PATH = Path(__file__).resolve().parent / "train_memory_dense_adapter_interaction_v1.py"
LOW_BOUNDARY = 0.947


def load_trainer_module() -> Any:
    sys.path.insert(0, str(TRAINER_PATH.parent))
    spec = importlib.util.spec_from_file_location("probe_trainer_mod", TRAINER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def stable_hex(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}|{key}".encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map-manifest", type=Path, required=True)
    ap.add_argument("--cache-manifest", type=Path, required=True)
    ap.add_argument("--extra-map-manifest", type=Path, action="append", default=[])
    ap.add_argument("--extra-cache-manifest", type=Path, action="append", default=[])
    ap.add_argument("--state-cache-manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-windows", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--latent-frames", type=int, default=21)
    ap.add_argument("--video-frames", type=int, default=81)
    ap.add_argument("--raw-stride", type=int, default=2)
    ap.add_argument("--video-height", type=int, default=480)
    ap.add_argument("--video-width", type=int, default=832)
    ap.add_argument("--vae-stride", type=int, default=8)
    ap.add_argument("--patch-size-hw", type=int, default=2)
    ap.add_argument("--split-key", default="match")
    ap.add_argument("--require-backend-id", default="bsp_faces_disp_gpu")
    ap.add_argument("--lingbot-repo", type=Path,
                    default=Path(str(source_path('lingbot', ''))),
                    help="wan package root (prepare_cam_chunk imports wan.utils.cam_utils)")
    args = ap.parse_args()

    trainer = load_trainer_module()
    trainer.add_lingbot_path(args.lingbot_repo)
    targs = argparse.Namespace(
        map_manifest=args.map_manifest,
        cache_manifest=args.cache_manifest,
        extra_map_manifest=list(args.extra_map_manifest),
        extra_cache_manifest=list(args.extra_cache_manifest),
        state_cache_manifest=args.state_cache_manifest,
        train_split="val",
        split_key=args.split_key,
        latent_frames=args.latent_frames,
        video_frames=args.video_frames,
        raw_stride=args.raw_stride,
        video_height=args.video_height,
        video_width=args.video_width,
        vae_stride=args.vae_stride,
        patch_size_hw=args.patch_size_hw,
        require_backend_id=args.require_backend_id,
        allow_static_dense_repeat=False,
        limit=None,
        log_loader_stages=False,
    )

    manifest_pairs = trainer.manifest_pairs_from_args(targs)
    state_rows = trainer.load_state_manifest_for_args(targs)
    if not state_rows:
        raise SystemExit("state cache manifest yielded no rows; probe needs sidecar coverage")

    pool: list[tuple[dict[str, Any], int]] = []
    for release_index, (map_manifest, cache_manifest) in enumerate(manifest_pairs):
        map_sha = trainer.sha256_file(map_manifest)
        records, needed_ids, _report = trainer.load_aligned_cache_records_lightweight(
            cache_manifest, args=targs, release_index=release_index, map_manifest_sha256=map_sha,
        )
        print(json.dumps({"event": "val_records_loaded", "release_index": release_index,
                          "cache_manifest": str(cache_manifest), "val_records": len(records)}), flush=True)
        for record in records:
            pool.append((record, release_index))

    covered = [
        (record, ridx) for record, ridx in pool
        if str(record.get("clip_id")) in state_rows
    ]
    print(json.dumps({"event": "pool", "val_total": len(pool), "sidecar_covered": len(covered)}), flush=True)
    if len(covered) < args.n_windows:
        raise SystemExit(f"not enough sidecar-covered val windows: {len(covered)} < {args.n_windows}")

    by_match: dict[str, list[tuple[dict[str, Any], int]]] = {}
    for record, ridx in covered:
        match_id = str(record.get("map_memory_match_id") or record.get("game_id"))
        by_match.setdefault(match_id, []).append((record, ridx))
    for match_id in by_match:
        by_match[match_id].sort(key=lambda item: stable_hex(args.seed, str(item[0].get("clip_id"))))
    match_order = sorted(by_match, key=lambda mid: stable_hex(args.seed, mid))
    if len(match_order) < 2:
        raise SystemExit("need >=2 val matches for cross-match partners")

    selected: list[tuple[dict[str, Any], int, str]] = []
    cursor = {mid: 0 for mid in match_order}
    while len(selected) < args.n_windows:
        progressed = False
        for mid in match_order:
            if len(selected) >= args.n_windows:
                break
            rows = by_match[mid]
            if cursor[mid] < len(rows):
                record, ridx = rows[cursor[mid]]
                cursor[mid] += 1
                selected.append((record, ridx, mid))
                progressed = True
        if not progressed:
            raise SystemExit("val pool exhausted before reaching n-windows")
    match_hist: dict[str, int] = {}
    for _, _, mid in selected:
        match_hist[mid] = match_hist.get(mid, 0) + 1
    print(json.dumps({"event": "selected", "n": len(selected), "per_match": match_hist}), flush=True)

    needed_ids_by_release: dict[int, set] = {}
    sample_ids_per_window: list[list[str]] = []
    for record, ridx, _mid in selected:
        sample_ids = trainer.resolve_record_sample_ids(
            record, latent_frames=args.latent_frames,
            allow_static_dense_repeat=targs.allow_static_dense_repeat,
        )
        chunk_ids = list(sample_ids[: args.latent_frames])
        if len(chunk_ids) != args.latent_frames:
            raise SystemExit(f"{record.get('clip_id')}: resolved {len(chunk_ids)} sample ids")
        sample_ids_per_window.append(chunk_ids)
        needed_ids_by_release.setdefault(ridx, set()).update(chunk_ids)

    release_by_index: dict[int, Any] = {}
    for ridx, ids in needed_ids_by_release.items():
        map_manifest, _cache_manifest = manifest_pairs[ridx]
        release_by_index[ridx] = trainer.build_lightweight_release_from_manifest(
            map_manifest, require_backend_id=args.require_backend_id, sample_ids=ids,
        )
        print(json.dumps({"event": "release_indexed", "release_index": ridx,
                          "needed_samples": len(ids)}), flush=True)

    device = torch.device("cpu")
    text_context = trainer.load_text_context(selected[0][0], device).to(torch.bfloat16).cpu()

    windows: list[dict[str, Any]] = []
    sigmas: list[float] = []
    t0 = time.time()
    for index, (record, ridx, mid) in enumerate(selected):
        clip_id = str(record.get("clip_id"))
        window_id = f"probe{index:02d}_{clip_id}"
        partner = None
        for k in range(1, len(selected)):
            cand = selected[(index + k) % len(selected)]
            if cand[2] != mid:
                partner = cand
                break
        if partner is None:
            raise SystemExit("no cross-match partner found")
        partner_record, partner_ridx, partner_mid = partner
        partner_ids = sample_ids_per_window[(index + k) % len(selected)]

        chunk_samples = [release_by_index[ridx].by_id[sid] for sid in sample_ids_per_window[index]]
        partner_chunk_samples = [release_by_index[partner_ridx].by_id[sid] for sid in partner_ids]

        x0, cond = trainer.load_latent_pair(record, device, torch.float32)
        if x0.shape[1] != args.latent_frames:
            raise SystemExit(f"{clip_id}: latent frames {x0.shape[1]} != {args.latent_frames}")
        sigma = (index + 0.5) / len(selected) * LOW_BOUNDARY
        sigmas.append(sigma)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(stable_hex(args.seed, window_id)[:12], 16))
        noise = torch.randn(x0.shape, generator=gen, dtype=torch.float32)
        xt = ((1.0 - sigma) * x0 + sigma * noise).to(torch.bfloat16)
        target = (noise - x0).to(torch.float32)
        timestep = torch.tensor([sigma * 1000.0], dtype=torch.float32)
        cam_chunk = trainer.prepare_cam_chunk(
            record, latent_frames=args.latent_frames, chunk_start=0,
            chunk_size=args.latent_frames, height=args.video_height,
            width=args.video_width, vae_stride=args.vae_stride,
            device=device, dtype=torch.bfloat16,
        )
        windows.append({
            "window_id": window_id,
            "match_id": mid,
            "partner_match_id": partner_mid,
            "record": record,
            "partner_record": partner_record,
            "chunk_samples": chunk_samples,
            "partner_chunk_samples": partner_chunk_samples,
            "xt": xt.cpu(),
            "timestep": timestep,
            "target": target.cpu(),
            "cond_chunk": cond.to(torch.bfloat16).cpu(),
            "cam_chunk": cam_chunk.cpu(),
            "sigma": sigma,
        })
        if (index + 1) % 8 == 0:
            print(json.dumps({"event": "assembled", "done": index + 1,
                              "elapsed_sec": round(time.time() - t0, 1)}), flush=True)

    first_dense = trainer.load_dense(windows[0]["chunk_samples"][0])
    if tuple(first_dense.shape) != (7, 240, 416):
        raise SystemExit(f"dense sanity failed: shape {tuple(first_dense.shape)}")

    payload = {
        "kind": "condition_probe_batch_v1",
        "split": "val",
        "seed": args.seed,
        "n_windows": len(windows),
        "sigma_band": [0.0, LOW_BOUNDARY],
        "sigmas": sigmas,
        "created_by": "tools/build_condition_probe_batch_v0.py",
        "manifests": [
            {"map_manifest": str(mp), "map_manifest_sha256": trainer.sha256_file(mp),
             "cache_manifest": str(cp), "cache_manifest_sha256": trainer.sha256_file(cp)}
            for mp, cp in manifest_pairs
        ],
        "state_cache_manifest": str(args.state_cache_manifest),
        "text_context": text_context,
        "windows": windows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    size_mb = args.out.stat().st_size / 2**20
    print(json.dumps({"event": "saved", "out": str(args.out), "size_mb": round(size_mb, 1)}), flush=True)

    reloaded = trainer.load_condition_probe_batch(args.out)
    print(json.dumps({"event": "acceptance", "loader": "load_condition_probe_batch",
                      "windows": len(reloaded["windows"]), "status": "PASS"}), flush=True)


if __name__ == "__main__":
    main()
