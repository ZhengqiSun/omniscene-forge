#!/usr/bin/env python3
"""Build Plan-B noised-context finetune cache for motion_q50 v2only.

This intentionally does not modify trainer code. Clean rows keep their original
latent_cache; degraded rows get a new .pt payload with the original video latent
and a VAE-encoded degraded first-frame condition.
"""

from __future__ import annotations
from runtime_paths import source_path

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


PROMPT_TEXT_CACHE = Path(str(source_path('assets', 'datasets/fast_cam_dust2_p0_32f_done_cpfs/text_cache/prompt_089a5a60240c0039.pt')))


def add_path(path: Path) -> None:
    text = str(path.resolve())
    if text not in sys.path:
        sys.path.insert(0, text)


def map_path(value: str | Path) -> Path:
    text = str(value)
    if text.startswith("/mnt/workspace/"):
        text = "/mnt/data/pku/" + text[len("/mnt/workspace/") :]
    return Path(text)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def choose_aug(clip_id: str) -> str:
    # Plan B: engine bucket is folded into degraded, yielding 80/20 degraded/clean.
    v = int(hashlib.sha256(clip_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "degraded" if v < 80 else "clean"


def degrade_image(img: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)
    arr = np.array(img.convert("RGB"))
    sigma = rng.uniform(1.0, 3.0)
    k = max(3, int(round(sigma * 4)) | 1)
    arr = cv2.GaussianBlur(arr, (k, k), sigmaX=sigma, sigmaY=sigma)
    quality = rng.randint(30, 60)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    arr = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    noise_std = rng.uniform(2.0, 6.0)
    noise = np.random.default_rng(seed).normal(0.0, noise_std, arr.shape)
    arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def build_condition(vae: Any, img: Image.Image, *, frames: int, height: int, width: int, chunk_size: int, device: torch.device) -> torch.Tensor:
    img_t = TF.to_tensor(img.convert("RGB")).sub_(0.5).div_(0.5).to(device)
    lat_h = height // 8
    lat_w = width // 8
    lat_f = (frames - 1) // 4 + 1
    lat_f = int(lat_f - (lat_f % chunk_size))
    effective_frames = (lat_f - 1) * 4 + 1
    msk = torch.ones(1, effective_frames, lat_h, lat_w, device=device)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]
    y = vae.encode(
        [
            torch.concat(
                [
                    torch.nn.functional.interpolate(img_t[None].cpu(), size=(height, width), mode="bicubic").transpose(0, 1),
                    torch.zeros(3, effective_frames - 1, height, width),
                ],
                dim=1,
            ).to(device)
        ]
    )[0]
    return torch.concat([msk, y]).to(torch.bfloat16).cpu()


def combine_map_manifests(paths: list[Path], out_path: Path) -> dict[str, Any]:
    manifests = [read_json(p) for p in paths]
    first = manifests[0]
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for p, m in zip(paths, manifests):
        if m.get("shape") != first.get("shape"):
            raise ValueError(f"{p}: shape mismatch")
        if m.get("channels") != first.get("channels"):
            raise ValueError(f"{p}: channels mismatch")
        if m.get("geometry_backend_id") != first.get("geometry_backend_id"):
            raise ValueError(f"{p}: geometry_backend_id mismatch")
        for s in m.get("samples", []):
            sid = str(s.get("sample_id", ""))
            if sid and sid not in seen:
                samples.append(s)
                seen.add(sid)
    out = {
        "kind": "anchor_ft_combined_memorymask_view_manifest_v0",
        "created_by": "tools/build_anchor_ft_noised_context_v0.py",
        "source_manifests": [str(p) for p in paths],
        "shape": first.get("shape"),
        "channels": first.get("channels"),
        "geometry_backend_id": first.get("geometry_backend_id"),
        "sample_count": len(samples),
        "samples": samples,
    }
    write_json(out_path, out)
    return out


def build(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    latents_dir = args.out_dir / "latents"
    degraded_img_dir = args.out_dir / "degraded_context_images"
    latents_dir.mkdir(parents=True, exist_ok=True)
    degraded_img_dir.mkdir(parents=True, exist_ok=True)

    source_rows: list[dict[str, Any]] = []
    for p in args.source_cache:
        source_rows.extend(read_jsonl(p))
    source_rows = [row for idx, row in enumerate(source_rows) if idx % args.num_shards == args.shard_index]
    source_map_paths = [map_path(p) for p in args.source_map_manifest]
    combine_map_manifests(source_map_paths, args.combined_map_manifest)
    combined_sha = sha256_file(args.combined_map_manifest)

    done_ids = set()
    if args.out_manifest.exists() and args.resume:
        for row in read_jsonl(args.out_manifest):
            done_ids.add(str(row["clip_id"]))
    elif args.out_manifest.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest without --resume: {args.out_manifest}")

    add_path(args.lingbot_repo)
    from wan.configs import WAN_CONFIGS
    from wan.modules.vae2_1 import Wan2_1_VAE

    device = torch.device(args.device)
    cfg = WAN_CONFIGS["i2v-A14B"]
    vae = Wan2_1_VAE(vae_pth=str(args.ckpt_dir / cfg.vae_checkpoint), device=device)

    counts = {"clean": 0, "degraded": 0, "skipped_existing": 0, "failed": 0}
    started = time.time()
    first_error = None
    for idx, row in enumerate(source_rows):
        clip_id = str(row["clip_id"])
        if clip_id in done_ids:
            counts["skipped_existing"] += 1
            continue
        aug = choose_aug(clip_id)
        out_row = dict(row)
        out_row["map_memory_manifest"] = str(args.combined_map_manifest)
        out_row["map_memory_manifest_sha256"] = combined_sha
        out_row["context_aug"] = aug
        out_row["context_aug_plan"] = "plan_b_noised_context_engine_folded_into_degraded_v0"
        try:
            if aug == "degraded":
                src_latent = map_path(row["latent_cache"])
                obj = torch.load(src_latent, map_location="cpu")
                image_path = map_path(row["image"])
                seed = int(hashlib.sha256(clip_id.encode("utf-8")).hexdigest()[:8], 16)
                img = degrade_image(Image.open(image_path), seed)
                img_path = degraded_img_dir / f"{clip_id}.jpg"
                img.save(img_path, quality=95)
                condition = build_condition(
                    vae,
                    img,
                    frames=args.video_frames,
                    height=args.video_height,
                    width=args.video_width,
                    chunk_size=args.chunk_size,
                    device=device,
                )
                payload = {
                    "clip_id": clip_id,
                    "latent": obj["latent"].to(torch.bfloat16).cpu(),
                    "condition": condition,
                    "latent_shape": list(obj["latent"].shape),
                    "condition_shape": list(condition.shape),
                    "latent_dtype": str(obj["latent"].dtype),
                    "condition_dtype": str(condition.dtype),
                    "context_aug": aug,
                    "source_latent_cache": str(src_latent),
                    "degraded_context_image": str(img_path),
                }
                latent_path = latents_dir / f"{clip_id}.pt"
                tmp = latent_path.with_suffix(".pt.tmp")
                torch.save(payload, tmp)
                tmp.replace(latent_path)
                out_row["latent_cache"] = str(latent_path)
                out_row["image"] = str(img_path)
                out_row["shape"] = payload["latent_shape"]
                out_row["condition_shape"] = payload["condition_shape"]
                out_row["dtype"] = "torch.bfloat16"
            else:
                out_row["latent_cache"] = str(map_path(row["latent_cache"]))
                out_row["image"] = str(map_path(row["image"]))
            for k in ["video", "poses", "intrinsics", "text_cache"]:
                if k in out_row:
                    out_row[k] = str(map_path(out_row[k]))
            if "text_cache" not in out_row:
                out_row["text_cache"] = str(PROMPT_TEXT_CACHE)
            append_jsonl(args.out_manifest, out_row)
            counts[aug] += 1
            append_jsonl(args.progress_jsonl, {"event": "done", "index": idx, "clip_id": clip_id, "context_aug": aug})
            if args.progress_every and (sum(counts[a] for a in ["clean", "degraded"]) % args.progress_every == 0):
                print(json.dumps({"event": "progress", **counts}, ensure_ascii=False), flush=True)
        except Exception as exc:
            counts["failed"] += 1
            first_error = first_error or repr(exc)
            append_jsonl(args.progress_jsonl, {"event": "failed", "index": idx, "clip_id": clip_id, "error": repr(exc)})
            if not args.continue_on_error:
                raise

    manifest_rows = read_jsonl(args.out_manifest)
    final_counts: dict[str, int] = {}
    for row in manifest_rows:
        final_counts[row["context_aug"]] = final_counts.get(row["context_aug"], 0) + 1
    report = {
        "kind": "anchor_ft_noised_context_build_report_v0",
        "status": "complete" if counts["failed"] == 0 else "failed",
        "plan": "Plan B: textured engine anchor unavailable; engine 50% bucket folded into degraded noised context",
        "source_cache": [str(p) for p in args.source_cache],
        "source_rows": len(source_rows),
        "out_manifest": str(args.out_manifest),
        "combined_map_manifest": str(args.combined_map_manifest),
        "combined_map_manifest_sha256": combined_sha,
        "manifest_rows": len(manifest_rows),
        "context_aug_counts": final_counts,
        "context_aug_ratios": {k: round(v / max(1, len(manifest_rows)), 6) for k, v in final_counts.items()},
        "new_degraded_latents_dir": str(latents_dir),
        "progress_jsonl": str(args.progress_jsonl),
        "counts_this_run": counts,
        "first_error": first_error,
        "elapsed_seconds": round(time.time() - started, 3),
        "free_bytes_after": shutil.disk_usage(args.out_dir).free,
    }
    write_json(args.out_report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-cache", type=Path, action="append", required=True)
    ap.add_argument("--source-map-manifest", type=Path, action="append", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--out-manifest", type=Path, required=True)
    ap.add_argument("--combined-map-manifest", type=Path, required=True)
    ap.add_argument("--out-report", type=Path, required=True)
    ap.add_argument("--progress-jsonl", type=Path, required=True)
    ap.add_argument("--lingbot-repo", type=Path, default=Path(str(source_path('lingbot', ''))))
    ap.add_argument("--ckpt-dir", type=Path, default=Path(str(source_path('assets', 'lingbot-world-base-cam'))))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--video-width", type=int, default=832)
    ap.add_argument("--video-height", type=int, default=480)
    ap.add_argument("--video-frames", type=int, default=81)
    ap.add_argument("--chunk-size", type=int, default=3)
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
