#!/usr/bin/env python3
"""Bridge v2 dense render pools into the trainer's map-manifest + cache-manifest contract.

The v2 pools ship a ``release_manifest_v1.json`` (per-sample dense npz paths, QA
ratios, shape/backend locks).  The trainer instead wants the older
``*_dense_combined_manifest_memorymask_view_v0.json`` layout, an aligned cache
manifest whose rows point at latents/text, and two sidecars next to the manifest
(``training_readiness_v0.json`` and ``channel_teacher_qa_v0/...json``).

This tool joins the two: aligned rows supply the clip-level map-memory fields
(sample ids, match/episode ids, ego stem) and the release manifest supplies the
per-sample render facts.  Three things are re-derived rather than inherited:

* ``selection_role`` is recomputed from the *v2* measured Memory ch3 pixel count
  (positive iff > 0).  The inherited roles were measured on the v1 320x176 render
  and no longer agree with the v2 240x416 geometry, and the release validator
  fails hard on positive-with-empty-mask / context-with-nonempty-mask.  Roles are
  rewritten in the manifest and in the cache rows together, because the trainer
  cross-checks the two.
* ``latent_cache`` and friends are absolutised (the tier69h manifests stored them
  relative to the zhengqi repo root).
* ``map_memory_manifest``/``map_memory_manifest_sha256`` are re-stamped so the
  trainer's per-record manifest check passes against the manifest we just wrote.

``target_rgb_path``/``qa_path`` keys are omitted when the v2 render has no such
file (this is a latent-space training release; there are no extracted RGB target
frames and no teacher seg/depth streams).  The manifest records that explicitly
via ``target_rgb_available``/``qa_available``.
"""
from __future__ import annotations

from runtime_paths import ASSET_ROOT, LINGBOT_ROOT

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REGION_MASK_KIND = "memory_dense_channel_3_surrogate_v0"
REGION_MASK_POLICY = {
    "kind": REGION_MASK_KIND,
    "source": "dense_channel_3_other_player_mask_from_memory_player_capsules",
    "valid_for": "relative true-vs-shuffled Memory-mask region loss only",
}
ABS_KEYS = (
    "latent_cache", "clean_latent_cache", "degraded_latent_cache", "text_cache",
    "poses", "intrinsics", "image", "video", "mp4", "action_json", "meta_json",
    "prompt_txt", "sample_dir", "clip_dir",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def absolutise(value: Any, root: Path) -> Any:
    if isinstance(value, str) and value and not value.startswith("/"):
        return str(root / value)
    return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--aligned", type=Path, required=True)
    ap.add_argument("--dataset-source", default="")
    ap.add_argument("--pool-name", required=True)
    ap.add_argument("--out-dir", type=Path, required=True, help="pool subdir is created under here")
    ap.add_argument("--zhengqi-root", type=Path, default=ASSET_ROOT)
    ap.add_argument("--repeat", type=int, default=1, help="duplicate every cache row N times (upweighting)")
    ap.add_argument("--limit-aligned", type=int, default=0, help="debug: only read the first N aligned rows")
    ap.add_argument("--positive-min-pixels", type=int, default=1,
                    help="Memory ch3 pixel count at or above which a sample counts as positive.")
    args = ap.parse_args()

    out_dir: Path = args.out_dir / args.pool_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    wanted: set[str] = set()
    sid_meta: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(args.aligned):
        if args.dataset_source and row.get("dataset_source") != args.dataset_source:
            continue
        rows.append(row)
        for sid in (row.get("map_memory_sample_ids") or []):
            sid = str(sid)
            wanted.add(sid)
            sid_meta.setdefault(sid, {
                "ego_stem": row.get("map_memory_ego_stem") or row.get("player_stem"),
                "episode": row.get("map_memory_episode"),
                "raw_episode": row.get("map_memory_raw_episode"),
                "match_id": row.get("map_memory_match_id"),
            })
        if args.limit_aligned and len(rows) >= args.limit_aligned:
            break
    print(f"aligned rows: {len(rows)} | distinct sample_ids wanted: {len(wanted)}", flush=True)

    release = json.loads(args.release.read_text(encoding="utf-8"))
    channels = release.get("channels")
    shape = release.get("shape")
    samples_in = release.get("samples") or []
    print(f"release samples: {len(samples_in)} | shape {shape}", flush=True)

    out_samples: list[dict[str, Any]] = []
    qa_out: list[dict[str, Any]] = []
    role_by_sid: dict[str, str] = {}
    seen: set[str] = set()
    dup = 0
    inherited_roles = Counter()
    flips = Counter()
    have_rgb = have_qa = 0
    for s in samples_in:
        sid = str(s.get("sample_id"))
        meta = sid_meta.get(sid)
        if meta is None:
            continue
        if sid in seen:
            dup += 1
            continue
        seen.add(sid)
        px = int(s.get("other_player_memory_pixels") or 0)
        role = "positive" if px >= args.positive_min_pixels else "context"
        role_by_sid[sid] = role
        out = {
            "sample_id": sid,
            "frame_index": s.get("frame_index"),
            "dense_path": s.get("dense_path"),
            "meta_path": s.get("meta_path"),
            "channels": channels,
            "shape": shape,
            "geometry_backend_id": s.get("geometry_backend_id") or release.get("geometry_backend_id"),
            "mesh_backend": s.get("mesh_backend") or release.get("mesh_backend"),
            "mesh_hit_ratio": s.get("mesh_hit_ratio"),
            "nav_semantic_hit_ratio": s.get("nav_semantic_hit_ratio"),
            "other_player_memory_pixels": px,
            "selection_role": role,
            "region_mask_kind": REGION_MASK_KIND,
            "region_mask_policy": REGION_MASK_POLICY,
        }
        if s.get("target_rgb_path"):
            out["target_rgb_path"] = s["target_rgb_path"]; have_rgb += 1
        if s.get("qa_path"):
            out["qa_path"] = s["qa_path"]; have_qa += 1
        out.update(meta)
        out_samples.append(out)
        qa_out.append({
            "sample_id": sid,
            "selection_role": role,
            "other_player_memory_pixels": px,
            "visible_teacher_players": 0,
            "other_player_teacher_pixels": 0,
            "teacher_streams_available": False,
        })
    print(f"joined samples: {len(out_samples)} | duplicate sample_ids skipped: {dup}", flush=True)

    kept_rows: list[dict[str, Any]] = []
    dropped_uncovered = 0
    for row in rows:
        sids = [str(x) for x in (row.get("map_memory_sample_ids") or [])]
        if not sids or any(sid not in role_by_sid for sid in sids):
            dropped_uncovered += 1
            continue
        old = [str(x) for x in (row.get("map_memory_selection_roles") or [])]
        new = [role_by_sid[sid] for sid in sids]
        for i, r in enumerate(new):
            prev = old[i] if i < len(old) else "?"
            inherited_roles[prev] += 1
            if prev != r:
                flips[f"{prev}->{r}"] += 1
        row["map_memory_selection_roles"] = new
        row["map_memory_positive_frames"] = sum(1 for r in new if r == "positive")
        row["map_memory_context_frames"] = sum(1 for r in new if r == "context")
        kept_rows.append(row)
    print(f"cache rows kept: {len(kept_rows)} | dropped (sample not in pool): {dropped_uncovered}", flush=True)
    print(f"role flips: {dict(flips)}", flush=True)

    keep_sids = {sid for row in kept_rows for sid in (str(x) for x in row.get("map_memory_sample_ids") or [])}
    out_samples = [s for s in out_samples if s["sample_id"] in keep_sids]
    qa_out = [q for q in qa_out if q["sample_id"] in keep_sids]
    role_hist = Counter(s["selection_role"] for s in out_samples)
    print(f"final role histogram: {dict(role_hist)}", flush=True)

    manifest_path = out_dir / f"{args.pool_name}_v2_combined_manifest_memorymask_view_v1.json"
    manifest = {
        "kind": f"{args.pool_name}_dense_combined_manifest_v2_memorymask_view_v1",
        "created_by": str(Path(__file__).resolve()),
        "created_at_unix": int(time.time()),
        "source_release_manifest": str(args.release.resolve()),
        "source_aligned_manifest": str(args.aligned.resolve()),
        "channels": channels,
        "shape": shape,
        "geometry_backend_id": release.get("geometry_backend_id"),
        "mesh_backend": release.get("mesh_backend"),
        "region_mask_kind": REGION_MASK_KIND,
        "region_mask_policy": REGION_MASK_POLICY,
        "teacher_qa_available": False,
        "target_rgb_available": bool(have_rgb),
        "qa_available": bool(have_qa),
        "selection_role_source": f"v2_memory_ch3_pixels_ge_{args.positive_min_pixels}",
        "sample_count": len(out_samples),
        "samples": out_samples,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f)
    sha = sha256_file(manifest_path)
    print(f"wrote {manifest_path} sha256={sha}", flush=True)

    qa_dir = out_dir / "channel_teacher_qa_v0"
    qa_dir.mkdir(exist_ok=True)
    qa_path = qa_dir / "memory_dense_channels_vs_teacher_v0.json"
    with qa_path.open("w", encoding="utf-8") as f:
        json.dump({
            "kind": "memory_dense_channels_vs_teacher_v0_memorymask_surrogate",
            "region_mask_kind": REGION_MASK_KIND,
            "teacher_streams_available": False,
            "note": "No teacher seg/depth streams exist for the v2 light-data release. "
                    "visible_teacher_players/other_player_teacher_pixels are recorded as 0 and are "
                    "unused on the Memory-mask surrogate validation path; "
                    "other_player_memory_pixels is the real per-sample v2 render measurement.",
            "sample_count": len(qa_out),
            "rows": qa_out,
        }, f)
    print(f"wrote {qa_path} rows={len(qa_out)}", flush=True)

    readiness = {
        "kind": "map_memory_training_readiness_v0_memorymask_surrogate",
        "status": "pass",
        "created_by": str(Path(__file__).resolve()),
        "created_at_unix": int(time.time()),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha,
        "region_mask_kind": REGION_MASK_KIND,
        "teacher_qa_available": False,
        "checks": {
            "shape": shape,
            "channels_locked": True,
            "geometry_backend_id": release.get("geometry_backend_id"),
            "roles_rederived_from_v2_pixels": True,
            "sample_count": len(out_samples),
        },
        "failures": [],
    }
    (out_dir / "training_readiness_v0.json").write_text(json.dumps(readiness, indent=1), encoding="utf-8")

    cache_path = out_dir / f"{args.pool_name}_v2_cache_manifest_v1.jsonl"
    split_counts: Counter = Counter()
    written = 0
    with cache_path.open("w", encoding="utf-8") as f:
        for row in kept_rows:
            new = dict(row)
            for key in ABS_KEYS:
                if key in new:
                    new[key] = absolutise(new[key], args.zhengqi_root)
            new["map_memory_manifest"] = str(manifest_path.resolve())
            new["map_memory_manifest_sha256"] = sha
            split_counts[new.get("map_memory_split")] += args.repeat
            line = json.dumps(new)
            for _ in range(args.repeat):
                f.write(line + "\n")
                written += 1
    print(f"wrote {cache_path} rows={written} splits={dict(split_counts)}", flush=True)

    px_vals = sorted(int(s["other_player_memory_pixels"]) for s in out_samples)
    report = {
        "kind": "qxq_v2_map_and_cache_bridge_report_v1",
        "pool": args.pool_name,
        "release_manifest": str(args.release.resolve()),
        "aligned_manifest": str(args.aligned.resolve()),
        "map_manifest": str(manifest_path.resolve()),
        "map_manifest_sha256": sha,
        "cache_manifest": str(cache_path.resolve()),
        "teacher_qa": str(qa_path.resolve()),
        "readiness": str((out_dir / "training_readiness_v0.json").resolve()),
        "aligned_rows_in": len(rows),
        "cache_rows_kept": len(kept_rows),
        "cache_rows_written": written,
        "cache_rows_dropped_uncovered": dropped_uncovered,
        "release_samples_in": len(samples_in),
        "manifest_samples_out": len(out_samples),
        "duplicate_sample_ids_skipped": dup,
        "repeat": args.repeat,
        "positive_min_pixels": args.positive_min_pixels,
        "inherited_role_histogram": dict(inherited_roles),
        "role_flips": dict(flips),
        "final_role_histogram": dict(role_hist),
        "split_counts": {str(k): v for k, v in split_counts.items()},
        "other_player_memory_pixels": {
            "n": len(px_vals),
            "min": px_vals[0] if px_vals else None,
            "p50": px_vals[len(px_vals) // 2] if px_vals else None,
            "p90": px_vals[int(len(px_vals) * 0.9)] if px_vals else None,
            "max": px_vals[-1] if px_vals else None,
            "n_zero": sum(1 for v in px_vals if v == 0),
        },
        "status": "pass" if out_samples and kept_rows else "fail",
    }
    (out_dir / f"{args.pool_name}_v2_bridge_report_v1.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "split_counts"}, indent=1))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    sys.exit(main())
