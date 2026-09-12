#!/usr/bin/env python3
"""Build a balanced manifest by filtering and mixing dense dataset manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def file_fingerprint(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        return {"path": str(path), "exists": False}
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(path.stat().st_size),
        "sha256": h.hexdigest(),
    }


def resolve(base: Path, sample: dict[str, Any], key: str, rel_key: str) -> Path:
    if sample.get(rel_key):
        rel = base / sample[rel_key]
        if rel.exists():
            return rel
    p = Path(sample[key])
    if p.exists():
        return p
    return base / "samples" / sample["sample_id"] / p.name


def sample_channels(sample: dict[str, Any]) -> list[str]:
    return list(sample.get("channels") or [])


def sample_stats(base: Path, sample: dict[str, Any]) -> dict[str, Any]:
    dense = np.load(resolve(base, sample, "dense_path", "dense_relpath"))["dense"]
    channels = sample_channels(sample)
    if "other_player_mask_from_memory_player_capsules" in channels:
        other_idx = channels.index("other_player_mask_from_memory_player_capsules")
        other_pixels = int((dense[other_idx] > 0.5).sum())
    elif dense.shape[0] >= 5:
        other_pixels = int(((dense[3] > 0.5) | (dense[4] > 0.5)).sum())
    else:
        other_pixels = 0
    return {
        "other_player_pixels": other_pixels,
        "nav_pixels": int((dense[2] > 0).sum()),
    }


def clone_sample(sample: dict[str, Any], source_manifest: Path, stats: dict[str, Any]) -> dict[str, Any]:
    out = dict(sample)
    out["source_manifest"] = str(source_manifest)
    out["balance_stats"] = stats
    return out


def normalized_player_params(manifest: dict[str, Any]) -> dict[str, Any]:
    params = dict(manifest.get("player_capsule_params") or {})
    if params and "player_mask_mode" not in params:
        params["player_mask_mode"] = "capsule"
    return params


def manifest_backend_report(manifest: dict[str, Any]) -> dict[str, Any]:
    ids = sorted({
        str(v)
        for v in [manifest.get("geometry_backend_id"), *(s.get("geometry_backend_id") for s in manifest.get("samples", []))]
        if v
    })
    signatures = sorted({
        str(v)
        for v in [*(manifest.get("backend_signatures") or []), *(s.get("backend_signature") for s in manifest.get("samples", []))]
        if v
    })
    features = sorted({
        str(feature)
        for s in manifest.get("samples", [])
        for feature in (s.get("backend_features") or [])
    } | {str(feature) for feature in (manifest.get("backend_features") or [])})
    return {
        "geometry_backend_ids": ids,
        "backend_signatures": signatures,
        "backend_features": features,
    }


def compatibility_report(base_manifest: dict[str, Any], enemy_manifest: dict[str, Any], require_backend_match: bool) -> dict[str, Any]:
    base_channels = base_manifest["samples"][0]["channels"] if base_manifest.get("samples") else []
    enemy_channels = enemy_manifest["samples"][0]["channels"] if enemy_manifest.get("samples") else []
    base_backend = manifest_backend_report(base_manifest)
    enemy_backend = manifest_backend_report(enemy_manifest)
    report = {
        "channels": base_channels,
        "base_player_capsule_params": normalized_player_params(base_manifest),
        "enemy_player_capsule_params": normalized_player_params(enemy_manifest),
        "base_camera_projection_params": base_manifest.get("camera_projection_params") or {},
        "enemy_camera_projection_params": enemy_manifest.get("camera_projection_params") or {},
        "base_backend": base_backend,
        "player_backend": enemy_backend,
    }
    mismatches = []
    if base_channels != enemy_channels:
        mismatches.append("channels")
    if report["base_player_capsule_params"] != report["enemy_player_capsule_params"]:
        mismatches.append("player_capsule_params")
    if report["base_camera_projection_params"] != report["enemy_camera_projection_params"]:
        mismatches.append("camera_projection_params")
    if base_backend["geometry_backend_ids"] and enemy_backend["geometry_backend_ids"]:
        if base_backend["geometry_backend_ids"] != enemy_backend["geometry_backend_ids"]:
            mismatches.append("geometry_backend_ids")
    elif require_backend_match:
        mismatches.append("missing_geometry_backend_provenance")
    if base_backend["backend_signatures"] and enemy_backend["backend_signatures"]:
        if base_backend["backend_signatures"] != enemy_backend["backend_signatures"]:
            mismatches.append("backend_signatures")
    elif require_backend_match:
        mismatches.append("missing_backend_signature")
    report["mismatches"] = mismatches
    hard_mismatches = [m for m in mismatches if m not in {"geometry_backend_ids", "backend_signatures"}]
    if hard_mismatches or (require_backend_match and mismatches):
        raise ValueError(f"Incompatible manifests for balanced mix: {mismatches}")
    return report


def load_teacher_qa_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    data = load_json(path)
    return {str(row.get("sample_id")): row for row in data.get("rows", []) if row.get("sample_id")}


def qa_depth_mae(row: dict[str, Any]) -> float | None:
    if row.get("depth_common_mae") is not None:
        return float(row["depth_common_mae"])
    depth_common = row.get("depth_common") or {}
    if depth_common.get("mae") is not None:
        return float(depth_common["mae"])
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-manifest", type=Path, required=True)
    ap.add_argument("--player-manifest", "--enemy-manifest", dest="player_manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-base", type=int, default=12)
    ap.add_argument("--max-player", "--max-enemy", dest="max_player", type=int, default=12)
    ap.add_argument("--min-player-pixels", "--min-enemy-pixels", dest="min_player_pixels", type=int, default=1)
    ap.add_argument("--require-backend-match", action="store_true")
    ap.add_argument("--base-teacher-qa", type=Path, default=None)
    ap.add_argument("--max-base-depth-common-mae", type=float, default=None)
    ap.add_argument("--min-base-hit-iou", type=float, default=None)
    ap.add_argument("--copy-files", action="store_true")
    args = ap.parse_args()

    base_manifest = load_json(args.base_manifest)
    player_manifest = load_json(args.player_manifest)
    compatibility = compatibility_report(base_manifest, player_manifest, args.require_backend_match)
    base_teacher_rows = load_teacher_qa_rows(args.base_teacher_qa)
    selected: list[dict[str, Any]] = []

    player_rows = []
    for sample in player_manifest["samples"]:
        stats = sample_stats(args.player_manifest.parent, sample)
        if stats["other_player_pixels"] >= args.min_player_pixels:
            player_rows.append((stats["other_player_pixels"], sample, stats))
    player_rows.sort(key=lambda row: row[0], reverse=True)
    for _, sample, stats in player_rows[: args.max_player]:
        selected.append(clone_sample(sample, args.player_manifest, stats))

    base_rows = []
    for sample in base_manifest["samples"]:
        stats = sample_stats(args.base_manifest.parent, sample)
        if stats["other_player_pixels"] == 0 and stats["nav_pixels"] > 0:
            qa = base_teacher_rows.get(str(sample.get("sample_id")), {})
            depth_mae = qa_depth_mae(qa) if qa else None
            hit_iou = qa.get("hit_iou")
            if args.max_base_depth_common_mae is not None and depth_mae is not None and float(depth_mae) > args.max_base_depth_common_mae:
                continue
            if args.min_base_hit_iou is not None and hit_iou is not None and float(hit_iou) < args.min_base_hit_iou:
                continue
            stats["teacher_qa"] = {
                "hit_iou": hit_iou,
                "depth_common_mae": depth_mae,
                "other_player_teacher_pixels": qa.get("other_player_teacher_pixels"),
                "other_player_memory_pixels": qa.get("other_player_memory_pixels"),
            } if qa else {}
            depth_rank = float(depth_mae) if depth_mae is not None else 999.0
            base_rows.append((depth_rank, -stats["nav_pixels"], sample, stats))
    base_rows.sort(key=lambda row: (row[0], row[1]))
    for _, _, sample, stats in base_rows[: args.max_base]:
        selected.append(clone_sample(sample, args.base_manifest, stats))

    geometry_backend_ids = sorted({
        str(v)
        for s in selected
        for v in [s.get("geometry_backend_id")]
        if v
    })
    backend_signatures = sorted({
        str(v)
        for s in selected
        for v in [s.get("backend_signature")]
        if v
    })
    backend_features = sorted({
        str(feature)
        for s in selected
        for feature in (s.get("backend_features") or [])
    })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_files:
        selected = [copy_sample(args.out_dir, s) for s in selected]

    out = {
        "kind": "balanced_memory_dense_manifest_v0",
        "policy": "Mix renderer-confirmed other-player samples with map/context samples. Training inputs still come from Map Memory dense tensors.",
        "base_manifest": str(args.base_manifest),
        "player_manifest": str(args.player_manifest),
        "input_provenance": {
            "base_manifest": file_fingerprint(args.base_manifest),
            "player_manifest": file_fingerprint(args.player_manifest),
            "base_teacher_qa": file_fingerprint(args.base_teacher_qa),
        },
        "compatibility": compatibility,
        "min_player_pixels": args.min_player_pixels,
        "base_teacher_qa": str(args.base_teacher_qa) if args.base_teacher_qa else None,
        "base_selection_policy": {
            "max_base_depth_common_mae": args.max_base_depth_common_mae,
            "min_base_hit_iou": args.min_base_hit_iou,
            "sort": "lowest teacher depth_common_mae first when --base-teacher-qa is supplied",
        },
        "sample_count": len(selected),
        "player_sample_count": sum(1 for s in selected if s["balance_stats"]["other_player_pixels"] >= args.min_player_pixels),
        "base_sample_count": sum(1 for s in selected if s["balance_stats"]["other_player_pixels"] < args.min_player_pixels),
        "channels": compatibility["channels"],
        "geometry_backend_id": geometry_backend_ids[0] if len(geometry_backend_ids) == 1 else ("mixed" if geometry_backend_ids else None),
        "geometry_backend_ids": geometry_backend_ids,
        "backend_signatures": backend_signatures,
        "backend_features": backend_features,
        "samples": selected,
    }
    path = args.out_dir / "manifest.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "manifest": str(path),
        "sample_count": out["sample_count"],
        "player_sample_count": out["player_sample_count"],
        "base_sample_count": out["base_sample_count"],
    }, ensure_ascii=False, indent=2))


def copy_sample(out_dir: Path, sample: dict[str, Any]) -> dict[str, Any]:
    source_manifest = Path(sample["source_manifest"])
    source_base = source_manifest.parent
    sample_dir = out_dir / "samples" / sample["sample_id"]
    sample_dir.mkdir(parents=True, exist_ok=True)
    mapping = [
        ("dense_path", "dense_relpath", "mesh_dense_condition_v0.npz"),
        ("target_rgb_path", "target_rgb_relpath", "target_rgb.png"),
        ("meta_path", "meta_relpath", "mesh_dense_condition_meta_v0.json"),
        ("qa_path", "qa_relpath", "mesh_dense_condition_qa_v0.png"),
    ]
    out = dict(sample)
    for key, rel_key, filename in mapping:
        src = resolve(source_base, sample, key, rel_key)
        dst = sample_dir / filename
        shutil.copy2(src, dst)
        out[key] = str(dst)
        out[rel_key] = str(dst.relative_to(out_dir))
    return out


if __name__ == "__main__":
    main()
