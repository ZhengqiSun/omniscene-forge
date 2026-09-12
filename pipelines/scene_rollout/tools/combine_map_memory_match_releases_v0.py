#!/usr/bin/env python3
"""Combine per-match Map Memory releases into one training manifest.

Each per-match release is already teacher-QA checked. This script copies their
self-contained samples into a collision-free multi-match layout, merges the QA
rows, reruns the strict readiness gate on the combined manifest, and writes a
small release JSON.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def resolve_sample_path(manifest_path: Path, sample: dict[str, Any], key: str, rel_key: str) -> Path:
    if sample.get(rel_key):
        p = manifest_path.parent / sample[rel_key]
        if p.exists():
            return p
    p = Path(sample[key])
    if p.exists():
        return p
    return manifest_path.parent / "samples" / sample["sample_id"] / p.name


def prefixed_id(match_id: str, sample_id: str) -> str:
    prefix = f"{match_id}__"
    return sample_id if sample_id.startswith(prefix) else f"{prefix}{sample_id}"


def normalized_episode(match_id: str, sample: dict[str, Any]) -> tuple[str, str | None]:
    raw_episode = sample.get("raw_episode") or sample.get("episode")
    episode = str(sample.get("episode") or "")
    if episode.startswith(f"{match_id}_"):
        return episode, str(raw_episode) if raw_episode is not None else None
    if raw_episode:
        return f"{match_id}_{raw_episode}", str(raw_episode)
    return f"{match_id}_{episode}", episode


def copy_sample(out_dir: Path, source_manifest: Path, sample: dict[str, Any], fallback_match_id: str) -> dict[str, Any]:
    old_id = str(sample["sample_id"])
    match_id = str(sample.get("match_id") or fallback_match_id)
    new_id = prefixed_id(match_id, old_id)
    episode, raw_episode = normalized_episode(match_id, sample)
    sample_dir = out_dir / "samples" / new_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    mapping = [
        ("dense_path", "dense_relpath", "mesh_dense_condition_v0.npz"),
        ("target_rgb_path", "target_rgb_relpath", "target_rgb.png"),
        ("meta_path", "meta_relpath", "mesh_dense_condition_meta_v0.json"),
        ("qa_path", "qa_relpath", "mesh_dense_condition_qa_v0.png"),
    ]
    out = dict(sample)
    out["sample_id"] = new_id
    out["source_sample_id"] = sample.get("source_sample_id", old_id)
    out["source_match_sample_id"] = old_id
    out["match_id"] = match_id
    out["raw_episode"] = raw_episode
    out["episode"] = episode
    out["source_manifest"] = str(source_manifest)
    for key, rel_key, filename in mapping:
        src = resolve_sample_path(source_manifest, sample, key, rel_key)
        dst = sample_dir / filename
        shutil.copy2(src, dst)
        out[key] = str(dst)
        out[rel_key] = str(dst.relative_to(out_dir))
    return out


def row_with_match(row: dict[str, Any], old_to_new: dict[str, dict[str, str | None]]) -> dict[str, Any]:
    old_id = str(row["sample_id"])
    mapped = old_to_new[old_id]
    out = dict(row)
    out["sample_id"] = mapped["sample_id"]
    out["source_match_sample_id"] = old_id
    out["match_id"] = mapped["match_id"]
    out["raw_episode"] = mapped["raw_episode"]
    out["episode"] = mapped["episode"]
    return out


def manifest_match_dirs(manifest: dict[str, Any]) -> list[str]:
    if isinstance(manifest.get("match_dirs"), list):
        return [str(path) for path in manifest["match_dirs"] if path]
    if manifest.get("match_dir"):
        return [str(manifest["match_dir"])]
    return []


def release_json_path(release_dir: Path) -> Path | None:
    for name in [
        "map_memory_multi_match_bsp_release_v0.json",
        "map_memory_multi_episode_bsp_release_v0.json",
    ]:
        path = release_dir / name
        if path.exists():
            return path
    return None


def source_release_entries(
    release_dir: Path,
    manifest: dict[str, Any],
    release: dict[str, Any],
    source_manifest: Path,
    teacher_qa: Path,
) -> list[dict[str, Any]]:
    nested = manifest.get("source_releases")
    if isinstance(nested, list) and nested:
        rows = []
        for row in nested:
            out = dict(row)
            out.setdefault("parent_release_dir", str(release_dir))
            rows.append(out)
        return rows
    match_dir = str(manifest.get("match_dir", ""))
    match_id = Path(match_dir).name or release_dir.name
    return [{
        "match_id": match_id,
        "release_dir": str(release_dir),
        "manifest": str(source_manifest),
        "teacher_qa": str(teacher_qa),
        "status": release.get("status"),
        "sample_count": manifest.get("sample_count"),
        "episode_count": manifest.get("episode_count"),
    }]


def unique_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for row in rows:
        key = (
            str(row.get("match_id")),
            str(row.get("release_dir")),
            str(row.get("manifest")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "hit_iou_mean": mean([float(r["hit_iou"]) for r in rows if r.get("hit_iou") is not None]),
        "semantic_binary_iou_mean": mean([
            float(r["semantic_binary_iou"]) for r in rows if r.get("semantic_binary_iou") is not None
        ]),
        "other_player_iou_mean": mean([
            float(r["other_player_iou"]) for r in rows if r.get("other_player_iou") is not None
        ]),
        "depth_common_mae_mean": mean([
            float(r["depth_common"]["mae"])
            for r in rows
            if r.get("depth_common", {}).get("mae") is not None
        ]),
        "other_player_depth_common_mae_mean": mean([
            float(r["other_player_depth_common"]["mae"])
            for r in rows
            if r.get("other_player_depth_common", {}).get("mae") is not None
        ]),
    }


def sample_stats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_match: dict[str, dict[str, int]] = {}
    by_episode: dict[str, dict[str, int]] = {}
    for sample in samples:
        role = str(sample.get("selection_role"))
        match_id = str(sample.get("match_id"))
        episode = str(sample.get("episode"))
        for table, key in [(by_match, match_id), (by_episode, episode)]:
            row = table.setdefault(key, {"total": 0, "positive": 0, "context": 0})
            row["total"] += 1
            row[role] = row.get(role, 0) + 1
    return {
        "sample_count": len(samples),
        "positive_sample_count": sum(1 for s in samples if s.get("selection_role") == "positive"),
        "context_sample_count": sum(1 for s in samples if s.get("selection_role") == "context"),
        "match_sample_counts": by_match,
        "episode_sample_counts": by_episode,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-dirs", nargs="+", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--bsp-parity", type=Path, default=Path("docs/assets/bsp_static_geometry_v0/obj_vs_bsp_disp_v0/geometry_backend_comparison_v0.json"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.force:
        shutil.rmtree(args.out_dir, ignore_errors=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifests = []
    samples = []
    qa_rows = []
    source_releases = []
    match_dirs = []
    for release_dir in args.release_dirs:
        manifest_path = release_dir / "manifest.json"
        qa_path = release_dir / "channel_teacher_qa_v0" / "memory_dense_channels_vs_teacher_v0.json"
        release_path = release_json_path(release_dir)
        manifest = load_json(manifest_path)
        qa = load_json(qa_path)
        release = load_json(release_path) if release_path and release_path.exists() else {}
        release_match_dirs = manifest_match_dirs(manifest)
        fallback_match_id = Path(release_match_dirs[0]).name if len(release_match_dirs) == 1 else release_dir.name
        old_to_new: dict[str, dict[str, str | None]] = {}
        copied = []
        for sample in manifest.get("samples", []):
            out = copy_sample(args.out_dir, manifest_path, sample, fallback_match_id)
            old_to_new[str(sample["sample_id"])] = {
                "sample_id": str(out["sample_id"]),
                "match_id": str(out["match_id"]),
                "episode": str(out["episode"]),
                "raw_episode": out.get("raw_episode"),
            }
            copied.append(out)
        manifests.append(manifest)
        samples.extend(copied)
        qa_rows.extend(row_with_match(row, old_to_new) for row in qa.get("rows", []))
        source_releases.extend(source_release_entries(release_dir, manifest, release, manifest_path, qa_path))
        match_dirs.extend(release_match_dirs)

    first = manifests[0]
    match_dirs = sorted(dict.fromkeys(path for path in match_dirs if path))
    source_releases = unique_rows(source_releases)
    backend_ids = sorted({str(s.get("geometry_backend_id")) for s in samples if s.get("geometry_backend_id")})
    backend_signatures = sorted({str(s.get("backend_signature")) for s in samples if s.get("backend_signature")})
    backend_features = sorted({str(f) for s in samples for f in (s.get("backend_features") or [])})
    manifest = {
        "kind": "multi_match_bsp_memory_dense_manifest_v0",
        "policy": "First-person Map Memory dense inputs only. Teacher streams are QA/target only.",
        "match_dirs": match_dirs,
        "source_releases": source_releases,
        "match_count": len({str(s.get("match_id")) for s in samples if s.get("match_id")}),
        "episode_count": len({s["episode"] for s in samples}),
        "channels": first.get("channels"),
        "shape": first.get("shape"),
        "geometry_backend_id": backend_ids[0] if len(backend_ids) == 1 else "mixed",
        "geometry_backend_ids": backend_ids,
        "backend_signatures": backend_signatures,
        "backend_features": backend_features,
        "player_capsule_params": first.get("player_capsule_params"),
        "camera_projection_params": first.get("camera_projection_params"),
        **sample_stats(samples),
        "samples": samples,
    }
    manifest_path = args.out_dir / "manifest.json"
    write_json(manifest_path, manifest)

    qa = {
        "kind": "memory_dense_channels_vs_teacher_v0",
        "manifest": str(manifest_path),
        "match_dirs": match_dirs,
        "sample_count": len(qa_rows),
        "missing_count": 0,
        "summary": summarize_rows(qa_rows),
        "rows": qa_rows,
        "missing": [],
        "policy": "Teacher streams are used only for QA. Memory dense tensors remain Map-rendered inputs.",
    }
    qa_path = args.out_dir / "channel_teacher_qa_v0" / "memory_dense_channels_vs_teacher_v0.json"
    write_json(qa_path, qa)

    tools = Path(__file__).resolve().parent
    run([
        sys.executable, str(tools / "check_memory_dense_manifest_v0.py"),
        "--manifest", str(manifest_path),
        "--max-samples", str(manifest["sample_count"]),
    ])
    run([
        sys.executable, str(tools / "check_map_memory_training_readiness_v0.py"),
        "--manifest", str(manifest_path),
        "--teacher-qa", str(qa_path),
        "--mode", "mixed",
        "--bsp-parity", str(args.bsp_parity),
        "--max-samples", str(manifest["sample_count"]),
        "--min-samples", str(manifest["sample_count"]),
        "--min-episodes", str(manifest["episode_count"]),
        "--require-geometry-backend-id", "bsp_faces_disp_gpu",
        "--out-json", str(args.out_dir / "training_readiness_v0.json"),
    ])
    run([
        sys.executable, str(tools / "summarize_memory_dense_dataset_v0.py"),
        "--manifest", str(manifest_path),
        "--out-json", str(args.out_dir / "summary_v0.json"),
        "--out-montage", str(args.out_dir / "qa_montage_v0.png"),
        "--max-montage-images", "8",
    ])

    readiness = load_json(args.out_dir / "training_readiness_v0.json")
    release = {
        "kind": "map_memory_multi_match_bsp_release_v0",
        "status": readiness.get("status"),
        "manifest": str(manifest_path),
        "teacher_qa": str(qa_path),
        "training_readiness": str(args.out_dir / "training_readiness_v0.json"),
        "match_count": manifest["match_count"],
        "episode_count": manifest["episode_count"],
        "sample_count": manifest["sample_count"],
        "positive_sample_count": manifest["positive_sample_count"],
        "context_sample_count": manifest["context_sample_count"],
        "teacher_qa_summary": qa["summary"],
        "readiness_failures": readiness.get("failures", []),
    }
    write_json(args.out_dir / "map_memory_multi_match_bsp_release_v0.json", release)
    print(json.dumps(release, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
