#!/usr/bin/env python3
"""Build a larger Map Memory release from already-rendered QA candidates.

This is the fast scale-up path after the curated v42 gate: it does not rerender
the map. It reuses per-episode positive/context candidate manifests and teacher
QA rows, keeps only rows that satisfy the same static geometry gates, copies the
self-contained samples into one multi-match release, and reruns readiness.
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


CHANNELS = [
    "env_depth_norm_from_world_obj_projection",
    "env_mesh_hit_mask",
    "env_nav_place_semantic_from_static_memory",
    "other_player_mask_from_memory_player_capsules",
    "other_player_depth_norm_from_memory_capsules",
    "other_player_yaw_sin_relative_to_ego_from_memory",
    "other_player_yaw_cos_relative_to_ego_from_memory",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("+ " + " ".join(cmd) + f" > {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-80:])
        raise RuntimeError(f"command failed with code {proc.returncode}: {' '.join(cmd)}\n{tail}")


def depth_mae(row: dict[str, Any]) -> float | None:
    if row.get("depth_common_mae") is not None:
        return float(row["depth_common_mae"])
    depth_common = row.get("depth_common") or {}
    if depth_common.get("mae") is not None:
        return float(depth_common["mae"])
    return None


def resolve_sample_path(manifest_path: Path, sample: dict[str, Any], key: str, rel_key: str) -> Path:
    if sample.get(rel_key):
        p = manifest_path.parent / sample[rel_key]
        if p.exists():
            return p
    raw = Path(sample[key])
    if raw.exists():
        return raw
    return manifest_path.parent / "samples" / sample["sample_id"] / raw.name


def prefixed_id(match_id: str, sample_id: str) -> str:
    prefix = f"{match_id}__"
    return sample_id if sample_id.startswith(prefix) else f"{prefix}{sample_id}"


def source_release_json(release_dir: Path) -> dict[str, Any]:
    for name in [
        "map_memory_multi_match_bsp_release_v0.json",
        "map_memory_multi_episode_bsp_release_v0.json",
    ]:
        path = release_dir / name
        if path.exists():
            return load_json(path)
    return {}


def manifest_match_dirs(manifest: dict[str, Any]) -> list[str]:
    if isinstance(manifest.get("match_dirs"), list):
        return [str(path) for path in manifest["match_dirs"] if path]
    if manifest.get("match_dir"):
        return [str(manifest["match_dir"])]
    return []


def qa_rows_by_id(path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(path)
    return {str(row["sample_id"]): row for row in data.get("rows", [])}


def positive_ok(row: dict[str, Any], args: argparse.Namespace) -> bool:
    visible = int(row.get("visible_teacher_players", 0) or 0)
    teacher_pixels = int(row.get("other_player_teacher_pixels", 0) or 0)
    memory_pixels = int(row.get("other_player_memory_pixels", 0) or 0)
    other_iou = float(row.get("other_player_iou", 0.0) or 0.0)
    hit_iou = float(row.get("hit_iou", 0.0) or 0.0)
    mae = depth_mae(row)
    return (
        visible > 0
        and teacher_pixels > 0
        and memory_pixels > 0
        and other_iou >= args.min_positive_other_iou
        and hit_iou >= args.min_hit_iou
        and mae is not None
        and mae <= args.max_depth_mae
    )


def context_ok(row: dict[str, Any], args: argparse.Namespace) -> bool:
    visible = int(row.get("visible_teacher_players", 0) or 0)
    teacher_pixels = int(row.get("other_player_teacher_pixels", 0) or 0)
    memory_pixels = int(row.get("other_player_memory_pixels", 0) or 0)
    hit_iou = float(row.get("hit_iou", 0.0) or 0.0)
    mae = depth_mae(row)
    return (
        visible == 0
        and teacher_pixels == 0
        and memory_pixels == 0
        and hit_iou >= args.min_hit_iou
        and mae is not None
        and mae <= args.max_depth_mae
    )


def rank_positive(item: tuple[dict[str, Any], dict[str, Any]]) -> tuple[float, float, float, int]:
    _, row = item
    mae = depth_mae(row)
    return (
        float(row.get("other_player_iou", 0.0) or 0.0),
        -float(mae if mae is not None else 999.0),
        float(row.get("hit_iou", 0.0) or 0.0),
        int(row.get("other_player_teacher_pixels", 0) or 0),
    )


def rank_context(item: tuple[dict[str, Any], dict[str, Any]]) -> tuple[float, float]:
    _, row = item
    mae = depth_mae(row)
    return (
        float(mae if mae is not None else 999.0),
        -float(row.get("hit_iou", 0.0) or 0.0),
    )


def select_from_candidate_manifest(
    manifest_path: Path,
    qa_path: Path,
    role: str,
    args: argparse.Namespace,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    manifest = load_json(manifest_path)
    rows = qa_rows_by_id(qa_path)
    selected = []
    for sample in manifest.get("samples", []):
        row = rows.get(str(sample["sample_id"]))
        if not row:
            continue
        if role == "positive" and positive_ok(row, args):
            selected.append((sample, row))
        elif role == "context" and context_ok(row, args):
            selected.append((sample, row))
    if role == "positive":
        selected.sort(key=rank_positive, reverse=True)
        if args.max_positive_per_episode is not None:
            selected = selected[: args.max_positive_per_episode]
    else:
        selected.sort(key=rank_context)
        if args.max_context_per_episode is not None:
            selected = selected[: args.max_context_per_episode]
    return selected


def normalized_episode(match_id: str, sample: dict[str, Any]) -> tuple[str, str | None]:
    raw_episode = sample.get("raw_episode") or sample.get("episode")
    episode = str(sample.get("episode") or "")
    if episode.startswith(f"{match_id}_"):
        return episode, str(raw_episode) if raw_episode is not None else None
    if raw_episode:
        return f"{match_id}_{raw_episode}", str(raw_episode)
    return f"{match_id}_{episode}", episode


def copy_selected_sample(
    out_dir: Path,
    manifest_path: Path,
    sample: dict[str, Any],
    qa_row: dict[str, Any],
    match_id: str,
    role: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    old_id = str(sample["sample_id"])
    new_id = prefixed_id(match_id, old_id)
    episode, raw_episode = normalized_episode(match_id, sample)
    sample_dir = out_dir / "samples" / new_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    out_sample = dict(sample)
    out_sample["sample_id"] = new_id
    out_sample["source_sample_id"] = sample.get("source_sample_id", old_id)
    out_sample["source_match_sample_id"] = old_id
    out_sample["match_id"] = match_id
    out_sample["episode"] = episode
    out_sample["raw_episode"] = raw_episode
    out_sample["selection_role"] = role
    out_sample["source_manifest"] = str(manifest_path)
    out_sample["channels"] = CHANNELS
    out_sample["shape"] = [7, 176, 320]
    out_sample["teacher_qa_at_selection"] = {
        "hit_iou": qa_row.get("hit_iou"),
        "depth_common_mae": depth_mae(qa_row),
        "other_player_iou": qa_row.get("other_player_iou"),
        "visible_teacher_players": qa_row.get("visible_teacher_players"),
        "other_player_teacher_pixels": qa_row.get("other_player_teacher_pixels"),
        "other_player_memory_pixels": qa_row.get("other_player_memory_pixels"),
    }

    for key, rel_key, filename in [
        ("dense_path", "dense_relpath", "mesh_dense_condition_v0.npz"),
        ("target_rgb_path", "target_rgb_relpath", "target_rgb.png"),
        ("meta_path", "meta_relpath", "mesh_dense_condition_meta_v0.json"),
        ("qa_path", "qa_relpath", "mesh_dense_condition_qa_v0.png"),
    ]:
        src = resolve_sample_path(manifest_path, sample, key, rel_key)
        dst = sample_dir / filename
        shutil.copy2(src, dst)
        out_sample[key] = str(dst)
        out_sample[rel_key] = str(dst.relative_to(out_dir))

    out_row = dict(qa_row)
    out_row["sample_id"] = new_id
    out_row["source_match_sample_id"] = old_id
    out_row["match_id"] = match_id
    out_row["episode"] = episode
    out_row["raw_episode"] = raw_episode
    out_row["selection_role"] = role
    return out_sample, out_row


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
        "depth_common_mae_mean": mean([v for r in rows if (v := depth_mae(r)) is not None]),
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
    ap.add_argument("--source-release-dirs", nargs="+", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--bsp-parity", type=Path, default=Path("docs/assets/bsp_static_geometry_v0/obj_vs_bsp_disp_v0/geometry_backend_comparison_v0.json"))
    ap.add_argument("--min-positive-other-iou", type=float, default=0.25)
    ap.add_argument("--min-hit-iou", type=float, default=0.90)
    ap.add_argument("--max-depth-mae", type=float, default=0.08)
    ap.add_argument("--max-positive-per-episode", type=int, default=None)
    ap.add_argument("--max-context-per-episode", type=int, default=None)
    ap.add_argument("--manifest-check-samples", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.force:
        shutil.rmtree(args.out_dir, ignore_errors=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []
    source_releases: list[dict[str, Any]] = []
    match_dirs: list[str] = []
    selection_report: dict[str, Any] = {}
    seen_ids: set[str] = set()
    first_manifest: dict[str, Any] | None = None

    for release_dir in args.source_release_dirs:
        release_manifest = load_json(release_dir / "manifest.json")
        first_manifest = first_manifest or release_manifest
        release = source_release_json(release_dir)
        release_match_dirs = manifest_match_dirs(release_manifest)
        if len(release_match_dirs) != 1:
            raise ValueError(f"{release_dir}: expected one match_dir, got {release_match_dirs}")
        match_dir = release_match_dirs[0]
        match_id = Path(match_dir).name
        match_dirs.append(match_dir)
        source_releases.append({
            "match_id": match_id,
            "release_dir": str(release_dir),
            "manifest": str(release_dir / "manifest.json"),
            "teacher_qa": str(release_dir / "channel_teacher_qa_v0" / "memory_dense_channels_vs_teacher_v0.json"),
            "status": release.get("status"),
            "sample_count": release_manifest.get("sample_count"),
            "episode_count": release_manifest.get("episode_count"),
        })

        for episode, paths in release_manifest.get("selection_report", {}).items():
            selected = []
            selected.extend(select_from_candidate_manifest(
                Path(paths["positive_manifest"]),
                Path(paths["positive_teacher_qa"]),
                "positive",
                args,
            ))
            selected.extend(select_from_candidate_manifest(
                Path(paths["context_manifest"]),
                Path(paths["context_teacher_qa"]),
                "context",
                args,
            ))
            ep_key = f"{match_id}_{episode}"
            selection_report[ep_key] = {
                "source_release_dir": str(release_dir),
                "raw_episode": episode,
                "positive_manifest": paths["positive_manifest"],
                "positive_teacher_qa": paths["positive_teacher_qa"],
                "context_manifest": paths["context_manifest"],
                "context_teacher_qa": paths["context_teacher_qa"],
                "positive_selected": 0,
                "context_selected": 0,
            }
            for sample, row in selected:
                role = "positive" if positive_ok(row, args) else "context"
                new_id = prefixed_id(match_id, str(sample["sample_id"]))
                if new_id in seen_ids:
                    continue
                seen_ids.add(new_id)
                out_sample, out_row = copy_selected_sample(args.out_dir, Path(paths["positive_manifest"]).parent.parent / ("positive" if role == "positive" else "context_candidates") / "manifest.json", sample, row, match_id, role)
                samples.append(out_sample)
                qa_rows.append(out_row)
                selection_report[ep_key][f"{role}_selected"] += 1

    if first_manifest is None:
        raise ValueError("no source manifests")
    if not samples:
        raise ValueError("selection produced no samples")

    backend_ids = sorted({str(s.get("geometry_backend_id")) for s in samples if s.get("geometry_backend_id")})
    backend_signatures = sorted({str(s.get("backend_signature")) for s in samples if s.get("backend_signature")})
    backend_features = sorted({str(f) for s in samples for f in (s.get("backend_features") or [])})
    manifest = {
        "kind": "multi_match_bsp_memory_dense_manifest_v0",
        "policy": "First-person Map Memory dense inputs only. Teacher streams are QA/target only.",
        "match_dirs": sorted(dict.fromkeys(match_dirs)),
        "source_releases": source_releases,
        "selection_policy": {
            "source": "existing per-episode candidate manifests and teacher QA",
            "positive": {
                "visible_teacher_players": ">0",
                "other_player_teacher_pixels": ">0",
                "other_player_memory_pixels": ">0",
                "min_other_player_iou": args.min_positive_other_iou,
                "min_hit_iou": args.min_hit_iou,
                "max_depth_common_mae": args.max_depth_mae,
                "max_positive_per_episode": args.max_positive_per_episode,
            },
            "context": {
                "visible_teacher_players": 0,
                "other_player_teacher_pixels": 0,
                "other_player_memory_pixels": 0,
                "min_hit_iou": args.min_hit_iou,
                "max_depth_common_mae": args.max_depth_mae,
                "max_context_per_episode": args.max_context_per_episode,
            },
        },
        "selection_report": selection_report,
        "match_count": len({str(s.get("match_id")) for s in samples if s.get("match_id")}),
        "episode_count": len({s["episode"] for s in samples}),
        "channels": CHANNELS,
        "shape": [7, 176, 320],
        "geometry_backend_id": backend_ids[0] if len(backend_ids) == 1 else "mixed",
        "geometry_backend_ids": backend_ids,
        "backend_signatures": backend_signatures,
        "backend_features": backend_features,
        "player_capsule_params": first_manifest.get("player_capsule_params"),
        "camera_projection_params": first_manifest.get("camera_projection_params"),
        **sample_stats(samples),
        "samples": samples,
    }
    manifest_path = args.out_dir / "manifest.json"
    write_json(manifest_path, manifest)

    qa = {
        "kind": "memory_dense_channels_vs_teacher_v0",
        "manifest": str(manifest_path),
        "match_dirs": manifest["match_dirs"],
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
    check_samples = args.manifest_check_samples or manifest["sample_count"]
    run_logged([
        sys.executable, str(tools / "check_memory_dense_manifest_v0.py"),
        "--manifest", str(manifest_path),
        "--max-samples", str(check_samples),
    ], args.out_dir / "logs" / "check_memory_dense_manifest_v0.log")
    run_logged([
        sys.executable, str(tools / "check_map_memory_training_readiness_v0.py"),
        "--manifest", str(manifest_path),
        "--teacher-qa", str(qa_path),
        "--mode", "mixed",
        "--bsp-parity", str(args.bsp_parity),
        "--max-samples", str(check_samples),
        "--min-samples", str(manifest["sample_count"]),
        "--min-episodes", str(manifest["episode_count"]),
        "--require-geometry-backend-id", "bsp_faces_disp_gpu",
        "--out-json", str(args.out_dir / "training_readiness_v0.json"),
    ], args.out_dir / "logs" / "check_map_memory_training_readiness_v0.log")
    run_logged([
        sys.executable, str(tools / "summarize_memory_dense_dataset_v0.py"),
        "--manifest", str(manifest_path),
        "--out-json", str(args.out_dir / "summary_v0.json"),
        "--out-montage", str(args.out_dir / "qa_montage_v0.png"),
        "--max-montage-images", "8",
    ], args.out_dir / "logs" / "summarize_memory_dense_dataset_v0.log")

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
        "logs": {
            "manifest_check": str(args.out_dir / "logs" / "check_memory_dense_manifest_v0.log"),
            "readiness": str(args.out_dir / "logs" / "check_map_memory_training_readiness_v0.log"),
            "summary": str(args.out_dir / "logs" / "summarize_memory_dense_dataset_v0.log"),
        },
    }
    write_json(args.out_dir / "map_memory_multi_match_bsp_release_v0.json", release)
    print(json.dumps(release, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
