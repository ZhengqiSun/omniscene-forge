#!/usr/bin/env python3
"""Build a multi-episode BSP+displacement Map Memory release.

This runner is the current map-side promotion path for adapter data. It keeps
the model input contract fixed to first-person 7-channel Map Memory dense
tensors, uses teacher streams only for sampling/QA, and produces one
self-contained manifest copied under the final release directory.
"""

from __future__ import annotations
from runtime_paths import source_path

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MATCH_DIR = Path(str(source_path('assets', 'csgo-datasets-fullsubset/32f1644d4f42c29d/030a506e5e77459c8d5078a06eec61ee')))
DEFAULT_BSP_FACES_NPZ = Path("docs/assets/bsp_static_geometry_v0/de_dust2_bsp_faces_v0.npz")
DEFAULT_EPISODES = ["Ep_000003", "Ep_000004", "Ep_000005"]
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


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def have_files(paths: list[Path]) -> bool:
    return all(path.exists() for path in paths)


def run_or_reuse(cmd: list[str], env: dict[str, str], outputs: list[Path], reuse_existing: bool) -> None:
    if reuse_existing and have_files(outputs):
        print("+ reuse " + " ".join(str(path) for path in outputs), flush=True)
        return
    run(cmd, env=env)


def gpu_env() -> dict[str, str]:
    env = os.environ.copy()
    cuda = env.get("CUDA_PATH", "/usr/local/cuda-12.6")
    env["CUDA_PATH"] = cuda
    env["PATH"] = f"{cuda}/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = f"{cuda}/lib64:" + env.get("LD_LIBRARY_PATH", "")
    return env


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


def resolve_sample_path(manifest_path: Path, sample: dict[str, Any], key: str, rel_key: str) -> Path:
    if sample.get(rel_key):
        p = manifest_path.parent / sample[rel_key]
        if p.exists():
            return p
    raw = Path(sample[key])
    if raw.exists():
        return raw
    return manifest_path.parent / "samples" / sample["sample_id"] / raw.name


def copy_sample(out_dir: Path, manifest_path: Path, sample: dict[str, Any], role: str, qa_row: dict[str, Any]) -> dict[str, Any]:
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
        src = resolve_sample_path(manifest_path, sample, key, rel_key)
        dst = sample_dir / filename
        shutil.copy2(src, dst)
        out[key] = str(dst)
        out[rel_key] = str(dst.relative_to(out_dir))
    out["source_manifest"] = str(manifest_path)
    out["source_sample_id"] = sample["sample_id"]
    out["selection_role"] = role
    out["teacher_qa_at_selection"] = {
        "hit_iou": qa_row.get("hit_iou"),
        "depth_common_mae": qa_depth_mae(qa_row),
        "other_player_iou": qa_row.get("other_player_iou"),
        "visible_teacher_players": qa_row.get("visible_teacher_players"),
        "other_player_teacher_pixels": qa_row.get("other_player_teacher_pixels"),
        "other_player_memory_pixels": qa_row.get("other_player_memory_pixels"),
    }
    return out


def qa_depth_mae(row: dict[str, Any]) -> float | None:
    if row.get("depth_common_mae") is not None:
        return float(row["depth_common_mae"])
    depth_common = row.get("depth_common") or {}
    if depth_common.get("mae") is not None:
        return float(depth_common["mae"])
    return None


def load_qa_rows(path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(path)
    return {str(row["sample_id"]): row for row in data.get("rows", [])}


def teacher_player_pixels(row: dict[str, Any]) -> int:
    return int(row.get("other_player_teacher_pixels", 0) or 0)


def memory_player_pixels(row: dict[str, Any]) -> int:
    return int(row.get("other_player_memory_pixels", 0) or 0)


def visible_teacher_players(row: dict[str, Any]) -> int:
    return int(row.get("visible_teacher_players", 0) or 0)


def select_positive_rows(
    manifest_path: Path,
    qa_path: Path,
    max_per_episode: int,
    min_other_iou: float,
    min_hit_iou: float | None,
    max_depth_mae: float | None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    manifest = load_json(manifest_path)
    qa_rows = load_qa_rows(qa_path)
    ranked: list[tuple[float, float, float, int, dict[str, Any], dict[str, Any]]] = []
    for sample in manifest.get("samples", []):
        row = qa_rows.get(sample["sample_id"])
        if not row:
            continue
        if visible_teacher_players(row) <= 0 and teacher_player_pixels(row) <= 0:
            continue
        if memory_player_pixels(row) <= 0:
            continue
        other_iou = float(row.get("other_player_iou", 0.0) or 0.0)
        if other_iou < min_other_iou:
            continue
        hit_iou = float(row.get("hit_iou", 0.0) or 0.0)
        depth_mae = qa_depth_mae(row)
        if min_hit_iou is not None and hit_iou < min_hit_iou:
            continue
        if max_depth_mae is not None:
            if depth_mae is None or depth_mae > max_depth_mae:
                continue
        depth_rank = -float(depth_mae) if depth_mae is not None else -999.0
        ranked.append((other_iou, depth_rank, hit_iou, teacher_player_pixels(row), sample, row))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [(sample, row) for _, _, _, _, sample, row in ranked[:max_per_episode]]


def select_context_rows(
    manifest_path: Path,
    qa_path: Path,
    max_per_episode: int,
    min_hit_iou: float,
    max_depth_mae: float,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    manifest = load_json(manifest_path)
    qa_rows = load_qa_rows(qa_path)
    ranked: list[tuple[float, float, dict[str, Any], dict[str, Any]]] = []
    for sample in manifest.get("samples", []):
        row = qa_rows.get(sample["sample_id"])
        if not row:
            continue
        if visible_teacher_players(row) != 0 or teacher_player_pixels(row) != 0:
            continue
        if memory_player_pixels(row) != 0:
            continue
        hit_iou = float(row.get("hit_iou", 0.0) or 0.0)
        depth_mae = qa_depth_mae(row)
        if depth_mae is None:
            continue
        if hit_iou < min_hit_iou or depth_mae > max_depth_mae:
            continue
        ranked.append((depth_mae, -hit_iou, sample, row))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [(sample, row) for _, _, sample, row in ranked[:max_per_episode]]


def sample_stats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_episode: dict[str, dict[str, int]] = {}
    for sample in samples:
        ep = str(sample.get("episode"))
        role = str(sample.get("selection_role"))
        row = by_episode.setdefault(ep, {"total": 0, "positive": 0, "context": 0})
        row["total"] += 1
        row[role] = row.get(role, 0) + 1
    return {
        "sample_count": len(samples),
        "positive_sample_count": sum(1 for s in samples if s.get("selection_role") == "positive"),
        "context_sample_count": sum(1 for s in samples if s.get("selection_role") == "context"),
        "episode_sample_counts": by_episode,
    }


def selection_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    return {
        "hit_iou_mean": mean([float(r.get("hit_iou")) for r in rows if r.get("hit_iou") is not None]),
        "depth_common_mae_mean": mean([v for r in rows if (v := qa_depth_mae(r)) is not None]),
        "other_player_iou_mean": mean([
            float(r.get("other_player_iou"))
            for r in rows
            if r.get("other_player_iou") is not None
        ]),
        "other_player_iou_min": min(
            [float(r.get("other_player_iou")) for r in rows if r.get("other_player_iou") is not None],
            default=None,
        ),
    }


def assert_compatible(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    if not manifests:
        raise ValueError("no manifests to combine")
    first_sample = manifests[0].get("samples", [])[0]
    channels = first_sample.get("channels")
    player_params = manifests[0].get("player_capsule_params") or {}
    camera_params = manifests[0].get("camera_projection_params") or {}
    mismatches = []
    for manifest in manifests:
        sample0 = manifest.get("samples", [])[0]
        if sample0.get("channels") != channels:
            mismatches.append("channels")
        if (manifest.get("player_capsule_params") or {}) != player_params:
            mismatches.append("player_capsule_params")
        if (manifest.get("camera_projection_params") or {}) != camera_params:
            mismatches.append("camera_projection_params")
    if mismatches:
        raise ValueError(f"incompatible source manifests: {sorted(set(mismatches))}")
    return {
        "channels": channels,
        "player_capsule_params": player_params,
        "camera_projection_params": camera_params,
    }


def build_episode(
    args: argparse.Namespace,
    tools: Path,
    env: dict[str, str],
    episode: str,
) -> dict[str, Path]:
    ep_root = args.work_dir / episode
    memory_dir = ep_root / "episode_memory_v0"
    positive_dir = ep_root / "positive"
    context_dir = ep_root / "context_candidates"
    positive_qa_dir = positive_dir / "channel_teacher_qa_v0"
    context_qa_dir = context_dir / "channel_teacher_qa_v0"
    candidate_json = positive_dir / "teacher_other_player_candidates_v0.json"

    run_or_reuse([
        sys.executable, str(tools / "build_episode_memory_v0.py"),
        "--match-dir", str(args.match_dir),
        "--episode", episode,
        "--out-dir", str(memory_dir),
    ], env=env, outputs=[memory_dir / "episode_memory_v0.npz", memory_dir / "episode_memory_meta_v0.json"], reuse_existing=args.reuse_existing)
    run_or_reuse([
        sys.executable, str(tools / "find_teacher_other_player_candidates_v0.py"),
        "--match-dir", str(args.match_dir),
        "--episode", episode,
        "--out-json", str(candidate_json),
        "--ego-limit", str(args.positive_ego_limit),
        "--frames-per-ego", str(args.positive_frames_per_ego),
        "--top-k", str(args.positive_top_k),
        "--start", "0",
        "--stop", str(args.stop),
        "--stride", "16",
        "--height", str(args.height),
        "--width", str(args.width),
        "--min-teacher-pixels", "32",
        "--min-pixel-percent", str(args.min_pixel_percent),
        "--max-scan-per-ego", "24",
    ], env=env, outputs=[candidate_json], reuse_existing=args.reuse_existing)

    common_export = [
        "--match-dir", str(args.match_dir),
        "--episode", episode,
        "--episode-memory-dir", str(memory_dir),
        "--width", str(args.width),
        "--height", str(args.height),
        "--fov-x", "110",
        "--far", "3000",
        "--player-radius", "14",
        "--player-height", "40",
        "--occlusion-tolerance", "40",
        "--min-visible-pixels", "48",
        "--player-mask-mode", "capsule",
        "--geometry-backend", "bsp_faces",
        "--bsp-faces-npz", str(args.bsp_faces_npz),
    ]
    run_or_reuse([
        sys.executable, str(tools / "export_memory_dense_dataset_v0.py"),
        *common_export,
        "--out-dir", str(positive_dir),
        "--candidate-json", str(candidate_json),
        "--ego-limit", str(args.positive_ego_limit),
        "--frames-per-ego", str(args.positive_frames_per_ego),
    ], env=env, outputs=[positive_dir / "manifest.json"], reuse_existing=args.reuse_existing)
    positive_manifest = positive_dir / "manifest.json"
    positive_count = int(load_json(positive_manifest).get("sample_count", 0))
    run_or_reuse([
        sys.executable, str(tools / "compare_memory_dense_channels_to_teacher_v0.py"),
        "--manifest", str(positive_manifest),
        "--match-dir", str(args.match_dir),
        "--out-dir", str(positive_qa_dir),
        "--max-samples", str(positive_count),
        "--make-qa",
    ], env=env, outputs=[positive_qa_dir / "memory_dense_channels_vs_teacher_v0.json"], reuse_existing=args.reuse_existing)

    run_or_reuse([
        sys.executable, str(tools / "export_memory_dense_dataset_v0.py"),
        *common_export,
        "--out-dir", str(context_dir),
        "--ego-limit", str(args.context_ego_limit),
        "--frames-per-ego", str(args.context_frames_per_ego),
        "--start", str(args.context_start),
        "--stop", str(args.stop),
        "--stride", str(args.context_stride),
    ], env=env, outputs=[context_dir / "manifest.json"], reuse_existing=args.reuse_existing)
    context_manifest = context_dir / "manifest.json"
    context_count = int(load_json(context_manifest).get("sample_count", 0))
    run_or_reuse([
        sys.executable, str(tools / "compare_memory_dense_channels_to_teacher_v0.py"),
        "--manifest", str(context_manifest),
        "--match-dir", str(args.match_dir),
        "--out-dir", str(context_qa_dir),
        "--max-samples", str(context_count),
        "--make-qa",
    ], env=env, outputs=[context_qa_dir / "memory_dense_channels_vs_teacher_v0.json"], reuse_existing=args.reuse_existing)

    return {
        "memory_dir": memory_dir,
        "positive_manifest": positive_manifest,
        "positive_qa": positive_qa_dir / "memory_dense_channels_vs_teacher_v0.json",
        "context_manifest": context_manifest,
        "context_qa": context_qa_dir / "memory_dense_channels_vs_teacher_v0.json",
        "candidate_json": candidate_json,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-dir", type=Path, default=DEFAULT_MATCH_DIR)
    ap.add_argument("--episodes", nargs="+", default=DEFAULT_EPISODES)
    ap.add_argument("--work-dir", type=Path, default=Path("output/map_memory_multi_episode_bsp_v0"))
    ap.add_argument("--out-dir", type=Path, default=Path("docs/assets/memory_dense_dataset_v35_bsp_multi_episode_h176"))
    ap.add_argument("--bsp-faces-npz", type=Path, default=DEFAULT_BSP_FACES_NPZ)
    ap.add_argument("--bsp-parity", type=Path, default=Path("docs/assets/bsp_static_geometry_v0/obj_vs_bsp_disp_v0/geometry_backend_comparison_v0.json"))
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=176)
    ap.add_argument("--stop", type=int, default=1800)
    ap.add_argument("--positive-ego-limit", type=int, default=10)
    ap.add_argument("--positive-frames-per-ego", type=int, default=3)
    ap.add_argument("--positive-top-k", type=int, default=30)
    ap.add_argument("--positive-keep-per-episode", type=int, default=18)
    ap.add_argument("--context-ego-limit", type=int, default=10)
    ap.add_argument("--context-frames-per-ego", type=int, default=8)
    ap.add_argument("--context-keep-per-episode", type=int, default=4)
    ap.add_argument("--context-start", type=int, default=0)
    ap.add_argument("--context-stride", type=int, default=96)
    ap.add_argument("--min-pixel-percent", type=float, default=0.02)
    ap.add_argument("--min-positive-other-iou", type=float, default=0.25)
    ap.add_argument("--min-positive-hit-iou", type=float, default=None)
    ap.add_argument("--max-positive-depth-mae", type=float, default=None)
    ap.add_argument("--min-context-hit-iou", type=float, default=0.90)
    ap.add_argument("--max-context-depth-mae", type=float, default=0.08)
    ap.add_argument("--min-positive-per-episode", type=int, default=3)
    ap.add_argument("--min-context-per-episode", type=int, default=1)
    ap.add_argument("--reuse-existing", action="store_true")
    ap.add_argument("--clean-out-dir", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    tools = Path(__file__).resolve().parent
    env = gpu_env()
    if args.force:
        shutil.rmtree(args.work_dir, ignore_errors=True)
        shutil.rmtree(args.out_dir, ignore_errors=True)
    elif args.clean_out_dir:
        shutil.rmtree(args.out_dir, ignore_errors=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    episode_outputs = {}
    for episode in args.episodes:
        episode_outputs[episode] = build_episode(args, tools, env, episode)

    source_manifests = []
    copied_samples = []
    selection_report = {}
    for episode, paths in episode_outputs.items():
        positive = select_positive_rows(
            paths["positive_manifest"],
            paths["positive_qa"],
            args.positive_keep_per_episode,
            args.min_positive_other_iou,
            args.min_positive_hit_iou,
            args.max_positive_depth_mae,
        )
        context = select_context_rows(
            paths["context_manifest"],
            paths["context_qa"],
            args.context_keep_per_episode,
            args.min_context_hit_iou,
            args.max_context_depth_mae,
        )
        if len(positive) < args.min_positive_per_episode:
            raise RuntimeError(f"{episode}: selected only {len(positive)} positive rows")
        if len(context) < args.min_context_per_episode:
            raise RuntimeError(f"{episode}: selected only {len(context)} teacher-empty context rows")
        source_manifests.extend([load_json(paths["positive_manifest"]), load_json(paths["context_manifest"])])
        selection_report[episode] = {
            "positive_selected": len(positive),
            "context_selected": len(context),
            "positive_selected_metrics": selection_metrics([row for _, row in positive]),
            "context_selected_metrics": selection_metrics([row for _, row in context]),
            "positive_manifest": str(paths["positive_manifest"]),
            "positive_teacher_qa": str(paths["positive_qa"]),
            "context_manifest": str(paths["context_manifest"]),
            "context_teacher_qa": str(paths["context_qa"]),
        }
        for sample, row in positive:
            copied_samples.append(copy_sample(args.out_dir, paths["positive_manifest"], sample, "positive", row))
        for sample, row in context:
            copied_samples.append(copy_sample(args.out_dir, paths["context_manifest"], sample, "context", row))

    compatibility = assert_compatible(source_manifests)
    stats = sample_stats(copied_samples)
    backend_ids = sorted({str(s.get("geometry_backend_id")) for s in copied_samples if s.get("geometry_backend_id")})
    backend_signatures = sorted({str(s.get("backend_signature")) for s in copied_samples if s.get("backend_signature")})
    backend_features = sorted({str(f) for s in copied_samples for f in (s.get("backend_features") or [])})
    manifest = {
        "kind": "multi_episode_bsp_memory_dense_manifest_v0",
        "policy": "First-person Map Memory dense inputs only. Teacher RGB/depth/seg/visibility streams are used for sampling, QA, and targets, not as input channels.",
        "match_dir": str(args.match_dir),
        "episodes": list(args.episodes),
        "episode_count": len(args.episodes),
        "channels": CHANNELS,
        "shape": [7, args.height, args.width],
        "geometry_backend_id": backend_ids[0] if len(backend_ids) == 1 else "mixed",
        "geometry_backend_ids": backend_ids,
        "backend_signatures": backend_signatures,
        "backend_features": backend_features,
        "player_capsule_params": compatibility["player_capsule_params"],
        "camera_projection_params": compatibility["camera_projection_params"],
        "selection_policy": {
            "positive": {
                "min_positive_other_iou": args.min_positive_other_iou,
                "min_hit_iou": args.min_positive_hit_iou,
                "max_depth_common_mae": args.max_positive_depth_mae,
                "min_positive_per_episode": args.min_positive_per_episode,
                "max_keep_per_episode": args.positive_keep_per_episode,
            },
            "context": {
                "teacher_other_player_pixels": 0,
                "memory_other_player_pixels": 0,
                "min_hit_iou": args.min_context_hit_iou,
                "max_depth_common_mae": args.max_context_depth_mae,
                "min_context_per_episode": args.min_context_per_episode,
                "max_keep_per_episode": args.context_keep_per_episode,
            },
        },
        "input_provenance": {
            "bsp_faces_npz": file_fingerprint(args.bsp_faces_npz),
            "bsp_parity": file_fingerprint(args.bsp_parity),
            "episode_memories": [file_fingerprint(paths["memory_dir"] / "episode_memory_meta_v0.json") for paths in episode_outputs.values()],
            "source_manifests": [
                file_fingerprint(path)
                for paths in episode_outputs.values()
                for path in [paths["positive_manifest"], paths["context_manifest"]]
            ],
            "teacher_qa": [
                file_fingerprint(path)
                for paths in episode_outputs.values()
                for path in [paths["positive_qa"], paths["context_qa"]]
            ],
        },
        "selection_report": selection_report,
        **stats,
        "samples": copied_samples,
    }
    manifest_path = args.out_dir / "manifest.json"
    write_json(manifest_path, manifest)

    qa_dir = args.out_dir / "channel_teacher_qa_v0"
    qa_json = qa_dir / "memory_dense_channels_vs_teacher_v0.json"
    run([
        sys.executable, str(tools / "compare_memory_dense_channels_to_teacher_v0.py"),
        "--manifest", str(manifest_path),
        "--match-dir", str(args.match_dir),
        "--out-dir", str(qa_dir),
        "--max-samples", str(manifest["sample_count"]),
        "--make-qa",
    ], env=env)
    run([
        sys.executable, str(tools / "check_memory_dense_manifest_v0.py"),
        "--manifest", str(manifest_path),
        "--max-samples", str(manifest["sample_count"]),
    ], env=env)
    run([
        sys.executable, str(tools / "check_map_memory_training_readiness_v0.py"),
        "--manifest", str(manifest_path),
        "--teacher-qa", str(qa_json),
        "--mode", "mixed",
        "--bsp-parity", str(args.bsp_parity),
        "--max-samples", str(manifest["sample_count"]),
        "--min-samples", str(manifest["sample_count"]),
        "--min-episodes", str(len(args.episodes)),
        "--require-geometry-backend-id", "bsp_faces_disp_gpu",
        "--out-json", str(args.out_dir / "training_readiness_v0.json"),
    ], env=env)
    run([
        sys.executable, str(tools / "summarize_memory_dense_dataset_v0.py"),
        "--manifest", str(manifest_path),
        "--out-json", str(args.out_dir / "summary_v0.json"),
        "--out-montage", str(args.out_dir / "qa_montage_v0.png"),
        "--max-montage-images", "8",
    ], env=env)

    teacher_qa = load_json(qa_json)
    readiness = load_json(args.out_dir / "training_readiness_v0.json")
    release = {
        "kind": "map_memory_multi_episode_bsp_release_v0",
        "status": readiness.get("status"),
        "manifest": str(manifest_path),
        "teacher_qa": str(qa_json),
        "training_readiness": str(args.out_dir / "training_readiness_v0.json"),
        "sample_count": manifest["sample_count"],
        "episode_count": manifest["episode_count"],
        "positive_sample_count": manifest["positive_sample_count"],
        "context_sample_count": manifest["context_sample_count"],
        "teacher_qa_summary": teacher_qa.get("summary"),
        "readiness_failures": readiness.get("failures", []),
        "known_limits": [
            "This is a multi-episode adapter-data gate, not a full-dataset export.",
            "BSP geometry currently uses visual faces plus expanded displacement surfaces; static-prop mesh parity and brush collision queries remain follow-up work.",
            "Other-player masks are capsule proxies. They are sufficient for adapter condition training gates, but not final silhouette supervision.",
        ],
    }
    write_json(args.out_dir / "map_memory_multi_episode_bsp_release_v0.json", release)
    print(json.dumps(release, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
