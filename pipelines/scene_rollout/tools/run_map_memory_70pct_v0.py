#!/usr/bin/env python3
"""Build and validate the 70-percent Map Memory release dataset.

This runner intentionally keeps the current engineering target conservative:
OBJ/GPU or BSP+displacement first-person depth, nav-place semantic fallback,
replay-state capsules, alive ego filtering, teacher-only QA, and strict
manifest gates. It is the reproducible map-side pipeline that should not rely on
hand-run shell fragments.
"""

from __future__ import annotations
from runtime_paths import source_path

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_MATCH_DIR = Path(str(source_path('assets', 'csgo-datasets-fullsubset/32f1644d4f42c29d/030a506e5e77459c8d5078a06eec61ee')))
DEFAULT_EPISODE = "Ep_000015"
DEFAULT_BSP_FACES_NPZ = Path("docs/assets/bsp_static_geometry_v0/de_dust2_bsp_faces_v0.npz")
DEFAULT_OUT_ROOT = Path("docs/assets")
OBJ_GPU_POSITIVE_SUFFIX = "v24_map70_positive_alive_h176"
OBJ_GPU_MIXED_SUFFIX = "v25_map70_mixed_alive_h176"
BSP_FACES_POSITIVE_SUFFIX = "v26_bsp_faces_positive_alive_h176"
BSP_FACES_MIXED_SUFFIX = "v27_bsp_faces_mixed_alive_h176"
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


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def gpu_env() -> dict[str, str]:
    env = os.environ.copy()
    cuda = env.get("CUDA_PATH", "/usr/local/cuda-12.6")
    env["CUDA_PATH"] = cuda
    env["PATH"] = f"{cuda}/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = f"{cuda}/lib64:" + env.get("LD_LIBRARY_PATH", "")
    return env


def normalize_geometry_backend(value: str) -> str:
    aliases = {
        "obj": "obj_gpu",
        "gpu": "obj_gpu",
        "obj/gpu": "obj_gpu",
        "obj_gpu": "obj_gpu",
        "bsp": "bsp_faces",
        "bsp_faces": "bsp_faces",
    }
    key = value.strip().lower()
    if key not in aliases:
        raise argparse.ArgumentTypeError(
            f"unsupported geometry backend {value!r}; use obj_gpu/obj/gpu or bsp_faces"
        )
    return aliases[key]


def default_out_dirs(geometry_backend: str, out_root: Path) -> tuple[Path, Path]:
    if geometry_backend == "bsp_faces":
        return (
            out_root / f"memory_dense_dataset_{BSP_FACES_POSITIVE_SUFFIX}",
            out_root / f"memory_dense_dataset_{BSP_FACES_MIXED_SUFFIX}",
        )
    return (
        out_root / f"memory_dense_dataset_{OBJ_GPU_POSITIVE_SUFFIX}",
        out_root / f"memory_dense_dataset_{OBJ_GPU_MIXED_SUFFIX}",
    )


def exporter_supports_arg(exporter: Path, arg_name: str) -> bool:
    text = exporter.read_text(encoding="utf-8")
    return arg_name in text


def export_geometry_args(args: argparse.Namespace, exporter: Path) -> list[str]:
    if args.geometry_backend == "obj_gpu":
        return ["--mesh-backend", "gpu"]

    if args.bsp_faces_npz is None:
        raise ValueError("--bsp-faces-npz is required when --geometry-backend bsp_faces")
    if not args.bsp_faces_npz.exists():
        raise FileNotFoundError(f"BSP faces npz not found: {args.bsp_faces_npz}")

    required_args = ["--geometry-backend", "--bsp-faces-npz"]
    missing = [name for name in required_args if not exporter_supports_arg(exporter, name)]
    if missing:
        raise RuntimeError(
            "bsp_faces was requested, but export_memory_dense_dataset_v0.py does not yet expose "
            f"{', '.join(missing)}. This runner is ready to pass BSP faces through; land the "
            "renderer/exporter backend first or use --geometry-backend obj_gpu."
        )
    return [
        "--geometry-backend", "bsp_faces",
        "--bsp-faces-npz", str(args.bsp_faces_npz),
    ]


def assert_metric(name: str, value: float | None, op: str, threshold: float) -> None:
    if value is None:
        raise AssertionError(f"{name} is missing")
    ok = value >= threshold if op == ">=" else value <= threshold
    if not ok:
        raise AssertionError(f"{name}={value:.6f} failed gate {op} {threshold:.6f}")


def validate_manifest(
    manifest_path: Path,
    expected_count_min: int,
    allowed_mesh_backends: set[str | None],
    require_channels: bool = True,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    samples = manifest.get("samples", [])
    if manifest.get("sample_count") != len(samples):
        raise AssertionError(f"{manifest_path}: sample_count does not match samples length")
    if len(samples) < expected_count_min:
        raise AssertionError(f"{manifest_path}: sample_count {len(samples)} < {expected_count_min}")
    for sample in samples:
        if tuple(sample.get("shape", [])) != (7, 176, 320):
            raise AssertionError(f"{sample.get('sample_id')}: bad shape {sample.get('shape')}")
        if require_channels and sample.get("channels") != CHANNELS:
            raise AssertionError(f"{sample.get('sample_id')}: channel contract mismatch")
        if sample.get("mesh_backend") not in allowed_mesh_backends:
            raise AssertionError(f"{sample.get('sample_id')}: unexpected mesh_backend {sample.get('mesh_backend')}")
    return manifest


def validate_ego_alive(manifest: dict[str, Any], episode_memory_dir: Path) -> None:
    import numpy as np

    meta = load_json(episode_memory_dir / "episode_memory_meta_v0.json")
    mem = np.load(episode_memory_dir / "episode_memory_v0.npz")
    for sample in manifest.get("samples", []):
        idx = meta["player_stems"].index(sample["ego_stem"])
        frame = int(sample["frame_index"])
        if not bool(mem["alive"][idx, frame]):
            raise AssertionError(f"{sample['sample_id']}: ego is dead/observer in release manifest")


def validate_qa(qa_path: Path, positive: bool) -> dict[str, Any]:
    data = load_json(qa_path)
    if data.get("missing_count") != 0:
        raise AssertionError(f"{qa_path}: missing_count={data.get('missing_count')}")
    summary = data["summary"]
    assert_metric("hit_iou_mean", summary.get("hit_iou_mean"), ">=", 0.90)
    assert_metric("depth_common_mae_mean", summary.get("depth_common_mae_mean"), "<=", 0.08)
    assert_metric("other_player_depth_common_mae_mean", summary.get("other_player_depth_common_mae_mean"), "<=", 0.03)
    if positive:
        assert_metric("positive other_player_iou_mean", summary.get("other_player_iou_mean"), ">=", 0.45)
        for row in data.get("rows", []):
            if int(row.get("visible_teacher_players", 0)) <= 0:
                raise AssertionError(f"{row.get('sample_id')}: positive QA row has no visible teacher players")
            if int(row.get("other_player_memory_pixels", 0)) <= 0:
                raise AssertionError(f"{row.get('sample_id')}: positive QA row has empty Memory player mask")
    else:
        assert_metric("mixed other_player_iou_mean", summary.get("other_player_iou_mean"), ">=", 0.55)
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-dir", type=Path, default=DEFAULT_MATCH_DIR)
    ap.add_argument("--episode", default=DEFAULT_EPISODE)
    ap.add_argument("--episode-memory-dir", type=Path, default=Path("output/episode_memory_ep15_v0"))
    ap.add_argument("--negative-manifest", type=Path, default=Path("docs/assets/memory_dense_dataset_v19_visibility_strict_h176/manifest.json"))
    ap.add_argument("--geometry-backend", type=normalize_geometry_backend, default="obj_gpu", help="Static geometry backend: obj_gpu/obj/gpu for the OBJ CUDA path, or bsp_faces for the BSP+displacement mesh npz.")
    ap.add_argument("--bsp-faces-npz", type=Path, default=DEFAULT_BSP_FACES_NPZ, help="BSP+displacement mesh npz used by --geometry-backend bsp_faces.")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--positive-out-dir", type=Path, default=None)
    ap.add_argument("--mixed-out-dir", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    tools = Path(__file__).resolve().parent
    exporter = tools / "export_memory_dense_dataset_v0.py"
    env = gpu_env()
    env["MAP_MEMORY_GEOMETRY_BACKEND"] = args.geometry_backend
    if args.bsp_faces_npz is not None:
        env["MAP_MEMORY_BSP_FACES_NPZ"] = str(args.bsp_faces_npz)

    default_positive, default_mixed = default_out_dirs(args.geometry_backend, args.out_root)
    positive = args.positive_out_dir or default_positive
    mixed = args.mixed_out_dir or default_mixed
    geometry_export_args = export_geometry_args(args, exporter)
    positive_allowed_mesh_backends: set[str | None] = (
        {"gpu", None} if args.geometry_backend == "obj_gpu" else {"bsp_faces_gpu", "bsp_faces_cpu"}
    )
    # Mixed releases may inherit the current v19 OBJ/GPU negative/context slice
    # until a backend-matched negative manifest is supplied by the caller.
    mixed_allowed_mesh_backends = set(positive_allowed_mesh_backends)
    mixed_allowed_mesh_backends.update({"gpu", None})
    if args.force:
        import shutil

        shutil.rmtree(positive, ignore_errors=True)
        shutil.rmtree(mixed, ignore_errors=True)

    candidate_json = positive / "teacher_other_player_candidates_v0.json"
    positive_manifest = positive / "manifest.json"
    positive_qa_dir = positive / "channel_teacher_qa_v0"
    positive_qa_json = positive_qa_dir / "memory_dense_channels_vs_teacher_v0.json"
    mixed_manifest = mixed / "manifest.json"
    mixed_qa_dir = mixed / "channel_teacher_qa_v0"
    mixed_qa_json = mixed_qa_dir / "memory_dense_channels_vs_teacher_v0.json"

    run([
        sys.executable, str(tools / "find_teacher_other_player_candidates_v0.py"),
        "--match-dir", str(args.match_dir),
        "--episode", args.episode,
        "--out-json", str(candidate_json),
        "--ego-limit", "10",
        "--frames-per-ego", "4",
        "--top-k", "24",
        "--start", "0",
        "--stop", "1800",
        "--stride", "16",
        "--height", "176",
        "--width", "320",
        "--min-teacher-pixels", "32",
        "--min-pixel-percent", "0.05",
        "--max-scan-per-ego", "24",
    ], env=env)

    run([
        sys.executable, str(exporter),
        "--match-dir", str(args.match_dir),
        "--episode", args.episode,
        "--out-dir", str(positive),
        "--episode-memory-dir", str(args.episode_memory_dir),
        "--candidate-json", str(candidate_json),
        "--ego-limit", "6",
        "--frames-per-ego", "3",
        "--width", "320",
        "--height", "176",
        "--fov-x", "110",
        "--far", "3000",
        "--player-radius", "14",
        "--player-height", "40",
        "--occlusion-tolerance", "40",
        "--min-visible-pixels", "48",
        "--player-mask-mode", "capsule",
        *geometry_export_args,
    ], env=env)

    positive_manifest_obj = validate_manifest(
        positive_manifest,
        expected_count_min=12,
        allowed_mesh_backends=positive_allowed_mesh_backends,
    )
    validate_ego_alive(positive_manifest_obj, args.episode_memory_dir)

    run([
        sys.executable, str(tools / "compare_memory_dense_channels_to_teacher_v0.py"),
        "--manifest", str(positive_manifest),
        "--match-dir", str(args.match_dir),
        "--out-dir", str(positive_qa_dir),
        "--max-samples", str(positive_manifest_obj["sample_count"]),
        "--make-qa",
    ], env=env)
    positive_qa = validate_qa(positive_qa_json, positive=True)

    run([
        sys.executable, str(tools / "build_balanced_manifest_v0.py"),
        "--base-manifest", str(args.negative_manifest),
        "--player-manifest", str(positive_manifest),
        "--out-dir", str(mixed),
        "--max-base", "4",
        "--max-player", str(positive_manifest_obj["sample_count"]),
        "--min-player-pixels", "1",
        "--copy-files",
    ], env=env)

    mixed_manifest_obj = validate_manifest(
        mixed_manifest,
        expected_count_min=16,
        allowed_mesh_backends=mixed_allowed_mesh_backends,
    )
    run([
        sys.executable, str(tools / "compare_memory_dense_channels_to_teacher_v0.py"),
        "--manifest", str(mixed_manifest),
        "--match-dir", str(args.match_dir),
        "--out-dir", str(mixed_qa_dir),
        "--max-samples", str(mixed_manifest_obj["sample_count"]),
        "--make-qa",
    ], env=env)
    mixed_qa = validate_qa(mixed_qa_json, positive=False)

    run([
        sys.executable, str(tools / "check_memory_dense_manifest_v0.py"),
        "--manifest", str(mixed_manifest),
        "--max-samples", str(mixed_manifest_obj["sample_count"]),
    ], env=env)
    run([
        sys.executable, str(tools / "summarize_memory_dense_dataset_v0.py"),
        "--manifest", str(mixed_manifest),
        "--out-json", str(mixed / "summary_v0.json"),
        "--out-montage", str(mixed / "qa_montage_v0.png"),
        "--max-montage-images", "6",
    ], env=env)

    release = {
        "kind": "map_memory_70pct_release_v0",
        "status": "pass",
        "contract": {
            "shape": [7, 176, 320],
            "channels": CHANNELS,
            "input_policy": "Map Memory dense only; teacher streams are QA/target only.",
            "state_policy": "Replay alive ego frames only for this release; dynamics rollout is external.",
            "geometry_backend": args.geometry_backend,
            "bsp_faces_npz": str(args.bsp_faces_npz) if args.geometry_backend == "bsp_faces" else None,
        },
        "positive_manifest": str(positive_manifest),
        "mixed_manifest": str(mixed_manifest),
        "negative_manifest": str(args.negative_manifest),
        "positive_sample_count": positive_manifest_obj["sample_count"],
        "mixed_sample_count": mixed_manifest_obj["sample_count"],
        "positive_qa_summary": positive_qa["summary"],
        "mixed_qa_summary": mixed_qa["summary"],
        "known_limits": [
            "bsp_faces currently means BSP visual faces plus expanded displacement surfaces; brush collision and static-prop meshes are still separate work.",
            "Player masks are capsule proxies; silhouette IoU is not final.",
            "Evaluation is still Ep_000015 smoke scale, not full dataset scale.",
        ],
    }
    write_json(mixed / "map_memory_70pct_release_v0.json", release)
    print(json.dumps(release, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
