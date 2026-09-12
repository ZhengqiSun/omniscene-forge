#!/usr/bin/env python3
"""Benchmark CPU and GPU Map Memory mesh renderers on one camera frame."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def import_tool(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timed(fn):
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start


def summarize_depth_diff(cpu_depth: np.ndarray, gpu_depth: np.ndarray) -> dict[str, Any]:
    cpu_hit = np.isfinite(cpu_depth)
    gpu_hit = np.isfinite(gpu_depth)
    both = cpu_hit & gpu_hit
    union = cpu_hit | gpu_hit
    out: dict[str, Any] = {
        "cpu_hit_ratio": float(cpu_hit.mean()),
        "gpu_hit_ratio": float(gpu_hit.mean()),
        "hit_iou": float(both.sum() / max(1, union.sum())),
        "cpu_only_pixels": int((cpu_hit & ~gpu_hit).sum()),
        "gpu_only_pixels": int((gpu_hit & ~cpu_hit).sum()),
        "both_hit_pixels": int(both.sum()),
    }
    if np.any(both):
        diff = np.abs(cpu_depth[both] - gpu_depth[both])
        out.update(
            {
                "abs_depth_diff_mean": float(diff.mean()),
                "abs_depth_diff_median": float(np.median(diff)),
                "abs_depth_diff_p95": float(np.percentile(diff, 95)),
                "abs_depth_diff_max": float(diff.max()),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-dir", type=Path, required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--ego-stem", required=True)
    ap.add_argument("--frame-index", type=int, required=True)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=176)
    ap.add_argument("--fov-x", type=float, default=110.0)
    ap.add_argument("--near", type=float, default=4.0)
    ap.add_argument("--far", type=float, default=3000.0)
    ap.add_argument("--pitch-sign", type=float, default=1.0)
    ap.add_argument("--max-triangles", type=int, default=90000)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    tools_dir = Path(__file__).resolve().parent
    dense_mod = import_tool(tools_dir / "build_mesh_dense_condition_v0.py", "mesh_dense_condition_v0")
    gpu_mod = import_tool(tools_dir / "gpu_mesh_renderer_v0.py", "gpu_mesh_renderer_v0")
    cache, cache_s = timed(lambda: dense_mod.load_renderer_cache(args.match_dir, tools_dir))

    episode_dir = args.match_dir / "train" / args.episode
    frames = dense_mod.load_json(episode_dir / f"{args.ego_stem}.json")
    frame = frames[args.frame_index]
    cam_pos = np.asarray(
        frame.get("camera_position") or [frame["x"], frame["y"], frame["z"] + 64.0],
        dtype=np.float32,
    )
    yaw = float(frame.get("yaw", frame.get("camera_rotation", [0, 0, 0])[2]))
    pitch = float(frame.get("pitch", frame.get("camera_rotation", [0, 0, 0])[1]))
    mesh_mod = cache["mesh_mod"]

    (projected, _), project_s = timed(
        lambda: mesh_mod.project_vertices(
            cache["vertices"],
            cam_pos,
            yaw,
            pitch,
            args.pitch_sign,
            args.width,
            args.height,
            args.fov_x,
        )
    )
    (cpu_depth, cpu_stats), cpu_s = timed(
        lambda: mesh_mod.rasterize_depth(
            projected,
            cache["faces"],
            args.width,
            args.height,
            args.near,
            args.far,
            args.max_triangles,
        )
    )

    gpu_renderer, gpu_cache_s = timed(lambda: gpu_mod.GpuMeshDepthRenderer.from_numpy(cache["vertices"], cache["faces"]))
    gpu_runs = []
    gpu_depth = None
    gpu_stats = None
    for idx in range(args.iters):
        (gpu_depth_i, gpu_stats_i), gpu_s = timed(
            lambda: gpu_mod.rasterize_depth_gpu(
                gpu_renderer,
                cam_pos,
                yaw,
                pitch,
                args.pitch_sign,
                args.width,
                args.height,
                args.fov_x,
                args.near,
                args.far,
                args.max_triangles,
            )
        )
        gpu_depth = gpu_depth_i
        gpu_stats = gpu_stats_i
        gpu_runs.append({"iter": idx, "render_s": gpu_s})

    assert gpu_depth is not None and gpu_stats is not None
    result = {
        "match_dir": str(args.match_dir),
        "episode": args.episode,
        "ego_stem": args.ego_stem,
        "frame_index": args.frame_index,
        "shape": [args.height, args.width],
        "mesh_vertices": int(len(cache["vertices"])),
        "mesh_faces": int(len(cache["faces"])),
        "cache_load_s": cache_s,
        "cpu": {
            "project_s": project_s,
            "raster_s": cpu_s,
            "total_mesh_s": project_s + cpu_s,
            "stats": cpu_stats,
        },
        "gpu": {
            "cache_to_gpu_s": gpu_cache_s,
            "runs": gpu_runs,
            "first_render_s": gpu_runs[0]["render_s"],
            "mean_render_s": float(np.mean([r["render_s"] for r in gpu_runs])),
            "best_render_s": float(np.min([r["render_s"] for r in gpu_runs])),
            "stats": gpu_stats,
        },
        "depth_compare_vs_cpu_v0": summarize_depth_diff(cpu_depth, gpu_depth),
        "note": (
            "CPU v0 uses screen-space linear z interpolation. GPU backend uses "
            "perspective-correct camera z, so depth differences on shared pixels "
            "are expected and are a correctness improvement, not just numerical noise."
        ),
    }
    result["speedup_vs_cpu_raster_best"] = float(cpu_s / max(result["gpu"]["best_render_s"], 1e-9))

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

