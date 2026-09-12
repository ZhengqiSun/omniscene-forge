#!/usr/bin/env python3
"""Diagnose Map Memory player projection and depth visibility for one frame."""

from __future__ import annotations

import argparse
import importlib.util
import json
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-dir", type=Path, required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--ego-stem", required=True)
    ap.add_argument("--frame-index", type=int, required=True)
    ap.add_argument("--episode-memory-dir", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=176)
    ap.add_argument("--fov-x", type=float, default=110.0)
    ap.add_argument("--far", type=float, default=3000.0)
    ap.add_argument("--near", type=float, default=4.0)
    ap.add_argument("--pitch-sign", type=float, default=1.0)
    ap.add_argument("--max-triangles", type=int, default=0)
    ap.add_argument("--player-radius", type=float, default=14.0)
    ap.add_argument("--player-height", type=float, default=40.0)
    ap.add_argument("--occlusion-tolerance", type=float, default=40.0)
    ap.add_argument("--min-visible-pixels", type=int, default=48)
    ap.add_argument("--player-mask-mode", choices=["capsule", "multipart"], default="capsule")
    ap.add_argument("--mesh-backend", choices=["cpu", "gpu", "auto"], default="gpu")
    args = ap.parse_args()

    tools_dir = Path(__file__).resolve().parent
    dense_mod = import_tool(tools_dir / "build_mesh_dense_condition_v0.py", "mesh_dense_condition_v0")
    cache = dense_mod.load_renderer_cache(args.match_dir, tools_dir)
    episode_memory = dense_mod.load_episode_memory(args.episode_memory_dir) if args.episode_memory_dir else None
    dense, mesh_depth_units, _rgb, meta, _qa_mod = dense_mod.render_memory_dense_condition(
        args.match_dir,
        args.episode,
        args.ego_stem,
        args.frame_index,
        args.width,
        args.height,
        args.fov_x,
        args.near,
        args.far,
        args.pitch_sign,
        args.max_triangles,
        cache=cache,
        episode_memory=episode_memory,
        player_radius=args.player_radius,
        player_height=args.player_height,
        occlusion_tolerance=args.occlusion_tolerance,
        min_visible_pixels=0,
        player_mask_mode=args.player_mask_mode,
        mesh_backend=args.mesh_backend,
    )

    episode_dir = args.match_dir / "train" / args.episode
    if episode_memory is None:
        ego_frame = dense_mod.load_json(episode_dir / f"{args.ego_stem}.json")[args.frame_index]
    else:
        ego_frame = dense_mod.frame_from_episode_memory(episode_memory, args.ego_stem, args.frame_index)
    cam_pos = np.asarray(ego_frame.get("camera_position") or [ego_frame["x"], ego_frame["y"], ego_frame["z"] + 64.0], dtype=np.float32)
    yaw = float(ego_frame["yaw"])
    pitch = float(ego_frame["pitch"])

    rows = []
    for stem in episode_memory["meta"]["player_stems"] if episode_memory else [p.stem for p in dense_mod.player_files(episode_dir)]:
        if stem == args.ego_stem:
            continue
        frame = dense_mod.frame_from_episode_memory(episode_memory, stem, args.frame_index) if episode_memory else dense_mod.load_json(episode_dir / f"{stem}.json")[args.frame_index]
        team_id, player_idx = dense_mod.parse_team_player(stem)
        row: dict[str, Any] = {
            "stem": stem,
            "team_id": team_id,
            "player_index": player_idx,
            "health": float(frame.get("health", 100)),
        }
        if not np.isfinite(frame["x"]) or float(frame.get("health", 100)) <= 0:
            row["reason"] = "dead_or_invalid"
            rows.append(row)
            continue
        base = np.asarray([frame["x"], frame["y"], frame["z"]], dtype=np.float32)
        samples = np.asarray([
            [base[0], base[1], base[2] + 8.0],
            [base[0], base[1], base[2] + args.player_height * 0.5],
            [base[0], base[1], base[2] + args.player_height],
        ], dtype=np.float32)
        proj, z_cam = cache["mesh_mod"].project_vertices(samples, cam_pos, yaw, pitch, args.pitch_sign, args.width, args.height, args.fov_x)
        valid = (z_cam > 1.0) & (z_cam < args.far)
        row["sample_uvz"] = [[float(proj[i, 0]), float(proj[i, 1]), float(z_cam[i])] for i in range(len(samples))]
        row["valid_projected_points"] = int(valid.sum())
        if not np.any(valid):
            row["reason"] = "behind_or_beyond_far"
            rows.append(row)
            continue
        u = float(np.mean(proj[valid, 0]))
        v_top = float(np.min(proj[valid, 1]))
        v_bot = float(np.max(proj[valid, 1]))
        z = float(np.mean(z_cam[valid]))
        pixel_radius = max(2.0, args.player_radius / max(z, 1.0) / np.tan(np.radians(args.fov_x) / 2.0) * args.width * 0.5)
        y_mid = (v_top + v_bot) * 0.5
        capsule_h = max(pixel_radius * 2.0, abs(v_bot - v_top) + pixel_radius * 2.0)
        raw_mask, _shape_meta = dense_mod.stamp_player_visibility_shape(
            u, y_mid, pixel_radius, capsule_h, args.player_mask_mode, args.height, args.width
        )
        visible_mask = raw_mask & np.isfinite(mesh_depth_units) & (z <= mesh_depth_units + args.occlusion_tolerance)
        finite_on_raw = raw_mask & np.isfinite(mesh_depth_units)
        if raw_mask.any() and finite_on_raw.any():
            env_vals = mesh_depth_units[finite_on_raw]
            row["env_depth_on_raw_min_median_max"] = [
                float(np.min(env_vals)),
                float(np.median(env_vals)),
                float(np.max(env_vals)),
            ]
        row.update({
            "center_uv": [u, y_mid],
            "z_cam": z,
            "pixel_radius": float(pixel_radius),
            "capsule_height_px": float(capsule_h),
            "raw_pixels": int(raw_mask.sum()),
            "raw_pixels_on_finite_mesh": int(finite_on_raw.sum()),
            "visible_pixels_after_depth_test": int(visible_mask.sum()),
            "would_pass_min_visible_pixels": int(visible_mask.sum()) >= args.min_visible_pixels,
            "reason": "kept" if int(visible_mask.sum()) >= args.min_visible_pixels else "below_min_visible_pixels",
        })
        rows.append(row)

    out = {
        "kind": "memory_player_visibility_diagnosis_v0",
        "sample": {
            "episode": args.episode,
            "ego_stem": args.ego_stem,
            "frame_index": args.frame_index,
        },
        "params": {
            "width": args.width,
            "height": args.height,
            "fov_x": args.fov_x,
            "player_radius": args.player_radius,
            "player_height": args.player_height,
            "occlusion_tolerance": args.occlusion_tolerance,
            "min_visible_pixels": args.min_visible_pixels,
            "mesh_backend": args.mesh_backend,
        },
        "mesh_hit_ratio": float(np.isfinite(mesh_depth_units).mean()),
        "kept_players_in_normal_render": meta.get("memory_projected_players", []),
        "rows": rows,
        "dense_other_player_pixels_with_min_zero": int((dense[3] > 0.5).sum()),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out_json": str(args.out_json),
        "mesh_hit_ratio": out["mesh_hit_ratio"],
        "rows": [
            {
                "stem": r["stem"],
                "reason": r["reason"],
                "raw": r.get("raw_pixels"),
                "visible": r.get("visible_pixels_after_depth_test"),
                "center_uv": r.get("center_uv"),
                "z_cam": r.get("z_cam"),
            }
            for r in rows
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
