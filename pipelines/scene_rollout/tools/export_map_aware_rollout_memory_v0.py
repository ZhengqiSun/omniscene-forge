#!/usr/bin/env python3
"""Export simulator-mode Episode Memory from map-aware Dynamics.

This mirrors evaluate_map_aware_dynamics_v0.py but writes the rolled-out state
back to the standard episode_memory_v0.npz schema consumed by the renderer.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def import_eval_module() -> Any:
    path = Path(__file__).with_name("evaluate_map_aware_dynamics_v0.py")
    spec = importlib.util.spec_from_file_location("evaluate_map_aware_dynamics_v0", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    mod = import_eval_module()
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-memory-dir", type=Path, required=True)
    ap.add_argument("--navmesh-path", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--motion-params", type=Path)
    ap.add_argument("--bsp-faces-npz", type=Path)
    ap.add_argument("--bsp-cell-size", type=float, default=128.0)
    ap.add_argument("--max-ground-slope-deg", type=float, default=55.0)
    ap.add_argument("--max-step-height", type=float, default=72.0)
    ap.add_argument("--player-radius", type=float, default=18.0)
    ap.add_argument("--player-height", type=float, default=72.0)
    ap.add_argument("--enable-bsp-collision", action="store_true")
    ap.add_argument("--initial-velocity", choices=["zero", "replay"], default="zero")
    ap.add_argument("--player-indices", default="")
    ap.add_argument("--start", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=128)
    ap.add_argument("--fps", type=float, default=16.0)
    ap.add_argument("--look-lag", type=int, default=1)
    ap.add_argument("--action-lag", type=int, default=0)
    ap.add_argument("--nav-tolerance", type=float, default=24.0)
    ap.add_argument("--base-speed", type=float, default=105.0)
    ap.add_argument("--accel", type=float, default=420.0)
    ap.add_argument("--friction", type=float, default=7.5)
    ap.add_argument("--walk-scale", type=float, default=0.52)
    ap.add_argument("--crouch-scale", type=float, default=0.45)
    ap.add_argument("--velocity-model", choices=["rule", "linear"], default="rule")
    ap.add_argument("--velocity-alpha", type=float, default=10.0)
    ap.add_argument("--velocity-min-speed", type=float, default=2.0)
    ap.add_argument("--max-speed", type=float, default=320.0)
    ap.add_argument("--max-substep-distance", type=float, default=8.0)
    args = ap.parse_args()

    src_meta = load_json(args.episode_memory_dir / "episode_memory_meta_v0.json")
    src = np.load(args.episode_memory_dir / "episode_memory_v0.npz")
    navmesh = mod.NavmeshIndex.from_json(args.navmesh_path)
    bsp_ground = mod.BspGroundIndex.from_npz(args.bsp_faces_npz, args.bsp_cell_size, args.max_ground_slope_deg) if args.bsp_faces_npz else None
    motion_params = load_json(args.motion_params) if args.motion_params else None
    player_count = len(src_meta["player_stems"])
    if args.player_indices:
        rollout_players = [int(x) for x in args.player_indices.split(",") if x.strip()]
    else:
        rollout_players = list(range(player_count))

    train_players = list(range(min(6, player_count)))
    look_scales = mod.fit_look_scales(src, train_players, 0, 1200, args.look_lag)
    velocity_weights = None
    velocity_fit = None
    if args.velocity_model == "linear":
        x_vel, y_vel = mod.build_velocity_samples(
            src, train_players, 0, 1200, args.fps, args.action_lag,
            args.velocity_min_speed,
        )
        if len(x_vel) == 0:
            raise ValueError("No velocity samples available for linear velocity model")
        velocity_weights = mod.fit_ridge(x_vel, y_vel, args.velocity_alpha)
        pred_vel = x_vel @ velocity_weights
        err = np.linalg.norm(pred_vel - y_vel, axis=1)
        velocity_fit = {
            "feature_count": int(x_vel.shape[1]),
            "train_samples": int(len(x_vel)),
            "alpha": args.velocity_alpha,
            "min_speed": args.velocity_min_speed,
            "one_step_velocity_rmse": float(np.sqrt(np.mean(err ** 2))),
            "one_step_velocity_mae": float(np.mean(np.abs(err))),
        }

    out_arrays = {name: src[name].copy() for name in src.files}
    pos = out_arrays["position"]
    camera_pos = out_arrays["camera_position"]
    yaw = out_arrays["yaw"]
    pitch = out_arrays["pitch"]
    alive = out_arrays["alive"]
    track_length = out_arrays["track_length"]
    stop = min(args.start + args.horizon, int(np.max(track_length)) - 1)
    rollout_rows = []

    eval_args = argparse.Namespace(
        fps=args.fps,
        look_lag=args.look_lag,
        action_lag=args.action_lag,
        nav_tolerance=args.nav_tolerance,
        base_speed=args.base_speed,
        accel=args.accel,
        friction=args.friction,
        walk_scale=args.walk_scale,
        crouch_scale=args.crouch_scale,
        max_step_height=args.max_step_height,
        player_radius=args.player_radius,
        player_height=args.player_height,
        enable_bsp_collision=args.enable_bsp_collision,
        initial_velocity=args.initial_velocity,
        velocity_model=args.velocity_model,
        velocity_alpha=args.velocity_alpha,
        velocity_min_speed=args.velocity_min_speed,
        max_speed=args.max_speed,
        max_substep_distance=args.max_substep_distance,
    )

    for pidx in rollout_players:
        if pidx < 0 or pidx >= player_count:
            raise ValueError(f"Invalid player index {pidx}")
        if stop >= int(track_length[pidx]):
            continue
        if args.start < 1 or not alive[pidx, args.start - 1 : stop + 1].all():
            continue
        pred = mod.rollout(src, pidx, args.start, stop - args.start, navmesh, bsp_ground, look_scales, motion_params, velocity_weights, eval_args)
        z_offset = float(camera_pos[pidx, args.start, 2] - pos[pidx, args.start, 2])
        for i, t in enumerate(range(args.start, stop + 1)):
            pos[pidx, t, 0:2] = pred["xy"][i].astype(np.float32)
            pos[pidx, t, 2] = float(pred["z"][i])
            camera_pos[pidx, t, 0:2] = pred["xy"][i].astype(np.float32)
            camera_pos[pidx, t, 2] = float(pred["z"][i] + z_offset)
            yaw[pidx, t] = float(pred["yaw"][i])
            pitch[pidx, t] = float(pred["pitch"][i])
        gt_xy = src["position"][pidx, args.start : stop + 1, :2].astype(np.float64)
        err = np.linalg.norm(pred["xy"] - gt_xy, axis=1)
        rollout_rows.append({
            "player_idx": int(pidx),
            "player_stem": src_meta["player_stems"][pidx],
            "start": args.start,
            "stop": stop,
            "final_error_xy": float(err[-1]),
            "mean_error_xy": float(err.mean()),
            "clipped_steps": int(pred["clipped_steps"]),
            "collision_clipped_steps": int(pred["collision_clipped_steps"]),
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.out_dir / "episode_memory_v0.npz"
    np.savez_compressed(npz_path, **out_arrays)
    meta = dict(src_meta)
    meta.update({
        "kind": "map_aware_simulator_episode_memory_v0",
        "source_episode_memory_dir": str(args.episode_memory_dir),
        "npz_path": str(npz_path),
        "rollout_policy": {
            "method": "map_aware_rule",
            "velocity_fit": velocity_fit,
            "navmesh_path": str(args.navmesh_path),
            "motion_params": str(args.motion_params) if args.motion_params else None,
            "bsp_faces_npz": str(args.bsp_faces_npz) if args.bsp_faces_npz else None,
            "start": args.start,
            "stop": stop,
            "horizon": stop - args.start,
            "rollout_players": rollout_players,
            "params": {
                "fps": args.fps,
                "look_lag": args.look_lag,
                "action_lag": args.action_lag,
                "nav_tolerance": args.nav_tolerance,
                "base_speed": args.base_speed,
                "accel": args.accel,
                "friction": args.friction,
                "walk_scale": args.walk_scale,
                "crouch_scale": args.crouch_scale,
                "velocity_model": args.velocity_model,
                "velocity_alpha": args.velocity_alpha,
                "velocity_min_speed": args.velocity_min_speed,
                "max_speed": args.max_speed,
                "max_substep_distance": args.max_substep_distance,
                "bsp_cell_size": args.bsp_cell_size,
                "max_ground_slope_deg": args.max_ground_slope_deg,
                "max_step_height": args.max_step_height,
                "player_radius": args.player_radius,
                "player_height": args.player_height,
                "enable_bsp_collision": args.enable_bsp_collision,
                "initial_velocity": args.initial_velocity,
            },
            "look_scales": look_scales,
        },
        "rollout_summary": {
            "player_count": len(rollout_rows),
            "mean_final_error_xy": float(np.mean([r["final_error_xy"] for r in rollout_rows])) if rollout_rows else None,
            "mean_error_xy": float(np.mean([r["mean_error_xy"] for r in rollout_rows])) if rollout_rows else None,
            "rows": rollout_rows,
        },
    })
    meta_path = args.out_dir / "episode_memory_meta_v0.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "npz": str(npz_path),
        "meta": str(meta_path),
        "rollout_summary": meta["rollout_summary"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
