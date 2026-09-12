#!/usr/bin/env python3
"""Fit simple state-dependent motion parameters from replay Episode Memory.

The output is a JSON parameter file consumed by map-aware Dynamics. It is still
a compact rule model, but it replaces one global speed constant with statistics
conditioned on walk/crouch/scope and action state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def condition_name(action: np.ndarray) -> str:
    if not action[:4].any():
        return "no_move"
    if action[5]:
        return "crouch"
    if action[6]:
        return "walk"
    if action[10]:
        return "scope"
    return "normal"


def robust_percentile(values: list[float], q: float, fallback: float) -> float:
    if not values:
        return fallback
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-memory-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--player-limit", type=int, default=6)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=1200)
    ap.add_argument("--fps", type=float, default=16.0)
    ap.add_argument("--action-lag", type=int, default=0)
    ap.add_argument("--speed-percentile", type=float, default=60.0)
    ap.add_argument("--accel-percentile", type=float, default=70.0)
    ap.add_argument("--friction-percentile", type=float, default=60.0)
    args = ap.parse_args()

    meta = load_json(args.episode_memory_dir / "episode_memory_meta_v0.json")
    mem = np.load(args.episode_memory_dir / "episode_memory_v0.npz")
    pos = mem["position"]
    actions = mem["actions"]
    alive = mem["alive"]
    track_length = mem["track_length"]
    dt = 1.0 / args.fps

    speeds: dict[str, list[float]] = {k: [] for k in ["normal", "walk", "crouch", "scope", "no_move"]}
    accel_mags: dict[str, list[float]] = {k: [] for k in ["normal", "walk", "crouch", "scope"]}
    stop_decay: list[float] = []
    rows = []

    for pidx in range(min(args.player_limit, pos.shape[0])):
        hi = min(int(track_length[pidx]) - 2, args.stop)
        for t in range(max(args.start + 1, 1), hi):
            if not alive[pidx, t - 1 : t + 2].all():
                continue
            vel_prev = (pos[pidx, t, :2] - pos[pidx, t - 1, :2]) / dt
            vel_next = (pos[pidx, t + 1, :2] - pos[pidx, t, :2]) / dt
            speed_next = float(np.linalg.norm(vel_next))
            action_t = min(max(t + args.action_lag, 0), actions.shape[1] - 1)
            cond = condition_name(actions[pidx, action_t])
            speeds[cond].append(speed_next)
            if cond != "no_move":
                accel_mags[cond].append(float(np.linalg.norm(vel_next - vel_prev) / dt))
            elif np.linalg.norm(vel_prev) > 10.0:
                decay = float(np.linalg.norm(vel_next) / max(np.linalg.norm(vel_prev), 1e-6))
                if 0.0 <= decay < 1.0:
                    stop_decay.append(decay)
            rows.append({
                "player_idx": int(pidx),
                "frame": int(t),
                "condition": cond,
                "speed": speed_next,
            })

    normal_speed = robust_percentile(speeds["normal"], args.speed_percentile, 110.0)
    params = {
        "kind": "motion_params_v0",
        "episode_memory_dir": str(args.episode_memory_dir),
        "fit_players": list(range(min(args.player_limit, pos.shape[0]))),
        "fit_frame_range": [args.start, args.stop],
        "fps": args.fps,
        "action_lag": args.action_lag,
        "percentiles": {
            "speed": args.speed_percentile,
            "accel": args.accel_percentile,
            "friction": args.friction_percentile,
        },
        "speed_by_condition": {
            "normal": normal_speed,
            "walk": robust_percentile(speeds["walk"], args.speed_percentile, normal_speed * 0.52),
            "crouch": robust_percentile(speeds["crouch"], args.speed_percentile, normal_speed * 0.45),
            "scope": robust_percentile(speeds["scope"], args.speed_percentile, normal_speed * 0.75),
            "no_move": 0.0,
        },
        "accel_by_condition": {
            key: robust_percentile(vals, args.accel_percentile, 420.0)
            for key, vals in accel_mags.items()
        },
        "friction": None,
        "counts": {
            "speed": {k: len(v) for k, v in speeds.items()},
            "accel": {k: len(v) for k, v in accel_mags.items()},
            "stop_decay": len(stop_decay),
        },
        "speed_summary": {
            k: {
                "mean": float(np.mean(v)) if v else None,
                "p50": robust_percentile(v, 50.0, 0.0),
                "p75": robust_percentile(v, 75.0, 0.0),
                "p90": robust_percentile(v, 90.0, 0.0),
            }
            for k, v in speeds.items()
        },
    }
    if stop_decay:
        decay = robust_percentile(stop_decay, args.friction_percentile, 0.5)
        params["friction"] = float(max(1.0, min(20.0, (1.0 - decay) / dt)))
    else:
        params["friction"] = 7.5

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "motion_params_v0.json"
    out_path.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(out_path),
        "speed_by_condition": params["speed_by_condition"],
        "accel_by_condition": params["accel_by_condition"],
        "friction": params["friction"],
        "counts": params["counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
