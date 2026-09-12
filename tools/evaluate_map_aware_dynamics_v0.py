#!/usr/bin/env python3
"""Evaluate map-aware inference Dynamics.

This is the first transition loop where Map Memory is part of the dynamics:
state_{t+1} = transition(state_t, action_t, map_memory). It uses navmesh area
connectivity and ground height before accepting each movement step. Learning is
not used here; this is a map-aware rule baseline to replace action-only rollout.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NavArea:
    area_id: int
    x0: float
    x1: float
    y0: float
    y1: float
    z_nw: float
    z_ne: float
    z_se: float
    z_sw: float


class NavmeshIndex:
    def __init__(self, areas: dict[int, NavArea], neighbors: dict[int, set[int]]) -> None:
        self.areas = areas
        self.neighbors = neighbors

    @classmethod
    def from_json(cls, path: Path) -> "NavmeshIndex":
        navmesh = load_json(path)
        areas: dict[int, NavArea] = {}
        neighbors: dict[int, set[int]] = {}
        for raw_id, area in navmesh["areas"].items():
            area_id = int(area.get("area_id", raw_id))
            nw = area["nw_corner"]
            se = area["se_corner"]
            x0, x1 = sorted((float(nw[0]), float(se[0])))
            y0, y1 = sorted((float(nw[1]), float(se[1])))
            areas[area_id] = NavArea(
                area_id=area_id,
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                z_nw=float(nw[2]),
                z_ne=float(area.get("ne_z", nw[2])),
                z_se=float(se[2]),
                z_sw=float(area.get("sw_z", se[2])),
            )
            neighbors.setdefault(area_id, set()).add(area_id)
            for linked in area.get("connections", {}).values():
                for linked_id in linked:
                    linked_id = int(linked_id)
                    neighbors.setdefault(area_id, set()).add(linked_id)
                    neighbors.setdefault(linked_id, set()).add(area_id)
        return cls(areas, neighbors)

    def area_distance(self, area_id: int, xy: np.ndarray) -> float:
        area = self.areas[area_id]
        x = float(xy[0])
        y = float(xy[1])
        dx = 0.0 if area.x0 <= x <= area.x1 else min(abs(x - area.x0), abs(x - area.x1))
        dy = 0.0 if area.y0 <= y <= area.y1 else min(abs(y - area.y0), abs(y - area.y1))
        return math.hypot(dx, dy)

    def nearest_area_id(self, xy: np.ndarray) -> int:
        return min(self.areas, key=lambda area_id: self.area_distance(area_id, xy))

    def candidate_area_ids(self, area_id: int) -> set[int]:
        return {a for a in self.neighbors.get(area_id, {area_id}) if a in self.areas}

    def locate_area(self, xy: np.ndarray, previous_area_id: int | None, tolerance: float) -> int:
        candidates = self.candidate_area_ids(previous_area_id) if previous_area_id is not None else set()
        candidates.add(previous_area_id) if previous_area_id is not None else None
        candidates = {a for a in candidates if a in self.areas}
        if candidates:
            best = min(candidates, key=lambda area_id: self.area_distance(area_id, xy))
            if self.area_distance(best, xy) <= tolerance:
                return best
        return self.nearest_area_id(xy)

    def project_to_reachable(self, xy: np.ndarray, area_id: int, tolerance: float) -> tuple[np.ndarray, int, bool]:
        candidates = self.candidate_area_ids(area_id)
        best_id = min(candidates, key=lambda a: self.area_distance(a, xy))
        best_dist = self.area_distance(best_id, xy)
        if best_dist <= tolerance:
            return xy, best_id, False
        area = self.areas[best_id]
        projected = np.asarray([
            min(max(float(xy[0]), area.x0), area.x1),
            min(max(float(xy[1]), area.y0), area.y1),
        ], dtype=np.float64)
        return projected, best_id, True

    def ground_z(self, xy: np.ndarray, area_id: int) -> float:
        area = self.areas[area_id]
        tx = 0.0 if area.x1 == area.x0 else (float(xy[0]) - area.x0) / (area.x1 - area.x0)
        ty = 0.0 if area.y1 == area.y0 else (float(xy[1]) - area.y0) / (area.y1 - area.y0)
        tx = min(max(tx, 0.0), 1.0)
        ty = min(max(ty, 0.0), 1.0)
        z_n = area.z_nw * (1.0 - tx) + area.z_ne * tx
        z_s = area.z_sw * (1.0 - tx) + area.z_se * tx
        return float(z_n * (1.0 - ty) + z_s * ty)

    def path_off_nav_fraction(self, xy: np.ndarray, tolerance: float) -> float:
        off = [self.area_distance(self.nearest_area_id(p), p) > tolerance for p in xy]
        return float(np.mean(off))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def angle_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + 180.0) % 360.0 - 180.0


def yaw_basis(yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    yaw = math.radians(float(yaw_deg))
    return (
        np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64),
        np.asarray([math.cos(yaw + math.pi / 2), math.sin(yaw + math.pi / 2)], dtype=np.float64),
    )


def wish_direction(action: np.ndarray, yaw_deg: float) -> np.ndarray:
    fwd, right = yaw_basis(yaw_deg)
    forward_axis = float(action[0]) - float(action[1])
    strafe_axis = float(action[3]) - float(action[2])
    wish = fwd * forward_axis + right * strafe_axis
    norm = float(np.linalg.norm(wish))
    return wish / norm if norm > 1e-6 else np.zeros(2, dtype=np.float64)


def fit_look_scales(mem: Any, player_indices: list[int], start: int, stop: int, look_lag: int) -> dict[str, float]:
    yaw = mem["yaw"]
    pitch = mem["pitch"]
    look = mem["look_delta"]
    alive = mem["alive"]
    xs_yaw, ys_yaw, xs_pitch, ys_pitch = [], [], [], []
    for pidx in player_indices:
        hi = min(stop, yaw.shape[1] - 1)
        for t in range(max(start, 0), hi):
            if not alive[pidx, t : t + 2].all():
                continue
            look_t = t + look_lag
            if look_t < 0 or look_t >= look.shape[1]:
                continue
            dx = float(look[pidx, look_t, 0])
            dy = float(look[pidx, look_t, 1])
            yaw_delta = float(angle_diff_deg(np.asarray(yaw[pidx, t + 1]), np.asarray(yaw[pidx, t])))
            pitch_delta = float(pitch[pidx, t + 1] - pitch[pidx, t])
            if abs(dx) > 1e-5 and abs(yaw_delta) < 90.0:
                xs_yaw.append(dx); ys_yaw.append(yaw_delta)
            if abs(dy) > 1e-5 and abs(pitch_delta) < 90.0:
                xs_pitch.append(dy); ys_pitch.append(pitch_delta)

    def slope(xs: list[float], ys: list[float], fallback: float) -> float:
        x = np.asarray(xs, dtype=np.float64)
        y = np.asarray(ys, dtype=np.float64)
        if len(x) < 8 or float(x @ x) <= 1e-9:
            return fallback
        return float((x @ y) / (x @ x))

    return {
        "yaw_scale": slope(xs_yaw, ys_yaw, 1.0),
        "pitch_scale": slope(xs_pitch, ys_pitch, 1.0),
        "yaw_fit_samples": len(xs_yaw),
        "pitch_fit_samples": len(xs_pitch),
    }


def target_speed(action: np.ndarray, base_speed: float, walk_scale: float, crouch_scale: float) -> float:
    speed = base_speed
    if action[6]:
        speed *= walk_scale
    if action[5]:
        speed *= crouch_scale
    if action[10]:
        speed *= 0.75
    return speed


def action_condition(action: np.ndarray) -> str:
    if not action[:4].any():
        return "no_move"
    if action[5]:
        return "crouch"
    if action[6]:
        return "walk"
    if action[10]:
        return "scope"
    return "normal"


def motion_value(params: dict[str, Any] | None, group: str, key: str, fallback: float) -> float:
    if not params:
        return fallback
    value = params.get(group, {}).get(key)
    return float(value) if value is not None else fallback


def step_map_aware(
    xy: np.ndarray,
    vel: np.ndarray,
    area_id: int,
    action: np.ndarray,
    yaw: float,
    navmesh: NavmeshIndex,
    dt: float,
    base_speed: float,
    accel: float,
    friction: float,
    walk_scale: float,
    crouch_scale: float,
    nav_tolerance: float,
    motion_params: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    wish = wish_direction(action, yaw)
    cond = action_condition(action)
    if float(np.linalg.norm(wish)) > 0.0:
        speed = motion_value(motion_params, "speed_by_condition", cond, target_speed(action, base_speed, walk_scale, crouch_scale))
        cond_accel = motion_value(motion_params, "accel_by_condition", cond, accel)
        desired = wish * speed
        max_delta = cond_accel * dt
        delta = desired - vel
        delta_norm = float(np.linalg.norm(delta))
        vel = vel + delta * min(1.0, max_delta / max(delta_norm, 1e-6))
    else:
        cond_friction = float(motion_params.get("friction", friction)) if motion_params else friction
        vel = vel * max(0.0, 1.0 - cond_friction * dt)

    proposal = xy + vel * dt
    proposal, next_area_id, clipped = navmesh.project_to_reachable(proposal, area_id, nav_tolerance)
    if clipped:
        vel = (proposal - xy) / dt
    return proposal, vel, next_area_id, clipped


def rollout(
    mem: Any,
    pidx: int,
    start: int,
    horizon: int,
        navmesh: NavmeshIndex,
        look_scales: dict[str, float],
        motion_params: dict[str, Any] | None,
        args: argparse.Namespace,
) -> dict[str, np.ndarray | int]:
    pos = mem["position"]
    yaw_gt = mem["yaw"]
    pitch_gt = mem["pitch"]
    actions = mem["actions"]
    look_delta = mem["look_delta"]
    dt = 1.0 / args.fps
    xy = pos[pidx, start, :2].astype(np.float64).copy()
    vel = (pos[pidx, start, :2] - pos[pidx, start - 1, :2]).astype(np.float64) / dt
    area_id = navmesh.locate_area(xy, None, args.nav_tolerance)
    yaw = float(yaw_gt[pidx, start])
    pitch = float(pitch_gt[pidx, start])
    pred_xy = [xy.copy()]
    pred_z = [navmesh.ground_z(xy, area_id)]
    pred_yaw = [yaw]
    pred_pitch = [pitch]
    clipped_steps = 0
    for k in range(horizon):
        t = start + k
        look_t = min(max(t + args.look_lag, 0), look_delta.shape[1] - 1)
        yaw = float((yaw + look_delta[pidx, look_t, 0] * look_scales["yaw_scale"] + 180.0) % 360.0 - 180.0)
        pitch = float(np.clip(pitch + look_delta[pidx, look_t, 1] * look_scales["pitch_scale"], -89.0, 89.0))
        area_id = navmesh.locate_area(xy, area_id, args.nav_tolerance)
        xy, vel, area_id, clipped = step_map_aware(
            xy, vel, area_id, actions[pidx, t], yaw, navmesh, dt,
            args.base_speed, args.accel, args.friction, args.walk_scale,
            args.crouch_scale, args.nav_tolerance, motion_params,
        )
        clipped_steps += int(clipped)
        pred_xy.append(xy.copy())
        pred_z.append(navmesh.ground_z(xy, area_id))
        pred_yaw.append(yaw)
        pred_pitch.append(pitch)
    return {
        "xy": np.asarray(pred_xy, dtype=np.float64),
        "z": np.asarray(pred_z, dtype=np.float64),
        "yaw": np.asarray(pred_yaw, dtype=np.float64),
        "pitch": np.asarray(pred_pitch, dtype=np.float64),
        "clipped_steps": clipped_steps,
    }


def evaluate(
    mem: Any,
    navmesh: NavmeshIndex,
    look_scales: dict[str, float],
    motion_params: dict[str, Any] | None,
    eval_players: list[int],
    starts: list[int],
    horizon: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    pos = mem["position"]
    yaw = mem["yaw"]
    pitch = mem["pitch"]
    alive = mem["alive"]
    track_length = mem["track_length"]
    rows = []
    for pidx in eval_players:
        for start in starts:
            if start < 1 or start + horizon >= int(track_length[pidx]):
                continue
            if not alive[pidx, start - 1 : start + horizon + 1].all():
                continue
            gt_xy = pos[pidx, start : start + horizon + 1, :2].astype(np.float64)
            gt_disp = float(np.linalg.norm(gt_xy[-1] - gt_xy[0]))
            if gt_disp < args.min_gt_displacement:
                continue
            pred = rollout(mem, pidx, start, horizon, navmesh, look_scales, motion_params, args)
            xy_err = np.linalg.norm(pred["xy"] - gt_xy, axis=1)
            gt_yaw = yaw[pidx, start : start + horizon + 1].astype(np.float64)
            gt_pitch = pitch[pidx, start : start + horizon + 1].astype(np.float64)
            yaw_err = np.abs(angle_diff_deg(pred["yaw"], gt_yaw))
            pitch_err = np.abs(pred["pitch"] - gt_pitch)
            rows.append({
                "player_idx": int(pidx),
                "start": int(start),
                "final_error_xy": float(xy_err[-1]),
                "mean_error_xy": float(xy_err.mean()),
                "final_yaw_error": float(yaw_err[-1]),
                "mean_yaw_error": float(yaw_err.mean()),
                "final_pitch_error": float(pitch_err[-1]),
                "mean_pitch_error": float(pitch_err.mean()),
                "gt_displacement": gt_disp,
                "pred_displacement": float(np.linalg.norm(pred["xy"][-1] - pred["xy"][0])),
                "off_nav_fraction": navmesh.path_off_nav_fraction(pred["xy"], args.nav_tolerance),
                "clipped_fraction": float(pred["clipped_steps"]) / max(1, horizon),
            })

    def stat(key: str, fn: Any = np.mean) -> float | None:
        vals = [r[key] for r in rows]
        return float(fn(vals)) if vals else None

    return {
        "sample_count": len(rows),
        "mean_final_error_xy": stat("final_error_xy"),
        "median_final_error_xy": stat("final_error_xy", np.median),
        "p90_final_error_xy": stat("final_error_xy", lambda x: np.percentile(x, 90)),
        "mean_error_xy": stat("mean_error_xy"),
        "mean_final_yaw_error": stat("final_yaw_error"),
        "mean_yaw_error": stat("mean_yaw_error"),
        "mean_final_pitch_error": stat("final_pitch_error"),
        "mean_pitch_error": stat("mean_pitch_error"),
        "mean_off_nav_fraction": stat("off_nav_fraction"),
        "mean_clipped_fraction": stat("clipped_fraction"),
        "rows": rows[:120],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-memory-dir", type=Path, required=True)
    ap.add_argument("--navmesh-path", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--train-player-limit", type=int, default=6)
    ap.add_argument("--train-start", type=int, default=0)
    ap.add_argument("--train-stop", type=int, default=1200)
    ap.add_argument("--eval-player-start", type=int, default=6)
    ap.add_argument("--eval-player-limit", type=int, default=4)
    ap.add_argument("--eval-start", type=int, default=20)
    ap.add_argument("--eval-stop", type=int, default=1200)
    ap.add_argument("--stride", type=int, default=120)
    ap.add_argument("--horizons", default="32,64,128,256")
    ap.add_argument("--fps", type=float, default=16.0)
    ap.add_argument("--look-lag", type=int, default=1)
    ap.add_argument("--nav-tolerance", type=float, default=24.0)
    ap.add_argument("--min-gt-displacement", type=float, default=50.0)
    ap.add_argument("--base-speed", type=float, default=105.0)
    ap.add_argument("--accel", type=float, default=420.0)
    ap.add_argument("--friction", type=float, default=7.5)
    ap.add_argument("--walk-scale", type=float, default=0.52)
    ap.add_argument("--crouch-scale", type=float, default=0.45)
    ap.add_argument("--motion-params", type=Path)
    args = ap.parse_args()

    meta = load_json(args.episode_memory_dir / "episode_memory_meta_v0.json")
    mem = np.load(args.episode_memory_dir / "episode_memory_v0.npz")
    navmesh = NavmeshIndex.from_json(args.navmesh_path)
    motion_params = load_json(args.motion_params) if args.motion_params else None
    player_count = len(meta["player_stems"])
    train_players = list(range(min(args.train_player_limit, player_count)))
    eval_players = list(range(args.eval_player_start, min(args.eval_player_start + args.eval_player_limit, player_count)))
    starts = list(range(args.eval_start, args.eval_stop, args.stride))
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    look_scales = fit_look_scales(mem, train_players, args.train_start, args.train_stop, args.look_lag)

    results = {
        str(h): evaluate(mem, navmesh, look_scales, motion_params, eval_players, starts, h, args)
        for h in horizons
    }
    out = {
        "kind": "map_aware_inference_dynamics_eval_v0",
        "policy": "Closed-loop inference Dynamics with navmesh area connectivity and ground-height queries inside the transition loop.",
        "episode_memory_dir": str(args.episode_memory_dir),
        "navmesh_path": str(args.navmesh_path),
        "train_players": train_players,
        "eval_players": eval_players,
        "starts": starts,
        "horizons": horizons,
        "params": {
            "fps": args.fps,
            "look_lag": args.look_lag,
            "nav_tolerance": args.nav_tolerance,
            "min_gt_displacement": args.min_gt_displacement,
            "base_speed": args.base_speed,
            "accel": args.accel,
            "friction": args.friction,
            "walk_scale": args.walk_scale,
            "crouch_scale": args.crouch_scale,
            "motion_params": str(args.motion_params) if args.motion_params else None,
        },
        "look_scales": look_scales,
        "results": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "map_aware_dynamics_eval_v0.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(out_path),
        "params": out["params"],
        "horizons": {
            h: {
                "final_xy": results[str(h)]["mean_final_error_xy"],
                "yaw": results[str(h)]["mean_final_yaw_error"],
                "off_nav": results[str(h)]["mean_off_nav_fraction"],
                "clipped": results[str(h)]["mean_clipped_fraction"],
                "samples": results[str(h)]["sample_count"],
            }
            for h in horizons
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
