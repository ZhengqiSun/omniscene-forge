#!/usr/bin/env python3
"""Evaluate map-aware inference Dynamics.

This is the first transition loop where Map Memory is part of the dynamics:
state_{t+1} = transition(state_t, action_t, map_memory). It uses navmesh area
connectivity and ground height before accepting each movement step. Learning is
not used here; this is a map-aware rule baseline to replace action-only rollout.

v1 fixes the Source-engine right-vector convention used by movement actions.
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


class BspGroundIndex:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, cell_size: float, max_slope_deg: float) -> None:
        self.vertices = vertices.astype(np.float64)
        self.faces = faces.astype(np.int64)
        self.cell_size = float(cell_size)
        self.max_slope_cos = math.cos(math.radians(max_slope_deg))
        tri = self.vertices[self.faces]
        self.tri = tri
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        norm = np.linalg.norm(normals, axis=1)
        valid = norm > 1e-8
        normals[valid] = normals[valid] / norm[valid, None]
        flip = normals[:, 2] < 0
        normals[flip] *= -1.0
        self.normals = normals
        self.walkable = valid & (normals[:, 2] >= self.max_slope_cos)
        normal_xy_norm = np.linalg.norm(normals[:, :2], axis=1)
        self.blocking = valid & (normal_xy_norm >= 0.25) & (normals[:, 2] < self.max_slope_cos)
        self.xy_min = tri[:, :, :2].min(axis=(0, 1))
        self.grid: dict[tuple[int, int], list[int]] = {}
        for idx in np.where(self.walkable)[0]:
            mn = tri[idx, :, :2].min(axis=0)
            mx = tri[idx, :, :2].max(axis=0)
            c0 = np.floor((mn - self.xy_min) / self.cell_size).astype(int)
            c1 = np.floor((mx - self.xy_min) / self.cell_size).astype(int)
            for cx in range(int(c0[0]), int(c1[0]) + 1):
                for cy in range(int(c0[1]), int(c1[1]) + 1):
                    self.grid.setdefault((cx, cy), []).append(int(idx))
        self.blocker_grid: dict[tuple[int, int], list[int]] = {}
        for idx in np.where(self.blocking)[0]:
            mn = tri[idx, :, :2].min(axis=0)
            mx = tri[idx, :, :2].max(axis=0)
            c0 = np.floor((mn - self.xy_min) / self.cell_size).astype(int)
            c1 = np.floor((mx - self.xy_min) / self.cell_size).astype(int)
            for cx in range(int(c0[0]), int(c1[0]) + 1):
                for cy in range(int(c0[1]), int(c1[1]) + 1):
                    self.blocker_grid.setdefault((cx, cy), []).append(int(idx))

    @classmethod
    def from_npz(cls, path: Path, cell_size: float, max_slope_deg: float) -> "BspGroundIndex":
        data = np.load(path)
        return cls(data["vertices"], data["faces"], cell_size, max_slope_deg)

    def cell(self, xy: np.ndarray) -> tuple[int, int]:
        c = np.floor((xy.astype(np.float64) - self.xy_min) / self.cell_size).astype(int)
        return int(c[0]), int(c[1])

    def cells_for_bbox(self, mn: np.ndarray, mx: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
        c0 = np.floor((mn.astype(np.float64) - self.xy_min) / self.cell_size).astype(int)
        c1 = np.floor((mx.astype(np.float64) - self.xy_min) / self.cell_size).astype(int)
        return (int(c0[0]), int(c0[1])), (int(c1[0]), int(c1[1]))

    @staticmethod
    def barycentric_xy(p: np.ndarray, tri_xy: np.ndarray) -> tuple[float, float, float] | None:
        a, b, c = tri_xy
        v0 = b - a
        v1 = c - a
        v2 = p - a
        den = float(v0[0] * v1[1] - v1[0] * v0[1])
        if abs(den) < 1e-8:
            return None
        u = float((v2[0] * v1[1] - v1[0] * v2[1]) / den)
        v = float((v0[0] * v2[1] - v2[0] * v0[1]) / den)
        w = 1.0 - u - v
        eps = -1e-5
        if u >= eps and v >= eps and w >= eps:
            return w, u, v
        return None

    def query(self, xy: np.ndarray, ref_z: float | None = None, search_radius_cells: int = 1) -> dict[str, float] | None:
        base = self.cell(xy)
        candidates: list[int] = []
        for dx in range(-search_radius_cells, search_radius_cells + 1):
            for dy in range(-search_radius_cells, search_radius_cells + 1):
                candidates.extend(self.grid.get((base[0] + dx, base[1] + dy), []))
        best = None
        best_score = float("inf")
        p = xy.astype(np.float64)
        for idx in candidates:
            tri = self.tri[idx]
            bc = self.barycentric_xy(p, tri[:, :2])
            if bc is None:
                continue
            z = float(bc[0] * tri[0, 2] + bc[1] * tri[1, 2] + bc[2] * tri[2, 2])
            score = abs(z - ref_z) if ref_z is not None else -z
            if score < best_score:
                best_score = score
                best = {
                    "z": z,
                    "normal_z": float(self.normals[idx, 2]),
                    "face_index": int(idx),
                }
        return best

    @staticmethod
    def point_segment_distance_xy(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ab = b - a
        den = float(ab @ ab)
        if den <= 1e-8:
            return float(np.linalg.norm(p - a))
        t = float(np.clip(((p - a) @ ab) / den, 0.0, 1.0))
        return float(np.linalg.norm(p - (a + ab * t)))

    def blocker_candidates(
        self,
        start_xy: np.ndarray,
        end_xy: np.ndarray,
        radius: float,
        z_min: float,
        z_max: float,
    ) -> list[int]:
        mn = np.minimum(start_xy, end_xy) - radius
        mx = np.maximum(start_xy, end_xy) + radius
        c0, c1 = self.cells_for_bbox(mn, mx)
        candidates: set[int] = set()
        for cx in range(c0[0], c1[0] + 1):
            for cy in range(c0[1], c1[1] + 1):
                candidates.update(self.blocker_grid.get((cx, cy), []))
        out = []
        for idx in candidates:
            tri = self.tri[idx]
            if float(tri[:, 2].max()) < z_min or float(tri[:, 2].min()) > z_max:
                continue
            out.append(int(idx))
        return out

    def resolve_horizontal_collision(
        self,
        start_xy: np.ndarray,
        end_xy: np.ndarray,
        vel: np.ndarray,
        ground_z: float,
        radius: float,
        player_height: float,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        xy = end_xy.astype(np.float64).copy()
        new_vel = vel.astype(np.float64).copy()
        clipped = False
        z_min = ground_z + 8.0
        z_max = ground_z + player_height
        candidates = self.blocker_candidates(start_xy, xy, radius, z_min, z_max)
        if not candidates:
            return xy, new_vel, False
        for _ in range(3):
            adjusted = False
            for idx in candidates:
                tri = self.tri[idx]
                n = self.normals[idx, :2].astype(np.float64)
                n_norm = float(np.linalg.norm(n))
                if n_norm <= 1e-6:
                    continue
                n = n / n_norm
                if float((start_xy - tri[0, :2]) @ n) < 0.0:
                    n = -n
                signed = float((xy - tri[0, :2]) @ n)
                min_edge_dist = min(
                    self.point_segment_distance_xy(xy, tri[0, :2], tri[1, :2]),
                    self.point_segment_distance_xy(xy, tri[1, :2], tri[2, :2]),
                    self.point_segment_distance_xy(xy, tri[2, :2], tri[0, :2]),
                )
                inside_xy = self.barycentric_xy(xy, tri[:, :2]) is not None
                moving_into = float((xy - start_xy) @ n) < 0.0 or float(new_vel @ n) < 0.0
                in_face_span = inside_xy or min_edge_dist <= radius * 1.5
                if moving_into and signed < radius and in_face_span:
                    xy = xy + n * (radius - signed + 1e-3)
                    inward = float(new_vel @ n)
                    if inward < 0.0:
                        new_vel = new_vel - n * inward
                    clipped = True
                    adjusted = True
            if not adjusted:
                break
        return xy, new_vel, clipped


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def angle_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + 180.0) % 360.0 - 180.0


def yaw_basis(yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    yaw = math.radians(float(yaw_deg))
    return (
        np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64),
        np.asarray([math.cos(yaw - math.pi / 2), math.sin(yaw - math.pi / 2)], dtype=np.float64),
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


def velocity_feature_vec(vel_xy: np.ndarray, yaw_deg: float, action: np.ndarray) -> np.ndarray:
    fwd, right = yaw_basis(yaw_deg)
    forward_axis = float(action[0]) - float(action[1])
    strafe_axis = float(action[3]) - float(action[2])
    wish = fwd * forward_axis + right * strafe_axis
    speed = float(np.linalg.norm(vel_xy))
    no_move = float(not action[:4].any())
    walk = float(action[6])
    crouch = float(action[5])
    scope = float(action[10])
    jump = float(action[4])
    return np.asarray([
        1.0,
        float(vel_xy[0]), float(vel_xy[1]), speed,
        float(wish[0]), float(wish[1]),
        float(fwd[0]), float(fwd[1]), float(right[0]), float(right[1]),
        forward_axis, strafe_axis,
        no_move, walk, crouch, scope, jump,
        float(vel_xy[0]) * no_move, float(vel_xy[1]) * no_move,
        float(vel_xy[0]) * walk, float(vel_xy[1]) * walk,
        float(wish[0]) * walk, float(wish[1]) * walk,
        float(wish[0]) * crouch, float(wish[1]) * crouch,
    ], dtype=np.float64)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xtx = x.T @ x
    reg = np.eye(xtx.shape[0], dtype=np.float64) * alpha
    reg[0, 0] = 0.0
    return np.linalg.solve(xtx + reg, x.T @ y)


def build_velocity_samples(
    mem: Any,
    player_indices: list[int],
    start: int,
    stop: int,
    fps: float,
    action_lag: int,
    min_speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    pos = mem["position"]
    yaw = mem["yaw"]
    actions = mem["actions"]
    alive = mem["alive"]
    track_length = mem["track_length"]
    dt = 1.0 / fps
    xs = []
    ys = []
    for pidx in player_indices:
        hi = min(int(track_length[pidx]) - 2, stop)
        for t in range(max(start + 1, 1), hi):
            if not alive[pidx, t - 1 : t + 2].all():
                continue
            action_t = min(max(t + action_lag, 0), actions.shape[1] - 1)
            action = actions[pidx, action_t]
            vel_t = (pos[pidx, t, :2] - pos[pidx, t - 1, :2]).astype(np.float64) / dt
            vel_next = (pos[pidx, t + 1, :2] - pos[pidx, t, :2]).astype(np.float64) / dt
            if float(np.linalg.norm(vel_t)) < min_speed and float(np.linalg.norm(vel_next)) < min_speed and not action[:4].any():
                continue
            xs.append(velocity_feature_vec(vel_t, float(yaw[pidx, t]), action))
            ys.append(vel_next)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def predict_rule_velocity(
    vel: np.ndarray,
    action: np.ndarray,
    yaw: float,
    dt: float,
    base_speed: float,
    accel: float,
    friction: float,
    walk_scale: float,
    crouch_scale: float,
    motion_params: dict[str, Any] | None,
) -> np.ndarray:
    wish = wish_direction(action, yaw)
    cond = action_condition(action)
    if float(np.linalg.norm(wish)) > 0.0:
        speed = motion_value(motion_params, "speed_by_condition", cond, target_speed(action, base_speed, walk_scale, crouch_scale))
        cond_accel = motion_value(motion_params, "accel_by_condition", cond, accel)
        desired = wish * speed
        max_delta = cond_accel * dt
        delta = desired - vel
        delta_norm = float(np.linalg.norm(delta))
        return vel + delta * min(1.0, max_delta / max(delta_norm, 1e-6))
    cond_friction = float(motion_params.get("friction", friction)) if motion_params else friction
    return vel * max(0.0, 1.0 - cond_friction * dt)


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
    current_z: float,
    bsp_ground: BspGroundIndex | None,
    max_step_height: float,
    player_radius: float,
    player_height: float,
    enable_bsp_collision: bool,
    velocity_model: str,
    velocity_weights: np.ndarray | None,
    max_speed: float,
    max_substep_distance: float,
) -> tuple[np.ndarray, np.ndarray, int, bool, bool, bool, bool, bool]:
    if velocity_model == "linear":
        if velocity_weights is None:
            raise ValueError("velocity_model=linear requires fitted velocity weights")
        vel = velocity_feature_vec(vel, yaw, action) @ velocity_weights
    else:
        vel = predict_rule_velocity(
            vel, action, yaw, dt, base_speed, accel, friction,
            walk_scale, crouch_scale, motion_params,
        )
    speed = float(np.linalg.norm(vel))
    if speed > max_speed:
        vel = vel * (max_speed / max(speed, 1e-6))

    origin = xy.copy()
    total_delta = vel * dt
    step_count = max(1, int(math.ceil(float(np.linalg.norm(total_delta)) / max(max_substep_distance, 1e-6))))
    sub_delta = total_delta / step_count
    next_area_id = area_id
    clipped = False
    collision_clipped = False
    nav_clipped = False
    ground_miss_clipped = False
    step_height_clipped = False
    for _ in range(step_count):
        prev_xy = xy.copy()
        proposal = xy + sub_delta
        proposal, next_area_id, sub_nav_clipped = navmesh.project_to_reachable(proposal, next_area_id, nav_tolerance)
        nav_clipped = nav_clipped or sub_nav_clipped
        if bsp_ground is not None and enable_bsp_collision:
            proposal, vel, sub_collision_clipped = bsp_ground.resolve_horizontal_collision(
                xy, proposal, vel, current_z, player_radius, player_height
            )
            collision_clipped = collision_clipped or sub_collision_clipped
        if bsp_ground is not None:
            ground = bsp_ground.query(proposal, current_z)
            if ground is None:
                clipped = True
                ground_miss_clipped = True
                break
            if abs(float(ground["z"]) - current_z) > max_step_height:
                clipped = True
                step_height_clipped = True
                break
            current_z = float(ground["z"])
        xy = proposal
        if sub_nav_clipped or not np.allclose(xy, prev_xy + sub_delta):
            sub_delta = xy - prev_xy
    if clipped or collision_clipped or nav_clipped:
        vel = (xy - origin) / dt
    return (
        xy,
        vel,
        next_area_id,
        clipped or nav_clipped,
        collision_clipped,
        nav_clipped,
        ground_miss_clipped,
        step_height_clipped,
    )


def rollout(
    mem: Any,
    pidx: int,
    start: int,
    horizon: int,
    navmesh: NavmeshIndex,
    bsp_ground: BspGroundIndex | None,
    look_scales: dict[str, float],
    motion_params: dict[str, Any] | None,
    velocity_weights: np.ndarray | None,
    args: argparse.Namespace,
) -> dict[str, np.ndarray | int]:
    pos = mem["position"]
    yaw_gt = mem["yaw"]
    pitch_gt = mem["pitch"]
    actions = mem["actions"]
    look_delta = mem["look_delta"]
    dt = 1.0 / args.fps
    xy = pos[pidx, start, :2].astype(np.float64).copy()
    if args.initial_velocity == "replay":
        vel = (pos[pidx, start, :2] - pos[pidx, start - 1, :2]).astype(np.float64) / dt
    else:
        vel = np.zeros(2, dtype=np.float64)
    area_id = navmesh.locate_area(xy, None, args.nav_tolerance)
    yaw = float(yaw_gt[pidx, start])
    pitch = float(pitch_gt[pidx, start])
    pred_xy = [xy.copy()]
    initial_nav_z = navmesh.ground_z(xy, area_id)
    initial_bsp = bsp_ground.query(xy, float(pos[pidx, start, 2])) if bsp_ground is not None else None
    current_z = float(initial_bsp["z"]) if initial_bsp is not None else initial_nav_z
    pred_z = [current_z]
    pred_yaw = [yaw]
    pred_pitch = [pitch]
    clipped_steps = 0
    collision_clipped_steps = 0
    nav_clipped_steps = 0
    ground_miss_clipped_steps = 0
    step_height_clipped_steps = 0
    action_counts = {k: 0 for k in ["normal", "walk", "crouch", "scope", "no_move"]}
    jump_steps = 0
    for k in range(horizon):
        t = start + k
        look_t = min(max(t + args.look_lag, 0), look_delta.shape[1] - 1)
        yaw = float((yaw + look_delta[pidx, look_t, 0] * look_scales["yaw_scale"] + 180.0) % 360.0 - 180.0)
        pitch = float(np.clip(pitch + look_delta[pidx, look_t, 1] * look_scales["pitch_scale"], -89.0, 89.0))
        area_id = navmesh.locate_area(xy, area_id, args.nav_tolerance)
        action_t = min(max(t + args.action_lag, 0), actions.shape[1] - 1)
        action = actions[pidx, action_t]
        cond = action_condition(action)
        action_counts[cond] += 1
        jump_steps += int(bool(action[4]))
        (
            xy,
            vel,
            area_id,
            clipped,
            collision_clipped,
            nav_clipped,
            ground_miss_clipped,
            step_height_clipped,
        ) = step_map_aware(
            xy, vel, area_id, action, yaw, navmesh, dt,
            args.base_speed, args.accel, args.friction, args.walk_scale,
            args.crouch_scale, args.nav_tolerance, motion_params,
            current_z, bsp_ground, args.max_step_height, args.player_radius,
            args.player_height, args.enable_bsp_collision, args.velocity_model,
            velocity_weights, args.max_speed, args.max_substep_distance,
        )
        clipped_steps += int(clipped)
        collision_clipped_steps += int(collision_clipped)
        nav_clipped_steps += int(nav_clipped)
        ground_miss_clipped_steps += int(ground_miss_clipped)
        step_height_clipped_steps += int(step_height_clipped)
        if bsp_ground is not None:
            ground = bsp_ground.query(xy, current_z)
            current_z = float(ground["z"]) if ground is not None else navmesh.ground_z(xy, area_id)
        else:
            current_z = navmesh.ground_z(xy, area_id)
        pred_xy.append(xy.copy())
        pred_z.append(current_z)
        pred_yaw.append(yaw)
        pred_pitch.append(pitch)
    return {
        "xy": np.asarray(pred_xy, dtype=np.float64),
        "z": np.asarray(pred_z, dtype=np.float64),
        "yaw": np.asarray(pred_yaw, dtype=np.float64),
        "pitch": np.asarray(pred_pitch, dtype=np.float64),
        "clipped_steps": clipped_steps,
        "collision_clipped_steps": collision_clipped_steps,
        "nav_clipped_steps": nav_clipped_steps,
        "ground_miss_clipped_steps": ground_miss_clipped_steps,
        "step_height_clipped_steps": step_height_clipped_steps,
        "action_counts": action_counts,
        "jump_steps": jump_steps,
    }


def evaluate(
    mem: Any,
    navmesh: NavmeshIndex,
    bsp_ground: BspGroundIndex | None,
    look_scales: dict[str, float],
    motion_params: dict[str, Any] | None,
    velocity_weights: np.ndarray | None,
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
            pred = rollout(mem, pidx, start, horizon, navmesh, bsp_ground, look_scales, motion_params, velocity_weights, args)
            xy_err = np.linalg.norm(pred["xy"] - gt_xy, axis=1)
            z_err = np.abs(pred["z"] - pos[pidx, start : start + horizon + 1, 2].astype(np.float64))
            gt_yaw = yaw[pidx, start : start + horizon + 1].astype(np.float64)
            gt_pitch = pitch[pidx, start : start + horizon + 1].astype(np.float64)
            yaw_err = np.abs(angle_diff_deg(pred["yaw"], gt_yaw))
            pitch_err = np.abs(pred["pitch"] - gt_pitch)
            pred_disp_vec = pred["xy"][-1] - pred["xy"][0]
            gt_disp_vec = gt_xy[-1] - gt_xy[0]
            pred_disp = float(np.linalg.norm(pred_disp_vec))
            gt_unit = gt_disp_vec / max(gt_disp, 1e-6)
            pred_minus_gt_disp = pred_disp_vec - gt_disp_vec
            along_error = float(pred_minus_gt_disp @ gt_unit)
            cross_error = float(abs(pred_minus_gt_disp[0] * gt_unit[1] - pred_minus_gt_disp[1] * gt_unit[0]))
            action_counts = pred["action_counts"]
            rows.append({
                "player_idx": int(pidx),
                "start": int(start),
                "final_error_xy": float(xy_err[-1]),
                "mean_error_xy": float(xy_err.mean()),
                "final_yaw_error": float(yaw_err[-1]),
                "mean_yaw_error": float(yaw_err.mean()),
                "final_z_error": float(z_err[-1]),
                "mean_z_error": float(z_err.mean()),
                "final_pitch_error": float(pitch_err[-1]),
                "mean_pitch_error": float(pitch_err.mean()),
                "gt_displacement": gt_disp,
                "pred_displacement": pred_disp,
                "displacement_ratio": pred_disp / max(gt_disp, 1e-6),
                "final_along_track_error": along_error,
                "final_cross_track_error": cross_error,
                "off_nav_fraction": navmesh.path_off_nav_fraction(pred["xy"], args.nav_tolerance),
                "clipped_fraction": float(pred["clipped_steps"]) / max(1, horizon),
                "collision_clipped_fraction": float(pred["collision_clipped_steps"]) / max(1, horizon),
                "nav_clipped_fraction": float(pred["nav_clipped_steps"]) / max(1, horizon),
                "ground_miss_clipped_fraction": float(pred["ground_miss_clipped_steps"]) / max(1, horizon),
                "step_height_clipped_fraction": float(pred["step_height_clipped_steps"]) / max(1, horizon),
                "jump_fraction": float(pred["jump_steps"]) / max(1, horizon),
                "no_move_fraction": float(action_counts["no_move"]) / max(1, horizon),
                "normal_move_fraction": float(action_counts["normal"]) / max(1, horizon),
                "walk_fraction": float(action_counts["walk"]) / max(1, horizon),
                "crouch_fraction": float(action_counts["crouch"]) / max(1, horizon),
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
        "mean_displacement_ratio": stat("displacement_ratio"),
        "mean_final_along_track_error": stat("final_along_track_error"),
        "mean_final_cross_track_error": stat("final_cross_track_error"),
        "mean_final_yaw_error": stat("final_yaw_error"),
        "mean_yaw_error": stat("mean_yaw_error"),
        "mean_final_z_error": stat("final_z_error"),
        "mean_z_error": stat("mean_z_error"),
        "mean_final_pitch_error": stat("final_pitch_error"),
        "mean_pitch_error": stat("mean_pitch_error"),
        "mean_off_nav_fraction": stat("off_nav_fraction"),
        "mean_clipped_fraction": stat("clipped_fraction"),
        "mean_collision_clipped_fraction": stat("collision_clipped_fraction"),
        "mean_nav_clipped_fraction": stat("nav_clipped_fraction"),
        "mean_ground_miss_clipped_fraction": stat("ground_miss_clipped_fraction"),
        "mean_step_height_clipped_fraction": stat("step_height_clipped_fraction"),
        "mean_jump_fraction": stat("jump_fraction"),
        "mean_no_move_fraction": stat("no_move_fraction"),
        "mean_normal_move_fraction": stat("normal_move_fraction"),
        "mean_walk_fraction": stat("walk_fraction"),
        "mean_crouch_fraction": stat("crouch_fraction"),
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
    ap.add_argument("--action-lag", type=int, default=0)
    ap.add_argument("--nav-tolerance", type=float, default=24.0)
    ap.add_argument("--min-gt-displacement", type=float, default=50.0)
    ap.add_argument("--base-speed", type=float, default=105.0)
    ap.add_argument("--accel", type=float, default=420.0)
    ap.add_argument("--friction", type=float, default=7.5)
    ap.add_argument("--walk-scale", type=float, default=0.52)
    ap.add_argument("--crouch-scale", type=float, default=0.45)
    ap.add_argument("--motion-params", type=Path)
    ap.add_argument("--velocity-model", choices=["rule", "linear"], default="rule")
    ap.add_argument("--velocity-alpha", type=float, default=10.0)
    ap.add_argument("--velocity-min-speed", type=float, default=2.0)
    ap.add_argument("--max-speed", type=float, default=320.0)
    ap.add_argument("--max-substep-distance", type=float, default=8.0)
    ap.add_argument("--bsp-faces-npz", type=Path)
    ap.add_argument("--bsp-cell-size", type=float, default=128.0)
    ap.add_argument("--max-ground-slope-deg", type=float, default=55.0)
    ap.add_argument("--max-step-height", type=float, default=72.0)
    ap.add_argument("--player-radius", type=float, default=18.0)
    ap.add_argument("--player-height", type=float, default=72.0)
    ap.add_argument("--enable-bsp-collision", action="store_true")
    ap.add_argument("--initial-velocity", choices=["zero", "replay"], default="zero")
    args = ap.parse_args()

    meta = load_json(args.episode_memory_dir / "episode_memory_meta_v0.json")
    mem = np.load(args.episode_memory_dir / "episode_memory_v0.npz")
    navmesh = NavmeshIndex.from_json(args.navmesh_path)
    motion_params = load_json(args.motion_params) if args.motion_params else None
    bsp_ground = BspGroundIndex.from_npz(args.bsp_faces_npz, args.bsp_cell_size, args.max_ground_slope_deg) if args.bsp_faces_npz else None
    player_count = len(meta["player_stems"])
    train_players = list(range(min(args.train_player_limit, player_count)))
    eval_players = list(range(args.eval_player_start, min(args.eval_player_start + args.eval_player_limit, player_count)))
    starts = list(range(args.eval_start, args.eval_stop, args.stride))
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    look_scales = fit_look_scales(mem, train_players, args.train_start, args.train_stop, args.look_lag)
    velocity_weights = None
    velocity_fit: dict[str, Any] | None = None
    if args.velocity_model == "linear":
        x_vel, y_vel = build_velocity_samples(
            mem, train_players, args.train_start, args.train_stop,
            args.fps, args.action_lag, args.velocity_min_speed,
        )
        if len(x_vel) == 0:
            raise ValueError("No velocity samples available for linear velocity model")
        velocity_weights = fit_ridge(x_vel, y_vel, args.velocity_alpha)
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

    results = {
        str(h): evaluate(mem, navmesh, bsp_ground, look_scales, motion_params, velocity_weights, eval_players, starts, h, args)
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
            "action_lag": args.action_lag,
            "nav_tolerance": args.nav_tolerance,
            "min_gt_displacement": args.min_gt_displacement,
            "base_speed": args.base_speed,
            "accel": args.accel,
            "friction": args.friction,
            "walk_scale": args.walk_scale,
            "crouch_scale": args.crouch_scale,
            "motion_params": str(args.motion_params) if args.motion_params else None,
            "velocity_model": args.velocity_model,
            "velocity_alpha": args.velocity_alpha,
            "velocity_min_speed": args.velocity_min_speed,
            "max_speed": args.max_speed,
            "max_substep_distance": args.max_substep_distance,
            "bsp_faces_npz": str(args.bsp_faces_npz) if args.bsp_faces_npz else None,
            "bsp_cell_size": args.bsp_cell_size,
            "max_ground_slope_deg": args.max_ground_slope_deg,
            "max_step_height": args.max_step_height,
            "player_radius": args.player_radius,
            "player_height": args.player_height,
            "enable_bsp_collision": args.enable_bsp_collision,
            "initial_velocity": args.initial_velocity,
        },
        "look_scales": look_scales,
        "velocity_fit": velocity_fit,
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
                "z": results[str(h)]["mean_final_z_error"],
                "disp_ratio": results[str(h)]["mean_displacement_ratio"],
                "along": results[str(h)]["mean_final_along_track_error"],
                "cross": results[str(h)]["mean_final_cross_track_error"],
                "off_nav": results[str(h)]["mean_off_nav_fraction"],
                "clipped": results[str(h)]["mean_clipped_fraction"],
                "collision_clipped": results[str(h)]["mean_collision_clipped_fraction"],
                "ground_miss": results[str(h)]["mean_ground_miss_clipped_fraction"],
                "step_height": results[str(h)]["mean_step_height_clipped_fraction"],
                "samples": results[str(h)]["sample_count"],
            }
            for h in horizons
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
