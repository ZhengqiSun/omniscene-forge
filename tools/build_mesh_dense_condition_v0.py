#!/usr/bin/env python3
"""Build a first-person dense condition from Map Memory.

Compared with build_dense_condition_v0.py, this replaces the environment depth
channel from the dataset depth video with a projected `_world_` OBJ depth map.
Player masks are generated from player pose capsules in the episode JSON. The
dataset segmentation stream is optional teacher/QA data, not an input source for
the dense tensor.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PLAYER_RADIUS = 18.0
PLAYER_HEIGHT = 72.0
PLAYER_OCCLUSION_TOLERANCE = 80.0
PLAYER_MIN_VISIBLE_PIXELS = 48
PLAYER_Z_OFFSET = 0.0
CAMERA_YAW_OFFSET = 0.0
CAMERA_PITCH_OFFSET = 0.0
NAV_SEMANTIC_Z_OFFSET = 2.0
PLAYER_MASK_MODE = "capsule"


def perspective_correct_z(w0: np.ndarray, w1: np.ndarray, w2: np.ndarray, zs: np.ndarray) -> np.ndarray:
    inv_z0 = 1.0 / max(float(zs[0]), 1e-6)
    inv_z1 = 1.0 / max(float(zs[1]), 1e-6)
    inv_z2 = 1.0 / max(float(zs[2]), 1e-6)
    inv_z = w0 * inv_z0 + w1 * inv_z1 + w2 * inv_z2
    return np.where(np.abs(inv_z) > 1e-8, 1.0 / inv_z, np.inf)


def import_tool(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_renderer_cache(match_dir: Path, tools_dir: Path | None = None, bsp_faces_npz: Path | None = None) -> dict[str, Any]:
    if tools_dir is None:
        tools_dir = Path(__file__).resolve().parent
    mesh_mod = import_tool(tools_dir / "build_mesh_projection_v0.py", "mesh_projection_v0")
    dense_mod = import_tool(tools_dir / "build_dense_condition_v0.py", "dense_condition_v0")
    obj_path = mesh_mod.find_world_obj(match_dir)
    vertices, faces = mesh_mod.load_obj_mesh(obj_path)
    navmesh = load_json(match_dir / "navmesh.json")
    nav_semantic = build_nav_semantic_triangles(navmesh)
    cache = {
        "mesh_mod": mesh_mod,
        "dense_mod": dense_mod,
        "obj_path": obj_path,
        "vertices": vertices,
        "faces": faces,
        "navmesh": navmesh,
        "nav_semantic": nav_semantic,
    }
    bsp_path = bsp_faces_npz or os.environ.get("MAP_MEMORY_BSP_FACES_NPZ")
    if bsp_path:
        load_bsp_faces_cache(cache, Path(bsp_path))
    return cache


def load_bsp_faces_cache(cache: dict[str, Any], bsp_faces_npz: Path) -> None:
    data = np.load(bsp_faces_npz)
    vertices = data["vertices"].astype(np.float32)
    faces = data["faces"].astype(np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"{bsp_faces_npz} vertices must be [N,3], got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"{bsp_faces_npz} faces must be [M,3], got {faces.shape}")
    cache["bsp_faces_npz"] = str(bsp_faces_npz)
    cache["bsp_vertices"] = vertices
    cache["bsp_faces"] = faces
    component_counts: dict[str, int] = {
        "bsp_vertices": int(len(vertices)),
        "bsp_faces": int(len(faces)),
    }
    features = ["visual_faces"]
    for key in ["base_vertices", "base_faces", "displacement_vertices", "displacement_faces"]:
        if key in data.files:
            component_counts[key] = int(len(data[key]))
    if component_counts.get("displacement_faces", 0) > 0:
        features.append("displacement")
    cache["bsp_component_counts"] = component_counts
    cache["bsp_backend_features"] = features


def describe_geometry_backend(mesh_backend: str, cache: dict[str, Any]) -> dict[str, Any]:
    if mesh_backend in {"gpu", "cpu"}:
        backend_id = f"obj_world_{mesh_backend}"
        return {
            "geometry_backend_id": backend_id,
            "geometry_backend_family": "obj_world",
            "backend_features": ["world_obj_visual_mesh"],
            "backend_signature": backend_id,
            "geometry_source_paths": {
                "world_obj": str(cache.get("obj_path")),
            },
            "component_counts": {
                "world_obj_vertices": int(len(cache["vertices"])),
                "world_obj_faces": int(len(cache["faces"])),
            },
        }
    if mesh_backend in {"bsp_faces_gpu", "bsp_faces_cpu"}:
        suffix = "gpu" if mesh_backend.endswith("_gpu") else "cpu"
        features = list(cache.get("bsp_backend_features", ["visual_faces"]))
        has_displacement = "displacement" in features
        backend_id = f"bsp_faces{'_disp' if has_displacement else ''}_{suffix}"
        return {
            "geometry_backend_id": backend_id,
            "geometry_backend_family": "bsp_faces",
            "backend_features": features,
            "backend_signature": f"{backend_id}:{Path(str(cache.get('bsp_faces_npz'))).name}",
            "geometry_source_paths": {
                "world_obj_reference": str(cache.get("obj_path")),
                "bsp_faces_npz": cache.get("bsp_faces_npz"),
            },
            "component_counts": {
                "world_obj_vertices": int(len(cache["vertices"])),
                "world_obj_faces": int(len(cache["faces"])),
                **cache.get("bsp_component_counts", {}),
            },
        }
    return {
        "geometry_backend_id": mesh_backend,
        "geometry_backend_family": "unknown",
        "backend_features": [],
        "backend_signature": mesh_backend,
        "geometry_source_paths": {"world_obj": str(cache.get("obj_path"))},
        "component_counts": {},
    }


def ensure_gpu_mesh_renderer(
    cache: dict[str, Any],
    tools_dir: Path | None = None,
    vertices_key: str = "vertices",
    faces_key: str = "faces",
    renderer_key: str = "gpu_mesh_renderer",
) -> tuple[Any, Any]:
    if tools_dir is None:
        tools_dir = Path(__file__).resolve().parent
    gpu_mod = cache.get("gpu_mesh_mod")
    if gpu_mod is None:
        gpu_mod = import_tool(tools_dir / "gpu_mesh_renderer_v0.py", "gpu_mesh_renderer_v0")
        cache["gpu_mesh_mod"] = gpu_mod
    gpu_renderer = cache.get(renderer_key)
    if gpu_renderer is None:
        gpu_renderer = gpu_mod.GpuMeshDepthRenderer.from_numpy(cache[vertices_key], cache[faces_key])
        cache[renderer_key] = gpu_renderer
    return gpu_mod, gpu_renderer


def area_corners(area: dict[str, Any], z_offset: float = NAV_SEMANTIC_Z_OFFSET) -> np.ndarray:
    nw = area["nw_corner"]
    se = area["se_corner"]
    ne_z = float(area.get("ne_z", nw[2]))
    sw_z = float(area.get("sw_z", se[2]))
    return np.asarray([
        [float(nw[0]), float(nw[1]), float(nw[2]) + z_offset],
        [float(se[0]), float(nw[1]), ne_z + z_offset],
        [float(se[0]), float(se[1]), float(se[2]) + z_offset],
        [float(nw[0]), float(se[1]), sw_z + z_offset],
    ], dtype=np.float32)


def build_nav_semantic_triangles(navmesh: dict[str, Any]) -> dict[str, Any]:
    place_names = list(navmesh.get("place_names", []))
    place_to_id = {name: idx + 1 for idx, name in enumerate(place_names)}
    vertices: list[np.ndarray] = []
    faces: list[list[int]] = []
    face_values: list[float] = []
    face_area_ids: list[int] = []
    face_places: list[str] = []
    denom = float(max(1, len(place_to_id) + 1))
    for area_id, area in navmesh["areas"].items():
        place = area.get("place_name") or ""
        place_id = place_to_id.get(place, 0)
        if place_id <= 0:
            continue
        base = len(vertices)
        corners = area_corners(area)
        vertices.extend([corners[i] for i in range(4)])
        value = (place_id + 1) / denom
        faces.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])
        face_values.extend([value, value])
        face_area_ids.extend([int(area_id), int(area_id)])
        face_places.extend([place, place])
    if not vertices:
        return {
            "vertices": np.zeros((0, 3), dtype=np.float32),
            "faces": np.zeros((0, 3), dtype=np.int32),
            "face_values": np.zeros((0,), dtype=np.float32),
            "face_area_ids": np.zeros((0,), dtype=np.int32),
            "face_places": [],
            "place_to_id": place_to_id,
        }
    return {
        "vertices": np.asarray(vertices, dtype=np.float32),
        "faces": np.asarray(faces, dtype=np.int32),
        "face_values": np.asarray(face_values, dtype=np.float32),
        "face_area_ids": np.asarray(face_area_ids, dtype=np.int32),
        "face_places": face_places,
        "place_to_id": place_to_id,
    }


def rasterize_face_values(
    projected: np.ndarray,
    z_cam: np.ndarray,
    faces: np.ndarray,
    face_values: np.ndarray,
    width: int,
    height: int,
    near: float,
    far: float,
    max_triangles: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    semantic = np.zeros((height, width), dtype=np.float32)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    drawn = 0
    skipped = 0
    if len(faces) == 0:
        return semantic, depth, {"triangles_drawn": 0, "triangles_skipped": 0, "hit_ratio": 0.0}
    step = 1 if max_triangles <= 0 else max(1, int(np.ceil(len(faces) / max_triangles)))
    for tri_idx in range(0, len(faces), step):
        tri = faces[tri_idx]
        pts = projected[tri, :2]
        zs = z_cam[tri]
        if not np.all(np.isfinite(pts)) or not np.all(np.isfinite(zs)):
            skipped += 1
            continue
        if np.any(zs <= near) or np.all(zs >= far):
            skipped += 1
            continue
        min_x = max(0, int(np.floor(np.min(pts[:, 0]))))
        max_x = min(width - 1, int(np.ceil(np.max(pts[:, 0]))))
        min_y = max(0, int(np.floor(np.min(pts[:, 1]))))
        max_y = min(height - 1, int(np.ceil(np.max(pts[:, 1]))))
        if min_x > max_x or min_y > max_y:
            skipped += 1
            continue
        x0, y0 = pts[0]
        x1, y1 = pts[1]
        x2, y2 = pts[2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(float(denom)) < 1e-6:
            skipped += 1
            continue
        yy, xx = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
        w0 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denom
        w1 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not np.any(inside):
            skipped += 1
            continue
        z = perspective_correct_z(w0, w1, w2, zs)
        region = depth[min_y : max_y + 1, min_x : max_x + 1]
        update = inside & (z >= near) & (z <= far) & (z < region)
        if not np.any(update):
            skipped += 1
            continue
        region[update] = z[update]
        semantic_region = semantic[min_y : max_y + 1, min_x : max_x + 1]
        semantic_region[update] = float(face_values[tri_idx])
        drawn += 1
    return semantic, depth, {
        "triangles_drawn": int(drawn),
        "triangles_skipped": int(skipped),
        "hit_ratio": float((semantic > 0).mean()),
    }


def render_nav_place_semantic(
    cache: dict[str, Any],
    cam_pos: np.ndarray,
    yaw: float,
    pitch: float,
    pitch_sign: float,
    width: int,
    height: int,
    fov_x: float,
    near: float,
    far: float,
    env_depth_units: np.ndarray,
    occlusion_tolerance: float = 24.0,
    camera_yaw_offset: float = CAMERA_YAW_OFFSET,
    camera_pitch_offset: float = CAMERA_PITCH_OFFSET,
) -> tuple[np.ndarray, dict[str, Any]]:
    nav = cache.get("nav_semantic")
    if not nav:
        return np.zeros((height, width), dtype=np.float32), {"nav_semantic": {"enabled": False}}
    projected, z_cam = cache["mesh_mod"].project_vertices(
        nav["vertices"],
        cam_pos,
        yaw + camera_yaw_offset,
        pitch + camera_pitch_offset,
        pitch_sign,
        width,
        height,
        fov_x,
    )
    semantic, semantic_depth, stats = rasterize_face_values(
        projected,
        z_cam,
        nav["faces"],
        nav["face_values"],
        width,
        height,
        near,
        far,
        max_triangles=100000,
    )
    visible = (semantic > 0) & ((~np.isfinite(env_depth_units)) | (semantic_depth <= env_depth_units + occlusion_tolerance))
    semantic = np.where(visible, semantic, 0.0).astype(np.float32)
    visible_places = summarize_visible_places(semantic, nav["place_to_id"])
    return semantic, {
        "nav_semantic": {
            "enabled": True,
            "place_count": len(nav["place_to_id"]),
            "hit_ratio_before_occlusion": stats["hit_ratio"],
            "hit_ratio_after_occlusion": float((semantic > 0).mean()),
            "visible_place_value_count": int(len(visible_places)),
            "visible_places": visible_places,
            "dominant_place": visible_places[0]["place"] if visible_places else None,
            "triangles_drawn": stats["triangles_drawn"],
            "triangles_skipped": stats["triangles_skipped"],
            "encoding": "place_id divided by number of navmesh places; 0 means no visible nav area",
            "place_to_id": nav["place_to_id"],
        }
    }


def summarize_visible_places(semantic: np.ndarray, place_to_id: dict[str, int]) -> list[dict[str, Any]]:
    if not place_to_id:
        return []
    id_to_place = {idx: name for name, idx in place_to_id.items()}
    denom = float(max(place_to_id.values()) + 1)
    total = int((semantic > 0).sum())
    if total == 0:
        return []
    place_ids = np.rint(semantic[semantic > 0] * denom - 1).astype(np.int32)
    unique, counts = np.unique(place_ids, return_counts=True)
    rows = []
    for place_id, count in zip(unique, counts):
        place = id_to_place.get(int(place_id))
        if not place:
            continue
        rows.append({
            "place": place,
            "place_id": int(place_id),
            "pixels": int(count),
            "fraction_of_visible_nav": float(count / total),
            "fraction_of_image": float(count / semantic.size),
        })
    rows.sort(key=lambda r: r["pixels"], reverse=True)
    return rows


def player_files(episode_dir: Path) -> list[Path]:
    return sorted(
        p for p in episode_dir.glob("*.json")
        if "_team_" in p.name and "_player_" in p.name and p.name.endswith("_inst_000.json")
    )


def parse_team_player(stem: str) -> tuple[int, int]:
    parts = stem.split("_")
    team = int(parts[parts.index("team") + 1])
    player = int(parts[parts.index("player") + 1])
    return team, player


def load_episode_memory(memory_dir: Path | None) -> dict[str, Any] | None:
    if memory_dir is None:
        return None
    return {
        "meta": load_json(memory_dir / "episode_memory_meta_v0.json"),
        "npz": np.load(memory_dir / "episode_memory_v0.npz"),
        "memory_dir": str(memory_dir),
    }


def frame_from_episode_memory(episode_memory: dict[str, Any], stem: str, frame_index: int) -> dict[str, Any]:
    meta = episode_memory["meta"]
    mem = episode_memory["npz"]
    idx = meta["player_stems"].index(stem)
    pos = mem["position"][idx, frame_index]
    cam = mem["camera_position"][idx, frame_index]
    return {
        "x": float(pos[0]),
        "y": float(pos[1]),
        "z": float(pos[2]),
        "camera_position": cam.astype(float).tolist(),
        "yaw": float(mem["yaw"][idx, frame_index]),
        "pitch": float(mem["pitch"][idx, frame_index]),
        "health": float(mem["health"][idx, frame_index]),
    }


def stamp_disk(channel: np.ndarray, cx: float, cy: float, radius: float, value: float) -> None:
    h, w = channel.shape
    x0 = max(0, int(np.floor(cx - radius)))
    x1 = min(w - 1, int(np.ceil(cx + radius)))
    y0 = max(0, int(np.floor(cy - radius)))
    y1 = min(h - 1, int(np.ceil(cy + radius)))
    if x1 < x0 or y1 < y0:
        return
    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
    region = channel[y0 : y1 + 1, x0 : x1 + 1]
    region[mask] = value


def stamp_ellipse(channel: np.ndarray, cx: float, cy: float, rx: float, ry: float, value: float) -> None:
    h, w = channel.shape
    rx = max(float(rx), 0.75)
    ry = max(float(ry), 0.75)
    x0 = max(0, int(np.floor(cx - rx)))
    x1 = min(w - 1, int(np.ceil(cx + rx)))
    y0 = max(0, int(np.floor(cy - ry)))
    y1 = min(h - 1, int(np.ceil(cy + ry)))
    if x1 < x0 or y1 < y0:
        return
    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    region = channel[y0 : y1 + 1, x0 : x1 + 1]
    region[mask] = value


def stamp_player_shape(
    target: np.ndarray,
    player_depth: np.ndarray,
    orient_sin: np.ndarray,
    orient_cos: np.ndarray,
    u: float,
    y_mid: float,
    pixel_radius: float,
    capsule_h: float,
    z_norm: float,
    yaw_sin: float,
    yaw_cos: float,
    mode: str,
) -> dict[str, Any]:
    if mode == "multipart":
        # A simple first-person silhouette: torso/head/legs. It is still
        # geometric Memory, but avoids the over-wide vertical capsule.
        parts = [
            ("legs", 0.70, 0.82, 0.86, 0.22),
            ("torso", 0.47, 1.04, 1.30, 0.46),
            ("head", 0.18, 0.62, 0.58, 0.18),
        ]
        for _, center_frac, rx_mul, ry_mul, _ in parts:
            cy = y_mid - capsule_h * 0.5 + capsule_h * center_frac
            rx = pixel_radius * rx_mul
            ry = pixel_radius * ry_mul
            stamp_ellipse(target, u, cy, rx, ry, 1.0)
            stamp_ellipse(player_depth, u, cy, rx, ry, z_norm)
            stamp_ellipse(orient_sin, u, cy, rx, ry, yaw_sin)
            stamp_ellipse(orient_cos, u, cy, rx, ry, yaw_cos)
        return {"mask_mode": "multipart", "parts": [p[0] for p in parts]}

    steps = max(2, int(capsule_h / max(pixel_radius, 1.0)))
    for yy in np.linspace(y_mid - capsule_h * 0.5, y_mid + capsule_h * 0.5, steps):
        stamp_disk(target, u, yy, pixel_radius, 1.0)
        stamp_disk(player_depth, u, yy, pixel_radius, z_norm)
        stamp_disk(orient_sin, u, yy, pixel_radius, yaw_sin)
        stamp_disk(orient_cos, u, yy, pixel_radius, yaw_cos)
    return {"mask_mode": "capsule", "steps": int(steps)}


def stamp_player_visibility_shape(
    u: float,
    y_mid: float,
    pixel_radius: float,
    capsule_h: float,
    mode: str,
    height: int,
    width: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    mask = np.zeros((height, width), dtype=np.float32)
    scratch = np.zeros_like(mask)
    meta = stamp_player_shape(
        mask,
        scratch,
        scratch,
        scratch,
        u,
        y_mid,
        pixel_radius,
        capsule_h,
        1.0,
        0.0,
        1.0,
        mode,
    )
    return mask > 0.5, meta


def project_memory_players(
    episode_dir: Path,
    ego_stem: str,
    frame_index: int,
    mesh_mod: Any,
    cam_pos: np.ndarray,
    yaw: float,
    pitch: float,
    pitch_sign: float,
    width: int,
    height: int,
    fov_x: float,
    far: float,
    env_depth_units: np.ndarray,
    episode_memory: dict[str, Any] | None = None,
    player_radius: float = PLAYER_RADIUS,
    player_height: float = PLAYER_HEIGHT,
    occlusion_tolerance: float = PLAYER_OCCLUSION_TOLERANCE,
    min_visible_pixels: int = PLAYER_MIN_VISIBLE_PIXELS,
    player_z_offset: float = PLAYER_Z_OFFSET,
    camera_yaw_offset: float = CAMERA_YAW_OFFSET,
    camera_pitch_offset: float = CAMERA_PITCH_OFFSET,
    player_mask_mode: str = PLAYER_MASK_MODE,
) -> tuple[np.ndarray, dict[str, Any]]:
    ego_team, _ = parse_team_player(ego_stem)
    other_player_mask = np.zeros((height, width), dtype=np.float32)
    player_depth = np.zeros((height, width), dtype=np.float32)
    orient_sin = np.zeros((height, width), dtype=np.float32)
    orient_cos = np.zeros((height, width), dtype=np.float32)
    players_meta: list[dict[str, Any]] = []

    if episode_memory is None:
        players_iter = [(path.stem, path) for path in player_files(episode_dir)]
    else:
        players_iter = [(stem, None) for stem in episode_memory["meta"]["player_stems"]]

    for stem, path in players_iter:
        if stem == ego_stem:
            continue
        team_id, player_idx = parse_team_player(stem)
        if episode_memory is None:
            assert path is not None
            frames = load_json(path)
            if frame_index >= len(frames):
                continue
            frame = frames[frame_index]
        else:
            frame = frame_from_episode_memory(episode_memory, stem, frame_index)
        if not np.isfinite(frame["x"]):
            continue
        if float(frame.get("health", 100)) <= 0:
            continue
        base = np.asarray([frame["x"], frame["y"], frame["z"] + player_z_offset], dtype=np.float32)
        yaw_other = float(frame.get("yaw", 0.0))
        samples = np.asarray([
            [base[0], base[1], base[2] + 8.0],
            [base[0], base[1], base[2] + player_height * 0.5],
            [base[0], base[1], base[2] + player_height],
        ], dtype=np.float32)
        proj, z_cam = mesh_mod.project_vertices(samples, cam_pos, yaw + camera_yaw_offset, pitch + camera_pitch_offset, pitch_sign, width, height, fov_x)
        valid = (z_cam > 1.0) & (z_cam < far)
        if not np.any(valid):
            continue
        u = float(np.mean(proj[valid, 0]))
        v_top = float(np.min(proj[valid, 1]))
        v_bot = float(np.max(proj[valid, 1]))
        z = float(np.mean(z_cam[valid]))
        pixel_radius = max(2.0, player_radius / max(z, 1.0) / np.tan(np.radians(fov_x) / 2.0) * width * 0.5)
        y_mid = (v_top + v_bot) * 0.5
        capsule_h = max(pixel_radius * 2.0, abs(v_bot - v_top) + pixel_radius * 2.0)
        if u < -pixel_radius or u >= width + pixel_radius or y_mid < -capsule_h or y_mid >= height + capsule_h:
            continue

        raw_player_mask, shape_meta = stamp_player_visibility_shape(
            u,
            y_mid,
            pixel_radius,
            capsule_h,
            player_mask_mode,
            height,
            width,
        )
        visible_pixels = raw_player_mask & np.isfinite(env_depth_units) & (z <= env_depth_units + occlusion_tolerance)
        if int(visible_pixels.sum()) < min_visible_pixels:
            continue
        tmp_target = np.zeros((height, width), dtype=np.float32)
        tmp_depth = np.zeros((height, width), dtype=np.float32)
        tmp_sin = np.zeros((height, width), dtype=np.float32)
        tmp_cos = np.zeros((height, width), dtype=np.float32)
        relative_yaw = yaw_other - yaw
        stamp_player_shape(
            tmp_target,
            tmp_depth,
            tmp_sin,
            tmp_cos,
            u,
            y_mid,
            pixel_radius,
            capsule_h,
            np.clip(z / far, 0.0, 1.0),
            np.sin(np.radians(relative_yaw)),
            np.cos(np.radians(relative_yaw)),
            player_mask_mode,
        )
        visible_update = visible_pixels & (tmp_target > 0.5)
        other_player_mask[visible_update] = 1.0
        player_depth[visible_update] = tmp_depth[visible_update]
        orient_sin[visible_update] = tmp_sin[visible_update]
        orient_cos[visible_update] = tmp_cos[visible_update]
        occluded_pixels = int(raw_player_mask.sum() - visible_pixels.sum())
        visible = True
        players_meta.append({
            "stem": stem,
            "team_id": team_id,
            "player_index": player_idx,
            "is_enemy": team_id != ego_team,
            "z_cam": z,
            "center_uv": [u, y_mid],
            "pixel_radius": pixel_radius,
            "capsule_height_px": capsule_h,
            "top_bottom_v": [y_mid - capsule_h * 0.5, y_mid + capsule_h * 0.5],
            "visible_by_depth_test": visible,
            "visible_pixels_after_depth_test": int(visible_pixels.sum()),
            "occluded_pixels_after_depth_test": occluded_pixels,
            "yaw_world_deg": yaw_other,
            "yaw_relative_to_ego_deg": relative_yaw,
            **shape_meta,
        })

    return (
        np.stack([other_player_mask, player_depth, orient_sin, orient_cos], axis=0).astype(np.float32),
        {"memory_projected_players": players_meta},
    )


def render_memory_dense_condition(
    match_dir: Path,
    episode: str,
    ego_stem: str,
    frame_index: int,
    width: int,
    height: int,
    fov_x: float,
    near: float,
    far: float,
    pitch_sign: float,
    max_triangles: int,
    cache: dict[str, Any] | None = None,
    episode_memory: dict[str, Any] | None = None,
    player_radius: float = PLAYER_RADIUS,
    player_height: float = PLAYER_HEIGHT,
    occlusion_tolerance: float = PLAYER_OCCLUSION_TOLERANCE,
    min_visible_pixels: int = PLAYER_MIN_VISIBLE_PIXELS,
    player_z_offset: float = PLAYER_Z_OFFSET,
    camera_yaw_offset: float = CAMERA_YAW_OFFSET,
    camera_pitch_offset: float = CAMERA_PITCH_OFFSET,
    player_mask_mode: str = PLAYER_MASK_MODE,
    mesh_backend: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], Any]:
    if cache is None:
        cache = load_renderer_cache(match_dir)
    mesh_mod = cache["mesh_mod"]
    dense_mod = cache["dense_mod"]
    obj_path = cache["obj_path"]
    vertices = cache["vertices"]
    faces = cache["faces"]

    episode_dir = match_dir / "train" / episode
    if episode_memory is None:
        frames = load_json(episode_dir / f"{ego_stem}.json")
        frame = frames[frame_index]
    else:
        frame = frame_from_episode_memory(episode_memory, ego_stem, frame_index)
    cam_pos = np.asarray(frame.get("camera_position") or [frame["x"], frame["y"], frame["z"] + 64.0], dtype=np.float32)
    yaw = float(frame.get("yaw", frame.get("camera_rotation", [0, 0, 0])[2]))
    pitch = float(frame.get("pitch", frame.get("camera_rotation", [0, 0, 0])[1]))

    camera_points = None
    if hasattr(mesh_mod, "project_vertices_with_camera"):
        projected, z_cam, camera_points = mesh_mod.project_vertices_with_camera(
            vertices,
            cam_pos,
            yaw + camera_yaw_offset,
            pitch + camera_pitch_offset,
            pitch_sign,
            width,
            height,
            fov_x,
        )
    else:
        projected, z_cam = mesh_mod.project_vertices(
            vertices,
            cam_pos,
            yaw + camera_yaw_offset,
            pitch + camera_pitch_offset,
            pitch_sign,
            width,
            height,
            fov_x,
        )
    if mesh_backend == "auto":
        mesh_backend = os.environ.get("MAP_MEMORY_MESH_BACKEND", "cpu")
    if mesh_backend == "gpu":
        gpu_mod, gpu_renderer = ensure_gpu_mesh_renderer(cache)
        mesh_depth_units, mesh_stats = gpu_mod.rasterize_depth_gpu(
            gpu_renderer,
            cam_pos,
            yaw + camera_yaw_offset,
            pitch + camera_pitch_offset,
            pitch_sign,
            width,
            height,
            fov_x,
            near,
            far,
            max_triangles,
        )
    elif mesh_backend == "bsp_faces_gpu":
        if "bsp_vertices" not in cache or "bsp_faces" not in cache:
            bsp_env = os.environ.get("MAP_MEMORY_BSP_FACES_NPZ")
            if not bsp_env:
                raise ValueError("mesh_backend=bsp_faces_gpu requires --bsp-faces-npz or MAP_MEMORY_BSP_FACES_NPZ")
            load_bsp_faces_cache(cache, Path(bsp_env))
        gpu_mod, gpu_renderer = ensure_gpu_mesh_renderer(
            cache,
            vertices_key="bsp_vertices",
            faces_key="bsp_faces",
            renderer_key="gpu_bsp_faces_renderer",
        )
        mesh_depth_units, mesh_stats = gpu_mod.rasterize_depth_gpu(
            gpu_renderer,
            cam_pos,
            yaw + camera_yaw_offset,
            pitch + camera_pitch_offset,
            pitch_sign,
            width,
            height,
            fov_x,
            near,
            far,
            max_triangles,
        )
        mesh_stats["bsp_faces_npz"] = cache.get("bsp_faces_npz")
    elif mesh_backend == "cpu" and camera_points is not None and hasattr(mesh_mod, "rasterize_depth_clipped"):
        mesh_depth_units, mesh_stats = mesh_mod.rasterize_depth_clipped(
            camera_points,
            faces,
            width,
            height,
            fov_x,
            near,
            far,
            max_triangles,
        )
    elif mesh_backend == "bsp_faces_cpu":
        if "bsp_vertices" not in cache or "bsp_faces" not in cache:
            bsp_env = os.environ.get("MAP_MEMORY_BSP_FACES_NPZ")
            if not bsp_env:
                raise ValueError("mesh_backend=bsp_faces_cpu requires --bsp-faces-npz or MAP_MEMORY_BSP_FACES_NPZ")
            load_bsp_faces_cache(cache, Path(bsp_env))
        bsp_camera_points = mesh_mod.world_to_camera(
            cache["bsp_vertices"],
            cam_pos,
            yaw + camera_yaw_offset,
            pitch + camera_pitch_offset,
            pitch_sign,
        )
        mesh_depth_units, mesh_stats = mesh_mod.rasterize_depth_clipped(
            bsp_camera_points,
            cache["bsp_faces"],
            width,
            height,
            fov_x,
            near,
            far,
            max_triangles,
        )
        mesh_stats["bsp_faces_npz"] = cache.get("bsp_faces_npz")
    elif mesh_backend == "cpu":
        mesh_depth_units, mesh_stats = mesh_mod.rasterize_depth(
            projected,
            faces,
            width,
            height,
            near,
            far,
            max_triangles,
        )
    else:
        raise ValueError(f"Unsupported mesh_backend: {mesh_backend!r}")
    mesh_stats["mesh_backend"] = mesh_backend
    env_depth = mesh_mod.normalize_depth(mesh_depth_units, far)
    mesh_hit = np.isfinite(mesh_depth_units).astype(np.float32)

    rgb_bgr = dense_mod.read_frame(episode_dir / f"{ego_stem}.mp4", frame_index)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    nav_semantic, nav_meta = render_nav_place_semantic(
        cache,
        cam_pos,
        yaw,
        pitch,
        pitch_sign,
        width,
        height,
        fov_x,
        near,
        far,
        mesh_depth_units,
        camera_yaw_offset=camera_yaw_offset,
        camera_pitch_offset=camera_pitch_offset,
    )
    env_semantic = np.where(
        nav_semantic > 0,
        nav_semantic,
        mesh_hit * (1.0 / float(max(1, len(cache.get("nav_semantic", {}).get("place_to_id", {})) + 1))),
    ).astype(np.float32)
    non_nav_mesh_value = 1.0 / float(max(1, len(cache.get("nav_semantic", {}).get("place_to_id", {})) + 1))
    nav_meta["env_semantic_encoding"] = {
        "zero": "no visible world mesh",
        "non_nav_mesh_value": non_nav_mesh_value,
        "nav_place_value": "(place_id + 1) / (place_count + 1)",
        "place_id_source": "navmesh place_to_id",
    }
    player_channels, player_meta = project_memory_players(
        episode_dir,
        ego_stem,
        frame_index,
        mesh_mod,
        cam_pos,
        yaw,
        pitch,
        pitch_sign,
        width,
        height,
        fov_x,
        far,
        mesh_depth_units,
        episode_memory=episode_memory,
        player_radius=player_radius,
        player_height=player_height,
        occlusion_tolerance=occlusion_tolerance,
        min_visible_pixels=min_visible_pixels,
        player_z_offset=player_z_offset,
        camera_yaw_offset=camera_yaw_offset,
        camera_pitch_offset=camera_pitch_offset,
        player_mask_mode=player_mask_mode,
    )

    dense = np.concatenate([
        env_depth[None, ...],
        mesh_hit[None, ...],
        env_semantic[None, ...],
        player_channels,
    ], axis=0).astype(np.float32)
    channels = [
        "env_depth_norm_from_world_obj_projection",
        "env_mesh_hit_mask",
        "env_nav_place_semantic_from_static_memory",
        "other_player_mask_from_memory_player_capsules",
        "other_player_depth_norm_from_memory_capsules",
        "other_player_yaw_sin_relative_to_ego_from_memory",
        "other_player_yaw_cos_relative_to_ego_from_memory",
    ]
    backend_desc = describe_geometry_backend(mesh_backend, cache)

    meta = {
        "match_dir": str(match_dir),
        "obj_path": str(obj_path),
        "episode": episode,
        "ego_stem": ego_stem,
        "frame_index": frame_index,
        "camera_position": cam_pos.astype(float).tolist(),
        "yaw": yaw,
        "pitch": pitch,
        "pitch_sign": pitch_sign,
        "fov_x": fov_x,
        "near": near,
        "far": far,
        "player_radius": player_radius,
        "player_height": player_height,
        "occlusion_tolerance": occlusion_tolerance,
        "min_visible_pixels": min_visible_pixels,
        "player_z_offset": player_z_offset,
        "camera_yaw_offset": camera_yaw_offset,
        "camera_pitch_offset": camera_pitch_offset,
        "player_mask_mode": player_mask_mode,
        "mesh_backend": mesh_backend,
        **backend_desc,
        "output_shape": list(dense.shape),
        "channels": channels,
        "mesh_vertices": int(len(vertices)),
        "mesh_faces": int(len(faces)),
        "bsp_faces_npz": cache.get("bsp_faces_npz"),
        "bsp_vertices": int(len(cache["bsp_vertices"])) if "bsp_vertices" in cache else None,
        "bsp_faces": int(len(cache["bsp_faces"])) if "bsp_faces" in cache else None,
        "mesh_hit_ratio": float(mesh_hit.mean()),
        "nav_semantic_hit_ratio": float((env_semantic > 0).mean()),
        "nav_place_hit_ratio": float((nav_semantic > 0).mean()),
        "non_nav_mesh_semantic_hit_ratio": float(((env_semantic > 0) & (nav_semantic <= 0)).mean()),
        "vertex_camera_z_minmax": [float(np.min(z_cam)), float(np.max(z_cam))],
        "mesh_stats": mesh_stats,
        "depth_note": "Environment depth is v0 projected OBJ z-depth, normalized by far; final renderer should use clipped ray casting and verify FOV/camera intrinsics.",
        "input_policy": "Dense tensor is generated from Map Memory: world OBJ plus player JSON pose/state. Dataset RGB/depth/seg are teacher QA only, not dense input sources.",
        "episode_memory_dir": episode_memory.get("memory_dir") if episode_memory else None,
        **nav_meta,
        **player_meta,
    }
    return dense, mesh_depth_units, rgb, meta, dense_mod


def render_memory_player_channels_only(
    match_dir: Path,
    episode: str,
    ego_stem: str,
    frame_index: int,
    width: int,
    height: int,
    fov_x: float,
    far: float,
    pitch_sign: float,
    mesh_depth_units: np.ndarray,
    cache: dict[str, Any],
    episode_memory: dict[str, Any] | None = None,
    player_radius: float = PLAYER_RADIUS,
    player_height: float = PLAYER_HEIGHT,
    occlusion_tolerance: float = PLAYER_OCCLUSION_TOLERANCE,
    min_visible_pixels: int = PLAYER_MIN_VISIBLE_PIXELS,
    player_z_offset: float = PLAYER_Z_OFFSET,
    camera_yaw_offset: float = CAMERA_YAW_OFFSET,
    camera_pitch_offset: float = CAMERA_PITCH_OFFSET,
    player_mask_mode: str = PLAYER_MASK_MODE,
) -> tuple[np.ndarray, dict[str, Any]]:
    episode_dir = match_dir / "train" / episode
    if episode_memory is None:
        frames = load_json(episode_dir / f"{ego_stem}.json")
        frame = frames[frame_index]
    else:
        frame = frame_from_episode_memory(episode_memory, ego_stem, frame_index)
    cam_pos = np.asarray(frame.get("camera_position") or [frame["x"], frame["y"], frame["z"] + 64.0], dtype=np.float32)
    yaw = float(frame.get("yaw", frame.get("camera_rotation", [0, 0, 0])[2]))
    pitch = float(frame.get("pitch", frame.get("camera_rotation", [0, 0, 0])[1]))
    return project_memory_players(
        episode_dir,
        ego_stem,
        frame_index,
        cache["mesh_mod"],
        cam_pos,
        yaw,
        pitch,
        pitch_sign,
        width,
        height,
        fov_x,
        far,
        mesh_depth_units,
        episode_memory=episode_memory,
        player_radius=player_radius,
        player_height=player_height,
        occlusion_tolerance=occlusion_tolerance,
        min_visible_pixels=min_visible_pixels,
        player_z_offset=player_z_offset,
        camera_yaw_offset=camera_yaw_offset,
        camera_pitch_offset=camera_pitch_offset,
        player_mask_mode=player_mask_mode,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-dir", type=Path, required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--ego-stem", required=True)
    ap.add_argument("--frame-index", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=180)
    ap.add_argument("--fov-x", type=float, default=90.0)
    ap.add_argument("--near", type=float, default=4.0)
    ap.add_argument("--far", type=float, default=3000.0)
    ap.add_argument("--pitch-sign", type=float, default=1.0)
    ap.add_argument("--max-triangles", type=int, default=0, help="0 keeps all visible triangles; positive values are a debug speed cap.")
    ap.add_argument("--episode-memory-dir", type=Path, default=None)
    ap.add_argument("--player-radius", type=float, default=PLAYER_RADIUS)
    ap.add_argument("--player-height", type=float, default=PLAYER_HEIGHT)
    ap.add_argument("--occlusion-tolerance", type=float, default=PLAYER_OCCLUSION_TOLERANCE)
    ap.add_argument("--min-visible-pixels", type=int, default=PLAYER_MIN_VISIBLE_PIXELS)
    ap.add_argument("--player-z-offset", type=float, default=PLAYER_Z_OFFSET)
    ap.add_argument("--camera-yaw-offset", type=float, default=CAMERA_YAW_OFFSET)
    ap.add_argument("--camera-pitch-offset", type=float, default=CAMERA_PITCH_OFFSET)
    ap.add_argument("--player-mask-mode", choices=["capsule", "multipart"], default=PLAYER_MASK_MODE)
    ap.add_argument("--mesh-backend", choices=["cpu", "gpu", "auto", "bsp_faces_cpu", "bsp_faces_gpu"], default="cpu")
    ap.add_argument("--bsp-faces-npz", type=Path, default=None)
    args = ap.parse_args()

    cache = load_renderer_cache(args.match_dir, bsp_faces_npz=args.bsp_faces_npz)
    dense, mesh_depth_units, rgb, meta, dense_mod = render_memory_dense_condition(
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
        episode_memory=load_episode_memory(args.episode_memory_dir),
        player_radius=args.player_radius,
        player_height=args.player_height,
        occlusion_tolerance=args.occlusion_tolerance,
        min_visible_pixels=args.min_visible_pixels,
        player_z_offset=args.player_z_offset,
        camera_yaw_offset=args.camera_yaw_offset,
        camera_pitch_offset=args.camera_pitch_offset,
        player_mask_mode=args.player_mask_mode,
        mesh_backend=args.mesh_backend,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "mesh_dense_condition_v0.npz", dense=dense, mesh_depth_units=mesh_depth_units)
    (args.out_dir / "mesh_dense_condition_meta_v0.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    dense_mod.make_qa(
        args.out_dir / "mesh_dense_condition_qa_v0.png",
        rgb,
        dense[0],
        dense[1],
        dense[3:],
        meta,
    )
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "shape": list(dense.shape),
        "mesh_hit_ratio": meta["mesh_hit_ratio"],
        "memory_projected_players": len(meta["memory_projected_players"]),
        "channels": meta["channels"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
