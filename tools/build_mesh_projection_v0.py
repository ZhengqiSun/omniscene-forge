#!/usr/bin/env python3
"""Render a first-person mesh-depth QA image from the dataset world OBJ.

This is a geometry-alignment smoke test, not the final renderer. It projects
the map's `_world_` OBJ into one player's camera and rasterizes a low-res depth
buffer. The goal is to verify that mesh, nav/player JSON coordinates, yaw/pitch,
and first-person videos can share one Memory query path.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_world_obj(match_dir: Path) -> Path:
    manifest = load_json(match_dir / "mesh_manifest.json")
    world = next((m for m in manifest if m.get("model_name") == "_world_"), None)
    if not world:
        raise FileNotFoundError("mesh_manifest.json has no _world_ entry")
    path = match_dir / "meshes" / world["mesh_file"]
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idxs = []
                for tok in line.split()[1:]:
                    # OBJ is 1-indexed. Tokens may be v, v/vt, or v/vt/vn.
                    idxs.append(int(tok.split("/")[0]) - 1)
                if len(idxs) >= 3:
                    # Fan triangulate just in case, though this dataset is triangles.
                    for i in range(1, len(idxs) - 1):
                        faces.append([idxs[0], idxs[i], idxs[i + 1]])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def read_frame(path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise IndexError(f"Could not read frame {frame_index} from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def camera_basis(yaw_deg: float, pitch_deg: float, pitch_sign: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg) * pitch_sign
    forward = np.array([
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        -math.sin(pitch),
    ], dtype=np.float32)
    forward /= max(np.linalg.norm(forward), 1e-6)

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    right /= max(np.linalg.norm(right), 1e-6)
    up = np.cross(right, forward)
    up /= max(np.linalg.norm(up), 1e-6)
    return right, up, forward


def project_vertices(
    vertices: np.ndarray,
    cam_pos: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    pitch_sign: float,
    width: int,
    height: int,
    fov_x_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    cam = world_to_camera(vertices, cam_pos, yaw_deg, pitch_deg, pitch_sign)
    return project_camera_points(cam, width, height, fov_x_deg)


def world_to_camera(
    vertices: np.ndarray,
    cam_pos: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    pitch_sign: float,
) -> np.ndarray:
    right, up, forward = camera_basis(yaw_deg, pitch_deg, pitch_sign)
    rel = vertices - cam_pos.reshape(1, 3)
    x_cam = rel @ right
    y_cam = rel @ up
    z_cam = rel @ forward
    return np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float32)


def project_camera_points(
    camera_points: np.ndarray,
    width: int,
    height: int,
    fov_x_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_cam = camera_points[:, 0]
    y_cam = camera_points[:, 1]
    z_cam = camera_points[:, 2]
    tan_x = math.tan(math.radians(fov_x_deg) / 2.0)
    tan_y = tan_x * height / width
    u = width * 0.5 + (x_cam / np.maximum(z_cam, 1e-6)) / tan_x * width * 0.5
    v = height * 0.5 - (y_cam / np.maximum(z_cam, 1e-6)) / tan_y * height * 0.5
    return np.stack([u, v, z_cam], axis=1).astype(np.float32), z_cam


def project_vertices_with_camera(
    vertices: np.ndarray,
    cam_pos: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    pitch_sign: float,
    width: int,
    height: int,
    fov_x_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cam = world_to_camera(vertices, cam_pos, yaw_deg, pitch_deg, pitch_sign)
    projected, z_cam = project_camera_points(cam, width, height, fov_x_deg)
    return projected, z_cam, cam


def clip_polygon_z_min(poly: np.ndarray, z_min: float) -> np.ndarray:
    """Clip a camera-space polygon against z >= z_min."""
    if len(poly) == 0:
        return poly
    out: list[np.ndarray] = []
    prev = poly[-1]
    prev_inside = bool(prev[2] >= z_min)
    for cur in poly:
        cur_inside = bool(cur[2] >= z_min)
        if cur_inside != prev_inside:
            denom = float(cur[2] - prev[2])
            if abs(denom) > 1e-8:
                t = (z_min - float(prev[2])) / denom
                out.append((prev + t * (cur - prev)).astype(np.float32))
        if cur_inside:
            out.append(cur.astype(np.float32))
        prev = cur
        prev_inside = cur_inside
    if not out:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(out, axis=0).astype(np.float32)


def rasterize_one_projected_triangle(
    depth: np.ndarray,
    t: np.ndarray,
    near: float,
    far: float,
) -> int:
    height, width = depth.shape
    min_u = max(0, int(np.floor(np.min(t[:, 0]))))
    max_u = min(width - 1, int(np.ceil(np.max(t[:, 0]))))
    min_v = max(0, int(np.floor(np.min(t[:, 1]))))
    max_v = min(height - 1, int(np.ceil(np.max(t[:, 1]))))
    if min_u > max_u or min_v > max_v:
        return 0
    xs = np.arange(min_u, max_u + 1, dtype=np.float32) + 0.5
    ys = np.arange(min_v, max_v + 1, dtype=np.float32) + 0.5
    xx, yy = np.meshgrid(xs, ys)
    p = np.stack([xx, yy], axis=-1)
    a = t[0, :2]
    b = t[1, :2]
    c = t[2, :2]
    v0 = b - a
    v1 = c - a
    v2 = p - a
    den = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(float(den)) < 1e-6:
        return 0
    inv_den = 1.0 / den
    w1 = (v2[..., 0] * v1[1] - v1[0] * v2[..., 1]) * inv_den
    w2 = (v0[0] * v2[..., 1] - v2[..., 0] * v0[1]) * inv_den
    w0 = 1.0 - w1 - w2
    mask = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
    if not np.any(mask):
        return 0
    inv_z0 = 1.0 / max(float(t[0, 2]), 1e-6)
    inv_z1 = 1.0 / max(float(t[1, 2]), 1e-6)
    inv_z2 = 1.0 / max(float(t[2, 2]), 1e-6)
    inv_z = w0 * inv_z0 + w1 * inv_z1 + w2 * inv_z2
    zbuf = np.where(np.abs(inv_z) > 1e-8, 1.0 / inv_z, np.inf)
    region = depth[min_v : max_v + 1, min_u : max_u + 1]
    update = mask & (zbuf >= near) & (zbuf <= far) & (zbuf < region)
    if not np.any(update):
        return 0
    region[update] = zbuf[update]
    return int(update.sum())


def perspective_correct_z(w0: np.ndarray, w1: np.ndarray, w2: np.ndarray, zs: np.ndarray) -> np.ndarray:
    inv_z0 = 1.0 / max(float(zs[0]), 1e-6)
    inv_z1 = 1.0 / max(float(zs[1]), 1e-6)
    inv_z2 = 1.0 / max(float(zs[2]), 1e-6)
    inv_z = w0 * inv_z0 + w1 * inv_z1 + w2 * inv_z2
    return np.where(np.abs(inv_z) > 1e-8, 1.0 / inv_z, np.inf)


def rasterize_depth(
    projected: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
    near: float,
    far: float,
    max_triangles: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    tri = projected[faces]
    z = tri[:, :, 2]
    in_front = np.all((z > near) & (z < far), axis=1)
    tri = tri[in_front]

    min_u = np.floor(np.min(tri[:, :, 0], axis=1)).astype(np.int32)
    max_u = np.ceil(np.max(tri[:, :, 0], axis=1)).astype(np.int32)
    min_v = np.floor(np.min(tri[:, :, 1], axis=1)).astype(np.int32)
    max_v = np.ceil(np.max(tri[:, :, 1], axis=1)).astype(np.int32)
    intersects = (max_u >= 0) & (min_u < width) & (max_v >= 0) & (min_v < height)
    tri = tri[intersects]
    min_u = np.clip(min_u[intersects], 0, width - 1)
    max_u = np.clip(max_u[intersects], 0, width - 1)
    min_v = np.clip(min_v[intersects], 0, height - 1)
    max_v = np.clip(max_v[intersects], 0, height - 1)

    mean_z = tri[:, :, 2].mean(axis=1)
    if max_triangles > 0 and len(tri) > max_triangles:
        # Debug-only speed cap. Production dense export should leave
        # max_triangles=0 so far-but-visible geometry is not silently dropped.
        keep = np.argpartition(mean_z, max_triangles)[:max_triangles]
        tri = tri[keep]
        min_u = min_u[keep]
        max_u = max_u[keep]
        min_v = min_v[keep]
        max_v = max_v[keep]

    depth = np.full((height, width), np.inf, dtype=np.float32)
    painted = 0
    skipped_large = 0
    for t, x0, x1, y0, y1 in zip(tri, min_u, max_u, min_v, max_v):
        if x1 < x0 or y1 < y0:
            continue
        # Extremely large projected triangles are usually near-plane artifacts in
        # this unclipped v0. They are better skipped than allowed to dominate QA.
        if (x1 - x0 + 1) * (y1 - y0 + 1) > width * height * 0.6:
            skipped_large += 1
            continue
        xs = np.arange(x0, x1 + 1, dtype=np.float32) + 0.5
        ys = np.arange(y0, y1 + 1, dtype=np.float32) + 0.5
        xx, yy = np.meshgrid(xs, ys)
        p = np.stack([xx, yy], axis=-1)
        a = t[0, :2]
        b = t[1, :2]
        c = t[2, :2]
        v0 = b - a
        v1 = c - a
        v2 = p - a
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(float(den)) < 1e-6:
            continue
        inv_den = 1.0 / den
        w1 = (v2[..., 0] * v1[1] - v1[0] * v2[..., 1]) * inv_den
        w2 = (v0[0] * v2[..., 1] - v2[..., 0] * v0[1]) * inv_den
        w0 = 1.0 - w1 - w2
        mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not np.any(mask):
            continue
        zbuf = perspective_correct_z(w0, w1, w2, t[:, 2])
        region = depth[y0 : y1 + 1, x0 : x1 + 1]
        update = mask & (zbuf >= near) & (zbuf <= far) & (zbuf < region)
        if np.any(update):
            region[update] = zbuf[update]
            painted += int(update.sum())

    return depth, {
        "triangles_after_front_cull": int(in_front.sum()),
        "triangles_after_frustum_cull": int(len(mean_z)),
        "triangles_rasterized": int(len(tri)),
        "painted_pixels": int(np.isfinite(depth).sum()),
        "painted_updates": int(painted),
        "skipped_large_triangles": int(skipped_large),
    }


def rasterize_depth_clipped(
    camera_points: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
    fov_x_deg: float,
    near: float,
    far: float,
    max_triangles: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    tri_cam = camera_points[faces]
    z = tri_cam[:, :, 2]
    potentially_visible = (np.max(z, axis=1) >= near) & (np.min(z, axis=1) <= far)
    tri_cam = tri_cam[potentially_visible]
    input_triangles = int(len(tri_cam))
    truncated = 0
    if max_triangles > 0 and len(tri_cam) > max_triangles:
        # Debug-only speed cap. Keep disabled for training-data export.
        mean_z = np.mean(np.clip(tri_cam[:, :, 2], near, far), axis=1)
        keep = np.argpartition(mean_z, max_triangles)[:max_triangles]
        tri_cam = tri_cam[keep]
        truncated = int(input_triangles - len(tri_cam))

    depth = np.full((height, width), np.inf, dtype=np.float32)
    painted = 0
    clipped_triangles = 0
    rasterized = 0
    for tri in tri_cam:
        poly = clip_polygon_z_min(tri, near)
        if len(poly) < 3:
            continue
        if len(poly) != 3:
            clipped_triangles += 1
        projected, _ = project_camera_points(poly, width, height, fov_x_deg)
        for idx in range(1, len(projected) - 1):
            t = np.stack([projected[0], projected[idx], projected[idx + 1]], axis=0)
            count = rasterize_one_projected_triangle(depth, t, near, far)
            if count > 0:
                rasterized += 1
                painted += count

    return depth, {
        "triangles_after_front_cull": int(potentially_visible.sum()),
        "triangles_after_frustum_cull": int(len(tri_cam)),
        "triangles_before_max_limit": input_triangles,
        "triangles_dropped_by_max_limit": truncated,
        "triangles_rasterized": int(rasterized),
        "painted_pixels": int(np.isfinite(depth).sum()),
        "painted_updates": int(painted),
        "skipped_large_triangles": 0,
        "near_clipped_triangles": int(clipped_triangles),
    }


def normalize_depth(depth: np.ndarray, far: float) -> np.ndarray:
    out = np.zeros_like(depth, dtype=np.float32)
    hit = np.isfinite(depth)
    out[hit] = np.clip(depth[hit] / far, 0.0, 1.0)
    return out


def gray(ch: np.ndarray) -> Image.Image:
    arr = np.clip(ch * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def make_qa(
    out_path: Path,
    rgb: np.ndarray,
    dataset_depth_rgb: np.ndarray | None,
    mesh_depth: np.ndarray,
    far: float,
    meta: dict[str, Any],
) -> None:
    h, w = mesh_depth.shape
    rgb_small = Image.fromarray(rgb).resize((w, h), Image.Resampling.BILINEAR)
    mesh_norm = normalize_depth(mesh_depth, far)
    mesh_img = gray(mesh_norm)
    hit = np.isfinite(mesh_depth)
    hit_img = Image.fromarray((hit.astype(np.uint8) * 255)).convert("RGB")

    overlay = np.asarray(rgb_small).astype(np.float32)
    color = np.zeros_like(overlay)
    color[..., 1] = 255
    color[..., 2] = 80
    overlay[hit] = overlay[hit] * 0.45 + color[hit] * 0.55
    overlay_img = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))

    panels: list[tuple[str, Image.Image]] = [("rgb", rgb_small), ("mesh depth", mesh_img), ("mesh hit overlay", overlay_img)]
    if dataset_depth_rgb is not None:
        ds = Image.fromarray(dataset_depth_rgb).resize((w, h), Image.Resampling.BILINEAR)
        panels.append(("dataset depth stream", ds))
        panels.append(("mesh hit mask", hit_img))
    else:
        panels.append(("mesh hit mask", hit_img))

    pad = 26
    cols = 3
    rows = math.ceil(len(panels) / cols)
    canvas = Image.new("RGB", (cols * w, rows * (h + pad)), (250, 250, 247))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, img) in enumerate(panels):
        x = (idx % cols) * w
        y = (idx // cols) * (h + pad)
        canvas.paste(img, (x, y + pad))
        draw.text((x + 8, y + 6), label, fill=(20, 20, 20))
    draw.text((8, canvas.height - 18), f"hit={meta['hit_ratio']:.3f}, yaw={meta['yaw']:.1f}, pitch={meta['pitch']:.1f}", fill=(20, 20, 20))
    canvas.save(out_path)


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
    ap.add_argument("--max-triangles", type=int, default=0, help="0 keeps all visible triangles; positive values are a speed/debug cap.")
    args = ap.parse_args()

    obj_path = find_world_obj(args.match_dir)
    vertices, faces = load_obj_mesh(obj_path)

    episode_dir = args.match_dir / "train" / args.episode
    frames = load_json(episode_dir / f"{args.ego_stem}.json")
    frame = frames[args.frame_index]
    cam_pos = np.asarray(frame.get("camera_position") or [frame["x"], frame["y"], frame["z"] + 64.0], dtype=np.float32)
    yaw = float(frame.get("yaw", frame.get("camera_rotation", [0, 0, 0])[2]))
    pitch = float(frame.get("pitch", frame.get("camera_rotation", [0, 0, 0])[1]))

    projected, z_cam, camera_points = project_vertices_with_camera(
        vertices, cam_pos, yaw, pitch, args.pitch_sign, args.width, args.height, args.fov_x
    )
    depth, stats = rasterize_depth_clipped(
        camera_points, faces, args.width, args.height, args.fov_x, args.near, args.far, args.max_triangles
    )

    rgb = read_frame(episode_dir / f"{args.ego_stem}.mp4", args.frame_index)
    dataset_depth = None
    depth_path = episode_dir / f"{args.ego_stem}_depth.mkv"
    if depth_path.exists():
        dataset_depth = read_frame(depth_path, args.frame_index)

    hit_ratio = float(np.isfinite(depth).mean())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "mesh_projection_depth_v0.npz", depth=depth, hit=np.isfinite(depth))

    meta = {
        "match_dir": str(args.match_dir),
        "obj_path": str(obj_path),
        "episode": args.episode,
        "ego_stem": args.ego_stem,
        "frame_index": args.frame_index,
        "camera_position": cam_pos.astype(float).tolist(),
        "yaw": yaw,
        "pitch": pitch,
        "pitch_sign": args.pitch_sign,
        "fov_x": args.fov_x,
        "near": args.near,
        "far": args.far,
        "width": args.width,
        "height": args.height,
        "mesh_vertices": int(len(vertices)),
        "mesh_faces": int(len(faces)),
        "vertex_camera_z_minmax": [float(np.min(z_cam)), float(np.max(z_cam))],
        "hit_ratio": hit_ratio,
        "stats": stats,
        "channels": ["mesh_env_depth_units", "mesh_hit_mask"],
        "note": "v0 mesh projection/rasterization for coordinate QA; final renderer should do clipped ray casting with semantic surfaces.",
    }
    (args.out_dir / "mesh_projection_meta_v0.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    make_qa(args.out_dir / "mesh_projection_qa_v0.png", rgb, dataset_depth, depth, args.far, meta)

    print(json.dumps({
        "out_dir": str(args.out_dir),
        "depth_shape": list(depth.shape),
        "hit_ratio": hit_ratio,
        "mesh_vertices": int(len(vertices)),
        "mesh_faces": int(len(faces)),
        "stats": stats,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
