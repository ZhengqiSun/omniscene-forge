#!/usr/bin/env python3
"""CUDA depth rasterizer for Map Memory world meshes.

This module is intentionally a narrow backend: it produces the same
``mesh_depth_units`` contract as ``build_mesh_projection_v0.rasterize_depth``
without the Python per-triangle raster loop.  It keeps the public renderer
stable while moving the expensive environment mesh pass onto the GPU.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import cupy as cp
except Exception as exc:  # pragma: no cover - exercised on non-CUDA machines.
    cp = None
    _CUPY_IMPORT_ERROR = exc
else:
    _CUPY_IMPORT_ERROR = None


_DEPTH_KERNEL_SRC = r"""
extern "C" __global__
void rasterize_depth_kernel(
    const float* __restrict__ tri,
    const int ntri,
    const int width,
    const int height,
    const float near_z,
    const float far_z,
    const int skip_large,
    float* __restrict__ depth,
    int* __restrict__ counters
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= ntri) {
        return;
    }

    const float* t = tri + tid * 9;
    float x0 = t[0], y0 = t[1], z0 = t[2];
    float x1 = t[3], y1 = t[4], z1 = t[5];
    float x2 = t[6], y2 = t[7], z2 = t[8];

    if (!(isfinite(x0) && isfinite(y0) && isfinite(z0) &&
          isfinite(x1) && isfinite(y1) && isfinite(z1) &&
          isfinite(x2) && isfinite(y2) && isfinite(z2))) {
        return;
    }
    if (!(z0 > near_z && z0 < far_z && z1 > near_z && z1 < far_z && z2 > near_z && z2 < far_z)) {
        return;
    }

    float minxf = fminf(x0, fminf(x1, x2));
    float maxxf = fmaxf(x0, fmaxf(x1, x2));
    float minyf = fminf(y0, fminf(y1, y2));
    float maxyf = fmaxf(y0, fmaxf(y1, y2));
    if (maxxf < 0.0f || minxf >= (float)width || maxyf < 0.0f || minyf >= (float)height) {
        return;
    }

    int minx = max(0, (int)floorf(minxf));
    int maxx = min(width - 1, (int)ceilf(maxxf));
    int miny = max(0, (int)floorf(minyf));
    int maxy = min(height - 1, (int)ceilf(maxyf));
    int bbox_area = (maxx - minx + 1) * (maxy - miny + 1);
    if (bbox_area <= 0) {
        return;
    }
    if (skip_large && bbox_area > (int)((float)(width * height) * 0.6f)) {
        atomicAdd(&counters[1], 1);
        return;
    }

    float den = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    if (fabsf(den) < 1.0e-6f) {
        return;
    }
    float inv_den = 1.0f / den;
    float inv_z0 = 1.0f / fmaxf(z0, 1.0e-6f);
    float inv_z1 = 1.0f / fmaxf(z1, 1.0e-6f);
    float inv_z2 = 1.0f / fmaxf(z2, 1.0e-6f);
    int local_updates = 0;

    for (int py = miny; py <= maxy; ++py) {
        float yy = (float)py + 0.5f;
        for (int px = minx; px <= maxx; ++px) {
            float xx = (float)px + 0.5f;
            float w1 = ((xx - x0) * (y2 - y0) - (x2 - x0) * (yy - y0)) * inv_den;
            float w2 = ((x1 - x0) * (yy - y0) - (xx - x0) * (y1 - y0)) * inv_den;
            float w0 = 1.0f - w1 - w2;
            if (w0 < -1.0e-5f || w1 < -1.0e-5f || w2 < -1.0e-5f) {
                continue;
            }

            // Perspective-correct camera z.  The old CPU v0 path linearly
            // interpolated z in screen space, which is wrong for slanted faces.
            float inv_z = w0 * inv_z0 + w1 * inv_z1 + w2 * inv_z2;
            if (fabsf(inv_z) < 1.0e-8f) {
                continue;
            }
            float z = 1.0f / inv_z;
            if (z < near_z || z > far_z) {
                continue;
            }

            int offset = py * width + px;
            int* depth_i = (int*)(depth + offset);
            int old_i = *depth_i;
            int new_i = __float_as_int(z);
            while (z < __int_as_float(old_i)) {
                int prev_i = atomicCAS(depth_i, old_i, new_i);
                if (prev_i == old_i) {
                    local_updates += 1;
                    break;
                }
                old_i = prev_i;
            }
        }
    }

    if (local_updates > 0) {
        atomicAdd(&counters[0], 1);
        atomicAdd(&counters[2], local_updates);
    }
}
"""


class GpuMeshDepthRenderer:
    """Reusable GPU cache for one static mesh."""

    def __init__(self, vertices_gpu: Any, faces_gpu: Any) -> None:
        self.vertices_gpu = vertices_gpu
        self.faces_gpu = faces_gpu

    @classmethod
    def from_numpy(cls, vertices: np.ndarray, faces: np.ndarray) -> "GpuMeshDepthRenderer":
        require_cupy()
        return cls(
            vertices_gpu=cp.asarray(vertices, dtype=cp.float32),
            faces_gpu=cp.asarray(faces, dtype=cp.int32),
        )


def require_cupy() -> None:
    if cp is None:
        raise RuntimeError(
            "cupy is required for GPU mesh rasterization. Install cupy-cuda12x "
            f"for this server. Original import error: {_CUPY_IMPORT_ERROR!r}"
        )


def camera_basis_np(yaw_deg: float, pitch_deg: float, pitch_sign: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg) * pitch_sign
    forward = np.array(
        [
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            -math.sin(pitch),
        ],
        dtype=np.float32,
    )
    forward /= max(float(np.linalg.norm(forward)), 1e-6)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    right /= max(float(np.linalg.norm(right)), 1e-6)
    up = np.cross(right, forward)
    up /= max(float(np.linalg.norm(up)), 1e-6)
    return right.astype(np.float32), up.astype(np.float32), forward.astype(np.float32)


def _prepare_projected_triangles_gpu(
    renderer: GpuMeshDepthRenderer,
    cam_pos: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    pitch_sign: float,
    width: int,
    height: int,
    fov_x_deg: float,
    near: float,
    far: float,
    max_triangles: int,
) -> tuple[Any, dict[str, Any]]:
    right, up, forward = camera_basis_np(yaw_deg, pitch_deg, pitch_sign)
    cam_pos_gpu = cp.asarray(cam_pos.reshape(1, 3), dtype=cp.float32)
    basis_gpu = cp.asarray(np.stack([right, up, forward], axis=1), dtype=cp.float32)
    cam = (renderer.vertices_gpu - cam_pos_gpu) @ basis_gpu
    tri_cam = cam[renderer.faces_gpu]
    z = tri_cam[:, :, 2]
    in_front = cp.all((z > near) & (z < far), axis=1)
    tri_cam = tri_cam[in_front]
    triangles_after_front = int(in_front.sum().get())
    if tri_cam.shape[0] == 0:
        return cp.empty((0, 3, 3), dtype=cp.float32), {
            "triangles_after_front_cull": 0,
            "triangles_after_frustum_cull": 0,
            "triangles_rasterized": 0,
            "triangles_before_max_limit": 0,
            "triangles_dropped_by_max_limit": 0,
        }

    tan_x = math.tan(math.radians(fov_x_deg) / 2.0)
    tan_y = tan_x * height / width
    x = tri_cam[:, :, 0]
    y = tri_cam[:, :, 1]
    z = tri_cam[:, :, 2]
    u = width * 0.5 + (x / cp.maximum(z, 1e-6)) / tan_x * width * 0.5
    v = height * 0.5 - (y / cp.maximum(z, 1e-6)) / tan_y * height * 0.5
    tri = cp.stack([u, v, z], axis=2).astype(cp.float32)

    min_u = cp.floor(cp.min(tri[:, :, 0], axis=1))
    max_u = cp.ceil(cp.max(tri[:, :, 0], axis=1))
    min_v = cp.floor(cp.min(tri[:, :, 1], axis=1))
    max_v = cp.ceil(cp.max(tri[:, :, 1], axis=1))
    intersects = (max_u >= 0) & (min_u < width) & (max_v >= 0) & (min_v < height)
    tri = tri[intersects]
    triangles_after_frustum = int(intersects.sum().get())
    before_limit = int(tri.shape[0])
    dropped = 0

    if max_triangles > 0 and tri.shape[0] > max_triangles:
        mean_z = tri[:, :, 2].mean(axis=1)
        keep = cp.argpartition(mean_z, max_triangles)[:max_triangles]
        tri = tri[keep]
        dropped = before_limit - int(tri.shape[0])

    stats = {
        "triangles_after_front_cull": triangles_after_front,
        "triangles_after_frustum_cull": triangles_after_frustum,
        "triangles_before_max_limit": before_limit,
        "triangles_dropped_by_max_limit": dropped,
        "triangles_rasterized": int(tri.shape[0]),
    }
    return cp.ascontiguousarray(tri, dtype=cp.float32), stats


def rasterize_depth_gpu(
    renderer: GpuMeshDepthRenderer,
    cam_pos: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    pitch_sign: float,
    width: int,
    height: int,
    fov_x_deg: float,
    near: float,
    far: float,
    max_triangles: int = 0,
    skip_large_triangles: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render camera-space z depth on GPU and return a NumPy depth buffer."""

    require_cupy()
    tri, stats = _prepare_projected_triangles_gpu(
        renderer,
        np.asarray(cam_pos, dtype=np.float32),
        yaw_deg,
        pitch_deg,
        pitch_sign,
        width,
        height,
        fov_x_deg,
        near,
        far,
        max_triangles,
    )
    depth = cp.full((height, width), cp.inf, dtype=cp.float32)
    counters = cp.zeros((3,), dtype=cp.int32)
    if tri.shape[0] > 0:
        kernel = cp.RawKernel(_DEPTH_KERNEL_SRC, "rasterize_depth_kernel")
        block = 128
        grid = (int((tri.shape[0] + block - 1) // block),)
        kernel(
            grid,
            (block,),
            (
                tri.ravel(),
                np.int32(tri.shape[0]),
                np.int32(width),
                np.int32(height),
                np.float32(near),
                np.float32(far),
                np.int32(1 if skip_large_triangles else 0),
                depth,
                counters,
            ),
        )
    cp.cuda.Stream.null.synchronize()
    counters_np = cp.asnumpy(counters)
    depth_np = cp.asnumpy(depth)
    stats.update(
        {
            "triangles_drawn": int(counters_np[0]),
            "painted_pixels": int(np.isfinite(depth_np).sum()),
            "painted_updates": int(counters_np[2]),
            "skipped_large_triangles": int(counters_np[1]),
            "backend": "cupy_raw_cuda_perspective_depth_v0",
        }
    )
    return depth_np, stats
