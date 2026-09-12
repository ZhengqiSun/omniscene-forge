from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange


def add_lingbot_import(lingbot_repo: Path) -> None:
    value = str(lingbot_repo.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def prepare_camera(poses_path: Path, intrinsics_path: Path, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    from wan.utils.cam_utils import compute_relative_poses, get_Ks_transformed, get_plucker_embeddings, interpolate_camera_poses

    c2ws = np.load(poses_path).astype("float32")
    intrinsics = torch.from_numpy(np.load(intrinsics_path).astype("float32"))
    c2ws_latent = interpolate_camera_poses(
        src_indices=np.linspace(0, 80, 81), src_rot_mat=c2ws[:, :3, :3],
        src_trans_vec=c2ws[:, :3, 3], tgt_indices=np.linspace(0, 80, 21),
    )
    c2ws_latent = compute_relative_poses(c2ws_latent, framewise=True).to(device)
    # Match official WanI2V: use the first calibration for every latent frame.
    ks = intrinsics[0].repeat(21, 1)
    ks = get_Ks_transformed(
        ks, height_org=480, width_org=832, height_resize=480, width_resize=832,
        height_final=480, width_final=832,
    ).to(device)
    rays = get_plucker_embeddings(c2ws_latent, ks, 480, 832, only_rays_d=False)
    rays = rearrange(rays, "f (h c1) (w c2) c -> (f h w) (c c1 c2)", c1=8, c2=8)
    rays = rearrange(rays[None], "b (f h w) c -> b c f h w", f=21, h=60, w=104)
    return rays.to(dtype)
