#!/usr/bin/env python3
"""Runtime-only MultiGen history guidance patch for LingBot WanI2V."""
from __future__ import annotations

import gc
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from einops import rearrange
from tqdm import tqdm


def install_history_guidance(pipe: Any) -> None:
    """Patch one WanI2V instance so CFG-uncond receives noised context frames.

    Extra generate kwargs are ``history_guidance``,
    ``history_guidance_noise_std`` (Gaussian std in normalized RGB [-1, 1]), and
    ``history_guidance_seed_offset``. The default std=0.5 is an implementation
    assumption because the task-provided MultiGen quote specifies the branch
    asymmetry but not an exact numeric noise scale.
    """
    original_generate = pipe.generate

    def generate_with_history_guidance(
        self: Any,
        input_prompt: str,
        img: Any,
        action_path: str | None = None,
        allow_act2cam: bool = False,
        action_string: str | None = None,
        vis_ui: bool = True,
        max_area: int = 720 * 1280,
        frame_num: int = 81,
        shift: float = 5.0,
        sample_solver: str = "unipc",
        sampling_steps: int = 40,
        guide_scale: float | tuple[float, float] = 5.0,
        n_prompt: str = "",
        seed: int = -1,
        offload_model: bool = True,
        history_guidance: bool = False,
        history_guidance_noise_std: float = 0.5,
        history_guidance_seed_offset: int = 9176,
    ):
        if not history_guidance or float(history_guidance_noise_std) <= 0.0:
            return original_generate(
                input_prompt, img,
                action_path=action_path,
                allow_act2cam=allow_act2cam,
                action_string=action_string,
                vis_ui=vis_ui,
                max_area=max_area,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                n_prompt=n_prompt,
                seed=seed,
                offload_model=offload_model,
            )

        from wan import image2video as base

        if allow_act2cam or action_string is not None or vis_ui:
            raise NotImplementedError("history guidance wrapper supports the AR sampler path only")

        if action_path is not None:
            c2ws = np.load(os.path.join(action_path, "poses.npy"))
            len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
            frame_num = min(frame_num, len_c2ws)
            c2ws = c2ws[:frame_num]
            if self.control_type == "act":
                wasd_action = np.load(os.path.join(action_path, "action.npy"))[:frame_num]
            else:
                wasd_action = None

        guide_scale = (guide_scale, guide_scale) if isinstance(guide_scale, float) else guide_scale
        img_clean = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img_clean.shape[1:]
        aspect_ratio = h / w
        lat_h = round(np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] // self.patch_size[1] * self.patch_size[1])
        lat_w = round(np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] // self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        lat_f = (F - 1) // self.vae_stride[0] + 1
        max_seq_len = lat_f * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(16, lat_f, lat_h, lat_w, dtype=torch.float32, generator=seed_g, device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device("cpu"))
            context_null = self.text_encoder([n_prompt], torch.device("cpu"))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        dit_cond_dict = None
        if action_path is not None:
            Ks = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()
            Ks = base.get_Ks_transformed(Ks, height_org=480, width_org=832, height_resize=h, width_resize=w, height_final=h, width_final=w)
            Ks = Ks[0]
            len_c2ws = len(c2ws)
            c2ws_infer = base.interpolate_camera_poses(
                src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
                src_rot_mat=c2ws[:, :3, :3],
                src_trans_vec=c2ws[:, :3, 3],
                tgt_indices=np.linspace(0, len_c2ws - 1, int((len_c2ws - 1) // 4) + 1),
            )
            c2ws_infer = base.compute_relative_poses(c2ws_infer, framewise=True)
            Ks = Ks.repeat(len(c2ws_infer), 1)
            c2ws_infer = c2ws_infer.to(self.device)
            Ks = Ks.to(self.device)
            if self.control_type == "act":
                wasd_action = torch.from_numpy(wasd_action[::4]).float().to(self.device)
            else:
                wasd_action = None
            only_rays_d = wasd_action is not None
            c2ws_plucker_emb = base.get_plucker_embeddings(c2ws_infer, Ks, h, w, only_rays_d=only_rays_d)
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb,
                "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
                c1=int(h // lat_h), c2=int(w // lat_w),
            )
            c2ws_plucker_emb = c2ws_plucker_emb[None, ...]
            c2ws_plucker_emb = rearrange(c2ws_plucker_emb, "b (f h w) c -> b c f h w", f=lat_f, h=lat_h, w=lat_w).to(self.param_dtype)
            if wasd_action is not None:
                wasd_action_tensor = wasd_action[:, None, None, :].repeat(1, h, w, 1)
                wasd_action_tensor = rearrange(
                    wasd_action_tensor,
                    "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
                    c1=int(h // lat_h), c2=int(w // lat_w),
                )
                wasd_action_tensor = wasd_action_tensor[None, ...]
                wasd_action_tensor = rearrange(wasd_action_tensor, "b (f h w) c -> b c f h w", f=lat_f, h=lat_h, w=lat_w).to(self.param_dtype)
                c2ws_plucker_emb = torch.cat([c2ws_plucker_emb, wasd_action_tensor], dim=1)
            dit_cond_dict = {"c2ws_plucker_emb": c2ws_plucker_emb.chunk(1, dim=0)}

        hg_g = torch.Generator(device=self.device)
        hg_g.manual_seed(int(seed) + int(history_guidance_seed_offset))
        img_null = (img_clean + float(history_guidance_noise_std) * torch.randn(
            img_clean.shape, generator=hg_g, device=img_clean.device, dtype=img_clean.dtype)).clamp(-1.0, 1.0)

        def encode_context(image_tensor: torch.Tensor) -> torch.Tensor:
            encoded = self.vae.encode([
                torch.concat([
                    torch.nn.functional.interpolate(image_tensor[None].cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
                    torch.zeros(3, F - 1, h, w),
                ], dim=1).to(self.device)
            ])[0]
            return torch.concat([msk, encoded])

        y_clean = encode_context(img_clean)
        y_null = encode_context(img_null)

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low_noise = getattr(self.low_noise_model, "no_sync", noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, "no_sync", noop_no_sync)

        with (amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync_low_noise(), no_sync_high_noise()):
            boundary = self.boundary * self.num_train_timesteps
            if sample_solver == "unipc":
                sample_scheduler = base.FlowUniPCMultistepScheduler(num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == "dpm++":
                sample_scheduler = base.FlowDPMSolverMultistepScheduler(num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False)
                sampling_sigmas = base.get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = base.retrieve_timesteps(sample_scheduler, device=self.device, sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            latent = noise
            arg_c = {"context": [context[0]], "seq_len": max_seq_len, "y": [y_clean], "dit_cond_dict": dit_cond_dict}
            arg_null = {"context": context_null, "seq_len": max_seq_len, "y": [y_null], "dit_cond_dict": dit_cond_dict}

            if offload_model:
                torch.cuda.empty_cache()

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = torch.stack([t]).to(self.device)
                model = self._prepare_model_for_timestep(t, boundary, offload_model)
                sample_guide_scale = guide_scale[1] if t.item() >= boundary else guide_scale[0]
                noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + sample_guide_scale * (noise_pred_cond - noise_pred_uncond)
                temp_x0 = sample_scheduler.step(noise_pred.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False, generator=seed_g)[0]
                latent = temp_x0.squeeze(0)
                x0 = [latent]
                del latent_model_input, timestep

            if offload_model:
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latent, x0
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        return videos[0] if self.rank == 0 else None

    pipe.generate = types.MethodType(generate_with_history_guidance, pipe)
