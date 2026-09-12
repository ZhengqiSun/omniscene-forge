from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .schema import LingBotSampleV1


def _to_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    rgb = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames])
    return torch.from_numpy(rgb).permute(3, 0, 1, 2).float().div_(127.5).sub_(1.0)


def decode_video(path: Path, indices: list[int] | None = None) -> torch.Tensor:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot decode {path}")
    frames: list[np.ndarray] = []
    if indices is None:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    else:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError(f"failed decoding frame {index} from {path}")
            frames.append(frame)
    cap.release()
    if len(frames) != 81:
        raise RuntimeError(f"decoded {len(frames)} frames from {path}, expected 81")
    if any(frame.shape[:2] != (480, 832) for frame in frames):
        raise RuntimeError(f"video frames from {path} are not 832x480")
    return _to_tensor(frames)


def load_initial_image(sample: LingBotSampleV1) -> torch.Tensor:
    if sample.initial_image:
        frame = cv2.imread(str(sample.initial_image), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"cannot read {sample.initial_image}")
        return _to_tensor([frame])[:, 0]
    cap = cv2.VideoCapture(str(sample.raw_video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot decode {sample.raw_video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, sample.raw_start)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed decoding initial frame {sample.raw_start} from {sample.raw_video}")
    if frame.shape[:2] != (480, 832):
        raise RuntimeError(f"initial frame from {sample.raw_video} is not 832x480")
    return _to_tensor([frame])[:, 0]


def load_target_video(sample: LingBotSampleV1) -> torch.Tensor:
    if sample.target_video:
        return decode_video(sample.target_video)
    if sample.raw_video:
        indices = [sample.raw_start + i * sample.raw_stride for i in range(81)]
        return decode_video(sample.raw_video, indices)
    raise RuntimeError(f"{sample.sample_id}: no pixel target")


@torch.no_grad()
def encode_video_and_condition(vae, video: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    video = video.to(device)
    latent = vae.encode([video])[0]
    first_only = torch.zeros_like(video)
    first_only[:, :1] = video[:, :1]
    condition_latent = vae.encode([first_only])[0]
    mask = torch.ones(1, 81, 60, 104, device=device)
    mask[:, 1:] = 0
    mask = torch.cat([mask[:, :1].repeat_interleave(4, dim=1), mask[:, 1:]], dim=1)
    mask = mask.view(1, 21, 4, 60, 104).transpose(1, 2)[0]
    return latent, torch.cat([mask.to(condition_latent.dtype), condition_latent])


@torch.no_grad()
def encode_initial_condition(vae, image: torch.Tensor, device: torch.device) -> torch.Tensor:
    first_only = torch.zeros(3, 81, 480, 832, dtype=image.dtype, device=device)
    first_only[:, 0] = image.to(device)
    condition_latent = vae.encode([first_only])[0]
    mask = torch.ones(1, 81, 60, 104, device=device)
    mask[:, 1:] = 0
    mask = torch.cat([mask[:, :1].repeat_interleave(4, dim=1), mask[:, 1:]], dim=1)
    mask = mask.view(1, 21, 4, 60, 104).transpose(1, 2)[0]
    return torch.cat([mask.to(condition_latent.dtype), condition_latent])
