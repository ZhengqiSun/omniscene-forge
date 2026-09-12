from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

SCHEMA_ID = "lingbot_sample_v1"
ALLOWED_SPLITS = {"train", "validation", "val", "test"}
FORBIDDEN_KEYS = {
    "dense", "dense_cache", "dense_path", "state", "state_cache",
    "player_mask", "teacher_player_mask", "world_events", "world_event",
    "memory", "memory_adapter", "map_memory_sample_ids",
}
ALLOWED_KEYS = {
    "schema_id", "sample_id", "split", "initial_image", "raw_video",
    "raw_start", "raw_stride", "target_video", "target_latent", "poses",
    "intrinsics", "prompt", "text_cache", "num_frames", "fps", "height",
    "width", "metadata",
}


class ManifestError(ValueError):
    pass


def _path(value: Any, root: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


@dataclass(frozen=True)
class LingBotSampleV1:
    sample_id: str
    split: str
    poses: Path
    intrinsics: Path
    num_frames: int
    fps: float
    height: int
    width: int
    initial_image: Path | None = None
    raw_video: Path | None = None
    raw_start: int | None = None
    raw_stride: int | None = None
    target_video: Path | None = None
    target_latent: Path | None = None
    prompt: str | None = None
    text_cache: Path | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any], root: Path, *, strict: bool = True) -> "LingBotSampleV1":
        keys = set(row)
        forbidden = sorted(keys & FORBIDDEN_KEYS)
        if forbidden:
            raise ManifestError(f"camera-only schema rejects privileged fields: {forbidden}")
        unknown = sorted(keys - ALLOWED_KEYS)
        if strict and unknown:
            raise ManifestError(f"unknown LingBotSampleV1 fields: {unknown}")
        if row.get("schema_id", SCHEMA_ID) != SCHEMA_ID:
            raise ManifestError(f"schema_id must be {SCHEMA_ID!r}")
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise ManifestError("sample_id is required")
        split = str(row.get("split", "")).lower()
        if split not in ALLOWED_SPLITS:
            raise ManifestError(f"invalid split {split!r}")
        initial = _path(row.get("initial_image"), root)
        raw = _path(row.get("raw_video"), root)
        if (initial is None) == (raw is None):
            raise ManifestError("exactly one of initial_image or raw_video is required")
        raw_start = row.get("raw_start")
        raw_stride = row.get("raw_stride")
        if raw is not None:
            if raw_start is None or raw_stride is None:
                raise ManifestError("raw_video requires raw_start and raw_stride")
            if int(raw_start) < 0 or int(raw_stride) < 1:
                raise ManifestError("raw_start must be >=0 and raw_stride >=1")
        elif raw_start is not None or raw_stride is not None:
            raise ManifestError("raw_start/raw_stride require raw_video")
        prompt = row.get("prompt")
        text_cache = _path(row.get("text_cache"), root)
        if prompt is not None and not str(prompt).strip():
            raise ManifestError("prompt cannot be empty")
        poses = _path(row.get("poses"), root)
        intrinsics = _path(row.get("intrinsics"), root)
        if poses is None or intrinsics is None:
            raise ManifestError("poses and intrinsics are required")
        return cls(
            sample_id=sample_id, split=split, poses=poses, intrinsics=intrinsics,
            num_frames=int(row.get("num_frames", 81)), fps=float(row.get("fps", 16)),
            height=int(row.get("height", 480)), width=int(row.get("width", 832)),
            initial_image=initial, raw_video=raw,
            raw_start=None if raw_start is None else int(raw_start),
            raw_stride=None if raw_stride is None else int(raw_stride),
            target_video=_path(row.get("target_video"), root),
            target_latent=_path(row.get("target_latent"), root),
            prompt=None if prompt is None else str(prompt), text_cache=text_cache,
            metadata=row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
        )

    def condition_paths(self) -> Iterable[Path]:
        yield self.poses
        yield self.intrinsics
        if self.initial_image:
            yield self.initial_image
        if self.raw_video:
            yield self.raw_video

    def training_paths(self) -> Iterable[Path]:
        yield from self.condition_paths()
        for path in (self.target_video, self.target_latent, self.text_cache):
            if path:
                yield path


def manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path, *, strict: bool = True) -> list[LingBotSampleV1]:
    root = path.resolve().parent
    samples: list[LingBotSampleV1] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                sample = LingBotSampleV1.from_row(row, root, strict=strict)
            except Exception as exc:
                raise ManifestError(f"{path}:{line_no}: {exc}") from exc
            if sample.sample_id in seen:
                raise ManifestError(f"{path}:{line_no}: duplicate sample_id {sample.sample_id!r}")
            seen.add(sample.sample_id)
            samples.append(sample)
    if not samples:
        raise ManifestError(f"empty manifest: {path}")
    return samples


def _video_info(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ManifestError(f"undecodable video: {path}")
    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def validate_sample(sample: LingBotSampleV1, *, mode: str, deep: bool = False) -> list[str]:
    errors: list[str] = []
    if (sample.num_frames, sample.fps, sample.height, sample.width) != (81, 16.0, 480, 832):
        errors.append("required output contract is 81 frames, 832x480, 16 FPS")
    paths = sample.condition_paths() if mode == "infer" else sample.training_paths()
    for path in paths:
        if not path.is_file():
            errors.append(f"missing file: {path}")
    if errors:
        return errors
    try:
        poses = np.load(sample.poses, mmap_mode="r")
        if poses.shape != (81, 4, 4):
            errors.append(f"poses shape {poses.shape} != (81,4,4)")
        elif not np.isfinite(poses).all():
            errors.append("poses contain NaN/Inf")
        else:
            if not np.allclose(poses[:, 3, :], np.array([0, 0, 0, 1]), atol=1e-4):
                errors.append("poses are not homogeneous camera-to-world matrices")
            rotations = np.asarray(poses[:, :3, :3])
            orth = rotations @ np.swapaxes(rotations, 1, 2)
            if not np.allclose(orth, np.eye(3), atol=2e-2):
                errors.append("pose rotations are not approximately orthonormal")
        intr = np.load(sample.intrinsics, mmap_mode="r")
        if intr.shape != (81, 4):
            errors.append(f"intrinsics shape {intr.shape} != (81,4)")
        elif not np.isfinite(intr).all():
            errors.append("intrinsics contain NaN/Inf")
        elif not np.allclose(intr, intr[0:1], rtol=1e-4, atol=1e-3):
            errors.append("intrinsics vary over time; official LingBot uses row 0 for all latent frames")
    except Exception as exc:
        errors.append(f"camera metadata load failed: {exc}")
    if sample.initial_image:
        image = cv2.imread(str(sample.initial_image), cv2.IMREAD_COLOR)
        if image is None:
            errors.append(f"undecodable initial image: {sample.initial_image}")
        elif image.shape[:2] != (480, 832):
            errors.append(f"initial image shape {image.shape[:2]} != (480,832)")
    if sample.raw_video:
        try:
            info = _video_info(sample.raw_video)
            if (info["height"], info["width"]) != (480, 832):
                errors.append(f"raw video resolution {info['width']}x{info['height']} != 832x480")
            last = int(sample.raw_start) + 80 * int(sample.raw_stride)
            if last >= info["frames"]:
                errors.append(f"raw window ends at frame {last}, video has {info['frames']} frames")
            effective_fps = info["fps"] / int(sample.raw_stride)
            if abs(effective_fps - 16.0) > 0.05:
                errors.append(f"raw effective FPS {effective_fps:.4f} != 16")
        except Exception as exc:
            errors.append(str(exc))
    if mode == "train":
        if sample.target_video is None and sample.target_latent is None and sample.raw_video is None:
            errors.append("training requires target_video, target_latent, or a raw_video window")
        if sample.prompt is None and sample.text_cache is None:
            errors.append("training requires prompt or text_cache")
        if sample.target_video:
            try:
                info = _video_info(sample.target_video)
                if (info["frames"], round(info["fps"]), info["height"], info["width"]) != (81, 16, 480, 832):
                    errors.append(f"target video contract mismatch: {info}")
            except Exception as exc:
                errors.append(str(exc))
        if sample.target_latent and deep:
            try:
                import torch
                obj = torch.load(sample.target_latent, map_location="cpu", weights_only=False)
                latent = obj.get("latent") if isinstance(obj, dict) else obj
                if tuple(latent.shape) != (16, 21, 60, 104):
                    errors.append(f"target latent shape {tuple(latent.shape)} != (16,21,60,104)")
                if isinstance(obj, dict) and "condition" in obj and tuple(obj["condition"].shape) != (20, 21, 60, 104):
                    errors.append(f"condition shape {tuple(obj['condition'].shape)} != (20,21,60,104)")
            except Exception as exc:
                errors.append(f"target latent load failed: {exc}")
    elif sample.prompt is None:
        errors.append("inference requires prompt; text_cache is train-only")
    return errors
