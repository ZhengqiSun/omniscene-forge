from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

SCHEMA_ID = "ScopeSampleV1"
VALID_SPLITS = {"train", "validation", "val", "test"}
MODEL_FIELDS = {
    "sample_id", "split", "initial_image", "raw_video", "raw_start",
    "raw_action_path", "scope_action_path", "prompt", "source_fps",
    "model_fps", "num_frames", "height", "width", "target_video",
    "target_latent",
}
EVALUATOR_ONLY_FIELDS = {
    "gt_video", "camera_pose", "camera_poses", "intrinsics", "group_id",
    "view_id", "dense", "state", "dense_path", "state_path",
    "player_mask", "world_events", "map_memory", "future_pose",
    "future_state",
}


@dataclass(frozen=True)
class ScopeSampleV1:
    sample_id: str
    split: str
    prompt: str
    source_fps: float
    model_fps: float
    num_frames: int
    height: int
    width: int
    initial_image: str | None = None
    raw_video: str | None = None
    raw_start: float | None = None
    raw_action_path: str | None = None
    scope_action_path: str | None = None
    target_video: str | None = None
    target_latent: str | None = None
    evaluator: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ScopeSampleV1":
        if row.get("schema_id", SCHEMA_ID) != SCHEMA_ID:
            raise ValueError(f"unsupported schema_id: {row.get('schema_id')!r}")
        allowed = {f.name for f in fields(cls)} | {"schema_id"} | EVALUATOR_ONLY_FIELDS
        unknown = sorted(set(row) - allowed)
        if unknown:
            raise ValueError(f"unknown fields: {unknown}")
        evaluator = dict(row.get("evaluator") or {})
        for key in EVALUATOR_ONLY_FIELDS:
            if key in row:
                evaluator[key] = row[key]
        values = {f.name: row.get(f.name) for f in fields(cls) if f.name != "evaluator"}
        values["evaluator"] = evaluator or None
        sample = cls(**values)
        sample.validate_contract()
        return sample

    def validate_contract(self) -> None:
        if not self.sample_id or self.split not in VALID_SPLITS:
            raise ValueError("sample_id must be non-empty and split must be train/validation/val/test")
        if bool(self.initial_image) == bool(self.raw_video):
            raise ValueError("provide exactly one of initial_image or raw_video")
        if self.raw_video and self.raw_start is None:
            raise ValueError("raw_video requires raw_start as a source-frame index")
        if bool(self.raw_action_path) == bool(self.scope_action_path):
            raise ValueError("provide exactly one of raw_action_path or scope_action_path")
        if self.num_frames not in (81, 101):
            raise ValueError("num_frames must be 81 (native) or 101 (candidate)")
        if (self.height, self.width, self.model_fps) != (480, 832, 20):
            raise ValueError("SCOPE generation contract is 480x832 at 20 FPS")
        if self.source_fps <= 0:
            raise ValueError("source_fps must be positive")

    def model_projection(self, mode: str) -> dict[str, Any]:
        """Return only fields the model-side dataset is authorized to observe."""
        if mode not in {"train", "infer"}:
            raise ValueError(mode)
        keys = MODEL_FIELDS if mode == "train" else MODEL_FIELDS - {"target_video", "target_latent"}
        return {key: getattr(self, key) for key in keys if getattr(self, key) is not None}

    def identity_projection(self) -> dict[str, Any]:
        """Non-model identity used only for output naming/provenance."""
        evaluator=self.evaluator or {}
        return {key:evaluator[key] for key in ("group_id","view_id") if key in evaluator}


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sha256(path: str | Path) -> str:
    return sha256_file(path)


def load_manifest(path: str | Path) -> list[ScopeSampleV1]:
    result: list[ScopeSampleV1] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                sample = ScopeSampleV1.from_dict(json.loads(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            if sample.sample_id in seen:
                raise ValueError(f"duplicate sample_id: {sample.sample_id}")
            seen.add(sample.sample_id)
            result.append(sample)
    if not result:
        raise ValueError("manifest is empty")
    return result


def resolve_path(manifest: str | Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(manifest).resolve().parent / path


def validate_paths(sample: ScopeSampleV1, manifest: str | Path, mode: str) -> list[str]:
    errors: list[str] = []
    projection = sample.model_projection(mode)
    path_fields = {"initial_image", "raw_video", "raw_action_path", "scope_action_path"}
    if mode == "train":
        path_fields |= {"target_video", "target_latent"}
        if not sample.target_video and not sample.target_latent:
            errors.append("training requires target_video or target_latent")
        if sample.target_latent:
            errors.append("target_latent is disabled until serialized Wan2.2 VAE shape/dtype is runtime-audited")
    for key in path_fields:
        if value := projection.get(key):
            if not resolve_path(manifest, value).is_file():
                errors.append(f"missing {key}: {value}")
    return errors
