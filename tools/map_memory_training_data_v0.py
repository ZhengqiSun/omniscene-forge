#!/usr/bin/env python3
"""Training data contract helpers for Map Memory dense adapter training.

This module is deliberately strict.  It admits versioned release manifests that
have passed the Map Memory readiness gate, resolves their sample sidecars, and
keeps teacher/QA fields out of the model-input path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CHANNELS = [
    "env_depth_norm_from_world_obj_projection",
    "env_mesh_hit_mask",
    "env_nav_place_semantic_from_static_memory",
    "other_player_mask_from_memory_player_capsules",
    "other_player_depth_norm_from_memory_capsules",
    "other_player_yaw_sin_relative_to_ego_from_memory",
    "other_player_yaw_cos_relative_to_ego_from_memory",
]
# v2 dense contract (416x240 -> 30x52 native, no interpolation)
DENSE_H, DENSE_W = 240, 416
SHAPE = (7, DENSE_H, DENSE_W)
REQUIRED_BACKEND_ID = "bsp_faces_disp_gpu"
MODEL_INPUT_KEYS = {"dense_path", "dense_relpath"}
TEACHER_OR_TARGET_KEYS = {
    "target_rgb_path",
    "target_rgb_relpath",
    "meta_path",
    "meta_relpath",
    "qa_path",
    "qa_relpath",
    "teacher_qa_at_selection",
}
MEMORY_MASK_REGION_KINDS = {
    "memory_dense_channel_3_surrogate_v0",
    "memory_dense_channel_3_player_mask_surrogate_v0",
}


def set_dense_hw(height: int, width: int) -> tuple[int, int, int]:
    """Switch the module-level dense grid at runtime and return the new SHAPE.

    The default is the v2 dense contract (416x240 -> 30x52 native, no interpolation).
    Every validator in this module reads the module-level ``SHAPE`` at call time,
    so callers that need to read a legacy pre-v2 release (or run a controlled
    grid ablation) can switch the contract here before loading anything.
    """
    global DENSE_H, DENSE_W, SHAPE
    height = int(height)
    width = int(width)
    if height <= 0 or width <= 0:
        raise ValueError(f"dense H/W must be positive, got {(height, width)}")
    DENSE_H, DENSE_W = height, width
    SHAPE = (len(CHANNELS), DENSE_H, DENSE_W)
    return SHAPE


def dense_hw() -> tuple[int, int]:
    """Current dense grid (H, W); v2 dense contract default."""
    return (DENSE_H, DENSE_W)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_bucket(text: str, modulo: int = 10_000) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulo


def resolve_sample_path(manifest_path: Path, sample: dict[str, Any], key: str, rel_key: str) -> Path:
    if sample.get(rel_key):
        local = manifest_path.parent / sample[rel_key]
        if local.exists():
            return local
    if sample.get("source_manifest") and sample.get(rel_key):
        source = Path(sample["source_manifest"]).parent / sample[rel_key]
        if source.exists():
            return source
    raw = Path(sample.get(key, ""))
    if raw.exists():
        return raw
    return manifest_path.parent / "samples" / str(sample.get("sample_id", "")) / raw.name


@dataclass(frozen=True)
class MapMemorySample:
    sample_id: str
    match_id: str
    episode: str
    raw_episode: str
    ego_stem: str
    frame_index: int
    selection_role: str
    dense_path: Path
    target_rgb_path: Path
    meta_path: Path
    qa_path: Path
    row: dict[str, Any]

    @property
    def split_group_episode(self) -> str:
        return self.episode

    @property
    def split_group_track(self) -> str:
        return f"{self.match_id}|{self.raw_episode}|{self.ego_stem}"


class MapMemoryRelease:
    def __init__(
        self,
        manifest_path: Path,
        manifest: dict[str, Any],
        readiness: dict[str, Any],
        teacher_qa: dict[str, Any],
        samples: list[MapMemorySample],
    ) -> None:
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.readiness = readiness
        self.teacher_qa = teacher_qa
        self.samples = samples
        self.by_id = {sample.sample_id: sample for sample in samples}

    @classmethod
    def load(
        cls,
        manifest_path: Path,
        *,
        readiness_path: Path | None = None,
        teacher_qa_path: Path | None = None,
        require_backend_id: str = REQUIRED_BACKEND_ID,
        validate_sidecars: bool = True,
    ) -> "MapMemoryRelease":
        manifest_path = manifest_path.resolve()
        manifest = load_json(manifest_path)
        readiness_path = readiness_path or manifest_path.parent / "training_readiness_v0.json"
        teacher_qa_path = teacher_qa_path or manifest_path.parent / "channel_teacher_qa_v0" / "memory_dense_channels_vs_teacher_v0.json"
        readiness = load_json(readiness_path)
        teacher_qa = load_json(teacher_qa_path)
        failures = validate_release_manifest(
            manifest_path,
            manifest,
            readiness,
            teacher_qa,
            require_backend_id=require_backend_id,
            validate_sidecars=validate_sidecars,
        )
        if failures:
            raise ValueError("Map Memory release validation failed:\n" + "\n".join(f"- {msg}" for msg in failures))
        qa_rows = {str(row["sample_id"]): row for row in teacher_qa.get("rows", [])}
        samples = [
            sample_from_manifest(manifest_path, sample, qa_rows[str(sample["sample_id"])])
            for sample in manifest.get("samples", [])
        ]
        return cls(manifest_path, manifest, readiness, teacher_qa, samples)

    def audit(self, *, dense_limit: int | None = None, hash_payloads: bool = False) -> dict[str, Any]:
        return audit_release(self, dense_limit=dense_limit, hash_payloads=hash_payloads)


def sample_from_manifest(manifest_path: Path, sample: dict[str, Any], qa_row: dict[str, Any]) -> MapMemorySample:
    return MapMemorySample(
        sample_id=str(sample["sample_id"]),
        match_id=str(sample["match_id"]),
        episode=str(sample["episode"]),
        raw_episode=str(sample["raw_episode"]),
        ego_stem=str(sample["ego_stem"]),
        frame_index=int(sample["frame_index"]),
        selection_role=str(sample["selection_role"]),
        dense_path=resolve_sample_path(manifest_path, sample, "dense_path", "dense_relpath"),
        target_rgb_path=resolve_sample_path(manifest_path, sample, "target_rgb_path", "target_rgb_relpath"),
        meta_path=resolve_sample_path(manifest_path, sample, "meta_path", "meta_relpath"),
        qa_path=resolve_sample_path(manifest_path, sample, "qa_path", "qa_relpath"),
        row=sample,
    )


def uses_memory_mask_region_surrogate(manifest: dict[str, Any], teacher_qa: dict[str, Any] | None = None) -> bool:
    """Return true for light-data releases whose region mask is explicit Memory ch3.

    These releases have no teacher seg/depth streams. They are only valid for
    relative true-vs-shuffled region comparisons, where both variants are scored
    on the same Memory-projected player mask.
    """
    candidates = [
        manifest.get("region_mask_kind"),
        manifest.get("region_mask_source"),
        manifest.get("region_mask_policy"),
    ]
    if isinstance(manifest.get("region_mask_policy"), dict):
        candidates.append(manifest["region_mask_policy"].get("kind"))
        candidates.append(manifest["region_mask_policy"].get("source"))
    if teacher_qa:
        candidates.extend([
            teacher_qa.get("region_mask_kind"),
            teacher_qa.get("region_mask_source"),
        ])
        if isinstance(teacher_qa.get("policy"), dict):
            candidates.append(teacher_qa["policy"].get("region_mask_kind"))
            candidates.append(teacher_qa["policy"].get("region_mask_source"))
    return any(str(value) in MEMORY_MASK_REGION_KINDS for value in candidates if value is not None)


def validate_release_manifest(
    manifest_path: Path,
    manifest: dict[str, Any],
    readiness: dict[str, Any],
    teacher_qa: dict[str, Any],
    *,
    require_backend_id: str = REQUIRED_BACKEND_ID,
    validate_sidecars: bool = True,
) -> list[str]:
    failures: list[str] = []
    samples = manifest.get("samples", [])
    qa_rows = {str(row.get("sample_id")): row for row in teacher_qa.get("rows", [])}
    memory_mask_surrogate = uses_memory_mask_region_surrogate(manifest, teacher_qa)
    if readiness.get("status") != "pass" and not memory_mask_surrogate:
        failures.append(f"readiness status is {readiness.get('status')!r}, expected 'pass'")
    if readiness.get("manifest") and Path(str(readiness["manifest"])).name != manifest_path.name:
        # Historical readiness files may store repo-relative paths; require at least the same basename.
        expected_parent = str(manifest_path.parent.name)
        if expected_parent not in str(readiness["manifest"]):
            failures.append(f"readiness manifest does not point to this release: {readiness.get('manifest')}")
    if readiness.get("failures") and not memory_mask_surrogate:
        failures.append(f"readiness failures are non-empty: {readiness.get('failures')}")
    if manifest.get("sample_count") != len(samples):
        failures.append("manifest sample_count does not match samples length")
    if teacher_qa.get("sample_count") != len(samples) or len(qa_rows) != len(samples):
        failures.append("teacher QA does not cover every sample")
    if teacher_qa.get("missing_count") not in (0, None):
        failures.append(f"teacher QA missing_count is {teacher_qa.get('missing_count')}")
    if manifest.get("shape") != list(SHAPE):
        failures.append(f"manifest shape is {manifest.get('shape')}, expected {list(SHAPE)}")
    if manifest.get("channels") != CHANNELS:
        failures.append("manifest channel order does not match current 7-channel contract")
    if manifest.get("geometry_backend_id") != require_backend_id:
        failures.append(f"manifest geometry_backend_id is {manifest.get('geometry_backend_id')}, expected {require_backend_id}")

    seen: set[str] = set()
    for sample in samples:
        sid = str(sample.get("sample_id", ""))
        if not sid:
            failures.append("sample missing sample_id")
            continue
        if sid in seen:
            failures.append(f"duplicate sample_id: {sid}")
        seen.add(sid)
        if sid not in qa_rows:
            failures.append(f"{sid}: missing teacher QA row")
        if sample.get("shape") != list(SHAPE):
            failures.append(f"{sid}: sample shape is {sample.get('shape')}, expected {list(SHAPE)}")
        if sample.get("channels") != CHANNELS:
            failures.append(f"{sid}: sample channel order mismatch")
        if sample.get("geometry_backend_id") != require_backend_id:
            failures.append(f"{sid}: backend {sample.get('geometry_backend_id')} != {require_backend_id}")
        role = sample.get("selection_role")
        if role not in {"positive", "context"}:
            failures.append(f"{sid}: invalid selection_role {role!r}")
        dense_text = f"{sample.get('dense_path', '')} {sample.get('dense_relpath', '')}"
        if "_depth.mkv" in dense_text or "_seg.mkv" in dense_text or "visibility" in dense_text:
            failures.append(f"{sid}: dense path appears to reference teacher streams")
        if validate_sidecars:
            for key, rel_key in [
                ("dense_path", "dense_relpath"),
                ("target_rgb_path", "target_rgb_relpath"),
                ("meta_path", "meta_relpath"),
                ("qa_path", "qa_relpath"),
            ]:
                path = resolve_sample_path(manifest_path, sample, key, rel_key)
                if not path.exists():
                    failures.append(f"{sid}: missing {key}: {path}")

        row = qa_rows.get(sid) or {}
        visible = int(row.get("visible_teacher_players", 0) or 0)
        teacher_pixels = int(row.get("other_player_teacher_pixels", 0) or 0)
        memory_pixels = int(row.get("other_player_memory_pixels", 0) or 0)
        if memory_mask_surrogate:
            if role == "positive" and memory_pixels <= 0:
                failures.append(f"{sid}: positive surrogate sample has empty Memory other-player mask")
            if role == "context" and memory_pixels != 0:
                failures.append(f"{sid}: context surrogate sample has Memory other-player pixels")
        else:
            if role == "positive":
                if visible <= 0 or teacher_pixels <= 0 or memory_pixels <= 0:
                    failures.append(f"{sid}: positive sample is not teacher/memory positive")
            if role == "context":
                if visible != 0 or teacher_pixels != 0 or memory_pixels != 0:
                    failures.append(f"{sid}: context sample is not teacher-empty and memory-empty")
    return failures


def validate_dense_array(dense: np.ndarray) -> list[str]:
    errors: list[str] = []
    if dense.shape != SHAPE:
        errors.append(f"dense shape {list(dense.shape)} != {list(SHAPE)}")
    if dense.dtype != np.float32:
        errors.append(f"dense dtype {dense.dtype} != float32")
    if not np.isfinite(dense).all():
        errors.append("dense contains NaN or Inf")
    if dense.shape == SHAPE:
        for idx in range(5):
            if float(dense[idx].min()) < -1e-5 or float(dense[idx].max()) > 1.0001:
                errors.append(f"channel {idx} outside [0,1]")
        for idx in (5, 6):
            if float(dense[idx].min()) < -1.0001 or float(dense[idx].max()) > 1.0001:
                errors.append(f"channel {idx} outside [-1,1]")
    return errors


def load_dense(sample: MapMemorySample) -> np.ndarray:
    dense = np.load(sample.dense_path)["dense"]
    errors = validate_dense_array(dense)
    if errors:
        raise ValueError(f"{sample.sample_id}: invalid dense tensor: {errors}")
    return dense


def audit_release(
    release: MapMemoryRelease,
    *,
    dense_limit: int | None = None,
    hash_payloads: bool = False,
) -> dict[str, Any]:
    samples = release.samples
    role_counts: dict[str, int] = {}
    match_counts: dict[str, int] = {}
    episode_counts: dict[str, int] = {}
    for sample in samples:
        role_counts[sample.selection_role] = role_counts.get(sample.selection_role, 0) + 1
        match_counts[sample.match_id] = match_counts.get(sample.match_id, 0) + 1
        episode_counts[sample.episode] = episode_counts.get(sample.episode, 0) + 1

    checked = samples if dense_limit is None else samples[:dense_limit]
    channel_min = np.full(7, np.inf, dtype=np.float64)
    channel_max = np.full(7, -np.inf, dtype=np.float64)
    channel_sum = np.zeros(7, dtype=np.float64)
    pixel_count = 0
    dense_errors: list[dict[str, Any]] = []
    dense_hashes: dict[str, int] = {}
    for sample in checked:
        try:
            dense = load_dense(sample)
        except Exception as exc:
            dense_errors.append({"sample_id": sample.sample_id, "error": str(exc)})
            continue
        channel_min = np.minimum(channel_min, dense.min(axis=(1, 2)))
        channel_max = np.maximum(channel_max, dense.max(axis=(1, 2)))
        channel_sum += dense.sum(axis=(1, 2))
        pixel_count += dense.shape[1] * dense.shape[2]
        if hash_payloads:
            h = hashlib.blake2b(dense.tobytes(), digest_size=16).hexdigest()
            dense_hashes[h] = dense_hashes.get(h, 0) + 1

    duplicate_dense_payloads = sum(count - 1 for count in dense_hashes.values() if count > 1)
    return {
        "kind": "map_memory_training_release_audit_v0",
        "manifest": str(release.manifest_path),
        "status": "pass" if not dense_errors else "fail",
        "sample_count": len(samples),
        "checked_dense_count": len(checked),
        "role_counts": role_counts,
        "match_count": len(match_counts),
        "match_counts": dict(sorted(match_counts.items())),
        "episode_count": len(episode_counts),
        "readiness_status": release.readiness.get("status"),
        "teacher_qa_summary": release.teacher_qa.get("summary"),
        "dense_channel_min": channel_min.tolist() if pixel_count else None,
        "dense_channel_max": channel_max.tolist() if pixel_count else None,
        "dense_channel_mean": (channel_sum / pixel_count).tolist() if pixel_count else None,
        "dense_errors": dense_errors[:50],
        "duplicate_dense_payloads_in_checked": duplicate_dense_payloads if hash_payloads else None,
    }


def split_samples(
    samples: Iterable[MapMemorySample],
    *,
    val_fraction: float,
    test_fraction: float,
    split_key: str = "episode",
    seed: int = 20260531,
) -> dict[str, list[MapMemorySample]]:
    if split_key not in {"episode", "track", "match"}:
        raise ValueError(f"split_key must be episode, track, or match; got {split_key}")
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be non-negative and sum to less than 1")

    buckets: dict[str, list[MapMemorySample]] = {"train": [], "val": [], "test": []}
    val_cut = int(val_fraction * 10_000)
    test_cut = int((val_fraction + test_fraction) * 10_000)
    for sample in samples:
        if split_key == "episode":
            group = sample.split_group_episode
        elif split_key == "track":
            group = sample.split_group_track
        else:
            group = sample.match_id
        bucket = stable_bucket(f"{seed}|{group}")
        split = "val" if bucket < val_cut else "test" if bucket < test_cut else "train"
        buckets[split].append(sample)
    return buckets


def split_report(splits: dict[str, list[MapMemorySample]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, rows in splits.items():
        out[name] = {
            "sample_count": len(rows),
            "match_count": len({s.match_id for s in rows}),
            "episode_count": len({s.episode for s in rows}),
            "role_counts": {
                role: sum(1 for s in rows if s.selection_role == role)
                for role in sorted({s.selection_role for s in rows})
            },
        }
    return out
