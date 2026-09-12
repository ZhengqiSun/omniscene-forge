#!/usr/bin/env python3
"""Project teacher non-ego player masks onto dense, latent, or token grids."""

from __future__ import annotations
from runtime_paths import source_path

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DENSE_HW = (176, 320)
DEFAULT_LATENT_HW = (60, 104)
DEFAULT_DATASET_ROOT = Path(str(source_path('assets', 'csgo-datasets-fullsubset/32f1644d4f42c29d')))
MEMORY_MASK_REGION_KINDS = {
    "memory_dense_channel_3_surrogate_v0",
    "memory_dense_channel_3_player_mask_surrogate_v0",
}


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


def _sample_get(sample: Any, key: str, default: Any = None) -> Any:
    if isinstance(sample, dict):
        if key in sample:
            return sample[key]
        row = sample.get("row")
        if isinstance(row, dict) and key in row:
            return row[key]
        return default
    if hasattr(sample, key):
        return getattr(sample, key)
    row = getattr(sample, "row", None)
    if isinstance(row, dict) and key in row:
        return row[key]
    return default


def _sample_row(sample: Any) -> dict[str, Any]:
    if isinstance(sample, dict):
        row = sample.get("row")
        return row if isinstance(row, dict) else sample
    row = getattr(sample, "row", None)
    return row if isinstance(row, dict) else {}


def _uses_memory_mask_surrogate(sample: Any) -> bool:
    row = _sample_row(sample)
    manifest_policy = row.get("region_mask_policy")
    candidates = [
        row.get("region_mask_kind"),
        row.get("region_mask_source"),
        _sample_get(sample, "region_mask_kind"),
        _sample_get(sample, "region_mask_source"),
    ]
    if isinstance(manifest_policy, dict):
        candidates.append(manifest_policy.get("kind"))
        candidates.append(manifest_policy.get("source"))
    return any(str(value) in MEMORY_MASK_REGION_KINDS for value in candidates if value is not None)


def _episode_dir_from_sample(
    sample: Any,
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
) -> Path:
    episode_dir = _sample_get(sample, "episode_dir")
    if episode_dir:
        return Path(episode_dir)

    match_dir = _sample_get(sample, "match_dir")
    if not match_dir:
        match_id = _sample_get(sample, "match_id")
        if not match_id:
            raise ValueError("sample must provide match_dir or match_id")
        match_dir = dataset_root / str(match_id)

    episode = _sample_get(sample, "raw_episode") or _sample_get(sample, "episode")
    if not episode:
        raise ValueError("sample must provide raw_episode or episode")
    episode = str(episode).split("_")[-2] + "_" + str(episode).split("_")[-1] if str(episode).count("_Ep_") else str(episode)
    if not episode.startswith("Ep_"):
        raise ValueError(f"cannot resolve raw episode name from {episode!r}")
    return Path(match_dir) / "train" / episode


def _is_context_sample(sample: Any) -> bool:
    return str(_sample_get(sample, "selection_role", "")).lower() == "context"


def _dense_mod():
    tools_dir = Path(__file__).resolve().parent
    return import_tool(tools_dir / "build_dense_condition_v0.py", "dense_condition_v0")


def _compare_mod():
    tools_dir = Path(__file__).resolve().parent
    return import_tool(
        tools_dir / "compare_memory_dense_channels_to_teacher_v0.py",
        "compare_memory_dense_channels_to_teacher_v0",
    )


def teacher_player_mask_dense(
    sample: Any,
    *,
    dense_hw: tuple[int, int] = DENSE_HW,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    force_compute_context: bool = False,
) -> np.ndarray:
    """Return bool [dense_h,dense_w] teacher mask for visible non-ego players.

    This intentionally follows the teacher QA path used by
    ``compare_memory_dense_channels_to_teacher_v0.py``: read the raw seg/depth
    streams, call ``build_dense_condition_v0.build_masks()``, then merge enemy
    and teammate masks. Context samples return all-False by default.
    """
    height, width = int(dense_hw[0]), int(dense_hw[1])
    if _uses_memory_mask_surrogate(sample):
        dense_path = _sample_get(sample, "dense_path")
        if not dense_path:
            raise ValueError("Memory-mask surrogate sample must provide dense_path")
        dense = np.load(Path(dense_path))["dense"]
        mask = dense[3] > 0.5
        if mask.shape != (height, width):
            mask = resize_mask_any(mask, (height, width))
        return mask

    if _is_context_sample(sample) and not force_compute_context:
        return np.zeros((height, width), dtype=bool)

    episode_dir = _episode_dir_from_sample(sample, dataset_root=dataset_root)
    ego_stem = str(_sample_get(sample, "ego_stem"))
    frame_index = int(_sample_get(sample, "frame_index"))
    dense_mod = _dense_mod()
    compare_mod = _compare_mod()

    teacher_player, _meta = compare_mod.teacher_other_player_channels(
        dense_mod,
        episode_dir,
        ego_stem,
        frame_index,
        height,
        width,
    )
    return teacher_player[0] > 0.5


def resize_mask_any(mask: np.ndarray, output_hw: tuple[int, int]) -> np.ndarray:
    """Area-pool a bool mask; output cell is true if any source coverage exists."""
    height, width = int(output_hw[0]), int(output_hw[1])
    if mask.shape == (height, width):
        return mask.astype(bool, copy=True)
    pooled = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
    return pooled > 0.0


def teacher_player_mask_latent(
    sample: Any,
    latent_hw: tuple[int, int] = DEFAULT_LATENT_HW,
) -> np.ndarray:
    """Return bool [latent_h,latent_w] teacher visible non-ego player pixels.

    The mask is first built on the frozen Memory dense grid [176,320] with the
    same full-frame teacher QA crop/FOV path, then area-pooled to ``latent_hw``.
    Use ``latent_hw=(60,104)`` for Wan latent cells or ``(30,52)`` for token
    grid region metrics.
    """
    dense_mask = teacher_player_mask_dense(sample, dense_hw=DENSE_HW)
    return resize_mask_any(dense_mask, latent_hw)


def _load_sample(manifest_path: Path, sample_id: str | None, sample_index: int) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    samples = manifest.get("samples", [])
    if sample_id is not None:
        for sample in samples:
            if str(sample.get("sample_id")) == sample_id:
                return sample
        raise KeyError(f"sample_id not found in manifest: {sample_id}")
    return samples[sample_index]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--sample-id")
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--latent-hw", type=int, nargs=2, default=list(DEFAULT_LATENT_HW), metavar=("H", "W"))
    ap.add_argument("--out", type=Path, help="Optional .npz output path with key 'mask'.")
    args = ap.parse_args()

    sample = _load_sample(args.manifest, args.sample_id, args.sample_index)
    mask = teacher_player_mask_latent(sample, latent_hw=(args.latent_hw[0], args.latent_hw[1]))
    info = {
        "sample_id": sample.get("sample_id"),
        "selection_role": sample.get("selection_role"),
        "shape": list(mask.shape),
        "dtype": "bool",
        "pixels": int(mask.sum()),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, mask=mask)
        info["out"] = str(args.out)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
