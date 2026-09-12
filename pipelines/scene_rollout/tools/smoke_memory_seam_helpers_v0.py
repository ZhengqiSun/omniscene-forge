#!/usr/bin/env python3
"""Smoke-test Memory-side P0 seam helper functions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from known_dense_patterns_v0 import known_quadrant_dense
from teacher_player_latent_mask_v0 import teacher_player_mask_dense, teacher_player_mask_latent


DEFAULT_MANIFEST = Path("docs/assets/memory_dense_dataset_v43_bsp_all5_match_large_h176/manifest.json")
DEFAULT_TEACHER_QA = Path(
    "docs/assets/memory_dense_dataset_v43_bsp_all5_match_large_h176/channel_teacher_qa_v0/"
    "memory_dense_channels_vs_teacher_v0.json"
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def assert_known_dense() -> dict[str, object]:
    quadrant_sums: dict[str, list[int]] = {}
    for q, name in enumerate(["tl", "tr", "bl", "br"]):
        dense = known_quadrant_dense(q)
        assert dense.shape == (7, 176, 320), dense.shape
        assert dense.dtype == np.float32, dense.dtype
        assert set(np.unique(dense).tolist()) == {0.0, 1.0}
        h, w = dense.shape[1:]
        sums = [
            int(dense[:, : h // 2, : w // 2].sum()),
            int(dense[:, : h // 2, w // 2 :].sum()),
            int(dense[:, h // 2 :, : w // 2].sum()),
            int(dense[:, h // 2 :, w // 2 :].sum()),
        ]
        assert sums[q] == 7 * 88 * 160, (name, sums)
        assert sum(sums) == sums[q], (name, sums)
        quadrant_sums[name] = sums
    return {"quadrant_sums": quadrant_sums}


def assert_teacher_masks(manifest_path: Path, teacher_qa_path: Path) -> dict[str, object]:
    manifest = load_json(manifest_path)
    teacher_qa = load_json(teacher_qa_path)
    qa_by_id = {row["sample_id"]: row for row in teacher_qa["rows"]}
    positive = next(sample for sample in manifest["samples"] if sample["selection_role"] == "positive")
    context = next(sample for sample in manifest["samples"] if sample["selection_role"] == "context")

    rows = []
    for sample in [positive, context]:
        dense = teacher_player_mask_dense(sample)
        latent = teacher_player_mask_latent(sample, latent_hw=(60, 104))
        token = teacher_player_mask_latent(sample, latent_hw=(30, 52))
        qa_row = qa_by_id[sample["sample_id"]]
        row = {
            "sample_id": sample["sample_id"],
            "selection_role": sample["selection_role"],
            "dense_pixels": int(dense.sum()),
            "qa_teacher_pixels": int(qa_row["other_player_teacher_pixels"]),
            "latent_pixels_60x104": int(latent.sum()),
            "token_pixels_30x52": int(token.sum()),
        }
        if sample["selection_role"] == "positive":
            assert row["dense_pixels"] == row["qa_teacher_pixels"], row
            assert row["latent_pixels_60x104"] > 0, row
            assert row["token_pixels_30x52"] > 0, row
        else:
            assert row["dense_pixels"] == 0, row
            assert row["latent_pixels_60x104"] == 0, row
            assert row["token_pixels_30x52"] == 0, row
        rows.append(row)
    return {"teacher_mask_rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--teacher-qa", type=Path, default=DEFAULT_TEACHER_QA)
    args = ap.parse_args()

    out = {
        "known_dense": assert_known_dense(),
        "teacher_masks": assert_teacher_masks(args.manifest, args.teacher_qa),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
