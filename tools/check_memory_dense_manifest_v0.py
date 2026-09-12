#!/usr/bin/env python3
"""Validate a Memory dense dataset manifest for adapter training plumbing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def resolve(path: str, manifest_path: Path) -> Path:
    p = Path(path)
    if p.exists():
        return p
    alt = manifest_path.parent / p.name
    if alt.exists():
        return alt
    return p


def sample_path(sample: dict, key: str, rel_key: str, manifest_path: Path) -> Path:
    if rel_key in sample:
        rel = manifest_path.parent / sample[rel_key]
        if rel.exists():
            return rel
    if "source_manifest" in sample:
        source_base = Path(sample["source_manifest"]).parent
        if rel_key in sample:
            rel = source_base / sample[rel_key]
            if rel.exists():
                return rel
    return resolve(sample[key], manifest_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--max-samples", type=int, default=4)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checked = []
    for sample in manifest["samples"][: args.max_samples]:
        dense_path = sample_path(sample, "dense_path", "dense_relpath", args.manifest)
        target_path = sample_path(sample, "target_rgb_path", "target_rgb_relpath", args.manifest)
        if not dense_path.exists():
            # Local copied manifests may keep server absolute paths. Fall back to
            # the same relative sample layout under the manifest directory.
            dense_path = args.manifest.parent / "samples" / sample["sample_id"] / "mesh_dense_condition_v0.npz"
        if not target_path.exists():
            target_path = args.manifest.parent / "samples" / sample["sample_id"] / "target_rgb.png"
        dense = np.load(dense_path)["dense"]
        target = Image.open(target_path)
        assert tuple(dense.shape) == tuple(sample["shape"]), (dense.shape, sample["shape"])
        assert dense.dtype == np.float32, dense.dtype
        assert target.size == (dense.shape[2], dense.shape[1]), (target.size, dense.shape)
        checked.append({
            "sample_id": sample["sample_id"],
            "dense_shape": list(dense.shape),
            "target_size": list(target.size),
            "dense_minmax": [float(dense.min()), float(dense.max())],
        })
    print(json.dumps({
        "manifest": str(args.manifest),
        "sample_count": manifest["sample_count"],
        "checked": checked,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
