#!/usr/bin/env python3
"""Known-pattern Map Memory dense tensors for token-alignment tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


DENSE_SHAPE = (7, 176, 320)
QUADRANTS = {
    0: "tl",
    1: "tr",
    2: "bl",
    3: "br",
    "tl": "tl",
    "top_left": "tl",
    "tr": "tr",
    "top_right": "tr",
    "bl": "bl",
    "bottom_left": "bl",
    "br": "br",
    "bottom_right": "br",
}


def quadrant_slices(height: int, width: int, quadrant: int | str = 0) -> tuple[slice, slice]:
    """Return row/column slices for one image quadrant."""
    q = QUADRANTS.get(quadrant)
    if q is None:
        raise ValueError(f"quadrant must be one of {sorted(QUADRANTS, key=str)}, got {quadrant!r}")
    rows = slice(0, height // 2) if q in {"tl", "tr"} else slice(height // 2, height)
    cols = slice(0, width // 2) if q in {"tl", "bl"} else slice(width // 2, width)
    return rows, cols


def known_quadrant_dense(
    quadrant: int | str = 0,
    *,
    shape: tuple[int, int, int] = DENSE_SHAPE,
    channels: Iterable[int] | None = None,
    value: float = 1.0,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Return a dense tensor with one quadrant set to ``value`` and all else zero.

    The default is the requested P0 seam-test sample: all 7 channels in the
    top-left quarter are 1.0, and the rest of the [7,176,320] tensor is 0.0.
    ``quadrant`` accepts 0/1/2/3 as TL/TR/BL/BR or string aliases.
    """
    if len(shape) != 3:
        raise ValueError(f"shape must be [channels,height,width], got {shape!r}")
    channel_count, height, width = (int(shape[0]), int(shape[1]), int(shape[2]))
    dense = np.zeros((channel_count, height, width), dtype=dtype)
    channel_ids = range(channel_count) if channels is None else channels
    rows, cols = quadrant_slices(height, width, quadrant)
    for channel in channel_ids:
        idx = int(channel)
        if idx < 0 or idx >= channel_count:
            raise ValueError(f"channel index {idx} outside [0,{channel_count})")
        dense[idx, rows, cols] = value
    return dense


def known_top_left_quarter_dense(**kwargs) -> np.ndarray:
    """Compatibility helper for the canonical top-left-quarter sample."""
    return known_quadrant_dense(0, **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quadrant", default="tl", choices=["tl", "tr", "bl", "br"])
    ap.add_argument("--out", type=Path, help="Optional .npz output path with key 'dense'.")
    args = ap.parse_args()

    dense = known_quadrant_dense(args.quadrant)
    info = {
        "shape": list(dense.shape),
        "dtype": str(dense.dtype),
        "quadrant": args.quadrant,
        "nonzero_pixels": int(np.count_nonzero(dense)),
        "sum": float(dense.sum()),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, dense=dense)
        info["out"] = str(args.out)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
