#!/usr/bin/env python3
"""Summarize a Memory dense dataset manifest for map-channel QA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_sample_path(base: Path, sample: dict[str, Any], key: str, rel_key: str) -> Path:
    if sample.get(rel_key):
        rel = base / sample[rel_key]
        if rel.exists():
            return rel
    if sample.get("source_manifest"):
        source_base = Path(sample["source_manifest"]).parent
        if sample.get(rel_key):
            rel = source_base / sample[rel_key]
            if rel.exists():
                return rel
    p = Path(sample[key])
    if p.exists():
        return p
    return base / "samples" / sample["sample_id"] / p.name


def make_montage(base: Path, samples: list[dict[str, Any]], out_path: Path, max_images: int) -> str | None:
    image_paths = []
    for sample in samples[:max_images]:
        p = resolve_sample_path(base, sample, "qa_path", "qa_relpath")
        if p.exists():
            image_paths.append((sample["sample_id"], p))
    if not image_paths:
        return None
    thumbs = []
    cell_w = 480
    label_h = 24
    for sample_id, path in image_paths:
        img = Image.open(path).convert("RGB")
        scale = cell_w / img.width
        thumb = img.resize((cell_w, max(1, int(img.height * scale))), Image.Resampling.BILINEAR)
        cell = Image.new("RGB", (cell_w, thumb.height + label_h), (248, 248, 245))
        cell.paste(thumb, (0, label_h))
        draw = ImageDraw.Draw(cell)
        draw.text((6, 5), sample_id[-32:], fill=(20, 20, 20))
        thumbs.append(cell)
    cols = min(2, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    cell_h = max(t.height for t in thumbs)
    canvas = Image.new("RGB", (cell_w * cols, cell_h * rows), (248, 248, 245))
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        canvas.paste(thumb, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return str(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-montage", type=Path, default=None)
    ap.add_argument("--max-montage-images", type=int, default=6)
    args = ap.parse_args()

    manifest = load_json(args.manifest)
    base = args.manifest.parent
    nav_hits = []
    mesh_hits = []
    projected_players = []
    place_pixels: dict[str, int] = {}
    dominant_places: dict[str, int] = {}
    channel_min = None
    channel_max = None

    for sample in manifest["samples"]:
        meta = load_json(resolve_sample_path(base, sample, "meta_path", "meta_relpath"))
        dense = np.load(resolve_sample_path(base, sample, "dense_path", "dense_relpath"))["dense"]
        cur_min = dense.reshape(dense.shape[0], -1).min(axis=1)
        cur_max = dense.reshape(dense.shape[0], -1).max(axis=1)
        channel_min = cur_min if channel_min is None else np.minimum(channel_min, cur_min)
        channel_max = cur_max if channel_max is None else np.maximum(channel_max, cur_max)
        nav_hits.append(float(meta.get("nav_semantic_hit_ratio", float((dense[2] > 0).mean()))))
        mesh_hits.append(float(meta.get("mesh_hit_ratio", float((dense[1] > 0).mean()))))
        projected_players.append(len(meta.get("memory_projected_players", [])))
        nav_meta = meta.get("nav_semantic", {})
        dominant = nav_meta.get("dominant_place")
        if dominant:
            dominant_places[dominant] = dominant_places.get(dominant, 0) + 1
        for row in nav_meta.get("visible_places", []):
            place = row["place"]
            place_pixels[place] = place_pixels.get(place, 0) + int(row["pixels"])

    summary = {
        "kind": "memory_dense_dataset_summary_v0",
        "manifest": str(args.manifest),
        "sample_count": int(manifest["sample_count"]),
        "channels": manifest["samples"][0]["channels"] if manifest["samples"] else [],
        "mesh_hit_ratio": stat(mesh_hits),
        "nav_semantic_hit_ratio": stat(nav_hits),
        "projected_players": stat(projected_players),
        "channel_min": channel_min.astype(float).tolist() if channel_min is not None else [],
        "channel_max": channel_max.astype(float).tolist() if channel_max is not None else [],
        "top_visible_places_by_pixels": sorted(place_pixels.items(), key=lambda x: x[1], reverse=True)[:12],
        "dominant_place_counts": sorted(dominant_places.items(), key=lambda x: x[1], reverse=True),
    }
    if args.out_montage:
        summary["montage_path"] = make_montage(base, manifest["samples"], args.out_montage, args.max_montage_images)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def stat(values: list[float] | list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "mean": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {"min": float(arr.min()), "mean": float(arr.mean()), "max": float(arr.max())}


if __name__ == "__main__":
    main()
