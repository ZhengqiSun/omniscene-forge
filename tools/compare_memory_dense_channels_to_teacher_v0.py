#!/usr/bin/env python3
"""Compare Memory dense channels against dataset teacher streams.

This script is map-side QA. It does not create training inputs from teacher
streams; it only evaluates an existing Memory dense manifest against RGB/depth/
seg/visibility data when those raw episode files are available.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


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


def resolve_sample_path(manifest_path: Path, sample: dict[str, Any], key: str, rel_key: str) -> Path:
    if sample.get(rel_key):
        p = manifest_path.parent / sample[rel_key]
        if p.exists():
            return p
    if sample.get("source_manifest"):
        base = Path(sample["source_manifest"]).parent
        if sample.get(rel_key):
            p = base / sample[rel_key]
            if p.exists():
                return p
    p = Path(sample[key])
    if p.exists():
        return p
    return manifest_path.parent / "samples" / sample["sample_id"] / p.name


def resize_nearest(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    return cv2.resize(arr, (width, height), interpolation=cv2.INTER_NEAREST)


def resize_area(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    return cv2.resize(arr, (width, height), interpolation=cv2.INTER_AREA)


def gray(ch: np.ndarray) -> Image.Image:
    arr = np.nan_to_num(ch, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8)).convert("RGB")


def mask_img(mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    out[mask] = color
    return Image.fromarray(out)


def diff_mask_img(rgb: np.ndarray, teacher: np.ndarray, memory: np.ndarray) -> Image.Image:
    out = rgb.copy().astype(np.float32) * 0.42
    tp = teacher & memory
    fn = teacher & ~memory
    fp = memory & ~teacher
    out[tp] = np.asarray([40, 220, 90], dtype=np.float32)
    out[fn] = np.asarray([255, 70, 70], dtype=np.float32)
    out[fp] = np.asarray([70, 140, 255], dtype=np.float32)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def depth_error_img(err: np.ndarray, valid: np.ndarray, max_err: float) -> Image.Image:
    norm = np.zeros_like(err, dtype=np.float32)
    norm[valid] = np.clip(err[valid] / max(max_err, 1e-6), 0.0, 1.0)
    out = np.zeros((*err.shape, 3), dtype=np.uint8)
    out[..., 0] = (norm * 255).astype(np.uint8)
    out[..., 1] = ((1.0 - norm) * 180 * valid).astype(np.uint8)
    return Image.fromarray(out)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def mae_stats(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> dict[str, float | None]:
    if not np.any(valid):
        return {"mae": None, "median": None, "p95": None}
    diff = np.abs(a[valid] - b[valid])
    return {
        "mae": float(diff.mean()),
        "median": float(np.median(diff)),
        "p95": float(np.percentile(diff, 95)),
    }


def teacher_other_player_channels(
    dense_mod: Any,
    episode_dir: Path,
    ego_stem: str,
    frame_index: int,
    height: int,
    width: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    seg_bgr = dense_mod.read_frame(episode_dir / f"{ego_stem}_seg.mkv", frame_index)
    depth_bgr = dense_mod.read_frame(episode_dir / f"{ego_stem}_depth.mkv", frame_index)
    depth_full = depth_bgr.astype(np.float32).mean(axis=2) / 255.0
    player_channels, meta = dense_mod.build_masks(
        episode_dir,
        ego_stem,
        frame_index,
        seg_bgr,
        depth_full,
        (height, width),
        tolerance=6,
    )
    other_mask = np.maximum(player_channels[0], player_channels[1])
    return np.stack([
        other_mask,
        player_channels[2],
        player_channels[3],
        player_channels[4],
    ], axis=0).astype(np.float32), meta


def teacher_env_channels(
    dense_mod: Any,
    episode_dir: Path,
    ego_stem: str,
    frame_index: int,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth_bgr = dense_mod.read_frame(episode_dir / f"{ego_stem}_depth.mkv", frame_index)
    seg_bgr = dense_mod.read_frame(episode_dir / f"{ego_stem}_seg.mkv", frame_index)
    depth = dense_mod.make_depth(depth_bgr, (height, width))
    semantic = dense_mod.make_semantic(seg_bgr, (height, width))
    hit = depth > 1e-5
    return depth, hit.astype(np.float32), semantic


def make_sample_qa(
    out_path: Path,
    rgb: np.ndarray,
    memory: np.ndarray,
    teacher_depth: np.ndarray,
    teacher_hit: np.ndarray,
    teacher_semantic: np.ndarray,
    teacher_player: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    h, w = memory.shape[1:]
    rgb_small = np.asarray(Image.fromarray(rgb).resize((w, h), Image.Resampling.BILINEAR))
    mem_depth = memory[0]
    mem_hit = memory[1] > 0.5
    mem_sem = memory[2] > 0
    mem_player = memory[3] > 0.5
    teacher_player_mask = teacher_player[0] > 0.5
    both_depth = (teacher_hit > 0.5) & mem_hit
    depth_err = np.abs(teacher_depth - mem_depth)
    panels = [
        ("rgb", Image.fromarray(rgb_small)),
        ("teacher depth", gray(teacher_depth)),
        ("memory depth", gray(mem_depth)),
        ("depth abs err", depth_error_img(depth_err, both_depth, max_err=0.25)),
        ("teacher hit", gray(teacher_hit)),
        ("memory hit", gray(memory[1])),
        ("hit diff", diff_mask_img(rgb_small, teacher_hit > 0.5, mem_hit)),
        ("teacher semantic", gray(teacher_semantic)),
        ("memory semantic", gray(memory[2])),
        ("teacher other player", mask_img(teacher_player_mask, (255, 190, 40))),
        ("memory other player", mask_img(mem_player, (70, 140, 255))),
        ("player diff", diff_mask_img(rgb_small, teacher_player_mask, mem_player)),
    ]
    pad = 26
    cols = 4
    rows = int(np.ceil(len(panels) / cols))
    canvas = Image.new("RGB", (cols * w, rows * (h + pad) + 22), (250, 250, 247))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, img) in enumerate(panels):
        x = (idx % cols) * w
        y = (idx // cols) * (h + pad)
        canvas.paste(img, (x, y + pad))
        draw.text((x + 8, y + 6), label, fill=(20, 20, 20))
    footer = (
        f"hit_iou={metrics['hit_iou']:.3f} "
        f"depth_mae={metrics['depth_common']['mae']} "
        f"other_player_iou={metrics['other_player_iou']:.3f}"
    )
    draw.text((8, canvas.height - 17), footer, fill=(20, 20, 20))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def sample_metrics(memory: np.ndarray, teacher_depth: np.ndarray, teacher_hit: np.ndarray, teacher_semantic: np.ndarray, teacher_player: np.ndarray) -> dict[str, Any]:
    mem_hit = memory[1] > 0.5
    teacher_hit_b = teacher_hit > 0.5
    common = mem_hit & teacher_hit_b
    mem_player = memory[3] > 0.5
    teacher_player_b = teacher_player[0] > 0.5
    player_common = mem_player & teacher_player_b
    true_positive = int(player_common.sum())
    false_positive = int((mem_player & ~teacher_player_b).sum())
    false_negative = int((teacher_player_b & ~mem_player).sum())
    memory_pixels = int(mem_player.sum())
    teacher_pixels = int(teacher_player_b.sum())
    return {
        "hit_iou": iou(mem_hit, teacher_hit_b),
        "memory_hit_ratio": float(mem_hit.mean()),
        "teacher_hit_ratio": float(teacher_hit_b.mean()),
        "depth_common": mae_stats(memory[0], teacher_depth, common),
        "semantic_binary_iou": iou(memory[2] > 0, teacher_semantic > 0),
        "other_player_iou": iou(mem_player, teacher_player_b),
        "other_player_memory_pixels": memory_pixels,
        "other_player_teacher_pixels": teacher_pixels,
        "other_player_true_positive_pixels": true_positive,
        "other_player_false_positive_pixels": false_positive,
        "other_player_false_negative_pixels": false_negative,
        "other_player_precision": None if memory_pixels == 0 else float(true_positive / memory_pixels),
        "other_player_recall": None if teacher_pixels == 0 else float(true_positive / teacher_pixels),
        "other_player_depth_common": mae_stats(memory[4], teacher_player[1], player_common) if memory.shape[0] > 4 else {"mae": None, "median": None, "p95": None},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--match-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-samples", type=int, default=12)
    ap.add_argument("--make-qa", action="store_true")
    args = ap.parse_args()

    tools_dir = Path(__file__).resolve().parent
    dense_mod = import_tool(tools_dir / "build_dense_condition_v0.py", "dense_condition_v0")
    manifest = load_json(args.manifest)
    rows = []
    missing = []
    for sample in manifest.get("samples", [])[: args.max_samples]:
        episode = sample["episode"]
        ego_stem = sample["ego_stem"]
        frame_index = int(sample["frame_index"])
        episode_dir = args.match_dir / "train" / episode
        required = [
            episode_dir / f"{ego_stem}.mp4",
            episode_dir / f"{ego_stem}_depth.mkv",
            episode_dir / f"{ego_stem}_seg.mkv",
            episode_dir / f"{ego_stem}_seg_colormap.json",
            episode_dir / f"{ego_stem}_player_visibility.json",
            episode_dir / "game_manifest.json",
        ]
        absent = [str(p) for p in required if not p.exists()]
        if absent:
            missing.append({"sample_id": sample["sample_id"], "missing": absent})
            continue
        dense_path = resolve_sample_path(args.manifest, sample, "dense_path", "dense_relpath")
        memory = np.load(dense_path)["dense"].astype(np.float32)
        h, w = memory.shape[1:]
        if memory.shape[0] != 7:
            raise ValueError(f"{sample['sample_id']} expected canonical 7-channel memory dense, got {memory.shape}")
        teacher_depth, teacher_hit, teacher_semantic = teacher_env_channels(dense_mod, episode_dir, ego_stem, frame_index, h, w)
        teacher_player, teacher_meta = teacher_other_player_channels(dense_mod, episode_dir, ego_stem, frame_index, h, w)
        metrics = sample_metrics(memory, teacher_depth, teacher_hit, teacher_semantic, teacher_player)
        row = {
            "sample_id": sample["sample_id"],
            "episode": episode,
            "ego_stem": ego_stem,
            "frame_index": frame_index,
            "visible_teacher_players": len(teacher_meta.get("visible_players", [])),
            **metrics,
        }
        if args.make_qa:
            rgb_bgr = dense_mod.read_frame(episode_dir / f"{ego_stem}.mp4", frame_index)
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            qa_path = args.out_dir / "samples" / sample["sample_id"] / "memory_vs_teacher_channels_v0.png"
            make_sample_qa(qa_path, rgb, memory, teacher_depth, teacher_hit, teacher_semantic, teacher_player, metrics)
            row["qa_path"] = str(qa_path)
        rows.append(row)

    summary = summarize(rows)
    out = {
        "kind": "memory_dense_channels_vs_teacher_v0",
        "manifest": str(args.manifest),
        "match_dir": str(args.match_dir),
        "sample_count": len(rows),
        "missing_count": len(missing),
        "summary": summary,
        "rows": rows,
        "missing": missing,
        "policy": "Teacher streams are used only for QA. Memory dense tensors remain Map-rendered inputs.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "memory_dense_channels_vs_teacher_v0.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_json": str(path), "sample_count": len(rows), "missing_count": len(missing), "summary": summary}, ensure_ascii=False, indent=2))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_of(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def nested_mean(key: str, subkey: str) -> float | None:
        vals = [r[key][subkey] for r in rows if r.get(key, {}).get(subkey) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "hit_iou_mean": mean_of("hit_iou"),
        "semantic_binary_iou_mean": mean_of("semantic_binary_iou"),
        "other_player_iou_mean": mean_of("other_player_iou"),
        "other_player_precision_mean": mean_of("other_player_precision"),
        "other_player_recall_mean": mean_of("other_player_recall"),
        "other_player_memory_pixels_total": int(sum(r.get("other_player_memory_pixels", 0) for r in rows)),
        "other_player_teacher_pixels_total": int(sum(r.get("other_player_teacher_pixels", 0) for r in rows)),
        "other_player_true_positive_pixels_total": int(sum(r.get("other_player_true_positive_pixels", 0) for r in rows)),
        "other_player_false_positive_pixels_total": int(sum(r.get("other_player_false_positive_pixels", 0) for r in rows)),
        "other_player_false_negative_pixels_total": int(sum(r.get("other_player_false_negative_pixels", 0) for r in rows)),
        "depth_common_mae_mean": nested_mean("depth_common", "mae"),
        "other_player_depth_common_mae_mean": nested_mean("other_player_depth_common", "mae"),
    }


if __name__ == "__main__":
    main()
