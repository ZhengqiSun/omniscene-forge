#!/usr/bin/env python3
"""Find first-person frames for teacher-visible or teacher-empty QA.

This is a QA/export helper for Map Memory. It scans real dataset
segmentation/visibility streams and emits a candidate JSON that can be consumed
by ``export_memory_dense_dataset_v0.py``. Teacher streams are used only to pick
frames for evaluation; the exported dense condition is still rendered from Map
Memory.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


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


def player_stems(episode_dir: Path) -> list[str]:
    return [
        p.stem
        for p in sorted(episode_dir.glob("*.json"))
        if "_team_" in p.name and "_player_" in p.name and p.name.endswith("_inst_000.json")
    ]


def alive_frame_ids(episode_dir: Path, ego_stem: str, start: int, stop: int | None) -> set[int]:
    frames = load_json(episode_dir / f"{ego_stem}.json")
    hi = min(len(frames), stop if stop is not None else len(frames))
    out: set[int] = set()
    for idx in range(start, hi):
        frame = frames[idx]
        if float(frame.get("health", 0)) <= 0:
            continue
        cam = frame.get("camera_position")
        if cam is None:
            continue
        try:
            if not np.isfinite(np.asarray(cam, dtype=np.float32)).all():
                continue
        except (TypeError, ValueError):
            continue
        out.add(idx)
    return out


def range_midpoints(visibility_path: Path, min_pixel_percent: float) -> list[tuple[int, float]]:
    vis = load_json(visibility_path)
    frames: dict[int, float] = {}
    for entry in vis.values():
        for row in entry.get("ranges", []):
            score = float(row.get("pixel_percent_max", 0.0))
            if score < min_pixel_percent:
                continue
            lo, hi = row["range"]
            frame = int((int(lo) + int(hi)) // 2)
            frames[frame] = max(frames.get(frame, 0.0), score)
    return sorted(frames.items(), key=lambda item: item[1], reverse=True)


def candidate_frame_ids(
    visibility_path: Path,
    start: int,
    stop: int | None,
    stride: int,
    min_pixel_percent: float,
) -> list[tuple[int, float]]:
    mids = range_midpoints(visibility_path, min_pixel_percent)
    sampled: dict[int, float] = {}
    for mid, score in mids:
        if mid < start:
            continue
        if stop is not None and mid >= stop:
            continue
        offsets = [0, -stride, stride] if stride > 0 else [0]
        for off in offsets:
            frame = mid + off
            if frame >= start and (stop is None or frame < stop):
                sampled[frame] = max(sampled.get(frame, 0.0), score)
    return sorted(sampled.items(), key=lambda item: item[1], reverse=True)


def context_frame_ids(
    alive_ids: set[int],
    start: int,
    stop: int | None,
    stride: int,
    top_k: int,
) -> list[tuple[int, float]]:
    hi = stop if stop is not None else (max(alive_ids) + 1 if alive_ids else start)
    out: list[tuple[int, float]] = []
    for frame in range(start, hi, max(1, stride)):
        if frame in alive_ids:
            out.append((frame, 0.0))
        if len(out) >= top_k:
            break
    return out


def count_teacher_players(
    dense_mod: Any,
    episode_dir: Path,
    ego_stem: str,
    frame_index: int,
    height: int,
    width: int,
    tolerance: int,
) -> tuple[int, int, list[dict[str, Any]]]:
    seg_bgr = dense_mod.read_frame(episode_dir / f"{ego_stem}_seg.mkv", frame_index)
    depth_norm_full = np.zeros(seg_bgr.shape[:2], dtype=np.float32)
    player_channels, meta = dense_mod.build_masks(
        episode_dir,
        ego_stem,
        frame_index,
        seg_bgr,
        depth_norm_full,
        (height, width),
        tolerance=tolerance,
    )
    other_mask = np.maximum(player_channels[0], player_channels[1]) > 0.5
    visible = list(meta.get("visible_players", []))
    return int(other_mask.sum()), len(visible), visible


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-dir", type=Path, required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--ego-limit", type=int, default=10)
    ap.add_argument("--frames-per-ego", type=int, default=6)
    ap.add_argument("--top-k", type=int, default=48)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=1800)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=176)
    ap.add_argument("--tolerance", type=int, default=6)
    ap.add_argument("--min-teacher-pixels", type=int, default=32)
    ap.add_argument("--min-pixel-percent", type=float, default=0.02)
    ap.add_argument("--max-scan-per-ego", type=int, default=24)
    ap.add_argument("--mode", choices=["positive", "context"], default="positive")
    ap.add_argument("--max-teacher-pixels", type=int, default=0)
    args = ap.parse_args()

    tools_dir = Path(__file__).resolve().parent
    dense_mod = import_tool(tools_dir / "build_dense_condition_v0.py", "dense_condition_v0")
    episode_dir = args.match_dir / "train" / args.episode
    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for ego_stem in player_stems(episode_dir)[: args.ego_limit]:
        vis_path = episode_dir / f"{ego_stem}_player_visibility.json"
        if not vis_path.exists():
            missing.append(str(vis_path))
            continue
        alive_ids = alive_frame_ids(episode_dir, ego_stem, args.start, args.stop)
        ego_rows: list[dict[str, Any]] = []
        if args.mode == "positive":
            prefilter = [
                item for item in candidate_frame_ids(
                    vis_path,
                    args.start,
                    args.stop,
                    args.stride,
                    args.min_pixel_percent,
                )
                if item[0] in alive_ids
            ][: args.max_scan_per_ego]
        else:
            prefilter = context_frame_ids(
                alive_ids,
                args.start,
                args.stop,
                args.stride,
                args.max_scan_per_ego,
            )
        for frame_index, visibility_score in prefilter:
            try:
                pixels, player_count, visible = count_teacher_players(
                    dense_mod,
                    episode_dir,
                    ego_stem,
                    frame_index,
                    args.height,
                    args.width,
                    args.tolerance,
                )
            except (FileNotFoundError, IndexError, cv2.error) as exc:
                missing.append(f"{ego_stem}:{frame_index}:{exc}")
                continue
            if args.mode == "positive":
                if pixels < args.min_teacher_pixels:
                    continue
            else:
                if pixels > args.max_teacher_pixels or player_count > 0:
                    continue
            ego_rows.append({
                "ego_stem": ego_stem,
                "frame_index": int(frame_index),
                "teacher_other_player_pixels": int(pixels),
                "visible_teacher_players": int(player_count),
                "visibility_prefilter_score": float(visibility_score),
                "visible_players": visible,
                "score": float(pixels + player_count * 1000) if args.mode == "positive" else float(-pixels),
            })
        ego_rows.sort(key=lambda r: r["score"], reverse=True)
        rows.extend(ego_rows[: args.frames_per_ego])

    rows.sort(key=lambda r: r["score"], reverse=True)
    rows = rows[: args.top_k]
    out = {
        "kind": "teacher_other_player_candidates_v0" if args.mode == "positive" else "teacher_empty_context_candidates_v0",
        "match_dir": str(args.match_dir),
        "episode": args.episode,
        "mode": args.mode,
        "policy": "Teacher RGB/depth/seg/visibility are used only to choose QA frames. Dense inputs must still be rendered from Map Memory.",
        "height": args.height,
        "width": args.width,
        "min_teacher_pixels": args.min_teacher_pixels,
        "max_teacher_pixels": args.max_teacher_pixels,
        "candidate_count": len(rows),
        "missing_count": len(missing),
        "candidates": rows,
        "missing": missing[:50],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out_json": str(args.out_json),
        "candidate_count": len(rows),
        "top": rows[: min(8, len(rows))],
        "missing_count": len(missing),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
