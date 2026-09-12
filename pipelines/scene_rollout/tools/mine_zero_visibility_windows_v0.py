#!/usr/bin/env python3
"""Mine alive five-second windows with no visible players.

The visibility JSON is treated as the source of truth. A window is accepted
only when no recorded visibility interval overlaps any of its 161 raw frames.
Canonical match-level splits are assigned with the project seed so held-out
examples cannot leak into training.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterable


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def stable_bucket(text: str, modulo: int = 10_000) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulo


def canonical_match_split(match_id: str, seed: int) -> str:
    bucket = stable_bucket(f"{seed}|{match_id}")
    return "val" if bucket < 500 else "test" if bucket < 1000 else "train"


def visibility_intervals(payload: dict[str, Any]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for player in payload.values():
        for row in player.get("ranges", []):
            bounds = row.get("range", [])
            if len(bounds) != 2:
                continue
            start, end = int(bounds[0]), int(bounds[1])
            if end >= start:
                intervals.append((start, end))
    intervals.sort()
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def overlaps(intervals: list[tuple[int, int]], start: int, end: int) -> bool:
    for left, right in intervals:
        if left > end:
            return False
        if right >= start:
            return True
    return False


def finite_camera(row: dict[str, Any]) -> bool:
    camera = row.get("camera_position")
    if not isinstance(camera, list) or len(camera) != 3:
        return False
    try:
        return all(math.isfinite(float(value)) for value in camera)
    except (TypeError, ValueError):
        return False


def angular_delta(a: float, b: float) -> float:
    return abs((float(b) - float(a) + 180.0) % 360.0 - 180.0)


def dynamic_score(rows: list[dict[str, Any]], indices: list[int]) -> float:
    sampled = [rows[index] for index in indices]
    travel = 0.0
    for first, second in zip(sampled, sampled[1:]):
        dx = float(second["x"]) - float(first["x"])
        dy = float(second["y"]) - float(first["y"])
        dz = float(second["z"]) - float(first["z"])
        travel += math.sqrt(dx * dx + dy * dy + dz * dz)
    yaw_motion = sum(angular_delta(a["yaw"], b["yaw"]) for a, b in zip(sampled, sampled[1:]))
    pitch_motion = sum(abs(float(b["pitch"]) - float(a["pitch"])) for a, b in zip(sampled, sampled[1:]))
    active_actions = sum(
        any(bool(row.get("action", {}).get(key)) for key in ("forward", "back", "left", "right", "jump", "crouch", "fire"))
        for row in sampled
    )
    return float(travel + 2.0 * yaw_motion + pitch_motion + 5.0 * active_actions)


def iter_candidates(
    visibility_path: Path,
    *,
    dataset_hash: str,
    window_frames: int,
    stride: int,
    split_seed: int,
) -> Iterable[dict[str, Any]]:
    suffix = "_player_visibility.json"
    stem = visibility_path.name[: -len(suffix)]
    episode_dir = visibility_path.parent
    episode = episode_dir.name
    match_id = episode_dir.parent.parent.name
    action_path = episode_dir / f"{stem}.json"
    mp4_path = episode_dir / f"{stem}.mp4"
    episode_info = episode_dir / f"{stem}_episode_info.json"
    video_manifest = episode_dir / f"{stem}_video_manifest.json"
    game_manifest = episode_dir / "game_manifest.json"
    world_events = episode_dir / "world_events.jsonl"
    required = [action_path, mp4_path, episode_info, video_manifest, game_manifest, world_events]
    if any(not path.is_file() for path in required):
        return

    intervals = visibility_intervals(load_json(visibility_path))
    rows = load_json(action_path)
    if not isinstance(rows, list) or len(rows) < window_frames:
        return
    latent_offsets = list(range(0, window_frames, 8))
    video_offsets = list(range(0, window_frames, 2))
    split = canonical_match_split(match_id, split_seed)

    for raw_start in range(0, len(rows) - window_frames + 1, stride):
        raw_end = raw_start + window_frames - 1
        if overlaps(intervals, raw_start, raw_end):
            continue
        latent_indices = [raw_start + offset for offset in latent_offsets]
        full_window = rows[raw_start : raw_end + 1]
        if any(float(row.get("health", 0) or 0) <= 0 for row in full_window):
            continue
        if any(not finite_camera(rows[index]) for index in latent_indices):
            continue
        if any(not all(math.isfinite(float(row.get(key, float("nan")))) for key in ("x", "y", "z", "yaw", "pitch")) for row in (rows[index] for index in latent_indices)):
            continue

        clip_id = f"{dataset_hash}_{match_id}_{episode}_{stem}_{raw_start:07d}"
        yield {
            "clip_id": clip_id,
            "hash": dataset_hash,
            "game_id": match_id,
            "episode": episode,
            "player_stem": stem,
            "mp4": str(mp4_path),
            "action_json": str(action_path),
            "episode_info": str(episode_info),
            "video_manifest": str(video_manifest),
            "game_manifest": str(game_manifest),
            "world_events": str(world_events),
            "player_visibility": str(visibility_path),
            "map_name": "de_dust2",
            "raw_start": raw_start,
            "raw_indices": [raw_start + offset for offset in video_offsets],
            "frame_count_start": raw_start,
            "frame_count_end": raw_end,
            "window_stride_raw": stride,
            "dynamic_score": dynamic_score(rows, latent_indices),
            "prompt": "first-person gameplay video in Counter-Strike, de_dust2 map",
            "positive_latent_frame_count": len(latent_indices),
            "positive_latent_frames": latent_indices,
            "latent_frame_count": len(latent_indices),
            "latent_frames": latent_indices,
            "selection_role": "context",
            "map_memory_selection_roles": ["context"] * len(latent_indices),
            "map_memory_positive_frames": 0,
            "map_memory_context_frames": len(latent_indices),
            "max_visibility_pixel_percent": 0.0,
            "visible_entity_count": 0,
            "selection_metric_note": "strict zero-visibility interval scan; full-window alive and finite-camera checks",
            "pose_version": "csgoflu_to_opencv_c2w_v1",
            "hfov_source": "fixed_unscoped_hfov_106.26",
            "map_memory_split": split,
            "map_memory_split_key": "match",
            "zero_visibility_contract": {
                "window_raw_frames": window_frames,
                "visibility_interval_overlap_count": 0,
                "alive_checked_on_all_raw_frames": True,
                "camera_checked_on_latent_frames": True,
            },
        }


def select_for_match(
    files: list[Path],
    *,
    quota: int,
    per_ego_limit: int,
    rng: random.Random,
    dataset_hash: str,
    window_frames: int,
    stride: int,
    split_seed: int,
) -> tuple[list[dict[str, Any]], int, list[dict[str, str]]]:
    if quota <= 0:
        return [], 0, []
    rng.shuffle(files)
    selected: list[dict[str, Any]] = []
    candidate_count = 0
    errors: list[dict[str, str]] = []
    for visibility_path in files:
        try:
            candidates = list(iter_candidates(
                visibility_path,
                dataset_hash=dataset_hash,
                window_frames=window_frames,
                stride=stride,
                split_seed=split_seed,
            ))
        except Exception as exc:
            errors.append({"path": str(visibility_path), "error": repr(exc)})
            continue
        candidate_count += len(candidates)
        candidates.sort(key=lambda row: (-float(row["dynamic_score"]), int(row["raw_start"])))
        if len(candidates) > per_ego_limit:
            candidates = candidates[:per_ego_limit]
        selected.extend(candidates)
        if len(selected) >= quota:
            break
    selected.sort(key=lambda row: (-float(row["dynamic_score"]), str(row["clip_id"])))
    return selected[:quota], candidate_count, errors


def distribute(total: int, match_ids: list[str]) -> dict[str, int]:
    if not match_ids:
        return {}
    base, remainder = divmod(total, len(match_ids))
    return {match_id: base + (index < remainder) for index, match_id in enumerate(match_ids)}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-target", type=int, default=18_000)
    parser.add_argument("--heldout-target", type=int, default=2_000)
    parser.add_argument("--per-ego-limit", type=int, default=4)
    parser.add_argument("--window-frames", type=int, default=161)
    parser.add_argument("--stride", type=int, default=162)
    parser.add_argument("--split-seed", type=int, default=20_260_531)
    parser.add_argument("--selection-seed", type=int, default=20_260_716)
    args = parser.parse_args()

    started = time.time()
    dataset_hash = args.dataset_root.name
    visibility_files = sorted(args.dataset_root.glob("*/train/Ep_*/*_player_visibility.json"))
    by_match: dict[str, list[Path]] = defaultdict(list)
    for path in visibility_files:
        by_match[path.parent.parent.parent.name].append(path)

    train_matches = sorted(match_id for match_id in by_match if canonical_match_split(match_id, args.split_seed) == "train")
    heldout_matches = sorted(match_id for match_id in by_match if canonical_match_split(match_id, args.split_seed) != "train")
    quotas = {
        **distribute(args.train_target, train_matches),
        **distribute(args.heldout_target, heldout_matches),
    }
    rng = random.Random(args.selection_seed)
    selected: list[dict[str, Any]] = []
    scanned_candidates: dict[str, int] = {}
    errors: list[dict[str, str]] = []
    for index, match_id in enumerate(train_matches + heldout_matches, start=1):
        rows, candidate_count, match_errors = select_for_match(
            list(by_match[match_id]),
            quota=quotas[match_id],
            per_ego_limit=args.per_ego_limit,
            rng=rng,
            dataset_hash=dataset_hash,
            window_frames=args.window_frames,
            stride=args.stride,
            split_seed=args.split_seed,
        )
        selected.extend(rows)
        scanned_candidates[match_id] = candidate_count
        errors.extend(match_errors)
        print(json.dumps({
            "progress": f"{index}/{len(train_matches) + len(heldout_matches)}",
            "match_id": match_id,
            "split": canonical_match_split(match_id, args.split_seed),
            "selected": len(rows),
            "quota": quotas[match_id],
        }), flush=True)

    selected.sort(key=lambda row: (row["map_memory_split"] != "train", row["game_id"], row["clip_id"]))
    for index, row in enumerate(selected):
        row["source_manifest_index"] = index
        row["zero_visibility_selection_source"] = "all188_strict_zero_visibility_v0"

    train_rows = [row for row in selected if row["map_memory_split"] == "train"]
    heldout_rows = [row for row in selected if row["map_memory_split"] != "train"]
    all_ids = [row["clip_id"] for row in selected]
    split_match_sets = {
        split: {row["game_id"] for row in selected if row["map_memory_split"] == split}
        for split in ("train", "val", "test")
    }
    leakage = bool(split_match_sets["train"] & (split_match_sets["val"] | split_match_sets["test"]))
    split_counts = Counter(row["map_memory_split"] for row in selected)
    match_counts = Counter(row["game_id"] for row in selected)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "zero_visibility_train_source_manifest_v0.jsonl", train_rows)
    write_jsonl(args.out_dir / "zero_visibility_heldout_exam_v0.jsonl", heldout_rows)
    write_jsonl(args.out_dir / "zero_visibility_all_source_manifest_v0.jsonl", selected)
    report = {
        "kind": "zero_visibility_window_mining_report_v0",
        "status": "pass" if len(train_rows) >= args.train_target and len(heldout_rows) >= args.heldout_target and not leakage and len(all_ids) == len(set(all_ids)) else "fail",
        "dataset_root": str(args.dataset_root),
        "contract": {
            "window_raw_frames": args.window_frames,
            "stride_raw_frames": args.stride,
            "visibility": "no recorded player visibility interval overlaps the full window",
            "alive": "ego health > 0 on every raw frame",
            "camera": "finite camera and pose on all 21 latent-aligned frames",
            "split": f"canonical match split, blake2b seed {args.split_seed}",
        },
        "targets": {"train": args.train_target, "heldout": args.heldout_target},
        "selected": {"total": len(selected), "train": len(train_rows), "heldout": len(heldout_rows)},
        "split_counts": dict(sorted(split_counts.items())),
        "match_counts": dict(sorted(match_counts.items())),
        "train_match_count": len(train_matches),
        "heldout_match_count": len(heldout_matches),
        "split_match_leakage": leakage,
        "duplicate_clip_ids": len(all_ids) - len(set(all_ids)),
        "visibility_files_found": len(visibility_files),
        "visibility_files_processed_or_skipped_after_quota": sum(len(by_match[key]) for key in by_match),
        "scanned_candidate_counts_by_match": scanned_candidates,
        "error_count": len(errors),
        "error_examples": errors[:50],
        "dynamic_score": {
            "train_min": min((row["dynamic_score"] for row in train_rows), default=None),
            "train_median": sorted((row["dynamic_score"] for row in train_rows))[len(train_rows) // 2] if train_rows else None,
            "heldout_min": min((row["dynamic_score"] for row in heldout_rows), default=None),
        },
        "elapsed_seconds": time.time() - started,
    }
    report_path = args.out_dir / "zero_visibility_mining_report_v0.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
