#!/usr/bin/env python3
"""Build a sharded visible-multiplayer candidate manifest.

The scanner is intentionally metadata-only: it reads player state, visibility
JSON, and file existence, but never decodes RGB/depth/seg videos.  It is meant
as the data gate before scaling Map Memory dense-adapter training beyond the
small fullsubset release.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PLAYER_RE = re.compile(r"^(?P<episode>Ep_\d+)_team_(?P<team>\d+)_player_(?P<player>\d+)_inst_(?P<inst>\d+)$")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_player_stem(stem: str) -> dict[str, Any] | None:
    m = PLAYER_RE.match(stem)
    if not m:
        return None
    return {
        "episode": m.group("episode"),
        "team": int(m.group("team")),
        "player": int(m.group("player")),
        "inst": int(m.group("inst")),
    }


def player_json_paths(episode_dir: Path) -> list[Path]:
    out: list[Path] = []
    for path in sorted(episode_dir.glob("*.json")):
        parsed = parse_player_stem(path.stem)
        if parsed is not None:
            out.append(path)
    return out


def player_visibility_paths(episode_dir: Path) -> list[Path]:
    return sorted(episode_dir.glob("*_player_visibility.json"))


def stem_from_visibility_path(path: Path) -> str:
    suffix = "_player_visibility"
    stem = path.stem
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def estimate_frame_count(episode_dir: Path, ego_stem: str) -> int | None:
    info_path = episode_dir / f"{ego_stem}_episode_info.json"
    if not info_path.exists():
        return None
    try:
        info = load_json(info_path)
    except (OSError, json.JSONDecodeError):
        return None
    try:
        start_tick = int(info["start_tick"])
        end_tick = int(info["end_tick"])
        tf_ratio = int(info.get("tf_ratio") or 4)
    except (KeyError, TypeError, ValueError):
        return None
    if tf_ratio <= 0 or end_tick < start_tick:
        return None
    return int(math.ceil((end_tick - start_tick + 1) / tf_ratio))


def any_child(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        next(path.iterdir())
        return True
    except StopIteration:
        return False
    except OSError:
        return False


def match_geometry_availability(match_dir: Path) -> dict[str, bool]:
    meshes_dir = match_dir / "meshes"
    return {
        "navmesh": (match_dir / "navmesh.json").exists(),
        "mesh_manifest": (match_dir / "mesh_manifest.json").exists(),
        "static_props": (match_dir / "static_props.json").exists(),
        "meshes_dir": meshes_dir.is_dir(),
        "meshes_obj_any": any_child(meshes_dir),
    }


def ego_core_availability(episode_dir: Path, ego_stem: str) -> dict[str, bool]:
    return {
        "ego_player_json": (episode_dir / f"{ego_stem}.json").exists(),
        "ego_rgb_mp4": (episode_dir / f"{ego_stem}.mp4").exists(),
        "ego_episode_info": (episode_dir / f"{ego_stem}_episode_info.json").exists(),
        "ego_video_manifest": (episode_dir / f"{ego_stem}_video_manifest.json").exists(),
        "ego_player_visibility": (episode_dir / f"{ego_stem}_player_visibility.json").exists(),
        "game_manifest": (episode_dir / "game_manifest.json").exists(),
        "world_events": (episode_dir / "world_events.jsonl").exists(),
    }


def ego_teacher_availability(episode_dir: Path, ego_stem: str) -> dict[str, bool]:
    return {
        "ego_depth_mkv": (episode_dir / f"{ego_stem}_depth.mkv").exists(),
        "ego_seg_mkv": (episode_dir / f"{ego_stem}_seg.mkv").exists(),
        "ego_seg_colormap_json": (episode_dir / f"{ego_stem}_seg_colormap.json").exists(),
        "ego_motion_bin": (episode_dir / f"{ego_stem}_motion.bin").exists(),
        "ego_hud_mkv": (episode_dir / f"{ego_stem}_hud.mkv").exists(),
    }


def iter_match_dirs(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"root does not exist: {root}")
    return [p for p in sorted(root.iterdir()) if p.is_dir()]


def iter_episode_dirs(match_dir: Path, split: str) -> list[Path]:
    split_dir = match_dir / split
    if not split_dir.is_dir():
        return []
    return [p for p in sorted(split_dir.iterdir()) if p.is_dir() and p.name.startswith("Ep_")]


def add_visibility_ranges(frame_targets: dict[int, dict[int, float]], target_idx: int, rows: Iterable[dict[str, Any]]) -> None:
    for row in rows:
        try:
            lo, hi = row["range"]
            lo_i = max(0, int(lo))
            hi_i = int(hi)
            score = float(row.get("pixel_percent_max", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if hi_i < lo_i:
            continue
        for frame in range(lo_i, hi_i + 1):
            targets = frame_targets.setdefault(frame, {})
            if score > targets.get(target_idx, -1.0):
                targets[target_idx] = score


def load_episode_visibility(
    episode_dir: Path,
    player_meta_by_idx: dict[int, dict[str, Any]],
    errors: list[str],
) -> dict[int, dict[int, dict[int, float]]]:
    """Return ego_idx -> frame -> target_idx -> max pixel percent."""
    out: dict[int, dict[int, dict[int, float]]] = {}
    for path in player_visibility_paths(episode_dir):
        ego_stem = stem_from_visibility_path(path)
        parsed = parse_player_stem(ego_stem)
        if parsed is None:
            continue
        ego_idx = int(parsed["player"])
        player_meta_by_idx.setdefault(ego_idx, {"stem": ego_stem, "team": int(parsed["team"]), "player": ego_idx})
        frame_targets: dict[int, dict[int, float]] = {}
        try:
            visibility = load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        if not isinstance(visibility, dict):
            errors.append(f"{path}: expected object")
            continue
        for target_key, entry in visibility.items():
            try:
                target_idx = int(target_key)
            except (TypeError, ValueError):
                continue
            if target_idx == ego_idx:
                continue
            if not isinstance(entry, dict):
                continue
            add_visibility_ranges(frame_targets, target_idx, entry.get("ranges", []))
        out[ego_idx] = frame_targets
    return out


def update_bool_counts(acc: dict[str, Counter], namespace: str, values: dict[str, bool]) -> None:
    for key, value in values.items():
        acc[f"{namespace}.{key}"]["true" if value else "false"] += 1


def top_heap_key(row: dict[str, Any]) -> tuple[float, int, int, int, int, int]:
    pixel_sum = float(sum(float(v) for v in row.get("visible_pixel_percent_max_by_player", {}).values()))
    return (
        int(row.get("visible_player_count", 0)),
        int(row.get("mutual_visible_player_count", 0)),
        int(row.get("cross_team_visible_player_count", 0)),
        int(row.get("pairwise_directed_edge_count_at_frame", 0)),
        pixel_sum,
        -int(row.get("frame", 0)),
    )


class CandidateWriter:
    def __init__(self, out_jsonl: Path, limit: int | None):
        self.out_jsonl = out_jsonl
        self.limit = limit if limit and limit > 0 else None
        self.written = 0
        self.seq = 0
        self.heap: list[tuple[tuple[float, int, int, int, int, int], int, dict[str, Any]]] = []
        self.handle = None
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if self.limit is None:
            self.handle = out_jsonl.open("w", encoding="utf-8")

    def add(self, row: dict[str, Any]) -> None:
        if self.limit is None:
            assert self.handle is not None
            self.handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.written += 1
            return
        item = (top_heap_key(row), self.seq, row)
        self.seq += 1
        if len(self.heap) < self.limit:
            heapq.heappush(self.heap, item)
        else:
            if item[0] > self.heap[0][0]:
                heapq.heapreplace(self.heap, item)

    def close(self) -> int:
        if self.handle is not None:
            self.handle.close()
            return self.written
        rows = [item[2] for item in sorted(self.heap, key=lambda item: (item[0], -item[1]), reverse=True)]
        with self.out_jsonl.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.written = len(rows)
        return self.written


def build_row(
    *,
    dataset_label: str,
    hash_id: str,
    match_dir: Path,
    episode_dir: Path,
    frame: int,
    raw_start: int,
    raw_end: int,
    window_frames: int,
    ego_idx: int,
    ego_meta: dict[str, Any],
    visible_targets: dict[int, float],
    per_ego_frame_targets: dict[int, dict[int, dict[int, float]]],
    player_meta_by_idx: dict[int, dict[str, Any]],
    geometry: dict[str, bool],
    core: dict[str, bool],
    teacher: dict[str, bool],
) -> dict[str, Any]:
    visible_indices = sorted(visible_targets, key=lambda idx: visible_targets[idx], reverse=True)
    visible_stems = [player_meta_by_idx.get(idx, {}).get("stem") for idx in visible_indices]
    ego_team = ego_meta.get("team")
    mutual_indices = [
        idx
        for idx in visible_indices
        if ego_idx in per_ego_frame_targets.get(idx, {}).get(frame, {})
    ]
    cross_team_indices = [
        idx
        for idx in visible_indices
        if player_meta_by_idx.get(idx, {}).get("team") is not None and player_meta_by_idx.get(idx, {}).get("team") != ego_team
    ]
    pairwise_edges = sum(len(frame_targets.get(frame, {})) for frame_targets in per_ego_frame_targets.values())
    ego_stem = str(ego_meta.get("stem") or f"player_{ego_idx}")
    sample_id = f"{dataset_label}__{match_dir.name}__{episode_dir.name}__{ego_stem}__f{frame:06d}"
    return {
        "sample_id": sample_id,
        "dataset_label": dataset_label,
        "hash": hash_id,
        "match": match_dir.name,
        "match_id": match_dir.name,
        "episode": episode_dir.name,
        "frame": int(frame),
        "raw_start": int(raw_start),
        "raw_end": int(raw_end),
        "raw_start_proposed_81f": int(raw_start) if window_frames == 81 else None,
        "raw_end_proposed_81f": int(raw_end) if window_frames == 81 else None,
        "window_frames": int(window_frames),
        "ego": {
            "player_index": int(ego_idx),
            "stem": ego_stem,
            "team": ego_team,
        },
        "ego_player_index": int(ego_idx),
        "ego_stem": ego_stem,
        "ego_team": ego_team,
        "visible_players": [
            {
                "player_index": int(idx),
                "stem": player_meta_by_idx.get(idx, {}).get("stem"),
                "team": player_meta_by_idx.get(idx, {}).get("team"),
                "pixel_percent_max": float(visible_targets[idx]),
                "mutual_visible": idx in mutual_indices,
            }
            for idx in visible_indices
        ],
        "visible_player_count": len(visible_indices),
        "visible_player_indices": [int(idx) for idx in visible_indices],
        "visible_player_stems": visible_stems,
        "visible_pixel_percent_max_by_player": {str(idx): float(visible_targets[idx]) for idx in visible_indices},
        "mutual_edges": [
            {"a": int(ego_idx), "b": int(idx), "frame": int(frame)}
            for idx in mutual_indices
        ],
        "mutual_visible_player_count": len(mutual_indices),
        "mutual_visible_player_indices": [int(idx) for idx in mutual_indices],
        "cross_team_count": len(cross_team_indices),
        "cross_team_visible_player_count": len(cross_team_indices),
        "pairwise_directed_edge_count_at_frame": int(pairwise_edges),
        "core_files": core,
        "core_files_available": core,
        "teacher_files": teacher,
        "teacher_files_available": teacher,
        "geometry_files": geometry,
        "geometry_assets_available": geometry,
        "source_episode_dir": str(episode_dir),
        "source_visibility_json": str(episode_dir / f"{ego_stem}_player_visibility.json"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="Hash root containing match directories.")
    ap.add_argument("--out-jsonl", type=Path, required=True)
    ap.add_argument("--out-summary", type=Path, required=True)
    ap.add_argument("--min-visible-players", type=int, default=2)
    ap.add_argument("--window-frames", type=int, default=81)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--max-matches", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="If >0, write only top-N rows while still counting all candidates.")
    ap.add_argument("--dataset-label", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--progress-every", type=int, default=10)
    args = ap.parse_args()

    if args.min_visible_players < 1:
        raise ValueError("--min-visible-players must be >= 1")
    if args.window_frames < 1 or args.window_frames % 2 == 0:
        raise ValueError("--window-frames must be a positive odd integer")
    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < count")

    root = args.root.resolve()
    dataset_label = args.dataset_label or root.name
    hash_id = root.name
    all_matches = iter_match_dirs(root)
    shard_matches = [p for i, p in enumerate(all_matches) if i % args.shard_count == args.shard_index]
    if args.max_matches and args.max_matches > 0:
        shard_matches = shard_matches[: args.max_matches]

    writer = CandidateWriter(args.out_jsonl, args.limit)
    half = args.window_frames // 2
    started = time.time()
    summary: dict[str, Any] = {
        "kind": "visible_multiplayer_manifest_summary_v0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "dataset_label": dataset_label,
        "hash": hash_id,
        "split": args.split,
        "args": {
            "min_visible_players": args.min_visible_players,
            "window_frames": args.window_frames,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "max_matches": args.max_matches,
            "limit": args.limit,
        },
        "total_match_dirs": len(all_matches),
        "selected_match_dirs": len(shard_matches),
        "matches_scanned": 0,
        "episodes_seen": 0,
        "episodes_with_visibility": 0,
        "visibility_files_scanned": 0,
        "player_json_files_seen": 0,
        "candidate_count_total": 0,
        "candidate_count_written": 0,
        "visible_player_count_hist": {},
        "mutual_visible_player_count_hist": {},
        "cross_team_visible_player_count_hist": {},
        "file_availability_weighted_by_candidate": {},
        "match_geometry_availability_by_match": {},
        "errors": [],
        "out_jsonl": str(args.out_jsonl),
        "out_summary": str(args.out_summary),
    }
    visible_hist: Counter = Counter()
    mutual_hist: Counter = Counter()
    cross_team_hist: Counter = Counter()
    file_counts: dict[str, Counter] = defaultdict(Counter)
    match_geometry_counts: dict[str, Counter] = defaultdict(Counter)
    errors: list[str] = []

    try:
        for match_i, match_dir in enumerate(shard_matches):
            geometry = match_geometry_availability(match_dir)
            for key, value in geometry.items():
                match_geometry_counts[key]["true" if value else "false"] += 1
            episodes = iter_episode_dirs(match_dir, args.split)
            summary["matches_scanned"] += 1
            summary["episodes_seen"] += len(episodes)
            for episode_dir in episodes:
                pjsons = player_json_paths(episode_dir)
                summary["player_json_files_seen"] += len(pjsons)
                player_meta_by_idx: dict[int, dict[str, Any]] = {}
                for pjson in pjsons:
                    parsed = parse_player_stem(pjson.stem)
                    if parsed is None:
                        continue
                    player_meta_by_idx[int(parsed["player"])] = {
                        "stem": pjson.stem,
                        "team": int(parsed["team"]),
                        "player": int(parsed["player"]),
                    }
                per_ego = load_episode_visibility(episode_dir, player_meta_by_idx, errors)
                if not per_ego:
                    continue
                summary["episodes_with_visibility"] += 1
                summary["visibility_files_scanned"] += len(per_ego)
                for ego_idx, frame_targets in per_ego.items():
                    ego_meta = player_meta_by_idx.get(ego_idx)
                    if ego_meta is None:
                        continue
                    ego_stem = str(ego_meta["stem"])
                    core = ego_core_availability(episode_dir, ego_stem)
                    teacher = ego_teacher_availability(episode_dir, ego_stem)
                    frame_count = estimate_frame_count(episode_dir, ego_stem)
                    for frame, targets in sorted(frame_targets.items()):
                        visible_targets = {idx: score for idx, score in targets.items() if idx != ego_idx}
                        if len(visible_targets) < args.min_visible_players:
                            continue
                        raw_start = int(frame) - half
                        raw_end = raw_start + args.window_frames - 1
                        if raw_start < 0:
                            continue
                        if frame_count is not None and raw_end >= frame_count:
                            continue
                        row = build_row(
                            dataset_label=dataset_label,
                            hash_id=hash_id,
                            match_dir=match_dir,
                            episode_dir=episode_dir,
                            frame=int(frame),
                            raw_start=raw_start,
                            raw_end=raw_end,
                            window_frames=args.window_frames,
                            ego_idx=ego_idx,
                            ego_meta=ego_meta,
                            visible_targets=visible_targets,
                            per_ego_frame_targets=per_ego,
                            player_meta_by_idx=player_meta_by_idx,
                            geometry=geometry,
                            core=core,
                            teacher=teacher,
                        )
                        summary["candidate_count_total"] += 1
                        visible_hist[str(row["visible_player_count"])] += 1
                        mutual_hist[str(row["mutual_visible_player_count"])] += 1
                        cross_team_hist[str(row["cross_team_visible_player_count"])] += 1
                        update_bool_counts(file_counts, "core", core)
                        update_bool_counts(file_counts, "teacher", teacher)
                        update_bool_counts(file_counts, "geometry", geometry)
                        writer.add(row)
            if args.progress_every > 0 and ((match_i + 1) % args.progress_every == 0 or match_i + 1 == len(shard_matches)):
                elapsed = time.time() - started
                print(
                    json.dumps(
                        {
                            "progress": f"{match_i + 1}/{len(shard_matches)}",
                            "matches_scanned": summary["matches_scanned"],
                            "episodes_seen": summary["episodes_seen"],
                            "candidates": summary["candidate_count_total"],
                            "elapsed_sec": round(elapsed, 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        summary["candidate_count_written"] = writer.close()

    elapsed = time.time() - started
    summary["elapsed_sec"] = round(elapsed, 3)
    summary["visible_player_count_hist"] = dict(sorted(visible_hist.items(), key=lambda kv: int(kv[0])))
    summary["mutual_visible_player_count_hist"] = dict(sorted(mutual_hist.items(), key=lambda kv: int(kv[0])))
    summary["cross_team_visible_player_count_hist"] = dict(sorted(cross_team_hist.items(), key=lambda kv: int(kv[0])))
    summary["file_availability_weighted_by_candidate"] = {
        key: dict(value)
        for key, value in sorted(file_counts.items())
    }
    summary["match_geometry_availability_by_match"] = {
        key: dict(value)
        for key, value in sorted(match_geometry_counts.items())
    }
    summary["errors"] = errors[:100]
    summary["error_count"] = len(errors)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
