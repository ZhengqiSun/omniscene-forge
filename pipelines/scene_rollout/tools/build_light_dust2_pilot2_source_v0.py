#!/usr/bin/env python3
"""Build and materialize a light-dust2 cross-session pilot2 source manifest.

This is a scoped Memory-side helper for the June 2026 sprint.  It mirrors the
pilot1 source contract: 81 RGB frames at 832x480/16fps, 81 camera poses and
intrinsics, and a JSONL source manifest with frame-exact raw indices.
"""

from __future__ import annotations
from runtime_paths import source_path

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np
from PIL import Image


PROMPT = "first-person gameplay video in Counter-Strike, de_dust2 map"
INTRINSICS_832_480 = np.asarray([312.00115966796875, 320.0011901855469, 416.0, 240.0], dtype=np.float32)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_dust2_match(match_dir: Path) -> bool:
    probes = list(match_dir.glob("train/Ep_*/game_manifest.json"))[:1]
    probes += list(match_dir.glob("train/Ep_*/*_episode_info.json"))[:1]
    for path in probes:
        try:
            if "dust2" in json.dumps(load_json(path), ensure_ascii=False).lower():
                return True
        except Exception:
            continue
    return False


def video_contract_ok(path: Path) -> bool:
    try:
        data = load_json(path)
    except Exception:
        return False
    streams = data if isinstance(data, list) else [data]
    return any(
        int(stream.get("width", 0)) == 1280
        and int(stream.get("height", 0)) == 720
        and abs(float(stream.get("fps", 0.0)) - 32.0) < 1e-3
        for stream in streams
    )


def player_index_from_stem(stem: str) -> str | None:
    match = re.search(r"player_(\d+)_", stem)
    if not match:
        return None
    return str(int(match.group(1)))


def pose_rotation_from_action_frame(frame: dict[str, Any]) -> np.ndarray:
    rotation = frame.get("camera_rotation") or [0.0, 0.0, 0.0]
    pitch = math.radians(float(rotation[1]))
    yaw = math.radians(float(rotation[2]))
    sy = math.sin(yaw)
    cy = math.cos(yaw)
    sp = math.sin(pitch)
    cp = math.cos(pitch)
    return np.asarray(
        [
            [sy, -cy * sp, cy * cp],
            [-cy, -sy * sp, sy * cp],
            [0.0, -cp, -sp],
        ],
        dtype=np.float32,
    )


def raw_indices_for_window(raw_start: int, *, video_frames: int, raw_stride: int) -> list[int]:
    return [int(raw_start + raw_stride * i) for i in range(video_frames)]


def latent_positions(video_frames: int, latent_frames: int) -> list[int]:
    vals = np.rint(np.linspace(0, video_frames - 1, latent_frames)).astype(np.int64)
    return [int(x) for x in vals.tolist()]


def visibility_intervals(visibility: dict[str, Any], ego_index: str | None) -> list[tuple[int, int, float, str]]:
    intervals: list[tuple[int, int, float, str]] = []
    for ent, payload in visibility.items():
        try:
            ent_norm = str(int(ent))
        except Exception:
            ent_norm = str(ent)
        if ego_index is not None and ent_norm == ego_index:
            continue
        for item in (payload or {}).get("ranges") or []:
            rr = item.get("range") or []
            if len(rr) != 2:
                continue
            a, b = int(rr[0]), int(rr[1])
            if b <= a:
                continue
            intervals.append((a, b, float(item.get("pixel_percent_max") or 0.0), ent_norm))
    return intervals


def positive_latent_frames(
    raw_frames: list[int],
    *,
    video_frames: int,
    latent_frames: int,
    intervals: list[tuple[int, int, float, str]],
) -> tuple[list[int], float, int]:
    positives: list[int] = []
    max_pct = 0.0
    visible_entities: set[str] = set()
    for pos in latent_positions(video_frames, latent_frames):
        frame = raw_frames[pos]
        hit = False
        for a, b, pct, ent in intervals:
            if a <= frame <= b:
                hit = True
                max_pct = max(max_pct, pct)
                visible_entities.add(ent)
        if hit:
            positives.append(frame)
    return positives, max_pct, len(visible_entities)


def candidate_rows_for_match(
    *,
    raw_root: Path,
    session: str,
    match_dir: Path,
    video_frames: int,
    raw_stride: int,
    latent_frames: int,
    window_stride_raw: int,
    max_rows: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode_dir in sorted((match_dir / "train").glob("Ep_*")):
        for action_json in sorted(episode_dir.glob("*_inst_000.json")):
            stem = action_json.name[:-5]
            video_manifest = episode_dir / f"{stem}_video_manifest.json"
            visibility_path = episode_dir / f"{stem}_player_visibility.json"
            mp4 = episode_dir / f"{stem}.mp4"
            episode_info = episode_dir / f"{stem}_episode_info.json"
            game_manifest = episode_dir / "game_manifest.json"
            world_events = episode_dir / "world_events.jsonl"
            if not (video_manifest.exists() and visibility_path.exists() and mp4.exists()):
                continue
            if not video_contract_ok(video_manifest):
                continue
            try:
                frames = load_json(action_json)
                visibility = load_json(visibility_path)
            except Exception:
                continue
            if not frames or not isinstance(frames, list):
                continue
            first = frames[0]
            required = ["x", "y", "z", "yaw", "pitch", "camera_position", "camera_rotation", "action"]
            if not all(key in first for key in required):
                continue
            intervals = visibility_intervals(visibility, player_index_from_stem(stem))
            if not intervals:
                continue
            max_start = len(frames) - ((video_frames - 1) * raw_stride) - 1
            raw_start = 16
            while raw_start <= max_start:
                raw_frames = raw_indices_for_window(raw_start, video_frames=video_frames, raw_stride=raw_stride)
                positives, max_pct, visible_count = positive_latent_frames(
                    raw_frames,
                    video_frames=video_frames,
                    latent_frames=latent_frames,
                    intervals=intervals,
                )
                if positives:
                    clip_id = f"{session}_{match_dir.name}_{episode_dir.name}_{stem}_{raw_start:07d}"
                    rows.append(
                        {
                            "clip_id": clip_id,
                            "hash": session,
                            "game_id": match_dir.name,
                            "episode": episode_dir.name,
                            "player_stem": stem,
                            "mp4": str(mp4),
                            "action_json": str(action_json),
                            "episode_info": str(episode_info),
                            "video_manifest": str(video_manifest),
                            "game_manifest": str(game_manifest),
                            "world_events": str(world_events),
                            "player_visibility": str(visibility_path),
                            "map_name": "de_dust2",
                            "raw_start": int(raw_start),
                            "raw_indices": raw_frames,
                            "frame_count_start": int(raw_frames[0]),
                            "frame_count_end": int(raw_frames[-1]),
                            "window_stride_raw": int(window_stride_raw),
                            "dynamic_score": float(len(positives) * 1000.0 + max_pct),
                            "prompt": PROMPT,
                            "positive_latent_frame_count": int(len(positives)),
                            "positive_latent_frames": positives,
                            "max_visibility_pixel_percent": float(max_pct),
                            "visible_entity_count": int(visible_count),
                            "pose_version": "csgoflu_to_opencv_c2w_v1",
                            "hfov_source": "fixed_unscoped_hfov_106.26",
                        }
                    )
                    if max_rows is not None and len(rows) >= max_rows:
                        rows.sort(key=lambda row: (float(row["dynamic_score"]), int(row["raw_start"])), reverse=True)
                        return rows
                raw_start += window_stride_raw
    rows.sort(key=lambda row: (float(row["dynamic_score"]), int(row["raw_start"])), reverse=True)
    return rows


def select_matches(args: argparse.Namespace) -> list[tuple[str, Path, str]]:
    plan: list[tuple[str, int, str]] = []
    for item in args.session_plan:
        session, count, split = item.split(":")
        plan.append((session, int(count), split))
    selected: list[tuple[str, Path, str]] = []
    for session, count, split in plan:
        session_dir = args.raw_root / session
        dust2 = [path for path in sorted(session_dir.iterdir()) if path.is_dir() and is_dust2_match(path)]
        if len(dust2) < count:
            raise ValueError(f"{session}: requested {count} dust2 matches, found {len(dust2)}")
        selected.extend((session, path, split) for path in dust2[:count])
    return selected


def read_frames_direct_resize(mp4: Path, raw_indices: list[int], *, width: int, height: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {mp4}")
    frames: list[np.ndarray] = []
    start, end = raw_indices[0], raw_indices[-1]
    wanted = set(raw_indices)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for raw in range(start, end + 1):
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"{mp4}: failed to read raw frame {raw}")
        if raw in wanted:
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            frames.append(resized)
    cap.release()
    if len(frames) != len(raw_indices):
        raise RuntimeError(f"{mp4}: got {len(frames)} frames, expected {len(raw_indices)}")
    return frames


def materialize(row: dict[str, Any], *, sample_dir: Path, width: int, height: int, fps: float) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    frames_bgr = read_frames_direct_resize(Path(row["mp4"]), [int(x) for x in row["raw_indices"]], width=width, height=height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(sample_dir / "video.mp4"), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open writer: {sample_dir / 'video.mp4'}")
    for frame in frames_bgr:
        writer.write(frame)
    writer.release()
    first_rgb = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2RGB)
    Image.fromarray(first_rgb).save(sample_dir / "image.jpg", quality=95)
    (sample_dir / "prompt.txt").write_text(PROMPT, encoding="utf-8")

    action_frames = load_json(Path(row["action_json"]))
    poses = np.tile(np.eye(4, dtype=np.float32), (len(row["raw_indices"]), 1, 1))
    for out_idx, raw_idx in enumerate(row["raw_indices"]):
        frame = action_frames[int(raw_idx)]
        cam_pos = frame.get("camera_position") or [float(frame["x"]), float(frame["y"]), float(frame.get("z", 0.0)) + 64.0]
        poses[out_idx, :3, :3] = pose_rotation_from_action_frame(frame)
        poses[out_idx, :3, 3] = np.asarray(cam_pos, dtype=np.float32)
    intrinsics = np.repeat(INTRINSICS_832_480[None, :], len(row["raw_indices"]), axis=0)
    np.save(sample_dir / "poses.npy", poses)
    np.save(sample_dir / "intrinsics.npy", intrinsics)

    meta = dict(row)
    meta.update(
        {
            "sample_dir": str(sample_dir),
            "video": str(sample_dir / "video.mp4"),
            "image": str(sample_dir / "image.jpg"),
            "poses": str(sample_dir / "poses.npy"),
            "intrinsics": str(sample_dir / "intrinsics.npy"),
            "meta_json": str(sample_dir / "meta.json"),
            "prompt_txt": str(sample_dir / "prompt.txt"),
            "materialize_policy": "raw RGB frames direct-resized 1280x720 -> 832x480, matching pilot1 sanity check",
        }
    )
    write_json(sample_dir / "meta.json", meta)


def build(args: argparse.Namespace) -> dict[str, Any]:
    matches = select_matches(args)
    args.out_root.mkdir(parents=True, exist_ok=True)
    clip_root = args.out_root / "clips"
    manifest_dir = args.out_root / "manifests"

    source_manifest = manifest_dir / "light_dust2_pilot2_source_manifest.jsonl"
    if args.materialize_only:
        rows = list(iter_jsonl(source_manifest))
        match_reports: list[dict[str, Any]] = []
    else:
        rows = []
        match_reports = []
        for session, match_dir, split in matches:
            candidates = candidate_rows_for_match(
                raw_root=args.raw_root,
                session=session,
                match_dir=match_dir,
                video_frames=args.video_frames,
                raw_stride=args.raw_stride,
                latent_frames=args.latent_frames,
                window_stride_raw=args.window_stride_raw,
                max_rows=args.max_candidates_per_match,
            )
            take = candidates[: args.clips_per_match]
            for row in take:
                row["map_memory_split"] = split
                row["map_memory_split_key"] = "session_match"
            rows.extend(take)
            match_reports.append(
                {
                    "session": session,
                    "match_id": match_dir.name,
                    "split": split,
                    "candidate_count": len(candidates),
                    "selected_count": len(take),
                    "best_dynamic_score": float(candidates[0]["dynamic_score"]) if candidates else None,
                }
            )

    rows.sort(key=lambda row: (row["map_memory_split"], row["hash"], row["game_id"], row["episode"], row["player_stem"], row["raw_start"]))
    for idx, row in enumerate(rows):
        sample_name = f"{idx:04d}_{row['game_id']}_{row['episode']}_{row['player_stem']}"
        sample_dir = clip_root / sample_name
        row["source_manifest_index"] = idx
        row["pilot_selection_source"] = "light_dust2_cross_session_pilot2_v0"
        row["sample_dir"] = str(sample_dir)
        row["clip_dir"] = str(sample_dir)
        row["video"] = str(sample_dir / "video.mp4")
        row["image"] = str(sample_dir / "image.jpg")
        row["poses"] = str(sample_dir / "poses.npy")
        row["intrinsics"] = str(sample_dir / "intrinsics.npy")
        row["meta_json"] = str(sample_dir / "meta.json")
        row["prompt_txt"] = str(sample_dir / "prompt.txt")
        should_materialize = (
            args.materialize
            and idx % args.materialize_num_shards == args.materialize_shard_index
        )
        if should_materialize:
            materialize(row, sample_dir=sample_dir, width=args.video_width, height=args.video_height, fps=args.fps)

    if not args.materialize_only:
        write_jsonl(source_manifest, rows)
        (manifest_dir / "light_dust2_pilot2_selected_clip_ids.txt").write_text(
            "\n".join(str(row["clip_id"]) for row in rows) + "\n",
            encoding="utf-8",
        )
    split_counts = Counter(str(row["map_memory_split"]) for row in rows)
    session_counts = Counter(str(row["hash"]) for row in rows)
    match_counts = Counter(str(row["game_id"]) for row in rows)
    report = {
        "kind": "light_dust2_pilot2_source_build_report_v0",
        "out_root": str(args.out_root),
        "source_manifest": str(source_manifest),
        "raw_root": str(args.raw_root),
        "session_plan": args.session_plan,
        "match_count": len({row["game_id"] for row in rows}),
        "clip_count": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "session_clip_counts": dict(sorted(session_counts.items())),
        "selected_match_count": len(match_counts),
        "clips_per_match_target": args.clips_per_match,
        "materialized": bool(args.materialize),
        "materialize_only": bool(args.materialize_only),
        "materialize_shard_index": args.materialize_shard_index,
        "materialize_num_shards": args.materialize_num_shards,
        "match_reports": match_reports,
    }
    write_json(args.out_root / "pilot2_source_build_report_v0.json", report)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, default=Path(str(source_path('assets', 'csgo-datasets'))))
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--session-plan", nargs="+", default=[
        "9ea6db4a23df49f4:25:train",
        "56bc2d36cc366395:25:train",
        "2bb7255d07049f06:9:val",
        "149450a323214632:21:test",
    ])
    ap.add_argument("--clips-per-match", type=int, default=40)
    ap.add_argument("--max-candidates-per-match", type=int, default=40)
    ap.add_argument("--video-width", type=int, default=832)
    ap.add_argument("--video-height", type=int, default=480)
    ap.add_argument("--video-frames", type=int, default=81)
    ap.add_argument("--raw-stride", type=int, default=2)
    ap.add_argument("--latent-frames", type=int, default=21)
    ap.add_argument("--window-stride-raw", type=int, default=162)
    ap.add_argument("--fps", type=float, default=16.0)
    ap.add_argument("--materialize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--materialize-only", action="store_true")
    ap.add_argument("--materialize-shard-index", type=int, default=0)
    ap.add_argument("--materialize-num-shards", type=int, default=1)
    args = ap.parse_args()
    report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
