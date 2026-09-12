#!/usr/bin/env python3
"""Reusable 10-ego, two-window AR demo pipeline.

The pipeline deliberately separates CPU preparation, dense rendering, state
construction, generation, verification, and packaging.  Every stage validates
the full 10-view contract and writes an explicit report before later stages are
allowed to run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


SESSION = "32f1644d4f42c29d"
PLAYER_COUNT = 10
WINDOW_RAW_SPAN = 160
SCENE_RAW_SPAN = 320
LATENT_STRIDE = 8
VIDEO_STRIDE = 2
LATENT_FRAMES = 21
VIDEO_FRAMES = 81
ALIGNMENT_REMAINDER = 4
DEFAULT_SEED = 20260722
DEFAULT_STEPS = 70
DEFAULT_FPS = 16
W5_LOW_SHA256 = "633ec804b3bc440837de735c27cd44bc053c634a6ca79f93f351594c4f3b185d"
W5_HIGH_SHA256 = "6c3f7ef5d4cbf9163d21ce3beb95d223807a0a77be16b3ccb166ea13ac0e638b"
EXACT_FRAME_OFFSETS = [0, 22, 46, 68, 92, 114, 138, 160]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_write_text(path, payload)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    atomic_write_text(path, payload)


def atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(path)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def aligned_latest_start(first_death: int, action_length: int) -> int | None:
    """Latest valid start where start+320 is strictly before the first death."""
    latest_end = min(first_death - 1, action_length - 1)
    latest_start = latest_end - SCENE_RAW_SPAN
    start = latest_start - ((latest_start - ALIGNMENT_REMAINDER) % LATENT_STRIDE)
    return start if start >= 0 else None


def first_death_frame(events_path: Path) -> int | None:
    deaths = []
    if not events_path.is_file():
        return None
    with events_path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            frame = event.get("frame_count")
            if event.get("event_type") == "player_death" and isinstance(frame, (int, float)):
                deaths.append(int(frame))
    return min(deaths) if deaths else None


def player_action_paths(episode_dir: Path) -> list[Path]:
    return sorted(episode_dir.glob("*_inst_000.json"))


def load_actions(path: Path) -> list[dict[str, Any]]:
    actions = read_json(path)
    if not isinstance(actions, list):
        raise RuntimeError(f"action file is not a JSON list: {path}")
    return actions


def validate_action_contract(actions: Sequence[dict[str, Any]], start: int, end: int, path: Path) -> None:
    if len(actions) <= end:
        raise RuntimeError(f"action file too short for frame {end}: {path}, len={len(actions)}")
    for frame in (start, start + WINDOW_RAW_SPAN, end):
        if int(actions[frame].get("frame_count", -1)) != frame:
            raise RuntimeError(f"frame_count/index mismatch at {frame}: {path}")


def motion_score(actions_by_player: Sequence[Sequence[dict[str, Any]]], start: int, end: int) -> dict[str, float]:
    distances: list[float] = []
    yaws: list[float] = []
    for actions in actions_by_player:
        distance = 0.0
        yaw_change = 0.0
        for frame in range(start + LATENT_STRIDE, end + 1, LATENT_STRIDE):
            previous = actions[frame - LATENT_STRIDE]
            current = actions[frame]
            p0 = (float(previous["x"]), float(previous["y"]), float(previous["z"]))
            p1 = (float(current["x"]), float(current["y"]), float(current["z"]))
            distance += math.dist(p0, p1)
            delta = (float(current["yaw"]) - float(previous["yaw"]) + 180.0) % 360.0 - 180.0
            yaw_change += abs(delta)
        distances.append(distance)
        yaws.append(yaw_change)
    return {
        "mean_distance_units": sum(distances) / len(distances),
        "min_distance_units": min(distances),
        "mean_abs_yaw_degrees": sum(yaws) / len(yaws),
        "score": sum(distances) / len(distances) + 0.5 * sum(yaws) / len(yaws),
    }


def candidate_rows(pattern: str) -> list[dict[str, Any]]:
    files = sorted(Path().glob(pattern)) if not os.path.isabs(pattern) else sorted(Path("/").glob(pattern[1:]))
    if not files:
        raise RuntimeError(f"no candidate manifests matched: {pattern}")
    rows: list[dict[str, Any]] = []
    for path in files:
        rows.extend(read_jsonl(path))
    return rows


def discover(args: argparse.Namespace) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    source_files = sorted(Path(args.project_root).glob(args.candidate_glob))
    if not source_files:
        raise RuntimeError(f"no source files matched {args.candidate_glob}")
    for source in source_files:
        for row in read_jsonl(source):
            grouped[(row["match_id"], row["episode"])].append(row)

    includes = set(args.include_scene or [])
    records = []
    rejected = []
    for (match_id, episode), rows in sorted(grouped.items()):
        scene_key = f"{match_id}:{episode}"
        if includes and scene_key not in includes:
            continue
        episode_dir = Path(rows[0]["action_json"]).parent
        actions_paths = player_action_paths(episode_dir)
        mp4_paths = sorted(episode_dir.glob("*_inst_000.mp4"))
        death = first_death_frame(episode_dir / "world_events.jsonl")
        reasons = []
        if len(actions_paths) != PLAYER_COUNT:
            reasons.append(f"action_count={len(actions_paths)}")
        if len(mp4_paths) != PLAYER_COUNT:
            reasons.append(f"video_count={len(mp4_paths)}")
        if death is None:
            reasons.append("missing_first_death")
        actions_by_player = []
        lengths = []
        if not reasons:
            for path in actions_paths:
                actions = load_actions(path)
                actions_by_player.append(actions)
                lengths.append(len(actions))
            start = aligned_latest_start(int(death), min(lengths))
            if start is None:
                reasons.append("no_safe_320_frame_window")
        if reasons:
            rejected.append({"scene": scene_key, "reasons": reasons})
            continue
        end = start + SCENE_RAW_SPAN
        for path, actions in zip(actions_paths, actions_by_player):
            validate_action_contract(actions, start, end, path)
        activity = motion_score(actions_by_player, start, end)
        template = rows[0]
        players = []
        for action_path in actions_paths:
            stem = action_path.name[:-5]
            mp4 = episode_dir / f"{stem}.mp4"
            visibility = episode_dir / f"{stem}_player_visibility.json"
            if not mp4.is_file() or not visibility.is_file():
                raise RuntimeError(f"missing paired source for {action_path}")
            players.append({
                "player_stem": stem,
                "action_json": str(action_path),
                "mp4": str(mp4),
                "player_visibility": str(visibility),
            })
        records.append({
            "scene_key": scene_key,
            "match_id": match_id,
            "episode": episode,
            "episode_dir": str(episode_dir),
            "raw_start": start,
            "raw_end": end,
            "first_death_frame": int(death),
            "strictly_before_first_death": end < int(death),
            "action_length_min": min(lengths),
            "activity": activity,
            "game_manifest": str(episode_dir / "game_manifest.json"),
            "template_row": template,
            "players": players,
        })

    if includes and len(records) != len(includes):
        found = {record["scene_key"] for record in records}
        raise RuntimeError(f"requested scenes failed discovery: {sorted(includes - found)}")
    if not includes:
        selected = []
        used_matches = set()
        for record in sorted(records, key=lambda item: item["activity"]["score"], reverse=True):
            if record["match_id"] in used_matches:
                continue
            selected.append(record)
            used_matches.add(record["match_id"])
            if len(selected) == args.scene_count:
                break
        records = selected
    if len(records) < args.scene_count:
        raise RuntimeError(f"only {len(records)} valid diverse scenes, need {args.scene_count}")
    records = sorted(records, key=lambda item: (item["match_id"], item["episode"]))[: args.scene_count]
    for index, record in enumerate(records, 1):
        record["scene_index"] = index
        record["scene_id"] = f"scene_{index:02d}_{record['match_id'][:8]}_{record['episode']}"

    payload_hash = sha256_bytes(json.dumps(records, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    report = {
        "kind": "multiscene_10ego_selection_v1",
        "status": "pass",
        "scene_count": len(records),
        "contract": {
            "players_per_scene": PLAYER_COUNT,
            "raw_span": SCENE_RAW_SPAN,
            "two_windows": [WINDOW_RAW_SPAN, WINDOW_RAW_SPAN],
            "alignment": f"start % {LATENT_STRIDE} == {ALIGNMENT_REMAINDER}",
            "death_rule": "raw_end < first_player_death_frame",
        },
        "source_manifests": [
            {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in source_files
        ],
        "selection_payload_sha256": payload_hash,
        "scenes": records,
        "rejected": rejected,
    }
    write_json(Path(args.selection_out), report)
    print(json.dumps({"status": "pass", "scenes": [r["scene_id"] for r in records]}, ensure_ascii=False))


def ffmpeg_executable() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_frame(source: Path, frame: int, output: Path) -> None:
    command = [
        ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-vf", f"select=eq(n\\,{frame})", "-vframes", "1", str(output),
    ]
    subprocess.run(command, check=True)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"failed to extract frame {frame} from {source}")


def prompt_text(template: dict[str, Any]) -> str:
    candidates = []
    if template.get("prompt_txt"):
        candidates.append(Path(template["prompt_txt"]))
    if template.get("clip_dir"):
        candidates.append(Path(template["clip_dir"]) / "prompt.txt")
    for path in candidates:
        if path.is_file():
            return path.read_text(encoding="utf-8")
    raise RuntimeError(f"template has no readable prompt: {candidates}")


def prepare(args: argparse.Namespace) -> None:
    import numpy as np

    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    selection = read_json(Path(args.selection))
    if selection.get("status") != "pass" or selection.get("scene_count", 0) < 1:
        raise RuntimeError("selection report is not pass")
    pose_module = load_module(project_root / "tools/build_light_dust2_pilot3_source_v0.py", "pose_source_v0")
    prepared = []
    for scene in selection["scenes"]:
        scene_root = output_root / "scenes" / scene["scene_id"]
        prompt = prompt_text(scene["template_row"])
        state_rows = []
        view_reports = []
        starts = [int(scene["raw_start"]), int(scene["raw_start"]) + WINDOW_RAW_SPAN]
        if starts[1] + WINDOW_RAW_SPAN >= int(scene["first_death_frame"]):
            raise RuntimeError(f"death gate failed during prepare: {scene['scene_id']}")
        for view_index, player in enumerate(scene["players"]):
            action_path = Path(player["action_json"])
            actions = load_actions(action_path)
            rows = []
            for window_index, start in enumerate(starts):
                end = start + WINDOW_RAW_SPAN
                validate_action_contract(actions, start, end, action_path)
                latent_frames = list(range(start, end + 1, LATENT_STRIDE))
                video_frames = list(range(start, end + 1, VIDEO_STRIDE))
                if len(latent_frames) != LATENT_FRAMES or len(video_frames) != VIDEO_FRAMES:
                    raise RuntimeError("frame geometry contract failed")
                stem = player["player_stem"]
                clip_id = f"{SESSION}_{scene['match_id']}_{scene['episode']}_{stem}_{start:07d}"
                clip_dir = scene_root / "clips" / clip_id
                clip_dir.mkdir(parents=True, exist_ok=True)
                image = clip_dir / "image.jpg"
                if not image.is_file():
                    extract_frame(Path(player["mp4"]), start, image)
                (clip_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
                poses = np.zeros((VIDEO_FRAMES, 4, 4), dtype=np.float32)
                poses[:, 3, 3] = 1.0
                for pose_index, raw_frame in enumerate(video_frames):
                    action = actions[raw_frame]
                    camera = action.get("camera_position")
                    if camera is None:
                        camera = [float(action["x"]), float(action["y"]), float(action.get("z", 0.0)) + 64.0]
                    poses[pose_index, :3, :3] = pose_module.pose_rotation_from_action_frame(action)
                    poses[pose_index, :3, 3] = np.asarray(camera, dtype=np.float32)
                intrinsics = np.repeat(pose_module.INTRINSICS_832_480[None, :], VIDEO_FRAMES, axis=0)
                np.save(clip_dir / "poses.npy", poses)
                np.save(clip_dir / "intrinsics.npy", intrinsics)
                template = dict(scene["template_row"])
                template.update({
                    "pair_id": f"{scene['scene_id']}__view_{view_index:02d}",
                    "side": f"view_{view_index:02d}",
                    "match_id": scene["match_id"],
                    "game_id": scene["match_id"],
                    "episode": scene["episode"],
                    "ego": stem,
                    "player_stem": stem,
                    "clip_id": clip_id,
                    "clip_dir": str(clip_dir),
                    "image": str(image),
                    "poses": str(clip_dir / "poses.npy"),
                    "intrinsics": str(clip_dir / "intrinsics.npy"),
                    "prompt_txt": str(clip_dir / "prompt.txt"),
                    "game_manifest": scene["game_manifest"],
                    "action_json": player["action_json"],
                    "mp4": player["mp4"],
                    "player_visibility": player["player_visibility"],
                    "raw_start": start,
                    "frame_count_start": start,
                    "frame_count_end": end,
                    "positive_latent_frames": latent_frames,
                    "raw_indices": latent_frames,
                    "positive_latent_frame_count": LATENT_FRAMES,
                    "latent_frames": LATENT_FRAMES,
                    "exact_aligned_raw_frames": [start + offset for offset in EXACT_FRAME_OFFSETS],
                    "phase2a_window_id": clip_id,
                    "phase2a_dense_sequence_manifest": str(
                        scene_root / "dense" / clip_id / "dense_sequence_manifest_v0.jsonl"
                    ),
                    "frame_contract": {
                        "gen_frame": "4 * latent_index",
                        "raw_frame": "raw_start + 8 * latent_index",
                    },
                })
                rows.append(template)
                state_rows.append({
                    "clip_id": clip_id,
                    "map_memory_raw_frame_indices": latent_frames,
                    "action_json": player["action_json"],
                    "player_stem": stem,
                })
            manifest = scene_root / "manifests" / f"view_{view_index:02d}_{player['player_stem']}.jsonl"
            write_jsonl(manifest, rows)
            view_reports.append({
                "view_index": view_index,
                "player_stem": player["player_stem"],
                "manifest": str(manifest),
                "clip_ids": [row["clip_id"] for row in rows],
            })
        if len(view_reports) != PLAYER_COUNT or len(state_rows) != PLAYER_COUNT * 2:
            raise RuntimeError(f"prepare cardinality failed: {scene['scene_id']}")
        state_input = scene_root / "state" / "state_input_20.jsonl"
        write_jsonl(state_input, state_rows)
        scene_report = {
            "kind": "multiscene_10ego_prepare_scene_v1",
            "status": "pass",
            "scene_id": scene["scene_id"],
            "match_id": scene["match_id"],
            "episode": scene["episode"],
            "raw_start": scene["raw_start"],
            "raw_end": scene["raw_end"],
            "first_death_frame": scene["first_death_frame"],
            "view_count": len(view_reports),
            "window_count": len(state_rows),
            "state_input": str(state_input),
            "views": view_reports,
        }
        write_json(scene_root / "PREPARE_REPORT_v1.json", scene_report)
        prepared.append(scene_report)
    overall = {
        "kind": "multiscene_10ego_prepare_v1",
        "status": "pass",
        "selection": str(Path(args.selection).resolve()),
        "scene_count": len(prepared),
        "scenes": prepared,
    }
    write_json(output_root / "PREPARE_REPORT_v1.json", overall)
    print(json.dumps({"status": "pass", "scenes": len(prepared), "views": len(prepared) * 10}))


def scene_root(output_root: Path, scene_id: str) -> Path:
    root = output_root.resolve() / "scenes" / scene_id
    if not (root / "PREPARE_REPORT_v1.json").is_file():
        raise RuntimeError(f"scene is not prepared: {root}")
    return root


def selected_view_manifests(root: Path, view_indices: Sequence[int] | None) -> list[Path]:
    manifests = sorted((root / "manifests").glob("view_*.jsonl"))
    if len(manifests) != PLAYER_COUNT:
        raise RuntimeError(f"expected 10 view manifests, got {len(manifests)}")
    if view_indices is None:
        return manifests
    selected = []
    for index in view_indices:
        if index < 0 or index >= len(manifests):
            raise RuntimeError(f"view index out of range: {index}")
        selected.append(manifests[index])
    return selected


def render_dense(args: argparse.Namespace) -> None:
    import numpy as np

    project_root = Path(args.project_root).resolve()
    root = scene_root(Path(args.output_root), args.scene_id)
    module = load_module(project_root / "tools/build_mesh_dense_condition_v0.py", "dense_builder_v0")
    manifests = selected_view_manifests(root, args.view)
    rows = [row for manifest in manifests for row in read_jsonl(manifest)]
    if args.window is not None:
        rows = [row for index, row in enumerate(rows) if index % 2 in set(args.window)]
    if not rows:
        raise RuntimeError("no dense rows selected")
    match_dir = Path(rows[0]["game_manifest"]).parents[2]
    bsp = project_root / "docs/assets/bsp_static_geometry_v0/de_dust2_bsp_faces_v0.npz"
    cache = module.load_renderer_cache(match_dir, project_root / "tools", bsp_faces_npz=bsp)
    completed = []
    for row in rows:
        output_dir = Path(row["phase2a_dense_sequence_manifest"]).parent
        samples_dir = output_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows = []
        for latent_index, raw_frame in enumerate(row["positive_latent_frames"]):
            sample_id = f"{row['game_id']}__{row['episode']}_{row['player_stem']}_f{int(raw_frame):06d}"
            sample_dir = samples_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            dense_path = sample_dir / "mesh_dense_condition_v0.npz"
            if not dense_path.is_file():
                dense, _, _, _, _ = module.render_memory_dense_condition(
                    match_dir, row["episode"], row["player_stem"], int(raw_frame),
                    320, 176, 106.26, 4.0, 3000.0, 1.0, 0,
                    cache=cache, player_mask_mode="capsule", mesh_backend="bsp_faces_gpu",
                )
                np.savez_compressed(dense_path, dense=dense.astype(np.float16))
            with np.load(dense_path) as payload:
                if "dense" not in payload or payload["dense"].shape[0] <= 0:
                    raise RuntimeError(f"invalid dense sample: {dense_path}")
            manifest_rows.append({
                "index": latent_index,
                "latent_index": latent_index,
                "raw_frame": int(raw_frame),
                "gen_frame": latent_index * 4,
                "sample_id": sample_id,
                "dense_path": str(dense_path),
            })
        if len(manifest_rows) != LATENT_FRAMES:
            raise RuntimeError(f"dense cardinality failed: {row['clip_id']}")
        write_jsonl(Path(row["phase2a_dense_sequence_manifest"]), manifest_rows)
        completed.append(row["clip_id"])
    report = {
        "kind": "multiscene_10ego_dense_shard_v1",
        "status": "pass",
        "scene_id": args.scene_id,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "completed_windows": len(completed),
        "clip_ids": completed,
    }
    report_path = root / "dense_reports" / f"dense_{os.getpid()}.json"
    write_json(report_path, report)
    print(json.dumps({"status": "pass", "completed_windows": len(completed), "report": str(report_path)}))


def build_state(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    root = scene_root(Path(args.output_root), args.scene_id)
    state_dir = root / "state" / "cache"
    state_dir.mkdir(parents=True, exist_ok=True)
    input_manifest = root / "state" / "state_input_20.jsonl"
    command = [
        sys.executable, str(project_root / "tools/build_state_channels_v0.py"),
        "--cache-manifest", str(input_manifest), "--out-dir", str(state_dir),
        "--manifest-name", "state_cache_manifest.jsonl", "--report-name", "state_builder_report.json",
    ]
    subprocess.run(command, cwd=project_root, check=True)
    manifest = state_dir / "state_cache_manifest.jsonl"
    rows = read_jsonl(manifest)
    if len(rows) != PLAYER_COUNT * 2:
        raise RuntimeError(f"state row count is {len(rows)}, expected 20")
    for row in rows:
        cache_path = Path(row["state_cache"])
        if not cache_path.is_absolute():
            cache_path = (state_dir / cache_path).resolve()
            row["state_cache"] = str(cache_path)
        if not cache_path.is_file():
            raise RuntimeError(f"missing state cache: {cache_path}")
    write_jsonl(manifest, rows)
    report = {
        "kind": "multiscene_10ego_state_v1",
        "status": "pass",
        "scene_id": args.scene_id,
        "records": len(rows),
        "manifest": str(manifest),
    }
    write_json(root / "STATE_REPORT_v1.json", report)
    print(json.dumps(report))


def validate_dense_and_state(root: Path, manifest: Path) -> None:
    rows = read_jsonl(manifest)
    if len(rows) != 2:
        raise RuntimeError(f"view manifest must contain exactly two windows: {manifest}")
    state_report = read_json(root / "STATE_REPORT_v1.json")
    state_rows = {row["clip_id"]: row for row in read_jsonl(Path(state_report["manifest"]))}
    for row in rows:
        dense_rows = read_jsonl(Path(row["phase2a_dense_sequence_manifest"]))
        if len(dense_rows) != LATENT_FRAMES:
            raise RuntimeError(f"dense manifest is incomplete: {row['clip_id']}")
        if row["clip_id"] not in state_rows:
            raise RuntimeError(f"state cache missing clip: {row['clip_id']}")


def generate_one(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    root = scene_root(Path(args.output_root), args.scene_id)
    manifests = selected_view_manifests(root, [args.view])
    manifest = manifests[0]
    validate_dense_and_state(root, manifest)
    state_manifest = Path(read_json(root / "STATE_REPORT_v1.json")["manifest"])
    output = root / "generation" / f"view_{args.view:02d}"
    command = [
        sys.executable, str(project_root / "tools/qxq_sample_candidate1_ar_v0.py"),
        "--source-manifest", str(manifest),
        "--state-cache-manifest", str(state_manifest),
        "--adapter-checkpoint-low", str(Path(args.checkpoint_low).resolve()),
        "--adapter-checkpoint-high", str(Path(args.checkpoint_high).resolve()),
        "--out-root", str(output), "--latent-frames", str(LATENT_FRAMES),
        "--chunk-size", "3", "--steps", str(args.steps), "--seed", str(args.seed), "--device-id", "0",
    ]
    if args.dry_run:
        print(json.dumps({"command": command, "output": str(output)}, ensure_ascii=False))
        return
    subprocess.run(command, cwd=project_root, check=True)
    report_path = output / "candidate1_ar_run_report_v0.json"
    report = read_json(report_path)
    if report.get("status") != "complete" or report.get("completed_windows") != 2 or report.get("failed_windows") != 0:
        raise RuntimeError(f"generation did not complete: {report_path}")
    print(json.dumps({"status": "pass", "scene_id": args.scene_id, "view": args.view, "report": str(report_path)}))


def video_probe(path: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    result = {
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    if result["fps"] <= 0:
        raise RuntimeError(f"invalid fps: {path}")
    result["duration_seconds"] = result["frames"] / result["fps"]
    return result


def verify(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    selection = read_json(Path(args.selection))
    scene_reports = []
    for scene in selection["scenes"]:
        root = scene_root(output_root, scene["scene_id"])
        prep = read_json(root / "PREPARE_REPORT_v1.json")
        state = read_json(root / "STATE_REPORT_v1.json")
        if prep.get("status") != "pass" or prep.get("view_count") != PLAYER_COUNT or state.get("records") != 20:
            raise RuntimeError(f"prepare/state gate failed: {scene['scene_id']}")
        reports = []
        for view in range(PLAYER_COUNT):
            manifest = selected_view_manifests(root, [view])[0]
            validate_dense_and_state(root, manifest)
            report_path = root / "generation" / f"view_{view:02d}" / "candidate1_ar_run_report_v0.json"
            report = read_json(report_path)
            if report.get("status") != "complete" or report.get("completed_windows") != 2 or report.get("failed_windows") != 0:
                raise RuntimeError(f"incomplete generation report: {report_path}")
            experts = report.get("experts", [])
            if [(x.get("step"), x.get("state_projector_loaded")) for x in experts] != [(75, True), (75, True)]:
                raise RuntimeError(f"checkpoint/state binding failed: {report_path}")
            if any(window.get("steps") != args.steps or window.get("seed") != args.seed for window in report["windows"]):
                raise RuntimeError(f"sampling provenance mismatch: {report_path}")
            stitched = Path(report["stitched"]["mp4"])
            if not stitched.is_absolute():
                stitched = Path(args.project_root).resolve() / stitched
            probe = video_probe(stitched)
            if probe["frames"] != 162 or abs(probe["fps"] - DEFAULT_FPS) > 1e-6:
                raise RuntimeError(f"stitched video contract failed: {stitched}: {probe}")
            reports.append({"view": view, "report": str(report_path), "stitched": str(stitched), "probe": probe})
        scene_reports.append({"scene_id": scene["scene_id"], "status": "pass", "views": reports})
    result = {
        "kind": "multiscene_10ego_generation_gate_v1",
        "status": "pass",
        "scene_count": len(scene_reports),
        "view_count": len(scene_reports) * PLAYER_COUNT,
        "seed": args.seed,
        "steps": args.steps,
        "checkpoint_sha256": {"low": W5_LOW_SHA256, "high": W5_HIGH_SHA256},
        "scenes": scene_reports,
    }
    write_json(output_root / "GENERATION_GATE_REPORT_v1.json", result)
    print(json.dumps({"status": "pass", "scenes": len(scene_reports), "views": len(scene_reports) * 10}))


def run_ffmpeg(arguments: list[str]) -> None:
    subprocess.run([ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-y", *arguments], check=True)


def package(args: argparse.Namespace) -> None:
    import cv2

    output_root = Path(args.output_root).resolve()
    gate = read_json(output_root / "GENERATION_GATE_REPORT_v1.json")
    if gate.get("status") != "pass":
        raise RuntimeError("generation gate is not pass")
    delivery = output_root / "delivery_unlabeled_10s"
    all_artifacts = []
    scene_manifests = []
    for scene_index, scene in enumerate(gate["scenes"], 1):
        scene_dir = delivery / f"scene_{scene_index:02d}"
        singles_dir = scene_dir / "01_single_views"
        pairs_dir = scene_dir / "02_two_view_pairs"
        grid_dir = scene_dir / "03_ten_view"
        qa_dir = scene_dir / ".qa"
        for directory in (singles_dir, pairs_dir, grid_dir, qa_dir):
            directory.mkdir(parents=True, exist_ok=True)
        singles = []
        source_records = []
        for record in scene["views"]:
            view = int(record["view"])
            source = Path(record["stitched"])
            output = singles_dir / f"view_{view + 1:02d}_10s.mp4"
            run_ffmpeg([
                "-i", str(source), "-vf", "select='between(n,0,79)+between(n,81,160)',setpts=N/16/TB",
                "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                "-pix_fmt", "yuv420p", "-r", "16", "-movflags", "+faststart", str(output),
            ])
            probe = video_probe(output)
            if probe["frames"] != 160 or abs(probe["duration_seconds"] - 10.0) > 1e-6:
                raise RuntimeError(f"exact 10-second contract failed: {output}: {probe}")
            singles.append(output)
            source_records.append({"view": view, "source": str(source), "output": str(output), "probe": probe})
        pairs = []
        for pair_index, (left, right) in enumerate(((0, 5), (1, 6), (2, 7), (3, 8), (4, 9)), 1):
            output = pairs_dir / f"pair_{pair_index:02d}_10s.mp4"
            run_ffmpeg([
                "-i", str(singles[left]), "-i", str(singles[right]),
                "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]",
                "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                "-pix_fmt", "yuv420p", "-r", "16", "-movflags", "+faststart", str(output),
            ])
            probe = video_probe(output)
            if probe["frames"] != 160 or probe["width"] != 1664:
                raise RuntimeError(f"pair contract failed: {output}: {probe}")
            pairs.append({"pair": pair_index, "views": [left + 1, right + 1], "output": str(output), "probe": probe})
        grid = grid_dir / "ten_views_5x2_10s.mp4"
        inputs = []
        for path in singles:
            inputs.extend(["-i", str(path)])
        filters = [
            f"[{index}:v]scale=480:270:force_original_aspect_ratio=decrease,pad=480:270:(ow-iw)/2:(oh-ih)/2:black[v{index}]"
            for index in range(10)
        ]
        layout = "|".join(f"{(index % 5) * 480}_{(index // 5) * 270}" for index in range(10))
        filters.append("".join(f"[v{index}]" for index in range(10)) + f"xstack=inputs=10:layout={layout}[v]")
        run_ffmpeg([
            *inputs, "-filter_complex", ";".join(filters), "-map", "[v]", "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-r", "16", "-movflags", "+faststart", str(grid),
        ])
        grid_probe = video_probe(grid)
        if grid_probe["frames"] != 160 or (grid_probe["width"], grid_probe["height"]) != (2400, 540):
            raise RuntimeError(f"grid contract failed: {grid}: {grid_probe}")
        capture = cv2.VideoCapture(str(grid))
        contact_frames = []
        for frame_index in (0, 40, 80, 120, 152):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"cannot read grid frame {frame_index}: {grid}")
            contact_frames.append(frame)
        capture.release()
        cv2.imwrite(str(qa_dir / "ten_views_contact.jpg"), cv2.vconcat(contact_frames), [cv2.IMWRITE_JPEG_QUALITY, 93])
        scene_videos = sorted(scene_dir.rglob("*.mp4"))
        artifacts = [
            {
                "path": str(path.relative_to(delivery)), "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path), "probe": video_probe(path), "contains_visible_labels": False,
            }
            for path in scene_videos
        ]
        all_artifacts.extend(artifacts)
        scene_manifests.append({
            "scene_id": scene["scene_id"], "single_views": source_records, "pairs": pairs,
            "grid": {"output": str(grid), "probe": grid_probe}, "artifacts": artifacts,
        })
    manifest = {
        "kind": "lingbot_w5_75_multiscene_10ego_10s_unlabeled_delivery_v1",
        "status": "pass",
        "global_checkpoint": "W5-75",
        "checkpoint_sha256": gate["checkpoint_sha256"],
        "generation": {"seed": gate["seed"], "sampling_steps": gate["steps"], "fps": 16, "frames": 160,
                       "duration_seconds": 10.0, "audio": False, "visible_labels": False},
        "counts": {"scenes": len(scene_manifests), "single_views": len(scene_manifests) * 10,
                   "two_view_pairs": len(scene_manifests) * 5, "ten_view_grids": len(scene_manifests),
                   "total_videos": len(all_artifacts)},
        "scenes": scene_manifests,
        "artifacts": all_artifacts,
    }
    write_json(delivery / "MANIFEST.json", manifest)
    (delivery / "README_zh.txt").write_text(
        "LingBot W5-75 多场景十视角 10 秒无标签视频\n"
        "每个场景包含 10 条单视角、5 条同步双视角和 1 条 5x2 十视角总览。\n"
        "所有视频均为 160 帧、16fps、精确 10 秒、H.264、无文字标签、无音频。\n",
        encoding="utf-8",
    )
    if args.copy_to:
        target = Path(args.copy_to).expanduser().resolve()
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(delivery, target)
    print(json.dumps(manifest["counts"], ensure_ascii=False))


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-root", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    command = subparsers.add_parser("discover")
    command.add_argument("--project-root", required=True)
    command.add_argument("--candidate-glob", default="output/demo_ext15_multiview_ckpt8500_20260716_v0/source_shard*.jsonl")
    command.add_argument("--selection-out", required=True)
    command.add_argument("--scene-count", type=int, default=3)
    command.add_argument("--include-scene", action="append")
    command.set_defaults(handler=discover)

    command = subparsers.add_parser("prepare")
    add_common(command)
    command.add_argument("--selection", required=True)
    command.set_defaults(handler=prepare)

    command = subparsers.add_parser("render-dense")
    add_common(command)
    command.add_argument("--scene-id", required=True)
    command.add_argument("--view", type=int, action="append")
    command.add_argument("--window", type=int, choices=(0, 1), action="append")
    command.set_defaults(handler=render_dense)

    command = subparsers.add_parser("build-state")
    add_common(command)
    command.add_argument("--scene-id", required=True)
    command.set_defaults(handler=build_state)

    command = subparsers.add_parser("generate-one")
    add_common(command)
    command.add_argument("--scene-id", required=True)
    command.add_argument("--view", type=int, required=True, choices=range(10))
    command.add_argument("--checkpoint-low", required=True)
    command.add_argument("--checkpoint-high", required=True)
    command.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    command.add_argument("--seed", type=int, default=DEFAULT_SEED)
    command.add_argument("--dry-run", action="store_true")
    command.set_defaults(handler=generate_one)

    command = subparsers.add_parser("verify")
    add_common(command)
    command.add_argument("--selection", required=True)
    command.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    command.add_argument("--seed", type=int, default=DEFAULT_SEED)
    command.set_defaults(handler=verify)

    command = subparsers.add_parser("package")
    add_common(command)
    command.add_argument("--copy-to")
    command.set_defaults(handler=package)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
