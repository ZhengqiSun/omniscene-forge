#!/usr/bin/env python3
"""Build and verify a no-future-pose Dynamics-to-LingBot demo."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


LATENT_STRIDE = 8
VIDEO_STRIDE = 2
LATENT_FRAMES = 21
VIDEO_FRAMES = 81
TARGET_HEIGHT = 480
TARGET_WIDTH = 832
DEFAULT_PROMPT = "first-person gameplay video in Counter-Strike, de_dust2 map"
STATE_FIELDS = ("position", "camera_position", "yaw", "pitch", "health", "alive")
CONTROL_FIELDS = ("actions", "look_delta")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def extract_frame(source: Path, frame: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source),
            "-vf", f"select=eq(n\\,{frame}),scale={TARGET_WIDTH}:{TARGET_HEIGHT}:flags=lanczos",
            "-vframes", "1", str(output),
        ],
        check=True,
    )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"failed to extract frame {frame} from {source}")
    from PIL import Image

    with Image.open(output) as image:
        if image.size != (TARGET_WIDTH, TARGET_HEIGHT):
            raise RuntimeError(f"initial frame contract failed: {output}: {image.size}")


def poisoned_future(memory: dict[str, np.ndarray], start: int) -> dict[str, np.ndarray]:
    """Destroy every future state field while preserving controls and the initial state."""
    poisoned = {key: value.copy() for key, value in memory.items()}
    for key in STATE_FIELDS:
        value = poisoned[key]
        if value.dtype == np.bool_:
            value[:, start + 1 :] = ~value[:, start + 1 :]
        else:
            value[:, start + 1 :] = np.asarray(987654.25, dtype=value.dtype)
    return poisoned


def rollout_equal(left: dict[str, Any], right: dict[str, Any]) -> tuple[bool, dict[str, float]]:
    diffs: dict[str, float] = {}
    for key in ("xy", "z", "yaw", "pitch"):
        a = np.asarray(left[key])
        b = np.asarray(right[key])
        diffs[key] = float(np.max(np.abs(a - b))) if a.size else 0.0
    return all(value == 0.0 for value in diffs.values()), diffs


def frame_contract(start: int, horizon: int) -> tuple[list[int], list[int]]:
    if horizon != 160:
        raise ValueError("v0 is intentionally fixed to one 5-second, 160-raw-frame window")
    latent = list(range(start, start + horizon + 1, LATENT_STRIDE))
    video = list(range(start, start + horizon + 1, VIDEO_STRIDE))
    if len(latent) != LATENT_FRAMES or len(video) != VIDEO_FRAMES:
        raise AssertionError("frame contract mismatch")
    return latent, video


def as_memory_dict(memory: Any) -> dict[str, np.ndarray]:
    return {key: np.asarray(memory[key]).copy() for key in memory.files}


def empty_like(value: np.ndarray) -> np.ndarray:
    if value.dtype == np.bool_:
        return np.zeros_like(value)
    if np.issubdtype(value.dtype, np.floating):
        return np.full_like(value, np.nan)
    return np.full_like(value, -1)


def strict_rollout(
    *,
    module: Any,
    source: dict[str, np.ndarray],
    source_meta: dict[str, Any],
    navmesh: Any,
    bsp_ground: Any,
    motion_params: dict[str, Any] | None,
    start: int,
    horizon: int,
    fps: float,
    look_lag: int,
    action_lag: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    player_count = len(source_meta["player_stems"])
    observed_fit_start = max(0, start - 512)
    look_scales = module.fit_look_scales(source, list(range(player_count)), observed_fit_start, start, look_lag)
    eval_args = argparse.Namespace(
        fps=fps,
        look_lag=look_lag,
        action_lag=action_lag,
        nav_tolerance=24.0,
        base_speed=105.0,
        accel=420.0,
        friction=7.5,
        walk_scale=0.52,
        crouch_scale=0.45,
        max_step_height=72.0,
        player_radius=18.0,
        player_height=72.0,
        enable_bsp_collision=False,
        initial_velocity="zero",
        velocity_model="rule",
        velocity_alpha=10.0,
        velocity_min_speed=2.0,
        max_speed=320.0,
        max_substep_distance=8.0,
    )
    poison = poisoned_future(source, start)
    predictions: list[dict[str, Any]] = []
    non_leak_rows = []
    for player_index in range(player_count):
        if not bool(source["alive"][player_index, start]):
            raise RuntimeError(f"player {player_index} is not alive at initial frame {start}")
        pred = module.rollout(
            source, player_index, start, horizon, navmesh, bsp_ground,
            look_scales, motion_params, None, eval_args,
        )
        poisoned_pred = module.rollout(
            poison, player_index, start, horizon, navmesh, bsp_ground,
            look_scales, motion_params, None, eval_args,
        )
        equal, diffs = rollout_equal(pred, poisoned_pred)
        if not equal:
            raise RuntimeError(f"future-state leakage detected for player {player_index}: {diffs}")
        predictions.append(pred)
        non_leak_rows.append({"player_index": player_index, "bitwise_invariant": equal, "max_abs_diff": diffs})

    output = {key: empty_like(value) for key, value in source.items()}
    for key in CONTROL_FIELDS:
        output[key][:, start : start + horizon + 1] = source[key][:, start : start + horizon + 1]
    for key in ("team_id", "player_index"):
        output[key] = source[key].copy()
    output["track_length"] = np.full_like(source["track_length"], start + horizon + 1)

    trajectory_rows = []
    for player_index, pred in enumerate(predictions):
        sl = slice(start, start + horizon + 1)
        camera_offset = float(source["camera_position"][player_index, start, 2] - source["position"][player_index, start, 2])
        output["position"][player_index, sl, :2] = pred["xy"].astype(np.float32)
        output["position"][player_index, sl, 2] = pred["z"].astype(np.float32)
        output["camera_position"][player_index, sl, :2] = pred["xy"].astype(np.float32)
        output["camera_position"][player_index, sl, 2] = (pred["z"] + camera_offset).astype(np.float32)
        output["yaw"][player_index, sl] = pred["yaw"].astype(np.float32)
        output["pitch"][player_index, sl] = pred["pitch"].astype(np.float32)
        initial_health = float(source["health"][player_index, start])
        output["health"][player_index, sl] = initial_health
        output["alive"][player_index, sl] = initial_health > 0.0
        initial_tick = int(source["tick"][player_index, start])
        output["tick"][player_index, sl] = initial_tick + np.arange(horizon + 1, dtype=output["tick"].dtype)
        gt = source["position"][player_index, sl].astype(np.float64)
        predicted = output["position"][player_index, sl].astype(np.float64)
        error = np.linalg.norm(predicted - gt, axis=1)
        trajectory_rows.append(
            {
                "player_index": player_index,
                "player_stem": source_meta["player_stems"][player_index],
                "final_gt_eval_error_3d": float(error[-1]),
                "mean_gt_eval_error_3d": float(error.mean()),
                "clipped_steps": int(pred["clipped_steps"]),
                "nav_clipped_steps": int(pred["nav_clipped_steps"]),
                "step_height_clipped_steps": int(pred["step_height_clipped_steps"]),
            }
        )
    audit = {
        "observed_state_frames": [observed_fit_start, start],
        "initial_state_frame": start,
        "future_control_frames": [start, start + horizon],
        "future_control_source": "recorded keyboard/mouse action stream; treated as exogenous policy input",
        "future_pose_source": "map-aware rule dynamics rollout",
        "initial_velocity": "zero",
        "velocity_model": "rule",
        "look_scale_fit_uses_only_observed_prefix": True,
        "future_gt_state_poison_test": {
            "status": "pass",
            "poisoned_fields": list(STATE_FIELDS),
            "rows": non_leak_rows,
        },
        "trajectory_gt_used_for": "post-rollout evaluation only; never copied into generation inputs",
        "trajectory_evaluation": trajectory_rows,
        "look_scales": look_scales,
    }
    return output, audit


def memory_frame(memory: dict[str, np.ndarray], player_index: int, frame_index: int) -> dict[str, Any]:
    position = memory["position"][player_index, frame_index]
    alive = bool(memory["alive"][player_index, frame_index])
    health = float(memory["health"][player_index, frame_index]) if alive else 0.0
    return {
        "x": float(position[0]),
        "y": float(position[1]),
        "z": float(position[2]),
        "camera_position": memory["camera_position"][player_index, frame_index].astype(float).tolist(),
        "yaw": float(memory["yaw"][player_index, frame_index]),
        "pitch": float(memory["pitch"][player_index, frame_index]),
        "health": health,
        "alive": alive,
    }


def make_engine_state_cache(
    path: Path,
    raw_frames: list[int],
    memory: dict[str, np.ndarray],
    memory_meta: dict[str, Any],
    player_index: int,
    state_module: Any,
    mesh_projection: Any,
    inference_source: str,
) -> dict[str, Any]:
    alive = memory["alive"][player_index, raw_frames].astype(np.float32)
    health = np.clip(memory["health"][player_index, raw_frames] / 100.0, 0.0, 1.0).astype(np.float32)
    health[alive <= 0.0] = 0.0
    dead_masks = np.zeros((len(raw_frames), 60, 104), dtype=np.uint8)
    projected_counts: list[int] = []
    projected_stems: list[list[str]] = []
    ego_team = int(memory["team_id"][player_index])
    frame_count = int(memory_meta["frame_count"])
    other_frames_by_stem: dict[str, list[dict[str, Any]]] = {}
    for other_index, stem in enumerate(memory_meta["player_stems"]):
        if other_index == player_index or int(memory["team_id"][other_index]) == ego_team:
            continue
        frames: list[dict[str, Any]] = [{} for _ in range(frame_count)]
        for frame_index in raw_frames:
            frames[frame_index] = memory_frame(memory, other_index, frame_index)
        other_frames_by_stem[str(stem)] = frames
    ego_stem = str(memory_meta["player_stems"][player_index])
    for output_index, frame_index in enumerate(raw_frames):
        mask, count, stems = state_module.project_dead_mask_for_frame(
            mesh_mod=mesh_projection,
            other_frames_by_stem=other_frames_by_stem,
            ego_frame=memory_frame(memory, player_index, frame_index),
            ego_stem=ego_stem,
            frame_index=frame_index,
            latent_h=60,
            latent_w=104,
            video_h=TARGET_HEIGHT,
            video_w=TARGET_WIDTH,
            fov_x=106.26,
            far=4096.0,
            pitch_sign=-1.0,
        )
        dead_masks[output_index] = mask
        projected_counts.append(int(count))
        projected_stems.append(stems)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ego_alive=alive,
        ego_health=health,
        opponent_dead_mask=dead_masks,
        raw_frames=np.asarray(raw_frames, dtype=np.int32),
    )
    return {
        "kind": "state_channels_v0_record",
        "state_cache": str(path),
        "shape": {
            "ego_alive": [len(raw_frames)],
            "ego_health": [len(raw_frames)],
            "opponent_dead_mask": [len(raw_frames), 60, 104],
        },
        "channels": ["ego_alive_constant_plane", "ego_health_norm_constant_plane", "opponent_dead_marker_mask"],
        "inference_source": inference_source,
        "stats": {
            "ego_alive_values": sorted(float(value) for value in np.unique(alive)),
            "ego_health_values": sorted(float(value) for value in np.unique(health)),
            "ego_alive_changes": int(np.count_nonzero(np.diff(alive))),
            "ego_health_changes": int(np.count_nonzero(np.diff(health))),
            "opponent_dead_mask_frames": int(sum(value > 0 for value in projected_counts)),
            "opponent_dead_mask_pixels": int(dead_masks.sum()),
            "dead_projected_counts": projected_counts,
            "dead_projected_stems": projected_stems,
        },
    }


def prepare(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    dynamics_path: Path | None = None
    native_audit_path: Path | None = None
    engine_mode = args.engine_memory_dir is not None
    if engine_mode:
        engine_memory_dir = args.engine_memory_dir.resolve()
        engine_meta_path = engine_memory_dir / "episode_memory_meta_v1.json"
        engine_npz_path = engine_memory_dir / "episode_memory_v1.npz"
        native_audit_path = engine_memory_dir.parent / "SRCSD_NATIVE_ROLLOUT_AUDIT_v1.json"
        for required_path in (engine_meta_path, engine_npz_path, native_audit_path):
            if not required_path.is_file():
                raise RuntimeError(f"required native inference artifact is missing: {required_path}")
        source_meta = read_json(engine_meta_path)
        native_audit = read_json(native_audit_path)
        if source_meta.get("kind") != "srcds_native_inference_episode_memory_v1":
            raise RuntimeError(f"unexpected native memory kind: {source_meta.get('kind')}")
        if source_meta.get("future_replay_actions_read") is not False or source_meta.get("future_replay_state_read") is not False:
            raise RuntimeError("native memory does not satisfy the no-future-replay contract")
        if source_meta.get("valid_frame_range") != [args.start, args.start + args.horizon]:
            raise RuntimeError(f"native memory frame range mismatch: {source_meta.get('valid_frame_range')}")
        if native_audit.get("status") != "pass" or int(native_audit.get("export_frames", 0)) != args.horizon + 1:
            raise RuntimeError(f"native rollout audit is not pass: {native_audit_path}")
        if native_audit.get("tick_step_values") != [4] or float(native_audit.get("max_position_step_at_32hz", math.inf)) > 50.0:
            raise RuntimeError(f"native rollout continuity contract failed: {native_audit_path}")
        with np.load(engine_npz_path) as payload:
            simulator = as_memory_dict(payload)
        required_fields = {"position", "camera_position", "yaw", "pitch", "health", "alive", "team_id", "track_length"}
        missing_fields = sorted(required_fields.difference(simulator))
        if missing_fields:
            raise RuntimeError(f"native memory fields missing: {missing_fields}")
        sl = slice(args.start, args.start + args.horizon + 1)
        for field in ("position", "camera_position", "yaw", "pitch", "health"):
            if not np.all(np.isfinite(simulator[field][:, sl])):
                raise RuntimeError(f"native memory has non-finite {field} in rollout range")
        source = simulator
        trajectory_rows = []
        for player_index, stem in enumerate(source_meta["player_stems"]):
            positions = simulator["position"][player_index, sl].astype(np.float64)
            health = simulator["health"][player_index, sl]
            alive = simulator["alive"][player_index, sl].astype(np.int8)
            trajectory_rows.append(
                {
                    "player_index": player_index,
                    "player_stem": stem,
                    "policy_source": source_meta["policy_source"],
                    "final_displacement_3d": float(np.linalg.norm(positions[-1] - positions[0])),
                    "max_step_3d": float(np.linalg.norm(np.diff(positions, axis=0), axis=1).max()),
                    "health_change_count": int(np.count_nonzero(np.diff(health))),
                    "alive_change_count": int(np.count_nonzero(np.diff(alive))),
                }
            )
        inference_audit = {
            "mode": "srcds_native_inference",
            "policy_source": source_meta["policy_source"],
            "initial_state_frame": args.start,
            "observed_replay_frames_read": [args.start - 1, args.start],
            "future_control_source": "native CS:GO bot AI",
            "future_pose_source": "native CS:GO srcds",
            "future_replay_actions_read": False,
            "future_replay_state_read": False,
            "future_gt_state_poison_test": {
                "status": "not_applicable_native_engine",
                "reason": "srcds receives only the serialized t0 plan; future replay arrays are absent from its process inputs",
                "rows": [],
            },
            "native_rollout_contract": native_audit,
            "trajectory_evaluation": trajectory_rows,
        }
        memory_kind = "srcds_native_inference_episode_memory_v1"
    else:
        replay_dir = output_root / "replay_memory_input_only"
        subprocess.run(
            [
                sys.executable, str(project_root / "tools/build_episode_memory_v0.py"),
                "--match-dir", str(args.match_dir), "--episode", args.episode, "--out-dir", str(replay_dir),
            ],
            cwd=project_root,
            check=True,
        )
        source_meta = read_json(replay_dir / "episode_memory_meta_v0.json")
        with np.load(replay_dir / "episode_memory_v0.npz") as payload:
            source = as_memory_dict(payload)
        if args.start < 512:
            raise RuntimeError("need at least 512 observed prefix frames for a stable no-future look calibration")
        if args.start + args.horizon >= int(np.min(source["track_length"])):
            raise RuntimeError("rollout range exceeds source control stream")
        dynamics_path = project_root / "tools/evaluate_map_aware_dynamics_v1.py"
        if not dynamics_path.is_file():
            raise RuntimeError(f"required Source-coordinate dynamics v1 is missing: {dynamics_path}")
        dynamics = load_module(dynamics_path, "dynamics_inference_eval_v1")
        navmesh = dynamics.NavmeshIndex.from_json(args.navmesh_path)
        bsp_ground = dynamics.BspGroundIndex.from_npz(args.bsp_faces, 128.0, 55.0)
        motion_params = read_json(args.motion_params)
        simulator, inference_audit = strict_rollout(
            module=dynamics,
            source=source,
            source_meta=source_meta,
            navmesh=navmesh,
            bsp_ground=bsp_ground,
            motion_params=motion_params,
            start=args.start,
            horizon=args.horizon,
            fps=args.fps,
            look_lag=args.look_lag,
            action_lag=args.action_lag,
        )
        memory_kind = "strict_dynamics_inference_episode_memory_v0"
    sim_dir = output_root / "simulator_memory"
    sim_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sim_dir / "episode_memory_v0.npz", **simulator)
    sim_meta = dict(source_meta)
    sim_meta.update(
        {
            "kind": memory_kind,
            "npz_path": str(sim_dir / "episode_memory_v0.npz"),
            "source_episode_memory": str(args.engine_memory_dir.resolve()) if engine_mode else str(replay_dir),
            "inference_audit": inference_audit,
        }
    )
    write_json(sim_dir / "episode_memory_meta_v0.json", sim_meta)

    dense_module = load_module(project_root / "tools/build_mesh_dense_condition_v0.py", "dense_inference_builder_v0")
    pose_module = load_module(project_root / "tools/build_light_dust2_pilot3_source_v0.py", "pose_inference_builder_v0")
    state_module = load_module(project_root / "tools/build_state_channels_v0.py", "state_inference_builder_v0")
    mesh_projection = load_module(project_root / "tools/build_mesh_projection_v0.py", "state_mesh_projection_v0")
    cache = dense_module.load_renderer_cache(args.match_dir, project_root / "tools", bsp_faces_npz=args.bsp_faces)
    simulator_memory = dense_module.load_episode_memory(sim_dir)
    latent_frames, video_frames = frame_contract(args.start, args.horizon)
    requested = [value.strip() for value in args.ego_stems.split(",") if value.strip()]
    if not requested:
        team2 = [stem for stem in source_meta["player_stems"] if "_team_2_" in stem]
        team3 = [stem for stem in source_meta["player_stems"] if "_team_3_" in stem]
        requested = [team2[0], team3[0]]
    if len(requested) != 2 or len(set(requested)) != 2:
        raise RuntimeError("v0 requires exactly two distinct ego stems")

    state_rows = []
    view_rows = []
    for view_index, stem in enumerate(requested):
        if stem not in source_meta["player_stems"]:
            raise RuntimeError(f"unknown ego stem: {stem}")
        player_index = source_meta["player_stems"].index(stem)
        clip_id = f"dynamics_inference_{args.episode}_{stem}_f{args.start:06d}"
        clip_dir = output_root / "clips" / clip_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        source_video = args.match_dir / "train" / args.episode / f"{stem}.mp4"
        extract_frame(source_video, args.start, clip_dir / "image.jpg")
        (clip_dir / "prompt.txt").write_text(DEFAULT_PROMPT + "\n", encoding="utf-8")
        poses = np.zeros((VIDEO_FRAMES, 4, 4), dtype=np.float32)
        poses[:, 3, 3] = 1.0
        for pose_index, raw_frame in enumerate(video_frames):
            frame = {
                "camera_rotation": [
                    0.0,
                    float(simulator["pitch"][player_index, raw_frame]),
                    float(simulator["yaw"][player_index, raw_frame]),
                ],
            }
            poses[pose_index, :3, :3] = pose_module.pose_rotation_from_action_frame(frame)
            poses[pose_index, :3, 3] = simulator["camera_position"][player_index, raw_frame]
        intrinsics = np.repeat(pose_module.INTRINSICS_832_480[None, :], VIDEO_FRAMES, axis=0)
        np.save(clip_dir / "poses.npy", poses)
        np.save(clip_dir / "intrinsics.npy", intrinsics)

        dense_root = output_root / "dense" / clip_id
        dense_rows = []
        for latent_index, raw_frame in enumerate(latent_frames):
            sample_id = f"dynamics_inference__{args.episode}_{stem}_f{raw_frame:06d}"
            sample_dir = dense_root / "samples" / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            dense_path = sample_dir / "mesh_dense_condition_v0.npz"
            dense, _, _, _, _ = dense_module.render_memory_dense_condition(
                args.match_dir,
                args.episode,
                stem,
                raw_frame,
                320,
                176,
                106.26,
                4.0,
                3000.0,
                1.0,
                0,
                cache=cache,
                episode_memory=simulator_memory,
                player_mask_mode="capsule",
                mesh_backend=args.mesh_backend,
            )
            np.savez_compressed(dense_path, dense=np.asarray(dense, dtype=np.float16))
            dense_rows.append(
                {
                    "index": latent_index,
                    "latent_index": latent_index,
                    "raw_frame": raw_frame,
                    "gen_frame": latent_index * 4,
                    "sample_id": sample_id,
                    "dense_path": str(dense_path),
                    "state_source": memory_kind,
                }
            )
        dense_manifest = dense_root / "dense_sequence_manifest_v0.jsonl"
        write_jsonl(dense_manifest, dense_rows)

        state_path = output_root / "state" / "cache" / f"{clip_id}.npz"
        state_row = make_engine_state_cache(
            state_path,
            latent_frames,
            simulator,
            source_meta,
            player_index,
            state_module,
            mesh_projection,
            "authoritative native CS:GO engine state at every latent frame"
            if engine_mode
            else "map-aware rule rollout state at every latent frame",
        )
        state_row["clip_id"] = clip_id
        state_rows.append(state_row)
        manifest_row = {
            "pair_id": "dynamics_inference_pair_v0",
            "side": f"view_{view_index:02d}",
            "match_id": args.match_dir.name,
            "game_id": args.match_dir.name,
            "episode": args.episode,
            "ego": stem,
            "player_stem": stem,
            "clip_id": clip_id,
            "clip_dir": str(clip_dir),
            "image": str(clip_dir / "image.jpg"),
            "poses": str(clip_dir / "poses.npy"),
            "intrinsics": str(clip_dir / "intrinsics.npy"),
            "prompt_txt": str(clip_dir / "prompt.txt"),
            "action_json": str(args.match_dir / "train" / args.episode / f"{stem}.json"),
            "raw_start": args.start,
            "frame_count_start": args.start,
            "frame_count_end": args.start + args.horizon,
            "positive_latent_frames": latent_frames,
            "raw_indices": latent_frames,
            "latent_frames": LATENT_FRAMES,
            "phase2a_window_id": clip_id,
            "phase2a_dense_sequence_manifest": str(dense_manifest),
            "inference_contract": {
                "future_pose_source": "native CS:GO srcds" if engine_mode else "map-aware rule dynamics",
                "future_gt_pose_read": False,
                "control_source": "native CS:GO bot AI" if engine_mode else "recorded keyboard/mouse stream",
                "future_replay_action_read": False if engine_mode else True,
                "future_replay_state_read": False,
                "simulator_memory": str(sim_dir),
            },
        }
        manifest_path = output_root / "manifests" / f"view_{view_index:02d}.jsonl"
        write_jsonl(manifest_path, [manifest_row])
        view_rows.append(
            {
                "view_index": view_index,
                "player_index": player_index,
                "player_stem": stem,
                "source_manifest": str(manifest_path),
                "dense_manifest": str(dense_manifest),
                "clip_dir": str(clip_dir),
            }
        )
    state_manifest = output_root / "state" / "state_cache_manifest.jsonl"
    write_jsonl(state_manifest, state_rows)
    bindings = {
        "bsp_faces": {"path": str(args.bsp_faces), "sha256": sha256_file(args.bsp_faces)},
        "navmesh": {"path": str(args.navmesh_path), "sha256": sha256_file(args.navmesh_path)},
    }
    if engine_mode:
        assert args.engine_memory_dir is not None and native_audit_path is not None
        engine_meta_path = args.engine_memory_dir / "episode_memory_meta_v1.json"
        engine_npz_path = args.engine_memory_dir / "episode_memory_v1.npz"
        bindings.update(
            {
                "native_rollout_audit": {"path": str(native_audit_path), "sha256": sha256_file(native_audit_path)},
                "native_episode_memory": {"path": str(engine_npz_path), "sha256": sha256_file(engine_npz_path)},
                "native_episode_meta": {"path": str(engine_meta_path), "sha256": sha256_file(engine_meta_path)},
            }
        )
    else:
        assert dynamics_path is not None
        bindings.update(
            {
                "dynamics_engine": {"path": str(dynamics_path), "sha256": sha256_file(dynamics_path)},
                "motion_params": {"path": str(args.motion_params), "sha256": sha256_file(args.motion_params)},
            }
        )
    report = {
        "kind": "dynamics_inference_demo_prepare_v0",
        "status": "pass",
        "inference_mode": "srcds_native_inference" if engine_mode else "map_aware_rule_baseline",
        "project_root": str(project_root),
        "match_dir": str(args.match_dir),
        "episode": args.episode,
        "start": args.start,
        "horizon": args.horizon,
        "fps": args.fps,
        "views": view_rows,
        "state_manifest": str(state_manifest),
        "simulator_memory": str(sim_dir),
        "inference_audit": inference_audit,
        "bindings": bindings,
    }
    write_json(output_root / "PREPARE_REPORT_v0.json", report)
    print(json.dumps({"status": "pass", "output_root": str(output_root), "views": requested}, ensure_ascii=False))


def generate_one(args: argparse.Namespace) -> None:
    report = read_json(args.output_root / "PREPARE_REPORT_v0.json")
    if report.get("status") != "pass":
        raise RuntimeError("prepare report is not pass")
    view = next((row for row in report["views"] if int(row["view_index"]) == args.view), None)
    if view is None:
        raise RuntimeError(f"unknown view index: {args.view}")
    generation_root = args.output_root / "generation" / f"view_{args.view:02d}"
    if generation_root.exists() and any(path.is_file() for path in generation_root.rglob("*")):
        raise RuntimeError(f"refusing to overwrite generation: {generation_root}")
    source_rows = read_jsonl(Path(view["source_manifest"]))
    if len(source_rows) != 1:
        raise RuntimeError("v0 runtime source manifest must contain exactly one row")
    source_row = dict(source_rows[0])
    original_clip_id = str(source_row["clip_id"])
    compatible_clip_id = f"{original_clip_id}_{int(source_row['raw_start']):07d}"
    source_row["clip_id"] = compatible_clip_id
    source_row["phase2a_window_id"] = compatible_clip_id
    source_row["original_inference_clip_id"] = original_clip_id
    runtime_dir = args.output_root / "runtime_manifests"
    runtime_source = runtime_dir / f"view_{args.view:02d}.jsonl"
    write_jsonl(runtime_source, [source_row])
    state_rows = read_jsonl(Path(report["state_manifest"]))
    matched = 0
    for state_row in state_rows:
        if str(state_row["clip_id"]) == original_clip_id:
            state_row["clip_id"] = compatible_clip_id
            state_row["original_inference_clip_id"] = original_clip_id
            matched += 1
    if matched != 1:
        raise RuntimeError(f"expected one state row for {original_clip_id}, got {matched}")
    runtime_state = runtime_dir / f"state_view_{args.view:02d}.jsonl"
    write_jsonl(runtime_state, state_rows)
    command = [
        sys.executable,
        str(args.qxq_sampler),
        "--source-manifest", str(runtime_source),
        "--state-cache-manifest", str(runtime_state),
        "--adapter-checkpoint-low", str(args.checkpoint_low),
        "--adapter-checkpoint-high", str(args.checkpoint_high),
        "--out-root", str(generation_root),
        "--latent-frames", str(LATENT_FRAMES),
        "--chunk-size", "3",
        "--steps", str(args.steps),
        "--shift", str(args.shift),
        "--guide", str(args.guide),
        "--size", f"{TARGET_WIDTH}*{TARGET_HEIGHT}",
        "--seed", str(args.seed),
        "--device-id", "0",
        "--fps", "16",
    ]
    if args.print_only:
        print(json.dumps({"command": command}, ensure_ascii=False))
        return
    subprocess.run(command, cwd=args.qxq_sampler.parents[1], check=True)
    generated = read_json(generation_root / "candidate1_ar_run_report_v0.json")
    if generated.get("status") != "complete" or generated.get("completed_windows") != 1:
        raise RuntimeError(f"generation did not complete: {generation_root}")
    print(json.dumps({"status": "pass", "view": args.view, "generation_root": str(generation_root)}))


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
    return result


def normalize_video_height(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source),
            "-vf", f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:flags=lanczos",
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
            "-pix_fmt", "yuv420p", "-r", "16", "-movflags", "+faststart", str(output),
        ],
        check=True,
    )


def verify(args: argparse.Namespace) -> None:
    prepare_report = read_json(args.output_root / "PREPARE_REPORT_v0.json")
    native_mode = prepare_report.get("inference_mode") == "srcds_native_inference"
    poison = prepare_report["inference_audit"]["future_gt_state_poison_test"]
    if native_mode:
        native_contract = prepare_report["inference_audit"]["native_rollout_contract"]
        if (
            native_contract.get("status") != "pass"
            or native_contract.get("tick_step_values") != [4]
            or float(native_contract.get("max_position_step_at_32hz", math.inf)) > 50.0
            or prepare_report["inference_audit"].get("future_replay_actions_read") is not False
            or prepare_report["inference_audit"].get("future_replay_state_read") is not False
        ):
            raise RuntimeError("native srcds rollout gate is not pass")
    elif poison.get("status") != "pass" or not all(row.get("bitwise_invariant") for row in poison["rows"]):
        raise RuntimeError("future GT poison gate is not pass")
    trajectory_by_stem = {
        row["player_stem"]: row for row in prepare_report["inference_audit"]["trajectory_evaluation"]
    }
    dynamics_path = args.project_root / "tools/evaluate_map_aware_dynamics_v1.py"
    if not native_mode and not dynamics_path.is_file():
        raise RuntimeError(f"required dynamics engine is missing: {dynamics_path}")
    outputs = []
    delivery = args.output_root / "delivery"
    delivery.mkdir(parents=True, exist_ok=True)
    for view in prepare_report["views"]:
        index = int(view["view_index"])
        from PIL import Image

        initial_image = Path(view["clip_dir"]) / "image.jpg"
        with Image.open(initial_image) as image:
            if image.size != (TARGET_WIDTH, TARGET_HEIGHT):
                raise RuntimeError(f"initial image contract failed: {initial_image}: {image.size}")
        trajectory = trajectory_by_stem[view["player_stem"]]
        if native_mode:
            if float(trajectory["max_step_3d"]) > 50.0:
                raise RuntimeError(f"selected native trajectory continuity gate failed: {trajectory}")
        else:
            if float(trajectory["mean_gt_eval_error_3d"]) > args.max_selected_mean_error:
                raise RuntimeError(f"selected trajectory error gate failed: {trajectory}")
            if int(trajectory["clipped_steps"]) > args.max_selected_clipped_steps:
                raise RuntimeError(f"selected trajectory clipping gate failed: {trajectory}")
        report_path = args.output_root / "generation" / f"view_{index:02d}" / "candidate1_ar_run_report_v0.json"
        generated = read_json(report_path)
        if generated.get("status") != "complete" or generated.get("completed_windows") != 1 or generated.get("failed_windows") != 0:
            raise RuntimeError(f"incomplete generation: {report_path}")
        if not all(expert.get("step") == 75 and expert.get("state_projector_loaded") is True for expert in generated["experts"]):
            raise RuntimeError(f"checkpoint/state binding failed: {report_path}")
        stitched = Path(generated["stitched"]["mp4"])
        if not stitched.is_absolute():
            stitched = args.project_root / stitched
        raw_model_video = stitched
        raw_probe = video_probe(raw_model_video)
        if raw_probe == {"frames": 81, "fps": 16.0, "width": TARGET_WIDTH, "height": 464}:
            stitched = delivery / "views" / f"view_{index:02d}_832x480.mp4"
            normalize_video_height(raw_model_video, stitched)
        probe = video_probe(stitched)
        if probe != {"frames": 81, "fps": 16.0, "width": TARGET_WIDTH, "height": TARGET_HEIGHT}:
            raise RuntimeError(f"video contract failed: {stitched}: {probe}")
        outputs.append(
            {
                "view_index": index,
                "player_stem": view["player_stem"],
                "video": str(stitched),
                "raw_model_video": str(raw_model_video),
                "raw_model_probe": raw_probe,
                "sha256": sha256_file(stitched),
                "probe": probe,
                "generation_report": str(report_path),
                "trajectory_evaluation": trajectory,
            }
        )
    side_by_side = delivery / "dynamics_inference_2view_5s_v0.mp4"
    subprocess.run(
        [
            ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-y",
            "-i", outputs[0]["video"], "-i", outputs[1]["video"],
            "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]",
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
            "-pix_fmt", "yuv420p", "-r", "16", "-movflags", "+faststart", str(side_by_side),
        ],
        check=True,
    )
    side_probe = video_probe(side_by_side)
    if side_probe != {"frames": 81, "fps": 16.0, "width": TARGET_WIDTH * 2, "height": TARGET_HEIGHT}:
        raise RuntimeError(f"side-by-side contract failed: {side_probe}")
    result = {
        "kind": "dynamics_inference_demo_delivery_v0",
        "status": "pass",
        "inference_mode": prepare_report.get("inference_mode"),
        "claim": (
            "native CS:GO inference after one observed replay state; future policy, pose, health, weapon, and alive state come from srcds"
            if native_mode
            else "action-conditioned inference after one observed state; no future GT pose/state is read by rollout or generation inputs"
        ),
        "scope_note": (
            "The engine state channels are sampled per latent frame; this rollout may naturally contain no damage or death event."
            if native_mode
            else "recorded keyboard/mouse controls are exogenous inputs; game events are held constant in this v0"
        ),
        "dynamics_engine": (
            prepare_report["bindings"]["native_rollout_audit"]
            if native_mode
            else {"path": str(dynamics_path), "sha256": sha256_file(dynamics_path)}
        ),
        "trajectory_gates": {
            "max_selected_mean_error_3d": args.max_selected_mean_error,
            "max_selected_clipped_steps": args.max_selected_clipped_steps,
        },
        "prepare_report": str(args.output_root / "PREPARE_REPORT_v0.json"),
        "future_gt_poison_gate": poison,
        "native_rollout_contract": prepare_report["inference_audit"].get("native_rollout_contract"),
        "views": outputs,
        "side_by_side": {
            "path": str(side_by_side),
            "sha256": sha256_file(side_by_side),
            "probe": side_probe,
        },
    }
    write_json(delivery / "INFERENCE_DEMO_AUDIT_v0.json", result)
    print(json.dumps({"status": "pass", "video": str(side_by_side), "audit": str(delivery / "INFERENCE_DEMO_AUDIT_v0.json")}))


def self_test(_: argparse.Namespace) -> None:
    source = {
        "position": np.zeros((2, 8, 3), dtype=np.float32),
        "camera_position": np.zeros((2, 8, 3), dtype=np.float32),
        "yaw": np.zeros((2, 8), dtype=np.float32),
        "pitch": np.zeros((2, 8), dtype=np.float32),
        "health": np.full((2, 8), 100.0, dtype=np.float32),
        "alive": np.ones((2, 8), dtype=np.bool_),
        "actions": np.zeros((2, 8, 13), dtype=np.bool_),
        "look_delta": np.zeros((2, 8, 2), dtype=np.float32),
    }
    poisoned = poisoned_future(source, 3)
    for key in CONTROL_FIELDS:
        assert np.array_equal(source[key], poisoned[key])
    for key in STATE_FIELDS:
        assert np.array_equal(source[key][:, :4], poisoned[key][:, :4])
        assert not np.array_equal(source[key][:, 4:], poisoned[key][:, 4:])
    latent, video = frame_contract(900, 160)
    assert len(latent) == 21 and len(video) == 81 and latent[-1] == 1060 and video[-1] == 1060
    equal, diffs = rollout_equal(
        {"xy": np.zeros((2, 2)), "z": np.zeros(2), "yaw": np.zeros(2), "pitch": np.zeros(2)},
        {"xy": np.zeros((2, 2)), "z": np.zeros(2), "yaw": np.zeros(2), "pitch": np.zeros(2)},
    )
    assert equal and all(value == 0.0 for value in diffs.values())
    dynamics = load_module(Path(__file__).with_name("evaluate_map_aware_dynamics_v1.py"), "dynamics_self_test_v1")
    forward, right = dynamics.yaw_basis(0.0)
    assert np.allclose(forward, [1.0, 0.0]) and np.allclose(right, [0.0, -1.0])
    right_action = np.zeros((13,), dtype=np.bool_)
    right_action[3] = True
    assert np.allclose(dynamics.wish_direction(right_action, 0.0), [0.0, -1.0])
    print(json.dumps({"status": "pass", "tests": 4}))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--project-root", type=Path, required=True)
    prep.add_argument("--match-dir", type=Path, required=True)
    prep.add_argument("--episode", required=True)
    prep.add_argument("--start", type=int, default=900)
    prep.add_argument("--horizon", type=int, default=160)
    prep.add_argument("--fps", type=float, default=32.0)
    prep.add_argument("--look-lag", type=int, default=1)
    prep.add_argument("--action-lag", type=int, default=0)
    prep.add_argument("--ego-stems", default="")
    prep.add_argument(
        "--engine-memory-dir",
        type=Path,
        default=None,
        help="Use a passed srcds native inference Episode Memory v1 instead of the Python rule baseline.",
    )
    prep.add_argument("--navmesh-path", type=Path, required=True)
    prep.add_argument("--bsp-faces", type=Path, required=True)
    prep.add_argument("--motion-params", type=Path, required=True)
    prep.add_argument("--mesh-backend", choices=["bsp_faces_cpu", "bsp_faces_gpu"], default="bsp_faces_cpu")
    prep.add_argument("--output-root", type=Path, required=True)
    prep.set_defaults(func=prepare)

    gen = sub.add_parser("generate-one")
    gen.add_argument("--output-root", type=Path, required=True)
    gen.add_argument("--view", type=int, required=True)
    gen.add_argument("--qxq-sampler", type=Path, required=True)
    gen.add_argument("--checkpoint-low", type=Path, required=True)
    gen.add_argument("--checkpoint-high", type=Path, required=True)
    gen.add_argument("--steps", type=int, default=70)
    gen.add_argument("--shift", type=float, default=10.0)
    gen.add_argument("--guide", type=float, default=5.0)
    gen.add_argument("--seed", type=int, default=20260722)
    gen.add_argument("--print-only", action="store_true")
    gen.set_defaults(func=generate_one)

    verify_cmd = sub.add_parser("verify")
    verify_cmd.add_argument("--project-root", type=Path, required=True)
    verify_cmd.add_argument("--output-root", type=Path, required=True)
    verify_cmd.add_argument("--max-selected-mean-error", type=float, default=120.0)
    verify_cmd.add_argument("--max-selected-clipped-steps", type=int, default=10)
    verify_cmd.set_defaults(func=verify)

    test = sub.add_parser("self-test")
    test.set_defaults(func=self_test)
    return ap


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
