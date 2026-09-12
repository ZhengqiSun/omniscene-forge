#!/usr/bin/env python3
"""Prepare and verify a 10-view, 10-second native-CS:GO inference demo."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

import run_dynamics_inference_demo_v0 as base


WINDOW_RAW_FRAMES = 160
WINDOW_COUNT = 2
TOTAL_RAW_FRAMES = WINDOW_RAW_FRAMES * WINDOW_COUNT
LATENT_STRIDE = 8
VIDEO_STRIDE = 2
LATENT_FRAMES = 21
VIDEO_FRAMES = 81
DELIVERY_FRAMES = 161


def native_inputs(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    memory_dir = args.engine_memory_dir.resolve()
    meta_path = memory_dir / "episode_memory_meta_v1.json"
    npz_path = memory_dir / "episode_memory_v1.npz"
    audit_path = memory_dir.parent / "SRCSD_NATIVE_ROLLOUT_AUDIT_v1.json"
    for path in (meta_path, npz_path, audit_path):
        if not path.is_file():
            raise RuntimeError(f"required native inference artifact is missing: {path}")
    meta = base.read_json(meta_path)
    audit = base.read_json(audit_path)
    if meta.get("kind") != "srcds_native_inference_episode_memory_v1":
        raise RuntimeError(f"unexpected native memory kind: {meta.get('kind')}")
    if meta.get("valid_frame_range") != [args.start, args.start + TOTAL_RAW_FRAMES]:
        raise RuntimeError(f"native memory range mismatch: {meta.get('valid_frame_range')}")
    if meta.get("future_replay_actions_read") is not False or meta.get("future_replay_state_read") is not False:
        raise RuntimeError("native memory violates the no-future-replay contract")
    if (
        audit.get("status") != "pass"
        or audit.get("export_frames") != TOTAL_RAW_FRAMES + 1
        or audit.get("tick_step_values") != [4]
        or float(audit.get("initial_position_error_max", math.inf)) > 0.05
        or float(audit.get("max_position_step_at_32hz", math.inf)) > 50.0
    ):
        raise RuntimeError(f"native rollout contract is not pass: {audit_path}")
    with np.load(npz_path) as payload:
        memory = base.as_memory_dict(payload)
    rollout = slice(args.start, args.start + TOTAL_RAW_FRAMES + 1)
    for field in ("position", "camera_position", "yaw", "pitch", "health"):
        if not np.all(np.isfinite(memory[field][:, rollout])):
            raise RuntimeError(f"non-finite {field} in native rollout")
    if len(meta["player_stems"]) != 10:
        raise RuntimeError(f"expected ten native players, got {len(meta['player_stems'])}")
    return memory, meta, audit


def frame_indices(start: int) -> tuple[list[int], list[int]]:
    latent = list(range(start, start + WINDOW_RAW_FRAMES + 1, LATENT_STRIDE))
    video = list(range(start, start + WINDOW_RAW_FRAMES + 1, VIDEO_STRIDE))
    if len(latent) != LATENT_FRAMES or len(video) != VIDEO_FRAMES:
        raise AssertionError("window frame contract mismatch")
    return latent, video


def prepare(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    memory, meta, native_audit = native_inputs(args)

    sim_dir = output_root / "simulator_memory"
    sim_dir.mkdir(parents=True)
    np.savez_compressed(sim_dir / "episode_memory_v0.npz", **memory)
    sim_meta = dict(meta)
    sim_meta.update(
        {
            "npz_path": str(sim_dir / "episode_memory_v0.npz"),
            "source_episode_memory": str(args.engine_memory_dir.resolve()),
        }
    )
    base.write_json(sim_dir / "episode_memory_meta_v0.json", sim_meta)

    dense_module = base.load_module(project_root / "tools/build_mesh_dense_condition_v0.py", "native10s_dense_v1")
    pose_module = base.load_module(project_root / "tools/build_light_dust2_pilot3_source_v0.py", "native10s_pose_v1")
    state_module = base.load_module(project_root / "tools/build_state_channels_v0.py", "native10s_state_v1")
    mesh_projection = base.load_module(project_root / "tools/build_mesh_projection_v0.py", "native10s_projection_v1")
    renderer_cache = dense_module.load_renderer_cache(
        args.match_dir, project_root / "tools", bsp_faces_npz=args.bsp_faces
    )
    simulator_memory = dense_module.load_episode_memory(sim_dir)

    state_rows: list[dict[str, Any]] = []
    view_rows: list[dict[str, Any]] = []
    for view_index, stem in enumerate(meta["player_stems"]):
        player_index = int(view_index)
        observed_image = output_root / "observed_images" / f"{stem}_f{args.start:06d}.jpg"
        source_video = args.match_dir / "train" / args.episode / f"{stem}.mp4"
        base.extract_frame(source_video, args.start, observed_image)
        manifest_rows: list[dict[str, Any]] = []
        window_rows: list[dict[str, Any]] = []
        for window_index in range(WINDOW_COUNT):
            raw_start = args.start + window_index * WINDOW_RAW_FRAMES
            latent_frames, video_frames = frame_indices(raw_start)
            clip_id = f"native10s_{args.episode}_{stem}_w{window_index:02d}_f{raw_start:06d}_{raw_start:07d}"
            clip_dir = output_root / "clips" / clip_id
            clip_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(observed_image, clip_dir / "image.jpg")
            (clip_dir / "prompt.txt").write_text(base.DEFAULT_PROMPT + "\n", encoding="utf-8")

            poses = np.zeros((VIDEO_FRAMES, 4, 4), dtype=np.float32)
            poses[:, 3, 3] = 1.0
            for pose_index, raw_frame in enumerate(video_frames):
                frame = {
                    "camera_rotation": [
                        0.0,
                        float(memory["pitch"][player_index, raw_frame]),
                        float(memory["yaw"][player_index, raw_frame]),
                    ]
                }
                poses[pose_index, :3, :3] = pose_module.pose_rotation_from_action_frame(frame)
                poses[pose_index, :3, 3] = memory["camera_position"][player_index, raw_frame]
            intrinsics = np.repeat(pose_module.INTRINSICS_832_480[None, :], VIDEO_FRAMES, axis=0)
            np.save(clip_dir / "poses.npy", poses)
            np.save(clip_dir / "intrinsics.npy", intrinsics)

            dense_root = output_root / "dense" / clip_id
            dense_rows = []
            for latent_index, raw_frame in enumerate(latent_frames):
                sample_id = f"native10s__{args.episode}_{stem}_w{window_index:02d}_f{raw_frame:06d}"
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
                    cache=renderer_cache,
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
                        "state_source": "srcds_native_inference_episode_memory_v1",
                    }
                )
            dense_manifest = dense_root / "dense_sequence_manifest_v0.jsonl"
            base.write_jsonl(dense_manifest, dense_rows)

            state_path = output_root / "state" / "cache" / f"{clip_id}.npz"
            state_row = base.make_engine_state_cache(
                state_path,
                latent_frames,
                memory,
                meta,
                player_index,
                state_module,
                mesh_projection,
                "authoritative native CS:GO engine state at every latent frame",
            )
            state_row["clip_id"] = clip_id
            state_rows.append(state_row)

            manifest_rows.append(
                {
                    "pair_id": f"native10s_view_{view_index:02d}",
                    "side": f"window_{window_index:02d}",
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
                    "action_json_consumed_by_sampler": False,
                    "raw_start": raw_start,
                    "frame_count_start": raw_start,
                    "frame_count_end": raw_start + WINDOW_RAW_FRAMES,
                    "positive_latent_frames": latent_frames,
                    "raw_indices": latent_frames,
                    "latent_frames": LATENT_FRAMES,
                    "phase2a_window_id": clip_id,
                    "phase2a_dense_sequence_manifest": str(dense_manifest),
                    "inference_contract": {
                        "future_pose_source": "native CS:GO srcds",
                        "future_gt_pose_read": False,
                        "future_replay_action_read": False,
                        "future_replay_state_read": False,
                        "window_context": "observed t0 image" if window_index == 0 else "previous generated tail frame",
                        "clip_image_role": "consumed" if window_index == 0 else "schema placeholder only",
                    },
                }
            )
            window_rows.append(
                {
                    "window_index": window_index,
                    "raw_range": [raw_start, raw_start + WINDOW_RAW_FRAMES],
                    "clip_id": clip_id,
                    "clip_dir": str(clip_dir),
                    "dense_manifest": str(dense_manifest),
                    "state_cache": str(state_path),
                    "pose_rotation_delta": float(np.max(np.abs(poses[-1, :3, :3] - poses[0, :3, :3]))),
                }
            )
        manifest_path = output_root / "manifests" / f"view_{view_index:02d}_{stem}.jsonl"
        base.write_jsonl(manifest_path, manifest_rows)
        view_rows.append(
            {
                "view_index": view_index,
                "player_index": player_index,
                "player_stem": stem,
                "source_manifest": str(manifest_path),
                "generation_root": str(output_root / "generation" / f"view_{view_index:02d}_{stem}"),
                "windows": window_rows,
            }
        )

    state_manifest = output_root / "state" / "state_cache_manifest.jsonl"
    base.write_jsonl(state_manifest, state_rows)
    report = {
        "kind": "native_dynamics_10view_10s_prepare_v1",
        "status": "pass",
        "inference_mode": "srcds_native_inference",
        "project_root": str(project_root),
        "match_dir": str(args.match_dir.resolve()),
        "episode": args.episode,
        "start": args.start,
        "horizon": TOTAL_RAW_FRAMES,
        "fps": 32.0,
        "view_count": 10,
        "windows_per_view": WINDOW_COUNT,
        "state_manifest": str(state_manifest),
        "simulator_memory": str(sim_dir),
        "native_rollout_contract": native_audit,
        "future_replay_actions_read": False,
        "future_replay_state_read": False,
        "views": view_rows,
        "bindings": {
            "native_episode_memory": {
                "path": str(args.engine_memory_dir.resolve() / "episode_memory_v1.npz"),
                "sha256": base.sha256_file(args.engine_memory_dir.resolve() / "episode_memory_v1.npz"),
            },
            "native_rollout_audit": {
                "path": str(args.engine_memory_dir.resolve().parent / "SRCSD_NATIVE_ROLLOUT_AUDIT_v1.json"),
                "sha256": base.sha256_file(args.engine_memory_dir.resolve().parent / "SRCSD_NATIVE_ROLLOUT_AUDIT_v1.json"),
            },
            "bsp_faces": {"path": str(args.bsp_faces.resolve()), "sha256": base.sha256_file(args.bsp_faces)},
        },
    }
    base.write_json(output_root / "PREPARE_REPORT_v1.json", report)
    print(json.dumps({"status": "pass", "views": 10, "windows": 20, "output_root": str(output_root)}))


def normalized_view(ffmpeg: str, window_paths: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    for path in window_paths:
        command.extend(["-i", str(path)])
    command.extend(
        [
            "-filter_complex",
            "[0:v]scale=832:480:flags=lanczos,setpts=PTS-STARTPTS[v0];"
            "[1:v]scale=832:480:flags=lanczos,select='not(eq(n,0))',setpts=N/(16*TB)[v1];"
            "[v0][v1]concat=n=2:v=1:a=0[out]",
            "-map", "[out]", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
            "-pix_fmt", "yuv420p", "-r", "16", "-movflags", "+faststart", str(output),
        ]
    )
    subprocess.run(command, check=True)


def make_mosaic(ffmpeg: str, inputs: list[Path], output: Path) -> None:
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    for path in inputs:
        command.extend(["-i", str(path)])
    scale = ";".join(f"[{index}:v]scale=416:240:flags=lanczos[v{index}]" for index in range(10))
    top = "".join(f"[v{index}]" for index in range(5)) + "hstack=inputs=5[top]"
    bottom = "".join(f"[v{index}]" for index in range(5, 10)) + "hstack=inputs=5[bottom]"
    filters = f"{scale};{top};{bottom};[top][bottom]vstack=inputs=2[out]"
    command.extend(
        [
            "-filter_complex", filters, "-map", "[out]", "-an", "-c:v", "libx264",
            "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "16",
            "-movflags", "+faststart", str(output),
        ]
    )
    subprocess.run(command, check=True)


def verify(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    report = base.read_json(output_root / "PREPARE_REPORT_v1.json")
    native = report["native_rollout_contract"]
    if (
        report.get("status") != "pass"
        or report.get("future_replay_actions_read") is not False
        or report.get("future_replay_state_read") is not False
        or native.get("status") != "pass"
        or native.get("export_frames") != TOTAL_RAW_FRAMES + 1
    ):
        raise RuntimeError("prepare/native contract is not pass")
    ffmpeg = base.ffmpeg_executable()
    delivery = output_root / "delivery"
    delivery.mkdir(parents=True, exist_ok=True)
    outputs = []
    view_paths = []
    for view in report["views"]:
        generation_root = Path(view["generation_root"])
        generated = base.read_json(generation_root / "candidate1_ar_run_report_v0.json")
        if generated.get("status") != "complete" or generated.get("completed_windows") != 2 or generated.get("failed_windows") != 0:
            raise RuntimeError(f"incomplete generation: {generation_root}")
        if not all(expert.get("step") == 75 and expert.get("state_projector_loaded") is True for expert in generated["experts"]):
            raise RuntimeError(f"checkpoint/state binding failed: {generation_root}")
        if len(generated["windows"]) != 2 or not all(window.get("frames") == 81 for window in generated["windows"]):
            raise RuntimeError(f"window frame contract failed: {generation_root}")
        if (
            generated["windows"][0].get("context_image_kind") != "original_clip_image"
            or generated["windows"][1].get("context_image_kind") != "generated_tail_frame"
        ):
            raise RuntimeError(f"autoregressive context contract failed: {generation_root}")
        window_paths = [Path(window["mp4"]) for window in generated["windows"]]
        view_index = int(view["view_index"])
        stem = str(view["player_stem"])
        output = delivery / "views" / f"view_{view_index:02d}_{stem}_10s_832x480.mp4"
        normalized_view(ffmpeg, window_paths, output)
        probe = base.video_probe(output)
        expected = {"frames": DELIVERY_FRAMES, "fps": 16.0, "width": 832, "height": 480}
        if probe != expected:
            raise RuntimeError(f"view delivery contract failed: {output}: {probe}")
        view_paths.append(output)
        outputs.append(
            {
                "view_index": view_index,
                "player_stem": stem,
                "video": str(output),
                "sha256": base.sha256_file(output),
                "probe": probe,
                "generation_report": str(generation_root / "candidate1_ar_run_report_v0.json"),
                "raw_window_videos": [str(path) for path in window_paths],
            }
        )
    mosaic = delivery / "native_dynamics_10view_10s_v1.mp4"
    make_mosaic(ffmpeg, view_paths, mosaic)
    mosaic_probe = base.video_probe(mosaic)
    expected_mosaic = {"frames": DELIVERY_FRAMES, "fps": 16.0, "width": 2080, "height": 480}
    if mosaic_probe != expected_mosaic:
        raise RuntimeError(f"mosaic contract failed: {mosaic_probe}")
    audit = {
        "kind": "native_dynamics_10view_10s_delivery_v1",
        "status": "pass",
        "claim": "ten synchronized views from one native CS:GO inference world; no future replay action/state/image is consumed",
        "view_count": 10,
        "duration_seconds": 10.0625,
        "native_rollout_contract": native,
        "checkpoint_contract": "W5 step75 low/high experts with state projectors",
        "views": outputs,
        "mosaic": {
            "path": str(mosaic),
            "sha256": base.sha256_file(mosaic),
            "probe": mosaic_probe,
        },
    }
    base.write_json(delivery / "NATIVE_DYNAMICS_10VIEW_10S_AUDIT_v1.json", audit)
    print(json.dumps({"status": "pass", "video": str(mosaic), "views": 10, "frames": DELIVERY_FRAMES}))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--project-root", type=Path, required=True)
    prep.add_argument("--match-dir", type=Path, required=True)
    prep.add_argument("--episode", required=True)
    prep.add_argument("--start", type=int, default=900)
    prep.add_argument("--engine-memory-dir", type=Path, required=True)
    prep.add_argument("--navmesh-path", type=Path, required=True)
    prep.add_argument("--bsp-faces", type=Path, required=True)
    prep.add_argument("--mesh-backend", choices=["bsp_faces_cpu", "bsp_faces_gpu"], default="bsp_faces_gpu")
    prep.add_argument("--output-root", type=Path, required=True)
    prep.set_defaults(func=prepare)
    verify_cmd = sub.add_parser("verify")
    verify_cmd.add_argument("--output-root", type=Path, required=True)
    verify_cmd.set_defaults(func=verify)
    return ap


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
