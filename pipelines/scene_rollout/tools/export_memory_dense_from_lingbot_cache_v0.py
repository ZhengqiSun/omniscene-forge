#!/usr/bin/env python3
"""Render a training-ready Map Memory dense release from LingBot cache rows.

The LingBot cache is window based: each row is an 81-frame RGB/camera clip with
21 latent frames.  This exporter renders one canonical Map Memory dense tensor
for every latent timestamp, computes teacher QA for that same frame, and only
accepts a cache row when all 21 latent frames pass the dense-data contract.

The paired output is:

- ``manifest.json`` plus dense/sample sidecars under ``samples/``
- ``channel_teacher_qa_v0/memory_dense_channels_vs_teacher_v0.json``
- ``training_readiness_v0.json``
- ``aligned_cache_manifest.jsonl`` with ``map_memory_sample_ids`` per latent
  frame, ready for ``train_memory_dense_adapter_v0.py train``
"""

from __future__ import annotations
from runtime_paths import source_path

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image

from build_memory_dense_aligned_cache_v0 import (
    DEFAULT_CACHE_MANIFEST,
    DEFAULT_SOURCE_MANIFEST,
    expected_sample_ids,
    latent_frame_source_positions,
    parse_clip_identity,
    validate_record_shapes,
)
from map_memory_training_data_v0 import CHANNELS, REQUIRED_BACKEND_ID, sha256_file, stable_bucket


DEFAULT_OUT_DIR = Path("output/memory_dense_adapter_v0/memory_dense_lingbot_cache_v0")
DEFAULT_RAW_ROOT = Path(str(source_path('assets', 'csgo-datasets-fullsubset')))
DEFAULT_BSP_FACES_NPZ = Path("docs/assets/bsp_static_geometry_v0/de_dust2_bsp_faces_v0.npz")


def import_tool(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def iter_jsonl(path: Path, limit: int | None = None, start_offset: int = 0):
    with path.open("r", encoding="utf-8") as f:
        count = 0
        yielded = 0
        for line in f:
            if not line.strip():
                continue
            if count < start_offset:
                count += 1
                continue
            yield json.loads(line)
            count += 1
            yielded += 1
            if limit is not None and yielded >= limit:
                break


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def clip_id_from_source_row(row: dict[str, Any]) -> str | None:
    clip_id = row.get("clip_id")
    if clip_id:
        return str(clip_id)
    required = ["hash", "game_id", "episode", "player_stem", "raw_start"]
    if all(key in row for key in required):
        return f"{row['hash']}_{row['game_id']}_{row['episode']}_{row['player_stem']}_{int(row['raw_start']):07d}"
    return None


def load_source_alignment_index(source_manifest: Path | None, *, video_frames: int, latent_frames: int) -> dict[str, dict[str, Any]]:
    if source_manifest is None or not source_manifest.exists():
        return {}
    positions = latent_frame_source_positions(video_frames, latent_frames)
    index: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(source_manifest):
        clip_id = clip_id_from_source_row(row)
        if not clip_id:
            continue
        slim = {
            key: row[key]
            for key in [
                "hash",
                "game_id",
                "episode",
                "player_stem",
                "raw_start",
                "frame_count_start",
                "frame_count_end",
                "round_freeze_end",
                "window_stride_raw",
                "dynamic_score",
                "mp4",
                "action_json",
                "episode_info",
                "video_manifest",
                "game_manifest",
                "world_events",
                "map_name",
                "pose_version",
                "hfov_source",
                "sample_dir",
                "clip_dir",
                "meta_json",
                "snapshot_source",
                "source_manifest_index",
                "source_shard_id",
                "source_shard_position_1based",
            ]
            if key in row
        }
        raw_indices = row.get("raw_indices")
        if isinstance(raw_indices, list):
            slim["raw_indices_len"] = len(raw_indices)
            slim["raw_indices_first"] = int(raw_indices[0]) if raw_indices else None
            slim["raw_indices_last"] = int(raw_indices[-1]) if raw_indices else None
            slim["raw_indices_latent_frames"] = [int(raw_indices[pos]) for pos in positions if pos < len(raw_indices)]
            if len(raw_indices) > 1:
                slim["raw_indices_stride_set"] = sorted({
                    int(raw_indices[i + 1]) - int(raw_indices[i])
                    for i in range(len(raw_indices) - 1)
                })
            else:
                slim["raw_indices_stride_set"] = []
        slim["clip_id"] = clip_id
        index[clip_id] = slim
    return index


def validate_source_alignment_summary(
    source: dict[str, Any] | None,
    raw_frames: list[int],
    *,
    raw_stride: int,
    video_frames: int,
) -> list[str]:
    if not source:
        return []
    errors: list[str] = []
    if source.get("raw_indices_len") != video_frames:
        errors.append(f"source raw_indices length {source.get('raw_indices_len')} != video_frames {video_frames}")
    if source.get("raw_indices_latent_frames") != raw_frames:
        errors.append(
            f"source raw_indices latent frames {source.get('raw_indices_latent_frames', [])[:5]}... "
            f"!= expected {raw_frames[:5]}..."
        )
    strides = source.get("raw_indices_stride_set")
    if strides is not None and strides != [raw_stride]:
        errors.append(f"source raw_indices strides {strides} != [{raw_stride}]")
    return errors


def depth_mae(row: dict[str, Any]) -> float | None:
    if row.get("depth_common_mae") is not None:
        return float(row["depth_common_mae"])
    depth_common = row.get("depth_common") or {}
    if depth_common.get("mae") is not None:
        return float(depth_common["mae"])
    return None


def summarize_qa(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_of(key: str) -> float | None:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def nested_mean(key: str, subkey: str) -> float | None:
        vals = [float(r[key][subkey]) for r in rows if r.get(key, {}).get(subkey) is not None]
        return float(np.mean(vals)) if vals else None

    positives = [
        r for r in rows
        if int(r.get("visible_teacher_players", 0) or 0) > 0
        or int(r.get("other_player_teacher_pixels", 0) or 0) > 0
    ]
    return {
        "hit_iou_mean": mean_of("hit_iou"),
        "semantic_binary_iou_mean": mean_of("semantic_binary_iou"),
        "other_player_iou_mean": mean_of("other_player_iou"),
        "positive_other_player_iou_mean": float(np.mean([float(r.get("other_player_iou", 0.0) or 0.0) for r in positives])) if positives else None,
        "positive_other_player_iou_min": min([float(r.get("other_player_iou", 0.0) or 0.0) for r in positives], default=None),
        "depth_common_mae_mean": nested_mean("depth_common", "mae"),
        "other_player_depth_common_mae_mean": nested_mean("other_player_depth_common", "mae"),
    }


def meta_matches_args(meta: dict[str, Any], dense: np.ndarray, args: argparse.Namespace) -> list[str]:
    mismatches: list[str] = []
    expected = {
        "fov_x": float(args.fov_x),
        "near": float(args.near),
        "far": float(args.far),
        "pitch_sign": float(args.pitch_sign),
        "player_radius": float(args.player_radius),
        "player_height": float(args.player_height),
        "occlusion_tolerance": float(args.occlusion_tolerance),
        "min_visible_pixels": int(args.min_visible_pixels),
        "player_z_offset": float(args.player_z_offset),
        "camera_yaw_offset": float(args.camera_yaw_offset),
        "camera_pitch_offset": float(args.camera_pitch_offset),
        "player_mask_mode": str(args.player_mask_mode),
        "player_screen_y_offset_px": float(args.player_screen_y_offset_px),
        "player_radius_scale": float(args.player_radius_scale),
        "mesh_backend": str(args.mesh_backend),
        "geometry_backend_id": REQUIRED_BACKEND_ID,
    }
    for key, value in expected.items():
        if key not in meta:
            mismatches.append(f"missing meta.{key}")
            continue
        got = meta[key]
        if isinstance(value, float):
            if abs(float(got) - value) > 1e-6:
                mismatches.append(f"meta.{key}={got!r} != {value!r}")
        elif got != value:
            mismatches.append(f"meta.{key}={got!r} != {value!r}")
    if list(dense.shape) != [7, args.height, args.width]:
        mismatches.append(f"dense.shape={list(dense.shape)} != {[7, args.height, args.width]}")
    if meta.get("channels") != CHANNELS:
        mismatches.append("meta.channels mismatch")
    return mismatches


def match_dir_for_identity(identity: dict[str, Any], raw_root: Path) -> Path:
    match_dir = raw_root / str(identity.get("hash", "32f1644d4f42c29d")) / str(identity["game_id"])
    if not match_dir.exists():
        raise FileNotFoundError(f"missing match_dir for {identity['clip_id']}: {match_dir}")
    return match_dir


def split_for_sample(sample: dict[str, Any], *, split_key: str, val_fraction: float, test_fraction: float, seed: int) -> str:
    if split_key == "episode":
        group = str(sample["episode"])
    elif split_key == "track":
        group = f"{sample['match_id']}|{sample['raw_episode']}|{sample['ego_stem']}"
    elif split_key == "match":
        group = str(sample["match_id"])
    else:
        raise ValueError(f"unknown split_key: {split_key}")
    bucket = stable_bucket(f"{seed}|{group}")
    val_cut = int(val_fraction * 10_000)
    test_cut = int((val_fraction + test_fraction) * 10_000)
    return "val" if bucket < val_cut else "test" if bucket < test_cut else "train"


def classify_sample(qa_row: dict[str, Any], args: argparse.Namespace) -> tuple[str | None, str | None]:
    visible = int(qa_row.get("visible_teacher_players", 0) or 0)
    teacher_pixels = int(qa_row.get("other_player_teacher_pixels", 0) or 0)
    memory_pixels = int(qa_row.get("other_player_memory_pixels", 0) or 0)
    hit_iou = float(qa_row.get("hit_iou", 0.0) or 0.0)
    mae = depth_mae(qa_row)
    other_iou = float(qa_row.get("other_player_iou", 0.0) or 0.0)
    if mae is None:
        return None, "missing_depth_common_mae"
    if hit_iou < args.min_hit_iou:
        return None, "hit_iou_below_gate"
    if mae > args.max_depth_mae:
        return None, "depth_mae_above_gate"
    teacher_positive = visible > 0 or teacher_pixels > 0
    memory_positive = memory_pixels > 0
    if teacher_positive and memory_positive:
        if other_iou < args.min_positive_other_iou:
            return None, "positive_other_iou_below_gate"
        return "positive", None
    if not teacher_positive and not memory_positive:
        return "context", None
    return None, "teacher_memory_other_player_mismatch"


def apply_teacher_positive_gate(
    qa_row: dict[str, Any],
    teacher_meta: dict[str, Any],
    *,
    min_teacher_player_pixels: int,
) -> dict[str, Any]:
    raw_visible_players = teacher_meta.get("visible_players", [])
    raw_teacher_pixels = int(qa_row.get("other_player_teacher_pixels", 0) or 0)
    gate_positive = raw_teacher_pixels >= min_teacher_player_pixels
    qa_row["raw_visible_teacher_players"] = len(raw_visible_players)
    qa_row["raw_other_player_teacher_pixels"] = raw_teacher_pixels
    qa_row["teacher_positive_gate_min_pixels"] = int(min_teacher_player_pixels)
    qa_row["visible_teacher_players"] = len(raw_visible_players) if gate_positive else 0
    qa_row["other_player_teacher_pixels"] = raw_teacher_pixels if gate_positive else 0
    qa_row["teacher_positive_gate_applied"] = True
    return qa_row


def render_or_load_sample(
    *,
    sample_id: str,
    match_dir: Path,
    episode: str,
    ego_stem: str,
    frame_index: int,
    out_dir: Path,
    cache: dict[str, Any],
    dense_mod: Any,
    qa_mod: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], str | None, str | None]:
    sample_dir = out_dir / "samples" / sample_id
    dense_path = sample_dir / "mesh_dense_condition_v0.npz"
    target_path = sample_dir / "target_rgb.png"
    meta_path = sample_dir / "mesh_dense_condition_meta_v0.json"
    qa_path = sample_dir / "mesh_dense_condition_qa_v0.png"
    render_dense_mod = cache["dense_mod"]

    if args.reuse_existing and dense_path.exists() and target_path.exists() and meta_path.exists() and qa_path.exists():
        dense = np.load(dense_path)["dense"].astype(np.float32)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        reuse_mismatches = meta_matches_args(meta, dense, args)
        if reuse_mismatches:
            shutil.rmtree(sample_dir, ignore_errors=True)
            dense = None
            meta = None
            rgb = None
        else:
            rgb = np.asarray(Image.open(target_path).convert("RGB"))
    else:
        dense = None
        meta = None
        rgb = None

    if dense is None or meta is None or rgb is None:
        sample_dir.mkdir(parents=True, exist_ok=True)
        dense, mesh_depth_units, rgb, meta, render_dense_mod = dense_mod.render_memory_dense_condition(
            match_dir,
            episode,
            ego_stem,
            frame_index,
            args.width,
            args.height,
            args.fov_x,
            args.near,
            args.far,
            args.pitch_sign,
            args.max_triangles,
            cache=cache,
            episode_memory=None,
            player_radius=args.player_radius,
            player_height=args.player_height,
            occlusion_tolerance=args.occlusion_tolerance,
            min_visible_pixels=args.min_visible_pixels,
            player_z_offset=args.player_z_offset,
            camera_yaw_offset=args.camera_yaw_offset,
            camera_pitch_offset=args.camera_pitch_offset,
            player_mask_mode=args.player_mask_mode,
            player_screen_y_offset_px=args.player_screen_y_offset_px,
            player_radius_scale=args.player_radius_scale,
            mesh_backend=args.mesh_backend,
        )
        np.savez_compressed(dense_path, dense=dense, mesh_depth_units=mesh_depth_units)
        Image.fromarray(rgb).resize((args.width, args.height), Image.Resampling.BILINEAR).save(target_path)
        meta["target_rgb_path"] = str(target_path)
        meta["target_policy"] = "Current-frame RGB target for LingBot latent-frame dense alignment; teacher streams are QA only."
        write_json(meta_path, meta)
        render_dense_mod.make_qa(qa_path, rgb, dense[0], dense[1], dense[3:], meta)

    episode_dir = match_dir / "train" / episode
    required = [
        episode_dir / f"{ego_stem}.mp4",
        episode_dir / f"{ego_stem}_depth.mkv",
        episode_dir / f"{ego_stem}_seg.mkv",
        episode_dir / f"{ego_stem}_player_visibility.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return {}, {}, None, f"missing_teacher_inputs:{missing[:3]}"

    teacher_depth, teacher_hit, teacher_semantic = qa_mod.teacher_env_channels(render_dense_mod, episode_dir, ego_stem, frame_index, args.height, args.width)
    teacher_player, teacher_meta = qa_mod.teacher_other_player_channels(render_dense_mod, episode_dir, ego_stem, frame_index, args.height, args.width)
    metrics = qa_mod.sample_metrics(dense, teacher_depth, teacher_hit, teacher_semantic, teacher_player)
    qa_row = {
        "sample_id": sample_id,
        "episode": f"{match_dir.name}_{episode}",
        "raw_episode": episode,
        "match_id": match_dir.name,
        "match_dir": str(match_dir),
        "ego_stem": ego_stem,
        "frame_index": frame_index,
        "visible_teacher_players": len(teacher_meta.get("visible_players", [])),
        **metrics,
    }
    qa_row = apply_teacher_positive_gate(
        qa_row,
        teacher_meta,
        min_teacher_player_pixels=args.min_teacher_player_pixels,
    )
    role, reject_reason = classify_sample(qa_row, args)
    if role is None:
        return {}, qa_row, None, reject_reason

    sample = {
        "sample_id": sample_id,
        "match_id": match_dir.name,
        "episode": f"{match_dir.name}_{episode}",
        "raw_episode": episode,
        "ego_stem": ego_stem,
        "frame_index": frame_index,
        "dense_path": str(dense_path),
        "dense_relpath": str(dense_path.relative_to(out_dir)),
        "target_rgb_path": str(target_path),
        "target_rgb_relpath": str(target_path.relative_to(out_dir)),
        "meta_path": str(meta_path),
        "meta_relpath": str(meta_path.relative_to(out_dir)),
        "qa_path": str(qa_path),
        "qa_relpath": str(qa_path.relative_to(out_dir)),
        "shape": list(dense.shape),
        "mesh_hit_ratio": meta["mesh_hit_ratio"],
        "nav_semantic_hit_ratio": meta.get("nav_semantic_hit_ratio"),
        "memory_projected_players": len(meta["memory_projected_players"]),
        "channels": meta["channels"],
        "mesh_backend": meta.get("mesh_backend"),
        "geometry_backend": "bsp_faces",
        "geometry_backend_id": meta.get("geometry_backend_id"),
        "geometry_backend_family": meta.get("geometry_backend_family"),
        "backend_features": meta.get("backend_features"),
        "backend_signature": meta.get("backend_signature"),
        "component_counts": meta.get("component_counts"),
        "selection_role": role,
        "teacher_qa_at_selection": {
            "hit_iou": qa_row.get("hit_iou"),
            "depth_common_mae": depth_mae(qa_row),
            "other_player_iou": qa_row.get("other_player_iou"),
            "visible_teacher_players": qa_row.get("visible_teacher_players"),
            "other_player_teacher_pixels": qa_row.get("other_player_teacher_pixels"),
            "other_player_memory_pixels": qa_row.get("other_player_memory_pixels"),
            "raw_visible_teacher_players": qa_row.get("raw_visible_teacher_players"),
            "raw_other_player_teacher_pixels": qa_row.get("raw_other_player_teacher_pixels"),
            "teacher_positive_gate_min_pixels": qa_row.get("teacher_positive_gate_min_pixels"),
        },
    }
    return sample, qa_row, role, None


def run_readiness_checker(manifest_path: Path, teacher_qa_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    readiness_path = args.out_dir / "training_readiness_v0.json"
    readiness_path.unlink(missing_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "check_map_memory_training_readiness_v0.py"),
        "--manifest",
        str(manifest_path),
        "--teacher-qa",
        str(teacher_qa_path),
        "--mode",
        "mixed",
        "--max-samples",
        str(max(1, args.readiness_max_samples)),
        "--min-samples",
        str(args.min_samples),
        "--min-episodes",
        str(args.min_episodes),
        "--require-geometry-backend-id",
        REQUIRED_BACKEND_ID,
        "--out-json",
        str(readiness_path),
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if readiness_path.exists():
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    else:
        readiness = {
            "kind": "map_memory_training_readiness_v0",
            "status": "fail",
            "manifest": str(manifest_path),
            "teacher_qa": str(teacher_qa_path),
            "failures": [proc.stdout[-4000:]],
        }
        write_json(readiness_path, readiness)
    readiness["checker_returncode"] = int(proc.returncode)
    if proc.returncode != 0:
        readiness["status"] = "fail"
        readiness["checker_output_tail"] = proc.stdout[-4000:]
        failures = list(readiness.get("failures", []))
        if proc.stdout:
            failures.append(proc.stdout[-4000:])
        readiness["failures"] = failures
        write_json(readiness_path, readiness)
    return readiness


def export_release(args: argparse.Namespace) -> dict[str, Any]:
    tools_dir = Path(__file__).resolve().parent
    dense_mod = import_tool(tools_dir / "build_mesh_dense_condition_v0.py", "mesh_dense_condition_v0")
    qa_mod = import_tool(tools_dir / "compare_memory_dense_channels_to_teacher_v0.py", "memory_teacher_qa_v0")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    renderer_caches: dict[str, dict[str, Any]] = {}
    samples_by_id: dict[str, dict[str, Any]] = {}
    qa_by_id: dict[str, dict[str, Any]] = {}
    aligned_rows: list[dict[str, Any]] = []
    source_index = load_source_alignment_index(
        args.source_manifest,
        video_frames=args.video_frames,
        latent_frames=args.latent_frames,
    )
    rejected_samples: Counter[str] = Counter()
    rejected_clips: Counter[str] = Counter()
    reject_metric_rows: list[dict[str, Any]] = []
    reject_examples: list[dict[str, Any]] = []
    scanned_clips = 0
    accepted_clips = 0
    rendered_samples = 0

    positions = latent_frame_source_positions(args.video_frames, args.latent_frames)
    for record in iter_jsonl(args.cache_manifest, limit=args.limit, start_offset=args.start_offset):
        scanned_clips += 1
        clip_id = str(record.get("clip_id", ""))
        if args.max_accepted_clips is not None and accepted_clips >= args.max_accepted_clips:
            break
        if args.progress_every > 0 and scanned_clips % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "event": "export_progress",
                        "scanned_clips": scanned_clips,
                        "accepted_clips": accepted_clips,
                        "rendered_samples": rendered_samples,
                        "rejected_clips": dict(sorted(rejected_clips.items())),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        try:
            source = source_index.get(clip_id)
            if args.require_source_raw_indices and not source:
                raise ValueError(f"source manifest has no row for {clip_id}")
            identity = parse_clip_identity(record, source)
            shape_errors = validate_record_shapes(
                record,
                latent_frames=args.latent_frames,
                video_height=args.video_height,
                video_width=args.video_width,
            )
            if shape_errors:
                raise ValueError("; ".join(shape_errors))
            match_dir = match_dir_for_identity(identity, args.raw_root)
            if match_dir.name not in renderer_caches:
                renderer_caches[match_dir.name] = dense_mod.load_renderer_cache(
                    match_dir,
                    tools_dir,
                    bsp_faces_npz=args.bsp_faces_npz,
                )
            raw_frames, sample_ids = expected_sample_ids(
                identity,
                video_frames=args.video_frames,
                raw_stride=args.raw_stride,
                latent_frames=args.latent_frames,
            )
            raw_errors = validate_source_alignment_summary(source, raw_frames, raw_stride=args.raw_stride, video_frames=args.video_frames)
            if args.require_source_raw_indices and raw_errors:
                raise ValueError("; ".join(raw_errors))
        except Exception as exc:
            rejected_clips["bad_clip_metadata"] += 1
            if len(reject_examples) < args.max_reject_examples:
                reject_examples.append({"clip_id": clip_id, "reason": "bad_clip_metadata", "detail": str(exc)})
            continue

        clip_samples: list[dict[str, Any]] = []
        clip_qa_rows: list[dict[str, Any]] = []
        pending_samples: dict[str, dict[str, Any]] = {}
        pending_qa_rows: dict[str, dict[str, Any]] = {}
        clip_reject_reason: str | None = None
        for sample_id, frame_index in zip(sample_ids, raw_frames):
            if sample_id in samples_by_id:
                clip_samples.append(samples_by_id[sample_id])
                clip_qa_rows.append(qa_by_id[sample_id])
                continue
            if sample_id in pending_samples:
                clip_samples.append(pending_samples[sample_id])
                clip_qa_rows.append(pending_qa_rows[sample_id])
                continue
            sample, qa_row, _, reject_reason = render_or_load_sample(
                sample_id=sample_id,
                match_dir=match_dir,
                episode=str(identity["episode"]),
                ego_stem=str(identity["player_stem"]),
                frame_index=int(frame_index),
                out_dir=args.out_dir,
                cache=renderer_caches[match_dir.name],
                dense_mod=dense_mod,
                qa_mod=qa_mod,
                args=args,
            )
            rendered_samples += 1
            if reject_reason is not None:
                rejected_samples[reject_reason] += 1
                clip_reject_reason = reject_reason
                reject_metric_rows.append(
                    {
                        "clip_id": clip_id,
                        "sample_id": sample_id,
                        "latent_frame_index": len(clip_samples),
                        "raw_frame_index": int(frame_index),
                        "reason": reject_reason,
                        "hit_iou": qa_row.get("hit_iou"),
                        "depth_common_mae": depth_mae(qa_row),
                        "other_player_iou": qa_row.get("other_player_iou"),
                        "visible_teacher_players": qa_row.get("visible_teacher_players"),
                        "other_player_teacher_pixels": qa_row.get("other_player_teacher_pixels"),
                        "raw_visible_teacher_players": qa_row.get("raw_visible_teacher_players"),
                        "raw_other_player_teacher_pixels": qa_row.get("raw_other_player_teacher_pixels"),
                        "other_player_memory_pixels": qa_row.get("other_player_memory_pixels"),
                    }
                )
                if len(reject_examples) < args.max_reject_examples:
                    reject_examples.append({"clip_id": clip_id, "sample_id": sample_id, "reason": reject_reason})
                if not args.keep_rejected_samples:
                    shutil.rmtree(args.out_dir / "samples" / sample_id, ignore_errors=True)
                break
            pending_samples[sample_id] = sample
            pending_qa_rows[sample_id] = qa_row
            clip_samples.append(sample)
            clip_qa_rows.append(qa_row)

        if clip_reject_reason is not None:
            rejected_clips[clip_reject_reason] += 1
            if not args.keep_rejected_samples:
                for pending_id in pending_samples:
                    shutil.rmtree(args.out_dir / "samples" / pending_id, ignore_errors=True)
            continue
        if len(clip_samples) != args.latent_frames:
            rejected_clips["incomplete_clip_samples"] += 1
            if not args.keep_rejected_samples:
                for pending_id in pending_samples:
                    shutil.rmtree(args.out_dir / "samples" / pending_id, ignore_errors=True)
            continue
        splits = {
            split_for_sample(
                sample,
                split_key=args.split_key,
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
                seed=args.seed,
            )
            for sample in clip_samples
        }
        if len(splits) != 1:
            rejected_clips["mixed_split_within_clip"] += 1
            if not args.keep_rejected_samples:
                for pending_id in pending_samples:
                    shutil.rmtree(args.out_dir / "samples" / pending_id, ignore_errors=True)
            continue
        samples_by_id.update(pending_samples)
        qa_by_id.update(pending_qa_rows)
        split = next(iter(splits))
        out_record = dict(record)
        source_for_record = source_index.get(clip_id)
        if source_for_record:
            for key, value in source_for_record.items():
                if key != "raw_indices":
                    out_record.setdefault(key, value)
        out_record.update(
            {
                "alignment_kind": "map_memory_dense_lingbot_latent_frame_exact_v0",
                "map_memory_manifest": str(args.out_dir / "manifest.json"),
                "map_memory_sample_ids": sample_ids,
                "map_memory_raw_frame_indices": raw_frames,
                "map_memory_split": split,
                "map_memory_split_key": args.split_key,
                "map_memory_match_id": identity["game_id"],
                "map_memory_episode": f"{identity['game_id']}_{identity['episode']}",
                "map_memory_raw_episode": identity["episode"],
                "map_memory_ego_stem": identity["player_stem"],
                "map_memory_track_id": f"{identity['game_id']}|{identity['episode']}|{identity['player_stem']}",
                "map_memory_selection_roles": [sample["selection_role"] for sample in clip_samples],
                "map_memory_positive_frames": sum(1 for sample in clip_samples if sample["selection_role"] == "positive"),
                "map_memory_context_frames": sum(1 for sample in clip_samples if sample["selection_role"] == "context"),
                "alignment_video_frames": args.video_frames,
                "alignment_raw_stride": args.raw_stride,
                "alignment_latent_frames": args.latent_frames,
                "alignment_latent_source_positions": positions,
                "alignment_exact_frame_match": True,
            }
        )
        aligned_rows.append(out_record)
        accepted_clips += 1

    samples = sorted(samples_by_id.values(), key=lambda row: row["sample_id"])
    qa_rows = [qa_by_id[sample["sample_id"]] for sample in samples]
    role_counts = Counter(sample["selection_role"] for sample in samples)
    match_counts = Counter(sample["match_id"] for sample in samples)
    episode_counts = Counter(sample["episode"] for sample in samples)
    backend_signatures = sorted({sample["backend_signature"] for sample in samples if sample.get("backend_signature")})
    manifest = {
        "kind": "memory_dense_lingbot_cache_release_v0",
        "source_cache_manifest": str(args.cache_manifest),
        "source_manifest": str(args.source_manifest) if args.source_manifest else None,
        "raw_root": str(args.raw_root),
        "sample_count": len(samples),
        "positive_sample_count": int(role_counts.get("positive", 0)),
        "context_sample_count": int(role_counts.get("context", 0)),
        "match_count": len(match_counts),
        "episode_count": len(episode_counts),
        "match_sample_counts": dict(sorted(match_counts.items())),
        "episode_sample_counts": dict(sorted(episode_counts.items())),
        "shape": [7, args.height, args.width],
        "channels": CHANNELS,
        "policy": "Inputs are generated from Map Memory using LingBot latent-frame timestamps; RGB/depth/seg/visibility streams are target/QA only.",
        "selection_policy": "Accept complete LingBot clips only when every latent frame passes teacher QA and exact Map Memory rendering.",
        "teacher_positive_policy": {
            "min_teacher_player_pixels": args.min_teacher_player_pixels,
            "reason": "Match renderer min_visible_pixels so tiny teacher-only specks do not reject complete clips.",
            "raw_teacher_counts_preserved": True,
        },
        "alignment": {
            "video_frames": args.video_frames,
            "raw_stride": args.raw_stride,
            "latent_frames": args.latent_frames,
            "latent_source_positions": positions,
            "frame_formula": "raw_start + raw_stride * latent_source_position",
            "nearest_or_repeat_used": False,
        },
        "player_capsule_params": {
            "player_radius": args.player_radius,
            "player_height": args.player_height,
            "occlusion_tolerance": args.occlusion_tolerance,
            "min_visible_pixels": args.min_visible_pixels,
            "player_mask_mode": args.player_mask_mode,
        },
        "camera_projection_params": {
            "fov_x": args.fov_x,
            "camera_yaw_offset": args.camera_yaw_offset,
            "camera_pitch_offset": args.camera_pitch_offset,
        },
        "mesh_backend": args.mesh_backend,
        "geometry_backend": "bsp_faces",
        "geometry_backend_id": samples[0].get("geometry_backend_id") if samples else REQUIRED_BACKEND_ID,
        "geometry_backend_family": samples[0].get("geometry_backend_family") if samples else "bsp_faces",
        "backend_features": samples[0].get("backend_features") if samples else ["visual_faces", "displacement"],
        "backend_signatures": backend_signatures,
        "bsp_faces_npz": str(args.bsp_faces_npz),
        "samples": samples,
    }
    manifest_path = args.out_dir / "manifest.json"
    write_json(manifest_path, manifest)
    map_manifest_sha256 = sha256_file(manifest_path)
    for row in aligned_rows:
        row["map_memory_manifest"] = str(manifest_path)
        row["map_memory_manifest_sha256"] = map_manifest_sha256

    teacher_qa = {
        "kind": "memory_dense_channels_vs_teacher_v0",
        "manifest": str(manifest_path),
        "sample_count": len(qa_rows),
        "missing_count": 0,
        "summary": summarize_qa(qa_rows),
        "rows": qa_rows,
        "missing": [],
        "policy": "Teacher streams are used only for QA. Memory dense tensors remain Map-rendered inputs.",
    }
    teacher_qa_path = args.out_dir / "channel_teacher_qa_v0" / "memory_dense_channels_vs_teacher_v0.json"
    write_json(teacher_qa_path, teacher_qa)

    readiness = run_readiness_checker(manifest_path, teacher_qa_path, args)
    readiness_path = args.out_dir / "training_readiness_v0.json"

    aligned_path = args.out_dir / "aligned_cache_manifest.jsonl"
    with aligned_path.open("w", encoding="utf-8") as f:
        for row in aligned_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    export_failures: list[str] = []
    if accepted_clips < args.min_accepted_clips:
        export_failures.append(f"accepted_clips {accepted_clips} < min_accepted_clips {args.min_accepted_clips}")
    if int(role_counts.get("positive", 0)) < args.min_positive_samples:
        export_failures.append(f"positive_sample_count {int(role_counts.get('positive', 0))} < min_positive_samples {args.min_positive_samples}")
    positive_other_iou_mean = teacher_qa["summary"].get("positive_other_player_iou_mean")
    if positive_other_iou_mean is not None and float(positive_other_iou_mean) < args.min_positive_other_iou_mean:
        export_failures.append(
            f"positive_other_player_iou_mean {positive_other_iou_mean} < {args.min_positive_other_iou_mean}"
        )
    metric_summary: dict[str, Any] = {}
    for key in ["hit_iou", "depth_common_mae", "other_player_iou"]:
        vals = [float(row[key]) for row in reject_metric_rows if row.get(key) is not None]
        if vals:
            metric_summary[f"rejected_{key}"] = {
                "count": len(vals),
                "mean": float(np.mean(vals)),
                "p50": float(np.percentile(vals, 50)),
                "p90": float(np.percentile(vals, 90)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
    report = {
        "kind": "memory_dense_lingbot_cache_export_report_v0",
        "status": "pass" if readiness["status"] == "pass" and not export_failures else "fail",
        "out_dir": str(args.out_dir),
        "manifest": str(manifest_path),
        "teacher_qa": str(teacher_qa_path),
        "readiness": str(readiness_path),
        "aligned_cache_manifest": str(aligned_path),
        "cache_manifest": str(args.cache_manifest),
        "source_manifest": str(args.source_manifest) if args.source_manifest else None,
        "source_rows_indexed": len(source_index),
        "map_manifest_sha256": map_manifest_sha256,
        "scanned_clips": scanned_clips,
        "start_offset": args.start_offset,
        "limit": args.limit,
        "accepted_clips": accepted_clips,
        "rendered_samples": rendered_samples,
        "accepted_unique_samples": len(samples),
        "role_counts": dict(sorted(role_counts.items())),
        "match_counts": dict(sorted(match_counts.items())),
        "episode_count": len(episode_counts),
        "rejected_clips": dict(sorted(rejected_clips.items())),
        "rejected_samples": dict(sorted(rejected_samples.items())),
        "reject_examples": reject_examples,
        "reject_metric_summary": metric_summary,
        "reject_metric_rows": reject_metric_rows[: args.max_reject_examples],
        "readiness_status": readiness["status"],
        "readiness_failures": readiness.get("failures", []),
        "export_failures": export_failures,
    }
    report_path = args.out_dir / "export_report_v0.json"
    write_json(report_path, report)
    if report["status"] != "pass":
        raise SystemExit(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-manifest", type=Path, default=DEFAULT_CACHE_MANIFEST)
    ap.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    ap.add_argument("--bsp-faces-npz", type=Path, default=DEFAULT_BSP_FACES_NPZ)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start-offset", type=int, default=0)
    ap.add_argument("--max-accepted-clips", type=int, default=None)
    ap.add_argument("--min-accepted-clips", type=int, default=1)
    ap.add_argument("--min-positive-samples", type=int, default=1)
    ap.add_argument("--min-samples", type=int, default=21)
    ap.add_argument("--min-episodes", type=int, default=1)
    ap.add_argument("--readiness-max-samples", type=int, default=4096)

    ap.add_argument("--video-width", type=int, default=832)
    ap.add_argument("--video-height", type=int, default=480)
    ap.add_argument("--video-frames", type=int, default=81)
    ap.add_argument("--raw-stride", type=int, default=2)
    ap.add_argument("--latent-frames", type=int, default=21)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=176)
    ap.add_argument("--fov-x", type=float, default=106.26)
    ap.add_argument("--near", type=float, default=4.0)
    ap.add_argument("--far", type=float, default=3000.0)
    ap.add_argument("--pitch-sign", type=float, default=1.0)
    ap.add_argument("--max-triangles", type=int, default=0)

    ap.add_argument("--player-radius", type=float, default=dense_mod.PLAYER_RADIUS)
    ap.add_argument("--player-height", type=float, default=dense_mod.PLAYER_HEIGHT)
    ap.add_argument("--occlusion-tolerance", type=float, default=dense_mod.PLAYER_OCCLUSION_TOLERANCE)
    ap.add_argument("--min-visible-pixels", type=int, default=48)
    ap.add_argument("--player-z-offset", type=float, default=0.0)
    ap.add_argument("--camera-yaw-offset", type=float, default=0.0)
    ap.add_argument("--camera-pitch-offset", type=float, default=0.0)
    ap.add_argument("--player-mask-mode", choices=["capsule", "multipart"], default=dense_mod.PLAYER_MASK_MODE)
    ap.add_argument("--player-screen-y-offset-px", type=float, default=dense_mod.PLAYER_SCREEN_Y_OFFSET_PX)
    ap.add_argument("--player-radius-scale", type=float, default=dense_mod.PLAYER_RADIUS_SCALE)
    ap.add_argument("--mesh-backend", choices=["bsp_faces_gpu", "bsp_faces_cpu"], default="bsp_faces_gpu")

    ap.add_argument("--min-hit-iou", type=float, default=0.90)
    ap.add_argument("--max-depth-mae", type=float, default=0.08)
    ap.add_argument("--min-teacher-player-pixels", type=int, default=None)
    ap.add_argument("--min-positive-other-iou", type=float, default=0.25)
    ap.add_argument("--min-positive-other-iou-mean", type=float, default=0.45)
    ap.add_argument("--split-key", choices=["episode", "track", "match"], default="episode")
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--test-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260531)
    ap.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--require-source-raw-indices", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--keep-rejected-samples", action="store_true")
    ap.add_argument("--max-reject-examples", type=int, default=50)
    ap.add_argument("--progress-every", type=int, default=100)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.min_teacher_player_pixels is None:
        args.min_teacher_player_pixels = args.min_visible_pixels
    report = export_release(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
