#!/usr/bin/env python3
"""Export a light-dust2 Map Memory dense release without teacher seg/depth.

This is intentionally scoped to the light dust2 pilot path.  The light dataset
has RGB, player JSON, and visibility, but no teacher seg/depth streams.  It
therefore writes an explicit Memory-mask surrogate contract: region metrics may
use dense channel 3, and the release must not be treated as teacher QA.
"""

from __future__ import annotations
from runtime_paths import source_path

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from PIL import Image

from build_memory_dense_aligned_cache_v0 import (
    expected_sample_ids,
    latent_frame_source_positions,
    parse_clip_identity,
    validate_record_shapes,
)
from map_memory_training_data_v0 import CHANNELS, REQUIRED_BACKEND_ID, sha256_file, validate_dense_array


DEFAULT_RAW_ROOT = Path(str(source_path('assets', 'csgo-datasets')))
DEFAULT_BSP_FACES_NPZ = Path("docs/assets/bsp_static_geometry_v0/de_dust2_bsp_faces_v0.npz")
REGION_MASK_KIND = "memory_dense_channel_3_surrogate_v0"


def import_tool(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def iter_jsonl(path: Path, limit: int | None = None):
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)
            count += 1
            if limit is not None and count >= limit:
                break


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_source_index(path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(iter_jsonl(path)):
        raw_start = int(row["raw_start"])
        keys = {
            str(row["clip_id"]),
            Path(str(row.get("clip_dir", ""))).name,
            f"{idx:04d}_{row['game_id']}_{row['episode']}_{row['player_stem']}",
            f"{idx:04d}_{row['game_id']}_{row['episode']}_{row['player_stem']}_{raw_start:07d}",
        }
        for key in keys:
            if key:
                index[key] = row
    return index


def match_dir_for_row(row: dict[str, Any], raw_root: Path) -> Path:
    match_dir = raw_root / str(row.get("hash", "32f1644d4f42c29d")) / str(row["game_id"])
    if not match_dir.exists():
        raise FileNotFoundError(f"missing light match dir: {match_dir}")
    return match_dir


def source_raw_frame_errors(source: dict[str, Any], raw_frames: list[int], *, video_frames: int, raw_stride: int) -> list[str]:
    raw_indices = source.get("raw_indices")
    if not isinstance(raw_indices, list):
        return ["source row has no raw_indices list"]
    if len(raw_indices) != video_frames:
        return [f"raw_indices length {len(raw_indices)} != {video_frames}"]
    positions = latent_frame_source_positions(video_frames, len(raw_frames))
    observed = [int(raw_indices[pos]) for pos in positions]
    if observed != raw_frames:
        return [f"latent raw frames {observed[:5]}... != expected {raw_frames[:5]}..."]
    if len(raw_indices) > 1:
        strides = sorted({int(raw_indices[i + 1]) - int(raw_indices[i]) for i in range(len(raw_indices) - 1)})
        if strides != [raw_stride]:
            return [f"raw index strides {strides} != [{raw_stride}]"]
    return []


def render_or_load_sample(
    *,
    sample_id: str,
    source: dict[str, Any],
    frame_index: int,
    out_dir: Path,
    cache: dict[str, Any],
    dense_mod: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    match_dir = match_dir_for_row(source, args.raw_root)
    sample_dir = out_dir / "samples" / sample_id
    dense_path = sample_dir / "mesh_dense_condition_v0.npz"
    target_path = sample_dir / "target_rgb.png"
    meta_path = sample_dir / "mesh_dense_condition_meta_v0.json"
    qa_path = sample_dir / "mesh_dense_condition_qa_v0.png"

    reuse_ok = False
    dense: np.ndarray | None = None
    meta: dict[str, Any] | None = None
    rgb: np.ndarray | None = None
    if args.reuse_existing and dense_path.exists() and target_path.exists() and meta_path.exists() and qa_path.exists():
        try:
            dense = np.load(dense_path)["dense"].astype(np.float32)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rgb = np.asarray(Image.open(target_path).convert("RGB"))
            reuse_ok = (
                list(dense.shape) == [7, args.height, args.width]
                and meta.get("player_mask_mode") == args.player_mask_mode
                and str(meta.get("mesh_backend")) == args.mesh_backend
                and abs(float(meta.get("fov_x", -1.0)) - float(args.fov_x)) < 1e-6
                and abs(float(meta.get("player_screen_y_offset_px", 0.0)) - float(args.player_screen_y_offset_px)) < 1e-6
                and abs(float(meta.get("player_radius_scale", 0.0)) - float(args.player_radius_scale)) < 1e-6
            )
        except Exception:
            reuse_ok = False

    if not reuse_ok:
        shutil.rmtree(sample_dir, ignore_errors=True)
        sample_dir.mkdir(parents=True, exist_ok=True)
        dense, mesh_depth_units, rgb, meta, render_dense_mod = dense_mod.render_memory_dense_condition(
            match_dir,
            str(source["episode"]),
            str(source["player_stem"]),
            int(frame_index),
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
        errors = validate_dense_array(dense)
        if errors:
            raise ValueError(f"{sample_id}: invalid dense: {errors}")
        np.savez_compressed(dense_path, dense=dense, mesh_depth_units=mesh_depth_units)
        Image.fromarray(rgb).resize((args.width, args.height), Image.Resampling.BILINEAR).save(target_path)
        meta["target_rgb_path"] = str(target_path)
        meta["target_policy"] = "Light pilot RGB target only; no teacher seg/depth streams are available."
        meta["region_mask_kind"] = REGION_MASK_KIND
        write_json(meta_path, meta)
        render_dense_mod.make_qa(qa_path, rgb, dense[0], dense[1], dense[3:], meta)
    assert dense is not None and meta is not None

    memory_pixels = int((dense[3] > 0.5).sum())
    role = "positive" if memory_pixels > 0 else "context"
    sample = {
        "sample_id": sample_id,
        "match_id": str(source["game_id"]),
        "episode": f"{source['game_id']}_{source['episode']}",
        "raw_episode": str(source["episode"]),
        "ego_stem": str(source["player_stem"]),
        "frame_index": int(frame_index),
        "dense_path": str(dense_path),
        "dense_relpath": str(dense_path.relative_to(out_dir)),
        "target_rgb_path": str(target_path),
        "target_rgb_relpath": str(target_path.relative_to(out_dir)),
        "meta_path": str(meta_path),
        "meta_relpath": str(meta_path.relative_to(out_dir)),
        "qa_path": str(qa_path),
        "qa_relpath": str(qa_path.relative_to(out_dir)),
        "shape": list(dense.shape),
        "mesh_hit_ratio": float(meta.get("mesh_hit_ratio", 0.0)),
        "nav_semantic_hit_ratio": meta.get("nav_semantic_hit_ratio"),
        "memory_projected_players": int(len(meta.get("memory_projected_players", []))),
        "channels": meta.get("channels", CHANNELS),
        "mesh_backend": meta.get("mesh_backend"),
        "geometry_backend": "bsp_faces",
        "geometry_backend_id": meta.get("geometry_backend_id"),
        "geometry_backend_family": meta.get("geometry_backend_family"),
        "backend_features": meta.get("backend_features"),
        "backend_signature": meta.get("backend_signature"),
        "component_counts": meta.get("component_counts"),
        "selection_role": role,
        "region_mask_kind": REGION_MASK_KIND,
        "teacher_qa_at_selection": {
            "available": False,
            "reason": "light dust2 pilot has no teacher seg/depth; region mask is Memory dense channel 3 surrogate",
            "other_player_memory_pixels": memory_pixels,
        },
    }
    qa_row = {
        "sample_id": sample_id,
        "episode": sample["episode"],
        "raw_episode": sample["raw_episode"],
        "match_id": sample["match_id"],
        "match_dir": str(match_dir),
        "ego_stem": sample["ego_stem"],
        "frame_index": int(frame_index),
        "visible_teacher_players": 0,
        "other_player_teacher_pixels": 0,
        "other_player_memory_pixels": memory_pixels,
        "other_player_iou": None,
        "hit_iou": None,
        "depth_common": {"mae": None, "median": None, "p95": None},
        "region_mask_kind": REGION_MASK_KIND,
        "teacher_qa_available": False,
        "teacher_qa_unavailable_reason": "light dust2 pilot has no teacher seg/depth",
    }
    return sample, qa_row


def summarize_samples(samples: list[dict[str, Any]], qa_rows: list[dict[str, Any]]) -> dict[str, Any]:
    memory_pixels = np.asarray([int(row.get("other_player_memory_pixels", 0) or 0) for row in qa_rows], dtype=np.float64)
    mesh_hit = np.asarray([float(sample.get("mesh_hit_ratio", 0.0) or 0.0) for sample in samples], dtype=np.float64)
    nav_hit = np.asarray([float(sample.get("nav_semantic_hit_ratio", 0.0) or 0.0) for sample in samples], dtype=np.float64)
    return {
        "teacher_qa_available": False,
        "region_mask_kind": REGION_MASK_KIND,
        "memory_player_pixels_mean": float(memory_pixels.mean()) if len(memory_pixels) else None,
        "memory_player_pixels_p50": float(np.percentile(memory_pixels, 50)) if len(memory_pixels) else None,
        "memory_player_pixels_min": int(memory_pixels.min()) if len(memory_pixels) else None,
        "memory_player_pixels_max": int(memory_pixels.max()) if len(memory_pixels) else None,
        "mesh_hit_ratio_mean": float(mesh_hit.mean()) if len(mesh_hit) else None,
        "mesh_hit_ratio_p50": float(np.percentile(mesh_hit, 50)) if len(mesh_hit) else None,
        "mesh_hit_ratio_min": float(mesh_hit.min()) if len(mesh_hit) else None,
        "nav_semantic_hit_ratio_mean": float(nav_hit.mean()) if len(nav_hit) else None,
    }


def export(args: argparse.Namespace) -> dict[str, Any]:
    tools_dir = Path(__file__).resolve().parent
    dense_mod = import_tool(tools_dir / "build_mesh_dense_condition_v0.py", "mesh_dense_condition_v0")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_index = load_source_index(args.source_manifest)
    renderer_caches: dict[str, dict[str, Any]] = {}
    samples_by_id: dict[str, dict[str, Any]] = {}
    qa_by_id: dict[str, dict[str, Any]] = {}
    aligned_rows: list[dict[str, Any]] = []
    reject_counts: Counter[str] = Counter()
    reject_examples: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    match_counts: Counter[str] = Counter()
    episode_counts: Counter[str] = Counter()

    positions = latent_frame_source_positions(args.video_frames, args.latent_frames)
    scanned = 0
    accepted = 0
    for record_index, record in enumerate(iter_jsonl(args.cache_manifest, limit=args.limit)):
        if args.num_shards > 1 and record_index % args.num_shards != args.shard_index:
            continue
        scanned += 1
        clip_id = str(record.get("clip_id", ""))
        source = source_index.get(clip_id)
        if source is None:
            reject_counts["missing_source_row"] += 1
            if len(reject_examples) < args.max_reject_examples:
                reject_examples.append({"clip_id": clip_id, "reason": "missing_source_row"})
            continue
        try:
            shape_errors = validate_record_shapes(
                record,
                latent_frames=args.latent_frames,
                video_height=args.video_height,
                video_width=args.video_width,
            )
            if shape_errors:
                raise ValueError("; ".join(shape_errors))
            identity = parse_clip_identity(record, source)
            raw_frames, sample_ids = expected_sample_ids(
                identity,
                video_frames=args.video_frames,
                raw_stride=args.raw_stride,
                latent_frames=args.latent_frames,
            )
            raw_errors = source_raw_frame_errors(source, raw_frames, video_frames=args.video_frames, raw_stride=args.raw_stride)
            if raw_errors:
                raise ValueError("; ".join(raw_errors))
            match_dir = match_dir_for_row(source, args.raw_root)
            if match_dir.name not in renderer_caches:
                renderer_caches[match_dir.name] = dense_mod.load_renderer_cache(
                    match_dir,
                    tools_dir,
                    bsp_faces_npz=args.bsp_faces_npz,
                )
            clip_samples: list[dict[str, Any]] = []
            clip_qa: list[dict[str, Any]] = []
            for sample_id, frame_index in zip(sample_ids, raw_frames):
                if sample_id not in samples_by_id:
                    sample, qa_row = render_or_load_sample(
                        sample_id=sample_id,
                        source=source,
                        frame_index=int(frame_index),
                        out_dir=args.out_dir,
                        cache=renderer_caches[match_dir.name],
                        dense_mod=dense_mod,
                        args=args,
                    )
                    samples_by_id[sample_id] = sample
                    qa_by_id[sample_id] = qa_row
                clip_samples.append(samples_by_id[sample_id])
                clip_qa.append(qa_by_id[sample_id])
            if len(clip_samples) != args.latent_frames:
                raise ValueError(f"incomplete clip samples: {len(clip_samples)}")
            split = str(source.get("map_memory_split") or record.get("map_memory_split") or "train")
            if split not in {"train", "val", "test"}:
                raise ValueError(f"invalid source split {split!r}")
            roles = [sample["selection_role"] for sample in clip_samples]
        except Exception as exc:
            reject_counts["bad_or_failed_clip"] += 1
            if len(reject_examples) < args.max_reject_examples:
                reject_examples.append({"clip_id": clip_id, "reason": "bad_or_failed_clip", "detail": str(exc)})
            continue

        out_record = dict(record)
        for key, value in source.items():
            if key != "raw_indices":
                out_record.setdefault(key, value)
        out_record.update({
            "alignment_kind": "map_memory_dense_lingbot_latent_frame_exact_v0",
            "map_memory_manifest": str(args.out_dir / "manifest.json"),
            "map_memory_sample_ids": sample_ids,
            "map_memory_raw_frame_indices": raw_frames,
            "map_memory_split": split,
            "map_memory_split_key": "match",
            "map_memory_match_id": identity["game_id"],
            "map_memory_episode": f"{identity['game_id']}_{identity['episode']}",
            "map_memory_raw_episode": identity["episode"],
            "map_memory_ego_stem": identity["player_stem"],
            "map_memory_track_id": f"{identity['game_id']}|{identity['episode']}|{identity['player_stem']}",
            "map_memory_selection_roles": roles,
            "map_memory_positive_frames": sum(1 for role in roles if role == "positive"),
            "map_memory_context_frames": sum(1 for role in roles if role == "context"),
            "alignment_video_frames": args.video_frames,
            "alignment_raw_stride": args.raw_stride,
            "alignment_latent_frames": args.latent_frames,
            "alignment_latent_source_positions": positions,
            "alignment_exact_frame_match": True,
            "region_mask_kind": REGION_MASK_KIND,
        })
        aligned_rows.append(out_record)
        accepted += 1
        split_counts[split] += 1
        role_counts.update(roles)
        match_counts[str(identity["game_id"])] += 1
        episode_counts[f"{identity['game_id']}_{identity['episode']}"] += 1
        if args.progress_every > 0 and accepted % args.progress_every == 0:
            print(json.dumps({"event": "light_export_progress", "accepted_clips": accepted, "samples": len(samples_by_id)}, ensure_ascii=False), flush=True)

    samples = sorted(samples_by_id.values(), key=lambda row: row["sample_id"])
    qa_rows = [qa_by_id[sample["sample_id"]] for sample in samples]
    sample_role_counts = Counter(sample["selection_role"] for sample in samples)
    sample_match_counts = Counter(sample["match_id"] for sample in samples)
    sample_episode_counts = Counter(sample["episode"] for sample in samples)
    backend_signatures = sorted({sample["backend_signature"] for sample in samples if sample.get("backend_signature")})
    summary = summarize_samples(samples, qa_rows)

    manifest = {
        "kind": "memory_dense_lingbot_light_dust2_pilot_release_v0",
        "source_cache_manifest": str(args.cache_manifest),
        "source_manifest": str(args.source_manifest),
        "raw_root": str(args.raw_root),
        "sample_count": len(samples),
        "positive_sample_count": int(sample_role_counts.get("positive", 0)),
        "context_sample_count": int(sample_role_counts.get("context", 0)),
        "match_count": len(sample_match_counts),
        "episode_count": len(sample_episode_counts),
        "match_sample_counts": dict(sorted(sample_match_counts.items())),
        "episode_sample_counts": dict(sorted(sample_episode_counts.items())),
        "shape": [7, args.height, args.width],
        "channels": CHANNELS,
        "policy": "Light dust2 pilot dense tensors are generated from Map Memory JSON/BSP only; no teacher seg/depth QA is available.",
        "selection_policy": "Rows come from player_visibility-positive source clips; per-frame role is positive iff Memory dense channel 3 has player pixels.",
        "region_mask_kind": REGION_MASK_KIND,
        "region_mask_policy": {
            "kind": REGION_MASK_KIND,
            "source": "dense_channel_3_other_player_mask_from_memory_player_capsules",
            "valid_for": "relative true-vs-shuffled Memory-mask region loss only",
            "not_valid_for": "teacher segmentation precision/recall, absolute presence, or position error",
        },
        "teacher_qa_available": False,
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
            "player_screen_y_offset_px": args.player_screen_y_offset_px,
            "player_radius_scale": args.player_radius_scale,
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
        "kind": "memory_dense_light_surrogate_channels_v0",
        "manifest": str(manifest_path),
        "sample_count": len(qa_rows),
        "missing_count": 0,
        "summary": summary,
        "rows": qa_rows,
        "missing": [],
        "region_mask_kind": REGION_MASK_KIND,
        "teacher_qa_available": False,
        "policy": {
            "region_mask_kind": REGION_MASK_KIND,
            "note": "This is not teacher segmentation QA. It exists so existing release loaders can carry Memory-mask region metadata.",
        },
    }
    teacher_qa_path = args.out_dir / "channel_teacher_qa_v0" / "memory_dense_channels_vs_teacher_v0.json"
    write_json(teacher_qa_path, teacher_qa)

    readiness = {
        "kind": "map_memory_training_readiness_v0",
        "status": "pass",
        "manifest": str(manifest_path),
        "teacher_qa": str(teacher_qa_path),
        "mode": "light_memory_mask_surrogate",
        "sample_count": len(samples),
        "failures": [],
        "region_mask_kind": REGION_MASK_KIND,
        "teacher_qa_available": False,
        "caveats": [
            "No teacher seg/depth streams are available for light dust2.",
            "Region metrics must use Memory dense channel 3 surrogate and remain relative true-vs-shuffled gates.",
        ],
    }
    readiness_path = args.out_dir / "training_readiness_v0.json"
    write_json(readiness_path, readiness)

    aligned_path = args.out_dir / "aligned_cache_manifest.jsonl"
    with aligned_path.open("w", encoding="utf-8") as f:
        for row in aligned_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    export_failures: list[str] = []
    if accepted < args.min_accepted_clips:
        export_failures.append(f"accepted_clips {accepted} < min_accepted_clips {args.min_accepted_clips}")
    if int(sample_role_counts.get("positive", 0)) < args.min_positive_samples:
        export_failures.append(f"positive_sample_count {int(sample_role_counts.get('positive', 0))} < {args.min_positive_samples}")

    report = {
        "kind": "memory_dense_lingbot_light_dust2_pilot_export_report_v0",
        "status": "pass" if not export_failures else "fail",
        "out_dir": str(args.out_dir),
        "manifest": str(manifest_path),
        "teacher_qa": str(teacher_qa_path),
        "readiness": str(readiness_path),
        "aligned_cache_manifest": str(aligned_path),
        "cache_manifest": str(args.cache_manifest),
        "source_manifest": str(args.source_manifest),
        "map_manifest_sha256": map_manifest_sha256,
        "scanned_clips": scanned,
        "accepted_clips": accepted,
        "accepted_unique_samples": len(samples),
        "split_counts": dict(sorted(split_counts.items())),
        "aligned_latent_frame_role_counts": dict(sorted(role_counts.items())),
        "sample_role_counts": dict(sorted(sample_role_counts.items())),
        "match_count": len(sample_match_counts),
        "episode_count": len(sample_episode_counts),
        "summary": summary,
        "region_mask_kind": REGION_MASK_KIND,
        "teacher_qa_available": False,
        "reject_counts": dict(sorted(reject_counts.items())),
        "reject_examples": reject_examples,
        "export_failures": export_failures,
    }
    report_path = args.out_dir / "export_report_v0.json"
    write_json(report_path, report)
    if report["status"] != "pass":
        raise SystemExit(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-manifest", type=Path, required=True)
    ap.add_argument("--source-manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    ap.add_argument("--bsp-faces-npz", type=Path, default=DEFAULT_BSP_FACES_NPZ)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--min-accepted-clips", type=int, default=1)
    ap.add_argument("--min-positive-samples", type=int, default=1)
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
    ap.add_argument("--player-radius", type=float, default=14.0)
    ap.add_argument("--player-height", type=float, default=40.0)
    ap.add_argument("--occlusion-tolerance", type=float, default=40.0)
    ap.add_argument("--min-visible-pixels", type=int, default=48)
    ap.add_argument("--player-z-offset", type=float, default=0.0)
    ap.add_argument("--camera-yaw-offset", type=float, default=0.0)
    ap.add_argument("--camera-pitch-offset", type=float, default=0.0)
    ap.add_argument("--player-mask-mode", choices=["capsule", "multipart"], default="capsule")
    ap.add_argument("--player-screen-y-offset-px", type=float, default=-5.5)
    ap.add_argument("--player-radius-scale", type=float, default=0.70)
    ap.add_argument("--mesh-backend", choices=["bsp_faces_gpu", "bsp_faces_cpu"], default="bsp_faces_gpu")
    ap.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-reject-examples", type=int, default=50)
    ap.add_argument("--progress-every", type=int, default=50)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must be in [0, num_shards)")
    report = export(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
