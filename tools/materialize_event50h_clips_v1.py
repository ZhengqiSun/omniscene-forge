#!/usr/bin/env python3
"""Materialize deterministic 81-frame event50h evaluator clips.

The video policy is inherited from build_light_dust2_pilot2_source_v0.py:
OpenCV frame-index seek followed by sequential decode, BGR INTER_LINEAR resize,
and BGR mp4v output at 16 fps and 832x480. The current aligned-row ``video``
field is never used as a processed input; raw pixels come only from ``mp4``.
"""

from __future__ import annotations
from runtime_paths import source_path

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
from typing import Any, Iterable

import av
import cv2
import numpy as np


PROTOCOL_VERSION = "event50h_processed_clip_v1_tier69h_compatible"
PRODUCER = "tools/materialize_event50h_clips_v1.py"
REFERENCE_PRODUCER = (
    str(source_path('scene', 'tools/build_light_dust2_pilot2_source_v0.py'))
)
DEFAULT_TIER_MANIFEST = Path(
    str(source_path('assets', 'prepared/tier69h/tier69h_v2_cache_manifest_v1.jsonl'))
)
WIDTH = 832
HEIGHT = 480
FPS = 16.0
RAW_FPS = 32.0
FRAME_COUNT = 81
RAW_STRIDE = 2
LATENT_POSITIONS = tuple(range(0, 81, 4))
CHECK_POSITIONS = (0, 20, 40, 60, 80)
FOURCC = "mp4v"
SAFE_CLIP_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class ContractError(RuntimeError):
    def __init__(self, reason: str, detail: Any):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "sha256": sha256_file(path)}


def source_stat(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def row_key(row: dict[str, Any]) -> tuple[str, str, str, int, str]:
    try:
        return (
            str(row["hash"]), str(row["game_id"]), str(row["episode"]),
            int(row["raw_start"]), str(row["player_stem"]),
        )
    except Exception as exc:
        raise ContractError("row_identity", repr(exc)) from exc


def load_rows(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    records: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    payloads: dict[tuple[str, str, str, int, str], str] = {}
    sources: dict[tuple[str, str, str, int, str], list[dict[str, Any]]] = {}
    conflicts: set[tuple[str, str, str, int, str]] = set()
    rejects: list[dict[str, Any]] = []
    input_rows = identical = 0
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                input_rows += 1
                try:
                    row = json.loads(line)
                    key = row_key(row)
                except Exception as exc:
                    rejects.append({"reason": "source_row_parse", "source": str(path),
                                    "line": line_number, "detail": repr(exc)})
                    continue
                payload = canonical_json(row)
                source = {"aligned_manifest": str(path), "line": line_number}
                if key not in records:
                    records[key], payloads[key], sources[key] = row, payload, [source]
                elif payloads[key] == payload:
                    identical += 1
                    sources[key].append(source)
                else:
                    conflicts.add(key)
                    rejects.append({"reason": "conflicting_duplicate", "key": list(key),
                                    "kept_sources": sources[key], "conflicting_source": source})
    for key in conflicts:
        records.pop(key, None)
    output = []
    for key in sorted(records):
        row = records[key]
        row_hash = sha256_bytes(payloads[key].encode("utf-8"))
        output.append({"row": row, "source_row_sha256": row_hash, "sources": sources[key]})
    stats = {
        "input_rows": input_rows, "unique_nonconflicting_rows": len(output),
        "identical_duplicate_rows": identical, "conflicting_keys": len(conflicts),
    }
    return output, stats, rejects


def inspect_row(entry: dict[str, Any], clip_dir: Path) -> dict[str, Any]:
    row = entry["row"]
    clip_id = str(row.get("clip_id", ""))
    if not clip_id or not SAFE_CLIP_ID.fullmatch(clip_id):
        raise ContractError("clip_id", clip_id)
    if row.get("dataset_source") not in (None, "event50h"):
        raise ContractError("dataset_source", row.get("dataset_source"))
    if int(row.get("alignment_video_frames", -1)) != FRAME_COUNT:
        raise ContractError("alignment_video_frames", row.get("alignment_video_frames"))
    if int(row.get("alignment_raw_stride", -1)) != RAW_STRIDE:
        raise ContractError("alignment_raw_stride", row.get("alignment_raw_stride"))

    raw_indices_value = row.get("raw_indices")
    if not isinstance(raw_indices_value, list) or len(raw_indices_value) not in (FRAME_COUNT, FRAME_COUNT + 1):
        raise ContractError("raw_indices_length", {"observed": getattr(raw_indices_value, "__len__", lambda: None)(),
                                                    "allowed": [FRAME_COUNT, FRAME_COUNT + 1]})
    raw_indices = [int(value) for value in raw_indices_value]
    selected = raw_indices[:FRAME_COUNT]
    if any(b - a != RAW_STRIDE for a, b in zip(raw_indices, raw_indices[1:])):
        raise ContractError("raw_indices_stride", raw_indices[:5])
    if int(row.get("raw_start", -1)) != selected[0]:
        raise ContractError("raw_start_mismatch", {"raw_start": row.get("raw_start"), "first": selected[0]})

    positions = row.get("alignment_latent_source_positions")
    if positions != list(LATENT_POSITIONS):
        raise ContractError("latent_positions", positions)
    map_indices = row.get("map_memory_raw_frame_indices")
    expected_map = [selected[position] for position in LATENT_POSITIONS]
    if map_indices != expected_map:
        raise ContractError("map_memory_raw_frame_indices", {"expected": expected_map, "observed": map_indices})

    raw_path = Path(str(row.get("mp4", "")))
    if not raw_path.is_file():
        raise ContractError("raw_mp4_missing", str(raw_path))
    poses_path = Path(str(row.get("poses", "")))
    intrinsics_path = Path(str(row.get("intrinsics", "")))
    if not poses_path.is_file() or not intrinsics_path.is_file():
        raise ContractError("camera_sidecar_missing", {"poses": str(poses_path), "intrinsics": str(intrinsics_path)})
    poses = np.load(poses_path, mmap_mode="r", allow_pickle=False)
    intrinsics = np.load(intrinsics_path, mmap_mode="r", allow_pickle=False)
    if poses.shape != (FRAME_COUNT, 4, 4) or poses.dtype != np.float32:
        raise ContractError("poses_contract", {"shape": list(poses.shape), "dtype": str(poses.dtype)})
    if intrinsics.shape != (FRAME_COUNT, 4) or intrinsics.dtype != np.float32:
        raise ContractError("intrinsics_contract", {"shape": list(intrinsics.shape), "dtype": str(intrinsics.dtype)})
    if not np.isfinite(poses).all() or not np.isfinite(intrinsics).all():
        raise ContractError("camera_nonfinite", {"poses": str(poses_path), "intrinsics": str(intrinsics_path)})

    raw_probe = probe_cv2(raw_path, decode=False)
    if not raw_probe["readable"]:
        raise ContractError("raw_mp4_unreadable", str(raw_path))
    if raw_probe["width"] != 1280 or raw_probe["height"] != 720 or abs(raw_probe["fps"] - RAW_FPS) > 1e-6:
        raise ContractError("raw_container_contract", raw_probe)
    if raw_probe["declared_frames"] <= selected[-1]:
        raise ContractError("raw_index_out_of_range", {"last": selected[-1], "frames": raw_probe["declared_frames"]})
    return {
        "clip_id": clip_id, "raw_path": raw_path, "selected_raw_indices": selected,
        "source_raw_indices_count": len(raw_indices), "ignored_trailing_raw_indices": raw_indices[FRAME_COUNT:],
        "poses_path": poses_path, "intrinsics_path": intrinsics_path,
        "clip_dir": clip_dir, "video_path": clip_dir / "video.mp4",
        "artifact_meta_path": clip_dir / "artifact_meta.json", "raw_probe": raw_probe,
    }


def probe_cv2(path: Path, *, decode: bool) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    readable = bool(cap.isOpened())
    result = {"path": str(path), "exists": path.is_file(), "readable": readable}
    if not readable:
        cap.release()
        return result
    result.update(
        declared_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        backend=cap.getBackendName(),
    )
    if decode:
        count = 0
        shapes: set[tuple[int, ...]] = set()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
            shapes.add(tuple(int(x) for x in frame.shape))
        result.update(decoded_frames=count, decoded_shapes=[list(value) for value in sorted(shapes)])
    cap.release()
    return result


def probe_av(path: Path) -> dict[str, Any]:
    with av.open(str(path), mode="r") as container:
        if len(container.streams.video) != 1:
            raise ContractError("video_stream_count", len(container.streams.video))
        stream = container.streams.video[0]
        context = stream.codec_context
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else None
        return {
            "container_format": container.format.name, "codec_name": context.name,
            "codec_tag": str(context.codec_tag),
            "pixel_format": context.format.name if context.format is not None else None,
            "stream_frames": int(stream.frames), "average_rate": float(stream.average_rate),
            "width": int(context.width), "height": int(context.height), "duration_seconds": duration,
        }


def read_raw_frames(path: Path, raw_indices: list[int]) -> list[np.ndarray]:
    if not raw_indices or raw_indices != sorted(set(raw_indices)):
        raise ContractError("read_raw_indices", raw_indices)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ContractError("raw_open", str(path))
    start, end = raw_indices[0], raw_indices[-1]
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
        cap.release()
        raise ContractError("frame_index_seek_failed", {"path": str(path), "frame": start})
    reported = float(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if abs(reported - start) > 0.5:
        cap.release()
        raise ContractError("frame_index_seek_position", {"requested": start, "reported": reported})
    wanted = set(raw_indices)
    frames: dict[int, np.ndarray] = {}
    try:
        for raw_index in range(start, end + 1):
            ok, frame_bgr = cap.read()
            if not ok:
                raise ContractError("raw_decode", {"path": str(path), "raw_index": raw_index})
            if raw_index in wanted:
                if frame_bgr.dtype != np.uint8 or frame_bgr.shape != (720, 1280, 3):
                    raise ContractError("raw_frame_contract", {"raw_index": raw_index,
                                                               "shape": list(frame_bgr.shape),
                                                               "dtype": str(frame_bgr.dtype)})
                frames[raw_index] = cv2.resize(frame_bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    finally:
        cap.release()
    if set(frames) != wanted:
        raise ContractError("raw_frames_missing", sorted(wanted - set(frames)))
    return [frames[index] for index in raw_indices]


def decode_processed_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ContractError("processed_open", str(path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.dtype != np.uint8 or frame.shape != (HEIGHT, WIDTH, 3):
                raise ContractError("processed_frame_contract", {"index": len(frames),
                                                                 "shape": list(frame.shape),
                                                                 "dtype": str(frame.dtype)})
            frames.append(frame)
    finally:
        cap.release()
    if len(frames) != FRAME_COUNT:
        raise ContractError("processed_decoded_frames", {"expected": FRAME_COUNT, "observed": len(frames)})
    return frames


def pixel_error(processed_frames: list[np.ndarray], raw_path: Path,
                selected_raw_indices: list[int]) -> dict[str, Any]:
    raw_check_indices = [selected_raw_indices[position] for position in CHECK_POSITIONS]
    references = read_raw_frames(raw_path, raw_check_indices)
    per_frame = []
    all_errors = []
    for position, raw_index, reference in zip(CHECK_POSITIONS, raw_check_indices, references):
        error = np.abs(processed_frames[position].astype(np.int16) - reference.astype(np.int16)).astype(np.uint8)
        all_errors.append(error.reshape(-1))
        per_frame.append({"clip_position": position, "raw_index": raw_index,
                          "mean": float(error.mean()), "p99": float(np.percentile(error, 99)),
                          "max": int(error.max())})
    combined = np.concatenate(all_errors)
    return {"positions": list(CHECK_POSITIONS), "per_frame": per_frame,
            "aggregate": {"mean": float(combined.mean()), "p99": float(np.percentile(combined, 99)),
                          "max": int(combined.max())}}


def validate_processed_container(path: Path) -> tuple[dict[str, Any], list[np.ndarray]]:
    cv_probe = probe_cv2(path, decode=True)
    av_probe = probe_av(path)
    errors = []
    if cv_probe.get("decoded_frames") != FRAME_COUNT or cv_probe.get("declared_frames") != FRAME_COUNT:
        errors.append("frames")
    if abs(float(cv_probe.get("fps", 0.0)) - FPS) > 1e-6:
        errors.append("fps")
    if (cv_probe.get("width"), cv_probe.get("height")) != (WIDTH, HEIGHT):
        errors.append("resolution")
    if av_probe["stream_frames"] not in (0, FRAME_COUNT):
        errors.append("av_stream_frames")
    if errors:
        raise ContractError("processed_container", {"errors": errors, "cv2": cv_probe, "pyav": av_probe})
    return {"cv2": cv_probe, "pyav": av_probe}, decode_processed_frames(path)


def load_tier_references(path: Path | None, wanted: set[tuple[str, str, str, int, str]]) -> dict[tuple[str, str, str, int, str], dict[str, Any]]:
    if path is None or not path.is_file() or not wanted:
        return {}
    references = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row_key(row)
            if key in wanted and key not in references:
                video = Path(str(row.get("video", "")))
                if video.is_file():
                    references[key] = row
            if len(references) == len(wanted):
                break
    return references


def tier_reference_error(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    raw_indices = [int(value) for value in row.get("raw_indices", [])[:FRAME_COUNT]]
    if len(raw_indices) != FRAME_COUNT:
        raise ContractError("tier_reference_raw_indices", len(raw_indices))
    path = Path(str(row["video"]))
    container, frames = validate_processed_container(path)
    return {"clip_id": row.get("clip_id"), "video": str(path), "container": container,
            "pixel_error": pixel_error(frames, Path(str(row["mp4"])), raw_indices)}


def expected_metadata(entry: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "producer": PRODUCER,
        "producer_sha256": sha256_file(Path(__file__).resolve()),
        "source_row_sha256": entry["source_row_sha256"],
        "clip_id": contract["clip_id"],
        "raw_source": str(contract["raw_path"]),
        "selected_raw_indices": contract["selected_raw_indices"],
        "output_video": str(contract["video_path"]),
        "poses": str(contract["poses_path"]),
        "intrinsics": str(contract["intrinsics_path"]),
    }


def validate_existing(entry: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    meta_path, video_path = contract["artifact_meta_path"], contract["video_path"]
    if not meta_path.is_file() or not video_path.is_file():
        raise ContractError("existing_output_incomplete", {"clip_dir": str(contract["clip_dir"]),
                                                           "meta": meta_path.is_file(), "video": video_path.is_file()})
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = expected_metadata(entry, contract)
    mismatches = {key: {"expected": value, "observed": meta.get(key)}
                  for key, value in expected.items() if meta.get(key) != value}
    if mismatches:
        raise ContractError("existing_metadata_mismatch", mismatches)
    actual_video = file_fingerprint(video_path)
    if meta.get("file_fingerprints", {}).get("video") != actual_video:
        raise ContractError("existing_video_fingerprint", {"metadata": meta.get("file_fingerprints", {}).get("video"),
                                                           "actual": actual_video})
    for name, path in (("poses", contract["poses_path"]), ("intrinsics", contract["intrinsics_path"])):
        actual = file_fingerprint(path)
        if meta.get("file_fingerprints", {}).get(name) != actual:
            raise ContractError(f"existing_{name}_fingerprint", {"metadata": meta.get("file_fingerprints", {}).get(name),
                                                                 "actual": actual})
    container, _ = validate_processed_container(video_path)
    return {"status": "reused", "clip_id": contract["clip_id"], "video": str(video_path),
            "artifact_meta": str(meta_path), "container": container,
            "pixel_error": meta["pixel_validation"],
            "tier69h_reference": meta.get("tier69h_reference"),
            "video_size_bytes": actual_video["size_bytes"],
            "artifact_size_bytes": actual_video["size_bytes"] + meta_path.stat().st_size}


def materialize_one(entry: dict[str, Any], output_dir: Path,
                    tier_reference: dict[str, Any] | None, dry_run: bool) -> dict[str, Any]:
    row = entry["row"]
    clip_id = str(row.get("clip_id", ""))
    final_dir = output_dir / "clips" / clip_id
    contract = inspect_row(entry, final_dir)
    if dry_run:
        return {"status": "planned", "clip_id": clip_id, "raw_source": str(contract["raw_path"]),
                "selected_raw_indices": contract["selected_raw_indices"],
                "source_raw_indices_count": contract["source_raw_indices_count"],
                "ignored_trailing_raw_indices": contract["ignored_trailing_raw_indices"],
                "output_video": str(contract["video_path"]), "raw_container": contract["raw_probe"]}
    if final_dir.exists():
        return validate_existing(entry, contract)

    clips_root = output_dir / "clips"
    clips_root.mkdir(parents=True, exist_ok=True)
    temp_dir = clips_root / f".{clip_id}.tmp.{os.getpid()}.{threading.get_ident()}"
    if temp_dir.exists():
        raise ContractError("temporary_output_exists", str(temp_dir))
    temp_dir.mkdir()
    temp_video = temp_dir / "video.mp4"
    try:
        frames_bgr = read_raw_frames(contract["raw_path"], contract["selected_raw_indices"])
        writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*FOURCC), FPS, (WIDTH, HEIGHT))
        if not writer.isOpened():
            raise ContractError("video_writer_open", str(temp_video))
        writer_backend = writer.getBackendName()
        try:
            for frame_bgr in frames_bgr:
                writer.write(frame_bgr)
        finally:
            writer.release()
        container, decoded_frames = validate_processed_container(temp_video)
        container["cv2"]["path"] = str(contract["video_path"])
        pixel_validation = pixel_error(decoded_frames, contract["raw_path"], contract["selected_raw_indices"])
        reference_validation = tier_reference_error(tier_reference)

        final_video = contract["video_path"]
        meta = {
            **expected_metadata(entry, contract),
            "reference_producer": REFERENCE_PRODUCER,
            "aligned_manifest_sources": entry["sources"],
            "source_raw_indices_count": contract["source_raw_indices_count"],
            "selected_raw_indices_rule": "raw_indices[:alignment_video_frames] after strict stride/map alignment validation",
            "ignored_trailing_raw_indices": contract["ignored_trailing_raw_indices"],
            "alignment_latent_source_positions": list(LATENT_POSITIONS),
            "map_memory_raw_frame_indices": [contract["selected_raw_indices"][p] for p in LATENT_POSITIONS],
            "raw_source_stat": source_stat(contract["raw_path"]),
            "video_policy": {
                "decoder": "cv2.VideoCapture", "decoder_native_color_order": "BGR",
                "seek": "CAP_PROP_POS_FRAMES exact index then sequential decode; no time seek",
                "resize": {"width": WIDTH, "height": HEIGHT, "interpolation": "cv2.INTER_LINEAR"},
                "writer": "cv2.VideoWriter", "writer_backend": writer_backend,
                "writer_input_color_order": "BGR", "rgb_conversion": "none",
                "fourcc_requested": FOURCC, "fps": FPS,
                "quality": {"explicitly_set": False, "value": None, "mode": "OpenCV/FFmpeg backend default"},
                "pixel_format": container["pyav"]["pixel_format"],
                "opencv_version": cv2.__version__, "pyav_version": av.__version__,
            },
            "container": container,
            "pixel_validation": pixel_validation,
            "tier69h_reference": reference_validation,
            "file_fingerprints": {
                "video": {**file_fingerprint(temp_video), "path": str(final_video)},
                "poses": file_fingerprint(contract["poses_path"]),
                "intrinsics": file_fingerprint(contract["intrinsics_path"]),
            },
        }
        write_json(temp_dir / "artifact_meta.json", meta)
        os.replace(temp_dir, final_dir)
        meta_path = final_dir / "artifact_meta.json"
        video_size = final_video.stat().st_size
        return {"status": "materialized", "clip_id": clip_id, "video": str(final_video),
                "artifact_meta": str(meta_path), "container": container,
                "pixel_error": pixel_validation, "tier69h_reference": reference_validation,
                "video_size_bytes": video_size,
                "artifact_size_bytes": video_size + meta_path.stat().st_size}
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise


def corrected_row(entry: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    row = dict(entry["row"])
    raw_source = str(row["mp4"])
    row.update({
        "mp4": raw_source, "video": result["video"],
        "sample_dir": str(Path(result["video"]).parent),
        "clip_dir": str(Path(result["video"]).parent),
        "artifact_meta": result["artifact_meta"],
        "processed_video_protocol": PROTOCOL_VERSION,
        "materialization_source_row_sha256": entry["source_row_sha256"],
        "materialization_aligned_manifest_sources": entry["sources"],
    })
    return row


def summarize_errors(results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregates = [r["pixel_error"]["aggregate"] for r in results if "pixel_error" in r]
    if not aggregates:
        return {}
    return {
        "clips": len(aggregates),
        "mean_error_mean": float(np.mean([v["mean"] for v in aggregates])),
        "mean_error_min": float(np.min([v["mean"] for v in aggregates])),
        "mean_error_max": float(np.max([v["mean"] for v in aggregates])),
        "p99_min": float(np.min([v["p99"] for v in aggregates])),
        "p99_max": float(np.max([v["p99"] for v in aggregates])),
        "max_error": int(np.max([v["max"] for v in aggregates])),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    entries, source_stats, source_rejects = load_rows(args.aligned_manifest)
    all_unique_rows = len(entries)
    selected = entries[:args.max_clips] if args.max_clips is not None else entries
    wanted = {row_key(entry["row"]) for entry in selected}
    tier_refs = load_tier_references(args.tier69h_manifest, wanted)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def task(entry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
        try:
            result = materialize_one(entry, args.output_dir, tier_refs.get(row_key(entry["row"])), args.dry_run)
            return entry, result, None
        except ContractError as exc:
            return entry, None, {"clip_id": entry["row"].get("clip_id"), "reason": exc.reason,
                                 "detail": exc.detail, "sources": entry["sources"]}
        except Exception as exc:
            return entry, None, {"clip_id": entry["row"].get("clip_id"), "reason": "unexpected_error",
                                 "detail": repr(exc), "sources": entry["sources"]}

    if args.workers == 1:
        outcomes = [task(entry) for entry in selected]
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            outcomes = list(pool.map(task, selected))

    successes = []
    corrected = []
    rejects = list(source_rejects)
    for entry, result, failure in outcomes:
        if failure is not None:
            rejects.append(failure)
        else:
            assert result is not None
            successes.append(result)
            if not args.dry_run:
                corrected.append(corrected_row(entry, result))
    successes.sort(key=lambda value: value["clip_id"])
    corrected.sort(key=row_key)
    rejects.sort(key=lambda value: (str(value.get("clip_id", "")), str(value.get("reason", ""))))

    write_jsonl(args.output_dir / "rejected_clips.jsonl", rejects)
    if not args.dry_run:
        write_jsonl(args.output_dir / "corrected_manifest_preview.jsonl", corrected)

    materialized = [result for result in successes if result["status"] in {"materialized", "reused"}]
    video_sizes = [result["video_size_bytes"] for result in materialized]
    artifact_sizes = [result["artifact_size_bytes"] for result in materialized]
    average_video = float(np.mean(video_sizes)) if video_sizes else None
    average_artifact = float(np.mean(artifact_sizes)) if artifact_sizes else None
    estimates = {}
    if average_video is not None and average_artifact is not None:
        for name, count in (("current_pair_raw_endpoints", 21640),
                            ("all_unique_event50h_aligned_rows", all_unique_rows)):
            estimates[name] = {
                "count": count,
                "video_bytes": int(round(average_video * count)),
                "video_gib": average_video * count / (1024 ** 3),
                "video_plus_metadata_bytes": int(round(average_artifact * count)),
                "video_plus_metadata_gib": average_artifact * count / (1024 ** 3),
            }
    report = {
        "protocol_version": PROTOCOL_VERSION, "producer": PRODUCER,
        "reference_producer": REFERENCE_PRODUCER,
        "dry_run": args.dry_run, "workers": args.workers,
        "aligned_manifests": [str(path) for path in args.aligned_manifest],
        "tier69h_manifest": str(args.tier69h_manifest) if args.tier69h_manifest else None,
        "source": source_stats, "all_unique_event50h_aligned_rows": all_unique_rows,
        "selected_clips": len(selected), "successful_clips": len(successes),
        "materialized_clips": sum(result["status"] == "materialized" for result in successes),
        "reused_clips": sum(result["status"] == "reused" for result in successes),
        "planned_clips": sum(result["status"] == "planned" for result in successes),
        "failed_clips": len(rejects), "failure_reasons": dict(Counter(r["reason"] for r in rejects)),
        "tier69h_reference_matches": len(tier_refs),
        "video_policy": {"frames": FRAME_COUNT, "fps": FPS, "resolution": [WIDTH, HEIGHT],
                         "fourcc": FOURCC, "resize_interpolation": "cv2.INTER_LINEAR",
                         "decoder_color_order": "BGR", "writer_color_order": "BGR",
                         "quality": "OpenCV/FFmpeg backend default; not explicitly set"},
        "pixel_error_summary": summarize_errors(materialized),
        "average_video_size_bytes": average_video,
        "average_video_plus_metadata_size_bytes": average_artifact,
        "disk_estimates": estimates,
        "results": successes,
        "outputs": {
            "report": str(args.output_dir / "materialization_report.json"),
            "rejected": str(args.output_dir / "rejected_clips.jsonl"),
            "corrected_preview": None if args.dry_run else str(args.output_dir / "corrected_manifest_preview.jsonl"),
        },
    }
    write_json(args.output_dir / "materialization_report.json", report)
    print(json.dumps({key: report[key] for key in (
        "dry_run", "source", "selected_clips", "successful_clips", "materialized_clips",
        "reused_clips", "planned_clips", "failed_clips", "failure_reasons",
        "tier69h_reference_matches", "pixel_error_summary", "average_video_size_bytes",
        "average_video_plus_metadata_size_bytes", "disk_estimates", "outputs",
    )}, ensure_ascii=False, indent=2, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-manifest", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tier69h-manifest", type=Path, default=DEFAULT_TIER_MANIFEST)
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_clips is not None and args.max_clips < 1:
        parser.error("--max-clips must be >= 1")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    return args


if __name__ == "__main__":
    run(parse_args())
