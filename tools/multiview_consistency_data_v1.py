#!/usr/bin/env python3
"""Read-only loader and RGB/camera preflight for consistency pair manifests.

OpenCV is used consistently with the existing repository readers: decoded
frames arrive as BGR and selected frames are converted at the decoder boundary
with cv2.COLOR_BGR2RGB. Raw frame indices never index clip videos.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
from PIL import Image, ImageDraw


PROTOCOL_VERSION = "multiview_consistency_mvp_v1_exacttick_matchsplit"
PREFLIGHT_VERSION = "multiview_consistency_data_preflight_v1"
LABELS = ("positive", "time_shift", "wrong_window", "cross_episode", "cross_match")
EXPECTED_VIDEO_FRAMES = 81
EXPECTED_SELECTED_FRAMES = 21
CONTACT_TIME_POSITIONS = (0, 5, 10, 15, 20)


class DataContractError(RuntimeError):
    def __init__(self, reason: str, detail: Any):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def validate_positions(endpoint: dict[str, Any]) -> list[int]:
    positions = endpoint.get("alignment_latent_source_positions")
    if not isinstance(positions, list) or len(positions) != EXPECTED_SELECTED_FRAMES:
        raise DataContractError("latent_positions_shape", {"value": positions})
    try:
        values = [int(value) for value in positions]
    except Exception as exc:
        raise DataContractError("latent_positions_dtype", str(exc)) from exc
    if values != sorted(set(values)):
        raise DataContractError("latent_positions_not_strictly_increasing", values)
    if values[0] < 0 or values[-1] >= EXPECTED_VIDEO_FRAMES:
        raise DataContractError("latent_positions_out_of_bounds", values)
    return values


def decode_clip_rgb(video_path: Path, positions: list[int]) -> tuple[np.ndarray, dict[str, Any]]:
    if not video_path.is_file():
        raise DataContractError("video_missing", str(video_path))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise DataContractError("video_open_failed", str(video_path))
    declared_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    declared_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    declared_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    declared_fps = float(cap.get(cv2.CAP_PROP_FPS))
    selected: dict[int, np.ndarray] = {}
    decoded_count = 0
    decoded_shape: tuple[int, int, int] | None = None
    wanted = set(positions)
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_bgr is None or frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
                raise DataContractError(
                    "decoded_frame_contract", {"frame": decoded_count, "shape": getattr(frame_bgr, "shape", None),
                                                "dtype": str(getattr(frame_bgr, "dtype", None))},
                )
            shape = tuple(int(value) for value in frame_bgr.shape)
            if decoded_shape is None:
                decoded_shape = shape
            elif shape != decoded_shape:
                raise DataContractError("decoded_frame_shape_changed", {"first": decoded_shape, "current": shape})
            if decoded_count in wanted:
                # Repository convention: VideoCapture produces BGR; convert here to RGB.
                selected[decoded_count] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            decoded_count += 1
    finally:
        cap.release()
    if decoded_count != EXPECTED_VIDEO_FRAMES:
        raise DataContractError(
            "decoded_frame_count", {"path": str(video_path), "decoded": decoded_count,
                                    "expected": EXPECTED_VIDEO_FRAMES, "container_declared": declared_frames},
        )
    missing = [position for position in positions if position not in selected]
    if missing:
        raise DataContractError("selected_video_frames_missing", missing)
    rgb_thwc = np.stack([selected[position] for position in positions], axis=0)
    rgb_tchw = np.ascontiguousarray(rgb_thwc.transpose(0, 3, 1, 2))
    if rgb_tchw.dtype != np.uint8 or rgb_tchw.shape[0:2] != (21, 3):
        raise DataContractError("selected_rgb_contract", {"shape": rgb_tchw.shape, "dtype": str(rgb_tchw.dtype)})
    metadata = {
        "path": str(video_path), "exists": True,
        "decoder": "cv2.VideoCapture", "decoder_native_color_order": "BGR",
        "rgb_conversion": "cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) at selected-frame decode boundary",
        "decoded_frames": decoded_count, "container_declared_frames": declared_frames,
        "container_width": declared_width, "container_height": declared_height, "container_fps": declared_fps,
        "decoded_hwc_shape": list(decoded_shape or ()), "selected_rgb_shape": list(rgb_tchw.shape),
        "selected_rgb_dtype": str(rgb_tchw.dtype),
    }
    return rgb_tchw, metadata


def load_camera_arrays(endpoint: dict[str, Any], positions: list[int]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    poses_path = Path(str(endpoint.get("poses", "")))
    intrinsics_path = Path(str(endpoint.get("intrinsics", "")))
    if not poses_path.is_file():
        raise DataContractError("poses_missing", str(poses_path))
    if not intrinsics_path.is_file():
        raise DataContractError("intrinsics_missing", str(intrinsics_path))
    try:
        poses_source = np.load(poses_path, mmap_mode="r", allow_pickle=False)
        intrinsics_source = np.load(intrinsics_path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise DataContractError("camera_array_load", str(exc)) from exc
    if poses_source.shape != (EXPECTED_VIDEO_FRAMES, 4, 4):
        raise DataContractError("poses_shape", {"path": str(poses_path), "shape": list(poses_source.shape)})
    if intrinsics_source.shape != (EXPECTED_VIDEO_FRAMES, 4):
        raise DataContractError("intrinsics_shape", {"path": str(intrinsics_path), "shape": list(intrinsics_source.shape)})
    if not np.issubdtype(poses_source.dtype, np.number):
        raise DataContractError("poses_dtype", str(poses_source.dtype))
    if not np.issubdtype(intrinsics_source.dtype, np.number):
        raise DataContractError("intrinsics_dtype", str(intrinsics_source.dtype))
    poses = np.asarray(poses_source[positions], dtype=np.float32).copy()
    intrinsics = np.asarray(intrinsics_source[positions], dtype=np.float32).copy()
    pose_nan, pose_inf = int(np.isnan(poses).sum()), int(np.isinf(poses).sum())
    intr_nan, intr_inf = int(np.isnan(intrinsics).sum()), int(np.isinf(intrinsics).sum())
    if pose_nan or pose_inf:
        raise DataContractError("poses_nonfinite", {"nan": pose_nan, "inf": pose_inf})
    if intr_nan or intr_inf:
        raise DataContractError("intrinsics_nonfinite", {"nan": intr_nan, "inf": intr_inf})
    metadata = {
        "poses_path": str(poses_path), "poses_source_shape": list(poses_source.shape),
        "poses_source_dtype": str(poses_source.dtype), "poses_selected_shape": list(poses.shape),
        "poses_output_dtype": str(poses.dtype), "poses_nan": pose_nan, "poses_inf": pose_inf,
        "intrinsics_path": str(intrinsics_path), "intrinsics_source_shape": list(intrinsics_source.shape),
        "intrinsics_source_dtype": str(intrinsics_source.dtype),
        "intrinsics_selected_shape": list(intrinsics.shape),
        "intrinsics_output_dtype": str(intrinsics.dtype), "intrinsics_nan": intr_nan, "intrinsics_inf": intr_inf,
    }
    return poses, intrinsics, metadata


def endpoint_ticks(endpoint: dict[str, Any], field: str, dtype: np.dtype[Any]) -> np.ndarray:
    values = endpoint.get(field)
    if not isinstance(values, list) or len(values) != EXPECTED_SELECTED_FRAMES:
        raise DataContractError(f"{field}_shape", values)
    try:
        return np.asarray([int(value) for value in values], dtype=dtype)
    except Exception as exc:
        raise DataContractError(f"{field}_dtype", str(exc)) from exc


def load_endpoint(endpoint: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if int(endpoint.get("alignment_video_frames", -1)) != EXPECTED_VIDEO_FRAMES:
        raise DataContractError("alignment_video_frames", endpoint.get("alignment_video_frames"))
    positions = validate_positions(endpoint)
    rgb, video_meta = decode_clip_rgb(Path(str(endpoint.get("video", ""))), positions)
    pose, intrinsics, camera_meta = load_camera_arrays(endpoint, positions)
    ticks = endpoint_ticks(endpoint, "world_ticks", np.dtype(np.int64))
    frame_counts = endpoint_ticks(endpoint, "frame_counts", np.dtype(np.int64))
    data = {"rgb": rgb, "pose": pose, "intrinsics": intrinsics,
            "ticks": ticks, "frame_counts": frame_counts, "metadata": endpoint}
    return data, {"positions": positions, "video": video_meta, "camera": camera_meta}


def load_pair_record(record: dict[str, Any]) -> dict[str, Any]:
    required = ("sample_id", "label", "split", "left", "right")
    missing = [field for field in required if field not in record]
    if missing:
        raise DataContractError("pair_required_fields", missing)
    if record.get("protocol_version") != PROTOCOL_VERSION:
        raise DataContractError("protocol_version", record.get("protocol_version"))
    if record["label"] not in LABELS:
        raise DataContractError("label", record["label"])
    left, left_meta = load_endpoint(record["left"])
    right, right_meta = load_endpoint(record["right"])
    if left["rgb"].shape != right["rgb"].shape:
        raise DataContractError("pair_rgb_shape_mismatch", {"left": left["rgb"].shape, "right": right["rgb"].shape})
    pair_metadata = {key: value for key, value in record.items() if key not in {"left", "right"}}
    result = {
        "sample_id": str(record["sample_id"]), "label": str(record["label"]),
        "is_consistent": bool(record.get("is_consistent", record["label"] == "positive")),
        "difficulty": record.get("negative_difficulty"), "split": str(record["split"]),
        "left_rgb": left["rgb"], "right_rgb": right["rgb"],
        "left_pose": left["pose"], "right_pose": right["pose"],
        "left_intrinsics": left["intrinsics"], "right_intrinsics": right["intrinsics"],
        "left_ticks": left["ticks"], "right_ticks": right["ticks"],
        "left_frame_counts": left["frame_counts"], "right_frame_counts": right["frame_counts"],
        "left_endpoint": record["left"], "right_endpoint": record["right"],
        "pair_metadata": pair_metadata,
        "load_metadata": {"left": left_meta, "right": right_meta},
    }
    return result


class PairManifestDataset:
    """Random-access JSONL dataset; only byte offsets are indexed in memory."""

    def __init__(self, pair_manifest: str | Path):
        self.path = Path(pair_manifest)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._offsets: list[int] = []
        self.label_counts: Counter[str] = Counter()
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                record = json.loads(line)
                self._offsets.append(offset)
                self.label_counts[str(record.get("label"))] += 1

    def __len__(self) -> int:
        return len(self._offsets)

    def read_record(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        with self.path.open("rb") as handle:
            handle.seek(self._offsets[index])
            return json.loads(handle.readline())

    def __getitem__(self, index: int) -> dict[str, Any]:
        return load_pair_record(self.read_record(index))


def deterministic_sample(path: Path, samples_per_label: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    heaps: dict[str, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
    manifest_counts: Counter[str] = Counter()
    rows = 0
    for line_number, record in iter_jsonl(path):
        rows += 1
        label = str(record.get("label"))
        manifest_counts[label] += 1
        if label not in LABELS:
            continue
        rank = int(hashlib.sha256(
            f"{PREFLIGHT_VERSION}|{label}|{record.get('sample_id')}".encode("utf-8")
        ).hexdigest(), 16)
        item = (-rank, -line_number, record)
        heap = heaps[label]
        if len(heap) < samples_per_label:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    selected = {
        label: [item[2] for item in sorted(heap, key=lambda value: (-value[0], -value[1]))]
        for label, heap in heaps.items()
    }
    return selected, {"rows": rows, "label_counts": dict(manifest_counts),
                      "selected_counts": {label: len(selected.get(label, [])) for label in LABELS}}


def relation_check(record: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    label, left, right = record["label"], record["left"], record["right"]
    left_match = (left["hash"], left["game_id"])
    right_match = (right["hash"], right["game_id"])
    left_raw, right_raw = left["map_memory_raw_frame_indices"], right["map_memory_raw_frame_indices"]
    disjoint = max(left_raw) < min(right_raw) or max(right_raw) < min(left_raw)
    checks = {
        "same_match": left_match == right_match,
        "same_episode": left_match == right_match and left["episode"] == right["episode"],
        "different_raw_start": int(left["raw_start"]) != int(right["raw_start"]),
        "raw_intervals_disjoint": disjoint,
    }
    if label == "positive":
        passed = checks["same_episode"] and not checks["different_raw_start"]
    elif label in {"time_shift", "wrong_window"}:
        passed = checks["same_episode"] and checks["different_raw_start"] and disjoint
    elif label == "cross_episode":
        passed = checks["same_match"] and left["episode"] != right["episode"]
    else:
        passed = left_match != right_match
    return bool(passed), checks


def array_memory(sample: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "left_rgb", "right_rgb", "left_pose", "right_pose", "left_intrinsics", "right_intrinsics",
        "left_ticks", "right_ticks", "left_frame_counts", "right_frame_counts",
    )
    breakdown = {key: int(sample[key].nbytes) for key in keys}
    total = sum(breakdown.values())
    return {"array_payload_bytes": total, "array_payload_mib": total / (1024 ** 2), "breakdown_bytes": breakdown,
            "excludes": "Python dictionaries, strings, decoder buffers, and endpoint metadata"}


def contact_sheet(sample: dict[str, Any], output_path: Path) -> None:
    left = sample["left_rgb"]
    right = sample["right_rgb"]
    _, _, height, width = left.shape
    header = 22
    canvas = Image.new("RGB", (width * 5, (height + header) * 2), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for column, time_index in enumerate(CONTACT_TIME_POSITIONS):
        for row_index, (name, frames) in enumerate((("left", left), ("right", right))):
            y = row_index * (height + header)
            draw.text((column * width + 4, y + 4), f"{name} t={time_index}", fill=(255, 255, 255))
            rgb_hwc = frames[time_index].transpose(1, 2, 0)
            canvas.paste(Image.fromarray(rgb_hwc, mode="RGB"), (column * width, y + header))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    selected, manifest_stats = deterministic_sample(args.pair_manifest, args.samples_per_label)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    success_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    endpoint_paths = endpoint_exists = 0
    decoded_frame_counts: Counter[int] = Counter()
    rgb_shapes: Counter[str] = Counter()
    rgb_dtypes: Counter[str] = Counter()
    pose_shapes: Counter[str] = Counter()
    pose_source_dtypes: Counter[str] = Counter()
    intr_shapes: Counter[str] = Counter()
    intr_source_dtypes: Counter[str] = Counter()
    nonfinite = Counter()
    positive_tick = Counter()
    relation_results: dict[str, Counter[str]] = defaultdict(Counter)
    contact_sheets: list[str] = []
    memory_example: dict[str, Any] | None = None
    sample_summaries: list[dict[str, Any]] = []

    for label in LABELS:
        sheets_written = 0
        for record in selected.get(label, []):
            for endpoint in (record.get("left", {}), record.get("right", {})):
                endpoint_paths += 1
                endpoint_exists += int(Path(str(endpoint.get("video", ""))).is_file())
            relation_passed, relation_detail = relation_check(record)
            relation_results[label]["passed" if relation_passed else "failed"] += 1
            try:
                sample = load_pair_record(record)
            except DataContractError as exc:
                failure_counts[exc.reason] += 1
                failures.append({"sample_id": record.get("sample_id"), "label": label,
                                 "reason": exc.reason, "detail": exc.detail})
                continue
            except Exception as exc:
                failure_counts["unexpected_error"] += 1
                failures.append({"sample_id": record.get("sample_id"), "label": label,
                                 "reason": "unexpected_error", "detail": repr(exc)})
                continue

            success_counts[label] += 1
            for side in ("left", "right"):
                load_meta = sample["load_metadata"][side]
                video_meta, camera_meta = load_meta["video"], load_meta["camera"]
                decoded_frame_counts[video_meta["decoded_frames"]] += 1
                rgb_shapes[str(video_meta["selected_rgb_shape"])] += 1
                rgb_dtypes[video_meta["selected_rgb_dtype"]] += 1
                pose_shapes[str(camera_meta["poses_selected_shape"])] += 1
                pose_source_dtypes[camera_meta["poses_source_dtype"]] += 1
                intr_shapes[str(camera_meta["intrinsics_selected_shape"])] += 1
                intr_source_dtypes[camera_meta["intrinsics_source_dtype"]] += 1
                nonfinite["pose_nan"] += camera_meta["poses_nan"]
                nonfinite["pose_inf"] += camera_meta["poses_inf"]
                nonfinite["intrinsics_nan"] += camera_meta["intrinsics_nan"]
                nonfinite["intrinsics_inf"] += camera_meta["intrinsics_inf"]
            if label == "positive":
                positive_tick["equal" if np.array_equal(sample["left_ticks"], sample["right_ticks"]) else "mismatch"] += 1
            if memory_example is None:
                memory_example = {"sample_id": sample["sample_id"], **array_memory(sample)}
            summary = {
                "sample_id": sample["sample_id"], "label": label,
                "left_rgb_shape": list(sample["left_rgb"].shape),
                "right_rgb_shape": list(sample["right_rgb"].shape),
                "pose_shape": list(sample["left_pose"].shape),
                "intrinsics_shape": list(sample["left_intrinsics"].shape),
                "rgb_dtype": str(sample["left_rgb"].dtype), "pose_dtype": str(sample["left_pose"].dtype),
                "intrinsics_dtype": str(sample["left_intrinsics"].dtype),
                "relation_passed": relation_passed, "relation": relation_detail,
            }
            sample_summaries.append(summary)
            if sheets_written < args.contact_sheets_per_label:
                sheet_path = args.output_dir / f"{label}__{sample['sample_id'][:16]}__contact.png"
                contact_sheet(sample, sheet_path)
                contact_sheets.append(str(sheet_path))
                sheets_written += 1

    report = {
        "preflight_version": PREFLIGHT_VERSION, "protocol_version": PROTOCOL_VERSION,
        "pair_manifest": str(args.pair_manifest), "manifest": manifest_stats,
        "requested_samples_per_label": args.samples_per_label,
        "successful_samples_by_label": dict(success_counts),
        "failed_samples": len(failures), "failure_reasons": dict(failure_counts), "failures": failures,
        "video": {
            "selected_endpoint_paths": endpoint_paths, "existing_endpoint_paths": endpoint_exists,
            "existence_rate": endpoint_exists / endpoint_paths if endpoint_paths else None,
            "decoded_frame_count_distribution": dict(decoded_frame_counts),
            "decoder_native_color_order": "BGR",
            "rgb_conversion_location": "decode_clip_rgb selected-frame boundary via cv2.COLOR_BGR2RGB",
            "selected_rgb_shape_distribution": dict(rgb_shapes), "selected_rgb_dtype_distribution": dict(rgb_dtypes),
            "resize_or_normalization": False,
        },
        "camera": {
            "pose_shape_distribution": dict(pose_shapes), "pose_source_dtype_distribution": dict(pose_source_dtypes),
            "intrinsics_shape_distribution": dict(intr_shapes),
            "intrinsics_source_dtype_distribution": dict(intr_source_dtypes), "nan_inf_counts": dict(nonfinite),
        },
        "positive_tick_equality": dict(positive_tick),
        "negative_relation_checks": {label: dict(counts) for label, counts in relation_results.items()},
        "single_pair_memory_estimate": memory_example,
        "contact_sheet_time_positions": list(CONTACT_TIME_POSITIONS),
        "contact_sheets": contact_sheets, "sample_summaries": sample_summaries,
        "prohibited_operations_performed": [],
    }
    report_path = args.output_dir / "preflight_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "pair_manifest", "successful_samples_by_label", "failed_samples", "failure_reasons", "video",
        "camera", "positive_tick_equality", "negative_relation_checks", "single_pair_memory_estimate",
        "contact_sheets",
    )}, ensure_ascii=False, indent=2, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-label", type=int, default=5)
    parser.add_argument("--contact-sheets-per-label", type=int, default=1)
    args = parser.parse_args()
    if args.samples_per_label < 1:
        parser.error("--samples-per-label must be >= 1")
    if args.contact_sheets_per_label < 0:
        parser.error("--contact-sheets-per-label must be >= 0")
    return args


def main() -> None:
    preflight(parse_args())


if __name__ == "__main__":
    main()
