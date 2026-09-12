#!/usr/bin/env python3
"""Score persistent person hallucinations on strict zero-visibility windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_path(value: str, report_path: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    if report_path is not None:
        candidate = report_path.parent / path
        if candidate.exists():
            return candidate
    return path


def visibility_intervals(payload: dict[str, Any]) -> list[tuple[int, int]]:
    intervals = []
    for player in payload.values():
        if not isinstance(player, dict):
            continue
        for item in player.get("ranges", []):
            bounds = item.get("range", [])
            if len(bounds) == 2 and int(bounds[1]) >= int(bounds[0]):
                intervals.append((int(bounds[0]), int(bounds[1])))
    return intervals


def overlaps(intervals: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(left <= end and right >= start for left, right in intervals)


def assert_zero_contract(row: dict[str, Any]) -> None:
    start = int(row.get("frame_count_start", row["raw_start"]))
    end = int(row.get("frame_count_end", start + 160))
    visibility_path = Path(str(row["player_visibility"]))
    with visibility_path.open("r", encoding="utf-8") as handle:
        visibility = json.load(handle)
    if overlaps(visibility_intervals(visibility), start, end):
        raise ValueError(f"{row['clip_id']}: visibility overlaps [{start}, {end}]")


def latent_frame_indices(row: dict[str, Any]) -> list[int]:
    for key in ("latent_frames", "positive_latent_frames"):
        values = row.get(key)
        if isinstance(values, list):
            return [int(value) for value in values]
    start = int(row["raw_start"])
    return list(range(start, start + 161, 8))


def load_generations(report_paths: list[Path], manifest_path: Path | None) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for report_path in report_paths:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for window in report.get("windows", []):
            if window.get("status") != "complete":
                continue
            clip_id = str(window["clip_id"])
            if clip_id in rows:
                raise ValueError(f"duplicate generated clip id: {clip_id}")
            rows[clip_id] = {
                "clip_id": clip_id,
                "generated_mp4": str(resolve_path(str(window["mp4"]), report_path)),
                "source": str(report_path),
            }
    if manifest_path is not None:
        for row in read_jsonl(manifest_path):
            clip_id = str(row["clip_id"])
            if clip_id in rows:
                raise ValueError(f"duplicate generated clip id: {clip_id}")
            value = row.get("generated_mp4") or row.get("mp4")
            if not value:
                raise ValueError(f"{clip_id}: generation manifest needs generated_mp4 or mp4")
            rows[clip_id] = {
                "clip_id": clip_id,
                "generated_mp4": str(resolve_path(str(value), manifest_path)),
                "source": str(manifest_path),
            }
    if not rows:
        raise ValueError("no complete generated windows found")
    return rows


def read_video_frames(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frames = {}
    for index in sorted(set(indices)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames[index] = frame
    cap.release()
    missing = sorted(set(indices) - set(frames))
    if missing:
        raise RuntimeError(f"missing frames {missing[:8]} from {path}")
    return frames


def bbox_center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5


def bbox_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def bbox_iou(first: list[float], second: list[float]) -> float:
    x0, y0 = max(first[0], second[0]), max(first[1], second[1])
    x1, y1 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = bbox_area(first) + bbox_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def in_ego_zone(box: list[float], width: int, height: int) -> bool:
    _center_x, center_y = bbox_center(box)
    bottom_connected = box[3] >= 0.90 * height and center_y >= 0.55 * height
    return bottom_connected


class Detector:
    def __init__(self, device: str, threshold: float, min_area_ratio: float) -> None:
        import torch
        import torchvision

        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.torch = torch
        self.device = torch.device(device)
        self.threshold = threshold
        self.min_area_ratio = min_area_ratio
        self.weights_name = str(weights)
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights).eval().to(self.device)

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[dict[str, Any]]]:
        tensors = [
            self.torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            .permute(2, 0, 1).float().div(255.0).to(self.device)
            for frame in frames
        ]
        with self.torch.inference_mode():
            outputs = self.model(tensors)
        results = []
        for frame, output in zip(frames, outputs):
            height, width = frame.shape[:2]
            detections = []
            for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
                score_value = float(score)
                if score_value < self.threshold:
                    break
                if int(label) != 1:
                    continue
                values = [round(float(value), 2) for value in box]
                if bbox_area(values) / (width * height) < self.min_area_ratio or in_ego_zone(values, width, height):
                    continue
                normalized = [
                    values[0] / width, values[1] / height,
                    values[2] / width, values[3] / height,
                ]
                detections.append({
                    "bbox": values,
                    "bbox_norm": [round(value, 6) for value in normalized],
                    "score": round(score_value, 6),
                    "area": round(bbox_area(values), 2),
                    "center": [round(value, 2) for value in bbox_center(values)],
                })
            results.append(detections)
        return results


def track_detections(frames: list[dict[str, Any]], max_gap: int) -> list[dict[str, Any]]:
    tracks: list[dict[str, Any]] = []
    for frame_position, frame in enumerate(frames):
        candidates = []
        for track_index, track in enumerate(tracks):
            gap = frame_position - int(track["last_position"])
            if gap > max_gap + 1:
                continue
            previous = track["last_bbox_norm"]
            px, py = bbox_center(previous)
            previous_diag = math.hypot(previous[2] - previous[0], previous[3] - previous[1])
            for detection_index, detection in enumerate(frame["detections"]):
                box = detection["bbox_norm"]
                cx, cy = bbox_center(box)
                distance = math.hypot(cx - px, cy - py)
                overlap = bbox_iou(previous, box)
                if overlap >= 0.05 or distance <= max(0.05, 0.75 * previous_diag):
                    candidates.append((-overlap, distance, track_index, detection_index))
        used_tracks, used_detections = set(), set()
        for _negative_iou, _distance, track_index, detection_index in sorted(candidates):
            if track_index in used_tracks or detection_index in used_detections:
                continue
            detection = frame["detections"][detection_index]
            detection["track_id"] = track_index
            tracks[track_index]["detections"].append({"sample_index": frame["sample_index"], **detection})
            tracks[track_index]["last_position"] = frame_position
            tracks[track_index]["last_bbox_norm"] = detection["bbox_norm"]
            used_tracks.add(track_index)
            used_detections.add(detection_index)
        for detection_index, detection in enumerate(frame["detections"]):
            if detection_index in used_detections:
                continue
            track_index = len(tracks)
            detection["track_id"] = track_index
            tracks.append({
                "track_id": track_index,
                "last_position": frame_position,
                "last_bbox_norm": detection["bbox_norm"],
                "detections": [{"sample_index": frame["sample_index"], **detection}],
            })
    return [
        {
            "track_id": track["track_id"],
            "frame_count": len(track["detections"]),
            "first_sample_index": min(row["sample_index"] for row in track["detections"]),
            "last_sample_index": max(row["sample_index"] for row in track["detections"]),
            "max_score": max(row["score"] for row in track["detections"]),
            "max_area": max(row["area"] for row in track["detections"]),
            "detections": track["detections"],
        }
        for track in tracks
    ]


def variant_summary(frames: list[dict[str, Any]], tracks: list[dict[str, Any]], min_track_frames: int) -> dict[str, Any]:
    persistent = [track for track in tracks if track["frame_count"] >= min_track_frames]
    persistent_samples = {
        detection["sample_index"] for track in persistent for detection in track["detections"]
    }
    detected_samples = {frame["sample_index"] for frame in frames if frame["detections"]}
    frame_count = len(frames)
    return {
        "frame_count": frame_count,
        "detection_count": sum(len(frame["detections"]) for frame in frames),
        "detection_frame_count": len(detected_samples),
        "detection_frame_rate": len(detected_samples) / frame_count if frame_count else None,
        "track_count": len(tracks),
        "persistent_track_count": len(persistent),
        "persistent_detection_frame_count": len(persistent_samples),
        "persistent_detection_frame_rate": len(persistent_samples) / frame_count if frame_count else None,
        "max_track_frames": max((track["frame_count"] for track in tracks), default=0),
    }


def score_variant(
    detector: Detector,
    frames_by_index: dict[int, np.ndarray],
    sample_to_video_frame: dict[int, int],
    batch_size: int,
    max_track_gap: int,
    min_track_frames: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sample_indices = sorted(sample_to_video_frame)
    output_rows = []
    for offset in range(0, len(sample_indices), batch_size):
        batch_indices = sample_indices[offset : offset + batch_size]
        batch_frames = [frames_by_index[sample_to_video_frame[index]] for index in batch_indices]
        detections = detector.detect_batch(batch_frames)
        for sample_index, rows in zip(batch_indices, detections):
            output_rows.append({
                "sample_index": sample_index,
                "video_frame": sample_to_video_frame[sample_index],
                "detections": rows,
            })
    tracks = track_detections(output_rows, max_track_gap)
    summary = variant_summary(output_rows, tracks, min_track_frames)
    return output_rows, tracks, summary


def annotate(frame: np.ndarray, detections: list[dict[str, Any]], title: str) -> np.ndarray:
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(output, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    for detection in detections:
        x0, y0, x1, y1 = [int(round(value)) for value in detection["bbox"]]
        cv2.rectangle(output, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(
            output,
            f"person {detection['score']:.2f} t{detection.get('track_id', -1)}",
            (x0, max(45, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
        )
    return output


def write_audits(out_dir: Path, windows: list[dict[str, Any]], limit: int, min_track_frames: int) -> list[str]:
    candidates = []
    for window in windows:
        track_lengths = {track["track_id"]: track["frame_count"] for track in window["generated_tracks"]}
        for frame in window["generated_frames"]:
            scores = [row["score"] for row in frame["detections"]]
            if scores:
                candidates.append((
                    max(scores),
                    max(track_lengths.get(row.get("track_id"), 0) >= min_track_frames for row in frame["detections"]),
                    window,
                    frame,
                ))
    audit_dir = out_dir / "audit_top_detections"
    audit_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for rank, (_score, _persistent, window, generated_row) in enumerate(
        sorted(candidates, reverse=True, key=lambda row: (row[1], row[0]))[:limit]
    ):
        sample_index = generated_row["sample_index"]
        gt_row = next(row for row in window["gt_frames"] if row["sample_index"] == sample_index)
        generated_frame = read_video_frames(Path(window["generated_mp4"]), [generated_row["video_frame"]])[generated_row["video_frame"]]
        gt_frame = read_video_frames(Path(window["gt_mp4"]), [gt_row["video_frame"]])[gt_row["video_frame"]]
        generated_image = annotate(generated_frame, generated_row["detections"], f"generated sample {sample_index}")
        gt_image = annotate(gt_frame, gt_row["detections"], f"strict-zero GT raw {gt_row['video_frame']}")
        if gt_image.shape[:2] != generated_image.shape[:2]:
            gt_image = cv2.resize(gt_image, (generated_image.shape[1], generated_image.shape[0]))
        output = np.concatenate([generated_image, gt_image], axis=1)
        path = audit_dir / f"{rank:03d}_{window['clip_id']}_s{sample_index:02d}.jpg"
        cv2.imwrite(str(path), output, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        written.append(str(path))
    return written


def aggregate(windows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    summaries = [window[f"{key}_summary"] for window in windows]
    frames = sum(summary["frame_count"] for summary in summaries)
    detection_frames = sum(summary["detection_frame_count"] for summary in summaries)
    persistent_frames = sum(summary["persistent_detection_frame_count"] for summary in summaries)
    return {
        "window_count": len(windows),
        "frame_count": frames,
        "detection_count": sum(summary["detection_count"] for summary in summaries),
        "detection_frame_rate": detection_frames / frames if frames else None,
        "persistent_track_count": sum(summary["persistent_track_count"] for summary in summaries),
        "persistent_track_count_per_window": sum(summary["persistent_track_count"] for summary in summaries) / len(windows) if windows else None,
        "persistent_detection_frame_rate": persistent_frames / frames if frames else None,
        "windows_with_persistent_detection": sum(summary["persistent_track_count"] > 0 for summary in summaries),
        "window_persistent_hallucination_rate": float(np.mean([summary["persistent_track_count"] > 0 for summary in summaries])) if summaries else None,
        "max_track_frames": max((summary["max_track_frames"] for summary in summaries), default=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--run-report", type=Path, action="append", default=[])
    parser.add_argument("--generation-manifest", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--min-bbox-area-ratio", type=float, default=0.00064)
    parser.add_argument("--generated-frame-stride", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=21)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--min-track-frames", type=int, default=2)
    parser.add_argument("--max-track-gap", type=int, default=1)
    parser.add_argument("--limit-windows", type=int, default=0)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--expected-windows", type=int, default=0)
    parser.add_argument("--expected-matches", type=int, default=0)
    parser.add_argument("--expected-per-match", type=int, default=0)
    parser.add_argument("--audit-images", type=int, default=24)
    parser.add_argument("--require-zero-contract", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-exam-contract", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not args.run_report and args.generation_manifest is None:
        parser.error("at least one --run-report or --generation-manifest is required")
    source_rows = read_jsonl(args.source_manifest)
    source_by_id = {str(row["clip_id"]): row for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise ValueError(f"duplicate source clip ids: {len(source_rows) - len(source_by_id)}")
    contract_failures = []
    source_match_counts: dict[str, int] = {}
    for row in source_rows:
        clip_id = str(row.get("clip_id"))
        match_id = row.get("game_id")
        if not match_id:
            contract_failures.append(f"{clip_id}: missing game_id")
            continue
        match_id = str(match_id)
        source_match_counts[match_id] = source_match_counts.get(match_id, 0) + 1
        if row.get("map_memory_split") not in {"val", "test"}:
            contract_failures.append(f"{clip_id}: non-heldout split {row.get('map_memory_split')}")
        if row.get("selection_role") != "context" or set(row.get("map_memory_selection_roles", [])) != {"context"}:
            contract_failures.append(f"{clip_id}: context role contract failed")
        if "max_visibility_pixel_percent" not in row or float(row["max_visibility_pixel_percent"] or 0.0) != 0.0:
            contract_failures.append(f"{clip_id}: strict-zero summary contract failed")
    if args.expected_per_match and any(count != args.expected_per_match for count in source_match_counts.values()):
        contract_failures.append(
            f"per-match counts are {sorted(source_match_counts.values())}, expected all {args.expected_per_match}"
        )
    if contract_failures and args.require_exam_contract:
        raise ValueError(f"source exam contract failed ({len(contract_failures)}): {contract_failures[:5]}")
    if args.selection_report is not None:
        selection = json.loads(args.selection_report.read_text(encoding="utf-8"))
        source_sha256 = hashlib.sha256(args.source_manifest.read_bytes()).hexdigest()
        if selection.get("status") != "pass" or selection.get("output_manifest_sha256") != source_sha256:
            raise ValueError("selection report status/hash does not bind to source manifest")
    generations = load_generations(args.run_report, args.generation_manifest)
    missing_generation_ids = [clip_id for clip_id in source_by_id if clip_id not in generations]
    unexpected_generation_ids = [clip_id for clip_id in generations if clip_id not in source_by_id]
    if missing_generation_ids and not args.allow_partial:
        raise ValueError(
            f"generation coverage incomplete: missing {len(missing_generation_ids)}/{len(source_by_id)}; "
            f"examples={missing_generation_ids[:5]}"
        )
    if args.limit_windows and not args.allow_partial:
        parser.error("--limit-windows requires explicit --allow-partial")
    clip_ids = [clip_id for clip_id in source_by_id if clip_id in generations]
    if args.limit_windows:
        clip_ids = clip_ids[: args.limit_windows]
    if not clip_ids:
        raise ValueError("no clip ids overlap source and generation manifests")
    selected_match_count = len({str(source_by_id[clip_id]["game_id"]) for clip_id in clip_ids})
    if args.expected_windows and len(clip_ids) != args.expected_windows:
        raise ValueError(f"matched windows {len(clip_ids)} != expected {args.expected_windows}")
    if args.expected_matches and selected_match_count != args.expected_matches:
        raise ValueError(f"matched matches {selected_match_count} != expected {args.expected_matches}")

    detector = Detector(args.device, args.score_threshold, args.min_bbox_area_ratio)
    windows = []
    for window_index, clip_id in enumerate(clip_ids):
        source = source_by_id[clip_id]
        if args.require_zero_contract:
            assert_zero_contract(source)
        generated_mp4 = Path(generations[clip_id]["generated_mp4"])
        gt_mp4 = Path(str(source["mp4"]))
        latent_frames = latent_frame_indices(source)[: args.max_samples]
        sample_count = len(latent_frames)
        generated_indices = [index * args.generated_frame_stride for index in range(sample_count)]
        generated_images = read_video_frames(generated_mp4, generated_indices)
        gt_images = read_video_frames(gt_mp4, latent_frames)
        sample_to_generated = {index: frame for index, frame in enumerate(generated_indices)}
        sample_to_gt = {index: frame for index, frame in enumerate(latent_frames)}
        generated_frames, generated_tracks, generated_summary = score_variant(
            detector, generated_images, sample_to_generated, args.batch_size, args.max_track_gap, args.min_track_frames
        )
        gt_frames, gt_tracks, gt_summary = score_variant(
            detector, gt_images, sample_to_gt, args.batch_size, args.max_track_gap, args.min_track_frames
        )
        windows.append({
            "clip_id": clip_id,
            "generated_mp4": str(generated_mp4),
            "gt_mp4": str(gt_mp4),
            "source_generation_record": generations[clip_id]["source"],
            "generated_summary": generated_summary,
            "gt_summary": gt_summary,
            "paired_excess_detection_frame_rate": generated_summary["detection_frame_rate"] - gt_summary["detection_frame_rate"],
            "paired_excess_persistent_frame_rate": generated_summary["persistent_detection_frame_rate"] - gt_summary["persistent_detection_frame_rate"],
            "generated_frames": generated_frames,
            "gt_frames": gt_frames,
            "generated_tracks": generated_tracks,
            "gt_tracks": gt_tracks,
        })
        print(json.dumps({
            "progress": f"{window_index + 1}/{len(clip_ids)}",
            "clip_id": clip_id,
            "generated_persistent_tracks": generated_summary["persistent_track_count"],
            "gt_persistent_tracks": gt_summary["persistent_track_count"],
        }), flush=True)

    generated_aggregate = aggregate(windows, "generated")
    gt_aggregate = aggregate(windows, "gt")
    paired_excess = [window["paired_excess_persistent_frame_rate"] for window in windows]
    paired_detection_excess = [
        window["paired_excess_detection_frame_rate"] for window in windows
    ]
    report = {
        "kind": "zero_visibility_hallucination_eval_v0",
        "status": "pass",
        "detector": "torchvision fasterrcnn_resnet50_fpn_v2 DEFAULT; COCO person; ego-zone filtered",
        "detector_weights": detector.weights_name,
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": hashlib.sha256(args.source_manifest.read_bytes()).hexdigest(),
        "selection_report": str(args.selection_report.resolve()) if args.selection_report else None,
        "selection_report_sha256": hashlib.sha256(args.selection_report.read_bytes()).hexdigest() if args.selection_report else None,
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items() if key not in {"run_report"}
        },
        "run_reports": [str(path.resolve()) for path in args.run_report],
        "matched_window_count": len(windows),
        "matched_match_count": selected_match_count,
        "coverage": {
            "source_window_count": len(source_rows),
            "generated_window_count": len(generations),
            "matched_window_count": len(clip_ids),
            "missing_generation_count": len(missing_generation_ids),
            "missing_generation_examples": missing_generation_ids[:20],
            "unexpected_generation_count": len(unexpected_generation_ids),
            "unexpected_generation_examples": unexpected_generation_ids[:20],
            "allow_partial": args.allow_partial,
        },
        "generated": generated_aggregate,
        "strict_zero_gt_detector_control": gt_aggregate,
        "paired_generated_minus_gt": {
            "detection_frame_rate_mean": float(np.mean(paired_detection_excess)),
            "detection_frame_rate_median": float(np.median(paired_detection_excess)),
            "persistent_detection_frame_rate_mean": float(np.mean(paired_excess)),
            "persistent_detection_frame_rate_median": float(np.median(paired_excess)),
            "windows_generated_detection_gt_clean": sum(
                window["generated_summary"]["detection_frame_count"] > 0
                and window["gt_summary"]["detection_frame_count"] == 0
                for window in windows
            ),
            "windows_generated_persistent_gt_clean": sum(
                window["generated_summary"]["persistent_track_count"] > 0
                and window["gt_summary"]["persistent_track_count"] == 0
                for window in windows
            ),
        },
        "windows": windows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "zero_visibility_hallucination_report_v0.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audits = write_audits(args.out_dir, windows, args.audit_images, args.min_track_frames)
    report["audit_images"] = audits
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = [
        "# Zero-visibility hallucination evaluation v0",
        "",
        f"- windows: {len(windows)}",
        f"- generated person-detection frame rate (primary): {generated_aggregate['detection_frame_rate']:.6f}",
        f"- strict-zero GT detector-control frame rate: {gt_aggregate['detection_frame_rate']:.6f}",
        f"- paired excess detection frame rate (primary): {float(np.mean(paired_detection_excess)):.6f}",
        f"- generated persistent frame rate: {generated_aggregate['persistent_detection_frame_rate']:.6f}",
        f"- strict-zero GT detector-control rate: {gt_aggregate['persistent_detection_frame_rate']:.6f}",
        f"- paired excess persistent frame rate: {float(np.mean(paired_excess)):.6f}",
        f"- generated-detected / GT-clean windows: {report['paired_generated_minus_gt']['windows_generated_detection_gt_clean']}/{len(windows)}",
        f"- generated windows with persistent detections: {generated_aggregate['windows_with_persistent_detection']}/{len(windows)}",
        f"- generated-positive / GT-clean windows: {report['paired_generated_minus_gt']['windows_generated_persistent_gt_clean']}/{len(windows)}",
    ]
    (args.out_dir / "SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "generated": generated_aggregate,
        "strict_zero_gt_detector_control": gt_aggregate,
        "paired_generated_minus_gt": report["paired_generated_minus_gt"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
