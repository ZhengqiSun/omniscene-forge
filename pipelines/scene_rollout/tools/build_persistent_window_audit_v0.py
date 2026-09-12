#!/usr/bin/env python3
"""Create one generated-versus-GT audit image per persistent-detection window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_frame(path: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"cannot read frame {index}: {path}")
    return frame


def resize_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    if frame.shape[0] == height:
        return frame
    width = max(1, int(round(frame.shape[1] * height / frame.shape[0])))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("status") != "pass" or report.get("matched_window_count") != 28:
        raise ValueError("input is not a passing exact 28-window report")
    expected = int(report.get("generated", {}).get("windows_with_persistent_detection", -1))
    windows = [window for window in report.get("windows", []) if window["generated_summary"]["persistent_track_count"] > 0]
    if len(windows) != expected:
        raise ValueError(f"persistent window rows {len(windows)} != aggregate {expected}")
    args.out_dir.mkdir(parents=True)
    records = []
    contact_tiles = []
    for rank, window in enumerate(
        sorted(windows, key=lambda item: item["generated_summary"]["persistent_detection_frame_rate"], reverse=True),
        start=1,
    ):
        track = max(window["generated_tracks"], key=lambda item: (item["frame_count"], item["max_score"]))
        detection = max(track["detections"], key=lambda item: item["score"])
        sample_index = int(detection["sample_index"])
        generated_frame_index = int(window["generated_frames"][sample_index]["video_frame"])
        gt_frame_index = int(window["gt_frames"][sample_index]["video_frame"])
        generated = read_frame(Path(window["generated_mp4"]), generated_frame_index)
        gt = read_frame(Path(window["gt_mp4"]), gt_frame_index)
        generated_shape = list(generated.shape)
        gt_shape = list(gt.shape)
        x1, y1, x2, y2 = [int(round(value)) for value in detection["bbox"]]
        cv2.rectangle(generated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            generated,
            f"generated track={track['track_id']} frames={track['frame_count']} score={detection['score']:.3f}",
            (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        cv2.putText(gt, f"strict-zero GT raw={gt_frame_index}", (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        gt = resize_to_height(gt, generated.shape[0])
        paired = np.concatenate([generated, gt], axis=1)
        clip_id = str(window["clip_id"])
        output = args.out_dir / f"{rank:02d}_{clip_id}.jpg"
        if not cv2.imwrite(str(output), paired, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"failed to write {output}")
        tile = cv2.resize(paired, (832, 240), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, f"window {rank:02d}", (12, 228), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        contact_tiles.append(tile)
        records.append({
            "rank": rank,
            "clip_id": clip_id,
            "persistent_track_count": window["generated_summary"]["persistent_track_count"],
            "persistent_detection_frame_rate": window["generated_summary"]["persistent_detection_frame_rate"],
            "selected_track_id": track["track_id"],
            "selected_track_frames": track["frame_count"],
            "selected_score": detection["score"],
            "sample_index": sample_index,
            "generated_video_frame": generated_frame_index,
            "gt_video_frame": gt_frame_index,
            "generated_frame_shape": generated_shape,
            "gt_frame_shape": gt_shape,
            "audit_image": str(output),
        })
    columns = 3
    blank = np.zeros_like(contact_tiles[0])
    padded = contact_tiles + [blank] * ((columns - len(contact_tiles) % columns) % columns)
    rows = [np.concatenate(padded[index : index + columns], axis=1) for index in range(0, len(padded), columns)]
    contact = args.out_dir / "persistent_windows_contact_sheet.jpg"
    if not cv2.imwrite(str(contact), np.concatenate(rows, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"failed to write {contact}")
    payload = {
        "kind": "persistent_window_visual_audit_v0",
        "status": "pass",
        "source_report": str(args.report.resolve()),
        "persistent_window_count": len(records),
        "policy": "one image per persistent window; longest track, then highest-score detection; generated left, aligned strict-zero GT right",
        "contact_sheet": str(contact),
        "windows": records,
    }
    (args.out_dir / "audit_manifest_v0.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "persistent_window_count": len(records), "contact_sheet": str(contact)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
