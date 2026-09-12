#!/usr/bin/env python3
"""Calibrate torchvision FasterRCNN person detection against CS:GO seg ground truth.

Purpose: quantify the detector ceiling for player detection on real CS:GO
first-person frames, as step 1 of the spatial-placement metric pipeline.

GT extraction follows tools/build_dense_condition_v0.py (color_mask /
build_masks): seg_colormap.json maps entity_handle -> RGB; player visibility
ranges gate which players to look for; ego handle is excluded (its color may
still paint first-person arms).

GT boxes: per non-ego player, connected components of the color mask; every
component with >= --min-gt-pixels pixels becomes one GT bbox.

Detector: fasterrcnn_resnet50_fpn_v2 (COCO, weights='DEFAULT'), CPU, person
class (label==1), score > 0.5. Matching: greedy by IoU, threshold 0.5.

Outputs: metrics JSON + 6 visualization PNGs (GT green, detection red,
bottom-center ego-arm zone blue) under --output-dir.

CPU only. Does not write anywhere outside --output-dir.
"""

from __future__ import annotations
from runtime_paths import source_path

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PLAYER_RE = re.compile(r"^Ep_\d+_team_(\d+)_player_(\d+)_inst_(\d+)$")

# Bottom-center zone where ego first-person arms/weapon live (fractions of W/H).
EGO_ZONE = {"x0": 0.25, "x1": 0.75, "y0": 0.55, "y1": 1.0}

AREA_BUCKETS = [
    ("small_lt1000", 0, 1000),
    ("mid_1000_5000", 1000, 5000),
    ("large_gt5000", 5000, float("inf")),
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_ego(stem: str) -> tuple[int, int]:
    m = PLAYER_RE.match(stem)
    if not m:
        raise ValueError(f"Cannot parse ego player stem: {stem}")
    return int(m.group(1)), int(m.group(2))


def color_mask(seg_bgr: np.ndarray, rgb: list[int], tolerance: int) -> np.ndarray:
    # OpenCV returns BGR frames. Seg colormap stores RGB bytes.
    target = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.int16)
    diff = np.abs(seg_bgr.astype(np.int16) - target.reshape(1, 1, 3))
    return (diff <= tolerance).all(axis=2)


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def bucket_of(bbox_area: float) -> str:
    for name, lo, hi in AREA_BUCKETS:
        if lo <= bbox_area < hi:
            return name
    return AREA_BUCKETS[-1][0]


def in_ego_zone(bbox: list[float], w: int, h: int) -> bool:
    cx = 0.5 * (bbox[0] + bbox[2])
    cy = 0.5 * (bbox[1] + bbox[3])
    return (
        EGO_ZONE["x0"] * w <= cx <= EGO_ZONE["x1"] * w
        and EGO_ZONE["y0"] * h <= cy <= EGO_ZONE["y1"] * h
    )


def pick_streams(root: Path, n_episodes: int) -> list[dict[str, Any]]:
    """Pick one ego stream from the first episode of each of the first
    n_episodes matches (sorted), for diversity across maps/matches."""
    streams: list[dict[str, Any]] = []
    for match_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        train = match_dir / "train"
        if not train.is_dir():
            continue
        for ep_dir in sorted(p for p in train.iterdir() if p.is_dir()):
            mp4s = sorted(ep_dir.glob("Ep_*_inst_000.mp4"))
            picked = None
            for mp4 in mp4s:
                stem = mp4.stem
                if (
                    (ep_dir / f"{stem}_seg.mkv").exists()
                    and (ep_dir / f"{stem}_seg_colormap.json").exists()
                    and (ep_dir / f"{stem}_player_visibility.json").exists()
                ):
                    picked = stem
                    break
            if picked:
                streams.append({"match": match_dir.name, "episode_dir": ep_dir, "stem": picked})
                break  # one episode per match
        if len(streams) >= n_episodes:
            break
    return streams


def sample_frame_indices(n_frames: int, k: int, skip_head: int = 16) -> list[int]:
    lo = min(skip_head, max(0, n_frames - 1))
    hi = max(lo + 1, n_frames - 8)
    idx = np.linspace(lo, hi - 1, k).round().astype(int)
    return sorted(set(int(i) for i in idx))


def extract_gt_boxes(
    seg_bgr: np.ndarray,
    colormap: dict[str, Any],
    visibility: dict[str, Any],
    ego_handle: str,
    frame_index: int,
    tolerance: int,
    min_pixels: int,
) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for _pid, entry in visibility.items():
        ranges = entry.get("ranges", [])
        if not any(r["range"][0] <= frame_index <= r["range"][1] for r in ranges):
            continue
        handle = str(entry.get("entity_handle"))
        if handle == ego_handle:
            continue
        cmap = colormap.get(handle)
        if not cmap:
            continue
        mask = color_mask(seg_bgr, cmap["color"], tolerance=tolerance)
        if int(mask.sum()) < min_pixels:
            continue
        n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        for cc in range(1, n_cc):
            x, y, w, h, area = stats[cc]
            if area < min_pixels:
                continue
            boxes.append(
                {
                    "bbox": [float(x), float(y), float(x + w), float(y + h)],
                    "mask_pixels": int(area),
                    "bbox_area": float(w * h),
                    "entity_handle": handle,
                }
            )
    return boxes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path(str(source_path('assets', 'csgo-datasets-fullsubset/32f1644d4f42c29d'))),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            str(source_path('project', 'output/qxq_detector_calibration_20260611'))
        ),
    )
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--frames-per-episode", type=int, default=14)
    ap.add_argument("--score-threshold", type=float, default=0.5)
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--color-tolerance", type=int, default=8)
    ap.add_argument("--min-gt-pixels", type=int, default=100)
    ap.add_argument("--num-vis", type=int, default=6)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch  # noqa: PLC0415  (heavy import after arg parsing)
    import torchvision  # noqa: PLC0415

    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = torch.device("cpu")
    print("[calib] loading fasterrcnn_resnet50_fpn_v2 (COCO DEFAULT weights)...")
    t0 = time.time()
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    model.eval().to(device)
    print(f"[calib] model ready in {time.time() - t0:.1f}s")

    streams = pick_streams(args.data_root, args.episodes)
    if not streams:
        raise SystemExit("No usable streams found under data root")
    print(f"[calib] streams: {[(s['match'][:8], s['episode_dir'].name, s['stem']) for s in streams]}")

    per_frame: list[dict[str, Any]] = []
    vis_candidates: list[dict[str, Any]] = []

    for stream in streams:
        ep_dir: Path = stream["episode_dir"]
        stem: str = stream["stem"]
        ego_team, ego_pidx = parse_ego(stem)
        game = load_json(ep_dir / "game_manifest.json")
        ego_handle = ""
        for p in game["players"]:
            if int(p["team_id"]) == ego_team and int(p["player_index"]) == ego_pidx:
                ego_handle = str(p["entity_handle"])
                break
        colormap = load_json(ep_dir / f"{stem}_seg_colormap.json")
        visibility = load_json(ep_dir / f"{stem}_player_visibility.json")

        rgb_cap = cv2.VideoCapture(str(ep_dir / f"{stem}.mp4"))
        seg_cap = cv2.VideoCapture(str(ep_dir / f"{stem}_seg.mkv"))
        n_frames = int(min(
            rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT), seg_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        ))
        indices = sample_frame_indices(n_frames, args.frames_per_episode)
        print(f"[calib] {stream['match'][:8]}/{ep_dir.name}/{stem}: {len(indices)} frames of {n_frames}")

        for fi in indices:
            rgb_cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok_rgb, rgb_bgr = rgb_cap.read()
            seg_cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok_seg, seg_bgr = seg_cap.read()
            if not ok_rgb or not ok_seg:
                print(f"[calib] WARN: failed to read frame {fi}, skipping")
                continue
            h, w = rgb_bgr.shape[:2]

            gt_boxes = extract_gt_boxes(
                seg_bgr, colormap, visibility, ego_handle, fi,
                args.color_tolerance, args.min_gt_pixels,
            )

            img = torch.from_numpy(
                cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            ).permute(2, 0, 1).float() / 255.0
            t0 = time.time()
            with torch.no_grad():
                out = model([img.to(device)])[0]
            infer_s = time.time() - t0

            keep = (out["labels"] == 1) & (out["scores"] > args.score_threshold)
            det_boxes = out["boxes"][keep].cpu().numpy().tolist()
            det_scores = out["scores"][keep].cpu().numpy().tolist()

            # Greedy IoU matching, highest IoU pairs first.
            pairs = []
            for gi, g in enumerate(gt_boxes):
                for di, d in enumerate(det_boxes):
                    v = iou_xyxy(g["bbox"], d)
                    if v >= args.iou_threshold:
                        pairs.append((v, gi, di))
            pairs.sort(reverse=True)
            gt_matched: dict[int, int] = {}
            det_matched: dict[int, int] = {}
            for v, gi, di in pairs:
                if gi in gt_matched or di in det_matched:
                    continue
                gt_matched[gi] = di
                det_matched[di] = gi

            dets = []
            for di, (d, s) in enumerate(zip(det_boxes, det_scores)):
                dets.append({
                    "bbox": [float(x) for x in d],
                    "score": float(s),
                    "matched": di in det_matched,
                    "in_ego_zone": in_ego_zone(d, w, h),
                })
            for gi, g in enumerate(gt_boxes):
                g["matched"] = gi in gt_matched
                g["bucket"] = bucket_of(g["bbox_area"])

            rec = {
                "match": stream["match"],
                "episode": ep_dir.name,
                "stem": stem,
                "frame_index": fi,
                "width": w,
                "height": h,
                "infer_seconds": round(infer_s, 2),
                "gt_boxes": gt_boxes,
                "detections": dets,
            }
            per_frame.append(rec)
            vis_candidates.append({"rec": rec, "rgb": rgb_bgr.copy()})
            print(
                f"[calib]   frame {fi}: gt={len(gt_boxes)} det={len(dets)} "
                f"matched={len(gt_matched)} ({infer_s:.1f}s)"
            )
        rgb_cap.release()
        seg_cap.release()

    # ---- Aggregate metrics ----
    n_gt = sum(len(r["gt_boxes"]) for r in per_frame)
    n_det = sum(len(r["detections"]) for r in per_frame)
    n_gt_matched = sum(sum(1 for g in r["gt_boxes"] if g["matched"]) for r in per_frame)
    n_det_matched = sum(sum(1 for d in r["detections"] if d["matched"]) for r in per_frame)

    bucket_stats = {}
    for name, _, _ in AREA_BUCKETS:
        gts = [g for r in per_frame for g in r["gt_boxes"] if g["bucket"] == name]
        m = sum(1 for g in gts if g["matched"])
        bucket_stats[name] = {
            "num_gt": len(gts),
            "num_matched": m,
            "recall": round(m / len(gts), 4) if gts else None,
        }

    ego_dets = [d for r in per_frame for d in r["detections"] if d["in_ego_zone"]]
    ego_unmatched = [d for d in ego_dets if not d["matched"]]
    n_det_outside_ego = n_det - len(ego_unmatched)

    summary = {
        "config": {
            "data_root": str(args.data_root),
            "episodes": [
                {"match": s["match"], "episode": s["episode_dir"].name, "stem": s["stem"]}
                for s in streams
            ],
            "frames_evaluated": len(per_frame),
            "detector": "torchvision fasterrcnn_resnet50_fpn_v2 COCO DEFAULT, CPU",
            "score_threshold": args.score_threshold,
            "iou_threshold": args.iou_threshold,
            "color_tolerance": args.color_tolerance,
            "min_gt_pixels": args.min_gt_pixels,
            "gt_definition": (
                "per non-ego visible player, 8-connected components of seg color "
                "mask; each component >= min_gt_pixels is one GT bbox"
            ),
            "ego_zone": EGO_ZONE,
        },
        "overall": {
            "num_gt": n_gt,
            "num_det": n_det,
            "num_gt_matched": n_gt_matched,
            "num_det_matched": n_det_matched,
            "recall": round(n_gt_matched / n_gt, 4) if n_gt else None,
            "precision": round(n_det_matched / n_det, 4) if n_det else None,
            "precision_excluding_unmatched_ego_zone": (
                round(n_det_matched / n_det_outside_ego, 4) if n_det_outside_ego else None
            ),
        },
        "recall_by_gt_bbox_area": bucket_stats,
        "ego_zone_analysis": {
            "zone": "bbox center in x[0.25W,0.75W] x y[0.55H,H]",
            "num_det_in_zone": len(ego_dets),
            "num_det_in_zone_unmatched": len(ego_unmatched),
            "mean_score_unmatched_in_zone": (
                round(float(np.mean([d["score"] for d in ego_unmatched])), 4)
                if ego_unmatched else None
            ),
        },
        "mean_infer_seconds_per_frame": round(
            float(np.mean([r["infer_seconds"] for r in per_frame])), 2
        ) if per_frame else None,
    }

    with (args.output_dir / "calibration_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (args.output_dir / "per_frame_results.json").open("w", encoding="utf-8") as f:
        json.dump(per_frame, f, indent=2)

    # ---- Visualizations: prefer frames with most GT boxes, ensure episode mix ----
    vis_candidates.sort(
        key=lambda c: (len(c["rec"]["gt_boxes"]), len(c["rec"]["detections"])),
        reverse=True,
    )
    chosen: list[dict[str, Any]] = []
    seen_eps: set[str] = set()
    for c in vis_candidates:  # first pass: one per episode
        ep = c["rec"]["episode"]
        if ep not in seen_eps:
            chosen.append(c)
            seen_eps.add(ep)
        if len(chosen) >= args.num_vis:
            break
    for c in vis_candidates:  # fill up
        if len(chosen) >= args.num_vis:
            break
        if c not in chosen:
            chosen.append(c)

    for c in chosen[: args.num_vis]:
        rec = c["rec"]
        img = c["rgb"]
        h, w = img.shape[:2]
        cv2.rectangle(
            img,
            (int(EGO_ZONE["x0"] * w), int(EGO_ZONE["y0"] * h)),
            (int(EGO_ZONE["x1"] * w), int(EGO_ZONE["y1"] * h)),
            (255, 128, 0), 1,
        )
        for g in rec["gt_boxes"]:
            x0, y0, x1, y1 = [int(v) for v in g["bbox"]]
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(img, f"GT {int(g['bbox_area'])}", (x0, max(12, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        for d in rec["detections"]:
            x0, y0, x1, y1 = [int(v) for v in d["bbox"]]
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.putText(img, f"{d['score']:.2f}", (x0, min(h - 4, y1 + 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        name = f"vis_{rec['match'][:8]}_{rec['episode']}_f{rec['frame_index']:05d}.png"
        cv2.imwrite(str(args.output_dir / name), img)
        print(f"[calib] wrote {name}")

    print("[calib] summary:")
    print(json.dumps(summary["overall"], indent=2))
    print(json.dumps(summary["recall_by_gt_bbox_area"], indent=2))
    print(json.dumps(summary["ego_zone_analysis"], indent=2))
    print(f"[calib] outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
