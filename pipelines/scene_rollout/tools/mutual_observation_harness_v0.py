#!/usr/bin/env python3
"""Mutual observation placement harness v1 over cleanv2 32-window outputs.

Expected players come from mesh_dense_condition_meta_v0.json next to each dense npz
(memory_projected_players from canonical capsule projection). Generated/GT evidence comes
from FasterRCNN person detections in the corresponding video frame.
"""
from __future__ import annotations
from runtime_paths import source_path
import argparse, glob, json, math, os, time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DENSE_W, DENSE_H = 320, 176
FULL_H = 480
RGB_STRIDE = 4
EGO_ZONE = {"x0": 0.25, "x1": 0.75, "y0": 0.55, "y1": 1.0}


def log(msg: str) -> None:
    print(f"[mutual_obs] {msg}", flush=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def bbox_center(b: list[float]) -> tuple[float, float]:
    return (b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5


def bbox_area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def in_ego_zone(b: list[float], w: int, h: int) -> bool:
    x, y = bbox_center(b)
    return EGO_ZONE["x0"] * w <= x <= EGO_ZONE["x1"] * w and EGO_ZONE["y0"] * h <= y <= EGO_ZONE["y1"] * h


def iou(a: list[float], b: list[float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    den = bbox_area(a) + bbox_area(b) - inter
    return inter / den if den > 0 else 0.0


class Detector:
    def __init__(self, device: str, threshold: float):
        import torch
        import torchvision
        self.torch = torch
        self.device = device
        self.threshold = threshold
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT").eval().to(device)

    def detect(self, frame_bgr: np.ndarray) -> list[dict[str, Any]]:
        import torch
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(self.device)
        with torch.no_grad():
            out = self.model([t])[0]
        boxes = out["boxes"].detach().cpu().numpy()
        labels = out["labels"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()
        h, w = frame_bgr.shape[:2]
        rows = []
        for b, lab, score in zip(boxes, labels, scores):
            if int(lab) != 1 or float(score) < self.threshold:
                continue
            bb = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
            if in_ego_zone(bb, w, h):
                continue
            rows.append({"bbox": bb, "score": float(score), "center": list(bbox_center(bb))})
        return rows


def read_frame(mp4: Path, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(mp4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def video_info(mp4: Path) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(mp4))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, w, h


def dense_to_video_box(player: dict[str, Any], w: int, h: int) -> list[float]:
    cu, cv = player["center_uv"]
    pr = float(player["pixel_radius"])
    v0, v1 = player["top_bottom_v"]
    x0, x1 = (cu - pr) * w / DENSE_W, (cu + pr) * w / DENSE_W
    if h < FULL_H:
        yoff = (FULL_H - h) * 0.5
        y0, y1 = v0 * FULL_H / DENSE_H - yoff, v1 * FULL_H / DENSE_H - yoff
    else:
        y0, y1 = v0 * h / DENSE_H, v1 * h / DENSE_H
    return [max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)]


def dense_to_video_center(player: dict[str, Any], w: int, h: int) -> list[float]:
    cu, cv = player["center_uv"]
    x = cu * w / DENSE_W
    if h < FULL_H:
        y = cv * FULL_H / DENSE_H - (FULL_H - h) * 0.5
    else:
        y = cv * h / DENSE_H
    return [float(x), float(y)]


def expected_from_meta(meta_path: Path, w: int, h: int, min_visible_px: int) -> list[dict[str, Any]]:
    meta = load_json(meta_path)
    out = []
    for p in meta.get("memory_projected_players", []):
        if not p.get("visible_by_depth_test", False):
            continue
        vis = int(p.get("visible_pixels_after_depth_test", 0))
        if vis < min_visible_px:
            continue
        bb = dense_to_video_box(p, w, h)
        if bbox_area(bb) <= 1:
            continue
        out.append({"stem": p.get("stem"), "visible_pixels": vis, "bbox": bb, "center": dense_to_video_center(p, w, h), "width": max(1.0, bb[2] - bb[0]), "raw": p})
    return out


def meta_for_dense(dense_path: str) -> Path:
    return Path(dense_path).with_name("mesh_dense_condition_meta_v0.json")


def load_manifest_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(report["dense_sequence_manifest"]).read_text().splitlines() if line.strip()]
    by_sid = {r.get("sample_id"): r for r in rows}
    picked = [by_sid[sid] for sid in report.get("sample_ids", []) if sid in by_sid]
    return picked if picked else rows[: int(report.get("eff_latent_frames", 21))]


def find_mp4(gen_root: Path, variant: str, clip_id: str) -> Path | None:
    patterns = [gen_root / variant / f"{clip_id}_{variant}_lf21_*.mp4", gen_root / variant / f"{clip_id}_{variant}_*.mp4", gen_root / f"{clip_id}_{variant}_lf21_*.mp4", gen_root / f"{clip_id}_{variant}_*.mp4"]
    for pat in patterns:
        g = sorted(glob.glob(str(pat)))
        if g:
            return Path(g[0])
    return None


def find_report(gen_root: Path, variant: str, clip_id: str) -> Path | None:
    patterns = [gen_root / variant / f"{clip_id}_{variant}_report.json", gen_root / f"{clip_id}_{variant}_report.json"]
    for pat in patterns:
        g = sorted(glob.glob(str(pat)))
        if g:
            return Path(g[0])
    return None


def gt_mp4_from_report(report: dict[str, Any]) -> Path:
    if report.get("gt_path"):
        return Path(report["gt_path"])
    return Path(report["clip_dir"]) / "video.mp4"


def match_expected(expected: list[dict[str, Any]], detections: list[dict[str, Any]], center_gate_scale: float) -> tuple[list[dict[str, Any]], set[int], set[int]]:
    candidates = []
    for ei, e in enumerate(expected):
        ex, ey = e["center"]
        gate = max(1.0, float(e["width"]) * center_gate_scale)
        for di, d in enumerate(detections):
            dx, dy = d["center"]
            dist = math.hypot(dx - ex, dy - ey)
            ov = iou(e["bbox"], d["bbox"])
            if dist < gate or ov > 0.05:
                candidates.append((dist, -ov, ei, di))
    candidates.sort()
    used_e, used_d, matches = set(), set(), []
    for dist, neg_iou, ei, di in candidates:
        if ei in used_e or di in used_d:
            continue
        e, d = expected[ei], detections[di]
        ratio = bbox_area(d["bbox"]) / max(1.0, bbox_area(e["bbox"]))
        matches.append({"expected_index": ei, "det_index": di, "stem": e.get("stem"), "center_error_px": float(dist), "bbox_size_ratio": float(ratio), "ok_center_lt_width": bool(dist < e["width"]), "iou": float(-neg_iou)})
        used_e.add(ei); used_d.add(di)
    return matches, used_e, used_d


def player_count_error_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errs = [int(r["n_detections"]) - int(r["n_expected"]) for r in rows]
    if not errs:
        return {"player_count_error_mean": None, "player_count_error_distribution": {}, "player_count_error_over_rate": None, "player_count_error_under_rate": None, "player_count_error_abs_mean": None}
    dist = {str(k): int(errs.count(k)) for k in sorted(set(errs))}
    return {
        "player_count_error_mean": float(np.mean(errs)),
        "player_count_error_abs_mean": float(np.mean(np.abs(errs))),
        "player_count_error_distribution": dist,
        "player_count_error_over_rate": float(np.mean([e > 0 for e in errs])),
        "player_count_error_under_rate": float(np.mean([e < 0 for e in errs])),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_expected = sum(r["n_expected"] for r in rows)
    n_matched = sum(r["n_matched"] for r in rows)
    n_det = sum(r["n_detections"] for r in rows)
    halluc = sum(max(0, r["n_detections"] - r["n_matched"]) for r in rows)
    center = [m["center_error_px"] for r in rows for m in r["matches"]]
    ratios = [m["bbox_size_ratio"] for r in rows for m in r["matches"]]
    correct = [m["ok_center_lt_width"] for r in rows for m in r["matches"]]
    out = {"frames": len(rows), "n_expected": n_expected, "n_matched": n_matched, "n_detections": n_det, "miss_rate": (n_expected - n_matched) / n_expected if n_expected else None, "hallucination_rate": halluc / n_det if n_det else None, "center_error_px_median": float(np.median(center)) if center else None, "center_error_px_mean": float(np.mean(center)) if center else None, "bbox_size_ratio_median": float(np.median(ratios)) if ratios else None, "placement_correct_rate": float(np.mean(correct)) if correct else None}
    out.update(player_count_error_summary(rows))
    return out


def add_temporal(rows: list[dict[str, Any]], jump_thresh: float) -> dict[str, Any]:
    by_key = defaultdict(list)
    for r in rows:
        for m in r["matches"]:
            e = r["expected"][m["expected_index"]]
            d = r["detections"][m["det_index"]]
            by_key[(r["clip_id"], e.get("stem"))].append((r["latent_index"], d["center"]))
    jumps = 0; total = 0
    for vals in by_key.values():
        vals.sort()
        for (_, a), (_, b) in zip(vals, vals[1:]):
            total += 1
            if math.hypot(b[0] - a[0], b[1] - a[1]) > jump_thresh:
                jumps += 1
    return {"jump_threshold_px": jump_thresh, "center_transitions": total, "teleport_count": jumps, "teleport_rate": jumps / total if total else None}


def mutual_success(rows: list[dict[str, Any]]) -> dict[str, Any]:
    idx = {}
    for r in rows:
        for m in r["matches"]:
            stem = r["expected"][m["expected_index"]].get("stem")
            idx[(r["ep"], r["raw_frame"], r["ego_stem"], stem)] = bool(m["ok_center_lt_width"])
    total = ok = 0
    for (ep, raw, ego, target), good in list(idx.items()):
        if (ep, raw, target, ego) in idx:
            if ego < target:
                total += 1
                ok += int(good and idx[(ep, raw, target, ego)])
    return {"mutual_pairs": total, "mutual_observation_success_rate": ok / total if total else None, "mutual_success_count": ok}


def eval_one_video(variant: str, mp4: Path, report: dict[str, Any], detector: Detector, args) -> list[dict[str, Any]]:
    nframes, w, h = video_info(mp4)
    rows = []
    manifest = load_manifest_rows(report)
    for row in manifest[: args.max_latents]:
        li = int(row.get("latent_index", row.get("index", 0)))
        frame_idx = int(row.get("gen_frame", li * RGB_STRIDE)) if variant != "gt" else int(row.get("gen_frame", li * RGB_STRIDE))
        if frame_idx >= nframes:
            frame_idx = min(nframes - 1, li * RGB_STRIDE)
        frame = read_frame(mp4, frame_idx)
        if frame is None:
            continue
        meta_path = meta_for_dense(row["dense_path"])
        expected = expected_from_meta(meta_path, w, h, args.min_visible_px) if meta_path.exists() else []
        detections = detector.detect(frame)
        matches, used_e, used_d = match_expected(expected, detections, args.center_gate_scale)
        rows.append({"variant": variant, "clip_id": report["clip_id"], "ep": report.get("player_stem", "").split("_team_")[0], "ego_stem": report.get("player_stem"), "latent_index": li, "raw_frame": int(row.get("raw_frame", -1)), "video_frame": frame_idx, "n_expected": len(expected), "n_detections": len(detections), "n_matched": len(matches), "expected": expected, "detections": detections, "matches": matches, "meta_path": str(meta_path), "mp4": str(mp4)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-root", type=Path, default=Path(str(source_path('project', 'output/cleanv2_verdict_20260703_v0/gen/v2'))))
    ap.add_argument("--out-dir", type=Path, default=Path(str(source_path('scene', 'output/mutual_obs_harness_20260705_v0'))))
    ap.add_argument("--variants", default="gt,base,true_dense,shuffled_dense")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--score-threshold", type=float, default=0.5)
    ap.add_argument("--min-visible-px", type=int, default=50)
    ap.add_argument("--center-gate-scale", type=float, default=1.0)
    ap.add_argument("--teleport-threshold-px", type=float, default=80.0)
    ap.add_argument("--max-latents", type=int, default=21)
    ap.add_argument("--limit-clips", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_reports = sorted((args.gen_root / "true_dense").glob("*_true_dense_report.json"))
    if not base_reports:
        base_reports = sorted(args.gen_root.glob("*_true_dense_report.json"))
    if args.limit_clips:
        base_reports = base_reports[: args.limit_clips]
    detector = Detector(args.device, args.score_threshold)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    all_rows, missing = [], []
    for i, trp in enumerate(base_reports):
        true_report = load_json(trp)
        clip_id = true_report["clip_id"]
        for variant in variants:
            if variant == "gt":
                report = true_report
                mp4 = gt_mp4_from_report(report)
            else:
                rp = find_report(args.gen_root, variant, clip_id)
                mp4 = find_mp4(args.gen_root, variant, clip_id)
                if rp is None or mp4 is None:
                    missing.append({"clip_id": clip_id, "variant": variant, "report": str(rp), "mp4": str(mp4)})
                    continue
                report = load_json(rp)
            t0 = time.time()
            rows = eval_one_video(variant, mp4, report, detector, args)
            all_rows.extend(rows)
            log(f"{i+1}/{len(base_reports)} {variant} {clip_id[:12]} rows={len(rows)} sec={time.time()-t0:.1f}")
    by_variant = {}
    for v in variants:
        rows = [r for r in all_rows if r["variant"] == v]
        s = summarize(rows)
        s.update(add_temporal(rows, args.teleport_threshold_px))
        s.update(mutual_success(rows))
        by_variant[v] = s
    out = {"kind": "mutual_observation_harness_v0", "status": "ok", "params": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "by_variant": by_variant, "missing": missing, "per_frame": all_rows}
    (args.out_dir / "mutual_observation_scores.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = ["# Mutual observation harness v1", "", "| variant | frames | expected | detections | count_err_mean | count_err_abs | over_rate | dist | miss_rate | halluc_rate |", "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    for v in variants:
        s = by_variant[v]
        md.append(f"| {v} | {s['frames']} | {s['n_expected']} | {s['n_detections']} | {s['player_count_error_mean']} | {s['player_count_error_abs_mean']} | {s['player_count_error_over_rate']} | {s['player_count_error_distribution']} | {s['miss_rate']} | {s['hallucination_rate']} |")
    (args.out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"WROTE {args.out_dir / 'mutual_observation_scores.json'} and {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
