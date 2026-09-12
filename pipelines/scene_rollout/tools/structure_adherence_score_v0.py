#!/usr/bin/env python3
"""World-structure adherence v0.

Scores generated videos by comparing monocular depth from RGB frames against the
dense mesh-depth condition. Main shuffled score uses the condition actually fed
to the model; an extra cross score uses the window's original condition.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr


VARIANTS = ("base", "true_dense", "shuffled_dense")


@dataclass
class DepthModel:
    name: str
    version: str
    device: str
    kind: str
    model: Any
    processor: Any = None
    transform: Any = None

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray) -> np.ndarray:
        if self.kind == "transformers":
            image = Image.fromarray(rgb)
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model(**inputs)
            pred = out.predicted_depth
            pred = F.interpolate(
                pred.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            return pred.detach().float().cpu().numpy()
        if self.kind == "midas":
            batch = self.transform(rgb).to(self.device)
            pred = self.model(batch)
            pred = F.interpolate(
                pred.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            return pred.detach().float().cpu().numpy()
        raise RuntimeError(f"unknown model kind: {self.kind}")


def package_version(name: str) -> str:
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception as exc:  # pragma: no cover
        return f"unavailable:{exc}"


def load_depth_model(device: str, model_name: str) -> DepthModel:
    errors: List[str] = []
    if model_name in ("auto", "midas-small"):
        try:
            model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            model.to(device).eval()
            return DepthModel(
                name="intel-isl/MiDaS:MiDaS_small",
                version=f"torch={torch.__version__}, timm={package_version('timm')}, hub=/root/.cache/torch/hub",
                device=device,
                kind="midas",
                model=model,
                transform=transforms.small_transform,
            )
        except Exception as exc:
            errors.append(f"MiDaS_small torch.hub failed: {type(exc).__name__}: {exc}")
            if model_name == "midas-small":
                raise

    if model_name in ("auto", "depth-anything-v2-small"):
        try:
            os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "5")
            os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "10")
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            hf_id = "depth-anything/Depth-Anything-V2-Small-hf"
            processor = AutoImageProcessor.from_pretrained(hf_id)
            model = AutoModelForDepthEstimation.from_pretrained(hf_id)
            model.to(device).eval()
            return DepthModel(
                name=hf_id,
                version=f"transformers={package_version('transformers')}, torch={torch.__version__}",
                device=device,
                kind="transformers",
                model=model,
                processor=processor,
            )
        except Exception as exc:
            errors.append(f"Depth-Anything-V2-Small-hf failed: {type(exc).__name__}: {exc}")
            if model_name != "auto":
                raise
    raise RuntimeError("Could not load a monocular depth model:\n" + "\n".join(errors))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def finite_float(x: Any) -> Optional[float]:
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def ranksafe_spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size < 32 or b.size < 32:
        return None
    if float(np.nanstd(a)) == 0.0 or float(np.nanstd(b)) == 0.0:
        return None
    val = spearmanr(a, b).statistic
    return finite_float(val)


def silog_rmse(pred: np.ndarray, target: np.ndarray) -> Optional[float]:
    mask = np.isfinite(pred) & np.isfinite(target) & (pred > 0) & (target > 0)
    if int(mask.sum()) < 32:
        return None
    d = np.log(pred[mask]) - np.log(target[mask])
    mse = float(np.mean(d * d) - np.mean(d) ** 2)
    return math.sqrt(max(0.0, mse))


def resize_nearest(arr: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)


def resize_linear(arr: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)


def load_condition(sample_dir: Path, shape_hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    npz_path = sample_dir / "mesh_dense_condition_v0.npz"
    data = np.load(npz_path)
    dense = data["dense"]
    mesh_depth = data["mesh_depth_units"].astype(np.float32)
    mesh_hit = dense[1].astype(np.float32) > 0.5
    player_mask = dense[3].astype(np.float32) > 0.05
    mesh_depth = resize_linear(mesh_depth, shape_hw)
    mesh_hit = resize_nearest(mesh_hit.astype(np.uint8), shape_hw).astype(bool)
    player_mask = resize_nearest(player_mask.astype(np.uint8), shape_hw).astype(bool)
    return mesh_depth, mesh_hit, player_mask


def score_depth_pair(mono: np.ndarray, mesh_depth: np.ndarray, mesh_hit: np.ndarray, player_mask: np.ndarray) -> Dict[str, Any]:
    valid = np.isfinite(mesh_depth) & (mesh_depth > 0) & mesh_hit & (~player_mask)
    n = int(valid.sum())
    if n < 32:
        return {"valid_pixels": n, "spearman": None, "silog_rmse": None, "mono_sign": None}

    mono_v = mono[valid].astype(np.float64)
    mesh_v = mesh_depth[valid].astype(np.float64)
    sp_pos = ranksafe_spearman(mono_v, mesh_v)
    sp_neg = ranksafe_spearman(-mono_v, mesh_v)
    if sp_neg is not None and (sp_pos is None or abs(sp_neg) > abs(sp_pos)):
        mono_v_for_log = -mono_v
        sp = sp_neg
        sign = -1
    else:
        mono_v_for_log = mono_v
        sp = sp_pos
        sign = 1

    shifted = mono_v_for_log - float(np.min(mono_v_for_log)) + 1e-3
    return {
        "valid_pixels": n,
        "spearman": sp,
        "spearman_pos": sp_pos,
        "spearman_neg": sp_neg,
        "silog_rmse": silog_rmse(shifted, mesh_v),
        "mono_sign": sign,
    }


def read_video_frames(path: Path, frame_indices: Sequence[int]) -> Dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    out: Dict[int, np.ndarray] = {}
    for idx in sorted(set(int(x) for x in frame_indices)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        out[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()
    return out


def find_sample_dirs(dense_root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in dense_root.glob("*/samples/*/mesh_dense_condition_v0.npz"):
        out[p.parent.name] = p.parent
    return out


def load_manifest_entries(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    p = Path(report["dense_sequence_manifest"])
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def original_sample_ids(report: Dict[str, Any]) -> List[str]:
    entries = load_manifest_entries(report)
    if entries:
        return [str(x["sample_id"]) for x in entries]
    return list(report.get("sample_ids", []))


def condition_frame_indices(report: Dict[str, Any], n: int) -> List[int]:
    entries = load_manifest_entries(report)
    frames = [int(x.get("gen_frame", i * 4)) for i, x in enumerate(entries[:n])]
    if len(frames) >= n:
        return frames[:n]
    total = int(report.get("frames", 81))
    if n <= 1:
        return [0]
    return [round(i * (total - 1) / (n - 1)) for i in range(n)]


def mp4_for_report(report_path: Path, report: Dict[str, Any]) -> Path:
    p = Path(report.get("mp4", ""))
    if p.exists():
        return p
    p2 = report_path.parents[3] / p
    if p2.exists():
        return p2
    matches = list(report_path.parent.glob(report_path.name.replace("_report.json", "_lf*_seed*.mp4")))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"mp4 not found for {report_path}")


def summarize_values(rows: List[Dict[str, Any]], variant: str, score_kind: str) -> Dict[str, Any]:
    sub = [r for r in rows if r["variant"] == variant and r["score_kind"] == score_kind]
    sp = np.array([r["spearman_median"] for r in sub if r["spearman_median"] is not None], dtype=float)
    si = np.array([r["silog_rmse_median"] for r in sub if r["silog_rmse_median"] is not None], dtype=float)
    return {
        "variant": variant,
        "score_kind": score_kind,
        "windows": len(sub),
        "median_spearman": finite_float(np.median(sp)) if sp.size else None,
        "median_silog_rmse": finite_float(np.median(si)) if si.size else None,
    }


def paired_delta(rows: List[Dict[str, Any]], left: str, right: str, score_kind: str) -> Dict[str, Any]:
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in rows:
        if r["score_kind"] == score_kind and r["variant"] in (left, right):
            by_key[(r["window_id"], r["variant"])] = r
    ds: List[float] = []
    for (wid, var), r in list(by_key.items()):
        if var != left:
            continue
        rr = by_key.get((wid, right))
        if not rr:
            continue
        a = r.get("spearman_median")
        b = rr.get("spearman_median")
        if a is not None and b is not None:
            ds.append(float(a) - float(b))
    arr = np.array(ds, dtype=float)
    return {
        "left_minus_right": f"{left}-{right}",
        "score_kind": score_kind,
        "paired_windows": int(arr.size),
        "median_delta_spearman": finite_float(np.median(arr)) if arr.size else None,
        "mean_delta_spearman": finite_float(np.mean(arr)) if arr.size else None,
    }


def score_sequence(
    depth_model: DepthModel,
    report_path: Path,
    sample_dirs: Dict[str, Path],
    out_root: Path,
    stride: int,
) -> List[Dict[str, Any]]:
    report = read_json(report_path)
    variant = str(report["variant"])
    window_id = str(report["window_id"])
    mp4 = mp4_for_report(report_path, report)
    fed_ids = list(report.get("sample_ids", []))
    true_ids = original_sample_ids(report)
    n = min(len(fed_ids), len(true_ids) if true_ids else len(fed_ids))
    keep = list(range(0, n, stride))
    frame_indices_all = condition_frame_indices(report, n)
    frame_indices = [frame_indices_all[i] for i in keep]
    frames = read_video_frames(mp4, frame_indices)

    score_sets = [("fed_condition", fed_ids)]
    if variant == "shuffled_dense":
        score_sets.append(("true_window_condition", true_ids))

    per_window: List[Dict[str, Any]] = []
    for score_kind, ids in score_sets:
        frame_scores: List[Dict[str, Any]] = []
        for i in keep:
            frame_idx = frame_indices_all[i]
            rgb = frames.get(frame_idx)
            sample_id = ids[i] if i < len(ids) else None
            if rgb is None or sample_id not in sample_dirs:
                frame_scores.append(
                    {
                        "slot_index": i,
                        "video_frame": frame_idx,
                        "sample_id": sample_id,
                        "status": "missing_frame_or_condition",
                    }
                )
                continue
            mono = depth_model.predict(rgb)
            mesh_depth, mesh_hit, player_mask = load_condition(sample_dirs[sample_id], rgb.shape[:2])
            s = score_depth_pair(mono, mesh_depth, mesh_hit, player_mask)
            s.update(
                {
                    "slot_index": i,
                    "video_frame": frame_idx,
                    "sample_id": sample_id,
                    "condition_dir": str(sample_dirs[sample_id]),
                    "status": "ok",
                }
            )
            frame_scores.append(s)

        sp = [x["spearman"] for x in frame_scores if x.get("spearman") is not None]
        si = [x["silog_rmse"] for x in frame_scores if x.get("silog_rmse") is not None]
        obj = {
            "variant": variant,
            "score_kind": score_kind,
            "window_id": window_id,
            "clip_id": report.get("clip_id"),
            "report_path": str(report_path),
            "mp4": str(mp4),
            "shuffled_source": report.get("shuffled_source"),
            "frames_requested": len(keep),
            "frames_scored": len(sp),
            "stride_on_condition_slots": stride,
            "spearman_median": finite_float(np.median(sp)) if sp else None,
            "spearman_mean": finite_float(np.mean(sp)) if sp else None,
            "silog_rmse_median": finite_float(np.median(si)) if si else None,
            "silog_rmse_mean": finite_float(np.mean(si)) if si else None,
            "frame_scores": frame_scores,
        }
        out_path = out_root / "per_window" / variant / f"{window_id}__{score_kind}.json"
        write_json(out_path, obj)
        per_window.append({k: v for k, v in obj.items() if k != "frame_scores"})

    if variant in ("base", "true_dense"):
        gt_scores: List[Dict[str, Any]] = []
        for i in keep:
            sample_id = true_ids[i] if i < len(true_ids) else None
            if sample_id not in sample_dirs:
                continue
            rgb_path = sample_dirs[sample_id] / "target_rgb.png"
            if not rgb_path.exists():
                continue
            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mono = depth_model.predict(rgb)
            mesh_depth, mesh_hit, player_mask = load_condition(sample_dirs[sample_id], rgb.shape[:2])
            s = score_depth_pair(mono, mesh_depth, mesh_hit, player_mask)
            s.update(
                {
                    "slot_index": i,
                    "video_frame": frame_indices_all[i],
                    "sample_id": sample_id,
                    "condition_dir": str(sample_dirs[sample_id]),
                    "status": "ok",
                }
            )
            gt_scores.append(s)
        sp = [x["spearman"] for x in gt_scores if x.get("spearman") is not None]
        si = [x["silog_rmse"] for x in gt_scores if x.get("silog_rmse") is not None]
        obj = {
            "variant": "gt",
            "score_kind": "true_window_condition",
            "window_id": window_id,
            "clip_id": report.get("clip_id"),
            "source_report_path": str(report_path),
            "frames_requested": len(keep),
            "frames_scored": len(sp),
            "stride_on_condition_slots": stride,
            "spearman_median": finite_float(np.median(sp)) if sp else None,
            "spearman_mean": finite_float(np.mean(sp)) if sp else None,
            "silog_rmse_median": finite_float(np.median(si)) if si else None,
            "silog_rmse_mean": finite_float(np.mean(si)) if si else None,
            "frame_scores": gt_scores,
        }
        out_path = out_root / "per_window" / "gt" / f"{window_id}__true_window_condition.json"
        write_json(out_path, obj)
        per_window.append({k: v for k, v in obj.items() if k != "frame_scores"})

    return per_window


def collect_reports(gen_root: Path) -> List[Path]:
    reports: List[Path] = []
    for variant in VARIANTS:
        reports.extend(sorted((gen_root / variant).glob("*_report.json")))
    return reports


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys() if k != "frame_scores"})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdict-root", type=Path, default=Path("output/verdict70h_20260706_v0"))
    ap.add_argument("--out-root", type=Path, default=Path("output/structure_adherence_20260706_v0"))
    ap.add_argument("--stride", type=int, default=3, help="stride over the 21 dense condition slots")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="auto", choices=["auto", "depth-anything-v2-small", "midas-small"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_unix": time.time(),
        "cwd": os.getcwd(),
        "host": platform.node(),
        "python": platform.python_version(),
        "package_versions": {
            "torch": torch.__version__,
            "timm": package_version("timm"),
            "opencv-python-headless": package_version("opencv-python-headless"),
            "decord": package_version("decord"),
            "scipy": package_version("scipy"),
            "pandas": package_version("pandas"),
            "transformers": package_version("transformers"),
        },
        "cuda_available": torch.cuda.is_available(),
        "verdict_root": str(args.verdict_root),
        "out_root": str(args.out_root),
        "stride_on_condition_slots": args.stride,
        "score_policy": {
            "main": "generated/GT monocular depth compared to mesh depth on valid mesh_hit pixels, excluding dense other_player_mask",
            "shuffled": "fed_condition uses report.sample_ids, i.e. the actually fed shuffled dense condition",
            "shuffled_cross": "true_window_condition uses the evaluated window's dense_sequence_manifest",
            "depth_sign": "per frame, choose depth or negative depth by larger absolute Spearman against mesh depth",
        },
    }
    write_json(args.out_root / "run_manifest_initial.json", manifest)

    sample_dirs = find_sample_dirs(args.verdict_root / "dense_backfill_v0")
    reports = collect_reports(args.verdict_root / "gen")
    if args.limit:
        reports = reports[: args.limit]
    depth_model = load_depth_model(args.device, args.model)
    manifest["depth_model"] = {
        "name": depth_model.name,
        "version": depth_model.version,
        "kind": depth_model.kind,
        "device": depth_model.device,
    }
    manifest["sample_dir_count"] = len(sample_dirs)
    manifest["report_count"] = len(reports)
    write_json(args.out_root / "run_manifest.json", manifest)

    rows: List[Dict[str, Any]] = []
    for idx, report_path in enumerate(reports):
        print(f"[{idx+1}/{len(reports)}] {report_path}", flush=True)
        rows.extend(score_sequence(depth_model, report_path, sample_dirs, args.out_root, args.stride))
        write_csv(args.out_root / "per_window_summary_partial.csv", rows)

    write_csv(args.out_root / "per_window_summary.csv", rows)
    summaries: List[Dict[str, Any]] = []
    for variant in ("gt", "base", "true_dense", "shuffled_dense"):
        for score_kind in ("fed_condition", "true_window_condition"):
            s = summarize_values(rows, variant, score_kind)
            if s["windows"]:
                summaries.append(s)
    deltas = [
        paired_delta(rows, "true_dense", "base", "fed_condition"),
        paired_delta(rows, "true_dense", "shuffled_dense", "fed_condition"),
        paired_delta(rows, "gt", "true_dense", "true_window_condition"),
    ]
    result = {
        "manifest": manifest,
        "elapsed_sec": time.time() - t0,
        "summary": summaries,
        "paired_deltas": deltas,
    }
    write_json(args.out_root / "summary.json", result)
    write_csv(args.out_root / "summary.csv", summaries + deltas)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
