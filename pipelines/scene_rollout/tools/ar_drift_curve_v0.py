#!/usr/bin/env python3
"""Quantify per-window AR drift against GT and teacher-context stitch outputs."""

from __future__ import annotations
from runtime_paths import source_path

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


WINDOW_RE = re.compile(r"_(\d{7})_true_dense")


@dataclass(frozen=True)
class WindowInput:
    window_id: str
    raw_start: int
    ar_mp4: Path
    teacher_mp4: Path
    manifest: Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(source_path('scene', '')))
    ap.add_argument(
        "--ar-dir",
        default="output/demo_1min_ar_20260706_v0/gen_true_dense_ar/out",
    )
    ap.add_argument(
        "--teacher-dir",
        default="output/demo_1min_20260706_v0/gen_true_dense/out",
    )
    ap.add_argument(
        "--out-dir",
        default="output/ar_drift_curve_20260706_v0",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lpips-batch", type=int, default=16)
    return ap.parse_args()


def stat_info(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path),
        "size": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
    }


def raw_start_from_name(path: Path) -> int:
    m = WINDOW_RE.search(path.name)
    if not m:
        raise ValueError(f"cannot parse raw start from {path.name}")
    return int(m.group(1))


def collect_windows(root: Path, ar_dir: Path, teacher_dir: Path) -> list[WindowInput]:
    ar_mp4s = sorted(ar_dir.glob("*_true_dense_ar_lf21_seed*.mp4"), key=raw_start_from_name)
    teacher_by_start = {raw_start_from_name(p): p for p in teacher_dir.glob("*_true_dense_lf21_seed*.mp4")}
    windows: list[WindowInput] = []
    for ar_mp4 in ar_mp4s:
        raw_start = raw_start_from_name(ar_mp4)
        teacher_mp4 = teacher_by_start.get(raw_start)
        if teacher_mp4 is None:
            raise FileNotFoundError(f"missing teacher mp4 for raw_start={raw_start}")
        report = ar_mp4.with_name(ar_mp4.name.replace("_true_dense_ar_lf21_seed20260629.mp4", "_true_dense_ar_report.json"))
        with report.open() as f:
            rep = json.load(f)
        manifest = Path(rep["dense_sequence_manifest"])
        if not manifest.is_absolute():
            manifest = root / manifest
        windows.append(
            WindowInput(
                window_id=rep["window_id"],
                raw_start=raw_start,
                ar_mp4=ar_mp4,
                teacher_mp4=teacher_mp4,
                manifest=manifest,
            )
        )
    return windows


def read_manifest_gt(root: Path, manifest: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest.open() as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            dense_path = Path(item["dense_path"])
            if not dense_path.is_absolute():
                dense_path = root / dense_path
            gt = dense_path.with_name("target_rgb.png")
            if not gt.exists():
                raise FileNotFoundError(f"missing GT target_rgb.png for {dense_path}")
            rows.append(
                {
                    "index": item["index"],
                    "gen_frame": int(item["gen_frame"]),
                    "raw_frame": int(item["raw_frame"]),
                    "sample_id": item["sample_id"],
                    "gt_path": gt,
                }
            )
    return rows


def read_video_frames(path: Path, frame_indices: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    out: dict[int, np.ndarray] = {}
    wanted = set(frame_indices)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for idx in range(total):
        ok, bgr = cap.read()
        if not ok:
            break
        if idx in wanted:
            out[idx] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cap.release()
    missing = sorted(wanted - set(out))
    if missing:
        raise RuntimeError(f"{path} missing frames {missing}")
    return out


def load_gt(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def resize_to_gt(rgb: np.ndarray, gt: np.ndarray) -> np.ndarray:
    h, w = gt.shape[:2]
    if rgb.shape[:2] == (h, w):
        return rgb
    return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = float(np.mean(diff * diff))
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel = (11, 11)
    sigma = 1.5
    mu_a = cv2.GaussianBlur(ga, kernel, sigma)
    mu_b = cv2.GaussianBlur(gb, kernel, sigma)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sig_a2 = cv2.GaussianBlur(ga * ga, kernel, sigma) - mu_a2
    sig_b2 = cv2.GaussianBlur(gb * gb, kernel, sigma) - mu_b2
    sig_ab = cv2.GaussianBlur(ga * gb, kernel, sigma) - mu_ab
    num = (2 * mu_ab + c1) * (2 * sig_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sig_a2 + sig_b2 + c2)
    return float(np.mean(num / (den + 1e-12)))


def lap_var(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tensor_batch(images: list[np.ndarray], device: torch.device) -> torch.Tensor:
    arr = np.stack(images).astype(np.float32) / 127.5 - 1.0
    arr = np.transpose(arr, (0, 3, 1, 2))
    return torch.from_numpy(arr).to(device)


def lpips_scores(model: Any, pairs: list[tuple[np.ndarray, np.ndarray]], device: torch.device, batch: int) -> list[float]:
    vals: list[float] = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch):
            chunk = pairs[i : i + batch]
            aa = tensor_batch([p[0] for p in chunk], device)
            bb = tensor_batch([p[1] for p in chunk], device)
            y = model(aa, bb).view(-1).detach().cpu().numpy()
            vals.extend(float(x) for x in y)
    return vals


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def std(xs: list[float]) -> float:
    return float(np.std(xs, ddof=0)) if xs else float("nan")


def main() -> None:
    args = parse_args()
    root = Path(args.project_root)
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import lpips

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    ar_dir = root / args.ar_dir
    teacher_dir = root / args.teacher_dir
    windows = collect_windows(root, ar_dir, teacher_dir)
    results: list[dict[str, Any]] = []

    for wi, w in enumerate(windows, start=1):
        gt_rows = read_manifest_gt(root, w.manifest)
        frame_ids = [r["gen_frame"] for r in gt_rows]
        ar_frames = read_video_frames(w.ar_mp4, frame_ids)
        teacher_frames = read_video_frames(w.teacher_mp4, frame_ids)

        ar_gt_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        teacher_gt_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        ar_teacher_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        ar_laps: list[float] = []
        teacher_laps: list[float] = []
        frame_rows: list[dict[str, Any]] = []

        for row in gt_rows:
            gt = load_gt(row["gt_path"])
            ar = resize_to_gt(ar_frames[row["gen_frame"]], gt)
            teacher = resize_to_gt(teacher_frames[row["gen_frame"]], gt)
            ar_gt_pairs.append((ar, gt))
            teacher_gt_pairs.append((teacher, gt))
            ar_teacher_pairs.append((ar, teacher))
            ar_laps.append(lap_var(ar))
            teacher_laps.append(lap_var(teacher))
            frame_rows.append(
                {
                    "manifest_index": row["index"],
                    "gen_frame": row["gen_frame"],
                    "raw_frame": row["raw_frame"],
                    "sample_id": row["sample_id"],
                    "gt_path": str(row["gt_path"]),
                    "ar_psnr_gt": psnr(ar, gt),
                    "teacher_psnr_gt": psnr(teacher, gt),
                    "ar_psnr_teacher": psnr(ar, teacher),
                    "ar_ssim_gt": ssim_gray(ar, gt),
                    "teacher_ssim_gt": ssim_gray(teacher, gt),
                    "ar_ssim_teacher": ssim_gray(ar, teacher),
                    "ar_laplacian_var": ar_laps[-1],
                    "teacher_laplacian_var": teacher_laps[-1],
                }
            )

        ar_lp_gt = lpips_scores(lpips_model, ar_gt_pairs, device, args.lpips_batch)
        teacher_lp_gt = lpips_scores(lpips_model, teacher_gt_pairs, device, args.lpips_batch)
        ar_lp_teacher = lpips_scores(lpips_model, ar_teacher_pairs, device, args.lpips_batch)
        for fr, a, t, at in zip(frame_rows, ar_lp_gt, teacher_lp_gt, ar_lp_teacher):
            fr["ar_lpips_gt"] = a
            fr["teacher_lpips_gt"] = t
            fr["ar_lpips_teacher"] = at

        ar_psnr = [fr["ar_psnr_gt"] for fr in frame_rows]
        teacher_psnr = [fr["teacher_psnr_gt"] for fr in frame_rows]
        ar_ssim = [fr["ar_ssim_gt"] for fr in frame_rows]
        teacher_ssim = [fr["teacher_ssim_gt"] for fr in frame_rows]
        result = {
            "window_index": wi,
            "window_id": w.window_id,
            "raw_start": w.raw_start,
            "inputs": {
                "ar_mp4": stat_info(w.ar_mp4),
                "teacher_mp4": stat_info(w.teacher_mp4),
                "manifest": stat_info(w.manifest),
            },
            "num_aligned_gt_frames": len(frame_rows),
            "alignment_note": "metrics use manifest gen_frame indices 0,4,...,80 aligned to target_rgb.png samples",
            "metrics": {
                "ar_vs_gt": {
                    "psnr": mean(ar_psnr),
                    "ssim": mean(ar_ssim),
                    "lpips": mean(ar_lp_gt),
                },
                "teacher_vs_gt": {
                    "psnr": mean(teacher_psnr),
                    "ssim": mean(teacher_ssim),
                    "lpips": mean(teacher_lp_gt),
                },
                "ar_vs_teacher": {
                    "psnr": mean([fr["ar_psnr_teacher"] for fr in frame_rows]),
                    "ssim": mean([fr["ar_ssim_teacher"] for fr in frame_rows]),
                    "lpips": mean(ar_lp_teacher),
                },
                "laplacian_var": {
                    "ar": mean(ar_laps),
                    "teacher": mean(teacher_laps),
                },
                "loss_ar_minus_teacher": {
                    "psnr_drop": mean(teacher_psnr) - mean(ar_psnr),
                    "ssim_drop": mean(teacher_ssim) - mean(ar_ssim),
                    "lpips_increase": mean(ar_lp_gt) - mean(teacher_lp_gt),
                },
            },
            "frames": frame_rows,
        }
        print(
            f"window {wi:02d} raw={w.raw_start}: "
            f"AR PSNR {result['metrics']['ar_vs_gt']['psnr']:.3f}, "
            f"teacher PSNR {result['metrics']['teacher_vs_gt']['psnr']:.3f}, "
            f"LPIPS loss {result['metrics']['loss_ar_minus_teacher']['lpips_increase']:.4f}",
            flush=True,
        )
        results.append(result)

    teacher_lpips = [r["metrics"]["teacher_vs_gt"]["lpips"] for r in results]
    teacher_lpips_sigma = std(teacher_lpips)
    lpips_losses = [r["metrics"]["loss_ar_minus_teacher"]["lpips_increase"] for r in results]
    lpips_threshold = 2.0 * teacher_lpips_sigma
    first_bad = None
    for r, loss in zip(results, lpips_losses):
        if loss > lpips_threshold:
            first_bad = r["window_index"]
            break
    usable_horizon = (first_bad - 1) if first_bad is not None else len(results)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(root),
        "device": str(device),
        "num_windows": len(results),
        "horizon_rule": "first window where AR LPIPS_vs_GT minus teacher LPIPS_vs_GT exceeds 2 sigma of teacher LPIPS_vs_GT across windows",
        "teacher_lpips_sigma": teacher_lpips_sigma,
        "lpips_loss_threshold_2sigma": lpips_threshold,
        "first_window_exceeding_threshold": first_bad,
        "usable_horizon_windows": usable_horizon,
        "windows": results,
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    xs = [r["window_index"] for r in results]
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ax = axs[0, 0]
    ax.plot(xs, [r["metrics"]["ar_vs_gt"]["psnr"] for r in results], "o-", label="AR vs GT")
    ax.plot(xs, [r["metrics"]["teacher_vs_gt"]["psnr"] for r in results], "o-", label="Stitched vs GT")
    ax.set_title("PSNR higher is better")
    ax.set_xlabel("Window")
    ax.set_ylabel("dB")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axs[0, 1]
    ax.plot(xs, [r["metrics"]["ar_vs_gt"]["ssim"] for r in results], "o-", label="AR vs GT")
    ax.plot(xs, [r["metrics"]["teacher_vs_gt"]["ssim"] for r in results], "o-", label="Stitched vs GT")
    ax.set_title("SSIM higher is better")
    ax.set_xlabel("Window")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axs[1, 0]
    ax.plot(xs, [r["metrics"]["ar_vs_gt"]["lpips"] for r in results], "o-", label="AR vs GT")
    ax.plot(xs, [r["metrics"]["teacher_vs_gt"]["lpips"] for r in results], "o-", label="Stitched vs GT")
    ax.plot(xs, lpips_losses, "o--", label="AR-teacher LPIPS loss")
    ax.axhline(lpips_threshold, color="red", linestyle=":", label="2 sigma threshold")
    if first_bad is not None:
        ax.axvline(first_bad, color="red", linestyle=":", alpha=0.7)
    ax.set_title("LPIPS lower is better")
    ax.set_xlabel("Window")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axs[1, 1]
    ax.plot(xs, [r["metrics"]["laplacian_var"]["ar"] for r in results], "o-", label="AR")
    ax.plot(xs, [r["metrics"]["laplacian_var"]["teacher"] for r in results], "o-", label="Stitched")
    ax.set_title("Laplacian variance sharpness")
    ax.set_xlabel("Window")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.suptitle(f"AR drift curve, usable horizon={usable_horizon} windows")
    png_path = out_dir / "ar_drift_curve_v0.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    conclusion = (
        f"AR LPIPS loss first exceeds 2sigma teacher fluctuation at window {first_bad}; "
        f"usable horizon = {usable_horizon} windows."
        if first_bad is not None
        else f"AR LPIPS loss never exceeds 2sigma teacher fluctuation; usable horizon = {usable_horizon} windows."
    )
    (out_dir / "summary.txt").write_text(
        "\n".join(
            [
                conclusion,
                f"teacher_lpips_sigma={teacher_lpips_sigma:.6f}",
                f"lpips_loss_threshold_2sigma={lpips_threshold:.6f}",
                f"metrics_json={metrics_path}",
                f"curve_png={png_path}",
            ]
        )
        + "\n"
    )
    print(conclusion)
    print(f"wrote {metrics_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
