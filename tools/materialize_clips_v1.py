#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# materialize_clips_v1.py — v2 scale-up 入库副本
#
# 来源路径:
#   /mnt/workspace/zhengqi/multiview-map-v0/output/memory_v2_all188_diversity_stride_manifest_20260629_v0/materialize_tier69h_supplement_cpuio_v0_w32/materialize_tier69h_supplement_cpuio_v0.py
# 源文件 sha256:
#   72831399bb72d125499ef93c7b5a5c4b7a73b9970ba06a384408e2acbb3a1b5d
# 移植日期: 2026-07-23
#
# 背景: 源脚本原先只存在于 zhengqi output 运行目录内、不在任何版本控制
# （审计判定"二次失传风险"），本文件为其原样入库副本。
# 相对源文件的最小改动（其余逻辑一行不动）:
#   1) 增加本注释块（来源路径 + 源文件 sha256 + 移植日期）;
#   2) report 的 kind 字符串中硬编码的 '_w32' 后缀改为 f'_w{args.workers}'
#      （审计发现的记账瑕疵）;
#   3) 审计项"硬编码输出路径默认值指向 zhengqi 树": 检查后确认
#      --out-dir / --report-dir 本就为 required=True 且无默认值，无需改动。
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REQUIRED = ["video.mp4", "image.jpg", "poses.npy", "intrinsics.npy", "meta.json", "prompt.txt"]
BASE_R = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)
INTRINSIC_ROW = np.array([312.00116, 320.0012, 416.0, 240.0], dtype=np.float32)
ACTION_CACHE: dict[str, dict[int, dict[str, Any]]] = {}


def safe_name(row: dict[str, Any], index: int) -> str:
    return f"{index:06d}_{row['game_id']}_{row['episode']}_{row['player_stem']}_{int(row['raw_start']):07d}"


def load_rows(path: Path, start_index: int, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start_index or not line.strip():
                continue
            row = json.loads(line)
            row["_source_line_index"] = i
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def clip_dir_for(row: dict[str, Any], out_dir: Path) -> Path:
    idx = int(row.get("source_manifest_index", row.get("_source_line_index")))
    return out_dir / safe_name(row, idx)


def complete(cdir: Path) -> bool:
    if not all((cdir / name).exists() and (cdir / name).stat().st_size > 0 for name in REQUIRED):
        return False
    try:
        poses = np.load(cdir / "poses.npy")
        intr = np.load(cdir / "intrinsics.npy")
        cap = cv2.VideoCapture(str(cdir / "video.mp4"))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        return poses.shape == (81, 4, 4) and intr.shape == (81, 4) and frames == 81 and width == 832 and height == 480 and abs(fps - 16.0) < 0.05
    except Exception:
        return False


def index_actions(action_json: str) -> dict[int, dict[str, Any]]:
    cached = ACTION_CACHE.get(action_json)
    if cached is not None:
        return cached
    data = json.load(open(action_json, "r", encoding="utf-8"))
    indexed = {int(r.get("frame_count")): r for r in data if "frame_count" in r}
    ACTION_CACHE[action_json] = indexed
    return indexed


def pose_from_action(rec: dict[str, Any]) -> np.ndarray:
    pos = rec.get("camera_position") or [rec.get("x", 0.0), rec.get("y", 0.0), float(rec.get("z", 0.0)) + 64.0]
    rot = rec.get("camera_rotation") or [0.0, rec.get("pitch", 0.0), rec.get("yaw", 0.0)]
    pitch = math.radians(float(rot[1] if len(rot) > 1 else 0.0))
    yaw = math.radians(float(rot[2] if len(rot) > 2 else rec.get("yaw", 0.0)))
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rz @ BASE_R @ rx
    mat[:3, 3] = np.asarray(pos, dtype=np.float32)
    return mat


def write_video_and_image(row: dict[str, Any], cdir: Path, width: int, height: int, fps: float) -> None:
    indices = [int(x) for x in row["raw_indices"][:81]]
    cap = cv2.VideoCapture(row["mp4"])
    if not cap.isOpened():
        raise RuntimeError(f"failed to open mp4: {row['mp4']}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_video = cdir / "video.mp4.tmp.mp4"
    vw = cv2.VideoWriter(str(tmp_video), fourcc, fps, (width, height))
    if not vw.isOpened():
        cap.release()
        raise RuntimeError(f"failed to open VideoWriter: {tmp_video}")
    frames: dict[int, np.ndarray] = {}
    first_idx, last_idx = indices[0], indices[-1]
    wanted = set(indices)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_idx)
    cur = first_idx
    while cur <= last_idx:
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            vw.release()
            raise RuntimeError(f"failed to read frame {cur} from {row['mp4']}")
        if cur in wanted:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            frames[cur] = frame
            vw.write(frame)
        cur += 1
    cap.release()
    vw.release()
    missing = [i for i in indices if i not in frames]
    if missing:
        raise RuntimeError(f"missing decoded frames: {missing[:5]}")
    os.replace(tmp_video, cdir / "video.mp4")
    tmp_img = cdir / "image.jpg.tmp.jpg"
    if not cv2.imwrite(str(tmp_img), frames[indices[0]], [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
        raise RuntimeError(f"failed to write image: {tmp_img}")
    os.replace(tmp_img, cdir / "image.jpg")


def materialize_one(args_tuple: tuple[dict[str, Any], str, int, int, float]) -> dict[str, Any]:
    row, out_dir_s, width, height, fps = args_tuple
    out_dir = Path(out_dir_s)
    idx = int(row.get("source_manifest_index", row.get("_source_line_index")))
    cdir = clip_dir_for(row, out_dir)
    t0 = time.perf_counter()
    if complete(cdir):
        return {"index": idx, "clip_id": row["clip_id"], "ok": True, "reused": True, "elapsed_s": 0.0, "bytes": sum((cdir / n).stat().st_size for n in REQUIRED)}
    cdir.mkdir(parents=True, exist_ok=True)
    try:
        write_video_and_image(row, cdir, width, height, fps)
        actions = index_actions(row["action_json"])
        poses = []
        for frame_idx in row["raw_indices"][:81]:
            rec = actions.get(int(frame_idx))
            if rec is None:
                raise RuntimeError(f"missing action frame_count {frame_idx} in {row['action_json']}")
            poses.append(pose_from_action(rec))
        np.save(cdir / "poses.npy", np.stack(poses).astype(np.float32))
        np.save(cdir / "intrinsics.npy", np.repeat(INTRINSIC_ROW[None, :], 81, axis=0).astype(np.float32))
        (cdir / "prompt.txt").write_text(str(row.get("prompt", "")), encoding="utf-8")
        meta = dict(row)
        meta.pop("_source_line_index", None)
        meta.update({
            "sample_dir": str(cdir.resolve()),
            "clip_dir": str(cdir.resolve()),
            "video": str((cdir / "video.mp4").resolve()),
            "image": str((cdir / "image.jpg").resolve()),
            "poses": str((cdir / "poses.npy").resolve()),
            "intrinsics": str((cdir / "intrinsics.npy").resolve()),
            "meta_json": str((cdir / "meta.json").resolve()),
            "prompt_txt": str((cdir / "prompt.txt").resolve()),
            "materialize_policy": "raw RGB frames direct-resized 1280x720 -> 832x480, matching pilot1 sanity check",
        })
        tmp_meta = cdir / "meta.json.tmp"
        tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_meta, cdir / "meta.json")
        if not complete(cdir):
            raise RuntimeError("clip incomplete after write")
        return {"index": idx, "clip_id": row["clip_id"], "ok": True, "reused": False, "elapsed_s": time.perf_counter() - t0, "bytes": sum((cdir / n).stat().st_size for n in REQUIRED)}
    except Exception as exc:
        return {"index": idx, "clip_id": row.get("clip_id"), "ok": False, "reused": False, "elapsed_s": time.perf_counter() - t0, "error": repr(exc), "traceback": traceback.format_exc()[-4000:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--report-dir", type=Path, required=True)
    ap.add_argument("--start-index", type=int, default=14222)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=16.0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    command = {"argv": os.sys.argv, "cwd": str(Path.cwd()), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    (args.report_dir / f"command_{os.getpid()}.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    rows = load_rows(args.source_manifest, args.start_index, args.limit)
    already = sum(1 for r in rows if complete(clip_dir_for(r, args.out_dir)))
    progress_path = args.report_dir / "progress.jsonl"
    failures = []
    done = 0
    t_all = time.perf_counter()
    with progress_path.open("a", encoding="utf-8") as pf, cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(materialize_one, (r, str(args.out_dir), args.width, args.height, args.fps)) for r in rows]
        for fut in cf.as_completed(futs):
            rec = fut.result()
            done += 1
            if not rec.get("ok"):
                failures.append(rec)
            pf.write(json.dumps({"event": "materialize_done", "done": done, "total": len(rows), **rec}, ensure_ascii=False) + "\n")
            pf.flush()
            if done % 100 == 0 or done == len(rows):
                print(json.dumps({"event": "progress", "done": done, "total": len(rows), "failures": len(failures), "elapsed_s": time.perf_counter() - t_all}, ensure_ascii=False), flush=True)
    elapsed = time.perf_counter() - t_all
    clip_dirs = [p for p in args.out_dir.iterdir() if p.is_dir()]
    file_counts = {name: 0 for name in REQUIRED}
    total_bytes = 0
    missing = []
    for r in rows:
        cdir = clip_dir_for(r, args.out_dir)
        miss = [name for name in REQUIRED if not ((cdir / name).exists() and (cdir / name).stat().st_size > 0)]
        if miss or not complete(cdir):
            missing.append({"index": int(r.get("source_manifest_index", r.get("_source_line_index"))), "clip_id": r.get("clip_id"), "missing": miss, "clip_dir": str(cdir)})
        for name in REQUIRED:
            p = cdir / name
            if p.exists() and p.stat().st_size > 0:
                file_counts[name] += 1
                total_bytes += p.stat().st_size
    report = {
        "kind": f"memory_v2_all188_tier69h_supplement_clip_materialize_report_v0_w{args.workers}",
        "source_manifest": str(args.source_manifest.resolve()),
        "start_index": args.start_index,
        "rows": len(rows),
        "workers": args.workers,
        "video_width": args.width,
        "video_height": args.height,
        "fps": args.fps,
        "already_complete_at_start": already,
        "elapsed_s": elapsed,
        "clips_per_s": len(rows) / max(elapsed, 1e-9),
        "remaining_clips_per_s": max(0, len(rows) - already) / max(elapsed, 1e-9),
        "clip_dirs": len(clip_dirs),
        "required_file_counts": file_counts,
        "missing_count": len(missing),
        "missing_examples": missing[:20],
        "failure_count": len(failures),
        "failure_examples": failures[:20],
        "total_output_bytes_required_files": total_bytes,
        "total_output_gib_required_files": total_bytes / (1024 ** 3),
        "progress_jsonl": str(progress_path.resolve()),
        "out_dir": str(args.out_dir.resolve()),
    }
    (args.report_dir / "materialize_report_v0.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
