#!/usr/bin/env python3
"""Build LingBot latent cache for tier69h materialized clips on DLC.

This is a DLC-safe copy of cache_light_dust2_pilot_latents_v0.py:
- maps /mnt/workspace paths to /mnt/data/pku inside the container
- writes only into the requested output directory
- appends JSONL progress as each clip completes
- skips existing complete .pt files idempotently
"""

from __future__ import annotations
from runtime_paths import source_path

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


PROMPT = "first-person gameplay video in Counter-Strike, de_dust2 map"
TEXT_CACHE = Path(str(source_path('assets', 'datasets/fast_cam_dust2_p0_32f_done_cpfs/text_cache/prompt_089a5a60240c0039.pt')))
MIN_FREE_BYTES = 800 * 1024**3


def add_path(path: Path) -> None:
    text = str(path.resolve())
    if text not in sys.path:
        sys.path.insert(0, text)


def map_path(value: str | Path) -> Path:
    text = str(value)
    if text.startswith("/mnt/workspace/"):
        text = "/mnt/data/pku/" + text[len("/mnt/workspace/") :]
    return Path(text)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_video_tensor(path: Path, *, frames: int, width: int, height: int, device: torch.device) -> torch.Tensor:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    out: list[np.ndarray] = []
    for _ in range(frames):
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"{path}: expected {frames} frames, got {len(out)}")
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    arr = np.stack(out, axis=0).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous().to(device)


def clip_cache_id(row: dict[str, Any]) -> str:
    sample_dir = row.get("sample_dir") or row.get("clip_dir")
    if sample_dir:
        return Path(str(sample_dir)).name
    return str(row["clip_id"])


def valid_latent_file(path: Path) -> tuple[bool, list[int] | None, list[int] | None]:
    if not path.exists() or path.stat().st_size < 1024 * 1024:
        return False, None, None
    try:
        obj = torch.load(path, map_location="cpu")
        if "latent" not in obj or "condition" not in obj:
            return False, None, None
        return True, list(obj["latent"].shape), list(obj["condition"].shape)
    except Exception:
        return False, None, None


def build_condition(vae: Any, image_path: Path, *, frames: int, height: int, width: int, chunk_size: int, device: torch.device) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    img_t = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device)
    lat_h = height // 8
    lat_w = width // 8
    lat_f = (frames - 1) // 4 + 1
    lat_f = int(lat_f - (lat_f % chunk_size))
    effective_frames = (lat_f - 1) * 4 + 1
    msk = torch.ones(1, effective_frames, lat_h, lat_w, device=device)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]
    y = vae.encode(
        [
            torch.concat(
                [
                    torch.nn.functional.interpolate(img_t[None].cpu(), size=(height, width), mode="bicubic").transpose(0, 1),
                    torch.zeros(3, effective_frames - 1, height, width),
                ],
                dim=1,
            ).to(device)
        ]
    )[0]
    return torch.concat([msk, y]).to(torch.bfloat16)


def manifest_row(row: dict[str, Any], cid: str, latent_path: Path, latent_shape: list[int], condition_shape: list[int], args: argparse.Namespace) -> dict[str, Any]:
    clip_dir = map_path(row.get("clip_dir") or row.get("sample_dir"))
    out = dict(row)
    out.update(
        {
            "clip_id": cid,
            "video": str(map_path(row.get("video") or clip_dir / "video.mp4")),
            "image": str(map_path(row.get("image") or clip_dir / "image.jpg")),
            "poses": str(map_path(row.get("poses") or clip_dir / "poses.npy")),
            "intrinsics": str(map_path(row.get("intrinsics") or clip_dir / "intrinsics.npy")),
            "latent_frames_expected": args.latent_frames,
            "latent_cache": str(latent_path),
            "dtype": "torch.bfloat16",
            "shape": latent_shape,
            "condition_shape": condition_shape,
            "prompt": row.get("prompt", PROMPT),
            "text_cache": str(args.text_cache),
            "text_shape": [16, 4096],
            "text_dtype": "torch.bfloat16",
        }
    )
    return out


def check_free_space(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"free space below 0.8T guard: {free} bytes at {path}")
    return free


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    add_path(args.lingbot_repo)
    from wan.configs import WAN_CONFIGS
    from wan.modules.vae2_1 import Wan2_1_VAE

    args.latents_dir.mkdir(parents=True, exist_ok=True)
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.progress_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.reset_manifest and args.out_manifest.exists():
        args.out_manifest.unlink()

    rows = list(iter_jsonl(args.source_manifest))
    selected = [row for idx, row in enumerate(rows) if idx % args.num_shards == args.shard_index]
    if args.limit is not None:
        selected = selected[: args.limit]

    device = torch.device(args.device)
    cfg = WAN_CONFIGS["i2v-A14B"]
    vae = Wan2_1_VAE(vae_pth=str(args.ckpt_dir / cfg.vae_checkpoint), device=device)

    started = time.time()
    done = skipped = failed = 0
    first_error = None
    for idx, row in enumerate(selected):
        cid = clip_cache_id(row)
        latent_path = args.latents_dir / f"{cid}.pt"
        t0 = time.time()
        try:
            free_before = check_free_space(args.latents_dir)
            ok, latent_shape, condition_shape = valid_latent_file(latent_path) if args.skip_existing else (False, None, None)
            if ok:
                skipped += 1
            else:
                clip_dir = map_path(row.get("clip_dir") or row.get("sample_dir"))
                video_path = map_path(row.get("video") or clip_dir / "video.mp4")
                image_path = map_path(row.get("image") or clip_dir / "image.jpg")
                video = load_video_tensor(
                    video_path,
                    frames=args.video_frames,
                    width=args.video_width,
                    height=args.video_height,
                    device=device,
                )
                with torch.no_grad():
                    latent = vae.encode([video])[0].to(torch.bfloat16)
                    condition = build_condition(
                        vae,
                        image_path,
                        frames=args.video_frames,
                        height=args.video_height,
                        width=args.video_width,
                        chunk_size=args.chunk_size,
                        device=device,
                    )
                payload = {
                    "clip_id": cid,
                    "latent": latent.cpu(),
                    "condition": condition.cpu(),
                    "latent_shape": list(latent.shape),
                    "condition_shape": list(condition.shape),
                    "latent_dtype": str(latent.dtype),
                    "condition_dtype": str(condition.dtype),
                }
                tmp = latent_path.with_suffix(".pt.tmp")
                torch.save(payload, tmp)
                tmp.replace(latent_path)
                latent_shape = payload["latent_shape"]
                condition_shape = payload["condition_shape"]
            done += 1
            out_row = manifest_row(row, cid, latent_path, latent_shape or [], condition_shape or [], args)
            append_jsonl(args.out_manifest, out_row)
            append_jsonl(
                args.progress_jsonl,
                {
                    "event": "clip_done",
                    "clip_id": cid,
                    "index_in_shard": idx,
                    "done": done,
                    "total": len(selected),
                    "skipped_existing": bool(ok),
                    "latent_cache": str(latent_path),
                    "latent_bytes": latent_path.stat().st_size,
                    "seconds": round(time.time() - t0, 3),
                    "free_bytes_before": free_before,
                    "free_bytes_after": shutil.disk_usage(args.latents_dir).free,
                },
            )
            if args.progress_every > 0 and done % args.progress_every == 0:
                print(json.dumps({"event": "cache_progress", "done": done, "total": len(selected), "skipped": skipped}, ensure_ascii=False), flush=True)
        except Exception as exc:
            failed += 1
            first_error = first_error or repr(exc)
            append_jsonl(args.progress_jsonl, {"event": "clip_failed", "clip_id": cid, "index_in_shard": idx, "error": repr(exc)})
            if not args.continue_on_error:
                raise

    report = {
        "kind": "tier69h_latent_cache_dlc_report_v0",
        "source_manifest": str(args.source_manifest),
        "out_manifest": str(args.out_manifest),
        "progress_jsonl": str(args.progress_jsonl),
        "latents_dir": str(args.latents_dir),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "source_rows": len(rows),
        "clips_this_shard": len(selected),
        "done": done,
        "skipped_existing": skipped,
        "failed": failed,
        "first_error": first_error,
        "elapsed_seconds": round(time.time() - started, 3),
        "free_bytes_after": shutil.disk_usage(args.latents_dir).free,
    }
    write_json(args.out_report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-manifest", type=Path, required=True)
    ap.add_argument("--latents-dir", type=Path, required=True)
    ap.add_argument("--out-manifest", type=Path, required=True)
    ap.add_argument("--out-report", type=Path, required=True)
    ap.add_argument("--progress-jsonl", type=Path, required=True)
    ap.add_argument("--lingbot-repo", type=Path, default=Path(str(source_path('lingbot', ''))))
    ap.add_argument("--ckpt-dir", type=Path, default=Path(str(source_path('assets', 'lingbot-world-base-cam'))))
    ap.add_argument("--text-cache", type=Path, default=TEXT_CACHE)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--reset-manifest", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--video-width", type=int, default=832)
    ap.add_argument("--video-height", type=int, default=480)
    ap.add_argument("--video-frames", type=int, default=81)
    ap.add_argument("--latent-frames", type=int, default=21)
    ap.add_argument("--chunk-size", type=int, default=3)
    ap.add_argument("--progress-every", type=int, default=25)
    args = ap.parse_args()
    build_cache(args)


if __name__ == "__main__":
    main()
