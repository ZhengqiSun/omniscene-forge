#!/usr/bin/env python3
"""Build LingBot VAE latent cache rows for materialized light-dust2 clips."""

from __future__ import annotations
from runtime_paths import source_path

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


TEXT_CACHE = Path(str(source_path('assets', 'datasets/fast_cam_dust2_p0_32f_done_cpfs/text_cache/prompt_089a5a60240c0039.pt')))
PROMPT = "first-person gameplay video in Counter-Strike, de_dust2 map"


def add_path(path: Path) -> None:
    text = str(path.resolve())
    if text not in sys.path:
        sys.path.insert(0, text)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous().to(device)
    return tensor


def clip_cache_id(row: dict[str, Any]) -> str:
    sample_dir = row.get("sample_dir") or row.get("clip_dir")
    if sample_dir:
        return Path(str(sample_dir)).name
    return f"{int(row['source_manifest_index']):04d}_{row['game_id']}_{row['episode']}_{row['player_stem']}"


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


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    add_path(args.lingbot_repo)
    from wan.configs import WAN_CONFIGS
    from wan.modules.vae2_1 import Wan2_1_VAE

    cfg = WAN_CONFIGS["i2v-A14B"]
    device = torch.device(args.device)
    vae = Wan2_1_VAE(vae_pth=str(args.ckpt_dir / cfg.vae_checkpoint), device=device)
    rows = list(iter_jsonl(args.source_manifest))
    selected = [row for idx, row in enumerate(rows) if idx % args.num_shards == args.shard_index]
    if args.limit is not None:
        selected = selected[: args.limit]
    args.latents_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(selected):
        cid = clip_cache_id(row)
        latent_path = args.latents_dir / f"{cid}.pt"
        video_path = Path(row["video"])
        image_path = Path(row["image"])
        if args.skip_existing and latent_path.exists():
            obj = torch.load(latent_path, map_location="cpu")
            latent_shape = list(obj["latent"].shape)
            condition_shape = list(obj["condition"].shape)
        else:
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
            torch.save(payload, latent_path)
            latent_shape = payload["latent_shape"]
            condition_shape = payload["condition_shape"]
        manifest_rows.append(
            {
                "clip_id": cid,
                "video": str(video_path),
                "image": str(image_path),
                "poses": str(row["poses"]),
                "intrinsics": str(row["intrinsics"]),
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
        if args.progress_every > 0 and (idx + 1) % args.progress_every == 0:
            print(json.dumps({"event": "cache_progress", "done": idx + 1, "total": len(selected)}, ensure_ascii=False), flush=True)
    write_jsonl(args.out_manifest, manifest_rows)
    report = {
        "kind": "light_dust2_latent_cache_report_v0",
        "source_manifest": str(args.source_manifest),
        "out_manifest": str(args.out_manifest),
        "latents_dir": str(args.latents_dir),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "clips_total_selected": len(rows),
        "clips_this_shard": len(selected),
        "manifest_rows": len(manifest_rows),
        "skip_existing": bool(args.skip_existing),
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-manifest", type=Path, required=True)
    ap.add_argument("--latents-dir", type=Path, required=True)
    ap.add_argument("--out-manifest", type=Path, required=True)
    ap.add_argument("--out-report", type=Path, required=True)
    ap.add_argument("--lingbot-repo", type=Path, default=Path(str(source_path('lingbot', ''))))
    ap.add_argument("--ckpt-dir", type=Path, default=Path(str(source_path('assets', 'lingbot-world-base-cam'))))
    ap.add_argument("--text-cache", type=Path, default=TEXT_CACHE)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--video-width", type=int, default=832)
    ap.add_argument("--video-height", type=int, default=480)
    ap.add_argument("--video-frames", type=int, default=81)
    ap.add_argument("--latent-frames", type=int, default=21)
    ap.add_argument("--chunk-size", type=int, default=3)
    ap.add_argument("--progress-every", type=int, default=25)
    args = ap.parse_args()
    report = build_cache(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
