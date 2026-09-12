#!/usr/bin/env python3
"""Compare Map Memory geometry backends on the same manifest samples."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


def import_tool(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def mae(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float | None:
    if not np.any(valid):
        return None
    return float(np.abs(a[valid] - b[valid]).mean())


def gray(ch: np.ndarray) -> Image.Image:
    arr = np.nan_to_num(ch, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8)).convert("RGB")


def diff_img(rgb: np.ndarray, base: np.ndarray, test: np.ndarray) -> Image.Image:
    out = rgb.copy().astype(np.float32) * 0.42
    both = base & test
    base_only = base & ~test
    test_only = test & ~base
    out[both] = np.asarray([40, 220, 90], dtype=np.float32)
    out[base_only] = np.asarray([255, 70, 70], dtype=np.float32)
    out[test_only] = np.asarray([70, 140, 255], dtype=np.float32)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def make_qa(path: Path, rgb: np.ndarray, obj_dense: np.ndarray, bsp_dense: np.ndarray, teacher_depth: np.ndarray | None) -> None:
    h, w = obj_dense.shape[1:]
    rgb_small = np.asarray(Image.fromarray(rgb).resize((w, h), Image.Resampling.BILINEAR))
    panels: list[tuple[str, Image.Image]] = [
        ("rgb", Image.fromarray(rgb_small)),
        ("obj depth", gray(obj_dense[0])),
        ("bsp depth", gray(bsp_dense[0])),
        ("obj hit", gray(obj_dense[1])),
        ("bsp hit", gray(bsp_dense[1])),
        ("hit diff", diff_img(rgb_small, obj_dense[1] > 0.5, bsp_dense[1] > 0.5)),
        ("obj player", gray(obj_dense[3])),
        ("bsp player", gray(bsp_dense[3])),
    ]
    if teacher_depth is not None:
        panels.append(("teacher depth", gray(teacher_depth)))
    pad = 26
    cols = 3
    rows = int(np.ceil(len(panels) / cols))
    canvas = Image.new("RGB", (cols * w, rows * (h + pad)), (250, 250, 247))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, img) in enumerate(panels):
        x = (idx % cols) * w
        y = (idx // cols) * (h + pad)
        canvas.paste(img, (x, y + pad))
        draw.text((x + 8, y + 6), label, fill=(20, 20, 20))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--match-dir", type=Path, required=True)
    ap.add_argument("--episode-memory-dir", type=Path, required=True)
    ap.add_argument("--bsp-faces-npz", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-samples", type=int, default=8)
    ap.add_argument("--obj-backend", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--bsp-backend", default="bsp_faces_gpu", choices=["bsp_faces_gpu", "bsp_faces_cpu"])
    ap.add_argument("--make-qa", action="store_true")
    args = ap.parse_args()

    tools_dir = Path(__file__).resolve().parent
    render_mod = import_tool(tools_dir / "build_mesh_dense_condition_v0.py", "mesh_dense_condition_v0")
    teacher_mod = import_tool(tools_dir / "build_dense_condition_v0.py", "teacher_dense_v0")
    manifest = load_json(args.manifest)
    obj_cache = render_mod.load_renderer_cache(args.match_dir, tools_dir)
    bsp_cache = render_mod.load_renderer_cache(args.match_dir, tools_dir, bsp_faces_npz=args.bsp_faces_npz)
    episode_memory = render_mod.load_episode_memory(args.episode_memory_dir)
    rows = []
    for sample in manifest["samples"][: args.max_samples]:
        common_kwargs = dict(
            match_dir=args.match_dir,
            episode=sample["episode"],
            ego_stem=sample["ego_stem"],
            frame_index=int(sample["frame_index"]),
            width=320,
            height=176,
            fov_x=110.0,
            near=4.0,
            far=3000.0,
            pitch_sign=1.0,
            max_triangles=0,
            episode_memory=episode_memory,
            player_radius=14.0,
            player_height=40.0,
            occlusion_tolerance=40.0,
            min_visible_pixels=48,
            player_mask_mode="capsule",
        )
        obj_dense, _obj_depth_units, obj_rgb, obj_meta, _ = render_mod.render_memory_dense_condition(
            cache=obj_cache,
            mesh_backend=args.obj_backend,
            **common_kwargs,
        )
        bsp_dense, _bsp_depth_units, _bsp_rgb, bsp_meta, _ = render_mod.render_memory_dense_condition(
            cache=bsp_cache,
            mesh_backend=args.bsp_backend,
            **common_kwargs,
        )
        obj_hit = obj_dense[1] > 0.5
        bsp_hit = bsp_dense[1] > 0.5
        both_hit = obj_hit & bsp_hit
        player_obj = obj_dense[3] > 0.5
        player_bsp = bsp_dense[3] > 0.5
        episode_dir = args.match_dir / "train" / sample["episode"]
        teacher_depth = None
        teacher_hit = None
        if (episode_dir / f"{sample['ego_stem']}_depth.mkv").exists():
            depth_bgr = teacher_mod.read_frame(episode_dir / f"{sample['ego_stem']}_depth.mkv", int(sample["frame_index"]))
            teacher_depth = teacher_mod.make_depth(depth_bgr, (176, 320))
            teacher_hit = teacher_depth > 1e-5
        row = {
            "sample_id": sample["sample_id"],
            "episode": sample["episode"],
            "ego_stem": sample["ego_stem"],
            "frame_index": int(sample["frame_index"]),
            "obj_hit_ratio": float(obj_hit.mean()),
            "bsp_hit_ratio": float(bsp_hit.mean()),
            "obj_bsp_hit_iou": iou(obj_hit, bsp_hit),
            "obj_bsp_depth_mae_common": mae(obj_dense[0], bsp_dense[0], both_hit),
            "obj_player_pixels": int(player_obj.sum()),
            "bsp_player_pixels": int(player_bsp.sum()),
            "obj_bsp_player_iou": iou(player_obj, player_bsp),
            "obj_backend": args.obj_backend,
            "bsp_backend": args.bsp_backend,
            "obj_mesh_stats": obj_meta.get("mesh_stats", {}),
            "bsp_mesh_stats": bsp_meta.get("mesh_stats", {}),
        }
        if teacher_hit is not None and teacher_depth is not None:
            row.update({
                "obj_teacher_hit_iou": iou(obj_hit, teacher_hit),
                "bsp_teacher_hit_iou": iou(bsp_hit, teacher_hit),
                "obj_teacher_depth_mae_common": mae(obj_dense[0], teacher_depth, obj_hit & teacher_hit),
                "bsp_teacher_depth_mae_common": mae(bsp_dense[0], teacher_depth, bsp_hit & teacher_hit),
            })
        if args.make_qa:
            qa_path = args.out_dir / "samples" / sample["sample_id"] / "obj_vs_bsp_geometry_v0.png"
            make_qa(qa_path, obj_rgb, obj_dense, bsp_dense, teacher_depth)
            row["qa_path"] = str(qa_path)
        rows.append(row)

    def mean(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    out = {
        "kind": "geometry_backend_comparison_v0",
        "manifest": str(args.manifest),
        "obj_backend": args.obj_backend,
        "bsp_backend": args.bsp_backend,
        "bsp_faces_npz": str(args.bsp_faces_npz),
        "sample_count": len(rows),
        "summary": {
            "obj_bsp_hit_iou_mean": mean("obj_bsp_hit_iou"),
            "obj_bsp_depth_mae_common_mean": mean("obj_bsp_depth_mae_common"),
            "obj_bsp_player_iou_mean": mean("obj_bsp_player_iou"),
            "obj_teacher_hit_iou_mean": mean("obj_teacher_hit_iou"),
            "bsp_teacher_hit_iou_mean": mean("bsp_teacher_hit_iou"),
            "obj_teacher_depth_mae_common_mean": mean("obj_teacher_depth_mae_common"),
            "bsp_teacher_depth_mae_common_mean": mean("bsp_teacher_depth_mae_common"),
        },
        "rows": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "geometry_backend_comparison_v0.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_json": str(out_path), "summary": out["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
