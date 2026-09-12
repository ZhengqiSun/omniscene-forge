#!/usr/bin/env python3
"""Export a small replay-mode Map Memory dense-condition dataset.

This is the v0 data factory for adapter training. It repeatedly calls the
Memory-only dense renderer on real episode ticks, so the model input is generated
from Map Memory rather than copied from dataset depth/seg streams.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


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


def player_stems(episode_dir: Path) -> list[str]:
    stems = []
    for path in sorted(episode_dir.glob("*.json")):
        name = path.name
        if "_team_" in name and "_player_" in name and name.endswith("_inst_000.json"):
            stems.append(path.stem)
    return stems


def choose_frames(frames: list[dict[str, Any]], start: int, stop: int | None, stride: int, max_frames: int) -> list[int]:
    hi = min(len(frames), stop if stop is not None else len(frames))
    out = []
    for idx in range(start, hi, stride):
        if frames[idx].get("health", 100) <= 0:
            continue
        out.append(idx)
        if len(out) >= max_frames:
            break
    return out


def load_episode_memory(memory_dir: Path | None) -> tuple[dict[str, Any] | None, Any | None]:
    if memory_dir is None:
        return None, None
    meta = load_json(memory_dir / "episode_memory_meta_v0.json")
    mem = np.load(memory_dir / "episode_memory_v0.npz")
    return meta, mem


def choose_frames_from_memory(mem: Any, player_idx: int, start: int, stop: int | None, stride: int, max_frames: int) -> list[int]:
    alive = mem["alive"][player_idx]
    hi = min(len(alive), stop if stop is not None else len(alive))
    out = []
    for idx in range(start, hi, stride):
        if not bool(alive[idx]):
            continue
        out.append(idx)
        if len(out) >= max_frames:
            break
    return out


def load_candidate_plan(path: Path | None) -> dict[str, list[int]]:
    if path is None:
        return {}
    data = load_json(path)
    out: dict[str, list[int]] = {}
    for row in data.get("candidates", []):
        out.setdefault(row["ego_stem"], []).append(int(row["frame_index"]))
    return {stem: sorted(dict.fromkeys(frames)) for stem, frames in out.items()}


def normalize_geometry_backend(value: str | None) -> str | None:
    if value is None:
        return None
    aliases = {
        "obj": "obj_gpu",
        "gpu": "obj_gpu",
        "obj/gpu": "obj_gpu",
        "obj_gpu": "obj_gpu",
        "bsp": "bsp_faces",
        "bsp_faces": "bsp_faces",
    }
    key = value.strip().lower()
    if key not in aliases:
        raise argparse.ArgumentTypeError(
            f"unsupported geometry backend {value!r}; use obj_gpu/obj/gpu or bsp_faces"
        )
    return aliases[key]


def render_mesh_backend(mesh_backend: str, geometry_backend: str | None) -> str:
    if geometry_backend is None:
        return mesh_backend
    if geometry_backend == "obj_gpu":
        return "gpu"
    if geometry_backend == "bsp_faces":
        return "bsp_faces_gpu"
    raise ValueError(f"Unsupported geometry_backend: {geometry_backend!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-dir", type=Path, required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--ego-limit", type=int, default=2)
    ap.add_argument("--frames-per-ego", type=int, default=4)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=1400)
    ap.add_argument("--stride", type=int, default=200)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=180)
    ap.add_argument("--fov-x", type=float, default=90.0)
    ap.add_argument("--far", type=float, default=3000.0)
    ap.add_argument("--max-triangles", type=int, default=0, help="0 keeps all visible triangles; positive values are a debug speed cap.")
    ap.add_argument("--episode-memory-dir", type=Path, default=None)
    ap.add_argument("--player-radius", type=float, default=18.0)
    ap.add_argument("--player-height", type=float, default=72.0)
    ap.add_argument("--occlusion-tolerance", type=float, default=80.0)
    ap.add_argument("--min-visible-pixels", type=int, default=48)
    ap.add_argument("--camera-yaw-offset", type=float, default=0.0)
    ap.add_argument("--camera-pitch-offset", type=float, default=0.0)
    ap.add_argument("--player-mask-mode", choices=["capsule", "multipart"], default="capsule")
    ap.add_argument("--mesh-backend", choices=["cpu", "gpu", "auto", "bsp_faces_cpu", "bsp_faces_gpu"], default="cpu")
    ap.add_argument("--geometry-backend", type=normalize_geometry_backend, default=None, help="Release-level geometry backend alias: obj_gpu or bsp_faces. Overrides --mesh-backend.")
    ap.add_argument("--bsp-faces-npz", type=Path, default=None)
    ap.add_argument("--candidate-json", type=Path, default=None)
    args = ap.parse_args()
    mesh_backend = render_mesh_backend(args.mesh_backend, args.geometry_backend)

    tools_dir = Path(__file__).resolve().parent
    dense_script = tools_dir / "build_mesh_dense_condition_v0.py"
    dense_mod = import_tool(dense_script, "mesh_dense_condition_v0")
    cache = dense_mod.load_renderer_cache(args.match_dir, tools_dir, bsp_faces_npz=args.bsp_faces_npz)

    episode_dir = args.match_dir / "train" / args.episode
    memory_meta, memory_npz = load_episode_memory(args.episode_memory_dir)
    episode_memory = None
    candidate_plan = load_candidate_plan(args.candidate_json)
    if args.episode_memory_dir is not None:
        episode_memory = dense_mod.load_episode_memory(args.episode_memory_dir)
    if candidate_plan:
        stems = list(candidate_plan.keys())[: args.ego_limit]
    elif memory_meta is not None:
        stems = memory_meta["player_stems"][: args.ego_limit]
    else:
        stems = player_stems(episode_dir)[: args.ego_limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    for ego_stem in stems:
        if candidate_plan:
            if memory_meta is not None and memory_npz is not None:
                mem_idx = memory_meta["player_stems"].index(ego_stem)
                alive = memory_npz["alive"][mem_idx]
                frame_ids = [idx for idx in candidate_plan.get(ego_stem, []) if idx < len(alive) and bool(alive[idx])][: args.frames_per_ego]
            else:
                frames = load_json(episode_dir / f"{ego_stem}.json")
                frame_ids = [idx for idx in candidate_plan.get(ego_stem, []) if idx < len(frames) and float(frames[idx].get("health", 0)) > 0][: args.frames_per_ego]
        elif memory_meta is not None and memory_npz is not None:
            mem_idx = memory_meta["player_stems"].index(ego_stem)
            frame_ids = choose_frames_from_memory(memory_npz, mem_idx, args.start, args.stop, args.stride, args.frames_per_ego)
        else:
            frames = load_json(episode_dir / f"{ego_stem}.json")
            frame_ids = choose_frames(frames, args.start, args.stop, args.stride, args.frames_per_ego)
        for frame_index in frame_ids:
            sample_id = f"{args.episode}_{ego_stem}_f{frame_index:06d}"
            sample_dir = args.out_dir / "samples" / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            dense, mesh_depth_units, rgb, meta, qa_mod = dense_mod.render_memory_dense_condition(
                args.match_dir,
                args.episode,
                ego_stem,
                frame_index,
                args.width,
                args.height,
                args.fov_x,
                4.0,
                args.far,
                1.0,
                args.max_triangles,
                cache=cache,
                episode_memory=episode_memory,
                player_radius=args.player_radius,
                player_height=args.player_height,
                occlusion_tolerance=args.occlusion_tolerance,
                min_visible_pixels=args.min_visible_pixels,
                camera_yaw_offset=args.camera_yaw_offset,
                camera_pitch_offset=args.camera_pitch_offset,
                player_mask_mode=args.player_mask_mode,
                mesh_backend=mesh_backend,
            )
            target_rgb = Image.fromarray(rgb).resize((args.width, args.height), Image.Resampling.BILINEAR)
            target_path = sample_dir / "target_rgb.png"
            target_rgb.save(target_path)
            npz_path = sample_dir / "mesh_dense_condition_v0.npz"
            np.savez_compressed(npz_path, dense=dense, mesh_depth_units=mesh_depth_units)
            meta_path = sample_dir / "mesh_dense_condition_meta_v0.json"
            meta["target_rgb_path"] = str(target_path)
            meta["target_policy"] = "Current-frame RGB target for initial adapter/data plumbing; next-frame target can be enabled once temporal windowing is fixed."
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            qa_mod.make_qa(
                sample_dir / "mesh_dense_condition_qa_v0.png",
                rgb,
                dense[0],
                dense[1],
                dense[3:],
                meta,
            )
            samples.append({
                "sample_id": sample_id,
                "episode": args.episode,
                "ego_stem": ego_stem,
                "frame_index": frame_index,
                "dense_path": str(npz_path),
                "dense_relpath": str(npz_path.relative_to(args.out_dir)),
                "target_rgb_path": str(target_path),
                "target_rgb_relpath": str(target_path.relative_to(args.out_dir)),
                "meta_path": str(meta_path),
                "meta_relpath": str(meta_path.relative_to(args.out_dir)),
                "qa_path": str(sample_dir / "mesh_dense_condition_qa_v0.png"),
                "qa_relpath": str((sample_dir / "mesh_dense_condition_qa_v0.png").relative_to(args.out_dir)),
                "shape": meta["output_shape"],
                "mesh_hit_ratio": meta["mesh_hit_ratio"],
                "nav_semantic_hit_ratio": meta.get("nav_semantic_hit_ratio"),
                "memory_projected_players": len(meta["memory_projected_players"]),
                "channels": meta["channels"],
                "mesh_backend": meta.get("mesh_backend"),
                "geometry_backend": args.geometry_backend,
                "geometry_backend_id": meta.get("geometry_backend_id"),
                "geometry_backend_family": meta.get("geometry_backend_family"),
                "backend_features": meta.get("backend_features"),
                "backend_signature": meta.get("backend_signature"),
                "component_counts": meta.get("component_counts"),
            })

    manifest = {
        "kind": "replay_mode_memory_dense_dataset_v0",
        "match_dir": str(args.match_dir),
        "episode": args.episode,
        "sample_count": len(samples),
        "ego_stems": stems,
        "policy": "Inputs are generated from Map Memory using real episode ticks; dataset RGB/depth/seg are QA/teacher only.",
        "episode_memory_dir": str(args.episode_memory_dir) if args.episode_memory_dir else None,
        "player_capsule_params": {
            "player_radius": args.player_radius,
            "player_height": args.player_height,
            "occlusion_tolerance": args.occlusion_tolerance,
            "min_visible_pixels": args.min_visible_pixels,
            "player_mask_mode": args.player_mask_mode,
        },
        "camera_projection_params": {
            "fov_x": args.fov_x,
            "camera_yaw_offset": args.camera_yaw_offset,
            "camera_pitch_offset": args.camera_pitch_offset,
        },
        "candidate_json": str(args.candidate_json) if args.candidate_json else None,
        "mesh_backend": mesh_backend,
        "geometry_backend": args.geometry_backend,
        "geometry_backend_id": samples[0].get("geometry_backend_id") if samples else None,
        "geometry_backend_family": samples[0].get("geometry_backend_family") if samples else None,
        "backend_features": samples[0].get("backend_features") if samples else None,
        "backend_signatures": sorted({s["backend_signature"] for s in samples if s.get("backend_signature")}),
        "bsp_faces_npz": str(args.bsp_faces_npz) if args.bsp_faces_npz else None,
        "renderer": str(dense_script),
        "samples": samples,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "sample_count": len(samples),
        "ego_stems": stems,
        "manifest": str(args.out_dir / "manifest.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
