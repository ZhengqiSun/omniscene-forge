#!/usr/bin/env python3
"""Compare replay vs simulator Episode Memory in Map Memory dense space."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


def import_tool(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float | None:
    ma = a > 0.5
    mb = b > 0.5
    union = int((ma | mb).sum())
    if union == 0:
        return None
    return float((ma & mb).sum() / union)


def channel_mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None:
    if not bool(mask.any()):
        return None
    return float(np.abs(a[mask] - b[mask]).mean())


def compare_dense(replay: np.ndarray, sim: np.ndarray) -> dict[str, Any]:
    replay_mask = replay[3] > 0.5
    sim_mask = sim[3] > 0.5
    common = replay_mask & sim_mask
    union = replay_mask | sim_mask
    return {
        "other_player_mask_iou": mask_iou(replay[3], sim[3]),
        "replay_other_player_pixels": int(replay_mask.sum()),
        "sim_other_player_pixels": int(sim_mask.sum()),
        "other_player_pixel_delta": int(sim_mask.sum() - replay_mask.sum()),
        "other_player_depth_mae_common": channel_mae(replay[4], sim[4], common),
        "other_player_depth_mae_union": channel_mae(replay[4], sim[4], union),
        "other_player_yaw_sin_mae_common": channel_mae(replay[5], sim[5], common),
        "other_player_yaw_cos_mae_common": channel_mae(replay[6], sim[6], common),
        "env_depth_mae_common_hit": channel_mae(replay[0], sim[0], (replay[1] > 0.5) & (sim[1] > 0.5)),
    }


def projected_stems(meta: dict[str, Any]) -> list[str]:
    return [str(p.get("stem")) for p in meta.get("memory_projected_players", []) if p.get("stem")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-dir", type=Path, required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--replay-memory-dir", type=Path, required=True)
    ap.add_argument("--sim-memory-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--ego-stems", default="")
    ap.add_argument("--frames", default="")
    ap.add_argument("--focus-player-stems", default="")
    ap.add_argument("--start", type=int, default=780)
    ap.add_argument("--stop", type=int, default=908)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=176)
    ap.add_argument("--fov-x", type=float, default=90.0)
    ap.add_argument("--near", type=float, default=4.0)
    ap.add_argument("--far", type=float, default=3000.0)
    ap.add_argument("--pitch-sign", type=float, default=1.0)
    ap.add_argument("--mesh-backend", choices=["cpu", "gpu", "auto", "bsp_faces_cpu", "bsp_faces_gpu"], default="bsp_faces_cpu")
    ap.add_argument("--bsp-faces-npz", type=Path, required=True)
    ap.add_argument("--max-triangles", type=int, default=0)
    args = ap.parse_args()

    tools_dir = Path(__file__).resolve().parent
    dense_mod = import_tool(tools_dir / "build_mesh_dense_condition_v0.py", "build_mesh_dense_condition_v0")
    cache = dense_mod.load_renderer_cache(args.match_dir, bsp_faces_npz=args.bsp_faces_npz)
    replay_memory = dense_mod.load_episode_memory(args.replay_memory_dir)
    sim_memory = dense_mod.load_episode_memory(args.sim_memory_dir)
    replay_meta = replay_memory["meta"]
    if args.ego_stems:
        ego_stems = [x.strip() for x in args.ego_stems.split(",") if x.strip()]
    else:
        ego_stems = [s for s in replay_meta["player_stems"] if "_team_2_" in s][:2]
    if args.frames:
        frames = [int(x) for x in args.frames.split(",") if x.strip()]
    else:
        frames = list(range(args.start, args.stop + 1, args.stride))
    focus_stems = {x.strip() for x in args.focus_player_stems.split(",") if x.strip()}

    rows = []
    for ego_stem in ego_stems:
        for frame_index in frames:
            replay_dense, _, _, replay_render_meta, _ = dense_mod.render_memory_dense_condition(
                args.match_dir,
                args.episode,
                ego_stem,
                frame_index,
                args.width,
                args.height,
                args.fov_x,
                args.near,
                args.far,
                args.pitch_sign,
                args.max_triangles,
                cache=cache,
                episode_memory=replay_memory,
                mesh_backend=args.mesh_backend,
            )
            sim_dense, _, _, sim_render_meta, _ = dense_mod.render_memory_dense_condition(
                args.match_dir,
                args.episode,
                ego_stem,
                frame_index,
                args.width,
                args.height,
                args.fov_x,
                args.near,
                args.far,
                args.pitch_sign,
                args.max_triangles,
                cache=cache,
                episode_memory=sim_memory,
                mesh_backend=args.mesh_backend,
            )
            row = {
                "ego_stem": ego_stem,
                "frame_index": int(frame_index),
                **compare_dense(replay_dense, sim_dense),
                "replay_visible_stems": projected_stems(replay_render_meta),
                "sim_visible_stems": projected_stems(sim_render_meta),
                "replay_visible_players": len(replay_render_meta.get("memory_projected_players", [])),
                "sim_visible_players": len(sim_render_meta.get("memory_projected_players", [])),
            }
            row["focus_visible_replay"] = sorted(focus_stems & set(row["replay_visible_stems"]))
            row["focus_visible_sim"] = sorted(focus_stems & set(row["sim_visible_stems"]))
            row["focus_visible"] = bool(row["focus_visible_replay"] or row["focus_visible_sim"])
            rows.append(row)

    def mean_of(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "sample_count": len(rows),
        "mean_other_player_mask_iou": mean_of("other_player_mask_iou"),
        "mean_replay_other_player_pixels": mean_of("replay_other_player_pixels"),
        "mean_sim_other_player_pixels": mean_of("sim_other_player_pixels"),
        "mean_other_player_pixel_delta": mean_of("other_player_pixel_delta"),
        "mean_other_player_depth_mae_common": mean_of("other_player_depth_mae_common"),
        "mean_other_player_depth_mae_union": mean_of("other_player_depth_mae_union"),
        "mean_other_player_yaw_sin_mae_common": mean_of("other_player_yaw_sin_mae_common"),
        "mean_other_player_yaw_cos_mae_common": mean_of("other_player_yaw_cos_mae_common"),
        "mean_env_depth_mae_common_hit": mean_of("env_depth_mae_common_hit"),
        "mean_replay_visible_players": mean_of("replay_visible_players"),
        "mean_sim_visible_players": mean_of("sim_visible_players"),
        "focus_visible_sample_count": int(sum(bool(r.get("focus_visible")) for r in rows)),
    }
    out = {
        "kind": "simulator_memory_dense_comparison_v0",
        "match_dir": str(args.match_dir),
        "episode": args.episode,
        "replay_memory_dir": str(args.replay_memory_dir),
        "sim_memory_dir": str(args.sim_memory_dir),
        "ego_stems": ego_stems,
        "frames": frames,
        "focus_player_stems": sorted(focus_stems),
        "params": {
            "width": args.width,
            "height": args.height,
            "mesh_backend": args.mesh_backend,
            "bsp_faces_npz": str(args.bsp_faces_npz),
        },
        "summary": summary,
        "rows": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "simulator_memory_dense_comparison_v0.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out_path), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
