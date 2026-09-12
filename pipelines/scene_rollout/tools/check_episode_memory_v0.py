#!/usr/bin/env python3
"""Smoke-check Dynamic Episode Memory against raw JSON for one snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory-dir", type=Path, required=True)
    ap.add_argument("--ego-stem", required=True)
    ap.add_argument("--frame-index", type=int, required=True)
    args = ap.parse_args()

    meta = load_json(args.memory_dir / "episode_memory_meta_v0.json")
    mem = np.load(args.memory_dir / "episode_memory_v0.npz")
    stems = meta["player_stems"]
    ego_idx = stems.index(args.ego_stem)
    frame_index = args.frame_index

    snapshot = {
        "episode": meta["episode"],
        "ego_stem": args.ego_stem,
        "frame_index": frame_index,
        "ego_position": mem["position"][ego_idx, frame_index].astype(float).tolist(),
        "ego_camera_position": mem["camera_position"][ego_idx, frame_index].astype(float).tolist(),
        "ego_yaw": float(mem["yaw"][ego_idx, frame_index]),
        "ego_pitch": float(mem["pitch"][ego_idx, frame_index]),
        "ego_alive": bool(mem["alive"][ego_idx, frame_index]),
        "alive_players": int(mem["alive"][:, frame_index].sum()),
        "action_names": meta["action_names"],
        "ego_actions": [
            name for name, active in zip(meta["action_names"], mem["actions"][ego_idx, frame_index])
            if bool(active)
        ],
    }

    raw_path = Path(meta["episode_dir"]) / f"{args.ego_stem}.json"
    raw = load_json(raw_path)[frame_index]
    diffs = {
        "position_l2": float(np.linalg.norm(mem["position"][ego_idx, frame_index] - np.asarray([raw["x"], raw["y"], raw["z"]], dtype=np.float32))),
        "camera_l2": float(np.linalg.norm(mem["camera_position"][ego_idx, frame_index] - np.asarray(raw["camera_position"], dtype=np.float32))),
        "yaw_abs": float(abs(float(mem["yaw"][ego_idx, frame_index]) - float(raw["yaw"]))),
        "pitch_abs": float(abs(float(mem["pitch"][ego_idx, frame_index]) - float(raw["pitch"]))),
    }
    print(json.dumps({"snapshot": snapshot, "raw_diff": diffs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
