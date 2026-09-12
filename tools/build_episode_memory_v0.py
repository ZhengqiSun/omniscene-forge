#!/usr/bin/env python3
"""Build replay-mode Dynamic Episode Memory from raw per-player JSON files.

The output is a compact cache used by the Map Memory renderer/data factory. It
keeps real episode ticks for training-time replay, while making the schema
explicit enough to later swap in an action-driven simulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ACTION_NAMES = [
    "forward",
    "back",
    "left",
    "right",
    "jump",
    "crouch",
    "walk",
    "fire",
    "reload",
    "use",
    "scope",
    "plant",
    "defuse",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def player_files(episode_dir: Path) -> list[Path]:
    return sorted(
        p for p in episode_dir.glob("*.json")
        if "_team_" in p.name and "_player_" in p.name and p.name.endswith("_inst_000.json")
    )


def parse_team_player(stem: str) -> tuple[int, int]:
    parts = stem.split("_")
    return int(parts[parts.index("team") + 1]), int(parts[parts.index("player") + 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-dir", type=Path, required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    episode_dir = args.match_dir / "train" / args.episode
    files = player_files(episode_dir)
    if not files:
        raise FileNotFoundError(f"No player JSON files in {episode_dir}")

    tracks = [load_json(p) for p in files]
    frame_count = max(len(t) for t in tracks)
    player_count = len(files)

    pos = np.full((player_count, frame_count, 3), np.nan, dtype=np.float32)
    camera_pos = np.full((player_count, frame_count, 3), np.nan, dtype=np.float32)
    yaw = np.full((player_count, frame_count), np.nan, dtype=np.float32)
    pitch = np.full((player_count, frame_count), np.nan, dtype=np.float32)
    health = np.zeros((player_count, frame_count), dtype=np.float32)
    alive = np.zeros((player_count, frame_count), dtype=np.bool_)
    tick = np.full((player_count, frame_count), -1, dtype=np.int64)
    actions = np.zeros((player_count, frame_count, len(ACTION_NAMES)), dtype=np.bool_)
    look_delta = np.zeros((player_count, frame_count, 2), dtype=np.float32)

    teams = []
    player_indices = []
    stems = []
    lengths = []

    for pi, (path, frames) in enumerate(zip(files, tracks)):
        team_id, player_idx = parse_team_player(path.stem)
        stems.append(path.stem)
        teams.append(team_id)
        player_indices.append(player_idx)
        lengths.append(len(frames))
        for ti, frame in enumerate(frames):
            pos[pi, ti] = [frame.get("x", np.nan), frame.get("y", np.nan), frame.get("z", np.nan)]
            cam = frame.get("camera_position") or [frame.get("x", np.nan), frame.get("y", np.nan), frame.get("z", 0.0) + 64.0]
            camera_pos[pi, ti] = cam
            yaw[pi, ti] = frame.get("yaw", np.nan)
            pitch[pi, ti] = frame.get("pitch", np.nan)
            health[pi, ti] = frame.get("health", 0)
            alive[pi, ti] = health[pi, ti] > 0
            tick[pi, ti] = frame.get("tick", -1)
            action = frame.get("action") or {}
            for ai, name in enumerate(ACTION_NAMES):
                actions[pi, ti, ai] = bool(action.get(name, False))
            look_delta[pi, ti] = [float(action.get("look_dx", 0.0)), float(action.get("look_dy", 0.0))]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.out_dir / "episode_memory_v0.npz"
    np.savez_compressed(
        npz_path,
        position=pos,
        camera_position=camera_pos,
        yaw=yaw,
        pitch=pitch,
        health=health,
        alive=alive,
        tick=tick,
        actions=actions,
        look_delta=look_delta,
        team_id=np.asarray(teams, dtype=np.int32),
        player_index=np.asarray(player_indices, dtype=np.int32),
        track_length=np.asarray(lengths, dtype=np.int32),
    )

    meta = {
        "kind": "dynamic_episode_memory_v0",
        "match_dir": str(args.match_dir),
        "episode": args.episode,
        "episode_dir": str(episode_dir),
        "npz_path": str(npz_path),
        "player_count": player_count,
        "frame_count": frame_count,
        "player_stems": stems,
        "team_id": teams,
        "player_index": player_indices,
        "track_length": lengths,
        "action_names": ACTION_NAMES,
        "schema": {
            "position": "[P,T,3] world player origin xyz",
            "camera_position": "[P,T,3] world camera xyz",
            "yaw": "[P,T] degrees",
            "pitch": "[P,T] degrees",
            "alive": "[P,T] bool from health > 0",
            "actions": "[P,T,A] bool action flags",
        },
        "policy": "Replay-mode Memory uses real episode states for training-time fidelity. Simulator-mode can later write the same schema from action rollout.",
    }
    meta_path = args.out_dir / "episode_memory_meta_v0.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "out_dir": str(args.out_dir),
        "npz": str(npz_path),
        "meta": str(meta_path),
        "player_count": player_count,
        "frame_count": frame_count,
        "shape_position": list(pos.shape),
        "alive_ratio": float(alive.mean()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
