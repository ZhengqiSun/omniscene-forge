from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from baselines.scope.actions.resample_actions import resample
from baselines.scope.scope_bridge.schema import sha256_file

BUTTON_COLS = ["right_trigger", "left_trigger", "south", "right_thumb", "west", "north"]
CONVERTER_VERSION = "csgo_scope_actions_v1"


def parse_csgo_rows(path: str | Path, start: int, count: int, signs: tuple[int, int, int, int]):
    rows = json.loads(Path(path).read_text())
    rows = rows[start:start + count]
    if len(rows) != count:
        raise ValueError(f"requested {count} raw actions at {start}, found {len(rows)}")
    movement, mouse, buttons = [], [], []
    sx, sy, rx, ry = signs
    for row in rows:
        action = row.get("action")
        if not isinstance(action, dict):
            raise ValueError("each raw row must contain an action object")
        x = float(bool(action.get("right"))) - float(bool(action.get("left")))
        y = float(bool(action.get("forward"))) - float(bool(action.get("back")))
        movement.append([sx * x, sy * y])
        mouse.append([rx * float(action.get("look_dx", 0.0)), ry * float(action.get("look_dy", 0.0))])
        buttons.append([
            float(bool(action.get("fire"))), float(bool(action.get("scope"))),
            float(bool(action.get("jump"))), 0.0, float(bool(action.get("reload"))), 0.0,
        ])
    return np.asarray(movement), np.asarray(mouse), np.asarray(buttons)


def load_calibration(path: str | Path) -> tuple[float, str]:
    data = json.loads(Path(path).read_text())
    if data.get("schema_id") != "ScopeMouseCalibrationV1" or data.get("fit_split") != "train":
        raise ValueError("mouse calibration must be ScopeMouseCalibrationV1 fitted on train")
    return float(data["gain"]), sha256_file(path)


def write_parquet(path: str | Path, actions, metadata: dict) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas and pyarrow are required to write SCOPE parquet") from exc
    data = {name: actions.keyboard[:, i].astype(np.int8) for i, name in enumerate(BUTTON_COLS)}
    data["j_left"] = [list(map(float, row[:2])) for row in actions.sticks]
    data["j_right"] = [list(map(float, row[2:])) for row in actions.sticks]
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_parquet(path, index=False)
    path.with_suffix(path.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def convert(args: argparse.Namespace) -> dict:
    signs = (args.left_x_sign, args.left_y_sign, args.right_x_sign, args.right_y_sign)
    if any(value not in (-1, 1) for value in signs):
        raise ValueError("all axis signs must be explicitly set to -1 or +1; official visual sign validation is pending")
    gain, calibration_hash = load_calibration(args.mouse_calibration)
    source_count = int(np.ceil(args.num_frames / args.target_fps * args.source_fps))
    move, mouse, buttons = parse_csgo_rows(args.raw_action_path, args.raw_start, source_count, signs)
    converted = resample(move, mouse, buttons, args.source_fps, args.target_fps, args.num_frames, gain)
    mouse_out = converted.sticks[:, 2:]
    metadata = {
        "schema_id": "ScopeActionParquetMetadataV1", "converter_version": CONVERTER_VERSION,
        "source_action_path": str(Path(args.raw_action_path).resolve()),
        "source_action_sha256": sha256_file(args.raw_action_path), "source_fps": args.source_fps,
        "target_fps": args.target_fps, "num_source_rows": source_count, "num_output_rows": args.num_frames,
        "resampling_policy": "timestamp_overlap:movement=weighted_state,buttons=OR,mouse=integrated_delta",
        "mouse_calibration_sha256": calibration_hash, "mouse_gain": gain,
        "axis_signs": {"j_left_x": signs[0], "j_left_y": signs[1], "j_right_x": signs[2], "j_right_y": signs[3]},
        "unsupported_raw_events": {"right_thumb_melee": "zero:no raw key", "north_weapon_switch": "zero:no raw event"},
        "events_before": {BUTTON_COLS[i]: int(buttons[:, i].sum()) for i in range(6)},
        "events_after": {BUTTON_COLS[i]: int(converted.keyboard[:, i].sum()) for i in range(6)},
        "mouse_clip_rate": float((np.abs(mouse_out) >= 1).mean()),
        "mouse_mean": mouse_out.mean(axis=0).tolist(), "mouse_std": mouse_out.std(axis=0).tolist(),
    }
    write_parquet(args.output, converted, metadata)
    return metadata


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-action-path", required=True); p.add_argument("--raw-start", type=int, required=True)
    p.add_argument("--source-fps", type=float, default=32); p.add_argument("--target-fps", type=float, default=20)
    p.add_argument("--num-frames", type=int, choices=(81, 101), required=True); p.add_argument("--mouse-calibration", required=True)
    for name in ("left-x-sign", "left-y-sign", "right-x-sign", "right-y-sign"):
        p.add_argument("--" + name, type=int, choices=(-1, 1), required=True)
    p.add_argument("--output", required=True)
    print(json.dumps(convert(p.parse_args()), indent=2))


if __name__ == "__main__": main()
