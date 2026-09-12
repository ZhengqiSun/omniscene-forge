#!/usr/bin/env python3
"""Validate and shard strict zero-visibility windows for dense rendering."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import random
from typing import Any


def load_miner(tools_dir: Path):
    path = tools_dir / "mine_zero_visibility_windows_v0.py"
    spec = importlib.util.spec_from_file_location("zero_visibility_miner_v0", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import miner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def enrich(row: dict[str, Any], index: int) -> dict[str, Any]:
    out = dict(row)
    frames = []
    for key in ("latent_frames", "positive_latent_frames"):
        values = out.get(key)
        if isinstance(values, list):
            frames = [int(value) for value in values]
            break
    if len(frames) != 21:
        raise ValueError(f"{out.get('clip_id')}: expected 21 latent frames, got {len(frames)}")
    out.update({
        "source_manifest_index": index,
        "latent_frame_count": len(frames),
        "latent_frames": frames,
        "positive_latent_frame_count": len(frames),
        "positive_latent_frames": frames,
        "selection_role": "context",
        "map_memory_selection_roles": ["context"] * len(frames),
        "map_memory_positive_frames": 0,
        "map_memory_context_frames": len(frames),
    })
    return out


def audit_row(row: dict[str, Any], miner: Any) -> dict[str, Any]:
    with open(row["player_visibility"], "r", encoding="utf-8") as handle:
        visibility = json.load(handle)
    intervals = miner.visibility_intervals(visibility)
    start = int(row["raw_start"])
    end = int(row["frame_count_end"])
    with open(row["action_json"], "r", encoding="utf-8") as handle:
        actions = json.load(handle)
    action_by_frame = {int(action.get("frame_count", index)): action for index, action in enumerate(actions)}
    full_window = [action_by_frame.get(frame) for frame in range(start, end + 1)]
    latent_actions = [action_by_frame.get(int(frame)) for frame in row["latent_frames"]]
    overlap = miner.overlaps(intervals, start, end)
    missing_action_frames = sum(action is None for action in full_window)
    min_health = min(
        (float(action.get("health", 0) or 0) for action in full_window if action is not None),
        default=0.0,
    )
    camera_ok = all(action is not None and miner.finite_camera(action) for action in latent_actions)
    pose_ok = all(
        action is not None
        and all(math.isfinite(float(action.get(key, float("nan")))) for key in ("x", "y", "z", "yaw", "pitch"))
        for action in latent_actions
    )
    paths_ok = all(Path(str(row[key])).is_file() for key in (
        "mp4", "action_json", "episode_info", "video_manifest", "game_manifest", "world_events", "player_visibility"
    ))
    passed = not overlap and missing_action_frames == 0 and min_health > 0 and camera_ok and pose_ok and paths_ok
    return {
        "clip_id": row["clip_id"],
        "split": row["map_memory_split"],
        "visibility_overlap": overlap,
        "missing_action_frames": missing_action_frames,
        "min_health": min_health,
        "camera_ok": camera_ok,
        "pose_ok": pose_ok,
        "paths_ok": paths_ok,
        "status": "pass" if passed else "fail",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-train", type=int, default=18_000)
    parser.add_argument("--expected-heldout", type=int, default=2_000)
    parser.add_argument("--shards", type=int, default=12)
    parser.add_argument("--audit-samples", type=int, default=200)
    parser.add_argument("--audit-seed", type=int, default=20_260_716)
    args = parser.parse_args()

    source_rows = read_jsonl(args.input_manifest)
    rows = [enrich(row, index) for index, row in enumerate(source_rows)]
    ids = [str(row["clip_id"]) for row in rows]
    splits = Counter(str(row.get("map_memory_split")) for row in rows)
    train_rows = [row for row in rows if row["map_memory_split"] == "train"]
    heldout_rows = [row for row in rows if row["map_memory_split"] in {"val", "test"}]
    train_matches = {row["game_id"] for row in train_rows}
    heldout_matches = {row["game_id"] for row in heldout_rows}
    role_failures = [row["clip_id"] for row in rows if row["selection_role"] != "context" or set(row["map_memory_selection_roles"]) != {"context"}]

    rng = random.Random(args.audit_seed)
    audit_rows = rng.sample(rows, min(args.audit_samples, len(rows)))
    miner = load_miner(Path(__file__).resolve().parent)
    audits = [audit_row(row, miner) for row in audit_rows]
    audit_failures = [row for row in audits if row["status"] != "pass"]
    failures: list[str] = []
    if len(train_rows) != args.expected_train:
        failures.append(f"train rows {len(train_rows)} != {args.expected_train}")
    if len(heldout_rows) != args.expected_heldout:
        failures.append(f"heldout rows {len(heldout_rows)} != {args.expected_heldout}")
    if len(ids) != len(set(ids)):
        failures.append(f"duplicate clip ids: {len(ids) - len(set(ids))}")
    if train_matches & heldout_matches:
        failures.append(f"match split leakage: {len(train_matches & heldout_matches)}")
    if role_failures:
        failures.append(f"context role failures: {len(role_failures)}")
    if audit_failures:
        failures.append(f"source truth audit failures: {len(audit_failures)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "zero_visibility_all_context_v0.jsonl", rows)
    write_jsonl(args.out_dir / "zero_visibility_train_context_v0.jsonl", train_rows)
    write_jsonl(args.out_dir / "zero_visibility_heldout_context_v0.jsonl", heldout_rows)
    shard_counts = []
    for index in range(args.shards):
        shard = rows[index::args.shards]
        write_jsonl(args.out_dir / f"render_shard_{index:02d}.jsonl", shard)
        shard_counts.append(len(shard))

    report = {
        "kind": "zero_visibility_render_prep_report_v0",
        "status": "pass" if not failures else "fail",
        "input_manifest": str(args.input_manifest),
        "output_dir": str(args.out_dir),
        "row_count": len(rows),
        "split_counts": dict(sorted(splits.items())),
        "train_rows": len(train_rows),
        "heldout_rows": len(heldout_rows),
        "train_match_count": len(train_matches),
        "heldout_match_count": len(heldout_matches),
        "split_match_leakage_count": len(train_matches & heldout_matches),
        "duplicate_clip_ids": len(ids) - len(set(ids)),
        "context_role_failure_count": len(role_failures),
        "audit_sample_count": len(audits),
        "audit_failure_count": len(audit_failures),
        "audit_examples": audits[:20],
        "audit_failure_examples": audit_failures[:20],
        "shard_counts": shard_counts,
        "failures": failures,
    }
    report_path = args.out_dir / "render_prep_report_v0.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
