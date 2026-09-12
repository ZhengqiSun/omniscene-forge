#!/usr/bin/env python3
"""Build causal, latent-aligned action features from aligned CS:GO clips."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from memory_dense_action_adapter_v0 import (
    ACTION_CHANNELS_V0,
    aggregate_action_window_v0,
)


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


@lru_cache(maxsize=8)
def load_action_frames(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: action JSON must be a non-empty list")
    return rows


def frames_inclusive(rows: list[dict[str, Any]], start: int, end: int) -> list[dict[str, Any]]:
    if start < 0 or end < start:
        raise ValueError(f"invalid raw action interval [{start}, {end}]")
    if end < len(rows):
        candidate = rows[start : end + 1]
        if candidate and int(candidate[0].get("frame_count", -1)) == start and int(candidate[-1].get("frame_count", -1)) == end:
            return candidate
    by_frame = {int(row["frame_count"]): row for row in rows if "frame_count" in row}
    missing = [index for index in range(start, end + 1) if index not in by_frame]
    if missing:
        raise ValueError(f"action JSON missing raw frames in [{start}, {end}]: first={missing[:8]}")
    return [by_frame[index] for index in range(start, end + 1)]


def build_clip(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    anchors = [int(value) for value in row.get("map_memory_raw_frame_indices") or []]
    if len(anchors) != 21 or any(b <= a for a, b in zip(anchors, anchors[1:])):
        raise ValueError(f"{row.get('clip_id')}: expected 21 increasing map_memory_raw_frame_indices")
    source = load_action_frames(str(row["action_json"]))
    vectors = []
    starts = []
    ends = []
    debug = []
    previous_weapon = None
    for index, anchor in enumerate(anchors):
        start = anchor if index == 0 else anchors[index - 1] + 1
        end = anchor
        raw_rows = frames_inclusive(source, start, end)
        vector, item = aggregate_action_window_v0(raw_rows, previous_weapon=previous_weapon)
        previous_weapon = str((raw_rows[-1].get("action") or {}).get("weapon_slot") or "")
        vectors.append(vector)
        starts.append(start)
        ends.append(end)
        debug.append(item)
    return (
        np.stack(vectors).astype(np.float32),
        np.asarray(starts, dtype=np.int32),
        np.asarray(ends, dtype=np.int32),
        debug,
    )


def cache_path_for(root: Path, clip_id: str) -> Path:
    shard = hashlib.sha1(clip_id.encode("utf-8")).hexdigest()[:2]
    return root / "cache" / shard / f"{clip_id}.npz"


def write_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--split", default=None)
    parser.add_argument("--require-label", default=None, help="Keep only rows containing this event50h label.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum unique clips; intended for smoke runs.")
    parser.add_argument("--existing", choices=["error", "skip"], default="error")
    args = parser.parse_args()

    source_manifest = args.cache_manifest.resolve()
    output_root = args.output_root.resolve()
    output_manifest = (args.output_manifest or (output_root / "action_cache_manifest_v0.jsonl")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    if output_manifest.exists() and args.existing == "error":
        raise FileExistsError(f"refusing to overwrite existing manifest: {output_manifest}")

    started = time.time()
    seen: set[str] = set()
    output_rows = []
    source_rows = duplicates = split_skips = label_skips = existing_skips = 0
    for line_number, row in iter_jsonl(source_manifest):
        source_rows += 1
        if args.split is not None and str(row.get("map_memory_split")) != args.split:
            split_skips += 1
            continue
        if args.require_label is not None and args.require_label not in (row.get("event50h_labels") or []):
            label_skips += 1
            continue
        clip_id = str(row["clip_id"])
        if clip_id in seen:
            duplicates += 1
            continue
        seen.add(clip_id)
        if args.limit is not None and len(output_rows) >= args.limit:
            break

        cache_path = cache_path_for(output_root, clip_id)
        if cache_path.exists():
            if args.existing == "error":
                raise FileExistsError(f"refusing to overwrite existing cache: {cache_path}")
            existing_skips += 1
            with np.load(cache_path, allow_pickle=False) as data:
                vector = np.asarray(data["action_vector"], dtype=np.float32)
                bin_starts = np.asarray(data["raw_bin_start"], dtype=np.int32)
                bin_ends = np.asarray(data["raw_bin_end"], dtype=np.int32)
        else:
            vector, bin_starts, bin_ends, _debug = build_clip(row)
            write_npz_atomic(
                cache_path,
                action_vector=vector,
                channels=np.asarray(ACTION_CHANNELS_V0),
                raw_frame_anchors=np.asarray(row["map_memory_raw_frame_indices"], dtype=np.int32),
                raw_bin_start=bin_starts,
                raw_bin_end=bin_ends,
            )

        fire_index = ACTION_CHANNELS_V0.index("fire_any")
        reload_index = ACTION_CHANNELS_V0.index("reload_any")
        switch_index = ACTION_CHANNELS_V0.index("weapon_switch_any")
        fire_fraction_index = ACTION_CHANNELS_V0.index("fire_fraction")
        reload_fraction_index = ACTION_CHANNELS_V0.index("reload_fraction")
        bin_lengths = bin_ends - bin_starts + 1
        output_rows.append(
            {
                "kind": "interaction_action_cache_v0",
                "clip_id": clip_id,
                "action_cache": str(cache_path),
                "source_cache_manifest": str(source_manifest),
                "source_line_number": line_number,
                "action_json": str(row["action_json"]),
                "map_memory_split": row.get("map_memory_split"),
                "map_memory_raw_frame_indices": row["map_memory_raw_frame_indices"],
                "shape": list(vector.shape),
                "channels": list(ACTION_CHANNELS_V0),
                "stats": {
                    "fire_latent_frames": int((vector[:, fire_index] > 0).sum()),
                    "reload_latent_frames": int((vector[:, reload_index] > 0).sum()),
                    "weapon_switch_latent_frames": int((vector[:, switch_index] > 0).sum()),
                    "raw_fire_frames": int(round(float((vector[:, fire_fraction_index] * bin_lengths).sum()))),
                    "raw_reload_frames": int(round(float((vector[:, reload_fraction_index] * bin_lengths).sum()))),
                },
                "event50h_labels": row.get("event50h_labels") or [],
                "event50h_primary_label": row.get("event50h_primary_label"),
                "aggregation": "causal trailing raw bins: latent0=[anchor0], latent_i=(anchor_i-1,anchor_i]",
            }
        )

    temp_manifest = output_manifest.with_name(output_manifest.name + f".tmp-{os.getpid()}")
    with temp_manifest.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temp_manifest, output_manifest)

    report = {
        "kind": "interaction_action_cache_build_report_v0",
        "source_cache_manifest": str(source_manifest),
        "output_root": str(output_root),
        "output_manifest": str(output_manifest),
        "source_rows_scanned": source_rows,
        "unique_clips_written": len(output_rows),
        "duplicate_rows_skipped": duplicates,
        "split_rows_skipped": split_skips,
        "label_rows_skipped": label_skips,
        "existing_cache_rows_reused": existing_skips,
        "limit": args.limit,
        "split": args.split,
        "require_label": args.require_label,
        "channels": list(ACTION_CHANNELS_V0),
        "channel_count": len(ACTION_CHANNELS_V0),
        "elapsed_sec": round(time.time() - started, 3),
    }
    report_path = output_manifest.with_suffix(output_manifest.suffix + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
