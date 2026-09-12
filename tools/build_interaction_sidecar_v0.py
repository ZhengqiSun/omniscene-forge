#!/usr/bin/env python3
"""Add interaction_v1 channels to an already-built state_channels_v2 cache.

Why a sidecar instead of build_state_channels_v1.py: v1 carries the stale v0
projection defaults (--far 4096, --pitch-sign -1) that v2 corrected, and
rebuilding opponent_dead_mask is the expensive part of the state cache. Our v2
cache is already built with the corrected projection, so this tool copies those
arrays through untouched and only computes the 9 interaction arrays from each
clip's action JSON. Output is a state_channels_v1_record manifest that both
train_memory_dense_adapter_state_v0.py (reads state fields, ignores the rest)
and train_memory_dense_adapter_interaction_v1.py (reads both) accept.

Writes only under --out-dir. Never mutates the input v0 cache.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interaction_channels_v0 import WEAPON_VOCABULARY, build_interaction_channels

EVENT_FIELDS = (
    "ego_fire_event",
    "ego_reload_event",
    "ego_weapon_switch_event",
    "ego_throw_event",
)
ONEHOT_FIELDS = (
    "ego_current_weapon_onehot",
    "ego_fire_weapon_onehot",
    "ego_reload_weapon_onehot",
    "ego_switch_target_onehot",
    "ego_throw_item_onehot",
)
STATE_FIELDS = ("ego_alive", "ego_health", "opponent_dead_mask")
INTERACTION_CHANNEL_NAMES = (
    "ego_fire_event_constant_plane",
    "ego_reload_event_constant_plane",
    "ego_weapon_switch_event_constant_plane",
    "ego_throw_event_constant_plane",
    "ego_current_weapon_onehot_constant_plane",
    "ego_fire_weapon_onehot_constant_plane",
    "ego_reload_weapon_onehot_constant_plane",
    "ego_switch_target_onehot_constant_plane",
    "ego_throw_item_onehot_constant_plane",
)


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def validate_interaction(
    interaction: dict[str, np.ndarray], *, clip_id: str, frame_count: int
) -> dict[str, np.ndarray]:
    required = EVENT_FIELDS + ONEHOT_FIELDS + ("ego_weapon_slot_name",)
    missing = [k for k in required if k not in interaction]
    if missing:
        raise ValueError(f"clip_id={clip_id}: missing interaction fields: {missing}")

    vocab = len(WEAPON_VOCABULARY)
    out: dict[str, np.ndarray] = {}
    for key in EVENT_FIELDS:
        value = np.asarray(interaction[key])
        if value.shape != (frame_count,):
            raise ValueError(
                f"clip_id={clip_id}: {key} shape={list(value.shape)}, expected=[{frame_count}]"
            )
        out[key] = value.astype(np.float32, copy=False)
    for key in ONEHOT_FIELDS:
        value = np.asarray(interaction[key])
        if value.shape != (frame_count, vocab):
            raise ValueError(
                f"clip_id={clip_id}: {key} shape={list(value.shape)}, "
                f"expected=[{frame_count},{vocab}]"
            )
        rows = value.astype(np.float32, copy=False)
        sums = rows.sum(axis=1)
        if not np.all(np.isclose(sums, 1.0) | np.isclose(sums, 0.0)):
            raise ValueError(f"clip_id={clip_id}: {key} rows are not one-hot/zero")
        out[key] = rows
    out["ego_weapon_slot_name"] = np.asarray(interaction["ego_weapon_slot_name"]).astype("<U64")
    return out


def event_counts(interaction: dict[str, np.ndarray]) -> dict[str, int]:
    return {
        key.removeprefix("ego_").removesuffix("_event"): int((interaction[key] > 0).sum())
        for key in EVENT_FIELDS
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-manifest", type=Path, required=True,
                    help="Existing state_channels_v0/v2 manifest (jsonl).")
    ap.add_argument("--cache-manifest", type=Path, action="append", default=[],
                    help="Aligned cache manifest(s) supplying action_json per clip_id. Repeatable.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="Max clips, taken in --cache-manifest order to match trainer --limit.")
    ap.add_argument("--state-dir-name", default="state_cache_v1")
    ap.add_argument("--manifest-name", default="state_cache_manifest_v1.jsonl")
    ap.add_argument("--report-name", default="interaction_sidecar_report_v0.json")
    ap.add_argument("--progress-every", type=int, default=200)
    args = ap.parse_args()

    if not args.cache_manifest:
        raise SystemExit("--cache-manifest is required at least once")

    # clip order must match the trainer's --limit ordering: cache manifests in
    # the order given, rows in file order.
    order: list[str] = []
    action_by_clip: dict[str, dict[str, Any]] = {}
    for manifest in args.cache_manifest:
        for row in iter_jsonl(manifest):
            clip_id = str(row.get("clip_id"))
            if clip_id in action_by_clip:
                continue
            action_by_clip[clip_id] = {
                "action_json": row.get("action_json"),
                "raw_frames": row.get("map_memory_raw_frame_indices") or row.get("raw_indices"),
                "latent_cache": row.get("latent_cache"),
                "source_cache_manifest": str(manifest),
            }
            order.append(clip_id)
    print(f"cache manifests: {len(args.cache_manifest)}  unique clips: {len(order)}", flush=True)

    state_by_clip: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(args.state_manifest):
        state_by_clip[str(row.get("clip_id"))] = row
    print(f"state manifest rows: {len(state_by_clip)}", flush=True)

    selected = [c for c in order if c in state_by_clip]
    if args.limit is not None:
        selected = selected[: args.limit]
    print(f"selected clips: {len(selected)}", flush=True)

    out_root = args.out_dir.resolve()
    state_dir = out_root / args.state_dir_name
    state_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / args.manifest_name
    written = 0
    skipped: list[dict[str, str]] = []
    totals = {k.removeprefix("ego_").removesuffix("_event"): 0 for k in EVENT_FIELDS}
    started = time.time()

    with manifest_path.open("w", encoding="utf-8") as out_handle:
        for index, clip_id in enumerate(selected):
            info = action_by_clip[clip_id]
            state_row = state_by_clip[clip_id]
            try:
                action_json = Path(str(info["action_json"]))
                raw_frames = [int(v) for v in (info["raw_frames"] or [])]
                if not raw_frames:
                    raise ValueError("empty map_memory_raw_frame_indices")

                frames = json.loads(action_json.read_text())
                if not isinstance(frames, list):
                    raise TypeError("action JSON must contain a top-level list")

                interaction = validate_interaction(
                    build_interaction_channels(frames, raw_frames),
                    clip_id=clip_id,
                    frame_count=len(raw_frames),
                )

                src_npz = args.state_manifest.parent / Path(str(state_row["state_cache"]))
                with np.load(src_npz, allow_pickle=False) as data:
                    payload = {k: data[k] for k in STATE_FIELDS}
                    payload["raw_frames"] = (
                        data["raw_frames"] if "raw_frames" in data.files
                        else np.asarray(raw_frames, dtype=np.int32)
                    )
                for key in EVENT_FIELDS + ONEHOT_FIELDS + ("ego_weapon_slot_name",):
                    payload[key] = interaction[key]

                if payload["ego_alive"].shape[0] != len(raw_frames):
                    raise ValueError(
                        f"state frame count {payload['ego_alive'].shape[0]} "
                        f"!= raw_frames {len(raw_frames)}"
                    )

                bucket = state_dir / clip_id[:2]
                bucket.mkdir(parents=True, exist_ok=True)
                dst = bucket / f"{clip_id}.npz"
                np.savez_compressed(dst, **payload)

                counts = event_counts(interaction)
                for key, value in counts.items():
                    totals[key] += value

                out_handle.write(json.dumps({
                    "kind": "state_channels_v1_record",
                    "clip_id": clip_id,
                    "state_cache": str(dst),
                    "source_cache_manifest": info["source_cache_manifest"],
                    "source_state_cache": str(src_npz),
                    "latent_cache": info["latent_cache"],
                    "action_json": str(action_json),
                    "map_memory_raw_frame_indices": raw_frames,
                    "weapon_vocabulary": list(WEAPON_VOCABULARY),
                    "weapon_vocab_size": len(WEAPON_VOCABULARY),
                    "shape": {k: list(np.asarray(payload[k]).shape)
                              for k in STATE_FIELDS + EVENT_FIELDS + ONEHOT_FIELDS},
                    "channels": list(state_row.get("channels", [])) + list(INTERACTION_CHANNEL_NAMES),
                    "metadata_arrays": ["ego_weapon_slot_name"],
                    "stats": {"interaction_event_counts": counts},
                }) + "\n")
                written += 1
            except Exception as exc:  # noqa: BLE001 - one bad clip must not kill the build
                skipped.append({"clip_id": clip_id, "error": f"{type(exc).__name__}: {exc}"})

            if args.progress_every and (index + 1) % args.progress_every == 0:
                rate = (index + 1) / max(1e-6, time.time() - started)
                print(f"  {index+1}/{len(selected)}  written={written} "
                      f"skipped={len(skipped)}  {rate:.1f} clip/s", flush=True)

    status = "pass" if written > 0 and not skipped else ("partial" if written > 0 else "fail")
    report = {
        "kind": "interaction_sidecar_report_v0",
        "status": status,
        "state_manifest": str(args.state_manifest),
        "cache_manifests": [str(p) for p in args.cache_manifest],
        "out_manifest": str(manifest_path),
        "selected": len(selected),
        "written_rows": written,
        "skipped_rows": len(skipped),
        "skipped_examples": skipped[:20],
        "interaction_event_totals": totals,
        "weapon_vocab_size": len(WEAPON_VOCABULARY),
        "elapsed_sec": round(time.time() - started, 2),
    }
    (out_root / args.report_name).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1)[:2000], flush=True)
    if status == "fail":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
