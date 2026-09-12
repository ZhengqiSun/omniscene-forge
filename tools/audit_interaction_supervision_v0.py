#!/usr/bin/env python3
"""Audit whether event-rich clips provide precise, usable action supervision."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


def event_key(event: dict[str, Any]) -> str:
    return str(event.get("relation")) if event.get("type") == "player_death" else str(event.get("type"))


def visible_at(vis: dict[str, Any], player_indices: list[int], frame: int) -> tuple[bool, int | None]:
    best = None
    for player_index in player_indices:
        for item in (vis.get(str(player_index), {}) or {}).get("ranges") or []:
            start, end = map(int, item["range"])
            distance = 0 if start <= frame <= end else min(abs(frame - start), abs(frame - end))
            best = distance if best is None else min(best, distance)
    return best == 0, best


def audit_events(path: Path, *, visibility_sample: int, seed: int) -> dict[str, Any]:
    primary = Counter()
    labels = Counter()
    totals = Counter()
    inside = Counter()
    sampled = Counter()
    rows = 0
    firefights_inside: list[tuple[dict[str, Any], dict[str, Any]]] = []
    rng = random.Random(seed)

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            primary[str(row.get("event50h_primary_label"))] += 1
            labels.update(str(value) for value in row.get("event50h_labels") or [])
            for event in row.get("event50h_events") or []:
                key = event_key(event)
                offset = int(event.get("raw_window_offset", int(event["frame"]) - int(row["raw_start"])))
                totals[key] += 1
                if 0 <= offset <= 162:
                    inside[key] += 1
                    sampled[key] += int(offset % 2 == 0)
                    if event.get("type") == "firefight":
                        item = (row, event)
                        if len(firefights_inside) < visibility_sample:
                            firefights_inside.append(item)
                        else:
                            index = rng.randrange(inside["firefight"])
                            if index < visibility_sample:
                                firefights_inside[index] = item

    visibility = Counter()
    distance_hist = Counter()
    for row, event in firefights_inside:
        with open(row["player_visibility"], encoding="utf-8") as handle:
            vis = json.load(handle)
        same, distance = visible_at(vis, [int(v) for v in event.get("visible_enemy_indices") or []], int(event["frame"]))
        visibility["same_frame"] += int(same)
        visibility["within_1"] += int(distance is not None and distance <= 1)
        visibility["within_2"] += int(distance is not None and distance <= 2)
        visibility["within_4"] += int(distance is not None and distance <= 4)
        visibility["missing_enemy_ranges"] += int(distance is None)
        if distance is not None:
            distance_hist[min(distance, 32)] += 1

    event_table = {}
    for key, total in totals.most_common():
        in_clip = inside[key]
        exact = sampled[key]
        event_table[key] = {
            "manifest_events": total,
            "inside_final_clip": in_clip,
            "inside_rate": in_clip / total if total else 0.0,
            "exactly_on_16fps_sample": exact,
            "exact_sample_rate_given_inside": exact / in_clip if in_clip else 0.0,
        }
    n = len(firefights_inside)
    return {
        "source_manifest": str(path.resolve()),
        "rows": rows,
        "primary_label_counts": dict(primary.most_common()),
        "label_counts": dict(labels.most_common()),
        "event_alignment": event_table,
        "firefight_visibility_audit": {
            "sample_size": n,
            "seed": seed,
            "criterion_in_source": "any fire in window AND any enemy visible anywhere in window",
            "enemy_visible_same_fire_frame": visibility["same_frame"],
            "enemy_visible_same_fire_frame_rate": visibility["same_frame"] / n if n else 0.0,
            "enemy_visible_within_1_raw_frame_rate": visibility["within_1"] / n if n else 0.0,
            "enemy_visible_within_2_raw_frames_rate": visibility["within_2"] / n if n else 0.0,
            "enemy_visible_within_4_raw_frames_rate": visibility["within_4"] / n if n else 0.0,
            "missing_enemy_ranges": visibility["missing_enemy_ranges"],
            "nearest_visible_distance_hist_capped32": {str(k): v for k, v in sorted(distance_hist.items())},
        },
    }


def audit_aligned(path: Path) -> dict[str, Any]:
    rows = 0
    clips: set[str] = set()
    splits = Counter()
    labels = Counter()
    primary = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            clips.add(str(row["clip_id"]))
            splits[str(row.get("map_memory_split"))] += 1
            labels.update(str(value) for value in row.get("event50h_labels") or [])
            primary[str(row.get("event50h_primary_label"))] += 1
    return {
        "aligned_cache_manifest": str(path.resolve()),
        "rows": rows,
        "unique_clips": len(clips),
        "duplicate_weight_rows": rows - len(clips),
        "split_counts": dict(splits.most_common()),
        "weighted_label_counts": dict(labels.most_common()),
        "weighted_primary_label_counts": dict(primary.most_common()),
    }


def audit_trainer(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    signals = ["action_json", "fire", "reload", "weapon_slot", "ammo_magazine", "look_dx", "look_dy"]
    mentions = {name: len(re.findall(rf"\b{re.escape(name)}\b", text)) for name in signals}
    return {
        "trainer": str(path.resolve()),
        "signal_identifier_mentions": mentions,
        "explicit_action_signal_mentioned": any(mentions.values()),
        "note": "Lexical fail-fast only; model-forward wiring still requires code review.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-source-manifest", type=Path, required=True)
    parser.add_argument("--aligned-cache-manifest", type=Path)
    parser.add_argument("--trainer", type=Path)
    parser.add_argument("--visibility-sample", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "kind": "interaction_supervision_audit_v0",
        "events": audit_events(
            args.event_source_manifest,
            visibility_sample=args.visibility_sample,
            seed=args.seed,
        ),
    }
    if args.aligned_cache_manifest:
        report["aligned_training_cache"] = audit_aligned(args.aligned_cache_manifest)
    if args.trainer:
        report["trainer_lexical_audit"] = audit_trainer(args.trainer)
    report["verdict"] = {
        "event50h_role": "event-rich sampling pool, not a precise action-conditioning cache",
        "required_next_gate": "build latent-aligned action cache and validate paired fire/no-fire response before full training",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {args.output}")
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
