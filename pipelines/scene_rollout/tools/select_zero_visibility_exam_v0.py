#!/usr/bin/env python3
"""Select a deterministic match-balanced zero-visibility held-out exam."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def rank(seed: int, clip_id: str) -> str:
    return hashlib.sha256(f"{seed}|{clip_id}".encode("utf-8")).hexdigest()


def enrich_context(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    frames = []
    for key in ("latent_frames", "positive_latent_frames"):
        values = output.get(key)
        if isinstance(values, list):
            frames = [int(value) for value in values]
            break
    if len(frames) != 21:
        raise ValueError(f"{output.get('clip_id')}: expected 21 latent frames, got {len(frames)}")
    output.update({
        "latent_frame_count": len(frames),
        "latent_frames": frames,
        "positive_latent_frame_count": len(frames),
        "positive_latent_frames": frames,
        "selection_role": "context",
        "map_memory_selection_roles": ["context"] * len(frames),
        "map_memory_positive_frames": 0,
        "map_memory_context_frames": len(frames),
    })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--per-match", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20_260_716)
    parser.add_argument("--expected-matches", type=int, default=14)
    args = parser.parse_args()

    rows = read_jsonl(args.input_manifest)
    by_match: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failures = []
    for row in rows:
        clip_id = str(row.get("clip_id"))
        match_value = row.get("game_id")
        if not match_value:
            failures.append(f"{clip_id}: missing game_id")
            continue
        match_id = str(match_value)
        split = str(row.get("map_memory_split"))
        roles = row.get("map_memory_selection_roles", [])
        if split not in {"val", "test"}:
            failures.append(f"{clip_id}: non-heldout split {split}")
        if ("selection_role" in row and row.get("selection_role") != "context") or (roles and set(roles) != {"context"}):
            failures.append(f"{clip_id}: non-context role")
        if "max_visibility_pixel_percent" not in row:
            failures.append(f"{clip_id}: missing visibility summary")
        elif float(row["max_visibility_pixel_percent"] or 0.0) != 0.0:
            failures.append(f"{clip_id}: nonzero visibility percent")
        by_match[match_id].append(row)

    if len(by_match) != args.expected_matches:
        failures.append(f"heldout matches {len(by_match)} != {args.expected_matches}")
    selected = []
    match_selection_counts = {}
    for match_id in sorted(by_match):
        candidates = sorted(by_match[match_id], key=lambda row: rank(args.seed, str(row["clip_id"])))
        picked = [enrich_context(row) for row in candidates[: args.per_match]]
        if len(picked) != args.per_match:
            failures.append(f"{match_id}: selected {len(picked)} != {args.per_match}")
        match_selection_counts[match_id] = len(picked)
        selected.extend(picked)
    selected.sort(key=lambda row: (str(row["game_id"]), rank(args.seed, str(row["clip_id"]))))
    clip_ids = [str(row["clip_id"]) for row in selected]
    if len(clip_ids) != len(set(clip_ids)):
        failures.append("duplicate selected clip ids")

    output_sha256 = None
    if not failures:
        write_jsonl(args.output_manifest, selected)
        output_sha256 = hashlib.sha256(args.output_manifest.read_bytes()).hexdigest()
    report: dict[str, Any] = {
        "kind": "zero_visibility_exam_selection_v0",
        "status": "pass" if not failures else "fail",
        "input_manifest": str(args.input_manifest.resolve()),
        "input_manifest_sha256": hashlib.sha256(args.input_manifest.read_bytes()).hexdigest(),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": output_sha256,
        "seed": args.seed,
        "policy": "sha256(seed|clip_id), first N per held-out match; no score-based cherry-picking",
        "per_match": args.per_match,
        "input_rows": len(rows),
        "heldout_match_count": len(by_match),
        "selected_rows": len(selected),
        "selected_val_rows": sum(row["map_memory_split"] == "val" for row in selected),
        "selected_test_rows": sum(row["map_memory_split"] == "test" for row in selected),
        "match_selection_counts": match_selection_counts,
        "selected_clip_ids": clip_ids,
        "failures": failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
