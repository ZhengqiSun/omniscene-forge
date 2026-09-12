#!/usr/bin/env python3
"""Summarize Map Memory dense/LingBot training-prep artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def summarize(base_dir: Path) -> dict[str, Any]:
    shard_rows: list[dict[str, Any]] = []
    shard_role_counts: Counter[str] = Counter()
    shard_reject_counts: Counter[str] = Counter()
    for report_path in sorted(base_dir.glob("fullsubset_shard_*/export_report_v0.json")):
        report = load_json(report_path)
        roles = report.get("role_counts") or {}
        rejects = report.get("rejected_clips") or {}
        shard_role_counts.update({str(k): int(v) for k, v in roles.items()})
        shard_reject_counts.update({str(k): int(v) for k, v in rejects.items()})
        shard_rows.append(
            {
                "shard": report_path.parent.name,
                "status": report.get("status"),
                "start_offset": report.get("start_offset"),
                "limit": report.get("limit"),
                "scanned_clips": report.get("scanned_clips"),
                "accepted_clips": report.get("accepted_clips"),
                "accepted_unique_samples": report.get("accepted_unique_samples"),
                "role_counts": roles,
                "readiness_status": report.get("readiness_status"),
                "export_failures": report.get("export_failures"),
                "rejected_clips": rejects,
            }
        )

    final_dirs = []
    for path in sorted(base_dir.glob("*/combined_report_v0.json")):
        report = load_json(path)
        manifest = Path(str(report.get("manifest", "")))
        aligned = Path(str(report.get("aligned_cache_manifest", "")))
        final_dirs.append(
            {
                "dir": str(path.parent),
                "status": report.get("status"),
                "accepted_clips": report.get("accepted_clips"),
                "accepted_unique_samples": report.get("accepted_unique_samples"),
                "role_counts": report.get("role_counts"),
                "split_counts": report.get("split_counts"),
                "readiness_status": report.get("readiness_status"),
                "failures": report.get("failures"),
                "manifest_exists": manifest.exists(),
                "aligned_rows": count_jsonl(aligned) if aligned.exists() else None,
            }
        )

    preflights = []
    preflight_paths = sorted(base_dir.glob("preflight*.json"))
    for child in sorted(base_dir.iterdir()) if base_dir.exists() else []:
        if child.is_dir() and child.name.startswith(("train", "final", "memory_dense")):
            preflight_paths.extend(sorted(child.glob("preflight*.json")))
            preflight_paths.extend(sorted(child.glob("preflight*_v0.json")))
    for path in preflight_paths:
        try:
            report = load_json(path)
        except Exception:
            continue
        if report.get("kind") != "memory_dense_adapter_train_preflight_v0":
            continue
        preflights.append(
            {
                "path": str(path),
                "status": report.get("status"),
                "record_count": report.get("record_count"),
                "checked_record_count": report.get("checked_record_count"),
                "release_gate": report.get("release_gate"),
                "optimizer_steps_run": report.get("optimizer_steps_run"),
            }
        )

    large_runs = []
    for path in sorted(base_dir.glob("runs/*/large_training_prep_summary_v0.json")):
        try:
            report = load_json(path)
        except Exception:
            continue
        if report.get("kind") != "memory_dense_lingbot_large_training_prep_summary_v0":
            continue
        filtered = report.get("filtered_manifest_report") or {}
        combined = report.get("combined_report") or {}
        preflight = report.get("preflight_report") or {}
        large_runs.append(
            {
                "path": str(path),
                "status": report.get("status"),
                "optimizer_steps_run": report.get("optimizer_steps_run"),
                "filtered_cache_rows": filtered.get("filtered_cache_rows"),
                "filtered_match_count": filtered.get("match_count"),
                "filtered_episode_count": filtered.get("episode_count"),
                "shard_status_counts": report.get("shard_status_counts"),
                "accepted_clips": combined.get("accepted_clips"),
                "accepted_unique_samples": combined.get("accepted_unique_samples"),
                "role_counts": combined.get("role_counts"),
                "split_counts": combined.get("split_counts"),
                "preflight_status": preflight.get("status"),
                "release_gate": preflight.get("release_gate"),
                "artifact_hashes": report.get("artifact_hashes"),
            }
        )

    return {
        "kind": "memory_dense_training_prep_summary_v0",
        "base_dir": str(base_dir),
        "shard_report_count": len(shard_rows),
        "shards": shard_rows,
        "aggregate_shard_role_counts": dict(sorted(shard_role_counts.items())),
        "aggregate_shard_rejected_clips": dict(sorted(shard_reject_counts.items())),
        "combined_releases": final_dirs,
        "preflights": preflights,
        "large_runs": large_runs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", type=Path, default=Path("output/memory_dense_adapter_v0"))
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()
    report = summarize(args.base_dir)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
