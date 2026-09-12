#!/usr/bin/env python3
"""Build the event-bucketed val/test eval subset for the M2 interaction ablation.

Joins aligned-cache manifests (clip_id + split + clip_dir) with the s6 interaction
sidecar manifest (stats.interaction_event_counts per clip), buckets clips into
fire / reload / weapon_switch / throw / no_event, and picks a deterministic,
purity-preferring sample per (pair, split, bucket):

  - a clip may qualify for several event buckets; each bucket ranks its candidates by
    (other-event count ascending, clip_id) so the cleanest examples come first — no RNG,
    same subset on every rerun (a seed only enters if you pass --shuffle-seed).
  - no_event = all four event counts zero.

Output: one JSONL row per selected (pair, split, bucket, clip). CPU-only, read-only
inputs; writes ONLY --out.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

EVENT_KEYS = ("fire", "reload", "weapon_switch", "throw")
BUCKETS = EVENT_KEYS + ("no_event",)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_sidecar_counts(path: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            counts = (row.get("stats") or {}).get("interaction_event_counts") or {}
            out[str(row.get("clip_id"))] = {k: int(counts.get(k, 0) or 0) for k in EVENT_KEYS}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", nargs=3, action="append", metavar=("NAME", "MAP_MANIFEST", "CACHE_MANIFEST"),
                    required=True, help="repeatable: dataset pair name + map manifest + cache manifest")
    ap.add_argument("--state-cache-manifest", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    ap.add_argument("--buckets", nargs="+", default=list(BUCKETS), choices=list(BUCKETS))
    ap.add_argument("--per-bucket", type=int, default=1,
                    help="clips per (pair, split, bucket); default 1 keeps GPU cost bounded")
    ap.add_argument("--min-event-count", type=int, default=1,
                    help="minimum count of the bucket's event within the clip")
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="optional: shuffle candidates before purity ranking (default: fully deterministic)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    side = load_sidecar_counts(args.state_cache_manifest)
    print(json.dumps({"event": "sidecar_loaded", "rows": len(side)}), flush=True)

    selected: list[dict[str, Any]] = []
    summary: dict[str, int] = {}
    for name, map_manifest, cache_manifest in args.pair:
        records = read_jsonl(Path(cache_manifest))
        for split in args.splits:
            split_records = [
                r for r in records
                if str(r.get("map_memory_split", r.get("split", ""))) == split
                and str(r.get("clip_id")) in side
            ]
            for bucket in args.buckets:
                candidates = []
                for r in split_records:
                    cid = str(r.get("clip_id"))
                    counts = side[cid]
                    total = sum(counts.values())
                    if bucket == "no_event":
                        if total != 0:
                            continue
                        purity_other = 0
                    else:
                        if counts[bucket] < args.min_event_count:
                            continue
                        purity_other = total - counts[bucket]
                    candidates.append((purity_other, cid, counts, r))
                if args.shuffle_seed is not None:
                    rng = random.Random(f"{args.shuffle_seed}:{name}:{split}:{bucket}")
                    rng.shuffle(candidates)
                candidates.sort(key=lambda t: (t[0], t[1]))
                picked = candidates[: args.per_bucket]
                summary[f"{name}/{split}/{bucket}"] = len(picked)
                if len(picked) < args.per_bucket:
                    print(json.dumps({
                        "event": "bucket_underfilled",
                        "pair": name, "split": split, "bucket": bucket,
                        "wanted": args.per_bucket, "got": len(picked),
                        "candidates": len(candidates),
                    }), flush=True)
                for purity_other, cid, counts, r in picked:
                    selected.append({
                        "kind": "interaction_eval_subset_row_v1",
                        "pair": name,
                        "map_manifest": str(map_manifest),
                        "cache_manifest": str(cache_manifest),
                        "split": split,
                        "bucket": bucket,
                        "clip_id": cid,
                        "event_counts": counts,
                        "purity_other_events": purity_other,
                        "clip_dir": r.get("clip_dir"),
                    })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "event": "subset_written",
        "out": str(args.out),
        "rows": len(selected),
        "per_bucket": args.per_bucket,
        "summary": summary,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
