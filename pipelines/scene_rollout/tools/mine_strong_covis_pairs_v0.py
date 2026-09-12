#!/usr/bin/env python3
"""Mine strict mutual-covisibility SymMVC pairs from held-out benchmark clips.

The output pair index is intentionally compatible with tools/symmvc_v2_score_v0.py.
It only reads held-out candidate rows and GT player_visibility JSON files.
"""
from __future__ import annotations
from runtime_paths import source_path

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_HELDOUT = Path(
    str(source_path('project', 'output/rank64_final_verdict_20260629_bridge/v2_fixed_motion_heldout_benchmark_candidate_clipdir_corrected_v0.jsonl'))
)
DEFAULT_OUT_DIR = Path(str(source_path('scene', 'output/symmvc_strong_pairs_20260705_v0')))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def player_index_from_stem(stem: str) -> int | None:
    m = re.search(r"player_(\d+)_", stem or "")
    return int(m.group(1)) if m else None


def norm_ranges(label: dict[str, Any] | None) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for rr in (label or {}).get("visible_ranges") or []:
        if len(rr) != 2:
            continue
        a, b = int(rr[0]), int(rr[1])
        if b < a:
            continue
        out.append((a, b))
    return out


def label_for(row: dict[str, Any], target_index: int) -> dict[str, Any] | None:
    for lab in row.get("gt_visibility_labels") or []:
        try:
            if int(lab.get("player_index")) == int(target_index):
                return lab
        except Exception:
            continue
    return None


def visible_at(ranges: list[tuple[int, int]], raw: int) -> bool:
    return any(a <= raw <= b for a, b in ranges)


def local_index(raw: int, start: int, end: int, n_frames: int) -> int | None:
    if raw < start or raw > end or n_frames <= 0:
        return None
    # Existing pair_index maps raw 654 in a 502..662/81-frame clip to local 76.
    li = int(round((raw - start) * (n_frames - 1) / max(1, end - start)))
    if 0 <= li < n_frames:
        return li
    return None


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_pair(row_a: dict[str, Any], row_b: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    idx_a = player_index_from_stem(row_a.get("player_stem", ""))
    idx_b = player_index_from_stem(row_b.get("player_stem", ""))
    if idx_a is None or idx_b is None or idx_a == idx_b:
        return None

    lab_ab = label_for(row_a, idx_b)
    lab_ba = label_for(row_b, idx_a)
    if not lab_ab or not lab_ba:
        return None

    start = max(int(row_a["frame_count_start"]), int(row_b["frame_count_start"]))
    end = min(int(row_a["frame_count_end"]), int(row_b["frame_count_end"]))
    if end < start:
        return None

    ranges_ab = norm_ranges(lab_ab)
    ranges_ba = norm_ranges(lab_ba)
    n_frames = min(int(row_a.get("positive_latent_frame_count") or 0), int(row_b.get("positive_latent_frame_count") or 0))
    if n_frames <= 1:
        n_frames = 81

    pix_ab = float(lab_ab.get("pixel_percent_max") or 0.0)
    pix_ba = float(lab_ba.get("pixel_percent_max") or 0.0)
    frames: list[list[float | int]] = []
    raw_mutual = 0
    seen_li: set[int] = set()
    for raw in range(start, end + 1):
        if visible_at(ranges_ab, raw) and visible_at(ranges_ba, raw):
            raw_mutual += 1
            li = local_index(raw, start, end, n_frames)
            if li is not None and li not in seen_li:
                seen_li.add(li)
                frames.append([int(li), int(raw), round(pix_ab, 4), round(pix_ba, 4)])
    frames.sort(key=lambda x: int(x[0]))
    if not frames:
        return None

    min_pix = min(pix_ab, pix_ba)
    score = min_pix * len(frames) + 0.01 * raw_mutual
    return {
        "game": row_a.get("match_id"),
        "ep": row_a.get("episode"),
        "window": [int(start), int(end)],
        "egoA": row_a.get("player_stem"),
        "egoB": row_b.get("player_stem"),
        "idxA": int(idx_a),
        "idxB": int(idx_b),
        "clipA_dir": row_a.get("clip_dir"),
        "clipB_dir": row_b.get("clip_dir"),
        "a_sees_b_maxpix": round(pix_ab, 4),
        "b_sees_a_maxpix": round(pix_ba, 4),
        "n_mutual_covis_frames": int(len(frames)),
        "mutual_covis_frames": frames,
        "benchmark_group_id": row_a.get("benchmark_group_id"),
        "benchmark_motion_class": row_a.get("benchmark_motion_class") or row_b.get("benchmark_motion_class"),
        "entry_kind": "strong_covis_heldout_mined_v0",
        "clipA_id": row_a.get("clip_id"),
        "clipB_id": row_b.get("clip_id"),
        "min_mutual_maxpix": round(min_pix, 4),
        "raw_mutual_covis_frames": int(raw_mutual),
        "score": round(score, 6),
        "source_heldout_jsonl": str(args.heldout_jsonl),
        "source_split_A": row_a.get("split"),
        "source_split_B": row_b.get("split"),
        "source_player_visibility_A": row_a.get("player_visibility"),
        "source_player_visibility_B": row_b.get("player_visibility"),
    }


def pass_gate(pair: dict[str, Any], min_frames: int, min_pixels: float) -> bool:
    return int(pair.get("n_mutual_covis_frames", 0)) >= min_frames and float(pair.get("min_mutual_maxpix", 0.0)) >= min_pixels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout-jsonl", type=Path, default=DEFAULT_HELDOUT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--min-mutual-frames", type=int, default=4)
    ap.add_argument("--min-mutual-pixels", type=float, default=50.0)
    ap.add_argument("--target-pairs", type=int, default=30)
    args = ap.parse_args()

    rows = read_rows(args.heldout_jsonl)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r.get("match_id"), r.get("episode"), int(r.get("frame_count_start", -1)), int(r.get("frame_count_end", -1)))].append(r)

    all_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for _g, items in groups.items():
        items = sorted(items, key=lambda r: r.get("clip_id") or "")
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                p = build_pair(items[i], items[j], args)
                if not p:
                    continue
                key = tuple(sorted([p["clipA_id"], p["clipB_id"]]))
                all_pairs[key] = p

    candidates = sorted(all_pairs.values(), key=lambda p: (-float(p["score"]), -float(p["min_mutual_maxpix"]), p["clipA_id"], p["clipB_id"]))
    strict = [p for p in candidates if pass_gate(p, args.min_mutual_frames, args.min_mutual_pixels)]

    thresholds = []
    for pix in [50.0, 40.0, 35.0, 30.0, 25.0, 20.0, 15.0, 10.0, 5.0, 2.0, 1.0, 0.0]:
        thresholds.append({
            "min_mutual_frames": args.min_mutual_frames,
            "min_mutual_pixels": pix,
            "count": sum(1 for p in candidates if pass_gate(p, args.min_mutual_frames, pix)),
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pair_index = strict[: args.target_pairs]
    pair_path = args.out_dir / "pair_index_strong_covis_v0.json"
    report_path = args.out_dir / "strong_covis_mining_report_v0.json"
    md_path = args.out_dir / "strong_covis_mining_report_v0.md"

    pair_path.write_text(json.dumps(pair_index, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        "kind": "strong_covis_pair_mining_v0",
        "created_at_unix": time.time(),
        "heldout_jsonl": str(args.heldout_jsonl),
        "output_pair_index": str(pair_path),
        "policy": "same match+episode+identical window among held-out benchmark clips; two different ego players; bidirectional gt_visibility_labels; local frame index compatible with existing SymMVC scorer",
        "gate": {"min_mutual_frames": args.min_mutual_frames, "min_mutual_pixels": args.min_mutual_pixels, "target_pairs": args.target_pairs},
        "input_rows": len(rows),
        "groups_total": len(groups),
        "groups_by_size": dict(sorted(Counter(len(v) for v in groups.values()).items())),
        "candidate_pairs_total": len(candidates),
        "strict_pairs_total": len(strict),
        "emitted_pairs": len(pair_index),
        "threshold_sensitivity": thresholds,
        "all_candidate_pairs": candidates,
        "emitted_pair_ids": [f"{p['clipA_id']}__{p['clipB_id']}" for p in pair_index],
    }
    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Strong Covis Pair Mining v0",
        "",
        f"heldout_jsonl: `{args.heldout_jsonl}`",
        f"input_rows: {len(rows)}; groups_total: {len(groups)}; candidate_pairs_total: {len(candidates)}",
        f"strict gate: mutual_frames >= {args.min_mutual_frames}, min bidirectional maxpix >= {args.min_mutual_pixels}",
        f"strict_pairs_total: {len(strict)}; emitted_pairs: {len(pair_index)}; target_pairs: {args.target_pairs}",
        "",
        "| min_pixels | count |",
        "|---:|---:|",
    ]
    for t in thresholds:
        lines.append(f"| {t['min_mutual_pixels']:.1f} | {t['count']} |")
    lines += ["", "| rank | clipA | clipB | frames | min_pix | a_pix | b_pix |", "|---:|---|---|---:|---:|---:|---:|"]
    for i, p in enumerate(candidates[:50], 1):
        lines.append(
            f"| {i} | {p['clipA_id']} | {p['clipB_id']} | {p['n_mutual_covis_frames']} | "
            f"{p['min_mutual_maxpix']} | {p['a_sees_b_maxpix']} | {p['b_sees_a_maxpix']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"WROTE {pair_path} pairs={len(pair_index)}")
    print(f"WROTE {report_path} candidates={len(candidates)} strict={len(strict)}")
    print(f"WROTE {md_path}")


if __name__ == "__main__":
    main()
