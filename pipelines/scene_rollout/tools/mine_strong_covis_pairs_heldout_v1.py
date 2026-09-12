#!/usr/bin/env python3
from __future__ import annotations
from runtime_paths import source_path

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(str(source_path('scene', '')))
DEFAULT_CANDIDATE_POOL = ROOT / "output/memory_v2_all188_diversity_stride_manifest_20260629_v0/candidate_pool_cache_v0.jsonl"
DEFAULT_TRAIN_CACHE = ROOT / "output/memory_v2_all188_diversity_stride_manifest_20260629_v0/train_inputs_tier69h_v0/tier69h_aligned_cache_memorymask_view_v0.jsonl"
DEFAULT_OUT_DIR = ROOT / "output/symmvc_strong_pairs_heldout_v1"


PLAYER_RE = re.compile(r"player_(\d+)_")


def stable_bucket(text: str, modulo: int = 10_000) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulo


def split_for_match(match_id: str, *, seed: int, val_fraction: float, test_fraction: float) -> tuple[str, int]:
    val_cut = int(val_fraction * 10_000)
    test_cut = int((val_fraction + test_fraction) * 10_000)
    bucket = stable_bucket(f"{seed}|{match_id}")
    split = "val" if bucket < val_cut else "test" if bucket < test_cut else "train"
    return split, bucket


def player_index(stem: str) -> int | None:
    m = PLAYER_RE.search(stem or "")
    return int(m.group(1)) if m else None


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def visibility_ranges(path: str, ego_idx: int, start: int, end: int) -> dict[int, dict[str, Any]]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[int, dict[str, Any]] = {}
    for k, v in obj.items():
        try:
            target = int(k)
        except Exception:
            continue
        if target == ego_idx:
            continue
        segs = []
        max_pix = 0.0
        raw_hits: set[int] = set()
        for rr in v.get("ranges") or []:
            pair = rr.get("range")
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            a, b = int(pair[0]), int(pair[1])
            lo, hi = max(a, start), min(b, end)
            if hi < lo:
                continue
            pix = float(rr.get("pixel_percent_max") or 0.0)
            max_pix = max(max_pix, pix)
            segs.append((lo, hi, pix))
            raw_hits.update(range(lo, hi + 1))
        if raw_hits:
            out[target] = {"ranges": segs, "frames": raw_hits, "pixel_percent_max": max_pix}
    return out


def local_index(raw: int, start: int, end: int, n_frames: int) -> int | None:
    if raw < start or raw > end or n_frames <= 0:
        return None
    li = int(round((raw - start) * (n_frames - 1) / max(1, end - start)))
    return li if 0 <= li < n_frames else None


def build_pair(row_a: dict[str, Any], row_b: dict[str, Any], vis_a: dict[int, Any], vis_b: dict[int, Any]) -> dict[str, Any] | None:
    idx_a = player_index(row_a.get("player_stem", ""))
    idx_b = player_index(row_b.get("player_stem", ""))
    if idx_a is None or idx_b is None or idx_a == idx_b:
        return None
    a_sees_b = vis_a.get(idx_b)
    b_sees_a = vis_b.get(idx_a)
    if not a_sees_b or not b_sees_a:
        return None

    start = max(int(row_a["frame_count_start"]), int(row_b["frame_count_start"]))
    end = min(int(row_a["frame_count_end"]), int(row_b["frame_count_end"]))
    if end < start:
        return None

    mutual_raw = sorted(a_sees_b["frames"] & b_sees_a["frames"] & set(range(start, end + 1)))
    if not mutual_raw:
        return None

    n_frames = min(int(row_a.get("alignment_video_frames") or 0), int(row_b.get("alignment_video_frames") or 0))
    if n_frames <= 1:
        n_frames = min(len(row_a.get("raw_indices") or []), len(row_b.get("raw_indices") or [])) or 81

    frames = []
    seen_li: set[int] = set()
    pix_percent_ab = float(a_sees_b["pixel_percent_max"])
    pix_percent_ba = float(b_sees_a["pixel_percent_max"])
    # Visibility sidecars store percent of 1280x720 image area; the SymMVC gate is in pixels.
    image_pixels = 1280 * 720
    pix_ab = pix_percent_ab * image_pixels / 100.0
    pix_ba = pix_percent_ba * image_pixels / 100.0
    for raw in mutual_raw:
        li = local_index(raw, start, end, n_frames)
        if li is not None and li not in seen_li:
            seen_li.add(li)
            frames.append([int(li), int(raw), round(pix_ab, 4), round(pix_ba, 4)])
    if not frames:
        return None

    min_pix = min(pix_ab, pix_ba)
    score = min_pix * len(frames) + 0.01 * len(mutual_raw)
    match_id = row_a.get("game_id") or row_a.get("match_id")
    return {
        "game": match_id,
        "match_id": match_id,
        "ep": row_a.get("episode"),
        "episode": row_a.get("episode"),
        "window": [int(start), int(end)],
        "egoA": row_a.get("player_stem"),
        "egoB": row_b.get("player_stem"),
        "idxA": int(idx_a),
        "idxB": int(idx_b),
        "clipA_dir": row_a.get("clip_dir"),
        "clipB_dir": row_b.get("clip_dir"),
        "a_sees_b_maxpix": round(pix_ab, 4),
        "b_sees_a_maxpix": round(pix_ba, 4),
        "a_sees_b_max_pixel_percent": round(pix_percent_ab, 6),
        "b_sees_a_max_pixel_percent": round(pix_percent_ba, 6),
        "pixel_count_conversion": {"width": 1280, "height": 720, "source_field": "player_visibility.*.ranges[].pixel_percent_max"},
        "n_mutual_covis_frames": int(len(frames)),
        "mutual_covis_frames": frames,
        "entry_kind": "strong_covis_heldout_mined_v1",
        "clipA_id": row_a.get("clip_id"),
        "clipB_id": row_b.get("clip_id"),
        "min_mutual_maxpix": round(min_pix, 4),
        "raw_mutual_covis_frames": int(len(mutual_raw)),
        "score": round(score, 6),
        "source_candidate_pool_A": {
            "mp4": row_a.get("mp4"),
            "player_visibility": row_a.get("player_visibility"),
            "raw_start": row_a.get("raw_start"),
            "frame_count_start": row_a.get("frame_count_start"),
            "frame_count_end": row_a.get("frame_count_end"),
            "map_memory_split": row_a.get("map_memory_split"),
            "map_memory_split_key": row_a.get("map_memory_split_key"),
        },
        "source_candidate_pool_B": {
            "mp4": row_b.get("mp4"),
            "player_visibility": row_b.get("player_visibility"),
            "raw_start": row_b.get("raw_start"),
            "frame_count_start": row_b.get("frame_count_start"),
            "frame_count_end": row_b.get("frame_count_end"),
            "map_memory_split": row_b.get("map_memory_split"),
            "map_memory_split_key": row_b.get("map_memory_split_key"),
        },
    }


def pass_gate(pair: dict[str, Any], min_frames: int, min_pixels: float) -> bool:
    return int(pair.get("n_mutual_covis_frames", 0)) >= min_frames and float(pair.get("min_mutual_maxpix", 0.0)) >= min_pixels


def file_info(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {"path": str(path), "size": st.st_size, "mtime_unix": st.st_mtime, "mtime_local": time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(st.st_mtime))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-pool", type=Path, default=DEFAULT_CANDIDATE_POOL)
    ap.add_argument("--train-cache", type=Path, default=DEFAULT_TRAIN_CACHE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--seed", type=int, default=20260531)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--test-fraction", type=float, default=0.05)
    ap.add_argument("--min-mutual-frames", type=int, default=4)
    ap.add_argument("--min-mutual-pixels", type=float, default=50.0)
    args = ap.parse_args()

    train_matches: set[str] = set()
    train_cache_rows = 0
    train_cache_field_counts = Counter()
    for row in read_jsonl(args.train_cache):
        train_cache_rows += 1
        match_id = str(row.get("game_id") or row.get("match_id"))
        train_matches.add(match_id)
        train_cache_field_counts[(row.get("map_memory_split"), row.get("map_memory_split_key"))] += 1

    split_rows = []
    split_by_name: dict[str, list[str]] = defaultdict(list)
    train_rule_matches: set[str] = set()
    heldout_matches: set[str] = set()
    for match_id in sorted(train_matches):
        split, bucket = split_for_match(match_id, seed=args.seed, val_fraction=args.val_fraction, test_fraction=args.test_fraction)
        split_rows.append({"match_id": match_id, "split": split, "bucket": bucket})
        split_by_name[split].append(match_id)
        if split == "train":
            train_rule_matches.add(match_id)
        else:
            heldout_matches.add(match_id)

    rows_by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    candidate_rows_total = 0
    candidate_rows_heldout = 0
    candidate_rows_train_rule = 0
    candidate_pool_field_counts = Counter()
    for row in read_jsonl(args.candidate_pool):
        candidate_rows_total += 1
        match_id = str(row.get("game_id") or row.get("match_id"))
        candidate_pool_field_counts[(row.get("map_memory_split"), row.get("map_memory_split_key"))] += 1
        if match_id in train_rule_matches:
            candidate_rows_train_rule += 1
            continue
        if match_id not in heldout_matches:
            continue
        candidate_rows_heldout += 1
        rows_by_group[(match_id, str(row.get("episode")))].append(row)

    visibility_cache: dict[tuple[str, int, int, int], dict[int, Any]] = {}

    def get_vis(row: dict[str, Any]) -> dict[int, Any]:
        idx = player_index(row.get("player_stem", ""))
        if idx is None:
            return {}
        key = (str(row.get("player_visibility")), idx, int(row["frame_count_start"]), int(row["frame_count_end"]))
        if key not in visibility_cache:
            visibility_cache[key] = visibility_ranges(key[0], key[1], key[2], key[3])
        return visibility_cache[key]

    pairs_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    overlap_pairs_examined = 0
    group_sizes = Counter()
    for group, rows in rows_by_group.items():
        rows = sorted(rows, key=lambda r: (r.get("player_stem") or "", int(r.get("frame_count_start") or 0), r.get("clip_id") or ""))
        group_sizes[len(rows)] += 1
        by_player: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_player[str(row.get("player_stem"))].append(row)
        stems = sorted(by_player)
        for i, stem_a in enumerate(stems):
            for stem_b in stems[i + 1:]:
                a_rows = by_player[stem_a]
                b_rows = by_player[stem_b]
                j0 = 0
                for ra in a_rows:
                    a0, a1 = int(ra["frame_count_start"]), int(ra["frame_count_end"])
                    while j0 < len(b_rows) and int(b_rows[j0]["frame_count_end"]) < a0:
                        j0 += 1
                    j = j0
                    while j < len(b_rows) and int(b_rows[j]["frame_count_start"]) <= a1:
                        rb = b_rows[j]
                        overlap_pairs_examined += 1
                        p = build_pair(ra, rb, get_vis(ra), get_vis(rb))
                        if p:
                            key = tuple(sorted([str(p["clipA_id"]), str(p["clipB_id"])]))
                            old = pairs_by_key.get(key)
                            if old is None or float(p["score"]) > float(old["score"]):
                                pairs_by_key[key] = p
                        j += 1

    candidates = sorted(
        pairs_by_key.values(),
        key=lambda p: (-float(p["score"]), -float(p["min_mutual_maxpix"]), -int(p["n_mutual_covis_frames"]), str(p["clipA_id"]), str(p["clipB_id"])),
    )
    strict = [p for p in candidates if pass_gate(p, args.min_mutual_frames, args.min_mutual_pixels)]
    thresholds = []
    for pix in [50.0, 40.0, 30.0, 25.0]:
        rows = [p for p in candidates if pass_gate(p, args.min_mutual_frames, pix)]
        thresholds.append({
            "min_mutual_frames": args.min_mutual_frames,
            "min_mutual_pixels": pix,
            "count": len(rows),
            "by_match": dict(sorted(Counter(p["match_id"] for p in rows).items())),
        })

    windows = []
    seen_windows = set()
    for p in strict:
        for side in ["A", "B"]:
            src = p[f"source_candidate_pool_{side}"]
            item = {
                "pair_id": f"{p['clipA_id']}__{p['clipB_id']}",
                "side": side,
                "match_id": p["match_id"],
                "episode": p["episode"],
                "ego": p[f"ego{side}"],
                "clip_id": p[f"clip{side}_id"],
                "clip_dir": p[f"clip{side}_dir"],
                "mp4": src["mp4"],
                "frame_count_start": src["frame_count_start"],
                "frame_count_end": src["frame_count_end"],
                "raw_start": src["raw_start"],
            }
            key = (item["clip_id"], item["pair_id"], item["side"])
            if key not in seen_windows:
                seen_windows.add(key)
                windows.append(item)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pair_path = args.out_dir / "pair_index_strong_covis_heldout_v1.json"
    report_path = args.out_dir / "strong_covis_heldout_v1_report.json"
    md_path = args.out_dir / "strong_covis_heldout_v1_report.md"
    windows_path = args.out_dir / "window_manifest_strong_covis_heldout_v1.jsonl"
    match_split_path = args.out_dir / "heldout_match_split_v1.json"

    pair_path.write_text(json.dumps(strict, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    with windows_path.open("w", encoding="utf-8") as f:
        for row in windows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    match_split = {
        "kind": "heldout_match_split_v1",
        "split_rule_source": "tools/map_memory_training_data_v0.py split_samples: stable_bucket(f'{seed}|{match_id}') with split_key='match'",
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "val_cut": int(args.val_fraction * 10_000),
        "test_cut": int((args.val_fraction + args.test_fraction) * 10_000),
        "source_train_cache": file_info(args.train_cache),
        "train_cache_rows": train_cache_rows,
        "match_count_total": len(train_matches),
        "split_counts_match": {k: len(v) for k, v in sorted(split_by_name.items())},
        "matches": split_rows,
        "heldout_matches": sorted(heldout_matches),
        "train_matches": sorted(train_rule_matches),
    }
    match_split_path.write_text(json.dumps(match_split, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    report = {
        "kind": "strong_covis_heldout_mining_v1",
        "created_at_unix": time.time(),
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "inputs": {"candidate_pool": file_info(args.candidate_pool), "train_cache": file_info(args.train_cache)},
        "output_files": {
            "pair_index": str(pair_path),
            "report_json": str(report_path),
            "report_md": str(md_path),
            "window_manifest": str(windows_path),
            "match_split": str(match_split_path),
        },
        "split_basis": {
            "code_file": "tools/map_memory_training_data_v0.py",
            "function": "split_samples",
            "split_key": "match",
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
            "heldout_definition": "val+test matches by stable_bucket(f'{seed}|{match_id}')",
            "train_cache_field_counts": {str(k): v for k, v in sorted(train_cache_field_counts.items())},
            "candidate_pool_field_counts": {str(k): v for k, v in sorted(candidate_pool_field_counts.items())},
        },
        "counts": {
            "train_cache_rows": train_cache_rows,
            "train_cache_match_count": len(train_matches),
            "candidate_rows_total": candidate_rows_total,
            "candidate_rows_train_rule_excluded": candidate_rows_train_rule,
            "candidate_rows_heldout": candidate_rows_heldout,
            "heldout_match_count": len(heldout_matches),
            "heldout_episode_groups": len(rows_by_group),
            "overlap_pairs_examined": overlap_pairs_examined,
            "candidate_pairs_with_any_mutual": len(candidates),
            "strict_pairs_total": len(strict),
            "window_manifest_rows": len(windows),
            "visibility_files_read": len({k[0] for k in visibility_cache}),
        },
        "heldout_matches_by_split": {k: sorted(v) for k, v in sorted(split_by_name.items()) if k != "train"},
        "train_match_count_by_rule": len(train_rule_matches),
        "strict_pairs_by_match": dict(sorted(Counter(p["match_id"] for p in strict).items())),
        "threshold_sensitivity": thresholds,
        "group_sizes": dict(sorted(group_sizes.items())),
        "strict_pair_stats": [
            {
                "rank": i + 1,
                "pair_id": f"{p['clipA_id']}__{p['clipB_id']}",
                "match_id": p["match_id"],
                "episode": p["episode"],
                "egoA": p["egoA"],
                "egoB": p["egoB"],
                "window": p["window"],
                "n_mutual_covis_frames": p["n_mutual_covis_frames"],
                "raw_mutual_covis_frames": p["raw_mutual_covis_frames"],
                "a_sees_b_maxpix": p["a_sees_b_maxpix"],
                "b_sees_a_maxpix": p["b_sees_a_maxpix"],
                "min_mutual_maxpix": p["min_mutual_maxpix"],
                "score": p["score"],
            }
            for i, p in enumerate(strict)
        ],
        "top_candidate_pairs_any_mutual": candidates[:100],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    lines = [
        "# Strong Covis Heldout Mining v1",
        "",
        f"candidate_pool: `{args.candidate_pool}`",
        f"train_cache: `{args.train_cache}`",
        "",
        "## Split",
        "",
        "Rule: `tools/map_memory_training_data_v0.py split_samples`, `split_key=match`, "
        f"`stable_bucket(f\"{args.seed}|{{match_id}}\")`, val < {int(args.val_fraction * 10000)}, test < {int((args.val_fraction + args.test_fraction) * 10000)}.",
        "",
        f"Held-out matches: {len(heldout_matches)} (val {len(split_by_name.get('val', []))}, test {len(split_by_name.get('test', []))}); train matches excluded: {len(train_rule_matches)}.",
        "",
        "## Counts",
        "",
        f"candidate rows total: {candidate_rows_total}",
        f"held-out candidate rows: {candidate_rows_heldout}",
        f"overlap pairs examined: {overlap_pairs_examined}",
        f"candidate pairs with any mutual covis: {len(candidates)}",
        f"strict pairs total: {len(strict)}",
        "",
        "## Threshold Sensitivity",
        "",
        "| min_pixels | count |",
        "|---:|---:|",
    ]
    for t in thresholds:
        lines.append(f"| {t['min_mutual_pixels']:.1f} | {t['count']} |")
    lines += ["", "## Strict Pairs By Match", "", "| match_id | count |", "|---|---:|"]
    for match_id, count in sorted(Counter(p["match_id"] for p in strict).items()):
        lines.append(f"| `{match_id}` | {count} |")
    lines += ["", "## Top Strict Pairs", "", "| rank | pair | match | ep | frames | min_pix | a_pix | b_pix |", "|---:|---|---|---|---:|---:|---:|---:|"]
    for i, p in enumerate(strict[:50], 1):
        lines.append(f"| {i} | `{p['clipA_id']}__{p['clipB_id']}` | `{p['match_id']}` | {p['episode']} | {p['n_mutual_covis_frames']} | {p['min_mutual_maxpix']} | {p['a_sees_b_maxpix']} | {p['b_sees_a_maxpix']} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"WROTE {pair_path} strict_pairs={len(strict)}")
    print(f"WROTE {report_path}")
    print(f"WROTE {windows_path} rows={len(windows)}")
    print(f"WROTE {match_split_path} heldout_matches={len(heldout_matches)}")


if __name__ == "__main__":
    main()
