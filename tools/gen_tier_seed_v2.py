#!/usr/bin/env python3
# QXQ parameterized v2 of tools/gen_tier69h_seed_v0.py (qxq fixed copy, base HEAD 2e5c485).
# Changes vs gen_tier69h_seed_v0.py:
#   1. Full CLI parameterization: seed/target manifest, report, out-dir (tmp+fresh cache),
#      seed/target hours, candidate cache (read-only reuse, optional sha256 reconciliation),
#      extra exclude manifests, summary path. No hardcoded output paths; this script never
#      writes anywhere under the zhengqi tree (the shared candidate cache is only read).
#   2. Select-time exclusion fix: round_robin_select now filters seed raw_keys AND all raw_keys
#      loaded from --exclude-manifest files. Scan-time exclusion is ineffective for extra
#      excludes because the candidate cache already exists and the scan is skipped, so the
#      filtering must happen at selection time. Seed skips and extra-exclude skips are counted
#      separately.
#   3. Canonical match-disjoint split relabel kept verbatim (canonical_match_split_v0,
#      seed 20260531, map_memory_split_key='match').
#   4. New raw-interval partial-overlap audit: selected supplement windows vs ALL rows of every
#      --exclude-manifest (including off-grid raw_start), grouped by (game_id, episode,
#      player_stem), window = [raw_start, raw_start + 162). Policy: record counts and first
#      examples in the report, do NOT reject (consistent with the ~8.6% event<->tier partial
#      overlap precedent).
#   5. Report gains: exclude_manifests, excluded_at_select, partial_overlap_report,
#      candidate cache sha256 reconciliation fields, seed_hours, cli_args. All original report
#      fields are kept (seed prefix check is renamed tier20_* -> seed_* because the seed is now
#      arbitrary, same semantics).
"""Generate a seed-expanded v2 tier source manifest (parameterized, select-time exclusion)."""
from runtime_paths import source_path
import argparse
import bisect
import collections
import hashlib
import heapq
import json
import os
import shutil
import time
from pathlib import Path

DEFAULT_MAN = Path(str(source_path('scene', 'output/memory_v2_all188_diversity_stride_manifest_20260629_v0')))
DEFAULT_CACHE = DEFAULT_MAN / "candidate_pool_cache_v0.jsonl"
DEFAULT_SUMMARY = DEFAULT_MAN / "v2_all188_diversity_stride_manifest_summary_v0.json"

SESSION_HASH = "32f1644d4f42c29d"
FRAME_STRIDE_RAW = 162
RAW_OFFSET = 16
LATENT_STEP_RAW = 8
LATENT_COUNT = 21
RAW_FRAME_STEP = 2

from canonical_match_split_v0 import match_to_split


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sha256_path(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_key(obj):
    game_id = obj.get("game_id") or obj.get("match_id")
    episode = obj.get("episode")
    player_stem = obj.get("player_stem")
    raw_start = obj.get("raw_start")
    if game_id is None or episode is None or player_stem is None or raw_start is None:
        cid = obj.get("clip_id")
        if cid:
            return cid
        raise KeyError(f"cannot derive raw key from keys={sorted(obj.keys())[:20]}")
    return f"{game_id}|{episode}|{player_stem}|{int(raw_start)}"


def clip_id(game_id, episode, player_stem, raw_start):
    return f"{SESSION_HASH}_{game_id}_{episode}_{player_stem}_{int(raw_start):07d}"


def load_raw_key_set(path):
    keys = set()
    rows = 0
    if not Path(path).exists():
        raise FileNotFoundError(path)
    for obj in read_jsonl(path):
        rows += 1
        keys.add(raw_key(obj))
    return keys, rows


def load_exclude_manifest(path):
    """Load one --exclude-manifest: raw_key set + per-track raw_start intervals (all rows,
    including off-grid raw_start) for the partial-overlap audit."""
    if not Path(path).exists():
        raise FileNotFoundError(path)
    keys = set()
    rows = 0
    missing_interval_fields = 0
    track_starts = collections.defaultdict(list)
    for obj in read_jsonl(path):
        rows += 1
        keys.add(raw_key(obj))
        game_id = obj.get("game_id") or obj.get("match_id")
        episode = obj.get("episode")
        player_stem = obj.get("player_stem")
        raw_start = obj.get("raw_start")
        if game_id is None or episode is None or player_stem is None or raw_start is None:
            missing_interval_fields += 1
            continue
        try:
            track_starts[(str(game_id), str(episode), str(player_stem))].append(int(raw_start))
        except (TypeError, ValueError):
            missing_interval_fields += 1
    for arr in track_starts.values():
        arr.sort()
    return {
        "path": str(path),
        "rows": rows,
        "unique_raw_keys": len(keys),
        "rows_missing_interval_fields": missing_interval_fields,
        "keys": keys,
        "track_starts": dict(track_starts),
    }


def partial_overlap_audit(selected, exclude_infos, max_examples=20):
    """Raw-interval intersection between selected supplement windows and every exclude
    manifest's rows, grouped by (game_id, episode, player_stem). Window length is
    FRAME_STRIDE_RAW raw frames: [raw_start, raw_start + 162). Record-only policy."""
    per_manifest = []
    total_pairs = 0
    for info in exclude_infos:
        track_starts = info["track_starts"]
        pairs = 0
        rows_with_overlap = 0
        identical_start_pairs = 0
        examples = []
        for cand in selected:
            tkey = (str(cand.get("game_id")), str(cand.get("episode")), str(cand.get("player_stem")))
            starts = track_starts.get(tkey)
            if not starts:
                continue
            rs = int(cand["raw_start"])
            lo = bisect.bisect_left(starts, rs - FRAME_STRIDE_RAW + 1)
            hi = bisect.bisect_left(starts, rs + FRAME_STRIDE_RAW)
            if lo >= hi:
                continue
            rows_with_overlap += 1
            for other in starts[lo:hi]:
                pairs += 1
                if other == rs:
                    identical_start_pairs += 1
                if len(examples) < max_examples:
                    examples.append({
                        "clip_id": cand.get("clip_id"),
                        "game_id": tkey[0],
                        "episode": tkey[1],
                        "player_stem": tkey[2],
                        "selected_raw_start": rs,
                        "exclude_raw_start": other,
                        "overlap_raw_frames": FRAME_STRIDE_RAW - abs(rs - other),
                    })
        total_pairs += pairs
        per_manifest.append({
            "exclude_manifest": info["path"],
            "exclude_rows_considered": info["rows"],
            "exclude_rows_missing_interval_fields": info["rows_missing_interval_fields"],
            "selected_rows_with_overlap": rows_with_overlap,
            "partial_overlap_pairs": pairs,
            "identical_start_pairs": identical_start_pairs,
            "examples_first_n": examples,
        })
    return {
        "policy": "record_only_do_not_reject (event<->tier ~8.6% partial-overlap precedent)",
        "window_raw_frames": FRAME_STRIDE_RAW,
        "window_definition": "[raw_start, raw_start + 162)",
        "selected_supplement_rows_checked": len(selected),
        "total_partial_overlap_pairs": total_pairs,
        "per_manifest": per_manifest,
    }


def visibility_stats(vis, latent_frames):
    positive = []
    max_pct = 0.0
    entities = set()
    for frame in latent_frames:
        frame_pos = False
        for entity_idx, item in vis.items():
            for r in item.get("ranges", []):
                lo, hi = r.get("range", [None, None])
                if lo is None:
                    continue
                if int(lo) <= frame <= int(hi):
                    frame_pos = True
                    entities.add(entity_idx)
                    try:
                        max_pct = max(max_pct, float(r.get("pixel_percent_max", 0.0)))
                    except Exception:
                        pass
        if frame_pos:
            positive.append(frame)
    return positive, max_pct, len(entities)


def frame_count_from_video_manifest(path):
    try:
        data = json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return None
    for key in ("frame_count", "num_frames", "n_frames", "frames"):
        val = data.get(key) if isinstance(data, dict) else None
        if isinstance(val, int):
            return val
        if isinstance(val, list):
            return len(val)
    if isinstance(data, dict):
        video = data.get("video") or data.get("video_info") or {}
        if isinstance(video, dict):
            for key in ("frame_count", "num_frames", "n_frames"):
                val = video.get(key)
                if isinstance(val, int):
                    return val
    return None


def frame_count_from_action_json(path):
    try:
        data = json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, list):
        max_frame = None
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("frame_count"), int):
                max_frame = item["frame_count"] if max_frame is None else max(max_frame, item["frame_count"])
        return (max_frame + 1) if max_frame is not None else len(data)
    if isinstance(data, dict):
        for key in ("frame_count", "num_frames", "n_frames"):
            if isinstance(data.get(key), int):
                return data[key]
        frames = data.get("frames")
        if isinstance(frames, list):
            return len(frames)
    return None


def build_candidate_from_sidecars(game_id, ep_dir, player_stem, raw_start, vis, map_name):
    latent_frames = [raw_start + LATENT_STEP_RAW * i for i in range(LATENT_COUNT)]
    pos_frames, max_pct, entity_count = visibility_stats(vis, latent_frames)
    if not pos_frames:
        return None
    raw_indices = list(range(raw_start, raw_start + FRAME_STRIDE_RAW + 1, RAW_FRAME_STEP))
    episode = ep_dir.name
    base = ep_dir / player_stem
    dynamic_score = round(len(pos_frames) * 1000.0 + max_pct, 4)
    return {
        "clip_id": clip_id(game_id, episode, player_stem, raw_start),
        "hash": SESSION_HASH,
        "game_id": game_id,
        "episode": episode,
        "player_stem": player_stem,
        "mp4": str(base.with_suffix(".mp4")),
        "action_json": str(base.with_suffix(".json")),
        "episode_info": str(ep_dir / f"{player_stem}_episode_info.json"),
        "video_manifest": str(ep_dir / f"{player_stem}_video_manifest.json"),
        "game_manifest": str(ep_dir / "game_manifest.json"),
        "world_events": str(ep_dir / "world_events.jsonl"),
        "player_visibility": str(ep_dir / f"{player_stem}_player_visibility.json"),
        "map_name": map_name,
        "raw_start": raw_start,
        "raw_indices": raw_indices,
        "frame_count_start": raw_start,
        "frame_count_end": raw_indices[-1],
        "window_stride_raw": FRAME_STRIDE_RAW,
        "dynamic_score": dynamic_score,
        "prompt": f"first-person gameplay video in Counter-Strike, {map_name} map" if map_name else "first-person gameplay video in Counter-Strike",
        "positive_latent_frame_count": len(pos_frames),
        "positive_latent_frames": pos_frames,
        "max_visibility_pixel_percent": round(max_pct, 4),
        "visible_entity_count": entity_count,
        "selection_metric_note": "visibility-positive direct scan; motion metrics intentionally not computed in fast tier manifest",
        "pose_version": "csgoflu_to_opencv_c2w_v1",
        "hfov_source": "fixed_unscoped_hfov_106.26",
        "map_memory_split": match_to_split(game_id),
        "map_memory_split_key": "match",
        "diversity_base_window_stride_raw": FRAME_STRIDE_RAW,
    }


def get_map_name(ep_dir):
    gm = ep_dir / "game_manifest.json"
    try:
        data = json.load(open(gm, "r", encoding="utf-8"))
    except Exception:
        return "de_dust2"
    for key in ("map_name", "map"):
        if isinstance(data, dict) and data.get(key):
            return data[key]
    return "de_dust2"


def scan_candidate_pool(raw_root, exclude_keys, cache_path, tmp_cache_path):
    counts = collections.Counter()
    total_positive = 0
    eligible = 0
    tracks = 0
    matches = set()
    episodes = set()
    log(f"scanning raw_root={raw_root}")
    with open(tmp_cache_path, "w", encoding="utf-8") as out:
        game_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir()])
        for gi, game_dir in enumerate(game_dirs, 1):
            game_id = game_dir.name
            train_dir = game_dir / "train"
            if not train_dir.exists():
                continue
            matches.add(game_id)
            for ep_dir in sorted([p for p in train_dir.iterdir() if p.is_dir()]):
                episodes.add((game_id, ep_dir.name))
                map_name = get_map_name(ep_dir)
                for vis_path in sorted(ep_dir.glob("*_player_visibility.json")):
                    player_stem = vis_path.name[:-len("_player_visibility.json")]
                    base = ep_dir / player_stem
                    mp4 = base.with_suffix(".mp4")
                    action_json = base.with_suffix(".json")
                    video_manifest = ep_dir / f"{player_stem}_video_manifest.json"
                    if not mp4.exists() or not action_json.exists() or not video_manifest.exists():
                        counts["missing_sidecar"] += 1
                        continue
                    try:
                        vis = json.load(open(vis_path, "r", encoding="utf-8"))
                    except Exception:
                        counts["bad_visibility_json"] += 1
                        continue
                    if not any(item.get("ranges") for item in vis.values()):
                        counts["no_visibility_intervals"] += 1
                        continue
                    frame_count = frame_count_from_video_manifest(video_manifest)
                    if not frame_count:
                        frame_count = frame_count_from_action_json(action_json)
                    if not frame_count:
                        counts["missing_frame_count"] += 1
                        continue
                    tracks += 1
                    max_start = frame_count - FRAME_STRIDE_RAW - 1
                    if max_start < RAW_OFFSET:
                        counts["too_short"] += 1
                        continue
                    kmax = (max_start - RAW_OFFSET) // FRAME_STRIDE_RAW
                    for k in range(kmax + 1):
                        rs = RAW_OFFSET + FRAME_STRIDE_RAW * k
                        cand = build_candidate_from_sidecars(game_id, ep_dir, player_stem, rs, vis, map_name)
                        if cand is None:
                            continue
                        total_positive += 1
                        key = raw_key(cand)
                        if key in exclude_keys:
                            counts["excluded_raw_key"] += 1
                            continue
                        cand["source_manifest_index"] = eligible
                        out.write(json.dumps(cand, ensure_ascii=False, sort_keys=False) + "\n")
                        eligible += 1
            if gi % 10 == 0:
                log(f"scan progress games={gi}/{len(game_dirs)} positive={total_positive} eligible={eligible}")
    os.replace(tmp_cache_path, cache_path)
    return {
        "positive_candidates_direct_scan": total_positive,
        "eligible_after_exclusion": eligible,
        "matches": len(matches),
        "episodes": len(episodes),
        "tracks_scanned": tracks,
        "counts": dict(counts),
    }


def round_robin_select(candidates, seed_keys, extra_exclude_keys, need):
    """Round-robin over (match, episode, time-bin) groups, highest dynamic_score first
    within each group. Select-time filter = seed_keys UNION extra_exclude_keys (fix vs v0,
    which only filtered seed_keys); seed skips and extra-exclude skips counted separately."""
    bins = collections.defaultdict(list)
    skipped_seed = 0
    skipped_extra = 0
    for cand in candidates:
        key = raw_key(cand)
        if key in seed_keys:
            skipped_seed += 1
            continue
        if key in extra_exclude_keys:
            skipped_extra += 1
            continue
        bkey = (cand["game_id"], cand["episode"], int(cand["raw_start"]) // FRAME_STRIDE_RAW)
        bins[bkey].append(cand)
    for arr in bins.values():
        arr.sort(key=lambda x: (-float(x.get("dynamic_score", 0.0)), x.get("player_stem", ""), int(x.get("raw_start", 0))))
    heap = []
    for idx, (bkey, arr) in enumerate(sorted(bins.items())):
        if arr:
            heapq.heappush(heap, (0, bkey[0], bkey[1], bkey[2], idx, bkey, 0))
    selected = []
    while heap and len(selected) < need:
        round_idx, _g, _e, _bin, idx, bkey, pos = heapq.heappop(heap)
        arr = bins[bkey]
        cand = dict(arr[pos])
        cand["tier_selection_source"] = "seed_plus_match_episode_timebin_round_robin_v2"
        cand["tier_selection_phase"] = "supplement"
        cand["tier_round_robin_group"] = f"{bkey[0]}|{bkey[1]}|bin{bkey[2]}"
        cand["tier_round_robin_rank_in_group"] = pos
        selected.append(cand)
        nxt = pos + 1
        if nxt < len(arr):
            heapq.heappush(heap, (round_idx + 1, bkey[0], bkey[1], bkey[2], idx, bkey, nxt))
    return selected, {
        "round_robin_groups": len(bins),
        "skipped_seed_candidates": skipped_seed,
        "skipped_extra_exclude_candidates": skipped_extra,
        "available_supplement_candidates": sum(len(v) for v in bins.values()),
    }


def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate a seed-expanded v2 tier source manifest: keep all seed rows "
                    "verbatim (canonical split relabel), then fill up to target_hours from the "
                    "candidate pool cache by match/episode/time-bin round robin, filtering seed "
                    "AND extra exclude manifests at selection time.")
    ap.add_argument("--seed-manifest", required=True, help="input seed manifest jsonl (kept verbatim as prefix)")
    ap.add_argument("--target-manifest", required=True, help="output manifest jsonl path")
    ap.add_argument("--report-path", required=True, help="output report json path")
    ap.add_argument("--out-dir", required=True, help="working dir for tmp files and freshly scanned cache")
    ap.add_argument("--seed-hours", required=True, type=float, help="hours represented by the seed manifest")
    ap.add_argument("--target-hours", required=True, type=float, help="target hours after expansion")
    ap.add_argument("--candidate-cache", default=str(DEFAULT_CACHE),
                    help="candidate pool cache jsonl, opened READ-ONLY when it exists "
                         f"(default: {DEFAULT_CACHE})")
    ap.add_argument("--candidate-cache-sha256", default=None,
                    help="optional expected sha256 (full 64-hex or prefix) of --candidate-cache; "
                         "mismatch or missing cache file -> hard fail")
    ap.add_argument("--exclude-manifest", action="append", default=[],
                    help="extra manifest jsonl whose raw_keys are filtered at SELECTION time "
                         "(repeatable); also drives the partial-overlap audit")
    ap.add_argument("--summary-path", default=str(DEFAULT_SUMMARY),
                    help="read-only summary json providing raw_root and selection_contract "
                         f"(default: {DEFAULT_SUMMARY})")
    return ap.parse_args()


def check_cache_sha(actual, expected):
    exp = expected.strip().lower()
    if not exp:
        raise ValueError("--candidate-cache-sha256 is empty")
    if len(exp) == 64:
        return actual == exp
    return actual.startswith(exp)


def main():
    start = time.time()
    args = parse_args()

    seed_manifest = Path(args.seed_manifest)
    target_manifest = Path(args.target_manifest)
    report_path = Path(args.report_path)
    out_dir = Path(args.out_dir)
    if args.seed_hours <= 0 or args.target_hours <= args.seed_hours:
        raise ValueError(f"need target_hours > seed_hours > 0, got seed={args.seed_hours} target={args.target_hours}")
    out_dir.mkdir(parents=True, exist_ok=True)
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = out_dir / (target_manifest.name + ".tmp")

    summary = json.load(open(args.summary_path, "r", encoding="utf-8"))
    raw_root = Path(summary["raw_root"])
    scan_exclusion_paths = [summary["selection_contract"]["exclude_heldout"]] + list(
        summary["selection_contract"]["exclude_already_rendered_v2_raw_keys"])

    log(f"loading seed manifest {seed_manifest}")
    seed_lines = []
    seed_keys = set()
    seed_clip_ids = []
    for line in open(seed_manifest, "r", encoding="utf-8"):
        if not line.strip():
            continue
        obj = json.loads(line)
        seed_lines.append(line if line.endswith("\n") else line + "\n")
        seed_keys.add(raw_key(obj))
        seed_clip_ids.append(obj.get("clip_id") or raw_key(obj))
    seed_rows = len(seed_lines)
    if seed_rows == 0:
        raise RuntimeError(f"seed manifest is empty: {seed_manifest}")
    target_rows = int(round(seed_rows * args.target_hours / args.seed_hours))
    need = target_rows - seed_rows
    log(f"seed_rows={seed_rows} seed_hours={args.seed_hours} target_hours={args.target_hours} "
        f"target_rows={target_rows} need_supplement={need}")
    if need <= 0:
        raise RuntimeError(f"nothing to supplement: target_rows={target_rows} <= seed_rows={seed_rows}")

    log("loading extra exclude manifests (select-time filter)")
    exclude_infos = []
    extra_exclude_keys = set()
    exclude_manifests_report = []
    for p in args.exclude_manifest:
        info = load_exclude_manifest(p)
        exclude_infos.append(info)
        extra_exclude_keys.update(info["keys"])
        exclude_manifests_report.append({
            "path": info["path"],
            "rows": info["rows"],
            "unique_raw_keys": info["unique_raw_keys"],
            "rows_missing_interval_fields": info["rows_missing_interval_fields"],
        })
        log(f"  exclude {p}: rows={info['rows']} unique_raw_keys={info['unique_raw_keys']}")
    log(f"extra exclude manifests={len(exclude_infos)} union_raw_keys={len(extra_exclude_keys)}")

    scan_exclude_keys = set()
    scan_exclusion_report = {}
    for p in scan_exclusion_paths:
        keys, rows = load_raw_key_set(p)
        scan_exclusion_report[p] = {"rows": rows, "unique_raw_keys": len(keys)}
        scan_exclude_keys.update(keys)
    log(f"loaded contract scan exclusions unique_raw_keys={len(scan_exclude_keys)} paths={len(scan_exclusion_paths)}")

    cache_path = Path(args.candidate_cache)
    if cache_path.exists():
        log(f"reusing candidate cache READ-ONLY {cache_path}")
        cache_sha = sha256_path(cache_path)
        if args.candidate_cache_sha256 is not None:
            if not check_cache_sha(cache_sha, args.candidate_cache_sha256):
                raise RuntimeError(
                    f"candidate cache sha256 mismatch: actual={cache_sha} "
                    f"expected={args.candidate_cache_sha256} path={cache_path}")
            log(f"candidate cache sha256 check OK ({cache_sha})")
        scan_report = {"cache_reused": True, "cache_path": str(cache_path)}
    else:
        if args.candidate_cache_sha256 is not None:
            raise RuntimeError(
                f"--candidate-cache-sha256 given but cache file missing: {cache_path}")
        cache_path = out_dir / "candidate_pool_cache_v2.jsonl"
        tmp_cache = out_dir / "candidate_pool_cache_v2.jsonl.tmp"
        log(f"candidate cache missing; scanning into {cache_path}")
        scan_report = scan_candidate_pool(raw_root, scan_exclude_keys, cache_path, tmp_cache)
        scan_report["cache_reused"] = False
        scan_report["cache_path"] = str(cache_path)
        cache_sha = sha256_path(cache_path)
        log(f"finished scan eligible={scan_report['eligible_after_exclusion']} cache={cache_path}")

    log("loading candidates from cache")
    candidates = list(read_jsonl(cache_path))
    log(f"candidate_cache_rows={len(candidates)}")
    selected, rr_report = round_robin_select(candidates, seed_keys, extra_exclude_keys, need)
    if len(selected) != need:
        raise RuntimeError(f"not enough supplement candidates selected={len(selected)} need={need}")

    log(f"writing target manifest {target_manifest}")
    split_row_counts = {"train": 0, "val": 0, "test": 0}
    relabelled_rows = 0
    with open(tmp_manifest, "w", encoding="utf-8") as out:
        for line in seed_lines:
            row = json.loads(line)
            canonical = match_to_split(str(row.get("game_id") or row.get("match_id")))
            if row.get("map_memory_split") != canonical or row.get("map_memory_split_key") != "match":
                relabelled_rows += 1
            row["map_memory_split"] = canonical
            row["map_memory_split_key"] = "match"
            split_row_counts[canonical] += 1
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
        for i, obj in enumerate(selected, seed_rows):
            obj["source_manifest_index"] = i
            canonical = match_to_split(str(obj.get("game_id") or obj.get("match_id")))
            if obj.get("map_memory_split") != canonical or obj.get("map_memory_split_key") != "match":
                relabelled_rows += 1
            obj["map_memory_split"] = canonical
            obj["map_memory_split_key"] = "match"
            split_row_counts[canonical] += 1
            out.write(json.dumps(obj, ensure_ascii=False, sort_keys=False) + "\n")
    shutil.move(str(tmp_manifest), str(target_manifest))

    log("running partial-overlap audit (record-only)")
    overlap_report = partial_overlap_audit(selected, exclude_infos, max_examples=20)
    log(f"partial_overlap total_pairs={overlap_report['total_partial_overlap_pairs']}")

    target_sha = sha256_path(target_manifest)
    first_n_ids = []
    with open(target_manifest, "r", encoding="utf-8") as f:
        for _, line in zip(range(seed_rows), f):
            first_n_ids.append(json.loads(line).get("clip_id"))
    subset_ok = first_n_ids == seed_clip_ids
    line_count = sum(1 for _ in open(target_manifest, "r", encoding="utf-8"))
    sample_count_estimate = line_count * LATENT_COUNT
    report = {
        "kind": "v2_all188_positive_diversity_tier_seed_expanded_source_manifest_report_v2",
        "created_unix": time.time(),
        "elapsed_sec": round(time.time() - start, 3),
        "rule_one_sentence": "Keep all seed manifest rows verbatim, then fill the remaining rows up to target_hours from the clean v2 visibility-positive candidate pool by round-robin over match/episode/time-bin groups (highest dynamic_score first within each group), filtering seed raw_keys AND all --exclude-manifest raw_keys at selection time.",
        "cli_args": {
            "seed_manifest": str(seed_manifest),
            "target_manifest": str(target_manifest),
            "report_path": str(report_path),
            "out_dir": str(out_dir),
            "seed_hours": args.seed_hours,
            "target_hours": args.target_hours,
            "candidate_cache": args.candidate_cache,
            "candidate_cache_sha256": args.candidate_cache_sha256,
            "exclude_manifest": list(args.exclude_manifest),
            "summary_path": str(args.summary_path),
        },
        "raw_root": str(raw_root),
        "summary_path": str(args.summary_path),
        "seed_manifest": str(seed_manifest),
        "target_manifest": str(target_manifest),
        "candidate_pool_cache": str(cache_path),
        "seed_rows": seed_rows,
        "seed_hours": args.seed_hours,
        "target_hours": args.target_hours,
        "target_rows": target_rows,
        "supplement_rows": len(selected),
        "manifest_rows": line_count,
        "manifest_sha256": target_sha,
        "candidate_pool_cache_rows": len(candidates),
        "candidate_pool_cache_sha256": cache_sha,
        "candidate_cache_sha256_expected": args.candidate_cache_sha256,
        "candidate_cache_sha256_check": ("ok" if args.candidate_cache_sha256 is not None else "not_requested"),
        "seed_subset_prefix_verbatim_clip_id_check": subset_ok,
        "estimated_sample_count_latent_frames": sample_count_estimate,
        "sample_count_estimate_formula": f"manifest_rows * {LATENT_COUNT} positive latent frame slots",
        "split_key": "match",
        "split_row_counts": split_row_counts,
        "split_relabelled_rows": relabelled_rows,
        "split_match_leakage": False,
        "split_note": "canonical match-disjoint split recomputed per row via canonical_match_split_v0 (seed 20260531); heldout rows are labeled val/test and must be excluded by train-split consumers",
        "exclusion_paths": scan_exclusion_paths,
        "exclusion_report": scan_exclusion_report,
        "exclude_manifests": exclude_manifests_report,
        "excluded_at_select": {
            "skipped_seed_candidates": rr_report["skipped_seed_candidates"],
            "skipped_extra_exclude_candidates": rr_report["skipped_extra_exclude_candidates"],
        },
        "partial_overlap_report": overlap_report,
        "scan_report": scan_report,
        "round_robin_report": rr_report,
        "contract": summary.get("selection_contract", {}),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    log(f"DONE manifest_rows={line_count} sha256={target_sha} subset_prefix={subset_ok} "
        f"estimated_sample_count={sample_count_estimate}")
    log(f"report={report_path}")


if __name__ == "__main__":
    main()
