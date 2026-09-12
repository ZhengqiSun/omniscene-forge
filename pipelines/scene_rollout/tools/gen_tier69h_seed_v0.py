#!/usr/bin/env python3
"""Generate tier69h seed-expanded manifest."""
from runtime_paths import source_path
import collections
import hashlib
import heapq
import json
import os
import time
from pathlib import Path

ROOT = Path(str(source_path('scene', '')))
OUT_DIR = ROOT / "output/memory_v2_all188_diversity_stride_manifest_20260629_v0"
SUMMARY_PATH = OUT_DIR / "v2_all188_diversity_stride_manifest_summary_v0.json"
SEED_MANIFEST = OUT_DIR / "v2_all188_positive_diversity_tier20h_source_manifest_v0.jsonl"
TARGET_MANIFEST = OUT_DIR / "v2_all188_positive_diversity_tier69h_source_manifest_v0.jsonl"
REPORT_PATH = OUT_DIR / "v2_all188_positive_diversity_tier69h_source_manifest_report_v0.json"
CACHE_PATH = OUT_DIR / "candidate_pool_cache_v0.jsonl"
TMP_CACHE = OUT_DIR / "candidate_pool_cache_v0.jsonl.tmp"
TMP_MANIFEST = OUT_DIR / "v2_all188_positive_diversity_tier69h_source_manifest_v0.jsonl.tmp"

SESSION_HASH = "32f1644d4f42c29d"
FRAME_STRIDE_RAW = 162
RAW_OFFSET = 16
LATENT_STEP_RAW = 8
LATENT_COUNT = 21
RAW_FRAME_STEP = 2
TIER20_HOURS = 20.0
TARGET_HOURS = 69.0


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
        "map_memory_split": "train",
        "map_memory_split_key": "v2_all188_diversity_stride",
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


def scan_candidate_pool(raw_root, exclude_keys):
    counts = collections.Counter()
    total_positive = 0
    eligible = 0
    tracks = 0
    matches = set()
    episodes = set()
    log(f"scanning raw_root={raw_root}")
    with open(TMP_CACHE, "w", encoding="utf-8") as out:
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
    os.replace(TMP_CACHE, CACHE_PATH)
    return {
        "positive_candidates_direct_scan": total_positive,
        "eligible_after_exclusion": eligible,
        "matches": len(matches),
        "episodes": len(episodes),
        "tracks_scanned": tracks,
        "counts": dict(counts),
    }


def load_candidates_from_cache():
    return list(read_jsonl(CACHE_PATH))


def round_robin_select(candidates, seed_keys, need):
    bins = collections.defaultdict(list)
    skipped_seed = 0
    for cand in candidates:
        key = raw_key(cand)
        if key in seed_keys:
            skipped_seed += 1
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
        cand["tier69_selection_source"] = "seed20h_plus_match_episode_timebin_round_robin_v0"
        cand["tier69_selection_phase"] = "supplement"
        cand["tier69_round_robin_group"] = f"{bkey[0]}|{bkey[1]}|bin{bkey[2]}"
        cand["tier69_round_robin_rank_in_group"] = pos
        selected.append(cand)
        nxt = pos + 1
        if nxt < len(arr):
            heapq.heappush(heap, (round_idx + 1, bkey[0], bkey[1], bkey[2], idx, bkey, nxt))
    return selected, {
        "round_robin_groups": len(bins),
        "skipped_seed_candidates": skipped_seed,
        "available_supplement_candidates": sum(len(v) for v in bins.values()),
    }


def main():
    start = time.time()
    summary = json.load(open(SUMMARY_PATH, "r", encoding="utf-8"))
    raw_root = Path(summary["raw_root"])
    exclusion_paths = [summary["selection_contract"]["exclude_heldout"]] + list(summary["selection_contract"]["exclude_already_rendered_v2_raw_keys"])

    log(f"loading seed manifest {SEED_MANIFEST}")
    seed_lines = []
    seed_keys = set()
    seed_clip_ids = []
    for line in open(SEED_MANIFEST, "r", encoding="utf-8"):
        if not line.strip():
            continue
        obj = json.loads(line)
        seed_lines.append(line if line.endswith("\n") else line + "\n")
        seed_keys.add(raw_key(obj))
        seed_clip_ids.append(obj.get("clip_id") or raw_key(obj))
    seed_rows = len(seed_lines)
    target_rows = int(round(seed_rows * TARGET_HOURS / TIER20_HOURS))
    need = target_rows - seed_rows
    log(f"seed_rows={seed_rows} target_rows={target_rows} need_supplement={need}")

    exclude_keys = set()
    exclusion_report = {}
    for p in exclusion_paths:
        keys, rows = load_raw_key_set(p)
        exclusion_report[p] = {"rows": rows, "unique_raw_keys": len(keys)}
        exclude_keys.update(keys)
    log(f"loaded exclusions unique_raw_keys={len(exclude_keys)} from paths={len(exclusion_paths)}")

    if CACHE_PATH.exists():
        scan_report = {"cache_reused": True, "cache_path": str(CACHE_PATH)}
        log(f"reusing candidate cache {CACHE_PATH}")
    else:
        scan_report = scan_candidate_pool(raw_root, exclude_keys)
        scan_report["cache_reused"] = False
        scan_report["cache_path"] = str(CACHE_PATH)
        log(f"finished scan eligible={scan_report['eligible_after_exclusion']} cache={CACHE_PATH}")

    log("loading candidates from cache")
    candidates = load_candidates_from_cache()
    log(f"candidate_cache_rows={len(candidates)}")
    selected, rr_report = round_robin_select(candidates, seed_keys, need)
    if len(selected) != need:
        raise RuntimeError(f"not enough supplement candidates selected={len(selected)} need={need}")

    log(f"writing target manifest {TARGET_MANIFEST}")
    with open(TMP_MANIFEST, "w", encoding="utf-8") as out:
        for line in seed_lines:
            out.write(line)
        for i, obj in enumerate(selected, seed_rows):
            obj["source_manifest_index"] = i
            out.write(json.dumps(obj, ensure_ascii=False, sort_keys=False) + "\n")
    os.replace(TMP_MANIFEST, TARGET_MANIFEST)

    target_sha = sha256_path(TARGET_MANIFEST)
    cache_sha = sha256_path(CACHE_PATH)
    first_n_ids = []
    with open(TARGET_MANIFEST, "r", encoding="utf-8") as f:
        for _, line in zip(range(seed_rows), f):
            first_n_ids.append(json.loads(line).get("clip_id"))
    subset_ok = first_n_ids == seed_clip_ids
    line_count = sum(1 for _ in open(TARGET_MANIFEST, "r", encoding="utf-8"))
    sample_count_estimate = line_count * LATENT_COUNT
    report = {
        "kind": "v2_all188_positive_diversity_tier69h_source_manifest_report_v0",
        "created_unix": time.time(),
        "elapsed_sec": round(time.time() - start, 3),
        "rule_one_sentence": "Tier69h keeps all tier20h rows verbatim, then fills the remaining rows from the same clean v2 visibility-positive candidate pool after exclusions by round-robin over match/episode/time-bin groups, taking highest dynamic_score candidates within each group.",
        "raw_root": str(raw_root),
        "seed_manifest": str(SEED_MANIFEST),
        "target_manifest": str(TARGET_MANIFEST),
        "candidate_pool_cache": str(CACHE_PATH),
        "seed_rows": seed_rows,
        "target_hours": TARGET_HOURS,
        "target_rows": target_rows,
        "supplement_rows": len(selected),
        "manifest_rows": line_count,
        "manifest_sha256": target_sha,
        "candidate_pool_cache_rows": len(candidates),
        "candidate_pool_cache_sha256": cache_sha,
        "tier20_subset_prefix_verbatim_clip_id_check": subset_ok,
        "estimated_sample_count_latent_frames": sample_count_estimate,
        "sample_count_estimate_formula": f"manifest_rows * {LATENT_COUNT} positive latent frame slots",
        "exclusion_paths": exclusion_paths,
        "exclusion_report": exclusion_report,
        "scan_report": scan_report,
        "round_robin_report": rr_report,
        "contract": summary.get("selection_contract", {}),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    log(f"DONE manifest_rows={line_count} sha256={target_sha} subset_prefix={subset_ok} estimated_sample_count={sample_count_estimate}")
    log(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
