#!/usr/bin/env python3
"""Build pre-full-train data eligibility reports.

This tool joins LingBot source/cache manifests with raw CS:GO metadata and
player visibility.  It does not export dense tensors or start training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


@lru_cache(maxsize=32768)
def cached_json(path_str: str) -> Any:
    return load_json(Path(path_str))


def parse_player_stem(stem: str) -> tuple[int | None, int | None]:
    parts = stem.split("_")
    try:
        team = int(parts[parts.index("team") + 1])
        player = int(parts[parts.index("player") + 1])
        return team, player
    except (ValueError, IndexError):
        return None, None


def hash_dir(root: Path, row: dict[str, Any]) -> Path:
    hash_id = str(row["hash"])
    game_id = str(row["game_id"])
    if root.name == hash_id or (root / game_id).exists():
        return root
    return root / hash_id


def episode_dir(root: Path, row: dict[str, Any]) -> Path:
    return hash_dir(root, row) / str(row["game_id"]) / "train" / str(row["episode"])


def match_dir(root: Path, row: dict[str, Any]) -> Path:
    return hash_dir(root, row) / str(row["game_id"])


@lru_cache(maxsize=8192)
def player_meta_for_episode(episode_dir_str: str) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    ep = Path(episode_dir_str)
    try:
        paths = list(ep.glob("*.json"))
    except OSError:
        return out
    for p in paths:
        stem = p.stem
        if "_team_" not in stem or "_player_" not in stem or "_inst_" not in stem:
            continue
        team, player = parse_player_stem(stem)
        if team is None or player is None:
            continue
        out[player] = {"stem": stem, "team": team, "player": player}
    return out


@lru_cache(maxsize=262144)
def file_exists(path_str: str) -> bool:
    return Path(path_str).exists()


@lru_cache(maxsize=262144)
def dir_has_child(path_str: str) -> bool:
    path = Path(path_str)
    if not path.is_dir():
        return False
    try:
        next(path.iterdir())
        return True
    except (StopIteration, OSError):
        return False


@lru_cache(maxsize=262144)
def core_files(ep: Path, stem: str) -> dict[str, bool]:
    return {
        "player_json": file_exists(str(ep / f"{stem}.json")),
        "rgb_mp4": file_exists(str(ep / f"{stem}.mp4")),
        "episode_info": file_exists(str(ep / f"{stem}_episode_info.json")),
        "video_manifest": file_exists(str(ep / f"{stem}_video_manifest.json")),
        "player_visibility": file_exists(str(ep / f"{stem}_player_visibility.json")),
        "game_manifest": file_exists(str(ep / "game_manifest.json")),
        "world_events": file_exists(str(ep / "world_events.jsonl")),
    }


@lru_cache(maxsize=262144)
def teacher_files(ep: Path, stem: str) -> dict[str, bool]:
    return {
        "depth_mkv": file_exists(str(ep / f"{stem}_depth.mkv")),
        "seg_mkv": file_exists(str(ep / f"{stem}_seg.mkv")),
        "seg_colormap_json": file_exists(str(ep / f"{stem}_seg_colormap.json")),
        "hud_mkv": file_exists(str(ep / f"{stem}_hud.mkv")),
        "motion_bin": file_exists(str(ep / f"{stem}_motion.bin")),
    }


@lru_cache(maxsize=65536)
def geometry_files(match: Path) -> dict[str, bool]:
    meshes_dir = match / "meshes"
    return {
        "navmesh": file_exists(str(match / "navmesh.json")),
        "mesh_manifest": file_exists(str(match / "mesh_manifest.json")),
        "static_props": file_exists(str(match / "static_props.json")),
        "meshes_dir": meshes_dir.is_dir(),
        "meshes_obj_any": dir_has_child(str(meshes_dir)),
    }


def all_true(values: dict[str, bool]) -> bool:
    return all(values.values())


def visibility_counts(
    visibility_path: Path,
    frames: list[int],
    player_meta: dict[int, dict[str, Any]],
    ego_team: int | None,
    min_visible_players: int,
) -> tuple[list[int], list[int], list[int], float, int]:
    if not file_exists(str(visibility_path)):
        return [], [], [], 0.0, 0
    try:
        visibility = cached_json(str(visibility_path))
    except (OSError, json.JSONDecodeError):
        return [], [], [], 0.0, 0
    rows_by_target: list[tuple[int, int | None, list[dict[str, Any]]]] = []
    if not isinstance(visibility, dict):
        return [], [], [], 0.0, 0
    for target_key, entry in visibility.items():
        try:
            target_idx = int(target_key)
        except (TypeError, ValueError):
            continue
        if not isinstance(entry, dict):
            continue
        target_team = player_meta.get(target_idx, {}).get("team")
        ranges = entry.get("ranges") or []
        rows_by_target.append((target_idx, target_team, ranges))

    visible_per_frame: list[int] = []
    cross_team_per_frame: list[int] = []
    pair_edges_per_frame: list[int] = []
    max_pixel = 0.0
    max_visible = 0
    for frame in frames:
        visible = 0
        cross = 0
        for _target_idx, target_team, ranges in rows_by_target:
            best = 0.0
            for row in ranges:
                try:
                    lo, hi = row["range"]
                    score = float(row.get("pixel_percent_max", 0.0))
                except (KeyError, TypeError, ValueError):
                    continue
                if int(lo) <= frame <= int(hi):
                    best = max(best, score)
            if best > 0:
                visible += 1
                max_pixel = max(max_pixel, best)
                if ego_team is not None and target_team is not None and target_team != ego_team:
                    cross += 1
        visible_per_frame.append(visible)
        cross_team_per_frame.append(cross)
        pair_edges_per_frame.append(visible)
        max_visible = max(max_visible, visible)
    positive_flags = [1 if n >= min_visible_players else 0 for n in visible_per_frame]
    return visible_per_frame, cross_team_per_frame, positive_flags, max_pixel, max_visible


def expected_map_memory_ids(row: dict[str, Any], frames: list[int]) -> list[str]:
    return [
        f"{row['game_id']}__{row['episode']}_{row['player_stem']}_f{int(frame):06d}"
        for frame in frames
    ]


def load_cache_index(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        clip_id = row.get("clip_id")
        if clip_id:
            out[str(clip_id)] = {
                "latent_cache": row.get("latent_cache"),
                "shape": row.get("shape"),
                "condition_shape": row.get("condition_shape"),
                "latent_frames_expected": row.get("latent_frames_expected"),
            }
    return out


def load_map_memory_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    data = load_json(path)
    rows: list[dict[str, Any]]
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("samples") or data.get("records") or data.get("items") or []
    else:
        rows = []
    ids = set()
    for row in rows:
        sample_id = row.get("sample_id") or row.get("id")
        if sample_id:
            ids.add(str(sample_id))
    return ids


def load_visible_keys(paths: list[Path], limit: int) -> tuple[set[tuple[str, str, str, str, int]], set[str], int]:
    keys: set[tuple[str, str, str, str, int]] = set()
    matches: set[str] = set()
    n = 0
    for path in paths:
        for row in iter_jsonl(path):
            n += 1
            matches.add(str(row.get("match_id") or row.get("match")))
            if len(keys) < limit or limit <= 0:
                try:
                    keys.add((
                        str(row["hash"]),
                        str(row.get("match_id") or row.get("match")),
                        str(row["episode"]),
                        str(row.get("ego_stem") or row.get("ego", {}).get("stem")),
                        int(row["raw_start"]),
                    ))
                except (KeyError, TypeError, ValueError):
                    pass
    return keys, matches, n


def split_for_match(match_id: str, seed: str, val_frac: float, test_frac: float) -> str:
    digest = hashlib.sha1(f"{seed}:{match_id}".encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16 ** 12)
    if value < test_frac:
        return "test"
    if value < test_frac + val_frac:
        return "val"
    return "train"


def counter_dict(counter: Counter) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda kv: str(kv[0]))}


def make_split_distribution_entry() -> dict[str, Any]:
    return {
        "rows": 0,
        "positive_clip_count": 0,
        "teacher_qa_ready": 0,
        "tier_counts": Counter(),
        "max_visible_players_hist": Counter(),
        "max_cross_team_visible_latent_hist": Counter(),
        "positive_latent_frames_hist": Counter(),
    }


def split_distribution_json(split_distribution: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split, stats in sorted(split_distribution.items()):
        out[split] = {
            "rows": int(stats["rows"]),
            "positive_clip_count": int(stats["positive_clip_count"]),
            "teacher_qa_ready": int(stats["teacher_qa_ready"]),
            "tier_counts": counter_dict(stats["tier_counts"]),
            "max_visible_players_hist": counter_dict(stats["max_visible_players_hist"]),
            "max_cross_team_visible_latent_hist": counter_dict(stats["max_cross_team_visible_latent_hist"]),
            "positive_latent_frames_hist": counter_dict(stats["positive_latent_frames_hist"]),
        }
    return out


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown_report(path: Path, summary: dict[str, Any], company_questions: list[str], zhengqi_questions: list[str]) -> None:
    lines = [
        "# Full Train Data Gate Report",
        "",
        "## Summary",
        "",
        f"- Source rows scanned: {summary['source_rows_scanned']:,}",
        f"- Rows with latent cache: {summary['rows_with_cache']:,}",
        f"- Trainable core candidates: {summary['tier_counts'].get('trainable_no_teacher', 0):,}",
        f"- Teacher-QA ready candidates: {summary['tier_counts'].get('teacher_qa_ready', 0):,}",
        f"- Metadata-only / blocked rows: {summary['tier_counts'].get('metadata_only_or_blocked', 0):,}",
        f"- Positive latent-frame clips: {summary['positive_clip_count']:,}",
        "",
        "## Split Stats",
        "",
        "| Split | Matches | Rows | Positive clips | Teacher-QA ready |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in ["train", "val", "test"]:
        s = summary["splits"].get(split, {})
        lines.append(
            f"| {split} | {len(s.get('matches', []))} | {s.get('rows', 0):,} | "
            f"{s.get('positive_clip_count', 0):,} | {s.get('teacher_qa_ready', 0):,} |"
        )
    lines += [
        "",
        "## Field Missing Rates",
        "",
        "| Field group | Missing / total |",
        "| --- | ---: |",
    ]
    for key, value in summary["missing_rates"].items():
        lines.append(f"| `{key}` | {value['missing']:,} / {value['total']:,} ({value['missing_rate']:.1%}) |")
    lines += [
        "",
        "## Company Questions",
        "",
    ]
    lines += [f"{i + 1}. {q}" for i, q in enumerate(company_questions)]
    lines += [
        "",
        "## Zhengqi Questions",
        "",
    ]
    lines += [f"{i + 1}. {q}" for i, q in enumerate(zhengqi_questions)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-manifest", type=Path, required=True)
    ap.add_argument("--cache-manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--raw-root-primary", type=Path, required=True)
    ap.add_argument("--raw-root-secondary", type=Path, default=None)
    ap.add_argument("--visible-jsonl", type=Path, action="append", default=[])
    ap.add_argument("--filter-visible-matches", action="store_true")
    ap.add_argument("--visible-key-limit", type=int, default=0)
    ap.add_argument("--map-memory-manifest", type=Path, default=None)
    ap.add_argument("--limit-source-rows", type=int, default=0)
    ap.add_argument("--max-output-rows", type=int, default=50000)
    ap.add_argument("--split-seed", default="map-memory-fulltrain-v0")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--min-visible-players", type=int, default=2)
    ap.add_argument("--latent-source-step", type=int, default=4)
    ap.add_argument("--dataset-label", default="fulltrain_data_gate_v0")
    ap.add_argument("--progress-every", type=int, default=25000)
    args = ap.parse_args()

    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_index = load_cache_index(args.cache_manifest)
    map_memory_ids = load_map_memory_ids(args.map_memory_manifest)
    visible_keys, visible_matches, visible_rows = load_visible_keys(args.visible_jsonl, args.visible_key_limit)

    contract = {
        "kind": "fulltrain_data_contract_v0",
        "required_for_trainable": [
            "source_manifest row",
            "cache_manifest clip_id with latent_cache",
            "raw RGB mp4",
            "player JSON",
            "episode_info",
            "video_manifest",
            "player_visibility",
            "game_manifest",
            "world_events",
            "constructible Map Memory dense sample ids for latent frames",
        ],
        "required_for_teacher_qa": [
            "trainable row",
            "depth_mkv",
            "seg_mkv",
            "seg_colormap_json",
            "hud_mkv",
            "motion_bin",
            "geometry strategy with mesh/static props or documented BSP replacement",
        ],
        "tiers": {
            "teacher_qa_ready": "trainable plus actual teacher media and primary geometry files",
            "trainable_no_teacher": "cache/core/visibility present and positive latent frames, but full teacher QA files absent",
            "metadata_only_or_blocked": "missing cache/core/visibility or no positive latent frames",
        },
    }
    write_json(args.out_dir / "data_contract_v0.json", contract)

    company_questions = [
        "full CPFS 是否本来就应该包含 mesh_manifest.json、static_props.json、meshes/？如果不是，geometry 是否统一从 OSS、BSP 或其他路径读取？",
        "OSS 有 mesh_manifest.json/static_props.json 但没有 meshes/ OBJ directory；OBJ mesh 是否被省略、另存，还是由 BSP/static geometry 替代？",
        "full32f 是否能补 actual teacher media：_depth.mkv、_seg.mkv、_seg_colormap.json、_hud.mkv、_motion.bin？",
        "episode_info 里 depth/hud/motion/mesh/static_props available=True 但实际文件不存在；这些 flag 是否过期或代表别的存储位置？",
        "是否有 canonical cross-mount manifest，标明 CPFS/OSS/其他路径分别负责哪些 asset，并带文件数/checksum？",
        "如果 teacher media 不再提供，是否接受 full train 用 RGB + player state + visibility + Map Memory self-check，teacher QA 降级到 fullsubset/小子集？",
    ]
    zhengqi_questions = [
        "1200 light pilot 的 dirty/untracked 脚本是否为最新 canonical，尤其是 export/combine/map_memory_training_data/teacher_player_latent_mask/train 脚本？",
        "region_mask_kind=memory_dense_channel_3_surrogate_v0 是否作为正式 region mask；若不是，full32f 是否必须等待 teacher seg？",
        "full train 前主 gate 是否仍是 small overfit true dense 赢 shuffled/player_shuffle，以及 holdout true dense 赢 blank/shuffled/player_shuffle？",
        "full32f 选样沿用 light pilot selection，还是接入 qxq visible-multiplayer/data-gate manifest？",
        "match-disjoint val/test match 列表由谁固定，是否已有项目级 split 文件？",
        "step250/500/750/1000 ablation 是否会产出 paired significance JSON；若 true-vs-shuffled fail，是否暂停扩大训练？",
    ]
    write_json(args.out_dir / "company_questions_v0.json", company_questions)
    write_json(args.out_dir / "zhengqi_questions_v0.json", zhengqi_questions)

    counters: dict[str, Counter] = defaultdict(Counter)
    split_stats: dict[str, dict[str, Any]] = {
        "train": {"matches": set(), "rows": 0, "positive_clip_count": 0, "teacher_qa_ready": 0},
        "val": {"matches": set(), "rows": 0, "positive_clip_count": 0, "teacher_qa_ready": 0},
        "test": {"matches": set(), "rows": 0, "positive_clip_count": 0, "teacher_qa_ready": 0},
    }
    split_distribution: dict[str, dict[str, Any]] = {
        "train": make_split_distribution_entry(),
        "val": make_split_distribution_entry(),
        "test": make_split_distribution_entry(),
    }
    match_stats: dict[str, Counter] = defaultdict(Counter)
    aligned_rows_written = 0
    source_rows_scanned = 0
    rows_with_cache = 0
    positive_clip_count = 0
    missing_counts: dict[str, Counter] = defaultdict(Counter)

    out_jsonl = args.out_dir / "aligned_eligibility_manifest.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as out_f:
        for row in iter_jsonl(args.source_manifest):
            if args.limit_source_rows and source_rows_scanned >= args.limit_source_rows:
                break
            source_rows_scanned += 1
            match_id = str(row.get("game_id"))
            if args.filter_visible_matches and visible_matches and match_id not in visible_matches:
                continue
            clip_id = str(row.get("clip_id"))
            stem = str(row.get("player_stem"))
            primary_ep = episode_dir(args.raw_root_primary, row)
            primary_match = match_dir(args.raw_root_primary, row)
            secondary_ep = episode_dir(args.raw_root_secondary, row) if args.raw_root_secondary else None
            secondary_match = match_dir(args.raw_root_secondary, row) if args.raw_root_secondary else None
            core_primary = core_files(primary_ep, stem)
            teacher_primary = teacher_files(primary_ep, stem)
            geom_primary = geometry_files(primary_match)
            teacher_secondary = teacher_files(secondary_ep, stem) if secondary_ep else {}
            geom_secondary = geometry_files(secondary_match) if secondary_match else {}

            cache = cache_index.get(clip_id)
            cache_ok = cache is not None and bool(cache.get("latent_cache"))
            rows_with_cache += int(cache_ok)
            raw_indices = [int(x) for x in row.get("raw_indices", [])]
            latent_positions = list(range(0, len(raw_indices), max(1, args.latent_source_step)))
            latent_frames = [raw_indices[i] for i in latent_positions if i < len(raw_indices)]
            ego_team, _ego_player = parse_player_stem(stem)
            player_meta = player_meta_for_episode(str(primary_ep))
            visibility_path = primary_ep / f"{stem}_player_visibility.json"
            visible_81, cross_81, positive_81_flags, max_pixel, max_visible = visibility_counts(
                visibility_path,
                raw_indices,
                player_meta,
                ego_team,
                args.min_visible_players,
            )
            latent_visible, latent_cross, latent_positive_flags, _latent_max_pixel, latent_max_visible = visibility_counts(
                visibility_path,
                latent_frames,
                player_meta,
                ego_team,
                args.min_visible_players,
            )
            positive_frames_81 = int(sum(positive_81_flags))
            positive_latent_frames = int(sum(latent_positive_flags))
            positive_clip = positive_latent_frames > 0
            positive_clip_count += int(positive_clip)
            counters["positive_latent_frame_count_hist"][positive_latent_frames] += 1
            counters["max_visible_players_hist"][max(max_visible, latent_max_visible)] += 1
            counters["max_cross_team_visible_latent_hist"][max(latent_cross) if latent_cross else 0] += 1

            expected_ids = expected_map_memory_ids(row, latent_frames)
            if map_memory_ids is None:
                map_memory_status = "not_checked_constructed_ids"
                missing_map_memory_ids = 0
            else:
                missing_map_memory_ids = sum(1 for sample_id in expected_ids if sample_id not in map_memory_ids)
                map_memory_status = "all_found" if missing_map_memory_ids == 0 else "missing_ids"

            exact_visible_key = (
                str(row.get("hash")),
                match_id,
                str(row.get("episode")),
                stem,
                int(row.get("raw_start", -1)),
            )
            visible_manifest_exact_raw_start = exact_visible_key in visible_keys if visible_keys else None

            core_ok = all_true(core_primary)
            trainable_no_teacher = bool(cache_ok and core_ok and positive_clip and map_memory_status != "missing_ids")
            teacher_ready_primary = bool(trainable_no_teacher and all_true(teacher_primary) and all_true(geom_primary))
            if teacher_ready_primary:
                tier = "teacher_qa_ready"
            elif trainable_no_teacher:
                tier = "trainable_no_teacher"
            else:
                tier = "metadata_only_or_blocked"
            counters["tier_counts"][tier] += 1

            for prefix, values in [
                ("core_primary", core_primary),
                ("teacher_primary", teacher_primary),
                ("geometry_primary", geom_primary),
                ("teacher_secondary", teacher_secondary),
                ("geometry_secondary", geom_secondary),
            ]:
                for key, value in values.items():
                    missing_counts[f"{prefix}.{key}"]["total"] += 1
                    missing_counts[f"{prefix}.{key}"]["missing"] += int(not value)

            split = split_for_match(match_id, args.split_seed, args.val_frac, args.test_frac)
            split_stats[split]["matches"].add(match_id)
            split_stats[split]["rows"] += 1
            split_stats[split]["positive_clip_count"] += int(positive_clip)
            split_stats[split]["teacher_qa_ready"] += int(teacher_ready_primary)
            split_distribution[split]["rows"] += 1
            split_distribution[split]["positive_clip_count"] += int(positive_clip)
            split_distribution[split]["teacher_qa_ready"] += int(teacher_ready_primary)
            split_distribution[split]["tier_counts"][tier] += 1
            split_distribution[split]["max_visible_players_hist"][max(max_visible, latent_max_visible)] += 1
            split_distribution[split]["max_cross_team_visible_latent_hist"][max(latent_cross) if latent_cross else 0] += 1
            split_distribution[split]["positive_latent_frames_hist"][positive_latent_frames] += 1
            match_stats[match_id]["rows"] += 1
            match_stats[match_id]["positive_clip_count"] += int(positive_clip)
            match_stats[match_id]["teacher_qa_ready"] += int(teacher_ready_primary)
            match_stats[match_id][f"split_{split}"] += 1

            out_row = {
                "clip_id": clip_id,
                "hash": row.get("hash"),
                "match_id": match_id,
                "episode": row.get("episode"),
                "player_stem": stem,
                "raw_start": row.get("raw_start"),
                "raw_indices_first_last": [raw_indices[0], raw_indices[-1]] if raw_indices else None,
                "latent_frames": latent_frames,
                "cache_ok": cache_ok,
                "core_primary_ok": core_ok,
                "teacher_primary_ok": all_true(teacher_primary),
                "geometry_primary_ok": all_true(geom_primary),
                "teacher_secondary_ok": all_true(teacher_secondary) if teacher_secondary else None,
                "geometry_secondary_ok": all_true(geom_secondary) if geom_secondary else None,
                "positive_frames_81": positive_frames_81,
                "positive_latent_frames": positive_latent_frames,
                "max_visible_players": max(max_visible, latent_max_visible),
                "max_visibility_pixel_percent": max_pixel,
                "max_cross_team_visible_latent": max(latent_cross) if latent_cross else 0,
                "map_memory_sample_ids": expected_ids,
                "map_memory_status": map_memory_status,
                "missing_map_memory_ids": missing_map_memory_ids,
                "visible_manifest_exact_raw_start": visible_manifest_exact_raw_start,
                "split": split,
                "eligibility_tier": tier,
            }
            if args.max_output_rows <= 0 or aligned_rows_written < args.max_output_rows:
                out_f.write(json.dumps(out_row, ensure_ascii=False, separators=(",", ":")) + "\n")
                aligned_rows_written += 1

            if args.progress_every and source_rows_scanned % args.progress_every == 0:
                print(json.dumps({
                    "rows": source_rows_scanned,
                    "written": aligned_rows_written,
                    "positive": positive_clip_count,
                    "elapsed_sec": round(time.time() - started, 1),
                }), flush=True)

    split_json = {
        split: {
            **{k: v for k, v in stats.items() if k != "matches"},
            "matches": sorted(stats["matches"]),
        }
        for split, stats in split_stats.items()
    }
    missing_rates = {
        key: {
            "missing": int(counts["missing"]),
            "total": int(counts["total"]),
            "missing_rate": float(counts["missing"] / counts["total"]) if counts["total"] else 0.0,
        }
        for key, counts in sorted(missing_counts.items())
    }
    match_json = {
        match_id: {str(k): int(v) for k, v in counts.items()}
        for match_id, counts in sorted(match_stats.items())
    }
    split_distribution_out = {
        "kind": "split_distribution_v0",
        "source": str(out_jsonl),
        "splits": split_distribution_json(split_distribution),
    }
    summary = {
        "kind": "fulltrain_data_gate_summary_v0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_label": args.dataset_label,
        "source_manifest": str(args.source_manifest),
        "cache_manifest": str(args.cache_manifest),
        "raw_root_primary": str(args.raw_root_primary),
        "raw_root_secondary": str(args.raw_root_secondary) if args.raw_root_secondary else None,
        "visible_jsonl": [str(p) for p in args.visible_jsonl],
        "visible_rows_loaded": visible_rows,
        "visible_keys_loaded": len(visible_keys),
        "source_rows_scanned": source_rows_scanned,
        "aligned_rows_written": aligned_rows_written,
        "rows_with_cache": rows_with_cache,
        "positive_clip_count": positive_clip_count,
        "tier_counts": counter_dict(counters["tier_counts"]),
        "positive_latent_frame_count_hist": counter_dict(counters["positive_latent_frame_count_hist"]),
        "max_visible_players_hist": counter_dict(counters["max_visible_players_hist"]),
        "max_cross_team_visible_latent_hist": counter_dict(counters["max_cross_team_visible_latent_hist"]),
        "missing_rates": missing_rates,
        "splits": split_json,
        "elapsed_sec": round(time.time() - started, 3),
        "outputs": {
            "aligned_eligibility_manifest": str(out_jsonl),
            "data_contract": str(args.out_dir / "data_contract_v0.json"),
            "split_matches": str(args.out_dir / "split_matches_v0.json"),
            "match_stats": str(args.out_dir / "match_stats_v0.json"),
            "split_distribution": str(args.out_dir / "split_distribution_v0.json"),
        },
    }
    write_json(args.out_dir / "summary_v0.json", summary)
    write_json(args.out_dir / "split_matches_v0.json", split_json)
    write_json(args.out_dir / "match_stats_v0.json", match_json)
    write_json(args.out_dir / "split_distribution_v0.json", split_distribution_out)
    write_markdown_report(args.out_dir / "FULLTRAIN_DATA_GATE_REPORT_v0.md", summary, company_questions, zhengqi_questions)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
