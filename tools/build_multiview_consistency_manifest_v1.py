#!/usr/bin/env python3
"""Build deterministic two-view consistency pair manifests from v2 aligned caches.

The builder writes indexes and reports only. It reads raw player JSON to enforce
pairwise 21-frame tick/frame-count equality for positives; it never decodes video,
shifts timestamps, interpolates frames, or creates model/data caches.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator


PROTOCOL_VERSION = "multiview_consistency_mvp_v1_exacttick_matchsplit"
SPLIT_SALT = "multiview_consistency_mvp_v1"
LABEL_DIFFICULTY = {
    "positive": None,
    "time_shift": "hard",
    "wrong_window": "hard",
    "cross_episode": "medium",
    "cross_match": "easy",
}
REFERENCE_FIELDS = (
    "clip_id", "hash", "game_id", "episode", "raw_start", "player_stem",
    "mp4", "video", "action_json", "poses", "intrinsics", "latent_cache",
    "map_memory_raw_frame_indices", "alignment_latent_source_positions",
    "alignment_video_frames", "alignment_exact_frame_match", "shape",
    "condition_shape", "player_visibility", "game_manifest", "episode_info",
    "video_manifest", "world_events", "map_name", "pose_version", "hfov_source",
)


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(*parts: Any) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def row_key(row: dict[str, Any]) -> tuple[str, str, str, int, str]:
    return (
        str(row["hash"]), str(row["game_id"]), str(row["episode"]),
        int(row["raw_start"]), str(row["player_stem"]),
    )


def group_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return row_key(row)[:4]


def episode_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return row_key(row)[:3]


def match_key(row: dict[str, Any]) -> tuple[str, str]:
    return row_key(row)[:2]


def endpoint_id(row: dict[str, Any]) -> str:
    root_hash, game_id, episode, raw_start, player_stem = row_key(row)
    return f"{root_hash}|{game_id}|{episode}|{raw_start:07d}|{player_stem}"


def split_for_match(key: tuple[str, str]) -> str:
    digest = hashlib.sha256(f"{SPLIT_SALT}|{key[0]}|{key[1]}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if value < 0.80:
        return "train"
    if value < 0.90:
        return "val"
    return "test"


def pair_ids(left: dict[str, Any], right: dict[str, Any]) -> tuple[str, str]:
    return tuple(sorted((endpoint_id(left), endpoint_id(right))))  # type: ignore[return-value]


def sample_id(label: str, left: dict[str, Any], right: dict[str, Any]) -> str:
    left_id, right_id = pair_ids(left, right)
    return stable_hash(PROTOCOL_VERSION, label, left_id, right_id)


def canonical_endpoints(left: dict[str, Any], right: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return (left, right) if endpoint_id(left) < endpoint_id(right) else (right, left)


def aligned_interval(row: dict[str, Any]) -> tuple[int, int] | None:
    frames = row.get("map_memory_raw_frame_indices")
    if not isinstance(frames, list) or len(frames) != 21:
        return None
    values = [int(value) for value in frames]
    return min(values), max(values)


def disjoint_intervals(left: dict[str, Any], right: dict[str, Any]) -> bool:
    a, b = aligned_interval(left), aligned_interval(right)
    return bool(a and b and (a[1] < b[0] or b[1] < a[0]))


def validate_aligned_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("hash", "game_id", "episode", "raw_start", "player_stem", "action_json"):
        if row.get(field) in (None, ""):
            errors.append(f"missing_{field}")
    for field in ("map_memory_raw_frame_indices", "alignment_latent_source_positions"):
        values = row.get(field)
        if not isinstance(values, list) or len(values) != 21:
            errors.append(f"{field}_not_21")
    if row.get("alignment_video_frames") is not None and int(row["alignment_video_frames"]) != 81:
        errors.append("alignment_video_frames_not_81")
    return errors


def load_and_deduplicate(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    dedup: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    payloads: dict[tuple[str, str, str, int, str], str] = {}
    source_counts: Counter[str] = Counter()
    identical_duplicates = 0
    conflicts: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for path in paths:
        for line_number, raw_row in iter_jsonl(path):
            source_counts[str(path)] += 1
            errors = validate_aligned_row(raw_row)
            if errors:
                rejected.append({
                    "stage": "source", "reason": "invalid_aligned_row", "detail": errors,
                    "source_manifest": str(path), "source_line": line_number,
                })
                continue
            key = row_key(raw_row)
            payload = canonical_json(raw_row)
            source = {"aligned_manifest": str(path), "line": line_number}
            if key not in dedup:
                row = dict(raw_row)
                row["_sources"] = [source]
                dedup[key] = row
                payloads[key] = payload
            elif payloads[key] == payload:
                identical_duplicates += 1
                dedup[key]["_sources"].append(source)
            else:
                conflict = {
                    "stage": "dedup", "reason": "conflicting_duplicate",
                    "row_key": list(key), "kept_sources": dedup[key]["_sources"],
                    "conflicting_source": source,
                    "kept_payload_sha256": hashlib.sha256(payloads[key].encode()).hexdigest(),
                    "conflicting_payload_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                }
                conflicts.append(conflict)
                rejected.append(conflict)

    stats = {
        "source_rows": sum(source_counts.values()),
        "source_rows_by_manifest": dict(source_counts),
        "deduplicated_rows": len(dedup),
        "identical_duplicate_rows_removed": identical_duplicates,
        "conflicting_duplicate_rows": len(conflicts),
        "invalid_source_rows": sum(1 for row in rejected if row["reason"] == "invalid_aligned_row"),
    }
    return list(dedup.values()), stats, rejected


def read_raw_signature_task(
    task: tuple[str, list[tuple[str, list[int]]]],
) -> dict[str, dict[str, Any]]:
    action_path, requests = task
    results: dict[str, dict[str, Any]] = {}
    try:
        with Path(action_path).open("r", encoding="utf-8") as handle:
            frames = json.load(handle)
        if not isinstance(frames, list):
            raise ValueError("raw JSON root is not a list")
    except Exception as exc:
        return {rid: {"valid": False, "error": f"raw_json_read_error:{exc}"} for rid, _ in requests}

    for rid, indices in requests:
        ticks: list[int] = []
        frame_counts: list[int] = []
        try:
            for index in indices:
                frame = frames[index]
                ticks.append(int(frame["tick"]))
                frame_counts.append(int(frame["frame_count"]))
        except Exception as exc:
            results[rid] = {"valid": False, "error": f"raw_frame_lookup_error:{exc}"}
            continue
        results[rid] = {"valid": True, "world_ticks": ticks, "frame_counts": frame_counts}
    return results


def populate_raw_signatures(
    rows: Iterable[dict[str, Any]],
    signatures: dict[str, dict[str, Any]],
    workers: int,
) -> None:
    by_path: dict[str, dict[str, list[int]]] = defaultdict(dict)
    for row in rows:
        rid = endpoint_id(row)
        if rid not in signatures:
            indices = row.get("map_memory_raw_frame_indices")
            if not isinstance(indices, list) or len(indices) != 21:
                signatures[rid] = {"valid": False, "error": "raw_indices_not_21"}
            else:
                by_path[str(row["action_json"])][rid] = [int(value) for value in indices]

    tasks = [
        (path, sorted(requests.items()))
        for path, requests in sorted(by_path.items())
    ]
    if workers == 1:
        result_iter = map(read_raw_signature_task, tasks)
        for result in result_iter:
            signatures.update(result)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(read_raw_signature_task, tasks, chunksize=8):
            signatures.update(result)


def first_mismatches(left: list[int], right: list[int], limit: int = 5) -> list[dict[str, int]]:
    return [
        {"position": index, "left": a, "right": b}
        for index, (a, b) in enumerate(zip(left, right)) if a != b
    ][:limit]


def positive_gate(
    left: dict[str, Any],
    right: dict[str, Any],
    signatures: dict[str, dict[str, Any]],
) -> tuple[bool, str | None, dict[str, Any]]:
    left_frames = [int(value) for value in left.get("map_memory_raw_frame_indices", [])]
    right_frames = [int(value) for value in right.get("map_memory_raw_frame_indices", [])]
    if len(left_frames) != 21 or len(right_frames) != 21:
        return False, "positive_raw_indices_not_21", {}
    if left_frames != right_frames:
        return False, "positive_raw_indices_mismatch", {"left": left_frames, "right": right_frames}
    left_sig, right_sig = signatures[endpoint_id(left)], signatures[endpoint_id(right)]
    if not left_sig["valid"] or not right_sig["valid"]:
        return False, "positive_raw_json_invalid", {
            "left_error": left_sig.get("error"), "right_error": right_sig.get("error"),
        }
    frame_equal = left_sig["frame_counts"] == right_sig["frame_counts"]
    tick_equal = left_sig["world_ticks"] == right_sig["world_ticks"]
    validity = {
        "raw_indices_equal": True,
        "frame_counts_equal": frame_equal,
        "world_ticks_equal": tick_equal,
        "all_21_valid": frame_equal and tick_equal,
    }
    if not frame_equal:
        return False, "positive_frame_count_mismatch", {
            "mismatches": first_mismatches(left_sig["frame_counts"], right_sig["frame_counts"]),
        }
    if not tick_equal:
        return False, "positive_tick_mismatch", {
            "mismatches": first_mismatches(left_sig["world_ticks"], right_sig["world_ticks"]),
        }
    return True, None, validity


def probe(items: list[Any], seed: str, limit: int) -> Iterator[Any]:
    if not items:
        return
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    count = len(items)
    start = int.from_bytes(digest[:8], "big") % count
    step = (int.from_bytes(digest[8:16], "big") % count) or 1
    while math.gcd(step, count) != 1:
        step = (step + 1) % count or 1
    for offset in range(min(count, limit)):
        yield items[(start + offset * step) % count]


def endpoint_reuse_limit(split: str, train_limit: int) -> int:
    return train_limit if split == "train" else 1


def can_reserve_pair(
    left: dict[str, Any],
    right: dict[str, Any],
    label: str,
    split: str,
    used_pairs: set[tuple[str, str]],
    endpoint_reuse: Counter[tuple[str, str, str]],
    train_limit: int,
) -> tuple[bool, str | None]:
    ids = pair_ids(left, right)
    if ids[0] == ids[1]:
        return False, "same_endpoint"
    if ids in used_pairs:
        return False, "duplicate_unordered_pair"
    if split_for_match(match_key(left)) != split or split_for_match(match_key(right)) != split:
        return False, "cross_split_leakage"
    limit = endpoint_reuse_limit(split, train_limit)
    if limit > 0 and any(endpoint_reuse[(split, label, rid)] >= limit for rid in ids):
        return False, "endpoint_reuse_limit"
    return True, None


def reserve_pair(
    left: dict[str, Any],
    right: dict[str, Any],
    label: str,
    split: str,
    used_pairs: set[tuple[str, str]],
    endpoint_reuse: Counter[tuple[str, str, str]],
) -> None:
    ids = pair_ids(left, right)
    used_pairs.add(ids)
    for rid in ids:
        endpoint_reuse[(split, label, rid)] += 1


def make_indexes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    rows_by_episode: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    rows_by_match: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
        rows_by_episode[episode_key(row)].append(row)
        rows_by_match[match_key(row)].append(row)
        rows_by_split[split_for_match(match_key(row))].append(row)
    for mapping in (groups, rows_by_episode, rows_by_match, rows_by_split):
        for key in mapping:
            mapping[key].sort(key=endpoint_id)
    group_views = {key: {str(row["player_stem"]): row for row in values} for key, values in groups.items()}
    groups_by_episode: dict[tuple[str, str, str], list[tuple[str, str, str, int]]] = defaultdict(list)
    for key in groups:
        groups_by_episode[key[:3]].append(key)
    for key in groups_by_episode:
        groups_by_episode[key].sort()
    return {
        "groups": groups, "group_views": group_views, "groups_by_episode": groups_by_episode,
        "rows_by_episode": rows_by_episode, "rows_by_match": rows_by_match,
        "rows_by_split": rows_by_split,
    }


def positive_candidates(indexes: dict[str, Any]) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    candidates: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for key in sorted(indexes["groups"]):
        rows = indexes["groups"][key]
        for i, left in enumerate(rows):
            for right in rows[i + 1:]:
                if left["player_stem"] == right["player_stem"]:
                    continue
                sid = sample_id("positive", left, right)
                candidates.append((sid, left, right))
    candidates.sort(key=lambda item: item[0])
    return candidates


def candidate_time_shifts(
    left: dict[str, Any], right: dict[str, Any], indexes: dict[str, Any], seed: str, limit: int,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    source_group = group_key(left)
    left_view, right_view = str(left["player_stem"]), str(right["player_stem"])
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for alternate_group in indexes["groups_by_episode"][episode_key(left)]:
        if alternate_group == source_group:
            continue
        views = indexes["group_views"][alternate_group]
        if left_view not in views or right_view not in views:
            continue
        alternate_left, alternate_right = views[left_view], views[right_view]
        if disjoint_intervals(left, alternate_right):
            candidates.append((left, alternate_right))
        if disjoint_intervals(right, alternate_left):
            candidates.append((right, alternate_left))
    candidates.sort(key=lambda pair: pair_ids(*pair))
    yield from probe(candidates, seed, limit)


def candidate_wrong_windows(
    left: dict[str, Any], right: dict[str, Any], indexes: dict[str, Any], seed: str, limit: int,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    source_group = group_key(left)
    anchors = [left, right]
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    source_views = set(indexes["group_views"][source_group])
    for other in indexes["rows_by_episode"][episode_key(left)]:
        other_group = group_key(other)
        if other_group == source_group:
            continue
        other_group_views = set(indexes["group_views"][other_group])
        for anchor in anchors:
            if anchor["player_stem"] == other["player_stem"] or not disjoint_intervals(anchor, other):
                continue
            pair_view_ids = {str(anchor["player_stem"]), str(other["player_stem"])}
            if pair_view_ids <= source_views and pair_view_ids <= other_group_views:
                continue  # This relation is time-shift eligible by protocol.
            candidates.append((anchor, other))
    candidates.sort(key=lambda pair: pair_ids(*pair))
    yield from probe(candidates, seed, limit)


def candidate_cross_episodes(
    left: dict[str, Any], right: dict[str, Any], indexes: dict[str, Any], seed: str, limit: int,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    anchors = (left, right)
    pool = indexes["rows_by_match"][match_key(left)]
    for other in probe(pool, seed, limit):
        if other["episode"] == left["episode"]:
            continue
        anchor_index = int(stable_hash(seed, endpoint_id(other))[:8], 16) % 2
        yield anchors[anchor_index], other


def candidate_cross_matches(
    left: dict[str, Any], right: dict[str, Any], indexes: dict[str, Any], seed: str, limit: int,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    anchors = (left, right)
    split = split_for_match(match_key(left))
    for other in probe(indexes["rows_by_split"][split], seed, limit):
        if match_key(other) == match_key(left):
            continue
        anchor_index = int(stable_hash(seed, endpoint_id(other))[:8], 16) % 2
        yield anchors[anchor_index], other


def endpoint_payload(row: dict[str, Any], signatures: dict[str, dict[str, Any]]) -> dict[str, Any]:
    signature = signatures[endpoint_id(row)]
    payload = {field: row.get(field) for field in REFERENCE_FIELDS if field in row}
    payload.update({
        "endpoint_id": endpoint_id(row),
        "split_key": f"{row['hash']}|{row['game_id']}",
        "world_ticks": signature["world_ticks"],
        "frame_counts": signature["frame_counts"],
        "aligned_manifest_sources": row["_sources"],
    })
    return payload


def pair_validity(left: dict[str, Any], right: dict[str, Any], signatures: dict[str, dict[str, Any]]) -> dict[str, Any]:
    left_sig, right_sig = signatures[endpoint_id(left)], signatures[endpoint_id(right)]
    raw_equal = left.get("map_memory_raw_frame_indices") == right.get("map_memory_raw_frame_indices")
    frames_equal = left_sig["frame_counts"] == right_sig["frame_counts"]
    ticks_equal = left_sig["world_ticks"] == right_sig["world_ticks"]
    return {
        "left_endpoint_raw_valid": bool(left_sig["valid"]),
        "right_endpoint_raw_valid": bool(right_sig["valid"]),
        "raw_indices_equal": raw_equal,
        "frame_counts_equal": frames_equal,
        "world_ticks_equal": ticks_equal,
        "frame_count_equal_mask": [a == b for a, b in zip(left_sig["frame_counts"], right_sig["frame_counts"])],
        "world_tick_equal_mask": [a == b for a, b in zip(left_sig["world_ticks"], right_sig["world_ticks"])],
        "all_21_endpoint_records_valid": True,
    }


def pair_payload(
    label: str,
    left: dict[str, Any],
    right: dict[str, Any],
    signatures: dict[str, dict[str, Any]],
    anchor_positive_id: str | None,
) -> dict[str, Any]:
    left, right = canonical_endpoints(left, right)
    split = split_for_match(match_key(left))
    sid = sample_id(label, left, right)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "sample_id": sid,
        "label": label,
        "is_consistent": label == "positive",
        "negative_difficulty": LABEL_DIFFICULTY[label],
        "split": split,
        "left": endpoint_payload(left, signatures),
        "right": endpoint_payload(right, signatures),
        "pair_validity": pair_validity(left, right, signatures),
        "provenance": {
            "builder": str(Path(__file__).resolve()),
            "anchor_positive_sample_id": anchor_positive_id,
            "selection_hash": stable_hash(PROTOCOL_VERSION, anchor_positive_id or sid, label),
        },
    }
    if label == "time_shift":
        earlier = left if int(left["raw_start"]) < int(right["raw_start"]) else right
        payload["time_shift"] = {
            "earlier_endpoint_id": endpoint_id(earlier),
            "raw_start_delta_right_minus_left": int(right["raw_start"]) - int(left["raw_start"]),
            "raw_intervals_disjoint": disjoint_intervals(left, right),
        }
    return payload


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows, source_stats, rejected = load_and_deduplicate(args.aligned_manifest)
    if source_stats["conflicting_duplicate_rows"] and not args.allow_conflicting_duplicates:
        report = {
            "protocol_version": PROTOCOL_VERSION, "status": "failed_conflicting_duplicates",
            "source": source_stats, "rejection_reasons": dict(Counter(row["reason"] for row in rejected)),
        }
        return report, rejected, {"train": [], "val": [], "test": []}

    indexes = make_indexes(rows)
    split_match_counts = Counter(split_for_match(key) for key in indexes["rows_by_match"])
    split_row_counts = Counter(split_for_match(match_key(row)) for row in rows)
    candidates = positive_candidates(indexes)
    candidate_count_before_limit = len(candidates)
    if args.max_positive_pairs is not None:
        candidates = candidates[: args.max_positive_pairs]

    signatures: dict[str, dict[str, Any]] = {}
    populate_raw_signatures(
        (row for _, left, right in candidates for row in (left, right)), signatures, args.raw_json_workers,
    )

    gate_passed: list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for sid, left, right in candidates:
        passed, reason, detail = positive_gate(left, right, signatures)
        if not passed:
            rejected.append({
                "stage": "positive_tick_gate", "reason": reason, "sample_id": sid,
                "left_endpoint_id": endpoint_id(left), "right_endpoint_id": endpoint_id(right), "detail": detail,
            })
            continue
        gate_passed.append((sid, left, right, detail))

    used_pairs: set[tuple[str, str]] = set()
    endpoint_reuse: Counter[tuple[str, str, str]] = Counter()
    emitted_positives: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    duplicate_pair_attempts = 0
    cross_split_leakage = 0
    for sid, left, right, _ in gate_passed:
        split = split_for_match(match_key(left))
        allowed, reason = can_reserve_pair(
            left, right, "positive", split, used_pairs, endpoint_reuse, args.train_endpoint_reuse_limit,
        )
        if not allowed:
            duplicate_pair_attempts += int(reason == "duplicate_unordered_pair")
            cross_split_leakage += int(reason == "cross_split_leakage")
            rejected.append({
                "stage": "positive_reuse_gate", "reason": reason, "sample_id": sid,
                "left_endpoint_id": endpoint_id(left), "right_endpoint_id": endpoint_id(right),
            })
            continue
        reserve_pair(left, right, "positive", split, used_pairs, endpoint_reuse)
        emitted_positives.append((sid, left, right))

    selected_negatives: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
    candidate_functions = {
        "time_shift": candidate_time_shifts,
        "wrong_window": candidate_wrong_windows,
        "cross_episode": candidate_cross_episodes,
        "cross_match": candidate_cross_matches,
    }
    for positive_id, left, right in emitted_positives:
        split = split_for_match(match_key(left))
        for label, candidate_function in candidate_functions.items():
            seed = stable_hash(PROTOCOL_VERSION, positive_id, label)
            chosen: tuple[dict[str, Any], dict[str, Any]] | None = None
            failure_counts: Counter[str] = Counter()
            for candidate_left, candidate_right in candidate_function(
                left, right, indexes, seed, args.negative_search_limit,
            ):
                allowed, reason = can_reserve_pair(
                    candidate_left, candidate_right, label, split, used_pairs, endpoint_reuse,
                    args.train_endpoint_reuse_limit,
                )
                if allowed:
                    chosen = candidate_left, candidate_right
                    break
                failure_counts[str(reason)] += 1
                duplicate_pair_attempts += int(reason == "duplicate_unordered_pair")
                cross_split_leakage += int(reason == "cross_split_leakage")
            if chosen is None:
                rejected.append({
                    "stage": "negative_sampling", "reason": f"no_{label}_candidate",
                    "anchor_positive_sample_id": positive_id, "split": split,
                    "candidate_failure_reasons": dict(failure_counts),
                })
                continue
            reserve_pair(*chosen, label, split, used_pairs, endpoint_reuse)
            selected_negatives.append((label, positive_id, chosen[0], chosen[1]))

    populate_raw_signatures(
        (row for _, _, left, right in selected_negatives for row in (left, right)), signatures,
        args.raw_json_workers,
    )

    outputs: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for sid, left, right in emitted_positives:
        outputs[split_for_match(match_key(left))].append(pair_payload("positive", left, right, signatures, None))
    invalid_negative_raw = 0
    for label, anchor_id, left, right in selected_negatives:
        left_sig, right_sig = signatures[endpoint_id(left)], signatures[endpoint_id(right)]
        if not left_sig["valid"] or not right_sig["valid"]:
            invalid_negative_raw += 1
            rejected.append({
                "stage": "negative_raw_gate", "reason": "negative_raw_json_invalid", "label": label,
                "anchor_positive_sample_id": anchor_id, "left_endpoint_id": endpoint_id(left),
                "right_endpoint_id": endpoint_id(right), "left_error": left_sig.get("error"),
                "right_error": right_sig.get("error"),
            })
            continue
        outputs[split_for_match(match_key(left))].append(pair_payload(label, left, right, signatures, anchor_id))

    for split in outputs:
        outputs[split].sort(key=lambda row: row["sample_id"])

    label_counts = {split: dict(Counter(row["label"] for row in values)) for split, values in outputs.items()}
    reuse_summary: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        reuse_summary[split] = {}
        for label in LABEL_DIFFICULTY:
            counts = [count for (s, candidate_label, _), count in endpoint_reuse.items() if s == split and candidate_label == label]
            reuse_summary[split][label] = {
                "endpoint_count": len(counts), "max_reuse": max(counts, default=0),
                "reuse_histogram": dict(Counter(counts)),
                "configured_limit": endpoint_reuse_limit(split, args.train_endpoint_reuse_limit),
            }

    emitted_pair_ids = [pair_ids_from_payload(row) for values in outputs.values() for row in values]
    duplicate_emitted_pairs = len(emitted_pair_ids) - len(set(emitted_pair_ids))
    actual_cross_split = sum(
        1 for values in outputs.values() for row in values
        if row["left"]["split_key"] != row["right"]["split_key"]
        and split_for_endpoint_payload(row["left"]) != split_for_endpoint_payload(row["right"])
    )
    rejection_counts = Counter(str(row["reason"]) for row in rejected)
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "dry_run" if args.dry_run else "complete",
        "aligned_manifests": [str(path) for path in args.aligned_manifest],
        "source": source_stats,
        "groups": len(indexes["groups"]),
        "multi_view_groups": sum(len(values) >= 2 for values in indexes["groups"].values()),
        "split": {
            "salt": SPLIT_SALT, "unit": ["hash", "game_id"], "fractions": [0.8, 0.1, 0.1],
            "match_counts": dict(split_match_counts), "deduplicated_row_counts": dict(split_row_counts),
        },
        "positive": {
            "pairwise_gate_candidates_before_max_limit": candidate_count_before_limit,
            "pairwise_gate_candidates_considered": len(candidates),
            "pairwise_tick_gate_passed": len(gate_passed),
            "emitted_after_reuse_gate": len(emitted_positives),
        },
        "output_counts_by_split_and_label": label_counts,
        "output_rows_by_split": {split: len(values) for split, values in outputs.items()},
        "endpoint_reuse": reuse_summary,
        "train_endpoint_reuse_limit": args.train_endpoint_reuse_limit,
        "negative_search_limit": args.negative_search_limit,
        "raw_json_workers": args.raw_json_workers,
        "invalid_negative_raw_pairs": invalid_negative_raw,
        "duplicate_pair_attempts_avoided": duplicate_pair_attempts,
        "duplicate_unordered_pairs_emitted": duplicate_emitted_pairs,
        "cross_split_candidate_attempts_rejected": cross_split_leakage,
        "cross_split_leakage_emitted": actual_cross_split,
        "rejected_pairs": len(rejected),
        "rejection_reasons": dict(rejection_counts),
    }
    return report, rejected, outputs


def pair_ids_from_payload(row: dict[str, Any]) -> tuple[str, str]:
    return tuple(sorted((row["left"]["endpoint_id"], row["right"]["endpoint_id"])))  # type: ignore[return-value]


def split_for_endpoint_payload(endpoint: dict[str, Any]) -> str:
    root_hash, game_id = endpoint["split_key"].split("|", 1)
    return split_for_match((root_hash, game_id))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/multiview_consistency_mvp_v1"))
    parser.add_argument("--dry-run", action="store_true", help="Run all gates/sampling and print the report without writing outputs.")
    parser.add_argument("--max-positive-pairs", type=int, default=None)
    parser.add_argument("--train-endpoint-reuse-limit", type=int, default=8, help="Per endpoint and label; 0 means unlimited. Val/test are always 1.")
    parser.add_argument("--negative-search-limit", type=int, default=512, help="Maximum deterministic probes per anchor/subtype.")
    parser.add_argument("--raw-json-workers", type=int, default=16, help="Worker processes for read-only raw JSON extraction.")
    parser.add_argument("--allow-conflicting-duplicates", action="store_true")
    args = parser.parse_args()
    if args.max_positive_pairs is not None and args.max_positive_pairs < 1:
        parser.error("--max-positive-pairs must be >= 1")
    if args.train_endpoint_reuse_limit < 0:
        parser.error("--train-endpoint-reuse-limit must be >= 0")
    if args.negative_search_limit < 1:
        parser.error("--negative-search-limit must be >= 1")
    if args.raw_json_workers < 1:
        parser.error("--raw-json-workers must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    report, rejected, outputs = build(args)
    if report["status"] == "failed_conflicting_duplicates":
        if not args.dry_run:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            write_json(args.output_dir / "build_report.json", report)
            write_jsonl(args.output_dir / "rejected_pairs.jsonl", rejected)
        raise RuntimeError(
            f"found {report['source']['conflicting_duplicate_rows']} conflicting duplicates; "
            "inspect rejected_pairs.jsonl or pass --allow-conflicting-duplicates"
        )
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in outputs.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)
    write_jsonl(args.output_dir / "rejected_pairs.jsonl", rejected)
    write_json(args.output_dir / "build_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
