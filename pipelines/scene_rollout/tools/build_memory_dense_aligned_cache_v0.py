#!/usr/bin/env python3
"""Build a frame-exact LingBot latent-cache manifest aligned to Map Memory dense.

This is a formal cache builder, not a smoke helper.  It only admits LingBot
clips whose every latent frame has an exact Map Memory dense sample in the
accepted release.  Rows with missing dense frames are rejected and reported; the
script never fills gaps by nearest-neighbor lookup or static repetition.
"""

from __future__ import annotations
from runtime_paths import source_path

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np

from map_memory_training_data_v0 import REQUIRED_BACKEND_ID, MapMemoryRelease, sha256_file, split_samples


DEFAULT_MAP_MANIFEST = Path("docs/assets/memory_dense_dataset_v43_bsp_all5_match_large_h176/manifest.json")
DEFAULT_CACHE_MANIFEST = Path(str(source_path('assets', 'datasets/fast_cam_dust2_p0_32f_done_cpfs/latents_full_32f/cache_manifest.jsonl')))
DEFAULT_SOURCE_MANIFEST = Path(str(source_path('assets', 'datasets/fast_cam_dust2_p0_32f_done_cpfs/manifests/32f_done_train_cache_input_cpfs.jsonl')))
DEFAULT_OUT_JSONL = Path("output/memory_dense_adapter_v0/aligned_cache_v43.jsonl")
DEFAULT_OUT_REPORT = Path("output/memory_dense_adapter_v0/aligned_cache_v43_report.json")

CLIP_ID_RE = re.compile(
    r"^(?P<hash>[^_]+)_(?P<game_id>[0-9a-fA-F]{32})_(?P<episode>Ep_\d{6})_"
    r"(?P<player_stem>Ep_\d{6}_team_\d+_player_\d+_inst_\d+)_(?P<raw_start>\d{7})$"
)

SOURCE_FIELDS_TO_PRESERVE = [
    "hash",
    "game_id",
    "episode",
    "player_stem",
    "raw_start",
    "raw_indices",
    "frame_count_start",
    "frame_count_end",
    "round_freeze_end",
    "window_stride_raw",
    "dynamic_score",
    "mp4",
    "action_json",
    "episode_info",
    "video_manifest",
    "game_manifest",
    "world_events",
    "map_name",
    "pose_version",
    "hfov_source",
    "sample_dir",
    "clip_dir",
    "meta_json",
    "snapshot_source",
    "source_manifest_index",
    "source_shard_id",
    "source_shard_position_1based",
]


def iter_jsonl(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        count = 0
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)
            count += 1
            if limit is not None and count >= limit:
                break


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def clip_id_from_source_row(row: dict[str, Any]) -> str | None:
    clip_id = row.get("clip_id")
    if clip_id:
        return str(clip_id)
    required = ["hash", "game_id", "episode", "player_stem", "raw_start"]
    if all(key in row for key in required):
        return f"{row['hash']}_{row['game_id']}_{row['episode']}_{row['player_stem']}_{int(row['raw_start']):07d}"
    return None


def parse_clip_identity(record: dict[str, Any], source: dict[str, Any] | None) -> dict[str, Any]:
    merged = source or {}
    clip_id = str(record.get("clip_id") or merged.get("clip_id") or "")
    parsed = CLIP_ID_RE.match(clip_id)
    out: dict[str, Any] = {"clip_id": clip_id}
    for key in ["hash", "game_id", "episode", "player_stem", "raw_start"]:
        if key in merged:
            out[key] = merged[key]
    if parsed:
        groups = parsed.groupdict()
        out.setdefault("hash", groups["hash"])
        out.setdefault("game_id", groups["game_id"])
        out.setdefault("episode", groups["episode"])
        out.setdefault("player_stem", groups["player_stem"])
        out.setdefault("raw_start", int(groups["raw_start"]))
    missing = [key for key in ["game_id", "episode", "player_stem", "raw_start"] if key not in out or out[key] in ("", None)]
    if missing:
        raise ValueError(f"{clip_id or '<unknown>'}: cannot derive {missing} from cache/source metadata")
    out["game_id"] = str(out["game_id"])
    out["episode"] = str(out["episode"])
    out["player_stem"] = str(out["player_stem"])
    out["raw_start"] = int(out["raw_start"])
    return out


def load_source_index(source_manifest: Path | None, wanted_clip_ids: set[str]) -> dict[str, dict[str, Any]]:
    if source_manifest is None or not source_manifest.exists():
        return {}
    index: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(source_manifest):
        clip_id = clip_id_from_source_row(row)
        if not clip_id or clip_id not in wanted_clip_ids:
            continue
        slim = {key: row[key] for key in SOURCE_FIELDS_TO_PRESERVE if key in row}
        slim["clip_id"] = clip_id
        index[clip_id] = slim
        if len(index) == len(wanted_clip_ids):
            break
    return index


def latent_frame_source_positions(video_frames: int, latent_frames: int) -> list[int]:
    if latent_frames < 1:
        raise ValueError("latent_frames must be >= 1")
    if latent_frames == 1:
        return [0]
    positions = np.linspace(0, video_frames - 1, latent_frames)
    rounded = np.rint(positions).astype(np.int64)
    if not np.allclose(positions, rounded, atol=1e-6):
        raise ValueError(
            f"video_frames={video_frames} and latent_frames={latent_frames} do not give integer source positions: "
            f"first={positions[:5].tolist()}"
        )
    return [int(x) for x in rounded.tolist()]


def expected_sample_ids(identity: dict[str, Any], *, video_frames: int, raw_stride: int, latent_frames: int) -> tuple[list[int], list[str]]:
    positions = latent_frame_source_positions(video_frames, latent_frames)
    raw_start = int(identity["raw_start"])
    raw_frames = [raw_start + pos * raw_stride for pos in positions]
    sample_ids = [
        f"{identity['game_id']}__{identity['episode']}_{identity['player_stem']}_f{frame_index:06d}"
        for frame_index in raw_frames
    ]
    return raw_frames, sample_ids


def validate_record_shapes(record: dict[str, Any], *, latent_frames: int, video_height: int, video_width: int) -> list[str]:
    errors: list[str] = []
    shape = record.get("shape")
    expected_hw = [video_height // 8, video_width // 8]
    if isinstance(shape, list) and len(shape) == 4:
        if int(shape[1]) != latent_frames:
            errors.append(f"shape latent frames {shape[1]} != {latent_frames}")
        if [int(shape[2]), int(shape[3])] != expected_hw:
            errors.append(f"shape latent hw {shape[2:4]} != {expected_hw}")
    else:
        errors.append("missing or invalid latent shape")
    condition_shape = record.get("condition_shape")
    if isinstance(condition_shape, list) and len(condition_shape) == 4:
        if int(condition_shape[1]) != latent_frames:
            errors.append(f"condition_shape latent frames {condition_shape[1]} != {latent_frames}")
        if [int(condition_shape[2]), int(condition_shape[3])] != expected_hw:
            errors.append(f"condition_shape latent hw {condition_shape[2:4]} != {expected_hw}")
    else:
        errors.append("missing or invalid condition_shape")
    expected = record.get("latent_frames_expected")
    if expected is not None and int(expected) != latent_frames:
        errors.append(f"latent_frames_expected {expected} != {latent_frames}")
    if record.get("dtype") == "metadata_only":
        errors.append("metadata_only cache row")
    for key in ["latent_cache", "text_cache", "poses", "intrinsics"]:
        if not record.get(key):
            errors.append(f"missing {key}")
    return errors


def validate_source_raw_indices(source: dict[str, Any] | None, raw_frames: list[int], *, raw_stride: int, video_frames: int) -> list[str]:
    if not source or "raw_indices" not in source:
        return []
    raw_indices = source["raw_indices"]
    if not isinstance(raw_indices, list):
        return ["source raw_indices is not a list"]
    if len(raw_indices) != video_frames:
        return [f"source raw_indices length {len(raw_indices)} != video_frames {video_frames}"]
    positions = latent_frame_source_positions(video_frames, len(raw_frames))
    observed = [int(raw_indices[pos]) for pos in positions]
    if observed != raw_frames:
        return [f"source raw_indices latent frames {observed[:5]}... != expected {raw_frames[:5]}..."]
    if len(raw_indices) > 1:
        strides = {int(raw_indices[i + 1]) - int(raw_indices[i]) for i in range(len(raw_indices) - 1)}
        if strides != {raw_stride}:
            return [f"source raw_indices strides {sorted(strides)} != [{raw_stride}]"]
    return []


def split_lookup_for_release(
    release: MapMemoryRelease,
    *,
    split_key: str,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, str]:
    splits = split_samples(
        release.samples,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        split_key=split_key,
        seed=seed,
    )
    out: dict[str, str] = {}
    for split, samples in splits.items():
        for sample in samples:
            out[sample.sample_id] = split
    return out


def sample_roles(release: MapMemoryRelease, sample_ids: list[str]) -> list[str]:
    return [release.by_id[sid].selection_role for sid in sample_ids]


def reject(reject_counts: Counter[str], reject_examples: list[dict[str, Any]], *, clip_id: str, reason: str, detail: Any, max_examples: int) -> None:
    reject_counts[reason] += 1
    if len(reject_examples) < max_examples:
        reject_examples.append({"clip_id": clip_id, "reason": reason, "detail": detail})


def build_aligned_cache(args: argparse.Namespace) -> dict[str, Any]:
    release = MapMemoryRelease.load(args.map_manifest, require_backend_id=args.require_backend_id)
    map_manifest_sha256 = sha256_file(args.map_manifest)
    split_by_sample_id = split_lookup_for_release(
        release,
        split_key=args.split_key,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    accepted = 0
    scanned = 0
    source_rows_loaded = 0
    split_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    match_counts: Counter[str] = Counter()
    episode_counts: Counter[str] = Counter()
    reject_counts: Counter[str] = Counter()
    reject_examples: list[dict[str, Any]] = []
    accepted_pending: list[tuple[dict[str, Any], dict[str, Any], list[int], list[str], list[str], str]] = []

    for record in iter_jsonl(args.cache_manifest, limit=args.limit):
        scanned += 1
        clip_id = str(record.get("clip_id", ""))
        try:
            identity = parse_clip_identity(record, None)
        except Exception as exc:
            reject(reject_counts, reject_examples, clip_id=clip_id, reason="bad_clip_identity", detail=str(exc), max_examples=args.max_reject_examples)
            continue
        shape_errors = validate_record_shapes(
            record,
            latent_frames=args.latent_frames,
            video_height=args.video_height,
            video_width=args.video_width,
        )
        if shape_errors:
            reject(reject_counts, reject_examples, clip_id=clip_id, reason="bad_cache_shape", detail=shape_errors, max_examples=args.max_reject_examples)
            continue
        raw_frames, sample_ids = expected_sample_ids(
            identity,
            video_frames=args.video_frames,
            raw_stride=args.raw_stride,
            latent_frames=args.latent_frames,
        )
        missing_ids = [sid for sid in sample_ids if sid not in release.by_id]
        if missing_ids:
            reject(
                reject_counts,
                reject_examples,
                clip_id=clip_id,
                reason="missing_map_memory_samples",
                detail={"missing_count": len(missing_ids), "first_missing": missing_ids[:5]},
                max_examples=args.max_reject_examples,
            )
            continue
        splits = {split_by_sample_id[sid] for sid in sample_ids}
        if len(splits) != 1:
            reject(reject_counts, reject_examples, clip_id=clip_id, reason="mixed_split_within_clip", detail=sorted(splits), max_examples=args.max_reject_examples)
            continue
        roles = sample_roles(release, sample_ids)
        split = next(iter(splits))
        accepted_pending.append((record, identity, raw_frames, sample_ids, roles, split))

    source_index = load_source_index(args.source_manifest, {str(row.get("clip_id", "")) for row, *_ in accepted_pending})
    source_rows_loaded = len(source_index)

    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for record, identity, raw_frames, sample_ids, roles, split in accepted_pending:
            clip_id = str(record.get("clip_id", ""))
            source = source_index.get(clip_id)
            if args.require_source_raw_indices and not source:
                reject(
                    reject_counts,
                    reject_examples,
                    clip_id=clip_id,
                    reason="missing_source_raw_indices",
                    detail=f"source manifest has no row for {clip_id}",
                    max_examples=args.max_reject_examples,
                )
                continue
            raw_errors = validate_source_raw_indices(source, raw_frames, raw_stride=args.raw_stride, video_frames=args.video_frames)
            if raw_errors and args.require_source_raw_indices:
                reject(reject_counts, reject_examples, clip_id=clip_id, reason="bad_source_raw_indices", detail=raw_errors, max_examples=args.max_reject_examples)
                continue
            out = dict(record)
            if source:
                for key, value in source.items():
                    if key == "raw_indices":
                        continue
                    out.setdefault(key, value)
            out.update(
                {
                    "alignment_kind": "map_memory_dense_lingbot_latent_frame_exact_v0",
                    "map_memory_manifest": str(args.map_manifest),
                    "map_memory_manifest_sha256": map_manifest_sha256,
                    "map_memory_sample_ids": sample_ids,
                    "map_memory_raw_frame_indices": raw_frames,
                    "map_memory_split": split,
                    "map_memory_split_key": args.split_key,
                    "map_memory_match_id": identity["game_id"],
                    "map_memory_episode": f"{identity['game_id']}_{identity['episode']}",
                    "map_memory_raw_episode": identity["episode"],
                    "map_memory_ego_stem": identity["player_stem"],
                    "map_memory_track_id": f"{identity['game_id']}|{identity['episode']}|{identity['player_stem']}",
                    "map_memory_selection_roles": roles,
                    "map_memory_positive_frames": sum(1 for role in roles if role == "positive"),
                    "map_memory_context_frames": sum(1 for role in roles if role == "context"),
                    "alignment_video_frames": args.video_frames,
                    "alignment_raw_stride": args.raw_stride,
                    "alignment_latent_frames": args.latent_frames,
                    "alignment_latent_source_positions": latent_frame_source_positions(args.video_frames, args.latent_frames),
                    "alignment_exact_frame_match": True,
                }
            )
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            accepted += 1
            split_counts[split] += 1
            match_counts[str(identity["game_id"])] += 1
            episode_counts[f"{identity['game_id']}_{identity['episode']}"] += 1
            role_counts.update(roles)

    report = {
        "kind": "map_memory_dense_aligned_cache_build_v0",
        "status": "pass" if accepted >= args.min_accepted and not (args.fail_on_rejects and reject_counts) else "fail",
        "map_manifest": str(args.map_manifest),
        "cache_manifest": str(args.cache_manifest),
        "source_manifest": str(args.source_manifest) if args.source_manifest else None,
        "out_jsonl": str(args.out_jsonl),
        "map_manifest_sha256": map_manifest_sha256,
        "scanned_cache_rows": scanned,
        "accepted_rows": accepted,
        "rejected_rows": scanned - accepted,
        "min_accepted": args.min_accepted,
        "split_key": args.split_key,
        "split_counts": dict(sorted(split_counts.items())),
        "match_counts": dict(sorted(match_counts.items())),
        "episode_count": len(episode_counts),
        "latent_frame_role_counts": dict(sorted(role_counts.items())),
        "reject_counts": dict(sorted(reject_counts.items())),
        "reject_examples": reject_examples,
        "alignment": {
            "video_frames": args.video_frames,
            "raw_stride": args.raw_stride,
            "latent_frames": args.latent_frames,
            "latent_source_positions": latent_frame_source_positions(args.video_frames, args.latent_frames),
            "frame_formula": "raw_start + raw_stride * latent_source_position",
            "nearest_or_repeat_used": False,
        },
        "source_rows_loaded": source_rows_loaded,
        "cache_clip_ids_seen": scanned,
    }
    if args.out_report:
        write_json(args.out_report, report)
    if report["status"] != "pass":
        raise SystemExit(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map-manifest", type=Path, default=DEFAULT_MAP_MANIFEST)
    ap.add_argument("--cache-manifest", type=Path, default=DEFAULT_CACHE_MANIFEST)
    ap.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    ap.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT_JSONL)
    ap.add_argument("--out-report", type=Path, default=DEFAULT_OUT_REPORT)
    ap.add_argument("--require-backend-id", default=REQUIRED_BACKEND_ID)
    ap.add_argument("--video-width", type=int, default=832)
    ap.add_argument("--video-height", type=int, default=480)
    ap.add_argument("--video-frames", type=int, default=81)
    ap.add_argument("--raw-stride", type=int, default=2)
    ap.add_argument("--latent-frames", type=int, default=21)
    ap.add_argument("--split-key", choices=["episode", "track", "match"], default="episode")
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--test-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260531)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-accepted", type=int, default=1)
    ap.add_argument("--fail-on-rejects", action="store_true")
    ap.add_argument("--hash-manifest", action="store_true", help="Deprecated; manifest hash is always written for train-safety.")
    ap.add_argument("--require-source-raw-indices", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-reject-examples", type=int, default=50)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    report = build_aligned_cache(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
