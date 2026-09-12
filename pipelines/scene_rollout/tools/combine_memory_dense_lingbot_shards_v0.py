#!/usr/bin/env python3
"""Combine strict LingBot-cache Map Memory export shards into one train release.

This is a promotion tool, not a sampling helper.  It accepts only shard outputs
from ``export_memory_dense_from_lingbot_cache_v0.py`` whose readiness check
passed, merges complete aligned cache rows, deduplicates samples by
``sample_id``, rewrites sample sidecar paths into a self-contained output
release, reruns the readiness checker, and rewrites every aligned row with the
final release manifest SHA-256.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from map_memory_training_data_v0 import CHANNELS, REQUIRED_BACKEND_ID, sha256_file


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_sample_path(shard_dir: Path, sample: dict[str, Any], key: str, rel_key: str) -> Path:
    rel = sample.get(rel_key)
    if rel:
        path = shard_dir / str(rel)
        if path.exists():
            return path
    raw = Path(str(sample.get(key, "")))
    if raw.exists():
        return raw
    raise FileNotFoundError(f"{sample.get('sample_id', '<unknown>')}: missing {key}")


def copy_or_link(src: Path, dst: Path, *, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def normalize_sample(
    sample: dict[str, Any],
    *,
    shard_dir: Path,
    out_dir: Path,
    link_mode: str,
) -> dict[str, Any]:
    sid = str(sample["sample_id"])
    sample_dir = out_dir / "samples" / sid
    files = [
        ("dense_path", "dense_relpath", "mesh_dense_condition_v0.npz"),
        ("target_rgb_path", "target_rgb_relpath", "target_rgb.png"),
        ("meta_path", "meta_relpath", "mesh_dense_condition_meta_v0.json"),
        ("qa_path", "qa_relpath", "mesh_dense_condition_qa_v0.png"),
    ]
    out = dict(sample)
    for key, rel_key, filename in files:
        src = resolve_sample_path(shard_dir, sample, key, rel_key)
        dst = sample_dir / filename
        copy_or_link(src, dst, mode=link_mode)
        out[key] = str(dst)
        out[rel_key] = str(dst.relative_to(out_dir))
    out["channels"] = CHANNELS
    return out


def sample_fingerprint(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "shape": sample.get("shape"),
        "channels": sample.get("channels"),
        "selection_role": sample.get("selection_role"),
        "match_id": sample.get("match_id"),
        "episode": sample.get("episode"),
        "raw_episode": sample.get("raw_episode"),
        "ego_stem": sample.get("ego_stem"),
        "frame_index": sample.get("frame_index"),
        "geometry_backend_id": sample.get("geometry_backend_id"),
        "backend_signature": sample.get("backend_signature"),
        "teacher_qa_at_selection": sample.get("teacher_qa_at_selection"),
    }


def qa_fingerprint(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "sample_id",
        "episode",
        "raw_episode",
        "match_id",
        "ego_stem",
        "frame_index",
        "visible_teacher_players",
        "other_player_teacher_pixels",
        "other_player_memory_pixels",
        "hit_iou",
        "semantic_binary_iou",
        "other_player_iou",
        "depth_common",
        "other_player_depth_common",
    ]
    return {key: row.get(key) for key in keys}


def summarize_qa(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_of(key: str) -> float | None:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    def nested_mean(key: str, subkey: str) -> float | None:
        vals = [float(r[key][subkey]) for r in rows if r.get(key, {}).get(subkey) is not None]
        return sum(vals) / len(vals) if vals else None

    positives = [
        r for r in rows
        if int(r.get("visible_teacher_players", 0) or 0) > 0
        or int(r.get("other_player_teacher_pixels", 0) or 0) > 0
    ]
    pos_ious = [float(r.get("other_player_iou", 0.0) or 0.0) for r in positives]
    return {
        "hit_iou_mean": mean_of("hit_iou"),
        "semantic_binary_iou_mean": mean_of("semantic_binary_iou"),
        "other_player_iou_mean": mean_of("other_player_iou"),
        "positive_other_player_iou_mean": sum(pos_ious) / len(pos_ious) if pos_ious else None,
        "positive_other_player_iou_min": min(pos_ious) if pos_ious else None,
        "depth_common_mae_mean": nested_mean("depth_common", "mae"),
        "other_player_depth_common_mae_mean": nested_mean("other_player_depth_common", "mae"),
    }


def is_positive_other_iou_mean_failure(message: Any) -> bool:
    text = str(message)
    return "positive_other_player_iou_mean" in text or "positive other_player_iou_mean below 0.45" in text


def allowed_relaxed_readiness_failures(failures: list[Any]) -> bool:
    return bool(failures) and all(is_positive_other_iou_mean_failure(item) for item in failures)


def validate_shard(
    shard_dir: Path,
    *,
    allow_report_fail_for_min_positive: bool,
    allow_positive_other_iou_mean_fail: bool,
) -> dict[str, Any]:
    report_path = shard_dir / "export_report_v0.json"
    manifest_path = shard_dir / "manifest.json"
    readiness_path = shard_dir / "training_readiness_v0.json"
    qa_path = shard_dir / "channel_teacher_qa_v0" / "memory_dense_channels_vs_teacher_v0.json"
    aligned_path = shard_dir / "aligned_cache_manifest.jsonl"
    for path in [report_path, manifest_path, readiness_path, qa_path, aligned_path]:
        if not path.exists():
            raise FileNotFoundError(f"{shard_dir}: missing required shard file {path.name}")

    report = load_json(report_path)
    readiness = load_json(readiness_path)
    if readiness.get("status") != "pass":
        if not (
            allow_positive_other_iou_mean_fail
            and allowed_relaxed_readiness_failures(list(readiness.get("failures") or []))
        ):
            raise ValueError(f"{shard_dir}: readiness status is {readiness.get('status')!r}")
    if report.get("readiness_status") not in (None, "pass"):
        if not allow_positive_other_iou_mean_fail:
            raise ValueError(f"{shard_dir}: report readiness_status is {report.get('readiness_status')!r}")
    if report.get("status") != "pass":
        failures = list(report.get("export_failures") or [])
        only_min_positive = failures and all("positive_sample_count" in str(item) for item in failures)
        only_positive_iou_mean = failures and all(is_positive_other_iou_mean_failure(item) for item in failures)
        if not (
            (allow_report_fail_for_min_positive and only_min_positive)
            or (allow_positive_other_iou_mean_fail and only_positive_iou_mean)
        ):
            raise ValueError(f"{shard_dir}: export report status is fail: {failures}")
    return {
        "report": report,
        "manifest": load_json(manifest_path),
        "readiness": readiness,
        "teacher_qa": load_json(qa_path),
        "aligned_rows": list(iter_jsonl(aligned_path)),
    }


def run_readiness_checker(manifest_path: Path, teacher_qa_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    readiness_path = args.out_dir / "training_readiness_v0.json"
    readiness_path.unlink(missing_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "check_map_memory_training_readiness_v0.py"),
        "--manifest",
        str(manifest_path),
        "--teacher-qa",
        str(teacher_qa_path),
        "--mode",
        "mixed",
        "--max-samples",
        str(max(1, args.readiness_max_samples)),
        "--min-samples",
        str(args.min_samples),
        "--min-episodes",
        str(args.min_episodes),
        "--require-geometry-backend-id",
        REQUIRED_BACKEND_ID,
        "--out-json",
        str(readiness_path),
    ]
    if args.allow_positive_other_iou_mean_fail:
        cmd.append("--allow-positive-other-iou-mean-below-threshold")
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if readiness_path.exists():
        readiness = load_json(readiness_path)
    else:
        readiness = {
            "kind": "map_memory_training_readiness_v0",
            "status": "fail",
            "manifest": str(manifest_path),
            "teacher_qa": str(teacher_qa_path),
            "failures": [proc.stdout[-4000:]],
        }
    readiness["checker_returncode"] = int(proc.returncode)
    if proc.returncode != 0:
        readiness["status"] = "fail"
        readiness["checker_output_tail"] = proc.stdout[-4000:]
        failures = list(readiness.get("failures", []))
        if proc.stdout:
            failures.append(proc.stdout[-4000:])
        readiness["failures"] = failures
        write_json(readiness_path, readiness)
    return readiness


def combine_shards(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_dirs = [path.resolve() for path in args.shard_dir]
    if not shard_dirs:
        raise ValueError("at least one --shard-dir is required")

    samples_by_id: dict[str, dict[str, Any]] = {}
    qa_by_id: dict[str, dict[str, Any]] = {}
    rows_by_clip_id: dict[str, dict[str, Any]] = {}
    shard_summaries: list[dict[str, Any]] = []
    reject_counts: Counter[str] = Counter()
    source_cache_manifests: set[str] = set()
    source_manifests: set[str] = set()
    raw_roots: set[str] = set()

    for shard_dir in shard_dirs:
        payload = validate_shard(
            shard_dir,
            allow_report_fail_for_min_positive=args.allow_shard_min_positive_fail,
            allow_positive_other_iou_mean_fail=args.allow_positive_other_iou_mean_fail,
        )
        manifest = payload["manifest"]
        teacher_qa = payload["teacher_qa"]
        report = payload["report"]
        reject_counts.update(report.get("rejected_clips") or {})
        if manifest.get("channels") != CHANNELS:
            raise ValueError(f"{shard_dir}: channel contract mismatch")
        if manifest.get("shape") != [7, 176, 320]:
            raise ValueError(f"{shard_dir}: shape {manifest.get('shape')} != [7,176,320]")
        if manifest.get("geometry_backend_id") != REQUIRED_BACKEND_ID:
            raise ValueError(f"{shard_dir}: backend {manifest.get('geometry_backend_id')} != {REQUIRED_BACKEND_ID}")

        source_cache_manifests.add(str(manifest.get("source_cache_manifest")))
        if manifest.get("source_manifest"):
            source_manifests.add(str(manifest.get("source_manifest")))
        raw_roots.add(str(manifest.get("raw_root")))

        shard_samples = {str(sample["sample_id"]): sample for sample in manifest.get("samples", [])}
        shard_qa = {str(row["sample_id"]): row for row in teacher_qa.get("rows", [])}
        for sid, sample in shard_samples.items():
            if sid not in shard_qa:
                raise ValueError(f"{shard_dir}: sample {sid} has no teacher QA row")
            normalized = normalize_sample(sample, shard_dir=shard_dir, out_dir=args.out_dir, link_mode=args.link_mode)
            if sid in samples_by_id:
                if sample_fingerprint(samples_by_id[sid]) != sample_fingerprint(normalized):
                    raise ValueError(f"{shard_dir}: duplicate sample {sid} has conflicting metadata")
            else:
                samples_by_id[sid] = normalized
            if sid in qa_by_id:
                if qa_fingerprint(qa_by_id[sid]) != qa_fingerprint(shard_qa[sid]):
                    raise ValueError(f"{shard_dir}: duplicate QA row {sid} has conflicting metrics")
            else:
                qa_by_id[sid] = dict(shard_qa[sid])

        for row in payload["aligned_rows"]:
            clip_id = str(row.get("clip_id", ""))
            if not clip_id:
                raise ValueError(f"{shard_dir}: aligned row missing clip_id")
            sample_ids = row.get("map_memory_sample_ids")
            if not isinstance(sample_ids, list) or len(sample_ids) != args.latent_frames:
                raise ValueError(f"{shard_dir}: {clip_id}: invalid map_memory_sample_ids")
            missing = [sid for sid in sample_ids if sid not in shard_samples and sid not in samples_by_id]
            if missing:
                raise ValueError(f"{shard_dir}: {clip_id}: aligned row references missing samples {missing[:5]}")
            if clip_id in rows_by_clip_id:
                if rows_by_clip_id[clip_id].get("map_memory_sample_ids") != sample_ids:
                    raise ValueError(f"{shard_dir}: duplicate clip {clip_id} has conflicting sample ids")
                continue
            rows_by_clip_id[clip_id] = dict(row)

        shard_summaries.append(
            {
                "shard_dir": str(shard_dir),
                "report_status": report.get("status"),
                "readiness_status": report.get("readiness_status"),
                "accepted_clips": report.get("accepted_clips"),
                "accepted_unique_samples": report.get("accepted_unique_samples"),
                "role_counts": report.get("role_counts"),
                "export_failures": report.get("export_failures"),
            }
        )

    samples = sorted(samples_by_id.values(), key=lambda row: row["sample_id"])
    qa_rows = [qa_by_id[sample["sample_id"]] for sample in samples]
    aligned_rows = sorted(rows_by_clip_id.values(), key=lambda row: str(row.get("clip_id", "")))
    role_counts = Counter(str(sample.get("selection_role")) for sample in samples)
    match_counts = Counter(str(sample.get("match_id")) for sample in samples)
    episode_counts = Counter(str(sample.get("episode")) for sample in samples)
    split_counts = Counter(str(row.get("map_memory_split")) for row in aligned_rows)
    row_role_counts = Counter()
    for row in aligned_rows:
        row_role_counts.update(row.get("map_memory_selection_roles") or [])

    backend_signatures = sorted({str(sample["backend_signature"]) for sample in samples if sample.get("backend_signature")})
    first = samples[0] if samples else {}
    manifest = {
        "kind": "memory_dense_lingbot_cache_release_v0"
        if not args.allow_positive_other_iou_mean_fail
        else "memory_dense_lingbot_cache_train_only_relaxed_positive_mean_v0",
        "combined_from_shards": [str(path) for path in shard_dirs],
        "source_cache_manifest": next(iter(source_cache_manifests)) if len(source_cache_manifests) == 1 else sorted(source_cache_manifests),
        "source_manifest": next(iter(source_manifests)) if len(source_manifests) == 1 else sorted(source_manifests),
        "raw_root": next(iter(raw_roots)) if len(raw_roots) == 1 else sorted(raw_roots),
        "sample_count": len(samples),
        "positive_sample_count": int(role_counts.get("positive", 0)),
        "context_sample_count": int(role_counts.get("context", 0)),
        "match_count": len(match_counts),
        "episode_count": len(episode_counts),
        "match_sample_counts": dict(sorted(match_counts.items())),
        "episode_sample_counts": dict(sorted(episode_counts.items())),
        "shape": [7, 176, 320],
        "channels": CHANNELS,
        "policy": "Inputs are generated from Map Memory using LingBot latent-frame timestamps; RGB/depth/seg/visibility streams are target/QA only.",
        "selection_policy": "Combined release contains only complete LingBot clips accepted by strict shard exports."
        if not args.allow_positive_other_iou_mean_fail
        else (
            "Train-only relaxed-positive-mean combine: complete LingBot clips with exact dense alignment; "
            "positive other-player IoU per-row min, hit/depth, teacher-only, and provenance gates remain enforced, "
            "but shard/release positive-other-player IoU mean below 0.45 is allowed."
        ),
        "promotion_policy": "release_quality_strict"
        if not args.allow_positive_other_iou_mean_fail
        else "train_only_relaxed_positive_mean",
        "alignment": {
            "video_frames": args.video_frames,
            "raw_stride": args.raw_stride,
            "latent_frames": args.latent_frames,
            "frame_formula": "raw_start + raw_stride * latent_source_position",
            "nearest_or_repeat_used": False,
        },
        "mesh_backend": first.get("mesh_backend", "bsp_faces_gpu"),
        "geometry_backend": "bsp_faces",
        "geometry_backend_id": REQUIRED_BACKEND_ID,
        "geometry_backend_family": first.get("geometry_backend_family", "bsp_faces"),
        "backend_features": first.get("backend_features", ["visual_faces", "displacement"]),
        "backend_signatures": backend_signatures,
        "samples": samples,
    }
    manifest_path = args.out_dir / "manifest.json"
    write_json(manifest_path, manifest)

    teacher_qa = {
        "kind": "memory_dense_channels_vs_teacher_v0",
        "manifest": str(manifest_path),
        "sample_count": len(qa_rows),
        "missing_count": 0,
        "summary": summarize_qa(qa_rows),
        "rows": qa_rows,
        "missing": [],
        "policy": "Teacher streams are used only for QA. Memory dense tensors remain Map-rendered inputs.",
    }
    teacher_qa_path = args.out_dir / "channel_teacher_qa_v0" / "memory_dense_channels_vs_teacher_v0.json"
    write_json(teacher_qa_path, teacher_qa)

    readiness = run_readiness_checker(manifest_path, teacher_qa_path, args)
    map_manifest_sha256 = sha256_file(manifest_path)
    for row in aligned_rows:
        row["map_memory_manifest"] = str(manifest_path)
        row["map_memory_manifest_sha256"] = map_manifest_sha256
    aligned_path = args.out_dir / "aligned_cache_manifest.jsonl"
    write_jsonl(aligned_path, aligned_rows)

    failures: list[str] = []
    if readiness.get("status") != "pass":
        failures.append(f"readiness status is {readiness.get('status')!r}")
    if len(aligned_rows) < args.min_accepted_clips:
        failures.append(f"accepted_clips {len(aligned_rows)} < min_accepted_clips {args.min_accepted_clips}")
    if len(samples) < args.min_samples:
        failures.append(f"sample_count {len(samples)} < min_samples {args.min_samples}")
    if len(episode_counts) < args.min_episodes:
        failures.append(f"episode_count {len(episode_counts)} < min_episodes {args.min_episodes}")
    if int(role_counts.get("positive", 0)) < args.min_positive_samples:
        failures.append(f"positive_sample_count {int(role_counts.get('positive', 0))} < min_positive_samples {args.min_positive_samples}")
    if args.require_train_split and split_counts.get("train", 0) <= 0:
        failures.append(f"aligned rows have no train split; split_counts={dict(sorted(split_counts.items()))}")

    report = {
        "kind": "memory_dense_lingbot_cache_combined_report_v0",
        "status": "pass" if not failures else "fail",
        "out_dir": str(args.out_dir),
        "manifest": str(manifest_path),
        "teacher_qa": str(teacher_qa_path),
        "readiness": str(args.out_dir / "training_readiness_v0.json"),
        "aligned_cache_manifest": str(aligned_path),
        "map_manifest_sha256": map_manifest_sha256,
        "shard_count": len(shard_dirs),
        "shards": shard_summaries,
        "accepted_clips": len(aligned_rows),
        "accepted_unique_samples": len(samples),
        "role_counts": dict(sorted(role_counts.items())),
        "latent_frame_role_counts": dict(sorted(row_role_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "match_counts": dict(sorted(match_counts.items())),
        "episode_count": len(episode_counts),
        "combined_rejected_clips_from_shards": dict(sorted(reject_counts.items())),
        "readiness_status": readiness.get("status"),
        "readiness_failures": readiness.get("failures", []),
        "failures": failures,
        "promotion_policy": manifest["promotion_policy"],
        "relaxed_gates": {
            "positive_other_player_iou_mean_below_0_45": bool(args.allow_positive_other_iou_mean_fail),
            "positive_other_player_iou_min_0_25_still_enforced": True,
            "hit_iou_and_depth_still_enforced": True,
            "exact_latent_frame_alignment_still_enforced": True,
            "teacher_streams_are_model_inputs": False,
        },
    }
    report_path = args.out_dir / "combined_report_v0.json"
    write_json(report_path, report)
    if report["status"] != "pass":
        raise SystemExit(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard-dir", type=Path, action="append", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--link-mode", choices=["hardlink", "copy"], default="hardlink")
    ap.add_argument("--allow-shard-min-positive-fail", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--allow-positive-other-iou-mean-fail", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--video-frames", type=int, default=81)
    ap.add_argument("--raw-stride", type=int, default=2)
    ap.add_argument("--latent-frames", type=int, default=21)
    ap.add_argument("--min-accepted-clips", type=int, default=1)
    ap.add_argument("--min-positive-samples", type=int, default=1)
    ap.add_argument("--min-samples", type=int, default=21)
    ap.add_argument("--min-episodes", type=int, default=1)
    ap.add_argument("--readiness-max-samples", type=int, default=4096)
    ap.add_argument("--require-train-split", action=argparse.BooleanOptionalAction, default=True)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    report = combine_shards(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
