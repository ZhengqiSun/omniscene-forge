#!/usr/bin/env python3
"""Combine light-dust2 pilot dense export shards.

This combiner is intentionally scoped to the light-data surrogate contract:
there are no teacher seg/depth streams, so the final release preserves
``memory_dense_channel_3_surrogate_v0`` metadata and does not run teacher-QA
readiness checks.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from map_memory_training_data_v0 import CHANNELS, REQUIRED_BACKEND_ID, sha256_file


REGION_MASK_KIND = "memory_dense_channel_3_surrogate_v0"


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


def link_or_copy(src: Path, dst: Path, *, mode: str) -> None:
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
    out = dict(sample)
    for key, rel_key, filename in [
        ("dense_path", "dense_relpath", "mesh_dense_condition_v0.npz"),
        ("target_rgb_path", "target_rgb_relpath", "target_rgb.png"),
        ("meta_path", "meta_relpath", "mesh_dense_condition_meta_v0.json"),
        ("qa_path", "qa_relpath", "mesh_dense_condition_qa_v0.png"),
    ]:
        src = resolve_sample_path(shard_dir, sample, key, rel_key)
        dst = sample_dir / filename
        link_or_copy(src, dst, mode=link_mode)
        out[key] = str(dst)
        out[rel_key] = str(dst.relative_to(out_dir))
    out["channels"] = CHANNELS
    out["region_mask_kind"] = REGION_MASK_KIND
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
        "region_mask_kind": sample.get("region_mask_kind"),
    }


def qa_fingerprint(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row.get("sample_id"),
        "episode": row.get("episode"),
        "raw_episode": row.get("raw_episode"),
        "match_id": row.get("match_id"),
        "ego_stem": row.get("ego_stem"),
        "frame_index": row.get("frame_index"),
        "other_player_memory_pixels": row.get("other_player_memory_pixels"),
        "region_mask_kind": row.get("region_mask_kind"),
        "teacher_qa_available": row.get("teacher_qa_available"),
    }


def summarize_light(samples: list[dict[str, Any]], qa_rows: list[dict[str, Any]]) -> dict[str, Any]:
    memory_pixels = np.asarray(
        [int(row.get("other_player_memory_pixels", 0) or 0) for row in qa_rows],
        dtype=np.float64,
    )
    mesh_hit = np.asarray(
        [float(sample.get("mesh_hit_ratio", 0.0) or 0.0) for sample in samples],
        dtype=np.float64,
    )
    nav_hit = np.asarray(
        [float(sample.get("nav_semantic_hit_ratio", 0.0) or 0.0) for sample in samples],
        dtype=np.float64,
    )
    return {
        "teacher_qa_available": False,
        "region_mask_kind": REGION_MASK_KIND,
        "memory_player_pixels_mean": float(memory_pixels.mean()) if len(memory_pixels) else None,
        "memory_player_pixels_p50": float(np.percentile(memory_pixels, 50)) if len(memory_pixels) else None,
        "memory_player_pixels_min": int(memory_pixels.min()) if len(memory_pixels) else None,
        "memory_player_pixels_max": int(memory_pixels.max()) if len(memory_pixels) else None,
        "mesh_hit_ratio_mean": float(mesh_hit.mean()) if len(mesh_hit) else None,
        "mesh_hit_ratio_p50": float(np.percentile(mesh_hit, 50)) if len(mesh_hit) else None,
        "mesh_hit_ratio_min": float(mesh_hit.min()) if len(mesh_hit) else None,
        "nav_semantic_hit_ratio_mean": float(nav_hit.mean()) if len(nav_hit) else None,
    }


def validate_light_shard(shard_dir: Path) -> dict[str, Any]:
    paths = {
        "report": shard_dir / "export_report_v0.json",
        "manifest": shard_dir / "manifest.json",
        "readiness": shard_dir / "training_readiness_v0.json",
        "teacher_qa": shard_dir / "channel_teacher_qa_v0" / "memory_dense_channels_vs_teacher_v0.json",
        "aligned": shard_dir / "aligned_cache_manifest.jsonl",
    }
    for key, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{shard_dir}: missing {key}: {path}")
    report = load_json(paths["report"])
    manifest = load_json(paths["manifest"])
    readiness = load_json(paths["readiness"])
    teacher_qa = load_json(paths["teacher_qa"])
    if report.get("status") != "pass":
        raise ValueError(f"{shard_dir}: report status {report.get('status')!r}: {report.get('export_failures')}")
    if readiness.get("status") != "pass":
        raise ValueError(f"{shard_dir}: readiness status {readiness.get('status')!r}")
    if manifest.get("region_mask_kind") != REGION_MASK_KIND:
        raise ValueError(f"{shard_dir}: unexpected region_mask_kind {manifest.get('region_mask_kind')!r}")
    if teacher_qa.get("region_mask_kind") != REGION_MASK_KIND:
        raise ValueError(f"{shard_dir}: unexpected QA region_mask_kind {teacher_qa.get('region_mask_kind')!r}")
    if manifest.get("channels") != CHANNELS:
        raise ValueError(f"{shard_dir}: channel contract mismatch")
    if manifest.get("shape") != [7, 176, 320]:
        raise ValueError(f"{shard_dir}: shape {manifest.get('shape')} != [7,176,320]")
    if manifest.get("geometry_backend_id") != REQUIRED_BACKEND_ID:
        raise ValueError(f"{shard_dir}: backend {manifest.get('geometry_backend_id')} != {REQUIRED_BACKEND_ID}")
    return {
        "report": report,
        "manifest": manifest,
        "readiness": readiness,
        "teacher_qa": teacher_qa,
        "aligned_rows": list(iter_jsonl(paths["aligned"])),
    }


def combine(args: argparse.Namespace) -> dict[str, Any]:
    shard_dirs = [path.resolve() for path in args.shard_dir]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    samples_by_id: dict[str, dict[str, Any]] = {}
    qa_by_id: dict[str, dict[str, Any]] = {}
    rows_by_clip_id: dict[str, dict[str, Any]] = {}
    shard_summaries: list[dict[str, Any]] = []
    source_cache_manifests: set[str] = set()
    source_manifests: set[str] = set()
    raw_roots: set[str] = set()
    reject_counts: Counter[str] = Counter()
    player_capsule_params: dict[str, Any] | None = None
    camera_projection_params: dict[str, Any] | None = None

    for shard_dir in shard_dirs:
        payload = validate_light_shard(shard_dir)
        manifest = payload["manifest"]
        teacher_qa = payload["teacher_qa"]
        report = payload["report"]
        if isinstance(manifest.get("player_capsule_params"), dict):
            if player_capsule_params is None:
                player_capsule_params = dict(manifest["player_capsule_params"])
            elif player_capsule_params != manifest["player_capsule_params"]:
                raise ValueError(f"{shard_dir}: player_capsule_params mismatch")
        if isinstance(manifest.get("camera_projection_params"), dict):
            if camera_projection_params is None:
                camera_projection_params = dict(manifest["camera_projection_params"])
            elif camera_projection_params != manifest["camera_projection_params"]:
                raise ValueError(f"{shard_dir}: camera_projection_params mismatch")
        source_cache_manifests.add(str(manifest.get("source_cache_manifest")))
        source_manifests.add(str(manifest.get("source_manifest")))
        raw_roots.add(str(manifest.get("raw_root")))
        reject_counts.update(report.get("reject_counts") or {})

        shard_samples = {str(sample["sample_id"]): sample for sample in manifest.get("samples", [])}
        shard_qa = {str(row["sample_id"]): row for row in teacher_qa.get("rows", [])}
        for sid, sample in shard_samples.items():
            if sid not in shard_qa:
                raise ValueError(f"{shard_dir}: sample {sid} has no surrogate QA row")
            normalized = normalize_sample(
                sample,
                shard_dir=shard_dir,
                out_dir=args.out_dir,
                link_mode=args.link_mode,
            )
            qa_row = dict(shard_qa[sid])
            qa_row["region_mask_kind"] = REGION_MASK_KIND
            qa_row["teacher_qa_available"] = False
            if sid in samples_by_id and sample_fingerprint(samples_by_id[sid]) != sample_fingerprint(normalized):
                raise ValueError(f"{shard_dir}: duplicate sample {sid} has conflicting metadata")
            samples_by_id.setdefault(sid, normalized)
            if sid in qa_by_id and qa_fingerprint(qa_by_id[sid]) != qa_fingerprint(qa_row):
                raise ValueError(f"{shard_dir}: duplicate surrogate QA row {sid} has conflicting metrics")
            qa_by_id.setdefault(sid, qa_row)

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
            out_row = dict(row)
            out_row["region_mask_kind"] = REGION_MASK_KIND
            if clip_id in rows_by_clip_id:
                if rows_by_clip_id[clip_id].get("map_memory_sample_ids") != sample_ids:
                    raise ValueError(f"{shard_dir}: duplicate clip {clip_id} has conflicting sample ids")
                continue
            rows_by_clip_id[clip_id] = out_row

        shard_summaries.append({
            "shard_dir": str(shard_dir),
            "accepted_clips": report.get("accepted_clips"),
            "accepted_unique_samples": report.get("accepted_unique_samples"),
            "split_counts": report.get("split_counts"),
            "sample_role_counts": report.get("sample_role_counts"),
            "summary": report.get("summary"),
        })

    samples = sorted(samples_by_id.values(), key=lambda row: row["sample_id"])
    qa_rows = [qa_by_id[sample["sample_id"]] for sample in samples]
    aligned_rows = sorted(rows_by_clip_id.values(), key=lambda row: str(row.get("clip_id", "")))
    role_counts = Counter(str(sample.get("selection_role")) for sample in samples)
    match_counts = Counter(str(sample.get("match_id")) for sample in samples)
    episode_counts = Counter(str(sample.get("episode")) for sample in samples)
    split_counts = Counter(str(row.get("map_memory_split")) for row in aligned_rows)
    latent_role_counts: Counter[str] = Counter()
    for row in aligned_rows:
        latent_role_counts.update(row.get("map_memory_selection_roles") or [])

    if args.expected_clips is not None and len(aligned_rows) != args.expected_clips:
        raise ValueError(f"accepted clips {len(aligned_rows)} != expected {args.expected_clips}")
    if args.expected_samples is not None and len(samples) != args.expected_samples:
        raise ValueError(f"samples {len(samples)} != expected {args.expected_samples}")

    backend_signatures = sorted({str(sample["backend_signature"]) for sample in samples if sample.get("backend_signature")})
    first = samples[0] if samples else {}
    manifest = {
        "kind": "memory_dense_lingbot_light_dust2_pilot_release_v0",
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
        "policy": "Light dust2 pilot dense tensors are generated from Map Memory JSON/BSP only; no teacher seg/depth QA is available.",
        "selection_policy": "Rows come from player_visibility-positive source clips; per-frame role is positive iff Memory dense channel 3 has player pixels.",
        "region_mask_kind": REGION_MASK_KIND,
        "region_mask_policy": {
            "kind": REGION_MASK_KIND,
            "source": "dense_channel_3_other_player_mask_from_memory_player_capsules",
            "valid_for": "relative true-vs-shuffled Memory-mask region loss only",
            "not_valid_for": "teacher segmentation precision/recall, absolute presence, or position error",
        },
        "teacher_qa_available": False,
        "alignment": {
            "video_frames": args.video_frames,
            "raw_stride": args.raw_stride,
            "latent_frames": args.latent_frames,
            "frame_formula": "raw_start + raw_stride * latent_source_position",
            "nearest_or_repeat_used": False,
        },
        "player_capsule_params": player_capsule_params,
        "camera_projection_params": camera_projection_params,
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
    manifest_sha = sha256_file(manifest_path)
    for row in aligned_rows:
        row["map_memory_manifest"] = str(manifest_path)
        row["map_memory_manifest_sha256"] = manifest_sha

    teacher_qa = {
        "kind": "memory_dense_light_surrogate_channels_v0",
        "manifest": str(manifest_path),
        "sample_count": len(qa_rows),
        "missing_count": 0,
        "summary": summarize_light(samples, qa_rows),
        "rows": qa_rows,
        "missing": [],
        "region_mask_kind": REGION_MASK_KIND,
        "teacher_qa_available": False,
        "policy": {
            "region_mask_kind": REGION_MASK_KIND,
            "note": "This is not teacher segmentation QA. It exists so release loaders can carry Memory-mask region metadata.",
        },
    }
    teacher_qa_path = args.out_dir / "channel_teacher_qa_v0" / "memory_dense_channels_vs_teacher_v0.json"
    write_json(teacher_qa_path, teacher_qa)

    readiness = {
        "kind": "map_memory_training_readiness_v0",
        "status": "pass",
        "manifest": str(manifest_path),
        "teacher_qa": str(teacher_qa_path),
        "mode": "light_memory_mask_surrogate",
        "sample_count": len(samples),
        "failures": [],
        "region_mask_kind": REGION_MASK_KIND,
        "teacher_qa_available": False,
        "caveats": [
            "No teacher seg/depth streams are available for light dust2.",
            "Region metrics must use Memory dense channel 3 surrogate and remain relative true-vs-shuffled gates.",
        ],
    }
    readiness_path = args.out_dir / "training_readiness_v0.json"
    write_json(readiness_path, readiness)

    aligned_path = args.out_dir / "aligned_cache_manifest.jsonl"
    write_jsonl(aligned_path, aligned_rows)

    failures: list[str] = []
    if len(aligned_rows) < args.min_accepted_clips:
        failures.append(f"accepted_clips {len(aligned_rows)} < min_accepted_clips {args.min_accepted_clips}")
    if int(role_counts.get("positive", 0)) < args.min_positive_samples:
        failures.append(f"positive_sample_count {int(role_counts.get('positive', 0))} < min_positive_samples {args.min_positive_samples}")
    if split_counts.get("train", 0) <= 0 or split_counts.get("test", 0) <= 0:
        failures.append(f"aligned rows need train and test split; split_counts={dict(sorted(split_counts.items()))}")

    report = {
        "kind": "memory_dense_lingbot_light_dust2_pilot_combined_report_v0",
        "status": "pass" if not failures else "fail",
        "out_dir": str(args.out_dir),
        "manifest": str(manifest_path),
        "teacher_qa": str(teacher_qa_path),
        "readiness": str(readiness_path),
        "aligned_cache_manifest": str(aligned_path),
        "map_manifest_sha256": manifest_sha,
        "shard_count": len(shard_dirs),
        "shards": shard_summaries,
        "accepted_clips": len(aligned_rows),
        "accepted_unique_samples": len(samples),
        "sample_role_counts": dict(sorted(role_counts.items())),
        "latent_frame_role_counts": dict(sorted(latent_role_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "match_count": len(match_counts),
        "episode_count": len(episode_counts),
        "combined_reject_counts_from_shards": dict(sorted(reject_counts.items())),
        "summary": teacher_qa["summary"],
        "region_mask_kind": REGION_MASK_KIND,
        "teacher_qa_available": False,
        "failures": failures,
    }
    report_path = args.out_dir / "combined_report_v0.json"
    write_json(report_path, report)
    if failures:
        raise SystemExit(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard-dir", type=Path, action="append", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--link-mode", choices=["hardlink", "copy"], default="hardlink")
    ap.add_argument("--video-frames", type=int, default=81)
    ap.add_argument("--raw-stride", type=int, default=2)
    ap.add_argument("--latent-frames", type=int, default=21)
    ap.add_argument("--min-accepted-clips", type=int, default=1)
    ap.add_argument("--min-positive-samples", type=int, default=1)
    ap.add_argument("--expected-clips", type=int)
    ap.add_argument("--expected-samples", type=int)
    return ap


def main() -> None:
    report = combine(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
