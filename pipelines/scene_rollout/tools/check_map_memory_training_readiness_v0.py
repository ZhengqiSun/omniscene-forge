#!/usr/bin/env python3
"""Check whether a Map Memory dense release is ready for adapter training.

The checker is intentionally stricter than the lightweight dataloader smoke. It
validates the model-input contract, teacher-only policy, teacher QA gates, and
optional BSP-vs-OBJ parity before a manifest is promoted for training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


CHANNELS = [
    "env_depth_norm_from_world_obj_projection",
    "env_mesh_hit_mask",
    "env_nav_place_semantic_from_static_memory",
    "other_player_mask_from_memory_player_capsules",
    "other_player_depth_norm_from_memory_capsules",
    "other_player_yaw_sin_relative_to_ego_from_memory",
    "other_player_yaw_cos_relative_to_ego_from_memory",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sample_path(manifest_path: Path, sample: dict[str, Any], key: str, rel_key: str) -> Path:
    if sample.get(rel_key):
        local = manifest_path.parent / sample[rel_key]
        if local.exists():
            return local
    if sample.get("source_manifest") and sample.get(rel_key):
        source = Path(sample["source_manifest"]).parent / sample[rel_key]
        if source.exists():
            return source
    raw = Path(sample[key])
    if raw.exists():
        return raw
    fallback = manifest_path.parent / "samples" / sample["sample_id"] / raw.name
    return fallback


def check(condition: bool, failures: list[str], message: str) -> None:
    if not condition:
        failures.append(message)


def mean_row(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def check_backend_provenance(
    manifest: dict[str, Any],
    min_episodes: int,
    min_samples: int,
    require_geometry_backend_id: str | None,
    failures: list[str],
) -> dict[str, Any]:
    samples = manifest.get("samples", [])
    episodes = sorted({str(s.get("episode")) for s in samples if s.get("episode")})
    backend_ids = sorted({
        str(v)
        for v in [manifest.get("geometry_backend_id"), *(s.get("geometry_backend_id") for s in samples)]
        if v
    })
    backend_signatures = sorted({
        str(v)
        for v in [*(manifest.get("backend_signatures") or []), *(s.get("backend_signature") for s in samples)]
        if v
    })
    backend_features = sorted({
        str(feature)
        for s in samples
        for feature in (s.get("backend_features") or [])
    } | {str(feature) for feature in (manifest.get("backend_features") or [])})

    check(len(samples) >= min_samples, failures, f"manifest has {len(samples)} samples, below min_samples={min_samples}")
    check(len(episodes) >= min_episodes, failures, f"manifest has {len(episodes)} episodes, below min_episodes={min_episodes}")
    if require_geometry_backend_id:
        bad = [
            s.get("sample_id", "<missing>")
            for s in samples
            if s.get("geometry_backend_id") != require_geometry_backend_id
        ]
        check(not bad, failures, f"samples missing required geometry_backend_id={require_geometry_backend_id}: {bad[:8]}")
        check(manifest.get("geometry_backend_id") == require_geometry_backend_id, failures, "manifest geometry_backend_id mismatch")
        check(bool(backend_signatures), failures, "required backend has no backend_signatures provenance")
    compatibility = manifest.get("compatibility")
    if isinstance(compatibility, dict):
        check(not compatibility.get("mismatches"), failures, f"manifest compatibility mismatches: {compatibility.get('mismatches')}")
    return {
        "episode_count": len(episodes),
        "episodes": episodes,
        "geometry_backend_ids": backend_ids,
        "backend_signatures": backend_signatures,
        "backend_features": backend_features,
    }


def check_manifest_contract(
    manifest_path: Path,
    manifest: dict[str, Any],
    max_samples: int,
    failures: list[str],
) -> dict[str, Any]:
    samples = manifest.get("samples", [])
    check(manifest.get("sample_count") == len(samples), failures, "manifest sample_count does not match samples length")
    check(len(samples) > 0, failures, "manifest has no samples")

    checked = []
    for sample in samples[:max_samples]:
        sid = sample.get("sample_id", "<missing>")
        check(tuple(sample.get("shape", [])) == (7, 176, 320), failures, f"{sid}: manifest shape is not [7,176,320]")
        check(sample.get("channels") == CHANNELS, failures, f"{sid}: channel names/order mismatch")
        dense_path = sample_path(manifest_path, sample, "dense_path", "dense_relpath")
        if not dense_path.exists():
            failures.append(f"{sid}: dense file missing: {dense_path}")
            continue
        for key, rel_key in [
            ("target_rgb_path", "target_rgb_relpath"),
            ("meta_path", "meta_relpath"),
            ("qa_path", "qa_relpath"),
        ]:
            sidecar = sample_path(manifest_path, sample, key, rel_key)
            check(sidecar.exists(), failures, f"{sid}: {key} missing: {sidecar}")
        dense = np.load(dense_path)["dense"]
        check(tuple(dense.shape) == (7, 176, 320), failures, f"{sid}: dense array shape is {list(dense.shape)}")
        check(dense.dtype == np.float32, failures, f"{sid}: dense dtype is {dense.dtype}, expected float32")
        check(bool(np.isfinite(dense).all()), failures, f"{sid}: dense contains non-finite values")
        check(float(dense.min()) >= -1.0001, failures, f"{sid}: dense min below -1")
        check(float(dense.max()) <= 1.0001, failures, f"{sid}: dense max above 1")
        for idx in [0, 1, 2, 3, 4]:
            ch = dense[idx]
            check(float(ch.min()) >= -0.0001, failures, f"{sid}: channel {idx} min below 0")
            check(float(ch.max()) <= 1.0001, failures, f"{sid}: channel {idx} max above 1")
        checked.append({
            "sample_id": sid,
            "dense_path": str(dense_path),
            "dense_min": float(dense.min()),
            "dense_max": float(dense.max()),
            "mesh_backend": sample.get("mesh_backend"),
        })
    return {"checked_sample_count": len(checked), "checked_samples": checked}


def check_teacher_policy(manifest: dict[str, Any], failures: list[str]) -> None:
    policy_text = " ".join(
        str(manifest.get(key, ""))
        for key in ["policy", "input_policy", "kind"]
    ).lower()
    check("teacher" not in policy_text or "qa" in policy_text or "target" in policy_text, failures, "manifest policy mentions teacher without QA/target context")
    for sample in manifest.get("samples", []):
        sid = sample.get("sample_id", "<missing>")
        dense_path = str(sample.get("dense_path", "")) + " " + str(sample.get("dense_relpath", ""))
        check("_depth.mkv" not in dense_path and "_seg.mkv" not in dense_path, failures, f"{sid}: dense path appears to reference teacher stream")


def check_teacher_qa(
    qa_path: Path,
    mode: str,
    expected_sample_count: int | None,
    failures: list[str],
    *,
    allow_positive_other_iou_mean_below_threshold: bool,
) -> dict[str, Any]:
    qa = load_json(qa_path)
    rows = qa.get("rows", [])
    summary = qa.get("summary", {})
    check(qa.get("missing_count") == 0, failures, f"{qa_path}: missing_count={qa.get('missing_count')}")
    if expected_sample_count is not None:
        check(len(rows) == expected_sample_count, failures, f"{qa_path}: QA row count {len(rows)} != manifest sample_count {expected_sample_count}")
        check(qa.get("sample_count") == expected_sample_count, failures, f"{qa_path}: QA sample_count {qa.get('sample_count')} != manifest sample_count {expected_sample_count}")
    hit_iou_mean = summary.get("hit_iou_mean")
    depth_common_mae_mean = summary.get("depth_common_mae_mean")
    check(
        hit_iou_mean is not None and float(hit_iou_mean) >= 0.90,
        failures,
        f"{qa_path}: hit_iou_mean below 0.90",
    )
    check(
        depth_common_mae_mean is not None and float(depth_common_mae_mean) <= 0.08,
        failures,
        f"{qa_path}: depth_common_mae_mean above 0.08",
    )

    positive_rows = [r for r in rows if int(r.get("visible_teacher_players", 0)) > 0 or int(r.get("other_player_teacher_pixels", 0)) > 0]
    empty_rows = [r for r in rows if int(r.get("visible_teacher_players", 0)) == 0 and int(r.get("other_player_teacher_pixels", 0)) == 0]

    if mode in {"positive", "mixed"}:
        if mode == "positive":
            check(len(positive_rows) == len(rows), failures, f"{qa_path}: positive mode contains teacher-empty rows")
        if positive_rows:
            pos_other_mean = mean_row(positive_rows, "other_player_iou")
            pos_other_min = min(float(r.get("other_player_iou", 0.0)) for r in positive_rows)
            if not allow_positive_other_iou_mean_below_threshold:
                check(pos_other_mean is not None and pos_other_mean >= 0.45, failures, f"{qa_path}: positive other_player_iou_mean below 0.45")
            check(pos_other_min >= 0.25, failures, f"{qa_path}: positive other_player_iou_min below 0.25")
            for r in positive_rows:
                sid = r.get("sample_id", "<missing>")
                check(int(r.get("visible_teacher_players", 0)) > 0, failures, f"{sid}: positive row has no visible teacher players")
                check(int(r.get("other_player_memory_pixels", 0)) > 0, failures, f"{sid}: positive row has empty Memory other-player mask")
    for r in empty_rows:
        sid = r.get("sample_id", "<missing>")
        check(int(r.get("other_player_memory_pixels", 0)) == 0, failures, f"{sid}: teacher-empty row has Memory other-player pixels")

    return {
        "qa_path": str(qa_path),
        "mode": mode,
        "sample_count": len(rows),
        "positive_row_count": len(positive_rows),
        "teacher_empty_row_count": len(empty_rows),
        "summary": summary,
        "relaxed_positive_other_iou_mean_below_threshold": bool(allow_positive_other_iou_mean_below_threshold),
    }


def check_bsp_parity(path: Path, failures: list[str]) -> dict[str, Any]:
    data = load_json(path)
    summary = data.get("summary", {})
    obj_hit = summary.get("obj_teacher_hit_iou_mean")
    bsp_hit = summary.get("bsp_teacher_hit_iou_mean")
    obj_depth = summary.get("obj_teacher_depth_mae_common_mean")
    bsp_depth = summary.get("bsp_teacher_depth_mae_common_mean")
    if obj_hit is not None and bsp_hit is not None:
        check(bsp_hit >= obj_hit - 0.03, failures, f"{path}: BSP teacher hit IoU regresses more than 0.03 vs OBJ")
    if obj_depth is not None and bsp_depth is not None:
        check(bsp_depth <= obj_depth + 0.02, failures, f"{path}: BSP teacher depth MAE regresses more than 0.02 vs OBJ")
    return {"bsp_parity_path": str(path), "summary": summary}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--teacher-qa", type=Path, required=True)
    ap.add_argument("--mode", choices=["positive", "mixed", "context"], default="mixed")
    ap.add_argument("--bsp-parity", type=Path, default=None)
    ap.add_argument("--max-samples", type=int, default=64)
    ap.add_argument("--min-samples", type=int, default=1)
    ap.add_argument("--min-episodes", type=int, default=1)
    ap.add_argument("--require-geometry-backend-id", default=None)
    ap.add_argument("--allow-positive-other-iou-mean-below-threshold", action="store_true")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    failures: list[str] = []
    manifest = load_json(args.manifest)
    report = {
        "kind": "map_memory_training_readiness_v0",
        "manifest": str(args.manifest),
        "teacher_qa": str(args.teacher_qa),
        "mode": args.mode,
        "contract": {
            "shape": [7, 176, 320],
            "channels": CHANNELS,
            "teacher_policy": "Teacher streams are GT/QA only and must not be model input channels.",
        },
        "manifest_check": check_manifest_contract(args.manifest, manifest, args.max_samples, failures),
    }
    report["backend_provenance_check"] = check_backend_provenance(
        manifest,
        args.min_episodes,
        args.min_samples,
        args.require_geometry_backend_id,
        failures,
    )
    check_teacher_policy(manifest, failures)
    report["teacher_qa_check"] = check_teacher_qa(
        args.teacher_qa,
        args.mode,
        manifest.get("sample_count"),
        failures,
        allow_positive_other_iou_mean_below_threshold=args.allow_positive_other_iou_mean_below_threshold,
    )
    if args.bsp_parity is not None:
        report["bsp_parity_check"] = check_bsp_parity(args.bsp_parity, failures)

    report["status"] = "pass" if not failures else "fail"
    report["failures"] = failures
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")
    print(text)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
