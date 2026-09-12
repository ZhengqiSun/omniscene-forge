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

# v2 dense contract (416x240 -> 30x52 native, no interpolation). Kept local on
# purpose: this checker stays importable without the training stack, so the grid is
# a CLI knob (--dense-hw) instead of an import from map_memory_training_data_v0.
DEFAULT_DENSE_HW = (240, 416)


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


def optional_sample_path(manifest_path: Path, sample: dict[str, Any], key: str, rel_key: str) -> Path | None:
    if not sample.get(key) and not sample.get(rel_key):
        return None
    return sample_path(manifest_path, sample, key, rel_key)


def load_sample_meta(manifest_path: Path, sample: dict[str, Any]) -> dict[str, Any]:
    path = optional_sample_path(manifest_path, sample, "meta_path", "meta_relpath")
    if path is None or not path.exists():
        return {}
    return load_json(path)


def check(condition: bool, failures: list[str], message: str) -> None:
    if not condition:
        failures.append(message)


def check_player(condition: bool, failures: list[str], warnings: list[str], message: str, policy: str) -> None:
    if condition or policy == "skip":
        return
    if policy == "warn":
        warnings.append(message)
    else:
        failures.append(message)


def mean_row(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def check_manifest_contract(
    manifest_path: Path,
    manifest: dict[str, Any],
    max_samples: int,
    expected_geometry_backend: str | None,
    allow_mixed_backend: bool,
    failures: list[str],
    dense_hw: tuple[int, int] = DEFAULT_DENSE_HW,
) -> dict[str, Any]:
    expected_shape = (len(CHANNELS), int(dense_hw[0]), int(dense_hw[1]))
    samples = manifest.get("samples", [])
    check(manifest.get("sample_count") == len(samples), failures, "manifest sample_count does not match samples length")
    check(len(samples) > 0, failures, "manifest has no samples")

    checked = []
    backend_ids: list[str] = []
    backend_signatures: list[str] = []
    for sample in samples[:max_samples]:
        sid = sample.get("sample_id", "<missing>")
        check(
            tuple(sample.get("shape", [])) == expected_shape,
            failures,
            f"{sid}: manifest shape {sample.get('shape')} is not {list(expected_shape)}",
        )
        check(sample.get("channels") == CHANNELS, failures, f"{sid}: channel names/order mismatch")
        meta = load_sample_meta(manifest_path, sample)
        geometry_backend_id = sample.get("geometry_backend_id") or meta.get("geometry_backend_id")
        backend_signature = sample.get("backend_signature") or meta.get("backend_signature")
        backend_features = sample.get("backend_features") or meta.get("backend_features") or []
        component_counts = sample.get("component_counts") or meta.get("component_counts") or {}
        if geometry_backend_id:
            backend_ids.append(str(geometry_backend_id))
        if backend_signature:
            backend_signatures.append(str(backend_signature))
        if expected_geometry_backend is not None:
            check(
                geometry_backend_id == expected_geometry_backend,
                failures,
                f"{sid}: geometry_backend_id={geometry_backend_id!r}, expected {expected_geometry_backend!r}",
            )
        if geometry_backend_id and str(geometry_backend_id).startswith("bsp_faces_disp"):
            check("displacement" in backend_features, failures, f"{sid}: BSP displacement backend lacks displacement feature provenance")
            check(int(component_counts.get("displacement_faces", 0)) > 0, failures, f"{sid}: BSP displacement backend has no displacement_faces count")
        dense_path = sample_path(manifest_path, sample, "dense_path", "dense_relpath")
        if not dense_path.exists():
            failures.append(f"{sid}: dense file missing: {dense_path}")
            continue
        dense = np.load(dense_path)["dense"]
        check(
            tuple(dense.shape) == expected_shape,
            failures,
            f"{sid}: dense array shape is {list(dense.shape)}, expected {list(expected_shape)}",
        )
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
            "geometry_backend_id": geometry_backend_id,
            "backend_signature": backend_signature,
        })
    unique_backend_ids = sorted(set(backend_ids))
    if expected_geometry_backend is not None:
        check(bool(backend_ids), failures, "expected geometry backend was requested but no sample records backend provenance")
    if not allow_mixed_backend and len(unique_backend_ids) > 1:
        failures.append(f"manifest mixes geometry backends: {unique_backend_ids}")
    return {
        "checked_sample_count": len(checked),
        "checked_samples": checked,
        "geometry_backend_ids": unique_backend_ids,
        "backend_signatures": sorted(set(backend_signatures)),
        "allow_mixed_backend": allow_mixed_backend,
        "expected_geometry_backend": expected_geometry_backend,
    }


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
    failures: list[str],
    warnings: list[str],
    min_hit_iou: float,
    max_depth_common_mae: float,
    min_positive_other_iou_mean: float,
    min_positive_other_iou_min: float,
    player_qa_policy: str,
) -> dict[str, Any]:
    qa = load_json(qa_path)
    rows = qa.get("rows", [])
    summary = qa.get("summary", {})
    check(qa.get("missing_count") == 0, failures, f"{qa_path}: missing_count={qa.get('missing_count')}")
    check(summary.get("hit_iou_mean", 0.0) >= min_hit_iou, failures, f"{qa_path}: hit_iou_mean below {min_hit_iou}")
    check(summary.get("depth_common_mae_mean", 999.0) <= max_depth_common_mae, failures, f"{qa_path}: depth_common_mae_mean above {max_depth_common_mae}")

    positive_rows = [r for r in rows if int(r.get("visible_teacher_players", 0)) > 0 or int(r.get("other_player_teacher_pixels", 0)) > 0]
    empty_rows = [r for r in rows if int(r.get("visible_teacher_players", 0)) == 0 and int(r.get("other_player_teacher_pixels", 0)) == 0]

    if mode in {"positive", "mixed"}:
        if mode == "positive":
            check(len(positive_rows) == len(rows), failures, f"{qa_path}: positive mode contains teacher-empty rows")
        if positive_rows:
            pos_other_mean = mean_row(positive_rows, "other_player_iou")
            pos_other_min = min(float(r.get("other_player_iou", 0.0)) for r in positive_rows)
            check_player(
                pos_other_mean is not None and pos_other_mean >= min_positive_other_iou_mean,
                failures,
                warnings,
                f"{qa_path}: positive other_player_iou_mean below {min_positive_other_iou_mean}",
                player_qa_policy,
            )
            check_player(
                pos_other_min >= min_positive_other_iou_min,
                failures,
                warnings,
                f"{qa_path}: positive other_player_iou_min below {min_positive_other_iou_min}",
                player_qa_policy,
            )
            for r in positive_rows:
                sid = r.get("sample_id", "<missing>")
                check_player(
                    int(r.get("visible_teacher_players", 0)) > 0,
                    failures,
                    warnings,
                    f"{sid}: positive row has no visible teacher players",
                    player_qa_policy,
                )
                check_player(
                    int(r.get("other_player_memory_pixels", 0)) > 0,
                    failures,
                    warnings,
                    f"{sid}: positive row has empty Memory other-player mask",
                    player_qa_policy,
                )
    for r in empty_rows:
        sid = r.get("sample_id", "<missing>")
        check_player(
            int(r.get("other_player_memory_pixels", 0)) == 0,
            failures,
            warnings,
            f"{sid}: teacher-empty row has Memory other-player pixels",
            player_qa_policy,
        )

    return {
        "qa_path": str(qa_path),
        "mode": mode,
        "player_qa_policy": player_qa_policy,
        "sample_count": len(rows),
        "positive_row_count": len(positive_rows),
        "teacher_empty_row_count": len(empty_rows),
        "summary": summary,
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
    ap.add_argument("--expected-geometry-backend", default=None)
    ap.add_argument("--allow-mixed-backend", action="store_true")
    ap.add_argument(
        "--dense-hw",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=list(DEFAULT_DENSE_HW),
        help=(
            "Dense condition map grid H W. Default is the v2 dense contract 240 416 "
            "(-> 30x52 native VAE grid == DiT token grid). Pass the legacy pre-v2 grid "
            "to audit an older release."
        ),
    )
    ap.add_argument("--max-samples", type=int, default=64)
    ap.add_argument("--min-hit-iou", type=float, default=0.90)
    ap.add_argument("--max-depth-common-mae", type=float, default=0.08)
    ap.add_argument("--min-positive-other-iou-mean", type=float, default=0.45)
    ap.add_argument("--min-positive-other-iou-min", type=float, default=0.25)
    ap.add_argument("--player-qa-policy", choices=["strict", "warn", "skip"], default="strict")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    failures: list[str] = []
    warnings: list[str] = []
    manifest = load_json(args.manifest)
    report = {
        "kind": "map_memory_training_readiness_v0",
        "manifest": str(args.manifest),
        "teacher_qa": str(args.teacher_qa),
        "mode": args.mode,
        "contract": {
            "shape": [len(CHANNELS), int(args.dense_hw[0]), int(args.dense_hw[1])],
            "channels": CHANNELS,
            "teacher_policy": "Teacher streams are GT/QA only and must not be model input channels.",
            "thresholds": {
                "min_hit_iou": args.min_hit_iou,
                "max_depth_common_mae": args.max_depth_common_mae,
                "min_positive_other_iou_mean": args.min_positive_other_iou_mean,
                "min_positive_other_iou_min": args.min_positive_other_iou_min,
                "player_qa_policy": args.player_qa_policy,
            },
        },
        "manifest_check": check_manifest_contract(
            args.manifest,
            manifest,
            args.max_samples,
            args.expected_geometry_backend,
            args.allow_mixed_backend,
            failures,
            dense_hw=(int(args.dense_hw[0]), int(args.dense_hw[1])),
        ),
    }
    check_teacher_policy(manifest, failures)
    report["teacher_qa_check"] = check_teacher_qa(
        args.teacher_qa,
        args.mode,
        failures,
        warnings,
        args.min_hit_iou,
        args.max_depth_common_mae,
        args.min_positive_other_iou_mean,
        args.min_positive_other_iou_min,
        args.player_qa_policy,
    )
    if args.bsp_parity is not None:
        report["bsp_parity_check"] = check_bsp_parity(args.bsp_parity, failures)

    report["status"] = "pass" if not failures else "fail"
    report["failures"] = failures
    report["warnings"] = warnings
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")
    print(text)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
