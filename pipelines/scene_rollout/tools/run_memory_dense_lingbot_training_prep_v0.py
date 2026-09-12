#!/usr/bin/env python3
"""Orchestrate strict LingBot-aligned Map Memory dense training preparation.

This script prepares data for the formal dense-adapter trainer.  It never runs
optimizer steps.  The final action is a real ``preflight-train`` pass plus a
JSON summary that records the exact train command to run afterwards.
"""

from __future__ import annotations
from runtime_paths import source_path

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any


DEFAULT_CACHE_MANIFEST = Path(
    str(source_path('assets', 'datasets/fast_cam_dust2_p0_32f_done_cpfs/latents_full_32f/cache_manifest.jsonl'))
)
DEFAULT_SOURCE_MANIFEST = Path(
    str(source_path('assets', 'datasets/fast_cam_dust2_p0_32f_done_cpfs/manifests/32f_done_train_cache_input_cpfs.jsonl'))
)
DEFAULT_RAW_ROOT = Path(str(source_path('assets', 'csgo-datasets-fullsubset')))
DEFAULT_LINGBOT_REPO = Path(str(source_path('lingbot', '')))
DEFAULT_BASE_DIR = Path("output/memory_dense_adapter_v0")
DEFAULT_BSP_FACES_NPZ = Path("docs/assets/bsp_static_geometry_v0/de_dust2_bsp_faces_v0.npz")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def git_revision() -> str | None:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def state_path(run_dir: Path) -> Path:
    return run_dir / "training_prep_state_v0.json"


def load_state(run_dir: Path) -> dict[str, Any]:
    path = state_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"missing run state: {path}")
    return load_json(path)


def save_state(run_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    write_json(state_path(run_dir), state)


def command_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    return env


def run_logged(cmd: list[str], log_path: Path, *, env: dict[str, str] | None = None) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + shell_join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, text=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    return {"cmd": cmd, "log": str(log_path), "returncode": int(proc.returncode)}


def hash_existing(path: Path) -> str | None:
    return sha256_file(path) if path.exists() and path.is_file() else None


def split_group_value(row: dict[str, Any], key: str) -> str:
    if key == "match":
        return str(row.get("map_memory_match_id") or row.get("game_id") or "")
    if key == "episode":
        return str(row.get("map_memory_episode") or row.get("episode") or "")
    if key == "track":
        return str(row.get("map_memory_track_id") or "")
    raise ValueError(f"unknown split group key: {key}")


def audit_split_leakage(aligned_path: Path, *, enforced_split_key: str, out_json: Path) -> dict[str, Any]:
    rows = list(iter_jsonl(aligned_path))
    audits: dict[str, Any] = {}
    hard_failures: list[str] = []
    for key in ["match", "episode", "track"]:
        group_splits: dict[str, set[str]] = {}
        group_rows: Counter[str] = Counter()
        for row in rows:
            group = split_group_value(row, key)
            split = str(row.get("map_memory_split", ""))
            if not group:
                continue
            group_splits.setdefault(group, set()).add(split)
            group_rows[group] += 1
        leaking = {
            group: sorted(splits)
            for group, splits in group_splits.items()
            if len(splits) > 1
        }
        audits[key] = {
            "group_count": len(group_splits),
            "multi_split_group_count": len(leaking),
            "multi_split_examples": [
                {"group": group, "splits": splits, "rows": int(group_rows[group])}
                for group, splits in list(sorted(leaking.items()))[:20]
            ],
        }
        if key == enforced_split_key and leaking:
            hard_failures.append(f"{key} split leakage: {len(leaking)} group(s) cross splits")

    report = {
        "kind": "memory_dense_lingbot_split_leakage_report_v0",
        "status": "pass" if not hard_failures else "fail",
        "aligned_cache_manifest": str(aligned_path),
        "row_count": len(rows),
        "enforced_split_key": enforced_split_key,
        "audits": audits,
        "failures": hard_failures,
    }
    write_json(out_json, report)
    if hard_failures:
        raise RuntimeError("split leakage audit failed:\n" + "\n".join(f"- {msg}" for msg in hard_failures))
    return report


def artifact_hashes(final_dir: Path) -> dict[str, str | None]:
    paths = {
        "manifest_sha256": final_dir / "manifest.json",
        "aligned_cache_manifest_sha256": final_dir / "aligned_cache_manifest.jsonl",
        "teacher_qa_sha256": final_dir / "channel_teacher_qa_v0" / "memory_dense_channels_vs_teacher_v0.json",
        "training_readiness_sha256": final_dir / "training_readiness_v0.json",
        "combined_report_sha256": final_dir / "combined_report_v0.json",
        "preflight_train_sha256": final_dir / "preflight_train_v0.json",
        "split_leakage_report_sha256": final_dir / "split_leakage_report_v0.json",
    }
    return {key: hash_existing(path) for key, path in paths.items()}


def infer_raw_match_exists(raw_root: Path, row: dict[str, Any]) -> bool:
    hash_id = str(row.get("hash", ""))
    game_id = str(row.get("game_id", ""))
    if hash_id and game_id and (raw_root / hash_id / game_id).exists():
        return True
    match_id = str(row.get("match_id", ""))
    if hash_id and match_id and (raw_root / hash_id / match_id).exists():
        return True
    return False


def build_filtered_manifests(args: argparse.Namespace) -> dict[str, Any]:
    filtered_dir = args.filtered_dir or args.base_dir / "filtered_manifests"
    filtered_dir.mkdir(parents=True, exist_ok=True)
    cache_out = filtered_dir / args.filtered_cache_name
    source_out = filtered_dir / args.filtered_source_name
    report_out = filtered_dir / args.filtered_report_name

    source_rows_by_clip: dict[str, dict[str, Any]] = {}
    eligible_clip_ids: set[str] = set()
    source_total = 0
    source_missing_clip = 0
    source_raw_missing = 0
    filtered_source_rows: list[dict[str, Any]] = []
    for row in iter_jsonl(args.source_manifest):
        source_total += 1
        clip_id = str(row.get("clip_id", ""))
        if not clip_id:
            source_missing_clip += 1
            continue
        source_rows_by_clip[clip_id] = row
        if not infer_raw_match_exists(args.raw_root, row):
            source_raw_missing += 1
            continue
        eligible_clip_ids.add(clip_id)
        filtered_source_rows.append(row)

    filtered_cache_rows: list[dict[str, Any]] = []
    cache_total = 0
    cache_missing_source = 0
    cache_not_raw_subset = 0
    for row in iter_jsonl(args.cache_manifest):
        cache_total += 1
        clip_id = str(row.get("clip_id", ""))
        if clip_id not in source_rows_by_clip:
            cache_missing_source += 1
            continue
        if clip_id not in eligible_clip_ids:
            cache_not_raw_subset += 1
            continue
        filtered_cache_rows.append(row)

    source_clip_set = {str(row.get("clip_id", "")) for row in filtered_source_rows}
    cache_clip_set = {str(row.get("clip_id", "")) for row in filtered_cache_rows}
    common_clip_set = source_clip_set & cache_clip_set
    filtered_source_rows = [row for row in filtered_source_rows if str(row.get("clip_id", "")) in common_clip_set]
    filtered_cache_rows = [row for row in filtered_cache_rows if str(row.get("clip_id", "")) in common_clip_set]
    match_counts: Counter[str] = Counter()
    episode_counts: Counter[str] = Counter()
    for row in filtered_source_rows:
        match_counts[str(row.get("game_id", ""))] += 1
        episode_counts[f"{row.get('game_id', '')}_{row.get('episode', '')}"] += 1

    write_jsonl(source_out, filtered_source_rows)
    write_jsonl(cache_out, filtered_cache_rows)
    report = {
        "kind": "memory_dense_lingbot_filtered_manifest_report_v0",
        "status": "pass" if filtered_cache_rows and len(filtered_cache_rows) == len(filtered_source_rows) else "fail",
        "created_at": utc_now(),
        "cache_manifest": str(args.cache_manifest),
        "source_manifest": str(args.source_manifest),
        "raw_root": str(args.raw_root),
        "filtered_cache_manifest": str(cache_out),
        "filtered_source_manifest": str(source_out),
        "source_total": source_total,
        "source_missing_clip": source_missing_clip,
        "source_raw_missing": source_raw_missing,
        "cache_total": cache_total,
        "cache_missing_source": cache_missing_source,
        "cache_not_raw_subset": cache_not_raw_subset,
        "filtered_cache_rows": len(filtered_cache_rows),
        "filtered_source_rows": len(filtered_source_rows),
        "match_count": len([m for m in match_counts if m]),
        "episode_count": len([e for e in episode_counts if e and not e.endswith("_")]),
        "match_counts": dict(sorted(match_counts.items())),
        "cache_sha256": sha256_file(cache_out),
        "source_sha256": sha256_file(source_out),
    }
    if report["status"] != "pass":
        report["failures"] = ["filtered cache/source manifests are empty or length-mismatched"]
    write_json(report_out, report)
    if report["status"] != "pass":
        raise SystemExit(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def shard_specs(row_count: int, args: argparse.Namespace, run_dir: Path) -> list[dict[str, Any]]:
    start = max(0, args.start_offset)
    end = row_count if args.limit is None else min(row_count, start + max(0, args.limit))
    if start >= end:
        raise ValueError(f"empty shard range: start_offset={start}, end={end}, row_count={row_count}")
    total = end - start
    if args.shard_size is not None:
        shard_size = max(1, args.shard_size)
        starts = list(range(start, end, shard_size))
    else:
        count = max(1, args.shard_count)
        shard_size = (total + count - 1) // count
        starts = list(range(start, end, shard_size))

    specs = []
    for idx, shard_start in enumerate(starts):
        shard_end = min(end, shard_start + shard_size)
        limit = shard_end - shard_start
        name = f"fullsubset_large_shard_{idx:02d}_{shard_start:04d}_{shard_end:04d}"
        specs.append(
            {
                "index": idx,
                "start_offset": shard_start,
                "limit": limit,
                "out_dir": str(run_dir / "shards" / name),
                "log": str(run_dir / "logs" / f"{name}.log"),
            }
        )
    return specs


def ensure_run_dir(args: argparse.Namespace) -> Path:
    run_name = args.run_name or datetime.now().strftime("memory_dense_lingbot_large_%Y%m%d_%H%M%S")
    run_dir = args.run_dir or args.base_dir / "runs" / run_name
    if run_dir.exists() and not args.resume and args.mode in {"plan", "launch-shards", "all"}:
        if args.clean_run_dir:
            shutil.rmtree(run_dir)
        else:
            raise FileExistsError(f"run dir already exists; pass --resume or --clean-run-dir: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def export_cmd(spec: dict[str, Any], filtered: dict[str, Any], args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "tools/export_memory_dense_from_lingbot_cache_v0.py",
        "--cache-manifest",
        str(filtered["filtered_cache_manifest"]),
        "--source-manifest",
        str(filtered["filtered_source_manifest"]),
        "--raw-root",
        str(args.raw_root),
        "--bsp-faces-npz",
        str(args.bsp_faces_npz),
        "--out-dir",
        str(spec["out_dir"]),
        "--start-offset",
        str(spec["start_offset"]),
        "--limit",
        str(spec["limit"]),
        "--max-accepted-clips",
        str(args.max_accepted_clips_per_shard),
        "--min-accepted-clips",
        str(args.min_accepted_clips_per_shard),
        "--min-positive-samples",
        str(args.min_positive_samples_per_shard),
        "--min-samples",
        str(args.min_samples_per_shard),
        "--min-episodes",
        str(args.min_episodes_per_shard),
        "--readiness-max-samples",
        str(args.readiness_max_samples),
        "--fov-x",
        str(args.fov_x),
        "--min-hit-iou",
        str(args.min_hit_iou),
        "--max-depth-mae",
        str(args.max_depth_mae),
        "--min-positive-other-iou",
        str(args.min_positive_other_iou),
        "--min-positive-other-iou-mean",
        str(args.min_positive_other_iou_mean),
        "--split-key",
        args.split_key,
        "--val-fraction",
        str(args.val_fraction),
        "--test-fraction",
        str(args.test_fraction),
        "--seed",
        str(args.seed),
        "--progress-every",
        str(args.progress_every),
        "--no-reuse-existing",
    ]
    return cmd


def initialize_or_update_plan(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    run_dir = ensure_run_dir(args)
    if args.resume and state_path(run_dir).exists():
        state = load_state(run_dir)
        return run_dir, state

    filtered = build_filtered_manifests(args)
    specs = shard_specs(int(filtered["filtered_cache_rows"]), args, run_dir)
    for spec in specs:
        spec["cmd"] = export_cmd(spec, filtered, args)
        spec["status"] = "planned"
    final_dir = run_dir / "final_release"
    state = {
        "kind": "memory_dense_lingbot_training_prep_state_v0",
        "created_at": utc_now(),
        "run_name": args.run_name or run_dir.name,
        "run_dir": str(run_dir),
        "mode": args.mode,
        "git_revision": git_revision(),
        "train_command_args": {
            "lingbot_repo": str(args.lingbot_repo),
            "train_split": args.train_split,
            "min_train_records": args.min_train_records,
            "min_train_positive_frames": args.min_train_positive_frames,
            "min_train_context_frames": args.min_train_context_frames,
            "min_train_matches": args.min_train_matches,
            "min_train_episodes": args.min_train_episodes,
            "train_nproc_per_node": args.train_nproc_per_node,
            "train_max_steps": args.train_max_steps,
            "train_out_dir": str(args.train_out_dir) if args.train_out_dir else None,
        },
        "filtered_manifest_report": filtered,
        "thresholds": threshold_summary(args),
        "parallel_shards": args.parallel_shards,
        "shards": specs,
        "final_dir": str(final_dir),
        "summary_json": str(run_dir / "large_training_prep_summary_v0.json"),
        "optimizer_steps_run": 0,
    }
    save_state(run_dir, state)
    return run_dir, state


def threshold_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "export": {
            "max_accepted_clips_per_shard": args.max_accepted_clips_per_shard,
            "min_accepted_clips_per_shard": args.min_accepted_clips_per_shard,
            "min_positive_samples_per_shard": args.min_positive_samples_per_shard,
            "min_hit_iou": args.min_hit_iou,
            "max_depth_mae": args.max_depth_mae,
            "min_positive_other_iou": args.min_positive_other_iou,
            "min_positive_other_iou_mean": args.min_positive_other_iou_mean,
        },
        "combine": {
            "min_final_accepted_clips": args.min_final_accepted_clips,
            "min_final_positive_samples": args.min_final_positive_samples,
            "min_final_samples": args.min_final_samples,
            "min_final_episodes": args.min_final_episodes,
        },
        "preflight": {
            "min_train_records": args.min_train_records,
            "min_train_positive_frames": args.min_train_positive_frames,
            "min_train_context_frames": args.min_train_context_frames,
            "min_train_matches": args.min_train_matches,
            "min_train_episodes": args.min_train_episodes,
            "preflight_records": args.preflight_records,
        },
    }


def run_one_shard(spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(str(spec["out_dir"]))
    if out_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to reuse shard out-dir without --resume: {out_dir}")
    result = run_logged([str(x) for x in spec["cmd"]], Path(str(spec["log"])), env=command_env(args))
    report_path = out_dir / "export_report_v0.json"
    report = load_json(report_path) if report_path.exists() else None
    status = "pass" if result["returncode"] == 0 and report and report.get("status") == "pass" else "fail"
    return {
        **spec,
        "status": status,
        "returncode": result["returncode"],
        "report": str(report_path) if report_path.exists() else None,
        "accepted_clips": report.get("accepted_clips") if report else None,
        "accepted_unique_samples": report.get("accepted_unique_samples") if report else None,
        "role_counts": report.get("role_counts") if report else None,
        "rejected_clips": report.get("rejected_clips") if report else None,
        "export_failures": report.get("export_failures") if report else ["export_report_v0.json missing"],
    }


def launch_shards(run_dir: Path, state: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    pending = [spec for spec in state["shards"] if args.resume and spec.get("status") != "pass" or not args.resume]
    if not pending:
        return state

    completed: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.parallel_shards)) as pool:
        futures = {pool.submit(run_one_shard, spec, args): int(spec["index"]) for spec in pending}
        for fut in as_completed(futures):
            idx = futures[fut]
            completed[idx] = fut.result()
            state["shards"] = [completed.get(int(s["index"]), s) for s in state["shards"]]
            save_state(run_dir, state)
            print(json.dumps({"event": "shard_finished", "index": idx, "status": completed[idx]["status"]}, ensure_ascii=False), flush=True)
    return state


def passed_shard_dirs(state: dict[str, Any], *, require_all: bool) -> list[Path]:
    failed = [s for s in state["shards"] if s.get("status") != "pass"]
    if failed and require_all:
        raise RuntimeError(f"{len(failed)} shard(s) did not pass; first={failed[0]}")
    dirs = [Path(str(s["out_dir"])) for s in state["shards"] if s.get("status") == "pass"]
    if not dirs:
        raise RuntimeError("no passed shards to combine")
    return dirs


def validate_shards_against_plan(state: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    clip_ids: set[str] = set()
    duplicate_clip_ids: list[str] = []
    filtered = state.get("filtered_manifest_report") or {}
    cache_manifest = str(filtered.get("filtered_cache_manifest", ""))
    source_manifest = str(filtered.get("filtered_source_manifest", ""))
    ranges = []
    for spec in state.get("shards", []):
        if spec.get("status") != "pass":
            continue
        report_path = Path(str(spec.get("report") or ""))
        if not report_path.exists():
            failures.append(f"shard {spec.get('index')}: missing report {report_path}")
            continue
        report = load_json(report_path)
        if int(report.get("start_offset", -1)) != int(spec.get("start_offset", -2)):
            failures.append(f"shard {spec.get('index')}: start_offset mismatch")
        if int(report.get("limit", -1)) != int(spec.get("limit", -2)):
            failures.append(f"shard {spec.get('index')}: limit mismatch")
        if cache_manifest and str(report.get("cache_manifest")) != cache_manifest:
            failures.append(f"shard {spec.get('index')}: cache manifest mismatch")
        if source_manifest and str(report.get("source_manifest")) != source_manifest:
            failures.append(f"shard {spec.get('index')}: source manifest mismatch")
        ranges.append((int(spec.get("start_offset", 0)), int(spec.get("start_offset", 0)) + int(spec.get("limit", 0)), spec.get("index")))
        aligned_path = Path(str(report.get("aligned_cache_manifest") or ""))
        if aligned_path.exists():
            for row in iter_jsonl(aligned_path):
                clip_id = str(row.get("clip_id", ""))
                if clip_id in clip_ids and len(duplicate_clip_ids) < 20:
                    duplicate_clip_ids.append(clip_id)
                clip_ids.add(clip_id)

    for (start_a, end_a, idx_a), (start_b, end_b, idx_b) in zip(sorted(ranges), sorted(ranges)[1:]):
        if end_a > start_b:
            failures.append(f"shard ranges overlap: {idx_a} [{start_a},{end_a}) and {idx_b} [{start_b},{end_b})")

    report = {
        "kind": "memory_dense_lingbot_shard_plan_audit_v0",
        "status": "pass" if not failures and not duplicate_clip_ids else "fail",
        "passed_shard_count": len(ranges),
        "accepted_unique_clip_ids": len(clip_ids),
        "duplicate_clip_id_examples": duplicate_clip_ids,
        "failures": failures + ([f"duplicate accepted clip ids: {len(duplicate_clip_ids)} example(s)"] if duplicate_clip_ids else []),
    }
    if report["status"] != "pass":
        raise RuntimeError("shard plan audit failed:\n" + "\n".join(f"- {msg}" for msg in report["failures"]))
    return report


def combine_cmd(shard_dirs: list[Path], state: dict[str, Any], args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "tools/combine_memory_dense_lingbot_shards_v0.py"]
    for shard_dir in shard_dirs:
        cmd += ["--shard-dir", str(shard_dir)]
    cmd += [
        "--out-dir",
        str(state["final_dir"]),
        "--min-accepted-clips",
        str(args.min_final_accepted_clips),
        "--min-positive-samples",
        str(args.min_final_positive_samples),
        "--min-samples",
        str(args.min_final_samples),
        "--min-episodes",
        str(args.min_final_episodes),
        "--readiness-max-samples",
        str(args.readiness_max_samples),
    ]
    return cmd


def run_combine(run_dir: Path, state: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    shard_dirs = passed_shard_dirs(state, require_all=args.require_all_shards)
    shard_plan_audit = validate_shards_against_plan(state)
    final_dir = Path(str(state["final_dir"]))
    if final_dir.exists() and not args.resume:
        if args.clean_final_dir:
            shutil.rmtree(final_dir)
        else:
            raise FileExistsError(f"final dir already exists; pass --resume or --clean-final-dir: {final_dir}")
    cmd = combine_cmd(shard_dirs, state, args)
    result = run_logged(cmd, run_dir / "logs" / "combine.log", env=command_env(args))
    write_json(run_dir / "shard_plan_audit_v0.json", shard_plan_audit)
    state["shard_plan_audit"] = str(run_dir / "shard_plan_audit_v0.json")
    state["combine"] = result
    report_path = final_dir / "combined_report_v0.json"
    state["combined_report"] = str(report_path) if report_path.exists() else None
    state["combined_status"] = load_json(report_path).get("status") if report_path.exists() else "missing"
    save_state(run_dir, state)
    if result["returncode"] != 0:
        raise RuntimeError(f"combine failed; see {result['log']}")
    split_report = audit_split_leakage(
        final_dir / "aligned_cache_manifest.jsonl",
        enforced_split_key=args.split_key,
        out_json=final_dir / "split_leakage_report_v0.json",
    )
    state["split_leakage_report"] = str(final_dir / "split_leakage_report_v0.json")
    state["split_leakage_status"] = split_report.get("status")
    save_state(run_dir, state)
    return state


def preflight_cmd(state: dict[str, Any], args: argparse.Namespace) -> list[str]:
    final_dir = Path(str(state["final_dir"]))
    return [
        sys.executable,
        "tools/train_memory_dense_adapter_v0.py",
        "preflight-train",
        "--map-manifest",
        str(final_dir / "manifest.json"),
        "--cache-manifest",
        str(final_dir / "aligned_cache_manifest.jsonl"),
        "--lingbot-repo",
        str(args.lingbot_repo),
        "--device",
        args.device,
        "--train-split",
        args.train_split,
        "--preflight-records",
        str(args.preflight_records),
        "--min-train-records",
        str(args.min_train_records),
        "--min-train-positive-frames",
        str(args.min_train_positive_frames),
        "--min-train-context-frames",
        str(args.min_train_context_frames),
        "--min-train-matches",
        str(args.min_train_matches),
        "--min-train-episodes",
        str(args.min_train_episodes),
        "--out-json",
        str(final_dir / "preflight_train_v0.json"),
    ]


def run_preflight(run_dir: Path, state: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cmd = preflight_cmd(state, args)
    result = run_logged(cmd, run_dir / "logs" / "preflight_train.log", env=command_env(args))
    final_dir = Path(str(state["final_dir"]))
    preflight_path = final_dir / "preflight_train_v0.json"
    state["preflight"] = result
    state["preflight_report"] = str(preflight_path) if preflight_path.exists() else None
    state["preflight_status"] = load_json(preflight_path).get("status") if preflight_path.exists() else "missing"
    save_state(run_dir, state)
    if result["returncode"] != 0:
        raise RuntimeError(f"preflight failed; see {result['log']}")
    return state


def train_command(state: dict[str, Any], args: argparse.Namespace) -> list[str]:
    final_dir = Path(str(state["final_dir"]))
    saved = state.get("train_command_args") or {}
    lingbot_repo = saved.get("lingbot_repo", str(args.lingbot_repo))
    train_split = saved.get("train_split", args.train_split)
    min_train_records = saved.get("min_train_records", args.min_train_records)
    min_train_positive_frames = saved.get("min_train_positive_frames", args.min_train_positive_frames)
    min_train_context_frames = saved.get("min_train_context_frames", args.min_train_context_frames)
    min_train_matches = saved.get("min_train_matches", args.min_train_matches)
    min_train_episodes = saved.get("min_train_episodes", args.min_train_episodes)
    nproc = saved.get("train_nproc_per_node", args.train_nproc_per_node)
    max_steps = saved.get("train_max_steps", args.train_max_steps)
    out_dir = saved.get("train_out_dir") or str(args.train_out_dir or Path(str(state["run_dir"])) / "train")
    return [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "tools/train_memory_dense_adapter_v0.py",
        "train",
        "--map-manifest",
        str(final_dir / "manifest.json"),
        "--cache-manifest",
        str(final_dir / "aligned_cache_manifest.jsonl"),
        "--lingbot-repo",
        str(lingbot_repo),
        "--train-split",
        str(train_split),
        "--min-train-records",
        str(min_train_records),
        "--min-train-positive-frames",
        str(min_train_positive_frames),
        "--min-train-context-frames",
        str(min_train_context_frames),
        "--min-train-matches",
        str(min_train_matches),
        "--min-train-episodes",
        str(min_train_episodes),
        "--max-steps",
        str(max_steps),
        "--out-dir",
        str(out_dir),
    ]


def summarize_run(run_dir: Path, state: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    final_dir = Path(str(state["final_dir"]))
    combined_path = final_dir / "combined_report_v0.json"
    preflight_path = final_dir / "preflight_train_v0.json"
    split_path = final_dir / "split_leakage_report_v0.json"
    shard_plan_path = run_dir / "shard_plan_audit_v0.json"
    combined = load_json(combined_path) if combined_path.exists() else None
    preflight = load_json(preflight_path) if preflight_path.exists() else None
    split_report = load_json(split_path) if split_path.exists() else None
    shard_plan_audit = load_json(shard_plan_path) if shard_plan_path.exists() else None
    shard_status_counts = Counter(str(s.get("status", "unknown")) for s in state.get("shards", []))
    summary = {
        "kind": "memory_dense_lingbot_large_training_prep_summary_v0",
        "created_at": utc_now(),
        "run_dir": str(run_dir),
        "status": "pass"
        if combined
        and combined.get("status") == "pass"
        and preflight
        and preflight.get("status") == "pass"
        and split_report
        and split_report.get("status") == "pass"
        and shard_plan_audit
        and shard_plan_audit.get("status") == "pass"
        and int(preflight.get("optimizer_steps_run", -1)) == 0
        else "pending_or_fail",
        "optimizer_steps_run": int(preflight.get("optimizer_steps_run", 0)) if preflight else 0,
        "git_revision": state.get("git_revision"),
        "filtered_manifest_report": state.get("filtered_manifest_report"),
        "thresholds": state.get("thresholds"),
        "artifact_hashes": artifact_hashes(final_dir),
        "shard_status_counts": dict(sorted(shard_status_counts.items())),
        "shard_plan_audit": shard_plan_audit,
        "shards": state.get("shards", []),
        "combined_report": combined,
        "split_leakage_report": split_report,
        "preflight_report": preflight,
        "next_train_command": train_command(state, args),
        "next_train_command_shell": shell_join(train_command(state, args)),
        "policy": {
            "starts_optimizer": False,
            "requires_clean_shards": True,
            "export_reuse_existing": False,
            "teacher_streams_are_model_inputs": False,
            "nearest_or_repeat_dense_filling": False,
        },
    }
    write_json(Path(str(state["summary_json"])), summary)
    state["summary_status"] = summary["status"]
    save_state(run_dir, state)
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["filter", "plan", "launch-shards", "collect", "all", "summarize"])
    ap.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--clean-run-dir", action="store_true")
    ap.add_argument("--clean-final-dir", action="store_true")

    ap.add_argument("--cache-manifest", type=Path, default=DEFAULT_CACHE_MANIFEST)
    ap.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    ap.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    ap.add_argument("--lingbot-repo", type=Path, default=DEFAULT_LINGBOT_REPO)
    ap.add_argument("--bsp-faces-npz", type=Path, default=DEFAULT_BSP_FACES_NPZ)
    ap.add_argument("--filtered-dir", type=Path, default=None)
    ap.add_argument("--filtered-cache-name", default="cache_manifest_fullsubset_5matches.jsonl")
    ap.add_argument("--filtered-source-name", default="source_manifest_fullsubset_5matches.jsonl")
    ap.add_argument("--filtered-report-name", default="filtered_manifest_report_v0.json")

    ap.add_argument("--start-offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard-count", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=None)
    ap.add_argument("--parallel-shards", type=int, default=1)
    ap.add_argument("--require-all-shards", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--max-accepted-clips-per-shard", type=int, default=50)
    ap.add_argument("--min-accepted-clips-per-shard", type=int, default=1)
    ap.add_argument("--min-positive-samples-per-shard", type=int, default=0)
    ap.add_argument("--min-samples-per-shard", type=int, default=21)
    ap.add_argument("--min-episodes-per-shard", type=int, default=1)
    ap.add_argument("--readiness-max-samples", type=int, default=4096)

    ap.add_argument("--fov-x", type=float, default=106.26)
    ap.add_argument("--min-hit-iou", type=float, default=0.90)
    ap.add_argument("--max-depth-mae", type=float, default=0.08)
    ap.add_argument("--min-positive-other-iou", type=float, default=0.25)
    ap.add_argument("--min-positive-other-iou-mean", type=float, default=0.45)
    ap.add_argument("--split-key", choices=["episode", "track", "match"], default="episode")
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--test-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260531)
    ap.add_argument("--progress-every", type=int, default=100)

    ap.add_argument("--min-final-accepted-clips", type=int, default=256)
    ap.add_argument("--min-final-positive-samples", type=int, default=16)
    ap.add_argument("--min-final-samples", type=int, default=5376)
    ap.add_argument("--min-final-episodes", type=int, default=8)

    ap.add_argument("--cuda-visible-devices", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--preflight-records", type=int, default=8)
    ap.add_argument("--min-train-records", type=int, default=128)
    ap.add_argument("--min-train-positive-frames", type=int, default=16)
    ap.add_argument("--min-train-context-frames", type=int, default=0)
    ap.add_argument("--min-train-matches", type=int, default=2)
    ap.add_argument("--min-train-episodes", type=int, default=8)

    ap.add_argument("--train-nproc-per-node", type=int, default=4)
    ap.add_argument("--train-max-steps", type=int, default=1000)
    ap.add_argument("--train-out-dir", type=Path, default=None)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "filter":
        print(json.dumps(build_filtered_manifests(args), ensure_ascii=False, indent=2))
        return

    if args.mode == "summarize":
        if not args.run_dir:
            raise ValueError("--run-dir is required for summarize")
        run_dir = args.run_dir
        state = load_state(run_dir)
        print(json.dumps(summarize_run(run_dir, state, args), ensure_ascii=False, indent=2))
        return

    if args.mode == "collect":
        if not args.run_dir:
            raise ValueError("--run-dir is required for collect")
        run_dir = args.run_dir
        state = load_state(run_dir)
        state = run_combine(run_dir, state, args)
        state = run_preflight(run_dir, state, args)
        print(json.dumps(summarize_run(run_dir, state, args), ensure_ascii=False, indent=2))
        return

    run_dir, state = initialize_or_update_plan(args)
    if args.mode == "plan":
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return

    if args.mode in {"launch-shards", "all"}:
        state = launch_shards(run_dir, state, args)
        if args.mode == "launch-shards":
            print(json.dumps(state, ensure_ascii=False, indent=2))
            return

    if args.mode == "all":
        state = run_combine(run_dir, state, args)
        state = run_preflight(run_dir, state, args)
        print(json.dumps(summarize_run(run_dir, state, args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
