#!/usr/bin/env python3
"""Resumable single-GPU worker for multiscene 10-ego generation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(path)


def parse_task(value: str) -> tuple[str, int]:
    try:
        scene_id, raw_view = value.rsplit(":", 1)
        view = int(raw_view)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("task must have form SCENE_ID:VIEW_INDEX") from error
    if not scene_id or view not in range(10):
        raise argparse.ArgumentTypeError("view index must be between 0 and 9")
    return scene_id, view


def generation_complete(output_root: Path, scene_id: str, view: int, seed: int, steps: int) -> bool:
    report = output_root / "scenes" / scene_id / "generation" / f"view_{view:02d}" / "candidate1_ar_run_report_v0.json"
    if not report.is_file():
        return False
    try:
        value = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if value.get("status") != "complete" or value.get("completed_windows") != 2 or value.get("failed_windows") != 0:
        return False
    windows = value.get("windows", [])
    return len(windows) == 2 and all(window.get("seed") == seed and window.get("steps") == steps for window in windows)


def run(command: list[str], project_root: Path, environment: dict[str, str]) -> None:
    subprocess.run(command, cwd=project_root, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--task", action="append", type=parse_task, required=True)
    parser.add_argument("--checkpoint-low", required=True)
    parser.add_argument("--checkpoint-high", required=True)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--steps", type=int, default=70)
    parser.add_argument("--worker-name", required=True)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    tool = project_root / "tools/run_multiscene_10ego_demo_v1.py"
    report_path = output_root / "worker_reports" / f"{args.worker_name}.json"
    report = {
        "kind": "multiscene_10ego_gpu_worker_v1",
        "status": "running",
        "worker_name": args.worker_name,
        "gpu": str(args.gpu),
        "pid": os.getpid(),
        "tasks": [{"scene_id": scene_id, "view": view, "status": "pending"} for scene_id, view in args.task],
        "seed": args.seed,
        "steps": args.steps,
    }
    atomic_json(report_path, report)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    try:
        for task_record, (scene_id, view) in zip(report["tasks"], args.task):
            if generation_complete(output_root, scene_id, view, args.seed, args.steps):
                task_record["status"] = "already_complete"
                atomic_json(report_path, report)
                continue
            task_record["status"] = "rendering_dense"
            task_record["started_unix"] = time.time()
            atomic_json(report_path, report)
            run([
                sys.executable, str(tool), "render-dense", "--project-root", str(project_root),
                "--output-root", str(output_root), "--scene-id", scene_id, "--view", str(view),
            ], project_root, environment)
            task_record["status"] = "generating"
            atomic_json(report_path, report)
            run([
                sys.executable, str(tool), "generate-one", "--project-root", str(project_root),
                "--output-root", str(output_root), "--scene-id", scene_id, "--view", str(view),
                "--checkpoint-low", str(Path(args.checkpoint_low).resolve()),
                "--checkpoint-high", str(Path(args.checkpoint_high).resolve()),
                "--steps", str(args.steps), "--seed", str(args.seed),
            ], project_root, environment)
            if not generation_complete(output_root, scene_id, view, args.seed, args.steps):
                raise RuntimeError(f"post-generation validation failed: {scene_id}:{view}")
            task_record["status"] = "complete"
            task_record["elapsed_seconds"] = time.time() - task_record["started_unix"]
            atomic_json(report_path, report)
        report["status"] = "pass"
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = repr(error)
        atomic_json(report_path, report)
        raise
    atomic_json(report_path, report)
    print(json.dumps({"status": "pass", "worker": args.worker_name, "tasks": len(args.task)}))


if __name__ == "__main__":
    main()
