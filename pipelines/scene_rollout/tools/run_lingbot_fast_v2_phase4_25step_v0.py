#!/usr/bin/env python3
"""Authorized Fast v2 Phase 4 run: exactly 25 local steps from total step 4."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping


TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(TOOLS_DIR))

from run_lingbot_fast_v2_phase3_smoke_v0 import (  # noqa: E402
    TrainingQualityError,
    _execute_fixed_training,
    atomic_json,
    best_effort_destroy_distributed,
    build_strict_preflight,
    load_json,
    microbatch_forward_budget,
    sha256_file,
    sha256_json,
    utc_now,
    validate_config as validate_phase3_config,
)


CONFIG_SCHEMA = "lingbot-fast-v2-phase4-25step-config/v1"
AUTHORIZATION_SCHEMA = "lingbot-fast-v2-phase4-launch-authorization/v1"
PREFLIGHT_KIND = "lingbot-fast-v2-phase4-25step-preflight/v1"
CHECKPOINT_KIND = "lingbot-fast-v2-phase4-25step-checkpoint/v1"
GATE_KIND = "lingbot-fast-v2-phase4-25step-gate/v1"
LOCAL_OPTIMIZER_STEPS = 25
SOURCE_OPTIMIZER_STEPS = 4
TOTAL_OPTIMIZER_STEPS = 29
DEFAULT_CONFIG = PROJECT_ROOT / "configs/lingbot_fast_v2_phase4_25step_from_step4_v0.json"
DEFAULT_REPORT = PROJECT_ROOT / "output/lingbot_fast_v2_phase4_25step_validation_v0.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "output/lingbot_fast_v2_phase4_25step_from_step4_v0"
EXPECTED_BASE_CONFIG = PROJECT_ROOT / "configs/lingbot_fast_v2_phase3_smoke_v0.json"
EXPECTED_AUTHORIZATION = PROJECT_ROOT / "configs/lingbot_fast_v2_phase4_25step_authorization_v0.json"
EXPECTED_AUTHORIZATION_SHA256 = "be9be39fb98f07d7b504901d2206fa4eaf5510f66070a77303024acb7266e343"
EXPECTED_SOURCE_GATE = PROJECT_ROOT / "output/lingbot_fast_v2_phase3_resume_smoke_canonical_retry6_v0/FAST_V2_PHASE3_SMOKE_GATE_REPORT_v0.json"
EXPECTED_SOURCE_GATE_SHA256 = "0a9d5e5ea0706023e246fb21221d4aa2cbe2713fc75781caf1bc89eb37f4c138"
EXPECTED_SOURCE_CHECKPOINT = PROJECT_ROOT / "output/lingbot_fast_v2_phase3_resume_smoke_canonical_retry6_v0/checkpoints/lingbot_fast_v2_phase3_smoke_step_000004_compact.pt"
EXPECTED_SOURCE_CHECKPOINT_SHA256 = "3e4bedb18b98299558a693de4fe0119d626e53b70a2eb055b00725847fdbd9c9"


class Phase4ContractError(RuntimeError):
    pass


def _expect(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise Phase4ContractError(f"{name} must be {expected!r}, got {actual!r}")


def validate_phase4_config(config: Mapping[str, Any]) -> dict[str, Any]:
    _expect("Phase4 schema", config.get("schema_version"), CONFIG_SCHEMA)
    for key in (
        "authorization", "source_gate", "source_checkpoint", "execution",
        "microbatch_schedule", "output",
    ):
        if not isinstance(config.get(key), Mapping):
            raise Phase4ContractError(f"config.{key} must be an object")
    execution = config["execution"]
    schedule = config["microbatch_schedule"]
    output = config["output"]
    authorization = config["authorization"]
    source_gate = config["source_gate"]
    source_checkpoint = config["source_checkpoint"]
    _expect("world size", execution.get("world_size"), 8)
    _expect("sequence parallel size", execution.get("sequence_parallel_size"), 8)
    _expect("local optimizer steps", execution.get("local_optimizer_steps"), 25)
    _expect("gradient accumulation", execution.get("gradient_accumulation_steps"), 2)
    _expect("hard stop total steps", execution.get("hard_stop_total_optimizer_steps"), 29)
    _expect("learning rate", float(execution.get("learning_rate")), 0.00005)
    _expect("LoRA learning rate", float(execution.get("lora_learning_rate")), 0.000005)
    _expect("weight decay", float(execution.get("weight_decay")), 0.01)
    _expect("max gradient norm", float(execution.get("max_gradient_norm")), 1.0)
    _expect(
        "projector gradient policy",
        execution.get("projector_gradient_must_be_finite_nonzero_each_step"),
        True,
    )
    _expect(
        "microbatch cycle",
        schedule.get("cycle_sources"),
        ["w3", "tier69h", "event50h", "tier69h"],
    )
    _expect("target chunk cycle", schedule.get("cycle_target_chunks"), [0, 1, 3, 4])
    _expect("timestep cycle", schedule.get("cycle_timestep_indices"), [0, 1, 2, 3])
    _expect("local global microbatches", schedule.get("local_global_microbatch_count"), 50)
    _expect(
        "realized source counts",
        schedule.get("realized_source_counts"),
        {"w3": 13, "tier69h": 25, "event50h": 12},
    )
    _expect("checkpoint kind", output.get("checkpoint_kind"), CHECKPOINT_KIND)
    _expect("gate kind", output.get("gate_kind"), GATE_KIND)
    _expect("automatic extension", output.get("authorizes_automatic_extension"), False)
    _expect("maximum local steps", output.get("maximum_local_optimizer_steps_authorized"), 25)
    _expect("fixed output directory", Path(str(output.get("directory"))).resolve(), DEFAULT_OUT_DIR)
    _expect("authorization path", Path(str(authorization.get("path"))).resolve(), EXPECTED_AUTHORIZATION)
    _expect("authorization sha256", authorization.get("sha256"), EXPECTED_AUTHORIZATION_SHA256)
    _expect("authorization CLI steps", authorization.get("required_cli_authorized_optimizer_steps"), 25)
    _expect("source gate path", Path(str(source_gate.get("path"))).resolve(), EXPECTED_SOURCE_GATE)
    _expect("source gate size", source_gate.get("size_bytes"), 327512)
    _expect("source gate sha256", source_gate.get("sha256"), EXPECTED_SOURCE_GATE_SHA256)
    _expect("source gate status", source_gate.get("required_status"), "pass")
    _expect("source gate total steps", source_gate.get("required_total_optimizer_steps"), 4)
    _expect("source gate projector status", source_gate.get("required_projector_gradient_status"), "pass")
    _expect("source gate projector all rank activation", source_gate.get("required_projector_all_steps_all_ranks_nonzero"), True)
    _expect("source gate Phase3 authorization", source_gate.get("required_authorizes_longer_training"), False)
    _expect("source checkpoint path", Path(str(source_checkpoint.get("path"))).resolve(), EXPECTED_SOURCE_CHECKPOINT)
    _expect("source checkpoint size", source_checkpoint.get("size_bytes"), 261681534)
    _expect("source checkpoint sha256", source_checkpoint.get("sha256"), EXPECTED_SOURCE_CHECKPOINT_SHA256)
    _expect("source checkpoint steps", source_checkpoint.get("optimizer_steps"), 4)
    _expect("source checkpoint kind", source_checkpoint.get("kind"), "lingbot-fast-v2-phase3-smoke-checkpoint/v1")

    base_path = Path(str(config.get("base_phase3_config", ""))).resolve()
    _expect("base Phase3 config path", base_path, EXPECTED_BASE_CONFIG)
    base = load_json(base_path)
    validate_phase3_config(base)
    _expect("base Phase3 config hash", sha256_json(base), config.get("base_phase3_config_sha256"))
    for key in ("learning_rate", "lora_learning_rate", "weight_decay", "max_gradient_norm"):
        _expect(f"Phase4 {key}", execution.get(key), base["execution"].get(key))
    _expect("Phase4 gradient accumulation", execution.get("gradient_accumulation_steps"), base["execution"].get("gradient_accumulation_steps"))
    return {
        "config_sha256": sha256_json(config),
        "base_phase3_config": base_path,
        "base_phase3_config_sha256": sha256_json(base),
    }


def phase4_microbatch_spec(
    *, optimizer_step: int, accumulation_microbatch: int, config: Mapping[str, Any]
) -> dict[str, Any]:
    if not 1 <= optimizer_step <= LOCAL_OPTIMIZER_STEPS:
        raise ValueError("Phase4 optimizer step must be in [1,25]")
    if accumulation_microbatch not in (0, 1):
        raise ValueError("Phase4 accumulation microbatch must be 0 or 1")
    local_index = (optimizer_step - 1) * 2 + accumulation_microbatch
    continued_index = SOURCE_OPTIMIZER_STEPS * 2 + local_index
    sources = ("w3", "tier69h", "event50h", "tier69h")
    target_chunks = (0, 1, 3, 4)
    timestep_indices = (0, 1, 2, 3)
    cycle_index = continued_index % len(sources)
    source = sources[cycle_index]
    source_ordinal = sum(
        sources[index % len(sources)] == source for index in range(continued_index)
    )
    target_chunk = target_chunks[cycle_index]
    timestep_index = timestep_indices[cycle_index]
    return {
        "global_microbatch_index": local_index,
        "continued_global_microbatch_index": continued_index,
        "optimizer_step": optimizer_step,
        "total_optimizer_step": SOURCE_OPTIMIZER_STEPS + optimizer_step,
        "accumulation_microbatch": accumulation_microbatch,
        "source": source,
        "source_ordinal": source_ordinal,
        "target_chunk": target_chunk,
        "timestep_index": timestep_index,
        "timestep": int(config["causal_runtime"]["selected_timesteps"][timestep_index]),
        "noise_seed": (
            int(config["execution"]["seed"])
            + 1000 * continued_index
            + 10 * target_chunk
            + timestep_index
        ),
    }


def validate_phase3_source_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    gate_contract = config["source_gate"]
    gate_path = Path(str(gate_contract["path"])).resolve()
    if not gate_path.is_file():
        raise Phase4ContractError(f"retry6 gate missing: {gate_path}")
    _expect("retry6 gate size", gate_path.stat().st_size, gate_contract["size_bytes"])
    _expect("retry6 gate sha256", sha256_file(gate_path), gate_contract["sha256"])
    gate = load_json(gate_path)
    for field, expected in (
        ("kind", gate_contract["kind"]),
        ("status", gate_contract["required_status"]),
        ("source_optimizer_steps", gate_contract["required_source_optimizer_steps"]),
        ("run_local_optimizer_steps", gate_contract["required_run_local_optimizer_steps"]),
        ("total_optimizer_steps", gate_contract["required_total_optimizer_steps"]),
        ("authorizes_longer_training", gate_contract["required_authorizes_longer_training"]),
    ):
        _expect(f"retry6 gate {field}", gate.get(field), expected)
    projector = gate.get("state_action_projector_gradient_gate")
    if not isinstance(projector, Mapping):
        raise Phase4ContractError("retry6 gate lacks projector gradient gate")
    _expect(
        "retry6 projector status",
        projector.get("status"),
        gate_contract["required_projector_gradient_status"],
    )
    per_step = projector.get("per_step")
    if not isinstance(per_step, list) or len(per_step) != 2:
        raise Phase4ContractError("retry6 projector gate must contain exactly two steps")
    for row in per_step:
        _expect("retry6 projector all-rank activation", row.get("all_ranks_nonzero"), True)
        norms = row.get("gradient_norms_by_rank")
        if not isinstance(norms, list) or len(norms) != 8 or any(
            not math.isfinite(float(value)) or float(value) <= 0.0 for value in norms
        ):
            raise Phase4ContractError("retry6 projector norms must be finite positive on 8 ranks")
    checkpoint = config["source_checkpoint"]
    checkpoint_path = Path(str(checkpoint["path"])).resolve()
    _expect("retry6 gate checkpoint path", Path(str(gate.get("checkpoint"))).resolve(), checkpoint_path)
    _expect("retry6 gate checkpoint sha256", gate.get("checkpoint_sha256"), checkpoint["sha256"])
    return {
        "status": "pass",
        "path": str(gate_path),
        "sha256": gate_contract["sha256"],
        "projector_all_steps_all_ranks_nonzero": True,
        "phase3_authorizes_longer_training": False,
    }


def validate_authorization_and_checkpoint(config: Mapping[str, Any]) -> dict[str, Any]:
    authorization_contract = config["authorization"]
    authorization_path = Path(str(authorization_contract["path"])).resolve()
    _expect("authorization sha256", sha256_file(authorization_path), authorization_contract["sha256"])
    authorization = load_json(authorization_path)
    _expect("authorization schema", authorization.get("schema_version"), AUTHORIZATION_SCHEMA)
    _expect("authorized local steps", authorization.get("authorized_local_optimizer_steps"), 25)
    _expect("authorization source steps", authorization.get("source_total_optimizer_steps"), 4)
    _expect("authorization hard stop", authorization.get("hard_stop_total_optimizer_steps"), 29)
    _expect("authorization single run", authorization.get("single_run_only"), True)
    _expect("authorization automatic extension", authorization.get("automatic_extension_forbidden"), True)
    checkpoint = config["source_checkpoint"]
    checkpoint_path = Path(str(checkpoint["path"])).resolve()
    if not checkpoint_path.is_file():
        raise Phase4ContractError(f"source checkpoint missing: {checkpoint_path}")
    _expect("source checkpoint size", checkpoint_path.stat().st_size, checkpoint["size_bytes"])
    _expect("source checkpoint sha256", sha256_file(checkpoint_path), checkpoint["sha256"])
    _expect("authorization checkpoint path", Path(str(authorization["source_checkpoint"])).resolve(), checkpoint_path)
    _expect("authorization checkpoint sha256", authorization.get("source_checkpoint_sha256"), checkpoint["sha256"])
    return {
        "status": "pass",
        "path": str(authorization_path),
        "sha256": authorization_contract["sha256"],
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_size_bytes": checkpoint["size_bytes"],
        "source_checkpoint_sha256": checkpoint["sha256"],
    }


def build_phase4_preflight(config_path: Path) -> dict[str, Any]:
    config = load_json(config_path)
    validation = validate_phase4_config(config)
    source_gate = validate_phase3_source_gate(config)
    authorization = validate_authorization_and_checkpoint(config)
    base_preflight = build_strict_preflight(validation["base_phase3_config"])
    _expect("base preflight status", base_preflight.get("status"), "pass")
    return {
        "kind": PREFLIGHT_KIND,
        "status": "pass",
        "mode": "phase4-25step-preflight",
        "created_at": utc_now(),
        "config": str(config_path.resolve()),
        "config_sha256": validation["config_sha256"],
        "authorization": authorization,
        "retry6_source_gate": source_gate,
        "base_phase3_preflight": base_preflight,
        "world_size": 8,
        "sequence_parallel_size": 8,
        "source_optimizer_steps": 4,
        "authorized_local_optimizer_steps": 25,
        "hard_stop_total_optimizer_steps": 29,
        "model_loaded": False,
        "cuda_initialized": False,
        "distributed_initialized": False,
        "phase3_gate_semantics_unchanged": True,
        "phase3_gate_authorizes_longer_training": False,
        "authorizes_only_this_fixed_phase4_run": True,
        "authorizes_automatic_extension": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validation-only", action="store_true")
    modes.add_argument("--execute-phase4", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--authorized-optimizer-steps", type=int, default=None)
    parser.add_argument("--run-token", default=None)
    parser.add_argument("--preflight-wait-seconds", type=int, default=7200)
    args = parser.parse_args(argv)
    args.config = args.config.resolve()
    args.report = args.report.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.execute_phase4:
        if args.authorized_optimizer_steps != 25:
            parser.error("--execute-phase4 requires --authorized-optimizer-steps 25")
        if args.out_dir != DEFAULT_OUT_DIR:
            parser.error(f"Phase4 output directory is fixed to {DEFAULT_OUT_DIR}")
    elif args.authorized_optimizer_steps is not None:
        parser.error("--authorized-optimizer-steps is valid only with --execute-phase4")
    if args.preflight_wait_seconds < 60:
        parser.error("--preflight-wait-seconds must be at least 60")
    return args


def execute_phase4(args: argparse.Namespace, phase4_config: Mapping[str, Any]) -> None:
    validation = validate_phase4_config(phase4_config)
    base_config = load_json(validation["base_phase3_config"])
    source = phase4_config["source_checkpoint"]
    args.resume_checkpoint = Path(str(source["path"])).resolve()
    args.resume_checkpoint_size_bytes = int(source["size_bytes"])
    args.resume_checkpoint_sha256 = str(source["sha256"])
    args.resume_checkpoint_optimizer_steps = int(source["optimizer_steps"])
    contract = {
        "execute_flag": "--execute-phase4",
        "local_optimizer_steps": LOCAL_OPTIMIZER_STEPS,
        "config_path": args.config,
        "config_sha256": validation["config_sha256"],
        "source_checkpoint_config_sha256": validation["base_phase3_config_sha256"],
        "build_preflight": lambda: build_phase4_preflight(args.config),
        "preflight_filename": phase4_config["output"]["preflight_filename"],
        "microbatch_spec": phase4_microbatch_spec,
        "expected_source_counts": phase4_config["microbatch_schedule"]["realized_source_counts"],
        "checkpoint_kind": CHECKPOINT_KIND,
        "checkpoint_smoke_only": False,
        "checkpoint_filename": "lingbot_fast_v2_phase4_step_{total_optimizer_steps:06d}_compact.pt",
        "gate_kind": GATE_KIND,
        "gate_filename": phase4_config["output"]["gate_filename"],
        "projector_must_be_finite_nonzero_each_step": True,
        "continuation_policy": (
            "This gate covers exactly 25 local optimizer steps ending at total step 29; "
            "it cannot authorize automatic extension."
        ),
        "failure_prefix": "Fast v2 Phase 4 25-step gate failed: ",
    }
    _execute_fixed_training(args, base_config, contract)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_json(args.config)
    validate_phase4_config(config)
    if args.execute_phase4:
        try:
            execute_phase4(args, config)
        except Exception as exc:
            best_effort_destroy_distributed()
            gate_path = args.out_dir / str(config["output"]["gate_filename"])
            if int(os.environ.get("RANK", "0")) == 0 and args.out_dir.is_dir() and not gate_path.is_file():
                atomic_json(gate_path, {
                    "kind": GATE_KIND,
                    "status": "quality_fail" if isinstance(exc, TrainingQualityError) else "fail",
                    "failed_at": utc_now(),
                    "config": str(args.config),
                    "config_sha256": sha256_json(config),
                    "error": f"{type(exc).__name__}: {exc}",
                    "source_optimizer_steps": 4,
                    "authorized_local_optimizer_steps": 25,
                    "hard_stop_total_optimizer_steps": 29,
                    "authorizes_longer_training": False,
                    "authorizes_automatic_extension": False,
                })
            raise
        return
    report = build_phase4_preflight(args.config)
    atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
