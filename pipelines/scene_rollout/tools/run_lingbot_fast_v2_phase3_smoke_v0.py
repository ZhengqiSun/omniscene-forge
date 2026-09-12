#!/usr/bin/env python3
"""Gated real-data two-step smoke trainer for LingBot Fast v2 direct tuning.

The default and ``--validation-only`` paths are CPU-only: they do not import
official Wan model code, initialize CUDA/distributed state, or load weights.
Only ``--execute-smoke`` may cross that boundary, after rank zero publishes a
strict hash-bound preflight and only when torchrun reports WORLD_SIZE=8.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Iterable, Iterator, Mapping


TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(TOOLS_DIR))

CONFIG_SCHEMA = "lingbot-fast-v2-phase3-smoke-config/v1"
PREFLIGHT_KIND = "lingbot-fast-v2-phase3-preflight/v1"
CHECKPOINT_KIND = "lingbot-fast-v2-phase3-smoke-checkpoint/v1"
GATE_KIND = "lingbot-fast-v2-phase3-smoke-gate/v1"
CHECKPOINT_NAMESPACE = "lingbot_fast_v2_direct"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/lingbot_fast_v2_phase3_smoke_v0.json"
DEFAULT_VALIDATION_REPORT = PROJECT_ROOT / "output/lingbot_fast_v2_phase3_validation_v0.json"
DEFAULT_SMOKE_DIR = PROJECT_ROOT / "output/lingbot_fast_v2_phase3_two_step_smoke_v0"
RUN_LOCAL_OPTIMIZER_STEPS = 2
INJECTION_CHECKPOINT_KIND = "lingbot-fast-v2-direct-finetune-checkpoint/v1"
INJECTION_SCHEMA_VERSION = "lingbot-fast-v2-native-injection/v1"
FSDP_WRAPPED_MODULE_SEGMENT = "_fsdp_wrapped_module"
PROJECTOR_STATE_KEYS = frozenset(
    {"fast_v2_state_proj.weight", "fast_v2_state_proj.bias"}
)


class Phase3ContractError(RuntimeError):
    """Raised before model load when a fixed Phase 3 contract is violated."""


class TrainingQualityError(RuntimeError):
    """Raised when a fixed training run must stop on a quality invariant."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def emit_rank_stage(rank: int, stage: str, **fields: Any) -> None:
    """Emit sparse, flush-safe markers for distributed hang diagnosis."""

    print(
        json.dumps(
            {"phase3_rank_stage": stage, "rank": rank, **fields},
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )


def best_effort_destroy_distributed() -> bool:
    """Destroy an already initialized group without importing torch on safe paths."""

    dist = sys.modules.get("torch.distributed")
    if dist is None:
        return False
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
            return True
    except Exception:
        return False
    return False


def run_collective_autograd_self_test(
    torch: Any,
    dist: Any,
    sequence_parallel: Any,
    *,
    rank: int,
    world: int,
    device: Any,
) -> dict[str, Any]:
    """Prove both sequence-parallel collectives preserve gradients."""

    if world != 8:
        raise Phase3ContractError(f"collective autograd self-test requires world=8, got {world}")
    with torch.enable_grad():
        value = (
            torch.arange(1, 1 + 8 * 8 * 2, device=device, dtype=torch.float32)
            .reshape(1, 8, 8, 2)
            .add_(rank * 1000)
            .requires_grad_(True)
        )
        exchanged = sequence_parallel.all_to_all(value, scatter_dim=2, gather_dim=1)
        gathered = sequence_parallel.gather_forward(exchanged, dim=1)
        if tuple(exchanged.shape) != (1, 64, 1, 2):
            raise RuntimeError(f"differentiable all_to_all shape drifted: {tuple(exchanged.shape)}")
        if tuple(gathered.shape) != (1, 512, 1, 2):
            raise RuntimeError(f"differentiable gather shape drifted: {tuple(gathered.shape)}")
        if not gathered.requires_grad or gathered.grad_fn is None:
            raise RuntimeError("sequence-parallel collective output detached from autograd")
        if not bool(gathered.isfinite().all().item()):
            raise RuntimeError("sequence-parallel collective output is non-finite")
        loss = gathered.square().mean()
        if not loss.requires_grad:
            raise RuntimeError("sequence-parallel collective self-test loss has no grad graph")
        loss.backward()
    if value.grad is None or not bool(value.grad.isfinite().all().item()):
        raise RuntimeError("sequence-parallel collective backward produced missing/non-finite grad")
    gradient_nonzero = int(value.grad.count_nonzero().item())
    if gradient_nonzero == 0:
        raise RuntimeError("sequence-parallel collective backward produced zero grad")
    checksum = torch.tensor(
        [gathered.detach().double().sum().item()], device=device, dtype=torch.float64
    )
    rank_checksums = [torch.zeros_like(checksum) for _ in range(world)]
    dist.all_gather(rank_checksums, checksum)
    checksum_values = [float(item.item()) for item in rank_checksums]
    if any(value != checksum_values[0] for value in checksum_values[1:]):
        raise RuntimeError(f"sequence-parallel gathered outputs differ by rank: {checksum_values}")
    report = {
        "status": "pass",
        "world_size": world,
        "input_shape": [1, 8, 8, 2],
        "all_to_all_shape": [1, 64, 1, 2],
        "gather_shape": [1, 512, 1, 2],
        "output_requires_grad": True,
        "gradient_finite": True,
        "gradient_nonzero_elements_per_rank": gradient_nonzero,
        "gathered_checksum_all_ranks": checksum_values,
    }
    del value, exchanged, gathered, loss, checksum, rank_checksums
    return report


def run_fsdp_sp_autograd_self_test(
    torch: Any,
    dist: Any,
    fsdp_class: Any,
    sequence_parallel: Any,
    *,
    rank: int,
    world: int,
    local_rank: int,
    device: Any,
) -> dict[str, Any]:
    """Exercise FSDP and SP backward together before loading the 14B model."""

    class TinyFSDPSequenceParallel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = torch.nn.Linear(16, 16)
            self.output_projection = torch.nn.Linear(16, 16)

        def forward(self, value: Any) -> Any:
            hidden = torch.nn.functional.silu(self.input_projection(value))
            hidden = sequence_parallel.all_to_all(
                hidden, scatter_dim=2, gather_dim=1
            )
            hidden = self.output_projection(hidden)
            return sequence_parallel.gather_forward(hidden, dim=1)

    torch.manual_seed(314159)
    tiny = TinyFSDPSequenceParallel().to(device=device, dtype=torch.float32)
    wrapped = fsdp_class(
        tiny,
        device_id=local_rank,
        use_orig_params=True,
        sync_module_states=True,
    )
    optimizer = torch.optim.SGD(wrapped.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    output_checksums: list[float] = []
    for microbatch in range(2):
        value = (
            torch.arange(1, 1 + 8 * 8 * 16, device=device, dtype=torch.float32)
            .reshape(1, 8, 8, 16)
            .add(rank * 1000 + microbatch * 10)
        )
        with ExitStack() as stack:
            if microbatch == 0:
                stack.enter_context(wrapped.no_sync())
            output = wrapped(value)
            if tuple(output.shape) != (1, 512, 1, 16):
                raise RuntimeError(f"tiny FSDP+SP output shape drifted: {tuple(output.shape)}")
            if not output.requires_grad or output.grad_fn is None:
                raise RuntimeError("tiny FSDP+SP output detached from autograd")
            loss = output.square().mean() / 2.0
            if not loss.requires_grad or not bool(loss.isfinite().item()):
                raise RuntimeError("tiny FSDP+SP loss is detached or non-finite")
            loss.backward()
        losses.append(float(loss.detach().cpu()))
        output_checksums.append(float(output.detach().double().sum().cpu()))
    gradients = [parameter.grad for parameter in wrapped.parameters() if parameter.grad is not None]
    if not gradients or any(not bool(gradient.isfinite().all().item()) for gradient in gradients):
        raise RuntimeError("tiny FSDP+SP gradients are missing or non-finite")
    gradient_nonzero = sum(int(gradient.count_nonzero().item()) for gradient in gradients)
    if gradient_nonzero == 0:
        raise RuntimeError("tiny FSDP+SP gradients are all zero")
    optimizer.step()
    checksum = torch.tensor(output_checksums, device=device, dtype=torch.float64)
    rank_checksums = [torch.zeros_like(checksum) for _ in range(world)]
    dist.all_gather(rank_checksums, checksum)
    checksum_rows = [[float(value) for value in row.cpu().tolist()] for row in rank_checksums]
    if any(row != checksum_rows[0] for row in checksum_rows[1:]):
        raise RuntimeError(f"tiny FSDP+SP outputs differ by rank: {checksum_rows}")
    dist.barrier()
    report = {
        "status": "pass",
        "world_size": world,
        "fsdp_process_group": "default",
        "sequence_parallel_process_group": "dedicated_same_membership",
        "gradient_accumulation_microbatches": 2,
        "synchronized_backward_pattern": [False, True],
        "losses": losses,
        "gradient_tensor_count": len(gradients),
        "gradient_nonzero_elements_per_rank": gradient_nonzero,
        "output_checksums_all_ranks": checksum_rows,
    }
    del optimizer, wrapped, tiny, value, output, loss, gradients, checksum, rank_checksums
    torch.cuda.empty_cache()
    return report


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def build_resume_checkpoint_binding(
    args: argparse.Namespace,
    *,
    run_local_optimizer_steps: int = RUN_LOCAL_OPTIMIZER_STEPS,
) -> dict[str, Any]:
    """Bind an explicitly supplied resume file before any model code loads."""

    if args.resume_checkpoint is None:
        return {
            "enabled": False,
            "source_checkpoint": None,
            "source_checkpoint_size_bytes": None,
            "source_checkpoint_sha256": None,
            "source_optimizer_steps": 0,
            "run_local_optimizer_steps": run_local_optimizer_steps,
            "total_optimizer_steps": run_local_optimizer_steps,
        }
    checkpoint = args.resume_checkpoint
    if not checkpoint.is_absolute():
        raise Phase3ContractError("resume checkpoint path must be absolute")
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise Phase3ContractError(f"resume checkpoint is not a file: {checkpoint}")
    actual_size = checkpoint.stat().st_size
    if actual_size != args.resume_checkpoint_size_bytes:
        raise Phase3ContractError(
            "resume checkpoint size mismatch: "
            f"expected={args.resume_checkpoint_size_bytes} actual={actual_size}"
        )
    actual_sha256 = sha256_file(checkpoint)
    if actual_sha256 != args.resume_checkpoint_sha256:
        raise Phase3ContractError(
            "resume checkpoint sha256 mismatch: "
            f"expected={args.resume_checkpoint_sha256} actual={actual_sha256}"
        )
    source_steps = int(args.resume_checkpoint_optimizer_steps)
    return {
        "enabled": True,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_size_bytes": actual_size,
        "source_checkpoint_sha256": actual_sha256,
        "source_optimizer_steps": source_steps,
        "run_local_optimizer_steps": run_local_optimizer_steps,
        "total_optimizer_steps": source_steps + run_local_optimizer_steps,
    }


def json_safe(value: Any) -> Any:
    """Normalize nested contract helper results to report-safe JSON values."""

    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase3ContractError(f"cannot load JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Phase3ContractError(f"JSON root must be an object: {path}")
    return value


def resolve_path(config: Mapping[str, Any], value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(str(config["project_root"])) / path).resolve()


def _expect(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise Phase3ContractError(f"{name} must be {expected!r}, got {actual!r}")


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on every non-negotiable smoke setting."""

    _expect("config schema", config.get("schema_version"), CONFIG_SCHEMA)
    for key in ("official", "data", "execution", "causal_runtime", "injection", "output"):
        if not isinstance(config.get(key), Mapping):
            raise Phase3ContractError(f"config.{key} must be an object")
    official = config["official"]
    data = config["data"]
    execution = config["execution"]
    causal = config["causal_runtime"]
    injection = config["injection"]
    output = config["output"]

    _expect("official source revision", official.get("source_revision"), "2648877f763a06cc743bcd919936da4d25f12e7b")
    _expect("total records", data.get("total_records"), 129342)
    _expect("W3 fraction", float(data.get("w3_fraction")), 0.25)
    _expect("general fraction", float(data.get("general_fraction")), 0.75)
    _expect("train split", data.get("train_split"), "train")
    _expect("cached latent frames", data.get("latent_cache_frames"), 21)
    _expect("usable latent frames", data.get("usable_latent_frames"), 20)
    releases = data.get("releases")
    if not isinstance(releases, list) or [item.get("name") for item in releases] != ["w3", "tier69h", "event50h"]:
        raise Phase3ContractError("release order must be W3, tier69h, event50h")
    _expect("release counts", [item.get("records") for item in releases], [18000, 46552, 64790])
    _expect("release count sum", sum(int(item["records"]) for item in releases), 129342)

    _expect("world size", execution.get("world_size"), 8)
    _expect("sequence parallel size", execution.get("sequence_parallel_size"), 8)
    _expect("optimizer steps", execution.get("optimizer_steps"), 2)
    _expect("gradient accumulation", execution.get("gradient_accumulation_steps"), 2)
    _expect("batch size per rank", execution.get("batch_size_per_rank"), 1)
    _expect(
        "global microbatch sources",
        execution.get("global_microbatch_sources"),
        ["w3", "tier69h", "event50h", "tier69h"],
    )
    _expect("global microbatch target chunks", execution.get("global_microbatch_target_chunks"), [0, 1, 3, 4])
    _expect("global microbatch timestep indices", execution.get("global_microbatch_timestep_indices"), [0, 1, 2, 3])
    _expect("precision", execution.get("precision"), "bfloat16")
    if not isinstance(execution.get("seed"), int):
        raise Phase3ContractError("execution seed must be an integer")

    _expect("timestep indices", causal.get("timestep_indices"), [0, 250, 500, 750])
    _expect("selected timesteps", causal.get("selected_timesteps"), [999, 967, 908, 768])
    _expect("shift", float(causal.get("shift")), 10.0)
    _expect("chunk size", causal.get("chunk_size"), 4)
    _expect("chunk count", causal.get("chunk_count"), 5)
    _expect("local attention", causal.get("local_attention_frames"), 18)
    _expect("sink", causal.get("sink_frames"), 6)
    _expect("clean t0", causal.get("clean_commit_timestep"), 0)
    _expect("mutable KV", causal.get("mutable_kv_cache"), True)

    _expect("LoRA rank", injection.get("lora_rank"), 16)
    _expect("LoRA alpha", float(injection.get("lora_alpha")), 16.0)
    _expect("LoRA dropout", float(injection.get("lora_dropout")), 0.0)
    _expect(
        "LoRA targets",
        injection.get("lora_targets"),
        [
            "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
            "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
            "ffn.0", "ffn.2",
        ],
    )
    _expect("adapter hidden dim", injection.get("adapter_hidden_dim"), 128)
    _expect("adapter block coverage", injection.get("wrap_first_blocks"), None)
    _expect("adapter residual scale", float(injection.get("residual_scale_init")), 1.0)
    _expect("external checkpoint", injection.get("external_checkpoint"), None)
    _expect("LOW/HIGH load", injection.get("load_or_merge_low_high_weights"), False)
    _expect("condition dim", injection.get("cond_dim"), 128)
    _expect("state/action projector", injection.get("state_action_projector"), "lingbot_fast_v2_state_action_projector/v1")

    _expect("checkpoint kind", output.get("checkpoint_kind"), CHECKPOINT_KIND)
    _expect("gate kind", output.get("gate_report_kind"), GATE_KIND)
    _expect("longer training authorization", output.get("authorizes_longer_training"), False)
    _expect("maximum authorized steps", output.get("maximum_optimizer_steps_authorized"), 2)
    return {"config_sha256": sha256_json(config), "release_names": [item["name"] for item in releases]}


def global_microbatch_spec(
    *, optimizer_step: int, accumulation_microbatch: int, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the rank-independent contract for one global SP microbatch."""

    grad_accum = int(config["execution"]["gradient_accumulation_steps"])
    if optimizer_step not in (1, 2) or not 0 <= accumulation_microbatch < grad_accum:
        raise ValueError("Phase 3 has exactly two steps with two accumulation microbatches")
    global_index = (optimizer_step - 1) * grad_accum + accumulation_microbatch
    sources = config["execution"]["global_microbatch_sources"]
    target_chunks = config["execution"]["global_microbatch_target_chunks"]
    timestep_indices = config["execution"]["global_microbatch_timestep_indices"]
    source = str(sources[global_index])
    source_ordinal = sum(1 for previous in sources[:global_index] if previous == source)
    target_chunk = int(target_chunks[global_index])
    timestep_index = int(timestep_indices[global_index])
    return {
        "global_microbatch_index": global_index,
        "optimizer_step": optimizer_step,
        "accumulation_microbatch": accumulation_microbatch,
        "source": source,
        "source_ordinal": source_ordinal,
        "target_chunk": target_chunk,
        "timestep_index": timestep_index,
        "timestep": int(config["causal_runtime"]["selected_timesteps"][timestep_index]),
        "noise_seed": int(config["execution"]["seed"]) + 1000 * global_index + 10 * target_chunk + timestep_index,
    }


def should_sync_microbatch(*, accumulation_microbatch: int, config: Mapping[str, Any]) -> bool:
    return accumulation_microbatch == int(config["execution"]["gradient_accumulation_steps"]) - 1


def microbatch_forward_budget(spec: Mapping[str, Any]) -> dict[str, int]:
    prefix_commits = int(spec["target_chunk"])
    return {
        "prefix_clean_commits": prefix_commits,
        "train_forwards": 1,
        "target_clean_commits": 1,
        "backward_calls": 1,
        "total_forwards": prefix_commits + 2,
    }


def validate_text_context_shape(shape: Iterable[int]) -> tuple[int, int]:
    values = tuple(int(value) for value in shape)
    if len(values) != 2 or values[1] != 4096 or not 1 <= values[0] <= 512:
        raise Phase3ContractError(f"text context must be [L,4096] with 1<=L<=512, got {values}")
    return values


def assert_consensus_summaries(
    summaries: Iterable[Mapping[str, Any]], *, expected_replicas: int = 8
) -> dict[str, Any]:
    """Fail unless all SP ranks report one identical global-microbatch input."""

    rows = [dict(row) for row in summaries]
    if len(rows) != expected_replicas:
        raise Phase3ContractError(f"SP consensus expected {expected_replicas} replicas, got {len(rows)}")
    canonical = rows[0]
    mismatched = [index for index, row in enumerate(rows) if row != canonical]
    if mismatched:
        raise Phase3ContractError(f"SP rank input mismatch at replicas {mismatched}")
    return {
        "replica_count": expected_replicas,
        "all_ranks_identical": True,
        "shared": canonical,
    }


def collapse_rank_microbatch_reports(
    rank_reports: Iterable[Mapping[str, Any]],
    *,
    expected_world: int = 8,
    expected_microbatches: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collapse eight SP replicas into exactly four global microbatch rows."""

    reports = [dict(report) for report in rank_reports]
    if len(reports) != expected_world:
        raise Phase3ContractError(f"gate expected {expected_world} rank reports, got {len(reports)}")
    by_global: dict[int, list[dict[str, Any]]] = {
        index: [] for index in range(expected_microbatches)
    }
    for report in reports:
        rows = report.get("microbatches")
        if not isinstance(rows, list) or len(rows) != expected_microbatches:
            raise Phase3ContractError(
                f"each SP rank must report exactly {expected_microbatches} microbatches"
            )
        for row in rows:
            by_global[int(row["global_microbatch_index"])].append(dict(row))
    global_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    identity_keys = (
        "global_microbatch_index", "optimizer_step", "accumulation_microbatch",
        "source", "source_ordinal", "clip_id", "target_chunk", "timestep", "noise_seed",
        "prefix_clean_commits", "train_forwards", "target_clean_commits",
        "backward_calls", "total_forwards", "synchronized_backward",
        "cache_indices_after_target_commit",
    )
    for global_index in range(expected_microbatches):
        replicas = by_global[global_index]
        identities = [{key: row.get(key) for key in identity_keys} for row in replicas]
        audit = assert_consensus_summaries(identities, expected_replicas=expected_world)
        losses = [float(row["loss"]) for row in replicas]
        if not all(math.isfinite(loss) for loss in losses):
            raise Phase3ContractError(f"global microbatch {global_index} has non-finite rank loss")
        shared = dict(replicas[0])
        shared["rank_replica_count"] = expected_world
        shared["rank_replicas_count_as_samples"] = False
        shared["rank_loss_min"] = min(losses)
        shared["rank_loss_max"] = max(losses)
        global_rows.append(shared)
        audits.append({"global_microbatch_index": global_index, **audit})
    return global_rows, audits


def expected_cache_indices_after_chunk(
    chunk_index: int, *, frame_seqlen: int = 1560, local_attention_frames: int = 18
) -> dict[str, int]:
    if not 0 <= chunk_index < 5:
        raise ValueError("chunk index must be in [0,4]")
    global_end = (chunk_index + 1) * 4 * frame_seqlen
    return {
        "global_end_index": global_end,
        "local_end_index": min(global_end, local_attention_frames * frame_seqlen),
    }


def smoke_mix_schedule(config: Mapping[str, Any]) -> dict[str, Any]:
    specs = [
        global_microbatch_spec(optimizer_step=step, accumulation_microbatch=micro, config=config)
        for step in (1, 2)
        for micro in range(2)
    ]
    sources = [spec["source"] for spec in specs]
    counts = {name: sources.count(name) for name in ("w3", "tier69h", "event50h")}
    if counts != {"w3": 1, "tier69h": 2, "event50h": 1}:
        raise Phase3ContractError(f"two-step sampler no longer realizes exact 25/75 mix: {counts}")
    return {
        "global_microbatches": 4,
        "source_counts": counts,
        "w3_fraction": counts["w3"] / 4,
        "general_fraction": (counts["tier69h"] + counts["event50h"]) / 4,
        "per_step": [sources[:2], sources[2:]],
        "microbatches": [{**spec, "forward_budget": microbatch_forward_budget(spec)} for spec in specs],
        "sp_replica_count_per_global_microbatch": 8,
        "rank_replicas_count_as_samples": False,
    }


def _validate_git_binding(source_root: Path, revision: str) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(source_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _expect("official source HEAD", head, revision)
    if status.strip():
        raise Phase3ContractError("official LingBot source worktree is not clean")
    return {"head": head, "clean": True}


def _verify_bound_file(path: Path, expected_hash: str, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise Phase3ContractError(f"missing or empty {label}: {path}")
    actual = sha256_file(path)
    _expect(f"{label} sha256", actual, expected_hash)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}


def validate_real_data_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate all real-data release gates without importing model code."""

    data = config["data"]
    ready_path = resolve_path(config, str(data["ready_report"]))
    ready = load_json(ready_path)
    _expect("ready kind", ready.get("kind"), "w3_zero_training_inputs_ready_no_train_v0")
    _expect("ready status", ready.get("status"), "pass")
    _expect("ready training_started", ready.get("training_started"), False)

    ready_bindings: dict[str, Any] = {}
    for path_key, hash_key in (
        ("aligned_manifest", "aligned_manifest_sha256"),
        ("aligned_report", "aligned_report_sha256"),
        ("w3_state_manifest", "w3_state_manifest_sha256"),
        ("w3_state_report", "w3_state_report_sha256"),
        ("combined_state_manifest", "combined_state_manifest_sha256"),
        ("combined_state_report", "combined_state_report_sha256"),
        ("schedule_report", "schedule_report_sha256"),
        ("preflight_report", "preflight_report_sha256"),
        ("training_index", "training_index_sha256"),
    ):
        ready_bindings[path_key] = _verify_bound_file(
            Path(str(ready[path_key])).resolve(), str(ready[hash_key]), label=f"ready.{path_key}"
        )

    _expect("configured preflight path", resolve_path(config, str(data["preflight_report"])), Path(str(ready["preflight_report"])).resolve())
    _expect("configured index path", resolve_path(config, str(data["training_index"])), Path(str(ready["training_index"])).resolve())
    _expect("configured state path", resolve_path(config, str(data["state_manifest"])), Path(str(ready["combined_state_manifest"])).resolve())

    preflight = load_json(Path(str(ready["preflight_report"])))
    _expect("legacy preflight status", preflight.get("status"), "pass")
    _expect("legacy preflight records", preflight.get("record_count"), 129342)
    _expect("legacy preflight optimizer steps", preflight.get("optimizer_steps_run"), 0)
    _expect("legacy preflight checked records", preflight.get("checked_record_count"), 8)
    _expect("legacy preflight release count", preflight.get("release_count"), 3)
    release_gate = preflight.get("release_gate", {})
    _expect("legacy release gate status", release_gate.get("total_train_record_count"), 129342)
    _expect("legacy release counts", [item.get("train_record_count") for item in release_gate.get("releases", [])], [18000, 46552, 64790])
    state_gate = preflight.get("state_channels_v0", {})
    _expect("legacy state enabled", state_gate.get("enabled"), True)
    checked = preflight.get("checked_records", [])
    if len(checked) != 8 or not all(item.get("latent_shape") == [16, 21, 60, 104] for item in checked):
        raise Phase3ContractError("legacy preflight latent probes drifted")
    if not all(item.get("condition_shape") == [20, 21, 60, 104] for item in checked):
        raise Phase3ContractError("legacy preflight image-condition probes drifted")
    if not all(item.get("state_token_shape") == [1, 32760, 128] for item in checked):
        raise Phase3ContractError("legacy preflight state probes drifted")

    release_bindings = []
    index_inputs = preflight.get("training_index_cache", {}).get("contract", {}).get("inputs", [])
    for release, index_item in zip(data["releases"], index_inputs):
        map_path = resolve_path(config, str(release["map_manifest"]))
        cache_path = resolve_path(config, str(release["cache_manifest"]))
        _expect(f"{release['name']} index map path", Path(str(index_item.get("map_manifest"))).resolve(), map_path)
        _expect(f"{release['name']} index cache path", Path(str(index_item.get("cache_manifest"))).resolve(), cache_path)
        _expect(f"{release['name']} index map hash", index_item.get("map_manifest_sha256"), release["map_manifest_sha256"])
        _expect(f"{release['name']} index cache hash", index_item.get("cache_manifest_sha256"), release["cache_manifest_sha256"])
        release_bindings.append({
            "name": release["name"],
            "records": release["records"],
            "map": _verify_bound_file(map_path, release["map_manifest_sha256"], label=f"{release['name']} map manifest"),
            "cache": _verify_bound_file(cache_path, release["cache_manifest_sha256"], label=f"{release['name']} cache manifest"),
        })
    if len(index_inputs) != 3:
        raise Phase3ContractError("training index must bind exactly three releases")

    schedule = load_json(Path(str(ready["schedule_report"])))
    _expect("existing W3 schedule status", schedule.get("status"), "pass")
    _expect("existing W3 schedule records", schedule.get("w3_train_records"), 18000)
    _expect("existing general schedule records", schedule.get("old_train_records"), 111342)
    return {
        "ready_report": str(ready_path),
        "ready_report_sha256": sha256_file(ready_path),
        "ready_bindings": ready_bindings,
        "release_bindings": release_bindings,
        "legacy_preflight": {
            "path": str(ready["preflight_report"]),
            "record_count": 129342,
            "optimizer_steps_run": 0,
            "checked_record_count": 8,
        },
    }


def build_strict_preflight(config_path: Path) -> dict[str, Any]:
    """Run every CPU-only gate required before the official model may load."""

    config_path = config_path.resolve(strict=True)
    config = load_json(config_path)
    structure = validate_config(config)
    official = config["official"]
    source_root = Path(str(official["source_root"])).resolve(strict=True)
    model_root = Path(str(official["model_root"])).resolve(strict=True)
    git_binding = _validate_git_binding(source_root, str(official["source_revision"]))

    from run_lingbot_fast_v2_zero_init_parity_v0 import (
        load_manifest as load_parity_manifest,
        validate_manifest_structure,
        validate_official_contract_and_model,
    )

    parity_path = resolve_path(config, str(official["parity_fixture"]))
    parity_manifest = load_parity_manifest(parity_path)
    parity_structure = validate_manifest_structure(parity_manifest)
    official_binding = json_safe(
        validate_official_contract_and_model(parity_manifest, source_root, model_root)
    )
    _expect("parity selected timesteps", parity_structure["selected_timesteps"], config["causal_runtime"]["selected_timesteps"])
    data_binding = validate_real_data_contract(config)
    mix = smoke_mix_schedule(config)
    report = {
        "kind": PREFLIGHT_KIND,
        "status": "pass",
        "mode": "validation-only",
        "created_at": utc_now(),
        "config": str(config_path),
        "config_sha256": structure["config_sha256"],
        "official_git": git_binding,
        "official_binding": official_binding,
        "parity_fixture": str(parity_path),
        "parity_fixture_sha256": sha256_file(parity_path),
        "data": data_binding,
        "smoke_mix": mix,
        "optimizer_steps_run": 0,
        "model_loaded": False,
        "cuda_initialized": False,
        "distributed_initialized": False,
        "authorizes_longer_training": False,
        "maximum_optimizer_steps_authorized": 2,
        "next_action": "This report permits only the explicit two-step smoke path; it cannot authorize continuation or longer training.",
    }
    return report


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"non-object JSONL row in {path}")
                yield value


def iter_json_array(path: Path, key: str = "samples") -> Iterator[dict[str, Any]]:
    """Stream objects from one top-level JSON array without loading its manifest."""

    decoder = json.JSONDecoder()
    marker = f'"{key}"'
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                raise ValueError(f"{path}: missing top-level array {key!r}")
            buffer += chunk
            marker_at = buffer.find(marker)
            if marker_at < 0:
                buffer = buffer[-len(marker) :]
                continue
            array_at = buffer.find("[", marker_at + len(marker))
            if array_at < 0:
                continue
            buffer = buffer[array_at + 1 :]
            break
        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                return
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    raise ValueError(f"{path}: truncated array {key!r}")
                buffer += chunk
                continue
            if not isinstance(value, dict):
                raise ValueError(f"{path}: {key} item is not an object")
            yield value
            buffer = buffer[end:]


def select_record(config: Mapping[str, Any], *, source: str, ordinal: int) -> tuple[dict[str, Any], Mapping[str, Any]]:
    release = next(item for item in config["data"]["releases"] if item["name"] == source)
    path = resolve_path(config, str(release["cache_manifest"]))
    selected = 0
    for row in iter_jsonl(path):
        if row.get("map_memory_split") != "train":
            continue
        roles = row.get("map_memory_selection_roles")
        if source != "w3" and (not isinstance(roles, list) or "positive" not in roles):
            continue
        if selected == ordinal:
            return row, release
        selected += 1
    raise Phase3ContractError(f"{source} manifest has no eligible record ordinal {ordinal}")


def load_selected_samples(config: Mapping[str, Any], release: Mapping[str, Any], sample_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    needed = set(sample_ids)
    found: dict[str, dict[str, Any]] = {}
    for sample in iter_json_array(resolve_path(config, str(release["map_manifest"]))):
        sample_id = str(sample.get("sample_id", ""))
        if sample_id in needed:
            found[sample_id] = sample
            if len(found) == len(needed):
                return found
    missing = sorted(needed - set(found))
    raise Phase3ContractError(f"validated map manifest is missing selected sample ids: {missing[:5]}")


def load_state_row(config: Mapping[str, Any], clip_id: str) -> dict[str, Any]:
    manifest = resolve_path(config, str(config["data"]["state_manifest"]))
    for row in iter_jsonl(manifest):
        if str(row.get("clip_id")) == clip_id:
            path = Path(str(row["state_cache"]))
            if not path.is_absolute():
                path = (manifest.parent / path).resolve()
            result = dict(row)
            result["state_cache"] = str(path)
            return result
    raise Phase3ContractError(f"combined state manifest has no row for {clip_id}")


def assert_finite_gradients(named_parameters: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    """Check every produced gradient and require at least one nonzero gradient."""

    checked = 0
    nonzero = 0
    missing: list[str] = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
            continue
        checked += 1
        if not bool(parameter.grad.isfinite().all().item()):
            raise RuntimeError(f"non-finite gradient: {name}")
        if bool(parameter.grad.count_nonzero().item()):
            nonzero += 1
    if checked == 0 or nonzero == 0:
        raise RuntimeError(f"no finite nonzero Fast v2 gradients; checked={checked} missing={missing[:10]}")
    return {"gradient_tensor_count": checked, "nonzero_gradient_tensor_count": nonzero, "missing_gradient_tensor_count": len(missing)}


def audit_gradient_presence(named_parameters: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    """Audit gradient metadata without synchronizing sharded CUDA tensors."""

    trainable = 0
    present = 0
    missing: list[str] = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        trainable += 1
        if parameter.grad is None:
            missing.append(name)
        else:
            present += 1
    if trainable == 0 or present == 0:
        raise RuntimeError(
            f"no Fast v2 gradients are present; trainable={trainable} missing={missing[:10]}"
        )
    return {
        "trainable_gradient_tensor_count": trainable,
        "present_gradient_tensor_count": present,
        "missing_gradient_tensor_count": len(missing),
        "per_tensor_cuda_value_sync_performed": False,
    }


def assert_trainable_allowlist(model: Any, state_projector: Any) -> dict[str, Any]:
    from lingbot_fast_v2_injection_v0 import fast_v2_trainable_parameters

    allowed_model = dict(fast_v2_trainable_parameters(model))
    actual_model = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(actual_model) != set(allowed_model) or not actual_model:
        raise RuntimeError(
            f"Fast v2 model trainable allowlist mismatch: unexpected={sorted(set(actual_model)-set(allowed_model))[:10]} "
            f"missing={sorted(set(allowed_model)-set(actual_model))[:10]}"
        )
    projector = {
        f"fast_v2_state_action_projector.{name}": parameter
        for name, parameter in state_projector.named_parameters()
        if parameter.requires_grad
    }
    if not projector or not all("fast_v2_state_proj" in name for name in projector):
        raise RuntimeError("Fast v2 state/action projector allowlist mismatch")
    return {
        "model_trainable_names": sorted(actual_model),
        "projector_trainable_names": sorted(projector),
        "model_trainable_parameters": sum(parameter.numel() for parameter in actual_model.values()),
        "projector_trainable_parameters": sum(parameter.numel() for parameter in projector.values()),
        "backbone_trainable_parameters": 0,
    }


def validate_checkpoint_envelope(
    envelope: Mapping[str, Any],
    *,
    expected_optimizer_steps: int = RUN_LOCAL_OPTIMIZER_STEPS,
    expected_config_sha256: str | None = None,
    expected_injection_config: Mapping[str, Any] | None = None,
    expected_source_binding: Mapping[str, Any] | None = None,
    expected_kind: str = CHECKPOINT_KIND,
    expected_smoke_only: bool = True,
    expected_run_local_optimizer_steps: int = RUN_LOCAL_OPTIMIZER_STEPS,
) -> None:
    """Validate the exact legacy or resume-smoke outer checkpoint envelope."""

    legacy_keys = {
        "kind", "namespace", "optimizer_steps", "smoke_only",
        "authorizes_longer_training", "source_checkpoint", "config_sha256",
        "fast_v2_injection", "fast_v2_state_action_projector",
    }
    resume_keys = legacy_keys | {
        "source_checkpoint_size_bytes", "source_checkpoint_sha256",
        "source_optimizer_steps", "run_local_optimizer_steps",
        "total_optimizer_steps",
    }
    if set(envelope) not in (legacy_keys, resume_keys):
        raise Phase3ContractError(
            "smoke checkpoint outer envelope keys are not exact: "
            f"actual={sorted(envelope)}"
        )
    _expect("smoke checkpoint kind", envelope.get("kind"), expected_kind)
    _expect("smoke checkpoint namespace", envelope.get("namespace"), CHECKPOINT_NAMESPACE)
    _expect("smoke checkpoint steps", envelope.get("optimizer_steps"), expected_optimizer_steps)
    _expect("smoke checkpoint scope", envelope.get("smoke_only"), expected_smoke_only)
    _expect("smoke checkpoint longer-training scope", envelope.get("authorizes_longer_training"), False)
    if expected_config_sha256 is not None:
        _expect("smoke checkpoint config", envelope.get("config_sha256"), expected_config_sha256)
    if not _is_sha256(envelope.get("config_sha256")):
        raise Phase3ContractError("smoke checkpoint config_sha256 is not lowercase sha256")

    if set(envelope) == legacy_keys:
        if envelope.get("source_checkpoint") is not None:
            raise Phase3ContractError("legacy Phase 3 smoke checkpoint cannot have a source checkpoint")
        if expected_source_binding is not None:
            raise Phase3ContractError("resume smoke checkpoint lacks source provenance fields")
    else:
        source_path = Path(str(envelope.get("source_checkpoint", "")))
        if not source_path.is_absolute():
            raise Phase3ContractError("resume smoke source_checkpoint must be absolute")
        if not _is_sha256(envelope.get("source_checkpoint_sha256")):
            raise Phase3ContractError("resume smoke source sha256 is not lowercase sha256")
        source_steps = envelope.get("source_optimizer_steps")
        local_steps = envelope.get("run_local_optimizer_steps")
        total_steps = envelope.get("total_optimizer_steps")
        if not isinstance(source_steps, int) or source_steps <= 0:
            raise Phase3ContractError("resume smoke source optimizer steps must be positive")
        _expect(
            "resume smoke run-local steps",
            local_steps,
            expected_run_local_optimizer_steps,
        )
        _expect("resume smoke total steps", total_steps, source_steps + local_steps)
        _expect("resume smoke optimizer steps", envelope.get("optimizer_steps"), total_steps)
        if not isinstance(envelope.get("source_checkpoint_size_bytes"), int) or int(
            envelope["source_checkpoint_size_bytes"]
        ) <= 0:
            raise Phase3ContractError("resume smoke source checkpoint size must be positive")
        if expected_source_binding is not None:
            for key in (
                "source_checkpoint", "source_checkpoint_size_bytes",
                "source_checkpoint_sha256", "source_optimizer_steps",
                "run_local_optimizer_steps", "total_optimizer_steps",
            ):
                _expect(f"resume smoke {key}", envelope.get(key), expected_source_binding.get(key))

    injection = envelope.get("fast_v2_injection")
    if not isinstance(injection, Mapping) or injection.get("namespace") != CHECKPOINT_NAMESPACE:
        raise Phase3ContractError("checkpoint lacks isolated Fast v2 injection envelope")
    if set(injection) != {"kind", "namespace", "injection_schema_version", "config", "state"}:
        raise Phase3ContractError("nested Fast v2 injection envelope keys are not exact")
    _expect("nested Fast v2 checkpoint kind", injection.get("kind"), INJECTION_CHECKPOINT_KIND)
    _expect("nested Fast v2 injection schema", injection.get("injection_schema_version"), INJECTION_SCHEMA_VERSION)
    if expected_injection_config is not None:
        _expect("nested Fast v2 injection config", injection.get("config"), dict(expected_injection_config))
    state = injection.get("state")
    if not isinstance(state, Mapping) or not state:
        raise Phase3ContractError("nested Fast v2 checkpoint state must be a non-empty mapping")
    noncanonical = sorted(
        name
        for name in state
        if FSDP_WRAPPED_MODULE_SEGMENT in str(name).split(".")
    )
    if noncanonical:
        raise Phase3ContractError(
            "nested Fast v2 checkpoint contains non-canonical FSDP wrapper segments: "
            f"{noncanonical[:10]}"
        )
    serialized_keys = json.dumps(sorted(injection.get("state", {}))).lower()
    if "memory_dense" in serialized_keys or "low" in serialized_keys or "high" in serialized_keys:
        raise Phase3ContractError("foreign LOW/HIGH or memory-dense keys appeared in Fast v2 checkpoint")
    projector_state = envelope.get("fast_v2_state_action_projector")
    if not isinstance(projector_state, Mapping) or set(projector_state) != PROJECTOR_STATE_KEYS:
        raise Phase3ContractError("state/action projector checkpoint keys are not exact")


def load_state_projector_checkpoint(projector: Any, state: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly copy the nested projector tensors before DDP wrapping."""

    expected = projector.state_dict()
    if set(state) != set(expected) or set(expected) != PROJECTOR_STATE_KEYS:
        raise Phase3ContractError("state/action projector load keys are not exact")
    for name, target in expected.items():
        value = state[name]
        if not hasattr(value, "shape"):
            raise Phase3ContractError(f"state/action projector value is not a tensor: {name}")
        if tuple(value.shape) != tuple(target.shape):
            raise Phase3ContractError(f"state/action projector shape mismatch for {name}")
        target.copy_(value.to(device=target.device, dtype=target.dtype))
    return {"strict": True, "tensor_count": len(expected), "keys": sorted(expected)}


def audit_compact_checkpoint_tensors(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """Fail if a saved tensor view retains a larger backing storage."""

    states = (
        envelope["fast_v2_injection"]["state"],
        envelope["fast_v2_state_action_projector"],
    )
    checked = 0
    for state in states:
        for name, tensor in state.items():
            if not hasattr(tensor, "untyped_storage"):
                raise Phase3ContractError(f"checkpoint value is not a tensor: {name}")
            expected_bytes = tensor.numel() * tensor.element_size()
            if tensor.untyped_storage().nbytes() != expected_bytes:
                raise Phase3ContractError(
                    f"checkpoint tensor storage is not compact: {name} "
                    f"storage={tensor.untyped_storage().nbytes()} expected={expected_bytes}"
                )
            checked += 1
    return {"status": "pass", "all_tensor_storages_compact": True, "tensor_count": checked}


def evaluate_projector_gradient_gate(
    rank_reports: Iterable[Mapping[str, Any]],
    *,
    source_optimizer_steps: int,
    required: bool,
    expected_steps: int = RUN_LOCAL_OPTIMIZER_STEPS,
    require_every_step_nonzero: bool = False,
) -> dict[str, Any]:
    """Require synchronized projector activation during a resumed smoke."""

    reports = [dict(report) for report in rank_reports]
    per_rank: list[list[float]] = []
    for report in reports:
        gradients = report.get("gradients")
        if not isinstance(gradients, list) or len(gradients) != expected_steps:
            raise Phase3ContractError(
                f"projector gradient gate requires {expected_steps} reports per rank"
            )
        values = [float(item["state_action_projector_gradient_norm"]) for item in gradients]
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise Phase3ContractError("projector gradient norms must be finite and non-negative")
        per_rank.append(values)
    per_step = []
    activated_steps = []
    for local_step in range(1, expected_steps + 1):
        values = [row[local_step - 1] for row in per_rank]
        all_ranks_nonzero = bool(values) and all(value > 0.0 for value in values)
        if all_ranks_nonzero:
            activated_steps.append(local_step)
        per_step.append({
            "run_local_optimizer_step": local_step,
            "total_optimizer_step": source_optimizer_steps + local_step,
            "gradient_norms_by_rank": values,
            "minimum_gradient_norm": min(values) if values else None,
            "maximum_gradient_norm": max(values) if values else None,
            "all_ranks_nonzero": all_ranks_nonzero,
        })
    activated = (
        len(activated_steps) == expected_steps
        if require_every_step_nonzero
        else bool(activated_steps)
    )
    return {
        "required_for_resume": required,
        "requires_every_step_nonzero": require_every_step_nonzero,
        "status": "pass" if (not required or activated) else "quality_fail",
        "activated": activated,
        "activated_run_local_steps": activated_steps,
        "all_observed_gradient_norms_zero": bool(per_rank) and all(
            value == 0.0 for row in per_rank for value in row
        ),
        "per_step": per_step,
    }


def _initialize_caches(torch: Any, config: Mapping[str, Any], device: Any, dtype: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = config["data"]
    causal = config["causal_runtime"]
    frame_seqlen = int(data["latent_height"]) * int(data["latent_width"]) // 4
    self_shape = (1, int(causal["local_attention_frames"]) * frame_seqlen, 5, 128)
    cross_shape = (1, 512, 40, 128)
    self_cache = [
        {
            "k": torch.zeros(self_shape, device=device, dtype=dtype),
            "v": torch.zeros(self_shape, device=device, dtype=dtype),
            "global_end_index": torch.zeros((), device=device, dtype=torch.long),
            "local_end_index": torch.zeros((), device=device, dtype=torch.long),
        }
        for _ in range(40)
    ]
    cross_cache = [
        {
            "k": torch.zeros(cross_shape, device=device, dtype=dtype),
            "v": torch.zeros(cross_shape, device=device, dtype=dtype),
            "is_init": torch.zeros((), device=device, dtype=torch.long),
        }
        for _ in range(40)
    ]
    return self_cache, cross_cache


def _detach_caches(caches: Iterable[dict[str, Any]]) -> None:
    for cache in caches:
        for key in ("k", "v"):
            cache[key] = cache[key].detach()


def _convert_x0(torch: Any, flow: Any, latent: Any, timestep: Any, scheduler: Any) -> Any:
    sigmas = scheduler.sigmas.double().to(flow.device)
    timesteps = scheduler.timesteps.double().to(flow.device)
    timestep_id = torch.argmin((timesteps - timestep.double().to(flow.device)).abs())
    sigma = sigmas[timestep_id].reshape(-1, 1, 1, 1)
    return (latent.double() - sigma * flow.double()).float()


def _load_real_batch(
    config: Mapping[str, Any], *, spec: Mapping[str, Any], device: Any, dtype: Any
) -> dict[str, Any]:
    import numpy as np
    import torch

    source = str(spec["source"])
    ordinal = int(spec["source_ordinal"])
    record, release = select_record(config, source=source, ordinal=ordinal)
    sample_ids = record.get("map_memory_sample_ids") or record.get("memory_dense_sample_ids")
    if not isinstance(sample_ids, list) or len(sample_ids) != 21:
        raise Phase3ContractError(f"{record.get('clip_id')}: expected 21 exact-frame sample ids")
    sample_ids = [str(value) for value in sample_ids[:20]]
    samples = load_selected_samples(config, release, sample_ids)
    dense_frames = []
    for sample_id in sample_ids:
        dense_path = Path(str(samples[sample_id]["dense_path"]))
        dense = np.load(dense_path)["dense"].astype("float32")
        if dense.shape != (7, 176, 320) or not np.isfinite(dense).all():
            raise RuntimeError(f"invalid selected dense payload: {dense_path} shape={dense.shape}")
        dense_frames.append(dense)
    dense = torch.from_numpy(np.stack(dense_frames)).to(device=device, dtype=dtype)

    state_row = load_state_row(config, str(record["clip_id"]))
    state_obj = np.load(state_row["state_cache"])
    alive = state_obj["ego_alive"].astype("float32")[:20]
    health = state_obj["ego_health"].astype("float32")[:20]
    dead = state_obj["opponent_dead_mask"].astype("float32")[:20]
    state = np.stack([
        np.stack((np.full_like(dead[index], alive[index]), np.full_like(dead[index], health[index]), dead[index]))
        for index in range(20)
    ])
    state_tensor = torch.from_numpy(state).to(device=device, dtype=dtype)

    latent_obj = torch.load(record["latent_cache"], map_location="cpu")
    x0 = latent_obj["latent"][:, :20].to(device=device, dtype=dtype)
    condition = latent_obj["condition"][:, :20].to(device=device, dtype=dtype)
    if tuple(x0.shape) != (16, 20, 60, 104) or tuple(condition.shape) != (20, 20, 60, 104):
        raise RuntimeError(f"{record['clip_id']}: selected latent/condition shape drift")
    text_obj = torch.load(record["text_cache"], map_location="cpu")
    context = text_obj["context"].to(device=device, dtype=dtype)
    validate_text_context_shape(context.shape)

    from lingbot_fast_v2_phase3_conditioning_v0 import prepare_fast_v2_camera_chunks

    _full_camera, all_camera_chunks = prepare_fast_v2_camera_chunks(
        record["poses"],
        record["intrinsics"],
        official_source_root=config["official"]["source_root"],
        device=device,
        dtype=dtype,
    )
    cameras = list(all_camera_chunks[: int(spec["target_chunk"]) + 1])
    return {
        "source": source,
        "clip_id": str(record["clip_id"]),
        "x0": x0,
        "condition": condition,
        "context": context,
        "dense": dense,
        "state": state_tensor,
        "cameras": cameras,
        "sample_ids": sample_ids,
        "latent_cache": str(record["latent_cache"]),
        "text_cache": str(record["text_cache"]),
        "poses": str(record["poses"]),
        "intrinsics": str(record["intrinsics"]),
        "state_cache": str(state_row["state_cache"]),
    }


def _execute_fixed_training(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    """Run one fixed, pre-authorized Fast v2 contract after strict preflight."""

    expected_world = int(config["execution"]["world_size"])
    local_optimizer_steps = int(contract["local_optimizer_steps"])
    gradient_accumulation_steps = int(config["execution"]["gradient_accumulation_steps"])
    expected_microbatches = local_optimizer_steps * gradient_accumulation_steps
    run_config_path = Path(str(contract.get("config_path", args.config))).resolve()
    run_config_sha256 = str(contract.get("config_sha256", sha256_json(config)))
    source_checkpoint_config_sha256 = str(
        contract.get("source_checkpoint_config_sha256", sha256_json(config))
    )
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world != expected_world:
        raise Phase3ContractError(
            f"{contract['execute_flag']} requires torchrun WORLD_SIZE={expected_world}, got {world}"
        )
    if not args.run_token:
        raise Phase3ContractError(
            f"{contract['execute_flag']} requires a fresh shared --run-token on all eight ranks"
        )
    resume_binding = build_resume_checkpoint_binding(
        args,
        run_local_optimizer_steps=local_optimizer_steps,
    )
    source_optimizer_steps = int(resume_binding["source_optimizer_steps"])
    total_optimizer_steps = int(resume_binding["total_optimizer_steps"])
    preflight_path = args.out_dir / str(contract["preflight_filename"])
    if rank == 0:
        if args.out_dir.exists():
            raise Phase3ContractError(f"refusing existing smoke output directory: {args.out_dir}")
        args.out_dir.mkdir(parents=True, exist_ok=False)
        preflight = contract["build_preflight"]()
        preflight["execution_run_token"] = args.run_token
        preflight["resume_checkpoint_binding"] = resume_binding
        atomic_json(preflight_path, preflight)
    else:
        deadline = time.time() + args.preflight_wait_seconds
        while not preflight_path.is_file() and time.time() < deadline:
            time.sleep(1)
        if not preflight_path.is_file():
            raise Phase3ContractError("rank-zero strict preflight did not arrive")
        preflight = load_json(preflight_path)
    _expect("strict preflight status", preflight.get("status"), "pass")
    _expect("strict preflight model_loaded", preflight.get("model_loaded"), False)
    _expect("strict preflight config hash", preflight.get("config_sha256"), run_config_sha256)
    _expect("strict preflight run token", preflight.get("execution_run_token"), args.run_token)
    _expect(
        "strict preflight resume checkpoint binding",
        preflight.get("resume_checkpoint_binding"),
        resume_binding,
    )

    # No torch/CUDA/distributed/model imports occur above this line.
    import types
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
        raise Phase3ContractError("--execute-smoke requires one visible CUDA device per local rank")
    torch.cuda.set_device(local_rank)
    collective_timeout = timedelta(seconds=180)
    dist.init_process_group("nccl", timeout=collective_timeout)
    sequence_parallel_group = dist.new_group(
        ranks=list(range(world)),
        backend="nccl",
        timeout=collective_timeout,
    )
    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16
    seed = int(config["execution"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    source_root = Path(str(config["official"]["source_root"])).resolve()
    sys.path.insert(0, str(source_root))
    from wan.distributed.fsdp import shard_model
    import wan.distributed.sequence_parallel as sequence_parallel
    from wan.modules.model_fast import WanModelFast
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    from lingbot_fast_v2_injection_v0 import (
        FastV2InjectionConfig,
        fast_v2_checkpoint_envelope,
        fast_v2_trainable_parameters,
        inject_fast_v2_native,
        load_fast_v2_checkpoint_envelope,
    )
    from lingbot_fast_v2_phase3_conditioning_v0 import (
        FastV2StateActionProjector,
        install_differentiable_sequence_parallel_collectives,
        partition_sequence_parallel,
    )

    collective_patch = install_differentiable_sequence_parallel_collectives(
        sequence_parallel,
        process_group=sequence_parallel_group,
    )
    collective_autograd = run_collective_autograd_self_test(
        torch,
        dist,
        sequence_parallel,
        rank=rank,
        world=world,
        device=device,
    )
    fsdp_sp_autograd = run_fsdp_sp_autograd_self_test(
        torch,
        dist,
        FSDP,
        sequence_parallel,
        rank=rank,
        world=world,
        local_rank=local_rank,
        device=device,
    )
    sp_attn_forward_causal = sequence_parallel.sp_attn_forward_causal
    sp_dit_forward_causal = sequence_parallel.sp_dit_forward_causal

    injection_cfg = FastV2InjectionConfig(
        cond_dim=128,
        adapter_hidden_dim=128,
        wrap_first_blocks=None,
        residual_scale_init=1.0,
        lora_rank=16,
        lora_alpha=16.0,
        lora_dropout=0.0,
        lora_targets=tuple(config["injection"]["lora_targets"]),
    )
    resume_envelope = None
    if resume_binding["enabled"]:
        emit_rank_stage(rank, "resume_checkpoint_outer_load_begin")
        try:
            resume_envelope = torch.load(
                resume_binding["source_checkpoint"],
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise Phase3ContractError(
                f"cannot load bound resume checkpoint: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(resume_envelope, Mapping):
            raise Phase3ContractError("resume checkpoint root must be a mapping")
        validate_checkpoint_envelope(
            resume_envelope,
            expected_optimizer_steps=source_optimizer_steps,
            expected_config_sha256=source_checkpoint_config_sha256,
            expected_injection_config=injection_cfg.to_dict(),
            expected_kind=str(
                contract.get("source_checkpoint_kind", CHECKPOINT_KIND)
            ),
            expected_smoke_only=bool(
                contract.get("source_checkpoint_smoke_only", True)
            ),
            expected_run_local_optimizer_steps=int(
                contract.get(
                    "source_checkpoint_run_local_optimizer_steps",
                    RUN_LOCAL_OPTIMIZER_STEPS,
                )
            ),
        )
        emit_rank_stage(rank, "resume_checkpoint_outer_load_complete")
    model = WanModelFast.from_pretrained(
        str(Path(str(config["official"]["model_root"])).resolve()),
        subfolder="transformers",
        torch_dtype=dtype,
        local_attn_size=18,
        sink_size=6,
    )
    model.requires_grad_(False)
    for block in model.blocks:
        block.self_attn.forward = types.MethodType(sp_attn_forward_causal, block.self_attn)
    injection_report = inject_fast_v2_native(model, injection_cfg, freeze_base=True)
    model.forward = types.MethodType(sp_dit_forward_causal, model)
    projector = FastV2StateActionProjector(cond_dim=128).to(device=device, dtype=dtype)
    resume_load_audit = None
    if resume_envelope is None:
        projector.assert_zero_initialization()
    else:
        load_fast_v2_checkpoint_envelope(
            model, resume_envelope["fast_v2_injection"], injection_cfg
        )
        projector_audit = load_state_projector_checkpoint(
            projector, resume_envelope["fast_v2_state_action_projector"]
        )
        resume_load_audit = {
            "status": "pass",
            "loaded_after_injection": True,
            "loaded_before_fsdp_and_ddp": True,
            "fast_v2_injection_strict": True,
            "state_action_projector": projector_audit,
            "optimizer_state_restored": False,
            "resume_semantics": "Fast v2 injection and state/action projector weights only",
        }
        injection_report["checkpoint_restored_after_fresh_injection"] = True
        resume_envelope = None
    trainable_audit = assert_trainable_allowlist(model, projector)

    model = shard_model(model, device_id=local_rank, use_lora=True)
    projector = torch.nn.parallel.DistributedDataParallel(projector, device_ids=[local_rank], output_device=local_rank)
    model.train()
    projector.train()
    model_named = list(fast_v2_trainable_parameters(model))
    projector_named = [(f"fast_v2_state_action_projector.{name}", value) for name, value in projector.named_parameters()]
    model_named_map = dict(model_named)
    actual_model_trainables = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if not model_named_map or set(model_named_map) != set(actual_model_trainables):
        raise RuntimeError(
            "post-FSDP Fast v2 trainable mismatch: "
            f"expected={len(model_named_map)} actual={len(actual_model_trainables)}"
        )
    if any(not parameter.requires_grad for parameter in model_named_map.values()):
        raise RuntimeError("post-FSDP Fast v2 parameter unexpectedly has requires_grad=False")
    post_fsdp_trainable_audit = {
        "model_trainable_tensor_count": len(model_named_map),
        "model_trainable_parameter_count": sum(
            parameter.numel() for parameter in model_named_map.values()
        ),
        "all_model_trainables_require_grad": True,
        "backbone_trainable_parameters": 0,
    }
    lora_params = [parameter for name, parameter in model_named if "fast_v2_lora_" in name]
    adapter_params = [parameter for name, parameter in model_named if "fast_v2_lora_" not in name]
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_params + [parameter for _, parameter in projector_named], "lr": float(config["execution"]["learning_rate"])},
            {"params": lora_params, "lr": float(config["execution"]["lora_learning_rate"])},
        ],
        weight_decay=float(config["execution"]["weight_decay"]),
    )

    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    scheduler.set_timesteps(1000, shift=10.0)
    timesteps = scheduler.timesteps[[0, 250, 500, 750]]
    actual_timesteps = [int(item.item()) for item in timesteps]
    _expect("runtime selected timesteps", actual_timesteps, [999, 967, 908, 768])
    frame_seqlen = 60 * 104 // 4
    seq_len = 4 * frame_seqlen
    losses: list[float] = []
    grad_reports: list[dict[str, Any]] = []
    local_microbatches: list[dict[str, Any]] = []
    input_consensus_audits: list[dict[str, Any]] = []
    optimizer_step_count = 0

    for step in range(1, local_optimizer_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_losses: list[float] = []
        for micro in range(gradient_accumulation_steps):
            spec = contract["microbatch_spec"](
                optimizer_step=step, accumulation_microbatch=micro, config=config
            )
            batch = _load_real_batch(config, spec=spec, device=device, dtype=dtype)
            target_chunk = int(spec["target_chunk"])
            timestep_cpu = timesteps[int(spec["timestep_index"])]
            x0_target = batch["x0"][:, target_chunk * 4 : (target_chunk + 1) * 4]
            generator = torch.Generator(device=device).manual_seed(int(spec["noise_seed"]))
            noise = torch.randn(x0_target.shape, generator=generator, device=device, dtype=torch.float32)
            input_summary = {
                **spec,
                "clip_id": batch["clip_id"],
                "sample_ids": batch["sample_ids"],
                "latent_cache": batch["latent_cache"],
                "text_cache": batch["text_cache"],
                "poses": batch["poses"],
                "intrinsics": batch["intrinsics"],
                "state_cache": batch["state_cache"],
                "text_shape": list(batch["context"].shape),
                "camera_shape": list(batch["cameras"][target_chunk].shape),
                "dense_shape": list(batch["dense"].shape),
                "state_shape": list(batch["state"].shape),
                "latent_shape": list(batch["x0"].shape),
                "image_condition_shape": list(batch["condition"].shape),
                "noise_shape": list(noise.shape),
                "text_sum": float(batch["context"].float().double().sum().cpu()),
                "camera_sum": float(batch["cameras"][target_chunk].float().double().sum().cpu()),
                "dense_sum": float(batch["dense"].float().double().sum().cpu()),
                "state_sum": float(batch["state"].float().double().sum().cpu()),
                "latent_sum": float(batch["x0"].float().double().sum().cpu()),
                "image_condition_sum": float(batch["condition"].float().double().sum().cpu()),
                "noise_sum": float(noise.double().sum().cpu()),
            }
            rank_inputs: list[Any] = [None for _ in range(world)]
            dist.all_gather_object(rank_inputs, input_summary)
            consensus = assert_consensus_summaries(rank_inputs, expected_replicas=world)
            input_consensus_audits.append({
                "global_microbatch_index": spec["global_microbatch_index"],
                **consensus,
            })

            self_cache, cross_cache = _initialize_caches(torch, config, device, dtype)
            cross_first = True
            prefix_commits = 0
            for prefix_chunk in range(target_chunk):
                start = prefix_chunk * 4
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                    full_tokens = projector(
                        batch["dense"][start : start + 4],
                        batch["state"][start : start + 4],
                        target_token_hw=(30, 52),
                    )
                    local_tokens = partition_sequence_parallel(full_tokens, rank=rank, world_size=world)
                    model(
                        x=[batch["x0"][:, start : start + 4]],
                        t=torch.zeros(1, device=device, dtype=timesteps.dtype),
                        context=[batch["context"]],
                        seq_len=seq_len,
                        y=[batch["condition"][:, start : start + 4]],
                        dit_cond_dict={
                            "c2ws_plucker_emb": (batch["cameras"][prefix_chunk],),
                            injection_cfg.cond_key: local_tokens,
                        },
                        kv_cache=self_cache,
                        crossattn_cache=cross_cache,
                        current_start=prefix_chunk * seq_len,
                        max_attention_size=18 * frame_seqlen,
                        frame_seqlen=frame_seqlen,
                        cross_attn_first_call=cross_first,
                    )
                cross_first = False
                prefix_commits += 1
                _detach_caches(self_cache)
                _detach_caches(cross_cache)

            current_latent = scheduler.add_noise(x0_target.float(), noise, timestep_cpu).to(dtype)
            target_start = target_chunk * 4
            synchronized_backward = should_sync_microbatch(
                accumulation_microbatch=micro, config=config
            )
            emit_rank_stage(
                rank,
                "training_forward_begin",
                optimizer_step=step,
                accumulation_microbatch=micro,
                synchronized_backward=synchronized_backward,
            )
            with torch.enable_grad(), ExitStack() as stack:
                if not torch.is_grad_enabled():
                    raise RuntimeError("training microbatch entered with autograd disabled")
                if not synchronized_backward:
                    stack.enter_context(model.no_sync())
                    stack.enter_context(projector.no_sync())
                with torch.amp.autocast("cuda", dtype=dtype):
                    full_tokens = projector(
                        batch["dense"][target_start : target_start + 4],
                        batch["state"][target_start : target_start + 4],
                        target_token_hw=(30, 52),
                    )
                    if not full_tokens.requires_grad or full_tokens.grad_fn is None:
                        raise RuntimeError("state/dense condition tokens detached from trainable projector")
                    local_tokens = partition_sequence_parallel(full_tokens, rank=rank, world_size=world)
                    prediction = model(
                        x=[current_latent],
                        t=timestep_cpu.reshape(1).to(device=device),
                        context=[batch["context"]],
                        seq_len=seq_len,
                        y=[batch["condition"][:, target_start : target_start + 4]],
                        dit_cond_dict={
                            "c2ws_plucker_emb": (batch["cameras"][target_chunk],),
                            injection_cfg.cond_key: local_tokens,
                        },
                        kv_cache=self_cache,
                        crossattn_cache=cross_cache,
                        current_start=target_chunk * seq_len,
                        max_attention_size=18 * frame_seqlen,
                        frame_seqlen=frame_seqlen,
                        cross_attn_first_call=cross_first,
                    )[0]
                    if not prediction.requires_grad or prediction.grad_fn is None:
                        raise RuntimeError(
                            "Fast v2 prediction detached from trainable LoRA/adapter graph"
                        )
                    target_flow = noise.to(prediction.dtype) - x0_target
                    raw_loss = F.mse_loss(prediction.float(), target_flow.float())
                    loss = raw_loss / 2.0
                    if not raw_loss.requires_grad or not loss.requires_grad:
                        raise RuntimeError("Fast v2 training loss detached from autograd")
                if not bool(loss.isfinite().item()):
                    raise RuntimeError(
                        f"non-finite loss at global microbatch {spec['global_microbatch_index']}"
                    )
                predicted_x0 = _convert_x0(
                    torch, prediction, current_latent, timestep_cpu, scheduler
                ).detach()
                emit_rank_stage(
                    rank,
                    "training_backward_begin",
                    optimizer_step=step,
                    accumulation_microbatch=micro,
                )
                loss.backward()
                emit_rank_stage(
                    rank,
                    "training_backward_returned",
                    optimizer_step=step,
                    accumulation_microbatch=micro,
                )
            _detach_caches(self_cache)
            _detach_caches(cross_cache)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                full_tokens = projector(
                    batch["dense"][target_start : target_start + 4],
                    batch["state"][target_start : target_start + 4],
                    target_token_hw=(30, 52),
                )
                local_tokens = partition_sequence_parallel(full_tokens, rank=rank, world_size=world)
                model(
                    x=[predicted_x0.to(dtype)],
                    t=torch.zeros(1, device=device, dtype=timesteps.dtype),
                    context=[batch["context"]],
                    seq_len=seq_len,
                    y=[batch["condition"][:, target_start : target_start + 4]],
                    dit_cond_dict={
                        "c2ws_plucker_emb": (batch["cameras"][target_chunk],),
                        injection_cfg.cond_key: local_tokens,
                    },
                    kv_cache=self_cache,
                    crossattn_cache=cross_cache,
                    current_start=target_chunk * seq_len,
                    max_attention_size=18 * frame_seqlen,
                    frame_seqlen=frame_seqlen,
                    cross_attn_first_call=False,
                )
            _detach_caches(self_cache)
            _detach_caches(cross_cache)
            expected_indices = expected_cache_indices_after_chunk(target_chunk)
            observed_indices = {
                (
                    int(cache["global_end_index"].item()),
                    int(cache["local_end_index"].item()),
                )
                for cache in self_cache
            }
            expected_pair = (
                expected_indices["global_end_index"],
                expected_indices["local_end_index"],
            )
            if observed_indices != {expected_pair}:
                raise RuntimeError(
                    f"cache index mismatch after target chunk {target_chunk}: {observed_indices} != {expected_pair}"
                )

            budget = microbatch_forward_budget(spec)
            if prefix_commits != budget["prefix_clean_commits"]:
                raise RuntimeError("prefix clean-commit budget drifted")
            loss_value = float(raw_loss.detach().cpu())
            step_losses.append(loss_value)
            local_microbatches.append({
                **spec,
                "clip_id": batch["clip_id"],
                "loss": loss_value,
                **budget,
                "synchronized_backward": synchronized_backward,
                "autograd_enabled": True,
                "condition_tokens_require_grad": True,
                "prediction_requires_grad": True,
                "loss_requires_grad": True,
                "cache_indices_after_target_commit": expected_indices,
            })
            emit_rank_stage(
                rank,
                "microbatch_complete",
                optimizer_step=step,
                accumulation_microbatch=micro,
            )

        named_trainables = list(model_named) + projector_named
        emit_rank_stage(rank, "gradient_presence_audit_begin", optimizer_step=step)
        grad_report = audit_gradient_presence(named_trainables)
        emit_rank_stage(rank, "gradient_presence_audit_complete", optimizer_step=step)
        max_gradient_norm = float(config["execution"]["max_gradient_norm"])
        emit_rank_stage(rank, "gradient_clip_begin", optimizer_step=step)
        model_grad_norm = model.clip_grad_norm_(max_gradient_norm)
        emit_rank_stage(rank, "model_gradient_clip_complete", optimizer_step=step)
        projector_grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in projector_named], max_gradient_norm
        )
        emit_rank_stage(rank, "gradient_clip_complete", optimizer_step=step)
        model_grad_norm_value = float(model_grad_norm.detach().cpu())
        projector_grad_norm_value = float(projector_grad_norm.detach().cpu())
        if not math.isfinite(projector_grad_norm_value) and contract.get(
            "projector_must_be_finite_nonzero_each_step", False
        ):
            raise TrainingQualityError(
                f"non-finite state/action projector gradient norm at optimizer step {step}"
            )
        if not math.isfinite(model_grad_norm_value) or not math.isfinite(projector_grad_norm_value):
            raise RuntimeError(f"non-finite gradient norm at optimizer step {step}")
        if projector_grad_norm_value == 0.0 and contract.get(
            "projector_must_be_finite_nonzero_each_step", False
        ):
            raise TrainingQualityError(
                f"zero state/action projector gradient norm at optimizer step {step}"
            )
        if model_grad_norm_value == 0.0 and projector_grad_norm_value == 0.0:
            raise RuntimeError(f"all global gradient norms are zero at optimizer step {step}")
        optimizer.step()
        optimizer_step_count += 1
        if len(step_losses) != gradient_accumulation_steps:
            raise RuntimeError(
                f"optimizer step {step} did not consume exactly "
                f"{gradient_accumulation_steps} microbatches"
            )
        step_loss = sum(step_losses) / float(gradient_accumulation_steps)
        if not math.isfinite(step_loss):
            raise RuntimeError(f"non-finite averaged loss at optimizer step {step}")
        losses.append(step_loss)
        grad_reports.append({
            **grad_report,
            "run_local_optimizer_step": step,
            "total_optimizer_step": source_optimizer_steps + step,
            "global_gradient_norms_finite": True,
            "at_least_one_global_gradient_norm_nonzero": True,
            "model_gradient_norm": model_grad_norm_value,
            "state_action_projector_gradient_norm": projector_grad_norm_value,
        })
        emit_rank_stage(
            rank,
            "optimizer_step_complete",
            run_local_optimizer_step=step,
            total_optimizer_step=source_optimizer_steps + step,
            loss=step_loss,
            model_gradient_norm=model_grad_norm_value,
            state_action_projector_gradient_norm=projector_grad_norm_value,
        )
    if optimizer_step_count != local_optimizer_steps:
        raise RuntimeError(f"optimizer step invariant violated: {optimizer_step_count}")

    checkpoint_path = args.out_dir / "checkpoints" / str(
        contract["checkpoint_filename"]
    ).format(total_optimizer_steps=total_optimizer_steps)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with FSDP.summon_full_params(
        model,
        recurse=True,
        writeback=False,
        rank0_only=True,
        offload_to_cpu=True,
    ):
        if rank == 0:
            root_model = model.module
            envelope = {
                "kind": str(contract["checkpoint_kind"]),
                "namespace": CHECKPOINT_NAMESPACE,
                "optimizer_steps": total_optimizer_steps,
                "smoke_only": bool(contract["checkpoint_smoke_only"]),
                "authorizes_longer_training": False,
                "source_checkpoint": resume_binding["source_checkpoint"],
                "config_sha256": run_config_sha256,
                "fast_v2_injection": fast_v2_checkpoint_envelope(root_model, injection_cfg),
                "fast_v2_state_action_projector": {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in projector.module.state_dict().items()
                },
            }
            if resume_binding["enabled"]:
                envelope.update({
                    key: resume_binding[key]
                    for key in (
                        "source_checkpoint_size_bytes", "source_checkpoint_sha256",
                        "source_optimizer_steps", "run_local_optimizer_steps",
                        "total_optimizer_steps",
                    )
                })
            validate_checkpoint_envelope(
                envelope,
                expected_optimizer_steps=total_optimizer_steps,
                expected_config_sha256=run_config_sha256,
                expected_injection_config=injection_cfg.to_dict(),
                expected_source_binding=resume_binding if resume_binding["enabled"] else None,
                expected_kind=str(contract["checkpoint_kind"]),
                expected_smoke_only=bool(contract["checkpoint_smoke_only"]),
                expected_run_local_optimizer_steps=local_optimizer_steps,
            )
            checkpoint_compaction_audit = audit_compact_checkpoint_tensors(envelope)
            checkpoint_temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            if checkpoint_temporary.exists():
                raise RuntimeError(f"refusing stale checkpoint temporary: {checkpoint_temporary}")
            torch.save(envelope, checkpoint_temporary)
            checkpoint_temporary.replace(checkpoint_path)
    gathered: list[Any] = [None for _ in range(world)]
    dist.all_gather_object(gathered, {
        "rank": rank,
        "microbatches": local_microbatches,
        "optimizer_step_losses": losses,
        "gradients": grad_reports,
        "input_consensus_audits": input_consensus_audits,
    })
    dist.barrier()
    gate_failure_message = None
    if rank == 0:
        global_microbatches, replica_audits = collapse_rank_microbatch_reports(
            gathered,
            expected_world=world,
            expected_microbatches=expected_microbatches,
        )
        source_counts = {
            name: sum(row["source"] == name for row in global_microbatches)
            for name in ("w3", "tier69h", "event50h")
        }
        failures = []
        projector_gradient_gate = evaluate_projector_gradient_gate(
            gathered,
            source_optimizer_steps=source_optimizer_steps,
            required=bool(resume_binding["enabled"]),
            expected_steps=local_optimizer_steps,
            require_every_step_nonzero=bool(
                contract.get("projector_must_be_finite_nonzero_each_step", False)
            ),
        )
        if projector_gradient_gate["status"] != "pass":
            failures.append(
                "state/action projector gradient activation contract failed"
            )
        rank_input_audits = [report["input_consensus_audits"] for report in gathered]
        if any(audits != rank_input_audits[0] for audits in rank_input_audits[1:]):
            failures.append("rank-local copies of runtime input consensus audit differ")
        if len(rank_input_audits[0]) != expected_microbatches or not all(
            audit.get("all_ranks_identical") is True for audit in rank_input_audits[0]
        ):
            failures.append(
                f"runtime input consensus did not cover exactly {expected_microbatches} "
                "global microbatches"
            )
        if source_counts != dict(contract["expected_source_counts"]):
            failures.append(f"global source mix mismatch: {source_counts}")
        if len(global_microbatches) != expected_microbatches:
            failures.append(
                f"global microbatch count is {len(global_microbatches)}, "
                f"expected {expected_microbatches}"
            )
        expected_sync = [
            micro == gradient_accumulation_steps - 1
            for _step in range(local_optimizer_steps)
            for micro in range(gradient_accumulation_steps)
        ]
        if [row["synchronized_backward"] for row in global_microbatches] != expected_sync:
            failures.append("no_sync boundary is not the final microbatch of each optimizer step")
        for row in global_microbatches:
            expected_budget = microbatch_forward_budget(row)
            for key, expected in expected_budget.items():
                if row.get(key) != expected:
                    failures.append(
                        f"global microbatch {row['global_microbatch_index']} {key}={row.get(key)} != {expected}"
                    )
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
            failures.append("isolated Fast v2 checkpoint is missing")
        gate_status = "pass"
        if failures:
            gate_status = (
                "quality_fail"
                if projector_gradient_gate["status"] == "quality_fail" and len(failures) == 1
                else "fail"
            )
        gate = {
            "kind": str(contract["gate_kind"]),
            "status": gate_status,
            "completed_at": utc_now(),
            "config": str(run_config_path),
            "config_sha256": run_config_sha256,
            "strict_preflight": str(preflight_path),
            "strict_preflight_sha256": sha256_file(preflight_path),
            "world_size": world,
            "sequence_parallel_replicas_per_global_microbatch": world,
            "optimizer_steps_run": optimizer_step_count,
            "expected_optimizer_steps": local_optimizer_steps,
            "source_optimizer_steps": source_optimizer_steps,
            "run_local_optimizer_steps": optimizer_step_count,
            "total_optimizer_steps": total_optimizer_steps,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "global_microbatch_count": len(global_microbatches),
            "source_counts": source_counts,
            "global_microbatches": global_microbatches,
            "rank_replica_consistency": replica_audits,
            "runtime_input_consensus": rank_input_audits[0],
            "rank_replicas_count_as_samples": False,
            "rank_reports": gathered,
            "trainable_allowlist": trainable_audit,
            "post_fsdp_trainable_audit": post_fsdp_trainable_audit,
            "differentiable_collective_patch": collective_patch,
            "collective_autograd_self_test": collective_autograd,
            "fsdp_sequence_parallel_autograd_self_test": fsdp_sp_autograd,
            "injection": injection_report,
            "resume_checkpoint_binding": resume_binding,
            "resume_load_audit": resume_load_audit,
            "state_action_projector_gradient_gate": projector_gradient_gate,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_kind": str(contract["checkpoint_kind"]),
            "checkpoint_compaction": checkpoint_compaction_audit,
            "foreign_low_high_weights_loaded_or_merged": False,
            "backbone_frozen": True,
            "official_causal_runtime": {
                "timestep_indices": [0, 250, 500, 750],
                "selected_timesteps": actual_timesteps,
                "shift": 10.0,
                "chunk_size": 4,
                "local_attention_frames": 18,
                "sink_frames": 6,
                "mutable_kv_cache": True,
                "clean_commit_timestep": 0,
            },
            "finite_losses": True,
            "finite_gradients": True,
            "authorizes_longer_training": False,
            "maximum_optimizer_steps_authorized": local_optimizer_steps,
            "continuation_policy": str(contract["continuation_policy"]),
            "failures": failures,
        }
        atomic_json(args.out_dir / str(contract["gate_filename"]), gate)
        if failures:
            gate_failure_message = str(contract["failure_prefix"]) + "; ".join(failures)
    failure_broadcast = [gate_failure_message]
    dist.broadcast_object_list(failure_broadcast, src=0)
    dist.barrier()
    dist.destroy_process_group(sequence_parallel_group)
    dist.destroy_process_group()
    if failure_broadcast[0] is not None:
        raise RuntimeError(str(failure_broadcast[0]))


def execute_smoke(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    """Execute only the fixed two-step Phase 3 smoke contract."""

    contract = {
        "execute_flag": "--execute-smoke",
        "local_optimizer_steps": RUN_LOCAL_OPTIMIZER_STEPS,
        "config_path": args.config,
        "config_sha256": sha256_json(config),
        "source_checkpoint_config_sha256": sha256_json(config),
        "build_preflight": lambda: build_strict_preflight(args.config),
        "preflight_filename": "STRICT_PREFLIGHT_BEFORE_MODEL_LOAD_v0.json",
        "microbatch_spec": global_microbatch_spec,
        "expected_source_counts": {"w3": 1, "tier69h": 2, "event50h": 1},
        "checkpoint_kind": CHECKPOINT_KIND,
        "checkpoint_smoke_only": True,
        "checkpoint_filename": (
            "lingbot_fast_v2_phase3_smoke_step_"
            "{total_optimizer_steps:06d}_compact.pt"
        ),
        "gate_kind": GATE_KIND,
        "gate_filename": "FAST_V2_PHASE3_SMOKE_GATE_REPORT_v0.json",
        "projector_must_be_finite_nonzero_each_step": False,
        "continuation_policy": (
            "This smoke-only gate records recovery quality but cannot authorize "
            "a 25-step or any longer training run."
        ),
        "failure_prefix": "Fast v2 Phase 3 smoke gate failed: ",
    }
    _execute_fixed_training(args, config, contract)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validation-only", action="store_true", help="Run CPU-only strict preflight (default).")
    modes.add_argument("--execute-smoke", action="store_true", help="Explicitly permit the fixed two-step 8-GPU smoke.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_VALIDATION_REPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SMOKE_DIR)
    parser.add_argument("--run-token", default=None, help="Fresh launch token shared by all torchrun ranks.")
    parser.add_argument("--preflight-wait-seconds", type=int, default=7200)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Absolute compact Phase 3 checkpoint path; valid only with --execute-smoke.",
    )
    parser.add_argument("--resume-checkpoint-size-bytes", type=int, default=None)
    parser.add_argument("--resume-checkpoint-sha256", default=None)
    parser.add_argument("--resume-checkpoint-optimizer-steps", type=int, default=None)
    args = parser.parse_args(argv)
    resume_values = (
        args.resume_checkpoint,
        args.resume_checkpoint_size_bytes,
        args.resume_checkpoint_sha256,
        args.resume_checkpoint_optimizer_steps,
    )
    if any(value is not None for value in resume_values) and not all(
        value is not None for value in resume_values
    ):
        parser.error(
            "resume requires --resume-checkpoint, --resume-checkpoint-size-bytes, "
            "--resume-checkpoint-sha256, and --resume-checkpoint-optimizer-steps together"
        )
    if args.resume_checkpoint is not None:
        if not args.execute_smoke:
            parser.error("resume checkpoint arguments require --execute-smoke")
        if not args.resume_checkpoint.is_absolute():
            parser.error("--resume-checkpoint must be an absolute path")
        if args.resume_checkpoint_size_bytes <= 0:
            parser.error("--resume-checkpoint-size-bytes must be positive")
        if not _is_sha256(args.resume_checkpoint_sha256):
            parser.error("--resume-checkpoint-sha256 must be 64 lowercase hexadecimal characters")
        if args.resume_checkpoint_optimizer_steps <= 0:
            parser.error("--resume-checkpoint-optimizer-steps must be positive")
        args.resume_checkpoint = args.resume_checkpoint.resolve()
    args.config = args.config.resolve()
    args.report = args.report.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.preflight_wait_seconds < 60:
        parser.error("--preflight-wait-seconds must be at least 60")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_json(args.config)
    validate_config(config)
    if args.execute_smoke:
        try:
            execute_smoke(args, config)
        except Exception as exc:
            best_effort_destroy_distributed()
            gate_path = args.out_dir / "FAST_V2_PHASE3_SMOKE_GATE_REPORT_v0.json"
            if (
                int(os.environ.get("RANK", "0")) == 0
                and args.out_dir.is_dir()
                and not gate_path.is_file()
            ):
                failure = {
                    "kind": GATE_KIND,
                    "status": "fail",
                    "failed_at": utc_now(),
                    "config": str(args.config),
                    "config_sha256": sha256_json(config),
                    "error": f"{type(exc).__name__}: {exc}",
                    "resume_checkpoint": (
                        str(args.resume_checkpoint) if args.resume_checkpoint is not None else None
                    ),
                    "resume_checkpoint_expected_sha256": args.resume_checkpoint_sha256,
                    "resume_checkpoint_expected_size_bytes": args.resume_checkpoint_size_bytes,
                    "resume_checkpoint_expected_optimizer_steps": args.resume_checkpoint_optimizer_steps,
                    "authorizes_longer_training": False,
                    "maximum_optimizer_steps_authorized": RUN_LOCAL_OPTIMIZER_STEPS,
                    "continuation_policy": "A failed smoke never authorizes resume, continuation, or longer training.",
                }
                atomic_json(gate_path, failure)
            raise
        return
    report = build_strict_preflight(args.config)
    atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
