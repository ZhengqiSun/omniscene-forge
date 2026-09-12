#!/usr/bin/env python3
"""Gated real-model zero-init parity runner for LingBot World v2 causal Fast.

Without --execute-real-model this script is read-only, imports no official Wan
model code, does not initialize distributed/CUDA, and only validates the bound
fixture, public contract, official source hashes, and all model file hashes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


TOOLS_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(TOOLS_DIR))

from lingbot_fast_v2_runtime_v0 import (  # noqa: E402
    CONTRACT_VERSION,
    DEFAULT_MODEL_ROOT,
    DEFAULT_SOURCE_ROOT,
    PUBLIC_CHUNK_SIZE,
    PUBLIC_LOCAL_ATTN_SIZE,
    PUBLIC_SHIFT,
    PUBLIC_SINK_SIZE,
    PUBLIC_TIMESTEP_INDICES,
    clean_commit_lifecycle,
    derive_public_contract,
    derive_scheduler_selection,
)


FIXTURE_SCHEMA_VERSION = "lingbot-fast-v2-zero-init-parity-fixture/v1"
REPORT_SCHEMA_VERSION = "lingbot-fast-v2-zero-init-parity-report/v1"
DEFAULT_MANIFEST = TOOLS_DIR / "lingbot_fast_v2_parity_fixture_v0.json"
EXPECTED_SOURCE_REVISION = "2648877f763a06cc743bcd919936da4d25f12e7b"
MODEL_SUBFOLDER = "transformers"


class ParityContractError(RuntimeError):
    """Raised before model load when the bound parity contract has drifted."""


@dataclass(frozen=True)
class BindingValidation:
    root: str
    file_count: int
    total_size_bytes: int
    all_sha256_match: bool


def _expect_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ParityContractError(f"{name} drifted: expected {expected!r}, got {actual!r}")


def load_manifest(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ParityContractError(f"invalid parity manifest JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ParityContractError("parity manifest root must be an object")
    return value


def validate_manifest_structure(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete deterministic fixture before any official model import."""

    _expect_equal("fixture schema", manifest.get("schema_version"), FIXTURE_SCHEMA_VERSION)
    _expect_equal("runtime contract", manifest.get("contract_version"), CONTRACT_VERSION)
    _expect_equal(
        "official source revision",
        manifest.get("official_source_revision"),
        EXPECTED_SOURCE_REVISION,
    )
    public = manifest.get("public_contract")
    execution = manifest.get("execution")
    injection = manifest.get("injection")
    fixture = manifest.get("deterministic_fixture")
    thresholds = manifest.get("thresholds")
    model_config = manifest.get("expected_model_config")
    bindings = manifest.get("bindings")
    for name, value in (
        ("public_contract", public),
        ("execution", execution),
        ("injection", injection),
        ("deterministic_fixture", fixture),
        ("thresholds", thresholds),
        ("expected_model_config", model_config),
        ("bindings", bindings),
    ):
        if not isinstance(value, Mapping):
            raise ParityContractError(f"manifest {name} must be an object")

    _expect_equal("timestep indices", public.get("timestep_indices"), list(PUBLIC_TIMESTEP_INDICES))
    _expect_equal("shift", float(public.get("shift")), PUBLIC_SHIFT)
    _expect_equal("chunk size", int(public.get("chunk_size")), PUBLIC_CHUNK_SIZE)
    _expect_equal("local attention", int(public.get("local_attn_size")), PUBLIC_LOCAL_ATTN_SIZE)
    _expect_equal("sink size", int(public.get("sink_size")), PUBLIC_SINK_SIZE)
    _expect_equal("clean commit timestep", int(public.get("clean_commit_timestep")), 0)
    selected_timesteps = public.get("selected_timesteps")
    if not isinstance(selected_timesteps, list) or len(selected_timesteps) != 4:
        raise ParityContractError("selected_timesteps must contain the four public steps")

    expected_execution = {
        "mode": "causal_fast",
        "world_size": 8,
        "sequence_parallel_size": 8,
        "dtype": "bfloat16",
        "batch_size": 1,
        "chunk_count": 5,
        "latent_frames": 20,
        "latent_height": 58,
        "latent_width": 104,
        "frame_seqlen": 1508,
        "max_sequence_length": 6032,
        "text_sequence_length": 512,
        "head_dim": 128,
        "local_num_heads": 5,
    }
    _expect_equal("execution contract", dict(execution), expected_execution)
    if execution["chunk_count"] * public["chunk_size"] != execution["latent_frames"]:
        raise ParityContractError("chunk_count * chunk_size must equal latent_frames")
    expected_frame_seqlen = execution["latent_height"] * execution["latent_width"] // 4
    _expect_equal("frame sequence length", execution["frame_seqlen"], expected_frame_seqlen)
    _expect_equal(
        "maximum sequence length",
        execution["max_sequence_length"],
        execution["frame_seqlen"] * public["chunk_size"],
    )

    from lingbot_fast_v2_injection_v0 import (  # CPU-only import
        FAST_V2_COND_KEY,
        OFFICIAL_FAST_V2_LORA_TARGETS,
        FastV2InjectionConfig,
    )

    _expect_equal("Fast v2 condition key", injection.get("cond_key"), FAST_V2_COND_KEY)
    _expect_equal("Fast v2 LoRA targets", injection.get("lora_targets"), list(OFFICIAL_FAST_V2_LORA_TARGETS))
    _expect_equal("Fast v2 LoRA rank", injection.get("lora_rank"), 16)
    _expect_equal("Fast v2 LoRA alpha", float(injection.get("lora_alpha")), 16.0)
    _expect_equal("external Fast v2 checkpoint", injection.get("external_checkpoint"), None)
    config = FastV2InjectionConfig(
        cond_dim=int(injection["cond_dim"]),
        adapter_hidden_dim=int(injection["adapter_hidden_dim"]),
        wrap_first_blocks=injection["wrap_first_blocks"],
        residual_scale_init=float(injection["residual_scale_init"]),
        lora_rank=int(injection["lora_rank"]),
        lora_alpha=float(injection["lora_alpha"]),
        lora_dropout=float(injection["lora_dropout"]),
        lora_targets=tuple(injection["lora_targets"]),
        cond_key=str(injection["cond_key"]),
    )
    config.validate()

    expected_model = {
        "dim": 5120,
        "ffn_dim": 13824,
        "freq_dim": 256,
        "in_dim": 36,
        "model_type": "i2v",
        "num_heads": 40,
        "num_layers": 40,
        "out_dim": 16,
        "text_len": 512,
    }
    _expect_equal("expected model config", dict(model_config), expected_model)
    _expect_equal("head dimension", execution["head_dim"], model_config["dim"] // model_config["num_heads"])
    _expect_equal(
        "local head count",
        execution["local_num_heads"],
        model_config["num_heads"] // execution["sequence_parallel_size"],
    )

    algorithm = "torch.Generator(device='cpu').manual_seed(seed); torch.randn(shape, dtype=torch.float32)"
    _expect_equal("fixture algorithm", fixture.get("algorithm"), algorithm)
    if not isinstance(fixture.get("base_seed"), int) or fixture["base_seed"] < 0:
        raise ParityContractError("fixture base_seed must be a non-negative integer")
    if not isinstance(fixture.get("chunk_seed_stride"), int) or fixture["chunk_seed_stride"] <= 0:
        raise ParityContractError("fixture chunk_seed_stride must be positive")
    tensor_specs = fixture.get("tensors")
    if not isinstance(tensor_specs, Mapping):
        raise ParityContractError("deterministic_fixture.tensors must be an object")
    expected_shapes = {
        "context": [512, 4096],
        "initial_latent_chunk": [16, 4, 58, 104],
        "image_condition_chunk": [20, 4, 58, 104],
        "camera_condition_chunk": [1, 384, 4, 58, 104],
        "adapter_condition_chunk": [1, 6032, 128],
        "transition_noise": [16, 4, 58, 104],
    }
    expected_runtime_dtypes = {
        "context": "bfloat16",
        "initial_latent_chunk": "float32",
        "image_condition_chunk": "bfloat16",
        "camera_condition_chunk": "bfloat16",
        "adapter_condition_chunk": "bfloat16",
        "transition_noise": "float32",
    }
    _expect_equal("fixture tensor names", sorted(tensor_specs), sorted(expected_shapes))
    seed_offsets: list[int] = []
    for name, shape in expected_shapes.items():
        spec = tensor_specs[name]
        if not isinstance(spec, Mapping):
            raise ParityContractError(f"fixture tensor {name} must be an object")
        _expect_equal(f"fixture {name} shape", spec.get("shape"), shape)
        _expect_equal(
            f"fixture {name} runtime dtype",
            spec.get("runtime_dtype"),
            expected_runtime_dtypes[name],
        )
        if not isinstance(spec.get("seed_offset"), int):
            raise ParityContractError(f"fixture {name} seed_offset must be an integer")
        seed_offsets.append(spec["seed_offset"])
        scale = spec.get("scale")
        if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ParityContractError(f"fixture {name} scale must be finite and positive")
    if len(seed_offsets) != len(set(seed_offsets)):
        raise ParityContractError("fixture tensor seed offsets must be unique")
    _expect_equal(
        "adapter sequence-parallel partition",
        tensor_specs["adapter_condition_chunk"].get("sequence_parallel_partition"),
        "contiguous_dim_1_by_rank",
    )
    transition = tensor_specs["transition_noise"]
    _expect_equal("transition count", transition.get("transition_count_per_chunk"), 3)
    _expect_equal("transition seed stride", transition.get("transition_seed_stride"), 1)

    expected_threshold_names = {
        "output_max_abs",
        "output_mean_abs",
        "x0_max_abs",
        "x0_mean_abs",
        "cache_max_abs",
        "cache_mean_abs",
    }
    _expect_equal("threshold names", set(thresholds), expected_threshold_names)
    for name, value in thresholds.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ParityContractError(f"threshold {name} must be finite and non-negative")

    source_bindings = bindings.get("official_source_files")
    model_bindings = bindings.get("model_files")
    if not isinstance(source_bindings, list) or not isinstance(model_bindings, list):
        raise ParityContractError("binding lists are required")
    expected_source_paths = {
        "generate.py",
        "run_fast.sh",
        "wan/image2video.py",
        "wan/modules/model_fast.py",
        "wan/distributed/sequence_parallel.py",
        "wan/distributed/fsdp.py",
        "wan/utils/fm_solvers_unipc.py",
        "wan/configs/shared_config.py",
        "wan/configs/wan_i2v_A14B.py",
    }
    expected_model_paths = {
        "config.json",
        "transformers/diffusion_pytorch_model.safetensors.index.json",
        *{f"transformers/model-{index:05d}-of-00008.safetensors" for index in range(1, 9)},
    }
    _validate_binding_entries(source_bindings, expected_source_paths, "source")
    _validate_binding_entries(model_bindings, expected_model_paths, "model")
    return {
        "fixture_id": manifest.get("fixture_id"),
        "injection_config": config,
        "selected_timesteps": [int(value) for value in selected_timesteps],
    }


def _validate_binding_entries(
    entries: Sequence[Mapping[str, Any]], expected_paths: set[str], label: str
) -> None:
    paths: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ParityContractError(f"{label} binding entry must be an object")
        relpath = entry.get("relpath")
        size = entry.get("size_bytes")
        digest = entry.get("sha256")
        if not isinstance(relpath, str) or Path(relpath).is_absolute() or ".." in Path(relpath).parts:
            raise ParityContractError(f"unsafe {label} binding relpath: {relpath!r}")
        if not isinstance(size, int) or size <= 0:
            raise ParityContractError(f"invalid {label} binding size for {relpath}")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ParityContractError(f"invalid {label} binding sha256 for {relpath}")
        paths.append(relpath)
    if len(paths) != len(set(paths)):
        raise ParityContractError(f"duplicate {label} binding relpaths")
    _expect_equal(f"{label} binding paths", set(paths), expected_paths)


def _sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def validate_file_bindings(root: Path, entries: Sequence[Mapping[str, Any]]) -> BindingValidation:
    root = root.resolve(strict=True)
    total = 0
    for entry in entries:
        path = (root / entry["relpath"]).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ParityContractError(f"binding escapes root: {path}") from exc
        if not path.is_file():
            raise ParityContractError(f"binding is not a regular file: {path}")
        size = path.stat().st_size
        _expect_equal(f"file size {entry['relpath']}", size, entry["size_bytes"])
        actual_sha = _sha256(path)
        _expect_equal(f"SHA-256 {entry['relpath']}", actual_sha, entry["sha256"])
        total += size
    return BindingValidation(
        root=str(root),
        file_count=len(entries),
        total_size_bytes=total,
        all_sha256_match=True,
    )


def _official_git_revision(source_root: Path) -> str:
    git_dir = source_root.resolve(strict=True) / ".git"
    head_path = git_dir / "HEAD"
    head = head_path.read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref = head.removeprefix("ref: ")
        ref_path = (git_dir / ref).resolve(strict=True)
        try:
            ref_path.relative_to(git_dir.resolve(strict=True))
        except ValueError as exc:
            raise ParityContractError(f"official Git HEAD ref escapes .git: {ref}") from exc
        head = ref_path.read_text(encoding="utf-8").strip()
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise ParityContractError(f"invalid official Git HEAD revision: {head!r}")
    return head


def validate_official_contract_and_model(
    manifest: Mapping[str, Any], source_root: Path, model_root: Path
) -> dict[str, Any]:
    """Hash every bound file and validate public scheduler/model metadata."""

    structure = validate_manifest_structure(manifest)
    bindings = manifest["bindings"]
    source_binding = validate_file_bindings(source_root, bindings["official_source_files"])
    model_binding = validate_file_bindings(model_root, bindings["model_files"])
    source_revision = _official_git_revision(source_root)
    _expect_equal(
        "official Git revision", source_revision, manifest["official_source_revision"]
    )

    public = derive_public_contract(source_root)
    _expect_equal("official public indices", public["outer_generate_timestep_indices"], list(PUBLIC_TIMESTEP_INDICES))
    _expect_equal("official public shift", public["shift"], PUBLIC_SHIFT)
    scheduler = derive_scheduler_selection(
        source_root,
        num_train_timesteps=public["num_train_timesteps"],
        shift=public["shift"],
        timestep_indices=public["outer_generate_timestep_indices"],
    )
    selected = [item["timestep"] for item in scheduler["selected"]]
    _expect_equal("selected official timesteps", selected, structure["selected_timesteps"])
    lifecycle = clean_commit_lifecycle(selected)
    _expect_equal("clean commit timestep", lifecycle[-1]["timestep"], 0)
    _expect_equal("clean commit flag", lifecycle[-1]["commits_clean_x0"], True)

    config_path = model_root.resolve(strict=True) / "config.json"
    index_path = model_root.resolve(strict=True) / MODEL_SUBFOLDER / "diffusion_pytorch_model.safetensors.index.json"
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    for key, expected in manifest["expected_model_config"].items():
        _expect_equal(f"model config {key}", model_config.get(key), expected)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = sorted(set(index.get("weight_map", {}).values()))
    expected_shards = [f"model-{i:05d}-of-00008.safetensors" for i in range(1, 9)]
    _expect_equal("model index shards", shards, expected_shards)
    _expect_equal("model declared bytes", int(index["metadata"]["total_size"]), 74177331456)
    return {
        "structure": structure,
        "source_binding": source_binding.__dict__,
        "source_revision": source_revision,
        "model_binding": model_binding.__dict__,
        "public_contract": public,
        "scheduler": scheduler,
        "lifecycle": list(lifecycle),
        "model_index_tensor_count": len(index["weight_map"]),
    }


def validation_only_report(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "mode": "validation-only",
        "passed": True,
        "fixture_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": _sha256(manifest_path.resolve()),
            "fixture_id": manifest["fixture_id"],
        },
        "validated_contract": {
            "source_binding": validation["source_binding"],
            "source_revision": validation["source_revision"],
            "model_binding": validation["model_binding"],
            "public_timestep_indices": list(PUBLIC_TIMESTEP_INDICES),
            "selected_timesteps": [item["timestep"] for item in validation["scheduler"]["selected"]],
            "shift": PUBLIC_SHIFT,
            "chunk_size": PUBLIC_CHUNK_SIZE,
            "local_attn_size": PUBLIC_LOCAL_ATTN_SIZE,
            "sink_size": PUBLIC_SINK_SIZE,
            "forward_count": len(validation["lifecycle"]) * manifest["execution"]["chunk_count"],
            "clean_commit_timestep": validation["lifecycle"][-1]["timestep"],
        },
        "safety": {
            "execute_real_model_flag": False,
            "official_model_imported": False,
            "model_weights_loaded": False,
            "cuda_initialized": False,
            "distributed_initialized": False,
            "training_started": False,
            "files_written": False,
        },
        "next_action": (
            "Run only in a later exclusive eight-GPU window with torchrun --nproc_per_node=8 "
            "and the explicit --execute-real-model flag."
        ),
    }


def _metric(torch: Any, name: str, category: str, left: Any, right: Any) -> dict[str, Any]:
    if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
        raise RuntimeError(f"parity tensor metadata mismatch for {name}")
    difference = (left.detach().float() - right.detach().float()).abs()
    finite = bool(torch.isfinite(difference).all().item())
    maximum = float(difference.max().item()) if difference.numel() else 0.0
    total = float(difference.double().sum().item())
    return {
        "name": name,
        "category": category,
        "shape": list(left.shape),
        "dtype": str(left.dtype),
        "max_abs": maximum,
        "sum_abs": total,
        "element_count": int(difference.numel()),
        "exact_equal": bool(torch.equal(left, right)),
        "finite": finite,
        "rank_aggregation": (
            "partitioned"
            if name.startswith("self_cache.") and name.rsplit(".", 1)[-1] in {"k", "v"}
            else "replicated"
        ),
    }


def _cache_metrics(torch: Any, self_left: Any, self_right: Any, cross_left: Any, cross_right: Any) -> list[dict[str, Any]]:
    if len(self_left) != len(self_right) or len(cross_left) != len(cross_right):
        raise RuntimeError("disabled/enabled cache layer counts differ")
    if len(self_left) != len(cross_left):
        raise RuntimeError("self/cross cache layer counts differ")
    metrics: list[dict[str, Any]] = []
    for layer, (left, right) in enumerate(zip(self_left, self_right)):
        for key in ("k", "v", "global_end_index", "local_end_index"):
            if key not in left or key not in right:
                raise RuntimeError(f"self cache layer {layer} is missing {key}")
            metrics.append(_metric(torch, f"self_cache.{layer}.{key}", "cache", left[key], right[key]))
    for layer, (left, right) in enumerate(zip(cross_left, cross_right)):
        for key in ("k", "v", "is_init"):
            if key not in left or key not in right:
                raise RuntimeError(f"cross cache layer {layer} is missing {key}")
            metrics.append(_metric(torch, f"cross_cache.{layer}.{key}", "cache", left[key], right[key]))
    return metrics


def _caches_exact(torch: Any, self_left: Any, self_right: Any, cross_left: Any, cross_right: Any) -> bool:
    if len(self_left) != len(self_right) or len(cross_left) != len(cross_right):
        raise RuntimeError("disabled/enabled cache layer counts differ before forward")
    if len(self_left) != len(cross_left):
        raise RuntimeError("self/cross cache layer counts differ before forward")
    for left, right in zip(self_left, self_right):
        if any(not torch.equal(left[key], right[key]) for key in ("k", "v", "global_end_index", "local_end_index")):
            return False
    for left, right in zip(cross_left, cross_right):
        if any(not torch.equal(left[key], right[key]) for key in ("k", "v", "is_init")):
            return False
    return True


def _make_tensor(torch: Any, fixture: Mapping[str, Any], name: str, *, chunk: int = 0, transition: int = 0) -> Any:
    spec = fixture["tensors"][name]
    seed = fixture["base_seed"] + chunk * fixture["chunk_seed_stride"] + spec["seed_offset"]
    if name == "transition_noise":
        seed += transition * spec["transition_seed_stride"]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(spec["shape"], generator=generator, dtype=torch.float32) * float(spec["scale"])


def _partition_sequence_parallel_condition(
    condition: Any,
    *,
    rank: int,
    world_size: int,
    expected_local_tokens: int,
) -> Any:
    if condition.ndim != 3 or condition.shape[0] != 1:
        raise RuntimeError("Fast v2 SP condition must be [1,L,C]")
    token_count = int(condition.shape[1])
    if token_count % world_size:
        raise RuntimeError(
            f"Fast v2 SP condition length {token_count} is not divisible by world size {world_size}"
        )
    local_tokens = token_count // world_size
    if local_tokens != expected_local_tokens:
        raise RuntimeError(
            f"Fast v2 SP local condition length {local_tokens} != expected {expected_local_tokens}"
        )
    if rank < 0 or rank >= world_size:
        raise RuntimeError(f"SP rank {rank} is outside world size {world_size}")
    start = rank * local_tokens
    return condition[:, start:start + local_tokens].contiguous()


def _scheduler_add_noise(scheduler: Any, latent: Any, noise: Any, next_timestep: Any) -> Any:
    if latent.ndim != 4 or tuple(noise.shape) != tuple(latent.shape):
        raise RuntimeError(
            f"official Fast add_noise requires matching [C,F,H,W] tensors, got "
            f"{tuple(latent.shape)} and {tuple(noise.shape)}"
        )
    if next_timestep.ndim != 0:
        raise RuntimeError("official Fast add_noise timestep must be a scheduler scalar")
    output = scheduler.add_noise(latent, noise, next_timestep)
    if tuple(output.shape) != tuple(latent.shape):
        raise RuntimeError("scheduler add_noise changed the Fast latent shape")
    return output


def _initialize_caches(torch: Any, manifest: Mapping[str, Any], device: Any, dtype: Any) -> tuple[Any, Any]:
    execution = manifest["execution"]
    model = manifest["expected_model_config"]
    self_shape = [
        execution["batch_size"],
        execution["frame_seqlen"] * manifest["public_contract"]["local_attn_size"],
        execution["local_num_heads"],
        execution["head_dim"],
    ]
    cross_shape = [
        execution["batch_size"],
        execution["text_sequence_length"],
        model["num_heads"],
        execution["head_dim"],
    ]
    self_cache = [
        {
            "k": torch.zeros(self_shape, dtype=dtype, device=device),
            "v": torch.zeros(self_shape, dtype=dtype, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
        }
        for _ in range(model["num_layers"])
    ]
    cross_cache = [
        {
            "k": torch.zeros(cross_shape, dtype=dtype, device=device),
            "v": torch.zeros(cross_shape, dtype=dtype, device=device),
            "is_init": torch.tensor(0, dtype=torch.int32, device=device),
        }
        for _ in range(model["num_layers"])
    ]
    return self_cache, cross_cache


def _convert_x0(torch: Any, flow: Any, latent: Any, timestep: Any, scheduler: Any) -> Any:
    original_dtype = flow.dtype
    flow_d = flow.double()
    latent_d = latent.double()
    sigmas = scheduler.sigmas.double().to(flow.device)
    timesteps = scheduler.timesteps.double().to(flow.device)
    timestep_d = timestep.double().to(flow.device)
    timestep_id = torch.argmin((timesteps - timestep_d).abs())
    sigma = sigmas[timestep_id].reshape(-1, 1, 1, 1)
    return (latent_d - sigma * flow_d).to(original_dtype)


def _merge_rank_metrics(rank_events: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    if not rank_events:
        raise RuntimeError("no rank parity events were gathered")
    event_count = len(rank_events[0])
    if any(len(events) != event_count for events in rank_events):
        raise RuntimeError("rank event counts differ")
    merged_events: list[dict[str, Any]] = []
    for event_index in range(event_count):
        templates = [events[event_index] for events in rank_events]
        first = templates[0]
        for other in templates[1:]:
            for key in ("chunk_index", "phase", "forward_index_in_chunk", "timestep"):
                if other[key] != first[key]:
                    raise RuntimeError(f"rank event metadata differs at event {event_index}")
        pre_forward_cache_exact = all(event["pre_forward_cache_exact"] for event in templates)
        pre_forward_latent_exact = all(event["pre_forward_latent_exact"] for event in templates)
        rank_metrics = [event["metrics"] for event in templates]
        metric_count = len(rank_metrics[0])
        if any(len(items) != metric_count for items in rank_metrics):
            raise RuntimeError("rank metric counts differ")
        merged_metrics: list[dict[str, Any]] = []
        for metric_index in range(metric_count):
            values = [items[metric_index] for items in rank_metrics]
            base = values[0]
            metadata_keys = ("name", "category", "shape", "dtype", "rank_aggregation")
            if any(any(value[key] != base[key] for key in metadata_keys) for value in values):
                raise RuntimeError("rank metric ordering differs")
            total = sum(value["sum_abs"] for value in values)
            count = sum(value["element_count"] for value in values)
            merged_metrics.append({
                "name": base["name"],
                "category": base["category"],
                "shape_per_rank": base["shape"],
                "dtype": base["dtype"],
                "max_abs": max(value["max_abs"] for value in values),
                "mean_abs": total / count if count else 0.0,
                "exact_equal": all(value["exact_equal"] for value in values),
                "finite": all(value["finite"] for value in values),
                "element_count_across_ranks": count,
                "rank_aggregation": base["rank_aggregation"],
            })
        merged_events.append({
            "chunk_index": first["chunk_index"],
            "phase": first["phase"],
            "forward_index_in_chunk": first["forward_index_in_chunk"],
            "timestep": first["timestep"],
            "x0_observable": first["x0_observable"],
            "pre_forward_cache_exact": pre_forward_cache_exact,
            "pre_forward_latent_exact": pre_forward_latent_exact,
            "metrics": merged_metrics,
        })
    return merged_events


def _apply_thresholds(events: Sequence[Mapping[str, Any]], thresholds: Mapping[str, float]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    maxima = {"output": 0.0, "x0": 0.0, "cache": 0.0}
    mean_maxima = {"output": 0.0, "x0": 0.0, "cache": 0.0}
    metric_count = 0
    for event_index, event in enumerate(events):
        if not event["pre_forward_cache_exact"]:
            failures.append({
                "event_index": event_index,
                "name": "pre_forward_cache_state",
                "category": "cache",
                "reason": "disabled and enabled branches did not enter the forward with identical caches",
            })
        if not event["pre_forward_latent_exact"]:
            failures.append({
                "event_index": event_index,
                "name": "pre_forward_latent_input",
                "category": "output",
                "reason": "disabled and enabled branches did not enter the forward with identical latent input",
            })
        for metric in event["metrics"]:
            category = metric["category"]
            metric_count += 1
            maxima[category] = max(maxima[category], metric["max_abs"])
            mean_maxima[category] = max(mean_maxima[category], metric["mean_abs"])
            max_limit = float(thresholds[f"{category}_max_abs"])
            mean_limit = float(thresholds[f"{category}_mean_abs"])
            passed = (
                metric["finite"]
                and metric["exact_equal"]
                and metric["max_abs"] <= max_limit
                and metric["mean_abs"] <= mean_limit
            )
            if not passed:
                failures.append({
                    "event_index": event_index,
                    "name": metric["name"],
                    "category": category,
                    "max_abs": metric["max_abs"],
                    "mean_abs": metric["mean_abs"],
                    "max_threshold": max_limit,
                    "mean_threshold": mean_limit,
                    "exact_equal": metric["exact_equal"],
                    "finite": metric["finite"],
                })
    return {
        "passed": not failures,
        "metric_count": metric_count,
        "global_max_abs_by_category": maxima,
        "largest_tensor_mean_abs_by_category": mean_maxima,
        "failure_count": len(failures),
        "failures": failures,
    }


def execute_real_model(
    args: argparse.Namespace,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    structure: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Enter the explicit eight-GPU path after fixture structure validation."""

    expected_world = manifest["execution"]["world_size"]
    world_env = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_env != expected_world:
        raise ParityContractError(
            f"--execute-real-model requires torchrun WORLD_SIZE={expected_world}, got {world_env}"
        )

    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise ParityContractError("--execute-real-model requires CUDA")
    if local_rank >= torch.cuda.device_count():
        raise ParityContractError(f"LOCAL_RANK={local_rank} is not a visible CUDA device")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    validation_payload: list[Any] = [None]
    if rank == 0:
        try:
            validation_payload[0] = {
                "ok": True,
                "validation": validate_official_contract_and_model(
                    manifest, args.source_root, args.model_root
                ),
            }
        except Exception as exc:  # broadcast failure so peers do not proceed to model load
            validation_payload[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    dist.broadcast_object_list(validation_payload, src=0)
    validation_message = validation_payload[0]
    if not validation_message["ok"]:
        dist.destroy_process_group()
        raise ParityContractError(validation_message["error"])
    validation = validation_message["validation"]
    dist.barrier()

    source_root = args.source_root.resolve(strict=True)
    sys.path.insert(0, str(source_root))
    import types
    from wan.distributed.fsdp import shard_model
    from wan.distributed.sequence_parallel import sp_attn_forward_causal, sp_dit_forward_causal
    from wan.modules.model_fast import WanModelFast
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    from lingbot_fast_v2_injection_v0 import inject_fast_v2_native, set_fast_v2_enabled

    dtype = torch.bfloat16
    model = WanModelFast.from_pretrained(
        str(args.model_root.resolve()),
        subfolder=MODEL_SUBFOLDER,
        torch_dtype=dtype,
        local_attn_size=PUBLIC_LOCAL_ATTN_SIZE,
        sink_size=PUBLIC_SINK_SIZE,
    )
    model.eval().requires_grad_(False)
    for block in model.blocks:
        block.self_attn.forward = types.MethodType(sp_attn_forward_causal, block.self_attn)
    injection_report = inject_fast_v2_native(
        model, structure["injection_config"], freeze_base=True
    )
    model.forward = types.MethodType(sp_dit_forward_causal, model)
    model = shard_model(model, device_id=local_rank, use_lora=True)
    model.eval()
    device = torch.device("cuda", local_rank)

    expected_toggle_counts = {
        "injected_block_count": len(injection_report["wrapped_blocks"]),
        "lora_linear_count": len(injection_report["wrapped_lora_names"]),
    }

    def set_injection_mode(enabled: bool) -> None:
        toggle = set_fast_v2_enabled(model, enabled)
        for key, expected in expected_toggle_counts.items():
            if toggle[key] != expected:
                raise RuntimeError(
                    f"FSDP wrapper hid Fast v2 modules: {key}={toggle[key]} expected {expected}"
                )

    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(1000, shift=PUBLIC_SHIFT)
    timesteps = scheduler.timesteps[list(PUBLIC_TIMESTEP_INDICES)]
    if [int(value.item()) for value in timesteps] != structure["selected_timesteps"]:
        raise RuntimeError("scheduler drift appeared after model load")

    fixture = manifest["deterministic_fixture"]
    context = _make_tensor(torch, fixture, "context").to(device=device, dtype=dtype)
    disabled_self, disabled_cross = _initialize_caches(torch, manifest, device, dtype)
    enabled_self, enabled_cross = _initialize_caches(torch, manifest, device, dtype)
    local_events: list[dict[str, Any]] = []
    disabled_cross_first_call = True
    enabled_cross_first_call = True

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        for chunk_index in range(manifest["execution"]["chunk_count"]):
            disabled_latent = _make_tensor(
                torch, fixture, "initial_latent_chunk", chunk=chunk_index
            ).to(device=device, dtype=torch.float32)
            enabled_latent = disabled_latent.clone()
            image_condition = _make_tensor(
                torch, fixture, "image_condition_chunk", chunk=chunk_index
            ).to(device=device, dtype=dtype)
            camera_condition = _make_tensor(
                torch, fixture, "camera_condition_chunk", chunk=chunk_index
            ).to(device=device, dtype=dtype)
            full_adapter_condition = _make_tensor(
                torch, fixture, "adapter_condition_chunk", chunk=chunk_index
            )
            local_adapter_condition = _partition_sequence_parallel_condition(
                full_adapter_condition,
                rank=rank,
                world_size=expected_world,
                expected_local_tokens=(
                    manifest["execution"]["max_sequence_length"] // expected_world
                ),
            ).to(device=device, dtype=dtype)
            del full_adapter_condition
            dit_condition = {
                "c2ws_plucker_emb": (camera_condition,),
                structure["injection_config"].cond_key: local_adapter_condition,
            }
            common = {
                "context": [context],
                "seq_len": manifest["execution"]["max_sequence_length"],
                "y": [image_condition],
                "dit_cond_dict": dit_condition,
                "current_start": chunk_index
                * PUBLIC_CHUNK_SIZE
                * manifest["execution"]["frame_seqlen"],
                "max_attention_size": PUBLIC_LOCAL_ATTN_SIZE
                * manifest["execution"]["frame_seqlen"],
                "frame_seqlen": manifest["execution"]["frame_seqlen"],
            }

            for timestep_index, timestep_cpu in enumerate(timesteps):
                timestep = timestep_cpu.reshape(1).to(device=device)
                pre_forward_latent_exact = bool(
                    torch.equal(disabled_latent, enabled_latent)
                )
                pre_forward_cache_exact = _caches_exact(
                    torch,
                    disabled_self,
                    enabled_self,
                    disabled_cross,
                    enabled_cross,
                )
                set_injection_mode(False)
                disabled_output = model(
                    x=[disabled_latent],
                    t=timestep,
                    kv_cache=disabled_self,
                    crossattn_cache=disabled_cross,
                    cross_attn_first_call=disabled_cross_first_call,
                    **common,
                )[0]
                disabled_cross_first_call = False
                set_injection_mode(True)
                enabled_output = model(
                    x=[enabled_latent],
                    t=timestep,
                    kv_cache=enabled_self,
                    crossattn_cache=enabled_cross,
                    cross_attn_first_call=enabled_cross_first_call,
                    **common,
                )[0]
                enabled_cross_first_call = False
                disabled_x0 = _convert_x0(
                    torch, disabled_output, disabled_latent, timestep_cpu, scheduler
                )
                enabled_x0 = _convert_x0(
                    torch, enabled_output, enabled_latent, timestep_cpu, scheduler
                )
                metrics = [
                    _metric(torch, "output", "output", disabled_output, enabled_output),
                    _metric(torch, "x0", "x0", disabled_x0, enabled_x0),
                    *_cache_metrics(
                        torch,
                        disabled_self,
                        enabled_self,
                        disabled_cross,
                        enabled_cross,
                    ),
                ]
                local_events.append({
                    "chunk_index": chunk_index,
                    "phase": "denoise",
                    "forward_index_in_chunk": timestep_index,
                    "timestep": int(timestep_cpu.item()),
                    "x0_observable": True,
                    "pre_forward_cache_exact": pre_forward_cache_exact,
                    "pre_forward_latent_exact": pre_forward_latent_exact,
                    "metrics": metrics,
                })
                if timestep_index < len(timesteps) - 1:
                    transition_noise = _make_tensor(
                        torch,
                        fixture,
                        "transition_noise",
                        chunk=chunk_index,
                        transition=timestep_index,
                    ).to(device=device, dtype=disabled_x0.dtype)
                    next_timestep = timesteps[timestep_index + 1]
                    disabled_latent = _scheduler_add_noise(
                        scheduler, disabled_x0, transition_noise, next_timestep
                    )
                    enabled_latent = _scheduler_add_noise(
                        scheduler, enabled_x0, transition_noise, next_timestep
                    )
                else:
                    disabled_latent = disabled_x0
                    enabled_latent = enabled_x0

            timestep_zero = torch.zeros(1, device=device, dtype=timesteps.dtype)
            pre_forward_latent_exact = bool(
                torch.equal(disabled_latent, enabled_latent)
            )
            pre_forward_cache_exact = _caches_exact(
                torch,
                disabled_self,
                enabled_self,
                disabled_cross,
                enabled_cross,
            )
            if disabled_cross_first_call or enabled_cross_first_call:
                raise RuntimeError("cross-cache first-call state did not advance independently")
            set_injection_mode(False)
            disabled_output = model(
                x=[disabled_latent],
                t=timestep_zero,
                kv_cache=disabled_self,
                crossattn_cache=disabled_cross,
                cross_attn_first_call=False,
                **common,
            )[0]
            set_injection_mode(True)
            enabled_output = model(
                x=[enabled_latent],
                t=timestep_zero,
                kv_cache=enabled_self,
                crossattn_cache=enabled_cross,
                cross_attn_first_call=False,
                **common,
            )[0]
            local_events.append({
                "chunk_index": chunk_index,
                "phase": "clean_commit",
                "forward_index_in_chunk": len(timesteps),
                "timestep": 0,
                "x0_observable": False,
                "pre_forward_cache_exact": pre_forward_cache_exact,
                "pre_forward_latent_exact": pre_forward_latent_exact,
                "metrics": [
                    _metric(torch, "output", "output", disabled_output, enabled_output),
                    *_cache_metrics(
                        torch,
                        disabled_self,
                        enabled_self,
                        disabled_cross,
                        enabled_cross,
                    ),
                ],
            })

    gathered: list[Any] | None = [None for _ in range(expected_world)] if rank == 0 else None
    dist.gather_object(local_events, gathered, dst=0)
    dist.barrier()
    result: dict[str, Any] | None = None
    if rank == 0:
        if gathered is None or any(events is None for events in gathered):
            raise RuntimeError("one or more ranks did not provide parity metrics")
        events = _merge_rank_metrics(gathered)
        verdict = _apply_thresholds(events, manifest["thresholds"])
        result = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "mode": "real-model-zero-init-parity",
            "passed": verdict["passed"],
            "fixture_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": _sha256(manifest_path.resolve()),
                "fixture_id": manifest["fixture_id"],
            },
            "validated_bindings": {
                "source": validation["source_binding"],
                "source_revision": validation["source_revision"],
                "model": validation["model_binding"],
            },
            "execution": manifest["execution"],
            "public_contract": manifest["public_contract"],
            "injection": injection_report,
            "comparison_contract": {
                "same_model_instance": True,
                "disabled_and_enabled_cache_sets_initialized_identically": True,
                "branches_advanced_in_lockstep": True,
                "pre_forward_latent_inputs_checked_exactly": True,
                "pre_forward_cache_states_checked_exactly": True,
                "external_adapter_checkpoints_loaded": False,
                "low_high_lora_weights_loaded_or_merged": False,
                "output_and_x0_compared_in_float32": True,
                "cache_tensors_compared_in_float32": True,
                "indices_compared_exactly": True,
            },
            "thresholds": manifest["thresholds"],
            "verdict": verdict,
            "events": events,
            "safety": {
                "execute_real_model_flag": True,
                "world_size": expected_world,
                "training_started": False,
                "optimizer_created": False,
                "backward_called": False,
            },
        }
    dist.destroy_process_group()
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument(
        "--execute-real-model",
        action="store_true",
        help=(
            "Explicitly permit importing/loading the bound 14B model and using CUDA. "
            "Requires an eight-process torchrun launch."
        ),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        help="Explicitly write the versioned report to a file whose parent already exists.",
    )
    return parser.parse_args(argv)


def _render(report: Mapping[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = args.manifest.resolve(strict=True)
    manifest = load_manifest(manifest_path)
    structure = validate_manifest_structure(manifest)

    if args.execute_real_model:
        report = execute_real_model(args, manifest_path, manifest, structure)
        rank = int(os.environ.get("RANK", "0"))
        if rank != 0:
            return 0
        assert report is not None
    else:
        validation = validate_official_contract_and_model(
            manifest, args.source_root, args.model_root
        )
        report = validation_only_report(manifest_path, manifest, validation)

    rendered = _render(report)
    if args.out_json is not None:
        if not args.out_json.parent.is_dir():
            raise FileNotFoundError(f"--out-json parent does not exist: {args.out_json.parent}")
        args.out_json.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
