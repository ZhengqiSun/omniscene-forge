#!/usr/bin/env python3
"""Pure contract helpers for the official LingBot World v2 Fast runtime."""

from __future__ import annotations
from runtime_paths import source_path

import ast
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import math
import operator
from pathlib import Path
import re
import sys
from typing import Any, Sequence


CONTRACT_VERSION = "lingbot-fast-v2-causal-runtime-contract/v1"
REPORT_SCHEMA_VERSION = "lingbot-fast-v2-causal-runtime-audit/v1"

DEFAULT_SOURCE_ROOT = Path(str(source_path('assets', 'external/lingbot-world-v2-code')))
DEFAULT_MODEL_ROOT = Path(str(source_path('assets', 'lingbot-world-v2-14b-causal-fast')))

PUBLIC_TIMESTEP_INDICES = (0, 250, 500, 750)
PUBLIC_SHIFT = 10.0
PUBLIC_CHUNK_SIZE = 4
PUBLIC_LOCAL_ATTN_SIZE = 18
PUBLIC_SINK_SIZE = 6


class ContractError(RuntimeError):
    """Raised when the official files no longer satisfy the bound contract."""


@dataclass(frozen=True)
class FrameBehavior:
    requested_frames: int
    normalized_requested_frames: int
    available_pose_frames: int | None
    normalized_available_pose_frames: int | None
    selected_input_frames: int
    pre_chunk_latent_frames: int
    usable_latent_frames: int
    dropped_latent_frames: int
    output_frames: int
    dropped_input_to_output_frames: int
    runtime_valid: bool
    warning: str | None


@dataclass(frozen=True)
class ChunkExpectation:
    chunk_id: int
    latent_start: int
    latent_end: int
    current_start: int
    current_end: int


@dataclass(frozen=True)
class CacheIndexExpectation:
    chunk_id: int
    current_start: int
    current_end: int
    global_end_before: int
    global_end_after: int
    local_start: int
    local_end: int
    evicted_tokens: int


def _positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def frame_behavior(
    requested_frames: int,
    *,
    available_pose_frames: int | None = None,
    vae_temporal_stride: int = 4,
    chunk_size: int = PUBLIC_CHUNK_SIZE,
) -> FrameBehavior:
    """Mirror official frame normalization and expose its chunk truncation.

    The official path first rounds video/pose counts down to ``4n+1``, then
    converts to VAE latents and drops the remainder that cannot fill a whole
    non-overlapping Fast chunk.
    """

    requested_frames = _positive_int("requested_frames", requested_frames)
    vae_temporal_stride = _positive_int("vae_temporal_stride", vae_temporal_stride)
    chunk_size = _positive_int("chunk_size", chunk_size)

    normalized_requested = (
        (requested_frames - 1) // vae_temporal_stride * vae_temporal_stride + 1
    )
    normalized_available: int | None = None
    if available_pose_frames is not None:
        available_pose_frames = _positive_int("available_pose_frames", available_pose_frames)
        normalized_available = (
            (available_pose_frames - 1) // vae_temporal_stride * vae_temporal_stride + 1
        )

    selected = normalized_requested
    if normalized_available is not None:
        selected = min(selected, normalized_available)
    pre_chunk_latents = (selected - 1) // vae_temporal_stride + 1
    usable_latents = pre_chunk_latents - pre_chunk_latents % chunk_size
    dropped_latents = pre_chunk_latents - usable_latents
    valid = usable_latents > 0
    output_frames = (
        (usable_latents - 1) * vae_temporal_stride + 1 if valid else 0
    )
    dropped_frames = selected - output_frames
    warning = None
    if not valid:
        warning = (
            f"{selected} selected input frames produce only {pre_chunk_latents} latent "
            f"frames, fewer than one chunk of {chunk_size}"
        )
    elif dropped_latents:
        warning = (
            f"{selected} selected input frames produce {pre_chunk_latents} latent frames; "
            f"the official non-overlapping chunk path uses {usable_latents} and decodes "
            f"{output_frames} output frames"
        )

    return FrameBehavior(
        requested_frames=requested_frames,
        normalized_requested_frames=normalized_requested,
        available_pose_frames=available_pose_frames,
        normalized_available_pose_frames=normalized_available,
        selected_input_frames=selected,
        pre_chunk_latent_frames=pre_chunk_latents,
        usable_latent_frames=usable_latents,
        dropped_latent_frames=dropped_latents,
        output_frames=output_frames,
        dropped_input_to_output_frames=dropped_frames,
        runtime_valid=valid,
        warning=warning,
    )


def official_non_overlapping_chunks(
    usable_latent_frames: int,
    *,
    frame_seqlen: int,
    chunk_size: int = PUBLIC_CHUNK_SIZE,
) -> tuple[ChunkExpectation, ...]:
    """Return the exact ``Tensor.split`` chunks and token cache offsets."""

    usable_latent_frames = _positive_int("usable_latent_frames", usable_latent_frames)
    frame_seqlen = _positive_int("frame_seqlen", frame_seqlen)
    chunk_size = _positive_int("chunk_size", chunk_size)
    if usable_latent_frames % chunk_size:
        raise ValueError(
            f"usable_latent_frames={usable_latent_frames} is not divisible by chunk_size={chunk_size}"
        )
    return tuple(
        ChunkExpectation(
            chunk_id=chunk_id,
            latent_start=latent_start,
            latent_end=latent_start + chunk_size,
            current_start=latent_start * frame_seqlen,
            current_end=(latent_start + chunk_size) * frame_seqlen,
        )
        for chunk_id, latent_start in enumerate(range(0, usable_latent_frames, chunk_size))
    )


def cache_shapes(
    *,
    batch_size: int,
    latent_height: int,
    latent_width: int,
    dim: int,
    num_heads: int,
    num_layers: int,
    sequence_parallel_size: int,
    text_sequence_length: int = 512,
    patch_size_hw: tuple[int, int] = (2, 2),
    local_attn_size: int = PUBLIC_LOCAL_ATTN_SIZE,
    usable_latent_frames: int | None = None,
) -> dict[str, Any]:
    """Compute official per-layer self/cross KV shapes without allocating them."""

    values = {
        "batch_size": batch_size,
        "latent_height": latent_height,
        "latent_width": latent_width,
        "dim": dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "sequence_parallel_size": sequence_parallel_size,
        "text_sequence_length": text_sequence_length,
        "patch_height": patch_size_hw[0],
        "patch_width": patch_size_hw[1],
    }
    values = {name: _positive_int(name, value) for name, value in values.items()}
    if values["num_heads"] % values["sequence_parallel_size"]:
        raise ValueError("num_heads must be divisible by sequence_parallel_size")
    if values["dim"] % values["num_heads"]:
        raise ValueError("dim must be divisible by num_heads")
    patch_area = values["patch_height"] * values["patch_width"]
    latent_area = values["latent_height"] * values["latent_width"]
    if latent_area % patch_area:
        raise ValueError("latent spatial area must be divisible by patch area")
    frame_seqlen = latent_area // patch_area
    if local_attn_size == -1:
        if usable_latent_frames is None:
            raise ValueError("usable_latent_frames is required for global attention")
        kv_frames = _positive_int("usable_latent_frames", usable_latent_frames)
    else:
        kv_frames = _positive_int("local_attn_size", local_attn_size)
    head_dim = values["dim"] // values["num_heads"]
    local_heads = values["num_heads"] // values["sequence_parallel_size"]
    return {
        "num_layers": values["num_layers"],
        "frame_seqlen": frame_seqlen,
        "head_dim": head_dim,
        "local_num_heads": local_heads,
        "self_kv_per_layer": [
            values["batch_size"], kv_frames * frame_seqlen, local_heads, head_dim
        ],
        "self_index_scalars_per_layer": ["global_end_index", "local_end_index"],
        "cross_kv_per_layer": [
            values["batch_size"],
            values["text_sequence_length"],
            values["num_heads"],
            head_dim,
        ],
        "cross_index_scalar_per_layer": "is_init",
    }


def cache_index_expectations(
    chunks: Sequence[ChunkExpectation],
    *,
    frame_seqlen: int,
    local_attn_size: int = PUBLIC_LOCAL_ATTN_SIZE,
    sink_size: int = PUBLIC_SINK_SIZE,
) -> tuple[CacheIndexExpectation, ...]:
    """Simulate official local-cache indices once per newly advancing chunk."""

    frame_seqlen = _positive_int("frame_seqlen", frame_seqlen)
    local_attn_size = _positive_int("local_attn_size", local_attn_size)
    sink_size = int(sink_size)
    if sink_size < 0 or sink_size >= local_attn_size:
        raise ValueError("sink_size must be non-negative and smaller than local_attn_size")
    cache_tokens = local_attn_size * frame_seqlen
    sink_tokens = sink_size * frame_seqlen
    global_end = 0
    local_end = 0
    out: list[CacheIndexExpectation] = []
    for chunk in chunks:
        num_new = chunk.current_end - chunk.current_start
        if num_new <= 0 or num_new > cache_tokens - sink_tokens:
            raise ValueError("a chunk must fit in the non-sink portion of the local cache")
        before = global_end
        evicted = 0
        if chunk.current_end > global_end and num_new + local_end > cache_tokens:
            evicted = num_new + local_end - cache_tokens
            rolled = local_end - evicted - sink_tokens
            if rolled < 0:
                raise ValueError("official cache roll would have a negative token count")
            new_local_end = local_end + chunk.current_end - global_end - evicted
        else:
            new_local_end = local_end + chunk.current_end - global_end
        local_start = new_local_end - num_new
        global_end = chunk.current_end
        local_end = new_local_end
        out.append(
            CacheIndexExpectation(
                chunk_id=chunk.chunk_id,
                current_start=chunk.current_start,
                current_end=chunk.current_end,
                global_end_before=before,
                global_end_after=global_end,
                local_start=local_start,
                local_end=local_end,
                evicted_tokens=evicted,
            )
        )
    return tuple(out)


def clean_commit_lifecycle(
    selected_timesteps: Sequence[int],
) -> tuple[dict[str, Any], ...]:
    """Describe official per-chunk forwards, including the final clean commit."""

    if not selected_timesteps:
        raise ValueError("selected_timesteps cannot be empty")
    events = [
        {
            "phase": "denoise",
            "forward_index": index,
            "timestep": int(timestep),
            "cache_position": "overwrite-current-chunk",
            "commits_clean_x0": False,
        }
        for index, timestep in enumerate(selected_timesteps)
    ]
    events.append(
        {
            "phase": "clean_commit",
            "forward_index": len(events),
            "timestep": 0,
            "cache_position": "overwrite-current-chunk",
            "commits_clean_x0": True,
        }
    )
    return tuple(events)


def canonical_latent_hw(
    *,
    image_height: int,
    image_width: int,
    max_area: int,
    vae_stride_hw: tuple[int, int] = (8, 8),
    patch_size_hw: tuple[int, int] = (2, 2),
) -> tuple[int, int]:
    """Mirror official spatial latent rounding."""

    image_height = _positive_int("image_height", image_height)
    image_width = _positive_int("image_width", image_width)
    max_area = _positive_int("max_area", max_area)
    aspect = image_height / image_width
    latent_height = round(
        math.sqrt(max_area * aspect)
        // vae_stride_hw[0]
        // patch_size_hw[0]
        * patch_size_hw[0]
    )
    latent_width = round(
        math.sqrt(max_area / aspect)
        // vae_stride_hw[1]
        // patch_size_hw[1]
        * patch_size_hw[1]
    )
    return int(latent_height), int(latent_width)


def sha256_binding(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> dict[str, Any]:
    """Bind one existing regular file by resolved path, byte size, and SHA-256."""

    path = path.resolve(strict=True)
    if not path.is_file():
        raise ContractError(f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _static_value(node: ast.AST) -> Any:
    """Evaluate only literals and simple arithmetic used in official defaults."""

    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        pass
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _static_value(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        operations = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
        }
        operation = operations.get(type(node.op))
        if operation is not None:
            return operation(_static_value(node.left), _static_value(node.right))
    raise ContractError(f"unsupported static expression: {ast.dump(node, include_attributes=False)}")


def _literal_assignment(path: Path, target: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        lhs = node.targets[0]
        dotted = None
        if isinstance(lhs, ast.Attribute) and isinstance(lhs.value, ast.Name):
            dotted = f"{lhs.value.id}.{lhs.attr}"
        if dotted == target:
            return _static_value(node.value)
    raise ContractError(f"could not find literal assignment {target} in {path}")


def _function_defaults(path: Path, class_name: str, function_name: str) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == function_name:
                names = [arg.arg for arg in item.args.args]
                defaults = [_static_value(value) for value in item.args.defaults]
                return dict(zip(names[-len(defaults):], defaults)) if defaults else {}
    raise ContractError(f"could not find {class_name}.{function_name} in {path}")


def _argparse_default(path: Path, option: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        try:
            first = ast.literal_eval(node.args[0])
        except (ValueError, TypeError):
            continue
        if first != option:
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                return _static_value(keyword.value)
    raise ContractError(f"could not find argparse default for {option} in {path}")


def _shell_flag(path: Path, option: str) -> int:
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"(?m){re.escape(option)}\s+(\d+)", text)
    if match is None:
        raise ContractError(f"could not find {option} in {path}")
    return int(match.group(1))


def derive_public_contract(source_root: Path) -> dict[str, Any]:
    """Resolve the effective Fast values from the official executable path."""

    source_root = source_root.resolve(strict=True)
    image2video = source_root / "wan/image2video.py"
    generate_py = source_root / "generate.py"
    config_py = source_root / "wan/configs/wan_i2v_A14B.py"
    shared_config_py = source_root / "wan/configs/shared_config.py"
    run_fast = source_root / "run_fast.sh"

    outer = _function_defaults(image2video, "WanI2VCausal", "generate")
    inner = _function_defaults(image2video, "WanI2VCausal", "_generate_causal_fast")
    resolved = {
        "outer_generate_timestep_indices": list(outer["timesteps_index"]),
        "shift": float(_literal_assignment(config_py, "i2v_A14B.sample_shift")),
        "chunk_size": int(_argparse_default(generate_py, "--chunk_size")),
        "local_attn_size": _shell_flag(run_fast, "--local_attn_size"),
        "sink_size": _shell_flag(run_fast, "--sink_size"),
        "num_train_timesteps": int(
            _literal_assignment(shared_config_py, "wan_shared_cfg.num_train_timesteps")
        ),
        "default_requested_frames": int(
            _literal_assignment(shared_config_py, "wan_shared_cfg.frame_num")
        ),
        "vae_temporal_stride": int(
            _literal_assignment(config_py, "i2v_A14B.vae_stride")[0]
        ),
        "patch_size": list(_literal_assignment(config_py, "i2v_A14B.patch_size")),
    }
    expected = {
        "outer_generate_timestep_indices": list(PUBLIC_TIMESTEP_INDICES),
        "shift": PUBLIC_SHIFT,
        "chunk_size": PUBLIC_CHUNK_SIZE,
        "local_attn_size": PUBLIC_LOCAL_ATTN_SIZE,
        "sink_size": PUBLIC_SINK_SIZE,
    }
    mismatches = {
        key: {"expected": value, "official": resolved[key]}
        for key, value in expected.items()
        if resolved[key] != value
    }
    if mismatches:
        raise ContractError(f"official public Fast contract drifted: {mismatches}")
    return {
        **resolved,
        "inner_fast_fallback_timestep_indices": list(inner["timesteps_index"]),
        "resolution_note": (
            "WanI2VCausal.generate passes its outer default into _generate_causal_fast, "
            "so the inner fallback is not used by the public generate path"
        ),
    }


def load_official_scheduler_class(source_root: Path):
    """Load only the official scheduler file, without importing the Wan package."""

    scheduler_path = source_root.resolve(strict=True) / "wan/utils/fm_solvers_unipc.py"
    module_name = "_lingbot_fast_v2_official_scheduler"
    spec = importlib.util.spec_from_file_location(module_name, scheduler_path)
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot load scheduler module from {scheduler_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.FlowUniPCMultistepScheduler


def derive_scheduler_selection(
    source_root: Path,
    *,
    num_train_timesteps: int,
    shift: float,
    timestep_indices: Sequence[int],
) -> dict[str, Any]:
    """Instantiate the official scheduler and extract selected timesteps/sigmas."""

    scheduler_class = load_official_scheduler_class(source_root)
    scheduler = scheduler_class(
        num_train_timesteps=num_train_timesteps,
        shift=1,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(num_train_timesteps, shift=shift)
    indices = tuple(int(index) for index in timestep_indices)
    if not indices or min(indices) < 0 or max(indices) >= len(scheduler.timesteps):
        raise ContractError(f"scheduler indices out of range: {indices}")
    return {
        "scheduler_class": scheduler_class.__name__,
        "construction": {
            "num_train_timesteps": int(num_train_timesteps),
            "initial_shift": 1,
            "use_dynamic_shifting": False,
            "set_timesteps_count": int(num_train_timesteps),
            "set_timesteps_shift": float(shift),
        },
        "selected": [
            {
                "outer_index": index,
                "timestep": int(scheduler.timesteps[index].item()),
                "sigma": float(scheduler.sigmas[index].item()),
            }
            for index in indices
        ],
        "timesteps_dtype": str(scheduler.timesteps.dtype),
        "sigmas_dtype": str(scheduler.sigmas.dtype),
        "timesteps_count": len(scheduler.timesteps),
        "sigmas_count_including_terminal": len(scheduler.sigmas),
    }


def _model_files(model_root: Path) -> tuple[Path, Path, list[Path]]:
    config_path = model_root / "config.json"
    index_path = model_root / "transformers/diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    expected_names = [f"model-{index:05d}-of-00008.safetensors" for index in range(1, 9)]
    if shard_names != expected_names:
        raise ContractError(f"index must bind exactly the canonical eight shards, got {shard_names}")
    shards = [index_path.parent / name for name in shard_names]
    return config_path, index_path, shards


def build_audit_report(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    model_root: Path = DEFAULT_MODEL_ROOT,
    requested_frames: int = 81,
    available_pose_frames: int | None = None,
    image_height: int = 480,
    image_width: int = 832,
    max_area: int = 480 * 832,
    sequence_parallel_size: int = 8,
) -> dict[str, Any]:
    """Build the complete versioned audit report without loading model weights."""

    source_root = source_root.resolve(strict=True)
    model_root = model_root.resolve(strict=True)
    public = derive_public_contract(source_root)
    scheduler = derive_scheduler_selection(
        source_root,
        num_train_timesteps=public["num_train_timesteps"],
        shift=public["shift"],
        timestep_indices=public["outer_generate_timestep_indices"],
    )
    behavior = frame_behavior(
        requested_frames,
        available_pose_frames=available_pose_frames,
        vae_temporal_stride=public["vae_temporal_stride"],
        chunk_size=public["chunk_size"],
    )
    if not behavior.runtime_valid:
        raise ContractError(behavior.warning or "frame request cannot produce a complete Fast chunk")

    model_config_path, model_index_path, shards = _model_files(model_root)
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    model_index = json.loads(model_index_path.read_text(encoding="utf-8"))
    latent_hw = canonical_latent_hw(
        image_height=image_height,
        image_width=image_width,
        max_area=max_area,
    )
    shapes = cache_shapes(
        batch_size=1,
        latent_height=latent_hw[0],
        latent_width=latent_hw[1],
        dim=int(model_config["dim"]),
        num_heads=int(model_config["num_heads"]),
        num_layers=int(model_config["num_layers"]),
        sequence_parallel_size=sequence_parallel_size,
        text_sequence_length=int(model_config["text_len"]),
        patch_size_hw=tuple(public["patch_size"][1:]),
        local_attn_size=public["local_attn_size"],
    )
    chunks = official_non_overlapping_chunks(
        behavior.usable_latent_frames,
        frame_seqlen=shapes["frame_seqlen"],
        chunk_size=public["chunk_size"],
    )
    cache_indices = cache_index_expectations(
        chunks,
        frame_seqlen=shapes["frame_seqlen"],
        local_attn_size=public["local_attn_size"],
        sink_size=public["sink_size"],
    )

    source_relpaths = [
        "generate.py",
        "run_fast.sh",
        "wan/image2video.py",
        "wan/modules/model_fast.py",
        "wan/utils/fm_solvers_unipc.py",
        "wan/configs/shared_config.py",
        "wan/configs/wan_i2v_A14B.py",
    ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "mode": "causal_fast",
        "safety": {
            "model_weights_loaded": False,
            "gpu_required": False,
            "operation": "filesystem metadata/content hashing plus CPU scheduler instantiation",
        },
        "official_roots": {"source": str(source_root), "model": str(model_root)},
        "resolved_public_cli_contract": public,
        "scheduler": scheduler,
        "frame_behavior": asdict(behavior),
        "canonical_geometry": {
            "input_image_hw": [image_height, image_width],
            "max_area": max_area,
            "latent_hw": list(latent_hw),
            "sequence_parallel_size": sequence_parallel_size,
        },
        "chunks": [asdict(item) for item in chunks],
        "cache": {
            "shapes": shapes,
            "index_expectations": [asdict(item) for item in cache_indices],
        },
        "per_chunk_lifecycle": list(
            clean_commit_lifecycle([item["timestep"] for item in scheduler["selected"]])
        ),
        "bindings": {
            "official_source_files": [
                sha256_binding(source_root / relpath) for relpath in source_relpaths
            ],
            "model_config": sha256_binding(model_config_path),
            "model_index": sha256_binding(model_index_path),
            "model_shards": [sha256_binding(path) for path in shards],
        },
        "model_index_summary": {
            "declared_tensor_bytes": int(model_index["metadata"]["total_size"]),
            "indexed_tensor_count": len(model_index["weight_map"]),
            "indexed_shard_count": len(shards),
        },
    }


def report_json(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
