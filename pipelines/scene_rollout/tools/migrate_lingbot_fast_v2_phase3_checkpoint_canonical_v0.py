#!/usr/bin/env python3
"""One-time CPU-only canonical migration for the retry5 Phase 3 checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping


TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(TOOLS_DIR))

SOURCE = PROJECT_ROOT / (
    "output/lingbot_fast_v2_phase3_two_step_smoke_gradnormfix_retry5_v0/"
    "checkpoints/lingbot_fast_v2_phase3_smoke_step_000002_compact.pt"
)
SOURCE_SIZE_BYTES = 261707952
SOURCE_SHA256 = "d21c0193751423ff4a86bad9be34b68498de6b1f9d41b8b54e0dac50287dffd2"
OUTPUT = SOURCE.with_name(SOURCE.stem + "_canonical.pt")
REPORT = SOURCE.parents[1] / "FAST_V2_PHASE3_CHECKPOINT_CANONICAL_MIGRATION_v0.json"
MIGRATION_SCHEMA = "lingbot-fast-v2-phase3-canonical-checkpoint-migration/v1"


class MigrationError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def exact_equal(left: Any, right: Any, torch: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(exact_equal(left[key], right[key], torch) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(exact_equal(a, b, torch) for a, b in zip(left, right))
        )
    return left == right


def metadata_without_injection_state(envelope: Mapping[str, Any]) -> dict[str, Any]:
    injection = envelope["fast_v2_injection"]
    return {
        **{key: value for key, value in envelope.items() if key != "fast_v2_injection"},
        "fast_v2_injection": {
            key: value for key, value in injection.items() if key != "state"
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    args.report = args.report.resolve()
    if args.source == args.output:
        parser.error("migration output must not overwrite source")
    return args


def migrate(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from lingbot_fast_v2_injection_v0 import (
        FSDP_WRAPPED_MODULE_SEGMENT,
        canonicalize_fast_v2_parameter_name,
        canonicalize_fast_v2_state_keys,
    )
    from run_lingbot_fast_v2_phase3_smoke_v0 import (
        audit_compact_checkpoint_tensors,
        atomic_json,
        sha256_file,
        validate_checkpoint_envelope,
    )

    if args.source != SOURCE.resolve():
        raise MigrationError(f"source path is locked to {SOURCE.resolve()}")
    if not args.source.is_file():
        raise MigrationError(f"source checkpoint is missing: {args.source}")
    if args.output.exists() or args.report.exists():
        raise MigrationError("refusing to overwrite canonical checkpoint or migration report")
    source_size = args.source.stat().st_size
    if source_size != SOURCE_SIZE_BYTES:
        raise MigrationError(
            f"source size mismatch: expected={SOURCE_SIZE_BYTES} actual={source_size}"
        )
    source_sha256 = sha256_file(args.source)
    if source_sha256 != SOURCE_SHA256:
        raise MigrationError(
            f"source sha256 mismatch: expected={SOURCE_SHA256} actual={source_sha256}"
        )

    envelope = torch.load(args.source, map_location="cpu", weights_only=True)
    if not isinstance(envelope, Mapping):
        raise MigrationError("source checkpoint root must be a mapping")
    injection = envelope.get("fast_v2_injection")
    if not isinstance(injection, Mapping) or not isinstance(injection.get("state"), Mapping):
        raise MigrationError("source lacks nested fast_v2_injection.state")
    source_state = injection["state"]
    source_metadata = metadata_without_injection_state(envelope)
    renamed_keys = [
        {
            "source": source_name,
            "canonical": canonicalize_fast_v2_parameter_name(source_name),
        }
        for source_name in source_state
    ]
    canonical_state = canonicalize_fast_v2_state_keys(source_state)
    if len(canonical_state) != len(source_state):
        raise MigrationError("canonical state tensor count drifted")
    if any(
        FSDP_WRAPPED_MODULE_SEGMENT in name for name in canonical_state
    ):
        raise MigrationError("canonical state retained _fsdp_wrapped_module")
    renamed_count = sum(item["source"] != item["canonical"] for item in renamed_keys)
    if renamed_count == 0:
        raise MigrationError("source checkpoint contains no FSDP wrapper segments to migrate")

    injection["state"] = canonical_state
    if not exact_equal(source_metadata, metadata_without_injection_state(envelope), torch):
        raise MigrationError("outer or nested injection metadata changed during migration")
    validate_checkpoint_envelope(
        envelope,
        expected_optimizer_steps=int(envelope["optimizer_steps"]),
        expected_config_sha256=str(envelope["config_sha256"]),
        expected_injection_config=injection["config"],
    )
    pre_save_compaction = audit_compact_checkpoint_tensors(envelope)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    if temporary.exists():
        raise MigrationError(f"refusing stale migration temporary file: {temporary}")
    try:
        torch.save(envelope, temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=True)
        validate_checkpoint_envelope(
            reloaded,
            expected_optimizer_steps=int(envelope["optimizer_steps"]),
            expected_config_sha256=str(envelope["config_sha256"]),
            expected_injection_config=injection["config"],
        )
        post_load_compaction = audit_compact_checkpoint_tensors(reloaded)
        reloaded_state = reloaded["fast_v2_injection"]["state"]
        tensor_values_exact = all(
            exact_equal(
                source_state[item["source"]],
                reloaded_state[item["canonical"]],
                torch,
            )
            for item in renamed_keys
        )
        metadata_exact = exact_equal(
            source_metadata, metadata_without_injection_state(reloaded), torch
        )
        if not tensor_values_exact or not metadata_exact:
            raise MigrationError("serialized canonical checkpoint is not exact")
        if any(FSDP_WRAPPED_MODULE_SEGMENT in name for name in reloaded_state):
            raise MigrationError("serialized canonical checkpoint retained wrapper segment")
        temporary.replace(args.output)
    finally:
        if temporary.exists():
            temporary.unlink()

    if sha256_file(args.source) != SOURCE_SHA256:
        raise MigrationError("source checkpoint changed during migration")
    report = {
        "schema": MIGRATION_SCHEMA,
        "status": "pass",
        "completed_at": utc_now(),
        "source": str(args.source),
        "source_bytes": source_size,
        "source_sha256": source_sha256,
        "canonical": str(args.output),
        "canonical_bytes": args.output.stat().st_size,
        "canonical_sha256": sha256_file(args.output),
        "source_preserved": True,
        "only_nested_fast_v2_injection_state_keys_changed": True,
        "outer_and_nested_injection_metadata_exact_equal": metadata_exact,
        "tensor_values_exact_equal": tensor_values_exact,
        "canonical_collision_count": 0,
        "source_tensor_count": len(source_state),
        "canonical_tensor_count": len(canonical_state),
        "renamed_key_count": renamed_count,
        "residual_fsdp_wrapper_segment_count": 0,
        "pre_save_compaction": pre_save_compaction,
        "post_load_compaction": post_load_compaction,
        "strict_outer_envelope_validation": "pass",
        "renamed_keys": renamed_keys,
    }
    atomic_json(args.report, report)
    return report


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = migrate(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
