#!/usr/bin/env python3
"""Zero-init parity gate scaffold for LingBot Fast v2 adapters and LoRA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


SCHEMA_VERSION = "lingbot-fast-v2-zero-init-parity-gate/v1"


def _binding(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def run_contract_only(tools_dir: Path) -> dict[str, Any]:
    """Instantiate only tiny CPU helpers and prove their initialized deltas are zero."""

    import torch
    import torch.nn as nn

    sys.path.insert(0, str(tools_dir.resolve()))
    from memory_dense_state_adapter_v0 import StateTokenProjector
    from memory_dense_wan_adapter_v0 import LoRALinear, ZeroInitMemoryDenseResidualAdapter

    torch.manual_seed(20260717)
    hidden = torch.randn(2, 5, 7, device="cpu")
    cond = torch.randn(2, 5, 3, device="cpu")
    adapter = ZeroInitMemoryDenseResidualAdapter(7, 3, 11, "cond_gated").cpu().eval()
    with torch.no_grad():
        adapter_output = adapter(hidden, cond)
    adapter_exact = torch.equal(adapter_output, hidden)
    adapter_out_zero = bool(torch.count_nonzero(adapter.out.weight) == 0) and bool(
        torch.count_nonzero(adapter.out.bias) == 0
    )

    base = nn.Linear(7, 9).cpu().eval()
    lora = LoRALinear(base, rank=2, alpha=2, dropout=0.0).cpu().eval()
    lora_input = torch.randn(2, 5, 7, device="cpu")
    with torch.no_grad():
        base_output = base(lora_input)
        lora_output = lora(lora_input)
    lora_exact = torch.equal(lora_output, base_output)
    lora_b_zero = bool(torch.count_nonzero(lora.lora_B) == 0)

    state = StateTokenProjector(state_channels=3, cond_dim=7).cpu().eval()
    with torch.no_grad():
        state_output = state(torch.randn(2, 3, 8, 8), target_token_hw=(2, 2))
    state_zero = bool(torch.count_nonzero(state_output) == 0)
    state_params_zero = bool(torch.count_nonzero(state.proj.weight) == 0) and bool(
        torch.count_nonzero(state.proj.bias) == 0
    )

    checks = {
        "residual_adapter_output_projection_zero": adapter_out_zero,
        "residual_adapter_exact_identity": adapter_exact,
        "lora_B_zero": lora_b_zero,
        "lora_exact_base_parity": lora_exact,
        "state_projector_parameters_zero": state_params_zero,
        "state_projector_output_zero": state_zero,
        "all_parameters_and_inputs_on_cpu": all(
            tensor.device.type == "cpu"
            for tensor in (
                hidden,
                cond,
                adapter.out.weight,
                lora.lora_A,
                lora.lora_B,
                state.proj.weight,
            )
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "contract-only",
        "passed": all(checks.values()),
        "checks": checks,
        "safety": {"gpu_used": False, "official_model_loaded": False, "weights_loaded": False},
        "bound_helpers": [
            _binding(tools_dir / "memory_dense_wan_adapter_v0.py"),
            _binding(tools_dir / "memory_dense_state_adapter_v0.py"),
        ],
        "future_real_model_parity": {
            "implemented": False,
            "required_comparison": (
                "same initialized Fast model/input/cache state with adapters disabled versus "
                "zero-initialized adapters enabled, including per-step cache tensors and output"
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract-only",
        action="store_true",
        required=True,
        help="Run tiny CPU initialization checks only; never load the official model or use a GPU.",
    )
    parser.add_argument("--tools-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_contract_only(args.tools_dir)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
