#!/usr/bin/env python3
"""CPU-only tests for Fast v2 native injection and the gated parity runner."""

from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lingbot_fast_v2_injection_v0 import (
    CHECKPOINT_KIND,
    CHECKPOINT_NAMESPACE,
    FAST_V2_COND_KEY,
    FSDP_WRAPPED_MODULE_SEGMENT,
    OFFICIAL_FAST_V2_LORA_TARGETS,
    FastV2InjectedBlock,
    FastV2InjectionConfig,
    FastV2LoRALinear,
    assert_fast_v2_zero_initialization,
    canonicalize_fast_v2_parameter_name,
    canonicalize_fast_v2_state_keys,
    fast_v2_checkpoint_envelope,
    fast_v2_trainable_parameters,
    inject_fast_v2_native,
    load_fast_v2_checkpoint_envelope,
    set_fast_v2_enabled,
    validate_official_fast_v2_topology,
)
from run_lingbot_fast_v2_zero_init_parity_v0 import (
    DEFAULT_MANIFEST,
    ParityContractError,
    _apply_thresholds,
    _merge_rank_metrics,
    _partition_sequence_parallel_condition,
    _scheduler_add_noise,
    execute_real_model,
    load_manifest,
    parse_args,
    validate_file_bindings,
    validate_manifest_structure,
)


class TinyAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.o(self.q(value) + self.k(value) + self.v(value))


class TinyOfficialFastBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.self_attn = TinyAttention(dim)
        self.cross_attn = TinyAttention(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, value: torch.Tensor, **_: object) -> torch.Tensor:
        value = value + self.self_attn(value)
        value = value + self.cross_attn(value)
        return value + self.ffn(value)


class TinyOfficialFastModel(nn.Module):
    def __init__(self, dim: int = 8, blocks: int = 2):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList([TinyOfficialFastBlock(dim) for _ in range(blocks)])

    def forward(self, value: torch.Tensor, **kwargs: object) -> torch.Tensor:
        for block in self.blocks:
            value = block(value, **kwargs)
        return value


class TinyFSDPWrapper(nn.Module):
    """Exercise nn.Module traversal used by real FSDP wrappers."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self._fsdp_wrapped_module = module

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        return self._fsdp_wrapped_module(*args, **kwargs)


def tiny_config() -> FastV2InjectionConfig:
    return FastV2InjectionConfig(
        cond_dim=3,
        adapter_hidden_dim=5,
        lora_rank=2,
        lora_alpha=2.0,
        lora_dropout=0.0,
    )


class FastV2NativeInjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260717)

    def test_topology_and_exact_official_target_names(self) -> None:
        model = TinyOfficialFastModel()
        report = validate_official_fast_v2_topology(model)
        expected = [
            f"blocks.{block}.{suffix}"
            for block in range(2)
            for suffix in OFFICIAL_FAST_V2_LORA_TARGETS
        ]
        self.assertEqual(report["checked_linear_names"], expected)

    def test_enabled_zero_init_matches_disabled_exactly(self) -> None:
        model = TinyOfficialFastModel()
        report = inject_fast_v2_native(model, tiny_config(), freeze_base=True)
        self.assertEqual(len(report["wrapped_blocks"]), 2)
        self.assertEqual(len(report["wrapped_lora_names"]), 20)
        self.assertTrue(all(isinstance(block, FastV2InjectedBlock) for block in model.blocks))
        for block in model.blocks:
            self.assertEqual(block.fast_v2_adapter_scale.ndim, 1)
            self.assertEqual(block.fast_v2_adapter_scale.numel(), 1)
        self.assertEqual(
            report["zero_initialization"]["fsdp_compatible_adapter_scale_count"],
            2,
        )

        value = torch.randn(1, 7, 8)
        condition = torch.randn(1, 7, 3)
        kwargs = {"dit_cond_dict": {FAST_V2_COND_KEY: condition}}
        model.eval()
        with torch.no_grad():
            set_fast_v2_enabled(model, False)
            disabled = model(value, **kwargs)
            set_fast_v2_enabled(model, True)
            enabled = model(value, **kwargs)
        self.assertTrue(torch.equal(disabled, enabled))
        self.assertTrue(assert_fast_v2_zero_initialization(model)["exact"])

        trainable = fast_v2_trainable_parameters(model)
        self.assertTrue(trainable)
        self.assertTrue(all(parameter.requires_grad for _, parameter in trainable))
        trainable_ids = {id(parameter) for _, parameter in trainable}
        self.assertTrue(
            all(
                (id(parameter) in trainable_ids) == parameter.requires_grad
                for parameter in model.parameters()
            )
        )

    def test_nonzero_lora_is_controlled_by_explicit_switch(self) -> None:
        model = TinyOfficialFastModel(blocks=1)
        inject_fast_v2_native(model, tiny_config())
        lora = next(module for module in model.modules() if isinstance(module, FastV2LoRALinear))
        lora.fast_v2_lora_B.data.fill_(0.25)
        value = torch.randn(1, 4, 8)
        condition = torch.randn(1, 4, 3)
        kwargs = {"dit_cond_dict": {FAST_V2_COND_KEY: condition}}
        with torch.no_grad():
            set_fast_v2_enabled(model, False)
            disabled = model(value, **kwargs)
            set_fast_v2_enabled(model, True)
            enabled = model(value, **kwargs)
        self.assertFalse(torch.equal(disabled, enabled))

    def test_enable_switch_traverses_fsdp_style_wrapper(self) -> None:
        model = TinyOfficialFastModel(blocks=1)
        report = inject_fast_v2_native(model, tiny_config())
        wrapped = TinyFSDPWrapper(model)
        toggle = set_fast_v2_enabled(wrapped, False)
        self.assertEqual(toggle["injected_block_count"], len(report["wrapped_blocks"]))
        self.assertEqual(toggle["lora_linear_count"], len(report["wrapped_lora_names"]))
        self.assertFalse(
            any(
                module.fast_v2_enabled
                for module in wrapped.modules()
                if isinstance(module, (FastV2InjectedBlock, FastV2LoRALinear))
            )
        )

    def test_enabled_adapter_requires_aligned_fast_v2_condition(self) -> None:
        model = TinyOfficialFastModel(blocks=1)
        inject_fast_v2_native(model, tiny_config())
        with self.assertRaisesRegex(ValueError, FAST_V2_COND_KEY):
            model(torch.randn(1, 4, 8))
        with self.assertRaisesRegex(ValueError, "does not align"):
            model(
                torch.randn(1, 4, 8),
                dit_cond_dict={FAST_V2_COND_KEY: torch.randn(1, 5, 3)},
            )

    def test_checkpoint_envelope_is_strict_and_rejects_foreign_state(self) -> None:
        model = TinyOfficialFastModel(blocks=1)
        config = tiny_config()
        inject_fast_v2_native(model, config)
        first_name, first_parameter = fast_v2_trainable_parameters(model)[0]
        oversized_storage = torch.zeros(first_parameter.numel() + 128)
        first_parameter.data = oversized_storage[: first_parameter.numel()].view_as(
            first_parameter
        )
        self.assertGreater(
            first_parameter.untyped_storage().nbytes(),
            first_parameter.numel() * first_parameter.element_size(),
        )
        envelope = fast_v2_checkpoint_envelope(model, config)
        self.assertEqual(envelope["kind"], CHECKPOINT_KIND)
        self.assertEqual(envelope["namespace"], CHECKPOINT_NAMESPACE)
        self.assertTrue(envelope["state"])
        compact = envelope["state"][first_name]
        self.assertEqual(
            compact.untyped_storage().nbytes(), compact.numel() * compact.element_size()
        )
        self.assertTrue(all("memory_dense" not in name for name in envelope["state"]))
        scale_state = {
            name: value
            for name, value in envelope["state"].items()
            if name.endswith(".fast_v2_adapter_scale")
        }
        self.assertEqual(len(scale_state), 1)
        self.assertTrue(
            all(value.ndim == 1 and value.numel() == 1 for value in scale_state.values())
        )
        load_fast_v2_checkpoint_envelope(model, envelope, config)

        foreign = copy.deepcopy(envelope)
        foreign["kind"] = "memory_dense_adapter_checkpoint_v0"
        foreign["namespace"] = "LOW_HIGH"
        with self.assertRaisesRegex(ValueError, "not a Fast v2"):
            load_fast_v2_checkpoint_envelope(model, foreign, config)

        partial = copy.deepcopy(envelope)
        partial["state"].pop(next(iter(partial["state"])))
        with self.assertRaisesRegex(ValueError, "keys are not exact"):
            load_fast_v2_checkpoint_envelope(model, partial, config)

        scalar_scale = copy.deepcopy(envelope)
        scale_name = next(
            name
            for name in scalar_scale["state"]
            if name.endswith(".fast_v2_adapter_scale")
        )
        scalar_scale["state"][scale_name] = scalar_scale["state"][scale_name].squeeze(0)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            load_fast_v2_checkpoint_envelope(model, scalar_scale, config)

    def test_fsdp_style_checkpoint_export_is_canonical_and_strictly_loadable(self) -> None:
        model = TinyOfficialFastModel(blocks=1)
        config = tiny_config()
        inject_fast_v2_native(model, config)
        for index, (_name, parameter) in enumerate(fast_v2_trainable_parameters(model), 1):
            parameter.data.fill_(index / 100.0)
        model.blocks[0] = TinyFSDPWrapper(model.blocks[0])
        wrapped_names = [name for name, _parameter in fast_v2_trainable_parameters(model)]
        self.assertTrue(wrapped_names)
        self.assertTrue(
            all(
                FSDP_WRAPPED_MODULE_SEGMENT in name.split(".")
                for name in wrapped_names
            )
        )

        envelope = fast_v2_checkpoint_envelope(model, config)
        self.assertTrue(
            all(
                FSDP_WRAPPED_MODULE_SEGMENT not in name.split(".")
                for name in envelope["state"]
            )
        )
        fresh = TinyOfficialFastModel(blocks=1)
        inject_fast_v2_native(fresh, config)
        fresh_names = {name for name, _parameter in fast_v2_trainable_parameters(fresh)}
        self.assertEqual(set(envelope["state"]), fresh_names)
        load_fast_v2_checkpoint_envelope(fresh, envelope, config)
        for name, parameter in fast_v2_trainable_parameters(fresh):
            self.assertTrue(torch.equal(parameter.detach().cpu(), envelope["state"][name]))

    def test_canonical_export_removes_only_exact_segments_and_rejects_collision(self) -> None:
        self.assertEqual(
            canonicalize_fast_v2_parameter_name(
                "blocks.0._fsdp_wrapped_module.fast_v2_adapter_scale"
            ),
            "blocks.0.fast_v2_adapter_scale",
        )
        retained = "blocks.0.prefix_fsdp_wrapped_module.fast_v2_adapter_scale"
        self.assertEqual(canonicalize_fast_v2_parameter_name(retained), retained)
        with self.assertRaisesRegex(ValueError, "canonical checkpoint key collision"):
            canonicalize_fast_v2_state_keys({
                "blocks.0.fast_v2_adapter_scale": torch.ones(1),
                "blocks.0._fsdp_wrapped_module.fast_v2_adapter_scale": torch.zeros(1),
            })

    def test_topology_drift_fails_closed(self) -> None:
        model = TinyOfficialFastModel(blocks=1)
        model.blocks[0].ffn[2] = nn.Identity()
        with self.assertRaisesRegex(TypeError, "must be Linear"):
            validate_official_fast_v2_topology(model)


class FastV2ParityContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest(DEFAULT_MANIFEST)

    def test_default_manifest_is_complete_and_deterministic(self) -> None:
        structure = validate_manifest_structure(self.manifest)
        self.assertEqual(structure["selected_timesteps"], [999, 967, 908, 768])
        self.assertEqual(structure["injection_config"].cond_key, FAST_V2_COND_KEY)
        self.assertEqual(structure["injection_config"].lora_rank, 16)
        self.assertEqual(structure["injection_config"].lora_alpha, 16.0)
        self.assertEqual(self.manifest["injection"]["lora_rank"], 16)
        self.assertEqual(self.manifest["injection"]["lora_alpha"], 16.0)
        self.assertIsNone(self.manifest["injection"]["external_checkpoint"])
        self.assertEqual(self.manifest["execution"]["chunk_count"], 5)
        self.assertEqual(self.manifest["public_contract"]["clean_commit_timestep"], 0)

    def test_contract_and_fixture_drift_fail_closed(self) -> None:
        drifted = copy.deepcopy(self.manifest)
        drifted["public_contract"]["shift"] = 5.0
        with self.assertRaisesRegex(ParityContractError, "shift drifted"):
            validate_manifest_structure(drifted)

        drifted = copy.deepcopy(self.manifest)
        drifted["deterministic_fixture"]["tensors"]["initial_latent_chunk"]["shape"][1] = 3
        with self.assertRaisesRegex(ParityContractError, "shape drifted"):
            validate_manifest_structure(drifted)

        drifted = copy.deepcopy(self.manifest)
        drifted["injection"]["external_checkpoint"] = "LOW.pt"
        with self.assertRaisesRegex(ParityContractError, "external Fast v2 checkpoint"):
            validate_manifest_structure(drifted)

    def test_file_sha_and_size_drift_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "bound.bin"
            path.write_bytes(b"fast-v2")
            entry = {
                "relpath": "bound.bin",
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            result = validate_file_bindings(root, [entry])
            self.assertTrue(result.all_sha256_match)
            path.write_bytes(b"drift-v2")
            with self.assertRaisesRegex(ParityContractError, "file size|SHA-256"):
                validate_file_bindings(root, [entry])

    def test_real_model_path_requires_explicit_flag_and_eight_processes(self) -> None:
        args = parse_args([])
        self.assertFalse(args.execute_real_model)
        structure = validate_manifest_structure(self.manifest)
        fake_args = types.SimpleNamespace(
            source_root=Path("/does/not/matter"),
            model_root=Path("/does/not/matter"),
        )
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=False):
            with self.assertRaisesRegex(ParityContractError, "WORLD_SIZE=8"):
                execute_real_model(fake_args, DEFAULT_MANIFEST, self.manifest, structure)

    def test_rank_metrics_have_exact_max_mean_and_thresholds(self) -> None:
        metric_a = {
            "name": "output",
            "category": "output",
            "shape": [2],
            "dtype": "torch.float32",
            "max_abs": 0.0,
            "sum_abs": 0.0,
            "element_count": 2,
            "exact_equal": True,
            "finite": True,
            "rank_aggregation": "replicated",
        }
        event = {
            "chunk_index": 0,
            "phase": "denoise",
            "forward_index_in_chunk": 0,
            "timestep": 999,
            "x0_observable": True,
            "pre_forward_cache_exact": True,
            "pre_forward_latent_exact": True,
            "metrics": [metric_a],
        }
        merged = _merge_rank_metrics([[copy.deepcopy(event)], [copy.deepcopy(event)]])
        self.assertEqual(merged[0]["metrics"][0]["max_abs"], 0.0)
        self.assertEqual(merged[0]["metrics"][0]["mean_abs"], 0.0)
        thresholds = {
            "output_max_abs": 0.0,
            "output_mean_abs": 0.0,
            "x0_max_abs": 0.0,
            "x0_mean_abs": 0.0,
            "cache_max_abs": 0.0,
            "cache_mean_abs": 0.0,
        }
        verdict = _apply_thresholds(merged, thresholds)
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["metric_count"], 1)

        shard_a = copy.deepcopy(event)
        shard_b = copy.deepcopy(event)
        for shard in (shard_a, shard_b):
            shard["metrics"][0].update({
                "name": "self_cache.0.k",
                "category": "cache",
                "rank_aggregation": "partitioned",
                "exact_equal": False,
            })
        shard_a["metrics"][0].update({"max_abs": 1.0, "sum_abs": 2.0})
        shard_b["metrics"][0].update({"max_abs": 3.0, "sum_abs": 6.0})
        cache_metric = _merge_rank_metrics([[shard_a], [shard_b]])[0]["metrics"][0]
        self.assertEqual(cache_metric["max_abs"], 3.0)
        self.assertEqual(cache_metric["mean_abs"], 2.0)
        self.assertEqual(cache_metric["element_count_across_ranks"], 4)
        self.assertEqual(cache_metric["rank_aggregation"], "partitioned")

        mismatched = copy.deepcopy(shard_b)
        mismatched["metrics"][0]["shape"] = [3]
        with self.assertRaisesRegex(RuntimeError, "rank metric ordering differs"):
            _merge_rank_metrics([[shard_a], [mismatched]])

    def test_sp_partition_and_scheduler_shape_contracts(self) -> None:
        condition = torch.arange(1 * 16 * 3).reshape(1, 16, 3)
        local = _partition_sequence_parallel_condition(
            condition, rank=2, world_size=4, expected_local_tokens=4
        )
        self.assertTrue(torch.equal(local, condition[:, 8:12]))
        with self.assertRaisesRegex(RuntimeError, "not divisible"):
            _partition_sequence_parallel_condition(
                condition[:, :15], rank=0, world_size=4, expected_local_tokens=4
            )

        class Scheduler:
            @staticmethod
            def add_noise(latent: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
                self.assertEqual(timestep.ndim, 0)
                return latent + noise

        latent = torch.zeros(2, 4, 3, 5)
        noise = torch.ones_like(latent)
        output = _scheduler_add_noise(Scheduler(), latent, noise, torch.tensor(7))
        self.assertEqual(tuple(output.shape), tuple(latent.shape))
        with self.assertRaisesRegex(RuntimeError, "matching"):
            _scheduler_add_noise(Scheduler(), latent, noise[:, :3], torch.tensor(7))


if __name__ == "__main__":
    unittest.main(verbosity=2)
