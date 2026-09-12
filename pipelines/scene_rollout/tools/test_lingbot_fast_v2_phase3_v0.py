#!/usr/bin/env python3
"""CPU-only focused tests for the Fast v2 Phase 3 smoke trainer."""

from __future__ import annotations
from runtime_paths import source_path

import copy
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn as nn


TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from lingbot_fast_v2_injection_v0 import (  # noqa: E402
    CHECKPOINT_NAMESPACE,
    FastV2InjectionConfig,
    fast_v2_checkpoint_envelope,
    inject_fast_v2_native,
)
from lingbot_fast_v2_phase3_conditioning_v0 import (  # noqa: E402
    FastV2StateActionProjector,
    differentiable_all_to_all,
    differentiable_gather_forward,
    install_differentiable_sequence_parallel_collectives,
    partition_sequence_parallel,
    prepare_fast_v2_camera_chunks,
)
from run_lingbot_fast_v2_phase3_smoke_v0 import (  # noqa: E402
    CHECKPOINT_KIND,
    DEFAULT_CONFIG,
    Phase3ContractError,
    audit_compact_checkpoint_tensors,
    audit_gradient_presence,
    assert_consensus_summaries,
    assert_finite_gradients,
    assert_trainable_allowlist,
    build_resume_checkpoint_binding,
    collapse_rank_microbatch_reports,
    evaluate_projector_gradient_gate,
    execute_smoke,
    expected_cache_indices_after_chunk,
    global_microbatch_spec,
    iter_json_array,
    json_safe,
    load_json,
    load_state_projector_checkpoint,
    microbatch_forward_budget,
    parse_args,
    should_sync_microbatch,
    smoke_mix_schedule,
    validate_text_context_shape,
    validate_checkpoint_envelope,
    validate_config,
)


class TinyAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)


class TinyFastBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.self_attn = TinyAttention(dim)
        self.cross_attn = TinyAttention(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, value: torch.Tensor, **_: object) -> torch.Tensor:
        return value


class TinyFastModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 8
        self.blocks = nn.ModuleList([TinyFastBlock(8)])


def tiny_injection_config() -> FastV2InjectionConfig:
    return FastV2InjectionConfig(cond_dim=8, adapter_hidden_dim=8, lora_rank=2, lora_alpha=2.0)


class Phase3ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_json(DEFAULT_CONFIG)

    def test_locked_config_and_exact_two_step_mix(self) -> None:
        report = validate_config(self.config)
        self.assertEqual(report["release_names"], ["w3", "tier69h", "event50h"])
        mix = smoke_mix_schedule(self.config)
        self.assertEqual(mix["global_microbatches"], 4)
        self.assertEqual(mix["source_counts"], {"w3": 1, "tier69h": 2, "event50h": 1})
        self.assertEqual(mix["w3_fraction"], 0.25)
        self.assertEqual(mix["general_fraction"], 0.75)
        self.assertEqual(
            [row["source"] for row in mix["microbatches"]],
            ["w3", "tier69h", "event50h", "tier69h"],
        )
        self.assertFalse(mix["rank_replicas_count_as_samples"])

    def test_route_changes_fail_closed(self) -> None:
        cases = [
            ("execution", "world_size", 4),
            ("execution", "optimizer_steps", 3),
            ("execution", "gradient_accumulation_steps", 1),
            ("causal_runtime", "chunk_size", 5),
            ("causal_runtime", "selected_timesteps", [999, 900, 800, 700]),
            ("injection", "lora_rank", 64),
            ("injection", "external_checkpoint", "LOW.pt"),
            ("injection", "load_or_merge_low_high_weights", True),
            ("output", "authorizes_longer_training", True),
        ]
        for section, key, value in cases:
            with self.subTest(section=section, key=key):
                changed = copy.deepcopy(self.config)
                changed[section][key] = value
                with self.assertRaises(Phase3ContractError):
                    validate_config(changed)

    def test_cli_defaults_to_validation_only_and_execution_is_explicit(self) -> None:
        default = parse_args([])
        self.assertFalse(default.execute_smoke)
        explicit = parse_args(["--execute-smoke"])
        self.assertTrue(explicit.execute_smoke)
        with self.assertRaises(SystemExit):
            parse_args(["--validation-only", "--execute-smoke"])

    def test_resume_cli_requires_complete_absolute_binding(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--execute-smoke", "--resume-checkpoint", "/tmp/source.pt"])
        with self.assertRaises(SystemExit):
            parse_args([
                "--resume-checkpoint", "/tmp/source.pt",
                "--resume-checkpoint-size-bytes", "1",
                "--resume-checkpoint-sha256", "a" * 64,
                "--resume-checkpoint-optimizer-steps", "2",
            ])
        with self.assertRaises(SystemExit):
            parse_args([
                "--execute-smoke", "--resume-checkpoint", "relative.pt",
                "--resume-checkpoint-size-bytes", "1",
                "--resume-checkpoint-sha256", "a" * 64,
                "--resume-checkpoint-optimizer-steps", "2",
            ])

    def test_resume_file_binding_rejects_size_and_sha_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "compact.pt"
            checkpoint.write_bytes(b"bound-checkpoint")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            args = parse_args([
                "--execute-smoke", "--resume-checkpoint", str(checkpoint),
                "--resume-checkpoint-size-bytes", str(checkpoint.stat().st_size),
                "--resume-checkpoint-sha256", digest,
                "--resume-checkpoint-optimizer-steps", "2",
            ])
            binding = build_resume_checkpoint_binding(args)
            self.assertEqual(binding["source_optimizer_steps"], 2)
            self.assertEqual(binding["run_local_optimizer_steps"], 2)
            self.assertEqual(binding["total_optimizer_steps"], 4)
            changed = copy.copy(args)
            changed.resume_checkpoint_size_bytes += 1
            with self.assertRaisesRegex(Phase3ContractError, "size mismatch"):
                build_resume_checkpoint_binding(changed)
            changed = copy.copy(args)
            changed.resume_checkpoint_sha256 = "0" * 64
            with self.assertRaisesRegex(Phase3ContractError, "sha256 mismatch"):
                build_resume_checkpoint_binding(changed)

    def test_execute_gate_rejects_non_eight_world_before_cuda_boundary(self) -> None:
        args = parse_args(["--execute-smoke", "--run-token", "cpu-test"])
        with mock.patch.dict("os.environ", {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}, clear=False):
            with self.assertRaisesRegex(Phase3ContractError, "WORLD_SIZE=8"):
                execute_smoke(args, self.config)

    def test_all_sp_ranks_share_source_record_and_noise_seed(self) -> None:
        spec = global_microbatch_spec(
            optimizer_step=2, accumulation_microbatch=0, config=self.config
        )
        summaries = [
            {
                "source": spec["source"],
                "source_ordinal": spec["source_ordinal"],
                "clip_id": "shared-clip",
                "target_chunk": spec["target_chunk"],
                "timestep": spec["timestep"],
                "noise_seed": spec["noise_seed"],
            }
            for _rank in range(8)
        ]
        audit = assert_consensus_summaries(summaries)
        self.assertTrue(audit["all_ranks_identical"])
        self.assertEqual(audit["replica_count"], 8)
        changed = copy.deepcopy(summaries)
        changed[7]["noise_seed"] += 1
        with self.assertRaisesRegex(Phase3ContractError, "rank input mismatch"):
            assert_consensus_summaries(changed)

    def test_sync_boundary_is_last_microbatch_of_each_step(self) -> None:
        self.assertEqual(
            [
                should_sync_microbatch(accumulation_microbatch=micro, config=self.config)
                for _step in (1, 2)
                for micro in (0, 1)
            ],
            [False, True, False, True],
        )

    def test_rank_reports_collapse_to_four_global_microbatches(self) -> None:
        rows = []
        for step in (1, 2):
            for micro in (0, 1):
                spec = global_microbatch_spec(
                    optimizer_step=step, accumulation_microbatch=micro, config=self.config
                )
                rows.append({
                    **spec,
                    "clip_id": f"clip-{spec['global_microbatch_index']}",
                    "loss": 1.0 + spec["global_microbatch_index"],
                    **microbatch_forward_budget(spec),
                    "synchronized_backward": should_sync_microbatch(
                        accumulation_microbatch=micro, config=self.config
                    ),
                })
        rank_reports = [{"rank": rank, "microbatches": copy.deepcopy(rows)} for rank in range(8)]
        global_rows, audits = collapse_rank_microbatch_reports(rank_reports)
        self.assertEqual(len(global_rows), 4)
        self.assertEqual(len(audits), 4)
        self.assertTrue(all(row["rank_replicas_count_as_samples"] is False for row in global_rows))
        self.assertTrue(all(audit["all_ranks_identical"] for audit in audits))

    def test_variable_text_context_lengths(self) -> None:
        self.assertEqual(validate_text_context_shape((1, 4096)), (1, 4096))
        self.assertEqual(validate_text_context_shape((16, 4096)), (16, 4096))
        self.assertEqual(validate_text_context_shape((512, 4096)), (512, 4096))
        for shape in ((0, 4096), (513, 4096), (16, 2048), (512, 4096, 1)):
            with self.subTest(shape=shape), self.assertRaises(Phase3ContractError):
                validate_text_context_shape(shape)

    def test_prefix_commit_and_single_train_forward_budget(self) -> None:
        budgets = [
            microbatch_forward_budget(
                global_microbatch_spec(
                    optimizer_step=step, accumulation_microbatch=micro, config=self.config
                )
            )
            for step in (1, 2)
            for micro in (0, 1)
        ]
        self.assertEqual([item["prefix_clean_commits"] for item in budgets], [0, 1, 3, 4])
        self.assertTrue(all(item["train_forwards"] == 1 for item in budgets))
        self.assertTrue(all(item["backward_calls"] == 1 for item in budgets))
        self.assertTrue(all(item["target_clean_commits"] == 1 for item in budgets))
        self.assertEqual([item["total_forwards"] for item in budgets], [2, 3, 5, 6])
        self.assertEqual(expected_cache_indices_after_chunk(4)["local_end_index"], 18 * 1560)


class Phase3ConditioningTests(unittest.TestCase):
    def test_training_collectives_use_autograd_apis_and_patch_only_runtime_module(self) -> None:
        all_to_all_source = inspect.getsource(differentiable_all_to_all)
        gather_source = inspect.getsource(differentiable_gather_forward)
        self.assertIn("dist_nn.all_to_all", all_to_all_source)
        self.assertIn("dist_nn.all_gather", gather_source)
        self.assertNotIn("dist.all_to_all(", all_to_all_source)
        self.assertNotIn("dist.all_gather(", gather_source)

        fake = type("FakeSequenceParallel", (), {})()
        fake.__name__ = "wan.distributed.sequence_parallel"
        fake.all_to_all = lambda value, **_: value
        fake.gather_forward = lambda value, **_: value
        report = install_differentiable_sequence_parallel_collectives(
            fake, process_group=object()
        )
        self.assertIs(fake.all_to_all, differentiable_all_to_all)
        self.assertIs(fake.gather_forward, differentiable_gather_forward)
        self.assertTrue(report["installed"])
        self.assertTrue(report["dedicated_process_group"])
        self.assertFalse(report["official_source_modified"])
        ungrouped = type("UngroupedSequenceParallel", (), {})()
        ungrouped.__name__ = "wan.distributed.sequence_parallel"
        ungrouped.all_to_all = lambda value, **_: value
        ungrouped.gather_forward = lambda value, **_: value
        with self.assertRaisesRegex(ValueError, "dedicated sequence-parallel process group"):
            install_differentiable_sequence_parallel_collectives(ungrouped)
        with self.assertRaises(ValueError):
            install_differentiable_sequence_parallel_collectives(
                type("Wrong", (), {"__name__": "wrong"})(), process_group=object()
            )

    def test_fast_v2_camera_route_builds_five_exact_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            poses = np.repeat(np.eye(4, dtype=np.float32)[None], 81, axis=0)
            poses[:, 0, 3] = np.linspace(0.0, 8.0, 81, dtype=np.float32)
            poses[:, 1, 3] = np.linspace(0.0, 1.0, 81, dtype=np.float32)
            intrinsics = np.repeat(
                np.array([[600.0, 600.0, 416.0, 240.0]], dtype=np.float32),
                81,
                axis=0,
            )
            poses_path = root / "poses.npy"
            intrinsics_path = root / "intrinsics.npy"
            np.save(poses_path, poses)
            np.save(intrinsics_path, intrinsics)
            full, chunks = prepare_fast_v2_camera_chunks(
                poses_path,
                intrinsics_path,
                official_source_root=Path(
                    str(source_path('assets', 'external/lingbot-world-v2-code'))
                ),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        self.assertEqual(tuple(full.shape), (1, 384, 20, 60, 104))
        self.assertEqual(len(chunks), 5)
        self.assertTrue(all(tuple(chunk.shape) == (1, 384, 4, 60, 104) for chunk in chunks))
        self.assertTrue(torch.isfinite(full).all())
        self.assertTrue(torch.equal(torch.cat(chunks, dim=2), full))
        source = inspect.getsource(prepare_fast_v2_camera_chunks)
        self.assertNotIn("only_rays_d", source)
        self.assertNotIn("train_memory_dense_adapter_state_v0", source)

    def test_zero_init_state_path_and_fixed_dense_packing(self) -> None:
        projector = FastV2StateActionProjector(dense_channels=7, state_channels=3, cond_dim=16)
        self.assertTrue(projector.assert_zero_initialization()["exact"])
        dense = torch.randn(4, 7, 8, 10)
        state = torch.randn(4, 3, 8, 10)
        output = projector(dense, state, target_token_hw=(2, 5))
        self.assertEqual(tuple(output.shape), (1, 40, 16))
        expected = torch.nn.functional.interpolate(dense, size=(2, 5), mode="bilinear", align_corners=False)
        expected = expected.permute(0, 2, 3, 1).reshape(1, 40, 7)
        self.assertTrue(torch.equal(output[..., :7], expected))
        self.assertEqual(int(output[..., 7:].count_nonzero()), 0)

    def test_sequence_parallel_partition_is_exact(self) -> None:
        tokens = torch.arange(1 * 32 * 3).reshape(1, 32, 3)
        parts = [partition_sequence_parallel(tokens, rank=rank, world_size=8) for rank in range(8)]
        self.assertTrue(torch.equal(torch.cat(parts, dim=1), tokens))
        with self.assertRaisesRegex(ValueError, "not divisible"):
            partition_sequence_parallel(tokens[:, :31], rank=0, world_size=8)


class Phase3SafetyTests(unittest.TestCase):
    def _checkpoint_envelope(self) -> tuple[dict[str, object], FastV2InjectionConfig]:
        model = TinyFastModel()
        config = tiny_injection_config()
        inject_fast_v2_native(model, config, freeze_base=True)
        projector = FastV2StateActionProjector(cond_dim=8)
        envelope: dict[str, object] = {
            "kind": CHECKPOINT_KIND,
            "namespace": CHECKPOINT_NAMESPACE,
            "optimizer_steps": 2,
            "smoke_only": True,
            "authorizes_longer_training": False,
            "source_checkpoint": None,
            "config_sha256": "a" * 64,
            "fast_v2_injection": fast_v2_checkpoint_envelope(model, config),
            "fast_v2_state_action_projector": {
                name: tensor.detach().clone()
                for name, tensor in projector.state_dict().items()
            },
        }
        return envelope, config

    def test_nested_contract_dataclasses_are_json_safe(self) -> None:
        @dataclass
        class Fixture:
            path: Path
            values: tuple[int, ...]

        normalized = json_safe({"fixture": Fixture(Path("x"), (1, 2))})
        self.assertEqual(normalized, {"fixture": {"path": "x", "values": [1, 2]}})
        json.dumps(normalized)

    def test_trainable_allowlist_rejects_backbone_parameter(self) -> None:
        model = TinyFastModel()
        projector = FastV2StateActionProjector(cond_dim=8)
        inject_fast_v2_native(model, tiny_injection_config(), freeze_base=True)
        audit = assert_trainable_allowlist(model, projector)
        self.assertEqual(audit["backbone_trainable_parameters"], 0)
        base = model.blocks[0].fast_v2_base_block.self_attn.q.fast_v2_base
        base.weight.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "allowlist mismatch"):
            assert_trainable_allowlist(model, projector)

    def test_finite_gradient_gate(self) -> None:
        parameter = nn.Parameter(torch.tensor([2.0]))
        (parameter.square().sum()).backward()
        report = assert_finite_gradients([("fast_v2_test", parameter)])
        self.assertEqual(report["nonzero_gradient_tensor_count"], 1)
        parameter.grad.fill_(float("nan"))
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            assert_finite_gradients([("fast_v2_test", parameter)])

    def test_sharded_gradient_presence_audit_defers_value_checks(self) -> None:
        present = nn.Parameter(torch.tensor([2.0]))
        missing = nn.Parameter(torch.tensor([3.0]))
        (present.square().sum()).backward()
        present.grad.fill_(float("nan"))
        report = audit_gradient_presence(
            [("fast_v2_present", present), ("fast_v2_missing", missing)]
        )
        self.assertEqual(report["present_gradient_tensor_count"], 1)
        self.assertEqual(report["missing_gradient_tensor_count"], 1)
        self.assertFalse(report["per_tensor_cuda_value_sync_performed"])
        with self.assertRaisesRegex(RuntimeError, "no Fast v2 gradients are present"):
            audit_gradient_presence([("fast_v2_missing", missing)])

    def test_checkpoint_envelope_is_smoke_only_and_rejects_foreign_keys(self) -> None:
        envelope, injection_config = self._checkpoint_envelope()
        validate_checkpoint_envelope(
            envelope,
            expected_config_sha256="a" * 64,
            expected_injection_config=injection_config.to_dict(),
        )
        foreign = copy.deepcopy(envelope)
        foreign["fast_v2_injection"]["state"] = {"memory_dense_adapter_LOW.weight": torch.zeros(1)}
        with self.assertRaises(Phase3ContractError):
            validate_checkpoint_envelope(foreign)
        noncanonical = copy.deepcopy(envelope)
        name, tensor = noncanonical["fast_v2_injection"]["state"].popitem()
        parts = name.split(".")
        parts.insert(2, "_fsdp_wrapped_module")
        noncanonical["fast_v2_injection"]["state"][".".join(parts)] = tensor
        with self.assertRaisesRegex(Phase3ContractError, "non-canonical FSDP"):
            validate_checkpoint_envelope(noncanonical)
        longer = copy.deepcopy(envelope)
        longer["authorizes_longer_training"] = True
        with self.assertRaises(Phase3ContractError):
            validate_checkpoint_envelope(longer)

        wrong_namespace = copy.deepcopy(envelope)
        wrong_namespace["namespace"] = "LOW_HIGH"
        with self.assertRaises(Phase3ContractError):
            validate_checkpoint_envelope(wrong_namespace)
        with self.assertRaisesRegex(Phase3ContractError, "config"):
            validate_checkpoint_envelope(envelope, expected_config_sha256="b" * 64)
        with self.assertRaisesRegex(Phase3ContractError, "steps"):
            validate_checkpoint_envelope(envelope, expected_optimizer_steps=3)

    def test_resume_output_envelope_has_exact_provenance(self) -> None:
        envelope, injection_config = self._checkpoint_envelope()
        binding = {
            "source_checkpoint": "/absolute/source_compact.pt",
            "source_checkpoint_size_bytes": 123,
            "source_checkpoint_sha256": "b" * 64,
            "source_optimizer_steps": 2,
            "run_local_optimizer_steps": 2,
            "total_optimizer_steps": 4,
        }
        envelope.update(binding)
        envelope["optimizer_steps"] = 4
        validate_checkpoint_envelope(
            envelope,
            expected_optimizer_steps=4,
            expected_config_sha256="a" * 64,
            expected_injection_config=injection_config.to_dict(),
            expected_source_binding=binding,
        )
        changed = copy.deepcopy(envelope)
        changed["source_optimizer_steps"] = 3
        with self.assertRaises(Phase3ContractError):
            validate_checkpoint_envelope(changed, expected_optimizer_steps=4)

    def test_projector_checkpoint_load_is_strict(self) -> None:
        source = FastV2StateActionProjector(cond_dim=8)
        source.fast_v2_state_proj.weight.data.fill_(0.25)
        source.fast_v2_state_proj.bias.data.fill_(-0.5)
        target = FastV2StateActionProjector(cond_dim=8)
        state = {name: tensor.detach().clone() for name, tensor in source.state_dict().items()}
        report = load_state_projector_checkpoint(target, state)
        self.assertTrue(report["strict"])
        for name, tensor in target.state_dict().items():
            self.assertTrue(torch.equal(tensor, state[name]))
        partial = dict(state)
        partial.pop("fast_v2_state_proj.bias")
        with self.assertRaisesRegex(Phase3ContractError, "keys"):
            load_state_projector_checkpoint(target, partial)

    def test_checkpoint_storage_audit_rejects_oversized_views(self) -> None:
        envelope, _config = self._checkpoint_envelope()
        report = audit_compact_checkpoint_tensors(envelope)
        self.assertTrue(report["all_tensor_storages_compact"])
        oversized = torch.zeros(129)
        envelope["fast_v2_state_action_projector"]["fast_v2_state_proj.bias"] = oversized[:128]
        with self.assertRaisesRegex(Phase3ContractError, "not compact"):
            audit_compact_checkpoint_tensors(envelope)

    def test_resume_projector_gradient_gate_requires_activation(self) -> None:
        zero_reports = [
            {
                "gradients": [
                    {"state_action_projector_gradient_norm": 0.0},
                    {"state_action_projector_gradient_norm": 0.0},
                ]
            }
            for _rank in range(8)
        ]
        failed = evaluate_projector_gradient_gate(
            zero_reports, source_optimizer_steps=2, required=True
        )
        self.assertEqual(failed["status"], "quality_fail")
        self.assertTrue(failed["all_observed_gradient_norms_zero"])
        self.assertEqual(
            [item["total_optimizer_step"] for item in failed["per_step"]], [3, 4]
        )
        active_reports = copy.deepcopy(zero_reports)
        for report in active_reports:
            report["gradients"][1]["state_action_projector_gradient_norm"] = 0.125
        passed = evaluate_projector_gradient_gate(
            active_reports, source_optimizer_steps=2, required=True
        )
        self.assertEqual(passed["status"], "pass")
        self.assertEqual(passed["activated_run_local_steps"], [2])

    def test_large_manifest_array_reader_streams_selected_objects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            payload = {"kind": "fixture", "padding": "x" * (1024 * 1024 + 17), "samples": [{"sample_id": str(i)} for i in range(5)]}
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual([item["sample_id"] for item in iter_json_array(path)], ["0", "1", "2", "3", "4"])


if __name__ == "__main__":
    unittest.main()
