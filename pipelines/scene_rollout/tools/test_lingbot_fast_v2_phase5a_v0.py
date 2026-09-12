#!/usr/bin/env python3
"""CPU-only contracts for the authorized Fast v2 Phase 5A 200-step run."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest import mock


TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from run_lingbot_fast_v2_phase3_smoke_v0 import (  # noqa: E402
    DEFAULT_CONFIG as PHASE3_CONFIG,
    evaluate_projector_gradient_gate,
    load_json,
)
from run_lingbot_fast_v2_phase4_25step_v0 import (  # noqa: E402
    DEFAULT_CONFIG as PHASE4_CONFIG,
    validate_phase4_config,
)
from run_lingbot_fast_v2_phase5a_200step_v0 import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_OUT_DIR,
    Phase5AContractError,
    execute_phase5a,
    parse_args,
    phase5a_microbatch_spec,
    validate_authorization_and_checkpoint,
    validate_phase4_source_gate,
    validate_phase5a_config,
)


class Phase5AConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_json(DEFAULT_CONFIG)
        cls.base = load_json(PHASE3_CONFIG)
        cls.phase4 = load_json(PHASE4_CONFIG)

    def test_phase5a_is_exactly_200_local_steps_from_total_step29(self) -> None:
        report = validate_phase5a_config(self.config)
        self.assertEqual(self.config["execution"]["local_optimizer_steps"], 200)
        self.assertEqual(self.config["execution"]["hard_stop_total_optimizer_steps"], 229)
        self.assertEqual(self.config["execution"]["gradient_accumulation_steps"], 2)
        self.assertEqual(self.config["output"]["maximum_local_optimizer_steps_authorized"], 200)
        self.assertFalse(self.config["output"]["authorizes_automatic_extension"])
        self.assertEqual(
            self.config["execution"]["resume_semantics"],
            "adapter-lora-projector-weights-only-with-fresh-adamw",
        )
        self.assertTrue(report["base_phase3_config"].is_absolute())

    def test_phase3_and_phase4_contracts_remain_unchanged(self) -> None:
        validate_phase4_config(self.phase4)
        self.assertEqual(self.base["execution"]["optimizer_steps"], 2)
        self.assertEqual(self.phase4["execution"]["local_optimizer_steps"], 25)
        self.assertEqual(self.phase4["execution"]["hard_stop_total_optimizer_steps"], 29)
        self.assertFalse(self.phase4["output"]["authorizes_automatic_extension"])

    def test_phase5a_cli_requires_explicit_exact_authorization(self) -> None:
        self.assertFalse(parse_args([]).execute_phase5a)
        with self.assertRaises(SystemExit):
            parse_args(["--execute-phase5a"])
        with self.assertRaises(SystemExit):
            parse_args(["--execute-phase5a", "--authorized-optimizer-steps", "199"])
        explicit = parse_args([
            "--execute-phase5a", "--authorized-optimizer-steps", "200",
            "--out-dir", str(DEFAULT_OUT_DIR),
        ])
        self.assertTrue(explicit.execute_phase5a)

    def test_phase5a_source_bindings_and_authorization_are_real(self) -> None:
        gate = validate_phase4_source_gate(self.config)
        authorization = validate_authorization_and_checkpoint(self.config)
        self.assertEqual(gate["status"], "pass")
        self.assertTrue(gate["projector_all_steps_all_ranks_nonzero"])
        self.assertFalse(gate["source_gate_authorizes_longer_training"])
        self.assertEqual(authorization["status"], "pass")
        self.assertEqual(authorization["source_checkpoint_size_bytes"], 261678092)

    def test_any_authorization_or_source_drift_fails_closed(self) -> None:
        cases = [
            ("execution", "local_optimizer_steps", 201),
            ("execution", "hard_stop_total_optimizer_steps", 230),
            ("authorization", "sha256", "0" * 64),
            ("source_gate", "sha256", "0" * 64),
            ("source_gate", "required_status", "quality_fail"),
            ("source_checkpoint", "size_bytes", 1),
            ("source_checkpoint", "sha256", "0" * 64),
            ("source_checkpoint", "config_sha256", "0" * 64),
            ("output", "authorizes_automatic_extension", True),
        ]
        for section, key, value in cases:
            with self.subTest(section=section, key=key):
                changed = copy.deepcopy(self.config)
                changed[section][key] = value
                with self.assertRaises(Phase5AContractError):
                    validate_phase5a_config(changed)

    def test_non_eight_world_fails_before_torch_or_model_boundary(self) -> None:
        args = parse_args([
            "--execute-phase5a", "--authorized-optimizer-steps", "200",
            "--run-token", "cpu-contract",
        ])
        with mock.patch.dict(
            os.environ,
            {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "WORLD_SIZE=8"):
                execute_phase5a(args, self.config)

    def test_phase5a_declares_exact_phase4_checkpoint_envelope_contract(self) -> None:
        args = parse_args([
            "--execute-phase5a", "--authorized-optimizer-steps", "200",
            "--run-token", "cpu-contract",
        ])
        with mock.patch(
            "run_lingbot_fast_v2_phase5a_200step_v0._execute_fixed_training"
        ) as training:
            execute_phase5a(args, self.config)
        contract = training.call_args.args[2]
        self.assertEqual(
            contract["source_checkpoint_kind"],
            "lingbot-fast-v2-phase4-25step-checkpoint/v1",
        )
        self.assertFalse(contract["source_checkpoint_smoke_only"])
        self.assertEqual(contract["source_checkpoint_run_local_optimizer_steps"], 25)
        self.assertEqual(
            contract["source_checkpoint_config_sha256"],
            self.config["source_checkpoint"]["config_sha256"],
        )


class Phase5AScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = load_json(PHASE3_CONFIG)

    def test_fixed_cycle_continues_without_reusing_step0_to29_records(self) -> None:
        rows = [
            phase5a_microbatch_spec(
                optimizer_step=step,
                accumulation_microbatch=micro,
                config=self.base,
            )
            for step in range(1, 201)
            for micro in range(2)
        ]
        self.assertEqual(len(rows), 400)
        self.assertEqual(
            [row["source"] for row in rows[:4]],
            ["event50h", "tier69h", "w3", "tier69h"],
        )
        self.assertEqual([row["target_chunk"] for row in rows[:4]], [3, 4, 0, 1])
        self.assertEqual([row["timestep"] for row in rows[:4]], [908, 768, 999, 967])
        self.assertEqual(rows[0]["continued_global_microbatch_index"], 58)
        self.assertEqual(rows[-1]["continued_global_microbatch_index"], 457)
        self.assertEqual(rows[0]["total_optimizer_step"], 30)
        self.assertEqual(rows[-1]["total_optimizer_step"], 229)
        self.assertEqual([row["source_ordinal"] for row in rows[:4]], [14, 29, 15, 30])
        counts = {
            source: sum(row["source"] == source for row in rows)
            for source in ("w3", "tier69h", "event50h")
        }
        self.assertEqual(counts, {"w3": 100, "tier69h": 200, "event50h": 100})
        self.assertEqual(len({row["noise_seed"] for row in rows}), 400)
        json.dumps(rows)

    def test_schedule_rejects_steps_outside_fixed_contract(self) -> None:
        for step in (0, 201):
            with self.assertRaises(ValueError):
                phase5a_microbatch_spec(
                    optimizer_step=step,
                    accumulation_microbatch=0,
                    config=self.base,
                )

    def test_projector_gate_requires_all_200_steps_on_all_ranks(self) -> None:
        reports = [
            {
                "gradients": [
                    {"state_action_projector_gradient_norm": 0.001}
                    for _step in range(200)
                ]
            }
            for _rank in range(8)
        ]
        passed = evaluate_projector_gradient_gate(
            reports,
            source_optimizer_steps=29,
            required=True,
            expected_steps=200,
            require_every_step_nonzero=True,
        )
        self.assertEqual(passed["status"], "pass")
        self.assertEqual(len(passed["activated_run_local_steps"]), 200)
        reports[7]["gradients"][99]["state_action_projector_gradient_norm"] = 0.0
        failed = evaluate_projector_gradient_gate(
            reports,
            source_optimizer_steps=29,
            required=True,
            expected_steps=200,
            require_every_step_nonzero=True,
        )
        self.assertEqual(failed["status"], "quality_fail")


if __name__ == "__main__":
    unittest.main(verbosity=2)
