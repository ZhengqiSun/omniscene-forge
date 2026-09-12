#!/usr/bin/env python3
"""CPU-only contracts for the authorized Fast v2 Phase 4 25-step run."""

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
    parse_args as parse_phase3_args,
    validate_config as validate_phase3_config,
)
from run_lingbot_fast_v2_phase4_25step_v0 import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_OUT_DIR,
    Phase4ContractError,
    execute_phase4,
    parse_args,
    phase4_microbatch_spec,
    validate_authorization_and_checkpoint,
    validate_phase3_source_gate,
    validate_phase4_config,
)


class Phase4ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_json(DEFAULT_CONFIG)
        cls.base = load_json(PHASE3_CONFIG)

    def test_phase4_is_exactly_25_local_steps_from_total_step4(self) -> None:
        report = validate_phase4_config(self.config)
        self.assertEqual(self.config["execution"]["local_optimizer_steps"], 25)
        self.assertEqual(self.config["execution"]["hard_stop_total_optimizer_steps"], 29)
        self.assertEqual(self.config["execution"]["gradient_accumulation_steps"], 2)
        self.assertEqual(self.config["output"]["maximum_local_optimizer_steps_authorized"], 25)
        self.assertFalse(self.config["output"]["authorizes_automatic_extension"])
        self.assertTrue(report["base_phase3_config"].is_absolute())

    def test_phase3_contract_remains_two_step_smoke_only(self) -> None:
        validate_phase3_config(self.base)
        self.assertEqual(self.base["execution"]["optimizer_steps"], 2)
        self.assertEqual(self.base["output"]["maximum_optimizer_steps_authorized"], 2)
        self.assertFalse(self.base["output"]["authorizes_longer_training"])
        with self.assertRaises(SystemExit):
            parse_phase3_args(["--authorized-optimizer-steps", "25"])

    def test_phase4_cli_requires_explicit_exact_authorization(self) -> None:
        default = parse_args([])
        self.assertFalse(default.execute_phase4)
        with self.assertRaises(SystemExit):
            parse_args(["--execute-phase4"])
        with self.assertRaises(SystemExit):
            parse_args(["--execute-phase4", "--authorized-optimizer-steps", "24"])
        explicit = parse_args([
            "--execute-phase4", "--authorized-optimizer-steps", "25",
            "--out-dir", str(DEFAULT_OUT_DIR),
        ])
        self.assertTrue(explicit.execute_phase4)

    def test_phase4_source_bindings_and_authorization_are_real(self) -> None:
        gate = validate_phase3_source_gate(self.config)
        authorization = validate_authorization_and_checkpoint(self.config)
        self.assertEqual(gate["status"], "pass")
        self.assertTrue(gate["projector_all_steps_all_ranks_nonzero"])
        self.assertFalse(gate["phase3_authorizes_longer_training"])
        self.assertEqual(authorization["status"], "pass")
        self.assertEqual(authorization["source_checkpoint_size_bytes"], 261681534)

    def test_any_authorization_or_source_drift_fails_closed(self) -> None:
        cases = [
            ("execution", "local_optimizer_steps", 26),
            ("execution", "hard_stop_total_optimizer_steps", 30),
            ("authorization", "sha256", "0" * 64),
            ("source_gate", "sha256", "0" * 64),
            ("source_gate", "required_status", "quality_fail"),
            ("source_checkpoint", "size_bytes", 1),
            ("source_checkpoint", "sha256", "0" * 64),
            ("output", "authorizes_automatic_extension", True),
        ]
        for section, key, value in cases:
            with self.subTest(section=section, key=key):
                changed = copy.deepcopy(self.config)
                changed[section][key] = value
                with self.assertRaises(Phase4ContractError):
                    validate_phase4_config(changed)

    def test_non_eight_world_fails_before_torch_or_model_boundary(self) -> None:
        args = parse_args([
            "--execute-phase4", "--authorized-optimizer-steps", "25",
            "--run-token", "cpu-contract",
        ])
        with mock.patch.dict(
            os.environ,
            {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "WORLD_SIZE=8"):
                execute_phase4(args, self.config)


class Phase4ScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = load_json(PHASE3_CONFIG)

    def test_fixed_cycle_continues_after_total_step4(self) -> None:
        rows = [
            phase4_microbatch_spec(
                optimizer_step=step,
                accumulation_microbatch=micro,
                config=self.base,
            )
            for step in range(1, 26)
            for micro in range(2)
        ]
        self.assertEqual(len(rows), 50)
        self.assertEqual([row["source"] for row in rows[:4]], ["w3", "tier69h", "event50h", "tier69h"])
        self.assertEqual([row["target_chunk"] for row in rows[:4]], [0, 1, 3, 4])
        self.assertEqual([row["timestep"] for row in rows[:4]], [999, 967, 908, 768])
        self.assertEqual(rows[0]["continued_global_microbatch_index"], 8)
        self.assertEqual(rows[-1]["continued_global_microbatch_index"], 57)
        self.assertEqual(rows[0]["total_optimizer_step"], 5)
        self.assertEqual(rows[-1]["total_optimizer_step"], 29)
        counts = {
            source: sum(row["source"] == source for row in rows)
            for source in ("w3", "tier69h", "event50h")
        }
        self.assertEqual(counts, {"w3": 13, "tier69h": 25, "event50h": 12})
        self.assertEqual(len({row["noise_seed"] for row in rows}), 50)
        json.dumps(rows)

    def test_schedule_rejects_steps_outside_fixed_contract(self) -> None:
        for step in (0, 26):
            with self.assertRaises(ValueError):
                phase4_microbatch_spec(
                    optimizer_step=step,
                    accumulation_microbatch=0,
                    config=self.base,
                )

    def test_projector_gate_requires_all_25_steps_on_all_ranks(self) -> None:
        reports = [
            {
                "gradients": [
                    {"state_action_projector_gradient_norm": 0.001}
                    for _step in range(25)
                ]
            }
            for _rank in range(8)
        ]
        passed = evaluate_projector_gradient_gate(
            reports,
            source_optimizer_steps=4,
            required=True,
            expected_steps=25,
            require_every_step_nonzero=True,
        )
        self.assertEqual(passed["status"], "pass")
        self.assertEqual(len(passed["activated_run_local_steps"]), 25)
        reports[7]["gradients"][12]["state_action_projector_gradient_norm"] = 0.0
        failed = evaluate_projector_gradient_gate(
            reports,
            source_optimizer_steps=4,
            required=True,
            expected_steps=25,
            require_every_step_nonzero=True,
        )
        self.assertEqual(failed["status"], "quality_fail")


if __name__ == "__main__":
    unittest.main(verbosity=2)
