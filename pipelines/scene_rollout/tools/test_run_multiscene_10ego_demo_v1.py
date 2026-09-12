#!/usr/bin/env python3
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_multiscene_10ego_demo_v1.py")
SPEC = importlib.util.spec_from_file_location("multiscene", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

WORKER_PATH = Path(__file__).with_name("run_multiscene_10ego_worker_v1.py")
WORKER_SPEC = importlib.util.spec_from_file_location("multiscene_worker", WORKER_PATH)
WORKER = importlib.util.module_from_spec(WORKER_SPEC)
WORKER_SPEC.loader.exec_module(WORKER)


class MultisceneContractTest(unittest.TestCase):
    def test_latest_start_is_aligned_and_strictly_before_death(self):
        start = MODULE.aligned_latest_start(first_death=1118, action_length=3971)
        self.assertEqual(start, 796)
        self.assertEqual(start % 8, 4)
        self.assertLess(start + 320, 1118)

    def test_latest_start_uses_shortest_action_sequence(self):
        start = MODULE.aligned_latest_start(first_death=5000, action_length=1001)
        self.assertEqual(start, 676)
        self.assertLessEqual(start + 320, 1000)

    def test_rejects_too_short_safe_interval(self):
        self.assertIsNone(MODULE.aligned_latest_start(first_death=300, action_length=1000))

    def test_action_contract_accepts_exact_indices(self):
        actions = [{"frame_count": index} for index in range(400)]
        MODULE.validate_action_contract(actions, 4, 324, Path("actions.json"))

    def test_action_contract_rejects_index_mismatch(self):
        actions = [{"frame_count": index} for index in range(400)]
        actions[164]["frame_count"] = 999
        with self.assertRaisesRegex(RuntimeError, "frame_count/index mismatch"):
            MODULE.validate_action_contract(actions, 4, 324, Path("actions.json"))

    def test_jsonl_is_atomic_and_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            rows = [{"clip_id": "a"}, {"clip_id": "b"}]
            MODULE.write_jsonl(path, rows)
            self.assertEqual(MODULE.read_jsonl(path), rows)

    def test_exact_frame_offsets_cover_window(self):
        self.assertEqual(MODULE.EXACT_FRAME_OFFSETS[0], 0)
        self.assertEqual(MODULE.EXACT_FRAME_OFFSETS[-1], 160)
        self.assertEqual(len(MODULE.EXACT_FRAME_OFFSETS), 8)

    def test_worker_task_parser(self):
        self.assertEqual(WORKER.parse_task("scene_01_example:9"), ("scene_01_example", 9))

    def test_worker_task_parser_rejects_invalid_view(self):
        with self.assertRaises(Exception):
            WORKER.parse_task("scene_01_example:10")


if __name__ == "__main__":
    unittest.main()
