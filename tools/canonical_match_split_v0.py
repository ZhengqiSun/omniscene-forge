#!/usr/bin/env python3
"""Canonical match-disjoint split. Single source of truth.

Authority: tools/mine_zero_visibility_windows_v0.py::canonical_match_split, seed 20260531.
Frozen cross-check (verified 2026-07-19 against
/mnt/data/pku/zhengqi/multiview-map-v0/output/w3_zero_visibility_20260716_v0/formal_18k_train_2k_heldout/):
  zero_visibility_mining_report_v0.json  sha256 4568a17e9d77b82bd5511746e6188be7b4f5c323dabe09dbf8903d2a9fcffc7c
  zero_visibility_heldout_exam_v0.jsonl  sha256 bdbd3931b50e7e5b0fbea4c2f15de0e17aa4ea69d12aaea9132c4a107ac7a72c
The hash function below reproduces the FROZEN_HELDOUT mapping exactly
(188-match universe: 174 train / 6 val / 8 test); _self_check() enforces this at import.
"""
import hashlib

CANONICAL_SPLIT_SEED = 20260531
CANONICAL_SPLIT_KEY = "match"

FROZEN_HELDOUT = {
    "35e1e77cb6324614b545b59507a6e382": "val", "40a32dbc30814e72afcf92f57adc348d": "val",
    "83e85253a57842d8976d02e6fc72cfd1": "val", "a09eb788e8e14f97a1495a2ac7aa8978": "val",
    "a0c859b08b6148299b5fa40964df4373": "val", "c0082d6819a9473bb762423ddb7a07c7": "val",
    "0732a78d6ae54c55b2fd185fc5771916": "test", "739f283c26c8499bbc9417df463711e3": "test",
    "7d1810b3b452407e9fd2c0fa1680ce52": "test", "a5bcca1c8f0249a5a3d44b350fff8a75": "test",
    "b6596b5c2c1840a2baee40ad6a08a793": "test", "c7a919a01cd3436f94337256ee2adcf9": "test",
    "c7c4d1deac1145fa8bd2df1bbb1416c8": "test", "fe5d0696d5af4570a31dee9748bbb383": "test",
}


def stable_bucket(text: str, modulo: int = 10_000) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulo


def match_to_split(match_id: str, seed: int = CANONICAL_SPLIT_SEED) -> str:
    if not match_id or not isinstance(match_id, str):
        raise ValueError(f"match_to_split needs a non-empty match id, got {match_id!r}")
    bucket = stable_bucket(f"{seed}|{match_id}")
    return "val" if bucket < 500 else "test" if bucket < 1000 else "train"


def record_match_id(record: dict) -> str:
    mid = record.get("game_id") or record.get("map_memory_match_id") or record.get("match_id")
    if not mid:
        raise ValueError(f"record {record.get('clip_id', '<unknown>')!r} has no game_id/map_memory_match_id/match_id")
    return str(mid)


def _self_check() -> None:
    for mid, want in FROZEN_HELDOUT.items():
        got = match_to_split(mid)
        if got != want:
            raise AssertionError(f"canonical split drift: {mid} -> {got}, frozen says {want}")


_self_check()
