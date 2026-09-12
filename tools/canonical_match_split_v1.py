#!/usr/bin/env python3
"""Canonical match-disjoint split v1: D16 heldout expansion (2026-08-11).

Extends v0 (6 val / 8 test / 174 train over the 188-match pool universe) to
15 val / 15 test / 158 train by promoting 16 named train matches. v0
assignments are untouched (segment extension, never threshold raising - see
research_notes/ablation_design_20260811/D16_split_expansion_audit.md).

Expansion rule (deterministic, frozen): among the 188 distinct game_ids of the
tier69h+event50h v2 cache manifests, take the train matches (bucket >= 1000)
in ascending stable_bucket order; the lowest 9 become val, the next 7 become
test. Frozen-list extension was chosen over threshold extension deliberately:
the raw dataset holds 1336 game dirs and only 188 are pooled - a pure
threshold band calibrated to hit 9+7 on the pool would silently sweep ~10x
unpooled matches into heldout for any future mining. The blast radius of this
file is exactly the 16 matches named below.

Calibration artefact: $MG/_claude_out/ablation_20260811/split_v1_calibration.json

Contamination note (protocol-binding): models with pre-v1 training lineage
(taskB, bigrun, anchor_ft) SAW the 16 promoted matches during training. Any
comparison against those checkpoints must use the v0 heldout (FROZEN_HELDOUT)
only; from-scratch v1-split arms are clean on all 30.
"""
from canonical_match_split_v0 import (  # noqa: F401  (re-exports are part of the API)
    CANONICAL_SPLIT_KEY,
    CANONICAL_SPLIT_SEED,
    FROZEN_HELDOUT,
    match_to_split,
    record_match_id,
    stable_bucket,
)

SPLIT_VERSION = "canonical_match_split_v1_20260811"

EXPANSION_HELDOUT = {
    "3b903c92c34342169a9c88f1639fd7ad": "val",   # bucket 1136
    "b712f0a8d3ec495a96b6a18fd9807027": "val",   # bucket 1251
    "767e388468f84e12a689f0533515ec48": "val",   # bucket 1315
    "a2ed74d8d10e492fba7730ed31d26a2a": "val",   # bucket 1358
    "b926063497d64735961a556cd708593f": "val",   # bucket 1393
    "bb2dc731b5334340bda9e269282626f0": "val",   # bucket 1602
    "7f33794d01f647788acdc5411262dc74": "val",   # bucket 1716
    "9baabd05353e43e99ef215b544ca20db": "val",   # bucket 1724
    "5673c42d94e24a6091c3f078a7ca311b": "val",   # bucket 1812
    "cd792f6e275a426a8488ad81dae7e0b9": "test",  # bucket 1883
    "03e53318a3714c8aa31854ff4f6f5767": "test",  # bucket 1921
    "5e622651c92b417fa890e218aa26c8fc": "test",  # bucket 1927
    "f54f4c997e204952a214921e3460c403": "test",  # bucket 2038
    "4455d06e22b6470bac72d271d1bd4e9a": "test",  # bucket 2078
    "9cff602351d846f596e87b251250965e": "test",  # bucket 2092
    "a789c83d12b74d498892c8b7b180246e": "test",  # bucket 2202
}

FROZEN_HELDOUT_V1 = {**FROZEN_HELDOUT, **EXPANSION_HELDOUT}


def match_to_split_v1(match_id: str, seed: int = CANONICAL_SPLIT_SEED) -> str:
    """v0 assignment for the original heldout bands; frozen-list promotion for
    the 16 expansion matches; train otherwise."""
    base = match_to_split(match_id, seed=seed)
    if base != "train":
        return base
    return EXPANSION_HELDOUT.get(match_id, "train")


def _self_check() -> None:
    for mid, want in FROZEN_HELDOUT.items():
        got = match_to_split_v1(mid)
        if got != want:
            raise AssertionError(f"v1 broke a v0 assignment: {mid} -> {got}, frozen v0 says {want}")
    for mid, want in EXPANSION_HELDOUT.items():
        if stable_bucket(f"{CANONICAL_SPLIT_SEED}|{mid}") < 1000:
            raise AssertionError(f"expansion match {mid} is not a v0 train match")
        got = match_to_split_v1(mid)
        if got != want:
            raise AssertionError(f"expansion drift: {mid} -> {got}, frozen says {want}")
    val_n = sum(1 for v in FROZEN_HELDOUT_V1.values() if v == "val")
    test_n = sum(1 for v in FROZEN_HELDOUT_V1.values() if v == "test")
    if (val_n, test_n) != (15, 15):
        raise AssertionError(f"v1 heldout counts drifted: val={val_n} test={test_n}, frozen says 15/15")


_self_check()

