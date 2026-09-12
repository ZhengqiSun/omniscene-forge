#!/usr/bin/env python3
"""Build fire, reload, weapon-switch, and throw interaction channels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


# Fixed from the audited action JSON ``action.weapon_slot`` and ``equipment.name``
# values. Keep this order stable: it is the serialized one-hot channel order.
WEAPON_VOCABULARY = (
    "unknown",
    "other",
    "knife",
    "c4",
    # Pistols.
    "glock_18",
    "usp_s",
    "p2000",
    "p250",
    "five_seven",
    "tec_9",
    "dual_berettas",
    "desert_eagle",
    # SMGs.
    "mac_10",
    "mp9",
    # Rifles.
    "famas",
    "galil_ar",
    "ak_47",
    "m4a1",
    # Sniper rifles.
    "ssg_08",
    "awp",
    # Grenades are deliberately separate item classes.
    "flashbang",
    "smoke_grenade",
    "he_grenade",
    "molotov",
    "incendiary_grenade",
    "decoy_grenade",
)

_WEAPON_NAME_TO_KEY = {
    "Knife": "knife",
    "C4": "c4",
    "Glock-18": "glock_18",
    "USP-S": "usp_s",
    "P2000": "p2000",
    "P250": "p250",
    "Five-SeveN": "five_seven",
    "Tec-9": "tec_9",
    "Dual Berettas": "dual_berettas",
    "Desert Eagle": "desert_eagle",
    "MAC-10": "mac_10",
    "MP9": "mp9",
    "FAMAS": "famas",
    "Galil AR": "galil_ar",
    "AK-47": "ak_47",
    "M4A1": "m4a1",
    "SSG 08": "ssg_08",
    "AWP": "awp",
    "Flashbang": "flashbang",
    "Smoke Grenade": "smoke_grenade",
    "HE Grenade": "he_grenade",
    "Molotov": "molotov",
    "Incendiary Grenade": "incendiary_grenade",
    "Decoy Grenade": "decoy_grenade",
}
_WEAPON_INDEX = {name: index for index, name in enumerate(WEAPON_VOCABULARY)}
_GRENADE_KEYS = frozenset(
    {
        "flashbang",
        "smoke_grenade",
        "he_grenade",
        "molotov",
        "incendiary_grenade",
        "decoy_grenade",
    }
)
_FIREARM_KEYS = frozenset(
    {
        "glock_18",
        "usp_s",
        "p2000",
        "p250",
        "five_seven",
        "tec_9",
        "dual_berettas",
        "desert_eagle",
        "mac_10",
        "mp9",
        "famas",
        "galil_ar",
        "ak_47",
        "m4a1",
        "ssg_08",
        "awp",
    }
)
_KNIFE_KEY = "knife"
_C4_KEY = "c4"


def _action(frame: dict[str, Any]) -> dict[str, Any]:
    value = frame.get("action")
    return value if isinstance(value, dict) else {}


def _weapon_slot(frame: dict[str, Any]) -> str:
    """Return normalized current weapon name; empty means unavailable."""
    value = _action(frame).get("weapon_slot", "")
    return str(value).strip() if value is not None else ""


def _weapon_key(weapon_name: str) -> str:
    if not weapon_name:
        return "unknown"
    return _WEAPON_NAME_TO_KEY.get(weapon_name, "other")


def _weapon_kind(weapon_name: str) -> str:
    key = _weapon_key(weapon_name)
    if key in _FIREARM_KEYS:
        return "firearm"
    if key in _GRENADE_KEYS:
        return "grenade"
    if key == _KNIFE_KEY:
        return "knife"
    if key == _C4_KEY:
        return "c4"
    return "other"


def _weapon_onehot(weapon_name: str) -> np.ndarray:
    value = np.zeros(len(WEAPON_VOCABULARY), dtype=np.float32)
    value[_WEAPON_INDEX[_weapon_key(weapon_name)]] = 1.0
    return value


def _event_value(value: Any) -> float:
    if isinstance(value, str):
        return float(value.strip().lower() in {"1", "true", "yes"})
    if isinstance(value, (bool, int, float, np.bool_, np.number)):
        return float(bool(value))
    return 0.0


def _explicit_throw_event(frame: dict[str, Any]) -> float | None:
    """Return an explicit throw pulse, or None when no throw field exists.

    The audited dataset has no throw field. This intentionally recognizes an
    explicit schema extension by meaning rather than guessing a single future
    spelling: any frame-level or action-level key containing ``throw`` wins over
    the grenade-fire fallback.
    """
    values: list[float] = []
    for source in (frame, _action(frame)):
        for key, value in source.items():
            if "throw" in str(key).lower():
                values.append(_event_value(value))
    return max(values, default=0.0) if values else None


def build_interaction_channels(
    frames: Sequence[dict[str, Any]],
    raw_frames: Sequence[int],
) -> dict[str, np.ndarray]:
    """Convert raw per-frame JSON records to latent-time interaction channels.

    Event alignment:
      latent index k receives any event that happened after the previous
      sampled raw frame and up to the current sampled raw frame.

    Returns:
      Existing compatible outputs:
        ego_fire_event:          float32 [F]
        ego_weapon_switch_event: float32 [F]
        ego_weapon_slot_name:    unicode [F]

      Added outputs:
        ego_reload_event, ego_throw_event: float32 [F]
        ego_current_weapon_onehot:          float32 [F,V]
        ego_fire_weapon_onehot:             float32 [F,V]
        ego_reload_weapon_onehot:           float32 [F,V]
        ego_switch_target_onehot:           float32 [F,V]
        ego_throw_item_onehot:              float32 [F,V]

    Fire/throw rule: raw fire becomes fire only for a firearm weapon_slot. For a
    grenade it becomes throw instead; knife, C4, unknown, and other items produce
    neither. An explicit frame/action key containing ``throw`` has priority for a
    grenade. No non-grenade item can produce a throw event or throw target.

    Reload rule: raw reload becomes reload only for a firearm weapon_slot. No
    grenade, knife, C4, unknown, or other item can produce reload or its target.
    """
    if not frames:
        raise ValueError("frames is empty")
    if not raw_frames:
        raise ValueError("raw_frames is empty")

    num_raw = len(frames)

    # Raw-frame event pulses.
    raw_fire = np.asarray(
        [_event_value(_action(frame).get("fire", False)) for frame in frames],
        dtype=np.float32,
    )
    raw_reload = np.asarray(
        [_event_value(_action(frame).get("reload", False)) for frame in frames],
        dtype=np.float32,
    )

    # Detect real weapon changes while ignoring empty/missing weapon_slot values.
    switch_raw = np.zeros(num_raw, dtype=np.float32)
    carried_weapon: list[str] = []
    last_valid_weapon = ""

    for index, frame in enumerate(frames):
        current = _weapon_slot(frame)

        if current:
            if last_valid_weapon and current != last_valid_weapon:
                switch_raw[index] = 1.0
            last_valid_weapon = current

        # Carry forward the last valid weapon over empty records.
        carried_weapon.append(last_valid_weapon)

    fire_raw = np.zeros(num_raw, dtype=np.float32)
    reload_raw = np.zeros(num_raw, dtype=np.float32)
    throw_raw = np.zeros(num_raw, dtype=np.float32)
    for index, frame in enumerate(frames):
        weapon_kind = _weapon_kind(carried_weapon[index])
        if raw_fire[index] > 0.0 and weapon_kind == "firearm":
            fire_raw[index] = 1.0
        if raw_reload[index] > 0.0 and weapon_kind == "firearm":
            reload_raw[index] = 1.0

        explicit_throw = _explicit_throw_event(frame)
        if weapon_kind == "grenade":
            if explicit_throw is not None:
                throw_raw[index] = explicit_throw
            elif raw_fire[index] > 0.0:
                throw_raw[index] = 1.0

    sampled_indices = [int(index) for index in raw_frames]

    for index in sampled_indices:
        if index < 0 or index >= num_raw:
            raise IndexError(
                f"raw frame index {index} outside valid range [0, {num_raw - 1}]"
            )

    fire_event = np.zeros(len(sampled_indices), dtype=np.float32)
    reload_event = np.zeros(len(sampled_indices), dtype=np.float32)
    switch_event = np.zeros(len(sampled_indices), dtype=np.float32)
    throw_event = np.zeros(len(sampled_indices), dtype=np.float32)
    sampled_weapon: list[str] = []
    current_weapon_onehot = np.zeros((len(sampled_indices), len(WEAPON_VOCABULARY)), dtype=np.float32)
    fire_weapon_onehot = np.zeros_like(current_weapon_onehot)
    reload_weapon_onehot = np.zeros_like(current_weapon_onehot)
    switch_target_onehot = np.zeros_like(current_weapon_onehot)
    throw_item_onehot = np.zeros_like(current_weapon_onehot)

    for latent_index, current_raw in enumerate(sampled_indices):
        if latent_index == 0:
            interval_start = current_raw
        else:
            interval_start = sampled_indices[latent_index - 1] + 1

        interval_end = current_raw

        if interval_start <= interval_end:
            fire_event[latent_index] = float(
                fire_raw[interval_start : interval_end + 1].max(initial=0.0)
            )
            reload_event[latent_index] = float(
                reload_raw[interval_start : interval_end + 1].max(initial=0.0)
            )
            switch_event[latent_index] = float(
                switch_raw[interval_start : interval_end + 1].max(initial=0.0)
            )
            throw_event[latent_index] = float(
                throw_raw[interval_start : interval_end + 1].max(initial=0.0)
            )

            event_outputs = (
                (fire_raw, fire_weapon_onehot),
                (reload_raw, reload_weapon_onehot),
                (switch_raw, switch_target_onehot),
                (throw_raw, throw_item_onehot),
            )
            for raw_event, onehot_output in event_outputs:
                event_offsets = np.flatnonzero(raw_event[interval_start : interval_end + 1] > 0.0)
                if event_offsets.size:
                    event_raw_index = interval_start + int(event_offsets[-1])
                    onehot_output[latent_index] = _weapon_onehot(carried_weapon[event_raw_index])

        endpoint_weapon = carried_weapon[current_raw]
        sampled_weapon.append(endpoint_weapon)
        current_weapon_onehot[latent_index] = _weapon_onehot(endpoint_weapon)

    return {
        "ego_fire_event": fire_event,
        "ego_weapon_switch_event": switch_event,
        "ego_weapon_slot_name": np.asarray(sampled_weapon, dtype="<U64"),
        "ego_reload_event": reload_event,
        "ego_throw_event": throw_event,
        "ego_current_weapon_onehot": current_weapon_onehot,
        "ego_fire_weapon_onehot": fire_weapon_onehot,
        "ego_reload_weapon_onehot": reload_weapon_onehot,
        "ego_switch_target_onehot": switch_target_onehot,
        "ego_throw_item_onehot": throw_item_onehot,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-json", required=True)
    parser.add_argument(
        "--raw-indices",
        default="",
        help="Comma-separated raw indices. Empty means use every raw frame.",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    path = Path(args.action_json)
    frames = json.loads(path.read_text())

    if not isinstance(frames, list):
        raise TypeError("action JSON must contain a top-level list")

    if args.raw_indices:
        raw_frames = [
            int(value.strip())
            for value in args.raw_indices.split(",")
            if value.strip()
        ]
    else:
        raw_frames = list(range(len(frames)))

    channels = build_interaction_channels(frames, raw_frames)

    print("sampled frames:", len(raw_frames))
    print(
        "fire event indices:",
        np.flatnonzero(channels["ego_fire_event"]).tolist(),
    )
    print(
        "switch event indices:",
        np.flatnonzero(channels["ego_weapon_switch_event"]).tolist(),
    )
    print(
        "reload event indices:",
        np.flatnonzero(channels["ego_reload_event"]).tolist(),
    )
    print(
        "throw event indices:",
        np.flatnonzero(channels["ego_throw_event"]).tolist(),
    )
    print(
        "weapon names:",
        sorted(set(channels["ego_weapon_slot_name"].tolist())),
    )
    print("weapon vocabulary:", list(WEAPON_VOCABULARY))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **channels)
        print("saved:", output)


if __name__ == "__main__":
    main()
