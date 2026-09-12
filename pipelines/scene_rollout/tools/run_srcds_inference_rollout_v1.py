#!/usr/bin/env python3
"""Run an authoritative CS:GO srcds rollout from one observed replay state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


ACTION_NAMES = (
    "forward", "back", "left", "right", "jump", "crouch", "walk",
    "fire", "reload", "use", "scope", "plant", "defuse",
)
BUTTON_MASKS = (8, 16, 512, 1024, 2, 4, 131072, 1, 8192, 32, 2048, 0, 0)
PLAYER_RE = re.compile(r"^(?P<episode>Ep_\d+)_team_(?P<team>\d+)_player_(?P<player>\d+)_inst_000$")
WEAPON_ENTITIES = {
    "": "weapon_knife",
    "Knife": "weapon_knife",
    "AK-47": "weapon_ak47",
    "AWP": "weapon_awp",
    "Glock-18": "weapon_glock",
    "USP-S": "weapon_usp_silencer",
    "M4A1": "weapon_m4a1_silencer",
    "M4A4": "weapon_m4a1",
    "C4": "weapon_c4",
    "Molotov": "weapon_molotov",
    "Incendiary Grenade": "weapon_incgrenade",
    "HE Grenade": "weapon_hegrenade",
    "Flashbang": "weapon_flashbang",
    "Smoke Grenade": "weapon_smokegrenade",
    "Decoy Grenade": "weapon_decoy",
    "P250": "weapon_p250",
    "Desert Eagle": "weapon_deagle",
    "FAMAS": "weapon_famas",
    "Galil AR": "weapon_galilar",
    "SSG 08": "weapon_ssg08",
    "MP9": "weapon_mp9",
    "MAC-10": "weapon_mac10",
    "UMP-45": "weapon_ump45",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signed_angle(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def angle_diff(current: float, previous: float) -> float:
    return (float(current) - float(previous) + 180.0) % 360.0 - 180.0


def discover_players(match_dir: Path, episode: str) -> list[tuple[str, Path, list[dict[str, Any]]]]:
    episode_dir = match_dir / "train" / episode
    rows = []
    for path in episode_dir.glob(f"{episode}_team_*_player_*_inst_000.json"):
        stem = path.stem
        match = PLAYER_RE.match(stem)
        if match is None:
            continue
        rows.append((stem, path, read_json(path)))
    rows.sort(key=lambda row: (int(PLAYER_RE.match(row[0]).group("team")), int(PLAYER_RE.match(row[0]).group("player"))))
    if len(rows) != 10:
        raise RuntimeError(f"expected 10 player tracks, got {len(rows)}: {episode_dir}")
    if [int(PLAYER_RE.match(row[0]).group("team")) for row in rows].count(2) != 5:
        raise RuntimeError("expected five team-2 players")
    if [int(PLAYER_RE.match(row[0]).group("team")) for row in rows].count(3) != 5:
        raise RuntimeError("expected five team-3 players")
    return rows


def build_initial_plan(
    players: list[tuple[str, Path, list[dict[str, Any]]]], start: int, fps: float
) -> tuple[list[str], list[dict[str, Any]]]:
    stems = []
    plan = []
    for index, (stem, path, track) in enumerate(players):
        if start <= 0 or start >= len(track):
            raise RuntimeError(f"invalid start frame for {stem}: {start}")
        previous = track[start - 1]
        initial = track[start]
        action = initial.get("action") or {}
        weapon_label = str(action.get("weapon_slot") or "")
        if weapon_label not in WEAPON_ENTITIES:
            raise RuntimeError(f"unmapped active weapon for {stem}: {weapon_label!r}")
        match = PLAYER_RE.match(stem)
        position = np.asarray([initial["x"], initial["y"], initial["z"]], dtype=np.float64)
        previous_position = np.asarray([previous["x"], previous["y"], previous["z"]], dtype=np.float64)
        velocity = (position - previous_position) * fps
        row = {
            "player_index": index,
            "player_stem": stem,
            "source_path": str(path),
            "team": int(match.group("team")),
            "position": position.tolist(),
            "yaw": signed_angle(initial["yaw"]),
            "pitch": signed_angle(initial["pitch"]),
            "velocity": velocity.tolist(),
            "health": int(initial.get("health", 100)),
            "armor": int(initial.get("armor", 0)),
            "weapon_label": weapon_label,
            "weapon_entity": WEAPON_ENTITIES[weapon_label],
            "observed_frames_read": [start - 1, start],
        }
        stems.append(stem)
        plan.append(row)
    return stems, plan


def write_engine_inputs(engine_dir: Path, plan: list[dict[str, Any]], horizon: int) -> dict[str, Path]:
    runtime = engine_dir / "csgo_ds" / "csgo" / "engine_inference"
    runtime.mkdir(parents=True, exist_ok=True)
    plan_path = runtime / "initial_state_v1.csv"
    lines = []
    for row in plan:
        pos = row["position"]
        vel = row["velocity"]
        lines.append(
            ",".join(
                [
                    str(row["player_index"]), str(row["team"]),
                    *(f"{float(value):.9f}" for value in pos),
                    f"{float(row['yaw']):.9f}", f"{float(row['pitch']):.9f}",
                    *(f"{float(value):.9f}" for value in vel),
                    str(row["health"]), str(row["armor"]), row["weapon_entity"],
                ]
            )
        )
    plan_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    horizon_path = runtime / "horizon_v1.txt"
    horizon_path.write_text(f"{horizon}\n", encoding="utf-8")
    return {
        "runtime": runtime,
        "plan": plan_path,
        "horizon": horizon_path,
        "ready": runtime / "ready_v1.txt",
        "done": runtime / "done_v1.txt",
        "export": runtime / "state_export_v1.jsonl",
    }


def stop_process_group(process: subprocess.Popen[Any], grace_seconds: float = 10.0) -> str:
    if process.poll() is not None:
        return f"already_exited:{process.returncode}"
    os.killpg(process.pid, signal.SIGINT)
    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if process.poll() is not None:
            return f"sigint_exit:{process.returncode}"
        time.sleep(0.25)
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5.0)
        return f"sigterm_exit:{process.returncode}"
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5.0)
        return f"sigkill_exit:{process.returncode}"


def run_srcds(engine_dir: Path, runtime_paths: dict[str, Path], out_root: Path, port: int, timeout: float) -> dict[str, Any]:
    ds = engine_dir / "csgo_ds"
    runtime32 = engine_dir / "runtime32" / "usr" / "lib32"
    loader = runtime32 / "ld-linux.so.2"
    if not loader.is_file():
        raise RuntimeError(f"project-local 32-bit ELF loader is missing: {loader}")
    for key in ("ready", "done", "export"):
        path = runtime_paths[key]
        if path.exists():
            archived = out_root / "preexisting_engine_files" / path.name
            archived.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(archived))
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["CUDA_VISIBLE_DEVICES"] = ""
    library_path = f"{runtime32}:{ds / 'bin'}"
    env["LD_LIBRARY_PATH"] = library_path
    log_path = out_root / "srcds_native_rollout.log"
    command = [
        "nice", "-n", "10", str(loader), "--library-path", library_path,
        "./srcds_linux", "-game", "csgo", "-console", "-usercon",
        "-tickrate", "128", "-port", str(port), "+sv_hibernate_when_empty", "0",
        "+sv_hibernate_ms", "0", "+game_type", "0", "+game_mode", "1",
        "+map", "de_dust2", "+exec", "server.cfg",
    ]
    old_exporter = ds / "csgo/addons/sourcemod/plugins/engine_smoke_export.smx"
    disabled_exporter = ds / "csgo/addons/sourcemod/plugins/disabled/engine_smoke_export.native_run_disabled.smx"
    if disabled_exporter.exists():
        raise RuntimeError(f"stale disabled exporter from an earlier run: {disabled_exporter}")
    if old_exporter.exists():
        shutil.move(str(old_exporter), str(disabled_exporter))
    started = time.time()
    process = None
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command, cwd=ds, env=env, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
            deadline = started + timeout
            while time.time() < deadline and not runtime_paths["done"].is_file():
                if process.poll() is not None:
                    raise RuntimeError(f"srcds exited before rollout completed: {process.returncode}; see {log_path}")
                time.sleep(0.25)
            if not runtime_paths["done"].is_file():
                stop_process_group(process)
                raise RuntimeError(f"srcds rollout timeout after {timeout}s; see {log_path}")
            stop_status = stop_process_group(process)
    finally:
        if process is not None and process.poll() is None:
            stop_process_group(process)
        if disabled_exporter.exists():
            shutil.move(str(disabled_exporter), str(old_exporter))
    export_copy = out_root / "srcds_state_export_v1.jsonl"
    shutil.copy2(runtime_paths["export"], export_copy)
    return {
        "command": command,
        "port": port,
        "pid": process.pid,
        "elapsed_s": time.time() - started,
        "stop_status": stop_status,
        "log": str(log_path),
        "export": str(export_copy),
    }


def parse_export(path: Path, horizon: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_frame = {int(row["frame_index"]): row for row in rows}
    expected = list(range(horizon + 1))
    if sorted(by_frame) != expected:
        raise RuntimeError(f"native export frames mismatch: got={sorted(by_frame)} expected={expected}")
    ordered = [by_frame[index] for index in expected]
    for row in ordered:
        players = row.get("players") or []
        if len(players) != 10 or [int(player["player_index"]) for player in players] != list(range(10)):
            raise RuntimeError(f"native export player contract failed at frame {row.get('frame_index')}")
    return ordered


def build_episode_memory(
    rows: list[dict[str, Any]], stems: list[str], start: int, horizon: int, fps: float, out_root: Path
) -> tuple[Path, Path, dict[str, Any]]:
    player_count = len(stems)
    length = start + horizon + 1
    position = np.full((player_count, length, 3), np.nan, dtype=np.float32)
    camera_position = np.full_like(position, np.nan)
    yaw = np.full((player_count, length), np.nan, dtype=np.float32)
    pitch = np.full_like(yaw, np.nan)
    velocity = np.full_like(position, np.nan)
    health = np.full((player_count, length), -1.0, dtype=np.float32)
    armor = np.full_like(health, -1.0)
    alive = np.zeros((player_count, length), dtype=np.bool_)
    actions = np.zeros((player_count, length, len(ACTION_NAMES)), dtype=np.bool_)
    look_delta = np.zeros((player_count, length, 2), dtype=np.float32)
    tick = np.full((player_count, length), -1, dtype=np.int64)
    team_id = np.zeros((player_count,), dtype=np.int32)
    player_index = np.zeros((player_count,), dtype=np.int32)
    weapon = [[""] * (horizon + 1) for _ in range(player_count)]

    for offset, row in enumerate(rows):
        frame = start + offset
        for player in row["players"]:
            index = int(player["player_index"])
            position[index, frame] = np.asarray(player["origin"], dtype=np.float32)
            camera_position[index, frame] = np.asarray(player["eye_position"], dtype=np.float32)
            yaw[index, frame] = float(player["eye_angles"][0]) % 360.0
            pitch[index, frame] = float(player["eye_angles"][1])
            velocity[index, frame] = np.asarray(player["velocity"], dtype=np.float32)
            health[index, frame] = max(0.0, float(player["health"]))
            armor[index, frame] = max(0.0, float(player["armor"]))
            alive[index, frame] = bool(player["alive"])
            buttons = int(player["buttons"])
            actions[index, frame] = np.asarray([bool(buttons & mask) if mask else False for mask in BUTTON_MASKS])
            tick[index, frame] = int(row["server_tick"])
            team_id[index] = int(player["team"])
            player_index[index] = index
            weapon[index][offset] = str(player["weapon"])
            if offset > 0:
                look_delta[index, frame, 0] = angle_diff(yaw[index, frame], yaw[index, frame - 1])
                look_delta[index, frame, 1] = pitch[index, frame] - pitch[index, frame - 1]

    memory_dir = out_root / "srcds_episode_memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    npz_path = memory_dir / "episode_memory_v1.npz"
    np.savez_compressed(
        npz_path,
        position=position,
        camera_position=camera_position,
        yaw=yaw,
        pitch=pitch,
        velocity=velocity,
        health=health,
        armor=armor,
        alive=alive,
        actions=actions,
        look_delta=look_delta,
        tick=tick,
        team_id=team_id,
        player_index=player_index,
        track_length=np.full((player_count,), length, dtype=np.int32),
    )
    meta = {
        "kind": "srcds_native_inference_episode_memory_v1",
        "npz_path": str(npz_path),
        "player_count": player_count,
        "frame_count": length,
        "valid_frame_range": [start, start + horizon],
        "player_stems": stems,
        "fps": fps,
        "policy_source": "native CS:GO bot AI after one observed replay state",
        "future_replay_actions_read": False,
        "future_replay_state_read": False,
        "state_fields": ["position", "camera_position", "yaw", "pitch", "velocity", "health", "armor", "alive", "weapon"],
        "action_names": list(ACTION_NAMES),
        "weapon_by_player_and_offset": weapon,
    }
    meta_path = memory_dir / "episode_memory_meta_v1.json"
    write_json(meta_path, meta)
    return npz_path, meta_path, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--match-dir", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=160)
    parser.add_argument("--fps", type=float, default=32.0)
    parser.add_argument("--port", type=int, default=27046)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    engine_dir = args.project_root.resolve() / "engine_smoke_v0"
    bridge = engine_dir / "csgo_ds/csgo/addons/sourcemod/plugins/engine_inference_bridge_v1.smx"
    if not bridge.is_file():
        raise RuntimeError(f"native inference bridge is missing: {bridge}")

    players = discover_players(args.match_dir.resolve(), args.episode)
    stems, plan = build_initial_plan(players, args.start, args.fps)
    runtime_paths = write_engine_inputs(engine_dir, plan, args.horizon)
    plan_report = {
        "kind": "srcds_native_inference_initial_plan_v1",
        "status": "pass",
        "match_dir": str(args.match_dir.resolve()),
        "episode": args.episode,
        "start": args.start,
        "horizon": args.horizon,
        "fps": args.fps,
        "observed_replay_frames_read": [args.start - 1, args.start],
        "future_replay_actions_read": False,
        "future_replay_state_read": False,
        "players": plan,
        "bindings": {
            "srcds_linux": {"path": str(engine_dir / "csgo_ds/srcds_linux"), "sha256": sha256_file(engine_dir / "csgo_ds/srcds_linux")},
            "bridge_plugin": {"path": str(bridge), "sha256": sha256_file(bridge)},
            "initial_state_csv": {"path": str(runtime_paths["plan"]), "sha256": sha256_file(runtime_paths["plan"])},
        },
    }
    write_json(output_root / "INITIAL_PLAN_AUDIT_v1.json", plan_report)
    runtime_report = run_srcds(engine_dir, runtime_paths, output_root, args.port, args.timeout)
    rows = parse_export(Path(runtime_report["export"]), args.horizon)
    npz_path, meta_path, memory_meta = build_episode_memory(rows, stems, args.start, args.horizon, args.fps, output_root)

    initial_errors = []
    for index, player in enumerate(rows[0]["players"]):
        expected = np.asarray(plan[index]["position"], dtype=np.float64)
        actual = np.asarray(player["origin"], dtype=np.float64)
        initial_errors.append(float(np.linalg.norm(actual - expected)))
    tick_steps = [int(right["server_tick"]) - int(left["server_tick"]) for left, right in zip(rows, rows[1:])]
    max_step_distance = 0.0
    for index in range(10):
        positions = np.asarray([row["players"][index]["origin"] for row in rows], dtype=np.float64)
        max_step_distance = max(max_step_distance, float(np.linalg.norm(np.diff(positions, axis=0), axis=1).max()))
    alive_changes = 0
    health_changes = 0
    for index in range(10):
        states = [bool(row["players"][index]["alive"]) for row in rows]
        healths = [int(row["players"][index]["health"]) for row in rows]
        alive_changes += sum(left != right for left, right in zip(states, states[1:]))
        health_changes += sum(left != right for left, right in zip(healths, healths[1:]))
    final_report = {
        "kind": "srcds_native_inference_rollout_v1",
        "status": "pass",
        "claim": "one observed state plus observed initial velocity; all future world state and policy come from native CS:GO",
        "initial_plan": str(output_root / "INITIAL_PLAN_AUDIT_v1.json"),
        "runtime": runtime_report,
        "export_frames": len(rows),
        "tick_step_values": sorted(set(tick_steps)),
        "initial_position_error_max": max(initial_errors),
        "initial_position_error_mean": float(np.mean(initial_errors)),
        "max_position_step_at_32hz": max_step_distance,
        "alive_change_count": alive_changes,
        "health_change_count": health_changes,
        "episode_memory": str(npz_path),
        "episode_memory_meta": str(meta_path),
        "memory_contract": memory_meta,
    }
    if (
        len(rows) != args.horizon + 1
        or sorted(set(tick_steps)) != [4]
        or max(initial_errors) > 0.05
        or max_step_distance > 50.0
    ):
        final_report["status"] = "fail"
        write_json(output_root / "SRCSD_NATIVE_ROLLOUT_AUDIT_v1.json", final_report)
        raise RuntimeError(f"native rollout contract failed: {final_report}")
    write_json(output_root / "SRCSD_NATIVE_ROLLOUT_AUDIT_v1.json", final_report)
    print(json.dumps({"status": "pass", "output_root": str(output_root), "frames": len(rows), "max_initial_error": max(initial_errors)}))


if __name__ == "__main__":
    main()
