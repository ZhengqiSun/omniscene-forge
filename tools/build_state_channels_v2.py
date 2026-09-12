#!/usr/bin/env python3
"""Build player-state condition cache v2 for Memory dense adapter training.

v2 (qxq, 2026-07-23) port of zhengqi tools/build_state_channels_v0.py (350L)
with two corrections and no other behavior change:
  1. Constants/helpers are imported from build_mesh_dense_condition_v0_droplog
     (qxq copy of the production renderer: capsule calibration
     PLAYER_RADIUS=14 / PLAYER_HEIGHT=40 / PLAYER_RADIUS_SCALE=0.70 /
     PLAYER_SCREEN_Y_OFFSET_PX=-5.5) instead of the stale qxq
     build_mesh_dense_condition_v0 (old 18/72/80 constants).
  2. Projection defaults corrected per the v2 render contract:
     --far 3000.0 (was 4096.0 in v0), --pitch-sign +1.0 (was -1.0 in v0).
The build report additionally echoes projection_params and the sha256 of the
imported constants module.

Outputs one compressed npz per aligned cache row plus a jsonl manifest keyed by
clip_id. Ego alive/health are stored as [21] scalars; opponent dead-marker is a
[21,60,104] uint8 mask and is expanded by the state trainer on demand.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from build_mesh_dense_condition_v0_droplog import (  # type: ignore
    CAMERA_PITCH_OFFSET,
    CAMERA_YAW_OFFSET,
    PLAYER_HEIGHT,
    PLAYER_RADIUS,
    PLAYER_RADIUS_SCALE,
    PLAYER_SCREEN_Y_OFFSET_PX,
    PLAYER_Z_OFFSET,
    import_tool,
    parse_team_player,
    stamp_ellipse,
)

_CONSTANTS_MODULE_PATH = ROOT / "tools" / "build_mesh_dense_condition_v0_droplog.py"

DEFAULT_CACHE_MANIFESTS = [
    Path("output/memory_v2_canonical_cleanup_20260628_v0/pilot1_aligned_cache_K11T2_motion_q50_v2only_v0.jsonl"),
    Path("output/memory_v2_canonical_cleanup_20260628_v0/pilot3_batch1_aligned_cache_K11T2_motion_q50_v2only_v0.jsonl"),
    Path("output/memory_v2_canonical_cleanup_20260628_v0/pilot3_batch2_aligned_cache_K11T2_motion_q50_v2only_v0.jsonl"),
]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


_JSON_CACHE: dict[str, Any] = {}
# player-frame jsons are large; an unbounded cache OOM-killed a 200GiB pod on the full run
_JSON_CACHE_MAX = 20
_EPISODE_PLAYER_JSONS_CACHE: dict[str, list[Path]] = {}


def read_json_cached(path: Path) -> Any:
    key = str(path)
    if key not in _JSON_CACHE:
        if len(_JSON_CACHE) >= _JSON_CACHE_MAX:
            _JSON_CACHE.clear()
        _JSON_CACHE[key] = read_json(path)
    return _JSON_CACHE[key]


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mtime_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def finite_player(frame: dict[str, Any]) -> bool:
    try:
        return bool(np.isfinite(float(frame["x"])) and np.isfinite(float(frame["y"])) and np.isfinite(float(frame["z"])))
    except Exception:
        return False


def ego_state(frames: list[dict[str, Any]], raw_frames: list[int]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    alive = np.ones((len(raw_frames),), dtype=np.float32)
    health = np.ones((len(raw_frames),), dtype=np.float32)
    debug: list[dict[str, Any]] = []
    for i, frame_index in enumerate(raw_frames):
        if frame_index < 0 or frame_index >= len(frames):
            h = 100.0
            missing = True
        else:
            h = float(frames[frame_index].get("health", 100.0))
            missing = "health" not in frames[frame_index]
        alive[i] = 1.0 if h > 0.0 else 0.0
        health[i] = max(0.0, min(1.0, h / 100.0))
        debug.append({"frame": int(frame_index), "raw_health": float(h), "alive": float(alive[i]), "health_norm": float(health[i]), "fallback_health": bool(missing)})
    return alive, health, debug


def episode_player_jsons(action_json: Path) -> list[Path]:
    episode_dir = action_json.parent
    return sorted(p for p in episode_dir.glob("*.json") if p.name.startswith("Ep_") and not p.name.endswith("_episode_info.json") and not p.name.endswith("_video_manifest.json") and not p.name.endswith("_player_visibility.json"))


def episode_player_jsons_cached(action_json: Path) -> list[Path]:
    key = str(action_json.parent)
    if key not in _EPISODE_PLAYER_JSONS_CACHE:
        _EPISODE_PLAYER_JSONS_CACHE[key] = episode_player_jsons(action_json)
    return _EPISODE_PLAYER_JSONS_CACHE[key]


def project_dead_mask_for_frame(
    *,
    mesh_mod: Any,
    other_frames_by_stem: dict[str, list[dict[str, Any]]],
    ego_frame: dict[str, Any],
    ego_stem: str,
    frame_index: int,
    latent_h: int,
    latent_w: int,
    video_h: int,
    video_w: int,
    fov_x: float,
    far: float,
    pitch_sign: float,
) -> tuple[np.ndarray, int, list[str]]:
    mask = np.zeros((latent_h, latent_w), dtype=np.uint8)
    cam_pos = np.asarray(ego_frame.get("camera_position") or [ego_frame["x"], ego_frame["y"], ego_frame["z"] + 64.0], dtype=np.float32)
    yaw = float(ego_frame.get("yaw", ego_frame.get("camera_rotation", [0, 0, 0])[2]))
    pitch = float(ego_frame.get("pitch", ego_frame.get("camera_rotation", [0, 0, 0])[1]))
    projected = 0
    projected_stems: list[str] = []
    for stem, frames in other_frames_by_stem.items():
        if stem == ego_stem or frame_index >= len(frames):
            continue
        frame = frames[frame_index]
        if not finite_player(frame):
            continue
        if float(frame.get("health", 100.0)) > 0.0:
            continue
        base = np.asarray([frame["x"], frame["y"], frame["z"] + PLAYER_Z_OFFSET], dtype=np.float32)
        samples = np.asarray([
            [base[0], base[1], base[2] + 4.0],
            [base[0], base[1], base[2] + PLAYER_HEIGHT * 0.18],
            [base[0], base[1], base[2] + PLAYER_HEIGHT * 0.35],
        ], dtype=np.float32)
        proj, z_cam = mesh_mod.project_vertices(samples, cam_pos, yaw + CAMERA_YAW_OFFSET, pitch + CAMERA_PITCH_OFFSET, pitch_sign, video_w, video_h, fov_x)
        valid = (z_cam > 1.0) & (z_cam < far)
        if not np.any(valid):
            continue
        u = float(np.mean(proj[valid, 0])) * latent_w / video_w
        v = float(np.mean(proj[valid, 1]) + PLAYER_SCREEN_Y_OFFSET_PX) * latent_h / video_h
        z = float(np.mean(z_cam[valid]))
        base_pixel_radius = PLAYER_RADIUS / max(z, 1.0) / np.tan(np.radians(fov_x) / 2.0) * video_w * 0.5
        rx = max(1.0, base_pixel_radius * PLAYER_RADIUS_SCALE * latent_w / video_w * 1.6)
        ry = max(1.0, rx * 0.45)
        if u < -rx or u >= latent_w + rx or v < -ry or v >= latent_h + ry:
            continue
        stamp_ellipse(mask, u, v, rx, ry, 1)
        projected += 1
        projected_stems.append(stem)
    return mask, projected, projected_stems


def build_one(record: dict[str, Any], out_dir: Path, mesh_mod: Any, *, video_h: int, video_w: int, latent_h: int, latent_w: int, fov_x: float, far: float, pitch_sign: float) -> tuple[dict[str, Any], dict[str, Any]]:
    clip_id = str(record["clip_id"])
    raw_frames = [int(x) for x in (record.get("map_memory_raw_frame_indices") or record["raw_indices"])]
    action_json = Path(record["action_json"])
    ego_stem = str(record.get("map_memory_ego_stem") or record.get("player_stem"))
    ego_frames = read_json_cached(action_json)
    alive, health, ego_debug = ego_state(ego_frames, raw_frames)

    other_frames_by_stem: dict[str, list[dict[str, Any]]] = {}
    for p in episode_player_jsons_cached(action_json):
        stem = p.stem
        if stem == ego_stem:
            continue
        # parse filter keeps only player files and drops unrelated jsons.
        try:
            parse_team_player(stem)
        except Exception:
            continue
        other_frames_by_stem[stem] = read_json_cached(p)

    dead_masks = np.zeros((len(raw_frames), latent_h, latent_w), dtype=np.uint8)
    dead_projected_counts: list[int] = []
    dead_stems_by_frame: list[list[str]] = []
    for i, frame_index in enumerate(raw_frames):
        if frame_index < 0 or frame_index >= len(ego_frames):
            dead_projected_counts.append(0)
            dead_stems_by_frame.append([])
            continue
        m, count, stems = project_dead_mask_for_frame(
            mesh_mod=mesh_mod,
            other_frames_by_stem=other_frames_by_stem,
            ego_frame=ego_frames[frame_index],
            ego_stem=ego_stem,
            frame_index=frame_index,
            latent_h=latent_h,
            latent_w=latent_w,
            video_h=video_h,
            video_w=video_w,
            fov_x=fov_x,
            far=far,
            pitch_sign=pitch_sign,
        )
        dead_masks[i] = m
        dead_projected_counts.append(int(count))
        dead_stems_by_frame.append(stems)

    rel = Path(sha1_text(clip_id)[:2]) / f"{clip_id}.npz"
    out_path = out_dir / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        ego_alive=alive.astype(np.float32),
        ego_health=health.astype(np.float32),
        opponent_dead_mask=dead_masks.astype(np.uint8),
        raw_frames=np.asarray(raw_frames, dtype=np.int32),
    )
    row = {
        "kind": "state_channels_v0_record",
        "clip_id": clip_id,
        "state_cache": str(out_path),
        "source_cache_manifest": record.get("_source_cache_manifest"),
        "latent_cache": record.get("latent_cache"),
        "map_memory_raw_frame_indices": raw_frames,
        "shape": {"ego_alive": [len(raw_frames)], "ego_health": [len(raw_frames)], "opponent_dead_mask": [len(raw_frames), latent_h, latent_w]},
        "channels": ["ego_alive_constant_plane", "ego_health_norm_constant_plane", "opponent_dead_marker_mask"],
        "stats": {
            "ego_alive_min": float(alive.min()) if len(alive) else None,
            "ego_health_min": float(health.min()) if len(health) else None,
            "ego_dead_frames": int((alive <= 0.0).sum()),
            "opponent_dead_mask_frames": int(sum(1 for x in dead_projected_counts if x > 0)),
            "opponent_dead_mask_pixels": int(dead_masks.sum()),
            "dead_projected_counts": dead_projected_counts,
        },
    }
    debug = {
        "clip_id": clip_id,
        "action_json": str(action_json),
        "ego_stem": ego_stem,
        "raw_frames": raw_frames,
        "ego_debug": ego_debug,
        "dead_stems_by_frame": dead_stems_by_frame,
    }
    return row, debug


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-manifest", type=Path, action="append", default=[])
    ap.add_argument("--out-dir", type=Path, default=Path("output/state_channels_v0"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--video-height", type=int, default=480)
    ap.add_argument("--video-width", type=int, default=832)
    ap.add_argument("--latent-height", type=int, default=60)
    ap.add_argument("--latent-width", type=int, default=104)
    ap.add_argument("--fov-x", type=float, default=106.26)
    ap.add_argument("--far", type=float, default=3000.0, help="Far clip for dead-marker projection; v2 corrected default (was 4096.0 in v0).")
    ap.add_argument("--pitch-sign", type=float, default=1.0, help="Pitch sign for projection; v2 corrected default (was -1.0 in v0).")
    ap.add_argument("--clean", action="store_true", help="Remove existing state cache/report files under --out-dir before building.")
    ap.add_argument("--state-dir-name", default="state_cache_v0")
    ap.add_argument("--manifest-name", default="state_cache_manifest_v0.jsonl")
    ap.add_argument("--report-name", default="state_cache_build_report_v0.json")
    args = ap.parse_args()

    manifests = args.cache_manifest or DEFAULT_CACHE_MANIFESTS
    state_dir = args.out_dir / args.state_dir_name
    if args.clean and args.out_dir.exists():
        for p in [state_dir, args.out_dir / args.manifest_name, args.out_dir / args.report_name]:
            if p.is_dir():
                import shutil

                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
    state_dir.mkdir(parents=True, exist_ok=True)
    mesh_mod = import_tool(ROOT / "tools" / "build_mesh_projection_v0.py", "mesh_projection_v0_state_v0")

    records: list[dict[str, Any]] = []
    for p in manifests:
        for row in iter_jsonl(p):
            row = dict(row)
            row["_source_cache_manifest"] = str(p)
            records.append(row)
    if args.limit is not None:
        records = records[: args.limit]

    out_manifest = args.out_dir / args.manifest_name
    debug_samples: list[dict[str, Any]] = []
    counts = Counter()
    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        row, debug = build_one(record, state_dir, mesh_mod, video_h=args.video_height, video_w=args.video_width, latent_h=args.latent_height, latent_w=args.latent_width, fov_x=args.fov_x, far=args.far, pitch_sign=args.pitch_sign)
        row["state_cache"] = str(Path(row["state_cache"]).relative_to(args.out_dir))
        rows.append(row)
        st = row["stats"]
        counts["rows"] += 1
        counts["ego_dead_windows"] += int(st["ego_dead_frames"] > 0)
        counts["opponent_dead_windows"] += int(st["opponent_dead_mask_frames"] > 0)
        counts["windows_without_any_death_signal"] += int(st["ego_dead_frames"] == 0 and st["opponent_dead_mask_frames"] == 0)
        counts["opponent_dead_mask_pixels"] += int(st["opponent_dead_mask_pixels"])
        if len(debug_samples) < 3:
            debug_samples.append(debug)
        if (idx + 1) % 100 == 0:
            print(json.dumps({"event": "progress", "done": idx + 1, "total": len(records)}, ensure_ascii=False), flush=True)

    with out_manifest.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    by_manifest = Counter(row["source_cache_manifest"] for row in rows)
    cache_files = sorted(state_dir.glob("*/*.npz"))
    report = {
        "kind": "state_channels_v0_build_report",
        "status": "pass",
        "out_dir": str(args.out_dir),
        "state_cache_manifest": str(out_manifest),
        "state_cache_dir": str(state_dir),
        "input_cache_manifests": [str(p) for p in manifests],
        "projection_params": {
            "far": float(args.far),
            "pitch_sign": float(args.pitch_sign),
            "fov_x": float(args.fov_x),
            "video_height": int(args.video_height),
            "video_width": int(args.video_width),
            "latent_height": int(args.latent_height),
            "latent_width": int(args.latent_width),
        },
        "constants_source_module": {
            "path": str(_CONSTANTS_MODULE_PATH),
            "sha256": sha256_file(_CONSTANTS_MODULE_PATH),
        },
        "input_rows": len(records),
        "written_rows": len(rows),
        "state_cache_file_count": len(cache_files),
        "counts": dict(counts),
        "death_window_stats": {
            "ego_dead_windows": int(counts["ego_dead_windows"]),
            "opponent_dead_windows": int(counts["opponent_dead_windows"]),
            "windows_without_any_death_signal": int(counts["windows_without_any_death_signal"]),
            "opponent_dead_mask_pixels": int(counts["opponent_dead_mask_pixels"]),
        },
        "rows_by_input_manifest": dict(by_manifest),
        "mtime_utc": {
            "manifest": mtime_utc(out_manifest),
            "report": None,
            "first_cache_file": mtime_utc(cache_files[0]) if cache_files else None,
            "last_cache_file": mtime_utc(cache_files[-1]) if cache_files else None,
        },
        "storage": "compressed npz per clip: ego scalar vectors [21] plus opponent_dead_mask uint8 [21,60,104]; constant planes are expanded by trainer on demand",
        "fallbacks": {"missing_ego_health": "health=1.0/alive=1.0", "missing_player_frame": "skip dead marker for that player/frame"},
        "health_audit_samples": debug_samples,
    }
    report_path = args.out_dir / args.report_name
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["mtime_utc"]["report"] = mtime_utc(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
