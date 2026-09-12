#!/usr/bin/env python3
"""Build a first-person dense-condition smoke sample from existing CS:GO streams.

This v0 does not ray cast BSP. It uses the dataset's already-rendered depth and
segmentation videos to produce the same tensor interface that the later renderer
should output:

  environment depth, coarse semantic, player mask/depth, and player orientation
  sin/cos. The historical teacher stream can still split enemy/team for
  diagnostics, but the canonical Memory input now merges them as other players.

The purpose is to lock down data alignment and downstream adapter I/O before
implementing the final BSP/collision-mesh renderer.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


PLAYER_RE = re.compile(r"^Ep_\d+_team_(\d+)_player_(\d+)_inst_(\d+)$")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_ego(stem: str) -> tuple[int, int]:
    m = PLAYER_RE.match(stem)
    if not m:
        raise ValueError(f"Cannot parse ego player stem: {stem}")
    return int(m.group(1)), int(m.group(2))


def read_frame(path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise IndexError(f"Could not read frame {frame_index} from {path}")
    return frame


def resize_nearest(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    return cv2.resize(arr, (width, height), interpolation=cv2.INTER_NEAREST)


def resize_area(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    return cv2.resize(arr, (width, height), interpolation=cv2.INTER_AREA)


def color_mask(seg_bgr: np.ndarray, rgb: list[int], tolerance: int) -> np.ndarray:
    # OpenCV returns BGR frames. Seg colormap stores RGB bytes.
    target = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.int16)
    diff = np.abs(seg_bgr.astype(np.int16) - target.reshape(1, 1, 3))
    return (diff <= tolerance).all(axis=2)


def build_player_metadata(episode_dir: Path, ego_stem: str) -> dict[str, Any]:
    ego_team, ego_player_index = parse_ego(ego_stem)
    game = load_json(episode_dir / "game_manifest.json")
    players = game["players"]
    by_handle = {str(p["entity_handle"]): p for p in players}
    ego = next(
        p for p in players
        if int(p["team_id"]) == ego_team and int(p["player_index"]) == ego_player_index
    )
    return {"ego": ego, "players": players, "by_handle": by_handle}


def build_masks(
    episode_dir: Path,
    ego_stem: str,
    frame_index: int,
    seg_bgr: np.ndarray,
    depth_norm_full: np.ndarray,
    output_hw: tuple[int, int],
    tolerance: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    meta = build_player_metadata(episode_dir, ego_stem)
    ego = meta["ego"]
    ego_team = int(ego["team_id"])
    colormap = load_json(episode_dir / f"{ego_stem}_seg_colormap.json")
    vis = load_json(episode_dir / f"{ego_stem}_player_visibility.json")

    h, w = output_hw
    enemy_mask = np.zeros(seg_bgr.shape[:2], dtype=bool)
    teammate_mask = np.zeros_like(enemy_mask)
    player_depth = np.zeros(seg_bgr.shape[:2], dtype=np.float32)
    orient_sin = np.zeros(seg_bgr.shape[:2], dtype=np.float32)
    orient_cos = np.zeros(seg_bgr.shape[:2], dtype=np.float32)
    found_players: list[dict[str, Any]] = []

    for player_id, entry in vis.items():
        ranges = entry.get("ranges", [])
        if not any(r["range"][0] <= frame_index <= r["range"][1] for r in ranges):
            continue
        handle = str(entry.get("entity_handle"))
        player = meta["by_handle"].get(handle)
        cmap = colormap.get(handle)
        if not player or not cmap:
            continue
        if int(player["team_id"]) == ego_team and int(player["player_index"]) == int(ego["player_index"]):
            continue
        mask = color_mask(seg_bgr, cmap["color"], tolerance=tolerance)
        pixels = int(mask.sum())
        if pixels == 0:
            continue
        is_enemy = int(player["team_id"]) != ego_team
        if is_enemy:
            enemy_mask |= mask
        else:
            teammate_mask |= mask
        player_depth[mask] = depth_norm_full[mask]

        other_stem = f"Ep_{episode_dir.name.split('_')[-1]}_team_{int(player['team_id'])}_player_{int(player['player_index']):04d}_inst_000"
        other_json = episode_dir / f"{other_stem}.json"
        yaw = None
        if other_json.exists():
            frames = load_json(other_json)
            if frame_index < len(frames):
                yaw = float(frames[frame_index].get("yaw", 0.0))
                orient_sin[mask] = math.sin(math.radians(yaw))
                orient_cos[mask] = math.cos(math.radians(yaw))
        found_players.append({
            "player_index": int(player["player_index"]),
            "team_id": int(player["team_id"]),
            "entity_handle": handle,
            "is_enemy": is_enemy,
            "pixels": pixels,
            "yaw": yaw,
        })

    enemy = resize_nearest(enemy_mask.astype(np.float32), h, w)
    teammate = resize_nearest(teammate_mask.astype(np.float32), h, w)
    pdepth = resize_area(player_depth, h, w)
    psin = resize_area(orient_sin, h, w)
    pcos = resize_area(orient_cos, h, w)
    return (
        np.stack([enemy, teammate, pdepth, psin, pcos], axis=0).astype(np.float32),
        {
            "ego": ego,
            "visible_players": found_players,
            "color_tolerance": tolerance,
        },
    )


def make_semantic(seg_bgr: np.ndarray, output_hw: tuple[int, int]) -> np.ndarray:
    # Coarse v0 semantics: foreground/object/player pixels are any non-black seg.
    # Exact class semantics need a stable dataset-level label map.
    nonzero = (seg_bgr.max(axis=2) > 0).astype(np.float32)
    return resize_nearest(nonzero, output_hw[0], output_hw[1]).astype(np.float32)


def make_depth(depth_bgr: np.ndarray, output_hw: tuple[int, int]) -> np.ndarray:
    # Existing depth stream is rgb24; OpenCV reads it as 8-bit BGR. For v0 we use
    # normalized luminance. The manifest records far=2048 units, but final depth
    # decoding should verify the original 16-bit packing.
    gray = depth_bgr.astype(np.float32).mean(axis=2) / 255.0
    return resize_area(gray, output_hw[0], output_hw[1]).astype(np.float32)


def overlay_mask(rgb: np.ndarray, enemy: np.ndarray, teammate: np.ndarray) -> np.ndarray:
    out = rgb.copy().astype(np.float32)
    enemy_full = cv2.resize(enemy, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0.5
    team_full = cv2.resize(teammate, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0.5
    out[enemy_full] = out[enemy_full] * 0.45 + np.array([255, 40, 40]) * 0.55
    out[team_full] = out[team_full] * 0.45 + np.array([40, 140, 255]) * 0.55
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_other_player(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.copy().astype(np.float32)
    full = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0.5
    out[full] = out[full] * 0.45 + np.array([255, 190, 40]) * 0.55
    return np.clip(out, 0, 255).astype(np.uint8)


def gray_image(ch: np.ndarray) -> Image.Image:
    arr = np.clip(ch * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def semantic_color_image(ch: np.ndarray) -> Image.Image:
    palette = np.asarray([
        [0, 0, 0],
        [230, 57, 70],
        [29, 53, 87],
        [69, 123, 157],
        [42, 157, 143],
        [233, 196, 106],
        [244, 162, 97],
        [131, 56, 236],
        [255, 0, 110],
        [58, 134, 255],
        [138, 201, 38],
        [255, 202, 58],
        [106, 76, 147],
        [25, 130, 196],
        [198, 40, 40],
        [0, 121, 107],
        [123, 31, 162],
        [251, 133, 0],
        [38, 70, 83],
        [2, 48, 71],
        [142, 202, 230],
        [33, 158, 188],
        [255, 183, 3],
        [251, 85, 114],
        [144, 190, 109],
        [249, 65, 68],
        [87, 117, 144],
    ], dtype=np.uint8)
    idx = np.rint(np.clip(ch, 0.0, 1.0) * 26.0).astype(np.int32)
    idx = np.clip(idx, 0, len(palette) - 1)
    return Image.fromarray(palette[idx])


def semantic_overlay(rgb: np.ndarray, semantic: np.ndarray) -> Image.Image:
    rgb_img = Image.fromarray(rgb).resize((semantic.shape[1], semantic.shape[0]), Image.Resampling.BILINEAR)
    rgb_arr = np.asarray(rgb_img).astype(np.float32)
    sem = np.asarray(semantic_color_image(semantic)).astype(np.float32)
    mask = semantic > 0
    out = rgb_arr.copy()
    out[mask] = out[mask] * 0.55 + sem[mask] * 0.45
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def mask_image(ch: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    arr = np.zeros((ch.shape[0], ch.shape[1], 3), dtype=np.uint8)
    arr[ch > 0.5] = color
    return Image.fromarray(arr)


def make_qa(
    out_path: Path,
    rgb: np.ndarray,
    depth: np.ndarray,
    semantic: np.ndarray,
    player_channels: np.ndarray,
    meta: dict[str, Any],
) -> None:
    h, w = depth.shape
    rgb_small = Image.fromarray(rgb).resize((w, h), Image.Resampling.BILINEAR)
    if player_channels.shape[0] == 4:
        overlay = Image.fromarray(overlay_other_player(rgb, player_channels[0])).resize((w, h), Image.Resampling.BILINEAR)
        mask_label = "other player mask"
        mask_panel = mask_image(player_channels[0], (255, 190, 40))
    else:
        overlay = Image.fromarray(overlay_mask(rgb, player_channels[0], player_channels[1])).resize((w, h), Image.Resampling.BILINEAR)
        mask_label = "enemy mask"
        mask_panel = mask_image(player_channels[0], (255, 40, 40))
    panels = [
        ("rgb", rgb_small),
        ("depth", gray_image(depth)),
        ("semantic", semantic_color_image(semantic)),
        ("player overlay", overlay),
        ("semantic overlay", semantic_overlay(rgb, semantic)),
        (mask_label, mask_panel),
    ]
    pad = 26
    canvas = Image.new("RGB", (w * 3, (h + pad) * 2), (250, 250, 247))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, img) in enumerate(panels):
        x = (idx % 3) * w
        y = (idx // 3) * (h + pad)
        canvas.paste(img, (x, y + pad))
        draw.text((x + 8, y + 6), label, fill=(20, 20, 20))
    visible = meta.get("visible_players")
    if visible is None:
        visible = meta.get("memory_projected_players", [])
    draw.text((8, canvas.height - 18), f"visible players with mask: {len(visible)}", fill=(20, 20, 20))
    canvas.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", required=True, type=Path)
    ap.add_argument("--ego-stem", required=True)
    ap.add_argument("--frame-index", type=int, required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--height", type=int, default=180)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--color-tolerance", type=int, default=6)
    args = ap.parse_args()

    rgb_bgr = read_frame(args.episode_dir / f"{args.ego_stem}.mp4", args.frame_index)
    depth_bgr = read_frame(args.episode_dir / f"{args.ego_stem}_depth.mkv", args.frame_index)
    seg_bgr = read_frame(args.episode_dir / f"{args.ego_stem}_seg.mkv", args.frame_index)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    output_hw = (args.height, args.width)
    env_depth = make_depth(depth_bgr, output_hw)
    env_semantic = make_semantic(seg_bgr, output_hw)
    depth_full = depth_bgr.astype(np.float32).mean(axis=2) / 255.0
    player_channels, player_meta = build_masks(
        args.episode_dir,
        args.ego_stem,
        args.frame_index,
        seg_bgr,
        depth_full,
        output_hw,
        args.color_tolerance,
    )

    dense = np.concatenate([
        env_depth[None, ...],
        env_semantic[None, ...],
        player_channels,
    ], axis=0).astype(np.float32)
    channels = [
        "env_depth_norm_from_depth_mkv",
        "env_semantic_nonzero_seg_v0",
        "enemy_mask_from_seg_colormap",
        "teammate_mask_from_seg_colormap",
        "player_depth_norm",
        "player_yaw_sin",
        "player_yaw_cos",
    ]
    meta = {
        "episode_dir": str(args.episode_dir),
        "ego_stem": args.ego_stem,
        "frame_index": args.frame_index,
        "output_shape": list(dense.shape),
        "channels": channels,
        "depth_note": "v0 normalized 8-bit luminance from depth.mkv; verify 16-bit packing before final training.",
        **player_meta,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "dense_condition_v0.npz", dense=dense)
    (args.out_dir / "dense_condition_meta_v0.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_qa(args.out_dir / "dense_condition_qa_v0.png", rgb, env_depth, env_semantic, player_channels, meta)
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "shape": list(dense.shape),
        "visible_players_with_mask": len(meta["visible_players"]),
        "channels": channels,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
