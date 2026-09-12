#!/usr/bin/env python3
"""Find pairs of release clips whose egos can see each other in the same time window.

Scans an aligned cache manifest (light_dust2_pilot release), groups rows by
(game_id, episode), keeps pairs of different egos whose raw windows overlap,
then checks both players' *_player_visibility.json for mutual visibility inside
the overlap. Outputs a ranked JSON so inference can run on real mutually-visible
multi-view moments.

Read-only over raw data; writes one JSON report.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

PLAYER_NUM_RE = re.compile(r"team_(?P<team>\d+)_player_(?P<player>\d+)")


def clip_motion(poses_path: str | None) -> dict:
    """Camera translation path length (game units) and total rotation (deg) from poses.npy."""
    out = {"path_len": None, "rot_deg": None}
    if not poses_path:
        return out
    try:
        poses = np.load(poses_path)  # (T,4,4) c2w
    except (OSError, ValueError):
        return out
    t = poses[:, :3, 3]
    out["path_len"] = float(np.linalg.norm(np.diff(t, axis=0), axis=1).sum())
    rot = 0.0
    for i in range(len(poses) - 1):
        r = poses[i, :3, :3].T @ poses[i + 1, :3, :3]
        c = max(-1.0, min(1.0, (np.trace(r) - 1.0) / 2.0))
        rot += math.degrees(math.acos(c))
    out["rot_deg"] = float(rot)
    return out


def player_idx(stem: str) -> int | None:
    m = PLAYER_NUM_RE.search(stem)
    return int(m.group("player")) if m else None


def visible_frames(vis_path: Path, target_idx: int, lo: int, hi: int) -> list[tuple[int, float]]:
    """Frames in [lo, hi] where target_idx is visible, with max pixel percent."""
    try:
        vis = json.loads(vis_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    entry = vis.get(str(target_idx))
    if not isinstance(entry, dict):
        return []
    out: dict[int, float] = {}
    for row in entry.get("ranges", []):
        try:
            rlo, rhi = row["range"]
            score = float(row.get("pixel_percent_max", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        for f in range(max(lo, int(rlo)), min(hi, int(rhi)) + 1):
            if score > out.get(f, -1.0):
                out[f] = score
    return sorted(out.items())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-manifest", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--window-raw-frames", type=int, default=160,
                    help="raw-frame span of one 81-frame clip window")
    ap.add_argument("--min-mutual-frames", type=int, default=8)
    ap.add_argument("--rank-by", choices=["visibility", "motion"], default="visibility")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.cache_manifest.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if (r.get("map_memory_split") or r.get("split")) == args.split]

    grp: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        grp[(r.get("game_id") or r.get("hash"), r.get("episode"))].append(r)

    pairs = []
    for (game, episode), items in grp.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if a["player_stem"] == b["player_stem"]:
                    continue
                sa, sb = int(a["raw_start"]), int(b["raw_start"])
                lo = max(sa, sb)
                hi = min(sa, sb) + args.window_raw_frames
                if hi - lo < args.window_raw_frames // 2:
                    continue
                ia, ib = player_idx(a["player_stem"]), player_idx(b["player_stem"])
                if ia is None or ib is None:
                    continue
                ep_dir = Path(a["action_json"]).parent
                vis_a = ep_dir / f"{a['player_stem']}_player_visibility.json"
                vis_b = ep_dir / f"{b['player_stem']}_player_visibility.json"
                a_sees_b = visible_frames(vis_a, ib, lo, hi)
                b_sees_a = visible_frames(vis_b, ia, lo, hi)
                mutual = sorted(set(f for f, _ in a_sees_b) & set(f for f, _ in b_sees_a))
                if len(mutual) < args.min_mutual_frames:
                    continue
                motion_a = clip_motion(a.get("poses"))
                motion_b = clip_motion(b.get("poses"))
                pairs.append({
                    "game_id": game,
                    "episode": episode,
                    "path_len_a": motion_a["path_len"],
                    "path_len_b": motion_b["path_len"],
                    "rot_deg_a": motion_a["rot_deg"],
                    "rot_deg_b": motion_b["rot_deg"],
                    "min_path_len": min(motion_a["path_len"] or 0.0, motion_b["path_len"] or 0.0),
                    "clip_a": a["clip_id"],
                    "clip_b": b["clip_id"],
                    "player_a": a["player_stem"],
                    "player_b": b["player_stem"],
                    "raw_start_a": sa,
                    "raw_start_b": sb,
                    "overlap_raw": [lo, hi],
                    "a_sees_b_frames": len(a_sees_b),
                    "b_sees_a_frames": len(b_sees_a),
                    "mutual_frames": len(mutual),
                    "mutual_frame_span": [mutual[0], mutual[-1]] if mutual else None,
                    "a_sees_b_max_pixel_pct": max((s for _, s in a_sees_b), default=0.0),
                    "b_sees_a_max_pixel_pct": max((s for _, s in b_sees_a), default=0.0),
                    "cross_team": ("team_2" in a["player_stem"]) != ("team_2" in b["player_stem"]),
                })
    if args.rank_by == "motion":
        # high camera motion on BOTH sides (the visible player's own path = how much
        # they move inside the other ego's frame), gated by enough mutual visibility
        pairs.sort(key=lambda p: (p["min_path_len"], p["mutual_frames"]), reverse=True)
    else:
        pairs.sort(key=lambda p: (p["mutual_frames"],
                                  min(p["a_sees_b_max_pixel_pct"], p["b_sees_a_max_pixel_pct"])),
                   reverse=True)
    report = {
        "kind": "mutual_visible_clip_pairs_v0",
        "cache_manifest": str(args.cache_manifest),
        "split": args.split,
        "row_count": len(rows),
        "pair_count": len(pairs),
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))
    print(f"split={args.split} rows={len(rows)} mutual pairs={len(pairs)} -> {args.output}")
    for p in pairs[:6]:
        print(f"  {p['episode']} {p['player_a']} <-> {p['player_b']} "
              f"mutual={p['mutual_frames']} pix=({p['a_sees_b_max_pixel_pct']:.1f},{p['b_sees_a_max_pixel_pct']:.1f}) "
              f"path=({p['path_len_a']:.0f},{p['path_len_b']:.0f}) rot=({p['rot_deg_a']:.0f},{p['rot_deg_b']:.0f}) "
              f"cross_team={p['cross_team']}")


if __name__ == "__main__":
    main()
