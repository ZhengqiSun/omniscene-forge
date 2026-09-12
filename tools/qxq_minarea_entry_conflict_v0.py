#!/usr/bin/env python3
"""V4: min-area gate vs mid-entry chunk conflict quantification (F1-4).

For every mid-entry aligned chunk (frame[start] role=context, later in-chunk
frame positive) in the pilot1/pilot2 motionclean train splits, load the memory
dense npz channel 3 for the 3 chunk frames, pool to the 30x52 token grid using
the exact teacher_player_latent_mask_v0.py path (mask = dense[3] > 0.5, then
cv2.resize float32 INTER_AREA, > 0.0), and count token cells.

Reports:
  - entry-frame cell count distribution (P10/P50/P90 etc.)
  - per-chunk aggregated (3-frame sum) cell counts
  - fraction of mid-entry chunks skipped under min_pixels in {1,2,4,8}
    (skip iff cells < min_pixels), both chunk-sum gating and per-frame
    entry-frame gating, split by pilot1/pilot2.

Read-only on Zhengqi tree; writes only the output JSON under qxq output dir.
"""
from runtime_paths import source_path

import json
import os
import sys
from collections import Counter

import numpy as np
import cv2

ZQ_ROOT = str(source_path('scene', ''))
RUNS = os.path.join(ZQ_ROOT, "output/memory_dense_adapter_v0/runs/memory_dense_lingbot_fullsubset_large_v0")
QXQ_OUT = str(source_path('project', 'output/qxq_a3_preflight_20260612'))
AUDIT_SUMMARY = os.path.join(QXQ_OUT, "train_entry_event_audit_v0.json")

LATENT_FRAMES = 21
CHUNK_SIZE = 3
CHUNK_STARTS = [0, 3, 6, 9, 12, 15, 18]
TOKEN_HW = (30, 52)        # token grid per task spec


def _default_dense_hw():
    """Dense grid from the real teacher path, not a local copy.

    teacher_player_latent_mask_v0 reads the grid from map_memory_training_data_v0
    at call time; this tool used to hardcode (176, 320), so after the v2 switch to
    (240, 416) the "exact replica" silently stopped replicating. Fall back to the
    legacy constant only if the teacher module cannot be imported.
    """
    import importlib.util
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        name = "teacher_player_latent_mask_v0"
        mod = sys.modules.get(name)
        if mod is None:
            spec = importlib.util.spec_from_file_location(
                name, os.path.join(tools_dir, name + ".py"))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod          # register before exec (dataclass lookup)
            try:
                spec.loader.exec_module(mod)
            except BaseException:
                sys.modules.pop(name, None)
                raise
        return tuple(mod.default_dense_hw())
    except Exception as exc:
        print("[warn] teacher_player_latent_mask_v0 unavailable (%r); "
              "falling back to legacy (176, 320)" % (exc,), file=sys.stderr)
        return (176, 320)


# Override with --dense-hw H W (needed to reproduce pre-v2 numbers on v1 npz).
if "--dense-hw" in sys.argv:
    _i = sys.argv.index("--dense-hw")
    DENSE_HW = (int(sys.argv[_i + 1]), int(sys.argv[_i + 2]))
else:
    DENSE_HW = _default_dense_hw()
MIN_PIXELS_GRID = [1, 2, 4, 8]


def resize_mask_any(mask, output_hw):
    """Exact replica of teacher_player_latent_mask_v0.resize_mask_any."""
    height, width = int(output_hw[0]), int(output_hw[1])
    if mask.shape == (height, width):
        return mask.astype(bool, copy=True)
    pooled = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
    return pooled > 0.0


def token_cells_from_dense_npz(path):
    dense = np.load(path)["dense"]
    mask = dense[3] > 0.5
    if mask.shape != DENSE_HW:
        # A two-hop resample (native -> DENSE_HW -> token) only ever inflates the
        # mask under INTER_AREA + >0; refuse it instead of reporting a wrong count.
        raise SystemExit(
            "dense npz native grid %s != DENSE_HW %s (%s).\n"
            "  This tool must run on the grid it is analysing. Pass "
            "--dense-hw %d %d to pin the legacy grid, or point it at v2 data."
            % (tuple(mask.shape), tuple(DENSE_HW), path, mask.shape[0], mask.shape[1])
        )
    tok = resize_mask_any(mask, TOKEN_HW)
    return int(tok.sum())


def pct(values, q):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def dist_stats(values):
    if not values:
        return {"n": 0}
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "p10": pct(values, 10),
        "p25": pct(values, 25),
        "p50": pct(values, 50),
        "p75": pct(values, 75),
        "p90": pct(values, 90),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "zero_count": int((a == 0).sum()),
    }


def audit_view(name, view_info):
    manifest_path = view_info["manifest"]
    cache_path = view_info["cache_manifest"]
    with open(manifest_path) as f:
        man = json.load(f)
    by_id = {}
    for s in man["samples"]:
        by_id[s["sample_id"]] = (s["selection_role"], s["dense_path"])

    rows = []
    with open(cache_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("map_memory_split") == "train":
                rows.append(r)

    chunks = []
    n_chunks_total = 0
    for r in rows:
        sids = r["map_memory_sample_ids"]
        if len(sids) != LATENT_FRAMES:
            raise ValueError(f"{r['clip_id']}: {len(sids)} sample ids")
        roles = [by_id[sid][0] for sid in sids]
        rb = [ro == "positive" for ro in roles]
        for start in CHUNK_STARTS:
            crb = rb[start:start + CHUNK_SIZE]
            n_chunks_total += 1
            if (not crb[0]) and any(crb[1:]):
                # mid-entry chunk; entry frame = first positive in chunk
                entry_off = next(i for i in range(1, CHUNK_SIZE) if crb[i])
                chunks.append({
                    "clip_id": r["clip_id"],
                    "chunk_start": start,
                    "entry_frame": start + entry_off,
                    "entry_offset_in_chunk": entry_off,
                    "pattern": "".join("P" if b else "C" for b in crb),
                    "sample_ids": sids[start:start + CHUNK_SIZE],
                })
    print(f"[{name}] train clips={len(rows)} aligned_chunks={n_chunks_total} mid_entry_chunks={len(chunks)}", flush=True)

    # load dense npz and compute token cells
    per_chunk = []
    pattern_counter = Counter()
    for k, ch in enumerate(chunks):
        cells = []
        for sid in ch["sample_ids"]:
            dense_rel = by_id[sid][1]
            dense_abs = dense_rel if os.path.isabs(dense_rel) else os.path.join(ZQ_ROOT, dense_rel)
            cells.append(token_cells_from_dense_npz(dense_abs))
        entry_cells = cells[ch["entry_offset_in_chunk"]]
        rec = {
            "clip_id": ch["clip_id"],
            "chunk_start": ch["chunk_start"],
            "entry_frame": ch["entry_frame"],
            "pattern": ch["pattern"],
            "frame_cells": cells,
            "entry_frame_cells": entry_cells,
            "chunk_cells_sum": int(sum(cells)),
            "chunk_cells_max": int(max(cells)),
        }
        per_chunk.append(rec)
        pattern_counter[ch["pattern"]] += 1
        if (k + 1) % 100 == 0:
            print(f"[{name}] processed {k + 1}/{len(chunks)}", flush=True)

    entry_cells_all = [r["entry_frame_cells"] for r in per_chunk]
    chunk_sum_all = [r["chunk_cells_sum"] for r in per_chunk]
    chunk_max_all = [r["chunk_cells_max"] for r in per_chunk]

    gates = {}
    for mp in MIN_PIXELS_GRID:
        n = len(per_chunk)
        skipped_sum = sum(1 for r in per_chunk if r["chunk_cells_sum"] < mp)
        skipped_max = sum(1 for r in per_chunk if r["chunk_cells_max"] < mp)
        skipped_entry = sum(1 for r in per_chunk if r["entry_frame_cells"] < mp)
        gates[str(mp)] = {
            "chunk_sum_skipped": skipped_sum,
            "chunk_sum_skip_rate": skipped_sum / n if n else None,
            "chunk_max_frame_skipped": skipped_max,
            "chunk_max_frame_skip_rate": skipped_max / n if n else None,
            "entry_frame_below": skipped_entry,
            "entry_frame_below_rate": skipped_entry / n if n else None,
        }

    return {
        "view": view_info["view"],
        "train_clips": len(rows),
        "aligned_chunks_total": n_chunks_total,
        "mid_entry_chunks": len(per_chunk),
        "patterns": dict(pattern_counter),
        "entry_frame_cells_dist": dist_stats(entry_cells_all),
        "chunk_cells_sum_dist": dist_stats(chunk_sum_all),
        "chunk_cells_max_dist": dist_stats(chunk_max_all),
        "min_pixels_gates": gates,
        "per_chunk": per_chunk,
    }


def main():
    with open(AUDIT_SUMMARY) as f:
        summary = json.load(f)
    views = summary["views"]

    out = {
        "audit": "qxq_minarea_entry_conflict_v0",
        "date": "2026-06-12",
        "task": "V4/F1-4: min-area gate vs mid-entry chunk conflict",
        "pooling": {
            "method": "replica of teacher_player_latent_mask_v0.py: mask = dense_npz['dense'][3] > 0.5; "
                      "resize_mask_any (cv2.resize float32 INTER_AREA, > 0.0) to DENSE_HW then to token grid",
            "dense_hw": list(DENSE_HW),
            "token_hw": list(TOKEN_HW),
            "source_lines": "teacher_player_latent_mask_v0.py L135-145 (surrogate dense[3]>0.5), "
                            "L168-173 (resize_mask_any), L175-187 (teacher_player_mask_latent)",
        },
        "gate_semantics": "chunk skipped iff cells < min_pixels; chunk_sum = sum of per-frame token cells over the 3 "
                          "chunk frames (task spec); chunk_max_frame and entry_frame variants reported as secondary",
        "min_pixels_grid": MIN_PIXELS_GRID,
        "views": {},
    }

    merged_entry = []
    merged_sum = []
    merged_max = []
    merged_n_chunks = 0
    for name, info in views.items():
        res = audit_view(name, info)
        out["views"][name] = res
        merged_entry += [r["entry_frame_cells"] for r in res["per_chunk"]]
        merged_sum += [r["chunk_cells_sum"] for r in res["per_chunk"]]
        merged_max += [r["chunk_cells_max"] for r in res["per_chunk"]]
        merged_n_chunks += res["mid_entry_chunks"]

    merged_gates = {}
    for mp in MIN_PIXELS_GRID:
        sk_sum = sum(1 for v in merged_sum if v < mp)
        sk_max = sum(1 for v in merged_max if v < mp)
        sk_ent = sum(1 for v in merged_entry if v < mp)
        merged_gates[str(mp)] = {
            "chunk_sum_skipped": sk_sum,
            "chunk_sum_skip_rate": sk_sum / merged_n_chunks if merged_n_chunks else None,
            "chunk_max_frame_skipped": sk_max,
            "chunk_max_frame_skip_rate": sk_max / merged_n_chunks if merged_n_chunks else None,
            "entry_frame_below": sk_ent,
            "entry_frame_below_rate": sk_ent / merged_n_chunks if merged_n_chunks else None,
        }
    out["merged"] = {
        "mid_entry_chunks": merged_n_chunks,
        "entry_frame_cells_dist": dist_stats(merged_entry),
        "chunk_cells_sum_dist": dist_stats(merged_sum),
        "chunk_cells_max_dist": dist_stats(merged_max),
        "min_pixels_gates": merged_gates,
    }

    out_path = os.path.join(QXQ_OUT, "minarea_entry_conflict_v0.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", out_path)

    # compact console report
    for name in out["views"]:
        v = out["views"][name]
        print(f"== {name} == mid_entry_chunks={v['mid_entry_chunks']}")
        d = v["entry_frame_cells_dist"]
        print(f"  entry cells P10/P50/P90 = {d['p10']}/{d['p50']}/{d['p90']} zero={d['zero_count']}")
        for mp in MIN_PIXELS_GRID:
            g = v["min_pixels_gates"][str(mp)]
            print(f"  min_pixels={mp}: chunk_sum skip {g['chunk_sum_skipped']} ({g['chunk_sum_skip_rate']:.4f}) "
                  f"entry_frame below {g['entry_frame_below']} ({g['entry_frame_below_rate']:.4f})")
    print("== merged ==", merged_n_chunks)
    for mp in MIN_PIXELS_GRID:
        g = out["merged"]["min_pixels_gates"][str(mp)]
        print(f"  min_pixels={mp}: chunk_sum skip_rate={g['chunk_sum_skip_rate']:.4f} "
              f"entry_frame_below_rate={g['entry_frame_below_rate']:.4f}")


if __name__ == "__main__":
    sys.exit(main())
