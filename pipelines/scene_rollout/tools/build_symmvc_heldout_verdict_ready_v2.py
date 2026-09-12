#!/usr/bin/env python3
from __future__ import annotations
from runtime_paths import source_path

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(str(source_path('scene', '')))
V1_DIR = ROOT / "output/symmvc_strong_pairs_heldout_v1"
DEFAULT_PAIR_INDEX = V1_DIR / "pair_index_strong_covis_heldout_v1.json"
DEFAULT_OUT_DIR = ROOT / "output/symmvc_strong_pairs_heldout_v2"
DEFAULT_MANIFESTS = [
    ROOT / "output/memory_v2_all188_diversity_stride_manifest_20260629_v0/step5_materialized_manifest_dlc_v0.jsonl",
    ROOT / "output/memory_v2_all188_diversity_stride_manifest_20260629_v0/v2_all188_positive_diversity_tier20h_source_manifest_v0.jsonl",
    ROOT / "output/memory_v2_all188_diversity_stride_manifest_20260629_v0/v2_all188_positive_diversity_tier69h_source_manifest_v0.jsonl",
]


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def norm_path(p: str | None) -> str | None:
    if not p:
        return None
    return p.replace("/mnt/workspace/zhengqi/", "/mnt/data/pku/zhengqi/")


def file_info(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {"path": str(path), "size": st.st_size, "mtime_unix": st.st_mtime, "mtime_local": time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(st.st_mtime))}


def load_clip_index(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    idx: dict[str, dict[str, Any]] = {}
    stats = []
    for path in paths:
        if not path.exists():
            stats.append({"path": str(path), "exists": False})
            continue
        n = 0
        with_dir = 0
        for row in read_jsonl(path):
            n += 1
            cid = row.get("clip_id")
            cdir = norm_path(row.get("clip_dir") or row.get("sample_dir"))
            if not cid or not cdir:
                continue
            with_dir += 1
            old = idx.get(cid)
            # Prefer an existing materialized directory with all scorer-required files.
            has_required = all((Path(cdir) / fn).exists() for fn in ["video.mp4", "poses.npy", "intrinsics.npy", "meta.json"])
            old_has_required = bool(old and old.get("has_required"))
            if old is None or (has_required and not old_has_required):
                out = dict(row)
                out["clip_dir"] = cdir
                out["has_required"] = has_required
                idx[str(cid)] = out
        stats.append({"path": str(path), "exists": True, "rows": n, "rows_with_clip_dir": with_dir, **file_info(path)})
    return idx, {"manifest_stats": stats, "unique_clip_ids": len(idx), "with_required_files": sum(1 for v in idx.values() if v.get("has_required"))}


def raw_map_from_clip(row: dict[str, Any]) -> dict[int, int] | None:
    cdir = Path(str(row["clip_dir"]))
    meta_path = cdir / "meta.json"
    raw_indices = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        raw_indices = meta.get("raw_indices")
    if raw_indices is None:
        raw_indices = row.get("raw_indices")
    if not isinstance(raw_indices, list) or not raw_indices:
        return None
    return {int(raw): i for i, raw in enumerate(raw_indices)}


def rebuild_pair(pair: dict[str, Any], clip_idx: dict[str, dict[str, Any]], min_pct: float, min_frames: int) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
    cid_a, cid_b = str(pair.get("clipA_id")), str(pair.get("clipB_id"))
    row_a, row_b = clip_idx.get(cid_a), clip_idx.get(cid_b)
    aux = {"clipA_id": cid_a, "clipB_id": cid_b}
    if row_a is None or row_b is None:
        return None, "missing_clip_manifest", aux
    if not row_a.get("has_required") or not row_b.get("has_required"):
        return None, "missing_required_materialized_files", {**aux, "clipA_dir": row_a.get("clip_dir"), "clipB_dir": row_b.get("clip_dir"), "A_has_required": row_a.get("has_required"), "B_has_required": row_b.get("has_required")}
    map_a = raw_map_from_clip(row_a)
    map_b = raw_map_from_clip(row_b)
    if map_a is None or map_b is None:
        return None, "missing_raw_indices", {**aux, "clipA_dir": row_a.get("clip_dir"), "clipB_dir": row_b.get("clip_dir")}
    frame_by_raw = {int(raw): (float(pa), float(pb)) for _, raw, pa, pb in pair.get("mutual_covis_frames") or []}
    aligned = []
    for raw in sorted(set(map_a) & set(map_b) & set(frame_by_raw)):
        pa, pb = frame_by_raw[raw]
        aligned.append([int(map_a[raw]), int(raw), float(pa), float(pb)])
    if not aligned:
        return None, "no_exact_raw_indices_intersection", {**aux, "clipA_dir": row_a.get("clip_dir"), "clipB_dir": row_b.get("clip_dir")}
    min_pair_pct = min(float(pair.get("a_sees_b_max_pixel_percent") or 0.0), float(pair.get("b_sees_a_max_pixel_percent") or 0.0))
    if min_pair_pct < min_pct:
        return None, "below_percent_gate", {**aux, "min_bidirectional_pixel_percent": min_pair_pct, "n_exact_aligned_frames": len(aligned)}
    if len(aligned) < min_frames:
        return None, "below_exact_aligned_frame_gate", {**aux, "min_bidirectional_pixel_percent": min_pair_pct, "n_exact_aligned_frames": len(aligned)}
    out = dict(pair)
    out["clipA_dir"] = row_a["clip_dir"]
    out["clipB_dir"] = row_b["clip_dir"]
    out["mutual_covis_frames_v1_original_count"] = int(pair.get("n_mutual_covis_frames") or 0)
    out["raw_mutual_covis_frames_v1_original"] = int(pair.get("raw_mutual_covis_frames") or 0)
    out["mutual_covis_frames"] = aligned
    out["n_mutual_covis_frames"] = len(aligned)
    out["entry_kind"] = "strong_covis_heldout_verdict_ready_v2_exact_raw_aligned"
    out["alignment_policy_v2"] = {
        "policy": "exact absolute raw frame intersection consumed as local scorer index",
        "clipA_raw_start": row_a.get("raw_start"),
        "clipB_raw_start": row_b.get("raw_start"),
        "clipA_dir": row_a["clip_dir"],
        "clipB_dir": row_b["clip_dir"],
        "min_bidirectional_pixel_percent": round(min_pair_pct, 6),
        "min_percent_gate": min_pct,
        "min_exact_aligned_frames_gate": min_frames,
    }
    out["min_bidirectional_pixel_percent"] = round(min_pair_pct, 6)
    out["score_v2"] = round(min_pair_pct * len(aligned), 6)
    return out, "kept", {**aux, "min_bidirectional_pixel_percent": min_pair_pct, "n_exact_aligned_frames": len(aligned)}


def percentile(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    xs = sorted(vals)
    pos = (len(xs) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-index", type=Path, default=DEFAULT_PAIR_INDEX)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--manifest", type=Path, action="append", default=[])
    ap.add_argument("--min-percent", type=float, default=0.3)
    ap.add_argument("--min-exact-aligned-frames", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=48)
    args = ap.parse_args()

    manifests = args.manifest or DEFAULT_MANIFESTS
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs_all = json.loads(args.pair_index.read_text(encoding="utf-8"))
    clip_idx, clip_report = load_clip_index(manifests)

    rebuilt = []
    reject_counts = Counter()
    reject_examples: dict[str, list[dict[str, Any]]] = {}
    for p in pairs_all:
        rp, reason, aux = rebuild_pair(p, clip_idx, args.min_percent, args.min_exact_aligned_frames)
        if rp is None:
            reject_counts[reason] += 1
            reject_examples.setdefault(reason, [])
            if len(reject_examples[reason]) < 10:
                reject_examples[reason].append(aux)
        else:
            rebuilt.append(rp)

    rebuilt.sort(key=lambda p: (-float(p["score_v2"]), -float(p["min_bidirectional_pixel_percent"]), -int(p["n_mutual_covis_frames"]), str(p["clipA_id"]), str(p["clipB_id"])))
    selected = rebuilt[: args.top_k]

    pair_path = args.out_dir / "pair_index_verdict_ready_v2.json"
    win_path = args.out_dir / "window_manifest_verdict_ready_v2.jsonl"
    report_path = args.out_dir / "verdict_ready_v2_build_report.json"
    sens_path = args.out_dir / "visibility_percent_sensitivity_v2.json"

    pair_path.write_text(json.dumps(selected, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    with win_path.open("w", encoding="utf-8") as f:
        for p in selected:
            for side in ["A", "B"]:
                f.write(json.dumps({
                    "pair_id": str(p.get("clipA_id")) + "__" + str(p.get("clipB_id")),
                    "side": side,
                    "match_id": p.get("match_id"),
                    "episode": p.get("episode"),
                    "ego": p.get(f"ego{side}"),
                    "clip_id": p.get(f"clip{side}_id"),
                    "clip_dir": p.get(f"clip{side}_dir"),
                    "raw_start": p.get("alignment_policy_v2", {}).get(f"clip{side}_raw_start"),
                    "exact_aligned_raw_frames": [int(x[1]) for x in p.get("mutual_covis_frames") or []],
                }, ensure_ascii=False, separators=(",", ":")) + "\n")

    all_min_pct = [min(float(p.get("a_sees_b_max_pixel_percent") or 0.0), float(p.get("b_sees_a_max_pixel_percent") or 0.0)) for p in pairs_all]
    materializable_min_pct = [float(p["min_bidirectional_pixel_percent"]) for p in rebuilt]
    sensitivity = []
    for pct in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]:
        rows = [p for p in rebuilt if float(p["min_bidirectional_pixel_percent"]) >= pct and int(p["n_mutual_covis_frames"]) >= args.min_exact_aligned_frames]
        sensitivity.append({"min_percent": pct, "min_exact_aligned_frames": args.min_exact_aligned_frames, "count": len(rows), "top48_available": min(len(rows), args.top_k)})
    sens_path.write_text(json.dumps({
        "all_v1_pairs": {"count": len(all_min_pct), "p50": percentile(all_min_pct, 50), "p90": percentile(all_min_pct, 90)},
        "materializable_exact_aligned_after_default_gates": {"count": len(materializable_min_pct), "p50": percentile(materializable_min_pct, 50), "p90": percentile(materializable_min_pct, 90)},
        "sensitivity": sensitivity,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    report = {
        "kind": "symmvc_heldout_verdict_ready_build_v2",
        "created_at_unix": time.time(),
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "input_pair_index": file_info(args.pair_index),
        "clip_index": clip_report,
        "params": {"min_percent": args.min_percent, "min_exact_aligned_frames": args.min_exact_aligned_frames, "top_k": args.top_k},
        "counts": {"input_pairs": len(pairs_all), "eligible_after_exact_alignment_and_gates": len(rebuilt), "selected_pairs": len(selected), "reject_counts": dict(sorted(reject_counts.items()))},
        "output_files": {"pair_index": str(pair_path), "window_manifest": str(win_path), "report_json": str(report_path), "sensitivity_json": str(sens_path)},
        "reject_examples": reject_examples,
        "selected_pair_visibility": [
            {"rank": i + 1, "pair_id": str(p.get("clipA_id")) + "__" + str(p.get("clipB_id")), "match_id": p.get("match_id"), "episode": p.get("episode"), "a_sees_b_max_pixel_percent": p.get("a_sees_b_max_pixel_percent"), "b_sees_a_max_pixel_percent": p.get("b_sees_a_max_pixel_percent"), "min_bidirectional_pixel_percent": p.get("min_bidirectional_pixel_percent"), "n_exact_aligned_frames": p.get("n_mutual_covis_frames"), "score_v2": p.get("score_v2")} for i, p in enumerate(selected)
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"WROTE {pair_path} selected_pairs={len(selected)} eligible={len(rebuilt)}")
    print(f"WROTE {win_path} rows={len(selected)*2}")
    print(f"WROTE {report_path}")
    print(f"WROTE {sens_path}")


if __name__ == "__main__":
    main()
