#!/usr/bin/env python3
"""Paired per-chunk diagnostic for memory dense adapter ablation eval JSONs.

Reads the `rows` section written by evaluate_memory_dense_adapter_ablation_v0.py
and answers: where does the true-vs-shuffled signal live? Buckets paired
region-loss differences by sigma (noise level), region pixel count, and clip,
so we can tell "effect is genuinely tiny" apart from "effect is diluted by
noise-dominated chunks".

Read-only over inputs; writes one analysis JSON per input eval file.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

BASELINES = ["shuffled_dense", "player_shuffle", "blank_dense"]
TRUE_VARIANT = "true_dense"


def chunk_key(row: dict) -> tuple:
    return (row.get("clip_id"), row.get("record_index"), row.get("chunk_ord"), row.get("chunk_start"))


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def median(xs: list[float]) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    return ys[mid] if n % 2 else 0.5 * (ys[mid - 1] + ys[mid])


def summarize_pairs(pairs: list[dict]) -> dict:
    diffs = [p["diff"] for p in pairs]
    base_losses = [p["base_region_loss"] for p in pairs]
    n = len(diffs)
    if n == 0:
        return {"n": 0}
    m = mean(diffs)
    var = sum((d - m) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    sem = math.sqrt(var / n) if n > 1 else 0.0
    mean_base = mean(base_losses)
    return {
        "n": n,
        "mean_paired_diff": m,
        "median_paired_diff": median(diffs),
        "sem": sem,
        "ci95": [m - 1.96 * sem, m + 1.96 * sem],
        "frac_true_better": sum(1 for d in diffs if d > 0) / n,
        "mean_baseline_region_loss": mean_base,
        "relative_reduction": (m / mean_base) if mean_base else None,
    }


def bucket_label_sigma(sigma: float, edges: list[float]) -> str:
    for i, e in enumerate(edges):
        if sigma <= e:
            return f"sigma_q{i + 1}_le_{e:.3f}"
    return f"sigma_q{len(edges) + 1}_gt_{edges[-1]:.3f}"


def bucket_label_pixels(px: int) -> str:
    if px <= 32:
        return "region_px_001_032"
    if px <= 64:
        return "region_px_033_064"
    if px <= 128:
        return "region_px_065_128"
    if px <= 256:
        return "region_px_129_256"
    return "region_px_257_plus"


def analyze(eval_path: Path, top_clips: int) -> dict:
    data = json.loads(eval_path.read_text())
    rows = data.get("rows") or []
    by_chunk: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_chunk[chunk_key(row)][row.get("variant")] = row

    result: dict = {
        "kind": "ablation_eval_rows_analysis_v0",
        "eval_json": str(eval_path),
        "adapter_checkpoint": data.get("adapter_checkpoint"),
        "eval_chunk_count": data.get("eval_chunk_count"),
        "comparisons": {},
        "notes": [],
    }

    for baseline in BASELINES:
        pairs: list[dict] = []
        sigma_mismatch = 0
        for key, variants in by_chunk.items():
            t = variants.get(TRUE_VARIANT)
            b = variants.get(baseline)
            if not t or not b:
                continue
            if t.get("region_loss") is None or b.get("region_loss") is None:
                continue
            if abs(float(t.get("sigma", 0.0)) - float(b.get("sigma", 0.0))) > 1e-6:
                sigma_mismatch += 1
                continue
            pairs.append(
                {
                    "clip_id": key[0],
                    "sigma": float(t["sigma"]),
                    "region_pixels": int(t.get("region_pixels") or 0),
                    "diff": float(b["region_loss"]) - float(t["region_loss"]),
                    "base_region_loss": float(b["region_loss"]),
                    "true_region_loss": float(t["region_loss"]),
                    "whole_diff": float(b.get("whole_frame_loss") or b["loss"]) - float(t.get("whole_frame_loss") or t["loss"]),
                }
            )
        comp: dict = {"overall": summarize_pairs(pairs), "sigma_mismatch_pairs_skipped": sigma_mismatch}

        if pairs:
            sigmas = sorted(p["sigma"] for p in pairs)
            qn = len(sigmas)
            edges = [sigmas[qn // 4], sigmas[qn // 2], sigmas[(3 * qn) // 4]]
            sigma_buckets: dict[str, list[dict]] = defaultdict(list)
            for p in pairs:
                sigma_buckets[bucket_label_sigma(p["sigma"], edges)].append(p)
            comp["by_sigma_quartile"] = {k: summarize_pairs(v) for k, v in sorted(sigma_buckets.items())}

            px_buckets: dict[str, list[dict]] = defaultdict(list)
            for p in pairs:
                px_buckets[bucket_label_pixels(p["region_pixels"])].append(p)
            comp["by_region_pixels"] = {k: summarize_pairs(v) for k, v in sorted(px_buckets.items())}

            clip_buckets: dict[str, list[dict]] = defaultdict(list)
            for p in pairs:
                clip_buckets[p["clip_id"]].append(p)
            clip_stats = []
            for cid, ps in clip_buckets.items():
                s = summarize_pairs(ps)
                s["clip_id"] = cid
                clip_stats.append(s)
            clip_stats.sort(key=lambda s: s["mean_paired_diff"], reverse=True)
            comp["clip_count"] = len(clip_stats)
            comp["clips_mean_diff_positive"] = sum(1 for s in clip_stats if s["mean_paired_diff"] > 0)
            comp["top_clips_true_better"] = clip_stats[:top_clips]
            comp["top_clips_true_worse"] = clip_stats[-top_clips:][::-1]

            comp["whole_frame_overall_mean_diff"] = mean([p["whole_diff"] for p in pairs])

        result["comparisons"][baseline] = comp

    return result


def print_summary(result: dict) -> None:
    print(f"== {result['eval_json']}")
    for baseline, comp in result["comparisons"].items():
        o = comp.get("overall") or {}
        if not o.get("n"):
            print(f"  {baseline}: no pairs")
            continue
        print(
            f"  {baseline}: n={o['n']} mean_diff={o['mean_paired_diff']:.5f} "
            f"frac_true_better={o['frac_true_better']:.3f} rel_reduction={o['relative_reduction']:.4f}"
        )
        for section in ["by_sigma_quartile", "by_region_pixels"]:
            for k, s in (comp.get(section) or {}).items():
                if not s.get("n"):
                    continue
                print(
                    f"    {k}: n={s['n']} mean_diff={s['mean_paired_diff']:.5f} "
                    f"frac={s['frac_true_better']:.3f} rel={s['relative_reduction']:.4f}"
                )
        if "clips_mean_diff_positive" in comp:
            print(f"    clips true-better: {comp['clips_mean_diff_positive']}/{comp['clip_count']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", type=Path, nargs="+", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--top-clips", type=int, default=8)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for eval_path in args.eval_json:
        result = analyze(eval_path, args.top_clips)
        out = args.output_dir / f"{eval_path.parent.name}__{eval_path.stem}_rows_analysis_v0.json"
        out.write_text(json.dumps(result, indent=1))
        print_summary(result)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
