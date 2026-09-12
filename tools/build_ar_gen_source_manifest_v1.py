#!/usr/bin/env python3
"""Canonical non-overlapping (stride == window_len) AR gen-source manifest builder.

Background
----------
The v0 AR source manifest
`output/demo_1min_20260706_v0/manifests/candidate1_true_dense_gen_source_manifest_v0.jsonl`
is a *sliding* window layout: consecutive windows advance by raw_start stride 80, yet each
window spans 160 raw frames (21 positive latent frames at raw stride 8 -> 8*20 = 160). That is
exactly 50% overlap. When `qxq_sample_candidate1_ar_v0.py` chains window N's generated tail
frame into window N+1 as the I2V image condition, the image condition leads window N+1's
dense/pose start by 160-80 = 80 raw frames (2.5 s @ 32fps raw -> the well-known lead bug).

A correct autoregressive chain needs the next window to begin exactly where the previous one
ended: next.raw_start == prev.raw_start + window_len_raw (stride == window_len), so the previous
tail frame is genuinely the next window's first conditioned frame.

Reverse-engineering basis (no original non-overlapping builder existed; v0's
`build_candidate1_gen_manifest_v0.py` only *decorates* pre-windowed rows from an upstream
sliding source and does not choose the stride). Row schema fields used here, confirmed by
inspecting the real manifest:
  - raw_start                     : first raw frame of the window (int)
  - raw_indices / positive_latent_frames : list of conditioned raw frames, stride 8, len 21
  - frame_count_start / frame_count_end  : raw span endpoints (end - start == window span)
  - positive_latent_frame_count   : 21 (=> window_len_raw = 8 * (n - 1))
  - frame_contract.raw_frame      : "raw_start + 8 * latent_index" (canonical raw stride 8)
window_len_raw is derived (priority order): raw_indices span, then frame_count_end-start, then
8*(positive_latent_frame_count-1).

Modes
-----
build (default): read an input manifest of already-windowed+decorated rows and emit the
  non-overlapping subset (stride == window_len). Rows pass through UNCHANGED, so the schema
  stays v0-compatible and the downstream sampler needs no edits (each kept row already carries
  its dense_sequence_manifest). Self-check: re-read the output and assert every adjacent
  raw_start diff == window_len, else exit non-zero.
--check-only <manifest.jsonl>: validate an existing manifest against the non-overlap /
  contiguity contract. Prints stride + window_len + verdict per adjacent pair; exits non-zero
  if any overlap (or gap) is found. Running it on the v0 manifest is the falsification test:
  it must report the 50% overlap.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def raw_span(row: dict[str, Any]) -> tuple[int, int]:
    """Return (raw_start, window_len_raw) for one window row.

    window_len_raw is the raw-frame distance from the first to the last conditioned frame.
    """
    ri = row.get("raw_indices") or row.get("positive_latent_frames")
    if isinstance(ri, list) and len(ri) >= 2:
        return int(ri[0]), int(ri[-1]) - int(ri[0])
    raw_start = int(row["raw_start"])
    if row.get("frame_count_start") is not None and row.get("frame_count_end") is not None:
        return raw_start, int(row["frame_count_end"]) - int(row["frame_count_start"])
    n_lat = int(row.get("positive_latent_frame_count") or row.get("latent_frames"))
    return raw_start, 8 * (n_lat - 1)


def uniform_window_len(rows: list[dict[str, Any]]) -> int:
    lens = {raw_span(r)[1] for r in rows}
    if len(lens) != 1:
        raise SystemExit(
            f"[build] windows do not share a single window_len_raw: {sorted(lens)}; "
            "cannot build a canonical stride==window_len tiling."
        )
    return next(iter(lens))


def check_only(manifest: Path) -> int:
    rows = read_jsonl(manifest)
    rows.sort(key=lambda r: int(r["raw_start"]) if "raw_start" in r else raw_span(r)[0])
    print(json.dumps({"mode": "check-only", "manifest": str(manifest), "rows": len(rows)}, ensure_ascii=False))
    if len(rows) < 2:
        print(json.dumps({"verdict": "ok", "reason": "fewer than 2 windows; nothing to overlap"}, ensure_ascii=False))
        return 0
    n_overlap = 0
    n_gap = 0
    for i in range(1, len(rows)):
        prev_start, window_len = raw_span(rows[i - 1])
        cur_start, _ = raw_span(rows[i])
        stride = cur_start - prev_start
        if stride < window_len:
            verdict = "OVERLAP"
            n_overlap += 1
        elif stride > window_len:
            verdict = "GAP"
            n_gap += 1
        else:
            verdict = "ok"
        print(json.dumps({
            "pair": [i - 1, i],
            "prev_raw_start": prev_start,
            "cur_raw_start": cur_start,
            "stride": stride,
            "window_len_raw": window_len,
            "overlap_raw": max(0, window_len - stride),
            "verdict": verdict,
        }, ensure_ascii=False))
    summary = {
        "summary": True,
        "pairs": len(rows) - 1,
        "overlap_pairs": n_overlap,
        "gap_pairs": n_gap,
        "contract_satisfied": (n_overlap == 0 and n_gap == 0),
        "note": "non-overlap AR contract requires stride == window_len for every adjacent pair.",
    }
    print(json.dumps(summary, ensure_ascii=False))
    if n_overlap > 0:
        return 2  # falsification signal: sliding/overlapping layout
    if n_gap > 0:
        return 3  # non-overlapping but non-contiguous (temporal gap between windows)
    return 0


def build(input_manifest: Path, output_manifest: Path, start_raw: int | None) -> int:
    rows = read_jsonl(input_manifest)
    if not rows:
        raise SystemExit(f"[build] empty input manifest: {input_manifest}")
    rows.sort(key=lambda r: int(r["raw_start"]) if "raw_start" in r else raw_span(r)[0])
    window_len = uniform_window_len(rows)
    by_start: dict[int, dict[str, Any]] = {}
    for r in rows:
        s = raw_span(r)[0]
        by_start.setdefault(s, r)  # first wins on duplicate raw_start
    next_start = int(start_raw) if start_raw is not None else raw_span(rows[0])[0]
    kept: list[dict[str, Any]] = []
    while next_start in by_start:
        kept.append(by_start[next_start])
        next_start += window_len
    if not kept:
        raise SystemExit(
            f"[build] no window found at start_raw={next_start}; available starts head="
            f"{sorted(by_start)[:5]}"
        )
    write_jsonl(output_manifest, kept)
    # Self-check: re-read and assert exact non-overlapping tiling.
    reread = read_jsonl(output_manifest)
    reread.sort(key=lambda r: int(r["raw_start"]) if "raw_start" in r else raw_span(r)[0])
    for i in range(1, len(reread)):
        prev_start, wl = raw_span(reread[i - 1])
        cur_start, _ = raw_span(reread[i])
        if cur_start - prev_start != wl:
            raise SystemExit(
                f"[build] self-check FAILED at pair ({i - 1},{i}): stride "
                f"{cur_start - prev_start} != window_len {wl}"
            )
    report = {
        "mode": "build",
        "input_manifest": str(input_manifest),
        "output_manifest": str(output_manifest),
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "window_len_raw": window_len,
        "first_raw_start": raw_span(kept[0])[0],
        "last_raw_start": raw_span(kept[-1])[0],
        "self_check": "passed: every adjacent raw_start diff == window_len",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-only", type=Path, default=None,
                    help="validate an existing manifest against the non-overlap contract; non-zero exit on overlap/gap.")
    ap.add_argument("--input", type=Path, default=None,
                    help="build mode: input manifest of already-windowed+decorated rows.")
    ap.add_argument("--output", type=Path, default=None,
                    help="build mode: output non-overlapping manifest path.")
    ap.add_argument("--start-raw", type=int, default=None,
                    help="build mode: raw_start of the first kept window (default: smallest raw_start in input).")
    args = ap.parse_args()

    if args.check_only is not None:
        sys.exit(check_only(args.check_only))

    if args.input is None or args.output is None:
        ap.error("build mode requires both --input and --output (or use --check-only)")
    sys.exit(build(args.input, args.output, args.start_raw))


if __name__ == "__main__":
    main()
