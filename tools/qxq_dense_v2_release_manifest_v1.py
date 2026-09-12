#!/usr/bin/env python3
"""Merge per-shard dense v2 render outputs into one release manifest.

run_dense_v2_gpu_v1.py writes ``{"samples": [...]}`` per shard and leaves two
fields out that ``check_map_memory_training_readiness_v0.check_manifest_contract``
requires: a top-level ``sample_count`` and a per-sample ``channels`` list. This
tool closes that gap after the fact, so the running renderer is never touched.

The channel contract is taken from the rendered sample metadata (what actually
went to disk) and asserted equal to the readiness checker's CHANNELS. They differ
from ``run_dense_v2_gpu_v1.CHANNELS``, whose ch0 is named
``env_depth_norm_from_bsp_faces`` while the meta/checker/loader all say
``env_depth_norm_from_world_obj_projection`` -- a stale name kept because it is a
contract identifier, not a description. Populating the manifest from the driver
constant instead would fail the checker with "channel names/order mismatch".

Usage:
  python3 tools/qxq_dense_v2_release_manifest_v1.py \
    --run-dir output/qxq_dense_v2_new120h_20260723 \
    --pool-name new120h \
    --out output/qxq_dense_v2_new120h_20260723/release_manifest_v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_DENSE_HW = (240, 416)
DEFAULT_BACKEND = "bsp_faces_disp_gpu"


def load_checker_channels() -> list[str] | None:
    """CHANNELS from the readiness checker, or None if it cannot be imported."""
    path = TOOLS_DIR / "check_map_memory_training_readiness_v0.py"
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_readiness_checker", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        return list(mod.CHANNELS)
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"[warn] could not import readiness checker CHANNELS: {exc!r}", file=sys.stderr)
        return None


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(base: Path, value: str) -> str:
    p = Path(value)
    return str(p if p.is_absolute() else (base / p).resolve())


def shard_path_base(shard_dir: Path, sample: dict | None) -> Path:
    """Base for relative sample paths.

    The driver stores paths built from ``--out-dir`` verbatim, so they are
    relative to the *cwd it ran under* (recorded in the command json), not to the
    shard dir. Probe the candidates against a known sample path and take the
    first that actually resolves; fall back to the shard dir.
    """
    candidates: list[Path] = []
    for cmd in sorted(shard_dir.glob("command_dense_v2_gpu_v1_resume_*.json")):
        try:
            cwd = json.loads(cmd.read_text(encoding="utf-8")).get("cwd")
        except Exception:
            continue
        if cwd and Path(cwd).is_dir():
            candidates.append(Path(cwd))
    candidates.extend([shard_dir, TOOLS_DIR.parent])
    probe = (sample or {}).get("meta_path") or (sample or {}).get("dense_path")
    if probe and not Path(probe).is_absolute():
        for base in candidates:
            if (base / probe).exists():
                return base
    return candidates[0] if candidates else shard_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None, help="run root; globs gpu_*/ shards")
    ap.add_argument("--shard-dir", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pool-name", required=True)
    ap.add_argument("--dense-hw", type=int, nargs=2, metavar=("H", "W"), default=list(DEFAULT_DENSE_HW))
    ap.add_argument("--expected-geometry-backend", default=DEFAULT_BACKEND)
    ap.add_argument("--meta-spot-check", type=int, default=32, help="metas to read for channel-contract verification")
    ap.add_argument("--allow-missing-report", action="store_true", help="for partial/in-flight inspection only")
    ap.add_argument(
        "--allow-duplicate-sample-ids", action="store_true",
        help="accept duplicate sample_ids as re-renders of the same (instance, frame): "
        "overlapping event windows render the same frame into more than one clip/shard. "
        "First occurrence wins; the dedup count still lands in the accept json.",
    )
    args = ap.parse_args()

    shards = list(args.shard_dir)
    if args.run_dir is not None:
        shards.extend(sorted(d for d in args.run_dir.glob("gpu_*") if d.is_dir()))
    shards = [d.resolve() for d in shards]
    if not shards:
        raise SystemExit("no shard dirs found")

    expected_shape = [7, int(args.dense_hw[0]), int(args.dense_hw[1])]
    failures: list[str] = []
    samples: list[dict] = []
    seen_ids: set[str] = set()
    dup_ids: list[str] = []
    shard_reports: list[dict] = []
    render_shas: set[str] = set()
    bsp_shas: set[str] = set()
    backends: set[str] = set()
    shapes: set[tuple] = set()

    for sd in shards:
        mpath = sd / "manifest.json"
        rpath = sd / "dense_v2_gpu_report_v1.json"
        if not mpath.exists():
            failures.append(f"{sd.name}: manifest.json missing (shard not finished?)")
            continue
        if rpath.exists():
            rep = json.loads(rpath.read_text(encoding="utf-8"))
            shard_reports.append({
                "shard": sd.name, "status": rep.get("status"),
                "sample_count": rep.get("sample_count"), "failure_count": rep.get("failure_count"),
                "shapes": rep.get("shapes"), "backend_ids": rep.get("backend_ids"),
                "render_module_sha256": rep.get("render_module_sha256"),
                "bsp_faces_npz_sha256": rep.get("bsp_faces_npz_sha256"),
                "clips_per_s": rep.get("clips_per_s"),
            })
            if rep.get("status") != "pass":
                failures.append(f"{sd.name}: report status={rep.get('status')!r}, expected 'pass'")
            if rep.get("render_module_sha256"):
                render_shas.add(str(rep["render_module_sha256"]))
            if rep.get("bsp_faces_npz_sha256"):
                bsp_shas.add(str(rep["bsp_faces_npz_sha256"]))
        elif not args.allow_missing_report:
            failures.append(f"{sd.name}: dense_v2_gpu_report_v1.json missing (shard not finished?)")

        man = json.loads(mpath.read_text(encoding="utf-8"))
        shard_samples = man.get("samples", [])
        base = shard_path_base(sd, shard_samples[0] if shard_samples else None)
        for s in shard_samples:
            sid = s.get("sample_id")
            if sid in seen_ids:
                dup_ids.append(str(sid))
                continue
            seen_ids.add(sid)
            shape = s.get("shape")
            if shape is not None:
                shapes.add(tuple(shape))
            bid = s.get("geometry_backend_id")
            if bid:
                backends.add(str(bid))
            rec = dict(s)
            rec["shard"] = sd.name
            for key in ("dense_path", "meta_path", "qa_path", "target_rgb_path"):
                if rec.get(key):
                    rec[key] = resolve(base, rec[key])
            samples.append(rec)

    # --- channel contract: source of truth is what was rendered ---
    meta_channels: set[tuple] = set()
    metas_read = 0
    metas_missing: list[str] = []
    for s in samples[: max(0, args.meta_spot_check)]:
        mp = s.get("meta_path")
        if not mp or not Path(mp).exists():
            metas_missing.append(str(mp))
            continue
        meta = json.loads(Path(mp).read_text(encoding="utf-8"))
        metas_read += 1
        ch = meta.get("channels")
        if ch:
            meta_channels.add(tuple(ch))
    channels: list[str] | None
    if samples and metas_read == 0:
        failures.append(
            f"could not read ANY sample meta ({len(metas_missing)} probed); "
            f"path resolution likely wrong, e.g. {metas_missing[:1]}"
        )
        channels = None
    elif len(meta_channels) > 1:
        failures.append(f"rendered metas disagree on channel list: {sorted(meta_channels)}")
        channels = sorted(meta_channels)[0]
    elif len(meta_channels) == 1:
        channels = list(next(iter(meta_channels)))
    else:
        if samples:
            failures.append("sample metas carry no 'channels' field")
        channels = None

    checker_channels = load_checker_channels()
    if channels is not None and checker_channels is not None and list(channels) != list(checker_channels):
        failures.append(
            "rendered channel list != readiness checker CHANNELS; refusing to emit a "
            f"manifest that cannot pass the contract.\n  rendered={list(channels)}\n  checker ={list(checker_channels)}"
        )

    if channels is not None:
        for s in samples:
            s["channels"] = list(channels)

    if dup_ids and not args.allow_duplicate_sample_ids:
        failures.append(f"{len(dup_ids)} duplicate sample_id across shards, e.g. {dup_ids[:3]}")
    bad_shapes = [list(x) for x in shapes if list(x) != expected_shape]
    if bad_shapes:
        failures.append(f"shape(s) != {expected_shape}: {bad_shapes}")
    bad_backends = sorted(b for b in backends if b != args.expected_geometry_backend)
    if bad_backends:
        failures.append(f"geometry_backend_id(s) != {args.expected_geometry_backend!r}: {bad_backends}")
    if len(render_shas) > 1:
        failures.append(f"shards disagree on render_module_sha256: {sorted(render_shas)}")
    if len(bsp_shas) > 1:
        failures.append(f"shards disagree on bsp_faces_npz_sha256: {sorted(bsp_shas)}")

    manifest = {
        "kind": "qxq_dense_v2_release_manifest_v1",
        "pool": args.pool_name,
        "sample_count": len(samples),          # <- required by check_manifest_contract
        "channels": list(channels) if channels else None,
        "dense_hw": [int(args.dense_hw[0]), int(args.dense_hw[1])],
        "shape": expected_shape,
        "geometry_backend_id": args.expected_geometry_backend,
        "shard_count": len(shards),
        "render_module_sha256": sorted(render_shas)[0] if len(render_shas) == 1 else sorted(render_shas),
        "bsp_faces_npz_sha256": sorted(bsp_shas)[0] if len(bsp_shas) == 1 else sorted(bsp_shas),
        "shard_reports": shard_reports,
        "samples": samples,                    # each now carries "channels"
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    accept = {
        "kind": "qxq_dense_v2_release_manifest_accept_v1",
        "pool": args.pool_name,
        "manifest": str(args.out.resolve()),
        "manifest_sha256": sha256_file(args.out),
        "status": "pass" if not failures else "fail",
        "sample_count": len(samples),
        "shard_count": len(shards),
        "shards_with_report": len(shard_reports),
        "distinct_shapes": sorted([list(x) for x in shapes], key=str),
        "distinct_backend_ids": sorted(backends),
        "render_module_sha256": sorted(render_shas),
        "bsp_faces_npz_sha256": sorted(bsp_shas),
        "channels": list(channels) if channels else None,
        "channels_match_readiness_checker": (
            None if checker_channels is None else (list(channels or []) == list(checker_channels))
        ),
        "duplicate_sample_ids": len(dup_ids),
        "failures": failures,
    }
    acc_path = args.out.with_suffix(".accept.json")
    acc_path.write_text(json.dumps(accept, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(accept, ensure_ascii=False, indent=1), flush=True)
    raise SystemExit(0 if not failures else 2)


if __name__ == "__main__":
    main()
