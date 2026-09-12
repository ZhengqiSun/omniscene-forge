from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from baselines.scope.actions.convert_csgo_to_scope import parse_csgo_rows
from baselines.scope.actions.resample_actions import resample
from baselines.scope.scope_bridge.schema import load_manifest, manifest_sha256, resolve_path


def assert_train_only(samples) -> None:
    if any(s.split != "train" for s in samples):
        raise ValueError("calibration manifest must contain train split only")


def main() -> None:
    p = argparse.ArgumentParser(description="Fit one global mouse gain using train rows only.")
    p.add_argument("--manifest", required=True); p.add_argument("--output", required=True)
    p.add_argument("--quantile", type=float, default=0.995); p.add_argument("--max-samples", type=int)
    args = p.parse_args()
    samples = load_manifest(args.manifest)
    assert_train_only(samples)
    values = []
    for sample in samples[:args.max_samples]:
        if not sample.raw_action_path:
            continue
        n = int(np.ceil(sample.num_frames / sample.model_fps * sample.source_fps))
        _, mouse, buttons = parse_csgo_rows(resolve_path(args.manifest, sample.raw_action_path), int(sample.raw_start or 0), n, (1, 1, 1, 1))
        actions = resample(np.zeros((n, 2)), mouse, buttons, sample.source_fps, sample.model_fps, sample.num_frames, None)
        values.append(np.abs(actions.sticks[:, 2:]).reshape(-1))
    if not values:
        raise ValueError("no raw_action_path rows available")
    flat = np.concatenate(values); gain = float(np.quantile(flat, args.quantile))
    if not np.isfinite(gain) or gain <= 0: raise ValueError("invalid fitted gain")
    normalized = np.clip(flat / gain, 0, 1)
    report = {
        "schema_id": "ScopeMouseCalibrationV1", "fit_split": "train",
        "train_manifest": str(Path(args.manifest).resolve()), "train_manifest_sha256": manifest_sha256(args.manifest),
        "quantile": args.quantile, "gain": gain, "sample_count": len(values), "value_count": len(flat),
        "clip_rate": float((flat / gain >= 1).mean()), "mean": float(normalized.mean()), "std": float(normalized.std()),
        "percentiles": {str(q): float(np.quantile(normalized, q)) for q in (0.5, 0.9, 0.95, 0.99, 0.999)},
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
