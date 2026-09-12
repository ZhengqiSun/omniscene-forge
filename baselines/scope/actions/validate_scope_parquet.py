from __future__ import annotations

import argparse
import json

import numpy as np

from baselines.scope.actions.convert_csgo_to_scope import BUTTON_COLS


def validate(path: str, expected_frames: int) -> dict:
    try: import pandas as pd
    except ImportError as exc: raise RuntimeError("pandas and pyarrow are required") from exc
    df = pd.read_parquet(path); errors = []
    if list(df.columns) != BUTTON_COLS + ["j_left", "j_right"]: errors.append(f"columns={list(df.columns)}")
    if len(df) != expected_frames: errors.append(f"rows={len(df)}, expected={expected_frames}")
    if all(name in df for name in BUTTON_COLS):
        key = df[BUTTON_COLS].to_numpy();
        if not np.isin(key, [0, 1]).all(): errors.append("binary columns contain values outside {0,1}")
    for name in ("j_left", "j_right"):
        if name in df:
            try: arr = np.asarray(df[name].tolist(), dtype=np.float32)
            except Exception: errors.append(f"{name} cannot convert to float32"); continue
            if arr.shape != (len(df), 2): errors.append(f"{name} shape={arr.shape}")
            elif not np.isfinite(arr).all() or (np.abs(arr) > 1).any(): errors.append(f"{name} must be finite in [-1,1]")
    return {"path": path, "rows": len(df), "errors": errors, "valid": not errors}


def main():
    p=argparse.ArgumentParser(); p.add_argument("path"); p.add_argument("--expected-frames",type=int,choices=(81,101),required=True); a=p.parse_args()
    report=validate(a.path,a.expected_frames); print(json.dumps(report,indent=2)); raise SystemExit(0 if report["valid"] else 2)


if __name__ == "__main__": main()
