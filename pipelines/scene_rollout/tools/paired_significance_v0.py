#!/usr/bin/env python3
"""配对显著性复判：判断 region_loss(true) 是否显著优于 baseline（shuffled 等）。

取代旧的 "0.5 × 跨帧 loss_std" 判据。核心：true / shuffled / blank / player_shuffle 是在
**同帧、同 noise/sigma** 下评估的，必须按 item 配对，差值里抵消帧难度方差。

用法：喂一个 ablation rows JSON（每行含 variant + 配对键 + region_loss），
对 (true vs 每个 baseline) 输出配对统计与"统一判据"判定。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

# 通过判据（与 training_codex.md "统一显著性判据" 一致）
CI_LEVEL = 0.95
PASS_FRAC = 0.65          # true 更好的帧占比下限
PASS_REL_REDUCTION = 0.10 # 相对降幅下限（实际意义）
N_BOOT = 10000


def item_key(row: dict) -> tuple:
    """同一帧/同一 noise 的配对键。ablation 对每个 item 固定 noise_seed+idx 与 sigma，
    所以 (record_index, chunk_ord) 足以配对；带上 sample_ids 更稳。"""
    return (row.get("record_index"), row.get("chunk_ord"), tuple(row.get("sample_ids", [])))


def region_loss(row: dict) -> float:
    # 用区域 loss；若该字段不存在，回退到 loss 并在报告里标注（应优先 region_loss）
    v = row.get("region_loss", row.get("loss"))
    return float(v)


def paired_series(rows: list[dict], variant_a: str, variant_b: str) -> tuple[np.ndarray, np.ndarray]:
    by_key_a, by_key_b = {}, {}
    for r in rows:
        if r.get("variant") == variant_a:
            by_key_a[item_key(r)] = region_loss(r)
        elif r.get("variant") == variant_b:
            by_key_b[item_key(r)] = region_loss(r)
    keys = [k for k in by_key_a if k in by_key_b]
    a = np.array([by_key_a[k] for k in keys], dtype=np.float64)  # true
    b = np.array([by_key_b[k] for k in keys], dtype=np.float64)  # baseline
    return a, b


def judge(true_loss: np.ndarray, base_loss: np.ndarray, rng: np.random.Generator) -> dict:
    d = base_loss - true_loss              # >0 表示 true 更好
    n = int(d.size)
    if n == 0:
        return {"n": 0, "error": "no paired items"}
    mean_d = float(d.mean())
    sem = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    # bootstrap CI of mean(d)
    boot = rng.choice(d, size=(N_BOOT, n), replace=True).mean(axis=1)
    ci_low, ci_high = (float(x) for x in np.percentile(boot, [(1 - CI_LEVEL) / 2 * 100, (1 + CI_LEVEL) / 2 * 100]))
    # 符号检验
    frac = float((d > 0).mean())
    # 正态近似 binomial 双侧 p（dependency-light）
    z = (frac - 0.5) / np.sqrt(0.25 / n)
    p_sign = float(2 * (1 - 0.5 * (1 + _erf(abs(z) / np.sqrt(2)))))
    # 效果量
    rel_reduction = float((base_loss.mean() - true_loss.mean()) / base_loss.mean())
    cohen_d = float(mean_d / d.std(ddof=1)) if n > 1 and d.std(ddof=1) > 0 else float("nan")
    passed = (ci_low > 0) and (frac >= PASS_FRAC) and (rel_reduction >= PASS_REL_REDUCTION)
    return {
        "n": n, "mean_true": float(true_loss.mean()), "mean_base": float(base_loss.mean()),
        "mean_paired_diff": mean_d, "sem": sem, "ci95": [ci_low, ci_high],
        "frac_true_better": frac, "sign_p_approx": p_sign,
        "relative_reduction": rel_reduction, "cohen_d": cohen_d,
        "criteria": {"ci_low>0": ci_low > 0, "frac>=0.65": frac >= PASS_FRAC,
                     "rel_reduction>=0.10": rel_reduction >= PASS_REL_REDUCTION},
        "status": "pass" if passed else "fail",
    }


def _erf(x: float) -> float:
    # Abramowitz-Stegun 7.1.26，避免 scipy 依赖
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return float(y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-json", type=Path, required=True, help="ablation 输出，含逐行 variant + region_loss")
    ap.add_argument("--true-variant", default="true_dense")
    ap.add_argument("--baselines", nargs="+", default=["shuffled_dense", "player_shuffle", "blank_dense"])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    obj = json.loads(args.rows_json.read_text())
    rows = obj.get("rows", obj if isinstance(obj, list) else [])
    rng = np.random.default_rng(0)
    report = {"true_variant": args.true_variant, "comparisons": {}}
    for base in args.baselines:
        a, b = paired_series(rows, args.true_variant, base)
        report["comparisons"][base] = judge(a, b, rng)
    # 主判据用 true vs shuffled
    report["primary"] = report["comparisons"].get("shuffled_dense")
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
