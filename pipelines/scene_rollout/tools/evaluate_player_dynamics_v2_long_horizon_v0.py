#!/usr/bin/env python3
"""Long-horizon autoregressive eval for player dynamics v2 checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import train_player_dynamics_v2 as v2  # noqa: E402


class LongWindowDataset:
    def __init__(self, manifest: dict[str, Any], split: str, horizon: int, limit_windows: int | None = None) -> None:
        self.manifest = manifest
        self.history = int(manifest["history"])
        self.stride = int(manifest["stride"])
        self.horizon = int(horizon)
        self.refs: list[tuple[str, int]] = []
        self._cache_path: str | None = None
        self._cache: Any = None
        for rec in manifest["episode_records"]:
            if rec["split"] != split:
                continue
            n = int(rec["frames"])
            windows = max(0, (n - self.history - self.horizon) // self.stride + 1)
            for i in range(windows):
                self.refs.append((rec["path"], i * self.stride))
        if limit_windows is not None:
            self.refs = self.refs[:limit_windows]

    def __len__(self) -> int:
        return len(self.refs)

    def _load(self, path: str) -> Any:
        if self._cache_path != path:
            self._cache = np.load(path, allow_pickle=True)
            self._cache_path = path
        return self._cache

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        path, s = self.refs[idx]
        z = self._load(path)
        h = self.history
        k = self.horizon
        return {
            "pose_h": z["pose"][s : s + h],
            "act_h": z["actions"][s : s + h],
            "y": z["target"][s + h : s + h + k],
            "act_y": z["actions"][s + h : s + h + k],
            "last_y": z["target"][s + h - 1],
            "prev_y": z["target"][s + h - 2],
        }


def target_to_pose_features(target: Any, health: Any) -> Any:
    import torch

    yaw = torch.deg2rad(torch.remainder(target[..., 3], 360.0))
    pitch = torch.deg2rad(torch.remainder(target[..., 4], 360.0))
    return torch.cat(
        [
            target[..., :3],
            torch.sin(yaw)[..., None],
            torch.cos(yaw)[..., None],
            torch.sin(pitch)[..., None],
            torch.cos(pitch)[..., None],
            health,
        ],
        dim=-1,
    )


def rollout_model(model: Any, pose_h: Any, act_h: Any, norm: dict[str, Any], horizon: int, chunk: int) -> tuple[Any, Any]:
    import torch

    pose_hist = pose_h.clone()
    act_hist = act_h.clone()
    pred_chunks = []
    logit_chunks = []
    while sum(x.shape[1] for x in pred_chunks) < horizon:
        pose_norm, act_norm = v2.normalize_inputs(pose_hist, act_hist, norm)
        residual_norm, logits = model(pose_norm, act_norm)
        last_y = v2.y_targets_from_pose(pose_hist[:, -1].detach().cpu().numpy())
        prev_y = v2.y_targets_from_pose(pose_hist[:, -2].detach().cpu().numpy())
        last_y_t = torch.from_numpy(last_y).to(pose_h.device, dtype=pose_h.dtype)
        prev_y_t = torch.from_numpy(prev_y).to(pose_h.device, dtype=pose_h.dtype)
        pred_abs = v2.decode_abs_predictions(residual_norm, last_y_t, prev_y_t, norm)
        take = min(chunk, horizon - sum(x.shape[1] for x in pred_chunks))
        pred_take = pred_abs[:, :take]
        logits_take = logits[:, :take]
        pred_chunks.append(pred_take)
        logit_chunks.append(logits_take)
        health = pose_hist[:, -1:, :, 7:8].repeat(1, take, 1, 1)
        next_pose = target_to_pose_features(pred_take, health)
        next_act = (logits_take >= 0.0).to(act_h.dtype)
        pose_hist = torch.cat([pose_hist[:, take:], next_pose], dim=1)
        act_hist = torch.cat([act_hist[:, take:], next_act], dim=1)
    return torch.cat(pred_chunks, dim=1), torch.cat(logit_chunks, dim=1)


def add_metric_sums(bucket: dict[str, dict[str, float]], pred: np.ndarray, y: np.ndarray, logits: np.ndarray | None, act_y: np.ndarray, horizons: list[int]) -> None:
    rows = v2.metric_accum(pred, y, logits, act_y, horizons)
    for h, rec in rows.items():
        dst = bucket.setdefault(h, {})
        for k, val in rec.items():
            dst[k] = dst.get(k, 0.0) + float(val)


def reduce_bucket(bucket: dict[str, dict[str, float]], horizons: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for h in horizons:
        key = str(h)
        rec = bucket.get(key, {})
        pose_count = int(rec.get("pose_count", 0))
        item: dict[str, Any] = {"pose_count": pose_count}
        if pose_count:
            item["pos3d_mean"] = float(rec.get("pos3d_sum", 0.0) / pose_count)
            item["xy_mean"] = float(rec.get("xy_sum", 0.0) / pose_count)
            item["yaw_deg_mean"] = float(rec.get("yaw_deg_sum", 0.0) / pose_count)
            item["pitch_deg_mean"] = float(rec.get("pitch_deg_sum", 0.0) / pose_count)
        action_count = int(rec.get("action_count", 0))
        if action_count:
            item["action_count"] = action_count
            item["action_bit_acc"] = float(rec.get("action_correct", 0.0) / action_count)
        out[key] = item
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--horizons", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit-windows", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    import __main__
    import torch
    from torch.utils.data import DataLoader

    __main__.train_cmd = v2.train_cmd
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    manifest = ckpt["manifest"]
    norm = v2.normalization_from_ckpt(ckpt)
    model_future = int(manifest["future"])
    horizons = sorted(set(int(h) for h in args.horizons if int(h) > 0))
    max_horizon = max(horizons)
    if max_horizon % model_future != 0:
        raise ValueError(f"max horizon {max_horizon} should be a multiple of checkpoint future {model_future}")

    ds = LongWindowDataset(manifest, args.split, max_horizon, args.limit_windows)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=v2.collate)
    device = v2.resolve_device(args.cpu)
    Model = v2.get_torch_model_class()
    model = Model(
        v2.POSE_DIM,
        v2.ACTION_DIM,
        model_future,
        d_model=ckpt["args"]["d_model"],
        layers=ckpt["args"]["layers"],
        heads=ckpt["args"]["heads"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    buckets: dict[str, dict[str, dict[str, float]]] = {"model": {}, "static": {}, "constant_velocity": {}}
    with torch.no_grad():
        for batch in loader:
            pose_h = batch["pose_h"].to(device)
            act_h = batch["act_h"].to(device)
            pred, logits = rollout_model(model, pose_h, act_h, norm, max_horizon, model_future)
            static, const = v2.baseline_preds(batch, max_horizon)
            y = batch["y"].numpy()
            act_y = batch["act_y"].numpy()
            add_metric_sums(buckets["model"], pred.cpu().numpy(), y, logits.cpu().numpy(), act_y, horizons)
            add_metric_sums(buckets["static"], static.numpy(), y, None, act_y, horizons)
            add_metric_sums(buckets["constant_velocity"], const.numpy(), y, None, act_y, horizons)

    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "windows_evaluated": len(ds),
        "horizons_frames": horizons,
        "horizons_seconds": [h / 32.0 for h in horizons],
        "rollout": {
            "mode": "autoregressive_chunks",
            "checkpoint_future": model_future,
            "history": int(manifest["history"]),
            "stride": int(manifest["stride"]),
            "action_feedback": "thresholded_model_logits",
            "health_feedback": "repeat_last_observed_health",
        },
        "metrics": {name: reduce_bucket(bucket, horizons) for name, bucket in buckets.items()},
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    v2.write_json(args.out_json, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
