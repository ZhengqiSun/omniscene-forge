#!/usr/bin/env python3
"""Player dynamics v0: data build, small transformer training, eval, and plots."""

from __future__ import annotations
from runtime_paths import source_path

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


BOOL_ACTIONS = [
    "forward",
    "back",
    "left",
    "right",
    "jump",
    "crouch",
    "walk",
    "fire",
    "reload",
    "use",
    "scope",
    "inspect",
    "plant",
    "defuse",
    "plant_success",
    "defuse_success",
    "has_helmet",
    "has_defuse_kit",
    "is_on_ladder",
]
ACTION_NAMES = BOOL_ACTIONS + ["scope_level_positive"]
POSE_DIM = 8
ACTION_DIM = len(ACTION_NAMES)
PLAYER_RE = re.compile(r"team_(\d+)_player_(\d+)_inst_000\.json$")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def player_jsons(ep_dir: Path) -> list[Path]:
    paths = []
    for p in ep_dir.glob("*_team_*_player_*_inst_000.json"):
        name = p.name
        if any(s in name for s in ["visibility", "manifest", "episode_info", "colormap"]):
            continue
        if PLAYER_RE.search(name):
            paths.append(p)
    return sorted(paths, key=player_sort_key)


def player_sort_key(path: Path) -> tuple[int, int, str]:
    m = PLAYER_RE.search(path.name)
    if not m:
        return (99, 9999, path.name)
    return (int(m.group(1)), int(m.group(2)), path.name)


def angle_to_sincos(deg: float) -> tuple[float, float]:
    rad = math.radians(float(deg) % 360.0)
    return math.sin(rad), math.cos(rad)


def angle_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def action_vector(action: dict[str, Any]) -> np.ndarray:
    vals = [1.0 if bool(action.get(k, False)) else 0.0 for k in BOOL_ACTIONS]
    vals.append(1.0 if float(action.get("scope_level", 0.0)) > 0.0 else 0.0)
    return np.asarray(vals, dtype=np.float32)


def pose_vector(row: dict[str, Any]) -> np.ndarray:
    yaw_s, yaw_c = angle_to_sincos(float(row.get("yaw", 0.0)))
    pitch_s, pitch_c = angle_to_sincos(float(row.get("pitch", 0.0)))
    return np.asarray(
        [
            float(row.get("x", 0.0)),
            float(row.get("y", 0.0)),
            float(row.get("z", 0.0)),
            yaw_s,
            yaw_c,
            pitch_s,
            pitch_c,
            float(row.get("health", 0.0)) / 100.0,
        ],
        dtype=np.float32,
    )


def y_targets_from_pose(pose: np.ndarray) -> np.ndarray:
    yaw = np.degrees(np.arctan2(pose[..., 3], pose[..., 4])) % 360.0
    pitch = np.degrees(np.arctan2(pose[..., 5], pose[..., 6])) % 360.0
    return np.concatenate([pose[..., :3], yaw[..., None], pitch[..., None]], axis=-1).astype(np.float32)


def y_targets_from_json_rows(rows: list[dict[str, Any]]) -> np.ndarray:
    vals = []
    for r in rows:
        vals.append([float(r.get("x", 0.0)), float(r.get("y", 0.0)), float(r.get("z", 0.0)), float(r.get("yaw", 0.0)) % 360.0, float(r.get("pitch", 0.0)) % 360.0])
    return np.asarray(vals, dtype=np.float32)


def resolve_device(cpu: bool) -> Any:
    import torch

    if cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be set to a subset of physical GPUs 2-7, e.g. CUDA_VISIBLE_DEVICES=2")
    ids = [x.strip() for x in visible.split(",") if x.strip()]
    in_dlc = bool(os.environ.get("DLC_JOB_ID") or os.environ.get("PAI_WORKSPACE_ID"))
    bad = [x for x in ids if x in {"0", "1"}]
    if bad and not in_dlc:
        raise RuntimeError(f"Refusing to use physical GPU(s) {bad}; set CUDA_VISIBLE_DEVICES to 2-7 only")
    return torch.device("cuda:0")


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out = {}
    for k, v in vars(args).items():
        out[k] = str(v) if isinstance(v, Path) else v
    return out


def episode_to_arrays(ep_dir: Path) -> dict[str, Any] | None:
    paths = player_jsons(ep_dir)
    if len(paths) != 10:
        return None
    raw = [load_json(p) for p in paths]
    n = min(len(r) for r in raw)
    if n < 64:
        return None
    pose = np.zeros((n, 10, POSE_DIM), dtype=np.float32)
    target = np.zeros((n, 10, 5), dtype=np.float32)
    actions = np.zeros((n, 10, ACTION_DIM), dtype=np.float32)
    ticks = np.zeros((n, 10), dtype=np.int64)
    rel_ms = np.zeros((n, 10), dtype=np.float32)
    for pi, rows in enumerate(raw):
        for t, row in enumerate(rows[:n]):
            pose[t, pi] = pose_vector(row)
            target[t, pi] = y_targets_from_json_rows([row])[0]
            actions[t, pi] = action_vector(row.get("action", {}))
            ticks[t, pi] = int(row.get("tick", -1))
            rel_ms[t, pi] = float(row.get("relativeTimeMs", np.nan))
    return {
        "pose": pose,
        "target": target,
        "actions": actions,
        "ticks": ticks,
        "relative_time_ms": rel_ms,
        "player_stems": [p.name.replace(".json", "") for p in paths],
    }


def inspect_cmd(args: argparse.Namespace) -> None:
    root = args.data_root
    matches = sorted([p for p in root.iterdir() if p.is_dir()])[: args.matches]
    out: dict[str, Any] = {"data_root": str(root), "matches_seen": []}
    for match in matches:
        eps = sorted((match / "train").glob("Ep_*"))[: args.episodes_per_match]
        match_rec = {"match": match.name, "episode_count": len(list((match / "train").glob("Ep_*"))), "episodes": []}
        for ep in eps:
            paths = player_jsons(ep)
            rec: dict[str, Any] = {"episode": ep.name, "player_jsons": len(paths)}
            if paths:
                rows0 = load_json(paths[0])
                rec["sample_player"] = paths[0].name
                rec["sample_len"] = len(rows0)
                rec["sample_fields"] = list(rows0[0].keys()) if rows0 else []
                rec["sample_action_fields"] = list(rows0[0].get("action", {}).keys()) if rows0 else []
                dts = [rows0[i + 1]["relativeTimeMs"] - rows0[i]["relativeTimeMs"] for i in range(min(len(rows0) - 1, 256))]
                rec["dt_ms_median"] = float(np.median(dts)) if dts else None
                lengths, tick_sets, frame0 = [], [], []
                for p in paths:
                    rows = load_json(p)
                    lengths.append(len(rows))
                    tick_sets.append({int(r.get("tick", -1)) for r in rows})
                    frame0.append({"file": p.name, "tick0": rows[0].get("tick"), "frame_count0": rows[0].get("frame_count"), "relativeTimeMs0": rows[0].get("relativeTimeMs")})
                rec["length_minmax"] = [int(min(lengths)), int(max(lengths))]
                rec["common_ticks"] = int(len(set.intersection(*tick_sets))) if tick_sets else 0
                rec["union_ticks"] = int(len(set.union(*tick_sets))) if tick_sets else 0
                rec["frame0_head"] = frame0[:3]
            match_rec["episodes"].append(rec)
        out["matches_seen"].append(match_rec)
    print(json.dumps(out, ensure_ascii=False, indent=2))


def build_cmd(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    out_dir = args.out_dir
    episodes_dir = out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    matches = sorted([p for p in args.data_root.iterdir() if p.is_dir()])
    val_matches = matches[-args.val_matches :]
    train_matches = matches[: -args.val_matches]
    split = {"train": [m.name for m in train_matches], "val": [m.name for m in val_matches]}
    records = []
    action_sum = np.zeros((ACTION_DIM,), dtype=np.float64)
    action_count = 0
    for split_name, match_list in [("train", train_matches), ("val", val_matches)]:
        for match in match_list:
            for ep in sorted((match / "train").glob("Ep_*")):
                arr = episode_to_arrays(ep)
                if arr is None:
                    continue
                n = int(arr["pose"].shape[0])
                windows = max(0, (n - args.history - args.future) // args.stride + 1)
                if windows <= 0:
                    continue
                rel = Path(split_name) / match.name / f"{ep.name}.npz"
                save_path = episodes_dir / rel
                save_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    save_path,
                    pose=arr["pose"],
                    target=arr["target"],
                    actions=arr["actions"],
                    ticks=arr["ticks"],
                    relative_time_ms=arr["relative_time_ms"],
                    player_stems=np.asarray(arr["player_stems"]),
                )
                action_sum += arr["actions"].reshape(-1, ACTION_DIM).sum(axis=0)
                action_count += int(arr["actions"].shape[0] * arr["actions"].shape[1])
                records.append(
                    {
                        "split": split_name,
                        "match": match.name,
                        "episode": ep.name,
                        "path": str(save_path),
                        "frames": n,
                        "windows": windows,
                        "players": 10,
                    }
                )
    train_windows = sum(r["windows"] for r in records if r["split"] == "train")
    val_windows = sum(r["windows"] for r in records if r["split"] == "val")
    manifest = {
        "kind": "player_dynamics_dataset_v0",
        "source_data_root": str(args.data_root),
        "out_dir": str(out_dir),
        "created_time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "history": args.history,
        "future": args.future,
        "stride": args.stride,
        "fps": 32.0,
        "alignment": "episode-local player JSON array index / frame_count; tick differs by 1-2 across ego files",
        "players_per_episode": 10,
        "pose_features": ["x", "y", "z", "sin_yaw", "cos_yaw", "sin_pitch", "cos_pitch", "health_0_1"],
        "target_features": ["x", "y", "z", "yaw_deg", "pitch_deg"],
        "action_features": ACTION_NAMES,
        "split": split,
        "episode_records": records,
        "summary": {
            "match_count": len(matches),
            "train_matches": len(train_matches),
            "val_matches": len(val_matches),
            "episodes": len(records),
            "train_windows": int(train_windows),
            "val_windows": int(val_windows),
            "total_windows": int(train_windows + val_windows),
            "action_positive_rate": (action_sum / max(action_count, 1)).astype(float).tolist(),
        },
    }
    write_json(out_dir / "manifest_player_dynamics_v0.json", manifest)
    write_json(out_dir / "split_player_dynamics_v0.json", split)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


@dataclass
class SampleRef:
    path: str
    start: int


class WindowDataset:
    def __init__(self, manifest: dict[str, Any], split: str, limit_windows: int | None = None) -> None:
        self.manifest = manifest
        self.history = int(manifest["history"])
        self.future = int(manifest["future"])
        self.stride = int(manifest["stride"])
        self.refs: list[SampleRef] = []
        for rec in manifest["episode_records"]:
            if rec["split"] != split:
                continue
            for i in range(int(rec["windows"])):
                self.refs.append(SampleRef(rec["path"], i * self.stride))
        if limit_windows:
            self.refs = self.refs[:limit_windows]
        self._cache_path = None
        self._cache = None

    def __len__(self) -> int:
        return len(self.refs)

    def _load(self, path: str) -> Any:
        if self._cache_path != path:
            self._cache = np.load(path, allow_pickle=True)
            self._cache_path = path
        return self._cache

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        ref = self.refs[idx]
        z = self._load(ref.path)
        s = ref.start
        h = self.history
        k = self.future
        pose_h = z["pose"][s : s + h]
        act_h = z["actions"][s : s + h]
        y = z["target"][s + h : s + h + k]
        act_y = z["actions"][s + h : s + h + k]
        last_y = z["target"][s + h - 1]
        prev_y = z["target"][s + h - 2]
        return {"pose_h": pose_h, "act_h": act_h, "y": y, "act_y": act_y, "last_y": last_y, "prev_y": prev_y}


def collate(batch: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    import torch

    return {k: torch.from_numpy(np.stack([b[k] for b in batch])).float() for k in batch[0].keys()}


def get_torch_model_class():
    import torch
    from torch import nn

    class PlayerDynamicsTransformer(nn.Module):
        def __init__(self, pose_dim: int, action_dim: int, future: int, players: int = 10, d_model: int = 256, layers: int = 4, heads: int = 8, dropout: float = 0.1) -> None:
            super().__init__()
            self.future = future
            self.players = players
            self.in_proj = nn.Linear(players * (pose_dim + action_dim), d_model)
            self.pos = nn.Parameter(torch.zeros(1, 512, d_model))
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.norm = nn.LayerNorm(d_model)
            self.pose_head = nn.Linear(d_model, future * players * 5)
            self.action_head = nn.Linear(d_model, future * players * action_dim)

        def forward(self, pose_h: Any, act_h: Any) -> tuple[Any, Any]:
            b, h, p, _ = pose_h.shape
            x = torch.cat([pose_h, act_h], dim=-1).reshape(b, h, p * (POSE_DIM + ACTION_DIM))
            x = self.in_proj(x) + self.pos[:, :h]
            x = self.norm(self.encoder(x)[:, -1])
            pose = self.pose_head(x).reshape(b, self.future, self.players, 5)
            action = self.action_head(x).reshape(b, self.future, self.players, ACTION_DIM)
            return pose, action

    return PlayerDynamicsTransformer


def normalize_targets(y: Any) -> Any:
    import torch

    out = y.clone()
    out[..., :3] = out[..., :3] / 1000.0
    out[..., 3:5] = out[..., 3:5] / 180.0 - 1.0
    return out


def denormalize_targets(y: Any) -> Any:
    out = y.clone()
    out[..., :3] = out[..., :3] * 1000.0
    out[..., 3:5] = (out[..., 3:5] + 1.0) * 180.0
    return out


def pose_smooth_l1_loss(pred: Any, target: Any) -> Any:
    import torch

    pos_delta = (pred[..., :3] - target[..., :3]) / 1000.0
    angle_delta = torch.remainder(pred[..., 3:5] - target[..., 3:5] + 180.0, 360.0) - 180.0
    angle_delta = angle_delta / 180.0
    delta = torch.cat([pos_delta, angle_delta], dim=-1)
    return torch.nn.functional.smooth_l1_loss(delta, torch.zeros_like(delta), beta=0.05)


def train_cmd(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader

    manifest = load_json(args.manifest)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    train_ds = WindowDataset(manifest, "train", args.limit_train_windows)
    val_ds = WindowDataset(manifest, "val", args.limit_val_windows)
    if len(train_ds) < args.batch_size:
        raise RuntimeError(f"train windows ({len(train_ds)}) must be >= batch_size ({args.batch_size})")
    if len(val_ds) == 0:
        raise RuntimeError("val windows must be non-empty")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate)
    device = resolve_device(args.cpu)
    Model = get_torch_model_class()
    model = Model(POSE_DIM, ACTION_DIM, int(manifest["future"]), d_model=args.d_model, layers=args.layers, heads=args.heads, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = torch.nn.BCEWithLogitsLoss()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "train_player_dynamics_v0.log"
    best_val = float("inf")
    step = 0
    param_count = sum(p.numel() for p in model.parameters())
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"event": "start", "device": str(device), "params": param_count, "train_windows": len(train_ds), "val_windows": len(val_ds), "pid": os.getpid()}) + "\n")
        while step < args.max_steps:
            for batch in train_loader:
                model.train()
                pose_h = batch["pose_h"].to(device)
                act_h = batch["act_h"].to(device)
                y = batch["y"].to(device)
                act_y = batch["act_y"].to(device)
                pred, logits = model(pose_h, act_h)
                pose_loss = pose_smooth_l1_loss(pred, y)
                act_loss = bce(logits, act_y)
                loss = pose_loss + args.action_loss_weight * act_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                step += 1
                if step % args.log_every == 0 or step == 1:
                    rec = {"event": "train", "step": step, "loss": float(loss.item()), "pose_loss": float(pose_loss.item()), "action_loss": float(act_loss.item())}
                    print(json.dumps(rec), flush=True)
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                if step % args.eval_every == 0 or step == args.max_steps:
                    val_loss = eval_loss(model, val_loader, device, args.action_loss_weight)
                    rec = {"event": "val", "step": step, "val_loss": val_loss}
                    print(json.dumps(rec), flush=True)
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    ckpt = {"model": model.state_dict(), "manifest": manifest, "args": jsonable_args(args), "params": param_count, "step": step, "val_loss": val_loss}
                    torch.save(ckpt, args.out_dir / "last_player_dynamics_v0.pt")
                    if val_loss < best_val:
                        best_val = val_loss
                        torch.save(ckpt, args.out_dir / "best_player_dynamics_v0.pt")
                if step >= args.max_steps:
                    break
    print(json.dumps({"out_dir": str(args.out_dir), "best_val_loss": best_val, "steps": step, "params": param_count}, indent=2))


def eval_loss(model: Any, loader: Any, device: Any, action_weight: float) -> float:
    import torch

    model.eval()
    vals = []
    bce = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            pose_h = batch["pose_h"].to(device)
            act_h = batch["act_h"].to(device)
            y = batch["y"].to(device)
            act_y = batch["act_y"].to(device)
            pred, logits = model(pose_h, act_h)
            loss = pose_smooth_l1_loss(pred, y) + action_weight * bce(logits, act_y)
            vals.append(float(loss.item()))
            if i >= 50:
                break
    return float(np.mean(vals)) if vals else float("nan")


def baseline_preds(batch: dict[str, Any], future: int) -> tuple[Any, Any]:
    import torch

    last = batch["last_y"]
    prev = batch["prev_y"]
    vel = last[..., :3] - prev[..., :3]
    static = last[:, None].repeat(1, future, 1, 1)
    const = static.clone()
    for k in range(future):
        const[:, k, :, :3] = last[..., :3] + vel * float(k + 1)
        const[:, k, :, 3:5] = last[..., 3:5]
    return static, const


def metric_accum(pred: np.ndarray, y: np.ndarray, logits: np.ndarray | None, act_y: np.ndarray, horizons: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for h in horizons:
        p = pred[:, h - 1]
        t = y[:, h - 1]
        disp = np.linalg.norm(p[..., :3] - t[..., :3], axis=-1)
        xy = np.linalg.norm(p[..., :2] - t[..., :2], axis=-1)
        yaw = angle_diff_deg(p[..., 3], t[..., 3])
        pitch = angle_diff_deg(p[..., 4], t[..., 4])
        rec = {
            "pose_count": int(disp.size),
            "pos3d_sum": float(disp.sum()),
            "xy_sum": float(xy.sum()),
            "yaw_deg_sum": float(yaw.sum()),
            "pitch_deg_sum": float(pitch.sum()),
        }
        if logits is not None:
            act_pred = (logits[:, h - 1] >= 0.0).astype(np.float32)
            ok = act_pred == act_y[:, h - 1]
            rec["action_count"] = int(ok.size)
            rec["action_correct"] = int(ok.sum())
        out[str(h)] = rec
    return out


def eval_cmd(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    manifest = ckpt["manifest"]
    ds = WindowDataset(manifest, args.split, args.limit_windows)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate)
    device = resolve_device(args.cpu)
    Model = get_torch_model_class()
    model = Model(POSE_DIM, ACTION_DIM, int(manifest["future"]), d_model=ckpt["args"]["d_model"], layers=ckpt["args"]["layers"], heads=ckpt["args"]["heads"], dropout=0.0).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    horizons = [min(8, int(manifest["future"])), min(16, int(manifest["future"]))]
    buckets = {"model": [], "static": [], "constant_velocity": [], "repeat_last_action": []}
    with torch.no_grad():
        for batch in loader:
            pose_h = batch["pose_h"].to(device)
            act_h = batch["act_h"].to(device)
            pred, logits = model(pose_h, act_h)
            static, const = baseline_preds(batch, int(manifest["future"]))
            y = batch["y"].numpy()
            act_y = batch["act_y"].numpy()
            buckets["model"].append(metric_accum(pred.cpu().numpy(), y, logits.cpu().numpy(), act_y, horizons))
            buckets["static"].append(metric_accum(static.numpy(), y, None, act_y, horizons))
            buckets["constant_velocity"].append(metric_accum(const.numpy(), y, None, act_y, horizons))
            last_action_logits = (batch["act_h"][:, -1:, :, :].repeat(1, int(manifest["future"]), 1, 1).numpy() * 2.0) - 1.0
            buckets["repeat_last_action"].append(metric_accum(static.numpy(), y, last_action_logits, act_y, horizons))
    metrics = reduce_metrics(buckets, horizons)
    metrics["repeat_last_action"]["note"] = "Action baseline repeats last observed action; pose columns are copied from static baseline only for shared metric schema."
    result = {"checkpoint": str(args.checkpoint), "split": args.split, "windows_evaluated": len(ds), "horizons_frames": horizons, "horizons_seconds": [h / 32.0 for h in horizons], "metrics": metrics}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out_json, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def reduce_metrics(buckets: dict[str, list[dict[str, Any]]], horizons: list[int]) -> dict[str, Any]:
    reduced: dict[str, Any] = {}
    for name, rows in buckets.items():
        reduced[name] = {}
        for h in horizons:
            key = str(h)
            pose_count = sum(r[key].get("pose_count", 0) for r in rows)
            rec = {}
            if pose_count:
                rec["pos3d_mean"] = float(sum(r[key].get("pos3d_sum", 0.0) for r in rows) / pose_count)
                rec["xy_mean"] = float(sum(r[key].get("xy_sum", 0.0) for r in rows) / pose_count)
                rec["yaw_deg_mean"] = float(sum(r[key].get("yaw_deg_sum", 0.0) for r in rows) / pose_count)
                rec["pitch_deg_mean"] = float(sum(r[key].get("pitch_deg_sum", 0.0) for r in rows) / pose_count)
            action_count = sum(r[key].get("action_count", 0) for r in rows)
            if action_count:
                rec["action_bit_acc"] = float(sum(r[key].get("action_correct", 0) for r in rows) / action_count)
            reduced[name][key] = rec
    return reduced


def plot_cmd(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    manifest = ckpt["manifest"]
    ds = WindowDataset(manifest, "val", None)
    device = resolve_device(args.cpu)
    Model = get_torch_model_class()
    model = Model(POSE_DIM, ACTION_DIM, int(manifest["future"]), d_model=ckpt["args"]["d_model"], layers=ckpt["args"]["layers"], heads=ckpt["args"]["heads"], dropout=0.0).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for idx in np.linspace(0, max(len(ds) - 1, 0), args.count, dtype=int):
        sample = collate([ds[int(idx)]])
        with torch.no_grad():
            pred, _ = model(sample["pose_h"].to(device), sample["act_h"].to(device))
        static, const = baseline_preds(sample, int(manifest["future"]))
        hist = y_targets_from_pose(sample["pose_h"].numpy()[0])
        gt = sample["y"].numpy()[0]
        pr = pred.cpu().numpy()[0]
        cv = const.numpy()[0]
        fig, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)
        for p in range(10):
            ax = axes.flat[p]
            ax.plot(hist[:, p, 0], hist[:, p, 1], color="0.55", linewidth=1.0, label="hist" if p == 0 else None)
            ax.plot(gt[:, p, 0], gt[:, p, 1], color="#1f77b4", linewidth=1.5, label="GT" if p == 0 else None)
            ax.plot(pr[:, p, 0], pr[:, p, 1], color="#d62728", linewidth=1.2, label="model" if p == 0 else None)
            ax.plot(cv[:, p, 0], cv[:, p, 1], color="#2ca02c", linestyle="--", linewidth=1.0, label="const vel" if p == 0 else None)
            ax.set_title(f"P{p}", fontsize=9)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, linewidth=0.3)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4)
        out = args.out_dir / f"val_window_{int(idx):06d}_trajectory.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        made.append(str(out))
    print(json.dumps({"plots": made}, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inspect")
    p.add_argument("--data-root", type=Path, default=Path(str(source_path('assets', 'csgo-datasets/32f1644d4f42c29d'))))
    p.add_argument("--matches", type=int, default=5)
    p.add_argument("--episodes-per-match", type=int, default=2)
    p.set_defaults(func=inspect_cmd)

    p = sub.add_parser("build")
    p.add_argument("--data-root", type=Path, default=Path(str(source_path('assets', 'csgo-datasets/32f1644d4f42c29d'))))
    p.add_argument("--out-dir", type=Path, default=Path(str(source_path('scene', 'output/dynamics_v0'))))
    p.add_argument("--history", type=int, default=32)
    p.add_argument("--future", type=int, default=16)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--val-matches", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260702)
    p.set_defaults(func=build_cmd)

    p = sub.add_parser("train")
    p.add_argument("--manifest", type=Path, default=Path(str(source_path('scene', 'output/dynamics_v0/manifest_player_dynamics_v0.json'))))
    p.add_argument("--out-dir", type=Path, default=Path(str(source_path('scene', 'output/dynamics_v0/train_v0'))))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--action-loss-weight", type=float, default=0.05)
    p.add_argument("--limit-train-windows", type=int, default=None)
    p.add_argument("--limit-val-windows", type=int, default=None)
    p.add_argument("--seed", type=int, default=20260702)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=train_cmd)

    p = sub.add_parser("eval")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-json", type=Path, default=Path(str(source_path('scene', 'output/dynamics_v0/eval_player_dynamics_v0.json'))))
    p.add_argument("--split", default="val")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit-windows", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=eval_cmd)

    p = sub.add_parser("plot")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path(str(source_path('scene', 'output/dynamics_v0/plots'))))
    p.add_argument("--count", type=int, default=5)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=plot_cmd)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
