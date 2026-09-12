from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import CHECKPOINT_KIND, PAIR_KIND
from .lora import load_lora_state_dict, lora_state_dict


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def base_identity(root: Path, expert: str, *, content_hash: bool = False) -> dict[str, Any]:
    subdir = root / f"{expert}_noise_model"
    index = subdir / "diffusion_pytorch_model.safetensors.index.json"
    config = subdir / "config.json"
    if not index.is_file() or not config.is_file():
        raise FileNotFoundError(f"incomplete {expert} expert under {subdir}")
    data = json.loads(index.read_text(encoding="utf-8"))
    shards = sorted(set(data.get("weight_map", {}).values()))
    missing = [name for name in shards if not (subdir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing checkpoint shards: {missing}")
    inventory = [(name, (subdir / name).stat().st_size) for name in shards]
    digest = hashlib.sha256()
    digest.update(sha256_file(index).encode())
    digest.update(sha256_file(config).encode())
    digest.update(json.dumps(inventory, separators=(",", ":")).encode())
    result = {
        "root": str(root.resolve()), "expert": expert,
        "structural_sha256": digest.hexdigest(), "index_sha256": sha256_file(index),
        "config_sha256": sha256_file(config), "shard_count": len(shards),
        "shard_bytes": sum(size for _, size in inventory), "shards": inventory,
    }
    if content_hash:
        shard_hashes = [(name, sha256_file(subdir / name)) for name in shards]
        content_digest = hashlib.sha256()
        content_digest.update(json.dumps(shard_hashes, separators=(",", ":")).encode())
        result["shard_sha256"] = shard_hashes
        result["content_sha256"] = content_digest.hexdigest()
    return result


def _identity_hash(identity: dict[str, Any]) -> str:
    return identity.get("content_sha256", identity["structural_sha256"])


def capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(path: Path, *, model, optimizer, scheduler, step: int, config: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": CHECKPOINT_KIND, "version": 1, "global_step": int(step),
        "expert": config["expert"], "base_identity": config["base_identity"],
        "manifest_sha256": config["manifest_sha256"], "lora_config": config["lora_config"],
        "lora": lora_state_dict(model), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "rng": capture_rng(), "effective_config": config,
    }
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    latest = path.parent / "latest.json"
    latest.write_text(json.dumps({"checkpoint": path.name, "global_step": step}, indent=2) + "\n", encoding="utf-8")


def load_checkpoint(path: Path, *, model, optimizer=None, scheduler=None, expected: dict[str, Any], restore_rng_state=True) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != CHECKPOINT_KIND:
        raise ValueError(f"unsupported checkpoint kind: {payload.get('kind')}")
    checks = {
        "expert": expected["expert"], "manifest_sha256": expected["manifest_sha256"],
    }
    for key, value in checks.items():
        if payload.get(key) != value:
            raise ValueError(f"resume {key} mismatch: {payload.get(key)!r} != {value!r}")
    if _identity_hash(payload.get("base_identity", {})) != _identity_hash(expected["base_identity"]):
        raise ValueError("resume base checkpoint identity mismatch")
    if payload.get("lora_config") != expected["lora_config"]:
        raise ValueError("resume LoRA configuration mismatch")
    load_lora_state_dict(model, payload["lora"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if restore_rng_state:
        restore_rng(payload["rng"])
    return int(payload["global_step"])


def load_pair(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("kind") != PAIR_KIND:
        raise ValueError(f"unsupported pair kind: {obj.get('kind')}")
    root = path.resolve().parent
    for key in ("low_checkpoint", "high_checkpoint"):
        item = Path(obj[key])
        obj[key] = str(item if item.is_absolute() else (root / item).resolve())
    return obj


def load_lora_for_inference(path: Path, *, model, expert: str, base: dict[str, Any]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != CHECKPOINT_KIND or payload.get("expert") != expert:
        raise ValueError(f"{path}: expected a {expert} {CHECKPOINT_KIND}")
    if _identity_hash(payload.get("base_identity", {})) != _identity_hash(base):
        raise ValueError(f"{path}: base checkpoint identity mismatch")
    load_lora_state_dict(model, payload["lora"])
    return {
        "path": str(path.resolve()), "sha256": sha256_file(path),
        "global_step": int(payload["global_step"]), "expert": expert,
        "manifest_sha256": payload["manifest_sha256"],
        "lora_config": payload["lora_config"],
    }
