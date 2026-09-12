from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from .schema import sha256_file

ACTION_TOKEN = ".action_attn."


def require_official_scope(scope_repo: str | Path) -> Path:
    repo = Path(scope_repo).resolve()
    required = [repo / "inference.py", repo / "diffsynth/models/scope_dit.py", repo / "diffsynth/pipelines/scope_pipeline.py"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete official SCOPE repo: {missing}")
    return repo


def import_official_inference(scope_repo: str | Path):
    repo = require_official_scope(scope_repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("scope_official_inference", repo / "inference.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def init_official_pipeline(scope_repo: str | Path, model_dir: str | Path):
    return import_official_inference(scope_repo).init_pipeline(str(Path(model_dir).resolve()))


def configure_actionmodule_only(dit) -> list[dict[str, Any]]:
    manifest = []
    for name, param in dit.named_parameters():
        allowed = ACTION_TOKEN in f".{name}."
        param.requires_grad_(allowed)
        manifest.append({"name": name, "shape": list(param.shape), "numel": param.numel(), "trainable": allowed})
    if not any(item["trainable"] for item in manifest):
        raise RuntimeError("no blocks.N.action_attn parameters found")
    return manifest


def assert_actionmodule_gradients(dit) -> None:
    offenders = [name for name, p in dit.named_parameters() if p.grad is not None and ACTION_TOKEN not in f".{name}."]
    if offenders:
        raise RuntimeError(f"gradient leaked outside ActionModule: {offenders[:10]}")
    trainable_without_grad = [name for name, p in dit.named_parameters() if p.requires_grad and p.grad is None]
    if trainable_without_grad:
        raise RuntimeError(f"trainable ActionModule parameters without gradient: {trainable_without_grad[:10]}")


def load_result_is_clean(missing: list[str], unexpected: list[str]) -> None:
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:20]}, unexpected={unexpected[:20]}")


def parameter_manifest_sha256(items: list[dict[str, Any]]) -> str:
    payload = json.dumps(items, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def official_checkpoint_fingerprint(model_dir: str | Path) -> str:
    """Content fingerprint over the released DiT, VAE, UMT5 and tokenizer files."""
    root=Path(model_dir).resolve()
    files=sorted(root.glob("model-*-of-*.safetensors")) or ([root/"SCOPE.safetensors"] if (root/"SCOPE.safetensors").is_file() else [])
    for required in (root/"models_t5_umt5-xxl-enc-bf16.pth",root/"Wan2.2_VAE.pth"):
        if not required.is_file(): raise FileNotFoundError(required)
        files.append(required)
    tokenizer=root/"google/umt5-xxl"
    if not tokenizer.is_dir(): raise FileNotFoundError(tokenizer)
    files.extend(sorted(p for p in tokenizer.rglob("*") if p.is_file()))
    if not files: raise FileNotFoundError("official SCOPE DiT shards not found")
    digest=hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode()+b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()
