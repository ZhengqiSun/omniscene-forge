#!/usr/bin/env python3
"""Base dual-expert ablation scorer v1 (R1 ruler, ablation design 2026-08-11).

Replaces zq/tools/evaluate_memory_dense_adapter_ablation_v0.py for the base
two-expert mainline: that scorer builds the Fast single-expert model and
silently drops LoRA plus both projectors when fed current checkpoints (C9).
This tool imports the trainer as a library and reuses its exact build path
(import_wan_model_base -> wrap -> inject_lora -> build_dense_encoder /
build_state_projector / build_interaction_projector) plus its token-assembly
and loss functions, so scorer and trainer cannot drift.

Design contract (verify-corrected):
- Model/geometry hyperparameters derive from ckpt['config'] (never hardcoded;
  interaction feature dim included), CLI can override paths only.
- Fail-closed checkpoint pre-check runs BEFORE any load: correct sub-dict key
  shapes (state['memory_dense_state_projector']['proj.weight'] == (cond_dim,3),
  interaction proj == (cond_dim, feature_dim)); trainer.load_checkpoint's
  projector fail-open path is therefore never trusted alone.
- Variants: true / shuffled_xmatch (full condition set from a different-match
  partner window, partner ids logged) / blank_dense (zero dense image through
  the current encoder; state,interaction absent) / disabled (COND_KEY omitted:
  adapter no-op, LoRA active - NOT the base model).
- Pairing: every variant shares the same window list, sigma grid (stratified
  over the LOW band) and per-window noise (sha-derived); rows carry match_id
  for match-cluster significance (R6).
- Region loss is reported raw with normalize='total' and NO training weight
  (protocol N-03: evaluation decouples from the training w; judge applies its
  own weighting downstream). whole_frame_loss is the primary metric for the
  state/interaction axes.
- --ckpt-check-only performs the fail-closed pre-check on CPU and exits.
- --a7-check (positive control, trained ckpt only): asserts condition tokens
  are non-zero and pred_true differs from pred_disabled on the first window -
  a scorer whose conditions never reach the model fails loudly instead of
  passing every relative check.
"""

from __future__ import annotations

from runtime_paths import ASSET_ROOT, LINGBOT_ROOT

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

TRAINER_PATH = Path(__file__).resolve().parent / "train_memory_dense_adapter_interaction_v1.py"
LOW_BOUNDARY_FALLBACK = 0.947
S4_DEFAULT = ASSET_ROOT / "prepared"


def load_trainer_module() -> Any:
    sys.path.insert(0, str(TRAINER_PATH.parent))
    spec = importlib.util.spec_from_file_location("r1_trainer_mod", TRAINER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def stable_hex(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}|{key}".encode("utf-8")).hexdigest()


def ckpt_precheck(ckpt: dict[str, Any], path: Path) -> dict[str, Any]:
    """Fail-closed namespace/shape audit. Returns derived dims."""
    for key in ("kind", "step", "state", "config"):
        if key not in ckpt:
            raise ValueError(f"{path}: checkpoint missing top-level key {key!r}")
    cfg = ckpt["config"]
    state = ckpt["state"]
    adapter_cfg = cfg.get("adapter_config") or {}
    cond_dim = int(adapter_cfg.get("cond_dim", 0))
    if cond_dim <= 0:
        raise ValueError(f"{path}: config.adapter_config.cond_dim missing")
    if not state.get("memory_dense_adapter"):
        raise ValueError(f"{path}: empty memory_dense_adapter state")
    lora_cfg = cfg.get("lora_config") or {}
    lora_rank = int(lora_cfg.get("rank") or 0)
    if lora_rank > 0 and not state.get("memory_dense_lora"):
        raise ValueError(f"{path}: config says lora rank {lora_rank} but memory_dense_lora state is empty")
    enc_cfg = cfg.get("dense_condition_encoder") or {}
    if enc_cfg.get("type") == "frozen_vae" and not state.get("memory_dense_encoder"):
        raise ValueError(f"{path}: frozen_vae encoder but memory_dense_encoder (projection) state is empty")

    derived = {"cond_dim": cond_dim, "lora_rank": lora_rank,
               "encoder_type": enc_cfg.get("type"), "step": int(ckpt.get("step", -1))}

    state_cfg = cfg.get("state_channels_v0") or {}
    if state_cfg.get("enabled"):
        sp = state.get("memory_dense_state_projector") or {}
        w = sp.get("proj.weight")
        n_state = len(state_cfg.get("channels") or [])
        if w is None or tuple(w.shape) != (cond_dim, n_state):
            raise ValueError(
                f"{path}: state projector proj.weight expected {(cond_dim, n_state)}, "
                f"got {None if w is None else tuple(w.shape)} (sub-dict key contract)"
            )
        derived["state_channels"] = n_state
    inter_cfg = cfg.get("interaction_channels_v1") or {}
    if inter_cfg.get("enabled"):
        ip = state.get("memory_dense_interaction_projector") or {}
        w = ip.get("proj.weight")
        feature_dim = int((inter_cfg.get("projector") or {}).get("feature_dim") or 0)
        if feature_dim <= 0:
            raise ValueError(f"{path}: interaction enabled but projector.feature_dim missing in config")
        if w is None or tuple(w.shape) != (cond_dim, feature_dim):
            raise ValueError(
                f"{path}: interaction projector proj.weight expected {(cond_dim, feature_dim)}, "
                f"got {None if w is None else tuple(w.shape)} (derive from ckpt config, never hardcode)"
            )
        derived["interaction_feature_dim"] = feature_dim
    return derived


def build_targs(cfg: dict[str, Any], cli: argparse.Namespace, split: str) -> argparse.Namespace:
    adapter_cfg = cfg["adapter_config"]
    lora_cfg = cfg.get("lora_config") or {}
    enc_cfg = cfg.get("dense_condition_encoder") or {}
    dense_hw = cfg.get("dense_native_hw") or [240, 416]
    video_hw = cfg.get("video_hw") or [480, 832]
    return argparse.Namespace(
        # data
        map_manifest=cli.map_manifest, cache_manifest=cli.cache_manifest,
        extra_map_manifest=list(cli.extra_map_manifest), extra_cache_manifest=list(cli.extra_cache_manifest),
        state_cache_manifest=cli.state_cache_manifest,
        train_split=split, split_key="match",
        latent_frames=int(cfg.get("latent_frames", 21)),
        video_frames=int(cli.video_frames), raw_stride=int(cli.raw_stride),
        video_height=int(video_hw[0]), video_width=int(video_hw[1]),
        vae_stride=int(adapter_cfg.get("vae_stride", 8)),
        patch_size_hw=int(adapter_cfg.get("wan_patch_size_hw", 2)),
        require_backend_id=cli.require_backend_id,
        allow_static_dense_repeat=False, limit=None, log_loader_stages=False,
        # model
        use_base_model=True, base_expert=cli.expert,
        base_sigma_min=None, base_sigma_max=None,
        lingbot_repo=cli.lingbot_repo, ckpt_dir=cli.ckpt_dir,
        cond_dim=int(adapter_cfg["cond_dim"]),
        adapter_hidden_dim=int(adapter_cfg.get("adapter_hidden_dim", 512)),
        encoder_hidden_dim=int(adapter_cfg.get("encoder_hidden_dim", 64)),
        dense_channels=int(adapter_cfg.get("dense_channels", 7)),
        adapter_residual_mode=str(adapter_cfg.get("residual_mode", "cond_gated")),
        adapter_wrap_first_blocks=int(adapter_cfg.get("wrap_first_blocks") or 0),
        adapter_residual_scale_init=float(adapter_cfg.get("residual_scale_init", 1.0)),
        activation_checkpoint_blocks=False,
        highfreq_branch=bool(adapter_cfg.get("highfreq_branch", False)),
        highfreq_kind=str(adapter_cfg.get("highfreq_kind", "laplacian")),
        highfreq_hidden_dim=int(adapter_cfg.get("highfreq_hidden_dim", 32)),
        highfreq_stride=int(adapter_cfg.get("highfreq_stride", 16)),
        image_to_wan_token_stride=int(adapter_cfg.get("image_to_wan_token_stride", 16)),
        dense_condition_encoder=str(enc_cfg.get("type", "frozen_vae")),
        only_player_dense_channels=False,
        dense_vae_pth=Path(enc_cfg["vae_pth"]) if enc_cfg.get("vae_pth") else None,
        dense_vae_packing=str(enc_cfg.get("packing", "img1_mask_img2_player_v0")),
        dense_hw=[int(dense_hw[0]), int(dense_hw[1])],
        dense_resize_mode=str(cfg.get("dense_resize_mode", "bilinear")),
        lora_rank=int(lora_cfg.get("rank") or 0),
        lora_targets=(
            ",".join(lora_cfg["targets"]) if isinstance(lora_cfg.get("targets"), list)
            else str(lora_cfg.get("targets") or "attn,mlp")
        ),
        lora_alpha=lora_cfg.get("alpha"), lora_dropout=float(lora_cfg.get("dropout") or 0.0),
        w4_freeze_lora=False, w4_freeze_state_projector=False,
        seed=int(cli.seed),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--n-windows", type=int, default=96)
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--noise-seed", type=int, default=20260811)
    ap.add_argument("--variants", default="true,shuffled_xmatch,blank_dense,disabled")
    ap.add_argument("--expert", choices=["low", "high"], default="low")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--ckpt-check-only", action="store_true")
    ap.add_argument("--a7-check", action="store_true",
                    help="Positive control on the first window (trained ckpt only): conditions must reach the model")
    ap.add_argument("--map-manifest", type=Path, default=S4_DEFAULT / "tier69h/tier69h_v2_combined_manifest_memorymask_view_v1.json")
    ap.add_argument("--cache-manifest", type=Path, default=S4_DEFAULT / "tier69h/tier69h_v2_cache_manifest_split_v1.jsonl")
    ap.add_argument("--extra-map-manifest", type=Path, action="append", default=None,
                    help="argparse append semantics: passing this REPLACES the event50h default (default=None resolved post-parse)")
    ap.add_argument("--extra-cache-manifest", type=Path, action="append", default=None)
    ap.add_argument("--state-cache-manifest", type=Path, default=ASSET_ROOT / "state/state_cache_manifest_v1.jsonl")
    ap.add_argument("--lingbot-repo", type=Path, default=LINGBOT_ROOT)
    ap.add_argument("--ckpt-dir", type=Path, default=ASSET_ROOT / "lingbot-world-base-cam")
    ap.add_argument("--video-frames", type=int, default=81)
    ap.add_argument("--raw-stride", type=int, default=2)
    ap.add_argument("--require-backend-id", default="bsp_faces_disp_gpu")
    cli = ap.parse_args()
    if cli.extra_map_manifest is None:
        cli.extra_map_manifest = [S4_DEFAULT / "event50h/event50h_v2_combined_manifest_memorymask_view_v1.json"]
    if cli.extra_cache_manifest is None:
        cli.extra_cache_manifest = [S4_DEFAULT / "event50h/event50h_v2_cache_manifest_split_v1.jsonl"]

    trainer = load_trainer_module()
    ckpt = torch.load(cli.ckpt, map_location="cpu")
    derived = ckpt_precheck(ckpt, cli.ckpt)
    print(json.dumps({"event": "ckpt_precheck", "path": str(cli.ckpt), **derived}), flush=True)
    if cli.ckpt_check_only:
        print(json.dumps({"event": "ckpt_check_only", "status": "PASS"}), flush=True)
        return

    cfg = ckpt["config"]
    targs = build_targs(cfg, cli, cli.split)
    device = torch.device(cli.device)
    dtype = torch.bfloat16
    variants = [v.strip() for v in cli.variants.split(",") if v.strip()]

    # ---- data: heldout windows (deterministic, per-match round-robin) ----
    manifest_pairs = trainer.manifest_pairs_from_args(targs)
    state_rows = trainer.load_state_manifest_for_args(targs)
    pool: list[tuple[dict[str, Any], int]] = []
    for ridx, (mp, cp) in enumerate(manifest_pairs):
        map_sha = trainer.sha256_file(mp)
        records, _ids, _rep = trainer.load_aligned_cache_records_lightweight(
            cp, args=targs, release_index=ridx, map_manifest_sha256=map_sha)
        pool.extend((r, ridx) for r in records)
    covered = [(r, x) for r, x in pool if str(r.get("clip_id")) in state_rows]
    print(json.dumps({"event": "pool", "split": cli.split, "total": len(pool), "sidecar_covered": len(covered)}), flush=True)
    by_match: dict[str, list[tuple[dict[str, Any], int]]] = {}
    for r, x in covered:
        by_match.setdefault(str(r.get("game_id")), []).append((r, x))
    for mid in by_match:
        by_match[mid].sort(key=lambda it: stable_hex(cli.seed, str(it[0].get("clip_id"))))
    order = sorted(by_match, key=lambda mid: stable_hex(cli.seed, mid))
    selected: list[tuple[dict[str, Any], int, str]] = []
    cur = {m: 0 for m in order}
    while len(selected) < cli.n_windows:
        moved = False
        for mid in order:
            if len(selected) >= cli.n_windows:
                break
            rows_m = by_match[mid]
            if cur[mid] < len(rows_m):
                r, x = rows_m[cur[mid]]
                cur[mid] += 1
                selected.append((r, x, mid))
                moved = True
        if not moved:
            break
    print(json.dumps({"event": "selected", "n": len(selected), "matches": len({m for _, _, m in selected})}), flush=True)

    needed: dict[int, set] = {}
    ids_per: list[list[str]] = []
    for r, x, _m in selected:
        sids = trainer.resolve_record_sample_ids(r, latent_frames=targs.latent_frames,
                                                 allow_static_dense_repeat=False)[: targs.latent_frames]
        ids_per.append(list(sids))
        needed.setdefault(x, set()).update(sids)
    releases = {x: trainer.build_lightweight_release_from_manifest(
        manifest_pairs[x][0], require_backend_id=targs.require_backend_id, sample_ids=ids)
        for x, ids in needed.items()}

    # ---- model ----
    WanModelBase, base_cfg = trainer.import_wan_model_base(targs.lingbot_repo)
    expert_sub = base_cfg.low_noise_checkpoint if cli.expert == "low" else base_cfg.high_noise_checkpoint
    model = WanModelBase.from_pretrained(str(targs.ckpt_dir), subfolder=expert_sub,
                                         torch_dtype=dtype, control_type="cam").to(device)
    model.requires_grad_(False)
    boundary = float(getattr(base_cfg, "boundary", LOW_BOUNDARY_FALLBACK))
    sigma_hi = boundary if cli.expert == "low" else 1.0
    sigma_lo = 0.0 if cli.expert == "low" else boundary
    config = trainer.build_adapter_config(targs)
    trainer.wrap_wan_model_fast_with_memory_dense_adapter(model, config, freeze_base=True)
    trainer.inject_lora_into_wan_model_fast(model, rank=targs.lora_rank, targets=targs.lora_targets,
                                            alpha=targs.lora_alpha, dropout=targs.lora_dropout)
    model.to(device=device, dtype=dtype)
    encoder = trainer.build_dense_encoder(targs, config, device=device, dtype=dtype)
    state_projector = trainer.build_state_projector(targs, device=device, dtype=dtype)
    interaction_projector = trainer.build_interaction_projector(targs, device=device, dtype=dtype)
    params = [p for p in model.parameters() if p.requires_grad] or [torch.nn.Parameter(torch.zeros(1))]
    dummy_optimizer = torch.optim.AdamW([{"params": params}])
    step = trainer.load_checkpoint(
        cli.ckpt, model=model, encoder=encoder, optimizer=dummy_optimizer,
        reset_optimizer=True, state_projector=state_projector,
        interaction_projector=interaction_projector,
    )
    model.eval(); encoder.eval()
    if state_projector is not None:
        state_projector.eval()
    if interaction_projector is not None:
        interaction_projector.eval()
    print(json.dumps({"event": "model_loaded", "step": step, "expert": cli.expert,
                      "boundary": boundary, "sigma_band": [sigma_lo, sigma_hi]}), flush=True)

    text_context = trainer.load_text_context(selected[0][0], device).to(dtype)
    wan_token_hw = (targs.video_height // targs.vae_stride // targs.patch_size_hw,
                    targs.video_width // targs.vae_stride // targs.patch_size_hw)

    def tokens_for(record, samples):
        toks, _ = trainer.dense_tokens_for_samples(encoder, samples, device=device, dtype=dtype,
                                                   target_token_hw=wan_token_hw)
        if state_projector is not None:
            st = trainer.state_tokens_for_record_chunk(
                state_projector, state_rows, record, chunk_start=0, chunk_size=len(samples),
                device=device, dtype=dtype, target_token_hw=wan_token_hw)
            if st is not None:
                toks = toks + st
        if interaction_projector is not None:
            it = trainer.interaction_tokens_for_record_chunk(
                interaction_projector, state_rows, record, chunk_start=0, chunk_size=len(samples),
                device=device, dtype=dtype, target_token_hw=wan_token_hw)
            if it is not None:
                toks = toks + it
        return toks

    cli.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = cli.out_dir / f"ablation_rows_{cli.split}_step{step}.jsonl"
    t0 = time.time()
    sums: dict[str, float] = {v: 0.0 for v in variants}
    paired_ts = 0
    F = torch.nn.functional
    with torch.no_grad(), rows_path.open("w", encoding="utf-8") as rows_f:
        for idx, (record, ridx, mid) in enumerate(selected):
            clip_id = str(record.get("clip_id"))
            samples = [releases[ridx].by_id[s] for s in ids_per[idx]]
            k = 1
            while selected[(idx + k) % len(selected)][2] == mid:
                k += 1
                if k > len(selected):
                    raise SystemExit("no cross-match partner available")
            p_rec, p_ridx, p_mid = selected[(idx + k) % len(selected)]
            p_samples = [releases[p_ridx].by_id[s] for s in ids_per[(idx + k) % len(selected)]]

            x0, cond = trainer.load_latent_pair(record, device, torch.float32)
            sigma = sigma_lo + (idx + 0.5) / len(selected) * (sigma_hi - sigma_lo)
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(stable_hex(cli.noise_seed, clip_id)[:12], 16))
            noise = torch.randn(x0.shape, generator=gen, dtype=torch.float32).to(device)
            xt = ((1.0 - sigma) * x0 + sigma * noise).to(dtype)
            target = (noise - x0).float()
            timestep = torch.tensor([sigma * 1000.0], device=device, dtype=torch.float32)
            cam = trainer.prepare_cam_chunk(record, latent_frames=targs.latent_frames, chunk_start=0,
                                            chunk_size=targs.latent_frames, height=targs.video_height,
                                            width=targs.video_width, vae_stride=targs.vae_stride,
                                            device=device, dtype=dtype)
            latent_hw = tuple(x0.shape[-2:])
            variant_tokens: dict[str, torch.Tensor | None] = {}
            for v in variants:
                if v == "true":
                    variant_tokens[v] = tokens_for(record, samples)
                elif v == "shuffled_xmatch":
                    variant_tokens[v] = tokens_for(p_rec, p_samples)
                elif v == "blank_dense":
                    ref = trainer.load_dense(samples[0])
                    zeros = torch.zeros((len(samples), *ref.shape), device=device, dtype=dtype)
                    bt, _ = encoder(zeros, target_token_hw=wan_token_hw)
                    variant_tokens[v] = bt.reshape(1, len(samples) * bt.shape[1], bt.shape[2]).to(dtype)
                elif v == "disabled":
                    variant_tokens[v] = None
                else:
                    raise SystemExit(f"unknown variant {v!r}")
            seq_len = targs.latent_frames * wan_token_hw[0] * wan_token_hw[1]
            preds: dict[str, torch.Tensor] = {}
            per: dict[str, dict[str, Any]] = {}
            for v in variants:
                dit_cond: dict[str, Any] = {"c2ws_plucker_emb": cam.chunk(1, dim=0)}
                if variant_tokens[v] is not None:
                    dit_cond[trainer.COND_KEY] = variant_tokens[v]
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    pred = model(x=[xt], t=timestep, context=[text_context], seq_len=seq_len,
                                 y=[cond.to(dtype)], dit_cond_dict=dit_cond)[0]
                preds[v] = pred
                whole = float(F.mse_loss(pred.float(), target).cpu())
                region, region_pixels = trainer.region_loss_for_samples(
                    pred, target, samples, latent_hw=latent_hw, min_pixels=1, normalize="total")
                per[v] = {"whole_frame_loss": whole,
                          "region_loss": None if region is None else float(region.cpu()),
                          "region_pixels": int(region_pixels)}
                sums[v] += whole
            if cli.a7_check and idx == 0:
                tt = variant_tokens.get("true")
                if tt is None or float(tt.abs().max()) == 0.0:
                    raise AssertionError("A7: true condition tokens are all-zero or missing")
                if "disabled" in preds and torch.allclose(preds["true"], preds["disabled"]):
                    raise AssertionError("A7: pred_true == pred_disabled; conditions never reach the model")
                print(json.dumps({"event": "a7_positive_control", "status": "PASS",
                                  "token_abs_max": float(tt.abs().max())}), flush=True)
            if "true" in per and "shuffled_xmatch" in per:
                if per["true"]["whole_frame_loss"] < per["shuffled_xmatch"]["whole_frame_loss"]:
                    paired_ts += 1
            rows_f.write(json.dumps({
                "clip_id": clip_id, "match_id": mid, "record_index": idx,
                "release_index": ridx, "chunk_ord": 0, "chunk_start": 0,
                "sigma": sigma, "split": cli.split,
                "partner_clip_id": str(p_rec.get("clip_id")), "partner_match_id": p_mid,
                "variants": per,
            }, ensure_ascii=False) + "\n")
            if (idx + 1) % 8 == 0:
                print(json.dumps({"event": "progress", "done": idx + 1,
                                  "elapsed_sec": round(time.time() - t0, 1)}), flush=True)

    n = len(selected)
    summary = {
        "kind": "base_ablation_summary_v1", "ckpt": str(cli.ckpt), "step": step,
        "expert": cli.expert, "split": cli.split, "n_windows": n,
        "variants": variants, "sigma_band": [sigma_lo, sigma_hi],
        "mean_whole_frame_loss": {v: sums[v] / n for v in variants},
        "true_lt_shuffled_count": paired_ts,
        "rows": str(rows_path),
        "governance": "raw numbers only; pass/fail judgement belongs to the R6 match-cluster judge (first round report-only)",
    }
    (cli.out_dir / f"ablation_summary_{cli.split}_step{step}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", **{k: summary[k] for k in ('step', 'split', 'n_windows', 'true_lt_shuffled_count', 'mean_whole_frame_loss')}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

