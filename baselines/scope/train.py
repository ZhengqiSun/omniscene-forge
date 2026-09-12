from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch

from baselines.scope.scope_bridge.checkpoint import load_checkpoint, save_checkpoint
from baselines.scope.scope_bridge.dataset import ScopeManifestDataset, assert_train_split
from baselines.scope.scope_bridge.model_loader import assert_actionmodule_gradients, configure_actionmodule_only, init_official_pipeline, official_checkpoint_fingerprint, parameter_manifest_sha256
from baselines.scope.scope_bridge.schema import manifest_sha256, sha256_file
from baselines.scope.actions.validate_scope_parquet import validate as validate_action_parquet


def git_commit(path: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision=Path(path).parent/(Path(path).name+".REVISION")
        if revision.is_file(): return revision.read_text().strip()
        raise RuntimeError(f"cannot establish git revision for {path}")


def parse_args():
    p=argparse.ArgumentParser(description="SCOPE ActionModule-only fine-tuning bridge")
    p.add_argument("--manifest",required=True); p.add_argument("--scope-repo",required=True); p.add_argument("--model-dir",required=True)
    p.add_argument("--mouse-calibration",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--project-repo",default=str(Path(__file__).resolve().parents[2]))
    p.add_argument("--max-steps",type=int,default=1); p.add_argument("--learning-rate",type=float,default=1e-5); p.add_argument("--global-batch-size",type=int,default=1)
    p.add_argument("--gradient-accumulation",type=int,default=1); p.add_argument("--gradient-clip",type=float,default=1.0); p.add_argument("--seed",type=int,default=0)
    p.add_argument("--split",choices=("train","test"),default="train"); p.add_argument("--resume"); p.add_argument("--dry-run",action="store_true"); p.add_argument("--debug-allow-test",action="store_true")
    p.add_argument("--activation-checkpointing",action="store_true"); p.add_argument("--cpu-offload",action="store_true")
    return p.parse_args()


def main() -> None:
    args=parse_args(); torch.manual_seed(args.seed)
    if args.split == "test" and not args.debug_allow_test:
        raise ValueError("refusing split=test; --debug-allow-test is required")
    dataset=ScopeManifestDataset(args.manifest,"train",{args.split}); debug=assert_train_split(dataset.samples,args.debug_allow_test)
    if not len(dataset): raise ValueError(f"manifest contains no split={args.split} rows")
    calibration=json.loads(Path(args.mouse_calibration).read_text())
    if calibration.get("fit_split") != "train": raise ValueError("mouse calibration was not fitted on train")
    summary={"manifest_sha256":manifest_sha256(args.manifest),"samples":len(dataset),"splits":sorted({s.split for s in dataset.samples}),"debug":debug,"trainable_scope":"blocks.*.action_attn only","forbidden_inputs":["camera_pose","dense","state","map_memory","player_mask","other_views"]}
    if args.dry_run:
        print(json.dumps(summary,indent=2)); return
    if args.global_batch_size != args.gradient_accumulation:
        raise NotImplementedError("current single-GPU bridge requires global_batch_size == gradient_accumulation")
    pipe=init_official_pipeline(args.scope_repo,args.model_dir)
    manifest=configure_actionmodule_only(pipe.dit); trainable=[p for p in pipe.dit.parameters() if p.requires_grad]
    for module in (pipe.vae, pipe.text_encoder, getattr(pipe,"dit2",None)):
        if module is not None:
            module.eval()
            for parameter in module.parameters(): parameter.requires_grad_(False)
    pipe.dit.train(); pipe.scheduler.set_timesteps(1000,training=True)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/"trainable_parameters.json").write_text(json.dumps(manifest,indent=2)+"\n")
    optimizer=torch.optim.AdamW(trainable,lr=args.learning_rate); scheduler=torch.optim.lr_scheduler.ConstantLR(optimizer)
    metadata={**summary,"action_calibration_sha256":sha256_file(args.mouse_calibration),"base_checkpoint_fingerprint":official_checkpoint_fingerprint(args.model_dir),"scope_commit":git_commit(args.scope_repo),"project_commit":git_commit(args.project_repo),"trainable_scope_sha256":parameter_manifest_sha256(manifest),"effective_config":vars(args)}
    if args.resume: start=load_checkpoint(args.resume,pipe.dit,optimizer,scheduler,metadata)
    else: start=0
    import sys
    sys.path.insert(0,args.scope_repo)
    from diffsynth.diffusion.loss import FlowMatchSFTLoss
    from diffsynth.pipelines.scope_pipeline import WanVideoUnit_PromptEmbedder
    from diffsynth.utils.data import LowMemoryVideo, crop_and_resize
    from inference import load_actions

    def prepare(row):
        if row.get("target_latent"):
            raise ValueError("target_latent remains disabled until serialized VAE dtype/shape is runtime-audited")
        if not row.get("target_video") or not row.get("scope_action_path"):
            raise ValueError(f"{row['sample_id']}: training requires target_video and converted scope_action_path")
        action_report=validate_action_parquet(row["scope_action_path"],row["num_frames"])
        if not action_report["valid"]: raise ValueError(f"{row['sample_id']}: invalid action parquet: {action_report['errors']}")
        source=LowMemoryVideo(row["target_video"])
        if len(source) != row["num_frames"]:
            raise ValueError(f"{row['sample_id']}: target has {len(source)} frames, expected {row['num_frames']}")
        frames=[crop_and_resize(source[i],row["height"],row["width"]) for i in range(len(source))]
        with torch.no_grad():
            pipe.load_models_to_device(["vae"])
            pixels=pipe.preprocess_video(frames)
            clean=pipe.vae.encode(pixels,device=pipe.device,tiled=True,tile_size=(30,52),tile_stride=(15,26)).to(dtype=pipe.torch_dtype,device=pipe.device)
            first_image=read_initial_image(row,frames[0])
            image=pipe.preprocess_image(first_image.resize((row["width"],row["height"]))).transpose(0,1)
            first=pipe.vae.encode([image],device=pipe.device,tiled=True,tile_size=(30,52),tile_stride=(15,26))
            context=WanVideoUnit_PromptEmbedder().encode_prompt(pipe,row["prompt"])
        keyboard,mouse=load_actions(row["scope_action_path"],row["num_frames"],device=str(pipe.device))
        if keyboard.shape[1] != row["num_frames"] or mouse.shape[1] != row["num_frames"]:
            raise ValueError("action/video timestamp length mismatch")
        return {"input_latents":clean,"first_frame_latents":first,"context":context,"keyboard_action":keyboard,"mouse_action":mouse,"fuse_vae_embedding_in_latents":True,"use_gradient_checkpointing":args.activation_checkpointing,"use_gradient_checkpointing_offload":args.cpu_offload}

    def read_initial_image(row, fallback):
        from PIL import Image
        if row.get("initial_image"): return Image.open(row["initial_image"]).convert("RGB")
        if row.get("raw_video"):
            return LowMemoryVideo(row["raw_video"])[int(row["raw_start"])]
        return fallback

    optimizer.zero_grad(set_to_none=True); rows=list(dataset); step=start
    while step < args.max_steps:
        row=rows[step % len(rows)]; loss=FlowMatchSFTLoss(pipe,**prepare(row)) / args.gradient_accumulation
        if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        if (step + 1) % args.gradient_accumulation == 0:
            assert_actionmodule_gradients(pipe.dit); torch.nn.utils.clip_grad_norm_(trainable,args.gradient_clip)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        step += 1
        save_checkpoint(out/f"step_{step:08d}.pt",pipe.dit,optimizer,scheduler,step,metadata)
        print(json.dumps({"step":step,"sample_id":row["sample_id"],"loss":float(loss.detach())*args.gradient_accumulation}))


if __name__ == "__main__": main()
