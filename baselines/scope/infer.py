from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import torch
from PIL import Image

from baselines.scope.scope_bridge.dataset import ScopeManifestDataset
from baselines.scope.scope_bridge.model_loader import init_official_pipeline, official_checkpoint_fingerprint
from baselines.scope.scope_bridge.schema import manifest_sha256, sha256_file
from baselines.scope.actions.validate_scope_parquet import validate as validate_action_parquet


def read_initial(row: dict) -> Image.Image:
    if "initial_image" in row: return Image.open(row["initial_image"]).convert("RGB")
    cmd=["ffmpeg","-v","error","-ss",str(row["raw_start"]),"-i",row["raw_video"],"-frames:v","1","-f","image2pipe","-vcodec","png","-"]
    import io
    cmd[4] = str(float(row["raw_start"]) / float(row["source_fps"]))
    return Image.open(io.BytesIO(subprocess.check_output(cmd))).convert("RGB")


def load_ft(path: str, dit) -> str:
    payload=torch.load(path,map_location="cpu",weights_only=False); state=payload.get("trainable_state")
    if not state: raise ValueError("not a scope_actionmodule_ft_v1 checkpoint")
    result=dit.load_state_dict(state,strict=False)
    if result.unexpected_keys: raise ValueError(f"unexpected fine-tuned keys: {result.unexpected_keys}")
    return sha256_file(path)


def verify_mp4(path: Path, frames: int, fps: int) -> dict:
    raw=subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-count_frames","-show_entries","stream=width,height,r_frame_rate,nb_read_frames","-of","json",str(path)],text=True)
    stream=json.loads(raw)["streams"][0]; num,den=map(int,stream["r_frame_rate"].split("/")); actual_fps=num/den
    actual=(int(stream["nb_read_frames"]),int(stream["width"]),int(stream["height"]),actual_fps)
    expected=(frames,832,480,float(fps))
    if actual != expected: raise RuntimeError(f"output contract mismatch: actual={actual}, expected={expected}")
    return {"ffprobe_frames":actual[0],"ffprobe_width":actual[1],"ffprobe_height":actual[2],"ffprobe_fps":actual[3]}


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--scope-repo",required=True); p.add_argument("--model-dir",required=True); p.add_argument("--output-dir",required=True)
    p.add_argument("--checkpoint"); p.add_argument("--sample-id",action="append"); p.add_argument("--limit",type=int); p.add_argument("--seed",type=int,action="append",default=[]); p.add_argument("--prompt",help="override manifest prompts for all selected samples")
    p.add_argument("--steps",type=int,default=30); p.add_argument("--cfg",type=float,default=5.0); p.add_argument("--mode",choices=("native81_20","candidate101_20"),default="native81_20")
    p.add_argument("--skip-existing",action="store_true"); p.add_argument("--dry-run",action="store_true"); return p.parse_args()


def main():
    args=parse_args(); ds=ScopeManifestDataset(args.manifest,"infer"); ids=set(args.sample_id or [])
    identities={s.sample_id:s.identity_projection() for s in ds.samples}
    rows=[r for r in ds if not ids or r["sample_id"] in ids][:args.limit]; expected=81 if args.mode=="native81_20" else 101
    for row in rows:
        if row["num_frames"] != expected: raise ValueError(f"{row['sample_id']}: mode expects {expected} action rows")
    if args.dry_run: print(json.dumps({"samples":[r["sample_id"] for r in rows],"mode":args.mode,"seeds":args.seed or [0],"authorized_fields":sorted(set().union(*(r.keys() for r in rows))) if rows else []},indent=2)); return
    pipe=init_official_pipeline(args.scope_repo,args.model_dir); checkpoint_hash=load_ft(args.checkpoint,pipe.dit) if args.checkpoint else official_checkpoint_fingerprint(args.model_dir)
    import sys
    sys.path.insert(0,args.scope_repo); from inference import load_actions, NEGATIVE_PROMPT
    from diffsynth.utils.data import save_video
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); mhash=manifest_sha256(args.manifest)
    for row in rows:
      prompt=args.prompt if args.prompt is not None else row["prompt"]
      action=row.get("scope_action_path")
      if not action: raise ValueError(f"{row['sample_id']}: inference requires converted scope_action_path")
      action_report=validate_action_parquet(action,expected)
      if not action_report["valid"]: raise ValueError(f"{row['sample_id']}: invalid action parquet: {action_report['errors']}")
      for seed in args.seed or [0]:
        stem=f"{row['sample_id']}__seed{seed}"; mp4=out/(stem+".mp4"); meta_path=out/(stem+".json")
        sidecar=Path(action).with_suffix(Path(action).suffix+".metadata.json")
        final_frames=81; final_fps=20 if args.mode=="native81_20" else 16
        meta={"sample_id":row["sample_id"],**identities.get(row["sample_id"],{}),"manifest_sha256":mhash,"input_sha256":sha256_file(row["initial_image"]) if row.get("initial_image") else sha256_file(row["raw_video"]),"action_sha256":sha256_file(action),"action_calibration_sha256":json.loads(sidecar.read_text()).get("mouse_calibration_sha256") if sidecar.is_file() else None,"checkpoint_sha256":checkpoint_hash,"model_mode":"S1" if args.checkpoint else "S0","prompt":prompt,"seed":seed,"cfg":args.cfg,"sampling_steps":args.steps,"generated_fps":20,"generated_frames":expected,"output_fps":final_fps,"output_frames":final_frames,"height":480,"width":832,"time_resampled":args.mode!="native81_20"}
        if mp4.exists() or meta_path.exists():
            existing=json.loads(meta_path.read_text()) if meta_path.exists() else {}
            if args.skip_existing and mp4.exists() and meta_path.exists() and all(existing.get(k)==v for k,v in meta.items()): continue
            raise FileExistsError(f"refusing to overwrite mismatched/incomplete output: {stem}")
        keyboard,mouse=load_actions(action,expected); image=read_initial(row); started=time.time()
        video=pipe(prompt=prompt,negative_prompt=NEGATIVE_PROMPT,input_image=image,num_frames=expected,num_inference_steps=args.steps,height=480,width=832,seed=seed,cfg_scale=args.cfg,tiled=True,keyboard_action=keyboard,mouse_action=mouse)
        if args.mode=="native81_20": save_video(video,str(mp4),fps=20,quality=5)
        else:
            source_path=out/(stem+".source101.mp4"); save_video(video,str(source_path),fps=20,quality=5)
            subprocess.run(["ffmpeg","-v","error","-i",str(source_path),"-vf","scale=832:480,fps=16","-frames:v","81","-c:v","libx264","-pix_fmt","yuv420p",str(mp4)],check=True)
            source_path.unlink()
        meta.update(verify_mp4(mp4,final_frames,final_fps)); meta.update({"runtime_seconds":time.time()-started,"torch":torch.__version__,"python":platform.python_version(),"gpu":torch.cuda.get_device_name() if torch.cuda.is_available() else None})
        meta_path.write_text(json.dumps(meta,indent=2,sort_keys=True)+"\n")


if __name__ == "__main__": main()
