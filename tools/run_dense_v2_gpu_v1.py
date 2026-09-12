#!/usr/bin/env python3
"""Dense-condition v2 GPU render driver (qxq tree, +120h scale contract).

v2 render contract (user-approved):
  - grid 416x240 (latent exactly 30x52), fov_x 106.26, near 4.0, far 3000.0,
    pitch_sign +1.0, max_triangles 0
  - capsule players: player_radius 14.0 / player_height 40.0 /
    player_radius_scale 0.70 / player_screen_y_offset_px -5.5 /
    occlusion_tolerance 40.0 / min_visible_pixels 85
    (angle-preserving: 48 * (416*240)/(320*176) = 85) / player_mask_mode capsule
  - renderer: tools/build_mesh_dense_condition_v0_droplog.py (production params
    + 6-gate drop meta); the old tools/build_mesh_dense_condition_v0.py is
    forbidden for v2.
  - backend bsp_faces_gpu -> geometry_backend_id must be bsp_faces_disp_gpu
  - storage: npz + meta json always; target_rgb.png / qa png sampled at
    --qa-sample-rate (default 1%, deterministic per clip_id).

Derived from zhengqi template
$MAN/dense_event50h_gpu_v0/run_dense_event50h_gpu_v0.py (not under version
control there; this driver brings the pipeline into the qxq repo).
"""
from __future__ import annotations

from runtime_paths import ASSET_ROOT, LINGBOT_ROOT
import argparse, hashlib, importlib.util, json, multiprocessing as mp, os, queue, sys, time, traceback
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
RENDER_MODULE_PATH = REPO_ROOT / "tools" / "build_mesh_dense_condition_v0_droplog.py"
DEFAULT_TOOLS_DIR = REPO_ROOT / "tools"
# qxq tree has no docs/assets/bsp_static_geometry_v0/de_dust2_bsp_faces_v0.npz
# (checked 2026-07-23; only an output-dir copy exists), so default to the
# zhengqi tree asset, read-only.
DEFAULT_BSP_FACES_NPZ = ASSET_ROOT / "geometry/de_dust2_bsp_faces_v0.npz"

CHANNELS=["env_depth_norm_from_bsp_faces","env_mesh_hit_mask","env_nav_place_semantic_from_static_memory","other_player_mask_from_memory_player_capsules","other_player_depth_norm_from_memory_capsules","other_player_yaw_sin_relative_to_ego_from_memory","other_player_yaw_cos_relative_to_ego_from_memory"]

# v2 contract values, passed explicitly on every render call (no reliance on
# renderer-module defaults).
CAPSULE_PARAMS: dict[str, Any] = {
    "near": 4.0,
    "pitch_sign": 1.0,
    "max_triangles": 0,
    "player_radius": 14.0,
    "player_height": 40.0,
    "player_radius_scale": 0.70,
    "player_screen_y_offset_px": -5.5,
    "occlusion_tolerance": 40.0,
    "min_visible_pixels": 85,
    "player_z_offset": 0.0,
    "camera_yaw_offset": 0.0,
    "camera_pitch_offset": 0.0,
    "player_mask_mode": "capsule",
}

def render_frames(row: dict) -> list[int]:
    """21-frame stride-8 latent grid for one clip. raw_start+8*i is the canonical
    grid; raw_indices[::4] is the equivalent fallback (stride-2 list, every 4th).
    positive_latent_frames is NOT a render list on tier manifests (it is the
    visible-frame subset) and is only used as a last resort."""
    rs = row.get("raw_start")
    if rs is not None:
        return [int(rs) + 8 * i for i in range(21)]
    ri = list(row.get("raw_indices") or [])
    if len(ri) >= 81:
        return [int(x) for x in ri[::4][:21]]
    return [int(x) for x in (row.get("positive_latent_frames") or [])][:21]


def expected_backend_id(mesh_backend: str) -> str:
    if mesh_backend == "bsp_faces_gpu":
        return "bsp_faces_disp_gpu"
    if mesh_backend == "bsp_faces_cpu":
        return "bsp_faces_disp_cpu"
    raise ValueError(f"unsupported mesh_backend {mesh_backend!r}")

def import_tool(path: Path, name: str):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(mod); return mod

def sha256_file(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()

def load_rows(path: Path, limit: int|None):
    rows=[]
    with path.open() as f:
        for i,line in enumerate(f):
            if not line.strip(): continue
            d=json.loads(line); d["_source_line_index"]=i; rows.append(d)
            if limit and len(rows)>=limit: break
    return rows

def qa_selected(clip_id: Any, qa_sample_rate: float) -> bool:
    """Deterministic per-clip QA sampling: stable sha256 bucket in [0,100).

    Python's builtin hash() is salted per process, so a stable digest is used
    instead to keep the selection identical across runs/workers/resume scans.
    """
    bucket=int.from_bytes(hashlib.sha256(str(clip_id).encode("utf-8")).digest()[:8],"big")%100
    return bucket < int(round(qa_sample_rate*100))

def render_clip(row: dict[str,Any], out_dir: Path, tools_dir: Path, bsp: Path, width:int, height:int, fov_x:float, far:float, qa_sample_rate:float, mesh_backend:str) -> dict[str,Any]:
    dense_mod=import_tool(RENDER_MODULE_PATH, f"dense_mod_worker_{os.getpid()}")
    match_dir=Path(row["game_manifest"]).parents[2]
    cache=dense_mod.load_renderer_cache(match_dir, tools_dir, bsp_faces_npz=bsp)
    clip_id=row["clip_id"]; episode=row["episode"]; ego=row["player_stem"]
    qa_keep=qa_selected(clip_id, qa_sample_rate)
    want_backend=expected_backend_id(mesh_backend)
    frames=render_frames(row)
    clip_rec={"clip_id":clip_id,"source_line_index":row.get("_source_line_index"),"sample_count":0,"failure_count":0,"failures":[],"elapsed_s":0.0,"bytes":0,"samples":[],"qa_sampled":qa_keep}
    t_clip=time.perf_counter()
    for frame in frames:
        sample_id=f"{row['game_id']}__{episode}_{ego}_f{int(frame):06d}"
        sdir=out_dir/"samples"/sample_id
        sdir.mkdir(parents=True, exist_ok=True)
        t0=time.perf_counter()
        try:
            dense, depth, rgb, meta, qa_mod = dense_mod.render_memory_dense_condition(match_dir, episode, ego, int(frame), width, height, fov_x, far=far, cache=cache, mesh_backend=mesh_backend, **CAPSULE_PARAMS)
            if list(dense.shape)!=[7,height,width]:
                raise RuntimeError(f"bad dense shape {dense.shape}")
            if meta.get("geometry_backend_id")!=want_backend:
                raise RuntimeError(f"bad backend {meta.get('geometry_backend_id')} (want {want_backend})")
            target_path=sdir/"target_rgb.png"; qa_path=sdir/"mesh_dense_condition_qa_v0.png"
            if qa_keep:
                target=Image.fromarray(rgb).resize((width,height), Image.Resampling.BILINEAR)
                target.save(target_path)
                meta["target_rgb_path"]=str(target_path)
            else:
                meta["target_rgb_path"]=None
            npz=sdir/"mesh_dense_condition_v0.npz"; np.savez_compressed(npz,dense=dense,mesh_depth_units=depth)
            meta["qa_sampled"]=qa_keep
            meta["target_policy"]="v2 GPU dense: npz+meta always stored; current-frame RGB target and qa png stored only for QA-sampled clips (deterministic sha256 bucket of clip_id)."
            meta_path=sdir/"mesh_dense_condition_meta_v0.json"; meta_path.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
            if qa_keep:
                qa_mod.make_qa(qa_path,rgb,dense[0],dense[1],dense[3:],meta)
            stored=[npz,meta_path]+([target_path,qa_path] if qa_keep else [])
            size=sum(p.stat().st_size for p in stored if p.exists())
            clip_rec["bytes"]+=size; clip_rec["sample_count"]+=1
            clip_rec["samples"].append({"sample_id":sample_id,"frame_index":int(frame),"dense_path":str(npz),"target_rgb_path":str(target_path) if qa_keep else None,"meta_path":str(meta_path),"qa_path":str(qa_path) if qa_keep else None,"qa_sampled":qa_keep,"render_s":time.perf_counter()-t0,"mesh_hit_ratio":meta.get("mesh_hit_ratio"),"nav_semantic_hit_ratio":meta.get("nav_semantic_hit_ratio"),"other_player_memory_pixels":int((dense[3]>0.5).sum()),"geometry_backend_id":meta.get("geometry_backend_id"),"mesh_backend":meta.get("mesh_backend"),"shape":list(dense.shape)})
        except Exception as e:
            clip_rec["failure_count"]+=1
            clip_rec["failures"].append({"sample_id":sample_id,"frame_index":int(frame),"error":repr(e),"traceback":traceback.format_exc()[-4000:]})
    clip_rec["elapsed_s"]=time.perf_counter()-t_clip
    return clip_rec

def expected_frames(row: dict[str,Any]) -> list[int]:
    return [int(x) for x in render_frames(row)]

def sample_paths(row: dict[str,Any], out_dir: Path, frame: int) -> dict[str,Path]:
    sample_id=f"{row['game_id']}__{row['episode']}_{row['player_stem']}_f{int(frame):06d}"
    sdir=out_dir/"samples"/sample_id
    return {
        "sample_id": sample_id,
        "sdir": sdir,
        "target_rgb_path": sdir/"target_rgb.png",
        "dense_path": sdir/"mesh_dense_condition_v0.npz",
        "meta_path": sdir/"mesh_dense_condition_meta_v0.json",
        "qa_path": sdir/"mesh_dense_condition_qa_v0.png",
    }

def sample_complete(paths: dict[str,Path], height:int, width:int, require_qa:bool) -> bool:
    """A sample is complete only if npz+meta exist AND the npz really loads with
    dense.shape == (7, height, width). QA pngs are required only for QA-sampled
    clips. Any load error or shape mismatch means incomplete -> re-render, so a
    mixed-resolution output dir can never silently keep stale products."""
    required=["dense_path","meta_path"]+(["target_rgb_path","qa_path"] if require_qa else [])
    for key in required:
        p=paths[key]
        if not p.exists() or p.stat().st_size <= 0:
            return False
    try:
        with np.load(paths["dense_path"]) as z:
            if tuple(z["dense"].shape)!=(7,height,width):
                return False
    except Exception:
        return False
    return True

def clip_complete(row: dict[str,Any], out_dir: Path, height:int, width:int, qa_sample_rate:float) -> bool:
    frames=expected_frames(row)
    require_qa=qa_selected(row["clip_id"], qa_sample_rate)
    return bool(frames) and all(sample_complete(sample_paths(row,out_dir,frame),height,width,require_qa) for frame in frames)

def existing_clip_record(row: dict[str,Any], out_dir: Path, width:int, height:int, qa_sample_rate:float) -> dict[str,Any]:
    qa_keep=qa_selected(row["clip_id"], qa_sample_rate)
    clip_rec={"clip_id":row["clip_id"],"source_line_index":row.get("_source_line_index"),"sample_count":0,"failure_count":0,"failures":[],"elapsed_s":0.0,"bytes":0,"samples":[],"skipped_existing":True,"qa_sampled":qa_keep}
    for frame in expected_frames(row):
        paths=sample_paths(row,out_dir,frame)
        meta={}
        try:
            meta=json.loads(paths["meta_path"].read_text(encoding="utf-8"))
        except Exception:
            meta={}
        player_pixels=0; shape=None
        try:
            with np.load(paths["dense_path"]) as z:
                dense=z["dense"]
                shape=list(dense.shape)
                player_pixels=int((dense[3]>0.5).sum())
        except Exception:
            pass
        size=sum(paths[k].stat().st_size for k in ["target_rgb_path","dense_path","meta_path","qa_path"] if paths[k].exists())
        clip_rec["bytes"]+=size; clip_rec["sample_count"]+=1
        clip_rec["samples"].append({"sample_id":paths["sample_id"],"frame_index":int(frame),"dense_path":str(paths["dense_path"]),"target_rgb_path":str(paths["target_rgb_path"]) if paths["target_rgb_path"].exists() else None,"meta_path":str(paths["meta_path"]),"qa_path":str(paths["qa_path"]) if paths["qa_path"].exists() else None,"qa_sampled":qa_keep,"render_s":0.0,"mesh_hit_ratio":meta.get("mesh_hit_ratio"),"nav_semantic_hit_ratio":meta.get("nav_semantic_hit_ratio"),"other_player_memory_pixels":player_pixels,"geometry_backend_id":meta.get("geometry_backend_id"),"mesh_backend":meta.get("mesh_backend"),"shape":shape,"skipped_existing":True})
    return clip_rec

def clip_timeout_record(row: dict[str,Any], timeout_s: float, elapsed_s: float) -> dict[str,Any]:
    return {"clip_id":row["clip_id"],"source_line_index":row.get("_source_line_index"),"sample_count":0,"failure_count":1,"failures":[{"clip_id":row["clip_id"],"source_line_index":row.get("_source_line_index"),"error":f"clip_timeout>{timeout_s:.1f}s","traceback":""}],"elapsed_s":elapsed_s,"bytes":0,"samples":[],"timeout":True}

def clip_process_entry(q, row: dict[str,Any], out_dir: Path, tools_dir: Path, bsp: Path, width:int, height:int, fov_x:float, far:float, qa_sample_rate:float, mesh_backend:str):
    try:
        q.put(("ok", render_clip(row,out_dir,tools_dir,bsp,width,height,fov_x,far,qa_sample_rate,mesh_backend)))
    except Exception:
        q.put(("err", {"clip_id":row.get("clip_id"),"source_line_index":row.get("_source_line_index"),"sample_count":0,"failure_count":1,"failures":[{"clip_id":row.get("clip_id"),"source_line_index":row.get("_source_line_index"),"error":"worker_crash","traceback":traceback.format_exc()[-4000:]}],"elapsed_s":0.0,"bytes":0,"samples":[]}))

def percentiles(vals):
    vals=[v for v in vals if v is not None]
    if not vals: return {}
    arr=np.asarray(vals,dtype=np.float64)
    return {str(p):float(np.percentile(arr,p)) for p in [0,10,25,50,75,90,100]}

def parity_check(gpu_manifest: list[dict[str,Any]], cpu_dir: Path, max_pairs:int):
    out=[]
    by_id={s["sample_id"]:s for c in gpu_manifest for s in c.get("samples",[])}
    for sample_id,s in list(by_id.items())[:max_pairs*3]:
        cpu_npz=cpu_dir/"samples"/sample_id/"mesh_dense_condition_v0.npz"
        if not cpu_npz.exists(): continue
        gpu=np.load(s["dense_path"])["dense"].astype(np.float32)
        cpu=np.load(cpu_npz)["dense"].astype(np.float32)
        diff=np.abs(cpu-gpu)
        out.append({"sample_id":sample_id,"same_shape":list(cpu.shape)==list(gpu.shape),"dense_mean_abs_diff":float(diff.mean()),"dense_max_abs_diff":float(diff.max()),"mesh_hit_iou":float(((cpu[1]>0.5)&(gpu[1]>0.5)).sum()/max(1,((cpu[1]>0.5)|(gpu[1]>0.5)).sum())),"nav_iou":float(((cpu[2]>0.5)&(gpu[2]>0.5)).sum()/max(1,((cpu[2]>0.5)|(gpu[2]>0.5)).sum())),"player_mask_iou":float(((cpu[3]>0.5)&(gpu[3]>0.5)).sum()/max(1,((cpu[3]>0.5)|(gpu[3]>0.5)).sum()))})
        if len(out)>=max_pairs: break
    return out

def expected_samples_from_rows(rows):
    total = 0
    per_clip = []
    for r in rows:
        frames = render_frames(r)
        n = min(len(frames), 21)
        per_clip.append({"clip_id": r.get("clip_id"), "source_line_index": r.get("_source_line_index"), "expected_samples": n})
        total += n
    return total, per_clip

def run_clip_pool(rows, out, tools_dir, bsp, args, progress):
    ctx=mp.get_context("spawn")
    pending=list(rows); active=[]; done=0; manifest=[]; failures=[]; total=len(rows)
    with progress.open("a",encoding="utf-8") as pf:
        while pending or active:
            while pending and len(active)<args.workers:
                row=pending.pop(0); q=ctx.Queue()
                p=ctx.Process(target=clip_process_entry,args=(q,row,out,tools_dir,bsp,args.width,args.height,args.fov_x,args.far,args.qa_sample_rate,args.mesh_backend))
                p.start(); active.append({"row":row,"queue":q,"process":p,"start":time.perf_counter()})
            next_active=[]
            for item in active:
                p=item["process"]; q=item["queue"]; row=item["row"]; start=item["start"]; rec=None
                try:
                    _status, rec=q.get_nowait()
                except queue.Empty:
                    rec=None
                if rec is None and not p.is_alive():
                    try:
                        _status, rec=q.get_nowait()
                    except queue.Empty:
                        rec={"clip_id":row.get("clip_id"),"source_line_index":row.get("_source_line_index"),"sample_count":0,"failure_count":1,"failures":[{"clip_id":row.get("clip_id"),"source_line_index":row.get("_source_line_index"),"error":f"worker_exitcode={p.exitcode}","traceback":""}],"elapsed_s":time.perf_counter()-start,"bytes":0,"samples":[]}
                if rec is None and time.perf_counter()-start > args.clip_timeout_s:
                    p.terminate(); p.join(10)
                    if p.is_alive():
                        p.kill(); p.join(10)
                    rec=clip_timeout_record(row,args.clip_timeout_s,time.perf_counter()-start)
                if rec is None:
                    next_active.append(item); continue
                p.join(2)
                if p.is_alive():
                    p.terminate(); p.join(10)
                    if p.is_alive():
                        p.kill(); p.join(10)
                    rec["worker_killed_after_result"]=True
                q.close(); q.join_thread()
                done+=1; manifest.append(rec); failures.extend(rec.get("failures",[]))
                pf.write(json.dumps({"done":done,"total":total,**{k:rec.get(k) for k in ["clip_id","sample_count","failure_count","elapsed_s","bytes"]},"qa_sampled":bool(rec.get("qa_sampled")),"timeout":bool(rec.get("timeout")),"skipped_existing":bool(rec.get("skipped_existing"))},ensure_ascii=False)+"\n"); pf.flush()
                if done%10==0 or done==total: print(json.dumps({"event":"dense_v2_gpu_progress","clips_done":done,"clips_total":total,"failures":len(failures)},ensure_ascii=False),flush=True)
            active=next_active
            if active:
                time.sleep(0.5)
    return manifest, failures

def main():
    ap=argparse.ArgumentParser(description="v2 dense-condition GPU render driver (416x240, capsule contract, droplog renderer)")
    ap.add_argument("--source-manifest",type=Path,required=True); ap.add_argument("--out-dir",type=Path,required=True); ap.add_argument("--cpu-reference-dir",type=Path,default=None)
    ap.add_argument("--clip-limit",type=int,default=0,help="0 = no limit (render every manifest row)")
    ap.add_argument("--workers",type=int,default=4); ap.add_argument("--width",type=int,default=416); ap.add_argument("--height",type=int,default=240); ap.add_argument("--fov-x",type=float,default=106.26); ap.add_argument("--far",type=float,default=3000.0); ap.add_argument("--parity-pairs",type=int,default=20); ap.add_argument("--clip-timeout-s",type=float,default=120.0); ap.add_argument("--skip-existing",action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument("--qa-sample-rate",type=float,default=0.01,help="fraction of clips that also store target_rgb.png + qa png (deterministic per clip_id)")
    ap.add_argument("--mesh-backend",choices=["bsp_faces_gpu","bsp_faces_cpu"],default="bsp_faces_gpu")
    ap.add_argument("--bsp-faces-npz",type=Path,default=DEFAULT_BSP_FACES_NPZ)
    args=ap.parse_args()
    if not RENDER_MODULE_PATH.exists():
        raise SystemExit(f"render module missing: {RENDER_MODULE_PATH}")
    bsp=args.bsp_faces_npz
    if not bsp.exists():
        raise SystemExit(f"bsp faces npz missing: {bsp}")
    tools_dir=DEFAULT_TOOLS_DIR
    out=args.out_dir; (out/"samples").mkdir(parents=True,exist_ok=True)
    rows=load_rows(args.source_manifest,args.clip_limit)
    expected_samples, expected_samples_by_clip = expected_samples_from_rows(rows)
    completed_rows=[]; todo_rows=[]
    for i,r in enumerate(rows):
        if args.skip_existing and clip_complete(r,out,args.height,args.width,args.qa_sample_rate):
            completed_rows.append(r)
        else:
            todo_rows.append(r)
        if (i+1)%500==0:
            print(json.dumps({"event":"resume_scan_progress","scanned":i+1,"total":len(rows),"complete":len(completed_rows)},ensure_ascii=False),flush=True)
    want_backend=expected_backend_id(args.mesh_backend)
    command={"argv":sys.argv,"cwd":str(Path.cwd()),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"started_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"source_manifest":str(args.source_manifest),"out_dir":str(out),"workers":args.workers,"clip_count":len(rows),"clip_limit":args.clip_limit,"skip_existing":args.skip_existing,"skipped_existing_count":len(completed_rows),"todo_clip_count":len(todo_rows),"clip_timeout_s":args.clip_timeout_s,"render_module_path":str(RENDER_MODULE_PATH),"render_module_sha256":sha256_file(RENDER_MODULE_PATH),"bsp_faces_npz":str(bsp),"bsp_faces_npz_sha256":sha256_file(bsp),"tools_dir":str(tools_dir),"width":args.width,"height":args.height,"fov_x":args.fov_x,"far":args.far,"capsule_params":CAPSULE_PARAMS,"qa_sample_rate":args.qa_sample_rate,"mesh_backend":args.mesh_backend,"geometry_backend_id_required":want_backend,"channels":CHANNELS}
    (out/f"command_dense_v2_gpu_v1_resume_{os.getpid()}.json").write_text(json.dumps(command,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"event":"resume_scan","clip_count":len(rows),"skipped_existing":len(completed_rows),"todo":len(todo_rows),"clip_timeout_s":args.clip_timeout_s,"qa_sample_rate":args.qa_sample_rate,"mesh_backend":args.mesh_backend},ensure_ascii=False),flush=True)
    progress=out/"progress.jsonl"; t_all=time.perf_counter()
    manifest=[existing_clip_record(r,out,args.width,args.height,args.qa_sample_rate) for r in completed_rows]; failures=[]
    new_manifest, failures = run_clip_pool(todo_rows,out,tools_dir,bsp,args,progress)
    manifest.extend(new_manifest)
    manifest=sorted(manifest,key=lambda r:r.get("source_line_index",0))
    manifest_path=out/"manifest.json"; manifest_path.write_text(json.dumps({"samples":[s for c in manifest for s in c.get("samples",[])]},ensure_ascii=False,indent=2),encoding="utf-8")
    flat=[s for c in manifest for s in c.get("samples",[])]
    mesh=[s["mesh_hit_ratio"] for s in flat]; nav=[s["nav_semantic_hit_ratio"] for s in flat]; pix=[s["other_player_memory_pixels"] for s in flat]
    parity=parity_check(manifest,args.cpu_reference_dir,args.parity_pairs) if args.cpu_reference_dir else []
    report={"kind":"dense_v2_gpu_v1_report","status":"pass" if len(failures)==0 and len(flat)==expected_samples and all(s.get("geometry_backend_id")==want_backend and s.get("shape")==[7,args.height,args.width] for s in flat) else "fail","out_dir":str(out.resolve()),"source_manifest":str(args.source_manifest.resolve()),"workers":args.workers,"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"clip_count":len(rows),"expected_samples":expected_samples,"expected_samples_by_clip_head":expected_samples_by_clip[:10],"sample_count":len(flat),"failure_count":len(failures),"failure_examples":failures[:5],"elapsed_s":time.perf_counter()-t_all,"clips_per_s":len(rows)/max(1e-9,time.perf_counter()-t_all),"samples_per_s":len(flat)/max(1e-9,time.perf_counter()-t_all),"mesh_hit_ratio_percentiles":percentiles(mesh),"nav_semantic_hit_ratio_percentiles":percentiles(nav),"other_player_memory_pixels_percentiles":percentiles(pix),"backend_ids":sorted(set(str(s.get("geometry_backend_id")) for s in flat)),"geometry_backend_id_required":want_backend,"shape_required":[7,args.height,args.width],"shapes":sorted(set(tuple(s.get("shape")) if s.get("shape") else ("missing",) for s in flat),key=str),"qa_sample_rate":args.qa_sample_rate,"qa_sampled_clips":sum(1 for c in manifest if c.get("qa_sampled")),"qa_sampled_samples":sum(1 for s in flat if s.get("qa_sampled")),"render_module_path":str(RENDER_MODULE_PATH),"render_module_sha256":command["render_module_sha256"],"bsp_faces_npz":str(bsp),"bsp_faces_npz_sha256":command["bsp_faces_npz_sha256"],"capsule_params":CAPSULE_PARAMS,"manifest":str(manifest_path.resolve()),"progress_jsonl":str(progress.resolve()),"manifest_sha256":sha256_file(manifest_path),"parity_cpu_reference_dir":str(args.cpu_reference_dir) if args.cpu_reference_dir else None,"parity_pairs":parity,"parity_summary":{}}
    if parity:
        for k in ["dense_mean_abs_diff","mesh_hit_iou","nav_iou","player_mask_iou"]:
            report["parity_summary"][k+"_mean"]=float(np.mean([p[k] for p in parity]))
            report["parity_summary"][k+"_min"]=float(np.min([p[k] for p in parity]))
    report_path=out/"dense_v2_gpu_report_v1.json"; report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    report["report_sha256"]=sha256_file(report_path)
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    raise SystemExit(0 if report["status"]=="pass" else 2)
if __name__=="__main__": main()
