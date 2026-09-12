#!/usr/bin/env python3
from __future__ import annotations
from runtime_paths import source_path

import argparse, json, math, os, random, sys, time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

ROOT = Path(str(source_path('scene', '')))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_OUT_DIR = ROOT / "output/symmvc_scene_covis_pairs_v3"
DEFAULT_HELDOUT = ROOT / "output/symmvc_strong_pairs_heldout_v1/heldout_match_split_v1.json"
DEFAULT_MANIFESTS = [
    ROOT / "output/memory_v2_all188_diversity_stride_manifest_20260629_v0/step5_materialized_manifest_dlc_v0.jsonl",
    ROOT / "output/memory_v2_all188_diversity_stride_manifest_20260629_v0/v2_all188_positive_diversity_tier20h_source_manifest_v0.jsonl",
    ROOT / "output/memory_v2_all188_diversity_stride_manifest_20260629_v0/v2_all188_positive_diversity_tier69h_source_manifest_v0.jsonl",
]
DATA_ROOT = Path(str(source_path('assets', 'csgo-datasets-fullsubset/32f1644d4f42c29d')))
DEFAULT_MESH_DIR = DATA_ROOT / "00a145c4184c4cf092acb25d2521ccee"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def norm_path(p: str | None) -> str | None:
    if not p: return None
    return p.replace("/mnt/workspace/zhengqi/", "/mnt/data/pku/zhengqi/")


def file_info(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {"path": str(path), "size": st.st_size, "mtime_unix": st.st_mtime, "mtime_local": time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(st.st_mtime))}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_clip_rows(manifests: list[Path], heldout: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    stats=[]
    for path in manifests:
        n=kept=with_req=0
        if not path.exists():
            stats.append({"path": str(path), "exists": False}); continue
        for r in read_jsonl(path):
            n += 1
            mid = str(r.get("game_id") or r.get("match_id"))
            if mid not in heldout: continue
            cdir = norm_path(r.get("clip_dir") or r.get("sample_dir"))
            if not cdir: continue
            req = all((Path(cdir)/fn).exists() for fn in ["video.mp4","poses.npy","intrinsics.npy","meta.json"])
            if not req: continue
            with_req += 1
            rr = dict(r); rr["clip_dir"] = cdir
            cid = str(rr.get("clip_id"))
            old = by_id.get(cid)
            if old is None or "tier69h_supplement" in cdir:
                by_id[cid] = rr
                kept += 1
        stats.append({"path": str(path), "exists": True, "rows": n, "heldout_rows_with_required": with_req, **file_info(path)})
    return list(by_id.values()), {"manifest_stats": stats, "unique_heldout_clips_with_required": len(by_id)}


def raw_map(row: dict[str, Any]) -> dict[int,int]:
    meta = load_json(Path(row["clip_dir"])/"meta.json")
    raws = meta.get("raw_indices") or row.get("raw_indices") or []
    return {int(r): i for i,r in enumerate(raws)}


def forward_from_pose(c2w: np.ndarray) -> np.ndarray:
    f = c2w[:3,2].astype(np.float64)
    n = np.linalg.norm(f)
    return f / max(n, 1e-9)


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a,b)), -1, 1))))


def camera_basis(yaw_deg: float, pitch_deg: float, pitch_sign: float=1.0):
    yaw = math.radians(yaw_deg); pitch = math.radians(pitch_deg) * pitch_sign
    forward = np.array([math.cos(pitch)*math.cos(yaw), math.cos(pitch)*math.sin(yaw), -math.sin(pitch)], dtype=np.float32)
    forward /= max(float(np.linalg.norm(forward)), 1e-6)
    world_up = np.array([0.0,0.0,1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6: right = np.array([0.0,-1.0,0.0], dtype=np.float32)
    right /= max(float(np.linalg.norm(right)), 1e-6)
    up = np.cross(right, forward); up /= max(float(np.linalg.norm(up)), 1e-6)
    return right, up, forward


def world_to_camera(points: np.ndarray, cam_pos: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    right, up, forward = camera_basis(yaw, pitch, 1.0)
    rel = points - cam_pos.reshape(1,3)
    return np.stack([rel @ right, rel @ up, rel @ forward], axis=1).astype(np.float32)


def project_camera_points(cam: np.ndarray, width: int, height: int, fov_x: float) -> np.ndarray:
    tan_x = math.tan(math.radians(fov_x)/2.0); tan_y = tan_x * height / width
    z = np.maximum(cam[:,2], 1e-6)
    u = width*0.5 + (cam[:,0]/z)/tan_x*width*0.5
    v = height*0.5 - (cam[:,1]/z)/tan_y*height*0.5
    return np.stack([u,v,cam[:,2]], axis=1).astype(np.float32)


def load_obj_mesh(path: Path):
    verts=[]; faces=[]
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                p=line.split(); verts.append([float(p[1]),float(p[2]),float(p[3])])
            elif line.startswith("f "):
                idx=[int(tok.split("/")[0])-1 for tok in line.split()[1:]]
                for i in range(1,len(idx)-1): faces.append([idx[0],idx[i],idx[i+1]])
    return np.asarray(verts,np.float32), np.asarray(faces,np.int32)



def match_dir_for_row(row: dict[str, Any]) -> Path:
    for key in ["game_manifest", "action_json", "mp4", "video_manifest", "player_visibility"]:
        val = row.get(key)
        if not val:
            continue
        cur = Path(str(val))
        for parent in [cur.parent, *cur.parents]:
            if (parent / "mesh_manifest.json").exists():
                return parent
    mid = str(row.get("game_id") or row.get("match_id"))
    fallback = DATA_ROOT / mid
    if (fallback / "mesh_manifest.json").exists():
        return fallback
    return DEFAULT_MESH_DIR

def find_world_obj(match_dir: Path) -> Path:
    manifest = load_json(match_dir/"mesh_manifest.json")
    world = next((m for m in manifest if m.get("model_name") == "_world_"), None)
    if not world: raise FileNotFoundError(match_dir/"mesh_manifest.json")
    return match_dir/"meshes"/world["mesh_file"]



def mesh_point_cloud(vertices: np.ndarray, faces: np.ndarray, max_points: int = 60000) -> np.ndarray:
    tri = vertices[faces]
    centers = tri.mean(axis=1).astype(np.float32)
    if len(centers) <= max_points:
        return centers
    rng = np.random.default_rng(20260705)
    idx = rng.choice(len(centers), size=max_points, replace=False)
    return centers[idx]


def point_zbuffer(points: np.ndarray, cam_pos, yaw, pitch, width, height, fov_x, near, far):
    cam = world_to_camera(points, cam_pos, yaw, pitch)
    pr = project_camera_points(cam, width, height, fov_x)
    u = np.floor(pr[:,0]).astype(np.int32); v = np.floor(pr[:,1]).astype(np.int32); z = pr[:,2]
    inside = (z > near) & (z < far) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    idx_inside = np.where(inside)[0]
    if len(idx_inside):
        np.minimum.at(depth, (v[idx_inside], u[idx_inside]), z[idx_inside].astype(np.float32))
    return cam, pr, inside, depth, u, v, z


def visible_indices_from_zbuffer(inside: np.ndarray, depth: np.ndarray, u: np.ndarray, v: np.ndarray, z: np.ndarray, tol_abs: float, tol_rel: float) -> np.ndarray:
    idx = np.where(inside)[0]
    if len(idx) == 0:
        return idx
    dz = depth[v[idx], u[idx]]
    tol = np.maximum(tol_abs, tol_rel * z[idx])
    return idx[np.isfinite(dz) & (np.abs(dz - z[idx]) <= tol)]

def frame_record(action_path: str, raw: int) -> dict[str, Any] | None:
    arr = ACTION_CACHE.get(action_path)
    if arr is None:
        arr = load_json(Path(action_path)); ACTION_CACHE[action_path] = arr
    if 0 <= raw < len(arr): return arr[raw]
    return None


def cam_from_frame(fr: dict[str, Any]):
    pos = np.asarray(fr.get("camera_position") or [fr["x"], fr["y"], fr["z"] + 64.0], dtype=np.float32)
    yaw = float(fr.get("yaw", fr.get("camera_rotation", [0,0,0])[2]))
    pitch = float(fr.get("pitch", fr.get("camera_rotation", [0,0,0])[1]))
    return pos, yaw, pitch


def sample_visible_world_points(depth: np.ndarray, cam_pos, yaw, pitch, n: int, fov_x: float, seed: int):
    h,w = depth.shape
    ys,xs = np.where(np.isfinite(depth))
    if len(xs) == 0: return np.zeros((0,3), np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
    u = xs[idx].astype(np.float32) + 0.5; v = ys[idx].astype(np.float32) + 0.5; z = depth[ys[idx], xs[idx]].astype(np.float32)
    tan_x = math.tan(math.radians(fov_x)/2.0); tan_y = tan_x*h/w
    x = ((u - w*0.5)/(w*0.5))*tan_x*z
    y = -((v - h*0.5)/(h*0.5))*tan_y*z
    right, up, forward = camera_basis(yaw, pitch, 1.0)
    pts = cam_pos.reshape(1,3) + x[:,None]*right.reshape(1,3) + y[:,None]*up.reshape(1,3) + z[:,None]*forward.reshape(1,3)
    return pts.astype(np.float32)



def covis_rate_for_frame(row_a, row_b, li_a, li_b, raw, mesh_points, args) -> dict[str, Any]:
    fra = frame_record(row_a["action_json"], raw); frb = frame_record(row_b["action_json"], raw)
    if fra is None or frb is None:
        return {"raw": raw, "status": "missing_action_frame", "rate": 0.0}
    posa, yawa, pitcha = cam_from_frame(fra); posb, yawb, pitchb = cam_from_frame(frb)
    _, _, inside_a, depth_a, ua, va, za = point_zbuffer(mesh_points, posa, yawa, pitcha, args.width, args.height, args.fov_x, args.near, args.far)
    vis_a = visible_indices_from_zbuffer(inside_a, depth_a, ua, va, za, args.depth_tol_abs, args.depth_tol_rel)
    if len(vis_a) == 0:
        return {"raw": raw, "li": li_a, "status": "no_visible_A", "rate": 0.0, "A_visible_points": 0}
    if len(vis_a) > args.samples_per_frame:
        rng = np.random.default_rng(raw + 17 * li_a)
        vis_a = rng.choice(vis_a, size=args.samples_per_frame, replace=False)
    pts = mesh_points[vis_a]
    _, _, inside_b_all, depth_b, ub_all, vb_all, zb_all = point_zbuffer(mesh_points, posb, yawb, pitchb, args.width, args.height, args.fov_x, args.near, args.far)
    cam_b = world_to_camera(pts, posb, yawb, pitchb)
    pr_b = project_camera_points(cam_b, args.width, args.height, args.fov_x)
    ub = np.floor(pr_b[:,0]).astype(np.int32); vb = np.floor(pr_b[:,1]).astype(np.int32); zb = pr_b[:,2]
    inside_b = (zb > args.near) & (zb < args.far) & (ub >= 0) & (ub < args.width) & (vb >= 0) & (vb < args.height)
    visible = np.zeros(len(pts), dtype=bool)
    if inside_b.any():
        idx = np.where(inside_b)[0]
        dd = depth_b[vb[idx], ub[idx]]
        tol = np.maximum(args.depth_tol_abs, args.depth_tol_rel * zb[idx])
        visible[idx] = np.isfinite(dd) & (np.abs(dd - zb[idx]) <= tol)
    return {"raw": int(raw), "li": int(li_a), "status": "ok", "rate": round(float(visible.mean()), 6), "sampled": int(len(pts)), "inside_B": int(inside_b.sum()), "visible_B": int(visible.sum()), "A_visible_points_total": int(len(vis_a)), "B_visible_points_total": int(visible_indices_from_zbuffer(inside_b_all, depth_b, ub_all, vb_all, zb_all, args.depth_tol_abs, args.depth_tol_rel).size), "distance_units": round(float(np.linalg.norm(posa-posb)), 3)}

def percentile(vals, q):
    if not vals: return None
    return round(float(np.percentile(np.asarray(vals, dtype=np.float64), q)), 6)


ACTION_CACHE: dict[str, Any] = {}
MESH_CACHE: dict[str, Any] = {}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--heldout-split", type=Path, default=DEFAULT_HELDOUT)
    ap.add_argument("--manifest", type=Path, action="append", default=[])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--top-k", type=int, default=60)
    ap.add_argument("--min-exact-frames", type=int, default=8)
    ap.add_argument("--fov-x", type=float, default=106.26)
    ap.add_argument("--angle-max", type=float, default=90.0)
    ap.add_argument("--distance-max", type=float, default=2000.0)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=180)
    ap.add_argument("--near", type=float, default=4.0)
    ap.add_argument("--far", type=float, default=3000.0)
    ap.add_argument("--max-triangles", type=int, default=0)
    ap.add_argument("--samples-per-frame", type=int, default=220)
    ap.add_argument("--depth-tol-abs", type=float, default=18.0)
    ap.add_argument("--depth-tol-rel", type=float, default=0.035)
    ap.add_argument("--max-pairs", type=int, default=0, help="debug cap after proxy sort; 0 means all")
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    held_obj=load_json(args.heldout_split)
    heldout=set(held_obj.get("heldout_matches") or [m["match_id"] for m in held_obj.get("matches",[]) if m.get("split") in ("val","test")])
    rows, clip_report = load_clip_rows(args.manifest or DEFAULT_MANIFESTS, heldout)
    by_group=defaultdict(list)
    for r in rows: by_group[(str(r.get("game_id")), str(r.get("episode")))].append(r)

    proxy=[]; reject=Counter(); examined=0
    pose_cache={}
    raw_cache={}
    for group, gr in by_group.items():
        by_player=defaultdict(list)
        for r in gr: by_player[str(r.get("player_stem"))].append(r)
        stems=sorted(by_player)
        for i,sa in enumerate(stems):
            for sb in stems[i+1:]:
                for a in by_player[sa]:
                    a0,a1=int(a["frame_count_start"]),int(a["frame_count_end"])
                    mapa=raw_cache.setdefault(a["clip_id"], raw_map(a))
                    pa=pose_cache.setdefault(a["clip_id"], np.load(Path(a["clip_dir"])/"poses.npy").astype(np.float64))
                    for b in by_player[sb]:
                        b0,b1=int(b["frame_count_start"]),int(b["frame_count_end"])
                        if min(a1,b1) < max(a0,b0): continue
                        examined += 1
                        mapb=raw_cache.setdefault(b["clip_id"], raw_map(b))
                        raws=sorted(set(mapa)&set(mapb))
                        if len(raws) < args.min_exact_frames:
                            reject["below_exact_raw_overlap"] += 1; continue
                        pb=pose_cache.setdefault(b["clip_id"], np.load(Path(b["clip_dir"])/"poses.npy").astype(np.float64))
                        frame_metrics=[]
                        for raw in raws:
                            ia,ib=mapa[raw],mapb[raw]
                            ca,cb=pa[ia][:3,3],pb[ib][:3,3]
                            dist=float(np.linalg.norm(ca-cb))
                            ang=angle_deg(forward_from_pose(pa[ia]), forward_from_pose(pb[ib]))
                            if ang < args.angle_max and dist < args.distance_max:
                                frame_metrics.append((raw,ia,ib,ang,dist))
                        if len(frame_metrics) < args.min_exact_frames:
                            reject["below_proxy_frame_gate"] += 1; continue
                        med_ang=median([x[3] for x in frame_metrics]); med_dist=median([x[4] for x in frame_metrics])
                        proxy_score=(args.angle_max-med_ang)/args.angle_max + max(0.0,(args.distance_max-med_dist)/args.distance_max) + 0.01*len(frame_metrics)
                        proxy.append((proxy_score, med_ang, med_dist, len(frame_metrics), a, b, frame_metrics))
    proxy.sort(key=lambda x: (-x[0], x[1], x[2], -x[3], str(x[4].get("clip_id")), str(x[5].get("clip_id"))))
    if args.max_pairs and len(proxy) > args.max_pairs: proxy = proxy[:args.max_pairs]

    out_pairs=[]; eval_rows=[]
    for rank,(pscore, med_ang, med_dist, nproxy, a,b,fms) in enumerate(proxy,1):
        mid=str(a.get("game_id")); mdir = match_dir_for_row(a)
        mesh=MESH_CACHE.get(mid)
        if mesh is None:
            vertices, faces = load_obj_mesh(find_world_obj(mdir)); mesh=mesh_point_cloud(vertices, faces); MESH_CACHE[mid]=mesh
        # evaluate up to 8 evenly spread proxy-passing exact frames for speed and scorer compatibility
        if len(fms) > 8:
            idxs=np.linspace(0,len(fms)-1,8).round().astype(int).tolist(); use=[fms[i] for i in idxs]
        else:
            use=fms
        frs=[]
        for raw,ia,ib,ang,dist in use:
            rec=covis_rate_for_frame(a,b,ia,ib,raw,mesh,args); rec["angle_pose_deg"]=round(float(ang),3); rec["distance_units_pose"]=round(float(dist),3); frs.append(rec)
        rates=[float(x.get("rate") or 0.0) for x in frs if x.get("status")=="ok"]
        if not rates:
            reject["no_ok_geometry_frames"] += 1; continue
        med_rate=float(np.median(rates)); mean_rate=float(np.mean(rates))
        pair={
            "game": mid, "match_id": mid, "ep": a.get("episode"), "episode": a.get("episode"),
            "window": [max(int(a["frame_count_start"]), int(b["frame_count_start"])), min(int(a["frame_count_end"]), int(b["frame_count_end"]))],
            "egoA": a.get("player_stem"), "egoB": b.get("player_stem"),
            "clipA_id": a.get("clip_id"), "clipB_id": b.get("clip_id"), "clipA_dir": a.get("clip_dir"), "clipB_dir": b.get("clip_dir"),
            "idxA": None, "idxB": None,
            "entry_kind": "scene_covis_heldout_v3_exact_raw_mesh_depth",
            "n_mutual_covis_frames": len(use),
            "mutual_covis_frames": [[int(ia), int(raw), round(med_rate*100000.0,4), round(med_rate*100000.0,4)] for raw,ia,ib,ang,dist in use],
            "a_sees_b_maxpix": round(med_rate*100000.0,4), "b_sees_a_maxpix": round(med_rate*100000.0,4), "min_mutual_maxpix": round(med_rate*100000.0,4),
            "scene_covis_median": round(med_rate,6), "scene_covis_mean": round(mean_rate,6), "scene_covis_frame_rates": [round(x,6) for x in rates],
            "pose_proxy": {"median_view_angle_deg": round(float(med_ang),3), "median_distance_units": round(float(med_dist),3), "proxy_passing_exact_frames": int(nproxy), "proxy_score": round(float(pscore),6)},
            "scene_covis_policy_v3": {"mesh_source_match_dir": str(mdir), "fov_x_deg": args.fov_x, "angle_max_deg": args.angle_max, "distance_max_units": args.distance_max, "render_size": [args.width,args.height], "samples_per_frame": args.samples_per_frame, "depth_tol_abs": args.depth_tol_abs, "depth_tol_rel": args.depth_tol_rel, "scored_frames": len(use)},
            "source_manifest_A": {k:a.get(k) for k in ["mp4","action_json","raw_start","frame_count_start","frame_count_end","map_memory_split","map_memory_split_key"]},
            "source_manifest_B": {k:b.get(k) for k in ["mp4","action_json","raw_start","frame_count_start","frame_count_end","map_memory_split","map_memory_split_key"]},
        }
        out_pairs.append(pair)
        eval_rows.append({"rank_proxy": rank, "pair_id": str(a.get("clip_id")) + "__" + str(b.get("clip_id")), "match_id": mid, "episode": a.get("episode"), "scene_covis_median": round(med_rate,6), "scene_covis_mean": round(mean_rate,6), "pose_median_angle_deg": round(float(med_ang),3), "pose_median_distance_units": round(float(med_dist),3), "frames": frs})
        print("GEOM {}/{} kept={} med={:.4f} mean={:.4f} {} {} {}__{}".format(rank, len(proxy), len(out_pairs), med_rate, mean_rate, mid, a.get("episode"), a.get("player_stem"), b.get("player_stem")), flush=True)
    out_pairs.sort(key=lambda p: (-float(p["scene_covis_median"]), -float(p["scene_covis_mean"]), float(p["pose_proxy"]["median_view_angle_deg"]), float(p["pose_proxy"]["median_distance_units"]), str(p["clipA_id"])))
    selected=out_pairs[:args.top_k]
    pair_path=args.out_dir/"pair_index_verdict_ready_v3.json"
    win_path=args.out_dir/"window_manifest_scene_covis_v3.jsonl"
    report_path=args.out_dir/"scene_covis_v3_report.json"
    details_path=args.out_dir/"scene_covis_v3_frame_details.json"
    pair_path.write_text(json.dumps(selected, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
    with win_path.open("w", encoding="utf-8") as f:
        for p in selected:
            for side in ["A","B"]:
                f.write(json.dumps({"pair_id": str(p["clipA_id"]) + "__" + str(p["clipB_id"]), "side": side, "match_id": p["match_id"], "episode": p["episode"], "ego": p[f"ego{side}"], "clip_id": p[f"clip{side}_id"], "clip_dir": p[f"clip{side}_dir"], "exact_aligned_raw_frames": [int(x[1]) for x in p["mutual_covis_frames"]]}, ensure_ascii=False, separators=(",",":"))+"\n")
    details_path.write_text(json.dumps(eval_rows, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
    vals=[float(p["scene_covis_median"]) for p in out_pairs]
    report={"kind":"symmvc_scene_covis_pairs_v3", "created_at_unix": time.time(), "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S %z"), "params": vars(args) | {"manifest": [str(p) for p in (args.manifest or DEFAULT_MANIFESTS)], "heldout_split": str(args.heldout_split)}, "inputs": {"heldout_split": file_info(args.heldout_split), **clip_report}, "counts": {"heldout_match_count": len(heldout), "clips_loaded": len(rows), "episode_groups": len(by_group), "overlap_pairs_examined": examined, "proxy_candidates": len(proxy), "geometry_scored_pairs": len(out_pairs), "selected_pairs": len(selected), "reject_counts": dict(sorted(reject.items()))}, "scene_covis_distribution_all_scored": {"min": percentile(vals,0), "p10": percentile(vals,10), "p25": percentile(vals,25), "p50": percentile(vals,50), "p75": percentile(vals,75), "p90": percentile(vals,90), "max": percentile(vals,100)}, "outputs": {"pair_index": str(pair_path), "window_manifest": str(win_path), "frame_details": str(details_path), "report_json": str(report_path)}, "top_selected": [{"rank":i+1,"pair_id":str(p["clipA_id"]) + "__" + str(p["clipB_id"]),"match_id":p["match_id"],"episode":p["episode"],"egoA":p["egoA"],"egoB":p["egoB"],"scene_covis_median":p["scene_covis_median"],"scene_covis_mean":p["scene_covis_mean"],"pose_proxy":p["pose_proxy"]} for i,p in enumerate(selected)]}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
    print(f"WROTE {pair_path} selected={len(selected)} scored={len(out_pairs)} proxy={len(proxy)}")
    print(f"WROTE {report_path}")

if __name__ == "__main__":
    main()
