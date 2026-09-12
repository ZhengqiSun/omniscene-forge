#!/usr/bin/env python3
# QXQ height-aware variant of zhengqi tools/symmvc_v2_score_v0.py.
# Source: /mnt/data/pku/zhengqi/multiview-map-v0/tools/symmvc_v2_score_v0.py
# Source sha256: 9b4c65ebb82c80dfa9231fcef50f46c09904e0a2408a6c3e8926047a12a1b717
# Only behavioral change vs source: the per-video principal-point shift y_off is
# derived from the ACTUAL decoded frame height as y_off = (480 - h) / 2, instead
# of the source hardcoded 8.0 for every generated video (which silently assumed
# all generated mp4s are 464x832 legacy center-crops). For 464 inputs the result
# is bit-identical (y_off=8.0); for fixed480 832x480 inputs y_off becomes 0.0.
# Measured heights and applied offsets are recorded in each pair result for audit.
"""SymMVC v2 diagnostics: expanded frames, quality gates, paired stats, tail metrics."""
from __future__ import annotations
from runtime_paths import source_path
import argparse, glob, json, math, os, tempfile, time
from pathlib import Path
from statistics import median

import cv2
import numpy as np

EGO_ZONE = {"x0": 0.25, "x1": 0.75, "y0": 0.55, "y1": 1.0}


def log(msg: str) -> None:
    print(f"[symmvc_v2] {msg}", flush=True)


def w2c(c2w: np.ndarray) -> np.ndarray:
    r, t = c2w[:3, :3], c2w[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ t
    return out


def k_eff(intr: np.ndarray, y_off: float) -> np.ndarray:
    fx, fy, cx, cy = [float(x) for x in intr]
    return np.array([[fx, 0, cx], [0, fy, cy - y_off], [0, 0, 1]], dtype=np.float64)


def proj_mat(c2w: np.ndarray, k: np.ndarray) -> np.ndarray:
    return k @ w2c(c2w)[:3, :]


def read_frame(mp4: str, idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(mp4)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def stat(vals: list[float], threshold: float | None = None) -> dict:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "median": None, "p90": None, "mean": None, "over_threshold_frac": None}
    out = {
        "n": int(arr.size),
        "median": round(float(np.median(arr)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "mean": round(float(arr.mean()), 4),
    }
    out["over_threshold_frac"] = None if threshold is None else round(float((arr > threshold).mean()), 4)
    return out


def in_ego_zone_xy(x: float, y: float, w: int, h: int) -> bool:
    return EGO_ZONE["x0"] * w <= x <= EGO_ZONE["x1"] * w and EGO_ZONE["y0"] * h <= y <= EGO_ZONE["y1"] * h


def in_boxes(x: float, y: float, boxes: list[list[float]]) -> bool:
    return any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in boxes)


class Detector:
    def __init__(self, device: str, thr: float):
        import torch
        import torchvision
        self.torch = torch
        self.thr = thr
        self.device = device
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT").eval().to(device)

    def boxes(self, frame_bgr: np.ndarray) -> list[list[float]]:
        import torch
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(self.device)
        with torch.no_grad():
            out = self.model([t])[0]
        keep = (out["labels"] == 1) & (out["scores"] >= self.thr)
        return out["boxes"][keep].detach().cpu().numpy().astype(float).tolist()


def measure_height_y_off(mp4: str, declared: float) -> tuple[float, int]:
    """Height-aware y-offset: y_off = (480 - actual_height) / 2.

    Generated videos are either 832x480 (fixed480 chain) or 832x464 (legacy
    MAX_AREA floor bug, a center-crop of 480 so cy shifts by 8). Falls back to
    the resolver's declared value when the height cannot be read.
    """
    cap = cv2.VideoCapture(mp4)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if h <= 0:
        return declared, -1
    return (480.0 - float(h)) / 2.0, h


def resolve_gen_video(gen_root: Path, clip_id: str, variant: str) -> tuple[str | None, float]:
    if variant == "gt":
        return None, 0.0
    pats = [str(gen_root / variant / f"{clip_id}_{variant}_lf21_*.mp4"), str(gen_root / variant / f"{clip_id}_{variant}_*.mp4")]
    for pat in pats:
        g = sorted(glob.glob(pat))
        if g:
            return g[0], 8.0
    return None, 8.0


def resolve_symmvc_video(symmvc_root: Path, clip_id: str, variant: str) -> tuple[str | None, float]:
    if variant == "gt":
        return None, 0.0
    for split in ("val", "test"):
        g = sorted(glob.glob(str(symmvc_root / split / f"{clip_id}_{variant}_lf21_*.mp4")))
        if g:
            return g[0], 8.0
    return None, 8.0


def resolve_video(pair: dict, which: str, variant: str, args) -> tuple[str | None, float]:
    clip_dir = Path(pair[f"clip{which}_dir"])
    if variant == "gt":
        return str(clip_dir / "video.mp4"), 0.0
    clip_id = pair.get(f"clip{which}_id") or clip_dir.name
    mp4, yoff = resolve_gen_video(args.gen_root, clip_id, variant)
    if mp4:
        return mp4, yoff
    return resolve_symmvc_video(args.symmvc_view_root, clip_id, variant)


def dense_sample_keys(pair: dict, which: str, raw_frame: int) -> list[str]:
    game = pair.get("game") or pair.get("match_id")
    episode = pair.get("episode") or pair.get("ep")
    ego = pair.get(f"ego{which}")
    if not game or not ego:
        clip_id = pair.get(f"clip{which}_id", "")
        parts = clip_id.split("_")
        game = game or (parts[1] if len(parts) > 1 else "")
        # clip_id layout: dataset_game_Ep_NNNNN_Ep_NNNNN_team_..._frame
        episode = episode or "_".join(parts[2:4])
        ego = ego or "_".join(parts[4:-1])
    keys = []
    if game and episode and ego:
        keys.append(f"{game}__{episode}_{ego}_f{int(raw_frame):06d}")
    if game and ego:
        keys.append(f"{game}__{ego}_f{int(raw_frame):06d}")
    return keys


def build_dense_index(root: Path) -> dict[str, Path]:
    if not root or not root.exists():
        return {}
    return {p.parent.name: p for p in root.glob("*/mesh_dense_condition_v0.npz")}


def load_player_mask(dense_index: dict[str, Path], keys: list[str], frame_shape: tuple[int, int], args) -> tuple[np.ndarray | None, dict]:
    path = None
    key = None
    for cand in keys:
        if cand in dense_index:
            key = cand
            path = dense_index[cand]
            break
    if path is None:
        return None, {"status": "missing_dense_npz", "keys": keys}
    try:
        z = np.load(path)
        dense = z["dense"]
        mask_small = dense[3] > float(args.dense_player_threshold)
    except Exception as e:
        return None, {"status": f"dense_load_error: {type(e).__name__}: {e}", "key": key, "path": str(path)}
    mask_pixels = int(mask_small.sum())
    info = {"status": "ok", "key": key, "path": str(path), "mask_pixels": mask_pixels, "mask_shape": list(mask_small.shape)}
    if mask_pixels < int(args.dense_min_mask_pixels):
        info["status"] = "too_few_mask_pixels"
        return None, info
    h, w = frame_shape
    mask = cv2.resize(mask_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    info["mask_pixels_resized"] = int(mask.sum())
    return mask, info


def points_in_mask(pts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    x = np.rint(pts[:, 0]).astype(np.int64)
    y = np.rint(pts[:, 1]).astype(np.int64)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    out = np.zeros(len(pts), dtype=bool)
    out[inside] = mask[y[inside], x[inside]]
    return out


OCCL_MESH_CACHE: dict[str, "np.ndarray"] = {}
OCCL_DEPTH_CACHE: dict[tuple, "np.ndarray"] = {}
OCCL_DEFAULT_MESH_DIR = str(source_path('assets', 'csgo-datasets-fullsubset/32f1644d4f42c29d/00a145c4184c4cf092acb25d2521ccee'))
ANGLE_BUCKET_EDGES = [(0.0, 10.0), (10.0, 30.0), (30.0, 60.0), (60.0, 180.1)]


def view_angle_deg(c2w_a, c2w_b) -> float:
    fa = c2w_a[:3, 2] / (np.linalg.norm(c2w_a[:3, 2]) + 1e-9)
    fb = c2w_b[:3, 2] / (np.linalg.norm(c2w_b[:3, 2]) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(float(fa @ fb), -1.0, 1.0))))


def angle_bucket_name(a: float) -> str:
    for lo, hi in ANGLE_BUCKET_EDGES:
        if lo <= a < hi:
            return f"{int(lo)}-{int(hi) if hi < 180 else 'inf'}"
    return "invalid"


def load_world_mesh_points(pair: dict, n_samples: int, seed: int = 20260705):
    """Area-weighted surface samples of the world mesh (mining-equivalent geometry source)."""
    mdir = Path((pair.get("scene_covis_policy_v3") or {}).get("mesh_source_match_dir") or OCCL_DEFAULT_MESH_DIR)
    key = f"{mdir}:{n_samples}"
    if key in OCCL_MESH_CACHE:
        return OCCL_MESH_CACHE[key]
    manifest = json.loads((mdir / "mesh_manifest.json").read_text(encoding="utf-8"))
    world = next(m for m in manifest if m.get("model_name") == "_world_")
    verts, facelist = [], []
    with (mdir / "meshes" / world["mesh_file"]).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split(); verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                for i in range(1, len(idx) - 1):
                    facelist.append([idx[0], idx[i], idx[i + 1]])
    tri = np.asarray(verts, np.float32)[np.asarray(facelist, np.int32)]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    fi = rng.choice(len(tri), size=n_samples, p=area / area.sum())
    r1 = np.sqrt(rng.random(n_samples, dtype=np.float32)); r2 = rng.random(n_samples, dtype=np.float32)
    pts = (1 - r1)[:, None] * tri[fi, 0] + (r1 * (1 - r2))[:, None] * tri[fi, 1] + (r1 * r2)[:, None] * tri[fi, 2]
    OCCL_MESH_CACHE[key] = pts.astype(np.float32)
    return OCCL_MESH_CACHE[key]


def env_depth_buffer(pair: dict, which: str, li: int, c2w, intr4, args):
    """Mining point_zbuffer semantics via K projection; cached per (clip_dir, li)."""
    key = (pair[f"clip{which}_dir"], int(li), int(args.occl_grid_w), int(args.occl_grid_h))
    if key in OCCL_DEPTH_CACHE:
        return OCCL_DEPTH_CACHE[key]
    pts = load_world_mesh_points(pair, int(args.occl_mesh_samples))
    fx, fy, cx, cy = [float(x) for x in intr4]
    r = c2w[:3, :3].T; t = (-r @ c2w[:3, 3])[:, None]
    camp = (r @ pts.T + t).T
    z = camp[:, 2]
    u = fx * camp[:, 0] / np.maximum(z, 1e-6) + cx
    v = fy * camp[:, 1] / np.maximum(z, 1e-6) + cy
    w, h = 832.0, 480.0
    inside = (z > args.occl_near) & (z < args.occl_far) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    gw, gh = int(args.occl_grid_w), int(args.occl_grid_h)
    depth = np.full((gh, gw), np.inf, dtype=np.float32)
    if inside.any():
        ub = np.clip((u[inside] * gw / w).astype(np.int32), 0, gw - 1)
        vb = np.clip((v[inside] * gh / h).astype(np.int32), 0, gh - 1)
        np.minimum.at(depth, (vb, ub), z[inside].astype(np.float32))
    OCCL_DEPTH_CACHE[key] = depth
    return depth


def occluded_flags(x3d, z_cam, c2w, intr4, depth, args):
    """visible_indices_from_zbuffer semantics: occluded iff finite env depth sits in front beyond tol."""
    fx, fy, cx, cy = [float(x) for x in intr4]
    r = c2w[:3, :3].T; t = (-r @ c2w[:3, 3])[:, None]
    camp = (r @ x3d.T + t).T
    u = fx * camp[:, 0] / np.maximum(camp[:, 2], 1e-6) + cx
    v = fy * camp[:, 1] / np.maximum(camp[:, 2], 1e-6) + cy
    gh, gw = depth.shape
    ub = np.clip((u * gw / 832.0).astype(np.int32), 0, gw - 1)
    vb = np.clip((v * gh / 480.0).astype(np.int32), 0, gh - 1)
    d = depth[vb, ub]
    tol = np.maximum(float(args.occl_tol_abs), float(args.occl_tol_rel) * z_cam)
    return np.isfinite(d) & ((z_cam - d) > tol)


def frame_metrics(pair: dict, variant: str, roma, det, dense_index, args, tmpd: str) -> dict:
    mp_a, yo_a = resolve_video(pair, "A", variant, args)
    mp_b, yo_b = resolve_video(pair, "B", variant, args)
    if not mp_a or not mp_b or not Path(mp_a).exists() or not Path(mp_b).exists():
        return {"status": "missing_video", "frames": [], "n_candidate_frames": min(len(pair["mutual_covis_frames"]), args.max_frames)}
    yo_a, h_meas_a = measure_height_y_off(mp_a, yo_a)
    yo_b, h_meas_b = measure_height_y_off(mp_b, yo_b)
    c_a = np.load(Path(pair["clipA_dir"]) / "poses.npy").astype(np.float64)
    k_a = np.load(Path(pair["clipA_dir"]) / "intrinsics.npy").astype(np.float64)
    c_b = np.load(Path(pair["clipB_dir"]) / "poses.npy").astype(np.float64)
    k_b = np.load(Path(pair["clipB_dir"]) / "intrinsics.npy").astype(np.float64)
    frames = sorted(pair["mutual_covis_frames"], key=lambda x: -(x[2] + x[3]))[: args.max_frames]
    n_candidates = len(frames)
    rows = []
    for li, raw, pix_ab, pix_ba in frames:
        f_a, f_b = read_frame(mp_a, int(li)), read_frame(mp_b, int(li))
        if f_a is None or f_b is None:
            rows.append({"li": li, "raw": raw, "status": "missing_frame"})
            continue
        pa, pb = os.path.join(tmpd, "a.png"), os.path.join(tmpd, "b.png")
        cv2.imwrite(pa, f_a); cv2.imwrite(pb, f_b)
        warp, cert = roma.match(pa, pb, device=args.device)
        matches, certs = roma.sample(warp, cert, num=args.num_matches)
        pts_a, pts_b = roma.to_pixel_coordinates(matches, f_a.shape[0], f_a.shape[1], f_b.shape[0], f_b.shape[1])
        pts_a = pts_a.detach().cpu().numpy().astype(np.float64)
        pts_b = pts_b.detach().cpu().numpy().astype(np.float64)
        certs = certs.detach().cpu().numpy().astype(np.float64)
        sel = certs > args.certainty
        row = {"li": int(li), "raw": int(raw), "pix_ab": float(pix_ab), "pix_ba": float(pix_ba), "matches_certainty": int(sel.sum()), "mean_confidence": round(float(certs[sel].mean()), 4) if sel.any() else None}
        row["view_angle_deg"] = round(view_angle_deg(c_a[int(li)], c_b[int(li)]), 3)
        if sel.sum() < 8:
            row.update({"status": "too_few_matches", "covisible": stat([]), "dynamic_player": stat([])})
            rows.append(row); continue
        pta, ptb = pts_a[sel], pts_b[sel]
        pma = proj_mat(c_a[int(li)], k_eff(k_a[int(li)], yo_a))
        pmb = proj_mat(c_b[int(li)], k_eff(k_b[int(li)], yo_b))
        x = cv2.triangulatePoints(pma, pmb, pta.T, ptb.T)
        x = (x[:3] / x[3]).T
        ma, mb = w2c(c_a[int(li)]), w2c(c_b[int(li)])
        za = (ma[:3, :3] @ x.T + ma[:3, 3:4]).T[:, 2]
        zb = (mb[:3, :3] @ x.T + mb[:3, 3:4]).T[:, 2]
        ca, cb = c_a[int(li)][:3, 3], c_b[int(li)][:3, 3]
        ra, rb = x - ca, x - cb
        ra /= np.linalg.norm(ra, axis=1, keepdims=True) + 1e-9
        rb /= np.linalg.norm(rb, axis=1, keepdims=True) + 1e-9
        ang = np.degrees(np.arccos(np.clip((ra * rb).sum(1), -1, 1)))
        good = (za > 0) & (zb > 0) & (ang > args.min_angle)
        def rp(pmat, xs):
            y = (pmat @ np.hstack([xs, np.ones((len(xs), 1))]).T).T
            return y[:, :2] / y[:, 2:3]
        errs = 0.5 * (np.linalg.norm(rp(pma, x) - pta, axis=1) + np.linalg.norm(rp(pmb, x) - ptb, axis=1))
        if args.occlusion_stratify:
            try:
                da = env_depth_buffer(pair, "A", int(li), c_a[int(li)], k_a[int(li)], args)
                db = env_depth_buffer(pair, "B", int(li), c_b[int(li)], k_b[int(li)], args)
                occ_a = occluded_flags(x, za, c_a[int(li)], k_a[int(li)], da, args)
                occ_b = occluded_flags(x, zb, c_b[int(li)], k_b[int(li)], db, args)
                occ = good & (occ_a | occ_b)
                vis = good & ~occ
                row["covisible_visible_errors"] = errs[vis].astype(float).tolist()
                row["occluded_errors"] = errs[occ].astype(float).tolist()
                row["occlusion_info"] = {
                    "status": "ok", "n_good": int(good.sum()), "n_visible": int(vis.sum()),
                    "n_occluded": int(occ.sum()), "occluded_frac": round(float(occ.sum() / max(int(good.sum()), 1)), 4),
                    "zbuf_fill_a": round(float(np.isfinite(da).mean()), 4), "zbuf_fill_b": round(float(np.isfinite(db).mean()), 4),
                    "tol_abs": float(args.occl_tol_abs), "tol_rel": float(args.occl_tol_rel),
                }
            except Exception as e:
                row["occlusion_info"] = {"status": f"error: {type(e).__name__}: {e}"}
        dyn = np.zeros(len(errs), dtype=bool)
        dyn_info = {"mode": "disabled"}
        if not args.no_dynamic:
            ma_mask, ma_info = load_player_mask(dense_index, dense_sample_keys(pair, "A", int(raw)), f_a.shape[:2], args)
            mb_mask, mb_info = load_player_mask(dense_index, dense_sample_keys(pair, "B", int(raw)), f_b.shape[:2], args)
            dyn_info = {"mode": "dense_player", "A": ma_info, "B": mb_info}
            if ma_mask is not None and mb_mask is not None:
                in_dyn = points_in_mask(pta, ma_mask) & points_in_mask(ptb, mb_mask)
                dyn = good & in_dyn
                dyn_info["matches_dynamic_player"] = int(dyn.sum())
                if int(dyn.sum()) < int(args.dense_min_dynamic_matches):
                    dyn_info["status"] = "too_few_dynamic_matches"
                    dyn[:] = False
                else:
                    dyn_info["status"] = "ok"
            else:
                dyn_info["status"] = "missing_or_small_mask"
        row.update({"status": "ok", "matches_good": int(good.sum()), "dynamic_player_info": dyn_info, "covisible_errors": errs[good].astype(float).tolist(), "dynamic_errors": errs[dyn].astype(float).tolist()})
        rows.append(row)
    return {"status": "ok", "videoA": mp_a, "videoB": mp_b, "video_height_a": h_meas_a, "video_height_b": h_meas_b, "y_off_a": yo_a, "y_off_b": yo_b, "n_candidate_frames": n_candidates, "frames": rows}


def wilcoxon_signed_rank(diffs: list[float]) -> dict:
    vals = [float(x) for x in diffs if x != 0 and math.isfinite(x)]
    n = len(vals)
    if n == 0:
        return {"n": 0, "stat": None, "p_value": None, "method": "no_nonzero_diffs"}
    try:
        from scipy.stats import wilcoxon
        res = wilcoxon(vals, alternative="two-sided", zero_method="wilcox")
        return {"n": n, "stat": float(res.statistic), "p_value": float(res.pvalue), "method": "scipy.wilcoxon"}
    except Exception:
        absvals = np.asarray([abs(x) for x in vals])
        order = np.argsort(absvals)
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n + 1)
        wpos = float(ranks[np.asarray(vals) > 0].sum())
        mean = n * (n + 1) / 4
        var = n * (n + 1) * (2 * n + 1) / 24
        z = (wpos - mean) / math.sqrt(var) if var > 0 else 0.0
        p = math.erfc(abs(z) / math.sqrt(2))
        return {"n": n, "stat": wpos, "p_value": p, "method": "normal_approx_no_tie_correction"}


def bootstrap_ci(vals: list[float], reps: int = 5000, seed: int = 123) -> list[float | None]:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    meds = [float(np.median(rng.choice(arr, size=arr.size, replace=True))) for _ in range(reps)]
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def summarize_pair(res: dict, threshold: float | None) -> dict:
    cov, dyn, confs, mcert = [], [], [], []
    vis_errs, occ_errs, angles = [], [], []
    status_counts: dict[str, int] = {}
    for fr in res.get("frames", []):
        cov += fr.get("covisible_errors", [])
        dyn += fr.get("dynamic_errors", [])
        vis_errs += fr.get("covisible_visible_errors", [])
        occ_errs += fr.get("occluded_errors", [])
        if fr.get("view_angle_deg") is not None: angles.append(float(fr["view_angle_deg"]))
        st = str(fr.get("status")); status_counts[st] = status_counts.get(st, 0) + 1
        if fr.get("mean_confidence") is not None: confs.append(fr["mean_confidence"])
        if fr.get("matches_certainty") is not None: mcert.append(fr["matches_certainty"])
    n_cand = res.get("n_candidate_frames")
    n_ok = sum(1 for f in res.get("frames", []) if f.get("status") == "ok")
    n_skip = sum(v for k, v in status_counts.items() if k in ("too_few_matches", "missing_frame"))
    med_ang = round(float(np.median(angles)), 3) if angles else None
    out = {"covisible": stat(cov, threshold), "dynamic_player": stat(dyn, threshold),
           "frames_ok": n_ok, "matched_points_total": int(sum(mcert)),
           "matched_points_mean": round(float(np.mean(mcert)), 3) if mcert else None,
           "mean_confidence": round(float(np.mean(confs)), 4) if confs else None}
    out["n_candidate_frames"] = n_cand
    out["coverage"] = round(n_ok / n_cand, 4) if n_cand else None
    out["skip_rate"] = round(n_skip / n_cand, 4) if n_cand else None
    out["frame_status_counts"] = status_counts
    out["view_angle_median_deg"] = med_ang
    out["view_angle_bucket"] = angle_bucket_name(med_ang) if med_ang is not None else None
    out["covisible_visible"] = stat(vis_errs, threshold)
    out["occluded"] = stat(occ_errs, threshold)
    out["occluded_point_frac"] = round(len(occ_errs) / max(len(vis_errs) + len(occ_errs), 1), 4) if (vis_errs or occ_errs) else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-index", type=Path, default=Path(str(source_path('project', 'output/rank64_verdict_20260629_1330_v0/symmvc_v2_motion/pair_index_v2_motion.json'))))
    ap.add_argument("--gen-root", type=Path, default=Path(str(source_path('project', 'output/cleanv2_verdict_20260703_v0/gen/v2'))))
    ap.add_argument("--symmvc-view-root", type=Path, default=Path(str(source_path('project', 'output/cleanv2_verdict_20260703_v0/gen_symmvc_view'))))
    ap.add_argument("--out-dir", type=Path, default=Path(str(source_path('scene', 'output/symmvc_v2_eval_20260705_v0'))))
    ap.add_argument("--variants", default="gt,base,true_dense,shuffled_dense")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--min-mutual-frames", type=int, default=4)
    ap.add_argument("--min-mutual-pixels", type=float, default=50.0)
    ap.add_argument("--num-matches", type=int, default=3000)
    ap.add_argument("--certainty", type=float, default=0.5)
    ap.add_argument("--min-angle", type=float, default=2.0)
    ap.add_argument("--det-threshold", type=float, default=0.5, help="legacy detector threshold; unused by dense dynamic mode")
    ap.add_argument("--dense-samples-root", type=Path, default=Path("output/verdict70h_20260706_v0/dense_backfill_v0/gpu_0_1_full/samples"))
    ap.add_argument("--dense-player-threshold", type=float, default=0.5)
    ap.add_argument("--dense-min-mask-pixels", type=int, default=48)
    ap.add_argument("--dense-min-dynamic-matches", type=int, default=8)
    ap.add_argument("--no-dynamic", action="store_true")
    ap.add_argument("--occlusion-stratify", action="store_true", help="stratify covisible errors into visible/occluded via mining-equivalent env z-buffer")
    ap.add_argument("--occl-mesh-samples", type=int, default=800000)
    ap.add_argument("--occl-grid-w", type=int, default=208)
    ap.add_argument("--occl-grid-h", type=int, default=120)
    ap.add_argument("--occl-near", type=float, default=4.0)
    ap.add_argument("--occl-far", type=float, default=3000.0)
    ap.add_argument("--occl-tol-abs", type=float, default=18.0)
    ap.add_argument("--occl-tol-rel", type=float, default=0.035)
    ap.add_argument("--upsample", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs_all = json.load(args.pair_index.open())
    pairs = [p for p in pairs_all if int(p.get("n_mutual_covis_frames", 0)) >= args.min_mutual_frames and min(float(p.get("a_sees_b_maxpix", 0)), float(p.get("b_sees_a_maxpix", 0))) >= args.min_mutual_pixels]
    pairs.sort(key=lambda p: -min(float(p["a_sees_b_maxpix"]), float(p["b_sees_a_maxpix"])))
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    log(f"quality gate kept {len(pairs)}/{len(pairs_all)} pairs min_frames={args.min_mutual_frames} min_pixels={args.min_mutual_pixels}")
    raw_quality = [{"pair": f"{p.get('clipA_id')}__{p.get('clipB_id')}", "n_mutual_covis_frames": p.get("n_mutual_covis_frames"), "min_maxpix": min(float(p.get("a_sees_b_maxpix", 0)), float(p.get("b_sees_a_maxpix", 0)))} for p in pairs_all]
    if not pairs:
        out = {"kind": "symmvc_v2_score_v0", "status": "no_pairs_after_quality_gate", "params": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "raw_pair_quality": raw_quality, "by_variant": {}, "paired_stats": {}}
        (args.out_dir / "symmvc_v2_scores.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        md = "# SymMVC v2 eval\n\nNo pairs passed quality gate.\n\n| all_pairs | kept | min_mutual_frames | min_mutual_pixels |\n|---:|---:|---:|---:|\n| %d | 0 | %d | %.1f |\n" % (len(pairs_all), args.min_mutual_frames, args.min_mutual_pixels)
        (args.out_dir / "summary.md").write_text(md, encoding="utf-8")
        log(f"WROTE {args.out_dir / 'symmvc_v2_scores.json'}")
        return
    import torch
    from romatch import roma_outdoor
    import romatch.models.matcher as matcher
    roma = roma_outdoor(device=args.device)
    orig_lc = matcher.local_correlation
    matcher.local_correlation = lambda *a, **k: orig_lc(*a, **{**k, "use_custom_corr": False})
    for m in roma.modules():
        if hasattr(m, "use_custom_corr"): m.use_custom_corr = False
    if not args.upsample:
        roma.upsample_preds = False
    det = None
    dense_index = {} if args.no_dynamic else build_dense_index(args.dense_samples_root)
    log(f"dense dynamic samples indexed {len(dense_index)} from {args.dense_samples_root}; mask_threshold={args.dense_player_threshold} min_mask_pixels={args.dense_min_mask_pixels} min_dynamic_matches={args.dense_min_dynamic_matches}")
    detailed = []
    with tempfile.TemporaryDirectory() as tmpd:
        for pi, pair in enumerate(pairs):
            pair_id = f"{pair.get('clipA_id')}__{pair.get('clipB_id')}"
            rec = {"pair_id": pair_id, "pair_meta": {k: pair.get(k) for k in ("game", "ep", "clipA_id", "clipB_id", "n_mutual_covis_frames", "a_sees_b_maxpix", "b_sees_a_maxpix")}, "variants": {}}
            for v in variants:
                t0 = time.time()
                try: vr = frame_metrics(pair, v, roma, det, dense_index, args, tmpd)
                except Exception as e: vr = {"status": f"error: {type(e).__name__}: {e}", "frames": []}
                vr["sec"] = round(time.time() - t0, 2)
                rec["variants"][v] = vr
                log(f"{pi+1}/{len(pairs)} {v} {pair_id[:18]} status={vr.get('status')} sec={vr['sec']}")
            detailed.append(rec)
    gt_p90_vals = []
    for rec in detailed:
        gt_sum = summarize_pair(rec["variants"].get("gt", {}), None)
        if gt_sum["covisible"]["p90"] is not None: gt_p90_vals.append(gt_sum["covisible"]["p90"])
    threshold = 2.0 * float(np.median(gt_p90_vals)) if gt_p90_vals else None
    by_pair = []
    for rec in detailed:
        row = {"pair_id": rec["pair_id"], "pair_meta": rec["pair_meta"], "variants": {}}
        gt_match_mean = summarize_pair(rec["variants"].get("gt", {}), threshold)["matched_points_mean"]
        for v in variants:
            s = summarize_pair(rec["variants"].get(v, {}), threshold)
            s["match_count_ratio_vs_gt"] = None if not gt_match_mean or not s["matched_points_mean"] else round(float(s["matched_points_mean"] / gt_match_mean), 4)
            row["variants"][v] = s
        by_pair.append(row)
    by_variant = {}
    for v in variants:
        meds = [p["variants"][v]["covisible"]["median"] for p in by_pair if p["variants"].get(v, {}).get("covisible", {}).get("median") is not None]
        p90s = [p["variants"][v]["covisible"]["p90"] for p in by_pair if p["variants"].get(v, {}).get("covisible", {}).get("p90") is not None]
        dmeds = [p["variants"][v]["dynamic_player"]["median"] for p in by_pair if p["variants"].get(v, {}).get("dynamic_player", {}).get("median") is not None]
        dp90s = [p["variants"][v]["dynamic_player"]["p90"] for p in by_pair if p["variants"].get(v, {}).get("dynamic_player", {}).get("p90") is not None]
        by_variant[v] = {
            "n_pairs": len(meds),
            "median_of_pair_medians": round(float(np.median(meds)), 4) if meds else None,
            "median_of_pair_p90": round(float(np.median(p90s)), 4) if p90s else None,
            "n_pairs_dyn": len(dmeds),
            "dynamic_median_of_pair_medians": round(float(np.median(dmeds)), 4) if dmeds else None,
            "dynamic_median_of_pair_p90": round(float(np.median(dp90s)), 4) if dp90s else None,
            "coverage_mean": round(float(np.mean([p["variants"][v]["coverage"] for p in by_pair if p["variants"].get(v, {}).get("coverage") is not None])), 4) if any(p["variants"].get(v, {}).get("coverage") is not None for p in by_pair) else None,
            "skip_rate_mean": round(float(np.mean([p["variants"][v]["skip_rate"] for p in by_pair if p["variants"].get(v, {}).get("skip_rate") is not None])), 4) if any(p["variants"].get(v, {}).get("skip_rate") is not None for p in by_pair) else None,
            "view_angle_bucket_pairs": {bk: sum(1 for p in by_pair if p["variants"].get(v, {}).get("view_angle_bucket") == bk) for bk in ["0-10", "10-30", "30-60", "60-inf"]},
            "covisible_median_by_angle_bucket": {bk: (round(float(np.median(vals)), 4) if (vals := [p["variants"][v]["covisible"]["median"] for p in by_pair if p["variants"].get(v, {}).get("view_angle_bucket") == bk and p["variants"][v]["covisible"]["median"] is not None]) else None) for bk in ["0-10", "10-30", "30-60", "60-inf"]},
            "visible_median_of_pair_medians": (lambda vv: round(float(np.median(vv)), 4) if vv else None)([p["variants"][v]["covisible_visible"]["median"] for p in by_pair if p["variants"].get(v, {}).get("covisible_visible", {}).get("median") is not None]),
            "occluded_median_of_pair_medians": (lambda vv: round(float(np.median(vv)), 4) if vv else None)([p["variants"][v]["occluded"]["median"] for p in by_pair if p["variants"].get(v, {}).get("occluded", {}).get("median") is not None]),
            "occluded_point_frac_mean": (lambda vv: round(float(np.mean(vv)), 4) if vv else None)([p["variants"][v]["occluded_point_frac"] for p in by_pair if p["variants"].get(v, {}).get("occluded_point_frac") is not None]),
        }
    paired = {}
    for channel in ["covisible", "dynamic_player"]:
        for a, b in [("true_dense", "shuffled_dense"), ("true_dense", "base")]:
            diffs = []
            for p in by_pair:
                av = p["variants"].get(a, {}).get(channel, {}).get("median")
                bv = p["variants"].get(b, {}).get(channel, {}).get("median")
                if av is not None and bv is not None: diffs.append(float(av) - float(bv))
            paired[f"{channel}:{a}_minus_{b}"] = {"diffs": diffs, "median_diff": round(float(np.median(diffs)), 4) if diffs else None, "wilcoxon": wilcoxon_signed_rank(diffs), "bootstrap_median_diff_95ci": bootstrap_ci(diffs)}
    out = {"kind": "symmvc_v2_score_v0", "status": "ok", "params": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "threshold_gt_p90_x2": threshold, "raw_pair_quality": raw_quality, "by_variant": by_variant, "paired_stats": paired, "by_pair": by_pair, "detailed_frames": detailed}
    (args.out_dir / "symmvc_v2_scores.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    lines = ["# SymMVC v2 eval", "", f"quality gate kept {len(pairs)}/{len(pairs_all)} pairs; threshold_gt_p90_x2={threshold}", f"dense_dynamic: samples_root={args.dense_samples_root} player_threshold={args.dense_player_threshold} min_mask_pixels={args.dense_min_mask_pixels} min_dynamic_matches={args.dense_min_dynamic_matches}", "", "| variant | n_pairs | cov median(pair median) | cov median(pair p90) | n_pairs_dyn | dyn median(pair median) | dyn median(pair p90) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for v in variants:
        s = by_variant[v]; lines.append(f"| {v} | {s['n_pairs']} | {s['median_of_pair_medians']} | {s['median_of_pair_p90']} | {s['n_pairs_dyn']} | {s['dynamic_median_of_pair_medians']} | {s['dynamic_median_of_pair_p90']} |")
    lines += ["", "| contrast | n | median diff | Wilcoxon p | bootstrap 95% CI |", "|---|---:|---:|---:|---|"]
    for k, s in paired.items(): lines.append(f"| {k} | {s['wilcoxon']['n']} | {s['median_diff']} | {s['wilcoxon']['p_value']} | {s['bootstrap_median_diff_95ci']} |")
    lines += ["", "| variant | coverage_mean | skip_rate_mean | angle buckets (pairs) | visible med | occluded med | occl frac |", "|---|---:|---:|---|---:|---:|---:|"]
    for v in variants:
        s = by_variant[v]
        lines.append(f"| {v} | {s.get('coverage_mean')} | {s.get('skip_rate_mean')} | {s.get('view_angle_bucket_pairs')} | {s.get('visible_median_of_pair_medians')} | {s.get('occluded_median_of_pair_medians')} | {s.get('occluded_point_frac_mean')} |")
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"WROTE {args.out_dir / 'symmvc_v2_scores.json'} and {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
