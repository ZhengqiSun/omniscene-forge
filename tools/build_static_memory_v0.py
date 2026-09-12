#!/usr/bin/env python3
"""Build Static Map Memory metadata for one CS:GO match/map directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def file_fingerprint(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        return {"path": str(path), "exists": False}
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "sha256": h.hexdigest(),
    }


def nav_bounds(navmesh: dict[str, Any]) -> dict[str, float]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for area in navmesh["areas"].values():
        nw = area["nw_corner"]
        se = area["se_corner"]
        xs.extend([float(nw[0]), float(se[0])])
        ys.extend([float(nw[1]), float(se[1])])
        zs.extend([float(nw[2]), float(se[2]), float(area.get("ne_z", nw[2])), float(area.get("sw_z", se[2]))])
    return {"x_min": min(xs), "x_max": max(xs), "y_min": min(ys), "y_max": max(ys), "z_min": min(zs), "z_max": max(zs)}


def world_mesh_entry(match_dir: Path) -> dict[str, Any]:
    manifest = load_json(match_dir / "mesh_manifest.json")
    entry = next(m for m in manifest if m.get("model_name") == "_world_")
    mesh_path = match_dir / "meshes" / entry["mesh_file"]
    return {**entry, "mesh_path": str(mesh_path)}


def obj_bounds(path: Path) -> tuple[dict[str, float], int, int]:
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    vertices = 0
    faces = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                vals = [float(x) for x in line.split()[1:4]]
                vertices += 1
                for i, val in enumerate(vals):
                    mins[i] = min(mins[i], val)
                    maxs[i] = max(maxs[i], val)
            elif line.startswith("f "):
                faces += 1
    return (
        {"x_min": mins[0], "x_max": maxs[0], "y_min": mins[1], "y_max": maxs[1], "z_min": mins[2], "z_max": maxs[2]},
        vertices,
        faces,
    )


def load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    data = load_json(path)
    return data if isinstance(data, dict) else {"value": data}


def npz_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    import numpy as np

    with np.load(path) as data:
        shapes = {key: list(data[key].shape) for key in data.files}
        counts = {
            "vertices": int(data["vertices"].shape[0]) if "vertices" in data else None,
            "faces": int(data["faces"].shape[0]) if "faces" in data else None,
            "base_vertices": int(data["base_vertices"].shape[0]) if "base_vertices" in data else None,
            "base_faces": int(data["base_faces"].shape[0]) if "base_faces" in data else None,
            "displacement_vertices": int(data["displacement_vertices"].shape[0]) if "displacement_vertices" in data else None,
            "displacement_faces": int(data["displacement_faces"].shape[0]) if "displacement_faces" in data else None,
        }
    return {"path": str(path), "arrays": shapes, "component_counts": counts}


def static_memory_v1(
    args: argparse.Namespace,
    navmesh: dict[str, Any],
    static_props: list[Any],
    world: dict[str, Any],
    mesh_bounds: dict[str, float],
    vertex_count: int,
    face_count: int,
) -> dict[str, Any]:
    nav_path = args.match_dir / "navmesh.json"
    static_props_path = args.match_dir / "static_props.json"
    mesh_manifest_path = args.match_dir / "mesh_manifest.json"
    mesh_path = Path(world["mesh_path"])
    inspection = load_optional_json(args.bsp_inspection_json)
    bsp_npz = npz_summary(args.bsp_faces_npz)
    bsp_counts = (inspection or {}).get("counts", {})
    brush_contents_counts = (inspection or {}).get("brush_contents_counts", {})
    static_prop_gamelump = (inspection or {}).get("static_prop_gamelump", {})
    nonempty_lumps = (inspection or {}).get("nonempty_lumps", [])
    nonempty_lump_names = [str(row.get("name", "")).lower() for row in nonempty_lumps if isinstance(row, dict)]
    geometry_backend_id = args.geometry_backend_id
    backend_features = ["world_obj_visual_mesh"]
    component_counts = {
        "obj_vertices": int(vertex_count),
        "obj_faces": int(face_count),
        "static_props_json": len(static_props),
    }
    if bsp_npz is not None:
        geometry_backend_id = geometry_backend_id or "bsp_faces_disp_gpu"
        backend_features = ["visual_faces", "displacement"]
        component_counts.update({k: v for k, v in bsp_npz["component_counts"].items() if v is not None})
    elif geometry_backend_id is None:
        geometry_backend_id = "obj_world_cpu"

    source_fingerprints = {
        "bsp": file_fingerprint(args.bsp_path),
        "bsp_inspection_json": file_fingerprint(args.bsp_inspection_json),
        "bsp_faces_npz": file_fingerprint(args.bsp_faces_npz),
        "mesh_manifest": file_fingerprint(mesh_manifest_path),
        "world_obj": file_fingerprint(mesh_path),
        "navmesh": file_fingerprint(nav_path),
        "static_props": file_fingerprint(static_props_path),
    }
    signature_parts = [
        geometry_backend_id,
        source_fingerprints["bsp_faces_npz"]["sha256"] if source_fingerprints["bsp_faces_npz"] and source_fingerprints["bsp_faces_npz"].get("sha256") else "",
        source_fingerprints["world_obj"]["sha256"] if source_fingerprints["world_obj"] and source_fingerprints["world_obj"].get("sha256") else "",
        source_fingerprints["navmesh"]["sha256"] if source_fingerprints["navmesh"] and source_fingerprints["navmesh"].get("sha256") else "",
    ]
    backend_signature = hashlib.sha256("|".join(signature_parts).encode("utf-8")).hexdigest()

    return {
        "schema_version": "StaticMemoryV1",
        "kind": "static_memory_cache_v1",
        "map_identity": {
            "map_name": navmesh.get("map_name"),
            "match_dir": str(args.match_dir),
            "bsp_path": str(args.bsp_path) if args.bsp_path else None,
            "bsp_version": (inspection or {}).get("version"),
            "bsp_map_revision": (inspection or {}).get("map_revision"),
        },
        "coordinate_system": {
            "units": "Source/CS:GO world units",
            "axes": "CS:GO world xyz",
            "camera_convention": "camera_position + yaw/pitch; OpenCV-style camera forward is used by dense renderer.",
        },
        "source_fingerprints": source_fingerprints,
        "geometry_backend": {
            "geometry_backend_id": geometry_backend_id,
            "backend_signature": backend_signature,
            "backend_features": backend_features,
            "component_counts": component_counts,
            "mesh_ref": str(args.bsp_faces_npz or mesh_path),
            "world_obj": {
                "path": str(mesh_path),
                "bounds": mesh_bounds,
                "vertices": int(vertex_count),
                "faces": int(face_count),
            },
            "bsp_faces_npz": bsp_npz,
        },
        "navmesh": {
            "path": str(nav_path),
            "area_count": len(navmesh["areas"]),
            "place_count": len(navmesh.get("place_names", [])),
            "place_names": navmesh.get("place_names", []),
            "bounds": nav_bounds(navmesh),
        },
        "static_props": {
            "json_path": str(static_props_path),
            "json_count": len(static_props),
            "gamelump_model_dict_count": static_prop_gamelump.get("dict_count"),
            "gamelump_prop_count": static_prop_gamelump.get("prop_count"),
            "status": "indexed_for_provenance_not_yet_merged_into_renderer",
        },
        "brush_collision": {
            "status": "inspected_not_yet_renderer_input",
            "brush_count": bsp_counts.get("brush"),
            "brushside_count": bsp_counts.get("brushside"),
            "contents_counts": brush_contents_counts,
        },
        "visibility_occlusion": {
            "status": "inspected_not_yet_renderer_input",
            "nonempty_lumps": nonempty_lumps,
            "has_visibility_lump": "visibility" in nonempty_lump_names,
            "has_occlusion_lump": "occlusion" in nonempty_lump_names,
        },
        "semantic_policy": {
            "encoding": "0=no hit; non-nav visible mesh uses fallback; nav place is encoded by place id.",
            "teacher_streams": "RGB/depth/seg/visibility are QA/target only, not StaticMemory inputs.",
        },
        "renderer_defaults": {
            "image_size": [176, 320],
            "channels": [
                "env_depth_norm_from_world_obj_projection",
                "env_mesh_hit_mask",
                "env_nav_place_semantic_from_static_memory",
                "other_player_mask_from_memory_player_capsules",
                "other_player_depth_norm_from_memory_capsules",
                "other_player_yaw_sin_relative_to_ego_from_memory",
                "other_player_yaw_cos_relative_to_ego_from_memory",
            ],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--bsp-path", type=Path, default=None)
    ap.add_argument("--bsp-inspection-json", type=Path, default=None)
    ap.add_argument("--bsp-faces-npz", type=Path, default=None)
    ap.add_argument("--geometry-backend-id", default=None)
    args = ap.parse_args()

    navmesh = load_json(args.match_dir / "navmesh.json")
    static_props = load_json(args.match_dir / "static_props.json")
    world = world_mesh_entry(args.match_dir)
    mesh_path = Path(world["mesh_path"])
    mesh_bounds, vertex_count, face_count = obj_bounds(mesh_path)

    meta = {
        "kind": "static_map_memory_v0",
        "match_dir": str(args.match_dir),
        "map_name": navmesh.get("map_name"),
        "world_mesh": {
            "path": str(mesh_path),
            "model_name": world["model_name"],
            "vertex_count_manifest": world.get("vertex_count"),
            "triangle_count_manifest": world.get("triangle_count"),
            "vertex_count_obj": vertex_count,
            "face_count_obj": face_count,
            "bounds": mesh_bounds,
        },
        "navmesh": {
            "path": str(args.match_dir / "navmesh.json"),
            "area_count": len(navmesh["areas"]),
            "place_count": len(navmesh.get("place_names", [])),
            "bounds": nav_bounds(navmesh),
        },
        "static_props": {
            "path": str(args.match_dir / "static_props.json"),
            "count": len(static_props),
        },
        "renderer_policy": "Use world_mesh for first-person env depth/occlusion, navmesh for walkability/place QA, static props/labels for future semantics.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "static_memory_meta_v0.json"
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    v1 = static_memory_v1(args, navmesh, static_props, world, mesh_bounds, vertex_count, face_count)
    v1_out = args.out_dir / "static_memory_v1.json"
    v1_out.write_text(json.dumps(v1, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "meta": str(out),
        "static_memory_v1": str(v1_out),
        "map_name": meta["map_name"],
        "mesh_vertices": vertex_count,
        "mesh_faces": face_count,
        "nav_areas": len(navmesh["areas"]),
        "static_props": len(static_props),
        "geometry_backend_id": v1["geometry_backend"]["geometry_backend_id"],
        "backend_signature": v1["geometry_backend"]["backend_signature"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
