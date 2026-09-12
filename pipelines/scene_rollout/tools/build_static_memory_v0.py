#!/usr/bin/env python3
"""Build Static Map Memory metadata for one CS:GO match/map directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
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
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "meta": str(out),
        "map_name": meta["map_name"],
        "mesh_vertices": vertex_count,
        "mesh_faces": face_count,
        "nav_areas": len(navmesh["areas"]),
        "static_props": len(static_props),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
