#!/usr/bin/env python3
"""Inspect Source/CS:GO VBSP static geometry resources.

This is the first BSP intake tool for Map Memory. It does not replace the
renderer yet; it verifies that a BSP is parseable, same-coordinate, and contains
the geometry/collision lumps needed to build a stricter static query layer.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np


HEADER_LUMPS = 64
LUMP_NAMES = {
    0: "entities",
    1: "planes",
    2: "texdata",
    3: "vertices",
    4: "visibility",
    5: "nodes",
    6: "texinfo",
    7: "faces",
    8: "lighting",
    9: "occlusion",
    10: "leaves",
    11: "faceids",
    12: "edges",
    13: "surfedges",
    14: "models",
    15: "worldlights",
    16: "leaffaces",
    17: "leafbrushes",
    18: "brushes",
    19: "brushsides",
    20: "areas",
    21: "areaportals",
    26: "dispinfo",
    27: "originalfaces",
    33: "dispverts",
    35: "gamelump",
    40: "pakfile",
    43: "cubemaps",
    44: "texdata_string_data",
    45: "texdata_string_table",
    46: "overlays",
    48: "physcollide",
    49: "dphysmodel",
    52: "leafwaterdata",
    53: "primitives",
    54: "primverts",
    55: "primindices",
}

STRUCT_SIZES = {
    "plane": 20,
    "vertex": 12,
    "edge": 4,
    "surfedge": 4,
    "face": 56,
    "model": 48,
    "brush": 12,
    "brushside": 8,
    "texinfo": 72,
    "texdata": 32,
    "dispinfo": 176,
    "dispvert": 20,
}

CONTENTS_FLAGS = {
    0x00000001: "SOLID",
    0x00000002: "WINDOW",
    0x00000004: "AUX",
    0x00000008: "GRATE",
    0x00000010: "SLIME",
    0x00000020: "WATER",
    0x00000040: "BLOCKLOS",
    0x00000080: "OPAQUE",
    0x00000100: "TESTFOGVOLUME",
    0x00000200: "UNUSED",
    0x00000800: "TEAM1",
    0x00001000: "TEAM2",
    0x00002000: "IGNORE_NODRAW_OPAQUE",
    0x00004000: "MOVEABLE",
    0x00008000: "AREAPORTAL",
    0x00010000: "PLAYERCLIP",
    0x00020000: "MONSTERCLIP",
    0x00040000: "CURRENT_0",
    0x00080000: "CURRENT_90",
    0x00100000: "CURRENT_180",
    0x00200000: "CURRENT_270",
    0x00400000: "CURRENT_UP",
    0x00800000: "CURRENT_DOWN",
    0x01000000: "ORIGIN",
    0x02000000: "MONSTER",
    0x04000000: "DEBRIS",
    0x08000000: "DETAIL",
    0x10000000: "TRANSLUCENT",
    0x20000000: "LADDER",
    0x40000000: "HITBOX",
}


def parse_header(blob: bytes) -> dict[str, Any]:
    if len(blob) < 8 + HEADER_LUMPS * 16 + 4:
        raise ValueError("File too small for VBSP header")
    ident, version = struct.unpack_from("<4sI", blob, 0)
    if ident != b"VBSP":
        raise ValueError(f"Unexpected BSP ident {ident!r}")
    lumps = []
    off = 8
    for idx in range(HEADER_LUMPS):
        fileofs, filelen, lump_version, fourcc = struct.unpack_from("<IIII", blob, off + idx * 16)
        lumps.append({
            "index": idx,
            "name": LUMP_NAMES.get(idx, f"lump_{idx}"),
            "offset": int(fileofs),
            "length": int(filelen),
            "version": int(lump_version),
            "fourcc": int(fourcc),
        })
    map_revision = struct.unpack_from("<I", blob, 8 + HEADER_LUMPS * 16)[0]
    return {
        "ident": ident.decode("ascii"),
        "version": int(version),
        "map_revision": int(map_revision),
        "lumps": lumps,
    }


def lump_bytes(blob: bytes, header: dict[str, Any], idx: int) -> bytes:
    lump = header["lumps"][idx]
    return blob[lump["offset"] : lump["offset"] + lump["length"]]


def count_lump(header: dict[str, Any], idx: int, struct_size: int) -> int | None:
    length = header["lumps"][idx]["length"]
    if length == 0:
        return 0
    if length % struct_size != 0:
        return None
    return length // struct_size


def parse_vertices(blob: bytes, header: dict[str, Any]) -> np.ndarray:
    data = lump_bytes(blob, header, 3)
    if len(data) % 12 != 0:
        raise ValueError("vertices lump size is not divisible by 12")
    return np.frombuffer(data, dtype="<f4").reshape(-1, 3).copy()


def parse_edges(blob: bytes, header: dict[str, Any]) -> np.ndarray:
    data = lump_bytes(blob, header, 12)
    if len(data) % 4 != 0:
        raise ValueError("edges lump size is not divisible by 4")
    return np.frombuffer(data, dtype="<u2").reshape(-1, 2).copy()


def parse_surfedges(blob: bytes, header: dict[str, Any]) -> np.ndarray:
    data = lump_bytes(blob, header, 13)
    if len(data) % 4 != 0:
        raise ValueError("surfedges lump size is not divisible by 4")
    return np.frombuffer(data, dtype="<i4").copy()


def parse_faces(blob: bytes, header: dict[str, Any], lump_idx: int = 7) -> np.ndarray:
    data = lump_bytes(blob, header, lump_idx)
    if len(data) % 56 != 0:
        raise ValueError(f"faces lump {lump_idx} size is not divisible by 56")
    dtype = np.dtype([
        ("planenum", "<u2"),
        ("side", "u1"),
        ("on_node", "u1"),
        ("firstedge", "<i4"),
        ("numedges", "<i2"),
        ("texinfo", "<i2"),
        ("dispinfo", "<i2"),
        ("surface_fog_volume_id", "<i2"),
        ("styles", "u1", (4,)),
        ("lightofs", "<i4"),
        ("area", "<f4"),
        ("lightmap_texture_mins_in_luxels", "<i4", (2,)),
        ("lightmap_texture_size_in_luxels", "<i4", (2,)),
        ("origface", "<i4"),
        ("numprims", "<u2"),
        ("firstprimid", "<u2"),
        ("smoothing_groups", "<u4"),
    ])
    return np.frombuffer(data, dtype=dtype).copy()


def parse_dispinfos(blob: bytes, header: dict[str, Any]) -> np.ndarray:
    data = lump_bytes(blob, header, 26)
    if len(data) % 176 != 0:
        raise ValueError("dispinfo lump size is not divisible by 176")
    dtype = np.dtype([
        ("start_position", "<f4", (3,)),
        ("disp_vert_start", "<i4"),
        ("disp_tri_start", "<i4"),
        ("power", "<i4"),
        ("min_tess", "<i4"),
        ("smoothing_angle", "<f4"),
        ("contents", "<i4"),
        ("map_face", "<u2"),
        ("lightmap_alpha_start", "<i4"),
        ("lightmap_sample_position_start", "<i4"),
        ("neighbors", "u1", (90,)),
        ("allowed_verts", "<u4", (10,)),
    ])
    return np.frombuffer(data, dtype=dtype).copy()


def parse_dispverts(blob: bytes, header: dict[str, Any]) -> np.ndarray:
    data = lump_bytes(blob, header, 33)
    if len(data) % 20 != 0:
        raise ValueError("dispverts lump size is not divisible by 20")
    dtype = np.dtype([
        ("vector", "<f4", (3,)),
        ("dist", "<f4"),
        ("alpha", "<f4"),
    ])
    return np.frombuffer(data, dtype=dtype).copy()


def parse_models(blob: bytes, header: dict[str, Any]) -> np.ndarray:
    data = lump_bytes(blob, header, 14)
    if len(data) % 48 != 0:
        raise ValueError("models lump size is not divisible by 48")
    dtype = np.dtype([
        ("mins", "<f4", (3,)),
        ("maxs", "<f4", (3,)),
        ("origin", "<f4", (3,)),
        ("headnode", "<i4"),
        ("firstface", "<i4"),
        ("numfaces", "<i4"),
    ])
    return np.frombuffer(data, dtype=dtype).copy()


def parse_brushes(blob: bytes, header: dict[str, Any]) -> np.ndarray:
    data = lump_bytes(blob, header, 18)
    if len(data) % 12 != 0:
        raise ValueError("brushes lump size is not divisible by 12")
    dtype = np.dtype([("firstside", "<i4"), ("numsides", "<i4"), ("contents", "<i4")])
    return np.frombuffer(data, dtype=dtype).copy()


def content_flag_names(contents: int) -> list[str]:
    value = int(contents) & 0xFFFFFFFF
    return [name for bit, name in CONTENTS_FLAGS.items() if value & bit]


def summarize_brush_contents(brushes: np.ndarray) -> dict[str, Any]:
    unique_contents, brush_counts = np.unique(brushes["contents"], return_counts=True)
    raw_counts = []
    for contents, count in zip(unique_contents.tolist(), brush_counts.tolist()):
        raw_counts.append({
            "contents": int(contents),
            "hex": f"0x{int(contents) & 0xFFFFFFFF:08x}",
            "flags": content_flag_names(int(contents)),
            "count": int(count),
        })
    flag_counts = []
    all_contents = [int(v) for v in brushes["contents"].tolist()]
    for bit, name in CONTENTS_FLAGS.items():
        count = sum(1 for contents in all_contents if (contents & bit) != 0)
        if count:
            flag_counts.append({
                "flag": name,
                "bit": int(bit),
                "hex": f"0x{bit:08x}",
                "brush_count": int(count),
            })
    return {
        "raw_counts": raw_counts,
        "flag_counts": flag_counts,
        "known_flags": [
            {"flag": name, "bit": int(bit), "hex": f"0x{bit:08x}"}
            for bit, name in CONTENTS_FLAGS.items()
        ],
    }


def gamelump_id_to_text(lump_id: int) -> str:
    raw = int(lump_id).to_bytes(4, "little", signed=False)
    return raw[::-1].decode("ascii", errors="replace")


def parse_gamelumps(blob: bytes, header: dict[str, Any]) -> dict[str, Any]:
    data = lump_bytes(blob, header, 35)
    if len(data) < 4:
        return {"count": 0, "directory": [], "static_prop": None}
    lump_count = struct.unpack_from("<i", data, 0)[0]
    directory = []
    static_prop = None
    entry_size = 16
    for idx in range(max(0, lump_count)):
        off = 4 + idx * entry_size
        if off + entry_size > len(data):
            break
        lump_id, flags, version, fileofs, filelen = struct.unpack_from("<IHHII", data, off)
        entry = {
            "index": int(idx),
            "id": int(lump_id),
            "id_text": gamelump_id_to_text(lump_id),
            "id_text_little_endian": int(lump_id).to_bytes(4, "little", signed=False).decode("ascii", errors="replace"),
            "flags": int(flags),
            "version": int(version),
            "file_offset": int(fileofs),
            "file_length": int(filelen),
        }
        directory.append(entry)
        if entry["id_text"] == "sprp":
            static_prop = parse_static_prop_gamelump(blob, entry)
    return {"count": int(lump_count), "directory": directory, "static_prop": static_prop}


def parse_static_prop_gamelump(blob: bytes, entry: dict[str, Any]) -> dict[str, Any]:
    start = int(entry["file_offset"])
    end = start + int(entry["file_length"])
    data = blob[start:end]
    out: dict[str, Any] = {
        "id_text": entry["id_text"],
        "version": int(entry["version"]),
        "byte_count": len(data),
        "dict_count": None,
        "leaf_count": None,
        "prop_count": None,
        "model_preview": [],
        "parseable": False,
    }
    if len(data) < 4:
        out["error"] = "static prop gamelump too small for dict count"
        return out
    pos = 0
    dict_count = struct.unpack_from("<i", data, pos)[0]
    pos += 4
    dict_bytes = dict_count * 128
    if dict_count < 0 or pos + dict_bytes + 4 > len(data):
        out["error"] = "static prop dict section exceeds gamelump bounds"
        return out
    model_names = []
    for idx in range(dict_count):
        raw = data[pos + idx * 128 : pos + (idx + 1) * 128]
        model_names.append(raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace"))
    pos += dict_bytes
    leaf_count = struct.unpack_from("<i", data, pos)[0]
    pos += 4
    leaf_bytes = leaf_count * 2
    if leaf_count < 0 or pos + leaf_bytes + 4 > len(data):
        out["error"] = "static prop leaf section exceeds gamelump bounds"
        out["dict_count"] = int(dict_count)
        out["model_preview"] = model_names[:20]
        return out
    pos += leaf_bytes
    prop_count = struct.unpack_from("<i", data, pos)[0]
    out.update({
        "dict_count": int(dict_count),
        "leaf_count": int(leaf_count),
        "prop_count": int(prop_count),
        "model_preview": model_names[:20],
        "parseable": True,
    })
    return out


def parse_planes(blob: bytes, header: dict[str, Any]) -> np.ndarray:
    data = lump_bytes(blob, header, 1)
    if len(data) % 20 != 0:
        raise ValueError("planes lump size is not divisible by 20")
    dtype = np.dtype([("normal", "<f4", (3,)), ("dist", "<f4"), ("type", "<i4")])
    return np.frombuffer(data, dtype=dtype).copy()


def extract_face_triangles(vertices: np.ndarray, edges: np.ndarray, surfedges: np.ndarray, faces: np.ndarray, max_faces: int = 0) -> np.ndarray:
    tris: list[list[int]] = []
    face_count = len(faces) if max_faces <= 0 else min(len(faces), max_faces)
    for face in faces[:face_count]:
        first = int(face["firstedge"])
        count = int(face["numedges"])
        if count < 3 or first < 0 or first + count > len(surfedges):
            continue
        polygon: list[int] = []
        for se in surfedges[first : first + count]:
            se_i = int(se)
            edge = edges[abs(se_i)]
            vert_idx = int(edge[0] if se_i >= 0 else edge[1])
            if 0 <= vert_idx < len(vertices):
                polygon.append(vert_idx)
        if len(polygon) < 3:
            continue
        for i in range(1, len(polygon) - 1):
            a, b, c = polygon[0], polygon[i], polygon[i + 1]
            if a != b and b != c and c != a:
                tris.append([a, b, c])
    return np.asarray(tris, dtype=np.int32)


def face_polygon_indices(edges: np.ndarray, surfedges: np.ndarray, face: np.void) -> list[int]:
    first = int(face["firstedge"])
    count = int(face["numedges"])
    if count < 3 or first < 0 or first + count > len(surfedges):
        return []
    polygon: list[int] = []
    for se in surfedges[first : first + count]:
        se_i = int(se)
        if abs(se_i) >= len(edges):
            return []
        edge = edges[abs(se_i)]
        polygon.append(int(edge[0] if se_i >= 0 else edge[1]))
    return polygon


def extract_face_triangles_filtered(
    vertices: np.ndarray,
    edges: np.ndarray,
    surfedges: np.ndarray,
    faces: np.ndarray,
    skip_displacements: bool,
    max_faces: int = 0,
) -> np.ndarray:
    tris: list[list[int]] = []
    face_count = len(faces) if max_faces <= 0 else min(len(faces), max_faces)
    for face in faces[:face_count]:
        if skip_displacements and int(face["dispinfo"]) >= 0:
            continue
        polygon = face_polygon_indices(edges, surfedges, face)
        polygon = [idx for idx in polygon if 0 <= idx < len(vertices)]
        if len(polygon) < 3:
            continue
        for i in range(1, len(polygon) - 1):
            a, b, c = polygon[0], polygon[i], polygon[i + 1]
            if a != b and b != c and c != a:
                tris.append([a, b, c])
    return np.asarray(tris, dtype=np.int32)


def rotate_quad_to_start(quad: np.ndarray, start_position: np.ndarray) -> np.ndarray:
    idx = int(np.argmin(np.linalg.norm(quad - start_position.reshape(1, 3), axis=1)))
    return np.concatenate([quad[idx:], quad[:idx]], axis=0)


def build_displacement_mesh(
    base_vertices: np.ndarray,
    edges: np.ndarray,
    surfedges: np.ndarray,
    faces: np.ndarray,
    dispinfos: np.ndarray,
    dispverts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    vertices: list[np.ndarray] = []
    tris: list[list[int]] = []
    skipped: list[dict[str, Any]] = []
    used_faces = 0
    powers: dict[int, int] = {}

    for face_idx, face in enumerate(faces):
        disp_idx = int(face["dispinfo"])
        if disp_idx < 0:
            continue
        if disp_idx >= len(dispinfos):
            skipped.append({"face": face_idx, "reason": "dispinfo index out of range", "dispinfo": disp_idx})
            continue
        polygon_idx = face_polygon_indices(edges, surfedges, face)
        if len(polygon_idx) != 4 or any(idx < 0 or idx >= len(base_vertices) for idx in polygon_idx):
            skipped.append({"face": face_idx, "reason": "displacement base face is not a valid quad", "num_vertices": len(polygon_idx)})
            continue
        info = dispinfos[disp_idx]
        power = int(info["power"])
        side = (1 << power) + 1
        start = int(info["disp_vert_start"])
        count = side * side
        if power < 2 or power > 4 or start < 0 or start + count > len(dispverts):
            skipped.append({"face": face_idx, "reason": "invalid displacement vertex span", "power": power, "start": start})
            continue

        quad = rotate_quad_to_start(base_vertices[np.asarray(polygon_idx, dtype=np.int32)], info["start_position"].astype(np.float32))
        a, b, c, d = quad
        offset = len(vertices)
        block = dispverts[start : start + count]
        for y in range(side):
            v = y / float(side - 1)
            for x in range(side):
                u = x / float(side - 1)
                base = (
                    (1.0 - u) * (1.0 - v) * a
                    + u * (1.0 - v) * b
                    + u * v * c
                    + (1.0 - u) * v * d
                )
                dv = block[y * side + x]
                vertices.append((base + dv["vector"].astype(np.float32) * float(dv["dist"])).astype(np.float32))
        for y in range(side - 1):
            for x in range(side - 1):
                v00 = offset + y * side + x
                v10 = v00 + 1
                v01 = v00 + side
                v11 = v01 + 1
                tris.append([v00, v10, v11])
                tris.append([v00, v11, v01])
        used_faces += 1
        powers[power] = powers.get(power, 0) + 1

    if vertices:
        out_vertices = np.stack(vertices, axis=0).astype(np.float32)
        out_faces = np.asarray(tris, dtype=np.int32)
    else:
        out_vertices = np.zeros((0, 3), dtype=np.float32)
        out_faces = np.zeros((0, 3), dtype=np.int32)
    meta = {
        "dispinfo_count": int(len(dispinfos)),
        "dispvert_count": int(len(dispverts)),
        "used_displacement_faces": int(used_faces),
        "skipped_displacement_faces": int(len(skipped)),
        "skipped_preview": skipped[:20],
        "power_counts": {str(k): int(v) for k, v in sorted(powers.items())},
        "generated_vertices": int(len(out_vertices)),
        "generated_faces": int(len(out_faces)),
    }
    return out_vertices, out_faces, meta


def obj_bounds(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    mins = np.full(3, np.inf, dtype=np.float64)
    maxs = np.full(3, -np.inf, dtype=np.float64)
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            xyz = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
            mins = np.minimum(mins, xyz)
            maxs = np.maximum(maxs, xyz)
            count += 1
    return {"vertex_count": count, "mins": mins.tolist(), "maxs": maxs.tolist(), "size": (maxs - mins).tolist()}


def nav_bounds(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    mins = np.full(3, np.inf, dtype=np.float64)
    maxs = np.full(3, -np.inf, dtype=np.float64)
    count = 0
    for area in data.get("areas", {}).values():
        pts = [area.get("nw_corner"), area.get("se_corner")]
        for key in ("ne_z", "sw_z"):
            pass
        for pt in pts:
            if pt is None:
                continue
            xyz = np.asarray(pt, dtype=np.float64)
            mins = np.minimum(mins, xyz)
            maxs = np.maximum(maxs, xyz)
            count += 1
    return {"area_corner_count": count, "mins": mins.tolist(), "maxs": maxs.tolist(), "size": (maxs - mins).tolist()}


def parse_entity_preview(blob: bytes, header: dict[str, Any]) -> dict[str, Any]:
    data = lump_bytes(blob, header, 0)
    text = data.decode("utf-8", errors="ignore")
    classnames: dict[str, int] = {}
    for part in text.split('"classname"')[1:]:
        tokens = part.split('"')
        if len(tokens) >= 2:
            cls = tokens[1]
            classnames[cls] = classnames.get(cls, 0) + 1
    return {
        "byte_count": len(data),
        "classname_counts_top": sorted(classnames.items(), key=lambda x: x[1], reverse=True)[:30],
        "preview": text[:2000],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bsp", type=Path, required=True)
    ap.add_argument("--world-obj", type=Path, default=None)
    ap.add_argument("--navmesh", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-npz", type=Path, default=None)
    ap.add_argument("--max-face-export", type=int, default=0)
    ap.add_argument("--no-displacements", action="store_true", help="Keep legacy visual faces only in exported vertices/faces.")
    args = ap.parse_args()

    blob = args.bsp.read_bytes()
    header = parse_header(blob)
    vertices = parse_vertices(blob, header)
    edges = parse_edges(blob, header)
    surfedges = parse_surfedges(blob, header)
    faces = parse_faces(blob, header, 7)
    models = parse_models(blob, header)
    brushes = parse_brushes(blob, header)
    planes = parse_planes(blob, header)
    dispinfos = parse_dispinfos(blob, header)
    dispverts = parse_dispverts(blob, header)
    gamelumps = parse_gamelumps(blob, header)
    legacy_tris = extract_face_triangles(vertices, edges, surfedges, faces, max_faces=args.max_face_export)
    if args.no_displacements:
        export_vertices = vertices.astype(np.float32)
        export_faces = legacy_tris.astype(np.int32)
        disp_vertices = np.zeros((0, 3), dtype=np.float32)
        disp_faces = np.zeros((0, 3), dtype=np.int32)
        displacement_meta = {
            "enabled": False,
            "dispinfo_count": int(len(dispinfos)),
            "dispvert_count": int(len(dispverts)),
        }
    else:
        base_tris = extract_face_triangles_filtered(
            vertices, edges, surfedges, faces, skip_displacements=True, max_faces=args.max_face_export
        )
        disp_vertices, disp_faces_local, displacement_meta = build_displacement_mesh(
            vertices, edges, surfedges, faces, dispinfos, dispverts
        )
        displacement_meta["enabled"] = True
        export_vertices = np.concatenate([vertices.astype(np.float32), disp_vertices], axis=0)
        disp_faces = (disp_faces_local + len(vertices)).astype(np.int32) if len(disp_faces_local) else disp_faces_local
        export_faces = np.concatenate([base_tris.astype(np.int32), disp_faces], axis=0)

    counts = {}
    for key, size in STRUCT_SIZES.items():
        idx = {
            "plane": 1,
            "vertex": 3,
            "edge": 12,
            "surfedge": 13,
            "face": 7,
            "model": 14,
            "brush": 18,
            "brushside": 19,
            "texinfo": 6,
            "texdata": 2,
            "dispinfo": 26,
            "dispvert": 33,
        }[key]
        counts[key] = count_lump(header, idx, size)

    nonempty_lumps = [
        {k: lump[k] for k in ("index", "name", "offset", "length", "version", "fourcc")}
        for lump in header["lumps"]
        if lump["length"] > 0
    ]
    model_rows = []
    for idx, model in enumerate(models[:16]):
        model_rows.append({
            "index": idx,
            "mins": model["mins"].astype(float).tolist(),
            "maxs": model["maxs"].astype(float).tolist(),
            "origin": model["origin"].astype(float).tolist(),
            "firstface": int(model["firstface"]),
            "numfaces": int(model["numfaces"]),
            "headnode": int(model["headnode"]),
        })
    brush_contents_summary = summarize_brush_contents(brushes)
    out = {
        "kind": "bsp_static_geometry_inspection_v0",
        "bsp": str(args.bsp),
        "file_size": args.bsp.stat().st_size,
        "ident": header["ident"],
        "version": header["version"],
        "map_revision": header["map_revision"],
        "nonempty_lumps": nonempty_lumps,
        "counts": counts,
        "bsp_vertex_bounds": {
            "mins": vertices.min(axis=0).astype(float).tolist(),
            "maxs": vertices.max(axis=0).astype(float).tolist(),
            "size": (vertices.max(axis=0) - vertices.min(axis=0)).astype(float).tolist(),
        },
        "world_model": model_rows[0] if model_rows else None,
        "model_preview": model_rows,
        "brush_contents": brush_contents_summary,
        "brush_contents_counts": brush_contents_summary["raw_counts"],
        "gamelumps": gamelumps,
        "static_prop_gamelump": gamelumps["static_prop"],
        "plane_count": int(len(planes)),
        "face_triangle_count": int(len(legacy_tris)),
        "exported_static_mesh": {
            "vertices": int(len(export_vertices)),
            "faces": int(len(export_faces)),
            "base_visual_faces_without_displacements": int(len(export_faces) - len(disp_faces)),
            "displacement_faces": int(len(disp_faces)),
            "policy": "vertices/faces in the exported npz include displacement surfaces unless --no-displacements is used.",
        },
        "displacement_mesh": displacement_meta,
        "entity_preview": parse_entity_preview(blob, header),
        "reference_bounds": {
            "world_obj": obj_bounds(args.world_obj),
            "navmesh": nav_bounds(args.navmesh),
        },
        "assessment": {
            "parseable_vbsp": True,
            "has_collision_brushes": bool(len(brushes) > 0 and header["lumps"][19]["length"] > 0),
            "has_visual_faces": bool(len(faces) > 0 and len(vertices) > 0),
            "has_static_prop_gamelump": bool(gamelumps["static_prop"] is not None),
            "next_step": "Build BSP face/brush backed static memory and compare first-person depth against current OBJ renderer.",
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out_npz:
        args.out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.out_npz,
            vertices=export_vertices.astype(np.float32),
            faces=export_faces.astype(np.int32),
            base_vertices=vertices.astype(np.float32),
            base_faces=legacy_tris.astype(np.int32),
            displacement_vertices=disp_vertices.astype(np.float32),
            displacement_faces=disp_faces.astype(np.int32),
        )
    print(json.dumps({
        "out_json": str(args.out_json),
        "out_npz": str(args.out_npz) if args.out_npz else None,
        "version": out["version"],
        "counts": counts,
        "bounds": out["bsp_vertex_bounds"],
        "assessment": out["assessment"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
