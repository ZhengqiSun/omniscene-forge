#!/usr/bin/env python3
"""Build a tier69h map manifest view that carries memory-mask surrogate fields."""
from __future__ import annotations
from runtime_paths import source_path
import argparse, json, time
from pathlib import Path
from typing import Any
REGION_MASK_KIND = "memory_dense_channel_3_surrogate_v0"
PROJECT_ROOT = Path(str(source_path('scene', '')))
WORKSPACE_ROOT = str(source_path('scene', ''))
DATA_ROOT = str(source_path('scene', ''))
PATH_KEYS = ("dense_path", "target_rgb_path", "meta_path", "qa_path")
REGION_MASK_POLICY = {
    "kind": REGION_MASK_KIND,
    "source": "dense_channel_3_other_player_mask_from_memory_player_capsules",
    "valid_for": "relative true-vs-shuffled Memory-mask region loss only",
    "not_valid_for": "teacher segmentation precision/recall, absolute presence, or position error",
}

def normalize_project_path(value: Any, project_root: Path) -> Any:
    if not isinstance(value, str) or not value:
        return value
    if value.startswith(WORKSPACE_ROOT):
        return DATA_ROOT + value[len(WORKSPACE_ROOT):]
    if value.startswith(DATA_ROOT):
        return value
    if value.startswith("output/") or value.startswith("tools/"):
        return str(project_root / value)
    return value

def build_view(src: Path, dst: Path, project_root: Path) -> dict[str, Any]:
    with src.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"{src}: manifest has no list samples")
    manifest["kind"] = str(manifest.get("kind", "tier69h_dense_combined_manifest_v0")) + "_memorymask_view_v0"
    manifest["source_manifest"] = str(src)
    manifest["memorymask_view_created_at_unix"] = time.time()
    manifest["map_memory_manifest_view_note"] = "Per-sample memory-mask surrogate fields added for region-loss teacher mask path."
    manifest["region_mask_kind"] = REGION_MASK_KIND
    manifest["region_mask_policy"] = REGION_MASK_POLICY
    manifest["teacher_qa_available"] = False
    if isinstance(manifest.get("source_manifests"), list):
        manifest["source_manifests"] = [normalize_project_path(v, project_root) for v in manifest["source_manifests"]]
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("sample is not an object")
        for key in PATH_KEYS:
            if key in sample:
                sample[key] = normalize_project_path(sample[key], project_root)
        sample["region_mask_kind"] = REGION_MASK_KIND
        sample["region_mask_policy"] = REGION_MASK_POLICY
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(dst)
    return {"src": str(src), "dst": str(dst), "sample_count": len(samples), "region_mask_kind": REGION_MASK_KIND}

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = ap.parse_args()
    print(json.dumps(build_view(args.src, args.dst, args.project_root), ensure_ascii=False, indent=2))
if __name__ == "__main__":
    main()
