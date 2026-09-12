from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from baselines.scope.scope_bridge.schema import load_manifest, manifest_sha256, validate_paths


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--manifest", required=True); p.add_argument("--mode", choices=("train", "infer"), required=True)
    p.add_argument("--split", action="append", default=[]); p.add_argument("--json-out"); p.add_argument("--deep",action="store_true")
    args = p.parse_args(); wanted = set(args.split); samples = load_manifest(args.manifest)
    samples = [s for s in samples if not wanted or s.split in wanted]
    failures=[]
    for s in samples:
        errors=validate_paths(s,args.manifest,args.mode)
        if args.deep:
            from PIL import Image
            from baselines.scope.scope_bridge.schema import resolve_path
            if s.initial_image:
                try:
                    with Image.open(resolve_path(args.manifest,s.initial_image)) as im:
                        if im.size != (s.width,s.height): errors.append(f"initial_image size={im.size}, expected={(s.width,s.height)}")
                except Exception as exc: errors.append(f"initial_image decode failed: {exc}")
            if args.mode=="train" and s.target_video:
                try:
                    raw=subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-count_frames","-show_entries","stream=width,height,r_frame_rate,nb_read_frames","-of","json",str(resolve_path(args.manifest,s.target_video))],text=True)
                    stream=json.loads(raw)["streams"][0]
                    if (int(stream["width"]),int(stream["height"])) != (s.width,s.height): errors.append(f"target size mismatch: {stream}")
                    if int(stream.get("nb_read_frames",-1)) != s.num_frames: errors.append(f"target frame count mismatch: {stream}")
                except Exception as exc: errors.append(f"target ffprobe failed: {exc}")
        if errors: failures.append({"sample_id":s.sample_id,"errors":errors})
    report = {"schema_id": "ScopeManifestValidationV1", "manifest": str(Path(args.manifest).resolve()), "manifest_sha256": manifest_sha256(args.manifest), "mode": args.mode, "sample_count": len(samples), "failed_count": len(failures), "failures": failures}
    text = json.dumps(report, indent=2); print(text)
    if args.json_out: Path(args.json_out).write_text(text + "\n")
    if failures: raise SystemExit(2)


if __name__ == "__main__": main()
