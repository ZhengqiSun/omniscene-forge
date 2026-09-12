from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schema import load_manifest, manifest_sha256, validate_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LingBotSampleV1 JSONL without loading model weights.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=["train", "infer"], required=True)
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--deep", action="store_true", help="Load target latent payloads and validate tensor shapes.")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    samples = load_manifest(args.manifest)
    wanted = set(args.split)
    samples = [sample for sample in samples if not wanted or sample.split in wanted]
    failures = []
    for sample in samples:
        errors = validate_sample(sample, mode=args.mode, deep=args.deep)
        if errors:
            failures.append({"sample_id": sample.sample_id, "errors": errors})
    report = {
        "schema_id": "lingbot_manifest_validation_v1",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha256(args.manifest),
        "mode": args.mode,
        "sample_count": len(samples),
        "failed_count": len(failures),
        "failures": failures,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
