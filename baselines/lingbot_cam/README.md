# LingBot camera-only baseline

This directory contains the project's camera-only LoRA training, inference, checkpoint pairing, manifest validation, and tests. It uses the bundled `vendor/lingbot` implementation. Base model and LoRA weights are supplied separately.

Run these commands from the repository root:

```bash
python multiview.py run baseline.lingbot-validate -- --help
python multiview.py run baseline.lingbot-train -- --help
python multiview.py run baseline.lingbot-infer -- --help
```

Package invocation is also supported: `python -m baselines.lingbot_cam.train ...`.

`schema.py` defines `lingbot_sample_v1`: exactly one initial image or raw video source, explicit poses and intrinsics, timing and dimensions, and a prompt or text cache. Training also requires a target. Privileged conditions such as dense, state, and world events are rejected.

Tests cover LoRA gradient boundaries, checkpoint restoration and base identity, media encoding interfaces, and the schema. Real GPU training and generation have not been reproduced in the current validation environment.
