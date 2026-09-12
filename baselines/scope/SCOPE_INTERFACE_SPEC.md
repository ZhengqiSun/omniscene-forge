# SCOPE Interface Specification

## ScopeSampleV1 JSONL

Required model fields are `sample_id`, `split`, `prompt`, `source_fps`, `model_fps=20`, `num_frames` (81 or 101), `height=480`, and `width=832`, plus exactly one initial source and one action source:

- initial source: `initial_image`, or `raw_video` + `raw_start` (source-frame index);
- action source: `raw_action_path`, or converted `scope_action_path`.

Training additionally permits exactly one of `target_video` or `target_latent`. `target_latent` is accepted by schema but must not be enabled until checkpoint/VAE runtime confirms serialized shape and dtype.

Evaluator-only fields (`gt_video`, camera poses/intrinsics, group/view IDs) are isolated into `evaluator`. Dense/state, Map Memory, player masks, world events, future pose/state are also evaluator-only/forbidden. `ScopeManifestDataset(mode="infer")` does not return, stat, or decode any target/GT field.

## Time protocols

| Protocol | Generation | Normalized evaluation output | Status |
|---|---|---|---|
| `native81_20` | 81 @ 20 FPS, 480x832, 4.05 s by frame-count convention | unchanged | Confirmed official shape |
| `candidate101_20` | 101 @ 20 FPS, 480x832, 5.05 s | timestamp resample to 81 @ 16 FPS | Proposed; GPU quality smoke required |

All action/video resampling uses physical timestamps. Frame count must satisfy `T % 4 == 1`.

## Model boundaries

S0 reads only initial RGB, prompt and `[T,10]` action. S1 has the same inference interface. S1 may read a future target only in training. It never reads camera pose, intrinsics, other views, dense/state, masks or world metadata.

Fine-tuned checkpoints contain trainable state, AdamW state, scheduler, global step, Python/NumPy/Torch/CUDA RNG, manifest/calibration/base fingerprints, both git commits, effective config and the trainable-parameter manifest hash. Resume refuses a mismatch in any provenance key.

## Output contract

Each MP4 has a JSON sidecar containing sample/group/view identity where available, manifest/input/action/checkpoint hashes, model mode, prompt, seed, CFG, steps, shape/FPS, calibration hash, runtime, GPU/software and resampling flag. Existing files are skipped only when all deterministic metadata fields match; otherwise overwrite is refused.
