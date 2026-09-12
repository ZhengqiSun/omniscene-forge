# SCOPE action-only baseline bridge

S0 uses official weights for zero-shot inference. S1 trains only `blocks.*.action_attn.*`. This directory contains the project's official-model loading boundary, flow matching, action conversion and calibration, checkpoints, and data validation. Official SCOPE source code and model weights must be supplied separately.

Run these commands from the repository root:

```bash
python multiview.py run baseline.scope-actions -- --help
python multiview.py run baseline.scope-validate -- --help
python multiview.py run baseline.scope-train -- --help
python multiview.py run baseline.scope-infer -- --help
```

Model inputs are initial RGB, a prompt, and per-frame actions. Camera, dense, state, and other-view information stays in the evaluator section and is not passed to the model. CSGO conversion requires explicit axis signs, and mouse gain must be fitted on the training split. See the [interface specification](SCOPE_INTERFACE_SPEC.md).

`--scope-repo` points to the official source tree; `--model-dir` points to the weights. Use an isolated environment matching that official version. The project's `vendor/lingbot` is not a substitute for the SCOPE runtime.

CPU tests cover the schema, actions, model parameter boundaries, and checkpoints. The official model has not been executed in the current validation environment.
