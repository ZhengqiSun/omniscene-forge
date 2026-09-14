# OmniScene Forge

**Scene-grounded multiview world generation.**

A research codebase for generating multiple views of a shared scene. It covers data and camera alignment, map and scene memory, condition rendering, model training, long-horizon generation, dynamics, multiview evaluation, and baselines. Interaction conditioning is one of the supported model variants.

[Project map](docs/PROJECT_MAP.md) · [Validation results](docs/VALIDATION.md) · [License](#license)

The repository brings together existing project implementations with their dependencies and file-level provenance. Model weights, research data, and runtime artifacts are supplied separately. Implementation relationships and validation limits are documented below.

## Project components

| Component | Coverage | Routes |
|---|---|---|
| Data | Video clips, camera/frame alignment, match splits, latent caches, multiview pairs | `data.*` |
| Maps and scenes | Static maps, episode player states, BSP/mesh projection, dense rendering | `geometry.*` |
| Conditioning | Dense, state v2, action caches, interaction sidecars | `conditions.*` |
| Training | Dense, base-expert, state, and interaction variants | `train.*` |
| Video generation | Dense/state/interaction sampling, autoregressive windows, history guidance | `infer.*` |
| Dynamics | Player dynamics v0–v3, motion fitting, long-horizon evaluation | `dynamics.*` |
| Scene rollout | Multiscene 10-ego generation, native ten-view dynamics, SRCDS integration | `rollout.*` |
| Evaluation | Co-visibility consistency, structure adherence, zero-visibility hallucination, AR drift, significance, ablations | `eval.*` |
| Baselines | Camera-only LingBot LoRA and an action-only SCOPE bridge | `baseline.*` |

The [project map](docs/PROJECT_MAP.md) describes the implemented connections and remaining gaps. In particular, the multiview pair loader is not yet connected to the current trainers. CLI imports, component tests, and end-to-end GPU execution are tracked separately in the [validation results](docs/VALIDATION.md).

## Repository layout

```text
multiview.py                  Project CLI and interaction configuration entrypoints
project_routes.py             Route catalog and isolated process launcher
tools/                        Current data, geometry, training, inference, and evaluation tools
pipelines/scene_rollout/       Scene and dynamics pipelines with their matching dependencies
baselines/lingbot_cam/         Camera-only LingBot baseline
baselines/scope/               SCOPE bridge and action conversion
vendor/lingbot/                Project-specific LingBot/Wan source and license
configs/interaction/          Portable interaction JSON recipes
tests/                        Interface, gradient, and multiview data tests
requirements/                 Base, data/evaluation, CUDA, and test dependencies
docs/                         Method relationships, usage, and validation limits
provenance/                   Source paths, selection records, and file hashes
scripts/                      Source assembly, path migration, and integrity checks
```

Some identically named modules differ between source families. Their matching directories and filenames are preserved. Routes start in the appropriate directory, and explicit dependencies across families remain intact. State modules, renderers, and checkpoints must match the protocol expected by each pipeline.

## Environment

Python 3.11 or newer is required. Use a dedicated environment with PyTorch and torchvision appropriate for your platform.

The following profile targets Linux with NVIDIA CUDA 12.4:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/cuda124.txt
python -m pip install -r requirements/base.txt -r requirements/data-eval.txt -r requirements/dev.txt
# Build GPU kernels for the target machine; CPU component tests do not need them.
# python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
```

For CPU development, install the appropriate CPU/platform builds of PyTorch and torchvision instead of the CUDA profile, then install the base, data/evaluation, and development requirements.

The recorded validation used macOS, Python 3.11, and PyTorch 2.6 on CPU. The Linux GPU environment has not been revalidated. SCOPE and the separate causal LingBot Fast v2 experiments require their matching external source trees and model environments; see [external assets](docs/ASSETS.md).

## Usage

List the available routes and inspect their original arguments:

```bash
python multiview.py list
python multiview.py run data.multiview -- --help
python multiview.py run train.state -- --help
python multiview.py run rollout.multiscene -- prepare --help
python multiview.py run baseline.lingbot-train -- --help
```

`run ROUTE -- ARGUMENTS` forwards arguments and subcommands to the selected tool. Place the launcher's `--dry-run` before the separator to print the command without executing it. This mode does not automatically launch multiple GPUs; use the corresponding trainer's torchrun interface for distributed training. Tool arguments containing relative paths are resolved from the selected route directory, so absolute data and output paths are recommended.

```bash
python multiview.py run train.state --dry-run -- train --out-dir /absolute/path/run
python multiview.py run rollout.dynamics -- self-test
```

The interaction variant also provides JSON recipes. Paths in these configurations are resolved relative to the configuration file:

```bash
cp configs/interaction/smoke.example.json configs/interaction/smoke.local.json
# Set the real asset paths, then inspect the launch command.
python multiview.py interaction-smoke --config configs/interaction/smoke.local.json --dry-run
```

See the [interaction guide](docs/INTERACTION.md), [pipeline inputs and outputs](docs/PROJECT_MAP.md), and baseline READMEs for details. Models, training data, maps, manifests, and engine installations must be supplied separately. Set `MULTIVIEW_ASSETS` to override the asset root. Paths embedded in existing manifests must be accessible on the target machine.

## Validation and provenance

### Demo 06/07 data preview

[Six historical demo views](docs/demo_reproduction/06_07/README.md) are available as a separate Release download, with context images, camera arrays, dense conditions, GT and historical reference videos. The accompanying records identify their shared LOW2700/HIGH2850 checkpoints. Project checkpoint binaries are **not** included; this is a data-format/protocol preview, not a GPU-validated inference release.

```bash
python scripts/check_codebase.py --help-smoke
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python multiview.py run rollout.dynamics -- self-test
```

The recorded checks include 58 passing tests, 53 successful CLI help checks, and four dynamics self-tests. End-to-end GPU training, generation, and research metrics have not been reproduced in this repository.

The [provenance record](docs/PROVENANCE.md) documents source selection and modifications. Historical experiment reports are not treated as evidence that the current codebase has reproduced those runs. Detailed research notes under `docs/` are currently in Chinese.

## License

Original project contributions are released under the [Apache License 2.0](LICENSE). Third-party code retains its original licenses and attribution; see [third-party notices](THIRD_PARTY_NOTICES.md).

`vendor/lingbot/wan/modules/animate/motion_encoder.py` is attributed to LIA and retains the noncommercial conditions of CC BY-NC 4.0. This optional animation module is not used by the registered multiview routes. Model weights and external datasets are subject to their own terms.
