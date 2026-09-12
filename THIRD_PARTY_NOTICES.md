# Third-party notices

OmniScene Forge's original project contributions are licensed under Apache 2.0.
That grant does not replace the licenses of third-party material. Existing source
attributions are retained. Model weights, datasets and external runtimes are not
included in the project's license grant.

| Material in this repository | Upstream attribution | License / notice |
|---|---|---|
| `vendor/lingbot/` | Modified LingBot/Wan tree from the original project upload | [Provided Apache 2.0 text](vendor/lingbot/LICENSE.txt); specific upstream notices below also apply |
| `vendor/lingbot/wan/modules/model_fast.py` | [Self-Forcing](https://github.com/guandeh17/Self-Forcing) functions, as attributed in its header | Apache 2.0; original attribution retained |
| `vendor/lingbot/wan/utils/fm_solvers*.py` and the Transformers-derived XLM-R implementation | Hugging Face Diffusers / Transformers, as attributed in source | Apache 2.0; original attribution retained |
| `vendor/lingbot/wan/modules/animate/clip.py` | [OpenAI CLIP](https://github.com/openai/CLIP) and [OpenCLIP](https://github.com/mlfoundations/open_clip), as attributed in its header | [CLIP MIT notice](licenses/CLIP-MIT.txt), [OpenCLIP MIT notice](licenses/OpenCLIP-MIT.txt) |
| `vendor/lingbot/wan/modules/animate/motion_encoder.py` | [LIA](https://github.com/wyhsirius/LIA), as attributed in its header | [CC BY-NC 4.0](licenses/LIA-CC-BY-NC-4.0.md), including its noncommercial condition |
| `vendor/lingbot/wan/utils/qwen_vl_utils.py` | `kq-chen/qwen-vl-utils`, as attributed in its header | Preserved as part of the provided LingBot tree; no claim of newly granted upstream rights |

The LIA-derived animation component is retained from the supplied vendor tree.
It is not used by the registered multiview routes and is not relicensed under
Apache 2.0. Do not interpret the repository's main license as permission for
commercial use of that component. Adaptation attribution is present in its source
header; OmniScene Forge has made no additional modifications to that file.

The CLIP, OpenCLIP and LIA license texts were retrieved from their official
repositories on 2026-09-12. This records their notices; it does not assert that the
vendor snapshot is identical to those repositories' current revisions.

The vendored T5 constructor was changed to defer CUDA-device selection until
construction. Project source-path changes are recorded in
[the complete source patch](provenance/source_patches.patch).

SCOPE and the separate official causal Fast v2 runtime remain external
dependencies. Their code, checkpoints and data are governed by their own terms;
this repository contains the project's integration code only.
