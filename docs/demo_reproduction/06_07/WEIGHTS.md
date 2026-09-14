# Checkpoint references for demos 06 and 07

Both groups use the **same pair of project checkpoints**. Every view within a group uses that pair. LOW and HIGH refer to the two noise experts, not separate demo groups.

These are expected local paths relative to the repository root. **They are not download links; neither checkpoint binary is distributed by this data preview.**

| Expert | Step | Expected file | Bytes |
|---|---|---|---|
| LOW | 2700 | `assets/checkpoints/demo-06-07/historical_low2700.pt` | 1,044,200,842 |
| HIGH | 2850 | `assets/checkpoints/demo-06-07/historical_high2850.pt` | 1,044,201,968 |

SHA256 for LOW:

```text
41a4befe41c8fb36b3a6016f681b52fce08ae0034842dff8aa22a755a568a155
```

SHA256 for HIGH:

```text
fb23a3f4badf122672113016436e674e34878338604ef7836cb215c8ba09db23
```

These hashes identify the inference-only exports, not the larger original training files. Each export preserves all 1,122 inference tensors, including adapter, LoRA and dense encoder projection; optimizer state and training-only configuration were removed. No quantization or retraining was performed. Hashes, sizes, and CPU tensor equality were recorded during local preparation. A new GPU reproduction is not claimed.

The machine-readable `weights.json` sets `download_url` to `null` for each project checkpoint. Obtain authorized copies from the maintainers separately; do not replace them with a newer checkpoint and call the result a reproduction of these historical outputs. No internal server paths are required or published.

## Official Base dependency

The project exports are not full standalone models. They require [robbyant/lingbot-world-base-cam](https://huggingface.co/robbyant/lingbot-world-base-cam/tree/6fc824ffc338d64c97c77e2eb8c0f4cfc24d82bd), pinned to revision:

```text
6fc824ffc338d64c97c77e2eb8c0f4cfc24d82bd
```

Supply both base noise experts, T5/text encoder, tokenizer and Wan VAE, approximately 160.25 GB of required files. A suggested local root is `assets/models/lingbot-world-base-cam/`. This is the Base camera model, **not** a Fast checkpoint. The Base remains external and subject to its upstream terms.

## Historical settings

Both groups: UniPC, 70 sampling steps, shift 3, CFG 5 for both experts, 21 latent frames, chunk size 3, 16 fps, max area 399360, legacy dense packing, no state or interaction condition. Seed is 20260701 for 06 and 20260629 for 07. The new JSON records describe these settings but do not add a runnable inference integration to this repository.
