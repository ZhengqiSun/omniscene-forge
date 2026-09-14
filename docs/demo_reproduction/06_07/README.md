# Demo 06/07: data samples and checkpoint references

This data-only preview contains two historical, author-selected multiview examples: six views, six input windows, and one shared LOW/HIGH checkpoint pair. **Checkpoint binaries are not included and no public download URL for those project checkpoints is provided.** This is not a validated one-command inference release.

## Download and inspect

Download `solaris_demo_samples_06_07_v1.zip` and `SHA256SUMS.txt` from the [06/07 data release](https://github.com/ZhengqiSun/omniscene-forge/releases/tag/demo-repro-06-07-v1). The ZIP contains data and metadata only, with no executable code or model weights.

From the repository root, after placing the downloaded files there:

```bash
sha256sum -c SHA256SUMS.txt
mkdir -p assets/demos/06_07
unzip solaris_demo_samples_06_07_v1.zip -d assets/demos/06_07
cd assets/demos/06_07
sha256sum -c samples_SHA256SUMS.txt
```

`assets/` is intentionally ignored by Git. The extracted `samples/06/` and `samples/07/` directories contain three views each. `metadata/` contains the protocol descriptions and weight references; `docs/` contains the data-format and checkpoint guides.

| Demo | Views | Source interval | Seed | Reference generation |
|---|---|---|---|---|
| 06: visibility transition | T2/P0003, P0006, P0011 | `aaebee4e89b94d0cac64445a13370419`, episode 4, raw indices 16-176 | 20260701 | 832 x 464, 81 frames, 16 fps |
| 07: moving cameras and players | T3/P0002, P0004, P0008 | `8790bb3c174541378f8325d5ad609a23`, episode 3, raw indices 664-824 | 20260629 | 832 x 464, 81 frames, 16 fps |

For 06, inspect how other players become visible as the views evolve. For 07, inspect corresponding player positions and poses across moving viewpoints. These are viewing intentions, not claims of perfect consistency or a measured causal improvement. The [previously uploaded composite videos and original author descriptions](https://github.com/ZhengqiSun/multiview-world-model/tree/d0a55ff546ae1dc9efac0f3500368cdf81de0685/assets/video/selected-20260905) are not duplicated in this archive.

## What is included

Each view supplies its original context JPEG, prompt, camera-to-world poses, intrinsics, 21 dense-condition NPZ files, a dense manifest, GT video, raw ego-stream JSON excerpt, sample metadata, and historical generated reference video. There are 180 original sample files: 126 dense NPZs, six GT videos, six reference generations, and 42 other input/metadata files. No state or interaction cache is required by these examples.

- [Data format and timing](DATA_FORMAT.md)
- [Required checkpoint files and Base dependency](WEIGHTS.md)
- [Demo 06 protocol](../../../configs/demo_reproduction/06_07/demo06.json)
- [Demo 07 protocol](../../../configs/demo_reproduction/06_07/demo07.json)
- [Sample/file manifest](../../../provenance/demo_reproduction/06_07/samples_manifest.json)

## Validation and compatibility boundary

The released input files are byte-preserved from the prepared sample packet. The six camera arrays, 126 dense arrays, frame mappings, GT/context alignment, and all 12 videos are checked locally. The two referenced inference-only checkpoints were previously checked tensor-by-tensor against their original training checkpoints; this release publishes their identity, not their bytes.

The JSON recipes are **protocol metadata, not configurations accepted by an existing launcher**. They use repository-root-relative paths; sample-local manifests resolve paths relative to their own window directory. No implicit path rewriting is promised.

These examples require the historical `img1_mask_img2_player_v0` dense packing and `max_area=399360`, without forcing the context image to a new aspect ratio. The recorded output is 832 x 464, while the context JPEG and GT are 832 x 480. The repository's later sampler behavior must not silently replace this historical protocol. At the reviewed base commit `f4f42baa444dd33acfff26dbac2b93b5a207fcd0`, the historical sampler dependency `qxq_sample_base_dense_v0.py` is absent and the val/test sampler includes a later size adjustment. This data-only change does not repair or reroute those scripts. Fresh-environment GPU reproduction and resource requirements remain unverified.

06 retains its historical `val` label; 07 has no recorded split label. Neither establishes that its match was unseen by every checkpoint ancestor. Future dense conditions come from replay/world-state information, not from a closed-loop simulator. Raw JSON excerpts are format examples, not the complete match telemetry needed to regenerate every dense condition.

## Data and model terms

This preview does not assign a new license to replay-derived data or project model weights. Do not infer that the repository's source-code Apache-2.0 license grants rights to those separate assets. The official Base model retains its upstream terms. Obtain applicable permission before reuse or redistribution. No credentials, internal storage paths, private audit files, or training-state checkpoints are included.
