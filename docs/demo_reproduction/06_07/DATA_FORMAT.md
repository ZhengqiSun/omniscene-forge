# Demo 06/07 data format

All paths below are relative to the extracted data root, `assets/demos/06_07/`.

```text
samples/<06-or-07>/<team_player>/
  reference_output.mp4       Historical generation; not an inference input
  window00/
    sample.json             Source interval, seed, protocol and relative paths
    image.jpg               Original I2V context, 832 x 480
    prompt.txt
    poses.npy               float32 [81, 4, 4]
    intrinsics.npy          float32 [81, 4], fx/fy/cx/cy
    dense_manifest.jsonl    21 ordered records
    dense/00.npz ... 20.npz  float32 dense [7, 176, 320]
    gt.mp4                  Reference only, 832 x 480, 81 frames at 16 fps
    raw_player_frames.json  161 consecutive source-frame records
```

## Time and multiview alignment

For raw start index `r`, output frame `j` corresponds to raw index `r + 2*j`, for `j=0..80`. Dense item `k` corresponds to raw index `r + 8*k` and output frame `4*k`, for `k=0..20`. Source sampling is 32 Hz; output is 16 fps. An 81-frame video has a playback duration of 5.0625 seconds and a first-to-last sample timestamp span of 5.0 seconds.

The three views within each group share match, episode and raw index range. Keep the original raw JSON timing fields: source-array/video indices are not a general substitute for `tick`, `epochTime` or `frame_count`. Different groups are different scenes and cannot be treated as one synchronized event.

## Camera inputs

`poses.npy` is OpenCV-style camera-to-world: columns right, down, forward; translation uses Source engine world units. `intrinsics.npy` holds `[fx, fy, cx, cy]` in the supplied materialized camera convention. Preserve the arrays and original JPEG; do not infer new intrinsics from an assumed FOV or independently resize the inputs.

GT remains at 832 x 480. Historical generation uses `max_area=399360` with the legacy aspect-ratio calculation, yielding 832 x 464. Do not crop or stretch GT and call it the original file. Derived display comparisons must be labeled separately.

## Dense conditions

Read `np.load(path, allow_pickle=False)['dense']`. Any additional arrays in the NPZ are source metadata, not extra condition channels.

| Index | Meaning |
|---|---|
| 0 | Environment depth under the source normalization contract |
| 1 | Environment mesh-hit mask |
| 2 | Static-memory navigation/place semantics |
| 3 | Other-player capsule mask |
| 4 | Other-player capsule depth |
| 5 | Other-player yaw sine relative to ego |
| 6 | Other-player yaw cosine relative to ego |

The historical encoder is `img1_mask_img2_player_v0`. Its two three-channel images are packed from `[ch0,ch2,ch3]` and `[ch4,ch5,ch6]`, with the encoder's original value mappings. **Channel 1 is stored but not consumed by this packing.** This is not the current `full7_yawangle_v2` protocol; do not silently substitute it or resize the dense grid to 240 x 416.

The files provide replay-derived future conditions over the entire window. They do not demonstrate that a simulator predicted those future states. GT and `reference_output.mp4` are inspection targets, not additional inference inputs. Prepared dense inputs mean teacher depth/segmentation videos, BSP/OBJ geometry, a complete replay and state/interaction caches are not required to consume this sample data.

## Raw JSON example

Each `raw_player_frames.json` wraps the original selected ego-stream records with `source_rate_hz`, `original_array_start_index` and `frames`. Frame objects include the original timing, position, yaw/pitch, camera position/rotation, render transform, action, health, armor, money and equipment fields. Preserve nested types and source values. This is not full match-wide other-player telemetry or a complete arbitrary-replay preprocessing release.

## Provenance and split

06: match `aaebee4e89b94d0cac64445a13370419`, episode 4, raw 16-176, source label `val`.

07: match `8790bb3c174541378f8325d5ad609a23`, episode 3, raw 664-824, source label unspecified.

These are six manually selected historical windows from two matches, not an unbiased benchmark. The labels do not certify whole-training-lineage isolation. Per-file SHA256 hashes and source clip IDs are provided in `samples_manifest.json` and `samples_SHA256SUMS.txt` in the archive.
