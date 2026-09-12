import json
from pathlib import Path

import pytest

from baselines.lingbot_cam.schema import LingBotSampleV1, ManifestError


def row(tmp_path: Path):
    return {
        "schema_id": "lingbot_sample_v1", "sample_id": "sample-1", "split": "train",
        "initial_image": str(tmp_path / "image.jpg"), "target_latent": str(tmp_path / "latent.pt"),
        "poses": str(tmp_path / "poses.npy"), "intrinsics": str(tmp_path / "intrinsics.npy"),
        "text_cache": str(tmp_path / "text.pt"), "num_frames": 81, "fps": 16,
        "height": 480, "width": 832,
    }


def test_parse_minimal_row(tmp_path):
    sample = LingBotSampleV1.from_row(row(tmp_path), tmp_path)
    assert sample.sample_id == "sample-1"
    assert sample.target_latent == tmp_path / "latent.pt"


@pytest.mark.parametrize("key", ["dense_cache", "state_cache", "player_mask", "world_events"])
def test_reject_privileged_fields(tmp_path, key):
    value = row(tmp_path)
    value[key] = "/forbidden"
    with pytest.raises(ManifestError, match="privileged"):
        LingBotSampleV1.from_row(value, tmp_path)


def test_condition_source_is_exclusive(tmp_path):
    value = row(tmp_path)
    value["raw_video"] = "/video.mp4"
    value["raw_start"] = 0
    value["raw_stride"] = 2
    with pytest.raises(ManifestError, match="exactly one"):
        LingBotSampleV1.from_row(value, tmp_path)


def test_prompt_and_text_cache_can_coexist_for_train_and_infer(tmp_path):
    value = row(tmp_path)
    value["prompt"] = "hello"
    sample = LingBotSampleV1.from_row(value, tmp_path)
    assert sample.prompt == "hello"
    assert sample.text_cache == tmp_path / "text.pt"
