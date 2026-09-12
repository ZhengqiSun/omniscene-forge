from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from baselines.lingbot_cam.media import load_initial_image
from baselines.lingbot_cam.schema import LingBotSampleV1


def test_raw_inference_decodes_only_initial_frame():
    sample = LingBotSampleV1(
        sample_id="raw", split="test", poses=Path("poses.npy"), intrinsics=Path("intrinsics.npy"),
        num_frames=81, fps=16, height=480, width=832, raw_video=Path("video.mp4"),
        raw_start=2446, raw_stride=2, prompt="prompt",
    )
    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.read.return_value = (True, np.zeros((480, 832, 3), dtype=np.uint8))
    with patch("baselines.lingbot_cam.media.cv2.VideoCapture", return_value=capture):
        image = load_initial_image(sample)
    capture.set.assert_called_once()
    capture.read.assert_called_once_with()
    assert image.shape == (3, 480, 832)
