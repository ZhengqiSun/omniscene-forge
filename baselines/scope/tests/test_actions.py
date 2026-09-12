import numpy as np

from baselines.scope.actions.resample_actions import resample
from baselines.scope.actions.calibrate_mouse import assert_train_only
from types import SimpleNamespace
import pytest


def test_diagonal_movement_norm_is_bounded():
    n=32; move=np.ones((n,2)); out=resample(move,np.zeros((n,2)),np.zeros((n,6)),32,20,20,1)
    assert np.linalg.norm(out.sticks[:,:2],axis=1).max() <= 1.000001


def test_short_discrete_event_survives_32_to_20():
    n=32; buttons=np.zeros((n,6)); buttons[7,0]=1
    out=resample(np.zeros((n,2)),np.zeros((n,2)),buttons,32,20,20,1)
    assert out.keyboard[:,0].sum() >= 1


def test_mouse_deltas_are_integrated_not_interpolated():
    n=32; mouse=np.ones((n,2)); out=resample(np.zeros((n,2)),mouse,np.zeros((n,6)),32,20,20,None)
    assert np.isclose(out.sticks[:,2:].sum(),64)


def test_action_shapes_and_ranges():
    n=130; out=resample(np.zeros((n,2)),np.zeros((n,2)),np.zeros((n,6)),32,20,81,1)
    assert out.keyboard.shape==(81,6) and out.sticks.shape==(81,4) and out.sticks.dtype==np.float32


def test_mouse_calibration_rejects_test_rows():
    with pytest.raises(ValueError): assert_train_only([SimpleNamespace(split="test")])
