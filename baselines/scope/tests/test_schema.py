import json

import pytest

from baselines.scope.scope_bridge.dataset import ScopeManifestDataset, assert_train_split
from baselines.scope.scope_bridge.schema import ScopeSampleV1


def row(**updates):
    value={"schema_id":"ScopeSampleV1","sample_id":"a","split":"train","initial_image":"a.png","scope_action_path":"a.parquet","prompt":"p","source_fps":32,"model_fps":20,"num_frames":81,"height":480,"width":832}
    value.update(updates); return value


def test_contract_and_inference_projection_hides_all_gt():
    sample=ScopeSampleV1.from_dict(row(target_video="future.mp4",gt_video="gt.mp4",dense_path="dense.pt",camera_poses="pose.npy"))
    projected=sample.model_projection("infer")
    assert "target_video" not in projected and "gt_video" not in projected and "dense_path" not in projected and "camera_poses" not in projected


def test_requires_exactly_one_image_source():
    with pytest.raises(ValueError): ScopeSampleV1.from_dict(row(raw_video="x.mp4",raw_start=0))


def test_unknown_field_rejected():
    with pytest.raises(ValueError): ScopeSampleV1.from_dict(row(secret="x"))


def test_test_training_rejected():
    with pytest.raises(ValueError): assert_train_split([ScopeSampleV1.from_dict(row(split="test"))])


def test_dataset_infer_does_not_stat_gt(tmp_path):
    image=tmp_path/"a.png"; image.write_bytes(b"x"); action=tmp_path/"a.parquet"; action.write_bytes(b"x")
    manifest=tmp_path/"m.jsonl"; manifest.write_text(json.dumps(row(initial_image="a.png",scope_action_path="a.parquet",gt_video="does-not-exist.mp4"))+"\n")
    item=ScopeManifestDataset(manifest,"infer")[0]
    assert set(item).isdisjoint({"gt_video","target_video","evaluator"})
