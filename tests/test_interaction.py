import json

import numpy as np
import pytest
import torch

from interaction_channels_v0 import build_interaction_channels
from memory_dense_interaction_adapter_v1 import InteractionTokenProjector, load_interaction_tensor
from memory_dense_state_adapter_v0 import verify_state_projection_contract, STATE_BUILD_REPORT_NAME
from sample_memory_dense_adapter_interaction_eval_v1 import choose_shuffled_interaction_clip


def test_event_producer_consumer_gradient_and_checkpoint(tmp_path):
    frames = [{'action':a} for a in [
        {'weapon_slot':'AK-47'}, {'weapon_slot':'AK-47','fire':True},
        {'weapon_slot':'AK-47','reload':True}, {'weapon_slot':'HE Grenade','fire':True},
        {'weapon_slot':'HE Grenade'}]]
    arrays = build_interaction_channels(frames,[0,2,4])
    np.testing.assert_array_equal(arrays['ego_fire_event'],[0,1,0])
    np.testing.assert_array_equal(arrays['ego_throw_event'],[0,0,1])
    path = tmp_path / 'state.npz'
    np.savez(path,**arrays)
    kwargs = dict(frame_indices=[0,1,2],device=torch.device('cpu'),dtype=torch.float32)
    features = load_interaction_tensor({'state_cache':str(path)},**kwargs)
    assert features.shape == (3,134)
    projector = InteractionTokenProjector()
    output = projector(features,target_token_hw=(2,3))
    assert output.shape == (1,18,128) and torch.count_nonzero(output) == 0
    optimizer = torch.optim.SGD(projector.parameters(),lr=0.1)
    (output - 1).square().mean().backward()
    assert projector.proj.weight.grad.norm() > 0
    optimizer.step()
    output = projector(features,target_token_hw=(2,3))
    assert torch.count_nonzero(output) > 0
    restored = InteractionTokenProjector()
    restored.load_state_dict(projector.state_dict())
    torch.testing.assert_close(output,restored(features,target_token_hw=(2,3)))
    arrays['ego_fire_weapon_onehot'][1] = 0
    np.savez(path,**arrays)
    with pytest.raises(ValueError,match='ego_fire_weapon_onehot'):
        load_interaction_tensor({'state_cache':str(path)},**kwargs)


def test_projection_contract_distinguishes_unknown_from_valid(tmp_path):
    manifest = tmp_path / 'state.jsonl'
    manifest.write_text('')
    assert verify_state_projection_contract(manifest,{},strict=True)['status'] == 'unknown'
    report = tmp_path / STATE_BUILD_REPORT_NAME
    report.write_text(json.dumps({'projection_params':{'far':3000.0,'pitch_sign':1.0}}))
    assert verify_state_projection_contract(manifest,{},strict=True)['status'] == 'ok'
    report.write_text(json.dumps({'projection_params':{'far':4096.0,'pitch_sign':-1.0}}))
    with pytest.raises(ValueError,match='projection contract'):
        verify_state_projection_contract(manifest,{},strict=True)


def test_shuffled_condition_preserves_split():
    records = [{'clip_id':k,'split':v} for k,v in [('a','val'),('b','val'),('c','train')]]
    rows = {k:{} for k in 'abc'}
    assert choose_shuffled_interaction_clip(records,rows,current_clip_id='a',split='val',offset=7) == 'b'
    with pytest.raises(ValueError):
        choose_shuffled_interaction_clip(records,rows,current_clip_id='c',split='train',offset=7)
