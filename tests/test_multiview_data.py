from copy import deepcopy
import numpy as np
import pytest

from build_multiview_consistency_manifest_v1 import endpoint_id, positive_gate
from multiview_consistency_data_v1 import DataContractError, load_camera_arrays
from canonical_match_split_v0 import FROZEN_HELDOUT
from canonical_match_split_v1 import EXPANSION_HELDOUT, match_to_split_v1


def test_positive_pair_requires_all_21_world_ticks_and_frame_counts():
    left={'hash':'session','game_id':'match','episode':'episode','raw_start':100,'player_stem':'ego1','map_memory_raw_frame_indices':list(range(100,261,8))}
    right={**left,'player_stem':'ego2'}
    signature={'valid':True,'frame_counts':list(range(100,261,8)),'world_ticks':list(range(1000,1161,8))}
    sigs={endpoint_id(left):deepcopy(signature),endpoint_id(right):deepcopy(signature)}
    assert positive_gate(left,right,sigs)[0]
    sigs[endpoint_id(right)]['world_ticks'][10]+=1
    assert positive_gate(left,right,sigs)[1]=='positive_tick_mismatch'
    sigs[endpoint_id(right)]=deepcopy(signature)
    sigs[endpoint_id(right)]['frame_counts'][10]+=1
    assert positive_gate(left,right,sigs)[1]=='positive_frame_count_mismatch'


def test_camera_selection_preserves_alignment_and_rejects_nonfinite(tmp_path):
    poses=np.tile(np.eye(4,dtype=np.float32),(81,1,1));poses[:,0,3]=np.arange(81)
    intr=np.tile(np.array([400,400,416,240],dtype=np.float32),(81,1))
    np.save(tmp_path/'poses.npy',poses);np.save(tmp_path/'intr.npy',intr)
    endpoint={'poses':str(tmp_path/'poses.npy'),'intrinsics':str(tmp_path/'intr.npy')}
    selected,ki,_=load_camera_arrays(endpoint,list(range(0,81,4)))
    np.testing.assert_array_equal(selected[:,0,3],np.arange(0,81,4))
    assert ki.shape==(21,4)
    poses[40,0,0]=np.nan;np.save(tmp_path/'poses.npy',poses)
    with pytest.raises(DataContractError,match='poses_nonfinite'):
        load_camera_arrays(endpoint,list(range(0,81,4)))


def test_split_v1_preserves_original_heldout_and_declared_expansion():
    for match,split in {**FROZEN_HELDOUT,**EXPANSION_HELDOUT}.items():
        assert match_to_split_v1(match)==split
