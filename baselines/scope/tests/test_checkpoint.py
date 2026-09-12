import pytest
import torch

from baselines.scope.scope_bridge.checkpoint import load_checkpoint, save_checkpoint


def metadata(value="a"):
    return {k:value for k in ("manifest_sha256","action_calibration_sha256","base_checkpoint_fingerprint","scope_commit","project_commit","trainable_scope_sha256")}


def test_checkpoint_resume_guard(tmp_path):
    model=torch.nn.Linear(2,1); opt=torch.optim.AdamW(model.parameters()); sched=torch.optim.lr_scheduler.ConstantLR(opt)
    path=tmp_path/"c.pt"; save_checkpoint(path,model,opt,sched,2,metadata())
    assert load_checkpoint(path,model,opt,sched,metadata())==2
    bad=metadata(); bad["manifest_sha256"]="wrong"
    with pytest.raises(ValueError): load_checkpoint(path,model,opt,sched,bad)
