import torch

from baselines.scope.scope_bridge.flow_matching import flow_target, interpolate_clean_noise, weighted_flow_mse


def test_official_flow_formulas():
    clean=torch.ones(2,3); noise=torch.zeros(2,3); sigma=torch.tensor([0.,1.])
    mixed=interpolate_clean_noise(clean,noise,sigma)
    assert torch.equal(mixed[0],clean[0]) and torch.equal(mixed[1],noise[1])
    assert torch.equal(flow_target(clean,noise),-clean)
    assert weighted_flow_mse(-clean,-clean).item()==0
