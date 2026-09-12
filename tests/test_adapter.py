import dataclasses

import torch
from torch import nn

from memory_dense_wan_adapter_v0 import MemoryDenseAdapterConfig, WanBlockMemoryDenseWrapper


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8,8)

    def forward(self,x,**kwargs):
        return self.linear(x)


def test_joint_checkpoint_preserves_outputs_and_condition_gradients():
    torch.manual_seed(4)
    cfg = MemoryDenseAdapterConfig(cond_dim=4,adapter_hidden_dim=6)
    direct = WanBlockMemoryDenseWrapper(Block(),8,cfg)
    recompute = WanBlockMemoryDenseWrapper(Block(),8,dataclasses.replace(cfg,activation_checkpoint_adapter=True))
    # Use nonzero adapter weights so this also exercises condition gradients.
    with torch.no_grad():
        for p in direct.parameters():
            p.uniform_(-0.2,0.2)
    recompute.load_state_dict(direct.state_dict())
    x = torch.randn(1,5,8)
    condition = torch.randn(1,5,4)
    outputs, grads = [], []
    for model in [direct,recompute]:
        model.train()
        cond = condition.clone().requires_grad_()
        out = model(x,dit_cond_dict={cfg.cond_key:cond})
        out.square().sum().backward()
        outputs.append(out)
        grads.append([cond.grad] + [p.grad for p in model.parameters()])
    torch.testing.assert_close(outputs[0],outputs[1])
    assert grads[0][0].norm() > 0
    for a,b in zip(*grads):
        torch.testing.assert_close(a,b)
