import torch

from baselines.scope.scope_bridge.model_loader import assert_actionmodule_gradients, configure_actionmodule_only


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.action_attn=torch.nn.Linear(2,2); self.ffn=torch.nn.Linear(2,2)


class Dit(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.blocks=torch.nn.ModuleList([Block(),Block()]); self.head=torch.nn.Linear(2,2)


def test_only_actionmodule_is_trainable_and_gets_gradient():
    dit=Dit(); manifest=configure_actionmodule_only(dit)
    assert all(item["trainable"] == (".action_attn." in f".{item['name']}.") for item in manifest)
    loss=sum(block.action_attn(torch.ones(1,2)).sum() for block in dit.blocks); loss.backward()
    assert_actionmodule_gradients(dit)
    assert all(p.grad is None for name,p in dit.named_parameters() if ".action_attn." not in f".{name}.")
