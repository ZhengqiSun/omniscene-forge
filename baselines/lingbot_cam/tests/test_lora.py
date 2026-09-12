import torch
import torch.nn as nn

from baselines.lingbot_cam.lora import assert_only_lora_trainable, inject_lora


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention(8)
        self.cross_attn = Attention(8)
        self.ffn = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])


def test_all_base_parameters_stay_frozen():
    model = Model().requires_grad_(False)
    report = inject_lora(model, rank=4, targets="attn,mlp", alpha=4, dropout=0)
    audit = assert_only_lora_trainable(model)
    assert report["wrapped_linear_count"] == 20
    assert audit["parameter_count"] == 2 * 4 * ((8 + 8) * 8 + (8 + 16) * 2)
    assert all("lora_" in name for name in audit["names"])
