import torch
import torch.nn as nn

from baselines.lingbot_cam.checkpoint import load_checkpoint, save_checkpoint
from baselines.lingbot_cam.lora import inject_lora, named_lora_parameters
from baselines.lingbot_cam.tests.test_lora import Model


def test_checkpoint_round_trip_restores_lora_optimizer_scheduler_and_step(tmp_path):
    model = Model().requires_grad_(False)
    inject_lora(model, rank=4, targets="attn,mlp", alpha=4, dropout=0)
    params = [p for _, p in named_lora_parameters(model)]
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    config = {
        "expert": "low", "manifest_sha256": "manifest",
        "base_identity": {"structural_sha256": "base"},
        "lora_config": {"rank": 4, "alpha": 4, "dropout": 0, "targets": "attn,mlp"},
    }
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler, step=1, config=config)
    before = {name: p.detach().clone() for name, p in named_lora_parameters(model)}
    for parameter in params:
        parameter.data.add_(1)
    step = load_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler, expected=config)
    assert step == 1
    assert all(torch.equal(before[name], parameter) for name, parameter in named_lora_parameters(model))
