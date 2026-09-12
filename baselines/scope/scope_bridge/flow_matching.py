from __future__ import annotations

import torch


def interpolate_clean_noise(clean: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Official DiffSynth FlowMatchScheduler.add_noise formula."""
    while sigma.ndim < clean.ndim:
        sigma = sigma.unsqueeze(-1)
    return (1.0 - sigma) * clean + sigma * noise


def flow_target(clean: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Official DiffSynth FlowMatchScheduler.training_target formula."""
    return noise - clean


def weighted_flow_mse(prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    loss = (prediction.float() - target.float()).square()
    if weight is not None:
        while weight.ndim < loss.ndim:
            weight = weight.unsqueeze(-1)
        loss = loss * weight
    return loss.mean()
