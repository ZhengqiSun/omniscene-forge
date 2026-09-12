#!/usr/bin/env python3

import unittest

import torch
import torch.nn as nn

from lingbot_fast_v2_dense_direct_tune_v0 import (
    COND_KEY,
    DenseDirectTuneConfig,
    FastV2DenseProjection,
    apply_dense_mode,
    audit_trainables,
    checkpoint_envelope,
    gradient_audit,
    inject_dense_direct_tune,
    load_checkpoint_envelope,
    set_dense_enabled,
)


class IdentityBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.base = nn.Linear(dim, dim)

    def forward(self, hidden, **_kwargs):
        return self.base(hidden)


class TinyModel(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList([IdentityBlock(dim) for _ in range(3)])


class DenseDirectTuneTest(unittest.TestCase):
    def config(self):
        return DenseDirectTuneConfig(
            cond_dim=5,
            hidden_dim=8,
            adapter_hidden_dim=4,
            block_indices=(0, 2),
            token_height=2,
            token_width=3,
            chunk_size=2,
        )

    def test_modes_and_projection_shape(self):
        config = self.config()
        projection = FastV2DenseProjection(config)
        dense = torch.arange(2 * 7 * 4 * 5, dtype=torch.float32).reshape(2, 7, 4, 5)
        true, _ = projection(dense, mode="true")
        shuffled_dense, audit = apply_dense_mode(dense, "shuffled")
        shuffled, _ = projection(dense, mode="shuffled")
        blank, _ = projection(dense, mode="blank")
        disabled, _ = projection(dense, mode="disabled")
        self.assertEqual(tuple(true.shape), (1, 12, 5))
        self.assertEqual(audit["shuffle_indices"], [1, 0])
        self.assertFalse(torch.equal(shuffled_dense, dense))
        self.assertFalse(torch.equal(shuffled, true))
        self.assertIsNotNone(blank)
        self.assertIsNone(disabled)

    def test_zero_parity_and_two_stage_gradient(self):
        torch.manual_seed(3)
        config = self.config()
        model = TinyModel(config.hidden_dim)
        projection = FastV2DenseProjection(config)
        inject_dense_direct_tune(model, config)
        audit_trainables(model, projection)
        hidden = torch.randn(1, 12, config.hidden_dim)
        dense = torch.randn(2, 7, 4, 5)
        cond, _ = projection(dense)
        set_dense_enabled(model, False)
        disabled = model.blocks[0](hidden, dit_cond_dict={})
        set_dense_enabled(model, True)
        enabled = model.blocks[0](hidden, dit_cond_dict={COND_KEY: cond})
        self.assertTrue(torch.equal(enabled, disabled))

        optimizer = torch.optim.SGD(
            [value for _, value in audit_trainables_named(model, projection)], lr=1e-2
        )
        optimizer.zero_grad(set_to_none=True)
        model.blocks[0](hidden, dit_cond_dict={COND_KEY: cond}).square().mean().backward()
        first = gradient_audit(audit_trainables_named(model, projection))
        self.assertTrue(first["adapter_any_nonzero"])
        self.assertFalse(first["projection_any_nonzero"])
        optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        cond, _ = projection(dense)
        model.blocks[0](hidden, dit_cond_dict={COND_KEY: cond}).square().mean().backward()
        second = gradient_audit(audit_trainables_named(model, projection))
        self.assertTrue(second["adapter_any_nonzero"])
        self.assertTrue(second["projection_any_nonzero"])

        envelope = checkpoint_envelope(model, projection, config, optimizer_step=2, metadata={})
        load_checkpoint_envelope(model, projection, envelope, config)


def audit_trainables_named(model, projection):
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ] + [
        (f"dense_projection.{name}", parameter)
        for name, parameter in projection.named_parameters()
        if parameter.requires_grad
    ]


if __name__ == "__main__":
    unittest.main()
