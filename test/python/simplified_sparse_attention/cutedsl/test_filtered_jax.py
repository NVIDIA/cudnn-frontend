# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Long-key selection, exact repair, and the key-major indexer gradient."""

import pytest
import torch

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from cudnn.ops import simplified_sparse_attention_jax
from cudnn.simplified_sparse_attention import jax_api
from ssa_reference import pack, sample
from test_simplified_sparse_attention import supported_gpu
from test_simplified_sparse_attention_jax import jax_inputs

pytestmark = [pytest.mark.L1, pytest.mark.usefixtures("supported_gpu")]


def selected_reference(inputs, indices):
    tensors, metadata, upstream = inputs
    q, k, v, iq, ik = [t.detach().double().requires_grad_() for t in tensors]
    lengths = (metadata.key_ends - metadata.key_starts).clamp(0, indices.shape[1])
    live = torch.arange(indices.shape[1], device="cuda")[None] < lengths[:, None]
    live = live | (~metadata.query_valid[:, None] & (torch.arange(indices.shape[1], device="cuda")[None] == 0))
    p = (torch.einsum("qhd,qkd->qhk", q, k[indices, 0]) / 16).masked_fill(~live[:, None], -torch.inf).softmax(-1)
    out = torch.einsum("qhk,qkd->qhd", p, v[indices, 0]).masked_fill(~metadata.query_valid[:, None, None], 0)
    student = ((iq[:, None] * ik[indices]).sum(-1) / 128**0.5).masked_fill(~live, -torch.inf).softmax(-1)
    teacher = p.detach().mean(1)
    loss = (teacher * ((teacher + 1e-10).log() - (student + 1e-10).log())).sum(-1)
    loss = loss.masked_fill(~metadata.query_valid, 0).sum() * 0.1 / metadata.loss_denominator[0]
    grads = torch.autograd.grad((out, loss), (q, k, v, iq, ik), (upstream.double(), torch.ones_like(loss)))
    return dict(zip(("output", "indexer_loss", "dq", "dk", "dv", "dq_idx", "dk_idx"), (out, loss, *grads)))


@pytest.mark.parametrize("topk", [512, 2048])
def test_filtered_training_and_changed_inputs(topk, monkeypatch):
    rows, keys = 129, 131329
    query, upstream = sample(rows)
    key, _ = sample(keys, heads=1, seed=33)
    tensors = [query[0], key[1], key[2], query[3], key[4]]
    _, metadata, _ = pack([sample(rows)])
    metadata.query_valid.fill_(True)
    metadata.key_starts.zero_()
    metadata.key_ends.fill_(keys // 2)
    inputs = tensors, metadata, upstream

    selected = []
    indexer = jax_api._indexer

    def observe(*args):
        result = indexer(*args)
        selected.append(result)
        return result

    monkeypatch.setattr(jax_api, "_indexer", observe)

    @jax.jit
    def train(*arrays):
        selected.clear()

        def operation(*values):
            return simplified_sparse_attention_jax(*values, *arrays[5:9], topk=topk)

        (output, loss), pullback = jax.vjp(operation, *arrays[:5])
        gradients = pullback((arrays[9], jnp.ones_like(loss)))
        # Observe the same selection: tied keys may differ between executions.
        return output, loss, *gradients, *selected[-1]

    # All 129 tied rows fail the certificate and require two repair chunks.
    for changed in (False, True):
        if changed:
            tensors[3] = query[3]
            metadata.key_starts.fill_(keys // 2)
            metadata.key_ends.fill_(keys)
            metadata.query_valid[::7] = False
        else:
            tensors[3] = torch.zeros_like(query[3])
        values, bounds, grad = jax_inputs(inputs)
        result = jax.block_until_ready(train(*values, *bounds, grad))
        converted = [torch.from_dlpack(x) for x in result]
        actual = dict(zip(("output", "indexer_loss", "dq", "dk", "dv", "dq_idx", "dk_idx"), converted[:7]))
        ids, probabilities = converted[-2:]
        valid = metadata.query_valid
        active = ids[valid]
        assert ((active >= metadata.key_starts[valid, None]) & (active < metadata.key_ends[valid, None])).all()
        assert (active[:, 1:] > active[:, :-1]).all()
        torch.testing.assert_close(probabilities[valid].sum(-1), torch.ones(int(valid.sum()), device="cuda"), rtol=1e-5, atol=1e-6)
        scores = tensors[3].float() @ tensors[4].float().T / 128**0.5
        columns = torch.arange(keys, device="cuda")
        allowed = (columns[None] >= metadata.key_starts[:, None]) & (columns[None] < metadata.key_ends[:, None])
        exact = scores.masked_fill(~allowed, -torch.inf).topk(topk, dim=-1).values
        selected = scores.gather(1, ids.long()).sort(dim=-1, descending=True).values
        torch.testing.assert_close(selected[valid], exact[valid], rtol=1e-5, atol=1e-6)
        ids = ids.masked_fill(~valid[:, None], 0).long()
        expected = selected_reference(inputs, ids)
        for name, want in expected.items():
            got = actual[name].double()
            assert torch.isfinite(got).all(), name
            if name == "indexer_loss":
                # Near-uniform KL needs an absolute FP32 rounding allowance.
                loss_atol = 16 * torch.finfo(torch.float32).eps * 0.1 * valid.sum().item() / metadata.loss_denominator.item()
                torch.testing.assert_close(got, want, rtol=0.01, atol=loss_atol)
                continue
            error = (got - want).norm() / want.norm().clamp_min(1e-12)
            assert error < 0.01, (name, error.item())
            torch.testing.assert_close(got, want, rtol=0.03, atol=0.005 * want.abs().max().item() + 1e-9)
