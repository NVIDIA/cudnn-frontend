# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public JAX forward and all five reverse-mode gradients."""

from functools import lru_cache
import pytest
import torch

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from cudnn.ops import simplified_sparse_attention_jax
from ssa_reference import check_reference, pack, sample
from test_simplified_sparse_attention import supported_gpu

pytestmark = [pytest.mark.L0, pytest.mark.usefixtures("supported_gpu")]


def jax_inputs(inputs):
    tensors, metadata, upstream = inputs
    values = tuple(jax.dlpack.from_dlpack(t) for t in tensors)
    bounds = tuple(jax.dlpack.from_dlpack(t) for t in (metadata.key_starts, metadata.key_ends, metadata.query_valid, metadata.loss_denominator))
    return values, bounds, jax.dlpack.from_dlpack(upstream)


@lru_cache(maxsize=None)
def runner(topk=512, coefficient=0.1, triangular=False, backward=True):
    @jax.jit
    def train(*arrays):
        tensors, metadata, upstream = arrays[:5], arrays[5:9], arrays[9]

        def operation(*values):
            return simplified_sparse_attention_jax(*values, *metadata, topk=topk, loss_coeff=coefficient, triangular_scores=triangular)

        if not backward:
            return operation(*tensors)
        (output, loss), pullback = jax.vjp(operation, *tensors)
        gradients = pullback((upstream, jnp.ones_like(loss)))
        return output, loss, *gradients

    return train


def execute(inputs, topk=512, coefficient=0.1, triangular=False, backward=True):
    tensors, metadata, upstream = jax_inputs(inputs)
    result = jax.block_until_ready(runner(topk, coefficient, triangular, backward)(*tensors, *metadata, upstream))
    return dict(zip(("output", "indexer_loss", "dq", "dk", "dv", "dq_idx", "dk_idx"), (torch.from_dlpack(x) for x in result)))


@pytest.mark.parametrize("heads", [16, 32, 96], ids=lambda n: f"bf16-h{n}")
@pytest.mark.parametrize("topk", [512, 1024, 2048], ids=lambda n: f"k{n}")
def test_jit_forward_backward(heads, topk):
    inputs = pack([sample(65, heads), sample(17, heads, seed=31)])
    before = [t.clone() for t in (*inputs[0], inputs[2])]
    actual = execute(inputs, topk)
    check_reference(inputs, actual, True, topk)
    for a, b in zip((*inputs[0], inputs[2]), before):
        assert torch.equal(a, b)


def test_selected_support():
    tensors, upstream = sample(1025)
    tensors[0], tensors[3] = tensors[0][:65], tensors[3][:65]
    _, metadata, _ = pack([sample(65)])
    metadata.key_ends.fill_(1025)
    inputs = tensors, metadata, upstream[:65]
    check_reference(inputs, execute(inputs), True)


def test_triangular_and_zero_loss():
    inputs = pack([sample(17)])
    check_reference(inputs, execute(inputs, triangular=True), True)
    zero = execute(inputs, coefficient=0)
    for name in ("indexer_loss", "dq_idx", "dk_idx"):
        assert torch.count_nonzero(zero[name]) == 0


@pytest.mark.parametrize("change", ["dtype", "heads", "topk", "bounds", "denominator", "empty"])
def test_invalid_inputs(change):
    tensors, metadata, _ = jax_inputs(pack([sample(8)]))
    tensors, metadata, topk = list(tensors), list(metadata), 512
    if change == "dtype":
        tensors[0] = tensors[0].astype(jnp.float32)
    elif change == "heads":
        tensors[0] = tensors[0][:, :8]
    elif change == "topk":
        topk = 32
    elif change == "bounds":
        metadata[0] = metadata[0][:-1]
    elif change == "denominator":
        metadata[3] = metadata[3][0]
    else:
        tensors[0] = tensors[0][:0]
    with pytest.raises((ValueError, TypeError)):
        jax.jit(lambda *x: simplified_sparse_attention_jax(*x, *metadata, topk=topk))(*tensors)


@pytest.mark.parametrize("captured", [False, True])
def test_mixed_architecture_rejected_before_launch(monkeypatch, captured):
    from types import SimpleNamespace
    from cudnn.simplified_sparse_attention import jax_api

    tensors, metadata, _ = jax_inputs(pack([sample(8)]))
    monkeypatch.setattr(jax, "local_devices", lambda **kw: [SimpleNamespace(compute_capability=cc) for cc in ("10.0", "10.3")])

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid placement reached the kernel")

    monkeypatch.setattr(jax_api, "_attention", forbidden)
    with pytest.raises(ValueError, match="only SM100 or only SM103"):
        if captured:
            jax.jit(lambda: simplified_sparse_attention_jax(*tensors, *metadata))()
        else:
            jax.jit(lambda *x: simplified_sparse_attention_jax(*x, *metadata))(*tensors)


def test_captured_jit_uses_native_buffers(monkeypatch):
    inputs = pack([sample(19)])
    tensors, metadata, _ = jax_inputs(inputs)

    def forbidden(*args, **kwargs):
        raise AssertionError("JAX execution allocated through PyTorch")

    with monkeypatch.context() as patch:
        for name in ("empty", "zeros", "empty_like", "zeros_like"):
            patch.setattr(torch, name, forbidden)
        result = jax.block_until_ready(jax.jit(lambda: simplified_sparse_attention_jax(*tensors, *metadata))())
    actual = dict(zip(("output", "indexer_loss"), (torch.from_dlpack(x) for x in result)))
    check_reference(inputs, actual, False)


def test_dense_workspace_index_limit():
    shapes = ((65536, 16, 256), (65536, 1, 256), (65536, 1, 256), (65536, 128), (65536, 128), (65536,), (65536,), (65536,), (1,))
    dtypes = (jnp.bfloat16,) * 5 + (jnp.int32, jnp.int32, jnp.bool_, jnp.float32)
    inputs = [jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in zip(shapes, dtypes)]
    with pytest.raises(ValueError, match="Dense score workspace exceeds"):
        jax.eval_shape(lambda *x: simplified_sparse_attention_jax(*x, triangular_scores=True), *inputs)
