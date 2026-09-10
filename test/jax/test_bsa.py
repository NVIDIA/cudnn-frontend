# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cudnn import block_sparse_attention_forward_jax as forward
from cudnn import block_sparse_attention_backward_jax as backward
from cudnn import block_sparse_attention_jax as attention

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


def inputs(layout="bhsd", d=64, variable=False):
    rng = np.random.default_rng(12)
    arrays = [jnp.asarray(rng.normal(size=s).astype(np.float32), dtype=jnp.bfloat16) for s in ((2, 3, 256, d), (2, 3, 512, d), (2, 3, 512, d))]
    if layout == "bshd":
        arrays = [jnp.transpose(x, (0, 2, 1, 3)) for x in arrays]
    indices = jnp.broadcast_to(jnp.array([3, 1, 0], jnp.int32), (2, 3, 2, 3))
    nums = jnp.broadcast_to(jnp.array([0, 3], jnp.int32), (2, 3, 2)) if variable else None
    return (*arrays, indices, nums)


def reference(q, k, v, indices, nums, layout):
    if layout == "bshd":
        q, k, v = [jnp.transpose(x, (0, 2, 1, 3)) for x in (q, k, v)]
    q, k, v = [x.astype(jnp.float32) for x in (q, k, v)]
    counts = jnp.full(indices.shape[:3], 2) if nums is None else nums
    valid = jnp.arange(indices.shape[-1]) < counts[..., None]
    mask = jnp.any((indices[..., None] == jnp.arange(k.shape[2] // 128)) & valid[..., None], axis=3)
    mask = jnp.repeat(jnp.repeat(mask, 128, axis=2), 128, axis=3)
    logits = jnp.einsum("bhsd,bhtd->bhst", q, k) * q.shape[-1] ** -0.5
    maximum = jnp.max(jnp.where(mask, logits, -jnp.inf), axis=-1, keepdims=True)
    maximum = jnp.where(jnp.isfinite(maximum), maximum, 0)
    weights = jnp.where(mask, jnp.exp(logits - maximum), 0)
    denom = jnp.sum(weights, axis=-1, keepdims=True)
    probs = weights / jnp.where(denom > 0, denom, 1)
    o = jnp.einsum("bhst,bhtd->bhsd", probs, v)
    lse = jnp.where(denom[..., 0] > 0, jnp.log(denom[..., 0]) + maximum[..., 0], -jnp.inf)
    return (jnp.transpose(o, (0, 2, 1, 3)) if layout == "bshd" else o), lse


@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("variable,bucket", [(False, 384), (True, 1)])
def test_numerics_and_grad(layout, d, variable, bucket):
    q, k, v, indices, nums = inputs(layout, d, variable)
    opts = dict(q2k_block_nums=nums, layout=layout, allow_empty_block_nums=variable)
    run = jax.jit(partial(forward, block_sparse_num=2, **{k: v for k, v in opts.items() if k != "q2k_block_nums"}))
    o, lse = run(q, k, v, indices, q2k_block_nums=nums)
    expected, expected_lse = reference(q, k, v, indices, nums, layout)
    np.testing.assert_allclose(o.astype(jnp.float32), expected, atol=3e-2, rtol=3e-2)
    np.testing.assert_allclose(lse, expected_lse, atol=2e-3, rtol=2e-3)
    do = jnp.asarray(np.random.default_rng(7).normal(size=q.shape), jnp.bfloat16)
    grads = jax.jit(partial(backward, block_sparse_num=2, layout=layout, allow_empty_block_nums=variable, bucket_size_blocks=bucket))(
        do, q, k, v, o, lse, indices, q2k_block_nums=nums
    )
    reference_grads = jax.grad(lambda q, k, v: jnp.sum(reference(q, k, v, indices, nums, layout)[0] * do.astype(jnp.float32)), argnums=(0, 1, 2))(q, k, v)
    for actual, desired in zip(grads, reference_grads):
        np.testing.assert_allclose(actual.astype(jnp.float32), desired.astype(jnp.float32), atol=3e-2, rtol=3e-2)
    grad_fn = jax.jit(
        jax.grad(
            lambda q, k, v: jnp.sum(attention(q, k, v, indices, 2, bucket_size_blocks=bucket, **opts).astype(jnp.float32) * do.astype(jnp.float32)),
            argnums=(0, 1, 2),
        )
    )
    for actual, desired in zip(grad_fn(q, k, v), reference_grads):
        np.testing.assert_allclose(actual.astype(jnp.float32), desired.astype(jnp.float32), atol=3e-2, rtol=3e-2)


def test_import_isolation():
    script = """
import sys
class NoTorch:
    def find_spec(self, fullname, *args):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError("torch import attempted")
sys.meta_path.insert(0, NoTorch())
import cudnn
assert callable(cudnn.BSA.block_sparse_attention_jax)
assert callable(cudnn.block_sparse_attention_backward_jax)
assert "torch" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, env=os.environ.copy())


@pytest.mark.parametrize(
    "options,error",
    [
        ({"layout": "bad"}, ValueError),
        ({"sparse_block_size": 64}, ValueError),
        ({"block_sparse_num": 1}, ValueError),
        ({"softmax_scale": float("nan")}, ValueError),
    ],
)
def test_unsupported(options, error):
    q, k, v, indices, _ = inputs()
    kwargs = dict(block_sparse_num=2)
    kwargs.update(options)
    with pytest.raises(error):
        jax.jit(partial(forward, **kwargs))(q, k, v, indices)


def test_async_repeated_calls_and_metadata_changes():
    q, k, v, indices, _ = inputs("bshd")
    do = jnp.ones_like(q)

    @jax.jit
    def run(q, k, v, indices, nums, do):
        opts = dict(q2k_block_nums=nums, layout="bshd", allow_empty_block_nums=True)
        o, lse = forward(q, k, v, indices, 2, **opts)
        grads = backward(do, q, k, v, o, lse, indices, 2, bucket_size_blocks=1, **opts)
        return o, lse, grads

    pending = []
    for i in range(12):
        qi = q + jnp.asarray(i / 100, jnp.bfloat16)
        ii = (indices + i) % 4
        nums = jnp.full(indices.shape[:3], i % 4, jnp.int32)
        pending.append((qi, ii, nums, run(qi, k, v, ii, nums, do)))
    for qi, ii, nums, (o, lse, grads) in pending:
        expected, expected_lse = reference(qi, k, v, ii, nums, "bshd")
        np.testing.assert_allclose(o.astype(jnp.float32), expected, atol=3e-2, rtol=3e-2)
        np.testing.assert_allclose(lse, expected_lse, atol=2e-3, rtol=2e-3)
        expected_grads = jax.grad(lambda q, k, v: reference(q, k, v, ii, nums, "bshd")[0].sum(), argnums=(0, 1, 2))(qi, k, v)
        for actual, desired in zip(grads, expected_grads):
            np.testing.assert_allclose(actual.astype(jnp.float32), desired.astype(jnp.float32), atol=3e-2, rtol=3e-2)
    q0, k0, v0, i0, _ = inputs("bshd")
    for actual, original in zip((q, k, v, indices), (q0, k0, v0, i0)):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
def test_eager_and_layout_lowering(layout):
    q, k, v, indices, _ = inputs(layout)
    eager = forward(q, k, v, indices, 2, layout=layout)
    fn = jax.jit(partial(forward, block_sparse_num=2, layout=layout))
    compiled = fn.lower(q, k, v, indices).compile()
    for a, b in zip(eager, compiled(q, k, v, indices)):
        np.testing.assert_array_equal(a, b)
    text = compiled.as_text().lower()
    assert "custom-call" in text
    assert "transpose(" not in text
    assert "copy(" not in text


@pytest.mark.parametrize("kind", ["dtype", "dimension", "tail", "gqa", "metadata_dtype", "metadata_shape"])
def test_tensor_contract(kind):
    q, k, v, indices, _ = inputs()
    if kind == "dtype":
        q = q.astype(jnp.float16)
    elif kind == "dimension":
        q, k, v = [x[..., :32] for x in (q, k, v)]
    elif kind == "tail":
        q = q[:, :, :255]
    elif kind == "gqa":
        k, v = k[:, :1], v[:, :1]
    elif kind == "metadata_dtype":
        indices = indices.astype(jnp.float32)
    else:
        indices = indices[:, :, :1]
    with pytest.raises((TypeError, ValueError)):
        forward(q, k, v, indices, 2)


def test_unsupported_forward_mode():
    q, k, v, indices, _ = inputs()
    with pytest.raises((TypeError, NotImplementedError)):
        jax.jvp(lambda q: attention(q, k, v, indices, 2), (q,), (jnp.ones_like(q),))


def test_single_block_variable_count_and_default_count():
    q = jnp.full((1, 1, 128, 64), 0.125, jnp.bfloat16)
    v = jnp.ones_like(q)
    indices = jnp.zeros((1, 1, 1, 1), jnp.int32)
    nums = jnp.ones((1, 1, 1), jnp.int32)
    o, lse = forward(q_tensor=q, k_tensor=q, v_tensor=v, q2k_block_index=indices, q2k_block_nums=nums)
    np.testing.assert_array_equal(o, v)
    grad = jax.jit(jax.grad(lambda q: attention(q, q, v, indices, q2k_block_nums=nums).astype(jnp.float32).sum()))(q)
    np.testing.assert_allclose(grad.astype(jnp.float32), 0, atol=1e-3)
    assert lse.shape == (1, 1, 128)


def test_runtime_isolation():
    script = """
import sys
class NoTorch:
    def find_spec(self, fullname, *args):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError("torch import attempted")
sys.meta_path.insert(0, NoTorch())
import jax
import jax.numpy as jnp
from cudnn import block_sparse_attention_jax as attention
q = jnp.ones((1, 1, 256, 64), jnp.bfloat16)
i = jnp.broadcast_to(jnp.arange(2, dtype=jnp.int32), (1, 1, 2, 2))
grad = jax.jit(jax.grad(lambda q: attention(q, q, q, i, 2).astype(jnp.float32).sum()))(q)
grad.block_until_ready()
assert jnp.all(jnp.abs(grad.astype(jnp.float32) - 1) < 0.01)
assert "torch" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, env=os.environ.copy(), timeout=120)
