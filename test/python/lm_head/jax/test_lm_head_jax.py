# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX LM-head native launches, replay and gradients."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from cudnn.ops import lm_head_logprob_backward_jax, lm_head_logprob_forward_jax, lm_head_logprob_jax

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def available():
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("requires a GPU")
    from cudnn._cake.loader import check_toolkit, device_arch

    if device_arch(0) != "sm_100a":
        pytest.skip("requires SM100")
    try:
        check_toolkit("sm_100a", "Cake JAX LM head")
    except (RuntimeError, NotImplementedError) as error:
        pytest.skip(str(error))
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def inputs(rows=17, hidden=512, vocab=512):
    rng = np.random.default_rng(93)
    x = jnp.asarray(rng.normal(0, 0.1, (rows, hidden)), jnp.bfloat16)
    weight = jnp.asarray(rng.normal(0, 0.1, (vocab, hidden)), jnp.bfloat16)
    labels = np.arange(rows, dtype=np.int64) % vocab
    labels[::3] = -100
    return x, weight, jnp.asarray(labels), jnp.asarray(rng.normal(size=rows), jnp.float32)


def reference(x, weight, labels):
    logits = jnp.dot(x.astype(jnp.float64), weight.astype(jnp.float64).T, precision=jax.lax.Precision.HIGHEST)
    lse = jax.scipy.special.logsumexp(logits, axis=-1)
    selected = jnp.take_along_axis(logits, jnp.maximum(labels, 0)[:, None], axis=-1)[:, 0]
    return jnp.where(labels == -100, 0, selected - lse), lse


@pytest.mark.parametrize("rows,hidden,vocab", [(17, 512, 512), (257, 1024, 768)])
def test_jit_forward_backward_replay(rows, hidden, vocab):
    x, weight, labels, dlogp = inputs(rows, hidden, vocab)
    forward = jax.jit(lm_head_logprob_forward_jax)
    backward = jax.jit(lm_head_logprob_backward_jax)
    for factor in (1, 0.5):
        current = (x * factor).astype(jnp.bfloat16)
        logp, lse = forward(current, weight, labels)
        dx, dw = backward(current, weight, labels, lse, dlogp)
        ref, ref_lse = reference(current, weight, labels)
        expected = jax.grad(lambda a, b: jnp.vdot(reference(a, b, labels)[0], dlogp), argnums=(0, 1))(current.astype(jnp.float64), weight.astype(jnp.float64))
        np.testing.assert_allclose(logp, ref, atol=0.01, rtol=0.003)
        np.testing.assert_allclose(lse, ref_lse, atol=0.01, rtol=0.003)
        for got, want in zip((dx, dw), expected):
            assert got.dtype == jnp.float32
            assert np.linalg.norm(np.asarray(got) - np.asarray(want)) / np.linalg.norm(np.asarray(want)) < 0.015
        repeated = backward(current, weight, labels, lse, dlogp)
        for got, again in zip((dx, dw), repeated):
            np.testing.assert_array_equal(got, again)
    hlo = forward.lower(x, weight, labels).as_text()
    assert "cudnn_cake_launch_" in hlo
    assert "callback" not in hlo


def test_custom_vjp_and_ignored_rows():
    x, weight, labels, dlogp = inputs()
    loss_grad = jax.jit(jax.value_and_grad(lambda a, b, y: jnp.vdot(lm_head_logprob_jax(a, b, y), dlogp), argnums=(0, 1)))
    value, gradients = loss_grad(x, weight, labels)
    logp, lse = lm_head_logprob_forward_jax(x, weight, labels)
    explicit = lm_head_logprob_backward_jax(x, weight, labels, lse, dlogp)
    np.testing.assert_allclose(value, jnp.vdot(logp, dlogp))
    for got, want in zip(gradients, explicit):
        assert got.dtype == jnp.bfloat16
        np.testing.assert_array_equal(got, want.astype(jnp.bfloat16))
    ignored = jnp.full_like(labels, -100)
    value, gradients = loss_grad(x, weight, ignored)
    assert float(value) == 0
    for grad in gradients:
        assert not np.count_nonzero(np.asarray(grad))


def test_input_contract():
    x, weight, labels, dlogp = inputs()
    with pytest.raises(ValueError, match="int64"):
        lm_head_logprob_jax(x, weight, labels.astype(jnp.int32))
    with pytest.raises(ValueError, match="BF16"):
        lm_head_logprob_jax(x.astype(jnp.float32), weight, labels)
    with pytest.raises(ValueError, match="requires x"):
        lm_head_logprob_jax(x, weight[:, :-1], labels)
    with pytest.raises(ValueError, match="FP32"):
        lm_head_logprob_backward_jax(x, weight, labels, dlogp, dlogp.astype(jnp.bfloat16))
