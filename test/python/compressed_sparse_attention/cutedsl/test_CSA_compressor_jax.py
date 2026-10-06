# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp


def _require_sm100():
    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("JAX has no CUDA device")
    from cudnn.tensor_adapter import get_compute_capability

    if get_compute_capability()[0] < 10:
        pytest.skip("CSA CuTeDSL kernels require compute capability >= 10")


def _make_inputs(seed, ratio, blocks, width):
    """Random single-segment pack of ``blocks * ratio`` tokens."""
    key = jax.random.key(seed)
    kv = jax.random.normal(key, (blocks * ratio, width), jnp.bfloat16)
    score = jax.random.normal(jax.random.fold_in(key, 1), kv.shape, jnp.bfloat16)
    ape = jax.random.normal(jax.random.fold_in(key, 2), (ratio, width), jnp.float32)
    cu = jnp.array([0, blocks * ratio], jnp.int32)
    cu_comp = jnp.array([0, blocks], jnp.int32)
    return kv, score, ape, cu, cu_comp


def _check_against_reference(reference, inputs, *, ratio, coff, out_tol, grad_tol, ape_tol):
    """Compare the fused forward/backward with ``reference`` and its autodiff gradients.

    Tolerances are ``(atol, rtol)``; ``grad_tol`` applies to grad_kv and grad_score.
    """
    from cudnn import csa_compressor_backward_jax_sm100, csa_compressor_forward_jax_sm100

    kv, score, ape, cu, cu_comp = inputs
    got = jax.jit(lambda x, s, a: csa_compressor_forward_jax_sm100(x, s, a, cu, cu_comp, total_comp=kv.shape[0] // ratio, ratio=ratio, coff=coff))(
        kv, score, ape
    )
    grads = csa_compressor_backward_jax_sm100(kv, score, ape, cu, cu_comp, jnp.ones_like(got), ratio=ratio, coff=coff)
    ref_grads = jax.grad(lambda x, s, a: reference(x, s, a).astype(jnp.float32).sum(), argnums=(0, 1, 2))(kv, score, ape)

    pairs = [(got, reference(kv, score, ape), out_tol)]
    pairs += zip(grads, ref_grads, (grad_tol, grad_tol, ape_tol))
    for actual, expected, (atol, rtol) in pairs:
        np.testing.assert_allclose(np.asarray(actual, np.float32), np.asarray(expected, np.float32), atol=atol, rtol=rtol)


@pytest.mark.L0
@pytest.mark.parametrize("head_dim", [128, 512])
def test_csa_compressor_jax_matches_overlap_reference(head_dim):
    _require_sm100()

    def reference(x, scores, bias):
        a_kv, b_kv = jnp.split(x.reshape(4, 4, 2 * head_dim), 2, axis=-1)
        a_score, b_score = jnp.split(scores.reshape(4, 4, 2 * head_dim) + bias, 2, axis=-1)
        shifted_kv = jnp.concatenate([jnp.zeros_like(a_kv[:1]), a_kv[:-1]], axis=0)
        shifted_score = jnp.concatenate([jnp.full_like(a_score[:1], -jnp.inf), a_score[:-1]], axis=0)
        values = jnp.concatenate([shifted_kv, b_kv], axis=1)
        logits = jnp.concatenate([shifted_score, b_score], axis=1)
        return jnp.sum(values * jax.nn.softmax(logits.astype(jnp.float32), axis=1).astype(values.dtype), axis=1)

    _check_against_reference(
        reference,
        _make_inputs(head_dim, ratio=4, blocks=4, width=2 * head_dim),
        ratio=4,
        coff=2,
        out_tol=(8e-3, 8e-3),
        grad_tol=(1e-7, 1e-6),
        ape_tol=(1e-6, 1e-6),
    )


@pytest.mark.L0
@pytest.mark.parametrize("coff", [1, 2])
@pytest.mark.parametrize("head_dim", [128, 512])
def test_hca_compressor_jax_matches_reference(head_dim, coff):
    _require_sm100()

    ratio, blocks, width = 128, 4, coff * head_dim

    def reference(x, scores, bias):
        values = x.astype(jnp.float32).reshape(blocks, ratio, width)
        logits = scores.astype(jnp.float32).reshape(blocks, ratio, width) + bias
        if coff == 2:
            a_kv, values = jnp.split(values, 2, axis=-1)
            a_score, logits = jnp.split(logits, 2, axis=-1)
            shifted_kv = jnp.concatenate([jnp.zeros_like(a_kv[:1]), a_kv[:-1]], axis=0)
            shifted_score = jnp.concatenate([jnp.full_like(a_score[:1], -jnp.inf), a_score[:-1]], axis=0)
            values = jnp.concatenate([shifted_kv, values], axis=1)
            logits = jnp.concatenate([shifted_score, logits], axis=1)
        return jnp.sum(values * jax.nn.softmax(logits, axis=1), axis=1).astype(jnp.bfloat16)

    # The ratio-128 kernels reorder reductions and may use a fast exp, so they match the
    # fp32 reference within tolerance rather than bitwise.
    _check_against_reference(
        reference,
        _make_inputs(head_dim + coff, ratio, blocks, width),
        ratio=ratio,
        coff=coff,
        out_tol=(1.6e-2, 0),
        grad_tol=(1.6e-2, 0),
        ape_tol=(1e-3, 0),
    )
