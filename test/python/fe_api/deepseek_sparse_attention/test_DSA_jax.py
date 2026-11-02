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
        pytest.skip("DSA CuTeDSL kernels require compute capability >= 10")


@pytest.mark.L0
def test_indexer_jax_matches_native_math():
    _require_sm100()
    from cudnn import indexer_forward_jax_sm100

    key = jax.random.key(4)
    q = jax.random.normal(key, (1, 16, 64, 128), jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (1, 4, 1, 128), jnp.bfloat16)
    weights = jax.random.normal(jax.random.fold_in(key, 2), (1, 16, 64), jnp.bfloat16)
    scale = 128**-0.5
    got = indexer_forward_jax_sm100(q, k, weights, sm_scale=scale)
    dots = jnp.einsum("bshd,bwkd->bhsw", q.astype(jnp.float32), k.astype(jnp.float32))
    ref = jnp.einsum("bhsw,bsh->bsw", jax.nn.relu(dots) * scale, weights.astype(jnp.float32))
    valid = jnp.isfinite(got)
    np.testing.assert_allclose(np.asarray(got[valid]), np.asarray(ref[valid]), atol=2e-5, rtol=2e-5)


@pytest.mark.L0
def test_sparse_attention_jax_forward_and_backward():
    _require_sm100()
    from cudnn import sparse_attention_backward_jax_sm100, sparse_attention_forward_jax_sm100

    key = jax.random.key(5)
    q = jax.random.normal(key, (1, 64, 512), jnp.bfloat16) / jnp.sqrt(512)
    kv = jax.random.normal(jax.random.fold_in(key, 1), (640, 512), jnp.bfloat16)
    indices = jnp.arange(640, dtype=jnp.int32)[None]
    lengths = jnp.array([640], jnp.int32)
    sinks = jnp.zeros((64,), jnp.float32)
    # Q is already scaled above; keep the native call and reference at scale 1.
    got, _, lse, _ = sparse_attention_forward_jax_sm100(q, kv, indices, lengths, sinks, softmax_scale=1.0)
    logits = jnp.einsum("hd,kd->hk", q[0].astype(jnp.float32), kv.astype(jnp.float32))
    maximum = jnp.maximum(jnp.max(logits, axis=-1), sinks)
    probabilities = jnp.exp(logits - maximum[:, None])
    ref = jnp.einsum("hk,kd->hd", probabilities, kv.astype(jnp.float32))
    ref /= (jnp.sum(probabilities, axis=-1) + jnp.exp(sinks - maximum))[:, None]
    np.testing.assert_allclose(np.asarray(got[0], np.float32), np.asarray(ref), atol=1e-2, rtol=2e-2)

    gradients = sparse_attention_backward_jax_sm100(q, kv, got, jnp.ones_like(got), lse, sinks, indices, lengths, softmax_scale=1.0)

    def reference_loss(x, y, s):
        reference_logits = jnp.einsum("thd,kd->thk", x.astype(jnp.float32), y.astype(jnp.float32))
        reference_max = jnp.maximum(jnp.max(reference_logits, axis=-1), s)
        exponentials = jnp.exp(reference_logits - reference_max[..., None])
        denominator = jnp.sum(exponentials, axis=-1) + jnp.exp(s - reference_max)
        output = jnp.einsum("thk,kd->thd", exponentials, y.astype(jnp.float32)) / denominator[..., None]
        return output.sum()

    reference_gradients = jax.grad(reference_loss, argnums=(0, 1, 2))(q, kv, sinks)
    np.testing.assert_allclose(np.asarray(gradients[0], np.float32), np.asarray(reference_gradients[0], np.float32), atol=4e-2, rtol=2e-2)
    np.testing.assert_allclose(np.asarray(gradients[1], np.float32), np.asarray(reference_gradients[1], np.float32), atol=3e-3, rtol=2e-2)
    np.testing.assert_allclose(np.asarray(gradients[2]), np.asarray(reference_gradients[2]), atol=2e-5, rtol=2e-2)


@pytest.mark.L0
@pytest.mark.parametrize("lengths", [(21, 33), (3, 33)])
def test_indexer_jax_thd_matches_independent_sequences(lengths):
    _require_sm100()
    from cudnn import indexer_forward_jax_sm100

    compressed = [n // 4 for n in lengths]
    cu_q = jnp.array([0, lengths[0], sum(lengths), sum(lengths)], jnp.int32)
    cu_k = jnp.array([0, compressed[0], sum(compressed), sum(compressed)], jnp.int32)
    # Extra capacity is poisoned: neither it nor the unused prefix slot is data.
    key = jax.random.key(12)
    q = jax.random.normal(key, (sum(lengths) + 1, 64, 128), jnp.bfloat16).at[-1].set(jnp.nan)
    k = jax.random.normal(jax.random.fold_in(key, 1), (sum(compressed) + 1, 1, 128), jnp.bfloat16).at[-1].set(jnp.nan)
    w = jax.random.normal(jax.random.fold_in(key, 2), q.shape[:2], jnp.bfloat16).at[-1].set(jnp.nan)
    bound = (max(lengths) + 3) // 4
    width = (bound + 3) // 4 * 4
    got = jax.jit(
        lambda x, y, z: indexer_forward_jax_sm100(
            x,
            y,
            z,
            sm_scale=128**-0.5,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max(lengths),
            max_seqlen_k=bound,
        )
    )(q, k, w)
    expected = np.full((q.shape[0], width), -np.inf, np.float32)
    qs = ks = 0
    for nq, nk in zip(lengths, compressed):
        dots = jnp.einsum("thd,kd->thk", q[qs : qs + nq].astype(jnp.float32), k[ks : ks + nk, 0].astype(jnp.float32))
        scores = jnp.einsum("thk,th->tk", jax.nn.relu(dots) * 128**-0.5, w[qs : qs + nq].astype(jnp.float32))
        causal = 4 * (jnp.arange(nk)[None] + 1) <= jnp.arange(nq)[:, None] + 1
        expected[qs : qs + nq, :nk] = np.asarray(jnp.where(causal, scores, -jnp.inf))
        qs, ks = qs + nq, ks + nk
    np.testing.assert_array_equal(np.isfinite(got), np.isfinite(expected))
    valid = np.isfinite(expected)
    np.testing.assert_allclose(np.asarray(got)[valid], expected[valid], atol=2e-5, rtol=2e-5)
