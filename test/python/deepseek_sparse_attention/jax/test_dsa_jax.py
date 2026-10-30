# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import pytest

from cudnn.frost.buffers import cutedsl_requirement_error

pytest.importorskip("cutlass")
requirement_error = cutedsl_requirement_error("DSA JAX tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from cudnn.ops import dsa

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def require_sm100():
    device = jax.local_devices()[0]
    if device.platform != "gpu" or not str(getattr(device, "compute_capability", "")).startswith("10."):
        pytest.skip("DSA JAX ops require the first visible GPU to be SM100-family")


def indexer_scores(q, k, weights, sm_scale):
    dots = jnp.einsum("bshd,bkd->bshk", q.astype(jnp.float32), k.astype(jnp.float32))
    return jnp.einsum("bshk,bsh->bsk", jax.nn.relu(dots), weights.astype(jnp.float32) * sm_scale)


def kl_score_gradient(target, predict, valid, grad_scale):
    """Clipped-log KL gradient w.r.t. the scores, as the kernels form it (dsa_reference.ref_indexer_backward)."""
    eps = math.exp(-100.0)
    g = -jnp.maximum(target, eps) * (predict >= eps) * grad_scale
    return jnp.where(valid, g - predict * g.sum(-1, keepdims=True), 0.0)


def assert_close_gradients(actual, expected):
    for got, ref in zip(actual, expected):
        got, ref = np.asarray(got, np.float32).ravel(), np.asarray(ref, np.float32).ravel()
        cosine = got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref))
        rms_rel = np.sqrt(np.mean((got - ref) ** 2)) / np.sqrt(np.mean(ref**2))
        assert cosine >= 0.999 and rms_rel <= 0.02, (cosine, rms_rel)


def test_indexer_jax_matches_native_math():
    key = jax.random.key(4)
    q = jax.random.normal(key, (1, 16, 64, 128), jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (1, 4, 1, 128), jnp.bfloat16)
    weights = jax.random.normal(jax.random.fold_in(key, 2), (1, 16, 64), jnp.bfloat16)
    scale = 128**-0.5
    got = dsa.indexer_forward_jax(q, k, weights, sm_scale=scale)
    ref = indexer_scores(q, k[:, :, 0], weights, scale)
    valid = jnp.isfinite(got)
    np.testing.assert_allclose(np.asarray(got[valid]), np.asarray(ref[valid]), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("dtype, num_cols", [(jnp.float32, 1003), (jnp.bfloat16, 1001)])
def test_indexer_top_k_jax_matches_lax_top_k(dtype, num_cols):
    top_k, rows = 64, 8
    scores = jax.random.normal(jax.random.key(6), (rows, num_cols), jnp.float32).astype(dtype)
    seq_lens = jnp.array([num_cols, num_cols - 1, 700, 65, top_k, 40, 1, 0], jnp.int32)
    got = np.asarray(jax.jit(lambda s, n: dsa.indexer_top_k_forward_jax(s, n, top_k=top_k))(scores, seq_lens))
    for row, length in enumerate(np.asarray(seq_lens)):
        if length <= top_k:
            expected = np.concatenate([np.arange(length), np.full(top_k - length, -1)])
            np.testing.assert_array_equal(got[row], expected)
        else:
            _, expected = jax.lax.top_k(scores[row, :length], top_k)
            np.testing.assert_array_equal(np.sort(got[row]), np.sort(np.asarray(expected)))


def test_dense_indexer_backward_jax_matches_reference():
    batch, seqlen_q, seqlen_k, ratio, sm_scale = 1, 256, 64, 4, 128**-0.5
    key = jax.random.key(7)
    q = jax.random.normal(key, (batch, seqlen_q, 64, 128), jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (batch, seqlen_k, 1, 128), jnp.bfloat16)
    weights = jax.random.normal(jax.random.fold_in(key, 2), (batch, seqlen_q, 64), jnp.bfloat16)
    valid = jnp.arange(seqlen_k)[None, None] < (jnp.arange(seqlen_q)[None, :, None] + 1) // ratio
    attn_score = jnp.where(valid, jax.random.uniform(jax.random.fold_in(key, 3), (batch, seqlen_q, seqlen_k)), 0.0)
    attn_l1norm = attn_score.sum(-1)
    grad_loss = jnp.array([1.5], jnp.float32)

    index_score, index_lse = dsa.indexer_forward_jax(q, k, weights, ratio=ratio, sm_scale=sm_scale, return_lse=True)
    got = dsa.dense_indexer_backward_jax(q, k, weights, attn_score, attn_l1norm, index_score, index_lse, grad_loss, ratio=ratio, sm_scale=sm_scale)

    scores, vjp = jax.vjp(lambda x, y, w: indexer_scores(x, y[:, :, 0], w, sm_scale), q, k, weights)
    predict = jnp.where(valid, jax.nn.softmax(jnp.where(valid, scores, -jnp.inf), axis=-1), 0.0)
    target = attn_score / jnp.maximum(attn_l1norm, 1e-10)[..., None]
    expected = vjp(kl_score_gradient(target, predict, valid, grad_loss[0] / (batch * seqlen_q)))
    assert_close_gradients(got, (expected[0], expected[2], expected[1][:, :, 0]))


def test_indexer_backward_jax_matches_reference_and_keeps_inputs():
    batch, seqlen_q, seqlen_k, topk, sm_scale = 1, 128, 1024, 512, 128**-0.5
    key = jax.random.key(8)
    q = jax.random.normal(key, (batch, seqlen_q, 64, 128), jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (batch, seqlen_k, 1, 128), jnp.bfloat16)
    weights = jax.random.normal(jax.random.fold_in(key, 2), (batch, seqlen_q, 64), jnp.bfloat16)
    indices = jax.vmap(lambda r: jax.random.permutation(r, seqlen_k)[:topk])(jax.random.split(jax.random.fold_in(key, 3), seqlen_q))
    valid = jnp.arange(topk)[None, None] < jnp.arange(1, seqlen_q + 1)[None, :, None] * 4
    indices = jnp.where(valid, indices[None], -1).astype(jnp.int32)
    grad_loss = jnp.array([0.5], jnp.float32)

    def selected_scores(x, y, w):
        gathered = jnp.take_along_axis(y[:, None, :, 0], jnp.maximum(indices, 0)[..., None], axis=2)
        dots = jnp.einsum("bshd,bstd->bsht", x.astype(jnp.float32), gathered.astype(jnp.float32))
        return jnp.einsum("bsht,bsh->bst", jax.nn.relu(dots), w.astype(jnp.float32) * sm_scale)

    scores, vjp = jax.vjp(selected_scores, q, k, weights)
    index_score = jnp.where(valid, jax.nn.softmax(jnp.where(valid, scores, -jnp.inf), axis=-1), 0.0)
    attn_score = jnp.where(valid, jax.nn.softmax(jax.random.normal(jax.random.fold_in(key, 4), scores.shape), axis=-1), 0.0)
    attn_score = attn_score / attn_score.sum(-1, keepdims=True)
    attn_score_before = np.asarray(attn_score)

    got = dsa.indexer_backward_jax(q, k, weights, attn_score, index_score, indices, grad_loss, sm_scale=sm_scale)

    np.testing.assert_array_equal(np.asarray(attn_score), attn_score_before)
    expected = vjp(kl_score_gradient(attn_score, index_score, valid, grad_loss[0] / (batch * seqlen_q)))
    assert_close_gradients(got, (expected[0], expected[2], expected[1][:, :, 0]))


def test_sparse_attention_jax_forward_and_backward():
    key = jax.random.key(5)
    q = jax.random.normal(key, (1, 64, 512), jnp.bfloat16) / jnp.sqrt(512)
    kv = jax.random.normal(jax.random.fold_in(key, 1), (640, 512), jnp.bfloat16)
    indices = jnp.arange(640, dtype=jnp.int32)[None]
    lengths = jnp.array([640], jnp.int32)
    sinks = jnp.zeros((64,), jnp.float32)
    got, _, lse, _ = dsa.sparse_attention_forward_jax(q, kv, indices, lengths, sinks, softmax_scale=1.0)
    logits = jnp.einsum("hd,kd->hk", q[0].astype(jnp.float32), kv.astype(jnp.float32))
    maximum = jnp.maximum(jnp.max(logits, axis=-1), sinks)
    probabilities = jnp.exp(logits - maximum[:, None])
    ref = jnp.einsum("hk,kd->hd", probabilities, kv.astype(jnp.float32))
    ref /= (jnp.sum(probabilities, axis=-1) + jnp.exp(sinks - maximum))[:, None]
    np.testing.assert_allclose(np.asarray(got[0], np.float32), np.asarray(ref), atol=1e-2, rtol=2e-2)

    gradients = dsa.sparse_attention_backward_jax(q, kv, got, jnp.ones_like(got), lse, sinks, indices, lengths, softmax_scale=1.0)

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
