# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX integration tests for DSA score-recompute kernels."""

import pytest


def _jax_runtime():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    cutlass_jax = pytest.importorskip("cutlass.jax")
    if not cutlass_jax.is_available():
        pytest.skip("Installed JAX version is unsupported by CUTLASS JAX")
    return jax, jnp


def _sm100_device(jax):
    supported = {100, 103, 107}
    try:
        devices = tuple(jax.local_devices(backend="gpu"))
    except RuntimeError as error:
        pytest.skip(f"A JAX GPU is not available: {error}")
    if not devices:
        pytest.skip("A JAX GPU is not available")
    capabilities = []
    for device in devices:
        reported = getattr(device, "compute_capability", None)
        try:
            if isinstance(reported, (tuple, list)):
                major, minor = (int(value) for value in reported[:2])
                capability = major * 10 + minor
            elif "." in str(reported):
                major, minor = (int(value) for value in str(reported).split(".", 1))
                capability = major * 10 + minor
            else:
                capability = int(reported)
                if capability < 10:
                    capability *= 10
        except (TypeError, ValueError):
            pytest.skip(f"JAX reported an unsupported compute capability {reported!r}")
        capabilities.append(capability)
    if len(set(capabilities)) != 1 or capabilities[0] not in supported:
        pytest.skip(f"Score recompute requires homogeneous SM100/SM103/SM107 devices, found {capabilities}")
    return devices[0], capabilities[0]


@pytest.mark.L0
def test_jax_score_recompute_eval_shapes(monkeypatch):
    jax, jnp = _jax_runtime()
    from cudnn._jax import JaxApiBase
    from cudnn.jax import (
        dense_attn_score_recompute_wrapper,
        dense_indexer_score_recompute_wrapper,
        sparse_attn_score_recompute_wrapper,
        sparse_indexer_score_recompute_wrapper,
    )

    monkeypatch.setattr(JaxApiBase, "_local_gpu_capabilities", staticmethod(lambda _operation_name: ((object(), 100),)))

    sparse_q = jax.ShapeDtypeStruct((1, 4, 32, 128), jnp.bfloat16)
    sparse_k = jax.ShapeDtypeStruct((1, 128, 128), jnp.bfloat16)
    sparse_weights = jax.ShapeDtypeStruct((1, 4, 32), jnp.bfloat16)
    sparse_lse = jax.ShapeDtypeStruct((1, 4, 32), jnp.float32)
    sparse_indices = jax.ShapeDtypeStruct((1, 4, 128), jnp.int32)
    predict = jax.eval_shape(
        lambda q, k, weights, indices: sparse_indexer_score_recompute_wrapper(
            q,
            k,
            weights,
            indices,
            target_compute_capability=100,
        ),
        sparse_q,
        sparse_k,
        sparse_weights,
        sparse_indices,
    )["predict"]
    target = jax.eval_shape(
        lambda q, k, lse, indices: sparse_attn_score_recompute_wrapper(
            q,
            k,
            lse,
            indices,
            softmax_scale=0.125,
            target_compute_capability=100,
        ),
        sparse_q,
        sparse_k,
        sparse_lse,
        sparse_indices,
    )["target"]
    assert (predict.shape, predict.dtype) == ((1, 4, 128), jnp.float32)
    assert (target.shape, target.dtype) == ((1, 4, 128), jnp.float32)

    dense_q = jax.ShapeDtypeStruct((1, 4, 32, 128), jnp.bfloat16)
    dense_k = jax.ShapeDtypeStruct((1, 8, 1, 128), jnp.bfloat16)
    dense_weights = jax.ShapeDtypeStruct((1, 4, 32), jnp.bfloat16)
    dense_lse = jax.ShapeDtypeStruct((1, 4, 32), jnp.float32)
    dense_indexer = jax.eval_shape(
        lambda q, k, weights: dense_indexer_score_recompute_wrapper(q, k, weights, target_compute_capability=100),
        dense_q,
        dense_k,
        dense_weights,
    )
    dense_attention = jax.eval_shape(
        lambda q, k, lse: dense_attn_score_recompute_wrapper(
            q,
            k,
            lse,
            softmax_scale=0.125,
            target_compute_capability=100,
        ),
        dense_q,
        dense_k,
        dense_lse,
    )
    for result in (dense_indexer, dense_attention):
        assert (result["out"].shape, result["out"].dtype) == ((1, 4, 8), jnp.float32)
        assert (result["denom"].shape, result["denom"].dtype) == ((1, 4), jnp.float32)


@pytest.mark.L0
def test_jax_dense_score_recompute_jit_and_numerics():
    jax, jnp = _jax_runtime()
    device, target_compute_capability = _sm100_device(jax)
    from cudnn.jax import dense_attn_score_recompute_wrapper, dense_indexer_score_recompute_wrapper

    batch, seqlen_q, seqlen_k = 1, 2, 128
    num_query_heads, head_dim = 32, 128
    ratio = 1
    softmax_scale = head_dim**-0.5
    q = jax.device_put(jax.random.normal(jax.random.key(40), (batch, seqlen_q, num_query_heads, head_dim), dtype=jnp.bfloat16), device)
    k = jax.device_put(jax.random.normal(jax.random.key(41), (batch, seqlen_k, 1, head_dim), dtype=jnp.bfloat16), device)
    weights = jax.device_put(
        jax.random.normal(jax.random.key(42), (batch, seqlen_q, num_query_heads), dtype=jnp.bfloat16) * jnp.bfloat16(1.0 / num_query_heads),
        device,
    )
    q_causal_offsets = jax.device_put(jnp.array([63], dtype=jnp.int32), device)

    qk = jnp.einsum("bqhd,bkd->bqhk", q.astype(jnp.float32), k[:, :, 0, :].astype(jnp.float32))
    scaled_qk = qk * softmax_scale
    row_max = jnp.max(scaled_qk, axis=-1)
    lse = row_max + jnp.log(jnp.sum(jnp.exp(scaled_qk - row_max[..., None]), axis=-1))

    @jax.jit
    def run_indexer(q, k, weights, offsets):
        return dense_indexer_score_recompute_wrapper(
            q,
            k,
            weights,
            ratio=ratio,
            q_causal_offsets=offsets,
            target_compute_capability=target_compute_capability,
        )

    @jax.jit
    def run_attention(q, k, lse, offsets):
        return dense_attn_score_recompute_wrapper(
            q,
            k,
            lse,
            softmax_scale,
            ratio=ratio,
            q_causal_offsets=offsets,
            target_compute_capability=target_compute_capability,
        )

    indexer_lowered = run_indexer.lower(q, k, weights, q_causal_offsets)
    attention_lowered = run_attention.lower(q, k, lse, q_causal_offsets)
    for lowered in (indexer_lowered, attention_lowered):
        stablehlo = lowered.as_text("stablehlo")
        assert stablehlo.count("stablehlo.custom_call") == 1
        assert "CuteDSLRT_NvJaxCutlassCall" in stablehlo

    indexer = indexer_lowered.compile()(q, k, weights, q_causal_offsets)
    attention = attention_lowered.compile()(q, k, lse, q_causal_offsets)
    attention["denom"].block_until_ready()

    q_positions = jnp.arange(seqlen_q, dtype=jnp.int32)[None, :, None]
    col_limits = (q_causal_offsets[:, None, None] + q_positions + 1) // ratio
    valid = jnp.arange(seqlen_k, dtype=jnp.int32)[None, None, :] < col_limits

    indexer_scores = jnp.sum(jnp.maximum(qk, 0.0) * weights.astype(jnp.float32)[..., None], axis=2)
    expected_indexer_out = jnp.where(valid, indexer_scores, -jnp.inf)
    indexer_max = jnp.max(expected_indexer_out, axis=-1)
    expected_indexer_denom = indexer_max + jnp.log(jnp.sum(jnp.exp(expected_indexer_out - indexer_max[..., None]), axis=-1))

    attention_scores = jnp.sum(jnp.exp(scaled_qk - lse[..., None]), axis=2)
    expected_attention_out = jnp.where(valid, attention_scores, -jnp.inf)
    expected_attention_denom = jnp.sum(jnp.where(valid, attention_scores, 0.0), axis=-1)

    assert jnp.allclose(indexer["out"], expected_indexer_out, atol=6e-3, rtol=2e-2)
    assert jnp.allclose(indexer["denom"], expected_indexer_denom, atol=6e-3, rtol=2e-2)
    assert jnp.allclose(attention["out"], expected_attention_out, atol=6e-3, rtol=2e-2)
    assert jnp.allclose(attention["denom"], expected_attention_denom, atol=6e-3, rtol=2e-2)
    assert jnp.all(jnp.isneginf(indexer["out"][~valid]))
    assert jnp.all(jnp.isneginf(attention["out"][~valid]))


@pytest.mark.L0
def test_jax_sparse_indexer_score_recompute_jit():
    jax, jnp = _jax_runtime()
    device, target_compute_capability = _sm100_device(jax)
    from cudnn.jax import sparse_indexer_score_recompute_wrapper

    batch, seqlen_q, seqlen_k = 1, 2, 128
    num_query_heads, head_dim, topk = 32, 128, 128
    q = jax.device_put(jax.random.normal(jax.random.key(10), (batch, seqlen_q, num_query_heads, head_dim), dtype=jnp.bfloat16), device)
    k = jax.device_put(jax.random.normal(jax.random.key(11), (batch, seqlen_k, head_dim), dtype=jnp.bfloat16), device)
    weights = jax.device_put(
        jax.random.normal(jax.random.key(12), (batch, seqlen_q, num_query_heads), dtype=jnp.bfloat16) * jnp.bfloat16(1.0 / num_query_heads),
        device,
    )
    indices = jax.device_put(
        jnp.broadcast_to(jnp.arange(seqlen_k - 1, -1, -1, dtype=jnp.int32), (batch, seqlen_q, topk)),
        device,
    )

    lowered = sparse_indexer_score_recompute_wrapper.lower(
        q,
        k,
        weights,
        indices,
        target_compute_capability=target_compute_capability,
    )
    stablehlo = lowered.as_text("stablehlo")
    assert stablehlo.count("stablehlo.custom_call") == 1
    assert "CuteDSLRT_NvJaxCutlassCall" in stablehlo
    predict = lowered.compile()(q, k, weights, indices)["predict"]
    predict.block_until_ready()

    k_gather = jax.vmap(lambda batch_k, batch_indices: batch_k[batch_indices])(k.astype(jnp.float32), indices)
    scores_per_head = jnp.einsum("bqhd,bqtd->bqht", q.astype(jnp.float32), k_gather)
    scores = jnp.sum(jnp.maximum(scores_per_head, 0.0) * weights.astype(jnp.float32)[..., None], axis=2)
    expected = jax.nn.softmax(scores, axis=-1)
    assert jnp.allclose(predict, expected, atol=3e-3, rtol=2e-2)


@pytest.mark.L0
def test_jax_sparse_attn_score_recompute_jit():
    jax, jnp = _jax_runtime()
    device, target_compute_capability = _sm100_device(jax)
    from cudnn.jax import sparse_attn_score_recompute_wrapper

    batch, seqlen_q, seqlen_k = 1, 2, 128
    num_query_heads, head_dim, topk = 32, 128, 128
    softmax_scale = head_dim**-0.5
    q = jax.device_put(jax.random.normal(jax.random.key(20), (batch, seqlen_q, num_query_heads, head_dim), dtype=jnp.bfloat16), device)
    k = jax.device_put(jax.random.normal(jax.random.key(21), (batch, seqlen_k, head_dim), dtype=jnp.bfloat16), device)
    indices = jax.device_put(
        jnp.broadcast_to(jnp.arange(seqlen_k - 1, -1, -1, dtype=jnp.int32), (batch, seqlen_q, topk)),
        device,
    )
    full_scores = jnp.einsum("bqhd,bkd->bqhk", q.astype(jnp.float32), k.astype(jnp.float32)) * softmax_scale
    row_max = jnp.max(full_scores, axis=-1)
    lse = row_max + jnp.log(jnp.sum(jnp.exp(full_scores - row_max[..., None]), axis=-1))

    lowered = sparse_attn_score_recompute_wrapper.lower(
        q,
        k,
        lse,
        indices,
        softmax_scale=softmax_scale,
        target_compute_capability=target_compute_capability,
    )
    stablehlo = lowered.as_text("stablehlo")
    assert stablehlo.count("stablehlo.custom_call") == 1
    assert "CuteDSLRT_NvJaxCutlassCall" in stablehlo
    target = lowered.compile()(q, k, lse, indices)["target"]
    target.block_until_ready()

    k_gather = jax.vmap(lambda batch_k, batch_indices: batch_k[batch_indices])(k.astype(jnp.float32), indices)
    sparse_scores = jnp.einsum("bqhd,bqtd->bqht", q.astype(jnp.float32), k_gather) * softmax_scale
    expected = jnp.sum(jnp.exp(sparse_scores - lse[..., None]), axis=2)
    expected /= jnp.sum(expected, axis=-1, keepdims=True)
    assert jnp.allclose(target, expected, atol=3e-3, rtol=2e-2)


@pytest.mark.L0
def test_jax_sparse_score_recompute_global_indices_and_partial_lengths():
    jax, jnp = _jax_runtime()
    device, target_compute_capability = _sm100_device(jax)
    from cudnn.jax import sparse_attn_score_recompute_wrapper, sparse_indexer_score_recompute_wrapper

    batch, seqlen_q, seqlen_k = 2, 1, 128
    num_query_heads, head_dim, topk = 32, 128, 128
    softmax_scale = head_dim**-0.5
    q = jax.device_put(jax.random.normal(jax.random.key(30), (batch, seqlen_q, num_query_heads, head_dim), dtype=jnp.bfloat16), device)
    k = jax.device_put(jax.random.normal(jax.random.key(31), (batch, seqlen_k, head_dim), dtype=jnp.bfloat16), device)
    weights = jax.device_put(
        jax.random.normal(jax.random.key(32), (batch, seqlen_q, num_query_heads), dtype=jnp.bfloat16) * jnp.bfloat16(1.0 / num_query_heads),
        device,
    )
    local_indices = jnp.broadcast_to(jnp.arange(topk, dtype=jnp.int32), (batch, seqlen_q, topk))
    global_indices = jax.device_put(local_indices + jnp.arange(batch, dtype=jnp.int32)[:, None, None] * seqlen_k, device)
    topk_length = jax.device_put(jnp.array([[0], [73]], dtype=jnp.int32), device)

    full_scores = jnp.einsum("bqhd,bkd->bqhk", q.astype(jnp.float32), k.astype(jnp.float32)) * softmax_scale
    row_max = jnp.max(full_scores, axis=-1)
    lse = row_max + jnp.log(jnp.sum(jnp.exp(full_scores - row_max[..., None]), axis=-1))

    @jax.jit
    def run(q, k, weights, lse, indices, lengths):
        predict = sparse_indexer_score_recompute_wrapper(
            q,
            k,
            weights,
            indices,
            topk_length=lengths,
            topk_indices_global=True,
            target_compute_capability=target_compute_capability,
        )["predict"]
        target = sparse_attn_score_recompute_wrapper(
            q,
            k,
            lse,
            indices,
            softmax_scale,
            topk_length=lengths,
            topk_indices_global=True,
            target_compute_capability=target_compute_capability,
        )["target"]
        return predict, target

    lowered = run.lower(q, k, weights, lse, global_indices, topk_length)
    assert lowered.as_text("stablehlo").count("stablehlo.custom_call") == 2
    predict, target = lowered.compile()(q, k, weights, lse, global_indices, topk_length)
    target.block_until_ready()

    k_gather = jax.vmap(lambda batch_k, batch_indices: batch_k[batch_indices])(k.astype(jnp.float32), local_indices)
    qk = jnp.einsum("bqhd,bqtd->bqht", q.astype(jnp.float32), k_gather)
    valid = jnp.arange(topk)[None, None, :] < topk_length[..., None]

    indexer_scores = jnp.sum(jnp.maximum(qk, 0.0) * weights.astype(jnp.float32)[..., None], axis=2)
    expected_predict = jax.nn.softmax(jnp.where(valid, indexer_scores, -jnp.inf), axis=-1)
    expected_predict = jnp.where(topk_length[..., None] > 0, expected_predict, 0.0)

    attention_scores = jnp.sum(jnp.exp(qk * softmax_scale - lse[..., None]), axis=2)
    attention_scores = jnp.where(valid, attention_scores, 0.0)
    attention_denom = jnp.sum(attention_scores, axis=-1, keepdims=True)
    expected_target = jnp.where(attention_denom > 0, attention_scores / jnp.maximum(attention_denom, 1e-12), 0.0)

    assert jnp.allclose(predict, expected_predict, atol=3e-3, rtol=2e-2)
    assert jnp.allclose(target, expected_target, atol=3e-3, rtol=2e-2)
    assert jnp.array_equal(predict[0], jnp.zeros_like(predict[0]))
    assert jnp.array_equal(target[0], jnp.zeros_like(target[0]))
    assert jnp.array_equal(predict[1, :, 73:], jnp.zeros_like(predict[1, :, 73:]))
    assert jnp.array_equal(target[1, :, 73:], jnp.zeros_like(target[1, :, 73:]))
