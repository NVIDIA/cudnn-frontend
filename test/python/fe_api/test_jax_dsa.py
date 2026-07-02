# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX integration tests for the DSA kernels."""

import pytest


def _jax_dependencies():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("cutlass.jax")
    return jax, jnp


def _gpu_device(jax, minimum_compute_capability):
    gpu_devices = [device for device in jax.local_devices() if device.platform == "gpu"]
    for device in gpu_devices:
        capability = getattr(device, "compute_capability", None)
        if capability is None:
            continue
        if isinstance(capability, (tuple, list)):
            major, minor = (int(value) for value in capability[:2])
            capability_number = major * 10 + minor
        elif "." in str(capability):
            major, minor = (int(value) for value in str(capability).split(".", 1))
            capability_number = major * 10 + minor
        else:
            capability_number = int(capability)
            if capability_number < 10:
                capability_number *= 10
        if capability_number >= minimum_compute_capability:
            return device
    pytest.skip(f"JAX SM{minimum_compute_capability}+ device is not available")


@pytest.mark.L0
def test_jax_dsa_abstract_shapes():
    jax, jnp = _jax_dependencies()

    from cudnn.jax import (
        TupleDict,
        indexer_forward_wrapper,
        indexer_top_k_wrapper,
        sparse_attn_score_recompute_wrapper,
        sparse_indexer_score_recompute_wrapper,
    )

    q = jax.ShapeDtypeStruct((1, 4, 32, 128), jnp.bfloat16)
    k = jax.ShapeDtypeStruct((1, 5, 1, 128), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((1, 4, 32), jnp.bfloat16)
    (scores,) = jax.eval_shape(
        lambda q, k, w: indexer_forward_wrapper(q, k, w, ratio=1),
        q,
        k,
        w,
    )
    assert scores.shape == (1, 4, 5)
    assert scores.dtype == jnp.float32

    input_values = jax.ShapeDtypeStruct((2, 64), jnp.float32)
    seq_lens = jax.ShapeDtypeStruct((2,), jnp.int32)
    topk_result = jax.eval_shape(
        lambda values, lengths: indexer_top_k_wrapper(
            values,
            lengths,
            top_k=8,
        ),
        input_values,
        seq_lens,
    )
    assert isinstance(topk_result, TupleDict)
    assert tuple(topk_result.keys()) == ("indices", "values")
    indices, values = topk_result
    assert indices.shape == (2, 8)
    assert indices.dtype == jnp.int32
    assert values.shape == (2, 8)
    assert values.dtype == jnp.float32

    sparse_q = jax.ShapeDtypeStruct((1, 4, 32, 128), jnp.bfloat16)
    sparse_k = jax.ShapeDtypeStruct((1, 128, 128), jnp.bfloat16)
    sparse_topk = jax.ShapeDtypeStruct((1, 4, 128), jnp.int32)
    sparse_weights = jax.ShapeDtypeStruct((1, 4, 32), jnp.bfloat16)
    predict = jax.eval_shape(
        sparse_indexer_score_recompute_wrapper,
        sparse_q,
        sparse_k,
        sparse_weights,
        sparse_topk,
    )["predict"]
    assert predict.shape == (1, 4, 128)
    assert predict.dtype == jnp.float32

    sparse_lse = jax.ShapeDtypeStruct((1, 4, 32), jnp.float32)
    target = jax.eval_shape(
        lambda q, k, lse, indices: sparse_attn_score_recompute_wrapper(
            q,
            k,
            lse,
            indices,
            softmax_scale=0.125,
        ),
        sparse_q,
        sparse_k,
        sparse_lse,
        sparse_topk,
    )["target"]
    assert target.shape == (1, 4, 128)
    assert target.dtype == jnp.float32


@pytest.mark.L0
def test_jax_indexer_forward_jit():
    jax, jnp = _jax_dependencies()
    device = _gpu_device(jax, 100)

    from cudnn.jax import indexer_forward_wrapper

    batch, seqlen_q, seqlen_k = 1, 4, 5
    num_query_heads, num_kv_heads, head_dim = 32, 1, 128
    sm_scale = 0.5
    q = jax.device_put(
        jax.random.normal(
            jax.random.key(0),
            (batch, seqlen_q, num_query_heads, head_dim),
            dtype=jnp.bfloat16,
        ),
        device,
    )
    k = jax.device_put(
        jax.random.normal(
            jax.random.key(1),
            (batch, seqlen_k, num_kv_heads, head_dim),
            dtype=jnp.bfloat16,
        ),
        device,
    )
    w = jax.device_put(
        jax.random.normal(
            jax.random.key(2),
            (batch, seqlen_q, num_query_heads),
            dtype=jnp.bfloat16,
        ),
        device,
    )

    @jax.jit
    def run(q, k, w):
        return indexer_forward_wrapper(
            q,
            k,
            w,
            ratio=1,
            sm_scale=sm_scale,
        )["scores"]

    lowered = run.lower(q, k, w)
    stablehlo = lowered.as_text("stablehlo")
    assert stablehlo.count("stablehlo.custom_call") == 1
    assert "CuteDSLRT_NvJaxCutlassCall" in stablehlo

    compiled = lowered.compile()
    q_second = -q
    scores_first = compiled(q, k, w)
    scores_second = compiled(q_second, k, w)
    scores_second.block_until_ready()

    k_f32 = jnp.repeat(k.astype(jnp.float32), num_query_heads, axis=2)
    valid = jnp.arange(seqlen_k)[None, :] < (seqlen_k - seqlen_q + jnp.arange(seqlen_q)[:, None] + 1)

    def reference(q_value):
        dots = jnp.einsum("bqhd,bkhd->bqkh", q_value.astype(jnp.float32), k_f32)
        result = sm_scale * jnp.sum(
            jnp.maximum(dots, 0.0) * w.astype(jnp.float32)[:, :, None, :],
            axis=-1,
        )
        return jnp.where(valid[None, :, :], result, -jnp.inf)

    for scores, expected in (
        (scores_first, reference(q)),
        (scores_second, reference(q_second)),
    ):
        assert scores.shape == (batch, seqlen_q, seqlen_k)
        assert jnp.array_equal(jnp.isneginf(scores), jnp.isneginf(expected))
        finite = jnp.isfinite(expected)
        assert jnp.allclose(scores[finite], expected[finite], atol=5e-2, rtol=2e-2)


@pytest.mark.L0
def test_jax_indexer_top_k_jit_uses_hidden_workspace():
    jax, jnp = _jax_dependencies()
    device = _gpu_device(jax, 90)

    from cudnn.jax import indexer_top_k_wrapper

    # More than 148 rows selects the large-occupancy configuration, whose
    # FP32 candidate capacity is 256. These 257 unique values all round to
    # the same FP16 coarse key, so every full-length row deterministically
    # spills one candidate to global scratch and reads it during refinement.
    num_rows, num_cols, top_k = 149, 257, 8
    row_values = 1.0 + jnp.arange(num_cols, dtype=jnp.float32) * 1.0e-6
    input_values = jax.device_put(
        jnp.broadcast_to(row_values, (num_rows, num_cols)),
        device,
    )
    seq_lens = jax.device_put(
        jnp.full((num_rows,), num_cols, dtype=jnp.int32).at[0].set(num_cols - 1),
        device,
    )

    @jax.jit
    def run(values, lengths):
        return indexer_top_k_wrapper(values, lengths, top_k=top_k)

    lowered = run.lower(input_values, seq_lens)
    stablehlo = lowered.as_text("stablehlo")
    assert stablehlo.count("stablehlo.custom_call") == 1
    assert "CuteDSLRT_NvJaxCutlassCall" in stablehlo
    # Float32 top-K uses two int32 scratch planes. The custom call returns this
    # allocation internally even though the public JAX result has only two leaves.
    assert f"{num_rows}x2x{num_cols}xi32" in stablehlo

    indices, values = lowered.compile()(input_values, seq_lens)
    indices.block_until_ready()
    valid = jnp.arange(num_cols)[None, :] < seq_lens[:, None]
    masked_values = jnp.where(valid, input_values, -jnp.inf)
    reference_values, reference_indices = jax.lax.top_k(masked_values, top_k)

    assert indices.dtype == jnp.int32
    assert values.dtype == jnp.float32
    assert jnp.array_equal(jnp.sort(indices, axis=-1), jnp.sort(reference_indices, axis=-1))
    assert jnp.allclose(
        jnp.sort(values, axis=-1),
        jnp.sort(reference_values, axis=-1),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.L0
def test_jax_sparse_indexer_score_recompute_jit():
    jax, jnp = _jax_dependencies()
    device = _gpu_device(jax, 100)

    from cudnn.jax import sparse_indexer_score_recompute_wrapper

    batch, seqlen_q, seqlen_k = 1, 2, 128
    num_query_heads, head_dim, topk = 32, 128, 128
    q = jax.device_put(
        jax.random.normal(
            jax.random.key(10),
            (batch, seqlen_q, num_query_heads, head_dim),
            dtype=jnp.bfloat16,
        ),
        device,
    )
    k = jax.device_put(
        jax.random.normal(
            jax.random.key(11),
            (batch, seqlen_k, head_dim),
            dtype=jnp.bfloat16,
        ),
        device,
    )
    weights = jax.device_put(
        jax.random.normal(
            jax.random.key(12),
            (batch, seqlen_q, num_query_heads),
            dtype=jnp.bfloat16,
        )
        * jnp.bfloat16(1.0 / num_query_heads),
        device,
    )
    topk_indices = jax.device_put(
        jnp.broadcast_to(
            jnp.arange(seqlen_k - 1, -1, -1, dtype=jnp.int32),
            (batch, seqlen_q, topk),
        ),
        device,
    )

    @jax.jit
    def run(q, k, weights, indices):
        return sparse_indexer_score_recompute_wrapper(
            q,
            k,
            weights,
            indices,
        )["predict"]

    lowered = run.lower(q, k, weights, topk_indices)
    stablehlo = lowered.as_text("stablehlo")
    assert stablehlo.count("stablehlo.custom_call") == 1
    assert "CuteDSLRT_NvJaxCutlassCall" in stablehlo

    predict = lowered.compile()(q, k, weights, topk_indices)
    predict.block_until_ready()

    k_gather = jax.vmap(lambda batch_k, batch_indices: batch_k[batch_indices])(
        k.astype(jnp.float32),
        topk_indices,
    )
    scores_per_head = jnp.einsum(
        "bqhd,bqtd->bqht",
        q.astype(jnp.float32),
        k_gather,
    )
    scores = jnp.sum(
        jnp.maximum(scores_per_head, 0.0) * weights.astype(jnp.float32)[..., None],
        axis=2,
    )
    expected = jax.nn.softmax(scores, axis=-1)

    assert predict.shape == (batch, seqlen_q, topk)
    assert predict.dtype == jnp.float32
    assert jnp.allclose(predict, expected, atol=3e-3, rtol=2e-2)


@pytest.mark.L0
def test_jax_sparse_attn_score_recompute_jit():
    jax, jnp = _jax_dependencies()
    device = _gpu_device(jax, 100)

    from cudnn.jax import sparse_attn_score_recompute_wrapper

    batch, seqlen_q, seqlen_k = 1, 2, 128
    num_query_heads, head_dim, topk = 32, 128, 128
    softmax_scale = head_dim**-0.5
    q = jax.device_put(
        jax.random.normal(
            jax.random.key(20),
            (batch, seqlen_q, num_query_heads, head_dim),
            dtype=jnp.bfloat16,
        ),
        device,
    )
    k = jax.device_put(
        jax.random.normal(
            jax.random.key(21),
            (batch, seqlen_k, head_dim),
            dtype=jnp.bfloat16,
        ),
        device,
    )
    topk_indices = jax.device_put(
        jnp.broadcast_to(
            jnp.arange(seqlen_k - 1, -1, -1, dtype=jnp.int32),
            (batch, seqlen_q, topk),
        ),
        device,
    )

    full_scores = (
        jnp.einsum(
            "bqhd,bkd->bqhk",
            q.astype(jnp.float32),
            k.astype(jnp.float32),
        )
        * softmax_scale
    )
    row_max = jnp.max(full_scores, axis=-1)
    lse = row_max + jnp.log(jnp.sum(jnp.exp(full_scores - row_max[..., None]), axis=-1))

    @jax.jit
    def run(q, k, lse, indices):
        return sparse_attn_score_recompute_wrapper(
            q,
            k,
            lse,
            indices,
            softmax_scale=softmax_scale,
        )["target"]

    lowered = run.lower(q, k, lse, topk_indices)
    stablehlo = lowered.as_text("stablehlo")
    assert stablehlo.count("stablehlo.custom_call") == 1
    assert "CuteDSLRT_NvJaxCutlassCall" in stablehlo

    target = lowered.compile()(q, k, lse, topk_indices)
    target.block_until_ready()

    k_gather = jax.vmap(lambda batch_k, batch_indices: batch_k[batch_indices])(
        k.astype(jnp.float32),
        topk_indices,
    )
    sparse_scores = (
        jnp.einsum(
            "bqhd,bqtd->bqht",
            q.astype(jnp.float32),
            k_gather,
        )
        * softmax_scale
    )
    expected = jnp.sum(jnp.exp(sparse_scores - lse[..., None]), axis=2)
    expected /= jnp.sum(expected, axis=-1, keepdims=True)

    assert target.shape == (batch, seqlen_q, topk)
    assert target.dtype == jnp.float32
    assert jnp.allclose(target, expected, atol=3e-3, rtol=2e-2)


@pytest.mark.L0
def test_jax_sparse_score_recompute_global_indices_and_partial_lengths():
    jax, jnp = _jax_dependencies()
    device = _gpu_device(jax, 100)

    from cudnn.jax import (
        sparse_attn_score_recompute_wrapper,
        sparse_indexer_score_recompute_wrapper,
    )

    batch, seqlen_q, seqlen_k = 2, 1, 128
    num_query_heads, head_dim, topk = 32, 128, 128
    softmax_scale = head_dim**-0.5
    q = jax.device_put(
        jax.random.normal(
            jax.random.key(30),
            (batch, seqlen_q, num_query_heads, head_dim),
            dtype=jnp.bfloat16,
        ),
        device,
    )
    k = jax.device_put(
        jax.random.normal(
            jax.random.key(31),
            (batch, seqlen_k, head_dim),
            dtype=jnp.bfloat16,
        ),
        device,
    )
    weights = jax.device_put(
        jax.random.normal(
            jax.random.key(32),
            (batch, seqlen_q, num_query_heads),
            dtype=jnp.bfloat16,
        )
        * jnp.bfloat16(1.0 / num_query_heads),
        device,
    )
    local_indices = jnp.broadcast_to(
        jnp.arange(topk, dtype=jnp.int32),
        (batch, seqlen_q, topk),
    )
    global_indices = jax.device_put(
        local_indices + jnp.arange(batch, dtype=jnp.int32)[:, None, None] * seqlen_k,
        device,
    )
    topk_length = jax.device_put(
        jnp.array([[0], [73]], dtype=jnp.int32),
        device,
    )

    full_scores = (
        jnp.einsum(
            "bqhd,bkd->bqhk",
            q.astype(jnp.float32),
            k.astype(jnp.float32),
        )
        * softmax_scale
    )
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
        )["predict"]
        target = sparse_attn_score_recompute_wrapper(
            q,
            k,
            lse,
            indices,
            softmax_scale,
            topk_length=lengths,
            topk_indices_global=True,
        )["target"]
        return predict, target

    lowered = run.lower(q, k, weights, lse, global_indices, topk_length)
    stablehlo = lowered.as_text("stablehlo")
    assert stablehlo.count("stablehlo.custom_call") == 2
    predict, target = lowered.compile()(
        q,
        k,
        weights,
        lse,
        global_indices,
        topk_length,
    )
    target.block_until_ready()

    k_gather = jax.vmap(lambda batch_k, batch_indices: batch_k[batch_indices])(
        k.astype(jnp.float32),
        local_indices,
    )
    qk = jnp.einsum(
        "bqhd,bqtd->bqht",
        q.astype(jnp.float32),
        k_gather,
    )
    valid = jnp.arange(topk)[None, None, :] < topk_length[..., None]

    indexer_scores = jnp.sum(
        jnp.maximum(qk, 0.0) * weights.astype(jnp.float32)[..., None],
        axis=2,
    )
    expected_predict = jax.nn.softmax(
        jnp.where(valid, indexer_scores, -jnp.inf),
        axis=-1,
    )
    expected_predict = jnp.where(topk_length[..., None] > 0, expected_predict, 0.0)

    attention_scores = jnp.sum(
        jnp.exp(qk * softmax_scale - lse[..., None]),
        axis=2,
    )
    attention_scores = jnp.where(valid, attention_scores, 0.0)
    attention_denom = jnp.sum(attention_scores, axis=-1, keepdims=True)
    expected_target = jnp.where(
        attention_denom > 0,
        attention_scores / jnp.maximum(attention_denom, 1e-12),
        0.0,
    )

    assert jnp.allclose(predict, expected_predict, atol=3e-3, rtol=2e-2)
    assert jnp.allclose(target, expected_target, atol=3e-3, rtol=2e-2)
    assert jnp.array_equal(predict[0], jnp.zeros_like(predict[0]))
    assert jnp.array_equal(target[0], jnp.zeros_like(target[0]))
    assert jnp.array_equal(predict[1, :, 73:], jnp.zeros_like(predict[1, :, 73:]))
    assert jnp.array_equal(target[1, :, 73:], jnp.zeros_like(target[1, :, 73:]))
