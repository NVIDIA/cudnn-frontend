# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX integration coverage for DSA backward adapters."""

import pytest


def _jax_runtime():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("cudnn")
    return jax, jnp


def _supported_gpu(jax, minimum_compute_capability):
    supported_targets = {90, 100, 103, 107}
    for device in jax.local_devices():
        if device.platform != "gpu":
            continue
        reported = getattr(device, "compute_capability", None)
        if isinstance(reported, (tuple, list)):
            major, minor = int(reported[0]), int(reported[1])
        else:
            text = str(reported)
            if "." in text:
                major_text, minor_text = text.split(".", 1)
                major, minor = int(major_text), int(minor_text)
            else:
                try:
                    major, minor = divmod(int(text), 10)
                except ValueError:
                    continue
        capability = major * 10 + minor
        if capability >= minimum_compute_capability and capability in supported_targets:
            return device, capability
    pytest.skip(f"A supported JAX SM{minimum_compute_capability}+ GPU is not available")


@pytest.mark.L0
def test_jax_dsa_backward_abstract_wrappers(monkeypatch):
    """Trace every backward signature on a CPU-only JAX installation."""

    jax, jnp = _jax_runtime()
    from cudnn._jax import JaxApiBase, disable_device_compatibility_checks
    from cudnn.deepseek_sparse_attention.indexer_backward.jax import (
        dense_indexer_backward_wrapper,
        indexer_backward_wrapper,
    )
    from cudnn.deepseek_sparse_attention.sparse_attention_backward.jax import sparse_attention_backward_wrapper

    def abstract_call(_self, _inputs, *, output_descs, **_options):
        return tuple(jnp.empty(desc.shape, dtype=desc.dtype) for desc in output_descs)

    monkeypatch.setattr(JaxApiBase, "_call_kernel", abstract_call)
    disable_device_compatibility_checks(True)
    try:
        q = jax.ShapeDtypeStruct((1, 2, 64, 128), jnp.bfloat16)
        weights = jax.ShapeDtypeStruct((1, 2, 64), jnp.bfloat16)
        k = jax.ShapeDtypeStruct((1, 128, 128), jnp.bfloat16)
        sparse_score = jax.ShapeDtypeStruct((1, 2, 128), jnp.float32)
        topk = jax.ShapeDtypeStruct((1, 2, 128), jnp.int32)
        sparse = jax.eval_shape(
            lambda q, w, k, target, predict, topk: indexer_backward_wrapper(
                q,
                w,
                k,
                target,
                predict,
                topk,
                target_compute_capability=100,
            ),
            q,
            weights,
            k,
            sparse_score,
            sparse_score,
            topk,
        )
        assert sparse["d_index_q"].shape == q.shape
        assert sparse["d_weights"].shape == weights.shape
        assert sparse["d_index_k"].shape == k.shape

        q_sparse_thd = jax.ShapeDtypeStruct((2, 64, 128), jnp.bfloat16)
        weights_sparse_thd = jax.ShapeDtypeStruct((2, 64), jnp.bfloat16)
        k_sparse_thd = jax.ShapeDtypeStruct((128, 128), jnp.bfloat16)
        score_sparse_thd = jax.ShapeDtypeStruct((2, 128), jnp.float32)
        topk_sparse_thd = jax.ShapeDtypeStruct((2, 128), jnp.int32)
        sparse_thd = jax.eval_shape(
            lambda q, w, k, target, predict, topk: indexer_backward_wrapper(
                q,
                w,
                k,
                target,
                predict,
                topk,
                topk_indices_global=True,
                target_compute_capability=100,
            ),
            q_sparse_thd,
            weights_sparse_thd,
            k_sparse_thd,
            score_sparse_thd,
            score_sparse_thd,
            topk_sparse_thd,
        )
        assert sparse_thd["d_index_q"].shape == q_sparse_thd.shape
        assert sparse_thd["d_weights"].shape == weights_sparse_thd.shape
        assert sparse_thd["d_index_k"].shape == k_sparse_thd.shape

        dense_score = jax.ShapeDtypeStruct((1, 2, 128), jnp.float32)
        dense_denom = jax.ShapeDtypeStruct((1, 2), jnp.float32)
        dense = jax.eval_shape(
            lambda q, w, k, target, denom, predict, lse: dense_indexer_backward_wrapper(
                q,
                w,
                k,
                target,
                denom,
                predict,
                lse,
                target_compute_capability=100,
            ),
            q,
            weights,
            k,
            dense_score,
            dense_denom,
            dense_score,
            dense_denom,
        )
        assert dense["d_index_q"].shape == q.shape
        assert dense["d_weights"].shape == weights.shape
        assert dense["d_index_k"].shape == k.shape

        q_thd = jax.ShapeDtypeStruct((2, 64, 128), jnp.bfloat16)
        weights_thd = jax.ShapeDtypeStruct((2, 64), jnp.bfloat16)
        k_thd = jax.ShapeDtypeStruct((128, 128), jnp.bfloat16)
        score_thd = jax.ShapeDtypeStruct((2, 128), jnp.float32)
        denom_thd = jax.ShapeDtypeStruct((2,), jnp.float32)
        cu = jax.ShapeDtypeStruct((2,), jnp.int32)
        dense_thd = jax.eval_shape(
            lambda q, w, k, target, denom, predict, lse, cu_q, cu_k: dense_indexer_backward_wrapper(
                q,
                w,
                k,
                target,
                denom,
                predict,
                lse,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=2,
                max_seqlen_k=128,
                target_compute_capability=90,
            ),
            q_thd,
            weights_thd,
            k_thd,
            score_thd,
            denom_thd,
            score_thd,
            denom_thd,
            cu,
            cu,
        )
        assert dense_thd["d_index_q"].shape == q_thd.shape
        assert dense_thd["d_index_k"].shape == k_thd.shape

        attn_q = jax.ShapeDtypeStruct((2, 64, 512), jnp.bfloat16)
        attn_kv = jax.ShapeDtypeStruct((128, 512), jnp.bfloat16)
        attn_out = jax.ShapeDtypeStruct((2, 64, 512), jnp.bfloat16)
        lse = jax.ShapeDtypeStruct((2, 64), jnp.float32)
        sink = jax.ShapeDtypeStruct((64,), jnp.float32)
        attn_topk = jax.ShapeDtypeStruct((2, 32), jnp.int32)
        attention = jax.eval_shape(
            lambda q, kv, out, dout, lse, sink, topk: sparse_attention_backward_wrapper(
                q,
                kv,
                out,
                dout,
                lse,
                sink,
                topk,
                target_compute_capability=100,
            ),
            attn_q,
            attn_kv,
            attn_out,
            attn_out,
            lse,
            sink,
            attn_topk,
        )
        assert attention["dq"].shape == attn_q.shape
        assert attention["dkv"].shape == attn_kv.shape
        assert attention["d_sink"].shape == sink.shape
    finally:
        disable_device_compatibility_checks(False)


@pytest.mark.L0
def test_jax_dense_indexer_backward_sm100_jit_and_numerics():
    jax, jnp = _jax_runtime()
    cutlass_jax = pytest.importorskip("cutlass.jax")
    if not cutlass_jax.is_available():
        pytest.skip("Installed JAX version is unsupported by CUTLASS JAX")
    device, compute_capability = _supported_gpu(jax, 100)

    from cudnn.deepseek_sparse_attention.indexer_backward.jax import dense_indexer_backward_wrapper

    batch, seqlen_q, seqlen_k = 1, 128, 128
    heads, head_dim = 64, 128
    sm_scale = 0.5
    index_q = jax.device_put(
        jax.random.normal(
            jax.random.key(40),
            (batch, seqlen_q, heads, head_dim),
            dtype=jnp.bfloat16,
        ),
        device,
    )
    weights = jax.device_put(
        jax.random.normal(
            jax.random.key(41),
            (batch, seqlen_q, heads),
            dtype=jnp.bfloat16,
        )
        * jnp.bfloat16(1.0 / heads),
        device,
    )
    index_k = jax.device_put(
        jax.random.normal(
            jax.random.key(42),
            (batch, seqlen_k, head_dim),
            dtype=jnp.bfloat16,
        ),
        device,
    )
    valid = jnp.arange(seqlen_k)[None, :] < jnp.arange(seqlen_q)[:, None] + 1
    attn_score = jax.device_put(
        jax.random.uniform(
            jax.random.key(43),
            (batch, seqlen_q, seqlen_k),
            dtype=jnp.float32,
            minval=0.1,
            maxval=1.0,
        ),
        device,
    )
    attn_score = jnp.where(valid[None, :, :], attn_score, 0.0)
    attn_l1norm = jnp.sum(attn_score, axis=-1)
    scores_per_head = jnp.einsum(
        "bqhd,bkd->bqhk",
        index_q.astype(jnp.float32),
        index_k.astype(jnp.float32),
    )
    index_score = jnp.sum(
        jnp.maximum(scores_per_head, 0.0) * (weights.astype(jnp.float32) * sm_scale)[..., None],
        axis=2,
    )
    index_lse = jax.nn.logsumexp(
        jnp.where(valid[None, :, :], index_score, -jnp.inf),
        axis=-1,
    )

    @jax.jit
    def run(q, w, k, target, target_denom, predict, predict_lse, grad_loss):
        return dense_indexer_backward_wrapper(
            q,
            w,
            k,
            target,
            target_denom,
            predict,
            predict_lse,
            grad_loss=grad_loss,
            sm_scale=sm_scale,
            target_compute_capability=compute_capability,
        )

    grad_loss = jax.device_put(jnp.ones((1,), dtype=jnp.float32), device)
    arguments = (
        index_q,
        weights,
        index_k,
        attn_score,
        attn_l1norm,
        index_score,
        index_lse,
        grad_loss,
    )
    lowered = run.lower(*arguments)
    stablehlo = lowered.as_text("stablehlo")
    assert stablehlo.count("stablehlo.custom_call") == 1
    assert "CuteDSLRT_NvJaxCutlassCall" in stablehlo

    compiled = lowered.compile()
    result = compiled(*arguments)
    result["d_index_k"].block_until_ready()
    assert result["d_index_q"].shape == index_q.shape
    assert result["d_weights"].shape == weights.shape
    assert result["d_index_k"].shape == index_k.shape
    assert all(jnp.all(jnp.isfinite(value)) for value in result.values())

    target = attn_score / attn_l1norm[..., None]
    predict = jax.nn.softmax(
        jnp.where(valid[None, :, :], index_score, -jnp.inf),
        axis=-1,
    )
    grad_scale = 1.0 / (batch * seqlen_q)
    g = jnp.where(
        valid[None, :, :],
        -jnp.maximum(target, jnp.exp(-100.0)) * grad_scale,
        0.0,
    )
    grad_signal = jnp.where(
        valid[None, :, :],
        g - predict * jnp.sum(g, axis=-1, keepdims=True),
        0.0,
    )

    def raw_scores(q, w, k):
        per_head = jnp.einsum("bqhd,bkd->bqhk", q, k)
        return jnp.sum(
            jnp.maximum(per_head, 0.0) * (w * sm_scale)[..., None],
            axis=2,
        )

    _, pullback = jax.vjp(
        raw_scores,
        index_q.astype(jnp.float32),
        weights.astype(jnp.float32),
        index_k.astype(jnp.float32),
    )
    reference = pullback(grad_signal)
    for name, expected in zip(("d_index_q", "d_weights", "d_index_k"), reference):
        actual_f32 = result[name].astype(jnp.float32).ravel()
        expected_f32 = expected.astype(jnp.float32).ravel()
        cosine = jnp.vdot(actual_f32, expected_f32) / (jnp.linalg.norm(actual_f32) * jnp.linalg.norm(expected_f32))
        relative_rms = jnp.sqrt(jnp.mean(jnp.square(actual_f32 - expected_f32))) / jnp.sqrt(jnp.mean(jnp.square(expected_f32)))
        assert cosine >= 0.97, f"{name} cosine similarity: {cosine}"
        assert relative_rms <= 0.55, f"{name} relative RMS error: {relative_rms}"

    zero_result = compiled(*arguments[:-1], jnp.zeros_like(grad_loss))
    zero_result["d_index_k"].block_until_ready()
    assert all(jnp.array_equal(value, jnp.zeros_like(value)) for value in zero_result.values())
