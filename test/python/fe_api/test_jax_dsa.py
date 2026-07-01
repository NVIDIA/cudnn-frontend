# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX proof-of-concept tests for the DSA indexer kernels."""

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

    from cudnn.jax import indexer_forward_wrapper, indexer_top_k_wrapper

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
    indices, values = jax.eval_shape(
        lambda values, lengths: indexer_top_k_wrapper(
            values,
            lengths,
            top_k=8,
        ),
        input_values,
        seq_lens,
    )
    assert indices.shape == (2, 8)
    assert indices.dtype == jnp.int32
    assert values.shape == (2, 8)
    assert values.dtype == jnp.float32


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
        ).scores

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
