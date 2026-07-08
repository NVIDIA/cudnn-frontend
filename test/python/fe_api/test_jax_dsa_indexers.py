# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX integration coverage for DSA indexer forward and top-K."""

import pytest


def _jax_runtime():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    cutlass_jax = pytest.importorskip("cutlass.jax")
    if not cutlass_jax.is_available():
        pytest.skip("Installed JAX version is unsupported by CUTLASS JAX")
    return jax, jnp


def _supported_gpu(jax, minimum=90):
    supported_targets = {90, 100, 103, 107}
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
                major, minor = int(reported[0]), int(reported[1])
            else:
                major_text, minor_text = str(reported).split(".", 1)
                major, minor = int(major_text), int(minor_text)
        except (TypeError, ValueError):
            pytest.skip(f"JAX reported an unsupported compute capability {reported!r}")
        capabilities.append(major * 10 + minor)

    if len(set(capabilities)) != 1:
        pytest.skip(f"DSA indexers require homogeneous local GPU targets, found {capabilities}")
    capability = capabilities[0]
    if capability < minimum or capability not in supported_targets:
        pytest.skip(f"A supported JAX SM{minimum}+ GPU is not available")
    return devices[0], capability


@pytest.mark.L0
def test_jax_dsa_indexer_abstract_shapes(monkeypatch):
    jax, jnp = _jax_runtime()
    from cudnn._jax import JaxApiBase
    from cudnn.jax import (
        compactify_wrapper,
        indexer_forward_wrapper,
        indexer_top_k_wrapper,
        local_to_global_wrapper,
    )

    monkeypatch.setattr(JaxApiBase, "_local_gpu_capabilities", staticmethod(lambda _operation_name: ((object(), 100),)))

    q = jax.ShapeDtypeStruct((1, 8, 32, 128), jnp.bfloat16)
    k = jax.ShapeDtypeStruct((1, 5, 1, 128), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((1, 8, 32), jnp.bfloat16)
    fixed = jax.eval_shape(
        lambda q, k, w: indexer_forward_wrapper(q, k, w, ratio=1, target_compute_capability=100),
        q,
        k,
        w,
    )["scores"]
    assert fixed.shape == (1, 8, 5)
    assert fixed.dtype == jnp.float32

    q_sequence_major = jax.ShapeDtypeStruct((8, 2, 32, 128), jnp.bfloat16)
    k_sequence_major = jax.ShapeDtypeStruct((5, 2, 1, 128), jnp.bfloat16)
    w_sequence_major = jax.ShapeDtypeStruct((8, 2, 32), jnp.bfloat16)
    sequence_major = jax.eval_shape(
        lambda q, k, w: indexer_forward_wrapper(
            q,
            k,
            w,
            ratio=1,
            q_layout="SBHD",
            k_layout="SBHD",
            w_layout="SBH",
            output_layout="SBK",
            target_compute_capability=100,
        ),
        q_sequence_major,
        k_sequence_major,
        w_sequence_major,
    )["scores"]
    assert sequence_major.shape == (8, 2, 5)
    assert sequence_major.dtype == jnp.float32

    q_thd = jax.ShapeDtypeStruct((8, 32, 128), jnp.bfloat16)
    k_thd = jax.ShapeDtypeStruct((5, 1, 128), jnp.bfloat16)
    w_thd = jax.ShapeDtypeStruct((8, 32), jnp.bfloat16)
    cu = jax.ShapeDtypeStruct((2,), jnp.int32)
    packed = jax.eval_shape(
        lambda q, k, w, cu_q, cu_k: indexer_forward_wrapper(
            q,
            k,
            w,
            ratio=1,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=8,
            max_seqlen_k=5,
            target_compute_capability=100,
        ),
        q_thd,
        k_thd,
        w_thd,
        cu,
        cu,
    )["scores"]
    assert packed.shape == (8, 5)

    values = jax.ShapeDtypeStruct((2, 64), jnp.float32)
    lengths = jax.ShapeDtypeStruct((2,), jnp.int32)
    with_values = jax.eval_shape(
        lambda values, lengths: indexer_top_k_wrapper(
            values,
            lengths,
            8,
            target_compute_capability=100,
        ),
        values,
        lengths,
    )
    assert with_values["indices"].shape == (2, 8)
    assert with_values["indices"].dtype == jnp.int32
    assert with_values["values"].shape == (2, 8)
    without_values = jax.eval_shape(
        lambda values, lengths: indexer_top_k_wrapper(
            values,
            lengths,
            8,
            return_val=False,
            target_compute_capability=100,
        ),
        values,
        lengths,
    )
    assert without_values["indices"].shape == (2, 8)
    assert without_values["values"] is None

    local = jax.ShapeDtypeStruct((1, 2, 8), jnp.int64)
    global_indices = jax.eval_shape(
        lambda indices: local_to_global_wrapper(indices, 64, target_compute_capability=100),
        local,
    )["indices"]
    assert global_indices.shape == local.shape
    assert global_indices.dtype == jnp.int32
    packed_global = jax.eval_shape(
        lambda indices, cu_q, cu_k: local_to_global_wrapper(
            indices,
            64,
            cu_q,
            cu_k,
            target_compute_capability=100,
        ),
        jax.ShapeDtypeStruct((2, 8), jnp.int32),
        jax.ShapeDtypeStruct((2,), jnp.int32),
        jax.ShapeDtypeStruct((2,), jnp.int32),
    )["indices"]
    assert packed_global.shape == (2, 8)

    compact = jax.eval_shape(
        lambda indices: compactify_wrapper(indices, target_compute_capability=100),
        jax.ShapeDtypeStruct((1, 2, 8), jnp.int32),
    )
    assert compact["indices"].shape == (2, 8)
    assert compact["topk_length"].shape == (2,)


@pytest.mark.L0
def test_jax_indexer_forward_jit_and_numerics():
    jax, jnp = _jax_runtime()
    device, capability = _supported_gpu(jax, 90)
    from cudnn.jax import indexer_forward_wrapper

    batch, seqlen_q, seqlen_k = 2, 8, 5
    num_query_heads, head_dim = 32, 128
    q = jax.device_put(
        jax.random.normal(jax.random.key(0), (batch, seqlen_q, num_query_heads, head_dim), dtype=jnp.bfloat16),
        device,
    )
    k = jax.device_put(
        jax.random.normal(jax.random.key(1), (batch, seqlen_k, 1, head_dim), dtype=jnp.bfloat16),
        device,
    )
    w = jax.device_put(
        jax.random.normal(jax.random.key(2), (batch, seqlen_q, num_query_heads), dtype=jnp.bfloat16),
        device,
    )

    lowered = indexer_forward_wrapper.lower(
        q,
        k,
        w,
        ratio=1,
        target_compute_capability=capability,
    )
    stablehlo = lowered.as_text("stablehlo")
    assert stablehlo.count("stablehlo.custom_call") == 1
    assert "CuteDSLRT_NvJaxCutlassCall" in stablehlo
    scores = lowered.compile()(q, k, w)["scores"]
    scores.block_until_ready()

    expanded_k = jnp.repeat(k.astype(jnp.float32), num_query_heads, axis=2)
    per_head = jnp.einsum("bqhd,bkhd->bqhk", q.astype(jnp.float32), expanded_k)
    expected = jnp.sum(jnp.maximum(per_head, 0.0) * w.astype(jnp.float32)[..., None], axis=2)
    valid = jnp.arange(seqlen_k)[None, :] < (jnp.arange(seqlen_q)[:, None] + 1)
    expected = jnp.where(valid[None, :, :], expected, -jnp.inf)
    assert jnp.array_equal(jnp.isneginf(scores), jnp.isneginf(expected))
    finite = jnp.isfinite(expected)
    assert jnp.allclose(scores[finite], expected[finite], atol=5e-2, rtol=2e-2)

    q_sequence_major = jnp.transpose(q, (1, 0, 2, 3))
    k_sequence_major = jnp.transpose(k, (1, 0, 2, 3))
    w_sequence_major = jnp.transpose(w, (1, 0, 2))
    sequence_major_lowered = indexer_forward_wrapper.lower(
        q_sequence_major,
        k_sequence_major,
        w_sequence_major,
        ratio=1,
        q_layout="SBHD",
        k_layout="SBHD",
        w_layout="SBH",
        output_layout="SBK",
        target_compute_capability=capability,
    )
    sequence_major_scores = sequence_major_lowered.compile()(
        q_sequence_major,
        k_sequence_major,
        w_sequence_major,
    )["scores"]
    sequence_major_scores.block_until_ready()
    expected_sequence_major = jnp.transpose(expected, (1, 0, 2))
    assert jnp.array_equal(
        jnp.isneginf(sequence_major_scores),
        jnp.isneginf(expected_sequence_major),
    )
    finite = jnp.isfinite(expected_sequence_major)
    assert jnp.allclose(
        sequence_major_scores[finite],
        expected_sequence_major[finite],
        atol=5e-2,
        rtol=2e-2,
    )


@pytest.mark.L0
def test_jax_indexer_top_k_jit_and_numerics():
    jax, jnp = _jax_runtime()
    device, capability = _supported_gpu(jax, 90)
    from cudnn.jax import indexer_top_k_wrapper

    num_rows, num_cols, top_k = 4, 64, 8
    row = jnp.arange(num_cols, dtype=jnp.float32)
    input_values = jax.device_put(jnp.broadcast_to(row, (num_rows, num_cols)), device)
    seq_lens = jax.device_put(jnp.asarray((64, 63, 62, 61), dtype=jnp.int32), device)
    lowered = indexer_top_k_wrapper.lower(
        input_values,
        seq_lens,
        top_k,
        target_compute_capability=capability,
    )
    stablehlo = lowered.as_text("stablehlo")
    assert stablehlo.count("stablehlo.custom_call") == 1
    assert "CuteDSLRT_NvJaxCutlassCall" in stablehlo
    result = lowered.compile()(input_values, seq_lens)
    result["indices"].block_until_ready()

    valid = jnp.arange(num_cols)[None, :] < seq_lens[:, None]
    reference_values, reference_indices = jax.lax.top_k(jnp.where(valid, input_values, -jnp.inf), top_k)
    assert jnp.array_equal(jnp.sort(result["indices"], axis=-1), jnp.sort(reference_indices, axis=-1))
    assert jnp.array_equal(jnp.sort(result["values"], axis=-1), jnp.sort(reference_values, axis=-1))
