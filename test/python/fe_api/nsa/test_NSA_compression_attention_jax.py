# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
ml_dtypes = pytest.importorskip("ml_dtypes")
pytest.importorskip("torch")  # Imported transitively by the NSA compression package.

import cutlass
import cutlass.cute as cute
import cutlass.jax
from cutlass.cutlass_dsl import extract_mlir_values, new_from_mlir_values
from cutlass.cute.typing import Int32


def _skip_unless_sm100():
    if not cutlass.jax.is_available():
        pytest.skip("CuTeDSL JAX extensions unavailable (jax >= 0.5 required)")
    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("JAX has no CUDA device")

    from cudnn.tensor_adapter import get_compute_capability

    major, _ = get_compute_capability()
    if major < 10:
        pytest.skip(f"Environment not supported: requires compute capability >= 10, found {major}")


def _alias_leading_batch_dim(tensor):
    shape = tensor.shape
    stride = tensor.stride
    return cute.make_tensor(
        tensor.iterator,
        cute.make_layout(
            (1, shape[0], shape[1], shape[2]),
            stride=(stride[0], stride[0], stride[1], stride[2]),
        ),
    )


def _allow_static_scheduler_values(fmha_helpers, monkeypatch):
    scheduler_type = fmha_helpers.FmhaStaticTileScheduler

    def extract(scheduler):
        values = []
        scheduler._value_counts = []
        for part in (scheduler._params, scheduler._current_work_linear_idx, scheduler._blk_coord, scheduler._grid_shape):
            part_values = extract_mlir_values(part)
            values.extend(part_values)
            scheduler._value_counts.append(len(part_values))
        return values

    def reconstruct(scheduler, values):
        parts = []
        offset = 0
        originals = (scheduler._params, scheduler._current_work_linear_idx, scheduler._blk_coord, scheduler._grid_shape)
        for original, count in zip(originals, scheduler._value_counts):
            parts.append(new_from_mlir_values(original, values[offset : offset + count]))
            offset += count
        return scheduler_type(*parts)

    monkeypatch.setattr(scheduler_type, "__extract_mlir_values__", extract)
    monkeypatch.setattr(scheduler_type, "__new_from_mlir_values__", reconstruct)


def _make_launcher(head_dim, monkeypatch):
    from cudnn.native_sparse_attention.compression import fmha_helpers
    from cudnn.native_sparse_attention.compression.fmha import BlackwellFusedMultiHeadAttentionForward

    # Static problem shapes elide scheduler MLIR values; use value-count-based
    # reconstruction so this test reaches the independent output-store path.
    _allow_static_scheduler_values(fmha_helpers, monkeypatch)
    kernel = BlackwellFusedMultiHeadAttentionForward(
        qk_acc_dtype=cutlass.Float32,
        pv_acc_dtype=cutlass.Float32,
        mma_tiler=(128, 128, head_dim),
        is_persistent=False,
        mask_type=fmha_helpers.MaskType.COMPRESSED_CAUSAL_MASK,
    )

    @cute.jit
    def launcher(
        stream,
        q,
        k,
        v,
        cum_seqlen_q,
        cum_seqlen_k,
        o,
        *,
        problem_size,
        scale_softmax_log2,
        scale_softmax,
    ):
        kernel(
            _alias_leading_batch_dim(q),
            _alias_leading_batch_dim(k),
            _alias_leading_batch_dim(v),
            _alias_leading_batch_dim(o),
            problem_size,
            cum_seqlen_q,
            cum_seqlen_k,
            None,
            scale_softmax_log2,
            scale_softmax,
            1.0,
            None,
            Int32(0),
            stream,
        )

    return launcher


def _reference(q, k, v):
    q = q.astype(np.float32)
    k = np.repeat(k.astype(np.float32), q.shape[1] // k.shape[1], axis=1)
    v = np.repeat(v.astype(np.float32), q.shape[1] // v.shape[1], axis=1)

    scores = np.einsum("qhd,khd->hqk", q, k) / math.sqrt(q.shape[-1])
    compression_factor = max(1, q.shape[0] // k.shape[0])
    q_coords = np.arange(q.shape[0])[:, None]
    k_coords = ((np.arange(k.shape[0]) + 1) * compression_factor - 1)[None, :]
    valid = k_coords <= q_coords

    scores = np.where(valid[None, :, :], scores, -np.inf)
    row_max = np.max(scores, axis=-1, keepdims=True)
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)
    probabilities = np.where(valid[None, :, :], np.exp(scores - row_max), 0.0)
    row_sum = probabilities.sum(axis=-1, keepdims=True)
    probabilities = np.divide(probabilities, row_sum, out=np.zeros_like(probabilities), where=row_sum != 0)
    return np.einsum("hqk,khd->qhd", probabilities, v)


@pytest.mark.L0
@pytest.mark.parametrize("seqlen_q", [127, 128, 129, 300])
def test_nsa_compression_jax_static_query_tile_boundaries(seqlen_q, monkeypatch):
    _skip_unless_sm100()

    head_dim = 64
    num_q_heads = 2
    num_kv_heads = 1
    seqlen_k = seqlen_q // 8
    rng = np.random.default_rng(0)
    q = (rng.standard_normal((seqlen_q, num_q_heads, head_dim), dtype=np.float32) * 0.1).astype(ml_dtypes.bfloat16)
    k = (rng.standard_normal((seqlen_k, num_kv_heads, head_dim), dtype=np.float32) * 0.1).astype(ml_dtypes.bfloat16)
    v = (rng.standard_normal((seqlen_k, num_kv_heads, head_dim), dtype=np.float32) * 0.1).astype(ml_dtypes.bfloat16)

    q_device, k_device, v_device = (jax.device_put(x) for x in (q, k, v))
    cum_seqlen_q = jax.device_put(np.array([0, seqlen_q], dtype=np.int32))
    cum_seqlen_k = jax.device_put(np.array([0, seqlen_k], dtype=np.int32))
    scale_softmax = 1.0 / math.sqrt(head_dim)
    o = cutlass.jax.cutlass_call(
        _make_launcher(head_dim, monkeypatch),
        output_shape_dtype=jax.ShapeDtypeStruct(q.shape, q.dtype),
        problem_size=(1, seqlen_q, seqlen_q, seqlen_k, num_q_heads, num_kv_heads, head_dim),
        scale_softmax_log2=scale_softmax * math.log2(math.e),
        scale_softmax=scale_softmax,
        use_static_tensors=True,
    )(q_device, k_device, v_device, cum_seqlen_q, cum_seqlen_k)

    actual = np.asarray(o).astype(np.float32)
    expected = _reference(q, k, v)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=2e-2, rtol=2e-2)
