# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental, single-device JAX native sparse attention forward ops (packed THD)."""

from dataclasses import dataclass
from functools import lru_cache
import math

import jax
import jax.numpy as jnp

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("JAX NSA")
if requirement_error:
    raise ImportError(requirement_error)

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32

from cudnn.jax import TensorSpec, call
from .compression import fmha_helpers
from .compression.fmha import BlackwellFusedMultiHeadAttentionForward
from .selection.NSA_select_attn_fwd_hmma import HopperSelectAttentionFwd
from .top_k.nsa_top_k_reduction_fwd import FineGrainedReductionQK

LOG2_E = math.log2(math.e)
# Presents a row-major (1, T, H, X) buffer to a kernel as (1, H, T, X).
HEAD_MAJOR = TensorSpec(mode=(0, 2, 1, 3))
# A row-major (1, T, H, D) buffer whose unit batch stride equals the token stride,
# as the PyTorch path's as_strided view gives the compression kernel.
TOKEN_BATCH = TensorSpec(layout=(2, 3, 1, 0))
# A (1, T, H) LSE stored head-major (contiguous along T), strided as the PyTorch path's view.
LSE_STORAGE = TensorSpec(layout=(1, 0, 2))
LSE_HEAD_MAJOR = TensorSpec(layout=(1, 0, 2), mode=(0, 2, 1))


def require_array(x, name, shape=None, dtype=None):
    if not isinstance(x, (jax.Array, jax.core.Tracer)):
        raise TypeError(f"{name} must be a JAX array")
    if shape is not None and tuple(x.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(x.shape)}")
    if dtype is not None and x.dtype not in (dtype if isinstance(dtype, tuple) else (dtype,)):
        raise TypeError(f"{name} must have dtype {dtype}, got {x.dtype}")


def target_arch(arrays, name, capabilities):
    placed = {d for x in arrays if isinstance(x, jax.Array) and not isinstance(x, jax.core.Tracer) for d in x.devices()}
    if len(placed) > 1:
        raise ValueError(f"all {name} arrays must be on the same GPU")
    # Traced arrays carry no device: under jax.jit the call runs on JAX's default device.
    device = next(iter(placed)) if placed else jax.devices()[0]
    capability = str(getattr(device, "compute_capability", ""))
    if jax.process_count() != 1 or device.platform != "gpu" or capability not in capabilities:
        raise ValueError(f"JAX {name} requires a single process and arrays on one GPU with compute capability in {capabilities}")
    return "sm_" + capability.replace(".", "") + "a"


def check_thd(q, k, v=None):
    for name, x in (("q_tensor", q), ("k_tensor", k)) + ((("v_tensor", v),) if v is not None else ()):
        require_array(x, name, dtype=(jnp.bfloat16, jnp.float16))
        if x.ndim != 3 or 0 in x.shape:
            raise ValueError(f"{name} must use the packed (T, H, D) layout with positive extents")
    if k.dtype != q.dtype or (v is not None and v.dtype != q.dtype):
        raise TypeError("q_tensor, k_tensor and v_tensor must have the same dtype")
    if q.shape[1] % k.shape[1]:
        raise ValueError("H_q must be a multiple of H_kv")


def check_offsets(cu_q, cu_k, dtype=jnp.int32):
    require_array(cu_q, "cum_seqlen_q_tensor", dtype=dtype)
    if cu_q.ndim != 1 or cu_q.shape[0] < 2:
        raise ValueError("cum_seqlen_q_tensor must have shape (batch + 1,)")
    if cu_k is not None:
        require_array(cu_k, "cum_seqlen_k_tensor", cu_q.shape, dtype)


def static_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive static int")


def static_scale(value, default):
    value = default if value is None else value
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("scale_softmax must be a finite, positive static float")
    return float(value)


def as_4d(x):
    return x.reshape(1, *x.shape)


# The kernels keep per-trace state on the instance, and the JAX bridge traces an
# adapter more than once, so each trace builds a fresh kernel object. Selection and
# top-k compile against static shapes, as on the PyTorch path; compression reads
# the unit batch stride, which static bridge tensors set to 0, so it stays dynamic.
@dataclass(frozen=True)
class SelectionConfig:
    t: int
    h_q: int
    h_kv: int
    d: int
    d_v: int
    k_slots: int
    dtype: str
    block_size: int
    is_causal: bool
    max_s_q: int
    scale: float
    arch: str


class Selection:
    def __init__(self, c):
        self.c = c

    @cute.jit
    def __call__(self, stream, q, k, v, block_indices, block_counts, cum_seqlen_q, o, l, m):
        c = self.c
        kernel = HopperSelectAttentionFwd(
            head_dim=c.d,
            value_dim=c.d_v,
            GQA_group_size=c.h_q // c.h_kv,
            block_size=c.block_size,
            dtype=cutlass.BFloat16 if c.dtype == "bfloat16" else cutlass.Float16,
            acc_dtype=cutlass.Float32,
            causal_within_selected_blocks=c.is_causal,
        )
        kernel(q, k, v, o, l, m, block_indices, block_counts, c.max_s_q, cum_seqlen_q, c.scale, stream)


@lru_cache(maxsize=128)
def selection_call(c):
    desc = jax.ShapeDtypeStruct
    stats = desc((1, c.t, c.h_q), jnp.float32)
    return call(
        Selection(c), output_shape_dtype=(desc((1, c.t, c.h_q, c.d_v), c.dtype), stats, stats), compile_options=f"--gpu-arch={c.arch}", use_static_tensors=True
    )


def selection_attention_forward(
    q_tensor,
    k_tensor,
    v_tensor,
    block_indices_tensor,
    block_counts_tensor,
    cum_seqlen_q_tensor,
    max_s_q,
    block_size=64,
    scale_softmax=None,
    *,
    is_causal=False,
):
    """Return ``{"o_tensor", "l_tensor", "m_tensor"}`` for NSA selection attention.

    ``q``/``k``/``v`` are packed (T, H, D) and share ``cum_seqlen_q_tensor``;
    ``l``/``m`` are (T, H_q, 1) FP32. ``max_s_q`` is the static per-sequence maximum.
    """
    check_thd(q_tensor, k_tensor, v_tensor)
    t, h_q, d = q_tensor.shape
    t_k, h_kv, _ = k_tensor.shape
    d_v = v_tensor.shape[2]
    require_array(k_tensor, "k_tensor", (t, h_kv, d))
    require_array(v_tensor, "v_tensor", (t, h_kv, d_v))
    check_offsets(cum_seqlen_q_tensor, None, (jnp.int32, jnp.int64))
    require_array(block_indices_tensor, "block_indices_tensor", dtype=jnp.int32)
    if block_indices_tensor.ndim != 3 or block_indices_tensor.shape[:2] != (t, h_kv):
        raise ValueError(f"block_indices_tensor must have shape ({t}, {h_kv}, K)")
    require_array(block_counts_tensor, "block_counts_tensor", (t, h_kv), jnp.int32)
    static_int(max_s_q, "max_s_q")
    if h_q // h_kv > 16:
        raise ValueError("H_q / H_kv must be at most 16")
    if block_size not in (16, 32, 64, 128):
        raise ValueError("block_size must be 16, 32, 64, or 128")
    if not isinstance(is_causal, bool):
        raise TypeError("is_causal must be a static bool")
    c = SelectionConfig(
        t,
        h_q,
        h_kv,
        d,
        d_v,
        block_indices_tensor.shape[2],
        str(q_tensor.dtype),
        block_size,
        is_causal,
        max_s_q,
        static_scale(scale_softmax, 1.0 / math.sqrt(d)),
        target_arch(
            (q_tensor, k_tensor, v_tensor, block_indices_tensor, block_counts_tensor, cum_seqlen_q_tensor), "NSA selection attention", ("9.0", "10.0", "10.3")
        ),
    )
    o, l, m = selection_call(c)(as_4d(q_tensor), as_4d(k_tensor), as_4d(v_tensor), block_indices_tensor, block_counts_tensor, cum_seqlen_q_tensor)
    return {"o_tensor": o.reshape(t, h_q, d_v), "l_tensor": l.reshape(t, h_q, 1), "m_tensor": m.reshape(t, h_q, 1)}


@dataclass(frozen=True)
class CompressionConfig:
    t: int
    t_kv: int
    h_q: int
    h_kv: int
    d: int
    batch: int
    dtype: str
    o_dtype: str
    scale: float
    arch: str


class Compression:
    def __init__(self, c):
        self.c = c

    @cute.jit
    def __call__(self, stream, q, k, v, cum_seqlen_q, cum_seqlen_k, o, lse):
        c = self.c
        kernel = BlackwellFusedMultiHeadAttentionForward(
            cutlass.Float32,
            cutlass.Float32,
            (128, 128, c.d),
            False,
            mask_type=fmha_helpers.MaskType.COMPRESSED_CAUSAL_MASK,
        )
        problem_size = tuple(Int32(x) for x in (c.batch, c.t, c.t, c.t_kv, c.h_q, c.h_kv, c.d))
        kernel(q, k, v, o, problem_size, cum_seqlen_q, cum_seqlen_k, lse, c.scale * LOG2_E, c.scale, 1.0, None, Int32(0), stream)


@lru_cache(maxsize=128)
def compression_call(c):
    desc = jax.ShapeDtypeStruct
    return call(
        Compression(c),
        output_shape_dtype=(desc((1, c.t, c.h_q, c.d), c.o_dtype), desc((1, c.t, c.h_q), jnp.float32)),
        input_spec=(TOKEN_BATCH,) * 3 + (None, None),
        output_spec=(TOKEN_BATCH, LSE_STORAGE),
        compile_options=f"--gpu-arch={c.arch}",
    )


def compression_attention_forward(
    q_tensor,
    k_tensor,
    v_tensor,
    cum_seqlen_q_tensor,
    cum_seqlen_k_tensor,
    o_dtype=None,
    scale_softmax=None,
):
    """Return ``{"o_tensor", "lse_tensor"}`` for NSA compression attention.

    ``k``/``v`` hold the compressed (T_kv, H_kv, D) blocks with their own
    ``cum_seqlen_k_tensor``; ``lse`` is (T, H_q, 1) FP32.
    """
    check_thd(q_tensor, k_tensor, v_tensor)
    t, h_q, d = q_tensor.shape
    t_kv, h_kv, _ = k_tensor.shape
    require_array(k_tensor, "k_tensor", (t_kv, h_kv, d))
    require_array(v_tensor, "v_tensor", (t_kv, h_kv, d))
    check_offsets(cum_seqlen_q_tensor, cum_seqlen_k_tensor, (jnp.int32, jnp.int64))
    if d not in (32, 64, 128):
        raise ValueError("head dimension must be 32, 64, or 128")
    o_dtype = q_tensor.dtype if o_dtype is None else jnp.dtype(o_dtype)
    if o_dtype not in (jnp.bfloat16, jnp.float16):
        raise TypeError("o_dtype must be bfloat16 or float16")
    c = CompressionConfig(
        t,
        t_kv,
        h_q,
        h_kv,
        d,
        cum_seqlen_q_tensor.shape[0] - 1,
        str(q_tensor.dtype),
        str(o_dtype),
        static_scale(scale_softmax, 1.0 / math.sqrt(d)),
        target_arch((q_tensor, k_tensor, v_tensor, cum_seqlen_q_tensor, cum_seqlen_k_tensor), "NSA compression attention", ("10.0", "10.3")),
    )
    o, lse = compression_call(c)(as_4d(q_tensor), as_4d(k_tensor), as_4d(v_tensor), cum_seqlen_q_tensor, cum_seqlen_k_tensor)
    return {"o_tensor": o.reshape(t, h_q, d), "lse_tensor": lse.reshape(t, h_q, 1)}


@dataclass(frozen=True)
class TopKConfig:
    t: int
    t_kv: int
    h_q: int
    h_kv: int
    d: int
    batch: int
    dtype: str
    max_s_q: int
    max_s_k: int
    k_value: int
    selection_block_size: int
    compress_stride: int
    is_causal: bool
    scale: float
    arch: str


class TopK:
    def __init__(self, c):
        self.c = c

    @cute.jit
    def __call__(self, stream, q, k, lse, cum_seqlen_q, cum_seqlen_k, scores, indices):
        c = self.c
        kernel = FineGrainedReductionQK(
            element_dtype=cutlass.BFloat16 if c.dtype == "bfloat16" else cutlass.Float16,
            acc_dtype=cutlass.Float32,
            k_value=c.k_value,
            selection_block_size=c.selection_block_size,
            compress_block_sliding_stride=c.compress_stride,
            mma_tiler=(128, 128, c.d),
            is_causal=c.is_causal,
        )
        problem_size = tuple(Int32(x) for x in (c.batch, c.max_s_q, c.max_s_k, c.h_q, c.h_kv, c.d))
        kernel(problem_size, q, k, lse, scores, indices, c.scale * LOG2_E, cum_seqlen_q, cum_seqlen_k, stream)


@lru_cache(maxsize=128)
def top_k_call(c):
    desc = jax.ShapeDtypeStruct
    shape = (1, c.t, c.h_kv, c.k_value)
    return call(
        TopK(c),
        output_shape_dtype=(desc(shape, jnp.float32), desc(shape, jnp.int32)),
        input_spec=(HEAD_MAJOR, HEAD_MAJOR, LSE_HEAD_MAJOR, None, None),
        output_spec=(HEAD_MAJOR, HEAD_MAJOR),
        compile_options=f"--gpu-arch={c.arch}",
        use_static_tensors=True,
    )


def top_k_reduction_forward(
    q_tensor,
    k_tensor,
    lse_tensor,
    cum_seqlen_q_tensor,
    cum_seqlen_k_tensor,
    max_s_q,
    max_s_k,
    k_value=16,
    selection_block_size=64,
    compress_stride=32,
    is_causal=True,
    scale_softmax=None,
):
    """Return ``{"topk_scores_tensor", "topk_indices_tensor"}``, each (T, H_kv, k_value).

    ``k_tensor`` holds the compressed keys and ``lse_tensor`` the (T, H_q, 1)
    compression-attention LSE. As in the PyTorch API, the first and neighboring
    blocks are excluded, so rows may hold -inf scores and -1 indices.
    """
    check_thd(q_tensor, k_tensor)
    t, h_q, d = q_tensor.shape
    t_kv, h_kv, _ = k_tensor.shape
    require_array(k_tensor, "k_tensor", (t_kv, h_kv, d))
    require_array(lse_tensor, "lse_tensor", dtype=jnp.float32)
    if lse_tensor.shape not in ((t, h_q), (t, h_q, 1)):
        raise ValueError(f"lse_tensor must have shape ({t}, {h_q}, 1)")
    check_offsets(cum_seqlen_q_tensor, cum_seqlen_k_tensor)
    for name, value in (
        ("max_s_q", max_s_q),
        ("max_s_k", max_s_k),
        ("k_value", k_value),
        ("selection_block_size", selection_block_size),
        ("compress_stride", compress_stride),
    ):
        static_int(value, name)
    if not isinstance(is_causal, bool):
        raise TypeError("is_causal must be a static bool")
    c = TopKConfig(
        t,
        t_kv,
        h_q,
        h_kv,
        d,
        cum_seqlen_q_tensor.shape[0] - 1,
        str(q_tensor.dtype),
        max_s_q,
        max_s_k,
        k_value,
        selection_block_size,
        compress_stride,
        is_causal,
        static_scale(scale_softmax, 1.0 / math.sqrt(d)),
        target_arch((q_tensor, k_tensor, lse_tensor, cum_seqlen_q_tensor, cum_seqlen_k_tensor), "NSA top-k reduction", ("10.0", "10.3")),
    )
    scores, indices = top_k_call(c)(as_4d(q_tensor), as_4d(k_tensor), lse_tensor.reshape(1, t, h_q), cum_seqlen_q_tensor, cum_seqlen_k_tensor)
    shape = (t, h_kv, k_value)
    return {"topk_scores_tensor": scores.reshape(shape), "topk_indices_tensor": indices.reshape(shape)}
