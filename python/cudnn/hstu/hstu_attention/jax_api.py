# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental, single-device SM10x JAX HSTU attention (packed THD)."""

from dataclasses import dataclass
from functools import lru_cache
import math

import jax
import jax.numpy as jnp

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("JAX HSTU attention")
if requirement_error:
    raise ImportError(requirement_error)

import cutlass.cute as cute
from cutlass.cute.typing import BFloat16, Float16, Int32

from cudnn.jax import TensorSpec, call, zeros_init
from ._kernels.hstu_bwd import HSTUAttentionBackwardSm100
from ._kernels.hstu_fwd import HSTUAttentionForwardSm100


@dataclass(frozen=True)
class Config:
    total_q: int
    total_k: int
    heads: int
    head_dim: int
    batch: int
    dtype: str
    max_seqlen_q: int
    max_seqlen_k: int
    window_left: int
    window_right: int
    is_causal: bool
    is_local: bool
    alpha: float
    scaling_seqlen: float
    arch: str


def require_array(x, name, shape=None, dtype=None):
    if not isinstance(x, (jax.Array, jax.core.Tracer)):
        raise TypeError(f"{name} must be a JAX array")
    if shape is not None and tuple(x.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(x.shape)}")
    if dtype is not None and x.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {x.dtype}")


def gpu_arch(arrays):
    placed = {d for x in arrays if isinstance(x, jax.Array) and not isinstance(x, jax.core.Tracer) for d in x.devices()}
    if len(placed) > 1:
        raise ValueError("all HSTU arrays must be on the same GPU")
    # Traced arrays carry no device: under jax.jit the call runs on JAX's default device.
    device = next(iter(placed)) if placed else jax.devices()[0]
    capability = str(getattr(device, "compute_capability", ""))
    if jax.process_count() != 1 or device.platform != "gpu" or capability not in ("10.0", "10.3"):
        raise ValueError("JAX HSTU attention requires a single process and arrays on one SM100 or SM103 GPU")
    return "sm_" + capability.replace(".", "") + "a"


def configuration(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, window_size, alpha, scaling_seqlen):
    for name, x in (("q", q), ("k", k), ("v", v)):
        require_array(x, name)
        if x.ndim != 3 or 0 in x.shape:
            raise ValueError(f"{name} must use the packed (T, H, D) layout with positive extents")
    if q.dtype not in (jnp.bfloat16, jnp.float16):
        raise TypeError(f"q must be bfloat16 or float16, got {q.dtype}")
    total_q, heads, head_dim = q.shape
    total_k = k.shape[0]
    require_array(k, "k", (total_k, heads, head_dim), q.dtype)
    require_array(v, "v", (total_k, heads, head_dim), q.dtype)
    if head_dim not in (32, 64, 128):
        raise ValueError(f"JAX HSTU attention supports head_dim 32, 64, or 128, got {head_dim}")
    require_array(cu_seqlens_q, "cu_seqlens_q", dtype=jnp.int32)
    require_array(cu_seqlens_k, "cu_seqlens_k", cu_seqlens_q.shape, jnp.int32)
    if cu_seqlens_q.ndim != 1 or cu_seqlens_q.shape[0] < 2:
        raise ValueError("cu_seqlens_q must have shape (batch + 1,) with batch >= 1")
    for name, value in (("max_seqlen_q", max_seqlen_q), ("max_seqlen_k", max_seqlen_k)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive static int")
    if max_seqlen_q < 2:
        raise ValueError("JAX HSTU attention does not support the max_seqlen_q == 1 decode path")
    if max_seqlen_q > max_seqlen_k:
        raise ValueError("max_seqlen_q must be <= max_seqlen_k")
    if len(window_size) != 2:
        raise ValueError(f"window_size must contain two integers, got {window_size}")
    left, right = (int(w) for w in window_size)
    for name, w in (("window_size[0]", left), ("window_size[1]", right)):
        if w < -1 or w > max_seqlen_k:
            raise ValueError(f"{name} must be -1 or in [0, {max_seqlen_k}], got {w}")
    left = max_seqlen_k if left < 0 else left
    right = max_seqlen_k if right < 0 else right
    is_causal = left == max_seqlen_k and right == 0
    is_local = (left < max_seqlen_k or right < max_seqlen_k) and not is_causal
    if not isinstance(alpha, (int, float)) or not math.isfinite(alpha):
        raise ValueError("alpha must be a finite static float")
    scaling_seqlen = float(max_seqlen_q if scaling_seqlen is None else scaling_seqlen)
    if not math.isfinite(scaling_seqlen) or scaling_seqlen <= 0:
        raise ValueError("scaling_seqlen must be positive and finite")
    arch = gpu_arch((q, k, v, cu_seqlens_q, cu_seqlens_k))
    return Config(
        total_q,
        total_k,
        heads,
        head_dim,
        cu_seqlens_q.shape[0] - 1,
        str(q.dtype),
        max_seqlen_q,
        max_seqlen_k,
        left,
        right,
        is_causal,
        is_local,
        float(alpha),
        scaling_seqlen,
        arch,
    )


# The kernels keep per-trace state on the instance, and the JAX bridge traces an
# adapter more than once, so each trace builds a fresh kernel object.
class Forward:
    def __init__(self, c):
        self.c = c

    @cute.jit
    def __call__(self, stream, q, k, v, cu_seqlens_q, cu_seqlens_k, o):
        c = self.c
        kernel = HSTUAttentionForwardSm100(
            head_dim=c.head_dim,
            is_causal=c.is_causal,
            is_local=c.is_local,
        )
        kernel(
            q,
            k,
            v,
            o,
            Int32(c.max_seqlen_q),
            Int32(c.max_seqlen_k),
            cu_seqlens_q,
            cu_seqlens_k,
            c.alpha,
            c.scaling_seqlen,
            stream,
            c.window_left,
            c.window_right,
            None,
            None,
            None,
            None,
            None,
        )


class Backward:
    def __init__(self, c):
        self.c = c

    @cute.jit
    def __call__(self, stream, q, k, v, do, cu_seqlens_q, cu_seqlens_k, dq, dk, dv, workspace):
        c = self.c
        kernel = HSTUAttentionBackwardSm100(
            element_dtype=Float16 if c.dtype == "float16" else BFloat16,
            head_dim=c.head_dim,
            tile_m=128,
            tile_n=128,
            is_causal=c.is_causal,
            is_local=c.is_local,
            use_2cta_instrs=c.head_dim == 128,
        )
        problem_shape = (Int32(c.max_seqlen_q), Int32(c.max_seqlen_k), Int32(c.head_dim), ((Int32(1), Int32(c.heads)), Int32(c.batch)))
        kernel(
            problem_shape,
            q,
            k,
            v,
            dq,
            dk,
            dv,
            do,
            cu_seqlens_q,
            cu_seqlens_k,
            Int32(c.window_left),
            Int32(c.window_right),
            None,
            c.alpha,
            c.scaling_seqlen,
            workspace,
            None,
            stream,
        )


# The backward kernel reads (T, H, D) buffers as (T, D, 1, H, 1) views. Callers
# reshape to (T, 1, H, 1, D); this mode presents that row-major buffer to the kernel.
BWD_QKV_SPEC = TensorSpec(mode=(0, 4, 1, 2, 3))


@lru_cache(maxsize=128)
def forward_call(c):
    out = jax.ShapeDtypeStruct((c.total_q, c.heads, c.head_dim), c.dtype)
    return call(Forward(c), output_shape_dtype=out, compile_options=f"--gpu-arch={c.arch}")


@lru_cache(maxsize=128)
def backward_call(c):
    desc = jax.ShapeDtypeStruct
    q5 = desc((c.total_q, 1, c.heads, 1, c.head_dim), c.dtype)
    k5 = desc((c.total_k, 1, c.heads, 1, c.head_dim), c.dtype)
    padding = c.batch * 128 if c.head_dim == 128 else 0
    workspace = desc((c.heads, c.total_q + padding, (c.head_dim + 7) // 8 * 8), jnp.float32)
    return call(
        Backward(c),
        output_shape_dtype=(q5, k5, k5, workspace),
        input_spec=(BWD_QKV_SPEC,) * 4 + (None, None),
        output_spec=(BWD_QKV_SPEC,) * 3 + (None,),
        initialized_outputs={3: zeros_init},
        compile_options=f"--gpu-arch={c.arch}",
    )


def hstu_attention_forward(
    q_tensor,
    k_tensor,
    v_tensor,
    cu_seqlens_q_tensor,
    cu_seqlens_k_tensor,
    max_seqlen_q,
    max_seqlen_k,
    window_size=(-1, -1),
    alpha=1.0,
    scaling_seqlen=None,
):
    """Return ``{"o_tensor": o}`` for packed (T, H, D) HSTU attention.

    ``cu_seqlens_*`` are runtime int32 device arrays; ``max_seqlen_*``, the window,
    ``alpha`` and ``scaling_seqlen`` are static. Sequence boundaries are a
    caller-validated contract: they are never copied to the host.
    """
    c = configuration(q_tensor, k_tensor, v_tensor, cu_seqlens_q_tensor, cu_seqlens_k_tensor, max_seqlen_q, max_seqlen_k, window_size, alpha, scaling_seqlen)
    return {"o_tensor": forward_call(c)(q_tensor, k_tensor, v_tensor, cu_seqlens_q_tensor, cu_seqlens_k_tensor)}


def hstu_attention_backward(
    do_tensor,
    q_tensor,
    k_tensor,
    v_tensor,
    cu_seqlens_q_tensor,
    cu_seqlens_k_tensor,
    max_seqlen_q,
    max_seqlen_k,
    window_size=(-1, -1),
    alpha=1.0,
    scaling_seqlen=None,
):
    """Return ``{"dq_tensor", "dk_tensor", "dv_tensor"}``. Pass the forward's inputs and options."""
    c = configuration(q_tensor, k_tensor, v_tensor, cu_seqlens_q_tensor, cu_seqlens_k_tensor, max_seqlen_q, max_seqlen_k, window_size, alpha, scaling_seqlen)
    require_array(do_tensor, "do_tensor", q_tensor.shape, q_tensor.dtype)
    q5, k5, v5, do5 = (x.reshape(x.shape[0], 1, c.heads, 1, c.head_dim) for x in (q_tensor, k_tensor, v_tensor, do_tensor))
    dq, dk, dv, _ = backward_call(c)(q5, k5, v5, do5, cu_seqlens_q_tensor, cu_seqlens_k_tensor)
    return {
        "dq_tensor": dq.reshape(q_tensor.shape),
        "dk_tensor": dk.reshape(k_tensor.shape),
        "dv_tensor": dv.reshape(v_tensor.shape),
    }
