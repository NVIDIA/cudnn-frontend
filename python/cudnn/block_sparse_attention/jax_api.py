# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental, single-device SM100 blk128 JAX block sparse attention."""

from dataclasses import dataclass
from functools import lru_cache, partial
import math

import jax
import jax.numpy as jnp

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("JAX BSA")
if requirement_error:
    raise ImportError(requirement_error)
if jax.version.__version_info__ < (0, 9, 1):
    raise ImportError("JAX BSA requires jax >= 0.9.1")

from cudnn.api_base import TupleDict
from cudnn.jax import TensorSpec, call, zeros_init
from .jax_kernels import Forward, Backward


@jax.tree_util.register_pytree_node_class
class BSAResult(TupleDict):
    def tree_flatten(self):
        return tuple(self.values()), tuple(self.keys())

    @classmethod
    def tree_unflatten(cls, keys, values):
        return cls(**dict(zip(keys, values)))


@dataclass(frozen=True)
class Config:
    qshape: tuple
    kshape: tuple
    ishape: tuple
    layout: str
    count: int
    variable: bool
    allow_empty: bool
    scale: float
    bucket_size: int
    device_id: int

    @property
    def bhsd(self):
        return self.qshape if self.layout == "bhsd" else (self.qshape[0], self.qshape[2], self.qshape[1], self.qshape[3])


def require_array(x, name, shape=None, dtype=None):
    if not isinstance(x, (jax.Array, jax.core.Tracer)):
        raise TypeError(f"{name} must be a JAX array")
    if shape is not None and tuple(x.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}, got {x.shape}")
    if dtype is not None and x.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {x.dtype}")
    if isinstance(x, jax.Array) and not isinstance(x, jax.core.Tracer) and (len(x.devices()) != 1 or next(iter(x.devices())).platform != "gpu"):
        raise ValueError(f"{name} must be on a single CUDA device")


def configuration(q_tensor, k_tensor, v_tensor, indices, nums, count, block_sizes, block_size, layout, scale, allow_empty, bucket_size):
    if layout not in ("bhsd", "bshd"):
        raise ValueError("layout must be 'bhsd' or 'bshd'")
    if block_sizes is not None or block_size != 128:
        raise ValueError("JAX BSA supports only sparse_block_size=128 and block_sizes=None")
    for name, x in (("q_tensor", q_tensor), ("k_tensor", k_tensor), ("v_tensor", v_tensor)):
        require_array(x, name, dtype=jnp.bfloat16)
        if x.ndim != 4 or any(not isinstance(n, int) or n <= 0 for n in x.shape):
            raise ValueError(f"{name} requires a positive, static rank-4 shape")
    b, h, sq, d = q_tensor.shape if layout == "bhsd" else (q_tensor.shape[0], q_tensor.shape[2], q_tensor.shape[1], q_tensor.shape[3])
    sk = k_tensor.shape[2 if layout == "bhsd" else 1]
    expected_k = (b, h, sk, d) if layout == "bhsd" else (b, sk, h, d)
    require_array(k_tensor, "k_tensor", expected_k)
    require_array(v_tensor, "v_tensor", expected_k)
    if d not in (64, 128) or sq % 128 or sk % 128:
        raise ValueError("JAX BSA requires D=64 or 128 and sequence lengths divisible by 128")
    require_array(indices, "q2k_block_index", dtype=jnp.int32)
    if indices.ndim != 4 or indices.shape[:3] != (b, h, sq // 128) or not 0 < indices.shape[3] <= sk // 128:
        raise ValueError("q2k_block_index must have shape [B,H,Sq/128,C], with 0<C<=Sk/128")
    capacity = indices.shape[-1]
    count = capacity if count is None else count
    if not isinstance(count, int) or isinstance(count, bool):
        raise TypeError("block_sparse_num must be a static integer")
    if nums is None:
        if count < 2 or count > capacity or count % 2:
            raise ValueError("fixed block_sparse_num must be even, between 2 and capacity")
    else:
        require_array(nums, "q2k_block_nums", (b, h, sq // 128), jnp.int32)
        if count < 0 or count > capacity:
            raise ValueError("block_sparse_num must be between 0 and capacity in variable-count mode")
    if not isinstance(allow_empty, bool):
        raise TypeError("allow_empty_block_nums must be a static bool")
    scale = d**-0.5 if scale is None else scale
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax_scale must be a finite, positive static scalar")
    if bucket_size is None:
        from .csrc.bwd.sm100_blk128.bsa_bwd_sm100 import sm100_blk128_bwd_default_bucketed_k2q_size_blocks

        bucket_size = sm100_blk128_bwd_default_bucketed_k2q_size_blocks(sq // 128, h)
    if not isinstance(bucket_size, int) or isinstance(bucket_size, bool) or not 0 < bucket_size <= 2**31 - 1:
        raise ValueError("bucket_size_blocks must be a positive static int32 integer")
    if sq // 128 * capacity > 2**31 - 1:
        raise ValueError("sparse edge capacity exceeds int32")
    devices = jax.local_devices()
    if jax.process_count() != 1 or len(devices) != 1 or devices[0].platform != "gpu" or str(getattr(devices[0], "compute_capability", "")) != "10.0":
        raise ValueError("JAX BSA currently requires one visible SM100 (compute capability 10.0) GPU")
    for x in (q_tensor, k_tensor, v_tensor, indices, nums):
        if isinstance(x, jax.Array) and not isinstance(x, jax.core.Tracer) and x.devices() != {devices[0]}:
            raise ValueError("all BSA arrays must be on the same visible SM100 GPU")
    return Config(
        tuple(q_tensor.shape),
        tuple(k_tensor.shape),
        tuple(indices.shape),
        layout,
        count,
        nums is not None,
        allow_empty,
        float(scale),
        bucket_size,
        devices[0].id,
    )


@lru_cache(maxsize=128)
def forward_call(c):
    b, h, sq, d = c.bhsd
    spec = TensorSpec(mode=(0, 2, 1, 3) if c.layout == "bshd" else None)
    return call(
        Forward(d, c.count, c.variable, c.allow_empty, c.scale),
        output_shape_dtype=(jax.ShapeDtypeStruct(c.qshape, jnp.bfloat16), jax.ShapeDtypeStruct((b, h, sq), jnp.float32)),
        input_spec=(spec,) * 3 + (None, None),
        output_spec=(spec, None),
        compile_options="--gpu-arch=sm_100a",
    )


@lru_cache(maxsize=128)
def backward_call(c):
    b, h, sq, d = c.bhsd
    sk = c.kshape[2 if c.layout == "bhsd" else 1]
    groups = (sq // 128 + c.bucket_size - 1) // c.bucket_size
    spec = TensorSpec(mode=(0, 2, 1, 3) if c.layout == "bhsd" else None)
    desc = jax.ShapeDtypeStruct
    qout, kout = desc(c.qshape, jnp.bfloat16), desc(c.kshape, jnp.bfloat16)
    stats = desc((b, h, sq), jnp.float32)
    counts = desc((b, h, groups, sk // 128), jnp.int32)
    offsets = desc((b, h, groups, sk // 128 + 1), jnp.int32)
    totals = desc((b, h, groups), jnp.int32)
    edges = desc((b, h, sq // 128 * (c.ishape[-1] if c.variable else c.count)), jnp.int32)
    dqacc = desc((b, h, sq * d), jnp.float32)
    dkacc = desc((b, h, sk * d) if groups > 1 else (1,), jnp.float32)
    # Output order matches Backward.__call__; counts and multi-bucket dK/dV accumulate.
    outputs = (qout, kout, kout, counts, offsets, totals, offsets, counts, edges, stats, stats, dqacc, dkacc, dkacc)
    initialized = {3: zeros_init}
    if groups > 1:
        initialized.update({12: zeros_init, 13: zeros_init})
    return call(
        Backward(d, c.count, c.variable, c.ishape[-1], c.bucket_size, groups, c.scale),
        output_shape_dtype=outputs,
        input_spec=(spec,) * 5 + (None,) * 3,
        output_spec=(spec,) * 3 + (None,) * 11,
        initialized_outputs=initialized,
        compile_options="--gpu-arch=sm_100a",
    )


def block_sparse_attention_forward_jax(
    q_tensor,
    k_tensor,
    v_tensor,
    q2k_block_index,
    block_sparse_num=None,
    block_sizes=None,
    q2k_block_nums=None,
    *,
    sparse_block_size=128,
    layout="bhsd",
    softmax_scale=None,
    allow_empty_block_nums=False,
):
    """Return ``(o_tensor, lse_tensor)``. Sparse metadata values are a caller-validated contract; see the JAX BSA documentation."""
    c = configuration(
        q_tensor,
        k_tensor,
        v_tensor,
        q2k_block_index,
        q2k_block_nums,
        block_sparse_num,
        block_sizes,
        sparse_block_size,
        layout,
        softmax_scale,
        allow_empty_block_nums,
        None,
    )
    o_tensor, lse_tensor = forward_call(c)(
        q_tensor, k_tensor, v_tensor, q2k_block_index, q2k_block_index if q2k_block_nums is None else q2k_block_nums
    )
    return BSAResult(o_tensor=o_tensor, lse_tensor=lse_tensor)


def block_sparse_attention_backward_jax(
    do_tensor,
    q_tensor,
    k_tensor,
    v_tensor,
    o_tensor,
    lse_tensor,
    q2k_block_index,
    block_sparse_num=None,
    block_sizes=None,
    q2k_block_nums=None,
    *,
    sparse_block_size=128,
    layout="bhsd",
    softmax_scale=None,
    allow_empty_block_nums=False,
    bucket_size_blocks=None,
):
    """Explicit first-order backward. Reuse the exact forward inputs, output, LSE, and options."""
    c = configuration(
        q_tensor,
        k_tensor,
        v_tensor,
        q2k_block_index,
        q2k_block_nums,
        block_sparse_num,
        block_sizes,
        sparse_block_size,
        layout,
        softmax_scale,
        allow_empty_block_nums,
        bucket_size_blocks,
    )
    require_array(do_tensor, "do_tensor", q_tensor.shape, jnp.bfloat16)
    require_array(o_tensor, "o_tensor", q_tensor.shape, jnp.bfloat16)
    require_array(lse_tensor, "lse_tensor", c.bhsd[:3], jnp.float32)
    dq_tensor, dk_tensor, dv_tensor, *workspace = backward_call(c)(
        q_tensor, k_tensor, v_tensor, do_tensor, o_tensor, lse_tensor, q2k_block_index, q2k_block_index if q2k_block_nums is None else q2k_block_nums
    )
    return BSAResult(dq_tensor=dq_tensor, dk_tensor=dk_tensor, dv_tensor=dv_tensor)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def attention(c, q, k, v, indices, nums):
    return forward_call(c)(q, k, v, indices, nums)[0]


def attention_forward(c, q, k, v, indices, nums):
    o, lse = forward_call(c)(q, k, v, indices, nums)
    return o, (q, k, v, o, lse, indices, nums)


def attention_backward(c, residual, do):
    q, k, v, o, lse, indices, nums = residual
    dq, dk, dv, *workspace = backward_call(c)(q, k, v, do, o, lse, indices, nums)
    return dq, dk, dv, None, None


attention.defvjp(attention_forward, attention_backward)


def block_sparse_attention_jax(
    q_tensor,
    k_tensor,
    v_tensor,
    q2k_block_index,
    block_sparse_num=None,
    block_sizes=None,
    q2k_block_nums=None,
    *,
    sparse_block_size=128,
    layout="bhsd",
    softmax_scale=None,
    allow_empty_block_nums=False,
    bucket_size_blocks=None,
):
    """Return O with first-order reverse-mode derivatives for Q, K, and V only."""
    c = configuration(
        q_tensor,
        k_tensor,
        v_tensor,
        q2k_block_index,
        q2k_block_nums,
        block_sparse_num,
        block_sizes,
        sparse_block_size,
        layout,
        softmax_scale,
        allow_empty_block_nums,
        bucket_size_blocks,
    )
    return attention(c, q_tensor, k_tensor, v_tensor, q2k_block_index, q2k_block_index if q2k_block_nums is None else q2k_block_nums)
