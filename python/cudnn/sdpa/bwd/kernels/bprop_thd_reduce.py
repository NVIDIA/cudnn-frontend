# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Origin-aware THD fold of compact per-Q-head dK/dV partials.

The shared vector reducer owns accumulation and conversion. This module only
maps sequence-local live rows to compact scratch and each output's independent
ragged origin; it adds no architecture-specific instructions or repacking.
"""

import cutlass
import cutlass.cute as cute

from cudnn.frost.tile_dsl.thd import thd_port_origin
from .bprop_chain_common import _reduce_group_vec

REDUCE_THREADS = 256
REDUCE_VECTOR = 8


@cute.kernel
def dkv_reduce_thd_origins_kernel(
    dk_part: cute.Tensor,
    dv_part: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    meta: cute.Tensor,
    origins: cute.Tensor,
    n_batch: cutlass.Int32,
    dk_origin: cutlass.Constexpr[int],
    dv_origin: cutlass.Constexpr[int],
    D: cutlass.Constexpr[int],
    group: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
):
    """Each live 16-byte output vector has one owner; both sums are FP32."""
    block, batch, kv_head = cute.arch.block_idx()
    tid = cute.arch.thread_idx()[0]
    local_vec = cutlass.Int64(block) * REDUCE_THREADS + cutlass.Int64(tid)
    local_pos = local_vec * REDUCE_VECTOR
    row = local_pos // D
    col = local_pos % D
    data = cutlass.make_array_view(meta)
    length = cutlass.Int64(data[batch])
    if row < length:
        start = cutlass.Int64(data[2 * n_batch + 1 + batch])
        h_kv = dk.shape[2]
        h_q = dk_part.shape[2]
        # The shared reducer decodes a compact logical [local row, KV head, D]
        # vector. Pointer biases place that vector at each physical output's
        # own origin and strides while its inputs retain compact KV prefixes.
        index = ((row * h_kv + cutlass.Int64(kv_head)) * D + col) // REDUCE_VECTOR
        k_token = thd_port_origin(origins, dk_origin, batch, start)
        v_token = thd_port_origin(origins, dv_origin, batch, start)
        k_bias = (
            k_token * cutlass.Int64(dk.stride[1]) + row * (cutlass.Int64(dk.stride[1]) - h_kv * D) + cutlass.Int64(kv_head) * (cutlass.Int64(dk.stride[2]) - D)
        )
        v_bias = (
            v_token * cutlass.Int64(dv.stride[1]) + row * (cutlass.Int64(dv.stride[1]) - h_kv * D) + cutlass.Int64(kv_head) * (cutlass.Int64(dv.stride[2]) - D)
        )
        _reduce_group_vec(
            dk_part.iterator.raw_ptr() + start * cutlass.Int64(dk_part.stride[1]),
            dk.iterator.raw_ptr() + k_bias,
            index,
            h_kv,
            h_q,
            D=D,
            group=group,
            io_dtype=io_dtype,
            wide=True,
        )
        _reduce_group_vec(
            dv_part.iterator.raw_ptr() + start * cutlass.Int64(dv_part.stride[1]),
            dv.iterator.raw_ptr() + v_bias,
            index,
            h_kv,
            h_q,
            D=D,
            group=group,
            io_dtype=io_dtype,
            wide=True,
        )


dkv_reduce_thd_origins_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def dkv_reduce_thd_origins_host(
    dk_part,
    dv_part,
    dk,
    dv,
    meta,
    origins,
    n_batch,
    max_skv,
    dk_origin: cutlass.Constexpr[int],
    dv_origin: cutlass.Constexpr[int],
    D: cutlass.Constexpr[int],
    group: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
    stream,
):
    """One fold launch; device lengths guard every input read and output write.

    Partials are compact [1,T,Hq,D] and outputs [1,Tphys,Hkv,D] with unit
    feature stride and vector-aligned token/head strides. D is shared by dK
    and dV and divisible by eight. The output origins may differ independently.
    """
    blocks = cute.math.max((cutlass.Int64(max_skv) * (D // REDUCE_VECTOR) + REDUCE_THREADS - 1) // REDUCE_THREADS, cutlass.Int64(1))
    dkv_reduce_thd_origins_kernel(dk_part, dv_part, dk, dv, meta, origins, n_batch, dk_origin, dv_origin, D, group, io_dtype).launch(
        grid=(blocks, n_batch, dk.shape[2]), block=(REDUCE_THREADS, 1, 1), stream=stream
    )
