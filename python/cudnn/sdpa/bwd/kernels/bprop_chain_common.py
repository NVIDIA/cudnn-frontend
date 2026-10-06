# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT
# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modifications are licensed under Apache-2.0. Pre-existing code retains
# its MIT terms; see LICENSING.md and THIRD_PARTY_LICENSES.txt.

"""Arch-neutral launch-chain kernels shared by the FROST SDPA backward engines.

Three small elementwise / reduction kernels that every large-head-dim backward
chain needs around its main pass, none of which touches an MMA or an
architecture-specific instruction:

* ``dot_do_o_kernel`` / ``dot_do_o_host`` -- the ``delta = rowsum(dO * O)``
  preprocess (plus the optional ``dq_accum`` / ``dq_sem`` zeroing the SM120
  fused kernel relies on);
* ``dkv_reduce_kernel`` / ``dkv_reduce_host`` -- the GQA fold of per-q-head
  dK / dV partials onto the KV heads (fixed-order fp32 accumulation, so it is
  deterministic); ``dkv_reduce_bounded_host`` is the same fold stopping at a
  device row limit (a packed THD chain's live kv total, so the caller's
  capacity tail is never written);
* ``dsink_kernel`` / ``dsink_host`` -- the attention-sink gradient;
* ``dot_do_o_scaled_host`` -- the per-tensor FP8 arm of the preprocess: O and
  dO are FP8 payloads, so ``delta`` is the raw fp8 dot product scaled by the
  device scalars ``descale_o * descale_dO`` (true units, never a host fold);
* ``fold_quant_kernel`` / ``fold_quant_host`` -- the quantized chain's tail:
  the GQA fold of per-q-head partials (fixed order, like ``dkv_reduce``) fused
  with the per-tensor FP8 epilogue -- ``* descale`` (an operand's descale the
  bf16 GEMM did not apply), the ``amax`` fold (a per-CTA max of the fp32
  pre-quantization values, ONE int32-bit-pattern ``atomicMax`` per CTA), ``* scale``
  and the cast to the gradient dtype.  ``group == 1`` makes it a plain quantize
  pass (dQ).  A persistent, 128-bit-load streaming pass (its measured geometry is
  documented at its constants).

They were written for the SM120 chain (``sm120/bprop_chain_f16.py``, which
re-exports them so its public names are unchanged) and are consumed unchanged
by the SM100 and SM107 adapters; a module at THIS level names the shared
ownership, per the ``kernels/__init__.py`` layout rule.  The ``dot`` /
``reduce`` / ``dsink`` bodies are byte-identical to their pre-move form apart
from the two trailing ``None``-specialized descale operands of the ``dot``
kernel (``None`` on every half-precision caller: the branch folds out).
"""

from typing import Optional, Type

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims

from cudnn.frost.tile_dsl.pointwise import atomic_max_f32_bits, fmax_f32, warp_abs_max_f32_shfl
from cudnn.sdpa.bwd.config_sm120 import ROW_ROUND
from cudnn.sdpa.bwd.kernels.sm120._common import _COPY_ELEMS, _LOG2E, ceil_div, wide_index

# ---------------------------------------------------------------------------
# Preprocess kernel: delta = rowsum(dO * O) + dq_accum / dq_sem zeroing
# ---------------------------------------------------------------------------

# The ``dot_do_o`` launch geometry every large-head chain uses: a 128-row q tile per block and
# 64-element (128 B) chunks per thread group.  The SM100 pointer host passes these as literals
# (``sm100/prepared_host.py``); the SM107 host reads them from here.
DOT_Q_TILE = 128
DOT_CHUNK_ELEMS = 64


@cute.kernel
def dot_do_o_kernel(
    o: cute.Tensor,  # [B, S_Q, H, D_V]
    do: cute.Tensor,  # [B, S_Q, H, D_V]
    delta: cute.Tensor,  # [B, H, S_Q_r128] fp32 out
    dq_accum: Optional[cute.Tensor],  # [B*S_Q_r128*H*D_QK] fp32 (zeroed here); None on the two-kernel dQ path
    dq_sem: Optional[cute.Tensor],  # [B*H*num_q_tiles] int32 relay turn counters (zeroed here when deterministic)
    q_tile: cutlass.Constexpr[int],
    D_QK: cutlass.Constexpr[int],  # dq_accum's head dim
    D_V: cutlass.Constexpr[int],  # O/dO's head dim
    chunk_elems: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    deterministic: cutlass.Constexpr[bool],
    descale_o: Optional[cute.Tensor],  # fp32 [1] device scalar (FP8 O payload); None = half-precision O
    descale_do: Optional[cute.Tensor],  # fp32 [1] device scalar (FP8 dO payload); paired with descale_o
):
    if cutlass.const_expr(use_pdl):
        cute.arch.griddepcontrol_launch_dependents()
    # Per-tensor FP8 O / dO: the row sum is of the RAW fp8 codes, so delta's true
    # value is the product with the two descales -- read once per thread from the
    # device scalars (never folded on the host).  None (the half chains) traces no
    # multiply at all, so their bodies are unchanged.
    dsc_o_do = descale_o.iterator.raw_ptr().load() * descale_do.iterator.raw_ptr().load() if cutlass.const_expr(descale_o is not None) else cutlass.Float32(1.0)
    q_block, head, batch = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    S_Q = o.shape[1]
    H = o.shape[2]
    S_Q_R = ceil_div(S_Q, ROW_ROUND) * ROW_ROUND
    Q_TILE = q_tile

    o_ptr = o.iterator.raw_ptr()
    do_ptr = do.iterator.raw_ptr()
    delta_ptr = delta.iterator.raw_ptr()
    if cutlass.const_expr(dq_accum is not None):
        dq_accum_ptr = dq_accum.iterator.raw_ptr()

    o_batch_stride, o_seq_stride, o_head_stride, _ = o.stride
    do_batch_stride, do_seq_stride, do_head_stride, _ = do.stride
    compact = (S_Q * H * D_V, H * D_V, D_V)
    io_strided = o.shape[3] != D_V or (o_batch_stride, o_seq_stride, o_head_stride) != compact or (do_batch_stride, do_seq_stride, do_head_stride) != compact
    if cutlass.const_expr(io_strided):
        o_base = wide_index(batch, o) * o_batch_stride + wide_index(q_block, o) * Q_TILE * o_seq_stride + wide_index(head, o) * o_head_stride
        do_base = wide_index(batch, do) * do_batch_stride + wide_index(q_block, do) * Q_TILE * do_seq_stride + wide_index(head, do) * do_head_stride
    else:
        row_stride = H * D_V
        base = ((batch * S_Q + q_block * Q_TILE) * H + head) * D_V
    delta_base = (batch * H + head) * S_Q_R + q_block * Q_TILE
    q_left = S_Q - q_block * Q_TILE

    threads_per_row = chunk_elems // _COPY_ELEMS
    rows_per_pass = 256 // threads_per_row
    col0 = (tidx % threads_per_row) * _COPY_ELEMS
    row0 = tidx // threads_per_row
    n_chunks = D_V // chunk_elems
    for rp in cutlass.range_constexpr(Q_TILE // rows_per_pass):
        row = row0 + rp * rows_per_pass
        acc = cutlass.Float32(0.0)
        if row < q_left:
            if cutlass.const_expr(io_strided):
                o_off = o_base + wide_index(row, o) * o_seq_stride + col0
                do_off = do_base + wide_index(row, do) * do_seq_stride + col0
                for chunk in cutlass.range_constexpr(n_chunks):
                    if cutlass.const_expr(o.shape[3] != D_V):
                        # Head dim is padded: gmem rows are only o.shape[3] wide.
                        if col0 + chunk * chunk_elems < o.shape[3]:
                            ov = (o_ptr + o_off + chunk * chunk_elems).load(count=_COPY_ELEMS)
                            dov = (do_ptr + do_off + chunk * chunk_elems).load(count=_COPY_ELEMS)
                            for kk in cutlass.range_constexpr(_COPY_ELEMS):
                                acc = acc + ov[kk].to(cutlass.Float32) * dov[kk].to(cutlass.Float32)
                    else:
                        ov = (o_ptr + o_off + chunk * chunk_elems).load(count=_COPY_ELEMS)
                        dov = (do_ptr + do_off + chunk * chunk_elems).load(count=_COPY_ELEMS)
                        for kk in cutlass.range_constexpr(_COPY_ELEMS):
                            acc = acc + ov[kk].to(cutlass.Float32) * dov[kk].to(cutlass.Float32)
            else:
                g_off = base + row * row_stride + col0
                for chunk in cutlass.range_constexpr(n_chunks):
                    ov = (o_ptr + g_off + chunk * chunk_elems).load(count=_COPY_ELEMS)
                    dov = (do_ptr + g_off + chunk * chunk_elems).load(count=_COPY_ELEMS)
                    for kk in cutlass.range_constexpr(_COPY_ELEMS):
                        acc = acc + ov[kk].to(cutlass.Float32) * dov[kk].to(cutlass.Float32)
        # Allreduce over the threads sharing the row (lane-contiguous).
        n_sh = 3 if cutlass.const_expr(threads_per_row == 8) else 2
        for sh in cutlass.range_constexpr(n_sh):
            acc = acc + prims.shfl_sync(
                thread_mask=0xFFFFFFFF,
                val=acc,
                offset=1 << (n_sh - 1 - sh),
                mask_and_clamp=0x1F,
                kind=prims.Shfl.BFLY,
            )
        if cutlass.const_expr(descale_o is not None):
            acc = acc * dsc_o_do
        if tidx % threads_per_row == 0:
            (delta_ptr + delta_base + row).store(acc)

    if cutlass.const_expr(use_pdl):
        cute.arch.griddepcontrol_wait()

    if cutlass.const_expr(dq_accum is not None):
        zero_rows_per_pass = 32 if cutlass.const_expr(D_QK == 32) else 16
        zero_threads_per_row = 256 // zero_rows_per_pass
        zero_row0 = tidx // zero_threads_per_row
        zero_col0 = (tidx % zero_threads_per_row) * 4
        zero4 = cutlass.Vector.from_elements(
            (
                cutlass.Float32(0.0),
                cutlass.Float32(0.0),
                cutlass.Float32(0.0),
                cutlass.Float32(0.0),
            ),
            cutlass.Float32,
        )
        dq_accum_base = ((batch * S_Q_R + q_block * Q_TILE) * H + head) * D_QK
        for im in cutlass.range_constexpr(Q_TILE // zero_rows_per_pass):
            for jn in cutlass.range_constexpr(D_QK // (zero_threads_per_row * 4)):
                addr = dq_accum_base + (zero_row0 + im * zero_rows_per_pass) * (H * D_QK) + zero_col0 + jn * zero_threads_per_row * 4
                (dq_accum_ptr + addr).store(zero4, alignment=16)

    if cutlass.const_expr(deterministic and dq_sem is not None):
        # Reset this q-tile's relay turn counter (PDL-ordered before the main
        # kernel's first acquire, like the dq_accum zeroing above).
        if tidx == 0:
            num_q_tiles = ceil_div(S_Q, Q_TILE)
            dq_sem_ptr = dq_sem.iterator.raw_ptr()
            (dq_sem_ptr + (batch * H + head) * num_q_tiles + q_block).store(cutlass.Int32(0))


dot_do_o_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def dot_do_o_host(
    o: cute.Tensor,
    do: cute.Tensor,
    delta: cute.Tensor,
    dq_accum: Optional[cute.Tensor],
    dq_sem: Optional[cute.Tensor],
    q_tile: cutlass.Constexpr[int],
    D_QK: cutlass.Constexpr[int],
    D_V: cutlass.Constexpr[int],
    chunk_elems: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    deterministic: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
):
    q_blocks = ceil_div(o.shape[1], q_tile)
    dot_do_o_kernel(o, do, delta, dq_accum, dq_sem, q_tile, D_QK, D_V, chunk_elems, use_pdl, deterministic, None, None).launch(
        grid=(q_blocks, o.shape[2], o.shape[0]),
        block=(256, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


@cute.jit
def dot_do_o_scaled_host(
    o: cute.Tensor,  # [B, S_Q, H, D_V] FP8 payload
    do: cute.Tensor,  # [B, S_Q, H, D_V] FP8 payload
    delta: cute.Tensor,  # [B, H, S_Q_r128] fp32 out, TRUE units
    descale_o: cute.Tensor,  # fp32 [1] device scalar
    descale_do: cute.Tensor,  # fp32 [1] device scalar
    q_tile: cutlass.Constexpr[int],
    D_V: cutlass.Constexpr[int],
    chunk_elems: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
):
    """``delta = rowsum(dO * O) * descale_o * descale_dO`` for the per-tensor FP8
    backward (cuDNN ``sdpa_fp8_backward``: O and dO are fp8 payloads with scalar
    descales).  The dQ accumulator / relay slots of the SM120 chain do not exist
    here, so they are not in the signature."""
    q_blocks = ceil_div(o.shape[1], q_tile)
    dot_do_o_kernel(o, do, delta, None, None, q_tile, D_V, D_V, chunk_elems, False, False, descale_o, descale_do).launch(
        grid=(q_blocks, o.shape[2], o.shape[0]),
        block=(256, 1, 1),
        stream=stream,
    )


# ---------------------------------------------------------------------------
# Reduce kernel: per-q-head dk_ws/dv_ws partials (io dtype) -> dK/dV over the group
# ---------------------------------------------------------------------------


@cute.jit
def _reduce_group_vec(
    ws_ptr,
    out_ptr,
    idx,
    h_kv,
    h_q,
    *,
    D: cutlass.Constexpr[int],
    group: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
    out_batch_stride: cutlass.Constexpr[int] = 0,
    out_seq_stride: cutlass.Constexpr[int] = 0,
    out_head_stride: cutlass.Constexpr[int] = 0,
    out_strided: cutlass.Constexpr[bool] = False,
    skv: cutlass.Constexpr[int] = 0,
):
    """Sum one 16 B output vector over the group's q-head partials (fp32,
    fixed order -> deterministic) and store it in the io dtype."""
    VEC = 8  # 8 elements per vector (16 bytes)
    pos = idx * VEC
    col = pos % D
    row = pos // D  # (b*S_KV + s)*H_KV + kv_head
    kv_head = row % h_kv
    b_seq = row // h_kv
    ws_base = (b_seq * h_q + kv_head * group) * D + col
    acc = cutlass.Array(cutlass.Float32, VEC)
    for e in cutlass.range_constexpr(VEC):
        acc[e] = cutlass.Float32(0.0)
    for g in cutlass.range_constexpr(group):
        part = (ws_ptr + ws_base + g * D).load(count=VEC)
        for e in cutlass.range_constexpr(VEC):
            acc[e] = acc[e] + part[e].to(cutlass.Float32)
    vec = cutlass.Vector.from_elements(tuple(acc[e].to(io_dtype) for e in range(VEC)), io_dtype)
    if cutlass.const_expr(out_strided):
        s_row = b_seq % skv
        b_idx = b_seq // skv
        (out_ptr + cutlass.Int64(b_idx) * out_batch_stride + cutlass.Int64(s_row) * out_seq_stride + cutlass.Int64(kv_head) * out_head_stride + col).store(
            vec, alignment=16
        )
    else:
        (out_ptr + pos).store(vec, alignment=16)


@cute.jit
def _reduce_group_vec_guarded(
    ws_ptr,
    out_ptr,
    idx,
    h_kv,
    h_q,
    *,
    D: cutlass.Constexpr[int],
    D_OUT: cutlass.Constexpr[int],
    group: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
    out_batch_stride: cutlass.Constexpr[int],
    out_seq_stride: cutlass.Constexpr[int],
    out_head_stride: cutlass.Constexpr[int],
    out_strided: cutlass.Constexpr[bool],
    skv: cutlass.Constexpr[int],
):
    """``_reduce_group_vec``, skipping the pad-column vectors when the output
    head dim is narrower than the padded workspace rows (those columns are
    zero)."""
    if cutlass.const_expr(D_OUT != D):
        if (idx * 8) % D < D_OUT:
            _reduce_group_vec(
                ws_ptr,
                out_ptr,
                idx,
                h_kv,
                h_q,
                D=D,
                group=group,
                io_dtype=io_dtype,
                out_batch_stride=out_batch_stride,
                out_seq_stride=out_seq_stride,
                out_head_stride=out_head_stride,
                out_strided=out_strided,
                skv=skv,
            )
    else:
        _reduce_group_vec(
            ws_ptr,
            out_ptr,
            idx,
            h_kv,
            h_q,
            D=D,
            group=group,
            io_dtype=io_dtype,
            out_batch_stride=out_batch_stride,
            out_seq_stride=out_seq_stride,
            out_head_stride=out_head_stride,
            out_strided=out_strided,
            skv=skv,
        )


@cute.jit
def _row_limit_value(row_limit: cute.Tensor):
    """The device row limit word (``row_limit[0]``) as an Int32."""
    return cutlass.Int32(cutlass.make_array_view(row_limit)[cutlass.Int32(0)])


@cute.kernel
def dkv_reduce_kernel(
    dk_ws: cute.Tensor,  # [B, S_KV, H_Q, D] io dtype (one dK partial per q head)
    dv_ws: cute.Tensor,  # [B, S_KV, H_Q, DV] io dtype (one dV partial per q head)
    dk: cute.Tensor,  # [B, S_KV, H_KV, D] io dtype out
    dv: cute.Tensor,  # [B, S_KV, H_KV, DV] io dtype out
    D_QK: cutlass.Constexpr[int],
    D_V: cutlass.Constexpr[int],
    group: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
    use_pdl: cutlass.Constexpr[bool],
    row_limit: Optional[cute.Tensor],  # int32 [1]: fold the kv rows [0, row_limit) of every batch only (a packed THD chain's live
    # total cu_k[B], read on device); None = every row of the extent.  The rows past the limit hold partials no kernel wrote.
):
    # One thread per 16 B output vector; serial fp32 accumulation over the
    # group's q-head slices (fixed order -> deterministic).
    if cutlass.const_expr(use_pdl):
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
    bidx, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    B = dk.shape[0]
    S_KV = dk.shape[1]
    H_KV = dk.shape[2]
    H_Q = H_KV * group
    VEC = 8  # 8 elements per vector (16 bytes)
    dk_ws_ptr = dk_ws.iterator.raw_ptr()
    dv_ws_ptr = dv_ws.iterator.raw_ptr()
    dk_ptr = dk.iterator.raw_ptr()
    dv_ptr = dv.iterator.raw_ptr()
    dk_batch_stride, dk_seq_stride, dk_head_stride, _ = dk.stride
    dv_batch_stride, dv_seq_stride, dv_head_stride, _ = dv.stride
    dk_strided = (dk_batch_stride, dk_seq_stride, dk_head_stride) != (S_KV * H_KV * D_QK, H_KV * D_QK, D_QK)
    dv_strided = (dv_batch_stride, dv_seq_stride, dv_head_stride) != (S_KV * H_KV * D_V, H_KV * D_V, D_V)
    gidx = bidx * 256 + tidx  # host launch 256 threads
    if cutlass.const_expr(D_QK == D_V):
        OUT_VECS = B * S_KV * H_KV * D_QK // VEC
        in_range = gidx < OUT_VECS
        if cutlass.const_expr(row_limit is not None):
            # The vector's kv row, decoded as _reduce_group_vec does: pos // D = (b * S_KV + s) * H_KV + kv_head.
            in_range = in_range & ((((gidx * VEC) // D_QK) // H_KV) % S_KV < _row_limit_value(row_limit))
        if in_range:
            _reduce_group_vec_guarded(
                dk_ws_ptr,
                dk_ptr,
                gidx,
                H_KV,
                H_Q,
                D=D_QK,
                D_OUT=dk.shape[3],
                group=group,
                io_dtype=io_dtype,
                out_batch_stride=dk_batch_stride,
                out_seq_stride=dk_seq_stride,
                out_head_stride=dk_head_stride,
                out_strided=dk_strided,
                skv=S_KV,
            )
            _reduce_group_vec_guarded(
                dv_ws_ptr,
                dv_ptr,
                gidx,
                H_KV,
                H_Q,
                D=D_QK,
                D_OUT=dv.shape[3],
                group=group,
                io_dtype=io_dtype,
                out_batch_stride=dv_batch_stride,
                out_seq_stride=dv_seq_stride,
                out_head_stride=dv_head_stride,
                out_strided=dv_strided,
                skv=S_KV,
            )
    else:
        # Unequal head dims: dK and dV vectors index different row widths, so
        # the flat thread range covers dK's vectors first, then dV's.
        K_VECS = B * S_KV * H_KV * D_QK // VEC
        V_VECS = B * S_KV * H_KV * D_V // VEC
        k_in_range = gidx < K_VECS
        if cutlass.const_expr(row_limit is not None):
            k_in_range = k_in_range & ((((gidx * VEC) // D_QK) // H_KV) % S_KV < _row_limit_value(row_limit))
        if k_in_range:
            _reduce_group_vec_guarded(
                dk_ws_ptr,
                dk_ptr,
                gidx,
                H_KV,
                H_Q,
                D=D_QK,
                D_OUT=dk.shape[3],
                group=group,
                io_dtype=io_dtype,
                out_batch_stride=dk_batch_stride,
                out_seq_stride=dk_seq_stride,
                out_head_stride=dk_head_stride,
                out_strided=dk_strided,
                skv=S_KV,
            )
        else:
            v_in_range = gidx < K_VECS + V_VECS
            if cutlass.const_expr(row_limit is not None):
                v_in_range = v_in_range & (((((gidx - K_VECS) * VEC) // D_V) // H_KV) % S_KV < _row_limit_value(row_limit))
            if v_in_range:
                _reduce_group_vec_guarded(
                    dv_ws_ptr,
                    dv_ptr,
                    gidx - K_VECS,
                    H_KV,
                    H_Q,
                    D=D_V,
                    D_OUT=dv.shape[3],
                    group=group,
                    io_dtype=io_dtype,
                    out_batch_stride=dv_batch_stride,
                    out_seq_stride=dv_seq_stride,
                    out_head_stride=dv_head_stride,
                    out_strided=dv_strided,
                    skv=S_KV,
                )


dkv_reduce_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def dkv_reduce_host(
    dk_ws: cute.Tensor,
    dv_ws: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    D_QK: cutlass.Constexpr[int],
    D_V: cutlass.Constexpr[int],
    group: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
):
    if cutlass.const_expr(D_QK == D_V):
        out_vecs = ceil_div(dk.shape[0] * dk.shape[1] * dk.shape[2] * D_QK, 8)
    else:
        # Split index space: one thread per dK vector plus one per dV vector.
        out_vecs = ceil_div(dk.shape[0] * dk.shape[1] * dk.shape[2] * (D_QK + D_V), 8)
    dkv_reduce_kernel(dk_ws, dv_ws, dk, dv, D_QK, D_V, group, io_dtype, use_pdl, None).launch(
        grid=(ceil_div(out_vecs, 256), 1, 1),
        block=(256, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


@cute.jit
def dkv_reduce_bounded_host(
    dk_ws: cute.Tensor,
    dv_ws: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    D_QK: cutlass.Constexpr[int],
    D_V: cutlass.Constexpr[int],
    group: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
    use_pdl: cutlass.Constexpr[bool],
    row_limit: cute.Tensor,
    stream: cuda_driver.CUstream,
):
    """:func:`dkv_reduce_host` folding only the kv rows below ``row_limit[0]`` (an int32 device word: a packed THD chain's
    live kv total ``cu_k[B]``).  The rows between the live total and the declared capacity hold partials no kernel wrote
    (dV and dK store through per-sequence clipped descriptors), so the fold must neither read them (a 0xFF-poisoned
    workspace is NaN there) nor write the caller's dK / dV at them: nothing past the packed total is written into the
    caller's gradients.  The grid is sized on the capacity (the limit is a device value, so a rebind with new lengths
    needs no host work); the vectors past the limit exit before their first load."""
    if cutlass.const_expr(D_QK == D_V):
        out_vecs = ceil_div(dk.shape[0] * dk.shape[1] * dk.shape[2] * D_QK, 8)
    else:
        out_vecs = ceil_div(dk.shape[0] * dk.shape[1] * dk.shape[2] * (D_QK + D_V), 8)
    dkv_reduce_kernel(dk_ws, dv_ws, dk, dv, D_QK, D_V, group, io_dtype, use_pdl, row_limit).launch(
        grid=(ceil_div(out_vecs, 256), 1, 1),
        block=(256, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


# ---------------------------------------------------------------------------
# Fold + quantize: per-q-head partials (fp32 under GQA, bf16 at MHA / on the bf16-dS twin) -> KV-head gradient in the io dtype
# with the per-tensor FP8 epilogue (descale, amax, scale, cast) -- one or two operand sets per launch
# ---------------------------------------------------------------------------

# The fold pass streams the partials (2 x 256 MiB of fp32 for the dV + dK pair at B=1, H_q=32, S_kv=8K, D=256), so its geometry
# is a bandwidth decision, MEASURED on Rubin (cc 10.7, 204 SMs, SM clock locked at 2376 MHz; CUDA events over 3 x 50 launches, the
# outputs and both amax words bitwise across every form tried):
# * ONE 16-byte load per partial per thread (``FOLD_QUANT_LOAD_BYTES``: 4 fp32 or 8 bf16 elements), 16-byte ALIGNED.  The previous
#   form read 8 contiguous elements per thread through a pointer whose arithmetic had dropped the tensor's alignment, and ptxas
#   scalarized every partial read into 32-bit loads (258 LDG.E per thread, no LDG.128): a warp-level load at a 32-byte per-thread
#   stride touched 8 lines for 128 bytes of data, and the pass ran at 41-56 % of the HBM pin whatever else changed.  Aligned
#   128-bit loads alone took S=8K from 51 % to 68 % of the pin; four contiguous elements per thread (every warp-level load 512
#   contiguous bytes) beat eight by 1-6 %.
# * A PERSISTENT grid-stride walk capped at ``FOLD_QUANT_CTAS_PER_SM`` CTAs of ``FOLD_QUANT_THREADS`` threads per SM in the GRID
#   (8 x 256 = 2048 threads per SM requested, not resident: the fp32 group-16 pair kernel compiles to 64 registers per thread, so at
#   most 4 such CTAs = 1024 threads are resident per SM and the grid's second half starts as the first drains): once the loads are
#   vectorized a one-item-per-CTA grid LOSES 7-41 % at S=32K, and a cap of 4 CTAs per SM loses 5-8 % against the 8 here.
# * The amax as a per-thread running max, a warp butterfly, the warp maxima through SMEM and ONE ``atomicMax`` per CTA -- at most the
#   cap (SMs x 8) per operand set per launch.  Measured against a plain-store per-CTA partial reduced by the last CTA: the atomic
#   form is faster at every S (the last CTA's serial reduce is a tail nothing hides), and removing every atomic from the previous
#   form was worth only +1.7..+5.2 % -- the atomics were never this pass's limit.
# Together: the dV + dK pair 0.0835 -> 0.0549 ms at S=8K (6.5 -> 9.9 TB/s, 77 % of the 12.85 TB/s pin), 0.305 -> 0.211 ms at
# S=32K (80 %), 0.0259 -> 0.0155 ms at S=2K.  The per-element arithmetic (the fixed g-order fp32 sum, the descale, the scale, the
# cast) and the max are unchanged, so the gradients and the amax words are bitwise the previous form's.
FOLD_QUANT_THREADS = 256
FOLD_QUANT_CTAS_PER_SM = 8
FOLD_QUANT_LOAD_BYTES = 16


def fold_quant_vec(ws_dtype) -> int:
    """Output elements per thread per item: one ``FOLD_QUANT_LOAD_BYTES`` load of the partials' dtype (4 fp32, 8 bf16)."""
    return FOLD_QUANT_LOAD_BYTES // (ws_dtype.width // 8)


def fold_quant_items(out_shape, D: int, vec: int) -> int:
    """Work items of ``FOLD_QUANT_THREADS x vec`` contiguous output elements covering the output's extent (host arithmetic over
    the static shape; the last item's tail threads exit before their first load)."""
    return ceil_div(out_shape[0] * out_shape[1] * out_shape[2] * D, vec * FOLD_QUANT_THREADS)


def fold_quant_ctas(items: int, sm_count: int) -> int:
    """Persistent CTAs for ONE operand set: the device's cap (``sm_count x FOLD_QUANT_CTAS_PER_SM``), never more than the items."""
    return max(1, min(items, sm_count * FOLD_QUANT_CTAS_PER_SM))


def fold_quant_pair_ctas(items_a: int, items_b: int, sm_count: int):
    """The pair launch's CTA split: the cap shared in proportion to the two sets' items (equal halves for dV + dK), each set at
    least one CTA and never more CTAs than items."""
    cap = sm_count * FOLD_QUANT_CTAS_PER_SM
    ctas_a = max(1, min(items_a, cap * items_a // max(1, items_a + items_b)))
    ctas_b = max(1, min(items_b, cap - ctas_a))
    return ctas_a, ctas_b


@cute.jit
def _fold_quant_set(
    ws: cute.Tensor,  # [B, S_WS, H_OUT * group, D] compact, the per-q-head partials (S_WS >= S_OUT: a padded extent); bf16 or fp32
    out: cute.Tensor,  # [B, S_OUT, H_OUT, D] the gradient in the graph's dtype: compact, or a packed THD gradient at its own token stride
    descale: Optional[cute.Tensor],  # fp32 [1]: the operand descale the bf16 GEMM did not apply (None = 1)
    scale: Optional[cute.Tensor],  # fp32 [1]: the gradient's FP8 scale (None = 1; 1.0 on half gradients)
    amax: Optional[cute.Tensor],  # fp32 [1]: max |value * descale| over the whole tensor, atomicMax'd (caller zeroes)
    D: cutlass.Constexpr[int],
    group: cutlass.Constexpr[int],
    out_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
    row_limit: Optional[cute.Tensor],  # int32 [1]: fold the rows [0, row_limit) of every batch only (a packed THD chain's live
    # total cu_k[B] / cu_q[B], read on device); None = every row of the output's extent.  The rows past the limit hold partials
    # no kernel wrote (per-sequence clipped stores), so they must reach neither the output nor the amax.
    cta,  # this CTA's index INTO THIS OPERAND SET (the pair kernel rebases set B's)
    n_ctas: cutlass.Constexpr[int],  # the set's persistent CTAs (``fold_quant_ctas`` / ``fold_quant_pair_ctas``)
    n_items: cutlass.Constexpr[int],  # the set's work items (``fold_quant_items``)
):
    """ONE operand set, CTA ``cta`` of ``n_ctas``: the items ``cta, cta + n_ctas, ...`` of ``n_items``, each ``FOLD_QUANT_THREADS``
    threads x ``VEC`` contiguous output elements, one aligned 16-byte load per partial per thread -- the body of
    :func:`fold_quant_kernel`, shared with the pair kernel.

    Per element ``acc = sum_g ws[b, s, h * group + g, :]`` in fp32, FIXED order (deterministic, like ``dkv_reduce``);
    ``true = acc * descale``; ``out = (true * scale).to(out_dtype)``; ``amax = max |true|`` as a per-thread running max over the
    CTA's items, a warp butterfly, the warp maxima through SMEM and ONE int32-bit-pattern ``atomicMax`` per CTA (the values are
    non-negative fp32, so the integer order is the float order; a zero maximum is skipped -- the caller zeroed the slot).  The
    partials are read in the WORKSPACE'S dtype: fp32 under GQA on the fp8 row (``.to(Float32)`` is the identity, so the group is
    summed from unrounded values and the gradient is rounded ONCE, here), bf16 at MHA and on the bf16-dS twin.  The output's
    extent bounds the walk, so a padded ``ws`` (rows past ``S_OUT``) is never read.  ``row_limit`` (None = the extent) bounds it
    further on device, the ``dkv_reduce_kernel`` pattern: a vector whose row is at or past the limit exits before its first load,
    so it contributes 0 to the amax and leaves the output row untouched (sdpa-invariants s5: an amax folds the live region only
    -- a 0xFF-poisoned unwritten partial is NaN, and NaN wins an integer-ordered atomicMax).  Every thread reaches the CTA
    barrier of the amax reduce: the item count is CTA-uniform and the row guard sits inside the loop.
    """
    tidx, _, _ = cute.arch.thread_idx()
    THREADS = FOLD_QUANT_THREADS
    WARPS = THREADS // 32
    VEC = fold_quant_vec(ws.element_type)
    B = out.shape[0]
    S_OUT = out.shape[1]
    H_OUT = out.shape[2]
    S_WS = ws.shape[1]
    H_WS = ws.shape[2]
    TOTAL = B * S_OUT * H_OUT * D
    ws_ptr = ws.iterator.raw_ptr()
    out_ptr = out.iterator.raw_ptr()
    # The OUTPUT may be a caller's tensor with a padded token stride (a packed THD gradient: token stride >= H * D, a multiple of 8
    # elements); the partials ``ws`` are always a compact workspace region.  Static strides (a plan fact), decided at trace time as
    # ``dkv_reduce_kernel`` does, so a compact output traces the plain linear store.
    out_batch_stride, out_seq_stride, out_head_stride, _ = out.stride
    out_strided = (out_batch_stride, out_seq_stride, out_head_stride) != (S_OUT * H_OUT * D, H_OUT * D, D)
    # The partials span ``group`` x the output: their element offset is promoted to Int64 when the span exceeds Int32 (the
    # output's own offset is the smaller one; ``wide_index``'s rule).
    ws_wide = cutlass.const_expr(ws.shape[0] * S_WS * H_WS * ws.shape[3] > 2**31 - 1)
    dsc = descale.iterator.raw_ptr().load() if cutlass.const_expr(descale is not None) else cutlass.Float32(1.0)
    sc = scale.iterator.raw_ptr().load() if cutlass.const_expr(scale is not None) else cutlass.Float32(1.0)
    limit = _row_limit_value(row_limit) if cutlass.const_expr(row_limit is not None) else cutlass.Int32(0)
    m = cutlass.Float32(0.0)
    n_iters = (cutlass.Int32(n_items) - cta + cutlass.Int32(n_ctas) - 1) // cutlass.Int32(n_ctas)
    for it in cutlass.range(n_iters):
        item = cta + it * cutlass.Int32(n_ctas)
        pos = (item * THREADS + tidx) * VEC
        live = pos < TOTAL
        if cutlass.const_expr(row_limit is not None):
            # The vector's row, decoded as below: pos // D = (b * S_OUT + s) * H_OUT + h.
            live = live & (((pos // D) // H_OUT) % S_OUT < limit)
        if live:
            col = pos % D
            row = pos // D  # (b * S_OUT + s) * H_OUT + h
            h = row % H_OUT
            bs = row // H_OUT
            s = bs % S_OUT
            b = bs // S_OUT
            b_ws = cutlass.Int64(b) if cutlass.const_expr(ws_wide) else b
            ws_base = ((b_ws * S_WS + s) * H_WS + h * group) * D + col
            acc = cutlass.Array(cutlass.Float32, VEC)
            for e in cutlass.range_constexpr(VEC):
                acc[e] = cutlass.Float32(0.0)
            for g in cutlass.range_constexpr(group):
                part = (ws_ptr + ws_base + g * D).load(alignment=FOLD_QUANT_LOAD_BYTES, count=VEC)
                for e in cutlass.range_constexpr(VEC):
                    acc[e] = acc[e] + part[e].to(cutlass.Float32)
            for e in cutlass.range_constexpr(VEC):
                acc[e] = acc[e] * dsc
                m = fmax_f32(m, cute.math.abs(acc[e]))
            vec = cutlass.Vector.from_elements(tuple((acc[e] * sc).to(out_dtype) for e in range(VEC)), out_dtype)
            if cutlass.const_expr(out_strided):
                # Int64 like _reduce_group_vec's strided store: a packed gradient with a padded token stride can push
                # ``s * out_seq_stride`` past 2^31 before the compact index does.
                (out_ptr + cutlass.Int64(b) * out_batch_stride + cutlass.Int64(s) * out_seq_stride + cutlass.Int64(h) * out_head_stride + col).store(
                    vec, alignment=min(FOLD_QUANT_LOAD_BYTES, VEC * (out_dtype.width // 8))
                )
            else:
                (out_ptr + pos).store(vec, alignment=min(FOLD_QUANT_LOAD_BYTES, VEC * (out_dtype.width // 8)))
    if cutlass.const_expr(amax is not None):
        # Every lane of every warp takes part (the guarded lanes hold 0), then the warp maxima meet in SMEM and lane 0 of warp 0
        # publishes the CTA's maximum once.
        m = warp_abs_max_f32_shfl(m)
        sRed = cutlass.Array(cutlass.Float32, WARPS, alignment=16, space=cutlass.AddressSpace.smem)
        if tidx % 32 == 0:
            sRed.subview(tidx // 32).store(m)
        prims.barrier_cta_sync()
        if tidx == 0:
            mm = sRed.subview(0).load()
            for w in cutlass.range_constexpr(1, WARPS):
                mm = fmax_f32(mm, sRed.subview(w).load())
            if mm > cutlass.Float32(0.0):
                atomic_max_f32_bits(amax, mm)


@cute.kernel
def fold_quant_kernel(
    ws: cute.Tensor,
    out: cute.Tensor,
    descale: Optional[cute.Tensor],
    scale: Optional[cute.Tensor],
    amax: Optional[cute.Tensor],
    D: cutlass.Constexpr[int],
    group: cutlass.Constexpr[int],
    out_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
    row_limit: Optional[cute.Tensor],
    n_ctas: cutlass.Constexpr[int],
    n_items: cutlass.Constexpr[int],
):
    """ONE operand set: :func:`_fold_quant_set` on every CTA (the dV fold at MHA; the twin's dQ fold)."""
    bidx, _, _ = cute.arch.block_idx()
    _fold_quant_set(ws, out, descale, scale, amax, D, group, out_dtype, row_limit, bidx, n_ctas, n_items)


fold_quant_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def fold_quant_pair_kernel(
    ws_a: cute.Tensor,
    out_a: cute.Tensor,
    descale_a: Optional[cute.Tensor],
    scale_a: Optional[cute.Tensor],
    amax_a: Optional[cute.Tensor],
    ws_b: cute.Tensor,
    out_b: cute.Tensor,
    descale_b: Optional[cute.Tensor],
    scale_b: Optional[cute.Tensor],
    amax_b: Optional[cute.Tensor],
    D: cutlass.Constexpr[int],
    group: cutlass.Constexpr[int],
    out_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
    row_limit: Optional[cute.Tensor],
    n_ctas_a: cutlass.Constexpr[int],
    n_items_a: cutlass.Constexpr[int],
    n_ctas_b: cutlass.Constexpr[int],
    n_items_b: cutlass.Constexpr[int],
):
    """TWO operand sets in ONE launch -- the fp8 row's dV and dK folds, which used to be two back-to-back launches of
    :func:`fold_quant_kernel`: CTAs ``[0, n_ctas_a)`` walk set A (dV), the rest set B (dK) with their index rebased.  The dispatch
    is CTA-uniform (every warp of a CTA takes the same arm, so the amax butterfly and the CTA barrier stay complete) and both arms
    are the single-set body, so the gradients and both amax values are bitwise what the two launches produced.  The sets share the
    group, the output dtype and the row limit (both are kv-row tensors of one chain); each has its own descale / scale / amax."""
    bidx, _, _ = cute.arch.block_idx()
    if bidx < n_ctas_a:
        _fold_quant_set(ws_a, out_a, descale_a, scale_a, amax_a, D, group, out_dtype, row_limit, bidx, n_ctas_a, n_items_a)
    else:
        _fold_quant_set(ws_b, out_b, descale_b, scale_b, amax_b, D, group, out_dtype, row_limit, bidx - n_ctas_a, n_ctas_b, n_items_b)


fold_quant_pair_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def fold_quant_host(
    ws: cute.Tensor,
    out: cute.Tensor,
    descale: Optional[cute.Tensor],
    scale: Optional[cute.Tensor],
    amax: Optional[cute.Tensor],
    D: cutlass.Constexpr[int],
    group: cutlass.Constexpr[int],
    out_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
    sm_count: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
    row_limit: Optional[cute.Tensor] = None,
):
    """Launch :func:`fold_quant_kernel` over the output's extent: ``fold_quant_ctas(items, sm_count)`` persistent CTAs of
    ``FOLD_QUANT_THREADS`` threads (``sm_count`` = the device's multiprocessors, a plan fact; a trace-time Constexpr, so it sits
    with the other Constexpr parameters ahead of ``stream`` -- both fold hosts are internal to the chain's prepared hosts, whose
    calls all carry it).  ``row_limit`` (appended, default None = dense) is an int32 ``[1]`` device word bounding the rows folded
    per batch -- a packed THD chain's live token total (``cu_k[B]`` for dV / dK, ``cu_q[B]`` for the bf16-dS twin's dQ): the grid
    stays sized on the capacity (a rebind with new lengths needs no host work) and the vectors past the limit exit before their
    first load."""
    items = fold_quant_items(out.shape, D, fold_quant_vec(ws.element_type))
    ctas = fold_quant_ctas(items, sm_count)
    fold_quant_kernel(ws, out, descale, scale, amax, D, group, out_dtype, row_limit, ctas, items).launch(
        grid=(ctas, 1, 1),
        block=(FOLD_QUANT_THREADS, 1, 1),
        stream=stream,
    )


@cute.jit
def fold_quant_pair_host(
    ws_a: cute.Tensor,
    out_a: cute.Tensor,
    descale_a: Optional[cute.Tensor],
    scale_a: Optional[cute.Tensor],
    amax_a: Optional[cute.Tensor],
    ws_b: cute.Tensor,
    out_b: cute.Tensor,
    descale_b: Optional[cute.Tensor],
    scale_b: Optional[cute.Tensor],
    amax_b: Optional[cute.Tensor],
    D: cutlass.Constexpr[int],
    group: cutlass.Constexpr[int],
    out_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
    sm_count: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
    row_limit: Optional[cute.Tensor] = None,
):
    """Launch :func:`fold_quant_pair_kernel`: set A's persistent CTAs then set B's, one grid (``fold_quant_pair_ctas``: the
    device's cap split in proportion to the sets' items -- the fp8 row's dV + dK folds; the same ``row_limit`` bounds both -- a
    packed THD chain's live kv total)."""
    items_a = fold_quant_items(out_a.shape, D, fold_quant_vec(ws_a.element_type))
    items_b = fold_quant_items(out_b.shape, D, fold_quant_vec(ws_b.element_type))
    ctas_a, ctas_b = fold_quant_pair_ctas(items_a, items_b, sm_count)
    fold_quant_pair_kernel(
        ws_a, out_a, descale_a, scale_a, amax_a, ws_b, out_b, descale_b, scale_b, amax_b, D, group, out_dtype, row_limit, ctas_a, items_a, ctas_b, items_b
    ).launch(
        grid=(ctas_a + ctas_b, 1, 1),
        block=(FOLD_QUANT_THREADS, 1, 1),
        stream=stream,
    )


@cute.kernel
def dsink_kernel(
    lse: cute.Tensor,  # [B, H_Q, S_Q] fp32 (natural log, sink folded in by the forward pass)
    delta: cute.Tensor,  # [B, H_Q, S_Q_r128] fp32 (dot_do_o output)
    sink: cute.Tensor,  # [H_Q] fp32 sink logits
    dsink: cute.Tensor,  # [H_Q] fp32 out
    seq_q_lens: Optional[cute.Tensor],  # [B] int32; None unless seq_q_lens_present
    use_pdl: cutlass.Constexpr[bool],
):
    """dsink[h] = -sum_{b,q} exp(sink[h] - lse[b,h,q]) * delta[b,h,q].

    One warp per query head, fixed reduction order -> bitwise deterministic."""
    if cutlass.const_expr(use_pdl):
        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()
    head, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    B = lse.shape[0]
    H_Q = lse.shape[1]
    S_Q = lse.shape[2]
    S_Q_R = delta.shape[2]
    lse_batch_stride, lse_head_stride, lse_seq_stride = lse.stride
    lse_ptr = lse.iterator.raw_ptr()
    delta_ptr = delta.iterator.raw_ptr()
    s_val = (sink.iterator.raw_ptr() + head).load()
    inf = cutlass.Float32(float("inf"))
    acc = cutlass.Float32(0.0)
    batch = cutlass.Int32(0)
    while batch < B:
        lse_base = wide_index(batch, lse) * lse_batch_stride + wide_index(head, lse) * lse_head_stride
        delta_base = (batch * H_Q + head) * S_Q_R
        q_bound = S_Q
        if cutlass.const_expr(seq_q_lens is not None):
            q_bound = cute.math.max(cutlass.Int32(0), cute.math.min(seq_q_lens[batch], cutlass.Int32(S_Q)))
        q = cutlass.Int32(tidx)
        while q < q_bound:
            lv = (lse_ptr + lse_base + wide_index(q, lse) * lse_seq_stride).load()
            # Padded / trimmed rows carry LSE = -inf: skip them (exp(sink - lse) overflows and inf * 0 = NaN).
            if lv > -inf and lv < inf:
                dd = (delta_ptr + delta_base + q).load()
                acc = acc + cute.math.exp2((s_val - lv) * cutlass.Float32(_LOG2E), fastmath=True) * dd
            q = q + 32
        batch = batch + 1
    for sh in cutlass.range_constexpr(5):
        acc = acc + prims.shfl_sync(
            thread_mask=0xFFFFFFFF,
            val=acc,
            offset=1 << (4 - sh),
            mask_and_clamp=0x1F,
            kind=prims.Shfl.BFLY,
        )
    if tidx == 0:
        (dsink.iterator.raw_ptr() + head).store(-acc)


dsink_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def dsink_host(
    lse: cute.Tensor,
    delta: cute.Tensor,
    sink: cute.Tensor,
    dsink: cute.Tensor,
    seq_q_lens: Optional[cute.Tensor],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
):
    dsink_kernel(lse, delta, sink, dsink, seq_q_lens, use_pdl).launch(
        grid=(lse.shape[1], 1, 1),
        block=(32, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )
