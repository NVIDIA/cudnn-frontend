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
  deterministic);
* ``dsink_kernel`` / ``dsink_host`` -- the attention-sink gradient.

They were written for the SM120 chain (``sm120/bprop_chain_f16.py``, which
re-exports them so its public names are unchanged) and are consumed unchanged
by the SM100 adapter; a module at THIS level names the shared ownership, per
the ``kernels/__init__.py`` layout rule.  Bodies are byte-identical to their
pre-move form.
"""

from typing import Optional, Type

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims

from cudnn.sdpa.bwd.config_sm120 import ROW_ROUND
from cudnn.sdpa.bwd.kernels.sm120._common import _COPY_ELEMS, _LOG2E, ceil_div

# ---------------------------------------------------------------------------
# Preprocess kernel: delta = rowsum(dO * O) + dq_accum / dq_sem zeroing
# ---------------------------------------------------------------------------


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
):
    if cutlass.const_expr(use_pdl):
        cute.arch.griddepcontrol_launch_dependents()
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
        o_base = batch * o_batch_stride + (q_block * Q_TILE) * o_seq_stride + head * o_head_stride
        do_base = batch * do_batch_stride + (q_block * Q_TILE) * do_seq_stride + head * do_head_stride
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
                o_off = o_base + row * o_seq_stride + col0
                do_off = do_base + row * do_seq_stride + col0
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
    dot_do_o_kernel(o, do, delta, dq_accum, dq_sem, q_tile, D_QK, D_V, chunk_elems, use_pdl, deterministic).launch(
        grid=(q_blocks, o.shape[2], o.shape[0]),
        block=(256, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
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
        (out_ptr + b_idx * out_batch_stride + s_row * out_seq_stride + kv_head * out_head_stride + col).store(vec, alignment=16)
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
        if gidx < OUT_VECS:
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
        if gidx < K_VECS:
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
            if gidx < K_VECS + V_VECS:
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
    dkv_reduce_kernel(dk_ws, dv_ws, dk, dv, D_QK, D_V, group, io_dtype, use_pdl).launch(
        grid=(ceil_div(out_vecs, 256), 1, 1),
        block=(256, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
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
        lse_base = batch * lse_batch_stride + head * lse_head_stride
        delta_base = (batch * H_Q + head) * S_Q_R
        q_bound = S_Q
        if cutlass.const_expr(seq_q_lens is not None):
            q_bound = cute.math.max(cutlass.Int32(0), cute.math.min(seq_q_lens[batch], cutlass.Int32(S_Q)))
        q = cutlass.Int32(tidx)
        while q < q_bound:
            lv = (lse_ptr + lse_base + q * lse_seq_stride).load()
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
