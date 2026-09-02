# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as _cuda_driver

TENSOR_MAP_QWORDS = 128 // 8


@cute.jit
def write_thd_meta(meta, ql, kl, lens_form: cutlass.Int32, n_batch: cutlass.Int32) -> None:
    """Single-thread body of the device-side THD metadata build (issue #552),
    shared by the SM100 setup kernel (which follows it with the per-batch O
    TMA descriptors) and the SM120 meta-only kernel. Writes the
    [seq_kv_lens(B) | cu_seqlens_q(B+1) | cu_seqlens_k(B+1)] buffer from the
    caller's length tensors — ``(B,)`` per-batch lengths (serial cumsum; B
    is small) or the ``(B+1,)`` cu prefix-sum form, per side via
    ``lens_form`` (bit 0: Q is cu, bit 1: KV is cu). cu prefixes are
    NORMALIZED (element 0 subtracted): the packed buffers are addressed from
    token 0, so a cu tensor sliced from a larger prefix means the same
    lengths — and the host can no longer validate ``cu[0] == 0`` (Rule 3),
    so an unnormalized base must not leak into the offsets the tiles and
    the dead-unit sentinel read. Callers run this under ``elect_sync``."""
    cuq0 = n_batch
    cuk0 = cutlass.Int32(2) * n_batch + cutlass.Int32(1)
    q_is_cu = (lens_form & cutlass.Int32(1)) != cutlass.Int32(0)
    kv_is_cu = (lens_form & cutlass.Int32(2)) != cutlass.Int32(0)
    if q_is_cu:
        base_q = cutlass.Int32(ql[0])
        for b in cutlass.range(0, n_batch + cutlass.Int32(1), 1, unroll=1):
            meta[cuq0 + b] = cutlass.Int32(ql[b]) - base_q
    else:
        acc = cutlass.Int32(0)
        meta[cuq0] = cutlass.Int32(0)
        for b in cutlass.range(0, n_batch, 1, unroll=1):
            acc = acc + cutlass.Int32(ql[b])
            meta[cuq0 + b + cutlass.Int32(1)] = acc
    if kv_is_cu:
        base_k = cutlass.Int32(kl[0])
        meta[cuk0] = cutlass.Int32(0)
        for b in cutlass.range(0, n_batch, 1, unroll=1):
            meta[cuk0 + b + cutlass.Int32(1)] = cutlass.Int32(kl[b + cutlass.Int32(1)]) - base_k
            meta[b] = cutlass.Int32(kl[b + cutlass.Int32(1)]) - cutlass.Int32(kl[b])
    else:
        acc_k = cutlass.Int32(0)
        meta[cuk0] = cutlass.Int32(0)
        for b in cutlass.range(0, n_batch, 1, unroll=1):
            lkv = cutlass.Int32(kl[b])
            meta[b] = lkv
            acc_k = acc_k + lkv
            meta[cuk0 + b + cutlass.Int32(1)] = acc_k


@cute.kernel
def build_thd_meta_kernel(
    meta_t: cute.Tensor,
    q_lens_t: cute.Tensor,
    kv_lens_t: cute.Tensor,
    lens_form: cutlass.Int32,
    n_batch: cutlass.Int32,
) -> None:
    """Meta-only THD setup (SM120: no per-batch O TMA descriptors — O stores
    are raw pointer writes predicated per row). One elected thread; the main
    kernel launched after it on the same stream sees the writes by kernel
    boundary ordering."""
    if nvvm.elect_sync():
        write_thd_meta(
            cutlass.make_array_view(meta_t),
            cutlass.make_array_view(q_lens_t),
            cutlass.make_array_view(kv_lens_t),
            lens_form,
            n_batch,
        )


@cute.kernel
def build_thd_meta_o_kv_descs_kernel(
    o_tensor: cute.Tensor,
    base_o_desc: cutlass.GridConstant[tmap.TensorMap],
    base_k_desc: cutlass.GridConstant[tmap.TensorMap],
    base_v_desc: cutlass.GridConstant[tmap.TensorMap],
    o_desc_words: cute.Tensor,
    meta_t: cute.Tensor,
    q_lens_t: cute.Tensor,
    kv_lens_t: cute.Tensor,
    lens_form: cutlass.Int32,
    n_qh: cutlass.Int32,
    n_batch: cutlass.Int32,
    o_row_stride: cutlass.Int32,
) -> None:
    """``build_thd_meta_o_descs_kernel`` + packed-total-clamped K/V descriptors
    (the FP8/MXFP8 THD flavors).

    The K/V loads tile in TILE_N rows, so the LAST sequence's tile steps past
    the packed KV total into the buffer's capacity tail — caller-owned bytes
    that may be NaN (test_mhas_v2 poisons them deliberately). Masked S columns
    are NaN-safe (the mask is a select), but BMM2's ``P·V`` is not
    (``0 · NaN == NaN`` wipes every valid row of the tile), and on cc10.3 the
    fused-LDTM row-max reduces S BEFORE the mask. So the setup thread also
    copies the K and V base descriptors into ``o_desc_words`` slots
    ``n_batch+1`` / ``n_batch+2`` with their seq extent (GLOBAL_DIM ord=2)
    patched to the packed total ``cu_k[B]`` — tile-tail loads past it become
    TMA-OOB and land as EXACT ZEROS in SMEM (zero V nulls the masked P·V
    terms; zero K keeps the pre-mask row-max finite). Slot ``n_batch`` stays
    the never-built dead-unit pad slot."""
    if nvvm.elect_sync():
        meta = cutlass.make_array_view(meta_t)
        write_thd_meta(meta, cutlass.make_array_view(q_lens_t), cutlass.make_array_view(kv_lens_t), lens_form, n_batch)
        cuq0 = n_batch
        o_ptr = o_tensor.iterator.raw_ptr()
        desc_base = o_desc_words.iterator.raw_ptr()
        src_words = Pointer(base_o_desc.get_ptr(), dtype=cutlass.Int64)
        row_elems = o_row_stride
        for b in cutlass.range(0, n_batch, 1, unroll=1):
            dptr = desc_base + b * cutlass.Int32(TENSOR_MAP_QWORDS)
            for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
                (dptr + i).store((src_words + i).load())
            cu_q_b = cutlass.Int32(meta[cuq0 + b])
            s_i = cutlass.Int32(meta[cuq0 + b + cutlass.Int32(1)]) - cu_q_b
            row_base = o_ptr + cu_q_b * row_elems
            nvvm.tensormap_replace(
                nvvm.TensormapField.GLOBAL_ADDRESS,
                dptr,
                new_value=row_base.toint(cutlass.Int64),
            )
            nvvm.tensormap_replace(
                nvvm.TensormapField.GLOBAL_DIM,
                dptr,
                new_value=s_i,
                ord=2,
            )
        t_kv = cutlass.Int32(meta[cutlass.Int32(3) * n_batch + cutlass.Int32(1)])  # cu_k[B]
        k_dptr = desc_base + (n_batch + cutlass.Int32(1)) * cutlass.Int32(TENSOR_MAP_QWORDS)
        k_src = Pointer(base_k_desc.get_ptr(), dtype=cutlass.Int64)
        for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
            (k_dptr + i).store((k_src + i).load())
        nvvm.tensormap_replace(nvvm.TensormapField.GLOBAL_DIM, k_dptr, new_value=t_kv, ord=2)
        v_dptr = desc_base + (n_batch + cutlass.Int32(2)) * cutlass.Int32(TENSOR_MAP_QWORDS)
        v_src = Pointer(base_v_desc.get_ptr(), dtype=cutlass.Int64)
        for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
            (v_dptr + i).store((v_src + i).load())
        nvvm.tensormap_replace(nvvm.TensormapField.GLOBAL_DIM, v_dptr, new_value=t_kv, ord=2)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU,
            from_proxy=nvvm.Proxy.GENERIC,
            to_proxy=nvvm.Proxy.TENSORMAP,
        )


@cute.kernel
def build_thd_meta_o_descs_kernel(
    o_tensor: cute.Tensor,
    base_o_desc: cutlass.GridConstant[tmap.TensorMap],
    o_desc_words: cute.Tensor,
    meta_t: cute.Tensor,
    q_lens_t: cute.Tensor,
    kv_lens_t: cute.Tensor,
    lens_form: cutlass.Int32,
    n_qh: cutlass.Int32,
    n_batch: cutlass.Int32,
    o_row_stride: cutlass.Int32,
) -> None:
    """Per-execute THD setup, one elected thread (issue #552, D2H removal):
    build the [seq_kv_lens(B) | cu_seqlens_q(B+1) | cu_seqlens_k(B+1)] metadata
    buffer DEVICE-side from the caller's length tensors — ``(B,)`` per-batch
    lengths (serial cumsum; B is small) or the ``(B+1,)`` cu prefix-sum form
    (NORMALIZED by subtracting element 0 — the packed buffers are addressed
    from token 0, so a cu tensor sliced from a larger prefix means the same
    lengths, and the host can no longer validate ``cu[0] == 0`` (Rule 3), so
    an unnormalized base must not leak into the offsets the tiles and the
    dead-unit sentinel read; per-batch KV lengths are adjacent diffs either
    way), per side via
    ``lens_form`` (bit 0: Q is cu, bit 1: KV is cu) — then build the per-batch
    O TMA descriptors from the cu_q values just written (same thread, program
    order). Replaces the host tolist → cumsum → H2D round-trip with work
    inside the setup launch that already existed for the descriptors."""
    if nvvm.elect_sync():
        meta = cutlass.make_array_view(meta_t)
        write_thd_meta(meta, cutlass.make_array_view(q_lens_t), cutlass.make_array_view(kv_lens_t), lens_form, n_batch)
        cuq0 = n_batch
        # Per-batch O descriptors, from the cu_q values written above (same
        # thread — plain program order, no fence needed for the meta reads).
        o_ptr = o_tensor.iterator.raw_ptr()
        desc_base = o_desc_words.iterator.raw_ptr()
        src_words = Pointer(base_o_desc.get_ptr(), dtype=cutlass.Int64)
        row_elems = o_row_stride
        for b in cutlass.range(0, n_batch, 1, unroll=1):
            dptr = desc_base + b * cutlass.Int32(TENSOR_MAP_QWORDS)
            for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
                (dptr + i).store((src_words + i).load())
            cu_q_b = cutlass.Int32(meta[cuq0 + b])
            s_i = cutlass.Int32(meta[cuq0 + b + cutlass.Int32(1)]) - cu_q_b
            row_base = o_ptr + cu_q_b * row_elems
            nvvm.tensormap_replace(
                nvvm.TensormapField.GLOBAL_ADDRESS,
                dptr,
                new_value=row_base.toint(cutlass.Int64),
            )
            nvvm.tensormap_replace(
                nvvm.TensormapField.GLOBAL_DIM,
                dptr,
                new_value=s_i,
                ord=2,
            )
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU,
            from_proxy=nvvm.Proxy.GENERIC,
            to_proxy=nvvm.Proxy.TENSORMAP,
        )
