# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared THD / varlen (packed ``[T,H,D]`` + ``cu_seqlens``) device helpers.

* :data:`TENSOR_MAP_QWORDS` — int64 words per 128-byte TMA descriptor.
* :func:`build_qkv_load_descs_kernel` — the separate device kernel that
  builds a per-(batch x head) TMA-descriptor array in GMEM for varlen
  loads/stores over a packed ``[T,H,D]`` tensor.
* :func:`build_h_descs_kernel` — its per-chunk-H sibling; derives the
  per-sequence H offsets from the token ``cu_seqlens`` in place of a
  caller-computed ``cu_h`` prefix array.
* :func:`build_state_descs_kernel` — per-(batch x head) descriptors over a
  DENSE ``[N, HO, K, V]`` state tensor (one entry per slot).
* :func:`downcast_state_kernel` — elementwise fp32 -> io copy of the
  initial state into the buffer the state descriptors read.
"""

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
import cutlass.experimental.cuda.tensor_map as _tma
from cutlass.base_dsl.typing import Pointer
import cuda.bindings.driver as _cuda_driver  # noqa: F401  (cute.compile pulls cuda)

TENSOR_MAP_QWORDS = 128 // 8


@cute.kernel
def build_qkv_load_descs_kernel(
    base_desc: cutlass.GridConstant[_tma.TensorMap],
    desc_words: cute.Tensor,
    cu_seqlens: cute.Tensor,
    base_ptr: cute.Tensor,
    n_batch: cutlass.Int32,
    n_heads: cutlass.Int32,
    head_group: cutlass.Int32,
    head_stride: cutlass.Int32,
    row_stride: cutlass.Int32,
    seq_ord: cutlass.Constexpr[int],
) -> None:
    """Build a per-(batch x head) TMA-descriptor ARRAY for a VARLEN (THD)
    Q/K/V/O tile over a packed ``[T,H,D]`` tensor + ``cu_seqlens``.  It
    re-points ``GLOBAL_ADDRESS`` per head so the flat output-head index
    ``head_idx`` maps to its KV head (``head_idx // head_group``),
    reproducing the GQA nested ``(h_r, h_v)`` head mode with stride-0
    replication.  The cute-side ``tma_tensor[None, None, head_idx]`` head
    indexing collapses to:

      * **identity (Q/O):** address head offset = ``head_idx * head_stride``
        (``head_group == 1``; ``head_idx // 1 == head_idx``).
      * **grouped (K/V):** address head offset =
        ``(head_idx // head_group) * head_stride`` — the stride-0 ``h_r``
        sub-mode means ``head_group`` consecutive Q heads share one KV head.

    Pass ``head_group == 1`` for identity and ``head_group == h_q // h_v``
    for K/V.  The descriptor array is laid out ``[batch][head]`` (head-minor):
    slot ``(b * n_heads + h)``.

    GLOBAL_ADDRESS folds BOTH the per-sequence token start
    (``cu_seqlens[b] * row_stride``) AND the per-head offset
    (``(head_idx // head_group) * head_stride``); GLOBAL_DIM[``seq_ord``] is
    capped to the per-sequence token COUNT (``cu_seqlens[b+1] - cu_seqlens[b]``)
    so a load box past the sequence end is OOB-clipped.  Because the
    GLOBAL_ADDRESS already points at the sequence's first token, the consumer
    issues the TMA with a token coordinate of **0** (not the absolute packed
    offset).

    Released via the GENERIC->TENSORMAP proxy fence; the consumer
    (``tma_load_tile`` with ``gmem_slice.desc_ptr`` set) acquire-fences each
    slot before the load.  ``seq_ord`` is a compile-time ord
    (``tensormap_replace`` ord must be a Python int)."""
    if nvvm.elect_sync():
        desc_base = desc_words.iterator.raw_ptr()
        src_words = Pointer(base_desc.get_ptr(), dtype=cutlass.Int64)
        cu = cutlass.make_array_view(cu_seqlens)
        base = base_ptr.iterator.raw_ptr()
        for b in cutlass.range(0, n_batch, 1, unroll=1):
            cu_b = cutlass.Int32(cu[b])
            s_b = cutlass.Int32(cu[b + cutlass.Int32(1)]) - cu_b
            # Int64: the H descs cross 2^31 elements near cu[b] ~ 2k
            # (row_stride = HO*DK*DV); inputs follow at larger B*T.
            tok0 = cutlass.Int64(cu_b) * cutlass.Int64(row_stride)
            for h in cutlass.range(0, n_heads, 1, unroll=1):
                slot = b * n_heads + h
                dptr = desc_base + slot * cutlass.Int32(TENSOR_MAP_QWORDS)
                for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
                    (dptr + i).store((src_words + i).load())
                kv_h = h // head_group
                head_off = cutlass.Int64(kv_h) * cutlass.Int64(head_stride)
                addr = base + (tok0 + head_off)
                nvvm.tensormap_replace(
                    nvvm.TensormapField.GLOBAL_ADDRESS,
                    dptr,
                    new_value=addr.toint(cutlass.Int64),
                )
                nvvm.tensormap_replace(
                    nvvm.TensormapField.GLOBAL_DIM,
                    dptr,
                    new_value=s_b,
                    ord=seq_ord,
                )
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU,
            from_proxy=nvvm.Proxy.GENERIC,
            to_proxy=nvvm.Proxy.TENSORMAP,
        )


@cute.kernel
def downcast_state_kernel(
    mS0: cute.Tensor,
    mOut: cute.Tensor,
    n: cutlass.Int32,
) -> None:
    """Flat elementwise copy of the fp32 initial state into the io-dtype
    buffer the backward's per-(b,h) state descriptors read (1-D views, one
    element per thread)."""
    idx = cutlass.Int32(cute.arch.block_idx()[0]) * cutlass.Int32(128) + cutlass.Int32(cute.arch.thread_idx()[0])
    if idx < n:
        mOut[idx] = mS0[idx].to(mOut.element_type)


@cute.kernel
def build_state_descs_kernel(
    base_desc: cutlass.GridConstant[_tma.TensorMap],
    desc_words: cute.Tensor,
    base_ptr: cute.Tensor,
    n_batch: cutlass.Int32,
    n_heads: cutlass.Int32,
    ent_stride: cutlass.Int32,
) -> None:
    """Per-(batch x head) TMA-descriptor array over a DENSE state tensor
    ``[N, HO, K, V]``: slot ``(b * n_heads + h)`` gets GLOBAL_ADDRESS
    pointing at its ``[K, V]`` tile (``slot * ent_stride`` elements in).
    All dims come baked from the base descriptor (one entry per slot), so
    only the address is patched."""
    if nvvm.elect_sync():
        desc_base = desc_words.iterator.raw_ptr()
        src_words = Pointer(base_desc.get_ptr(), dtype=cutlass.Int64)
        base = base_ptr.iterator.raw_ptr()
        n_slots = n_batch * n_heads
        for s in cutlass.range(0, n_slots, 1, unroll=1):
            dptr = desc_base + s * cutlass.Int32(TENSOR_MAP_QWORDS)
            for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
                (dptr + i).store((src_words + i).load())
            addr = base + cutlass.Int64(s) * cutlass.Int64(ent_stride)
            nvvm.tensormap_replace(
                nvvm.TensormapField.GLOBAL_ADDRESS,
                dptr,
                new_value=addr.toint(cutlass.Int64),
            )
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU,
            from_proxy=nvvm.Proxy.GENERIC,
            to_proxy=nvvm.Proxy.TENSORMAP,
        )


@cute.kernel
def build_h_descs_kernel(
    base_desc: cutlass.GridConstant[_tma.TensorMap],
    desc_words: cute.Tensor,
    cu_seqlens: cute.Tensor,
    base_ptr: cute.Tensor,
    n_batch: cutlass.Int32,
    n_heads: cutlass.Int32,
    head_stride: cutlass.Int32,
    row_stride: cutlass.Int32,
    every_n: cutlass.Int32,
    seq_ord: cutlass.Constexpr[int],
) -> None:
    """Per-(batch x head) TMA-descriptor array for the per-chunk H tensor.

    Unlike :func:`build_qkv_load_descs_kernel` this derives the per-sequence
    H offsets from the TOKEN ``cu_seqlens`` on the fly instead of taking a
    caller-computed ``cu_h`` prefix array: ``count_b = (seqlen_b - 1) //
    every_n`` (0 for empty sequences), running-prefix-summed.
    GLOBAL_DIM[``seq_ord``] is capped to ``count_b`` exactly as the
    cu_h-differences cap did.  H heads are identity-mapped (no GQA
    grouping)."""
    if nvvm.elect_sync():
        desc_base = desc_words.iterator.raw_ptr()
        src_words = Pointer(base_desc.get_ptr(), dtype=cutlass.Int64)
        cu = cutlass.make_array_view(cu_seqlens)
        base = base_ptr.iterator.raw_ptr()
        run = cutlass.Int32(0)
        for b in cutlass.range(0, n_batch, 1, unroll=1):
            s_tok = cutlass.Int32(cu[b + cutlass.Int32(1)]) - cutlass.Int32(cu[b])
            cnt = (s_tok - cutlass.Int32(1)) // every_n
            cnt = cnt if s_tok > 0 else cutlass.Int32(0)
            h0 = run
            run = run + cnt
            ent0 = cutlass.Int64(h0) * cutlass.Int64(row_stride)
            for h in cutlass.range(0, n_heads, 1, unroll=1):
                slot = b * n_heads + h
                dptr = desc_base + slot * cutlass.Int32(TENSOR_MAP_QWORDS)
                for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
                    (dptr + i).store((src_words + i).load())
                head_off = cutlass.Int64(h) * cutlass.Int64(head_stride)
                addr = base + (ent0 + head_off)
                nvvm.tensormap_replace(
                    nvvm.TensormapField.GLOBAL_ADDRESS,
                    dptr,
                    new_value=addr.toint(cutlass.Int64),
                )
                nvvm.tensormap_replace(
                    nvvm.TensormapField.GLOBAL_DIM,
                    dptr,
                    new_value=cnt,
                    ord=seq_ord,
                )
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU,
            from_proxy=nvvm.Proxy.GENERIC,
            to_proxy=nvvm.Proxy.TENSORMAP,
        )
