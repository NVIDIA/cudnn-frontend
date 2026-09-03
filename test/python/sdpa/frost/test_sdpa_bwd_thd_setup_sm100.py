# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``build_thd_bwd_setup_kernel``: the SM100 backward's THD setup launch.

Unit-level on purpose.  Everything this kernel produces is consumed by kernels
that do not exist yet, and both of its outputs fail SILENTLY downstream:

* the metadata words -- a reader offset and a writer offset that disagree by one
  make units decode garbage batches rather than raising;
* the per-sequence output descriptors -- their whole job is that stage 3's last
  M tile of a sequence, which OVERSHOOTS into the next sequence's rows with a
  live accumulator behind it, gets clipped by hardware.  ``compute-sanitizer``
  cannot see a TMA store that is in bounds for its descriptor but out of bounds
  for the sequence, so the descriptor extent IS the guard and a sentinel spike
  is the only thing that proves it (frost-gotchas.md, "Validation diagnostics").

Testing it here rather than through the chain means a failure names the setup
kernel instead of surfacing as wrong gradients three kernels later.
"""

from __future__ import annotations

import pytest
import torch

from frost_test_utils import _dsl_installed, requires_dsl, requires_pre_rubin_blackwell

pytestmark = [pytest.mark.L0, requires_pre_rubin_blackwell, requires_dsl]

# Sequence lengths chosen so every interesting case is present at once: not a
# multiple of the block granularity (70, 33, 5), exactly one block (128), a
# length spanning two blocks (130 kv), and lengths shorter than the probe's
# store box (33, 5) so the clip has something to clip.
_LENS_Q = (70, 33, 128, 5)
_LENS_KV = (64, 40, 130, 8)
_GRAN = 128  # the blocked workspace's row granularity: stage 2's per-CTA store box
_D = 64  # 64 fp32 = 256 B rows: the unswizzled TMA STG inner-box cap exactly
_PROBE_ROWS = 64  # store box; overshoots sequences 1 and 3
_SENTINEL = -7.0

# The DSL imports and the probe kernel sit at MODULE level, behind the same
# probe `requires_dsl` uses.  A `@cute.kernel` defined inside a test function
# resolves its names from the DEFINING MODULE's globals, not from the enclosing
# function's locals, so a nested one dies at trace time with
# `NameError: name 'cute' is not defined`.
if _dsl_installed():
    import cuda.bindings.driver as _cuda_driver
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cutlass.experimental import primitives as nvvm
    from cutlass.experimental.cuda import tensor_map as tmap

    from cudnn.frost.tile_dsl.handles import SmemTile, tma_slice_runtime_desc
    from cudnn.frost.tile_dsl.thd import THD_BWD_META_WORDS, THD_SETUP_THREADS
    from cudnn.frost.tile_dsl.tma import tma_store_commit, tma_store_tile, tma_store_wait
    from cudnn.sdpa.bwd.kernels.thd_helpers import DQ_SLOT_BASE, THD_BWD_DESC_SLOTS, build_thd_bwd_setup_kernel

    @cute.jit
    def _setup_host(dq_t, dk_t, dv_t, q_t, do_t, k_t, v_t, dw, meta_t, ql, kl, stream):
        box = (1, _PROBE_ROWS, 1, _D)
        order = (3, 2, 1, 0)  # innermost-first -> coords are (d, head, seq, batch)
        mk = lambda t: tmap.create_tensor_map_tiled_from_view(t, box_dims=box, stride_order=order, swizzle=tmap.TensorMapSwizzle.none)
        build_thd_bwd_setup_kernel(
            dq_t,
            dk_t,
            dv_t,
            mk(dq_t),
            mk(dk_t),
            mk(dv_t),
            mk(q_t),
            mk(do_t),
            mk(k_t),
            mk(v_t),
            dw,
            meta_t,
            ql,
            kl,
            cutlass.Int32(0),  # lens_form: both sides arrive as per-batch lengths
            cutlass.Int32(1),  # n_qh
            cutlass.Int32(len(_LENS_Q)),
            cutlass.Int32(_D),  # row strides, in ELEMENTS: H*D with H == 1
            cutlass.Int32(_D),
            cutlass.Int32(_D),
            cutlass.Int32(_GRAN),  # ws_gran
            cutlass.Int32(_GRAN),  # cga_tile_m
            cutlass.Int32(8),  # n_clusters
        ).launch(grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream)

    @cute.kernel
    def _probe_kernel(dw: cute.Tensor, slot_base: cutlass.Int32) -> None:
        """One block per sequence: store a full ``_PROBE_ROWS`` box through that
        sequence's dQ descriptor, marked with the sequence's own id."""
        bidx = cute.arch.block_idx()[0]
        tidx, _, _ = cute.arch.thread_idx()
        smem = cutlass.Array(cutlass.Float32, _PROBE_ROWS * _D, alignment=1024, space=cutlass.AddressSpace.smem)
        marker = cutlass.Float32(bidx + cutlass.Int32(1))
        for i in cutlass.range(tidx, cutlass.Int32(_PROBE_ROWS * _D), cutlass.Int32(128), unroll=1):
            smem.subview(i).store(marker)
        # The TMA reads SMEM the loop above wrote through the generic proxy.
        nvvm.fence_proxy("async.shared", space="cta")
        cute.arch.barrier()
        if tidx == cutlass.Int32(0):
            tile = SmemTile(
                base=smem,
                elems_per_stage=_PROBE_ROWS * _D,
                leading_byte_offset=0,
                stride_byte_offset=0,
                layout=0,
                tma_loads_per_tile=1,
                tma_granu_elems=_D,
                tma_subtile_stride_elems=0,
            )
            desc_ptr = (dw.iterator.raw_ptr() + (slot_base + bidx) * cutlass.Int32(16)).tospace(cutlass.AddressSpace.generic)
            # Row coord 0 of THIS sequence: the descriptor already carries the
            # sequence's base row, so the box spans [0, _PROBE_ROWS) of it.
            tma_store_tile(tile, tma_slice_runtime_desc(desc_ptr, cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0)))
            tma_store_commit()
            tma_store_wait(0)

    _probe_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    @cute.jit
    def _probe_host(dw, slot_base, stream):
        _probe_kernel(dw, slot_base).launch(grid=(len(_LENS_Q), 1, 1), block=(128, 1, 1), stream=stream)


def _build(dev):
    """Run the setup kernel over ``_LENS_*``; returns (meta list, desc_words, dQ)."""
    b = len(_LENS_Q)
    t_q, t_kv = sum(_LENS_Q), sum(_LENS_KV)
    # Deliberately OVER-allocated past the packed totals: that capacity tail is
    # exactly what the clamped input descriptors have to put out of reach, and
    # what the output descriptors must never be able to reach.
    dq = torch.full((1, t_q + 3 * _PROBE_ROWS, 1, _D), _SENTINEL, dtype=torch.float32, device=dev)
    dk = torch.full((1, t_kv + _PROBE_ROWS, 1, _D), _SENTINEL, dtype=torch.float32, device=dev)
    dv = torch.full_like(dk, _SENTINEL)
    q = torch.zeros((1, t_q + _PROBE_ROWS, 1, _D), dtype=torch.float32, device=dev)
    do = torch.zeros_like(q)
    k = torch.zeros((1, t_kv + _PROBE_ROWS, 1, _D), dtype=torch.float32, device=dev)
    v = torch.zeros_like(k)

    q_lens = torch.tensor(_LENS_Q, dtype=torch.int32, device=dev)
    kv_lens = torch.tensor(_LENS_KV, dtype=torch.int32, device=dev)
    meta = torch.zeros(THD_BWD_META_WORDS(b), dtype=torch.int32, device=dev)
    desc_words = torch.zeros(THD_BWD_DESC_SLOTS(b) * 16, dtype=torch.int64, device=dev)

    t = lambda x: from_dlpack(x, assumed_align=16)
    stream = _cuda_driver.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    args = (t(dq), t(dk), t(dv), t(q), t(do), t(k), t(v), t(desc_words), t(meta), t(q_lens), t(kv_lens), stream)
    cute.compile(_setup_host, *args)(*args)
    torch.cuda.synchronize()
    return meta.cpu().tolist(), desc_words, dq


def test_thd_bwd_setup_metadata():
    """Every metadata word, against a host-side reference."""
    from cudnn.frost.tile_dsl.thd import THD_CTR_OFF, THD_LIVE_OFF, THD_REMAP_OFF, THD_ROWOFF_OFF

    b = len(_LENS_Q)
    meta, _, _ = _build("cuda")

    cu_q = [0]
    for s in _LENS_Q:
        cu_q.append(cu_q[-1] + s)
    cu_k = [0]
    for s in _LENS_KV:
        cu_k.append(cu_k[-1] + s)

    assert meta[0:b] == list(_LENS_KV), "seq_kv_lens"
    assert meta[b : 2 * b + 1] == cu_q, "cu_seqlens_q"
    assert meta[2 * b + 1 : 3 * b + 2] == cu_k, "cu_seqlens_k"

    # batch_remap: descending Q length, ties on the lower original index.
    want = sorted(range(b), key=lambda i: (-_LENS_Q[i], i))
    assert meta[THD_REMAP_OFF(b) : THD_REMAP_OFF(b) + b] == want, "batch_remap"

    # One unit per (q-block, head); n_qh == 1 here.
    live = sum(-(-s // _GRAN) for s in _LENS_Q)
    assert meta[THD_LIVE_OFF(b)] == live, "live unit total"
    assert meta[THD_CTR_OFF(b)] == 8, "claim counter seeded at n_clusters"

    # row_off: the BLOCKED workspace prefix, each block padded up to _GRAN.
    row = [0]
    for s in _LENS_Q:
        row.append(row[-1] + -(-s // _GRAN) * _GRAN)
    assert meta[THD_ROWOFF_OFF(b) : THD_ROWOFF_OFF(b) + b + 1] == row, "row_off"


def test_thd_bwd_setup_output_descriptors_clip_at_the_sequence():
    """A store through sequence b's descriptor lands at cu_q[b] and STOPS.

    This is the property stage 3's epilogue will rest on: its last M tile of a
    sequence overshoots, and nothing but the descriptor's patched extent keeps
    that overshoot out of the next sequence's rows.
    """
    b = len(_LENS_Q)
    _, desc_words, dq = _build("cuda")

    t = lambda x: from_dlpack(x, assumed_align=16)
    stream = _cuda_driver.CUstream(torch.cuda.current_stream("cuda").cuda_stream)
    slot = cutlass.Int32(DQ_SLOT_BASE(b))
    cute.compile(_probe_host, t(desc_words), slot, stream)(t(desc_words), slot, stream)
    torch.cuda.synchronize()

    flat = dq[0, :, 0, :]
    cu = [0]
    for s in _LENS_Q:
        cu.append(cu[-1] + s)
    for i, s in enumerate(_LENS_Q):
        wrote = min(s, _PROBE_ROWS)
        got = flat[cu[i] : cu[i] + wrote]
        assert torch.all(got == float(i + 1)), f"seq {i}: rows [0,{wrote}) should carry marker {i + 1}"
        # Rows past the box but inside the sequence belong to nobody's store.
        tail = flat[cu[i] + wrote : cu[i + 1]]
        assert torch.all(tail == _SENTINEL), f"seq {i}: rows [{wrote},{s}) were written when they should not be"
    # And the over-allocated capacity tail past the packed total stays pristine.
    assert torch.all(flat[cu[-1] :] == _SENTINEL), "capacity tail was written"
