# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""THD / varlen setup launch for the SM100 SDPA backward chain.

One launch per execute, ahead of stages 1-3, building the facts only the DEVICE
can know: the packed totals and per-sequence lengths live in ``cu_seqlens``, and
reading them on the host would be a D2H sync per iteration (issue #552).

* the shared THD metadata buffer (:mod:`cudnn.frost.tile_dsl.thd`), extended
  with the ``row_off(B+1)`` block offsets of the blocked S/dS workspace;
* the persistent scheduler's live-unit total and claim counter.

**Descriptors are deliberately NOT built here.**  A TMA descriptor's box dims,
swizzle and dim ORDER are the consuming kernel's private geometry -- stage 2
addresses ``(d, head, seq, batch)`` so its sequence axis is ``ord=2``, while
stage 3's C operand is ``(n, m, h, b)`` and its sequence axis is ``ord=1`` -- so
a shared builder would either take those from a caller with no business knowing
them, or hardcode one stage's answer and silently mis-patch the other.  Each
stage patches its own descriptors from its own ``_host`` with
``tile_dsl.thd.emit_seq_descs`` / ``emit_clamped_desc``; this launch publishes
the metadata they all read.
"""

from cutlass.experimental import primitives as nvvm

from typing import Optional

import cutlass
import cutlass.cute as cute

from cudnn.frost.tile_dsl.thd import (
    THD_BWD_META_WORDS,
    THD_ROWOFF_OFF,
    THD_SETUP_THREADS,
    write_thd_batch_remap,
    write_thd_live_and_ctr,
    write_thd_meta,
    write_thd_port_origins,
    write_thd_row_offsets,
)

__all__ = [
    "THD_BWD_META_WORDS",
    "THD_ROWOFF_OFF",
    "THD_SETUP_THREADS",
    "build_thd_bwd_setup_kernel",
    "thd_bwd_setup_host",
    "build_thd_meta_kernel",
    "thd_meta_host",
    "ORIGIN_PORTS",
    "build_thd_meta_origins_kernel",
    "thd_meta_origins_host",
]


@cute.kernel
def build_thd_bwd_setup_kernel(
    meta_t: cute.Tensor,
    q_lens_t: cute.Tensor,
    kv_lens_t: cute.Tensor,
    lens_form: cutlass.Int32,
    n_qh: cutlass.Int32,
    n_batch: cutlass.Int32,
    ws_gran: cutlass.Int32,
    cga_tile_m: cutlass.Int32,
    n_clusters: cutlass.Int32,
) -> None:
    """Metadata, blocked-workspace row offsets, live-unit total, claim counter.

    ``ws_gran`` is the S/dS block granularity (stage 2's per-CTA store box);
    ``cga_tile_m`` is stage 2's unit height.  They are the same number today but
    are passed separately so a tile change cannot silently redefine the
    workspace layout.
    """
    tidx, _, _ = cute.arch.thread_idx()
    nthreads, _, _ = cute.arch.block_dim()
    # Warp 0's leader only: elect_sync elects one thread PER WARP, and this
    # block is THD_SETUP_THREADS wide for the parallel ranking below.
    if nvvm.elect_sync() and tidx < cutlass.Int32(32):
        meta = cutlass.make_array_view(meta_t)
        write_thd_meta(meta, cutlass.make_array_view(q_lens_t), cutlass.make_array_view(kv_lens_t), lens_form, n_batch)
        # Same thread, so plain program order carries the cu_seqlens it just
        # wrote into the block-offset prefix sum.
        write_thd_row_offsets(meta, n_batch, ws_gran)
    # Outside the elect: every thread helps rank the batches.  The barrier makes
    # the cu_seqlens_q written above visible to the whole block first.  Stage 2's
    # decode walks this permutation, so skipping it leaves the region
    # uninitialized and units decode garbage batches.
    cute.arch.barrier()
    write_thd_batch_remap(cutlass.make_array_view(meta_t), n_batch, cutlass.Int32(tidx), cutlass.Int32(nthreads))
    cute.arch.barrier()
    write_thd_live_and_ctr(cutlass.make_array_view(meta_t), n_batch, n_qh, cga_tile_m, n_clusters, cutlass.Int32(tidx))


build_thd_bwd_setup_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def thd_bwd_setup_host(meta_t, q_lens_t, kv_lens_t, lens_form, n_qh, n_batch, ws_gran, cga_tile_m, n_clusters, stream=None):
    """One-block launch of the metadata builder.

    Lives here rather than in the adapter because a `@cute.jit` defined inside a
    method closes over the kernel and its block width, and the DSL requires a
    code object with no free variables.
    """
    build_thd_bwd_setup_kernel(meta_t, q_lens_t, kv_lens_t, lens_form, n_qh, n_batch, ws_gran, cga_tile_m, n_clusters).launch(
        grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream
    )


@cute.kernel
def build_thd_meta_kernel(
    meta_t: cute.Tensor,
    q_lens_t: cute.Tensor,
    kv_lens_t: cute.Tensor,
    lens_form: cutlass.Int32,
    n_batch: cutlass.Int32,
) -> None:
    """Lengths -> ``[seq_kv_lens(B) | cu_seqlens_q(B+1) | cu_seqlens_k(B+1)]`` only.

    The metadata a kernel that takes ``cu_seqlens`` on device needs and
    nothing else: no blocked-workspace row offsets, no batch ranking, no claim
    counter.  One thread does the serial cumsum (B is small); no warp
    primitives, so it runs on every architecture the SDPA kernels do -- the
    SM80 backward reads ``cu_q`` / ``cu_k`` straight out of this buffer.
    """
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == cutlass.Int32(0):
        write_thd_meta(cutlass.make_array_view(meta_t), cutlass.make_array_view(q_lens_t), cutlass.make_array_view(kv_lens_t), lens_form, n_batch)


build_thd_meta_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def thd_meta_host(meta_t, q_lens_t, kv_lens_t, lens_form, n_batch, stream=None):
    """One-warp launch of :func:`build_thd_meta_kernel` (see ``thd_bwd_setup_host``
    for why the launch wrapper lives at module level)."""
    build_thd_meta_kernel(meta_t, q_lens_t, kv_lens_t, lens_form, n_batch).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


# Ports whose caller buffers a backward may address at bound ragged offsets, in
# the order the origin spec and the offset pointers use.
ORIGIN_PORTS = ("q", "k", "v", "o", "do", "dq", "dk", "dv", "stats")


@cute.kernel
def build_thd_meta_origins_kernel(
    meta_t: cute.Tensor,
    q_lens_t: cute.Tensor,
    kv_lens_t: cute.Tensor,
    lens_form: cutlass.Int32,
    n_batch: cutlass.Int32,
    org_t: cute.Tensor,
    ro_q: Optional[cute.Tensor],
    ro_k: Optional[cute.Tensor],
    ro_v: Optional[cute.Tensor],
    ro_o: Optional[cute.Tensor],
    ro_do: Optional[cute.Tensor],
    ro_dq: Optional[cute.Tensor],
    ro_dk: Optional[cute.Tensor],
    ro_dv: Optional[cute.Tensor],
    ro_stats: Optional[cute.Tensor],
    origins: cutlass.Constexpr,
) -> None:
    """:func:`build_thd_meta_kernel` plus the caller-buffer token origins of the
    ports ``origins`` materializes (one ``(row, mult, ts)`` or ``None`` per
    :data:`ORIGIN_PORTS` entry).  Thread 0 writes the compact metadata; every
    thread writes origins, which do not depend on it."""
    tidx, _, _ = cute.arch.thread_idx()
    nthreads, _, _ = cute.arch.block_dim()
    if tidx == cutlass.Int32(0):
        write_thd_meta(cutlass.make_array_view(meta_t), cutlass.make_array_view(q_lens_t), cutlass.make_array_view(kv_lens_t), lens_form, n_batch)
    ro = (ro_q, ro_k, ro_v, ro_o, ro_do, ro_dq, ro_dk, ro_dv, ro_stats)
    for i in cutlass.range_constexpr(len(ORIGIN_PORTS)):
        if cutlass.const_expr(origins[i] is not None):
            write_thd_port_origins(org_t, ro[i], origins[i][0], origins[i][1], origins[i][2], n_batch, cutlass.Int32(tidx), cutlass.Int32(nthreads))


build_thd_meta_origins_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def thd_meta_origins_host(meta_t, q_lens_t, kv_lens_t, lens_form, n_batch, org_t, ro, origins: cutlass.Constexpr, stream=None):
    """One-block launch of :func:`build_thd_meta_origins_kernel`; ``ro`` is the
    nine offset tensors (``None`` for a port without origins)."""
    build_thd_meta_origins_kernel(meta_t, q_lens_t, kv_lens_t, lens_form, n_batch, org_t, *ro, origins).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)
