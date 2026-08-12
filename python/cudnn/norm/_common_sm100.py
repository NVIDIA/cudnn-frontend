"""Shared device-side building blocks for the sm_100 norm forward kernels.

These are the CUTLASS-primitive equivalents of the old plain-cute-dsl helpers:

- :func:`stage_row` stages a whole global row into shared memory once with
  ``cp.async`` (``cudnn.frost.tile_dsl.tma.load_tile`` -> ``nvvm.cp_async_shared_global``),
  so the two-pass norm reads X once instead of twice.
- :func:`block_reduce_sum2` reduces two per-thread partials across the CTA
  (warp-shuffle + a small ``SmemAllocator`` scratch), broadcasting the result.

Every flavor's forward kernel (layernorm / rmsnorm / groupnorm / instancenorm /
batchnorm) is built from these. ``block_threads`` is a ``Constexpr`` so the warp
count is known at trace time.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.primitives as nvvm  # noqa: F401 — assert the primitive layer is present

from cudnn.frost.tile_dsl.tma import load_tile, cp_async_commit, cp_async_wait

# Staging modes (kernel Constexpr). NONE = read X from global; CPASYNC = per-thread
# cp.async into smem; BULK = single cp.async.bulk (TMA-family, mbarrier-completed).
STAGE_NONE = 0
STAGE_CPASYNC = 1
STAGE_BULK = 2


@cute.jit
def stage_row(sX, gX_row, M: cutlass.Constexpr, tid, V: cutlass.Constexpr,
              block_threads: cutlass.Constexpr, elem_bytes: cutlass.Constexpr) -> None:
    """Per-thread cp.async-stage one ``M``-element global row into ``sX``.

    Requires ``M % (block_threads * V) == 0``.
    """
    load_tile(
        sX.iterator, gX_row.iterator, M, tid,
        num_threads=block_threads, elems_per_copy=V, elem_bytes=elem_bytes,
    )
    cp_async_commit()
    cp_async_wait(0)
    cute.arch.sync_threads()


@cute.jit
def stage_row_bulk(sX, mbar, gX_row, nbytes: cutlass.Constexpr, tid) -> None:
    """TMA bulk-async stage: one ``cp.async.bulk`` copies the whole row into ``sX``.

    One thread issues the transfer and the CTA waits on ``mbar`` (an ``Int64``
    smem mbarrier). ``nbytes = M * elem_bytes`` must be a multiple of 16.
    """
    if tid == 0:
        cute.arch.mbarrier_init(mbar.iterator, 1)
    cute.arch.sync_threads()
    if tid == 0:
        cute.arch.mbarrier_arrive_and_expect_tx(mbar.iterator, nbytes)
        nvvm.cp_async_bulk_shared_cluster_global(sX.iterator, gX_row.iterator, mbar.iterator, nbytes)
    while not cute.arch.mbarrier_try_wait(mbar.iterator, 0):
        pass
    cute.arch.sync_threads()


@cute.jit
def stage_two_bulk(sX, sDY, mbar, gX_row, gDY_row, nbytes: cutlass.Constexpr, tid) -> None:
    """TMA bulk-async stage of two rows (X and DY) into ``sX``/``sDY`` on one
    mbarrier (expect 2x bytes, two bulk copies, single wait)."""
    if tid == 0:
        cute.arch.mbarrier_init(mbar.iterator, 1)
    cute.arch.sync_threads()
    if tid == 0:
        cute.arch.mbarrier_arrive_and_expect_tx(mbar.iterator, 2 * nbytes)
        nvvm.cp_async_bulk_shared_cluster_global(sX.iterator, gX_row.iterator, mbar.iterator, nbytes)
        nvvm.cp_async_bulk_shared_cluster_global(sDY.iterator, gDY_row.iterator, mbar.iterator, nbytes)
    while not cute.arch.mbarrier_try_wait(mbar.iterator, 0):
        pass
    cute.arch.sync_threads()


@cute.jit
def block_reduce_sum2(v1, v2, tid, red, block_threads: cutlass.Constexpr):
    """Reduce two independent partials across the CTA; both broadcast to all.

    ``red`` is an fp32 smem scratch tensor of length ``2 * (block_threads // 32)``.
    Returns ``(sum(v1), sum(v2))``.
    """
    nwarps: cutlass.Constexpr = block_threads // 32
    warp = tid // 32
    lane = tid % 32
    r1 = cute.arch.warp_reduction_sum(v1)
    r2 = cute.arch.warp_reduction_sum(v2)
    if lane == 0:
        red[warp] = r1
        red[nwarps + warp] = r2
    cute.arch.sync_threads()
    if tid == 0:
        a1 = cutlass.Float32(0.0)
        a2 = cutlass.Float32(0.0)
        for w in range(nwarps):
            a1 = a1 + red[w]
            a2 = a2 + red[nwarps + w]
        red[0] = a1
        red[nwarps] = a2
    cute.arch.sync_threads()
    s1 = red[0]
    s2 = red[nwarps]
    cute.arch.sync_threads()
    return s1, s2


def red_scratch_len(block_threads: int) -> int:
    """Length of the fp32 reduction scratch for :func:`block_reduce_sum2`."""
    return 2 * (block_threads // 32)
