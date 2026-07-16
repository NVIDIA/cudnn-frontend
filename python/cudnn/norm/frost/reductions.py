"""Block-level reduction helpers shared by all FROST norm kernels.

The norm kernels use a one-CTA-per-reduction-group layout: every thread block
owns one normalization group and cooperatively reduces the group's elements.
These helpers perform an intra-CTA sum reduction (warp-shuffle within each warp,
then a cross-warp combine through a small shared-memory scratch).

``BLOCK_THREADS`` must be a multiple of the 32-lane warp size. The helpers take
the compile-time thread count as a ``cutlass.Constexpr`` so the warp count is
known at trace time.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute


@cute.jit
def block_reduce_sum(
    val: cutlass.Float32,
    tid: cutlass.Int32,
    block_threads: cutlass.Constexpr,
) -> cutlass.Float32:
    """Reduce ``val`` across all threads in the CTA; result broadcast to all."""
    nwarps = block_threads // 32
    smem = cute.arch.alloc_smem(cutlass.Float32, nwarps, 4)
    warp = tid // 32
    lane = tid % 32
    v = cute.arch.warp_reduction_sum(val)
    if lane == 0:
        smem[warp] = v
    cute.arch.sync_threads()
    if tid == 0:
        acc = cutlass.Float32(0.0)
        for w in range(nwarps):
            acc = acc + smem[w]
        smem[0] = acc
    cute.arch.sync_threads()
    total = smem[0]
    cute.arch.sync_threads()
    return total


@cute.jit
def block_reduce_sum2(
    v1: cutlass.Float32,
    v2: cutlass.Float32,
    tid: cutlass.Int32,
    block_threads: cutlass.Constexpr,
) -> cute.typing.Tuple:
    """Reduce two independent values across the CTA in a single barrier pair.

    Returns ``(sum(v1), sum(v2))``, both broadcast to every thread.
    """
    nwarps = block_threads // 32
    smem = cute.arch.alloc_smem(cutlass.Float32, 2 * nwarps, 4)
    warp = tid // 32
    lane = tid % 32
    r1 = cute.arch.warp_reduction_sum(v1)
    r2 = cute.arch.warp_reduction_sum(v2)
    if lane == 0:
        smem[warp] = r1
        smem[nwarps + warp] = r2
    cute.arch.sync_threads()
    if tid == 0:
        a1 = cutlass.Float32(0.0)
        a2 = cutlass.Float32(0.0)
        for w in range(nwarps):
            a1 = a1 + smem[w]
            a2 = a2 + smem[nwarps + w]
        smem[0] = a1
        smem[nwarps] = a2
    cute.arch.sync_threads()
    s1 = smem[0]
    s2 = smem[nwarps]
    cute.arch.sync_threads()
    return s1, s2
