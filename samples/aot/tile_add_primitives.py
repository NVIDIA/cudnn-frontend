"""The same TMA tile add, one level down: cutlass.experimental.primitives.

``cutedsl_tma_add_engine`` writes this kernel with ``cpasync.make_tiled_tma_atom``
and ``cute.copy``, which wrap the descriptor and the copy instruction in an
"atom" and hide where the descriptor lives. This file is the explicit spelling
of the identical kernel: build the TMA descriptor yourself, issue the PTX
bulk-tensor copy yourself, drive the mbarrier yourself.

It is here because the two are worth reading side by side. The atom version is
what you want to write; this one is what it lowers to, and it is the level FROST
works at, so it is the level you end up debugging at.

Same arithmetic, same tile, same result:

    c = a + b     fp32, 2-D, 128x64 tiles, operands loaded by TMA

The canonical form for these primitives is the DKG tutorial at
``examples/CuTeDSL/experimental/primitives/tutorial/04_tma_load.py``. Four things
in it are load-bearing and none of them fail at compile time:

* the descriptor parameter must be ``cutlass.GridConstant[cuda.TensorMap]``. A
  TMA descriptor read from kernel parameter space has to be grid-constant; pass
  a bare ``TensorMap`` and the copy faults with a misaligned address, nowhere
  near the actual mistake.
* ``elect_sync()`` elects one lane PER WARP, so a block wider than a warp needs
  its own warp guard on top. Without it every warp issues the copies and
  quadruples the transaction count.
* ``mbarrier_arrive_expect_tx`` must precede the copies, and its byte count
  covers every copy that signals the same barrier -- hence the sum of both
  descriptors' ``global_tx_bytes()``.
* the wait spins on the timelimit variant, which retries on a tick-timeout and
  carries ``.acquire.cta`` ordering, so the TMA writes are visible once it
  returns.

    python tile_add_primitives.py
"""

import cuda.bindings.driver as drv
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.experimental import primitives as prims

BM, BN = 128, 64
THREADS = 128


@cute.kernel
def tile_add_kernel(
    desc_a: cutlass.GridConstant[cuda.TensorMap],
    desc_b: cutlass.GridConstant[cuda.TensorMap],
    mC: cute.Tensor,
):
    bidx, bidy, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    sA = cutlass.Array(cutlass.Float32, (BM, BN), space=cutlass.AddressSpace.smem)
    sB = cutlass.Array(cutlass.Float32, (BM, BN), space=cutlass.AddressSpace.smem)
    mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)

    # One warp, then one lane within it. elect_sync() alone would leave one
    # lane per warp, i.e. four issuers in a 128-thread block.
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp == 0:
        if prims.elect_sync():
            prims.prefetch_tensormap(desc_a.get_ptr())
            prims.prefetch_tensormap(desc_b.get_ptr())
            prims.mbarrier_init(mbar, 1)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)

    if warp == 0:
        if prims.elect_sync():
            # Both copies signal this barrier, so the promised byte count is
            # the sum. Promised BEFORE the copies, or TMA can decrement a
            # counter that is not set yet.
            prims.mbarrier_arrive_expect_tx(
                mbar, desc_a.global_tx_bytes() + desc_b.global_tx_bytes()
            )
            # Coordinates are element indices in the DESCRIPTOR's dimension
            # order -- innermost (N) first, the reverse of box_dims below.
            prims.cp_async_bulk_tensor_shared_cta_global(
                sA, desc_a.get_ptr(), (bidy * BN, bidx * BM), mbar
            )
            prims.cp_async_bulk_tensor_shared_cta_global(
                sB, desc_b.get_ptr(), (bidy * BN, bidx * BM), mbar
            )

    while not prims.mbarrier_try_wait_parity(mbar, 0, time_limit=10_000_000):
        pass

    for e in cutlass.range_constexpr(BM * BN // THREADS):
        flat = e * THREADS + tidx
        r = flat // BN
        c = flat % BN
        mC[(bidx * BM + r, bidy * BN + c)] = sA[(r, c)] + sB[(r, c)]


@cute.jit
def tile_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, stream: drv.CUstream):
    # Host side, once per call, against the pointers that actually arrived.
    # box_dims is in the TENSOR's mode order; swizzle none means the tile lands
    # linearly in shared memory, which is what the plain 2-D Array above is.
    descs = [
        cuda.create_tensor_map_tiled_from_view(
            t, box_dims=(BM, BN), stride_order=(1, 0), swizzle=cuda.TensorMapSwizzle.none
        )
        for t in (mA, mB)
    ]
    m, n = mC.shape
    tile_add_kernel(descs[0], descs[1], mC).launch(
        grid=(m // BM, n // BN, 1), block=(THREADS, 1, 1), stream=stream
    )


def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")

    M, N = 1024, 512
    a = torch.randn(M, N, device="cuda", dtype=torch.float32)
    b = torch.randn(M, N, device="cuda", dtype=torch.float32)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    stream = drv.CUstream(torch.cuda.current_stream().cuda_stream)

    compiled = cute.compile(
        tile_add,
        *[from_dlpack(t, assumed_align=16) for t in (a, b, c)],
        stream,
        options="--enable-tvm-ffi",
    )
    compiled(a, b, c, stream)
    torch.cuda.synchronize()

    err = (c - (a + b)).abs().max().item()
    print(f"tile_add via primitives  {M}x{N}  max |err| = {err}")
    return 0 if err == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
