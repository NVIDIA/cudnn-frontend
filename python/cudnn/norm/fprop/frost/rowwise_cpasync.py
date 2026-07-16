"""cp.async smem-staged forward kernel for the per-sample norm family.

The two-pass norm (reduce, then normalize) normally reads X from global memory
twice. This kernel stages the whole row into shared memory **once** with
``cp.async`` (asynchronous global->shared, bypassing L1/registers), then both
passes read from smem -- cutting global X traffic from 2x to 1x.

Alignment: ``cp.async`` requires 128-bit-aligned copies. We recast the row to
``Int128`` elements (each inherently 16-byte aligned) for the copy, then recast
the smem buffer back to the element type for the math.

Constraints (enforced by :func:`cudnn.norm.frost.config.choose_cpasync_config`):
the row must fit in static shared memory (<=48 KB on sm_80), ``M % V == 0`` and
``(M // V) % block_threads == 0``. ``M`` is a compile-time constant (static smem).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
from cutlass.cute.runtime import from_dlpack

from cudnn.norm.frost.reductions import block_reduce_sum2


def _dyn(t):
    return from_dlpack(t, assumed_align=16).mark_layout_dynamic()


_CACHE = {}


@cute.kernel
def _fwd_cpasync_kernel(
    mX: cute.Tensor,
    mY: cute.Tensor,
    mGamma: cute.Tensor,
    mBeta: cute.Tensor,
    mMean: cute.Tensor,
    mRstd: cute.Tensor,
    gps: cutlass.Int32,
    cpg: cutlass.Int32,
    span: cutlass.Int32,
    eps: cutlass.Float32,
    M: cutlass.Constexpr,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
    V: cutlass.Constexpr,
    block_threads: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    r, _, _ = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()
    sRow = smem.allocate_tensor(mX.element_type, cute.make_layout(M), byte_alignment=16)

    # --- stage the row global -> smem via cp.async (128-bit aligned) ---
    sRow128 = cute.recast_tensor(sRow, cutlass.Int128)
    mX128 = cute.recast_tensor(mX, cutlass.Int128)
    row128 = mX128[r, None]
    atom = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.Int128, num_bits_per_copy=128)
    tiled = cute.make_tiled_copy_tv(atom, cute.make_layout(block_threads), cute.make_layout(1))
    thr = tiled.get_slice(tid)
    nv = M // V
    nchunk = nv // block_threads
    gchunks = cute.zipped_divide(row128, (block_threads,))
    schunks = cute.zipped_divide(sRow128, (block_threads,))
    ci = 0
    while ci < nchunk:
        cute.copy(atom, thr.partition_S(gchunks[None, ci]), thr.partition_D(schunks[None, ci]))
        ci = ci + 1
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()

    Mf = cutlass.Float32(M)

    # --- pass 1: reduce from smem ---
    s1 = cutlass.Float32(0.0)
    s2 = cutlass.Float32(0.0)
    j = tid
    while j < M:
        x = sRow[j].to(cutlass.Float32)
        s1 = s1 + x
        s2 = s2 + x * x
        j = j + block_threads
    s1, s2 = block_reduce_sum2(s1, s2, tid, block_threads)

    mean = cutlass.Float32(0.0)
    var = cutlass.Float32(0.0)
    if has_mean:
        mean = s1 / Mf
        var = s2 / Mf - mean * mean
    else:
        var = s2 / Mf
    rstd = cute.math.rsqrt(var + eps)
    if tid == 0:
        if has_mean:
            mMean[r] = mean
        mRstd[r] = rstd

    # --- pass 2: normalize from smem, vectorized store to Y ---
    base_c = (r % gps) * cpg
    row_y = mY[r, None]
    yv = cute.zipped_divide(row_y, (V,))
    vi = tid
    while vi < nv:
        j0 = vi * V
        yf = cute.make_fragment_like(yv[None, vi])
        for e in cutlass.range_constexpr(V):
            x = sRow[j0 + e].to(cutlass.Float32)
            xhat = (x - mean) * rstd
            c = base_c + (j0 + e) // span
            g = mGamma[c].to(cutlass.Float32)
            val = g * xhat
            if has_beta:
                val = val + mBeta[c].to(cutlass.Float32)
            yf[e] = val.to(mY.element_type)
        cute.autovec_copy(yf, yv[None, vi])
        vi = vi + block_threads


@cute.jit
def _fwd_cpasync_host(
    mX: cute.Tensor,
    mY: cute.Tensor,
    mGamma: cute.Tensor,
    mBeta: cute.Tensor,
    mMean: cute.Tensor,
    mRstd: cute.Tensor,
    R: cutlass.Int32,
    gps: cutlass.Int32,
    cpg: cutlass.Int32,
    span: cutlass.Int32,
    eps: cutlass.Float32,
    M: cutlass.Constexpr,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
    V: cutlass.Constexpr,
    block_threads: cutlass.Constexpr,
) -> None:
    _fwd_cpasync_kernel(
        mX, mY, mGamma, mBeta, mMean, mRstd,
        gps, cpg, span, eps, M, has_mean, has_beta, V, block_threads,
    ).launch(grid=(R, 1, 1), block=(block_threads, 1, 1))


def rowwise_forward_cpasync(spec, x, gamma, beta, *, eps, V, block_threads):
    """cp.async smem-staged per-sample forward. Returns ``(y, mean, rstd)``."""
    import torch

    from cudnn.norm.frost.dtypes import torch_dtype_to_str

    io_dtype = torch_dtype_to_str(x.dtype)
    has_beta = beta is not None
    if beta is None:
        beta = torch.zeros(spec.gamma_len, dtype=x.dtype, device=x.device)

    y = torch.empty_like(x)
    rstd = torch.empty(spec.R, dtype=torch.float32, device=x.device)
    mean = torch.empty(spec.R, dtype=torch.float32, device=x.device)

    runtime_args = (
        _dyn(x), _dyn(y), _dyn(gamma), _dyn(beta), _dyn(mean), _dyn(rstd),
        cutlass.Int32(spec.R),
        cutlass.Int32(spec.groups_per_sample), cutlass.Int32(spec.channels_per_group),
        cutlass.Int32(spec.gamma_inner_span), cutlass.Float32(eps),
    )
    ce_args = (spec.M, spec.has_mean, has_beta, V, block_threads)
    key = (io_dtype, spec.M, spec.has_mean, has_beta, V, block_threads)
    fn = _CACHE.get(key)
    if fn is None:
        fn = cute.compile(_fwd_cpasync_host, *runtime_args, *ce_args)
        _CACHE[key] = fn
    fn(*runtime_args)
    return y, mean, rstd
