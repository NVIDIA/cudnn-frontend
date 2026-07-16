"""Vectorized forward kernel for the per-sample norm family.

Same math as :mod:`cudnn.norm.fprop.frost.rowwise`, but each thread loads/stores
``V`` contiguous elements at a time through a register fragment
(``cute.autovec_copy`` emits a 128-bit ``ld.global.v4`` / ``st.global.v4`` when
the slice is contiguous and aligned). ``V`` is chosen so ``V * sizeof(elem)`` is
16 bytes (fp32 -> 4, fp16/bf16 -> 8). The big X/Y tensors are moved with vector
ops; the small per-channel gamma/beta are read scalar (they stay hot in L2).

Requires ``M % V == 0`` and 16-byte-aligned rows; the dispatcher falls back to
the scalar kernel otherwise.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from cudnn.norm.frost.reductions import block_reduce_sum2


def _dyn(t):
    return from_dlpack(t).mark_layout_dynamic()


_CACHE = {}


@cute.kernel
def _rowwise_fwd_vec_kernel(
    mX: cute.Tensor,
    mY: cute.Tensor,
    mGamma: cute.Tensor,
    mBeta: cute.Tensor,
    mMean: cute.Tensor,
    mRstd: cute.Tensor,
    M: cutlass.Int32,
    gps: cutlass.Int32,
    cpg: cutlass.Int32,
    span: cutlass.Int32,
    eps: cutlass.Float32,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
    V: cutlass.Constexpr,
    block_threads: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    r, _, _ = cute.arch.block_idx()

    Mf = cutlass.Float32(M)
    nv = M // V
    row_x = mX[r, None]
    xv = cute.zipped_divide(row_x, (V,))

    # --- pass 1: vectorized reduce of sum(x) and sum(x^2) ---
    s1 = cutlass.Float32(0.0)
    s2 = cutlass.Float32(0.0)
    vi = tid
    while vi < nv:
        gx = xv[None, vi]
        xf = cute.make_fragment_like(gx)
        cute.autovec_copy(gx, xf)
        for e in cutlass.range_constexpr(V):
            x = xf[e].to(cutlass.Float32)
            s1 = s1 + x
            s2 = s2 + x * x
        vi = vi + block_threads
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

    # --- pass 2: vectorized normalize + affine ---
    base_c = (r % gps) * cpg
    row_y = mY[r, None]
    yv = cute.zipped_divide(row_y, (V,))
    vi = tid
    while vi < nv:
        j0 = vi * V
        gx = xv[None, vi]
        gy = yv[None, vi]
        xf = cute.make_fragment_like(gx)
        yf = cute.make_fragment_like(gy)
        cute.autovec_copy(gx, xf)
        for e in cutlass.range_constexpr(V):
            x = xf[e].to(cutlass.Float32)
            xhat = (x - mean) * rstd
            c = base_c + (j0 + e) // span
            g = mGamma[c].to(cutlass.Float32)
            val = g * xhat
            if has_beta:
                val = val + mBeta[c].to(cutlass.Float32)
            yf[e] = val.to(mY.element_type)
        cute.autovec_copy(yf, gy)
        vi = vi + block_threads


@cute.jit
def _rowwise_fwd_vec_host(
    mX: cute.Tensor,
    mY: cute.Tensor,
    mGamma: cute.Tensor,
    mBeta: cute.Tensor,
    mMean: cute.Tensor,
    mRstd: cute.Tensor,
    R: cutlass.Int32,
    M: cutlass.Int32,
    gps: cutlass.Int32,
    cpg: cutlass.Int32,
    span: cutlass.Int32,
    eps: cutlass.Float32,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
    V: cutlass.Constexpr,
    block_threads: cutlass.Constexpr,
) -> None:
    _rowwise_fwd_vec_kernel(
        mX, mY, mGamma, mBeta, mMean, mRstd,
        M, gps, cpg, span, eps, has_mean, has_beta, V, block_threads,
    ).launch(grid=(R, 1, 1), block=(block_threads, 1, 1))


def rowwise_forward_vec(spec, x, gamma, beta, *, eps, V, block_threads):
    """Vectorized per-sample forward. Returns ``(y, mean, rstd)``."""
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
        cutlass.Int32(spec.R), cutlass.Int32(spec.M),
        cutlass.Int32(spec.groups_per_sample), cutlass.Int32(spec.channels_per_group),
        cutlass.Int32(spec.gamma_inner_span), cutlass.Float32(eps),
    )
    ce_args = (spec.has_mean, has_beta, V, block_threads)
    key = (io_dtype, spec.has_mean, has_beta, V, block_threads)
    fn = _CACHE.get(key)
    if fn is None:
        fn = cute.compile(_rowwise_fwd_vec_host, *runtime_args, *ce_args)
        _CACHE[key] = fn
    fn(*runtime_args)
    return y, mean, rstd
