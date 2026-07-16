"""Vectorized backward kernel for the per-sample norm family.

Same math as :mod:`cudnn.norm.bprop.frost.rowwise`; DY / X are loaded and DX is
stored ``V`` elements at a time via register fragments (128-bit vector ops).
Parameter grads ``dgamma``/``dbeta`` still accumulate with fp32 atomics.

Requires ``M % V == 0`` and 16-byte-aligned rows.
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
def _rowwise_bwd_vec_kernel(
    mDY: cute.Tensor,
    mX: cute.Tensor,
    mGamma: cute.Tensor,
    mMean: cute.Tensor,
    mRstd: cute.Tensor,
    mDX: cute.Tensor,
    mDGamma: cute.Tensor,
    mDBeta: cute.Tensor,
    M: cutlass.Int32,
    gps: cutlass.Int32,
    cpg: cutlass.Int32,
    span: cutlass.Int32,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
    V: cutlass.Constexpr,
    block_threads: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    r, _, _ = cute.arch.block_idx()

    Mf = cutlass.Float32(M)
    mean = cutlass.Float32(0.0)
    if has_mean:
        mean = mMean[r]
    rstd = mRstd[r]
    base_c = (r % gps) * cpg
    nv = M // V

    row_dy = mDY[r, None]
    row_x = mX[r, None]
    dyv = cute.zipped_divide(row_dy, (V,))
    xv = cute.zipped_divide(row_x, (V,))

    # --- pass 1: reductions + dgamma/dbeta atomics ---
    s_dxhat = cutlass.Float32(0.0)
    s_dxhat_xhat = cutlass.Float32(0.0)
    vi = tid
    while vi < nv:
        j0 = vi * V
        gdy = dyv[None, vi]
        gx = xv[None, vi]
        dyf = cute.make_fragment_like(gdy)
        xf = cute.make_fragment_like(gx)
        cute.autovec_copy(gdy, dyf)
        cute.autovec_copy(gx, xf)
        for e in cutlass.range_constexpr(V):
            c = base_c + (j0 + e) // span
            x = xf[e].to(cutlass.Float32)
            dy = dyf[e].to(cutlass.Float32)
            g = mGamma[c].to(cutlass.Float32)
            xhat = (x - mean) * rstd
            dxhat = dy * g
            s_dxhat = s_dxhat + dxhat
            s_dxhat_xhat = s_dxhat_xhat + dxhat * xhat
            cute.arch.atomic_add(mDGamma.iterator + c, dy * xhat)
            if has_beta:
                cute.arch.atomic_add(mDBeta.iterator + c, dy)
        vi = vi + block_threads

    s_dxhat, s_dxhat_xhat = block_reduce_sum2(s_dxhat, s_dxhat_xhat, tid, block_threads)
    a = cutlass.Float32(0.0)
    if has_mean:
        a = s_dxhat / Mf
    b = s_dxhat_xhat / Mf

    # --- pass 2: dx (vectorized store) ---
    row_dx = mDX[r, None]
    dxv = cute.zipped_divide(row_dx, (V,))
    vi = tid
    while vi < nv:
        j0 = vi * V
        gdy = dyv[None, vi]
        gx = xv[None, vi]
        gdx = dxv[None, vi]
        dyf = cute.make_fragment_like(gdy)
        xf = cute.make_fragment_like(gx)
        dxf = cute.make_fragment_like(gdx)
        cute.autovec_copy(gdy, dyf)
        cute.autovec_copy(gx, xf)
        for e in cutlass.range_constexpr(V):
            c = base_c + (j0 + e) // span
            x = xf[e].to(cutlass.Float32)
            dy = dyf[e].to(cutlass.Float32)
            g = mGamma[c].to(cutlass.Float32)
            xhat = (x - mean) * rstd
            dxhat = dy * g
            dx = rstd * (dxhat - a - xhat * b)
            dxf[e] = dx.to(mDX.element_type)
        cute.autovec_copy(dxf, gdx)
        vi = vi + block_threads


@cute.jit
def _rowwise_bwd_vec_host(
    mDY: cute.Tensor,
    mX: cute.Tensor,
    mGamma: cute.Tensor,
    mMean: cute.Tensor,
    mRstd: cute.Tensor,
    mDX: cute.Tensor,
    mDGamma: cute.Tensor,
    mDBeta: cute.Tensor,
    R: cutlass.Int32,
    M: cutlass.Int32,
    gps: cutlass.Int32,
    cpg: cutlass.Int32,
    span: cutlass.Int32,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
    V: cutlass.Constexpr,
    block_threads: cutlass.Constexpr,
) -> None:
    _rowwise_bwd_vec_kernel(
        mDY, mX, mGamma, mMean, mRstd, mDX, mDGamma, mDBeta,
        M, gps, cpg, span, has_mean, has_beta, V, block_threads,
    ).launch(grid=(R, 1, 1), block=(block_threads, 1, 1))


def rowwise_backward_vec(spec, dy, x, gamma, mean, rstd, *, has_beta, V, block_threads):
    """Vectorized per-sample backward. Returns ``(dx, dgamma, dbeta)``."""
    import torch

    from cudnn.norm.frost.dtypes import torch_dtype_to_str

    io_dtype = torch_dtype_to_str(x.dtype)
    dx = torch.empty_like(x)
    dgamma = torch.zeros(spec.gamma_len, dtype=torch.float32, device=x.device)
    dbeta = torch.zeros(spec.gamma_len, dtype=torch.float32, device=x.device)

    runtime_args = (
        _dyn(dy), _dyn(x), _dyn(gamma), _dyn(mean), _dyn(rstd),
        _dyn(dx), _dyn(dgamma), _dyn(dbeta),
        cutlass.Int32(spec.R), cutlass.Int32(spec.M),
        cutlass.Int32(spec.groups_per_sample), cutlass.Int32(spec.channels_per_group),
        cutlass.Int32(spec.gamma_inner_span),
    )
    ce_args = (spec.has_mean, has_beta, V, block_threads)
    key = (io_dtype, spec.has_mean, has_beta, V, block_threads)
    fn = _CACHE.get(key)
    if fn is None:
        fn = cute.compile(_rowwise_bwd_vec_host, *runtime_args, *ce_args)
        _CACHE[key] = fn
    fn(*runtime_args)
    return dx, dgamma, (dbeta if has_beta else None)
