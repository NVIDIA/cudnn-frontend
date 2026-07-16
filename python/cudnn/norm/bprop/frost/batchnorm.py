"""BatchNorm backward.

Per channel ``c`` (one CTA), with ``xhat = (x - mean) * rstd``::

    dbeta_c  = sum over (n, s) of dy
    dgamma_c = sum over (n, s) of dy * xhat
    dx = gamma_c * rstd * (dy - dbeta_c / count - xhat * dgamma_c / count)

Because each channel is reduced by a single CTA, dgamma/dbeta need no atomics.
Input viewed as ``[N, C, S]``.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from cudnn.norm.frost.reductions import block_reduce_sum2


def _dyn(t):
    return from_dlpack(t).mark_layout_dynamic()


# Compiled-kernel cache keyed by (io_dtype, has_beta, block_threads).
_CACHE = {}


@cute.kernel
def _bn_bwd_kernel(
    mDY: cute.Tensor,
    mX: cute.Tensor,
    mGamma: cute.Tensor,
    mMean: cute.Tensor,
    mRstd: cute.Tensor,
    mDX: cute.Tensor,
    mDGamma: cute.Tensor,
    mDBeta: cute.Tensor,
    N: cutlass.Int32,
    S: cutlass.Int32,
    has_beta: cutlass.Constexpr,
    block_threads: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    c, _, _ = cute.arch.block_idx()
    count = N * S
    countf = cutlass.Float32(count)

    mean = mMean[c].to(cutlass.Float32)
    rstd = mRstd[c].to(cutlass.Float32)
    g = mGamma[c].to(cutlass.Float32)

    # --- reduce dy and dy*xhat over the channel ---
    s_dy = cutlass.Float32(0.0)
    s_dy_xhat = cutlass.Float32(0.0)
    i = tid
    while i < count:
        n = i // S
        s = i % S
        x = mX[n, c, s].to(cutlass.Float32)
        dy = mDY[n, c, s].to(cutlass.Float32)
        xhat = (x - mean) * rstd
        s_dy = s_dy + dy
        s_dy_xhat = s_dy_xhat + dy * xhat
        i = i + block_threads
    s_dy, s_dy_xhat = block_reduce_sum2(s_dy, s_dy_xhat, tid, block_threads)

    if tid == 0:
        mDGamma[c] = s_dy_xhat
        if has_beta:
            mDBeta[c] = s_dy

    a = s_dy / countf
    b = s_dy_xhat / countf
    i = tid
    while i < count:
        n = i // S
        s = i % S
        x = mX[n, c, s].to(cutlass.Float32)
        dy = mDY[n, c, s].to(cutlass.Float32)
        xhat = (x - mean) * rstd
        dx = g * rstd * (dy - a - xhat * b)
        mDX[n, c, s] = dx.to(mDX.element_type)
        i = i + block_threads


@cute.jit
def _bn_bwd_host(
    mDY: cute.Tensor,
    mX: cute.Tensor,
    mGamma: cute.Tensor,
    mMean: cute.Tensor,
    mRstd: cute.Tensor,
    mDX: cute.Tensor,
    mDGamma: cute.Tensor,
    mDBeta: cute.Tensor,
    C: cutlass.Int32,
    N: cutlass.Int32,
    S: cutlass.Int32,
    has_beta: cutlass.Constexpr,
    block_threads: cutlass.Constexpr,
) -> None:
    _bn_bwd_kernel(
        mDY, mX, mGamma, mMean, mRstd, mDX, mDGamma, mDBeta,
        N, S, has_beta, block_threads,
    ).launch(grid=(C, 1, 1), block=(block_threads, 1, 1))


def batchnorm_backward(spec, dy, x, gamma, saved_mean, saved_rstd, *, has_beta, block_threads):
    """Launch BatchNorm backward. Returns ``(dx, dgamma, dbeta)``.

    ``dbeta`` is ``None`` when ``has_beta`` is False.
    """
    import torch

    from cudnn.norm.frost.dtypes import torch_dtype_to_str

    io_dtype = torch_dtype_to_str(x.dtype)
    dx = torch.empty_like(x)
    dgamma = torch.zeros(spec.C, dtype=torch.float32, device=x.device)
    dbeta = torch.zeros(spec.C, dtype=torch.float32, device=x.device)

    runtime_args = (
        _dyn(dy), _dyn(x), _dyn(gamma), _dyn(saved_mean), _dyn(saved_rstd),
        _dyn(dx), _dyn(dgamma), _dyn(dbeta),
        cutlass.Int32(spec.C), cutlass.Int32(spec.N), cutlass.Int32(spec.S),
    )
    ce_args = (has_beta, block_threads)
    key = (io_dtype, has_beta, block_threads)
    fn = _CACHE.get(key)
    if fn is None:
        fn = cute.compile(_bn_bwd_host, *runtime_args, *ce_args)
        _CACHE[key] = fn
    fn(*runtime_args)
    return dx, dgamma, (dbeta if has_beta else None)
