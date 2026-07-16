"""BatchNorm forward.

BatchNorm normalizes across the batch *per channel*: for channel ``c`` the
reduction group is ``{x[n, c, s] : all n, all spatial s}``. One CTA owns one
channel. Optionally maintains running mean/var with momentum (training) or
consumes them directly (inference).

Input is viewed as ``[N, C, S]`` (S = product of spatial dims, 1 for 2-D).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from cudnn.norm.frost.reductions import block_reduce_sum2


def _dyn(t):
    return from_dlpack(t).mark_layout_dynamic()


# Compiled-kernel cache keyed by (io_dtype, has_beta, training, update_running, block_threads).
_CACHE = {}


@cute.kernel
def _bn_fwd_kernel(
    mX: cute.Tensor,
    mY: cute.Tensor,
    mGamma: cute.Tensor,
    mBeta: cute.Tensor,
    mSavedMean: cute.Tensor,
    mSavedRstd: cute.Tensor,
    mRunMean: cute.Tensor,
    mRunVar: cute.Tensor,
    N: cutlass.Int32,
    S: cutlass.Int32,
    eps: cutlass.Float32,
    momentum: cutlass.Float32,
    has_beta: cutlass.Constexpr,
    training: cutlass.Constexpr,
    update_running: cutlass.Constexpr,
    block_threads: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    c, _, _ = cute.arch.block_idx()
    count = N * S
    countf = cutlass.Float32(count)

    mean = cutlass.Float32(0.0)
    var = cutlass.Float32(0.0)
    rstd = cutlass.Float32(0.0)
    if training:
        s1 = cutlass.Float32(0.0)
        s2 = cutlass.Float32(0.0)
        i = tid
        while i < count:
            n = i // S
            s = i % S
            x = mX[n, c, s].to(cutlass.Float32)
            s1 = s1 + x
            s2 = s2 + x * x
            i = i + block_threads
        s1, s2 = block_reduce_sum2(s1, s2, tid, block_threads)
        mean = s1 / countf
        var = s2 / countf - mean * mean
        rstd = cute.math.rsqrt(var + eps)
        if tid == 0:
            mSavedMean[c] = mean
            mSavedRstd[c] = rstd
            if update_running:
                # PyTorch semantics: running_var uses the unbiased estimator.
                unbiased = var * countf / (countf - 1.0)
                mRunMean[c] = (1.0 - momentum) * mRunMean[c] + momentum * mean
                mRunVar[c] = (1.0 - momentum) * mRunVar[c] + momentum * unbiased
    else:
        mean = mRunMean[c].to(cutlass.Float32)
        var = mRunVar[c].to(cutlass.Float32)
        rstd = cute.math.rsqrt(var + eps)

    g = mGamma[c].to(cutlass.Float32)
    b = mBeta[c].to(cutlass.Float32) if has_beta else cutlass.Float32(0.0)
    i = tid
    while i < count:
        n = i // S
        s = i % S
        x = mX[n, c, s].to(cutlass.Float32)
        y = g * ((x - mean) * rstd) + b
        mY[n, c, s] = y.to(mY.element_type)
        i = i + block_threads


@cute.jit
def _bn_fwd_host(
    mX: cute.Tensor,
    mY: cute.Tensor,
    mGamma: cute.Tensor,
    mBeta: cute.Tensor,
    mSavedMean: cute.Tensor,
    mSavedRstd: cute.Tensor,
    mRunMean: cute.Tensor,
    mRunVar: cute.Tensor,
    C: cutlass.Int32,
    N: cutlass.Int32,
    S: cutlass.Int32,
    eps: cutlass.Float32,
    momentum: cutlass.Float32,
    has_beta: cutlass.Constexpr,
    training: cutlass.Constexpr,
    update_running: cutlass.Constexpr,
    block_threads: cutlass.Constexpr,
) -> None:
    _bn_fwd_kernel(
        mX, mY, mGamma, mBeta, mSavedMean, mSavedRstd, mRunMean, mRunVar,
        N, S, eps, momentum, has_beta, training, update_running, block_threads,
    ).launch(grid=(C, 1, 1), block=(block_threads, 1, 1))


def batchnorm_forward(
    spec, x, gamma, beta, *, eps, momentum, training,
    running_mean=None, running_var=None, block_threads,
):
    """Launch BatchNorm forward on torch tensors.

    Returns ``(y, saved_mean, saved_rstd)``. When ``training`` and running
    tensors are supplied, they are updated in place.
    """
    import torch

    from cudnn.norm.frost.dtypes import torch_dtype_to_str

    io_dtype = torch_dtype_to_str(x.dtype)
    has_beta = beta is not None
    if beta is None:
        beta = torch.zeros(spec.C, dtype=x.dtype, device=x.device)

    update_running = bool(training and running_mean is not None and running_var is not None)
    if running_mean is None:
        running_mean = torch.zeros(spec.C, dtype=torch.float32, device=x.device)
    if running_var is None:
        running_var = torch.ones(spec.C, dtype=torch.float32, device=x.device)

    y = torch.empty_like(x)
    saved_mean = torch.empty(spec.C, dtype=torch.float32, device=x.device)
    saved_rstd = torch.empty(spec.C, dtype=torch.float32, device=x.device)

    runtime_args = (
        _dyn(x), _dyn(y), _dyn(gamma), _dyn(beta),
        _dyn(saved_mean), _dyn(saved_rstd), _dyn(running_mean), _dyn(running_var),
        cutlass.Int32(spec.C), cutlass.Int32(spec.N), cutlass.Int32(spec.S),
        cutlass.Float32(eps), cutlass.Float32(momentum),
    )
    ce_args = (has_beta, bool(training), update_running, block_threads)
    key = (io_dtype, has_beta, bool(training), update_running, block_threads)
    fn = _CACHE.get(key)
    if fn is None:
        fn = cute.compile(_bn_fwd_host, *runtime_args, *ce_args)
        _CACHE[key] = fn
    fn(*runtime_args)
    return y, saved_mean, saved_rstd
