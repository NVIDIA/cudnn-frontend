"""Forward kernel for the per-sample norm family (LayerNorm / RMSNorm /
InstanceNorm / GroupNorm).

One CTA owns one normalization group (a row of the ``[R, M]`` view). The kernel
does a two-pass computation: reduce for ``mean``/``rstd`` over ``M``, then
normalize and apply the affine ``y = gamma * xhat + beta``. Statistics are
computed and stored in fp32; inputs/outputs may be bf16/fp16/fp32.

The affine channel index for element ``(r, j)`` is
``(r % gps) * cpg + (j // span)`` (see :mod:`cudnn.norm.frost.config`).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from cudnn.norm.frost.reductions import block_reduce_sum2


def _dyn(t):
    """Wrap a torch tensor as a dynamic-layout cute tensor (shape/stride generic)."""
    return from_dlpack(t).mark_layout_dynamic()


# Compiled-kernel cache keyed by (io_dtype, has_mean, has_beta, block_threads).
_CACHE = {}


@cute.kernel
def _rowwise_fwd_kernel(
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
    block_threads: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    r, _, _ = cute.arch.block_idx()

    Mf = cutlass.Float32(M)

    # --- pass 1: reduce sum(x) and sum(x^2) over the group ---
    s1 = cutlass.Float32(0.0)
    s2 = cutlass.Float32(0.0)
    j = tid
    while j < M:
        x = mX[r, j].to(cutlass.Float32)
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

    # --- pass 2: normalize + affine ---
    base_c = (r % gps) * cpg
    j = tid
    while j < M:
        x = mX[r, j].to(cutlass.Float32)
        xhat = (x - mean) * rstd
        c = base_c + (j // span)
        g = mGamma[c].to(cutlass.Float32)
        y = g * xhat
        if has_beta:
            y = y + mBeta[c].to(cutlass.Float32)
        mY[r, j] = y.to(mY.element_type)
        j = j + block_threads


@cute.jit
def _rowwise_fwd_host(
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
    block_threads: cutlass.Constexpr,
) -> None:
    _rowwise_fwd_kernel(
        mX, mY, mGamma, mBeta, mMean, mRstd,
        M, gps, cpg, span, eps, has_mean, has_beta, block_threads,
    ).launch(grid=(R, 1, 1), block=(block_threads, 1, 1))


def rowwise_forward(spec, x, gamma, beta, *, eps, block_threads):
    """Launch the per-sample forward norm on torch tensors.

    ``x`` is the contiguous ``[R, M]`` view. ``gamma``/``beta`` are length
    ``gamma_len`` fp/bf tensors (``beta`` may be a dummy when ``has_beta`` is
    False). Returns ``(y, mean, rstd)`` torch tensors (mean is a dummy zero
    tensor when the variant has no centering).
    """
    import torch

    from cudnn.norm.frost.dtypes import torch_dtype_to_str

    io_dtype = torch_dtype_to_str(x.dtype)
    has_beta = beta is not None
    if beta is None:
        beta = torch.zeros(spec.gamma_len, dtype=x.dtype, device=x.device)

    y = torch.empty_like(x)
    rstd = torch.empty(spec.R, dtype=torch.float32, device=x.device)
    mean = torch.empty(spec.R, dtype=torch.float32, device=x.device)

    # Runtime args are re-passed each call; Constexpr args are baked at compile
    # time and MUST NOT be passed again to the compiled callable.
    runtime_args = (
        _dyn(x), _dyn(y), _dyn(gamma), _dyn(beta), _dyn(mean), _dyn(rstd),
        cutlass.Int32(spec.R), cutlass.Int32(spec.M),
        cutlass.Int32(spec.groups_per_sample), cutlass.Int32(spec.channels_per_group),
        cutlass.Int32(spec.gamma_inner_span), cutlass.Float32(eps),
    )
    ce_args = (spec.has_mean, has_beta, block_threads)
    key = (io_dtype, spec.has_mean, has_beta, block_threads)
    fn = _CACHE.get(key)
    if fn is None:
        fn = cute.compile(_rowwise_fwd_host, *runtime_args, *ce_args)
        _CACHE[key] = fn
    fn(*runtime_args)
    return y, mean, rstd
