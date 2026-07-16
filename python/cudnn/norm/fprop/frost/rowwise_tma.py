"""TMA (Tensor Memory Accelerator) smem-staged forward kernel for the per-sample
norm family.  **Requires sm_90+ (Hopper / Blackwell).**

Like the cp.async variant, this stages each row into shared memory once, but uses
a TMA bulk-tensor copy (``cp.async.bulk.tensor``) driven by a hardware descriptor
and an mbarrier, instead of per-thread ``cp.async``. TMA moves the whole tile
with a single instruction issued by one thread, freeing the rest of the CTA and
giving the best global->shared bandwidth on Hopper+.

Status: this box is an A100 (sm_80), which has no TMA, so this path cannot run
here. The TMA descriptor/atom construction (``make_tiled_tma_atom``), mbarrier
setup, and the ``cute.copy(..., tma_bar_ptr=)`` bulk-copy issue follow the public
CuTe-DSL TMA API. Compiling the full kernel for ``CUTE_DSL_ARCH=sm_90a`` on this
sm_80 box currently trips a CuTe-DSL region-isolation check (``cute.crd2idx op
using value defined outside the region``) at the ``local_tile`` /
``tma_partition`` tile-selection step -- a DSL-usage detail to resolve and then
runtime-validate on real sm_90+ hardware. Until then, use
:mod:`~cudnn.norm.fprop.frost.rowwise_cpasync` (cp.async, sm_80) which is tested
and gives the same "read X once" staging benefit.

``M`` is a compile-time constant (static shared memory + static TMA descriptor);
tensors are passed with static layout (TMA descriptors require static shapes).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
from cutlass.cute.runtime import from_dlpack

from cudnn.norm.frost.reductions import block_reduce_sum2

_CACHE = {}


def _sm_major() -> int:
    import torch

    if not torch.cuda.is_available():
        return 0
    return torch.cuda.get_device_capability(0)[0]


@cute.kernel
def _fwd_tma_kernel(
    tma_atom: cute.CopyAtom,
    mTMA: cute.Tensor,
    mY: cute.Tensor,
    mGamma: cute.Tensor,
    mBeta: cute.Tensor,
    mMean: cute.Tensor,
    mRstd: cute.Tensor,
    gps: cutlass.Int32,
    cpg: cutlass.Int32,
    span: cutlass.Int32,
    eps: cutlass.Float32,
    smem_layout: cutlass.Constexpr,
    elem_ty: cutlass.Constexpr,
    M: cutlass.Constexpr,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
    V: cutlass.Constexpr,
    block_threads: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    r, _, _ = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()
    sX = smem.allocate_tensor(elem_ty, smem_layout, byte_alignment=128)
    mbar = smem.allocate_array(cutlass.Int64, 1)

    # --- TMA load of this CTA's (1, M) row tile into smem ---
    gX = cute.local_tile(mTMA, (1, M), (r, 0))
    tAsX, tAgX = cpasync.tma_partition(
        tma_atom, 0, cute.make_layout(1),
        cute.group_modes(sX, 0, cute.rank(sX)),
        cute.group_modes(gX, 0, cute.rank(gX)),
    )
    if tid == 0:
        cute.arch.mbarrier_init(mbar, 1)
    cute.arch.sync_threads()
    nbytes = M * (elem_ty.width // 8)
    if tid == 0:
        cute.arch.mbarrier_arrive_and_expect_tx(mbar, nbytes)
        cute.copy(tma_atom, tAgX, tAsX, tma_bar_ptr=mbar)
    cute.arch.mbarrier_wait(mbar, 0)
    cute.arch.sync_threads()

    sRow = cute.recast_tensor(sX, elem_ty)  # (1, M) -> flat access via [0, j]
    Mf = cutlass.Float32(M)

    # --- pass 1: reduce from smem ---
    s1 = cutlass.Float32(0.0)
    s2 = cutlass.Float32(0.0)
    j = tid
    while j < M:
        x = sRow[0, j].to(cutlass.Float32)
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
    nv = M // V
    vi = tid
    while vi < nv:
        j0 = vi * V
        yf = cute.make_fragment_like(yv[None, vi])
        for e in cutlass.range_constexpr(V):
            x = sRow[0, j0 + e].to(cutlass.Float32)
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
def _fwd_tma_host(
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
    smem_layout = cute.make_layout((1, M))
    tma_info = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mX, smem_layout, (1, M),
    )
    _fwd_tma_kernel(
        tma_info.atom, tma_info.tma_tensor, mY, mGamma, mBeta, mMean, mRstd,
        gps, cpg, span, eps, smem_layout, mX.element_type, M,
        has_mean, has_beta, V, block_threads,
    ).launch(grid=(R, 1, 1), block=(block_threads, 1, 1))


def rowwise_forward_tma(spec, x, gamma, beta, *, eps, V, block_threads):
    """TMA smem-staged per-sample forward (sm_90+). Returns ``(y, mean, rstd)``.

    Raises on pre-sm_90 GPUs (use the cp.async path there).
    """
    import torch

    from cudnn.norm.frost.dtypes import torch_dtype_to_str

    if _sm_major() < 9:
        raise RuntimeError(
            "rowwise_forward_tma requires sm_90+ (Hopper/Blackwell); this GPU has no "
            "TMA. Use impl='cpasync' or impl='vec' instead."
        )

    io_dtype = torch_dtype_to_str(x.dtype)
    has_beta = beta is not None
    if beta is None:
        beta = torch.zeros(spec.gamma_len, dtype=x.dtype, device=x.device)

    y = torch.empty_like(x)
    rstd = torch.empty(spec.R, dtype=torch.float32, device=x.device)
    mean = torch.empty(spec.R, dtype=torch.float32, device=x.device)

    # TMA descriptors require static shapes: do NOT mark_layout_dynamic.
    a = lambda t: from_dlpack(t, assumed_align=16)
    runtime_args = (
        a(x), a(y), a(gamma), a(beta), a(mean), a(rstd),
        cutlass.Int32(spec.R),
        cutlass.Int32(spec.groups_per_sample), cutlass.Int32(spec.channels_per_group),
        cutlass.Int32(spec.gamma_inner_span), cutlass.Float32(eps),
    )
    ce_args = (spec.M, spec.has_mean, has_beta, V, block_threads)
    key = (io_dtype, spec.R, spec.M, spec.has_mean, has_beta, V, block_threads)
    fn = _CACHE.get(key)
    if fn is None:
        fn = cute.compile(_fwd_tma_host, *runtime_args, *ce_args)
        _CACHE[key] = fn
    fn(*runtime_args)
    return y, mean, rstd
