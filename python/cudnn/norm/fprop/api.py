# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""High-level torch-tensor dispatch for the sm_100 norm forward kernels.

Takes torch tensors in their natural layout, derives the
:class:`~cudnn.norm.config_sm100.RowwiseSpec` / :class:`BatchNormSpec`, resolves
the launch :class:`Cfg`, reshapes to the canonical view each kernel expects, and
launches. This is the direct (non-graph) entry point used by the tests; the
cuDNN graph engine (``cudnn.norm.fprop.engines``) lowers onto the same kernels.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..config_sm100 import (
    STAGE_NONE,
    Cfg,
    NormVariant,
    ROWWISE_VARIANTS,
    TemplateParams,
    batchnorm_spec,
    choose_block_threads,
    make_cfg,
    make_warp_cfg,
    rowwise_spec,
    vector_width,
)

# LN/RMS reduce the last dim (M == C == gamma_len) and can use the warp-per-row
# kernel; GN/IN have a different affine mapping and stay on the row-reduce kernel.
_WARP_VARIANTS = (NormVariant.LAYER_NORM, NormVariant.RMS_NORM)
from ..dtypes import DTYPE_BYTES, torch_dtype_to_str
from .kernels import (
    batchnorm_nchw_sm100,
    batchnorm_nhwc_sm100,
    batchnorm_sm100,
    groupnorm_nhwc_sm100,
    groupnorm_sm100,
    instancenorm_nhwc_sm100,
    instancenorm_sm100,
    layernorm_sm100,
    rmsnorm_sm100,
)

_ROWWISE_KERNEL = {
    NormVariant.LAYER_NORM: layernorm_sm100,
    NormVariant.RMS_NORM: rmsnorm_sm100,
    NormVariant.GROUP_NORM: groupnorm_sm100,
    NormVariant.INSTANCE_NORM: instancenorm_sm100,
}


def _flat(t):
    """Per-channel vectors as 1-D, whatever rank the caller passed.

    The kernels index gamma/beta/mean/rstd FLAT (``mG[c]``, ``mMean[row]``), so a
    rank-3 ``scale`` of shape (1, 1, C) or a rank-3 ``mean`` of (N, S, 1) -- which
    is exactly what a cuDNN graph hands over -- indexes wrong and produces silently
    incorrect dx / dgamma. (dbeta survives, because it never touches these.) These
    are contiguous per-channel vectors, so the reshape is a view.
    """
    return t if t is None or t.dim() == 1 else t.reshape(-1)


def _as_variant(v) -> NormVariant:
    return v if isinstance(v, NormVariant) else NormVariant(v)


# Variants with a native channels-last forward kernel.
_NHWC_NATIVE = (NormVariant.BATCH_NORM, NormVariant.INSTANCE_NORM, NormVariant.GROUP_NORM)


def _nhwc_kernel_serves(variant, shape, io, num_groups):
    """True when a NATIVE channels-last kernel will actually take this shape.

    Keeping an NHWC input un-``contiguous()``-ed is only sound if one of those
    kernels runs. When none does, the fall-through views the tensor as the
    rowwise or (N, C, S) shape the contiguous kernels expect -- and for
    BatchNorm that view EXISTS in channels-last memory (merging H and W is
    legal there), so reshape silently returns a strided tensor instead of
    copying. The kernel then addresses it as if it were contiguous. fp32
    [4,24,8,8], whose C=24 no NHWC tile accepts, came back as 6144 untouched
    NaNs that way.

    So the layout decision has to ask the kernels, not just the variant.
    """
    from cudnn.norm.dtypes import DTYPE_BYTES

    from .kernels import batchnorm_nhwc_sm100, groupnorm_nhwc_sm100, instancenorm_nhwc_sm100

    if len(shape) != 4:
        return False
    eb = DTYPE_BYTES[io]
    C = int(shape[1])
    if variant == NormVariant.BATCH_NORM:
        return batchnorm_nhwc_sm100.nhwc_cfg(C, eb) is not None
    if variant == NormVariant.INSTANCE_NORM:
        return instancenorm_nhwc_sm100.nhwc_cfg(C, eb) is not None
    if variant == NormVariant.GROUP_NORM:
        if not num_groups or C % int(num_groups):
            return False
        return groupnorm_nhwc_sm100.eligible(C, C // int(num_groups), eb)
    return False


def _deliver(y, mean, rstd, out_y, out_mean, out_rstd):
    """Return the results, landing them in the caller's buffers if the kernel could not.

    ``out=`` is honoured either way. The kernels write those buffers directly
    when the layout allows; when it does not -- an NHWC input that no native
    kernel serves has to be made contiguous first, and the caller's
    channels-last buffer then no longer matches -- the copy happens here
    instead. Silently leaving an ``out=`` buffer untouched is exactly the kind
    of result that looks fine until someone reads the buffer.

    Returning the caller's own tensor keeps the engine's copy a no-op rather
    than a second one.
    """
    out = []
    for got, want in ((y, out_y), (mean, out_mean), (rstd, out_rstd)):
        if want is None or got is None or want.data_ptr() == got.data_ptr():
            out.append(got)
        else:
            want.copy_(got.reshape(want.shape))
            out.append(want)
    return tuple(out)


def _require_io_dtype(x, **operands):
    """Reject operands the kernels would reinterpret rather than convert.

    gamma/beta/DY are read AS the I/O dtype. The graph engines already decline
    mixed precision at check time; the torch API had no such gate, so an fp32
    gamma against a bf16 input was read bit-for-bit as bf16. It was worse than a
    local mistake: the compile cache keys on the INPUT dtype, not the
    parameter's, so a shape first compiled with matching operands would serve a
    later mismatched call out of cache and return garbage (measured 3.1
    absolute against an fp64 reference). Failing here makes that diagnosable.
    """
    for name, t in operands.items():
        if t is not None and t.dtype != x.dtype:
            raise ValueError(
                f"cudnn.norm: {name} is {t.dtype} but the input is {x.dtype}; these kernels "
                f"read {name} at the input dtype rather than converting it. Cast {name} first."
            )


def _stat_out(t, n):
    """Flat view of a caller's mean/rstd buffer, or None when it is not usable.

    The kernels write statistics as a flat fp32 array of exactly ``n`` entries;
    anything else (wrong count, wrong dtype, strided) gets a temporary instead.
    """
    import torch

    if t is None or t.dtype != torch.float32 or t.numel() != n or not t.is_contiguous():
        return None
    return t.reshape(-1)


def _is_nhwc(x) -> bool:
    """True for a 4-D tensor whose memory is channels-last (NHWC)."""
    import torch

    return x.dim() == 4 and x.is_contiguous(memory_format=torch.channels_last)


def fprop_workspace_bytes(variant, shape, io, *, normalized_shape=None, num_groups=None, nhwc=False):
    """Upper bound on the scratch :func:`norm_fprop` may carve for this call.

    Takes the MAX over every kernel the routing could pick rather than
    reproducing the if/else chain. The routing is performance-tuned and moves;
    a sizer that drifted from it would under-report, and the kernels would
    quietly fall back to allocating -- which is the thing the workspace exists
    to prevent. Over-reporting only costs address space.

    Returns 0 for the routes that need no scratch at all, which is every
    rowwise LayerNorm / RMSNorm / GroupNorm / InstanceNorm forward.
    """
    from .kernels import batchnorm_nchw_sm100, batchnorm_nhwc_sm100, groupnorm_nhwc_sm100, instancenorm_nhwc_sm100

    variant = _as_variant(variant)
    shape = tuple(int(v) for v in shape)
    best = 0
    # Only an unsupported SHAPE is tolerated here. A broad except would turn a
    # genuine bug into a silent 0, and a 0 means the kernels allocate -- the
    # exact failure this function exists to prevent, and an invisible one.
    try:
        if variant in (NormVariant.GROUP_NORM, NormVariant.INSTANCE_NORM) and nhwc and len(shape) == 4:
            N, C, H, W = shape
            spec = rowwise_spec(variant, shape, normalized_shape=normalized_shape, num_groups=num_groups)
            mod = groupnorm_nhwc_sm100 if variant == NormVariant.GROUP_NORM else instancenorm_nhwc_sm100
            best = max(best, mod.scratch_bytes(spec, TemplateParams(variant=variant, io_dtype=io), (N, H * W, C)))
        elif variant == NormVariant.BATCH_NORM:
            spec = batchnorm_spec(shape)
            params = TemplateParams(variant=variant, io_dtype=io)
            best = max(best, batchnorm_nchw_sm100.scratch_bytes(params=params, spec=spec, shape=(spec.N, spec.C, spec.S)))
            if nhwc:
                best = max(best, batchnorm_nhwc_sm100.scratch_bytes(params=params, spec=spec, shape=(spec.N * spec.S, spec.C)))
    except ValueError:  # a shape no kernel accepts needs no workspace
        return 0
    return best


def norm_fprop(
    variant,
    x,
    gamma=None,
    beta=None,
    *,
    eps: float = 1e-5,
    normalized_shape: Optional[Sequence[int]] = None,
    num_groups: Optional[int] = None,
    momentum: float = 0.1,
    training: bool = True,
    running_mean=None,
    running_var=None,
    out=None,
    ws=None,
):
    """Forward pass for any of the five norm variants.

    Returns ``(y, mean, rstd)``. ``y`` has the input's shape; ``mean``/``rstd``
    are fp32 per-group statistics (flat). For RMSNorm ``mean`` is a placeholder.

    ``out`` is an optional ``(y, mean, rstd)`` of caller-owned buffers to write
    in place. The graph engines pass the graph's own tensors, which is what lets
    execute() allocate nothing -- and it drops a full-tensor copy that otherwise
    costs the graph path ~1.6x the direct one. A buffer that cannot be used is
    ignored and a temporary takes its place, so this stays an optimisation
    rather than a contract the caller has to satisfy exactly.

    ``ws`` is a :class:`cudnn.norm.utils.Workspace` the cooperative kernels
    carve their scratch from; size it with :func:`fprop_workspace_bytes`. Left
    at None (the torch API) they allocate as before.
    """
    import torch

    variant = _as_variant(variant)
    # BN/IN/GN have dedicated channels-last kernels, so preserve NHWC inputs
    # instead of paying a transpose in ``.contiguous()``. The output keeps the
    # input's layout, which is what makes a channels-last forward -> backward
    # chain stay on the native NHWC kernels end to end. Only when one of those
    # kernels will actually run, though -- see _nhwc_kernel_serves.
    _io = torch_dtype_to_str(x.dtype)
    if not (variant in _NHWC_NATIVE and _is_nhwc(x) and _nhwc_kernel_serves(variant, x.shape, _io, num_groups)):
        x = x.contiguous()
    gamma, beta = _flat(gamma), _flat(beta)
    running_mean, running_var = _flat(running_mean), _flat(running_var)
    _require_io_dtype(x, gamma=gamma, beta=beta)
    io = torch_dtype_to_str(x.dtype)

    # Every route below views the output with the SAME expression it views x, so
    # a y buffer is usable exactly when it is laid out like the (already
    # normalised) x. Checked once here rather than per route.
    out_y, out_mean, out_rstd = out if out is not None else (None, None, None)
    # ``w_y`` is what the kernels may write IN PLACE; ``out_y`` stays the caller's
    # buffer so _deliver can still copy into it when they cannot. Keeping only the
    # validated one would silently leave an out= buffer untouched.
    w_y = out_y
    if w_y is not None and (tuple(w_y.shape) != tuple(x.shape) or w_y.stride() != x.stride() or w_y.dtype != x.dtype):
        w_y = None

    if variant in ROWWISE_VARIANTS:
        spec = rowwise_spec(variant, x.shape, normalized_shape=normalized_shape, num_groups=num_groups)
        if gamma is None:
            gamma = torch.ones(spec.gamma_len, dtype=x.dtype, device=x.device)
        params = TemplateParams(variant=variant, io_dtype=io, has_beta=(beta is not None))

        # --- channels-last GN/IN: route BEFORE the rowwise reshape, which would
        # silently materialise a transpose for a channels-last tensor. ---
        if variant == NormVariant.GROUP_NORM and _is_nhwc(x):
            N, C, H, W = (int(v) for v in x.shape)
            if groupnorm_nhwc_sm100.eligible(C, int(spec.channels_per_group), DTYPE_BYTES[io]):
                x3 = x.permute(0, 2, 3, 1).reshape(N, H * W, C)  # view, no copy
                o3 = w_y.permute(0, 2, 3, 1).reshape(N, H * W, C) if w_y is not None else None
                y3, mean, rstd = groupnorm_nhwc_sm100.forward(
                    spec,
                    x3,
                    gamma,
                    beta,
                    eps=eps,
                    cfg=None,
                    params=params,
                    out_y=o3,
                    out_mean=_stat_out(out_mean, spec.R),
                    out_rstd=_stat_out(out_rstd, spec.R),
                    ws=ws,
                )
                return _deliver(y3.reshape(N, H, W, C).permute(0, 3, 1, 2), mean, rstd, out_y, out_mean, out_rstd)
        if variant == NormVariant.INSTANCE_NORM and _is_nhwc(x):
            N, C, H, W = (int(v) for v in x.shape)
            if instancenorm_nhwc_sm100.nhwc_cfg(C, DTYPE_BYTES[io]) is not None:
                x3 = x.permute(0, 2, 3, 1).reshape(N, H * W, C)  # view, no copy
                o3 = w_y.permute(0, 2, 3, 1).reshape(N, H * W, C) if w_y is not None else None
                y3, mean, rstd = instancenorm_nhwc_sm100.forward(
                    spec,
                    x3,
                    gamma,
                    beta,
                    eps=eps,
                    cfg=None,
                    params=params,
                    out_y=o3,
                    out_mean=_stat_out(out_mean, spec.R),
                    out_rstd=_stat_out(out_rstd, spec.R),
                    ws=ws,
                )
                return _deliver(y3.reshape(N, H, W, C).permute(0, 3, 1, 2), mean, rstd, out_y, out_mean, out_rstd)

        x2d = x.reshape(spec.R, spec.M)
        o2d = w_y.reshape(spec.R, spec.M) if w_y is not None else None
        om, orstd = _stat_out(out_mean, spec.R), _stat_out(out_rstd, spec.R)

        # Warp-per-row kernel for LN/RMS when the row fits the register budget.
        if variant in _WARP_VARIANTS:
            # A long row first: one CTA per row caps parallelism and has to stage the
            # whole row, so past ~16K elements splitting it across a CGA -- each CTA
            # reducing its slice, then a reduction through distributed shared memory --
            # beats it by 2-4x. See layernorm_cga_sm100 for the measured crossover.
            from .kernels import layernorm_cga_sm100

            if layernorm_cga_sm100.should_use(spec.M, spec.R, DTYPE_BYTES[io]):
                y2, mean, rstd = layernorm_cga_sm100.forward(spec, x2d, gamma, beta, eps=eps, params=params, out_y=o2d, out_mean=om, out_rstd=orstd)
                return _deliver(y2.reshape(x.shape), mean, rstd, out_y, out_mean, out_rstd)

            wcfg = make_warp_cfg(params, spec.M)
            if wcfg is not None:
                from .kernels import layernorm_warp_sm100

                # The warp kernels cache the whole gamma/beta row (and, pipelined, the
                # staged X rows) in smem, so a long enough row does not fit. Check before
                # committing: otherwise the launch fails in the driver instead of falling
                # through to the staged path, which handles any C.
                if layernorm_warp_sm100.eligible(spec, params, wcfg, beta is not None):
                    y2, mean, rstd = layernorm_warp_sm100.forward(
                        spec, x2d, gamma, beta, eps=eps, wcfg=wcfg, params=params, out_y=o2d, out_mean=om, out_rstd=orstd
                    )
                    return _deliver(y2.reshape(x.shape), mean, rstd, out_y, out_mean, out_rstd)

        # GN/IN with too few (sample, group) rows to fill the machine: the rowwise
        # kernel is one CTA per row, so N=2/C=256/G=2 puts four CTAs on 148 SMs.
        # Split each row across a cluster and reduce through distributed shared memory.
        if variant in (NormVariant.GROUP_NORM, NormVariant.INSTANCE_NORM):
            from .kernels import groupnorm_cga_sm100

            if groupnorm_cga_sm100.eligible(spec.M, spec.R, DTYPE_BYTES[io]):
                y2, mean, rstd = groupnorm_cga_sm100.forward(spec, x2d, gamma, beta, eps=eps, params=params, out_y=o2d, out_mean=om, out_rstd=orstd)
                return _deliver(y2.reshape(x.shape), mean, rstd, out_y, out_mean, out_rstd)

        cfg = make_cfg(params, spec.M)
        y2, mean, rstd = _ROWWISE_KERNEL[variant].forward(spec, x2d, gamma, beta, eps=eps, cfg=cfg, params=params, out_y=o2d, out_mean=om, out_rstd=orstd)
        return _deliver(y2.reshape(x.shape), mean, rstd, out_y, out_mean, out_rstd)

    if variant == NormVariant.BATCH_NORM:
        spec = batchnorm_spec(x.shape)
        if gamma is None:
            gamma = torch.ones(spec.C, dtype=x.dtype, device=x.device)
        update_running = bool(training and running_mean is not None and running_var is not None)
        params = TemplateParams(
            variant=variant,
            io_dtype=io,
            has_beta=(beta is not None),
            training=training,
            update_running=update_running,
        )
        eb = DTYPE_BYTES[io]

        # --- NHWC (channels-last): fused cooperative split-K kernel over [M=N*H*W, C] ---
        if _is_nhwc(x) and batchnorm_nhwc_sm100.nhwc_cfg(spec.C, eb) is not None:
            N, C, H, W = (int(v) for v in x.shape)
            x2d = x.permute(0, 2, 3, 1).reshape(N * H * W, C)  # view, no copy
            o2d = out_y.permute(0, 2, 3, 1).reshape(N * H * W, C) if w_y is not None else None
            y2d, sm, sr = batchnorm_nhwc_sm100.forward(
                spec,
                x2d,
                gamma,
                beta,
                eps=eps,
                momentum=momentum,
                training=training,
                running_mean=running_mean,
                running_var=running_var,
                cfg=None,
                params=params,
                out_y=o2d,
                ws=ws,
                out_mean=_stat_out(out_mean, spec.C),
                out_rstd=_stat_out(out_rstd, spec.C),
            )
            y = y2d.reshape(N, H, W, C).permute(0, 3, 1, 2)
            return _deliver(y, sm, sr, out_y, out_mean, out_rstd)

        # --- NCHW: split-K over the (n, c) rows, each of S contiguous elements ---
        if batchnorm_nchw_sm100.nchw_cfg(spec.C, spec.N, spec.S, eb) is not None:
            x3d = x.reshape(spec.N, spec.C, spec.S)
            o3d = w_y.reshape(spec.N, spec.C, spec.S) if w_y is not None else None
            y3, sm, sr = batchnorm_nchw_sm100.forward(
                spec,
                x3d,
                gamma,
                beta,
                eps=eps,
                momentum=momentum,
                training=training,
                running_mean=running_mean,
                running_var=running_var,
                cfg=None,
                params=params,
                out_y=o3d,
                ws=ws,
                out_mean=_stat_out(out_mean, spec.C),
                out_rstd=_stat_out(out_rstd, spec.C),
            )
            return _deliver(y3.reshape(x.shape), sm, sr, out_y, out_mean, out_rstd)

        cfg = Cfg(block_threads=choose_block_threads(spec.count), V=vector_width(eb), stage_mode=STAGE_NONE, vec=False, elem_bytes=eb)
        x3d = x.reshape(spec.N, spec.C, spec.S)
        o3d = w_y.reshape(spec.N, spec.C, spec.S) if w_y is not None else None
        y3, sm, sr = batchnorm_sm100.forward(
            spec,
            x3d,
            gamma,
            beta,
            eps=eps,
            momentum=momentum,
            training=training,
            running_mean=running_mean,
            running_var=running_var,
            cfg=cfg,
            params=params,
            out_y=o3d,
            out_mean=_stat_out(out_mean, spec.C),
            out_rstd=_stat_out(out_rstd, spec.C),
        )
        return _deliver(y3.reshape(x.shape), sm, sr, out_y, out_mean, out_rstd)

    raise ValueError(f"unknown norm variant {variant}")
