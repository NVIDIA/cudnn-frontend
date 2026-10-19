# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""High-level torch-tensor dispatch for the sm_100 norm backward kernels.

Takes ``dy`` plus the forward's saved ``x``, ``gamma``, ``mean``, ``rstd``,
derives the layout + launch :class:`Cfg`, and launches. Returns
``(dx, dgamma, dbeta)`` (``dbeta`` is ``None`` when ``has_beta`` is False);
``dgamma``/``dbeta`` are fp32.
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
    rowwise_spec,
    vector_width,
)
from ..dtypes import DTYPE_BYTES, torch_dtype_to_str
from .kernels import (
    batchnorm_nchw_sm100,
    batchnorm_nhwc_sm100,
    groupnorm_fast_sm100,
    groupnorm_nhwc_sm100,
    instancenorm_nhwc_sm100,
    instancenorm_warp_sm100,
    batchnorm_sm100,
    groupnorm_sm100,
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


def _deliver(dx, dgamma, dbeta, out_dx, out_dgamma, out_dbeta):
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
    for got, want in ((dx, out_dx), (dgamma, out_dgamma), (dbeta, out_dbeta)):
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
    """Flat view of a caller's dgamma/dbeta buffer, or None when it is not usable."""
    import torch

    if t is None or t.dtype != torch.float32 or t.numel() != n or not t.is_contiguous():
        return None
    return t.reshape(-1)


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


def _as_variant(v) -> NormVariant:
    return v if isinstance(v, NormVariant) else NormVariant(v)


def _is_nhwc(t) -> bool:
    """True for a 4-D tensor whose memory is channels-last (NHWC)."""
    import torch

    return t.dim() == 4 and t.is_contiguous(memory_format=torch.channels_last)


def bprop_workspace_bytes(variant, shape, io, *, normalized_shape=None, num_groups=None, has_beta=True, nhwc=False):
    """Upper bound on the scratch :func:`norm_bprop` may carve for this call.

    Max over every kernel the routing could pick, for the same reason as the
    forward sizer: the routing is tuned and moves, and a sizer that drifted
    from it would under-report, sending the kernels back to torch for the
    partial buffers the workspace exists to hold. Over-reporting costs only
    address space -- and the dominant term, LayerNorm's per-CTA dgamma
    partials, is bounded by the grid cap rather than by the input size.
    """
    from .kernels import (
        batchnorm_nchw_sm100,
        batchnorm_nhwc_sm100,
        groupnorm_cga_sm100,
        groupnorm_fast_sm100,
        groupnorm_nhwc_sm100,
        instancenorm_nhwc_sm100,
        instancenorm_warp_sm100,
        layernorm_cga_sm100,
        layernorm_sm100,
    )

    variant = _as_variant(variant)
    shape = tuple(int(v) for v in shape)
    params = TemplateParams(variant=variant, io_dtype=io, has_beta=has_beta)
    best = 0
    # Only an unsupported SHAPE is tolerated; see fprop_workspace_bytes.
    try:
        if variant in (NormVariant.LAYER_NORM, NormVariant.RMS_NORM):
            spec = rowwise_spec(variant, shape, normalized_shape=normalized_shape, num_groups=num_groups)
            for mod in (layernorm_sm100, layernorm_cga_sm100):
                best = max(best, mod.scratch_bytes(spec, params, has_beta=has_beta))
        elif variant in (NormVariant.GROUP_NORM, NormVariant.INSTANCE_NORM):
            spec = rowwise_spec(variant, shape, normalized_shape=normalized_shape, num_groups=num_groups)
            for mod in (groupnorm_cga_sm100, groupnorm_fast_sm100, instancenorm_warp_sm100):
                best = max(best, mod.scratch_bytes(spec, params, has_beta=has_beta))
            if nhwc and len(shape) == 4:
                N, C, H, W = shape
                mod = groupnorm_nhwc_sm100 if variant == NormVariant.GROUP_NORM else instancenorm_nhwc_sm100
                best = max(best, mod.scratch_bytes(spec, params, (N, H * W, C), has_beta=has_beta))
        elif variant == NormVariant.BATCH_NORM:
            spec = batchnorm_spec(shape)
            best = max(best, batchnorm_nchw_sm100.scratch_bytes(spec, params, (spec.N, spec.C, spec.S), has_beta=has_beta))
            if nhwc:
                best = max(best, batchnorm_nhwc_sm100.scratch_bytes(spec, params, (spec.N * spec.S, spec.C), has_beta=has_beta))
    except ValueError:  # a shape no kernel accepts needs no workspace
        return 0
    return best


def norm_bprop(
    variant,
    dy,
    x,
    gamma,
    mean,
    rstd,
    *,
    normalized_shape: Optional[Sequence[int]] = None,
    num_groups: Optional[int] = None,
    has_beta: bool = True,
    out=None,
    ws=None,
):
    """Backward pass for any of the five norm variants.

    ``mean``/``rstd`` are the fp32 statistics returned by ``norm_fprop``.
    Returns ``(dx, dgamma, dbeta)``.

    ``out`` is an optional ``(dx, dgamma, dbeta)`` of caller-owned buffers to
    write in place, exactly as in :func:`norm_fprop` -- the graph engines pass
    the graph's own tensors so execute() allocates nothing and does not copy DX.
    Kernels that accumulate into dgamma/dbeta clear the buffer first, so a
    caller need not hand over zeroed memory.

    ``ws`` is a :class:`cudnn.norm.utils.Workspace` the kernels carve their
    partial buffers from; size it with :func:`bprop_workspace_bytes`.
    """
    variant = _as_variant(variant)
    # BatchNorm has a native channels-last backward, so preserve NHWC instead of
    # paying the transpose that .contiguous() would do.
    _nhwc_native = variant in (NormVariant.BATCH_NORM, NormVariant.INSTANCE_NORM, NormVariant.GROUP_NORM)
    _io = torch_dtype_to_str(x.dtype)
    if not (_nhwc_native and _is_nhwc(x) and _is_nhwc(dy) and _nhwc_kernel_serves(variant, x.shape, _io, num_groups)):
        x = x.contiguous()
        dy = dy.contiguous()
    gamma, mean, rstd = _flat(gamma), _flat(mean), _flat(rstd)
    _require_io_dtype(x, gamma=gamma, dy=dy)
    io = torch_dtype_to_str(x.dtype)

    # dx is viewed with the same expression as x in every route below, so it is
    # usable exactly when it is laid out like the (already normalised) x.
    out_dx, out_dgamma, out_dbeta = out if out is not None else (None, None, None)
    # ``w_dx`` is what the kernels may write IN PLACE; ``out_dx`` stays the caller's
    # buffer so _deliver can still copy into it when they cannot.
    w_dx = out_dx
    if w_dx is not None and (tuple(w_dx.shape) != tuple(x.shape) or w_dx.stride() != x.stride() or w_dx.dtype != x.dtype):
        w_dx = None

    if variant in ROWWISE_VARIANTS:
        spec = rowwise_spec(variant, x.shape, normalized_shape=normalized_shape, num_groups=num_groups)
        params = TemplateParams(variant=variant, io_dtype=io, has_beta=has_beta)
        cfg = make_cfg(params, spec.M, staged_rows=2)  # backward stages X and DY
        # GroupNorm, channels-last: a group is cpg ADJACENT channels, contiguous within
        # a pixel in NHWC, so the channel tile is the group itself.
        if variant == NormVariant.GROUP_NORM and _is_nhwc(x):
            N, C, H, W = (int(v) for v in x.shape)
            if groupnorm_nhwc_sm100.eligible(C, int(spec.channels_per_group), DTYPE_BYTES[io]):
                x3 = x.permute(0, 2, 3, 1).reshape(N, H * W, C)
                dy3 = dy.permute(0, 2, 3, 1).reshape(N, H * W, C)
                odx3 = w_dx.permute(0, 2, 3, 1).reshape(N, H * W, C) if w_dx is not None else None
                dx3, dgamma, dbeta = groupnorm_nhwc_sm100.backward(
                    spec,
                    dy3,
                    x3,
                    gamma,
                    mean,
                    rstd,
                    has_beta=has_beta,
                    cfg=cfg,
                    params=params,
                    out_dx=odx3,
                    ws=ws,
                    out_dgamma=_stat_out(out_dgamma, spec.gamma_len),
                    out_dbeta=_stat_out(out_dbeta, spec.gamma_len),
                )
                return _deliver(dx3.reshape(N, H, W, C).permute(0, 3, 1, 2), dgamma, dbeta, out_dx, out_dgamma, out_dbeta)
        # InstanceNorm, channels-last: an IN group is strided by C in NHWC, so the
        # rowwise view does not exist -- use the per-image NHWC kernel instead of
        # transposing.
        if variant == NormVariant.INSTANCE_NORM and _is_nhwc(x):
            N, C, H, W = (int(v) for v in x.shape)
            x3 = x.permute(0, 2, 3, 1).reshape(N, H * W, C)
            dy3 = dy.permute(0, 2, 3, 1).reshape(N, H * W, C)
            if instancenorm_nhwc_sm100.nhwc_cfg(C, DTYPE_BYTES[io]) is not None:
                odx3 = w_dx.permute(0, 2, 3, 1).reshape(N, H * W, C) if w_dx is not None else None
                dx3, dgamma, dbeta = instancenorm_nhwc_sm100.backward(
                    spec,
                    dy3,
                    x3,
                    gamma,
                    mean,
                    rstd,
                    has_beta=has_beta,
                    cfg=cfg,
                    params=params,
                    out_dx=odx3,
                    ws=ws,
                    out_dgamma=_stat_out(out_dgamma, spec.gamma_len),
                    out_dbeta=_stat_out(out_dbeta, spec.gamma_len),
                )
                return _deliver(dx3.reshape(N, H, W, C).permute(0, 3, 1, 2), dgamma, dbeta, out_dx, out_dgamma, out_dbeta)
        x2d = x.reshape(spec.R, spec.M)
        dy2d = dy.reshape(spec.R, spec.M)
        odx = w_dx.reshape(spec.R, spec.M) if w_dx is not None else None
        odg, odb = _stat_out(out_dgamma, spec.gamma_len), _stat_out(out_dbeta, spec.gamma_len)
        # GN/IN: the atomic-free map (channel set fixed per CTA) when it applies.
        # InstanceNorm with short rows: a WARP per row, so the row reduction is a
        # shuffle rather than a block reduce through shared memory.
        if variant == NormVariant.INSTANCE_NORM and instancenorm_warp_sm100.eligible(spec, DTYPE_BYTES[io]):
            dx, dgamma, dbeta = instancenorm_warp_sm100.backward(
                spec, dy2d, x2d, gamma, mean, rstd, has_beta=has_beta, cfg=cfg, params=params, out_dx=odx, out_dgamma=odg, out_dbeta=odb, ws=ws
            )
            return _deliver(dx.reshape(x.shape), dgamma, dbeta, out_dx, out_dgamma, out_dbeta)
        # Long rows and few rows: groupnorm_fast_sm100 declines when the row will not
        # fit its shared-memory stage, and caps its grid at R. Both dropped through to
        # the original one-atomic-per-element kernel at 0.00-0.01 of achievable.
        if variant in (NormVariant.GROUP_NORM, NormVariant.INSTANCE_NORM):
            from .kernels import groupnorm_cga_sm100

            if groupnorm_cga_sm100.should_use(spec, DTYPE_BYTES[io]):
                dx, dgamma, dbeta = groupnorm_cga_sm100.backward(
                    spec, dy2d, x2d, gamma, mean, rstd, has_beta=has_beta, params=params, out_dx=odx, out_dgamma=odg, out_dbeta=odb, ws=ws
                )
                return _deliver(dx.reshape(x.shape), dgamma, dbeta, out_dx, out_dgamma, out_dbeta)
        if variant in (NormVariant.GROUP_NORM, NormVariant.INSTANCE_NORM) and groupnorm_fast_sm100.eligible(spec, DTYPE_BYTES[io]):
            dx, dgamma, dbeta = groupnorm_fast_sm100.backward(
                spec, dy2d, x2d, gamma, mean, rstd, has_beta=has_beta, cfg=cfg, params=params, out_dx=odx, out_dgamma=odg, out_dbeta=odb, ws=ws
            )
            return _deliver(dx.reshape(x.shape), dgamma, dbeta, out_dx, out_dgamma, out_dbeta)
        # LN/RMS with a long row: split it across a CGA and reduce c1/c2 through
        # distributed shared memory, exactly as the forward does. Only past the
        # measured crossover -- the pipelined kernel below is better for short rows.
        if variant in (NormVariant.LAYER_NORM, NormVariant.RMS_NORM):
            from .kernels import layernorm_cga_sm100

            if layernorm_cga_sm100.should_use(spec.M, spec.R, DTYPE_BYTES[io]):
                dx, dgamma, dbeta = layernorm_cga_sm100.backward(
                    spec, dy2d, x2d, gamma, mean, rstd, has_beta=has_beta, params=params, out_dx=odx, out_dgamma=odg, out_dbeta=odb, ws=ws
                )
                return _deliver(dx.reshape(x.shape), dgamma, dbeta, out_dx, out_dgamma, out_dbeta)
        dx, dgamma, dbeta = _ROWWISE_KERNEL[variant].backward(
            spec, dy2d, x2d, gamma, mean, rstd, has_beta=has_beta, cfg=cfg, params=params, out_dx=odx, out_dgamma=odg, out_dbeta=odb, ws=ws
        )
        return _deliver(dx.reshape(x.shape), dgamma, dbeta, out_dx, out_dgamma, out_dbeta)

    if variant == NormVariant.BATCH_NORM:
        spec = batchnorm_spec(x.shape)
        params = TemplateParams(variant=variant, io_dtype=io, has_beta=has_beta)
        eb = DTYPE_BYTES[io]
        cfg = Cfg(block_threads=choose_block_threads(spec.count), V=vector_width(eb), stage_mode=STAGE_NONE, vec=False, elem_bytes=eb)
        x3d = x.reshape(spec.N, spec.C, spec.S)
        dy3d = dy.reshape(spec.N, spec.C, spec.S)
        odx3d = w_dx.reshape(spec.N, spec.C, spec.S) if w_dx is not None else None
        odg, odb = _stat_out(out_dgamma, spec.C), _stat_out(out_dbeta, spec.C)
        # NHWC: the forward's fused cooperative split-K with both streams cached.
        if _is_nhwc(x) and batchnorm_nhwc_sm100.nhwc_cfg(spec.C, eb) is not None:
            N, C, H, W = (int(v) for v in x.shape)
            x2d = x.permute(0, 2, 3, 1).reshape(N * H * W, C)
            dy2d = dy.permute(0, 2, 3, 1).reshape(N * H * W, C)
            odx2 = w_dx.permute(0, 2, 3, 1).reshape(N * H * W, C) if w_dx is not None else None
            dx2, dgamma, dbeta = batchnorm_nhwc_sm100.backward(
                spec,
                dy2d,
                x2d,
                gamma,
                mean,
                rstd,
                has_beta=has_beta,
                cfg=cfg,
                params=params,
                out_dx=odx2,
                ws=ws,
                out_dgamma=_stat_out(out_dgamma, spec.C),
                out_dbeta=_stat_out(out_dbeta, spec.C),
            )
            return _deliver(dx2.reshape(N, H, W, C).permute(0, 3, 1, 2), dgamma, dbeta, out_dx, out_dgamma, out_dbeta)
        # NCHW: split-K over the batch, fixed channels per warp (no atomics).
        if batchnorm_nchw_sm100.nchw_cfg(spec.C, spec.N, spec.S, eb) is not None:
            dx, dgamma, dbeta = batchnorm_nchw_sm100.backward(
                spec, dy3d, x3d, gamma, mean, rstd, has_beta=has_beta, cfg=cfg, params=params, out_dx=odx3d, out_dgamma=odg, out_dbeta=odb, ws=ws
            )
            return _deliver(dx.reshape(x.shape), dgamma, dbeta, out_dx, out_dgamma, out_dbeta)
        dx, dgamma, dbeta = batchnorm_sm100.backward(
            spec, dy3d, x3d, gamma, mean, rstd, has_beta=has_beta, cfg=cfg, params=params, out_dx=odx3d, out_dgamma=odg, out_dbeta=odb, ws=ws
        )
        return _deliver(dx.reshape(x.shape), dgamma, dbeta, out_dx, out_dgamma, out_dbeta)

    raise ValueError(f"unknown norm variant {variant}")
