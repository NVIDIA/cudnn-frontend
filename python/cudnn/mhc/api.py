# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic Cake four-stream mHC training stages."""

import math
from functools import lru_cache

import torch

from cudnn._buffer_utils import check_buffers_disjoint
from cudnn._torch_stream import launch_stream, stream_context
from cudnn._tupledict import TupleDict


@lru_cache(maxsize=2)
def _backend(head=False):
    from cudnn._cake import loader

    return loader.load_package("cake_mhc_head_det_train" if head else "cake_mhc_det_train")


def _tensor(name, value, shape, dtype, device):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape) or value.dtype != dtype or value.device != device:
        raise ValueError(f"{name} must have shape {tuple(shape)}, dtype {dtype}, on {device}")
    if not value.is_contiguous() or value.data_ptr() % 32:
        raise ValueError(f"{name} must be contiguous and 32-byte aligned")


def _input(x, ndim=3):
    if not isinstance(x, torch.Tensor) or not x.is_cuda or x.ndim != ndim or (ndim == 3 and x.shape[1] != 4):
        raise ValueError("mHC requires CUDA x[T,4,H] (or grad_out[T,H] for head backward)")
    t, h = int(x.shape[0]), int(x.shape[-1])
    if t < 1 or h not in (4096, 5120, 7168) or x.numel() > 2**31 - 1:
        raise NotImplementedError("mHC requires positive rows, H=4096/5120/7168, and int32 element offsets")
    _tensor("x", x, x.shape, torch.bfloat16, x.device)
    from cudnn._cake import loader

    arch = loader.device_arch(x.device.index)
    loader.check_toolkit(arch, "mHC")
    if arch == "sm_107a" and h != 4096:
        raise NotImplementedError("SM107 mHC requires H=4096")
    return t, h, arch


def _constants(rms_eps, hc_eps, sinkhorn_iters):
    for name, value in (("rms_eps", rms_eps), ("hc_eps", hc_eps)):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive host scalar")
    if isinstance(sinkhorn_iters, bool) or not isinstance(sinkhorn_iters, int) or not 1 <= sinkhorn_iters <= 64:
        raise ValueError("sinkhorn_iters must be a host integer in [1,64]")


def mhc_workspace_size(rows, hidden, *, backward=False, device=None):
    """Bytes for prepare or backward-finish scratch."""
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise ValueError("rows must be a positive host integer")
    device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
    return _backend()._workspace_bytes("bwd" if backward else "fwd", rows, device, hidden)


def _output(value, shape, dtype, device):
    return torch.empty(shape, dtype=dtype, device=device) if value is None else value


def _workspace(value, kind, t, h, device):
    b = _backend()
    value = _output(value, (b._workspace_bytes(kind, t, device, h),), torch.uint8, device)
    views = b._workspace_views(value, kind, rows=t, device=device, hidden=h)
    return value, views


def _launch(kind, x, values, reads, writes, t, h, arch, stream, *, workspace=None, sinkhorn_iters=20):
    b = _backend()
    path = b._PATH_FORWARD if kind.startswith("fwd") else b._PATH_BACKWARD
    check_buffers_disjoint(writes, reads)
    # Resolve every native stage before the first ticket reset or kernel launch.
    b._route_plan(arch, kind, t, h, x.device.index, path, sinkhorn_iters)
    if workspace is not None:
        workspace[:256].zero_()
    b._run_route(arch, kind, values, rows=t, hidden=h, device=x.device, path=path, sinkhorn_iters=sinkhorn_iters)
    launch = launch_stream(stream, x.device)
    for _, tensor in reads + writes:
        tensor.record_stream(launch)


def mhc_prepare(x, fn, scale, base, *, rms_eps=1e-5, hc_eps=1e-6, sinkhorn_iters=20, out=None, stats=None, workspace=None, stream=None):
    """Return collapsed BF16 branch input and FP32 saved statistics."""
    t, h, arch = _input(x)
    _constants(rms_eps, hc_eps, sinkhorn_iters)
    b = _backend()
    for name, value, shape in (("fn", fn, (24, 4 * h)), ("scale", scale, (3,)), ("base", base, (24,))):
        _tensor(name, value, shape, torch.float32, x.device)
    with torch.cuda.device(x.device), stream_context(stream, x.device):
        out = _output(out, (t, h), torch.bfloat16, x.device)
        stats = _output(stats, (t, 64), torch.float32, x.device)
        _tensor("out", out, (t, h), torch.bfloat16, x.device)
        _tensor("stats", stats, (t, 64), torch.float32, x.device)
        workspace, ws = _workspace(workspace, "fwd", t, h, x.device)
        values = b._forward_values(
            x, fn, scale, base, stats, ws, rows=t, hidden=h, device=x.device, rms_eps=rms_eps, hc_eps=hc_eps, sinkhorn_iters=sinkhorn_iters
        )
        values["u"] = out
        _launch(
            "fwd_prepare",
            x,
            values,
            [("x", x), ("fn", fn), ("scale", scale), ("base", base)],
            [("out", out), ("stats", stats), ("workspace", workspace)],
            t,
            h,
            arch,
            stream,
            workspace=workspace,
            sinkhorn_iters=sinkhorn_iters,
        )
    return TupleDict(out=out, stats=stats)


def mhc_finish(x, branch, stats, *, out=None, stream=None):
    """Mix residual streams with the external branch output."""
    t, h, arch = _input(x)
    _tensor("branch", branch, (t, h), torch.bfloat16, x.device)
    _tensor("stats", stats, (t, 64), torch.float32, x.device)
    with torch.cuda.device(x.device), stream_context(stream, x.device):
        out = _output(out, x.shape, torch.bfloat16, x.device)
        _tensor("out", out, x.shape, torch.bfloat16, x.device)
        _launch(
            "fwd_finish",
            x,
            dict(x=x, b=branch, stats=stats, y=out, rows=t),
            [("x", x), ("branch", branch), ("stats", stats)],
            [("out", out)],
            t,
            h,
            arch,
            stream,
        )
    return TupleDict(out=out)


def mhc_backward_prepare(x, branch, stats, grad_out, *, grad_branch=None, grad_stats=None, stream=None):
    """Return the branch-output gradient and saved coefficient gradients."""
    t, h, arch = _input(x)
    for name, value, shape, dtype in (
        ("branch", branch, (t, h), torch.bfloat16),
        ("stats", stats, (t, 64), torch.float32),
        ("grad_out", grad_out, x.shape, torch.bfloat16),
    ):
        _tensor(name, value, shape, dtype, x.device)
    with torch.cuda.device(x.device), stream_context(stream, x.device):
        grad_branch = _output(grad_branch, (t, h), torch.bfloat16, x.device)
        grad_stats = _output(grad_stats, (t, 32), torch.float32, x.device)
        _tensor("grad_branch", grad_branch, (t, h), torch.bfloat16, x.device)
        _tensor("grad_stats", grad_stats, (t, 32), torch.float32, x.device)
        _launch(
            "bwd_prepare",
            x,
            dict(x=x, b=branch, stats=stats, dy=grad_out, db=grad_branch, gstats=grad_stats, rows=t),
            [("x", x), ("branch", branch), ("stats", stats), ("grad_out", grad_out)],
            [("grad_branch", grad_branch), ("grad_stats", grad_stats)],
            t,
            h,
            arch,
            stream,
        )
    return TupleDict(grad_branch=grad_branch, grad_stats=grad_stats)


def mhc_backward_finish(
    x,
    grad_out,
    grad_u,
    fn,
    scale,
    base,
    stats,
    grad_stats,
    *,
    rms_eps=1e-5,
    hc_eps=1e-6,
    sinkhorn_iters=20,
    dx=None,
    dfn=None,
    dscale=None,
    dbase=None,
    workspace=None,
    stream=None,
):
    """Return BF16 dx and deterministic FP32 mapping/scale/base gradients."""
    t, h, arch = _input(x)
    _constants(rms_eps, hc_eps, sinkhorn_iters)
    b = _backend()
    for name, value, shape, dtype in (
        ("grad_out", grad_out, x.shape, torch.bfloat16),
        ("grad_u", grad_u, (t, h), torch.bfloat16),
        ("fn", fn, (24, 4 * h), torch.float32),
        ("scale", scale, (3,), torch.float32),
        ("base", base, (24,), torch.float32),
        ("stats", stats, (t, 64), torch.float32),
        ("grad_stats", grad_stats, (t, 32), torch.float32),
    ):
        _tensor(name, value, shape, dtype, x.device)
    with torch.cuda.device(x.device), stream_context(stream, x.device):
        dx = _output(dx, x.shape, torch.bfloat16, x.device)
        dfn = _output(dfn, fn.shape, torch.float32, x.device)
        dscale = _output(dscale, scale.shape, torch.float32, x.device)
        dbase = _output(dbase, base.shape, torch.float32, x.device)
        for name, value, shape, dtype in (
            ("dx", dx, x.shape, torch.bfloat16),
            ("dfn", dfn, fn.shape, torch.float32),
            ("dscale", dscale, scale.shape, torch.float32),
            ("dbase", dbase, base.shape, torch.float32),
        ):
            _tensor(name, value, shape, dtype, x.device)
        workspace, ws = _workspace(workspace, "bwd", t, h, x.device)
        plan = b._loader.host_plan(t, b._loader.sm_count(x.device.index), arch, h)
        values = dict(
            fn=fn,
            fnTp=ws["fnTp"],
            fnp=ws["fnTp"],
            x=x,
            du=grad_u,
            stats=stats,
            gstats=grad_stats,
            scale=scale,
            base=base,
            dzp=ws["dzp"],
            tokp=ws["tokp"],
            param_partial=ws["param_partial"],
            rows=t,
            t_pad=plan["t_pad"],
            rows_per_cta=plan["rows_per_cta"],
            hc_eps=float(hc_eps),
            sinkhorn_iters=sinkhorn_iters,
            x_flat=x.view(t, 4 * h),
            dy_flat=grad_out.view(t, 4 * h),
            dzp_rows=ws["dzp_rows"],
            dx_flat=dx.view(t, 4 * h),
            dfn_part=ws["dfn_part"],
            ticket=ws["ticket"],
            parts=plan["parts"],
            rows_per_chunk=plan["rows_per_chunk"],
            dfn=dfn,
            dscale=dscale,
            dbase=dbase,
            chunks=plan["chunks"],
            groups=plan["groups"],
            accumulate=0,
        )
        _launch(
            "bwd_finish",
            x,
            values,
            [("x", x), ("grad_out", grad_out), ("grad_u", grad_u), ("fn", fn), ("scale", scale), ("base", base), ("stats", stats), ("grad_stats", grad_stats)],
            [("dx", dx), ("dfn", dfn), ("dscale", dscale), ("dbase", dbase), ("workspace", workspace)],
            t,
            h,
            arch,
            stream,
            workspace=workspace,
            sinkhorn_iters=sinkhorn_iters,
        )
    return TupleDict(dx=dx, dfn=dfn, dscale=dscale, dbase=dbase)


def _head(x, out, backward, stream):
    t, h, arch = _input(x, 2 if backward else 3)
    b = _backend(True)
    shape = (t, 4, h) if backward else (t, h)
    with torch.cuda.device(x.device), stream_context(stream, x.device):
        out = _output(out, shape, torch.bfloat16, x.device)
        _tensor("out", out, shape, torch.bfloat16, x.device)
        check_buffers_disjoint([("out", out)], [("input", x)])
        b._run("head_bwd" if backward else "head_fwd", arch, dict(dy=x, dx=out) if backward else dict(x=x, y=out), rows=t, hidden=h, device=x.device)
        launch = launch_stream(stream, x.device)
        x.record_stream(launch)
        out.record_stream(launch)
    return TupleDict(out=out)


def mhc_head_forward(x, *, out=None, stream=None):
    """Mean the four BF16 residual streams in FP32, then round once."""
    return _head(x, out, False, stream)


def mhc_head_backward(grad_out, *, out=None, stream=None):
    """Broadcast BF16(grad_out / 4) to each of four streams."""
    return _head(grad_out, out, True, stream)


__all__ = ["mhc_workspace_size", "mhc_prepare", "mhc_finish", "mhc_backward_prepare", "mhc_backward_finish", "mhc_head_forward", "mhc_head_backward"]
