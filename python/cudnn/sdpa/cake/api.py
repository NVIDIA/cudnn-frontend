# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packed BF16 causal GQA with caller-owned output and scratch support."""

from __future__ import annotations

import math
import warnings
from functools import lru_cache

import torch

from cudnn._buffer_utils import check_buffers_disjoint
from cudnn._torch_stream import launch_stream, stream_context
from cudnn._tupledict import TupleDict


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= 2**31 - 1:
        raise ValueError(f"{name} must be a positive int32 host integer")
    return value


def gqa_attention_workspace_size(total_q, num_heads_q, num_segments, *, backward=False):
    """Scratch bytes for packed GQA; the forward uses a four-byte tile counter."""
    for name, value in (("total_q", total_q), ("num_heads_q", num_heads_q), ("num_segments", num_segments)):
        _positive_int(name, value)
    if num_heads_q % 2:
        raise ValueError("num_heads_q must be even")
    if not backward:
        return 4
    rows = ((2 * total_q + 128 * num_segments + 127) // 128) * 128
    return (num_heads_q // 2) * rows * (256 + 2) * 4


def _tensor(name, value, shape, dtype, device, alignment=16):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != tuple(shape) or value.dtype != dtype or value.device != device:
        raise ValueError(f"{name} must have shape {tuple(shape)}, dtype {dtype}, and device {device}")
    if not value.is_contiguous() or value.data_ptr() % alignment:
        raise ValueError(f"{name} must be contiguous and {alignment}-byte aligned")


def _inputs(q, k, v, cu_q, cu_k, scale):
    if not isinstance(q, torch.Tensor) or q.ndim != 3 or not q.is_cuda:
        raise ValueError("q must be a CUDA tensor with shape [Tq, Hq, 256]")
    if not isinstance(k, torch.Tensor) or k.ndim != 3:
        raise ValueError("k must have shape [Tk, Hkv, 256]")
    tq, hq, dim = q.shape
    tk, hkv, _ = k.shape
    if tq < 1 or tk < 1 or hq < 1 or hkv < 1 or dim != 256 or hq % hkv or (hq // hkv) % 2:
        raise NotImplementedError("GQA requires positive token/head counts, D=256, and an even Hq/Hkv ratio")
    if max(q.numel(), k.numel()) > 2**31 - 1:
        raise NotImplementedError("GQA operands must fit int32 element offsets")
    for name, value, shape in (("q", q, (tq, hq, 256)), ("k", k, (tk, hkv, 256)), ("v", v, (tk, hkv, 256))):
        _tensor(name, value, shape, torch.bfloat16, q.device)
    if not isinstance(cu_q, torch.Tensor) or cu_q.ndim != 1 or cu_q.numel() < 2:
        raise ValueError("cu_seqlens_q must have shape [num_segments + 1], with at least one segment")
    for name, value in (("cu_seqlens_q", cu_q), ("cu_seqlens_k", cu_k)):
        _tensor(name, value, cu_q.shape, torch.int32, q.device, 4)
    segments = cu_q.numel() - 1
    if max(segments, hkv) > 65535:
        raise NotImplementedError("GQA requires num_segments and Hkv <= 65535")
    scale = 256**-0.5 if scale is None else scale
    if isinstance(scale, bool) or not isinstance(scale, (float, int)) or not math.isfinite(scale):
        raise ValueError("softmax_scale must be a finite host scalar")
    return dict(total_q=tq, total_k=tk, num_heads_q=hq, num_heads_kv=hkv, num_segments=segments), float(scale)


def _scratch(buffer, size, device):
    if not isinstance(buffer, torch.Tensor) or buffer.ndim != 1 or buffer.numel() < size:
        raise ValueError(f"workspace must be a uint8 vector with at least {size} bytes")
    _tensor("workspace", buffer, buffer.shape, torch.uint8, device, 256)


def _nondeterminism():
    if torch.are_deterministic_algorithms_enabled():
        message = "GQA backward has no deterministic implementation: dQ uses FP32 atomic additions"
        if torch.is_deterministic_algorithms_warn_only_enabled():
            warnings.warn(message, UserWarning, stacklevel=3)
        else:
            raise RuntimeError(message)


class _Plan:
    def __init__(self, device, backward):
        from cudnn._cake import loader

        loader.load_package("cake_gqa_nondet_train")
        from cudnn._cake._vendor.cake_gqa_nondet_train import cake_backend, cake_jit

        arch = loader.device_arch(device)
        loader.check_toolkit(arch, "GQA")
        self.device = torch.device("cuda", device)
        self.backward = backward
        self.jit = cake_jit
        self.sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        self.entries = {}
        for stage in (("bwd_pre", "bwd_bf16", "bwd_post") if backward else ("fwd",)):
            record = cake_backend.stage_record(stage, arch)
            self.entries[stage] = (cake_backend.load(record, arch).run, cake_backend.parameters(record))

    def _launch(self, stage, values, dims):
        import tvm_ffi

        entry, names = self.entries[stage]
        grid = self.jit.grid(stage, sm_count=self.sm_count, **dims)
        with tvm_ffi.use_torch_stream():
            entry(*(values[name] for name in names), *grid)

    def forward(self, q, k, v, cu_q, cu_k, out, lse, workspace, scale, stream):
        dims, scale = _inputs(q, k, v, cu_q, cu_k, scale)
        if q.device != self.device:
            raise ValueError(f"plan belongs to {self.device}, got {q.device}")
        _tensor("out", out, q.shape, torch.bfloat16, q.device)
        _tensor("lse", lse, (q.shape[1], q.shape[0]), torch.float32, q.device)
        _scratch(workspace, 4, q.device)
        check_buffers_disjoint(
            (("out", out), ("lse", lse), ("workspace", workspace)),
            (("q", q), ("k", k), ("v", v), ("cu_seqlens_q", cu_q), ("cu_seqlens_k", cu_k)),
        )
        with torch.cuda.device(self.device), stream_context(stream, self.device):
            counter = workspace[:4].view(torch.uint32)
            counter.zero_()
            self._launch(
                "fwd",
                dict(
                    Q=q,
                    K=k,
                    V=v,
                    O=out,
                    LSE=lse,
                    cu_seqlens_q=cu_q,
                    cu_seqlens_k=cu_k,
                    tile_counter=counter,
                    num_segments=dims["num_segments"],
                    num_heads_q=dims["num_heads_q"],
                    group=dims["num_heads_q"] // dims["num_heads_kv"],
                    total_q=dims["total_q"],
                    softmax_scale_log2=scale * math.log2(math.e),
                ),
                dims,
            )

    def gradients(self, q, k, v, out, lse, grad_out, cu_q, cu_k, dq, dk, dv, workspace, scale, max_seqlen_k, stream):
        dims, scale = _inputs(q, k, v, cu_q, cu_k, scale)
        if q.device != self.device:
            raise ValueError(f"plan belongs to {self.device}, got {q.device}")
        for name, value, shape, dtype in (
            ("out", out, q.shape, torch.bfloat16),
            ("grad_out", grad_out, q.shape, torch.bfloat16),
            ("lse", lse, (q.shape[1], q.shape[0]), torch.float32),
            ("dq", dq, q.shape, torch.bfloat16),
            ("dk", dk, k.shape, torch.bfloat16),
            ("dv", dv, k.shape, torch.bfloat16),
        ):
            _tensor(name, value, shape, dtype, q.device)
        if max_seqlen_k is not None:
            _positive_int("max_seqlen_k", max_seqlen_k)
        size = gqa_attention_workspace_size(dims["total_q"], dims["num_heads_q"], dims["num_segments"], backward=True)
        _scratch(workspace, size, q.device)
        check_buffers_disjoint(
            (("dq", dq), ("dk", dk), ("dv", dv), ("workspace", workspace)),
            (("q", q), ("k", k), ("v", v), ("out", out), ("lse", lse), ("grad_out", grad_out), ("cu_seqlens_q", cu_q), ("cu_seqlens_k", cu_k)),
        )
        with torch.cuda.device(self.device), stream_context(stream, self.device):
            regions = {
                name: workspace[offset : offset + 4 * math.prod(shape)].view(torch.float32).view(shape)
                for name, (offset, shape) in self.jit.scratch_layout(dims["total_q"], dims["num_heads_q"], dims["num_segments"]).items()
            }
            common = dict(
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                num_segments=dims["num_segments"],
                rows_pad=self.jit.rows_pad(dims["total_q"], dims["num_segments"]),
                T_q=dims["total_q"],
                H_q=dims["num_heads_q"],
                softmax_scale=scale,
            )
            self._launch(
                "bwd_pre", dict(common, O=out, dO=grad_out, LSE=lse, LSE_log2=regions["lse_log2"], dPsum=regions["dpsum"], dQ_accum=regions["dq_accum"]), dims
            )
            self._launch(
                "bwd_bf16",
                dict(
                    common,
                    Q=q,
                    dO=grad_out,
                    K=k,
                    V=v,
                    **regions,
                    dk_out=dk,
                    dv_out=dv,
                    h_kv=dims["num_heads_kv"],
                    group=dims["num_heads_q"] // dims["num_heads_kv"],
                    softmax_scale_log2=scale * math.log2(math.e),
                    row_map=None,
                ),
                dict(dims, max_seqlen_k=max_seqlen_k),
            )
            self._launch("bwd_post", dict(common, dQ_accum=regions["dq_accum"], dQ=dq), dims)


@lru_cache(maxsize=32)
def _plan(device, backward):
    return _Plan(device, backward)


def gqa_attention_forward(q, k, v, cu_seqlens_q, cu_seqlens_k, *, softmax_scale=None, out=None, lse=None, workspace=None, stream=None):
    """Return ``(out, lse)`` for packed bottom-right causal GQA, D=256.

    Supply every output and workspace to avoid allocation after warmup.
    Prefix values must describe the packed operands; they stay on device.
    """
    _inputs(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale)
    plan = _plan(q.device.index, False)
    with torch.cuda.device(q.device), stream_context(stream, q.device):
        out = torch.empty_like(q) if out is None else out
        lse = torch.empty((q.shape[1], q.shape[0]), dtype=torch.float32, device=q.device) if lse is None else lse
        workspace = torch.empty(4, dtype=torch.uint8, device=q.device) if workspace is None else workspace
        plan.forward(q, k, v, cu_seqlens_q, cu_seqlens_k, out, lse, workspace, softmax_scale, stream)
        for value in (q, k, v, cu_seqlens_q, cu_seqlens_k, out, lse, workspace):
            value.record_stream(launch_stream(stream, q.device))
    return TupleDict(out=out, lse=lse)


def gqa_attention_backward(
    q, k, v, out, lse, grad_out, cu_seqlens_q, cu_seqlens_k, *, softmax_scale=None, max_seqlen_k=None, dq=None, dk=None, dv=None, workspace=None, stream=None
):
    """Return ``(dq, dk, dv)``. The dQ reduction is nondeterministic.

    ``max_seqlen_k`` must bound every key segment. Omitting it uses Tk,
    which is always a safe bound and requires no host read.
    """
    dims, _ = _inputs(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale)
    if max_seqlen_k is not None:
        _positive_int("max_seqlen_k", max_seqlen_k)
    _nondeterminism()
    plan = _plan(q.device.index, True)
    with torch.cuda.device(q.device), stream_context(stream, q.device):
        dq = torch.empty_like(q) if dq is None else dq
        dk = torch.empty_like(k) if dk is None else dk
        dv = torch.empty_like(v) if dv is None else dv
        if workspace is None:
            size = gqa_attention_workspace_size(dims["total_q"], dims["num_heads_q"], dims["num_segments"], backward=True)
            workspace = torch.empty(size, dtype=torch.uint8, device=q.device)
        plan.gradients(q, k, v, out, lse, grad_out, cu_seqlens_q, cu_seqlens_k, dq, dk, dv, workspace, softmax_scale, max_seqlen_k, stream)
        for value in (q, k, v, out, lse, grad_out, cu_seqlens_q, cu_seqlens_k, dq, dk, dv, workspace):
            value.record_stream(launch_stream(stream, q.device))
    return TupleDict(dq=dq, dk=dk, dv=dv)


__all__ = ["gqa_attention_forward", "gqa_attention_backward", "gqa_attention_workspace_size"]
