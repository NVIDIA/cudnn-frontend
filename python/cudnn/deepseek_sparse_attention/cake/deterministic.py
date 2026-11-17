# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic sparse attention and Lightning indexer native stages."""

import json
import math
from functools import lru_cache
from pathlib import Path

import torch

from cudnn._buffer_utils import check_buffers_disjoint
from cudnn._torch_stream import launch_stream, stream_context
from cudnn._tupledict import TupleDict


@lru_cache(maxsize=1)
def _package():
    from cudnn._cake import loader

    package = loader.load_package("cake_dsa_sparse_det")
    from cudnn._cake._vendor.cake_dsa_sparse_det import cake_build

    return cake_build, json.loads((Path(package.__file__).parent / "registry.json").read_text())


@lru_cache(maxsize=128)
def _entry(stage, device):
    from cudnn._cake import loader

    arch = loader.device_arch(device)
    loader.check_toolkit(arch, "deterministic DSA")
    build, records = _package()
    record = records[stage]
    entry = getattr(build.load_library(f"{record['name']}_{arch}"), record["ffi_entry"])
    return entry, record["arg_plan"]


def _call(stage, device, grid, **values):
    import tvm_ffi

    entry, args = _entry(stage, device)
    values.update(zip(("grid_x", "grid_y", "grid_z"), grid))
    args = [values[name] for _, name in args]
    with tvm_ffi.use_torch_stream():
        entry(*(v.detach() if isinstance(v, torch.Tensor) and v.requires_grad else v for v in args))


def _tensor(name, tensor, shape, dtype, device):
    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype or tensor.device != device:
        raise ValueError(f"{name} must have shape {tuple(shape)}, dtype {dtype}, on {device}")
    if not tensor.is_contiguous() or tensor.data_ptr() % 16:
        raise ValueError(f"{name} must be contiguous and 16-byte aligned")


def _inputs(q, kv, indices=None, *, indexer=False):
    if not isinstance(q, torch.Tensor) or q.ndim != 3 or not q.is_cuda:
        raise ValueError("q must be a CUDA tensor with shape [T,H,D]")
    t, h, d = q.shape
    if t < 1 or h not in (8, 16, 32, 64) or d not in ((128,) if indexer else (512, 576)):
        raise NotImplementedError("DSA requires positive rows, H in {8,16,32,64}, and D=128 for indexer or D=512/576 for attention")
    if not isinstance(kv, torch.Tensor) or kv.ndim != 2 or kv.shape[0] < 1:
        raise ValueError("kv must have shape [Tk,D] with positive Tk")
    _tensor("q", q, (t, h, d), torch.bfloat16, q.device)
    _tensor("kv", kv, (kv.shape[0], d), torch.bfloat16, q.device)
    if max(q.numel(), kv.numel()) > 2**31 - 1:
        raise NotImplementedError("DSA operands must fit int32 element offsets")
    if indices is not None:
        multiple = (64 if h <= 16 else 32) if indexer else 64
        if indices.ndim != 2 or indices.shape[0] != t or indices.shape[1] < 1 or indices.shape[1] % multiple:
            raise ValueError(f"indices must have shape [T,K] with positive K divisible by {multiple}")
        _tensor("indices", indices, indices.shape, torch.int32, q.device)
        if indices.numel() > 2**31 - 1:
            raise NotImplementedError("index slots must fit int32 offsets")
    return t, h, d, kv.shape[0]


def dsa_prepare_segments(indices, num_keys):
    """Allocate stable key-group metadata outside execution/capture.

    Rebuild all three outputs whenever indices change. They describe the
    flattened indices in row-major order, with invalid keys excluded.
    """
    if isinstance(num_keys, bool) or not isinstance(num_keys, int) or not 0 < num_keys < 2**31:
        raise ValueError("num_keys must be a positive int32 host integer")
    if not isinstance(indices, torch.Tensor) or indices.ndim != 2 or indices.dtype != torch.int32 or not indices.is_contiguous() or indices.numel() > 2**31 - 1:
        raise ValueError("indices must be a contiguous int32 matrix with int32 slot offsets")
    if indices.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prepare segments before CUDA graph capture")
    keys = indices.reshape(-1)
    keys = torch.where((keys < 0) | (keys >= num_keys), -1, keys)
    keys, order = torch.sort(keys, stable=True)
    bounds = torch.searchsorted(keys, torch.arange(num_keys + 1, dtype=torch.int32, device=indices.device), out_int32=True)
    return TupleDict(order=order.to(torch.int32), seg_start=bounds[:-1], seg_count=bounds[1:] - bounds[:-1])


def _segments(order, starts, counts, indices, keys):
    for name, value, shape in (("order", order, (indices.numel(),)), ("seg_start", starts, (keys,)), ("seg_count", counts, (keys,))):
        _tensor(name, value, shape, torch.int32, indices.device)


def _layout(rows, heads, dim, keys, topk, *, indexer=False):
    sizes = [("partial", rows * topk * dim * 4)]
    if not indexer:
        sizes += [("delta", rows * heads * 4), ("dkv", keys * dim * 4)]
    layout = {}
    offset = 0
    for name, size in sizes:
        layout[name] = (offset, size)
        offset += (size + 255) // 256 * 256
    return layout, offset


def dsa_det_workspace_size(rows, heads, dim, num_keys, top_k, *, indexer=False):
    """Bytes for full-row deterministic backward; no query chunking."""
    for value in (rows, heads, dim, num_keys, top_k):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("workspace geometry must use positive host integers")
    return _layout(rows, heads, dim, num_keys, top_k, indexer=indexer)[1]


def _scratch(workspace, size, device):
    if not isinstance(workspace, torch.Tensor) or workspace.ndim != 1 or workspace.numel() < size:
        raise ValueError(f"workspace requires at least {size} bytes")
    _tensor("workspace", workspace, workspace.shape, torch.uint8, device)
    if workspace.data_ptr() % 256:
        raise ValueError("workspace must be 256-byte aligned")


def _region(workspace, layout, name, shape):
    start, size = layout[name]
    return workspace[start : start + size].view(torch.float32).view(shape)


def _record(stream, device, values):
    launch = launch_stream(stream, device)
    for _, tensor in values:
        tensor.record_stream(launch)


def _scale(value, dim):
    value = dim**-0.5 if value is None else value
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError("softmax_scale must be a finite positive host scalar")
    return float(value)


def dsa_det_forward(q, kv, indices, *, softmax_scale=None, out=None, lse=None, workspace=None, stream=None):
    """Return BF16 out and base-2 FP32 lse for flat sparse K=V attention."""
    t, h, d, k = _inputs(q, kv, indices)
    scale = _scale(softmax_scale, d)
    with torch.cuda.device(q.device), stream_context(stream, q.device):
        out = torch.empty((t, h, 512), dtype=torch.bfloat16, device=q.device) if out is None else out
        lse = torch.empty((t, h), dtype=torch.float32, device=q.device) if lse is None else lse
        workspace = torch.empty(256, dtype=torch.uint8, device=q.device) if workspace is None else workspace
        _tensor("out", out, (t, h, 512), torch.bfloat16, q.device)
        _tensor("lse", lse, (t, h), torch.float32, q.device)
        _scratch(workspace, 256, q.device)
        writes = [("out", out), ("lse", lse), ("workspace", workspace)]
        reads = [("q", q), ("kv", kv), ("indices", indices)]
        check_buffers_disjoint(writes, reads)
        block = 16 if h <= 16 else 32
        stage = f"dsa_attention_fwd_h{block}_d512t{d-512}"
        _entry(stage, q.device.index)
        sink = workspace[: h * 4].view(torch.float32)
        sink.fill_(-torch.inf)
        _call(
            stage,
            q.device.index,
            (t, (h + block - 1) // block, 1),
            q=q,
            kv=kv,
            sink=sink,
            indices=indices,
            out=out,
            lse=lse,
            heads=h,
            topk=indices.shape[1],
            scale_log2=scale * math.log2(math.e),
        )
        _record(stream, q.device, reads + writes)
    return TupleDict(out=out, lse=lse)


def dsa_det_backward(q, kv, indices, out, lse, grad_out, order, seg_start, seg_count, *, softmax_scale=None, dq=None, dkv=None, workspace=None, stream=None):
    """Deterministic gradients using stable segments prepared for these indices."""
    t, h, d, k = _inputs(q, kv, indices)
    scale = _scale(softmax_scale, d)
    topk = indices.shape[1]
    _segments(order, seg_start, seg_count, indices, k)
    for name, value, shape, dtype in (
        ("out", out, (t, h, 512), torch.bfloat16),
        ("grad_out", grad_out, (t, h, 512), torch.bfloat16),
        ("lse", lse, (t, h), torch.float32),
    ):
        _tensor(name, value, shape, dtype, q.device)
    layout, size = _layout(t, h, d, k, topk)
    with torch.cuda.device(q.device), stream_context(stream, q.device):
        dq = torch.empty_like(q) if dq is None else dq
        dkv = torch.empty_like(kv) if dkv is None else dkv
        workspace = torch.empty(size, dtype=torch.uint8, device=q.device) if workspace is None else workspace
        _tensor("dq", dq, q.shape, torch.bfloat16, q.device)
        _tensor("dkv", dkv, kv.shape, torch.bfloat16, q.device)
        _scratch(workspace, size, q.device)
        reads = [
            ("q", q),
            ("kv", kv),
            ("indices", indices),
            ("out", out),
            ("lse", lse),
            ("grad_out", grad_out),
            ("order", order),
            ("seg_start", seg_start),
            ("seg_count", seg_count),
        ]
        writes = [("dq", dq), ("dkv", dkv), ("workspace", workspace)]
        check_buffers_disjoint(writes, reads)
        block = 16 if h <= 16 else 32
        stage = f"dsa_attention_bwd_h{'32x2' if h==64 else block}_d512t{d-512}"
        for name in ("dsa_attention_delta", stage, "dsa_segmented_reduce_split4"):
            _entry(name, q.device.index)
        partial = _region(workspace, layout, "partial", (t * topk, d))
        delta = _region(workspace, layout, "delta", (t, h))
        dkv32 = _region(workspace, layout, "dkv", (k, d))
        dkv32.zero_()
        _call("dsa_attention_delta", q.device.index, (t, 1, 1), o=out, do=grad_out, delta=delta, heads=h)
        _call(
            stage,
            q.device.index,
            (2 * t if h == 64 else t, 1, 1),
            q=q,
            kv=kv,
            do=grad_out,
            indices=indices,
            lse=lse,
            delta=delta,
            dq=dq,
            partial=partial,
            heads=h,
            topk=topk,
            num_head_blocks=(h + block - 1) // block,
            sm_scale=scale,
            scale_log2=scale * math.log2(math.e),
        )
        _call(
            "dsa_segmented_reduce_split4",
            q.device.index,
            (k, 1, 1),
            partial=partial,
            order=order,
            seg_start=seg_start,
            seg_count=seg_count,
            out=dkv32,
            dim=d,
            num_out=k,
        )
        dkv.copy_(dkv32)
        _record(stream, q.device, reads + writes)
    return TupleDict(dq=dq, dkv=dkv)


def dsa_indexer_logits(q, k, weights, key_start, key_end, *, logits=None, stream=None):
    """FP32 weighted ReLU logits [Tq,Tk], with per-row [start,end) key ranges."""
    t, h, d, nk = _inputs(q, k, indexer=True)
    _tensor("weights", weights, (t, h), torch.float32, q.device)
    for name, value in (("key_start", key_start), ("key_end", key_end)):
        _tensor(name, value, (t,), torch.int32, q.device)
    with torch.cuda.device(q.device), stream_context(stream, q.device):
        logits = torch.empty((t, nk), dtype=torch.float32, device=q.device) if logits is None else logits
        _tensor("logits", logits, (t, nk), torch.float32, q.device)
        reads = [("q", q), ("k", k), ("weights", weights), ("key_start", key_start), ("key_end", key_end)]
        writes = [("logits", logits)]
        check_buffers_disjoint(writes, reads)
        stage = f"dsa_indexer_logits_h{h}"
        _entry(stage, q.device.index)
        _entry("dsa_indexer_clean", q.device.index)
        _call(
            stage,
            q.device.index,
            ((t + 64 // h - 1) // (64 // h), 1, 1),
            index_q=q,
            index_k=k,
            weights=weights,
            cu_ks=key_start,
            cu_ke=key_end,
            logits=logits,
            seq_len=t,
            seq_len_kv=nk,
        )
        _call("dsa_indexer_clean", q.device.index, (t, 1, 1), logits=logits, cu_ks=key_start, cu_ke=key_end, seq_len=t, seq_len_kv=nk)
        _record(stream, q.device, reads + writes)
    return TupleDict(logits=logits)


def dsa_indexer_backward(q, k, weights, indices, grad_scores, order, seg_start, seg_count, *, dq=None, dk=None, dweights=None, workspace=None, stream=None):
    """Deterministic selected-score gradients; dk and dweights are FP32."""
    t, h, d, nk = _inputs(q, k, indices, indexer=True)
    topk = indices.shape[1]
    _tensor("weights", weights, (t, h), torch.float32, q.device)
    _tensor("grad_scores", grad_scores, indices.shape, torch.float32, q.device)
    _segments(order, seg_start, seg_count, indices, nk)
    layout, size = _layout(t, h, d, nk, topk, indexer=True)
    with torch.cuda.device(q.device), stream_context(stream, q.device):
        dq = torch.empty_like(q) if dq is None else dq
        dk = torch.empty(k.shape, dtype=torch.float32, device=q.device) if dk is None else dk
        dweights = torch.empty_like(weights) if dweights is None else dweights
        workspace = torch.empty(size, dtype=torch.uint8, device=q.device) if workspace is None else workspace
        for name, value, shape, dtype in (
            ("dq", dq, q.shape, torch.bfloat16),
            ("dk", dk, k.shape, torch.float32),
            ("dweights", dweights, weights.shape, torch.float32),
        ):
            _tensor(name, value, shape, dtype, q.device)
        _scratch(workspace, size, q.device)
        reads = [
            ("q", q),
            ("k", k),
            ("weights", weights),
            ("indices", indices),
            ("grad_scores", grad_scores),
            ("order", order),
            ("seg_start", seg_start),
            ("seg_count", seg_count),
        ]
        writes = [("dq", dq), ("dk", dk), ("dweights", dweights), ("workspace", workspace)]
        check_buffers_disjoint(writes, reads)
        stage = f"dsa_indexer_bwd_h{max(16,h)}"
        _entry(stage, q.device.index)
        _entry("dsa_segmented_reduce_split4", q.device.index)
        partial = _region(workspace, layout, "partial", (t * topk, d))
        dk.zero_()
        _call(
            stage,
            q.device.index,
            (t, 1, 1),
            index_q=q,
            index_k=k,
            weights=weights,
            topk_indices=indices,
            grad_scores=grad_scores,
            d_index_q=dq,
            d_weights=dweights,
            partial=partial,
            heads=h,
            topk=topk,
        )
        _call(
            "dsa_segmented_reduce_split4",
            q.device.index,
            (nk, 1, 1),
            partial=partial,
            order=order,
            seg_start=seg_start,
            seg_count=seg_count,
            out=dk,
            dim=d,
            num_out=nk,
        )
        _record(stream, q.device, reads + writes)
    return TupleDict(dq=dq, dk=dk, dweights=dweights)


__all__ = ["dsa_prepare_segments", "dsa_det_workspace_size", "dsa_det_forward", "dsa_det_backward", "dsa_indexer_logits", "dsa_indexer_backward"]
