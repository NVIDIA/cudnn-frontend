# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native 64-head Cake sparse attention with explicit training state."""

import math
from functools import lru_cache

import torch

from cudnn._buffer_utils import check_buffers_disjoint
from cudnn._torch_stream import launch_stream, stream_context
from cudnn._tupledict import TupleDict


@lru_cache(maxsize=1)
def _backend():
    from cudnn._cake import loader

    loader.load_package("cake_dsa_sparse_attention_nondet_train")
    from cudnn._cake._vendor.cake_dsa_sparse_attention_nondet_train import cake_backend

    return cake_backend


def _tensor(name, value, shape, dtype, device):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != dtype or value.device != device:
        raise ValueError(f"{name} must have shape {shape}, dtype {dtype}, on {device}")
    if not value.is_contiguous() or value.data_ptr() % 16:
        raise ValueError(f"{name} must be contiguous and 16-byte aligned")


def _inputs(q, kv, indices, q_rope, k_rope, scale, dout=None):
    if not isinstance(q, torch.Tensor) or not q.is_cuda:
        raise ValueError("q must be a CUDA tensor")
    if (q_rope is None) != (k_rope is None):
        raise ValueError("q_rope and k_rope must both be supplied or omitted")
    b = _backend()
    if q_rope is None:
        dims = b.validate_dsa_nope_inputs(q, kv, indices, dout=dout)
    else:
        dims = b.validate_dsa_train_inputs(q, q_rope, kv, k_rope, indices, dout=dout)
    for value in (kv, indices, q_rope, k_rope, dout):
        if value is not None and value.device != q.device:
            raise ValueError("all operands must be on the same CUDA device")
    if not indices.is_contiguous() or indices.data_ptr() % 16:
        raise ValueError("indices must be contiguous and 16-byte aligned")
    if max(*dims, q.numel(), kv.numel(), *q.stride(), *kv.stride()) > 2**31 - 1:
        raise NotImplementedError("sparse attention operands must fit int32 offsets")
    scale = (1 / 16 if q_rope is None else 576**-0.5) if scale is None else scale
    if isinstance(scale, bool) or not isinstance(scale, (float, int)) or not math.isfinite(scale):
        raise ValueError("softmax_scale must be a finite host scalar")
    return dims, float(scale)


def dsa_sparse_workspace_size(total_q, total_k, top_k, *, rope=True, backward=False):
    """Scratch bytes for a single key pass, including unread ABI operands."""
    for name, value in (("total_q", total_q), ("total_k", total_k), ("top_k", top_k)):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= 2**31 - 1:
            raise ValueError(f"{name} must be a positive int32 host integer")
    b = _backend()
    layout = b.workspace_layout if rope else b.workspace_layout_nope
    return int(layout(total_q, total_k, top_k, backward=backward, key_passes=1, dkv_direct=False)["total"]) + 256


def _execute(q, kv, indices, q_rope, k_rope, scale, out, lse, residual, workspace, stream, dout=None, gradients=None):
    from cudnn._cake import loader

    b = _backend()
    backward = dout is not None
    dims, scale = _inputs(q, kv, indices, q_rope, k_rope, scale, dout)
    tq, tk, topk = dims
    rope = q_rope is not None
    _tensor("out", out, (tq, 64, 512), torch.bfloat16, q.device)
    _tensor("lse", lse, (tq, 64), torch.float32, q.device)
    _tensor("residual", residual, (tq, 64, 512), torch.bfloat16, q.device)
    if backward:
        shapes = ((tq, 64, 512), (tk, 512), (tq, 64, 64), (tk, 64)) if rope else ((tq, 64, 512), (tk, 512))
        for (name, tensor), shape in zip(gradients.items(), shapes):
            _tensor(name, tensor, shape, torch.float32 if name == "dkv" and not rope else torch.bfloat16, q.device)
    size = dsa_sparse_workspace_size(*dims, rope=rope, backward=backward)
    if not isinstance(workspace, torch.Tensor) or workspace.ndim != 1 or workspace.numel() < size:
        raise ValueError(f"workspace must be a CUDA uint8 vector of at least {size} bytes")
    _tensor("workspace", workspace, tuple(workspace.shape), torch.uint8, q.device)
    if workspace.data_ptr() % 256:
        raise ValueError("workspace must be 256-byte aligned")
    reads = [("q", q), ("kv", kv), ("indices", indices), ("q_rope", q_rope), ("k_rope", k_rope)]
    writes = [("workspace", workspace)]
    if backward:
        reads += [("out", out), ("lse", lse), ("residual", residual), ("grad_out", dout)]
        writes += list(gradients.items())
    else:
        writes += [("out", out), ("lse", lse), ("residual", residual)]
    check_buffers_disjoint(writes, reads)
    arch = loader.device_arch(q.device.index)
    loader.check_toolkit(arch, "DSA sparse attention")
    outputs = dict(out=out, lse=lse, o_lo=residual)
    if backward:
        outputs.update(dq_latent=gradients["dq"])
        outputs.update(dict(dq_rope=gradients["dq_rope"], dkv_latent=gradients["dkv"], dk_rope=gradients["dk_rope"]) if rope else dict(dkv=gradients["dkv"]))
    options = dict(topk_length=None, dout=dout, softmax_scale=scale, outputs=outputs, backward=backward, key_passes=1, dkv_direct=False)
    plan = b._plan(q, q_rope, kv, k_rope, indices, dkv_fp32=False, **options) if rope else b._plan_nope(q, kv, indices, **options)
    with torch.cuda.device(q.device), stream_context(stream, q.device):
        lengths = b._carve(workspace, plan.layout, "topk_length", torch.int32, (tq,))
        tensors = dict(q_latent=q, kv_latent=kv, indices=indices, out=out, lse=lse, o_lo=residual, topk_length=lengths)
        if rope:
            tensors.update(q_rope=q_rope, k_rope=k_rope)
        if backward:
            tensors.update(dout=dout, **outputs)
            tensors.update((b._scratch if rope else b._scratch_nope)(plan, workspace))
            if rope:
                tensors.update(dkv_latent_fp32=tensors["dkv_latent_acc"], dk_rope_fp32=tensors["dk_rope_acc"])
        if rope:
            values = b._bound_values(plan, tensors)
        else:
            # NoPE has unused RoPE and prefix operands; keep them in caller scratch.
            dummy = workspace[plan.layout["total"] : plan.layout["total"] + 16]
            values = dict(plan.constants, **tensors, indices_out=indices, cu_seqlens_q=dummy.view(torch.int32), cu_seqlens_k=dummy.view(torch.int32))
            if backward:
                acc = tensors["dkv_latent_acc"]
                values.update(
                    dkv_dst_map=lengths,
                    dkv_f32=acc,
                    src_latent=acc,
                    src_rope=dummy.view(torch.float32),
                    dst_latent=dummy.view(torch.bfloat16),
                    dst_rope=dummy.view(torch.bfloat16),
                    dst_latent_f32=gradients["dkv"],
                    dst_rope_f32=dummy.view(torch.float32),
                    dst_packed=acc,
                    dst_map=lengths,
                    dq_partial=tensors["delta"],
                    key_scratch=lengths,
                    pass_counts=lengths,
                )
        calls = b._launches(plan, values)
        if backward:
            tensors["dkv_latent_acc"].zero_()
            if rope:
                tensors["dk_rope_acc"].zero_()
        else:
            lengths.fill_(topk)
        with b._ffi_stream_context(q.device.index):
            for key in plan.backward_order if backward else plan.forward_stages:
                calls[key]()
        launch = launch_stream(stream, q.device)
        for _, value in reads + writes:
            if value is not None:
                value.record_stream(launch)


def dsa_sparse_forward(q, kv, indices, *, q_rope=None, k_rope=None, softmax_scale=None, out=None, lse=None, residual=None, workspace=None, stream=None):
    """Return ``(out, lse, residual)``; retain all three for explicit backward.

    q is [Tq, 64, 512], kv is shared K=V [Tk, 512], and indices contains
    global key rows. Optional RoPE operands have width 64. No causal mask is
    added; -1 and out-of-range slots are holes.
    """
    dims, scale = _inputs(q, kv, indices, q_rope, k_rope, softmax_scale)
    with torch.cuda.device(q.device), stream_context(stream, q.device):
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device) if out is None else out
        lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device) if lse is None else lse
        residual = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device) if residual is None else residual
        workspace = (
            torch.empty(dsa_sparse_workspace_size(*dims, rope=q_rope is not None), dtype=torch.uint8, device=q.device) if workspace is None else workspace
        )
        _execute(q, kv, indices, q_rope, k_rope, scale, out, lse, residual, workspace, stream)
    return TupleDict(out=out, lse=lse, residual=residual)


def dsa_sparse_backward(
    q,
    kv,
    indices,
    out,
    lse,
    residual,
    grad_out,
    *,
    q_rope=None,
    k_rope=None,
    softmax_scale=None,
    dq=None,
    dkv=None,
    dq_rope=None,
    dk_rope=None,
    workspace=None,
    stream=None,
):
    """Return explicit gradients; dKV/dK use nondeterministic FP32 atomics.

    NoPE dkv is FP32; RoPE dkv and dk_rope are BF16. dq is BF16.
    """
    dims, scale = _inputs(q, kv, indices, q_rope, k_rope, softmax_scale, grad_out)
    _backend()._alert_nondeterministic("dsa_sparse_backward")
    rope = q_rope is not None
    if not rope and (dq_rope is not None or dk_rope is not None):
        raise ValueError("RoPE gradients require RoPE inputs")
    with torch.cuda.device(q.device), stream_context(stream, q.device):
        gradients = dict(
            dq=torch.empty(q.shape, dtype=torch.bfloat16, device=q.device) if dq is None else dq,
            dkv=torch.empty(kv.shape, dtype=torch.bfloat16 if rope else torch.float32, device=q.device) if dkv is None else dkv,
        )
        if rope:
            gradients.update(
                dq_rope=torch.empty(q_rope.shape, dtype=torch.bfloat16, device=q.device) if dq_rope is None else dq_rope,
                dk_rope=torch.empty(k_rope.shape, dtype=torch.bfloat16, device=q.device) if dk_rope is None else dk_rope,
            )
        workspace = (
            torch.empty(dsa_sparse_workspace_size(*dims, rope=rope, backward=True), dtype=torch.uint8, device=q.device) if workspace is None else workspace
        )
        _execute(q, kv, indices, q_rope, k_rope, scale, out, lse, residual, workspace, stream, grad_out, gradients)
    return TupleDict(**gradients)


__all__ = ["dsa_sparse_forward", "dsa_sparse_backward", "dsa_sparse_workspace_size"]
