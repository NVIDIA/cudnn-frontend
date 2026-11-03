# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pooled DSA indexer training: pooled keys, pool-level target, and the backward chain."""

from contextlib import contextmanager
from typing import Optional

import torch

from cudnn.api_base import TupleDict
from cudnn.deepseek_sparse_attention.indexer_backward.api import indexer_backward_wrapper
from cudnn.deepseek_sparse_attention.utils.runtime import torch_stream_context

_WARPS = 4
_kernels = {}


def _stream_handle(stream, device):
    import cuda.bindings.driver as cuda

    if stream is None:
        return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    if isinstance(stream, torch.cuda.Stream):
        return cuda.CUstream(stream.cuda_stream)
    return stream


@contextmanager
def _on_stream(stream, device):
    """Resolve ``stream`` to a driver handle and run the wrapper's torch work on it."""
    handle = _stream_handle(stream, device)
    with torch_stream_context(handle, device):
        yield handle


def _tensor_arg(t: torch.Tensor):
    """Dynamic-layout CuTe tensor so one compiled kernel serves every shape."""
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.dim() - 1)


def _grid(total_tasks: int, device) -> tuple:
    """Dynamic launch scalars: (blocks, warp stride, iterations) as Int32 values."""
    from cutlass import Int32

    sms = torch.cuda.get_device_properties(device).multi_processor_count
    blocks = max(1, min((total_tasks + _WARPS - 1) // _WARPS, 4 * sms))
    total_warps = blocks * _WARPS
    iters = max(1, (total_tasks + total_warps - 1) // total_warps)
    return Int32(blocks), Int32(total_warps), Int32(iters), blocks


def _compile(kind, pool_size, dtype_tag, device, *sample_args):
    from ._pool_sm100 import PoolKeysKernel, UnpoolBackwardKernel
    import cutlass.cute as cute

    key = (kind, pool_size, dtype_tag, device.index)
    fn = _kernels.get(key)
    if fn is None:
        obj = PoolKeysKernel(pool_size, dtype_tag) if kind == "pool" else UnpoolBackwardKernel(pool_size, dtype_tag)
        fn = cute.compile(obj, *sample_args)
        _kernels[key] = fn
    return fn


def _check_pool_inputs(keys, gates, ape, pool_size):
    if keys.dim() != 3 or keys.shape[-1] != 128:
        raise ValueError(f"keys must be (B, S_k, 128), got {tuple(keys.shape)}")
    if gates.shape != keys.shape:
        raise ValueError("gates must match keys")
    if ape.shape != (pool_size, 128) or ape.dtype != torch.float32:
        raise ValueError(f"ape must be fp32 ({pool_size}, 128)")
    if keys.dtype != torch.bfloat16 or gates.dtype != torch.bfloat16:
        raise ValueError("keys and gates must be bf16")
    if pool_size not in (4, 16):
        raise ValueError("pool_size must be 4 or 16")


def pooled_keys_forward(keys, gates, ape, pool_size=4, out=None, stream=None) -> TupleDict:
    """softmax(gates + ape) weighted pooled keys over complete pools. Returns ``{'pooled'}``."""
    _check_pool_inputs(keys, gates, ape, pool_size)
    b, s_k, d = keys.shape
    n_pools = s_k // pool_size
    with _on_stream(stream, keys.device) as handle:
        keys, gates, ape = keys.contiguous(), gates.contiguous(), ape.contiguous()
        if out is None:
            out = torch.empty(b, n_pools, d, dtype=torch.bfloat16, device=keys.device)
        tag = "bf16" if out.dtype == torch.bfloat16 else "fp32"
        blocks, stride, iters, _ = _grid(b * n_pools, keys.device)
        args = (_tensor_arg(keys), _tensor_arg(gates), _tensor_arg(ape), _tensor_arg(out), blocks, stride, iters, handle)
        _compile("pool", pool_size, tag, keys.device, *args)(*args)
    return TupleDict(pooled=out)


def unpool_backward(keys, gates, ape, d_pooled, pool_size=4, d_keys=None, d_gates=None, d_ape=None, workspace=None, stream=None) -> TupleDict:
    """Backward of ``pooled_keys_forward``. Returns ``{'d_keys', 'd_gates', 'd_ape'}``."""
    _check_pool_inputs(keys, gates, ape, pool_size)
    b, s_k, d = keys.shape
    n_pools = s_k // pool_size
    if d_pooled.shape != (b, n_pools, d) or d_pooled.dtype != torch.float32:
        raise ValueError(f"d_pooled must be fp32 ({b}, {n_pools}, {d})")
    with _on_stream(stream, keys.device) as handle:
        keys, gates, ape, d_pooled = keys.contiguous(), gates.contiguous(), ape.contiguous(), d_pooled.contiguous()
        if d_keys is None:
            d_keys = torch.empty_like(keys)
        if d_gates is None:
            d_gates = torch.empty_like(gates)
        if d_keys.dtype != d_gates.dtype or d_keys.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("d_keys and d_gates must share a bf16 or fp32 dtype")
        tag = "bf16" if d_keys.dtype == torch.bfloat16 else "fp32"
        blocks, stride, iters, n_blocks = _grid(b * n_pools, keys.device)
        partials = workspace
        if partials is None:
            partials = torch.empty(n_blocks * _WARPS, pool_size, d, dtype=torch.float32, device=keys.device)
        args = (
            _tensor_arg(keys),
            _tensor_arg(gates),
            _tensor_arg(ape),
            _tensor_arg(d_pooled),
            _tensor_arg(d_keys),
            _tensor_arg(d_gates),
            _tensor_arg(partials),
            blocks,
            stride,
            iters,
            handle,
        )
        _compile("unpool", pool_size, tag, keys.device, *args)(*args)
        use = n_pools * pool_size
        if use < s_k:
            d_keys[:, use:].zero_()
            d_gates[:, use:].zero_()
        reduced = partials[: n_blocks * _WARPS].sum(dim=0)
        if d_ape is None:
            d_ape = reduced
        else:
            d_ape.copy_(reduced)
    return TupleDict(d_keys=d_keys, d_gates=d_gates, d_ape=d_ape)


def pooled_indexer_target(token_target: torch.Tensor, pool_ids: torch.Tensor, pool_size: int = 4, out=None) -> TupleDict:
    """Pool-level teacher: the token teacher mass of each selected pool, renormalized; tail slots dropped."""
    b, s, k = pool_ids.shape
    grouped = token_target[..., : k * pool_size].reshape(b, s, k, pool_size).sum(dim=-1)
    grouped = grouped.masked_fill(pool_ids < 0, 0.0)
    grouped = grouped / grouped.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    if out is not None:
        out.copy_(grouped)
        grouped = out
    return TupleDict(target=grouped)


def _recompute_predict(index_q, pooled, weights, pool_ids, sm_scale, handle):
    from cudnn.deepseek_sparse_attention.score_recompute._interface_sm100 import sparse_indexer_score_recompute

    b, s, k = pool_ids.shape
    predict = torch.empty(b, s, k, dtype=torch.float32, device=index_q.device)
    sparse_indexer_score_recompute(
        index_q.contiguous(),
        pooled.contiguous(),
        weights.contiguous(),
        pool_ids.to(torch.int32).contiguous(),
        out=predict,
        sm_scale=sm_scale,
        topk_indices_global=False,
        current_stream=handle,
    )
    return predict


def pooled_indexer_backward(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    gates: torch.Tensor,
    ape: torch.Tensor,
    pool_ids: torch.Tensor,
    target: torch.Tensor,
    grad_loss: torch.Tensor,
    pool_size: int = 4,
    sm_scale: float = 1.0,
    loss_coeff: float = 1.0,
    pooled: Optional[torch.Tensor] = None,
    predict: Optional[torch.Tensor] = None,
    d_index_q: Optional[torch.Tensor] = None,
    d_weights: Optional[torch.Tensor] = None,
    d_keys: Optional[torch.Tensor] = None,
    d_gates: Optional[torch.Tensor] = None,
    d_ape: Optional[torch.Tensor] = None,
    stream=None,
) -> TupleDict:
    """Indexer-loss backward on pooled keys. Returns ``{'d_index_q', 'd_weights', 'd_keys', 'd_gates', 'd_ape', 'd_pooled'}``.

    ``target`` is consumed (overwritten by the score-grad stage); ``predict`` is
    the softmax over the selected pools' ``sm_scale``-scaled scores and is
    recomputed when omitted.
    """
    with _on_stream(stream, keys.device) as handle:
        if pooled is None:
            pooled = pooled_keys_forward(keys, gates, ape, pool_size, stream=handle)["pooled"]
        if predict is None:
            predict = _recompute_predict(index_q, pooled, weights, pool_ids, sm_scale, handle)
        b, n_pools, d = pooled.shape
        d_pooled = torch.empty(b, n_pools, d, dtype=torch.float32, device=keys.device)
        grads = indexer_backward_wrapper(
            index_q,
            weights,
            pooled,
            target,
            predict,
            pool_ids,
            grad_loss,
            sm_scale=sm_scale,
            loss_coeff=loss_coeff,
            d_index_q=d_index_q,
            d_weights=d_weights,
            d_index_k=d_pooled,
            stream=handle,
        )
        unpooled = unpool_backward(keys, gates, ape, d_pooled, pool_size, d_keys=d_keys, d_gates=d_gates, d_ape=d_ape, stream=handle)
    return TupleDict(
        d_index_q=grads["d_index_q"],
        d_weights=grads["d_weights"],
        d_keys=unpooled["d_keys"],
        d_gates=unpooled["d_gates"],
        d_ape=unpooled["d_ape"],
        d_pooled=d_pooled,
    )


def pooled_expand_compact(
    pool_ids: torch.Tensor,
    positions: torch.Tensor,
    pool_size: int = 4,
    seqlen_k: Optional[int] = None,
    global_ids: bool = True,
    token_ids=None,
    lengths=None,
    width_multiple: int = 64,
    stream=None,
) -> TupleDict:
    """Compact training expansion. Returns ``{'token_ids' (B, S, K*P + P - 1) INT32, 'topk_length' (B, S) INT32}``.

    Slot group ``g`` holds pool ``pool_ids[g]`` for the valid pools (``-1`` ids
    are expected last), the incomplete tail follows, the rest is ``-1``. The
    width is rounded up to ``width_multiple`` slots (the sparse score recompute
    kernels take 64-slot blocks). With ``global_ids`` the ids carry the
    ``batch * seqlen_k`` offset the flat sparse attention kernels expect.
    """
    from ._expand_sm100 import ExpandCompactKernel
    import cutlass.cute as cute
    from cutlass import Int32

    b, s, k = pool_ids.shape
    width = k * pool_size + pool_size - 1
    if width_multiple > 1:
        width = ((width + width_multiple - 1) // width_multiple) * width_multiple
    if global_ids and seqlen_k is None:
        raise ValueError("seqlen_k is required for global ids")
    with _on_stream(stream, pool_ids.device) as handle:
        if pool_ids.dtype != torch.int32:
            pool_ids = pool_ids.to(torch.int32)
        pool_ids = pool_ids.contiguous()
        positions = positions.to(torch.int32).contiguous()
        if token_ids is None:
            token_ids = torch.empty(b, s, width, dtype=torch.int32, device=pool_ids.device)
        if lengths is None:
            lengths = torch.empty(b, s, dtype=torch.int32, device=pool_ids.device)
        blocks, stride, iters, _ = _grid(b * s, pool_ids.device)
        key = ("expand", pool_size, bool(global_ids), pool_ids.device.index)
        fn = _kernels.get(key)
        args = (
            _tensor_arg(pool_ids),
            _tensor_arg(positions),
            _tensor_arg(token_ids),
            _tensor_arg(lengths),
            Int32(int(seqlen_k or 0)),
            blocks,
            stride,
            iters,
            handle,
        )
        if fn is None:
            fn = cute.compile(ExpandCompactKernel(pool_size, bool(global_ids)), *args)
            _kernels[key] = fn
        fn(*args)
    return TupleDict(token_ids=token_ids, topk_length=lengths)


def pooled_sparse_attention_backward(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    pool_ids: torch.Tensor,
    positions: torch.Tensor,
    pool_size: int = 4,
    softmax_scale: Optional[float] = None,
    attn_sink: Optional[torch.Tensor] = None,
    token_ids: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    dq: Optional[torch.Tensor] = None,
    dkv: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    stream=None,
) -> TupleDict:
    """Sparse attention backward over the selected pools plus the tail.

    ``q``/``out``/``dout`` are (B, S, H, D), ``kv`` is (B, S_k, D), ``lse`` is
    (B, S, H) FP32, ``pool_ids`` (B, S, K) INT32 local, ``positions`` (B, S).
    Returns ``{'dq' (B, S, H, D), 'dkv' (B, S_k, D), 'd_sink', 'token_ids', 'topk_length'}``.
    """
    from cudnn.deepseek_sparse_attention.sparse_attention_backward.api import sparse_attention_backward_wrapper

    b, s, h, d = q.shape
    s_k = kv.shape[1]
    with _on_stream(stream, q.device) as handle:
        if token_ids is None or topk_length is None:
            expanded = pooled_expand_compact(pool_ids, positions, pool_size, seqlen_k=s_k, global_ids=True, stream=handle)
            token_ids, topk_length = expanded["token_ids"], expanded["topk_length"]
        if attn_sink is None:
            attn_sink = torch.full((h,), float("-inf"), dtype=torch.float32, device=q.device)
        result = sparse_attention_backward_wrapper(
            q.reshape(b * s, h, d),
            kv.reshape(b * s_k, d),
            out.reshape(b * s, h, -1),
            dout.reshape(b * s, h, -1),
            lse.reshape(b * s, h),
            attn_sink,
            token_ids.reshape(b * s, -1),
            softmax_scale=softmax_scale,
            topk_length=topk_length.reshape(b * s),
            dq=None if dq is None else dq.reshape(b * s, h, d),
            dkv=None if dkv is None else dkv.reshape(b * s_k, d),
            deterministic=deterministic,
            stream=handle,
        )
    return TupleDict(
        dq=result["dq"].reshape(b, s, h, d), dkv=result["dkv"].reshape(b, s_k, d), d_sink=result["d_sink"], token_ids=token_ids, topk_length=topk_length
    )


def pooled_teacher_recompute(
    q_attn: torch.Tensor,
    k_attn: torch.Tensor,
    lse: torch.Tensor,
    token_ids: torch.Tensor,
    topk_length: torch.Tensor,
    k_pools: int,
    pool_size: int = 4,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    stream=None,
) -> TupleDict:
    """Pool-level teacher straight from the sparse attention score recompute. Returns ``{'target' (B, S, K)}``.

    ``token_ids`` is the compact pool-grouped expansion with global ids and
    ``topk_length`` its per-row valid count including the tail; the tail is
    excluded by rounding the length down to whole pools.
    """
    from cudnn.deepseek_sparse_attention.score_recompute._interface_sm100 import _sparse_attn_score_recompute

    b, s, h, d = q_attn.shape
    if softmax_scale is None:
        softmax_scale = d**-0.5
    with _on_stream(stream, q_attn.device) as handle:
        if out is None:
            out = torch.empty(b, s, k_pools, dtype=torch.float32, device=q_attn.device)
        pool_tokens = (topk_length.to(torch.int32) // pool_size * pool_size).contiguous()
        _sparse_attn_score_recompute(
            q_attn.contiguous(),
            k_attn.contiguous(),
            lse.contiguous().float(),
            token_ids.to(torch.int32).contiguous(),
            float(softmax_scale),
            out=out,
            topk_length=pool_tokens,
            topk_indices_global=True,
            pool_size=pool_size,
            current_stream=handle,
        )
    return TupleDict(target=out)


def pooled_sparse_attention_backward_with_teacher(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    pool_ids: torch.Tensor,
    positions: torch.Tensor,
    k_pools: int,
    pool_size: int = 4,
    softmax_scale: Optional[float] = None,
    token_ids: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    dq: Optional[torch.Tensor] = None,
    dkv: Optional[torch.Tensor] = None,
    target: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    stream=None,
) -> TupleDict:
    """Sparse attention backward that also emits the pool-level teacher.

    Same layouts as ``pooled_sparse_attention_backward``; ``target`` is
    (B, S, K) FP32, renormalized over the valid pools (zero rows when a query
    has no complete pool). Returns ``{'dq', 'dkv', 'd_sink', 'target', 'token_ids', 'topk_length'}``.
    """
    from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm100 as _iface
    from cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100_common import flash_attn_bwd_sm100_workspace_size

    b, s, h, d = q.shape
    s_k = kv.shape[1]
    if softmax_scale is None:
        softmax_scale = d**-0.5
    with _on_stream(stream, q.device) as handle:
        if token_ids is None or topk_length is None:
            expanded = pooled_expand_compact(pool_ids, positions, pool_size, seqlen_k=s_k, global_ids=True, stream=handle)
            token_ids, topk_length = expanded["token_ids"], expanded["topk_length"]
        q2, kv2, out2, dout2, lse2 = (
            q.reshape(b * s, h, d).contiguous(),
            kv.reshape(b * s_k, d).contiguous(),
            out.reshape(b * s, h, -1).contiguous(),
            dout.reshape(b * s, h, -1).contiguous(),
            lse.reshape(b * s, h).float().contiguous(),
        )
        if dq is None:
            dq = torch.empty_like(q)
        if dkv is None:
            dkv = torch.empty_like(kv)
        if target is None:
            target = torch.empty(b, s, k_pools, dtype=torch.float32, device=q.device)
        target.zero_()
        d_sink = torch.empty(h, dtype=torch.float32, device=q.device)
        attn_sink = torch.full((h,), float("-inf"), dtype=torch.float32, device=q.device)
        if workspace is None:
            workspace = torch.empty(flash_attn_bwd_sm100_workspace_size(b * s, b * s_k, d, h), dtype=torch.uint8, device=q.device)
        _iface.flash_attn_bwd_sm100(
            q2,
            kv2,
            out2,
            dout2,
            lse2,
            attn_sink,
            token_ids.reshape(b * s, -1).contiguous(),
            softmax_scale=float(softmax_scale),
            topk_length=topk_length.reshape(b * s).contiguous(),
            dq=dq.view(b * s, h, d),
            dkv=dkv.view(b * s_k, d),
            current_stream=handle,
            workspace=workspace,
            d_sink=d_sink,
            pool_target=target.view(b * s, k_pools),
            pool_size=pool_size,
        )
        target.div_(target.sum(dim=-1, keepdim=True).clamp(min=1e-12))
    return TupleDict(dq=dq, dkv=dkv, d_sink=d_sink, target=target, token_ids=token_ids, topk_length=topk_length)


def pooled_indexer_select(
    index_q: torch.Tensor,
    pooled: torch.Tensor,
    weights: torch.Tensor,
    k_pools: int,
    pool_size: int = 4,
    sm_scale: float = 1.0,
    deterministic: bool = False,
    stream=None,
) -> TupleDict:
    """Select the top pools and return their softmax in one launch. Returns ``{'pool_ids' (B, S, K) INT32 local, 'predict' (B, S, K) FP32, 'logits'}``.

    The ratio-causal rule of the compressed indexer forward with ``ratio = pool_size``
    is the pooled eligibility rule; ``predict`` feeds ``pooled_indexer_backward``
    directly, so the backward needs no predict recompute.
    """
    from cudnn.deepseek_sparse_attention.indexer_forward.api import indexer_forward_top_k_wrapper

    with _on_stream(stream, index_q.device) as handle:
        result = indexer_forward_top_k_wrapper(
            index_q,
            pooled.unsqueeze(2),
            weights,
            top_k=k_pools,
            ratio=pool_size,
            sm_scale=sm_scale,
            return_softmax=True,
            topk_indices_global=False,
            deterministic=deterministic,
            stream=handle,
        )
    return TupleDict(pool_ids=result["indices"], predict=result["softmax"], logits=result["logits"])
