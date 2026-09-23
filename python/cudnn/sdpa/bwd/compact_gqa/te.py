# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in TE adapter for host-certified sequence packing."""

import functools
import inspect
import json
import operator
import os
from numbers import Integral
import weakref

import torch

_prefixes = {}
_plans = {}
_workspace = None
_workspace_key = None
_workspace_capacity = 0
_step = 0
_enabled = False
_installed = False
_counts = {}
_native_only = os.environ.get("CUDNN_COMPACT_GQA_NATIVE_ONLY") == "1"
_trace_packs = os.environ.get("CUDNN_COMPACT_GQA_TRACE_PACKS") == "1"
_packing_work = {}
_unmatched_packing_calls = 0


def register_packing(tensor, host_offsets):
    """Register immutable GPU prefixes created from these CPU offsets."""
    if torch.is_tensor(host_offsets):
        raise TypeError("Packing offsets must be host integers.")
    if any(not isinstance(n, Integral) for n in host_offsets):
        raise TypeError("Packing offsets must contain host integers.")
    offsets = tuple(operator.index(n) for n in host_offsets)
    if len(offsets) < 2 or offsets[0] != 0 or any(b < a for a, b in zip(offsets, offsets[1:])):
        return
    if tensor.dtype != torch.int32 or not tensor.is_cuda or not tensor.is_contiguous() or tensor.numel() != len(offsets):
        return
    key = (tensor.device, tensor.data_ptr())

    def expired(ref):
        if _prefixes.get(key, (None,))[0] is ref:
            _prefixes.pop(key, None)

    _prefixes[key] = (weakref.ref(tensor, expired), tensor._version, offsets)


def _host_offsets(tensor):
    if tensor is None or tensor.dtype != torch.int32 or not tensor.is_contiguous():
        return None
    entry = _prefixes.get((tensor.device, tensor.data_ptr()))
    if entry is None:
        return None
    owner, version, offsets = entry
    owner = owner()
    if owner is not None and owner._version == version == tensor._version and tensor.numel() == len(offsets):
        return offsets
    return None


def _certified(tensor, tokens):
    offsets = _host_offsets(tensor)
    return offsets is not None and offsets[-1] == tokens


def _packing(x):
    logical = _host_offsets(x["cu_seqlens_q"])
    if logical is None or logical != _host_offsets(x["cu_seqlens_kv"]):
        return None
    physical = []
    for name in ("cu_seqlens_q_padded", "cu_seqlens_kv_padded"):
        tensor = x.get(name)
        physical.append(logical if tensor is None else _host_offsets(tensor))
    offsets = physical[0]
    if offsets is None or offsets != physical[1] or len(offsets) != len(logical) or offsets[-1] != x["q"].shape[0]:
        return None
    lengths = tuple(b - a for a, b in zip(logical, logical[1:]))
    if any(length > b - a for length, a, b in zip(lengths, offsets, offsets[1:])):
        return None
    if max(lengths) > min(x["max_seqlen_q"], x["max_seqlen_kv"]):
        return None
    return offsets, lengths


def eligible(x):
    """Use host metadata only; prefix contents are certified by the packer."""
    q = x["q"]
    if not q.is_cuda or torch.cuda.get_device_capability(q.device) != (10, 7):
        return False
    tokens = q.shape[0]
    if tokens < 1:
        return False
    if x["qkv_layout"] != "thd_thd_thd" or x["dqkv_layout"] != "thd_thd_thd":
        return False
    if x["o_format"] != "thd" or x["do_format"] != "thd":
        return False
    if x["dropout"] != 0 or x["attn_bias_type"] != "no_bias" or x["softmax_type"] != "vanilla":
        return False
    if x["attn_mask_type"] not in ("causal", "padding_causal") or x["attn_scale"] not in (None, 1 / 16):
        return False
    if tuple(x["window_size"]) not in ((-1, -1), (-1, 0)) or x["cuda_graph"] or x["deterministic"]:
        return False
    if any(x.get(key) is not None for key in ("s_quantizer", "dp_quantizer", "dqkv_quantizer")):
        return False
    for name, heads in (("q", 8), ("k", 1), ("v", 1), ("o", 8), ("d_o", 8)):
        t = x[name]
        if t.device != q.device or t.dtype != torch.bfloat16 or tuple(t.shape) != (tokens, heads, 256) or not t.is_contiguous():
            return False
    if _packing(x) is None:
        return False
    if not x["aux_ctx_tensors"]:
        return False
    lse = x["aux_ctx_tensors"][0]
    return lse.device == q.device and lse.dtype == torch.float32 and lse.is_contiguous() and tuple(lse.shape) in ((tokens, 8), (tokens, 8, 1))


def report():
    if _counts:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
        data = {"step": _step, "rank": rank, "native_only": _native_only, **_counts}
        if _trace_packs:
            data["packing_work"] = [{"offsets": offsets, "lengths": lengths, "calls": count} for (offsets, lengths), count in sorted(_packing_work.items())]
            data["unmatched_packing_calls"] = _unmatched_packing_calls
        print("COMPACT_GQA " + json.dumps(data), flush=True)


def set_enabled(enabled):
    """Release candidate scratch when switching to a native control."""
    global _enabled, _workspace, _workspace_key, _workspace_capacity
    _enabled = bool(enabled)
    if not _enabled:
        _workspace = None
        _workspace_key = None
        _workspace_capacity = 0


def begin_step():
    """Enable candidate backward for each training batch."""
    global _step, _counts, _unmatched_packing_calls
    install()
    report()
    _step += 1
    _counts = {}
    _packing_work.clear()
    _unmatched_packing_calls = 0
    set_enabled(not _native_only)


def install():
    global _installed
    if _installed:
        return
    from transformer_engine.pytorch.attention.dot_product_attention import backends

    original = backends.fused_attn_bwd
    signature = inspect.signature(original)

    @functools.wraps(original)
    def backward(*args, **kwargs):
        global _workspace, _workspace_key, _workspace_capacity, _unmatched_packing_calls
        params = signature.bind(*args, **kwargs)
        params.apply_defaults()
        x = params.arguments
        if _trace_packs:
            packing = _packing(x)
            if packing is None:
                _unmatched_packing_calls += 1
            else:
                _packing_work[packing] = _packing_work.get(packing, 0) + 1
        if not _enabled or not eligible(x):
            reason = "native_control" if _native_only else "native_startup" if not _enabled else "native_fallback"
            _counts[reason] = _counts.get(reason, 0) + 1
            return original(*args, **kwargs)
        from .api import CompactGqaBackward, compact_gqa_backward

        q = x["q"]
        stream = torch.cuda.current_stream(q.device).cuda_stream
        offsets, lengths = _packing(x)
        capacity = max(1, max(lengths))
        key = (q.device, stream, x["aux_ctx_tensors"][0].ndim)
        if key not in _plans:
            plan = CompactGqaBackward(
                q, x["k"], x["v"], x["o"], x["d_o"], x["aux_ctx_tensors"][0], max_seqlen=capacity, query_rows=32768, groups=4, fast_store=True
            )
            plan.compile(stream)
            _plans[key] = plan
        plan = _plans[key]
        if _workspace_key != (q.device, stream) or _workspace_capacity < capacity:
            _workspace = torch.empty(plan.scratch_workspace_bytes(capacity), dtype=torch.uint8, device=q.device)
            _workspace_key = (q.device, stream)
            _workspace_capacity = capacity
        if plan._workspace_ref is None or plan._workspace_ref() is not _workspace:
            plan.initialize_workspace(_workspace, stream, max_seqlen=_workspace_capacity)
        result = compact_gqa_backward(
            q,
            x["k"],
            x["v"],
            x["o"],
            x["d_o"],
            x["aux_ctx_tensors"][0],
            plan=plan,
            workspace=_workspace,
            current_stream=stream,
            sequence_offsets=offsets,
            sequence_lengths=lengths,
        )
        _counts["candidate"] = _counts.get("candidate", 0) + 1
        return [result["dq"], result["dk"], result["dv"], None, None]

    backends.fused_attn_bwd = backward
    _installed = True
