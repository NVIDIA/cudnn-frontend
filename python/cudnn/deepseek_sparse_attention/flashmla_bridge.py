# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional FlashMLA forward bridge for cuDNN DSA training.

This module does not contain or vendor a sparse-attention forward kernel.  It
dynamically calls ``flash_mla.flash_mla_sparse_fwd`` from the external,
MIT-licensed `deepseek-ai/FlashMLA <https://github.com/deepseek-ai/FlashMLA>`_
package, then connects its ``out`` and KV-only ``lse`` to the cuDNN Frontend
DSA backward and score-recompute APIs.

The adapter was developed against upstream FlashMLA commit
``15f13e5030374295491c5ce31b02d7e63a7772c6``.  FlashMLA remains a separate,
optional dependency; no FlashMLA or vLLM implementation source is copied into
cuDNN Frontend.

The initial contract is deliberately narrow: exact SM100 (validated on B200),
BF16, one flat MQA KV stream, QK dimension 512 or 576, V dimension 512, and
16/32/64/128 query heads.  FlashMLA itself launches 64- or 128-head kernels, so smaller
cuDNN backward head counts are zero-padded to 64 for forward and sliced back
before returning.  KV and aligned raw-forward Top-K inputs use zero-copy
singleton-head views.  The training and score-recompute paths materialize
safety-normalized indices because FlashMLA's invalid-index contract is wider
than the current cuDNN backward contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import math
from typing import Any, Callable, Optional, Tuple

import torch

from cudnn.api_base import TupleDict

_FLASHMLA_INSTALL_HINT = "Install the official deepseek-ai/FlashMLA package and make sure its " "compiled extension is importable in this Python environment."
_SM100_CAPABILITY = (10, 0)
_SUPPORTED_HEADS = (16, 32, 64, 128)
_SUPPORTED_HEAD_DIMS = (512, 576)
_VALUE_DIM = 512
_MAX_C_STRIDE = (1 << 31) - 1
_SCORE_TOPK_TILE = 128


class FlashMLABridgeUnavailableError(RuntimeError):
    """The external FlashMLA sparse-forward dependency is unavailable."""


@dataclass(frozen=True)
class FlashMLASparseForwardPlan:
    """Host-side launch adaptation for the external FlashMLA kernel."""

    num_heads: int
    launch_num_heads: int
    head_dim: int
    value_dim: int
    topk: int
    launch_topk: int
    topk_tile: int

    @property
    def pads_heads(self) -> bool:
        return self.launch_num_heads != self.num_heads

    @property
    def pads_topk(self) -> bool:
        return self.launch_topk != self.topk


@dataclass(frozen=True)
class _FlashMLALaunchInputs:
    q: torch.Tensor
    kv: torch.Tensor
    indices: torch.Tensor
    attn_sink: Optional[torch.Tensor]
    topk_length: Optional[torch.Tensor]
    plan: FlashMLASparseForwardPlan


def _require_plain_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def plan_flashmla_sparse_forward(
    num_heads: int,
    head_dim: int,
    topk: int,
) -> FlashMLASparseForwardPlan:
    """Plan the minimal H/Top-K padding without importing FlashMLA or CUDA.

    On the pinned upstream SM100 implementation, the H64 path consumes Top-K
    tiles of 64.  H128/D512 uses its 64-wide small-Top-K path through K1280;
    other H128 cases use 128-wide tiles.  H16/H32 are padded only to H64;
    already aligned Top-K tensors remain zero-copy in the raw forward adapter.
    """

    num_heads = _require_plain_int(num_heads, "num_heads")
    head_dim = _require_plain_int(head_dim, "head_dim")
    topk = _require_plain_int(topk, "topk")
    if num_heads not in _SUPPORTED_HEADS:
        raise ValueError(f"num_heads must be one of {_SUPPORTED_HEADS}, got {num_heads}")
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(f"head_dim must be one of {_SUPPORTED_HEAD_DIMS}, got {head_dim}")
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")

    launch_num_heads = 64 if num_heads <= 64 else 128
    use_h128_small_topk = launch_num_heads == 128 and head_dim == 512 and topk <= 1280
    topk_tile = 64 if launch_num_heads == 64 or use_h128_small_topk else 128
    return FlashMLASparseForwardPlan(
        num_heads=num_heads,
        launch_num_heads=launch_num_heads,
        head_dim=head_dim,
        value_dim=_VALUE_DIM,
        topk=topk,
        launch_topk=_round_up(topk, topk_tile),
        topk_tile=topk_tile,
    )


def _resolve_flashmla_sparse_fwd() -> Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Resolve the official external entry point without caching failures."""

    try:
        flash_mla = import_module("flash_mla")
    except Exception as exc:
        raise FlashMLABridgeUnavailableError(f"Cannot import optional dependency 'flash_mla'. {_FLASHMLA_INSTALL_HINT}") from exc

    sparse_fwd = getattr(flash_mla, "flash_mla_sparse_fwd", None)
    if not callable(sparse_fwd):
        raise FlashMLABridgeUnavailableError("The imported 'flash_mla' package does not export a callable " f"'flash_mla_sparse_fwd'. {_FLASHMLA_INSTALL_HINT}")
    return sparse_fwd


def _normalize_softmax_scale(softmax_scale: Optional[float], head_dim: int) -> float:
    scale = 1.0 / math.sqrt(head_dim) if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"softmax_scale must be finite and positive, got {scale}")
    return scale


def _check_tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    return value


def _validate_flashmla_contract(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
    topk_length: Optional[torch.Tensor],
    softmax_scale: Optional[float],
) -> tuple[FlashMLASparseForwardPlan, float]:
    q = _check_tensor(q, "q")
    kv = _check_tensor(kv, "kv")
    indices = _check_tensor(indices, "indices")
    if attn_sink is not None:
        attn_sink = _check_tensor(attn_sink, "attn_sink")
    if topk_length is not None:
        topk_length = _check_tensor(topk_length, "topk_length")

    if q.ndim != 3:
        raise ValueError(f"q must have shape (S_q, H, D), got {tuple(q.shape)}")
    if kv.ndim != 2:
        raise ValueError(f"kv must have shape (S_kv, D), got {tuple(kv.shape)}")
    if indices.ndim != 2:
        raise ValueError(f"indices must have shape (S_q, topk), got {tuple(indices.shape)}")
    if q.shape[0] <= 0 or kv.shape[0] <= 0:
        raise ValueError(f"q and kv sequence lengths must be positive, got {q.shape[0]} and {kv.shape[0]}")

    s_q, num_heads, head_dim = q.shape
    if kv.shape[1] != head_dim:
        raise ValueError(f"kv must have shape (S_kv, {head_dim}), got {tuple(kv.shape)}")
    if indices.shape[0] != s_q:
        raise ValueError(f"indices must have shape ({s_q}, topk), got {tuple(indices.shape)}")
    plan = plan_flashmla_sparse_forward(num_heads, head_dim, indices.shape[1])

    if q.dtype != torch.bfloat16:
        raise TypeError(f"q must have dtype torch.bfloat16, got {q.dtype}")
    if kv.dtype != torch.bfloat16:
        raise TypeError(f"kv must have dtype torch.bfloat16, got {kv.dtype}")
    if indices.dtype != torch.int32:
        raise TypeError(f"indices must have dtype torch.int32, got {indices.dtype}")

    if attn_sink is not None:
        if attn_sink.shape != (num_heads,):
            raise ValueError(f"attn_sink must have shape {(num_heads,)}, got {tuple(attn_sink.shape)}")
        if attn_sink.dtype != torch.float32:
            raise TypeError(f"attn_sink must have dtype torch.float32, got {attn_sink.dtype}")
    if topk_length is not None:
        if topk_length.shape != (s_q,):
            raise ValueError(f"topk_length must have shape {(s_q,)}, got {tuple(topk_length.shape)}")
        if topk_length.dtype != torch.int32:
            raise TypeError(f"topk_length must have dtype torch.int32, got {topk_length.dtype}")

    tensors = [q, kv, indices]
    if attn_sink is not None:
        tensors.append(attn_sink)
    if topk_length is not None:
        tensors.append(topk_length)
    if q.device.type != "cuda":
        raise RuntimeError(f"the FlashMLA bridge is CUDA-only; q is on {q.device}")
    mismatched = [str(t.device) for t in tensors if t.device != q.device]
    if mismatched:
        raise ValueError(f"all inputs must share q's device {q.device}; mismatches: {mismatched}")
    capability = torch.cuda.get_device_capability(q.device)
    if capability != _SM100_CAPABILITY:
        raise RuntimeError(
            "this prototype is fail-closed to exact SM100 (validated on NVIDIA B200); "
            f"device {q.device} has compute capability {capability[0]}.{capability[1]}"
        )

    for name, tensor in (
        ("q", q),
        ("kv", kv),
        ("indices", indices),
        ("attn_sink", attn_sink),
        ("topk_length", topk_length),
    ):
        if tensor is None:
            continue
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name}'s last dimension must be contiguous, got stride {tensor.stride()}")
        if any(stride < 0 or stride > _MAX_C_STRIDE for stride in tensor.stride()):
            raise ValueError(f"{name} has a stride that cannot be represented by FlashMLA's int32 ABI: {tensor.stride()}")

    return plan, _normalize_softmax_scale(softmax_scale, head_dim)


def _prepare_flashmla_launch_inputs(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
    topk_length: Optional[torch.Tensor],
    plan: FlashMLASparseForwardPlan,
) -> _FlashMLALaunchInputs:
    """Apply only the singleton-head views and padding described by ``plan``."""

    if plan.pads_heads:
        q_launch = q.new_zeros((q.shape[0], plan.launch_num_heads, plan.head_dim))
        q_launch[:, : plan.num_heads, :].copy_(q)
        if attn_sink is None:
            sink_launch = None
        else:
            sink_launch = attn_sink.new_zeros((plan.launch_num_heads,))
            sink_launch[: plan.num_heads].copy_(attn_sink)
    else:
        q_launch = q
        sink_launch = attn_sink

    # FlashMLA models this as one MQA KV head.  unsqueeze is a zero-copy view.
    kv_launch = kv.unsqueeze(1)
    if plan.pads_topk:
        indices_launch = indices.new_full((indices.shape[0], 1, plan.launch_topk), -1)
        indices_launch[:, 0, : plan.topk].copy_(indices)
    else:
        # Likewise, aligned flat Top-K indices gain only a singleton-head view.
        indices_launch = indices.unsqueeze(1)

    return _FlashMLALaunchInputs(
        q=q_launch,
        kv=kv_launch,
        indices=indices_launch,
        attn_sink=sink_launch,
        topk_length=topk_length,
        plan=plan,
    )


def _mask_invalid_sparse_indices(
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    s_kv: int,
) -> torch.Tensor:
    """Map every invalid/inactive slot to ``-1`` without changing positions."""

    topk = indices.shape[1]
    valid = (indices >= 0) & (indices < s_kv)
    if topk_length is not None:
        bounded_length = topk_length.clamp(min=0, max=topk)
        positions = torch.arange(topk, dtype=torch.int32, device=indices.device).unsqueeze(0)
        valid = valid & (positions < bounded_length.unsqueeze(1))
    return torch.where(valid, indices, -1)


def _normalize_cudnn_sparse_metadata(
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    s_kv: int,
    *,
    _compactify: Optional[Callable[[torch.Tensor], Any]] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Make FlashMLA metadata safe for cuDNN backward and recompute.

    FlashMLA treats every negative or ``>= S_kv`` index as invalid.  Current
    cuDNN DSA backward only guards negative sentinels without lengths, and it
    assumes every entry in a compact active prefix is valid when lengths are
    present.  Normalize high sentinels to ``-1`` in both cases.  For the
    length form, also mask the inactive suffix, compact valid entries to the
    front, and derive a new length.  This is asynchronous GPU work: there is
    no device-to-host validation or synchronization in the adapter.

    Out-of-contract length values are clamped to the physical ``[0, K]``
    envelope before masking.  An asynchronous device assert would poison the
    CUDA context on failure, while a strict host check would synchronize the
    hot path; clamping gives the downstream kernels a memory-safe contract.
    """

    normalized = _mask_invalid_sparse_indices(indices, topk_length, s_kv)
    if topk_length is None:
        return normalized, None

    if _compactify is None:
        # Lazy import preserves the optional CuTe DSL boundary at module import.
        from .indexer_top_k.api import compactify_wrapper

        _compactify = compactify_wrapper
    compact = _compactify(normalized)
    return compact["indices"], compact["topk_length"]


def _validate_flashmla_outputs(
    outputs: Any,
    launch: _FlashMLALaunchInputs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
        raise RuntimeError("flash_mla_sparse_fwd must return (output, max_logits, lse)")
    output, max_logits, lse = outputs
    for name, tensor in (("output", output), ("max_logits", max_logits), ("lse", lse)):
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"FlashMLA returned non-tensor {name}: {type(tensor).__name__}")
        if tensor.device != launch.q.device:
            raise RuntimeError(f"FlashMLA returned {name} on {tensor.device}, expected {launch.q.device}")

    plan = launch.plan
    expected_output = (launch.q.shape[0], plan.launch_num_heads, plan.value_dim)
    expected_aux = (launch.q.shape[0], plan.launch_num_heads)
    if output.shape != expected_output or output.dtype != torch.bfloat16:
        raise RuntimeError(f"FlashMLA output must be BF16 {expected_output}, got {output.dtype} {tuple(output.shape)}")
    if max_logits.shape != expected_aux or max_logits.dtype != torch.float32:
        raise RuntimeError(f"FlashMLA max_logits must be FP32 {expected_aux}, got {max_logits.dtype} {tuple(max_logits.shape)}")
    if lse.shape != expected_aux or lse.dtype != torch.float32:
        raise RuntimeError(f"FlashMLA lse must be FP32 {expected_aux}, got {lse.dtype} {tuple(lse.shape)}")

    if plan.pads_heads:
        output = output[:, : plan.num_heads, :].contiguous()
        max_logits = max_logits[:, : plan.num_heads].contiguous()
        lse = lse[:, : plan.num_heads].contiguous()
    return output, max_logits, lse


def _run_flashmla_sparse_forward(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
    topk_length: Optional[torch.Tensor],
    softmax_scale: Optional[float],
) -> tuple[TupleDict, float]:
    plan, scale = _validate_flashmla_contract(q, kv, indices, attn_sink, topk_length, softmax_scale)
    sparse_fwd = _resolve_flashmla_sparse_fwd()
    # The external dispatcher and current cuDNN wrappers inspect current-device
    # state.  Keep preparation, launch, and padded-output slicing under q's
    # device guard so a multi-GPU caller does not accidentally route by another
    # device or enqueue adapter work on another device's current stream.
    with torch.cuda.device(q.device):
        launch = _prepare_flashmla_launch_inputs(q, kv, indices, attn_sink, topk_length, plan)
        outputs = sparse_fwd(
            launch.q,
            launch.kv,
            launch.indices,
            sm_scale=scale,
            d_v=plan.value_dim,
            attn_sink=launch.attn_sink,
            topk_length=launch.topk_length,
        )
        output, max_logits, lse = _validate_flashmla_outputs(outputs, launch)
    return TupleDict(output=output, max_logits=max_logits, lse=lse), scale


def flashmla_sparse_forward_wrapper(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    softmax_scale: Optional[float] = None,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> TupleDict:
    """Call the external FlashMLA sparse forward without autograd wiring.

    ``q`` is ``(S_q, H, D)``, ``kv`` is ``(S_kv, D)``, and ``indices`` is
    ``(S_q, topk)`` with global positions into the flat KV stream.  Invalid
    indices may be ``-1`` or at least ``S_kv``.  ``topk_length``, when given,
    must contain values in ``[0, topk]``.  The returned ``lse`` excludes the
    attention sink, matching cuDNN DSA backward.

    Use :func:`flashmla_cudnn_sparse_attention_wrapper` for training.
    """

    result, _ = _run_flashmla_sparse_forward(q, kv, indices, attn_sink, topk_length, softmax_scale)
    return result


class _FlashMLACudnnSparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, indices, attn_sink, topk_length, softmax_scale):
        _validate_flashmla_contract(q, kv, indices, attn_sink, topk_length, softmax_scale)
        # Fail before launching metadata-normalization kernels when the
        # optional external forward is absent or incompatible.
        _resolve_flashmla_sparse_fwd()
        with torch.cuda.device(q.device):
            safe_indices, safe_topk_length = _normalize_cudnn_sparse_metadata(indices, topk_length, kv.shape[0])
            result, scale = _run_flashmla_sparse_forward(q, kv, safe_indices, attn_sink, safe_topk_length, softmax_scale)
        output, max_logits, lse = result
        ctx.softmax_scale = scale
        ctx.has_topk_length = safe_topk_length is not None
        tensors = [q, kv, output, lse, attn_sink, safe_indices]
        if safe_topk_length is not None:
            tensors.append(safe_topk_length)
        ctx.save_for_backward(*tensors)
        ctx.mark_non_differentiable(max_logits, lse)
        return output, max_logits, lse

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dout, _dmax_logits, _dlse):
        if dout is None:
            return (None,) * 6
        saved = ctx.saved_tensors
        q, kv, output, lse, attn_sink, indices = saved[:6]
        topk_length = saved[6] if ctx.has_topk_length else None

        # Import lazily so importing the bridge remains cheap and does not JIT
        # or initialize the CuTe DSL backward until a gradient is requested.
        from .sparse_attention_backward.api import sparse_attention_backward_wrapper

        with torch.cuda.device(q.device):
            result = sparse_attention_backward_wrapper(
                q,
                kv,
                output,
                dout,
                lse,
                attn_sink,
                indices,
                softmax_scale=ctx.softmax_scale,
                topk_length=topk_length,
            )
        needs = ctx.needs_input_grad
        return (
            result["dq"] if needs[0] else None,
            result["dkv"] if needs[1] else None,
            None,
            result["d_sink"] if needs[3] else None,
            None,
            None,
        )


def flashmla_cudnn_sparse_attention_wrapper(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: Optional[float] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> TupleDict:
    """Training bridge: external FlashMLA forward plus cuDNN DSA backward.

    Gradients are provided for ``q``, ``kv``, and ``attn_sink``.  Top-K
    indices/lengths and ``softmax_scale`` are non-differentiable.  FlashMLA's
    ``max_logits`` and KV-only ``lse`` are returned for diagnostics and score
    recompute but are explicitly non-differentiable outputs.  Before forward,
    invalid indices are normalized to the current cuDNN backward contract;
    with ``topk_length``, valid active entries are compacted and a safe length
    is derived.  Forward and backward therefore consume identical metadata.
    """

    if attn_sink is None:
        raise ValueError("attn_sink is required by the cuDNN DSA training backward")
    output, max_logits, lse = _FlashMLACudnnSparseAttention.apply(q, kv, indices, attn_sink, topk_length, softmax_scale)
    return TupleDict(output=output, max_logits=max_logits, lse=lse)


def flashmla_sparse_score_recompute_wrapper(
    q: torch.Tensor,
    kv: torch.Tensor,
    lse: torch.Tensor,
    indices: torch.Tensor,
    softmax_scale: Optional[float] = None,
    topk_length: Optional[torch.Tensor] = None,
    stream: Optional[Any] = None,
) -> TupleDict:
    """Run cuDNN attention-score recompute from a FlashMLA forward tuple.

    This is a single-flat-sequence adapter over
    :func:`DSA.sparse_attn_score_recompute_wrapper`.  It safety-normalizes the
    wider FlashMLA invalid-index contract in place, masks the inactive suffix,
    and uses the non-compact recompute path so every returned target slot stays
    aligned with the caller's original ``indices`` position.  The launch-only
    tail is padded to a conservative 128-slot tile and sliced away afterward.
    """

    if stream is not None:
        raise NotImplementedError("the B200 bridge prototype supports only the current PyTorch stream")

    _, scale = _validate_flashmla_contract(q, kv, indices, None, topk_length, softmax_scale)
    lse = _check_tensor(lse, "lse")
    expected_lse = (q.shape[0], q.shape[1])
    if lse.shape != expected_lse or lse.dtype != torch.float32:
        raise ValueError(f"lse must be FP32 {expected_lse}, got {lse.dtype} {tuple(lse.shape)}")
    if lse.device != q.device:
        raise ValueError(f"lse must be on q's device {q.device}, got {lse.device}")
    if lse.stride(-1) != 1:
        raise ValueError(f"lse's last dimension must be contiguous, got stride {lse.stride()}")

    from .score_recompute.api import sparse_attn_score_recompute_wrapper

    with torch.cuda.device(q.device):
        safe_indices = _mask_invalid_sparse_indices(indices, topk_length, kv.shape[0])
        launch_topk = _round_up(indices.shape[1], _SCORE_TOPK_TILE)
        if launch_topk == indices.shape[1]:
            launch_indices = safe_indices
        else:
            launch_indices = indices.new_full((indices.shape[0], launch_topk), -1)
            launch_indices[:, : indices.shape[1]].copy_(safe_indices)
        result = sparse_attn_score_recompute_wrapper(
            q.unsqueeze(0),
            kv.unsqueeze(0),
            lse.unsqueeze(0),
            launch_indices.unsqueeze(0),
            scale,
            qhead_per_kv_head=q.shape[1],
            topk_length=None,
            topk_indices_global=False,
            stream=stream,
        )
        target = result["target"].squeeze(0)[:, : indices.shape[1]].contiguous()
    return TupleDict(target=target, indices=safe_indices)


__all__ = [
    "FlashMLABridgeUnavailableError",
    "FlashMLASparseForwardPlan",
    "plan_flashmla_sparse_forward",
    "flashmla_sparse_forward_wrapper",
    "flashmla_cudnn_sparse_attention_wrapper",
    "flashmla_sparse_score_recompute_wrapper",
]
