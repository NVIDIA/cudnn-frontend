# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Local CuTe kernels and autograd for simplified sparse attention.

Callers supply local queries, globally ordered keys, and physical row bounds.
Distributed communication, model projections, and packed-layout construction
belong to the calling framework.
"""

from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch

_SUPPORTED_QUERY_HEADS = (16, 32, 96)


_KV_HEADS = 1


_HEAD_DIM = 256


_INDEX_DIM = 128


_MATURE_TOPK = 512


_SUPPORTED_TOPKS = (512, 1024, 2048)


_KERNEL_MODULES: dict[str, object] = {}


# Opt-in debugging aid. The checks synchronize the host on every call, so the
# production path never runs them; they raise at the offending call instead.
_DEBUG_CHECKS = os.environ.get("SSA_DEBUG_CHECKS", "0") == "1"


_ATTENTION_FORWARD_KERNEL_VARIANTS = {
    (16, 512): ("attention_teacher_forward_hq16", "run"),
    (16, 1024): ("sparse_attention_large", "run"),
    (16, 2048): ("sparse_attention_large", "run"),
    (32, 512): ("attention_teacher_forward_hq32", "run"),
    (32, 1024): ("sparse_attention_large_h32_lookahead", "run"),
    (32, 2048): ("sparse_attention_large", "run"),
    (96, 512): ("attention_teacher_forward_hq96", "run"),
    (96, 1024): ("sparse_attention_h96_big", "run_group"),
    (96, 2048): ("sparse_attention_h96_big", "run_group"),
}


_QUALIFIED_KERNEL_VARIANTS = {
    "indexer_forward_topk": "indexer_forward_topk_tcgen05",
    "indexer_aux_grad_precompute": "indexer_aux_grad_precompute_tcgen05",
}


_QUALIFIED_ENV_DEFAULTS = {
    "FUSED_IKET": "0",
    "SSA_FILTERED_SELECT": "compact",
    "SSA_FUSED_CHECK": "0",
    "SSA_FUSED_FALLBACK_MODE": "rows",
    "SSA_FUSED_FAST_CNT": "1",
    "SSA_FUSED_FILTER_C_TILES": "14",
    "SSA_FUSED_FINE_BOUNDARY": "1",
    "SSA_FUSED_KMAJOR": "0",
    "SSA_FUSED_LOG_EVERY": "1",
    "SSA_FUSED_MARGIN": "0",
    "SSA_FUSED_MASK_WALK": "1",
    "SSA_FUSED_NOSYNC": "0",
    "SSA_FUSED_RAW_CMP": "1",
    "SSA_FUSED_SAMPLER": "k1",
    "SSA_FUSED_SAMPLE_MODE": "token",
    "SSA_FUSED_SAMPLE_RANK_DEN": "4",
    "SSA_FUSED_SAMPLE_RANK_NUM": "9",
    "SSA_FUSED_SAMPLE_SPAN_SKIP": "1",
    "SSA_FUSED_SAMPLE_STRIDE": "16",
    "SSA_FUSED_SPAN_SKIP": "1",
    "SSA_HQ16_PATH_DIAGNOSTIC": "0",
    "SSA_HQ32_DIAGNOSTIC_SINGLE_BF16_DS": "0",
    "SSA_SPMM_SCAT_ALIGN": "4",
    "SSA_SPMM_SCAT_WIN": "16",
    "SSA_TOPK": str(_MATURE_TOPK),
    "SPMM_IKET": "0",
    "TOPK_IKET": "0",
}


_QUALIFIED_ENV_UNSET = ("SSA_FUSED_TIE_CAP",)


# Optional diagnostics that leave every output unchanged may be switched on.
_QUALIFIED_ENV_CHOICES = {
    "SSA_FUSED_FALLBACK_KINDS": ("0", "1"),
    "SSA_FUSED_SCR_COUNTS": ("0", "1"),
}


_QUALIFIED_LOADED_MODULE_DEFAULTS = {
    "sparse_attention_backward_tcgen05": {
        # The low-level kernel contains additional directly qualified variants.
        # The production adapter remains fail-closed through the explicit
        # Hq/TopK forward map above; these values describe the immutable
        # capability set expected from the loaded backward file.
        "SUPPORTED_HQ": (16, 32, 96),
        "SUPPORTED_TOPK": (512, 1024, 2048),
    },
    "sparse_attention_backward_hq16": {"_PATH_DIAGNOSTIC": False},
    "sparse_attention_backward_hq32": {"_DIAGNOSTIC_SINGLE_BF16_DS": False},
    "indexer_aux_grad_precompute_tcgen05": {
        "TOPK": _MATURE_TOPK,
        "SCAT_WIN": 16,
        "SCAT_ALIGN": 4,
        "_SPMM_IKET": False,
    },
    "indexer_forward_topk_fused_tcgen05": {
        "_FUSED_IKET": False,
        "_TIE_CAP_OVERRIDE": None,
        "TIE_CAP": 0,
        "FILTER_C_TILES": 14,
        "_LOG_EVERY": 1,
        "SAMPLE_MARGIN": 0,
        "FINE_BOUNDARY": True,
        "SAMPLE_RANK_NUM": 9,
        "SAMPLE_RANK_DEN": 4,
        "SAMPLER": "k1",
        "K1_STRIDE": 64,
        "K1_KEEP_MIN": 24,
        "K1_RISK": 1e-6,
        "SAMPLE_MODE": "token",
        "SAMPLE_STRIDE": 16,
        "MASK_WALK": True,
        "FAST_CNT": True,
        "RAW_CMP": True,
        "SPAN_SKIP": True,
        "SAMPLE_SPAN_SKIP": True,
        "KMAJOR": False,
    },
    "cute_dsl_filter_topk.filtered_top_k_varlen_util": {"_TOPK_IKET": False},
}


@dataclass(frozen=True)
class RowMetadata:
    """Row metadata for the CuTe kernels' physical Q/K coordinate system."""

    key_starts: torch.Tensor
    key_ends: torch.Tensor
    query_valid: torch.Tensor
    loss_denominator: torch.Tensor
    triangular_scores: bool
    mask_invalid_rows: bool
    all_query_rows_valid: bool = False
    fuse_boundary_ops: bool = False


def _validate_production_environment() -> None:
    """Reject component-only tuning routes at the production adapter boundary."""
    invalid = []
    for name, expected in _QUALIFIED_ENV_DEFAULTS.items():
        value = os.environ.get(name)
        if value is not None and value != expected:
            invalid.append(f"{name}={value!r} (expected {expected!r})")
    for name in _QUALIFIED_ENV_UNSET:
        value = os.environ.get(name)
        if value is not None:
            invalid.append(f"{name}={value!r} (expected unset)")
    for name, allowed in _QUALIFIED_ENV_CHOICES.items():
        value = os.environ.get(name)
        if value is not None and value not in allowed:
            invalid.append(f"{name}={value!r} (expected one of {allowed!r})")

    module_prefix = f"{__package__}.kernels."
    for suffix, expected_attributes in _QUALIFIED_LOADED_MODULE_DEFAULTS.items():
        module = sys.modules.get(f"{module_prefix}{suffix}")
        if module is None:
            continue
        for attribute, expected in expected_attributes.items():
            value = getattr(module, attribute, None)
            if value != expected:
                invalid.append(f"loaded {suffix}.{attribute}={value!r} (expected {expected!r})")

    if invalid:
        raise RuntimeError(
            "Simplified sparse attention production routing rejects unqualified component environment " "overrides: " + "; ".join(sorted(invalid))
        )


def _kernel_module(name: str, *, query_heads: Optional[int] = None, topk: Optional[int] = None):
    """Load only the selected production kernel variants."""
    _validate_production_environment()
    key = _QUALIFIED_KERNEL_VARIANTS.get(name, name)
    if name == "attention_teacher_forward":
        route = (query_heads, topk)
        variant = _ATTENTION_FORWARD_KERNEL_VARIANTS.get(route)
        if variant is None:
            raise NotImplementedError(
                f"Simplified sparse attention has no {name} specialization for Hq={query_heads}, TopK={topk}; "
                f"supported routes are {tuple(_ATTENTION_FORWARD_KERNEL_VARIANTS)}."
            )
        key = variant[0]
    elif name == "sparse_attention_backward":
        if query_heads not in _SUPPORTED_QUERY_HEADS:
            raise NotImplementedError(
                f"Simplified sparse attention has no {name} specialization for Hq={query_heads}; " f"supported values are {_SUPPORTED_QUERY_HEADS}."
            )
        key = "sparse_attention_backward_tcgen05"
    module = _KERNEL_MODULES.get(key)
    if module is None:
        module = importlib.import_module(f"{__package__}.kernels.{key}")
        if name == "indexer_aux_grad_precompute" and getattr(module, "TOPK", None) != _MATURE_TOPK:
            raise RuntimeError(
                "Simplified sparse attention requires the auxiliary-gradient module to be specialized for "
                f"top-K {_MATURE_TOPK}, got {getattr(module, 'TOPK', None)!r}."
            )
        _KERNEL_MODULES[key] = module
        _validate_production_environment()
    return module


def _attention_forward_entry(*, query_heads: int, topk: int):
    module = _kernel_module("attention_teacher_forward", query_heads=query_heads, topk=topk)
    _, entry_name = _ATTENTION_FORWARD_KERNEL_VARIANTS[(query_heads, topk)]
    return getattr(module, entry_name)


@contextmanager
def _nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
) -> None:
    query_length = query.size(0)
    key_length = key.size(0)
    if query_length <= 0 or key_length <= 0:
        raise NotImplementedError("Simplified sparse attention requires nonempty query and key/value streams.")
    if value.size(0) != key_length or q_idx.size(0) != query_length or k_idx.size(0) != key_length:
        raise ValueError(
            "Simplified sparse attention Q/K/V/indexer row counts disagree: "
            f"Tq={query_length}, Tk={key_length}, V={value.size(0)}, "
            f"Iq={q_idx.size(0)}, Ik={k_idx.size(0)}."
        )
    tensors = (query, key, value, q_idx, k_idx)
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        raise NotImplementedError(
            "Simplified sparse attention requires BF16 Q/K/V/indexer activations, got " f"{tuple(str(tensor.dtype) for tensor in tensors)}."
        )
    if not query.is_cuda or any(tensor.device != query.device for tensor in tensors[1:]):
        raise NotImplementedError("Simplified sparse attention requires all inputs on one CUDA device.")
    if torch.cuda.get_device_capability(query.device) not in ((10, 0), (10, 3)):
        raise NotImplementedError(
            "Simplified sparse attention kernels support SM100 and SM103 only; got capability " f"{torch.cuda.get_device_capability(query.device)}."
        )
    if torch.cuda.is_current_stream_capturing():
        raise NotImplementedError("Simplified sparse attention is not CUDA-graph safe.")

    qkv_layouts = (
        ("query", query, query.size(1) * _HEAD_DIM, query_length),
        ("key", key, _HEAD_DIM, key_length),
        ("value", value, _HEAD_DIM, key_length),
    )
    for name, tensor, row_elements, rows in qkv_layouts:
        # Kernels index each operand only over that operand's token extent. A single bound based
        # on max(Tq,Tk) incorrectly rejects a compact H32 query when K is much longer than Q.
        max_token_stride = (2**31 - 1) // (2 * max(rows - 1, 1))
        compact_inner = tensor.stride(-1) == 1
        if name == "query":
            compact_inner = compact_inner and tensor.stride(-2) == _HEAD_DIM
        if (
            not compact_inner
            or tensor.stride(0) < row_elements
            or tensor.stride(0) % 8 != 0
            or tensor.stride(0) > max_token_stride
            or tensor.data_ptr() % 16 != 0
        ):
            raise NotImplementedError(
                f"Simplified sparse attention requires compact-inner aligned {name} rows; "
                f"shape={tuple(tensor.shape)}, stride={tensor.stride()}, "
                f"max_token_stride={max_token_stride}."
            )


def _normalize_layout(
    layout: RowMetadata, *, query_length: int, key_length: int, device: torch.device, topk: int
) -> tuple[RowMetadata, torch.Tensor, torch.Tensor]:
    starts = layout.key_starts.to(device=device, dtype=torch.int32).reshape(-1).contiguous()
    ends = layout.key_ends.to(device=device, dtype=torch.int32).reshape(-1).contiguous()
    valid = layout.query_valid.to(device=device, dtype=torch.bool).reshape(-1).contiguous()
    denominator = layout.loss_denominator.to(device=device, dtype=torch.float32).reshape(1)
    if starts.shape != (query_length,) or ends.shape != (query_length,):
        raise ValueError("Simplified sparse attention key_starts/key_ends must have shape [Tq].")
    if valid.shape != (query_length,):
        raise ValueError("Simplified sparse attention query-valid mask must have shape [Tq].")
    if layout.triangular_scores and query_length != key_length:
        raise ValueError("Triangular Simplified sparse attention scores require Tq == Tk.")
    if starts.device.type == "cpu":
        if not bool(((starts >= 0) & (starts <= ends) & (ends <= key_length)).all()):
            raise ValueError("Simplified sparse attention received out-of-range row bounds.")
        if layout.triangular_scores:
            row = torch.arange(query_length, dtype=starts.dtype, device=device)
            if not bool((ends <= row + 1).all()):
                raise ValueError("Triangular Simplified sparse attention bounds extend above the causal diagonal.")

    effective_ends = ends if layout.all_query_rows_valid else torch.where(valid, ends, starts)
    raw_lengths = (effective_ends - starts).clamp(min=0, max=topk).contiguous()
    attention_lengths = raw_lengths.clamp_min(1).contiguous()
    return (
        RowMetadata(
            starts,
            effective_ends,
            valid,
            denominator,
            layout.triangular_scores,
            layout.mask_invalid_rows,
            layout.all_query_rows_valid,
            layout.fuse_boundary_ops,
        ),
        raw_lengths,
        attention_lengths,
    )


def _check_finite_indexer_inputs(q_idx: torch.Tensor, k_idx: torch.Tensor) -> None:
    """SSA_DEBUG_CHECKS: reject NaN/Inf indexer activations before top-k."""
    problems = []
    for name, tensor in (("q_idx", q_idx), ("k_idx", k_idx)):
        bad = ~torch.isfinite(tensor).all(dim=1)
        count = int(bad.sum())
        if count:
            first = int(torch.nonzero(bad)[0])
            problems.append(f"{count}/{tensor.size(0)} {name} rows contain NaN/Inf (first row {first})")
    if problems:
        raise RuntimeError("Simplified sparse attention debug check: " + "; ".join(problems) + ".")


def _check_index_bounds(indices: torch.Tensor, layout: RowMetadata) -> None:
    """SSA_DEBUG_CHECKS: reject selected ids outside their row's key interval before the gather.

    Valid top-k output is ascending inside [start, end) and padded with
    clamp(end-1, 0), which also covers empty rows.
    """
    upper = (layout.key_ends - 1).clamp_min(0)[:, None]
    lower = torch.minimum(layout.key_starts[:, None], upper)
    bad = ((indices < lower) | (indices > upper)).any(dim=1) & layout.query_valid
    count = int(bad.sum())
    if count:
        row = int(torch.nonzero(bad)[0])
        values = indices[row][(indices[row] < lower[row]) | (indices[row] > upper[row])]
        raise RuntimeError(
            f"Simplified sparse attention debug check: top-k returned ids outside the key "
            f"bounds in {count} rows (first row {row}, bounds [{int(layout.key_starts[row])}, "
            f"{int(layout.key_ends[row])}), ids {values[:4].tolist()})."
        )


def _run_sparse_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    layout: RowMetadata,
    topk: int,
    kv_ready_event: torch.cuda.Event | None = None,
):
    _validate_inputs(query, key, value, q_idx, k_idx)
    layout, raw_lengths, attention_lengths = _normalize_layout(layout, query_length=query.size(0), key_length=key.size(0), device=query.device, topk=topk)
    indices = torch.empty((query.size(0), topk), device=query.device, dtype=torch.int32)
    predict = torch.empty_like(indices, dtype=torch.float32)
    output = torch.empty_like(query)
    lse = torch.empty((query.size(0), query.size(1)), device=query.device, dtype=torch.float32)
    teacher = torch.empty_like(predict)
    softmax_max = torch.empty_like(lse)
    softmax_inv_sum = torch.empty_like(lse)

    if _DEBUG_CHECKS:
        _check_finite_indexer_inputs(q_idx, k_idx)
    with _nvtx_range("cudnn/simplified_sparse_attention/indexer_forward_topk"):
        _kernel_module("indexer_forward_topk").run(
            q_idx,
            k_idx,
            indices,
            predict,
            layout.key_starts,
            layout.key_ends,
            triangular=layout.triangular_scores,
            score_impl="auto",
            fusion="auto",
            topk_impl="auto",
            topk_overflow_policy="REREAD_ALWAYS",
        )
    if layout.mask_invalid_rows:
        # Attention kernels use a one-slot sentinel for physical padding rows.
        # Make every gather safe even if a top-k implementation leaves an empty
        # row unwritten; raw_lengths remains zero for the auxiliary loss.
        indices.masked_fill_(~layout.query_valid[:, None], 0)
    if _DEBUG_CHECKS:
        _check_index_bounds(indices, layout)
    if kv_ready_event is not None:
        # The producer event covers communication and any layout permutation.
        # Indexer work does not read main K/V and can run before this join.
        torch.cuda.current_stream(query.device).wait_event(kv_ready_event)
    with _nvtx_range("cudnn/simplified_sparse_attention/sparse_attention_forward"):
        _attention_forward_entry(query_heads=query.size(1), topk=topk)(
            query,
            key,
            value,
            indices,
            attention_lengths,
            output,
            lse,
            teacher,
            softmax_max,
            softmax_inv_sum,
        )
    if layout.mask_invalid_rows:
        output.masked_fill_(~layout.query_valid.view(-1, 1, 1), 0.0)
    return (
        output,
        lse,
        teacher,
        softmax_max,
        softmax_inv_sum,
        indices,
        raw_lengths,
        attention_lengths,
        predict,
        layout,
    )


class _SparseAttentionTrainingFunction(torch.autograd.Function):
    """Sparse QKV backward plus precomputed indexer-loss gradients."""

    @staticmethod
    def forward(ctx, query, key, value, q_idx, k_idx, layout, indexer_topk, loss_coeff, kv_ready_event=None):
        (
            output,
            lse,
            teacher,
            _softmax_max,
            _softmax_inv_sum,
            indices,
            lengths,
            attention_lengths,
            predict,
            layout,
        ) = _run_sparse_forward(
            query,
            key,
            value,
            q_idx,
            k_idx,
            layout,
            indexer_topk,
            **({} if kv_ready_event is None else {"kv_ready_event": kv_ready_event}),
        )

        indexer_loss = torch.empty((), device=query.device, dtype=torch.float32)
        unit_dq_idx = torch.empty_like(q_idx)
        ctx.fuse_boundary_ops = layout.fuse_boundary_ops
        ctx.all_query_rows_valid = layout.all_query_rows_valid
        ctx.defer_dk = (
            layout.fuse_boundary_ops and layout.all_query_rows_valid and query.shape[0] == 8192 and key.shape[0] in (524288, 1048576) and indexer_topk == 2048
        )
        # The deferred key gradient keeps only its key-major edge records, so the
        # full-length unit gradient (Tk x 128) is not allocated on that path.
        unit_dk_idx = None if ctx.defer_dk else torch.empty_like(k_idx)
        with _nvtx_range("cudnn/simplified_sparse_attention/indexer_aux_grad_precompute"):
            deferred_dk = _kernel_module("indexer_aux_grad_precompute").run(
                q_idx,
                k_idx,
                indices,
                lengths,
                predict,
                teacher,
                indexer_loss,
                unit_dq_idx,
                unit_dk_idx,
                layout.loss_denominator,
                loss_coeff=loss_coeff,
                rectangular=not layout.triangular_scores,
                spmm_impl="auto",
                **({"defer_dk": True} if ctx.defer_dk else {}),
            )
        if (deferred_dk is not None) != ctx.defer_dk:
            raise RuntimeError("SSA indexer aux kernel did not honor the deferred key gradient")
        if not layout.all_query_rows_valid:
            unit_dq_idx.masked_fill_(~layout.query_valid.view(-1, 1), 0.0)
        ctx.save_for_backward(
            query,
            key,
            value,
            output,
            lse,
            indices,
            attention_lengths,
            layout.query_valid,
            unit_dq_idx,
            # Saved, not held on ctx, so autograd frees the deferred edge records with
            # this node's other saved tensors.
            *(deferred_dk.tensors if ctx.defer_dk else (unit_dk_idx,)),
        )
        ctx.set_materialize_grads(False)
        return output, indexer_loss

    @staticmethod
    def backward(ctx, grad_output, grad_indexer_loss):
        (
            query,
            key,
            value,
            output,
            lse,
            indices,
            attention_lengths,
            query_valid,
            unit_dq_idx,
            *key_gradient_state,
        ) = ctx.saved_tensors
        grad_query = grad_key = grad_value = None
        if grad_output is not None and any(ctx.needs_input_grad[:3]):
            grad_output = grad_output.contiguous().view_as(query)
            if not ctx.all_query_rows_valid:
                # contiguous() may preserve caller-owned storage; never mask it in place.
                grad_output = grad_output.masked_fill(~query_valid.view(-1, 1, 1), 0.0)
            dq = torch.empty(query.shape, device=query.device, dtype=query.dtype)
            dk = torch.empty(key.shape, device=key.device, dtype=key.dtype)
            dv = torch.empty(value.shape, device=value.device, dtype=value.dtype)
            with _nvtx_range("cudnn/simplified_sparse_attention/sparse_attention_backward"):
                backward = _kernel_module("sparse_attention_backward", query_heads=query.size(1))
                if ctx.fuse_boundary_ops and query.size(1) == 96:
                    backward = _kernel_module("sparse_attention_workspace")
                backward.run(
                    query,
                    key,
                    value,
                    output,
                    grad_output,
                    lse,
                    indices,
                    attention_lengths,
                    dq,
                    dk,
                    dv,
                )
            grad_query = dq if ctx.needs_input_grad[0] else None
            grad_key = dk if ctx.needs_input_grad[1] else None
            grad_value = dv if ctx.needs_input_grad[2] else None

        grad_q_idx = grad_k_idx = None
        if grad_indexer_loss is not None:
            if ctx.needs_input_grad[3]:
                grad_q_idx = unit_dq_idx * grad_indexer_loss.to(unit_dq_idx.dtype)
            if ctx.needs_input_grad[4]:
                if ctx.defer_dk:
                    aux = _kernel_module("indexer_aux_grad_precompute")
                    grad_k_idx = aux.DeferredKeyGradient.from_saved(*key_gradient_state).finish(grad_indexer_loss)
                else:
                    (unit_dk_idx,) = key_gradient_state
                    grad_k_idx = unit_dk_idx * grad_indexer_loss.to(unit_dk_idx.dtype)
        gradients = (grad_query, grad_key, grad_value, grad_q_idx, grad_k_idx, None, None, None)
        return gradients + (None,) if len(ctx.needs_input_grad) == 9 else gradients


def sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    layout: RowMetadata,
    topk: int,
    *,
    loss_coeff: float = 0.1,
    kv_ready_event: torch.cuda.Event | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute sparse attention and its selected-support indexer objective.

    Q is [Tq, Hq, 256], K/V are [Tk, 1, 256], and indexer Q/K are
    [Tq, 128]/[Tk, 128]. Inputs are BF16 on one SM100 or SM103 CUDA device. Hq is
    16, 32, or 96 and topk is 512, 1024, or 2048. Tq may differ from Tk;
    ``layout`` defines each query's allowed key interval and padding mask.

    The returned FP32 scalar already includes ``loss_coeff`` (default 0.1)
    and ``layout.loss_denominator``. Do not apply that
    coefficient a second time. The output has the same shape as Q; neither
    result performs context-parallel communication. ``kv_ready_event`` marks
    completion of an external K/V producer (including layout permutation). It
    is joined after indexer work, immediately before sparse attention. Q,
    indexer Q/K and metadata must already be ready on the caller stream.
    The caller must protect K/V allocator lifetimes on both streams.
    """
    _validate_production_environment()
    if topk not in _SUPPORTED_TOPKS:
        raise NotImplementedError(f"Simplified sparse attention requires top-K in {_SUPPORTED_TOPKS}, got {topk}.")
    if query.ndim != 3 or query.shape[1] not in _SUPPORTED_QUERY_HEADS or query.shape[2] != _HEAD_DIM:
        raise NotImplementedError(f"Simplified sparse attention requires Q [Tq,Hq,256] with Hq in {_SUPPORTED_QUERY_HEADS}.")
    for name, tensor in (("key", key), ("value", value)):
        if tensor.ndim != 3 or tensor.shape[1:] != (_KV_HEADS, _HEAD_DIM):
            raise NotImplementedError(f"Simplified sparse attention requires {name} [Tk,1,256].")
    for name, tensor in (("q_idx", q_idx), ("k_idx", k_idx)):
        if tensor.ndim != 2 or tensor.shape[1] != _INDEX_DIM:
            raise NotImplementedError(f"Simplified sparse attention requires {name} [T,128].")
    q_idx, k_idx = q_idx.contiguous(), k_idx.contiguous()
    args = (query, key, value, q_idx, k_idx, layout, topk, float(loss_coeff))
    if kv_ready_event is not None:
        args += (kv_ready_event,)
    return _SparseAttentionTrainingFunction.apply(*args)


def reserve_indexer(max_query_len: int, max_key_len: int, topk: int) -> None:
    """Allocate and warm the indexer's persistent state for known maxima.

    ``max_query_len``/``max_key_len`` are the largest ``q_idx``/``k_idx``
    row counts this process will pass to :func:`sparse_attention` with
    ``triangular_scores=False``. Call once per device before training, on the
    CUDA stream that will run :func:`sparse_attention`. Storage is sized to
    the maxima without headroom, and kernels and the repair graph are
    prepared, so calls within the maxima never reallocate or recapture.
    Without it, or above the maxima, state grows on demand. Extents that do
    not use persistent state allocate nothing.
    """
    _validate_production_environment()
    if topk not in _SUPPORTED_TOPKS:
        raise NotImplementedError(f"Simplified sparse attention requires top-K in {_SUPPORTED_TOPKS}, got {topk}.")
    if max_query_len <= 0 or max_key_len <= 0:
        raise ValueError(f"reserve_indexer requires positive extents, got {max_query_len}, {max_key_len}.")
    if torch.cuda.is_current_stream_capturing():
        raise NotImplementedError("Simplified sparse attention is not CUDA-graph safe.")
    _kernel_module("indexer_forward_topk").reserve(max_query_len, max_key_len, topk)


__all__ = ["RowMetadata", "reserve_indexer", "sparse_attention"]
