# SPDX-License-Identifier: BSD-3-Clause
"""APIBase adapters and allocating wrappers for Flex Attention execution."""

from __future__ import annotations

import math
from collections import OrderedDict
from contextlib import nullcontext
from typing import Optional

from cuda.bindings import driver as cuda
import torch

from cudnn.api_base import APIBase, TensorDesc, TupleDict
from cudnn.flex_attention.dispatch import (
    _BwdPreallocated,
    _block_sparse_runtime_tuple,
    _compile_flex_attn_fwd,
    _flex_attn_bwd,
    _launch_flex_attn_fwd,
    _prepare_flex_attn_fwd,
)
from cudnn.flex_attention.plan.mask_plan import MaskPlan, validate_arbitrary_attention_plan
from cudnn.flex_attention.plan.validation import validate_call_options
from cudnn.flex_attention.runtime.dsl_utils import _is_aligned_layout

_CACHE_CAPACITY = 64
_WORKSPACE_ALIGNMENT = 128
_FWD_CACHE: OrderedDict = OrderedDict()
_BWD_CACHE: OrderedDict = OrderedDict()


def _cache_get(cache: OrderedDict, key):
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _cache_put(cache: OrderedDict, key, value) -> None:
    cache[key] = value
    cache.move_to_end(key)
    if len(cache) > _CACHE_CAPACITY:
        cache.popitem(last=False)


def _tensor_signature(tensor: Optional[torch.Tensor]):
    if tensor is None:
        return None
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def _plan_signature(mask_plan: MaskPlan) -> tuple:
    packed_plan, _, _ = mask_plan._runtime_args

    def signature(plan):
        return validate_arbitrary_attention_plan(block_sparse_tensors=plan).compile_key if plan is not None else None

    metadata = mask_plan.metadata
    return (
        metadata.mode,
        metadata.arch,
        metadata.device,
        metadata.dtype,
        metadata.batch_size,
        metadata.total_q,
        metadata.total_k,
        metadata.max_seqlen_q,
        metadata.max_seqlen_k,
        metadata.num_q_heads,
        metadata.num_kv_heads,
        metadata.head_dim,
        metadata.head_dim_v,
        metadata.hmask,
        metadata.pack_gqa,
        metadata.has_backward,
        signature(packed_plan),
        signature(getattr(packed_plan, "bwd_tensors", None)),
        signature(getattr(packed_plan, "dq_tensors", None)),
        packed_plan.narrow_workset,
    )


def _sequence_args(mask_plan: MaskPlan) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[int], Optional[int]]:
    _, cu_seqlens_q, cu_seqlens_k = mask_plan._runtime_args
    metadata = mask_plan.metadata
    if not mask_plan._is_varlen:
        return None, None, None, None
    return cu_seqlens_q, cu_seqlens_k, metadata.max_seqlen_q, metadata.max_seqlen_k


def _as_torch_stream(stream: cuda.CUstream | torch.cuda.Stream, device: torch.device) -> torch.cuda.Stream:
    if isinstance(stream, torch.cuda.Stream):
        if stream.device != device:
            raise ValueError(f"stream must be on {device}, got {stream.device}")
        return stream
    handle = int(stream)
    if handle in (0, 1, 2):
        return torch.cuda.default_stream(device)
    return torch.cuda.ExternalStream(handle, device=device)


def _stream_context(stream: Optional[cuda.CUstream | torch.cuda.Stream], device: torch.device):
    if stream is None:
        return nullcontext()
    return torch.cuda.stream(_as_torch_stream(stream, device))


def _plan_tensors(mask_plan: MaskPlan) -> tuple[torch.Tensor, ...]:
    packed_plan, cu_seqlens_q, cu_seqlens_k = mask_plan._runtime_args
    tensors = [cu_seqlens_q, cu_seqlens_k]
    for plan in (packed_plan, getattr(packed_plan, "bwd_tensors", None), getattr(packed_plan, "dq_tensors", None)):
        if plan is not None:
            tensors.extend(_block_sparse_runtime_tuple(plan))
    return tuple(tensor for tensor in tensors if isinstance(tensor, torch.Tensor))


def _record_streams(
    tensors: tuple[Optional[torch.Tensor], ...],
    stream: Optional[cuda.CUstream | torch.cuda.Stream],
    device: torch.device,
) -> None:
    if stream is None:
        return
    consumer = _as_torch_stream(stream, device)
    for tensor in tensors:
        if tensor is not None and tensor.is_cuda:
            tensor.record_stream(consumer)


def _align_workspace_bytes(nbytes: int) -> int:
    return -(-int(nbytes) // _WORKSPACE_ALIGNMENT) * _WORKSPACE_ALIGNMENT


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for extent in shape:
        result *= int(extent)
    return result


def _workspace_bytes(specs: tuple) -> int:
    total = 0
    for _, shape, dtype in specs:
        if shape is not None:
            total += _align_workspace_bytes(_numel(shape) * dtype.itemsize)
    return total


class _WorkspaceCarver:
    def __init__(self, workspace: Optional[torch.Tensor], required_bytes: int, device: torch.device, owner: str):
        if required_bytes == 0:
            if workspace is not None and workspace.device != device:
                raise ValueError(f"{owner} workspace must be on {device}")
            self._flat = None
            self._offset = 0
            return
        if workspace is None:
            raise ValueError(f"{owner} requires a {required_bytes}-byte uint8 CUDA workspace")
        if workspace.device != device or not workspace.is_cuda:
            raise ValueError(f"{owner} workspace must be a CUDA tensor on {device}")
        if not workspace.is_contiguous():
            raise ValueError(f"{owner} workspace must be contiguous")
        try:
            flat = workspace.view(torch.uint8).view(-1)
        except RuntimeError as exc:
            raise ValueError(f"{owner} workspace must provide a contiguous byte view") from exc
        if flat.numel() < required_bytes:
            raise ValueError(f"{owner} requires {required_bytes} workspace bytes, got {flat.numel()}")
        if flat.data_ptr() % _WORKSPACE_ALIGNMENT != 0:
            raise ValueError(f"{owner} workspace must be {_WORKSPACE_ALIGNMENT}-byte aligned")
        self._flat = flat
        self._offset = 0

    def take(self, shape: Optional[tuple[int, ...]], dtype: torch.dtype) -> Optional[torch.Tensor]:
        if shape is None:
            return None
        nbytes = _numel(shape) * dtype.itemsize
        start = self._offset
        end = start + nbytes
        self._offset += _align_workspace_bytes(nbytes)
        return self._flat[start:end].view(dtype).view(shape)


def _validate_runtime_tensor(tensor: Optional[torch.Tensor], desc: Optional[TensorDesc], name: str, align_bytes: int = 16) -> None:
    if (tensor is None) != (desc is None):
        raise ValueError(f"{name} presence must match the compiled configuration")
    if tensor is None:
        return
    if tuple(tensor.shape) != tuple(desc.shape) or tuple(tensor.stride()) != tuple(desc.stride) or tensor.dtype != desc.dtype or tensor.device != desc.device:
        raise ValueError(f"{name} shape, stride, dtype, and device must match the compiled descriptor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not _is_aligned_layout(tensor, align_bytes):
        raise ValueError(f"{name} must use a native {align_bytes}-byte-aligned layout; implicit copies are not supported")


def _validate_plan(mask_plan: MaskPlan) -> None:
    if not isinstance(mask_plan, MaskPlan):
        raise TypeError("mask_plan must be returned by create_mask_plan")


def _resolve_scale(softmax_scale: Optional[float], head_dim: int) -> float:
    validate_call_options(softmax_scale=softmax_scale, deterministic=False, return_lse=False)
    return 1.0 / math.sqrt(head_dim) if softmax_scale is None else float(softmax_scale)


class FlexAttentionFwd(APIBase):
    """Explicit compile/execute API for reusable-plan Flex Attention forward."""

    def __init__(
        self,
        sample_q: torch.Tensor,
        sample_k: torch.Tensor,
        sample_v: torch.Tensor,
        sample_o: torch.Tensor,
        sample_mask_plan: MaskPlan,
        sample_lse: Optional[torch.Tensor] = None,
        *,
        sm90_use_smem_mask_pipeline: bool = True,
    ) -> None:
        super().__init__()
        _validate_plan(sample_mask_plan)
        self._warn_experimental_api()
        self.q_desc = self._make_tensor_desc(sample_q, name="q")
        self.k_desc = self._make_tensor_desc(sample_k, name="k")
        self.v_desc = self._make_tensor_desc(sample_v, name="v")
        self.o_desc = self._make_tensor_desc(sample_o, name="o")
        self.lse_desc = self._make_tensor_desc(sample_lse, name="lse")
        self._sample_q = sample_q
        self._sample_k = sample_k
        self._sample_v = sample_v
        self._sample_o = sample_o
        self._sample_lse = sample_lse
        self._sample_mask_plan = sample_mask_plan
        self._sm90_use_smem_mask_pipeline = sm90_use_smem_mask_pipeline
        self._dispatch_compile_key = None
        self.workspace_size = 0

    def _prepare(self, q, k, v, mask_plan):
        packed_plan, _, _ = mask_plan._runtime_args
        cu_q, cu_k, max_q, max_k = _sequence_args(mask_plan)
        return _prepare_flex_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            pack_gqa=mask_plan.metadata.pack_gqa,
            block_sparse_tensors=packed_plan,
            has_lse=self.lse_desc is not None,
            sm90_use_smem_mask_pipeline=self._sm90_use_smem_mask_pipeline,
        )

    def check_support(self) -> bool:
        if self._is_supported:
            return True
        self._sample_mask_plan._validate_runtime(self._sample_q, self._sample_k, self._sample_v)
        for name, tensor in (("q", self._sample_q), ("k", self._sample_k), ("v", self._sample_v), ("o", self._sample_o)):
            _validate_runtime_tensor(tensor, getattr(self, f"{name}_desc"), name)
        _validate_runtime_tensor(self._sample_lse, self.lse_desc, "lse", align_bytes=4)
        dispatch = self._prepare(self._sample_q, self._sample_k, self._sample_v, self._sample_mask_plan)
        if dispatch.total_q == 0 or dispatch.total_k == 0:
            raise ValueError("FlexAttentionFwd APIBase requires non-empty sample tensors")
        if tuple(self._sample_o.shape) != dispatch.output_shape or self._sample_o.dtype != self._sample_q.dtype:
            raise ValueError(f"o must have shape {dispatch.output_shape} and dtype {self._sample_q.dtype}")
        if self._sample_lse is not None and (tuple(self._sample_lse.shape) != dispatch.lse_shape or self._sample_lse.dtype != torch.float32):
            raise ValueError(f"lse must have shape {dispatch.lse_shape} and dtype torch.float32")
        self._dispatch_compile_key = dispatch.compile_key
        self.workspace_size = _WORKSPACE_ALIGNMENT if dispatch.arch == 90 else 0
        self._is_supported = True
        return True

    def compile(self) -> None:
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        dispatch = self._prepare(self._sample_q, self._sample_k, self._sample_v, self._sample_mask_plan)
        cu_q, cu_k, _, _ = _sequence_args(self._sample_mask_plan)
        scheduler = torch.empty((1,), dtype=torch.int32, device=self._sample_q.device) if dispatch.arch == 90 else None
        self._compiled_kernel = _compile_flex_attn_fwd(
            dispatch,
            self._sample_q,
            self._sample_k,
            self._sample_v,
            self._sample_o,
            self._sample_lse,
            1.0 / math.sqrt(dispatch.head_dim),
            cu_q,
            cu_k,
            scheduler,
        )
        self._sample_q = self._sample_k = self._sample_v = None
        self._sample_o = self._sample_lse = self._sample_mask_plan = None

    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        mask_plan: MaskPlan,
        lse_tensor: Optional[torch.Tensor] = None,
        *,
        workspace: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        current_stream: Optional[cuda.CUstream | torch.cuda.Stream] = None,
    ) -> None:
        if self._compiled_kernel is None:
            raise RuntimeError("FlexAttentionFwd kernel is not compiled")
        _validate_plan(mask_plan)
        for name, tensor in (("q", q_tensor), ("k", k_tensor), ("v", v_tensor), ("o", o_tensor)):
            _validate_runtime_tensor(tensor, getattr(self, f"{name}_desc"), name)
        _validate_runtime_tensor(lse_tensor, self.lse_desc, "lse", align_bytes=4)
        mask_plan._validate_runtime(q_tensor, k_tensor, v_tensor)
        dispatch = self._prepare(q_tensor, k_tensor, v_tensor, mask_plan)
        if dispatch.compile_key != self._dispatch_compile_key:
            raise ValueError("runtime tensors or MaskPlan do not match the compiled Flex Attention forward configuration")
        carver = _WorkspaceCarver(workspace, self.workspace_size, q_tensor.device, "FlexAttentionFwd")
        scheduler = carver.take((1,), torch.int32) if dispatch.arch == 90 else None
        cu_q, cu_k, _, _ = _sequence_args(mask_plan)
        scale = _resolve_scale(softmax_scale, dispatch.head_dim)
        with torch.cuda.device(q_tensor.device), _stream_context(current_stream, q_tensor.device):
            _launch_flex_attn_fwd(
                dispatch,
                self._compiled_kernel,
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                lse_tensor,
                scale,
                cu_q,
                cu_k,
                scheduler,
            )
        _record_streams(
            (q_tensor, k_tensor, v_tensor, o_tensor, lse_tensor, workspace, *_plan_tensors(mask_plan)),
            current_stream,
            q_tensor.device,
        )


class FlexAttentionBwd(APIBase):
    """Explicit compile/execute API for reusable-plan Flex Attention backward."""

    def __init__(
        self,
        sample_q: torch.Tensor,
        sample_k: torch.Tensor,
        sample_v: torch.Tensor,
        sample_o: torch.Tensor,
        sample_do: torch.Tensor,
        sample_lse: torch.Tensor,
        sample_dq: torch.Tensor,
        sample_dk: torch.Tensor,
        sample_dv: torch.Tensor,
        sample_mask_plan: MaskPlan,
        sample_dlse: Optional[torch.Tensor] = None,
        *,
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        _validate_plan(sample_mask_plan)
        self._warn_experimental_api()
        for name, tensor in (
            ("q", sample_q),
            ("k", sample_k),
            ("v", sample_v),
            ("o", sample_o),
            ("do", sample_do),
            ("lse", sample_lse),
            ("dq", sample_dq),
            ("dk", sample_dk),
            ("dv", sample_dv),
            ("dlse", sample_dlse),
        ):
            setattr(self, f"{name}_desc", self._make_tensor_desc(tensor, name=name))
            setattr(self, f"_sample_{name}", tensor)
        self._sample_mask_plan = sample_mask_plan
        self.deterministic = deterministic
        self.workspace_size = 0

    def _call_dispatch(self, *, compile_only=False, validate_only=False, preallocated=None, compiled=None, compile_outputs=None, **runtime):
        mask_plan = runtime.pop("mask_plan")
        packed_plan, _, _ = mask_plan._runtime_args
        cu_q, cu_k, max_q, max_k = _sequence_args(mask_plan)
        return _flex_attn_bwd(
            runtime["q"],
            runtime["k"],
            runtime["v"],
            runtime["o"],
            runtime["do"],
            runtime["lse"],
            softmax_scale=runtime.get("softmax_scale"),
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            deterministic=self.deterministic,
            block_sparse_tensors=packed_plan,
            dlse=runtime.get("dlse"),
            _preallocated=preallocated,
            _compiled=compiled,
            _compile_only=compile_only,
            _native_inputs=True,
            _validate_only=validate_only,
            _compile_outputs=compile_outputs,
        )

    def check_support(self) -> bool:
        if self._is_supported:
            return True
        if type(self.deterministic) is not bool:
            raise TypeError("deterministic must be a bool")
        self._sample_mask_plan._validate_runtime(self._sample_q, self._sample_k, self._sample_v)
        align_bytes = 128 if self._sample_q.shape[-1] == 256 and self._sample_v.shape[-1] == 256 else 16
        for name in ("q", "k", "v", "o", "do", "dq", "dk", "dv"):
            _validate_runtime_tensor(getattr(self, f"_sample_{name}"), getattr(self, f"{name}_desc"), name, align_bytes=align_bytes)
        _validate_runtime_tensor(self._sample_lse, self.lse_desc, "lse", align_bytes=4)
        _validate_runtime_tensor(self._sample_dlse, self.dlse_desc, "dlse", align_bytes=4)
        if self._sample_dq.shape != self._sample_q.shape or self._sample_dq.dtype != self._sample_q.dtype:
            raise ValueError("dq must match q shape and dtype")
        if self._sample_dk.shape != self._sample_k.shape or self._sample_dk.dtype != self._sample_k.dtype:
            raise ValueError("dk must match k shape and dtype")
        if self._sample_dv.shape != self._sample_v.shape or self._sample_dv.dtype != self._sample_v.dtype:
            raise ValueError("dv must match v shape and dtype")
        if self._sample_q.numel() == 0 or self._sample_k.numel() == 0:
            raise ValueError("FlexAttentionBwd APIBase requires non-empty sample tensors")
        self._call_dispatch(
            validate_only=True,
            q=self._sample_q,
            k=self._sample_k,
            v=self._sample_v,
            o=self._sample_o,
            do=self._sample_do,
            lse=self._sample_lse,
            dlse=self._sample_dlse,
            mask_plan=self._sample_mask_plan,
        )
        self._is_supported = True
        return True

    def compile(self) -> None:
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        _, _, _, compiled = self._call_dispatch(
            compile_only=True,
            compile_outputs=(self._sample_dq, self._sample_dk, self._sample_dv),
            q=self._sample_q,
            k=self._sample_k,
            v=self._sample_v,
            o=self._sample_o,
            do=self._sample_do,
            lse=self._sample_lse,
            dlse=self._sample_dlse,
            mask_plan=self._sample_mask_plan,
        )
        self._compiled_kernel = compiled
        self.workspace_size = _workspace_bytes(compiled.workspace_specs)
        for name in ("q", "k", "v", "o", "do", "lse", "dq", "dk", "dv", "dlse"):
            setattr(self, f"_sample_{name}", None)
        self._sample_mask_plan = None

    def _carve_workspace(self, workspace: Optional[torch.Tensor], device: torch.device, dq, dk, dv) -> _BwdPreallocated:
        carver = _WorkspaceCarver(workspace, self.workspace_size, device, "FlexAttentionBwd")
        values = {name: carver.take(shape, dtype) for name, shape, dtype in self._compiled_kernel.workspace_specs}
        return _BwdPreallocated(dq=dq, dk=dk, dv=dv, **values)

    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        do_tensor: torch.Tensor,
        lse_tensor: torch.Tensor,
        dq_tensor: torch.Tensor,
        dk_tensor: torch.Tensor,
        dv_tensor: torch.Tensor,
        mask_plan: MaskPlan,
        dlse_tensor: Optional[torch.Tensor] = None,
        *,
        workspace: Optional[torch.Tensor],
        softmax_scale: Optional[float] = None,
        current_stream: Optional[cuda.CUstream | torch.cuda.Stream] = None,
    ) -> None:
        if self._compiled_kernel is None:
            raise RuntimeError("FlexAttentionBwd kernels are not compiled")
        _validate_plan(mask_plan)
        align_bytes = 128 if q_tensor.shape[-1] == 256 and v_tensor.shape[-1] == 256 else 16
        runtime_tensors = {
            "q": q_tensor,
            "k": k_tensor,
            "v": v_tensor,
            "o": o_tensor,
            "do": do_tensor,
            "lse": lse_tensor,
            "dq": dq_tensor,
            "dk": dk_tensor,
            "dv": dv_tensor,
            "dlse": dlse_tensor,
        }
        for name, tensor in runtime_tensors.items():
            _validate_runtime_tensor(tensor, getattr(self, f"{name}_desc"), name, align_bytes=4 if name in ("lse", "dlse") else align_bytes)
        mask_plan._validate_runtime(q_tensor, k_tensor, v_tensor)
        preallocated = self._carve_workspace(workspace, q_tensor.device, dq_tensor, dk_tensor, dv_tensor)
        scale = _resolve_scale(softmax_scale, q_tensor.shape[-1])
        with torch.cuda.device(q_tensor.device), _stream_context(current_stream, q_tensor.device):
            self._call_dispatch(
                preallocated=preallocated,
                compiled=self._compiled_kernel,
                q=q_tensor,
                k=k_tensor,
                v=v_tensor,
                o=o_tensor,
                do=do_tensor,
                lse=lse_tensor,
                dlse=dlse_tensor,
                softmax_scale=scale,
                mask_plan=mask_plan,
            )
        _record_streams(
            (
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                do_tensor,
                lse_tensor,
                dq_tensor,
                dk_tensor,
                dv_tensor,
                dlse_tensor,
                workspace,
                *_plan_tensors(mask_plan),
            ),
            current_stream,
            q_tensor.device,
        )


def _flex_attention_forward(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    *,
    mask_plan: MaskPlan,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    stream: Optional[cuda.CUstream | torch.cuda.Stream] = None,
) -> TupleDict:
    """Allocate outputs and execute reusable-plan Flex Attention forward."""

    _validate_plan(mask_plan)
    validate_call_options(softmax_scale=softmax_scale, deterministic=False, return_lse=return_lse)
    mask_plan._validate_runtime(q_tensor, k_tensor, v_tensor)
    metadata = mask_plan.metadata
    output_shape = (
        (metadata.total_q, metadata.num_q_heads, metadata.head_dim_v)
        if metadata.mode == "varlen"
        else (metadata.batch_size, metadata.total_q // metadata.batch_size, metadata.num_q_heads, metadata.head_dim_v)
    )
    lse_shape = (
        (metadata.num_q_heads, metadata.total_q)
        if metadata.mode == "varlen"
        else (metadata.batch_size, metadata.num_q_heads, metadata.total_q // metadata.batch_size)
    )
    needs_lse = return_lse or any(tensor.requires_grad for tensor in (q_tensor, k_tensor, v_tensor))
    with torch.cuda.device(q_tensor.device), _stream_context(stream, q_tensor.device):
        o_tensor = torch.empty(output_shape, dtype=q_tensor.dtype, device=q_tensor.device)
        lse_tensor = torch.empty(lse_shape, dtype=torch.float32, device=q_tensor.device) if needs_lse else None
    if metadata.total_q == 0 or metadata.total_k == 0:
        with torch.cuda.device(q_tensor.device), _stream_context(stream, q_tensor.device):
            o_tensor.zero_()
            if lse_tensor is not None:
                lse_tensor.fill_(float("-inf"))
        return TupleDict(o_tensor=o_tensor, lse_tensor=lse_tensor)

    key = (
        _tensor_signature(q_tensor),
        _tensor_signature(k_tensor),
        _tensor_signature(v_tensor),
        _tensor_signature(o_tensor),
        _tensor_signature(lse_tensor),
        _plan_signature(mask_plan),
    )
    api = _cache_get(_FWD_CACHE, key)
    if api is None:
        api = FlexAttentionFwd(q_tensor, k_tensor, v_tensor, o_tensor, mask_plan, lse_tensor)
        api.check_support()
        api.compile()
        _cache_put(_FWD_CACHE, key, api)
    with torch.cuda.device(q_tensor.device), _stream_context(stream, q_tensor.device):
        workspace = torch.empty((api.workspace_size,), dtype=torch.uint8, device=q_tensor.device) if api.workspace_size else None
    api.execute(
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        mask_plan,
        lse_tensor,
        workspace=workspace,
        softmax_scale=softmax_scale,
        current_stream=stream,
    )
    return TupleDict(o_tensor=o_tensor, lse_tensor=lse_tensor)


def _flex_attention_backward(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    o_tensor: torch.Tensor,
    do_tensor: torch.Tensor,
    lse_tensor: torch.Tensor,
    *,
    mask_plan: MaskPlan,
    softmax_scale: Optional[float] = None,
    deterministic: bool = False,
    dlse_tensor: Optional[torch.Tensor] = None,
    stream: Optional[cuda.CUstream | torch.cuda.Stream] = None,
) -> TupleDict:
    """Allocate gradients/workspace and execute reusable-plan Flex Attention backward."""

    _validate_plan(mask_plan)
    validate_call_options(softmax_scale=softmax_scale, deterministic=deterministic, return_lse=False)
    mask_plan._validate_runtime(q_tensor, k_tensor, v_tensor)
    with torch.cuda.device(q_tensor.device), _stream_context(stream, q_tensor.device):
        dq_tensor = torch.empty_like(q_tensor, memory_format=torch.preserve_format)
        dk_tensor = torch.empty_like(k_tensor, memory_format=torch.preserve_format)
        dv_tensor = torch.empty_like(v_tensor, memory_format=torch.preserve_format)
    if q_tensor.numel() == 0 or k_tensor.numel() == 0:
        with torch.cuda.device(q_tensor.device), _stream_context(stream, q_tensor.device):
            dq_tensor.zero_()
            dk_tensor.zero_()
            dv_tensor.zero_()
        return TupleDict(dq_tensor=dq_tensor, dk_tensor=dk_tensor, dv_tensor=dv_tensor)

    key = (
        _tensor_signature(q_tensor),
        _tensor_signature(k_tensor),
        _tensor_signature(v_tensor),
        _tensor_signature(o_tensor),
        _tensor_signature(do_tensor),
        _tensor_signature(lse_tensor),
        _tensor_signature(dq_tensor),
        _tensor_signature(dk_tensor),
        _tensor_signature(dv_tensor),
        _tensor_signature(dlse_tensor),
        _plan_signature(mask_plan),
        deterministic,
    )
    api = _cache_get(_BWD_CACHE, key)
    if api is None:
        api = FlexAttentionBwd(
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            do_tensor,
            lse_tensor,
            dq_tensor,
            dk_tensor,
            dv_tensor,
            mask_plan,
            dlse_tensor,
            deterministic=deterministic,
        )
        api.check_support()
        api.compile()
        _cache_put(_BWD_CACHE, key, api)
    with torch.cuda.device(q_tensor.device), _stream_context(stream, q_tensor.device):
        workspace = torch.empty((api.workspace_size,), dtype=torch.uint8, device=q_tensor.device)
    api.execute(
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        do_tensor,
        lse_tensor,
        dq_tensor,
        dk_tensor,
        dv_tensor,
        mask_plan,
        dlse_tensor,
        workspace=workspace,
        softmax_scale=softmax_scale,
        current_stream=stream,
    )
    return TupleDict(dq_tensor=dq_tensor, dk_tensor=dk_tensor, dv_tensor=dv_tensor)


__all__ = [
    "FlexAttentionFwd",
    "FlexAttentionBwd",
]
