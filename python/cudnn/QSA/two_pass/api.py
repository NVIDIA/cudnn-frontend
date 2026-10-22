# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""APIBase wrapper for the GQA index-list sparse attention (QSA) backward on SM100 / SM103 -- ``qsa_bwd_sm100.py``."""

from __future__ import annotations

from typing import Optional

import torch
from cuda.bindings import driver as cuda

from cudnn._torch_stream import device_context, stream_context
from cudnn.api_base import APIBase, TupleDict, WorkspaceCarver
from cudnn.frost.buffers import cutedsl_requirement_error

GROUP_SIZE = 12  # query heads per KV head
NUM_KV_HEADS = (1, 2)  # 24 / 2 is the full model, 12 / 1 is one tensor-parallel rank of two
HEAD_DIM = 256
TOPK_WIDTHS = (2051, 2052)  # 512 blocks x 4 tokens + 3 tail tokens, or with a fourth, always empty tail slot
SOFTMAX_SCALE = HEAD_DIM**-0.5
_SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))


def _dsl_cuda_version() -> tuple[int, int]:
    from cutlass.base_dsl.version_info import CUDA_VERSION

    return CUDA_VERSION.major, CUDA_VERSION.minor


def _byte_span(tensor: torch.Tensor) -> tuple[int, int]:
    # Exact for the contiguous layouts check_support admits.
    return tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size()


def _overlaps(lhs: tuple[int, int], rhs: tuple[int, int]) -> bool:
    return lhs[0] < rhs[1] and rhs[0] < lhs[1]


class _QsaTwoPassBackward(APIBase):
    """QSA backward: dQ, dK and dV from Q, K, V, O, dO, the forward LSE and the selected token indices.

    The geometry is 12 query heads per KV head with 2 KV heads (the model) or 1 (one of two tensor-parallel ranks),
    read from the sample shapes. The T rows are one sequence, or with ``cu_seqlens`` (int32 ``[B + 1]`` prefix sums)
    B packed sequences of at most ``max_seq_len`` tokens that attend within themselves. ``topk_idxs`` row ``i`` lists
    the first token (its row in the packed buffer) of each of its ``min((p + 1) // 4, 512)`` complete 4-token blocks,
    p the row's position in its sequence (the four ids of a block are consecutive and the blocks start at the
    sequence's first token), then the ``(p + 1) % 4`` tail tokens, then ``-1``. Up to 2051 tokens every row is
    therefore its whole causal prefix. The kernels read the block ids only and derive the tail block from the
    position. Rows that break this layout are not checked. FP32 saved output pairs with head-major ``[Hq, T]`` LSE
    and is read natively; BF16 saved output pairs with the earlier token-major ``[T, Hq]`` LSE contract.
    """

    def __init__(
        self,
        sample_q: torch.Tensor,  # (T, 12 * H_kv, 256) bf16
        sample_k: torch.Tensor,  # (T, H_kv, 256) bf16, H_kv in (1, 2)
        sample_v: torch.Tensor,  # (T, H_kv, 256) bf16
        sample_out: torch.Tensor,  # (T, 12 * H_kv, 256) bf16 or fp32
        sample_dout: torch.Tensor,  # (T, 12 * H_kv, 256) bf16
        sample_lse: torch.Tensor,  # fp32 natural-log LSE: (Hq, T) with fp32 out, otherwise (T, Hq)
        sample_topk_idxs: torch.Tensor,  # (T, 2051) or (T, 2052) int32
        sample_dq: Optional[torch.Tensor] = None,
        sample_dk: Optional[torch.Tensor] = None,
        sample_dv: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        sample_cu_seqlens: Optional[torch.Tensor] = None,  # (B + 1,) int32 sequence bounds of the packed rows
        max_seq_len: Optional[int] = None,  # longest packed sequence, T when omitted
    ):
        """Capture the sample tensor contract. Missing output samples default to the input layouts."""
        super().__init__()
        self._warn_experimental_api()
        self.descs = {
            "q": self._make_tensor_desc(sample_q, name="sample_q"),
            "k": self._make_tensor_desc(sample_k, name="sample_k"),
            "v": self._make_tensor_desc(sample_v, name="sample_v"),
            "out": self._make_tensor_desc(sample_out, name="sample_out"),
            "dout": self._make_tensor_desc(sample_dout, name="sample_dout"),
            "lse": self._make_tensor_desc(sample_lse, name="sample_lse"),
            "topk_idxs": self._make_tensor_desc(sample_topk_idxs, name="sample_topk_idxs"),
            "dq": self._make_tensor_desc(sample_q if sample_dq is None else sample_dq, name="sample_dq"),
            "dk": self._make_tensor_desc(sample_k if sample_dk is None else sample_dk, name="sample_dk"),
            "dv": self._make_tensor_desc(sample_v if sample_dv is None else sample_dv, name="sample_dv"),
        }
        if sample_cu_seqlens is not None:
            self.descs["cu_seqlens"] = self._make_tensor_desc(sample_cu_seqlens, name="sample_cu_seqlens")
        self.softmax_scale = softmax_scale
        self.max_seq_len = max_seq_len
        self._kernel_module = None
        self._plan = None

    def check_support(self) -> bool:
        """Validate the DSL version, device, dtypes, shapes and layouts."""
        error = cutedsl_requirement_error(self.__class__.__name__)
        self._not_implemented_error_if(error is not None, error or "")

        q_desc = self.descs["q"]
        self._value_error_if(q_desc.device.type != "cuda" or q_desc.device.index is None, f"q must live on an indexed CUDA device, got {q_desc.device}")
        capability = torch.cuda.get_device_capability(q_desc.device)
        self._not_implemented_error_if(
            capability not in _SUPPORTED_CAPABILITIES,
            f"{self.__class__.__name__} requires SM100 or SM103, found SM{capability[0]}{capability[1]}",
        )
        self._not_implemented_error_if(
            self.softmax_scale not in (None, SOFTMAX_SCALE),
            f"{self.__class__.__name__} supports only softmax_scale {SOFTMAX_SCALE}, got {self.softmax_scale}",
        )
        k_desc = self.descs["k"]
        self._value_error_if(q_desc.ndim != 3 or k_desc.ndim != 3, f"q and k must be 3-D (T, heads, {HEAD_DIM}), got {q_desc.shape} and {k_desc.shape}")
        seq_len = q_desc.shape[0]
        num_kv_heads = k_desc.shape[1]
        self._not_implemented_error_if(
            num_kv_heads not in NUM_KV_HEADS,
            f"{self.__class__.__name__} supports {NUM_KV_HEADS} KV heads with {GROUP_SIZE} query heads each, got k shape {k_desc.shape}",
        )
        num_q_heads = GROUP_SIZE * num_kv_heads
        out_dtype = self.descs["out"].dtype
        self._value_error_if(out_dtype not in (torch.bfloat16, torch.float32), "out must be BF16 or FP32")
        fp32_saved_state = out_dtype == torch.float32
        idx_desc = self.descs["topk_idxs"]
        topk_width = idx_desc.shape[1] if idx_desc.ndim == 2 and idx_desc.shape[1] in TOPK_WIDTHS else TOPK_WIDTHS[0]
        expected = {
            "q": ((seq_len, num_q_heads, HEAD_DIM), torch.bfloat16),
            "k": ((seq_len, num_kv_heads, HEAD_DIM), torch.bfloat16),
            "v": ((seq_len, num_kv_heads, HEAD_DIM), torch.bfloat16),
            "out": ((seq_len, num_q_heads, HEAD_DIM), out_dtype),
            "dout": ((seq_len, num_q_heads, HEAD_DIM), torch.bfloat16),
            "lse": ((num_q_heads, seq_len) if fp32_saved_state else (seq_len, num_q_heads), torch.float32),
            "topk_idxs": ((seq_len, topk_width), torch.int32),
            "dq": ((seq_len, num_q_heads, HEAD_DIM), torch.bfloat16),
            "dk": ((seq_len, num_kv_heads, HEAD_DIM), torch.bfloat16),
            "dv": ((seq_len, num_kv_heads, HEAD_DIM), torch.bfloat16),
        }
        num_seqs = 0
        if "cu_seqlens" in self.descs:
            cu_desc = self.descs["cu_seqlens"]
            self._value_error_if(cu_desc.ndim != 1 or cu_desc.shape[0] < 2, f"cu_seqlens must be 1-D with at least 2 entries, got {cu_desc.shape}")
            num_seqs = cu_desc.shape[0] - 1
            expected["cu_seqlens"] = ((num_seqs + 1,), torch.int32)
        for name, (shape, dtype) in expected.items():
            desc = self.descs[name]
            self._value_error_if(desc.shape != shape, f"{name} must have shape {shape}, got {desc.shape}")
            self._check_dtype(desc, dtype, name=name)
            self._value_error_if(desc.device != q_desc.device, f"{name} must be on {q_desc.device}, got {desc.device}")
            # The kernels address the declared layout natively and never repack (python/cudnn/AGENTS.md Rule 2).
            self._not_implemented_error_if(
                not desc.is_contiguous(),
                f"{self.__class__.__name__} addresses only contiguous {name}, got shape {desc.shape} strides {desc.stride}",
            )
        max_seq_len = seq_len if self.max_seq_len is None else int(self.max_seq_len)
        self._value_error_if(not 1 <= max_seq_len <= seq_len, f"max_seq_len must be in [1, {seq_len}], got {max_seq_len}")

        from . import qsa_bwd_sm100 as kernel

        max_len = kernel.max_packed_len(num_kv_heads)
        self._value_error_if(
            not 1 <= seq_len <= max_len,
            f"sequence length must be in [1, {max_len}], got {seq_len}",
        )
        self._kernel_module = kernel
        self._plan = kernel.LaunchPlan(
            seq_len, torch.cuda.get_device_properties(q_desc.device).multi_processor_count, num_kv_heads, num_seqs, max_seq_len, fp32_saved_state
        )
        if num_kv_heads == 1 and self._plan.has_sparse:
            # The cu12 compiler crashes compiling the four-query key-major kernel
            # in both DSL 4.7 and 4.8. Gate the actual compiler, not torch/CUDA_HOME.
            compiler_cuda = _dsl_cuda_version()
            self._not_implemented_error_if(
                compiler_cuda[0] < 13,
                f"QSA 12Q/1KV sparse backward requires the CUDA 13 build of CuTe DSL; found compiler CUDA {compiler_cuda[0]}.{compiler_cuda[1]}. "
                "Install 'nvidia-cutlass-dsl[cu13]>=4.7.0' with a CUDA 13 compatible driver.",
            )
        self._is_supported = True
        return True

    def compile(self) -> None:
        """Compile the launch sequence the plan's sequence length needs."""
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        with torch.cuda.device(self.descs["q"].device):
            self._compiled_kernel = self._kernel_module.compile_kernels(self._plan)

    def scratch_workspace_bytes(self) -> int:
        """Caller workspace bytes one execution needs."""
        self._ensure_support_checked()
        return self._plan.workspace_bytes()

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        dout: torch.Tensor,
        lse: torch.Tensor,
        topk_idxs: torch.Tensor,
        dq: torch.Tensor,
        dk: torch.Tensor,
        dv: torch.Tensor,
        workspace: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ):
        """Launch the backward on ``current_stream`` and return ``(dq, dk, dv)``."""
        self._runtime_error_if(self._compiled_kernel is None, "plan not compiled; call compile() first")
        tensors = {"q": q, "k": k, "v": v, "out": out, "dout": dout, "lse": lse, "topk_idxs": topk_idxs, "dq": dq, "dk": dk, "dv": dv}
        if ("cu_seqlens" in self.descs) != (cu_seqlens is not None):
            raise ValueError("cu_seqlens must be passed exactly when the plan was built with sample_cu_seqlens")
        if cu_seqlens is not None:
            tensors["cu_seqlens"] = cu_seqlens
        for name, tensor in tensors.items():
            desc = self.descs[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
            if tuple(tensor.shape) != desc.shape or tuple(tensor.stride()) != desc.stride:
                raise ValueError(f"{name} must match the compiled shape {desc.shape} and strides {desc.stride}")
            if tensor.dtype != desc.dtype or tensor.device != desc.device:
                raise ValueError(f"{name} must match the compiled dtype {desc.dtype} and device {desc.device}")
            scalar_lse = name == "lse" and self.descs["out"].dtype == torch.float32
            alignment = 4 if name in ("topk_idxs", "cu_seqlens") or scalar_lse else 16
            if tensor.data_ptr() % alignment:
                raise ValueError(f"{name} must be {alignment}-byte aligned")
        spans = {name: _byte_span(tensor) for name, tensor in tensors.items()}
        for first, second in (("dq", "dk"), ("dq", "dv"), ("dk", "dv")):
            if _overlaps(spans[first], spans[second]):
                raise ValueError(f"{first} and {second} must not overlap")
        if workspace is not None:
            if workspace.device != q.device:
                raise ValueError(f"workspace must be on {q.device}, got {workspace.device}")
            carved = (workspace.data_ptr(), workspace.data_ptr() + self.scratch_workspace_bytes())
            for name, span in spans.items():
                if _overlaps(carved, span):
                    raise ValueError(f"workspace must not overlap {name}")
        scratch = self._plan.carve(WorkspaceCarver(workspace, self.scratch_workspace_bytes(), self.__class__.__name__))
        with device_context(q.device):
            stream = self._get_default_stream(current_stream)
            self._kernel_module.launch(self._plan, self._compiled_kernel, q, k, v, out, dout, lse, topk_idxs, dq, dk, dv, scratch, stream, cu_seqlens)
        return dq, dk, dv


_cache_of__QsaTwoPassBackwardObjects: dict = {}


def _backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    topk_idxs: torch.Tensor,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    stream: Optional[cuda.CUstream] = None,
    workspace: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seq_len: Optional[int] = None,
) -> TupleDict:
    """High-level wrapper returning ``{'dq', 'dk', 'dv'}``."""
    inputs = (q, k, v, out, dout, lse, topk_idxs)
    key = (q.device, softmax_scale, max_seq_len) + tuple((tuple(t.shape), t.stride(), t.dtype) for t in inputs)
    key += tuple(None if t is None else (tuple(t.shape), t.stride(), t.dtype) for t in (dq, dk, dv, cu_seqlens))
    obj = _cache_of__QsaTwoPassBackwardObjects.get(key)
    if obj is None:
        obj = _QsaTwoPassBackward(
            *inputs, sample_dq=dq, sample_dk=dk, sample_dv=dv, softmax_scale=softmax_scale, sample_cu_seqlens=cu_seqlens, max_seq_len=max_seq_len
        )
        obj.check_support()
        obj.compile()
        _cache_of__QsaTwoPassBackwardObjects[key] = obj

    with torch.cuda.device(q.device), stream_context(stream, q.device):
        dq = torch.empty_like(q) if dq is None else dq
        dk = torch.empty_like(k) if dk is None else dk
        dv = torch.empty_like(v) if dv is None else dv
        if workspace is None:
            workspace = torch.empty(obj.scratch_workspace_bytes(), dtype=torch.uint8, device=q.device)
    obj.execute(*inputs, dq, dk, dv, workspace=workspace, current_stream=stream, cu_seqlens=cu_seqlens)
    return TupleDict(dq=dq, dk=dk, dv=dv)
