# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter of the SM100 index-list sparse (QSA) d256 backward -- ``kernels/sm100/bprop_sparse_d256_f16.py``."""

from __future__ import annotations

from typing import Optional

import torch
from cuda.bindings import driver as cuda

from cudnn._torch_stream import device_context, stream_context
from cudnn.api_base import APIBase, TupleDict, WorkspaceCarver
from cudnn.frost.buffers import cutedsl_requirement_error

NUM_Q_HEADS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 256
TOPK_WIDTH = 2051  # 512 blocks x 4 tokens + 3 tail tokens
SOFTMAX_SCALE = HEAD_DIM**-0.5
_SUPPORTED_CAPABILITIES = ((10, 0),)


def _byte_span(tensor: torch.Tensor) -> tuple[int, int]:
    # Exact for the contiguous layouts check_support admits.
    return tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size()


def _overlaps(lhs: tuple[int, int], rhs: tuple[int, int]) -> bool:
    return lhs[0] < rhs[1] and rhs[0] < lhs[1]


class SparseGqaBwdDslSm100(APIBase):
    """QSA backward: dQ, dK and dV from Q, K, V, O, dO, the forward LSE and the selected token indices."""

    def __init__(
        self,
        sample_q: torch.Tensor,  # (T, 24, 256) bf16
        sample_k: torch.Tensor,  # (T, 2, 256) bf16
        sample_v: torch.Tensor,  # (T, 2, 256) bf16
        sample_out: torch.Tensor,  # (T, 24, 256) bf16
        sample_dout: torch.Tensor,  # (T, 24, 256) bf16
        sample_lse: torch.Tensor,  # (T, 24) fp32, natural-log LSE of the scaled scores
        sample_topk_idxs: torch.Tensor,  # (T, 2051) int32
        sample_dq: Optional[torch.Tensor] = None,
        sample_dk: Optional[torch.Tensor] = None,
        sample_dv: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
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
        self.softmax_scale = softmax_scale
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
            f"{self.__class__.__name__} requires SM100, found SM{capability[0]}{capability[1]}",
        )
        self._not_implemented_error_if(
            self.softmax_scale not in (None, SOFTMAX_SCALE),
            f"{self.__class__.__name__} supports only softmax_scale {SOFTMAX_SCALE}, got {self.softmax_scale}",
        )
        self._value_error_if(q_desc.ndim != 3, f"q must be 3-D (T, {NUM_Q_HEADS}, {HEAD_DIM}), got {q_desc.shape}")
        seq_len = q_desc.shape[0]
        expected = {
            "q": ((seq_len, NUM_Q_HEADS, HEAD_DIM), torch.bfloat16),
            "k": ((seq_len, NUM_KV_HEADS, HEAD_DIM), torch.bfloat16),
            "v": ((seq_len, NUM_KV_HEADS, HEAD_DIM), torch.bfloat16),
            "out": ((seq_len, NUM_Q_HEADS, HEAD_DIM), torch.bfloat16),
            "dout": ((seq_len, NUM_Q_HEADS, HEAD_DIM), torch.bfloat16),
            "lse": ((seq_len, NUM_Q_HEADS), torch.float32),
            "topk_idxs": ((seq_len, TOPK_WIDTH), torch.int32),
            "dq": ((seq_len, NUM_Q_HEADS, HEAD_DIM), torch.bfloat16),
            "dk": ((seq_len, NUM_KV_HEADS, HEAD_DIM), torch.bfloat16),
            "dv": ((seq_len, NUM_KV_HEADS, HEAD_DIM), torch.bfloat16),
        }
        for name, (shape, dtype) in expected.items():
            desc = self.descs[name]
            self._value_error_if(desc.shape != shape, f"{name} must have shape {shape}, got {desc.shape}")
            self._check_dtype(desc, dtype, name=name)
            self._value_error_if(desc.device != q_desc.device, f"{name} must be on {q_desc.device}, got {desc.device}")
            # The kernels address the declared layout natively and never repack (Rule 2).
            self._not_implemented_error_if(
                not desc.is_contiguous(),
                f"{self.__class__.__name__} addresses only contiguous {name}, got shape {desc.shape} strides {desc.stride}",
            )

        from .kernels.sm100 import bprop_sparse_d256_f16 as kernel

        self._value_error_if(
            not 1 <= seq_len <= kernel.MAX_SEQ_LEN,
            f"sequence length must be in [1, {kernel.MAX_SEQ_LEN}], got {seq_len}",
        )
        self._kernel_module = kernel
        self._plan = kernel.LaunchPlan(seq_len, torch.cuda.get_device_properties(q_desc.device).multi_processor_count)
        self._is_supported = True
        return True

    def compile(self) -> None:
        """Compile the launch sequence the plan's sequence length needs."""
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        with torch.cuda.device(self.descs["q"].device):
            self._compiled_kernel = self._kernel_module.compile_kernels(self._plan.has_sparse)

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
    ):
        """Launch the backward on ``current_stream`` and return ``(dq, dk, dv)``."""
        self._runtime_error_if(self._compiled_kernel is None, "plan not compiled; call compile() first")
        tensors = {"q": q, "k": k, "v": v, "out": out, "dout": dout, "lse": lse, "topk_idxs": topk_idxs, "dq": dq, "dk": dk, "dv": dv}
        for name, tensor in tensors.items():
            desc = self.descs[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
            if tuple(tensor.shape) != desc.shape or tuple(tensor.stride()) != desc.stride:
                raise ValueError(f"{name} must match the compiled shape {desc.shape} and strides {desc.stride}")
            if tensor.dtype != desc.dtype or tensor.device != desc.device:
                raise ValueError(f"{name} must match the compiled dtype {desc.dtype} and device {desc.device}")
            alignment = 4 if name == "topk_idxs" else 16
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
            self._kernel_module.launch(self._plan, self._compiled_kernel, q, k, v, out, dout, lse, topk_idxs, dq, dk, dv, scratch, stream)
        return dq, dk, dv


_cache_of_SparseGqaBwdDslSm100Objects: dict = {}


def sparse_gqa_bwd_wrapper_sm100(
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
) -> TupleDict:
    """High-level wrapper returning ``{'dq', 'dk', 'dv'}``."""
    inputs = (q, k, v, out, dout, lse, topk_idxs)
    key = (q.device, softmax_scale) + tuple((tuple(t.shape), t.stride(), t.dtype) for t in inputs)
    key += tuple(None if t is None else (tuple(t.shape), t.stride(), t.dtype) for t in (dq, dk, dv))
    obj = _cache_of_SparseGqaBwdDslSm100Objects.get(key)
    if obj is None:
        obj = SparseGqaBwdDslSm100(*inputs, sample_dq=dq, sample_dk=dk, sample_dv=dv, softmax_scale=softmax_scale)
        obj.check_support()
        obj.compile()
        _cache_of_SparseGqaBwdDslSm100Objects[key] = obj

    with torch.cuda.device(q.device), stream_context(stream, q.device):
        dq = torch.empty_like(q) if dq is None else dq
        dk = torch.empty_like(k) if dk is None else dk
        dv = torch.empty_like(v) if dv is None else dv
        if workspace is None:
            workspace = torch.empty(obj.scratch_workspace_bytes(), dtype=torch.uint8, device=q.device)
    obj.execute(*inputs, dq, dk, dv, workspace=workspace, current_stream=stream)
    return TupleDict(dq=dq, dk=dk, dv=dv)
