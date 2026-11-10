# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared SM90 packed-THD IndexerForward + Top-K execution.

The plan fixes tensor capacities and generated-code choices while keeping the
contents of the packed prefix tensors, causal offsets, and candidate offsets as
device-resident runtime data.  That is the useful CUDA Graph boundary for a
training stack: storage addresses and launch topology remain fixed after
capture, while a caller may update the metadata in place before replay.
"""

from __future__ import annotations

from typing import Optional

import cuda.bindings.driver as cuda
import torch

from cudnn._torch_stream import record_streams
from cudnn.api_base import APIBase, WorkspaceCarver, ws_align
from cudnn.deepseek_sparse_attention.utils.runtime import device_capability
from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error

from ._compressed_top_k_sm90 import indexer_fwd_compress_topk_sm90
from ._interface_sm90 import indexer_fwd as indexer_fwd_sm90
from ..indexer_top_k.compress_top_k_sm100 import precompile_compress_stage2_topk_varlen


def _desc(tensor: Optional[torch.Tensor], owner: APIBase, name: str):
    """Snapshot a sample tensor contract while preserving optional absence."""
    return None if tensor is None else owner._make_tensor_desc(tensor, name=name)


def _same_tensor_contract(tensor: Optional[torch.Tensor], desc, name: str) -> None:
    """Require an execute binding to match its prepared descriptor exactly."""
    if desc is None:
        if tensor is not None:
            raise ValueError(f"{name} must be None: this plan was prepared without it")
        return
    if tensor is None:
        raise ValueError(f"{name} is required: this plan was prepared with it")
    if tensor.dtype != desc.dtype or tensor.device != desc.device:
        raise ValueError(f"{name} must have dtype/device {desc.dtype}/{desc.device}, got {tensor.dtype}/{tensor.device}")
    if tuple(tensor.shape) != tuple(desc.shape) or tuple(tensor.stride()) != tuple(desc.stride):
        raise ValueError(f"{name} must have shape/stride {tuple(desc.shape)}/{tuple(desc.stride)}, " f"got {tuple(tensor.shape)}/{tuple(tensor.stride())}")


def _check_output(tensor: Optional[torch.Tensor], name: str, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device, required: bool) -> None:
    """Validate a caller-owned output without allocating or converting it."""
    if not required:
        if tensor is not None:
            raise ValueError(f"{name} must be None: this output was not enabled when the plan was prepared")
        return
    if tensor is None:
        raise ValueError(f"{name} is required: this output was enabled when the plan was prepared")
    if tensor.dtype != dtype or tensor.device != device or tuple(tensor.shape) != shape or not tensor.is_contiguous():
        raise ValueError(f"{name} must be a contiguous {dtype} CUDA tensor of shape {shape} on {device}")


class IndexerForwardTopKPlan(APIBase):
    """Prepared SM90 fused indexer execution for fixed-capacity packed THD.

    The plan compiles the compact IndexerForward scorer and radix Top-K stages
    once, retains their launchables, and owns no device memory.  Q/K/W and all
    metadata tensors are caller-owned CUDA tensors.  ``execute`` also requires
    caller-owned outputs and a contiguous workspace of at least
    :meth:`get_workspace_size` bytes.  Independent executions may share the
    plan, but concurrent executions need disjoint outputs and workspaces.

    The Q/K storage extents and prefix tensor lengths are plan capacities.  On
    every execution the caller may change prefix, causal-offset, and candidate-
    offset *contents* in place, provided the value contract remains true:

    * prefixes start at zero, are monotonic, and end at the fixed Q/K extents;
    * every segment length is within ``max_seqlen_q/k``;
    * ``cand_batch_offsets`` matches those prefixes and ``ratio``;
    * its final offset is at most ``max_candidate_count``.

    These values intentionally stay on device; ``execute`` never reads them to
    the host.  Tensor addresses and host scalar arguments captured by a CUDA
    Graph remain fixed until that graph is rebuilt.  The first version returns
    local per-segment KV ids and supports contiguous packed THD on exact SM90,
    ratio 4, D128, Hq64/Hkv1, and BF16 or per-token E4M3 Q/K only.
    """

    def __init__(
        self,
        sample_q: torch.Tensor,
        sample_k: torch.Tensor,
        sample_w: torch.Tensor,
        sample_cu_seqlens_q: torch.Tensor,
        sample_cu_seqlens_k: torch.Tensor,
        sample_cand_batch_offsets: torch.Tensor,
        *,
        top_k: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        max_candidate_count: int,
        ratio: int = 4,
        sm_scale: float = 1.0,
        precision: str = "bf16",
        sample_q_scale: Optional[torch.Tensor] = None,
        sample_k_scale: Optional[torch.Tensor] = None,
        sample_q_causal_offsets: Optional[torch.Tensor] = None,
        return_softmax: bool = True,
        return_lse: bool = False,
        deterministic: bool = False,
    ):
        super().__init__()
        self._warn_experimental_api()
        self.q_desc = self._make_tensor_desc(sample_q, name="sample_q")
        self.k_desc = self._make_tensor_desc(sample_k, name="sample_k")
        self.w_desc = self._make_tensor_desc(sample_w, name="sample_w")
        self.cu_q_desc = self._make_tensor_desc(sample_cu_seqlens_q, name="sample_cu_seqlens_q")
        self.cu_k_desc = self._make_tensor_desc(sample_cu_seqlens_k, name="sample_cu_seqlens_k")
        self.cand_offsets_desc = self._make_tensor_desc(sample_cand_batch_offsets, name="sample_cand_batch_offsets")
        self.q_scale_desc = _desc(sample_q_scale, self, "sample_q_scale")
        self.k_scale_desc = _desc(sample_k_scale, self, "sample_k_scale")
        self.q_causal_offsets_desc = _desc(sample_q_causal_offsets, self, "sample_q_causal_offsets")

        self.top_k = int(top_k)
        self.max_seqlen_q = int(max_seqlen_q)
        self.max_seqlen_k = int(max_seqlen_k)
        self.max_candidate_count = int(max_candidate_count)
        self.ratio = int(ratio)
        self.sm_scale = float(sm_scale)
        self.precision = str(precision).lower()
        self.return_softmax = bool(return_softmax)
        self.return_lse = bool(return_lse)
        self.deterministic = bool(deterministic)
        self._compile_samples = (
            sample_q,
            sample_k,
            sample_w,
            sample_cu_seqlens_q,
            sample_cu_seqlens_k,
            sample_cand_batch_offsets,
            sample_q_scale,
            sample_k_scale,
            sample_q_causal_offsets,
        )

    @property
    def total_q_capacity(self) -> int:
        """Return the fixed packed-query storage capacity."""
        return int(self.q_desc.shape[0])

    @property
    def total_k_capacity(self) -> int:
        """Return the fixed packed-key storage capacity."""
        return int(self.k_desc.shape[0])

    @property
    def segment_capacity(self) -> int:
        """Return the fixed number of packed segment slots."""
        return int(self.cu_q_desc.shape[0]) - 1

    def check_support(self) -> bool:
        """Validate the bounded SM90 specialization before compilation."""
        self._logger.debug("Entering check_support")
        device = self.q_desc.device
        self._value_error_if(device.type != "cuda", f"sample_q must be on CUDA, got {device}")
        self._runtime_error_if(device_capability(device) != (9, 0), "IndexerForwardTopKPlan currently supports SM90 only")
        self._value_error_if(self.precision not in ("bf16", "fp8"), "precision must be 'bf16' or 'fp8'")
        self._not_implemented_error_if(self.ratio != 4, f"SM90 prepared IndexerForwardTopK is validated only for ratio=4, got {self.ratio}")
        self._value_error_if(not 0 < self.top_k <= 2048, f"top_k must be in [1, 2048], got {self.top_k}")
        dsl_error = cutedsl_requirement_error("SM90 prepared IndexerForwardTopK")
        if dsl_error:
            raise ImportError(dsl_error)
        arch_error = cutedsl_arch_requirement_error(device_capability(device))
        self._not_implemented_error_if(bool(arch_error), arch_error or "")
        self._value_error_if(self.max_candidate_count < 0, "max_candidate_count must be non-negative")
        self._value_error_if(self.max_seqlen_q < 0 or self.max_seqlen_k < 0, "max_seqlen_q/k must be non-negative")

        self._value_error_if(self.q_desc.ndim != 3 or self.k_desc.ndim != 3 or self.w_desc.ndim != 2, "packed THD requires q(Tq,H,D), k(Tk,Hkv,D), and w(Tq,H)")
        tq, hq, dq = self.q_desc.shape
        tk, hkv, dk = self.k_desc.shape
        self._not_implemented_error_if((hq, hkv, dq, dk) != (64, 1, 128, 128), f"SM90 prepared path requires Hq=64, Hkv=1, D=128; got {(hq, hkv, dq, dk)}")
        self._value_error_if(tuple(self.w_desc.shape) != (tq, hq), f"w must have shape {(tq, hq)}, got {self.w_desc.shape}")
        self._value_error_if(self.max_seqlen_q > tq or self.max_seqlen_k > tk, "max_seqlen_q/k cannot exceed the fixed Q/K storage extents")

        all_descs = (
            (self.q_desc, "q"),
            (self.k_desc, "k"),
            (self.w_desc, "w"),
            (self.cu_q_desc, "cu_seqlens_q"),
            (self.cu_k_desc, "cu_seqlens_k"),
            (self.cand_offsets_desc, "cand_batch_offsets"),
        )
        for desc, name in all_descs:
            self._value_error_if(desc.device != device, f"{name} must be on {device}, got {desc.device}")
            self._not_implemented_error_if(not desc.is_contiguous(), f"{name} must be contiguous")
        self._value_error_if(self.cu_q_desc.dtype != torch.int32 or self.cu_k_desc.dtype != torch.int32, "cu_seqlens_q/k must be int32")
        self._value_error_if(
            self.cu_q_desc.ndim != 1 or self.cu_k_desc.shape != self.cu_q_desc.shape or self.cu_q_desc.shape[0] < 2,
            "cu_seqlens_q/k must be 1D tensors of equal shape (segment_capacity + 1) >= 2",
        )
        self._value_error_if(
            self.cand_offsets_desc.dtype != torch.int64 or self.cand_offsets_desc.shape != self.cu_q_desc.shape,
            "cand_batch_offsets must be int64 with the same shape as cu_seqlens_q",
        )

        if self.precision == "bf16":
            self._value_error_if(self.q_desc.dtype != torch.bfloat16 or self.k_desc.dtype != torch.bfloat16, "BF16 plan requires BF16 q/k")
            self._value_error_if(self.w_desc.dtype not in (torch.bfloat16, torch.float32), "BF16 plan requires BF16 or FP32 w")
            self._value_error_if(self.q_scale_desc is not None or self.k_scale_desc is not None, "BF16 plan does not accept q_scale/k_scale")
        else:
            self._value_error_if(self.q_desc.dtype != torch.float8_e4m3fn or self.k_desc.dtype != torch.float8_e4m3fn, "FP8 plan requires E4M3 q/k")
            self._value_error_if(self.w_desc.dtype != torch.bfloat16, "FP8 plan requires BF16 w")
            self._value_error_if(self.q_scale_desc is None or self.k_scale_desc is None, "FP8 plan requires q_scale/k_scale")
            if self.q_scale_desc is not None and self.k_scale_desc is not None:
                for desc, name, shape in (
                    (self.q_scale_desc, "q_scale", (tq, hq)),
                    (self.k_scale_desc, "k_scale", (tk, hkv)),
                ):
                    self._value_error_if(
                        desc.dtype != torch.float32 or desc.device != device or tuple(desc.shape) != shape or not desc.is_contiguous(),
                        f"{name} must be contiguous FP32 with shape {shape} on {device}",
                    )

        if self.q_causal_offsets_desc is not None:
            desc = self.q_causal_offsets_desc
            self._value_error_if(
                desc.dtype != torch.int32 or desc.device != device or tuple(desc.shape) != (self.segment_capacity,) or not desc.is_contiguous(),
                f"q_causal_offsets must be contiguous int32 with shape ({self.segment_capacity},) on {device}",
            )
        self._is_supported = True
        return True

    def compile(self) -> None:
        """Compile both SM90 stages from the sample signatures; launch nothing."""
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        q, k, w, cu_q, cu_k, cand_offsets, q_scale, k_scale, q_causal_offsets = self._compile_samples
        score = indexer_fwd_sm90(
            q,
            k,
            w,
            ratio=self.ratio,
            qhead_per_kv_head=64,
            sm_scale=self.sm_scale,
            out=None,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=self.max_seqlen_q,
            max_seqlen_k=self.max_seqlen_k,
            q_causal_offsets=q_causal_offsets,
            precision=self.precision,
            q_scale=q_scale,
            k_scale=k_scale,
            return_lse=self.return_lse,
            lse_out=None,
            is_compressed_logits=True,
            cand_batch_offsets=cand_offsets,
            _compile_only=True,
        )
        topk = precompile_compress_stage2_topk_varlen(
            cu_q,
            cu_k,
            cand_offsets,
            self.max_seqlen_q,
            self.max_seqlen_k,
            self.top_k,
            self.ratio,
            block_threads=256,
            q_causal_offsets=q_causal_offsets,
            return_softmax=self.return_softmax,
            deterministic=self.deterministic,
        )
        self._compiled_kernel = (score, topk)
        self._compile_samples = None

    def scratch_workspace_bytes(self) -> int:
        """Exact caller workspace bytes for the prepared candidate capacity."""
        return ws_align(self.max_candidate_count * torch.float32.itemsize)

    def get_workspace_size(self) -> int:
        """Alias matching graph-plan workspace queries."""
        return self.scratch_workspace_bytes()

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        w: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        cand_batch_offsets: torch.Tensor,
        out_indices: torch.Tensor,
        out_logits: torch.Tensor,
        *,
        workspace: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        q_causal_offsets: Optional[torch.Tensor] = None,
        softmax_out: Optional[torch.Tensor] = None,
        lse_out: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        """Launch the prepared two-stage pipeline into caller-owned storage.

        ``q``, ``k``, ``w``, prefixes, optional scales/causal offsets, and all
        outputs must exactly match the dtype, shape, stride, device, and
        optional-presence contract captured at preparation.  ``out_indices``
        is INT32 local-per-segment KV ids; ``out_logits`` is FP32.  Optional
        ``softmax_out`` and ``lse_out`` are FP32 and are required exactly when
        enabled by the plan.  Padding slots use the underlying Top-K contract:
        index ``-1``, logit ``-inf``, and softmax ``0``.

        ``workspace`` is contiguous caller storage on the plan device and must
        remain alive until work on ``current_stream`` completes.  Execution
        performs no JIT, allocation, staging, host synchronization, or device
        metadata read.  The caller must order input/metadata updates before the
        launch stream; all plan kernels run on that stream.  During CUDA Graph
        replay, tensor contents may change in place but captured addresses,
        shapes, host scalars, and launch topology remain fixed.
        """
        self._ensure_support_checked()
        if self._compiled_kernel is None:
            raise ValueError("IndexerForwardTopKPlan is not compiled; call compile() before execute() or use prepare_indexer_forward_top_k()")
        for tensor, desc, name in (
            (q, self.q_desc, "q"),
            (k, self.k_desc, "k"),
            (w, self.w_desc, "w"),
            (cu_seqlens_q, self.cu_q_desc, "cu_seqlens_q"),
            (cu_seqlens_k, self.cu_k_desc, "cu_seqlens_k"),
            (cand_batch_offsets, self.cand_offsets_desc, "cand_batch_offsets"),
            (q_scale, self.q_scale_desc, "q_scale"),
            (k_scale, self.k_scale_desc, "k_scale"),
            (q_causal_offsets, self.q_causal_offsets_desc, "q_causal_offsets"),
        ):
            _same_tensor_contract(tensor, desc, name)

        device = self.q_desc.device
        out_shape = (self.total_q_capacity, self.top_k)
        _check_output(out_indices, "out_indices", out_shape, torch.int32, device, True)
        _check_output(out_logits, "out_logits", out_shape, torch.float32, device, True)
        _check_output(softmax_out, "softmax_out", out_shape, torch.float32, device, self.return_softmax)
        _check_output(lse_out, "lse_out", (self.total_q_capacity,), torch.float32, device, self.return_lse)
        if not isinstance(workspace, torch.Tensor) or workspace.device != device:
            raise ValueError(f"workspace must be a torch.Tensor on {device}")
        candidate = WorkspaceCarver(workspace, self.scratch_workspace_bytes(), "IndexerForwardTopKPlan").take(self.max_candidate_count, torch.float32)

        indexer_fwd_compress_topk_sm90(
            q,
            k,
            w,
            self.top_k,
            self.ratio,
            64,
            cu_seqlens_q,
            cu_seqlens_k,
            self.max_seqlen_q,
            self.max_seqlen_k,
            sm_scale=self.sm_scale,
            precision=self.precision,
            q_scale=q_scale,
            k_scale=k_scale,
            topk_indices_global=False,
            cand_buffer=candidate,
            out_indices=out_indices,
            out_logits=out_logits,
            cand_batch_offsets=cand_batch_offsets,
            q_causal_offsets=q_causal_offsets,
            return_softmax=self.return_softmax,
            softmax_out=softmax_out,
            return_lse=self.return_lse,
            lse_out=lse_out,
            deterministic=self.deterministic,
            stream=current_stream,
            _compiled_score=self._compiled_kernel[0],
            _compiled_topk=self._compiled_kernel[1],
        )
        record_streams(
            (
                q,
                k,
                w,
                q_scale,
                k_scale,
                cu_seqlens_q,
                cu_seqlens_k,
                cand_batch_offsets,
                q_causal_offsets,
                out_indices,
                out_logits,
                softmax_out,
                lse_out,
                workspace,
            ),
            current_stream,
            device,
        )


def prepare_indexer_forward_top_k(
    sample_q: torch.Tensor,
    sample_k: torch.Tensor,
    sample_w: torch.Tensor,
    sample_cu_seqlens_q: torch.Tensor,
    sample_cu_seqlens_k: torch.Tensor,
    sample_cand_batch_offsets: torch.Tensor,
    *,
    top_k: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    max_candidate_count: int,
    ratio: int = 4,
    sm_scale: float = 1.0,
    precision: str = "bf16",
    sample_q_scale: Optional[torch.Tensor] = None,
    sample_k_scale: Optional[torch.Tensor] = None,
    sample_q_causal_offsets: Optional[torch.Tensor] = None,
    return_softmax: bool = True,
    return_lse: bool = False,
    deterministic: bool = False,
) -> IndexerForwardTopKPlan:
    """Validate and compile a reusable SM90 packed-THD IndexerForward+Top-K plan.

    ``sample_q`` is ``[Tq, 64, 128]``, ``sample_k`` is ``[Tk, 1, 128]``, and
    ``sample_w`` is ``[Tq, 64]``.  They must be contiguous CUDA tensors on an
    exact SM90 device.  BF16 precision takes BF16 Q/K and BF16 or FP32 W; FP8
    precision takes E4M3 Q/K, BF16 W, and contiguous FP32 per-token
    ``sample_q_scale``/``sample_k_scale``.  Prefix samples are equal-shaped
    contiguous INT32 ``[segments + 1]`` tensors; candidate offsets are INT64
    with that shape.  Optional causal offsets are INT32 ``[segments]``.

    ``top_k``, ``ratio``, precision, optional outputs, and determinism select
    bounded generated-code variants.  ``max_seqlen_q/k`` and
    ``max_candidate_count`` describe plan/workspace capacity without entering
    the compile key.  ``sm_scale`` is a host FP32 scoring scalar fixed by the
    plan and by any graph capture, but does not specialize compilation.  Pass
    raw BF16 W and use this scalar for uniform softmax/head-normalization
    factors to avoid an extra BF16 rounding or accidental double scaling.
    Prefix contents, packed totals, pointers, and streams are runtime data.
    The factory compiles both stages but launches neither and allocates no
    device memory.  Allocate :meth:`IndexerForwardTopKPlan.get_workspace_size`
    bytes before calling :meth:`IndexerForwardTopKPlan.execute` or beginning
    CUDA Graph capture.
    """
    plan = IndexerForwardTopKPlan(
        sample_q,
        sample_k,
        sample_w,
        sample_cu_seqlens_q,
        sample_cu_seqlens_k,
        sample_cand_batch_offsets,
        top_k=top_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        max_candidate_count=max_candidate_count,
        ratio=ratio,
        sm_scale=sm_scale,
        precision=precision,
        sample_q_scale=sample_q_scale,
        sample_k_scale=sample_k_scale,
        sample_q_causal_offsets=sample_q_causal_offsets,
        return_softmax=return_softmax,
        return_lse=return_lse,
        deterministic=deterministic,
    )
    plan.check_support()
    plan.compile()
    return plan


__all__ = ["IndexerForwardTopKPlan", "prepare_indexer_forward_top_k"]
