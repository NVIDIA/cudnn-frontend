# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""APIBase integration for SM90 and SM100 DSA sparse-prefill forward."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import torch

from cudnn.api_base import APIBase, TupleDict
from . import _interface_sm100 as _iface_sm100
from ._support_sm90 import sm90_dsa_cutedsl_requirement_error

# SM100 (H, D_qk) -> supported indexer_topk. H96 runs the H64 kernel on two
# overlapping 64-head windows, and H128 D576 (and H128 D512 with
# indexer_topk=2048) on two disjoint ones (see
# ``_interface_sm100._HEAD64_WINDOW_STARTS``).
_SUPPORTED_VARIANTS = {
    (64, 512): (0, 512, 1024, 2048),
    (64, 576): (0, 512, 1024, 2048),
    (96, 512): (0, 512, 1024, 2048),
    (96, 576): (0, 512, 1024, 2048),
    (128, 512): (0, 512, 1024, 2048),
    (128, 576): (0, 512, 1024, 2048),
}
_SM90_SUPPORTED_HEADS = (16, 32, 64, 96, 128)
_SM90_SUPPORTED_HEAD_DIMS = (512, 576)


class SparseAttentionForward(APIBase):
    """Sparse prefill forward for the supported Hopper and Blackwell shapes.

    SM90 ``compile()`` builds a storage-free symbolic launchable; its
    ``execute()`` requires caller-owned outputs and performs no allocation or
    JIT. The SM100 interface retains its concrete-tensor cache. Runtime
    sequence and top-k extents remain dynamic on both architectures.
    """

    def __init__(
        self,
        sample_q: torch.Tensor,  # (total_S_q, H, D_qk) FP16/BF16
        sample_kv: torch.Tensor,  # (total_S_kv, D_qk) FP16/BF16, K=V latent
        sample_topk_idxs: torch.Tensor,  # (total_S_q, logical_K) INT32
        sample_attn_sink: Optional[torch.Tensor] = None,  # (H,) FP32
        sample_topk_length: Optional[torch.Tensor] = None,  # (total_S_q,) INT32
        softmax_scale: Optional[float] = None,
        indexer_topk: int = 0,
    ):
        """Capture a sparse-attention declaration for support checking and compilation."""
        super().__init__()
        self.q_desc = self._make_tensor_desc(sample_q, name="sample_q")
        self.kv_desc = self._make_tensor_desc(sample_kv, name="sample_kv")
        self.topk_idxs_desc = self._make_tensor_desc(sample_topk_idxs, name="sample_topk_idxs")
        self.attn_sink_desc = self._make_tensor_desc(sample_attn_sink, name="sample_attn_sink")
        self.topk_length_desc = self._make_tensor_desc(sample_topk_length, name="sample_topk_length")
        self.softmax_scale = None if softmax_scale is None else float(softmax_scale)
        self.indexer_topk = int(indexer_topk)
        self.num_heads: Optional[int] = None
        self.head_dim: Optional[int] = None
        self.head_dim_v: Optional[int] = None
        self.logical_topk: Optional[int] = None
        self._device_major: Optional[int] = None

    def check_support(self) -> bool:
        """Validate the sampled declaration and select its architecture path."""
        self._value_error_if(self.q_desc.ndim != 3, f"Q must be 3-D (total_S_q, H, D_qk), got {self.q_desc.shape}")
        self._value_error_if(self.kv_desc.ndim != 2, f"KV must be 2-D (total_S_kv, D_qk), got {self.kv_desc.shape}")
        self._value_error_if(
            self.topk_idxs_desc.ndim != 2,
            f"topk_idxs must be 2-D (total_S_q, logical_K), got {self.topk_idxs_desc.shape}",
        )

        total_s_q, num_heads, head_dim = self.q_desc.shape
        total_s_kv, kv_head_dim = self.kv_desc.shape
        topk_s_q, logical_topk = self.topk_idxs_desc.shape
        del total_s_kv

        self._check_dtype(self.q_desc, [torch.float16, torch.bfloat16], name="Q")
        self._check_dtype(self.kv_desc, self.q_desc.dtype, name="KV", extra_error_msg="KV must have same dtype as Q")
        self._check_dtype(self.topk_idxs_desc, torch.int32, name="topk_idxs")
        self._value_error_if(kv_head_dim != head_dim, f"KV head dimension ({kv_head_dim}) must match Q ({head_dim})")
        self._value_error_if(topk_s_q != total_s_q, f"topk_idxs first dimension ({topk_s_q}) must match Q ({total_s_q})")

        ref_device = self.q_desc.device
        descs = [self.q_desc, self.kv_desc, self.topk_idxs_desc]
        if self.attn_sink_desc is not None:
            descs.append(self.attn_sink_desc)
        if self.topk_length_desc is not None:
            descs.append(self.topk_length_desc)
        self._value_error_if(ref_device.type != "cuda", f"Q must live on CUDA, got {ref_device}")
        self._value_error_if(
            any(desc.device != ref_device for desc in descs),
            f"All inputs must share Q's device {ref_device}, got {[desc.device for desc in descs]}",
        )

        capability = torch.cuda.get_device_capability(ref_device)
        major = capability[0]
        variant = (num_heads, head_dim)
        if major == 9:
            self._value_error_if(
                num_heads not in _SM90_SUPPORTED_HEADS or head_dim not in _SM90_SUPPORTED_HEAD_DIMS,
                f"SM90 SparseAttentionForward supports H in {_SM90_SUPPORTED_HEADS} and D_qk in {_SM90_SUPPORTED_HEAD_DIMS}, got {variant}",
            )
            supported_indexer_topk = (0, 512, 1024, 2048)
        else:
            self._value_error_if(
                variant not in _SUPPORTED_VARIANTS,
                "SM100 SparseAttentionForward supports only (H, D_qk) in " f"{tuple(_SUPPORTED_VARIANTS)}, got {variant}",
            )
            supported_indexer_topk = _SUPPORTED_VARIANTS.get(variant, ())
        if supported_indexer_topk:
            self._value_error_if(
                self.indexer_topk not in supported_indexer_topk,
                f"indexer_topk={self.indexer_topk} is unsupported for (H, D_qk)={variant}; expected one of {supported_indexer_topk}",
            )
        self._value_error_if(self.indexer_topk < 0, f"indexer_topk must be nonnegative, got {self.indexer_topk}")
        self._value_error_if(
            self.indexer_topk > logical_topk,
            f"indexer_topk ({self.indexer_topk}) must not exceed logical K ({logical_topk})",
        )

        if self.attn_sink_desc is not None:
            self._check_dtype(self.attn_sink_desc, torch.float32, name="attn_sink")
            self._value_error_if(
                self.attn_sink_desc.shape != (num_heads,),
                f"attn_sink must have shape {(num_heads,)}, got {self.attn_sink_desc.shape}",
            )
        if self.topk_length_desc is not None:
            self._check_dtype(self.topk_length_desc, torch.int32, name="topk_length")
            self._value_error_if(
                self.topk_length_desc.shape != (total_s_q,),
                f"topk_length must have shape {(total_s_q,)}, got {self.topk_length_desc.shape}",
            )

        self._runtime_error_if(
            major not in (9, 10),
            f"SparseAttentionForward requires an SM90 or SM100-family GPU, found SM{capability[0]}{capability[1]}",
        )
        if major == 9:
            dsl_error = sm90_dsa_cutedsl_requirement_error()
            self._runtime_error_if(dsl_error is not None, dsl_error or "")
            self._runtime_error_if(capability != (9, 0), f"SM90 SparseAttentionForward requires compute capability (9, 0), found {capability}")
        elif major == 10:
            # This also rejects capabilities without an architecture-specific
            # compiler target (for example an unmapped SM10x).
            _iface_sm100._gpu_arch_flag(ref_device)

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.head_dim_v = 512
        self.logical_topk = logical_topk
        self._device_major = major
        self._is_supported = True
        return True

    def compile(self) -> None:
        """Compile the SM90 symbolic plan or prime the existing SM100 path."""
        self._ensure_support_checked()
        if self._device_major == 9:
            from . import _interface_sm90 as _iface_sm90

            self._compiled_kernel = _iface_sm90.compile_sparse_attention_forward_sm90(
                self.q_desc.device.index,
                self.q_desc.dtype,
                self.head_dim,
                self.attn_sink_desc is not None,
                self.topk_length_desc is not None,
                self.indexer_topk,
            )
            return
        # The interface compiles from concrete execute-time tensors after
        # stream-ordered normalization and top-k padding.
        self._compiled_kernel = True

    def execute(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        topk_idxs: torch.Tensor,
        *,
        attn_sink: Optional[torch.Tensor] = None,
        topk_length: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        max_logits: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        lse_indexer: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Execute a compiled sparse-attention plan.

        On SM90 this is the allocation-free, CUDA Graph-compatible entry:
        ``out``, ``max_logits``, ``lse``, and (when enabled) ``lse_indexer``
        must be preallocated, contiguous, and disjoint from all inputs and
        other outputs. Runtime H, Dqk, dtypes, and optional-operand presence
        must match the compiled declaration; sequence and logical top-k
        extents may change. No workspace is required.

        ``current_stream`` is either ``None`` for PyTorch's current stream or
        a concrete CUDA driver stream on Q's device. SM100 retains its existing
        direct execution behavior.
        """
        if self._compiled_kernel is None:
            raise ValueError("SparseAttentionForward kernel not compiled")
        scale = self.softmax_scale if softmax_scale is None else float(softmax_scale)
        if self._device_major == 9:
            from . import _interface_sm90 as _iface_sm90

            expected = (self.num_heads, self.head_dim, self.q_desc.dtype, self.kv_desc.dtype, self.topk_idxs_desc.dtype)
            actual = (
                q.shape[1] if q.ndim == 3 else None,
                q.shape[2] if q.ndim == 3 else None,
                q.dtype,
                kv.dtype,
                topk_idxs.dtype,
            )
            if actual != expected:
                raise ValueError(f"SM90 runtime (H, Dqk, Q dtype, KV dtype, index dtype) {actual} does not match the compiled plan {expected}")
            if (attn_sink is not None) != (self.attn_sink_desc is not None):
                raise ValueError("SM90 runtime attn_sink presence must match the compiled plan")
            if (topk_length is not None) != (self.topk_length_desc is not None):
                raise ValueError("SM90 runtime topk_length presence must match the compiled plan")

            return _iface_sm90.sparse_attention_forward_sm90(
                q,
                kv,
                topk_idxs,
                attn_sink=attn_sink,
                topk_length=topk_length,
                softmax_scale=scale,
                indexer_topk=self.indexer_topk,
                out=out,
                max_logits=max_logits,
                lse=lse,
                lse_indexer=lse_indexer,
                current_stream=current_stream,
                compiled_kernel=self._compiled_kernel,
            )
        return _iface_sm100.sparse_attention_forward_sm100(
            q,
            kv,
            topk_idxs,
            attn_sink=attn_sink,
            topk_length=topk_length,
            softmax_scale=scale,
            indexer_topk=self.indexer_topk,
            out=out,
            max_logits=max_logits,
            lse=lse,
            lse_indexer=lse_indexer,
            current_stream=current_stream,
        )


_cache_of_sparse_attention_forward_objects: dict = {}


def sparse_attention_forward_wrapper(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_idxs: torch.Tensor,
    *,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    indexer_topk: int = 0,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Allocate, normalize, and run SM90/SM100 sparse-prefill attention.

    ``lse_indexer`` is ``None`` when ``indexer_topk == 0``.  The wrapper
    accepts arbitrary logical K. Blackwell pads indices internally to a
    multiple of 64; Hopper handles the tail directly. It may allocate aligned
    input copies and outputs on ``stream``; use ``SparseAttentionForward`` with
    caller-owned outputs for an allocation-free CUDA Graph capture path.
    When ``topk_length`` is present, rows shorter than ``indexer_topk`` retain
    the source kernel's ``+inf`` ``lse_indexer`` sentinel.
    """

    scale = 1.0 / math.sqrt(q.shape[-1]) if softmax_scale is None else float(softmax_scale)
    capability = torch.cuda.get_device_capability(q.device) if q.is_cuda else None
    if capability is not None and capability[0] == 9:
        dsl_error = sm90_dsa_cutedsl_requirement_error()
        if dsl_error is not None:
            raise RuntimeError(dsl_error)
        from . import _interface_sm90 as _iface_sm90

        q, kv, topk_idxs, attn_sink, topk_length = _iface_sm90.normalize_wrapper_inputs(q, kv, topk_idxs, attn_sink, topk_length, stream)
    num_heads = int(q.shape[1]) if q.ndim == 3 else None
    head_dim = int(q.shape[2]) if q.ndim == 3 else None
    # The API object owns lifecycle state, not compiled shape descriptors.
    # Execute-time normalization revalidates every tensor, so dynamic
    # sequence/top-k extents and layouts can safely share it. Device identity
    # already fixes capability; querying it again here only adds host overhead.
    key = (
        q.device,
        num_heads,
        head_dim,
        q.dtype,
        kv.dtype,
        topk_idxs.dtype,
        attn_sink is not None,
        topk_length is not None,
        int(indexer_topk),
        None if capability is None else tuple(capability),
    )
    obj = _cache_of_sparse_attention_forward_objects.get(key)
    if obj is None:
        obj = SparseAttentionForward(
            sample_q=q,
            sample_kv=kv,
            sample_topk_idxs=topk_idxs,
            sample_attn_sink=attn_sink,
            sample_topk_length=topk_length,
            indexer_topk=indexer_topk,
        )
        obj.check_support()
        obj.compile()
        _cache_of_sparse_attention_forward_objects[key] = obj

    outputs = {}
    if capability is not None and capability[0] == 9:
        with _iface_sm100._launch_context(q.device, stream) as (stream, consumer):
            outputs["out"] = torch.empty((q.shape[0], q.shape[1], 512), dtype=q.dtype, device=q.device)
            outputs["max_logits"] = torch.empty((q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
            outputs["lse"] = torch.empty((q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
            outputs["lse_indexer"] = torch.empty_like(outputs["lse"]) if indexer_topk else None
            _iface_sm100._record_stream(tuple(outputs.values()), consumer)
    out, max_logits, lse, lse_indexer = obj.execute(
        q,
        kv,
        topk_idxs,
        attn_sink=attn_sink,
        topk_length=topk_length,
        softmax_scale=scale,
        out=outputs.get("out"),
        max_logits=outputs.get("max_logits"),
        lse=outputs.get("lse"),
        lse_indexer=outputs.get("lse_indexer"),
        current_stream=stream,
    )
    return TupleDict(out=out, max_logits=max_logits, lse=lse, lse_indexer=lse_indexer)
