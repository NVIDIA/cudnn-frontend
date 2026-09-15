# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""APIBase integration for the SM100 DSA sparse-prefill forward kernels."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import torch

from cudnn.api_base import APIBase, TupleDict
from . import _interface_sm100 as _iface_sm100

_SUPPORTED_VARIANTS = {
    (64, 512): (0, 512, 1024, 2048),
    (64, 576): (0, 512, 1024, 2048),
    (128, 512): (0, 512, 1024),
}


class SparseAttentionForward(APIBase):
    """Sparse prefill forward for the supported SM100 MQA shapes.

    Compilation is deferred to :meth:`execute` because the architecture
    interface owns padding, output allocation, and the structurally keyed
    compiled-kernel cache.  Runtime extents and normalized layouts remain
    dynamic.  ``compile()`` still performs the normal ``APIBase`` lifecycle
    gate.
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

    def check_support(self) -> bool:
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

        variant = (num_heads, head_dim)
        self._value_error_if(
            variant not in _SUPPORTED_VARIANTS,
            "SparseAttentionForward supports only (H, D_qk) in " f"{tuple(_SUPPORTED_VARIANTS)}, got {variant}",
        )
        if variant in _SUPPORTED_VARIANTS:
            supported_indexer_topk = _SUPPORTED_VARIANTS[variant]
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
        self._runtime_error_if(
            capability[0] != 10,
            f"SparseAttentionForward requires an SM100-family GPU, found SM{capability[0]}{capability[1]}",
        )
        # This also rejects capabilities without an architecture-specific
        # compiler target (for example an unmapped SM10x).
        _iface_sm100._gpu_arch_flag(ref_device)

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.head_dim_v = 512
        self.logical_topk = logical_topk
        self._is_supported = True
        return True

    def compile(self) -> None:
        self._ensure_support_checked()
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
        if self._compiled_kernel is None:
            raise ValueError("SparseAttentionForward kernel not compiled")
        scale = self.softmax_scale if softmax_scale is None else float(softmax_scale)
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
    """Run SM100 sparse-prefill attention and return four stable keys.

    ``lse_indexer`` is ``None`` when ``indexer_topk == 0``.  The wrapper
    accepts arbitrary logical K and pads indices internally to a multiple of
    64 with invalid ``-1`` slots.
    """

    scale = 1.0 / math.sqrt(q.shape[-1]) if softmax_scale is None else float(softmax_scale)
    num_heads = int(q.shape[1]) if q.ndim == 3 else None
    head_dim = int(q.shape[2]) if q.ndim == 3 else None
    # The API object owns lifecycle state, not compiled shape descriptors.
    # Execute-time normalization revalidates every tensor, so dynamic
    # sequence/top-k extents and layouts can safely share it.
    key = (
        q.device,
        torch.cuda.get_device_capability(q.device) if q.device.type == "cuda" else None,
        num_heads,
        head_dim,
        q.dtype,
        kv.dtype,
        topk_idxs.dtype,
        attn_sink is not None,
        topk_length is not None,
        int(indexer_topk),
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

    out, max_logits, lse, lse_indexer = obj.execute(
        q,
        kv,
        topk_idxs,
        attn_sink=attn_sink,
        topk_length=topk_length,
        softmax_scale=scale,
        current_stream=stream,
    )
    return TupleDict(out=out, max_logits=max_logits, lse=lse, lse_indexer=lse_indexer)
