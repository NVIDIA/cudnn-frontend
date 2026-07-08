"""APIBase wrapper for DeepSeek Sparse Attention backward.

The wrapper dispatches to the Hopper (SM90) or Blackwell (SM100) CuTe DSL
implementation based on the active CUDA device. It consumes the ``out`` and
``lse`` tensors produced by the DSA sparse-attention forward path.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import cuda.bindings.driver as cuda

from cudnn import data_type
from cudnn.api_base import APIBase, TupleDict

from . import _interface_sm100 as _iface_sm100
from .op import SparseAttentionBackwardOp


def _compact_for_kernel(desc):
    """Describe tensors that the Torch backends normalize before lowering."""

    return desc.compact_like(
        cudnn_dtype=desc.cudnn_dtype,
        shape=desc.shape,
        name=desc.name,
        init_value=desc.init_value,
    )


class SparseAttentionBackward(APIBase):
    def __init__(
        self,
        sample_q: torch.Tensor,  # (total_S_q, H, D) BF16
        sample_kv: torch.Tensor,  # (total_S_kv, D) BF16 (K=V)
        sample_out: torch.Tensor,  # (total_S_q, H, D_v)
        sample_dout: torch.Tensor,  # (total_S_q, H, D_v)
        sample_lse: torch.Tensor,  # (total_S_q, H) FP32, KV-only LSE
        sample_attn_sink: torch.Tensor,  # (H,) FP32
        sample_topk_idxs: torch.Tensor,  # (total_S_q, topk_max) INT32
        sample_dq: Optional[torch.Tensor] = None,
        sample_dkv: Optional[torch.Tensor] = None,
        sample_topk_length: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        block_tile: int = 64,
    ):
        super().__init__()
        self.q_desc = self._make_tensor_desc(sample_q, name="sample_q")
        self.kv_desc = self._make_tensor_desc(sample_kv, name="sample_kv")
        self.out_desc = self._make_tensor_desc(sample_out, name="sample_out")
        self.dout_desc = self._make_tensor_desc(sample_dout, name="sample_dout")
        self.lse_desc = self._make_tensor_desc(sample_lse, name="sample_lse")
        self.attn_sink_desc = self._make_tensor_desc(sample_attn_sink, name="sample_attn_sink")
        self.topk_idxs_desc = self._make_tensor_desc(sample_topk_idxs, name="sample_topk_idxs")
        self.topk_length_desc = self._make_tensor_desc(sample_topk_length, name="sample_topk_length")
        self.dq_desc = (
            self._make_tensor_desc(sample_dq, name="sample_dq")
            if sample_dq is not None
            else self.q_desc.compact_like(
                cudnn_dtype=self.q_desc.cudnn_dtype,
                shape=self.q_desc.shape,
                name="sample_dq",
            )
        )
        self.dkv_desc = (
            self._make_tensor_desc(sample_dkv, name="sample_dkv")
            if sample_dkv is not None
            else self.kv_desc.compact_like(
                cudnn_dtype=self.kv_desc.cudnn_dtype,
                shape=self.kv_desc.shape,
                name="sample_dkv",
            )
        )
        self.d_sink_desc = self.attn_sink_desc.compact_like(
            cudnn_dtype=data_type.FLOAT,
            shape=self.attn_sink_desc.shape,
            name="sample_d_sink",
        )
        self._op = SparseAttentionBackwardOp(
            q=_compact_for_kernel(self.q_desc),
            kv=_compact_for_kernel(self.kv_desc),
            output=_compact_for_kernel(self.out_desc),
            doutput=_compact_for_kernel(self.dout_desc),
            lse=_compact_for_kernel(self.lse_desc),
            attn_sink=self.attn_sink_desc,
            topk_idxs=self.topk_idxs_desc,
            topk_length=self.topk_length_desc,
            dq=self.dq_desc,
            dkv=self.dkv_desc,
            d_sink=self.d_sink_desc,
            softmax_scale=softmax_scale,
            block_tile=block_tile,
        )
        self.block_tile = self._op.block_tile
        self.softmax_scale = softmax_scale

    def check_support(self) -> bool:
        major, minor = torch.cuda.get_device_capability(self.q_desc.device)
        compute_capability = major * 10 + minor
        self._runtime_error_if(
            compute_capability < 90,
            f"SparseAttentionBackward requires SM90+, found SM{compute_capability}",
        )
        self._op.check_support()
        if compute_capability >= 100 and self.q_desc.cudnn_dtype != data_type.BFLOAT16:
            raise ValueError("SparseAttentionBackward on SM100+ currently requires bfloat16 inputs")

        devices = {
            desc.device
            for desc in (
                self.q_desc,
                self.kv_desc,
                self.out_desc,
                self.dout_desc,
                self.lse_desc,
                self.attn_sink_desc,
                self.topk_idxs_desc,
                self.topk_length_desc,
                self.dq_desc,
                self.dkv_desc,
                self.d_sink_desc,
            )
            if desc is not None
        }
        self._value_error_if(len(devices) != 1, f"All tensors must be on the same device, got {sorted(map(str, devices))}")

        self._is_supported = True
        return True

    def compile(self) -> None:
        self._ensure_support_checked()
        # The architecture-specific interfaces manage their own compile caches.
        # Priming requires real tensors, so compilation is deferred to execute().
        self._compiled_kernel = True

    def execute(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        out: torch.Tensor,
        dout: torch.Tensor,
        lse: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        dq: Optional[torch.Tensor] = None,
        dkv: Optional[torch.Tensor] = None,
        topk_length: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        major, _ = torch.cuda.get_device_capability(q.device)
        scale = self._op.softmax_scale if softmax_scale is None else softmax_scale
        if major == 9:
            from . import _interface_sm90 as _iface_sm90

            return _iface_sm90.flash_attn_bwd_sm90(
                q,
                kv,
                out,
                dout,
                lse,
                attn_sink=attn_sink,
                topk_idxs=topk_idxs,
                softmax_scale=scale,
                topk_length=topk_length,
                dq=dq,
                dkv=dkv,
                need_d_sink=True,
                current_stream=current_stream,
            )
        return _iface_sm100.flash_attn_bwd_sm100(
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            softmax_scale=scale,
            topk_length=topk_length,
            dq=dq,
            dkv=dkv,
            current_stream=current_stream,
        )


_cache_of_SparseAttentionBackwardObjects: dict = {}


def sparse_attention_backward_wrapper(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: Optional[float] = None,
    topk_length: Optional[torch.Tensor] = None,
    dq: Optional[torch.Tensor] = None,
    dkv: Optional[torch.Tensor] = None,
    block_tile: int = 64,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """High-level wrapper. Returns ``{'dq', 'dkv', 'd_sink'}``.

    Dispatches to SM90 or SM100 based on the active CUDA device. The returned
    ``d_sink`` is computed from ``attn_sink`` and ``dout``.
    """
    key = (
        q.dtype,
        q.shape,
        kv.shape,
        out.shape,
        dout.shape,
        lse.shape,
        attn_sink.shape,
        topk_idxs.shape,
        topk_length is not None,
        int(block_tile),
        softmax_scale,
    )
    obj = _cache_of_SparseAttentionBackwardObjects.get(key)
    if obj is None:
        obj = SparseAttentionBackward(
            sample_q=q,
            sample_kv=kv,
            sample_out=out,
            sample_dout=dout,
            sample_lse=lse,
            sample_attn_sink=attn_sink,
            sample_topk_idxs=topk_idxs,
            sample_topk_length=topk_length,
            softmax_scale=softmax_scale,
            block_tile=block_tile,
        )
        obj.check_support()
        obj.compile()
        _cache_of_SparseAttentionBackwardObjects[key] = obj

    dq_out, dkv_out, d_sink_out = obj.execute(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        dq=dq,
        dkv=dkv,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        current_stream=stream,
    )
    return TupleDict(dq=dq_out, dkv=dkv_out, d_sink=d_sink_out)
