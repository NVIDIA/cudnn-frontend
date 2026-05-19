"""APIBase wrapper for DeepSeek Sparse Attention backward.

The wrapper dispatches to the Hopper (SM90) or Blackwell (SM100) CuTe DSL
implementation based on the active CUDA device. It consumes the ``out`` and
``lse`` tensors produced by the DSA sparse-attention forward path.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import cuda.bindings.driver as cuda

from cudnn.api_base import APIBase, TupleDict

from . import _interface_sm100 as _iface_sm100


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
        self.block_tile = int(block_tile)
        self.softmax_scale = softmax_scale

    def check_support(self) -> bool:
        major, _ = torch.cuda.get_device_capability()
        self._runtime_error_if(
            major < 9,
            f"SparseAttentionBackward requires SM90+, found SM{major}",
        )
        self._value_error_if(
            self.q_desc.ndim != 3,
            f"Q must be 3-D (total_S_q, H, D), got {self.q_desc.shape}",
        )
        self._value_error_if(
            self.kv_desc.ndim != 2,
            f"KV must be 2-D (total_S_kv, D), got {self.kv_desc.shape}",
        )
        self._check_dtype(self.q_desc, [torch.float16, torch.bfloat16], name="Q")
        self._check_dtype(
            self.kv_desc,
            self.q_desc.dtype,
            name="KV",
            extra_error_msg="KV must have same dtype as Q",
        )
        self._check_dtype(self.lse_desc, torch.float32, name="LSE")
        self._check_dtype(self.attn_sink_desc, torch.float32, name="attn_sink")
        self._check_dtype(self.topk_idxs_desc, torch.int32, name="topk_idxs")

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
        major, _ = torch.cuda.get_device_capability()
        scale = self.softmax_scale if softmax_scale is None else softmax_scale
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
        assert obj.check_support()
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
