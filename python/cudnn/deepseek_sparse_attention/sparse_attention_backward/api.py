# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""APIBase wrapper for DeepSeek Sparse Attention backward.

The wrapper dispatches to the Hopper (SM90) or Blackwell (SM100) CuTe DSL
implementation based on the active CUDA device. It consumes the ``out`` and
``lse`` tensors produced by the DSA sparse-attention forward path.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import cuda.bindings.driver as cuda

from cudnn.deepseek_sparse_attention.utils.runtime import device_capability

from cudnn.api_base import APIBase, TupleDict
from cudnn._torch_stream import contiguous_on_stream
from cudnn.deepseek_sparse_attention.utils.runtime import resolve_stream, torch_stream_context

from . import _interface_sm100 as _iface_sm100
from . import _interface_sm100_d576 as _iface_d576


class SparseAttentionBackward(APIBase):
    """Validated architecture-dispatch wrapper for DSA backward."""

    def __init__(
        self,
        sample_q: torch.Tensor,  # (total_S_q, H, D) FP16/BF16
        sample_kv: torch.Tensor,  # (total_S_kv, D) FP16/BF16 (K=V)
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
        deterministic: bool = False,
    ):
        """Capture the sample tensor contract and execution policy."""
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
        self.deterministic = bool(deterministic)
        self._backend = None
        self._two_cta_split_count = 1

    def check_support(self) -> bool:
        """Validate the device, dtype, shape, and deterministic contracts."""
        self._value_error_if(self.q_desc.device.type != "cuda", f"Q must live on CUDA, got {self.q_desc.device}")
        capability = torch.cuda.get_device_capability(self.q_desc.device)
        major, _ = capability
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
        self._check_dtype(self.out_desc, self.q_desc.dtype, name="out", extra_error_msg="out must have same dtype as Q")
        self._check_dtype(self.dout_desc, self.q_desc.dtype, name="dout", extra_error_msg="dout must have same dtype as Q")
        if self.topk_length_desc is not None:
            self._check_dtype(self.topk_length_desc, torch.int32, name="topk_length")

        # Device placement + cross-tensor device consistency. The SM90/SM100
        # kernels are CUDA-only and reject CPU or cross-device inputs at
        # execution time (see the is_cuda / same-device assert in
        # ``_interface_sm100.flash_attn_bwd_sm100``), so a placement mismatch
        # must fail the support gate here rather than compile/launch and crash.
        ref_device = self.q_desc.device
        named_descriptors = [
            ("q", self.q_desc),
            ("kv", self.kv_desc),
            ("out", self.out_desc),
            ("dout", self.dout_desc),
            ("lse", self.lse_desc),
            ("attn_sink", self.attn_sink_desc),
            ("topk_idxs", self.topk_idxs_desc),
        ]
        if self.topk_length_desc is not None:
            named_descriptors.append(("topk_length", self.topk_length_desc))
        descriptors = [desc for _, desc in named_descriptors]
        self._value_error_if(
            any(desc.device != ref_device for desc in descriptors),
            f"All inputs must share Q's device {ref_device}, got {[desc.device for desc in descriptors]}",
        )

        # Cross-tensor shape contract: every companion tensor is indexed with
        # coordinates derived from Q, so a mismatched shape silently reads or
        # writes out of place at execution time instead of failing.
        total_s_q, num_heads, head_dim = self.q_desc.shape
        self._value_error_if(
            self.deterministic and (major != 10 or num_heads not in _iface_sm100._DETERMINISTIC_HEAD_COUNTS),
            f"deterministic DSA backward requires SM100 and heads in {_iface_sm100._DETERMINISTIC_HEAD_COUNTS}, found SM{major} H{num_heads}",
        )
        # The SM100 kernel is tiled only for head_dim in {512, 576} (the 576
        # MLA case splits QK=576 / V=512); any other head_dim compiles to a
        # layout that indexes shared memory out of bounds and crashes.
        self._value_error_if(
            head_dim not in (512, 576),
            f"head_dim must be 512 or 576, got {head_dim}",
        )
        head_dim_v = 512 if head_dim == 576 else head_dim
        expected_o_shape = (total_s_q, num_heads, head_dim_v)
        self._value_error_if(
            self.kv_desc.shape[1] != head_dim,
            f"KV must have shape (total_S_kv, {head_dim}), got {self.kv_desc.shape}",
        )
        self._value_error_if(
            self.out_desc.shape != expected_o_shape,
            f"out must have shape {expected_o_shape}, got {self.out_desc.shape}",
        )
        self._value_error_if(
            self.dout_desc.shape != expected_o_shape,
            f"dout must have shape {expected_o_shape}, got {self.dout_desc.shape}",
        )
        self._value_error_if(
            self.lse_desc.shape != (total_s_q, num_heads),
            f"LSE must have shape {(total_s_q, num_heads)}, got {self.lse_desc.shape}",
        )
        self._value_error_if(
            self.attn_sink_desc.shape != (num_heads,),
            f"attn_sink must have shape {(num_heads,)}, got {self.attn_sink_desc.shape}",
        )
        self._value_error_if(
            self.topk_idxs_desc.ndim != 2 or self.topk_idxs_desc.shape[0] != total_s_q,
            f"topk_idxs must have shape ({total_s_q}, topk_max), got {self.topk_idxs_desc.shape}",
        )
        if self.topk_length_desc is not None:
            self._value_error_if(
                self.topk_length_desc.shape != (total_s_q,),
                f"topk_length must have shape {(total_s_q,)}, got {self.topk_length_desc.shape}",
            )

        # Resolve the SM100 backend once so compile()/execute() and the wrapper
        # agree on the route; the interface repeats the selection for direct
        # callers of ``flash_attn_bwd_sm100``.
        self._backend = None
        if major == 10:
            self._backend, _ = _iface_sm100._select_sm100_backend(
                num_heads,
                head_dim,
                head_dim_v=head_dim_v,
                dtype=self.q_desc.dtype,
                max_topk=self.topk_idxs_desc.shape[1],
                device_capability=capability,
                deterministic=self.deterministic,
                is_contiguous=all(desc.is_contiguous() for desc in descriptors),
            )
        if self._backend == _iface_d576.BACKEND:
            self._value_error_if(total_s_q <= 0 or self.kv_desc.shape[0] <= 0, "Q and KV sequence extents must be positive")
            self._two_cta_split_count = _iface_d576._split_count(
                total_s_q, self.topk_idxs_desc.shape[1], torch.cuda.get_device_properties(self.q_desc.device).multi_processor_count
            )
        else:
            # The kernels read the declared layout natively and execute never
            # repacks (R5); the D576 route already rejects strides at execute.
            for name, desc in named_descriptors:
                self._not_implemented_error_if(
                    not desc.is_contiguous(),
                    f"SparseAttentionBackward addresses only contiguous {name}; got shape {desc.shape} strides {desc.stride}",
                )

        self._is_supported = True
        return True

    def compile(self) -> None:
        self._ensure_support_checked()
        if self._backend == _iface_d576.BACKEND:
            with torch.cuda.device(self.q_desc.device):
                self._compiled_kernel = _iface_d576._compile_d576_2cta(
                    torch.cuda.get_device_capability(self.q_desc.device),
                    self.topk_idxs_desc.shape[1],
                    self.topk_length_desc is not None,
                    self.q_desc.shape[0] == 1,
                    self._two_cta_split_count > 1,
                )
            return
        # The architecture-specific interfaces manage their own compile caches.
        # Priming requires real tensors, so compilation is deferred to execute().
        self._compiled_kernel = True

    def scratch_workspace_bytes(self) -> int:
        """Return reusable per-execution scratch for the selected backend."""
        self._ensure_support_checked()
        major, _ = device_capability(self.q_desc.device)
        total_s_q, num_heads, head_dim = self.q_desc.shape
        total_s_kv = self.kv_desc.shape[0]
        if major == 9:
            from . import _interface_sm90 as _iface_sm90

            return _iface_sm90.flash_attn_bwd_sm90_workspace_size(total_s_q, total_s_kv, head_dim, num_heads)
        return _iface_sm100.flash_attn_bwd_sm100_workspace_size(
            total_s_q,
            total_s_kv,
            head_dim,
            num_heads,
            self.deterministic,
        )

    def execute(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        out: torch.Tensor,
        dout: torch.Tensor,
        lse: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        dq: torch.Tensor,
        dkv: torch.Tensor,
        topk_length: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        current_stream: Optional[cuda.CUstream] = None,
        workspace: Optional[torch.Tensor] = None,
        *,
        d_sink: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dispatch one validated execution to the active GPU architecture.

        Every backend requires ``dq``, ``dkv``, ``d_sink`` and (when
        ``scratch_workspace_bytes()`` is non-zero) ``workspace`` to be
        caller-provided; execute never allocates. ``d_sink`` is keyword-only so
        the positional order of the pre-existing parameters is unchanged. The
        H128/D576 two-CTA route additionally never compiles here.
        """
        for name, tensor in (("dq", dq), ("dkv", dkv), ("d_sink", d_sink)):
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"{name} must be preallocated; use sparse_attention_backward_wrapper for automatic output allocation")
        scale = self.softmax_scale if softmax_scale is None else softmax_scale
        if self._backend == _iface_d576.BACKEND:
            if self._compiled_kernel is None:
                raise RuntimeError("call compile() before execute()")
            # Execution must match the declaration used for routing/compilation.
            # In particular, neither presence nor absence of lengths is ignored.
            for name, tensor, desc in (
                ("q", q, self.q_desc),
                ("kv", kv, self.kv_desc),
                ("out", out, self.out_desc),
                ("dout", dout, self.dout_desc),
                ("lse", lse, self.lse_desc),
                ("attn_sink", attn_sink, self.attn_sink_desc),
                ("topk_idxs", topk_idxs, self.topk_idxs_desc),
                ("topk_length", topk_length, self.topk_length_desc),
            ):
                if (tensor is None) != (desc is None):
                    raise ValueError(f"{name} presence must match the compiled plan")
                if desc is not None and (tuple(tensor.shape) != desc.shape or tensor.dtype != desc.dtype or tensor.device != desc.device):
                    raise ValueError(f"{name} shape, dtype, and device must match the compiled plan")
            return _iface_d576._execute_d576_2cta(
                self._compiled_kernel,
                q,
                kv,
                out,
                dout,
                lse,
                attn_sink,
                topk_idxs,
                topk_length,
                dq,
                dkv,
                d_sink,
                workspace,
                scale,
                self._two_cta_split_count,
                current_stream,
            )
        # Resolve the architecture from Q's device rather than the ambient current
        # device, and launch under that device context, matching check_support().
        major, _ = device_capability(q.device)
        with torch.cuda.device(q.device):
            if major == 9:
                from . import _interface_sm90 as _iface_sm90

                return _iface_sm90.flash_attn_bwd_sm90(
                    q,
                    kv,
                    out,
                    dout,
                    lse,
                    attn_sink,
                    topk_idxs,
                    dq,
                    dkv,
                    d_sink,
                    workspace,
                    softmax_scale=scale,
                    topk_length=topk_length,
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
                d_sink=d_sink,
                deterministic=self.deterministic,
                workspace=workspace,
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
    deterministic: bool = False,
    stream: Optional[cuda.CUstream] = None,
    workspace: Optional[torch.Tensor] = None,
) -> TupleDict:
    """High-level wrapper. Returns ``{'dq', 'dkv', 'd_sink'}``.

    Dispatches to SM90 or SM100 from the input device and tensor metadata. The
    returned ``d_sink`` is computed from ``attn_sink`` and ``dout``. Set
    ``deterministic=True`` for bitwise-reproducible
    H16/H32/H64/H96/H128 gradients on SM100. The optional reusable uint8
    ``workspace`` must hold at least
    ``SparseAttentionBackward.scratch_workspace_bytes()`` bytes.
    """
    # The plan addresses contiguous tensors only (check_support declines the rest, R5).
    # The wrapper is the eager torch-op layer: it normalises strided inputs here, on the
    # launch stream, as it always did, and stages a strided caller output so the
    # caller's tensor keeps its identity.
    with torch.cuda.device(q.device):
        launch_stream = resolve_stream(stream)
    # R1 staging: each original is record_stream'ed on the launch stream before it is rebound.
    q, kv, out, dout, lse, attn_sink, topk_idxs, topk_length = (
        contiguous_on_stream(t, launch_stream, q.device) for t in (q, kv, out, dout, lse, attn_sink, topk_idxs, topk_length)
    )
    key = (
        q.device,
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
        bool(deterministic),
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
            deterministic=deterministic,
        )
        assert obj.check_support()
        obj.compile()
        _cache_of_SparseAttentionBackwardObjects[key] = obj

    dq_user, dkv_user = dq, dkv
    with torch.cuda.device(q.device):
        # execute() never allocates: outputs and scratch are provided here, ordered with the launch stream (R2).
        with torch_stream_context(launch_stream):
            if workspace is None:
                workspace_bytes = obj.scratch_workspace_bytes()
                if workspace_bytes:
                    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=q.device)
            if dq is None:
                dq = torch.empty_like(q)
            elif not dq.is_contiguous():
                dq = torch.empty(dq.shape, dtype=dq.dtype, device=dq.device)  # staged; copied back into the caller's tensor below
            if dkv is None:
                dkv = torch.empty_like(kv)
            elif not dkv.is_contiguous():
                dkv = torch.empty(dkv.shape, dtype=dkv.dtype, device=dkv.device)
            d_sink = torch.empty_like(attn_sink)

    dq_out, dkv_out, d_sink_out = obj.execute(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        dq,
        dkv,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        workspace=workspace,
        current_stream=launch_stream,
        d_sink=d_sink,
    )
    if (dq_user is not None and dq_user is not dq_out) or (dkv_user is not None and dkv_user is not dkv_out):
        with torch.cuda.device(q.device), torch_stream_context(launch_stream):
            if dq_user is not None and dq_user is not dq_out:
                dq_user.copy_(dq_out)
                dq_out = dq_user
            if dkv_user is not None and dkv_user is not dkv_out:
                dkv_user.copy_(dkv_out)
                dkv_out = dkv_user
    return TupleDict(dq=dq_out, dkv=dkv_out, d_sink=d_sink_out)
