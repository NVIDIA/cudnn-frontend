# SPDX-License-Identifier: Apache-2.0

"""Public API for the Triton NVFP4 QAT attention backward kernels."""

from __future__ import annotations

import math
from collections import OrderedDict
from contextlib import contextmanager
from typing import Iterator, Optional

import cuda.bindings.driver as cuda
import torch

from cudnn.api_base import APIBase, TupleDict

from ._workspace import nvfp4_workspace_layout

_SUPPORTED_CAPABILITIES = {(10, 0), (10, 3), (12, 0), (12, 1)}


@contextmanager
def _stream_context(current_stream: Optional[cuda.CUstream], device: torch.device) -> Iterator[None]:
    """Run torch allocations and Triton launches on ``current_stream``."""
    with torch.cuda.device(device):
        if current_stream is None:
            yield
            return
        stream_handle = int(current_stream)
        torch_current = torch.cuda.current_stream(device)
        if stream_handle == torch_current.cuda_stream:
            yield
            return
        torch_default = torch.cuda.default_stream(device)
        launch_stream = torch_default if stream_handle == torch_default.cuda_stream else torch.cuda.ExternalStream(stream_handle, device=device)
        with torch.cuda.stream(launch_stream):
            yield


class Nvfp4AttentionQatBackward(APIBase):
    """Explicit lifecycle API for NVFP4 fake-quantized SDPA backward.

    Q, K, V, the high-precision forward output, and dO use contiguous BHSD
    storage. ``lse`` is the natural-log softmax statistic with shape
    ``(B, H, S_q)``. The caller supplies output tensors and a byte workspace to
    :meth:`execute`; the convenience wrapper below owns those allocations.
    """

    def __init__(
        self,
        sample_q: torch.Tensor,
        sample_k: torch.Tensor,
        sample_v: torch.Tensor,
        sample_high_precision_o: torch.Tensor,
        sample_do: torch.Tensor,
        sample_lse: torch.Tensor,
        *,
        is_causal: bool = False,
        softmax_scale: Optional[float] = None,
    ):
        """Capture the tensor contract and compile-time attention options."""
        super().__init__()
        self._warn_experimental_api()
        self.q_desc = self._make_tensor_desc(sample_q, name="q")
        self.k_desc = self._make_tensor_desc(sample_k, name="k")
        self.v_desc = self._make_tensor_desc(sample_v, name="v")
        self.high_precision_o_desc = self._make_tensor_desc(sample_high_precision_o, name="high_precision_o")
        self.do_desc = self._make_tensor_desc(sample_do, name="dO")
        self.lse_desc = self._make_tensor_desc(sample_lse, name="lse")
        self.is_causal = bool(is_causal)
        self.softmax_scale = None if softmax_scale is None else float(softmax_scale)
        self._workspace_bytes: Optional[int] = None
        self._launch_config: Optional[tuple[int, int, int, int, int]] = None

    def check_support(self) -> bool:
        """Validate the contract and derive workspace and launch metadata."""
        activations = (
            self.q_desc,
            self.k_desc,
            self.v_desc,
            self.high_precision_o_desc,
            self.do_desc,
        )
        for desc in activations:
            self._value_error_if(desc.ndim != 4, f"{desc.name} must be rank-4 BHSD, got shape {desc.shape}")
            self._check_dtype(desc, torch.bfloat16, name=desc.name)
            self._value_error_if(not desc.is_contiguous(), f"{desc.name} must use contiguous BHSD storage")

        self._value_error_if(self.lse_desc.ndim != 3, f"lse must be rank-3 BHS, got shape {self.lse_desc.shape}")
        self._check_dtype(self.lse_desc, torch.float32, name="lse")
        self._value_error_if(not self.lse_desc.is_contiguous(), "lse must be contiguous")

        batch, heads, seqlen_q, head_dim = self.q_desc.shape
        batch_k, heads_k, seqlen_kv, head_dim_k = self.k_desc.shape
        self._value_error_if(min(batch, heads, seqlen_q, seqlen_kv) < 1, "B, H, S_q, and S_kv must be positive")
        self._value_error_if((batch_k, heads_k) != (batch, heads), "q and k must have matching batch and head counts")
        self._value_error_if(self.v_desc.shape != self.k_desc.shape, "v must have the same shape as k")
        self._value_error_if(head_dim != 128 or head_dim_k != 128, f"NVFP4 attention QAT backward requires D=128, got {head_dim} and {head_dim_k}")
        expected_q_shape = (batch, heads, seqlen_q, head_dim)
        self._value_error_if(self.high_precision_o_desc.shape != expected_q_shape, "high_precision_o must have the same shape as q")
        self._value_error_if(self.do_desc.shape != expected_q_shape, "dO must have the same shape as q")
        self._value_error_if(self.lse_desc.shape != (batch, heads, seqlen_q), f"lse must have shape {(batch, heads, seqlen_q)}")
        self._value_error_if(self.is_causal and seqlen_q != seqlen_kv, "causal QAT backward requires equal query and key/value sequence lengths")

        device = self.q_desc.device
        self._runtime_error_if(device.type != "cuda", f"NVFP4 attention QAT backward requires CUDA tensors, got {device}")
        for desc in (*activations[1:], self.lse_desc):
            self._value_error_if(desc.device != device, f"{desc.name} must be on {device}, got {desc.device}")
        capability = torch.cuda.get_device_capability(device)
        self._not_implemented_error_if(
            capability not in _SUPPORTED_CAPABILITIES,
            f"NVFP4 attention QAT backward supports SM100, SM103, SM120, and SM121, found SM{capability[0]}{capability[1]}",
        )

        if self.softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(head_dim)
        self._value_error_if(not math.isfinite(self.softmax_scale) or self.softmax_scale <= 0.0, "softmax_scale must be finite and positive")

        _, self._workspace_bytes = nvfp4_workspace_layout(
            self.q_desc.shape,
            self.k_desc.shape,
            self.v_desc.shape,
            self.lse_desc.shape,
        )

        optimized_sm100 = capability == (10, 0) and not self.is_causal and seqlen_kv % 16 == 0
        block_size = 64 if optimized_sm100 else 32
        num_warps = 8 if optimized_sm100 else 4
        fallback_stages = 2 if capability[0] == 12 and max(seqlen_q, seqlen_kv) >= 8192 else 3
        dq_num_stages = 2 if optimized_sm100 else fallback_stages
        dkdv_num_stages = 3 if optimized_sm100 else fallback_stages
        self._launch_config = (block_size, block_size, num_warps, dq_num_stages, dkdv_num_stages)
        self._is_supported = True
        return True

    def compile(self) -> None:
        """Compile the shape- and architecture-specialized Triton kernels."""
        self._ensure_support_checked()
        assert self.softmax_scale is not None
        assert self._launch_config is not None
        block_m, block_n, num_warps, dq_num_stages, dkdv_num_stages = self._launch_config
        with torch.cuda.device(self.q_desc.device):
            from ._interface import compile_nvfp4_attention_qat_backward

            self._compiled_kernel = compile_nvfp4_attention_qat_backward(
                self.q_desc.shape,
                self.k_desc.shape,
                self.q_desc.stride,
                self.k_desc.stride,
                softmax_scale=self.softmax_scale,
                is_causal=self.is_causal,
                block_m=block_m,
                block_n=block_n,
                num_warps=num_warps,
                dq_num_stages=dq_num_stages,
                dkdv_num_stages=dkdv_num_stages,
            )

    def scratch_workspace_bytes(self) -> int:
        """Return the caller-owned workspace size required by ``execute``."""
        self._ensure_support_checked()
        assert self._workspace_bytes is not None
        return self._workspace_bytes

    @staticmethod
    def _validate_runtime_tensor(tensor: torch.Tensor, desc, name: str) -> None:
        """Require a runtime tensor to match its plan-time descriptor."""
        if tuple(tensor.shape) != desc.shape:
            raise ValueError(f"{name} must have shape {desc.shape}, got {tuple(tensor.shape)}")
        if tensor.dtype != desc.dtype:
            raise ValueError(f"{name} must have dtype {desc.dtype}, got {tensor.dtype}")
        if tuple(tensor.stride()) != desc.stride:
            raise ValueError(f"{name} must have strides {desc.stride}, got {tuple(tensor.stride())}")
        if tensor.device != desc.device:
            raise ValueError(f"{name} must be on {desc.device}, got {tensor.device}")
        if tensor.data_ptr() % 16 != 0:
            raise ValueError(f"{name} address must be 16-byte aligned")

    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        high_precision_o_tensor: torch.Tensor,
        do_tensor: torch.Tensor,
        lse_tensor: torch.Tensor,
        dq_tensor: torch.Tensor,
        dk_tensor: torch.Tensor,
        dv_tensor: torch.Tensor,
        workspace: torch.Tensor,
        *,
        softmax_scale: Optional[float] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        """Launch the compiled backward into caller-owned outputs and workspace."""
        if self._compiled_kernel is None:
            raise RuntimeError("Nvfp4AttentionQatBackward is not compiled")

        runtime_inputs = (
            (q_tensor, self.q_desc, "q"),
            (k_tensor, self.k_desc, "k"),
            (v_tensor, self.v_desc, "v"),
            (high_precision_o_tensor, self.high_precision_o_desc, "high_precision_o"),
            (do_tensor, self.do_desc, "dO"),
            (lse_tensor, self.lse_desc, "lse"),
        )
        for tensor, desc, name in runtime_inputs:
            self._validate_runtime_tensor(tensor, desc, name)
        for tensor, desc, name in (
            (dq_tensor, self.q_desc, "dQ"),
            (dk_tensor, self.k_desc, "dK"),
            (dv_tensor, self.v_desc, "dV"),
        ):
            self._validate_runtime_tensor(tensor, desc, name)

        required_workspace = self.scratch_workspace_bytes()
        if workspace.dtype != torch.uint8 or workspace.ndim != 1 or not workspace.is_contiguous():
            raise ValueError("workspace must be a contiguous rank-1 torch.uint8 tensor")
        if workspace.device != q_tensor.device:
            raise ValueError(f"workspace must be on {q_tensor.device}, got {workspace.device}")
        if workspace.numel() < required_workspace:
            raise ValueError(f"workspace must contain at least {required_workspace} bytes, got {workspace.numel()}")
        if workspace.data_ptr() % 16 != 0:
            raise ValueError("workspace address must be 16-byte aligned")

        scale = self.softmax_scale if softmax_scale is None else float(softmax_scale)
        assert scale is not None
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("softmax_scale must be finite and positive")
        assert self._launch_config is not None
        block_m, block_n, num_warps, dq_num_stages, dkdv_num_stages = self._launch_config

        with _stream_context(current_stream, q_tensor.device):
            from ._interface import run_nvfp4_attention_qat_backward

            run_nvfp4_attention_qat_backward(
                q_tensor,
                k_tensor,
                v_tensor,
                high_precision_o_tensor,
                do_tensor,
                lse_tensor,
                dq_tensor,
                dk_tensor,
                dv_tensor,
                workspace,
                softmax_scale=scale,
                is_causal=self.is_causal,
                block_m=block_m,
                block_n=block_n,
                num_warps=num_warps,
                dq_num_stages=dq_num_stages,
                dkdv_num_stages=dkdv_num_stages,
            )


_OBJECT_CACHE_LIMIT = 64
_OBJECT_CACHE: OrderedDict[tuple[object, ...], Nvfp4AttentionQatBackward] = OrderedDict()


def nvfp4_attention_qat_backward(
    do_tensor: torch.Tensor,
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    high_precision_o_tensor: torch.Tensor,
    lse_tensor: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    is_causal: bool = False,
    dq_tensor: Optional[torch.Tensor] = None,
    dk_tensor: Optional[torch.Tensor] = None,
    dv_tensor: Optional[torch.Tensor] = None,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Compute STE gradients for NVFP4 fake-quantized scaled dot-product attention.

    ``high_precision_o_tensor`` is the forward ``softmax(QK^T) @ V`` value
    formed from fake-quantized Q/K/V before probability fake quantization.
    ``lse_tensor`` is the matching natural-log softmax statistic.
    """

    if q_tensor.ndim != 4:
        raise ValueError(f"q must be rank-4 BHSD, got shape {tuple(q_tensor.shape)}")
    requested_scale = None if softmax_scale is None else float(softmax_scale)
    key = tuple(
        (tensor.device, tuple(tensor.shape), tensor.dtype, tuple(tensor.stride()))
        for tensor in (q_tensor, k_tensor, v_tensor, high_precision_o_tensor, do_tensor, lse_tensor)
    ) + (bool(is_causal),)
    op = _OBJECT_CACHE.get(key)
    if op is not None:
        _OBJECT_CACHE.move_to_end(key)
    else:
        op = Nvfp4AttentionQatBackward(
            q_tensor,
            k_tensor,
            v_tensor,
            high_precision_o_tensor,
            do_tensor,
            lse_tensor,
            is_causal=is_causal,
            softmax_scale=requested_scale,
        )
        op.check_support()
        op.compile()
        _OBJECT_CACHE[key] = op
        if len(_OBJECT_CACHE) > _OBJECT_CACHE_LIMIT:
            _OBJECT_CACHE.popitem(last=False)

    scale = 1.0 / math.sqrt(q_tensor.shape[-1]) if requested_scale is None else requested_scale
    with _stream_context(current_stream, q_tensor.device):
        if dq_tensor is None:
            dq_tensor = torch.empty_strided(q_tensor.shape, q_tensor.stride(), dtype=q_tensor.dtype, device=q_tensor.device)
        if dk_tensor is None:
            dk_tensor = torch.empty_strided(k_tensor.shape, k_tensor.stride(), dtype=k_tensor.dtype, device=k_tensor.device)
        if dv_tensor is None:
            dv_tensor = torch.empty_strided(v_tensor.shape, v_tensor.stride(), dtype=v_tensor.dtype, device=v_tensor.device)
        workspace = torch.empty(op.scratch_workspace_bytes(), dtype=torch.uint8, device=q_tensor.device)
        op.execute(
            q_tensor,
            k_tensor,
            v_tensor,
            high_precision_o_tensor,
            do_tensor,
            lse_tensor,
            dq_tensor,
            dk_tensor,
            dv_tensor,
            workspace,
            softmax_scale=scale,
            current_stream=current_stream,
        )

    return TupleDict(dq_tensor=dq_tensor, dk_tensor=dk_tensor, dv_tensor=dv_tensor)
