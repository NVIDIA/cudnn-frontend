# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Tuple
import logging
import math

from cuda.bindings import driver as cuda
import cutlass
import torch

import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_stream

from cudnn.api_base import APIBase, TupleDict
from cudnn.datatypes import _convert_to_cutlass_data_type

from .fmha_forward_sm100_d256 import BlackwellFusedMultiHeadAttentionForward
from ..fmha_utils import MaskEnum


class SdpafwdSm100D256(APIBase):
    """API class for d=256 SDPA forward (SM100+) using the FE OSS CUTE DSL kernel."""

    def __init__(
        self,
        sample_q: torch.Tensor,
        sample_k: torch.Tensor,
        sample_v: torch.Tensor,
        sample_o: torch.Tensor,
        sample_lse: torch.Tensor,
        sample_cum_seqlen_q: Optional[torch.Tensor] = None,
        sample_cum_seqlen_k: Optional[torch.Tensor] = None,
        max_s_q: Optional[int] = None,
        max_s_k: Optional[int] = None,
        qk_acc_dtype: torch.dtype = torch.float32,
        pv_acc_dtype: torch.dtype = torch.float32,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        is_causal: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        scale_softmax: Optional[float] = None,
        scale_output: float = 1.0,
    ):
        super().__init__()
        self._kernel = BlackwellFusedMultiHeadAttentionForward

        self._warn_experimental_api()
        self._logger.debug("Entering __init__")

        if sample_cum_seqlen_q is not None:
            if sample_cum_seqlen_q.numel() < 2:
                raise ValueError("sample_cum_seqlen_q must contain at least 2 elements")
            self._sample_s_q_max = int((sample_cum_seqlen_q[1:] - sample_cum_seqlen_q[:-1]).max().item())
        else:
            self._sample_s_q_max = None

        if sample_cum_seqlen_k is not None:
            if sample_cum_seqlen_k.numel() < 2:
                raise ValueError("sample_cum_seqlen_k must contain at least 2 elements")
            self._sample_s_k_max = int((sample_cum_seqlen_k[1:] - sample_cum_seqlen_k[:-1]).max().item())
        else:
            self._sample_s_k_max = None

        self.s_q_max = int(max_s_q) if max_s_q is not None else self._sample_s_q_max
        self.s_k_max = int(max_s_k) if max_s_k is not None else self._sample_s_k_max

        self.q_desc = self._make_tensor_desc(sample_q, name="q")
        self.k_desc = self._make_tensor_desc(sample_k, name="k")
        self.v_desc = self._make_tensor_desc(sample_v, name="v")
        self.o_desc = self._make_tensor_desc(sample_o, name="o")
        self.lse_desc = self._unpad_tensor_to_ndim(self._make_tensor_desc(sample_lse, name="lse"), self.q_desc.ndim - 1, "lse")
        self.cum_seqlen_q_desc = self._unpad_tensor_to_ndim(self._make_tensor_desc(sample_cum_seqlen_q, name="cum_seqlen_q"), 1, "cum_seqlen_q")
        self.cum_seqlen_k_desc = self._unpad_tensor_to_ndim(self._make_tensor_desc(sample_cum_seqlen_k, name="cum_seqlen_k"), 1, "cum_seqlen_k")

        self.qk_acc_dtype = qk_acc_dtype
        self.pv_acc_dtype = pv_acc_dtype
        self.mma_tiler_mn = mma_tiler_mn
        self.is_causal = is_causal
        self.window_size_left, self.window_size_right = window_size
        self.scale_softmax = scale_softmax
        self.scale_output = float(scale_output)

        self.input_layout = None
        self.dtype = None
        self.problem_size = None
        self.mask_type = None

        self.h_k = None
        self.h_q = None
        self.head_dim = None
        self.batch_size = None

        self._logger.debug("__init__ completed")

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        if self.cum_seqlen_q_desc is None and self.cum_seqlen_k_desc is None:
            self.input_layout = "B,H,S,D"
            for desc_name in ["q_desc", "k_desc", "v_desc", "o_desc"]:
                tensor_desc = getattr(self, desc_name)
                self._value_error_if(tensor_desc.ndim != 4, f"{tensor_desc.name} must be rank-4 for B,H,S,D layout, got {tensor_desc.ndim}")
                self._value_error_if(
                    tensor_desc.stride_order != (3, 1, 2, 0),
                    f"{tensor_desc.name} must have d,h,s,b stride order (3, 1, 2, 0), got {tensor_desc.stride_order}",
                )
                setattr(self, desc_name, tensor_desc.transpose(1, 2))
            self.lse_desc = self._unpad_tensor_to_ndim(self.lse_desc, 3, "lse")
            self._value_error_if(self.lse_desc is None, "sample_lse is required")
            self._value_error_if(not self.lse_desc.is_contiguous(), "lse_tensor must be contiguous for B,H,S,D layout")
        elif self.cum_seqlen_q_desc is not None and self.cum_seqlen_k_desc is not None:
            self.input_layout = "T,H,D"
            for desc_name in ["q_desc", "k_desc", "v_desc", "o_desc"]:
                tensor_desc = getattr(self, desc_name)
                if tensor_desc.ndim == 3:
                    setattr(self, desc_name, tensor_desc.unsqueeze(0))
                elif tensor_desc.ndim == 4:
                    self._value_error_if(
                        tensor_desc.shape[0] != 1,
                        f"{tensor_desc.name} must have batch dimension 1 for T,H,D layout (1, t, h, d), got {tensor_desc.shape[0]}",
                    )
                else:
                    raise ValueError(f"{tensor_desc.name} must be rank-3 or rank-4 for T,H,D layout, got {tensor_desc.ndim}")
            self.lse_desc = self._unpad_tensor_to_ndim(self.lse_desc, 2, "lse")
        else:
            raise ValueError(f"cum_seqlen_q and cum_seqlen_k must be both None or both not None, got {self.cum_seqlen_q_desc} and {self.cum_seqlen_k_desc}")

        b, s_qo, h_qo, d_qk = self.q_desc.shape
        _, s_kv, h_kv, d_v = self.v_desc.shape

        self._check_tensor_shape(self.q_desc, (b, s_qo, h_qo, d_qk), name="Q")
        self._check_tensor_shape(self.k_desc, (b, s_kv, h_kv, d_qk), name="K")
        self._check_tensor_shape(self.v_desc, (b, s_kv, h_kv, d_v), name="V")
        self._check_tensor_shape(self.o_desc, (b, s_qo, h_qo, d_v), name="O")

        self._value_error_if(self.lse_desc is None, "sample_lse is required")
        if self.input_layout == "B,H,S,D":
            self._check_tensor_shape(self.lse_desc, (b, h_qo, s_qo), name="LSE")
        else:
            self._check_tensor_shape(self.lse_desc, (s_qo, h_qo), name="LSE")

        self._value_error_if(d_qk != d_v, f"D_qk must match D_v, got {d_qk} and {d_v}")
        self._value_error_if(h_qo % h_kv != 0, f"H_q must be divisible by H_k, got {h_qo} and {h_kv}")

        self.dtype = self._check_dtype(self.q_desc, [torch.float16, torch.bfloat16], name="Q")
        for tensor_desc in [self.k_desc, self.v_desc, self.o_desc]:
            self._check_dtype(tensor_desc, self.dtype, name=tensor_desc.name, extra_error_msg=f"{tensor_desc.name} must match Q dtype")
        self._check_dtype(self.lse_desc, torch.float32, name="LSE")
        self._check_dtype(self.qk_acc_dtype, torch.float32, name="qk_acc_dtype", extra_error_msg="Only float32 accumulator is supported")
        self._check_dtype(self.pv_acc_dtype, torch.float32, name="pv_acc_dtype", extra_error_msg="Only float32 accumulator is supported")

        if self.input_layout == "T,H,D":
            self._check_dtype(self.cum_seqlen_q_desc, [torch.int32], name="cum_seqlen_q")
            self._check_dtype(self.cum_seqlen_k_desc, [torch.int32], name="cum_seqlen_k")
            self._value_error_if(
                self.cum_seqlen_q_desc.shape != self.cum_seqlen_k_desc.shape,
                f"cum_seqlen_q and cum_seqlen_k must have same shape, got {self.cum_seqlen_q_desc.shape} and {self.cum_seqlen_k_desc.shape}",
            )
            self.batch_size = int(self.cum_seqlen_q_desc.shape[0] - 1)
            self._value_error_if(self.batch_size <= 0, f"Invalid varlen batch_size={self.batch_size}")
            self._value_error_if(self._sample_s_q_max is None, "sample_cum_seqlen_q is required for T,H,D layout")
            self._value_error_if(self._sample_s_k_max is None, "sample_cum_seqlen_k is required for T,H,D layout")
            if self.s_q_max is None:
                self.s_q_max = self._sample_s_q_max
            if self.s_k_max is None:
                self.s_k_max = self._sample_s_k_max
            self._value_error_if(
                self.s_q_max < self._sample_s_q_max,
                f"max_s_q must be >= inferred max from sample_cum_seqlen_q ({self._sample_s_q_max}), got {self.s_q_max}",
            )
            self._value_error_if(
                self.s_k_max < self._sample_s_k_max,
                f"max_s_k must be >= inferred max from sample_cum_seqlen_k ({self._sample_s_k_max}), got {self.s_k_max}",
            )
        else:
            self.batch_size = b
            self.s_q_max = s_qo
            self.s_k_max = s_kv

        self.h_k = h_kv
        self.h_q = h_qo
        self.head_dim = d_qk

        self._value_error_if(self.head_dim != 256, f"head_dim must be 256, got {self.head_dim}")
        self._value_error_if(self.mma_tiler_mn != (128, 128), f"mma_tiler_mn must be (128, 128), got {self.mma_tiler_mn}")

        if self.scale_softmax is None or self.scale_softmax == 0.0:
            self.scale_softmax = 1.0 / math.sqrt(self.head_dim)

        if self.is_causal:
            self.window_size_right = 0

        self._value_error_if(
            self.window_size_left >= self.s_k_max - 1,
            f"window_size_left must be less than s_k_max - 1 (s_k_max={self.s_k_max}), got {self.window_size_left}",
        )
        self._value_error_if(
            self.window_size_right >= self.s_q_max - 1,
            f"window_size_right must be less than s_q_max - 1 (s_q_max={self.s_q_max}), got {self.window_size_right}",
        )
        if not self.is_causal:
            self._not_implemented_error_if(
                (self.window_size_left, self.window_size_right) != (-1, -1),
                f"window_size must be (-1, -1) for non-causal mode, got {self.window_size_left} and {self.window_size_right}",
            )

        self.mask_type = MaskEnum.WINDOW_MASK_INFERENCE
        if (not self.is_causal) and (self.window_size_left, self.window_size_right) == (-1, -1):
            if self.input_layout == "T,H,D" or s_kv % self.mma_tiler_mn[1] != 0:
                self.mask_type = MaskEnum.RESIDUAL_MASK

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = self.q_desc.device
        major, minor = torch.cuda.get_device_capability(device)
        compute_capability = major * 10 + minor
        if compute_capability < 100:
            raise RuntimeError(f"SdpafwdSm100D256 requires SM100+, found SM{compute_capability} on device {device}")

        self.problem_size = (
            self.batch_size,
            self.s_q_max,
            self.s_k_max,
            self.h_q,
            self.h_k,
            self.head_dim,
        )

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def compile(self) -> None:
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            self._logger.debug("Kernel already compiled; skipping recompilation")
            return

        sdpa_fwd = self._kernel(
            qk_acc_dtype=_convert_to_cutlass_data_type(self.qk_acc_dtype),
            pv_acc_dtype=_convert_to_cutlass_data_type(self.pv_acc_dtype),
            mma_tiler=(*self.mma_tiler_mn, self.head_dim),
            is_persistent=False,
            mask_type=self.mask_type,
        )
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
        scale_softmax_log2 = self.scale_softmax * math.log2(math.exp(1.0))
        window_size_left = None if self.window_size_left < 0 else cutlass.Int32(self.window_size_left)
        window_size_right = None if self.window_size_right < 0 else cutlass.Int32(self.window_size_right)

        self._logger.debug("Compiling sdpa forward kernel with cute.compile")
        _compiled_kernel = cute.compile(
            sdpa_fwd,
            q_tensor=self._make_fake_cute_tensor_from_desc(self.q_desc, assumed_align=64),
            k_tensor=self._make_fake_cute_tensor_from_desc(self.k_desc, assumed_align=64),
            v_tensor=self._make_fake_cute_tensor_from_desc(self.v_desc, assumed_align=64),
            o_tensor=self._make_fake_cute_tensor_from_desc(self.o_desc, assumed_align=64),
            problem_size=self.problem_size,
            cum_seqlen_q=self._make_fake_cute_tensor_from_desc(self.cum_seqlen_q_desc, assumed_align=16),
            cum_seqlen_k=self._make_fake_cute_tensor_from_desc(self.cum_seqlen_k_desc, assumed_align=16),
            lse_tensor=self._make_fake_cute_tensor_from_desc(self.lse_desc, assumed_align=16),
            scale_softmax_log2=scale_softmax_log2,
            scale_softmax=self.scale_softmax,
            scale_output=self.scale_output,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            stream=fake_stream,
            options="--enable-tvm-ffi",
        )

        def tensor_api(
            q_tensor: torch.Tensor,
            k_tensor: torch.Tensor,
            v_tensor: torch.Tensor,
            o_tensor: torch.Tensor,
            lse_tensor: torch.Tensor,
            cum_seqlen_q: Optional[torch.Tensor],
            cum_seqlen_k: Optional[torch.Tensor],
            scale_softmax: float,
            scale_output: float,
            stream: cuda.CUstream,
        ) -> None:
            if self.input_layout == "B,H,S,D":
                q_tensor, k_tensor, v_tensor, o_tensor = (
                    q_tensor.transpose(1, 2),
                    k_tensor.transpose(1, 2),
                    v_tensor.transpose(1, 2),
                    o_tensor.transpose(1, 2),
                )
                lse_tensor = self._unpad_tensor_to_ndim(lse_tensor, 3, "lse_tensor")
            elif self.input_layout == "T,H,D":
                q_tensor, k_tensor, v_tensor, o_tensor = (
                    q_tensor.unsqueeze(0) if q_tensor.ndim == 3 else q_tensor,
                    k_tensor.unsqueeze(0) if k_tensor.ndim == 3 else k_tensor,
                    v_tensor.unsqueeze(0) if v_tensor.ndim == 3 else v_tensor,
                    o_tensor.unsqueeze(0) if o_tensor.ndim == 3 else o_tensor,
                )
                lse_tensor = self._unpad_tensor_to_ndim(lse_tensor, 2, "lse_tensor")
                cum_seqlen_q = self._unpad_tensor_to_ndim(cum_seqlen_q, 1, "cum_seqlen_q")
                cum_seqlen_k = self._unpad_tensor_to_ndim(cum_seqlen_k, 1, "cum_seqlen_k")
            else:
                raise NotImplementedError(f"Invalid input layout: {self.input_layout}")

            _compiled_kernel(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                self.problem_size,
                cum_seqlen_q,
                cum_seqlen_k,
                lse_tensor,
                scale_softmax * math.log2(math.exp(1.0)),
                scale_softmax,
                scale_output,
                window_size_left,
                window_size_right,
                stream,
            )

        self._compiled_kernel = tensor_api
        self._logger.debug("Kernel compiled successfully")

    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        lse_tensor: torch.Tensor,
        cum_seqlen_q_tensor: Optional[torch.Tensor] = None,
        cum_seqlen_k_tensor: Optional[torch.Tensor] = None,
        scale_softmax: Optional[float] = None,
        scale_output: Optional[float] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        self._logger.debug("Entering execute")
        current_stream = self._get_default_stream(current_stream)

        if self._compiled_kernel is None:
            raise RuntimeError("SdpafwdSm100D256 kernel is not compiled")

        if self.input_layout == "T,H,D":
            self._value_error_if(cum_seqlen_q_tensor is None, "cum_seqlen_q_tensor is required for T,H,D layout")
            self._value_error_if(cum_seqlen_k_tensor is None, "cum_seqlen_k_tensor is required for T,H,D layout")
        elif self.input_layout == "B,H,S,D":
            self._value_error_if(cum_seqlen_q_tensor is not None, "cum_seqlen_q_tensor must be None for B,H,S,D layout")
            self._value_error_if(cum_seqlen_k_tensor is not None, "cum_seqlen_k_tensor must be None for B,H,S,D layout")
        else:
            raise NotImplementedError(f"Invalid input layout: {self.input_layout}")

        scale_softmax_val = self.scale_softmax if (scale_softmax is None or scale_softmax == 0.0) else scale_softmax
        scale_output_val = self.scale_output if scale_output is None else float(scale_output)

        self._compiled_kernel(
            q_tensor=q_tensor,
            k_tensor=k_tensor,
            v_tensor=v_tensor,
            o_tensor=o_tensor,
            lse_tensor=lse_tensor,
            cum_seqlen_q=cum_seqlen_q_tensor,
            cum_seqlen_k=cum_seqlen_k_tensor,
            scale_softmax=scale_softmax_val,
            scale_output=scale_output_val,
            stream=current_stream,
        )
        self._logger.debug("Execute completed")


_logger = logging.getLogger(__name__)
_cache_of_SdpafwdSm100D256Objects = {}


def _allocate_lse_tensor(
    q_tensor: torch.Tensor,
    cum_seqlen_q_tensor: Optional[torch.Tensor],
) -> torch.Tensor:
    if cum_seqlen_q_tensor is None:
        if q_tensor.ndim != 4:
            raise ValueError(f"Expected BHSD q_tensor to be rank-4, got {q_tensor.ndim}")
        return torch.empty((q_tensor.shape[0], q_tensor.shape[1], q_tensor.shape[2]), dtype=torch.float32, device=q_tensor.device)

    if q_tensor.ndim == 3:
        return torch.empty((q_tensor.shape[0], q_tensor.shape[1]), dtype=torch.float32, device=q_tensor.device)
    if q_tensor.ndim == 4:
        return torch.empty((q_tensor.shape[1], q_tensor.shape[2]), dtype=torch.float32, device=q_tensor.device)
    raise ValueError(f"Expected THD q_tensor to be rank-3 or rank-4, got {q_tensor.ndim}")


def sdpa_fwd_wrapper_sm100_d256(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    cum_seqlen_q_tensor: Optional[torch.Tensor] = None,
    cum_seqlen_k_tensor: Optional[torch.Tensor] = None,
    max_s_q: Optional[int] = None,
    max_s_k: Optional[int] = None,
    qk_acc_dtype: torch.dtype = torch.float32,
    pv_acc_dtype: torch.dtype = torch.float32,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    is_causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    scale_softmax: Optional[float] = None,
    scale_output: float = 1.0,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Convenience wrapper for the d=256 SDPA forward SM100 kernel."""

    o_tensor = torch.empty_like(q_tensor)
    lse_tensor = _allocate_lse_tensor(q_tensor, cum_seqlen_q_tensor)

    cache_max_s_q = max_s_q
    cache_max_s_k = max_s_k
    if cache_max_s_q is None and cum_seqlen_q_tensor is not None:
        cache_max_s_q = int((cum_seqlen_q_tensor[1:] - cum_seqlen_q_tensor[:-1]).max().item())
    if cache_max_s_k is None and cum_seqlen_k_tensor is not None:
        cache_max_s_k = int((cum_seqlen_k_tensor[1:] - cum_seqlen_k_tensor[:-1]).max().item())

    cache_key = (
        q_tensor.shape,
        k_tensor.shape,
        v_tensor.shape,
        q_tensor.stride(),
        k_tensor.stride(),
        v_tensor.stride(),
        q_tensor.dtype,
        k_tensor.dtype,
        v_tensor.dtype,
        cum_seqlen_q_tensor.shape if cum_seqlen_q_tensor is not None else None,
        cum_seqlen_q_tensor.stride() if cum_seqlen_q_tensor is not None else None,
        cum_seqlen_q_tensor.dtype if cum_seqlen_q_tensor is not None else None,
        cum_seqlen_k_tensor.shape if cum_seqlen_k_tensor is not None else None,
        cum_seqlen_k_tensor.stride() if cum_seqlen_k_tensor is not None else None,
        cum_seqlen_k_tensor.dtype if cum_seqlen_k_tensor is not None else None,
        cache_max_s_q,
        cache_max_s_k,
        qk_acc_dtype,
        pv_acc_dtype,
        mma_tiler_mn,
        is_causal,
        window_size,
        scale_softmax,
        float(scale_output),
        q_tensor.device,
    )

    if cache_key in _cache_of_SdpafwdSm100D256Objects:
        _logger.debug("sdpa_fwd_wrapper_sm100_d256: Using cached SdpafwdSm100D256 object")
        sdpa_fwd = _cache_of_SdpafwdSm100D256Objects[cache_key]
    else:
        _logger.debug("sdpa_fwd_wrapper_sm100_d256: No cached object found, creating new SdpafwdSm100D256 object")
        sdpa_fwd = SdpafwdSm100D256(
            sample_q=q_tensor,
            sample_k=k_tensor,
            sample_v=v_tensor,
            sample_o=o_tensor,
            sample_lse=lse_tensor,
            sample_cum_seqlen_q=cum_seqlen_q_tensor,
            sample_cum_seqlen_k=cum_seqlen_k_tensor,
            max_s_q=max_s_q,
            max_s_k=max_s_k,
            qk_acc_dtype=qk_acc_dtype,
            pv_acc_dtype=pv_acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            is_causal=is_causal,
            window_size=window_size,
            scale_softmax=scale_softmax,
            scale_output=scale_output,
        )
        assert sdpa_fwd.check_support(), "Unsupported configuration"
        sdpa_fwd.compile()
        _cache_of_SdpafwdSm100D256Objects[cache_key] = sdpa_fwd

    sdpa_fwd.execute(
        q_tensor=q_tensor,
        k_tensor=k_tensor,
        v_tensor=v_tensor,
        o_tensor=o_tensor,
        lse_tensor=lse_tensor,
        cum_seqlen_q_tensor=cum_seqlen_q_tensor,
        cum_seqlen_k_tensor=cum_seqlen_k_tensor,
        scale_softmax=scale_softmax,
        scale_output=scale_output,
        current_stream=current_stream,
    )

    return TupleDict(
        o_tensor=o_tensor,
        lse_tensor=lse_tensor,
    )


# ===========================================================================
# SM80 (A100)
# ===========================================================================
"""cuDNN-frontend wrapper around the SM80 (A100) SDPA prefill kernels.

The kernels live at ``kernels/prefill_f16_sm80.py`` (generic) and
``kernels/prefill_d256_f16_sm80.py`` (d=256) — vendored from a since-retired
internal tile repo (provenance: ``kernels/__init__.py``), building on the
shared FROST tile library ``cudnn/frost/tile_dsl/``.  This file is the
thin adapter that:

  1. Maps the cuDNN-frontend BHSD tensor convention to the BSHD layout
     the kernel expects (zero-cost — torch.transpose just swaps
     strides).
  2. Picks the smallest kernel "flavor" (gptoss / llama / dsv3 / qwen)
     whose ``(D_QK, D_V)`` envelope covers the user's head dim and
     zero-pads V along the ``d_v`` axis when ``D_V_user < flavor.D_V``.
     Q/K padding is handled by the kernel's own ``is_even_k = False``
     path.
  3. Implements the ``cudnn.api_base.APIBase`` contract
     (``check_support`` / ``compile`` / ``execute``) so the kernel
     plugs into ``cudnn.sdpa`` alongside the SM100 d=256 path.  The
     ``cudnn.frost.sdpa`` forward engine lowers onto this adapter.

Coverage today:
  * dtype:     FP16 / BF16
  * mask:      none / causal / SWA / bottom-right alignment / per-batch
               padding (``seq_kv_lens`` + optional ``seq_len_q``) /
               causal right-band widening (``window_size_right > 0``)
  * features:  bias, ALiBi, learned sinks, block_mask (128x128), fused
               RoPE, LSE + score_max / score_sum_exp stats
  * varlen:    packed THD via ``cum_seqlen_q/k`` (wrapper-level path)
  * scheduler: auto (default / lpt / lpt_l2 picked from per-flavor table)
  * layout:    BHSD (B, H, S, D) logical, BSHD-physical stride order
               (3, 1, 2, 0), size-1 dims wildcarded
  * GQA:       H_q % H_kv == 0
  * head dims: any (D_QK, D_V) inside the qwen (256, 256) envelope
"""  # noqa: E501 — SM80 section notes

# ---------------------------------------------------------------------------
# Lazy import of the kernel.
# ---------------------------------------------------------------------------
# ``cute.compile`` pulls in cutlass and traces a kernel on first
# invocation — ~1.5 s.  Defer the kernel module import until the user
# actually calls ``compile()`` / ``execute()`` so ``import cudnn`` stays
# fast and the CuTe DSL is not hard-required just to import the package.
_KERNEL_MOD = {}


def _stream_ctx(current_stream):
    """Context manager dispatching onto ``current_stream`` (a ``cuda.CUstream``
    or raw stream int); the kernels launch on torch's current stream, so an
    ExternalStream context routes them.  ``None`` keeps the current stream, and
    a raw handle equal to torch's current/default stream reuses that torch
    stream object rather than wrapping it: ``ExternalStream(0)`` breaks
    re-execution on some torch builds (NGC), where every launch after the
    compile run silently no-ops (all-zero outputs; caught by test_mhas_v2's
    determinism re-run).  Mirrors gemm/cutedsl/grouped/backend_utils.py."""
    import contextlib

    if current_stream is None:
        return contextlib.nullcontext()
    handle = int(current_stream)
    torch_current = torch.cuda.current_stream()
    if handle in (0, 1, 2) or handle == torch_current.cuda_stream:
        return contextlib.nullcontext()
    torch_default = torch.cuda.default_stream()
    if handle == torch_default.cuda_stream:
        return torch.cuda.stream(torch_default)
    return torch.cuda.stream(torch.cuda.ExternalStream(handle))


# Flavors that route to the dedicated d=256 kernel (symmetric K+V prefetch);
# all others use the shared generic kernel.  Mirrors the upstream
# ``_sm80_dispatch._SM80_D256_FLAVORS``.
_D256_FLAVORS = ("qwen",)


def _load_kernel_module(flavor: str = ""):
    """Lazily import + cache the SM80 kernel module for ``flavor``.  qwen (d=256)
    routes to ``prefill_d256_f16_sm80`` (symmetric K+V prefetch); the rest
    use ``prefill_f16_sm80``."""
    key = "d256" if flavor in _D256_FLAVORS else "f16"
    if key not in _KERNEL_MOD:
        if key == "d256":
            from .kernels import prefill_d256_f16_sm80 as _mod
        else:
            from .kernels import prefill_f16_sm80 as _mod
        _KERNEL_MOD[key] = _mod
    return _KERNEL_MOD[key]


# ---------------------------------------------------------------------------
# Flavor tables — sourced from config_sm80 (the single source of truth, like
# config_sm100 / config_sm120 for the DSL engines).
# ---------------------------------------------------------------------------
from . import config_sm80

_FLAVOR_CFGS = {
    "gptoss": config_sm80.GPTOSS_CFG,
    "llama": config_sm80.LLAMA_CFG,
    "dsv3": config_sm80.DSV3_CFG,
    "qwen": config_sm80.QWEN_CFG,
}

# (D_QK, D_V) envelope per flavor.
_FLAVOR_DIMS = {name: (cfg.D_QK, cfg.D_V) for name, cfg in _FLAVOR_CFGS.items()}

# (tile_m, num_warps, tile_n) per flavor — frozen from the A100 perf sweep.
_FLAVOR_KNOBS = {name: (cfg.TILE_M, cfg.NUM_WARPS, cfg.TILE_N) for name, cfg in _FLAVOR_CFGS.items()}

# Causal L2 budget (MiB) for ``sched=lpt_l2`` per flavor.  Larger d_qk
# inflates the per-(B, H) resident set so dsv3 needs a smaller group.
_FLAVOR_CAUSAL_L2_MIB = {
    "llama": 16,
    "gptoss": 16,
    "dsv3": 8,
    "qwen": 8,
}

# Ascending (D_QK, D_V) order so _pick_flavor walks closest-from-above.
_SUPPORTED_FLAVORS = ("gptoss", "llama", "dsv3", "qwen")


def _pick_flavor(d_qk: int, d_v: int) -> str:
    """Smallest kernel flavor whose ``(D_QK, D_V)`` envelope covers
    ``(d_qk, d_v)``.  Exact-match wins when both axes match; otherwise
    walk gptoss → llama → dsv3 → qwen and pick the first that fits.
    Raises if nothing fits (heads bigger than the qwen envelope are not
    supported on SM80 yet)."""
    for flavor in _SUPPORTED_FLAVORS:
        fdqk, fdv = _FLAVOR_DIMS[flavor]
        if d_qk == fdqk and d_v == fdv:
            return flavor
    for flavor in _SUPPORTED_FLAVORS:
        fdqk, fdv = _FLAVOR_DIMS[flavor]
        if d_qk <= fdqk and d_v <= fdv:
            return flavor
    raise ValueError(
        f"SM80 SDPA: no flavor envelope covers (D_QK={d_qk}, D_V={d_v}).  "
        f"Supported envelopes: {_FLAVOR_DIMS}.  Heads larger than qwen "
        "(256/256) are not yet ported to SM80."
    )


def _pad_last_dim(t: torch.Tensor, new_last: int) -> torch.Tensor:
    """Zero-pad the trailing dim of a fp16 tensor up to ``new_last``."""
    old_last = t.shape[-1]
    if old_last == new_last:
        return t
    if old_last > new_last:
        raise ValueError(f"_pad_last_dim: tensor's last dim {old_last} exceeds target {new_last}")
    pad = torch.zeros(
        (*t.shape[:-1], new_last - old_last),
        dtype=t.dtype,
        device=t.device,
    )
    return torch.cat([t, pad], dim=-1).contiguous()


# ---------------------------------------------------------------------------
# Scheduler resolver — mirrors the upstream ``dispatch_sm80`` heuristic.
# ---------------------------------------------------------------------------
def _resolve_scheduler(
    *,
    scheduler: str,
    flavor: str,
    is_causal: bool,
    swa_window: int,
    skv: int,
) -> Tuple[str, int]:
    """Return ``(sched_token, sched_l2_mib)`` to pass to the kernel."""
    l2_mib = _FLAVOR_CAUSAL_L2_MIB[flavor]
    if scheduler == "auto":
        if is_causal:
            return "lpt_l2", l2_mib
        if swa_window > 0:
            # SWA heuristic — LPT wins for 1K ≤ SKV ≤ 16K.
            return ("lpt" if 1024 <= skv <= 16384 else "default"), l2_mib
        return "default", l2_mib
    if scheduler in ("natural", "default"):
        return "default", l2_mib
    if scheduler == "lpt":
        return "lpt", l2_mib
    if scheduler == "lpt_l2":
        return "lpt_l2", l2_mib
    raise ValueError(f"SM80 SDPA: scheduler must be 'auto' / 'default' / 'natural' / " f"'lpt' / 'lpt_l2', got {scheduler!r}")


# ---------------------------------------------------------------------------
# APIBase subclass.
# ---------------------------------------------------------------------------
class SdpafwdSm80(APIBase):
    """API class for SM80 (A100) SDPA forward.

    Mirrors the calling convention of ``SdpafwdSm100D256`` so callers
    can swap between SM80 / SM100 with no API changes.
    """

    def __init__(
        self,
        sample_q: torch.Tensor,
        sample_k: torch.Tensor,
        sample_v: torch.Tensor,
        sample_o: torch.Tensor,
        sample_lse: torch.Tensor,
        is_causal: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        scale_softmax: Optional[float] = None,
        scale_output: float = 1.0,
        scheduler: str = "auto",
        causal_bottom_right: bool = False,
        has_seq_kv_lens: bool = False,
        return_score_stats: bool = False,
    ):
        super().__init__()
        self._warn_experimental_api()
        self._logger.debug("Entering __init__")

        self.q_desc = self._make_tensor_desc(sample_q, name="q")
        self.k_desc = self._make_tensor_desc(sample_k, name="k")
        self.v_desc = self._make_tensor_desc(sample_v, name="v")
        self.o_desc = self._make_tensor_desc(sample_o, name="o")
        self.lse_desc = self._make_tensor_desc(sample_lse, name="lse")

        self.is_causal = is_causal
        self.window_size_left, self.window_size_right = window_size
        self.scale_softmax = scale_softmax
        self.scale_output = float(scale_output)
        self.scheduler = scheduler
        # Bottom-right causal alignment (cuDNN diagonal_alignment=BOTTOM_RIGHT)
        # and per-batch padded KV lengths (seq_kv_lens) — both compile-time
        # config on the kernel; the actual lengths flow in at execute().
        self.causal_bottom_right = bool(causal_bottom_right)
        self.has_seq_kv_lens = bool(has_seq_kv_lens)
        self.return_score_stats = bool(return_score_stats)

        # Filled by check_support().
        self.batch_size: Optional[int] = None
        self.s_q_max: Optional[int] = None
        self.s_k_max: Optional[int] = None
        self.h_q: Optional[int] = None
        self.h_kv: Optional[int] = None
        self.head_dim_qk: Optional[int] = None
        self.head_dim_v: Optional[int] = None
        self.flavor: Optional[str] = None
        self.flavor_d_qk: Optional[int] = None
        self.flavor_d_v: Optional[int] = None
        self.tile_m: Optional[int] = None
        self.num_warps: Optional[int] = None
        self.tile_n: Optional[int] = None
        self.sched_token: Optional[str] = None
        self.sched_l2_mib: Optional[int] = None
        self.mask_token: Optional[str] = None
        self.swa_window_runtime: int = 0
        self.right_bound: int = 0

        self._logger.debug("__init__ completed")

    # ------------------------------------------------------------------
    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        # ---- layout: B, H, S, D (the cuDNN-FE convention) -------------
        # Require BSHD-physical stride order (3, 1, 2, 0).  SIZE-1 dims (S=1
        # decode, H=1 MQA) make torch's contiguous-tensor stride order
        # ambiguous for that axis, so treat size-1 dims as wildcards: drop them
        # from both the actual and the required order before comparing — the
        # underlying data is still BSHD-contiguous and the kernel reads it fine.
        _REQ = (3, 1, 2, 0)
        for desc_name in ["q_desc", "k_desc", "v_desc", "o_desc"]:
            d = getattr(self, desc_name)
            self._value_error_if(
                d.ndim != 4,
                f"{d.name} must be rank-4 (B, H, S, D); got {d.ndim}",
            )
            _shape = d.shape
            _act = tuple(ax for ax in d.stride_order if _shape[ax] != 1)
            _exp = tuple(ax for ax in _REQ if _shape[ax] != 1)
            self._value_error_if(
                _act != _exp,
                f"{d.name} must have d, h, s, b stride order (3, 1, 2, 0) " f"(size-1 dims wildcarded); got {d.stride_order} shape {_shape}",
            )

        b, h_qo, s_qo, d_qk = self.q_desc.shape
        _, h_kv, s_kv, _ = self.k_desc.shape
        _, _, _, d_v = self.v_desc.shape

        self._check_tensor_shape(self.q_desc, (b, h_qo, s_qo, d_qk), name="Q")
        self._check_tensor_shape(self.k_desc, (b, h_kv, s_kv, d_qk), name="K")
        self._check_tensor_shape(self.v_desc, (b, h_kv, s_kv, d_v), name="V")
        self._check_tensor_shape(self.o_desc, (b, h_qo, s_qo, d_v), name="O")

        # ---- positive extents (kernel cannot launch on B/H/S = 0) ----
        for label, val in (
            ("B", b),
            ("H_q", h_qo),
            ("H_kv", h_kv),
            ("S_q", s_qo),
            ("S_kv", s_kv),
            ("D_QK", d_qk),
            ("D_V", d_v),
        ):
            self._value_error_if(
                int(val) <= 0,
                f"{label} must be > 0; got {val}",
            )

        self._value_error_if(
            h_qo % h_kv != 0,
            f"H_q ({h_qo}) must be divisible by H_kv ({h_kv}) for GQA / MQA",
        )

        # ---- head-dim envelope -------------------------------------------
        # The flavor envelope tops out at qwen (D_QK=256, D_V=256).  Reject
        # larger heads with a clear message instead of waiting for the
        # flavor picker to raise.
        max_d_qk = max(fdqk for fdqk, _ in _FLAVOR_DIMS.values())
        max_d_v = max(fdv for _, fdv in _FLAVOR_DIMS.values())
        self._value_error_if(
            d_qk > max_d_qk or d_v > max_d_v,
            f"SM80 SDPA: head dim (D_QK={d_qk}, D_V={d_v}) exceeds "
            f"supported envelope (D_QK<={max_d_qk}, D_V<={max_d_v}).  "
            f"Larger heads are not yet ported.",
        )

        # ---- dtype: FP16 or BF16 (both ride one SM80 mma pipeline) ----
        self.dtype = self._check_dtype(self.q_desc, [torch.float16, torch.bfloat16], name="Q")
        for desc in [self.k_desc, self.v_desc, self.o_desc]:
            self._check_dtype(
                desc,
                self.dtype,
                name=desc.name,
                extra_error_msg=f"{desc.name} must match Q dtype (FP16/BF16 on SM80)",
            )
        self._check_dtype(self.lse_desc, torch.float32, name="LSE")

        # LSE shape: cuDNN-FE convention is [B, H, S_q] (matches SM100).
        self._check_tensor_shape(self.lse_desc, (b, h_qo, s_qo), name="LSE")
        self._value_error_if(
            not self.lse_desc.is_contiguous(),
            "LSE must be contiguous on SM80",
        )

        # ---- arch ----------------------------------------------------
        self._value_error_if(
            not torch.cuda.is_available(),
            "CUDA must be available for SM80 SDPA",
        )
        device = self.q_desc.device
        major, minor = torch.cuda.get_device_capability(device)
        self._value_error_if(
            (major, minor) != (8, 0),
            f"SdpafwdSm80 requires SM80 (A100); found SM{major}{minor} on {device}",
        )

        # ---- flavor + knobs ------------------------------------------
        self.flavor = _pick_flavor(d_qk, d_v)
        self.flavor_d_qk, self.flavor_d_v = _FLAVOR_DIMS[self.flavor]
        self.tile_m, self.num_warps, self.tile_n = _FLAVOR_KNOBS[self.flavor]

        # Bottom-right alignment shifts the band to the corner; it must affect
        # SOMETHING — a causal upper bound (is_causal) and/or a left
        # sliding-window (window_size_left >= 0).  A bare BR with neither is a
        # no-op (dense), which the caller shouldn't request.
        self._value_error_if(
            self.causal_bottom_right and not (self.is_causal or self.window_size_left >= 0),
            "SM80 SDPA: causal_bottom_right requires is_causal=True and/or a " "left sliding-window (window_size_left >= 0).",
        )

        # ---- mask -----------------------------------------------------
        swa_left = self.window_size_left
        swa_right = self.window_size_right
        self.right_bound = 0
        # window_size_left W follows the cuDNN convention: attend to [q-W, q]
        # (W past tokens + self).  The kernel's swa_window IS that W directly
        # (it masks col < q - swa_window, i.e. keeps [q-swa_window, q]) — so
        # W==0 is a valid 1-token window (keep k>=q), NOT "no window".
        # swa_left < 0 = no left bound.
        if self.is_causal:
            # Causal-only (no window) → "causal".  Causal + left window →
            # "causal_swa" (MASK_CAUSAL|MASK_SWA — causal upper bound kept).
            # window_size_right > 0 widens the causal upper bound into a right
            # band (k <= q + right) via the kernel's runtime right_bound.
            self.mask_token = "causal" if swa_left < 0 else "causal_swa"
            self.swa_window_runtime = max(0, swa_left) if swa_left >= 0 else 0
            self.right_bound = max(0, swa_right)
        elif swa_left >= 0:
            # A left window alone selects SWA; window_size_right is only
            # meaningful with is_causal=True.
            self._not_implemented_error_if(
                swa_right > 0,
                "SM80 SDPA: non-causal SWA with window_size_right > 0 is not supported " "(window_size_right is only meaningful with is_causal=True)",
            )
            self.mask_token = "swa"
            self.swa_window_runtime = swa_left
        else:
            # window_size=(-1, r) without is_causal: a bare right bound has no
            # diagonal to anchor to — reject rather than silently pick a mask
            # (the THD path resolves the same input to "none").
            self._not_implemented_error_if(
                swa_right >= 0,
                "SM80 SDPA: window_size_right without a left window or is_causal=True has no effect; pass is_causal=True or a left window",
            )
            self.mask_token = "none"
            self.swa_window_runtime = 0

        # ---- scheduler -----------------------------------------------
        # Surface unsupported tokens up front (otherwise _resolve_scheduler
        # raises later with the same info — but pre-checking here keeps
        # check_support() side-effect-free on failure).
        _VALID_SCHED_TOKENS = ("auto", "natural", "default", "lpt", "lpt_l2")
        self._value_error_if(
            self.scheduler not in _VALID_SCHED_TOKENS,
            f"scheduler must be one of {_VALID_SCHED_TOKENS}; got {self.scheduler!r}",
        )
        self.sched_token, self.sched_l2_mib = _resolve_scheduler(
            scheduler=self.scheduler,
            flavor=self.flavor,
            is_causal=self.is_causal,
            swa_window=self.swa_window_runtime,
            skv=int(s_kv),
        )

        # ---- softmax scale -------------------------------------------
        if self.scale_softmax is None or self.scale_softmax == 0.0:
            self.scale_softmax = 1.0 / math.sqrt(d_qk)

        self.batch_size = int(b)
        self.s_q_max = int(s_qo)
        self.s_k_max = int(s_kv)
        self.h_q = int(h_qo)
        self.h_kv = int(h_kv)
        self.head_dim_qk = int(d_qk)
        self.head_dim_v = int(d_v)

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    # ------------------------------------------------------------------
    def compile(self) -> None:
        """Eager-warm the kernel's internal ``@lru_cache``.

        The kernel module owns its own per-shape ``cute.compile`` cache
        (see ``prefill_f16_sm80._compile_cached``).  We don't need
        to plumb a second cache here — calling ``forward()`` once with
        the sample shapes warms the binary so the first ``execute()``
        call is fast.  We deliberately skip the real launch by issuing
        a zero-element call when possible; today the kernel doesn't
        gracefully early-exit on B=0, so for now we just rely on the
        first ``execute()`` to trigger the compile.  That keeps memory
        traffic to zero here while still preserving the
        APIBase contract.
        """
        self._logger.debug("Entering compile (no-op — kernel self-caches)")
        self._ensure_support_checked()
        # Mark as "compiled" — the kernel's own lru_cache will JIT on
        # first execute() and reuse from then on.
        self._compiled_kernel = True
        self._logger.debug("compile completed")

    # ------------------------------------------------------------------
    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        lse_tensor: torch.Tensor,
        score_max_tensor: Optional[torch.Tensor] = None,
        score_sum_tensor: Optional[torch.Tensor] = None,
        scale_softmax: Optional[float] = None,
        scale_output: Optional[float] = None,
        current_stream: Optional[cuda.CUstream] = None,
        seq_kv_lens: Optional[torch.Tensor] = None,
        seq_len_q: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        alibi: bool = False,
        sinks: Optional[torch.Tensor] = None,
        cum_seqlen_q_tensor: Optional[torch.Tensor] = None,
        cum_seqlen_k_tensor: Optional[torch.Tensor] = None,
        max_s_q: Optional[int] = None,
        rope_freqs: Optional[torch.Tensor] = None,
        block_mask: Optional[torch.Tensor] = None,
    ) -> None:
        self._logger.debug("Entering execute")
        if self._compiled_kernel is None:
            raise RuntimeError("SdpafwdSm80 is not compiled")

        scale_softmax_val = self.scale_softmax if (scale_softmax is None or scale_softmax == 0.0) else float(scale_softmax)
        scale_output_val = self.scale_output if scale_output is None else float(scale_output)
        self._value_error_if(
            scale_output_val != 1.0,
            f"SM80 SDPA: scale_output != 1.0 is not supported yet (got {scale_output_val})",
        )

        kernel = _load_kernel_module(self.flavor)

        # --- Feature operands (bias / ALiBi / sink) -----------------------
        # ALiBi: the cuDNN matrix is a boolean flag; derive the per-Q-head
        # slopes from H with the standard power-of-2 recursion (matches
        # test_mhas.compute_ref + the kernel's forward divides by scale).
        alibi_slopes = None
        if alibi:
            h_q = q_tensor.shape[1]  # BHSD
            alibi_slopes = kernel.default_alibi_slopes(h_q).to(q_tensor.device)
        # Bias arrives as [1, H, S_q, S_kv] (head-major) — the kernel's forward
        # consumes exactly that shape (no BSHD transpose for the bias tile).
        bias_arg = bias_tensor
        self._not_implemented_error_if(
            cum_seqlen_q_tensor is not None,
            "SdpafwdSm80.execute does not serve ragged/THD (cum_seqlen_*) " "inputs; use sdpa_fwd_wrapper_sm80, whose THD path packs and " "launches them.",
        )

        # Convert BHSD → BSHD for the kernel; transpose is a stride-only
        # op so zero-cost.
        Q = q_tensor.transpose(1, 2).contiguous() if not q_tensor.transpose(1, 2).is_contiguous() else q_tensor.transpose(1, 2)
        K = k_tensor.transpose(1, 2).contiguous() if not k_tensor.transpose(1, 2).is_contiguous() else k_tensor.transpose(1, 2)
        V = v_tensor.transpose(1, 2).contiguous() if not v_tensor.transpose(1, 2).is_contiguous() else v_tensor.transpose(1, 2)
        # The kernel writes O in BSHD; we re-route by allocating a BSHD
        # scratch O and copying back into the user-provided BHSD O at
        # the end (we cannot pass the BHSD-strided O directly because
        # the kernel's TMA STG path expects contiguous BSHD strides).
        # Same with LSE: kernel writes [B, H, SQ] fp32 which matches
        # cuDNN-FE — no transpose needed.

        # Pad V along d_v if the flavor envelope is wider than the user's tensor.
        pad_v = self.head_dim_v < self.flavor_d_v
        if pad_v:
            V = _pad_last_dim(V, self.flavor_d_v)

        # Run the kernel.  Build the kwargs then drop any the chosen kernel's
        # forward() doesn't accept, so a future kernel with a narrower
        # signature degrades to a TypeError-free launch.  (Both current
        # kernels accept the full set, including ``seq_len_q``.)
        import inspect as _inspect

        _fwd_kwargs = dict(
            scale=scale_softmax_val,
            return_lse=True,
            tile_m=self.tile_m,
            num_warps=self.num_warps,
            tile_n=self.tile_n,
            d_qk=self.flavor_d_qk,
            d_v=self.flavor_d_v,
            mask=self.mask_token,
            swa_window=int(self.swa_window_runtime),
            right_bound=int(self.right_bound),
            causal_bottom_right=self.causal_bottom_right,
            seq_kv_lens=seq_kv_lens,
            seq_len_q=seq_len_q,
            bias=bias_arg,
            alibi_slopes=alibi_slopes,
            sinks=sinks,
            return_score_stats=self.return_score_stats,
            sched=self.sched_token,
            sched_l2_mib=self.sched_l2_mib,
            # RoPE freqs (cuDNN graph.rope `freqs`, [max_s,1,1,d_qk] angles).
            # The kernel rotates Q+K in SMEM; both SM80 kernels accept it
            # (filtered out below for any future kernel that doesn't).
            rope_freqs=rope_freqs,
            # Block-mask (cuDNN `block_mask`, bit-packed uint8 128x128 sparsity):
            # the kernel skips QK + SV mma for off blocks.
            block_mask=block_mask,
        )
        _accepted = _inspect.signature(kernel.forward).parameters
        _fwd_kwargs = {k: v for k, v in _fwd_kwargs.items() if k in _accepted}
        with _stream_ctx(current_stream):
            _res = kernel.forward(Q, K, V, **_fwd_kwargs)
        if self.return_score_stats:
            O_buf, LSE_buf, SMAX_buf, SSUM_buf = _res
        else:
            O_buf, LSE_buf = _res

        # Slice off any V-padding the wrapper applied, then copy back into the
        # user's BHSD-strided O / LSE tensors.
        if pad_v:
            O_buf = O_buf[..., : self.head_dim_v]
        # SCORE_MAX / SCORE_SUM are [B, H, SQ] (same as LSE) — copy directly.
        if self.return_score_stats:
            if score_max_tensor is not None:
                score_max_tensor.copy_(SMAX_buf)
            if score_sum_tensor is not None:
                score_sum_tensor.copy_(SSUM_buf)
        # O_buf is BSHD → transpose back to BHSD and copy into o_tensor.
        o_tensor.copy_(O_buf.transpose(1, 2))
        lse_tensor.copy_(LSE_buf)
        self._logger.debug("execute completed")


# ---------------------------------------------------------------------------
# Functional wrapper (matches the SM100 D256 surface).
# ---------------------------------------------------------------------------
_cache_of_objects: dict = {}


def _allocate_lse_tensor_sm80(q_tensor: torch.Tensor) -> torch.Tensor:
    if q_tensor.ndim != 4:
        raise ValueError(f"Expected BHSD q_tensor to be rank-4, got {q_tensor.ndim}")
    b, h, s_q, _ = q_tensor.shape
    return torch.empty((b, h, s_q), dtype=torch.float32, device=q_tensor.device)


def _thd_forward(
    q, k, v, *, cu_q, cu_k, max_s_q, scale_softmax, is_causal, window_size, causal_bottom_right, bias_tensor, alibi, sinks, return_score_stats=False
):
    """THD / varlen forward: q/k/v are PACKED ``[1, T, H, D]`` (already BSHD —
    no transpose), cu_q/cu_k are ``[B+1]`` cumulative seqlens.  Routes straight
    to the kernel's THD path (graph-safe over-provisioned grid), reusing the
    flavor-pick + d-pad.  Returns packed ``[1, T_q, H, D_v]`` O + LSE."""
    import inspect as _inspect

    d_qk = q.shape[-1]
    d_v = v.shape[-1]
    h_q = q.shape[2]
    flavor = _pick_flavor(d_qk, d_v)
    fdqk, fdv = _FLAVOR_DIMS[flavor]
    tile_m, num_warps, tile_n = _FLAVOR_KNOBS[flavor]
    # Resolve the default scale from the USER's head dim before padding: the
    # kernel would otherwise derive 1/sqrt(D) from the padded flavor width
    # (e.g. 1/sqrt(128) for a d=96 llama-flavor call) — silently wrong.
    if scale_softmax is None or scale_softmax == 0.0:
        scale_softmax = 1.0 / math.sqrt(d_qk)
    if d_qk < fdqk:
        q = _pad_last_dim(q, fdqk)
        k = _pad_last_dim(k, fdqk)
    pad_v = d_v < fdv
    if pad_v:
        v = _pad_last_dim(v, fdv)
    # mask token from cuDNN's (is_causal, window_size=(left,right)).
    wl, wr = window_size
    if is_causal and wl >= 0:
        mask_token, swa = "causal_swa", wl
    elif is_causal:
        mask_token, swa = "causal", 0
    elif wl >= 0:
        mask_token, swa = "swa", wl
    else:
        mask_token, swa = "none", 0
    right_bound = wr if (is_causal and wr is not None and wr > 0) else 0
    kernel = _load_kernel_module(flavor)
    alibi_slopes = kernel.default_alibi_slopes(h_q).to(q.device) if alibi else None
    fwd_kwargs = dict(
        scale=scale_softmax,
        return_lse=True,
        tile_m=tile_m,
        num_warps=num_warps,
        tile_n=tile_n,
        d_qk=fdqk,
        d_v=fdv,
        mask=mask_token,
        swa_window=int(swa),
        right_bound=int(right_bound),
        causal_bottom_right=bool(causal_bottom_right),
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_s_q=int(max_s_q),
        bias=bias_tensor,
        alibi_slopes=alibi_slopes,
        sinks=sinks,
        return_score_stats=return_score_stats,
    )
    acc = _inspect.signature(kernel.forward).parameters
    fwd_kwargs = {kk: vv for kk, vv in fwd_kwargs.items() if kk in acc}
    _res = kernel.forward(q, k, v, **fwd_kwargs)
    if return_score_stats:
        O_buf, LSE_buf, SMAX_buf, SSUM_buf = _res
    else:
        O_buf, LSE_buf = _res
    if pad_v:
        O_buf = O_buf[..., :d_v].contiguous()
    if return_score_stats:
        return TupleDict(o_tensor=O_buf, lse_tensor=LSE_buf, score_max=SMAX_buf, score_sum_exp=SSUM_buf)
    return TupleDict(o_tensor=O_buf, lse_tensor=LSE_buf)


def sdpa_fwd_wrapper_sm80(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    is_causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    scale_softmax: Optional[float] = None,
    scale_output: float = 1.0,
    scheduler: str = "auto",
    current_stream: Optional[cuda.CUstream] = None,
    causal_bottom_right: bool = False,
    seq_kv_lens: Optional[torch.Tensor] = None,
    seq_len_q: Optional[torch.Tensor] = None,
    bias_tensor: Optional[torch.Tensor] = None,
    alibi: bool = False,
    sinks: Optional[torch.Tensor] = None,
    cum_seqlen_q_tensor: Optional[torch.Tensor] = None,
    cum_seqlen_k_tensor: Optional[torch.Tensor] = None,
    max_s_q: Optional[int] = None,
    return_score_stats: bool = False,
    rope_freqs: Optional[torch.Tensor] = None,
    block_mask: Optional[torch.Tensor] = None,
) -> TupleDict:
    """SM80 (A100) SDPA forward.

    Returns ``TupleDict(o_tensor=..., lse_tensor=...)`` matching the
    SM100 wrapper's contract.
    """

    if q_tensor.ndim != 4 or v_tensor.ndim != 4:
        raise ValueError(f"Q and V must be rank-4 BHSD; got Q={q_tensor.ndim}D V={v_tensor.ndim}D")

    # THD / varlen: q/k/v are PACKED [1, T, H, D] (BSHD) + cu_seqlens.  Handled
    # by a dedicated path that skips the dense BHSD transpose + dense O alloc.
    if cum_seqlen_q_tensor is not None:
        if max_s_q is None:
            raise ValueError("THD path requires max_s_q (host int) for the grid")
        # Reject dense-only features up front: _thd_forward does not plumb
        # them, and silently computing without a requested feature is worse
        # than an error.
        for label, present in (
            ("rope_freqs", rope_freqs is not None),
            ("block_mask", block_mask is not None),
            ("seq_kv_lens", seq_kv_lens is not None),
            ("seq_len_q", seq_len_q is not None),
            ("scale_output != 1.0", scale_output not in (None, 1.0)),
            ('scheduler != "auto"', scheduler not in (None, "auto")),
        ):
            if present:
                raise NotImplementedError(f"SM80 SDPA THD (cum_seqlen_*) path does not support {label}; the dense path serves it")
        with _stream_ctx(current_stream):
            return _thd_forward(
                q_tensor,
                k_tensor,
                v_tensor,
                cu_q=cum_seqlen_q_tensor,
                cu_k=cum_seqlen_k_tensor,
                max_s_q=max_s_q,
                scale_softmax=scale_softmax,
                is_causal=is_causal,
                window_size=window_size,
                causal_bottom_right=causal_bottom_right,
                bias_tensor=bias_tensor,
                alibi=alibi,
                sinks=sinks,
                return_score_stats=return_score_stats,
            )

    b, h_q, s_q, _ = q_tensor.shape
    d_v = v_tensor.shape[-1]
    # O takes Q's leading shape but V's head dim — supports dsv3-style
    # D_QK != D_V.  Match Q's dtype + device and the cuDNN-FE BSHD-physical
    # stride order (3, 1, 2, 0) that ``check_support`` enforces: allocate as
    # contiguous (B, S, H, D) then transpose to a (B, H, S, D) view.
    o_tensor = torch.empty(
        (b, s_q, h_q, d_v),
        dtype=q_tensor.dtype,
        device=q_tensor.device,
    ).transpose(1, 2)
    lse_tensor = _allocate_lse_tensor_sm80(q_tensor)
    # SCORE_MAX / SCORE_SUM share LSE's [B, H, SQ] shape.
    if return_score_stats:
        score_max_t = _allocate_lse_tensor_sm80(q_tensor)
        score_sum_t = _allocate_lse_tensor_sm80(q_tensor)
    else:
        score_max_t = score_sum_t = None

    cache_key = (
        q_tensor.shape,
        k_tensor.shape,
        v_tensor.shape,
        q_tensor.stride(),
        k_tensor.stride(),
        v_tensor.stride(),
        q_tensor.dtype,
        is_causal,
        window_size,
        scale_softmax,
        float(scale_output),
        scheduler,
        causal_bottom_right,
        seq_kv_lens is not None,
        bias_tensor is not None,
        (bias_tensor.dtype if bias_tensor is not None else None),
        bool(alibi),
        sinks is not None,
        cum_seqlen_q_tensor is not None,
        bool(return_score_stats),
        rope_freqs is not None,
        block_mask is not None,
        q_tensor.device,
    )

    sdpa_fwd = _cache_of_objects.get(cache_key)
    if sdpa_fwd is None:
        _logger.debug("sdpa_fwd_wrapper_sm80: building new SdpafwdSm80")
        sdpa_fwd = SdpafwdSm80(
            sample_q=q_tensor,
            sample_k=k_tensor,
            sample_v=v_tensor,
            sample_o=o_tensor,
            sample_lse=lse_tensor,
            is_causal=is_causal,
            window_size=window_size,
            scale_softmax=scale_softmax,
            scale_output=scale_output,
            scheduler=scheduler,
            causal_bottom_right=causal_bottom_right,
            has_seq_kv_lens=seq_kv_lens is not None,
            return_score_stats=return_score_stats,
        )
        assert sdpa_fwd.check_support(), "Unsupported configuration"
        sdpa_fwd.compile()
        _cache_of_objects[cache_key] = sdpa_fwd

    sdpa_fwd.execute(
        q_tensor=q_tensor,
        k_tensor=k_tensor,
        v_tensor=v_tensor,
        o_tensor=o_tensor,
        lse_tensor=lse_tensor,
        score_max_tensor=score_max_t,
        score_sum_tensor=score_sum_t,
        scale_softmax=scale_softmax,
        scale_output=scale_output,
        current_stream=current_stream,
        seq_kv_lens=seq_kv_lens,
        seq_len_q=seq_len_q,
        bias_tensor=bias_tensor,
        alibi=alibi,
        sinks=sinks,
        cum_seqlen_q_tensor=cum_seqlen_q_tensor,
        cum_seqlen_k_tensor=cum_seqlen_k_tensor,
        max_s_q=max_s_q,
        rope_freqs=rope_freqs,
        block_mask=block_mask,
    )

    if return_score_stats:
        return TupleDict(o_tensor=o_tensor, lse_tensor=lse_tensor, score_max=score_max_t, score_sum_exp=score_sum_t)
    return TupleDict(o_tensor=o_tensor, lse_tensor=lse_tensor)
