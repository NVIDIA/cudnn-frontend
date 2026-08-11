# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Tuple
import inspect
import logging
import math

from cuda.bindings import driver as cuda
import torch

import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_stream

from cudnn.api_base import APIBase, TupleDict
from cudnn.datatypes import _convert_to_cutlass_data_type

from .fmha_backward_sm100_2kernel import BlackwellFusedMultiHeadAttentionBackward
from ..fmha_utils import MaskEnum


class SdpabwdSm100D256(APIBase):
    """API class for d=256 SDPA backward (SM100+) using 2-kernel implementation.

    Input/output layout follows the kernel contract:
    - Q/O/dQ/dO: `(B, S_q, H_k * H_r, D)`
    - K/V/dK/dV: `(B, S_k, H_k, D)`
    - LSE: `(B, H_k * H_r, S_q_like)`
    - varlen mode: `cum_seqlen_q` and `cum_seqlen_k` are required and `B` in tensors must be 1.
    """

    def __init__(
        self,
        sample_q: torch.Tensor,
        sample_k: torch.Tensor,
        sample_v: torch.Tensor,
        sample_o: torch.Tensor,
        sample_do: torch.Tensor,
        sample_lse: torch.Tensor,
        sample_dq: torch.Tensor,
        sample_dk: torch.Tensor,
        sample_dv: torch.Tensor,
        sample_cum_seqlen_q: Optional[torch.Tensor] = None,
        sample_cum_seqlen_k: Optional[torch.Tensor] = None,
        max_s_q: Optional[int] = None,
        max_s_k: Optional[int] = None,
        acc_dtype: torch.dtype = torch.float32,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        dkdv_mma_tiler_mn: Tuple[int, int] = (128, 64),
        is_causal: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        scale_softmax: Optional[float] = None,
    ):
        """Initialize SDPA backward API configuration and sample tensor signature.

        Args:
            sample_q: Sample Q tensor with shape `(B, S_q, H_k, H_r, D)`.
            sample_k: Sample K tensor with shape `(B, S_k, H_k, 1, D)`.
            sample_v: Sample V tensor with shape `(B, S_k, H_k, 1, D)`.
            sample_o: Sample forward output O tensor with shape `(B, S_q, H_k, H_r, D)`.
            sample_do: Sample gradient dO tensor with shape `(B, S_q, H_k, H_r, D)`.
            sample_lse: Sample LSE tensor with shape `(B, H_k, H_r, S_q_like)`.
            sample_dq: Sample output-gradient buffer dQ with shape `(B, S_q, H_k, H_r, D)`.
            sample_dk: Sample output-gradient buffer dK with shape `(B, S_k, H_k, 1, D)`.
            sample_dv: Sample output-gradient buffer dV with shape `(B, S_k, H_k, 1, D)`.
            sample_cum_seqlen_q: Optional cumulative query sequence lengths (int32, 1D) for varlen mode.
            sample_cum_seqlen_k: Optional cumulative key/value sequence lengths (int32, 1D) for varlen mode.
            max_s_q: Optional maximum query sequence length. If omitted, inferred from `sample_cum_seqlen_q` in varlen mode.
            max_s_k: Optional maximum key/value sequence length. If omitted, inferred from `sample_cum_seqlen_k` in varlen mode.
            acc_dtype: Accumulator dtype. Must be `torch.float32`.
            mma_tiler_mn: dQ kernel MMA tile `(M, N)`. Current supported value is `(128, 128)`.
            dkdv_mma_tiler_mn: dK/dV kernel MMA tile `(M, N)`. Current implementation requires first dim `128`.
            is_causal: Whether to enable causal masking behavior.
            window_size: Sliding-window tuple `(left, right)`. Use `(-1, -1)` for full window.
            scale_softmax: Optional softmax scaling factor. Defaults to `1/sqrt(D)` when omitted or set to `0.0`.
        """
        super().__init__()
        self._kernel = BlackwellFusedMultiHeadAttentionBackward

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

        # Tensor descriptors used for compile-time signatures.
        self.q_desc = self._make_tensor_desc(sample_q, name="q")
        self.k_desc = self._make_tensor_desc(sample_k, name="k")
        self.v_desc = self._make_tensor_desc(sample_v, name="v")
        self.o_desc = self._make_tensor_desc(sample_o, name="o")
        self.do_desc = self._make_tensor_desc(sample_do, name="do")
        self.lse_desc = self._unpad_tensor_to_ndim(self._make_tensor_desc(sample_lse, name="lse"), self.q_desc.ndim - 1, "lse")
        self.dq_desc = self._make_tensor_desc(sample_dq, name="dq")
        self.dk_desc = self._make_tensor_desc(sample_dk, name="dk")
        self.dv_desc = self._make_tensor_desc(sample_dv, name="dv")
        self.cum_seqlen_q_desc = self._unpad_tensor_to_ndim(self._make_tensor_desc(sample_cum_seqlen_q, name="cum_seqlen_q"), 1, "cum_seqlen_q")
        self.cum_seqlen_k_desc = self._unpad_tensor_to_ndim(self._make_tensor_desc(sample_cum_seqlen_k, name="cum_seqlen_k"), 1, "cum_seqlen_k")

        self.acc_dtype = acc_dtype
        self.mma_tiler_mn = mma_tiler_mn
        self.dkdv_mma_tiler_mn = dkdv_mma_tiler_mn
        self.is_causal = is_causal
        self.window_size_left, self.window_size_right = window_size
        self.scale_softmax = scale_softmax

        self.input_layout = None
        # self.varlen = False
        self.dtype = None
        self.problem_shape = None
        self.mask_type = None

        self.h_k = None
        self.h_r = None
        self.h_q = None
        self.head_dim = None
        self.batch_size = None
        self.s_q_total = None
        self.s_k_total = None
        self.workspace_shape = None
        self.workspace_torch = None

        self._logger.debug("__init__ completed")

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        # shape normalization and validation
        self._logger.debug("Checking shape normalization and validation")
        if self.cum_seqlen_q_desc is None and self.cum_seqlen_k_desc is None:
            self._logger.info("cum_seqlen_q and cum_seqlen_k not provided, inferring B,H,S,D layout")
            self.input_layout = "B,H,S,D"
            # (b, h, s, d) -> (b, s, h, d)
            for desc_name in ["q_desc", "k_desc", "v_desc", "o_desc", "do_desc", "dq_desc", "dk_desc", "dv_desc"]:
                tensor_desc = getattr(self, desc_name)
                self._value_error_if(tensor_desc.ndim != 4, f"{tensor_desc.name} must be rank-4 for B,H,S,D layout, got {tensor_desc.ndim}")
                self._value_error_if(
                    tensor_desc.stride_order != (3, 1, 2, 0), f"{tensor_desc.name} must have d,h,s,b stride order (3, 1, 2, 0), got {tensor_desc.stride_order}"
                )
                setattr(self, desc_name, tensor_desc.transpose(1, 2))
            self._value_error_if(not self.lse_desc.is_contiguous(), "lse_tensor must be contiguous for B,H,S,D layout")
        elif self.cum_seqlen_q_desc is not None and self.cum_seqlen_k_desc is not None:
            self._logger.info("cum_seqlen_q and cum_seqlen_k provided, inferring T,H,D layout")
            self.input_layout = "T,H,D"
            # (t, h, d) -> (1, t, h, d)
            for desc_name in ["q_desc", "k_desc", "v_desc", "o_desc", "do_desc", "dq_desc", "dk_desc", "dv_desc"]:
                tensor_desc = getattr(self, desc_name)
                if tensor_desc.ndim == 3:
                    setattr(self, desc_name, tensor_desc.unsqueeze(0))
                elif tensor_desc.ndim == 4:
                    self._value_error_if(
                        tensor_desc.shape[0] != 1, f"{tensor_desc.name} must have batch dimension 1 for T,H,D layout (1, t, h, d), got {tensor_desc.shape[0]}"
                    )
                else:
                    raise ValueError(f"{tensor_desc.name} must be rank-3 or rank-4 for T,H,D layout, got {tensor_desc.ndim}")
            self.lse_desc = self.lse_desc.unsqueeze(0).transpose(1, 2)
        else:
            raise ValueError(f"cum_seqlen_q and cum_seqlen_k must be both None or both not None, got {self.cum_seqlen_q_desc} and {self.cum_seqlen_k_desc}")

        b, s_qo, h_qo, d_qk = self.q_desc.shape
        _, s_kv, h_kv, d_v = self.v_desc.shape
        self._check_tensor_shape(self.q_desc, (b, s_qo, h_qo, d_qk), name="Q")
        self._check_tensor_shape(self.k_desc, (b, s_kv, h_kv, d_qk), name="K")
        self._check_tensor_shape(self.v_desc, (b, s_kv, h_kv, d_v), name="V")
        self._check_tensor_shape(self.o_desc, (b, s_qo, h_qo, d_v), name="O")
        self._check_tensor_shape(self.do_desc, (b, s_qo, h_qo, d_v), name="dO")
        self._check_tensor_shape(self.dq_desc, (b, s_qo, h_qo, d_v), name="dQ")
        self._check_tensor_shape(self.dk_desc, (b, s_kv, h_kv, d_v), name="dK")
        self._check_tensor_shape(self.dv_desc, (b, s_kv, h_kv, d_v), name="dV")
        self.lse_desc = self._unpad_tensor_to_ndim(self.lse_desc, 3, name="LSE")
        self._check_tensor_shape(self.lse_desc, (b, h_qo, s_qo), name="LSE")

        self._value_error_if(d_qk != d_v, f"D_qk must match D_v, got {d_qk} and {d_v}")
        self._value_error_if(h_qo % h_kv != 0, f"H_q must be divisible by H_k, got {h_qo} and {h_kv}")

        self.dtype = self._check_dtype(self.q_desc, [torch.float16, torch.bfloat16], name="Q")
        for tensor_desc in [self.k_desc, self.v_desc, self.o_desc, self.do_desc, self.dq_desc, self.dk_desc, self.dv_desc]:
            self._check_dtype(tensor_desc, self.dtype, name=tensor_desc.name, extra_error_msg=f"{tensor_desc.name} must match Q dtype")
        self._check_dtype(self.lse_desc, torch.float32, name="LSE")
        self._check_dtype(self.acc_dtype, torch.float32, name="acc_dtype", extra_error_msg="Only float32 accumulator is supported")

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
        self.h_r = h_qo // h_kv
        self.h_q = h_qo
        self.head_dim = d_qk

        self._value_error_if(self.head_dim != 256, f"head_dim must be 256, got {self.head_dim}")
        self._value_error_if(self.mma_tiler_mn != (128, 128), f"mma_tiler_mn must be (128, 128), got {self.mma_tiler_mn}")
        self._value_error_if(self.dkdv_mma_tiler_mn != (128, 64), f"dkdv_mma_tiler_mn must be (128, 64), got {self.dkdv_mma_tiler_mn}")

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
        if (not self.is_causal) and (self.input_layout == "T,H,D" or s_qo % self.mma_tiler_mn[0] != 0):
            self.mask_type = MaskEnum.RESIDUAL_MASK

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = self.q_desc.device
        major, minor = torch.cuda.get_device_capability(device)
        compute_capability = major * 10 + minor
        if compute_capability < 100:
            raise RuntimeError(f"SdpabwdSm100D256 requires SM100+, found SM{compute_capability} on device {device}")

        self.problem_shape = (
            self.s_q_max,
            self.s_k_max,
            self.head_dim,
            ((self.h_r, self.h_k), self.batch_size),
        )
        # Workspace follows kernel's expected shape contract.
        self.workspace_shape = BlackwellFusedMultiHeadAttentionBackward.get_workspace_size(
            s_qo,
            self.head_dim,
            self.h_q,
            b,
            _convert_to_cutlass_data_type(self.acc_dtype),
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

        sdpa_bwd = self._kernel(
            element_dtype=_convert_to_cutlass_data_type(self.dtype),
            acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
            mma_tiler=(*self.mma_tiler_mn, self.head_dim),
            dkdv_mma_tiler=self.dkdv_mma_tiler_mn,
            varlen=(self.input_layout == "T,H,D"),
            is_causal=self.is_causal,
            mask_type=self.mask_type,
            window_size_left=self.window_size_left,
            window_size_right=self.window_size_right,
        )
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)

        self.workspace_torch = torch.empty(self.workspace_shape, dtype=torch.uint8, device=self.q_desc.device)

        self._logger.debug("Compiling sdpa backward kernel with cute.compile")
        _compiled_kernel = cute.compile(
            sdpa_bwd,
            problem_shape=self.problem_shape,
            Q=self._make_fake_cute_tensor_from_desc(self.q_desc, assumed_align=64),
            K=self._make_fake_cute_tensor_from_desc(self.k_desc, assumed_align=64),
            V=self._make_fake_cute_tensor_from_desc(self.v_desc, assumed_align=64),
            O=self._make_fake_cute_tensor_from_desc(self.o_desc, assumed_align=64),
            dQ=self._make_fake_cute_tensor_from_desc(self.dq_desc, assumed_align=64),
            dK=self._make_fake_cute_tensor_from_desc(self.dk_desc, assumed_align=64),
            dV=self._make_fake_cute_tensor_from_desc(self.dv_desc, assumed_align=64),
            dO=self._make_fake_cute_tensor_from_desc(self.do_desc, assumed_align=64),
            LSE=self._make_fake_cute_tensor_from_desc(self.lse_desc, assumed_align=64),
            cumulative_s_q=self._make_fake_cute_tensor_from_desc(self.cum_seqlen_q_desc, assumed_align=16),
            cumulative_s_k=self._make_fake_cute_tensor_from_desc(self.cum_seqlen_k_desc, assumed_align=16),
            scale_softmax=self.scale_softmax,
            workspace=self._make_fake_cute_tensor_like(self.workspace_torch, assumed_align=16, name="workspace"),
            stream=fake_stream,
            options="--enable-tvm-ffi",
        )

        def tensor_api(
            q_tensor: torch.Tensor,
            k_tensor: torch.Tensor,
            v_tensor: torch.Tensor,
            o_tensor: torch.Tensor,
            dq_tensor: torch.Tensor,
            dk_tensor: torch.Tensor,
            dv_tensor: torch.Tensor,
            do_tensor: torch.Tensor,
            lse_tensor: torch.Tensor,
            cumulative_s_q: Optional[torch.Tensor],
            cumulative_s_k: Optional[torch.Tensor],
            scale_softmax: float,
            workspace: torch.Tensor,
            stream: cuda.CUstream,
        ) -> None:
            if self.input_layout == "B,H,S,D":
                q_tensor, k_tensor, v_tensor, o_tensor, dq_tensor, dk_tensor, dv_tensor, do_tensor = (
                    q_tensor.transpose(1, 2),
                    k_tensor.transpose(1, 2),
                    v_tensor.transpose(1, 2),
                    o_tensor.transpose(1, 2),
                    dq_tensor.transpose(1, 2),
                    dk_tensor.transpose(1, 2),
                    dv_tensor.transpose(1, 2),
                    do_tensor.transpose(1, 2),
                )
            elif self.input_layout == "T,H,D":
                q_tensor, k_tensor, v_tensor, o_tensor, dq_tensor, dk_tensor, dv_tensor, do_tensor = (
                    q_tensor.unsqueeze(0) if q_tensor.ndim == 3 else q_tensor,
                    k_tensor.unsqueeze(0) if k_tensor.ndim == 3 else k_tensor,
                    v_tensor.unsqueeze(0) if v_tensor.ndim == 3 else v_tensor,
                    o_tensor.unsqueeze(0) if o_tensor.ndim == 3 else o_tensor,
                    dq_tensor.unsqueeze(0) if dq_tensor.ndim == 3 else dq_tensor,
                    dk_tensor.unsqueeze(0) if dk_tensor.ndim == 3 else dk_tensor,
                    dv_tensor.unsqueeze(0) if dv_tensor.ndim == 3 else dv_tensor,
                    do_tensor.unsqueeze(0) if do_tensor.ndim == 3 else do_tensor,
                )
                lse_tensor = lse_tensor.unsqueeze(0).transpose(1, 2)
                cumulative_s_q = self._unpad_tensor_to_ndim(cumulative_s_q, 1, "cum_seqlen_q")
                cumulative_s_k = self._unpad_tensor_to_ndim(cumulative_s_k, 1, "cum_seqlen_k")
            lse_tensor = self._unpad_tensor_to_ndim(lse_tensor, 3, "lse_tensor")
            _compiled_kernel(
                self.problem_shape,
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                dq_tensor,
                dk_tensor,
                dv_tensor,
                do_tensor,
                lse_tensor,
                cumulative_s_q,
                cumulative_s_k,
                scale_softmax,
                workspace,
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
        do_tensor: torch.Tensor,
        lse_tensor: torch.Tensor,
        dq_tensor: torch.Tensor,
        dk_tensor: torch.Tensor,
        dv_tensor: torch.Tensor,
        cum_seqlen_q_tensor: Optional[torch.Tensor] = None,
        cum_seqlen_k_tensor: Optional[torch.Tensor] = None,
        scale_softmax: Optional[float] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        self._logger.debug("Entering execute")
        current_stream = self._get_default_stream(current_stream)

        if self._compiled_kernel is None:
            raise RuntimeError("SdpabwdSm100D256 kernel is not compiled")

        if self.input_layout == "T,H,D":
            self._value_error_if(cum_seqlen_q_tensor is None, "cum_seqlen_q_tensor is required for T,H,D layout")
            self._value_error_if(cum_seqlen_k_tensor is None, "cum_seqlen_k_tensor is required for T,H,D layout")
        elif self.input_layout == "B,H,S,D":
            self._value_error_if(cum_seqlen_q_tensor is not None, "cum_seqlen_q_tensor must be None for B,H,S,D layout")
            self._value_error_if(cum_seqlen_k_tensor is not None, "cum_seqlen_k_tensor must be None for B,H,S,D layout")
        else:
            raise NotImplementedError(f"Invalid input layout: {self.input_layout}")

        scale_softmax_val = self.scale_softmax if (scale_softmax is None or scale_softmax == 0.0) else scale_softmax

        with torch.cuda.stream(torch.cuda.ExternalStream(int(current_stream))):
            self.workspace_torch.zero_()

        self._compiled_kernel(
            q_tensor=q_tensor,
            k_tensor=k_tensor,
            v_tensor=v_tensor,
            o_tensor=o_tensor,
            dq_tensor=dq_tensor,
            dk_tensor=dk_tensor,
            dv_tensor=dv_tensor,
            do_tensor=do_tensor,
            lse_tensor=lse_tensor,
            cumulative_s_q=cum_seqlen_q_tensor,
            cumulative_s_k=cum_seqlen_k_tensor,
            scale_softmax=scale_softmax_val,
            workspace=self.workspace_torch,
            stream=current_stream,
        )
        self._logger.debug("Execute completed")


_logger = logging.getLogger(__name__)
_cache_of_SdpabwdSm100D256Objects = {}


def sdpa_bwd_wrapper_sm100_d256(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    o_tensor: torch.Tensor,
    do_tensor: torch.Tensor,
    lse_tensor: torch.Tensor,
    cum_seqlen_q_tensor: Optional[torch.Tensor] = None,
    cum_seqlen_k_tensor: Optional[torch.Tensor] = None,
    max_s_q: Optional[int] = None,
    max_s_k: Optional[int] = None,
    acc_dtype: torch.dtype = torch.float32,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    dkdv_mma_tiler_mn: Tuple[int, int] = (128, 64),
    is_causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    scale_softmax: Optional[float] = None,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Convenience wrapper for d=256 SDPA backward SM100 kernel.

    Returns:
        TupleDict: `(dq_tensor, dk_tensor, dv_tensor)`
    """

    dq_tensor = torch.empty_like(q_tensor)
    dk_tensor = torch.empty_like(k_tensor)
    dv_tensor = torch.empty_like(v_tensor)

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
        o_tensor.shape,
        do_tensor.shape,
        lse_tensor.shape,
        q_tensor.stride(),
        k_tensor.stride(),
        v_tensor.stride(),
        o_tensor.stride(),
        do_tensor.stride(),
        lse_tensor.stride(),
        q_tensor.dtype,
        k_tensor.dtype,
        v_tensor.dtype,
        o_tensor.dtype,
        do_tensor.dtype,
        lse_tensor.dtype,
        cum_seqlen_q_tensor.shape if cum_seqlen_q_tensor is not None else None,
        cum_seqlen_q_tensor.stride() if cum_seqlen_q_tensor is not None else None,
        cum_seqlen_q_tensor.dtype if cum_seqlen_q_tensor is not None else None,
        cum_seqlen_k_tensor.shape if cum_seqlen_k_tensor is not None else None,
        cum_seqlen_k_tensor.stride() if cum_seqlen_k_tensor is not None else None,
        cum_seqlen_k_tensor.dtype if cum_seqlen_k_tensor is not None else None,
        cache_max_s_q,
        cache_max_s_k,
        acc_dtype,
        mma_tiler_mn,
        dkdv_mma_tiler_mn,
        is_causal,
        window_size,
        scale_softmax,
        q_tensor.device,
    )

    if cache_key in _cache_of_SdpabwdSm100D256Objects:
        _logger.debug("sdpa_bwd_wrapper_sm100_d256: Using cached SdpabwdSm100D256 object")
        sdpa_bwd = _cache_of_SdpabwdSm100D256Objects[cache_key]
        sdpa_bwd.execute(
            q_tensor=q_tensor,
            k_tensor=k_tensor,
            v_tensor=v_tensor,
            o_tensor=o_tensor,
            do_tensor=do_tensor,
            lse_tensor=lse_tensor,
            dq_tensor=dq_tensor,
            dk_tensor=dk_tensor,
            dv_tensor=dv_tensor,
            cum_seqlen_q_tensor=cum_seqlen_q_tensor,
            cum_seqlen_k_tensor=cum_seqlen_k_tensor,
            scale_softmax=scale_softmax,
            current_stream=current_stream,
        )
    else:
        _logger.debug("sdpa_bwd_wrapper_sm100_d256: No cached object found, creating new SdpabwdSm100D256 object")
        sdpa_bwd = SdpabwdSm100D256(
            sample_q=q_tensor,
            sample_k=k_tensor,
            sample_v=v_tensor,
            sample_o=o_tensor,
            sample_do=do_tensor,
            sample_lse=lse_tensor,
            sample_dq=dq_tensor,
            sample_dk=dk_tensor,
            sample_dv=dv_tensor,
            sample_cum_seqlen_q=cum_seqlen_q_tensor,
            sample_cum_seqlen_k=cum_seqlen_k_tensor,
            max_s_q=max_s_q,
            max_s_k=max_s_k,
            acc_dtype=acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            dkdv_mma_tiler_mn=dkdv_mma_tiler_mn,
            is_causal=is_causal,
            window_size=window_size,
            scale_softmax=scale_softmax,
        )
        assert sdpa_bwd.check_support(), "Unsupported configuration"
        sdpa_bwd.compile()
        sdpa_bwd.execute(
            q_tensor=q_tensor,
            k_tensor=k_tensor,
            v_tensor=v_tensor,
            o_tensor=o_tensor,
            do_tensor=do_tensor,
            lse_tensor=lse_tensor,
            dq_tensor=dq_tensor,
            dk_tensor=dk_tensor,
            dv_tensor=dv_tensor,
            cum_seqlen_q_tensor=cum_seqlen_q_tensor,
            cum_seqlen_k_tensor=cum_seqlen_k_tensor,
            scale_softmax=scale_softmax,
            current_stream=current_stream,
        )
        _cache_of_SdpabwdSm100D256Objects[cache_key] = sdpa_bwd

    return TupleDict(
        dq_tensor=dq_tensor,
        dk_tensor=dk_tensor,
        dv_tensor=dv_tensor,
    )


# ===========================================================================
# SM80 (A100)
# ===========================================================================
"""cuDNN-frontend wrapper around the SM80 (A100) SDPA BACKWARD kernel.

Companion to the SM80 forward adapter above in ``fwd/api.py``.  The kernels live at
``kernels/bprop_f16_sm80.py`` (generic, fully parameterized on d_qk/d_v)
and ``kernels/bprop_d64_f16_sm80.py`` (d=64 perf variant the adapter
routes to when the call qualifies) —
vendored from the upstream tile repo (provenance: ``kernels/__init__.py``).

This adapter:
  1. Maps cuDNN-FE BHSD <-> kernel BSHD (zero-cost transpose).
  2. Picks the kernel flavor envelope (gptoss(64,64) / llama(128,128) /
     dsv3(192,128) / qwen(256,256)); requires ``d_qk >= d_v``.
  3. Implements the APIBase contract so backward plugs into ``cudnn.sdpa``.
     The ``cudnn.frost.sdpa`` backward engine lowers onto this adapter.
  4. Passes the full feature kwarg superset and drops any the kernel's
     ``backward()`` doesn't yet accept (``inspect.signature`` filter) — so
     features (mask / bias / dBias / alibi / sink / rope / seqlens / THD)
     auto-wire the moment they land in the kernel.

Coverage tracks the kernel: FP16/BF16, dense + causal/SWA/BR/padded masks,
bias (+dBias), alibi, sink (+dSink), GQA/MQA, RoPE, block_mask, THD, and a
``deterministic`` dQ mode (ordered KV-tile reduction) — gated per-feature.
"""  # noqa: E501 — SM80 section notes

# ---------------------------------------------------------------------------
# Lazy import of the BPROP kernel.
# ---------------------------------------------------------------------------
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


# The generic kernel now supports d_qk != d_v (split sub-groups) and d up to
# 256, so the envelope spans gptoss(64,64) / llama(128,128) / dsv3(192,128) /
# qwen(256,256).  Kernel constraint: d_qk >= d_v (the per-sub-group split).
# The flavor only sets the d-pad target; the kernel derives qo_stages / drop-sDQ
# from the (padded) d_qk for the A100 SMEM budget.
from ..fwd import config_sm80 as _fwd_config_sm80

_FLAVOR_DIMS = {
    name: (cfg.D_QK, cfg.D_V)
    for name, cfg in (
        ("gptoss", _fwd_config_sm80.GPTOSS_CFG),
        ("llama", _fwd_config_sm80.LLAMA_CFG),
        ("dsv3", _fwd_config_sm80.DSV3_CFG),
        ("qwen", _fwd_config_sm80.QWEN_CFG),
    )
}
_SUPPORTED_FLAVORS = ("gptoss", "llama", "dsv3", "qwen")


def _load_kernel_module(key: str = "f16"):
    """Lazily import + cache an SM80 BPROP kernel module.

    ``"f16"`` is the GENERIC kernel (``bprop_f16_sm80``): fully parameterized
    on d_qk/d_v with the full feature set (masks / bias / dBias / alibi /
    sink / rope / block_mask / THD / deterministic).  ``"d64"`` is the
    dedicated plain-dense d=64 MHA perf variant (~2x faster on A100); it
    supports NO features — its ``backward(**_ignored)`` silently swallows
    every feature kwarg, so callers must never rely on the signature filter
    and only select it through :func:`_d64_fast_path_eligible`.
    """
    if key not in _KERNEL_MOD:
        if key == "d64":
            from .kernels import bprop_d64_f16_sm80 as _mod
        else:
            from .kernels import bprop_f16_sm80 as _mod

        _KERNEL_MOD[key] = _mod
    return _KERNEL_MOD[key]


def _d64_fast_path_eligible(*, d_qk, d_v, h_q, h_kv, s_q, s_kv, mask_token, right_bound, causal_bottom_right, bw_kwargs) -> bool:
    """Whether the dedicated d=64 kernel can serve this call EXACTLY.

    The perf variant computes a plain dense MHA backward and nothing else;
    every condition here guards a feature it would silently ignore.
    """
    d64 = _load_kernel_module("d64")
    if (d_qk, d_v) != (64, 64) or h_q != h_kv:
        return False
    if s_q % d64.M_BLOCK != 0 or s_kv % d64.N_BLOCK != 0:
        return False
    if mask_token != "none" or right_bound != 0 or causal_bottom_right:
        return False
    for feature in ("seq_kv_lens", "seq_len_q", "bias", "alibi_slopes", "sinks", "rope_freqs", "block_mask"):
        if bw_kwargs.get(feature) is not None:
            return False
    if bw_kwargs.get("deterministic"):
        return False
    return True


def _pick_flavor(d_qk: int, d_v: int) -> str:
    """Smallest BPROP flavor whose ``(D_QK, D_V)`` envelope covers
    ``(d_qk, d_v)`` (fdqk >= d_qk and fdv >= d_v); the user's heads are padded
    up to the flavor dim.  The kernel supports d_qk != d_v but requires the
    (padded) d_qk >= d_v — the flavor list guarantees this (every flavor has
    fdqk >= fdv, and a d_qk < d_v case lands on an equal-d flavor after pad)."""
    for flavor in _SUPPORTED_FLAVORS:
        fdqk, fdv = _FLAVOR_DIMS[flavor]
        if d_qk == fdqk and d_v == fdv:
            return flavor
    for flavor in _SUPPORTED_FLAVORS:
        fdqk, fdv = _FLAVOR_DIMS[flavor]
        if d_qk <= fdqk and d_v <= fdv:
            return flavor
    raise ValueError(f"SM80 BPROP: no flavor envelope covers (D_QK={d_qk}, D_V={d_v}); " f"supported: {_FLAVOR_DIMS}.")


def _pad_last_dim(t: torch.Tensor, new_last: int) -> torch.Tensor:
    """Zero-pad the trailing dim of an fp16/bf16 tensor up to ``new_last``."""
    old_last = t.shape[-1]
    if old_last == new_last:
        return t
    if old_last > new_last:
        raise ValueError(f"_pad_last_dim: tensor's last dim {old_last} exceeds target {new_last}")
    pad = torch.zeros((*t.shape[:-1], new_last - old_last), dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=-1).contiguous()


def _bshd(t: torch.Tensor) -> torch.Tensor:
    """BHSD → BSHD (stride-only transpose; contiguous-ify only if needed)."""
    x = t.transpose(1, 2)
    return x if x.is_contiguous() else x.contiguous()


# ---------------------------------------------------------------------------
# APIBase subclass.
# ---------------------------------------------------------------------------
class SdpabwdSm80(APIBase):
    """SM80 (A100) SDPA backward.

    Mirrors ``SdpafwdSm80``.  Inputs are the forward activations (Q/K/V/O), the
    loss gradient dO, and the forward stats LSE.  Outputs dQ/dK/dV (+ dBias when
    an additive bias is present).
    """

    def __init__(
        self,
        sample_q: torch.Tensor,
        sample_k: torch.Tensor,
        sample_v: torch.Tensor,
        sample_o: torch.Tensor,
        sample_do: torch.Tensor,
        sample_lse: torch.Tensor,
        is_causal: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        scale_softmax: Optional[float] = None,
        causal_bottom_right: bool = False,
        has_seq_kv_lens: bool = False,
        has_bias: bool = False,
    ):
        super().__init__()
        self._warn_experimental_api()
        self._logger.debug("Entering __init__ (bwd)")

        self.q_desc = self._make_tensor_desc(sample_q, name="q")
        self.k_desc = self._make_tensor_desc(sample_k, name="k")
        self.v_desc = self._make_tensor_desc(sample_v, name="v")
        self.o_desc = self._make_tensor_desc(sample_o, name="o")
        self.do_desc = self._make_tensor_desc(sample_do, name="dO")
        self.lse_desc = self._make_tensor_desc(sample_lse, name="lse")

        self.is_causal = is_causal
        self.window_size_left, self.window_size_right = window_size
        self.scale_softmax = scale_softmax
        self.causal_bottom_right = bool(causal_bottom_right)
        self.has_seq_kv_lens = bool(has_seq_kv_lens)
        self.has_bias = bool(has_bias)

        # Filled by check_support().
        self.flavor: Optional[str] = None
        self.flavor_d_qk: Optional[int] = None
        self.flavor_d_v: Optional[int] = None
        self.mask_token: Optional[str] = None
        self.swa_window_runtime: int = 0
        self.right_bound: int = 0
        self.head_dim_qk: Optional[int] = None
        self.head_dim_v: Optional[int] = None
        self._logger.debug("__init__ (bwd) completed")

    # ------------------------------------------------------------------
    def check_support(self) -> bool:
        self._logger.debug("Entering check_support (bwd)")

        _REQ = (3, 1, 2, 0)
        for desc_name in ["q_desc", "k_desc", "v_desc", "o_desc", "do_desc"]:
            d = getattr(self, desc_name)
            self._value_error_if(d.ndim != 4, f"{d.name} must be rank-4 (B, H, S, D); got {d.ndim}")
            _shape = d.shape
            _act = tuple(ax for ax in d.stride_order if _shape[ax] != 1)
            _exp = tuple(ax for ax in _REQ if _shape[ax] != 1)
            self._value_error_if(
                _act != _exp, f"{d.name} must have d,h,s,b stride order (3,1,2,0) " f"(size-1 dims wildcarded); got {d.stride_order} shape {_shape}"
            )

        b, h_qo, s_qo, d_qk = self.q_desc.shape
        _, h_kv, s_kv, _ = self.k_desc.shape
        _, _, _, d_v = self.v_desc.shape

        self._check_tensor_shape(self.q_desc, (b, h_qo, s_qo, d_qk), name="Q")
        self._check_tensor_shape(self.k_desc, (b, h_kv, s_kv, d_qk), name="K")
        self._check_tensor_shape(self.v_desc, (b, h_kv, s_kv, d_v), name="V")
        self._check_tensor_shape(self.o_desc, (b, h_qo, s_qo, d_v), name="O")
        self._check_tensor_shape(self.do_desc, (b, h_qo, s_qo, d_v), name="dO")

        for label, val in (("B", b), ("H_q", h_qo), ("H_kv", h_kv), ("S_q", s_qo), ("S_kv", s_kv), ("D_QK", d_qk), ("D_V", d_v)):
            self._value_error_if(int(val) <= 0, f"{label} must be > 0; got {val}")

        self._value_error_if(h_qo % h_kv != 0, f"H_q ({h_qo}) must be divisible by H_kv ({h_kv}) for GQA / MQA")

        # Kernel supports d_qk != d_v (split sub-groups) but requires d_qk >= d_v
        # (a d_qk < d_v case is padded up to an equal-d flavor by _pick_flavor).
        self._value_error_if(d_qk < d_v, f"SM80 BPROP requires D_QK >= D_V; got D_QK={d_qk}, D_V={d_v}")
        max_dqk = max(fdqk for fdqk, _ in _FLAVOR_DIMS.values())
        max_dv = max(fdv for _, fdv in _FLAVOR_DIMS.values())
        self._value_error_if(
            d_qk > max_dqk or d_v > max_dv,
            f"SM80 BPROP: head dim (D_QK={d_qk}, D_V={d_v}) exceeds supported " f"envelope (D_QK<={max_dqk}, D_V<={max_dv}); larger heads not yet ported.",
        )

        self.dtype = self._check_dtype(self.q_desc, [torch.float16, torch.bfloat16], name="Q")
        for desc in [self.k_desc, self.v_desc, self.o_desc, self.do_desc]:
            self._check_dtype(desc, self.dtype, name=desc.name, extra_error_msg=f"{desc.name} must match Q dtype (FP16/BF16)")
        self._check_dtype(self.lse_desc, torch.float32, name="LSE")
        self._check_tensor_shape(self.lse_desc, (b, h_qo, s_qo), name="LSE")
        self._value_error_if(not self.lse_desc.is_contiguous(), "LSE must be contiguous on SM80")

        self._value_error_if(not torch.cuda.is_available(), "CUDA must be available for SM80 BPROP")
        device = self.q_desc.device
        major, minor = torch.cuda.get_device_capability(device)
        self._value_error_if((major, minor) != (8, 0), f"SdpabwdSm80 requires SM80 (A100); found SM{major}{minor} on {device}")

        self.flavor = _pick_flavor(d_qk, d_v)
        self.flavor_d_qk, self.flavor_d_v = _FLAVOR_DIMS[self.flavor]
        self.head_dim_qk = int(d_qk)
        self.head_dim_v = int(d_v)

        # ---- mask token (same resolution as the forward adapter) ------
        swa_left = self.window_size_left
        swa_right = self.window_size_right
        self.right_bound = 0
        if self.is_causal:
            self.mask_token = "causal" if swa_left < 0 else "causal_swa"
            self.swa_window_runtime = max(0, swa_left) if swa_left >= 0 else 0
            self.right_bound = max(0, swa_right)
        elif swa_left >= 0:
            # A left window alone selects SWA; window_size_right is only
            # meaningful with is_causal=True.
            self._not_implemented_error_if(swa_right > 0, "SM80 BPROP: non-causal SWA with window_size_right > 0 unsupported")
            self.mask_token = "swa"
            self.swa_window_runtime = swa_left
        else:
            # window_size=(-1, r) without is_causal: a bare right bound has no
            # diagonal to anchor to — reject rather than silently pick a mask
            # (mirrors the forward adapter and the THD path).
            self._not_implemented_error_if(
                swa_right >= 0,
                "SM80 BPROP: window_size_right without a left window or is_causal=True has no effect; pass is_causal=True or a left window",
            )
            self.mask_token = "none"
            self.swa_window_runtime = 0

        self._value_error_if(
            self.causal_bottom_right and not (self.is_causal or self.window_size_left >= 0),
            "SM80 BPROP: causal_bottom_right requires is_causal and/or a left window",
        )

        if self.scale_softmax is None or self.scale_softmax == 0.0:
            self.scale_softmax = 1.0 / math.sqrt(d_qk)

        self._is_supported = True
        self._logger.debug("check_support (bwd) completed")
        return True

    # ------------------------------------------------------------------
    def compile(self) -> None:
        """No-op — the kernel module owns its own per-shape ``lru_cache``;
        first ``execute()`` JITs and reuses thereafter."""
        self._logger.debug("Entering compile (bwd, no-op — kernel self-caches)")
        self._ensure_support_checked()
        self._compiled_kernel = True
        self._logger.debug("compile (bwd) completed")

    # ------------------------------------------------------------------
    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        do_tensor: torch.Tensor,
        lse_tensor: torch.Tensor,
        dq_tensor: torch.Tensor,
        dk_tensor: torch.Tensor,
        dv_tensor: torch.Tensor,
        dbias_tensor: Optional[torch.Tensor] = None,
        dsink_tensor: Optional[torch.Tensor] = None,
        scale_softmax: Optional[float] = None,
        current_stream: Optional[cuda.CUstream] = None,
        seq_kv_lens: Optional[torch.Tensor] = None,
        seq_len_q: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        alibi: bool = False,
        sinks: Optional[torch.Tensor] = None,
        rope_freqs: Optional[torch.Tensor] = None,
        block_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> None:
        self._logger.debug("Entering execute (bwd)")
        if self._compiled_kernel is None:
            raise RuntimeError("SdpabwdSm80 is not compiled")
        scale_val = self.scale_softmax if (scale_softmax is None or scale_softmax == 0.0) else float(scale_softmax)

        kernel = _load_kernel_module()

        alibi_slopes = None
        if alibi:
            h_q = q_tensor.shape[1]  # BHSD
            alibi_slopes = kernel.default_alibi_slopes(h_q).to(q_tensor.device)

        # BHSD → BSHD for the kernel.
        Q, K, V = _bshd(q_tensor), _bshd(k_tensor), _bshd(v_tensor)
        O, dO = _bshd(o_tensor), _bshd(do_tensor)

        pad_v = self.head_dim_v < self.flavor_d_v
        pad_qk = self.head_dim_qk < self.flavor_d_qk
        if pad_qk:
            Q = _pad_last_dim(Q, self.flavor_d_qk)
            K = _pad_last_dim(K, self.flavor_d_qk)
        if pad_v:
            V = _pad_last_dim(V, self.flavor_d_v)
            O = _pad_last_dim(O, self.flavor_d_v)
            dO = _pad_last_dim(dO, self.flavor_d_v)

        # Build the feature-kwarg superset; drop any the kernel doesn't accept.
        bw_kwargs = dict(
            scale=scale_val,
            mask=self.mask_token,
            swa_window=int(self.swa_window_runtime),
            right_bound=int(self.right_bound),
            causal_bottom_right=self.causal_bottom_right,
            seq_kv_lens=seq_kv_lens,
            seq_len_q=seq_len_q,
            bias=bias_tensor,
            alibi_slopes=alibi_slopes,
            sinks=sinks,
            rope_freqs=rope_freqs,
            block_mask=block_mask,
            deterministic=bool(deterministic),
        )
        # Route plain dense MHA d=64 calls to the dedicated perf kernel
        # (~2x faster on A100).  The gate must stay exhaustive: the d64
        # kernel's ``backward(**_ignored)`` silently swallows any feature
        # kwarg it does not implement, so an under-gated call would produce
        # wrong gradients rather than an error.
        if _d64_fast_path_eligible(
            d_qk=self.head_dim_qk,
            d_v=self.head_dim_v,
            h_q=q_tensor.shape[1],
            h_kv=k_tensor.shape[1],
            s_q=q_tensor.shape[2],
            s_kv=k_tensor.shape[2],
            mask_token=self.mask_token,
            right_bound=int(self.right_bound),
            causal_bottom_right=self.causal_bottom_right,
            bw_kwargs=bw_kwargs,
        ):
            kernel = _load_kernel_module("d64")
            self._logger.debug("execute (bwd): routing to the dedicated d64 kernel")
        accepted = inspect.signature(kernel.backward).parameters
        bw_kwargs = {kk: vv for kk, vv in bw_kwargs.items() if kk in accepted}

        with _stream_ctx(current_stream):
            res = kernel.backward(Q, K, V, dO, O, lse_tensor, **bw_kwargs)
        dQ_k, dK_k, dV_k = res[0], res[1], res[2]
        # backward() appends optional grads in a FIXED order: dBias (if bias),
        # then dSink (if sinks).  Reconstruct positions from what we passed.
        _idx = 3
        dBias_k = None
        dSink_k = None
        if bias_tensor is not None:
            dBias_k = res[_idx]
            _idx += 1
        if sinks is not None:
            dSink_k = res[_idx]
            _idx += 1

        # Slice off any d-padding, transpose BSHD → BHSD, copy into user tensors.
        if pad_qk:
            dQ_k = dQ_k[..., : self.head_dim_qk]
            dK_k = dK_k[..., : self.head_dim_qk]
        if pad_v:
            dV_k = dV_k[..., : self.head_dim_v]
        dq_tensor.copy_(dQ_k.transpose(1, 2))
        dk_tensor.copy_(dK_k.transpose(1, 2))
        dv_tensor.copy_(dV_k.transpose(1, 2))
        if dbias_tensor is not None and dBias_k is not None:
            # dBias is head-major [., H, SQ, SKV] (like bias) — no transpose.
            dbias_tensor.copy_(dBias_k.to(dbias_tensor.dtype))
        if dsink_tensor is not None and dSink_k is not None:
            dsink_tensor.copy_(dSink_k.to(dsink_tensor.dtype))
        self._logger.debug("execute (bwd) completed")


# ---------------------------------------------------------------------------
# THD / varlen backward (mirrors fwd/api.py::_thd_forward).
# ---------------------------------------------------------------------------
def _thd_backward(q, k, v, o, do, lse, *, cu_q, cu_k, scale_softmax, is_causal, window_size, causal_bottom_right, alibi=False, sinks=None, deterministic=False):
    """THD / varlen backward: q/k/v/o/do are PACKED ``[1, T, H, D]`` (BSHD,
    B==1 — no transpose), ``lse`` is packed ``[1, H, T_q]`` (head-major,
    matching the kernel's THD LSE layout), and cu_q/cu_k are ``[n_seq+1]``
    cumulative seqlens.  Routes straight to the kernel's THD backward
    (over-provisioned grid; MHA-only), reusing the flavor-pick + d-pad.
    Returns packed ``[1, T, H, D]`` dQ/dK/dV (BHSD-equivalent for B==1)."""
    d_qk = q.shape[-1]
    d_v = v.shape[-1]
    h_q = q.shape[2]
    flavor = _pick_flavor(d_qk, d_v)
    fdqk, fdv = _FLAVOR_DIMS[flavor]
    # Resolve the default scale from the USER's head dim before padding: the
    # kernel would otherwise derive 1/sqrt(D) from the padded flavor width
    # (e.g. 1/sqrt(128) for a d=96 llama-flavor call) — silently wrong
    # gradients.  Mirrors the forward THD path.
    if scale_softmax is None or scale_softmax == 0.0:
        scale_softmax = 1.0 / math.sqrt(d_qk)
    pad_qk = d_qk < fdqk
    pad_v = d_v < fdv
    if pad_qk:
        q = _pad_last_dim(q, fdqk)
        k = _pad_last_dim(k, fdqk)
    if pad_v:
        v = _pad_last_dim(v, fdv)
        o = _pad_last_dim(o, fdv)
        do = _pad_last_dim(do, fdv)
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
    kernel = _load_kernel_module()
    alibi_slopes = kernel.default_alibi_slopes(h_q).to(q.device) if alibi else None
    sinks_t = sinks.to(dtype=torch.float32, device=q.device).reshape(h_q).contiguous() if sinks is not None else None
    bw_kwargs = dict(
        scale=scale_softmax,
        mask=mask_token,
        swa_window=int(swa),
        right_bound=int(right_bound),
        causal_bottom_right=bool(causal_bottom_right),
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        alibi_slopes=alibi_slopes,
        sinks=sinks_t,
        deterministic=bool(deterministic),
    )
    acc = inspect.signature(kernel.backward).parameters
    bw_kwargs = {kk: vv for kk, vv in bw_kwargs.items() if kk in acc}
    res = kernel.backward(q, k, v, do, o, lse, **bw_kwargs)
    dQ_k, dK_k, dV_k = res[0], res[1], res[2]
    _idx = 3
    dSink_k = None
    if sinks_t is not None:
        dSink_k = res[_idx]
        _idx += 1
    if pad_qk:
        dQ_k = dQ_k[..., :d_qk].contiguous()
        dK_k = dK_k[..., :d_qk].contiguous()
    if pad_v:
        dV_k = dV_k[..., :d_v].contiguous()
    out = TupleDict(dq_tensor=dQ_k, dk_tensor=dK_k, dv_tensor=dV_k)
    if dSink_k is not None:
        out["dsink_tensor"] = dSink_k
    return out


# ---------------------------------------------------------------------------
# Functional wrapper (mirrors the forward surface).
# ---------------------------------------------------------------------------
_cache_of_objects: dict = {}


def sdpa_bwd_wrapper_sm80(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    o_tensor: torch.Tensor,
    do_tensor: torch.Tensor,
    lse_tensor: torch.Tensor,
    is_causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    scale_softmax: Optional[float] = None,
    causal_bottom_right: bool = False,
    current_stream: Optional[cuda.CUstream] = None,
    seq_kv_lens: Optional[torch.Tensor] = None,
    seq_len_q: Optional[torch.Tensor] = None,
    bias_tensor: Optional[torch.Tensor] = None,
    alibi: bool = False,
    sinks: Optional[torch.Tensor] = None,
    rope_freqs: Optional[torch.Tensor] = None,
    block_mask: Optional[torch.Tensor] = None,
    cum_seqlen_q_tensor: Optional[torch.Tensor] = None,
    cum_seqlen_k_tensor: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> TupleDict:
    """SM80 (A100) SDPA backward.

    Returns ``TupleDict(dq_tensor=..., dk_tensor=..., dv_tensor=...
    [, dbias_tensor=...])`` (BHSD grads; dBias head-major [., H, SQ, SKV]).
    """
    # THD / varlen: q/k/v/o/dO are PACKED [1, T, H, D] (BSHD) + cu_seqlens;
    # lse is packed [1, H, T_q].  Dedicated path that skips the dense BHSD
    # transpose + dense grad alloc (mirrors fwd/api.py's THD branch).
    if cum_seqlen_q_tensor is not None:
        # Reject dense-only features up front: _thd_backward accepts only
        # alibi/sinks/deterministic, and silently computing gradients without
        # a requested feature is worse than an error.
        for label, present in (
            ("bias_tensor", bias_tensor is not None),
            ("rope_freqs", rope_freqs is not None),
            ("block_mask", block_mask is not None),
            ("seq_kv_lens", seq_kv_lens is not None),
            ("seq_len_q", seq_len_q is not None),
        ):
            if present:
                raise NotImplementedError(f"SM80 SDPA THD (cum_seqlen_*) backward does not support {label}; the dense path serves it")
        with _stream_ctx(current_stream):
            return _thd_backward(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                do_tensor,
                lse_tensor,
                cu_q=cum_seqlen_q_tensor,
                cu_k=cum_seqlen_k_tensor,
                scale_softmax=scale_softmax,
                is_causal=is_causal,
                window_size=window_size,
                causal_bottom_right=causal_bottom_right,
                alibi=alibi,
                sinks=sinks,
                deterministic=deterministic,
            )
    for nm, t in (("Q", q_tensor), ("V", v_tensor), ("O", o_tensor), ("dO", do_tensor)):
        if t.ndim != 4:
            raise ValueError(f"{nm} must be rank-4 BHSD; got {t.ndim}D")

    # Allocate grad outputs in cuDNN-FE BHSD-physical stride order (3,1,2,0):
    # contiguous (B, S, H, D) then transpose to a (B, H, S, D) view.
    b, h_q, s_q, d_qk = q_tensor.shape
    d_v = v_tensor.shape[-1]
    dq = torch.empty((b, s_q, h_q, d_qk), dtype=q_tensor.dtype, device=q_tensor.device).transpose(1, 2)
    # dK/dV take K/V leading shape (GQA: h_kv heads).
    h_kv, s_kv = k_tensor.shape[1], k_tensor.shape[2]
    dk = torch.empty((b, s_kv, h_kv, d_qk), dtype=q_tensor.dtype, device=q_tensor.device).transpose(1, 2)
    dv = torch.empty((b, s_kv, h_kv, d_v), dtype=q_tensor.dtype, device=q_tensor.device).transpose(1, 2)
    # dBias: fp32, same shape as bias ([., H, SQ, SKV]).
    dbias = torch.zeros_like(bias_tensor, dtype=torch.float32) if bias_tensor is not None else None
    # dSink: fp32 [H] (sink-logit gradient).
    dsink = torch.zeros(h_q, dtype=torch.float32, device=q_tensor.device) if sinks is not None else None

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
        causal_bottom_right,
        seq_kv_lens is not None,
        bias_tensor is not None,
        (bias_tensor.dtype if bias_tensor is not None else None),
        bool(alibi),
        sinks is not None,
        rope_freqs is not None,
        block_mask is not None,
        q_tensor.device,
    )
    sdpa_bwd = _cache_of_objects.get(cache_key)
    if sdpa_bwd is None:
        _logger.debug("sdpa_bwd_wrapper_sm80: building new SdpabwdSm80")
        sdpa_bwd = SdpabwdSm80(
            sample_q=q_tensor,
            sample_k=k_tensor,
            sample_v=v_tensor,
            sample_o=o_tensor,
            sample_do=do_tensor,
            sample_lse=lse_tensor,
            is_causal=is_causal,
            window_size=window_size,
            scale_softmax=scale_softmax,
            causal_bottom_right=causal_bottom_right,
            has_seq_kv_lens=seq_kv_lens is not None,
            has_bias=bias_tensor is not None,
        )
        assert sdpa_bwd.check_support(), "Unsupported configuration"
        sdpa_bwd.compile()
        _cache_of_objects[cache_key] = sdpa_bwd

    sdpa_bwd.execute(
        q_tensor=q_tensor,
        k_tensor=k_tensor,
        v_tensor=v_tensor,
        o_tensor=o_tensor,
        do_tensor=do_tensor,
        lse_tensor=lse_tensor,
        dq_tensor=dq,
        dk_tensor=dk,
        dv_tensor=dv,
        dbias_tensor=dbias,
        dsink_tensor=dsink,
        scale_softmax=scale_softmax,
        current_stream=current_stream,
        seq_kv_lens=seq_kv_lens,
        seq_len_q=seq_len_q,
        bias_tensor=bias_tensor,
        alibi=alibi,
        sinks=sinks,
        rope_freqs=rope_freqs,
        block_mask=block_mask,
        deterministic=deterministic,
    )

    out = TupleDict(dq_tensor=dq, dk_tensor=dk, dv_tensor=dv)
    if dbias is not None:
        out["dbias_tensor"] = dbias
    if dsink is not None:
        out["dsink_tensor"] = dsink
    return out
