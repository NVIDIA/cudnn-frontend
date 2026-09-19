# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FE API for the subchannel-scaled (second-level SFA2) grouped GEMM weight gradient on SM100+.

Per expert ``e`` with token range ``[k0, k1)`` split into ``sgk``-token scale blocks ``b``::

    dW[e] = sum_b (A[:, b] @ B[b, :]) * sfa2[:, b_idx]      # f32, ascending b
    dW[e] = (dW[e] * global_scale_a[e] * global_scale_b[e]).to(bf16)

``A = a_fp4 * sfa`` and ``B = b_fp4 * sfb`` are the first-level NVFP4 operands (K = the ragged
token axis), and ``sfa2`` is the per-(hidden row x sgk tokens) f32 second-level A scale (no
second-level B scale in wgrad). Every expert's token count must be a multiple of ``sgk`` so
that second-level scale blocks never straddle experts.
"""

from __future__ import annotations

import os
from typing import Literal, Optional, Tuple

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass.cute.runtime import from_dlpack, make_fake_stream

from cudnn.api_base import APIBase, TensorDesc, TupleDict, ceil_div, get_device_type
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.gemm.cutedsl.grouped.unfused._bf16_api import _validate_pointer_tensor
from cudnn.tensor_adapter import canonicalize_unit_dim_strides, detect_framework, framework_dtype, get_shape, is_torch_tensor

from ..backend_utils import _torch_stream_context
from ..moe_utils import MoEWeightMode, WGradInputOrder
from ..wgrad.api import _wgrad_tensor_signature
from .grouped_gemm_wgrad_subchannel_scaled import (
    BlockScaledSubChannelMoEGroupedGemmWgradKernel,
    DEFAULT_CTA_SHAPE,
    VALID_CLUSTER_SHAPES,
    VALID_CTA_SHAPES,
)

# NVFP4 first level: e4m3 scale per (1, 16) block; the MMA K tile is 256 tokens.
_SF_VEC_SIZE = 16
_MMA_TILER_K = 256


def _get_rubin_kernel():
    from .moe_blockscaled_grouped_gemm_wgrad_subchannel_scaled_rubin import (
        BlockScaledSubChannelMoEGroupedGemmWgradKernelSm107,
    )

    return BlockScaledSubChannelMoEGroupedGemmWgradKernelSm107


def _round_up(a: int, b: int) -> int:
    return ceil_div(a, b) * b


class GroupedGemmWgradSubchannelScaledSm100(APIBase):
    """Subchannel-scaled grouped GEMM wgrad (2Dx2D) FE API for SM100+ GPUs.

    :param sample_a: ``(hidden, tokens_sum)`` packed fp4 dY^T, K(token)-major
    :param sample_b: ``(tokens_sum, intermediate)`` packed fp4 activations, K(token)-major
    :param sample_sfa: ``(round_up(hidden, 128), round_up(tokens_sum / 16, 4))`` e4m3 first-level A
        scales in the assembled (MMA-tiled, per-expert) block-scaled layout
    :param sample_sfb: same for B over ``intermediate``
    :param sample_sfa2: ``(hidden, tokens_sum / sgk)`` f32 second-level A scales, hidden-contiguous
        (strides ``(1, hidden)`` -- the rowwise-quant producer's ``sf2`` view, consumed zero-copy)
    :param sample_offsets: ``(expert_cnt,)`` int32 cumulative END token offsets
    :param sgk: second-level scale group size along the token axis (multiple of 256)
    :param sample_wgrad: dense mode: ``(expert_cnt, hidden, intermediate)`` bf16 output
    :param sample_wgrad_expert / num_experts / wgrad_shape / wgrad_dtype: discrete (per-expert
        pointer array) output mode, as in ``GroupedGemmWgradSm100``
    :param sample_global_scale_a / sample_global_scale_b: optional ``(expert_cnt,)`` f32 per-expert
        tensorwise scales (applied as ``alpha = gsa[e] * gsb[e]``; both or neither)
    :param mma_tiler_mn: ``(128, 128)`` or ``(256, 128)`` (2-CTA MMA iff M == 256)
    :param cluster_shape_mn: cluster shape; default ``(2, 1)`` for 2-CTA tilers, ``(1, 1)`` otherwise
    :param sf_fp8_dtype_override: ``"e5m3"`` reinterprets the e4m3-typed first-level scales as
        e5m3 (Rubin only)
    :param accumulate_on_output: TMA reduce-add into the caller's output instead of storing
    """

    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_b: torch.Tensor,
        sample_sfa: torch.Tensor,
        sample_sfb: torch.Tensor,
        sample_sfa2: torch.Tensor,
        sample_offsets: torch.Tensor,
        sgk: int,
        sample_wgrad: Optional[torch.Tensor] = None,
        sample_wgrad_expert: Optional[torch.Tensor] = None,
        num_experts: Optional[int] = None,
        wgrad_shape: Optional[Tuple[int, int]] = None,
        wgrad_dtype: Optional[torch.dtype] = None,
        sample_global_scale_a: Optional[torch.Tensor] = None,
        sample_global_scale_b: Optional[torch.Tensor] = None,
        acc_dtype: Optional[torch.dtype] = None,
        mma_tiler_mn: Tuple[int, int] = DEFAULT_CTA_SHAPE,
        cluster_shape_mn: Optional[Tuple[int, int]] = None,
        sf_fp8_dtype_override: Optional[Literal["e5m3"]] = None,
        accumulate_on_output: bool = False,
    ):
        if sample_a is not None and not is_torch_tensor(sample_a):
            raise ValueError(
                "GroupedGemmWgradSubchannelScaledSm100 supports torch tensors only: its fp4 operands and "
                "hidden-contiguous second-level scales require layouts that are not expressible as row-major JAX arrays"
            )
        import torch

        acc_dtype = framework_dtype(acc_dtype, "torch") if acc_dtype is not None else torch.float32
        wgrad_dtype = framework_dtype(wgrad_dtype, "torch") if wgrad_dtype is not None else None
        super().__init__()
        self._warn_experimental_api()
        self.input_order = WGradInputOrder.Tensor2D

        if sample_wgrad is not None and num_experts is None:
            self.weight_mode = MoEWeightMode.DENSE
        elif sample_wgrad is None and num_experts is not None:
            self.weight_mode = MoEWeightMode.DISCRETE
            if wgrad_shape is None or wgrad_dtype is None:
                raise ValueError("wgrad_shape and wgrad_dtype are required in discrete mode")
        else:
            raise ValueError("Provide either sample_wgrad for dense mode or (num_experts, wgrad_shape, wgrad_dtype) for discrete mode, but not both")

        self._interpret_uint8_as_fp4x2 = True
        self.sample_a_tensor = sample_a if self._is_fp4x2(sample_a) else None
        self.sample_b_tensor = sample_b if self._is_fp4x2(sample_b) else None
        self.a_desc = self._make_tensor_desc(sample_a, name="sample_a")
        self.b_desc = self._make_tensor_desc(sample_b, name="sample_b")
        self.sfa_desc = self._make_tensor_desc(sample_sfa, name="sample_sfa")
        self.sfb_desc = self._make_tensor_desc(sample_sfb, name="sample_sfb")
        self.sfa2_desc = self._make_tensor_desc(sample_sfa2, name="sample_sfa2")
        self.offsets_desc = self._make_tensor_desc(sample_offsets, name="sample_offsets")
        self.global_scale_a_desc = self._make_tensor_desc(sample_global_scale_a, name="sample_global_scale_a")
        self.global_scale_b_desc = self._make_tensor_desc(sample_global_scale_b, name="sample_global_scale_b")
        self.sf_vec_size = _SF_VEC_SIZE
        self.sgk = int(sgk)
        self.sf_fp8_dtype_override = sf_fp8_dtype_override
        tokens_sum_a = self.a_desc.shape[1]
        tokens_sum_b = self.b_desc.shape[0]
        self._value_error_if(
            tokens_sum_a != tokens_sum_b,
            f"sample_a and sample_b token dimensions must match, got {tokens_sum_a} and {tokens_sum_b}",
        )
        self._offset_values = self._validate_offsets(sample_offsets, tokens_sum_a, name="sample_offsets")
        self._scale_cols = _round_up(ceil_div(tokens_sum_a, self.sf_vec_size), 4)

        if self.weight_mode == MoEWeightMode.DENSE:
            self.wgrad_desc = self._make_tensor_desc(sample_wgrad, name="sample_wgrad")
            self.expert_cnt = self.wgrad_desc.shape[0]
            self.wgrad_shape = self.wgrad_desc.shape[1:]
            self.wgrad_dtype = self.wgrad_desc.dtype
            self.single_expert_wgrad_desc = TensorDesc(
                dtype=self.wgrad_desc.dtype,
                shape=self.wgrad_desc.shape[1:],
                stride=self.wgrad_desc.stride[1:],
                stride_order=tuple(i for i, s in sorted(enumerate(self.wgrad_desc.stride[1:]), key=lambda x: x[1])),
                device=self.wgrad_desc.device,
                name="single_expert_wgrad",
            )
        else:  # MoEWeightMode.DISCRETE
            self.expert_cnt = num_experts
            self.wgrad_shape = tuple(wgrad_shape)
            self.wgrad_dtype = wgrad_dtype
            self.wgrad_desc = None
            if sample_wgrad_expert is not None:
                self.single_expert_wgrad_desc = self._make_tensor_desc(sample_wgrad_expert, name="sample_wgrad_expert")
            else:
                self.single_expert_wgrad_desc = TensorDesc(
                    dtype=wgrad_dtype,
                    shape=self.wgrad_shape,
                    stride=(self.wgrad_shape[1], 1),
                    stride_order=(1, 0),
                    device=self.a_desc.device,
                    name="single_expert_wgrad",
                )

        self.acc_dtype = acc_dtype
        self.mma_tiler_mn = tuple(mma_tiler_mn)
        self.use_2cta_instrs = self.mma_tiler_mn[0] == 256
        self.cluster_shape_mn = tuple(cluster_shape_mn) if cluster_shape_mn is not None else ((2, 1) if self.use_2cta_instrs else (1, 1))
        self.accumulate_on_output = accumulate_on_output
        self._kernel = _get_rubin_kernel() if self._is_rubin_kernel else BlockScaledSubChannelMoEGroupedGemmWgradKernel
        self._workspace = None

    def _validate_offsets(self, offsets_tensor: torch.Tensor, tokens_sum: int, name: str) -> Tuple[int, ...]:
        self._value_error_if(offsets_tensor.ndim != 1, f"{name} must be rank-1, got shape {tuple(offsets_tensor.shape)}")

        offset_values = tuple(int(offset) for offset in offsets_tensor.detach().cpu().tolist())
        prev_offset = 0
        for idx, offset in enumerate(offset_values):
            self._value_error_if(
                offset < prev_offset,
                f"{name} must be a non-decreasing cumulative sum, but index {idx} has {offset} after {prev_offset}",
            )
            prev_offset = offset

        self._value_error_if(not offset_values, f"{name} cannot be empty")
        self._value_error_if(
            offset_values[-1] > tokens_sum,
            f"{name} last value must not exceed total tokens {tokens_sum}, got {offset_values[-1]}",
        )
        return offset_values

    def check_support(self) -> bool:
        import torch

        m, tokens_sum = self._tensor_shape(self.a_desc, name="sample_a")
        _, n = self._tensor_shape(self.b_desc, name="sample_b")

        _ = self._check_tensor_shape(self.a_desc, (m, tokens_sum), "sample_a")
        _ = self._check_tensor_shape(self.b_desc, (tokens_sum, n), "sample_b")
        _ = self._check_tensor_shape(self.sfa_desc, (_round_up(m, 128), self._scale_cols), "sample_sfa")
        _ = self._check_tensor_shape(self.sfb_desc, (_round_up(n, 128), self._scale_cols), "sample_sfb")
        _ = self._check_tensor_shape(self.offsets_desc, (self.expert_cnt,), "sample_offsets")

        dtype = self._check_dtype(self.a_desc, [torch.float4_e2m1fn_x2, torch.uint8], "sample_a")
        self._check_dtype(self.b_desc, dtype, "sample_b", extra_error_msg="sample_b must have the same dtype as sample_a")
        self._check_dtype(self.sfa_desc, torch.float8_e4m3fn, "sample_sfa", extra_error_msg="sample_sfa must have dtype float8_e4m3fn (NVFP4 first level)")
        self._check_dtype(self.sfb_desc, torch.float8_e4m3fn, "sample_sfb", extra_error_msg="sample_sfb must have dtype float8_e4m3fn (NVFP4 first level)")
        self._check_dtype(self.offsets_desc, torch.int32, "sample_offsets", extra_error_msg="sample_offsets must be int32")
        self._check_dtype(self.wgrad_dtype, torch.bfloat16, "wgrad_dtype", extra_error_msg="wgrad_dtype must be bfloat16")
        self._value_error_if(self.acc_dtype != torch.float32, f"acc_dtype must be float32, got {self.acc_dtype}")

        # The tcgen05 block-scaled MMA consumes K-major fp4 operands; the kernel transposes B by
        # stride swap, so both A (hidden, tokens) and B (tokens, intermediate) must be token-innermost.
        self._value_error_if(
            self.a_desc.stride[1] != 1 or self.b_desc.stride[0] != 1,
            "sample_a (hidden, tokens_sum) and sample_b (tokens_sum, intermediate) must be K(token)-major",
        )
        # SF atom layouts tile M/N in 128-row atoms.
        self._value_error_if(m % 128 != 0 or n % 128 != 0, f"hidden and intermediate must be multiples of 128, got hidden={m}, intermediate={n}")

        # Second-level scale group and its grid.
        self._value_error_if(
            self.sgk <= 0 or self.sgk % _MMA_TILER_K != 0,
            f"sgk must be a positive multiple of the {_MMA_TILER_K}-token MMA K tile, got {self.sgk}",
        )
        self._value_error_if(
            tokens_sum % self.sgk != 0,
            f"tokens_sum ({tokens_sum}) must be a multiple of sgk ({self.sgk})",
        )
        # Whole scale blocks per expert: token_offset // sgk is the exact per-expert SFA2 column
        # offset only when no block straddles experts. This is checked on the sample offsets; the
        # same contract holds for every execute() with different offsets.
        prev = 0
        for idx, end in enumerate(self._offset_values):
            self._value_error_if(
                (end - prev) % self.sgk != 0,
                f"per-expert token count must be sgk-aligned so second-level scale blocks never straddle experts, "
                f"but expert {idx} has {end - prev} tokens (sgk={self.sgk})",
            )
            prev = end
        scale_cols = tokens_sum // self.sgk
        _ = self._check_tensor_shape(self.sfa2_desc, (m, scale_cols), "sample_sfa2")
        self._check_dtype(self.sfa2_desc, torch.float32, "sample_sfa2", extra_error_msg="sample_sfa2 must be float32")
        expected_sfa2_stride = canonicalize_unit_dim_strides((m, scale_cols), (1, m))
        self._value_error_if(
            tuple(self.sfa2_desc.stride) != tuple(expected_sfa2_stride),
            f"sample_sfa2 must be hidden-contiguous with strides {expected_sfa2_stride} (the rowwise-quant producer's sf2 view), "
            f"got {tuple(self.sfa2_desc.stride)}",
        )

        # torch has no e5m3 dtype and TVM-FFI cannot marshal FloatNV8E5M3FNU, so e5m3 scale
        # factors arrive as e4m3 storage of the same width and the Rubin kernel reinterprets them.
        self._value_error_if(
            self.sf_fp8_dtype_override not in (None, "e5m3"),
            f"sf_fp8_dtype_override must be None or 'e5m3', got {self.sf_fp8_dtype_override!r}",
        )
        if self.sf_fp8_dtype_override == "e5m3":
            self._value_error_if(
                not self._is_rubin_kernel,
                f"sf_fp8_dtype_override='e5m3' requires Rubin (SM107), got device type {self._device_type!r}",
            )

        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_tensor_shape(self.wgrad_desc, (self.expert_cnt, m, n), "sample_wgrad")
            self._value_error_if(
                tuple(self.wgrad_desc.stride) != tuple(canonicalize_unit_dim_strides((self.expert_cnt, m, n), (m * n, n, 1))),
                f"sample_wgrad must be a contiguous (expert_cnt, hidden, intermediate) tensor, got strides {tuple(self.wgrad_desc.stride)}",
            )
        else:
            self._check_tensor_shape(self.wgrad_shape, (m, n), "wgrad_shape")
            self._check_tensor_shape(self.single_expert_wgrad_desc, (m, n), "single_expert_wgrad")
            self._check_dtype(
                self.single_expert_wgrad_desc,
                self.wgrad_dtype,
                "sample_wgrad_expert",
                extra_error_msg="sample_wgrad_expert must have the same dtype as wgrad_dtype",
            )

        self._value_error_if(self.mma_tiler_mn not in VALID_CTA_SHAPES, f"mma_tiler_mn must be one of {VALID_CTA_SHAPES}, got {self.mma_tiler_mn}")
        self._value_error_if(
            self.cluster_shape_mn not in VALID_CLUSTER_SHAPES, f"cluster_shape_mn must be one of {VALID_CLUSTER_SHAPES}, got {self.cluster_shape_mn}"
        )
        self._value_error_if(
            self.cluster_shape_mn[0] % (2 if self.use_2cta_instrs else 1) != 0,
            f"cluster_shape_mn[0] must be even for the 2-CTA (M=256) tiler, got {self.cluster_shape_mn[0]}",
        )

        has_global_scale = self.global_scale_a_desc is not None or self.global_scale_b_desc is not None
        if has_global_scale:
            self._value_error_if(
                self.global_scale_a_desc is None or self.global_scale_b_desc is None,
                "sample_global_scale_a and sample_global_scale_b must be provided together",
            )
            self._check_tensor_shape(self.global_scale_a_desc, (self.expert_cnt,), "sample_global_scale_a")
            self._check_tensor_shape(self.global_scale_b_desc, (self.expert_cnt,), "sample_global_scale_b")
            self._check_dtype(self.global_scale_a_desc, torch.float32, "sample_global_scale_a")
            self._check_dtype(self.global_scale_b_desc, torch.float32, "sample_global_scale_b")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        compute_capability = major * 10 + minor
        if compute_capability < 100:
            raise RuntimeError(f"GroupedGemmWgradSubchannelScaled requires SM100+ compute capability, but found SM{compute_capability} on device {device}")

        self._is_supported = True
        return True

    def compile(self) -> None:
        import torch

        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return

        kernel = self._kernel(
            sf_vec_size=self.sf_vec_size,
            sgk=self.sgk,
            acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
            use_2cta_instrs=self.use_2cta_instrs,
            mma_tiler_mn=self.mma_tiler_mn,
            cluster_shape_mn=self.cluster_shape_mn,
            accumulate_on_output=self.accumulate_on_output,
            expert_cnt=self.expert_cnt,
            weight_mode=self.weight_mode,
            input_order=self.input_order,
            sf_fp8_dtype_override=self.sf_fp8_dtype_override,
        )

        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(self.cluster_shape_mn[0] * self.cluster_shape_mn[1])
        self._workspace = torch.empty(max(kernel.get_workspace_bytes(), 1), dtype=torch.uint8, device=self.a_desc.device)
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)

        if self.weight_mode == MoEWeightMode.DENSE:
            self._compile_dense(kernel, max_active_clusters, fake_stream)
        else:
            self._compile_discrete(kernel, max_active_clusters, fake_stream)

        if self.sample_a_tensor is not None:
            del self.sample_a_tensor
        if self.sample_b_tensor is not None:
            del self.sample_b_tensor

    def _make_operand_fakes(self):
        """A/B/SFA/SFB/SFA2 fake tensors with the ragged token axis dynamic (one compile serves
        every token distribution)."""
        a_fake = (
            from_dlpack(self.sample_a_tensor, assumed_align=16, enable_tvm_ffi=True).mark_compact_shape_dynamic(
                mode=1,
                stride_order=self.sample_a_tensor.dim_order(),
                divisibility=16,
            )
            if self.sample_a_tensor is not None
            else self._make_fake_cute_compact_tensor(
                dtype=self.a_desc.dtype,
                shape=self.a_desc.shape,
                stride_order=self.a_desc.stride_order,
                assumed_align=16,
                dynamic_mode=1,
                divisibility=16,
            )
        )
        b_fake = (
            from_dlpack(self.sample_b_tensor, assumed_align=16, enable_tvm_ffi=True).mark_compact_shape_dynamic(
                mode=0,
                stride_order=self.sample_b_tensor.dim_order(),
                divisibility=16,
            )
            if self.sample_b_tensor is not None
            else self._make_fake_cute_compact_tensor(
                dtype=self.b_desc.dtype,
                shape=self.b_desc.shape,
                stride_order=self.b_desc.stride_order,
                assumed_align=16,
                dynamic_mode=0,
                divisibility=16,
            )
        )
        sfa_fake = self._make_fake_cute_compact_tensor(
            dtype=self.sfa_desc.dtype,
            shape=self.sfa_desc.shape,
            stride_order=self.sfa_desc.stride_order,
            assumed_align=16,
            dynamic_mode=1,
            divisibility=4,
        )
        sfb_fake = self._make_fake_cute_compact_tensor(
            dtype=self.sfb_desc.dtype,
            shape=self.sfb_desc.shape,
            stride_order=self.sfb_desc.stride_order,
            assumed_align=16,
            dynamic_mode=1,
            divisibility=4,
        )
        # (hidden, tokens_sum / sgk), hidden-contiguous; the column count follows the token axis.
        sfa2_fake = self._make_fake_cute_compact_tensor(
            dtype=self.sfa2_desc.dtype,
            shape=self.sfa2_desc.shape,
            stride_order=self.sfa2_desc.stride_order,
            assumed_align=16,
            dynamic_mode=1,
            divisibility=1,
        )
        return a_fake, b_fake, sfa_fake, sfb_fake, sfa2_fake

    def _compile_dense(self, kernel, max_active_clusters, fake_stream) -> None:
        a_fake, b_fake, sfa_fake, sfb_fake, sfa2_fake = self._make_operand_fakes()
        wgrad_fake = self._make_fake_cute_tensor_from_desc(self.wgrad_desc, assumed_align=16)
        offsets_fake = self._make_fake_cute_tensor_from_desc(self.offsets_desc, assumed_align=4)
        workspace_fake = from_dlpack(self._workspace, assumed_align=128, enable_tvm_ffi=True)
        gs_a_fake = self._make_fake_cute_tensor_from_desc(self.global_scale_a_desc, assumed_align=4)
        gs_b_fake = self._make_fake_cute_tensor_from_desc(self.global_scale_b_desc, assumed_align=4)

        compiled = cute.compile(
            kernel,
            a_fake,
            b_fake,
            sfa_fake,
            sfb_fake,
            sfa2_fake,
            wgrad_fake,
            offsets_fake,
            workspace_fake,
            max_active_clusters,
            fake_stream,
            gs_a_fake,
            gs_b_fake,
            None,
            options="--enable-tvm-ffi",
        )

        cached_workspace = from_dlpack(self._workspace, assumed_align=128, enable_tvm_ffi=True)

        def tensor_api(
            a_tensor: torch.Tensor,
            b_tensor: torch.Tensor,
            sfa_tensor: torch.Tensor,
            sfb_tensor: torch.Tensor,
            sfa2_tensor: torch.Tensor,
            wgrad_tensor: torch.Tensor,
            offsets_tensor: torch.Tensor,
            stream: cuda.CUstream,
            global_scale_a: Optional[torch.Tensor],
            global_scale_b: Optional[torch.Tensor],
        ) -> None:
            compiled(
                a_tensor,
                b_tensor,
                sfa_tensor,
                sfb_tensor,
                sfa2_tensor,
                wgrad_tensor,
                offsets_tensor,
                cached_workspace,
                stream,
                global_scale_a,
                global_scale_b,
                None,
            )

        self._compiled_kernel = tensor_api

    def _compile_discrete(self, kernel, max_active_clusters, fake_stream) -> None:
        import torch

        a_fake, b_fake, sfa_fake, sfb_fake, sfa2_fake = self._make_operand_fakes()
        offsets_fake = self._make_fake_cute_tensor_from_desc(self.offsets_desc, assumed_align=4)
        workspace_fake = from_dlpack(self._workspace, assumed_align=128, enable_tvm_ffi=True)
        gs_a_fake = self._make_fake_cute_tensor_from_desc(self.global_scale_a_desc, assumed_align=4)
        gs_b_fake = self._make_fake_cute_tensor_from_desc(self.global_scale_b_desc, assumed_align=4)
        wgrad_ptrs_placeholder = torch.empty((self.expert_cnt,), dtype=torch.int64, device=self.a_desc.device)
        wgrad_ptrs_fake = from_dlpack(wgrad_ptrs_placeholder, assumed_align=8, enable_tvm_ffi=True).iterator
        single_expert_fake = self._make_fake_cute_tensor(
            dtype=self.single_expert_wgrad_desc.dtype,
            shape=self.single_expert_wgrad_desc.shape,
            stride=self.single_expert_wgrad_desc.stride,
            assumed_align=16,
        )

        compiled = cute.compile(
            kernel,
            a_fake,
            b_fake,
            sfa_fake,
            sfb_fake,
            sfa2_fake,
            wgrad_ptrs_fake,
            offsets_fake,
            workspace_fake,
            max_active_clusters,
            fake_stream,
            gs_a_fake,
            gs_b_fake,
            single_expert_fake,
            options="--enable-tvm-ffi",
        )

        cached_workspace = from_dlpack(self._workspace, assumed_align=128, enable_tvm_ffi=True)
        single_expert_placeholder = torch.empty_strided(
            self.single_expert_wgrad_desc.shape,
            self.single_expert_wgrad_desc.stride,
            dtype=self.single_expert_wgrad_desc.dtype,
            device=self.single_expert_wgrad_desc.device,
        )
        cached_single_expert = from_dlpack(single_expert_placeholder, assumed_align=16, enable_tvm_ffi=True)

        def tensor_api(
            a_tensor: torch.Tensor,
            b_tensor: torch.Tensor,
            sfa_tensor: torch.Tensor,
            sfb_tensor: torch.Tensor,
            sfa2_tensor: torch.Tensor,
            wgrad_ptrs: torch.Tensor,
            offsets_tensor: torch.Tensor,
            stream: cuda.CUstream,
            global_scale_a: Optional[torch.Tensor],
            global_scale_b: Optional[torch.Tensor],
        ) -> None:
            compiled(
                a_tensor,
                b_tensor,
                sfa_tensor,
                sfb_tensor,
                sfa2_tensor,
                wgrad_ptrs.data_ptr(),
                offsets_tensor,
                cached_workspace,
                stream,
                global_scale_a,
                global_scale_b,
                cached_single_expert,
            )

        self._compiled_kernel = tensor_api

    def execute(
        self,
        a_tensor: torch.Tensor,
        b_tensor: torch.Tensor,
        sfa_tensor: torch.Tensor,
        sfb_tensor: torch.Tensor,
        sfa2_tensor: torch.Tensor,
        offsets_tensor: torch.Tensor,
        wgrad_tensor: Optional[torch.Tensor] = None,
        wgrad_ptrs: Optional[torch.Tensor] = None,
        global_scale_a: Optional[torch.Tensor] = None,
        global_scale_b: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        import torch

        current_stream = self._get_default_stream(current_stream)
        self._runtime_error_if(self._compiled_kernel is None, "Kernel not compiled; call compile() first")

        if self.weight_mode == MoEWeightMode.DENSE:
            self._value_error_if(wgrad_tensor is None, "wgrad_tensor is required in dense mode")
            self._compiled_kernel(
                a_tensor,
                b_tensor,
                sfa_tensor,
                sfb_tensor,
                sfa2_tensor,
                wgrad_tensor,
                offsets_tensor,
                current_stream,
                global_scale_a,
                global_scale_b,
            )
            return

        if wgrad_ptrs is None:
            self._value_error_if(wgrad_tensor is None, "Provide wgrad_tensor or wgrad_ptrs in discrete mode")
            self._value_error_if(wgrad_tensor.ndim != 3, f"wgrad_tensor must be rank-3, got {tuple(wgrad_tensor.shape)}")
            self._value_error_if(not wgrad_tensor.is_cuda, f"wgrad_tensor must be a CUDA tensor, got {wgrad_tensor.device}")
            if wgrad_tensor.shape[0] == 0:
                wgrad_ptrs = torch.empty((0,), dtype=torch.int64, device=wgrad_tensor.device)
            else:
                expert_stride_bytes = wgrad_tensor.stride(0) * wgrad_tensor.element_size()
                ptrs = [wgrad_tensor.data_ptr() + i * expert_stride_bytes for i in range(wgrad_tensor.shape[0])]
                wgrad_ptrs = torch.tensor(ptrs, dtype=torch.int64, device=wgrad_tensor.device)
        _validate_pointer_tensor(wgrad_ptrs, "wgrad_ptrs", self.expert_cnt)
        self._compiled_kernel(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            sfa2_tensor,
            wgrad_ptrs,
            offsets_tensor,
            current_stream,
            global_scale_a,
            global_scale_b,
        )


_cache_of_GroupedGemmWgradSubchannelScaledSm100Objects = {}


def grouped_gemm_wgrad_subchannel_scaled_wrapper_sm100(
    a_tensor: torch.Tensor,
    b_tensor: torch.Tensor,
    sfa_tensor: torch.Tensor,
    sfb_tensor: torch.Tensor,
    sfa2_tensor: torch.Tensor,
    offsets_tensor: torch.Tensor,
    sgk: int,
    output_mode: str = "dense",
    wgrad_tensor: Optional[torch.Tensor] = None,
    wgrad_ptrs: Optional[torch.Tensor] = None,
    global_scale_a: Optional[torch.Tensor] = None,
    global_scale_b: Optional[torch.Tensor] = None,
    acc_dtype: Optional[torch.dtype] = None,
    wgrad_dtype: Optional[torch.dtype] = None,
    mma_tiler_mn: Tuple[int, int] = DEFAULT_CTA_SHAPE,
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    sf_fp8_dtype_override: Optional[Literal["e5m3"]] = None,
    accumulate_on_output: bool = False,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Compile (cached) and execute the subchannel-scaled grouped GEMM wgrad; returns ``wgrad_tensor``.

    The compiled kernel is cached on the operand signatures with the token axis dynamic, so
    successive calls with different per-expert token counts reuse one compile. When
    ``wgrad_tensor``/``wgrad_ptrs`` are omitted the wrapper allocates the ``(expert_cnt, hidden,
    intermediate)`` bf16 output (zero-initialized when ``accumulate_on_output``)."""
    framework = detect_framework(a_tensor)
    if framework != "torch":
        raise ValueError(f"Unsupported tensor framework '{framework}' for grouped_gemm_wgrad_subchannel_scaled_wrapper_sm100; pass torch tensors")
    import torch

    acc_dtype = _convert_to_cutlass_data_type(acc_dtype) if acc_dtype is not None else cutlass.Float32
    wgrad_dtype = _convert_to_cutlass_data_type(wgrad_dtype) if wgrad_dtype is not None else cutlass.BFloat16
    if output_mode not in ("dense", "discrete"):
        raise ValueError(f'output_mode must be "dense" or "discrete", got {output_mode}')
    if len(get_shape(a_tensor)) != 2 or len(get_shape(b_tensor)) != 2:
        raise ValueError("a_tensor and b_tensor must both be rank-2")
    hidden, tokens_sum = get_shape(a_tensor)
    tokens_b, intermediate = get_shape(b_tensor)
    if tokens_sum != tokens_b:
        raise ValueError(f"a_tensor and b_tensor token dimensions must match, got {tokens_sum} and {tokens_b}")
    if len(get_shape(offsets_tensor)) != 1:
        raise ValueError(f"offsets_tensor must be rank-1, got shape {get_shape(offsets_tensor)}")
    expert_cnt = get_shape(offsets_tensor)[0]
    if output_mode == "dense" and wgrad_ptrs is not None:
        raise ValueError("dense output_mode forbids wgrad_ptrs")
    if wgrad_ptrs is not None:
        _validate_pointer_tensor(wgrad_ptrs, "wgrad_ptrs", expert_cnt)
    if wgrad_tensor is None and wgrad_ptrs is None:
        allocator = torch.zeros if accumulate_on_output else torch.empty
        with _torch_stream_context(current_stream, a_tensor.device):
            wgrad_tensor = allocator((expert_cnt, hidden, intermediate), dtype=framework_dtype(wgrad_dtype, "torch"), device=a_tensor.device)

    cache_key = (
        get_device_type(),
        output_mode,
        _wgrad_tensor_signature(a_tensor, dynamic_dims=(1,), exact_stride=False),
        _wgrad_tensor_signature(b_tensor, dynamic_dims=(0,), exact_stride=False),
        _wgrad_tensor_signature(sfa_tensor, dynamic_dims=(1,), exact_stride=False),
        _wgrad_tensor_signature(sfb_tensor, dynamic_dims=(1,), exact_stride=False),
        _wgrad_tensor_signature(sfa2_tensor, dynamic_dims=(1,), exact_stride=False),
        _wgrad_tensor_signature(offsets_tensor, exact_stride=True),
        _wgrad_tensor_signature(wgrad_tensor, exact_stride=True),
        _wgrad_tensor_signature(wgrad_ptrs, exact_stride=True),
        _wgrad_tensor_signature(global_scale_a, exact_stride=True),
        _wgrad_tensor_signature(global_scale_b, exact_stride=True),
        int(sgk),
        acc_dtype,
        wgrad_dtype,
        tuple(mma_tiler_mn),
        tuple(cluster_shape_mn) if cluster_shape_mn is not None else None,
        sf_fp8_dtype_override,
        accumulate_on_output,
        int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0")),
    )
    op = _cache_of_GroupedGemmWgradSubchannelScaledSm100Objects.get(cache_key)
    if op is None:
        common = dict(
            sample_a=a_tensor,
            sample_b=b_tensor,
            sample_sfa=sfa_tensor,
            sample_sfb=sfb_tensor,
            sample_sfa2=sfa2_tensor,
            sample_offsets=offsets_tensor,
            sgk=sgk,
            sample_global_scale_a=global_scale_a,
            sample_global_scale_b=global_scale_b,
            acc_dtype=acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sf_fp8_dtype_override=sf_fp8_dtype_override,
            accumulate_on_output=accumulate_on_output,
        )
        if output_mode == "dense":
            common["sample_wgrad"] = wgrad_tensor
        else:
            sample_wgrad_expert = (
                wgrad_tensor[0]
                if wgrad_tensor is not None
                else torch.empty((hidden, intermediate), dtype=framework_dtype(wgrad_dtype, "torch"), device=a_tensor.device)
            )
            common.update(
                sample_wgrad_expert=sample_wgrad_expert,
                num_experts=expert_cnt,
                wgrad_shape=(hidden, intermediate),
                wgrad_dtype=wgrad_dtype,
            )
        op = GroupedGemmWgradSubchannelScaledSm100(**common)
        if not op.check_support():
            raise RuntimeError("Unsupported configuration")
        op.compile()
        _cache_of_GroupedGemmWgradSubchannelScaledSm100Objects[cache_key] = op
    op.execute(
        a_tensor=a_tensor,
        b_tensor=b_tensor,
        sfa_tensor=sfa_tensor,
        sfb_tensor=sfb_tensor,
        sfa2_tensor=sfa2_tensor,
        offsets_tensor=offsets_tensor,
        wgrad_tensor=wgrad_tensor,
        wgrad_ptrs=wgrad_ptrs,
        global_scale_a=global_scale_a,
        global_scale_b=global_scale_b,
        current_stream=current_stream,
    )
    return TupleDict(wgrad_tensor=wgrad_tensor)


__all__ = ["GroupedGemmWgradSubchannelScaledSm100", "grouped_gemm_wgrad_subchannel_scaled_wrapper_sm100"]
