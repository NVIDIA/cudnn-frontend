# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""FE API for grouped GEMM GLU + Hadamard forward fusion."""

import logging
import os
from typing import Optional, Tuple

from cuda.bindings import driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode
from cutlass.cute.runtime import from_dlpack, make_fake_stream

from cudnn.api_base import APIBase, TupleDict, ceil_div, is_power_of_2
from cudnn.datatypes import _convert_to_cutlass_data_type

from ..moe_utils import MoEWeightMode
from .hadamard_utils import HADAMARD_SIZE, hadamard_matrix
from .moe_blockscaled_grouped_gemm_glu_hadamard import BlockScaledMoEGroupedGemmGluHadamardKernel


def _reinterpret_raw_grouped_fp4_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.uint8:
        cute_tensor = from_dlpack(tensor, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=1)
        cute_tensor.element_type = cutlass.Float4E2M1FN
        return cute_tensor
    return tensor


class GroupedGemmGluHadamardSm100(APIBase):
    """Dense grouped GEMM GLU forward kernel with fused Hadamard transform."""

    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_b: torch.Tensor,
        sample_c: torch.Tensor,
        sample_d: torch.Tensor,
        sample_sfa: torch.Tensor,
        sample_sfb: torch.Tensor,
        sample_padded_offsets: torch.Tensor,
        sample_alpha: torch.Tensor,
        sample_prob: torch.Tensor,
        sample_amax: Optional[torch.Tensor] = None,
        sample_bias: Optional[torch.Tensor] = None,
        sample_hadamard: Optional[torch.Tensor] = None,
        acc_dtype: torch.dtype = torch.float32,
        mma_tiler_mn: Tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[Tuple[int, int]] = None,
        sf_vec_size: int = 16,
        vector_f32: bool = False,
        m_aligned: int = 256,
        act_func: str = "swiglu",
        use_dynamic_sched: bool = False,
    ):
        super().__init__()

        self._warn_experimental_api()
        self._interpret_uint8_as_fp4x2 = True
        self._sample_a_tensor = sample_a
        self._sample_b_tensor = sample_b

        self.a_desc = self._make_tensor_desc(sample_a, name="sample_a", interpret_uint8_as_fp4x2=False)
        self.b_desc = self._make_tensor_desc(sample_b, name="sample_b", interpret_uint8_as_fp4x2=False)
        self.c_desc = self._make_tensor_desc(sample_c, name="sample_c")
        self.d_desc = self._make_tensor_desc(sample_d, name="sample_d")
        self.sfa_desc = self._make_tensor_desc(sample_sfa, name="sample_sfa")
        self.sfb_desc = self._make_tensor_desc(sample_sfb, name="sample_sfb")
        self.padded_offsets_desc = self._make_tensor_desc(sample_padded_offsets, name="sample_padded_offsets")
        self.alpha_desc = self._make_tensor_desc(sample_alpha, name="sample_alpha")
        self.prob_desc = self._make_tensor_desc(sample_prob, name="sample_prob")
        self.bias_desc = self._make_tensor_desc(sample_bias, name="sample_bias")
        self.expert_cnt = self.padded_offsets_desc.shape[0]
        if sample_amax is None:
            sample_amax = torch.empty((self.expert_cnt, 1), dtype=torch.float32, device=sample_a.device)
        self.amax_desc = self._make_tensor_desc(sample_amax, name="sample_amax")
        if sample_hadamard is None:
            self.hadamard_tensor = self._make_hadamard_tensor(sample_a.device)
        else:
            self.hadamard_tensor = self._normalize_hadamard_tensor(
                sample_hadamard,
                device=sample_a.device,
                name="sample_hadamard",
            )

        self.acc_dtype = acc_dtype
        self.mma_tiler_mn = mma_tiler_mn
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn if cluster_shape_mn is not None else ((2, 1) if self.use_2cta_instrs else (1, 1))
        self.sf_vec_size = sf_vec_size
        self.vector_f32 = vector_f32
        self.m_aligned = m_aligned
        self.act_func = act_func
        self.use_dynamic_sched = use_dynamic_sched
        self.weight_mode = MoEWeightMode.DENSE
        self._kernel = BlockScaledMoEGroupedGemmGluHadamardKernel
        self.num_cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
        self._workspace = None

    @staticmethod
    def _make_hadamard_tensor(device: torch.device) -> torch.Tensor:
        return hadamard_matrix(HADAMARD_SIZE, dtype=torch.bfloat16, device=device).t().contiguous()

    @classmethod
    def _normalize_hadamard_tensor(
        cls,
        hadamard_tensor: torch.Tensor,
        *,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        expected_shape = (HADAMARD_SIZE, HADAMARD_SIZE)
        if tuple(hadamard_tensor.shape) != expected_shape:
            raise ValueError(f"{name} tensor shape mismatch: expected {expected_shape}, got {tuple(hadamard_tensor.shape)}")
        if hadamard_tensor.dtype != torch.bfloat16 or hadamard_tensor.device != device:
            hadamard_tensor = hadamard_tensor.to(device=device, dtype=torch.bfloat16)
        if not hadamard_tensor.is_contiguous():
            hadamard_tensor = hadamard_tensor.contiguous()
        return hadamard_tensor

    def check_support(self) -> bool:
        tensor_m, k, _ = self._tensor_shape(self.a_desc, name="sample_a")
        n, _, l = self._tensor_shape(self.b_desc, name="sample_b")
        _, n_c, _ = self._tensor_shape(self.c_desc, name="sample_c")
        _, n_d, _ = self._tensor_shape(self.d_desc, name="sample_d")

        self._value_error_if(l != self.expert_cnt, f"B L dimension ({l}) must match expert_cnt ({self.expert_cnt})")
        self._value_error_if(n % 64 != 0, f"N must be divisible by 64, got {n}")
        self._value_error_if((n // 2) % HADAMARD_SIZE != 0, f"N/2 must be divisible by {HADAMARD_SIZE}, got {n // 2}")

        self._check_tensor_shape(self.a_desc, (tensor_m, k, 1), "A")
        self._check_tensor_shape(self.b_desc, (n, k, l), "B")
        self._check_tensor_shape(self.c_desc, (tensor_m, n, 1), "C")
        self._check_tensor_shape(self.d_desc, (tensor_m, n // 2, 1), "D")
        self._check_tensor_shape(self.sfa_desc, (32, 4, ceil_div(tensor_m, 128), 4, ceil_div(ceil_div(k, self.sf_vec_size), 4), 1), "SFA")
        self._check_tensor_shape(self.sfb_desc, (32, 4, ceil_div(n, 128), 4, ceil_div(ceil_div(k, self.sf_vec_size), 4), l), "SFB")
        self._check_tensor_shape(self.padded_offsets_desc, (l,), "padded_offsets")
        self._check_tensor_shape(self.alpha_desc, (l,), "alpha")
        self._check_tensor_shape(self.prob_desc, (tensor_m, 1, 1), "prob")
        self._check_tensor_shape(self.bias_desc, (n, l), "bias")
        self._check_tensor_shape(self.amax_desc, (l, 1), "amax")
        self._check_tensor_shape(self.hadamard_tensor, (HADAMARD_SIZE, HADAMARD_SIZE), "hadamard")

        self._check_tensor_stride(self.a_desc, stride=[(k, 1, tensor_m * k)], name="A", extra_error_msg="A must have k-major layout")
        self._check_tensor_stride(self.b_desc, stride=[(k, 1, n * k)], name="B", extra_error_msg="B must have k-major layout")
        self._check_tensor_stride(self.c_desc, stride=[(n_c, 1, tensor_m * n_c)], name="C", extra_error_msg="C must have n-major layout")
        self._check_tensor_stride(self.d_desc, stride=[(n_d, 1, tensor_m * n_d)], name="D", extra_error_msg="D must have n-major layout")
        self._check_tensor_stride(self.bias_desc, stride=[(1, n)], name="bias")

        self.ab_dtype = self._check_dtype(
            self.a_desc,
            dtype=[torch.float4_e2m1fn_x2, torch.uint8],
            name="A",
        )
        self._check_dtype(self.b_desc, dtype=self.ab_dtype, name="B", extra_error_msg="B must match A dtype")
        self.sf_dtype = self._check_dtype(self.sfa_desc, dtype=[torch.float8_e8m0fnu, torch.float8_e4m3fn], name="SFA")
        self._check_dtype(self.sfb_desc, dtype=self.sf_dtype, name="SFB", extra_error_msg="SFB must match SFA dtype")
        self.c_dtype = self._check_dtype(self.c_desc, dtype=[torch.float16, torch.bfloat16], name="C")
        self.d_dtype = self._check_dtype(self.d_desc, dtype=[torch.float16, torch.bfloat16], name="D")
        self._check_dtype(self.alpha_desc, dtype=torch.float32, name="alpha")
        self._check_dtype(self.prob_desc, dtype=torch.float32, name="prob")
        self._check_dtype(self.bias_desc, dtype=[torch.float16, torch.bfloat16, torch.float32], name="bias")
        self._check_dtype(self.amax_desc, dtype=torch.float32, name="amax")
        self._check_dtype(self.hadamard_tensor, dtype=torch.bfloat16, name="hadamard")
        self._check_dtype(self.acc_dtype, dtype=torch.float32, name="acc_dtype")

        self._value_error_if(self.sf_vec_size not in [16, 32], f"sf_vec_size must be 16 or 32, got {self.sf_vec_size}")
        self._value_error_if(self.act_func not in ["swiglu", "geglu"], f"act_func must be 'swiglu' or 'geglu', got {self.act_func}")
        self._value_error_if(
            not self.use_2cta_instrs or self.mma_tiler_mn != (256, 256), f"Hadamard fusion requires mma_tiler_mn=(256, 256), got {self.mma_tiler_mn}"
        )
        self._value_error_if(self.cluster_shape_mn[0] % 2 != 0, f"cluster_shape_mn[0] must be divisible by 2, got {self.cluster_shape_mn[0]}")
        self._value_error_if(
            not (
                self.cluster_shape_mn[0] * self.cluster_shape_mn[1] <= 16
                and self.cluster_shape_mn[0] > 0
                and self.cluster_shape_mn[1] > 0
                and self.cluster_shape_mn[0] <= 4
                and self.cluster_shape_mn[1] <= 4
                and is_power_of_2(self.cluster_shape_mn[0])
                and is_power_of_2(self.cluster_shape_mn[1])
            ),
            f"Invalid cluster shape: {self.cluster_shape_mn}",
        )
        self._value_error_if(
            self.m_aligned != BlockScaledMoEGroupedGemmGluHadamardKernel.FIX_PAD_SIZE,
            f"m_aligned must be {BlockScaledMoEGroupedGemmGluHadamardKernel.FIX_PAD_SIZE}, got {self.m_aligned}",
        )
        self._value_error_if(self.expert_cnt > 1024, f"expert_cnt must be <= 1024, got {self.expert_cnt}")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        compute_capability = major * 10 + minor
        if compute_capability < 100:
            raise RuntimeError(f"GroupedGemmGluHadamardSm100 requires SM100+, found SM{compute_capability}")

        if not self._kernel.can_implement(
            _convert_to_cutlass_data_type(self.ab_dtype, interpret_uint8_as_fp4x2=self._interpret_uint8_as_fp4x2),
            _convert_to_cutlass_data_type(self.sf_dtype),
            self.sf_vec_size,
            _convert_to_cutlass_data_type(self.acc_dtype),
            _convert_to_cutlass_data_type(self.d_dtype),
            self.use_2cta_instrs,
            self.mma_tiler_mn,
            self.cluster_shape_mn,
            self.m_aligned,
            n,
            k,
            l,
            "k",
            "k",
            "n",
            self.m_aligned,
        ):
            raise RuntimeError("Unsupported grouped GEMM GLU hadamard configuration")

        self._is_supported = True
        return True

    def compile(self) -> None:
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        if self.a_desc.shape[0] == 0:
            return

        kernel = self._kernel(
            sf_vec_size=self.sf_vec_size,
            acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
            use_2cta_instrs=self.use_2cta_instrs,
            mma_tiler_mn=self.mma_tiler_mn,
            cluster_shape_mn=self.cluster_shape_mn,
            vectorized_f32=self.vector_f32,
            expert_cnt=self.expert_cnt,
            weight_mode=MoEWeightMode.DENSE,
            use_dynamic_sched=self.use_dynamic_sched,
            act_func=self.act_func,
            enable_bias=self.bias_desc is not None,
        )

        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(self.cluster_shape_mn[0] * self.cluster_shape_mn[1])
        max_active_clusters -= self.num_cluster_overlap_margin
        self._value_error_if(max_active_clusters <= 0, "max_active_clusters must be > 0 after overlap margin")
        self._workspace = torch.empty(max(kernel.get_workspace_bytes(), 1), dtype=torch.uint8, device="cuda")
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
        fake_workspace_ptr = cute.runtime.nullptr(dtype=cutlass.Uint8, assumed_align=128)

        valid_m = cute.sym_int(divisibility=self.m_aligned)
        tensor_m_128 = cute.sym_int()
        stride_tensor_m_128 = cute.sym_int(divisibility=32 * 4 * 4)

        a_cute_fake = self._make_fake_cute_compact_tensor(
            dtype=self.a_desc.dtype,
            shape=(valid_m, self.a_desc.shape[1], 1),
            stride_order=self.a_desc.stride_order,
            dynamic_mode=self.a_desc.stride_order[0],
            divisibility=32 if self._is_fp4x2(self.ab_dtype) else 16,
        )
        c_cute_fake = self._make_fake_cute_compact_tensor(
            dtype=self.c_desc.dtype,
            shape=(valid_m, self.c_desc.shape[1], 1),
            stride_order=self.c_desc.stride_order,
            dynamic_mode=self.c_desc.stride_order[0],
            divisibility=8 if self._is_f16(self.c_desc) else 16,
        )
        d_cute_fake = self._make_fake_cute_compact_tensor(
            dtype=self.d_desc.dtype,
            shape=(valid_m, self.d_desc.shape[1], 1),
            stride_order=self.d_desc.stride_order,
            dynamic_mode=self.d_desc.stride_order[0],
            divisibility=8 if self._is_f16(self.d_desc) else 16,
        )
        prob_cute_fake = self._make_fake_cute_tensor(
            dtype=self.prob_desc.dtype,
            shape=(valid_m, 1, 1),
            stride=self.prob_desc.stride,
        )
        sfa_cute_fake = self._make_fake_cute_tensor(
            dtype=self.sfa_desc.dtype,
            shape=(32, 4, tensor_m_128, 4, self.sfa_desc.shape[4], 1),
            stride=(16, 4, self.sfa_desc.stride[2], 1, 512, stride_tensor_m_128),
        )
        b_cute_fake = self._make_fake_cute_tensor_from_desc(self.b_desc, assumed_align=16)
        sfb_cute_fake = self._make_fake_cute_tensor_from_desc(self.sfb_desc, assumed_align=16)
        alpha_cute_fake = self._make_fake_cute_tensor_from_desc(self.alpha_desc, assumed_align=16)
        padded_offsets_cute_fake = self._make_fake_cute_tensor_from_desc(self.padded_offsets_desc, assumed_align=16)
        amax_cute_fake = self._make_fake_cute_tensor_from_desc(self.amax_desc, assumed_align=16)
        bias_cute_fake = self._make_fake_cute_tensor_from_desc(self.bias_desc, assumed_align=16)
        hadamard_cute_fake = self._make_fake_cute_tensor_like(self.hadamard_tensor, assumed_align=16, name="sample_hadamard")
        cached_linear_offset = cutlass.Float32(1.0 if self.act_func == "geglu" else 0.0)

        compiled_kernel = cute.compile(
            kernel,
            _reinterpret_raw_grouped_fp4_tensor(self._sample_a_tensor) if self.a_desc.dtype == torch.uint8 else a_cute_fake,
            _reinterpret_raw_grouped_fp4_tensor(self._sample_b_tensor) if self.b_desc.dtype == torch.uint8 else b_cute_fake,
            sfa_cute_fake,
            sfb_cute_fake,
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int64(0),
            OperandMajorMode.K,
            fake_workspace_ptr,
            c_cute_fake,
            d_cute_fake,
            amax_cute_fake,
            padded_offsets_cute_fake,
            alpha_cute_fake,
            prob_cute_fake,
            hadamard_cute_fake,
            bias_cute_fake,
            max_active_clusters,
            fake_stream,
            cached_linear_offset,
            options="--enable-tvm-ffi",
        )

        cached_workspace_ptr = from_dlpack(self._workspace, assumed_align=128).iterator

        def tensor_api(
            a_tensor: torch.Tensor,
            b_tensor: torch.Tensor,
            c_tensor: torch.Tensor,
            d_tensor: torch.Tensor,
            sfa_tensor: torch.Tensor,
            sfb_tensor: torch.Tensor,
            padded_offsets: torch.Tensor,
            alpha_tensor: torch.Tensor,
            prob_tensor: torch.Tensor,
            hadamard_tensor: torch.Tensor,
            amax_tensor: Optional[torch.Tensor],
            bias_tensor: Optional[torch.Tensor],
            stream: cuda.CUstream,
        ) -> None:
            compiled_kernel(
                a_tensor,
                b_tensor,
                sfa_tensor,
                sfb_tensor,
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int64(0),
                cached_workspace_ptr,
                c_tensor,
                d_tensor,
                amax_tensor,
                padded_offsets,
                alpha_tensor,
                prob_tensor,
                hadamard_tensor,
                bias_tensor,
                stream,
                cached_linear_offset,
            )

        self._compiled_kernel = tensor_api

    def execute(
        self,
        a_tensor: torch.Tensor,
        b_tensor: torch.Tensor,
        c_tensor: torch.Tensor,
        d_tensor: torch.Tensor,
        sfa_tensor: torch.Tensor,
        sfb_tensor: torch.Tensor,
        padded_offsets: torch.Tensor,
        alpha_tensor: torch.Tensor,
        prob_tensor: torch.Tensor,
        hadamard_tensor: Optional[torch.Tensor] = None,
        amax_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        self._ensure_support_checked()
        if self._compiled_kernel is None:
            raise RuntimeError("Kernel has not been compiled")
        if a_tensor.shape[0] == 0:
            return
        if current_stream is None:
            current_stream = cuda.CUstream(torch.cuda.current_stream(a_tensor.device).cuda_stream)
        if hadamard_tensor is None:
            hadamard_tensor = self.hadamard_tensor
        else:
            hadamard_tensor = self._normalize_hadamard_tensor(
                hadamard_tensor,
                device=a_tensor.device,
                name="hadamard",
            )

        self._compiled_kernel(
            _reinterpret_raw_grouped_fp4_tensor(a_tensor),
            _reinterpret_raw_grouped_fp4_tensor(b_tensor),
            c_tensor,
            d_tensor,
            sfa_tensor,
            sfb_tensor,
            padded_offsets,
            alpha_tensor,
            prob_tensor,
            hadamard_tensor,
            amax_tensor,
            bias_tensor,
            current_stream,
        )


_logger = logging.getLogger(__name__)
_cache_of_GroupedGemmGluHadamardSm100Objects = {}


def grouped_gemm_glu_hadamard_wrapper_sm100(
    a_tensor: torch.Tensor,
    b_tensor: torch.Tensor,
    sfa_tensor: torch.Tensor,
    sfb_tensor: torch.Tensor,
    padded_offsets: torch.Tensor,
    alpha_tensor: torch.Tensor,
    prob_tensor: torch.Tensor,
    bias_tensor: Optional[torch.Tensor] = None,
    acc_dtype: torch.dtype = torch.float32,
    c_dtype: torch.dtype = torch.bfloat16,
    d_dtype: torch.dtype = torch.bfloat16,
    cd_major: str = "n",
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    sf_vec_size: int = 16,
    vector_f32: bool = False,
    m_aligned: int = 256,
    act_func: str = "swiglu",
    use_dynamic_sched: bool = False,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """High-level wrapper for grouped GEMM GLU + Hadamard forward fusion."""

    valid_m = a_tensor.shape[0]
    n_full, _, l = b_tensor.shape
    n_out = n_full // 2

    if cd_major != "n":
        raise ValueError(f"cd_major must be 'n', got {cd_major}")

    c_tensor = torch.empty_strided((valid_m, n_full, 1), (n_full, 1, valid_m * n_full), dtype=c_dtype, device=a_tensor.device)
    d_tensor = torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=d_dtype, device=a_tensor.device)
    amax_tensor = None
    if d_dtype in [torch.bfloat16, torch.float16]:
        amax_tensor = torch.full((l, 1), float("-inf"), dtype=torch.float32, device=a_tensor.device)

    if valid_m == 0:
        return TupleDict(c_tensor=c_tensor, d_tensor=d_tensor, amax_tensor=amax_tensor)

    def stride_order(tensor: torch.Tensor) -> Tuple[int, ...]:
        return tuple(i for i, _ in sorted(enumerate(tensor.stride()), key=lambda item: item[1]))

    def tensor_signature(tensor: Optional[torch.Tensor]) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        return tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype

    def dynamic_m_tensor_signature(
        tensor: Optional[torch.Tensor], static_shape_suffix: Optional[Tuple[int, ...]], dynamic_stride_dims: Tuple[int, ...] = ()
    ) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        stride_signature = tuple(None if idx in dynamic_stride_dims else value for idx, value in enumerate(tensor.stride()))
        return static_shape_suffix, stride_signature, tensor.dtype

    cache_key = (
        act_func,
        a_tensor.shape[1:],
        tuple(b_tensor.shape),
        c_tensor.shape[1:],
        a_tensor.dtype,
        b_tensor.dtype,
        c_tensor.dtype,
        d_tensor.dtype,
        stride_order(a_tensor),
        stride_order(b_tensor),
        stride_order(c_tensor),
        *dynamic_m_tensor_signature(sfa_tensor, (sfa_tensor.shape[4], 1), dynamic_stride_dims=(5,)),
        *tensor_signature(sfb_tensor),
        *tensor_signature(alpha_tensor),
        *dynamic_m_tensor_signature(prob_tensor, (1, 1)),
        *tensor_signature(bias_tensor),
        *tensor_signature(padded_offsets),
        acc_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        sf_vec_size,
        vector_f32,
        m_aligned,
        use_dynamic_sched,
    )

    if cache_key in _cache_of_GroupedGemmGluHadamardSm100Objects:
        api = _cache_of_GroupedGemmGluHadamardSm100Objects[cache_key]
    else:
        api = GroupedGemmGluHadamardSm100(
            sample_a=a_tensor,
            sample_b=b_tensor,
            sample_c=c_tensor,
            sample_d=d_tensor,
            sample_sfa=sfa_tensor,
            sample_sfb=sfb_tensor,
            sample_padded_offsets=padded_offsets,
            sample_alpha=alpha_tensor,
            sample_prob=prob_tensor,
            sample_amax=amax_tensor,
            sample_bias=bias_tensor,
            acc_dtype=acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sf_vec_size=sf_vec_size,
            vector_f32=vector_f32,
            m_aligned=m_aligned,
            act_func=act_func,
            use_dynamic_sched=use_dynamic_sched,
        )
        api.check_support()
        api.compile()
        _cache_of_GroupedGemmGluHadamardSm100Objects[cache_key] = api

    api.execute(
        a_tensor=a_tensor,
        b_tensor=b_tensor,
        c_tensor=c_tensor,
        d_tensor=d_tensor,
        sfa_tensor=sfa_tensor,
        sfb_tensor=sfb_tensor,
        padded_offsets=padded_offsets,
        alpha_tensor=alpha_tensor,
        prob_tensor=prob_tensor,
        amax_tensor=amax_tensor,
        bias_tensor=bias_tensor,
        current_stream=current_stream,
    )
    return TupleDict(c_tensor=c_tensor, d_tensor=d_tensor, amax_tensor=amax_tensor)
