from .dense_blockscaled_gemm_persistent_amax import (
    Sm100BlockScaledPersistentDenseGemmKernel,
    Sm100BlockScaledPersistentDenseGemmKernelNoDlpack,
)

from cuda.bindings import driver as cuda
import torch
from typing import Tuple, Optional
from packaging import version

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.api_base import APIBase, is_power_of_2, ceil_div


class GemmAmaxSm100(APIBase):
    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_b: torch.Tensor,
        sample_sfa: torch.Tensor,
        sample_sfb: torch.Tensor,
        sample_c: torch.Tensor,
        sample_amax: torch.Tensor,
        acc_dtype: torch.dtype = torch.float32,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        cluster_shape_mn: Tuple[int, int] = (1, 1),
        sf_vec_size: int = 32,
    ):
        super().__init__()

        self._logger.warning("GemmAmaxSm100 is an experimental API")
        self._logger.debug("Entering __init__")

        self.sample_a = sample_a
        self.sample_b = sample_b
        self.sample_sfa = sample_sfa
        self.sample_sfb = sample_sfb
        self.sample_c = sample_c
        self.sample_amax = self._pad_tensor_to_ndim(sample_amax, 3, "sample_amax")
        self.acc_dtype = acc_dtype
        self.mma_tiler_mn = mma_tiler_mn
        self.cluster_shape_mn = cluster_shape_mn
        self.sf_vec_size = sf_vec_size

        # used to reshape sfa/sfb tensors to atom layout
        self.atom_m = (32, 4)
        self.atom_k = 4

        self._interpret_uint8_as_fp4x2 = True
        self._logger.debug(
            f"__init__ completed with args: sample_a {sample_a.shape}, sample_b {sample_b.shape}, sample_sfa {sample_sfa.shape}, sample_sfb {sample_sfb.shape}, sample_c {sample_c.shape}, sample_amax {self.sample_amax.shape}, acc_dtype {acc_dtype}, mma_tiler_mn {mma_tiler_mn}, cluster_shape_mn {cluster_shape_mn}, sf_vec_size {sf_vec_size}"
        )

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        self._logger.debug("Checking dtypes and sf_vec_size")
        ab_dtype = self._check_dtype(
            self.sample_a,
            dtype=[torch.float4_e2m1fn_x2, torch.uint8, torch.float8_e5m2, torch.float8_e4m3fn],
            name="A",
        )
        self._check_dtype(
            self.sample_b,
            dtype=ab_dtype,
            name="B",
            extra_error_msg="A and B tensor dtypes must match",
        )
        if ab_dtype == torch.uint8:
            self._logger.warning("Uint8 ab_dtype will be interpreted as packed fp4, not as native uint8")

        self._value_error_if(
            self.sf_vec_size not in {16, 32},
            f"Unsupported sf_vec_size: received {self.sf_vec_size}, expected {{16, 32}}",
        )

        sf_dtype = self._check_dtype(
            self.sample_sfa,
            dtype=[torch.float8_e8m0fnu, torch.float8_e4m3fn, torch.int8],
            name="sfa",
        )
        self._check_dtype(
            self.sample_sfb,
            dtype=sf_dtype,
            name="sfb",
            extra_error_msg="sfa and sfb tensor dtypes must match",
        )
        if sf_dtype == torch.int8:
            self._logger.warning("Int8 sf_dtype will be interpreted as float8_e8m0fnu, not as native int8")

        self._value_error_if(
            sf_dtype == torch.float8_e4m3fn and self.sf_vec_size == 32,
            "Unsupported sf_dtype and sf_vec_size combination: float8_e4m3fn and 32 is not supported",
        )
        self._value_error_if(
            ab_dtype in {torch.float8_e5m2, torch.float8_e4m3fn} and self.sf_vec_size == 16,
            f"Unsupported ab_dtype and sf_vec_size combination: {{float8_e5m2, float8_e4m3fn}} and 16 is not supported",
        )

        c_dtype = self._check_dtype(
            self.sample_c,
            dtype=[torch.float32, torch.float16, torch.bfloat16, torch.float8_e5m2, torch.float8_e4m3fn, torch.float4_e2m1fn_x2, torch.uint8],
            name="C",
        )
        self._value_error_if(
            self._is_fp4x2(c_dtype) and not self._is_fp4x2(ab_dtype),
            f"Unsupported c_dtype and ab_dtype combination: fp4 c_dtype requires fp4 ab_dtype, got {ab_dtype}",
        )
        self._not_implemented_error_if(
            self._is_fp8(c_dtype) and self._is_fp8(ab_dtype),
            "Unsupported c_dtype and ab_dtype combination: fp8 ab_dtype and fp8 c_dtype (fails to launch)",
        )
        self._check_dtype(
            self.acc_dtype,
            dtype=torch.float32,
            name="Accumulator",
            extra_error_msg="Accumulator must be float32",
        )

        self.ab_dtype = ab_dtype
        self.c_dtype = c_dtype

        self._logger.debug("Checking tensor layout")
        m, k, l = self._tensor_shape(self.sample_a, name="sample_a")
        n, _, _ = self._tensor_shape(self.sample_b, name="sample_b")
        _, _, _ = self._tensor_shape(self.sample_c, name="sample_c")
        _, _, m_div_atom_m0_m1, _, sf_k_div_atom_k, _ = self.sample_sfa.shape
        _, _, n_div_atom_m0_m1, _, sf_k_div_atom_k, _ = self.sample_sfb.shape

        self._check_tensor_shape(self.sample_a, (m, k, l), "A")
        self._check_tensor_shape(self.sample_b, (n, k, l), "B")
        self._check_tensor_shape(self.sample_c, (m, n, l), "C")
        self._check_tensor_shape(
            self.sample_sfa,
            (self.atom_m[0], self.atom_m[1], m_div_atom_m0_m1, self.atom_k, sf_k_div_atom_k, l),
            "sfa",
        )
        self._check_tensor_shape(
            self.sample_sfb,
            (self.atom_m[0], self.atom_m[1], n_div_atom_m0_m1, self.atom_k, sf_k_div_atom_k, l),
            "sfb",
        )
        self._check_tensor_shape(self.sample_amax, (1, 1, 1), "amax")

        expected_m_div_atom = ceil_div(m, self.atom_m[0] * self.atom_m[1])
        expected_n_div_atom = ceil_div(n, self.atom_m[0] * self.atom_m[1])
        self._value_error_if(
            m_div_atom_m0_m1 != expected_m_div_atom,
            f"Input/Output shape mismatch: expected m_div_atom_m0_m1 (sfa.shape[2]) = {expected_m_div_atom}, got {m_div_atom_m0_m1}",
        )
        self._value_error_if(
            n_div_atom_m0_m1 != expected_n_div_atom,
            f"Input/Output shape mismatch: expected n_div_atom_m0_m1 (sfb.shape[2]) = {expected_n_div_atom}, got {n_div_atom_m0_m1}",
        )

        # Check tensor strides
        a_stride, self.a_stride_order = self._check_tensor_stride(
            self.sample_a,
            stride=[(1, m, m * k), (k, 1, m * k)],
            name="A",
        )
        b_stride, self.b_stride_order = self._check_tensor_stride(
            self.sample_b,
            stride=[(1, n, n * k), (k, 1, n * k)],
            name="B",
        )
        c_stride, self.c_stride_order = self._check_tensor_stride(
            self.sample_c,
            stride=[(1, m, m * n), (n, 1, m * n)],
            name="C",
        )

        # Derive major mode from stride order
        self.a_major = "m" if self.a_stride_order == (0, 1, 2) else "k"
        self.b_major = "n" if self.b_stride_order == (0, 1, 2) else "k"
        self.c_major = "m" if self.c_stride_order == (0, 1, 2) else "n"

        self._value_error_if(
            self._is_fp4x2(ab_dtype) and not (self.a_major == "k" and self.b_major == "k"),
            f"Unsupported A or B tensor stride: Float4 tensors require k-major layout for hardware efficiency, got {self.a_major} and {self.b_major}",
        )
        self._value_error_if(
            self._is_fp4x2(c_dtype) and self.c_major == "m",
            f"Unsupported C tensor stride: Float4 tensors require n-major layout for hardware efficiency, got {self.c_major}",
        )

        self._logger.debug("Checking mma tiler and cluster shape")
        self._value_error_if(
            self.mma_tiler_mn[0] not in [128, 256],
            f"Unsupported mma_tiler_mn[0]: expected {{128, 256}}, got {self.mma_tiler_mn[0]}",
        )
        self._value_error_if(
            self.mma_tiler_mn[1] not in [128, 256],
            f"Unsupported mma_tiler_mn[1]: expected {{128, 256}}, got {self.mma_tiler_mn[1]}",
        )
        self._not_implemented_error_if(
            self.mma_tiler_mn[0] == 256,
            "mma_tiler_mn[0] == 256 currently hangs",
        )
        self._value_error_if(
            self._is_fp4x2(self.ab_dtype) and self.mma_tiler_mn[1] == 256 and k <= 128,
            f"mma_tiler_mn (X, 256) requires k > 128 (packed x2), got {k}",
        )
        self._value_error_if(
            not (self.cluster_shape_mn[0] % (2 if self.mma_tiler_mn[0] == 256 else 1) == 0),
            "Illegal cluster shape",
        )
        self._not_implemented_error_if(
            self.mma_tiler_mn == (128, 256) and self.sf_vec_size == 16 and c_dtype in {torch.float32, torch.float16, torch.bfloat16},
            "mma_tiler_mn (128, 256), sf_vec_size 16, c_dtype {torch.float32, torch.float16, torch.bfloat16} fails to launch",
        )

        # Special cluster shape check for scale factor multicasts.
        # Due to limited size of scale factors, we can't multicast among more than 4 CTAs.
        self._value_error_if(
            not (
                self.cluster_shape_mn[0] <= 4
                and self.cluster_shape_mn[1] <= 4
                and self.cluster_shape_mn[0] > 0
                and self.cluster_shape_mn[1] > 0
                and is_power_of_2(self.cluster_shape_mn[0])
                and is_power_of_2(self.cluster_shape_mn[1])
            ),
            f"Invalid cluster shape: expected cluster_shape_mn values in {{1, 2, 4}}, got {self.cluster_shape_mn}",
        )

        self._logger.debug("Checking tensor alignment")

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // (_convert_to_cutlass_data_type(dtype).width)
            return num_major_elements % num_contiguous_elements == 0

        self._value_error_if(
            not (
                check_contigous_16B_alignment(ab_dtype, self.a_major == "m", (m, k, l))
                and check_contigous_16B_alignment(ab_dtype, self.b_major == "n", (n, k, l))
                and check_contigous_16B_alignment(c_dtype, self.c_major == "m", (m, n, l))
            ),
            "Unsupported tensor alignment: tensors must be 16B aligned",
        )

        self._logger.debug("Checking environment")
        self._runtime_error_if(not torch.cuda.is_available(), "CUDA is not available")
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        compute_capability = major * 10 + minor
        self._runtime_error_if(
            compute_capability < 100,
            f"GemmAmax requires SM100+ compute capability, but found SM{compute_capability} on device {device}",
        )
        self._runtime_error_if(
            compute_capability == 103,
            "cuteDSL GemmAmax is not supported on SM103",
        )

        is_ab_fp4 = self._is_fp4x2(self.ab_dtype)
        is_c_fp4 = self._is_fp4x2(self.c_dtype)
        is_ab_fp8 = self._is_fp8(self.ab_dtype)
        torch_version = version.parse(torch.__version__)
        _fp8_dlpack_supported = version.parse(torch_version.base_version) >= version.parse("2.10.0")
        use_no_dlpack_kernel = is_ab_fp4 or is_c_fp4 or (is_ab_fp8 and not _fp8_dlpack_supported)

        if use_no_dlpack_kernel:
            self._logger.debug("Running no_dlpack kernel wrapper due to fp4 dtype or fp8 dtype on incompatible torch version")
            self._kernel = Sm100BlockScaledPersistentDenseGemmKernelNoDlpack
        else:
            self._kernel = Sm100BlockScaledPersistentDenseGemmKernel

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def compile(self, current_stream: Optional[cuda.CUstream] = None) -> None:
        self._logger.debug("Entering compile")
        current_stream = self._get_default_stream(current_stream)
        self._ensure_support_checked()

        gemm_amax = self._kernel(
            sf_vec_size=self.sf_vec_size,
            mma_tiler_mn=self.mma_tiler_mn,
            cluster_shape_mn=self.cluster_shape_mn,
        )
        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(self.cluster_shape_mn[0] * self.cluster_shape_mn[1])

        if self._kernel is Sm100BlockScaledPersistentDenseGemmKernel:
            self._logger.debug("Compiling gemm_amax")
            self._compiled_kernel = cute.compile(
                gemm_amax,
                a_tensor=from_dlpack(self.sample_a, assumed_align=16),
                b_tensor=from_dlpack(self.sample_b, assumed_align=16),
                sfa_tensor=from_dlpack(self.sample_sfa, assumed_align=16),
                sfb_tensor=from_dlpack(self.sample_sfb, assumed_align=16),
                c_tensor=from_dlpack(self.sample_c, assumed_align=16),
                amax_tensor=from_dlpack(self.sample_amax, assumed_align=16),
                max_active_clusters=max_active_clusters,
                stream=current_stream,
            )
        elif self._kernel is Sm100BlockScaledPersistentDenseGemmKernelNoDlpack:
            # Create cute pointers/tensors manually to avoid DLPack requirements
            # amax is never fp4 or fp8 and is safe to use directly with dlpack
            self._logger.debug("Compiling gemm_amax (no dlpack)")

            is_ab_fp4 = self._is_fp4x2(self.ab_dtype)
            is_c_fp4 = self._is_fp4x2(self.c_dtype)
            a_ptr, a_shape, a_stride_order = self._make_cute_tensor_descriptor(self.sample_a, assumed_align=32 if is_ab_fp4 else 16, name="A")
            b_ptr, b_shape, b_stride_order = self._make_cute_tensor_descriptor(self.sample_b, assumed_align=32 if is_ab_fp4 else 16, name="B")
            c_ptr, c_shape, c_stride_order = self._make_cute_tensor_descriptor(self.sample_c, assumed_align=32 if is_c_fp4 else 16, name="C")
            sfa_ptr, sfa_shape, sfa_stride_order = self._make_cute_tensor_descriptor(self.sample_sfa, assumed_align=16, name="sfa")
            sfb_ptr, sfb_shape, sfb_stride_order = self._make_cute_tensor_descriptor(self.sample_sfb, assumed_align=16, name="sfb")

            self._compiled_kernel = cute.compile(
                gemm_amax,
                a_ptr=a_ptr,
                a_shape=a_shape,
                a_order=a_stride_order,
                b_ptr=b_ptr,
                b_shape=b_shape,
                b_order=b_stride_order,
                sfa_ptr=sfa_ptr,
                sfa_shape=sfa_shape,
                sfa_order=sfa_stride_order,
                sfb_ptr=sfb_ptr,
                sfb_shape=sfb_shape,
                sfb_order=sfb_stride_order,
                c_ptr=c_ptr,
                c_shape=c_shape,
                c_order=c_stride_order,
                amax_cute=from_dlpack(self.sample_amax, assumed_align=16),
                max_active_clusters=max_active_clusters,
                stream=current_stream,
            )
        else:
            raise NotImplementedError(f"Unreachable: invalid kernel type {self._kernel}")
        self._logger.debug("Kernel compiled successfully")

    def execute(
        self,
        a_tensor: torch.Tensor,
        b_tensor: torch.Tensor,
        sfa_tensor: torch.Tensor,
        sfb_tensor: torch.Tensor,
        c_tensor: torch.Tensor,
        amax_tensor: torch.Tensor,
        current_stream: Optional[cuda.CUstream] = None,
        skip_compile: bool = False,
    ) -> None:
        self._logger.debug("Entering execute")
        current_stream = self._get_default_stream(current_stream)

        amax_tensor = self._pad_tensor_to_ndim(amax_tensor, 3, "amax_tensor")

        is_ab_fp4 = self._is_fp4x2(self.ab_dtype)
        is_c_fp4 = self._is_fp4x2(self.c_dtype)

        if not skip_compile:
            self._runtime_error_if(
                self._compiled_kernel is None,
                "GemmAmaxSm100 kernel not compiled; call compile() first or use execute(skip_compile=True)",
            )
            self._logger.debug("Executing with compiled kernel")

            if self._kernel is Sm100BlockScaledPersistentDenseGemmKernel:
                self._compiled_kernel(
                    a_tensor=from_dlpack(a_tensor, assumed_align=16),
                    b_tensor=from_dlpack(b_tensor, assumed_align=16),
                    sfa_tensor=from_dlpack(sfa_tensor, assumed_align=16),
                    sfb_tensor=from_dlpack(sfb_tensor, assumed_align=16),
                    c_tensor=from_dlpack(c_tensor, assumed_align=16),
                    amax_tensor=from_dlpack(amax_tensor, assumed_align=16),
                    stream=current_stream,
                )
            elif self._kernel is Sm100BlockScaledPersistentDenseGemmKernelNoDlpack:
                a_ptr = self._make_cute_pointer(a_tensor, assumed_align=32 if is_ab_fp4 else 16)
                b_ptr = self._make_cute_pointer(b_tensor, assumed_align=32 if is_ab_fp4 else 16)
                c_ptr = self._make_cute_pointer(c_tensor, assumed_align=32 if is_c_fp4 else 16)
                sfa_ptr = self._make_cute_pointer(sfa_tensor, assumed_align=16)
                sfb_ptr = self._make_cute_pointer(sfb_tensor, assumed_align=16)

                self._compiled_kernel(
                    a_ptr=a_ptr,
                    b_ptr=b_ptr,
                    sfa_ptr=sfa_ptr,
                    sfb_ptr=sfb_ptr,
                    c_ptr=c_ptr,
                    amax_cute=from_dlpack(amax_tensor, assumed_align=16),
                    stream=current_stream,
                )
            else:
                raise NotImplementedError(f"Unreachable: invalid kernel type {self._kernel}")
            self._logger.debug("Executed with compiled kernel successfully")
        else:
            self._logger.debug("Executing without compiled kernel (JIT)")
            gemm_amax = self._kernel(
                sf_vec_size=self.sf_vec_size,
                mma_tiler_mn=self.mma_tiler_mn,
                cluster_shape_mn=self.cluster_shape_mn,
            )
            hardware_info = cutlass.utils.HardwareInfo()
            max_active_clusters = hardware_info.get_max_active_clusters(self.cluster_shape_mn[0] * self.cluster_shape_mn[1])

            if self._kernel is Sm100BlockScaledPersistentDenseGemmKernel:
                gemm_amax(
                    a_tensor=from_dlpack(a_tensor, assumed_align=16),
                    b_tensor=from_dlpack(b_tensor, assumed_align=16),
                    sfa_tensor=from_dlpack(sfa_tensor, assumed_align=16),
                    sfb_tensor=from_dlpack(sfb_tensor, assumed_align=16),
                    c_tensor=from_dlpack(c_tensor, assumed_align=16),
                    amax_tensor=from_dlpack(amax_tensor, assumed_align=16),
                    max_active_clusters=max_active_clusters,
                    stream=current_stream,
                )
            elif self._kernel is Sm100BlockScaledPersistentDenseGemmKernelNoDlpack:
                a_ptr, a_shape, a_stride_order = self._make_cute_tensor_descriptor(a_tensor, assumed_align=32 if is_ab_fp4 else 16, name="A")
                b_ptr, b_shape, b_stride_order = self._make_cute_tensor_descriptor(b_tensor, assumed_align=32 if is_ab_fp4 else 16, name="B")
                c_ptr, c_shape, c_stride_order = self._make_cute_tensor_descriptor(c_tensor, assumed_align=32 if is_c_fp4 else 16, name="C")
                sfa_ptr, sfa_shape, sfa_stride_order = self._make_cute_tensor_descriptor(sfa_tensor, assumed_align=16, name="sfa")
                sfb_ptr, sfb_shape, sfb_stride_order = self._make_cute_tensor_descriptor(sfb_tensor, assumed_align=16, name="sfb")

                gemm_amax(
                    a_ptr=a_ptr,
                    a_shape=a_shape,
                    a_order=a_stride_order,
                    b_ptr=b_ptr,
                    b_shape=b_shape,
                    b_order=b_stride_order,
                    sfa_ptr=sfa_ptr,
                    sfa_shape=sfa_shape,
                    sfa_order=sfa_stride_order,
                    sfb_ptr=sfb_ptr,
                    sfb_shape=sfb_shape,
                    sfb_order=sfb_stride_order,
                    c_ptr=c_ptr,
                    c_shape=c_shape,
                    c_order=c_stride_order,
                    amax_cute=from_dlpack(amax_tensor, assumed_align=16),
                    max_active_clusters=max_active_clusters,
                    stream=current_stream,
                )
            else:
                raise NotImplementedError(f"Unreachable: invalid kernel type {self._kernel}")
        self._logger.debug("Executed successfully")


import logging

_logger = logging.getLogger(__name__)
_cache_of_GemmAmaxSm100Objects = {}


def gemm_amax_wrapper_sm100(
    a_tensor: torch.Tensor,
    b_tensor: torch.Tensor,
    sfa_tensor: torch.Tensor,
    sfb_tensor: torch.Tensor,
    c_major: str = "n",
    c_dtype: torch.dtype = torch.float32,
    acc_dtype: torch.dtype = torch.float32,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    cluster_shape_mn: Tuple[int, int] = (1, 1),
    sf_vec_size: int = 32,
    stream: Optional[cuda.CUstream] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    _logger.debug("gemm_amax_wrapper_sm100: Creating empty output tensors c and amax")

    m, _, l = a_tensor.shape
    n, _, l = b_tensor.shape
    c_tensor = None
    if c_major == "m":
        c_tensor = torch.empty_strided((m, n, l), (1, m, m * n), dtype=c_dtype, device=a_tensor.device)
    elif c_major == "n":
        c_tensor = torch.empty_strided((m, n, l), (n, 1, m * n), dtype=c_dtype, device=a_tensor.device)
    else:
        raise ValueError(f"c_major must be either 'm' or 'n', got {c_major}")
    amax_tensor = torch.full((1, 1, 1), -float("inf"), device=a_tensor.device, dtype=torch.float32)

    cache_key = (
        a_tensor.shape,
        b_tensor.shape,
        sfa_tensor.shape,
        sfb_tensor.shape,
        a_tensor.dtype,
        b_tensor.dtype,
        sfa_tensor.dtype,
        sfb_tensor.dtype,
        a_tensor.stride(),
        b_tensor.stride(),
        sfa_tensor.stride(),
        sfb_tensor.stride(),
        c_major,
        c_dtype,
        acc_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        sf_vec_size,
    )
    if cache_key in _cache_of_GemmAmaxSm100Objects:
        _logger.debug("gemm_amax_wrapper_sm100: Using previously cached GemmAmaxSm100 object")
        gemm_amax = _cache_of_GemmAmaxSm100Objects[cache_key]
        gemm_amax.execute(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            sfa_tensor=sfa_tensor,
            sfb_tensor=sfb_tensor,
            c_tensor=c_tensor,
            amax_tensor=amax_tensor,
            current_stream=stream,
        )
    else:
        _logger.debug("gemm_amax_wrapper_sm100: No previously cached GemmAmaxSm100 object found, creating new GemmAmaxSm100 object")
        gemm_amax = GemmAmaxSm100(
            sample_a=a_tensor,
            sample_b=b_tensor,
            sample_sfa=sfa_tensor,
            sample_sfb=sfb_tensor,
            sample_c=c_tensor,
            sample_amax=amax_tensor,
            acc_dtype=acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sf_vec_size=sf_vec_size,
        )
        assert gemm_amax.check_support()
        gemm_amax.compile(current_stream=stream)
        gemm_amax.execute(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            sfa_tensor=sfa_tensor,
            sfb_tensor=sfb_tensor,
            c_tensor=c_tensor,
            amax_tensor=amax_tensor,
            current_stream=stream,
        )
        _cache_of_GemmAmaxSm100Objects[cache_key] = gemm_amax

    return c_tensor, amax_tensor
