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
from cutlass.cute.runtime import from_dlpack, make_ptr

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.api_base import APIBase


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
        self.sample_amax = sample_amax
        if self.sample_amax.dim() < 3:
            self._logger.info(
                f"Reshaping sample_amax to (1, 1, 1) from {self.sample_amax.shape}"
            )
            for _ in range(3 - self.sample_amax.dim()):
                self.sample_amax = self.sample_amax.unsqueeze(-1)
        self.acc_dtype = acc_dtype
        self.mma_tiler_mn = mma_tiler_mn
        self.cluster_shape_mn = cluster_shape_mn
        self.sf_vec_size = sf_vec_size

        # used to reshape sfa/sfb tensors to atom layout
        self.atom_m = (32, 4)
        self.atom_k = 4

        self._logger.debug(
            f"__init__ completed with args: sample_a {sample_a.shape}, sample_b {sample_b.shape}, sample_sfa {sample_sfa.shape}, sample_sfb {sample_sfb.shape}, sample_c {sample_c.shape}, sample_amax {sample_amax.shape}, acc_dtype {acc_dtype}, mma_tiler_mn {mma_tiler_mn}, cluster_shape_mn {cluster_shape_mn}, sf_vec_size {sf_vec_size}"
        )

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        ab_dtype = self.sample_a.dtype
        sf_dtype = self.sample_sfa.dtype
        c_dtype = self.sample_c.dtype

        self._logger.debug("Checking dtypes and sf_vec_size")
        if self.sample_a.dtype != self.sample_b.dtype:
            raise ValueError(
                f"A and B tensor dtypes must match, got {self.sample_a.dtype} and {self.sample_b.dtype}"
            )
        if ab_dtype not in {
            torch.float4_e2m1fn_x2,
            torch.uint8,
            torch.float8_e5m2,
            torch.float8_e4m3fn,
        }:
            raise ValueError(
                f"Unsupported ab_dtype: received {ab_dtype}, expected {{float4_e2m1fn_x2, uint8, float8_e5m2, float8_e4m3fn}}"
            )
        if ab_dtype == torch.uint8:
            self._logger.warning(
                "Uint8 ab_dtype will be interpreted as packed fp4, not as native uint8"
            )
        if self.sf_vec_size not in {16, 32}:
            raise ValueError(
                f"Unsupported sf_vec_size: received {self.sf_vec_size}, expected {{16, 32}}"
            )
        if sf_dtype not in {
            torch.float8_e8m0fnu,
            torch.float8_e4m3fn,
            torch.int8,
        }:
            raise ValueError(
                f"Unsupported sf_dtype: received {sf_dtype}, expected {{float8_e8m0fnu, float8_e4m3fn, int8}}"
            )
        if sf_dtype == torch.int8:
            self._logger.warning(
                "Int8 sf_dtype will be interpreted as float8_e8m0fnu, not as native int8"
            )
        if sf_dtype == torch.float8_e4m3fn and self.sf_vec_size == 32:
            raise ValueError(
                "Unsupported sf_dtype and sf_vec_size combination: float8_e4m3fn and 32 is not supported"
            )
        if (
            ab_dtype in {torch.float8_e5m2, torch.float8_e4m3fn}
            and self.sf_vec_size == 16
        ):
            raise ValueError(
                f"Unsupported ab_dtype and sf_vec_size combination: {{float8_e5m2, float8_e4m3fn}} and 16 is not supported"
            )
        if c_dtype not in {
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float8_e5m2,
            torch.float8_e4m3fn,
            torch.float4_e2m1fn_x2,
            torch.uint8,
        }:
            raise ValueError(
                f"Unsupported c_dtype: received {c_dtype}, expected {{float32, float16, bfloat16, float8_e5m2, float8_e4m3fn, float4_e2m1fn_x2, uint8}}"
            )
        if c_dtype in {torch.float4_e2m1fn_x2, torch.uint8}:
            if ab_dtype not in {torch.float4_e2m1fn_x2, torch.uint8}:
                raise ValueError(
                    f"Unsupported c_dtype and ab_dtype combination: fp4 c_dtype requires fp4 ab_dtype, got {ab_dtype}"
                )  # Kernel fails to launch with other ab_dtype
        if c_dtype in {torch.float8_e5m2, torch.float8_e4m3fn} and ab_dtype in {
            torch.float8_e5m2,
            torch.float8_e4m3fn,
        }:
            raise NotImplementedError(
                f"fp8 ab_dtype and fp8 c_dtype currently fails to launch"
            )
        if not (self.acc_dtype == torch.float32):
            raise ValueError(
                f"Unsupported acc_dtype: received {self.acc_dtype}, expected {{float32}}"
            )
        self.ab_dtype = ab_dtype
        self.c_dtype = c_dtype

        self._logger.debug("Checking tensor layout")
        m, k, l = self.sample_a.shape
        n, k, l = self.sample_b.shape
        m_, n_, l = self.sample_c.shape
        _, _, m_div_atom_m0_m1, _, sf_k_div_atom_k, l = self.sample_sfa.shape
        _, _, n_div_atom_m0_m1, _, sf_k_div_atom_k, l = self.sample_sfb.shape
        _, _, _ = self.sample_amax.shape

        if self.sample_a.shape != (m, k, l):
            raise ValueError(
                f"Input/Output shape mismatch: expected A tensor shape {m, k, l}, got {self.sample_a.shape}"
            )
        if self.sample_b.shape != (n, k, l):
            raise ValueError(
                f"Input/Output shape mismatch: expected B tensor shape {n, k, l}, got {self.sample_b.shape}"
            )
        if c_dtype == torch.float4_e2m1fn_x2 or c_dtype == torch.uint8:
            if self.sample_c.shape != (
                m,
                (n + 1) // 2,
                l,
            ):
                raise ValueError(
                    f"Input/Output shape mismatch: expected C tensor shape {m, (n + 1) // 2, l}, got {self.sample_c.shape}"
                )
        else:
            if self.sample_c.shape != (m, n, l):
                raise ValueError(
                    f"Input/Output shape mismatch: expected C tensor shape {m, n, l}, got {self.sample_c.shape}"
                )
        if self.sample_sfa.shape != (
            self.atom_m[0],
            self.atom_m[1],
            m_div_atom_m0_m1,
            self.atom_k,
            sf_k_div_atom_k,
            l,
        ):
            raise ValueError(
                f"Input/Output shape mismatch: expected sfa tensor shape {self.atom_m[0], self.atom_m[1], m_div_atom_m0_m1, self.atom_k, sf_k_div_atom_k, l}, got {self.sample_sfa.shape}"
            )
        if self.sample_sfb.shape != (
            self.atom_m[0],
            self.atom_m[1],
            n_div_atom_m0_m1,
            self.atom_k,
            sf_k_div_atom_k,
            l,
        ):
            raise ValueError(
                f"Input/Output shape mismatch: expected sfb tensor shape {self.atom_m[0], self.atom_m[1], n_div_atom_m0_m1, self.atom_k, sf_k_div_atom_k, l}, got {self.sample_sfb.shape}"
            )
        if self.sample_amax.shape != (1, 1, 1):
            raise ValueError(
                f"Input/Output shape mismatch: expected amax tensor shape {1, 1, 1}, got {self.sample_amax.shape}"
            )
        if m_div_atom_m0_m1 != (m + self.atom_m[0] * self.atom_m[1] - 1) // (
            self.atom_m[0] * self.atom_m[1]
        ):
            raise ValueError(
                f"Input/Output shape mismatch: expected m_div_atom_m0_m1 (sfa.shape[2]) = {(m + self.atom_m[0] * self.atom_m[1] - 1) // (self.atom_m[0] * self.atom_m[1])}, got {m_div_atom_m0_m1}"
            )
        if n_div_atom_m0_m1 != (n + self.atom_m[0] * self.atom_m[1] - 1) // (
            self.atom_m[0] * self.atom_m[1]
        ):
            raise ValueError(
                f"Input/Output shape mismatch: expected n_div_atom_m0_m1 (sfb.shape[2]) = {(n + self.atom_m[0] * self.atom_m[1] - 1) // (self.atom_m[0] * self.atom_m[1])}, got {n_div_atom_m0_m1}"
            )
        if self.sample_a.stride() == (1, m, m * k):
            self.a_major = "m"
        elif self.sample_a.stride() == (k, 1, m * k):
            self.a_major = "k"
        else:
            raise ValueError(
                f"Unsupported A tensor stride: expected {{(1, m, m * k), (k, 1, m * k)}}, got {self.sample_a.stride()}"
            )
        if self.sample_b.stride() == (1, n, n * k):
            self.b_major = "n"
        elif self.sample_b.stride() == (k, 1, n * k):
            self.b_major = "k"
        else:
            raise ValueError(
                f"Unsupported B tensor stride: expected {{(1, n, n * k), (k, 1, n * k)}}, got {self.sample_b.stride()}"
            )
        if self.sample_c.stride() == (1, m_, m_ * n_):
            self.c_major = "m"
        elif self.sample_c.stride() == (n_, 1, m_ * n_):
            self.c_major = "n"
        else:
            raise ValueError(
                f"Unsupported C tensor stride: expected {{(1, m, m * n), (n, 1, m * n)}}, got {self.sample_c.stride()}"
            )

        if ab_dtype in {torch.float4_e2m1fn_x2, torch.uint8} and not (
            self.a_major == "k" and self.b_major == "k"
        ):
            raise ValueError(
                f"Unsupported A or B tensor stride: Float4 tensors require k-major layout for hardware efficiency, got {self.a_major} and {self.b_major}"
            )
        if c_dtype in {torch.float4_e2m1fn_x2, torch.uint8} and self.c_major == "m":
            raise ValueError(
                f"Unsupported C tensor stride: Float4 tensors require n-major layout for hardware efficiency, got {self.c_major}"
            )

        self._logger.debug("Checking mma tiler and cluster shape")
        if self.mma_tiler_mn[0] not in [128, 256]:
            raise ValueError(
                f"Unsupported mma_tiler_mn[0]: expected {{128, 256}}, got {self.mma_tiler_mn[0]}"
            )
        if self.mma_tiler_mn[1] not in [128, 256]:
            raise ValueError(
                f"Unsupported mma_tiler_mn[1]: expected {{128, 256}}, got {self.mma_tiler_mn[1]}"
            )
        if self.mma_tiler_mn[0] == 256:
            raise NotImplementedError("mma_tiler_mn[0] == 256 currently hangs")
        if (
            self.ab_dtype in {torch.float4_e2m1fn_x2, torch.uint8}
            and self.mma_tiler_mn[1] == 256
            and k <= 128
        ):
            raise ValueError(
                f"mma_tiler_mn (X, 256) requires k > 128 (packed x2), got {k}"
            )
        if not (
            self.cluster_shape_mn[0] % (2 if self.mma_tiler_mn[0] == 256 else 1) == 0
        ):
            raise ValueError("Illegal cluster shape")
        if (
            self.mma_tiler_mn == (128, 256)
            and self.sf_vec_size == 16
            and c_dtype in {torch.float32, torch.float16, torch.bfloat16}
        ):
            raise NotImplementedError(
                "mma_tiler_mn (128, 256), sf_vec_size 16, c_dtype {torch.float32, torch.float16, torch.bfloat16} fails to launch"
            )

        # Special cluster shape check for scale factor multicasts.
        # Due to limited size of scale factors, we can't multicast among more than 4 CTAs.
        def is_power_of_2(x):
            return x > 0 and (x & (x - 1)) == 0

        if not (
            self.cluster_shape_mn[0] <= 4
            and self.cluster_shape_mn[1] <= 4
            and self.cluster_shape_mn[0] > 0
            and self.cluster_shape_mn[1] > 0
            and is_power_of_2(self.cluster_shape_mn[0])
            and is_power_of_2(self.cluster_shape_mn[1])
        ):
            raise ValueError(
                f"Invalid cluster shape: expected cluster_shape_mn values in {{1, 2, 4}}, got {self.cluster_shape_mn}"
            )

        self._logger.debug("Checking tensor alignment")

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = (
                16 * 8 // (_convert_to_cutlass_data_type(dtype).width)
            )
            return num_major_elements % num_contiguous_elements == 0

        if not (
            check_contigous_16B_alignment(ab_dtype, self.a_major == "m", (m, k, l))
            and check_contigous_16B_alignment(ab_dtype, self.b_major == "n", (n, k, l))
            and check_contigous_16B_alignment(c_dtype, self.c_major == "m", (m, n, l))
        ):
            raise ValueError(
                "Unsupported tensor alignment: tensors must be 16B aligned"
            )

        self._logger.debug("Checking environment")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        compute_capability = major * 10 + minor
        if compute_capability < 100:
            raise RuntimeError(
                f"GemmAmax requires SM100+ compute capability, but found SM{compute_capability} on device {device}"
            )
        if compute_capability == 103:
            raise RuntimeError("cuteDSL GemmAmax is not supported on SM103")

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def compile(self, current_stream: Optional[cuda.CUstream] = None) -> None:
        self._logger.debug("Entering compile")
        current_stream = self._get_default_stream(current_stream)
        self._ensure_support_checked()

        is_ab_fp4 = self.ab_dtype in {torch.float4_e2m1fn_x2, torch.uint8}
        is_c_fp4 = self.c_dtype in {torch.float4_e2m1fn_x2, torch.uint8}
        torch_version = version.parse(torch.__version__)
        _fp8_dlpack_supported = version.parse(
            torch_version.base_version
        ) >= version.parse("2.10.0")
        use_no_dlpack_kernel = is_ab_fp4 or is_c_fp4 or not _fp8_dlpack_supported

        if use_no_dlpack_kernel:
            self._logger.debug(
                "Running no_dlpack kernel wrapper due to fp4 dtype or fp8 dtype on incompatible torch version"
            )
            self._kernel = Sm100BlockScaledPersistentDenseGemmKernelNoDlpack
        else:
            self._kernel = Sm100BlockScaledPersistentDenseGemmKernel

        gemm_amax = self._kernel(
            sf_vec_size=self.sf_vec_size,
            mma_tiler_mn=self.mma_tiler_mn,
            cluster_shape_mn=self.cluster_shape_mn,
        )
        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(
            self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        )

        if not use_no_dlpack_kernel:
            sample_a_cute = from_dlpack(self.sample_a, assumed_align=16)
            sample_b_cute = from_dlpack(self.sample_b, assumed_align=16)

            sample_c_cute = from_dlpack(self.sample_c, assumed_align=16)

            self._logger.debug("Compiling gemm_amax")
            self._compiled_kernel = cute.compile(
                gemm_amax,
                a_tensor=sample_a_cute,
                b_tensor=sample_b_cute,
                sfa_tensor=from_dlpack(self.sample_sfa, assumed_align=16),
                sfb_tensor=from_dlpack(self.sample_sfb, assumed_align=16),
                c_tensor=sample_c_cute,
                amax_tensor=from_dlpack(self.sample_amax, assumed_align=16),
                max_active_clusters=max_active_clusters,
                stream=current_stream,
            )
        else:  # use_no_dlpack
            # Create cute pointers/tensors manually to avoid DLPack requirements
            # amax is never fp4 or fp8 and is safe to use directly with dlpack
            self._logger.debug("Compiling gemm_amax (no dlpack)")
            a_ptr = make_ptr(
                (
                    cutlass.Float4E2M1FN
                    if is_ab_fp4
                    else _convert_to_cutlass_data_type(self.sample_a.dtype)
                ),
                self.sample_a.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=32 if is_ab_fp4 else 16,
            )
            b_ptr = make_ptr(
                (
                    cutlass.Float4E2M1FN
                    if is_ab_fp4
                    else _convert_to_cutlass_data_type(self.sample_b.dtype)
                ),
                self.sample_b.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=32 if is_ab_fp4 else 16,
            )
            c_ptr = make_ptr(
                (
                    cutlass.Float4E2M1FN
                    if is_c_fp4
                    else _convert_to_cutlass_data_type(self.sample_c.dtype)
                ),
                self.sample_c.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=32 if is_c_fp4 else 16,
            )
            sfa_ptr = make_ptr(
                (
                    cutlass.Float8E8M0FNU
                    if self.sample_sfa.dtype == torch.int8
                    else _convert_to_cutlass_data_type(self.sample_sfa.dtype)
                ),
                self.sample_sfa.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            sfb_ptr = make_ptr(
                (
                    cutlass.Float8E8M0FNU
                    if self.sample_sfb.dtype == torch.int8
                    else _convert_to_cutlass_data_type(self.sample_sfb.dtype)
                ),
                self.sample_sfb.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            a_shape = (
                tuple(
                    dim * 2 if i == 1 else dim
                    for i, dim in enumerate(self.sample_a.shape)
                )
                if is_ab_fp4
                else tuple(self.sample_a.shape)
            )
            b_shape = (
                tuple(
                    dim * 2 if i == 1 else dim
                    for i, dim in enumerate(self.sample_b.shape)
                )
                if is_ab_fp4
                else tuple(self.sample_b.shape)
            )
            c_shape = (
                tuple(
                    dim * 2 if i == 1 else dim
                    for i, dim in enumerate(self.sample_c.shape)
                )
                if is_c_fp4
                else tuple(self.sample_c.shape)
            )
            sfa_shape = tuple(self.sample_sfa.shape)
            sfb_shape = tuple(self.sample_sfb.shape)

            a_order = (1, 0, 2) if self.a_major == "k" else (0, 1, 2)
            b_order = (1, 0, 2) if self.b_major == "k" else (0, 1, 2)
            c_order = (1, 0, 2) if self.c_major == "n" else (0, 1, 2)
            _sfa_strides = self.sample_sfa.stride()
            _sfb_strides = self.sample_sfb.stride()
            sfa_order = tuple(
                sorted(range(len(sfa_shape)), key=lambda i: _sfa_strides[i])
            )
            sfb_order = tuple(
                sorted(range(len(sfb_shape)), key=lambda i: _sfb_strides[i])
            )

            self._compiled_kernel = cute.compile(
                gemm_amax,
                a_ptr=a_ptr,
                a_shape=a_shape,
                a_order=a_order,
                b_ptr=b_ptr,
                b_shape=b_shape,
                b_order=b_order,
                sfa_ptr=sfa_ptr,
                sfa_shape=sfa_shape,
                sfa_order=sfa_order,
                sfb_ptr=sfb_ptr,
                sfb_shape=sfb_shape,
                sfb_order=sfb_order,
                c_ptr=c_ptr,
                c_shape=c_shape,
                c_order=c_order,
                amax_cute=from_dlpack(self.sample_amax, assumed_align=16),
                max_active_clusters=max_active_clusters,
                stream=current_stream,
            )
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

        if amax_tensor.dim() < 3:
            self._logger.info(
                f"Reshaping amax_tensor to (1, 1, 1) from {amax_tensor.shape}"
            )
            for _ in range(3 - amax_tensor.dim()):
                amax_tensor = amax_tensor.unsqueeze(-1)

        is_ab_fp4 = self.ab_dtype in {torch.float4_e2m1fn_x2, torch.uint8}
        is_c_fp4 = self.c_dtype in {torch.float4_e2m1fn_x2, torch.uint8}
        torch_version = version.parse(torch.__version__)
        _fp8_dlpack_supported = version.parse(
            torch_version.base_version
        ) >= version.parse("2.10.0")
        use_no_dlpack_kernel = is_ab_fp4 or is_c_fp4 or not _fp8_dlpack_supported

        if not use_no_dlpack_kernel:
            a_tensor_cute = from_dlpack(a_tensor, assumed_align=16)
            b_tensor_cute = from_dlpack(b_tensor, assumed_align=16)
            c_tensor_cute = from_dlpack(c_tensor, assumed_align=16)

            if not skip_compile:
                if self._compiled_kernel is None:
                    raise RuntimeError(
                        "GemmAmaxSm100 kernel not compiled; call compile() first or use execute(skip_compile=True)"
                    )
                self._logger.debug("Executing with compiled kernel")
                self._compiled_kernel(
                    a_tensor=a_tensor_cute,
                    b_tensor=b_tensor_cute,
                    sfa_tensor=from_dlpack(sfa_tensor, assumed_align=16),
                    sfb_tensor=from_dlpack(sfb_tensor, assumed_align=16),
                    c_tensor=c_tensor_cute,
                    amax_tensor=from_dlpack(amax_tensor, assumed_align=16),
                    stream=current_stream,
                )
                self._logger.debug("Executed with compiled kernel successfully")
            else:
                self._logger.debug("Executing without compiled kernel (JIT)")
                gemm_amax = self._kernel(
                    sf_vec_size=self.sf_vec_size,
                    mma_tiler_mn=self.mma_tiler_mn,
                    cluster_shape_mn=self.cluster_shape_mn,
                )
                gemm_amax(
                    a_tensor=a_tensor_cute,
                    b_tensor=b_tensor_cute,
                    sfa_tensor=from_dlpack(sfa_tensor, assumed_align=16),
                    sfb_tensor=from_dlpack(sfb_tensor, assumed_align=16),
                    c_tensor=c_tensor_cute,
                    amax_tensor=from_dlpack(amax_tensor, assumed_align=16),
                    stream=current_stream,
                )
        else:  # use_no_dlpack
            a_ptr = make_ptr(
                (
                    cutlass.Float4E2M1FN
                    if is_ab_fp4
                    else _convert_to_cutlass_data_type(a_tensor.dtype)
                ),
                a_tensor.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=32 if is_ab_fp4 else 16,
            )
            b_ptr = make_ptr(
                (
                    cutlass.Float4E2M1FN
                    if is_ab_fp4
                    else _convert_to_cutlass_data_type(b_tensor.dtype)
                ),
                b_tensor.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=32 if is_ab_fp4 else 16,
            )
            c_ptr = make_ptr(
                (
                    cutlass.Float4E2M1FN
                    if is_c_fp4
                    else _convert_to_cutlass_data_type(c_tensor.dtype)
                ),
                c_tensor.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=32 if is_c_fp4 else 16,
            )
            sfa_ptr = make_ptr(
                (
                    cutlass.Float8E8M0FNU
                    if sfa_tensor.dtype == torch.int8
                    else _convert_to_cutlass_data_type(sfa_tensor.dtype)
                ),
                sfa_tensor.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            sfb_ptr = make_ptr(
                (
                    cutlass.Float8E8M0FNU
                    if sfb_tensor.dtype == torch.int8
                    else _convert_to_cutlass_data_type(sfb_tensor.dtype)
                ),
                sfb_tensor.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )

            if not skip_compile:
                if self._compiled_kernel is None:
                    raise RuntimeError(
                        "GemmAmaxSm100 kernel not compiled; call compile() first or use execute(skip_compile=True)"
                    )
                self._logger.debug("Executing with compiled kernel")
                self._compiled_kernel(
                    a_ptr=a_ptr,
                    b_ptr=b_ptr,
                    sfa_ptr=sfa_ptr,
                    sfb_ptr=sfb_ptr,
                    c_ptr=c_ptr,
                    amax_cute=from_dlpack(amax_tensor, assumed_align=16),
                    stream=current_stream,
                )
                self._logger.debug("Executed with compiled kernel successfully")
            else:
                self._logger.debug("Executing without compiled kernel (JIT)")
                gemm_amax = self._kernel(
                    sf_vec_size=self.sf_vec_size,
                    mma_tiler_mn=self.mma_tiler_mn,
                    cluster_shape_mn=self.cluster_shape_mn,
                )

                a_shape = (
                    tuple(
                        dim * 2 if i == 1 else dim
                        for i, dim in enumerate(self.sample_a.shape)
                    )
                    if is_ab_fp4
                    else tuple(self.sample_a.shape)
                )
                b_shape = (
                    tuple(
                        dim * 2 if i == 1 else dim
                        for i, dim in enumerate(self.sample_b.shape)
                    )
                    if is_ab_fp4
                    else tuple(self.sample_b.shape)
                )
                c_shape = (
                    tuple(
                        dim * 2 if i == 1 else dim
                        for i, dim in enumerate(self.sample_c.shape)
                    )
                    if is_c_fp4
                    else tuple(self.sample_c.shape)
                )
                sfa_shape = tuple(sfa_tensor.shape)
                sfb_shape = tuple(sfb_tensor.shape)
                a_order = (1, 0, 2) if self.a_major == "k" else (0, 1, 2)
                b_order = (1, 0, 2) if self.b_major == "k" else (0, 1, 2)
                c_order = (1, 0, 2) if self.c_major == "n" else (0, 1, 2)
                _sfa_strides = sfa_tensor.stride()
                _sfb_strides = sfb_tensor.stride()
                sfa_order = tuple(
                    sorted(range(len(sfa_shape)), key=lambda i: _sfa_strides[i])
                )
                sfb_order = tuple(
                    sorted(range(len(sfb_shape)), key=lambda i: _sfb_strides[i])
                )
                hardware_info = cutlass.utils.HardwareInfo()

                gemm_amax(
                    a_ptr=a_ptr,
                    a_shape=a_shape,
                    a_order=a_order,
                    b_ptr=b_ptr,
                    b_shape=b_shape,
                    b_order=b_order,
                    sfa_ptr=sfa_ptr,
                    sfa_shape=sfa_shape,
                    sfa_order=sfa_order,
                    sfb_ptr=sfb_ptr,
                    sfb_shape=sfb_shape,
                    sfb_order=sfb_order,
                    c_ptr=c_ptr,
                    c_shape=c_shape,
                    c_order=c_order,
                    amax_cute=from_dlpack(amax_tensor, assumed_align=16),
                    max_active_clusters=cutlass.utils.HardwareInfo().get_max_active_clusters(
                        self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
                    ),
                    stream=current_stream,
                )
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
        c_tensor = torch.empty_strided(
            (m, n, l), (1, m, m * n), dtype=c_dtype, device=a_tensor.device
        )
    elif c_major == "n":
        c_tensor = torch.empty_strided(
            (m, n, l), (n, 1, m * n), dtype=c_dtype, device=a_tensor.device
        )
    else:
        raise ValueError(f"c_major must be either 'm' or 'n', got {c_major}")
    amax_tensor = torch.full(
        (1, 1, 1), -float("inf"), device=a_tensor.device, dtype=torch.float32
    )

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
        _logger.debug(
            "gemm_amax_wrapper_sm100: Using previously cached GemmAmaxSm100 object"
        )
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
        _logger.debug(
            "gemm_amax_wrapper_sm100: No previously cached GemmAmaxSm100 object found, creating new GemmAmaxSm100 object"
        )
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
