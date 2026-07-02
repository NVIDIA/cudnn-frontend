from .dense_blockscaled_gemm_persistent_amax import (
    Sm100BlockScaledPersistentDenseGemmKernel,
)

from cuda.bindings import driver as cuda
import logging
import os
import torch
from typing import Tuple, Optional

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_stream

from cudnn.api_base import ApiBaseTorch, TupleDict
from cudnn.gemm_validation import resolve_max_active_clusters

from .validation import validate_gemm_amax


class GemmAmaxSm100(ApiBaseTorch):
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
        self._interpret_uint8_as_fp4x2 = True

        self._warn_experimental_api()
        self._logger.debug("Entering __init__")
        self._kernel = Sm100BlockScaledPersistentDenseGemmKernel

        self.a_desc = self._make_tensor_desc(sample_a, name="sample_a")
        self.b_desc = self._make_tensor_desc(sample_b, name="sample_b")
        self.sfa_desc = self._make_tensor_desc(sample_sfa, name="sample_sfa")
        self.sfb_desc = self._make_tensor_desc(sample_sfb, name="sample_sfb")
        self.c_desc = self._make_tensor_desc(sample_c, name="sample_c")
        self.amax_desc = self._make_tensor_desc(sample_amax, name="sample_amax")

        self.acc_dtype = acc_dtype
        self.mma_tiler_mn = mma_tiler_mn
        self.cluster_shape_mn = cluster_shape_mn
        self.sf_vec_size = sf_vec_size

        self.num_cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
        self._logger.debug(f"setting num_cluster_overlap_margin: {self.num_cluster_overlap_margin}")

        self._logger.debug(
            f"__init__ completed with args: sample_a {self.a_desc.shape}, sample_b {self.b_desc.shape}, sample_sfa {self.sfa_desc.shape}, sample_sfb {self.sfb_desc.shape}, sample_c {self.c_desc.shape}, sample_amax {self.amax_desc.shape}, acc_dtype {acc_dtype}, mma_tiler_mn {mma_tiler_mn}, cluster_shape_mn {cluster_shape_mn}, sf_vec_size {sf_vec_size}"
        )

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        self._logger.debug("Resolving kernel configuration")
        self.amax_desc = self._pad_tensor_to_ndim(self.amax_desc, 3, "amax")
        self.mma_tiler_mn = self._kernel.require_mma_tiler(self.mma_tiler_mn)
        self.cluster_shape_mn = self._kernel.require_cluster_shape(
            self.cluster_shape_mn,
            mma_tiler_mn=self.mma_tiler_mn,
        )

        self._logger.debug("Checking shared tensor and configuration contract")
        plan = validate_gemm_amax(
            self.a_desc,
            self.b_desc,
            self.sfa_desc,
            self.sfb_desc,
            self.c_desc,
            self.amax_desc,
            acc_dtype=self.acc_dtype,
            sf_vec_size=self.sf_vec_size,
            supported_sf_vec_sizes=self._kernel.SF_VEC_SIZES,
            mma_tiler_mn=self.mma_tiler_mn,
        )
        self.ab_dtype = self.a_desc.dtype
        self.c_dtype = self.c_desc.dtype
        self.a_major = plan.a_major
        self.b_major = plan.b_major
        self.c_major = plan.c_major

        if self.a_desc.dtype == torch.uint8:
            self._logger.warning("Uint8 ab_dtype will be interpreted as packed fp4, not as native uint8")

        self._logger.debug("Checking environment")
        self._runtime_error_if(not torch.cuda.is_available(), "CUDA is not available")
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        compute_capability = major * 10 + minor
        self._runtime_error_if(
            compute_capability < 100,
            f"GemmAmax requires SM100+ compute capability, but found SM{compute_capability} on device {device}",
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

        gemm_amax = self._kernel(
            sf_vec_size=self.sf_vec_size,
            mma_tiler_mn=self.mma_tiler_mn,
            cluster_shape_mn=self.cluster_shape_mn,
        )
        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = resolve_max_active_clusters(
            hardware_info.get_max_active_clusters(self.cluster_shape_mn[0] * self.cluster_shape_mn[1]),
            self.num_cluster_overlap_margin,
        )

        self._logger.debug("Compiling gemm_amax")
        a_cute = self._make_fake_cute_tensor_from_desc(self.a_desc, assumed_align=16)
        b_cute = self._make_fake_cute_tensor_from_desc(self.b_desc, assumed_align=16)
        sfa_cute = self._make_fake_cute_tensor_from_desc(self.sfa_desc, assumed_align=16)
        sfb_cute = self._make_fake_cute_tensor_from_desc(self.sfb_desc, assumed_align=16)
        c_cute = self._make_fake_cute_tensor_from_desc(self.c_desc, assumed_align=16)
        amax_cute = self._make_fake_cute_tensor_from_desc(self.amax_desc, assumed_align=16)
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)

        _compiled_kernel = cute.compile(
            gemm_amax,
            a_tensor=a_cute,
            b_tensor=b_cute,
            sfa_tensor=sfa_cute,
            sfb_tensor=sfb_cute,
            c_tensor=c_cute,
            amax_tensor=amax_cute,
            max_active_clusters=max_active_clusters,
            stream=fake_stream,
            options="--enable-tvm-ffi",
        )

        def tensor_api(
            a_tensor: torch.Tensor,
            b_tensor: torch.Tensor,
            sfa_tensor: torch.Tensor,
            sfb_tensor: torch.Tensor,
            c_tensor: torch.Tensor,
            amax_tensor: torch.Tensor,
            stream: cuda.CUstream,
        ):
            amax_tensor = self._pad_tensor_to_ndim(amax_tensor, 3, "amax")

            return _compiled_kernel(
                a_tensor,
                b_tensor,
                sfa_tensor,
                sfb_tensor,
                c_tensor,
                amax_tensor,
                stream,
            )

        self._compiled_kernel = tensor_api

        for attr_name in tuple(vars(self)):
            if attr_name.startswith("sample_"):
                setattr(self, attr_name, None)

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
    ) -> None:
        self._logger.debug("Entering execute")
        current_stream = self._get_default_stream(current_stream)

        self._runtime_error_if(
            self._compiled_kernel is None,
            "GemmAmaxSm100 kernel not compiled; call compile() first",
        )
        self._logger.debug("Executing with compiled kernel")

        self._compiled_kernel(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            sfa_tensor=sfa_tensor,
            sfb_tensor=sfb_tensor,
            c_tensor=c_tensor,
            amax_tensor=amax_tensor,
            stream=current_stream,
        )
        self._logger.debug("Executed with compiled kernel successfully")


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
) -> TupleDict:

    _logger.debug("gemm_amax_wrapper_sm100: Creating empty output tensors c and amax")

    m, _, batch = a_tensor.shape
    n, _, _b_batch = b_tensor.shape
    c_tensor = None
    if c_major == "m":
        c_tensor = torch.empty_strided((m, n, batch), (1, m, m * n), dtype=c_dtype, device=a_tensor.device)
    elif c_major == "n":
        c_tensor = torch.empty_strided((m, n, batch), (n, 1, m * n), dtype=c_dtype, device=a_tensor.device)
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
        gemm_amax.compile()
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

    return TupleDict(
        c_tensor=c_tensor,
        amax_tensor=amax_tensor,
    )
