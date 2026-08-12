# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .dense_blockscaled_gemm_persistent_amax import (
    Sm100BlockScaledPersistentDenseGemmKernel,
)

from cuda.bindings import driver as cuda
import os
from typing import Any, Tuple, Optional

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_stream

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.api_base import APIBase, TensorDesc, TupleDict, is_power_of_2, ceil_div
from cudnn.tensor_adapter import (
    canonicalize_unit_dim_strides,
    cuda_is_available,
    default_stream,
    detect_framework,
    framework_dtype,
    get_compute_capability,
    get_shape,
    get_strides,
)


class GemmAmaxSm100(APIBase):
    """Blockscaled GEMM with amax epilogue for SM100.

    Tensor parameters are type-erased: torch tensors and JAX arrays are both accepted
    (any DLPack-capable tensor with .shape/.dtype works for metadata). torch is only
    imported when torch tensors/dtypes are passed; likewise for JAX. Dtype parameters
    accept torch dtypes, numpy/ml_dtypes dtypes, or cutlass types.
    """

    def __init__(
        self,
        sample_a: Any,
        sample_b: Any,
        sample_sfa: Any,
        sample_sfb: Any,
        sample_c: Any,
        sample_amax: Any,
        acc_dtype: Any = cutlass.Float32,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        cluster_shape_mn: Tuple[int, int] = (1, 1),
        sf_vec_size: int = 32,
    ):
        super().__init__()

        self._warn_experimental_api()
        self._logger.debug("Entering __init__")

        self.a_desc = self._make_tensor_desc(sample_a, name="sample_a", canonical=True)
        self.b_desc = self._make_tensor_desc(sample_b, name="sample_b", canonical=True)
        self.sfa_desc = self._make_tensor_desc(sample_sfa, name="sample_sfa", canonical=True)
        self.sfb_desc = self._make_tensor_desc(sample_sfb, name="sample_sfb", canonical=True)
        self.c_desc = self._make_tensor_desc(sample_c, name="sample_c", canonical=True)
        self.amax_desc = self._make_tensor_desc(sample_amax, name="sample_amax", canonical=True)

        self.acc_dtype = _convert_to_cutlass_data_type(acc_dtype)
        self.mma_tiler_mn = mma_tiler_mn
        self.cluster_shape_mn = cluster_shape_mn
        self.sf_vec_size = sf_vec_size

        self.num_cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
        self._logger.debug(f"setting num_cluster_overlap_margin: {self.num_cluster_overlap_margin}")

        # used to reshape sfa/sfb tensors to atom layout
        self.atom_m = (32, 4)
        self.atom_k = 4

        self._interpret_uint8_as_fp4x2 = True
        self._logger.debug(
            f"__init__ completed with args: sample_a {self.a_desc.shape}, sample_b {self.b_desc.shape}, sample_sfa {self.sfa_desc.shape}, sample_sfb {self.sfb_desc.shape}, sample_c {self.c_desc.shape}, sample_amax {self.amax_desc.shape}, acc_dtype {acc_dtype}, mma_tiler_mn {mma_tiler_mn}, cluster_shape_mn {cluster_shape_mn}, sf_vec_size {sf_vec_size}"
        )

    def _check_sf_shape(self, sf_desc: TensorDesc, l: int, name: str) -> Tuple[int, int]:
        """Validate a scale-factor tensor shape and return (mn_div_atom_m0_m1, sf_k_div_atom_k).

        SF tensors are accepted in either of two equivalent forms (byte-identical memory):
        - the torch-style logical atom view (Atom_M0, Atom_M1, MN', Atom_K, K', L), i.e. a
          C-contiguous (L, MN', K', Atom_M0, Atom_M1, Atom_K) allocation permuted by
          (3, 4, 1, 5, 2, 0), or
        - that physical C-contiguous shape (L, MN', K', Atom_M0, Atom_M1, Atom_K) directly,
          for frameworks such as JAX that cannot express the permuted (strided) view.

        The kernel rebuilds the SF layout from the A/B shapes and consumes only the SF base
        pointer, so the memory must be exactly the C-contiguous physical allocation in both
        forms: strides are validated too (a shape-matching but differently-strided tensor
        would silently produce wrong results).
        """
        shape = sf_desc.shape
        self._value_error_if(len(shape) != 6, f"{name} tensor must be 6-D, got shape {shape}")
        atom_m0, atom_m1, atom_k = self.atom_m[0], self.atom_m[1], self.atom_k
        atom_elems = atom_m0 * atom_m1 * atom_k
        if shape[0] == atom_m0 and shape[1] == atom_m1 and shape[3] == atom_k:
            mn_div_atom_m0_m1, sf_k_div_atom_k = shape[2], shape[4]
            atom_shape = (atom_m0, atom_m1, mn_div_atom_m0_m1, atom_k, sf_k_div_atom_k, l)
            self._check_tensor_shape(sf_desc, atom_shape, name)
            _ = self._check_tensor_stride(
                sf_desc,
                stride=[
                    canonicalize_unit_dim_strides(
                        atom_shape,
                        (atom_m1 * atom_k, atom_k, sf_k_div_atom_k * atom_elems, 1, atom_elems, mn_div_atom_m0_m1 * sf_k_div_atom_k * atom_elems),
                    )
                ],
                name=name,
                extra_error_msg=f"{name} atom view must be the (3, 4, 1, 5, 2, 0) permutation of a C-contiguous physical allocation",
            )
        else:
            mn_div_atom_m0_m1, sf_k_div_atom_k = shape[1], shape[2]
            physical_shape = (l, mn_div_atom_m0_m1, sf_k_div_atom_k, atom_m0, atom_m1, atom_k)
            self._check_tensor_shape(sf_desc, physical_shape, name)
            _ = self._check_tensor_stride(
                sf_desc,
                stride=[
                    canonicalize_unit_dim_strides(
                        physical_shape,
                        (mn_div_atom_m0_m1 * sf_k_div_atom_k * atom_elems, sf_k_div_atom_k * atom_elems, atom_elems, atom_m1 * atom_k, atom_k, 1),
                    )
                ],
                name=name,
                extra_error_msg=f"{name} in the physical (L, MN', K', Atom_M0, Atom_M1, Atom_K) form must be C-contiguous",
            )
        return mn_div_atom_m0_m1, sf_k_div_atom_k

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        self._logger.debug("Checking dtypes and sf_vec_size")
        # Tensor descriptors are canonical (cutlass dtypes), so validation is framework-neutral.
        ab_dtype = self._check_dtype(
            self.a_desc,
            dtype=[cutlass.Float4E2M1FN, cutlass.Uint8, cutlass.Float8E5M2, cutlass.Float8E4M3FN],
            name="A",
        )
        self._check_dtype(
            self.b_desc,
            dtype=ab_dtype,
            name="B",
            extra_error_msg="A and B tensor dtypes must match",
        )
        if ab_dtype is cutlass.Uint8:
            self._logger.warning("Uint8 ab_dtype will be interpreted as packed fp4, not as native uint8")

        self._value_error_if(
            self.sf_vec_size not in {16, 32},
            f"Unsupported sf_vec_size: received {self.sf_vec_size}, expected {{16, 32}}",
        )

        sf_dtype = self._check_dtype(
            self.sfa_desc,
            dtype=[cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN, cutlass.Int8],
            name="sfa",
        )
        self._check_dtype(
            self.sfb_desc,
            dtype=sf_dtype,
            name="sfb",
            extra_error_msg="sfa and sfb tensor dtypes must match",
        )
        if sf_dtype is cutlass.Int8:
            self._logger.warning("Int8 sf_dtype will be interpreted as float8_e8m0fnu, not as native int8")

        self._value_error_if(
            sf_dtype is cutlass.Float8E4M3FN and self.sf_vec_size == 32,
            "Unsupported sf_dtype and sf_vec_size combination: float8_e4m3fn and 32 is not supported",
        )
        self._value_error_if(
            ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and self.sf_vec_size == 16,
            f"Unsupported ab_dtype and sf_vec_size combination: {{float8_e5m2, float8_e4m3fn}} and 16 is not supported",
        )

        c_dtype = self._check_dtype(
            self.c_desc,
            dtype=[cutlass.Float32, cutlass.Float16, cutlass.BFloat16, cutlass.Float8E5M2, cutlass.Float8E4M3FN, cutlass.Float4E2M1FN, cutlass.Uint8],
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
            dtype=cutlass.Float32,
            name="Accumulator",
            extra_error_msg="Accumulator must be float32",
        )

        self.ab_dtype = ab_dtype
        self.c_dtype = c_dtype

        self._logger.debug("Checking tensor layout")
        m, k, l = self.a_desc.shape
        n, _, _ = self.b_desc.shape

        self._check_tensor_shape(self.a_desc, (m, k, l), "A")
        self._check_tensor_shape(self.b_desc, (n, k, l), "B")
        self._check_tensor_shape(self.c_desc, (m, n, l), "C")
        m_div_atom_m0_m1, sf_k_div_atom_k = self._check_sf_shape(self.sfa_desc, l, "sfa")
        n_div_atom_m0_m1, sfb_k_div_atom_k = self._check_sf_shape(self.sfb_desc, l, "sfb")
        self._value_error_if(
            sf_k_div_atom_k != sfb_k_div_atom_k,
            f"sfa and sfb tensor K' mismatch: got {sf_k_div_atom_k} and {sfb_k_div_atom_k}",
        )
        self.amax_desc = self._pad_tensor_to_ndim(self.amax_desc, 3, "amax")
        self._check_tensor_shape(self.amax_desc, (1, 1, 1), "amax")

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
        _ = self._check_tensor_stride(
            self.a_desc,
            stride=[(1, m, m * k), (k, 1, m * k)],
            name="A",
        )
        _ = self._check_tensor_stride(
            self.b_desc,
            stride=[(1, n, n * k), (k, 1, n * k)],
            name="B",
        )
        _ = self._check_tensor_stride(
            self.c_desc,
            stride=[(1, m, m * n), (n, 1, m * n)],
            name="C",
        )

        # Derive major mode from stride order
        self.a_major = "m" if self.a_desc.stride_order == (0, 1, 2) else "k"
        self.b_major = "n" if self.b_desc.stride_order == (0, 1, 2) else "k"
        self.c_major = "m" if self.c_desc.stride_order == (0, 1, 2) else "n"

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
            self.mma_tiler_mn == (128, 256) and self.sf_vec_size == 16 and c_dtype in {cutlass.Float32, cutlass.Float16, cutlass.BFloat16},
            "mma_tiler_mn (128, 256), sf_vec_size 16, c_dtype {float32, float16, bfloat16} fails to launch",
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
        self._runtime_error_if(not cuda_is_available(), "CUDA is not available")
        major, minor = get_compute_capability()
        compute_capability = major * 10 + minor
        self._runtime_error_if(
            compute_capability < 100,
            f"GemmAmax requires SM100+ compute capability, but found SM{compute_capability} on the current device",
        )

        self._kernel = Sm100BlockScaledPersistentDenseGemmKernel

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def _compile_kernel(self):
        """Compile the kernel and return the raw TVM-FFI callable."""
        self._ensure_support_checked()

        gemm_amax = self._kernel(
            sf_vec_size=self.sf_vec_size,
            mma_tiler_mn=self.mma_tiler_mn,
            cluster_shape_mn=self.cluster_shape_mn,
        )
        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(self.cluster_shape_mn[0] * self.cluster_shape_mn[1])
        max_active_clusters -= self.num_cluster_overlap_margin
        self._value_error_if(
            max_active_clusters <= 0,
            "max_active_clusters must be > 0 after applying overlap margin; reduce CUDNNFE_CLUSTER_OVERLAP_MARGIN",
        )

        self._logger.debug("Compiling gemm_amax")
        a_cute = self._make_fake_cute_tensor_from_desc(self.a_desc, assumed_align=16)
        b_cute = self._make_fake_cute_tensor_from_desc(self.b_desc, assumed_align=16)
        sfa_cute = self._make_fake_cute_tensor_from_desc(self.sfa_desc, assumed_align=16)
        sfb_cute = self._make_fake_cute_tensor_from_desc(self.sfb_desc, assumed_align=16)
        c_cute = self._make_fake_cute_tensor_from_desc(self.c_desc, assumed_align=16)
        amax_cute = self._make_fake_cute_tensor_from_desc(self.amax_desc, assumed_align=16)
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)

        return cute.compile(
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

    def compile(self) -> None:
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            self._logger.debug("Kernel already compiled; skipping recompilation")
            return

        _compiled_kernel = self._compile_kernel()

        def tensor_api(
            a_tensor: Any,
            b_tensor: Any,
            sfa_tensor: Any,
            sfb_tensor: Any,
            c_tensor: Any,
            amax_tensor: Any,
            stream: cuda.CUstream,
        ):
            amax_tensor = self._pad_tensor_to_ndim(amax_tensor, 3, "amax")

            # The TVM-FFI callable converts any DLPack-capable tensor itself
            # (C fast path for torch, __dlpack__ protocol otherwise).
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
        a_tensor: Any,
        b_tensor: Any,
        sfa_tensor: Any,
        sfb_tensor: Any,
        c_tensor: Any,
        amax_tensor: Any,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        self._logger.debug("Entering execute")
        if current_stream is None:
            # torch inputs stay ordered with the caller's current torch stream;
            # other frameworks (e.g. JAX) default to the CUDA legacy default stream.
            current_stream = default_stream(detect_framework(a_tensor))

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


import logging

_logger = logging.getLogger(__name__)
_cache_of_GemmAmaxSm100Objects = {}


def gemm_amax_wrapper_sm100(
    a_tensor: Any,
    b_tensor: Any,
    sfa_tensor: Any,
    sfb_tensor: Any,
    c_major: str = "n",
    c_dtype: Any = cutlass.Float32,
    acc_dtype: Any = cutlass.Float32,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    cluster_shape_mn: Tuple[int, int] = (1, 1),
    sf_vec_size: int = 32,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:

    _logger.debug("gemm_amax_wrapper_sm100: Creating empty output tensors c and amax")

    framework = detect_framework(a_tensor)
    c_dtype = _convert_to_cutlass_data_type(c_dtype)
    acc_dtype = _convert_to_cutlass_data_type(acc_dtype)

    m, _, l = get_shape(a_tensor)
    n, _, l = get_shape(b_tensor)
    if c_major not in ("m", "n"):
        raise ValueError(f"c_major must be either 'm' or 'n', got {c_major}")

    if framework == "torch":
        import torch

        if c_major == "m":
            c_tensor = torch.empty_strided((m, n, l), (1, m, m * n), dtype=framework_dtype(c_dtype, "torch"), device=a_tensor.device)
        else:
            c_tensor = torch.empty_strided((m, n, l), (n, 1, m * n), dtype=framework_dtype(c_dtype, "torch"), device=a_tensor.device)
        amax_tensor = torch.full((1, 1, 1), -float("inf"), device=a_tensor.device, dtype=torch.float32)
    elif framework == "jax":
        import jax
        import jax.numpy as jnp

        if c_major == "m":
            raise ValueError("JAX arrays are row-major; only c_major='n' is supported for JAX inputs")
        if l != 1:
            raise ValueError("JAX inputs must have batch dim L == 1; batch-outermost (L-major) layouts are not expressible as JAX arrays")
        device = a_tensor.device
        c_tensor = jnp.empty((m, n, l), dtype=framework_dtype(c_dtype, "jax"), device=device)
        amax_tensor = jnp.full((1, 1, 1), -float("inf"), dtype=jnp.float32, device=device)
        # The kernel writes into these buffers on the launch stream; make sure XLA has
        # finished materializing them before the kernel runs.
        jax.block_until_ready((c_tensor, amax_tensor))
    else:
        raise ValueError(f"Unsupported tensor framework '{framework}' for gemm_amax_wrapper_sm100; pass torch tensors or JAX arrays")

    cache_key = (
        get_shape(a_tensor),
        get_shape(b_tensor),
        get_shape(sfa_tensor),
        get_shape(sfb_tensor),
        _convert_to_cutlass_data_type(a_tensor.dtype),
        _convert_to_cutlass_data_type(b_tensor.dtype),
        _convert_to_cutlass_data_type(sfa_tensor.dtype),
        _convert_to_cutlass_data_type(sfb_tensor.dtype),
        canonicalize_unit_dim_strides(get_shape(a_tensor), get_strides(a_tensor)),
        canonicalize_unit_dim_strides(get_shape(b_tensor), get_strides(b_tensor)),
        canonicalize_unit_dim_strides(get_shape(sfa_tensor), get_strides(sfa_tensor)),
        canonicalize_unit_dim_strides(get_shape(sfb_tensor), get_strides(sfb_tensor)),
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
