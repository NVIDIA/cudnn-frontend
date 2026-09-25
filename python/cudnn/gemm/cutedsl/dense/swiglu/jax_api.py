# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX-native (XLA custom call) entry point for the GEMM + SwiGLU kernels.

Built on :func:`cudnn.jax.call` (CuTeDSL's native JAX bridge). Both the standard and
the blockscaled (MXFP8) quantized kernels are supported: the quantized kernel's
optional amax/sfc/norm_const parameters are compile-time ``None`` constants inside the
``@cute.jit`` adapter for the JAX-reachable configurations.
"""

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp

import cutlass
import cutlass.cute as cute
import cutlass.utils

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.tensor_adapter import framework_dtype
from cudnn.jax import call, gemm_operand_spec, row_major_desc as _make_desc, sf_atom_spec
from .api import GemmSwigluSm100

# config_key -> (kernel instance, max_active_clusters). The instance does not vary
# with problem shapes, so the key holds only kernel-construction config — a shared
# instance keeps cutlass_call's compile cache warm (its FunctionSpec keys on the
# constexpr kwargs). Shape/dtype validation is cached separately per full signature.
_kernel_cache: dict = {}
_validated_configs: set = set()


@cute.jit
def _swiglu_adapter(stream, a, b, ab12, c, *, kernel, mac, alpha):
    kernel(a, b, ab12, c, alpha, mac, stream)


@cute.jit
def _swiglu_quant_adapter(stream, a, b, sfa, sfb, c, ab12, *, kernel, mac, alpha):
    # amax/sfc/norm_const are compile-time Nones: fp8-C and fp4-A/B configurations
    # (which would produce those outputs) are not reachable from JAX.
    kernel(a, b, sfa, sfb, c, ab12, None, None, None, alpha, mac, stream)


def gemm_swiglu_jax_sm100(
    a_tensor: Any,
    b_tensor: Any,
    alpha: float = 1.0,
    ab12_dtype: Any = cutlass.Float32,
    c_dtype: Any = cutlass.Float16,
    acc_dtype: Any = cutlass.Float32,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    ### Quantize only arguments
    sfa_tensor: Optional[Any] = None,
    sfb_tensor: Optional[Any] = None,
    sf_vec_size: int = 16,
    vector_f32: bool = False,
    ab12_stages: int = 4,
) -> Tuple[Any, Any]:
    """GEMM + SwiGLU as an XLA custom call; usable eagerly or under jax.jit.

    A (M, K, 1) and B (N, K, 1) are k-major C-contiguous JAX arrays (or tracers);
    optional SFA/SFB (blockscaled MXFP8 path) in the physical atom shape
    (1, MN', K', 32, 4, 4). Returns a plain ``(ab12_tensor, c_tensor)`` tuple of
    fresh n-major JAX arrays. ``alpha`` is a static (trace-time) parameter.
    """
    ab12_dtype = _convert_to_cutlass_data_type(ab12_dtype)
    c_dtype = _convert_to_cutlass_data_type(c_dtype)
    acc_dtype = _convert_to_cutlass_data_type(acc_dtype)

    m, _, l = a_tensor.shape
    n, _, _ = b_tensor.shape
    if l != 1:
        raise ValueError("JAX inputs must have batch dim L == 1; batch-outermost (L-major) layouts are not expressible as JAX arrays")

    is_quantized = sfa_tensor is not None and sfb_tensor is not None
    if (sfa_tensor is None) != (sfb_tensor is None):
        raise ValueError("Provide both sfa_tensor and sfb_tensor for the quantized kernel, or neither")

    config_key = (
        is_quantized,
        ab12_dtype,
        c_dtype,
        acc_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        sf_vec_size,
        vector_f32,
        ab12_stages,
    )
    validation_key = (
        tuple(a_tensor.shape),
        tuple(b_tensor.shape),
        _convert_to_cutlass_data_type(a_tensor.dtype),
        _convert_to_cutlass_data_type(b_tensor.dtype),
        tuple(sfa_tensor.shape) if is_quantized else None,
        tuple(sfb_tensor.shape) if is_quantized else None,
        _convert_to_cutlass_data_type(sfa_tensor.dtype) if is_quantized else None,
        _convert_to_cutlass_data_type(sfb_tensor.dtype) if is_quantized else None,
        config_key,
    )

    if validation_key not in _validated_configs:
        gemm = GemmSwigluSm100(
            sample_a=_make_desc(tuple(a_tensor.shape), a_tensor.dtype, "sample_a"),
            sample_b=_make_desc(tuple(b_tensor.shape), b_tensor.dtype, "sample_b"),
            sample_ab12=_make_desc((m, n, l), ab12_dtype, "sample_ab12"),
            sample_c=_make_desc((m, n // 2, l), c_dtype, "sample_c"),
            alpha=alpha,
            acc_dtype=acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sample_sfa=_make_desc(tuple(sfa_tensor.shape), sfa_tensor.dtype, "sample_sfa") if is_quantized else None,
            sample_sfb=_make_desc(tuple(sfb_tensor.shape), sfb_tensor.dtype, "sample_sfb") if is_quantized else None,
            sf_vec_size=sf_vec_size,
            vector_f32=vector_f32,
            ab12_stages=ab12_stages,
        )
        assert gemm.check_support()
        if config_key not in _kernel_cache:
            if is_quantized:
                kernel = gemm._kernel(
                    sf_vec_size=sf_vec_size,
                    mma_tiler_mn=mma_tiler_mn,
                    cluster_shape_mn=gemm.cluster_shape_mn,
                    vector_f32=vector_f32,
                    ab12_stages=ab12_stages,
                )
            else:
                kernel = gemm._kernel(
                    acc_dtype=acc_dtype,
                    use_2cta_instrs=(mma_tiler_mn[0] == 256),
                    mma_tiler_mn=mma_tiler_mn,
                    cluster_shape_mn=gemm.cluster_shape_mn,
                )
            mac = cutlass.utils.HardwareInfo().get_max_active_clusters(gemm.cluster_shape_mn[0] * gemm.cluster_shape_mn[1]) - gemm.num_cluster_overlap_margin
            _kernel_cache[config_key] = (kernel, mac)
        _validated_configs.add(validation_key)
    kernel, mac = _kernel_cache[config_key]

    operand = gemm_operand_spec()
    out_types = (
        jax.ShapeDtypeStruct((m, n, l), framework_dtype(ab12_dtype, "jax")),  # ab12
        jax.ShapeDtypeStruct((m, n // 2, l), framework_dtype(c_dtype, "jax")),  # c
    )
    # Neither output needs zeroing: the kernel writes ab12 and c in full over (m, n, l)
    # and (m, n // 2, l). Zero-filling them was a full-size device write per dispatch.
    if not is_quantized:
        ab12_tensor, c_tensor = call(
            _swiglu_adapter,
            output_shape_dtype=out_types,
            input_spec=(operand, operand),
            output_spec=(operand, operand),
            kernel=kernel,
            mac=mac,
            alpha=float(alpha),
        )(a_tensor, b_tensor)
    else:
        sf = sf_atom_spec()
        c_tensor, ab12_tensor = call(
            _swiglu_quant_adapter,
            output_shape_dtype=(out_types[1], out_types[0]),
            input_spec=(operand, operand, sf, sf),
            output_spec=(operand, operand),
            kernel=kernel,
            mac=mac,
            alpha=float(alpha),
        )(a_tensor, b_tensor, sfa_tensor, sfb_tensor)

    return ab12_tensor, c_tensor
