# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX-native (XLA custom call) entry point for the blockscaled GEMM + amax kernel.

Built on :func:`cudnn.jax.call` (CuTeDSL's native JAX bridge): the kernel runs on
XLA's compute stream, outputs are XLA-managed, and the call composes with ``jax.jit``
and CUDA graph capture. No manual synchronization is needed.
"""

from typing import Any, Tuple

import jax
import jax.numpy as jnp

import cutlass
import cutlass.cute as cute
import cutlass.utils

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.tensor_adapter import framework_dtype
from cudnn.jax import call, gemm_operand_spec, row_major_desc as _make_desc, sf_atom_spec, zeros_init
from .api import GemmAmaxSm100

# config_key -> (kernel instance, max_active_clusters). The instance does not vary
# with problem shapes, so the key holds only kernel-construction config — a shared
# instance keeps cutlass_call's compile cache warm (its FunctionSpec keys on the
# constexpr kwargs). Shape/dtype validation is cached separately per full signature.
_kernel_cache: dict = {}
_validated_configs: set = set()


@cute.jit
def _amax_adapter(stream, a, b, sfa, sfb, c, amax, *, kernel, mac):
    # Destination-passing kernel signature; amax arrives pre-initialized (donated input).
    kernel(a, b, sfa, sfb, c, amax, mac, stream)


def gemm_amax_jax_sm100(
    a_tensor: Any,
    b_tensor: Any,
    sfa_tensor: Any,
    sfb_tensor: Any,
    c_dtype: Any = cutlass.Float32,
    acc_dtype: Any = cutlass.Float32,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    cluster_shape_mn: Tuple[int, int] = (1, 1),
    sf_vec_size: int = 32,
) -> Tuple[Any, Any]:
    """Blockscaled GEMM + amax as an XLA custom call; usable eagerly or under jax.jit.

    Arguments are JAX arrays (or tracers): A (M, K, 1) and B (N, K, 1) k-major
    C-contiguous, SFA/SFB in the physical atom shape (1, MN', K', 32, 4, 4).
    Returns a plain ``(c_tensor, amax_tensor)`` tuple of fresh JAX arrays; C is n-major.
    """
    c_dtype = _convert_to_cutlass_data_type(c_dtype)
    acc_dtype = _convert_to_cutlass_data_type(acc_dtype)

    m, _, l = a_tensor.shape
    n, _, _ = b_tensor.shape
    if l != 1:
        raise ValueError("JAX inputs must have batch dim L == 1; batch-outermost (L-major) layouts are not expressible as JAX arrays")

    config_key = (
        c_dtype,
        acc_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        sf_vec_size,
    )
    validation_key = (
        tuple(a_tensor.shape),
        tuple(b_tensor.shape),
        tuple(sfa_tensor.shape),
        tuple(sfb_tensor.shape),
        _convert_to_cutlass_data_type(a_tensor.dtype),
        _convert_to_cutlass_data_type(b_tensor.dtype),
        _convert_to_cutlass_data_type(sfa_tensor.dtype),
        _convert_to_cutlass_data_type(sfb_tensor.dtype),
        config_key,
    )

    if validation_key not in _validated_configs:
        # Validation reuses the class API's check_support on metadata-only descriptors
        # (works for jax.jit tracers, which expose only .shape/.dtype).
        gemm = GemmAmaxSm100(
            sample_a=_make_desc(tuple(a_tensor.shape), a_tensor.dtype, "sample_a"),
            sample_b=_make_desc(tuple(b_tensor.shape), b_tensor.dtype, "sample_b"),
            sample_sfa=_make_desc(tuple(sfa_tensor.shape), sfa_tensor.dtype, "sample_sfa"),
            sample_sfb=_make_desc(tuple(sfb_tensor.shape), sfb_tensor.dtype, "sample_sfb"),
            sample_c=_make_desc((m, n, l), c_dtype, "sample_c"),
            sample_amax=_make_desc((1, 1, 1), cutlass.Float32, "sample_amax"),
            acc_dtype=acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sf_vec_size=sf_vec_size,
        )
        assert gemm.check_support()
        if config_key not in _kernel_cache:
            kernel = gemm._kernel(
                sf_vec_size=sf_vec_size,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
            )
            mac = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1]) - gemm.num_cluster_overlap_margin
            _kernel_cache[config_key] = (kernel, mac)
        _validated_configs.add(validation_key)
    kernel, mac = _kernel_cache[config_key]

    operand = gemm_operand_spec()
    sf = sf_atom_spec()
    c_tensor, amax_tensor = call(
        _amax_adapter,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((m, n, l), framework_dtype(c_dtype, "jax")),
            jax.ShapeDtypeStruct((1, 1, 1), jnp.float32),
        ),
        input_spec=(operand, operand, sf, sf),
        output_spec=(operand, None),
        # Both outputs are donated pre-initialized inputs: amax needs the zero identity
        # (signed-int atomic max of non-negative values), and routing c through the
        # donated-input path gives it the explicit (1, 0, 2) layout spec -- the bridge's
        # leading-dim inference rejects trailing-unit-dim buffers on pure results.
        initialized_outputs={0: zeros_init, 1: zeros_init},
        kernel=kernel,
        mac=mac,
    )(a_tensor, b_tensor, sfa_tensor, sfb_tensor)

    return c_tensor, amax_tensor
