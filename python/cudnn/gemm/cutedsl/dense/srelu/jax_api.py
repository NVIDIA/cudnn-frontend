# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX-native (XLA custom call) entry points for the GEMM + sReLU forward and
backward (dsReLU) kernels, built on :func:`cudnn.jax.call`.

The kernels' optional sfd/amax/norm_const parameters are compile-time ``None``
constants inside the ``@cute.jit`` adapters for the JAX-reachable (non-fp8-D)
configurations.
"""

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp

import cutlass
import cutlass.cute as cute
import cutlass.utils

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.tensor_adapter import framework_dtype
from cudnn.jax import TensorSpec, call, gemm_operand_spec, row_major_desc as _make_desc, sf_atom_spec, zeros_init
from .api import GemmSreluSm100
from ..dsrelu.api import GemmDsreluSm100

# config_key -> (kernel instance, max_active_clusters). The instances do not vary
# with problem shapes, so the keys hold only kernel-construction config — a shared
# instance keeps cutlass_call's compile cache warm (its FunctionSpec keys on the
# constexpr kwargs). Shape/dtype validation is cached separately per full signature.
_srelu_kernel_cache: dict = {}
_srelu_validated_configs: set = set()
_dsrelu_kernel_cache: dict = {}
_dsrelu_validated_configs: set = set()

# Constexpr epilogues, identical to the class APIs' compile()
_SRELU_EPILOGUE = lambda x: cute.where(x > 0, x, cute.full_like(x, 0)) ** 2  # noqa: E731
_DSRELU_EPILOGUE = lambda x, y: cute.where(x > 0, x, cute.full_like(x, 0)) * 2 * y  # noqa: E731


def _prob_spec() -> TensorSpec:
    # (m, 1, 1) with m innermost: explicit ranks because trailing unit dims make
    # leading-dim inference ambiguous
    return TensorSpec(layout=(0, 1, 2))


@cute.jit
def _srelu_adapter(stream, a, b, sfa, sfb, prob, c, d, *, kernel, mac, alpha):
    # sfd/amax/norm_const are compile-time Nones (fp8-D configs are not reachable from JAX)
    kernel(
        a_tensor=a,
        b_tensor=b,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        c_tensor=c,
        d_tensor=d,
        prob_tensor=prob,
        amax_tensor=None,
        sfd_tensor=None,
        norm_const_tensor=None,
        alpha=alpha,
        max_active_clusters=mac,
        stream=stream,
        epilogue_op=_SRELU_EPILOGUE,
    )


@cute.jit
def _dsrelu_adapter(stream, a, b, sfa, sfb, c, prob, d, dprob, *, kernel, mac, alpha):
    kernel(
        a_tensor=a,
        b_tensor=b,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        c_tensor=c,
        d_tensor=d,
        prob_tensor=prob,
        dprob_tensor=dprob,
        amax_tensor=None,
        sfd_tensor=None,
        norm_const_tensor=None,
        alpha=alpha,
        max_active_clusters=mac,
        stream=stream,
        epilogue_op=_DSRELU_EPILOGUE,
    )


def gemm_srelu_jax_sm100(
    a_tensor: Any,
    b_tensor: Any,
    sfa_tensor: Any,
    sfb_tensor: Any,
    prob_tensor: Any,
    alpha: float = 1.0,
    c_dtype: Any = cutlass.BFloat16,
    d_dtype: Any = cutlass.BFloat16,
    acc_dtype: Any = cutlass.Float32,
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    sf_vec_size: int = 16,
    vector_f32: bool = False,
) -> Tuple[Any, Any]:
    """Blockscaled GEMM + sReLU as an XLA custom call; usable eagerly or under jax.jit.

    A (M, K, 1) / B (N, K, 1) k-major C-contiguous fp8 arrays, SFA/SFB in the physical
    atom shape (1, MN', K', 32, 4, 4), prob (M, 1, 1) float32. Returns a plain
    ``(c_tensor, d_tensor)`` tuple of fresh n-major JAX arrays. fp8 ``d_dtype``
    configurations (which produce sfd/norm_const outputs) are not reachable from JAX.
    """
    c_dtype = _convert_to_cutlass_data_type(c_dtype)
    d_dtype = _convert_to_cutlass_data_type(d_dtype)
    acc_dtype = _convert_to_cutlass_data_type(acc_dtype)

    m, _, l = a_tensor.shape
    n, _, _ = b_tensor.shape
    if l != 1:
        raise ValueError("JAX inputs must have batch dim L == 1; batch-outermost (L-major) layouts are not expressible as JAX arrays")
    if d_dtype in (cutlass.Float8E4M3FN, cutlass.Float8E5M2):
        raise ValueError("fp8 d_dtype requires sfd/norm_const outputs, which are not reachable from JAX; use the eager wrapper with torch tensors")

    config_key = (
        c_dtype,
        d_dtype,
        acc_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        sf_vec_size,
        vector_f32,
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
    if validation_key not in _srelu_validated_configs:
        gemm = GemmSreluSm100(
            sample_a=_make_desc(tuple(a_tensor.shape), a_tensor.dtype, "sample_a"),
            sample_b=_make_desc(tuple(b_tensor.shape), b_tensor.dtype, "sample_b"),
            sample_c=_make_desc((m, n, l), c_dtype, "sample_c"),
            sample_d=_make_desc((m, n, l), d_dtype, "sample_d"),
            sample_sfa=_make_desc(tuple(sfa_tensor.shape), sfa_tensor.dtype, "sample_sfa"),
            sample_sfb=_make_desc(tuple(sfb_tensor.shape), sfb_tensor.dtype, "sample_sfb"),
            sample_prob=_make_desc((m, 1, 1), cutlass.Float32, "sample_prob"),
            alpha=alpha,
            acc_dtype=acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sf_vec_size=sf_vec_size,
            vector_f32=vector_f32,
        )
        assert gemm.check_support()
        if config_key not in _srelu_kernel_cache:
            kernel = gemm._kernel(
                sf_vec_size=sf_vec_size,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=gemm.cluster_shape_mn,
                vector_f32=vector_f32,
            )
            mac = cutlass.utils.HardwareInfo().get_max_active_clusters(gemm.cluster_shape_mn[0] * gemm.cluster_shape_mn[1]) - gemm.num_cluster_overlap_margin
            _srelu_kernel_cache[config_key] = (kernel, mac)
        _srelu_validated_configs.add(validation_key)
    kernel, mac = _srelu_kernel_cache[config_key]

    operand = gemm_operand_spec()
    sf = sf_atom_spec()
    c_tensor, d_tensor = call(
        _srelu_adapter,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((m, n, l), framework_dtype(c_dtype, "jax")),
            jax.ShapeDtypeStruct((m, n, l), framework_dtype(d_dtype, "jax")),
        ),
        input_spec=(operand, operand, sf, sf, _prob_spec()),
        output_spec=(operand, operand),
        # Donated pre-initialized outputs: the bridge's leading-dim inference rejects
        # trailing-unit-dim buffers on pure results.
        initialized_outputs={0: zeros_init, 1: zeros_init},
        kernel=kernel,
        mac=mac,
        alpha=float(alpha),
    )(a_tensor, b_tensor, sfa_tensor, sfb_tensor, prob_tensor)

    return c_tensor, d_tensor


def gemm_dsrelu_jax_sm100(
    a_tensor: Any,
    b_tensor: Any,
    c_tensor: Any,
    sfa_tensor: Any,
    sfb_tensor: Any,
    prob_tensor: Any,
    alpha: float = 1.0,
    d_dtype: Any = cutlass.BFloat16,
    acc_dtype: Any = cutlass.Float32,
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    sf_vec_size: int = 16,
    vector_f32: bool = False,
) -> Tuple[Any, Any]:
    """Blockscaled GEMM + dsReLU (backward) as an XLA custom call.

    Returns a plain ``(d_tensor, dprob_tensor)`` tuple; dprob is a zero-initialized
    atomic-add accumulator managed by the bridge.
    """
    d_dtype = _convert_to_cutlass_data_type(d_dtype)
    acc_dtype = _convert_to_cutlass_data_type(acc_dtype)

    m, _, l = a_tensor.shape
    n, _, _ = b_tensor.shape
    if l != 1:
        raise ValueError("JAX inputs must have batch dim L == 1; batch-outermost (L-major) layouts are not expressible as JAX arrays")
    if d_dtype in (cutlass.Float8E4M3FN, cutlass.Float8E5M2):
        raise ValueError("fp8 d_dtype requires sfd/norm_const outputs, which are not reachable from JAX; use the eager wrapper with torch tensors")

    config_key = (
        d_dtype,
        acc_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        sf_vec_size,
        vector_f32,
    )
    validation_key = (
        tuple(a_tensor.shape),
        tuple(b_tensor.shape),
        tuple(c_tensor.shape),
        tuple(sfa_tensor.shape),
        tuple(sfb_tensor.shape),
        _convert_to_cutlass_data_type(a_tensor.dtype),
        _convert_to_cutlass_data_type(b_tensor.dtype),
        _convert_to_cutlass_data_type(c_tensor.dtype),
        _convert_to_cutlass_data_type(sfa_tensor.dtype),
        _convert_to_cutlass_data_type(sfb_tensor.dtype),
        config_key,
    )
    if validation_key not in _dsrelu_validated_configs:
        gemm = GemmDsreluSm100(
            sample_a=_make_desc(tuple(a_tensor.shape), a_tensor.dtype, "sample_a"),
            sample_b=_make_desc(tuple(b_tensor.shape), b_tensor.dtype, "sample_b"),
            sample_c=_make_desc(tuple(c_tensor.shape), c_tensor.dtype, "sample_c"),
            sample_d=_make_desc((m, n, l), d_dtype, "sample_d"),
            sample_dprob=_make_desc((m, 1, l), cutlass.Float32, "sample_dprob"),
            sample_sfa=_make_desc(tuple(sfa_tensor.shape), sfa_tensor.dtype, "sample_sfa"),
            sample_sfb=_make_desc(tuple(sfb_tensor.shape), sfb_tensor.dtype, "sample_sfb"),
            sample_prob=_make_desc((m, 1, 1), cutlass.Float32, "sample_prob"),
            alpha=alpha,
            acc_dtype=acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sf_vec_size=sf_vec_size,
            vector_f32=vector_f32,
        )
        assert gemm.check_support()
        if config_key not in _dsrelu_kernel_cache:
            kernel = gemm._kernel(
                sf_vec_size=sf_vec_size,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=gemm.cluster_shape_mn,
                vector_f32=vector_f32,
            )
            mac = cutlass.utils.HardwareInfo().get_max_active_clusters(gemm.cluster_shape_mn[0] * gemm.cluster_shape_mn[1]) - gemm.num_cluster_overlap_margin
            _dsrelu_kernel_cache[config_key] = (kernel, mac)
        _dsrelu_validated_configs.add(validation_key)
    kernel, mac = _dsrelu_kernel_cache[config_key]

    operand = gemm_operand_spec()
    sf = sf_atom_spec()
    d_tensor, dprob_tensor = call(
        _dsrelu_adapter,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((m, n, l), framework_dtype(d_dtype, "jax")),
            jax.ShapeDtypeStruct((m, 1, l), jnp.float32),
        ),
        input_spec=(operand, operand, sf, sf, operand, _prob_spec()),
        output_spec=(operand, _prob_spec()),
        # d is donated for the trailing-unit-dim layout spec; dprob is a genuine
        # zero-initialized atomic-add accumulator.
        initialized_outputs={0: zeros_init, 1: zeros_init},
        kernel=kernel,
        mac=mac,
        alpha=float(alpha),
    )(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor, prob_tensor)

    return d_tensor, dprob_tensor
