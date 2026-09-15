# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared metadata validation for canonical grouped MXFP8 JAX entry points."""

import os

import cutlass
import cutlass.utils
import jax
import jax.numpy as jnp
import ml_dtypes

from cudnn.api_base import ceil_div
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.jax import TensorSpec, row_major_desc
from cudnn.tensor_adapter import framework_dtype

kernel_cache = {}
validated_configs = set()


def row_spec(array):
    return TensorSpec(layout=tuple(reversed(range(len(array.shape)))))


def sf_array(array):
    if _convert_to_cutlass_data_type(array.dtype) is cutlass.Uint8:
        return array.view(ml_dtypes.float8_e8m0fnu)
    return array


def sf_zeros(shape_dtype):
    return jnp.zeros(shape_dtype.shape, jnp.uint8).view(shape_dtype.dtype)


def sf_shape(rows, cols):
    return (1, ceil_div(rows, 128), ceil_div(ceil_div(cols, 32), 4), 32, 4, 4)


def output_type(shape, dtype):
    return jax.ShapeDtypeStruct(shape, framework_dtype(dtype, "jax"))


def grouped_plan(api_type, inputs, outputs, *, backward, mma_tiler_mn, cluster_shape_mn):
    a, b = inputs["a"], inputs["b"]
    if a.ndim != 2 or b.ndim != 3:
        raise ValueError("A must have shape (m, k) and B (experts, n, k)")
    m, k = a.shape
    experts, n, bk = b.shape
    if m <= 0 or experts <= 0 or k != bk:
        raise ValueError("Expected nonempty A/B with matching K and at least one expert")
    if m % 256:
        raise ValueError("A rows must be padded to a multiple of 256")
    fp8 = (cutlass.Float8E4M3FN, cutlass.Float8E5M2)
    for name in ("a", "b"):
        if _convert_to_cutlass_data_type(inputs[name].dtype) not in fp8:
            raise ValueError(f"{name} must be MXFP8 e4m3 or e5m2; packed FP4 is unsupported")
    for name in ("sfa", "sfb"):
        if _convert_to_cutlass_data_type(inputs[name].dtype) is not cutlass.Float8E8M0FNU:
            raise ValueError(f"{name} must contain E8M0 scale bytes")
    for name in ("alpha", "beta", "norm_const"):
        if name in inputs and _convert_to_cutlass_data_type(inputs[name].dtype) is not cutlass.Float32:
            raise ValueError(f"{name} must be float32")
    if inputs["prob"].shape != (m,):
        raise ValueError("prob must have shape (m,)")
    if inputs["padded_offsets"].shape != (experts,) or _convert_to_cutlass_data_type(inputs["padded_offsets"].dtype) is not cutlass.Int32:
        raise ValueError("padded_offsets must have shape (experts,) and dtype int32")
    d = outputs["d_row" if backward else "d"]
    if _convert_to_cutlass_data_type(d.dtype) not in fp8:
        raise ValueError("The MXFP8 JAX entry point requires FP8 output dtype")
    margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
    config = (backward, experts, mma_tiler_mn, cluster_shape_mn, margin)
    signature = tuple((name, tuple(t.shape), str(t.dtype)) for name, t in (*inputs.items(), *outputs.items()))
    validation_key = (config, signature)
    if validation_key not in validated_configs:
        samples = {f"sample_{name}": row_major_desc(t.shape, t.dtype, f"sample_{name}") for name, t in (*inputs.items(), *outputs.items())}
        api = api_type(**samples, sf_vec_size=32, mma_tiler_mn=mma_tiler_mn, cluster_shape_mn=cluster_shape_mn)
        api.check_support()
        if config not in kernel_cache:
            kwargs = dict(
                sf_vec_size=32,
                acc_dtype=cutlass.Float32,
                use_2cta_instrs=api.use_2cta_instrs,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=api.cluster_shape_mn,
                discrete_col_sfd=False,
                expert_cnt=experts,
                use_mono_increase_expert_idx=True,
            )
            if backward:
                kwargs["vectorized_f32"] = False
            else:
                kwargs.update(vector_f32=False, generate_sfd=True)
            mac = cutlass.utils.HardwareInfo().get_max_active_clusters(api.cluster_shape_mn[0] * api.cluster_shape_mn[1]) - margin
            if mac <= 0:
                raise ValueError("CUDNNFE_CLUSTER_OVERLAP_MARGIN leaves no active clusters")
            kernel_cache[config] = (api._kernel(**kwargs), mac)
        validated_configs.add(validation_key)
    return kernel_cache[config]
