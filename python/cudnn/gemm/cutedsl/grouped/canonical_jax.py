# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared metadata validation for canonical grouped MXFP8 JAX entry points."""

import math
import os
from functools import lru_cache

import cutlass
import cutlass.utils
import jax
import ml_dtypes

from cudnn.api_base import TupleDict, ceil_div
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.frost.buffers import cutedsl_arch_requirement_error
from cudnn.jax import TensorSpec, call, row_major_desc
from cudnn.tensor_adapter import detect_framework, framework_dtype

jax.tree_util.register_pytree_node(
    TupleDict,
    lambda value: (tuple(value.values()), tuple(value.keys())),
    lambda keys, values: TupleDict(zip(keys, values)),
)


kernel_cache = {}
validated_configs = set()


def row_spec(array):
    return TensorSpec(layout=tuple(reversed(range(len(array.shape)))))


def sf_array(array):
    if _convert_to_cutlass_data_type(array.dtype) is cutlass.Uint8:
        return array.view(ml_dtypes.float8_e8m0fnu)
    return array


def sf_shape(rows, cols, sf_vec_size=32):
    return (1, ceil_div(rows, 128), ceil_div(ceil_div(cols, sf_vec_size), 4), 32, 4, 4)


fp8_dtypes = (cutlass.Float8E4M3FN, cutlass.Float8E5M2)


def operand_dtype(array):
    """JAX has no packed FP4 dtype, so FP4 operands travel as uint8 with two values per byte."""
    dtype = _convert_to_cutlass_data_type(array.dtype)
    return cutlass.Float4E2M1FN if dtype is cutlass.Uint8 else dtype


def check_block_scaled_operands(a, b, sfa, sfb, sf_vec_size):
    """Validate the A/B and scale dtypes; return the A/B dtype and the scale dtype."""
    ab_dtype = operand_dtype(a)
    if ab_dtype not in (*fp8_dtypes, cutlass.Float4E2M1FN) or operand_dtype(b) is not ab_dtype:
        raise ValueError("a and b must have the same dtype: MXFP8 e4m3 or e5m2, or packed FP4 as uint8")
    sf_dtype = _convert_to_cutlass_data_type(sfa.dtype)
    if sf_dtype not in (cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN) or _convert_to_cutlass_data_type(sfb.dtype) is not sf_dtype:
        raise ValueError("sfa and sfb must both contain E8M0 scale bytes, or both E4M3 for NVFP4")
    if sf_vec_size not in (16, 32) or (ab_dtype in fp8_dtypes and (sf_vec_size != 32 or sf_dtype is not cutlass.Float8E8M0FNU)):
        raise ValueError("sf_vec_size must be 32 with E8M0 scales for MXFP8, and 16 or 32 for FP4")
    return ab_dtype, sf_dtype


def check_packed_sf_size(inputs, sizes, rest_k):
    for name, rows, groups in sizes:
        if math.prod(inputs[name].shape) != 512 * ceil_div(rows, 128) * rest_k * groups:
            raise ValueError(f"{name.upper()} must contain the complete MMA-packed scale buffer")


def output_type(shape, dtype):
    return jax.ShapeDtypeStruct(shape, framework_dtype(dtype, "jax"))


def check_grouped_shapes(inputs, outputs, *, backward, sf_vec_size=32):
    """Validate the canonical GLU/dGLU operands; return the A/B dtype, the scale dtype and the logical K."""
    a, b = inputs["a"], inputs["b"]
    if a.ndim != 2 or b.ndim != 3:
        raise ValueError("A must have shape (m, k) and B (experts, n, k)")
    m, k = a.shape
    experts, n, bk = b.shape
    if m <= 0 or experts <= 0 or k != bk:
        raise ValueError("Expected nonempty A/B with matching K and at least one expert")
    if m % 256:
        raise ValueError("A rows must be padded to a multiple of 256")
    ab_dtype, sf_dtype = check_block_scaled_operands(a, b, inputs["sfa"], inputs["sfb"], sf_vec_size)
    for name, shape in (("alpha", (experts,)), ("beta", (experts,)), ("norm_const", (1,))):
        if name in inputs and (tuple(inputs[name].shape) != shape or _convert_to_cutlass_data_type(inputs[name].dtype) is not cutlass.Float32):
            raise ValueError(f"{name} must have shape {shape} and dtype float32")
    if inputs["prob"].shape != (m,):
        raise ValueError("prob must have shape (m,)")
    if inputs["padded_offsets"].shape != (experts,) or _convert_to_cutlass_data_type(inputs["padded_offsets"].dtype) is not cutlass.Int32:
        raise ValueError("padded_offsets must have shape (experts,) and dtype int32")
    d_dtype = _convert_to_cutlass_data_type(outputs["d_row" if backward else "d"].dtype)
    if ab_dtype in fp8_dtypes:
        if d_dtype not in fp8_dtypes:
            raise ValueError("The MXFP8 JAX entry point requires FP8 output dtype")
        if backward and d_dtype is not cutlass.Float8E4M3FN:
            raise ValueError("d_dtype must be e4m3 for JAX backward; the packed backward quantizer does not support e5m2")
    elif d_dtype not in (cutlass.BFloat16, cutlass.Float16):
        raise ValueError("FP4 inputs require a BF16 or FP16 d_dtype")
    return ab_dtype, sf_dtype, k * (2 if ab_dtype is cutlass.Float4E2M1FN else 1)


def grouped_plan(api_type, inputs, outputs, *, mma_tiler_mn, cluster_shape_mn, discrete_col_sfd):
    if check_grouped_shapes(inputs, outputs, backward=True)[0] not in fp8_dtypes:
        raise ValueError("a and b must be MXFP8 e4m3 or e5m2; packed FP4 is unsupported")
    experts = inputs["b"].shape[0]
    margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
    config = (experts, mma_tiler_mn, cluster_shape_mn, margin, discrete_col_sfd)
    signature = tuple((name, tuple(t.shape), str(t.dtype)) for name, t in (*inputs.items(), *outputs.items()))
    validation_key = (config, signature)
    if validation_key not in validated_configs:
        samples = {f"sample_{name}": row_major_desc(t.shape, t.dtype, f"sample_{name}") for name, t in (*inputs.items(), *outputs.items())}
        api = api_type(**samples, sf_vec_size=32, mma_tiler_mn=mma_tiler_mn, cluster_shape_mn=cluster_shape_mn, discrete_col_sfd=discrete_col_sfd)
        api.check_support()
        if config not in kernel_cache:
            kwargs = dict(
                sf_vec_size=32,
                acc_dtype=cutlass.Float32,
                use_2cta_instrs=api.use_2cta_instrs,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=api.cluster_shape_mn,
                discrete_col_sfd=discrete_col_sfd,
                expert_cnt=experts,
                use_mono_increase_expert_idx=True,
                vectorized_f32=False,
            )
            mac = cutlass.utils.HardwareInfo().get_max_active_clusters(api.cluster_shape_mn[0] * api.cluster_shape_mn[1]) - margin
            if mac <= 0:
                raise ValueError("CUDNNFE_CLUSTER_OVERLAP_MARGIN leaves no active clusters")
            kernel_cache[config] = (api._kernel(**kwargs), mac)
        validated_configs.add(validation_key)
    return kernel_cache[config]


def check_jax_inputs(inputs):
    for name, tensor in inputs.items():
        argument = name if name == "padded_offsets" else f"{name}_tensor"
        if tensor is None:
            raise ValueError(f"{argument} is required for the JAX MXFP8 path")
        if detect_framework(tensor) != "jax":
            raise ValueError(f"{argument} must be a JAX array or tracer")
    if inputs["a"].ndim != 2 or inputs["b"].ndim != 3:
        raise ValueError("JAX requires canonical A (m,k) and B (experts,n,k) layouts")


def check_bf16_dense_inputs(inputs):
    """Validate the canonical BF16 dense operands shared by GLU and dGLU; return (m, k, experts, n)."""
    check_jax_inputs(inputs)
    a, b = inputs["a"], inputs["b"]
    m, k = a.shape
    experts, n, b_k = b.shape
    if m <= 0 or experts <= 0 or k != b_k:
        raise ValueError("Expected nonempty A/B with matching K and at least one expert")
    if m % 256:
        raise ValueError("A rows must be padded to a multiple of 256")
    if _convert_to_cutlass_data_type(b.dtype) is not cutlass.BFloat16:
        raise ValueError("a and b must both be bfloat16")
    for name, shape in (("alpha", (experts,)), ("beta", (experts,)), ("prob", (m,))):
        if name in inputs and (tuple(inputs[name].shape) != shape or _convert_to_cutlass_data_type(inputs[name].dtype) is not cutlass.Float32):
            raise ValueError(f"{name} must have shape {shape} and dtype float32")
    if inputs["padded_offsets"].shape != (experts,) or _convert_to_cutlass_data_type(inputs["padded_offsets"].dtype) is not cutlass.Int32:
        raise ValueError("padded_offsets must have shape (experts,) and dtype int32")
    if experts > 1024:
        raise ValueError(f"expert count must be <= 1024, got {experts}")
    return m, k, experts, n


@lru_cache(maxsize=128)
def grouped_call(adapter, kernel, mac, input_types, output_types, *, initialized=(), **scalars):
    """initialized: (output index, initializer) pairs for the outputs the kernel accumulates into."""
    return call(
        adapter,
        output_shape_dtype=output_types,
        input_spec=tuple(row_spec(t) for t in input_types),
        output_spec=tuple(row_spec(t) for t in output_types),
        initialized_outputs=dict(initialized) or None,
        kernel=kernel,
        mac=mac,
        **scalars,
    )


def check_jax_wrapper_options(
    *,
    acc_dtype,
    cd_major,
    sf_vec_size,
    vector_f32,
    m_aligned,
    current_stream,
    epilogue_op=None,
    dprob_tensor_buf=None,
    amax_tensor_buf=None,
    bias_tensor=None,
    b_ptrs=None,
    sfb_ptrs=None,
    b_major="k",
    use_dynamic_sched=False,
    use_single_group_runtime_offsets=False,
    sf_fp8_dtype_override=None,
    scheduler_counter_tensor=None,
    sf_vec_sizes=(32,),
):
    options = {
        "acc_dtype": acc_dtype is None or _convert_to_cutlass_data_type(acc_dtype) is cutlass.Float32,
        "cd_major": cd_major == "n",
        "sf_vec_size": sf_vec_size in sf_vec_sizes,
        "vector_f32": not vector_f32,
        "m_aligned": m_aligned == 256,
        "current_stream": current_stream is None,
        "epilogue_op": epilogue_op in (None, "none", "identity"),
        "dprob_tensor_buf": dprob_tensor_buf is None,
        "amax_tensor_buf": amax_tensor_buf is None,
        "bias_tensor": bias_tensor is None,
        "b_ptrs": b_ptrs is None,
        "sfb_ptrs": sfb_ptrs is None,
        "b_major": b_major == "k",
        "use_dynamic_sched": not use_dynamic_sched,
        "use_single_group_runtime_offsets": not use_single_group_runtime_offsets,
        "sf_fp8_dtype_override": sf_fp8_dtype_override is None,
        "scheduler_counter_tensor": scheduler_counter_tensor is None,
    }
    for name, supported in options.items():
        if not supported:
            raise ValueError(f"{name} is unsupported for the JAX MXFP8 path")


def check_rubin_dsl():
    requirement = cutedsl_arch_requirement_error((10, 7))
    if requirement:
        raise RuntimeError(requirement)
