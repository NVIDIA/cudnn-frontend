# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX-native (XLA custom call) entry point for grouped GEMM GLU + Hadamard + quant,
built on :func:`cudnn.jax.call`.

``grouped_gemm_glu_hadamard_quant_jax_forward`` is the canonical forward with dense
weights for packed FP4 inputs. Its NVFP4 outputs are packed too: uint8 arrays with
two values per byte. ``padded_offsets`` values cannot be host-validated under
tracing; malformed offsets are the caller's responsibility here (the eager wrapper
validates them).
"""

import os

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass.cute.nvgpu import OperandMajorMode

from cudnn.api_base import TupleDict, ceil_div, get_device_type
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.frost.buffers import cutedsl_state
from cudnn.tensor_adapter import get_compute_capability
from ..canonical import kernel_facing_b, kernel_facing_fp4, kernel_facing_mx, kernel_facing_prob
from ..canonical_jax import check_block_scaled_operands, check_jax_inputs, check_packed_sf_size, check_rubin_dsl, grouped_call, output_type, sf_array, sf_shape
from ..moe_utils import MoEWeightMode
from .moe_blockscaled_grouped_gemm_glu_hadamard_quant import BlockScaledMoEGroupedGemmGluHadamardQuantKernel

kernel_cache = {}
validated_configs = set()


def glu_hadamard_quant_kernel_type():
    if get_device_type() == "rubin":
        check_rubin_dsl()
        from .moe_blockscaled_grouped_gemm_glu_hadamard_quant_rubin import BlockScaledMoEGroupedGemmGluHadamardQuantKernel as RubinKernel

        return RubinKernel
    return BlockScaledMoEGroupedGemmGluHadamardQuantKernel


@cute.jit
def grouped_glu_hadamard_quant_adapter(
    stream,
    a,
    b,
    sfa,
    sfb,
    padded_offsets,
    alpha,
    prob,
    *outputs,
    kernel,
    mac,
    d_quant,
    rht_rowwise,
    rht_colwise,
    rht_quant,
    sf_cols,
    linear_offset,
    norm_const,
    rht_norm_const,
):
    # outputs: c, d, [sfd], then [rht, [sfrht]]
    d = kernel_facing_mx(kernel_facing_fp4(outputs[1]))
    # The kernel indexes the output scales through the MMA atom view (32, 4, rows, 4, cols, 1) of the
    # physical (1, rows, cols, 32, 4, 4) buffer.
    sf_rows = (d.shape[0] + 127) // 128
    sf_layout = cute.make_layout((32, 4, sf_rows, 4, sf_cols, 1), stride=(16, 4, 512 * sf_cols, 1, 512, 512 * sf_cols * sf_rows))
    sfd = cute.make_tensor(outputs[2].iterator, sf_layout) if cutlass.const_expr(d_quant) else None
    rest = outputs[2 + int(d_quant) :]
    rht = None
    sfrht = None
    if cutlass.const_expr(rht_rowwise):
        rht = kernel_facing_mx(kernel_facing_fp4(rest[0]))
        if cutlass.const_expr(rht_quant):
            sfrht = cute.make_tensor(rest[1].iterator, sf_layout)
    if cutlass.const_expr(rht_colwise):
        # Flat per-expert ragged data and scales.
        rht = kernel_facing_fp4(rest[0])
        sfrht = rest[1]
    kernel(
        a=kernel_facing_mx(kernel_facing_fp4(a)),
        b=kernel_facing_b(kernel_facing_fp4(b)),
        sfa=sfa,
        sfb=sfb,
        n=cutlass.Int32(0),
        k=cutlass.Int32(0),
        b_stride_size=cutlass.Int64(0),
        b_major_mode=OperandMajorMode.K,
        workspace_ptr=cute.make_ptr(cutlass.Uint8, 0, cute.AddressSpace.gmem, assumed_align=128),
        c=kernel_facing_mx(outputs[0]),
        d=d,
        sfd=sfd,
        rht=rht,
        sfrht=sfrht,
        padded_offsets=padded_offsets,
        alpha=alpha,
        prob=kernel_facing_prob(prob),
        bias=None,
        max_active_clusters=mac,
        stream=stream,
        linear_offset=cutlass.Float32(linear_offset),
        norm_const=cutlass.Float32(norm_const),
        rht_norm_const=cutlass.Float32(rht_norm_const),
    )


def glu_hadamard_quant_plan(inputs, mma_tiler_mn, cluster_shape_mn, sf_vec_size, act_func, d_quant, rht_rowwise, rht_colwise, rht_quant, glu_alpha, glu_limit):
    a, b = inputs["a"], inputs["b"]
    m, k = a.shape
    experts, n, b_k = b.shape
    if m <= 0 or experts <= 0 or k != b_k:
        raise ValueError("Expected nonempty A/B with matching K and at least one expert")
    if m % 256:
        raise ValueError("A rows must be padded to a multiple of 256")
    ab_dtype, sf_dtype = check_block_scaled_operands(a, b, inputs["sfa"], inputs["sfb"], sf_vec_size)
    if ab_dtype is not cutlass.Float4E2M1FN:
        raise ValueError("a and b must be packed FP4 as uint8; this fusion is wired for FP4 inputs only")
    k *= 2
    if act_func not in ("swiglu", "geglu", "srelu") or (act_func == "srelu" and (d_quant or rht_quant)):
        raise ValueError(f"act_func must be 'swiglu' or 'geglu', or 'srelu' without NVFP4 outputs, got {act_func!r}")
    if (d_quant or rht_quant) and sf_dtype is not cutlass.Float8E4M3FN:
        raise ValueError("NVFP4 outputs require E4M3 scale factors")
    if tuple(inputs["alpha"].shape) != (experts,) or _convert_to_cutlass_data_type(inputs["alpha"].dtype) is not cutlass.Float32:
        raise ValueError(f"alpha must have shape {(experts,)} and dtype float32")
    if inputs["prob"].shape != (m,) or _convert_to_cutlass_data_type(inputs["prob"].dtype) not in (cutlass.Float32, cutlass.BFloat16):
        raise ValueError("prob must have shape (m,) and dtype float32 or bfloat16")
    if inputs["padded_offsets"].shape != (experts,) or _convert_to_cutlass_data_type(inputs["padded_offsets"].dtype) is not cutlass.Int32:
        raise ValueError("padded_offsets must have shape (experts,) and dtype int32")
    kernel_type = glu_hadamard_quant_kernel_type()
    use_2cta_instrs = mma_tiler_mn[0] == 256
    cluster_shape_mn = tuple(cluster_shape_mn or ((2, 1) if use_2cta_instrs else (1, 1)))
    margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
    config = (get_device_type(), experts, tuple(mma_tiler_mn), cluster_shape_mn, margin, sf_vec_size, act_func, rht_rowwise, rht_colwise, glu_alpha, glu_limit)
    validation_key = (config, d_quant, rht_quant, tuple((name, tuple(t.shape), str(t.dtype)) for name, t in inputs.items()))
    if validation_key not in validated_configs:
        check_packed_sf_size(inputs, (("sfa", m, 1), ("sfb", n, experts)), ceil_div(ceil_div(k, sf_vec_size), 4))
        if experts > 1024:
            raise ValueError(f"expert count must be <= 1024, got {experts}")
        if not kernel_type.can_implement(
            ab_dtype,
            sf_dtype,
            sf_vec_size,
            cutlass.Float32,
            cutlass.Float4E2M1FN if d_quant else cutlass.BFloat16,
            use_2cta_instrs,
            tuple(mma_tiler_mn),
            cluster_shape_mn,
            m,
            n,
            k,
            experts,
            "k",
            "k",
            "n",
            kernel_type.FIX_PAD_SIZE,
            rht_quant=rht_quant,
            d_quant=d_quant,
        ):
            raise ValueError("Unsupported grouped GEMM GLU hadamard quant dtype, tile, cluster, alignment, or layout configuration")
        validated_configs.add(validation_key)
    if config not in kernel_cache:
        major, minor = get_compute_capability()
        if major < 10:
            raise RuntimeError(f"Grouped GEMM GLU hadamard quant requires SM100+, found SM{major}{minor}")
        mac = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1]) - margin
        if mac <= 0:
            raise ValueError("CUDNNFE_CLUSTER_OVERLAP_MARGIN leaves no active clusters")
        kernel = kernel_type(
            sf_vec_size=sf_vec_size,
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=tuple(mma_tiler_mn),
            cluster_shape_mn=cluster_shape_mn,
            vectorized_f32=False,
            expert_cnt=experts,
            weight_mode=MoEWeightMode.DENSE,
            act_func=act_func,
            rht_rowwise=rht_rowwise,
            rht_per_expert=rht_colwise,
            glu_alpha=glu_alpha,
            glu_limit=glu_limit,
        )
        kernel_cache[config] = (kernel, mac)
    return kernel_cache[config]


def grouped_gemm_glu_hadamard_quant_jax_forward(
    a_tensor,
    b_tensor,
    sfa_tensor,
    sfb_tensor,
    padded_offsets,
    alpha_tensor,
    prob_tensor,
    *,
    c_dtype=cutlass.BFloat16,
    d_dtype=cutlass.BFloat16,
    rht_rowwise_dtype=None,
    rht_colwise_dtype=None,
    norm_const=1.0,
    rht_norm_const=1.0,
    glu_alpha=None,
    glu_limit=None,
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=None,
    sf_vec_size=16,
    act_func="swiglu",
):
    """Canonical FP4 grouped GEMM + GLU + Hadamard + quant forward, eagerly or under jax.jit.

    Packed FP4 A (m,k/2) and dense B (experts,n,k/2) as uint8 with two values
    per byte, explicit alpha (experts,) fp32, prob (m,) fp32/bf16, and int32
    padded_offsets (experts,). Offsets must be nondecreasing multiples of 256
    within [0,m]; m is padded to 256. SF buffers contain packed MMA-tiled
    bytes at any dense rank. Output modes are dtype driven, with
    ``cutlass.Float4E2M1FN`` selecting NVFP4: ``d_dtype`` is BF16, giving D
    (m,n_out), or NVFP4, giving packed uint8 D (m,n_out/2) and the physical
    6-D E4M3 SFD. ``rht_rowwise_dtype`` (BF16 or NVFP4) adds the rowwise
    Hadamard output in the same layouts; ``rht_colwise_dtype`` (NVFP4 only)
    adds flat per-expert packed data (m*n_out/2,) and scales
    (m*n_out/sf_vec_size,). At most one RHT output may be requested. n_out is
    n/2, or n for ``act_func="srelu"``. ``norm_const``, ``rht_norm_const`` and
    the GLU scalars are compile-time constants of the traced call: pass Python
    numbers, since a traced JAX value raises ConcretizationTypeError. Returns a
    TupleDict with the eager wrapper's keys, including the pre-activation C
    (m,n). Rows at or past padded_offsets[-1] are unspecified, as in the torch
    path. SM100+, with the Rubin kernel on SM107, and the default 256x256
    tile. Needs
    nvidia-cutlass-dsl >= 4.8.
    """
    if not hasattr(cutlass, "FloatNV8E5M3FNU"):
        version = cutedsl_state()[1]
        found = "unknown version" if version is None else " ".join(version)
        raise RuntimeError(f"GLU + Hadamard + quant from JAX requires nvidia-cutlass-dsl >= 4.8; found {found}")
    d_quant = _convert_to_cutlass_data_type(d_dtype) is cutlass.Float4E2M1FN
    if rht_rowwise_dtype is not None and rht_colwise_dtype is not None:
        raise ValueError("Request at most one of rht_rowwise_dtype or rht_colwise_dtype")
    rht_dtype = rht_rowwise_dtype if rht_rowwise_dtype is not None else rht_colwise_dtype
    rht_quant = rht_dtype is not None and _convert_to_cutlass_data_type(rht_dtype) is cutlass.Float4E2M1FN
    for name, dtype, quant in (("d_dtype", d_dtype, d_quant), ("rht_rowwise_dtype", rht_rowwise_dtype, rht_quant)):
        if dtype is not None and not quant and _convert_to_cutlass_data_type(dtype) is not cutlass.BFloat16:
            raise ValueError(f"{name} must be bfloat16 or cutlass.Float4E2M1FN")
    if rht_colwise_dtype is not None and not rht_quant:
        raise ValueError("rht_colwise_dtype must be cutlass.Float4E2M1FN")
    inputs = dict(a=a_tensor, b=b_tensor, sfa=sfa_tensor, sfb=sfb_tensor, padded_offsets=padded_offsets, alpha=alpha_tensor, prob=prob_tensor)
    check_jax_inputs(inputs)
    inputs["sfa"] = sf_array(sfa_tensor)
    inputs["sfb"] = sf_array(sfb_tensor)
    m, n = a_tensor.shape[0], b_tensor.shape[1]
    n_out = n if act_func == "srelu" else n // 2
    kernel, mac = glu_hadamard_quant_plan(
        inputs,
        mma_tiler_mn,
        cluster_shape_mn,
        sf_vec_size,
        act_func,
        d_quant,
        rht_rowwise_dtype is not None,
        rht_colwise_dtype is not None,
        rht_quant,
        glu_alpha,
        glu_limit,
    )
    packed = output_type((m, n_out // 2), cutlass.Uint8)
    plain = output_type((m, n_out), cutlass.BFloat16)
    scales = output_type(sf_shape(m, n_out, sf_vec_size), cutlass.Float8E4M3FN)
    outputs = dict(c=output_type((m, n), c_dtype), d=packed if d_quant else plain)
    if d_quant:
        outputs["sfd"] = scales
    if rht_rowwise_dtype is not None:
        outputs["rht_rowwise"] = packed if rht_quant else plain
        if rht_quant:
            outputs["sfrht_rowwise"] = scales
    if rht_colwise_dtype is not None:
        outputs["rht_colwise"] = output_type((m * n_out // 2,), cutlass.Uint8)
        outputs["sfrht_colwise"] = output_type((m * n_out // sf_vec_size,), cutlass.Float8E4M3FN)
    result = grouped_call(
        grouped_glu_hadamard_quant_adapter,
        kernel,
        mac,
        tuple(output_type(t.shape, t.dtype) for t in inputs.values()),
        tuple(outputs.values()),
        d_quant=d_quant,
        rht_rowwise=rht_rowwise_dtype is not None,
        rht_colwise=rht_colwise_dtype is not None,
        rht_quant=rht_quant,
        sf_cols=scales.shape[2],
        linear_offset=1.0 if act_func == "geglu" else 0.0,
        norm_const=float(norm_const),
        rht_norm_const=float(rht_norm_const),
    )(*inputs.values())
    result = dict(zip(outputs, result))
    return TupleDict(
        c_tensor=result["c"],
        d_tensor=result["d"],
        sfd_tensor=result.get("sfd"),
        rht_rowwise_tensor=result.get("rht_rowwise"),
        sfrht_rowwise_tensor=result.get("sfrht_rowwise"),
        rht_colwise_tensor=result.get("rht_colwise"),
        sfrht_colwise_tensor=result.get("sfrht_colwise"),
    )
