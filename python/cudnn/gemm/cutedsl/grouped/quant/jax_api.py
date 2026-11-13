# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX-native (XLA custom call) entry point for grouped GEMM quant, built on
:func:`cudnn.jax.call`.

``grouped_gemm_quant_jax_forward`` is the canonical block-scaled grouped GEMM with
dense weights: MXFP8 or packed FP4 A ``(m, k)``, B ``(experts, n, k)`` or column-wise
``(experts, k, n)``, and already MMA-packed scale bytes, running the SM100 or Rubin
(SM107) block-scaled quant kernel. ``padded_offsets`` values cannot be host-validated under tracing;
malformed offsets are the caller's responsibility here (the eager wrapper
validates them).
"""

import os

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass.cute.nvgpu import OperandMajorMode

from cudnn.api_base import TupleDict, ceil_div, get_device_type
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.tensor_adapter import detect_framework, get_compute_capability
from cudnn.jax import neg_inf_init
from ..canonical import kernel_facing_fp4
from ..canonical_jax import check_block_scaled_operands, check_packed_sf_size, check_rubin_dsl, fp8_dtypes, grouped_call, output_type, sf_array, sf_shape
from ..moe_utils import MoEWeightMode
from .grouped_gemm_quant import BlockScaledMoEGroupedGemmQuantKernel

kernel_cache = {}
validated_configs = set()


def quant_kernel_type():
    if get_device_type() == "rubin":
        check_rubin_dsl()
        from .moe_blockscaled_grouped_gemm_quant_rubin import BlockScaledMoEGroupedGemmQuantKernel as RubinKernel

        return RubinKernel
    return BlockScaledMoEGroupedGemmQuantKernel


@cute.jit
def grouped_quant_adapter(stream, a, b, sfa, sfb, padded_offsets, alpha, *rest, kernel, mac, rubin, n_major, has_prob, low_precision, has_amax):
    # rest: [prob], [norm_const], d, then d_col, sfd_row, sfd_col for FP8 output or [amax] otherwise
    a = kernel_facing_fp4(a)
    b = kernel_facing_fp4(b)
    amax = rest[-1] if cutlass.const_expr(has_amax) else None
    prob = rest[0] if cutlass.const_expr(has_prob) else None
    outputs = rest[int(has_prob) + int(low_precision) :]
    norm_const = rest[int(has_prob)] if cutlass.const_expr(low_precision) else None
    d = outputs[0]
    # The kernel always builds the D_col store descriptor; without quantization it is D, as in the torch path.
    d_col = outputs[1] if cutlass.const_expr(low_precision) else d
    sfd_row = outputs[2] if cutlass.const_expr(low_precision) else None
    sfd_col = outputs[3] if cutlass.const_expr(low_precision) else None
    b_major_mode = OperandMajorMode.MN if cutlass.const_expr(n_major) else OperandMajorMode.K
    workspace_ptr = cute.make_ptr(cutlass.Uint8, 0, cute.AddressSpace.gmem, assumed_align=128)
    if cutlass.const_expr(rubin):
        kernel(
            a=a,
            b=b,
            sfb=sfb,
            n=cutlass.Int32(0),
            k=cutlass.Int32(0),
            b_stride_size=cutlass.Int64(0),
            b_major_mode=b_major_mode,
            workspace_ptr=workspace_ptr,
            c=d,
            d=d,
            d_col=d_col,
            sfa=sfa,
            sfd_row_tensor=sfd_row,
            sfd_col_tensor=sfd_col,
            amax_tensor=amax,
            norm_const_tensor=norm_const,
            padded_offsets=padded_offsets,
            alpha=alpha,
            bias=None,
            prob=prob,
            max_active_clusters=mac,
            stream=stream,
        )
    else:
        kernel(
            a=a,
            b=b,
            sfb=sfb,
            n=cutlass.Int32(0),
            k=cutlass.Int32(0),
            b_stride_size=cutlass.Int64(0),
            b_major_mode=b_major_mode,
            workspace_ptr=workspace_ptr,
            d=d,
            d_col=d_col,
            sfa=sfa,
            sfd_row_tensor=sfd_row,
            sfd_col_tensor=sfd_col,
            amax_tensor=amax,
            norm_const_tensor=norm_const,
            padded_offsets=padded_offsets,
            alpha=alpha,
            row_scale=None,
            bias=None,
            prob=prob,
            max_active_clusters=mac,
            stream=stream,
        )


def quant_plan(inputs, d, b_major, mma_tiler_mn, cluster_shape_mn, discrete_col_sfd, sf_vec_size):
    a, b = inputs["a"], inputs["b"]
    if b_major not in ("k", "n"):
        raise ValueError(f"b_major must be 'k' or 'n', got {b_major!r}")
    if a.ndim != 2 or b.ndim != 3:
        raise ValueError("JAX requires canonical A (m,k) and B (experts,n,k), or (experts,k,n) with b_major='n'")
    m, k = a.shape
    experts = b.shape[0]
    n, b_k = (b.shape[2], b.shape[1]) if b_major == "n" else b.shape[1:]
    if m <= 0 or experts <= 0 or k != b_k:
        raise ValueError("Expected nonempty A/B with matching K and at least one expert")
    if m % 256:
        raise ValueError("A rows must be padded to a multiple of 256")
    ab_dtype, sf_dtype = check_block_scaled_operands(a, b, inputs["sfa"], inputs["sfb"], sf_vec_size)
    if ab_dtype is cutlass.Float4E2M1FN:
        if b_major != "k":
            raise ValueError("b_major must be 'k' for packed FP4")
        k *= 2
    for name, shape in (("alpha", (experts,)), ("norm_const", (1,))):
        if name in inputs and (tuple(inputs[name].shape) != shape or _convert_to_cutlass_data_type(inputs[name].dtype) is not cutlass.Float32):
            raise ValueError(f"{name} must have shape {shape} and dtype float32")
    if "prob" in inputs and (inputs["prob"].shape != (m,) or _convert_to_cutlass_data_type(inputs["prob"].dtype) not in (cutlass.Float32, cutlass.BFloat16)):
        raise ValueError("prob must have shape (m,) and dtype float32 or bfloat16")
    if inputs["padded_offsets"].shape != (experts,) or _convert_to_cutlass_data_type(inputs["padded_offsets"].dtype) is not cutlass.Int32:
        raise ValueError("padded_offsets must have shape (experts,) and dtype int32")
    d_dtype = _convert_to_cutlass_data_type(d.dtype)
    low_precision = d_dtype in fp8_dtypes
    if low_precision and ab_dtype not in fp8_dtypes:
        raise ValueError("FP4 inputs require a BF16 or FP16 d_dtype")
    use_2cta_instrs = mma_tiler_mn[0] == 256
    cluster_shape_mn = tuple(cluster_shape_mn or ((2, 1) if use_2cta_instrs else (1, 1)))
    margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
    config = (get_device_type(), experts, tuple(mma_tiler_mn), cluster_shape_mn, margin, discrete_col_sfd, low_precision, sf_vec_size)
    validation_key = (config, b_major, str(d.dtype), tuple((name, tuple(t.shape), str(t.dtype)) for name, t in inputs.items()))
    if validation_key not in validated_configs:
        check_packed_sf_size(inputs, (("sfa", m, 1), ("sfb", n, experts)), ceil_div(ceil_div(k, sf_vec_size), 4))
        if experts > 1024:
            raise ValueError(f"expert count must be <= 1024, got {experts}")
        kernel_type = quant_kernel_type()
        # can_implement models K-major B only; column-wise B additionally needs a 16-byte row stride.
        if (b_major == "n" and n % 16) or not kernel_type.can_implement(
            ab_dtype,
            sf_dtype,
            sf_vec_size,
            cutlass.Float32,
            d_dtype,
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
        ):
            raise ValueError("Unsupported grouped GEMM quant dtype, tile, cluster, alignment, or layout configuration")
        validated_configs.add(validation_key)
    if config not in kernel_cache:
        major, minor = get_compute_capability()
        if major * 10 + minor < 100:
            raise RuntimeError(f"Grouped GEMM quant requires SM100+ compute capability, but found SM{major}{minor}")
        mac = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1]) - margin
        if mac <= 0:
            raise ValueError("CUDNNFE_CLUSTER_OVERLAP_MARGIN leaves no active clusters")
        kernel = quant_kernel_type()(
            sf_vec_size=sf_vec_size,
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=tuple(mma_tiler_mn),
            cluster_shape_mn=cluster_shape_mn,
            vectorized_f32=False,
            generate_sfd=low_precision,
            discrete_col_sfd=discrete_col_sfd,
            enable_bias=False,
            expert_cnt=experts,
            weight_mode=MoEWeightMode.DENSE,
            **({"generate_c": False} if get_device_type() == "rubin" else {}),
        )
        kernel_cache[config] = (kernel, mac)
    return kernel_cache[config]


def grouped_gemm_quant_jax_forward(
    a_tensor,
    b_tensor,
    sfa_tensor,
    sfb_tensor,
    padded_offsets,
    alpha_tensor,
    prob_tensor=None,
    norm_const_tensor=None,
    *,
    d_dtype=cutlass.BFloat16,
    b_major="k",
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=None,
    discrete_col_sfd=False,
    sf_vec_size=32,
    generate_amax=False,
):
    """Canonical block-scaled grouped GEMM with optional output quantization, eagerly or under jax.jit.

    A (m,k), dense B (experts,n,k) or column-wise (experts,k,n) with
    ``b_major="n"``, explicit alpha (experts,) fp32, int32 padded_offsets
    (experts,), and optional prob (m,) fp32/bf16. Offsets must be nondecreasing
    multiples of 256 within [0,m]; m is padded to 256. A/B are MXFP8, or packed
    FP4 as uint8 with two values per byte (K is twice the array extent; K-major
    B only). SF buffers contain packed MMA-tiled bytes at any dense rank: E8M0
    with ``sf_vec_size=32`` for MXFP8 (uint8 bit patterns also accepted), E8M0
    or E4M3 (NVFP4, ``sf_vec_size=16``) for FP4. With a BF16/FP16 ``d_dtype``
    only D (m,n) is produced, plus the per-expert amax (experts,1) with
    ``generate_amax``. With an FP8 ``d_dtype``, which needs MXFP8 inputs,
    norm_const (1,) fp32 is required and D, D_col and the physical 6-D SFD
    buffers are produced; ``discrete_col_sfd=True`` packs column scales by
    expert. Returns a TupleDict with the eager wrapper's keys. Outputs are not
    initialized, so rows at or past padded_offsets[-1] are unspecified, as in
    the torch path. Runs the Rubin quant kernel on SM107 and the SM100 kernel
    otherwise.
    """
    d_dtype = _convert_to_cutlass_data_type(d_dtype)
    low_precision = d_dtype in fp8_dtypes
    has_amax = generate_amax and not low_precision
    if low_precision and norm_const_tensor is None:
        raise ValueError("norm_const_tensor is required when FP8 inputs are used with FP8 output")
    inputs = dict(a=a_tensor, b=b_tensor, sfa=sfa_tensor, sfb=sfb_tensor, padded_offsets=padded_offsets, alpha=alpha_tensor)
    if prob_tensor is not None:
        inputs["prob"] = prob_tensor
    if low_precision:
        inputs["norm_const"] = norm_const_tensor
    for name, tensor in inputs.items():
        argument = name if name == "padded_offsets" else f"{name}_tensor"
        if tensor is None:
            raise ValueError(f"{argument} is required for the JAX MXFP8 path")
        if detect_framework(tensor) != "jax":
            raise ValueError(f"{argument} must be a JAX array or tracer")
    inputs["sfa"] = sf_array(sfa_tensor)
    inputs["sfb"] = sf_array(sfb_tensor)
    m = a_tensor.shape[0]
    n = b_tensor.shape[-1 if b_major == "n" else -2]
    outputs = dict(d=output_type((m, n), d_dtype))
    if low_precision:
        outputs.update(
            d_col=output_type((m, n), d_dtype),
            sfd_row=output_type(sf_shape(m, n, sf_vec_size), cutlass.Float8E8M0FNU),
            sfd_col=output_type(sf_shape(n, m, sf_vec_size), cutlass.Float8E8M0FNU),
        )
    elif has_amax:
        outputs["amax"] = output_type((b_tensor.shape[0], 1), cutlass.Float32)
    kernel, mac = quant_plan(inputs, outputs["d"], b_major, mma_tiler_mn, cluster_shape_mn, discrete_col_sfd, sf_vec_size)
    result = grouped_call(
        grouped_quant_adapter,
        kernel,
        mac,
        tuple(output_type(t.shape, t.dtype) for t in inputs.values()),
        tuple(outputs.values()),
        initialized=((1, neg_inf_init),) if has_amax else (),
        rubin=get_device_type() == "rubin",
        n_major=b_major == "n",
        has_prob=prob_tensor is not None,
        low_precision=low_precision,
        has_amax=has_amax,
    )(*inputs.values())
    return TupleDict(
        d_tensor=result[0],
        d_col_tensor=result[1] if low_precision else None,
        amax_tensor=result[1] if has_amax else None,
        sfd_row_tensor=result[2] if low_precision else None,
        sfd_col_tensor=result[3] if low_precision else None,
    )
