# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX-native (XLA custom call) entry points for grouped GEMM dGLU, built on
:func:`cudnn.jax.call`.

``grouped_gemm_dglu_jax_sm100`` serves two modes. With ``b_tensor`` it is the
canonical MXFP8 backward: dense A ``(m, k)``, B ``(experts, n, k)``, saved C
``(m, 2n)`` and already MMA-packed scale bytes, running the SM100 or Rubin (SM107)
block-scaled dGLU kernel. With ``b_ptrs`` it is the BF16 backward in discrete
weight mode. The per-expert weight pointers travel as a regular device array
whose *values* are raw addresses — the referenced weight buffers are not visible
to XLA, so the caller must keep them alive (and unmoved) across every execution
of the traced computation. ``dprob`` and ``dbias`` are kernel-accumulated
outputs, so unlike the eager wrapper they are not caller-provided buffers here:
both are donated zero-initialized outputs of the custom call. ``padded_offsets`` values cannot be
host-validated under tracing; malformed offsets are the caller's responsibility
here (the eager wrapper validates them).
"""

import math
import os
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass.cute.nvgpu import OperandMajorMode

from cudnn.api_base import TupleDict, ceil_div, get_device_type
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.tensor_adapter import framework_dtype, get_compute_capability
from cudnn.jax import call, gemm_operand_spec, zeros_init
from ..canonical import kernel_facing_b, kernel_facing_mx, kernel_facing_prob
from ..canonical_jax import check_grouped_shapes, check_jax_inputs, grouped_call, output_type, sf_array, sf_shape
from ..moe_utils import MoEWeightMode
from ..unfused.jax_api import _pointer_count, _prob_spec
from .moe_blockscaled_grouped_gemm_dglu_dbias import BlockScaledMoEGroupedGemmDgluDbiasKernel
from .moe_grouped_gemm_dglu_dbias import MoEGroupedGemmDgluDbiasBf16Kernel

# cache_key -> (kernel instance, max_active_clusters, workspace_bytes); reusing the
# instance keeps cutlass_call's compile cache warm (its FunctionSpec keys on the
# constexpr kwargs).
_kernel_cache: dict = {}

_output_dtypes = (cutlass.BFloat16, cutlass.Float16, cutlass.Float32)


@cute.jit
def _grouped_dglu_bf16_adapter(stream, a, c, b_ptrs, padded_offsets, alpha, beta, prob, d, dprob, workspace, *, kernel, n, k, mac, linear_offset):
    # Discrete-mode b is a raw pointer to the device int64[] of per-expert base
    # addresses; the packed uint8 (or int64) input buffer recasts for free.
    b_arg = cute.recast_ptr(b_ptrs.iterator, dtype=cutlass.Int64)
    kernel(
        a=a,
        b=b_arg,
        n=cutlass.Int32(n),
        k=cutlass.Int32(k),
        b_stride_size=cutlass.Int64(k),  # uniform k-major per-expert (n, k) weights
        b_major_mode=OperandMajorMode.K,
        workspace_ptr=workspace.iterator,
        c=c,
        d=d,
        padded_offsets=padded_offsets,
        alpha=alpha,
        beta=beta,
        prob=prob,
        dprob=dprob,
        linear_offset=cutlass.Float32(linear_offset),
        dbias_tensor=None,
        max_active_clusters=mac,
        stream=stream,
    )


@cute.jit
def _grouped_dglu_bf16_dbias_adapter(stream, a, c, b_ptrs, padded_offsets, alpha, beta, prob, d, dprob, dbias, workspace, *, kernel, n, k, mac, linear_offset):
    b_arg = cute.recast_ptr(b_ptrs.iterator, dtype=cutlass.Int64)
    kernel(
        a=a,
        b=b_arg,
        n=cutlass.Int32(n),
        k=cutlass.Int32(k),
        b_stride_size=cutlass.Int64(k),  # uniform k-major per-expert (n, k) weights
        b_major_mode=OperandMajorMode.K,
        workspace_ptr=workspace.iterator,
        c=c,
        d=d,
        padded_offsets=padded_offsets,
        alpha=alpha,
        beta=beta,
        prob=prob,
        dprob=dprob,
        linear_offset=cutlass.Float32(linear_offset),
        dbias_tensor=dbias,
        max_active_clusters=mac,
        stream=stream,
    )


def grouped_gemm_dglu_jax_sm100(
    a_tensor: Any,
    c_tensor: Any,
    padded_offsets: Any,
    alpha_tensor: Any,
    beta_tensor: Any,
    b_ptrs: Any = None,
    n: Optional[int] = None,
    prob_tensor: Any = None,
    d_dtype: Any = cutlass.BFloat16,
    acc_dtype: Any = cutlass.Float32,
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    vector_f32: bool = False,
    act_func: str = "dswiglu",
    linear_offset: Optional[float] = None,
    generate_dbias: bool = False,
    use_dynamic_sched: bool = False,
    *,
    b_tensor: Any = None,
    sfa_tensor: Any = None,
    sfb_tensor: Any = None,
    norm_const_tensor: Any = None,
    discrete_col_sfd: bool = False,
    geglu_alpha: float = 1.702,
    glu_clamp_max: float = 7.0,
    glu_clamp_min: float = -7.0,
) -> Any:
    """Grouped GEMM dGLU backward as an XLA custom call.

    With ``b_tensor``: canonical MXFP8 dense weights, eagerly or under jax.jit.
    A (m,k), B (experts,n,k), saved C (m,2n), prob (m,) fp32/bf16, explicit
    alpha/beta (experts,) fp32, norm_const (1,) fp32, and int32 padded_offsets
    (experts,). Offsets must be nondecreasing multiples of 256 within [0,m]; m
    is padded to 256. SF buffers contain packed E8M0 MMA-tiled bytes, at any
    dense rank (uint8 bit patterns also accepted). FP8 A/B and an e4m3
    ``d_dtype`` are required. ``act_func`` is "dswiglu" or "dgeglu"; the
    activation scalars are compile-time constants of the traced call.
    ``discrete_col_sfd=True`` packs column scales by expert. Returns a
    TupleDict with the eager wrapper's keys: natural 2-D D_row/D_col, dprob
    (m,) and physical 6-D SF buffers. Only dprob is zero-initialized, for
    accumulation; other rows at or past padded_offsets[-1] are unspecified.
    Runs the Rubin dGLU kernel on SM107 and the SM100 kernel otherwise.

    With ``b_ptrs``: BF16 discrete weights.

    Same contract as the eager wrapper's BF16 discrete mode: A ``(m, k, 1)`` k-major
    C-contiguous bfloat16, C ``(m, 2n, 1)`` n-major forward pre-activations,
    ``padded_offsets (experts,)`` int32 cumulative 256-aligned row offsets, ``alpha``
    and ``beta`` ``(experts,)`` float32, ``prob (m, 1, 1)`` float32, and ``b_ptrs``
    holding per-expert ``(n, k)`` k-major bfloat16 weight base addresses (packed
    little-endian uint8, 8 bytes per pointer — or int64 with x64 mode). ``n`` is the
    per-expert weight N (half the pre-activation width). ``linear_offset`` defaults
    per ``act_func`` (1.0 for ``"dgeglu"``, 0.0 for ``"dswiglu"``) and is a
    compile-time constant of the traced call. Returns ``(d_row_tensor, dprob_tensor,
    dbias_tensor)`` with ``dbias_tensor`` None unless ``generate_dbias``; rows
    at/past ``padded_offsets[-1]`` come back zero-filled (the outputs are donated
    zero-initialized buffers, matching the eager contract of a caller-zeroed
    ``dprob``).
    """
    if b_tensor is not None:
        unsupported = {
            "b_ptrs": b_ptrs is not None,
            "n": n is not None,
            "acc_dtype": _convert_to_cutlass_data_type(acc_dtype) is not cutlass.Float32,
            "vector_f32": vector_f32,
            "use_dynamic_sched": use_dynamic_sched,
            "generate_dbias": generate_dbias,
        }
        for name, rejected in unsupported.items():
            if rejected:
                raise ValueError(f"{name} is unsupported for the JAX MXFP8 path")
        return blockscaled_dglu_jax(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            c_tensor=c_tensor,
            sfa_tensor=sfa_tensor,
            sfb_tensor=sfb_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            beta_tensor=beta_tensor,
            prob_tensor=prob_tensor,
            norm_const_tensor=norm_const_tensor,
            d_dtype=d_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            discrete_col_sfd=discrete_col_sfd,
            act_func=act_func,
            linear_offset=linear_offset,
            geglu_alpha=geglu_alpha,
            glu_clamp_max=glu_clamp_max,
            glu_clamp_min=glu_clamp_min,
        )
    if b_ptrs is None or prob_tensor is None or any(t is not None for t in (sfa_tensor, sfb_tensor, norm_const_tensor)):
        raise ValueError("BF16 discrete weights take b_ptrs and prob_tensor and no scale tensors; pass b_tensor for canonical MXFP8")
    d_dtype = _convert_to_cutlass_data_type(d_dtype)
    acc_dtype = _convert_to_cutlass_data_type(acc_dtype)

    if len(a_tensor.shape) != 3 or a_tensor.shape[2] != 1:
        raise ValueError(f"a_tensor must have shape (m, k, 1), got {tuple(a_tensor.shape)}")
    m, k, _ = a_tensor.shape
    if m % 256 != 0:
        raise ValueError(f"a_tensor M dimension must be 256-aligned, got {m}")
    if _convert_to_cutlass_data_type(a_tensor.dtype) is not cutlass.BFloat16:
        raise ValueError(f"BF16 discrete a_tensor must have dtype bfloat16, got {a_tensor.dtype}; use b_tensor for dense MXFP8")
    if n is None or n <= 0 or n % 32 != 0:
        raise ValueError(f"n must be positive and divisible by 32, got {n}")
    two_n = 2 * n
    if tuple(c_tensor.shape) != (m, two_n, 1):
        raise ValueError(f"c_tensor must have shape ({m}, {two_n}, 1), got {tuple(c_tensor.shape)}")
    c_dtype = _convert_to_cutlass_data_type(c_tensor.dtype)
    if c_dtype not in _output_dtypes or d_dtype not in _output_dtypes:
        raise ValueError(f"BF16 discrete c_tensor/d_dtype must be BF16, FP16, or FP32, got {c_dtype}/{d_dtype}")
    if acc_dtype is not cutlass.Float32:
        raise ValueError(f"acc_dtype must be float32, got {acc_dtype}")
    if act_func not in ("dswiglu", "dgeglu"):
        raise ValueError(f"act_func must be 'dswiglu' or 'dgeglu', got {act_func}")
    if linear_offset is None:
        linear_offset = 1.0 if act_func == "dgeglu" else 0.0

    expert_cnt = _pointer_count(b_ptrs)
    if expert_cnt <= 0 or expert_cnt > 1024:
        raise ValueError(f"expert count must be in [1, 1024], got {expert_cnt}")
    if tuple(padded_offsets.shape) != (expert_cnt,):
        raise ValueError(f"padded_offsets must have shape ({expert_cnt},), got {tuple(padded_offsets.shape)}")
    if _convert_to_cutlass_data_type(padded_offsets.dtype) is not cutlass.Int32:
        raise ValueError(f"padded_offsets must have dtype int32, got {padded_offsets.dtype}")
    if tuple(alpha_tensor.shape) != (expert_cnt,) or _convert_to_cutlass_data_type(alpha_tensor.dtype) is not cutlass.Float32:
        raise ValueError(f"alpha_tensor must be ({expert_cnt},) float32, got {tuple(alpha_tensor.shape)} {alpha_tensor.dtype}")
    if tuple(beta_tensor.shape) != (expert_cnt,) or _convert_to_cutlass_data_type(beta_tensor.dtype) is not cutlass.Float32:
        raise ValueError(f"beta_tensor must be ({expert_cnt},) float32, got {tuple(beta_tensor.shape)} {beta_tensor.dtype}")
    if tuple(prob_tensor.shape) != (m, 1, 1) or _convert_to_cutlass_data_type(prob_tensor.dtype) is not cutlass.Float32:
        raise ValueError(f"prob_tensor must be ({m}, 1, 1) float32, got {tuple(prob_tensor.shape)} {prob_tensor.dtype}")

    use_2cta_instrs = mma_tiler_mn[0] == 256
    cluster_shape_mn = tuple(cluster_shape_mn or ((2, 1) if use_2cta_instrs else (1, 1)))

    if not MoEGroupedGemmDgluDbiasBf16Kernel.can_implement(
        cutlass.BFloat16,
        c_dtype,
        d_dtype,
        acc_dtype,
        use_2cta_instrs,
        tuple(mma_tiler_mn),
        cluster_shape_mn,
        m,
        n,
        k,
        expert_cnt,
        "k",
        "k",
        "n",
        MoEGroupedGemmDgluDbiasBf16Kernel.FIX_PAD_SIZE,
        act_func,
    ):
        raise ValueError("Unsupported BF16 grouped GEMM dGLU tile, cluster, alignment, or layout configuration")

    cache_key = (
        expert_cnt,
        c_dtype,
        d_dtype,
        acc_dtype,
        tuple(mma_tiler_mn),
        cluster_shape_mn,
        vector_f32,
        act_func,
        use_dynamic_sched,
    )
    entry = _kernel_cache.get(cache_key)
    if entry is None:
        kernel = MoEGroupedGemmDgluDbiasBf16Kernel(
            acc_dtype=acc_dtype,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=tuple(mma_tiler_mn),
            cluster_shape_mn=cluster_shape_mn,
            vectorized_f32=vector_f32,
            expert_cnt=expert_cnt,
            weight_mode=MoEWeightMode.DISCRETE,
            use_dynamic_sched=use_dynamic_sched,
            act_func=act_func,
        )
        overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
        mac = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1]) - overlap_margin
        if mac <= 0:
            raise ValueError("max_active_clusters must be > 0 after applying CUDNNFE_CLUSTER_OVERLAP_MARGIN")
        entry = (kernel, mac, max(kernel.get_workspace_bytes(), 1))
        _kernel_cache[cache_key] = entry
    kernel, mac, workspace_bytes = entry

    operand = gemm_operand_spec()
    prob_spec = _prob_spec()
    output_shape_dtype = [
        jax.ShapeDtypeStruct((m, two_n, 1), framework_dtype(d_dtype, "jax")),
        jax.ShapeDtypeStruct((m, 1, 1), jnp.float32),  # dprob (kernel-accumulated)
    ]
    output_spec = [operand, prob_spec]
    if generate_dbias:
        output_shape_dtype.append(jax.ShapeDtypeStruct((expert_cnt, two_n, 1), framework_dtype(cutlass.BFloat16, "jax")))
        output_spec.append(operand)
    output_shape_dtype.append(jax.ShapeDtypeStruct((workspace_bytes,), jnp.uint8))
    output_spec.append(None)

    results = call(
        _grouped_dglu_bf16_dbias_adapter if generate_dbias else _grouped_dglu_bf16_adapter,
        output_shape_dtype=tuple(output_shape_dtype),
        input_spec=(operand, operand, None, None, None, None, prob_spec),
        output_spec=tuple(output_spec),
        # All outputs donated: d for the trailing-unit-dim layout spec (and defined
        # bytes past the last offset); dprob/dbias because the kernel accumulates
        # into them (atomic add) and expects zeroed buffers; the workspace because
        # the helper kernel writes the per-expert TMA descriptors into it (XLA
        # inputs are immutable).
        initialized_outputs={index: zeros_init for index in range(len(output_shape_dtype))},
        kernel=kernel,
        n=int(n),
        k=int(k),
        mac=mac,
        linear_offset=float(linear_offset),
    )(a_tensor, c_tensor, b_ptrs, padded_offsets, alpha_tensor, beta_tensor, prob_tensor)

    if generate_dbias:
        d_row_tensor, dprob_tensor, dbias_tensor, _workspace = results
        return d_row_tensor, dprob_tensor, dbias_tensor
    d_row_tensor, dprob_tensor, _workspace = results
    return d_row_tensor, dprob_tensor, None


kernel_cache = {}
validated_configs = set()


def dglu_kernel_type():
    if get_device_type() == "rubin":
        from .moe_blockscaled_grouped_gemm_dglu_rubin import BlockScaledMoEGroupedGemmDgluKernel

        return BlockScaledMoEGroupedGemmDgluKernel
    return BlockScaledMoEGroupedGemmDgluDbiasKernel


@cute.jit
def grouped_dglu_adapter(
    stream,
    a,
    b,
    c,
    sfa,
    sfb,
    offsets,
    alpha,
    beta,
    prob,
    norm_const,
    d_row,
    d_col,
    dprob,
    sfd_row,
    sfd_col,
    *,
    kernel,
    mac,
    rubin,
    linear_offset,
    geglu_alpha,
    glu_clamp_max,
    glu_clamp_min,
):
    # Metadata-only CuTe views preserve the shared kernel ABI without data copies.
    kernel(
        a=kernel_facing_mx(a),
        b=kernel_facing_b(b),
        sfb=sfb,
        n=cutlass.Int32(0),
        k=cutlass.Int32(0),
        b_stride_size=cutlass.Int64(0),
        b_major_mode=OperandMajorMode.K,
        workspace_ptr=cute.make_ptr(cutlass.Uint8, 0, cute.AddressSpace.gmem, assumed_align=128),
        c=kernel_facing_mx(c),
        d=kernel_facing_mx(d_row),
        d_col=kernel_facing_mx(d_col),
        sfa=sfa,
        sfd_row_tensor=sfd_row,
        sfd_col_tensor=sfd_col,
        amax_tensor=None,
        norm_const_tensor=norm_const,
        padded_offsets=offsets,
        alpha=alpha,
        beta=beta,
        prob=kernel_facing_prob(prob),
        dprob=kernel_facing_prob(dprob),
        dbias_tensor=None,
        max_active_clusters=mac,
        stream=stream,
        linear_offset=cutlass.Float32(linear_offset) if cutlass.const_expr(rubin) else linear_offset,
        geglu_alpha=geglu_alpha,
        glu_clamp_max=glu_clamp_max,
        glu_clamp_min=glu_clamp_min,
    )


def dglu_plan(inputs, outputs, mma_tiler_mn, cluster_shape_mn, discrete_col_sfd, act_func):
    check_grouped_shapes(inputs, outputs, backward=True)
    if act_func not in ("dswiglu", "dgeglu"):
        raise ValueError(f"act_func must be 'dswiglu' or 'dgeglu' for the JAX MXFP8 path, got {act_func!r}")
    m, k = inputs["a"].shape
    experts, n, _ = inputs["b"].shape
    if inputs["c"].shape != (m, 2 * n):
        raise ValueError(f"C must have shape {(m, 2 * n)}, got {inputs['c'].shape}")
    if _convert_to_cutlass_data_type(inputs["c"].dtype) not in (cutlass.BFloat16, cutlass.Float16):
        raise ValueError("C must be BF16 or FP16")
    if _convert_to_cutlass_data_type(inputs["a"].dtype) is not _convert_to_cutlass_data_type(inputs["b"].dtype):
        raise ValueError("A and B must have the same MXFP8 dtype")
    if _convert_to_cutlass_data_type(inputs["prob"].dtype) not in (cutlass.Float32, cutlass.BFloat16):
        raise ValueError("prob must be float32 or bfloat16")
    if experts > 1024:
        raise ValueError("expert count must be <= 1024")
    use_2cta = mma_tiler_mn[0] == 256
    cluster = tuple(cluster_shape_mn or ((2, 1) if use_2cta else (1, 1)))
    margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
    key = (get_device_type(), experts, tuple(mma_tiler_mn), cluster, margin, discrete_col_sfd, act_func)
    signature = (key, tuple((name, tuple(t.shape), str(t.dtype)) for name, t in (*inputs.items(), *outputs.items())))
    if signature not in validated_configs:
        for name, rows, groups in (("sfa", m, 1), ("sfb", n, experts)):
            expected = groups * 512 * ceil_div(rows, 128) * ceil_div(ceil_div(k, 32), 4)
            if math.prod(inputs[name].shape) != expected:
                raise ValueError(f"{name.upper()} must contain the complete MMA-packed scale buffer")
        kernel_type = dglu_kernel_type()
        if not kernel_type.can_implement(
            _convert_to_cutlass_data_type(inputs["a"].dtype),
            cutlass.Float8E8M0FNU,
            32,
            cutlass.Float32,
            _convert_to_cutlass_data_type(outputs["d_row"].dtype),
            use_2cta,
            tuple(mma_tiler_mn),
            cluster,
            m,
            n,
            k,
            experts,
            "k",
            "k",
            "n",
            kernel_type.FIX_PAD_SIZE,
            act_func,
        ):
            raise ValueError("Unsupported grouped GEMM dGLU tile, cluster, alignment, or layout configuration")
        validated_configs.add(signature)
    if key not in kernel_cache:
        major, minor = get_compute_capability()
        if major * 10 + minor < 100:
            raise RuntimeError(f"Grouped GEMM dGLU requires SM100+ compute capability, found SM{major}{minor}")
        mac = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster[0] * cluster[1]) - margin
        if mac <= 0:
            raise ValueError("CUDNNFE_CLUSTER_OVERLAP_MARGIN leaves no active clusters")
        kernel = dglu_kernel_type()(
            sf_vec_size=32,
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=use_2cta,
            mma_tiler_mn=tuple(mma_tiler_mn),
            cluster_shape_mn=cluster,
            vectorized_f32=False,
            discrete_col_sfd=discrete_col_sfd,
            expert_cnt=experts,
            weight_mode=MoEWeightMode.DENSE,
            act_func=act_func,
        )
        kernel_cache[key] = (kernel, mac)
    return kernel_cache[key]


def blockscaled_dglu_jax(
    *,
    a_tensor,
    b_tensor,
    c_tensor,
    sfa_tensor,
    sfb_tensor,
    padded_offsets,
    alpha_tensor,
    beta_tensor,
    prob_tensor,
    norm_const_tensor,
    d_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    discrete_col_sfd,
    act_func,
    linear_offset,
    geglu_alpha,
    glu_clamp_max,
    glu_clamp_min,
):
    inputs = dict(
        a=a_tensor,
        b=b_tensor,
        c=c_tensor,
        sfa=sfa_tensor,
        sfb=sfb_tensor,
        padded_offsets=padded_offsets,
        alpha=alpha_tensor,
        beta=beta_tensor,
        prob=prob_tensor,
        norm_const=norm_const_tensor,
    )
    check_jax_inputs(inputs)
    inputs["sfa"] = sf_array(sfa_tensor)
    inputs["sfb"] = sf_array(sfb_tensor)
    m, n = a_tensor.shape[0], b_tensor.shape[1]
    outputs = dict(
        d_row=output_type((m, 2 * n), d_dtype),
        d_col=output_type((m, 2 * n), d_dtype),
        dprob=output_type((m,), cutlass.Float32),
        sfd_row=output_type(sf_shape(m, 2 * n), cutlass.Float8E8M0FNU),
        sfd_col=output_type(sf_shape(2 * n, m), cutlass.Float8E8M0FNU),
    )
    kernel, mac = dglu_plan(inputs, outputs, mma_tiler_mn, cluster_shape_mn, discrete_col_sfd, act_func)
    if linear_offset is None:
        linear_offset = 1.0 if act_func == "dgeglu" else 0.0
    result = grouped_call(
        grouped_dglu_adapter,
        kernel,
        mac,
        tuple(output_type(t.shape, t.dtype) for t in inputs.values()),
        tuple(outputs.values()),
        zero_dprob=True,
        rubin=get_device_type() == "rubin",
        linear_offset=float(linear_offset),
        geglu_alpha=float(geglu_alpha),
        glu_clamp_max=float(glu_clamp_max),
        glu_clamp_min=float(glu_clamp_min),
    )(*inputs.values())
    return TupleDict(
        d_row_tensor=result[0],
        d_col_tensor=result[1],
        dprob_tensor=result[2],
        dbias_tensor=None,
        amax_tensor=None,
        sfd_row_tensor=result[3],
        sfd_col_tensor=result[4],
    )
