# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX-native (XLA custom call) entry points for grouped GEMM GLU, built on
:func:`cudnn.jax.call`.

``grouped_gemm_glu_jax_sm100`` serves two modes. With ``b_tensor`` it is the
canonical MXFP8 forward: dense A ``(m, k)``, B ``(experts, n, k)`` and already
MMA-packed scale bytes, running the SM100 or Rubin (SM107) block-scaled GLU
kernel. With ``b_ptrs`` it is the BF16 forward in discrete weight mode. The
per-expert weight pointers travel as a regular device array whose *values* are raw
addresses — the referenced weight buffers are not visible to XLA, so the caller
must keep them alive (and unmoved) across every execution of the traced
computation. ``padded_offsets`` values cannot be host-validated under tracing;
malformed offsets are the caller's responsibility here (the eager wrapper
validates them).
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
from cudnn.jax import call, gemm_operand_spec
from ..canonical_jax import check_grouped_shapes, check_jax_inputs, grouped_call, output_type, sf_array, sf_shape
from ..moe_utils import MoEWeightMode
from ..unfused.jax_api import _pointer_count, _prob_spec
from .moe_blockscaled_grouped_gemm_glu_bias import BlockScaledMoEGroupedGemmGluBiasKernel
from .moe_grouped_gemm_glu_bias import MoEGroupedGemmGluBiasBf16Kernel

# cache_key -> (kernel instance, max_active_clusters, workspace_bytes); reusing the
# instance keeps cutlass_call's compile cache warm (its FunctionSpec keys on the
# constexpr kwargs).
_kernel_cache: dict = {}

_output_dtypes = (cutlass.BFloat16, cutlass.Float16, cutlass.Float32)

_JAX_BLOCK_SCALED_ERROR = (
    "the block-scaled grouped GEMM GLU backend is not expressible as JAX arrays "
    "(its scale-factor tensors use an MMA-interleaved layout with no row-major equivalent); "
    "only the BF16 backend supports JAX inputs"
)


@cute.jit
def _grouped_glu_bf16_adapter(stream, a, b_ptrs, padded_offsets, alpha, prob, d, c, workspace, *, kernel, n, k, mac, linear_offset):
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
        prob=prob,
        bias=None,
        max_active_clusters=mac,
        stream=stream,
        linear_offset=cutlass.Float32(linear_offset),
    )


def grouped_gemm_glu_jax_sm100(
    a_tensor: Any,
    padded_offsets: Any,
    alpha_tensor: Any,
    b_ptrs: Any = None,
    n: Optional[int] = None,
    prob_tensor: Any = None,
    c_dtype: Any = cutlass.BFloat16,
    d_dtype: Any = cutlass.BFloat16,
    acc_dtype: Any = cutlass.Float32,
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    vector_f32: bool = False,
    act_func: str = "swiglu",
    linear_offset: Optional[float] = None,
    generate_c: bool = False,
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
    """Grouped GEMM GLU forward as an XLA custom call.

    With ``b_tensor``: canonical MXFP8 dense weights, eagerly or under jax.jit.
    A (m,k), B (experts,n,k), prob (m,) fp32/bf16, explicit alpha (experts,)
    fp32, norm_const (1,) fp32, and int32 padded_offsets (experts,). Offsets
    must be nondecreasing multiples of 256 within [0,m]; m is padded to 256.
    SF buffers contain packed E8M0 MMA-tiled bytes, at any dense rank (uint8
    bit patterns also accepted). FP8 A/B and an explicit FP8 ``d_dtype`` are
    required. ``act_func`` is "swiglu" or "geglu"; the activation scalars are
    compile-time constants of the traced call. ``discrete_col_sfd=True`` packs
    column scales by expert. Returns a TupleDict with the eager wrapper's keys:
    natural 2-D C/D/D_col (C None unless ``generate_c``) and physical 6-D SF
    buffers. Rows at or past padded_offsets[-1] are unspecified. Runs the Rubin
    GLU kernel on SM107 and the SM100 kernel otherwise.

    With ``b_ptrs``: BF16 discrete weights.

    Same contract as the eager wrapper's BF16 discrete mode: A ``(m, k, 1)`` k-major
    C-contiguous bfloat16, ``padded_offsets (experts,)`` int32 cumulative 256-aligned
    row offsets, ``alpha (experts,)`` float32, ``prob (m, 1, 1)`` float32, and
    ``b_ptrs`` holding per-expert ``(n, k)`` k-major bfloat16 weight base addresses
    (packed little-endian uint8, 8 bytes per pointer — or int64 with x64 mode).
    ``n`` is the full weight N before the GLU split; ``d`` comes back ``(m, n // 2, 1)``.
    ``linear_offset`` defaults per ``act_func`` (1.0 for ``"geglu"``, 0.0 for
    ``"swiglu"``) and is a compile-time constant of the traced call. Returns
    ``(d_tensor, c_tensor)`` with ``c_tensor`` None unless ``generate_c``. Rows
    at/past ``padded_offsets[-1]`` are unspecified.
    """
    if b_tensor is not None:
        unsupported = {
            "b_ptrs": b_ptrs is not None,
            "n": n is not None,
            "acc_dtype": _convert_to_cutlass_data_type(acc_dtype) is not cutlass.Float32,
            "vector_f32": vector_f32,
            "use_dynamic_sched": use_dynamic_sched,
        }
        for name, rejected in unsupported.items():
            if rejected:
                raise ValueError(f"{name} is unsupported for the JAX MXFP8 path")
        return blockscaled_glu_jax(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            sfa_tensor=sfa_tensor,
            sfb_tensor=sfb_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            prob_tensor=prob_tensor,
            norm_const_tensor=norm_const_tensor,
            c_dtype=c_dtype,
            d_dtype=d_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            discrete_col_sfd=discrete_col_sfd,
            act_func=act_func,
            linear_offset=linear_offset,
            geglu_alpha=geglu_alpha,
            glu_clamp_max=glu_clamp_max,
            glu_clamp_min=glu_clamp_min,
            generate_c=generate_c,
        )
    if b_ptrs is None or prob_tensor is None or any(t is not None for t in (sfa_tensor, sfb_tensor, norm_const_tensor)):
        raise ValueError("BF16 discrete weights take b_ptrs and prob_tensor and no scale tensors; pass b_tensor for canonical MXFP8")
    c_dtype = _convert_to_cutlass_data_type(c_dtype)
    d_dtype = _convert_to_cutlass_data_type(d_dtype)
    acc_dtype = _convert_to_cutlass_data_type(acc_dtype)

    if len(a_tensor.shape) != 3 or a_tensor.shape[2] != 1:
        raise ValueError(f"a_tensor must have shape (m, k, 1), got {tuple(a_tensor.shape)}")
    m, k, _ = a_tensor.shape
    if m % 256 != 0:
        raise ValueError(f"a_tensor M dimension must be 256-aligned, got {m}")
    if _convert_to_cutlass_data_type(a_tensor.dtype) is not cutlass.BFloat16:
        raise ValueError(f"a_tensor must have dtype bfloat16, got {a_tensor.dtype}; " + _JAX_BLOCK_SCALED_ERROR)
    if n is None or n <= 0 or n % 64 != 0:
        raise ValueError(f"n must be positive and divisible by 64 for paired GLU blocks, got {n}")
    if c_dtype not in _output_dtypes or d_dtype not in _output_dtypes:
        raise ValueError(f"c_dtype/d_dtype must be BF16, FP16, or FP32, got {c_dtype}/{d_dtype}; " + _JAX_BLOCK_SCALED_ERROR)
    if acc_dtype is not cutlass.Float32:
        raise ValueError(f"acc_dtype must be float32, got {acc_dtype}")
    if act_func not in ("swiglu", "geglu"):
        raise ValueError(f"act_func must be 'swiglu' or 'geglu', got {act_func}")
    if linear_offset is None:
        linear_offset = 1.0 if act_func == "geglu" else 0.0

    expert_cnt = _pointer_count(b_ptrs)
    if expert_cnt <= 0 or expert_cnt > 1024:
        raise ValueError(f"expert count must be in [1, 1024], got {expert_cnt}")
    if tuple(padded_offsets.shape) != (expert_cnt,):
        raise ValueError(f"padded_offsets must have shape ({expert_cnt},), got {tuple(padded_offsets.shape)}")
    if _convert_to_cutlass_data_type(padded_offsets.dtype) is not cutlass.Int32:
        raise ValueError(f"padded_offsets must have dtype int32, got {padded_offsets.dtype}")
    if tuple(alpha_tensor.shape) != (expert_cnt,) or _convert_to_cutlass_data_type(alpha_tensor.dtype) is not cutlass.Float32:
        raise ValueError(f"alpha_tensor must be ({expert_cnt},) float32, got {tuple(alpha_tensor.shape)} {alpha_tensor.dtype}")
    if tuple(prob_tensor.shape) != (m, 1, 1) or _convert_to_cutlass_data_type(prob_tensor.dtype) is not cutlass.Float32:
        raise ValueError(f"prob_tensor must be ({m}, 1, 1) float32, got {tuple(prob_tensor.shape)} {prob_tensor.dtype}")

    use_2cta_instrs = mma_tiler_mn[0] == 256
    cluster_shape_mn = tuple(cluster_shape_mn or ((2, 1) if use_2cta_instrs else (1, 1)))

    if not MoEGroupedGemmGluBiasBf16Kernel.can_implement(
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
        MoEGroupedGemmGluBiasBf16Kernel.FIX_PAD_SIZE,
    ):
        raise ValueError("Unsupported BF16 grouped GEMM GLU tile, cluster, alignment, or layout configuration")

    cache_key = (
        expert_cnt,
        c_dtype,
        d_dtype,
        acc_dtype,
        tuple(mma_tiler_mn),
        cluster_shape_mn,
        vector_f32,
        act_func,
        generate_c,
        use_dynamic_sched,
    )
    entry = _kernel_cache.get(cache_key)
    if entry is None:
        kernel = MoEGroupedGemmGluBiasBf16Kernel(
            acc_dtype=acc_dtype,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=tuple(mma_tiler_mn),
            cluster_shape_mn=cluster_shape_mn,
            vectorized_f32=vector_f32,
            expert_cnt=expert_cnt,
            weight_mode=MoEWeightMode.DISCRETE,
            use_dynamic_sched=use_dynamic_sched,
            act_func=act_func,
            enable_bias=False,
            generate_c=generate_c,
        )
        overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
        mac = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1]) - overlap_margin
        if mac <= 0:
            raise ValueError("max_active_clusters must be > 0 after applying CUDNNFE_CLUSTER_OVERLAP_MARGIN")
        entry = (kernel, mac, max(kernel.get_workspace_bytes(), 1))
        _kernel_cache[cache_key] = entry
    kernel, mac, workspace_bytes = entry

    n_out = n // 2
    operand = gemm_operand_spec()
    d_tensor, c_tensor, _workspace = call(
        _grouped_glu_bf16_adapter,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((m, n_out, 1), framework_dtype(d_dtype, "jax")),
            jax.ShapeDtypeStruct((m, n, 1), framework_dtype(c_dtype, "jax")),
            jax.ShapeDtypeStruct((workspace_bytes,), jnp.uint8),
        ),
        input_spec=(operand, None, None, None, _prob_spec()),
        output_spec=(operand, operand, None),
        # Only zero what the kernel does not write. Zero-filling an output the
        # kernel overwrites is a full-size device write on every dispatch, and it
        # scales with the output -- the dominant host-visible cost of the JAX path.
        # Nothing here qualifies: the kernel writes d/c over the addressed rows and the
        # descriptor helper writes the workspace before the kernel reads it. Rows at or
        # past padded_offsets[-1] are consequently unspecified, matching what the torch
        # wrapper has always returned from empty_strided.
        kernel=kernel,
        n=int(n),
        k=int(k),
        mac=mac,
        linear_offset=float(linear_offset),
    )(a_tensor, b_ptrs, padded_offsets, alpha_tensor, prob_tensor)

    return d_tensor, (c_tensor if generate_c else None)


kernel_cache = {}
validated_configs = set()


@cute.jit
def grouped_glu_adapter(
    stream,
    a,
    b,
    sfa,
    sfb,
    padded_offsets,
    alpha,
    prob,
    norm_const,
    c,
    d,
    d_col,
    sfd_row,
    sfd_col,
    *,
    kernel,
    mac,
    linear_offset,
    geglu_alpha,
    glu_clamp_max,
    glu_clamp_min,
):
    kernel(
        a=a,
        b=b,
        sfb=sfb,
        n=cutlass.Int32(0),
        k=cutlass.Int32(0),
        b_stride_size=cutlass.Int64(0),
        b_major_mode=OperandMajorMode.K,
        workspace_ptr=cute.make_ptr(cutlass.Uint8, 0, cute.AddressSpace.gmem, assumed_align=128),
        c=c,
        d=d,
        d_col=d_col,
        sfa=sfa,
        sfd_row_tensor=sfd_row,
        sfd_col_tensor=sfd_col,
        amax_tensor=None,
        norm_const_tensor=norm_const,
        padded_offsets=padded_offsets,
        alpha=alpha,
        prob=prob,
        bias=None,
        max_active_clusters=mac,
        stream=stream,
        linear_offset=cutlass.Float32(linear_offset),
        geglu_alpha=cutlass.Float32(geglu_alpha),
        glu_clamp_max=cutlass.Float32(glu_clamp_max),
        glu_clamp_min=cutlass.Float32(glu_clamp_min),
    )


def glu_kernel_type():
    if get_device_type() == "rubin":
        from .moe_blockscaled_grouped_gemm_glu_rubin import BlockScaledMoEGroupedGemmGluKernel

        return BlockScaledMoEGroupedGemmGluKernel
    return BlockScaledMoEGroupedGemmGluBiasKernel


def glu_plan(inputs, outputs, mma_tiler_mn, cluster_shape_mn, discrete_col_sfd, act_func):
    check_grouped_shapes(inputs, outputs, backward=False)
    if act_func not in ("swiglu", "geglu"):
        raise ValueError(f"act_func must be 'swiglu' or 'geglu' for the JAX MXFP8 path, got {act_func!r}")
    m, k = inputs["a"].shape
    experts, n, _ = inputs["b"].shape
    use_2cta_instrs = mma_tiler_mn[0] == 256
    cluster_shape_mn = tuple(cluster_shape_mn or ((2, 1) if use_2cta_instrs else (1, 1)))
    margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
    config = (experts, tuple(mma_tiler_mn), cluster_shape_mn, margin, discrete_col_sfd, act_func)
    validation_key = (config, tuple((name, tuple(t.shape), str(t.dtype)) for name, t in (*inputs.items(), *outputs.items())))
    if validation_key not in validated_configs:
        rest_k = ceil_div(ceil_div(k, 32), 4)
        for name, rows, groups in (("sfa", m, 1), ("sfb", n, experts)):
            if math.prod(inputs[name].shape) != 512 * ceil_div(rows, 128) * rest_k * groups:
                raise ValueError(f"{name.upper()} must contain the complete MMA-packed scale buffer")
        if _convert_to_cutlass_data_type(inputs["prob"].dtype) not in (cutlass.Float32, cutlass.BFloat16):
            raise ValueError("prob must be float32 or bfloat16")
        if experts > 1024:
            raise ValueError(f"expert count must be <= 1024, got {experts}")
        kernel_type = glu_kernel_type()
        if not kernel_type.can_implement(
            _convert_to_cutlass_data_type(inputs["a"].dtype),
            cutlass.Float8E8M0FNU,
            32,
            cutlass.Float32,
            _convert_to_cutlass_data_type(outputs["d"].dtype),
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
            raise ValueError("Unsupported grouped GEMM GLU tile, cluster, alignment, or layout configuration")
        validated_configs.add(validation_key)
    if config not in kernel_cache:
        major, minor = get_compute_capability()
        if major * 10 + minor < 100:
            raise RuntimeError(f"Grouped GEMM GLU requires SM100+ compute capability, but found SM{major}{minor}")
        mac = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1]) - margin
        if mac <= 0:
            raise ValueError("CUDNNFE_CLUSTER_OVERLAP_MARGIN leaves no active clusters")
        kernel = glu_kernel_type()(
            sf_vec_size=32,
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=tuple(mma_tiler_mn),
            cluster_shape_mn=cluster_shape_mn,
            vectorized_f32=False,
            generate_sfd=True,
            discrete_col_sfd=discrete_col_sfd,
            expert_cnt=experts,
            weight_mode=MoEWeightMode.DENSE,
            act_func=act_func,
        )
        kernel_cache[config] = (kernel, mac)
    return kernel_cache[config]


def blockscaled_glu_jax(
    *,
    a_tensor,
    b_tensor,
    sfa_tensor,
    sfb_tensor,
    padded_offsets,
    alpha_tensor,
    prob_tensor,
    norm_const_tensor,
    c_dtype,
    d_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    discrete_col_sfd,
    act_func,
    linear_offset,
    geglu_alpha,
    glu_clamp_max,
    glu_clamp_min,
    generate_c,
):
    inputs = dict(
        a=a_tensor,
        b=b_tensor,
        sfa=sfa_tensor,
        sfb=sfb_tensor,
        padded_offsets=padded_offsets,
        alpha=alpha_tensor,
        prob=prob_tensor,
        norm_const=norm_const_tensor,
    )
    check_jax_inputs(inputs)
    inputs["sfa"] = sf_array(sfa_tensor)
    inputs["sfb"] = sf_array(sfb_tensor)
    m = a_tensor.shape[0]
    n = b_tensor.shape[1]
    outputs = dict(
        c=output_type((m, n), c_dtype),
        d=output_type((m, n // 2), d_dtype),
        d_col=output_type((m, n // 2), d_dtype),
        sfd_row=output_type(sf_shape(m, n // 2), cutlass.Float8E8M0FNU),
        sfd_col=output_type(sf_shape(n // 2, m), cutlass.Float8E8M0FNU),
    )
    kernel, mac = glu_plan(inputs, outputs, mma_tiler_mn, cluster_shape_mn, discrete_col_sfd, act_func)
    if linear_offset is None:
        linear_offset = 1.0 if act_func == "geglu" else 0.0
    result = grouped_call(
        grouped_glu_adapter,
        kernel,
        mac,
        tuple(output_type(t.shape, t.dtype) for t in inputs.values()),
        tuple(outputs.values()),
        backward=False,
        linear_offset=float(linear_offset),
        geglu_alpha=float(geglu_alpha),
        glu_clamp_max=float(glu_clamp_max),
        glu_clamp_min=float(glu_clamp_min),
    )(*inputs.values())
    return TupleDict(
        c_tensor=result[0] if generate_c else None,
        d_tensor=result[1],
        d_col_tensor=result[2],
        amax_tensor=None,
        sfd_row_tensor=result[3],
        sfd_col_tensor=result[4],
    )
