# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX-native (XLA custom call) entry points for the grouped GEMM dSReLU backward,
built on :func:`cudnn.jax.call`.

``grouped_gemm_dsrelu_jax_sm100`` serves two modes. With ``b_tensor`` it is the
canonical block-scaled backward with dense weights: MXFP8 or packed FP4 A
``(m, k)``, B ``(experts, n, k)``, saved C ``(m, n)`` and already MMA-packed scale
bytes. With ``b_ptrs`` it is the discrete FP8 mode, the same contract the eager
wrapper enforces for JAX arrays. Scale-factor
tensors travel in the physical C-contiguous atom shape ``(L, MN', K', 32, 4, 4)``:
the kernel rebuilds each SF layout from the GEMM shapes via
``tile_atom_to_shape_SF`` and consumes only the SF base pointers, so the
permuted torch atom view is never needed. The per-expert weight/SFB pointers
travel as regular device arrays whose *values* are raw addresses -- the
referenced buffers are not visible to XLA, so the caller must keep them alive
(and unmoved) across every execution of the traced computation.
``padded_offsets`` values cannot be host-validated under tracing; malformed
offsets are the caller's responsibility here (the eager wrapper validates them).
"""

import os
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass.cute.nvgpu import OperandMajorMode

from cudnn.api_base import TupleDict, ceil_div
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.tensor_adapter import framework_dtype, get_compute_capability
from cudnn.jax import TensorSpec, call, gemm_operand_spec, neg_inf_init, zeros_init
from ..canonical import kernel_facing_b, kernel_facing_fp4, kernel_facing_mx, kernel_facing_prob
from ..canonical_jax import check_block_scaled_operands, check_jax_inputs, check_packed_sf_size, fp8_dtypes, grouped_call, output_type, sf_array, sf_shape
from ..moe_utils import MoEWeightMode
from ..unfused.jax_api import _pointer_count
from .moe_blockscaled_grouped_gemm_dsrelu_quant import BlockScaledMoEGroupedGemmQuantBwdKernel, EpilogueType

# cache_key -> (kernel instance, max_active_clusters, workspace_bytes); reusing the
# instance keeps cutlass_call's compile cache warm (its FunctionSpec keys on the
# constexpr kwargs).
_kernel_cache: dict = {}

_fp8_dtypes = (cutlass.Float8E4M3FN, cutlass.Float8E5M2)
_c_dtypes = (cutlass.Float32, cutlass.Float16, cutlass.BFloat16, cutlass.Float8E4M3FN, cutlass.Float8E5M2)


def _prob_spec() -> TensorSpec:
    # (m, 1, 1) with m innermost: explicit ranks because trailing unit dims make
    # leading-dim inference ambiguous
    return TensorSpec(layout=(0, 1, 2))


def _sf_physical_spec() -> TensorSpec:
    # Physical C-contiguous (L, MN', K', 32, 4, 4) atom form; explicit ranks because
    # the extent-1 L dim makes leading-dim inference ambiguous. The kernel rebuilds
    # the SF layout from the GEMM shapes and consumes only the base pointer.
    return TensorSpec(layout=(5, 4, 3, 2, 1, 0))


def _as_e8m0_array(scale: Any, name: str) -> Any:
    """Present a uint8 E8M0-bit-pattern array as float8_e8m0fnu (free bitcast, jit-safe)."""
    dtype = _convert_to_cutlass_data_type(scale.dtype)
    if dtype is cutlass.Uint8:
        import ml_dtypes

        return scale.view(ml_dtypes.float8_e8m0fnu)
    if dtype is not cutlass.Float8E8M0FNU:
        raise ValueError(f"{name} must be float8_e8m0fnu (or uint8 bit patterns), got {scale.dtype}; fp8 inputs require e8m0 scale factors with sf_vec_size=32")
    return scale


@cute.jit
def _grouped_dsrelu_adapter(
    stream,
    a,
    c,
    sfa,
    b_ptrs,
    sfb_ptrs,
    padded_offsets,
    alpha,
    prob,
    norm_const,
    d_row,
    d_col,
    d_srelu,
    sfd_row,
    sfd_col,
    sfd_col_d_srelu,
    dprob,
    workspace,
    *,
    kernel,
    n,
    k,
    b_stride,
    b_major_mode,
    mac,
):
    # Discrete-mode b/sfb are raw pointers to the device int64[] of per-expert base
    # addresses; the packed uint8 (or int64) input buffers recast for free.
    b_arg = cute.recast_ptr(b_ptrs.iterator, dtype=cutlass.Int64)
    sfb_arg = cute.recast_ptr(sfb_ptrs.iterator, dtype=cutlass.Int64)
    kernel(
        a=a,
        b=b_arg,
        sfb=sfb_arg,
        n=cutlass.Int32(n),
        k=cutlass.Int32(k),
        b_stride_size=cutlass.Int64(b_stride),
        b_major_mode=b_major_mode,
        workspace_ptr=workspace.iterator,
        c=c,
        d=d_row,
        d_col=d_col,
        sfa=sfa,
        sfd_row_tensor=sfd_row,
        sfd_col_tensor=sfd_col,
        amax_tensor=None,  # amax is only produced for bf16/fp16 D, unreachable with fp8 inputs
        norm_const_tensor=norm_const,
        padded_offsets=padded_offsets,
        alpha=alpha,
        prob=prob,
        dprob=dprob,
        dbias_tensor=None,
        d_srelu=d_srelu,
        sfd_col_d_srelu_tensor=sfd_col_d_srelu,
        max_active_clusters=mac,
        stream=stream,
    )


@cute.jit
def _grouped_dsrelu_dbias_adapter(
    stream,
    a,
    c,
    sfa,
    b_ptrs,
    sfb_ptrs,
    padded_offsets,
    alpha,
    prob,
    norm_const,
    d_row,
    d_col,
    d_srelu,
    sfd_row,
    sfd_col,
    sfd_col_d_srelu,
    dprob,
    dbias,
    workspace,
    *,
    kernel,
    n,
    k,
    b_stride,
    b_major_mode,
    mac,
):
    b_arg = cute.recast_ptr(b_ptrs.iterator, dtype=cutlass.Int64)
    sfb_arg = cute.recast_ptr(sfb_ptrs.iterator, dtype=cutlass.Int64)
    kernel(
        a=a,
        b=b_arg,
        sfb=sfb_arg,
        n=cutlass.Int32(n),
        k=cutlass.Int32(k),
        b_stride_size=cutlass.Int64(b_stride),
        b_major_mode=b_major_mode,
        workspace_ptr=workspace.iterator,
        c=c,
        d=d_row,
        d_col=d_col,
        sfa=sfa,
        sfd_row_tensor=sfd_row,
        sfd_col_tensor=sfd_col,
        amax_tensor=None,
        norm_const_tensor=norm_const,
        padded_offsets=padded_offsets,
        alpha=alpha,
        prob=prob,
        dprob=dprob,
        dbias_tensor=dbias,
        d_srelu=d_srelu,
        sfd_col_d_srelu_tensor=sfd_col_d_srelu,
        max_active_clusters=mac,
        stream=stream,
    )


def grouped_gemm_dsrelu_jax_sm100(
    a_tensor: Any,
    c_tensor: Any,
    sfa_tensor: Any,
    padded_offsets: Any,
    alpha_tensor: Any,
    prob_tensor: Any,
    b_ptrs: Any = None,
    sfb_ptrs: Any = None,
    n: Optional[int] = None,
    norm_const_tensor: Any = None,
    b_dtype: Any = None,
    b_major: str = "k",
    d_dtype: Any = cutlass.Float8E4M3FN,
    acc_dtype: Any = cutlass.Float32,
    generate_dbias: bool = False,
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    sf_vec_size: int = 32,
    vector_f32: bool = False,
    discrete_col_sfd: bool = False,
    use_dynamic_sched: bool = False,
    use_dsrelu_reuse: bool = False,
    *,
    b_tensor: Any = None,
    sfb_tensor: Any = None,
) -> Any:
    """Grouped GEMM dSReLU backward as an XLA custom call.

    With ``b_tensor``: canonical block-scaled dense weights, eagerly or under
    jax.jit. A (m,k), B (experts,n,k), saved C (m,n), prob (m,) fp32/bf16,
    explicit alpha (experts,) fp32, and int32 padded_offsets (experts,). Offsets
    must be nondecreasing multiples of 256 within [0,m]; m is padded to 256. A/B
    are MXFP8, or packed FP4 as uint8 with two values per byte (K is twice the
    array extent). SF buffers contain packed MMA-tiled bytes at any dense rank:
    E8M0 with ``sf_vec_size=32`` for MXFP8 (uint8 bit patterns also accepted),
    E8M0 or E4M3 (NVFP4, ``sf_vec_size=16``) for FP4. Returns a TupleDict with
    the eager wrapper's keys. An FP8 ``d_dtype`` needs MXFP8 inputs and
    norm_const (1,) fp32, and returns D_row, D_col, D_srelu (m,n) and the
    physical 6-D SFD buffers. A BF16/FP16 ``d_dtype`` returns D_row, D_srelu and
    the per-expert amax (experts,1). dprob (m,) is zero-initialized for
    accumulation; other rows at or past padded_offsets[-1] are unspecified.
    dbias, dynamic scheduling and dSReLU reuse are unsupported. SM100+, as in
    Torch.

    With ``b_ptrs``: discrete FP8 weights.

    Same contract as the eager wrapper's JAX (discrete, fp8) mode: ``a (m, k, 1)``
    k-major C-contiguous fp8, ``c (m, n, 1)`` n-major forward activations,
    ``sfa`` in the physical atom shape ``(1, m/128, K', 32, 4, 4)`` as
    ``float8_e8m0fnu`` (or uint8 bit patterns), ``padded_offsets (experts,)``
    int32 cumulative 256-aligned row offsets, ``alpha (experts,)`` float32,
    ``prob (m, 1, 1)`` float32, ``norm_const (1,)`` float32, and
    ``b_ptrs``/``sfb_ptrs`` holding per-expert weight/SFB base addresses (packed
    little-endian uint8, 8 bytes per pointer -- or int64 with x64 mode).

    Returns, in the eager wrapper's key order,
    ``(d_row, d_col, d_srelu, dprob, dbias, amax, sfd_row, sfd_col, sfd_col_d_srelu)``
    with ``dbias`` None unless ``generate_dbias`` and ``amax`` always None (fp8
    inputs force fp8 D, which produces SFD outputs instead of amax). SF outputs
    come back in the physical atom form; rows at/past ``padded_offsets[-1]`` come
    back zero-filled (the outputs are donated zero-initialized buffers).
    """
    d_dtype = _convert_to_cutlass_data_type(d_dtype)
    acc_dtype = _convert_to_cutlass_data_type(acc_dtype)
    b_dtype = _convert_to_cutlass_data_type(b_dtype) if b_dtype is not None else None
    if b_tensor is not None:
        unsupported = {
            "b_ptrs": b_ptrs is not None,
            "sfb_ptrs": sfb_ptrs is not None,
            "n": n is not None,
            "b_dtype": b_dtype is not None,
            "b_major": b_major != "k",
            "acc_dtype": acc_dtype is not cutlass.Float32,
            "generate_dbias": generate_dbias,
            "vector_f32": vector_f32,
            "use_dynamic_sched": use_dynamic_sched,
            "use_dsrelu_reuse": use_dsrelu_reuse,
        }
        for name, rejected in unsupported.items():
            if rejected:
                raise ValueError(f"{name} is unsupported for the JAX block-scaled path")
        return blockscaled_dsrelu_jax(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            c_tensor=c_tensor,
            sfa_tensor=sfa_tensor,
            sfb_tensor=sfb_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            prob_tensor=prob_tensor,
            norm_const_tensor=norm_const_tensor,
            d_dtype=d_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            discrete_col_sfd=discrete_col_sfd,
            sf_vec_size=sf_vec_size,
        )
    if b_ptrs is None or sfb_ptrs is None or sfb_tensor is not None:
        raise ValueError("Discrete weights take b_ptrs and sfb_ptrs; pass b_tensor and sfb_tensor for canonical dense weights")

    if len(a_tensor.shape) != 3 or a_tensor.shape[2] != 1:
        raise ValueError(f"a_tensor must have shape (m, k, 1), got {tuple(a_tensor.shape)}")
    m, k, _ = a_tensor.shape
    if m % 256 != 0:
        raise ValueError(f"a_tensor M dimension must be 256-aligned, got {m}")
    ab_dtype = _convert_to_cutlass_data_type(a_tensor.dtype)
    if ab_dtype in (cutlass.Float4E2M1FN, cutlass.Uint8) or b_dtype in (cutlass.Float4E2M1FN, cutlass.Uint8):
        raise ValueError(
            "Packed fp4 A/B tensors (float4_e2m1fn / raw uint8) are not expressible as JAX arrays "
            "(JAX has no packed fp4 dtype); use fp8 inputs from JAX, or torch tensors for fp4"
        )
    if ab_dtype not in _fp8_dtypes:
        raise ValueError(f"a_tensor must be fp8 (float8_e4m3fn/float8_e5m2), got {a_tensor.dtype}")
    if b_dtype is not None and b_dtype is not ab_dtype:
        raise ValueError(f"b_dtype ({b_dtype}) must match a_tensor dtype ({ab_dtype})")
    if n is None or n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if b_major not in ("k", "n"):
        raise ValueError(f"b_major must be 'k' or 'n', got {b_major}")
    if d_dtype not in _fp8_dtypes:
        raise ValueError(f"d_dtype must be fp8 (float8_e4m3fn/float8_e5m2) when a/b are fp8, got {d_dtype}")
    if acc_dtype is not cutlass.Float32:
        raise ValueError(f"acc_dtype must be float32, got {acc_dtype}")
    if sf_vec_size != 32:
        raise ValueError(f"sf_vec_size must be 32 for fp8 inputs, got {sf_vec_size}")

    c_dtype = _convert_to_cutlass_data_type(c_tensor.dtype)
    if tuple(c_tensor.shape) != (m, n, 1):
        raise ValueError(f"c_tensor must have shape ({m}, {n}, 1), got {tuple(c_tensor.shape)}")
    if c_dtype not in _c_dtypes:
        raise ValueError(f"c_tensor must be fp32, fp16, bf16, or fp8, got {c_tensor.dtype}")
    if c_dtype in _fp8_dtypes and vector_f32:
        raise ValueError("Invalid configuration: fp8 c_dtype and vector_f32 is not supported. Please use vector_f32=False or c_dtype=bfloat16 instead")

    sfa_tensor = _as_e8m0_array(sfa_tensor, "sfa_tensor")
    rest_k = ceil_div(ceil_div(k, sf_vec_size), 4)
    if tuple(sfa_tensor.shape) != (1, ceil_div(m, 128), rest_k, 32, 4, 4):
        raise ValueError(
            f"sfa_tensor must be the physical atom shape (1, {ceil_div(m, 128)}, {rest_k}, 32, 4, 4) "
            f"for (m, k)=({m}, {k}) with sf_vec_size={sf_vec_size}, got {tuple(sfa_tensor.shape)}"
        )

    expert_cnt = _pointer_count(b_ptrs)
    if expert_cnt <= 0 or expert_cnt > 1024:
        raise ValueError(f"expert count must be in [1, 1024], got {expert_cnt}")
    if _pointer_count(sfb_ptrs, "sfb_ptrs") != expert_cnt:
        raise ValueError(f"sfb_ptrs must hold {expert_cnt} pointers to match b_ptrs")
    if tuple(padded_offsets.shape) != (expert_cnt,):
        raise ValueError(f"padded_offsets must have shape ({expert_cnt},), got {tuple(padded_offsets.shape)}")
    if _convert_to_cutlass_data_type(padded_offsets.dtype) is not cutlass.Int32:
        raise ValueError(f"padded_offsets must have dtype int32, got {padded_offsets.dtype}")
    if tuple(alpha_tensor.shape) != (expert_cnt,) or _convert_to_cutlass_data_type(alpha_tensor.dtype) is not cutlass.Float32:
        raise ValueError(f"alpha_tensor must be ({expert_cnt},) float32, got {tuple(alpha_tensor.shape)} {alpha_tensor.dtype}")
    if tuple(prob_tensor.shape) != (m, 1, 1) or _convert_to_cutlass_data_type(prob_tensor.dtype) is not cutlass.Float32:
        raise ValueError(f"prob_tensor must be ({m}, 1, 1) float32, got {tuple(prob_tensor.shape)} {prob_tensor.dtype}")
    if norm_const_tensor is None:
        raise ValueError("norm_const_tensor is required: fp8 inputs with e8m0 scale factors and fp8 D always generate SFD outputs")
    if tuple(norm_const_tensor.shape) != (1,) or _convert_to_cutlass_data_type(norm_const_tensor.dtype) is not cutlass.Float32:
        raise ValueError(f"norm_const_tensor must be (1,) float32, got {tuple(norm_const_tensor.shape)} {norm_const_tensor.dtype}")

    use_2cta_instrs = mma_tiler_mn[0] == 256
    cluster_shape_mn = tuple(cluster_shape_mn or ((2, 1) if use_2cta_instrs else (1, 1)))

    if not BlockScaledMoEGroupedGemmQuantBwdKernel.can_implement(
        ab_dtype,
        cutlass.Float8E8M0FNU,
        sf_vec_size,
        acc_dtype,
        d_dtype,
        use_2cta_instrs,
        tuple(mma_tiler_mn),
        cluster_shape_mn,
        m,
        n,
        k,
        expert_cnt,
        "k",
        b_major,
        "n",
        BlockScaledMoEGroupedGemmQuantBwdKernel.FIX_PAD_SIZE,
    ):
        raise ValueError("Unsupported grouped GEMM dSReLU tile, cluster, alignment, or layout configuration")

    cache_key = (
        expert_cnt,
        ab_dtype,
        c_dtype,
        d_dtype,
        acc_dtype,
        b_major,
        generate_dbias,
        tuple(mma_tiler_mn),
        cluster_shape_mn,
        sf_vec_size,
        vector_f32,
        discrete_col_sfd,
        use_dynamic_sched,
        use_dsrelu_reuse,
    )
    entry = _kernel_cache.get(cache_key)
    if entry is None:
        kernel = BlockScaledMoEGroupedGemmQuantBwdKernel(
            sf_vec_size=sf_vec_size,
            acc_dtype=acc_dtype,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=tuple(mma_tiler_mn),
            cluster_shape_mn=cluster_shape_mn,
            vectorized_f32=vector_f32,
            generate_sfd=True,  # fp8 inputs + e8m0 SF + fp8 D always generate SFD
            discrete_col_sfd=discrete_col_sfd,
            expert_cnt=expert_cnt,
            weight_mode=MoEWeightMode.DISCRETE,
            use_dynamic_sched=use_dynamic_sched,
            epilogue_type=EpilogueType.DSRELU.value,
            generate_dbias=generate_dbias,
            generate_d_srelu=True,
            use_dsrelu_reuse=use_dsrelu_reuse,
        )
        overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
        mac = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1]) - overlap_margin
        if mac <= 0:
            raise ValueError("max_active_clusters must be > 0 after applying CUDNNFE_CLUSTER_OVERLAP_MARGIN")
        entry = (kernel, mac, max(kernel.get_workspace_bytes(), 1))
        _kernel_cache[cache_key] = entry
    kernel, mac, workspace_bytes = entry

    d_jax_dtype = framework_dtype(d_dtype, "jax")
    sf_jax_dtype = framework_dtype(cutlass.Float8E8M0FNU, "jax")
    sfd_row_shape = (1, ceil_div(m, 128), ceil_div(ceil_div(n, sf_vec_size), 4), 32, 4, 4)
    sfd_col_shape = (1, ceil_div(n, 128), ceil_div(ceil_div(m, sf_vec_size), 4), 32, 4, 4)

    output_shape_dtype = [
        jax.ShapeDtypeStruct((m, n, 1), d_jax_dtype),  # d_row
        jax.ShapeDtypeStruct((m, n, 1), d_jax_dtype),  # d_col
        jax.ShapeDtypeStruct((m, n, 1), d_jax_dtype),  # d_srelu
        jax.ShapeDtypeStruct(sfd_row_shape, sf_jax_dtype),
        jax.ShapeDtypeStruct(sfd_col_shape, sf_jax_dtype),
        jax.ShapeDtypeStruct(sfd_col_shape, sf_jax_dtype),  # sfd_col_d_srelu
        jax.ShapeDtypeStruct((m, 1, 1), jnp.float32),  # dprob (atomic-add accumulator)
    ]
    operand = gemm_operand_spec()
    sf = _sf_physical_spec()
    output_spec = [operand, operand, operand, sf, sf, sf, _prob_spec()]
    if generate_dbias:
        # dbias (experts, n, 1) with n innermost shares the GEMM-operand stride ranks.
        output_shape_dtype.append(jax.ShapeDtypeStruct((expert_cnt, n, 1), framework_dtype(cutlass.BFloat16, "jax")))
        output_spec.append(operand)
    output_shape_dtype.append(jax.ShapeDtypeStruct((workspace_bytes,), jnp.uint8))
    output_spec.append(None)

    results = call(
        _grouped_dsrelu_dbias_adapter if generate_dbias else _grouped_dsrelu_adapter,
        output_shape_dtype=tuple(output_shape_dtype),
        input_spec=(operand, operand, sf, None, None, None, None, _prob_spec(), None),
        output_spec=tuple(output_spec),
        # ALL outputs donated: d/sfd for the trailing-/leading-unit-dim layout specs
        # (and defined bytes past the last padded offset); dprob and dbias because the
        # kernel accumulates them with atomic adds; the workspace because the helper
        # kernel writes the per-expert TMA descriptors into it (XLA inputs are immutable).
        initialized_outputs={i: zeros_init for i in range(len(output_shape_dtype))},
        kernel=kernel,
        n=int(n),
        k=int(k),
        b_stride=int(k if b_major == "k" else n),
        b_major_mode=OperandMajorMode.K if b_major == "k" else OperandMajorMode.MN,
        mac=mac,
    )(a_tensor, c_tensor, sfa_tensor, b_ptrs, sfb_ptrs, padded_offsets, alpha_tensor, prob_tensor, norm_const_tensor)

    d_row, d_col, d_srelu, sfd_row, sfd_col, sfd_col_d_srelu, dprob = results[:7]
    dbias = results[7] if generate_dbias else None
    # Eager wrapper key order: (d_row, d_col, d_srelu, dprob, dbias, amax, sfd_row,
    # sfd_col, sfd_col_d_srelu); amax is always None (fp8 D produces SFD, not amax).
    return d_row, d_col, d_srelu, dprob, dbias, None, sfd_row, sfd_col, sfd_col_d_srelu


kernel_cache = {}
validated_configs = set()


@cute.jit
def grouped_dsrelu_dense_adapter(stream, a, b, c, sfa, sfb, padded_offsets, alpha, prob, *rest, kernel, mac, low_precision):
    # rest: [norm_const], then d_row, d_col, d_srelu, dprob, sfd_row, sfd_col, sfd_col_d_srelu for FP8 output
    # or d_row, d_srelu, dprob, amax otherwise
    norm_const = rest[0] if cutlass.const_expr(low_precision) else None
    outputs = rest[int(low_precision) :]
    d_row = kernel_facing_mx(outputs[0])
    kernel(
        a=kernel_facing_mx(kernel_facing_fp4(a)),
        b=kernel_facing_b(kernel_facing_fp4(b)),
        sfb=sfb,
        n=cutlass.Int32(0),
        k=cutlass.Int32(0),
        b_stride_size=cutlass.Int64(0),
        b_major_mode=OperandMajorMode.K,
        workspace_ptr=cute.make_ptr(cutlass.Uint8, 0, cute.AddressSpace.gmem, assumed_align=128),
        c=kernel_facing_mx(c),
        d=d_row,
        # Without quantization the kernel only needs a D_col descriptor and does not write through it.
        d_col=kernel_facing_mx(outputs[1]) if cutlass.const_expr(low_precision) else d_row,
        sfa=sfa,
        sfd_row_tensor=outputs[4] if cutlass.const_expr(low_precision) else None,
        sfd_col_tensor=outputs[5] if cutlass.const_expr(low_precision) else None,
        amax_tensor=None if cutlass.const_expr(low_precision) else outputs[3],
        norm_const_tensor=norm_const,
        padded_offsets=padded_offsets,
        alpha=alpha,
        prob=kernel_facing_prob(prob),
        dprob=kernel_facing_prob(outputs[3] if cutlass.const_expr(low_precision) else outputs[2]),
        dbias_tensor=None,
        d_srelu=kernel_facing_mx(outputs[2] if cutlass.const_expr(low_precision) else outputs[1]),
        sfd_col_d_srelu_tensor=outputs[6] if cutlass.const_expr(low_precision) else None,
        max_active_clusters=mac,
        stream=stream,
    )


def dsrelu_plan(inputs, outputs, mma_tiler_mn, cluster_shape_mn, discrete_col_sfd, sf_vec_size):
    a, b = inputs["a"], inputs["b"]
    m, k = a.shape
    experts, n, b_k = b.shape
    if m <= 0 or experts <= 0 or k != b_k:
        raise ValueError("Expected nonempty A/B with matching K and at least one expert")
    if m % 256:
        raise ValueError("A rows must be padded to a multiple of 256")
    ab_dtype, sf_dtype = check_block_scaled_operands(a, b, inputs["sfa"], inputs["sfb"], sf_vec_size)
    if ab_dtype is cutlass.Float4E2M1FN:
        k *= 2
    if inputs["c"].shape != (m, n) or _convert_to_cutlass_data_type(inputs["c"].dtype) not in _c_dtypes:
        raise ValueError(f"C must have shape {(m, n)} and dtype fp32, fp16, bf16, or fp8")
    for name, shape in (("alpha", (experts,)), ("norm_const", (1,))):
        if name in inputs and (tuple(inputs[name].shape) != shape or _convert_to_cutlass_data_type(inputs[name].dtype) is not cutlass.Float32):
            raise ValueError(f"{name} must have shape {shape} and dtype float32")
    if inputs["prob"].shape != (m,) or _convert_to_cutlass_data_type(inputs["prob"].dtype) not in (cutlass.Float32, cutlass.BFloat16):
        raise ValueError("prob must have shape (m,) and dtype float32 or bfloat16")
    if inputs["padded_offsets"].shape != (experts,) or _convert_to_cutlass_data_type(inputs["padded_offsets"].dtype) is not cutlass.Int32:
        raise ValueError("padded_offsets must have shape (experts,) and dtype int32")
    d_dtype = _convert_to_cutlass_data_type(outputs["d_row"].dtype)
    low_precision = d_dtype in fp8_dtypes
    if low_precision != (ab_dtype in fp8_dtypes):
        raise ValueError("MXFP8 inputs require an FP8 d_dtype and FP4 inputs a BF16 or FP16 d_dtype")
    use_2cta_instrs = mma_tiler_mn[0] == 256
    cluster_shape_mn = tuple(cluster_shape_mn or ((2, 1) if use_2cta_instrs else (1, 1)))
    margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
    config = (experts, tuple(mma_tiler_mn), cluster_shape_mn, margin, discrete_col_sfd, low_precision, sf_vec_size)
    validation_key = (config, tuple((name, tuple(t.shape), str(t.dtype)) for name, t in (*inputs.items(), *outputs.items())))
    if validation_key not in validated_configs:
        check_packed_sf_size(inputs, (("sfa", m, 1), ("sfb", n, experts)), ceil_div(ceil_div(k, sf_vec_size), 4))
        if experts > 1024:
            raise ValueError(f"expert count must be <= 1024, got {experts}")
        if not BlockScaledMoEGroupedGemmQuantBwdKernel.can_implement(
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
            BlockScaledMoEGroupedGemmQuantBwdKernel.FIX_PAD_SIZE,
        ):
            raise ValueError("Unsupported grouped GEMM dSReLU dtype, tile, cluster, alignment, or layout configuration")
        validated_configs.add(validation_key)
    if config not in kernel_cache:
        major, minor = get_compute_capability()
        if major < 10:
            raise RuntimeError(f"Grouped GEMM dSReLU requires SM100+, found SM{major}{minor}")
        mac = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1]) - margin
        if mac <= 0:
            raise ValueError("CUDNNFE_CLUSTER_OVERLAP_MARGIN leaves no active clusters")
        kernel = BlockScaledMoEGroupedGemmQuantBwdKernel(
            sf_vec_size=sf_vec_size,
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=tuple(mma_tiler_mn),
            cluster_shape_mn=cluster_shape_mn,
            vectorized_f32=False,
            generate_sfd=low_precision,
            discrete_col_sfd=discrete_col_sfd,
            expert_cnt=experts,
            weight_mode=MoEWeightMode.DENSE,
            epilogue_type=EpilogueType.DSRELU.value,
            generate_d_srelu=True,
        )
        kernel_cache[config] = (kernel, mac)
    return kernel_cache[config]


def blockscaled_dsrelu_jax(
    *,
    a_tensor,
    b_tensor,
    c_tensor,
    sfa_tensor,
    sfb_tensor,
    padded_offsets,
    alpha_tensor,
    prob_tensor,
    norm_const_tensor,
    d_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    discrete_col_sfd,
    sf_vec_size,
):
    low_precision = d_dtype in fp8_dtypes
    inputs = dict(a=a_tensor, b=b_tensor, c=c_tensor, sfa=sfa_tensor, sfb=sfb_tensor, padded_offsets=padded_offsets, alpha=alpha_tensor, prob=prob_tensor)
    if low_precision:
        inputs["norm_const"] = norm_const_tensor
    check_jax_inputs(inputs)
    inputs["sfa"] = sf_array(sfa_tensor)
    inputs["sfb"] = sf_array(sfb_tensor)
    m = a_tensor.shape[0]
    experts, n = b_tensor.shape[:2]
    d = output_type((m, n), d_dtype)
    dprob = output_type((m,), cutlass.Float32)
    if low_precision:
        sfd_col = output_type(sf_shape(n, m, sf_vec_size), cutlass.Float8E8M0FNU)
        outputs = dict(d_row=d, d_col=d, d_srelu=d, dprob=dprob, sfd_row=output_type(sf_shape(m, n, sf_vec_size), cutlass.Float8E8M0FNU), sfd_col=sfd_col)
        outputs["sfd_col_d_srelu"] = sfd_col
    else:
        outputs = dict(d_row=d, d_srelu=d, dprob=dprob, amax=output_type((experts, 1), cutlass.Float32))
    kernel, mac = dsrelu_plan(inputs, outputs, mma_tiler_mn, cluster_shape_mn, discrete_col_sfd, sf_vec_size)
    result = grouped_call(
        grouped_dsrelu_dense_adapter,
        kernel,
        mac,
        tuple(output_type(t.shape, t.dtype) for t in inputs.values()),
        tuple(outputs.values()),
        initialized=((3, zeros_init),) if low_precision else ((2, zeros_init), (3, neg_inf_init)),
        low_precision=low_precision,
    )(*inputs.values())
    result = dict(zip(outputs, result))
    return TupleDict(
        d_row_tensor=result["d_row"],
        d_col_tensor=result.get("d_col"),
        d_srelu_tensor=result["d_srelu"],
        dprob_tensor=result["dprob"],
        dbias_tensor=None,
        amax_tensor=result.get("amax"),
        sfd_row_tensor=result.get("sfd_row"),
        sfd_col_tensor=result.get("sfd_col"),
        sfd_col_d_srelu_tensor=result.get("sfd_col_d_srelu"),
    )
