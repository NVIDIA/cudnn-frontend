# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX-native (XLA custom call) entry points for grouped GEMM wgrad, built on
:func:`cudnn.jax.call`.

``grouped_gemm_wgrad_jax_sm100`` serves two modes. With ``sfa_tensor`` it is the
canonical MXFP8 wgrad: row-major A ``(m, tokens)`` and B ``(tokens, n)`` with
already MMA-packed scale bytes, writing a dense ``(experts, m, n)`` XLA output on
the SM100 or Rubin (SM107) block-scaled wgrad kernel. Without it, it is the BF16
wgrad in discrete output mode.

BF16 output pointer situation: the per-expert weight gradients are *not* XLA outputs.
``wgrad_ptrs`` is a regular input array whose values are raw device addresses of
caller-owned ``(m, n)`` row-major buffers; the kernel writes through them, so
under jit those buffers live outside XLA's buffer management. The caller must
keep them alive (and unmoved) across every execution of the traced computation,
and must order reads of them on the returned token (``jax.block_until_ready``,
or any data dependency on it). The token is the kernel's never-read discrete-mode
single-expert template output — an ``(m, n)`` zero-filled donated buffer whose
only job is to give the custom call an XLA-managed result (preventing dead-code
elimination) and to carry completion ordering; its contents are meaningless.

``offsets_tensor`` values cannot be host-validated under tracing; malformed
per-expert offsets (non-cumulative or not 256-aligned) are the caller's
responsibility here (the eager wrapper validates them). Only the total token
count — a static shape — is checked for 256-alignment.
"""

import math
import os
from typing import Any, Optional, Tuple, Union

import jax
import jax.numpy as jnp

import cutlass
import cutlass.cute as cute
import cutlass.utils

from cudnn.api_base import TupleDict, ceil_div, get_device_type
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.tensor_adapter import detect_framework, framework_dtype, get_compute_capability
from cudnn.jax import call, zeros_init
from ..canonical_jax import grouped_call, output_type, sf_array
from ..moe_utils import MoEWeightMode, WGradInputOrder
from ..unfused.jax_api import _pointer_count
from .moe_blockscaled_grouped_gemm_wgrad import BlockScaledMoEGroupedGemmWgradKernel
from .moe_grouped_gemm_wgrad import MoEGroupedGemmWgradBF16Kernel

# cache_key -> (kernel instance, max_active_clusters, workspace_bytes); reusing the
# instance keeps cutlass_call's compile cache warm (its FunctionSpec keys on the
# constexpr kwargs).
_kernel_cache: dict = {}

_output_dtypes = (cutlass.BFloat16, cutlass.Float16, cutlass.Float32)

_BLOCK_SCALED_JAX_ERROR = "pass sfa_tensor/sfb_tensor for MXFP8; FP4 wgrad is unsupported for JAX"


@cute.jit
def _wgrad_bf16_adapter(stream, a, b, offsets, wgrad_ptrs, wgrad_template, workspace, *, kernel, mac):
    # Discrete-mode out is a raw pointer to the device int64[] of per-expert base
    # addresses; the packed uint8 (or int64) input buffer recasts for free.
    out_arg = cute.recast_ptr(wgrad_ptrs.iterator, dtype=cutlass.Int64)
    kernel(
        mat_a=a,
        mat_b=b,
        out=out_arg,
        offs=offsets,
        workspace=workspace,
        max_active_clusters=mac,
        stream=stream,
        # Never-read (m, n) layout/dtype template for the per-expert TMA descriptors;
        # doubles as the XLA-managed token output returned to the caller.
        out_single_expert=wgrad_template,
    )


def grouped_gemm_wgrad_jax_sm100(
    a_tensor: Any,
    b_tensor: Any,
    offsets_tensor: Any,
    wgrad_ptrs: Any = None,
    wgrad_dtype: Any = cutlass.BFloat16,
    acc_dtype: Any = cutlass.Float32,
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    accumulate_on_output: bool = False,
    input_order: Union[WGradInputOrder, str] = WGradInputOrder.Tensor2D,
    *,
    sfa_tensor: Any = None,
    sfb_tensor: Any = None,
) -> Any:
    """Grouped GEMM wgrad as an XLA custom call.

    With ``sfa_tensor``: canonical MXFP8, eagerly or under jax.jit. A
    (m,tokens) and B (tokens,n) row-major with the same FP8 dtype, int32
    offsets_tensor (experts,) holding nondecreasing cumulative token ends that
    are multiples of 128, and SFA/SFB with packed E8M0 MMA-tiled bytes at any
    dense rank (uint8 bit patterns also accepted): for each expert in order,
    its (round_up(m or n, 128), tokens_i / 32) scales in 128x4 atoms. Returns a
    TupleDict with ``wgrad_tensor`` (experts, m, n) in ``wgrad_dtype``; experts
    without tokens come back zero. Runs the Rubin wgrad kernel on SM107 and
    the SM100 kernel otherwise.

    With ``wgrad_ptrs``: BF16 discrete per-expert output pointers.

    Same contract as the eager wrapper's discrete mode with explicit ``wgrad_ptrs``:
    ``a_tensor (m, tokens_sum)`` K-major and ``b_tensor (tokens_sum, n)`` N-major (both
    plain C-contiguous bfloat16 JAX arrays), ``offsets_tensor (experts,)`` int32
    cumulative 256-aligned token end-offsets, and ``wgrad_ptrs`` holding the per-expert
    ``(m, n)`` row-major ``wgrad_dtype`` output base addresses (packed little-endian
    uint8, 8 bytes per pointer — or int64 with x64 mode). The kernel writes each
    expert's weight gradient through those caller-owned buffers (with
    ``accumulate_on_output`` it TMA-reduces into them, so pre-zero/pre-seed them);
    empty experts are zero-filled.

    Returns an opaque ``(m, n)`` token array: block on it (or thread it through a data
    dependency) before reading the external per-expert buffers. Its values are
    unspecified. Dense (single 3-D wgrad tensor) output is available through the eager
    wrapper only.
    """
    wgrad_dtype = _convert_to_cutlass_data_type(wgrad_dtype)
    acc_dtype = _convert_to_cutlass_data_type(acc_dtype)
    input_order = WGradInputOrder(input_order)
    if sfa_tensor is not None:
        unsupported = {
            "wgrad_ptrs": wgrad_ptrs is not None,
            "acc_dtype": acc_dtype is not cutlass.Float32,
            "accumulate_on_output": accumulate_on_output,
            "input_order": input_order is not WGradInputOrder.Tensor2D,
        }
        for name, rejected in unsupported.items():
            if rejected:
                raise ValueError(f"{name} is unsupported for the JAX MXFP8 path")
        return blockscaled_wgrad_jax(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            sfa_tensor=sfa_tensor,
            sfb_tensor=sfb_tensor,
            offsets_tensor=offsets_tensor,
            wgrad_dtype=wgrad_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
        )
    if wgrad_ptrs is None or sfb_tensor is not None:
        raise ValueError("BF16 wgrad takes wgrad_ptrs and no scale tensors; pass sfa_tensor/sfb_tensor for canonical MXFP8")

    if len(a_tensor.shape) != 2:
        raise ValueError(f"a_tensor must have shape (m, tokens_sum), got {tuple(a_tensor.shape)}")
    if len(b_tensor.shape) != 2:
        raise ValueError(f"b_tensor must have shape (tokens_sum, n), got {tuple(b_tensor.shape)}")
    m, tokens_sum = a_tensor.shape
    tokens_b, n = b_tensor.shape
    if tokens_b != tokens_sum:
        raise ValueError(f"a_tensor and b_tensor token dimensions must match, got {tokens_sum} and {tokens_b}")
    for name, tensor in (("a_tensor", a_tensor), ("b_tensor", b_tensor)):
        if _convert_to_cutlass_data_type(tensor.dtype) is not cutlass.BFloat16:
            raise ValueError(f"{name} must have dtype bfloat16, got {tensor.dtype}; {_BLOCK_SCALED_JAX_ERROR}")
    if tokens_sum % MoEGroupedGemmWgradBF16Kernel.FIX_PAD_SIZE != 0:
        raise ValueError(f"total token count must be {MoEGroupedGemmWgradBF16Kernel.FIX_PAD_SIZE}-aligned, got {tokens_sum}")
    if wgrad_dtype not in _output_dtypes:
        raise ValueError(f"wgrad_dtype must be BF16, FP16, or FP32, got {wgrad_dtype}")
    if acc_dtype is not cutlass.Float32:
        raise ValueError(f"acc_dtype must be float32, got {acc_dtype}")

    expert_cnt = _pointer_count(wgrad_ptrs, "wgrad_ptrs")
    if expert_cnt <= 0:
        raise ValueError(f"expert count must be > 0, got {expert_cnt}")
    if tuple(offsets_tensor.shape) != (expert_cnt,):
        raise ValueError(f"offsets_tensor must have shape ({expert_cnt},), got {tuple(offsets_tensor.shape)}")
    if _convert_to_cutlass_data_type(offsets_tensor.dtype) is not cutlass.Int32:
        raise ValueError(f"offsets_tensor must have dtype int32, got {offsets_tensor.dtype}")

    use_2cta_instrs = mma_tiler_mn[0] == 256
    cluster_shape_mn = tuple(cluster_shape_mn or ((2, 1) if use_2cta_instrs else (1, 1)))

    # Per-expert token counts live in device memory (tracing-safe: no host reads), so
    # feed can_implement a synthetic 256-aligned split with the correct static sum;
    # it exercises every shape/tile/alignment rule that does not depend on the split.
    synthetic_group_k = [int(tokens_sum)] + [0] * (expert_cnt - 1)
    if not MoEGroupedGemmWgradBF16Kernel.can_implement(
        cutlass.BFloat16,
        wgrad_dtype,
        acc_dtype,
        use_2cta_instrs,
        tuple(mma_tiler_mn),
        cluster_shape_mn,
        m,
        n,
        synthetic_group_k,
        expert_cnt,
        "k",  # C-contiguous (m, tokens_sum) a_tensor is K-major
        "n",  # C-contiguous (tokens_sum, n) b_tensor is N-major
        MoEWeightMode.DISCRETE,
        input_order,
    ):
        raise ValueError("Unsupported BF16 grouped GEMM wgrad tile, cluster, alignment, or layout configuration")

    cache_key = (
        expert_cnt,
        wgrad_dtype,
        acc_dtype,
        tuple(mma_tiler_mn),
        cluster_shape_mn,
        accumulate_on_output,
        input_order,
    )
    entry = _kernel_cache.get(cache_key)
    if entry is None:
        kernel = MoEGroupedGemmWgradBF16Kernel(
            acc_dtype=acc_dtype,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=tuple(mma_tiler_mn),
            cluster_shape_mn=cluster_shape_mn,
            accumulate_on_output=accumulate_on_output,
            expert_cnt=expert_cnt,
            weight_mode=MoEWeightMode.DISCRETE,
            input_order=input_order,
        )
        overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
        mac = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1]) - overlap_margin
        if mac <= 0:
            raise ValueError("max_active_clusters must be > 0 after applying CUDNNFE_CLUSTER_OVERLAP_MARGIN")
        entry = (kernel, mac, max(kernel.get_workspace_bytes(), 1))
        _kernel_cache[cache_key] = entry
    kernel, mac, workspace_bytes = entry

    token, _workspace = call(
        _wgrad_bf16_adapter,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((m, n), framework_dtype(wgrad_dtype, "jax")),
            jax.ShapeDtypeStruct((workspace_bytes,), jnp.uint8),
        ),
        # Only zero what the kernel does not write. Zero-filling an output the
        # kernel overwrites is a full-size device write on every dispatch, and it
        # scales with the output -- the dominant host-visible cost of the JAX path.
        # The template keeps its zeros: the kernel writes through wgrad_ptrs and never
        # touches this buffer, so zeros are the only defined value it can carry. The
        # workspace is written by the descriptor helper before the kernel reads it.
        initialized_outputs={0: zeros_init},
        kernel=kernel,
        mac=mac,
    )(a_tensor, b_tensor, offsets_tensor, wgrad_ptrs)

    return token


kernel_cache = {}
validated_configs = set()


def wgrad_kernel_type():
    if get_device_type() == "rubin":
        from .moe_blockscaled_grouped_gemm_wgrad_rubin import BlockScaledMoEGroupedGemmWgradRubinKernel

        return BlockScaledMoEGroupedGemmWgradRubinKernel
    return BlockScaledMoEGroupedGemmWgradKernel


@cute.jit
def grouped_wgrad_adapter(stream, a, b, sfa, sfb, offsets, wgrad, workspace, *, kernel, mac, sfa_rows, sfb_rows):
    # The kernel rebuilds the MMA-tiled scale layouts from these 2-D shapes and reads only the base pointers.
    # The scale column count must stay dynamic, as in the torch path: a static count that fits in one K tile
    # traps the kernel.
    sf_cols = a.shape[1] // 32
    kernel(
        mat_a=a,
        mat_b=b,
        scale_a=cute.make_tensor(sfa.iterator, cute.make_layout((sfa_rows, sf_cols))),
        scale_b=cute.make_tensor(sfb.iterator, cute.make_layout((sfb_rows, sf_cols))),
        out=wgrad,
        offs=offsets,
        workspace=workspace,
        max_active_clusters=mac,
        stream=stream,
    )


def wgrad_plan(inputs, wgrad, mma_tiler_mn, cluster_shape_mn):
    a, b = inputs["a"], inputs["b"]
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("A must have shape (m, tokens) and B (tokens, n)")
    m, tokens = a.shape
    n = b.shape[1]
    experts = wgrad.shape[0]
    ab_dtype = _convert_to_cutlass_data_type(a.dtype)
    if ab_dtype not in (cutlass.Float8E4M3FN, cutlass.Float8E5M2) or _convert_to_cutlass_data_type(b.dtype) is not ab_dtype:
        raise ValueError("A and B must have the same MXFP8 dtype, e4m3 or e5m2; packed FP4 is unsupported")
    for name in ("sfa", "sfb"):
        if _convert_to_cutlass_data_type(inputs[name].dtype) is not cutlass.Float8E8M0FNU:
            raise ValueError(f"{name} must contain E8M0 scale bytes")
    if inputs["offsets"].shape != (experts,) or _convert_to_cutlass_data_type(inputs["offsets"].dtype) is not cutlass.Int32:
        raise ValueError("offsets_tensor must have shape (experts,) and dtype int32")
    if _convert_to_cutlass_data_type(wgrad.dtype) not in _output_dtypes:
        raise ValueError(f"wgrad_dtype must be BF16, FP16, or FP32, got {wgrad.dtype}")
    if m <= 0 or n <= 0 or experts <= 0 or tokens % 128:
        raise ValueError("Expected nonempty A/B, at least one expert, and a token count that is a multiple of 128")
    if any(size not in (128, 256) for size in mma_tiler_mn):
        raise ValueError(f"mma_tiler_mn entries must be 128 or 256, got {mma_tiler_mn}")
    use_2cta_instrs = mma_tiler_mn[0] == 256
    cluster_shape_mn = tuple(cluster_shape_mn or ((2, 1) if use_2cta_instrs else (1, 1)))
    if cluster_shape_mn[0] % (2 if use_2cta_instrs else 1) or math.prod(cluster_shape_mn) > 16 or any(size & (size - 1) for size in cluster_shape_mn):
        raise ValueError(f"Unsupported cluster_shape_mn {cluster_shape_mn}")
    margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
    config = (experts, tuple(mma_tiler_mn), cluster_shape_mn, margin)
    validation_key = (config, tuple((name, tuple(t.shape), str(t.dtype)) for name, t in inputs.items()))
    if validation_key not in validated_configs:
        for name, rows in (("sfa", m), ("sfb", n)):
            if math.prod(inputs[name].shape) != 128 * ceil_div(rows, 128) * tokens // 32:
                raise ValueError(f"{name.upper()} must contain the complete MMA-packed scale buffer")
        validated_configs.add(validation_key)
    if config not in kernel_cache:
        major, minor = get_compute_capability()
        if major * 10 + minor < 100:
            raise RuntimeError(f"Grouped GEMM wgrad requires SM100+ compute capability, but found SM{major}{minor}")
        mac = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1]) - margin
        if mac <= 0:
            raise ValueError("CUDNNFE_CLUSTER_OVERLAP_MARGIN leaves no active clusters")
        kernel = wgrad_kernel_type()(
            sf_vec_size=32,
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=tuple(mma_tiler_mn),
            cluster_shape_mn=cluster_shape_mn,
            expert_cnt=experts,
            weight_mode=MoEWeightMode.DENSE,
        )
        kernel_cache[config] = (kernel, mac, max(kernel.get_workspace_bytes(), 1))
    return kernel_cache[config]


def blockscaled_wgrad_jax(*, a_tensor, b_tensor, sfa_tensor, sfb_tensor, offsets_tensor, wgrad_dtype, mma_tiler_mn, cluster_shape_mn):
    inputs = dict(a=a_tensor, b=b_tensor, sfa=sfa_tensor, sfb=sfb_tensor, offsets=offsets_tensor)
    for name, tensor in inputs.items():
        if tensor is None:
            raise ValueError(f"{name}_tensor is required for the JAX MXFP8 path")
        if detect_framework(tensor) != "jax":
            raise ValueError(f"{name}_tensor must be a JAX array or tracer")
    inputs["sfa"] = sf_array(sfa_tensor)
    inputs["sfb"] = sf_array(sfb_tensor)
    m, n = a_tensor.shape[0], b_tensor.shape[-1]
    wgrad = output_type((offsets_tensor.shape[0], m, n), wgrad_dtype)
    kernel, mac, workspace_bytes = wgrad_plan(inputs, wgrad, mma_tiler_mn, cluster_shape_mn)
    result = grouped_call(
        grouped_wgrad_adapter,
        kernel,
        mac,
        tuple(output_type(t.shape, t.dtype) for t in inputs.values()),
        (wgrad, output_type((workspace_bytes,), cutlass.Uint8)),
        zero_dprob=False,
        sfa_rows=128 * ceil_div(m, 128),
        sfb_rows=128 * ceil_div(n, 128),
    )(*inputs.values())
    return TupleDict(wgrad_tensor=result[0])
