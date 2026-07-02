# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for dense-output grouped GEMM weight gradients on SM100."""

from __future__ import annotations

from functools import lru_cache
import os
from typing import Any, NamedTuple, Optional

import jax.numpy as jnp
from cutlass.jax import TensorSpec

from ..._jax.cutedsl import BufferSpec, call_cutedsl
from ..._jax.gemm import require_16_byte_extent, require_array
from ..._jax.grouped_gemm import grouped_workspace_tensor_spec
from ..._jax.validation import require_dtype
from ...gemm_validation import ceil_div, require_shape, resolve_max_active_clusters


class GroupedGemmWgradResult(NamedTuple):
    """Functional dense weight gradient."""

    wgrad_tensor: Any


def wgrad_scale_shape(
    rows: int,
    tokens: int,
    sf_vec_size: int,
) -> tuple[int, int]:
    """Return the packed 2Dx2D scale-buffer shape."""

    return (
        ceil_div(rows, 128) * 128,
        ceil_div(ceil_div(tokens, sf_vec_size), 4) * 4,
    )


def wgrad_a_tensor_spec() -> TensorSpec:
    """Describe ``A=(hidden,tokens)`` with tokens contiguous."""

    return TensorSpec(layout=(1, 0), mode=(0, 1), ptr_assumed_align=16)


def wgrad_b_tensor_spec() -> TensorSpec:
    """Describe ``B=(tokens,intermediate)`` with tokens contiguous."""

    return TensorSpec(layout=(0, 1), mode=(0, 1), ptr_assumed_align=16)


def wgrad_scale_tensor_spec() -> TensorSpec:
    """Describe the packed two-dimensional scale buffers."""

    return TensorSpec(layout=(1, 0), mode=(0, 1), ptr_assumed_align=16)


def wgrad_output_tensor_spec() -> TensorSpec:
    """Describe contiguous ``(experts,hidden,intermediate)`` output."""

    return TensorSpec(layout=(2, 1, 0), mode=(0, 1, 2), ptr_assumed_align=16)


@lru_cache(maxsize=None)
def _make_launcher(
    *,
    acc_dtype: Any,
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    sf_vec_size: int,
    accumulate_on_output: bool,
    expert_cnt: int,
    input_order: str,
    has_global_scale: bool,
    cluster_overlap_margin: int,
):
    def launch(stream, *args):
        import cutlass
        from cutlass.jax import jax_to_cutlass_dtype

        from ..moe_utils import MoEWeightMode, WGradInputOrder
        from .moe_blockscaled_grouped_gemm_wgrad import (
            BlockScaledMoEGroupedGemmWgradKernel,
        )

        arg_idx = 0

        def take():
            nonlocal arg_idx
            value = args[arg_idx]
            arg_idx += 1
            return value

        a_tensor = take()
        b_tensor = take()
        sfa_tensor = take()
        sfb_tensor = take()
        offsets_tensor = take()
        global_scale_a = take() if has_global_scale else None
        global_scale_b = take() if has_global_scale else None
        wgrad_tensor = take()
        workspace = take()
        if arg_idx != len(args):
            raise RuntimeError(f"Unexpected grouped wgrad argument count: consumed {arg_idx}, " f"received {len(args)}")

        kernel = BlockScaledMoEGroupedGemmWgradKernel(
            sf_vec_size=sf_vec_size,
            acc_dtype=jax_to_cutlass_dtype(acc_dtype),
            use_2cta_instrs=(mma_tiler_mn[0] == BlockScaledMoEGroupedGemmWgradKernel.TWO_CTA_MMA_TILER_M),
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            accumulate_on_output=accumulate_on_output,
            expert_cnt=expert_cnt,
            weight_mode=MoEWeightMode.DENSE,
            input_order=WGradInputOrder(input_order),
        )
        max_active_clusters = resolve_max_active_clusters(
            cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1]),
            cluster_overlap_margin,
        )
        kernel(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            wgrad_tensor,
            offsets_tensor,
            workspace,
            max_active_clusters,
            stream,
            global_scale_a,
            global_scale_b,
            None,
        )

    return launch


def grouped_gemm_wgrad_wrapper_sm100(
    a_tensor: Any,
    b_tensor: Any,
    sfa_tensor: Any,
    sfb_tensor: Any,
    offsets_tensor: Any,
    global_scale_a: Optional[Any] = None,
    global_scale_b: Optional[Any] = None,
    acc_dtype: Any = None,
    wgrad_dtype: Any = None,
    mma_tiler_mn: tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[tuple[int, int]] = None,
    sf_vec_size: int = 32,
    accumulate_on_output: bool = False,
    input_order: str = "tensor2d",
) -> GroupedGemmWgradResult:
    """Compute dense expert weight gradients using the SM100 grouped kernel.

    ``a_tensor`` has shape ``(hidden,tokens)`` and ``b_tensor`` has shape
    ``(tokens,intermediate)``. ``offsets_tensor`` contains each expert's
    cumulative token end offset and determines grouping at runtime; its static
    length determines the output shape
    ``(experts,hidden,intermediate)``. Offset values must be non-decreasing,
    aligned to ``sf_vec_size``, and no greater than ``tokens``.

    The JAX surface exposes only the dense output mode. The weight gradient and
    TMA-descriptor workspace are XLA-owned buffers; raw per-expert pointer
    outputs are not accepted. With ``accumulate_on_output=True``, XLA supplies
    a zero-initialized output to the reducing epilogue.

    ``input_order='tensor_ragged'`` is supported when A and B have already been
    packed expert-by-expert according to the runtime offsets. JAX cannot infer
    or perform that value-dependent packing from abstract shapes.
    """

    from .moe_blockscaled_grouped_gemm_wgrad import (
        BlockScaledMoEGroupedGemmWgradKernel,
    )

    kernel = BlockScaledMoEGroupedGemmWgradKernel
    a_shape = require_array("a_tensor", a_tensor, 2)
    b_shape = require_array("b_tensor", b_tensor, 2)
    hidden, tokens = a_shape
    b_tokens, intermediate = b_shape
    if b_tokens != tokens:
        raise ValueError("a_tensor and b_tensor token dimensions must match, got " f"{a_shape} and {b_shape}")
    dimensions = {
        "hidden": hidden,
        "tokens": tokens,
        "intermediate": intermediate,
    }
    nonpositive = [f"{name}={value}" for name, value in dimensions.items() if value <= 0]
    if nonpositive:
        raise ValueError("Grouped-wgrad dimensions must be positive, got " + ", ".join(nonpositive))

    ab_dtype = require_dtype(
        "a_tensor.dtype",
        a_tensor,
        (jnp.float8_e4m3fn, jnp.float8_e5m2),
    )
    require_dtype("b_tensor.dtype", b_tensor, (ab_dtype,))
    if sf_vec_size != kernel.FP8_SF_VEC_SIZE:
        raise ValueError("The JAX FP8 grouped-wgrad path requires " f"sf_vec_size={kernel.FP8_SF_VEC_SIZE}, got {sf_vec_size}")

    sfa_shape = require_array("sfa_tensor", sfa_tensor, 2)
    sfb_shape = require_array("sfb_tensor", sfb_tensor, 2)
    require_shape(
        "sfa_tensor",
        sfa_shape,
        wgrad_scale_shape(hidden, tokens, sf_vec_size),
    )
    require_shape(
        "sfb_tensor",
        sfb_shape,
        wgrad_scale_shape(intermediate, tokens, sf_vec_size),
    )
    sf_dtype = require_dtype(
        "sfa_tensor.dtype",
        sfa_tensor,
        (jnp.float8_e8m0fnu,),
    )
    require_dtype("sfb_tensor.dtype", sfb_tensor, (sf_dtype,))

    offsets_shape = require_array("offsets_tensor", offsets_tensor, 1)
    (expert_cnt,) = offsets_shape
    if expert_cnt <= 0:
        raise ValueError(f"offsets_tensor must contain at least one expert, got {expert_cnt}")
    require_dtype("offsets_tensor.dtype", offsets_tensor, (jnp.int32,))

    has_global_scale = global_scale_a is not None or global_scale_b is not None
    if has_global_scale:
        if global_scale_a is None or global_scale_b is None:
            raise ValueError("global_scale_a and global_scale_b must be provided together")
        require_shape(
            "global_scale_a",
            require_array("global_scale_a", global_scale_a, 1),
            (expert_cnt,),
        )
        require_shape(
            "global_scale_b",
            require_array("global_scale_b", global_scale_b, 1),
            (expert_cnt,),
        )
        require_dtype("global_scale_a.dtype", global_scale_a, (jnp.float32,))
        require_dtype("global_scale_b.dtype", global_scale_b, (jnp.float32,))

    acc_dtype = require_dtype(
        "acc_dtype",
        acc_dtype,
        (jnp.float32,),
        default=jnp.float32,
    )
    wgrad_dtype = require_dtype(
        "wgrad_dtype",
        wgrad_dtype,
        (jnp.bfloat16, jnp.float16, jnp.float32),
        default=jnp.bfloat16,
    )
    mma_tiler_mn = kernel.require_mma_tiler(mma_tiler_mn)
    if cluster_shape_mn is None:
        cluster_shape_mn = (2, 1) if mma_tiler_mn[0] == kernel.TWO_CTA_MMA_TILER_M else (1, 1)
    cluster_shape_mn = kernel.require_cluster_shape(
        cluster_shape_mn,
        mma_tiler_mn=mma_tiler_mn,
    )
    input_order_value = kernel.require_input_order(input_order).value

    require_16_byte_extent("a_tensor", tokens, ab_dtype)
    require_16_byte_extent("b_tensor", tokens, ab_dtype)
    require_16_byte_extent("wgrad_tensor", intermediate, wgrad_dtype)

    inputs = [a_tensor, b_tensor, sfa_tensor, sfb_tensor, offsets_tensor]
    input_specs = [
        wgrad_a_tensor_spec(),
        wgrad_b_tensor_spec(),
        wgrad_scale_tensor_spec(),
        wgrad_scale_tensor_spec(),
        None,
    ]
    if has_global_scale:
        inputs.extend((global_scale_a, global_scale_b))
        input_specs.extend((None, None))

    output_shape = (expert_cnt, hidden, intermediate)
    (wgrad_tensor,) = call_cutedsl(
        _make_launcher(
            acc_dtype=acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sf_vec_size=sf_vec_size,
            accumulate_on_output=bool(accumulate_on_output),
            expert_cnt=expert_cnt,
            input_order=input_order_value,
            has_global_scale=has_global_scale,
            cluster_overlap_margin=int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0")),
        ),
        inputs,
        outputs=(
            BufferSpec(
                "wgrad_tensor",
                output_shape,
                wgrad_dtype,
                tensor_spec=wgrad_output_tensor_spec(),
                fill_value=0.0 if accumulate_on_output else None,
            ),
        ),
        workspaces=(
            BufferSpec(
                "workspace",
                (
                    kernel.get_dense_workspace_bytes(
                        expert_cnt,
                        input_order_value,
                    ),
                ),
                jnp.uint8,
                tensor_spec=grouped_workspace_tensor_spec(),
            ),
        ),
        input_specs=input_specs,
        use_static_tensors=True,
    )
    return GroupedGemmWgradResult(wgrad_tensor=wgrad_tensor)


__all__ = ["GroupedGemmWgradResult", "grouped_gemm_wgrad_wrapper_sm100"]
