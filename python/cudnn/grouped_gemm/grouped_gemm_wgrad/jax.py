# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for dense-output grouped GEMM weight gradients on SM100."""

from __future__ import annotations

from functools import partial
import os
from typing import Any, Optional

import jax
import jax.numpy as jnp

from .._jax_api import (
    ApiBaseJax,
    GROUPED_WORKSPACE_ALIGNMENT,
    make_buffer_desc,
    TWO_CTA_MMA_TILER_M,
    TupleDict,
    as_dtype,
    as_gemm_tensor_desc,
    call_cutedsl,
    ceil_div,
    grouped_wgrad_workspace_bytes,
    normalize_wgrad_input_order,
    require_16_byte_extent,
    require_array,
    require_dtype,
    require_grouped_cluster_shape,
    require_grouped_mma_tiler,
)


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


WGRAD_ALIGNMENT = 16
WGRAD_A_STRIDE_ORDER = (1, 0)
WGRAD_B_STRIDE_ORDER = (0, 1)
WGRAD_OUTPUT_STRIDE_ORDER = (2, 1, 0)


def _launch(
    stream,
    *args,
    acc_dtype: Any,
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    sf_vec_size: int,
    accumulate_on_output: bool,
    expert_cnt: int,
    input_order: str,
    has_global_scale: bool,
    max_active_clusters: int,
):
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
        raise RuntimeError(
            f"Unexpected grouped wgrad argument count: consumed {arg_idx}, "
            f"received {len(args)}"
        )

    kernel = BlockScaledMoEGroupedGemmWgradKernel(
        sf_vec_size=sf_vec_size,
        acc_dtype=jax_to_cutlass_dtype(acc_dtype),
        use_2cta_instrs=(mma_tiler_mn[0] == TWO_CTA_MMA_TILER_M),
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        accumulate_on_output=accumulate_on_output,
        expert_cnt=expert_cnt,
        weight_mode=MoEWeightMode.DENSE,
        input_order=WGradInputOrder(input_order),
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


def _grouped_gemm_wgrad_impl(
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
    cluster_overlap_margin: int = 0,
    *,
    _validate_only: bool = False,
) -> TupleDict | dict[str, Any]:
    """Compute dense expert weight gradients using the SM100 grouped kernel.

    ``a_tensor`` has shape ``(hidden,tokens)`` and ``b_tensor`` has shape
    ``(tokens,intermediate)``. ``offsets_tensor`` contains each expert's
    cumulative token end offset and determines grouping at runtime; its static
    length determines the output shape
    ``(experts,hidden,intermediate)``. Offset values must be non-decreasing,
    aligned to ``sf_vec_size``, and no greater than ``tokens``.

    The JAX surface exposes only the dense output mode; raw per-expert pointer
    outputs are not accepted. The output is inferred from the inputs and
    allocated by JAX. When ``accumulate_on_output=True``, the reducing epilogue
    starts from a zero-initialized output.

    ``input_order='tensor_ragged'`` is supported when A and B have already been
    packed expert-by-expert according to the runtime offsets. JAX cannot infer
    or perform that value-dependent packing from abstract shapes.
    """

    a_shape = require_array(
        a_tensor,
        name="a_tensor",
        rank=2,
        dtype=(jnp.float8_e4m3fn, jnp.float8_e5m2),
    )
    ab_dtype = as_dtype(a_tensor)
    b_shape = require_array(b_tensor, name="b_tensor", rank=2, dtype=ab_dtype)
    hidden, tokens = a_shape
    b_tokens, intermediate = b_shape
    if b_tokens != tokens:
        raise ValueError(
            "a_tensor and b_tensor token dimensions must match, got "
            f"{a_shape} and {b_shape}"
        )
    dimensions = {
        "hidden": hidden,
        "tokens": tokens,
        "intermediate": intermediate,
    }
    nonpositive = [
        f"{name}={value}" for name, value in dimensions.items() if value <= 0
    ]
    if nonpositive:
        raise ValueError(
            "Grouped-wgrad dimensions must be positive, got " + ", ".join(nonpositive)
        )

    if sf_vec_size != 32:
        raise ValueError(
            f"The JAX FP8 grouped-wgrad path requires sf_vec_size=32, got {sf_vec_size}"
        )

    require_array(
        sfa_tensor,
        name="sfa_tensor",
        shape=wgrad_scale_shape(hidden, tokens, sf_vec_size),
        dtype=jnp.float8_e8m0fnu,
    )
    sf_dtype = as_dtype(sfa_tensor)
    require_array(
        sfb_tensor,
        name="sfb_tensor",
        shape=wgrad_scale_shape(intermediate, tokens, sf_vec_size),
        dtype=sf_dtype,
    )

    offsets_shape = require_array(
        offsets_tensor,
        name="offsets_tensor",
        rank=1,
        dtype=jnp.int32,
    )
    (expert_cnt,) = offsets_shape
    if expert_cnt <= 0:
        raise ValueError(
            f"offsets_tensor must contain at least one expert, got {expert_cnt}"
        )
    output_shape = (expert_cnt, hidden, intermediate)

    has_global_scale = global_scale_a is not None or global_scale_b is not None
    if has_global_scale:
        if global_scale_a is None or global_scale_b is None:
            raise ValueError(
                "global_scale_a and global_scale_b must be provided together"
            )
        require_array(
            global_scale_a,
            name="global_scale_a",
            shape=(expert_cnt,),
            dtype=jnp.float32,
        )
        require_array(
            global_scale_b,
            name="global_scale_b",
            shape=(expert_cnt,),
            dtype=jnp.float32,
        )

    acc_dtype = require_dtype(
        acc_dtype,
        (jnp.float32,),
        name="acc_dtype",
        default=jnp.float32,
    )
    valid_wgrad_dtypes = (jnp.bfloat16, jnp.float16, jnp.float32)
    wgrad_dtype = require_dtype(
        wgrad_dtype,
        valid_wgrad_dtypes,
        name="wgrad_dtype",
        default=jnp.bfloat16,
    )
    mma_tiler_mn = require_grouped_mma_tiler(mma_tiler_mn, allowed_m=(128, 256))
    if cluster_shape_mn is None:
        cluster_shape_mn = (2, 1) if mma_tiler_mn[0] == TWO_CTA_MMA_TILER_M else (1, 1)
    cluster_shape_mn = require_grouped_cluster_shape(
        cluster_shape_mn,
        mma_tiler_mn=mma_tiler_mn,
    )
    input_order_value = normalize_wgrad_input_order(input_order).value

    require_16_byte_extent("a_tensor", tokens, ab_dtype)
    require_16_byte_extent("b_tensor", tokens, ab_dtype)
    require_16_byte_extent("wgrad_tensor", intermediate, wgrad_dtype)

    if _validate_only:
        return {
            "acc_dtype": acc_dtype,
            "wgrad_dtype": wgrad_dtype,
            "mma_tiler_mn": mma_tiler_mn,
            "cluster_shape_mn": cluster_shape_mn,
        }

    inputs = [a_tensor, b_tensor, sfa_tensor, sfb_tensor, offsets_tensor]
    input_descs = [
        as_gemm_tensor_desc(
            "a_tensor",
            a_tensor,
            public_stride_order=WGRAD_A_STRIDE_ORDER,
            ptr_assumed_align=WGRAD_ALIGNMENT,
        ),
        as_gemm_tensor_desc(
            "b_tensor",
            b_tensor,
            public_stride_order=WGRAD_B_STRIDE_ORDER,
            ptr_assumed_align=WGRAD_ALIGNMENT,
        ),
        as_gemm_tensor_desc(
            "sfa_tensor",
            sfa_tensor,
            public_stride_order=WGRAD_A_STRIDE_ORDER,
            ptr_assumed_align=WGRAD_ALIGNMENT,
        ),
        as_gemm_tensor_desc(
            "sfb_tensor",
            sfb_tensor,
            public_stride_order=WGRAD_A_STRIDE_ORDER,
            ptr_assumed_align=WGRAD_ALIGNMENT,
        ),
        as_gemm_tensor_desc("offsets_tensor", offsets_tensor),
    ]
    if has_global_scale:
        inputs.extend((global_scale_a, global_scale_b))
        input_descs.extend(
            (
                as_gemm_tensor_desc("global_scale_a", global_scale_a),
                as_gemm_tensor_desc("global_scale_b", global_scale_b),
            )
        )

    (result_wgrad_tensor,) = call_cutedsl(
        _launch,
        inputs,
        input_descs=input_descs,
        static_args={
            "acc_dtype": acc_dtype,
            "mma_tiler_mn": mma_tiler_mn,
            "cluster_shape_mn": cluster_shape_mn,
            "sf_vec_size": sf_vec_size,
            "accumulate_on_output": bool(accumulate_on_output),
            "expert_cnt": expert_cnt,
            "input_order": input_order_value,
            "has_global_scale": has_global_scale,
            "cluster_overlap_margin": int(cluster_overlap_margin),
        },
        outputs=(
            make_buffer_desc(
                "wgrad_tensor",
                output_shape,
                wgrad_dtype,
                public_stride_order=WGRAD_OUTPUT_STRIDE_ORDER,
                ptr_assumed_align=WGRAD_ALIGNMENT,
                init_value=0.0 if accumulate_on_output else None,
            ),
        ),
        workspaces=(
            make_buffer_desc(
                "workspace",
                (
                    grouped_wgrad_workspace_bytes(
                        expert_cnt,
                        input_order_value,
                    ),
                ),
                jnp.uint8,
                ptr_assumed_align=GROUPED_WORKSPACE_ALIGNMENT,
            ),
        ),
    )
    return TupleDict(wgrad_tensor=result_wgrad_tensor)


class GroupedGemmWgradSm100(ApiBaseJax):
    """Sample-signature-bound JAX callable for grouped GEMM weight gradients."""

    def __init__(
        self,
        sample_a_tensor: Any,
        sample_b_tensor: Any,
        sample_sfa_tensor: Any,
        sample_sfb_tensor: Any,
        sample_offsets_tensor: Any,
        sample_global_scale_a: Optional[Any] = None,
        sample_global_scale_b: Optional[Any] = None,
        acc_dtype: Any = None,
        wgrad_dtype: Any = None,
        mma_tiler_mn: tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[tuple[int, int]] = None,
        sf_vec_size: int = 32,
        accumulate_on_output: bool = False,
        input_order: str = "tensor2d",
    ) -> None:
        super().__init__()
        self._sample_descs = {
            "a_tensor": self.make_tensor_desc(
                sample_a_tensor,
                public_stride_order=WGRAD_A_STRIDE_ORDER,
                ptr_assumed_align=WGRAD_ALIGNMENT,
                name="sample_a_tensor",
            ),
            "b_tensor": self.make_tensor_desc(
                sample_b_tensor,
                public_stride_order=WGRAD_B_STRIDE_ORDER,
                ptr_assumed_align=WGRAD_ALIGNMENT,
                name="sample_b_tensor",
            ),
            "sfa_tensor": self.make_tensor_desc(
                sample_sfa_tensor,
                public_stride_order=WGRAD_A_STRIDE_ORDER,
                ptr_assumed_align=WGRAD_ALIGNMENT,
                name="sample_sfa_tensor",
            ),
            "sfb_tensor": self.make_tensor_desc(
                sample_sfb_tensor,
                public_stride_order=WGRAD_A_STRIDE_ORDER,
                ptr_assumed_align=WGRAD_ALIGNMENT,
                name="sample_sfb_tensor",
            ),
            "offsets_tensor": self.make_tensor_desc(
                sample_offsets_tensor, name="sample_offsets_tensor"
            ),
            "global_scale_a": self.make_optional_tensor_desc(
                sample_global_scale_a, name="sample_global_scale_a"
            ),
            "global_scale_b": self.make_optional_tensor_desc(
                sample_global_scale_b, name="sample_global_scale_b"
            ),
        }
        self._config = {
            "acc_dtype": self.as_optional_dtype(acc_dtype),
            "wgrad_dtype": self.as_optional_dtype(wgrad_dtype),
            "mma_tiler_mn": tuple(mma_tiler_mn),
            "cluster_shape_mn": (
                None if cluster_shape_mn is None else tuple(cluster_shape_mn)
            ),
            "sf_vec_size": sf_vec_size,
            "accumulate_on_output": accumulate_on_output,
            "input_order": input_order,
            "cluster_overlap_margin": int(
                os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0")
            ),
        }

        self._sample_descs = self.freeze_mapping(self._sample_descs)
        self._config = self.freeze_mapping(self._config)

    def _check_support(self) -> None:
        resolved = _grouped_gemm_wgrad_impl(
            self._sample_descs["a_tensor"],
            self._sample_descs["b_tensor"],
            self._sample_descs["sfa_tensor"],
            self._sample_descs["sfb_tensor"],
            self._sample_descs["offsets_tensor"],
            self._sample_descs["global_scale_a"],
            self._sample_descs["global_scale_b"],
            **self._config,
            _validate_only=True,
        )
        self._config = self.freeze_mapping({**self._config, **resolved})

    def __call__(
        self,
        a_tensor: Any,
        b_tensor: Any,
        sfa_tensor: Any,
        sfb_tensor: Any,
        offsets_tensor: Any,
        global_scale_a: Optional[Any] = None,
        global_scale_b: Optional[Any] = None,
    ) -> TupleDict:
        return super().__call__(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            offsets_tensor,
            global_scale_a,
            global_scale_b,
        )

    def _call_impl(
        self,
        a_tensor: Any,
        b_tensor: Any,
        sfa_tensor: Any,
        sfb_tensor: Any,
        offsets_tensor: Any,
        global_scale_a: Optional[Any] = None,
        global_scale_b: Optional[Any] = None,
    ) -> TupleDict:
        values = {
            "a_tensor": a_tensor,
            "b_tensor": b_tensor,
            "sfa_tensor": sfa_tensor,
            "sfb_tensor": sfb_tensor,
            "offsets_tensor": offsets_tensor,
            "global_scale_a": global_scale_a,
            "global_scale_b": global_scale_b,
        }
        self.check_tensor_signatures(self._sample_descs, values)
        return _grouped_gemm_wgrad_impl(**values, **self._config)


@partial(
    jax.jit,
    static_argnames=(
        "acc_dtype",
        "wgrad_dtype",
        "mma_tiler_mn",
        "cluster_shape_mn",
        "sf_vec_size",
        "accumulate_on_output",
        "input_order",
    ),
)
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
) -> TupleDict:
    """Compute dense expert weight gradients using the SM100 grouped kernel.

    The output is inferred and allocated by JAX. Set
    ``accumulate_on_output=True`` to use the reducing epilogue with a
    zero-initialized output. Pointer-table outputs are not supported.
    """

    op = GroupedGemmWgradSm100(
        a_tensor,
        b_tensor,
        sfa_tensor,
        sfb_tensor,
        offsets_tensor,
        global_scale_a,
        global_scale_b,
        acc_dtype=acc_dtype,
        wgrad_dtype=wgrad_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        accumulate_on_output=accumulate_on_output,
        input_order=input_order,
    )
    return op(
        a_tensor,
        b_tensor,
        sfa_tensor,
        sfb_tensor,
        offsets_tensor,
        global_scale_a,
        global_scale_b,
    )


__all__ = [
    "GroupedGemmWgradSm100",
    "grouped_gemm_wgrad_wrapper_sm100",
]
