# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for dense-output grouped GEMM weight gradients on SM100."""

from __future__ import annotations

from functools import partial
import os
from typing import Any, Optional

import jax
import jax.numpy as jnp
from cutlass.jax import TensorSpec

from .._jax_api import (
    ApiBaseJax,
    make_buffer_desc,
    TWO_CTA_MMA_TILER_M,
    TupleDict,
    as_dtype,
    call_cutedsl,
    ceil_div,
    grouped_wgrad_workspace_bytes,
    grouped_workspace_tensor_spec,
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
    wgrad_tensor: Optional[Any] = None,
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
    outputs are not accepted. When ``accumulate_on_output=True``, an optional
    ``wgrad_tensor`` seeds the reducing epilogue and the returned value contains
    the seed plus the newly computed gradients. Omitting the seed preserves the
    simple fresh-output behavior by starting from zero.

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

    if wgrad_tensor is not None and not accumulate_on_output:
        raise ValueError("wgrad_tensor is only valid when accumulate_on_output=True")

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
    if wgrad_tensor is not None:
        require_array(
            wgrad_tensor,
            name="wgrad_tensor",
            shape=output_shape,
            dtype=valid_wgrad_dtypes,
        )
        seed_dtype = as_dtype(wgrad_tensor)
        wgrad_dtype = require_dtype(
            wgrad_dtype,
            valid_wgrad_dtypes,
            name="wgrad_dtype",
            default=seed_dtype,
        )
        if wgrad_dtype != seed_dtype:
            raise ValueError(
                "wgrad_dtype must match wgrad_tensor.dtype, got "
                f"{wgrad_dtype} and {seed_dtype}"
            )
    else:
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

    (result_wgrad_tensor,) = call_cutedsl(
        _launch,
        inputs,
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
                tensor_spec=wgrad_output_tensor_spec(),
                init_value=(
                    0.0 if accumulate_on_output and wgrad_tensor is None else None
                ),
            ),
        ),
        output_seeds=(wgrad_tensor,),
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
                tensor_spec=grouped_workspace_tensor_spec(),
            ),
        ),
        input_specs=input_specs,
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
        *,
        sample_wgrad_tensor: Optional[Any] = None,
    ) -> None:
        super().__init__()
        a_spec = wgrad_a_tensor_spec()
        b_spec = wgrad_b_tensor_spec()
        scale_spec = wgrad_scale_tensor_spec()
        self._sample_descs = {
            "a_tensor": self.make_tensor_desc(
                sample_a_tensor, tensor_spec=a_spec, name="sample_a_tensor"
            ),
            "b_tensor": self.make_tensor_desc(
                sample_b_tensor, tensor_spec=b_spec, name="sample_b_tensor"
            ),
            "sfa_tensor": self.make_tensor_desc(
                sample_sfa_tensor, tensor_spec=scale_spec, name="sample_sfa_tensor"
            ),
            "sfb_tensor": self.make_tensor_desc(
                sample_sfb_tensor, tensor_spec=scale_spec, name="sample_sfb_tensor"
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
            "wgrad_tensor": self.make_optional_tensor_desc(
                sample_wgrad_tensor,
                tensor_spec=wgrad_output_tensor_spec(),
                name="sample_wgrad_tensor",
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
            self._sample_descs["wgrad_tensor"],
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
        wgrad_tensor: Optional[Any] = None,
    ) -> TupleDict:
        return super().__call__(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            offsets_tensor,
            global_scale_a,
            global_scale_b,
            wgrad_tensor,
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
        wgrad_tensor: Optional[Any] = None,
    ) -> TupleDict:
        values = {
            "a_tensor": a_tensor,
            "b_tensor": b_tensor,
            "sfa_tensor": sfa_tensor,
            "sfb_tensor": sfb_tensor,
            "offsets_tensor": offsets_tensor,
            "global_scale_a": global_scale_a,
            "global_scale_b": global_scale_b,
            "wgrad_tensor": wgrad_tensor,
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
    *,
    wgrad_tensor: Optional[Any] = None,
) -> TupleDict:
    """Compute dense expert weight gradients using the SM100 grouped kernel.

    Set ``accumulate_on_output=True`` and pass ``wgrad_tensor`` to accumulate
    into an existing JAX value. If accumulation is enabled without a seed, the
    returned gradient starts from zero. Pointer-table outputs are not supported.
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
        sample_wgrad_tensor=wgrad_tensor,
    )
    return op(
        a_tensor,
        b_tensor,
        sfa_tensor,
        sfb_tensor,
        offsets_tensor,
        global_scale_a,
        global_scale_b,
        wgrad_tensor,
    )


__all__ = [
    "GroupedGemmWgradSm100",
    "grouped_gemm_wgrad_wrapper_sm100",
]
