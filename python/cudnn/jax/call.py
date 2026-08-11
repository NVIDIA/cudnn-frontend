# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.jax.call: cutlass.jax.cutlass_call with cuDNN conveniences."""

from typing import Any, Callable, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp

import cutlass.jax
from cutlass.jax import TensorSpec, cutlass_call

if not cutlass.jax.is_available():  # pragma: no cover - guarded import surface
    raise ImportError(
        "cudnn.jax requires the CuTeDSL JAX extensions (cutlass.jax), which need jax >= 0.5; " "install/upgrade jax (`pip install --group jax` from a checkout)"
    )


def row_major_desc(shape, dtype, name: str):
    """Metadata-only TensorDesc for a C-contiguous (row-major) JAX buffer.

    Built from aval metadata so this works for jax.jit tracers as well as concrete
    arrays (tracers expose .shape/.dtype but no device or DLPack); used to reuse the
    class APIs' check_support for validation.
    """
    from cudnn.api_base import TensorDesc
    from cudnn.datatypes import _convert_to_cutlass_data_type
    from cudnn.tensor_adapter import Device

    shape = tuple(shape)
    strides, acc = [1] * len(shape), 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = acc
        acc *= shape[i]
    stride = tuple(strides)
    return TensorDesc(
        dtype=_convert_to_cutlass_data_type(dtype),
        shape=shape,
        stride=stride,
        stride_order=TensorDesc._compute_stride_order(shape, stride),
        device=Device("cuda", 0),
        name=name,
    )


def zeros_init(shape_dtype: jax.ShapeDtypeStruct) -> jax.Array:
    """Zero-filled initializer for accumulator outputs (e.g. atomic-max amax, atomic-add dprob)."""
    return jnp.zeros(shape_dtype.shape, shape_dtype.dtype)


def neg_inf_init(shape_dtype: jax.ShapeDtypeStruct) -> jax.Array:
    """-inf-filled initializer for max-accumulator outputs."""
    return jnp.full(shape_dtype.shape, -float("inf"), shape_dtype.dtype)


def gemm_operand_spec() -> TensorSpec:
    """Spec for the (MN, K, 1)-shaped k-/n-major GEMM operands and outputs.

    The trailing unit batch dim makes leading-dim inference ambiguous for a
    C-contiguous buffer, so the minor-to-major stride ranks are declared
    explicitly: K/N innermost (rank 0), MN next (rank 1), L outermost (rank 2).
    """
    return TensorSpec(layout=(1, 0, 2))


def sf_atom_spec() -> TensorSpec:
    """Spec presenting a physical C-contiguous (L, MN', K', 32, 4, 4) scale-factor
    buffer to the kernel in the logical MMA atom view (32, 4, MN', 4, K', L).

    ``mode`` remaps dimensions without materializing a transpose, so the kernel sees
    exactly the layout the torch path compiles from its permuted view.
    """
    return TensorSpec(mode=(3, 4, 1, 5, 2, 0))


def call(
    fn: Callable[..., None],
    *,
    output_shape_dtype: Any,
    input_spec: Optional[Sequence[Optional[TensorSpec]]] = None,
    output_spec: Optional[Sequence[Optional[TensorSpec]]] = None,
    initialized_outputs: Optional[Mapping[int, Callable[[jax.ShapeDtypeStruct], jax.Array]]] = None,
    input_output_aliases: Optional[dict[int, int]] = None,
    allow_cuda_graph: bool = True,
    compile_options: Optional[str] = None,
    use_static_tensors: bool = False,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Invoke a ``@cute.jit`` kernel adapter from JAX; see :func:`cutlass.jax.cutlass_call`.

    Same contract as ``cutlass_call`` plus:

    initialized_outputs: ``{output_index: init_fn}`` for outputs the kernel
        *accumulates into* rather than fully writes (atomic max/add). For each entry,
        ``init_fn(ShapeDtypeStruct) -> jax.Array`` produces the pre-initialized buffer
        (e.g. :func:`zeros_init`), which is appended as a trailing input and donated to
        that output via ``input_output_aliases`` — the bridge drops aliased inputs from
        the kernel's argument list, so ``fn``'s signature stays exactly the kernel's.
    """
    output_leaves = jax.tree.leaves(
        output_shape_dtype,
        is_leaf=lambda x: hasattr(x, "shape") and hasattr(x, "dtype"),
    )
    initialized_outputs = dict(initialized_outputs or {})
    input_output_aliases = dict(input_output_aliases or {})

    def wrapper(*arrays: Any) -> Any:
        inits = []
        aliases = dict(input_output_aliases)
        extra_specs = []
        for offset, (out_index, init_fn) in enumerate(sorted(initialized_outputs.items())):
            inits.append(init_fn(output_leaves[out_index]))
            aliases[len(arrays) + offset] = out_index
            extra_specs.append(output_spec[out_index] if output_spec is not None else None)

        full_input_spec = input_spec
        if inits and input_spec is not None:
            full_input_spec = tuple(input_spec) + tuple(extra_specs)

        return cutlass_call(
            fn,
            output_shape_dtype=output_shape_dtype,
            input_spec=full_input_spec,
            output_spec=output_spec,
            input_output_aliases=aliases,
            allow_cuda_graph=allow_cuda_graph,
            compile_options=compile_options,
            use_static_tensors=use_static_tensors,
            **kwargs,
        )(*arrays, *inits)

    return wrapper
