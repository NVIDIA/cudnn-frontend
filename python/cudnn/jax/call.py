# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.jax.call: cutlass.jax.cutlass_call with cuDNN conveniences."""

from functools import lru_cache, partial

import cutlass.cute as cute

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


@lru_cache(maxsize=128)
def initialized_output_adapter(fn, num_inputs, output_positions):
    @cute.jit
    def adapter(stream, *args, **kwargs):
        inputs = args[:num_inputs]
        outputs = tuple(args[num_inputs + i] for i in output_positions)
        fn(stream, *inputs, *outputs, **kwargs)

    return adapter


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
        that output via ``input_output_aliases``. An adapter restores output order
        because the bridge retains aliased inputs and omits aliased outputs.
        Initializers require flat positional inputs and a single output descriptor
        or flat output sequence.
    """
    invoke = partial(
        cutlass_call,
        allow_cuda_graph=allow_cuda_graph,
        compile_options=compile_options,
        use_static_tensors=use_static_tensors,
        **kwargs,
    )
    if not initialized_outputs:
        return invoke(fn, output_shape_dtype=output_shape_dtype, input_spec=input_spec, output_spec=output_spec, input_output_aliases=input_output_aliases)

    single_output = hasattr(output_shape_dtype, "shape") and hasattr(output_shape_dtype, "dtype")
    if single_output:
        output_shape_dtype = (output_shape_dtype,)
        if isinstance(output_spec, TensorSpec):
            output_spec = (output_spec,)
    if not isinstance(output_shape_dtype, (tuple, list)) or any(not hasattr(x, "shape") for x in output_shape_dtype):
        raise ValueError("initialized_outputs requires a single output descriptor or flat output sequence")
    if any(i < 0 or i >= len(output_shape_dtype) for i in initialized_outputs):
        raise ValueError("initialized output index out of range")
    input_output_aliases = dict(input_output_aliases or {})
    if set(initialized_outputs).intersection(input_output_aliases.values()):
        raise ValueError("an initialized output cannot also alias an explicit input")

    initializers = tuple(sorted(initialized_outputs.items()))
    initialized_indices = tuple(i for i, _ in initializers)
    aliased_indices = set(input_output_aliases.values())
    ordinary_indices = tuple(i for i in range(len(output_shape_dtype)) if i not in initialized_indices and i not in aliased_indices)
    # The runtime ABI appends result buffers; aliased results must trail
    # the non-aliased results consumed by the compiled adapter.
    order = ordinary_indices + tuple(sorted(set(initialized_indices) | aliased_indices))
    result_positions = tuple(order.index(i) for i in range(len(order)))
    buffer_positions = {index: position for position, index in enumerate(initialized_indices + ordinary_indices)}
    output_positions = tuple(buffer_positions[i] for i in range(len(order)) if i not in aliased_indices)
    specs = tuple(output_spec) if output_spec is not None else (None,) * len(order)

    @lru_cache(maxsize=128)
    def initialized_call(num_inputs):
        aliases = dict(input_output_aliases)
        aliases.update({num_inputs + offset: i for offset, i in enumerate(initialized_indices)})
        full_input_spec = (tuple(input_spec) if input_spec is not None else (None,) * num_inputs) + tuple(specs[i] for i in initialized_indices)
        return invoke(
            initialized_output_adapter(fn, num_inputs, output_positions),
            output_shape_dtype=tuple(output_shape_dtype[i] for i in order),
            input_spec=full_input_spec,
            output_spec=tuple(specs[i] for i in order),
            input_output_aliases={i: result_positions[o] for i, o in aliases.items()},
        )

    def wrapper(*arrays: Any) -> Any:
        if any(not hasattr(a, "shape") for a in arrays):
            raise ValueError("initialized_outputs requires flat array inputs")
        inits = [init(output_shape_dtype[i]) for i, init in initializers]
        result = initialized_call(len(arrays))(*arrays, *inits)
        restored = [result[i] for i in result_positions]
        if single_output:
            return restored[0]
        return tuple(restored) if isinstance(output_shape_dtype, tuple) else restored

    return wrapper
