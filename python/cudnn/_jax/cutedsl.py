# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Internal JAX tracing adapter for frontend-only CuTe DSL operations.

This module is intentionally independent of :mod:`cudnn.api_base`.  The
existing API base implements an eager, preallocated-output lifecycle for
PyTorch.  JAX needs a functional call whose output buffers are described while
the function is being traced.  ``call_cutedsl`` bridges that difference using
``cutlass.jax.cutlass_call``.

The adapter also models temporary device workspaces as hidden custom-call
results.  XLA owns those buffers and can reuse their storage according to the
compiled program's liveness analysis.  Buffers that must start at zero or at a
constant value are materialized as JAX inputs and aliased to the corresponding
custom-call results.

Operator wrappers are responsible for target-neutral shape, dtype, layout,
and support inference before invoking this adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import cutlass.jax as cutlass_jax
from cutlass.jax import TensorSpec


@dataclass(frozen=True)
class BufferSpec:
    """Shape, dtype, tensor metadata, and optional fill value for a result.

    A spec passed through ``outputs`` is returned to the caller.  A spec passed
    through ``workspaces`` is supplied to the CuTe launcher after the public
    outputs and then omitted from the JAX-visible result. ``tensor_spec`` is a
    native :class:`cutlass.jax.TensorSpec`; ``None`` asks CUTLASS to infer its
    default from the shape and dtype. ``fill_value=None`` leaves the result
    uninitialized; any other value initializes it before the kernel launch.
    """

    name: str
    shape: Tuple[Any, ...]
    dtype: Any
    tensor_spec: TensorSpec | None = None
    fill_value: Any = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("BufferSpec.name must not be empty")
        object.__setattr__(self, "shape", tuple(self.shape))


@dataclass(frozen=True)
class _CallPlan:
    num_user_inputs: int
    all_results: Tuple[BufferSpec, ...]
    num_public_results: int
    # For every result, either the input index containing its aliased storage or
    # None when cutlass_call must provide a fresh output buffer.
    result_input_sources: Tuple[Optional[int], ...]
    input_output_aliases: Tuple[Tuple[int, int], ...]
    initialized_result_indices: Tuple[int, ...]

    @property
    def num_total_inputs(self) -> int:
        return self.num_user_inputs + len(self.initialized_result_indices)


def _build_call_plan(
    *,
    num_user_inputs: int,
    outputs: Sequence[BufferSpec],
    workspaces: Sequence[BufferSpec],
) -> _CallPlan:
    outputs = tuple(outputs)
    workspaces = tuple(workspaces)
    if not outputs:
        raise ValueError("call_cutedsl requires at least one public output")

    all_results = outputs + workspaces
    if not all(isinstance(spec, BufferSpec) for spec in all_results):
        raise TypeError("outputs and workspaces must contain only BufferSpec values")
    names = [spec.name for spec in all_results]
    if len(set(names)) != len(names):
        raise ValueError(f"Buffer names must be unique, got {names}")

    aliases = []
    result_sources: list[Optional[int]] = [None] * len(all_results)
    initialized_indices = []
    next_input_idx = num_user_inputs
    for result_idx, spec in enumerate(all_results):
        if spec.fill_value is None:
            continue
        result_sources[result_idx] = next_input_idx
        aliases.append((next_input_idx, result_idx))
        initialized_indices.append(result_idx)
        next_input_idx += 1

    return _CallPlan(
        num_user_inputs=num_user_inputs,
        all_results=all_results,
        num_public_results=len(outputs),
        result_input_sources=tuple(result_sources),
        input_output_aliases=tuple(aliases),
        initialized_result_indices=tuple(initialized_indices),
    )


def _make_launch_adapter_uncached(
    fn: Callable[..., None],
    num_user_inputs: int,
    num_total_inputs: int,
    result_input_sources: Tuple[Optional[int], ...],
) -> Callable[..., None]:
    """Reconstruct canonical ``inputs, outputs, workspaces`` launcher order."""

    def launch_adapter(stream: Any, *args: Any, **kwargs: Any) -> None:
        input_args = args[:num_total_inputs]
        fresh_results = iter(args[num_total_inputs:])
        canonical_results = []
        for source in result_input_sources:
            if source is None:
                canonical_results.append(next(fresh_results))
            else:
                canonical_results.append(input_args[source])

        try:
            next(fresh_results)
        except StopIteration:
            pass
        else:
            raise RuntimeError("CuTe launcher received more result buffers than expected")

        fn(
            stream,
            *input_args[:num_user_inputs],
            *canonical_results,
            **kwargs,
        )

    return launch_adapter


@lru_cache(maxsize=None)
def _make_launch_adapter_cached(
    fn: Callable[..., None],
    num_user_inputs: int,
    num_total_inputs: int,
    result_input_sources: Tuple[Optional[int], ...],
) -> Callable[..., None]:
    # Reusing the wrapper identity matters because CUTLASS includes the function
    # object in its compilation-cache key.
    return _make_launch_adapter_uncached(
        fn,
        num_user_inputs,
        num_total_inputs,
        result_input_sources,
    )


def _make_launch_adapter(fn: Callable[..., None], plan: _CallPlan) -> Callable[..., None]:
    try:
        return _make_launch_adapter_cached(
            fn,
            plan.num_user_inputs,
            plan.num_total_inputs,
            plan.result_input_sources,
        )
    except TypeError:
        # Callable objects are normally hashable, including @cute.jit functions.
        # Falling back keeps the adapter useful for unusual unhashable callables.
        return _make_launch_adapter_uncached(
            fn,
            plan.num_user_inputs,
            plan.num_total_inputs,
            plan.result_input_sources,
        )


def _call_cutedsl_with_modules(
    fn: Callable[..., None],
    inputs: Sequence[Any],
    *,
    outputs: Sequence[BufferSpec],
    workspaces: Sequence[BufferSpec] = (),
    input_specs: Optional[Sequence[Optional[TensorSpec]]] = None,
    static_args: Optional[Mapping[str, Any]] = None,
    allow_cuda_graph: bool = True,
    compile_options: Any = None,
    use_static_tensors: bool = False,
    jax_module: Any,
    jax_numpy_module: Any,
    cutlass_jax_module: Any,
) -> Tuple[Any, ...]:
    """Dependency-injected implementation used by tests and ``call_cutedsl``."""

    inputs = tuple(inputs)
    static_args = dict(static_args or {})
    reserved_static_args = {
        "allow_cuda_graph",
        "compile_options",
        "fn",
        "input_mode",
        "input_output_aliases",
        "input_spec",
        "output_mode",
        "output_shape_dtype",
        "output_spec",
        "use_static_tensors",
    }
    conflicts = sorted(reserved_static_args.intersection(static_args))
    if conflicts:
        raise ValueError("static_args contains names reserved by cutlass_call: " + ", ".join(conflicts))
    plan = _build_call_plan(
        num_user_inputs=len(inputs),
        outputs=outputs,
        workspaces=workspaces,
    )

    if input_specs is None:
        normalized_input_specs: Tuple[Optional[TensorSpec], ...] = (None,) * len(inputs)
    else:
        normalized_input_specs = tuple(input_specs)
        if len(normalized_input_specs) != len(inputs):
            spec_label = "spec" if len(inputs) == 1 else "specs"
            raise ValueError(f"Expected {len(inputs)} input tensor {spec_label}, got " f"{len(normalized_input_specs)}")

    for input_idx, value in enumerate(inputs):
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise TypeError(
                "call_cutedsl inputs must be a flat sequence of array-like "
                f"values with shape and dtype metadata; input #{input_idx} "
                f"is {type(value).__name__}"
            )

    initialized_buffers = []
    initialized_specs = []
    for result_idx in plan.initialized_result_indices:
        spec = plan.all_results[result_idx]
        initialized_buffers.append(jax_numpy_module.full(spec.shape, spec.fill_value, dtype=spec.dtype))
        initialized_specs.append(spec.tensor_spec)

    cutlass_input_specs = normalized_input_specs + tuple(initialized_specs)
    cutlass_output_specs = tuple(x.tensor_spec for x in plan.all_results)
    result_shape_dtypes = tuple(jax_module.ShapeDtypeStruct(x.shape, x.dtype) for x in plan.all_results)

    launcher = _make_launch_adapter(fn, plan)
    call = cutlass_jax_module.cutlass_call(
        launcher,
        output_shape_dtype=result_shape_dtypes,
        input_spec=cutlass_input_specs,
        output_spec=cutlass_output_specs,
        input_output_aliases=dict(plan.input_output_aliases),
        allow_cuda_graph=allow_cuda_graph,
        compile_options=compile_options,
        use_static_tensors=use_static_tensors,
        **static_args,
    )
    all_results = call(*inputs, *initialized_buffers)

    # Passing a tuple to output_shape_dtype asks cutlass_call for multiple
    # results, including the one-result case.  Keep a defensive normalization
    # here so a compatible bridge implementation can still be substituted.
    if not isinstance(all_results, (tuple, list)):
        all_results = (all_results,)
    if len(all_results) != len(plan.all_results):
        raise RuntimeError(f"CuTe call returned {len(all_results)} buffers; expected " f"{len(plan.all_results)}")
    return tuple(all_results[: plan.num_public_results])


def call_cutedsl(
    fn: Callable[..., None],
    inputs: Sequence[Any],
    *,
    outputs: Sequence[BufferSpec],
    workspaces: Sequence[BufferSpec] = (),
    input_specs: Optional[Sequence[Optional[TensorSpec]]] = None,
    static_args: Optional[Mapping[str, Any]] = None,
    allow_cuda_graph: bool = True,
    compile_options: Any = None,
    use_static_tensors: bool = False,
) -> Tuple[Any, ...]:
    """Invoke a CuTe DSL launcher as a functional JAX operation.

    The launcher's canonical signature is::

        fn(stream, *inputs, *outputs, *workspaces, **static_args) -> None

    Call this function while tracing a JAX function (normally inside
    ``jax.jit``).  Public results are returned as a tuple in ``outputs`` order;
    workspace results are hidden.  All output and workspace sizes must be
    derivable from abstract input metadata, not from runtime tensor values.

    Inputs must be a flat sequence of array-like values; nested pytrees are
    intentionally outside the FE ABI. ``input_specs`` and
    ``BufferSpec.tensor_spec`` accept native ``cutlass.jax.TensorSpec`` objects;
    validation is delegated to ``cutlass.jax.cutlass_call``. Filled buffers are
    passed as internal aliased inputs so the public JAX function remains
    functional.
    """

    return _call_cutedsl_with_modules(
        fn,
        inputs,
        outputs=outputs,
        workspaces=workspaces,
        input_specs=input_specs,
        static_args=static_args,
        allow_cuda_graph=allow_cuda_graph,
        compile_options=compile_options,
        use_static_tensors=use_static_tensors,
        jax_module=jax,
        jax_numpy_module=jnp,
        cutlass_jax_module=cutlass_jax,
    )


__all__ = [
    "BufferSpec",
    "call_cutedsl",
]
