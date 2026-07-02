# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API base, tensor metadata, and CuTe DSL call adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Tuple, final

import cutlass
import cutlass.cute as cute
import cutlass.jax as cutlass_jax
import jax
import jax.numpy as jnp
from cutlass.jax import TensorSpec

from ..api_base import ApiBase, TensorDesc
from .validation import (
    as_dtype as _as_dtype,
    as_optional_dtype as _as_optional_dtype,
    require_dtype as _require_dtype,
)

_NO_DEFAULT = object()


@dataclass(frozen=True, kw_only=True)
class JaxTensorDesc(TensorDesc):
    """Abstract JAX tensor metadata and its declared custom-call layout.

    The descriptor reads only shape and dtype from an array-like value.  Its
    shape and stride describe the modes presented to the kernel after the
    declared ``TensorSpec.mode`` permutation.  The stride is derived from the
    compact layout requested from XLA; it never inspects a physical JAX buffer
    or device.
    """

    jax_layout: tuple[int, ...]
    jax_mode: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        rank = len(tuple(self.shape))
        layout = tuple(self.jax_layout)
        mode = tuple(range(rank)) if self.jax_mode is None else tuple(self.jax_mode)
        dimensions = tuple(range(rank))
        if tuple(sorted(layout)) != dimensions:
            raise ValueError(f"jax_layout must be a permutation for rank {rank}, got {layout}")
        if tuple(sorted(mode)) != dimensions:
            raise ValueError(f"jax_mode must be a permutation for rank {rank}, got {mode}")

        expected_order = tuple(sorted(dimensions, key=lambda dim: layout[mode[dim]]))
        if self.stride_order is None:
            object.__setattr__(self, "stride_order", expected_order)
        elif tuple(self.stride_order) != expected_order:
            raise ValueError(f"stride_order must agree with jax_layout and jax_mode: expected {expected_order}, got {tuple(self.stride_order)}")

        object.__setattr__(self, "dtype", jnp.dtype(self.dtype))
        object.__setattr__(self, "jax_layout", layout)
        object.__setattr__(self, "jax_mode", mode)
        super().__post_init__()

        expected_stride: list[Any] = [None] * rank
        running: Any = 1
        for dim in expected_order:
            expected_stride[dim] = running
            running *= self.shape[dim]
        if self.stride != tuple(expected_stride):
            raise ValueError(
                "stride must describe the compact layout declared by jax_layout and jax_mode: " f"expected {tuple(expected_stride)}, got {self.stride}"
            )

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        layout: Sequence[int] | None = None,
        mode: Sequence[int] | None = None,
        name: str = "",
    ) -> "JaxTensorDesc":
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise TypeError(f"{name or 'value'} must expose shape and dtype metadata")
        input_shape = tuple(value.shape)
        if layout is None:
            layout = tuple(range(len(input_shape) - 1, -1, -1))
        if mode is None:
            mode = tuple(range(len(input_shape)))
        mode = tuple(mode)
        if tuple(sorted(mode)) != tuple(range(len(input_shape))):
            raise ValueError(f"mode must be a permutation for rank {len(input_shape)}, got {mode}")
        shape = tuple(input_shape[dim] for dim in mode)
        return cls(
            dtype=value.dtype,
            shape=shape,
            jax_layout=tuple(layout),
            jax_mode=mode,
            name=name,
        )

    @property
    def layout(self) -> tuple[int, ...]:
        return self.jax_layout

    @property
    def mode(self) -> tuple[int, ...]:
        assert self.jax_mode is not None
        return self.jax_mode


class ApiBaseJax(ApiBase, ABC):
    """Base for sample-signature-bound, traceable JAX callable objects.

    Instances are intentionally not wrapped in ``jax.jit``. The application
    owns JIT, sharding, donation, and device-placement policy.
    """

    def __init__(self) -> None:
        super().__init__()
        self._configuration_frozen = False

    def __setattr__(self, name: str, value: Any) -> None:
        """Invalidate pre-call support state and reject post-call mutation.

        JAX caches an executable by callable identity, not by the current
        contents of a callable object's attributes. Compile-affecting state may
        therefore change before the first invocation, but becomes immutable as
        soon as this object has participated in tracing or execution.
        """

        if self.__dict__.get("_configuration_frozen", False):
            raise AttributeError(
                f"{self.__class__.__name__} configuration is immutable after its first call; " "construct a new instance for different static options"
            )

        if name not in {"_configuration_frozen", "_is_supported"} and self.__dict__.get("_is_supported", False):
            object.__setattr__(self, "_is_supported", False)
        object.__setattr__(self, name, value)

    @final
    def check_support(self) -> bool:
        """Validate and cache sample metadata and static configuration."""

        if self._is_supported:
            return True
        if not self._check_support():
            return False
        object.__setattr__(self, "_is_supported", True)
        return True

    @abstractmethod
    def _check_support(self) -> bool:
        """Implement operation-specific support validation."""

    @abstractmethod
    def _call_impl(self, *args: Any, **kwargs: Any) -> Any:
        """Lower the operation using invocation-time JAX arrays."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke this object with JAX arrays matching its sample signature."""

        self._ensure_support_checked()
        object.__setattr__(self, "_configuration_frozen", True)
        return self._call_impl(*args, **kwargs)

    def make_tensor_desc(
        self,
        value: Any,
        *,
        layout: Sequence[int] | None = None,
        mode: Sequence[int] | None = None,
        name: str = "",
    ) -> JaxTensorDesc:
        """Return abstract JAX metadata without reading array values."""

        return JaxTensorDesc.from_value(value, layout=layout, mode=mode, name=name)

    def as_dtype(self, value: Any) -> Any:
        """Return a JAX dtype without retaining a dtype-bearing value."""

        return _as_dtype(value)

    def as_optional_dtype(self, value: Any | None) -> Any | None:
        """Return ``None`` or a JAX dtype without retaining the source value."""

        return _as_optional_dtype(value)

    def require_dtype(
        self,
        name: str,
        value: Any,
        valid_dtypes: Iterable[Any],
        *,
        default: Any = _NO_DEFAULT,
    ) -> Any:
        """Return a supported dtype from a dtype-like value or descriptor."""

        if default is _NO_DEFAULT:
            return _require_dtype(name, value, valid_dtypes)
        return _require_dtype(name, value, valid_dtypes, default=default)

    def make_optional_tensor_desc(
        self,
        value: Any | None,
        *,
        layout: Sequence[int] | None = None,
        mode: Sequence[int] | None = None,
        name: str = "",
    ) -> JaxTensorDesc | None:
        """Return metadata for an optional sample value."""

        if value is None:
            return None
        return self.make_tensor_desc(value, layout=layout, mode=mode, name=name)

    def check_tensor_signature(
        self,
        value: Any,
        expected: JaxTensorDesc,
        *,
        name: str = "",
    ) -> JaxTensorDesc:
        """Validate an invocation-time value against a sample descriptor."""

        if hasattr(value, "shape") and len(tuple(value.shape)) != expected.ndim:
            raise ValueError(f"{name} tensor shape mismatch: expected rank {expected.ndim}, got {tuple(value.shape)}")
        actual = self.make_tensor_desc(
            value,
            layout=expected.layout,
            mode=expected.mode,
            name=name,
        )
        self.check_tensor_shape(actual, expected.shape, name)
        self.check_dtype(actual, expected.dtype_name, name)
        return actual

    def check_optional_tensor_signature(
        self,
        value: Any | None,
        expected: JaxTensorDesc | None,
        *,
        name: str = "",
    ) -> JaxTensorDesc | None:
        """Validate optional-operand presence and, when present, its signature."""

        if value is None and expected is None:
            return None
        if value is None or expected is None:
            expected_presence = "present" if expected is not None else "absent"
            actual_presence = "present" if value is not None else "absent"
            raise ValueError(f"{name} presence mismatch: expected {expected_presence}, got {actual_presence}")
        return self.check_tensor_signature(value, expected, name=name)

    def get_jax_callable(self) -> Callable[..., Any]:
        """Return this stable, un-jitted callable object."""

        return self


@dataclass(frozen=True)
class BufferSpec:
    """Shape, dtype, tensor metadata, and optional fill value for a result.

    A spec passed through ``outputs`` is returned to the caller. A spec passed
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
    result_input_sources: Tuple[Optional[int], ...]
    input_output_aliases: Tuple[Tuple[int, int], ...]
    initialized_result_indices: Tuple[int, ...]

    @property
    def num_total_inputs(self) -> int:
        return self.num_user_inputs + len(self.initialized_result_indices)


@dataclass(frozen=True)
class _LaunchConfig:
    """Immutable compile-time state for the shared CuTe launch adapter."""

    fn: Callable[..., None]
    num_user_inputs: int
    num_total_inputs: int
    result_input_sources: Tuple[Optional[int], ...]
    static_args: Tuple[Tuple[str, Any], ...]


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


@cute.jit(preprocess=False)
def _launch_adapter(stream, *args, config: cutlass.Constexpr):
    """Reconstruct canonical ``inputs, outputs, workspaces`` launcher order."""

    input_args = args[: config.num_total_inputs]
    fresh_result_idx = config.num_total_inputs
    canonical_results = []
    for source in config.result_input_sources:
        if source is None:
            canonical_results.append(args[fresh_result_idx])
            fresh_result_idx += 1
        else:
            canonical_results.append(input_args[source])

    if fresh_result_idx != len(args):
        raise RuntimeError("CuTe launcher received more result buffers than expected")

    config.fn(
        stream,
        *input_args[: config.num_user_inputs],
        *canonical_results,
        **dict(config.static_args),
    )


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
    ``jax.jit``). Public results are returned as a tuple in ``outputs`` order;
    workspace results are hidden. All output and workspace sizes must be
    derivable from abstract input metadata, not from runtime tensor values.

    Inputs must be a flat sequence of array-like values; nested pytrees are
    intentionally outside the FE ABI. ``input_specs`` and
    ``BufferSpec.tensor_spec`` accept native ``cutlass.jax.TensorSpec`` objects;
    validation is delegated to ``cutlass.jax.cutlass_call``. Filled buffers are
    passed as internal aliased inputs so the public JAX function remains
    functional. ``fn`` and every value in ``static_args`` must be immutable and
    hashable because they participate in JAX and CUTLASS compilation cache keys.
    """

    inputs = tuple(inputs)
    static_args = dict(static_args or {})
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
        initialized_buffers.append(jnp.full(spec.shape, spec.fill_value, dtype=spec.dtype))
        initialized_specs.append(spec.tensor_spec)

    cutlass_input_specs = normalized_input_specs + tuple(initialized_specs)
    cutlass_output_specs = tuple(x.tensor_spec for x in plan.all_results)
    result_shape_dtypes = tuple(jax.ShapeDtypeStruct(x.shape, x.dtype) for x in plan.all_results)

    launch_config = _LaunchConfig(
        fn=fn,
        num_user_inputs=plan.num_user_inputs,
        num_total_inputs=plan.num_total_inputs,
        result_input_sources=plan.result_input_sources,
        static_args=tuple(sorted(static_args.items())),
    )
    call = cutlass_jax.cutlass_call(
        _launch_adapter,
        output_shape_dtype=result_shape_dtypes,
        input_spec=cutlass_input_specs,
        output_spec=cutlass_output_specs,
        input_output_aliases=dict(plan.input_output_aliases),
        allow_cuda_graph=allow_cuda_graph,
        compile_options=compile_options,
        use_static_tensors=use_static_tensors,
        config=launch_config,
    )
    all_results = call(*inputs, *initialized_buffers)

    if not isinstance(all_results, (tuple, list)):
        all_results = (all_results,)
    if len(all_results) != len(plan.all_results):
        raise RuntimeError(f"CuTe call returned {len(all_results)} buffers; expected " f"{len(plan.all_results)}")
    return tuple(all_results[: plan.num_public_results])


__all__ = ["ApiBaseJax", "BufferSpec", "JaxTensorDesc", "call_cutedsl"]
