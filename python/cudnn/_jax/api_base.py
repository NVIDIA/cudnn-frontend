# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API base, tensor metadata, and CuTe DSL call adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Optional, Tuple, final

import cutlass
import cutlass.cute as cute
import cutlass.jax as cutlass_jax
import jax
import jax.numpy as jnp
from cutlass.jax import TensorSpec

from ..api_base import ApiBase, TensorDesc, TupleDict

_NO_DEFAULT = object()


def as_dtype(value: Any) -> Any:
    """Return a JAX dtype without retaining a dtype-bearing value."""

    # Scalar dtype classes such as numpy.float32 expose an instance-level
    # dtype descriptor, so only unwrap dtype-bearing values, not classes.
    if not isinstance(value, type) and hasattr(value, "dtype"):
        value = value.dtype
    return jnp.dtype(value)


def as_optional_dtype(value: Any | None) -> Any | None:
    """Return ``None`` or a JAX dtype without retaining the source value."""

    return None if value is None else as_dtype(value)


def require_dtype(
    value: Any,
    valid_dtypes: Iterable[Any],
    *,
    name: str | None = None,
    default: Any = _NO_DEFAULT,
) -> Any:
    """Return a supported dtype from a dtype-like value or object with ``dtype``.

    The diagnostic name defaults to ``"<value.name>.dtype"`` for named
    dtype-bearing values, then ``"dtype"``.
    """

    if not name:
        value_name = getattr(value, "name", None) if not isinstance(value, type) and hasattr(value, "dtype") else None
        name = f"{value_name}.dtype" if value_name else "dtype"

    if value is None:
        if default is _NO_DEFAULT:
            raise ValueError(f"{name} must not be None")
        value = default

    dtype = as_dtype(value)
    valid_dtypes = tuple(as_dtype(item) for item in valid_dtypes)
    if dtype not in valid_dtypes:
        supported = ", ".join(item.name for item in valid_dtypes)
        raise ValueError(f"{name} must be one of {{{supported}}}, got {dtype}")
    return dtype


def require_array(
    value: Any,
    *,
    name: str | None = None,
    rank: int | Iterable[int] | None = None,
    shape: Sequence[Any] | None = None,
    dtype: Any | Iterable[Any] | None = None,
) -> tuple[Any, ...]:
    """Validate array metadata and return its shape.

    The diagnostic name defaults to a non-empty ``value.name``, then
    ``"value"``. ``dtype`` accepts either one dtype or an iterable of supported
    dtypes. Validation applies to the value's exposed shape. For a
    :class:`JaxTensorDesc`, that is the kernel-visible shape; use
    :attr:`JaxTensorDesc.array_shape` when validating a public JAX array against
    a descriptor.
    """

    name = name or getattr(value, "name", None) or "value"
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must have shape and dtype metadata")

    actual_shape = tuple(value.shape)
    if rank is not None:
        valid_ranks = (rank,) if isinstance(rank, int) else tuple(rank)
        if len(actual_shape) not in valid_ranks:
            if len(valid_ranks) == 1:
                raise ValueError(f"{name} must have rank {valid_ranks[0]}, got shape {actual_shape}")
            expected = ", ".join(str(item) for item in valid_ranks)
            raise ValueError(f"{name} must have one of ranks {{{expected}}}, got shape {actual_shape}")

    if shape is not None:
        expected_shape = tuple(shape)
        if actual_shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {actual_shape}")

    if dtype is not None:
        is_dtype_collection = isinstance(dtype, Iterable) and not isinstance(dtype, (str, bytes, type)) and not hasattr(dtype, "dtype")
        valid_dtypes = tuple(dtype) if is_dtype_collection else (dtype,)
        require_dtype(value, valid_dtypes, name=f"{name}.dtype")

    return actual_shape


def _flatten_tuple_dict(value: TupleDict) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Flatten the current mapping values in insertion order."""

    keys = tuple(dict.keys(value))
    children = tuple(dict.__getitem__(value, key) for key in keys)
    return children, keys


def _flatten_tuple_dict_with_keys(value: TupleDict):
    children, keys = _flatten_tuple_dict(value)
    return tuple((jax.tree_util.DictKey(key), child) for key, child in zip(keys, children)), keys


def _unflatten_tuple_dict(keys: tuple[Any, ...], children: Iterable[Any]) -> TupleDict:
    return TupleDict(zip(keys, children))


if not getattr(TupleDict, "_jax_pytree_registered", False):
    jax.tree_util.register_pytree_with_keys(
        TupleDict,
        _flatten_tuple_dict_with_keys,
        _unflatten_tuple_dict,
        _flatten_tuple_dict,
    )
    TupleDict._jax_pytree_registered = True


def _resolve_layout_mode(
    rank: int,
    *,
    tensor_spec: TensorSpec | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    layout = None if tensor_spec is None else tensor_spec.layout
    mode = None if tensor_spec is None else tensor_spec.mode

    dimensions = tuple(range(rank))
    layout = tuple(reversed(dimensions)) if layout is None else tuple(layout)
    mode = dimensions if mode is None else tuple(mode)
    if tuple(sorted(layout)) != dimensions:
        raise ValueError(f"layout must be a permutation for rank {rank}, got {layout}")
    if tuple(sorted(mode)) != dimensions:
        raise ValueError(f"mode must be a permutation for rank {rank}, got {mode}")
    return layout, mode


@dataclass(frozen=True, kw_only=True)
class JaxTensorDesc(TensorDesc):
    """Abstract JAX tensor metadata and its CUTLASS lowering specification.

    The descriptor reads only shape and dtype from an array-like value.  Its
    shape and stride describe the modes presented to the kernel after the
    declared ``TensorSpec.mode`` permutation.  The stride is derived from the
    compact layout requested from XLA; it never inspects a physical JAX buffer
    or device. ``tensor_spec`` is retained unchanged for use by
    :func:`call_cutedsl`; ``None`` preserves CUTLASS's inferred-default
    behavior. The ``layout`` and ``mode`` properties expose normalized
    defaults without rewriting that native object. Prefer :meth:`from_value`
    for input metadata because it applies ``TensorSpec.mode`` to the public
    array shape. A shape passed directly to the constructor must already be in
    kernel-visible mode order.
    """

    tensor_spec: TensorSpec | None = None
    jax_layout: tuple[int, ...] = field(init=False)
    jax_mode: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        rank = len(tuple(self.shape))
        layout, mode = _resolve_layout_mode(
            rank,
            tensor_spec=self.tensor_spec,
        )
        dimensions = tuple(range(rank))

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
        tensor_spec: TensorSpec | None = None,
        layout: Sequence[int] | None = None,
        mode: Sequence[int] | None = None,
        name: str = "",
    ) -> "JaxTensorDesc":
        """Capture array metadata together with its native lowering spec.

        ``layout`` and ``mode`` are convenience forms that construct a native
        ``TensorSpec``. Supplying neither keeps ``tensor_spec=None`` so CUTLASS
        can infer its complete default, including divisibility.
        """

        input_shape = require_array(value, name=name or None)
        if tensor_spec is not None:
            if layout is not None or mode is not None:
                raise ValueError("tensor_spec cannot be combined with explicit layout or mode")
        elif layout is not None or mode is not None:
            tensor_spec = TensorSpec(
                layout=None if layout is None else tuple(layout),
                mode=None if mode is None else tuple(mode),
            )

        _, mode = _resolve_layout_mode(len(input_shape), tensor_spec=tensor_spec)
        shape = tuple(input_shape[dim] for dim in mode)
        return cls(
            dtype=value.dtype,
            shape=shape,
            tensor_spec=tensor_spec,
            name=name,
        )

    @property
    def layout(self) -> tuple[int, ...]:
        return self.jax_layout

    @property
    def mode(self) -> tuple[int, ...]:
        assert self.jax_mode is not None
        return self.jax_mode

    @property
    def array_shape(self) -> tuple[Any, ...]:
        """Return the public JAX shape before ``TensorSpec.mode`` is applied."""

        shape: list[Any] = [None] * self.ndim
        for kernel_dim, array_dim in enumerate(self.mode):
            shape[array_dim] = self.shape[kernel_dim]
        return tuple(shape)


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
        self._check_support()
        object.__setattr__(self, "_is_supported", True)
        return True

    @abstractmethod
    def _check_support(self) -> None:
        """Validate and resolve operation-specific configuration or raise."""

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
        tensor_spec: TensorSpec | None = None,
        layout: Sequence[int] | None = None,
        mode: Sequence[int] | None = None,
        name: str = "",
    ) -> JaxTensorDesc:
        """Return abstract JAX metadata using a declared CUTLASS tensor spec."""

        return JaxTensorDesc.from_value(
            value,
            tensor_spec=tensor_spec,
            layout=layout,
            mode=mode,
            name=name,
        )

    def as_dtype(self, value: Any) -> Any:
        """Return a JAX dtype without retaining a dtype-bearing value."""

        return as_dtype(value)

    def as_optional_dtype(self, value: Any | None) -> Any | None:
        """Return ``None`` or a JAX dtype without retaining the source value."""

        return as_optional_dtype(value)

    def require_dtype(
        self,
        value: Any,
        valid_dtypes: Iterable[Any],
        *,
        name: str | None = None,
        default: Any = _NO_DEFAULT,
    ) -> Any:
        """Return a supported dtype from a dtype-like value or descriptor."""

        return require_dtype(value, valid_dtypes, name=name, default=default)

    def make_optional_tensor_desc(
        self,
        value: Any | None,
        *,
        tensor_spec: TensorSpec | None = None,
        layout: Sequence[int] | None = None,
        mode: Sequence[int] | None = None,
        name: str = "",
    ) -> JaxTensorDesc | None:
        """Return metadata for an optional sample value."""

        if value is None:
            return None
        return self.make_tensor_desc(
            value,
            tensor_spec=tensor_spec,
            layout=layout,
            mode=mode,
            name=name,
        )

    def check_tensor_signature(
        self,
        value: Any,
        expected: JaxTensorDesc,
        *,
        name: str = "",
    ) -> JaxTensorDesc:
        """Validate an invocation-time value against a sample descriptor."""

        actual_shape = require_array(value, name=name or None)
        if actual_shape != expected.array_shape:
            raise ValueError(f"{name} tensor shape mismatch: expected {expected.array_shape}, got {actual_shape}")
        actual = self.make_tensor_desc(
            value,
            tensor_spec=expected.tensor_spec,
            name=name,
        )
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

    @staticmethod
    def freeze_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return an immutable copy suitable for persistent callable state."""

        return MappingProxyType(dict(values))

    def check_tensor_signatures(
        self,
        expected: Mapping[str, JaxTensorDesc | None],
        values: Mapping[str, Any],
    ) -> None:
        """Validate invocation metadata against named sample descriptors."""

        for name, expected_desc in expected.items():
            self.check_optional_tensor_signature(values[name], expected_desc, name=name)

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
    use_static_tensors: bool = True,
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
    validation is delegated to ``cutlass.jax.cutlass_call``. Tensor shapes and
    strides are compile-time constants by default; pass
    ``use_static_tensors=False`` for a future symbolic-shape path. Filled
    buffers are passed as internal aliased inputs so the public JAX function
    remains functional. ``fn`` and every value in ``static_args`` must be
    immutable and hashable because they participate in JAX and CUTLASS
    compilation cache keys.
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


__all__ = [
    "ApiBaseJax",
    "BufferSpec",
    "JaxTensorDesc",
    "TupleDict",
    "as_dtype",
    "as_optional_dtype",
    "call_cutedsl",
    "require_array",
    "require_dtype",
]
