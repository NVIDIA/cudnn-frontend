# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX adapters for framework-neutral FE-OSS API metadata."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, final

import jax.numpy as jnp

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


__all__ = ["ApiBaseJax", "JaxTensorDesc"]
