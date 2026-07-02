# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Framework-neutral metadata and validation support for FE-OSS APIs.

The eager Torch lifecycle is loaded only when one of its compatibility names
is requested.  JAX wrappers can therefore reuse :class:`TensorDesc` and
:class:`ApiBase` without importing Torch, CUDA bindings, or CuTe DSL.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from importlib import import_module
import logging
from typing import Any, TYPE_CHECKING

from ._experimental_warnings import warn_experimental_api_once

if TYPE_CHECKING:
    from .api_base_torch import APIBase, ApiBaseTorch, TorchTensorDesc


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def is_power_of_2(n: int) -> bool:
    """Return whether ``n`` is a positive power of two."""

    return n > 0 and (n & (n - 1)) == 0


_DTYPE_ALIASES = {
    "half": "float16",
    "float": "float32",
    "double": "float64",
    "long": "int64",
    # Torch names the storage type; the logical element remains E2M1.
    "float4_e2m1fn_x2": "float4_e2m1fn",
    # CUTLASS numeric classes omit separators used by framework dtypes.
    "float4e2m1fn": "float4_e2m1fn",
    "float8e4m3fn": "float8_e4m3fn",
    "float8e5m2": "float8_e5m2",
    "float8e8m0fnu": "float8_e8m0fnu",
}

_DTYPE_BITS = {
    "bool": 1,
    "float4_e2m1fn": 4,
    "float8_e4m3fn": 8,
    "float8_e5m2": 8,
    "float8_e8m0fnu": 8,
    "int8": 8,
    "uint8": 8,
    "bfloat16": 16,
    "float16": 16,
    "int16": 16,
    "uint16": 16,
    "float32": 32,
    "int32": 32,
    "uint32": 32,
    "float64": 64,
    "int64": 64,
    "uint64": 64,
}


def _storage_dtype_name(dtype: Any) -> str:
    name = getattr(dtype, "name", None)
    if not isinstance(name, str):
        name = getattr(dtype, "__name__", None)
    if not isinstance(name, str):
        name = str(dtype)
    return name.rsplit(".", 1)[-1].lower()


def canonical_dtype_name(dtype: Any) -> str:
    """Return a stable logical dtype name without importing a framework."""

    name = _storage_dtype_name(dtype)
    return _DTYPE_ALIASES.get(name, name)


def dtype_bits(dtype: Any) -> int | None:
    """Return the logical element width when it can be determined safely."""

    name = canonical_dtype_name(dtype)
    if name in _DTYPE_BITS:
        return _DTYPE_BITS[name]
    itemsize = getattr(dtype, "itemsize", None)
    if isinstance(itemsize, int) and itemsize > 0:
        return itemsize * 8
    return None


def _compact_stride(shape: tuple[Any, ...], order: tuple[int, ...]) -> tuple[Any, ...]:
    stride: list[Any] = [None] * len(shape)
    running: Any = 1
    for dim in order:
        stride[dim] = running
        running *= shape[dim]
    return tuple(stride)


@dataclass(frozen=True, kw_only=True)
class TensorDesc:
    """Logical tensor metadata shared by framework adapters.

    ``shape`` and ``stride`` are expressed in the mode order presented to the
    kernel.  For Torch that is the observed tensor order unless a wrapper
    explicitly transforms the descriptor.  For JAX the adapter applies the
    declared ``TensorSpec.mode`` permutation.  JAX strides describe the compact
    layout requested from XLA; they are not inspected from a physical buffer.
    ``dtype_name`` is logical; ``storage_dtype_name`` and ``packing`` retain
    adapter-specific storage details such as Torch's packed-uint8 FP4 ABI.
    """

    dtype: Any
    shape: tuple[Any, ...]
    stride: tuple[Any, ...] | None = None
    stride_order: tuple[int, ...] | None = None
    packing: str = "native"
    name: str = ""
    ndim: int = field(init=False)
    storage_dtype_name: str = field(init=False)
    dtype_name: str = field(init=False)
    element_bits: int | None = field(init=False)

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        stride = None if self.stride is None else tuple(self.stride)
        stride_order = None if self.stride_order is None else tuple(self.stride_order)
        ndim = len(shape)
        packing = self.packing
        storage_dtype_name = _storage_dtype_name(self.dtype)
        if storage_dtype_name == "float4_e2m1fn_x2" and packing == "native":
            packing = "fp4x2"
        if packing not in {"native", "fp4x2"}:
            raise ValueError(f"Unsupported tensor packing {packing!r}")

        if stride is not None and len(stride) != ndim:
            raise ValueError(f"Stride rank mismatch: expected {ndim}, got {len(stride)}")
        if stride_order is not None:
            if len(stride_order) != ndim:
                raise ValueError(f"Stride order rank mismatch: expected {ndim}, got {len(stride_order)}")
            if tuple(sorted(stride_order)) != tuple(range(ndim)):
                raise ValueError(f"Stride order must be a permutation of [0, {ndim - 1}], got {stride_order}")
        if stride is None and stride_order is not None:
            stride = _compact_stride(shape, stride_order)

        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "stride", stride)
        object.__setattr__(self, "stride_order", stride_order)
        object.__setattr__(self, "ndim", ndim)
        object.__setattr__(self, "packing", packing)
        if packing == "fp4x2" and storage_dtype_name not in {
            "uint8",
            "float4_e2m1fn",
            "float4_e2m1fn_x2",
        }:
            raise ValueError(f"fp4x2 packing requires uint8 or float4 storage, got {storage_dtype_name}")
        dtype_name = "float4_e2m1fn" if packing == "fp4x2" else canonical_dtype_name(self.dtype)
        object.__setattr__(self, "storage_dtype_name", storage_dtype_name)
        object.__setattr__(self, "dtype_name", dtype_name)
        object.__setattr__(self, "element_bits", _DTYPE_BITS.get(dtype_name, dtype_bits(self.dtype)))

    def size(self, dim: int | None = None) -> tuple[Any, ...] | Any:
        """Return the logical shape or one dimension."""

        if dim is None:
            return self.shape
        return self.shape[dim]

    @property
    def is_fp4(self) -> bool:
        return self.dtype_name.startswith("float4_")

    @property
    def is_fp8(self) -> bool:
        return self.dtype_name.startswith("float8_")

    @property
    def is_f16(self) -> bool:
        return self.dtype_name in {"float16", "bfloat16"}


def _shape_of(value: TensorDesc | Sequence[Any] | Any) -> tuple[Any, ...]:
    if isinstance(value, TensorDesc) or hasattr(value, "shape"):
        return tuple(value.shape)
    return tuple(value)


def _dtype_name_of(value: TensorDesc | Any) -> str:
    if isinstance(value, TensorDesc):
        return value.dtype_name
    if not isinstance(value, type) and hasattr(value, "dtype"):
        value = value.dtype
    return canonical_dtype_name(value)


class ApiBase:
    """Framework-neutral base for descriptor validation.

    This class deliberately has no compile or execute lifecycle.  Framework
    adapters own those policies while operation validators share descriptors.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._is_supported = False

    def _warn_experimental_api(self) -> None:
        warn_experimental_api_once(self._logger, self.__class__.__name__)

    def check_tensor_shape(
        self,
        tensor_or_shape: TensorDesc | Sequence[Any] | Any,
        expected: Sequence[Any] | Collection[Sequence[Any]],
        name: str = "",
    ) -> tuple[Any, ...]:
        """Validate a logical tensor shape and return it as a tuple."""

        actual = _shape_of(tensor_or_shape)
        if isinstance(expected, tuple):
            if actual != expected:
                raise ValueError(f"{name} tensor shape mismatch: expected {expected}, got {actual}")
        else:
            choices = [tuple(shape) for shape in expected]
            if actual not in choices:
                raise ValueError(f"{name} tensor shape mismatch: expected one of {choices}, got {actual}")
        return actual

    def check_tensor_stride(
        self,
        tensor_or_stride: TensorDesc | Sequence[Any] | None,
        stride: Sequence[Any] | Collection[Sequence[Any]] | None = None,
        stride_order: Sequence[int] | Collection[Sequence[int]] | None = None,
        name: str = "",
        extra_error_msg: str = "",
    ) -> tuple[tuple[Any, ...] | None, tuple[int, ...] | None]:
        """Validate a kernel-visible layout contract."""

        if tensor_or_stride is None:
            return None, None
        elif isinstance(tensor_or_stride, TensorDesc):
            actual_stride = tensor_or_stride.stride
            actual_order = tensor_or_stride.stride_order
        else:
            actual_stride = tuple(tensor_or_stride)
            actual_order = tuple(dim for dim, _ in sorted(enumerate(actual_stride), key=lambda item: item[1]))

        def with_context(message: str) -> str:
            if extra_error_msg:
                return f"{message}: {extra_error_msg}"
            return message

        if stride is not None:
            if not isinstance(stride, (tuple, list)):
                raise ValueError(with_context(f"Expected stride to be a tuple or list, got {type(stride)}"))
            expected = tuple(stride) if isinstance(stride, tuple) else [tuple(item) for item in stride]
            stride_matches = actual_stride in expected if isinstance(expected, list) else actual_stride == expected
            if not stride_matches:
                qualifier = "one of " if isinstance(expected, list) else ""
                raise ValueError(with_context(f"{name} tensor stride mismatch: expected {qualifier}{expected}, got {actual_stride}"))
        if stride_order is not None:
            if not isinstance(stride_order, (tuple, list)):
                raise ValueError(with_context(f"Expected stride order to be a tuple or list, got {type(stride_order)}"))
            expected_order = tuple(stride_order) if isinstance(stride_order, tuple) else [tuple(item) for item in stride_order]
            order_matches = actual_order in expected_order if isinstance(expected_order, list) else actual_order == expected_order
            if not order_matches:
                qualifier = "one of " if isinstance(expected_order, list) else ""
                raise ValueError(with_context(f"{name} tensor stride order mismatch: expected {qualifier}{expected_order}, got {actual_order}"))
        return actual_stride, actual_order

    def check_dtype(
        self,
        tensor_or_dtype: TensorDesc | Any,
        expected: Any | Collection[Any],
        name: str = "",
        extra_error_msg: str = "",
    ) -> str:
        """Validate a dtype through its framework-neutral logical name."""

        actual = _dtype_name_of(tensor_or_dtype)
        if isinstance(expected, Collection) and not isinstance(expected, (str, bytes)):
            choices = tuple(_dtype_name_of(item) for item in expected)
        else:
            choices = (_dtype_name_of(expected),)
        if actual not in choices:
            supported = ", ".join(choices)
            error_msg = f"{name} dtype mismatch: expected one of {{{supported}}}, got {actual}"
            if extra_error_msg:
                error_msg += f": {extra_error_msg}"
            raise ValueError(error_msg)
        return actual

    # Compatibility spellings used by the existing Torch wrappers.
    _check_tensor_shape = check_tensor_shape
    _check_tensor_stride = check_tensor_stride
    _check_dtype = check_dtype

    @staticmethod
    def value_error_if(condition: bool, error_msg: str) -> None:
        if condition:
            raise ValueError(error_msg)

    @staticmethod
    def not_implemented_error_if(condition: bool, error_msg: str) -> None:
        if condition:
            raise NotImplementedError(error_msg)

    @staticmethod
    def runtime_error_if(condition: bool, error_msg: str) -> None:
        if condition:
            raise RuntimeError(error_msg)

    _value_error_if = value_error_if
    _not_implemented_error_if = not_implemented_error_if
    _runtime_error_if = runtime_error_if

    def _ensure_support_checked(self) -> None:
        """Run an adapter's support check once before lowering or compilation."""

        if self._is_supported:
            return
        check_support = getattr(self, "check_support", None)
        if check_support is None:
            raise NotImplementedError(f"{self.__class__.__name__} does not define check_support()")
        self._logger.info(f"{self.__class__.__name__}: check_support not previously called, calling now")
        if not check_support():
            raise AssertionError("Unsupported configuration")


class TupleDict(dict):
    """Dictionary with value-order iteration and integer indexing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._keys = list(self.keys())

    def __iter__(self):
        return (self[key] for key in self._keys)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            if key < 0 or key >= len(self._keys):
                raise IndexError(f"index {key} out of range for TupleDict with {len(self._keys)} items")
            key = self._keys[key]
        return super().__getitem__(key)


_TORCH_EXPORTS = {"TorchTensorDesc", "ApiBaseTorch", "APIBase"}


def __getattr__(name: str) -> Any:
    if name not in _TORCH_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(".api_base_torch", __package__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_TORCH_EXPORTS))


__all__ = [
    "APIBase",
    "ApiBase",
    "ApiBaseTorch",
    "TensorDesc",
    "TorchTensorDesc",
    "TupleDict",
    "canonical_dtype_name",
    "ceil_div",
    "dtype_bits",
    "is_power_of_2",
]
