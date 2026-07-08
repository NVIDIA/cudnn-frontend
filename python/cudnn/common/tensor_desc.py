# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Framework-neutral tensor metadata for operation kernels."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from operator import index
from typing import Any, Generic, TypeVar

from .. import data_type

DTypeT = TypeVar("DTypeT")


@dataclass(frozen=True)
class TensorDesc(Generic[DTypeT]):
    """Framework-neutral tensor signature used by operation kernels.

    ``stride_order`` lists dimension indices from the smallest stride to the
    largest. Framework adapters may extend this descriptor with storage or
    lowering metadata, but operation kernels only consume the fields defined
    here and :attr:`cudnn_dtype`. ``init_value`` is a scalar used to initialize
    inferred output or workspace storage; ``None`` leaves the storage
    uninitialized.
    """

    dtype: DTypeT
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    stride_order: tuple[int, ...]
    ndim: int = field(init=False)
    name: str = ""
    init_value: bool | int | float | None = None

    def __post_init__(self) -> None:
        shape = self._integer_tuple(self.shape, "shape")
        stride = self._integer_tuple(self.stride, "stride")
        stride_order = self._integer_tuple(self.stride_order, "stride_order")
        ndim = len(shape)

        if self.init_value is not None and not isinstance(self.init_value, (bool, Real)):
            raise TypeError(f"TensorDesc.init_value must be a real scalar or None, got {self.init_value!r}")

        if any(size < 0 for size in shape):
            raise ValueError(f"TensorDesc.shape entries must be non-negative, got {shape}")
        if any(value < 0 for value in stride):
            raise ValueError(f"TensorDesc.stride entries must be non-negative, got {stride}")
        if len(stride) != ndim:
            raise ValueError(f"Stride rank mismatch: expected {ndim}, got {len(stride)}")
        if len(stride_order) != ndim:
            raise ValueError(f"Stride order rank mismatch: expected {ndim}, got {len(stride_order)}")
        if tuple(sorted(stride_order)) != tuple(range(ndim)):
            raise ValueError(f"Stride order must be a permutation of [0, {ndim - 1}], got {stride_order}")
        ordered_strides = tuple(stride[dimension] for dimension in stride_order)
        if any(left > right for left, right in zip(ordered_strides, ordered_strides[1:])):
            raise ValueError(f"Stride order {stride_order} is inconsistent with stride {stride}")

        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "stride", stride)
        object.__setattr__(self, "stride_order", stride_order)
        object.__setattr__(self, "ndim", ndim)

    @staticmethod
    def _integer_tuple(values, field_name: str) -> tuple[int, ...]:
        normalized = []
        for value in values:
            if isinstance(value, bool):
                raise TypeError(f"TensorDesc.{field_name} entries must be integers, got {value!r}")
            try:
                normalized.append(index(value))
            except TypeError as error:
                raise TypeError(f"TensorDesc.{field_name} entries must be integers, got {value!r}") from error
        return tuple(normalized)

    @property
    def cudnn_dtype(self) -> data_type:
        """Return the canonical dtype consumed by operation kernels.

        Framework-specific subclasses override this property when ``dtype`` is
        native to that framework.
        """

        if not isinstance(self.dtype, data_type):
            raise TypeError(f"{type(self).__name__}.dtype must provide a cudnn.data_type mapping, " f"got {type(self.dtype).__name__}")
        return self.dtype

    def compact_like(
        self,
        *,
        cudnn_dtype: data_type,
        shape: tuple[int, ...],
        stride_order: tuple[int, ...] | None = None,
        name: str = "",
        init_value: bool | int | float | None = None,
    ) -> "TensorDesc[Any]":
        """Create a compact descriptor derived from this descriptor.

        The framework-neutral implementation returns a canonical descriptor.
        Framework-specific descriptors override this method to preserve their
        native dtype and allocation metadata.
        """

        return make_compact_tensor_desc(
            dtype=cudnn_dtype,
            shape=shape,
            stride_order=stride_order,
            name=name,
            init_value=init_value,
        )

    def is_compact(self, stride_order: tuple[int, ...] | None = None) -> bool:
        """Return whether the descriptor is contiguous in the given order.

        Size-one dimensions may carry any non-negative stride because they do
        not change the addressed storage. When no order is supplied, the
        descriptor's canonical ``stride_order`` is used.
        """

        order = self.stride_order if stride_order is None else self._integer_tuple(stride_order, "stride_order")
        if len(order) != self.ndim or tuple(sorted(order)) != tuple(range(self.ndim)):
            raise ValueError(f"Stride order must be a permutation of [0, {self.ndim - 1}], got {order}")

        expected_stride = 1
        for dimension in order:
            size = self.shape[dimension]
            if size != 1 and self.stride[dimension] != expected_stride:
                return False
            expected_stride *= max(size, 1)
        return True


def make_compact_tensor_desc(
    *,
    dtype: data_type,
    shape: tuple[int, ...],
    stride_order: tuple[int, ...] | None = None,
    name: str = "",
    init_value: bool | int | float | None = None,
) -> TensorDesc[data_type]:
    """Construct a framework-neutral descriptor for a compact tensor.

    ``stride_order`` lists dimensions from fastest varying to slowest varying.
    It defaults to the conventional row-major order.
    """

    if not isinstance(dtype, data_type):
        raise TypeError(f"dtype must be a cudnn.data_type, got {type(dtype).__name__}")

    shape = TensorDesc._integer_tuple(shape, "shape")
    if any(size < 0 for size in shape):
        raise ValueError(f"TensorDesc.shape entries must be non-negative, got {shape}")

    if stride_order is None:
        stride_order = tuple(reversed(range(len(shape))))
    else:
        stride_order = TensorDesc._integer_tuple(stride_order, "stride_order")

    if len(stride_order) != len(shape):
        raise ValueError(f"Stride order rank mismatch: expected {len(shape)}, got {len(stride_order)}")
    if tuple(sorted(stride_order)) != tuple(range(len(shape))):
        raise ValueError(f"Stride order must be a permutation of [0, {len(shape) - 1}], got {stride_order}")

    stride = [0] * len(shape)
    running = 1
    for dimension in stride_order:
        stride[dimension] = running
        running *= max(shape[dimension], 1)

    return TensorDesc(
        dtype=dtype,
        shape=shape,
        stride=tuple(stride),
        stride_order=stride_order,
        name=name,
        init_value=init_value,
    )


__all__ = ["TensorDesc", "make_compact_tensor_desc"]
