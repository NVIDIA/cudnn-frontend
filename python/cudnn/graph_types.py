# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure Python data types for cuDNN Frontend graph representation.

This module provides Python dataclasses for tensor attributes and node types,
enabling native Python access to graph structure.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union


class NodeType(Enum):
    """Operation node types. Maps to INode::Type in node_interface.h."""

    # Only the op types exercised by this version are listed. Add more as
    # needed, following the block-scale / MoE / reduction examples (enum entry
    # here + a builder in graph_native + inference in nodes + lowering).
    COMPOSITE = auto()
    CONV_FPROP = auto()
    CONV_DGRAD = auto()
    CONV_WGRAD = auto()
    GENSTATS = auto()
    RESHAPE = auto()
    SLICE = auto()
    CONCATENATE = auto()
    TRANSPOSE = auto()
    ROPE = auto()
    ROPE_BWD = auto()
    MOE_GROUPED_MATMUL_BWD = auto()
    MATMUL = auto()
    MATMUL_FP8 = auto()
    POINTWISE = auto()
    REDUCTION = auto()
    RMSNORM = auto()
    RMSNORM_BWD = auto()
    LAYERNORM = auto()
    LAYERNORM_BWD = auto()
    ADALAYERNORM = auto()
    ADALAYERNORM_BWD = auto()
    INSTANCENORM = auto()
    INSTANCENORM_BWD = auto()
    BATCHNORM = auto()
    BATCHNORM_INFERENCE = auto()
    BATCHNORM_BWD = auto()
    SDPA = auto()
    SDPA_BWD = auto()
    SDPA_FP8 = auto()
    SDPA_FP8_BWD = auto()
    SDPA_MXFP8 = auto()
    SDPA_MXFP8_BWD = auto()
    MOE_GROUPED_MATMUL = auto()
    BLOCK_SCALE_QUANTIZE = auto()
    BLOCK_SCALE_DEQUANTIZE = auto()


@dataclass(eq=False)  # identity-based hash/eq: uid/name are mutable
class Tensor:
    """Pure Python representation of tensor attributes.

    Mirrors cudnn_frontend::graph::Tensor_attributes from graph_properties.h.

    Attributes:
        name: Tensor identifier
        data_type: Data type (uses cudnn.data_type values)
        dim: Dimensions of the tensor
        stride: Memory strides
        is_virtual: True if tensor is an intermediate (not I/O)
        is_pass_by_value: True if tensor is a scalar passed at execution
        pass_by_value: Embedded constant value (for fused scalars)
        uid: Unique identifier for backend mapping
        uid_assigned: True if UID was explicitly assigned
        reordering_type: Memory layout transformation type
        ragged_offset: Tensor for variable-length tensor offsets
    """

    name: str = ""
    data_type: Any = None
    dim: List[int] = field(default_factory=list)
    stride: List[int] = field(default_factory=list)
    is_virtual: bool = False
    is_pass_by_value: bool = False
    pass_by_value: Optional[Union[int, float]] = None
    uid: int = 0
    uid_assigned: bool = False
    # user-assigned vs IR-inferred layout: only USER-assigned dim/stride are
    # pushed to the lowered C++ output tensors — inferred values are
    # provisional (row-major) and the backend applies its own classic
    # per-op layout inference (e.g. channels-last conv). Internal inference
    # writes the attributes directly and leaves these False.
    dim_assigned: bool = False
    stride_assigned: bool = False
    reordering_type: Any = None
    ragged_offset: Optional["Tensor"] = None
    ragged_offset_multiplier: int = 1
    scalar_type: Any = None  # cudnn.scalar_type for tensor_scalar-created scalars
    # weakref to the owning graph (set at registration): identity mutations
    # (set_name / set_uid) delegate to the graph so its indexes stay coherent.
    owner: Any = field(default=None, repr=False)

    def __setattr__(self, name, value):
        # direct attribute writes freeze with the owning graph (the fluent
        # setters are guarded separately and give a richer error)
        if getattr(self, "_frozen", False) and name != "_frozen":
            raise RuntimeError(f"cannot set Tensor.{name}: the owning graph is frozen after lowering/planning")
        object.__setattr__(self, name, value)

    def _guard(self, what: str = "mutate a tensor attribute") -> None:
        g = self.owner() if self.owner is not None else None
        if g is not None:
            g._check_mutable(what)

    def set_output(self, value: bool) -> "Tensor":
        """Mark this tensor as an output (non-virtual) or intermediate (virtual)."""
        self._guard()
        self.is_virtual = not value
        return self

    def set_data_type(self, dtype: Any) -> "Tensor":
        """Set the data type."""
        self._guard()
        self.data_type = dtype
        return self

    def set_name(self, name: str) -> "Tensor":
        """Set the tensor name (graph-owned tensors re-index atomically)."""
        g = self.owner() if self.owner is not None else None
        if g is not None:
            g._rename_tensor(self, name)
        else:
            self.name = name
        return self

    def set_dim(self, dim: List[int]) -> "Tensor":
        """Set the tensor dimensions (user-assigned: pushed at lowering)."""
        self._guard()
        self.dim = list(dim)
        self.dim_assigned = True
        return self

    def set_stride(self, stride: List[int]) -> "Tensor":
        """Set the tensor strides (user-assigned: pushed at lowering)."""
        self._guard()
        self.stride = list(stride)
        self.stride_assigned = True
        return self

    def set_uid(self, uid: int) -> "Tensor":
        """Set the tensor UID (graph-owned tensors re-index atomically; a
        colliding auto-assigned uid is renumbered, user-user conflicts raise)."""
        g = self.owner() if self.owner is not None else None
        if g is not None:
            g._reuid_tensor(self, uid)
        else:
            self.uid = uid
            self.uid_assigned = True
        return self

    def set_ragged_offset(self, ragged_offset: "Tensor") -> "Tensor":
        """Set the ragged-offset tensor (variable-length layouts)."""
        self._guard()
        self.ragged_offset = ragged_offset
        return self

    def set_ragged_offset_multiplier(self, multiplier: int) -> "Tensor":
        """Set the ragged-offset unit size in tensor elements."""
        self._guard()
        self.ragged_offset_multiplier = multiplier
        return self

    def set_reordering_type(self, reordering_type: Any) -> "Tensor":
        """Set the memory reordering layout (e.g. F8_128x4)."""
        self._guard()
        self.reordering_type = reordering_type
        return self

    def set_is_pass_by_value(self, value: bool) -> "Tensor":
        """Mark the tensor as a host pass-by-value scalar."""
        self._guard()
        self.is_pass_by_value = value
        return self

    def set_is_virtual(self, value: bool) -> "Tensor":
        """Set virtualness directly (classic parity; inverse of set_output)."""
        self._guard()
        self.is_virtual = value
        return self

    def get_uid(self) -> int:
        return self.uid

    def get_name(self) -> str:
        return self.name

    def get_dim(self) -> List[int]:
        return list(self.dim)  # a copy, like the classic pybind getter

    def get_stride(self) -> List[int]:
        return list(self.stride)  # a copy, like the classic pybind getter

    def get_data_type(self) -> Any:
        # classic parity: the pybind getter returns the cudnn enum even when
        # the user set a torch dtype (classic converts at set time)
        dt = self.data_type
        if dt is not None and type(dt).__module__ == "torch":
            from .datatypes import _torch_to_cudnn_data_type

            return _torch_to_cudnn_data_type(dt)
        return self.data_type

    def get_is_virtual(self) -> bool:
        return self.is_virtual

    def get_is_pass_by_value(self) -> bool:
        return self.is_pass_by_value

    def get_reordering_type(self) -> Any:
        return self.reordering_type

    def get_ragged_offset_multiplier(self) -> int:
        return self.ragged_offset_multiplier

    def validate(self) -> None:
        """Validate tensor configuration."""
        if not self.dim:
            raise ValueError(f"Tensor '{self.name}' dims not set.")
        if not self.stride:
            raise ValueError(f"Tensor '{self.name}' strides not set.")
        if len(self.dim) != len(self.stride):
            raise ValueError(f"Tensor '{self.name}' dim/stride length mismatch: " f"{len(self.dim)} vs {len(self.stride)}")
        if self.is_virtual and self.is_pass_by_value:
            raise ValueError(f"Tensor '{self.name}' can't be both virtual and pass_by_value.")

    # NOTE: hash/eq are object identity (dataclass eq=False). uid and name are
    # mutable, so value-based hashing would violate the dict-key invariant.
