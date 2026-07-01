"""Pure Python data types for cuDNN Frontend graph representation.

This module provides Python dataclasses for tensor attributes and node types,
enabling native Python access to graph structure.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union


class NodeType(Enum):
    """Operation node types. Maps to INode::Type in node_interface.h."""

    COMPOSITE = auto()
    BATCHNORM = auto()
    BATCHNORM_INFERENCE = auto()
    CONV_DGRAD = auto()
    CONV_FPROP = auto()
    CONV_WGRAD = auto()
    DBN = auto()
    DIN = auto()
    DLN = auto()
    DRN = auto()
    GENSTATS = auto()
    INSTANCENORM = auto()
    LAYERNORM = auto()
    MATMUL = auto()
    MATMUL_FP8 = auto()
    POINTWISE = auto()
    REDUCTION = auto()
    RESAMPLE = auto()
    RESHAPE = auto()
    RMSNORM = auto()
    SDPA = auto()
    SDPA_BWD = auto()
    SDPA_FP8 = auto()
    SLICE = auto()
    ADALAYERNORM = auto()
    BN_FINALIZE = auto()
    CONCATENATE = auto()
    MOE_GROUPED_MATMUL = auto()
    BLOCK_SCALE_QUANTIZE = auto()
    BLOCK_SCALE_DEQUANTIZE = auto()


@dataclass
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
    reordering_type: Any = None
    ragged_offset: Optional["Tensor"] = None

    def set_output(self, value: bool) -> "Tensor":
        """Mark this tensor as an output (non-virtual) or intermediate (virtual)."""
        self.is_virtual = not value
        return self

    def set_data_type(self, dtype: Any) -> "Tensor":
        """Set the data type."""
        self.data_type = dtype
        return self

    def set_name(self, name: str) -> "Tensor":
        """Set the tensor name."""
        self.name = name
        return self

    def set_dim(self, dim: List[int]) -> "Tensor":
        """Set the tensor dimensions."""
        self.dim = dim
        return self

    def set_stride(self, stride: List[int]) -> "Tensor":
        """Set the tensor strides."""
        self.stride = stride
        return self

    def set_uid(self, uid: int) -> "Tensor":
        """Set the tensor UID."""
        self.uid = uid
        self.uid_assigned = True
        return self

    def get_uid(self) -> int:
        return self.uid

    def get_name(self) -> str:
        return self.name

    def get_dim(self) -> List[int]:
        return self.dim

    def get_stride(self) -> List[int]:
        return self.stride

    def get_data_type(self) -> Any:
        return self.data_type

    def get_is_virtual(self) -> bool:
        return self.is_virtual

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

    def __hash__(self) -> int:
        """Hash based on UID for use as dict key."""
        return hash(self.uid)

    def __eq__(self, other: object) -> bool:
        """Equality based on UID."""
        if isinstance(other, Tensor):
            return self.uid == other.uid
        return False
