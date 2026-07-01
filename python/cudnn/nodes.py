"""Pure Python node class for cuDNN Frontend graph representation.

Simple, Pythonic design - everything stored directly on the Node.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Any

from .graph_types import NodeType, Tensor

if TYPE_CHECKING:
    from .graph_native import GraphContext


class Node:
    """A single operation node in the computation graph.

    All operation-specific parameters are stored in the `params` dict,
    keeping the design simple and flexible.

    Attributes:
        name: Operation name
        node_type: Type of operation (MATMUL, CONV_FPROP, POINTWISE, etc.)
        inputs: Dict mapping port names to input Tensor
        outputs: Dict mapping port names to output Tensor
        params: Dict of operation-specific parameters (padding, stride, mode, etc.)
        compute_data_type: Data type for computation
    """

    def __init__(
        self,
        name: str,
        node_type: NodeType,
        compute_data_type: Any = None,
    ):
        self.name = name
        self.node_type = node_type
        self.compute_data_type = compute_data_type
        self.inputs: Dict[str, Tensor] = {}
        self.outputs: Dict[str, Tensor] = {}
        self.params: Dict[str, Any] = {}

    def validate(self) -> None:
        """Validate node configuration."""
        for port_name, tensor in self.inputs.items():
            if tensor is None:
                raise ValueError(f"Node '{self.name}': Input '{port_name}' is None")
        for port_name, tensor in self.outputs.items():
            if tensor is None:
                raise ValueError(f"Node '{self.name}': Output '{port_name}' is None")

        if self.node_type == NodeType.MATMUL:
            self._validate_matmul()

    def infer_properties(self, context: "GraphContext") -> None:
        """Infer unset tensor properties from context and inputs."""
        # Fill data types from context
        for tensor in self.inputs.values():
            if tensor and tensor.data_type is None:
                tensor.data_type = context.intermediate_data_type if tensor.is_virtual else context.io_data_type

        for tensor in self.outputs.values():
            if tensor and tensor.data_type is None:
                tensor.data_type = context.intermediate_data_type if tensor.is_virtual else context.io_data_type

        # Operation-specific inference
        if self.node_type == NodeType.MATMUL:
            self._infer_matmul()
        elif self.node_type == NodeType.CONV_FPROP:
            self._infer_conv_fprop()
        elif self.node_type == NodeType.CONV_DGRAD:
            self._infer_conv_dgrad()
        elif self.node_type == NodeType.POINTWISE:
            self._infer_pointwise()
        elif self.node_type == NodeType.SDPA:
            self._infer_sdpa()
        elif self.node_type == NodeType.SDPA_BWD:
            self._infer_sdpa_backward()

    def _validate_matmul(self) -> None:
        """Validate matmul dimensions: C = A @ B."""
        a = self.inputs.get("A")
        b = self.inputs.get("B")
        if not (a and b and a.dim and b.dim):
            return
        if a.dim[-1] != b.dim[-2]:
            raise ValueError(f"Node '{self.name}': Inner dimensions must match for matmul: " f"A{a.dim} @ B{b.dim}")

    def _infer_matmul(self) -> None:
        """Infer output dims for matmul: C = A @ B."""
        a = self.inputs.get("A")
        b = self.inputs.get("B")
        c = self.outputs.get("C")

        if not (a and b and c):
            return

        if not c.dim and a.dim and b.dim:
            # Output shape: [..., M, N] where M=A[-2], N=B[-1]
            ndim = max(len(a.dim), len(b.dim))
            c_dim = [1] * ndim

            # Last two dims: M from A, N from B
            if len(a.dim) >= 2:
                c_dim[-2] = a.dim[-2]
            if len(b.dim) >= 2:
                c_dim[-1] = b.dim[-1]

            # Broadcast batch dims
            for i in range(ndim - 2):
                a_idx = i - (ndim - len(a.dim))
                b_idx = i - (ndim - len(b.dim))
                a_val = a.dim[a_idx] if 0 <= a_idx < len(a.dim) - 2 else 1
                b_val = b.dim[b_idx] if 0 <= b_idx < len(b.dim) - 2 else 1
                c_dim[i] = max(a_val, b_val)

            c.dim = c_dim

        if not c.stride and c.dim:
            c.stride = _row_major_stride(c.dim)

    def _infer_conv_fprop(self) -> None:
        """Infer output dims for convolution."""
        x = self.inputs.get("X")
        w = self.inputs.get("W")
        y = self.outputs.get("Y")

        if not (x and w and y):
            return

        if not y.dim and x.dim and w.dim:
            pre_pad = self.params.get("pre_padding", [0] * (len(x.dim) - 2))
            post_pad = self.params.get("post_padding", [0] * (len(x.dim) - 2))
            stride = self.params.get("stride", [1] * (len(x.dim) - 2))
            dilation = self.params.get("dilation", [1] * (len(x.dim) - 2))

            y_dim = [0] * len(x.dim)
            y_dim[0] = x.dim[0]  # N
            y_dim[1] = w.dim[0]  # K

            for i in range(2, len(x.dim)):
                idx = i - 2
                eff_filter = (w.dim[i] - 1) * dilation[idx] + 1
                y_dim[i] = (x.dim[i] + pre_pad[idx] + post_pad[idx] - eff_filter) // stride[idx] + 1

            y.dim = y_dim

        if not y.stride and y.dim:
            y.stride = _row_major_stride(y.dim)

    def _infer_conv_dgrad(self) -> None:
        """Infer output dims for conv data gradient."""
        dy = self.inputs.get("DY")
        w = self.inputs.get("W")
        dx = self.outputs.get("DX")

        if not (dy and w and dx):
            return

        if not dx.dim and dy.dim and w.dim:
            pre_pad = self.params.get("pre_padding", [0] * (len(dy.dim) - 2))
            post_pad = self.params.get("post_padding", [0] * (len(dy.dim) - 2))
            stride = self.params.get("stride", [1] * (len(dy.dim) - 2))
            dilation = self.params.get("dilation", [1] * (len(dy.dim) - 2))

            # Reverse of conv_fprop: compute input size from output size
            dx_dim = [0] * len(dy.dim)
            dx_dim[0] = dy.dim[0]  # N
            dx_dim[1] = w.dim[1]  # C (input channels from filter)

            for i in range(2, len(dy.dim)):
                idx = i - 2
                eff_filter = (w.dim[i] - 1) * dilation[idx] + 1
                # x_dim[i] = (y_dim[i] - 1) * stride + eff_filter - pre_pad - post_pad
                dx_dim[i] = (dy.dim[i] - 1) * stride[idx] + eff_filter - pre_pad[idx] - post_pad[idx]

            dx.dim = dx_dim

        if not dx.stride and dx.dim:
            dx.stride = _row_major_stride(dx.dim)

    def _infer_pointwise(self) -> None:
        """Infer output dims for pointwise (broadcast inputs)."""
        out = self.outputs.get("OUT_0")
        if not out:
            return

        if not out.dim:
            # Find largest input shape
            max_dim = []
            for tensor in self.inputs.values():
                if tensor and tensor.dim:
                    if len(tensor.dim) > len(max_dim):
                        max_dim = tensor.dim.copy()
                    elif len(tensor.dim) == len(max_dim):
                        max_dim = [max(a, b) for a, b in zip(max_dim, tensor.dim)]
            if max_dim:
                out.dim = max_dim

        if not out.stride and out.dim:
            out.stride = _row_major_stride(out.dim)

    def _infer_sdpa(self) -> None:
        """Infer output dims for scaled dot-product attention.

        O has same shape as V: [B, H, S_kv, D] or [B, S_kv, H, D]
        stats has shape [B, H, S_q, 1] for softmax stats
        """
        q = self.inputs.get("Q")
        v = self.inputs.get("V")
        o = self.outputs.get("O")
        stats = self.outputs.get("stats")

        if not (q and v and o):
            return

        # Output O has same shape as V
        if not o.dim and v.dim:
            o.dim = v.dim.copy()
        if not o.stride and o.dim:
            o.stride = _row_major_stride(o.dim)

        # Stats output: [B, H, S_q, 1]
        if stats and not stats.dim and q.dim:
            # Assuming [B, H, S_q, D] layout
            stats.dim = [q.dim[0], q.dim[1], q.dim[2], 1]
        if stats and not stats.stride and stats.dim:
            stats.stride = _row_major_stride(stats.dim)

    def _infer_sdpa_backward(self) -> None:
        """Infer output dims for SDPA backward.

        dQ has same shape as Q
        dK has same shape as K
        dV has same shape as V
        """
        q = self.inputs.get("Q")
        k = self.inputs.get("K")
        v = self.inputs.get("V")
        dq = self.outputs.get("dQ")
        dk = self.outputs.get("dK")
        dv = self.outputs.get("dV")

        if dq and not dq.dim and q and q.dim:
            dq.dim = q.dim.copy()
        if dq and not dq.stride and dq.dim:
            dq.stride = _row_major_stride(dq.dim)

        if dk and not dk.dim and k and k.dim:
            dk.dim = k.dim.copy()
        if dk and not dk.stride and dk.dim:
            dk.stride = _row_major_stride(dk.dim)

        if dv and not dv.dim and v and v.dim:
            dv.dim = v.dim.copy()
        if dv and not dv.stride and dv.dim:
            dv.stride = _row_major_stride(dv.dim)

    def __repr__(self) -> str:
        return f"Node({self.name!r}, {self.node_type.name})"


def _row_major_stride(dim: List[int]) -> List[int]:
    """Compute row-major (C-contiguous) strides."""
    if not dim:
        return []
    stride = [1] * len(dim)
    for i in range(len(dim) - 2, -1, -1):
        stride[i] = stride[i + 1] * dim[i + 1]
    return stride
