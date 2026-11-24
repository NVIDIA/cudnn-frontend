# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-reduction BF16 matmul and matmul + column bias for RL forward graphs."""

from dataclasses import dataclass

from cudnn import behavior_note, data_type, numerical_note, tensor_reordering
from cudnn.datatypes import _dlpack_code_bits, _library_type
from cudnn.engines.base import BaseEngine, CompiledPlan, require_stream
from cudnn.frost import device as devices
from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error, cutedsl_state
from cudnn.frost.workspace import Workspace
from cudnn.graph_types import NodeType, Tensor

from . import chunks_for, schedule_for


@dataclass(frozen=True)
class _Matmul:
    a: Tensor
    b: Tensor
    out: Tensor
    bias: Tensor | None
    m: int
    n: int
    k: int
    lda: int
    b_major: str


def _match(graph) -> _Matmul:
    nodes = graph.nodes
    if len(nodes) not in (1, 2) or nodes[0].node_type != NodeType.MATMUL:
        raise NotImplementedError("rl_gemm requires matmul with an optional column bias")
    if any(graph._cpp_graph_kwargs.get(key) for key in ("is_dynamic_shape_enabled", "is_override_shape_enabled")):
        raise NotImplementedError("rl_gemm requires fixed matrix shapes")
    mm = nodes[0]
    if mm.params["padding"] != 0 or mm.compute_data_type is None or _library_type(mm.compute_data_type) != data_type.FLOAT:
        raise NotImplementedError("rl_gemm requires FP32 matmul computation and zero padding")
    a, b, out = mm.inputs["A"], mm.inputs["B"], mm.outputs["C"]
    shape_a, shape_b = tuple(a.dim), tuple(b.dim)
    if len(shape_a) not in (2, 3) or len(shape_b) != len(shape_a) or shape_a[:-2] not in ((), (1,)) or shape_b[:-2] != shape_a[:-2]:
        raise NotImplementedError("rl_gemm requires rank-2 matrices or singleton rank-3 batches")
    m, k = shape_a[-2:]
    kb, n = shape_b[-2:]
    if m <= 0 or min(k, n) <= 0 or k != kb or k % 8 or n % 8:
        raise NotImplementedError("rl_gemm requires M > 0 and positive matching K and N multiples of 8")
    if max(m, n, k) >= 2**31 or schedule_for(m, n, chunks_for(n, k)).grid_x(m, n) >= 2**31:
        raise NotImplementedError("rl_gemm exceeds the kernel's signed 32-bit shape or grid limits")
    matrix_shape = shape_a[:-2] + (m, n)
    if tuple(out.dim) != matrix_shape:
        raise NotImplementedError("rl_gemm requires a matmul output of shape (..., M, N)")

    bias = None
    if len(nodes) == 2:
        add = nodes[1]
        if add.node_type != NodeType.POINTWISE or add.params.get("mode") not in ("add", "bias"):
            raise NotImplementedError("rl_gemm supports a column bias after matmul")
        if (
            add.compute_data_type is None
            or _library_type(add.compute_data_type) != data_type.FLOAT
            or out.get_data_type() != data_type.FLOAT
            or not out.is_virtual
            or out.ragged_offset is not None
            or out.is_pass_by_value
            or out.reordering_type not in (None, tensor_reordering.NONE)
        ):
            raise NotImplementedError("rl_gemm requires a virtual FP32 accumulator and FP32 bias addition")
        first, second = add.inputs["IN_0"], add.inputs["IN_1"]
        if out is first:
            bias = second
        elif out is second:
            bias = first
        else:
            raise NotImplementedError("rl_gemm requires the matmul accumulator as an input to the bias addition")
        if tuple(bias.dim) not in ((n,), (1, n), (1, 1, n)) or bias.stride[-1] != 1:
            raise NotImplementedError("rl_gemm requires a contiguous bias vector broadcast over rows")
        out = add.outputs["OUT_0"]

    for tensor in (a, b, out) + ((bias,) if bias is not None else ()):
        if tensor.is_virtual or tensor.get_data_type() != data_type.BFLOAT16:
            raise NotImplementedError("rl_gemm requires BF16 input, output and bias buffers")
        if tensor.is_pass_by_value or tensor.ragged_offset is not None or tensor.reordering_type not in (None, tensor_reordering.NONE):
            raise NotImplementedError("rl_gemm requires dense device tensors without storage reordering")
    if tuple(out.dim) != matrix_shape or out.stride[-1] != 1 or (m > 1 and out.stride[-2] != n):
        raise NotImplementedError("rl_gemm requires a row-major (..., M, N) output")
    lda = a.stride[-2] if m > 1 else k
    if a.stride[-1] != 1 or lda < k or lda % 8:
        raise NotImplementedError("rl_gemm requires row-major A with an aligned leading dimension")
    if tuple(b.stride[-2:]) == (n, 1):
        b_major = "n"
    elif tuple(b.stride[-2:]) == (1, k):
        b_major = "k"
    else:
        raise NotImplementedError("rl_gemm requires compact row-major or column-major B")
    return _Matmul(a, b, out, bias, m, n, k, lda, b_major)


def _geometry(shape, strides):
    # The graph commonly describes a 2-D input with a leading singleton
    # batch. Neither the dimension nor its arbitrary stride addresses a byte.
    return tuple((size, step) for size, step in zip(shape, strides) if size != 1)


class _RlGemmPlan(CompiledPlan):
    takes_variant_pack = True

    def __init__(self, op, kernel, device):
        self.kernel = kernel
        self.device = device

        def binding(tensor):
            layout = _geometry(tensor.dim, tensor.stride)
            nbytes = 2 * (1 + sum((size - 1) * step for size, step in layout))
            return tensor.uid, nbytes, layout

        self.inputs = [binding(t) for t in (op.a, op.b, op.bias) if t is not None]
        self.output = binding(op.out)
        self.dtype = _dlpack_code_bits(data_type.BFLOAT16)

    def get_workspace_size(self):
        return self.kernel.workspace_bytes

    def _buffer(self, pack, uid, nbytes, layout):
        index = pack.index_of(uid)
        ptr = pack.native.pointer(index)
        if not ptr or ptr % 16:
            raise ValueError(f"rl_gemm: tensor UID {uid} requires a non-null 16-byte aligned pointer")
        observed = pack.observed_bytes(index)
        if observed >= 0 and observed < nbytes:
            raise ValueError(f"rl_gemm: tensor UID {uid} requires {nbytes} bytes, got {observed}")
        kind, ordinal = pack.observed_device(index)
        if kind >= 0 and (kind != 2 or ordinal != self.device):
            raise ValueError(f"rl_gemm: tensor UID {uid} must be on CUDA device {self.device}")
        if index not in pack.graph_described:
            if pack.native.dtype(index) != self.dtype or _geometry(pack.native.shape(index), pack.native.stride(index)) != layout:
                raise ValueError(f"rl_gemm: tensor UID {uid} must match its declared BF16 shape and strides")
        return ptr, ptr + nbytes

    def execute(self, graph, pack, ctx):
        if pack.overridden or ctx.context is not None:
            raise ValueError("rl_gemm requires its declared tensor geometry and takes no context")
        if pack.device != self.device:
            raise ValueError(f"rl_gemm: plan was built for device {self.device}, attempted execute on device {pack.device}")
        reads = [self._buffer(pack, *desc) for desc in self.inputs]
        out = self._buffer(pack, *self.output)
        writes = [out]
        if self.kernel.workspace_bytes:
            Workspace.over(pack, self.kernel.workspace_bytes, "rl_gemm", align=16)
            workspace_device = getattr(ctx.workspace, "__dlpack_device__", None)
            if workspace_device is not None and tuple(workspace_device()) != (2, self.device):
                raise ValueError(f"rl_gemm: workspace must be on CUDA device {self.device}")
            writes.append((pack.workspace, pack.workspace + self.kernel.workspace_bytes))
        for i, (begin, end) in enumerate(writes):
            if any(begin < hi and lo < end for lo, hi in reads + writes[:i]):
                raise ValueError("rl_gemm: output and workspace must be disjoint from each other and all inputs")
        self.kernel.execute(reads[0][0], reads[1][0], out[0], reads[2][0] if len(reads) == 3 else None, pack.workspace, require_stream(ctx.stream, "rl_gemm"))


class RlGemmEngine(BaseEngine):
    name = "rl_gemm"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)
    numerical_notes = (numerical_note.RL, numerical_note.TENSOR_CORE)

    def check_support(self, graph):
        self._operation(graph)

    def _operation(self, graph):
        op = graph._facts_for(_match)
        cc = devices.compute_capability(devices.current_device())
        if cc not in ((10, 0), (10, 3), (10, 7)):
            raise NotImplementedError(f"rl_gemm requires SM100, SM103 or SM107, got {cc}")
        if not cutedsl_state()[0]:
            raise NotImplementedError("rl_gemm requires nvidia-cutlass-dsl >= 4.8.0")
        reason = cutedsl_requirement_error("rl_gemm", (4, 8, 0)) or cutedsl_arch_requirement_error(cc)
        if reason:
            raise NotImplementedError(reason)
        return op

    def build_plan(self, graph, plan, ctx=None):
        from .kernel import build

        if plan.knobs:
            raise NotImplementedError("rl_gemm accepts no tuning knobs")
        op = self._operation(graph)
        device = devices.current_device()
        kernel = build(op.m, op.n, op.k, op.lda, op.b_major, op.bias is not None, device)
        return _RlGemmPlan(op, kernel, device)
