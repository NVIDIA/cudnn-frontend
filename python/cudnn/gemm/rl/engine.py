# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-reduction matmul engines for RL forward graphs."""

from dataclasses import dataclass

from cudnn import behavior_note, data_type, numerical_note, tensor_reordering
from cudnn.datatypes import _dlpack_code_bits, _library_type
from cudnn.engines.base import BaseEngine, CompiledPlan, require_stream
from cudnn.frost import device as devices
from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error, cutedsl_state
from cudnn.frost.workspace import Workspace, check_workspace_device
from cudnn.graph_types import NodeType, Tensor, storage_geometry, storage_slot_bytes

from . import chunks_for, schedule_for
from ._graph import ScaledMatmul, is_scaled, match_scaled


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


def _match(graph) -> _Matmul | ScaledMatmul:
    nodes = graph.nodes
    if any(graph._cpp_graph_kwargs.get(key) for key in ("is_dynamic_shape_enabled", "is_override_shape_enabled")):
        raise NotImplementedError("rl_gemm requires fixed matrix shapes")
    if is_scaled(nodes):
        return match_scaled(nodes)
    if len(nodes) not in (1, 2) or nodes[0].node_type != NodeType.MATMUL:
        raise NotImplementedError("rl_gemm requires a supported GEMM")
    mm = nodes[0]
    if mm.params.get("enable_pdl"):
        raise NotImplementedError("rl_gemm supports PDL for mixed MXFP8/MXFP4 GEMM")
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
            dtype = tensor.get_data_type()
            shape, strides = storage_geometry(tensor.dim, tensor.stride, dtype)
            layout = _geometry(shape, strides)
            nbytes = storage_slot_bytes(dtype) * (1 + sum((size - 1) * step for size, step in layout))
            reordered = tensor.get_reordering_type() == tensor_reordering.F8_128x4
            return tensor.uid, nbytes, layout, _dlpack_code_bits(dtype), 4 if dtype == data_type.FLOAT else 16, reordered

        self.scaled = isinstance(op, ScaledMatmul)
        extra = (op.sa, op.sb, op.alpha) if self.scaled else (op.bias,)
        self.inputs = [binding(t) for t in (op.a, op.b, *extra) if t is not None]
        self.output = binding(op.out)

    def get_workspace_size(self):
        return self.kernel.workspace_bytes

    def _buffer(self, pack, uid, nbytes, layout, dtype, alignment, reordered):
        index = pack.index_of(uid)
        ptr = pack.native.pointer(index)
        if not ptr or ptr % alignment:
            raise ValueError(f"rl_gemm: tensor UID {uid} requires a non-null {alignment}-byte aligned pointer")
        observed = pack.observed_bytes(index)
        if observed >= 0 and observed < nbytes:
            raise ValueError(f"rl_gemm: tensor UID {uid} requires {nbytes} bytes, got {observed}")
        kind, ordinal = pack.observed_device(index)
        if kind >= 0 and (kind != 2 or ordinal != self.device):
            raise ValueError(f"rl_gemm: tensor UID {uid} must be on CUDA device {self.device}")
        if index not in pack.graph_described:
            actual = _geometry(pack.native.shape(index), pack.native.stride(index))
            if reordered:
                # F8_128x4 is a physical byte layout. Producers can present its
                # contiguous storage as a flat blob or a transpose of a matrix.
                extent = 1
                for size, stride in sorted(actual, key=lambda axis: axis[1]):
                    if stride != extent:
                        raise ValueError(f"rl_gemm: reordered scale UID {uid} requires a dense buffer")
                    extent *= size
                if extent < nbytes:
                    raise ValueError(f"rl_gemm: reordered scale UID {uid} requires {nbytes} bytes, got {extent}")
            elif actual != layout:
                raise ValueError(f"rl_gemm: tensor UID {uid} must match its declared shape and strides")
            if pack.native.dtype(index) != dtype:
                raise ValueError(f"rl_gemm: tensor UID {uid} must match its declared dtype")
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
            check_workspace_device(ctx.workspace, pack.workspace, self.device, "rl_gemm")
            writes.append((pack.workspace, pack.workspace + self.kernel.workspace_bytes))
        for i, (begin, end) in enumerate(writes):
            if any(begin < hi and lo < end for lo, hi in reads + writes[:i]):
                raise ValueError("rl_gemm: output and workspace must be disjoint from each other and all inputs")
        stream = require_stream(ctx.stream, "rl_gemm")
        if self.scaled:
            self.kernel.execute(reads[0][0], reads[1][0], out[0], reads[2][0], reads[3][0], reads[4][0] if len(reads) == 5 else None, pack.workspace, stream)
        else:
            self.kernel.execute(reads[0][0], reads[1][0], out[0], reads[2][0] if len(reads) == 3 else None, pack.workspace, stream)


class RlGemmEngine(BaseEngine):
    name = "rl_gemm"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)
    numerical_notes = (numerical_note.RL, numerical_note.TENSOR_CORE)

    def check_support(self, graph):
        self._operation(graph)

    def _operation(self, graph):
        op = graph._facts_for(_match)
        device = devices.current_device()
        cc = devices.compute_capability(device)
        if cc not in ((10, 0), (10, 3), (10, 7)):
            raise NotImplementedError(f"rl_gemm requires SM100, SM103 or SM107, got {cc}")
        if isinstance(op, ScaledMatmul):
            from ._quantized.config import schedule_for as scaled_schedule

            if op.kind == "mxfp8_mxfp4" and cc != (10, 7) and op.k % 128:
                raise NotImplementedError("rl_gemm mixed MXFP8/MXFP4 on SM100/SM103 requires K aligned to 128")
            scaled_schedule(op, cc, devices.multiprocessor_count(device))
        if not cutedsl_state()[0]:
            raise NotImplementedError("rl_gemm requires nvidia-cutlass-dsl >= 4.8.0")
        reason = cutedsl_requirement_error("rl_gemm", (4, 8, 0)) or cutedsl_arch_requirement_error(cc)
        if reason:
            raise NotImplementedError(reason)
        return op

    def build_plan(self, graph, plan, ctx=None):
        if plan.knobs:
            raise NotImplementedError("rl_gemm accepts no tuning knobs")
        op = self._operation(graph)
        device = devices.current_device()
        if isinstance(op, ScaledMatmul):
            from .quantized import build

            kernel = build(op, device)
        else:
            from .kernel import build

            kernel = build(op.m, op.n, op.k, op.lda, op.b_major, op.bias is not None, device)
        return _RlGemmPlan(op, kernel, device)
