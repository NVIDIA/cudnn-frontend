# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional gather fusion with a caller-workspace materialization alternative.

A route is prepared from an independent graph snapshot. Declining a fused
loader never consumes or rewrites the graph that the next route analyzes.
"""

from dataclasses import dataclass, replace
from math import prod

from cudnn.graph_types import Tensor
from cudnn.frost import buffers
from cudnn.frost.workspace import Workspace, WorkspaceLayout

from .dtypes import DTYPE_BYTES
from .fusion_ir import IndexMap, Dtype, FusionChain
from .graph_analyzer import GemmBinding, _analyze_state, _state_from_graph
from .layouts import checked_layout

_BUFFER_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32", "int32": "int32", "int64": "int64"}


@dataclass(frozen=True)
class GatherStage:
    source: Tensor
    indices: Tensor
    output: Tensor
    access: IndexMap
    source_stride: tuple[int, ...]
    indices_stride: tuple[int, ...]
    output_stride: tuple[int, ...]
    dtype: Dtype
    indices_dtype: Dtype
    observable: bool


@dataclass(frozen=True)
class ExecutionRoute:
    chain: FusionChain | None
    binding: GemmBinding
    gathers: tuple[GatherStage, ...] = ()


def execution_routes(graph):
    """Yield fused first, then an independently analyzed materialized route.

    Invalid public gather contracts are errors, not optimization misses. Each
    route's storage/backend restrictions may decline without invalidating the
    other. No device storage is allocated during analysis or compilation.
    """
    from cudnn._gemm_validate import validate_node
    from .compiler import _graph_shapes_fixed

    for node in graph.nodes:
        if node.node_type.name == "GATHER":
            validate_node(node)
    state = _state_from_graph(graph)
    from .compiler import _check_block_scale_sign

    _check_block_scale_sign(state)
    gathers = [op for op in state["ops"] if op.cudnn_name == "gather"]
    if not gathers:
        chain, binding = _analyze_state(state)
        yield ExecutionRoute(chain, binding)
        return
    try:
        chain, binding = _analyze_state(state)
    except (NotImplementedError, ValueError):
        # Storage projections can decline a mathematically valid gather. The
        # unmodified snapshot below remains available to the materializer.
        pass
    else:
        yield ExecutionRoute(chain, binding)
    if not _graph_shapes_fixed(graph):
        raise NotImplementedError("materialized gather currently requires fixed graph shapes; dynamic/override graphs need a workspace shape contract")
    meta = {tid: replace(m) for tid, m in state["tensor_meta"].items()}
    stages = []
    for op in gathers:
        source, indices, out = (meta[tid] for tid in (*op.inputs, op.output))
        if not source.is_input or not indices.is_input:
            raise NotImplementedError("materialized gather inputs must be graph inputs or earlier materialized gathers")
        if source.dtype not in _BUFFER_DTYPES or out.dtype != source.dtype or indices.dtype not in ("int32", "int64"):
            raise NotImplementedError("materialized gather supports FP16/BF16/FP32/INT32/INT64 sources and INT32/INT64 indices without casting")
        access = IndexMap.take(indices.name, source.dim, indices.dim, dict(op.op_attrs)["axis"])
        for item in (source, indices, out):
            if len(item.dim) != len(item.stride) or any(s < 0 for s in item.stride):
                raise NotImplementedError("materialized gather requires nonnegative strides matching the tensor rank")
            span = 1 + sum((d - 1) * s for d, s in zip(item.dim, item.stride))
            if span * DTYPE_BYTES[item.dtype] > (1 << 63) - 1:
                raise NotImplementedError("materialized gather storage exceeds signed Int64 geometry")
        if out.dim != access.consumer_shape:
            raise ValueError("gather output shape does not match its access domain")
        checked_layout(out.dim, out.stride, context="gather output", writable=True)
        if (prod(out.dim) + 255) // 256 > (1 << 31) - 1:
            raise NotImplementedError("materialized gather exceeds CUDA's one-dimensional grid capacity")
        observable = out.output_index is not None
        stages.append(
            GatherStage(source.tensor, indices.tensor, out.tensor, access, source.stride, indices.stride, out.stride, source.dtype, indices.dtype, observable)
        )
        meta[op.output] = replace(out, is_input=True, output_index=None)
    remaining = [op for op in state["ops"] if op.cudnn_name != "gather"]
    if remaining:
        chain, binding = _analyze_state({**state, "ops": remaining, "tensor_meta": meta})
        if chain.has_moe:
            raise NotImplementedError("materialized gather currently composes with dense GEMM plans, not grouped routing")
        if any(meta[id(t)].dtype not in _BUFFER_DTYPES for t in binding.bound_tensors()):
            raise NotImplementedError("materialized gather GEMM binding currently supports FP16/BF16/FP32/INT32/INT64 buffers only")
    else:
        if not any(stage.observable for stage in stages):
            raise NotImplementedError("materialized gather graph has no observable output")
        chain, binding = None, GemmBinding()
    yield ExecutionRoute(chain, binding, tuple(stages))


def available_routes(graph):
    routes = []
    try:
        for route in execution_routes(graph):
            routes.append(route)
    except (NotImplementedError, ValueError):
        if not routes:
            raise
    return tuple(routes)


def _span(shape, stride):
    return 1 + sum((d - 1) * s for d, s in zip(shape, stride))


def render_gather(stage):
    """Lower the complete IndexMap through the shared integer renderer."""
    from .dtypes import DTYPE_TO_CUTLASS
    from .epilogue_codegen import _render_index_expr
    from .fusion_ir import IndexExpr

    shape = stage.access.consumer_shape
    coordinates = {i: f"((position // {prod(shape[i + 1:])}) % {d})" for i, d in enumerate(shape)}
    loads = []
    predicates = []

    def load(ref, coords, extent):
        if ref != stage.indices.get_name():
            raise ValueError("gather stage references an unbound index tensor")
        offset = " + ".join(f"({c}) * {s}" for c, s in zip(coords, stage.indices_stride))
        name = f"index_{len(loads)}"
        loads.append(f"        {name} = cutlass.Int64(indices.iterator[{offset}])")
        predicates.append(f"({name} >= 0) & ({name} < {extent})")
        return name

    source_coords = stage.access.lower(tuple(IndexExpr.coordinate(i) for i in range(len(shape))), lambda expr: _render_index_expr(expr, coordinates), load)
    source_offset = " + ".join(f"({c}) * {s}" for c, s in zip(source_coords, stage.source_stride))
    output_offset = " + ".join(f"cutlass.Int64({coordinates[i]}) * {s}" for i, s in enumerate(stage.output_stride))
    dtype, indices_dtype = DTYPE_TO_CUTLASS[stage.dtype], DTYPE_TO_CUTLASS[stage.indices_dtype]
    return f"""# SPDX-License-Identifier: Apache-2.0
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_tensor, make_fake_stream

@cute.kernel
def gather(source, indices, output):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    position = cutlass.Int64(block) * 256 + thread
    if position < {prod(shape)}:
{chr(10).join(loads)}
        if {" & ".join(predicates)}:
            output.iterator[{output_offset}] = source.iterator[{source_offset}]
        else:
            output.iterator[{output_offset}] = {dtype}(0)

gather.set_name_prefix("cudnn", remove_cutlass_symbol=True)

@cute.jit
def launch(source, indices, output, stream):
    gather(source, indices, output).launch(grid=({(prod(shape) + 255) // 256}, 1, 1), block=(256, 1, 1), stream=stream)

def compile():
    return cute.compile(launch,
        make_fake_tensor({dtype}, ({_span(stage.access.producer_shape, stage.source_stride)},), (1,), assumed_align={DTYPE_BYTES[stage.dtype]}),
        make_fake_tensor({indices_dtype}, ({_span(stage.indices.get_dim(), stage.indices_stride)},), (1,), assumed_align={DTYPE_BYTES[stage.indices_dtype]}),
        make_fake_tensor({dtype}, ({_span(shape, stage.output_stride)},), (1,), assumed_align={DTYPE_BYTES[stage.dtype]}),
        make_fake_stream(use_tvm_ffi_env_stream=False), options="--enable-tvm-ffi")
"""


class MaterializedGatherPlan:
    """Precompiled gather launches followed by the original GEMM plan.

    Device intermediates and nested GEMM scratch are caller-workspace regions,
    not plan-owned allocations. Every buffer is validated before the first write.
    """

    def __init__(self, route, gemm=None):
        from .compiler import _import_kernel

        from cudnn.frost.device import current_device

        if gemm is not None and hasattr(gemm, "lowered") and gemm.lowered is None:
            raise NotImplementedError(f"materialized GEMM has no executable binding: {gemm.declined}")
        self.device = current_device()
        self.route, self.gemm = route, gemm
        self.chain = route.chain
        stages = route.gathers
        scratch = {id(stage.output) for stage in stages if not stage.observable}
        bound = list(route.binding.bound_tensors())
        for stage in stages:
            bound.extend((stage.source, stage.indices))
            if stage.observable:
                bound.append(stage.output)
        bound = list({id(t): t for t in bound if id(t) not in scratch}.values())
        self.binding = GemmBinding(a_operands=bound)
        self._kernels = tuple(_import_kernel(render_gather(stage)).compile() for stage in stages)
        self._layout = WorkspaceLayout()
        self._regions = tuple(
            (stage, self._layout.add(_span(stage.access.consumer_shape, stage.output_stride) * DTYPE_BYTES[stage.dtype]))
            for stage in stages
            if not stage.observable
        )
        self._gemm_offset = self._layout.add(int(getattr(gemm, "workspace_bytes", 0) or 0))
        self.workspace_bytes = self._layout.size
        self._tensors = tuple(bound)
        from .graph_analyzer import _map_dtype

        self._metadata = {id(t): (tuple(t.get_dim()), tuple(t.get_stride()), _BUFFER_DTYPES.get(_map_dtype(t.get_data_type()))) for t in bound}
        for stage in stages:
            for tensor, shape, stride, dtype in (
                (stage.source, stage.access.producer_shape, stage.source_stride, stage.dtype),
                (stage.indices, tuple(stage.indices.get_dim()), stage.indices_stride, stage.indices_dtype),
                (stage.output, stage.access.consumer_shape, stage.output_stride, stage.dtype),
            ):
                self._metadata[id(tensor)] = (shape, stride, _BUFFER_DTYPES[dtype])

    def __call__(self, variant_pack, workspace=None, stream=None):
        from cuda.bindings import driver

        if workspace is not None and not isinstance(workspace, Workspace):
            workspace = Workspace(workspace, self.workspace_bytes, "materialized_gather", device=self.device)
        if self.workspace_bytes and workspace is None:
            raise ValueError(f"materialized gather requires a {self.workspace_bytes}-byte caller workspace")
        values = dict(variant_pack)
        if any(t not in values for t in self._tensors):
            raise ValueError("materialized gather variant pack is missing a prepared operand")
        for stage, offset in self._regions:
            shape = stage.access.consumer_shape
            values[stage.output] = workspace.view(offset, _BUFFER_DTYPES[stage.dtype], (_span(shape, stage.output_stride),)).metadata_view(
                shape, stage.output_stride
            )
        # Fixed-shape storage specialization must not silently accept a changed
        # stride or dtype at execute. The native pack may supply graph geometry.
        for tensor, value in values.items():
            expected = self._metadata.get(id(tensor))
            if expected is not None:
                shape, stride, dtype = expected
                if tuple(value.shape) != shape or tuple(value.stride()) != stride or buffers.dtype_name(value) != dtype:
                    raise ValueError(f"materialized gather {tensor.get_name()!r} binding differs from its prepared shape/stride/dtype")
                device = value.__dlpack_device__()
                if tuple(device) != (2, self.device):
                    raise ValueError("materialized gather bindings must be CUDA buffers on the compiled device")
        # Reject overlap conservatively. All inputs stay live through the gather
        # sequence and GEMM, so graph-node order cannot make aliasing safe.
        output_tensors = [stage.output for stage in self.route.gathers]
        if self.gemm is not None:
            output_tensors.extend(self.gemm.binding.outputs)
        output_tensors = list({id(t): t for t in output_tensors}.values())
        if self.workspace_bytes:
            scratch_buffer = workspace.view(0, "uint8", (self.workspace_bytes,))
            if tuple(scratch_buffer.__dlpack_device__()) != (2, self.device):
                raise ValueError("materialized gather workspace must be on the compiled device")
            base = buffers.data_ptr(scratch_buffer)
            for tensor in self._tensors:
                value = values[tensor]
                ptr = buffers.data_ptr(value)
                if base < ptr + _span(value.shape, value.stride()) * value.element_size() and ptr < base + self.workspace_bytes:
                    raise ValueError("materialized gather workspace must not overlap caller operands")
        for output_tensor in output_tensors:
            output = values[output_tensor]
            lo = buffers.data_ptr(output)
            hi = lo + _span(output.shape, output.stride()) * output.element_size()
            for tensor, value in values.items():
                if tensor is output_tensor:
                    continue
                other_lo = buffers.data_ptr(value)
                other_hi = other_lo + _span(value.shape, value.stride()) * value.element_size()
                if lo < other_hi and other_lo < hi:
                    raise ValueError("materialized gather outputs/workspace must not overlap other operands")
        if self.gemm is not None:
            validate = getattr(self.gemm, "validate_operands", None)
            if validate is not None:
                validate([values[tensor] for tensor in self.gemm.bound])
        prepared = []
        for stage in self.route.gathers:
            carriers = []
            for tensor in (stage.source, stage.indices, stage.output):
                value = values[tensor]
                shape, stride, dtype = self._metadata[id(tensor)]
                device = value.__dlpack_device__()[1]
                carriers.append(buffers.DeviceView(buffers.data_ptr(value), (_span(shape, stride),), dtype, device))
            prepared.append(carriers)
        nested = None
        if self.gemm is not None and getattr(self.gemm, "workspace_bytes", 0):
            nested = workspace.view(self._gemm_offset, "uint8", (self.gemm.workspace_bytes,))
        cuda_stream = stream if isinstance(stream, driver.CUstream) else driver.CUstream(0 if stream is None else int(stream))
        for kernel, carriers in zip(self._kernels, prepared):
            kernel(*carriers, cuda_stream)
        if self.gemm is not None:
            self.gemm(values, workspace=nested, stream=stream)


def probe_execution_plan(graph, config=None, *, backend=None):
    """Cheap admission across loader alternatives, without JIT or allocations."""
    if backend is None:
        from . import compiler as backend
    from .compiler import _graph_dynamic_shapes, _graph_shapes_fixed

    backend.probe_cutedsl()
    reasons = []
    for route in available_routes(graph):
        if route.chain is None:
            if config is not None:
                raise NotImplementedError("standalone gather does not accept a GEMM tile config")
            return
        try:
            selected = config or backend.plan_config(route.chain, dynamic_shapes=_graph_dynamic_shapes(graph), shapes_fixed=_graph_shapes_fixed(graph))
            backend.probe_chain(route.chain, selected)
        except (NotImplementedError, ValueError) as exc:
            reasons.append(str(exc))
        else:
            return
    raise NotImplementedError("no fused or materialized gather plan: " + "; ".join(reasons))


def build_execution_plan(graph, config=None, *, knobs=None, backend=None):
    """The graph API and direct compiled-plan API share the same route selection."""
    if backend is None:
        from . import compiler as backend
    from .compiler import _graph_dynamic_shapes, _graph_shapes_fixed

    backend.probe_cutedsl()
    reasons = []
    for route in available_routes(graph):
        try:
            if route.chain is None:
                if knobs is not None or config is not None:
                    raise NotImplementedError("standalone gather has no GEMM tile knobs or config")
                return MaterializedGatherPlan(route)
            selected = config or backend.plan_config(
                route.chain, knobs=knobs, dynamic_shapes=_graph_dynamic_shapes(graph), shapes_fixed=_graph_shapes_fixed(graph)
            )
            backend.probe_chain(route.chain, selected)
            compiled = backend.jit_from_chain(route.chain, route.binding, selected)
            return MaterializedGatherPlan(route, compiled) if route.gathers else compiled
        except (NotImplementedError, ValueError) as exc:
            reasons.append(str(exc))
    raise NotImplementedError("no fused or materialized gather plan: " + "; ".join(reasons))
