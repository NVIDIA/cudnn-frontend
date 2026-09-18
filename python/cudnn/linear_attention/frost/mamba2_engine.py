# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Mamba-2 SSD plans using the linear-attention graph lifecycle."""

from functools import lru_cache
import json
import math

import cudnn
from cudnn.engines.base import BaseEngine
from cudnn.frost import buffers
from cudnn.frost.device import build_device, current_device
from cudnn.frost.workspace import WorkspaceLayout, carve_plan
from ..mamba2_graph_analyzer import analyze
from .engine import FrostLaPlan

_BF16_PORTS = frozenset(("x", "dt", "B", "C", "z", "dO", "O", "ungated_out", "dX", "dDt", "dB", "dC", "dZ"))


class Mamba2FrostEngine(BaseEngine):
    name = "mamba2_frost"
    behavior_notes = (cudnn.behavior_note.RUNTIME_COMPILATION,)

    def check_support(self, graph):
        facts = graph._facts_for(analyze)
        if facts is None:
            raise NotImplementedError("Mamba2FrostEngine requires exactly one MAMBA2/MAMBA2_BWD node")
        if facts.invalid:
            raise NotImplementedError(f"Mamba2FrostEngine: {facts.invalid}")
        sm = buffers.current_sm()
        if sm != 100:
            raise NotImplementedError(f"Mamba2FrostEngine requires SM100; found SM{sm}")
        installed, version = buffers.cutedsl_state()
        if not installed:
            raise NotImplementedError("Mamba2FrostEngine requires nvidia-cutlass-dsl >= 4.7.0")
        if buffers.cutedsl_too_old(version):
            raise NotImplementedError(buffers.cutedsl_requirement_error("Mamba2FrostEngine"))
        if facts.head_dim != 64 or facts.state_dim != 64 or facts.chunk_size != 32:
            raise NotImplementedError("Mamba2FrostEngine requires head_dim=state_dim=64 and chunk_size=32")
        if facts.dt_softplus is not True:
            raise NotImplementedError("Mamba2FrostEngine currently requires dt_softplus=True")
        if facts.intermediate_dtype not in ("float32", "bfloat16"):
            raise NotImplementedError("Mamba2FrostEngine: intermediate_dtype must be float32 or bfloat16")
        for name, tensor in facts.tensors.items():
            want = cudnn.data_type.BFLOAT16 if name in _BF16_PORTS else cudnn.data_type.FLOAT
            if name == "state_checkpoints" and facts.intermediate_dtype == "bfloat16":
                want = cudnn.data_type.BFLOAT16
            if tensor.get_data_type() != want:
                raise NotImplementedError(f"Mamba2FrostEngine: {name} must be {want}, got {tensor.get_data_type()}")
            if not buffers.is_contiguous(tensor.dim, tensor.stride):
                raise NotImplementedError(f"Mamba2FrostEngine: {name} must be contiguous, got strides {tensor.stride}")

    def build_plan(self, graph, plan, ctx=None):
        self.check_support(graph)
        handle = ctx.handle if ctx is not None else None
        device = handle.device.ordinal if hasattr(handle, "device") else None
        with build_device(device):
            return FrostLaPlan(CompiledMamba2(next(iter(graph.nodes)), graph._facts_for(analyze)))


@lru_cache(maxsize=128)
def _compile_step(component, geometry, options, descriptors, device):
    """Compile from plan metadata only, before any execute-time buffer exists."""
    import cutlass
    from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
    from cudnn.frost.compiled_cache import compile_cached
    from .kernel.mamba2_prefill_f16 import Mamba2Prefill
    from .kernel.mamba2_state_scan_f16 import Mamba2StateScan
    from .kernel.mamba2_bprop_f16 import Mamba2BackwardChunks
    from .kernel.mamba2_reduce import Mamba2BackwardReduce
    from .kernel.mamba2_gate_fwd import Mamba2GateForward
    from .kernel.mamba2_gate_bwd import Mamba2GateBackward

    b, length, h, g = geometry
    constructors = {
        "forward": lambda: Mamba2Prefill(b, length, h, g, parallel_scores=True),
        "gate_forward": lambda: Mamba2GateForward(b * length * h * 64),
        "gate_backward": lambda: Mamba2GateBackward(b, length, h),
        "scan": lambda: Mamba2StateScan(b, length, h, g, split_directions=options[0], reverse_only=options[1], early_tmem_release=True),
        "chunks": lambda: Mamba2BackwardChunks(b, length, h, g, 2, early_tmem_release=True, reuse_j_storage=True),
        "reduce": lambda: Mamba2BackwardReduce(b, length, h, g),
    }
    types = {"bfloat16": cutlass.BFloat16, "float32": cutlass.Float32}
    args = [None if d is None else make_fake_compact_tensor(types[d[0]], (d[1],), assumed_align=16) for d in descriptors]
    key = json.dumps(("mamba2_v1", component, geometry, options, descriptors, device))
    return compile_cached(constructors[component](), *args, make_fake_stream(), cache_key=key, options="--enable-tvm-ffi")


class CompiledMamba2:
    """Fixed-shape native pipeline over caller inputs, outputs and workspace."""

    def __init__(self, node, facts):
        self.plan_name = "Mamba2FrostEngine"
        self.device = current_device()
        self.facts = facts
        self.steps = []
        self.metadata = {}
        for name, tensor in facts.tensors.items():
            dtype = "bfloat16" if tensor.get_data_type() == cudnn.data_type.BFLOAT16 else "float32"
            self.metadata[name] = (dtype, tuple(tensor.dim))
        layout = WorkspaceLayout()
        regions = []

        def scratch(name, dtype, shape):
            self.metadata[name] = (dtype, tuple(shape))
            n = math.prod(shape)
            regions.append((name, layout.add(n * buffers.DTYPE_ITEMSIZE[dtype]), dtype, (n,)))
            return name

        def present(name):
            return name if name in facts.tensors else None

        def step(component, names, options=()):
            descriptors = tuple(None if name is None else (self.metadata[name][0], math.prod(self.metadata[name][1])) for name in names)
            compiled = _compile_step(component, (facts.batch, facts.length, facts.heads, facts.groups), tuple(options), descriptors, self.device)
            self.steps.append((compiled, tuple(names)))

        xshape = (facts.batch, facts.length, facts.heads, 64)
        state_shape = (facts.batch, facts.heads, 64, 64)
        blocks = (facts.batch, facts.heads, (facts.length + 31) // 32)
        checkpoints_shape = blocks + (64, 64)
        intermediate = facts.intermediate_dtype
        if not facts.is_bwd:
            raw = scratch("_raw", "float32", xshape) if present("z") else "O"
            final = present("final_state") or scratch("_final", "float32", state_shape)
            step(
                "forward",
                ("x", "dt", "A", "B", "C", present("D"), present("dt_bias"), None, present("initial_state"), raw, final, None, present("state_checkpoints")),
            )
            if present("z"):
                step("gate_forward", (raw, "z", "O", "ungated_out"))
        else:
            dy = "dO"
            ddp = scratch("_ddp", "float32", blocks)
            if present("z"):
                dy = scratch("_dy", "bfloat16", xshape)
                step("gate_backward", ("x", "z", "ungated_out", "dO", dy, "dZ", ddp))
            seeds = present("state_checkpoints") or scratch("_seeds", intermediate, checkpoints_shape)
            adjoints = scratch("_adjoints", intermediate, checkpoints_shape)
            di = present("d_initial_state") or scratch("_di", "float32", state_shape)
            step(
                "scan",
                ("x", dy, "dt", "A", "B", "C", present("dt_bias"), present("initial_state"), present("d_final_state"), seeds, adjoints, di),
                (bool(present("initial_state") or present("d_final_state") or facts.length < 128), bool(present("state_checkpoints"))),
            )
            dbp, dcp = scratch("_dbp", intermediate, xshape), scratch("_dcp", intermediate, xshape)
            dap, dbiasp = scratch("_dap", "float32", blocks), scratch("_dbiasp", "float32", blocks)
            step(
                "chunks",
                (
                    "x",
                    dy,
                    "dt",
                    "A",
                    "B",
                    "C",
                    present("D"),
                    present("dt_bias"),
                    seeds,
                    adjoints,
                    "dX",
                    dbp,
                    dcp,
                    "dDt",
                    dap,
                    None if present("z") else ddp,
                    dbiasp,
                ),
            )
            dd = present("dD") or scratch("_dD", "float32", (facts.heads,))
            dbias = present("d_dt_bias") or scratch("_d_dt_bias", "float32", (facts.heads,))
            step("reduce", (dbp, dcp, dap, ddp, dbiasp, "dB", "dC", "dA", dd, dbias))
        if not layout.size:
            layout.add(1)  # FrostLaPlan's caller-workspace protocol always has a base pointer.
        self.workspace_size = layout.size
        self.region_names = [r[0] for r in regions]
        self.carve = carve_plan(self.plan_name, [(offset, dtype, shape) for _, offset, dtype, shape in regions])

    def workspace_bytes(self):
        return self.workspace_size

    def bind(self, names):
        self.names = tuple(names)

    def run(self, views, workspace, stream):
        import cuda.bindings.driver as cuda

        values = {}
        for name, view in zip(self.names, views):
            ptr, shape, strides, dtype, device = buffers.probe(view)
            expected_dtype, expected_shape = self.metadata[name]
            if tuple(shape) != expected_shape or dtype != expected_dtype or device != self.device or not buffers.is_contiguous(shape, strides):
                raise ValueError(f"{self.plan_name}: {name} must match its declared contiguous shape, dtype and device")
            if ptr % 16:
                raise ValueError(f"{self.plan_name}: {name} must have a 16-byte aligned address")
            values[name] = buffers.DeviceView(ptr, (math.prod(shape),), dtype, device)
        values.update(zip(self.region_names, workspace.carve(self.carve)))
        cu_stream = cuda.CUstream(int(stream))
        for compiled, names in self.steps:
            compiled(*(values[name] if name is not None else None for name in names), cu_stream)
