# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental SM120 SwiGLU graph specialization with native weight and offset bindings.

Credit: NVIDIA Frost/CuTeDSL/CUTLASS and the explicit parent-binding work in
Yanqin/Yihua's PR1090 integration. This specialization is our research and implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

from cudnn.frost import buffers
from .graph_analyzer import analyze_with_binding

ENGINE = "frost_moe_swiglu_simt_sm120"


@dataclass(frozen=True)
class WeightBinding:
    tensor: object
    source: object
    byte_offset: int


@dataclass(frozen=True)
class SimtSpec:
    binding: object
    tokens: object
    gate: WeightBinding
    up: WeightBinding
    output: object
    offsets: object
    rows: int


def analyze(graph, *, dynamic_shapes=False):
    """Validate the entire graph and its declared parent relationship, on CPU."""
    if dynamic_shapes:
        raise NotImplementedError("SM120 SwiGLU currently requires fixed graph dimensions")
    chain, binding = analyze_with_binding(graph)
    if not chain.has_moe or chain.has_block_scale or chain.num_gemms != 2:
        raise NotImplementedError("SM120 SwiGLU requires two ordinary grouped matmuls")
    if chain.num_a_operands != 1 or chain.num_b_operands != 2 or set(chain.gemm_operands) != {(0, 0), (0, 1)}:
        raise NotImplementedError("SM120 SwiGLU requires one shared token input and two distinct weights")
    if chain.aux_tensors or chain.reductions or chain.quants or chain.mainloop_a_ops or chain.mainloop_b_ops:
        raise NotImplementedError("SM120 SwiGLU supports only the SwiGLU epilogue")
    mm, moe = chain.matmul, chain.moe
    if (mm.a_dtype, mm.b_dtype, mm.accum_dtype, mm.out_dtype, mm.a_major, mm.b_major) != ("bf16", "bf16", "fp32", "fp32", "k", "k"):
        raise NotImplementedError("SM120 SwiGLU requires BF16 operands and FP32 projections")
    if moe.mode != "none" or moe.offset_dtype != "int32" or moe.weight_layout is not None or moe.num_groups != moe.num_experts:
        raise NotImplementedError("SM120 SwiGLU requires one int32 offset group per canonical expert")
    if not 1 <= mm.M <= 4096 or (mm.N, mm.K, moe.num_experts) != (768, 2048, 128):
        raise NotImplementedError("SM120 SwiGLU requires 1<=R<=4096, N768, K2048 and E128")
    if len(chain.ops) != 2 or len(chain.output_specs) != 1 or len(binding.outputs) != 1:
        raise NotImplementedError("SM120 SwiGLU requires exactly swish then multiply and one output")
    swish, mul = chain.ops
    if swish.op != "swish" or mul.op != "mul" or swish.aux is not None or mul.aux is not None:
        raise NotImplementedError("SM120 SwiGLU requires swish(gate)*up")
    gate_ref = swish.resolved_parent_idx(0)
    if gate_ref not in (-1, -2) or {mul.resolved_parent_idx(1), mul.parent_idx_b} != {0, -3 - gate_ref}:
        raise NotImplementedError("SM120 SwiGLU requires independent gate and up projections")
    if dict(swish.attrs) not in ({}, {"swish_beta": 1.0}) or mul.attrs:
        raise NotImplementedError("SM120 SwiGLU supports the standard unit-beta SwiGLU")
    if swish.compute_dtype != "fp32" or swish.out_dtype != "fp32" or mul.compute_dtype != "fp32":
        raise NotImplementedError("SM120 SwiGLU must keep the intermediate activation in FP32")
    output = chain.output_specs[0]
    if output.source_ref != 1 or output.dtype != "bf16" or mul.out_dtype not in ("fp32", "bf16"):
        raise NotImplementedError("SM120 SwiGLU requires one BF16 final output")
    if chain.has_mainloop_fusion or chain.mainloop_a_load_dtype or chain.mainloop_b_load_dtype:
        raise NotImplementedError("SM120 SwiGLU supports no operand transform")
    views = {id(view.tensor): view for view in binding.operand_slices}
    gate_index = -1 - gate_ref
    gate_tensor = binding.b_operands[chain.gemm_operands[gate_index][1]]
    up_tensor = binding.b_operands[chain.gemm_operands[1 - gate_index][1]]
    pitch = (3145728, 1, 2048)

    def weight(tensor):
        if tuple(tensor.get_dim()) != (128, 2048, 768) or tuple(tensor.get_stride()) not in (pitch, (1572864, 1, 2048)):
            raise NotImplementedError("SM120 SwiGLU requires its native canonical expert weight strides")
        view = views.get(id(tensor))
        if view is None:
            return WeightBinding(tensor, tensor, 0)
        if view.parent_dim != (128, 2048, 1536) or view.parent_stride != pitch or view.byte_offset not in (0, 3145728):
            raise NotImplementedError("SM120 SwiGLU requires a complete canonical parent slice")
        return WeightBinding(tensor, view.parent, view.byte_offset)

    x, y, off = binding.a_operands[0], binding.outputs[0], binding.first_token_offset
    r = mm.M
    for tensor, dim, stride in [(x, (1, r, 2048), (r * 2048, 2048, 1)), (y, (1, r, 768), (r * 768, 768, 1)), (off, (128, 1, 1), (1, 1, 1))]:
        if tuple(tensor.get_dim()) != dim or tuple(tensor.get_stride()) != stride:
            raise NotImplementedError("SM120 SwiGLU requires compact tokens/output and 128 group starts")
    return SimtSpec(binding, x, weight(gate_tensor), weight(up_tensor), y, off, r)


@dataclass(frozen=True)
class KernelParams:
    device_ordinal: int
    source_digest: str
    grid_ctas: int = 564


def _template_paths():
    root = Path(__file__).resolve().parent / "sm120/kernel_templates"
    return root / "sm120_moe_swiglu_simt.py", root / "sm120_moe_swiglu_simt_small.py"


def device_params():
    from cudnn.frost import device
    from .compiler import probe_cutedsl

    probe_cutedsl()
    ordinal = device.resolve_device(None)
    if device.compute_capability(ordinal) != (12, 0) or device.multiprocessor_count(ordinal) != 188:
        raise NotImplementedError("SM120 SwiGLU specialization currently requires the 188-SM SM120 device")
    paths = [Path(__file__), *_template_paths()]
    digest = hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()
    return KernelParams(ordinal, digest)


def validate_knobs(knobs):
    if knobs is not None and (not isinstance(knobs, dict) or knobs):
        raise ValueError("SM120 SwiGLU specialization has no tunable knobs")
    return None


class SimtCompiled:
    workspace_bytes = 0

    def __init__(self, spec, params):
        from cudnn.frost.template_loader import load_template

        self.spec, self.binding, self.params = spec, spec.binding, params
        # This fixed-domain plan owns its declaration. Runtime buffers and their
        # metadata are checked on every call; no buffer or address is retained.
        operands = []
        for tensor in self.binding.bound_tensors():
            dim, stride = tuple(tensor.get_dim()), tuple(tensor.get_stride())
            dtype = "int32" if tensor is spec.offsets else "bfloat16"
            extent = 1 + sum((int(d) - 1) * int(s) for d, s in zip(dim, stride))
            operands.append((tensor, dim, stride, dtype, extent * (4 if dtype == "int32" else 2)))
        self._operand_specs = tuple(operands)
        self._expert_strides = (int(spec.gate.tensor.get_stride()[0] * 2), int(spec.up.tensor.get_stride()[0] * 2))
        regular, small = _template_paths()
        self.generated_path = small if spec.rows <= 8 else regular
        self.module = load_template(str(self.generated_path), params, tag="moe_swiglu_simt_sm120")
        self.kernel = self.module.compile()

    def __call__(self, pack, *, stream):
        from cuda.bindings import driver as cuda

        if stream is None:
            raise ValueError("SM120 SwiGLU execution requires an explicit stream")
        spec = self.spec
        ranges = {}
        for tensor, dim, stride, dtype, nbytes in self._operand_specs:
            buf = pack[tensor]
            if tuple(buf.shape) != dim or tuple(buf.stride()) != stride or buffers.dtype_name(buf) != dtype:
                raise ValueError("SM120 SwiGLU operand must match its declared shape, stride and dtype")
            address = int(buf.data_ptr())
            if tuple(map(int, buf.__dlpack_device__())) != (2, self.params.device_ordinal) or address <= 0 or address % 16:
                raise ValueError("SM120 SwiGLU requires 16-byte aligned CUDA operands on its compiled device")
            ranges[id(tensor)] = (address, address + nbytes)
        start, end = ranges[id(spec.output)]
        for tensor, _, _, _, _ in self._operand_specs:
            if tensor is spec.output:
                continue
            a, b = ranges[id(tensor)]
            if not (end <= a or b <= start):
                raise ValueError("SM120 SwiGLU output must not overlap an input")
        self.kernel(
            ranges[id(spec.tokens)][0],
            ranges[id(spec.gate.source)][0] + spec.gate.byte_offset,
            ranges[id(spec.up.source)][0] + spec.up.byte_offset,
            ranges[id(spec.offsets)][0],
            ranges[id(spec.output)][0],
            int(spec.rows),
            self._expert_strides[0],
            self._expert_strides[1],
            cuda.CUstream(stream),
        )


def build(graph, knobs):
    from .compiler import _graph_dynamic_shapes

    validate_knobs(knobs)
    spec = analyze(graph, dynamic_shapes=_graph_dynamic_shapes(graph))
    return SimtCompiled(spec, device_params())
