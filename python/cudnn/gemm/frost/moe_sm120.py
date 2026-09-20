# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit SM120 BF16 grouped projection using the existing graph contract.

Credit: NVIDIA Frost/CuTeDSL/CUTLASS; Yanqin Zhai's PR1090 and Yanqin/Yihua
collaboration; NVIDIA CuTeDSL MegaMoE coordinate reasoning. This implementation
adapts and combines those contributions for the supported SM120 device.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import hashlib

from cudnn.frost import buffers
from .graph_analyzer import analyze_with_binding
from .knobs import GemmKnobs

ENGINE = "frost_moe_fc2_sm120"


def geometry(policy=1):
    return GemmKnobs(
        pipeline_arch=120,
        cta_tile_m=128,
        cta_tile_n=8,
        cta_tile_k_bytes=128,
        mma_tile_m=16,
        mma_tile_n=8,
        mma_tile_k_bytes=32,
        cga_size_m=1,
        cga_size_n=1,
        warps_m=4,
        warps_n=1,
        swap_ab=True,
        moe_sched_policy=policy,
    )


@dataclass(frozen=True)
class Sm120Knobs:
    geometry: GemmKnobs = field(default_factory=geometry)
    ab_stages: int = 2

    def __post_init__(self):
        if self.geometry not in (geometry(0), geometry(1)):
            raise ValueError("SM120 projection requires its exact M128N8 configuration")
        if type(self.ab_stages) is not int or self.ab_stages != 2:
            raise ValueError("SM120 projection STAGES must be 2")

    def to_public(self):
        import cudnn

        return {**self.geometry.to_public(), cudnn.knob_type.STAGES: self.ab_stages}

    @classmethod
    def from_public(cls, public):
        import cudnn

        fields, stages = {}, 2
        seen = set()
        for key, value in public.items():
            key = cudnn.knob_type(int(key))
            if key in seen:
                raise ValueError("SM120 projection knob must be specified once")
            seen.add(key)
            if key == cudnn.knob_type.STAGES:
                stages = value
            else:
                fields[key] = value
        return cls(GemmKnobs.from_public(fields), stages)


def _knobs(value):
    if value is None:
        return Sm120Knobs()
    if isinstance(value, Sm120Knobs):
        return value
    if isinstance(value, GemmKnobs):
        return Sm120Knobs(value)
    if isinstance(value, dict):
        return Sm120Knobs.from_public(value)
    raise ValueError("SM120 projection requires its exact public knob record")


@dataclass(frozen=True)
class ProjectionSpec:
    binding: object
    tokens: object
    weight: object
    output: object
    offsets: object
    rows: int
    features: int
    reduction: int
    experts: int
    groups: int


def analyze(graph, *, dynamic_shapes=False):
    if dynamic_shapes:
        raise NotImplementedError("SM120 projection currently requires fixed graph dimensions")
    chain, binding = analyze_with_binding(graph)
    if not chain.has_moe or chain.has_block_scale or chain.num_gemms != 1:
        raise NotImplementedError("SM120 projection requires one ordinary grouped matmul")
    if chain.num_a_operands != 1 or chain.num_b_operands != 1 or chain.gemm_operands != [(0, 0)]:
        raise NotImplementedError("SM120 projection requires one token input and one expert weight")
    if (
        chain.ops
        or chain.aux_tensors
        or chain.reductions
        or chain.quants
        or chain.has_mainloop_fusion
        or chain.mainloop_a_load_dtype
        or chain.mainloop_b_load_dtype
        or binding.operand_slices
    ):
        raise NotImplementedError("SM120 projection supports only a direct projection without operand transforms")
    mm, moe = chain.matmul, chain.moe
    if (mm.a_dtype, mm.b_dtype, mm.accum_dtype, mm.out_dtype, mm.a_major, mm.b_major) != ("bf16", "bf16", "fp32", "bf16", "k", "k"):
        raise NotImplementedError("SM120 projection requires BF16 operands/output and FP32 accumulation")
    if moe.mode != "none" or moe.offset_dtype != "int32" or moe.weight_layout is not None:
        raise NotImplementedError("SM120 projection requires canonical expert weights and int32 group offsets")
    r, n, k, e, g = mm.M, mm.N, mm.K, moe.num_experts, moe.num_groups
    if not (1 <= r <= 65537 and 1 <= e <= 128 and 1 <= g <= 128 and 16 <= n <= 8192 and n % 16 == 0 and 8 <= k <= 8192 and k % 8 == 0):
        raise NotImplementedError("SM120 projection requires 1<=R<=65537, 1<=E,G<=128, 16<=N<=8192 divisible by16 and 8<=K<=8192 divisible by8")
    if len(chain.output_specs) != 1 or len(binding.outputs) != 1:
        raise NotImplementedError("SM120 projection requires exactly one BF16 output")
    output = chain.output_specs[0]
    if output.source_ref != -1 or output.dtype != "bf16" or output.quant_idx is not None:
        raise NotImplementedError("SM120 projection requires the direct BF16 projection output")
    x, w, y, off = binding.a_operands[0], binding.b_operands[0], binding.outputs[0], binding.first_token_offset
    for tensor, dim, stride in (
        (x, (1, r, k), (r * k, k, 1)),
        (w, (e, k, n), (n * k, 1, k)),
        (y, (1, r, n), (r * n, n, 1)),
        (off, (g, 1, 1), (1, 1, 1)),
    ):
        if tuple(tensor.get_dim()) != dim or tuple(tensor.get_stride()) != stride:
            raise NotImplementedError("SM120 projection requires compact tokens/output/offsets and canonical contiguous expert weight storage")
    return ProjectionSpec(binding, x, w, y, off, r, n, k, e, g)


@dataclass(frozen=True)
class KernelParams:
    device_ordinal: int
    grid_ctas: int
    source_digest: str
    static_sched: bool = True
    ab_stages: int = 2


def device_params():
    from cudnn.frost import device
    from .compiler import probe_cutedsl

    probe_cutedsl()
    ordinal = device.resolve_device(None)
    if device.compute_capability(ordinal) != (12, 0) or device.multiprocessor_count(ordinal) != 188:
        raise NotImplementedError("SM120 projection currently supports the 188-SM SM120 device")
    if device.shared_memory_per_block_optin(ordinal) < 40 * 1024:
        raise NotImplementedError("SM120 projection requires at least40KiB shared memory per CTA")
    root = Path(__file__).resolve().parent
    files = [Path(__file__), root / "sm120/kernel_templates/sm120_moe_fc2.py", root / "kernel_templates/moe_scheduler.py"]
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()
    return KernelParams(ordinal, 188, digest)


class ProjectionCompiled:
    workspace_bytes = 128

    def __init__(self, spec, params):
        from cudnn.frost.template_loader import load_template

        self.spec, self.binding, self.params = spec, spec.binding, params
        path = Path(__file__).resolve().parent / "sm120/kernel_templates/sm120_moe_fc2.py"
        self.module = load_template(str(path), params, tag="moe_fc2_sm120")
        self.generated_path = path
        self.kernel = self.module.compile()

    def __call__(self, pack, workspace, *, stream):
        from cuda.bindings import driver as cuda

        if stream is None:
            raise ValueError("SM120 projection execution requires an explicit stream")
        spec = self.spec
        tensors = (spec.tokens, spec.weight, spec.output, spec.offsets)
        operands = [pack[t] for t in tensors]
        pointers = []
        for tensor, buf, dtype in zip(tensors, operands, ("bfloat16", "bfloat16", "bfloat16", "int32")):
            if tuple(buf.shape) != tuple(tensor.get_dim()) or tuple(buf.stride()) != tuple(tensor.get_stride()) or buffers.dtype_name(buf) != dtype:
                raise ValueError("SM120 projection operand must match its declared shape, stride and dtype")
            pointer = int(buf.data_ptr())
            if tuple(map(int, buf.__dlpack_device__())) != (2, self.params.device_ordinal) or pointer <= 0 or pointer % (4 if dtype == "int32" else 16):
                raise ValueError("SM120 projection requires aligned CUDA operands on its compiled device")
            pointers.append(pointer)
        sizes = (spec.rows * spec.reduction * 2, spec.experts * spec.features * spec.reduction * 2, spec.rows * spec.features * 2, spec.groups * 4)
        output, output_end = pointers[2], pointers[2] + sizes[2]
        for index in (0, 1, 3):
            if output < pointers[index] + sizes[index] and pointers[index] < output_end:
                raise ValueError("SM120 projection output must not overlap an input")
        if workspace is None:
            raise ValueError("SM120 projection requires its caller-owned workspace")
        scratch = workspace.view(0, "int64", (16,))
        scratch_pointer = int(scratch.data_ptr())
        if tuple(map(int, scratch.__dlpack_device__())) != (2, self.params.device_ordinal) or scratch_pointer <= 0 or scratch_pointer % 128:
            raise ValueError("SM120 projection workspace must be 128-byte aligned on its compiled device")
        for pointer, size in zip(pointers, sizes):
            if scratch_pointer < pointer + size and pointer < scratch_pointer + self.workspace_bytes:
                raise ValueError("SM120 projection workspace must not overlap an operand")
        x, w, y, off = operands
        a, b, c = w.permute(2, 1, 0), x.permute(1, 2, 0), y.permute(2, 1, 0)
        problem = (spec.features, spec.rows, spec.reduction, spec.experts, spec.groups, *a.stride(), *b.stride(), *c.stride())
        self.kernel(problem, off.view(-1), scratch, a, b, c, stream=cuda.CUstream(stream))


def build(graph, knobs):
    from .compiler import _graph_dynamic_shapes

    spec = analyze(graph, dynamic_shapes=_graph_dynamic_shapes(graph))
    try:
        knobs = _knobs(knobs)
    except ValueError as exc:
        raise NotImplementedError(str(exc)) from exc
    params = replace(device_params(), static_sched=bool(knobs.geometry.moe_sched_policy), ab_stages=knobs.ab_stages)
    return ProjectionCompiled(spec, params)
