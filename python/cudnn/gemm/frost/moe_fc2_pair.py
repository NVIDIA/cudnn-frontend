# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small-row BF16 grouped projection with a paired, identity-store MMA.

Credit: NVIDIA Frost, KF624, Yanqin Zhai's PR1090, CUTLASS example113,
and the canonical rank5 pairing used by the SwiGLU sibling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from cudnn.frost import buffers
from .graph_analyzer import analyze_with_binding
from .knobs import GemmKnobs
from .moe_pair import KernelParams, device_params, pair_knobs

ENGINE = "frost_moe_fc2_pair"


@dataclass(frozen=True)
class Fc2Knobs:
    """Exact paired-FC2 geometry and A/B pipeline depth in public knob terms.

    Missing STAGES preserves the original twelve-stage plan. The six-stage
    alternative comes from KF2132/e459dff; its benefit depends on the workload.
    """

    geometry: GemmKnobs = field(default_factory=pair_knobs)
    ab_stages: int = 12

    def __post_init__(self):
        if self.geometry != pair_knobs():
            raise ValueError("paired FC2 supports only its exact M128N8 static configuration")
        if isinstance(self.ab_stages, bool) or not isinstance(self.ab_stages, int) or self.ab_stages not in (6, 12):
            raise ValueError("paired FC2 STAGES must be 6 or 12")

    def to_public(self):
        import cudnn

        public = self.geometry.to_public()
        if self.ab_stages != 12:
            public[cudnn.knob_type.STAGES] = self.ab_stages
        return public

    @classmethod
    def from_public(cls, public):
        import cudnn

        geometry = {}
        stages = 12
        seen_stages = False
        for knob, value in public.items():
            knob = cudnn.knob_type(int(knob))
            if knob == cudnn.knob_type.STAGES:
                if seen_stages:
                    raise ValueError("paired FC2 STAGES must be specified once")
                stages, seen_stages = value, True
            else:
                geometry[knob] = value
        return cls(GemmKnobs.from_public(geometry), stages)


def _fc2_knobs(knobs):
    if knobs is None:
        return Fc2Knobs()
    if isinstance(knobs, Fc2Knobs):
        return knobs
    if isinstance(knobs, GemmKnobs):
        return Fc2Knobs(geometry=knobs)
    if isinstance(knobs, dict):
        return Fc2Knobs.from_public(knobs)
    raise ValueError("paired FC2 requires its exact public knob record")


@dataclass(frozen=True)
class Fc2KernelParams(KernelParams):
    # Frozen template parameters also distinguish the persistent compile key.
    ab_stages: int = 12


@dataclass(frozen=True)
class Fc2Spec:
    binding: object
    tokens: object
    weight: object
    output: object
    offsets: object
    rows: int
    features: int
    reduction: int
    experts: int


def analyze_fc2(graph, *, dynamic_shapes=False):
    """Recognize only a direct BF16 grouped projection; no epilogue is ignored."""
    if dynamic_shapes:
        raise NotImplementedError("paired FC2 currently requires fixed graph dimensions")
    chain, binding = analyze_with_binding(graph)
    if not chain.has_moe or chain.has_block_scale or chain.num_gemms != 1:
        raise NotImplementedError("paired FC2 requires one ordinary grouped matmul")
    if chain.num_a_operands != 1 or chain.num_b_operands != 1 or chain.gemm_operands != [(0, 0)]:
        raise NotImplementedError("paired FC2 requires one token input and one expert weight")
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
        raise NotImplementedError("paired FC2 supports only a direct projection without operand transforms")
    mm, moe = chain.matmul, chain.moe
    if (mm.a_dtype, mm.b_dtype, mm.accum_dtype, mm.out_dtype, mm.a_major, mm.b_major) != ("bf16", "bf16", "fp32", "bf16", "k", "k"):
        raise NotImplementedError("paired FC2 requires BF16 operands/output and FP32 accumulation")
    if moe.mode != "none" or moe.offset_dtype != "int32" or moe.weight_layout is not None or moe.num_groups != moe.num_experts:
        raise NotImplementedError("paired FC2 requires one int32 offset group per canonical expert")
    if not 1 <= mm.M <= 513 or mm.N <= 0 or mm.N % 128 or mm.K <= 0 or mm.K % 64 or moe.num_experts <= 0:
        raise NotImplementedError("paired FC2 requires 1<=R<=513, output width divisible by128 and K divisible by64")
    if len(chain.output_specs) != 1 or len(binding.outputs) != 1:
        raise NotImplementedError("paired FC2 requires exactly one BF16 output")
    output = chain.output_specs[0]
    if output.source_ref != -1 or output.dtype != "bf16" or output.quant_idx is not None:
        raise NotImplementedError("paired FC2 requires the direct BF16 projection output")
    tx, tw, ty, offsets = binding.a_operands[0], binding.b_operands[0], binding.outputs[0], binding.first_token_offset
    e, n, k, rows = moe.num_experts, mm.N, mm.K, mm.M
    if tuple(tw.get_dim()) != (e, k, n):
        raise NotImplementedError("paired FC2 requires canonical expert weight dimensions")
    se, sk, sn = tw.get_stride()
    if sk != 1 or sn < k or se < (n - 1) * sn + k or se % 8 or sn % 8:
        raise NotImplementedError("paired FC2 requires nonoverlapping weight rows/experts with16-byte TMA strides")
    for tensor, dim, stride in (
        (tx, (1, rows, k), (rows * k, k, 1)),
        (ty, (1, rows, n), (rows * n, n, 1)),
        (offsets, (e, 1, 1), (1, 1, 1)),
    ):
        if tuple(tensor.get_dim()) != dim or tuple(tensor.get_stride()) != stride:
            raise NotImplementedError("paired FC2 requires compact tokens/output and contiguous expert offsets")
    return Fc2Spec(binding, tx, tw, ty, offsets, rows, n, k, e)


class Fc2Compiled:
    """Native metadata binding; compilation and descriptors are plan-time work."""

    def __init__(self, spec, params):
        from cudnn.frost.template_loader import load_template

        self.spec, self.binding = spec, spec.binding
        self.workspace_bytes = (params.grid_ctas * 2 + 1) * 128
        path = Path(__file__).resolve().parent / "sm100/kernel_templates/sm100_moe_fc2_pair.py"
        self.module = load_template(str(path), params, tag="moe_fc2_pair")
        self.kernel = self.module.compile()

    def __call__(self, pack, workspace, *, stream):
        from cuda.bindings import driver as cuda

        spec = self.spec
        tensors = (spec.tokens, spec.weight, spec.output, spec.offsets)
        operands = [pack[tensor] for tensor in tensors]
        for tensor, buf, dtype in zip(tensors, operands, ("bfloat16", "bfloat16", "bfloat16", "int32")):
            if tuple(buf.shape) != tuple(tensor.get_dim()) or tuple(buf.stride()) != tuple(tensor.get_stride()) or buffers.dtype_name(buf) != dtype:
                raise ValueError("paired FC2 operand must match its declared shape, stride and dtype")
            if int(buf.__dlpack_device__()[0]) != 2 or int(buf.data_ptr()) % (4 if dtype == "int32" else 16):
                raise ValueError("paired FC2 requires aligned CUDA operands")
        x, weight, out, offsets = operands
        a, b, c = x.permute(1, 2, 0), weight.permute(2, 1, 0), out.permute(1, 2, 0)
        scratch = workspace.view(0, "int64", (self.workspace_bytes // 8,))
        # The paired ABI counts channels per half; identity stores write both halves.
        problem = (spec.rows, spec.features // 2, spec.reduction, spec.experts, spec.experts, *a.stride(), *b.stride(), *c.stride())
        self.kernel(problem, offsets.view(-1), scratch, a, b, b, c, stream=cuda.CUstream(stream))


def build_fc2(graph, knobs):
    from .compiler import _graph_dynamic_shapes

    spec = analyze_fc2(graph, dynamic_shapes=_graph_dynamic_shapes(graph))
    try:
        knobs = _fc2_knobs(knobs)
    except ValueError as exc:
        raise NotImplementedError(str(exc)) from exc
    params = Fc2KernelParams(**asdict(device_params()), ab_stages=knobs.ab_stages)
    return Fc2Compiled(spec, params)
