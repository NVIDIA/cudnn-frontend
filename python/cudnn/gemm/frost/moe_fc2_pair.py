# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small-row BF16 grouped projection with a paired, identity-store MMA.

Credit: NVIDIA Frost, KF624, Yanqin Zhai's PR1090, CUTLASS example113,
and the canonical rank5 pairing used by the SwiGLU sibling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cudnn.frost import buffers
from .graph_analyzer import analyze_with_binding
from .moe_pair import device_params, pair_knobs

ENGINE = "frost_moe_fc2_pair"


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
    if not 1 <= mm.M <= 8 or mm.N <= 0 or mm.N % 128 or mm.K <= 0 or mm.K % 64 or moe.num_experts <= 0:
        raise NotImplementedError("paired FC2 requires 1<=R<=8, output width divisible by128 and K divisible by64")
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
    if knobs is not None and knobs != pair_knobs():
        raise NotImplementedError("paired FC2 supports only its exact M128N8 static configuration")
    return Fc2Compiled(spec, device_params())
