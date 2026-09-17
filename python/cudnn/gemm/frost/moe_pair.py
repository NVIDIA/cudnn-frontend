# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit canonical-parent SwiGLU specialization, sharing public GEMM knobs.

Credit: NVIDIA Frost, KF624, Yanqin Zhai's NVIDIA/cudnn-frontend PR1090,
CUTLASS example113, canonical rank5 pairing and TRT gated-row motivation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path

from cudnn.frost import buffers
from .graph_analyzer import analyze_with_binding
from .knobs import GemmKnobs
from .tile_config import by_name

ENGINE = "frost_moe_swiglu_pair"
CONFIG = "CONFIG_sm100_128x8x128_128x8x32_cluster1x1_1ctamma_swapAB"


def pair_knobs():
    # M counts the physical gate+up projection rows before SwiGLU; one CTA
    # produces 64 output features for up to eight routed tokens.
    return replace(GemmKnobs.from_config(by_name(CONFIG)), moe_sched_policy=1)


@dataclass(frozen=True)
class PairSpec:
    binding: object
    tokens: object
    weight: object
    output: object
    offsets: object
    rows: int
    features: int
    reduction: int
    experts: int


def analyze_pair(graph, *, dynamic_shapes=False):
    """Validate the entire graph and its declared parent relationship, on CPU."""
    if dynamic_shapes:
        raise NotImplementedError("paired MoE currently requires fixed graph dimensions")
    chain, binding = analyze_with_binding(graph)
    if not chain.has_moe or chain.has_block_scale or chain.num_gemms != 2:
        raise NotImplementedError("paired MoE requires two ordinary grouped matmuls")
    if chain.num_a_operands != 1 or chain.num_b_operands != 2 or set(chain.gemm_operands) != {(0, 0), (0, 1)}:
        raise NotImplementedError("paired MoE requires one shared token input and two distinct weights")
    if chain.aux_tensors or chain.reductions or chain.quants or chain.mainloop_a_ops or chain.mainloop_b_ops:
        raise NotImplementedError("paired MoE supports only the SwiGLU epilogue")
    mm, moe = chain.matmul, chain.moe
    if (mm.a_dtype, mm.b_dtype, mm.accum_dtype, mm.out_dtype, mm.a_major, mm.b_major) != ("bf16", "bf16", "fp32", "fp32", "k", "k"):
        raise NotImplementedError("paired MoE requires BF16 operands and FP32 projections")
    if moe.offset_dtype != "int32" or moe.weight_layout is not None or moe.num_groups != moe.num_experts:
        raise NotImplementedError("paired MoE requires one int32 offset group per canonical expert")
    if not 1 <= mm.M <= 513 or mm.N <= 0 or mm.K <= 0 or moe.num_experts <= 0 or mm.N % 64 or mm.K % 64:
        raise NotImplementedError("paired MoE requires 1<=R<=513 and N,K divisible by64")
    if len(chain.ops) != 2 or len(chain.output_specs) != 1 or len(binding.outputs) != 1:
        raise NotImplementedError("paired MoE requires exactly swish then multiply and one output")
    swish, mul = chain.ops
    if swish.op != "swish" or mul.op != "mul" or swish.aux is not None or mul.aux is not None:
        raise NotImplementedError("paired MoE requires swish(gate)*up")
    gate_ref = swish.resolved_parent_idx(0)
    if gate_ref not in (-1, -2) or {mul.resolved_parent_idx(1), mul.parent_idx_b} != {0, -3 - gate_ref}:
        raise NotImplementedError("paired MoE requires independent gate and up projections")
    if dict(swish.attrs) not in ({}, {"swish_beta": 1.0}) or mul.attrs:
        raise NotImplementedError("paired MoE supports the standard unit-beta SwiGLU")
    if swish.compute_dtype != "fp32" or swish.out_dtype != "fp32" or mul.compute_dtype != "fp32":
        raise NotImplementedError("paired MoE must keep the intermediate activation in FP32")
    output = chain.output_specs[0]
    if output.source_ref != 1 or output.dtype != "bf16" or mul.out_dtype not in ("fp32", "bf16"):
        raise NotImplementedError("paired MoE requires one BF16 final output")
    views = {id(view.tensor): view for view in binding.operand_slices}
    if len(views) != 2:
        raise NotImplementedError("paired MoE requires explicit slices of one parent weight")
    gate_index = -1 - gate_ref
    gate = views.get(id(binding.b_operands[chain.gemm_operands[gate_index][1]]))
    up = views.get(id(binding.b_operands[chain.gemm_operands[1 - gate_index][1]]))
    if gate is None or up is None or gate.parent is not up.parent:
        raise NotImplementedError("paired MoE weights must declare the same parent")
    e, n, k, r = moe.num_experts, mm.N, mm.K, mm.M
    if gate.parent_dim != (e, k, 2 * n) or gate.byte_offset != 0 or up.byte_offset != n * gate.parent_stride[2] * 2:
        raise NotImplementedError("paired MoE requires complete [gate,up] halves of its parent")
    se, sk, sn = gate.parent_stride
    if sk != 1 or se % 8 or sn % 8:
        raise NotImplementedError("paired MoE weight TMA strides must be16-byte aligned")
    tx, ty, offsets = binding.a_operands[0], binding.outputs[0], binding.first_token_offset
    geometry = (
        (tx, (1, r, k), (r * k, k, 1)),
        (ty, (1, r, n), (r * n, n, 1)),
        (offsets, (e, 1, 1), (1, 1, 1)),
    )
    for tensor, dim, stride in geometry:
        if tuple(tensor.get_dim()) != dim or tuple(tensor.get_stride()) != stride:
            raise NotImplementedError("paired MoE requires compact tokens/output and a contiguous offset vector")
    return PairSpec(binding, tx, gate.parent, ty, offsets, r, n, k, e)


@dataclass(frozen=True)
class KernelParams:
    grid_ctas: int
    l2_budget_bytes: int
    helper_digest: str


def device_params():
    from cudnn.frost import device
    from .compiler import probe_cutedsl

    probe_cutedsl()  # before importing any DSL-version-specific kernel API
    ordinal = device.resolve_device(None)
    if device.compute_capability(ordinal) != (10, 0):
        raise NotImplementedError("paired MoE currently supports SM100")
    smem = max(device.shared_memory_per_block_optin(ordinal), device.oversized_shared_memory_per_block(ordinal))
    if smem < 224 * 1024:
        raise NotImplementedError("paired MoE requires at least224KiB shared memory per CTA")
    root = Path(__file__).resolve().parent
    helpers = [root / "sm100/kernel_templates/_tile_helpers.py", root / "kernel_templates/moe_scheduler.py"]
    # Include native helper source in the plan-time compile identity.
    digest = hashlib.sha256(b"".join(path.read_bytes() for path in helpers)).hexdigest()
    return KernelParams(device.multiprocessor_count(ordinal), device.l2_cache_bytes(ordinal) // 3, digest)


class PairedCompiled:
    """Framework-neutral callable consumed by the existing Frost graph plan."""

    def __init__(self, spec, params):
        from cudnn.frost.template_loader import load_template

        self.spec = spec
        self.binding = spec.binding
        self.workspace_bytes = (params.grid_ctas * 2 + 1) * 128
        path = Path(__file__).resolve().parent / "sm100/kernel_templates/sm100_moe_swiglu_pair.py"
        self.module = load_template(str(path), params, tag="moe_swiglu_pair")
        self.kernel = self.module.compile()  # all compilation belongs to build

    def __call__(self, pack, workspace, *, stream):
        from cuda.bindings import driver as cuda

        spec = self.spec
        tensors = (spec.tokens, spec.weight, spec.output, spec.offsets)
        operands = [pack[tensor] for tensor in tensors]
        for tensor, buf, dtype in zip(tensors, operands, ("bfloat16", "bfloat16", "bfloat16", "int32")):
            if tuple(buf.shape) != tuple(tensor.get_dim()) or tuple(buf.stride()) != tuple(tensor.get_stride()) or buffers.dtype_name(buf) != dtype:
                raise ValueError("paired MoE operand must match its declared shape, stride and dtype")
            if int(buf.__dlpack_device__()[0]) != 2 or int(buf.data_ptr()) % (4 if dtype == "int32" else 16):
                raise ValueError("paired MoE requires aligned CUDA operands")
        x, parent, out, offsets = operands
        a, b, c = x.permute(1, 2, 0), parent.permute(2, 1, 0), out.permute(1, 2, 0)
        scratch = workspace.view(0, "int64", (self.workspace_bytes // 8,))
        problem = (spec.rows, spec.features, spec.reduction, spec.experts, spec.experts, *a.stride(), *b.stride(), *c.stride())
        self.kernel(problem, offsets.view(-1), scratch, a, b, b, c, stream=cuda.CUstream(stream))


def build_pair(graph, knobs):
    from .compiler import _graph_dynamic_shapes

    spec = analyze_pair(graph, dynamic_shapes=_graph_dynamic_shapes(graph))
    expected = pair_knobs()
    if knobs is not None and knobs != expected:
        raise NotImplementedError("paired MoE currently supports only its exact M128N8 static configuration")
    params = device_params()
    return PairedCompiled(spec, params)
