#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark-local ModelOpt 0.46.0 NVFP4 adapter for Qwen-Image.

This module is deliberately not a general quantization framework.  It recognizes
the exact fourteen ``nn.Linear`` roles in the pinned Diffusers Qwen-Image block,
collects one synthetic full-precision max-calibration pass, freezes ModelOpt's
two-level NVFP4 scales, and installs three strict benchmark treatments:

``A``
    Original bf16 linears/GELU FFNs.
``B``
    Original bf16 non-MLP linears plus ``cudnn.gemm.ops.gelu_mlp``.
``C``
    Native NVFP4 for all fourteen logical linears.  The two FFNs use a private
    FROST FC1+bias+GELU+requant graph so the FC2 input quantization is fused.

Weights are quantized once during setup.  Activations use fixed max-calibrated
global scales and are quantized at runtime.  There is no fallback: structure,
shape, route, cache-sharing, or kernel-contract drift raises.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import copy
import types

MODELOPT_RECIPE = {
    "id": "qwen-image-modelopt-0.46.0-nvfp4-max-interior-v1",
    "project": "NVIDIA/Model-Optimizer",
    "repo_url": "https://github.com/NVIDIA/Model-Optimizer",
    "release": "0.46.0",
    "commit": "43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a",
    "upstream_anchor_args": "--model qwen-image --format fp4 --quant-algo max",
    "proxy_overrides": {
        "model_dtype": "BFloat16 (ModelOpt CLI default is Half)",
        "trt_high_precision_dtype": "BFloat16 (ModelOpt CLI default is Half)",
        "calibration": "one deterministic synthetic pass instead of the upstream calibration workload",
        "depth": "four representative blocks projected onto full-model blocks [2, 20, 39, 57]",
        "weights": "offline one-time weight prepacking/compression during benchmark setup",
    },
    "alignment_scope": "Linear placement plus NVFP4 format/block scaling and max-policy; not exact dtype, calibration state, or workload",
    "quantize_mha": False,
    "linear_format": "NVFP4 E2M1, block_size=16, E4M3 block scales",
    "calibration": "one deterministic synthetic bf16 max-observer pass; frozen before timing",
    "attention": "bf16 core; ModelOpt quantize_mha is false",
    "scope": "representative quantized middle transformer blocks",
    "full_model_quantized_blocks": [2, 57],
    "full_model_excluded_blocks": [0, 1, 58, 59],
    "numerical_claim_eligible": False,
    "sources": {
        "selection": "examples/diffusers/quantization/quantize.py",
        "qwen_defaults": "examples/diffusers/quantization/models_utils.py",
        "preset": "modelopt_recipes/configs/ptq/presets/diffusers/nvfp4.yaml",
        "numerics": "modelopt_recipes/configs/numerics/nvfp4.yaml",
        "mha_policy": "examples/diffusers/quantization/utils.py",
        "real_backend": "modelopt/torch/quantization/backends/nvfp4_gemm.py",
    },
    "source_permalinks": {
        "selection": "https://github.com/NVIDIA/Model-Optimizer/blob/43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a/examples/diffusers/quantization/quantize.py",
        "qwen_defaults": "https://github.com/NVIDIA/Model-Optimizer/blob/43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a/examples/diffusers/quantization/models_utils.py",
        "preset": "https://github.com/NVIDIA/Model-Optimizer/blob/43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a/modelopt_recipes/configs/ptq/presets/diffusers/nvfp4.yaml",
        "numerics": "https://github.com/NVIDIA/Model-Optimizer/blob/43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a/modelopt_recipes/configs/numerics/nvfp4.yaml",
        "mha_policy": "https://github.com/NVIDIA/Model-Optimizer/blob/43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a/examples/diffusers/quantization/utils.py",
        "real_backend": "https://github.com/NVIDIA/Model-Optimizer/blob/43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a/modelopt/torch/quantization/backends/nvfp4_gemm.py",
    },
}

ARM_CONFIGS = {
    "A": {"generic_linear": "bf16", "mlp": "torch"},
    "B": {"generic_linear": "bf16", "mlp": "cudnn_bf16"},
    "C": {"generic_linear": "nvfp4", "mlp": "nvfp4"},
}

ROLE_ORDER = (
    "img_mod.1",
    "txt_mod.1",
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_out.0",
    "attn.to_add_out",
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
)

MLP_ROLES = frozenset(
    {
        "img_mlp.net.0.proj",
        "img_mlp.net.2",
        "txt_mlp.net.0.proj",
        "txt_mlp.net.2",
    }
)

_MODELOPT_FP8_MAX = 448.0
_NVFP4_E2M1_MAX = 6.0
_BLOCK_SIZE = 16
_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
_LINEAR_REFERENCE_RTOL = 2.0e-2
_LINEAR_REFERENCE_ATOL = 2.0e-1
_FUSED_HIDDEN_REFERENCE_REL_L2 = 2.5e-1
_FUSED_HIDDEN_ORACLE_REL_L2 = 1.2e-1


def _ceil_to(value, multiple):
    return ((int(value) + multiple - 1) // multiple) * multiple


def representative_middle_blocks(layers):
    """Evenly map a depth-reduced proxy onto Qwen-Image's full blocks 2..57."""
    if not isinstance(layers, int) or not 1 <= layers <= 56:
        raise ValueError(f"representative middle-block count must be in [1, 56], got {layers!r}")
    if layers == 1:
        return [30]
    lo, span = 2, 55
    # Round halves up, not Python's round-to-even.  Four layers map exactly to
    # the declared review anchors [2, 20, 39, 57].
    result = [lo + (2 * i * span + layers - 1) // (2 * (layers - 1)) for i in range(layers)]
    if len(set(result)) != layers or result[0] != 2 or result[-1] != 57:
        raise AssertionError(f"invalid representative middle-block mapping: {result}")
    return result


def three_arm_orders():
    """Position- and first-order-carryover-balanced design for three arms."""
    orders = (
        (0, 1, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
        (0, 2, 1),
        (1, 0, 2),
    )
    positions = Counter((position, treatment) for order in orders for position, treatment in enumerate(order))
    carryover = Counter(pair for order in orders for pair in zip(order, order[1:]))
    if set(positions.values()) != {2} or len(positions) != 9:
        raise AssertionError(f"unbalanced three-arm positions: {positions}")
    if set(carryover.values()) != {2} or len(carryover) != 6:
        raise AssertionError(f"unbalanced three-arm carryover: {carryover}")
    return orders


def expected_input_shapes(shape):
    """Exact activation shape at each Linear boundary for one proxy block."""
    batch = shape["bs"]
    hidden = shape["hidden"]
    ffn = shape["ffn"]
    image = shape["image_tokens"]
    text = shape["text_tokens"]
    return {
        "img_mod.1": (batch, hidden),
        "txt_mod.1": (batch, hidden),
        "attn.to_q": (batch, image, hidden),
        "attn.to_k": (batch, image, hidden),
        "attn.to_v": (batch, image, hidden),
        "attn.add_q_proj": (batch, text, hidden),
        "attn.add_k_proj": (batch, text, hidden),
        "attn.add_v_proj": (batch, text, hidden),
        "attn.to_out.0": (batch, image, hidden),
        "attn.to_add_out": (batch, text, hidden),
        "img_mlp.net.0.proj": (batch, image, hidden),
        "img_mlp.net.2": (batch, image, ffn),
        "txt_mlp.net.0.proj": (batch, text, hidden),
        "txt_mlp.net.2": (batch, text, ffn),
    }


def expected_weight_shapes(shape):
    hidden, ffn = shape["hidden"], shape["ffn"]
    return {
        "img_mod.1": (6 * hidden, hidden),
        "txt_mod.1": (6 * hidden, hidden),
        "attn.to_q": (hidden, hidden),
        "attn.to_k": (hidden, hidden),
        "attn.to_v": (hidden, hidden),
        "attn.add_q_proj": (hidden, hidden),
        "attn.add_k_proj": (hidden, hidden),
        "attn.add_v_proj": (hidden, hidden),
        "attn.to_out.0": (hidden, hidden),
        "attn.to_add_out": (hidden, hidden),
        "img_mlp.net.0.proj": (ffn, hidden),
        "img_mlp.net.2": (hidden, ffn),
        "txt_mlp.net.0.proj": (ffn, hidden),
        "txt_mlp.net.2": (hidden, ffn),
    }


def expected_plan_contracts(shape):
    """The seven distinct low-precision plan contracts in the pinned proxy."""
    hidden, ffn = shape["hidden"], shape["ffn"]
    image, text = shape["image_tokens"], shape["text_tokens"]
    contracts = (
        (1, 6 * hidden, hidden, "linear_bias"),
        (text, hidden, hidden, "linear_bias"),
        (image, hidden, hidden, "linear_bias"),
        (text, hidden, ffn, "linear_bias"),
        (image, hidden, ffn, "linear_bias"),
        (text, ffn, hidden, "linear_bias_gelu_nvfp4"),
        (image, ffn, hidden, "linear_bias_gelu_nvfp4"),
    )
    if len(set(contracts)) != 7:
        raise ValueError(f"proxy dimensions collapse distinct NVFP4 plan contracts: {contracts}")
    return contracts


def expected_route_delta(arm, layers):
    if arm not in ARM_CONFIGS:
        raise ValueError(f"unknown arm {arm!r}")
    zero_roles = {f"transformer_blocks.{block}.{role}": 0 for block in range(layers) for role in ROLE_ORDER}
    common = {
        "weight_pack_calls": 0,
        "plan_build_calls": 0,
        "fallback_calls": 0,
        "forward_scopes": 1,
        "nvfp4_linear_by_role": zero_roles,
    }
    if arm == "A":
        return {
            **common,
            "bf16_linear_calls": 14 * layers,
            "nvfp4_linear_calls": 0,
            "activation_quant_logical": 0,
            "activation_quant_physical": 0,
            "activation_quant_standalone": 0,
            "activation_quant_fused": 0,
            "activation_cache_hits": 0,
            "mlp_calls": {"torch": 2 * layers, "cudnn_bf16": 0, "nvfp4": 0},
        }
    if arm == "B":
        return {
            **common,
            "bf16_linear_calls": 14 * layers,
            "nvfp4_linear_calls": 0,
            "activation_quant_logical": 0,
            "activation_quant_physical": 0,
            "activation_quant_standalone": 0,
            "activation_quant_fused": 0,
            "activation_cache_hits": 0,
            "mlp_calls": {"torch": 0, "cudnn_bf16": 2 * layers, "nvfp4": 0},
        }
    common["nvfp4_linear_by_role"] = {f"transformer_blocks.{block}.{role}": 1 for block in range(layers) for role in ROLE_ORDER}
    physical = 1 + 8 * layers
    return {
        **common,
        "bf16_linear_calls": 0,
        "nvfp4_linear_calls": 14 * layers,
        "activation_quant_logical": 14 * layers,
        "activation_quant_physical": physical,
        "activation_quant_standalone": 1 + 6 * layers,
        "activation_quant_fused": 2 * layers,
        "activation_cache_hits": 14 * layers - physical,
        "mlp_calls": {"torch": 0, "cudnn_bf16": 0, "nvfp4": 2 * layers},
    }


def counter_delta(after, before):
    """Subtract two counter snapshots with identical nested structure."""
    if set(after) != set(before):
        raise ValueError("counter snapshots have different keys")
    result = {}
    for key in after:
        if isinstance(after[key], dict):
            if not isinstance(before[key], dict):
                raise ValueError(f"counter type changed at {key}")
            result[key] = counter_delta(after[key], before[key])
        else:
            result[key] = int(after[key]) - int(before[key])
    return result


def _resolve_path(module, path):
    value = module
    for component in path.split("."):
        value = value[int(component)] if component.isdigit() else getattr(value, component)
    return value


def _activation_group(block_index, role):
    if role in ("img_mod.1", "txt_mod.1"):
        return "all_blocks.modulation"
    if role in ("attn.to_q", "attn.to_k", "attn.to_v"):
        return f"block{block_index}.image_qkv"
    if role in ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"):
        return f"block{block_index}.text_qkv"
    return f"block{block_index}.{role}"


@dataclass
class _LinearEntry:
    qualified_name: str
    block_index: int
    role: str
    module: object
    original_forward: object
    input_shape: tuple
    weight_shape: tuple
    activation_group: str
    activation_amax: object = None
    activation_global_scale: object = None
    weight_amax: object = None
    weight_global_scale: object = None
    alpha: object = None
    packed_weight: object = None
    weight_scale_factors: object = None

    @property
    def m(self):
        result = 1
        for value in self.input_shape[:-1]:
            result *= value
        return result

    @property
    def k(self):
        return self.input_shape[-1]

    @property
    def n(self):
        return self.weight_shape[0]


def _collect_entries(model, shape, torch):
    if model.__class__.__module__ != "diffusers.models.transformers.transformer_qwenimage" or model.__class__.__name__ != "QwenImageTransformer2DModel":
        raise TypeError(f"expected pinned Diffusers QwenImageTransformer2DModel, got {type(model)!r}")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise NotImplementedError("ModelOpt NVFP4 benchmark treatment is inference-only")
    if getattr(model, "_compiled_call_impl", None) is not None or getattr(model, "peft_config", None):
        raise NotImplementedError("torch.compile and PEFT/LoRA are outside the ModelOpt NVFP4 benchmark treatment")
    blocks = getattr(model, "transformer_blocks", None)
    if not isinstance(blocks, torch.nn.ModuleList) or len(blocks) != shape["layers"]:
        raise TypeError(f"expected {shape['layers']} Qwen-Image transformer blocks, got {type(blocks)!r}/{len(blocks) if blocks is not None else None}")

    input_shapes = expected_input_shapes(shape)
    weight_shapes = expected_weight_shapes(shape)
    hook_fields = (
        "_forward_hooks",
        "_forward_pre_hooks",
        "_backward_hooks",
        "_backward_pre_hooks",
    )
    entries = []
    for block_index, block in enumerate(blocks):
        if block.__class__.__module__ != "diffusers.models.transformers.transformer_qwenimage" or block.__class__.__name__ != "QwenImageTransformerBlock":
            raise TypeError(f"transformer_blocks[{block_index}] is not the pinned QwenImageTransformerBlock: {type(block)!r}")
        if getattr(block, "zero_cond_t", None) is not False:
            raise NotImplementedError("shared modulation quantization requires pinned zero_cond_t=False")
        actual_linears = {name for name, child in block.named_modules() if type(child) is torch.nn.Linear}
        if actual_linears != set(ROLE_ORDER):
            raise TypeError(f"transformer_blocks[{block_index}] Linear roles changed: got {sorted(actual_linears)}, expected {sorted(ROLE_ORDER)}")
        for role in ROLE_ORDER:
            module = _resolve_path(block, role)
            qualified = f"transformer_blocks.{block_index}.{role}"
            if "forward" in module.__dict__:
                raise NotImplementedError(f"{qualified} already has an instance-level forward override")
            if module.bias is None or tuple(module.weight.shape) != weight_shapes[role] or tuple(module.bias.shape) != (weight_shapes[role][0],):
                raise ValueError(
                    f"{qualified} shape changed: weight={tuple(module.weight.shape)}, bias={None if module.bias is None else tuple(module.bias.shape)}, "
                    f"expected {weight_shapes[role]}/{(weight_shapes[role][0],)}"
                )
            for tensor_name, tensor in (
                ("weight", module.weight),
                ("bias", module.bias),
            ):
                if tensor.dtype != torch.bfloat16 or tensor.device.type != "cuda" or not tensor.is_contiguous():
                    raise NotImplementedError(f"{qualified}.{tensor_name} must be contiguous bf16 CUDA")
            if module.training or any(getattr(module, field, None) for field in hook_fields) or torch.nn.utils.parametrize.is_parametrized(module):
                raise NotImplementedError(f"training, hooks, or parametrizations on {qualified} are outside the benchmark treatment")
            entries.append(
                _LinearEntry(
                    qualified_name=qualified,
                    block_index=block_index,
                    role=role,
                    module=module,
                    original_forward=module.forward,
                    input_shape=input_shapes[role],
                    weight_shape=weight_shapes[role],
                    activation_group=_activation_group(block_index, role),
                )
            )
    return entries


def collect_max_calibration(model, shape, forward_call):
    """Collect one untimed BF16 max-observer pass for all exact Linear inputs."""
    import torch

    entries = _collect_entries(model, shape, torch)
    by_module = {entry.module: entry for entry in entries}
    maxima = {}
    calls = Counter()
    handles = []

    def observe(module, args):
        entry = by_module[module]
        if len(args) != 1 or tuple(args[0].shape) != entry.input_shape:
            raise ValueError(f"{entry.qualified_name} calibration input changed: got {[tuple(x.shape) for x in args]}, expected {entry.input_shape}")
        value = args[0]
        if value.dtype != torch.bfloat16 or value.device.type != "cuda" or not value.is_contiguous():
            raise NotImplementedError(f"{entry.qualified_name} calibration input must be contiguous bf16 CUDA")
        observed = value.detach().abs().amax().float()
        maxima[entry.qualified_name] = observed if entry.qualified_name not in maxima else torch.maximum(maxima[entry.qualified_name], observed)
        calls[entry.qualified_name] += 1

    try:
        for entry in entries:
            handles.append(entry.module.register_forward_pre_hook(observe))
        with torch.inference_mode():
            output = forward_call()
    finally:
        for handle in handles:
            handle.remove()

    expected_names = {entry.qualified_name for entry in entries}
    if set(maxima) != expected_names or set(calls.values()) != {1}:
        raise RuntimeError(f"calibration did not visit every Linear exactly once: calls={dict(calls)}")
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("synthetic calibration produced non-finite model output")
    for name, value in maxima.items():
        if not bool(torch.isfinite(value)) or not bool(value > 0):
            raise RuntimeError(f"invalid calibration amax for {name}: {value}")

    # These groups receive mathematically identical BF16 values in the pinned
    # call graph.  Exact equality lets one packed activation serve all consumers.
    grouped = {}
    for entry in entries:
        grouped.setdefault(entry.activation_group, []).append(entry.qualified_name)
    for group, names in grouped.items():
        values = [float(maxima[name].item()) for name in names]
        if len(names) > 1 and any(value != values[0] for value in values[1:]):
            raise RuntimeError(f"shared activation group {group} calibrated different maxima: {dict(zip(names, values))}")
    return {
        "amax": maxima,
        "calls": dict(calls),
        "metadata": {
            "method": "synthetic_bf16_max",
            "passes": 1,
            "frozen_before_timing": True,
            "amax": {name: float(value.item()) for name, value in maxima.items()},
        },
    }


def _tensor_signature(tensor):
    """Identity plus metadata for a buffer baked into a resolved binding."""
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def _run_resolved_with_temporary_output(compiled, resolved, output_id, output, *, stream):
    """Bind one per-call output without retaining it in the prepared mapping."""
    if resolved.get(output_id) is not None:
        raise RuntimeError("prepared NVFP4 output slot is already occupied")
    resolved[output_id] = output
    try:
        return compiled.run_resolved(resolved, stream=stream)
    finally:
        # Do not let the per-entry cache pin every full-model activation.
        resolved.pop(output_id, None)


@dataclass
class _PreparedNvfp4Binding:
    entry: object
    resolved: dict
    resolved_refs: dict
    resolved_signatures: dict
    activation_packed: object
    activation_scale_factors: object
    activation_global_scale: object
    packed_weight: object
    weight_scale_factors: object
    alpha: object
    bias: object
    activation_packed_signature: tuple
    activation_scale_signature: tuple
    activation_global_scale_signature: tuple
    packed_weight_signature: tuple
    weight_scale_signature: tuple
    alpha_signature: tuple
    bias_signature: tuple
    output_id: int | None = None
    hidden_global_scale: object = None
    hidden_global_scale_source: object = None
    hidden_global_scale_signature: tuple | None = None
    hidden_global_scale_source_signature: tuple | None = None
    output_packed: object = None
    output_scale_factors: object = None
    output_packed_signature: tuple | None = None
    output_scale_signature: tuple | None = None


def _validate_resolved_cache(prepared, label):
    if set(prepared.resolved) != set(prepared.resolved_refs):
        raise RuntimeError(f"{label} resolved binding keys changed after preparation")
    for tensor_id, expected in prepared.resolved_refs.items():
        current = prepared.resolved.get(tensor_id)
        if current is not expected or _tensor_signature(current) != prepared.resolved_signatures[tensor_id]:
            raise RuntimeError(f"{label} resolved binding changed after preparation")


def _validate_cached_tensor(label, current, cached, signature):
    if current is not cached or _tensor_signature(current) != signature:
        raise RuntimeError(f"{label} changed after NVFP4 preparation")


class _Nvfp4LinearPlan:
    def __init__(self, torch, cudnn, build_gemm_plan, *, m, n, k, device):
        self.torch = torch
        self.m, self.n, self.k = m, n, k
        sf_k = k // _BLOCK_SIZE
        fp4, fp8 = cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E4M3
        reorder = dict(reordering_type=cudnn.tensor_reordering.F8_128x4)
        with torch.cuda.device(device):
            graph = cudnn.pygraph(
                io_data_type=cudnn.data_type.BFLOAT16,
                intermediate_data_type=cudnn.data_type.FLOAT,
                compute_data_type=cudnn.data_type.FLOAT,
            )
            self.A = graph.tensor(name="A", dim=[1, m, k], stride=[m * k, k, 1], data_type=fp4)
            # Graph descriptors remain logical.  The variant pack carries the
            # larger F8_128x4 physical blob produced by nvfp4_quantize.
            self.SFA = graph.tensor(
                name="SFA",
                dim=[1, m, sf_k],
                stride=[m * sf_k, sf_k, 1],
                data_type=fp8,
                **reorder,
            )
            self.B = graph.tensor(name="B", dim=[1, k, n], stride=[k * n, 1, k], data_type=fp4)
            self.SFB = graph.tensor(
                name="SFB",
                dim=[1, sf_k, n],
                stride=[sf_k * n, 1, sf_k],
                data_type=fp8,
                **reorder,
            )
            self.ALPHA = graph.tensor(
                name="ALPHA",
                dim=[1, 1, 1],
                stride=[1, 1, 1],
                data_type=cudnn.data_type.FLOAT,
            )
            self.BIAS = graph.tensor(
                name="BIAS",
                dim=[1, 1, n],
                stride=[n, n, 1],
                data_type=cudnn.data_type.BFLOAT16,
            )
            ad = graph.block_scale_dequantize(input=self.A, descale=self.SFA, block_size=[1, _BLOCK_SIZE])
            bd = graph.block_scale_dequantize(input=self.B, descale=self.SFB, block_size=[_BLOCK_SIZE, 1])
            acc = graph.matmul(A=ad, B=bd, name="linear")
            corrected = graph.mul(a=acc, b=self.ALPHA, name="modelopt_alpha")
            # ModelOpt's real backend requests a BF16 GEMM result, then performs
            # the official Linear bias add.  Preserve both observable boundaries.
            corrected.set_data_type(cudnn.data_type.BFLOAT16)
            self.Y = graph.bias(input=corrected, bias=self.BIAS, name="bias")
            self.Y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
            self.compiled = build_gemm_plan(graph)
        self.device = device
        self._prepared = {}

    def prepare(self, resolve_variant_pack, activation_buffers, entry):
        """Resolve the stable per-entry bindings once; Y remains per-call."""
        torch = self.torch
        packed, scale_factors = activation_buffers
        if id(entry) in self._prepared:
            raise RuntimeError(f"duplicate prepared NVFP4 binding for {entry.qualified_name}")
        variant_pack = {
            self.A: packed.view(torch.float4_e2m1fn_x2).unsqueeze(0),
            self.SFA: scale_factors.view(torch.float8_e4m3fn).unsqueeze(0),
            self.B: entry.packed_weight.view(torch.float4_e2m1fn_x2).unsqueeze(0),
            self.SFB: entry.weight_scale_factors.view(torch.float8_e4m3fn).unsqueeze(0),
            self.ALPHA: entry.alpha,
            self.BIAS: entry.module.bias.view(1, 1, self.n),
            self.Y: None,
        }
        resolved = resolve_variant_pack(variant_pack, self.compiled.binding)
        expected = {id(tensor) for tensor in self.compiled.bound}
        if set(resolved) != expected or resolved.get(id(self.Y)) is not None:
            raise RuntimeError(f"incomplete prepared NVFP4 binding for {entry.qualified_name}")
        resolved.pop(id(self.Y))
        resolved_refs = dict(resolved)
        self._prepared[id(entry)] = _PreparedNvfp4Binding(
            entry=entry,
            resolved=resolved,
            resolved_refs=resolved_refs,
            resolved_signatures={tensor_id: _tensor_signature(tensor) for tensor_id, tensor in resolved_refs.items()},
            activation_packed=packed,
            activation_scale_factors=scale_factors,
            activation_global_scale=entry.activation_global_scale,
            packed_weight=entry.packed_weight,
            weight_scale_factors=entry.weight_scale_factors,
            alpha=entry.alpha,
            bias=entry.module.bias,
            activation_packed_signature=_tensor_signature(packed),
            activation_scale_signature=_tensor_signature(scale_factors),
            activation_global_scale_signature=_tensor_signature(entry.activation_global_scale),
            packed_weight_signature=_tensor_signature(entry.packed_weight),
            weight_scale_signature=_tensor_signature(entry.weight_scale_factors),
            alpha_signature=_tensor_signature(entry.alpha),
            bias_signature=_tensor_signature(entry.module.bias),
            output_id=id(self.Y),
        )

    def validate_prepared(self, activation_buffers, entry):
        prepared = self._prepared.get(id(entry))
        if prepared is None or prepared.entry is not entry:
            raise RuntimeError(f"missing prepared NVFP4 binding for {entry.qualified_name}")
        packed, scale_factors = activation_buffers
        _validate_cached_tensor(
            f"{entry.qualified_name} A",
            packed,
            prepared.activation_packed,
            prepared.activation_packed_signature,
        )
        _validate_cached_tensor(
            f"{entry.qualified_name} SFA",
            scale_factors,
            prepared.activation_scale_factors,
            prepared.activation_scale_signature,
        )
        _validate_cached_tensor(
            f"{entry.qualified_name} activation global scale",
            entry.activation_global_scale,
            prepared.activation_global_scale,
            prepared.activation_global_scale_signature,
        )
        _validate_cached_tensor(
            f"{entry.qualified_name} B",
            entry.packed_weight,
            prepared.packed_weight,
            prepared.packed_weight_signature,
        )
        _validate_cached_tensor(
            f"{entry.qualified_name} SFB",
            entry.weight_scale_factors,
            prepared.weight_scale_factors,
            prepared.weight_scale_signature,
        )
        _validate_cached_tensor(
            f"{entry.qualified_name} alpha",
            entry.alpha,
            prepared.alpha,
            prepared.alpha_signature,
        )
        _validate_cached_tensor(
            f"{entry.qualified_name} bias",
            entry.module.bias,
            prepared.bias,
            prepared.bias_signature,
        )
        if prepared.output_id in prepared.resolved:
            raise RuntimeError(f"{entry.qualified_name} prepared output slot retained a tensor")
        _validate_resolved_cache(prepared, entry.qualified_name)

    def _binding(self, activation, entry, alpha, bias):
        prepared = self._prepared.get(id(entry))
        if prepared is None or prepared.entry is not entry:
            raise RuntimeError(f"missing prepared NVFP4 binding for {entry.qualified_name}")
        if (
            activation.packed is not prepared.activation_packed
            or activation.scale_factors is not prepared.activation_scale_factors
            or activation.global_scale is not prepared.activation_global_scale
        ):
            raise RuntimeError(f"{entry.qualified_name} activation buffers changed after NVFP4 preparation")
        if (
            entry.packed_weight is not prepared.packed_weight
            or entry.weight_scale_factors is not prepared.weight_scale_factors
            or alpha is not prepared.alpha
            or bias is not prepared.bias
        ):
            raise RuntimeError(f"{entry.qualified_name} weight-scale or bias binding changed after NVFP4 preparation")
        return prepared

    def __call__(self, activation, weight, alpha, bias):
        torch = self.torch
        prepared = self._binding(activation, weight, alpha, bias)
        output = torch.empty((1, self.m, self.n), dtype=torch.bfloat16, device=self.device)
        stream = torch.cuda.current_stream(self.device).cuda_stream
        _run_resolved_with_temporary_output(
            self.compiled,
            prepared.resolved,
            prepared.output_id,
            output,
            stream=stream,
        )
        return output.squeeze(0)

    def run_unprepared(self, activation, weight, alpha, bias):
        """Dynamic-buffer path used only by the untimed numerical contract gate."""
        torch = self.torch
        output = torch.empty((1, self.m, self.n), dtype=torch.bfloat16, device=self.device)
        stream = torch.cuda.current_stream(self.device).cuda_stream
        self.compiled(
            {
                self.A: activation.packed.view(torch.float4_e2m1fn_x2).unsqueeze(0),
                self.SFA: activation.scale_factors.view(torch.float8_e4m3fn).unsqueeze(0),
                self.B: weight.packed_weight.view(torch.float4_e2m1fn_x2).unsqueeze(0),
                self.SFB: weight.weight_scale_factors.view(torch.float8_e4m3fn).unsqueeze(0),
                self.ALPHA: alpha,
                self.BIAS: bias.view(1, 1, self.n),
                self.Y: output,
            },
            stream=stream,
        )
        return output.squeeze(0)


class _Nvfp4FusedFc1Plan:
    def __init__(self, torch, cudnn, build_gemm_plan, *, m, n, k, device):
        self.torch = torch
        self.m, self.n, self.k = m, n, k
        pm = _ceil_to(m, 128)
        sf_k = k // _BLOCK_SIZE
        sn = _ceil_to(n // _BLOCK_SIZE, 4)
        fp4, fp8 = cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E4M3
        reorder = dict(reordering_type=cudnn.tensor_reordering.F8_128x4)
        with torch.cuda.device(device):
            graph = cudnn.pygraph(
                io_data_type=cudnn.data_type.BFLOAT16,
                intermediate_data_type=cudnn.data_type.FLOAT,
                compute_data_type=cudnn.data_type.FLOAT,
            )
            self.A = graph.tensor(name="A", dim=[1, m, k], stride=[m * k, k, 1], data_type=fp4)
            self.SFA = graph.tensor(
                name="SFA",
                dim=[1, m, sf_k],
                stride=[m * sf_k, sf_k, 1],
                data_type=fp8,
                **reorder,
            )
            self.B = graph.tensor(name="B", dim=[1, k, n], stride=[k * n, 1, k], data_type=fp4)
            self.SFB = graph.tensor(
                name="SFB",
                dim=[1, sf_k, n],
                stride=[sf_k * n, 1, sf_k],
                data_type=fp8,
                **reorder,
            )
            self.ALPHA = graph.tensor(
                name="ALPHA",
                dim=[1, 1, 1],
                stride=[1, 1, 1],
                data_type=cudnn.data_type.FLOAT,
            )
            self.BIAS = graph.tensor(
                name="BIAS",
                dim=[1, 1, n],
                stride=[n, n, 1],
                data_type=cudnn.data_type.BFLOAT16,
            )
            self.HIDDEN_GLOBAL_SCALE = graph.tensor(
                name="HIDDEN_GLOBAL_SCALE",
                dim=[1, 1, 1],
                stride=[1, 1, 1],
                data_type=cudnn.data_type.FLOAT,
            )
            ad = graph.block_scale_dequantize(input=self.A, descale=self.SFA, block_size=[1, _BLOCK_SIZE])
            bd = graph.block_scale_dequantize(input=self.B, descale=self.SFB, block_size=[_BLOCK_SIZE, 1])
            acc = graph.matmul(A=ad, B=bd, name="fc1")
            corrected = graph.mul(a=acc, b=self.ALPHA, name="modelopt_alpha")
            corrected.set_data_type(cudnn.data_type.BFLOAT16)
            pre = graph.bias(input=corrected, bias=self.BIAS, name="bias")
            pre.set_data_type(cudnn.data_type.BFLOAT16)
            hidden = graph.gelu_approx_tanh(input=pre, name="gelu_tanh")
            # Diffusers' eager GELU returns BF16; FC2 observes and quantizes that
            # value, not an unrounded FP32 epilogue intermediate.
            hidden.set_data_type(cudnn.data_type.BFLOAT16)
            hidden_scaled = graph.mul(
                a=hidden,
                b=self.HIDDEN_GLOBAL_SCALE,
                name="hidden_modelopt_global_scale",
            )
            self.QH, self.SH = graph.block_scale_quantize(
                input=hidden_scaled,
                block_size=_BLOCK_SIZE,
                axis=-1,
                name="hidden_nvfp4",
            )
            self.QH.set_output(True).set_data_type(fp4)
            self.SH.set_dim([1, pm, sn]).set_stride([pm * sn, sn, 1])
            self.SH.set_output(True).set_data_type(fp8).set_reordering_type(cudnn.tensor_reordering.F8_128x4)
            self.compiled = build_gemm_plan(graph)
            self.qh = torch.empty((m, n // 2), dtype=torch.uint8, device=device)
            self.sh = torch.empty((pm, sn), dtype=torch.uint8, device=device)
        self.device = device
        self._prepared = {}

    def prepare(self, resolve_variant_pack, activation_buffers, entry, hidden_global_scale):
        """Resolve stable input, weight, and fixed output taps for one FC1."""
        torch = self.torch
        packed, scale_factors = activation_buffers
        if id(entry) in self._prepared:
            raise RuntimeError(f"duplicate prepared NVFP4 binding for {entry.qualified_name}")
        hidden_global_scale_source = hidden_global_scale
        hidden_global_scale = hidden_global_scale_source.reshape(1, 1, 1)
        variant_pack = {
            self.A: packed.view(torch.float4_e2m1fn_x2).unsqueeze(0),
            self.SFA: scale_factors.view(torch.float8_e4m3fn).unsqueeze(0),
            self.B: entry.packed_weight.view(torch.float4_e2m1fn_x2).unsqueeze(0),
            self.SFB: entry.weight_scale_factors.view(torch.float8_e4m3fn).unsqueeze(0),
            self.ALPHA: entry.alpha,
            self.BIAS: entry.module.bias.view(1, 1, self.n),
            self.HIDDEN_GLOBAL_SCALE: hidden_global_scale,
            self.QH: self.qh.view(torch.int8).unsqueeze(0),
            self.SH: self.sh.view(torch.float8_e4m3fn).unsqueeze(0),
        }
        resolved = resolve_variant_pack(variant_pack, self.compiled.binding)
        expected = {id(tensor) for tensor in self.compiled.bound}
        if set(resolved) != expected:
            raise RuntimeError(f"incomplete prepared NVFP4 binding for {entry.qualified_name}")
        resolved_refs = dict(resolved)
        self._prepared[id(entry)] = _PreparedNvfp4Binding(
            entry=entry,
            resolved=resolved,
            resolved_refs=resolved_refs,
            resolved_signatures={tensor_id: _tensor_signature(tensor) for tensor_id, tensor in resolved_refs.items()},
            activation_packed=packed,
            activation_scale_factors=scale_factors,
            activation_global_scale=entry.activation_global_scale,
            packed_weight=entry.packed_weight,
            weight_scale_factors=entry.weight_scale_factors,
            alpha=entry.alpha,
            bias=entry.module.bias,
            activation_packed_signature=_tensor_signature(packed),
            activation_scale_signature=_tensor_signature(scale_factors),
            activation_global_scale_signature=_tensor_signature(entry.activation_global_scale),
            packed_weight_signature=_tensor_signature(entry.packed_weight),
            weight_scale_signature=_tensor_signature(entry.weight_scale_factors),
            alpha_signature=_tensor_signature(entry.alpha),
            bias_signature=_tensor_signature(entry.module.bias),
            hidden_global_scale=hidden_global_scale,
            hidden_global_scale_source=hidden_global_scale_source,
            hidden_global_scale_signature=_tensor_signature(hidden_global_scale),
            hidden_global_scale_source_signature=_tensor_signature(hidden_global_scale_source),
            output_packed=self.qh,
            output_scale_factors=self.sh,
            output_packed_signature=_tensor_signature(self.qh),
            output_scale_signature=_tensor_signature(self.sh),
        )

    def validate_prepared(self, activation_buffers, entry, hidden_global_scale_source):
        prepared = self._prepared.get(id(entry))
        if prepared is None or prepared.entry is not entry:
            raise RuntimeError(f"missing prepared NVFP4 binding for {entry.qualified_name}")
        packed, scale_factors = activation_buffers
        _validate_cached_tensor(
            f"{entry.qualified_name} A",
            packed,
            prepared.activation_packed,
            prepared.activation_packed_signature,
        )
        _validate_cached_tensor(
            f"{entry.qualified_name} SFA",
            scale_factors,
            prepared.activation_scale_factors,
            prepared.activation_scale_signature,
        )
        _validate_cached_tensor(
            f"{entry.qualified_name} activation global scale",
            entry.activation_global_scale,
            prepared.activation_global_scale,
            prepared.activation_global_scale_signature,
        )
        _validate_cached_tensor(
            f"{entry.qualified_name} B",
            entry.packed_weight,
            prepared.packed_weight,
            prepared.packed_weight_signature,
        )
        _validate_cached_tensor(
            f"{entry.qualified_name} SFB",
            entry.weight_scale_factors,
            prepared.weight_scale_factors,
            prepared.weight_scale_signature,
        )
        _validate_cached_tensor(
            f"{entry.qualified_name} alpha",
            entry.alpha,
            prepared.alpha,
            prepared.alpha_signature,
        )
        _validate_cached_tensor(
            f"{entry.qualified_name} bias",
            entry.module.bias,
            prepared.bias,
            prepared.bias_signature,
        )
        _validate_cached_tensor(
            f"{entry.qualified_name} HGS source",
            hidden_global_scale_source,
            prepared.hidden_global_scale_source,
            prepared.hidden_global_scale_source_signature,
        )
        if _tensor_signature(prepared.hidden_global_scale) != prepared.hidden_global_scale_signature:
            raise RuntimeError(f"{entry.qualified_name} HGS view changed after NVFP4 preparation")
        _validate_cached_tensor(
            f"{entry.qualified_name} QH",
            self.qh,
            prepared.output_packed,
            prepared.output_packed_signature,
        )
        _validate_cached_tensor(
            f"{entry.qualified_name} SH",
            self.sh,
            prepared.output_scale_factors,
            prepared.output_scale_signature,
        )
        _validate_resolved_cache(prepared, entry.qualified_name)

    def _binding(self, activation, entry, alpha, bias):
        prepared = self._prepared.get(id(entry))
        if prepared is None or prepared.entry is not entry:
            raise RuntimeError(f"missing prepared NVFP4 binding for {entry.qualified_name}")
        if (
            activation.packed is not prepared.activation_packed
            or activation.scale_factors is not prepared.activation_scale_factors
            or activation.global_scale is not prepared.activation_global_scale
        ):
            raise RuntimeError(f"{entry.qualified_name} activation buffers changed after NVFP4 preparation")
        if (
            entry.packed_weight is not prepared.packed_weight
            or entry.weight_scale_factors is not prepared.weight_scale_factors
            or alpha is not prepared.alpha
            or bias is not prepared.bias
            or self.qh is not prepared.output_packed
            or self.sh is not prepared.output_scale_factors
        ):
            raise RuntimeError(f"{entry.qualified_name} weight-scale or bias binding changed after NVFP4 preparation")
        return prepared

    def __call__(self, activation, weight, alpha, bias):
        prepared = self._binding(activation, weight, alpha, bias)
        stream = self.torch.cuda.current_stream(self.device).cuda_stream
        self.compiled.run_resolved(prepared.resolved, stream=stream)
        return _QuantizedActivation(
            self.qh,
            self.sh,
            prepared.hidden_global_scale_source,
            source=None,
            group="fused_hidden",
        )

    def run_unprepared(self, activation, weight, alpha, bias, hidden_global_scale):
        """Dynamic-buffer path used only by the untimed numerical contract gate."""
        torch = self.torch
        stream = torch.cuda.current_stream(self.device).cuda_stream
        self.compiled(
            {
                self.A: activation.packed.view(torch.float4_e2m1fn_x2).unsqueeze(0),
                self.SFA: activation.scale_factors.view(torch.float8_e4m3fn).unsqueeze(0),
                self.B: weight.packed_weight.view(torch.float4_e2m1fn_x2).unsqueeze(0),
                self.SFB: weight.weight_scale_factors.view(torch.float8_e4m3fn).unsqueeze(0),
                self.ALPHA: alpha,
                self.BIAS: bias.view(1, 1, self.n),
                self.HIDDEN_GLOBAL_SCALE: hidden_global_scale,
                # FROST's generated host ABI uses Int8 as the byte carrier for
                # packed FP4 output taps (the logical graph dtype remains E2M1).
                self.QH: self.qh.view(torch.int8).unsqueeze(0),
                self.SH: self.sh.view(torch.float8_e4m3fn).unsqueeze(0),
            },
            stream=stream,
        )
        return _QuantizedActivation(self.qh, self.sh, hidden_global_scale, source=None, group="fused_hidden")


@dataclass
class _QuantizedActivation:
    packed: object
    scale_factors: object
    global_scale: object
    source: object
    group: str


class QwenImageModelOptNvfp4Adapter:
    """Installed, fail-closed dispatcher for the three benchmark arms."""

    def __init__(self, model, shape, calibration):
        import torch
        import cudnn
        import cudnn.gemm.frost  # noqa: F401 -- installs graph recording hooks
        from cudnn.gemm.frost.graph_analyzer import (
            build_gemm_plan,
            resolve_variant_pack,
        )
        from cudnn.gemm.ops._nvfp4_quantize import nvfp4_quantize
        from cudnn.gemm.ops import gelu_mlp

        self.torch = torch
        self.cudnn = cudnn
        self.nvfp4_quantize = nvfp4_quantize
        self.gelu_mlp = gelu_mlp
        self.shape = dict(shape)
        self.layers = shape["layers"]
        self.entries = _collect_entries(model, shape, torch)
        self.by_name = {entry.qualified_name: entry for entry in self.entries}
        if set(calibration) != {"amax", "calls", "metadata"} or set(calibration["amax"]) != set(self.by_name):
            raise ValueError("calibration does not match the exact Qwen-Image Linear role set")
        self.calibration_metadata = copy.deepcopy(calibration["metadata"])
        self.counters = {
            "bf16_linear_calls": 0,
            "nvfp4_linear_calls": 0,
            "activation_quant_logical": 0,
            "activation_quant_physical": 0,
            "activation_quant_standalone": 0,
            "activation_quant_fused": 0,
            "activation_cache_hits": 0,
            "weight_pack_calls": 0,
            "plan_build_calls": 0,
            "fallback_calls": 0,
            "forward_scopes": 0,
            "mlp_calls": {"torch": 0, "cudnn_bf16": 0, "nvfp4": 0},
            "nvfp4_linear_by_role": {entry.qualified_name: 0 for entry in self.entries},
        }
        self._selected = None
        self._active = False
        self._active_role_order = []
        self._activation_cache = {}
        self._installed_generic = {}
        self._installed_mod = {}
        self._installed_mlp = {}
        self._activation_buffers = {}
        self._linear_plans = {}
        self._fused_fc1_plans = {}

        group_scale = {}
        device = self.entries[0].module.weight.device
        self._device = device
        self._stream = torch.cuda.current_stream(device).cuda_stream
        for entry in self.entries:
            entry.activation_amax = calibration["amax"][entry.qualified_name]
            scale = (_MODELOPT_FP8_MAX * _NVFP4_E2M1_MAX / entry.activation_amax).reshape(1).contiguous()
            existing = group_scale.get(entry.activation_group)
            if existing is None:
                group_scale[entry.activation_group] = scale
            elif float(existing.item()) != float(scale.item()):
                raise RuntimeError(f"activation group {entry.activation_group} has unequal frozen global scales")
            entry.activation_global_scale = group_scale[entry.activation_group]

            entry.weight_amax = entry.module.weight.detach().abs().amax().float()
            if not bool(torch.isfinite(entry.weight_amax)) or not bool(entry.weight_amax > 0):
                raise RuntimeError(f"invalid weight amax for {entry.qualified_name}: {entry.weight_amax}")
            entry.weight_global_scale = (_MODELOPT_FP8_MAX * _NVFP4_E2M1_MAX / entry.weight_amax).reshape(1).contiguous()
            entry.alpha = (1.0 / (entry.activation_global_scale * entry.weight_global_scale)).reshape(1, 1, 1).contiguous()
            entry.packed_weight, entry.weight_scale_factors = nvfp4_quantize(
                entry.module.weight.detach(),
                entry.weight_global_scale,
                pre_quant_scale=None,
                enable_pdl=True,
            )
            self._validate_quantized_buffers(
                entry.packed_weight,
                entry.weight_scale_factors,
                entry.n,
                entry.k,
                f"{entry.qualified_name} weight",
            )
            self.counters["weight_pack_calls"] += 1

        for entry in self.entries:
            key = (entry.m, entry.k)
            if key not in self._activation_buffers:
                self._activation_buffers[key] = (
                    torch.empty((entry.m, entry.k // 2), dtype=torch.uint8, device=device),
                    torch.empty(
                        (_ceil_to(entry.m, 128), _ceil_to(entry.k // _BLOCK_SIZE, 4)),
                        dtype=torch.uint8,
                        device=device,
                    ),
                )

        linear_shapes = {(entry.m, entry.n, entry.k) for entry in self.entries if entry.role not in MLP_ROLES or entry.role.endswith("net.2")}
        fused_shapes = {(entry.m, entry.n, entry.k) for entry in self.entries if entry.role.endswith("net.0.proj")}
        with torch.cuda.device(device):
            for m, n, k in sorted(linear_shapes):
                key = (device.index, self._stream, m, n, k, "linear_bias")
                self._linear_plans[key] = _Nvfp4LinearPlan(torch, cudnn, build_gemm_plan, m=m, n=n, k=k, device=device)
                self.counters["plan_build_calls"] += 1
            for m, n, k in sorted(fused_shapes):
                key = (device.index, self._stream, m, n, k, "linear_bias_gelu_nvfp4")
                self._fused_fc1_plans[key] = _Nvfp4FusedFc1Plan(torch, cudnn, build_gemm_plan, m=m, n=n, k=k, device=device)
                self.counters["plan_build_calls"] += 1
        actual_contracts = {(m, n, k, epilogue) for _, _, m, n, k, epilogue in (*self._linear_plans, *self._fused_fc1_plans)}
        if actual_contracts != set(expected_plan_contracts(shape)):
            raise RuntimeError(f"NVFP4 plan contracts changed: got={sorted(actual_contracts)}, expected={sorted(expected_plan_contracts(shape))}")
        if self.counters["weight_pack_calls"] != 14 * self.layers or self.counters["plan_build_calls"] != 7:
            raise RuntimeError(f"unexpected setup counts: {self.counters}")

        # Resolve stable typed views and graph bindings once per logical Linear.
        # Plans remain shared by shape, but weights/biases are per-entry.  FC2's
        # activation storage is the fixed output tap of its matching FC1 plan.
        for entry in self.entries:
            if entry.role.endswith("net.0.proj"):
                second_name = entry.qualified_name.removesuffix("net.0.proj") + "net.2"
                second = self.by_name.get(second_name)
                if second is None:
                    raise RuntimeError(f"cannot prepare fused FC1 without {second_name}")
                self._fused_fc1_plan(entry).prepare(
                    resolve_variant_pack,
                    self._activation_buffers[(entry.m, entry.k)],
                    entry,
                    second.activation_global_scale,
                )
            elif entry.role.endswith("net.2"):
                first_name = entry.qualified_name.removesuffix("net.2") + "net.0.proj"
                first = self.by_name.get(first_name)
                if first is None:
                    raise RuntimeError(f"cannot prepare FC2 without {first_name}")
                first_plan = self._fused_fc1_plan(first)
                self._linear_plan(entry).prepare(
                    resolve_variant_pack,
                    (first_plan.qh, first_plan.sh),
                    entry,
                )
            else:
                self._linear_plan(entry).prepare(
                    resolve_variant_pack,
                    self._activation_buffers[(entry.m, entry.k)],
                    entry,
                )
        prepared_linear = sum(len(plan._prepared) for plan in self._linear_plans.values())
        prepared_fused = sum(len(plan._prepared) for plan in self._fused_fc1_plans.values())
        if (prepared_linear, prepared_fused) != (12 * self.layers, 2 * self.layers):
            raise RuntimeError("unexpected prepared NVFP4 binding counts: " f"linear={prepared_linear}, fused_fc1={prepared_fused}")
        self.binding_mode = {
            "name": "pre_resolved_run_resolved",
            "scope": "per_linear_entry",
            "stable_typed_views_cached": True,
            "stable_binding_validation": "full reference/signature preflight in select('C'); hot path object identity only",
            "linear_output": "fresh allocation; resolved Y slot cleared in finally",
            "fused_fc1_outputs": "plan-owned fixed QH/SH consumed immediately by FC2 on the same guarded stream",
            "concurrency": "single-thread, non-reentrant, one prepared CUDA stream; mutation after select is unsupported",
            "private_lowered_call": False,
            "prepared_linear_entries": prepared_linear,
            "prepared_fused_fc1_entries": prepared_fused,
        }

        self._install_generic_linears()
        self._install_modulations(model)
        self._install_mlps(model)

    def _validate_quantized_buffers(self, packed, scale_factors, rows, k, label):
        torch = self.torch
        expected_packed = (rows, k // 2)
        expected_scale = (_ceil_to(rows, 128), _ceil_to(k // _BLOCK_SIZE, 4))
        if packed.dtype != torch.uint8 or not packed.is_contiguous() or tuple(packed.shape) != expected_packed:
            raise RuntimeError(f"{label} packed output changed: got {packed.dtype}/{tuple(packed.shape)}/{packed.stride()}, expected uint8/{expected_packed}")
        if scale_factors.dtype != torch.uint8 or not scale_factors.is_contiguous() or tuple(scale_factors.shape) != expected_scale:
            raise RuntimeError(
                f"{label} scale-factor output changed: got {scale_factors.dtype}/{tuple(scale_factors.shape)}/{scale_factors.stride()}, "
                f"expected uint8/{expected_scale}"
            )

    @staticmethod
    def _sample_indices(extent):
        return tuple(sorted({0, int(extent) // 2, int(extent) - 1}))

    def _dequantize_rows(self, packed, scale_factors, logical_rows, k, row_indices, label):
        """Decode selected rows without using a cuDNN/FROST layout helper."""
        torch = self.torch
        rows = torch.tensor(tuple(row_indices), dtype=torch.long, device=self._device)
        if packed.dtype != torch.uint8 or tuple(packed.shape) != (logical_rows, k // 2):
            raise RuntimeError(f"{label} packed reference input changed: {packed.dtype}/{tuple(packed.shape)}")
        expected_sf = (_ceil_to(logical_rows, 128), _ceil_to(k // _BLOCK_SIZE, 4))
        if scale_factors.dtype != torch.uint8 or tuple(scale_factors.shape) != expected_sf:
            raise RuntimeError(f"{label} scale reference input changed: {scale_factors.dtype}/{tuple(scale_factors.shape)}")

        selected = packed.index_select(0, rows)
        lut = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=self._device)
        low = lut[(selected & 0xF).long()]
        high = lut[(selected >> 4).long()]
        decoded = torch.stack((low, high), dim=-1).flatten(-2)

        sf_columns = k // _BLOCK_SIZE
        column_groups = _ceil_to(sf_columns, 4) // 4
        blocks = torch.arange(sf_columns, dtype=torch.long, device=self._device).unsqueeze(0)
        row_grid = rows.unsqueeze(1)
        addresses = ((row_grid // 128) * column_groups + (blocks // 4)) * 512 + (row_grid % 32) * 16 + ((row_grid % 128) // 32) * 4 + (blocks % 4)
        flat_scales = scale_factors.view(torch.float8_e4m3fn).flatten()
        if int(addresses.max().item()) >= flat_scales.numel():
            raise RuntimeError(f"{label} F8_128x4 reference address exceeds the physical blob")
        logical_scales = flat_scales[addresses].float()
        return decoded * logical_scales.repeat_interleave(_BLOCK_SIZE, dim=1)

    def _reference_linear_samples(self, activation, entry, row_indices, column_indices):
        torch = self.torch
        x = self._dequantize_rows(
            activation.packed,
            activation.scale_factors,
            entry.m,
            entry.k,
            row_indices,
            f"{entry.qualified_name} activation",
        )
        weight = self._dequantize_rows(
            entry.packed_weight,
            entry.weight_scale_factors,
            entry.n,
            entry.k,
            column_indices,
            f"{entry.qualified_name} weight",
        )
        corrected = (x @ weight.t()) * entry.alpha.float().reshape(())
        corrected = corrected.to(torch.bfloat16)
        columns = torch.tensor(tuple(column_indices), dtype=torch.long, device=self._device)
        return (corrected + entry.module.bias.index_select(0, columns)).to(torch.bfloat16)

    def _check_linear_samples(self, actual, expected, *, label):
        torch = self.torch
        actual_f, expected_f = actual.float(), expected.float()
        if not bool(torch.isfinite(actual_f).all()) or not bool(torch.isfinite(expected_f).all()):
            raise RuntimeError(f"{label} numerical reference contains a non-finite value")
        absolute = (actual_f - expected_f).abs()
        allowed = _LINEAR_REFERENCE_ATOL + _LINEAR_REFERENCE_RTOL * expected_f.abs()
        violations = int((absolute > allowed).sum().item())
        rel_l2 = float((actual_f - expected_f).norm() / expected_f.norm().clamp_min(1.0e-12))
        maximum = float(absolute.max())
        result = {
            "rel_l2": rel_l2,
            "max_abs": maximum,
            "rtol": _LINEAR_REFERENCE_RTOL,
            "atol": _LINEAR_REFERENCE_ATOL,
            "elements": actual.numel(),
            "violations": violations,
        }
        if violations:
            raise RuntimeError(f"{label} failed independent dequantized-operand reference: {result}")
        return result

    def _quantize_gate_input(self, entry, value, label):
        packed, scale_factors = self.nvfp4_quantize(
            value,
            entry.activation_global_scale,
            pre_quant_scale=None,
            enable_pdl=True,
        )
        self._validate_quantized_buffers(packed, scale_factors, entry.m, entry.k, label)
        return _QuantizedActivation(
            packed,
            scale_factors,
            entry.activation_global_scale,
            source=value,
            group=f"gate:{label}",
        )

    def _make_gate_input(self, entry, generator):
        torch = self.torch
        # Keep the deterministic probe near the frozen calibration range.  The
        # reference consumes the packed values, so clipping cannot mask a graph
        # layout/orientation/alpha defect.
        amplitude = max(float(entry.activation_amax.item()) / 8.0, 2.0**-12)
        return (
            torch.randn(
                (entry.m, entry.k),
                dtype=torch.bfloat16,
                device=self._device,
                generator=generator,
            )
            * amplitude
        ).contiguous()

    def _stage_prepared_gate_activation(self, plan, entry, activation):
        """Copy a gate probe into the exact buffers cached by one entry."""
        prepared = plan._prepared.get(id(entry))
        if prepared is None or prepared.entry is not entry:
            raise RuntimeError(f"numerical gate cannot resolve prepared binding for {entry.qualified_name}")
        if tuple(prepared.activation_packed.shape) != tuple(activation.packed.shape) or tuple(prepared.activation_scale_factors.shape) != tuple(
            activation.scale_factors.shape
        ):
            raise RuntimeError(f"numerical gate prepared buffers changed for {entry.qualified_name}")
        if float(prepared.activation_global_scale.item()) != float(activation.global_scale.item()):
            raise RuntimeError(f"numerical gate global scale changed for {entry.qualified_name}")
        prepared.activation_packed.copy_(activation.packed)
        prepared.activation_scale_factors.copy_(activation.scale_factors)
        return _QuantizedActivation(
            prepared.activation_packed,
            prepared.activation_scale_factors,
            prepared.activation_global_scale,
            source=None,
            group=f"prepared_gate:{entry.qualified_name}",
        )

    def _run_prepared_binding_parity(self, generator):
        """Exercise every per-entry resolved map against the dynamic gate path."""
        torch = self.torch
        direct_checked = 0
        fused_checked = 0
        sequential_fc2_checked = 0
        qkv_outputs = {}

        for entry in self.entries:
            value = self._make_gate_input(entry, generator)
            activation = self._quantize_gate_input(entry, value, f"prepared parity {entry.qualified_name}")
            if entry.role.endswith("net.0.proj"):
                second_name = entry.qualified_name.removesuffix("net.0.proj") + "net.2"
                second = self.by_name.get(second_name)
                if second is None:
                    raise RuntimeError(f"prepared parity cannot resolve FC2 for {entry.qualified_name}")
                plan = self._fused_fc1_plan(entry)
                hidden_scale = plan._prepared[id(entry)].hidden_global_scale
                dynamic_hidden = plan.run_unprepared(
                    activation,
                    entry,
                    entry.alpha,
                    entry.module.bias,
                    hidden_scale,
                )
                dynamic_packed = dynamic_hidden.packed.clone()
                dynamic_scales = dynamic_hidden.scale_factors.clone()
                prepared_activation = self._stage_prepared_gate_activation(plan, entry, activation)
                prepared_hidden = plan(
                    prepared_activation,
                    entry,
                    entry.alpha,
                    entry.module.bias,
                )
                if not torch.equal(dynamic_packed, prepared_hidden.packed) or not torch.equal(dynamic_scales, prepared_hidden.scale_factors):
                    raise RuntimeError(f"{entry.qualified_name} prepared fused QH/SH differs from dynamic binding")

                # The fixed QH/SH taps are benchmark-safe only because FC2
                # consumes them immediately on the same prepared stream.
                second_plan = self._linear_plan(second)
                dynamic_output = second_plan.run_unprepared(prepared_hidden, second, second.alpha, second.module.bias)
                prepared_output = second_plan(prepared_hidden, second, second.alpha, second.module.bias)
                if not torch.equal(dynamic_output, prepared_output):
                    raise RuntimeError(f"{entry.qualified_name} -> {second.qualified_name} prepared FC2 differs from dynamic binding")
                fused_checked += 1
                sequential_fc2_checked += 1
                continue

            plan = self._linear_plan(entry)
            dynamic_output = plan.run_unprepared(activation, entry, entry.alpha, entry.module.bias)
            prepared_activation = self._stage_prepared_gate_activation(plan, entry, activation)
            prepared_output = plan(prepared_activation, entry, entry.alpha, entry.module.bias)
            if not torch.equal(dynamic_output, prepared_output):
                raise RuntimeError(f"{entry.qualified_name} prepared output differs from dynamic binding")
            direct_checked += 1
            if entry.role in (
                "attn.to_q",
                "attn.to_k",
                "attn.to_v",
                "attn.add_q_proj",
                "attn.add_k_proj",
                "attn.add_v_proj",
            ):
                qkv_outputs.setdefault(entry.activation_group, []).append(prepared_output)

        for group, outputs in qkv_outputs.items():
            pointers = [output.data_ptr() for output in outputs]
            if len(outputs) != 3 or len(set(pointers)) != 3:
                raise RuntimeError(f"{group} prepared Q/K/V outputs do not have three independent Y allocations: {pointers}")
        if (
            direct_checked != 12 * self.layers
            or fused_checked != 2 * self.layers
            or sequential_fc2_checked != 2 * self.layers
            or len(qkv_outputs) != 2 * self.layers
        ):
            raise RuntimeError(
                "prepared parity coverage changed: "
                f"direct={direct_checked}, fused={fused_checked}, "
                f"sequential_fc2={sequential_fc2_checked}, qkv_groups={len(qkv_outputs)}"
            )
        return {
            "status": "passed",
            "comparison": "bitwise prepared run_resolved vs dynamic variant-pack binding",
            "logical_entries_checked": direct_checked + fused_checked,
            "direct_entries_checked": direct_checked,
            "fused_fc1_entries_checked": fused_checked,
            "sequential_fc1_fc2_pairs_checked": sequential_fc2_checked,
            "qkv_groups_with_three_live_distinct_outputs": len(qkv_outputs),
            "fused_output_lifetime": "plan-owned QH/SH consumed immediately by FC2 on the same guarded stream",
        }

    def _reference_fused_hidden(self, activation, entry, row_indices):
        torch = self.torch
        x = self._dequantize_rows(
            activation.packed,
            activation.scale_factors,
            entry.m,
            entry.k,
            row_indices,
            f"{entry.qualified_name} activation",
        )
        chunks = []
        # Chunk the independent decode so the image-token contract does not
        # materialize the full 12288x3072 weight as FP32 at once.
        for begin in range(0, entry.n, 1024):
            end = min(begin + 1024, entry.n)
            columns = tuple(range(begin, end))
            weight = self._dequantize_rows(
                entry.packed_weight,
                entry.weight_scale_factors,
                entry.n,
                entry.k,
                columns,
                f"{entry.qualified_name} weight",
            )
            corrected = ((x @ weight.t()) * entry.alpha.float().reshape(())).to(torch.bfloat16)
            pre = (corrected + entry.module.bias[begin:end]).to(torch.bfloat16)
            chunks.append(torch.nn.functional.gelu(pre, approximate="tanh").to(torch.bfloat16))
        return torch.cat(chunks, dim=1)

    @staticmethod
    def _relative_l2(torch, actual, expected):
        return float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1.0e-12))

    def run_focused_numerical_gate(self):
        """Validate all seven plans once, outside route counts and timed work."""
        torch = self.torch
        if self._active:
            raise RuntimeError("cannot run the NVFP4 numerical gate during a model forward")
        self._validate_prepared_bindings()
        before = self.snapshot()
        expected_contracts = expected_plan_contracts(self.shape)
        generator = torch.Generator(device=self._device).manual_seed(0x4E56465034)
        results = []

        def representative(m, n, k, *, fused):
            for candidate in self.entries:
                candidate_fused = candidate.role.endswith("net.0.proj")
                candidate_linear = candidate.role not in MLP_ROLES or candidate.role.endswith("net.2")
                if (candidate_fused if fused else candidate_linear) and (
                    candidate.m,
                    candidate.n,
                    candidate.k,
                ) == (m, n, k):
                    return candidate
            raise RuntimeError(f"no representative Linear for numerical contract {(m, n, k, fused)}")

        with torch.inference_mode(), torch.cuda.device(self._device):
            prepared_parity = self._run_prepared_binding_parity(generator)
            for m, n, k, epilogue in expected_contracts:
                fused = epilogue == "linear_bias_gelu_nvfp4"
                first = representative(m, n, k, fused=fused)
                value = self._make_gate_input(first, generator)
                activation = self._quantize_gate_input(first, value, f"gate {first.qualified_name}")
                row_indices = self._sample_indices(m)

                if not fused:
                    column_indices = self._sample_indices(n)
                    actual_full = self._linear_plan(first).run_unprepared(activation, first, first.alpha, first.module.bias)
                    rows = torch.tensor(row_indices, dtype=torch.long, device=self._device)
                    columns = torch.tensor(column_indices, dtype=torch.long, device=self._device)
                    actual = actual_full.index_select(0, rows).index_select(1, columns)
                    expected = self._reference_linear_samples(activation, first, row_indices, column_indices)
                    results.append(
                        {
                            "contract": {"m": m, "n": n, "k": k, "epilogue": epilogue},
                            "representative_role": first.qualified_name,
                            "sample_rows": list(row_indices),
                            "sample_columns": list(column_indices),
                            "output": self._check_linear_samples(actual, expected, label=f"gate {first.qualified_name}"),
                        }
                    )
                    continue

                second_name = first.qualified_name.removesuffix("net.0.proj") + "net.2"
                second = self.by_name.get(second_name)
                if second is None or second.m != first.m or second.k != first.n:
                    raise RuntimeError(f"fused numerical gate cannot resolve FC2 for {first.qualified_name}")
                hidden = self._fused_fc1_plan(first).run_unprepared(
                    activation,
                    first,
                    first.alpha,
                    first.module.bias,
                    second.activation_global_scale.reshape(1, 1, 1),
                )
                expected_hidden = self._reference_fused_hidden(activation, first, row_indices)
                fused_hidden_scaled = self._dequantize_rows(
                    hidden.packed,
                    hidden.scale_factors,
                    first.m,
                    first.n,
                    row_indices,
                    f"gate {first.qualified_name} fused hidden",
                )
                hidden_scale = second.activation_global_scale.float().reshape(())
                hidden_rel_l2 = self._relative_l2(torch, fused_hidden_scaled / hidden_scale, expected_hidden)

                oracle_packed, oracle_scale = self.nvfp4_quantize(
                    expected_hidden.contiguous(),
                    second.activation_global_scale,
                    pre_quant_scale=None,
                    enable_pdl=True,
                )
                oracle_scaled = self._dequantize_rows(
                    oracle_packed,
                    oracle_scale,
                    len(row_indices),
                    first.n,
                    tuple(range(len(row_indices))),
                    f"gate {first.qualified_name} standalone hidden oracle",
                )
                oracle_rel_l2 = self._relative_l2(torch, fused_hidden_scaled, oracle_scaled)
                hidden_metrics = {
                    "rel_l2_vs_bf16_gelu_reference": hidden_rel_l2,
                    "limit_vs_bf16_gelu_reference": _FUSED_HIDDEN_REFERENCE_REL_L2,
                    "rel_l2_vs_standalone_quant_oracle": oracle_rel_l2,
                    "limit_vs_standalone_quant_oracle": _FUSED_HIDDEN_ORACLE_REL_L2,
                }
                if not all(math_value == math_value and abs(math_value) != float("inf") for math_value in (hidden_rel_l2, oracle_rel_l2)):
                    raise RuntimeError(f"gate {first.qualified_name} fused hidden produced non-finite metrics: {hidden_metrics}")
                if hidden_rel_l2 > _FUSED_HIDDEN_REFERENCE_REL_L2 or oracle_rel_l2 > _FUSED_HIDDEN_ORACLE_REL_L2:
                    raise RuntimeError(f"gate {first.qualified_name} fused hidden failed reference: {hidden_metrics}")

                column_indices = self._sample_indices(second.n)
                actual_full = self._linear_plan(second).run_unprepared(hidden, second, second.alpha, second.module.bias)
                rows = torch.tensor(row_indices, dtype=torch.long, device=self._device)
                columns = torch.tensor(column_indices, dtype=torch.long, device=self._device)
                actual = actual_full.index_select(0, rows).index_select(1, columns)
                expected = self._reference_linear_samples(hidden, second, row_indices, column_indices)
                results.append(
                    {
                        "contract": {"m": m, "n": n, "k": k, "epilogue": epilogue},
                        "representative_role": first.qualified_name,
                        "sample_rows": list(row_indices),
                        "sample_columns": list(column_indices),
                        "fused_hidden": hidden_metrics,
                        "fc2_output": self._check_linear_samples(
                            actual,
                            expected,
                            label=f"gate {first.qualified_name} -> {second.qualified_name}",
                        ),
                    }
                )
            torch.cuda.synchronize(self._device)

        if self.snapshot() != before:
            raise RuntimeError(f"focused numerical gate contaminated route counters: before={before}, after={self.snapshot()}")
        if len(results) != 7 or [tuple((*item["contract"].values(),)) for item in results] != list(expected_contracts):
            raise RuntimeError(f"focused numerical gate did not cover the exact seven contracts: {results}")
        return {
            "status": "passed",
            "contracts_checked": len(results),
            "prepared_binding": prepared_parity,
            "reference": "independent E2M1 decode plus F8_128x4 address map and sampled FP32 matmul; fused hidden also checked against BF16 tanh-GELU and standalone quantization",
            "sampling": "first/middle/last rows and output columns; every full FROST plan still executes its exact M/N/K",
            "results": results,
        }

    def plan_provenance(self):
        records = []
        for key, plan in sorted(
            (*self._linear_plans.items(), *self._fused_fc1_plans.items()),
            key=lambda item: item[0][2:],
        ):
            device, stream, m, n, k, epilogue = key
            records.append(
                {
                    "device": device,
                    "stream": stream,
                    "m": m,
                    "n": n,
                    "k": k,
                    "epilogue": epilogue,
                    "tile_config": plan.compiled.config.name,
                    "generated_path": str(plan.compiled.generated_path.resolve()),
                }
            )
        if len(records) != 7:
            raise RuntimeError(f"expected seven low-precision plans, got {records}")
        return records

    def _install_generic_linears(self):
        for entry in self.entries:
            if entry.role in MLP_ROLES or entry.role in ("img_mod.1", "txt_mod.1"):
                continue

            def bf16_forward(_module, hidden_states, *args, _entry=entry, **kwargs):
                self._validate_call(_entry, hidden_states, args, kwargs)
                self.counters["bf16_linear_calls"] += 1
                return _entry.original_forward(hidden_states)

            def nvfp4_forward(_module, hidden_states, *args, _entry=entry, **kwargs):
                self._validate_call(_entry, hidden_states, args, kwargs)
                activation = self._quantize_activation(_entry, hidden_states)
                self._record_nvfp4_role(_entry)
                output = self._linear_plan(_entry)(activation, _entry, _entry.alpha, _entry.module.bias)
                return output.reshape(*hidden_states.shape[:-1], _entry.n)

            self._installed_generic[entry.module] = {
                "original": entry.original_forward,
                "bf16": types.MethodType(bf16_forward, entry.module),
                "nvfp4": types.MethodType(nvfp4_forward, entry.module),
            }

    def _install_modulations(self, model):
        """Own the raw-temb boundary so shared quantization is identity-guarded.

        The Linear inputs themselves are distinct outputs of two identical SiLU
        modules, so comparing their Tensor identities would reject valid sharing.
        The pinned block forward does pass the exact same raw ``temb`` object to
        every img/txt modulation Sequential.  Intercepting that boundary both
        proves the sharing key and lets C compute SiLU only for the first use.
        """
        torch = self.torch
        for block_index, block in enumerate(model.transformer_blocks):
            for stream in ("img", "txt"):
                module = getattr(block, f"{stream}_mod")
                name = f"transformer_blocks.{block_index}.{stream}_mod"
                entry = self.by_name[f"{name}.1"]
                if type(module) is not torch.nn.Sequential or len(module) != 2:
                    raise TypeError(f"{name} must be the pinned two-node Sequential")
                if type(module[0]) is not torch.nn.SiLU or module[0].inplace or module[1] is not entry.module:
                    raise TypeError(f"{name} must be non-inplace SiLU followed by its exact Linear")
                if "forward" in module.__dict__:
                    raise NotImplementedError(f"{name} already has an instance-level forward override")
                original = module.forward

                def bf16_forward(
                    _module,
                    temb,
                    *args,
                    _name=name,
                    _entry=entry,
                    _original=original,
                    **kwargs,
                ):
                    self._validate_mod_call(_name, _entry, temb, args, kwargs)
                    self.counters["bf16_linear_calls"] += 1
                    return _original(temb)

                def nvfp4_forward(
                    _module,
                    temb,
                    *args,
                    _name=name,
                    _entry=entry,
                    _silu=module[0],
                    **kwargs,
                ):
                    self._validate_mod_call(_name, _entry, temb, args, kwargs)
                    activation = self._quantize_modulation(_entry, temb, _silu)
                    self._record_nvfp4_role(_entry)
                    output = self._linear_plan(_entry)(activation, _entry, _entry.alpha, _entry.module.bias)
                    return output.reshape(*temb.shape[:-1], _entry.n)

                self._installed_mod[module] = {
                    "original": original,
                    "bf16": types.MethodType(bf16_forward, module),
                    "nvfp4": types.MethodType(nvfp4_forward, module),
                }

    def _install_mlps(self, model):
        torch = self.torch
        for block_index, block in enumerate(model.transformer_blocks):
            for stream in ("img", "txt"):
                module = getattr(block, f"{stream}_mlp")
                name = f"transformer_blocks.{block_index}.{stream}_mlp"
                if module.__class__.__module__ != "diffusers.models.attention" or module.__class__.__name__ != "FeedForward":
                    raise TypeError(f"{name} is not the pinned Diffusers FeedForward: {type(module)!r}")
                if "forward" in module.__dict__:
                    raise NotImplementedError(f"{name} already has an instance-level forward override")
                first = self.by_name[f"transformer_blocks.{block_index}.{stream}_mlp.net.0.proj"]
                second = self.by_name[f"transformer_blocks.{block_index}.{stream}_mlp.net.2"]
                net = getattr(module, "net", None)
                if not isinstance(net, torch.nn.ModuleList) or len(net) != 3:
                    raise TypeError(f"{name}.net must contain exactly GELU, Dropout, and output projection")
                activation, dropout, output = net
                if (
                    activation.__class__.__module__ != "diffusers.models.activations"
                    or activation.__class__.__name__ != "GELU"
                    or getattr(activation, "approximate", None) != "tanh"
                    or getattr(activation, "proj", None) is not first.module
                    or type(dropout) is not torch.nn.Dropout
                    or dropout.p != 0.0
                    or dropout.inplace
                    or output is not second.module
                ):
                    raise TypeError(f"{name} must be the pinned Linear -> GELU(tanh) -> non-inplace Dropout(0) -> Linear")
                original = module.forward

                def torch_forward(
                    _module,
                    hidden_states,
                    *args,
                    _name=name,
                    _original=original,
                    **kwargs,
                ):
                    self._validate_mlp_call(_name, hidden_states, args, kwargs)
                    self.counters["mlp_calls"]["torch"] += 1
                    self.counters["bf16_linear_calls"] += 2
                    return _original(hidden_states)

                def cudnn_forward(
                    _module,
                    hidden_states,
                    *args,
                    _name=name,
                    _first=first,
                    _second=second,
                    **kwargs,
                ):
                    self._validate_mlp_call(_name, hidden_states, args, kwargs)
                    self.counters["mlp_calls"]["cudnn_bf16"] += 1
                    self.counters["bf16_linear_calls"] += 2
                    return self.gelu_mlp(
                        hidden_states,
                        _first.module.weight,
                        _first.module.bias,
                        _second.module.weight,
                        _second.module.bias,
                    )

                def nvfp4_forward(
                    _module,
                    hidden_states,
                    *args,
                    _name=name,
                    _first=first,
                    _second=second,
                    **kwargs,
                ):
                    self._validate_mlp_call(_name, hidden_states, args, kwargs)
                    self.counters["mlp_calls"]["nvfp4"] += 1
                    activation = self._quantize_activation(_first, hidden_states)
                    self._record_nvfp4_role(_first)
                    hidden = self._fused_fc1_plan(_first)(
                        activation,
                        _first,
                        _first.alpha,
                        _first.module.bias,
                    )
                    self.counters["activation_quant_logical"] += 1
                    self.counters["activation_quant_physical"] += 1
                    self.counters["activation_quant_fused"] += 1
                    self._record_nvfp4_role(_second)
                    output = self._linear_plan(_second)(hidden, _second, _second.alpha, _second.module.bias)
                    return output.reshape(*hidden_states.shape[:-1], _second.n)

                self._installed_mlp[module] = {
                    "original": original,
                    "torch": types.MethodType(torch_forward, module),
                    "cudnn_bf16": types.MethodType(cudnn_forward, module),
                    "nvfp4": types.MethodType(nvfp4_forward, module),
                }

    def _validate_call(self, entry, hidden_states, args, kwargs):
        if not self._active:
            raise RuntimeError(f"{entry.qualified_name} called outside adapter.forward_scope()")
        if args or kwargs:
            raise NotImplementedError(f"{entry.qualified_name} accepts only hidden_states in this benchmark")
        if tuple(hidden_states.shape) != entry.input_shape:
            raise ValueError(f"{entry.qualified_name} expected {entry.input_shape}, got {tuple(hidden_states.shape)}")
        if hidden_states.dtype != self.torch.bfloat16 or hidden_states.device.type != "cuda" or not hidden_states.is_contiguous():
            raise NotImplementedError(f"{entry.qualified_name} requires contiguous bf16 CUDA activation")
        self._validate_stream(hidden_states.device)

    def _validate_mlp_call(self, name, hidden_states, args, kwargs):
        if not self._active:
            raise RuntimeError(f"{name} called outside adapter.forward_scope()")
        if args or kwargs:
            raise NotImplementedError(f"{name} accepts only hidden_states in this benchmark")
        if hidden_states.dtype != self.torch.bfloat16 or hidden_states.device.type != "cuda" or not hidden_states.is_contiguous():
            raise NotImplementedError(f"{name} requires contiguous bf16 CUDA activation")
        self._validate_stream(hidden_states.device)

    def _validate_mod_call(self, name, entry, temb, args, kwargs):
        if not self._active:
            raise RuntimeError(f"{name} called outside adapter.forward_scope()")
        if args or kwargs:
            raise NotImplementedError(f"{name} accepts only temb in this benchmark")
        if tuple(temb.shape) != entry.input_shape:
            raise ValueError(f"{name} expected raw temb shape {entry.input_shape}, got {tuple(temb.shape)}")
        if temb.dtype != self.torch.bfloat16 or temb.device.type != "cuda" or not temb.is_contiguous():
            raise NotImplementedError(f"{name} requires contiguous bf16 CUDA temb")
        self._validate_stream(temb.device)

    def _validate_stream(self, device):
        current = self.torch.cuda.current_stream(device).cuda_stream
        if device != self._device or current != self._stream:
            raise NotImplementedError(
                "the benchmark-local NVFP4 adapter is prepared for exactly one (device, stream); "
                f"prepared={(self._device, self._stream)}, current={(device, current)}"
            )

    def _linear_plan(self, entry):
        return self._linear_plans[(self._device.index, self._stream, entry.m, entry.n, entry.k, "linear_bias")]

    def _fused_fc1_plan(self, entry):
        return self._fused_fc1_plans[
            (
                self._device.index,
                self._stream,
                entry.m,
                entry.n,
                entry.k,
                "linear_bias_gelu_nvfp4",
            )
        ]

    def _validate_prepared_bindings(self):
        """Full stable-binding preflight, called outside the timed forward."""
        self._validate_stream(self._device)
        for entry in self.entries:
            if entry.role.endswith("net.0.proj"):
                second_name = entry.qualified_name.removesuffix("net.0.proj") + "net.2"
                second = self.by_name.get(second_name)
                if second is None:
                    raise RuntimeError(f"cannot validate fused FC1 without {second_name}")
                self._fused_fc1_plan(entry).validate_prepared(
                    self._activation_buffers[(entry.m, entry.k)],
                    entry,
                    second.activation_global_scale,
                )
            elif entry.role.endswith("net.2"):
                first_name = entry.qualified_name.removesuffix("net.2") + "net.0.proj"
                first = self.by_name.get(first_name)
                if first is None:
                    raise RuntimeError(f"cannot validate FC2 without {first_name}")
                first_plan = self._fused_fc1_plan(first)
                self._linear_plan(entry).validate_prepared((first_plan.qh, first_plan.sh), entry)
            else:
                self._linear_plan(entry).validate_prepared(self._activation_buffers[(entry.m, entry.k)], entry)

    def _quantize_modulation(self, entry, temb, silu):
        key = (entry.m, entry.k)
        cached = self._activation_cache.get(key)
        if cached is not None and cached.group == entry.activation_group:
            # Unlike equal amax values, this proves every reuse came from the
            # exact raw conditioning Tensor in the pinned call graph.
            if cached.source is not temb:
                raise RuntimeError("Qwen-Image modulation stopped sharing the exact raw temb Tensor")
            if cached.global_scale is not entry.activation_global_scale:
                raise RuntimeError("Qwen-Image modulation reused a different frozen global-scale tensor")
            self.counters["activation_quant_logical"] += 1
            self.counters["activation_cache_hits"] += 1
            return cached
        activated = silu(temb)
        self._validate_call(entry, activated, (), {})
        return self._quantize_activation(entry, activated, cache_source=temb)

    def _quantize_activation(self, entry, hidden_states, *, cache_source=None):
        self.counters["activation_quant_logical"] += 1
        source = hidden_states if cache_source is None else cache_source
        key = (entry.m, entry.k)
        cached = self._activation_cache.get(key)
        if cached is not None and cached.group == entry.activation_group:
            if cached.global_scale is not entry.activation_global_scale:
                raise RuntimeError(f"{entry.activation_group} reused with a different frozen global-scale tensor")
            if cached.source is not source:
                raise RuntimeError(f"{entry.activation_group} inputs stopped sharing the exact Tensor object")
            self.counters["activation_cache_hits"] += 1
            return cached

        packed, scale_factors = self._activation_buffers[key]
        result_packed, result_scale = self.nvfp4_quantize(
            hidden_states.view(entry.m, entry.k),
            entry.activation_global_scale,
            pre_quant_scale=None,
            out=packed,
            scale_factors=scale_factors,
            enable_pdl=True,
        )
        if result_packed.data_ptr() != packed.data_ptr() or result_scale.data_ptr() != scale_factors.data_ptr():
            raise RuntimeError("nvfp4_quantize ignored the caller-provided activation output buffers")
        self._validate_quantized_buffers(
            result_packed,
            result_scale,
            entry.m,
            entry.k,
            f"{entry.qualified_name} activation",
        )
        result = _QuantizedActivation(
            packed,
            scale_factors,
            entry.activation_global_scale,
            source,
            entry.activation_group,
        )
        self._activation_cache[key] = result
        self.counters["activation_quant_physical"] += 1
        self.counters["activation_quant_standalone"] += 1
        return result

    def _record_nvfp4_role(self, entry):
        self.counters["nvfp4_linear_calls"] += 1
        self.counters["nvfp4_linear_by_role"][entry.qualified_name] += 1
        self._active_role_order.append(entry.qualified_name)

    def select(self, arm):
        if arm not in ARM_CONFIGS:
            raise ValueError(f"unknown ModelOpt NVFP4 benchmark arm {arm!r}")
        if self._active:
            raise RuntimeError("cannot switch treatments during a model forward")
        if arm == "C":
            self._validate_prepared_bindings()
        valid_generic = {value for forwards in self._installed_generic.values() for key, value in forwards.items() if key != "original"}
        valid_mod = {value for forwards in self._installed_mod.values() for key, value in forwards.items() if key != "original"}
        valid_mlp = {value for forwards in self._installed_mlp.values() for key, value in forwards.items() if key != "original"}
        if any(module.forward not in valid_generic and "forward" in module.__dict__ for module in self._installed_generic):
            raise RuntimeError("a Qwen-Image generic Linear was modified after adapter installation")
        if any(module.forward not in valid_mlp and "forward" in module.__dict__ for module in self._installed_mlp):
            raise RuntimeError("a Qwen-Image FeedForward was modified after adapter installation")
        if any(module.forward not in valid_mod and "forward" in module.__dict__ for module in self._installed_mod):
            raise RuntimeError("a Qwen-Image modulation Sequential was modified after adapter installation")
        config = ARM_CONFIGS[arm]
        for module, forwards in self._installed_generic.items():
            module.forward = forwards[config["generic_linear"]]
        for module, forwards in self._installed_mlp.items():
            module.forward = forwards[config["mlp"]]
        for module, forwards in self._installed_mod.items():
            module.forward = forwards[config["generic_linear"]]
        self._selected = arm

    @contextmanager
    def forward_scope(self, arm):
        if arm != self._selected:
            raise RuntimeError(f"forward scope arm {arm} does not match selected treatment {self._selected}")
        if self._active:
            raise RuntimeError("nested model forwards are outside the benchmark treatment")
        before = self.snapshot()
        self._active = True
        self._active_role_order = []
        self._activation_cache = {}
        self.counters["forward_scopes"] += 1
        completed = False
        try:
            yield
            completed = True
        finally:
            self._active = False
            self._activation_cache = {}
            if completed:
                delta = counter_delta(self.snapshot(), before)
                expected = expected_route_delta(arm, self.layers)
                if delta != expected:
                    raise RuntimeError(f"{arm} adapter route mismatch: delta={delta}, expected={expected}")
                if arm == "C":
                    expected_order = [f"transformer_blocks.{block}.{role}" for block in range(self.layers) for role in ROLE_ORDER]
                    if self._active_role_order != expected_order:
                        raise RuntimeError(f"NVFP4 Linear call order changed: got {self._active_role_order}, expected {expected_order}")

    def snapshot(self):
        return copy.deepcopy(self.counters)

    def metadata(self):
        return {
            "recipe": copy.deepcopy(MODELOPT_RECIPE),
            "representative_full_blocks": representative_middle_blocks(self.layers),
            "role_order": list(ROLE_ORDER),
            "role_shapes": {
                entry.qualified_name: {
                    "input": list(entry.input_shape),
                    "weight": list(entry.weight_shape),
                    "gemm_mkn": [entry.m, entry.k, entry.n],
                    "activation_group": entry.activation_group,
                    "activation_amax": float(entry.activation_amax.item()),
                    "weight_amax": float(entry.weight_amax.item()),
                    "activation_global_scale": float(entry.activation_global_scale.item()),
                    "weight_global_scale": float(entry.weight_global_scale.item()),
                    "alpha": float(entry.alpha.item()),
                }
                for entry in self.entries
            },
            "calibration": copy.deepcopy(self.calibration_metadata),
            "setup_counts": {
                "weight_pack_calls": 14 * self.layers,
                "plan_build_calls": 7,
            },
            "binding_mode": copy.deepcopy(self.binding_mode),
            "plan_provenance": self.plan_provenance(),
            "expected_c_forward": expected_route_delta("C", self.layers),
        }

    def restore(self):
        if self._active:
            raise RuntimeError("cannot restore adapter during a model forward")
        for module, forwards in self._installed_generic.items():
            if module.forward in forwards.values():
                module.__dict__.pop("forward", None)
        for module, forwards in self._installed_mlp.items():
            if module.forward in forwards.values():
                module.__dict__.pop("forward", None)
        for module, forwards in self._installed_mod.items():
            if module.forward in forwards.values():
                module.__dict__.pop("forward", None)
        self._selected = None


def install_modelopt_nvfp4_dispatch(model, shape, calibration):
    return QwenImageModelOptNvfp4Adapter(model, shape, calibration)
