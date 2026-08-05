#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark fused cuDNN Frontend activations vs unfused Transformer Engine kernels.

Supported activations:
    swiglu, dswiglu, srelu, dsrelu, geglu, dgeglu

Fused path:
    cuDNN Frontend grouped GEMM + activation wrappers

Unfused TE path:
    TE MXFP8 quantize -> TE grouped GEMM -> optional pointwise scale
    -> TE activation + MXFP8 quantize

Both paths are warmed up, captured with CUDA Graphs, and timed with graph replay.
Functional tests can also be run with --functional-only. They use shared
TE-quantized MXFP8 inputs for the fused and reference paths and compare
dequantized outputs. For GeGLU/DGeGLU, the functional oracle uses an eager
PyTorch reference by default and matches cuDNN Frontend's Megatron-style
QuickGeGLU, clamp, and linear-offset semantics; TE's tex.geglu uses tanh-GELU
and is intentionally different. For dSReLU, the oracle matches cuDNN Frontend's operand order where
the grouped GEMM output is the saved SReLU input and c_tensor is the upstream
gradient.

Shape convention:
    E = number of experts
    T = tokens per expert
    K = input / gradient hidden size
    N = activation output hidden size

Forward GLU activations use a grouped GEMM with output width 2N and activation
output width N. Backward GLU activations use grouped GEMM output width N and
activation-gradient output width 2N. When comparing against a forward GLU shape,
pass --glu-bprop-tokens as half of --tokens for dswiglu/dgeglu. SReLU/dSReLU use
width N.

Example:
    python cutedsl_fusion_benchmarks.py --activation all --experts 8 --tokens 4096 --glu-bprop-tokens 2048 --k 8192 --n 4096
    python cutedsl_fusion_benchmarks.py --activation all --functional-only
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import inspect
import io
import os
import statistics
import sys
from collections.abc import Callable

import torch
import torch.nn.functional as F

ACTIVATIONS = ("swiglu", "dswiglu", "srelu", "dsrelu", "geglu", "dgeglu")
FORWARD_GLU = {"swiglu", "geglu"}
BACKWARD_GLU = {"dswiglu", "dgeglu"}
FORWARD_RELU = {"srelu"}
BACKWARD_RELU = {"dsrelu"}

MXFP8_BLOCK = 32
M_ALIGNED = 256
GEGLU_ALPHA = 1.702
GEGLU_LINEAR_OFFSET = 1.0
GEGLU_CLAMP_MAX = 7.0
GEGLU_CLAMP_MIN = -7.0

_COMPILED_GEGLU_REFERENCE: Callable[[torch.Tensor], torch.Tensor] | None = None
_COMPILED_DGEGLU_REFERENCE: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def parse_dtype(name: str) -> torch.dtype:
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16"):
        return torch.float16
    raise argparse.ArgumentTypeError("dtype must be bf16 or fp16")


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def check_dims(tokens: int, k: int, n: int) -> None:
    dims = {"tokens": tokens, "k": k, "n": n, "2*n": 2 * n}
    bad = [name for name, value in dims.items() if value % MXFP8_BLOCK != 0]
    if bad:
        raise ValueError(f"MXFP8 dimensions must be divisible by {MXFP8_BLOCK}: {bad}")
    if tokens % M_ALIGNED:
        raise ValueError(f"tokens must be a multiple of {M_ALIGNED} for this benchmark")
    if k % 64 or n % 64:
        raise ValueError("cuDNN grouped-GEMM fusion expects k and n to be multiples of 64")


def activation_shape(activation: str, n: int) -> tuple[int, int | None, int]:
    """Return GEMM N, optional fused C input width, and activation output width."""
    if activation in FORWARD_GLU:
        return 2 * n, None, n
    if activation in BACKWARD_GLU:
        return n, 2 * n, 2 * n
    if activation in FORWARD_RELU:
        return n, None, n
    if activation in BACKWARD_RELU:
        return n, n, n
    raise ValueError(f"Unsupported activation: {activation}")


def benchmark_tokens_for_activation(args: argparse.Namespace, activation: str) -> int:
    if activation in BACKWARD_GLU and args.glu_bprop_tokens is not None:
        return args.glu_bprop_tokens
    return args.tokens


def random_layout_tensor(
    *,
    groups: int,
    mode0: int,
    mode1: int,
    mode0_major: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create a tensor in cuDNN Frontend grouped-GEMM layout."""
    shape = (groups, mode1, mode0) if mode0_major else (groups, mode0, mode1)
    order = (2, 1, 0) if mode0_major else (1, 2, 0)
    src = torch.empty(shape, dtype=torch.float32, device=device).uniform_(-2.0, 2.0)
    return src.permute(order).to(dtype)


def random_scale_tensor(
    *,
    groups: int,
    mn: int,
    k: int,
    sf_vec_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create MXFP8 scale factors in cuDNN Frontend's M32x4xK4 layout."""
    sf_k = ceil_div(k, sf_vec_size)
    base_shape = (groups, ceil_div(mn, 128), ceil_div(sf_k, 4), 32, 4, 4)
    scale = torch.empty(base_shape, dtype=torch.float32, device=device).uniform_(1.0, 3.0)
    return scale.permute(3, 4, 1, 5, 2, 0).to(torch.int8).to(dtype)


def view_as_fp8_e4m3(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(dtype=torch.float8_e4m3fn)


def view_as_e8m0(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(dtype=torch.float8_e8m0fnu)


def k_major_3d(tensor: torch.Tensor) -> torch.Tensor:
    rows, cols = tensor.shape
    return tensor.as_strided((rows, cols, 1), (cols, 1, rows * cols))


def compact_sf_to_cudnn_layout(
    scale: torch.Tensor,
    *,
    mn: int,
    nk: int,
    groups: int,
) -> torch.Tensor:
    """Convert compact MXFP8 scales to cuDNN Frontend's M32x4xK4 layout."""
    sf_k = ceil_div(nk, MXFP8_BLOCK)
    m_blocks = ceil_div(mn, 128)
    k_blocks = ceil_div(sf_k, 4)
    padded = torch.zeros(
        (m_blocks * 128, k_blocks * 4, groups),
        dtype=scale.dtype,
        device=scale.device,
    )
    padded[:mn, :sf_k, :] = scale.reshape(mn, sf_k, groups)
    base_data = padded.view(m_blocks, 4, 32, k_blocks, 4, groups).permute(5, 0, 3, 2, 1, 4)

    # Keep the same strided tensor contract as cuDNN Frontend test utilities.
    base = torch.empty(
        (groups, m_blocks, k_blocks, 32, 4, 4),
        dtype=scale.dtype,
        device=scale.device,
    )
    base.copy_(base_data)
    return base.permute(3, 4, 1, 5, 2, 0)


def cudnn_layout_to_compact_sf(
    scale: torch.Tensor,
    *,
    mn: int,
    nk: int,
    groups: int,
) -> torch.Tensor:
    """Convert cuDNN Frontend's M32x4xK4 scale layout back to compact scales."""
    sf_k = ceil_div(nk, MXFP8_BLOCK)
    m_blocks = ceil_div(mn, 128)
    k_blocks = ceil_div(sf_k, 4)
    base = scale.permute(5, 2, 4, 0, 1, 3)
    padded = (
        base.permute(1, 4, 3, 2, 5, 0)
        .contiguous()
        .view(
            m_blocks * 128,
            k_blocks * 4,
            groups,
        )
    )
    return padded[:mn, :sf_k, :]


def dequantize_cudnn_mxfp8_output(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    *,
    out_n: int,
    norm_const: float,
) -> torch.Tensor:
    """Dequantize cuDNN FP8 output using the generated row scale factors."""
    m = tensor.shape[0]
    compact = cudnn_layout_to_compact_sf(scale, mn=m, nk=out_n, groups=1).to(torch.float32)
    expanded = compact.unsqueeze(2).expand(m, compact.shape[1], MXFP8_BLOCK, 1)
    expanded = expanded.reshape(m, compact.shape[1] * MXFP8_BLOCK, 1)[:, :out_n, :]
    return tensor.to(torch.float32) * expanded / norm_const


def glu_interleaved_indices(n: int, *, device: torch.device) -> torch.Tensor:
    """Column order expected by cuDNN GLU kernels for a logical [gate, up] tensor."""
    if n % MXFP8_BLOCK:
        raise ValueError(f"GLU width must be divisible by {MXFP8_BLOCK}, got {n}")
    gate = torch.arange(n, dtype=torch.long, device=device).view(-1, MXFP8_BLOCK)
    up = torch.arange(n, 2 * n, dtype=torch.long, device=device).view(-1, MXFP8_BLOCK)
    return torch.stack((gate, up), dim=1).reshape(-1)


def pack_glu_interleaved(tensor: torch.Tensor) -> torch.Tensor:
    """Pack logical [gate, up] columns into [gate32, up32, ...] GLU layout."""
    n2 = tensor.shape[1]
    if n2 % 2:
        raise ValueError(f"GLU tensor width must be even, got {n2}")
    indices = glu_interleaved_indices(n2 // 2, device=tensor.device)
    return tensor.index_select(1, indices).contiguous()


def unpack_glu_interleaved(tensor: torch.Tensor) -> torch.Tensor:
    """Unpack [gate32, up32, ...] GLU columns back to logical [gate, up]."""
    n2 = tensor.shape[1]
    if n2 % 2:
        raise ValueError(f"GLU tensor width must be even, got {n2}")
    indices = glu_interleaved_indices(n2 // 2, device=tensor.device)
    out = torch.empty_like(tensor)
    out.index_copy_(1, indices, tensor)
    return out


def require_torch_compile() -> None:
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is required for --functional-reference-backend torch_compile")


def quick_gelu(gate: torch.Tensor) -> torch.Tensor:
    """Megatron QuickGELU approximation used by cuDNN FE's GeGLU wrappers."""
    return gate * torch.sigmoid(GEGLU_ALPHA * gate)


def quick_geglu_formula(input_tensor: torch.Tensor) -> torch.Tensor:
    """Megatron-style QuickGeGLU with cuDNN FE clamp and linear-offset settings."""
    gate, up = input_tensor.to(torch.float32).chunk(2, dim=-1)
    gate = gate.clamp(max=GEGLU_CLAMP_MAX)
    up = up.clamp(min=GEGLU_CLAMP_MIN, max=GEGLU_CLAMP_MAX)
    out = quick_gelu(gate) * (up + GEGLU_LINEAR_OFFSET)
    return out.to(input_tensor.dtype)


def compiled_geglu_formula() -> Callable[[torch.Tensor], torch.Tensor]:
    global _COMPILED_GEGLU_REFERENCE
    require_torch_compile()
    if _COMPILED_GEGLU_REFERENCE is None:
        _COMPILED_GEGLU_REFERENCE = torch.compile(quick_geglu_formula, fullgraph=True, dynamic=False)
    return _COMPILED_GEGLU_REFERENCE


def cudnn_geglu_reference(input_tensor: torch.Tensor, quantizer, backend: str) -> object:
    if backend == "torch_compile":
        return quantizer(compiled_geglu_formula()(input_tensor))
    return quantizer(quick_geglu_formula(input_tensor))


def cudnn_dgeglu_formula(
    acc_tensor: torch.Tensor,
    activation_input: torch.Tensor,
    prob_tensor: torch.Tensor,
) -> torch.Tensor:
    """Derivative for cuDNN FE's clamped, offset Megatron-style QuickGeGLU."""
    grad = acc_tensor.to(torch.float32) * prob_tensor.to(torch.float32)
    gate_raw, up_raw = activation_input.to(torch.float32).chunk(2, dim=-1)
    gate = gate_raw.clamp(max=GEGLU_CLAMP_MAX)
    up = up_raw.clamp(min=GEGLU_CLAMP_MIN, max=GEGLU_CLAMP_MAX)
    sigmoid = torch.sigmoid(GEGLU_ALPHA * gate)

    dgate = grad * sigmoid * (1.0 + GEGLU_ALPHA * gate * (1.0 - sigmoid))
    dgate = dgate * (up + GEGLU_LINEAR_OFFSET)
    dup = grad * gate * sigmoid

    dgate = torch.where(gate_raw <= GEGLU_CLAMP_MAX, dgate, torch.zeros_like(dgate))
    up_mask = (up_raw >= GEGLU_CLAMP_MIN) & (up_raw <= GEGLU_CLAMP_MAX)
    dup = torch.where(up_mask, dup, torch.zeros_like(dup))
    return torch.cat((dgate, dup), dim=-1).to(activation_input.dtype)


def compiled_dgeglu_formula() -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    global _COMPILED_DGEGLU_REFERENCE
    require_torch_compile()
    if _COMPILED_DGEGLU_REFERENCE is None:
        _COMPILED_DGEGLU_REFERENCE = torch.compile(cudnn_dgeglu_formula, fullgraph=True, dynamic=False)
    return _COMPILED_DGEGLU_REFERENCE


def cudnn_dgeglu_reference(
    acc_tensor: torch.Tensor,
    activation_input: torch.Tensor,
    prob_tensor: torch.Tensor,
    quantizer,
    backend: str,
) -> object:
    if backend == "torch_compile":
        return quantizer(compiled_dgeglu_formula()(acc_tensor, activation_input, prob_tensor))
    return quantizer(cudnn_dgeglu_formula(acc_tensor, activation_input, prob_tensor))


def cudnn_dsrelu_formula(acc_tensor: torch.Tensor, grad_tensor: torch.Tensor) -> torch.Tensor:
    """cuDNN FE dSReLU math: 2 * relu(acc) * upstream_grad for alpha=prob=1."""
    acc_relu = torch.relu(acc_tensor.to(torch.float32))
    out = 2.0 * acc_relu * grad_tensor.to(torch.float32)
    return out.to(grad_tensor.dtype)


def cudnn_dsrelu_reference(acc_tensor: torch.Tensor, grad_tensor: torch.Tensor, quantizer) -> object:
    return quantizer(cudnn_dsrelu_formula(acc_tensor, grad_tensor))


def pytorch_activation_formula(
    *,
    activation: str,
    tensor: torch.Tensor,
    activation_input: torch.Tensor | None,
    prob: torch.Tensor,
) -> torch.Tensor:
    """PyTorch pointwise baseline with cuDNN FE semantics for GeGLU/dSReLU cases."""
    if activation == "swiglu":
        gate, up = tensor.to(torch.float32).chunk(2, dim=-1)
        return (F.silu(gate) * up).to(tensor.dtype)
    if activation == "geglu":
        return quick_geglu_formula(tensor)
    if activation == "srelu":
        return torch.relu(tensor.to(torch.float32)).square().to(tensor.dtype)

    if activation_input is None:
        raise ValueError(f"{activation} requires activation_input")

    if activation == "dswiglu":
        grad = tensor.to(torch.float32) * prob.to(torch.float32)
        gate, up = activation_input.to(torch.float32).chunk(2, dim=-1)
        sigmoid = torch.sigmoid(gate)
        silu = gate * sigmoid
        dsilu = sigmoid * (1.0 + gate * (1.0 - sigmoid))
        return torch.cat((grad * up * dsilu, grad * silu), dim=-1).to(activation_input.dtype)
    if activation == "dgeglu":
        return cudnn_dgeglu_formula(tensor, activation_input, prob)
    if activation == "dsrelu":
        return cudnn_dsrelu_formula(tensor, activation_input)

    raise ValueError(f"Unsupported activation: {activation}")


def make_cudnn_inputs(
    *,
    activation: str,
    experts: int,
    tokens: int,
    k: int,
    n: int,
    c_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor | None]:
    m = experts * tokens
    gemm_n, c_cols, _ = activation_shape(activation, n)
    sf_vec_size = 32

    out: dict[str, torch.Tensor | None] = {
        "a_tensor": random_layout_tensor(
            groups=1,
            mode0=m,
            mode1=k,
            mode0_major=False,
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        "b_tensor": random_layout_tensor(
            groups=experts,
            mode0=gemm_n,
            mode1=k,
            mode0_major=False,
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        "sfa_tensor": random_scale_tensor(
            groups=1,
            mn=m,
            k=k,
            sf_vec_size=sf_vec_size,
            dtype=torch.float8_e8m0fnu,
            device=device,
        ),
        "sfb_tensor": random_scale_tensor(
            groups=experts,
            mn=gemm_n,
            k=k,
            sf_vec_size=sf_vec_size,
            dtype=torch.float8_e8m0fnu,
            device=device,
        ),
        "padded_offsets": torch.arange(
            tokens,
            m + 1,
            tokens,
            dtype=torch.int32,
            device=device,
        ),
        "alpha_tensor": torch.ones(experts, dtype=torch.float32, device=device),
        "beta_tensor": torch.ones(experts, dtype=torch.float32, device=device),
        "prob_tensor": torch.ones(m, 1, 1, dtype=torch.float32, device=device),
        "dprob_tensor": torch.zeros(m, 1, 1, dtype=torch.float32, device=device),
        "norm_const_tensor": torch.tensor([0.01], dtype=torch.float32, device=device),
        "c_tensor": None,
    }

    if c_cols is not None:
        out["c_tensor"] = random_layout_tensor(
            groups=1,
            mode0=m,
            mode1=c_cols,
            mode0_major=False,
            dtype=c_dtype,
            device=device,
        )
    return out


def make_mxfp8_quantizer(*, rowwise: bool, columnwise: bool):
    import transformer_engine.pytorch  # noqa: F401
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    return MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=rowwise, columnwise=columnwise)


def capture_cuda_graph(fn: Callable[[], object], *, warmup: int) -> tuple[torch.cuda.CUDAGraph, object]:
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outputs = fn()
    torch.cuda.synchronize()
    return graph, graph_outputs


def cuda_graph_time_ms(
    graph: torch.cuda.CUDAGraph,
    *,
    iters: int,
    samples: int,
) -> list[float]:
    out = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(samples):
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        out.append(start.elapsed_time(end) / iters)
    return out


def output_get(outputs: object, names: tuple[str, ...], indices: tuple[int, ...]) -> torch.Tensor:
    """Fetch a named tensor from cuDNN TupleDict-like outputs with tuple fallback."""
    for name in names:
        try:
            value = outputs[name]  # type: ignore[index]
            if isinstance(value, torch.Tensor):
                return value
        except (KeyError, IndexError, TypeError, AttributeError):
            pass
    for index in indices:
        try:
            value = outputs[index]  # type: ignore[index]
            if isinstance(value, torch.Tensor):
                return value
        except (KeyError, IndexError, TypeError, AttributeError):
            pass
    keys = list(outputs.keys()) if hasattr(outputs, "keys") else type(outputs).__name__  # type: ignore[attr-defined]
    raise AssertionError(f"could not find output tensor {names}; available={keys}")


def check_tensor_contract(
    *,
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    if tuple(tensor.shape) != shape:
        raise AssertionError(f"{name} shape mismatch: got {tuple(tensor.shape)}, expected {shape}")
    if tensor.dtype != dtype:
        raise AssertionError(f"{name} dtype mismatch: got {tensor.dtype}, expected {dtype}")
    if not tensor.is_cuda:
        raise AssertionError(f"{name} must be a CUDA tensor")


def check_mxfp8_payload(*, tensor: object, rowwise_data: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tuple(rowwise_data.shape) != shape:
        raise AssertionError(f"unfused MXFP8 rowwise payload shape mismatch: got {tuple(rowwise_data.shape)}, expected {shape}")
    if rowwise_data.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise AssertionError("unfused MXFP8 rowwise payload dtype mismatch: " f"got {rowwise_data.dtype}, expected packed uint8 or float8_e4m3fn")
    if not rowwise_data.is_cuda:
        raise AssertionError("unfused MXFP8 rowwise payload must be a CUDA tensor")

    if getattr(tensor, "_fp8_dtype", None) is None:
        raise AssertionError("unfused output is missing MXFP8 dtype metadata")
    if getattr(tensor, "_rowwise_scale_inv", None) is None:
        raise AssertionError("unfused output is missing rowwise MXFP8 scales")


def check_functional_outputs(
    *,
    activation: str,
    fused_outputs: object,
    unfused_output: object,
    m: int,
    out_n: int,
    dtype: torch.dtype,
) -> None:
    if activation in FORWARD_GLU or activation in FORWARD_RELU:
        fused_primary = output_get(fused_outputs, ("d_tensor", "D"), (1, 0))
        fused_name = "fused d_tensor"
    else:
        fused_primary = output_get(fused_outputs, ("d_row_tensor", "d_tensor", "D"), (0,))
        fused_name = "fused d_row_tensor"

    check_tensor_contract(
        name=fused_name,
        tensor=fused_primary,
        shape=(m, out_n, 1),
        dtype=torch.float8_e4m3fn,
    )

    if not hasattr(unfused_output, "shape"):
        raise AssertionError(f"unfused output has no shape attribute: {type(unfused_output).__name__}")
    if tuple(unfused_output.shape) != (m, out_n):  # type: ignore[attr-defined]
        raise AssertionError(f"unfused output shape mismatch: got {tuple(unfused_output.shape)}, expected {(m, out_n)}")  # type: ignore[attr-defined]
    if getattr(unfused_output, "dtype", None) != dtype:
        raise AssertionError(f"unfused output dtype mismatch: got {getattr(unfused_output, 'dtype', None)}, expected {dtype}")

    rowwise_data = getattr(unfused_output, "_rowwise_data", None)
    if rowwise_data is None:
        raise AssertionError("unfused output is not rowwise MXFP8-quantized")
    check_mxfp8_payload(tensor=unfused_output, rowwise_data=rowwise_data, shape=(m, out_n))


def make_shared_functional_case(
    *,
    activation: str,
    tex,
    run_grouped_gemm,
    experts: int,
    tokens: int,
    m: int,
    k: int,
    n: int,
    dtype: torch.dtype,
    device: torch.device,
    wrappers,
    use_dynamic_sched: bool,
    reference_backend: str,
) -> tuple[Callable[[], object], Callable[[], object]]:
    gemm_n, c_cols, _ = activation_shape(activation, n)
    norm_const = 0.01

    a = torch.randn(m, k, dtype=dtype, device=device) * 0.1
    weight = torch.randn(experts, k, gemm_n, dtype=dtype, device=device) * 0.1

    q_a = make_mxfp8_quantizer(rowwise=True, columnwise=False)
    q_w = make_mxfp8_quantizer(rowwise=True, columnwise=True)
    q_out = make_mxfp8_quantizer(rowwise=True, columnwise=True)

    a_parts = [q_a(a[i * tokens : (i + 1) * tokens]) for i in range(experts)]
    w_parts = [q_w(weight[i]) for i in range(experts)]
    fused_weight = weight
    if activation in FORWARD_GLU:
        fused_weight = torch.stack(
            [pack_glu_interleaved(weight[i]) for i in range(experts)],
            dim=0,
        )
    fused_w_parts = [q_w(fused_weight[i]) for i in range(experts)]

    a_payload = torch.cat([view_as_fp8_e4m3(part._rowwise_data) for part in a_parts], dim=0).contiguous()
    a_scale = torch.cat([view_as_e8m0(part._rowwise_scale_inv) for part in a_parts], dim=0)
    b_payload = torch.stack(
        [view_as_fp8_e4m3(part._columnwise_data).T.contiguous() for part in fused_w_parts],
        dim=0,
    ).permute(1, 2, 0)
    b_scale = torch.stack(
        [view_as_e8m0(part._columnwise_scale_inv).T.contiguous() for part in fused_w_parts],
        dim=2,
    )

    act_input = None
    c_tensor = None
    if c_cols is not None:
        act_input = torch.randn(m, c_cols, dtype=dtype, device=device) * 0.1
        fused_act_input = pack_glu_interleaved(act_input) if activation in BACKWARD_GLU else act_input
        c_tensor = k_major_3d(fused_act_input)

    fused_inputs: dict[str, torch.Tensor | None] = {
        "a_tensor": k_major_3d(a_payload),
        "b_tensor": b_payload,
        "sfa_tensor": compact_sf_to_cudnn_layout(
            a_scale.reshape(m, k // MXFP8_BLOCK, 1),
            mn=m,
            nk=k,
            groups=1,
        ),
        "sfb_tensor": compact_sf_to_cudnn_layout(
            b_scale,
            mn=gemm_n,
            nk=k,
            groups=experts,
        ),
        "padded_offsets": torch.arange(
            tokens,
            m + 1,
            tokens,
            dtype=torch.int32,
            device=device,
        ),
        "alpha_tensor": torch.ones(experts, dtype=torch.float32, device=device),
        "beta_tensor": torch.ones(experts, dtype=torch.float32, device=device),
        "prob_tensor": torch.ones(m, 1, 1, dtype=torch.float32, device=device),
        "dprob_tensor": torch.zeros(m, 1, 1, dtype=torch.float32, device=device),
        "norm_const_tensor": torch.tensor([norm_const], dtype=torch.float32, device=device),
        "c_tensor": c_tensor,
    }

    fused = make_fused_fn(
        activation=activation,
        inputs=fused_inputs,
        dtype=dtype,
        use_dynamic_sched=use_dynamic_sched,
        wrappers=wrappers,
    )

    gemm_out = torch.empty(m, gemm_n, dtype=dtype, device=device)
    prob = torch.ones(m, 1, dtype=dtype, device=device)
    dact_dy = torch.empty_like(gemm_out)

    if activation in FORWARD_GLU or activation in FORWARD_RELU:

        def unfused() -> object:
            run_grouped_gemm(w_parts, a_parts, gemm_out)
            if activation == "geglu":
                return cudnn_geglu_reference(gemm_out, q_out, reference_backend)
            return getattr(tex, activation)(gemm_out, q_out)

        return fused, unfused

    if activation in BACKWARD_GLU or activation in BACKWARD_RELU:

        def unfused() -> object:
            run_grouped_gemm(w_parts, a_parts, gemm_out)
            if activation == "dgeglu":
                return cudnn_dgeglu_reference(gemm_out, act_input, prob, q_out, reference_backend)
            if activation == "dsrelu":
                return cudnn_dsrelu_reference(gemm_out, act_input, q_out)
            torch.mul(gemm_out, prob, out=dact_dy)
            return getattr(tex, activation)(dact_dy, act_input, q_out)

        return fused, unfused

    raise ValueError(f"Unsupported activation: {activation}")


def compare_functional_outputs(
    *,
    activation: str,
    fused_outputs: object,
    unfused_output: object,
    m: int,
    out_n: int,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> tuple[float, float, float]:
    check_functional_outputs(
        activation=activation,
        fused_outputs=fused_outputs,
        unfused_output=unfused_output,
        m=m,
        out_n=out_n,
        dtype=dtype,
    )

    if activation in FORWARD_GLU or activation in FORWARD_RELU:
        fused_primary = output_get(fused_outputs, ("d_tensor", "D"), (1, 0))
    else:
        fused_primary = output_get(fused_outputs, ("d_row_tensor", "d_tensor", "D"), (0,))

    sfd_row = output_get(fused_outputs, ("sfd_row_tensor",), (4, 5))
    fused_deq = dequantize_cudnn_mxfp8_output(
        fused_primary,
        sfd_row,
        out_n=out_n,
        norm_const=0.01,
    ).reshape(m, out_n)
    if activation in BACKWARD_GLU:
        fused_deq = unpack_glu_interleaved(fused_deq)
    unfused_deq = unfused_output.dequantize(dtype=torch.float32).reshape(m, out_n)  # type: ignore[attr-defined]

    diff = (fused_deq - unfused_deq).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    p99_abs = torch.quantile(diff.flatten(), 0.99).item()

    try:
        torch.testing.assert_close(fused_deq, unfused_deq, atol=atol, rtol=rtol)
    except AssertionError as exc:
        raise AssertionError(
            f"{activation} output mismatch: max_abs={max_abs:.6g}, " f"p99_abs={p99_abs:.6g}, mean_abs={mean_abs:.6g}, " f"atol={atol}, rtol={rtol}"
        ) from exc

    return max_abs, p99_abs, mean_abs


def run_functional_tests(
    *,
    activations: tuple[str, ...],
    args,
    tex,
    general_grouped_gemm,
    wrappers,
    device: torch.device,
) -> None:
    e = args.functional_experts
    t = args.functional_tokens
    k = args.functional_k
    n = args.functional_n
    m = e * t
    check_dims(t, k, n)

    print()
    print("functional tests:")
    print("  scope: shared-input dequantized output comparison")
    print(f"  shape: experts={e}, tokens/expert={t}, M={m}, K={k}, N={n}")
    print(f"  tolerance: atol={args.functional_atol}, rtol={args.functional_rtol}")
    if any(activation in {"geglu", "dgeglu"} for activation in activations):
        print("  geglu reference: " f"{args.functional_reference_backend} Megatron-style QuickGeGLU + clamp + linear-offset semantics")
    if "dsrelu" in activations:
        print("  dsrelu reference: cuDNN FE operand order, GEMM output is saved activation input")

    run_grouped_gemm = make_grouped_gemm_runner(
        general_grouped_gemm,
        experts=e,
        tokens=t,
        dtype=args.dtype,
    )

    with torch.no_grad():
        for index, activation in enumerate(activations):
            torch.manual_seed(args.seed + 1000 + index)
            _, _, out_n = activation_shape(activation, n)
            fused, unfused = make_shared_functional_case(
                activation=activation,
                tex=tex,
                run_grouped_gemm=run_grouped_gemm,
                experts=e,
                tokens=t,
                m=m,
                k=k,
                n=n,
                dtype=args.dtype,
                device=device,
                wrappers=wrappers,
                use_dynamic_sched=not args.no_dynamic_sched,
                reference_backend=args.functional_reference_backend,
            )

            fused_outputs = fused()
            unfused_output = unfused()
            torch.cuda.synchronize()
            max_abs, p99_abs, mean_abs = compare_functional_outputs(
                activation=activation,
                fused_outputs=fused_outputs,
                unfused_output=unfused_output,
                m=m,
                out_n=out_n,
                dtype=args.dtype,
                atol=args.functional_atol,
                rtol=args.functional_rtol,
            )
            print(f"  {activation:<8} PASS " f"max_abs={max_abs:.4g} p99_abs={p99_abs:.4g} mean_abs={mean_abs:.4g}")


def make_grouped_gemm_runner(general_grouped_gemm, *, experts: int, tokens: int, dtype: torch.dtype):
    """Handle TE grouped-GEMM signature differences across NGC and TE main."""
    params = inspect.signature(general_grouped_gemm).parameters
    if "quantization_params" in params:

        def run(w_parts, a_parts, gemm_out) -> None:
            general_grouped_gemm(
                w_parts,
                a_parts,
                [gemm_out],
                [None] * experts,
                dtype,
                layout="NN",
                m_splits=[tokens] * experts,
                use_split_accumulator=True,
                single_output=True,
            )

        return run

    from transformer_engine.pytorch.module.base import get_multi_stream_cublas_workspace

    workspace = get_multi_stream_cublas_workspace()

    def run(w_parts, a_parts, gemm_out) -> None:
        general_grouped_gemm(
            w_parts,
            a_parts,
            [gemm_out],
            dtype,
            workspace,
            layout="NN",
            m_splits=[tokens] * experts,
            use_split_accumulator=True,
            single_output=True,
        )

    return run


def make_fused_fn(
    *,
    activation: str,
    inputs: dict[str, torch.Tensor | None],
    dtype: torch.dtype,
    use_dynamic_sched: bool,
    wrappers,
):
    common = {
        "a_tensor": inputs["a_tensor"],
        "b_tensor": inputs["b_tensor"],
        "sfa_tensor": inputs["sfa_tensor"],
        "sfb_tensor": inputs["sfb_tensor"],
        "padded_offsets": inputs["padded_offsets"],
        "alpha_tensor": inputs["alpha_tensor"],
        "norm_const_tensor": inputs["norm_const_tensor"],
        "acc_dtype": torch.float32,
        "d_dtype": torch.float8_e4m3fn,
        "cd_major": "n",
        "mma_tiler_mn": (256, 256),
        "cluster_shape_mn": (2, 1),
        "sf_vec_size": 32,
        "vector_f32": False,
        "m_aligned": M_ALIGNED,
        "discrete_col_sfd": False,
    }

    def current_stream():
        return wrappers["cuda"].CUstream(torch.cuda.current_stream().cuda_stream)

    if activation in FORWARD_GLU:

        def fused() -> object:
            return wrappers["glu"](
                **common,
                prob_tensor=inputs["prob_tensor"],
                c_dtype=dtype,
                act_func=activation,
                use_dynamic_sched=use_dynamic_sched,
                current_stream=current_stream(),
            )

        return fused

    if activation in FORWARD_RELU:

        def fused() -> object:
            return wrappers["srelu"](
                **common,
                prob_tensor=inputs["prob_tensor"],
                c_dtype=dtype,
                use_dynamic_sched=use_dynamic_sched,
                current_stream=current_stream(),
            )

        return fused

    if activation in BACKWARD_GLU:

        def fused() -> object:
            inputs["dprob_tensor"].zero_()
            return wrappers["dglu"](
                **common,
                c_tensor=inputs["c_tensor"],
                beta_tensor=inputs["beta_tensor"],
                prob_tensor=inputs["prob_tensor"],
                dprob_tensor=inputs["dprob_tensor"],
                generate_dbias=False,
                act_func=activation,
                use_dynamic_sched=use_dynamic_sched,
                current_stream=current_stream(),
            )

        return fused

    if activation in BACKWARD_RELU:

        def fused() -> object:
            inputs["dprob_tensor"].zero_()
            return wrappers["dsrelu"](
                **common,
                c_tensor=inputs["c_tensor"],
                prob_tensor=inputs["prob_tensor"],
                dprob_tensor=inputs["dprob_tensor"],
                generate_dbias=False,
                use_dynamic_sched=use_dynamic_sched,
                current_stream=current_stream(),
            )

        return fused

    raise ValueError(f"Unsupported activation: {activation}")


def make_unfused_te_fn(
    *,
    activation: str,
    tex,
    run_grouped_gemm,
    experts: int,
    tokens: int,
    m: int,
    k: int,
    n: int,
    dtype: torch.dtype,
    device: torch.device,
):
    gemm_n, c_cols, _ = activation_shape(activation, n)

    a = torch.randn(m, k, dtype=dtype, device=device) * 0.1
    weight = torch.randn(experts, k, gemm_n, dtype=dtype, device=device) * 0.1

    q_a = make_mxfp8_quantizer(rowwise=True, columnwise=False)
    q_w = make_mxfp8_quantizer(rowwise=True, columnwise=True)
    q_out = make_mxfp8_quantizer(rowwise=True, columnwise=True)

    a_parts = [q_a(a[i * tokens : (i + 1) * tokens]) for i in range(experts)]
    w_parts = [q_w(weight[i]) for i in range(experts)]
    gemm_out = torch.empty(m, gemm_n, dtype=dtype, device=device)

    prob = torch.ones(m, 1, dtype=dtype, device=device)
    dact_dy = torch.empty_like(gemm_out)
    act_input = None
    if c_cols is not None:
        act_input = torch.randn(m, c_cols, dtype=dtype, device=device) * 0.1

    if activation in FORWARD_GLU or activation in FORWARD_RELU:

        def unfused() -> object:
            run_grouped_gemm(w_parts, a_parts, gemm_out)
            return getattr(tex, activation)(gemm_out, q_out)

        return unfused

    if activation in BACKWARD_GLU or activation in BACKWARD_RELU:

        def unfused() -> object:
            run_grouped_gemm(w_parts, a_parts, gemm_out)
            torch.mul(gemm_out, prob, out=dact_dy)
            return getattr(tex, activation)(dact_dy, act_input, q_out)

        return unfused

    raise ValueError(f"Unsupported activation: {activation}")


def make_pytorch_baseline_fn(
    *,
    activation: str,
    run_grouped_gemm,
    experts: int,
    tokens: int,
    m: int,
    k: int,
    n: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """TE MXFP8 grouped GEMM followed by PyTorch pointwise activation and TE quantize."""
    gemm_n, c_cols, _ = activation_shape(activation, n)

    a = torch.randn(m, k, dtype=dtype, device=device) * 0.1
    weight = torch.randn(experts, k, gemm_n, dtype=dtype, device=device) * 0.1

    q_a = make_mxfp8_quantizer(rowwise=True, columnwise=False)
    q_w = make_mxfp8_quantizer(rowwise=True, columnwise=True)
    q_out = make_mxfp8_quantizer(rowwise=True, columnwise=True)

    a_parts = [q_a(a[i * tokens : (i + 1) * tokens]) for i in range(experts)]
    w_parts = [q_w(weight[i]) for i in range(experts)]
    gemm_out = torch.empty(m, gemm_n, dtype=dtype, device=device)

    prob = torch.ones(m, 1, dtype=dtype, device=device)
    act_input = None
    if c_cols is not None:
        act_input = torch.randn(m, c_cols, dtype=dtype, device=device) * 0.1

    def pytorch_baseline() -> object:
        run_grouped_gemm(w_parts, a_parts, gemm_out)
        out = pytorch_activation_formula(
            activation=activation,
            tensor=gemm_out,
            activation_input=act_input,
            prob=prob,
        )
        return q_out(out)

    return pytorch_baseline


def benchmark_activation(
    *,
    activation: str,
    args,
    tex,
    general_grouped_gemm,
    wrappers,
    device: torch.device,
) -> tuple[str, float, float, float | None, list[float], list[float], list[float] | None]:
    e, k, n = args.experts, args.k, args.n
    t = benchmark_tokens_for_activation(args, activation)
    m = e * t
    gemm_n, c_cols, out_n = activation_shape(activation, n)

    print()
    print(f"activation: {activation}")
    print(f"shape: experts={e}, tokens/expert={t}, M={m}, K={k}, N={n}")
    print(f"fused GEMM N={gemm_n}, fused C input N={c_cols}, activation output N={out_n}")

    fused_inputs = make_cudnn_inputs(
        activation=activation,
        experts=e,
        tokens=t,
        k=k,
        n=n,
        c_dtype=args.dtype,
        device=device,
    )
    fused = make_fused_fn(
        activation=activation,
        inputs=fused_inputs,
        dtype=args.dtype,
        use_dynamic_sched=not args.no_dynamic_sched,
        wrappers=wrappers,
    )

    run_grouped_gemm = make_grouped_gemm_runner(
        general_grouped_gemm,
        experts=e,
        tokens=t,
        dtype=args.dtype,
    )
    unfused = make_unfused_te_fn(
        activation=activation,
        tex=tex,
        run_grouped_gemm=run_grouped_gemm,
        experts=e,
        tokens=t,
        m=m,
        k=k,
        n=n,
        dtype=args.dtype,
        device=device,
    )
    fused_graph, fused_outputs = capture_cuda_graph(fused, warmup=args.warmup)
    unfused_graph, unfused_outputs = capture_cuda_graph(unfused, warmup=args.warmup)
    graph_outputs = [fused_outputs, unfused_outputs]

    pytorch_graph = None
    pytorch_ms = None
    pytorch_avg = None
    if args.include_pytorch_baseline:
        pytorch_baseline = make_pytorch_baseline_fn(
            activation=activation,
            run_grouped_gemm=run_grouped_gemm,
            experts=e,
            tokens=t,
            m=m,
            k=k,
            n=n,
            dtype=args.dtype,
            device=device,
        )
        pytorch_graph, pytorch_outputs = capture_cuda_graph(pytorch_baseline, warmup=args.warmup)
        graph_outputs.append(pytorch_outputs)

    fused_ms = cuda_graph_time_ms(fused_graph, iters=args.iters, samples=args.samples)
    unfused_ms = cuda_graph_time_ms(unfused_graph, iters=args.iters, samples=args.samples)
    if pytorch_graph is not None:
        pytorch_ms = cuda_graph_time_ms(pytorch_graph, iters=args.iters, samples=args.samples)

    fused_avg = statistics.mean(fused_ms)
    unfused_avg = statistics.mean(unfused_ms)
    if pytorch_ms is not None:
        pytorch_avg = statistics.mean(pytorch_ms)
    print(f"fused cuDNN avg: {fused_avg:.3f} ms")
    print(f"unfused TE avg:  {unfused_avg:.3f} ms")
    print(f"speedup vs TE:   {unfused_avg / fused_avg:.2f}x")
    if pytorch_avg is not None:
        print(f"PyTorch avg:     {pytorch_avg:.3f} ms")
        print(f"speedup vs torch:{pytorch_avg / fused_avg:6.2f}x")
    print(f"fused samples:   {[round(x, 3) for x in fused_ms]}")
    print(f"TE samples:      {[round(x, 3) for x in unfused_ms]}")
    if pytorch_ms is not None:
        print(f"PyTorch samples: {[round(x, 3) for x in pytorch_ms]}")
    del graph_outputs
    return activation, fused_avg, unfused_avg, pytorch_avg, fused_ms, unfused_ms, pytorch_ms


class _Tee:
    """Write to several streams at once (e.g. real stdout and a capture buffer)."""

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


# Lines kept in output.txt: device, dtype, library versions, and shapes.
OUTPUT_TXT_PREFIXES = (
    "device:",
    "dtype:",
    "torch:",
    "transformer_engine:",
    "nvidia-cudnn-frontend:",
    "nvidia-cutlass-dsl:",
    "shape:",
    "base shape:",
    "GLU bprop shape:",
)

# Per-activation shape lines move next to each bar in the chart, so they are
# dropped from the chart's header block (but kept in output.txt).
SHAPE_PREFIXES = ("shape:", "base shape:", "GLU bprop shape:")

DISPLAY_NAMES = {
    "swiglu": "SwiGLU",
    "dswiglu": "dSwiGLU",
    "srelu": "SReLU",
    "dsrelu": "dSReLU",
    "geglu": "QuickGeGLU",
    "dgeglu": "dQuickGeGLU",
}


# Preferred header fonts; matplotlib falls back to the first one available.
HEADER_FONT = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
HEADER_ACCENT = "#00866a"

# Prettier labels for the environment metadata lines in the chart header.
LABEL_PRETTY = {
    "device": "Device",
    "dtype": "Dtype",
    "torch": "PyTorch",
    "transformer_engine": "Transformer Engine",
    "nvidia-cudnn-frontend": "cuDNN Frontend",
    "nvidia-cutlass-dsl": "cuTLASS DSL",
}


def display_name(activation: str) -> str:
    return DISPLAY_NAMES.get(activation, activation)


def family_color(activation: str) -> str:
    """Bar color grouped by activation family (GLU=green, ReLU=blue, GeGLU=amber)."""
    if activation in ("swiglu", "dswiglu"):
        return "#0db58f"
    if activation in ("srelu", "dsrelu"):
        return "#6c8ff0"
    return "#f2a51f"


def format_summary(results: list, *, include_pytorch: bool) -> list[str]:
    """Render the benchmark summary table as a list of lines."""
    lines = ["summary:"]
    if include_pytorch:
        lines.append(f"{'activation':<10} {'cudnn_ms':>10} {'te_ms':>10} {'pytorch_ms':>12} " f"{'te/cudnn':>10} {'torch/cudnn':>12}")
        for activation, fused_avg, unfused_avg, pytorch_avg, _, _, _ in results:
            assert pytorch_avg is not None
            lines.append(
                f"{activation:<10} {fused_avg:10.3f} {unfused_avg:10.3f} {pytorch_avg:12.3f} "
                f"{unfused_avg / fused_avg:9.2f}x {pytorch_avg / fused_avg:11.2f}x"
            )
    else:
        lines.append(f"{'activation':<10} {'cudnn_ms':>10} {'te_ms':>10} {'te/cudnn':>10}")
        for activation, fused_avg, unfused_avg, _, _, _, _ in results:
            lines.append(f"{activation:<10} {fused_avg:10.3f} {unfused_avg:10.3f} {unfused_avg / fused_avg:9.2f}x")
    return lines


def write_results_png(
    results: list,
    *,
    include_pytorch: bool,
    shapes: dict[str, str],
    info_lines: list[str],
    path: str,
) -> None:
    """Render speedups as a horizontal bar chart with a metadata header block."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        from matplotlib.patches import Patch
    except ImportError:
        print(f"matplotlib not available; skipping {path}")
        return

    # Only request fonts that exist so matplotlib does not warn on every glyph.
    available = {f.name for f in font_manager.fontManager.ttflist}
    header_font = [f for f in HEADER_FONT if f in available] or ["sans-serif"]

    activations = [r[0] for r in results]
    te_speedups = [unfused / fused for _, fused, unfused, _, _, _, _ in results]
    torch_speedups = [(pytorch / fused) if pytorch is not None else None for _, fused, _, pytorch, _, _, _ in results]
    have_torch = include_pytorch and all(s is not None for s in torch_speedups)

    # Header highlights: prominent speedup stats plus the environment metadata.
    avg_te = sum(te_speedups) / len(te_speedups)
    best_idx = max(range(len(te_speedups)), key=lambda i: te_speedups[i])
    stat_groups = [
        (f"{avg_te:.2f}x", "average speedup vs TE"),
        (f"{te_speedups[best_idx]:.2f}x", f"best: {display_name(activations[best_idx])}"),
    ]
    if have_torch:
        avg_torch = sum(torch_speedups) / len(torch_speedups)
        stat_groups.append((f"{avg_torch:.2f}x", "average speedup vs PyTorch"))
    stat_groups.append((str(len(results)), "activations benchmarked"))

    env_pairs = []
    for line in info_lines:
        key, _, value = line.partition(":")
        env_pairs.append((LABEL_PRETTY.get(key.strip(), key.strip()), value.strip()))
    env_rows = (len(env_pairs) + 1) // 2

    # Round the x-axis up to a clean 0.5 multiple that clears every bar.
    raw_max = max(list(te_speedups) + [s for s in torch_speedups if s is not None] + [1.0])
    tick_max = 0.5
    while tick_max < raw_max + 1e-9:
        tick_max += 0.5
    tick_max = max(tick_max, 1.5)
    gutter = 0.55 * tick_max  # left label column, in data units

    n = len(results)
    head_h = 1.35 + 0.42 * env_rows
    chart_h = 0.95 * n + 0.7
    fig = plt.figure(figsize=(14.0, head_h + chart_h + 0.5))
    fig.suptitle(
        "cuDNN fused-epilogue speedups vs Transformer Engine",
        fontsize=20,
        fontweight="bold",
        color="#0f172a",
        family=header_font,
        x=0.07,
        ha="left",
    )
    grid = fig.add_gridspec(2, 1, height_ratios=[head_h, chart_h], hspace=0.08)

    ax_info = fig.add_subplot(grid[0])
    ax_info.axis("off")

    # Stat highlights across the top: big accent number + muted caption.
    n_stat = len(stat_groups)
    for i, (value, caption) in enumerate(stat_groups):
        x = i / n_stat
        ax_info.text(x, 0.98, value, transform=ax_info.transAxes, va="top", ha="left", fontsize=26, fontweight="bold", color=HEADER_ACCENT, family=header_font)
        ax_info.text(x, 0.60, caption, transform=ax_info.transAxes, va="top", ha="left", fontsize=12, color="#64748b", family=header_font)

    # Thin divider between the highlights and the environment metadata.
    ax_info.plot([0.0, 1.0], [0.50, 0.50], transform=ax_info.transAxes, color="#e2e8f0", lw=1.2)

    # Environment metadata as a two-column label/value grid.
    label_x = (0.0, 0.50)
    value_x = (0.16, 0.74)
    row_top = 0.40
    row_step = 0.40 / max(env_rows, 1)
    for idx, (label, value) in enumerate(env_pairs):
        col = idx % 2
        y = row_top - (idx // 2) * row_step
        ax_info.text(
            label_x[col], y, label, transform=ax_info.transAxes, va="top", ha="left", fontsize=12, fontweight="bold", color="#334155", family=header_font
        )
        ax_info.text(value_x[col], y, value, transform=ax_info.transAxes, va="top", ha="left", fontsize=12, color="#0f172a", family=header_font)

    ax = fig.add_subplot(grid[1])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_yticks([])
    ax.set_xlim(-gutter, tick_max * 1.12)
    ax.set_ylim(-0.7, n - 0.3)
    ax.invert_yaxis()

    xticks = []
    tick = 0.0
    while tick <= tick_max + 1e-9:
        xticks.append(round(tick, 1))
        tick += 0.5
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:g}x" if x > 0 else "0" for x in xticks], color="#64748b")
    ax.tick_params(axis="x", length=0)
    for x in xticks:
        ax.axvline(x, color="#eef2f7", lw=1.0, zorder=0)
    ax.axvline(0.0, color="#cbd5e1", lw=1.2, zorder=1)
    ax.axvline(1.0, color="#94a3b8", lw=1.4, ls=(0, (5, 7)), zorder=1)
    ax.text(1.0, -0.62, "TE parity", color="#64748b", fontsize=11, ha="center", va="bottom")

    label_pad = tick_max * 0.02
    for i, activation in enumerate(activations):
        color = family_color(activation)
        shape = shapes.get(activation, "")
        ax.text(-gutter * 0.96, i - 0.14, display_name(activation), ha="left", va="center", fontweight="bold", fontsize=13, color="#0f172a")
        if shape:
            ax.text(-gutter * 0.96, i + 0.18, shape, ha="left", va="center", fontsize=9.5, color="#64748b")

        if have_torch:
            ax.barh(i - 0.17, te_speedups[i], height=0.30, color=color, zorder=3)
            ax.barh(i + 0.17, torch_speedups[i], height=0.30, color=color, alpha=0.5, zorder=3)
            ax.text(te_speedups[i] + label_pad, i - 0.17, f"{te_speedups[i]:.2f}x", ha="left", va="center", fontsize=11, fontweight="bold", color="#0f172a")
            ax.text(torch_speedups[i] + label_pad, i + 0.17, f"{torch_speedups[i]:.2f}x", ha="left", va="center", fontsize=10, color="#475569")
        else:
            ax.barh(i, te_speedups[i], height=0.5, color=color, zorder=3)
            ax.text(te_speedups[i] + label_pad, i, f"{te_speedups[i]:.2f}x", ha="left", va="center", fontsize=12, fontweight="bold", color="#0f172a")

    if have_torch:
        handles = [
            Patch(facecolor="#475569", label="cuDNN vs TE"),
            Patch(facecolor="#475569", alpha=0.5, label="cuDNN vs PyTorch"),
        ]
        ax.legend(
            handles=handles,
            loc="lower right",
            bbox_to_anchor=(1.0, 1.0),
            ncol=2,
            frameon=False,
            fontsize=10,
        )

    ax.set_xlabel("speedup (higher is better)", color="#475569", fontsize=11)

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def write_output_files(
    *,
    output_dir: str,
    verbose_text: str,
    results: list,
    args: argparse.Namespace,
) -> None:
    """Write output_verbose.txt, output.txt, and results.png into output_dir."""
    include_pytorch = args.include_pytorch_baseline
    verbose_path = os.path.join(output_dir, "output_verbose.txt")
    with open(verbose_path, "w") as handle:
        handle.write(verbose_text)
    print(f"wrote {verbose_path}")

    summary_lines = format_summary(results, include_pytorch=include_pytorch) if results else []
    # Preserve order but drop duplicates (per-activation "shape:" lines repeat the top-level one).
    key_lines: list[str] = []
    seen: set[str] = set()
    for line in verbose_text.splitlines():
        if line.startswith(OUTPUT_TXT_PREFIXES) and line not in seen:
            seen.add(line)
            key_lines.append(line)
    output_path = os.path.join(output_dir, "output.txt")
    with open(output_path, "w") as handle:
        for line in key_lines:
            handle.write(line + "\n")
        if summary_lines:
            handle.write("\n")
            for line in summary_lines:
                handle.write(line + "\n")
    print(f"wrote {output_path}")

    if results:
        # Per-activation GEMM shape: experts x tokens/expert x GEMM-N x K.
        shapes = {}
        for activation, *_ in results:
            tokens = benchmark_tokens_for_activation(args, activation)
            gemm_n, _, _ = activation_shape(activation, args.n)
            shapes[activation] = f"GEMM: {args.experts} x {tokens} x {gemm_n} x {args.k}"
        # The chart header keeps device/dtype/version info; shapes move next to each bar.
        chart_info_lines = [line for line in key_lines if not line.startswith(SHAPE_PREFIXES)]
        write_results_png(
            results,
            include_pytorch=include_pytorch,
            shapes=shapes,
            info_lines=chart_info_lines,
            path=os.path.join(output_dir, "results.png"),
        )


def run_benchmarks(args: argparse.Namespace) -> list:
    """Run functional tests and/or benchmarks, printing progress. Returns timing results."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)

    import transformer_engine.pytorch  # noqa: F401
    import transformer_engine_torch as tex
    from cuda.bindings import driver as cuda
    from cudnn import (
        grouped_gemm_dglu_wrapper_sm100,
        grouped_gemm_dsrelu_wrapper_sm100,
        grouped_gemm_glu_wrapper_sm100,
        grouped_gemm_srelu_wrapper_sm100,
    )
    from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm

    torch.manual_seed(args.seed)

    print(f"device: {torch.cuda.get_device_name(args.device)}")
    if not args.functional_only:
        if args.glu_bprop_tokens is None:
            print(f"shape: experts={args.experts}, tokens/expert={args.tokens}, " f"M={args.experts * args.tokens}, K={args.k}, N={args.n}")
        else:
            print(f"base shape: experts={args.experts}, tokens/expert={args.tokens}, " f"M={args.experts * args.tokens}, K={args.k}, N={args.n}")
            print(
                f"GLU bprop shape: experts={args.experts}, "
                f"tokens/expert={args.glu_bprop_tokens}, "
                f"M={args.experts * args.glu_bprop_tokens}, K={args.k}, N={args.n}"
            )
    print(f"dtype: {args.dtype}")
    print(f"torch: {torch.__version__}")
    print(f"transformer_engine: {package_version('transformer-engine')}")
    print(f"nvidia-cudnn-frontend: {package_version('nvidia-cudnn-frontend')}")
    print(f"nvidia-cutlass-dsl: {package_version('nvidia-cutlass-dsl')}")

    wrappers = {
        "cuda": cuda,
        "glu": grouped_gemm_glu_wrapper_sm100,
        "srelu": grouped_gemm_srelu_wrapper_sm100,
        "dsrelu": grouped_gemm_dsrelu_wrapper_sm100,
        "dglu": grouped_gemm_dglu_wrapper_sm100,
    }
    activations = ACTIVATIONS if args.activation == "all" else (args.activation,)

    if args.functional_test or args.functional_only:
        run_functional_tests(
            activations=activations,
            args=args,
            tex=tex,
            general_grouped_gemm=general_grouped_gemm,
            wrappers=wrappers,
            device=device,
        )
        if args.functional_only:
            return []

    for tokens in sorted({benchmark_tokens_for_activation(args, activation) for activation in activations}):
        check_dims(tokens, args.k, args.n)
    print("timing mode: CUDA graph replay")

    results = [
        benchmark_activation(
            activation=activation,
            args=args,
            tex=tex,
            general_grouped_gemm=general_grouped_gemm,
            wrappers=wrappers,
            device=device,
        )
        for activation in activations
    ]

    if len(results) > 1:
        print()
        for line in format_summary(results, include_pytorch=args.include_pytorch_baseline):
            print(line)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", choices=("all",) + ACTIVATIONS, default="all")
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=4096, help="tokens per expert")
    parser.add_argument(
        "--glu-bprop-tokens",
        type=int,
        default=None,
        help="tokens per expert for dswiglu/dgeglu; use half of --tokens for paired GLU bprop",
    )
    parser.add_argument("--k", type=int, default=8192, help="input / gradient hidden size")
    parser.add_argument("--n", type=int, default=4096, help="activation output hidden size")
    parser.add_argument("--dtype", type=parse_dtype, default=torch.bfloat16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--no-dynamic-sched", action="store_true")
    parser.add_argument(
        "--include-pytorch-baseline",
        action="store_true",
        help="also time the TE GEMM + PyTorch pointwise baseline",
    )
    parser.add_argument(
        "--functional-test",
        action="store_true",
        help="run small execution/shape/dtype tests before benchmarking",
    )
    parser.add_argument(
        "--functional-only",
        action="store_true",
        help="run only the functional tests and skip timing",
    )
    parser.add_argument("--functional-experts", type=int, default=2)
    parser.add_argument("--functional-tokens", type=int, default=256)
    parser.add_argument("--functional-k", type=int, default=128)
    parser.add_argument("--functional-n", type=int, default=256)
    parser.add_argument("--functional-atol", type=float, default=0.5)
    parser.add_argument("--functional-rtol", type=float, default=0.5)
    parser.add_argument(
        "--functional-reference-backend",
        choices=("torch_compile", "torch"),
        default="torch",
        help="reference backend for cuDNN-specific GeGLU/DGeGLU functional checks",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "if set, write output_verbose.txt (full stdout), output.txt (device/version/"
            "dtype/shape/summary), and results.png (summary table) to this directory"
        ),
    )
    args = parser.parse_args()

    if args.output_dir is None:
        run_benchmarks(args)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    buffer = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, buffer)
    try:
        results = run_benchmarks(args)
    finally:
        sys.stdout = original_stdout

    write_output_files(
        output_dir=args.output_dir,
        verbose_text=buffer.getvalue(),
        results=results,
        args=args,
    )


if __name__ == "__main__":
    main()
