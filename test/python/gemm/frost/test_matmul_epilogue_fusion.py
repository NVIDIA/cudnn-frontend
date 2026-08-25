# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Epilogue-fusion correctness gate over (op × broadcast_mode × config × dtype),
plus multi-output (set_output taps) and fan-out/fan-in DAG epilogues.

Companion to test_matmul (pure matmul); this exercises epilogue codegen +
template integration. Small-integer inputs keep the FP32 reduction exact, so
atol=1e-1/rtol=1e-2 only has to absorb dtype downcast + transcendental ULPs.
Runs as a script too (forwards argv to pytest on this file).
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import sys
import textwrap
from dataclasses import dataclass
from typing import Callable

import pytest
import torch

from gemm_test_utils import (
    requires_sm100,
    Plan as _plan,
    vp as _vp,
    resolve as _resolve,
)

# Module-wide GPU gate — every test here is end-to-end and needs a B200.
pytestmark = [pytest.mark.L0, requires_sm100]


import cudnn
import cudnn.gemm.frost  # noqa: F401  — installs the cudnn.pygraph recorder hook
from cudnn.gemm.frost import compiler, epilogue_codegen
from cudnn.gemm.frost.epilogue_codegen import generate
from cudnn.gemm.frost.graph_analyzer import analyze
from cudnn.gemm.frost.tile_config import by_name

# Dtype tables (dup of test_matmul to keep the files independent)

_TORCH_DTYPE = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
    "fp8_e8m0": torch.float8_e8m0fnu,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int32": torch.int32,
}
_CUDNN_DTYPE = {
    "bf16": cudnn.data_type.BFLOAT16,
    "fp16": cudnn.data_type.HALF,
    "fp32": cudnn.data_type.FLOAT,
    "fp8_e4m3": cudnn.data_type.FP8_E4M3,
    "fp8_e5m2": cudnn.data_type.FP8_E5M2,
    "fp8_e8m0": cudnn.data_type.FP8_E8M0,
    "int8": cudnn.data_type.INT8,
    "uint8": cudnn.data_type.UINT8,
    "int32": cudnn.data_type.INT32,
}


# Chain spec — list of (op_name, broadcast_mode | None) tuples
Op = str  # cuDNN frontend method name, e.g. "relu", "add", "bias"
Bcast = str  # "scalar" | "per_row" | "per_col" | "per_elem"
Step = tuple[Op, Bcast | None]  # (op, bcast); bcast=None means unary
Chain = tuple[Step, ...]


def _chain_id(chain: Chain) -> str:
    parts = []
    for op, bcast in chain:
        parts.append(op if bcast is None else f"{op}_{bcast}")
    return "__".join(parts)


def _layout_id(p: tuple[str, str]) -> str:
    return f"A{p[0]}_B{p[1]}"


# Default axes — one config, one dtype, one shape per fusion-API case.
_DEFAULT_CONFIG = "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"
_DEFAULT_IN_DT = "bf16"
_DEFAULT_OUT_DT = "bf16"
_DEFAULT_SHAPE: tuple[int, int, int] = (256, 256, 128)
_DEFAULT_BATCHED_SHAPE: tuple[int, int, int, int] = (2, 256, 256, 128)
_INPUT_LAYOUTS: tuple[tuple[str, str], ...] = (
    ("k", "k"),
    ("m", "k"),
    ("k", "n"),
    ("m", "n"),
)
_NONPACKED_CONFIGS: tuple[str, ...] = (
    "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
    "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
    "CONFIG_sm100_256x128x128_128x128x32_cluster1x1_1ctamma",  # num_mma_m=2
)
_NONPACKED_AUX_BCAST_MODES: tuple[Bcast, ...] = ("per_col", "per_elem")


# Group A: every unary op (skip `exp` — magnitude overflow with small-int inputs)
_UNARY_CHAINS: tuple[Chain, ...] = tuple(
    ((op, None),)
    for op in (
        "identity",
        "relu",
        "gelu",
        "gelu_approx_tanh",
        "swish",
        "sigmoid",
        "tanh",
        "abs",
        "neg",
    )
)

# Group B: every binary op × every broadcast mode (skip `div` — div-by-zero)
_BCAST_MODES: tuple[Bcast, ...] = ("scalar", "per_row", "per_col", "per_elem")
_BINARY_CHAINS: tuple[Chain, ...] = tuple(((op, bcast),) for op in ("add", "mul", "sub") for bcast in _BCAST_MODES)

# `bias` uses a different cuDNN API method than `add` (separate recorder patch)
_BIAS_CHAINS: tuple[Chain, ...] = ((("bias", "per_col"),),)

# Group C: canonical multi-op chains
_MULTI_CHAINS: tuple[Chain, ...] = (
    (("bias", "per_col"), ("gelu_approx_tanh", None)),  # FFN canonical
    (("bias", "per_row"), ("swish", None)),  # per-row bias
    (("bias", "scalar"), ("relu", None)),  # scalar bias
    (("add", "per_elem"), ("sigmoid", None)),  # per-elem aux, biggest IO
    (("mul", "per_col"), ("tanh", None)),  # scale → tanh
)

_LAYOUT_CHAINS: tuple[Chain, ...] = (
    (("relu", None),),
    (("bias", "per_col"), ("gelu_approx_tanh", None)),
    (("add", "per_elem"), ("sigmoid", None)),
)

# Default-axis cases: chain varies, everything else fixed.
_DEFAULT_AXIS_CHAINS: tuple[Chain, ...] = _UNARY_CHAINS + _BINARY_CHAINS + _BIAS_CHAINS + _MULTI_CHAINS


# Cross-config subset: canonical chains × {cta2 basic, cta2 cluster-m=128}
_CROSS_CONFIG_NAMES: tuple[str, ...] = (
    "CONFIG_sm100_128x128x128_128x128x32_cluster2x1_2ctamma",  # cta_group=2 baseline
    "CONFIG_sm100_64x64x128_64x64x32_cluster2x4_2ctamma",  # cta2 cluster-m=128 (cta_tile_m=64)
    # CTA tile split across two MMA instructions along M. Per-row aux is
    # prefetched from `row`, which is rebound per M block, so the aux-view block
    # has to sit INSIDE that loop — this is what catches it if it drifts out.
    "CONFIG_sm100_256x128x128_128x128x32_cluster1x1_1ctamma",  # num_mma_m=2
    "CONFIG_sm100_128x128x128_64x128x32_cluster2x1_2ctamma",  # num_mma_m=2, 2x2 DP drain
)
_CROSS_CHAINS: tuple[Chain, ...] = (
    (("relu", None),),
    (("bias", "per_col"), ("gelu_approx_tanh", None)),
    (("mul", "per_col"), ("tanh", None)),
)


# Cross-dtype subset: canonical chains × {fp16, fp8_e4m3→fp16}
_CROSS_DTYPE_PAIRS: tuple[tuple[str, str], ...] = (
    ("fp16", "fp16"),
    ("fp8_e4m3", "fp16"),
)
_EPILOGUE_DTYPES: tuple[str, ...] = (
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_e8m0",
    "int8",
    "uint8",
    "bf16",
    "fp16",
    "fp32",
    "int32",
)


@dataclass(frozen=True)
class BatchedFusionCase:
    chain: Chain
    batched_aux: bool = False


_BATCHED_FUSION_CASES: tuple[BatchedFusionCase, ...] = (
    BatchedFusionCase((("relu", None),)),
    BatchedFusionCase((("bias", "per_col"), ("gelu_approx_tanh", None))),
    BatchedFusionCase((("add", "per_elem"), ("sigmoid", None))),
    BatchedFusionCase((("add", "scalar"), ("relu", None)), batched_aux=True),
    BatchedFusionCase((("bias", "per_row"), ("swish", None)), batched_aux=True),
    BatchedFusionCase((("sub", "per_col"),), batched_aux=True),
    BatchedFusionCase((("mul", "per_elem"), ("tanh", None)), batched_aux=True),
)


def _batched_case_id(case: BatchedFusionCase) -> str:
    aux = "batched_aux" if case.batched_aux else "shared_aux"
    return f"{_chain_id(case.chain)}__{aux}"


@dataclass(frozen=True)
class Rank3BroadcastCase:
    aux_shape: tuple[int, int, int]
    aux_on_rhs: bool


@dataclass(frozen=True)
class MixedDtypeBroadcastCase:
    aux_shape: tuple[int, int, int]
    aux_on_rhs: bool
    aux_dtype: str
    out_dtype: str


def _rank3_broadcast_case_id(case: Rank3BroadcastCase) -> str:
    parts = ["B" if case.aux_shape[0] != 1 else "1"]
    parts.append("M" if case.aux_shape[1] != 1 else "1")
    parts.append("N" if case.aux_shape[2] != 1 else "1")
    side = "rhs" if case.aux_on_rhs else "lhs"
    return f"{''.join(parts)}__{side}"


_RANK3_BROADCAST_CASES: tuple[Rank3BroadcastCase, ...] = tuple(
    Rank3BroadcastCase((b, m, n), aux_on_rhs)
    for aux_on_rhs in (True, False)
    for b in (1, _DEFAULT_BATCHED_SHAPE[0])
    for m in (1, _DEFAULT_BATCHED_SHAPE[1])
    for n in (1, _DEFAULT_BATCHED_SHAPE[2])
)


def _mixed_dtype_broadcast_case_id(case: MixedDtypeBroadcastCase) -> str:
    base = _rank3_broadcast_case_id(Rank3BroadcastCase(case.aux_shape, case.aux_on_rhs))
    return f"{base}__aux_{case.aux_dtype}__out_{case.out_dtype}"


_MIXED_DTYPE_BROADCAST_CASES: tuple[MixedDtypeBroadcastCase, ...] = (
    MixedDtypeBroadcastCase((1, 1, 1), True, "fp8_e4m3", "fp32"),
    MixedDtypeBroadcastCase((1, 1, 1), False, "int8", "bf16"),
    MixedDtypeBroadcastCase((_DEFAULT_BATCHED_SHAPE[0], 1, 1), True, "fp8_e5m2", "fp16"),
    MixedDtypeBroadcastCase((_DEFAULT_BATCHED_SHAPE[0], 1, 1), False, "uint8", "int32"),
    MixedDtypeBroadcastCase((1, _DEFAULT_BATCHED_SHAPE[1], 1), True, "fp8_e8m0", "fp8_e4m3"),
    MixedDtypeBroadcastCase((1, _DEFAULT_BATCHED_SHAPE[1], 1), False, "bf16", "fp8_e5m2"),
    MixedDtypeBroadcastCase((_DEFAULT_BATCHED_SHAPE[0], _DEFAULT_BATCHED_SHAPE[1], 1), True, "fp16", "uint8"),
    MixedDtypeBroadcastCase((_DEFAULT_BATCHED_SHAPE[0], _DEFAULT_BATCHED_SHAPE[1], 1), False, "fp32", "int8"),
    MixedDtypeBroadcastCase((1, 1, _DEFAULT_BATCHED_SHAPE[2]), True, "int32", "fp8_e8m0"),
    MixedDtypeBroadcastCase((1, 1, _DEFAULT_BATCHED_SHAPE[2]), False, "fp8_e4m3", "fp32"),
    MixedDtypeBroadcastCase((_DEFAULT_BATCHED_SHAPE[0], 1, _DEFAULT_BATCHED_SHAPE[2]), True, "int8", "bf16"),
    MixedDtypeBroadcastCase(
        (_DEFAULT_BATCHED_SHAPE[0], 1, _DEFAULT_BATCHED_SHAPE[2]),
        False,
        "fp8_e5m2",
        "fp16",
    ),
    MixedDtypeBroadcastCase(
        (1, _DEFAULT_BATCHED_SHAPE[1], _DEFAULT_BATCHED_SHAPE[2]),
        True,
        "uint8",
        "int32",
    ),
    MixedDtypeBroadcastCase(
        (1, _DEFAULT_BATCHED_SHAPE[1], _DEFAULT_BATCHED_SHAPE[2]),
        False,
        "fp8_e8m0",
        "fp8_e4m3",
    ),
    MixedDtypeBroadcastCase(
        (
            _DEFAULT_BATCHED_SHAPE[0],
            _DEFAULT_BATCHED_SHAPE[1],
            _DEFAULT_BATCHED_SHAPE[2],
        ),
        True,
        "bf16",
        "fp8_e5m2",
    ),
    MixedDtypeBroadcastCase(
        (
            _DEFAULT_BATCHED_SHAPE[0],
            _DEFAULT_BATCHED_SHAPE[1],
            _DEFAULT_BATCHED_SHAPE[2],
        ),
        False,
        "fp16",
        "uint8",
    ),
)


# Reference (torch fp32, then downcast)

_TORCH_UNARY: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "identity": lambda x: x,
    "relu": torch.relu,
    "gelu": lambda x: torch.nn.functional.gelu(x),  # erf-based
    "gelu_approx_tanh": lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
    "swish": torch.nn.functional.silu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "abs": torch.abs,
    "neg": torch.neg,
    "exp": torch.exp,
    "ceil": torch.ceil,
    "floor": torch.floor,
    "erf": torch.erf,
    "log": torch.log,
    "reciprocal": torch.reciprocal,
    "rsqrt": torch.rsqrt,
    "sqrt": torch.sqrt,
}

_TORCH_BINARY: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "add": lambda x, y: x + y,
    "bias": lambda x, y: x + y,
    "mul": lambda x, y: x * y,
    "sub": lambda x, y: x - y,
    "div": lambda x, y: x / y,
    "max": torch.maximum,
    "min": torch.minimum,
    "pow": torch.pow,
    "add_square": lambda x, y: x + y * y,
}


# Aux shape / runtime data helpers


def _aux_dim_stride(bcast: Bcast, M: int, N: int) -> tuple[list[int], list[int]]:
    """cuDNN dim+stride for an aux broadcasting in the given mode. The analyzer
    infers bcast_mode from these dims, so the shape drives the codegen branch."""
    if bcast == "scalar":
        return [1, 1], [1, 1]
    if bcast == "per_row":
        return [M, 1], [1, 1]
    if bcast == "per_col":
        return [1, N], [N, 1]
    if bcast == "per_elem":
        return [M, N], [N, 1]
    raise AssertionError(f"unknown bcast {bcast!r}")


def _aux_dim_stride_batched(bcast: Bcast, batch: int, M: int, N: int, batched_aux: bool) -> tuple[list[int], list[int]]:
    if not batched_aux:
        return _aux_dim_stride(bcast, M, N)
    if bcast == "scalar":
        return [batch, 1, 1], [1, 1, 1]
    if bcast == "per_row":
        return [batch, M, 1], [M, 1, 1]
    if bcast == "per_col":
        return [batch, 1, N], [N, N, 1]
    if bcast == "per_elem":
        return [batch, M, N], [M * N, N, 1]
    raise AssertionError(f"unknown bcast {bcast!r}")


def _rank3_aux_dim_stride(
    aux_shape: tuple[int, int, int],
) -> tuple[list[int], list[int]]:
    b, m, n = aux_shape
    return [b, m, n], [m * n, n, 1]


def _nonpacked_aux_dim_stride(bcast: Bcast, batch: int, M: int, N: int) -> tuple[list[int], list[int]]:
    pad = 16
    if bcast == "per_col":
        return [batch, 1, N], [N + pad, N + pad, 1]
    if bcast == "per_elem":
        return [batch, M, N], [M * (N + pad), N + pad, 1]
    raise ValueError(f"unsupported nonpacked aux bcast {bcast!r}")


def _mkaux(bcast: Bcast, M: int, N: int, dtype: str, seed: int) -> torch.Tensor:
    """Small-integer aux; range matches `_mkdata` so the FP32 reduction is exact."""
    torch.manual_seed(seed)
    if bcast == "scalar":
        shape = (1, 1)
    elif bcast == "per_row":
        shape = (M, 1)
    elif bcast == "per_col":
        shape = (1, N)
    elif bcast == "per_elem":
        shape = (M, N)
    else:
        raise AssertionError(f"unknown bcast {bcast!r}")
    return torch.empty(*shape, dtype=torch.int32).random_(-2, 2).to(dtype=_TORCH_DTYPE[dtype], device="cuda")


def _mkaux_batched(
    bcast: Bcast,
    batch: int,
    M: int,
    N: int,
    dtype: str,
    seed: int,
    batched_aux: bool,
) -> torch.Tensor:
    if not batched_aux:
        return _mkaux(bcast, M, N, dtype, seed)

    torch.manual_seed(seed)
    if bcast == "scalar":
        shape = (batch, 1, 1)
    elif bcast == "per_row":
        shape = (batch, M, 1)
    elif bcast == "per_col":
        shape = (batch, 1, N)
    elif bcast == "per_elem":
        shape = (batch, M, N)
    else:
        raise AssertionError(f"unknown bcast {bcast!r}")
    return torch.empty(*shape, dtype=torch.int32).random_(-2, 2).to(dtype=_TORCH_DTYPE[dtype], device="cuda")


def _mkaux_rank3_broadcast(
    aux_shape: tuple[int, int, int],
    dtype: str,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.empty(*aux_shape, dtype=torch.int32).random_(-2, 2).to(dtype=_TORCH_DTYPE[dtype], device="cuda")


def _mkaux_nonpacked(bcast: Bcast, batch: int, M: int, N: int, dtype: str, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    pad = 16
    if bcast == "per_col":
        storage = torch.empty(batch, 1, N + pad, dtype=torch.int32, device="cuda").random_(-2, 2)
    elif bcast == "per_elem":
        storage = torch.empty(batch, M, N + pad, dtype=torch.int32, device="cuda").random_(-2, 2)
    else:
        raise ValueError(f"unsupported nonpacked aux bcast {bcast!r}")
    storage = storage.to(dtype=_TORCH_DTYPE[dtype])
    return storage[:, :, :N]


def _mkaux_zero_stride_per_elem(batch: int, M: int, N: int, dtype: str, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    base = torch.empty(N, dtype=torch.int32, device="cuda").random_(-2, 2).to(dtype=_TORCH_DTYPE[dtype])
    return torch.as_strided(base, (batch, M, N), (0, 0, 1))


def _mkdata(
    M: int,
    N: int,
    K: int,
    in_dt: str,
    out_dt: str,
    seed: int = 0,
    a_major: str = "k",
    b_major: str = "k",
    out_major: str = "n",
):
    """A, B, C tensors (rank-3, batch=1). Small-integer inputs ⇒ exact FP32 matmul."""
    torch.manual_seed(seed)
    rng = (-3, 3) if in_dt.startswith("fp8") else (-2, 2)
    a_shape = (1, M, K) if a_major == "k" else (1, K, M)
    b_shape = (1, N, K) if b_major == "k" else (1, K, N)
    a = torch.empty(*a_shape, dtype=torch.int32).random_(*rng).to(dtype=_TORCH_DTYPE[in_dt], device="cuda")
    b = torch.empty(*b_shape, dtype=torch.int32).random_(*rng).to(dtype=_TORCH_DTYPE[in_dt], device="cuda")
    if a_major == "m":
        a = a.transpose(1, 2)
    if b_major == "n":
        b = b.transpose(1, 2)
    if out_major == "m":
        c = torch.empty(1, N, M, dtype=_TORCH_DTYPE[out_dt], device="cuda").transpose(1, 2)
    else:
        c = torch.empty(1, M, N, dtype=_TORCH_DTYPE[out_dt], device="cuda")
    return a, b, c


def _mkbatched_data(
    batch: int,
    M: int,
    N: int,
    K: int,
    in_dt: str,
    out_dt: str,
    seed: int = 0,
    a_major: str = "k",
    b_major: str = "k",
):
    torch.manual_seed(seed)
    rng = (-3, 3) if in_dt.startswith("fp8") else (-2, 2)
    a_shape = (batch, M, K) if a_major == "k" else (batch, K, M)
    b_shape = (batch, N, K) if b_major == "k" else (batch, K, N)
    a = torch.empty(*a_shape, dtype=torch.int32).random_(*rng).to(dtype=_TORCH_DTYPE[in_dt], device="cuda")
    b = torch.empty(*b_shape, dtype=torch.int32).random_(*rng).to(dtype=_TORCH_DTYPE[in_dt], device="cuda")
    if a_major == "m":
        a = a.transpose(1, 2)
    if b_major == "n":
        b = b.transpose(1, 2)
    c = torch.empty(batch, M, N, dtype=_TORCH_DTYPE[out_dt], device="cuda")
    return a, b, c


def _mkbatched_nonpacked_data(
    batch: int,
    M: int,
    N: int,
    K: int,
    in_dt: str,
    out_dt: str,
    seed: int = 0,
):
    torch.manual_seed(seed)
    rng = (-3, 3) if in_dt.startswith("fp8") else (-2, 2)
    pad = 16
    a_storage = torch.empty(batch, M, K + pad, dtype=torch.int32, device="cuda").random_(*rng)
    a_storage = a_storage.to(dtype=_TORCH_DTYPE[in_dt])
    b_storage = torch.empty(batch, N, K + pad, dtype=torch.int32, device="cuda").random_(*rng)
    b_storage = b_storage.to(dtype=_TORCH_DTYPE[in_dt])
    c_storage = torch.empty(batch, M, N + pad, dtype=_TORCH_DTYPE[out_dt], device="cuda")
    a = a_storage[:, :, :K]
    b = b_storage[:, :, :K]
    c = c_storage[:, :, :N]
    return a, b, c


def _a_stride_batched(M: int, K: int, a_major: str) -> list[int]:
    return [M * K, K, 1] if a_major == "k" else [M * K, 1, M]


def _b_stride_batched(N: int, K: int, b_major: str) -> list[int]:
    return [N * K, 1, K] if b_major == "k" else [N * K, N, 1]


def _apply_unary(g: cudnn.pygraph, op: str, cur, name: str):
    return getattr(g, op)(input=cur, name=name)


def _apply_binary(g: cudnn.pygraph, op: str, lhs, rhs, name: str):
    if op == "bias":
        return g.bias(input=lhs, bias=rhs, name=name)
    if op in {"max", "min", "pow"}:
        return getattr(g, op)(input0=lhs, input1=rhs, name=name)
    return getattr(g, op)(a=lhs, b=rhs, name=name)


# Graph builder — drives the cuDNN frontend API + the recorder


def _build_graph(
    M: int,
    N: int,
    K: int,
    in_dt: str,
    out_dt: str,
    chain: Chain,
    a_major: str = "k",
    b_major: str = "k",
    out_major: str = "n",
) -> tuple[cudnn.pygraph, list[str]]:
    """Build a `cudnn.pygraph` for a matmul + the given chain. Returns
    (graph, aux_names); aux_names is the in-chain order the call site must
    pass aux in (matching chain.aux_tensors)."""
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[in_dt],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=_a_stride_batched(M, K, a_major))
    B = g.tensor(name="B", dim=[1, K, N], stride=_b_stride_batched(N, K, b_major))
    cur = g.matmul(A=A, B=B, name="mm")

    aux_names: list[str] = []
    for i, (op, bcast) in enumerate(chain):
        if bcast is None:
            cur = _apply_unary(g, op, cur, f"u{i}")
        else:
            dim, stride = _aux_dim_stride(bcast, M, N)
            aux_name = f"aux{i}"
            aux = g.tensor(name=aux_name, dim=dim, stride=stride)
            aux_names.append(aux_name)
            cur = _apply_binary(g, op, cur, aux, f"o{i}")
    if out_major == "m":
        cur.set_stride([M * N, 1, M])
    cur.set_output(True)
    if out_dt != in_dt:
        cur.set_data_type(_CUDNN_DTYPE[out_dt])
    return g, aux_names


def _build_nonpacked_epilogue_graph(
    batch: int,
    M: int,
    N: int,
    K: int,
    in_dt: str,
    out_dt: str,
    aux_bcast: Bcast,
) -> tuple[cudnn.pygraph, list[str]]:
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[in_dt],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[batch, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[batch, K, N], stride=_b_stride_batched(N, K, "k"))
    aux_dim, aux_stride = _nonpacked_aux_dim_stride(aux_bcast, batch, M, N)
    aux = g.tensor(name="aux0", dim=aux_dim, stride=aux_stride)
    cur = g.matmul(A=A, B=B, name="mm")
    cur = g.add(a=cur, b=aux, name="add")
    cur = g.relu(input=cur, name="relu")
    cur.set_output(True)
    if out_dt != in_dt:
        cur.set_data_type(_CUDNN_DTYPE[out_dt])
    return g, ["aux0"]


def _build_zero_stride_aux_epilogue_graph(
    batch: int,
    M: int,
    N: int,
    K: int,
    in_dt: str,
    out_dt: str,
) -> tuple[cudnn.pygraph, list[str]]:
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[in_dt],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[batch, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[batch, K, N], stride=_b_stride_batched(N, K, "k"))
    aux = g.tensor(name="aux0", dim=[batch, M, N], stride=[0, 0, 1])
    cur = g.matmul(A=A, B=B, name="mm")
    cur = g.add(a=cur, b=aux, name="add")
    cur = g.relu(input=cur, name="relu")
    cur.set_output(True)
    if out_dt != in_dt:
        cur.set_data_type(_CUDNN_DTYPE[out_dt])
    return g, ["aux0"]


def _build_batched_graph(
    batch: int,
    M: int,
    N: int,
    K: int,
    in_dt: str,
    out_dt: str,
    chain: Chain,
    *,
    batched_aux: bool,
    a_major: str = "k",
    b_major: str = "k",
) -> tuple[cudnn.pygraph, list[str]]:
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[in_dt],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[batch, M, K], stride=_a_stride_batched(M, K, a_major))
    B = g.tensor(name="B", dim=[batch, K, N], stride=_b_stride_batched(N, K, b_major))
    cur = g.matmul(A=A, B=B, name="mm")

    aux_names: list[str] = []
    for i, (op, bcast) in enumerate(chain):
        if bcast is None:
            cur = _apply_unary(g, op, cur, f"u{i}")
        else:
            dim, stride = _aux_dim_stride_batched(bcast, batch, M, N, batched_aux)
            aux_name = f"aux{i}"
            aux = g.tensor(name=aux_name, dim=dim, stride=stride)
            aux_names.append(aux_name)
            cur = _apply_binary(g, op, cur, aux, f"o{i}")
    cur.set_output(True)
    if out_dt != in_dt:
        cur.set_data_type(_CUDNN_DTYPE[out_dt])
    return g, aux_names


def _build_rank3_broadcast_graph(
    batch: int,
    M: int,
    N: int,
    K: int,
    in_dt: str,
    out_dt: str,
    case: Rank3BroadcastCase,
) -> tuple[cudnn.pygraph, list[str]]:
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[in_dt],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[batch, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[batch, K, N], stride=_b_stride_batched(N, K, "k"))
    aux_dim, aux_stride = _rank3_aux_dim_stride(case.aux_shape)
    aux = g.tensor(name="aux0", dim=aux_dim, stride=aux_stride)
    cur = g.matmul(A=A, B=B, name="mm")
    if case.aux_on_rhs:
        cur = g.sub(a=cur, b=aux, name="sub")
    else:
        cur = g.sub(a=aux, b=cur, name="sub")
    cur.set_output(True)
    if out_dt != in_dt:
        cur.set_data_type(_CUDNN_DTYPE[out_dt])
    return g, ["aux0"]


def _build_mixed_dtype_broadcast_graph(
    batch: int,
    M: int,
    N: int,
    K: int,
    case: MixedDtypeBroadcastCase,
) -> cudnn.pygraph:
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[_DEFAULT_IN_DT],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[batch, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[batch, K, N], stride=_b_stride_batched(N, K, "k"))
    aux_dim, aux_stride = _rank3_aux_dim_stride(case.aux_shape)
    aux = g.tensor(
        name="aux0",
        dim=aux_dim,
        stride=aux_stride,
        data_type=_CUDNN_DTYPE[case.aux_dtype],
    )
    cur = g.matmul(A=A, B=B, name="mm")
    if case.aux_on_rhs:
        cur = g.add(a=cur, b=aux, name="add")
    else:
        cur = g.add(a=aux, b=cur, name="add")
    cur.set_output(True).set_data_type(_CUDNN_DTYPE[case.out_dtype])
    return g


def _build_epilogue_dtype_graph(
    M: int,
    N: int,
    K: int,
    aux_dtype: str,
    out_dtype: str,
) -> cudnn.pygraph:
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[_DEFAULT_IN_DT],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    aux = g.tensor(
        name="aux0",
        dim=[1, 1, N],
        stride=[N, N, 1],
        data_type=_CUDNN_DTYPE[aux_dtype],
    )
    cur = g.matmul(A=A, B=B, name="mm")
    cur = g.add(a=cur, b=aux, name="add")
    cur.set_output(True).set_data_type(_CUDNN_DTYPE[out_dtype])
    return g


def _reference(
    a: torch.Tensor,
    b: torch.Tensor,
    aux_runtime: dict[str, torch.Tensor],
    chain: Chain,
    out_dt: str,
) -> torch.Tensor:
    cur = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    for i, (op, bcast) in enumerate(chain):
        if bcast is None:
            cur = _TORCH_UNARY[op](cur)
        else:
            aux_t = aux_runtime[f"aux{i}"].to(torch.float32)
            cur = _TORCH_BINARY[op](cur, aux_t)
    return cur.to(_TORCH_DTYPE[out_dt])


def _rank3_broadcast_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    aux: torch.Tensor,
    out_dt: str,
    aux_on_rhs: bool,
) -> torch.Tensor:
    cur = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    aux_f32 = aux.to(torch.float32)
    if aux_on_rhs:
        cur = cur - aux_f32
    else:
        cur = aux_f32 - cur
    return cur.to(_TORCH_DTYPE[out_dt])


# Config lookup


# Test bodies — each parametrized axis is a separate test for clear IDs


def _run_case(
    M: int,
    N: int,
    K: int,
    in_dt: str,
    out_dt: str,
    config_name: str,
    chain: Chain,
    a_major: str = "k",
    b_major: str = "k",
    out_major: str = "n",
) -> None:
    """Common run/check body: build graph, JIT, launch, compare to torch."""
    cfg, cta_group = _resolve(config_name)
    g, aux_names = _build_graph(M, N, K, in_dt, out_dt, chain, a_major, b_major, out_major)
    try:
        compiled = _plan(g, config=cfg, cta_group=cta_group)
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else ""
        pytest.fail(f"JIT compile failed: {type(e).__name__}: {first[:200]}", pytrace=False)

    a, b, c = _mkdata(
        M,
        N,
        K,
        in_dt,
        out_dt,
        seed=0,
        a_major=a_major,
        b_major=b_major,
        out_major=out_major,
    )
    aux_runtime: dict[str, torch.Tensor] = {}
    for i, (op, bcast) in enumerate(chain):
        if bcast is None:
            continue
        aux_runtime[f"aux{i}"] = _mkaux(bcast, M, N, in_dt, seed=10 + i)

    compiled(_vp(compiled, a, b, c, *[aux_runtime[n] for n in aux_names]))
    torch.cuda.synchronize()

    ref = _reference(a, b, aux_runtime, chain, out_dt)
    try:
        torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)
    except AssertionError as e:
        diff = (c.to(torch.float32) - ref.to(torch.float32)).abs()
        bad = int((diff > 1e-1).sum().item())
        pytest.fail(
            f"\n  config:    {config_name}"
            f"\n  dtype:     {in_dt} -> {out_dt}"
            f"\n  shape:     {M}x{N}x{K}"
            f"\n  layout:    A{a_major}/B{b_major}"
            f"\n  chain:     {_chain_id(chain)}"
            f"\n  bad:       {bad}/{diff.numel()} ({100 * bad / diff.numel():.2f}%)"
            f"\n  max|diff|: {float(diff.max().item()):.4g}"
            f"\n  max|ref|:  {float(ref.abs().max().item()):.4g}"
            f"\n  sample c[0,0,:8]   = {c[0, 0, :8].to(torch.float32).tolist()}"
            f"\n  sample ref[0,0,:8] = {ref[0, 0, :8].to(torch.float32).tolist()}"
            f"\n  inner: {str(e).splitlines()[0]}",
            pytrace=False,
        )


def _run_batched_case(
    batch: int,
    M: int,
    N: int,
    K: int,
    in_dt: str,
    out_dt: str,
    config_name: str,
    case: BatchedFusionCase,
) -> None:
    cfg, cta_group = _resolve(config_name)
    g, aux_names = _build_batched_graph(batch, M, N, K, in_dt, out_dt, case.chain, batched_aux=case.batched_aux)
    try:
        compiled = _plan(g, config=cfg, cta_group=cta_group)
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else ""
        pytest.fail(f"JIT compile failed: {type(e).__name__}: {first[:200]}", pytrace=False)

    a, b, c = _mkbatched_data(batch, M, N, K, in_dt, out_dt, seed=0)
    aux_runtime: dict[str, torch.Tensor] = {}
    for i, (op, bcast) in enumerate(case.chain):
        if bcast is None:
            continue
        aux_runtime[f"aux{i}"] = _mkaux_batched(
            bcast,
            batch,
            M,
            N,
            in_dt,
            seed=10 + i,
            batched_aux=case.batched_aux,
        )

    compiled(_vp(compiled, a, b, c, *[aux_runtime[n] for n in aux_names]))
    torch.cuda.synchronize()

    ref = _reference(a, b, aux_runtime, case.chain, out_dt)
    try:
        torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)
    except AssertionError as e:
        diff = (c.to(torch.float32) - ref.to(torch.float32)).abs()
        bad = int((diff > 1e-1).sum().item())
        pytest.fail(
            f"\n  config:    {config_name}"
            f"\n  dtype:     {in_dt} -> {out_dt}"
            f"\n  shape:     B{batch} {M}x{N}x{K}"
            f"\n  chain:     {_batched_case_id(case)}"
            f"\n  bad:       {bad}/{diff.numel()} ({100 * bad / diff.numel():.2f}%)"
            f"\n  max|diff|: {float(diff.max().item()):.4g}"
            f"\n  max|ref|:  {float(ref.abs().max().item()):.4g}"
            f"\n  sample c[0,0,:8]   = {c[0, 0, :8].to(torch.float32).tolist()}"
            f"\n  sample ref[0,0,:8] = {ref[0, 0, :8].to(torch.float32).tolist()}"
            f"\n  inner: {str(e).splitlines()[0]}",
            pytrace=False,
        )


def _run_rank3_broadcast_case(case: Rank3BroadcastCase) -> None:
    batch, M, N, K = _DEFAULT_BATCHED_SHAPE
    cfg, cta_group = _resolve(_DEFAULT_CONFIG)
    g, aux_names = _build_rank3_broadcast_graph(batch, M, N, K, _DEFAULT_IN_DT, _DEFAULT_OUT_DT, case)
    try:
        compiled = _plan(g, config=cfg, cta_group=cta_group)
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else ""
        pytest.fail(f"JIT compile failed: {type(e).__name__}: {first[:200]}", pytrace=False)

    a, b, c = _mkbatched_data(batch, M, N, K, _DEFAULT_IN_DT, _DEFAULT_OUT_DT, seed=0)
    aux = _mkaux_rank3_broadcast(case.aux_shape, _DEFAULT_IN_DT, seed=23)
    compiled(_vp(compiled, a, b, c, *[aux for _ in aux_names]))
    torch.cuda.synchronize()

    ref = _rank3_broadcast_reference(a, b, aux, _DEFAULT_OUT_DT, case.aux_on_rhs)
    try:
        torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)
    except AssertionError as e:
        diff = (c.to(torch.float32) - ref.to(torch.float32)).abs()
        bad = int((diff > 1e-1).sum().item())
        pytest.fail(
            f"\n  config:    {_DEFAULT_CONFIG}"
            f"\n  dtype:     {_DEFAULT_IN_DT} -> {_DEFAULT_OUT_DT}"
            f"\n  shape:     B{batch} {M}x{N}x{K}"
            f"\n  aux shape: {case.aux_shape}"
            f"\n  aux side:  {'rhs' if case.aux_on_rhs else 'lhs'}"
            f"\n  bad:       {bad}/{diff.numel()} ({100 * bad / diff.numel():.2f}%)"
            f"\n  max|diff|: {float(diff.max().item()):.4g}"
            f"\n  max|ref|:  {float(ref.abs().max().item()):.4g}"
            f"\n  sample c[0,0,:8]   = {c[0, 0, :8].to(torch.float32).tolist()}"
            f"\n  sample ref[0,0,:8] = {ref[0, 0, :8].to(torch.float32).tolist()}"
            f"\n  inner: {str(e).splitlines()[0]}",
            pytrace=False,
        )


def _run_epilogue_dtype_case(aux_dtype: str, out_dtype: str) -> None:
    M, N, K = 128, 128, 128
    cfg, cta_group = _resolve(_DEFAULT_CONFIG)
    g = _build_epilogue_dtype_graph(M, N, K, aux_dtype, out_dtype)
    try:
        compiled = _plan(g, config=cfg, cta_group=cta_group, force_stg_epi=True)
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else ""
        pytest.fail(f"JIT compile failed: {type(e).__name__}: {first[:200]}", pytrace=False)

    a = torch.zeros(1, M, K, dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")
    b = torch.zeros(1, N, K, dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")
    c = torch.empty(1, M, N, dtype=_TORCH_DTYPE[out_dtype], device="cuda")
    aux_i32 = torch.arange(N, dtype=torch.int32, device="cuda") % 2 + 1
    aux = aux_i32.reshape(1, 1, N).to(_TORCH_DTYPE[aux_dtype])

    compiled(_vp(compiled, a, b, c, aux))
    torch.cuda.synchronize()

    ref = aux.to(torch.float32).expand(1, M, N).to(_TORCH_DTYPE[out_dtype])
    # Exact compare: test values 1/2 round-trip exactly through the covered dtypes.
    torch.testing.assert_close(
        c.to(torch.float32),
        ref.to(torch.float32),
        atol=0,
        rtol=0,
    )


def _run_mixed_dtype_broadcast_case(case: MixedDtypeBroadcastCase) -> None:
    batch, M, N, K = _DEFAULT_BATCHED_SHAPE
    cfg, cta_group = _resolve(_DEFAULT_CONFIG)
    g = _build_mixed_dtype_broadcast_graph(batch, M, N, K, case)
    try:
        compiled = _plan(g, config=cfg, cta_group=cta_group, force_stg_epi=True)
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else ""
        pytest.fail(f"JIT compile failed: {type(e).__name__}: {first[:200]}", pytrace=False)

    a = torch.zeros(batch, M, K, dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")
    b = torch.zeros(batch, N, K, dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")
    row_bit = torch.arange(M, dtype=torch.int32, device="cuda") % 2
    col_bit = torch.arange(N, dtype=torch.int32, device="cuda") % 2
    batch_bit = torch.arange(batch, dtype=torch.int32, device="cuda") % 2
    a[:, :, 0] = (row_bit.unsqueeze(0) + batch_bit.unsqueeze(1)).to(_TORCH_DTYPE[_DEFAULT_IN_DT])
    b[:, :, 0] = 1
    a[:, :, 1] = 1
    b[:, :, 1] = col_bit.unsqueeze(0).to(_TORCH_DTYPE[_DEFAULT_IN_DT])
    if case.out_dtype == "fp8_e8m0":
        # e8m0 has no mantissa; use exactly-representable 1s to isolate dtype
        # plumbing + broadcast indexing.
        a.zero_()
        b.zero_()
        a[:, :, 0] = 1
        b[:, :, 0] = 1
    c = torch.empty(batch, M, N, dtype=_TORCH_DTYPE[case.out_dtype], device="cuda")
    aux_elems = case.aux_shape[0] * case.aux_shape[1] * case.aux_shape[2]
    if case.out_dtype == "fp8_e8m0":
        aux_i32 = torch.ones(aux_elems, dtype=torch.int32, device="cuda")
    else:
        aux_i32 = torch.arange(aux_elems, dtype=torch.int32, device="cuda") % 2 + 1
    aux = aux_i32.reshape(case.aux_shape).to(_TORCH_DTYPE[case.aux_dtype])

    assert compiled.chain.aux_tensors[0].dtype == case.aux_dtype
    assert compiled.chain.output_dtype == case.out_dtype
    assert compiled.chain.ops[0].aux_on_rhs is case.aux_on_rhs

    compiled(_vp(compiled, a, b, c, aux))
    torch.cuda.synchronize()

    ref = (torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32)) + aux.to(torch.float32)).to(_TORCH_DTYPE[case.out_dtype])
    # Exact compare: test data round-trips exactly for the chosen output dtype.
    torch.testing.assert_close(
        c.to(torch.float32),
        ref.to(torch.float32),
        atol=0,
        rtol=0,
    )


# Group A+B+bias+multi: chain varies, everything else fixed.
@pytest.mark.parametrize(
    "chain",
    _DEFAULT_AXIS_CHAINS,
    ids=[_chain_id(c) for c in _DEFAULT_AXIS_CHAINS],
)
def test_fusion_default_axis(chain: Chain) -> None:
    """API-surface gate: one case per (op, broadcast_mode) cell, default axes."""
    M, N, K = _DEFAULT_SHAPE
    _run_case(M, N, K, _DEFAULT_IN_DT, _DEFAULT_OUT_DT, _DEFAULT_CONFIG, chain)


# Representative M-major chains: unary and canonical aux+unary.
_M_MAJOR_CHAINS: tuple[Chain, ...] = (
    (("relu", None),),
    (("bias", "per_col"), ("gelu_approx_tanh", None)),
)


@pytest.mark.parametrize(
    "chain",
    _M_MAJOR_CHAINS,
    ids=[_chain_id(c) for c in _M_MAJOR_CHAINS],
)
def test_fusion_m_major(chain: Chain) -> None:
    """Epilogue fusion with M-major output."""
    M, N, K = _DEFAULT_SHAPE
    _run_case(M, N, K, _DEFAULT_IN_DT, _DEFAULT_OUT_DT, _DEFAULT_CONFIG, chain, out_major="m")


def test_rank1_per_col_fusion() -> None:
    """Rank-1 [N] aux broadcasts across M and must index with col_j."""
    M, N, K = _DEFAULT_SHAPE
    cfg, cta_group = _resolve(_DEFAULT_CONFIG)
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[_DEFAULT_IN_DT],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    bias = g.tensor(name="bias", dim=[N], stride=[1])
    cur = g.matmul(A=A, B=B, name="mm")
    cur = g.bias(input=cur, bias=bias, name="b")
    cur.set_output(True)

    compiled = _plan(g, config=cfg, cta_group=cta_group)
    a, b, c = _mkdata(M, N, K, _DEFAULT_IN_DT, _DEFAULT_OUT_DT, seed=0)
    bias_runtime = (torch.arange(N, dtype=torch.int32) % 5 - 2).to(dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")

    compiled(_vp(compiled, a, b, c, bias_runtime))
    torch.cuda.synchronize()

    ref = (torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32)) + bias_runtime.to(torch.float32)).to(_TORCH_DTYPE[_DEFAULT_OUT_DT])
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)


def test_rank1_scalar_fusion() -> None:
    """Rank-1 [1] aux broadcasts across all output elements."""
    M, N, K = _DEFAULT_SHAPE
    cfg, cta_group = _resolve(_DEFAULT_CONFIG)
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[_DEFAULT_IN_DT],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    scale = g.tensor(name="scale", dim=[1], stride=[1])
    cur = g.matmul(A=A, B=B, name="mm")
    cur = g.add(a=cur, b=scale, name="add")
    cur.set_output(True)

    compiled = _plan(g, config=cfg, cta_group=cta_group)
    a, b, c = _mkdata(M, N, K, _DEFAULT_IN_DT, _DEFAULT_OUT_DT, seed=0)
    scale_runtime = torch.tensor([2], dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")

    compiled(_vp(compiled, a, b, c, scale_runtime))
    torch.cuda.synchronize()

    ref = (torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32)) + scale_runtime.to(torch.float32)).to(_TORCH_DTYPE[_DEFAULT_OUT_DT])
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)


def test_batched_rank1_per_col_fusion() -> None:
    """Rank-1 [N] aux broadcasts across batch and M for rank-3 matmul output."""
    batch, M, N, K = _DEFAULT_BATCHED_SHAPE
    cfg, cta_group = _resolve(_DEFAULT_CONFIG)
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[_DEFAULT_IN_DT],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[batch, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[batch, K, N], stride=[K * N, 1, K])
    bias = g.tensor(name="bias", dim=[N], stride=[1])
    cur = g.matmul(A=A, B=B, name="mm")
    cur = g.bias(input=cur, bias=bias, name="b")
    cur.set_output(True)

    compiled = _plan(g, config=cfg, cta_group=cta_group)
    a, b, c = _mkbatched_data(batch, M, N, K, _DEFAULT_IN_DT, _DEFAULT_OUT_DT, seed=0)
    bias_runtime = (torch.arange(N, dtype=torch.int32) % 5 - 2).to(dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")

    compiled(_vp(compiled, a, b, c, bias_runtime))
    torch.cuda.synchronize()

    ref = (torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32)) + bias_runtime.to(torch.float32)).to(_TORCH_DTYPE[_DEFAULT_OUT_DT])
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    "case",
    _RANK3_BROADCAST_CASES,
    ids=[_rank3_broadcast_case_id(c) for c in _RANK3_BROADCAST_CASES],
)
def test_rank3_batched_broadcast_patterns(case: Rank3BroadcastCase) -> None:
    """Rank-3 aux, every [B/1, M/1, N/1] broadcast pattern. `sub` (non-commutative)
    covers both aux-on-RHS and aux-on-LHS to check port direction is preserved."""
    _run_rank3_broadcast_case(case)


@pytest.mark.parametrize(
    "aux_bcast",
    _NONPACKED_AUX_BCAST_MODES,
    ids=list(_NONPACKED_AUX_BCAST_MODES),
)
@pytest.mark.parametrize(
    "config_name",
    _NONPACKED_CONFIGS,
    ids=[n.removeprefix("CONFIG_") for n in _NONPACKED_CONFIGS],
)
def test_nonpacked_epilogue_aux_and_output(
    config_name: str,
    aux_bcast: Bcast,
) -> None:
    batch, M, N, K = 2, 256, 256, 256
    in_dt = out_dt = "bf16"
    cfg, cta_group = _resolve(config_name)
    g, aux_names = _build_nonpacked_epilogue_graph(batch, M, N, K, in_dt, out_dt, aux_bcast)
    try:
        compiled = _plan(g, config=cfg, cta_group=cta_group)
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else ""
        pytest.fail(
            f"JIT compile failed: {type(e).__name__}: {first[:200]}",
            pytrace=False,
        )

    a, b, c = _mkbatched_nonpacked_data(batch, M, N, K, in_dt, out_dt, seed=0)
    aux = _mkaux_nonpacked(aux_bcast, batch, M, N, in_dt, seed=13)
    assert not a.is_contiguous() and not b.is_contiguous()
    assert not c.is_contiguous() and not aux.is_contiguous()

    compiled(_vp(compiled, a, b, c, *[aux for _ in aux_names]))
    torch.cuda.synchronize()

    ref = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    ref = torch.relu(ref + aux.to(torch.float32)).to(_TORCH_DTYPE[out_dt])
    try:
        torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)
    except AssertionError as e:
        diff = (c.to(torch.float32) - ref.to(torch.float32)).abs()
        bad = int((diff > 1e-1).sum().item())
        pytest.fail(
            f"\n  config:    {config_name}"
            f"\n  aux bcast: {aux_bcast}"
            f"\n  strides:   A{tuple(a.stride())} B{tuple(b.stride())} "
            f"C{tuple(c.stride())} aux{tuple(aux.stride())}"
            f"\n  bad:       {bad}/{diff.numel()} ({100 * bad / diff.numel():.2f}%)"
            f"\n  max|diff|: {float(diff.max().item()):.4g}"
            f"\n  inner:     {str(e).splitlines()[0]}",
            pytrace=False,
        )


@pytest.mark.parametrize(
    "config_name",
    _NONPACKED_CONFIGS,
    ids=[n.removeprefix("CONFIG_") for n in _NONPACKED_CONFIGS],
)
def test_zero_stride_epilogue_aux_broadcast(
    config_name: str,
) -> None:
    batch, M, N, K = 2, 256, 256, 256
    in_dt = out_dt = "bf16"
    cfg, cta_group = _resolve(config_name)
    g, aux_names = _build_zero_stride_aux_epilogue_graph(batch, M, N, K, in_dt, out_dt)
    compiled = _plan(g, config=cfg, cta_group=cta_group)

    a, b, c = _mkbatched_nonpacked_data(batch, M, N, K, in_dt, out_dt, seed=0)
    aux = _mkaux_zero_stride_per_elem(batch, M, N, in_dt, seed=13)
    assert aux.stride() == (0, 0, 1)
    assert not c.is_contiguous()

    compiled(_vp(compiled, a, b, c, *[aux for _ in aux_names]))
    torch.cuda.synchronize()

    ref = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    ref = torch.relu(ref + aux.to(torch.float32)).to(_TORCH_DTYPE[out_dt])
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)


# Cross-config subset.
@pytest.mark.parametrize(
    "chain",
    _CROSS_CHAINS,
    ids=[_chain_id(c) for c in _CROSS_CHAINS],
)
@pytest.mark.parametrize(
    "config_name",
    _CROSS_CONFIG_NAMES,
    ids=[n.replace("CONFIG_", "").replace("_sm100", "") for n in _CROSS_CONFIG_NAMES],
)
def test_fusion_cross_config(config_name: str, chain: Chain) -> None:
    """Canonical chains on non-default configs — catches template-side bugs
    (e.g. the 2×2 DP TMEM layout for cluster-m=128)."""
    M, N, K = _DEFAULT_SHAPE
    _run_case(M, N, K, _DEFAULT_IN_DT, _DEFAULT_OUT_DT, config_name, chain)


# Cross-dtype subset.
@pytest.mark.parametrize(
    "chain",
    _CROSS_CHAINS,
    ids=[_chain_id(c) for c in _CROSS_CHAINS],
)
@pytest.mark.parametrize(
    "in_dt,out_dt",
    _CROSS_DTYPE_PAIRS,
    ids=[f"{p[0]}->{p[1]}" for p in _CROSS_DTYPE_PAIRS],
)
def test_fusion_cross_dtype(in_dt: str, out_dt: str, chain: Chain) -> None:
    """Canonical chains with FP16 and FP8→FP16 — catches dtype-specific codegen bugs."""
    M, N, K = _DEFAULT_SHAPE
    _run_case(M, N, K, in_dt, out_dt, _DEFAULT_CONFIG, chain)


@pytest.mark.parametrize("aux_dtype", _EPILOGUE_DTYPES)
def test_epilogue_aux_dtypes(aux_dtype: str) -> None:
    _run_epilogue_dtype_case(aux_dtype, "fp32")


@pytest.mark.parametrize("out_dtype", _EPILOGUE_DTYPES)
def test_epilogue_output_dtypes(out_dtype: str) -> None:
    _run_epilogue_dtype_case("bf16", out_dtype)


def test_epilogue_uint8_output_preserves_values_above_int8_positive_range() -> None:
    M, N, K = 128, 128, 128
    cfg, cta_group = _resolve(_DEFAULT_CONFIG)
    g = _build_epilogue_dtype_graph(M, N, K, "fp32", "uint8")
    try:
        compiled = _plan(g, config=cfg, cta_group=cta_group, force_stg_epi=True)
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else ""
        pytest.fail(f"JIT compile failed: {type(e).__name__}: {first[:200]}", pytrace=False)

    a = torch.zeros(1, M, K, dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")
    b = torch.zeros(1, N, K, dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")
    c = torch.empty(1, M, N, dtype=torch.uint8, device="cuda")
    aux = torch.full((1, 1, N), 200.0, dtype=torch.float32, device="cuda")

    compiled(_vp(compiled, a, b, c, aux))
    torch.cuda.synchronize()

    ref = torch.full((1, M, N), 200, dtype=torch.uint8, device="cuda")
    torch.testing.assert_close(c, ref, atol=0, rtol=0)


def test_epilogue_int32_compute_type() -> None:
    M, N, K = 128, 128, 128
    cfg, cta_group = _resolve(_DEFAULT_CONFIG)
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[_DEFAULT_IN_DT],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    aux = g.tensor(
        name="aux0",
        dim=[1, 1, N],
        stride=[N, N, 1],
        data_type=cudnn.data_type.INT32,
    )
    cur = g.matmul(A=A, B=B, name="mm")
    cur = g.add(
        a=cur,
        b=aux,
        name="add_i32",
        compute_data_type=cudnn.data_type.INT32,
    )
    cur.set_output(True).set_data_type(cudnn.data_type.INT32)
    try:
        compiled = _plan(g, config=cfg, cta_group=cta_group, force_stg_epi=True)
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else ""
        pytest.fail(f"JIT compile failed: {type(e).__name__}: {first[:200]}", pytrace=False)

    a = torch.zeros(1, M, K, dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")
    b = torch.zeros(1, N, K, dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")
    c = torch.empty(1, M, N, dtype=torch.int32, device="cuda")
    aux_runtime = (torch.arange(N, dtype=torch.int32, device="cuda") % 5 - 2).reshape(1, 1, N)

    compiled(_vp(compiled, a, b, c, aux_runtime))
    torch.cuda.synchronize()

    ref = aux_runtime.expand(1, M, N)
    torch.testing.assert_close(c, ref, atol=0, rtol=0)


@pytest.mark.parametrize(
    "op",
    (
        "ceil",
        "floor",
        "erf",
        "log",
        "reciprocal",
        "rsqrt",
        "sqrt",
    ),
)
def test_additional_unary_epilogue_ops(op: str) -> None:
    M, N, K = 128, 128, 128
    cfg, cta_group = _resolve(_DEFAULT_CONFIG)
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[_DEFAULT_IN_DT],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    cur = g.matmul(A=A, B=B, name="mm")
    cur = _apply_unary(g, op, cur, "new_unary")
    cur.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    try:
        compiled = _plan(g, config=cfg, cta_group=cta_group, force_stg_epi=True)
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else ""
        pytest.fail(f"JIT compile failed: {type(e).__name__}: {first[:200]}", pytrace=False)

    a = torch.zeros(1, M, K, dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")
    b = torch.zeros(1, N, K, dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")
    a[:, :, 0] = 1
    b[:, :, 0] = 4
    c = torch.empty(1, M, N, dtype=torch.float32, device="cuda")

    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    ref = _TORCH_UNARY[op](mm)
    torch.testing.assert_close(c, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("bcast", _BCAST_MODES)
@pytest.mark.parametrize("aux_on_rhs", (True, False), ids=("rhs", "lhs"))
@pytest.mark.parametrize("op", ("max", "min", "pow", "add_square"))
def test_additional_binary_epilogue_ops(
    op: str,
    aux_on_rhs: bool,
    bcast: Bcast,
) -> None:
    M, N, K = 128, 128, 128
    cfg, cta_group = _resolve(_DEFAULT_CONFIG)
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[_DEFAULT_IN_DT],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    dim, stride = _aux_dim_stride(bcast, M, N)
    aux = g.tensor(name="aux0", dim=dim, stride=stride)
    cur = g.matmul(A=A, B=B, name="mm")
    if aux_on_rhs:
        cur = _apply_binary(g, op, cur, aux, "new_binary")
    else:
        cur = _apply_binary(g, op, aux, cur, "new_binary")
    cur.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    try:
        compiled = _plan(g, config=cfg, cta_group=cta_group, force_stg_epi=True)
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else ""
        pytest.fail(f"JIT compile failed: {type(e).__name__}: {first[:200]}", pytrace=False)

    a = torch.zeros(1, M, K, dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")
    b = torch.zeros(1, N, K, dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")
    a[:, :, 0] = 1
    b[:, :, 0] = 3
    c = torch.empty(1, M, N, dtype=torch.float32, device="cuda")
    aux_runtime = torch.full(
        tuple(dim),
        2,
        dtype=_TORCH_DTYPE[_DEFAULT_IN_DT],
        device="cuda",
    )

    compiled(_vp(compiled, a, b, c, aux_runtime))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    aux_f32 = aux_runtime.to(torch.float32)
    if aux_on_rhs:
        ref = _TORCH_BINARY[op](mm, aux_f32)
    else:
        ref = _TORCH_BINARY[op](aux_f32, mm)
    torch.testing.assert_close(c, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "case",
    _MIXED_DTYPE_BROADCAST_CASES,
    ids=[_mixed_dtype_broadcast_case_id(c) for c in _MIXED_DTYPE_BROADCAST_CASES],
)
def test_mixed_dtype_rank3_broadcast_patterns(
    case: MixedDtypeBroadcastCase,
) -> None:
    _run_mixed_dtype_broadcast_case(case)


@pytest.mark.parametrize(
    "chain",
    _LAYOUT_CHAINS,
    ids=[_chain_id(c) for c in _LAYOUT_CHAINS],
)
@pytest.mark.parametrize(
    "a_major,b_major",
    _INPUT_LAYOUTS,
    ids=[_layout_id(p) for p in _INPUT_LAYOUTS],
)
def test_fusion_input_layouts(a_major: str, b_major: str, chain: Chain) -> None:
    """Representative fused epilogues with A K/M-major and B K/N-major inputs."""
    M, N, K = 256, 256, 256
    _run_case(
        M,
        N,
        K,
        _DEFAULT_IN_DT,
        _DEFAULT_OUT_DT,
        _DEFAULT_CONFIG,
        chain,
        a_major,
        b_major,
    )


@pytest.mark.parametrize(
    "case",
    _BATCHED_FUSION_CASES,
    ids=[_batched_case_id(c) for c in _BATCHED_FUSION_CASES],
)
def test_batched_fusion_default_axis(case: BatchedFusionCase) -> None:
    """Rank-3 matmul + fused epilogues: unary-only, batch-shared aux, and
    per-batch rank-3 aux."""
    batch, M, N, K = _DEFAULT_BATCHED_SHAPE
    _run_batched_case(batch, M, N, K, _DEFAULT_IN_DT, _DEFAULT_OUT_DT, _DEFAULT_CONFIG, case)


# ---------------------------------------------------------------------------
# Multi-output: set_output(True) intermediates (matmul tap + per-op taps),
# fan-out/fan-in DAGs, reduction taps — checked against a torch reference.
# ---------------------------------------------------------------------------


def _mkdata_bf16(M: int, N: int, K: int, B: int = 1):
    torch.manual_seed(0)
    a = torch.empty(B, M, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    b = torch.empty(B, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    return a, b


def test_matmul_tap_only() -> None:
    """Tap the raw matmul output. Outputs in recorder order: matmul tap (BF16),
    then relu (BF16)."""
    M, N, K = 256, 256, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    Y = g.relu(input=C, name="r")
    Y.set_output(True)

    plan = _plan(g)
    a, b = _mkdata_bf16(M, N, K)
    c_term = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    c_tap = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    plan(_vp(plan, a, b, [c_tap, c_term]))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    torch.testing.assert_close(c_tap, mm.to(torch.bfloat16), atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_term, torch.relu(mm).to(torch.bfloat16), atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("N,want_chunk", [(256, 32), (384, 32), (40, 8), (24, 8)])
def test_reduction_only_chain_sizes_its_chunk_from_n(N: int, want_chunk: int) -> None:
    """A chain whose ONLY output is a reduction has no dense store to size the
    epilogue chunk. It used to be sized off `chain.output_dtype`, which INVENTS a
    bf16 output when there are no specs -- so the chunk came from a tensor that
    does not exist, capped by that tensor's memory-access width. N is the real
    bound: the chunk walks N and nothing else constrains it.

    Nothing exercised this path before; the whole suite ran with it never taken."""
    from cudnn.gemm.frost.compiler import _epi_chunk_elems, _store_modes, _epi_vec_bytes
    from cudnn.gemm.frost.tile_config import by_name

    M, K = 128, 64
    cfg = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1")

    def build():
        g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
        B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
        Y = g.relu(input=g.matmul(A=A, B=B, name="mm"), name="relu")
        R = g.reduction(input=Y, mode=cudnn.reduction_mode.MAX, name="red")
        R.set_dim([1, 1, 1]).set_stride([1, 1, 1]).set_output(True).set_data_type(cudnn.data_type.FLOAT)
        return g, A, B, R

    g, _, _, _ = build()
    chain = analyze(g)
    assert not chain.output_specs and len(chain.reductions) == 1
    assert _epi_chunk_elems(chain, cfg, 1, False) == want_chunk
    assert _store_modes(chain, cfg, 1) == ("stg",)

    torch.manual_seed(0)
    a = torch.randn(1, M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1, N, K, device="cuda", dtype=torch.bfloat16)
    r = torch.zeros(1, 1, 1, device="cuda", dtype=torch.float32)
    gg, aa, bb, rr = build()
    compiler.jit_from_cudnn_graph(gg, config=cfg, cta_group=1)({aa: a, bb: b, rr: r})
    torch.cuda.synchronize()
    ref = torch.relu(torch.einsum("bmk,bnk->bmn", a.float(), b.float())).max()
    assert abs(r.item() - ref.item()) < 1e-3, (r.item(), ref.item())


@pytest.mark.parametrize(
    "mode,ref_fn",
    (
        (cudnn.reduction_mode.ADD, lambda x: x.sum(dim=(0, 1, 2), keepdim=True)),
        (
            cudnn.reduction_mode.AMAX,
            lambda x: x.abs().amax(dim=(0, 1, 2), keepdim=True),
        ),
        (cudnn.reduction_mode.MAX, lambda x: x.amax(dim=(0, 1, 2), keepdim=True)),
        (cudnn.reduction_mode.MIN, lambda x: x.amin(dim=(0, 1, 2), keepdim=True)),
    ),
    ids=("add", "amax", "max", "min"),
)
def test_epilogue_reduction_tap_scalar(mode, ref_fn) -> None:
    """Materialize the normal epilogue output plus a scalar reduction tap."""
    M, N, K = 128, 128, 64
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    Y = g.relu(input=C, name="relu")
    Y.set_output(True)
    R = g.reduction(input=Y, mode=mode, name="red")
    R.set_dim([1, 1, 1]).set_stride([1, 1, 1])
    R.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    plan = _plan(g)
    a, b = _mkdata_bf16(M, N, K)
    c_term = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    c_red = torch.empty(1, 1, 1, dtype=torch.float32, device="cuda")
    plan(_vp(plan, a, b, [c_term, c_red]))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    ref_term = torch.relu(mm)
    torch.testing.assert_close(c_term, ref_term.to(torch.bfloat16), atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_red, ref_fn(ref_term), atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    "mode,ref_fn",
    (
        (cudnn.reduction_mode.ADD, lambda x: x.sum(dim=(0, 1, 2), keepdim=True)),
        (
            cudnn.reduction_mode.AMAX,
            lambda x: x.abs().amax(dim=(0, 1, 2), keepdim=True),
        ),
        (cudnn.reduction_mode.MAX, lambda x: x.amax(dim=(0, 1, 2), keepdim=True)),
        (cudnn.reduction_mode.MIN, lambda x: x.amin(dim=(0, 1, 2), keepdim=True)),
    ),
    ids=("add", "amax", "max", "min"),
)
def test_epilogue_reduction_tap_scalar_int32(mode, ref_fn) -> None:
    """Int32 reduction uses int32 output/compute atomic semantics."""
    M, N, K = 128, 128, 64
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(cudnn.data_type.INT32)
    R = g.reduction(
        input=C,
        mode=mode,
        name="red_i32",
        compute_data_type=cudnn.data_type.INT32,
    )
    R.set_dim([1, 1, 1]).set_stride([1, 1, 1])
    R.set_output(True).set_data_type(cudnn.data_type.INT32)

    plan = _plan(g)
    a, b = _mkdata_bf16(M, N, K)
    c_term = torch.empty(1, M, N, dtype=torch.int32, device="cuda")
    c_red = torch.empty(1, 1, 1, dtype=torch.int32, device="cuda")
    plan(_vp(plan, a, b, [c_term, c_red]))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float()).to(torch.int32)
    torch.testing.assert_close(c_term, mm, atol=0, rtol=0)
    torch.testing.assert_close(c_red, ref_fn(mm).to(torch.int32), atol=0, rtol=0)


@pytest.mark.parametrize(
    "mode,red_shape,ref_dims",
    (
        (cudnn.reduction_mode.ADD, "per_batch", (1, 2)),
        (cudnn.reduction_mode.ADD, "per_row", (0, 2)),
        (cudnn.reduction_mode.ADD, "per_col", (0, 1)),
        (cudnn.reduction_mode.AMAX, "per_col", (0, 1)),
        (cudnn.reduction_mode.MAX, "per_col", (0, 1)),
        (cudnn.reduction_mode.MIN, "per_col", (0, 1)),
    ),
    ids=(
        "add_per_batch",
        "add_per_row",
        "add_per_col",
        "amax_per_col",
        "max_per_col",
        "min_per_col",
    ),
)
def test_epilogue_reduction_tap_partial(mode, red_shape: str, ref_dims: tuple[int, ...]) -> None:
    """Reduction taps keep rank 3 and set only the reduced dimensions to 1."""
    B, M, N, K = 2, 128, 128, 64
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[B, M, K], stride=[M * K, K, 1])
    B_t = g.tensor(name="B", dim=[B, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B_t, name="mm")
    Y = g.relu(input=C, name="relu")
    Y.set_output(True)
    R = g.reduction(input=Y, mode=mode, name="red")
    red_dims = {
        "per_batch": [B, 1, 1],
        "per_row": [1, M, 1],
        "per_col": [1, 1, N],
    }[red_shape]
    R.set_dim(red_dims).set_stride([red_dims[1] * red_dims[2], red_dims[2], 1])
    R.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    plan = _plan(g)
    a, b = _mkdata_bf16(M, N, K, B)
    c_term = torch.empty(B, M, N, dtype=torch.bfloat16, device="cuda")
    c_red = torch.empty(*red_dims, dtype=torch.float32, device="cuda")
    plan(_vp(plan, a, b, [c_term, c_red]))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    ref_term = torch.relu(mm)
    if mode == cudnn.reduction_mode.AMAX:
        ref_red = ref_term.abs().amax(dim=ref_dims, keepdim=True)
    elif mode == cudnn.reduction_mode.MAX:
        ref_red = ref_term.amax(dim=ref_dims, keepdim=True)
    elif mode == cudnn.reduction_mode.MIN:
        ref_red = ref_term.amin(dim=ref_dims, keepdim=True)
    else:
        ref_red = ref_term.sum(dim=ref_dims, keepdim=True)
    torch.testing.assert_close(c_term, ref_term.to(torch.bfloat16), atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_red, ref_red, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    "mode,red_shape,ref_dims",
    (
        (cudnn.reduction_mode.ADD, "per_batch", (1, 2)),
        (cudnn.reduction_mode.ADD, "per_row", (0, 2)),
        (cudnn.reduction_mode.ADD, "per_col", (0, 1)),
        (cudnn.reduction_mode.AMAX, "per_col", (0, 1)),
    ),
    ids=("add_per_batch", "add_per_row", "add_per_col", "amax_per_col"),
)
def test_epilogue_reduction_tap_strided_output(mode, red_shape: str, ref_dims: tuple[int, ...]) -> None:
    """Reduction output atomics honor the runtime output stride."""
    B, M, N, K = 2, 128, 128, 64
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[B, M, K], stride=[M * K, K, 1])
    B_t = g.tensor(name="B", dim=[B, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B_t, name="mm")
    Y = g.relu(input=C, name="relu")
    Y.set_output(True)
    R = g.reduction(input=Y, mode=mode, name="red")
    red_dims, red_stride = {
        "per_batch": ([B, 1, 1], [2, 1, 1]),
        "per_row": ([1, M, 1], [0, 2, 1]),
        "per_col": ([1, 1, N], [0, 0, 2]),
    }[red_shape]
    R.set_dim(red_dims).set_stride(red_stride)
    R.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    plan = _plan(g)
    a, b = _mkdata_bf16(M, N, K, B)
    c_term = torch.empty(B, M, N, dtype=torch.bfloat16, device="cuda")
    c_red = torch.empty_strided(tuple(red_dims), tuple(red_stride), dtype=torch.float32, device="cuda")
    assert not c_red.is_contiguous()
    plan(_vp(plan, a, b, [c_term, c_red]))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    ref_term = torch.relu(mm)
    if mode == cudnn.reduction_mode.AMAX:
        ref_red = ref_term.abs().amax(dim=ref_dims, keepdim=True)
    else:
        ref_red = ref_term.sum(dim=ref_dims, keepdim=True)
    torch.testing.assert_close(c_term, ref_term.to(torch.bfloat16), atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_red, ref_red, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    "mode,ref_fn",
    (
        (cudnn.reduction_mode.ADD, lambda x: x.sum(dim=(0, 1, 2), keepdim=True)),
        (
            cudnn.reduction_mode.AMAX,
            lambda x: x.abs().amax(dim=(0, 1, 2), keepdim=True),
        ),
        (cudnn.reduction_mode.MAX, lambda x: x.amax(dim=(0, 1, 2), keepdim=True)),
        (cudnn.reduction_mode.MIN, lambda x: x.amin(dim=(0, 1, 2), keepdim=True)),
    ),
    ids=("add", "amax", "max", "min"),
)
def test_epilogue_reduction_tap_scalar_big_cgrp_multi_cta(mode, ref_fn) -> None:
    """Reduction taps use global atomics across many CTA contributors."""
    B, M, N, K = 2, 512, 384, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[B, M, K], stride=[M * K, K, 1])
    B_t = g.tensor(name="B", dim=[B, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B_t, name="mm")
    C.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    R = g.reduction(input=C, mode=mode, name="red")
    R.set_dim([1, 1, 1]).set_stride([1, 1, 1])
    R.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    cfg = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster4x2")
    plan = _plan(g, config=cfg, cta_group=2)
    a, b = _mkdata_bf16(M, N, K, B)
    c_term = torch.empty(B, M, N, dtype=torch.float32, device="cuda")
    c_red = torch.empty(1, 1, 1, dtype=torch.float32, device="cuda")
    plan(_vp(plan, a, b, [c_term, c_red]))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    torch.testing.assert_close(c_term, mm, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_red, ref_fn(mm), atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    "mode,ref_fn",
    (
        (cudnn.reduction_mode.ADD, lambda x: x.sum(dim=(0, 1, 2), keepdim=True)),
        (
            cudnn.reduction_mode.AMAX,
            lambda x: x.abs().amax(dim=(0, 1, 2), keepdim=True),
        ),
        (cudnn.reduction_mode.MAX, lambda x: x.amax(dim=(0, 1, 2), keepdim=True)),
        (cudnn.reduction_mode.MIN, lambda x: x.amin(dim=(0, 1, 2), keepdim=True)),
    ),
    ids=("add", "amax", "max", "min"),
)
def test_epilogue_reduction_tap_scalar_int32_big_cgrp_multi_cta(mode, ref_fn) -> None:
    """Int32 reduction atomics work across many CTA contributors."""
    B, M, N, K = 2, 512, 384, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[B, M, K], stride=[M * K, K, 1])
    B_t = g.tensor(name="B", dim=[B, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B_t, name="mm")
    C.set_output(True).set_data_type(cudnn.data_type.INT32)
    R = g.reduction(
        input=C,
        mode=mode,
        name="red_i32",
        compute_data_type=cudnn.data_type.INT32,
    )
    R.set_dim([1, 1, 1]).set_stride([1, 1, 1])
    R.set_output(True).set_data_type(cudnn.data_type.INT32)

    cfg = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster4x2")
    plan = _plan(g, config=cfg, cta_group=2)
    a, b = _mkdata_bf16(M, N, K, B)
    c_term = torch.empty(B, M, N, dtype=torch.int32, device="cuda")
    c_red = torch.empty(1, 1, 1, dtype=torch.int32, device="cuda")
    plan(_vp(plan, a, b, [c_term, c_red]))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float()).to(torch.int32)
    torch.testing.assert_close(c_term, mm, atol=0, rtol=0)
    torch.testing.assert_close(c_red, ref_fn(mm).to(torch.int32), atol=0, rtol=0)


def test_mid_op_tap() -> None:
    """Tap an intermediate fusion-op result. Chain: matmul -> bias -> relu;
    outputs in recorder order: bias tap (FP32), then relu (BF16)."""
    M, N, K = 256, 256, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    bias = g.tensor(name="bias", dim=[1, 1, N], stride=[N, N, 1])
    C = g.matmul(A=A, B=B, name="mm")
    Cb = g.bias(input=C, bias=bias, name="b")
    Cb.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    Y = g.relu(input=Cb, name="r")
    Y.set_output(True)

    plan = _plan(g)
    a, b = _mkdata_bf16(M, N, K)
    bias_t = torch.randn(1, 1, N, device="cuda", dtype=torch.bfloat16)
    c_term = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    c_tap = torch.empty(1, M, N, dtype=torch.float32, device="cuda")
    plan(_vp(plan, a, b, [c_tap, c_term], bias_t))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    ref_after_bias = mm + bias_t.float()
    torch.testing.assert_close(c_tap, ref_after_bias, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_term, torch.relu(ref_after_bias).to(torch.bfloat16), atol=1e-1, rtol=1e-2)


def test_dag_two_branches_from_matmul() -> None:
    """Matmul fan-out to two branches (relu, gelu). Outputs in recorder order:
    relu (slot 0), then gelu (slot 1)."""
    M, N, K = 256, 256, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    Y1 = g.relu(input=C, name="r")
    Y1.set_output(True)  # → slot 0
    Y2 = g.gelu_approx_tanh(input=C, name="g")
    Y2.set_output(True)  # → slot 1

    plan = _plan(g)
    a, b = _mkdata_bf16(M, N, K)
    c_term = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    c_tap = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    plan(_vp(plan, a, b, [c_tap, c_term]))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    ref_gelu = torch.nn.functional.gelu(mm, approximate="tanh").to(torch.bfloat16)
    ref_relu = torch.relu(mm).to(torch.bfloat16)
    torch.testing.assert_close(c_term, ref_gelu, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_tap, ref_relu, atol=1e-1, rtol=1e-2)


def test_dag_two_branches_with_per_branch_ops() -> None:
    """Fan-out where each branch has its own pointwise ops (bias + activation)."""
    M, N, K = 256, 256, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    bias1 = g.tensor(name="bias1", dim=[1, 1, N], stride=[N, N, 1])
    bias2 = g.tensor(name="bias2", dim=[1, 1, N], stride=[N, N, 1])
    C = g.matmul(A=A, B=B, name="mm")
    # Branch 1: set_output first → slot 0
    Cb1 = g.bias(input=C, bias=bias1, name="b1")
    Y1 = g.relu(input=Cb1, name="r")
    Y1.set_output(True)
    # Branch 2: set_output last → slot 1
    Cb2 = g.bias(input=C, bias=bias2, name="b2")
    Y2 = g.gelu_approx_tanh(input=Cb2, name="g")
    Y2.set_output(True)

    plan = _plan(g)
    a, b = _mkdata_bf16(M, N, K)
    bias1_t = torch.randn(1, 1, N, device="cuda", dtype=torch.bfloat16)
    bias2_t = torch.randn(1, 1, N, device="cuda", dtype=torch.bfloat16)
    c_term = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    c_tap = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    plan(_vp(plan, a, b, [c_tap, c_term], bias1_t, bias2_t))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    ref_gelu = torch.nn.functional.gelu(mm + bias2_t.float(), approximate="tanh").to(torch.bfloat16)
    ref_relu = torch.relu(mm + bias1_t.float()).to(torch.bfloat16)
    torch.testing.assert_close(c_term, ref_gelu, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_tap, ref_relu, atol=1e-1, rtol=1e-2)


def test_fan_in_relu_plus_gelu() -> None:
    """Fan-in: Y = add(relu(C), gelu(C)) — two op-results re-merged, Y is the only output."""
    M, N, K = 256, 256, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    R = g.relu(input=C, name="r")
    G = g.gelu_approx_tanh(input=C, name="g")
    Y = g.add(a=R, b=G, name="add")
    Y.set_output(True)

    plan = _plan(g)
    a, b = _mkdata_bf16(M, N, K)
    c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    plan(_vp(plan, a, b, c))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    ref = (torch.relu(mm) + torch.nn.functional.gelu(mm, approximate="tanh")).to(torch.bfloat16)
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)


def test_fan_in_with_intermediate_tap() -> None:
    """Fan-in + tap: relu and gelu both feed `add`, and relu is also set_output as a tap."""
    M, N, K = 256, 256, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    R = g.relu(input=C, name="r")
    R.set_output(True).set_data_type(cudnn.data_type.FLOAT)  # tap relu's output → slot 0
    G = g.gelu_approx_tanh(input=C, name="g")
    Y = g.mul(a=R, b=G, name="mul")
    Y.set_output(True)  # → slot 1

    plan = _plan(g)
    a, b = _mkdata_bf16(M, N, K)
    c_term = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    c_relu = torch.empty(1, M, N, dtype=torch.float32, device="cuda")
    plan(_vp(plan, a, b, [c_relu, c_term]))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    relu_ref = torch.relu(mm)
    gelu_ref = torch.nn.functional.gelu(mm, approximate="tanh")
    torch.testing.assert_close(c_relu, relu_ref, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_term, (relu_ref * gelu_ref).to(torch.bfloat16), atol=1e-1, rtol=1e-2)


def test_complex_fusion_dag() -> None:
    """Everything at once: matmul tap + intermediate taps (varied dtypes),
    fan-out into two branches (bias-relu, scale-gelu), fan-in via add, three aux
    broadcast modes (per-row/per-col/scalar). Asserts encode the exact op topo
    and output slot order."""
    M, N, K = 256, 256, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    bias_row = g.tensor(name="bias_row", dim=[1, M, 1], stride=[M, 1, 1])  # per-row
    scale_col = g.tensor(name="scale_col", dim=[1, 1, N], stride=[N, N, 1])  # per-col
    alpha = g.tensor(name="alpha", dim=[1, 1, 1], stride=[1, 1, 1])  # scalar

    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(cudnn.data_type.HALF)  # tap

    # Branch 1: bias + relu
    T1 = g.bias(input=C, bias=bias_row, name="b1")
    T1.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)  # tap
    R1 = g.relu(input=T1, name="r")
    R1.set_output(True).set_data_type(cudnn.data_type.FLOAT)  # tap

    # Branch 2: scale + gelu_tanh
    T2 = g.mul(a=C, b=scale_col, name="m1")
    G1 = g.gelu_approx_tanh(input=T2, name="g")

    # Fan-in: re-merge branches via add
    S = g.add(a=R1, b=G1, name="add")
    S.set_output(True).set_data_type(cudnn.data_type.HALF)  # tap

    # Final: scale by scalar and tanh — terminal
    SC = g.mul(a=S, b=alpha, name="m2")
    Y = g.tanh(input=SC, name="t")
    Y.set_output(True)  # terminal (BF16)

    plan = _plan(g)

    # Validate IR structure
    chain = analyze(g)
    assert len(chain.ops) == 7
    assert [op.op for op in chain.ops] == [
        "add",
        "relu",
        "mul",
        "gelu_tanh",
        "add",
        "mul",
        "tanh",
    ]
    assert chain.ops[0].parent_idx == -1 and chain.ops[0].aux == "bias_row"  # bias on C
    assert chain.ops[1].parent_idx == 0  # relu on T1
    assert chain.ops[2].parent_idx == -1 and chain.ops[2].aux == "scale_col"  # mul on C
    assert chain.ops[3].parent_idx == 2  # gelu on T2
    assert chain.ops[4].parent_idx == 1 and chain.ops[4].parent_idx_b == 3  # add(R1, G1) fan-in
    assert chain.ops[4].aux is None  # fan-in has no aux
    assert chain.ops[5].parent_idx == 4 and chain.ops[5].aux == "alpha"  # mul by scalar
    assert chain.ops[6].parent_idx == 5  # tanh on SC
    assert chain.output_specs[-1].source_ref == 6  # last set_output = tanh

    outs = chain.outputs
    assert [(o.source, o.dtype) for o in outs] == [
        ("matmul", "fp16"),
        ("op_0", "bf16"),
        ("op_1", "fp32"),
        ("op_4", "fp16"),
        ("op_6", "bf16"),
    ]

    # Run + verify against torch reference
    a, b = _mkdata_bf16(M, N, K)
    bias_row_t = torch.randn(1, M, 1, device="cuda", dtype=torch.bfloat16)
    scale_col_t = torch.randn(1, 1, N, device="cuda", dtype=torch.bfloat16)
    alpha_t = torch.tensor([[[2.0]]], device="cuda", dtype=torch.bfloat16)

    c_term = torch.empty(1, M, N, device="cuda", dtype=torch.bfloat16)
    c_mm = torch.empty(1, M, N, device="cuda", dtype=torch.float16)
    c_bias = torch.empty(1, M, N, device="cuda", dtype=torch.bfloat16)
    c_relu = torch.empty(1, M, N, device="cuda", dtype=torch.float32)
    c_addS = torch.empty(1, M, N, device="cuda", dtype=torch.float16)

    plan(
        _vp(
            plan,
            a,
            b,
            [c_mm, c_bias, c_relu, c_addS, c_term],
            bias_row_t,
            scale_col_t,
            alpha_t,
        )
    )
    torch.cuda.synchronize()

    # Reference: compute in fp32, but round at each tensor's declared narrow
    # dtype before the next op reads it (matches cuDNN tensor semantics).
    def _round(x, dt):
        return x.to(dt).float()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    c = _round(mm, torch.float16)  # C: fp16 (feeds both branches)
    t1 = _round(c + bias_row_t.float(), torch.bfloat16)  # T1: bf16
    r1 = torch.relu(t1)  # R1: fp32 (no round)
    t2 = c * scale_col_t.float()  # T2: virtual fp32
    g1 = torch.nn.functional.gelu(t2, approximate="tanh")  # G1: virtual fp32
    s = _round(r1 + g1, torch.float16)  # S: fp16
    sc = s * alpha_t.float()  # SC: virtual fp32
    y = torch.tanh(sc)  # terminal

    torch.testing.assert_close(c_mm, mm.to(torch.float16), atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_bias, t1.to(torch.bfloat16), atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_relu, r1, atol=1e-3, rtol=1e-3)  # fp32 tight
    torch.testing.assert_close(c_addS, s.to(torch.float16), atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_term, y.to(torch.bfloat16), atol=1e-1, rtol=1e-2)


def test_two_taps_matmul_and_mid_op() -> None:
    """Both a matmul tap AND a mid-op tap in one chain (matmul -> bias -> gelu_tanh).
    Outputs in recorder order: matmul tap (FP32), bias tap (BF16), gelu (BF16)."""
    M, N, K = 256, 256, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    bias = g.tensor(name="bias", dim=[1, 1, N], stride=[N, N, 1])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    Cb = g.bias(input=C, bias=bias, name="b")
    Cb.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    Y = g.gelu_approx_tanh(input=Cb, name="g")
    Y.set_output(True)

    plan = _plan(g)
    a, b = _mkdata_bf16(M, N, K)
    bias_t = torch.randn(1, 1, N, device="cuda", dtype=torch.bfloat16)
    c_term = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    c_tap_mm = torch.empty(1, M, N, dtype=torch.float32, device="cuda")
    c_tap_bias = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    plan(_vp(plan, a, b, [c_tap_mm, c_tap_bias, c_term], bias_t))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    after_bias = mm + bias_t.float()
    after_gelu = torch.nn.functional.gelu(after_bias, approximate="tanh")

    torch.testing.assert_close(c_tap_mm, mm, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_tap_bias, after_bias.to(torch.bfloat16), atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_term, after_gelu.to(torch.bfloat16), atol=1e-1, rtol=1e-2)


def test_virtual_bf16_intermediate_loses_precision() -> None:
    """A pure-virtual bf16 intermediate (via bf16 intermediate_data_type) rounds
    the running value, so the kernel matches a rounded reference and differs from
    an fp32-throughout one. Fractional scale so the post-mul value isn't bf16-exact."""
    M, N, K = 256, 256, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.BFLOAT16,  # virtuals are bf16
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    scale = g.tensor(name="scale", dim=[1, 1, N], stride=[N, N, 1])
    C = g.matmul(A=A, B=B, name="mm")
    T = g.mul(a=C, b=scale, name="m")  # rounding bites here
    Y = g.gelu_approx_tanh(input=T, name="g")  # terminal
    Y.set_data_type(cudnn.data_type.FLOAT)  # fp32 output for tight compare
    Y.set_output(True)

    chain = analyze(g)
    assert chain.matmul.out_dtype == "bf16"  # C rounded
    assert chain.ops[0].out_dtype == "bf16"  # T rounded before gelu
    plan = _plan(g)

    a, b = _mkdata_bf16(M, N, K)
    scale_t = torch.randn(1, 1, N, device="cuda", dtype=torch.bfloat16)
    c_out = torch.empty(1, M, N, device="cuda", dtype=torch.float32)
    plan(_vp(plan, a, b, [c_out], scale_t))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())

    # Reference WITH rounding at the bf16 virtuals (what the kernel should do)
    c_r = mm.to(torch.bfloat16).float()
    t_r = (c_r * scale_t.float()).to(torch.bfloat16).float()
    y_rounded = torch.nn.functional.gelu(t_r, approximate="tanh")

    # Reference WITHOUT intermediate rounding (the old/wrong behavior)
    y_fp32 = torch.nn.functional.gelu(mm * scale_t.float(), approximate="tanh")

    torch.testing.assert_close(c_out, y_rounded, atol=1e-1, rtol=1e-2)
    # ...and the rounding actually mattered (else the test proves nothing)
    assert (y_rounded - y_fp32).abs().max().item() > 0.1, "intermediate rounding had no measurable effect — pick data that exercises it"


def test_m_major_op_tap() -> None:
    """M-major first output + tanh extra output (2CTA). The FP32 gelu output
    binds first (recorder order) and rides the runtime M-major strides; the
    BF16 tanh output is an extra dense output, stored compact N-major (a
    declared M-major on an extra output is rejected)."""
    M, N, K = 256, 256, 128
    tap_dt = cudnn.data_type.FLOAT
    tap_tdt = torch.float32
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    R = g.gelu_approx_tanh(input=C, name="r")
    R.set_stride([M * N, 1, M])  # op-tap: M-major
    R.set_output(True).set_data_type(tap_dt)
    Y = g.tanh(input=R, name="t")
    Y.set_output(True)

    cfg = by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1")
    plan = _plan(g, config=cfg, cta_group=2)

    a, b = _mkdata_bf16(M, N, K)
    c_term = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    c_tap = torch.empty(1, N, M, dtype=tap_tdt, device="cuda").transpose(1, 2)
    plan(_vp(plan, a, b, [c_tap, c_term]))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    tap = torch.nn.functional.gelu(mm, approximate="tanh")
    rounded_tap = tap.to(tap_tdt)
    torch.testing.assert_close(c_tap, rounded_tap, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(
        c_term,
        torch.tanh(rounded_tap.float()).to(torch.bfloat16),
        atol=1e-1,
        rtol=1e-2,
    )


def test_m_major_extra_output() -> None:
    """M-major on a NON-first output: the extra dense output rides its own
    runtime strides via the per-element scatter path (Phase E)."""
    M, N, K = 256, 256, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    R = g.gelu_approx_tanh(input=C, name="r")
    R.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    Y = g.tanh(input=R, name="t")
    Y.set_stride([M * N, 1, M])  # extra output: M-major
    Y.set_output(True)

    cfg = by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1")
    plan = _plan(g, config=cfg, cta_group=2)
    assert plan.chain.output_specs[0].major == "n" and plan.chain.output_specs[1].major == "m"

    a, b = _mkdata_bf16(M, N, K)
    c_first = torch.empty(1, M, N, dtype=torch.float32, device="cuda")
    c_mmaj = torch.empty(1, N, M, dtype=torch.bfloat16, device="cuda").transpose(1, 2)
    plan(_vp(plan, a, b, [c_first, c_mmaj]))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    tap = torch.nn.functional.gelu(mm, approximate="tanh").to(torch.float32)
    torch.testing.assert_close(c_first, tap, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_mmaj.float(), torch.tanh(tap.float()).to(torch.bfloat16).float(), atol=1e-1, rtol=1e-2)


# Standalone CLI shim — forwards remaining argv to pytest on this file


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))


# --- full cuDNN pointwise-mode coverage (new ops, 2026-07-20) ---------------


@pytest.mark.parametrize("mode,dims", (("per_col", [1, 1, 128]), ("per_elem", [1, 128, 128])))
def test_coordinate_reading_aux_is_clamped_on_the_tma_arm(mode: str, dims: list[int]) -> None:
    """The TMA arm hands the snippet a subtile base with no N or M bound, so a
    `per_col` / `per_elem` LDG at `col_j + k` reads past the aux allocation.
    The clamped fallback keeps every address inside both extents; the STG arm
    already carries `col_j + vsize <= N` and must stay byte-identical."""
    M = N = K = 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    stride = [dims[1] * dims[2], dims[2], 1]
    x = g.tensor(name="x", dim=dims, stride=stride, data_type=cudnn.data_type.FLOAT)
    g.mul(a=C, b=x, name="w").set_output(True)
    chain = analyze(g)

    stg = generate(chain, tma_slots=frozenset()).epilogue
    tma = generate(chain, tma_slots=frozenset({0})).epilogue

    assert "_auxv_x" not in stg, "the STG arm's own predicate already bounds the read — no fallback wanted"
    assert ".load(count=vsize, alignment=ALIGN_AUX_x)" in stg

    assert "cute.math.min(cutlass.Int32(col_j) + _auxk, cutlass.Int32(N) - 1)" in tma
    assert ("cute.math.min(cutlass.Int32(row), cutlass.Int32(M) - 1)" in tma) == (mode == "per_elem")
    # The whole-chunk load is still the fast path; the clamp is the tail only.
    assert "if (col_j + vsize <= N)" in tma
    assert ".load(count=vsize, alignment=ALIGN_AUX_x)" in tma


@pytest.mark.parametrize(
    "cfg_name,dts,want",
    [
        ("CONFIG_sm100_128x256x128_128x256x32_cluster1x1", ["bf16", "bf16"], ("tma", "tma")),
        ("CONFIG_sm100_128x256x128_128x256x32_cluster1x1", ["bf16", "fp32"], ("tma", "tma")),
        # epi_n is 64 only on this family, and 64 fp32 columns is a 256-byte row --
        # wider than any swizzle in the table, so the fp32 output falls back alone.
        ("CONFIG_sm100_256x256x128_128x256x32_cluster1x1", ["bf16", "fp32"], ("tma", "stg")),
        ("CONFIG_sm100_256x256x128_128x256x32_cluster1x1", ["fp32", "bf16"], ("tma", "tma")),
    ],
)
def test_epi_n_is_one_shared_column_count_not_a_per_output_one(cfg_name: str, dts: list[str], want: tuple) -> None:
    """The kernel renders ONE `epi_n`, from the primary slot's dtype. What is per
    output is the ROW BYTES (`epi_n * its own element width`) and with them the
    swizzle -- so a second TMA output is judged against the SHARED column count,
    not against the one its own dtype would have produced. Asking per output
    computes a width the kernel never renders, which at the 256x256 family lets
    an fp32 output onto a 256-byte staging row that has no swizzle."""
    from cudnn.gemm.frost.compiler import _epi_vec_bytes, _store_modes
    from cudnn.gemm.frost.tile_config import by_name

    CU = {"bf16": cudnn.data_type.BFLOAT16, "fp32": cudnn.data_type.FLOAT}
    M = N = K = 256
    g = cudnn.pygraph(io_data_type=CU["bf16"], intermediate_data_type=CU["fp32"], compute_data_type=CU["fp32"])
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    cur = g.relu(input=g.matmul(A=A, B=B, name="mm"), name="r")
    cur.set_output(True).set_data_type(CU[dts[0]])
    g.mul(a=cur, b=cur, name="sq").set_output(True).set_data_type(CU[dts[1]])

    chain = analyze(g)
    cfg = by_name(cfg_name)
    assert _store_modes(chain, cfg, 1) == want


@pytest.mark.parametrize(
    "majors,want",
    [
        (("n", "n"), ("tma", "tma")),
        (("n", "m"), ("tma", "tma")),
        (("m", "m"), ("tma", "tma")),
        (("m", "n"), ("tma", "tma")),
    ],
)
def test_outputs_of_either_layout_share_one_epilogue(majors: tuple, want: tuple) -> None:
    """Layout and store mode are independent axes: every output rides the same
    row-per-lane fragment and differs only in how its store stages it, so one
    epilogue serves any mix of m/n-major and any mix of tma/stg."""
    from cudnn.gemm.frost.compiler import _epi_vec_bytes, _store_modes, jit_from_cudnn_graph
    from cudnn.gemm.frost.tile_config import by_name

    M = N = K = 256
    cfg = by_name("CONFIG_sm100_128x256x128_128x256x32_cluster1x1")

    def build():
        g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
        B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
        R = g.relu(input=g.matmul(A=A, B=B, name="mm"), name="r")
        Y = g.gelu_approx_tanh(input=R, name="ge")
        for t, mj in zip((R, Y), majors):
            if mj == "m":
                t.set_stride([M * N, 1, M])
            t.set_output(True)
        return g, A, B, R, Y

    g, *_ = build()
    chain = analyze(g)
    assert _store_modes(chain, cfg, 1) == want

    torch.manual_seed(0)
    a = torch.randn(1, M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1, N, K, device="cuda", dtype=torch.bfloat16)

    def run(force_stg: bool):
        slack, outs, raws = 4096, [], []
        for mj in majors:
            raw = torch.full((2 * M * N + slack,), 0xAB, device="cuda", dtype=torch.uint8)
            v = raw.view(torch.bfloat16)[: M * N]
            outs.append(v.view(1, N, M).transpose(1, 2) if mj == "m" else v.view(1, M, N))
            raws.append((raw, raw[2 * M * N :].clone()))
        gg, aa, bb, rr, yy = build()
        jit_from_cudnn_graph(gg, config=cfg, cta_group=1, force_stg_epi=force_stg)({aa: a, bb: b, rr: outs[0], yy: outs[1]})
        torch.cuda.synchronize()
        for raw, tail in raws:
            assert torch.equal(raw[2 * M * N :], tail), "a store ran past its output"
        return [o.contiguous() for o in outs]

    for got, ref in zip(run(False), run(True)):
        assert torch.equal(got.view(torch.uint8), ref.view(torch.uint8))


def test_both_renderers_emit_the_tma_output_COUNT_not_a_flag() -> None:
    """`n_tma_outputs` sizes the descriptor prefetch, and on MoE also the
    tensormap scratch and the per-CTA workspace stride. The block-scale renderer
    used to emit `1 if use_tma_store_epi else 0` -- a FLAG -- which under-counts
    the moment a block-scale graph puts two outputs on the surface, so the second
    descriptor's scratch and workspace slot overlap somebody else's."""
    import re

    from cudnn.gemm.frost.compiler import (
        _epi_vec_bytes,
        _render_block_scale_tile_constants,
        _render_tile_constants,
        _tma_slots_for,
    )
    from cudnn.gemm.frost.tile_config import by_name

    E4, E8, RE = cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E8M0, cudnn.tensor_reordering.F8_128x4
    M = N = K = 256
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1], data_type=E4)
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K], data_type=E4)
    sfa = g.tensor(name="sfa", dim=[1, M, K // 32], stride=[M * K // 32, K // 32, 1], data_type=E8, reordering_type=RE)
    sfb = g.tensor(name="sfb", dim=[1, K // 32, N], stride=[K * N // 32, 1, K // 32], data_type=E8, reordering_type=RE)
    c = g.matmul(
        A=g.block_scale_dequantize(input=A, descale=sfa, block_size=[1, 32], name="dqa"),
        B=g.block_scale_dequantize(input=B, descale=sfb, block_size=[32, 1], name="dqb"),
        name="mm",
        compute_data_type=cudnn.data_type.FLOAT,
    )
    R = g.relu(input=c, name="r")
    R.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    g.mul(a=R, b=R, name="sq").set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

    chain = analyze(g)
    assert chain.has_block_scale
    cfg = by_name("CONFIG_sm100_128x256x128_128x256x32_cluster1x1")
    want = len(_tma_slots_for(chain, cfg, 1))
    assert want == 2, "this graph is meant to put BOTH outputs on the surface"

    for render in (
        _render_block_scale_tile_constants(cfg, chain, 1, use_tma_store_epi=True),
        _render_tile_constants(cfg, chain, 1),
    ):
        m = re.search(r"^n_tma_outputs = (\d+)$", render, re.M)
        assert m, "every renderer must emit n_tma_outputs"
        assert int(m.group(1)) == want, f"emitted {m.group(1)}, real TMA slots {want}"


def test_a_quant_output_must_carry_a_quantized_data_dtype() -> None:
    """An output holding `quant_idx` IS that node's quantized data, so its dtype
    has to be one the quantizer emits. The check needs BOTH halves -- the dtype
    is on the OutputSpec, the quant node is on the chain -- so it lives in
    `FusionChain.__post_init__` beside the other cross-references, not on either
    piece alone. ue5m3 is deliberately absent: it is a SCALE format
    (`BlockQuantizeSpec.scale_dtype`, validated there), carried as an opaque
    byte, never quantized DATA."""
    from cudnn.gemm.frost.fusion_ir import (
        QUANT_DATA_DTYPES,
        BlockQuantizeSpec,
        FusionChain,
        MatmulSpec,
        OutputSpec,
        gemm_source,
    )

    assert "fp8_e5m3" not in QUANT_DATA_DTYPES
    assert "fp8_e5m3" in BlockQuantizeSpec(source_ref=gemm_source(0), block_size=32, scale_dtype="fp8_e5m3").scale_dtype

    mm = MatmulSpec(M=256, N=256, K=256, batch=1, a_batch=1, b_batch=1)
    q = BlockQuantizeSpec(source_ref=gemm_source(0), block_size=32)

    def build(dtype: str) -> FusionChain:
        return FusionChain(matmul=mm, ops=[], quants=[q], output_specs=[OutputSpec(source_ref=gemm_source(0), dtype=dtype, quant_idx=0)])

    for dtype in QUANT_DATA_DTYPES:
        build(dtype)
    for dtype in ("bf16", "fp32", "fp8_e5m3"):
        with pytest.raises(ValueError, match="quantized DATA"):
            build(dtype)
    # It is NOT a constraint on an ordinary output of the same dtype.
    FusionChain(matmul=mm, ops=[], output_specs=[OutputSpec(source_ref=gemm_source(0), dtype="bf16")])


@pytest.mark.parametrize(
    "cfg_name,cta_group",
    [
        ("CONFIG_sm100_128x128x128_128x128x32_cluster1x1", 1),
        ("CONFIG_sm100_128x256x128_128x256x32_cluster2x1", 2),
        ("CONFIG_sm100_256x256x128_128x256x32_cluster1x1", 1),
        ("CONFIG_sm100_64x128x128_64x128x32_cluster1x1", 1),
    ],
)
@pytest.mark.parametrize("second", ["bf16", "fp32"])
def test_smem_d_reserve_matches_what_the_templates_stage(cfg_name: str, cta_group: int, second: str) -> None:
    """The epilogue ring's per-stage size is computed in THREE places: the host
    reserve (`_smem_d_bytes`, subtracted from the AB budget), the template's own
    `epi_subtile_elems`, and each TMA output's typed view of a slot. They must
    agree -- a reserve that is too small overflows the ring, one that is too large
    silently costs an `ab_stage`.

    The template's formula is READ here, not restated, so changing it fails this
    rather than drifting."""
    import cutlass
    import cutlass.cute as cute
    import cutlass.experimental.cuda.tensor_map as _tma
    import cutlass.experimental.primitives as nvvm
    from cudnn.gemm.frost.compiler import (
        _EPI_SMEM_STAGES,
        _epi_n,
        _epi_vec_bytes,
        _epi_stage_rows,
        _render_tile_constants,
        _smem_d_bytes,
        _tma_out_dtypes,
        _tma_slots_for,
    )
    from cudnn.gemm.frost.dtypes import DTYPE_BYTES
    from cudnn.gemm.frost.tile_config import by_name

    tmpl = pathlib.Path(cudnn.__file__).parent / "gemm" / "frost" / "kernel_templates"
    formulas = set()
    for f in sorted(tmpl.glob("sm*.py")):
        got = [l.strip() for l in f.read_text().split("\n") if l.strip().startswith("epi_subtile_elems = ")]
        assert len(got) == 1, f.name
        formulas.add(got[0])
    assert len(formulas) == 1, f"templates disagree on the ring slot size: {formulas}"

    CU = {"bf16": cudnn.data_type.BFLOAT16, "fp32": cudnn.data_type.FLOAT}
    M = N = K = 256
    g = cudnn.pygraph(io_data_type=CU["bf16"], intermediate_data_type=CU["fp32"], compute_data_type=CU["fp32"])
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    cur = g.relu(input=g.matmul(A=A, B=B, name="mm"), name="r")
    cur.set_output(True).set_data_type(CU["bf16"])
    g.mul(a=cur, b=cur, name="sq").set_output(True).set_data_type(CU[second])
    chain = analyze(g)
    cfg = by_name(cfg_name)

    ns = {"cutlass": cutlass, "cute": cute, "nvvm": nvvm, "_tma": _tma, "num_epilogue_warps": 4}
    exec(_render_tile_constants(cfg, chain, cta_group), ns)
    exec(formulas.pop(), ns)
    slot_bytes = ns["epi_subtile_elems"] * (ns["cd_dtype"].width // 8)

    assert _EPI_SMEM_STAGES * slot_bytes + 16 == _smem_d_bytes(cfg, chain, cta_group)

    epi_n = _epi_n(cfg, cta_group, chain.output_dtype)
    slots = _tma_slots_for(chain, cfg, cta_group)
    for _slot, dt in _tma_out_dtypes(chain, slots):
        view = _epi_stage_rows(cfg, cta_group) * epi_n * DTYPE_BYTES[dt]
        assert view <= slot_bytes, f"{dt} view {view}B does not fit a {slot_bytes}B slot"


@pytest.mark.parametrize("cfg_name,cta_group", [("CONFIG_sm100_128x128x128_128x128x32_cluster1x1", 1), ("CONFIG_sm100_128x256x128_128x256x32_cluster2x1", 2)])
@pytest.mark.parametrize("block_size", (16, 32))
def test_sub_byte_output_takes_the_tma_store(cfg_name: str, cta_group: int, block_size: int) -> None:
    """An fp4 quant DATA output rides the TMA arm. It is the one dtype whose
    element is narrower than the carrier its descriptor is declared with, so
    `epi_n` (LOGICAL columns) and the ring row, the box and the store coordinate
    (all PACKED) part company -- mixing the two is what kept it on STG.

    Measured 4096x4096x4096 nvfp4 out, `128x128` 1ctamma: 83.0 vs 85.5 us."""
    from cudnn.gemm.frost.compiler import _epi_row_bytes, _epi_vec_bytes, _store_modes, jit_from_cudnn_graph
    from cudnn.gemm.frost.tile_config import by_name

    M = K = 256
    N = 256
    cfg = by_name(cfg_name)

    def build():
        g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
        B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
        Y = g.relu(input=g.matmul(A=A, B=B, name="mm"), name="r")
        q, sf = g.block_scale_quantize(input=Y, block_size=block_size, axis=2, name="q")
        q.set_output(True).set_data_type(cudnn.data_type.FP4_E2M1)
        sf.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)
        return g, A, B, q, sf

    g, _, _, _, _ = build()
    chain = analyze(g)
    assert _store_modes(chain, cfg, cta_group) == ("tma", "stg")
    # the packed row is half the logical column count, in bytes
    assert _epi_row_bytes("fp4_e2m1", 32) == 16 and _epi_row_bytes("bf16", 32) == 64

    torch.manual_seed(0)
    a = torch.randn(1, M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1, N, K, device="cuda", dtype=torch.bfloat16)

    def run(force_stg: bool):
        nbytes, slack = M * (N // 2), 4096
        raw = torch.full((nbytes + slack,), 0xAB, device="cuda", dtype=torch.uint8)
        data = raw[:nbytes].view(1, M, N // 2)
        data.zero_()
        tail = raw[nbytes:].clone()
        scale = torch.zeros(1, M, N // block_size, device="cuda", dtype=torch.float8_e8m0fnu)
        gg, aa, bb, qq, ss = build()
        jit_from_cudnn_graph(gg, config=cfg, cta_group=cta_group, force_stg_epi=force_stg)({aa: a, bb: b, qq: data, ss: scale})
        torch.cuda.synchronize()
        assert torch.equal(raw[nbytes:], tail), "the store ran past the packed output"
        return data.clone(), scale.view(torch.uint8).clone()

    got_d, got_s = run(False)
    ref_d, ref_s = run(True)
    assert torch.equal(got_d, ref_d)
    assert torch.equal(got_s, ref_s)


def test_only_a_tma_stored_output_widens_the_ring() -> None:
    """The ring slot spans the widest TMA-stored output's element width. An
    output that takes STG never touches the ring, so it must not size it -- a
    wide STG-only output beside a narrow TMA one used to double the reserve and
    take the bytes straight out of `ab_stages`."""
    from cudnn.gemm.frost.compiler import _epi_slot_widen, _epi_vec_bytes, _smem_d_bytes, _store_modes
    from cudnn.gemm.frost.tile_config import by_name

    CU = {"bf16": cudnn.data_type.BFLOAT16, "fp32": cudnn.data_type.FLOAT}
    M = N = K = 256

    def build(second: str) -> FusionChain:
        g = cudnn.pygraph(io_data_type=CU["bf16"], intermediate_data_type=CU["fp32"], compute_data_type=CU["fp32"])
        A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
        B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
        cur = g.relu(input=g.matmul(A=A, B=B, name="mm"), name="r")
        cur.set_output(True).set_data_type(CU["bf16"])
        g.mul(a=cur, b=cur, name="sq").set_output(True).set_data_type(CU[second])
        return analyze(g)

    # epi_n = 64 here, so an fp32 output's 256-byte staging row has no swizzle
    # and it declines the TMA arm.
    stg = by_name("CONFIG_sm100_256x256x128_128x256x32_cluster1x1")
    chain = build("fp32")
    assert _store_modes(chain, stg, 1) == ("tma", "stg")
    assert _epi_slot_widen(chain, stg, 1) == 1
    assert _smem_d_bytes(stg, chain, 1) == _smem_d_bytes(stg, build("bf16"), 1)

    # ...but at epi_n = 32 the same fp32 output IS TMA-stored, and then it does.
    tma = by_name("CONFIG_sm100_128x256x128_128x256x32_cluster1x1")
    chain = build("fp32")
    assert _store_modes(chain, tma, 1) == ("tma", "tma")
    assert _epi_slot_widen(chain, tma, 1) == 2


def test_more_than_one_output_can_take_the_tma_store() -> None:
    """Every TMA-stored output walks the SAME SMEM ring -- the slot is sized for
    the widest and each takes its own typed view -- so there is no budget to
    spend: an output takes the surface iff it is eligible."""
    from cudnn.gemm.frost.compiler import _epi_slot_widen, _epi_vec_bytes, _store_modes
    from cudnn.gemm.frost.tile_config import by_name
    from cudnn.gemm.frost.graph_analyzer import analyze as _an

    CU = {"bf16": cudnn.data_type.BFLOAT16, "fp32": cudnn.data_type.FLOAT}
    M = N = K = 256
    g = cudnn.pygraph(io_data_type=CU["bf16"], intermediate_data_type=CU["fp32"], compute_data_type=CU["fp32"])
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    cur = g.relu(input=g.matmul(A=A, B=B, name="mm"), name="r")
    cur.set_output(True).set_data_type(CU["bf16"])
    g.mul(a=cur, b=cur, name="sq").set_output(True).set_data_type(CU["fp32"])
    chain = _an(g)

    cfg = by_name("CONFIG_sm100_128x256x128_128x256x32_cluster1x1")
    assert _store_modes(chain, cfg, 1) == ("tma", "tma")
    # The ring is typed as the primary slot's dtype, so a wider companion widens
    # the SLOT rather than adding a second ring.
    assert _epi_slot_widen(chain, cfg, 1) == 2


def test_every_dense_store_on_the_tma_arm_carries_the_row_bound() -> None:
    """The TMA arm runs the epilogue body for EVERY row of the tile and clips only
    its own store (the C descriptor's extent does that in hardware). Every other
    store in the body has to re-apply the bound itself.

    A quant tap's DATA store is the one that does not go through
    `_emit_tap_store`, and it was missed: `generate` appends it directly for
    `spec.quant_idx is not None`. Unguarded it writes the tile's overhang rows,
    which on a plain matmul run past the output buffer into whatever the
    allocator put next -- observed as ~55 % of a neighbouring SCALE buffer coming
    back as zero, and only on the first launch in a fresh process, because a warm
    allocator lays the buffers out differently."""
    M, N, K = 384, 256, 256
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    R = g.relu(input=g.matmul(A=A, B=B, name="mm"), name="r")
    R.set_output(True)
    q, sc = g.block_scale_quantize(input=R, block_size=32, axis=-1, name="qr")
    q.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    sc.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)
    chain = analyze(g)

    # A quant pins the chunk to its block size; slot 0 is bf16 -> 32 * 2 bytes.
    kw = dict(vec_bytes_epi=64, output_elem_bytes=2)
    tma = generate(chain, tma_slots=frozenset({0}), **kw).epilogue
    stg = generate(chain, tma_slots=frozenset(), **kw).epilogue

    guarded = [ln for ln in tma.splitlines() if ln.startswith("    (gC_tap_")]
    bare = [ln for ln in tma.splitlines() if ln.startswith("(gC_tap_")]
    assert guarded, "the TMA arm must emit its tap stores under a bound"
    assert not bare, f"unguarded store(s) on the TMA arm: {bare}"
    # The STG arm inherits `row < M` / `col_j + vsize <= N` from the drain, so its
    # stores must stay bare -- a guard there would change every shipping kernel.
    assert not [ln for ln in stg.splitlines() if ln.strip().startswith("if (row <")]


def test_coordinate_reading_pointwise_is_not_a_store_rule() -> None:
    """The store rule may not consult how a pointwise input reads its
    coordinates. Every arm hands the snippet the SAME row-per-lane fragment, so
    `row` / `col_j` are always its real coordinates; asking "does this graph read
    coordinates" in the gate is the shape of the bug, not the fix."""
    for fn in (compiler._output_store_mode,):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        # The prose may name these; the CODE may not.
        body = tree.body[0].body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]
        src = "\n".join(ast.unparse(node) for node in body)
        for probe in ("bcast_mode", "grouped_by_moe", "gen_index"):
            assert probe not in src, f"{fn.__name__} consults {probe}: that is a pointwise concern"


def test_m_major_scatter_covers_the_whole_chunk() -> None:
    """An M-major output stores one element per chunk column through its own
    strides. The trip count must follow the chunk -- it was once pinned at 16,
    exact only at `epi_n == 32` and dropping half the output at the 64 the wide
    tiles reach."""
    from cudnn.gemm.frost.epilogue_codegen import _emit_mmajor_scatter

    for vsize in (8, 16, 32, 64):
        lines = _emit_mmajor_scatter(0, 0, "vec_out", "bf16", 1, vsize)
        assert sum(".store(" in l for l in lines) == vsize, vsize


def test_tma_staged_values_reach_the_store_as_vectors() -> None:
    """`store_swizzled` picks its path by sniffing whether the value's `.shape`
    is a tuple: a `cutlass.Vector` reports `(N,)` and gets the per-16-byte-granule
    scatter the SMEM swizzle needs, while `cute.make_rmem_tensor(N, ...).load()`
    is a `TensorSSA` whose `.shape` is the bare int `N` -- taken for a scalar,
    XORed once on the row base and written CONTIGUOUSLY, so each row's tail
    spills into the next. Every emitter that can reach the TMA arm must hand on
    a Vector."""
    src = pathlib.Path(epilogue_codegen.__file__).read_text()

    sites = src.count("cute.make_rmem_tensor(")
    converted = src.count(".load().to_vector()")
    assert sites == 4, (
        f"epilogue_codegen has {sites} make_rmem_tensor sites, expected 4. A new one either "
        f"converts with .load().to_vector() or its feature stays off the TMA arm -- decide which, "
        f"then update this test."
    )
    assert converted == 5, f"expected exactly 5 converted rmem loads, found {converted}"
    # The two block-quantize emitters were the last holdouts: their result now
    # reaches a TMA-stored dense output, so they must convert like the rest.
    assert src.count("_out = cute.make_rmem_tensor(") == 2
    assert src.count("_vec = {p}_out.load().to_vector()") == 2, "the block-quantize emitters must hand on a Vector"


_PW_M, _PW_N, _PW_K = 128, 128, 128


def _pw_matmul_graph():
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, _PW_M, _PW_K], stride=[_PW_M * _PW_K, _PW_K, 1])
    B = g.tensor(name="B", dim=[1, _PW_K, _PW_N], stride=[_PW_K * _PW_N, 1, _PW_K])
    C = g.matmul(A=A, B=B, name="mm")
    return g, C


def _pw_elem(gen):
    return ([1, _PW_M, _PW_N], [_PW_M * _PW_N, _PW_N, 1], gen)


def _pw_scalar(v):
    return ([1, 1, 1], [1, 1, 1], lambda: torch.full((1, 1, 1), v, device="cuda", dtype=torch.float32))


def _sigp(x):
    return torch.sigmoid(x)


_PW_CASES = {
    "elu": (lambda g, C, x: g.elu(input=C), lambda acc, x: torch.nn.functional.elu(acc), [], "close"),
    "softplus": (lambda g, C, x: g.softplus(input=C), lambda acc, x: torch.nn.functional.softplus(acc), [], "close"),
    "tan": (
        lambda g, C, x: g.tan(input=g.mul(a=C, b=x[0], name="sc")),
        lambda acc, x: torch.tan(acc * 0.3),
        [_pw_scalar(0.3)],
        "close",
    ),
    "logical_not": (
        lambda g, C, x: g.logical_not(input=g.relu(input=C, name="r")),
        lambda acc, x: (torch.relu(acc) == 0).float(),
        [],
        "frac",
    ),
    "leaky_relu": (
        lambda g, C, x: g.leaky_relu(input=C, negative_slope=0.1),
        lambda acc, x: torch.nn.functional.leaky_relu(acc, 0.1),
        [],
        "close",
    ),
    "relu_clips": (
        lambda g, C, x: g.relu(input=C, lower_clip=0.5, upper_clip=3.0),
        lambda acc, x: torch.clamp(acc, 0.5, 3.0),
        [],
        "close",
    ),
    "relu_leaky_upper": (
        lambda g, C, x: g.relu(input=C, negative_slope=0.2, upper_clip=2.0),
        lambda acc, x: torch.clamp(torch.nn.functional.leaky_relu(acc, 0.2), max=2.0),
        [],
        "close",
    ),
    "swish_beta": (
        lambda g, C, x: g.swish(input=C, swish_beta=2.0),
        lambda acc, x: acc * torch.sigmoid(2.0 * acc),
        [],
        "close",
    ),
    "gen_index_row": (
        lambda g, C, x: g.gen_index(input=C, axis=1),
        lambda acc, x: torch.arange(_PW_M, device="cuda", dtype=torch.float32).view(-1, 1).expand(_PW_M, _PW_N).contiguous(),
        [],
        "close",
    ),
    "gen_index_col": (
        lambda g, C, x: g.gen_index(input=C, axis=2),
        lambda acc, x: torch.arange(_PW_N, device="cuda", dtype=torch.float32).view(1, -1).expand(_PW_M, _PW_N).contiguous(),
        [],
        "close",
    ),
    "mod": (
        lambda g, C, x: g.mod(input0=C, input1=x[0]),
        lambda acc, x: torch.fmod(acc, x[0][0]),
        [_pw_elem(lambda: torch.rand(1, _PW_M, _PW_N, device="cuda") + 0.5)],
        "frac",
    ),
    "cmp_le": (
        lambda g, C, x: g.cmp_le(input=C, comparison=x[0]),
        lambda acc, x: (acc <= x[0][0]).float(),
        [_pw_elem(lambda: torch.randn(1, _PW_M, _PW_N, device="cuda"))],
        "frac",
    ),
    "cmp_ge": (
        lambda g, C, x: g.cmp_ge(input=C, comparison=x[0]),
        lambda acc, x: (acc >= x[0][0]).float(),
        [_pw_elem(lambda: torch.randn(1, _PW_M, _PW_N, device="cuda"))],
        "frac",
    ),
    "cmp_lt": (
        lambda g, C, x: g.cmp_lt(input=C, comparison=x[0]),
        lambda acc, x: (acc < x[0][0]).float(),
        [_pw_elem(lambda: torch.randn(1, _PW_M, _PW_N, device="cuda"))],
        "frac",
    ),
    "cmp_gt": (
        lambda g, C, x: g.cmp_gt(input=C, comparison=x[0]),
        lambda acc, x: (acc > x[0][0]).float(),
        [_pw_elem(lambda: torch.randn(1, _PW_M, _PW_N, device="cuda"))],
        "frac",
    ),
    "cmp_eq_floor": (
        lambda g, C, x: g.cmp_eq(input=g.floor(input=C, name="f"), comparison=x[0]),
        lambda acc, x: (torch.floor(acc) == x[0][0]).float(),
        [_pw_elem(lambda: torch.randint(-2, 3, (1, _PW_M, _PW_N), device="cuda").float())],
        "frac",
    ),
    "cmp_neq_floor": (
        lambda g, C, x: g.cmp_neq(input=g.floor(input=C, name="f"), comparison=x[0]),
        lambda acc, x: (torch.floor(acc) != x[0][0]).float(),
        [_pw_elem(lambda: torch.randint(-2, 3, (1, _PW_M, _PW_N), device="cuda").float())],
        "frac",
    ),
    "logical_and": (
        lambda g, C, x: g.logical_and(a=g.cmp_gt(input=C, comparison=x[0], name="m1"), b=x[1]),
        lambda acc, x: ((acc > 0) & (x[1][0] != 0)).float(),
        [_pw_scalar(0.0), _pw_elem(lambda: torch.randint(0, 2, (1, _PW_M, _PW_N), device="cuda").float())],
        "frac",
    ),
    "logical_or": (
        lambda g, C, x: g.logical_or(a=g.cmp_gt(input=C, comparison=x[0], name="m1"), b=x[1]),
        lambda acc, x: ((acc > 0) | (x[1][0] != 0)).float(),
        [_pw_scalar(0.0), _pw_elem(lambda: torch.randint(0, 2, (1, _PW_M, _PW_N), device="cuda").float())],
        "frac",
    ),
    "relu_backward": (
        lambda g, C, x: g.relu_backward(loss=C, input=x[0]),
        lambda acc, x: acc * (x[0][0] > 0).float(),
        [_pw_elem(lambda: torch.randn(1, _PW_M, _PW_N, device="cuda"))],
        "frac",
    ),
    "relu_backward_clips": (
        lambda g, C, x: g.relu_backward(loss=C, input=x[0], lower_clip=-1.0, upper_clip=1.0),
        lambda acc, x: acc * ((x[0][0] > -1.0) & (x[0][0] < 1.0)).float(),
        [_pw_elem(lambda: torch.randn(1, _PW_M, _PW_N, device="cuda"))],
        "frac",
    ),
    "leaky_relu_backward": (
        lambda g, C, x: g.leaky_relu_backward(loss=C, input=x[0], negative_slope=0.1),
        lambda acc, x: acc * torch.where(x[0][0] > 0, 1.0, 0.1),
        [_pw_elem(lambda: torch.randn(1, _PW_M, _PW_N, device="cuda"))],
        "frac",
    ),
    "sigmoid_backward": (
        lambda g, C, x: g.sigmoid_backward(loss=C, input=x[0]),
        lambda acc, x: acc * _sigp(x[0][0]) * (1 - _sigp(x[0][0])),
        [_pw_elem(lambda: torch.randn(1, _PW_M, _PW_N, device="cuda"))],
        "close",
    ),
    "tanh_backward": (
        lambda g, C, x: g.tanh_backward(loss=C, input=x[0]),
        lambda acc, x: acc * (1 - torch.tanh(x[0][0]) ** 2),
        [_pw_elem(lambda: torch.randn(1, _PW_M, _PW_N, device="cuda"))],
        "close",
    ),
    "elu_backward": (
        lambda g, C, x: g.elu_backward(loss=C, input=x[0]),
        lambda acc, x: acc * torch.where(x[0][0] > 0, torch.ones_like(x[0][0]), torch.exp(x[0][0])),
        [_pw_elem(lambda: torch.randn(1, _PW_M, _PW_N, device="cuda"))],
        "frac",
    ),
    "gelu_backward": (
        lambda g, C, x: g.gelu_backward(loss=C, input=x[0]),
        lambda acc, x: acc * (0.5 * (1 + torch.erf(x[0][0] * 0.7071067811865476)) + x[0][0] * torch.exp(-x[0][0] ** 2 / 2) * 0.3989422804014327),
        [_pw_elem(lambda: torch.randn(1, _PW_M, _PW_N, device="cuda"))],
        "close",
    ),
    "gelu_tanh_backward": (
        lambda g, C, x: g.gelu_approx_tanh_backward(loss=C, input=x[0]),
        lambda acc, x: acc
        * (
            0.5 * (1 + torch.tanh(0.7978845608028654 * (x[0][0] + 0.044715 * x[0][0] ** 3)))
            + 0.5
            * x[0][0]
            * (1 - torch.tanh(0.7978845608028654 * (x[0][0] + 0.044715 * x[0][0] ** 3)) ** 2)
            * 0.7978845608028654
            * (1 + 0.134145 * x[0][0] ** 2)
        ),
        [_pw_elem(lambda: torch.randn(1, _PW_M, _PW_N, device="cuda"))],
        "close",
    ),
    "softplus_backward": (
        lambda g, C, x: g.softplus_backward(loss=C, input=x[0]),
        lambda acc, x: acc * _sigp(x[0][0]),
        [_pw_elem(lambda: torch.randn(1, _PW_M, _PW_N, device="cuda"))],
        "close",
    ),
    "swish_backward": (
        lambda g, C, x: g.swish_backward(loss=C, input=x[0], swish_beta=2.0),
        lambda acc, x: acc * (_sigp(2.0 * x[0][0]) + 2.0 * x[0][0] * _sigp(2.0 * x[0][0]) * (1 - _sigp(2.0 * x[0][0]))),
        [_pw_elem(lambda: torch.randn(1, _PW_M, _PW_N, device="cuda"))],
        "close",
    ),
    "binary_select": (
        lambda g, C, x: g.binary_select(input0=C, input1=x[0], mask=x[1]),
        lambda acc, x: torch.where(x[1][0] != 0, acc, x[0][0]),
        [
            _pw_elem(lambda: torch.randn(1, _PW_M, _PW_N, device="cuda")),
            _pw_elem(lambda: torch.randint(0, 2, (1, _PW_M, _PW_N), device="cuda").float()),
        ],
        "frac",
    ),
    "scale_alias": (
        lambda g, C, x: g.scale(input=C, scale=x[0]),
        lambda acc, x: acc * 0.5,
        [_pw_scalar(0.5)],
        "close",
    ),
}


@requires_sm100
@pytest.mark.parametrize("case", sorted(_PW_CASES), ids=str)
def test_pointwise_mode_coverage(case):
    build, ref_fn, aux_specs, mode = _PW_CASES[case]
    g, C = _pw_matmul_graph()
    aux_tensors = [g.tensor(name=f"x{i}", dim=dims, stride=strides, data_type=cudnn.data_type.FLOAT) for i, (dims, strides, _gen) in enumerate(aux_specs)]
    y = build(g, C, aux_tensors)
    y.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)

    compiled = _plan(g, config=by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1"), cta_group=1)

    torch.manual_seed(0)
    a = torch.randn(1, _PW_M, _PW_K, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(1, _PW_N, _PW_K, device="cuda", dtype=torch.bfloat16)
    aux_bufs = [gen() for (_d, _s, gen) in aux_specs]
    out = torch.empty(1, _PW_M, _PW_N, device="cuda", dtype=torch.bfloat16)
    bd = compiled.binding
    vp = {bd.a_operands[0]: a, bd.b_operands[0]: b, bd.outputs[0]: out}
    vp.update({t: buf for t, buf in zip(bd.aux, [aux_bufs[i] for i in _pw_aux_order(compiled, aux_bufs)])})
    compiled(vp)
    torch.cuda.synchronize()

    acc = torch.einsum("bmk,bnk->bmn", a.float(), b.float())[0]
    ref = ref_fn(acc, aux_bufs).to(torch.bfloat16).float()
    got = out[0].float()
    if mode == "close":
        torch.testing.assert_close(got, ref, atol=5e-2, rtol=5e-2)
    else:
        bad = (~torch.isclose(got, ref, atol=5e-2, rtol=5e-2)).float().mean().item()
        assert bad < 2e-3, bad


def _pw_aux_order(compiled, aux_bufs):
    order = []
    for ref in compiled.chain.aux_tensors:
        order.append(int(ref.name[1:]))
    return order


def _prod_dim1(s):
    """Product along dim 1 on the CPU -- torch's device-side `prod` NVRTC-JITs its kernel."""
    return s.cpu().prod(dim=1, keepdim=True).to(s.device)


_RED_CASES = {
    "avg_full": (cudnn.reduction_mode.AVG, (1, 1, 1), lambda s: s.mean().view(1, 1, 1), 0.0),
    "avg_row": (cudnn.reduction_mode.AVG, (1, _PW_M, 1), lambda s: s.mean(dim=1, keepdim=True).view(1, _PW_M, 1), 0.0),
    "avg_col": (cudnn.reduction_mode.AVG, (1, 1, _PW_N), lambda s: s.mean(dim=0, keepdim=True).view(1, 1, _PW_N), 0.0),
    "norm1_row": (cudnn.reduction_mode.NORM1, (1, _PW_M, 1), lambda s: s.abs().sum(dim=1, keepdim=True).view(1, _PW_M, 1), 0.0),
    "mul_row": (cudnn.reduction_mode.MUL, (1, _PW_M, 1), lambda s: _prod_dim1(s).view(1, _PW_M, 1), 1.0),
    "mul_no_zeros_row": (
        cudnn.reduction_mode.MUL_NO_ZEROS,
        (1, _PW_M, 1),
        lambda s: _prod_dim1(torch.where(s == 0, torch.ones_like(s), s)).view(1, _PW_M, 1),
        1.0,
    ),
}


@requires_sm100
@pytest.mark.parametrize("case", sorted(_RED_CASES), ids=str)
def test_reduction_mode_coverage(case):
    mode, dim, ref_fn, init = _RED_CASES[case]
    g, C = _pw_matmul_graph()
    zero = g.tensor(name="x0", dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    if case.startswith("mul_no_zeros"):
        src = g.mul(a=g.add(a=g.sigmoid(input=C, name="sg"), b=zero, name="sh"), b=g.cmp_gt(input=C, comparison=zero, name="mk"), name="s")
    elif case.startswith("mul"):
        src = g.add(a=g.sigmoid(input=C, name="sg"), b=zero, name="s")
    else:
        src = g.swish(input=C, name="s")
    src.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)
    red = g.reduction(input=src, mode=mode, name="red")
    red.set_dim(list(dim)).set_stride([dim[1] * dim[2], dim[2], 1])
    red.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    compiled = _plan(g, config=by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1"), cta_group=1)

    torch.manual_seed(0)
    a = torch.randn(1, _PW_M, _PW_K, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(1, _PW_N, _PW_K, device="cuda", dtype=torch.bfloat16)
    z = torch.full((1, 1, 1), 0.7, device="cuda", dtype=torch.float32)
    out = torch.empty(1, _PW_M, _PW_N, device="cuda", dtype=torch.bfloat16)
    r = torch.full(dim, init, device="cuda", dtype=torch.float32)
    bd = compiled.binding
    vp = {bd.a_operands[0]: a, bd.b_operands[0]: b, bd.outputs[0]: out, bd.outputs[1]: r}
    if bd.aux:
        vp[bd.aux[0]] = z
    compiled(vp)
    torch.cuda.synchronize()

    acc = torch.einsum("bmk,bnk->bmn", a.float(), b.float())[0]
    if case.startswith("mul_no_zeros"):
        s = (torch.sigmoid(acc) + 0.7) * (acc > 0.7).float()
    elif case.startswith("mul"):
        s = torch.sigmoid(acc) + 0.7
    else:
        s = acc * torch.sigmoid(acc)
    s = s.to(torch.bfloat16).float()
    ref = ref_fn(s.double()).float()
    torch.testing.assert_close(r, ref, atol=5e-2, rtol=2e-2)


@requires_sm100
def test_norm2_is_the_one_reduction_mode_this_engine_declines():
    """It is the only mode that is not finished when the kernel is.

    Its taps land through cross-CTA atomics, so the square root cannot go in the
    epilogue -- it has to run after every CTA has contributed. That is a device
    operation this engine does not own, and the version that borrowed the
    caller's ``sqrt_()`` worked only while the caller passed a torch tensor.

    Declining costs nothing reachable: the BACKEND refuses a norm2 reduction
    descriptor while the graph is still being lowered, so no public
    ``execute()`` ever gets a plan for one either (see
    ``test_public_execute_flavors.py::test_norm2_reduction_is_refused_at_build``).
    Recorded here because the mode is otherwise in every list of the reductions
    the epilogue supports.
    """
    from cudnn.gemm.frost.compiler import probe_supported

    g, C = _pw_matmul_graph()
    src = g.swish(input=C, name="s")
    src.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)
    red = g.reduction(input=src, mode=cudnn.reduction_mode.NORM2, name="red")
    red.set_dim([1, _PW_M, 1]).set_stride([_PW_M, 1, 1])
    red.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    with pytest.raises(NotImplementedError, match="square root"):
        probe_supported(g)
