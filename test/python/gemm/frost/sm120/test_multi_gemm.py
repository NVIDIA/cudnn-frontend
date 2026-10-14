# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-GEMM: K parallel matmuls (shared shape/layout, shared-or-distinct
operands) feeding one shared pointwise epilogue. Single-GEMM must stay unchanged."""

from __future__ import annotations

import cudnn
import cudnn.gemm.frost  # noqa: F401  (installs hook)
import pytest
import torch

from gemm_test_utils import (
    requires_sm120,
    Plan as _plan,
    vp_mg as _vp_mg,
    reduction_ref as _reduction_ref,
)

from cudnn.gemm.frost.epilogue_codegen import generate
from cudnn.gemm.frost.graph_analyzer import analyze
from cudnn.gemm.frost.tile_config import by_name

pytestmark = pytest.mark.L0


def _graph(io=cudnn.data_type.BFLOAT16):
    return cudnn.pygraph(
        io_data_type=io,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )


def _A(g, M, K, name="A"):
    return g.tensor(name=name, dim=[1, M, K], stride=[M * K, K, 1])


def _B(g, K, N, name="B"):
    return g.tensor(name=name, dim=[1, K, N], stride=[K * N, 1, K])


# SM120 runs one fixed 16x16x32 warp-MMA pair with no cluster and no CTA pair: the
# parent's 1-CTA geometries map to the 4x2 compute-warp grid, its 2-CTA ones to 2x4.
_N128_CFG = by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2")


# --- Analyzer + codegen (no GPU) ---


def test_single_gemm_unchanged() -> None:
    """A single matmul still analyzes as num_gemms==1."""
    M, N, K = 256, 256, 128
    g = _graph()
    A, B = _A(g, M, K), _B(g, K, N)
    C = g.matmul(A=A, B=B, name="mm")
    Y = g.relu(input=C, name="r")
    Y.set_output(True)
    chain = analyze(g)
    assert chain.num_gemms == 1
    assert chain.num_a_operands == 1 and chain.num_b_operands == 1
    assert chain.gemm_operands == [(0, 0)]
    assert not chain.is_multi_gemm
    # single-GEMM codegen uses the legacy `vec_f32` root, never `vec_f32_1`
    epi = generate(chain).epilogue
    assert "vec_f32_1" not in epi


def test_dual_gemm_shared_a_detected() -> None:
    """silu(A@B0) * (A@B1) * scale: 2 GEMMs, A shared, B distinct."""
    M, N, K = 256, 256, 128
    g = _graph()
    A = _A(g, M, K)
    B0, B1 = _B(g, K, N, "B0"), _B(g, K, N, "B1")
    sc = g.tensor(name="scale", dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    C0 = g.matmul(A=A, B=B0, name="mm0")
    C1 = g.matmul(A=A, B=B1, name="mm1")
    S0 = g.swish(input=C0, name="s")
    MU = g.mul(a=S0, b=C1, name="m")
    DQ = g.mul(a=MU, b=sc, name="d")
    DQ.set_output(True)
    chain = analyze(g)
    assert chain.num_gemms == 2
    assert chain.num_a_operands == 1 and chain.num_b_operands == 2
    assert chain.gemm_operands == [(0, 0), (0, 1)]
    # the re-merging mul is a fan-in over GEMM 0 (via silu) and GEMM 1 directly
    from cudnn.gemm.frost.fusion_ir import gemm_source

    mul_op = chain.ops[1]
    assert mul_op.op == "mul"
    assert mul_op.parent_idx == 0  # silu result (op 0)
    assert mul_op.parent_idx_b == gemm_source(1)  # GEMM 1 output (-2)
    # codegen binds both GEMM roots into the shared chain
    epi = generate(chain).epilogue
    assert "vec_f32" in epi and "vec_f32_1" in epi


def test_distinct_operands_detected() -> None:
    """(A0@B0), (A1@B1): 2 distinct A and 2 distinct B operands."""
    M, N, K = 256, 128, 128
    g = _graph()
    A0, A1 = _A(g, M, K, "A0"), _A(g, M, K, "A1")
    B0, B1 = _B(g, K, N, "B0"), _B(g, K, N, "B1")
    C0 = g.matmul(A=A0, B=B0, name="m0")
    C1 = g.matmul(A=A1, B=B1, name="m1")
    R = g.relu(input=C0, name="r")
    Y = g.add(a=R, b=C1, name="a")
    Y.set_output(True)
    chain = analyze(g)
    assert chain.num_gemms == 2
    assert chain.num_a_operands == 2 and chain.num_b_operands == 2
    assert chain.gemm_operands == [(0, 0), (1, 1)]


def test_three_gemm_detected() -> None:
    """C0*C1 + C2 over three GEMMs sharing A."""
    M, N, K = 256, 128, 128
    g = _graph()
    A = _A(g, M, K)
    Bs = [_B(g, K, N, f"B{i}") for i in range(3)]
    Cs = [g.matmul(A=A, B=Bs[i], name=f"m{i}") for i in range(3)]
    P = g.mul(a=Cs[0], b=Cs[1], name="p")
    Y = g.add(a=P, b=Cs[2], name="ad")
    Y.set_output(True)
    chain = analyze(g)
    assert chain.num_gemms == 3
    assert chain.num_a_operands == 1 and chain.num_b_operands == 3
    assert chain.gemm_operands == [(0, 0), (0, 1), (0, 2)]
    epi = generate(chain).epilogue
    assert "vec_f32_1" in epi and "vec_f32_2" in epi


def test_multi_gemm_reduction_from_fused_output_detected() -> None:
    """A reduction tap may consume the fused multi-GEMM terminal op."""
    M, N, K = 256, 128, 128
    g = _graph()
    A = _A(g, M, K)
    B0, B1 = _B(g, K, N, "B0"), _B(g, K, N, "B1")
    C0 = g.matmul(A=A, B=B0, name="m0")
    C1 = g.matmul(A=A, B=B1, name="m1")
    Y = g.add(a=C0, b=C1, name="add")
    Y.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    R = g.reduction(input=Y, mode=cudnn.reduction_mode.AMAX, name="red")
    R.set_dim([1, 1, 1]).set_stride([1, 1, 1])
    R.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    chain = analyze(g)
    assert chain.is_multi_gemm
    assert len(chain.reductions) == 1
    assert chain.reductions[0].source_ref == 0
    assert [o.source for o in chain.outputs] == ["op_0", "reduction_0"]
    epi = generate(chain).epilogue
    assert "gC_tap_0_ptr" in epi
    assert "atomic_fmax" in epi


def test_heterogeneous_gemms_rejected() -> None:
    """Parallel GEMMs with different shapes are out of POC scope."""
    g = _graph()
    A = _A(g, 256, 128)
    B0 = _B(g, 128, 256, "B0")
    B1 = _B(g, 128, 128, "B1")  # different N
    C0 = g.matmul(A=A, B=B0, name="m0")
    # mm1 needs its own A of matching K → shape mismatch across the two GEMMs
    A2 = _A(g, 256, 128, "A2")
    C1 = g.matmul(A=A2, B=B1, name="m1")
    Y = g.add(a=C0, b=C1, name="a")
    Y.set_output(True)
    with pytest.raises(ValueError, match="must share shape / layout / dtype"):
        analyze(g)


_N256_C2_CFG = by_name("CONFIG_sm120_128x256x128_16x16x32_cluster1x1_warps2x4")


@requires_sm120
def test_multi_gemm_2ctamma_compiles() -> None:
    """The parent's 2-CTA geometry (128x256, here the 2x4 warp grid) compiles a
    dual-GEMM graph: one A and two B tiles per stage leave a single AB stage."""
    M, N, K = 256, 256, 128
    g = _graph()
    A = _A(g, M, K)
    B0, B1 = _B(g, K, N, "B0"), _B(g, K, N, "B1")
    C0 = g.matmul(A=A, B=B0, name="m0")
    C1 = g.matmul(A=A, B=B1, name="m1")
    Y = g.add(a=C0, b=C1, name="a")
    Y.set_output(True)
    compiled = _plan(g, _N256_C2_CFG, cta_group=1)
    assert compiled.chain.num_gemms == 2


def test_multi_gemm_mainloop_template_rejected() -> None:
    """A multi-GEMM graph never selects a mainloop template (registry guard rejects it)."""
    from cudnn.gemm.frost.kernel_registry import TEMPLATES

    M, N, K = 256, 256, 128
    g = _graph()
    A = _A(g, M, K)
    B0, B1 = _B(g, K, N, "B0"), _B(g, K, N, "B1")
    C0 = g.matmul(A=A, B=B0, name="m0")
    C1 = g.matmul(A=A, B=B1, name="m1")
    Y = g.add(a=C0, b=C1, name="a")
    Y.set_output(True)
    chain = analyze(g)
    mainloop_tmpl = next(t for t in TEMPLATES if t.supports_mainloop_fusion)
    assert not mainloop_tmpl.supports_multi_gemm
    assert mainloop_tmpl.accepts(chain, _N256_C2_CFG) is not None


# --- End-to-end correctness (GPU) ---

_GPU = requires_sm120


def _rand(M, N, K, scale=1.0):
    a = torch.randn(1, M, K, device="cuda", dtype=torch.bfloat16) * scale
    b = torch.randn(1, N, K, device="cuda", dtype=torch.bfloat16) * scale
    return a, b


def _mm(a, b):
    return torch.einsum("bmk,bnk->bmn", a.float(), b.float())


def _mk_bf16(M: int, N: int, K: int, B: int = 1):
    torch.manual_seed(0)
    a = torch.randn(B, M, K, device="cuda", dtype=torch.bfloat16) * 0.4
    b0 = torch.randn(B, N, K, device="cuda", dtype=torch.bfloat16) * 0.4
    b1 = torch.randn(B, N, K, device="cuda", dtype=torch.bfloat16) * 0.4
    return a, b0, b1


def _mk_int8(M: int, N: int, K: int, B: int = 1):
    torch.manual_seed(0)
    a = torch.randint(-4, 4, (B, M, K), dtype=torch.int8, device="cuda")
    b0 = torch.randint(-4, 4, (B, N, K), dtype=torch.int8, device="cuda")
    b1 = torch.randint(-4, 4, (B, N, K), dtype=torch.int8, device="cuda")
    return a, b0, b1


def _build_dual_add_reduction(
    B,
    M,
    N,
    K,
    mode,
    red_dims,
    *,
    out_dtype=cudnn.data_type.FLOAT,
    red_dtype=cudnn.data_type.FLOAT,
    red_compute=cudnn.data_type.FLOAT,
):
    g = _graph()
    A = g.tensor(name="A", dim=[B, M, K], stride=[M * K, K, 1])
    B0 = g.tensor(name="B0", dim=[B, K, N], stride=[K * N, 1, K])
    B1 = g.tensor(name="B1", dim=[B, K, N], stride=[K * N, 1, K])
    C0 = g.matmul(A=A, B=B0, name="m0")
    C1 = g.matmul(A=A, B=B1, name="m1")
    Y = g.add(a=C0, b=C1, name="add")
    Y.set_output(True).set_data_type(out_dtype)
    R = g.reduction(input=Y, mode=mode, name="red", compute_data_type=red_compute)
    R.set_dim(red_dims).set_stride([red_dims[1] * red_dims[2], red_dims[2], 1])
    R.set_output(True).set_data_type(red_dtype)
    return g


def _build_dual_add_reduction_int32(B, M, N, K, mode, red_dims):
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.INT8,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.INT32,
    )
    A = g.tensor(name="A", dim=[B, M, K], stride=[M * K, K, 1])
    B0 = g.tensor(name="B0", dim=[B, K, N], stride=[K * N, 1, K])
    B1 = g.tensor(name="B1", dim=[B, K, N], stride=[K * N, 1, K])
    C0 = g.matmul(A=A, B=B0, name="m0")
    C1 = g.matmul(A=A, B=B1, name="m1")
    C0.set_data_type(cudnn.data_type.INT32)
    C1.set_data_type(cudnn.data_type.INT32)
    Y = g.add(a=C0, b=C1, name="add")
    Y.set_output(True).set_data_type(cudnn.data_type.INT32)
    R = g.reduction(
        input=Y,
        mode=mode,
        name="red",
        compute_data_type=cudnn.data_type.INT32,
    )
    R.set_dim(red_dims).set_stride([red_dims[1] * red_dims[2], red_dims[2], 1])
    R.set_output(True).set_data_type(cudnn.data_type.INT32)
    return g


def _assert_red_close(actual, expected, mode, *, exact=False):
    if exact:
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        return
    tol = dict(atol=1e-1, rtol=1e-2)
    if mode == cudnn.reduction_mode.ADD:
        tol = dict(atol=2e-1, rtol=2e-2)
    torch.testing.assert_close(actual, expected, **tol)


# The parent's split CTA tile (two MMA-M blocks per CTA): on sm120 a 256-row tile
# is four 16-row m-fragments per warp of the 4x2 grid.
_SPLIT_CFG = by_name("CONFIG_sm120_256x128x128_16x16x32_cluster1x1_warps4x2")


@_GPU
def test_multi_gemm_times_split_tile_over_tmem_budget_is_rejected() -> None:
    # The parent's TMEM column budget is a tcgen05 fact; sm120 keeps the
    # accumulators in registers and its one hard multi-GEMM gate is SMEM: a
    # 256x256 stage of A + two B tiles (96 KB) plus the epilogue staging misses it.
    M, N, K = 256, 256, 128
    g = _graph()
    A = _A(g, M, K)
    B0, B1 = _B(g, K, N, "B0"), _B(g, K, N, "B1")
    Y = g.mul(a=g.swish(input=g.matmul(A=A, B=B0, name="mm0")), b=g.matmul(A=A, B=B1, name="mm1"))
    Y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    with pytest.raises(NotImplementedError, match="does not fit the SMEM budget"):
        _plan(g, by_name("CONFIG_sm120_256x256x128_16x16x32_cluster1x1_warps4x2"), cta_group=1)


@_GPU
@pytest.mark.parametrize("config", [_N128_CFG, _SPLIT_CFG], ids=["single_mma", "num_mma_m2"])
def test_dual_silu_mul_end_to_end(config) -> None:
    M, N, K = 256, 256, 128
    g = _graph()
    A = _A(g, M, K)
    B0, B1 = _B(g, K, N, "B0"), _B(g, K, N, "B1")
    sc = g.tensor(name="scale", dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    C0 = g.matmul(A=A, B=B0, name="mm0")
    C1 = g.matmul(A=A, B=B1, name="mm1")
    S0 = g.swish(input=C0, name="s")
    MU = g.mul(a=S0, b=C1, name="m")
    DQ = g.mul(a=MU, b=sc, name="d")
    DQ.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    compiled = _plan(g, config, cta_group=1)

    torch.manual_seed(0)
    a, b0 = _rand(M, N, K, 0.4)
    _, b1 = _rand(M, N, K, 0.4)
    scale_t = torch.tensor([[[0.5]]], device="cuda", dtype=torch.float32)
    c = torch.zeros(1, M, N, device="cuda", dtype=torch.bfloat16)
    compiled(_vp_mg(compiled, [(a, b0), (a, b1)], c, scale_t))
    torch.cuda.synchronize()
    ref = (torch.nn.functional.silu(_mm(a, b0)) * _mm(a, b1) * 0.5).bfloat16()
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)


@_GPU
def test_distinct_operands_end_to_end() -> None:
    """relu(A0@B0) + (A1@B1) — no shared operands."""
    M, N, K = 256, 128, 128
    g = _graph()
    A0, A1 = _A(g, M, K, "A0"), _A(g, M, K, "A1")
    B0, B1 = _B(g, K, N, "B0"), _B(g, K, N, "B1")
    C0 = g.matmul(A=A0, B=B0, name="m0")
    C1 = g.matmul(A=A1, B=B1, name="m1")
    R = g.relu(input=C0, name="r")
    Y = g.add(a=R, b=C1, name="a")
    Y.set_output(True)
    compiled = _plan(g, _N128_CFG, cta_group=1)

    torch.manual_seed(1)
    a0, b0 = _rand(M, N, K, 0.4)
    a1, b1 = _rand(M, N, K, 0.4)
    c = torch.zeros(1, M, N, device="cuda", dtype=torch.bfloat16)
    compiled(_vp_mg(compiled, [(a0, b0), (a1, b1)], c))
    torch.cuda.synchronize()
    ref = (torch.relu(_mm(a0, b0)) + _mm(a1, b1)).bfloat16()
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)


@_GPU
def test_three_gemm_end_to_end() -> None:
    """C0*C1 + C2 over three GEMMs sharing A (N=128 → fits TMEM)."""
    M, N, K = 256, 128, 128
    g = _graph()
    A = _A(g, M, K)
    Bs = [_B(g, K, N, f"B{i}") for i in range(3)]
    Cs = [g.matmul(A=A, B=Bs[i], name=f"m{i}") for i in range(3)]
    P = g.mul(a=Cs[0], b=Cs[1], name="p")
    Y = g.add(a=P, b=Cs[2], name="ad")
    Y.set_output(True)
    compiled = _plan(g, _N128_CFG, cta_group=1)
    assert compiled.chain.num_gemms == 3

    torch.manual_seed(2)
    a = torch.randn(1, M, K, device="cuda", dtype=torch.bfloat16) * 0.3
    bs = [torch.randn(1, N, K, device="cuda", dtype=torch.bfloat16) * 0.3 for _ in range(3)]
    c = torch.zeros(1, M, N, device="cuda", dtype=torch.bfloat16)
    compiled(_vp_mg(compiled, [(a, bs[0]), (a, bs[1]), (a, bs[2])], c))
    torch.cuda.synchronize()
    ref = (_mm(a, bs[0]) * _mm(a, bs[1]) + _mm(a, bs[2])).bfloat16()
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    "mode,ref_dims",
    (
        (cudnn.reduction_mode.ADD, (0, 1, 2)),
        (cudnn.reduction_mode.AMAX, (0, 1, 2)),
        (cudnn.reduction_mode.MAX, (0, 1, 2)),
        (cudnn.reduction_mode.MIN, (0, 1, 2)),
    ),
    ids=("add", "amax", "max", "min"),
)
@_GPU
def test_multi_gemm_reduction_scalar_fp32(mode, ref_dims) -> None:
    M, N, K = 256, 128, 128
    g = _build_dual_add_reduction(1, M, N, K, mode, [1, 1, 1])
    compiled = _plan(g, _N128_CFG, cta_group=1)

    a, b0, b1 = _mk_bf16(M, N, K)
    c_term = torch.empty(1, M, N, dtype=torch.float32, device="cuda")
    c_red = torch.empty(1, 1, 1, dtype=torch.float32, device="cuda")
    compiled(_vp_mg(compiled, [(a, b0), (a, b1)], [c_term, c_red]))
    torch.cuda.synchronize()

    ref_term = _mm(a, b0) + _mm(a, b1)
    torch.testing.assert_close(c_term, ref_term, atol=1e-1, rtol=1e-2)
    _assert_red_close(c_red, _reduction_ref(c_term, mode, ref_dims), mode)


@pytest.mark.parametrize(
    "mode,ref_dims",
    (
        (cudnn.reduction_mode.ADD, (0, 1, 2)),
        (cudnn.reduction_mode.AMAX, (0, 1, 2)),
        (cudnn.reduction_mode.MAX, (0, 1, 2)),
        (cudnn.reduction_mode.MIN, (0, 1, 2)),
    ),
    ids=("add", "amax", "max", "min"),
)
@_GPU
def test_multi_gemm_reduction_scalar_int32(mode, ref_dims) -> None:
    M, N, K = 256, 128, 64
    g = _build_dual_add_reduction_int32(1, M, N, K, mode, [1, 1, 1])
    compiled = _plan(g, _N128_CFG, cta_group=1)

    a, b0, b1 = _mk_int8(M, N, K)
    c_term = torch.empty(1, M, N, dtype=torch.int32, device="cuda")
    c_red = torch.empty(1, 1, 1, dtype=torch.int32, device="cuda")
    compiled(_vp_mg(compiled, [(a, b0), (a, b1)], [c_term, c_red]))
    torch.cuda.synchronize()

    ref_term = torch.einsum("bmk,bnk->bmn", a.cpu().to(torch.int64), b0.cpu().to(torch.int64))
    ref_term += torch.einsum("bmk,bnk->bmn", a.cpu().to(torch.int64), b1.cpu().to(torch.int64))
    torch.testing.assert_close(c_term.cpu().to(torch.int64), ref_term, atol=0, rtol=0)
    _assert_red_close(
        c_red.cpu().to(torch.int64),
        _reduction_ref(ref_term, mode, ref_dims).to(torch.int64),
        mode,
        exact=True,
    )


@pytest.mark.parametrize(
    "mode,red_dims,red_stride,ref_dims",
    (
        (cudnn.reduction_mode.ADD, [1, 256, 1], [0, 2, 1], (0, 2)),
        (cudnn.reduction_mode.AMAX, [1, 1, 128], [0, 0, 2], (0, 1)),
    ),
    ids=("add_per_row", "amax_per_col"),
)
@_GPU
def test_multi_gemm_reduction_strided_output_fp32(mode, red_dims, red_stride, ref_dims) -> None:
    M, N, K = 256, 128, 128
    g = _build_dual_add_reduction(1, M, N, K, mode, red_dims)
    compiled = _plan(g, _N128_CFG, cta_group=1)

    a, b0, b1 = _mk_bf16(M, N, K)
    c_term = torch.empty(1, M, N, dtype=torch.float32, device="cuda")
    c_red = torch.empty_strided(tuple(red_dims), tuple(red_stride), dtype=torch.float32, device="cuda")
    assert not c_red.is_contiguous()
    compiled(_vp_mg(compiled, [(a, b0), (a, b1)], [c_term, c_red]))
    torch.cuda.synchronize()

    ref_term = _mm(a, b0) + _mm(a, b1)
    torch.testing.assert_close(c_term, ref_term, atol=1e-1, rtol=1e-2)
    _assert_red_close(c_red, _reduction_ref(c_term, mode, ref_dims), mode)


@pytest.mark.parametrize(
    "mode,ref_fn",
    (
        (cudnn.reduction_mode.ADD, lambda x: x.sum(dim=(0, 1, 2), keepdim=True)),
        (
            cudnn.reduction_mode.AMAX,
            lambda x: x.abs().amax(dim=(0, 1, 2), keepdim=True),
        ),
    ),
    ids=("add", "amax"),
)
@_GPU
def test_multi_gemm_reduction_big_cgrp_multi_cta(mode, ref_fn) -> None:
    """Dense multi-GEMM reductions use global atomics across many CTAs."""
    B, M, N, K = 2, 512, 256, 128
    cfg = by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps2x4")
    g = _build_dual_add_reduction(B, M, N, K, mode, [1, 1, 1])
    compiled = _plan(g, cfg, cta_group=1)

    a, b0, b1 = _mk_bf16(M, N, K, B)
    c_term = torch.empty(B, M, N, dtype=torch.float32, device="cuda")
    c_red = torch.empty(1, 1, 1, dtype=torch.float32, device="cuda")
    compiled(_vp_mg(compiled, [(a, b0), (a, b1)], [c_term, c_red]))
    torch.cuda.synchronize()

    ref_term = _mm(a, b0) + _mm(a, b1)
    torch.testing.assert_close(c_term, ref_term, atol=1e-1, rtol=1e-2)
    _assert_red_close(c_red, ref_fn(c_term), mode)


@_GPU
def test_single_gemm_still_runs_on_1ctamma() -> None:
    """Regression: a plain single-GEMM relu through the multi-GEMM-capable sm120 template."""
    M, N, K = 256, 256, 128
    g = _graph()
    A, B = _A(g, M, K), _B(g, K, N)
    C = g.matmul(A=A, B=B, name="mm")
    Y = g.relu(input=C, name="r")
    Y.set_output(True)
    compiled = _plan(g, _N128_CFG, cta_group=1)

    torch.manual_seed(3)
    a, b = _rand(M, N, K)
    c = torch.zeros(1, M, N, device="cuda", dtype=torch.bfloat16)
    compiled({A: a, B: b, Y: c})
    torch.cuda.synchronize()
    torch.testing.assert_close(c, torch.relu(_mm(a, b)).bfloat16(), atol=1e-1, rtol=1e-2)


# --------------------------------------------------------------------------- #
# sm120 regressions: the dense template's multi-GEMM -- one register
# accumulator per GEMM, one SMEM tile per DISTINCT operand per stage, each
# operand's fragments loaded once per k-block and reused by every GEMM reading
# it -- across warp grids, operand sharing patterns, MN-major / 8-bit operands,
# swap-AB, odd N; the SMEM gate shared by funnel, probe and renderer; and an
# automatic pick that keeps two AB stages.
# --------------------------------------------------------------------------- #

_MG_GUARD = 256


def _sm120_mg_graph(kind, M, N, K, *, io="bf16", out="bf16", a_major="k", b_major="k", ld=None):
    """kind: add (A@B0 + A@B1) | sharedb (A0@B + A1@B) | distinct (relu(A0@B0) + A1@B1)
    | triple (A@B0 * A@B1 + A@B2). Returns (graph, {name: tensor}, gemm pairs, output)."""
    dt = {"bf16": cudnn.data_type.BFLOAT16, "fp8": cudnn.data_type.FP8_E4M3, "int8": cudnn.data_type.INT8}[io]
    g = cudnn.pygraph(
        io_data_type=dt,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.INT32 if io == "int8" else cudnn.data_type.FLOAT,
    )

    def a(name):
        return g.tensor(name=name, dim=[1, M, K], stride=[M * K, K, 1] if a_major == "k" else [M * K, 1, M], data_type=dt)

    def b(name):
        return g.tensor(name=name, dim=[1, K, N], stride=[K * N, 1, K] if b_major == "k" else [K * N, N, 1], data_type=dt)

    pairs = {
        "add": [("A", "B0"), ("A", "B1")],
        "sharedb": [("A0", "B"), ("A1", "B")],
        "distinct": [("A0", "B0"), ("A1", "B1")],
        "triple": [("A", "B0"), ("A", "B1"), ("A", "B2")],
    }[kind]
    t = {}
    for an, bn in pairs:
        t.setdefault(an, a(an))
        t.setdefault(bn, b(bn))
    cs = [g.matmul(A=t[an], B=t[bn], name=f"m{i}") for i, (an, bn) in enumerate(pairs)]
    if io == "int8":
        for c in cs:
            c.set_data_type(cudnn.data_type.INT32)
    if kind == "distinct":
        y = g.add(a=g.relu(input=cs[0], name="r"), b=cs[1], name="y")
    elif kind == "triple":
        y = g.add(a=g.mul(a=cs[0], b=cs[1], name="p"), b=cs[2], name="y")
    else:
        y = g.add(a=cs[0], b=cs[1], name="y")
    y.set_output(True).set_data_type({"bf16": cudnn.data_type.BFLOAT16, "fp32": cudnn.data_type.FLOAT, "int32": cudnn.data_type.INT32}[out])
    y.set_dim([1, M, N]).set_stride([M * (ld or N), ld or N, 1])
    return g, t, pairs, y


_SM120_MG_CASES = {
    # id -> (kind, config, M, N, K, kwargs)
    "add_4x2_1stage": ("add", "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2", 200, 200, 320, {}),
    "add_2x4": ("add", "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps2x4", 200, 200, 320, {}),
    "add_8x1_k64": ("add", "CONFIG_sm120_128x128x64_16x16x32_cluster1x1_warps8x1", 200, 200, 320, {}),
    "add_1x8_k64": ("add", "CONFIG_sm120_128x128x64_16x16x32_cluster1x1_warps1x8", 200, 200, 320, {}),
    "add_64x128": ("add", "CONFIG_sm120_64x128x128_16x16x32_cluster1x1_warps2x4", 200, 200, 320, {}),
    "add_128x48_tail_group": ("add", "CONFIG_sm120_128x48x128_16x16x32_cluster1x1_warps8x1", 200, 200, 320, {}),
    "add_256x64_k64": ("add", "CONFIG_sm120_256x64x64_16x16x32_cluster1x1_warps4x2", 200, 200, 320, {}),
    "sharedb_2a": ("sharedb", "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2", 256, 128, 128, {}),
    "distinct_2a_2b": ("distinct", "CONFIG_sm120_128x64x128_16x16x32_cluster1x1_warps4x2", 256, 128, 128, {}),
    "triple_tails": ("triple", "CONFIG_sm120_128x64x128_16x16x32_cluster1x1_warps4x2", 200, 136, 320, {}),
    "add_m_major_a": ("add", "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2", 256, 256, 256, {"a_major": "m"}),
    "add_n_major_b": ("add", "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2", 256, 256, 256, {"b_major": "n"}),
    "add_fp8": ("add", "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2", 256, 256, 256, {"io": "fp8"}),
    "add_fp8_mn_major": ("add", "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2", 256, 256, 256, {"io": "fp8", "a_major": "m", "b_major": "n"}),
    "add_int8_int32": ("add", "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2", 256, 128, 64, {"io": "int8", "out": "int32"}),
    "add_odd_n_fp32": ("add", "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2", 256, 255, 256, {"out": "fp32"}),
    "add_ld257": ("add", "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2", 256, 256, 256, {"ld": 257}),
    "add_swap_ab": ("add", "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2", 200, 136, 256, {"swap": True}),
    "distinct_swap_ab": ("distinct", "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2", 128, 256, 128, {"swap": True}),
    "triple_swap_ab": ("triple", "CONFIG_sm120_128x64x128_16x16x32_cluster1x1_warps4x2", 136, 200, 128, {"swap": True}),
}


@requires_sm120
@pytest.mark.parametrize("case", list(_SM120_MG_CASES))
def test_sm120_dense_multi_gemm_numerics(case):
    """Exact (small-integer data, polynomial epilogues): every GEMM gets its own
    accumulator and its own operands; nothing is written outside the output."""
    from dataclasses import replace

    from cudnn.gemm.frost.sm120 import compiler as C

    kind, cfg_name, M, N, K, kw = _SM120_MG_CASES[case]
    kw = dict(kw)
    swap = kw.pop("swap", False)
    io, out, ld = kw.get("io", "bf16"), kw.get("out", "bf16"), kw.get("ld")
    g, t, pairs, y = _sm120_mg_graph(kind, M, N, K, **kw)
    compiled = C.jit_from_cudnn_graph(g, config=replace(by_name(cfg_name), swap_ab=swap))
    assert compiled.chain.num_gemms == len(pairs)
    torch.manual_seed(7)
    tdt = {"bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn, "int8": torch.int8}[io]
    data = {}
    for name in t:
        rows = M if name.startswith("A") else N
        x = torch.randint(-3 if io != "bf16" else -2, 4 if io != "bf16" else 3, (1, rows, K), device="cuda").to(tdt)
        mn_major = (name.startswith("A") and kw.get("a_major") == "m") or (name.startswith("B") and kw.get("b_major") == "n")
        data[name] = x.transpose(1, 2).contiguous().transpose(1, 2) if mn_major else x
    prods = [torch.einsum("bmk,bnk->bmn", data[an].double(), data[bn].double()) for an, bn in pairs]
    ref = {"add": lambda p: p[0] + p[1], "sharedb": lambda p: p[0] + p[1], "distinct": lambda p: p[0].relu() + p[1], "triple": lambda p: p[0] * p[1] + p[2]}[
        kind
    ](prods)
    odt = {"bf16": torch.bfloat16, "fp32": torch.float32, "int32": torch.int32}[out]
    pitch = ld or N
    nbytes = M * pitch * torch.empty((), dtype=odt).element_size()
    raw = torch.full((nbytes + 2 * _MG_GUARD,), 0xA5, dtype=torch.uint8, device="cuda")
    body = raw[_MG_GUARD : _MG_GUARD + nbytes].view(odt).view(1, M, pitch)
    compiled({**{t[n]: data[n] for n in t}, y: body[:, :, :N]})
    torch.cuda.synchronize()
    torch.testing.assert_close(body[:, :, :N].double(), (ref.to(torch.int64) if out == "int32" else ref.to(odt)).double(), atol=0, rtol=0)
    assert bool((raw[:_MG_GUARD] == 0xA5).all()) and bool((raw[-_MG_GUARD:] == 0xA5).all()), "a store ran past the output"
    if pitch > N:
        assert bool((body[:, :, N:].contiguous().view(torch.uint8) == 0xA5).all()), "a store wrote into the row padding"


@requires_sm120
@pytest.mark.parametrize("kind", ["add", "triple"])
@pytest.mark.parametrize("shape", [(128, 4096, 4096), (1024, 4096, 4096), (4096, 4096, 4096), (8192, 14336, 4096)], ids=lambda s: "x".join(map(str, s)))
def test_sm120_dense_multi_gemm_auto_pick_keeps_two_ab_stages(kind, shape):
    """The automatic pick for a dense multi-GEMM chain holds >= 2 AB stages (a
    one-stage ring serializes every K tile's TMA load and MMA; measured 1.3-1.4x
    slower for a dual SwiGLU), and the full probe accepts it."""
    from cudnn.gemm.frost.kernel_registry import select_template
    from cudnn.gemm.frost.sm120 import compiler as C

    chain = analyze(_sm120_mg_graph(kind, *shape)[0])
    cfg = C.plan_config(chain)
    assert cfg.pipeline == "sm120" and cfg.split_k_slices == 1
    assert select_template(chain, cfg).multi_gemm_ab_stages(chain, cfg)[0] >= 2, cfg.name
    C.probe_chain(chain, cfg)


@requires_sm120
@pytest.mark.parametrize("kind", ["add", "triple"])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 4096, 4096),
        (1, 14336, 4096),
        (16, 4096, 4096),
        (32, 14336, 4096),
        (64, 4096, 4096),
        (8, 11008, 4096),
        (96, 18432, 2048),
        (64, 64, 16384),
        (256, 256, 16384),
        (256, 4096, 4096),
        (512, 1024, 8192),
        (1024, 4096, 4096),
    ],
    ids=lambda s: "x".join(map(str, s)),
)
def test_sm120_dense_multi_gemm_auto_pick_sizes_the_tile_to_the_problem(monkeypatch, kind, shape):
    """At decode and skinny long-K sizes the automatic multi-GEMM pick carries no
    CTA rows the problem does not need -- at most max(32, M rounded up to a power
    of two) -- takes a 128-byte K tile while the GEMM streams its weights
    (M <= 256), holds >= 2 AB stages, and the full probe accepts it. The
    select_config pick it replaced was a 128-row tile on every shape here, up to
    4.9x slower than the best tile (1x4096x4096 dual SwiGLU: 125.9 vs 25.4 us;
    64x64x16384: 120.8 vs 66.5 us)."""
    from cudnn.gemm.frost.kernel_registry import select_template
    from cudnn.gemm.frost.sm120 import compiler as C

    monkeypatch.setattr(C, "_sm_count", lambda: 188)
    M = shape[0]
    chain = analyze(_sm120_mg_graph(kind, *shape)[0])
    cfg = C.plan_config(chain)
    assert cfg.cta_tile_m <= max(32, 1 << max(0, M - 1).bit_length()), cfg.name
    if M <= C._DENSE_MULTI_GEMM_STREAMING_M:
        assert cfg.cta_tile_k_bytes == 128, cfg.name
    assert cfg.split_k_slices == 1 and not cfg.swap_ab
    assert select_template(chain, cfg).multi_gemm_ab_stages(chain, cfg)[0] >= 2, cfg.name
    C.probe_chain(chain, cfg)


@requires_sm120
@pytest.mark.parametrize("kind", ["add", "triple"])
@pytest.mark.parametrize("shape", [(1, 1000, 512), (24, 520, 768), (96, 1000, 256), (64, 64, 2048)], ids=lambda s: "x".join(map(str, s)))
def test_sm120_dense_multi_gemm_small_m_auto_plan_runs_exactly(kind, shape):
    """The automatic plan at decode sizes (32- and 64-row tiles, a 16x16 warp tile
    on 2x4 warps, M tails down to one row) through the public graph API: exact
    on small-integer data, nothing written outside the output."""
    from cudnn.engines import is_python_engine

    M, N, K = shape
    g, t, pairs, y = _sm120_mg_graph(kind, M, N, K)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    frost = [i for i, plan in enumerate(g.plans) if is_python_engine(plan.engine_id)]
    assert frost, "no FROST plan listed"
    g.select_plan(frost[0])
    g.check_support()
    g.build_plans()
    torch.manual_seed(31)
    lo, hi = (-1, 2) if kind == "triple" else (-2, 3)  # the gated product stays exact in fp32
    data = {name: torch.randint(lo, hi, (1, M if name.startswith("A") else N, K), device="cuda").to(torch.bfloat16) for name in t}
    nbytes = M * N * 2
    raw = torch.full((nbytes + 2 * _MG_GUARD,), 0xA5, dtype=torch.uint8, device="cuda")
    out = raw[_MG_GUARD : _MG_GUARD + nbytes].view(torch.bfloat16).view(1, M, N)
    g.execute({**{t[n]: data[n] for n in t}, y: out}, torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda"))
    torch.cuda.synchronize()
    prods = [torch.einsum("bmk,bnk->bmn", data[an].double(), data[bn].double()) for an, bn in pairs]
    ref = prods[0] * prods[1] + prods[2] if kind == "triple" else prods[0] + prods[1]
    torch.testing.assert_close(out.double(), ref.to(torch.bfloat16).double(), atol=0, rtol=0)
    assert bool((raw[:_MG_GUARD] == 0xA5).all()) and bool((raw[-_MG_GUARD:] == 0xA5).all()), "a store ran past the output"


@requires_sm120
@pytest.mark.parametrize("kind", ["add", "distinct", "triple"])
def test_sm120_dense_multi_gemm_probe_and_funnel_agree(kind):
    """kernel_registry.accepts() / candidates() and probe_chain agree on every sm120
    catalog point of both orientations for a multi-GEMM chain -- the SMEM rule
    (Sm120KernelTemplate.multi_gemm_reject) declines the same tiles in both."""
    from dataclasses import replace

    from cudnn.gemm.frost import kernel_registry as R
    from cudnn.gemm.frost.fusion_ir import swap_ab
    from cudnn.gemm.frost.sm120 import compiler as C
    from cudnn.gemm.frost.tile_config import CATALOG

    chain = analyze(_sm120_mg_graph(kind, 512, 512, 512)[0])
    sm120 = [c for c in CATALOG if c.pipeline == "sm120"]
    declined_for_smem = 0
    disagree = []
    for cfg in sm120 + [replace(c, swap_ab=True) for c in sm120]:
        lowered = swap_ab(chain) if cfg.swap_ab else chain
        funnel = R.select_template(lowered, cfg).accepts(lowered, cfg)
        declined_for_smem += bool(funnel and "SMEM budget" in funnel)
        try:
            C.probe_chain(chain, cfg)
            probe = None
        except (NotImplementedError, ValueError) as e:
            probe = str(e)
        if (funnel is None) != (probe is None):
            disagree.append((cfg.name, cfg.swap_ab, funnel, probe))
    assert not disagree, disagree[:3]
    if kind != "add":  # every cataloged tile holds one A + two B tiles; four or more operand tiles miss on the large ones
        assert declined_for_smem > 0
    assert any(c.pipeline == "sm120" for _t, c in R.candidates(chain))


@requires_sm120
def test_sm120_dense_swiglu_public_graph_runs_the_frost_plan():
    """Through the public graph API (create_execution_plans -> check_support ->
    build_plans -> execute) a dense dual SwiGLU lists and runs a FROST plan on
    sm120 (it had none before the dense template took multi-GEMM)."""
    from cudnn.engines import is_python_engine

    M, N, K = 384, 1000, 512
    g = _graph()
    A = _A(g, M, K)
    B0, B1 = _B(g, K, N, "B0"), _B(g, K, N, "B1")
    sc = g.tensor(name="scale", dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    Y = g.mul(a=g.mul(a=g.swish(input=g.matmul(A=A, B=B0, name="m0"), name="s"), b=g.matmul(A=A, B=B1, name="m1"), name="m"), b=sc, name="d")
    Y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    frost = [i for i, plan in enumerate(g.plans) if is_python_engine(plan.engine_id)]
    assert frost, "no FROST plan listed for a dense dual SwiGLU"
    g.select_plan(frost[0])
    g.check_support()
    g.build_plans()
    torch.manual_seed(5)
    a, b0 = _rand(M, N, K, 0.3)
    _, b1 = _rand(M, N, K, 0.3)
    scale = torch.tensor([[[0.5]]], device="cuda", dtype=torch.float32)
    out = torch.full((1, M, N), float("nan"), device="cuda", dtype=torch.bfloat16)
    g.execute({A: a, B0: b0, B1: b1, sc: scale, Y: out}, torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda"))
    torch.cuda.synchronize()
    ref = torch.nn.functional.silu(_mm(a, b0)) * _mm(a, b1) * 0.5
    torch.testing.assert_close(out.float(), ref, atol=1e-1, rtol=1e-2)


# --------------------------------------------------------------------------- #
# sm120 regressions: multi-GEMM graphs the fused kernel cannot honour -- a GEMM's
# own result tapped beside the fused epilogue, a GEMM operand some node computes
# (a pointwise op on an operand, chained GEMMs), GEMM outputs that declare
# different dtypes -- are declined by check_support and the build alike, where
# the plan used to run and leave the graph's contract silently unmet. The
# no-epilogue multi-GEMM rounds each GEMM once, to its own output's dtype, and
# stores the layout each of its outputs declares; five or more operand rings
# get a ring depth whose real (1024-B aligned) SMEM layout fits the device.
# --------------------------------------------------------------------------- #

_BF16, _FP32 = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT


def _sm120_unservable_mg_graph(case):
    """(graph, the decline's message fragment) for a multi-GEMM graph the shared
    analyzer lowers only in part."""
    M = N = K = 256
    g = _graph()
    A = _A(g, M, K)
    B0, B1 = _B(g, K, N, "B0"), _B(g, K, N, "B1")
    y = None
    if case == "tap_one_gemm":
        m0, m1 = g.matmul(A=A, B=B0, name="m0"), g.matmul(A=A, B=B1, name="m1")
        m1.set_output(True).set_data_type(_BF16)
        y, frag = g.sub(a=m0, b=m1, name="y"), "GEMM 'm1'"
    elif case == "tap_both_gemms_swiglu":
        m0, m1 = g.matmul(A=A, B=B0, name="m0"), g.matmul(A=A, B=B1, name="m1")
        m0.set_output(True).set_data_type(_BF16)
        m1.set_output(True).set_data_type(_BF16)
        y, frag = g.mul(a=g.swish(input=m0, name="s"), b=m1, name="y"), "is not written by the fused kernel"
    elif case in ("relu_on_shared_a", "relu_on_shared_a_materialized"):
        ra = g.relu(input=A, name="ra")
        ra.set_data_type(_BF16)
        if case.endswith("materialized"):
            ra.set_output(True)
        y, frag = g.sub(a=g.matmul(A=ra, B=B0, name="m0"), b=g.matmul(A=ra, B=B1, name="m1"), name="y"), "is computed inside the graph"
    elif case == "relu_on_one_b":
        rb = g.relu(input=B1, name="rb")
        rb.set_data_type(_BF16).set_stride([K * N, 1, K])  # B0's K-major layout: one GEMM geometry
        y, frag = g.sub(a=g.matmul(A=A, B=B0, name="m0"), b=g.matmul(A=A, B=rb, name="m1"), name="y"), "is computed inside the graph"
    elif case in ("chained_gemms", "chained_gemms_materialized"):
        m0 = g.matmul(A=A, B=B0, name="m0")
        m0.set_data_type(_BF16)
        frag = "is computed inside the graph"
        if case.endswith("materialized"):
            m0.set_output(True)
            frag = "GEMM 'm0'"
        y = g.relu(input=g.matmul(A=m0, B=B1, name="m1"), name="y")
    elif case in ("dtypes_fp32_bf16", "dtypes_bf16_fp32"):
        m0, m1 = g.matmul(A=A, B=B0, name="m0"), g.matmul(A=A, B=B1, name="m1")
        m0.set_data_type(_FP32 if case == "dtypes_fp32_bf16" else _BF16)
        m1.set_data_type(_BF16 if case == "dtypes_fp32_bf16" else _FP32)
        y, frag = g.sub(a=m0, b=m1, name="y"), "declare different dtypes"
    elif case == "reduction_beside_no_epilogue":
        m0, m1 = g.matmul(A=A, B=B0, name="m0"), g.matmul(A=A, B=B1, name="m1")
        m0.set_output(True).set_data_type(_BF16)
        m1.set_output(True).set_data_type(_BF16)
        r = g.reduction(input=m0, mode=cudnn.reduction_mode.AMAX, name="r")
        r.set_dim([1, 1, 1]).set_stride([1, 1, 1]).set_output(True).set_data_type(_FP32)
        frag = "the result of reduction 'r'"
    else:
        raise ValueError(case)
    if y is not None:
        y.set_output(True).set_data_type(_FP32)
    return g, frag


@requires_sm120
@pytest.mark.parametrize(
    "case",
    [
        "tap_one_gemm",
        "tap_both_gemms_swiglu",
        "relu_on_shared_a",
        "relu_on_shared_a_materialized",
        "relu_on_one_b",
        "chained_gemms",
        "chained_gemms_materialized",
        "dtypes_fp32_bf16",
        "dtypes_bf16_fp32",
        "reduction_beside_no_epilogue",
    ],
)
def test_sm120_multi_gemm_graph_the_kernel_cannot_honour_is_declined(case):
    """The analyzer builds a multi-GEMM chain with no per-GEMM output beside a
    fused epilogue, no mainloop fusion and one dtype for every GEMM output, and
    does not decline a graph that asks for more: the plan ran, left a tapped GEMM
    output unwritten, read a buffer the graph was to compute, or rounded a GEMM
    with another one's dtype -- listed first, with nothing raised. Now
    check_support (probe_supported) and the build (jit_from_cudnn_graph) both
    decline, so the public plan list carries no FROST plan for the graph."""
    import re

    from cudnn.engines import is_python_engine
    from cudnn.gemm.frost.sm120 import compiler as C

    g, frag = _sm120_unservable_mg_graph(case)
    g.validate()  # derived tensors (relu(A), a chained GEMM's A) get their extents here
    g.build_operation_graph()
    with pytest.raises(NotImplementedError, match=re.escape(frag)):
        C.probe_supported(g)
    with pytest.raises(NotImplementedError, match=re.escape(frag)):
        C.jit_from_cudnn_graph(g, config=_N128_CFG)
    g, _ = _sm120_unservable_mg_graph(case)
    g.validate()
    g.build_operation_graph()
    try:
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    except cudnn.cudnnGraphNotSupportedError:
        return  # the backend declines this graph too: no engine proposes a plan
    assert not [i for i, plan in enumerate(g.plans) if is_python_engine(plan.engine_id)], "a FROST plan is listed for a graph it cannot honour"


_SM120_NOEPI_TORCH = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32, "fp8_e4m3": torch.float8_e4m3fn}
_SM120_NOEPI_CUDNN = {"bf16": _BF16, "fp16": cudnn.data_type.HALF, "fp32": _FP32, "fp8_e4m3": cudnn.data_type.FP8_E4M3}


def _sm120_noepi_graph(M, N, K, outs):
    """A no-epilogue dual GEMM (A shared): each GEMM's own result is an output,
    declared as ``outs[g] = (dtype, major, ld)``."""
    g = _graph()
    A = _A(g, M, K)
    Bs = [_B(g, K, N, f"B{i}") for i in range(len(outs))]
    ys = []
    for i, (dt, major, ld) in enumerate(outs):
        y = g.matmul(A=A, B=Bs[i], name=f"m{i}")
        y.set_output(True).set_data_type(_SM120_NOEPI_CUDNN[dt])
        if major == "m":
            y.set_dim([1, M, N]).set_stride([N * (ld or M), 1, ld or M])
        elif ld is not None:
            y.set_dim([1, M, N]).set_stride([M * ld, ld, 1])
        ys.append(y)
    return g, A, Bs, ys


@requires_sm120
@pytest.mark.parametrize("dts", [("bf16", "fp32"), ("fp32", "bf16"), ("bf16", "fp8_e4m3"), ("fp8_e4m3", "fp16")], ids="-".join)
def test_sm120_no_epilogue_multi_gemm_rounds_each_gemm_to_its_own_dtype(dts):
    """A no-epilogue multi-GEMM's outputs are its GEMMs' own results, each in its
    own declared dtype: each is rounded ONCE, to that dtype. The epilogue used to
    round every GEMM to GEMM 0's dtype first -- an fp32 output beside a bf16 one
    came back bf16-rounded, an fp16 one beside an fp8 one fp8-rounded. Data are
    multiples of 1/64, so the fp32 accumulation is exact and the store's rounding
    is the only one."""
    from cudnn.gemm.frost.sm120 import compiler as C

    M, N, K = 256, 192, 64
    g, A, Bs, ys = _sm120_noepi_graph(M, N, K, [(dt, "n", None) for dt in dts])
    compiled = C.jit_from_cudnn_graph(g, config=_N128_CFG)
    torch.manual_seed(11)
    a = (torch.randint(-64, 65, (1, M, K), device="cuda").float() / 64).to(torch.bfloat16)
    bs = [(torch.randint(-64, 65, (1, N, K), device="cuda").float() / 64).to(torch.bfloat16) for _ in dts]
    outs = [torch.full((1, M, N), float("nan"), device="cuda").to(_SM120_NOEPI_TORCH[dt]) for dt in dts]
    compiled({A: a, **dict(zip(Bs, bs)), **dict(zip(ys, outs))})
    torch.cuda.synchronize()
    for i, dt in enumerate(dts):
        exact = torch.einsum("bmk,bnk->bmn", a.double(), bs[i].double())
        torch.testing.assert_close(outs[i].double(), exact.float().to(_SM120_NOEPI_TORCH[dt]).double(), atol=0, rtol=0, msg=lambda m, i=i: f"GEMM {i}: {m}")


@requires_sm120
@pytest.mark.parametrize(
    "outs,swap",
    [
        ((("bf16", "m", None), ("fp32", "n", None)), False),
        ((("bf16", "n", 200), ("fp32", "m", 264)), False),
        ((("fp16", "m", None), ("bf16", "n", None)), True),
    ],
    ids=["m_major_beside_n_major", "padded_n_major_beside_padded_m_major", "swap_ab"],
)
def test_sm120_no_epilogue_multi_gemm_stores_each_declared_layout(outs, swap):
    """The analyzer models a no-epilogue multi-GEMM's per-GEMM outputs as compact
    N-major whatever the graph declares, so an M-major (or padded) one built a
    kernel whose call refused that very buffer. The sm120 entry points record
    each output's declared layout: the kernel stores it -- exact, nothing written
    outside it (canaries before, after and in every row's padding)."""
    from dataclasses import replace

    from cudnn.gemm.frost.sm120 import compiler as C

    M, N, K = 200, 136, 192
    g, A, Bs, ys = _sm120_noepi_graph(M, N, K, list(outs))
    compiled = C.jit_from_cudnn_graph(g, config=replace(_N128_CFG, swap_ab=swap))
    torch.manual_seed(13)
    a = torch.randint(-2, 3, (1, M, K), device="cuda").to(torch.bfloat16)
    bs = [torch.randint(-2, 3, (1, N, K), device="cuda").to(torch.bfloat16) for _ in outs]
    bufs = []
    for dt, major, ld in outs:
        inner, outer = (N, M) if major == "n" else (M, N)
        pitch = ld or inner
        nbytes = outer * pitch * torch.empty((), dtype=_SM120_NOEPI_TORCH[dt]).element_size()
        raw = torch.full((nbytes + 2 * _MG_GUARD,), 0xA5, dtype=torch.uint8, device="cuda")
        body = raw[_MG_GUARD : _MG_GUARD + nbytes].view(_SM120_NOEPI_TORCH[dt]).view(1, outer, pitch)
        view = body[:, :, :inner] if major == "n" else body[:, :, :inner].transpose(1, 2)
        bufs.append((raw, body, view, inner))
    compiled({A: a, **dict(zip(Bs, bs)), **{y: buf[2] for y, buf in zip(ys, bufs)}})
    torch.cuda.synchronize()
    for i, ((dt, _major, _ld), (raw, body, view, inner)) in enumerate(zip(outs, bufs)):
        ref = torch.einsum("bmk,bnk->bmn", a.double(), bs[i].double()).to(_SM120_NOEPI_TORCH[dt])
        torch.testing.assert_close(view.double(), ref.double(), atol=0, rtol=0, msg=lambda m, i=i: f"GEMM {i}: {m}")
        assert bool((raw[:_MG_GUARD] == 0xA5).all()) and bool((raw[-_MG_GUARD:] == 0xA5).all()), f"GEMM {i}: a store ran past the output"
        if body.shape[-1] > inner:
            assert bool((body[:, :, inner:].contiguous().view(torch.uint8) == 0xA5).all()), f"GEMM {i}: a store wrote into the padding"


@requires_sm120
def test_sm120_no_epilogue_m_major_outputs_public_graph_runs_the_frost_plan():
    """Through the public graph API a no-epilogue dual GEMM with an M-major
    per-GEMM output lists and runs a FROST plan that writes both outputs in the
    declared layouts (check_support and the build read them the same way)."""
    from cudnn.engines import is_python_engine

    M, N, K = 384, 256, 256
    g, A, Bs, ys = _sm120_noepi_graph(M, N, K, [("bf16", "m", None), ("bf16", "n", None)])
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    frost = [i for i, plan in enumerate(g.plans) if is_python_engine(plan.engine_id)]
    assert frost, "no FROST plan listed for a no-epilogue dual GEMM with an M-major output"
    g.select_plan(frost[0])
    g.check_support()
    g.build_plans()
    torch.manual_seed(17)
    a = torch.randint(-2, 3, (1, M, K), device="cuda").to(torch.bfloat16)
    bs = [torch.randint(-2, 3, (1, N, K), device="cuda").to(torch.bfloat16) for _ in Bs]
    y0 = torch.full((1, N, M), float("nan"), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    y1 = torch.full((1, M, N), float("nan"), device="cuda", dtype=torch.bfloat16)
    g.execute({A: a, Bs[0]: bs[0], Bs[1]: bs[1], ys[0]: y0, ys[1]: y1}, torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda"))
    torch.cuda.synchronize()
    for y, b in ((y0, bs[0]), (y1, bs[1])):
        torch.testing.assert_close(y, torch.einsum("bmk,bnk->bmn", a.double(), b.double()).to(torch.bfloat16), atol=0, rtol=0)


@requires_sm120
@pytest.mark.parametrize(
    "num_gemms,config_name",
    [(5, "CONFIG_sm120_128x144x32_16x16x32_cluster1x1_warps8x1"), (8, "CONFIG_sm120_128x48x32_16x16x32_cluster1x1_warps8x1")],
    ids=["1A5B_128x144x32", "1A8B_128x48x32"],
)
@pytest.mark.parametrize("io", ["bf16", "fp8"])
def test_sm120_many_operand_multi_gemm_fits_the_smem_it_allocates(num_gemms, config_name, io):
    """Five or more operand rings: every ring is 1024-B aligned, and the linear
    per-stage count that sizes the ring never saw the pad in front of each -- it
    admitted 3 / 5 stages here, a 101888 / 103936-B layout the device refused at
    launch (opt-in 101376 B). The ring depth now comes off the template's real
    layout: the funnel and the probe admit the plan, the layout fits, and the
    kernel launches and computes exactly."""
    from cudnn.gemm.frost import kernel_registry as R
    from cudnn.gemm.frost.sm120 import compiler as C
    from cudnn.gemm.frost.tile_config import _sm_smem_budget_bytes

    M = N = K = 512
    dt = {"bf16": _BF16, "fp8": cudnn.data_type.FP8_E4M3}[io]
    g = _graph(dt)
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1], data_type=dt)
    Bs = [g.tensor(name=f"B{i}", dim=[1, K, N], stride=[K * N, 1, K], data_type=dt) for i in range(num_gemms)]
    y = g.matmul(A=A, B=Bs[0], name="m0")
    for i in range(1, num_gemms):
        y = g.add(a=y, b=g.matmul(A=A, B=Bs[i], name=f"m{i}"), name=f"s{i}")
    y.set_output(True).set_data_type(_FP32)
    chain = analyze(g)
    cfg = by_name(config_name)
    tmpl = R.select_template(chain, cfg)
    assert tmpl.accepts(chain, cfg) is None
    C.probe_chain(chain, cfg)
    stages = tmpl.multi_gemm_ab_stages(chain, cfg)[0]
    assert stages >= 1 and tmpl.multi_gemm_smem_bytes(chain, cfg, stages) <= _sm_smem_budget_bytes()
    compiled = C.jit_from_cudnn_graph(g, config=cfg)
    torch.manual_seed(19)
    tdt = {"bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}[io]
    a = torch.randint(-2, 3, (1, M, K), device="cuda").to(tdt)
    bs = [torch.randint(-2, 3, (1, N, K), device="cuda").to(tdt) for _ in Bs]
    out = torch.full((1, M, N), float("nan"), device="cuda")
    compiled({A: a, **dict(zip(Bs, bs)), y: out})
    torch.cuda.synchronize()
    ref = sum(torch.einsum("bmk,bnk->bmn", a.double(), b.double()) for b in bs)
    torch.testing.assert_close(out.double(), ref, atol=0, rtol=0)


# --------------------------------------------------------------------------- #
# sm120 regressions: a multi-GEMM chain with a block-32 quantize whose every
# drainable geometry exceeds the quant pick's accumulator budget (three GEMMs
# with a 32-row col block: 32x32 is the smallest warp tile that drains it; five
# or more with a row block) gets an automatic plan -- the budget is a speed
# preference, never a gate. The pick used to keep a config the drain declines,
# so check_support and the public plan list declined chains the registry funnel
# lists (and the parent tree serves).
# --------------------------------------------------------------------------- #


def _sm120_mg_quant_graph(num_b, quants, M, N, K, gated=False):
    """A @ B_i for i < num_b, summed (or ``gated``: A@B0 * A@B1 + A@B2), then one
    MXFP8 quantize per ``quants`` axis (block 32, e4m3 data, compact e8m0 scales)."""
    g = _graph()
    A = _A(g, M, K)
    Bs = [_B(g, K, N, f"B{i}") for i in range(num_b)]
    ms = [g.matmul(A=A, B=b, name=f"m{i}") for i, b in enumerate(Bs)]
    if gated:
        y = g.add(a=g.mul(a=ms[0], b=ms[1], name="p"), b=ms[2], name="y")
    else:
        y = ms[0]
        for i in range(1, num_b):
            y = g.add(a=y, b=ms[i], name=f"s{i}")
    outs = []
    for qi, axis in enumerate(quants):
        Q, Sf = g.block_scale_quantize(input=y, block_size=32, axis=axis, name=f"q{qi}")
        Q.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
        dims = [1, M // 32, N] if axis == 1 else [1, M, N // 32]
        Sf.set_dim(dims).set_stride([dims[1] * dims[2], dims[2], 1]).set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)
        outs.append((axis, Q, Sf))
    return g, A, Bs, outs


def _mxfp8_ref(x, axis):
    """(e4m3 data, e8m0 scales) of an fp32 [1, M, N] tensor quantized in 32-element blocks along ``axis``."""
    _, M, N = x.shape
    blk = x.view(1, M, N // 32, 32) if axis == -1 else x.view(1, M // 32, 32, N)
    amax = blk.abs().amax(-1 if axis == -1 else 2)
    sf = amax / 448.0
    sf = torch.where(sf > 0, torch.pow(2.0, torch.ceil(torch.log2(torch.where(sf > 0, sf, torch.ones_like(sf))))), torch.zeros_like(sf))
    scale = sf.to(torch.float8_e8m0fnu)
    inv = torch.where(scale.float() > 0, 1.0 / scale.float(), torch.zeros_like(sf))
    inv = inv.unsqueeze(-1) if axis == -1 else inv.unsqueeze(2)
    return (blk * inv).clamp(-448, 448).to(torch.float8_e4m3fn).view(1, M, N), scale


_SM120_MGQ_CASES = {
    # id -> (num B operands, quant axes, gated)
    "triple_gated_dual32": (3, (-1, 1), True),
    "triple_col32": (3, (1,), False),
    "1A5B_row32": (5, (-1,), False),
    "1A5B_col32": (5, (1,), False),
    "1A6B_dual32": (6, (-1, 1), False),
}


@requires_sm120
@pytest.mark.parametrize("case", list(_SM120_MGQ_CASES))
@pytest.mark.parametrize("shape", [(512, 4096, 1024), (4096, 512, 1024)], ids=lambda s: "x".join(map(str, s)))
def test_sm120_multi_gemm_quant_auto_pick_past_the_register_budget(case, shape):
    """plan_config hands the probe a geometry the row-per-lane drain walks, though
    none fits the 64-register accumulator budget: check_support accepts the graph
    and the public plan list carries a FROST plan (it used to keep select_config's
    16-column warp tile, which the drain declines -- and no engine proposed a plan)."""
    from cudnn.engines import is_python_engine
    from cudnn.gemm.frost.sm120 import compiler as C

    num_b, quants, gated = _SM120_MGQ_CASES[case]
    g, *_ = _sm120_mg_quant_graph(num_b, quants, *shape, gated=gated)
    chain = analyze(g)
    cfg = C.plan_config(chain)
    C.probe_chain(chain, cfg)
    assert chain.num_gemms * cfg.warp_tile_m * cfg.warp_tile_n > 32 * C._ROW_PER_LANE_QUANT_ACC_REGS, cfg.name
    C.probe_supported(g)
    g, *_ = _sm120_mg_quant_graph(num_b, quants, *shape, gated=gated)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    assert [i for i, plan in enumerate(g.plans) if is_python_engine(plan.engine_id)], "no FROST plan listed"


@requires_sm120
@pytest.mark.parametrize("case", ["triple_gated_dual32", "1A5B_row32", "1A5B_col32"])
def test_sm120_multi_gemm_quant_over_budget_plan_runs_exactly(case):
    """The over-budget automatic plan, through the public graph API, writes every
    scale and data element exactly (small-integer data: the fp32 accumulation is
    exact, so the e8m0 scale and the e4m3 rounding are the only roundings)."""
    from cudnn.engines import is_python_engine

    num_b, quants, gated = _SM120_MGQ_CASES[case]
    M, N, K = 256, 512, 256
    g, A, Bs, outs = _sm120_mg_quant_graph(num_b, quants, M, N, K, gated=gated)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    frost = [i for i, plan in enumerate(g.plans) if is_python_engine(plan.engine_id)]
    assert frost, "no FROST plan listed"
    g.select_plan(frost[0])
    g.check_support()
    g.build_plans()
    torch.manual_seed(29)
    a = torch.randint(-2, 3, (1, M, K), device="cuda").to(torch.bfloat16)
    bs = [torch.randint(-2, 3, (1, N, K), device="cuda").to(torch.bfloat16) for _ in Bs]
    pack = {A: a, **dict(zip(Bs, bs))}
    bufs = []
    for axis, Q, Sf in outs:
        q = torch.zeros(1, M, N, device="cuda", dtype=torch.float8_e4m3fn)
        s = torch.full(tuple(Sf.get_dim()), 0x7F, device="cuda", dtype=torch.uint8).view(torch.float8_e8m0fnu)
        pack[Q], pack[Sf] = q, s
        bufs.append((axis, q, s))
    g.execute(pack, torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda"))
    torch.cuda.synchronize()
    p = [torch.einsum("bmk,bnk->bmn", a.double(), b.double()) for b in bs]
    x = (p[0] * p[1] + p[2] if gated else sum(p)).float()
    for axis, q, s in bufs:
        qref, sref = _mxfp8_ref(x, axis)
        assert torch.equal(s.view(torch.uint8), sref.view(torch.uint8)), f"axis {axis}: scales differ"
        assert torch.equal(q.float(), qref.float()), f"axis {axis}: data differ"


# --------------------------------------------------------------------------- #
# sm120 regressions: an N-major B multi-GEMM whose parallel GEMMs cap the N tile
# below a 128-byte swizzle group (e4m3 with 3+ GEMMs, bf16 with 5+) gets an
# automatic plan. select_config declined it with the tcgen05 group rule, so
# check_support and the public plan list had no FROST plan although the sm120
# MN-major group is the CTA K tile's width and the registry funnel listed 110-138.
# --------------------------------------------------------------------------- #


def _sm120_nmajor_mg_graph(io, num_b, M, N, K, a_major="k"):
    dt = {"bf16": _BF16, "fp8": cudnn.data_type.FP8_E4M3}[io]
    g = _graph(dt)
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1] if a_major == "k" else [M * K, 1, M], data_type=dt)
    Bs = [g.tensor(name=f"B{i}", dim=[1, K, N], stride=[K * N, N, 1], data_type=dt) for i in range(num_b)]
    y = g.matmul(A=A, B=Bs[0], name="m0")
    for i in range(1, num_b):
        y = g.add(a=y, b=g.matmul(A=A, B=Bs[i], name=f"m{i}"), name=f"s{i}")
    y.set_output(True).set_data_type(_FP32)
    return g, A, Bs, y


@requires_sm120
@pytest.mark.parametrize(
    "io,num_b,a_major",
    [("fp8", 3, "k"), ("fp8", 3, "m"), ("fp8", 4, "k"), ("bf16", 5, "k"), ("bf16", 8, "k")],
    ids=["fp8_1A3B", "fp8_1A3B_m_major_a", "fp8_1A4B", "bf16_1A5B", "bf16_1A8B"],
)
def test_sm120_n_major_b_multi_gemm_gets_an_automatic_plan(io, num_b, a_major):
    """The automatic config is probe-accepted, holds >= 2 AB stages, and its N-major
    B tile is a whole number of K-tile-wide swizzle groups; through the public
    graph API the FROST plan is listed and runs exactly (small-integer data)."""
    from cudnn.engines import is_python_engine
    from cudnn.gemm.frost.kernel_registry import select_template
    from cudnn.gemm.frost.sm120 import compiler as C

    M, N, K = 256, 144, 256
    chain = analyze(_sm120_nmajor_mg_graph(io, num_b, M, N, K, a_major)[0])
    cfg = C.plan_config(chain)
    C.probe_chain(chain, cfg)
    elem = 1 if io == "fp8" else 2
    assert cfg.cta_tile_n % (cfg.cta_tile_k_bytes // elem) == 0, cfg.name
    assert select_template(chain, cfg).multi_gemm_ab_stages(chain, cfg)[0] >= 2, cfg.name
    g, A, Bs, y = _sm120_nmajor_mg_graph(io, num_b, M, N, K, a_major)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    frost = [i for i, plan in enumerate(g.plans) if is_python_engine(plan.engine_id)]
    assert frost, "no FROST plan listed"
    g.select_plan(frost[0])
    g.check_support()
    g.build_plans()
    torch.manual_seed(31)
    tdt = {"bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}[io]
    a = torch.randint(-2, 3, (1, M, K), device="cuda").to(tdt)
    a_rt = a.transpose(1, 2).contiguous().transpose(1, 2) if a_major == "m" else a
    bs = [torch.randint(-2, 3, (1, K, N), device="cuda").to(tdt) for _ in Bs]
    out = torch.full((1, M, N), float("nan"), device="cuda")
    g.execute({A: a_rt, **dict(zip(Bs, bs)), y: out}, torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda"))
    torch.cuda.synchronize()
    ref = sum(torch.einsum("bmk,bkn->bmn", a.double(), b.double()) for b in bs)
    torch.testing.assert_close(out.double(), ref, atol=0, rtol=0)
