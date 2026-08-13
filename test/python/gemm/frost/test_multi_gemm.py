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
    requires_sm100,
    Plan as _plan,
    vp_mg as _vp_mg,
    reduction_ref as _reduction_ref,
)

from cudnn.gemm.frost.epilogue_codegen import generate
from cudnn.gemm.frost.graph_analyzer import analyze
from cudnn.gemm.frost.tile_config import CATALOG, DEFAULT_CONFIG, by_name

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


_N128_CFG = next(c for c in CATALOG if c.cta_tile_m == 128 and c.cta_tile_n == 128 and c.cta_tile_k_bytes == 128 and c.cgrp_size_m == 1 and c.cgrp_size_n == 1)


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


_N256_C2_CFG = next(
    c for c in CATALOG if c.cta_tile_m == 128 and c.cta_tile_n == 256 and c.cta_tile_k_bytes == 128 and c.cgrp_size_m == 2 and c.cgrp_size_n == 1
)


@requires_sm100
def test_multi_gemm_2ctamma_compiles() -> None:
    """The 2-CTA-MMA CLC template (cluster2x1) compiles a dual-GEMM graph."""
    M, N, K = 256, 256, 128
    g = _graph()
    A = _A(g, M, K)
    B0, B1 = _B(g, K, N, "B0"), _B(g, K, N, "B1")
    C0 = g.matmul(A=A, B=B0, name="m0")
    C1 = g.matmul(A=A, B=B1, name="m1")
    Y = g.add(a=C0, b=C1, name="a")
    Y.set_output(True)
    compiled = _plan(g, _N256_C2_CFG, cta_group=2)
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
    mainloop_tmpl = next(t for t in TEMPLATES if t.mainloop)
    assert not mainloop_tmpl.supports_multi_gemm
    assert mainloop_tmpl.accepts(chain, _N256_C2_CFG) is not None


# --- End-to-end correctness (GPU) ---

_GPU = requires_sm100


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


# A split CTA tile competes with multi-GEMM for the SAME 512 TMEM columns: one
# acc stage holds num_gemms x num_mma_m x cols_per_mma_m.
_SPLIT_CFG = by_name("CONFIG_sm100_256x128x128_128x128x32_cluster1x1")


@_GPU
def test_multi_gemm_times_split_tile_over_tmem_budget_is_rejected() -> None:
    M, N, K = 256, 256, 128
    g = _graph()
    A = _A(g, M, K)
    B0, B1 = _B(g, K, N, "B0"), _B(g, K, N, "B1")
    Y = g.mul(a=g.swish(input=g.matmul(A=A, B=B0, name="mm0")), b=g.matmul(A=A, B=B1, name="mm1"))
    Y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    with pytest.raises(NotImplementedError, match="TMEM columns"):
        # 2 GEMMs x 2 M blocks x 256 cols = 1024
        _plan(g, by_name("CONFIG_sm100_256x256x128_128x256x32_cluster1x1"), cta_group=1)


@_GPU
@pytest.mark.parametrize("config", [DEFAULT_CONFIG, _SPLIT_CFG], ids=["single_mma", "num_mma_m2"])
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
    sm = torch.cuda.get_device_capability()
    if sm[0] * 10 + sm[1] not in (100, 110):
        pytest.skip("int8 multi-GEMM requires SM100 or SM110")
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
    cfg = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster4x2")
    g = _build_dual_add_reduction(B, M, N, K, mode, [1, 1, 1])
    compiled = _plan(g, cfg, cta_group=2)

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
    """Regression: a plain single-GEMM relu through the modified 1ctamma path."""
    M, N, K = 256, 256, 128
    g = _graph()
    A, B = _A(g, M, K), _B(g, K, N)
    C = g.matmul(A=A, B=B, name="mm")
    Y = g.relu(input=C, name="r")
    Y.set_output(True)
    compiled = _plan(g, DEFAULT_CONFIG, cta_group=1)

    torch.manual_seed(3)
    a, b = _rand(M, N, K)
    c = torch.zeros(1, M, N, device="cuda", dtype=torch.bfloat16)
    compiled({A: a, B: b, Y: c})
    torch.cuda.synchronize()
    torch.testing.assert_close(c, torch.relu(_mm(a, b)).bfloat16(), atol=1e-1, rtol=1e-2)
