# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for mainloop fusion: A' = op(A), C = A' @ B.

Covers the analyzer (upstream ops land in mainloop_a/b_ops, kept out of the
epilogue), codegen (generate_mainloop), and end-to-end JIT+run vs torch.
"""

from __future__ import annotations

import cudnn
import cudnn.gemm.frost  # noqa: F401  (installs hook)
import pytest
import torch

from gemm_test_utils import (
    requires_int8_mma,
    requires_sm100,
    Plan as _plan,
    vp as _vp,
    kw as _kw,
)


from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
from cudnn.gemm.frost.epilogue_codegen import generate_mainloop
from cudnn.gemm.frost.graph_analyzer import analyze

pytestmark = pytest.mark.L0


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def _mainloop_graph(
    op: str,
    M: int,
    N: int,
    K: int,
    io_dtype=cudnn.data_type.BFLOAT16,
    out_major: str = "n",
):
    return _mainloop_graph_ab(op, "none", M, N, K, io_dtype, out_major)


def _mainloop_graph_ab(
    aop: str,
    bop: str,
    M: int,
    N: int,
    K: int,
    io_dtype=cudnn.data_type.BFLOAT16,
    out_major: str = "n",
    a_major: str = "k",
    b_major: str = "k",
):
    """Graph with an optional unary op on A and/or B ("none" = no op)."""
    g = cudnn.pygraph(
        io_data_type=io_dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1] if a_major == "k" else [M * K, 1, M])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K] if b_major == "k" else [K * N, N, 1])
    Ai = getattr(g, aop)(input=A, name="aop").set_data_type(io_dtype) if aop != "none" else A
    Bi = getattr(g, bop)(input=B, name="bop").set_data_type(io_dtype) if bop != "none" else B
    C = g.matmul(A=Ai, B=Bi, name="mm")
    if out_major == "m":
        C.set_stride([M * N, 1, M])  # column-major (M-contiguous)
    C.set_output(True)
    return g


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


def test_analyzer_detects_mainloop_op() -> None:
    chain = analyze(_mainloop_graph("abs", 256, 256, 128))
    assert chain.has_mainloop_fusion
    assert [op.op for op in chain.mainloop_a_ops] == ["abs"]
    # The abs op must NOT leak into the epilogue chain.
    assert chain.ops == []
    assert chain.matmul.a_dtype == "bf16"
    assert (chain.matmul.M, chain.matmul.N, chain.matmul.K) == (256, 256, 128)


def test_analyzer_no_mainloop_for_epilogue_op() -> None:
    # relu on the matmul OUTPUT is an epilogue op, not mainloop.
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, 256, 128], stride=[256 * 128, 128, 1])
    B = g.tensor(name="B", dim=[1, 128, 256], stride=[128 * 256, 1, 128])
    C = g.matmul(A=A, B=B, name="mm")
    Y = g.relu(input=C, name="r")
    Y.set_output(True)
    chain = analyze(g)
    assert not chain.has_mainloop_fusion
    assert [op.op for op in chain.ops] == ["relu"]


def test_analyzer_detects_b_and_both() -> None:
    # B-only.
    cb = analyze(_mainloop_graph_ab("none", "abs", 256, 256, 128))
    assert cb.has_mainloop_fusion_b and not cb.has_mainloop_fusion_a
    assert [op.op for op in cb.mainloop_b_ops] == ["abs"]
    # Both A and B.
    cab = analyze(_mainloop_graph_ab("abs", "relu", 256, 256, 128))
    assert cab.has_mainloop_fusion_a and cab.has_mainloop_fusion_b
    assert [op.op for op in cab.mainloop_a_ops] == ["abs"]
    assert [op.op for op in cab.mainloop_b_ops] == ["relu"]
    assert cab.ops == []


def test_zero_preserving_registry() -> None:
    """The K-OOB-mask decision rests on ZERO_PRESERVING_OPS: an op is listed
    iff f(0)=0 for any other operand (unary f(0)=0, plus mul). Everything else
    — including div (aux/0=inf) — must be non-preserving so the mask fires."""
    from cudnn.gemm.frost.fusion_ir import (
        ZERO_PRESERVING_OPS,
        UNARY_OPS,
        BINARY_OPS,
        FusionOp,
    )
    from cudnn.gemm.frost.compiler import _mainloop_chain_zero_preserving as zp

    # Pin membership both ways so a newly-added op can't silently slip in/out.
    assert {"sigmoid", "exp", "cos", "log", "reciprocal", "rsqrt", "softplus", "logical_not", "leaky_relu", "gen_index"} == {
        o for o in UNARY_OPS if o not in ZERO_PRESERVING_OPS
    }
    assert {
        "add",
        "sub",
        "div",
        "max",
        "min",
        "pow",
        "add_square",
        "mod",
        "logical_and",
        "logical_or",
        "cmp_eq",
        "cmp_neq",
        "cmp_gt",
        "cmp_ge",
        "cmp_lt",
        "cmp_le",
        "relu_backward",
        "leaky_relu_backward",
        "swish_backward",
        "sigmoid_backward",
        "tanh_backward",
        "elu_backward",
        "gelu_backward",
        "gelu_tanh_backward",
        "softplus_backward",
    } == {o for o in BINARY_OPS if o not in ZERO_PRESERVING_OPS}
    assert "mul" in ZERO_PRESERVING_OPS

    assert zp([]) is True  # no fusion -> safe
    assert zp([FusionOp(op="relu")]) is True
    assert zp([FusionOp(op="mul", aux="s")]) is True
    assert zp([FusionOp(op="cos")]) is False
    assert zp([FusionOp(op="abs"), FusionOp(op="cos")]) is False  # net f(0)!=0
    assert zp([FusionOp(op="div", aux="s", aux_on_rhs=True)]) is False  # aux/0=inf


def test_codegen_mainloop_b_snippet() -> None:
    chain = analyze(_mainloop_graph_ab("none", "relu", 256, 256, 128))
    snip = generate_mainloop(chain, "b")
    assert "ml_f32_b = ml_vec_b.to(cutlass.Float32)" in snip
    assert "ml_out_b = " in snip
    # A snippet is empty when only B is fused.
    assert generate_mainloop(chain, "a") == "pass"


def test_analyzer_chain_of_two_unary_a_ops() -> None:
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, 256, 128], stride=[256 * 128, 128, 1])
    B = g.tensor(name="B", dim=[1, 128, 256], stride=[128 * 256, 1, 128])
    Aa = g.abs(input=A, name="absA")
    Ar = g.relu(input=Aa, name="reluA")  # abs then relu, both on A
    C = g.matmul(A=Ar, B=B, name="mm")
    C.set_output(True)
    chain = analyze(g)
    assert [op.op for op in chain.mainloop_a_ops] == ["abs", "relu"]


# ---------------------------------------------------------------------------
# Codegen
# ---------------------------------------------------------------------------


def test_codegen_mainloop_snippet_abs() -> None:
    chain = analyze(_mainloop_graph("abs", 256, 256, 128))
    snip = generate_mainloop(chain, "a")
    assert "ml_f32_a = ml_vec_a.to(cutlass.Float32)" in snip
    assert "cute.math.abs(ml_f32_a)" in snip
    assert "ml_out_a = " in snip and ".to(cutlass.BFloat16)" in snip


def test_codegen_mainloop_empty_when_no_fusion() -> None:
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, 256, 128], stride=[256 * 128, 128, 1])
    B = g.tensor(name="B", dim=[1, 128, 256], stride=[128 * 256, 1, 128])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True)
    assert generate_mainloop(analyze(g), "a") == "pass"
    assert generate_mainloop(analyze(g), "b") == "pass"


# ---------------------------------------------------------------------------
# End-to-end (GPU)
# ---------------------------------------------------------------------------

_TORCH_OP = {"abs": torch.abs, "relu": torch.relu, "neg": torch.neg, "cos": torch.cos}


def _run_e2e(op, cfg_name, M, N, K, io_dtype, torch_dtype, out_major="n"):
    g = _mainloop_graph(op, M, N, K, io_dtype=io_dtype, out_major=out_major)
    plan = _plan(g, **_kw(cfg_name))
    assert analyze(g).has_mainloop_fusion
    torch.manual_seed(0)
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-3, 3).to(dtype=torch_dtype, device="cuda")
    b = torch.empty(1, N, K, dtype=torch.int32).random_(-3, 3).to(dtype=torch_dtype, device="cuda")
    if out_major == "m":
        c = torch.empty(1, N, M, dtype=torch_dtype, device="cuda").transpose(1, 2)
    else:
        c = torch.empty(1, M, N, dtype=torch_dtype, device="cuda")
    plan(_vp(plan, a, b, c))
    torch.cuda.synchronize()
    ref = torch.einsum("bmk,bnk->bmn", _TORCH_OP[op](a.float()), b.float()).to(torch_dtype)
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("op", ["abs", "relu", "neg"])
@pytest.mark.parametrize(
    "cfg",
    [
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",  # CTA_1, TMA-store epi
        "CONFIG_sm100_64x128x128_64x128x32_cluster1x1_1ctamma",  # CTA_1, M=64 STG epi
        "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",  # CTA_2, cluster MMA m=256
        "CONFIG_sm100_64x256x128_64x256x32_cluster2x1_2ctamma",  # CTA_2, cluster MMA m=128
        "CONFIG_sm100_128x40x128_128x40x32_cluster1x1_1ctamma",  # N%32!=0: B-transform tail round
        "CONFIG_sm100_128x48x128_128x48x32_cluster2x1_2ctamma",  # CTA_2 N%16, B tile smaller than one round
        # CTA tile split across two MMA instructions along M: the 12-warp mainloop
        # transforms the whole tile in SMEM, the MMA warp then walks the M blocks.
        "CONFIG_sm100_256x128x128_128x128x32_cluster1x1_1ctamma",  # num_mma_m=2
        "CONFIG_sm100_128x128x128_64x128x32_cluster2x1_2ctamma",  # num_mma_m=2, 2x2 DP drain
    ],
)
@requires_sm100
def test_e2e_mainloop_bf16(op, cfg) -> None:
    _run_e2e(op, cfg, 512, 512, 256, cudnn.data_type.BFLOAT16, torch.bfloat16)


@pytest.mark.parametrize(
    "op,cfg",
    [
        ("relu", "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"),
        ("relu", "CONFIG_sm100_64x128x128_64x128x32_cluster1x1_1ctamma"),
    ],
    ids=["relu-cta128", "relu-cta64"],
)
@requires_sm100
def test_m_major_mainloop(op, cfg) -> None:
    _run_e2e(op, cfg, 512, 512, 256, cudnn.data_type.BFLOAT16, torch.bfloat16, out_major="m")


@pytest.mark.parametrize(
    "cfg",
    [
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
        "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
    ],
)
@requires_sm100
def test_e2e_mainloop_fp16(cfg) -> None:
    _run_e2e("abs", cfg, 512, 512, 256, cudnn.data_type.HALF, torch.float16)


@pytest.mark.parametrize("op", ["abs", "relu", "neg"])
@pytest.mark.parametrize(
    "cfg",
    [
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
        "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
    ],
)
@requires_sm100
@requires_int8_mma
def test_e2e_mainloop_int8(op, cfg) -> None:
    """INT8 mainloop fusion: f(int8 A) @ int8 B → int32 acc → fp32 out.
    Exercises the integer idesc + int32→fp32 widen; bit-exact vs int reference."""
    M, N, K = 512, 512, 256
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.INT8,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.INT32,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    Ai = getattr(g, op)(input=A, name="aop")
    Ai.set_data_type(cudnn.data_type.INT8)
    C = g.matmul(A=Ai, B=B, name="mm")
    C.set_output(True)
    C.set_data_type(cudnn.data_type.FLOAT)

    plan = _plan(g, **_kw(cfg))
    assert analyze(g).has_mainloop_fusion
    assert analyze(g).matmul.accum_dtype == "int32"

    torch.manual_seed(0)
    a = torch.randint(-8, 8, (1, M, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-8, 8, (1, N, K), dtype=torch.int8, device="cuda")
    c = torch.empty(1, M, N, dtype=torch.float32, device="cuda")
    plan(_vp(plan, a, b, c))
    torch.cuda.synchronize()

    ref = torch.einsum("bmk,bnk->bmn", _TORCH_OP[op](a.cpu().to(torch.int64)), b.cpu().to(torch.int64)).float()
    diff = (c.cpu() - ref).abs().max().item()
    assert diff == 0.0, f"{cfg} {op}: max|diff|={diff} (expected bit-exact)"


@requires_sm100
def test_e2e_mainloop_multicast_and_oob() -> None:
    # multicast_a + multicast_b cluster, plus M-OOB / K-OOB shape.
    _run_e2e(
        "abs",
        "CONFIG_sm100_128x256x128_128x256x32_cluster2x2_2ctamma",
        255,
        512,
        240,
        cudnn.data_type.BFLOAT16,
        torch.bfloat16,
    )


def _run_e2e_ab(aop, bop, cfg_name, M, N, K):
    g = _mainloop_graph_ab(aop, bop, M, N, K)
    plan = _plan(g, **_kw(cfg_name))
    torch.manual_seed(0)
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-3, 3).to(dtype=torch.bfloat16, device="cuda")
    b = torch.empty(1, N, K, dtype=torch.int32).random_(-3, 3).to(dtype=torch.bfloat16, device="cuda")
    c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    plan(_vp(plan, a, b, c))
    torch.cuda.synchronize()
    fa = _TORCH_OP[aop] if aop != "none" else (lambda x: x)
    fb = _TORCH_OP[bop] if bop != "none" else (lambda x: x)
    ref = torch.einsum("bmk,bnk->bmn", fa(a.float()), fb(b.float())).to(torch.bfloat16)
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    "aop,bop",
    [("none", "abs"), ("abs", "relu"), ("relu", "neg")],
)
@pytest.mark.parametrize(
    "cfg",
    [
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",  # CTA_1
        "CONFIG_sm100_64x128x128_64x128x32_cluster1x1_1ctamma",  # CTA_1 M=64 STG epi
        "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",  # CTA_2 m=256
        "CONFIG_sm100_128x256x128_128x256x32_cluster2x2_2ctamma",  # CTA_2 multicast_a
        "CONFIG_sm100_128x256x128_128x256x32_cluster4x1_2ctamma",  # CTA_2 multicast_b
        "CONFIG_sm100_64x256x128_64x256x32_cluster2x1_2ctamma",  # CTA_2 cluster MMA m=128
    ],
)
@requires_sm100
def test_e2e_mainloop_ab(aop, bop, cfg) -> None:
    _run_e2e_ab(aop, bop, cfg, 512, 1024, 256)


# ---------------------------------------------------------------------------
# Scalar-aux mainloop binary ops (e.g. A * alpha before the MMA)
# ---------------------------------------------------------------------------


def _scaled_graph(scale_a: bool, scale_b: bool, M, N, K):
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    Ai, Bi = A, B
    if scale_a:
        al = g.tensor(name="alpha", dim=[1, 1, 1], stride=[1, 1, 1])
        Ai = g.mul(a=A, b=al, name="sA")
        Ai.set_data_type(cudnn.data_type.BFLOAT16)
    if scale_b:
        be = g.tensor(name="beta", dim=[1, 1, 1], stride=[1, 1, 1])
        Bi = g.mul(a=B, b=be, name="sB")
        Bi.set_data_type(cudnn.data_type.BFLOAT16)
    C = g.matmul(A=Ai, B=Bi, name="mm")
    C.set_output(True)
    return g


def test_analyzer_scalar_aux_mainloop() -> None:
    chain = analyze(_scaled_graph(True, True, 256, 256, 128))
    assert [op.op for op in chain.mainloop_a_ops] == ["mul"]
    assert [op.op for op in chain.mainloop_b_ops] == ["mul"]
    assert chain.mainloop_a_ops[0].aux == "alpha"
    assert chain.mainloop_b_ops[0].aux == "beta"
    assert {t.name for t in chain.aux_tensors} == {"alpha", "beta"}
    assert all(t.bcast_mode == "scalar" for t in chain.aux_tensors)


def test_analyzer_rejects_per_row_mainloop_aux() -> None:
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, 256, 128], stride=[256 * 128, 128, 1])
    B = g.tensor(name="B", dim=[1, 128, 256], stride=[128 * 256, 1, 128])
    rv = g.tensor(name="rv", dim=[1, 256, 1], stride=[256, 1, 1])  # per-row of A
    As = g.mul(a=A, b=rv, name="rowscale")
    C = g.matmul(A=As, B=B, name="mm")
    C.set_output(True)
    with pytest.raises(ValueError, match="SCALAR"):
        analyze(g)


def _run_scaled(scale_a, scale_b, cfg_name, M, N, K, av=2.0, bv=0.5):
    g = _scaled_graph(scale_a, scale_b, M, N, K)
    plan = _plan(g, **_kw(cfg_name))
    torch.manual_seed(0)
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-3, 3).to(dtype=torch.bfloat16, device="cuda")
    b = torch.empty(1, N, K, dtype=torch.int32).random_(-3, 3).to(dtype=torch.bfloat16, device="cuda")
    c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    auxmap = {
        "alpha": torch.full((1, 1, 1), av, dtype=torch.bfloat16, device="cuda"),
        "beta": torch.full((1, 1, 1), bv, dtype=torch.bfloat16, device="cuda"),
    }
    plan(_vp(plan, a, b, c, *[auxmap[n] for n in [t.name for t in analyze(g).aux_tensors]]))
    torch.cuda.synchronize()
    aa = a.float() * (av if scale_a else 1.0)
    bb = b.float() * (bv if scale_b else 1.0)
    ref = torch.einsum("bmk,bnk->bmn", aa, bb).to(torch.bfloat16)
    torch.testing.assert_close(c, ref, atol=2e-1, rtol=2e-2)


@pytest.mark.parametrize("scale_a,scale_b", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize(
    "cfg",
    [
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
        "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
    ],
)
@requires_sm100
def test_e2e_scalar_aux_mainloop(scale_a, scale_b, cfg) -> None:
    _run_scaled(scale_a, scale_b, cfg, 512, 512, 256)


# ---------------------------------------------------------------------------
# K-OOB mask: cos (f(0)=1) on BOTH operands, partial last K-tile
# ---------------------------------------------------------------------------


def _as_major(t, major):
    """(1, X, K) buffer laid out K-contiguous ("k") or X-contiguous ("m"/"n")."""
    return t if major == "k" else t.transpose(1, 2).contiguous().transpose(1, 2)


def _run_cos2(cfg_name, M, N, K, a_major="k", b_major="k"):
    """cos(A) @ cos(B): both chains map 0->1, so a partial last K-tile needs the
    mainloop's swizzle-aware OOB zeroing (works for any K%8, incl. K%16!=0)."""
    g = _mainloop_graph_ab("cos", "cos", M, N, K, a_major=a_major, b_major=b_major)
    plan = _plan(g, **_kw(cfg_name))
    torch.manual_seed(0)
    a = (torch.rand(1, M, K, device="cuda") * 6 - 3).to(torch.bfloat16)
    b = (torch.rand(1, N, K, device="cuda") * 6 - 3).to(torch.bfloat16)
    c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    plan(_vp(plan, _as_major(a, a_major), _as_major(b, b_major), c))
    torch.cuda.synchronize()
    ref = torch.einsum("bmk,bnk->bmn", torch.cos(a.float()), torch.cos(b.float())).to(torch.bfloat16)
    torch.testing.assert_close(c, ref, atol=6e-1, rtol=2e-2)


@pytest.mark.parametrize("K", [288, 264, 200, 256])  # incl. K%16 != 0 (partial OOB block)
@pytest.mark.parametrize(
    "cfg",
    [
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",  # CTA_1
        "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",  # CTA_2 m=256
        "CONFIG_sm100_64x256x128_64x256x32_cluster2x1_2ctamma",  # CTA_2 cluster MMA m=128
    ],
)
@requires_sm100
def test_e2e_cos_both_koob(cfg, K) -> None:
    _run_cos2(cfg, 240, 272, K)


@pytest.mark.parametrize("a_major,b_major", [("m", "k"), ("k", "n"), ("m", "n")])
@pytest.mark.parametrize("K", [200, 256])  # 200 = partial last K-tile, 256 = exact
@pytest.mark.parametrize(
    "cfg",
    [
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
        "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
    ],
)
@requires_sm100
def test_e2e_cos_both_koob_mn_major(cfg, K, a_major, b_major) -> None:
    """The OOB mask reads the K coordinate off the flat SMEM index, which differs
    between a K-major tile (K fast) and an M-major one (M-groups)."""
    _run_cos2(cfg, 240, 272, K, a_major=a_major, b_major=b_major)


@pytest.mark.parametrize("op", ["abs", "relu"])
@pytest.mark.parametrize("a_major,b_major", [("m", "k"), ("k", "n"), ("m", "n")])
@pytest.mark.parametrize(
    "cfg",
    [
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
        "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
    ],
)
@requires_sm100
def test_e2e_mainloop_mn_major_operands(cfg, op, a_major, b_major) -> None:
    M, N, K = 512, 512, 256
    g = _mainloop_graph_ab(op, op, M, N, K, a_major=a_major, b_major=b_major)
    plan = _plan(g, **_kw(cfg))
    assert analyze(g).matmul.a_major == a_major
    assert analyze(g).matmul.b_major == b_major
    torch.manual_seed(0)
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-3, 3).to(dtype=torch.bfloat16, device="cuda")
    b = torch.empty(1, N, K, dtype=torch.int32).random_(-3, 3).to(dtype=torch.bfloat16, device="cuda")
    c = torch.zeros(1, M, N, dtype=torch.bfloat16, device="cuda")
    plan(_vp(plan, _as_major(a, a_major), _as_major(b, b_major), c))
    torch.cuda.synchronize()
    ref = torch.einsum("bmk,bnk->bmn", _TORCH_OP[op](a.float()), _TORCH_OP[op](b.float())).to(torch.bfloat16)
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("a_pad,b_pad", [(16, 0), (0, 16), (16, 16)])
@pytest.mark.parametrize(
    "cfg",
    [
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
        "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
    ],
)
@requires_sm100
def test_e2e_mainloop_padded_operands(cfg, a_pad, b_pad) -> None:
    """A K-padded (sliced) operand: the mainloop host must build its TMA layout
    from the runtime strides, not from an assumed-compact one."""
    M, N, K = 256, 256, 256
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * (K + a_pad), K + a_pad, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[N * (K + b_pad), 1, K + b_pad])
    Ai = g.abs(input=A, name="aop").set_data_type(cudnn.data_type.BFLOAT16)
    C = g.matmul(A=Ai, B=B, name="mm")
    C.set_output(True)
    plan = _plan(g, **_kw(cfg))

    torch.manual_seed(0)
    a_store = torch.empty(1, M, K + a_pad, dtype=torch.int32).random_(-3, 3).to(dtype=torch.bfloat16, device="cuda")
    b_store = torch.empty(1, N, K + b_pad, dtype=torch.int32).random_(-3, 3).to(dtype=torch.bfloat16, device="cuda")
    a, b = a_store[:, :, :K], b_store[:, :, :K]
    c = torch.zeros(1, M, N, dtype=torch.bfloat16, device="cuda")
    plan(_vp(plan, a, b, c))
    torch.cuda.synchronize()
    ref = torch.einsum("bmk,bnk->bmn", a.float().abs(), b.float()).to(torch.bfloat16)
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)


@requires_sm100
def test_e2e_cos_a_only_oob_ok() -> None:
    # cos on A only: B's OOB is raw zero-fill, so A_oob*B_oob=0 with no mask
    # needed even though cos(0)=1. K=264 (partial OOB) must still be correct.
    M, N, K = 256, 256, 264
    g = _mainloop_graph_ab("cos", "none", M, N, K)
    plan = _plan(g)
    torch.manual_seed(0)
    a = (torch.rand(1, M, K, device="cuda") * 6 - 3).to(torch.bfloat16)
    b = (torch.rand(1, N, K, device="cuda") * 6 - 3).to(torch.bfloat16)
    c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    plan(_vp(plan, a, b, c))
    torch.cuda.synchronize()
    ref = torch.einsum("bmk,bnk->bmn", torch.cos(a.float()), b.float()).to(torch.bfloat16)
    torch.testing.assert_close(c, ref, atol=6e-1, rtol=2e-2)


# ---------------------------------------------------------------------------
# Strict dtype: no implicit fp32 → MMA-dtype cast
# ---------------------------------------------------------------------------


def test_fp32_matmul_rejected_no_implicit_cast() -> None:
    """fp32 matmul is rejected at JIT (no TF32 path, no implicit fp32→bf16 cast).
    analyze() succeeds; the compiler's arch-aware gate rejects fp32×fp32→fp32."""

    def _fp32_graph(with_op: bool):
        g = cudnn.pygraph(
            io_data_type=cudnn.data_type.BFLOAT16,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        A = g.tensor(
            name="A",
            dim=[1, 128, 64],
            stride=[128 * 64, 64, 1],
            data_type=cudnn.data_type.FLOAT,
        )
        B = g.tensor(
            name="B",
            dim=[1, 64, 128],
            stride=[64 * 128, 128, 1],
            data_type=cudnn.data_type.FLOAT,
        )
        Ai = g.relu(input=A, name="r") if with_op else A
        C = g.matmul(A=Ai, B=B, name="mm")
        C.set_output(True)
        return g

    for with_op in (False, True):
        analyze(_fp32_graph(with_op))  # IR construction is fine — no dtype judgment
        gg = _fp32_graph(with_op)
        with pytest.raises(NotImplementedError, match="does not support input/acc dtype combo"):
            jit_from_cudnn_graph(gg, **_kw("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"))


# Mixed-input mainloop (dtype cast): the fused operand is LOADED narrower than
# the MMA reads (e.g. int8 A -> identity -> bf16 MMA); the mainloop warps widen
# the narrow tile into the MMA SMEM tile before the MMA.


def _mixed_input_graph(
    cast_a: bool,
    cast_b: bool,
    M: int,
    N: int,
    K: int,
    load=cudnn.data_type.INT8,
    tin=cudnn.data_type.BFLOAT16,
):
    """`identity(A_load) @ identity(B_load)` casting load->tin before the MMA.
    A non-cast operand stays at `tin` (no mainloop op on that side)."""
    g = cudnn.pygraph(
        io_data_type=tin,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    if cast_a:
        A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1], data_type=load)
        Ai = g.identity(input=A, name="pwa")
        Ai.set_data_type(tin)
    else:
        A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
        Ai = A
    if cast_b:
        B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K], data_type=load)
        Bi = g.identity(input=B, name="pwb")
        Bi.set_data_type(tin)
    else:
        B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
        Bi = B
    C = g.matmul(A=Ai, B=Bi, name="mm")
    C.set_output(True)
    return g, A, B, C


def test_analyzer_mixed_input_records_load_dtype() -> None:
    chain = analyze(_mixed_input_graph(True, False, 256, 256, 128)[0])
    # MMA reads bf16 (the explicit dtype feeding the matmul); A is LOADED int8.
    assert chain.matmul.a_dtype == "bf16"
    assert chain.matmul.b_dtype == "bf16"
    assert chain.mainloop_a_cast and chain.mainloop_a_load_dtype == "int8"
    assert not chain.mainloop_b_cast
    assert [op.op for op in chain.mainloop_a_ops] == ["identity"]


def test_analyzer_same_dtype_mainloop_is_not_a_cast() -> None:
    # abs(bf16) -> bf16: dtype-preserving, NOT a cast (regression guard).
    chain = analyze(_mainloop_graph("abs", 256, 256, 128))
    assert chain.matmul.a_dtype == "bf16"
    assert not chain.mainloop_a_cast and chain.mainloop_a_load_dtype is None


@pytest.mark.parametrize("cast_a,cast_b", [(True, False), (False, True), (True, True)], ids=["A", "B", "AB"])
@pytest.mark.parametrize(
    "cfg",
    [
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
        "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
    ],
)
@requires_sm100
def test_e2e_mixed_input_int8_bf16(cfg, cast_a, cast_b) -> None:
    M, N, K = 512, 512, 512
    g, A, B, C = _mixed_input_graph(cast_a, cast_b, M, N, K)
    plan = _plan(g, **_kw(cfg))
    assert analyze(g).has_mainloop_fusion
    assert analyze(g).mainloop_a_cast == cast_a
    assert analyze(g).mainloop_b_cast == cast_b
    torch.manual_seed(0)
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-4, 4).to(torch.int8 if cast_a else torch.bfloat16).cuda()
    b = torch.empty(1, N, K, dtype=torch.int32).random_(-4, 4).to(torch.int8 if cast_b else torch.bfloat16).cuda()
    c = torch.empty(1, M, N, dtype=torch.bfloat16).cuda()
    plan(_vp(plan, a, b, c))
    torch.cuda.synchronize()
    ref = torch.einsum("bmk,bnk->bmn", a.float(), b.float()).to(torch.bfloat16)
    # Bit-exact: identity is lossless and the values fit bf16 mantissa exactly.
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)


@requires_sm100
def test_e2e_mixed_input_koob() -> None:
    # K not a multiple of the K-tile (576 = 9 * 64): TMA zero-fills the partial
    # tile; int8 0 -> identity -> bf16 0, so K-OOB is harmless (no koob_fix).
    M, N, K = 512, 512, 576
    g, A, B, C = _mixed_input_graph(True, False, M, N, K)
    plan = _plan(g, **_kw("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"))
    torch.manual_seed(0)
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-4, 4).to(torch.int8).cuda()
    b = torch.empty(1, N, K, dtype=torch.int32).random_(-4, 4).to(torch.bfloat16).cuda()
    c = torch.empty(1, M, N, dtype=torch.bfloat16).cuda()
    plan(_vp(plan, a, b, c))
    torch.cuda.synchronize()
    ref = torch.einsum("bmk,bnk->bmn", a.float(), b.float()).to(torch.bfloat16)
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    "load,mma,torch_load,cu_mma",
    [
        ("int8", "bf16", torch.int8, cudnn.data_type.BFLOAT16),
        ("int8", "fp16", torch.int8, cudnn.data_type.HALF),
        ("e4m3", "bf16", torch.float8_e4m3fn, cudnn.data_type.BFLOAT16),
        (
            "int8",
            "e4m3",
            torch.int8,
            cudnn.data_type.FP8_E4M3,
        ),  # int->fp8 fold workaround
        (
            "int8",
            "e5m2",
            torch.int8,
            cudnn.data_type.FP8_E5M2,
        ),  # int->fp8 fold workaround
        ("bf16", "e4m3", torch.bfloat16, cudnn.data_type.FP8_E4M3),  # narrowing to fp8
    ],
)
@requires_sm100
def test_e2e_mixed_input_cast_targets(load, mma, torch_load, cu_mma) -> None:
    """Cast A (loaded `load`) -> `mma` dtype; B is the same `mma` dtype. int->fp8
    exercises the fold workaround (else int->fp32->fp8 folds to NaN)."""
    cu_load = {
        "int8": cudnn.data_type.INT8,
        "e4m3": cudnn.data_type.FP8_E4M3,
        "bf16": cudnn.data_type.BFLOAT16,
    }[load]
    M, N, K = 512, 512, 512
    g = cudnn.pygraph(
        io_data_type=cu_mma,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1], data_type=cu_load)
    Ai = g.identity(input=A, name="pw")
    Ai.set_data_type(cu_mma)
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=Ai, B=B, name="mm")
    C.set_output(True)
    C.set_data_type(cudnn.data_type.BFLOAT16)
    plan = _plan(g, **_kw("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"))
    load_lit = {"int8": "int8", "e4m3": "fp8_e4m3", "bf16": "bf16"}[load]
    assert analyze(g).mainloop_a_cast and analyze(g).mainloop_a_load_dtype == load_lit
    torch.manual_seed(0)
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-4, 4).to(torch_load).cuda()
    b = (
        torch.empty(1, N, K, dtype=torch.int32)
        .random_(-4, 4)
        .to(
            {
                cudnn.data_type.BFLOAT16: torch.bfloat16,
                cudnn.data_type.HALF: torch.float16,
                cudnn.data_type.FP8_E4M3: torch.float8_e4m3fn,
                cudnn.data_type.FP8_E5M2: torch.float8_e5m2,
            }[cu_mma]
        )
        .cuda()
    )
    c = torch.empty(1, M, N, dtype=torch.bfloat16).cuda()
    plan(_vp(plan, a, b, c))
    torch.cuda.synchronize()
    ref = torch.einsum("bmk,bnk->bmn", a.float(), b.float()).to(torch.bfloat16)
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)


def test_codegen_int_to_fp8_fold_war() -> None:
    """GPU-free guard for the int->fp8 fold workaround (foot-gun #3):
    generate_mainloop inserts `+ cutlass.full_like(., 0.0)` before the fp8
    narrowing for integer-loaded operands (else int->fp32->fp8 folds to NaN);
    absent for int->bf16."""
    # int8 -> fp8: workaround present, narrows to fp8.
    chain_fp8 = analyze(
        _mixed_input_graph(
            True,
            False,
            256,
            256,
            256,
            load=cudnn.data_type.INT8,
            tin=cudnn.data_type.FP8_E4M3,
        )[0]
    )
    snip = generate_mainloop(chain_fp8, "a")
    assert "cutlass.full_like" in snip and "0.0" in snip
    assert ".to(cutlass.Float8E4M3FN)" in snip
    # int8 -> bf16: no fp8 narrowing, so no workaround.
    chain_bf16 = analyze(
        _mixed_input_graph(
            True,
            False,
            256,
            256,
            256,
            load=cudnn.data_type.INT8,
            tin=cudnn.data_type.BFLOAT16,
        )[0]
    )
    assert "cutlass.full_like" not in generate_mainloop(chain_bf16, "a")


@requires_sm100
def test_e2e_mixed_input_int8_fp8_2ctamma() -> None:
    """int8 -> fp8 (fold workaround) on the 2-CTA template, to cover both staging
    paths (1ctamma is covered by test_e2e_mixed_input_cast_targets)."""
    M, N, K = 512, 512, 512
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.FP8_E4M3,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1], data_type=cudnn.data_type.INT8)
    Ai = g.identity(input=A, name="pw")
    Ai.set_data_type(cudnn.data_type.FP8_E4M3)
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=Ai, B=B, name="mm")
    C.set_output(True)
    C.set_data_type(cudnn.data_type.BFLOAT16)
    plan = _plan(g, **_kw("CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma"))
    assert analyze(g).mainloop_a_cast and analyze(g).matmul.a_dtype == "fp8_e4m3"
    torch.manual_seed(0)
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-4, 4).to(torch.int8).cuda()
    b = torch.empty(1, N, K, dtype=torch.int32).random_(-4, 4).to(torch.float8_e4m3fn).cuda()
    c = torch.empty(1, M, N, dtype=torch.bfloat16).cuda()
    plan(_vp(plan, a, b, c))
    torch.cuda.synchronize()
    ref = torch.einsum("bmk,bnk->bmn", a.float(), b.float()).to(torch.bfloat16)
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)
