# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Epilogue ops that compute in INT32, on the sm120 tree.

The shared epilogue emitter computes only some pointwise ops on Int32 vectors; the
others use Float32-only CuTe DSL math. FROST declines those in check_support, so the
default build_plans() walk reaches an engine that computes them, instead of
cute.compile raising a TypeError or DSL error (comparisons, floor, mod, ...) or, for
div, the int32 output holding the bit pattern of the fp32 quotient. It also declines
an int32-compute op that reads or writes fp8, whose int32 <-> fp8 casts fail in the
NVVM backend. gen_index computing in int32 gives exact indices past 2**24.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from gemm_test_utils import requires_sm120

import cudnn
import cudnn.gemm.frost  # noqa: F401  -- installs the cudnn.pygraph recorder hook
from cudnn.engines.base import decline_types
from cudnn.gemm.frost.engine import FrostGemmEngine
from cudnn.gemm.frost.fusion_ir import FusionOp, MoeSwapAbSpec
from cudnn.gemm.frost.tile_config import by_name

pytestmark = pytest.mark.L0

_I8 = cudnn.data_type.INT8
_I32 = cudnn.data_type.INT32
_F32 = cudnn.data_type.FLOAT
_E4M3 = cudnn.data_type.FP8_E4M3
_E5M2 = cudnn.data_type.FP8_E5M2
_CFG = "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"
M = N = K = 128

# Graph-API keywords of the multi-input ops below; the GEMM result is the first
# operand and a per-column int32 aux ``z`` fills the others.
_OPERANDS = {
    **{op: ("a", "b") for op in ("add", "sub", "mul", "div", "add_square", "logical_and")},
    **{op: ("input0", "input1") for op in ("max", "min", "mod", "pow")},
    **{op: ("input", "comparison") for op in ("cmp_lt", "cmp_eq")},
    "relu_backward": ("loss", "input"),
    "binary_select": ("input0", "input1", "mask"),
}


def _graph(op: str, out=_I32, aux=_I32, gemm_out=None):
    """int8 A x int8 B -> one pointwise op -> ``out`` (int32 by default), all computing
    in INT32 (the graph default of an int8 GEMM)."""
    g = cudnn.pygraph(io_data_type=_I8, intermediate_data_type=_F32, compute_data_type=_I32)
    a = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    b = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    c = g.matmul(A=a, B=b, name="mm")
    if gemm_out is not None:
        c.set_data_type(gemm_out)
    z = None
    if op in _OPERANDS:
        z = g.tensor(name="z", dim=[1, 1, N], stride=[N, N, 1], data_type=aux)
        y = getattr(g, op)(**dict(zip(_OPERANDS[op], (c, z, z))), name="op")
    elif op == "leaky_relu":
        y = g.leaky_relu(input=c, negative_slope=0.5, name="op")
    elif op == "gen_index":
        y = g.add(a=c, b=g.gen_index(input=c, axis=1, name="gi").set_data_type(_I32), name="op")
    else:
        y = getattr(g, op)(input=c, name="op")
    y.set_output(True).set_data_type(out)
    return g, a, b, z, y


# One op per way the Float32-only emitters fail on Int32 vectors: the floor-based
# step forms (comparisons, logical ops, backward masks), floor / ceil, the `/` of mod
# (a DSL error) and of div (no error, wrong bits), and the exp2 / rcp / sqrt / powf math.
_NO_INT32_EMITTER = ("cmp_lt", "cmp_eq", "logical_and", "logical_not", "floor", "ceil", "mod", "div", "exp", "sigmoid", "sqrt", "pow", "relu_backward")

_INT32_EMITTER = ("identity", "relu", "leaky_relu", "abs", "neg", "add", "sub", "mul", "max", "min", "add_square", "binary_select", "gen_index")


@requires_sm120
@pytest.mark.parametrize("op", _NO_INT32_EMITTER)
def test_sm120_int32_compute_op_without_an_int32_emitter_is_declined(op):
    """check_support declines it, and so does the direct JIT entry: a decline, not a
    TypeError / DSL error out of cute.compile, nor a plan that writes wrong bits."""
    from cudnn.gemm.frost.sm120 import compiler

    g, *_ = _graph(op)
    with pytest.raises(NotImplementedError, match=rf"\('{op}'\) computes in int32"):
        FrostGemmEngine().check_support(g)
    with pytest.raises(NotImplementedError, match=rf"\('{op}'\) computes in int32"):
        compiler.jit_from_cudnn_graph(g, config=by_name(_CFG))


@requires_sm120
@pytest.mark.parametrize("op", _INT32_EMITTER)
def test_sm120_int32_compute_op_with_an_int32_emitter_is_accepted(op):
    """The decline keeps every op the emitter computes in Int32."""
    g, *_ = _graph(op)
    FrostGemmEngine().check_support(g)


def _reference(op, a, b, z):
    c = torch.einsum("bmk,bnk->bmn", a.cpu().long(), b.cpu().long())
    z = z.cpu().long()
    ref = {
        "cmp_lt": lambda: (c < z).long(),
        "floor": lambda: c,
        "div": lambda: torch.div(c, z, rounding_mode="trunc"),
        "mod": lambda: torch.fmod(c, z),
    }[op]()
    return ref.to(torch.int32)


@requires_sm120
@pytest.mark.parametrize("op", ("cmp_lt", "floor", "div", "mod"))
def test_sm120_int32_compute_default_walk_computes_exactly(op):
    """The public default walk, no plan pinned. FROST used to be its first plan:
    cmp_lt / floor / mod aborted build_plans() with a TypeError or DSL error, and div
    built and stored the fp32 quotient's bits as int32."""
    g, A, B, Z, Y = _graph(op)
    try:
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        g.build_plans()
    except decline_types() as exc:
        pytest.skip(f"no engine in this cuDNN computes the int32 {op} graph: {str(exc)[:160]}")
    torch.manual_seed(0)
    a = torch.randint(-4, 4, (1, M, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-4, 4, (1, N, K), dtype=torch.int8, device="cuda")
    z = torch.arange(N, dtype=torch.int32, device="cuda").reshape(1, 1, N) % 7 - 3
    z = torch.where(z == 0, torch.full_like(z, 2), z)  # div / mod by a nonzero integer of either sign
    y = torch.full((1, M, N), -12345, dtype=torch.int32, device="cuda")
    workspace = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    g.execute({A: a, B: b, Z: z, Y: y}, workspace)
    torch.cuda.synchronize()
    torch.testing.assert_close(y.cpu(), _reference(op, a, b, z), atol=0, rtol=0)


# An op with an int32 emitter (add) whose output, aux or GEMM input is fp8: the int32 <-> fp8
# casts around it do not compile (the NVVM backend raises CompilerDiagnosticError).
_FP8_EDGES = {
    "fp8_e4m3 output": dict(out=_E4M3),
    "fp8_e5m2 output": dict(out=_E5M2),
    "fp8_e4m3 aux": dict(aux=_E4M3),
    "fp8_e4m3 GEMM output": dict(gemm_out=_E4M3),
}


@requires_sm120
@pytest.mark.parametrize("edge", tuple(_FP8_EDGES))
def test_sm120_int32_compute_with_fp8_input_or_output_is_declined(edge):
    """check_support and the direct JIT entry decline it, instead of cute.compile failing."""
    from cudnn.gemm.frost.sm120 import compiler

    g, *_ = _graph("add", **_FP8_EDGES[edge])
    match = rf"\('add'\) computes in int32 but reads or writes {edge.split()[0]}"
    with pytest.raises(NotImplementedError, match=match):
        FrostGemmEngine().check_support(g)
    with pytest.raises(NotImplementedError, match=match):
        compiler.jit_from_cudnn_graph(g, config=by_name(_CFG))


@requires_sm120
def test_sm120_int32_gen_index_of_an_fp8_gemm_output_is_accepted():
    """gen_index reads only the shape of its input, so an fp8 GEMM output under an int32
    gen_index needs no conversion and FROST keeps the graph."""
    g = cudnn.pygraph(io_data_type=_I8, intermediate_data_type=_F32, compute_data_type=_I32)
    a = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    b = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    c = g.matmul(A=a, B=b, name="mm").set_data_type(_E4M3)
    g.gen_index(input=c, axis=1, name="gi").set_output(True).set_data_type(_I32)
    FrostGemmEngine().check_support(g)


_TORCH_FP8 = {_E4M3: torch.float8_e4m3fn, _E5M2: torch.float8_e5m2}


@requires_sm120
@pytest.mark.parametrize("edge", ("fp8_e4m3 output", "fp8_e5m2 output", "fp8_e4m3 aux"))
def test_sm120_int32_compute_with_fp8_default_walk_computes_exactly(edge):
    """The public default walk for an int32 add with an fp8 output or aux. FROST used to
    be its first plan, and build_plans() raised CompilerDiagnosticError (not a decline
    type) out of the NVVM backend."""
    edge_kw = _FP8_EDGES[edge]
    g, A, B, Z, Y = _graph("add", **edge_kw)
    try:
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        g.build_plans()
    except decline_types() as exc:
        pytest.skip(f"no engine in this cuDNN computes the int32 add with an {edge}: {str(exc)[:160]}")
    torch.manual_seed(0)
    # Entries in {-1, 0, 1} keep |A @ B + z| <= 131, inside the fp8 range; z is exact in fp8.
    a = torch.randint(-1, 2, (1, M, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-1, 2, (1, N, K), dtype=torch.int8, device="cuda")
    z = torch.arange(N, dtype=torch.int32, device="cuda").reshape(1, 1, N) % 7 - 3
    z_in = z.float().to(_TORCH_FP8[edge_kw["aux"]]) if "aux" in edge_kw else z
    out_dtype = _TORCH_FP8.get(edge_kw.get("out"), torch.int32)
    if out_dtype == torch.int32:
        y = torch.full((1, M, N), -12345, dtype=torch.int32, device="cuda")
    else:
        y = torch.full((1, M, N), float("nan"), device="cuda").to(out_dtype)
    workspace = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    g.execute({A: a, B: b, Z: z_in, Y: y}, workspace)
    torch.cuda.synchronize()
    ref = torch.einsum("bmk,bnk->bmn", a.cpu().long(), b.cpu().long()) + z.cpu().long()
    if out_dtype == torch.int32:
        torch.testing.assert_close(y.cpu(), ref.to(torch.int32), atol=0, rtol=0)
    else:
        torch.testing.assert_close(y.cpu().float(), ref.float().to(out_dtype).float(), atol=0, rtol=0)


@requires_sm120
def test_sm120_int32_gen_index_is_exact_past_2_24():
    """A row index above 2**24 has no fp32 representation: with int32 compute and an
    int32 output, row 16777217 was written as 16777216. A == 0, so Y == row index."""
    from cudnn.gemm.frost.sm120 import compiler

    rows, cols, depth = (1 << 24) + 64, 4, 16
    g = cudnn.pygraph(io_data_type=_I8, intermediate_data_type=_F32, compute_data_type=_I32)
    A = g.tensor(name="A", dim=[1, rows, depth], stride=[rows * depth, depth, 1])
    B = g.tensor(name="B", dim=[1, depth, cols], stride=[depth * cols, 1, depth])
    C = g.matmul(A=A, B=B, name="mm")
    Y = g.add(a=C, b=g.gen_index(input=C, axis=1, name="gi").set_data_type(_I32), name="ad")
    Y.set_output(True).set_data_type(_I32)
    compiled = compiler.jit_from_cudnn_graph(g, config=by_name(_CFG))
    assert [(op.op, op.compute_dtype) for op in compiled.chain.ops] == [("gen_index", "int32"), ("add", "int32")]
    try:
        a = torch.zeros(1, rows, depth, dtype=torch.int8, device="cuda")
        b = torch.zeros(1, cols, depth, dtype=torch.int8, device="cuda")
        y = torch.full((1, rows, cols), -1, dtype=torch.int32, device="cuda")
    except torch.cuda.OutOfMemoryError:
        pytest.skip("not enough free device memory for the 2**24-row operands")
    compiled({A: a, B: b, Y: y})
    torch.cuda.synchronize()
    row_index = torch.arange(rows, dtype=torch.int32, device="cuda").view(1, rows, 1)
    tail = slice((1 << 24) - 64, rows)
    torch.testing.assert_close(y[:, tail], row_index[:, tail].expand(1, -1, cols), atol=0, rtol=0)
    assert bool((y == row_index).all())


@requires_sm120
@pytest.mark.parametrize("swap_ab", (False, True))
def test_sm120_int32_gen_index_scatters_moe_rows(swap_ab):
    """MoE SCATTER replaces the token-axis index with the routed destination row; with
    int32 compute it is built from that Int32 row too, on token axis 1 or, under
    swap-AB, token axis 2 (small extents: a compile and placement check of the int32
    form)."""
    from cudnn.gemm.frost.sm120 import compiler

    rows, n, k, top_k = 387, 96, 96, 3
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=_F32, compute_data_type=_F32)
    x = g.tensor(name="tokens", dim=[1, rows, k], stride=[rows * k, k, 1])
    w = g.tensor(name="weights", dim=[3, k, n], stride=[k * n, 1, k])
    offsets = g.tensor(name="offsets", dim=[7, 1, 1], stride=[1, 1, 1], data_type=_I32)
    indices = g.tensor(name="indices", dim=[1, rows, 1], stride=[rows, 1, 1], data_type=_I32)
    ks = g.tensor(name="ks", dim=[1, rows, 1], stride=[rows, 1, 1], data_type=_I32)
    y = g.moe_grouped_matmul(x, w, offsets, token_index=indices, token_ks=ks, top_k=top_k, mode=cudnn.moe_grouped_matmul_mode.SCATTER)
    y.set_data_type(_F32)
    row_index = g.gen_index(input=y, axis=1, compute_data_type=_I32).set_data_type(_I32)
    col_index = g.gen_index(input=y, axis=2, compute_data_type=_I32).set_data_type(_I32)
    out = g.add(a=row_index, b=col_index, compute_data_type=_I32)
    out.set_output(True).set_data_type(_I32)
    compiled = compiler.jit_from_cudnn_graph(g, config=replace(by_name(_CFG), swap_ab=swap_ab))
    assert compiled.chain.moe.mode == "scatter"
    assert isinstance(compiled.chain.moe, MoeSwapAbSpec) == swap_ab
    assert [(op.op, op.compute_dtype) for op in compiled.chain.ops] == [("gen_index", "int32"), ("gen_index", "int32"), ("add", "int32")]
    destination = torch.randperm(rows, device="cuda", dtype=torch.int32)
    vp = {
        x: torch.ones(1, rows, k, device="cuda", dtype=torch.bfloat16),
        w: torch.ones(3, n, k, device="cuda", dtype=torch.bfloat16),
        offsets: torch.tensor([0, 3, 3, 134, 257, 258, rows], device="cuda", dtype=torch.int32),
        indices: (destination // top_k).view(1, rows, 1),
        ks: (destination % top_k).view(1, rows, 1),
        out: torch.full((1, rows, n), -7, device="cuda", dtype=torch.int32),
    }
    compiled(vp)
    torch.cuda.synchronize()
    expected = torch.arange(rows, device="cuda", dtype=torch.int32).view(1, rows, 1) + torch.arange(n, device="cuda", dtype=torch.int32).view(1, 1, n)
    torch.testing.assert_close(vp[out], expected, atol=0, rtol=0)


@pytest.mark.parametrize("axis", (1, 2))
def test_int32_gen_index_is_emitted_from_int32_coordinates(axis):
    """Host-side pin of both gen_index forms (the column form cannot reach 2**24 on
    sm120, whose grid.y caps N at 65535 tiles). fp32 compute keeps its Float32 index."""
    from cudnn.gemm.frost.epilogue_codegen import _emit_op

    def emit(compute):
        op = FusionOp("gen_index", attrs=(("axis", float(axis)),), compute_dtype=compute, out_dtype=compute)
        lines, _ = _emit_op(op, "_c", 0, {}, vsize=8)
        return "\n".join(lines)

    int32_src, fp32_src = emit("int32"), emit("fp32")
    assert "cutlass.Int32(" in int32_src and "Float32" not in int32_src, int32_src
    assert "cutlass.Float32(" in fp32_src and "Int32" not in fp32_src, fp32_src
