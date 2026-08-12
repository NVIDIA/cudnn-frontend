# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The build-time recipe, and the closure lowered from it.

``_lower`` captures the recipe into a call path that loops over it flat;
``launch`` interprets the same recipe by walking the operand structure. That is
a compiler beside its interpreter, and the two can drift -- an earlier
hand-written version of the emitted path lost the operand batch check and pinned
an fp4 output at N instead of N/2, both of which made it accept or reject calls
the general path did not.

So the differentials below are the point of this file: every case and every
flavor runs through BOTH entry points, and the two must return the same verdict
and the same numbers. The rest asserts that the fast path is the one a public
``execute()`` actually takes, and that what it declines is declined on purpose.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from gemm_test_utils import requires_sm100

import cudnn
import cudnn.gemm.frost  # noqa: F401 — installs the cudnn.pygraph recorder hook
from cudnn.engines import is_python_engine
from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
from cudnn.gemm.frost.graph_analyzer import resolve_variant_pack
from cudnn.gemm.frost.recipe import _output_rule, expected_shape

pytestmark = pytest.mark.L0

BF16 = cudnn.data_type.BFLOAT16
F32 = cudnn.data_type.FLOAT
M = N = 256
K = 128


# --- the output shape rule, which is pure data ------------------------------


def _spec(**kw):
    base = dict(source="matmul", dtype="bf16", dim=None, is_reduction=False, is_quant_scale=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _chain(batch=1, reductions=()):
    return SimpleNamespace(matmul=SimpleNamespace(batch=batch), reductions=list(reductions))


def test_dense_output_follows_m_and_n():
    assert expected_shape(_output_rule(_spec(), _chain(batch=3)), 128, 256) == (3, 128, 256)


def test_fp4_dense_output_is_half_as_wide():
    """fp4 packs two elements per stored slot, so the last axis is N/2.

    The hand-written launch path this replaces pinned it at N, which rejected a
    legal call. One rule, read by both paths, is the answer to that.
    """
    assert expected_shape(_output_rule(_spec(dtype="fp4_e2m1"), _chain()), 128, 256) == (1, 128, 128)


def test_a_reduced_axis_collapses_to_one():
    chain = _chain(batch=2, reductions=[SimpleNamespace(grouped_by_moe=False)])
    rule = _output_rule(_spec(source="reduction_0", is_reduction=True, dim=(1, 128, 1)), chain)
    assert expected_shape(rule, 512, 256) == (1, 512, 1)


def test_a_quant_scale_output_is_fixed_at_build():
    rule = _output_rule(_spec(source="quant_scale_0", is_quant_scale=True, dim=(1, 128, 4)), _chain())
    assert expected_shape(rule, 999, 999) == (1, 128, 4)


# --- which flavors lower, and which decline ---------------------------------


def _plain_graph(m=M, n=N, k=K, batch=1):
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    A = g.tensor(name="A", dim=[batch, m, k], stride=[m * k, k, 1])
    B = g.tensor(name="B", dim=[batch, k, n], stride=[k * n, 1, k])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(BF16)
    return g


def _reduction_graph():
    g = _plain_graph()
    Y = [t for t in g._nodes[-1].outputs.values()][0]
    R = g.reduction(input=Y, mode=cudnn.reduction_mode.ADD, name="red")
    R.set_dim([1, 1, 1]).set_stride([1, 1, 1]).set_output(True).set_data_type(F32)
    return g


def _aux_graph():
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    bias = g.tensor(name="bias", dim=[1, 1, N], stride=[N, N, 1], data_type=F32)
    C = g.matmul(A=A, B=B, name="mm")
    Y = g.add(a=C, b=bias, name="bias_add")
    Y.set_output(True).set_data_type(BF16)
    return g


def _epilogue_graph():
    g = _plain_graph()
    mm = [t for t in g._nodes[-1].outputs.values()][0]
    mm.set_output(False)
    g.relu(input=mm, name="relu").set_output(True).set_data_type(BF16)
    return g


def _two_output_graph():
    g = _plain_graph()
    mm = [t for t in g._nodes[-1].outputs.values()][0]
    g.relu(input=mm, name="relu").set_output(True).set_data_type(BF16)
    return g


def _multi_gemm_graph():
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B0 = g.tensor(name="B0", dim=[1, K, N], stride=[K * N, 1, K])
    B1 = g.tensor(name="B1", dim=[1, K, N], stride=[K * N, 1, K])
    Y = g.add(a=g.matmul(A=A, B=B0, name="mm0"), b=g.matmul(A=A, B=B1, name="mm1"), name="sum")
    Y.set_output(True).set_data_type(BF16)
    return g


# Each entry is (build, how many distinct B operands, how many outputs, how many aux).
_FLAVORS = (
    (_plain_graph, 1, 1, 0),
    (_epilogue_graph, 1, 1, 0),
    (_aux_graph, 1, 1, 1),
    (_two_output_graph, 1, 2, 0),
    (_reduction_graph, 1, 2, 0),
    (_multi_gemm_graph, 2, 1, 0),
)
_FLAVOR_IDS = ("plain", "epilogue", "aux", "two_outputs", "reduction", "multi_gemm")


@requires_sm100
@pytest.mark.parametrize("build,n_b,n_out,n_aux", _FLAVORS, ids=_FLAVOR_IDS)
def test_which_flavors_lower(build, n_b, n_out, n_aux):
    """Every flavor lowers, because what differs between them is a table entry.

    Aux, extra outputs, a reduction's seed and a multi-GEMM's operand list all
    used to be branches in a launcher, so the emitted path served only the one
    shape with none of them. They are ``arg_plan`` and ``seeds`` now, which the
    same loop reads -- so the list below is a list of table shapes, not of code
    paths, and adding to it is data.
    """
    compiled = jit_from_cudnn_graph(build())
    assert compiled.lowered is not None
    r = compiled.recipe
    assert (len(r.inputs) - 1, len(r.outputs), len(r.aux)) == (n_b, n_out, n_aux)
    # One launch argument per bound operand: nothing dropped, nothing passed twice.
    assert len(r.arg_plan) == len(r.inputs) + len(r.outputs) + len(r.aux) + len(r.sf)


@requires_sm100
def test_a_post_kernel_finalize_is_declined():
    """``norm2`` takes a square root through the caller's buffer after the kernel.

    The backend refuses that reduction while the graph is being lowered, so it
    never reaches a plan -- but the recipe records ``sqrt`` and the lowered path
    declines on it, because the alternative is a device operation the engine
    does not own and would have to borrow off whatever the caller passed.
    """
    compiled = jit_from_cudnn_graph(_plain_graph())
    assert compiled.lowered is not None
    compiled.recipe = replace(compiled.recipe, outputs=(replace(compiled.recipe.outputs[0], sqrt=True),))
    assert compiled._lower() is None


@requires_sm100
def test_public_execute_takes_the_lowered_path():
    """The lowered path is what a user's ``execute()`` runs, not a side door."""
    g = _plain_graph()
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    frost = [i for i, p in enumerate(g.plans) if is_python_engine(p.engine_id)]
    if not frost:
        pytest.skip("no FROST plan for this graph")
    g.select_plan(frost[0])
    g.check_support()
    g.build_plans()
    assert g._compiled_plans[g._plan_index]._lowered is not None


# --- the differential -------------------------------------------------------


def _bind(compiled, a_bufs, b_bufs, out_bufs, aux_bufs=()):
    """The operand list in bound-tensor order, which is what both paths take."""
    bd = compiled.binding
    pack = {}
    for role, bufs in ((bd.a_operands, a_bufs), (bd.b_operands, b_bufs), (bd.outputs, out_bufs), (bd.aux, aux_bufs)):
        pack.update(zip(role, bufs))
    resolved = resolve_variant_pack(pack, bd)
    return [resolved[id(t)] for t in bd.bound_tensors()]


def _bound_buffers(compiled, a, b, c):
    return _bind(compiled, [a], [b], [c])


def _buffers_for(compiled):
    """One buffer per bound tensor, sized from the graph's own declaration.

    Every flavor's operands follow from the recipe, so the differential below
    does not need a hand-written variant pack per flavor -- which is the same
    reason the launch path does not need a launcher per flavor.
    """
    r = compiled.recipe
    bufs = [None] * len(compiled.binding.bound_tensors())
    for op in r.inputs:
        rows = N if op.is_b else M
        bufs[op.index] = torch.randn(op.batch, rows, K // op.kpack, dtype=torch.bfloat16, device="cuda")
    for out in r.outputs:
        dtype = torch.float32 if out.raw else torch.bfloat16
        bufs[out.index] = torch.zeros(expected_shape(out.rule, M, N), dtype=dtype, device="cuda")
    for x in r.aux:
        bufs[x.index] = torch.randn(tuple(int(d) for d in x.ref.dim), dtype=torch.float32, device="cuda")
    assert all(b is not None for b in bufs)
    return bufs


@requires_sm100
@pytest.mark.parametrize("build,n_b,n_out,n_aux", _FLAVORS, ids=_FLAVOR_IDS)
def test_every_flavor_agrees_with_the_interpreter(build, n_b, n_out, n_aux):
    """One loop serves six flavors, so one of them drifting is the failure mode.

    The interpreter is the reference: it reads the same recipe but assembles the
    launch by walking the operand structure, which is the thing the lowered path
    replaced with a table. Bit-exact is the bar -- the two issue the same kernel
    with the same arguments or they do not agree.

    Except for a reduction output, which is not bit-reproducible against ITSELF:
    the taps land through cross-CTA float atomics, so the order varies with
    scheduling. Measured on this shape, the same path run six times spreads
    0.0049 while the two paths differ by 0.00024 -- twenty times smaller than
    the noise, so the tolerance below is the noise floor and not a slackened bar.
    """
    compiled = jit_from_cudnn_graph(build())
    if compiled.lowered is None:
        pytest.skip("this build does not lower (no tvm-ffi front door)")
    operands = _buffers_for(compiled)
    outs = compiled.recipe.outputs

    runs = []
    for run in (compiled.lowered, compiled.launch):
        for o in outs:
            operands[o.index].zero_()
        run(operands, stream=None)
        torch.cuda.synchronize()
        runs.append([operands[o.index].clone() for o in outs])
    for o, fast, slow in zip(outs, *runs):
        exact = o.init is None
        torch.testing.assert_close(fast, slow, atol=0 if exact else 1e-2, rtol=0 if exact else 1e-5)
    assert runs[0][0].abs().sum() > 0  # a path that wrote nothing would also "agree"


def _verdict(run, operands, c):
    c.zero_()
    try:
        run(operands, stream=None)
    except ValueError:
        return "rejected", None
    torch.cuda.synchronize()
    return "ran", c.clone()


def _operands(m=M, n=N, k=K, batch=1):
    a = torch.randn(batch, m, k, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(batch, n, k, dtype=torch.bfloat16, device="cuda")
    c = torch.empty(batch, m, n, dtype=torch.bfloat16, device="cuda")
    return a, b, c


def _good():
    return _operands()


def _short_k():
    a, _, c = _operands()
    return a, torch.randn(1, N, K // 2, dtype=torch.bfloat16, device="cuda"), c


def _wrong_major():
    a, _, c = _operands()
    return a, torch.randn(1, K, N, dtype=torch.bfloat16, device="cuda").transpose(1, 2), c


def _wrong_output_shape():
    a, b, _ = _operands()
    return a, b, torch.empty(1, M, N // 2, dtype=torch.bfloat16, device="cuda")


def _misaligned_output():
    a, b, _ = _operands()
    # One element in is one element off every alignment the epilogue wants.
    return a, b, torch.empty(1, M, N + 8, dtype=torch.bfloat16, device="cuda")[:, :, 1 : N + 1]


def _misaligned_a():
    """One bf16 element in is 2 bytes in, and TMA wants a 16-byte base."""
    _, b, c = _operands()
    wide = torch.randn(1, M, K + 8, dtype=torch.bfloat16, device="cuda")
    return wide[:, :, 1 : K + 1], b, c


def _padded_rows():
    """Legal: the outer stride is free, only the contiguous extent is pinned."""
    a, b, c = _operands()
    return a, b, torch.empty(1, M, N * 2, dtype=torch.bfloat16, device="cuda")[:, :, :N]


@requires_sm100
@pytest.mark.parametrize(
    "case",
    (_good, _short_k, _wrong_major, _wrong_output_shape, _misaligned_output, _misaligned_a, _padded_rows),
    ids=lambda f: f.__name__.strip("_"),
)
def test_lowered_and_interpreted_agree(case):
    """Same operands, both entry points, one verdict.

    An unsound fast path shows up here as an accept where the general path
    rejects -- which is exactly how the two regressions this replaced would have
    read.
    """
    compiled = jit_from_cudnn_graph(_plain_graph())
    if compiled.lowered is None:
        pytest.skip("this build does not lower (no tvm-ffi front door)")

    a, b, c = case()
    fast, fast_out = _verdict(compiled.lowered, _bound_buffers(compiled, a, b, c), c)
    slow, slow_out = _verdict(compiled.launch, _bound_buffers(compiled, a, b, c), c)
    assert fast == slow, f"lowered says {fast}, interpreted says {slow}"
    if fast == "ran":
        torch.testing.assert_close(fast_out, slow_out, atol=0, rtol=0)
        ref = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
        torch.testing.assert_close(fast_out.float(), ref, atol=2e-1, rtol=2e-2)


def _matmul_on(batch, m, n, k, want_frost, a, b, c):
    """Build the plain graph, pin a backend or a FROST plan, run it."""
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    A = g.tensor(name="A", dim=[batch, m, k], stride=[m * k, k, 1])
    B = g.tensor(name="B", dim=[batch, k, n], stride=[k * n, 1, k])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(BF16)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    hits = [i for i, p in enumerate(g.plans) if is_python_engine(p.engine_id) == want_frost]
    if not hits:
        return None
    g.select_plan(hits[0])
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    g.execute({A: a, B: b, C: c}, ws)
    torch.cuda.synchronize()
    return c


@requires_sm100
@pytest.mark.parametrize("batch", (1, 2), ids=("b1", "b2"))
@pytest.mark.parametrize("m,n,k", [(m, n, k) for m in (1, 128) for n in (1, 128) for k in (1, 128)], ids=str)
def test_a_degenerate_extent_is_refused_or_matches_the_backend(monkeypatch, batch, m, n, k):
    """An extent of 1 leaves its axis's stride free, so two majors can look alike.

    ``(batch, M, 1)`` k-major carries stride 1 on axis 1 AND axis 2, and nothing
    in the description says which one the kernel should read as K. What keeps
    that from mattering is the TMA rule: the contiguous extent must divide
    ``128 // bits``, whose smallest value is 4, and 1 divides none of them -- so
    a unit contiguous extent never reaches a launch. This asserts the property
    that argument implies rather than the argument: every degenerate shape is
    either refused or agrees with the backend.
    """
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    torch.manual_seed(0)
    a = torch.randn(batch, m, k, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(batch, n, k, dtype=torch.bfloat16, device="cuda")

    got = {}
    for want_frost in (False, True):
        c = torch.zeros(batch, m, n, dtype=torch.bfloat16, device="cuda")
        try:
            # None = no plan of that kind. For FROST that IS a refusal, and the
            # one every K == 1 shape takes: the graph-time gate applies the same
            # TMA rule, so the degenerate contiguous extent never gets a plan.
            ran = _matmul_on(batch, m, n, k, want_frost, a, b, c)
            got[want_frost] = "refused" if ran is None else ran
        except (ValueError, NotImplementedError, cudnn.cudnnGraphNotSupportedError):
            got[want_frost] = "refused"
    if got[False] == "refused":
        pytest.skip("the backend has nothing to compare against for this shape")
    if got[True] == "refused":
        return  # refusing is always allowed; computing something else is not
    torch.testing.assert_close(got[True].float(), got[False].float(), atol=2e-1, rtol=2e-2)


@requires_sm100
def test_an_operand_reporting_the_declaration_agrees_with_the_backend(monkeypatch):
    """At N == K the two axis orders are the SAME shape and the SAME stride.

    The graph declares B ``[batch, K, N]`` k-major, stride ``[K*N, 1, K]``; this
    engine's direct-call API takes ``(batch, N, K)``, stride ``(K*N, 1, N)``.
    Identical tuples when N == K, so the description cannot say which the caller
    meant -- and the backend does not ask, it computes from the descriptor. A
    python plan has to reach the same answer, because the caller does not choose
    which plan the heuristics land on.
    """
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    torch.manual_seed(0)
    d = 128
    a = torch.randn(1, d, d, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(1, d, d, dtype=torch.bfloat16, device="cuda").transpose(1, 2)
    assert (tuple(b.shape), tuple(b.stride())) == ((1, d, d), (d * d, 1, d))  # == the declaration

    out = {}
    for want_frost in (False, True):
        g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
        A = g.tensor(name="A", dim=[1, d, d], stride=[d * d, d, 1])
        B = g.tensor(name="B", dim=[1, d, d], stride=[d * d, 1, d])
        C = g.matmul(A=A, B=B, name="mm")
        C.set_output(True).set_data_type(BF16)
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        hits = [i for i, p in enumerate(g.plans) if is_python_engine(p.engine_id) == want_frost]
        if not hits:
            pytest.skip("no plan of the requested kind for this graph")
        g.select_plan(hits[0])
        g.check_support()
        g.build_plans()
        c = torch.zeros(1, d, d, dtype=torch.bfloat16, device="cuda")
        ws = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
        g.execute({A: a, B: b, C: c}, ws)
        torch.cuda.synchronize()
        out[want_frost] = c
    torch.testing.assert_close(out[True].float(), out[False].float(), atol=2e-1, rtol=2e-2)


@requires_sm100
@pytest.mark.parametrize("n,k", ((256, 128), (128, 128)), ids=("n!=k", "n==k"))
def test_bare_addresses_run_at_a_smaller_live_shape(n, k):
    """A bare address wears the graph's layout AND the graph's allocation size.

    ``override_shapes`` then says the call runs a smaller problem inside it, so
    the live shape matches neither the declaration nor a caller's buffer. The
    backend takes this; a python plan that inferred the axis order from the
    shape refused it.
    """
    MB, NB, KB = 256, n, 128
    m, live_k = 128, k // 2
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    A = g.tensor(name="A", uid=1, dim=[1, MB, KB], stride=[MB * KB, KB, 1])
    B = g.tensor(name="B", uid=2, dim=[1, KB, NB], stride=[KB * NB, 1, KB])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(BF16).set_uid(3)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    frost = [i for i, p in enumerate(g.plans) if is_python_engine(p.engine_id)]
    if not frost:
        pytest.skip("no FROST plan for this graph")
    g.select_plan(frost[0])
    g.check_support()
    g.build_plans()

    a = torch.randn(1, MB, KB, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(1, NB, KB, dtype=torch.bfloat16, device="cuda")
    c = torch.zeros(1, MB, NB, dtype=torch.bfloat16, device="cuda")
    ws = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    g.execute(
        {1: a.data_ptr(), 2: b.data_ptr(), 3: c.data_ptr()},
        ws,
        override_uids=[1, 2, 3],
        override_shapes=[[1, m, live_k], [1, live_k, NB], [1, m, NB]],
        override_strides=[[MB * KB, KB, 1], [KB * NB, 1, KB], [MB * NB, NB, 1]],
    )
    torch.cuda.synchronize()
    ref = torch.einsum("bmk,bnk->bmn", a[:, :m, :live_k].float(), b[:, :, :live_k].float())
    torch.testing.assert_close(c[:, :m].float(), ref, atol=2e-1, rtol=2e-2)


@requires_sm100
def test_operand_batch_is_checked():
    """The graph pins each operand's batch, and a launch that ignored it read
    one batch of A against three of B."""
    compiled = jit_from_cudnn_graph(_plain_graph(batch=2))
    a, b, c = _operands(batch=2)
    one_batch_b = b[:1].contiguous()
    for run in (compiled.lowered, compiled.launch):
        if run is None:
            continue
        with pytest.raises(ValueError):
            run(_bound_buffers(compiled, a, one_batch_b, c), stream=None)
