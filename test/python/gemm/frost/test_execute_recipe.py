# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The build-time recipe, and the one call path lowered from it.

``_lower`` captures the recipe into the loop that launches; ``explain`` reads the
same recipe to say what is wrong with a call that loop refused, and runs nothing.
There is no second executor to differential against, and that is deliberate: two
readings of one wrong plan agreeing proves nothing, which is exactly how the
axis-order bug survived a differential that ran both.

What replaces it is the BACKEND -- the two tests below that run a shape on a
cuDNN plan and a FROST plan and require the same numbers -- and the invariant
that makes a rejection meaningful: the set of calls the launch path refuses
should equal the set of illegal calls.

Only one direction of that is cheap to assert. ``deferrals`` must be empty for a
legal call of every flavor, which says nothing legal is refused; a call the fast
path ACCEPTS never reaches the counter, so the other direction is covered
case by case below and by the backend differentials.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from gemm_test_utils import kw, requires_sm100, to_blocked

import cudnn
import cudnn.gemm.frost  # noqa: F401 — installs the cudnn.pygraph recorder hook
from cudnn.engines import is_python_engine
from cudnn.gemm.frost.compiler import jit_from_cudnn_graph, probe_supported
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


def _shared_operand_graph(d=K):
    """``matmul(A, A)``: one tensor in two roles, so both bind ONE pack slot.

    At d x d the same buffer is a legal A (k-major) and a legal B (n-major), and
    the graph declares it once -- which is exactly the shape where re-labelling
    the SLOT into B's axis order also re-labels what A reads.
    """
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    A = g.tensor(name="A", dim=[1, d, d], stride=[d * d, d, 1])
    C = g.matmul(A=A, B=A, name="mm")
    C.set_output(True).set_data_type(BF16)
    return g


def _row_reduction_graph(pad=4):
    """A per-row tap, declared into a padded buffer when ``pad`` > 1."""
    g = _plain_graph()
    Y = [t for t in g._nodes[-1].outputs.values()][0]
    R = g.reduction(input=Y, mode=cudnn.reduction_mode.ADD, name="red")
    R.set_dim([1, M, 1]).set_stride([M * pad, pad, 1]).set_output(True).set_data_type(F32)
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


def _multi_gemm_graph(dense_output=True):
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B0 = g.tensor(name="B0", dim=[1, K, N], stride=[K * N, 1, K])
    B1 = g.tensor(name="B1", dim=[1, K, N], stride=[K * N, 1, K])
    Y = g.add(a=g.matmul(A=A, B=B0, name="mm0"), b=g.matmul(A=A, B=B1, name="mm1"), name="sum")
    Y.set_data_type(BF16).set_output(dense_output)
    if not dense_output:
        R = g.reduction(input=Y, mode=cudnn.reduction_mode.ADD, name="red")
        R.set_dim([1, 1, 1]).set_stride([1, 1, 1]).set_output(True).set_data_type(F32)
    return g


def _multi_gemm_reduction_only_graph():
    """Two matmuls whose only output is a tap: the batch comes off the operands."""
    return _multi_gemm_graph(dense_output=False)


# Each entry is (build, how many distinct B operands, how many outputs, how many aux).
_FLAVORS = (
    (_plain_graph, 1, 1, 0),
    (_epilogue_graph, 1, 1, 0),
    (_aux_graph, 1, 1, 1),
    (_two_output_graph, 1, 2, 0),
    (_reduction_graph, 1, 2, 0),
    (_multi_gemm_graph, 2, 1, 0),
    (_multi_gemm_reduction_only_graph, 2, 1, 0),
)
_FLAVOR_IDS = ("plain", "epilogue", "aux", "two_outputs", "reduction", "multi_gemm", "multi_gemm_tap_only")


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
@pytest.mark.parametrize("build,n_b,n_out,n_aux", _FLAVORS, ids=_FLAVOR_IDS)
def test_a_legal_call_of_every_flavor_is_refused_for_nothing(build, n_b, n_out, n_aux):
    """The invariant, per flavor: refusing a legal call is a bug, not a detour.

    While there was an interpreter behind it, a fast path that gave up on a
    legal call cost time and nothing else, so nothing asserted it did not. This
    is what that assertion looks like: every reason the launch path can refuse
    is counted, and a legal call of every flavor must trigger none of them.
    """
    compiled = jit_from_cudnn_graph(build())
    if compiled.lowered is None:
        pytest.skip(f"this build does not lower: {compiled.declined}")
    operands = _buffers_for(compiled)
    for o in compiled.recipe.outputs:
        operands[o.index].zero_()
    compiled.lowered(operands, stream=None)
    torch.cuda.synchronize()
    assert operands[compiled.recipe.outputs[0].index].abs().sum() > 0
    assert dict(compiled.deferrals) == {}


@requires_sm100
def test_a_padded_reduction_output_stays_on_the_fast_path():
    """A tap declared into a padded buffer is a legal call, not a slow one.

    The fast path could seed exactly one dense run, so it handed a padded tap
    to the interpreter -- which seeded it by calling ``fill_()`` on whatever the
    caller passed. Both are gone: the seed is the driver's 2D memset, and it
    must touch every element the view covers and nothing between them.
    """
    compiled = jit_from_cudnn_graph(_row_reduction_graph())
    if compiled.lowered is None:
        pytest.skip(f"this build does not lower: {compiled.declined}")
    operands = _buffers_for(compiled)
    tap = [o for o in compiled.recipe.outputs if o.init is not None][0]
    pad = torch.full((1, M, 4), -1.0, dtype=torch.float32, device="cuda")
    operands[tap.index] = pad[:, :, :1]

    compiled.lowered(operands, stream=None)
    torch.cuda.synchronize()
    assert dict(compiled.deferrals) == {}
    a, b = (operands[op.index] for op in compiled.recipe.inputs)
    torch.testing.assert_close(pad[:, :, 0], torch.einsum("bmk,bnk->bm", a.float(), b.float()), atol=1.0, rtol=2e-2)
    assert torch.equal(pad[:, :, 1:], torch.full((1, M, 3), -1.0, device="cuda"))


@requires_sm100
def test_every_decline_names_its_reason():
    """A plan without a fast path says which rule denied it, not just ``None``."""
    compiled = jit_from_cudnn_graph(_plain_graph())
    assert compiled.lowered is not None and compiled.declined is None
    compiled.recipe = replace(compiled.recipe, workspace_bytes=4096)
    assert compiled._lower() is None
    assert compiled.declined == "needs workspace"


@requires_sm100
def test_a_deferral_is_counted_under_the_rule_that_caused_it():
    """Only an ILLEGAL call leaves the fast path, and it says which rule sent it."""
    compiled = jit_from_cudnn_graph(_plain_graph())
    if compiled.lowered is None:
        pytest.skip("this build does not lower")
    a, b, c = _wrong_major()
    operands = _bound_buffers(compiled, a, b, c)
    with pytest.raises(ValueError):
        compiled.lowered(operands, stream=None)
    assert dict(compiled.deferrals) == {"input layout": 1}


@requires_sm100
@pytest.mark.parametrize("build,n_b,n_out,n_aux", _FLAVORS, ids=_FLAVOR_IDS)
def test_an_operand_in_the_graph_s_axis_order_stays_on_the_fast_path(build, n_b, n_out, n_aux):
    """A bare device address wears the graph's layout, and that is legal.

    cuDNN declares a matmul's B ``[batch, K, N]`` where this kernel reads
    ``(batch, N, K)``, so an operand the pack described FROM the graph -- a bare
    address has no geometry of its own -- arrives with its axes the other way
    round. Re-labelling one is a permute the recipe already knows the shape of,
    so it is not a reason to leave the fast path; it used to be, and a legal
    call paid a whole interpreted pass for it.
    """
    compiled = jit_from_cudnn_graph(build())
    if compiled.lowered is None:
        pytest.skip(f"this build does not lower: {compiled.declined}")
    operands = _buffers_for(compiled)
    outs = compiled.recipe.outputs
    borrowed = tuple(True for _ in operands)

    runs = []
    for order in (None, borrowed):
        for o in outs:
            operands[o.index].zero_()
        # Same buffers either way: what changes is which axis order they claim,
        # and the graph's is the one the declaration already describes them in.
        compiled.lowered(_as_declared(compiled, operands) if order else operands, order, stream=None)
        torch.cuda.synchronize()
        runs.append([operands[o.index].clone() for o in outs])
    assert dict(compiled.deferrals) == {}
    for o, own, graph in zip(outs, *runs):
        exact = o.init is None
        torch.testing.assert_close(own, graph, atol=0 if exact else 1e-2, rtol=0 if exact else 1e-5)


@requires_sm100
def test_a_post_kernel_finalize_is_refused_before_a_plan_exists():
    """``norm2`` takes a square root through the caller's buffer after the kernel.

    That is a device operation this engine does not own, and it has one call
    path -- so the GRAPH is declined and goes to the backend, rather than being
    compiled into a kernel only a second executor could run. Which executor a
    graph needs is not a question this engine wants to be able to ask.
    """
    g = _plain_graph()
    Y = [t for t in g._nodes[-1].outputs.values()][0]
    R = g.reduction(input=Y, mode=cudnn.reduction_mode.NORM2, name="red")
    R.set_dim([1, 1, 1]).set_stride([1, 1, 1]).set_output(True).set_data_type(F32)
    with pytest.raises(NotImplementedError, match="square root"):
        probe_supported(g)


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


def _bind(compiled, a_bufs, b_bufs, out_bufs, aux_bufs=(), sfa_bufs=(), sfb_bufs=()):
    """The operand list in bound-tensor order, which is what both paths take."""
    bd = compiled.binding
    pack = {}
    for role, bufs in (
        (bd.a_operands, a_bufs),
        (bd.b_operands, b_bufs),
        (bd.sfa_operands, sfa_bufs),
        (bd.sfb_operands, sfb_bufs),
        (bd.outputs, out_bufs),
        (bd.aux, aux_bufs),
    ):
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


def _as_declared(compiled, operands):
    """The same memory, each input re-labelled the way the graph declares it.

    What the pack hands over for a bare address: with no geometry of its own,
    the declaration IS the description, so B arrives ``[batch, K, N]``.
    """
    out = list(operands)
    for op in compiled.recipe.inputs:
        inverse = [0, 0, 0]
        for role, axis in enumerate(op.declared):
            inverse[axis] = role
        out[op.index] = operands[op.index].permute(*inverse)
    return out


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
    "case,verdict",
    (
        (_good, "ran"),
        (_padded_rows, "ran"),  # the outer stride is free
        (_short_k, "rejected"),
        (_wrong_major, "rejected"),
        (_wrong_output_shape, "rejected"),
        (_misaligned_output, "rejected"),
        (_misaligned_a, "rejected"),
    ),
    ids=lambda x: x if isinstance(x, str) else x.__name__.strip("_"),
)
def test_the_launch_path_accepts_exactly_the_legal_calls(case, verdict):
    """The invariant the second executor used to make untestable.

    While ``lowered`` could hand anything it was unsure of to an interpreter
    that ran it anyway, "the fast path rejects a legal call" was a performance
    bug at worst and nothing asserted otherwise. There is one path now, so a
    rejection IS the answer -- which makes both directions real failures:
    refusing a legal call breaks it, and accepting an illegal one runs a kernel
    over memory the caller did not describe.

    ``deferrals`` pins down the first direction only, and this table is how the
    second is covered: each illegal case is named and must be refused. A
    rejection with nothing counted would be drift between the guards and the
    diagnostics, which ``explain`` raises on separately.
    """
    compiled = jit_from_cudnn_graph(_plain_graph())
    if compiled.lowered is None:
        pytest.skip(f"this build does not lower: {compiled.declined}")

    a, b, c = case()
    got, out = _verdict(compiled.lowered, _bound_buffers(compiled, a, b, c), c)
    assert got == verdict
    assert bool(compiled.deferrals) == (verdict == "rejected")
    if verdict == "ran":
        torch.testing.assert_close(out.float(), torch.einsum("bmk,bnk->bmn", a.float(), b.float()), atol=2e-1, rtol=2e-2)


@requires_sm100
def test_one_buffer_in_two_roles_reads_each_role_s_own_axis_order():
    """``matmul(A, A)`` binds ONE slot as both operands.

    The two roles read that memory through different axis maps -- A as
    ``[b, m, k]``, B as the graph's ``[b, k, n]`` -- so the launch path carries
    a view per ROLE. Re-labelling the slot instead is cheaper and silently hands
    the other role a transposed view: the numbers come out as ``A @ A`` where
    the graph says ``A @ A.T``, and no rule fires, so the checker then finds
    nothing wrong and raises the drift error instead.
    """
    compiled = jit_from_cudnn_graph(_shared_operand_graph())
    if compiled.lowered is None:
        pytest.skip(f"this build does not lower: {compiled.declined}")
    assert len({op.index for op in compiled.recipe.inputs}) == 1  # one slot, two roles

    a = torch.randn(1, K, K, dtype=torch.bfloat16, device="cuda")
    c = torch.zeros(1, K, K, dtype=torch.bfloat16, device="cuda")
    compiled.lowered(_bind(compiled, [a], [a], [c]), stream=None)
    torch.cuda.synchronize()
    assert dict(compiled.deferrals) == {}
    # The graph declares B [b, K, N], so B's N axis is the buffer's LAST -- which
    # makes the product A @ A, not A @ A.T. Asymmetric by construction.
    torch.testing.assert_close(c.float(), (a.float() @ a.float()), atol=2e-1, rtol=2e-2)


@requires_sm100
def test_a_reduction_output_of_the_wrong_width_is_refused_before_anything_is_written():
    """The seed is a 32-bit word and the count is the buffer's numel.

    A tap bound to a narrower buffer therefore writes twice the bytes it owns,
    and the launch's own dtype check comes too late to help -- by then the fill
    has already run. So the width is a rule of this call, checked with the
    caller's memory still untouched. The canaries are the assertion: rejecting
    is not enough if it rejects afterwards.
    """
    compiled = jit_from_cudnn_graph(_reduction_graph())
    if compiled.lowered is None:
        pytest.skip(f"this build does not lower: {compiled.declined}")
    operands = _buffers_for(compiled)
    tap = [o for o in compiled.recipe.outputs if o.init is not None][0]
    block = torch.full((16,), -7.0, dtype=torch.float16, device="cuda")
    operands[tap.index] = block[:1].view(1, 1, 1)

    with pytest.raises(ValueError, match="element"):
        compiled.lowered(operands, stream=None)
    torch.cuda.synchronize()
    assert dict(compiled.deferrals) == {"reduction seed dtype": 1}
    assert torch.equal(block, torch.full((16,), -7.0, dtype=torch.float16, device="cuda"))


@requires_sm100
def test_the_checker_is_loud_when_it_cannot_find_the_fault():
    """The one failure mode that writing the rules twice introduces.

    ``lowered``'s guards are fused for speed and ``explain``'s are written for
    the message: two spellings of one set of rules. If they drift, a call gets
    refused and the checker then finds nothing wrong with it -- so a call the
    checker considers legal has to be loud rather than a quiet return, and
    distinct from the ValueError a real rejection raises.
    """
    compiled = jit_from_cudnn_graph(_plain_graph())
    a, b, c = _good()
    with pytest.raises(RuntimeError, match="no rule explains"):
        compiled.explain(_bound_buffers(compiled, a, b, c))


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


BS_M = BS_N = 128
BS_K = 256
BS_BLOCK = 16


def _nvfp4_graph(gemms=1):
    """One or two nvfp4 block-scaled matmuls, sharing A when there are two."""
    sf_k = BS_K // BS_BLOCK
    fp4, fp8 = cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E4M3
    reorder = dict(reordering_type=cudnn.tensor_reordering.F8_128x4)
    g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=F32, compute_data_type=F32)
    A = g.tensor(name="A", dim=[1, BS_M, BS_K], stride=[BS_M * BS_K, BS_K, 1], data_type=fp4)
    SFA = g.tensor(name="SFA", dim=[1, BS_M, sf_k], stride=[BS_M * sf_k, sf_k, 1], data_type=fp8, **reorder)
    Ad = g.block_scale_dequantize(input=A, descale=SFA, block_size=[1, BS_BLOCK])
    products = []
    for i in range(gemms):
        B = g.tensor(name=f"B{i}", dim=[1, BS_K, BS_N], stride=[BS_K * BS_N, 1, BS_K], data_type=fp4)
        SFB = g.tensor(name=f"SFB{i}", dim=[1, sf_k, BS_N], stride=[sf_k * BS_N, 1, sf_k], data_type=fp8, **reorder)
        Bd = g.block_scale_dequantize(input=B, descale=SFB, block_size=[BS_BLOCK, 1])
        products.append(g.matmul(A=Ad, B=Bd, name=f"mm{i}"))
    out = products[0] if gemms == 1 else g.add(a=products[0], b=products[1], name="sum")
    out.set_output(True).set_data_type(cudnn.data_type.HALF)
    return g


def _nvfp4_buffers(compiled):
    """Packed fp4 operands and their F8_128x4 scale blobs, in bound order."""
    sf_k = BS_K // BS_BLOCK
    n_b = len(compiled.binding.b_operands)
    a = torch.randint(0, 256, (1, BS_M, BS_K // 2), dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)
    sfa = to_blocked(torch.randint(1, 4, (BS_M, sf_k), device="cuda").to(torch.float8_e4m3fn)).view(1, BS_M, sf_k)
    bs = [torch.randint(0, 256, (1, BS_N, BS_K // 2), dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2) for _ in range(n_b)]
    sfbs = [to_blocked(torch.randint(1, 4, (BS_N, sf_k), device="cuda").to(torch.float8_e4m3fn)).view(1, BS_N, sf_k) for _ in range(n_b)]
    out = torch.zeros(1, BS_M, BS_N, dtype=torch.float16, device="cuda")
    return _bind(compiled, [a], bs, [out], sfa_bufs=[sfa], sfb_bufs=sfbs), out


@requires_sm100
@pytest.mark.parametrize("gemms", (1, 2), ids=("single", "multi"))
def test_block_scale_lowers_and_runs(gemms):
    """Block scale is the flavor with the most per-call table in it.

    Its scale factors ride in the launch argument list but NOT in
    ``problem_size``, its blob size is re-synthesized from M/N/K rather than read
    off the buffer, and the multi-GEMM form sends one A stride triple for every
    operand instead of one each. Four recipe fields, so it is the flavor most
    likely to reach the launch with an argument list that does not match the
    kernel's signature -- which is what running it here catches. What the
    numbers should BE is checked against torch through the public entry point.
    """
    # Two GEMMs do not fit the auto-selected cta_n=256 in TMEM, so pin a
    # geometry that does; which config the engine picks for one GEMM is the
    # public execute test's job, not this one's.
    cfg = kw("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma") if gemms > 1 else {}
    compiled = jit_from_cudnn_graph(_nvfp4_graph(gemms), **cfg)
    assert compiled.block_scale
    if compiled.lowered is None:
        pytest.skip("this build does not lower (no tvm-ffi front door)")
    assert bool(compiled.recipe.shared_layout) == (gemms > 1)

    operands, out = _nvfp4_buffers(compiled)
    compiled.lowered(operands, stream=None)
    torch.cuda.synchronize()
    assert dict(compiled.deferrals) == {}
    assert out.abs().sum() > 0


@requires_sm100
def test_a_scale_factor_of_the_wrong_rank_is_refused_rather_than_crashing():
    """A scale factor is relabelled like every other head, so its rank is a rule.

    The blob checks -- alignment, one dense run, the size the template
    re-synthesizes -- all pass for a flat rank-1 blob, and it then reached
    ``permute(1, 2, 0)`` in the launch argument list and raised from INSIDE the
    body that is not allowed to raise. Both the guard and the checker name it
    now, so it is a rejection with a reason like any other.
    """
    compiled = jit_from_cudnn_graph(_nvfp4_graph(1))
    if compiled.lowered is None:
        pytest.skip(f"this build does not lower: {compiled.declined}")
    operands, _out = _nvfp4_buffers(compiled)
    sf = compiled.recipe.sf[0]
    operands[sf.index] = operands[sf.index].reshape(-1)

    with pytest.raises(ValueError, match="rank-3"):
        compiled.lowered(operands, stream=None)
    assert dict(compiled.deferrals) == {"scale-factor blob": 1}


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
    if compiled.lowered is None:
        pytest.skip(f"this build does not lower: {compiled.declined}")
    a, b, c = _operands(batch=2)
    one_batch_b = b[:1].contiguous()
    with pytest.raises(ValueError):
        compiled.lowered(_bound_buffers(compiled, a, one_batch_b, c), stream=None)
