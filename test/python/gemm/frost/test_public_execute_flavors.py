# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every gemm flavor, driven through the PUBLIC ``graph.execute()``.

The rest of this directory calls the compiled object directly with torch
tensors. That skips the whole engine wrapper -- operand binding, the variant
pack, and the buffer conversion ``execute()`` performs -- so a break that only
appears when the engine hands the kernel what the pack holds is invisible to
it. A reduction output shipped in exactly that state: the epilogue initializes
it with ``fill_`` and finalizes ``norm2`` with ``sqrt_``, both of which the
compiled object used to receive as torch tensors and now does not.

So this file exists to exercise the same flavors the direct-call tests cover,
but through the entry point a user actually has.
"""

import pytest
import torch

import cudnn
from cudnn.engines import is_python_engine

pytestmark = pytest.mark.L0

_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the FROST gemm engine claims SM100",
)

BF16 = cudnn.data_type.BFLOAT16
F32 = cudnn.data_type.FLOAT
M = N = 128
K = 64


@pytest.fixture(autouse=True)
def _frost_opt_in(monkeypatch):
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")


def _pin_frost(g):
    """Select the FROST plan, or skip when it does not claim this graph."""
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    frost = [i for i, p in enumerate(g.plans) if is_python_engine(p.engine_id)]
    if not frost:
        pytest.skip("no FROST plan for this graph")
    g.select_plan(frost[0])
    g.check_support()
    g.build_plans()
    return g


def _operands():
    a = torch.randn(1, M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(1, N, K, dtype=torch.bfloat16, device="cuda")
    return a, b, torch.einsum("bmk,bnk->bmn", a.float(), b.float())


def _run(g, data):
    ws = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    g.execute(data, ws)
    torch.cuda.synchronize()


@_GPU
def test_plain_matmul():
    """The control: if this fails the harness is wrong, not the flavor."""
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(BF16)
    _pin_frost(g)

    a, b, ref = _operands()
    c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    _run(g, {A: a, B: b, C: c})
    torch.testing.assert_close(c, ref.to(torch.bfloat16), atol=1e-1, rtol=1e-2)


@_GPU
def test_epilogue_fusion():
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    Y = g.relu(input=C, name="relu")
    Y.set_output(True).set_data_type(BF16)
    _pin_frost(g)

    a, b, ref = _operands()
    y = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    _run(g, {A: a, B: b, Y: y})
    torch.testing.assert_close(y, torch.relu(ref).to(torch.bfloat16), atol=1e-1, rtol=1e-2)


@_GPU
def test_aux_tensor():
    """An aux operand reaches the kernel reshaped to its fake's rank, not permuted."""
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    bias = g.tensor(name="bias", dim=[1, 1, N], stride=[N, N, 1], data_type=F32)
    Y = g.add(a=g.matmul(A=A, B=B, name="mm"), b=bias, name="bias_add")
    Y.set_output(True).set_data_type(BF16)
    _pin_frost(g)

    a, b, ref = _operands()
    bias_buf = torch.randn(1, 1, N, dtype=torch.float32, device="cuda")
    y = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    _run(g, {A: a, B: b, bias: bias_buf, Y: y})
    torch.testing.assert_close(y, (ref + bias_buf).to(torch.bfloat16), atol=1e-1, rtol=1e-2)


@_GPU
@pytest.mark.parametrize("buffer_shape", ((1, 1, N), (1, N), (N,)), ids=("rank3", "rank2", "rank1"))
def test_aux_buffer_rank_need_not_match_the_declaration(buffer_shape):
    """A bias declared ``[1, 1, N]`` may be handed over as ``[N]``.

    Same rule as an operand's axis order: the descriptor defines the tensor and
    the pack supplies only a pointer, so the backend takes any rank here and a
    python plan must reach the same answer. The reshape that makes the kernel's
    wrapper agree read the rank off an ``ndim`` attribute with a default, and the
    pack's buffers carry no ``ndim`` -- so every operand arriving through
    ``execute()`` looked like it already matched, and these were refused.
    """
    out = {}
    for want_frost in (False, True):
        g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
        A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
        B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
        bias = g.tensor(name="bias", dim=[1, 1, N], stride=[N, N, 1], data_type=F32)
        Y = g.add(a=g.matmul(A=A, B=B, name="mm"), b=bias, name="bias_add")
        Y.set_output(True).set_data_type(BF16)
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        hits = [i for i, p in enumerate(g.plans) if is_python_engine(p.engine_id) == want_frost]
        if not hits:
            pytest.skip("no plan of the requested kind for this graph")
        g.select_plan(hits[0])
        g.check_support()
        g.build_plans()

        torch.manual_seed(0)
        a = torch.randn(1, M, K, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(1, N, K, dtype=torch.bfloat16, device="cuda")
        bias_buf = torch.randn(buffer_shape, dtype=torch.float32, device="cuda")
        y = torch.zeros(1, M, N, dtype=torch.bfloat16, device="cuda")
        _run(g, {A: a, B: b, bias: bias_buf, Y: y})
        out[want_frost] = y
    torch.testing.assert_close(out[True], out[False], atol=1e-1, rtol=1e-2)


@_GPU
def test_two_dense_outputs():
    """Two stored outputs: each contributes its own stride triple, in order."""
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(BF16)
    Y = g.relu(input=C, name="relu")
    Y.set_output(True).set_data_type(BF16)
    _pin_frost(g)

    a, b, ref = _operands()
    c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    y = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    _run(g, {A: a, B: b, C: c, Y: y})
    torch.testing.assert_close(c, ref.to(torch.bfloat16), atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(y, torch.relu(ref).to(torch.bfloat16), atol=1e-1, rtol=1e-2)


@_GPU
def test_multi_gemm():
    """Two matmuls sharing A and an epilogue: the operands the kernel takes are
    the DISTINCT ones, which is what the pack binds."""
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B0 = g.tensor(name="B0", dim=[1, K, N], stride=[K * N, 1, K])
    B1 = g.tensor(name="B1", dim=[1, K, N], stride=[K * N, 1, K])
    Y = g.add(a=g.matmul(A=A, B=B0, name="mm0"), b=g.matmul(A=A, B=B1, name="mm1"), name="sum")
    Y.set_output(True).set_data_type(BF16)
    _pin_frost(g)

    a = torch.randn(1, M, K, dtype=torch.bfloat16, device="cuda")
    b0 = torch.randn(1, N, K, dtype=torch.bfloat16, device="cuda")
    b1 = torch.randn(1, N, K, dtype=torch.bfloat16, device="cuda")
    y = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    _run(g, {A: a, B0: b0, B1: b1, Y: y})
    ref = torch.einsum("bmk,bnk->bmn", a.float(), b0.float()) + torch.einsum("bmk,bnk->bmn", a.float(), b1.float())
    torch.testing.assert_close(y, ref.to(torch.bfloat16), atol=2e-1, rtol=2e-2)


@_GPU
@pytest.mark.parametrize(
    "mode,reference",
    (
        (cudnn.reduction_mode.ADD, lambda t: t.sum().reshape(1, 1, 1)),
        (cudnn.reduction_mode.AMAX, lambda t: t.abs().max().reshape(1, 1, 1)),
        (cudnn.reduction_mode.MAX, lambda t: t.max().reshape(1, 1, 1)),
        (cudnn.reduction_mode.MIN, lambda t: t.min().reshape(1, 1, 1)),
    ),
    ids=("add", "amax", "max", "min"),
)
def test_reduction_output(mode, reference):
    """A reduction tap alongside the epilogue output.

    The epilogue writes the tap's initial value before the kernel runs and, for
    ``norm2``, takes a square root after it. Both are device operations on a
    caller buffer, which is where a buffer that is only a description rather
    than a tensor shows up.
    """
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    Y = g.relu(input=C, name="relu")
    Y.set_output(True).set_data_type(BF16)
    R = g.reduction(input=Y, mode=mode, name="red")
    R.set_dim([1, 1, 1]).set_stride([1, 1, 1])
    R.set_output(True).set_data_type(F32)
    _pin_frost(g)

    a, b, ref = _operands()
    y = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    r = torch.empty(1, 1, 1, dtype=torch.float32, device="cuda")
    _run(g, {A: a, B: b, Y: y, R: r})

    relu = torch.relu(ref)
    torch.testing.assert_close(y, relu.to(torch.bfloat16), atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(r, reference(relu), atol=1e-1, rtol=1e-2)


@_GPU
def test_int32_reduction_seed_is_packed_as_int32():
    """A memset moves bits, so the identity has to be packed as the dtype.

    int32's identities are the ends of its range, which is exactly where the
    difference shows: -2**31 packed as float is 0xcf000000, and a MAX reduction
    seeded with that returns -822083584 for any input below it.
    """
    I32 = cudnn.data_type.INT32
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    bias = g.tensor(name="bias", dim=[1, 1, N], stride=[N, N, 1], data_type=I32)
    C = g.matmul(A=A, B=B, name="mm")
    Y = g.add(a=C, b=bias, name="add_i32", compute_data_type=I32)
    Y.set_output(True).set_data_type(I32)
    R = g.reduction(input=Y, mode=cudnn.reduction_mode.MAX, name="red", compute_data_type=I32)
    R.set_dim([1, 1, 1]).set_stride([1, 1, 1]).set_output(True).set_data_type(I32)
    _pin_frost(g)

    floor = -2_000_000_000  # below the float-packed seed, above int32's minimum
    a = torch.zeros(1, M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.zeros(1, N, K, dtype=torch.bfloat16, device="cuda")
    y = torch.empty(1, M, N, dtype=torch.int32, device="cuda")
    r = torch.empty(1, 1, 1, dtype=torch.int32, device="cuda")
    _run(g, {A: a, B: b, bias: torch.full((1, 1, N), floor, dtype=torch.int32, device="cuda"), Y: y, R: r})
    assert int(r.item()) == floor


@_GPU
def test_block_scale_matmul():
    """nvfp4 + F8_128x4 scale blobs, through the public entry point.

    The scale factors are the operands the recipe treats least like the others:
    they ride in the launch argument list but not in ``problem_size``, and their
    size is re-synthesized from M/N/K rather than read off the buffer, so a blob
    that is not one dense byte run is read out of bounds with no fault.
    """
    from gemm_test_utils import E2M1, to_blocked, unpack_fp4

    bs_m = bs_n = 128
    bs_k, block = 256, 16
    sf_k = bs_k // block
    fp4, fp8 = cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E4M3
    reorder = dict(reordering_type=cudnn.tensor_reordering.F8_128x4)

    g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=F32, compute_data_type=F32)
    A = g.tensor(name="A", dim=[1, bs_m, bs_k], stride=[bs_m * bs_k, bs_k, 1], data_type=fp4)
    B = g.tensor(name="B", dim=[1, bs_k, bs_n], stride=[bs_k * bs_n, 1, bs_k], data_type=fp4)
    SFA = g.tensor(name="SFA", dim=[1, bs_m, sf_k], stride=[bs_m * sf_k, sf_k, 1], data_type=fp8, **reorder)
    SFB = g.tensor(name="SFB", dim=[1, sf_k, bs_n], stride=[sf_k * bs_n, 1, sf_k], data_type=fp8, **reorder)
    C = g.matmul(
        A=g.block_scale_dequantize(input=A, descale=SFA, block_size=[1, block]),
        B=g.block_scale_dequantize(input=B, descale=SFB, block_size=[block, 1]),
        name="mm",
    )
    C.set_output(True).set_data_type(cudnn.data_type.HALF)
    _pin_frost(g)
    assert g._compiled_plans[g._plan_index]._lowered is not None

    torch.manual_seed(0)
    lut = torch.tensor(E2M1, dtype=torch.float32, device="cuda")
    a_u8 = torch.randint(0, 256, (1, bs_m, bs_k // 2), dtype=torch.uint8, device="cuda")
    b_u8 = torch.randint(0, 256, (1, bs_n, bs_k // 2), dtype=torch.uint8, device="cuda")
    sfa = torch.randint(1, 4, (bs_m, sf_k), device="cuda").to(torch.float8_e4m3fn)
    sfb = torch.randint(1, 4, (bs_n, sf_k), device="cuda").to(torch.float8_e4m3fn)
    c = torch.zeros(1, bs_m, bs_n, dtype=torch.float16, device="cuda")
    _run(
        g,
        {
            A: a_u8.view(torch.float4_e2m1fn_x2),
            B: b_u8.view(torch.float4_e2m1fn_x2),
            SFA: to_blocked(sfa).view(1, bs_m, sf_k),
            SFB: to_blocked(sfb).view(1, bs_n, sf_k),
            C: c,
        },
    )
    a_s = unpack_fp4(a_u8, lut).view(bs_m, bs_k) * sfa.float().repeat_interleave(block, 1)
    b_s = unpack_fp4(b_u8, lut).view(bs_n, bs_k) * sfb.float().repeat_interleave(block, 1)
    torch.testing.assert_close(c.float(), (a_s @ b_s.t()).reshape(1, bs_m, bs_n), atol=2e-1, rtol=2e-2)


@_GPU
def test_norm2_reduction_is_refused_at_build():
    """``norm2`` is the one reduction mode with a post-kernel finalize.

    It never reaches one: the backend refuses the reduction descriptor while
    the graph is being lowered, so no plan exists and ``execute()`` is never
    called. Recorded because the finalize would otherwise look like a live path
    that needs a device-side square root.
    """
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    Y = g.relu(input=C, name="relu")
    Y.set_output(True).set_data_type(BF16)
    R = g.reduction(input=Y, mode=cudnn.reduction_mode.NORM2, name="red")
    R.set_dim([1, 1, 1]).set_stride([1, 1, 1])
    R.set_output(True).set_data_type(F32)
    g.validate()
    with pytest.raises(RuntimeError, match="NOT_SUPPORTED"):
        g.build_operation_graph()


@_GPU
def test_bare_address_operands():
    """The backend has always taken a raw device address; so must a python plan.

    The operand a bare address describes is the GRAPH's declaration, which
    orders a matmul's B ``[batch, K, N]`` where a caller allocates it
    ``(batch, N, K)``. Reading an extent by axis position answers one of those
    and not the other, so the recipe records which axis carries M/N/K instead --
    and re-labelling one into the other is a permute, which is why this stays on
    the fast path rather than costing an interpreted pass for being legal.
    """
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    A = g.tensor(name="A", uid=1, dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", uid=2, dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(BF16).set_uid(3)
    _pin_frost(g)

    a, b, ref = _operands()
    c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    _run(g, {1: a.data_ptr(), 2: b.data_ptr(), 3: c.data_ptr()})
    torch.testing.assert_close(c, ref.to(torch.bfloat16), atol=1e-1, rtol=1e-2)
    assert dict(g._compiled_plans[g._plan_index]._compiled.deferrals) == {}


@_GPU
def test_bare_address_workspace():
    """A workspace passed as a raw address has no measurable size.

    Zero means "the pack could not measure it", not "empty" -- the backend
    takes a raw workspace pointer without checking either, so an engine that
    needs scratch must not refuse one. This drives the unknown-capacity path
    through both `Workspace.over` and the C carve's bounds check.
    """
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(BF16)
    _pin_frost(g)

    a, b, ref = _operands()
    c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    ws = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    g.execute({A: a, B: b, C: c}, ws.data_ptr())
    torch.cuda.synchronize()
    torch.testing.assert_close(c, ref.to(torch.bfloat16), atol=1e-1, rtol=1e-2)


@_GPU
def test_undersized_workspace_still_rejected():
    """A workspace whose size IS known is still bounds-checked."""
    from cudnn.engines.base import VariantPack
    from cudnn.frost.workspace import Workspace

    tiny = torch.empty(16, dtype=torch.uint8, device="cuda")
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    A = g.tensor(name="A", uid=1, dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", uid=2, dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(BF16).set_uid(3)
    _pin_frost(g)

    a, b, _ = _operands()
    c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    pack = g._normalize(g._uid_to_data({1: a, 2: b, 3: c}), tiny)
    assert pack.workspace_bytes == 16
    with pytest.raises(ValueError, match=r"needs a .*-byte workspace"):
        Workspace.over(pack, 4096, "probe")


@_GPU
def test_unknown_size_workspace_refuses_to_measure_its_tail():
    """``remaining()`` cannot answer for a workspace whose size is unknown."""
    from cudnn.frost.workspace import Workspace

    ws = torch.empty(4096, dtype=torch.uint8, device="cuda")
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    A = g.tensor(name="A", uid=1, dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", uid=2, dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(BF16).set_uid(3)
    _pin_frost(g)

    a, b, _ = _operands()
    c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    pack = g._normalize(g._uid_to_data({1: a, 2: b, 3: c}), ws.data_ptr())
    assert pack.workspace_bytes == 0
    carver = Workspace.over(pack, 1024, "probe")
    with pytest.raises(ValueError, match="size is unknown"):
        carver.remaining()


# --- seeding a padded reduction output, which the engine owns ----------------


@pytest.mark.parametrize(
    "shape,stride,want",
    [
        ((1, 8, 4), (32, 4, 1), [(32, 1)]),  # dense whatever rank it was declared at
        ((1, 8, 1), (32, 4, 1), [(8, 4)]),  # a per-row scalar: one strided run
        ((2, 8, 4), (64, 8, 1), [(16, 8), (4, 1)]),  # padded rows: batch merges into the row count
        ((1, 1, 1), (1, 1, 1), []),  # one element
    ],
)
def test_a_layout_collapses_to_the_runs_a_memset_can_cover(shape, stride, want):
    """Unit axes carry no elements and adjacent dense axes are one run.

    Collapsing first is what keeps the seed to a single memset for a contiguous
    tap and one per batch for a padded one, rather than one per row.
    """
    from cudnn.frost.buffers import collapse_layout

    assert collapse_layout(shape, stride) == want


@pytest.mark.parametrize(
    "shape,stride",
    [
        ((1, 8, 1), (0, 0, 1)),  # stride 0 over a real extent
        ((1, 4, 4), (16, 2, 1)),  # rows closer together than they are wide
        ((1, 2, 2), (4, 2, 2)),  # two axes landing on the same elements
    ],
)
def test_a_reduction_output_that_writes_an_element_twice_is_rejected(shape, stride):
    """Two elements at one address is a write race, not a layout to support.

    The rule is that each axis clears the whole span of the one below it.
    Checking the innermost pair alone -- pitch against width -- passes the last
    case, whose width is 1 and whose two axes both land on element 2.
    """
    from cudnn.frost.buffers import fill_word_strided_async, strided_fill_plan

    assert strided_fill_plan(shape, stride) is None
    with pytest.raises(ValueError, match="twice"):
        fill_word_strided_async(0, shape, stride, 4, 0, None)


@_GPU
@pytest.mark.parametrize("shape,stride", [((1, 8, 1), (32, 4, 1)), ((2, 8, 4), (64, 8, 1)), ((3, 5, 1), (7, 1, 1))])
def test_a_padded_seed_writes_its_own_elements_and_no_others(shape, stride):
    """The engine seeds a padded output itself, without the caller's ``fill_()``.

    Borrowing that method worked only while the buffer happened to be a torch
    tensor -- and queued on torch's stream, not the one the kernel will run on.
    What it has to get right is exactly this: every element the view covers, and
    nothing between them.
    """
    from cudnn.frost.buffers import fill_word_strided_async, init_word

    span = 1 + sum((d - 1) * s for d, s in zip(shape, stride))
    flat = torch.zeros(span, dtype=torch.float32, device="cuda")
    view = torch.as_strided(flat, shape, stride)
    fill_word_strided_async(flat.data_ptr(), shape, stride, 4, init_word("fp32", 3.5), None)
    torch.cuda.synchronize()

    expected = torch.zeros(span, dtype=torch.float32, device="cuda")
    torch.as_strided(expected, shape, stride).fill_(3.5)
    assert torch.equal(flat, expected)
    assert torch.equal(view, torch.full(shape, 3.5, device="cuda"))


@_GPU
def test_a_padded_reduction_output_is_seeded_on_the_kernel_s_stream():
    """The case that used to reach ``tensor.fill_()``.

    A tap declared into a padded buffer is legal and goes through the public
    path like any other. Seeding it by calling ``fill_()`` on the caller's
    tensor queued on torch's current stream rather than the one the kernel runs
    on -- the same stream only by luck -- and only worked at all while that
    buffer was a torch tensor. It is the driver's 2D memset now, so this asserts
    both halves: the tap is right, and the padding it does not own is untouched.
    """
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    Y = g.relu(input=C, name="relu")
    Y.set_output(True).set_data_type(BF16)
    R = g.reduction(input=Y, mode=cudnn.reduction_mode.ADD, name="red")
    R.set_dim([1, M, 1]).set_stride([M * 4, 4, 1]).set_output(True).set_data_type(F32)
    _pin_frost(g)

    a, b, ref = _operands()
    y = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    pad = torch.full((1, M, 4), -7.0, dtype=torch.float32, device="cuda")
    _run(g, {A: a, B: b, Y: y, R: pad[:, :, :1]})

    torch.testing.assert_close(pad[:, :, 0], torch.relu(ref).sum(dim=2), atol=1.0, rtol=2e-2)
    assert torch.equal(pad[:, :, 1:], torch.full((1, M, 3), -7.0, device="cuda"))
