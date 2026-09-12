# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""graph.execute() normalizes the variant pack once, and both paths see the same thing.

The property under test is that a caller never has to know whether the
heuristics landed the graph on the cuDNN backend or on a python engine. Before
normalization the backend accepted a bare device address (its `_ptr` had
`if type(d) is int: return d`) while a python engine did not — `resolve_node_buffers`
handed the engine the caller's object untouched and `frost.buffers.probe` then
raised "buffer of type int exposes neither __cuda_array_interface__ nor
__dlpack__". Same call, two answers, and the caller does not pick the plan.
"""

import threading

import pytest
import torch

import cudnn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="execute() needs a device to run a plan on")

M = N = K = 64


def _matmul_graph():
    """A graph the cuDNN backend serves."""
    a = torch.randn(1, M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(1, K, N, dtype=torch.bfloat16, device="cuda")
    c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, compute_data_type=cudnn.data_type.FLOAT)
    A, B = g.tensor_like(a), g.tensor_like(b)
    C = g.matmul(A=A, B=B)
    C.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    g.build_plans()
    return g, {A: a, B: b, C: c}, (a, b, c)


@pytest.mark.L0
@pytest.mark.parametrize(
    "form",
    ["tensor_keys", "uid_keys", "int_values", "int_values_and_workspace"],
)
def test_every_variant_pack_form_still_works(form):
    """The four shapes a variant pack has always been allowed to take."""
    g, vp, (a, b, c) = _matmul_graph()
    handle = cudnn.create_handle()
    ws = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    ws_arg = ws

    if form == "uid_keys":
        vp = {t.get_uid(): v for t, v in vp.items()}
    elif form == "int_values":
        vp = {t: v.data_ptr() for t, v in vp.items()}
    elif form == "int_values_and_workspace":
        vp = {t.get_uid(): v.data_ptr() for t, v in vp.items()}
        ws_arg = ws.data_ptr()

    g.execute(vp, ws_arg, handle=handle)
    torch.cuda.synchronize()
    ref = (a.float() @ b.float()).to(torch.bfloat16)
    torch.testing.assert_close(c, ref, atol=0.2, rtol=0.05)


@pytest.mark.L0
def test_missing_operand_names_the_tensor():
    g, vp, _ = _matmul_graph()
    handle = cudnn.create_handle()
    ws = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    partial = dict(list(vp.items())[:-1])
    with pytest.raises(ValueError, match="missing a buffer for tensor uid"):
        g.execute(partial, ws, handle=handle)


@pytest.mark.L0
def test_operand_order_is_the_backend_order():
    """The layout is the backend's own, ascending by uid — not a python guess.

    A walk over node ports cannot produce it: a tensor's ragged_offset is a
    user operand but hangs off the Tensor rather than off a port, and the slots
    the graph fills itself (pass-by-value scalars, slice replacement
    destinations, workspace modifications) must be excluded.
    """
    g, _, _ = _matmul_graph()
    order = g._variant_pack_uids()
    assert order == sorted(order), f"not ascending: {order}"
    assert order == list(g._lowered_graph._get_variant_pack_uids_sorted())


@pytest.mark.L0
def test_execute_is_reentrant():
    """One built graph, many threads, each with its own buffers.

    The pointer array is per call for this reason. Sharing one across calls
    hands each thread the other's pointers — silently, because every pointer in
    it is individually valid, so the failure is a wrong number and not a raise.
    """
    g, _, _ = _matmul_graph()
    handle = cudnn.create_handle()
    uids = g._variant_pack_uids()
    wrong = [0] * 8

    def worker(i):
        a = torch.full((1, M, K), float(i + 1), dtype=torch.bfloat16, device="cuda")
        b = torch.eye(K, N, dtype=torch.bfloat16, device="cuda").unsqueeze(0)
        c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
        # one workspace per thread: it is scratch the plan writes, so sharing
        # it would be the very crossing this test is looking for
        ws = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
        want = float(i + 1)
        for _ in range(200):
            g.execute({uids[0]: a, uids[1]: b, uids[2]: c}, ws, handle=handle)
            torch.cuda.synchronize()
            if c[0, 0, 0].item() != want:
                wrong[i] += 1

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(wrong) == 0, f"crossed buffers between threads: {wrong}"


@pytest.mark.L0
def test_describing_tensor_matches_the_dataclass():
    """``describing_tensor`` skips ``Tensor.__init__``, so every field it does
    not set has to resolve to the same default the dataclass would have given
    it. A new field with a ``default_factory`` gets no class attribute and
    would raise here rather than reach an engine as a missing attribute."""
    import dataclasses

    from cudnn.graph_types import Tensor, describing_tensor

    fast = describing_tensor(7, (4, 3), (6, 1), cudnn.data_type.FLOAT)
    slow = Tensor(uid=7, dim=(4, 3), stride=(6, 1), data_type=cudnn.data_type.FLOAT)
    for f in dataclasses.fields(Tensor):
        assert getattr(fast, f.name) == getattr(slow, f.name), f.name


@pytest.mark.L0
def test_shape_overrides_reach_a_migrated_plan_as_a_pack():
    """Overrides do not change what a python plan is handed.

    They exist so the backend can re-describe a tensor it lowered at another
    shape. A python engine reads the shape off the buffer, which is what the
    pack already carries. Branching on them used to send a migrated plan the
    raw uid map, which it cannot read.
    """
    total, h, d, nseq = 256, 4, 128, 2
    dt = cudnn.data_type.BFLOAT16
    g = cudnn.pygraph()
    q = g.tensor([total, h, d], data_type=dt, name="q")
    k = g.tensor([total, h, d], data_type=dt, name="k")
    v = g.tensor([total, h, d], data_type=dt, name="v")
    gate = g.tensor([total, h], data_type=cudnn.data_type.FLOAT, name="g")
    beta = g.tensor([total, h], data_type=cudnn.data_type.FLOAT, name="beta")
    cu = g.tensor([nseq + 1], data_type=cudnn.data_type.INT32, name="cu_seqlens")
    out, _fs, _h = g.gdn(q=q, k=k, v=v, g=gate, beta=beta, cu_seqlens=cu, scale=1.0 / d**0.5, name="gdn")
    out.set_output(True).set_data_type(dt)
    try:
        g.build()
    except Exception as exc:  # no python engine on this arch -- nothing to assert
        pytest.skip(f"no GDN engine here: {exc}")
    if not g._compiled_plans[g._plan_index].takes_variant_pack:
        pytest.skip("the selected plan has not migrated to the variant pack")

    per = total // nseq
    data = {
        q: torch.randn(total, h, d, dtype=torch.bfloat16, device="cuda"),
        k: torch.randn(total, h, d, dtype=torch.bfloat16, device="cuda"),
        v: torch.randn(total, h, d, dtype=torch.bfloat16, device="cuda"),
        gate: torch.rand(total, h, device="cuda").log(),
        beta: torch.rand(total, h, device="cuda"),
        cu: torch.tensor([0, per, 2 * per], dtype=torch.int32, device="cuda"),
        out: torch.empty(total, h, d, dtype=torch.bfloat16, device="cuda"),
    }
    ws = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    g.execute(data, ws)
    torch.cuda.synchronize()
    plain = data[out].clone()

    data[out].zero_()
    g.execute(data, ws, override_uids=[q.get_uid()], override_shapes=[[total, h, d]], override_strides=[[h * d, d, 1]])
    torch.cuda.synchronize()
    torch.testing.assert_close(data[out], plain)


# ---------------------------------------------------------------------------
# The declaration is the contract: a caller buffer whose own geometry disagrees
# with the graph's but covers its bytes is described AS the declaration, the
# way a bare address is. This is how the backend has always read a buffer
# (pointer only), and it is what lets a 2-D matrix serve a [1, m, k] tensor or
# a flat blob serve a reordered scale tensor on a python plan too.
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_a_buffer_that_disagrees_with_the_declaration_is_described_from_it():
    g, vp, (a, b, c) = _matmul_graph()
    A, B, C = vp.keys()
    ws = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    two_d = {A: a.view(M, K), B: b.view(K, N), C: c}  # what FlashInfer binds
    pack = g._normalize(g._uid_to_data(two_d), ws)
    for t, buf in ((A, a), (B, b)):
        i = pack.index_of(t)
        assert list(pack.native.shape(i)) == list(t.get_dim()) and list(pack.native.stride(i)) == list(t.get_stride())
        assert i in pack.graph_described  # an engine reads it in the graph's axis order
        assert pack.native.pointer(i) == buf.data_ptr()
    assert pack.index_of(C) not in pack.graph_described  # bound as declared: its own description stands
    # ... and the backend plan runs the 2-D binding bit-identically to the 3-D one.
    g.execute(vp, ws)
    torch.cuda.synchronize()
    ref = c.clone()
    c.zero_()
    g.execute(two_d, ws)
    torch.cuda.synchronize()
    torch.testing.assert_close(c, ref)


@pytest.mark.L0
def test_a_buffer_too_small_for_the_declaration_keeps_its_own_description():
    g, vp, (a, b, c) = _matmul_graph()
    A, B, C = vp.keys()
    ws = torch.empty(1, dtype=torch.uint8, device="cuda")
    half = a[:, : M // 2, :]  # a strided view spanning fewer slots than [1, M, K]
    pack = g._normalize(g._uid_to_data({A: half, B: b, C: c}), ws)
    i = pack.index_of(A)
    assert list(pack.native.shape(i)) == [1, M // 2, K] and i not in pack.graph_described


@pytest.mark.L0
def test_a_narrower_buffer_covering_the_slot_count_but_not_the_bytes_is_not_re_described():
    g, vp, (a, b, c) = _matmul_graph()
    A, B, C = vp.keys()
    ws = torch.empty(1, dtype=torch.uint8, device="cuda")
    as_bytes = torch.empty(a.numel(), dtype=torch.uint8, device="cuda")  # as many SLOTS as [1, M, K] bf16, half the bytes
    pack = g._normalize(g._uid_to_data({A: as_bytes, B: b, C: c}), ws)
    i = pack.index_of(A)
    assert list(pack.native.shape(i)) == [a.numel()] and i not in pack.graph_described


@pytest.mark.L0
def test_storage_geometry_packs_fp4_two_per_slot():
    from cudnn.graph_types import storage_geometry as _storage_geometry

    bf16, fp4 = cudnn.data_type.BFLOAT16, cudnn.data_type.FP4_E2M1
    assert _storage_geometry([1, 256, 256], [65536, 256, 1], bf16) == ((1, 256, 256), (65536, 256, 1))
    # row-major A [1, M, K]: K halves, the outer strides halve with it
    assert _storage_geometry([1, 256, 256], [65536, 256, 1], fp4) == ((1, 256, 128), (32768, 128, 1))
    # column-major B [1, K, N] (stride 1 on K): K halves, N's stride halves
    assert _storage_geometry([1, 256, 512], [131072, 1, 256], fp4) == ((1, 128, 512), (65536, 1, 128))
    assert _storage_geometry([1, 256, 255], [65280, 255, 1], fp4) is None  # no slot geometry spells an odd extent


@pytest.mark.L0
def test_a_bare_address_for_an_fp4_tensor_is_lent_the_storage_geometry():
    """A bare address borrows the declaration; for fp4 that is the x2 SLOT
    geometry, so an engine's per-slot packing factor applies to it like to any
    typed buffer (the caller guarantees the allocation covers the bytes)."""
    g = cudnn.pygraph(io_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    a = g.tensor(name="a", dim=[1, 256, 256], stride=[65536, 256, 1], data_type=cudnn.data_type.FP4_E2M1)
    b = g.tensor(name="b", dim=[1, 256, 512], stride=[131072, 1, 256], data_type=cudnn.data_type.FP4_E2M1)
    _, ta = g._describe(0x1000, a.get_uid())
    _, tb = g._describe(0x2000, b.get_uid())
    assert (tuple(ta.dim), tuple(ta.stride)) == ((1, 256, 128), (32768, 128, 1))
    assert (tuple(tb.dim), tuple(tb.stride)) == ((1, 128, 512), (65536, 1, 128))
