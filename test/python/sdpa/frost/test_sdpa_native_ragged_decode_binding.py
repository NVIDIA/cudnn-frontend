# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native ragged decode preserves per-port offsets and current packed capacities."""

import pytest

import cudnn
from cudnn.sdpa.fwd import prepared as prep
from test_sdpa_native_dense_binding import _pack
from test_sdpa_native_split_binding import _split_fixture

pytestmark = [pytest.mark.L0]


def _fixture(dtype="bfloat16", i64=False, stats="NH", rank=3, total=None):
    s, facts, frames, combined = _split_fixture(dtype=dtype, paged=True, lse=stats is not None)
    s.ragged, s.ragged_i64, s.total_q = True, i64, total
    s.ragged_lse_head_major = stats == "HN"
    s.ragged_divs = (s.qh * s.d_qk, s.qh * s.d_v, s.qh if stats != "HN" else 1)
    if rank == 3:
        for role in ("q", "o"):
            facts[role] = facts[role]._replace(shape=(s.b, s.qh, s.d_qk), strides=(s.qh * s.d_qk, s.d_qk, 1))
    if stats == "NH":
        facts["lse"] = facts["lse"]._replace(shape=(s.b, s.qh), strides=(s.qh, 1))
    elif stats == "HN":
        facts["lse"] = facts["lse"]._replace(shape=(s.qh, s.b), strides=(s.b, 1))
    for i, role in enumerate(("ragged_q", "ragged_o", "ragged_lse")):
        if role == "ragged_lse" and stats is None:
            continue
        facts[role] = prep.BufferFacts(0x60000 + i * 0x10000, "int64" if i64 else "int32", (2, 0), s.b + 1, (s.b + 1,), (1,))
    s.native = cudnn._pybind_module._SdpaDenseBinder(s)
    return s, facts, frames, combined


def _equal(s, facts, workspace=0x100000, stream=17):
    expected = prep.bind_dense_split(s, facts, workspace, stream, stream)
    actual = s.native.bind_split(_pack(facts), prep._NATIVE_DENSE_INDICES, workspace, stream)
    assert (actual is None) == (expected is None)
    if actual is not None:
        assert list(actual[0]) == expected[0]
        assert actual[1] == expected[1]
    return actual


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("i64", [False, True])
@pytest.mark.parametrize("stats", [None, "NH", "HN"])
@pytest.mark.parametrize("rank", [3, 4])
def test_ragged_frames_match_python_and_rebind_every_port(dtype, i64, stats, rank):
    s, facts, frames, combined = _fixture(dtype, i64, stats, rank)
    first = _equal(s, facts)
    fresh = {name: f._replace(ptr=f.ptr + 0x100000) for name, f in facts.items()}
    current = _equal(s, fresh, workspace=0x400000, stream=29)
    assert current[0][s.index["ragged_q_addr"]] != first[0][s.index["ragged_q_addr"]]
    assert current[1][8:11] != first[1][8:11]
    assert s.native.execute(_pack(fresh), prep._NATIVE_DENSE_INDICES, 29, workspace=0x400000)
    assert frames == [current[0]] and combined == [current[1]]


@pytest.mark.parametrize("i64", [False, True])
@pytest.mark.parametrize("role", ["ragged_q", "ragged_o", "ragged_lse"])
@pytest.mark.parametrize("change", ["missing", "dtype", "alignment", "device", "span", "size", "stride"])
def test_ragged_bad_offsets_rejected_before_either_launch_after_warmup(i64, role, change):
    s, facts, frames, combined = _fixture(i64=i64)
    _equal(s, facts)
    value = facts[role]
    updates = dict(
        dtype={"dtype": "int32" if i64 else "int64"},
        alignment={"ptr": value.ptr + (4 if i64 else 1)},
        device={"device": (1, 0)},
        span={"span": s.b},
        size={"shape": (s.b,)},
        stride={"strides": (2,), "span": 2 * s.b + 1},
    )
    bad = dict(facts)
    if change == "missing":
        del bad[role]
    else:
        bad[role] = value._replace(**updates[change])
    with pytest.raises(ValueError):
        prep.bind_dense_split(s, bad, 0x100000, 17, 17)
    with pytest.raises(ValueError):
        s.native.execute(_pack(bad), prep._NATIVE_DENSE_INDICES, 17, workspace=0x100000)
    assert frames == combined == []
    _equal(s, facts)


@pytest.mark.parametrize("role", ["q", "o", "lse"])
@pytest.mark.parametrize("stats", ["NH", "HN"])
def test_ragged_capacities_are_observed_after_geometry_cache_warmup(role, stats):
    s, facts, frames, combined = _fixture(stats=stats)
    _equal(s, facts)
    value = facts[role]
    # Warm geometry is identical; the producer's current byte span controls both
    # launch suppression and final-store bounds, never a cached token capacity.
    for span in (0, 1, value.span // 2, value.span):
        changed = dict(facts, **{role: value._replace(span=span)})
        current = _equal(s, changed)
        launched = s.native.execute(_pack(changed), prep._NATIVE_DENSE_INDICES, 17, workspace=0x100000)
        assert launched == (current is not None)
        if launched:
            assert frames.pop() == current[0] and combined.pop() == current[1]
        else:
            assert frames == combined == []


def test_ragged_empty_output_skips_offsets_and_kv_validation():
    s, facts, frames, combined = _fixture()
    facts = {role: facts[role] for role in ("q", "o", "lse")}
    facts["o"] = facts["o"]._replace(span=0, ptr=0)
    assert _equal(s, facts) is None
    assert not s.native.execute(_pack(facts), prep._NATIVE_DENSE_INDICES, 17, workspace=0x100000)
    assert frames == combined == []


@pytest.mark.parametrize("role", ["q", "o"])
def test_ragged_unknown_producer_span_is_rejected(role):
    s, facts, frames, combined = _fixture()
    _equal(s, facts)
    facts[role] = facts[role]._replace(span=-1)
    with pytest.raises(ValueError, match="sized buffer"):
        prep.bind_dense_split(s, facts, 0x100000, 17, 17)
    with pytest.raises(ValueError, match="sized buffer"):
        s.native.execute(_pack(facts), prep._NATIVE_DENSE_INDICES, 17, workspace=0x100000)
    assert frames == combined == []


@pytest.mark.parametrize("stats", [None, "NH", "HN"])
def test_ragged_declared_total_requires_final_capacity_and_clamps_query(stats):
    s, facts, frames, combined = _fixture(stats=stats, total=2)
    assert _equal(s, facts)[0][s.index["problem_size"]][-1] == 2
    for role in ("o",) if stats is None else ("o", "lse"):
        bad = dict(facts, **{role: facts[role]._replace(span=0)})
        with pytest.raises(ValueError, match="packed Q total"):
            prep.bind_dense_split(s, bad, 0x100000, 17, 17)
        with pytest.raises(ValueError, match="packed Q total"):
            s.native.execute(_pack(bad), prep._NATIVE_DENSE_INDICES, 17, workspace=0x100000)
    assert frames == combined == []


def test_ragged_wide_token_table_strides_and_per_port_divisors():
    s, facts, _, _ = _fixture(i64=True)
    token_strides = (2**33, 2**33 + 1)
    for role, ts in zip(("q", "o"), token_strides):
        f = facts[role]
        facts[role] = f._replace(strides=(ts, 128, 1), span=3 * ts + 8 * 128)
    for role in ("block_table", "block_table_v"):
        f = facts[role]
        facts[role] = f._replace(strides=(2**33, 1), span=3 * 2**33 + 8)
    s.ragged_divs = (*token_strides, s.qh)
    s.native = cudnn._pybind_module._SdpaDenseBinder(s)
    partial, combined = _equal(s, facts)
    assert partial[s.index["q_strides"]][0] == 2**33
    assert combined[6][1] == 2**33 + 1 and combined[11] == s.ragged_divs


def _graph_case(monkeypatch, request, *, i64, hnd, stats, wide=False, python_binding=False, standalone=False, square_stats=False):
    import inspect
    import torch
    from frost_test_utils import select_engine, _dsl_installed
    from test_sdpa_fwd_decode_d128_sm100 import _pools, _gather_kv, _ref

    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)) or not _dsl_installed():
        pytest.skip("native ragged decode requires SM100/SM103 and CuTe DSL")
    if wide and torch.cuda.mem_get_info()[0] < 80 << 30:
        pytest.skip("physical wide output and live page-table rows need 80 GiB free")

    def storage(shape, strides, dtype):
        # Another xdist worker can allocate after the free-memory check.
        # Only unavailable test storage is skippable; launches/assertions are not.
        try:
            return torch.empty_strided(shape, strides, device="cuda", dtype=dtype)
        except torch.OutOfMemoryError:
            if wide:
                pytest.skip("insufficient free GPU memory for physical wide-stride storage")
            raise

    b, h, hk, d, page, pages = (2 if wide else 3), 8, 2, 128, 16, 8
    cap = 2 if wide else (h if square_stats else 6)
    dtype = torch.bfloat16
    q_stride = 2**32 + h * d if wide else h * d
    q = storage((cap, h, d), (q_stride, d, 1), dtype)
    q.copy_(torch.randn(cap, h, d, device="cuda", dtype=dtype))
    o_stride = 2**32 + 2 * h * d if wide else h * d + 16
    o = storage((cap, h, d), (o_stride, d, 1), dtype)
    lse = torch.empty((h, cap) if stats == "HN" else (cap, h), device="cuda", dtype=torch.float32)
    kp, vp, table = _pools(b, hk, d, page, pages, hnd, dtype, seed=19)
    k, v = (kp, vp) if hnd else (kp.transpose(1, 2), vp.transpose(1, 2))
    tables = [table.clone(), table.flip(1).contiguous()]
    if wide:
        narrow = tables
        tables = [storage((b, pages), (2**32 + 32, 1), torch.int32) for _ in range(2)]
        for dst, src in zip(tables, narrow):
            dst.copy_(src)
    lengths = torch.tensor([17, 119] if wide else [17, 99, 119], device="cuda", dtype=torch.int32).view(b, 1, 1, 1)
    q_lengths = torch.tensor([1, 1] if wide else [1, 0, 1], device="cuda", dtype=torch.int32).view(b, 1, 1, 1)
    q_rows = [0, 1, 2] if wide else [1, 2, 2, 3]
    o_rows = [0, 1, 2] if wide else [2, 0, 4, 5]
    lse_rows = [0, 1, 2] if wide else [0, 3, 4, 5]
    divisors = (q_stride, o_stride, 1 if stats == "HN" else h)
    offset_dtype = torch.int64 if i64 else torch.int32
    offsets = [
        (torch.tensor(rows, device="cuda", dtype=torch.int64) * div).to(offset_dtype).view(b + 1, 1, 1, 1)
        for rows, div in zip((q_rows, o_rows, lse_rows), divisors)
    ]
    handle = cudnn.create_handle()
    request.addfinalizer(lambda: cudnn.destroy_handle(handle))
    graph = cudnn.pygraph(
        handle=handle, io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT
    )
    q_t = graph.tensor(dim=[b, h, 1, d], stride=[q_stride, d, q_stride, 1], data_type=cudnn.data_type.BFLOAT16)
    k_t, v_t = graph.tensor_like(k), graph.tensor_like(v)
    qt, kt = graph.tensor_like(q_lengths), graph.tensor_like(lengths)
    table_t = [graph.tensor(dim=[b, 1, pages, 1], stride=[t.stride(0), pages, 1, 1], data_type=cudnn.data_type.INT32) for t in tables]
    ro_t = [graph.tensor_like(t) for t in offsets]
    q_t.set_ragged_offset(ro_t[0])
    out_t, lse_t = graph.sdpa(
        q=q_t,
        k=k_t,
        v=v_t,
        generate_stats=stats is not None,
        attn_scale=d**-0.5,
        use_padding_mask=True,
        seq_len_q=qt,
        seq_len_kv=kt,
        paged_attention_k_table=table_t[0],
        paged_attention_v_table=table_t[1],
        paged_attention_max_seq_len_kv=pages * page,
    )
    out_t.set_output(True).set_dim([b, h, 1, d]).set_stride([o_stride, d, o_stride, 1]).set_ragged_offset(ro_t[1])
    if stats is not None:
        lse_t.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim([b, h, 1, 1])
        lse_t.set_stride([1, cap, 1, 1] if stats == "HN" else [h, 1, h, 1]).set_ragged_offset(ro_t[2])
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    select_engine(graph, "sdpa_fwd_prefill_sm100")
    graph.check_support()
    graph.build_plans()
    launch = graph._compiled_plans[graph._plan_index]._prepared
    assert isinstance(launch, prep.PreparedDenseLaunch) and launch.spec.ragged
    if python_binding:
        launch.spec.native = None
    else:
        assert launch.spec.native is not None, "ragged decode must bind natively"
        monkeypatch.setattr(prep, "facts_of_roles", lambda *a: pytest.fail("native ragged path rebuilt Python facts"))
        monkeypatch.setattr(prep, "bind_dense_split", lambda *a: pytest.fail("native ragged path used the Python binder"))
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    def pack():
        result = {q_t: q, k_t: k, v_t: v, out_t: o, qt: q_lengths, kt: lengths}
        result.update(zip(table_t, tables))
        result.update(zip(ro_t[:2], offsets[:2]))
        if stats is not None:
            result[lse_t], result[ro_t[2]] = lse, offsets[2]
        return result

    def poison():
        o.fill_(123)
        lse.fill_(123)

    def check():
        live_o, live_lse = set(), set()
        for batch in range(b):
            if not int(q_lengths[batch].item()):
                continue
            qr, op, sp = (int(t[batch].item()) // div for t, div in zip(offsets, divisors))
            count = int(lengths[batch].item())
            expected_o, expected_lse = _ref(
                q[qr : qr + 1], _gather_kv(kp, tables[0][batch], count, hnd), _gather_kv(vp, tables[1][batch], count, hnd), 1, d**-0.5
            )
            torch.testing.assert_close(o[op : op + 1].float(), expected_o, atol=0.025, rtol=0.01)
            live_o.add(op)
            if stats is not None:
                actual = lse[:, sp] if stats == "HN" else lse[sp]
                torch.testing.assert_close(actual, expected_lse[:, 0], atol=0.005, rtol=0.001)
                live_lse.add(sp)
        for row in set(range(cap)) - live_o:
            assert torch.all(o[row] == 123).item()
        if stats is not None:
            for row in set(range(cap)) - live_lse:
                assert torch.all((lse[:, row] if stats == "HN" else lse[row]) == 123).item()

    api = inspect.getclosurevars(graph._compiled_plans[graph._plan_index]._compiled).nonlocals["api"]
    assert api._dense_spec is launch.spec

    def call():
        if standalone:
            api.execute(
                q,
                k,
                v,
                o,
                lse_tensor=lse if stats is not None else None,
                seq_q_lens=q_lengths,
                seq_kv_lens=lengths,
                block_table=tables[0],
                block_table_v=tables[1],
                workspace=workspace,
                ragged_q=offsets[0],
                ragged_o=offsets[1],
                ragged_lse=offsets[2] if stats is not None else None,
            )
        else:
            cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)
            graph.execute(pack(), workspace, handle=handle)

    poison()
    call()
    torch.cuda.synchronize()
    check()
    retained = (q, o, lse, offsets, workspace)
    q = storage(q.shape, q.stride(), q.dtype).copy_(q).add_(0.1)
    # Rebind equal-layout outputs; retain the wide physical span without copying its gaps.
    o = storage(o.shape, o.stride(), o.dtype)
    lse = torch.empty_like(lse)
    offsets = [t.clone() for t in offsets]
    workspace = torch.empty_like(workspace)
    poison()
    call()
    torch.cuda.synchronize()
    check()
    bad = pack()
    bad[ro_t[1]] = offsets[1][:1]
    workspace.fill_(37)
    poison()
    with pytest.raises(ValueError):
        graph.execute(bad, workspace, handle=handle)
    torch.cuda.synchronize()
    assert torch.all(workspace == 37).item() and torch.all(o == 123).item()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    stream.synchronize()
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture, stream=stream):
        torch.cuda.set_sync_debug_mode("error")
        try:
            call()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    q.mul_(0.9)
    lengths.sub_(1)
    tables[0].copy_(tables[0].flip(1))
    if not wide:
        for target, rows, div in zip(offsets, ([0, 1, 1, 2], [1, 3, 5, 6], [2, 0, 5, 6]), divisors):
            target.copy_(torch.tensor(rows, device="cuda", dtype=torch.int64).mul(div).to(offset_dtype).view_as(target))
    poison()
    capture.replay()
    torch.cuda.synchronize()
    check()
    capture.reset()
    del retained


@pytest.mark.parametrize("i64", [False, True])
@pytest.mark.parametrize("hnd", [False, True])
@pytest.mark.parametrize("stats", [None, "NH", "HN"])
def test_ragged_graph_independent_origins_fresh_buffers_and_replay(i64, hnd, stats, monkeypatch, request):
    _graph_case(monkeypatch, request, i64=i64, hnd=hnd, stats=stats)


@pytest.mark.parametrize("python_binding", [False, True], ids=["native", "python"])
def test_ragged_wide_live_output_and_page_table_strides(python_binding, monkeypatch, request):
    _graph_case(monkeypatch, request, i64=True, hnd=True, stats="HN", wide=True, python_binding=python_binding)


@pytest.mark.parametrize("i64", [False, True])
@pytest.mark.parametrize("stats", [None, "NH", "HN"])
def test_ragged_standalone_independent_origins_and_capture(i64, stats, monkeypatch, request):
    _graph_case(monkeypatch, request, i64=i64, hnd=True, stats=stats, standalone=True)


@pytest.mark.parametrize("python_binding", [False, True], ids=["native", "python"])
@pytest.mark.parametrize("stats", ["NH", "HN"])
def test_ragged_square_stats_follow_declared_packing(stats, python_binding, monkeypatch, request):
    _graph_case(monkeypatch, request, i64=True, hnd=True, stats=stats, standalone=True, python_binding=python_binding, square_stats=True)
