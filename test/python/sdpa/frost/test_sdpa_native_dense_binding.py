# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native/Python binding contracts and bounded SM100 graph execution checks."""

import ast
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import cudnn
from cudnn.sdpa.fwd import prepared as prep

pytestmark = [pytest.mark.L0]


def _fixture(dtype="bfloat16", paged=False, hnd=False, lse=True, lengths=True, d=128, sq=1):
    s = prep.DenseLaunchSpec()
    s.b, s.qh, s.kh, s.d_qk, s.d_v = 4, 8, 2, d, d
    s.s_q_max, s.s_k_max, s.page_size, s.tile_n = sq, 128, 16, 128
    s.paged, s.paged_hnd, s.has_lse, s.has_sink = paged, hnd, lse, False
    s.seq_kv_present, s.seq_q_present = lengths, lengths
    s.device_index, s.split, s.window_right = 0, 1, 0
    s.ragged = s.fp32_partial = s.shape_fixed = s.lpt_grid_fixed = s.kv_tail_native = s.causal = s.causal_bottom_right = False
    s.gate_expect = s.quant = s.combine = None
    s.expect = dict.fromkeys(("q", "k", "v", "o"), dtype)
    host_path = Path(prep.__file__).parent / f"kernels/sm100/decode_d{256 if d == 256 else 128}_f16.py"
    host = next(n for n in ast.parse(host_path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_host")
    s.order = [arg.arg for arg in host.args.args if "Constexpr" not in ast.unparse(arg.annotation)]
    s.index = {name: i for i, name in enumerate(s.order)}
    s.template = [None] * len(s.order)
    s.template[s.index["seq_q_lens_addr"]] = 0
    s.owner = object()
    frames = []
    s.fn = lambda *frame: frames.append(frame)
    facts = {}
    for i, (role, h, seq) in enumerate((("q", 8, sq), ("k", 2, 128), ("v", 2, 128), ("o", 8, sq))):
        shape, strides = (4, h, seq, d), (seq * h * d, d, h * d, 1)
        if paged and role in ("k", "v"):
            shape = (32, h, 16, d)
            strides = (16 * h * d, 16 * d, d, 1) if hnd else (16 * h * d, d, h * d, 1)
        span = sum((n - 1) * st for n, st in zip(shape, strides)) + 1
        facts[role] = prep.BufferFacts(0x1000 * (i + 1), dtype, (2, 0), span, shape, strides)
    if lengths:
        for i, role in enumerate(("seq_q_lens", "seq_kv_lens")):
            facts[role] = prep.BufferFacts(0x10000 + i * 0x1000, "int32", (2, 0), 4, (4,), (1,))
    if lse:
        facts["lse"] = prep.BufferFacts(0x20000, "float32", (2, 0), 32 * sq, (4, 8, sq), (8 * sq, 1, 8))
    if paged:
        for i, role in enumerate(("block_table", "block_table_v")):
            facts[role] = prep.BufferFacts(0x30000 + i * 0x1000, "int32", (2, 0), 32, (4, 8), (8, 1))
    s.native = cudnn._pybind_module._SdpaDenseBinder(s)
    return s, facts, frames


def _pack(facts):
    pack = cudnn._pybind_module.VariantPackNative(len(prep._NATIVE_DENSE_ROLES))
    for i, role in enumerate(prep._NATIVE_DENSE_ROLES):
        prep._set_native_fact(pack, i, facts.get(role))
    return pack


def _native(s, facts, stream=17):
    return s.native.bind(_pack(facts), tuple(range(len(prep._NATIVE_DENSE_ROLES))), stream)


def _equal(s, facts, stream=17):
    actual = _native(s, facts, stream)
    assert list(actual) == prep.bind_dense(s, facts, stream, stream)
    return actual


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("paged,hnd", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("lse", [False, True])
@pytest.mark.parametrize("lengths", [False, True])
@pytest.mark.parametrize("d,sq", [(64, 1), (128, 1), (256, 1), (256, 4)])
def test_native_dense_matches_python_and_rebinds(dtype, paged, hnd, lse, lengths, d, sq):
    s, facts, _ = _fixture(dtype, paged, hnd, lse, lengths, d=d, sq=sq)
    first = _equal(s, facts)
    changed = {name: f._replace(ptr=f.ptr + 0x100000) for name, f in facts.items()}
    second = _equal(s, changed, stream=23)
    assert first[s.index["q_ptr"]] == facts["q"].ptr
    assert second[s.index["q_ptr"]] == changed["q"].ptr
    assert first[s.index["stream"]] == 17 and second[s.index["stream"]] == 23


@pytest.mark.parametrize("role", ["q", "k", "v", "o", "lse", "seq_q_lens", "seq_kv_lens", "block_table", "block_table_v"])
@pytest.mark.parametrize("change", ["missing", "short", "misaligned", "wrong_device", "host", "wrong_dtype"])
@pytest.mark.parametrize("d,sq", [(128, 1), (64, 1), (256, 4)])
def test_native_dense_rechecks_current_storage_after_warmup(role, change, d, sq):
    s, facts, frames = _fixture(paged=True, d=d, sq=sq)
    _equal(s, facts)
    f = facts[role]
    updates = {
        "short": {"span": 0},
        "misaligned": {"ptr": f.ptr + 1},
        "wrong_device": {"device": (2, 1)},
        "host": {"device": (1, 0)},
        "wrong_dtype": {"dtype": "float32" if f.dtype != "float32" else "bfloat16"},
    }
    changed = dict(facts)
    if change == "missing":
        del changed[role]
    else:
        changed[role] = f._replace(**updates[change])
    with pytest.raises(ValueError):
        prep.bind_dense(s, changed, 17, 17)
    with pytest.raises(ValueError):
        s.native.execute(_pack(changed), tuple(range(len(prep._NATIVE_DENSE_ROLES))), 17)
    assert frames == []


@pytest.mark.parametrize(
    "role,updates",
    [
        ("q", {"shape": (5, 8, 1, 128)}),
        ("q", {"shape": (4, 7, 1, 128)}),
        ("q", {"shape": (4, 8, 2, 128)}),
        ("q", {"strides": (1024, 129, 1024, 1)}),
        ("q", {"strides": (1024, 128, 1024, 2)}),
        ("q", {"strides": (-1024, 128, 1024, 1)}),
        ("o", {"shape": (2, 8, 1, 128)}),
        ("lse", {"strides": (1, 1, 1)}),
        ("lse", {"shape": (4, 7, 1)}),
        ("seq_q_lens", {"strides": (2,), "span": 7}),
        ("seq_kv_lens", {"shape": (3,), "span": 3}),
        ("k", {"shape": (31, 2, 16, 128)}),
        ("k", {"strides": (4096, 2048, 128, 1)}),
        ("k", {"strides": (4097, 128, 256, 1)}),
        ("block_table", {"ptr": 0}),
        ("block_table", {"strides": (8, -1)}),
        ("block_table_v", {"shape": (4, 7)}),
        ("block_table_v", {"shape": (3, 8)}),
    ],
)
def test_native_dense_geometry_errors_match_python(role, updates):
    s, facts, _ = _fixture(paged=True)
    _equal(s, facts)
    changed = dict(facts, **{role: facts[role]._replace(**updates)})
    with pytest.raises(ValueError):
        prep.bind_dense(s, changed, 17, 17)
    with pytest.raises(ValueError):
        _native(s, changed)


@pytest.mark.parametrize("d", [64, 128, 256])
def test_native_dense_wide_independent_tables_and_dynamic_geometry(d):
    s, facts, _ = _fixture(paged=True, d=d)
    for batch in (4, 2, 1, 4):
        changed = dict(facts)
        for role in ("q", "o"):
            f = facts[role]
            stride = 2**32 + 8 * d
            changed[role] = f._replace(shape=(batch, 8, 1, d), strides=(stride, d, 8 * d, 1), span=(batch - 1) * stride + 8 * d)
        changed["block_table"] = facts["block_table"]._replace(shape=(4, 1, 8, 1), strides=(2**33, 1, 3, 1), span=(batch - 1) * 2**33 + 22)
        changed["block_table_v"] = facts["block_table_v"]._replace(strides=(2**33, 3), span=(batch - 1) * 2**33 + 22)
        actual = _equal(s, changed)
        assert actual[s.index["table_strides"]] == (2**33, 3)
        assert actual[s.index["block_table_ptr"]] != actual[s.index["block_table_v_ptr"]]
        assert actual[s.index["problem_size"]][0] == batch


@pytest.mark.parametrize("fixed", [None, "shape_fixed", "lpt_grid_fixed"])
def test_native_decode_query_overrides_respect_plan_specialization(fixed):
    s, facts, _ = _fixture(d=256, sq=4)
    if fixed:
        setattr(s, fixed, True)
        s.native = cudnn._pybind_module._SdpaDenseBinder(s)
    _equal(s, facts)
    changed = dict(facts)
    for role in ("q", "o"):
        changed[role] = facts[role]._replace(shape=(4, 8, 2, 256))
    if fixed:
        with pytest.raises(ValueError):
            _native(s, changed)
        with pytest.raises(ValueError):
            prep.bind_dense(s, changed, 17, 17)
    else:
        assert _equal(s, changed)[s.index["problem_size"]][3] == 2
    _equal(s, facts)


def test_native_standalone_scale_override_does_not_mutate_plan():
    s, facts, frames = _fixture(d=256, sq=4)
    s.template[s.index["scale_softmax_log2"]] = 0.5
    s.native = cudnn._pybind_module._SdpaDenseBinder(s)
    for scale in (0.25, 0.0, 0.5):
        s.native.execute(_pack(facts), prep._NATIVE_DENSE_INDICES, 17, scale)
        assert frames[-1][s.index["scale_softmax_log2"]] == scale
        assert s.template[s.index["scale_softmax_log2"]] == 0.5
    assert _equal(s, facts)[s.index["scale_softmax_log2"]] == 0.5


def test_native_dense_geometry_cache_never_caches_observations(monkeypatch):
    original = prep._dense_role_layout
    calls = []

    def observe(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(prep, "_dense_role_layout", observe)
    s, facts, _ = _fixture()
    _native(s, facts)
    initial_calls = len(calls)
    assert initial_calls > 0
    _native(s, {name: f._replace(ptr=f.ptr + 0x100000) for name, f in facts.items()})
    assert len(calls) == initial_calls
    with pytest.raises(ValueError):
        _native(s, dict(facts, q=facts["q"]._replace(span=1)))
    assert len(calls) == initial_calls
    changed = dict(facts)
    for role in ("q", "k", "v", "o"):
        changed[role] = facts[role]._replace(shape=(2, *facts[role].shape[1:]))
    assert _equal(s, changed)[s.index["problem_size"]][0] == 2
    assert _equal(s, facts)[s.index["problem_size"]][0] == 4


def test_native_dense_graph_hot_path_does_not_materialize_python_facts(monkeypatch):
    s, facts, frames = _fixture(paged=True)
    launch = prep.PreparedDenseLaunch.__new__(prep.PreparedDenseLaunch)
    launch.spec = s
    launch._roles = prep._NATIVE_DENSE_ROLES
    launch._uids = tuple(range(len(launch._roles)))
    launch._indices = None
    launch._native_indices = None
    monkeypatch.setattr(prep, "facts_of_roles", lambda *args: pytest.fail("native graph path rebuilt Python facts"))
    monkeypatch.setattr(prep, "bind_dense", lambda *args: pytest.fail("native graph path used Python binder"))
    launch.execute(SimpleNamespace(native=_pack(facts), index_of=launch._uids.index), 0, 17, 17)
    changed = dict(facts, q=facts["q"]._replace(span=1))
    with pytest.raises(ValueError):
        launch.execute(SimpleNamespace(native=_pack(changed), index_of=launch._uids.index), 0, 23, 23)
    assert len(frames) == 1


def test_native_dense_frames_are_per_call():
    s, facts, frames = _fixture(paged=True)
    _native(s, facts)  # initialize pure geometry before concurrent entry

    def execute(i):
        changed = {name: f._replace(ptr=f.ptr + i * 0x100000) for name, f in facts.items()}
        s.native.execute(_pack(changed), tuple(range(len(prep._NATIVE_DENSE_ROLES))), 17 + i)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(execute, range(12)))
    assert {(f[s.index["q_ptr"]], f[s.index["stream"]]) for f in frames} == {(facts["q"].ptr + i * 0x100000, 17 + i) for i in range(12)}


def test_native_dense_optional_and_tail_contract():
    s, facts, _ = _fixture(lse=False, lengths=False)
    _equal(s, facts)
    extra = prep.BufferFacts(0x20000, "float32", (2, 0), 8, (8,), (1,))
    for role in ("lse", "sinks", "gate"):
        with pytest.raises(ValueError):
            _native(s, dict(facts, **{role: extra}))
        with pytest.raises(ValueError):
            prep.bind_dense(s, dict(facts, **{role: extra}), 17, 17)
    changed = dict(facts)
    for role in ("k", "v"):
        changed[role] = facts[role]._replace(shape=(4, 2, 127, 128))
    with pytest.raises(ValueError):
        _native(s, changed)
    with pytest.raises(ValueError):
        prep.bind_dense(s, changed, 17, 17)


@pytest.mark.parametrize("right", [-4, -1, 0, 1])
def test_native_dense_signed_causal_tail_contract(right):
    s, facts, _ = _fixture(lengths=False)
    s.causal, s.window_right = True, right
    s.native = cudnn._pybind_module._SdpaDenseBinder(s)
    for role in ("k", "v"):
        facts[role] = facts[role]._replace(shape=(4, 2, 127, 128))
    _equal(s, facts)


def test_native_dense_empty_geometry_and_observed_bytes():
    s, facts, _ = _fixture()
    changed = dict(facts, q=facts["q"]._replace(shape=(), strides=()))
    with pytest.raises(ValueError):
        _native(s, changed)
    with pytest.raises(ValueError):
        prep.bind_dense(s, changed, 17, 17)
    pack = _pack(facts)
    q_index = prep._NATIVE_DENSE_ROLES.index("q")
    q = facts["q"]
    # The effective half descriptor must not turn a short producer byte span
    # into enough storage. Cache hits retain this distinction on every call.
    _equal(s, facts)
    pack.set_operand(q_index, q.ptr, q.shape, q.strides, 4, 16, 1, q.span * 2 - 1, 2, 0)
    with pytest.raises(ValueError, match="observed storage"):
        s.native.bind(pack, tuple(range(len(prep._NATIVE_DENSE_ROLES))), 17)


def test_native_dense_shared_table_stride_host_contract():
    s, facts, _ = _fixture(paged=True)
    assert "table_v_strides" not in s.index
    _equal(s, facts)
    changed = dict(facts, block_table_v=facts["block_table_v"]._replace(strides=(16, 2), span=63))
    with pytest.raises(ValueError, match="matching K/V table strides"):
        _native(s, changed)
    with pytest.raises(ValueError, match="matching K/V table strides"):
        prep.bind_dense(s, changed, 17, 17)


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("d,sq", [(64, 1), (128, 1), (256, 1), (256, 4)])
@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
def test_native_dense_graph_fresh_bindings_and_changed_replay(paged, d, sq, dtype_name, monkeypatch, request, wide_tables=False, has_sink=False):
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("native dense binding is bounded to SM100")
    from frost_test_utils import _dsl_installed
    from test_sdpa_fwd_decode_d128_sm100 import _SM100_ID, _gather_kv, _ref
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    if not _dsl_installed():
        pytest.skip("needs the supported CuTe DSL")
    gen = torch.Generator(device="cuda").manual_seed(107)
    b, h, kh, sk, page = 2, 8, 2, 256 if d == 256 else 128, 128 if d == 256 else 16
    dtype = getattr(torch, dtype_name)
    q = torch.randn((b, sq, h, d), device="cuda", dtype=dtype, generator=gen).transpose(1, 2)
    if paged:
        pool_shape = (b * sk // page, page, kh, d)
        k = torch.randn(pool_shape, device="cuda", dtype=dtype, generator=gen).transpose(1, 2)
        v = torch.randn(pool_shape, device="cuda", dtype=dtype, generator=gen).transpose(1, 2)
    else:
        k = torch.randn((b, sk, kh, d), device="cuda", dtype=dtype, generator=gen).transpose(1, 2)
        v = torch.randn((b, sk, kh, d), device="cuda", dtype=dtype, generator=gen).transpose(1, 2)
    # Padded output storage detects a wrong stride or an accidental compact store.
    o_storage = torch.empty((b, sq, h, d + 16), device="cuda", dtype=dtype).transpose(1, 2)
    o = o_storage[..., :d]
    lse = torch.empty((b, h, sq, 1), device="cuda", dtype=torch.float32)
    q_lens = torch.full((b, 1, 1, 1), sq, device="cuda", dtype=torch.int32)
    kv_lens = torch.tensor([sk, 63], device="cuda", dtype=torch.int32).view(b, 1, 1, 1)
    tables = [torch.arange(b * sk // page, device="cuda", dtype=torch.int32).view(b, 1, sk // page, 1)]
    tables.append(tables[0].flip(2).clone())
    if wide_tables:
        # Two live batch rows force device address arithmetic to traverse the
        # wide stride. A singleton only checks the host argument's type.
        wide = [torch.empty_strided(t.shape, (2**32 + 32, 1, 1, 1), device="cuda", dtype=t.dtype) for t in tables]
        for dst, src in zip(wide, tables):
            dst.copy_(src)
        tables = wide
    inputs = [q, k, v, q_lens, kv_lens] + (tables if paged else [])
    if has_sink:
        inputs.append(torch.linspace(-2, 6, h, device="cuda", dtype=torch.float32).view(1, h, 1, 1))
    handle = cudnn.create_handle()
    request.addfinalizer(lambda: cudnn.destroy_handle(handle))
    graph = cudnn.pygraph(
        handle=handle,
        io_data_type=cudnn.data_type.BFLOAT16 if dtype == torch.bfloat16 else cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        is_override_shape_enabled=True,
    )
    desc = [graph.tensor_like(t).set_uid(i + 1) for i, t in enumerate(inputs)]
    kwargs = dict(paged_attention_k_table=desc[5], paged_attention_v_table=desc[6], paged_attention_max_seq_len_kv=sk) if paged else {}
    if has_sink:
        kwargs["sink_token"] = desc[-1]
    out, stats = graph.sdpa(
        q=desc[0], k=desc[1], v=desc[2], generate_stats=True, attn_scale=d**-0.5, use_padding_mask=True, seq_len_q=desc[3], seq_len_kv=desc[4], **kwargs
    )
    out.set_uid(100).set_output(True).set_dim(o.shape).set_stride(o.stride())
    stats.set_uid(101).set_output(True).set_dim(lse.shape).set_stride(lse.stride()).set_data_type(cudnn.data_type.FLOAT)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plan(_SM100_ID, SdpaFwdKnobs(sched_policy=0, tile_m=128, tile_n=128, cga=2 if d == 256 else 1, pack_gqa=True, split_kv=1))
    graph.build_plan_at_index(0)
    launch = graph._compiled_plans[graph._plan_index]._prepared
    assert isinstance(launch, prep.PreparedDenseLaunch) and launch.spec.native is not None
    monkeypatch.setattr(prep, "facts_of_roles", lambda *args: pytest.fail("native graph rebuilt Python facts"))
    monkeypatch.setattr(prep, "bind_dense", lambda *args: pytest.fail("native graph used Python binding"))
    workspace = torch.empty(max(graph.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    uids = tuple(range(1, len(inputs) + 1)) + (100, 101)
    completions = []
    finish = graph._finish_ordered

    def complete(*args):
        completions.append(1)
        return finish(*args)

    monkeypatch.setattr(graph, "_finish_ordered", complete)

    def call(buffers=None, call_uids=None, call_workspace=None, **overrides):
        cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)
        graph.execute(
            (*inputs, o, lse) if buffers is None else buffers,
            workspace if call_workspace is None else call_workspace,
            handle=handle,
            tensor_uids=uids if call_uids is None else call_uids,
            **overrides,
        )

    def check(active_batch=b):
        torch.cuda.synchronize()
        for i in range(active_batch):
            keys, vals = inputs[1][i].transpose(0, 1), inputs[2][i].transpose(0, 1)
            length = int(inputs[4][i].item())
            if paged:
                keys = _gather_kv(inputs[1].transpose(1, 2), inputs[5][i].reshape(-1), length, False)
                vals = _gather_kv(inputs[2].transpose(1, 2), inputs[6][i].reshape(-1), length, False)
            expected_o, expected_lse = _ref(
                inputs[0][i].transpose(0, 1), keys[:length], vals[:length], int(inputs[3][i].item()), d**-0.5, sink=inputs[-1] if has_sink else None
            )
            torch.testing.assert_close(o[i].transpose(0, 1).float(), expected_o, atol=0.03, rtol=0.01)
            torch.testing.assert_close(lse[i, :, :, 0], expected_lse, atol=0.005, rtol=0.001)
        assert torch.all(o_storage[..., d:] == 123).item()

    def poison():
        o_storage.fill_(123)
        lse.fill_(float("nan"))

    poison()
    torch.cuda.set_sync_debug_mode("error")
    try:
        call()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    check()
    assert not completions, "native ordered tensors rebuilt a Python VariantPack"

    class PythonBuffer:
        # A supported producer without the native exchange protocol must finish
        # the current observation before calling the same validated binder.
        def __init__(self, tensor):
            self.tensor = tensor
            self.shape, self.dtype, self.device = tensor.shape, tensor.dtype, tensor.device

        def data_ptr(self):
            return self.tensor.data_ptr()

        def stride(self):
            return self.tensor.stride()

        def element_size(self):
            return self.tensor.element_size()

    for wrap_workspace in (False, True):
        poison()
        if wrap_workspace:
            call(call_workspace=PythonBuffer(workspace))
        else:
            call(buffers=(PythonBuffer(inputs[0]), *inputs[1:], o, lse))
        check()
    assert len(completions) == 2
    completions.clear()

    # Even this zero-scratch plan validates a workspace supplied by the caller.
    # Both invalid calls otherwise have valid writable outputs, so poisoning
    # detects an accidental launch without risking an invalid device address.
    for invalid in (("duplicate_uid", "strided_workspace", "short_sink") if has_sink else ("duplicate_uid", "strided_workspace")):
        poison()
        with pytest.raises(ValueError):
            if invalid == "duplicate_uid":
                call(call_uids=(uids[1], *uids[1:]))
            elif invalid == "strided_workspace":
                call(call_workspace=torch.empty(8, device="cuda", dtype=torch.uint8)[::2])
            else:
                call(buffers=(*inputs[:-1], inputs[-1].view(-1)[:1], o, lse))
        torch.cuda.synchronize()
        assert torch.all(o_storage == 123).item()
        assert torch.isnan(lse).all().item()
        call(buffers=tuple(reversed((*inputs, o, lse))), call_uids=tuple(reversed(uids)))
        check()
    completions.clear()

    # The same artifact accepts a smaller batch through overrides and then
    # returns to the declared batch. Pool capacities stay unchanged when paged.
    smaller = tuple(t if (paged and i in (1, 2)) or (has_sink and i == len(inputs) - 1) else t[:1] for i, t in enumerate((*inputs, o, lse)))
    poison()
    call(
        buffers=smaller,
        override_uids=uids,
        override_shapes=tuple(tuple(t.shape) for t in smaller),
        override_strides=tuple(tuple(t.stride()) for t in smaller),
    )
    check(active_batch=1)
    assert torch.all(o_storage[1:] == 123).item()
    call()
    check()
    assert not completions

    owners = (inputs, o_storage, lse, workspace)
    inputs = [t.clone() for t in inputs]
    o_storage = torch.empty_like(o_storage)
    o = o_storage[..., :d]
    lse, workspace = torch.empty_like(lse), torch.empty_like(workspace)
    poison()
    call()
    check()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    stream.synchronize()
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture, stream=stream):
            call()
        inputs[0].add_(0.17)
        inputs[1].mul_(0.9)
        if has_sink:
            inputs[-1].add_(2.0)
        inputs[3][1].zero_()
        if paged:
            inputs[5].copy_(inputs[5].flip(2))
            inputs[6].copy_(inputs[6].flip(2))
        poison()
        capture.replay()
        check()
        assert not completions
    finally:
        capture.reset()
    del owners


def test_native_dense_unknown_storage_contract_is_preserved():
    s, facts, _ = _fixture(paged=True)
    _equal(s, {name: f._replace(span=-1, device=(-1, -1)) for name, f in facts.items()})


@pytest.mark.parametrize("d,sq", [(64, 1), (128, 1), (256, 4)])
def test_native_decode_standalone_rebinds_scale_and_capture(d, sq, monkeypatch, has_sink=False):
    import torch

    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("native decode needs SM100")
    from frost_test_utils import _dsl_installed

    if not _dsl_installed():
        pytest.skip("needs the supported CuTe DSL")
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    gen = torch.Generator(device="cuda").manual_seed(139)
    q = torch.randn((2, sq, 8, d), device="cuda", dtype=torch.bfloat16, generator=gen).transpose(1, 2)
    k = torch.randn((2, 128, 2, d), device="cuda", dtype=torch.bfloat16, generator=gen).transpose(1, 2)
    v = torch.randn(k.shape, device="cuda", dtype=k.dtype, generator=gen).transpose(1, 2).contiguous().transpose(1, 2)
    o = torch.empty_like(q)
    lse = torch.empty((2, 8, sq), device="cuda", dtype=torch.float32)
    q_lens = torch.full((2,), sq, device="cuda", dtype=torch.int32)
    kv_lens = torch.full((2,), 128, device="cuda", dtype=torch.int32)
    sinks = torch.linspace(-2, 6, 8, device="cuda", dtype=torch.float32).view(1, 8, 1, 1) if has_sink else None
    api = SdpaFwdDslSm100(
        has_sink=has_sink,
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=o,
        sample_lse=lse,
        split_kv=1,
        pack_gqa=True,
        cga=2 if d == 256 else 1,
        seq_q_lens_present=True,
        seq_kv_lens_present=True,
    )
    api.check_support()
    api.compile()
    assert api._dense_spec.native is not None
    monkeypatch.setattr(prep, "facts_of_tensor", lambda *args: pytest.fail("standalone native path rebuilt Python facts"))
    monkeypatch.setattr(prep, "bind_dense", lambda *args: pytest.fail("standalone native path used Python binding"))

    def call(scale):
        api.execute(q, k, v, o, lse_tensor=lse, scale_softmax=scale, seq_q_lens=q_lens, seq_kv_lens=kv_lens, sinks=sinks)

    def check(scale):
        scores = torch.matmul(q.double(), k.double().repeat_interleave(4, dim=1).transpose(-1, -2)) * scale
        if has_sink:
            scores = torch.cat((scores, sinks.double().view(1, 8, 1, 1).expand(2, 8, sq, 1)), dim=-1)
        probabilities = scores.softmax(-1)[..., :128]
        expected = torch.matmul(probabilities, v.double().repeat_interleave(4, dim=1))
        torch.testing.assert_close(o.float(), expected.float(), atol=0.005, rtol=0.01)
        torch.testing.assert_close(lse.double(), scores.logsumexp(-1), atol=0.005, rtol=0.001)

    for scale in (d**-0.5, 0.5 * d**-0.5):
        o.fill_(float("nan"))
        lse.fill_(float("nan"))
        torch.cuda.set_sync_debug_mode("error")
        try:
            call(scale)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        check(scale)
        q, k, v = q.clone(), k.clone(), v.clone()
        if has_sink:
            sinks = sinks.clone().add_(0.5)
    if has_sink:
        valid_sink = sinks
        for sinks in (None, valid_sink.bfloat16(), valid_sink.view(-1)[:1], torch.empty(16, device="cuda")[::2]):
            o.fill_(123)
            with pytest.raises(ValueError, match="sink"):
                call(d**-0.5)
            torch.cuda.synchronize()
            assert torch.all(o == 123).item()
        sinks = valid_sink
    saved_lens = q_lens, kv_lens
    for role in ("q", "kv"):
        oversized = torch.ones((3,), device="cuda", dtype=torch.int32)
        q_lens, kv_lens = (oversized, saved_lens[1]) if role == "q" else (saved_lens[0], oversized)
        with pytest.raises(ValueError, match="must have B"):
            call(d**-0.5)
    q_lens, kv_lens = saved_lens
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    capture = torch.cuda.CUDAGraph()
    scale = 0.75 * d**-0.5
    try:
        with torch.cuda.stream(stream):
            call(scale)
        stream.synchronize()
        with torch.cuda.graph(capture, stream=stream):
            call(scale)
        q.add_(0.25)
        v.mul_(0.5)
        if has_sink:
            sinks.add_(1.0)
        o.fill_(float("nan"))
        lse.fill_(float("nan"))
        capture.replay()
        check(scale)
    finally:
        capture.reset()


@pytest.mark.parametrize("d", [64, 256])
def test_native_decode_new_flavors_keep_int64_table_stride(d, monkeypatch, request):
    test_native_dense_graph_fresh_bindings_and_changed_replay(True, d, 1, "bfloat16", monkeypatch, request, wide_tables=True)
