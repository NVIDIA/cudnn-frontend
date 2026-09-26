# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""The prepared THD f16 launch (``cudnn.sdpa.fwd.prepared``) behind ``graph.execute()``.

One THD f16 plan, executed through the graph's normalized VariantPack: the plan is prepared,
capacities come from the caller's observed spans (not the graph's ragged declaration), every
call binds an independent frame, outputs are fully written, degenerate inputs are handled or
rejected before any launch, and execute allocates nothing and never synchronizes.
"""

from __future__ import annotations

import math

import pytest
import torch

import cudnn
from cudnn.sdpa.fwd import prepared as prep_mod
from cudnn.sdpa.fwd.engines import engine_name
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

pytestmark = [pytest.mark.L0]

DEV = torch.device("cuda")


def _thd_graph(b, ql, kl, hq, hk, d, *, ragged_batch_stride=None, causal=True, override_enabled=False, dtype=cudnn.data_type.BFLOAT16):
    """A THD bf16 graph the way FlashInfer declares it: BHSD dims with ragged offsets, cu_seq_len
    lengths, token-major Stats. ``ragged_batch_stride`` mimics FlashInfer's small declared batch
    stride (the declaration's span is then far below the buffer's)."""
    g = cudnn.pygraph(
        io_data_type=dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        is_override_shape_enabled=override_enabled,
    )
    q_bs = ragged_batch_stride if ragged_batch_stride is not None else ql * hq * d
    kv_bs = ragged_batch_stride if ragged_batch_stride is not None else kl * hk * d
    tq = g.tensor(dim=[b, hq, ql, d], stride=[q_bs, d, hq * d, 1], data_type=dtype, name="q")
    tk = g.tensor(dim=[b, hk, kl, d], stride=[kv_bs, d, hk * d, 1], data_type=dtype, name="k")
    tv = g.tensor(dim=[b, hk, kl, d], stride=[kv_bs, d, hk * d, 1], data_type=dtype, name="v")
    i32 = cudnn.data_type.INT32
    t_cu_q = g.tensor(dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=i32, name="cu_q")
    t_cu_kv = g.tensor(dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=i32, name="cu_kv")
    off_q = g.tensor(dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=i32, name="off_q")
    off_kv = g.tensor(dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=i32, name="off_kv")
    off_lse = g.tensor(dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=i32, name="off_lse")
    tq.set_ragged_offset(off_q)
    tk.set_ragged_offset(off_kv)
    tv.set_ragged_offset(off_kv)
    to, ts = g.sdpa(
        name="sdpa",
        q=tq,
        k=tk,
        v=tv,
        generate_stats=True,
        attn_scale=1.0 / math.sqrt(d),
        use_causal_mask=causal,
        use_padding_mask=True,
        cu_seq_len_q=t_cu_q,
        cu_seq_len_kv=t_cu_kv,
        max_total_seq_len_q=b * ql,
        max_total_seq_len_kv=b * kl,
    )
    to.set_output(True).set_dim([b, hq, ql, d]).set_stride([q_bs, d, hq * d, 1]).set_ragged_offset(off_q)
    ts.set_output(True).set_dim([b, hq, ql, 1]).set_stride([ql * hq, 1, hq, 1]).set_data_type(cudnn.data_type.FLOAT).set_ragged_offset(off_lse)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    want = engine_name()
    g.select_plan(next(i for i, n in enumerate(names) if n == want or n.startswith(want + "[")))
    g.check_support()
    g.build_plans()
    return g, dict(q=tq, k=tk, v=tv, o=to, stats=ts, cu_q=t_cu_q, cu_kv=t_cu_kv, off_q=off_q, off_kv=off_kv, off_lse=off_lse)


def _buffers(b, ql, kl, hq, hk, d, seed=0, dtype=torch.bfloat16):
    torch.manual_seed(seed)
    q = torch.randn(b * ql, hq, d, device=DEV, dtype=dtype)
    k = torch.randn(b * kl, hk, d, device=DEV, dtype=dtype)
    v = torch.randn(b * kl, hk, d, device=DEV, dtype=dtype)
    o = torch.empty(b * ql, hq, d, device=DEV, dtype=dtype)
    lse = torch.empty(b * ql, hq, device=DEV, dtype=torch.float32)
    cu_q = (torch.arange(0, b + 1, device=DEV, dtype=torch.int32) * ql).contiguous()
    cu_kv = (torch.arange(0, b + 1, device=DEV, dtype=torch.int32) * kl).contiguous()
    return dict(
        q=q,
        k=k,
        v=v,
        o=o,
        lse=lse,
        cu_q=cu_q,
        cu_kv=cu_kv,
        off_q=(cu_q * hq * d).to(torch.int32),
        off_kv=(cu_kv * hk * d).to(torch.int32),
        off_lse=(cu_q * hq).to(torch.int32),
    )


def _pack(t, bufs):
    return {
        t["q"]: bufs["q"],
        t["k"]: bufs["k"],
        t["v"]: bufs["v"],
        t["o"]: bufs["o"],
        t["stats"]: bufs["lse"],
        t["cu_q"]: bufs["cu_q"],
        t["cu_kv"]: bufs["cu_kv"],
        t["off_q"]: bufs["off_q"],
        t["off_kv"]: bufs["off_kv"],
        t["off_lse"]: bufs["off_lse"],
    }


def _reference(bufs, b, ql, kl, hq, hk, d, causal=True):
    """fp32 causal (or full) attention per sequence with GQA broadcast; returns O (T, H, D) and LSE (T, H)."""
    q, k, v = bufs["q"].float(), bufs["k"].float(), bufs["v"].float()
    o = torch.empty(b * ql, hq, d, device=DEV)
    lse = torch.empty(b * ql, hq, device=DEV)
    g = hq // hk
    for i in range(b):
        qi = q[i * ql : (i + 1) * ql].transpose(0, 1)  # (H, ql, d)
        ki = k[i * kl : (i + 1) * kl].transpose(0, 1).repeat_interleave(g, 0)
        vi = v[i * kl : (i + 1) * kl].transpose(0, 1).repeat_interleave(g, 0)
        s = qi @ ki.transpose(1, 2) / math.sqrt(d)
        if causal:  # use_causal_mask: top-left aligned diagonal
            row = torch.arange(ql, device=DEV).view(-1, 1)
            col = torch.arange(kl, device=DEV).view(1, -1)
            s = s.masked_fill(col > row, float("-inf"))
        lse[i * ql : (i + 1) * ql] = torch.logsumexp(s, dim=-1).transpose(0, 1)
        o[i * ql : (i + 1) * ql] = (torch.softmax(s, dim=-1) @ vi).transpose(0, 1)
    return o, lse


def _plan(g):
    return g._compiled_plans[g._plan_index]


class _Recorder:
    """Records every frame the prepared launch hands to the positional entry."""

    def __init__(self, spec):
        self.spec, self.frames, self._fn = spec, [], spec.fn
        spec.fn = self
        self._native = getattr(spec, "native", None)
        if self._native is not None:
            # Native plans retain the official launch entry at prepare time.
            # Re-prepare after installing this test-only recorder.
            spec.native = cudnn._pybind_module._SdpaThdBinder(spec)

    def __call__(self, *frame):
        self.frames.append(dict(zip(self.spec.order, frame)))
        return self._fn(*frame)

    def restore(self):
        self.spec.fn = self._fn
        if self._native is not None:
            self.spec.native = self._native


@requires_pre_rubin_blackwell
@requires_dsl
def test_thd_f16_plan_is_prepared_and_binds_the_variant_pack():
    b, ql, kl, hq, hk, d = 4, 4, 64, 8, 2, 128
    g, t = _thd_graph(b, ql, kl, hq, hk, d)
    plan = _plan(g)
    assert isinstance(plan._prepared, prep_mod.PreparedThdLaunch)
    assert plan.takes_variant_pack is True
    bufs = _buffers(b, ql, kl, hq, hk, d)
    ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    g.execute(_pack(t, bufs), ws)
    torch.cuda.synchronize()
    o_ref, lse_ref = _reference(bufs, b, ql, kl, hq, hk, d)
    torch.testing.assert_close(bufs["o"].float(), o_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(bufs["lse"], lse_ref, atol=1e-3, rtol=1e-3)


@requires_pre_rubin_blackwell
@requires_dsl
def test_capacity_comes_from_the_observed_span_not_the_ragged_declaration():
    """FlashInfer declares Q with a batch stride far below one batch's rows; the pack re-describes the
    caller's (T, H, D) buffer with that geometry (graph_described). The token capacity must still be
    the buffer's T, or the TMA extent truncates and live rows go unwritten (the first version of
    this path bound 67 instead of 256)."""
    b, ql, kl, hq, hk, d = 64, 4, 64, 8, 2, 128
    g, t = _thd_graph(b, ql, kl, hq, hk, d, ragged_batch_stride=1024)
    plan = _plan(g)
    rec = _Recorder(plan._prepared.spec)
    try:
        bufs = _buffers(b, ql, kl, hq, hk, d)
        ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
        bufs["o"].fill_(float("nan"))
        bufs["lse"].fill_(float("nan"))
        g.execute(_pack(t, bufs), ws)
        torch.cuda.synchronize()
    finally:
        rec.restore()
    assert len(rec.frames) == 1
    problem = rec.frames[0]["problem_size"]
    assert problem[3] == b * ql and problem[4] == b * kl, problem
    assert not torch.isnan(bufs["o"]).any() and not torch.isnan(bufs["lse"]).any(), "poisoned outputs must be fully overwritten"
    o_ref, lse_ref = _reference(bufs, b, ql, kl, hq, hk, d)
    torch.testing.assert_close(bufs["o"].float(), o_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(bufs["lse"], lse_ref, atol=1e-3, rtol=1e-3)


@requires_pre_rubin_blackwell
@requires_dsl
def test_each_call_binds_its_own_buffers():
    b, ql, kl, hq, hk, d = 4, 4, 64, 8, 2, 128
    g, t = _thd_graph(b, ql, kl, hq, hk, d)
    ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    rec = _Recorder(_plan(g)._prepared.spec)
    try:
        a, c = _buffers(b, ql, kl, hq, hk, d, seed=1), _buffers(b, ql, kl, hq, hk, d, seed=2)
        g.execute(_pack(t, a), ws)
        g.execute(_pack(t, c), ws)
        torch.cuda.synchronize()
    finally:
        rec.restore()
    fa, fc = rec.frames
    assert fa["q_ptr"] == a["q"].data_ptr() and fc["q_ptr"] == c["q"].data_ptr() and fa["q_ptr"] != fc["q_ptr"]
    assert fa["o_ptr"] != fc["o_ptr"] and fa["lse_ptr"] != fc["lse_ptr"]
    for bufs in (a, c):
        o_ref, lse_ref = _reference(bufs, b, ql, kl, hq, hk, d)
        torch.testing.assert_close(bufs["o"].float(), o_ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(bufs["lse"], lse_ref, atol=1e-3, rtol=1e-3)


@requires_pre_rubin_blackwell
@requires_dsl
def test_shrinking_lengths_and_zero_capacity():
    """Shorter ragged lengths on the same buffers leave rows past each length untouched (the plan
    reads lengths on the device); an empty Q buffer binds no launch at all."""
    b, ql, kl, hq, hk, d = 4, 8, 64, 8, 2, 128
    g, t = _thd_graph(b, ql, kl, hq, hk, d)
    ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    bufs = _buffers(b, ql, kl, hq, hk, d)
    live = 3  # tokens per sequence actually present: the packed buffer holds b*live live rows, the rest is slack
    cu_q = (torch.arange(0, b + 1, device=DEV, dtype=torch.int32) * live).contiguous()
    bufs["cu_q"], bufs["off_q"], bufs["off_lse"] = cu_q, (cu_q * hq * d).to(torch.int32), (cu_q * hq).to(torch.int32)
    bufs["o"].fill_(7.0)
    bufs["lse"].fill_(7.0)
    g.execute(_pack(t, bufs), ws)
    torch.cuda.synchronize()
    o = bufs["o"].float()
    assert not (o[: b * live] == 7.0).all(dim=-1).any(), "live rows must be written"
    assert (o[b * live :] == 7.0).all(), "rows past the packed total are not the kernel's to write"
    # zero capacity: no Q token addressable -> bind returns None, nothing is launched
    rec = _Recorder(_plan(g)._prepared.spec)
    try:
        empty = dict(
            bufs,
            q=torch.empty(0, hq, d, device=DEV, dtype=torch.bfloat16),
            o=torch.empty(0, hq, d, device=DEV, dtype=torch.bfloat16),
            lse=torch.empty(0, hq, device=DEV),
        )
        g.execute(_pack(t, empty), ws)
        torch.cuda.synchronize()
    finally:
        rec.restore()
    assert rec.frames == []


@requires_pre_rubin_blackwell
@requires_dsl
def test_empty_operands_never_launch_whatever_their_geometry():
    """A zero-element producer spans nothing, wherever its empty axis sits: a strided empty O (whose
    affine span formula would otherwise claim capacity) and an empty interleaved Q slice (whose
    formula goes negative) both bind no launch; nothing is written."""
    b, ql, kl, hq, hk, d = 2, 8, 64, 8, 2, 128
    g, t = _thd_graph(b, ql, kl, hq, hk, d)
    ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    bufs = _buffers(b, ql, kl, hq, hk, d)
    bufs["lse"].fill_(7.0)
    empty_o = torch.empty(b, hq, ql, d, device=DEV, dtype=torch.bfloat16)[:, :, :0, :]  # (2, 8, 0, 128), strides (8192, 1024, 128, 1)
    empty_q = torch.empty(0, 2 * hq, d, device=DEV, dtype=torch.bfloat16)[:, :hq]  # interleaved slice: negative affine span
    rec = _Recorder(_plan(g)._prepared.spec)
    try:
        g.execute(_pack(t, dict(bufs, o=empty_o)), ws)
        g.execute(_pack(t, dict(bufs, q=empty_q, o=empty_q.clone())), ws)
        torch.cuda.synchronize()
    finally:
        rec.restore()
    assert rec.frames == [], "an empty operand must not reach the launch"
    assert (bufs["lse"] == 7.0).all(), "nothing is written when nothing is addressable"


def test_padded_stats_accept_the_declared_view_or_contiguous_storage():
    """Padded Stats: the declared per-batch strides apply over the caller's storage. The declared
    (B, H, S) view, the same allocation as contiguous (B, S, H) or flat storage all bind; a strided
    view that is not the declared layout does not."""
    from types import SimpleNamespace

    B, H, S = 2, 4, 16
    spec = SimpleNamespace(lse_padded=True, lse_stride=(S * H, 1, H), qh=H, s_q_max=S, lse_head_major=False, lse_head_stride=0)  # (B, S, H) storage order

    def facts(shape, strides):
        n = math.prod(shape)
        return prep_mod.BufferFacts(4096, "float32", (2, 0), 1 + sum((e - 1) * st for e, st in zip(shape, strides)) if n else 0, tuple(shape), tuple(strides))

    prep_mod._stats_layout_is_the_compiled_kind(spec, facts((B, H, S, 1), (S * H, 1, H, 1)))  # the declared view
    prep_mod._stats_layout_is_the_compiled_kind(spec, facts((B, S, H), (S * H, H, 1)))  # the storage itself
    prep_mod._stats_layout_is_the_compiled_kind(spec, facts((B * S * H,), (1,)))  # flat storage
    with pytest.raises(ValueError, match="declared"):
        prep_mod._stats_layout_is_the_compiled_kind(
            spec, facts((B, H, S, 1), (2 * S * H, 2 * S, 2, 1))
        )  # a gapped (B, H, S) view: neither the declaration nor storage


@requires_dsl
@pytest.mark.parametrize("rubin", [False, True], ids=["sm100", "sm107"])
@pytest.mark.parametrize("flavor", [(128, 128), (192, 128), (256, 256), (512, 512)], ids=["d128", "d192x128", "d256", "d512"])
def test_every_f16_host_slot_is_bound_by_both_launch_paths(rubin, flavor):
    """Host-side, no GPU: every runtime slot the eight f16 templates declare (SM107's gate slots included)
    is in the dense adapter's vocabulary and in the prepared THD launch's, so a host that grows a slot is
    declined at plan time on every arch, not at first launch on the one arch that carries it."""
    import inspect

    from cudnn.sdpa.fwd import api_dsl, prepared
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    for thd in (False, True):
        mod = api_dsl._load_sm100_kernel_module(flavor, TemplateParams(dtype_qkv=2, thd_varlen=thd, seq_kv_lens_present=thd), rubin=rubin)
        assert mod.EXPLICIT_ABI is True
        slots = {n for n, p in inspect.signature(mod._host).parameters.items() if "Constexpr" not in str(p.annotation)}
        dense = prepared._FILLED_AT_BUILD_DENSE | prepared._FILLED_PER_CALL_DENSE
        thd = prepared._FILLED_AT_BUILD | prepared._FILLED_PER_CALL
        assert slots <= dense, (mod.__name__, sorted(slots - dense))
        assert slots <= thd, (mod.__name__, sorted(slots - thd))
        if "gate_strides" in slots:  # a Tuple slot: both launches must pass a tuple even when the gate is absent
            assert "gate_strides" in prepared._FILLED_AT_BUILD and "gate_strides" in prepared._FILLED_AT_BUILD_DENSE


@requires_pre_rubin_blackwell
@requires_dsl
def test_capture_without_a_handle_stream_records_the_launch():
    """No handle stream: the launch goes to the caller's CURRENT torch stream (the tensor path's rule), so a
    CUDA-graph capture on a side stream records the kernel instead of running it eagerly on the legacy
    stream and leaving the graph empty: outputs are untouched by the capture, replay reproduces the eager
    result, and a second replay follows new inputs."""
    b, ql, kl, hq, hk, d = 8, 4, 256, 8, 2, 128
    g, t = _thd_graph(b, ql, kl, hq, hk, d)
    ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    bufs = _buffers(b, ql, kl, hq, hk, d)
    pack = _pack(t, bufs)
    g.execute(pack, ws)
    torch.cuda.synchronize()
    eager = (bufs["o"].clone(), bufs["lse"].clone())
    bufs["o"].zero_()
    bufs["lse"].zero_()
    side = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side):
        with torch.cuda.graph(graph, stream=side):
            g.execute(pack, ws)
    torch.cuda.synchronize()
    assert (bufs["o"] == 0).all() and (bufs["lse"] == 0).all(), "a capture must record the launch, not run it"
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(bufs["o"], eager[0]) and torch.equal(bufs["lse"], eager[1])
    bufs["q"].copy_(torch.randn_like(bufs["q"]))
    graph.replay()
    torch.cuda.synchronize()
    replayed = (bufs["o"].clone(), bufs["lse"].clone())
    g.execute(pack, ws)
    torch.cuda.synchronize()
    assert torch.equal(replayed[0], bufs["o"]) and torch.equal(replayed[1], bufs["lse"]), "replay must follow the new inputs"


@requires_pre_rubin_blackwell
@requires_dsl
def test_bare_address_and_wrong_device_are_rejected_before_launch():
    b, ql, kl, hq, hk, d = 4, 4, 64, 8, 2, 128
    g, t = _thd_graph(b, ql, kl, hq, hk, d)
    ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    bufs = _buffers(b, ql, kl, hq, hk, d)
    rec = _Recorder(_plan(g)._prepared.spec)
    try:
        with pytest.raises(ValueError, match="bare address"):
            g.execute(_pack(t, dict(bufs, q=bufs["q"].data_ptr())), ws)
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            g.execute(_pack(t, dict(bufs, k=bufs["k"].cpu())), ws)
        with pytest.raises((ValueError, RuntimeError, TypeError)):  # auxiliary roles carry the same device rule
            g.execute(_pack(t, dict(bufs, cu_q=bufs["cu_q"].cpu())), ws)
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            g.execute(_pack(t, dict(bufs, lse=bufs["lse"].cpu())), ws)
    finally:
        rec.restore()
    assert rec.frames == [], "a rejected call must not reach the launch"


@requires_pre_rubin_blackwell
@requires_dsl
def test_zero_kv_clamp_and_runtime_strides(monkeypatch):
    """All-zero KV lengths bind descriptor-only K/V views of live Q/O storage (no private allocation
    at build or execute); K/V sliced from a fused slab bind their runtime token stride; a
    layout TMA cannot express is rejected before launch."""

    def no_private_allocation(*args, **kwargs):
        pytest.fail("prepared launch allocated private device memory")

    monkeypatch.setattr(prep_mod._buffers, "DeviceBuffer", no_private_allocation)
    b, ql, kl, hq, hk, d = 4, 4, 64, 8, 2, 128
    g, t = _thd_graph(b, ql, kl, hq, hk, d)
    ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    bufs = _buffers(b, ql, kl, hq, hk, d)
    spec = _plan(g)._prepared.spec
    rec = _Recorder(spec)
    try:
        zero_kv = dict(bufs, k=torch.empty(0, hk, d, device=DEV, dtype=torch.bfloat16), v=torch.empty(0, hk, d, device=DEV, dtype=torch.bfloat16))
        zero_kv["cu_kv"] = torch.zeros(b + 1, device=DEV, dtype=torch.int32)
        zero_kv["off_kv"] = torch.zeros(b + 1, device=DEV, dtype=torch.int32)
        zero_kv["o"].fill_(float("nan"))  # The descriptor-only V alias must never be read.
        zero_kv["lse"].fill_(float("nan"))
        g.execute(_pack(t, zero_kv), ws)
        torch.cuda.synchronize()
    finally:
        rec.restore()
    frame = rec.frames[-1]
    assert frame["problem_size"][4] == 1 and frame["v_ptr"] == frame["o_ptr"] and frame["k_ptr"] == frame["q_ptr"]
    assert torch.count_nonzero(zero_kv["o"]) == 0
    assert torch.isneginf(zero_kv["lse"]).all()
    # K / V sliced out of a fused (T, 2*H_kv, D) slab: a runtime token stride of 2*H_kv*D, bound as such
    slab = torch.randn(b * kl, 2 * hk, d, device=DEV, dtype=torch.bfloat16)
    k_view, v_view = slab[:, :hk], slab[:, hk:]
    fused = dict(bufs, k=k_view, v=v_view)
    fused["o"].fill_(float("nan"))
    rec = _Recorder(spec)
    try:
        g.execute(_pack(t, fused), ws)
        torch.cuda.synchronize()
    finally:
        rec.restore()
    frame = rec.frames[-1]
    assert frame["k_strides"] == (2 * hk * d, 2 * hk * d, d) and frame["v_strides"] == (2 * hk * d, 2 * hk * d, d), (frame["k_strides"], frame["v_strides"])
    assert frame["k_ptr"] == k_view.data_ptr() and frame["v_ptr"] == v_view.data_ptr()
    o_ref, lse_ref = _reference(dict(fused, k=k_view.contiguous(), v=v_view.contiguous()), b, ql, kl, hq, hk, d)
    torch.testing.assert_close(fused["o"].float(), o_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(fused["lse"], lse_ref, atol=1e-3, rtol=1e-3)
    # a head stride below the head dim is not TMA-expressible: declined before launch
    rec = _Recorder(spec)
    try:
        with pytest.raises(ValueError, match="head stride"):
            g.execute(
                _pack(
                    t,
                    dict(
                        bufs, k=torch.randn(b * kl, hk, 2 * d, device=DEV, dtype=torch.bfloat16)[:, :, :d].as_strided((b * kl, hk, d), (hk * 2 * d, d // 2, 1))
                    ),
                ),
                ws,
            )
    finally:
        rec.restore()
    assert rec.frames == []


class _Tripwire:
    """Fails the test if the adapter, the heuristics/lowering or the compiler is re-entered during execute."""

    def __init__(self):
        import cudnn.frost.compiled_cache as cc
        import cudnn.sdpa.fwd.engines as eng
        from cudnn.sdpa.fwd import api_dsl

        self._targets = [(cc, "compile_cached"), (eng, "lower_dsl_prefill"), (api_dsl.SdpaFwdDslSm100, "execute"), (api_dsl.SdpaFwdDslSm100, "compile")]
        self._saved = [(m, n, getattr(m, n)) for m, n in self._targets]

    def __enter__(self):
        for m, n, _ in self._saved:

            def trip(*a, _n=n, **k):
                raise AssertionError(f"{_n} must not run during a prepared execute")

            setattr(m, n, trip)
        return self

    def __exit__(self, *exc):
        for m, n, f in self._saved:
            setattr(m, n, f)


@requires_pre_rubin_blackwell
@requires_dsl
def test_bounded_override_through_graph_execute():
    """The decisive case: a real shape override through public graph.execute() on the same prepared
    artifact — fewer sequences than declared — then back to the declared geometry; the frame carries
    the override, outputs match an independent reference, and neither the adapter, the heuristics
    nor the compiler run. An override outside the domain (more sequences than declared) is
    rejected before any output seed or launch."""
    b, ql, kl, hq, hk, d = 8, 4, 64, 8, 2, 128
    g, t = _thd_graph(b, ql, kl, hq, hk, d, override_enabled=True)
    plan = _plan(g)
    ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    spec = plan._prepared.spec
    rec = _Recorder(spec)
    try:
        # declared geometry first
        full = _buffers(b, ql, kl, hq, hk, d, seed=3)
        with _Tripwire():
            g.execute(_pack(t, full), ws)
        torch.cuda.synchronize()
        o_ref, lse_ref = _reference(full, b, ql, kl, hq, hk, d)
        torch.testing.assert_close(full["o"].float(), o_ref, atol=2e-2, rtol=2e-2)
        # override: b2 = 3 sequences, every per-batch / per-token operand re-described through the public API
        b2 = 3
        small = _buffers(b2, ql, kl, hq, hk, d, seed=4)
        small["o"].fill_(float("nan"))
        small["lse"].fill_(float("nan"))
        uids = [t[n].get_uid() for n in ("q", "k", "v", "o", "stats", "cu_q", "cu_kv", "off_q", "off_kv", "off_lse")]
        shapes = [[b2, hq, ql, d], [b2, hk, kl, d], [b2, hk, kl, d], [b2, hq, ql, d], [b2, hq, ql, 1]] + [[b2 + 1, 1, 1, 1]] * 5
        strides = [[ql * hq * d, d, hq * d, 1], [kl * hk * d, d, hk * d, 1], [kl * hk * d, d, hk * d, 1], [ql * hq * d, d, hq * d, 1], [ql * hq, 1, hq, 1]] + [
            [1, 1, 1, 1]
        ] * 5
        with _Tripwire():
            g.execute(_pack(t, small), ws, override_uids=uids, override_shapes=shapes, override_strides=strides)
        torch.cuda.synchronize()
        frame = rec.frames[-1]
        assert frame["problem_size"][0] == b2 and frame["problem_size"][3] == b2 * ql and frame["problem_size"][4] == b2 * kl, frame["problem_size"]
        assert not torch.isnan(small["o"]).any() and not torch.isnan(small["lse"]).any()
        o_ref, lse_ref = _reference(small, b2, ql, kl, hq, hk, d)
        torch.testing.assert_close(small["o"].float(), o_ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(small["lse"], lse_ref, atol=1e-3, rtol=1e-3)
        # back to the declared geometry on the same plan
        again = _buffers(b, ql, kl, hq, hk, d, seed=5)
        with _Tripwire():
            g.execute(_pack(t, again), ws)
        torch.cuda.synchronize()
        assert rec.frames[-1]["problem_size"][0] == b
        o_ref, _ = _reference(again, b, ql, kl, hq, hk, d)
        torch.testing.assert_close(again["o"].float(), o_ref, atol=2e-2, rtol=2e-2)
        # outside the domain: more sequences than the plan was prepared for -> rejected, no seed, no launch
        n_before = len(rec.frames)
        big = _buffers(b + 2, ql, kl, hq, hk, d, seed=6)
        big["lse"].fill_(7.0)
        shapes_big = [[b + 2, hq, ql, d], [b + 2, hk, kl, d], [b + 2, hk, kl, d], [b + 2, hq, ql, d], [b + 2, hq, ql, 1]] + [[b + 3, 1, 1, 1]] * 5
        with pytest.raises(ValueError, match="prepared for"):
            g.execute(_pack(t, big), ws, override_uids=uids, override_shapes=shapes_big, override_strides=strides)
        assert len(rec.frames) == n_before and (big["lse"] == 7.0).all()
        # the Stats layout is fixed by the artifact (token-major here): an override describing another layout is rejected
        n_before = len(rec.frames)
        odd = _buffers(b, ql, kl, hq, hk, d, seed=7)
        odd["lse"].fill_(7.0)
        with pytest.raises(ValueError, match="token-major lse_tensor"):
            g.execute(_pack(t, odd), ws, override_uids=[t["stats"].get_uid()], override_shapes=[[b, hq, ql, 1]], override_strides=[[ql * hq, ql, 1, 1]])
        assert len(rec.frames) == n_before and (odd["lse"] == 7.0).all()
    finally:
        rec.restore()


@requires_pre_rubin_blackwell
@requires_dsl
def test_execute_allocates_nothing_and_never_synchronizes():
    """Warmed-path check: after one execute, further executes add no torch allocation and trigger no
    torch-visible synchronization. Driver-side allocation is excluded by construction (the spec
    allocates its resources at build; see test_zero_kv_clamp_and_stride_mismatch)."""
    b, ql, kl, hq, hk, d = 4, 4, 64, 8, 2, 128
    g, t = _thd_graph(b, ql, kl, hq, hk, d)
    ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    bufs = _buffers(b, ql, kl, hq, hk, d)
    pack = _pack(t, bufs)
    g.execute(pack, ws)  # warm: dummies, indices
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(5):
            g.execute(pack, ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before


@requires_pre_rubin_blackwell
@requires_dsl
def test_graph_and_standalone_execute_bind_the_same_frame():
    """The graph plan and the adapter's execute() are the same core: identical frames, up to the stream."""
    b, ql, kl, hq, hk, d = 4, 4, 64, 8, 2, 128
    g, t = _thd_graph(b, ql, kl, hq, hk, d)
    ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    bufs = _buffers(b, ql, kl, hq, hk, d)
    plan = _plan(g)
    rec = _Recorder(plan._prepared.spec)
    try:
        g.execute(_pack(t, bufs), ws)
        torch.cuda.synchronize()
        prepared_frame = rec.frames[-1]
        plan._prepared, plan.takes_variant_pack = None, False
        try:
            g.execute(_pack(t, bufs), ws)
            torch.cuda.synchronize()
        finally:
            plan._prepared, plan.takes_variant_pack = rec.spec and plan._compiled.prepared, True
        standalone_frame = rec.frames[-1]
    finally:
        rec.restore()
    assert len(rec.frames) == 2
    for name in rec.spec.order:
        if name == "stream":
            continue
        assert str(prepared_frame[name]) == str(standalone_frame[name]), name


# --- dense (padded) f16 through the same prepared machinery ------------------------------------------------


def _dense_graph(
    b,
    h,
    hk,
    s_q,
    s_kv,
    d,
    *,
    causal=True,
    bshd_storage=True,
    d_v=None,
    bottom_right=False,
    split_kv=1,
    stats=True,
    stats_log2=False,
    o_stride=None,
    override_enabled=False,
):
    """A dense bf16 graph declared in BSHD storage (the zero-copy layout) or BHSD (which the tensor path
    repacks and the prepared launch therefore declines)."""
    d_v = d if d_v is None else d_v
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        is_override_shape_enabled=override_enabled,
    )

    def st(hh, s, dd):
        return [s * hh * dd, dd, hh * dd, 1] if bshd_storage else [hh * s * dd, s * dd, dd, 1]

    tq = g.tensor(dim=[b, h, s_q, d], stride=st(h, s_q, d), data_type=cudnn.data_type.BFLOAT16, name="q")
    tk = g.tensor(dim=[b, hk, s_kv, d], stride=st(hk, s_kv, d), data_type=cudnn.data_type.BFLOAT16, name="k")
    tv = g.tensor(dim=[b, hk, s_kv, d_v], stride=st(hk, s_kv, d_v), data_type=cudnn.data_type.BFLOAT16, name="v")
    mask = dict(use_causal_mask_bottom_right=True) if (causal and bottom_right) else dict(use_causal_mask=causal)
    to, ts = g.sdpa(name="sdpa", q=tq, k=tk, v=tv, generate_stats=stats, stats_use_log2=stats_log2, attn_scale=1.0 / math.sqrt(d), **mask)
    to.set_output(True).set_dim([b, h, s_q, d_v]).set_stride(st(h, s_q, d_v) if o_stride is None else o_stride)
    if stats:
        ts.set_output(True).set_dim([b, h, s_q, 1]).set_stride([h * s_q, s_q, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    want = engine_name()
    # the FROST row's unsplit plan (the heuristics may rank a split first on long KV)
    idx = next((i for i, n in enumerate(names) if (n == want or n.startswith(want + "[")) and g.plans[i].knobs.split_kv == 1), None)
    if idx is None:
        idx = next(i for i, n in enumerate(names) if n == want or n.startswith(want + "["))
    if split_kv > 1:
        from dataclasses import replace

        chosen = g.plans[idx]
        g.create_execution_plan(chosen.engine_id, replace(chosen.knobs, split_kv=split_kv))
        idx = len(g.plans) - 1
    g.select_plan(idx)
    g.check_support()
    g.build_plans()
    return g, dict(q=tq, k=tk, v=tv, o=to, stats=ts)


def _dense_buffers(b, h, hk, s_q, s_kv, d, *, seed=0):
    torch.manual_seed(seed)

    def mk(hh, s):
        return torch.randn(b, s, hh, d, device=DEV, dtype=torch.bfloat16).transpose(1, 2)  # BSHD storage, (B, H, S, D) view

    return dict(
        q=mk(h, s_q),
        k=mk(hk, s_kv),
        v=mk(hk, s_kv),
        o=torch.empty(b, s_q, h, d, device=DEV, dtype=torch.bfloat16).transpose(1, 2),
        lse=torch.empty(b, h, s_q, 1, device=DEV, dtype=torch.float32),
    )


def _dense_reference(bufs, causal=True):
    q, k, v = bufs["q"].float(), bufs["k"].float(), bufs["v"].float()
    b, h, s_q, d = q.shape
    hk = k.shape[1]
    k = k.repeat_interleave(h // hk, dim=1)
    v = v.repeat_interleave(h // hk, dim=1)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)
    if causal:  # the graph's use_causal_mask: top-left diagonal
        mask = torch.ones(s_q, k.shape[2], device=DEV, dtype=torch.bool).tril(diagonal=0)
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v), torch.logsumexp(scores, dim=-1)


def _dense_pack(t, bufs):
    return {t["q"]: bufs["q"], t["k"]: bufs["k"], t["v"]: bufs["v"], t["o"]: bufs["o"], t["stats"]: bufs["lse"]}


@requires_pre_rubin_blackwell
@requires_dsl
@pytest.mark.parametrize("s_q", [1, 4])
def test_decode_d128_prepared_rebind_and_capture(s_q, monkeypatch):
    """The small-Q decode tile uses the prepared binder and remains correct with new
    buffers, an explicit non-current stream and changed-input CUDA-graph replay."""

    def no_private_allocation(*args, **kwargs):
        pytest.fail("prepared launch allocated private device memory")

    monkeypatch.setattr(prep_mod._buffers, "DeviceBuffer", no_private_allocation)
    b, h, hk, sk, d = 4, 8, 2, 128, 128
    g, t = _dense_graph(b, h, hk, s_q, sk, d, causal=False)
    plan = _plan(g)
    assert plan._compiled.kernel_template == "decode_d128_f16"
    assert isinstance(plan._prepared, prep_mod.PreparedDenseLaunch)
    ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    stream = torch.cuda.Stream()
    handle = cudnn.create_handle()
    cudnn.set_stream(handle, stream.cuda_stream)

    def check(bufs):
        o_ref, lse_ref = _dense_reference(bufs, causal=False)
        torch.testing.assert_close(bufs["o"].float(), o_ref, atol=5e-2, rtol=3e-2)
        torch.testing.assert_close(bufs["lse"].squeeze(-1), lse_ref, atol=5e-3, rtol=1e-3)

    for seed in (11, 12):
        bufs = _dense_buffers(b, h, hk, s_q, sk, d, seed=seed)
        bufs["o"].fill_(float("nan"))
        bufs["lse"].fill_(float("nan"))
        stream.wait_stream(torch.cuda.current_stream())
        mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            g.execute(_dense_pack(t, bufs), ws, handle=handle)
        finally:
            torch.cuda.set_sync_debug_mode(mode)
        stream.synchronize()
        check(bufs)

    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture, stream=stream):
        g.execute(_dense_pack(t, bufs), ws, handle=handle)
    bufs["q"].mul_(0.75)
    bufs["o"].fill_(float("nan"))
    bufs["lse"].fill_(float("nan"))
    capture.replay()
    torch.cuda.synchronize()
    check(bufs)


@requires_pre_rubin_blackwell
@requires_dsl
def test_dense_f16_plan_is_prepared_and_matches_reference():
    """A dense bf16 graph declared in BSHD storage executes through the prepared dense launch (no
    tensor arm, no copies) and reproduces the reference O / LSE; a BHSD-declared graph (a layout the
    tensor path repacks) keeps the tensor arm."""
    b, h, hk, s_q, s_kv, d = 2, 8, 2, 256, 512, 128
    g, t = _dense_graph(b, h, hk, s_q, s_kv, d)
    plan = _plan(g)
    assert plan._prepared is not None and plan.takes_variant_pack
    assert type(plan._prepared).__name__ == "PreparedDenseLaunch"
    bufs = _dense_buffers(b, h, hk, s_q, s_kv, d)
    ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    rec = _Recorder(plan._prepared.spec)
    try:
        g.execute(_dense_pack(t, bufs), ws)
        torch.cuda.synchronize()
    finally:
        rec.restore()
    assert len(rec.frames) == 1
    fr = rec.frames[0]
    assert fr["problem_size"] == (b, h, hk, s_q, s_kv, 0) and fr["q_strides"] == (s_q * h * d, h * d, d)
    o_ref, lse_ref = _dense_reference(bufs)
    torch.testing.assert_close(bufs["o"].float(), o_ref, atol=5e-2, rtol=3e-2)
    torch.testing.assert_close(bufs["lse"].squeeze(-1), lse_ref, atol=5e-2, rtol=3e-2)
    g2, t2 = _dense_graph(b, h, hk, s_q, s_kv, d, bshd_storage=False)
    assert _plan(g2)._prepared is None, "a BHSD-declared graph is repacked by the tensor path, not bound zero-copy"
    bhsd = {n: bufs[n].contiguous() for n in ("q", "k", "v")}
    o2, lse2 = torch.empty(b, h, s_q, d, device=DEV, dtype=torch.bfloat16), torch.empty(b, h, s_q, 1, device=DEV, dtype=torch.float32)
    g2.execute(
        {t2["q"]: bhsd["q"], t2["k"]: bhsd["k"], t2["v"]: bhsd["v"], t2["o"]: o2, t2["stats"]: lse2},
        torch.empty(max(g2.get_workspace_size(), 1), device=DEV, dtype=torch.uint8),
    )
    torch.cuda.synchronize()
    assert torch.equal(o2, bufs["o"]) and torch.equal(lse2, bufs["lse"]), "the prepared dense launch and the tensor arm must agree bit for bit"


@requires_pre_rubin_blackwell
@requires_dsl
def test_dense_prepared_runs_a_smaller_batch_and_rejects_out_of_envelope():
    """The dense binder reads B and S from the operands: a smaller batch runs on the same artifact
    (bounded override through the public API), a larger one is rejected before launch."""
    b, h, hk, s_q, s_kv, d = 4, 8, 2, 128, 256, 128
    g, t = _dense_graph(b, h, hk, s_q, s_kv, d)
    plan = _plan(g)
    ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    uids = [t[n].get_uid() for n in ("q", "k", "v", "o", "stats")]

    def geometry(bb):
        shapes = [[bb, h, s_q, d], [bb, hk, s_kv, d], [bb, hk, s_kv, d], [bb, h, s_q, d], [bb, h, s_q, 1]]
        strides = [[s_q * h * d, d, h * d, 1], [s_kv * hk * d, d, hk * d, 1], [s_kv * hk * d, d, hk * d, 1], [s_q * h * d, d, h * d, 1], [h * s_q, s_q, 1, 1]]
        return dict(override_uids=uids, override_shapes=shapes, override_strides=strides)

    small = _dense_buffers(2, h, hk, s_q, s_kv, d, seed=1)
    rec = _Recorder(plan._prepared.spec)
    try:
        g.execute(_dense_pack(t, small), ws, **geometry(2))
        torch.cuda.synchronize()
        assert rec.frames[-1]["problem_size"][0] == 2
        o_ref, _ = _dense_reference(small)
        torch.testing.assert_close(small["o"].float(), o_ref, atol=5e-2, rtol=3e-2)
        big = _dense_buffers(8, h, hk, s_q, s_kv, d, seed=2)
        with pytest.raises(ValueError, match="envelope"):
            g.execute(_dense_pack(t, big), ws, **geometry(8))
    finally:
        rec.restore()
    assert len(rec.frames) == 1, "the rejected call must not reach the launch"


@requires_pre_rubin_blackwell
@requires_dsl
def test_dense_prepared_keeps_the_compiled_kv_tail_contract():
    """A smaller geometry must still be in the artifact's supported domain, not just inside the
    allocation envelope: an unmasked, unpadded d128 plan (TILE_N 128) visits whole KV tiles only, so
    overriding S_kv 256 -> 129 is rejected before launch while 256 -> 128 runs and matches the reference."""
    b, h, hk, s_q, s_kv, d = 2, 8, 2, 128, 256, 128
    g, t = _dense_graph(b, h, hk, s_q, s_kv, d, causal=False)
    plan = _plan(g)
    assert type(plan._prepared).__name__ == "PreparedDenseLaunch"
    ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    bufs = _dense_buffers(b, h, hk, s_q, s_kv, d)

    def kv_override(s):
        return dict(
            override_uids=[t["k"].get_uid(), t["v"].get_uid()],
            override_shapes=[[b, hk, s, d], [b, hk, s, d]],
            override_strides=[[s_kv * hk * d, d, hk * d, 1]] * 2,  # the allocation's strides: a prefix of each sequence
        )

    rec = _Recorder(plan._prepared.spec)
    try:
        with pytest.raises(ValueError, match="multiple of 128"):
            g.execute(_dense_pack(t, bufs), ws, **kv_override(129))
        assert rec.frames == [], "the partial tile must be refused before any launch"
        g.execute(_dense_pack(t, bufs), ws, **kv_override(128))
        torch.cuda.synchronize()
        assert rec.frames[-1]["problem_size"] == (b, h, hk, s_q, 128, 0)
    finally:
        rec.restore()
    prefix = dict(bufs, k=bufs["k"][:, :, :128], v=bufs["v"][:, :, :128])
    o_ref, lse_ref = _dense_reference(prefix, causal=False)
    torch.testing.assert_close(bufs["o"].float(), o_ref, atol=5e-2, rtol=3e-2)
    torch.testing.assert_close(bufs["lse"].squeeze(-1), lse_ref, atol=5e-2, rtol=3e-2)


@requires_pre_rubin_blackwell
@requires_dsl
def test_dense_prepared_pins_extents_a_shape_dependent_lowering_read():
    """The d192 lowering rewrites a SQUARE bottom-right mask as top-left (equal only while S_q == S_kv):
    such an artifact serves exactly the declared extents, so a rectangular override is rejected before
    launch; the declared geometry runs. A d128 plan has no such canonicalization and keeps runtime extents."""
    from cudnn.sdpa.fwd.config_sm100 import d192_square_br_as_tl

    b, h, hk, s, d, d_v = 1, 4, 1, 4224, 192, 128  # 4096 < S <= 8192: the rewrite's range
    g, t = _dense_graph(b, h, hk, s, s, d, causal=True, bottom_right=True, d_v=d_v)
    plan = _plan(g)
    assert type(plan._prepared).__name__ == "PreparedDenseLaunch"
    spec = plan._prepared.spec
    api = plan._executor.__self__ if hasattr(plan, "_executor") else None  # informative only
    assert spec.shape_fixed, "the square bottom-right rewrite must be recorded"
    assert d192_square_br_as_tl is not None
    ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    torch.manual_seed(3)
    q = torch.randn(b, s, h, d, device=DEV, dtype=torch.bfloat16).transpose(1, 2)
    k = torch.randn(b, s, hk, d, device=DEV, dtype=torch.bfloat16).transpose(1, 2)
    v = torch.randn(b, s, hk, d_v, device=DEV, dtype=torch.bfloat16).transpose(1, 2)
    o = torch.empty(b, s, h, d_v, device=DEV, dtype=torch.bfloat16).transpose(1, 2)
    lse = torch.empty(b, h, s, 1, device=DEV, dtype=torch.float32)
    pack = {t["q"]: q, t["k"]: k, t["v"]: v, t["o"]: o, t["stats"]: lse}
    rec = _Recorder(spec)
    try:
        uids = [t[n].get_uid() for n in ("q", "k", "v", "o", "stats")]
        s_q2, s_kv2 = 128, 256
        shapes = [[b, h, s_q2, d], [b, hk, s_kv2, d], [b, hk, s_kv2, d_v], [b, h, s_q2, d_v], [b, h, s_q2, 1]]
        strides = [[s * h * d, d, h * d, 1], [s * hk * d, d, hk * d, 1], [s * hk * d_v, d_v, hk * d_v, 1], [s * h * d_v, d_v, h * d_v, 1], [h * s, s, 1, 1]]
        with pytest.raises(ValueError, match="lowered for exactly"):
            g.execute(pack, ws, override_uids=uids, override_shapes=shapes, override_strides=strides)
        assert rec.frames == []
        g.execute(pack, ws)
        torch.cuda.synchronize()
        assert rec.frames[-1]["problem_size"] == (b, h, hk, s, s, 0)
    finally:
        rec.restore()
    g2, _ = _dense_graph(2, 8, 2, 256, 512, 128)
    assert not _plan(g2)._prepared.spec.shape_fixed


def test_dense_layout_rule_covers_every_tma_global_stride_and_the_lse_aliasing():
    """Host-side: the shared dense layout rule holds the batch stride to the 16-byte TMA rule like the seq
    and head strides (the DSL floors a misaligned stride to TMA units, so admission would mis-address);
    extent-1 axes are canonicalized before the rule; the Stats binder refuses aliasing strides."""
    from cudnn.sdpa.fwd.config_sm100 import canonical_bhsd_strides, dense_bind_strides

    b, h, s, d = 2, 8, 128, 128
    compact = (s * h * d, d, h * d, 1)
    assert dense_bind_strides((b, h, s, d), compact, 2) == (s * h * d, h * d, d)
    padded_ok = (s * h * d + 8, d, h * d, 1)  # 16 bytes of batch padding (bf16)
    assert dense_bind_strides((b, h, s, d), padded_ok, 2) == (s * h * d + 8, h * d, d)
    padded_bad = (s * h * d + 4, d, h * d, 1)  # 8 bytes: not a TMA global stride
    assert dense_bind_strides((b, h, s, d), padded_bad, 2) is None
    # singleton axes: a one-KV-head K keeps its allocation's head stride, an S=1 Q its seq stride
    assert canonical_bhsd_strides((1, 1, s, d), (99999, 77777, d, 1)) == (s * d, d, d, 1)
    assert dense_bind_strides((1, 1, s, d), (99999, 77777, d, 1), 2) == (s * d, d, d)
    assert dense_bind_strides((3, 4, 1, 512), (4 * 512, 512, 512, 1), 2) == (4 * 512, 4 * 512, 512)
    assert prep_mod._covering((2, 8, 128), (8 * 128, 128, 1)) and prep_mod._covering((2, 8, 128), (8 * 128, 1, 8))
    assert not prep_mod._covering((2, 8, 128), (8 * 128, 64, 1)), "head stride 64 aliases rows 64.. of the previous head"


def test_paged_pools_must_be_the_compiled_in_page_layout_kind():
    """Host-side: the paged binder refuses a pool whose in-page (row, head) order differs from the one the
    artifact was compiled for (``paged_hnd`` orders the TMA descriptors), for K and for V alike."""
    from types import SimpleNamespace

    kh, ps, d, n_pages, b, max_pages = 2, 16, 128, 8, 2, 4
    spec = SimpleNamespace(paged=True, paged_hnd=False, page_size=ps, kh=kh, d_qk=d, d_v=d, device_index=0)
    order = ["k_ptr", "v_ptr", "k_strides", "v_strides", "block_table_ptr", "block_table_v_ptr", "table_strides", "n_pages"]
    ix = {n: i for i, n in enumerate(order)}

    def pool(hnd):
        # container (n_pages, KH, page_size, D): NHD = (page, row, head, d) storage; HND = heads outermost within the page
        strides = (kh * ps * d, d, kh * d, 1) if not hnd else (kh * ps * d, ps * d, d, 1)
        return prep_mod.BufferFacts(4096, "bfloat16", (2, 0), n_pages * kh * ps * d, (n_pages, kh, ps, d), strides)

    table = prep_mod.BufferFacts(8192, "int32", (2, 0), b * max_pages, (b, 1, max_pages, 1), (max_pages, max_pages, 1, 1))
    facts = dict(block_table=table, block_table_v=table)
    frame = [None] * len(order)
    assert prep_mod._bind_paged_kv(spec, frame, ix, facts, pool(False), pool(False), b) == max_pages * ps
    for k_hnd, v_hnd in ((True, False), (False, True)):
        with pytest.raises(ValueError, match="compiled for NHD"):
            prep_mod._bind_paged_kv(spec, list(frame), ix, facts, pool(k_hnd), pool(v_hnd), b)


@requires_pre_rubin_blackwell
@requires_dsl
@pytest.mark.parametrize("d", [128, 256])
@pytest.mark.parametrize("stats,stats_log2", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("output_layout", ["bshd", "bhsd_padded", "bhsd_offset"])
def test_split_prepared_strided_output_rebind_and_capture(d, stats, stats_log2, output_layout, monkeypatch):
    """Split execution binds workspace per call and writes the declared O without a copy kernel."""
    b, h, hk, sq, sk = 2, 8, 2, 4, 1024
    stride = (h * sq * (d + 8), sq * (d + 8), d + 8, 1) if output_layout != "bshd" else (sq * h * d, d, h * d, 1)
    g, t = _dense_graph(b, h, hk, sq, sk, d, causal=False, split_kv=2, stats=stats, stats_log2=stats_log2, o_stride=stride)
    plan = _plan(g)
    assert isinstance(plan._prepared, prep_mod.PreparedDenseLaunch), "split plans must not fall back to the tensor adapter"
    assert plan._prepared.spec.combine is not None
    assert plan._compiled.kernel_template == f"decode_d{d}_f16"
    workspaces = [torch.empty(g.get_workspace_size(), dtype=torch.uint8, device=DEV) for _ in range(2)]
    bufs = _dense_buffers(b, h, hk, sq, sk, d, seed=123)
    backing = torch.full((b, h, sq, d + 8), 42.0, dtype=torch.bfloat16, device=DEV) if output_layout != "bshd" else None
    if backing is not None:
        if output_layout == "bhsd_offset":
            storage = torch.full((backing.numel() + 1,), 42.0, dtype=backing.dtype, device=DEV)
            backing = storage[1:].view(backing.shape)
        bufs["o"] = backing[..., :d]
    pack = {t[n]: bufs[n] for n in ("q", "k", "v", "o")}
    if stats:
        pack[t["stats"]] = bufs["lse"]
    stream = torch.cuda.Stream()
    handle = cudnn.create_handle()
    cudnn.set_stream(handle, stream.cuda_stream)

    def check():
        q, k, v = (bufs[n].double() for n in ("q", "k", "v"))
        scores = q @ k.repeat_interleave(h // hk, dim=1).transpose(-1, -2) / math.sqrt(d)
        ref = scores.softmax(-1) @ v.repeat_interleave(h // hk, dim=1)
        torch.testing.assert_close(bufs["o"].double(), ref, atol=5e-3, rtol=3e-2)
        if stats:
            ref_lse = scores.logsumexp(-1) * (math.log2(math.e) if stats_log2 else 1.0)
            torch.testing.assert_close(bufs["lse"].squeeze(-1).double(), ref_lse, atol=1e-4, rtol=1e-4)
        if backing is not None:
            assert torch.all(backing[..., d:] == 42.0), "combine must preserve padding beyond D"

    def forbidden(*args, **kwargs):
        raise AssertionError("prepared split must not use tensor adapter, copies or execution-time compilation")

    monkeypatch.setattr(plan._compiled, "execute_resolved", forbidden)
    import cutlass.cute as cute

    monkeypatch.setattr(cute, "compile", forbidden)
    for ws in workspaces:
        bufs["q"].mul_(0.75)
        bufs["o"].fill_(float("nan"))
        bufs["lse"].fill_(float("nan"))
        stream.wait_stream(torch.cuda.current_stream())
        mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            g.execute(pack, ws, handle=handle)
        finally:
            torch.cuda.set_sync_debug_mode(mode)
        stream.synchronize()
        check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        g.execute(pack, workspaces[1], handle=handle)
    bufs["q"].mul_(0.5)
    bufs["o"].fill_(float("nan"))
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    check()
    with pytest.raises(ValueError, match="workspace"):
        g.execute(pack, workspaces[0][:1], handle=handle)
    with pytest.raises(ValueError, match="aligned"):
        g.execute(pack, torch.empty(g.get_workspace_size() + 1, dtype=torch.uint8, device=DEV)[1:], handle=handle)
    with pytest.raises(ValueError, match="CUDA device"):
        g.execute(pack, torch.empty(g.get_workspace_size(), dtype=torch.uint8), handle=handle)
    # A valid smaller allocation cannot change the fixed split-workspace geometry.
    with pytest.raises(ValueError, match="declared"):
        g.execute(pack, workspaces[0], override_uids=[t["q"].get_uid()], override_shapes=[[1, h, sq, d]], override_strides=[bufs["q"].stride()])


@requires_pre_rubin_blackwell
@requires_dsl
@pytest.mark.parametrize("kh", [1, 2])
@pytest.mark.parametrize("k_hnd,v_hnd", [(False, False), (True, True), (False, True), (True, False)])
def test_paged_adapter_rejects_mixed_pool_layout_at_support_check(kh, k_hnd, v_hnd):
    """One compiled page-layout specialization must cover both pools, including singleton KH."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    b, h, sq, d, page_size, n_pages = 2, 8, 4, 128, 16, 8
    q = torch.empty((b, sq, h, d), dtype=torch.bfloat16, device=DEV).transpose(1, 2)
    o = torch.empty_like(q)

    def pool(hnd):
        if hnd:
            return torch.empty((n_pages, kh, page_size, d), dtype=q.dtype, device=DEV)
        return torch.empty((n_pages, page_size, kh, d), dtype=q.dtype, device=DEV).transpose(1, 2)

    api = SdpaFwdDslSm100(
        sample_q=q,
        sample_k=pool(k_hnd),
        sample_v=pool(v_hnd),
        sample_o=o,
        seq_kv_lens_present=True,
        paged_page_size=page_size,
        paged_max_seq_len_kv=64,
        split_kv=1,
        pack_gqa=False,
    )
    if k_hnd == v_hnd:
        assert api.check_support()
    else:
        with pytest.raises(NotImplementedError, match="paged K and V pools must use the same HND or NHD layout kind"):
            api.check_support()


@requires_pre_rubin_blackwell
@requires_dsl
@pytest.mark.parametrize("d", [128, 256])
def test_prepared_split_survives_collecting_another_plan_during_capture(d):
    """A compiled, abandoned plan may be collected inside another plan's capture."""
    import gc
    import weakref

    b, h, hk, sq, sk = 2, 8, 2, 4, 256
    graph, tensors = _dense_graph(b, h, hk, sq, sk, d, causal=False, split_kv=2)
    assert isinstance(_plan(graph)._prepared, prep_mod.PreparedDenseLaunch)
    bufs = _dense_buffers(b, h, hk, sq, sk, d)
    workspace = torch.empty(max(graph.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    graph.execute(_dense_pack(tensors, bufs), workspace)
    torch.cuda.synchronize()
    reference_o, reference_lse = _dense_reference(bufs, causal=False)
    capture, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    previous_stream = torch.cuda.current_stream()
    stream.wait_stream(previous_stream)
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        victim, _ = _dense_graph(b, h, hk, sq, sk, d, causal=False, split_kv=2)
        victim._lifetime_cycle = victim
        victim_ref = weakref.ref(victim)
        del victim
        assert victim_ref() is not None
        with torch.cuda.graph(capture, stream=stream, capture_error_mode="global"):
            gc.collect()
            assert victim_ref() is None
            graph.execute(_dense_pack(tensors, bufs), workspace)
        bufs["o"].fill_(float("nan"))
        bufs["lse"].fill_(float("nan"))
        capture.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(bufs["o"].float(), reference_o, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(bufs["lse"].squeeze(-1), reference_lse, atol=1e-3, rtol=1e-3)
    finally:
        torch.cuda.set_stream(previous_stream)
        if was_enabled:
            gc.enable()


@requires_pre_rubin_blackwell
@requires_dsl
@pytest.mark.parametrize("d", [128, 256])
@pytest.mark.parametrize("split", [1, 2])
def test_override_enabled_dense_plan_honors_bounded_kv(d, split):
    """Override admission preserves prepared plans, including split-KV and Stats."""
    b, h, hk, sq, sk = 1, 8, 2, 4, 2048
    graph, t = _dense_graph(b, h, hk, sq, sk, d, causal=False, split_kv=split, override_enabled=True)
    plan = _plan(graph)
    assert isinstance(plan._prepared, prep_mod.PreparedDenseLaunch)
    bufs = _dense_buffers(b, h, hk, sq, sk, d, seed=19)
    bufs["q"].zero_()
    bufs["k"].zero_()
    bufs["v"].fill_(9)
    bufs["v"][:, :, :1024].fill_(1)
    bufs["o"].fill_(float("nan"))
    bufs["lse"].fill_(float("nan"))
    ws = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device=DEV)
    with _Tripwire():
        graph.execute(
            _dense_pack(t, bufs),
            ws,
            override_uids=[t[n].get_uid() for n in ("k", "v")],
            override_shapes=[[b, hk, 1024, d]] * 2,
            override_strides=[bufs[n].stride() for n in ("k", "v")],
        )
    torch.cuda.synchronize()
    torch.testing.assert_close(bufs["o"], torch.ones_like(bufs["o"]), atol=0, rtol=0)
    torch.testing.assert_close(bufs["lse"], torch.full_like(bufs["lse"], math.log(1024)), atol=2e-6, rtol=0)


@requires_pre_rubin_blackwell
@requires_dsl
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("ordered", [False, True])
def test_native_thd_rebind_stream_capture_and_standalone(dtype, ordered, monkeypatch):
    """Both entry points use native admission with fresh buffers/workspace/stream.

    Replaying after input mutation checks that capture bound the current stream,
    and poisoned outputs catch incomplete writes after warm launches.
    """
    b, ql, kl, hq, hk, d = 2, 8, 64, 8, 2, 128
    fe_dtype = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g, t = _thd_graph(b, ql, kl, hq, hk, d, dtype=fe_dtype)
    plan = _plan(g)
    assert plan._prepared.spec.native is not None
    bufs = _buffers(b, ql, kl, hq, hk, d, dtype=dtype)
    ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    monkeypatch.setattr(prep_mod, "facts_of_roles", lambda *args: pytest.fail("native graph launch must not rebuild Python facts"))
    monkeypatch.setattr(prep_mod, "_bind_thd_python", lambda *args: pytest.fail("native f16 contract must not fall back to Python binding"))

    def execute(buffers, workspace):
        bindings = _pack(t, buffers)
        if ordered:
            items = list(reversed(list(bindings.items())))
            g.execute([buffer for _, buffer in items], workspace, tensor_uids=[tensor.get_uid() for tensor, _ in items])
        else:
            g.execute(bindings, workspace)

    def verify():
        o_ref, lse_ref = _reference(bufs, b, ql, kl, hq, hk, d)
        torch.testing.assert_close(bufs["o"].float(), o_ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(bufs["lse"], lse_ref, atol=1e-3, rtol=1e-3)

    with torch.cuda.stream(stream):
        execute(bufs, ws)
    stream.synchronize()
    verify()
    with torch.cuda.stream(stream):
        # Verify each route independently: otherwise a no-op standalone path
        # could leave the preceding graph's correct outputs untouched.
        bufs["o"].fill_(float("nan"))
        bufs["lse"].fill_(float("nan"))
        # The tensor/standalone route must share the same native evaluator.
        prepared = plan._prepared
        plan._prepared, plan.takes_variant_pack = None, False
        try:
            execute(bufs, ws)
        finally:
            plan._prepared, plan.takes_variant_pack = prepared, True
    stream.synchronize()
    verify()
    bufs = _buffers(b, ql, kl, hq, hk, d, seed=4, dtype=dtype)
    ws = torch.empty_like(ws)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        bufs["o"].fill_(float("nan"))
        bufs["lse"].fill_(float("nan"))
        execute(bufs, ws)
    stream.synchronize()
    verify()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        execute(bufs, ws)
    with torch.cuda.stream(stream):
        bufs["q"].mul_(0.5)
        bufs["o"].fill_(float("nan"))
        bufs["lse"].fill_(float("nan"))
        graph.replay()
    stream.synchronize()
    verify()
    # Replacement/replanning state must never patch an earlier captured frame.
    # The captured graph still refers to the first buffers and workspace.
    old_bufs, old_ws = bufs, ws
    replacement = _buffers(b, ql, kl, hq, hk, d, seed=9, dtype=dtype)
    new_ws = torch.empty_like(ws)
    execute(replacement, new_ws)
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        old_bufs["q"].mul_(0.5)
        graph.replay()
    stream.synchronize()
    assert old_ws is ws  # retain capture's metadata/workspace owners through replay
    verify()


@requires_pre_rubin_blackwell
@requires_dsl
def test_native_thd_concurrent_streams_use_independent_frames():
    from concurrent.futures import ThreadPoolExecutor

    b, ql, kl, hq, hk, d = 2, 8, 64, 8, 2, 128
    g, t = _thd_graph(b, ql, kl, hq, hk, d)
    assert _plan(g)._prepared.spec.native is not None
    buffers = [_buffers(b, ql, kl, hq, hk, d, seed=i + 1) for i in range(2)]
    workspaces = [torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8) for _ in range(2)]
    streams = [torch.cuda.Stream() for _ in range(2)]
    handles = [cudnn.create_handle() for _ in range(2)]
    for handle, stream in zip(handles, streams):
        cudnn.set_stream(handle, stream.cuda_stream)
    torch.cuda.synchronize()

    def run(i):
        for _ in range(3):
            g.execute(_pack(t, buffers[i]), workspaces[i], handle=handles[i])
        streams[i].synchronize()

    try:
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(run, range(2)))
        for bufs in buffers:
            o_ref, lse_ref = _reference(bufs, b, ql, kl, hq, hk, d)
            torch.testing.assert_close(bufs["o"].float(), o_ref, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(bufs["lse"], lse_ref, atol=1e-3, rtol=1e-3)
    finally:
        try:
            for stream in streams:
                stream.synchronize()
        finally:
            for handle in handles:
                cudnn.destroy_handle(handle)


@requires_pre_rubin_blackwell
@requires_dsl
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("d", [128, 256, 512])
def test_thd_output_row_stride_above_int32_reaches_device_descriptors(d):
    """B=2 must step the wide stride; a singleton ABI-only probe misses truncation in setup."""
    b, ql, kl, hq, hk = 2, 1, 64, 4, 2
    row_stride = 2**32 + hq * d
    if torch.cuda.mem_get_info()[0] < 2 * row_stride + 2**30:
        pytest.skip("wide physical row-stride regression needs 9 GiB free")
    g, t = _thd_graph(b, ql, kl, hq, hk, d, causal=False, override_enabled=True)
    assert _plan(g)._prepared.spec.native is not None
    bufs = _buffers(b, ql, kl, hq, hk, d)
    bufs["o"] = torch.empty_strided((b * ql, hq, d), (row_stride, d, 1), device=DEV, dtype=torch.bfloat16)
    ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    overrides = dict(override_uids=[t["o"].get_uid()], override_shapes=[[b, hq, ql, d]], override_strides=[[hq * d, d, row_stride, 1]])
    for replay in (False, True):
        if replay:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                g.execute(_pack(t, bufs), ws, **overrides)
            bufs["v"].mul_(0.5)
        bufs["o"].fill_(float("nan"))
        bufs["lse"].fill_(float("nan"))
        if replay:
            graph.replay()
        else:
            g.execute(_pack(t, bufs), ws, **overrides)
        o_ref, lse_ref = _reference(bufs, b, ql, kl, hq, hk, d, causal=False)
        torch.testing.assert_close(bufs["o"].float(), o_ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(bufs["lse"], lse_ref, atol=1e-3, rtol=1e-3)
