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


def _thd_graph(b, ql, kl, hq, hk, d, *, ragged_batch_stride=None, causal=True):
    """A THD bf16 graph the way FlashInfer declares it: BHSD dims with ragged offsets, cu_seq_len
    lengths, token-major Stats. ``ragged_batch_stride`` mimics FlashInfer's small declared batch
    stride (the declaration's span is then far below the buffer's)."""
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q_bs = ragged_batch_stride if ragged_batch_stride is not None else ql * hq * d
    kv_bs = ragged_batch_stride if ragged_batch_stride is not None else kl * hk * d
    tq = g.tensor(dim=[b, hq, ql, d], stride=[q_bs, d, hq * d, 1], data_type=cudnn.data_type.BFLOAT16, name="q")
    tk = g.tensor(dim=[b, hk, kl, d], stride=[kv_bs, d, hk * d, 1], data_type=cudnn.data_type.BFLOAT16, name="k")
    tv = g.tensor(dim=[b, hk, kl, d], stride=[kv_bs, d, hk * d, 1], data_type=cudnn.data_type.BFLOAT16, name="v")
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


def _buffers(b, ql, kl, hq, hk, d, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(b * ql, hq, d, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(b * kl, hk, d, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(b * kl, hk, d, device=DEV, dtype=torch.bfloat16)
    o = torch.empty(b * ql, hq, d, device=DEV, dtype=torch.bfloat16)
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

    def __call__(self, *frame):
        self.frames.append(dict(zip(self.spec.order, frame)))
        return self._fn(*frame)

    def restore(self):
        self.spec.fn = self._fn


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
        assert slots <= api_dsl.SdpaFwdDslSm100._DENSE_EXPLICIT_SLOTS, (mod.__name__, sorted(slots - api_dsl.SdpaFwdDslSm100._DENSE_EXPLICIT_SLOTS))
        assert slots <= prepared._FILLED_AT_BUILD | prepared._FILLED_PER_CALL, (
            mod.__name__,
            sorted(slots - prepared._FILLED_AT_BUILD - prepared._FILLED_PER_CALL),
        )
        if "gate_strides" in slots:  # a Tuple slot: the launch must pass a tuple even when the gate is absent
            assert "gate_strides" in prepared._FILLED_AT_BUILD


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
def test_zero_kv_clamp_and_runtime_strides():
    """All-zero KV lengths bind the V stub the spec allocated at build (no allocation, no first-use
    initialization during execute); K/V sliced from a fused slab bind their runtime token stride; a
    layout TMA cannot express is rejected before launch."""
    b, ql, kl, hq, hk, d = 4, 4, 64, 8, 2, 128
    g, t = _thd_graph(b, ql, kl, hq, hk, d)
    ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    bufs = _buffers(b, ql, kl, hq, hk, d)
    spec = _plan(g)._prepared.spec
    assert "v_stub" in spec._dummies and "sinks" in spec._dummies, "resources exist before the first execute"
    stub_before = spec.dummy("v_stub")
    rec = _Recorder(spec)
    try:
        zero_kv = dict(bufs, k=torch.empty(0, hk, d, device=DEV, dtype=torch.bfloat16), v=torch.empty(0, hk, d, device=DEV, dtype=torch.bfloat16))
        zero_kv["cu_kv"] = torch.zeros(b + 1, device=DEV, dtype=torch.int32)
        zero_kv["off_kv"] = torch.zeros(b + 1, device=DEV, dtype=torch.int32)
        g.execute(_pack(t, zero_kv), ws)
        torch.cuda.synchronize()
    finally:
        rec.restore()
    frame = rec.frames[-1]
    assert frame["problem_size"][4] == 1 and frame["v_ptr"] == stub_before and frame["k_ptr"] == frame["q_ptr"]
    assert spec.dummy("v_stub") == stub_before
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
    g, t = _thd_graph(b, ql, kl, hq, hk, d)
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
