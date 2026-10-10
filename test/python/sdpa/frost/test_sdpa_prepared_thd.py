# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""The prepared THD f16 launch (``cudnn.sdpa.fwd.prepared``) behind ``graph.execute()``.

One THD f16 plan, executed through the graph's normalized VariantPack: the plan is prepared,
capacities come from the caller's observed spans (not the graph's ragged declaration), every
call binds an independent frame, outputs are fully written, degenerate inputs are handled or
rejected before any launch, and execute allocates nothing and never synchronizes.
"""

from __future__ import annotations

import sdpa_binding_reference as binding_reference

import math
from itertools import accumulate

import pytest
import torch

import cudnn
from cudnn.sdpa.fwd import prepared as prep_mod
from cudnn.sdpa.fwd.engines import engine_name
from frost_test_utils import _dsl_installed, requires_blackwell, requires_blackwell_geforce, requires_dsl, requires_pre_rubin_blackwell

pytestmark = [pytest.mark.L0]

DEV = torch.device("cuda")


def _thd_graph(
    b,
    ql,
    kl,
    hq,
    hk,
    d,
    *,
    ragged_batch_stride=None,
    causal=True,
    override_enabled=False,
    dtype=cudnn.data_type.BFLOAT16,
    arch="sm100",
    stats_head_stride=0,
    has_sink=False,
):
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
    sink = g.tensor(dim=[1, hq, 1, 1], stride=[hq, 1, 1, 1], data_type=cudnn.data_type.FLOAT) if has_sink else None
    to, ts = g.sdpa(
        sink_token=sink,
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
    if stats_head_stride:
        ts.set_stride([hq * stats_head_stride, stats_head_stride, 1, 1])
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    want = engine_name(arch=arch)
    g.select_plan(next(i for i, n in enumerate(names) if n == want or n.startswith(want + "[")))
    g.check_support()
    g.build_plans()
    tensors = dict(q=tq, k=tk, v=tv, o=to, stats=ts, cu_q=t_cu_q, cu_kv=t_cu_kv, off_q=off_q, off_kv=off_kv, off_lse=off_lse)
    if has_sink:
        tensors["sink"] = sink
    return g, tensors


def _buffers(b, ql, kl, hq, hk, d, seed=0, dtype=torch.bfloat16, *, generator=None):
    if generator is None:
        torch.manual_seed(seed)
    q = torch.randn(b * ql, hq, d, device=DEV, dtype=dtype, generator=generator)
    k = torch.randn(b * kl, hk, d, device=DEV, dtype=dtype, generator=generator)
    v = torch.randn(b * kl, hk, d, device=DEV, dtype=dtype, generator=generator)
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
            spec.native = type(self._native)(spec)

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

    binding_reference._stats_layout_is_the_compiled_kind(spec, facts((B, H, S, 1), (S * H, 1, H, 1)))  # the declared view
    binding_reference._stats_layout_is_the_compiled_kind(spec, facts((B, S, H), (S * H, H, 1)))  # the storage itself
    binding_reference._stats_layout_is_the_compiled_kind(spec, facts((B * S * H,), (1,)))  # flat storage
    with pytest.raises(ValueError, match="declared"):
        binding_reference._stats_layout_is_the_compiled_kind(
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
    try:
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
    finally:
        graph.reset()


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
    dtype=cudnn.data_type.BFLOAT16,
    arch="sm100",
):
    """A dense half graph declared in BSHD storage (the zero-copy layout) or BHSD (which the tensor path
    repacks and the prepared launch therefore declines)."""
    d_v = d if d_v is None else d_v
    g = cudnn.pygraph(
        io_data_type=dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        is_override_shape_enabled=override_enabled,
    )

    def st(hh, s, dd):
        return [s * hh * dd, dd, hh * dd, 1] if bshd_storage else [hh * s * dd, s * dd, dd, 1]

    tq = g.tensor(dim=[b, h, s_q, d], stride=st(h, s_q, d), data_type=dtype, name="q")
    tk = g.tensor(dim=[b, hk, s_kv, d], stride=st(hk, s_kv, d), data_type=dtype, name="k")
    tv = g.tensor(dim=[b, hk, s_kv, d_v], stride=st(hk, s_kv, d_v), data_type=dtype, name="v")
    mask = dict(use_causal_mask_bottom_right=True) if (causal and bottom_right) else dict(use_causal_mask=causal)
    to, ts = g.sdpa(name="sdpa", q=tq, k=tk, v=tv, generate_stats=stats, stats_use_log2=stats_log2, attn_scale=1.0 / math.sqrt(d), **mask)
    to.set_output(True).set_dim([b, h, s_q, d_v]).set_stride(st(h, s_q, d_v) if o_stride is None else o_stride)
    if stats:
        ts.set_output(True).set_dim([b, h, s_q, 1]).set_stride([h * s_q, s_q, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    want = engine_name(arch=arch)
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
    try:
        with torch.cuda.graph(capture, stream=stream):
            g.execute(_dense_pack(t, bufs), ws, handle=handle)
        bufs["q"].mul_(0.75)
        bufs["o"].fill_(float("nan"))
        bufs["lse"].fill_(float("nan"))
        capture.replay()
        torch.cuda.synchronize()
        check(bufs)
    finally:
        capture.reset()


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
    assert binding_reference._bind_paged_kv(spec, frame, ix, facts, pool(False), pool(False), b) == max_pages * ps
    for k_hnd, v_hnd in ((True, False), (False, True)):
        with pytest.raises(ValueError, match="compiled for NHD"):
            binding_reference._bind_paged_kv(spec, list(frame), ix, facts, pool(k_hnd), pool(v_hnd), b)


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
    try:
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
    finally:
        graph.reset()


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
    monkeypatch.setattr(prep_mod, "_bind_thd_python", lambda *args: pytest.fail("native f16 contract must not fall back to Python binding"), raising=False)

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
    try:
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
    finally:
        graph.reset()


@requires_pre_rubin_blackwell
@requires_dsl
def test_native_thd_concurrent_streams_use_independent_frames():
    from concurrent.futures import ThreadPoolExecutor

    b, ql, kl, hq, hk, d = 2, 8, 64, 8, 2, 128
    g, t = _thd_graph(b, ql, kl, hq, hk, d)
    assert _plan(g)._prepared.spec.native is not None
    # This checks concurrent per-call frames after preparation. CuTe DSL
    # 4.7.0/4.7.1 can leak its process-global initialization lock when two
    # threads first enter the same cold module, hanging a later cold launch.
    # Load it serially, retaining these owners so the parallel calls below
    # must bind genuinely different storage (including fresh workspace).
    warm_buffers = _buffers(b, ql, kl, hq, hk, d, seed=0)
    warm_workspace = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    g.execute(_pack(t, warm_buffers), warm_workspace)
    buffers = [_buffers(b, ql, kl, hq, hk, d, seed=i + 1) for i in range(2)]
    workspaces = [torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8) for _ in range(2)]
    for bufs, workspace in zip(buffers, workspaces):
        bufs["o"].fill_(float("nan"))
        bufs["lse"].fill_(float("nan"))
        workspace.fill_(0xBD)
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
    graph = torch.cuda.CUDAGraph()
    try:
        for replay in (False, True):
            if replay:
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
    finally:
        graph.reset()


@requires_blackwell
@requires_dsl
@pytest.mark.parametrize("hnd", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "d,cga,causal,window,ql,splits,packed",
    [
        (128, None, False, None, 19, 1, False),
        (128, None, True, None, 19, 1, False),
        (256, None, False, None, 19, 1, False),
        (256, None, True, None, 19, 1, False),
        (128, 1, False, None, 19, 1, False),
        (128, 1, True, None, 19, 1, False),
        (128, 1, True, 15, 19, 1, False),
        pytest.param(128, 1, True, None, 1, 1, False, id="single-query-prefill"),
        (64, 1, False, None, 19, 3, False),
        (64, 1, True, None, 19, 3, False),
        (64, 1, True, 15, 19, 3, False),
        (64, 1, False, None, 19, 3, True),
        (64, 1, True, None, 19, 3, True),
        (64, 1, True, 15, 19, 3, True),
        (256, 2, False, None, 19, 3, False),
        (256, 2, True, None, 19, 3, False),
    ],
)
def test_native_paged_thd_capture_and_rebind(hnd, dtype, d, cga, causal, window, ql, splits, monkeypatch, packed):
    """Prepared and standalone launches bind fresh pools/tables without Python admission."""
    arch = "sm107" if torch.cuda.get_device_capability() == (10, 7) else "sm100"
    if cga == 1 and splits == 1 and arch != "sm107" and not (torch.cuda.get_device_capability() == (10, 0) and ql > 1):
        pytest.skip("The unsplit D128 cga1 paged prefill leg requires SM100 Q>1 or SM107")
    if d == 64 and arch == "sm107":
        pytest.skip("Native D64 paged split is qualified on SM100/SM103")
    if d == 256 and splits > 1 and arch != "sm107":
        pytest.skip("Paged D256 split and PackGQA are qualified on SM107")
    from test_sdpa_fwd_paged_sm100 import _pools

    b, h, hk, page, pages = 2, 8, 2, 16, 5
    dt = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    rng = torch.Generator(device=DEV).manual_seed(714)
    _, _, k, v, table = _pools(b, hk, d, page, pages, hnd, dtype, generator=rng)
    q = torch.randn(b * ql, h, d, device=DEV, dtype=dtype, generator=rng)
    bufs = dict(q=q, k=k, v=v, o=torch.empty_like(q), lse=torch.empty(b * ql, h, device=DEV))
    bufs.update(
        cu_q=torch.arange(b + 1, device=DEV, dtype=torch.int32) * ql,
        seq_kv=torch.tensor([pages * page - 3, pages * page - 11], device=DEV, dtype=torch.int32),
        k_table=table,
        v_table=table.flip(2),
    )
    bufs["off_q"], bufs["off_lse"] = bufs["cu_q"] * h * d, bufs["cu_q"] * h
    g = cudnn.pygraph(io_data_type=dt, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, is_override_shape_enabled=True)
    t = {n: g.tensor_like(x) for n, x in bufs.items() if n not in ("q", "o", "lse")}
    t["q"] = g.tensor(dim=[b, h, ql, d], stride=[ql * h * d, d, h * d, 1], data_type=dt).set_ragged_offset(t["off_q"])
    t["o"], t["lse"] = g.sdpa(
        q=t["q"],
        k=t["k"],
        v=t["v"],
        generate_stats=True,
        attn_scale=1 / math.sqrt(d),
        use_padding_mask=True,
        use_causal_mask_bottom_right=causal,
        diagonal_band_left_bound=window + 1 if window is not None else None,
        cu_seq_len_q=t["cu_q"],
        seq_len_kv=t["seq_kv"],
        max_total_seq_len_q=b * ql,
        paged_attention_k_table=t["k_table"],
        paged_attention_v_table=t["v_table"],
        paged_attention_max_seq_len_kv=page * pages,
    )
    t["o"].set_output(True).set_dim([b, h, ql, d]).set_stride([ql * h * d, d, h * d, 1]).set_ragged_offset(t["off_q"])
    t["lse"].set_output(True).set_dim([b, h, ql, 1]).set_stride([ql * h, 1, h, 1]).set_data_type(cudnn.data_type.FLOAT).set_ragged_offset(t["off_lse"])
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    index = next(i for i, name in enumerate(names) if name == engine_name(arch=arch) or name.startswith(engine_name(arch=arch) + "["))
    g.select_plan(index)
    if cga is not None:
        engine, knobs = g.get_engine_and_knobs_at_index(index)
        g.create_execution_plan(engine, {**knobs, cudnn.knob_type.TILE_CGA_M: cga, cudnn.knob_type.SPLIT_KV: splits, cudnn.knob_type.PACK_GQA: packed})
        g.select_plan(g.get_execution_plan_count() - 1)
        _, selected = g.get_engine_and_knobs_at_index(g._plan_index)
        assert selected[cudnn.knob_type.SPLIT_KV] == splits and selected[cudnn.knob_type.TILE_CGA_M] == cga
    g.check_support()
    g.build_plans()
    plan = _plan(g)
    assert plan._prepared.spec.native is not None and plan._prepared.spec.paged
    ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    monkeypatch.setattr(prep_mod, "_bind_thd_python", lambda *a: pytest.fail("paged native binder fell back"), raising=False)
    monkeypatch.setattr(prep_mod, "facts_of_roles", lambda *a: pytest.fail("paged graph rebuilt Python facts"))

    def execute(current, workspace, *, override=True):
        g.execute(
            {t[n]: x for n, x in current.items()},
            workspace,
            override_uids=[t[n].get_uid() for n in ("k", "v", "k_table", "v_table")] if override else None,
            override_shapes=[list(current[n].shape) for n in ("k", "v", "k_table", "v_table")] if override else None,
            override_strides=[list(current[n].stride()) for n in ("k", "v", "k_table", "v_table")] if override else None,
        )

    def check(current):
        for i, length in enumerate(current["seq_kv"].tolist()):
            dense = {}
            for n in ("k", "v"):
                ids = current[n + "_table"][i, 0, :, 0].long()
                dense[n] = current[n][ids].transpose(1, 2).reshape(-1, hk, d)[:length].repeat_interleave(h // hk, 1).double()
            qi = current["q"][i * ql : (i + 1) * ql].double()
            scores = torch.einsum("qhd,khd->hqk", qi, dense["k"]) / math.sqrt(d)
            if causal:
                scores.masked_fill_(torch.arange(length, device=DEV)[None, :] > torch.arange(ql, device=DEV)[:, None] + length - ql, -float("inf"))
            if window is not None:
                scores.masked_fill_(torch.arange(length, device=DEV)[None, :] < torch.arange(ql, device=DEV)[:, None] + length - ql - window, -float("inf"))
            ref = torch.einsum("hqk,khd->qhd", scores.softmax(-1), dense["v"])
            torch.testing.assert_close(current["o"][i * ql : (i + 1) * ql].float(), ref.float(), atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(current["lse"][i * ql : (i + 1) * ql], scores.logsumexp(-1).T.float(), atol=1e-3, rtol=1e-3)

    execute(bufs, ws)
    check(bufs)
    # Use fresh addresses and nonunit table columns in the same compiled graph.
    bufs = {n: x.clone() for n, x in bufs.items()}
    for n in ("k_table", "v_table"):
        old = bufs[n]
        raw = torch.zeros((b, 1, pages * 3, 1), device=DEV, dtype=torch.int32)
        bufs[n] = raw[:, :, ::3]
        bufs[n].copy_(old)
    ws = torch.empty_like(ws)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            execute(bufs, ws)
            with torch.cuda.graph(graph, stream=stream):
                torch.cuda.set_sync_debug_mode("error")
                try:
                    execute(bufs, ws)
                finally:
                    torch.cuda.set_sync_debug_mode("default")
        stream.synchronize()
        replacement = {n: x.clone() for n, x in bufs.items()}
        replacement["q"].mul_(0.75)
        replacement["o"].fill_(float("nan"))
        replacement["lse"].fill_(float("nan"))
        execute(replacement, torch.empty_like(ws))
        check(replacement)
        with torch.cuda.stream(stream):
            bufs["q"].mul_(0.5)
            bufs["o"].fill_(float("nan"))
            bufs["lse"].fill_(float("nan"))
            graph.replay()
        stream.synchronize()
        check(bufs)
    finally:
        graph.reset()
    # The tensor executor uses graph-declared strides and does not accept
    # overrides. Restore those strides before exercising its native admission.
    for n in ("k_table", "v_table"):
        bufs[n] = bufs[n].contiguous()
    prepared = plan._prepared
    plan._prepared, plan.takes_variant_pack = None, False
    try:
        bufs["o"].fill_(float("nan"))
        bufs["lse"].fill_(float("nan"))
        execute(bufs, ws, override=False)
        check(bufs)
    finally:
        plan._prepared, plan.takes_variant_pack = prepared, True


@requires_blackwell
@requires_dsl
@pytest.mark.parametrize("d,b", [(64, 3), (96, 3), (128, 3), (200, 3), (256, 3), (96, 1), (128, 1)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_thd_scheduler_policies_replay_changed_ragged_metadata(d, b, dtype):
    """Every public policy covers the same live rows, including empty sequences/KV."""
    rubin = torch.cuda.get_device_capability() == (10, 7)
    if rubin and d not in (128, 256):
        pytest.skip("Only D128/D256 half flavors admit LPT on SM107")
    hq, hk, qcap, kcap = 16, 2, 1025, 2305
    io_type = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g, t = _thd_graph(b, qcap, kcap, hq, hk, d, dtype=io_type, arch="sm107" if rubin else "sm100")
    engine, knobs = g.get_engine_and_knobs_at_index(g._plan_index)
    plans = []
    for policy in ((0, 1) if rubin else (0, 1, 2)):
        g.create_execution_plan(engine, {**knobs, cudnn.knob_type.SCHED_POLICY: policy})
        index = g.get_execution_plan_count() - 1
        g.build_plan_at_index(index)
        plans.append(index)
    bufs = _buffers(b, qcap, kcap, hq, hk, d, dtype=dtype, generator=torch.Generator(device=DEV).manual_seed(91))
    workspaces, captures = [], []
    for index in plans:
        g.build_plan_at_index(index)
        ws = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device=DEV)
        g.execute(_pack(t, bufs), ws)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            g.execute(_pack(t, bufs), ws)
        workspaces.append(ws)
        captures.append(graph)
    try:
        lengths = (
            (([513, 0, 1025], [769, 0, 2049]), ([0, 513, 1025], [0, 0, 1793]))
            if b > 1
            else (([qcap], [kcap]), ([257], [769]), ([1], [1]), ([128], [0]), ([0], [kcap]), ([qcap], [kcap]))
        )
        for ql, kl in lengths:
            cq, ck = [0, *accumulate(ql)], [0, *accumulate(kl)]
            for name, values in (
                ("cu_q", cq),
                ("cu_kv", ck),
                ("off_q", [x * hq * d for x in cq]),
                ("off_kv", [x * hk * d for x in ck]),
                ("off_lse", [x * hq for x in cq]),
            ):
                bufs[name].copy_(torch.tensor(values, dtype=torch.int32, device=DEV))
            bufs["q"].mul_(-0.5)
            ref_o = torch.zeros(cq[-1], hq, d, dtype=torch.float32, device=DEV)
            ref_s = torch.full((cq[-1], hq), -float("inf"), dtype=torch.float32, device=DEV)
            for batch, (nq, nk) in enumerate(zip(ql, kl)):
                if nq == 0 or nk == 0:
                    continue
                q = bufs["q"][cq[batch] : cq[batch + 1]].float().transpose(0, 1)
                k = bufs["k"][ck[batch] : ck[batch + 1]].float().transpose(0, 1).repeat_interleave(hq // hk, 0)
                v = bufs["v"][ck[batch] : ck[batch + 1]].float().transpose(0, 1).repeat_interleave(hq // hk, 0)
                score = q @ k.transpose(1, 2) / math.sqrt(d)
                score.masked_fill_(torch.arange(nk, device=DEV)[None, :] > torch.arange(nq, device=DEV)[:, None], -float("inf"))
                ref_o[cq[batch] : cq[batch + 1]] = (score.softmax(-1) @ v).transpose(0, 1)
                ref_s[cq[batch] : cq[batch + 1]] = score.logsumexp(-1).transpose(0, 1)
            natural = None
            for policy, graph in enumerate(captures):
                bufs["o"].fill_(float("nan"))
                bufs["lse"].fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                got_o, got_s = bufs["o"][: cq[-1]], bufs["lse"][: cq[-1]]
                torch.testing.assert_close(got_o.float(), ref_o, atol=2e-2, rtol=2e-2)
                torch.testing.assert_close(got_s, ref_s, atol=1e-3, rtol=1e-3)
                assert torch.isnan(bufs["o"][cq[-1] :]).all(), "policy must not write past live packed Q"
                assert torch.isnan(bufs["lse"][cq[-1] :]).all(), "policy must not write Stats past live packed Q"
                if policy == 0:
                    natural = got_o.clone(), got_s.clone()
                else:
                    torch.testing.assert_close(got_o, natural[0], atol=0, rtol=0)
                    torch.testing.assert_close(got_s, natural[1], atol=0, rtol=0)
    finally:
        for graph in captures:
            graph.reset()


@requires_pre_rubin_blackwell
@requires_dsl
@pytest.mark.parametrize("hnd", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("page", [8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("geometry", ["d128_packed", "d128_short", "d128_long"])
def test_packed_thd_grid_capture_changes_lengths(hnd, dtype, page, geometry):
    """Live scheduling and the packed grid preserve O/Stats under old captures."""
    from test_sdpa_fwd_paged_sm100 import _pools

    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Live-length scheduler is initially admitted only on SM100")
    b, h, hk, d, qcap, kcap = (1, 32, 8, 128, 2049, 2560)
    if geometry == "d128_short":
        qcap = 1025
    elif geometry == "d128_long":
        qcap = 2305
    dt = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    torch.manual_seed(191)
    _, _, k, v, table = _pools(b, hk, d, page, kcap // page, hnd, dtype)
    q = torch.randn(b * qcap, h, d, device=DEV, dtype=dtype)
    bufs = dict(q=q, k=k, v=v, o=torch.empty_like(q), lse=torch.empty(b * qcap, h, device=DEV))
    lse_tokens = bufs["lse"]
    bufs.update(
        cu_q=torch.arange(b + 1, device=DEV, dtype=torch.int32) * qcap,
        seq_kv=torch.full((b,), qcap, device=DEV, dtype=torch.int32),
        k_table=table,
        v_table=table.flip(2),
    )
    bufs["off_q"], bufs["off_lse"] = bufs["cu_q"] * h * d, bufs["cu_q"] * lse_tokens.stride(0)
    g = cudnn.pygraph(
        io_data_type=dt,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    t = {n: g.tensor_like(x) for n, x in bufs.items() if n not in ("q", "o", "lse")}
    t["q"] = g.tensor(dim=[b, h, qcap, d], stride=[qcap * h * d, d, h * d, 1], data_type=dt).set_ragged_offset(t["off_q"])
    t["o"], t["lse"] = g.sdpa(
        q=t["q"],
        k=t["k"],
        v=t["v"],
        generate_stats=True,
        attn_scale=1 / math.sqrt(d),
        use_causal_mask_bottom_right=True,
        use_padding_mask=True,
        cu_seq_len_q=t["cu_q"],
        seq_len_kv=t["seq_kv"],
        max_total_seq_len_q=b * qcap,
        paged_attention_k_table=t["k_table"],
        paged_attention_v_table=t["v_table"],
        paged_attention_max_seq_len_kv=kcap,
    )
    t["o"].set_output(True).set_dim([b, h, qcap, d]).set_stride([qcap * h * d, d, h * d, 1]).set_ragged_offset(t["off_q"])
    t["lse"].set_output(True).set_dim([b, h, qcap, 1]).set_stride([qcap * h, 1, h, 1]).set_data_type(cudnn.data_type.FLOAT).set_ragged_offset(t["off_lse"])
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    index = next(i for i, name in enumerate(names) if name == engine_name() or name.startswith(engine_name() + "["))
    engine, knobs = g.get_engine_and_knobs_at_index(index)
    captures, workspaces = [], []
    for policy in (0, 1):
        chosen = {**knobs, cudnn.knob_type.SCHED_POLICY: policy}
        chosen.update({cudnn.knob_type.PACK_GQA: 1, cudnn.knob_type.TILE_CGA_M: 2, cudnn.knob_type.SPLIT_KV: 1})
        g.create_execution_plan(engine, chosen)
        g.build_plan_at_index(g.get_execution_plan_count() - 1)
        spec = _plan(g)._prepared.spec
        resident = torch.cuda.get_device_properties(0).multi_processor_count // 2
        # Independent worklist oracle: every 128-token tile has eight packed heads.
        work = len({(row // 128, head // 4) for row in range(qcap) for head in range(h)})
        expanded = dtype == torch.bfloat16 and policy == 1 and resident < work <= 2 * resident
        assert spec.template[spec.index["n_thd_units"]] == (work if expanded else min(resident, ((qcap + 511) // 512) * h))
        ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
        pack = {t[n]: x for n, x in bufs.items()}
        g.execute(pack, ws)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            g.execute(pack, ws)
        captures.append(graph)
        workspaces.append(ws)
    try:
        lengths = (([qcap], [qcap]), ([257], [769]), ([1], [1]), ([128], [0]), ([0], [kcap]))
        for ql, kl in lengths:
            cq = [0, *accumulate(ql)]
            for name, values in (("cu_q", cq), ("off_q", [x * h * d for x in cq]), ("off_lse", [x * lse_tokens.stride(0) for x in cq]), ("seq_kv", kl)):
                bufs[name].copy_(torch.tensor(values, device=DEV, dtype=torch.int32))
            bufs["q"].mul_(-0.5)
            ref_o = torch.zeros(cq[-1], h, d, device=DEV, dtype=torch.float64)
            ref_s = torch.full((cq[-1], h), -float("inf"), device=DEV, dtype=torch.float64)
            for i, (nq, nk) in enumerate(zip(ql, kl)):
                if not nq or not nk:
                    continue
                dense = {}
                for n in ("k", "v"):
                    ids = bufs[n + "_table"][i, 0, :, 0].long()
                    dense[n] = bufs[n][ids].transpose(1, 2).reshape(-1, hk, d)[:nk].repeat_interleave(h // hk, 1).double()
                scores = torch.einsum("qhd,khd->hqk", bufs["q"][cq[i] : cq[i + 1]].double(), dense["k"]) / math.sqrt(d)
                scores.masked_fill_(torch.arange(nk, device=DEV)[None, :] > torch.arange(nq, device=DEV)[:, None] + nk - nq, -float("inf"))
                ref_o[cq[i] : cq[i + 1]] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), dense["v"])
                ref_s[cq[i] : cq[i + 1]] = scores.logsumexp(-1).T
            natural = None
            for graph in captures:
                bufs["o"].fill_(float("nan"))
                bufs["lse"].fill_(float("nan"))
                for _ in range(256 if cq[-1] <= 1 else 1):
                    graph.replay()
                torch.cuda.synchronize()
                got_o, got_s = bufs["o"][: cq[-1]], lse_tokens[: cq[-1]]
                torch.testing.assert_close(got_o.float(), ref_o.float(), atol=1.2e-2, rtol=1.2e-2)
                torch.testing.assert_close(got_s, ref_s.float(), atol=1e-3, rtol=1e-3)
                assert torch.isnan(bufs["o"][cq[-1] :]).all()
                assert torch.isnan(lse_tokens[cq[-1] :]).all()
                if natural is None:
                    natural = got_o.clone(), got_s.clone()
                else:
                    torch.testing.assert_close(got_o, natural[0], atol=0, rtol=0)
                    torch.testing.assert_close(got_s, natural[1], atol=0, rtol=0)
    finally:
        for graph in captures:
            graph.reset()


@requires_dsl
@pytest.mark.parametrize("arch", [pytest.param("sm100", marks=requires_blackwell), pytest.param("sm120", marks=requires_blackwell_geforce)])
@pytest.mark.parametrize("d", [128, 256, 512])
@pytest.mark.parametrize("backend_lowering", [True, False])
def test_thd_cache_shape_grid_tracks_runtime_capacity(d, arch, backend_lowering, monkeypatch):
    """One large cache-shape artifact, small changing batches and captured device lengths."""
    if torch.cuda.get_device_capability() == (10, 7):
        pytest.skip("SM107 overlaunch admission is qualified separately")
    if not backend_lowering:

        def decline(self):
            raise cudnn.cudnnGraphNotSupportedError("backend lowering unavailable for this test")

        monkeypatch.setattr(cudnn.pygraph, "_backend_lowerable", lambda self: False)
        monkeypatch.setattr(cudnn.pygraph, "_lower_backend_graph", decline)
    hq, hk = 4, 2
    g, t = _thd_graph(4096, 65536, 65536, hq, hk, d, override_enabled=True, arch=arch)
    if not backend_lowering:
        assert g._lowered_graph is None
    plan = _plan(g)
    spec = plan._prepared.spec
    template = list(spec.template)
    rec = _Recorder(spec)
    ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    torch.manual_seed(5350)

    def cumulative(lengths):
        return [0] + list(accumulate(lengths))

    def reference(bufs, qlens, klens):
        oq, ok = cumulative(qlens), cumulative(klens)
        expected_o = torch.empty_like(bufs["o"], dtype=torch.float32)
        expected_lse = torch.empty_like(bufs["lse"])
        for i, (nq, nk) in enumerate(zip(qlens, klens)):
            if nq == 0:
                continue
            q = bufs["q"][oq[i] : oq[i + 1]].float().transpose(0, 1)
            k = bufs["k"][ok[i] : ok[i + 1]].float().transpose(0, 1).repeat_interleave(hq // hk, 0)
            v = bufs["v"][ok[i] : ok[i + 1]].float().transpose(0, 1).repeat_interleave(hq // hk, 0)
            scores = q @ k.transpose(1, 2) / math.sqrt(d)
            mask = torch.arange(nk, device=DEV)[None, :] > torch.arange(nq, device=DEV)[:, None]
            scores.masked_fill_(mask, -float("inf"))
            expected_o[oq[i] : oq[i + 1]] = (scores.softmax(-1) @ v).transpose(0, 1)
            expected_lse[oq[i] : oq[i + 1]] = scores.logsumexp(-1).transpose(0, 1)
        return expected_o, expected_lse

    captured = None
    try:
        for qlens in ([517], [1, 3, 0, 513], [517]):
            b, total = len(qlens), sum(qlens)
            klens = [n + 17 for n in qlens]
            bufs = {
                name: torch.randn(n, heads, d, device=DEV, dtype=torch.bfloat16)
                for name, n, heads in (("q", total, hq), ("k", sum(klens), hk), ("v", sum(klens), hk))
            }
            bufs.update(o=torch.empty_like(bufs["q"]), lse=torch.empty(total, hq, device=DEV))
            for name, values, width in (("cu_q", qlens, 1), ("cu_kv", klens, 1), ("off_q", qlens, hq * d), ("off_kv", klens, hk * d), ("off_lse", qlens, hq)):
                bufs[name] = torch.tensor(cumulative(values), device=DEV, dtype=torch.int32) * width
            sq, sk = max(qlens), max(klens)
            uids = [t[n].get_uid() for n in ("q", "k", "v", "o", "stats", "cu_q", "cu_kv", "off_q", "off_kv", "off_lse")]
            shapes = [[b, hq, sq, d], [b, hk, sk, d], [b, hk, sk, d], [b, hq, sq, d], [b, hq, sq, 1]] + [[b + 1, 1, 1, 1]] * 5
            strides = [
                [sq * hq * d, d, hq * d, 1],
                [sk * hk * d, d, hk * d, 1],
                [sk * hk * d, d, hk * d, 1],
                [sq * hq * d, d, hq * d, 1],
                [sq * hq, 1, hq, 1],
            ] + [[1, 1, 1, 1]] * 5
            # Ragged offsets are caller slots in either operand layout: engines
            # that address padded THD read them on device (the SM80 backward),
            # so the Python layout keeps them when the backend cannot lower.
            operands = set(g._variant_pack_uids())
            assert all(uid in operands for uid in uids)
            overrides = [(uid, shape, stride) for uid, shape, stride in zip(uids, shapes, strides) if uid in operands]
            uids, shapes, strides = map(list, zip(*overrides))
            pack = _pack(t, bufs)

            def run():
                g.execute(pack, ws, override_uids=uids, override_shapes=shapes, override_strides=strides)

            with _Tripwire():
                previous = torch.cuda.get_sync_debug_mode()
                torch.cuda.set_sync_debug_mode("error")
                try:
                    run()
                finally:
                    torch.cuda.set_sync_debug_mode(previous)
                captured = torch.cuda.CUDAGraph()
                with torch.cuda.graph(captured):
                    run()
            bound = ((total - 1) // spec.cga_tile_m + b) * hq
            assert rec.frames[-1]["n_thd_units"] <= bound
            assert spec.template == template
            for lengths in (qlens, ([129, 129, 129, 130] if b == 4 else qlens)):
                kv_lengths = [n + 17 for n in lengths]
                for name, values, width in (
                    ("cu_q", lengths, 1),
                    ("cu_kv", kv_lengths, 1),
                    ("off_q", lengths, hq * d),
                    ("off_kv", kv_lengths, hk * d),
                    ("off_lse", lengths, hq),
                ):
                    bufs[name].copy_(torch.tensor(cumulative(values), device=DEV, dtype=torch.int32) * width)
                bufs["q"].mul_(0.75)
                bufs["o"].fill_(float("nan"))
                bufs["lse"].fill_(float("nan"))
                captured.replay()
                expected_o, expected_lse = reference(bufs, lengths, kv_lengths)
                torch.testing.assert_close(bufs["o"].float(), expected_o, atol=2e-2, rtol=2e-2)
                torch.testing.assert_close(bufs["lse"], expected_lse, atol=1e-3, rtol=1e-3)
            assert _plan(g) is plan
            captured.reset()
            captured = None
    finally:
        if captured is not None:
            captured.reset()
        rec.restore()


# Define DSL probes at module scope: the tracer resolves names from the
# defining module, not from a test function's local imports.
if _dsl_installed():
    import cuda.bindings.driver as _prefix_cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cudnn.frost.tile_dsl.thd import write_thd_batch_remap, write_thd_live_and_ctr, write_thd_prefix_warp

    @cute.kernel
    def _prefix_probe(meta_t, q, k, b: cutlass.Int32, flags: cutlass.Int32):
        tid, _, _ = cute.arch.thread_idx()
        warp, lane = cutlass.Int32(tid) // 32, cutlass.Int32(tid) % 32
        meta = cutlass.make_array_view(meta_t)
        if warp == 0:
            write_thd_prefix_warp(meta, cutlass.make_array_view(q), b, b, (flags & 1) != 0, lane, store_lengths=False)
        if warp == 1:
            write_thd_prefix_warp(meta, cutlass.make_array_view(k), b, 2 * b + 1, (flags & 2) != 0, lane, store_lengths=True)

        cute.arch.barrier()
        write_thd_batch_remap(meta, b, cutlass.Int32(tid), cutlass.Int32(64))
        cute.arch.barrier()
        write_thd_live_and_ctr(meta, b, cutlass.Int32(3), cutlass.Int32(128), cutlass.Int32(7), cutlass.Int32(tid))

    _prefix_probe.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    @cute.jit
    def _prefix_host(m, q, k, b: cutlass.Int32, flags: cutlass.Int32, stream):
        _prefix_probe(m, q, k, b, flags).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


@pytest.fixture(scope="module")
def _prefix_runner():
    compiled = None

    def run(meta, q, k, b, flags):
        nonlocal compiled
        tensor = lambda x: from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=0)
        args = (tensor(meta), tensor(q), tensor(k), cutlass.Int32(b), cutlass.Int32(flags), _prefix_cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        if compiled is None:
            compiled = cute.compile(_prefix_host, *args)
        compiled(*args)

    return run


@requires_blackwell
@requires_dsl
@pytest.mark.parametrize("b", [0, 1, 2, 4, 7, 8, 9, 31, 32, 33, 63, 64, 65, 255, 256, 257, 1024])
@pytest.mark.parametrize("flags", range(4))
def test_parallel_thd_metadata_matches_lengths_and_normalized_cu(b, flags, _prefix_runner):
    """Exact metadata across warp boundaries, zero lengths, and sliced prefixes."""
    q = (torch.arange(b, dtype=torch.int32) * 17) % 257
    k = (torch.arange(b, dtype=torch.int32) * 31) % 1025
    cq = torch.cat((torch.zeros(1, dtype=torch.int32), q.cumsum(0, dtype=torch.int32)))
    ck = torch.cat((torch.zeros(1, dtype=torch.int32), k.cumsum(0, dtype=torch.int32)))
    q_in = (cq + 17 if flags & 1 else q).to(DEV)
    k_in = (ck + 31 if flags & 2 else k).to(DEV)
    meta = torch.full((4 * b + 4,), -12345, dtype=torch.int32, device=DEV)
    _prefix_runner(meta, q_in, k_in, b, flags)
    got = meta.cpu()
    torch.testing.assert_close(got[: 3 * b + 2], torch.cat((k, cq, ck)), rtol=0, atol=0)
    remap = torch.tensor(sorted(range(b), key=lambda i: (-int(q[i]), i)), dtype=torch.int32)
    live = int(((q + 127) // 128).sum()) * 3
    torch.testing.assert_close(got[3 * b + 2 :], torch.cat((remap, torch.tensor([live, 7], dtype=torch.int32))), rtol=0, atol=0)


@requires_blackwell
@requires_dsl
@pytest.mark.parametrize("hnd", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("page", [16, 128])
@pytest.mark.parametrize("d,cga", [(64, 2), (128, 2), (256, 2), (128, 1)])
def test_thd_lpt_paged_capture_changes_full_and_prefix_lengths(hnd, dtype, page, d, cga):
    """All policies preserve live full/prefix, mixed, and empty requests under capture."""
    from test_sdpa_fwd_paged_sm100 import _pools

    cc = torch.cuda.get_device_capability()
    rubin = cc == (10, 7)
    if cc not in ((10, 0), (10, 3), (10, 7)) or (rubin and d == 64):
        pytest.skip("Paged LPT requires a qualified SM100/SM103/SM107 flavor")
    if cga == 1 and cc not in ((10, 0), (10, 7)):
        pytest.skip("The unsplit D128 cga1 paged prefill leg requires SM100 Q>1 or SM107")
    arch = "sm107" if rubin else "sm100"
    b, h, hk, qcap, kcap = 3, 8, 1, 1025, 2304
    dt = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    rng = torch.Generator(device=DEV).manual_seed(191)
    _, _, k, v, table = _pools(b, hk, d, page, kcap // page, hnd, dtype, generator=rng)
    q = torch.randn(b * qcap, h, d, device=DEV, dtype=dtype, generator=rng)
    bufs = dict(q=q, k=k, v=v, o=torch.empty_like(q), lse=torch.empty(b * qcap, h, device=DEV))
    bufs.update(
        cu_q=torch.arange(b + 1, device=DEV, dtype=torch.int32) * qcap,
        seq_kv=torch.full((b,), qcap, device=DEV, dtype=torch.int32),
        k_table=table,
        v_table=table.flip(2),
    )
    bufs["off_q"], bufs["off_lse"] = bufs["cu_q"] * h * d, bufs["cu_q"] * h
    g = cudnn.pygraph(io_data_type=dt, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    t = {n: g.tensor_like(x) for n, x in bufs.items() if n not in ("q", "o", "lse")}
    t["q"] = g.tensor(dim=[b, h, qcap, d], stride=[qcap * h * d, d, h * d, 1], data_type=dt).set_ragged_offset(t["off_q"])
    t["o"], t["lse"] = g.sdpa(
        q=t["q"],
        k=t["k"],
        v=t["v"],
        generate_stats=True,
        attn_scale=1 / math.sqrt(d),
        use_causal_mask_bottom_right=True,
        use_padding_mask=True,
        cu_seq_len_q=t["cu_q"],
        seq_len_kv=t["seq_kv"],
        max_total_seq_len_q=b * qcap,
        paged_attention_k_table=t["k_table"],
        paged_attention_v_table=t["v_table"],
        paged_attention_max_seq_len_kv=kcap,
    )
    t["o"].set_output(True).set_dim([b, h, qcap, d]).set_stride([qcap * h * d, d, h * d, 1]).set_ragged_offset(t["off_q"])
    t["lse"].set_output(True).set_dim([b, h, qcap, 1]).set_stride([qcap * h, 1, h, 1]).set_data_type(cudnn.data_type.FLOAT).set_ragged_offset(t["off_lse"])
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    index = next(i for i, name in enumerate(names) if name == engine_name(arch=arch) or name.startswith(engine_name(arch=arch) + "["))
    engine, knobs = g.get_engine_and_knobs_at_index(index)
    if d == 128:
        knobs = {**knobs, cudnn.knob_type.PACK_GQA: True, cudnn.knob_type.TILE_CGA_M: cga, cudnn.knob_type.SPLIT_KV: 1}
    captures, workspaces = [], []
    for policy in ((0, 1) if rubin else (0, 1, 2)):
        g.create_execution_plan(engine, {**knobs, cudnn.knob_type.SCHED_POLICY: policy})
        g.build_plan_at_index(g.get_execution_plan_count() - 1)
        ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
        pack = {t[n]: x for n, x in bufs.items()}
        g.execute(pack, ws)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            g.execute(pack, ws)
        captures.append(graph)
        workspaces.append(ws)
    try:
        for ql, kl in (([1025, 513, 0], [1025, 2049, 0]), ([0, 1025, 513], [0, 1025, 0]), ([257, 0, 1025], [769, 0, 1025])):
            cq = [0, *accumulate(ql)]
            for name, values in (("cu_q", cq), ("off_q", [x * h * d for x in cq]), ("off_lse", [x * h for x in cq]), ("seq_kv", kl)):
                bufs[name].copy_(torch.tensor(values, device=DEV, dtype=torch.int32))
            bufs["q"].mul_(-0.5)
            ref_o = torch.zeros(cq[-1], h, d, device=DEV, dtype=torch.float64)
            ref_s = torch.full((cq[-1], h), -float("inf"), device=DEV, dtype=torch.float64)
            for i, (nq, nk) in enumerate(zip(ql, kl)):
                if not nq or not nk:
                    continue
                dense = {}
                for n in ("k", "v"):
                    ids = bufs[n + "_table"][i, 0, :, 0].long()
                    dense[n] = bufs[n][ids].transpose(1, 2).reshape(-1, hk, d)[:nk].repeat_interleave(h // hk, 1).double()
                scores = torch.einsum("qhd,khd->hqk", bufs["q"][cq[i] : cq[i + 1]].double(), dense["k"]) / math.sqrt(d)
                scores.masked_fill_(torch.arange(nk, device=DEV)[None, :] > torch.arange(nq, device=DEV)[:, None] + nk - nq, -float("inf"))
                ref_o[cq[i] : cq[i + 1]] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), dense["v"])
                ref_s[cq[i] : cq[i + 1]] = scores.logsumexp(-1).T
            natural = None
            for graph in captures:
                bufs["o"].fill_(float("nan"))
                bufs["lse"].fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                got_o, got_s = bufs["o"][: cq[-1]], bufs["lse"][: cq[-1]]
                torch.testing.assert_close(got_o.float(), ref_o.float(), atol=1.2e-2, rtol=1.2e-2)
                torch.testing.assert_close(got_s, ref_s.float(), atol=1e-3, rtol=1e-3)
                assert torch.isnan(bufs["o"][cq[-1] :]).all()
                assert torch.isnan(bufs["lse"][cq[-1] :]).all()
                if natural is None:
                    natural = got_o.clone(), got_s.clone()
                else:
                    torch.testing.assert_close(got_o, natural[0], atol=0, rtol=0)
                    torch.testing.assert_close(got_s, natural[1], atol=0, rtol=0)
    finally:
        for graph in captures:
            graph.reset()


@requires_blackwell
@requires_dsl
@pytest.mark.parametrize("hnd", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("page", [16, 128])
@pytest.mark.parametrize(
    "geometry,stats_layout,stats_log2,splits",
    [
        ("sink_d128_split", "NH", False, 4),
        ("sink_d128_split_b1_gqa", "HN", True, 8),
        ("sink_d128_gqa16_split_gqa", "NH", False, 3),
        ("sink_d128_mha_split", "NH", True, 3),
        ("d128_split", "NH", False, 4),
        ("d128_split", "HN", False, 4),
        ("d128_split", "NH", True, 4),
        ("d128_split", "HN", True, 4),
        ("d128_split", "NH", False, 3),
        ("d128_split", "HN", False, 3),
        ("d128_split", "NH", True, 3),
        ("d128_split", "HN", True, 3),
        ("d128_split_gqa", "NH", False, 2),
        ("d128_split_gqa", "HN", True, 3),
        ("d128_prefill_b1_default_cga_gqa", "HN", True, 1),
        ("d128_prefill_b1_gqa16_gqa", "HN", True, 1),
        ("d128_prefill_gqa", "padded", False, 1),
        ("d128_split_b1", "HN", False, 4),
        ("d128_split_b1_gqa", "NH", True, 3),
        ("d128_gqa8_split_gqa", "HN", True, 3),
        ("d128_gqa16_split_gqa", "NH", False, 3),
        ("d128_gqa16_split_gqa", "HN", True, 4),
        ("d128_mha_split", "NH", False, 3),
        ("d64_split", "NH", False, 4),
        ("d64_split", "HN", True, 3),
        ("d64_split_default_cga", "HN", True, 3),
        ("d64_split_default_cga_gqa", "NH", False, 3),
        ("d64_split_b1", "NH", True, 3),
        ("d64_split_b1", "HN", False, 4),
        ("d64_split_gqa", "NH", False, 4),
        ("d64_split_gqa", "HN", True, 3),
        ("d64_split_b1_gqa", "NH", True, 3),
        ("d64_split_b1_gqa", "HN", False, 8),
        ("d256_split", "NH", False, 4),
        ("d256_split", "HN", True, 3),
        ("d256_split_default_cga", "HN", True, 3),
        ("d256_split_b1", "HN", False, 4),
        ("d256_split_b1", "NH", True, 3),
        ("d256_packed_gqa", "NH", False, 1),
        ("d256_gqa8_packed_gqa", "HN", True, 1),
        ("d256_gqa8_packed_b1_gqa", "NH", True, 1),
        ("d256_packed_b1_gqa", "HN", False, 1),
        ("d256_packed_gqa", "padded", False, 1),
    ],
)
def test_paged_thd_split_capture_lengths_and_stats(hnd, dtype, page, geometry, stats_layout, stats_log2, splits):
    """Explicit paged split/packing preserves O/Stats under changed lengths and retained captures."""
    import inspect

    from test_sdpa_fwd_paged_sm100 import _pools

    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Live-length scheduler is admitted on SM100, SM103 and SM107")
    if geometry.startswith("d128_prefill") and "_default_cga" not in geometry and torch.cuda.get_device_capability() == (10, 3):
        pytest.skip("The single-CTA unsplit paged prefill extension is not qualified on SM103")
    has_sink = geometry.startswith("sink_")
    if has_sink:
        if torch.cuda.get_device_capability() != (10, 7):
            pytest.skip("Paged sink-aware split is qualified on SM107")
        geometry = geometry.removeprefix("sink_")
    arch = "sm107" if torch.cuda.get_device_capability() == (10, 7) else "sm100"
    b, h, hk, d, qcap, kcap = (1 if "_b1" in geometry else 3), 8, 2, 128, 1025, 2304
    if has_sink:
        qcap = 9
    if geometry.startswith("d64"):
        if arch == "sm107":
            pytest.skip("Native D64 paged split is qualified on SM100/SM103")
        d = 64
    if geometry.startswith("d256"):
        if arch != "sm107" and splits == 1:
            pytest.skip("Paged D256 unsplit PackGQA is qualified on SM107")
        d = 256
    if "_gqa16_" in geometry:
        h, hk = 16, 1
    elif "_gqa8_" in geometry:
        h, hk = 32, 4
    elif "_mha_" in geometry:
        hk = h
    dt = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    rng = torch.Generator(device=DEV).manual_seed(191)
    _, _, k, v, table = _pools(b, hk, d, page, kcap // page, hnd, dtype, generator=rng)
    if has_sink:
        # Serving caches can concatenate K/V inside each token; preserve the
        # physical HND/NHD layout and test the strided views, without a gather.
        kv = torch.cat((k, v) if hnd else (k.transpose(1, 2), v.transpose(1, 2)), dim=-1)
        k, v = kv.split(d, dim=-1)
        if not hnd:
            k, v = k.transpose(1, 2), v.transpose(1, 2)
    spare = 17 if b == 1 else 0
    q = torch.randn(b * qcap + spare, h, d, device=DEV, dtype=dtype, generator=rng)
    bufs = dict(q=q, k=k, v=v, o=torch.empty_like(q), lse=torch.empty(b * qcap + spare, h, device=DEV))
    if stats_layout == "HN":
        bufs["lse"] = torch.empty(h, b * qcap + 17, device=DEV)
    if stats_layout == "padded":
        bufs["lse"] = torch.empty(b, h, qcap, device=DEV)
    lse_tokens = bufs["lse"].T if stats_layout == "HN" else bufs["lse"]
    bufs.update(
        cu_q=torch.arange(b + 1, device=DEV, dtype=torch.int32) * qcap,
        seq_kv=torch.full((b,), qcap, device=DEV, dtype=torch.int32),
        k_table=table,
        v_table=table.flip(2),
    )
    if has_sink:
        bufs["sink"] = torch.linspace(7, 10, h, device=DEV)
        bufs["sink"][0] = -torch.inf
    bufs["off_q"], bufs["off_lse"] = bufs["cu_q"] * h * d, bufs["cu_q"] * lse_tokens.stride(0)
    g = cudnn.pygraph(
        io_data_type=dt,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        is_override_shape_enabled=stats_layout == "HN",
    )
    t = {n: g.tensor_like(x) for n, x in bufs.items() if n not in ("q", "o", "lse")}
    if has_sink:
        t["sink"].set_dim([1, h, 1, 1]).set_stride([h, 1, 1, 1])
    t["q"] = g.tensor(dim=[b, h, qcap, d], stride=[qcap * h * d, d, h * d, 1], data_type=dt).set_ragged_offset(t["off_q"])
    t["o"], t["lse"] = g.sdpa(
        q=t["q"],
        k=t["k"],
        v=t["v"],
        generate_stats=True,
        attn_scale=1 / math.sqrt(d),
        sink_token=t.get("sink"),
        stats_use_log2=stats_log2,
        use_causal_mask_bottom_right=True,
        use_padding_mask=True,
        cu_seq_len_q=t["cu_q"],
        seq_len_kv=t["seq_kv"],
        # Override-enabled split plans require an explicit total bound.
        max_total_seq_len_q=None if spare and stats_layout == "NH" else b * qcap,
        paged_attention_k_table=t["k_table"],
        paged_attention_v_table=t["v_table"],
        paged_attention_max_seq_len_kv=kcap,
    )
    t["o"].set_output(True).set_dim([b, h, qcap, d]).set_stride([qcap * h * d, d, h * d, 1]).set_ragged_offset(t["off_q"])
    t["lse"].set_output(True).set_dim([b, h, qcap, 1]).set_data_type(cudnn.data_type.FLOAT)
    if stats_layout == "padded":
        t["lse"].set_stride([*bufs["lse"].stride(), 1])
    else:
        t["lse"].set_stride([qcap * h, 1, h, 1] if stats_layout == "NH" else [bufs["lse"].numel(), lse_tokens.stride(1), 1, 1]).set_ragged_offset(t["off_lse"])
    kwargs = {}
    if stats_layout == "HN":
        # The effective Stats geometry includes the extra per-head capacity.
        kwargs = dict(
            override_uids=[t["lse"].get_uid()],
            override_shapes=[[1, h, lse_tokens.shape[0], 1]],
            override_strides=[[bufs["lse"].numel(), lse_tokens.stride(1), 1, 1]],
        )
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    index = next(i for i, name in enumerate(names) if name == engine_name(arch=arch) or name.startswith(engine_name(arch=arch) + "["))
    engine, knobs = g.get_engine_and_knobs_at_index(index)
    captures, workspaces = [], []
    for policy in ((0,) if arch == "sm107" and d == 128 else (0, 1)):
        chosen = {
            **knobs,
            cudnn.knob_type.SCHED_POLICY: policy,
            cudnn.knob_type.PACK_GQA: int(geometry.endswith("_gqa")),
            cudnn.knob_type.TILE_CGA_M: 2 if d == 256 else 1,
            cudnn.knob_type.SPLIT_KV: splits,
        }
        if "_default_cga" in geometry:
            chosen.pop(cudnn.knob_type.TILE_CGA_M)
        g.create_execution_plan(engine, chosen)
        g.build_plan_at_index(g.get_execution_plan_count() - 1)
        api = inspect.getclosurevars(_plan(g)._compiled.default_stream).nonlocals["api"]
        workspace_bytes = g.get_workspace_size()
        spec = _plan(g)._prepared.spec
        assert spec.native is not None
        if splits > 1:
            assert spec.split_workspace.splits == splits
            assert api.packed_thd_split and api._thd_spec.split_workspace == spec.split_workspace
            assert api.template_params().thd_batch_one == (b == 1)
        else:
            assert spec.split_workspace is None and not api.packed_thd_split
        assert workspace_bytes == spec.scratch_bytes == api.scratch_workspace_bytes()
        ws = torch.empty(max(workspace_bytes, 1), device=DEV, dtype=torch.uint8)
        pack = {t[n]: x for n, x in bufs.items()}
        g.execute(pack, ws, **kwargs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            g.execute(pack, ws, **kwargs)
        captures.append(graph)
        workspaces.append(ws)
        from cuda.bindings import driver as cuda_driver

        def standalone(workspace):
            api.execute(
                bufs["q"],
                bufs["k"],
                bufs["v"],
                bufs["o"],
                lse_tensor=bufs["lse"],
                sinks=bufs.get("sink"),
                seq_q_lens=bufs["cu_q"],
                seq_kv_lens=bufs["seq_kv"],
                block_table=bufs["k_table"][:, 0, :, 0],
                block_table_v=bufs["v_table"][:, 0, :, 0],
                workspace=workspace,
                current_stream=cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream),
            )

        if splits > 1:
            with pytest.raises(ValueError, match="caller-owned workspace"):
                standalone(None)
        with pytest.raises(ValueError, match="requires a .* workspace"):
            standalone(ws[:-1])
        standalone(ws)
        standalone_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(standalone_graph):
            standalone(ws)
        captures.append(standalone_graph)
    try:
        lengths = (
            (([1025, 513, 0], [1025, 2049, 0]), ([0, 1025, 513], [0, 1025, 0]), ([257, 0, 1025], [769, 0, 1025]))
            if b > 1
            else (([qcap], [qcap]), ([257], [769]), ([1], [1]), ([128], [0]), ([0], [kcap]), ([qcap], [qcap]))
        )
        if has_sink:
            lengths = (
                (([9, 4, 0], [2049, 1025, 0]), ([0, 9, 4], [0, 0, 2]), ([4, 0, 9], [128, 0, 2304]))
                if b > 1
                else (([9], [2049]), ([4], [2]), ([8], [0]), ([0], [kcap]), ([9], [2304]))
            )
        for iteration, (ql, kl) in enumerate(lengths):
            if has_sink:
                bufs["sink"][1:].sub_(3.0 if iteration else 0.0)
            cq = [0, *accumulate(ql)]
            for name, values in (("cu_q", cq), ("off_q", [x * h * d for x in cq]), ("off_lse", [x * lse_tokens.stride(0) for x in cq]), ("seq_kv", kl)):
                bufs[name].copy_(torch.tensor(values, device=DEV, dtype=torch.int32))
            bufs["q"].mul_(-0.5)
            ref_o = torch.zeros(cq[-1], h, d, device=DEV, dtype=torch.float64)
            ref_s = torch.full((cq[-1], h), -float("inf"), device=DEV, dtype=torch.float64)
            for i, (nq, nk) in enumerate(zip(ql, kl)):
                if not nq or (not nk and not has_sink):
                    continue
                dense = {}
                for n in ("k", "v"):
                    ids = bufs[n + "_table"][i, 0, :, 0].long()
                    dense[n] = bufs[n][ids].transpose(1, 2).reshape(-1, hk, d)[:nk].repeat_interleave(h // hk, 1).double()
                scores = torch.einsum("qhd,khd->hqk", bufs["q"][cq[i] : cq[i + 1]].double(), dense["k"]) / math.sqrt(d)
                scores.masked_fill_(torch.arange(nk, device=DEV)[None, :] > torch.arange(nq, device=DEV)[:, None] + nk - nq, -float("inf"))
                if has_sink:
                    scores = torch.cat((scores, bufs["sink"].double()[:, None, None].expand(h, nq, 1)), -1)
                weights = scores.softmax(-1)[..., :nk].nan_to_num()
                ref_o[cq[i] : cq[i + 1]] = torch.einsum("hqk,khd->qhd", weights, dense["v"])
                ref_s[cq[i] : cq[i + 1]] = scores.logsumexp(-1).T * (math.log2(math.e) if stats_log2 else 1)
            natural = None
            for graph in captures:
                bufs["o"].fill_(float("nan"))
                bufs["lse"].fill_(float("nan"))
                for _ in range(256 if cq[-1] <= 1 else 1):
                    graph.replay()
                torch.cuda.synchronize()
                got_o = bufs["o"][: cq[-1]]
                if stats_layout == "padded":
                    got_s = torch.cat([bufs["lse"][i, :, :nq].T for i, nq in enumerate(ql)])
                    for i, nq in enumerate(ql):
                        assert torch.isneginf(bufs["lse"][i, :, nq:]).all()
                else:
                    got_s = lse_tokens[: cq[-1]]
                    assert torch.isnan(lse_tokens[cq[-1] :]).all()
                torch.testing.assert_close(got_o.float(), ref_o.float(), atol=1.2e-2, rtol=1.2e-2)
                torch.testing.assert_close(got_s, ref_s.float(), atol=1e-3, rtol=1e-3)
                assert torch.isnan(bufs["o"][cq[-1] :]).all()
                if natural is None:
                    natural = got_o.clone(), got_s.clone()
                else:
                    torch.testing.assert_close(got_o, natural[0], atol=0, rtol=0)
                    torch.testing.assert_close(got_s, natural[1], atol=0, rtol=0)
            if has_sink and iteration == 0:
                # Fresh execution binds a new pointer; older captures still own
                # the old binding even after subsequent calls reuse the plan.
                old_sink = bufs["sink"]
                bufs["sink"] = torch.full_like(old_sink, 1000)
                g.execute({t[n]: x for n, x in bufs.items()}, ws, **kwargs)
                torch.cuda.synchronize()
                assert torch.count_nonzero(bufs["o"][: cq[-1]]) == 0
                expected_sink_stats = 1000 * (math.log2(math.e) if stats_log2 else 1)
                torch.testing.assert_close(lse_tokens[: cq[-1]], torch.full_like(lse_tokens[: cq[-1]], expected_sink_stats))
                captures[0].replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(bufs["o"][: cq[-1]], natural[0], atol=0, rtol=0)
                torch.testing.assert_close(lse_tokens[: cq[-1]], natural[1], atol=0, rtol=0)
                bufs["sink"] = old_sink
    finally:
        for graph in captures:
            graph.reset()


@requires_blackwell
@requires_dsl
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("python_binding", [False, True], ids=["native", "python"])
def test_hn_stride_override_reuses_plan_and_old_capture(dtype, python_binding):
    """Vary compact HN stride, then replay older independent captures with no compile or adaptation."""
    b, qmax, kl, hq, hk, d = 2, 16, 64, 8, 2, 128
    if torch.cuda.get_device_capability() == (10, 7):
        pytest.skip("SM107 overlaunch admission is qualified separately")
    arch = "sm100"
    dt = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g, t = _thd_graph(b, qmax, kl, hq, hk, d, override_enabled=True, dtype=dt, arch=arch, stats_head_stride=b * qmax)
    prepared = _plan(g)._prepared
    assert prepared.spec.native is not None
    if python_binding:
        binding_reference.use_reference(prepared.spec)
    ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    retained = []
    for i, ql in enumerate((16, 9, 13, 16)):
        bufs = _buffers(b, ql, kl, hq, hk, d, seed=17 + i, dtype=dtype)
        bufs["lse"] = torch.full((hq, b * ql), float("nan"), device=DEV)
        bufs["off_lse"] = bufs["cu_q"]
        kwargs = dict(
            override_uids=[t["stats"].get_uid()],
            override_shapes=[[b, hq, ql, 1]],
            override_strides=[[hq * b * ql, b * ql, 1, 1]],
        )
        with _Tripwire():
            g.execute(_pack(t, bufs), ws, **kwargs)
            cg = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cg):
                g.execute(_pack(t, bufs), ws, **kwargs)
        torch.cuda.synchronize()
        expected_o, expected_lse = _reference(bufs, b, ql, kl, hq, hk, d)
        torch.testing.assert_close(bufs["o"].float(), expected_o, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(bufs["lse"].T, expected_lse, atol=1e-3, rtol=1e-3)
        assert _plan(g)._prepared is prepared
        retained.append((cg, bufs, ql))
    try:
        for cg, bufs, ql in reversed(retained):
            bufs["q"].mul_(0.75)
            bufs["v"].add_(0.1)
            bufs["o"].fill_(float("nan"))
            bufs["lse"].fill_(float("nan"))
            cg.replay()
            torch.cuda.synchronize()
            expected_o, expected_lse = _reference(bufs, b, ql, kl, hq, hk, d)
            torch.testing.assert_close(bufs["o"].float(), expected_o, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(bufs["lse"].T, expected_lse, atol=1e-3, rtol=1e-3)
    finally:
        for cg, _, _ in retained:
            cg.reset()
    # An HN artifact cannot become NH, nor may override geometry enlarge storage.
    before = bufs["lse"].clone()
    with pytest.raises(ValueError, match="token axis contiguous"):
        g.execute(_pack(t, bufs), ws, override_uids=[t["stats"].get_uid()], override_shapes=[[b, hq, ql, 1]], override_strides=[[hq * ql, 1, hq, 1]])
    with pytest.raises(ValueError, match="storage"):
        g.execute(_pack(t, bufs), ws, override_uids=[t["stats"].get_uid()], override_shapes=[[b, hq, qmax, 1]], override_strides=[[hq * 1024, 1024, 1, 1]])
    torch.testing.assert_close(bufs["lse"], before)


@requires_blackwell
@requires_dsl
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "splits,stats_layout,stats_log2",
    [(1, "NH", False), (2, "HN", False), (3, "NH", True), (8, "HN", True), (3, None, False), (None, "NH", False), (None, "HN", True), (None, None, False)],
)
@pytest.mark.parametrize("batch", [1, 3])
def test_mla_thd_fixed_split_capture(dtype, splits, stats_layout, stats_log2, batch, monkeypatch, cudnn_handle):
    _nonpaged_thd_split_capture(dtype, splits, stats_layout, stats_log2, batch, monkeypatch, cudnn_handle)


def _nonpaged_thd_split_capture(dtype, splits, stats_layout, stats_log2, batch, monkeypatch, cudnn_handle, *, d=192, pack_gqa=False, causal=True, qcap=129):
    """Explicit/automatic plans preserve rebased views, live lengths and output layouts."""
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Nonpaged packed split is admitted on SM100, SM103 and SM107")
    arch = "sm107" if torch.cuda.get_device_capability() == (10, 7) else "sm100"
    b, h, hk, d, dv, kcap = batch, 4, 2, d, (256 if d == 256 else 128), 513
    pitch = dv + 128
    if splits is None:
        if dtype != torch.bfloat16 and d != 128:
            pytest.skip("Automatic nonpaged split placement is currently measured for BF16 (FP16 for D128)")
        hk, kcap = h, 4097
        monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    tq, tk = b * qcap, b * kcap
    spare = 17 if b == 1 else 0
    rng = torch.Generator(device=DEV).manual_seed(192128)
    dt = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    bufs = {
        "q": torch.randn(tq + 3 + spare, h, d, device=DEV, dtype=dtype, generator=rng)[3:],
        "k": torch.randn(tk + 5, hk, d, device=DEV, dtype=dtype, generator=rng)[5:],
    }
    v_storage = torch.randn(tk + 5, hk, pitch, device=DEV, dtype=dtype, generator=rng)
    o_storage = torch.full((tq + 3 + spare, h, pitch), float("nan"), device=DEV, dtype=dtype)
    bufs.update(v=v_storage[5:, :, 128:], o=o_storage[3:, :, 64 : 64 + dv])
    if stats_layout is not None:
        bufs["lse"] = torch.empty((h, tq + 17) if stats_layout == "HN" else (tq + spare, h), device=DEV)
    for name in ("cu_q", "cu_kv", "off_q", "off_k", "off_v", "off_o", "off_lse"):
        bufs[name] = torch.zeros(b + 1, dtype=torch.int32, device=DEV)
    g = cudnn.pygraph(
        io_data_type=dt,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        is_override_shape_enabled=stats_layout == "HN",
    )
    t = {n: g.tensor_like(x) for n, x in bufs.items() if n not in ("q", "k", "v", "o", "lse")}
    for n, heads, cap, width in (("q", h, qcap, d), ("k", hk, kcap, d), ("v", hk, kcap, dv)):
        x = bufs[n]
        t[n] = g.tensor(dim=[b, heads, cap, width], stride=[cap * x.stride(0), x.stride(1), x.stride(0), 1], data_type=dt)
        t[n].set_ragged_offset(t["off_" + n])
    t["o"], stats = g.sdpa(
        q=t["q"],
        k=t["k"],
        v=t["v"],
        generate_stats=stats_layout is not None,
        attn_scale=d**-0.5,
        stats_use_log2=stats_log2,
        use_causal_mask_bottom_right=causal,
        use_padding_mask=True,
        cu_seq_len_q=t["cu_q"],
        cu_seq_len_kv=t["cu_kv"],
        # HN cases exercise overrides for explicit and automatic plans.
        max_total_seq_len_q=None if spare and stats_layout != "HN" else tq,
        max_total_seq_len_kv=tk,
    )
    t["o"].set_output(True).set_dim([b, h, qcap, dv]).set_stride([qcap * h * pitch, pitch, h * pitch, 1]).set_ragged_offset(t["off_o"])
    if stats_layout is not None:
        t["lse"] = stats
        stride = [h * (tq + 17), tq + 17, 1, 1] if stats_layout == "HN" else [qcap * h, 1, h, 1]
        stats.set_output(True).set_dim([b, h, qcap, 1]).set_stride(stride).set_data_type(cudnn.data_type.FLOAT).set_ragged_offset(t["off_lse"])
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    index = next(i for i, name in enumerate(names) if name == engine_name(arch=arch) or name.startswith(engine_name(arch=arch) + "["))
    engine, knobs = g.get_engine_and_knobs_at_index(index)
    if splits is None:
        # Exercise the public default without pinning a particular split count.
        g.build_plans()
    else:
        g.create_execution_plan(
            engine, {**knobs, cudnn.knob_type.TILE_CGA_M: 2 if d == 256 else 1, cudnn.knob_type.SPLIT_KV: splits, cudnn.knob_type.PACK_GQA: int(pack_gqa)}
        )
        g.build_plan_at_index(g.get_execution_plan_count() - 1)
    ws = torch.empty(g.get_workspace_size(), device=DEV, dtype=torch.uint8)
    pack = {t[n]: x for n, x in bufs.items() if n in t}
    # The effective HN descriptor includes the padding in each head's capacity.
    overrides = (
        {}
        if stats_layout != "HN"
        else dict(
            override_uids=[t["lse"].get_uid()],
            override_shapes=[[1, h, tq + 17, 1]],
            override_strides=[[h * (tq + 17), tq + 17, 1, 1]],
        )
    )

    def lengths(ql, kl):
        ql, kl = [min(qcap, q) for q in ql[:b]], kl[:b]
        # Prefix bases describe sliced length arrays; tensor pointers carry physical origins.
        cq, ck = [0, *accumulate(ql)], [0, *accumulate(kl)]
        values = dict(
            cu_q=[x + 3 for x in cq],
            cu_kv=[x + 5 for x in ck],
            off_q=[x * h * d for x in cq],
            off_k=[x * hk * d for x in ck],
            off_v=[x * hk * pitch for x in ck],
            off_o=[x * h * pitch for x in cq],
            off_lse=[x * (1 if stats_layout == "HN" else h) for x in cq],
        )
        for n, x in values.items():
            bufs[n].copy_(torch.tensor(x, device=DEV, dtype=torch.int32))
        return cq, ck

    handle = cudnn_handle
    previous_stream = cudnn.get_stream(handle)

    def execute():
        # The same stream contract also works if a later heuristic chooses backend.
        cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)
        g.execute(pack, ws, handle=handle, **overrides)

    graph = None
    try:
        lengths([65, 129, 0], [257, 513, 0])
        execute()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            execute()
        for ql, kl in (([65, 129, 0], [257, 513, 0]), ([0, 33, 1], [0, 17, 0]), ([1, 0, 65], [1, 0, 129]), ([17, 0, 0], [0, 0, 0])):
            ql, kl = [min(qcap, q) for q in ql[:b]], kl[:b]
            cq, ck = lengths(ql, kl)
            bufs["q"].mul_(-0.5)
            o_storage.fill_(float("nan"))
            lse = None if stats_layout is None else (bufs["lse"].T if stats_layout == "HN" else bufs["lse"])
            if lse is not None:
                lse.fill_(float("nan"))
            # Execute and replay must bind existing storage without constructing new buffers.
            with monkeypatch.context() as m:
                m.setattr(torch, "empty", lambda *args, **kwargs: pytest.fail("execute allocated a tensor"))
                execute()
                graph.replay()
            for i, (nq, nk) in enumerate(zip(ql, kl)):
                if not nq:
                    continue
                q = bufs["q"][cq[i] : cq[i + 1]].double().transpose(0, 1)
                k = bufs["k"][ck[i] : ck[i + 1]].double().transpose(0, 1).repeat_interleave(h // hk, 0)
                v = bufs["v"][ck[i] : ck[i + 1]].double().transpose(0, 1).repeat_interleave(h // hk, 0)
                scores = q @ k.transpose(-1, -2) * d**-0.5
                if causal:
                    scores.masked_fill_(torch.arange(nk, device=DEV)[None, :] > torch.arange(nq, device=DEV)[:, None] + nk - nq, -float("inf"))
                prob = scores.softmax(-1).nan_to_num()
                ref = prob @ v
                bound = torch.finfo(dtype).eps / 2 * (prob @ v.abs() + ref.abs()) + 2e-5
                error = (bufs["o"][cq[i] : cq[i + 1]].transpose(0, 1).double() - ref).abs()
                assert torch.all(error <= bound), float((error / bound).max())
                if lse is not None:
                    ref_lse = scores.logsumexp(-1) * (math.log2(math.e) if stats_log2 else 1)
                    torch.testing.assert_close(lse[cq[i] : cq[i + 1]].T.double(), ref_lse, atol=3e-4, rtol=0)
            assert torch.isnan(o_storage[:3]).all() and torch.isnan(o_storage[3 + cq[-1] :]).all()
            assert torch.isnan(o_storage[..., :64]).all() and torch.isnan(o_storage[..., 64 + dv :]).all()
            if lse is not None:
                assert torch.isnan(lse[: cq[0]]).all() and torch.isnan(lse[cq[-1] :]).all()
    finally:
        if graph is not None:
            graph.reset()
        cudnn.set_stream(handle, previous_stream)


@requires_blackwell
@requires_dsl
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize(
    "splits,stats_layout,stats_log2,pack_gqa,causal",
    [
        (2, "HN", False, False, True),
        (3, "NH", True, True, True),
        (8, "HN", True, True, False),
        (3, None, False, False, False),
        (None, None, False, False, True),
        (None, "NH", False, False, True),
        (None, "HN", True, False, True),
    ],
)
def test_d128_nonpaged_thd_split_capture(dtype, batch, splits, stats_layout, stats_log2, pack_gqa, causal, monkeypatch, cudnn_handle):
    """Ragged D128 reuses packed partials for both packed and unpacked GQA."""
    _nonpaged_thd_split_capture(dtype, splits, stats_layout, stats_log2, batch, monkeypatch, cudnn_handle, d=128, pack_gqa=pack_gqa, causal=causal)


@requires_blackwell
@requires_dsl
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize(
    "splits,stats_layout,stats_log2,causal,qcap",
    [(2, "HN", False, True, 129), (3, "NH", True, False, 129), (8, None, False, True, 129), (8, "HN", True, False, 1)],
)
def test_d256_nonpaged_thd_split_capture(dtype, batch, splits, stats_layout, stats_log2, causal, qcap, monkeypatch, cudnn_handle):
    """The two-CTA D256 partials preserve packed offsets, tails and Stats."""
    _nonpaged_thd_split_capture(dtype, splits, stats_layout, stats_log2, batch, monkeypatch, cudnn_handle, d=256, causal=causal, qcap=qcap)
