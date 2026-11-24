# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""SM90 additive bias on the D512 body (engine 20540): declared storage, routing, rebinding and capture."""

import cudnn
import pytest
import torch

from frost_test_utils import _SM, requires_dsl

THREE_WG = "sdpa_fwd_sm90_f16_three_wg"
requires_hopper = pytest.mark.skipif(_SM != 90, reason="needs an SM90 Hopper GPU, have " + ("none" if _SM is None else f"sm_{_SM}"))
pytestmark = [pytest.mark.L0, requires_dsl]

B, H, HK, SQ, SK, PITCH = 2, 4, 2, 67, 139, 144


def _case(
    dtype=torch.bfloat16,
    bias_fp32=True,
    broadcast=(1, 1),
    mask="none",
    d=(128, 128),
    scale=None,
    stats=True,
    stats_log2=False,
    sink=False,
):
    """A graph pinned to engine 20540 over strided Q/K/V/O, a key-padded bias view and strided Stats."""
    torch.manual_seed(381)
    d_qk, d_v = d
    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    bufs, tensors, vp, storage = {}, {}, {}, {}
    for name, heads, seq, width in (("q", H, SQ, d_qk), ("k", HK, SK, d_qk), ("v", HK, SK, d_v), ("o", H, SQ, d_v)):
        raw = torch.randn(B, seq, heads, width + 8, dtype=dtype, device="cuda") * 0.4
        if name == "o":
            raw.fill_(123)
        buf = raw[..., :width].transpose(1, 2)
        storage[name], bufs[name] = raw, buf
        if name != "o":
            tensors[name] = g.tensor_like(buf, name=name)
            vp[tensors[name]] = buf
    bdtype = torch.float32 if bias_fp32 else dtype
    raw = torch.randn(*broadcast, SQ, PITCH, dtype=bdtype, device="cuda") * 1.5
    bufs["bias"], storage["bias"] = raw[..., :SK], raw
    tensors["bias"] = g.tensor_like(bufs["bias"], name="bias")
    vp[tensors["bias"]] = bufs["bias"]
    kwargs = {}
    if mask in ("causal", "window"):
        kwargs["use_causal_mask"] = True
    if mask == "band":
        kwargs = {"diagonal_band_right_bound": 7}
    if mask == "padded":
        for name, values in (("seq_q", [51, SQ]), ("seq_kv", [0, 107])):
            buf = torch.tensor(values, dtype=torch.int32, device="cuda").view(B, 1, 1, 1)
            tensors[name] = g.tensor_like(buf, name=name)
            bufs[name], vp[tensors[name]] = buf, buf
        kwargs.update(use_padding_mask=True, seq_len_q=tensors["seq_q"], seq_len_kv=tensors["seq_kv"])
    if mask == "window":
        kwargs["diagonal_band_left_bound"] = 25
    if sink:
        bufs["sinks"] = torch.randn(1, H, 1, 1, device="cuda", dtype=torch.float32)
        tensors["sinks"] = g.tensor_like(bufs["sinks"], name="sink")
        vp[tensors["sinks"]] = bufs["sinks"]
        kwargs["sink_token"] = tensors["sinks"]
    o, stats_tensor = g.sdpa(
        q=tensors["q"],
        k=tensors["k"],
        v=tensors["v"],
        bias=tensors["bias"],
        attn_scale=d_qk**-0.5 if scale is None else scale,
        generate_stats=stats,
        stats_use_log2=stats_log2,
        **kwargs,
    )
    o.set_output(True).set_dim(bufs["o"].shape).set_stride(bufs["o"].stride())
    tensors["o"], vp[o] = o, bufs["o"]
    if stats:
        raw = torch.full((B, H, SQ * 2), float("nan"), device="cuda")
        storage["lse"], bufs["lse"] = raw, raw[..., ::2]
        stats_tensor.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim([B, H, SQ, 1]).set_stride([H * SQ * 2, SQ * 2, 2, 1])
        tensors["lse"], vp[stats_tensor] = stats_tensor, bufs["lse"]
    g.validate()
    g.build_operation_graph()
    from cudnn.engines.manifest import MANIFEST
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    engine_id = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd").offered_ids()[THREE_WG]
    g.create_execution_plan(engine_id, SdpaFwdKnobs(tile_m=64, tile_n=64, cga=1, pack_gqa=False, split_kv=1))
    g.select_plan(len(g.plans) - 1)
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    return g, vp, ws, bufs, tensors, storage


def _check(bufs, storage, mask, scale=None, stats_log2=False):
    """Compare O and Stats with an FP64 reference: logits are scale * Q.K + bias, then masks."""
    q, k, v = (bufs[n].double() for n in ("q", "k", "v"))
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    scores = (q @ k.repeat_interleave(H // HK, 1).transpose(-1, -2)) * scale + bufs["bias"].double()
    qr = torch.arange(q.shape[-2], device="cuda")[:, None]
    kr = torch.arange(k.shape[-2], device="cuda")[None, :]
    if mask in ("causal", "window", "band"):
        # Top-left alignment: the graph API refuses a bottom-right causal mask together with a bias.
        scores.masked_fill_(kr > qr + (7 if mask == "band" else 0), -float("inf"))
    if mask == "padded":
        for batch in range(q.shape[0]):
            scores[batch, :, int(bufs["seq_q"].flatten()[batch]) :] = -float("inf")
            scores[batch, :, :, int(bufs["seq_kv"].flatten()[batch]) :] = -float("inf")
    if mask == "window":
        scores.masked_fill_(kr < qr - 24, -float("inf"))
    logits = scores
    if "sinks" in bufs:
        logits = torch.cat((scores, bufs["sinks"].double().expand(q.shape[0], q.shape[1], q.shape[2], 1)), -1)
    ref = logits.softmax(-1).nan_to_num()[..., : k.shape[-2]] @ v.repeat_interleave(H // HK, 1)
    torch.testing.assert_close(bufs["o"].double(), ref, atol=2e-3, rtol=2e-2)
    if "lse" in bufs:
        expected = logits.logsumexp(-1) * (1.4426950408889634 if stats_log2 else 1)
        torch.testing.assert_close(bufs["lse"].double(), expected, atol=3e-4, rtol=3e-4)
        assert torch.isnan(storage["lse"][..., 1::2]).all()
    assert torch.all(storage["o"][..., bufs["o"].shape[-1] :] == 123)


def _rebind(bufs, vp, tensors, storage, names):
    """Move each named operand to fresh storage with the same declared view."""
    for name in names:
        storage[name] = storage[name].clone()
        if name == "bias":
            bufs[name] = storage[name][..., :SK]
        elif name == "lse":
            bufs[name] = storage[name][..., ::2]
        else:
            bufs[name] = storage[name][..., : bufs[name].shape[-1]].transpose(1, 2)
        vp[tensors[name]] = bufs[name]


def _capture_replay(g, vp, ws, cudnn_handle, mutate, check):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    captured = torch.cuda.CUDAGraph()
    previous_stream = cudnn.get_stream(cudnn_handle)
    try:
        cudnn.set_stream(cudnn_handle, stream.cuda_stream)
        with torch.cuda.graph(captured, stream=stream):
            g.execute(vp, ws, handle=cudnn_handle)
        torch.cuda.current_stream().wait_stream(stream)
        mutate()
        captured.replay()
        check()
    finally:
        cudnn.set_stream(cudnn_handle, previous_stream)
        captured.reset()


@requires_hopper
@pytest.mark.parametrize(
    "dtype,bias_fp32,broadcast,mask",
    [
        (torch.float16, False, (1, 1), "none"),
        (torch.float16, True, (2, 4), "none"),
        (torch.bfloat16, False, (2, 1), "causal"),
        (torch.bfloat16, True, (1, 4), "window"),
        (torch.bfloat16, True, (2, 4), "padded"),
    ],
)
def test_sm90_bias_rebind_capture(dtype, bias_fp32, broadcast, mask, monkeypatch, cudnn_handle):
    import cutlass.cute as cute

    g, vp, ws, bufs, tensors, storage = _case(dtype, bias_fp32, broadcast, mask)
    compiled = g._compiled_plans[g._plan_index]
    assert compiled._prepared is not None and compiled._prepared.spec.has_bias
    assert compiled._compiled.kernel_module.PARAMS.has_bias
    monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("execute must not compile"))
    for iteration in range(2):
        if iteration:
            _rebind(bufs, vp, tensors, storage, ("q", "k", "v", "o", "bias", "lse"))
        g.execute(vp, ws)
        _check(bufs, storage, mask)

        def mutate():
            bufs["bias"].mul_(-0.7)
            bufs["v"].mul_(0.6)
            bufs["o"].fill_(float("nan"))

        _capture_replay(g, vp, ws, cudnn_handle, mutate, lambda: _check(bufs, storage, mask))
    torch.cuda.set_sync_debug_mode("error")
    try:
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        g.execute(vp, ws)
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    finally:
        torch.cuda.set_sync_debug_mode("default")


@requires_hopper
@pytest.mark.parametrize("d", [(256, 256), (512, 512), (96, 96), (192, 128)], ids=["d256", "d512", "d96-envelope", "d192x128-envelope"])
def test_sm90_bias_head_dims(d):
    g, vp, ws, bufs, _, storage = _case(torch.bfloat16, False, (2, 4), "causal", d=d)
    g.execute(vp, ws)
    _check(bufs, storage, "causal")


def _standalone(bufs, scale=128**-0.5, **kw):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    return SdpaFwdDslSm90(
        sample_q=bufs["q"],
        sample_k=bufs["k"],
        sample_v=bufs["v"],
        sample_o=bufs["o"],
        sample_lse=bufs["lse"],
        sample_bias=bufs["bias"],
        scale_softmax=scale,
        pack_gqa=False,
        **kw,
    )


@requires_hopper
@pytest.mark.parametrize("dtype,bias_fp32", [(torch.float16, True), (torch.bfloat16, False)])
def test_sm90_bias_standalone_and_validation(dtype, bias_fp32, monkeypatch):
    g, vp, ws, bufs, tensors, storage = _case(dtype, bias_fp32, (2, 4))
    api = _standalone(bufs)
    api.check_support()
    api.compile()
    assert api.scratch_workspace_bytes() == 0
    assert api.kernel_template == "prefill_d512_f16"
    original = api._dense_spec.fn
    launches = []

    def record(*args):
        launches.append(args)
        return original(*args)

    monkeypatch.setattr(api._dense_spec, "fn", record)
    for iteration in range(2):
        if iteration:
            _rebind(bufs, vp, tensors, storage, ("bias",))
            bufs["bias"].mul_(-0.6)
        api.execute(bufs["q"], bufs["k"], bufs["v"], bufs["o"], bufs["lse"], bias_tensor=bufs["bias"])
        _check(bufs, storage, "none")
        g.execute(vp, ws)
        _check(bufs, storage, "none")
        from cuda.bindings import driver as cuda_driver

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        captured = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(captured, stream=stream):
                api.execute(
                    bufs["q"], bufs["k"], bufs["v"], bufs["o"], bufs["lse"], bias_tensor=bufs["bias"], current_stream=cuda_driver.CUstream(stream.cuda_stream)
                )
            torch.cuda.current_stream().wait_stream(stream)
            bufs["bias"].mul_(1.2)
            captured.replay()
            _check(bufs, storage, "none")
        finally:
            captured.reset()
    assert len(launches) == 4
    for bad in (None, bufs["bias"][..., :-1], bufs["bias"].to(torch.float64), bufs["bias"][..., ::2]):
        bufs["o"].fill_(123)
        bufs["lse"].fill_(456)
        with pytest.raises(ValueError, match="bias"):
            api.execute(bufs["q"], bufs["k"], bufs["v"], bufs["o"], bufs["lse"], bias_tensor=bad)
        assert len(launches) == 4
        assert torch.all(bufs["o"] == 123)
        assert torch.all(bufs["lse"] == 456)
    api.execute(bufs["q"], bufs["k"], bufs["v"], bufs["o"], bufs["lse"], bias_tensor=bufs["bias"])
    _check(bufs, storage, "none")


@requires_hopper
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mask", ["band", "padded"])
def test_sm90_bias_mask_composition(dtype, mask):
    g, vp, ws, bufs, _, storage = _case(dtype, True, (2, 4), mask)
    # A dead row spans every KV tile, the unmasked interior included.
    bufs["bias"][:, :, 0, :] = -float("inf")
    g.execute(vp, ws)
    _check(bufs, storage, mask)


@requires_hopper
@pytest.mark.parametrize("scale", [-0.25, 0.0, 1e-38, 1e12, 3e38])
def test_sm90_bias_scale_composition(scale):
    g, vp, ws, bufs, _, storage = _case(torch.bfloat16, True, (2, 4), scale=scale)
    bufs["bias"].mul_(20)
    if scale > 1e30:
        bufs["q"].mul_(0.001)
    g.execute(vp, ws)
    _check(bufs, storage, "none", scale)
    api = _standalone(bufs, scale=scale)
    api.check_support()
    api.compile()
    for runtime_scale in (scale, scale * 0.5):
        api.execute(bufs["q"], bufs["k"], bufs["v"], bufs["o"], bufs["lse"], bias_tensor=bufs["bias"], scale_softmax=runtime_scale)
        _check(bufs, storage, "none", runtime_scale)


@requires_hopper
@pytest.mark.parametrize("stats,stats_log2,sink", [(False, False, False), (True, True, False), (True, False, True)])
def test_sm90_bias_optional_stats_and_sink(stats, stats_log2, sink):
    g, vp, ws, bufs, _, storage = _case(stats=stats, stats_log2=stats_log2, sink=sink)
    bufs["bias"][:, :, 0, :] = -float("inf")
    g.execute(vp, ws)
    _check(bufs, storage, "none", stats_log2=stats_log2)


@requires_hopper
@pytest.mark.parametrize("logit", [3e38, -3e38])
def test_sm90_bias_large_finite_logits(logit):
    g, vp, ws, bufs, _, storage = _case()
    bufs["bias"].fill_(logit)
    g.execute(vp, ws)
    _check(bufs, storage, "none")


@requires_hopper
@pytest.mark.parametrize(
    "bias_fp32,fill", [(True, -1e9), (True, -1e30), (True, "finfo"), (False, "finfo")], ids=["fp32-1e9", "fp32-1e30", "fp32-min", "io-min"]
)
def test_sm90_bias_large_negative_finite_masks(bias_fp32, fill):
    """HF-style masks: masked keys carry a large finite negative bias, not -inf, so a row's logits spread far past the
    fast EX2's input range; the masked keys still get probability 0 and the rest a normal softmax."""
    g, vp, ws, bufs, _, storage = _case(torch.bfloat16, bias_fp32, (2, 1))
    value = torch.finfo(bufs["bias"].dtype).min if fill == "finfo" else fill
    bufs["bias"][1, :, :, 100:] = value
    bufs["bias"][0, :, :, 10:40] = value
    g.execute(vp, ws)
    _check(bufs, storage, "none")


@requires_hopper
@pytest.mark.parametrize("sink_logit", [-float("inf"), float("inf")])
def test_sm90_bias_sink_endpoints(sink_logit, cudnn_handle):
    g, vp, ws, bufs, _, storage = _case(sink=True)
    bufs["bias"][:, :, 0, :] = -float("inf")
    bufs["sinks"].fill_(sink_logit)
    g.execute(vp, ws)
    _check(bufs, storage, "none")

    def mutate():
        bufs["sinks"].fill_(0.7)

    _capture_replay(g, vp, ws, cudnn_handle, mutate, lambda: _check(bufs, storage, "none"))


@requires_hopper
def test_sm90_bias_declined_on_plain_specialization():
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    _, _, _, bufs, _, storage = _case()
    api = SdpaFwdDslSm90(bufs["q"], bufs["k"], bufs["v"], bufs["o"], sample_lse=bufs["lse"], pack_gqa=False)
    api.check_support()
    api.compile()
    bufs["o"].fill_(123)
    bufs["lse"].fill_(456)
    with pytest.raises(ValueError, match="without a bias"):
        api.execute(bufs["q"], bufs["k"], bufs["v"], bufs["o"], bufs["lse"], bias_tensor=bufs["bias"])
    assert torch.all(bufs["o"] == 123)
    assert torch.all(bufs["lse"] == 456)
    api.execute(bufs["q"], bufs["k"], bufs["v"], bufs["o"], bufs["lse"])
    _check({**bufs, "bias": torch.zeros_like(bufs["bias"])}, storage, "none")


# ---- Host-only admission: no GPU needed.


class _Bias:
    def __init__(self, dtype=cudnn.data_type.FLOAT):
        self._dtype = dtype

    def get_data_type(self):
        return self._dtype


def _facts(**change):
    from cudnn.sdpa import graph_analyzer as ga

    s_q, s_kv = change.get("s_q", SQ), change.get("s_kv", SK)
    base = dict(
        b=B,
        h_q=H,
        h_kv=HK,
        s_q=SQ,
        s_kv=SK,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.BFLOAT16,
        dtype_o=cudnn.data_type.BFLOAT16,
        device_cc=(9, 0),
        has_bias=True,
        bias_t=_Bias(),
        bias_dim=(1, 1, s_q, s_kv),
        bias_stride=(s_q * PITCH, s_q * PITCH, PITCH, 1),
    )
    base.update(change)
    return ga.SdpaGraphFacts(**base)


def _caps(name):
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    return next(s.capabilities for s in ENGINE_SPECS if s.name == name)


def test_sm90_bias_row_claim():
    assert _caps(THREE_WG).bias
    assert not _caps(THREE_WG).bias_io_dtype_only
    for name in ("sdpa_fwd_sm90_d64_one_wg", "sdpa_fwd_sm90_d128_multi_q"):
        assert not _caps(name).bias, name


# (id, fact changes, knob changes, served, a bias-specific decline names "bias")
_GATE = [
    ("d128", {}, {}, True, False),
    ("d256", {"d_qk": 256, "d_v": 256}, {}, True, False),
    ("d512-io-bias", {"d_qk": 512, "d_v": 512, "bias_t": _Bias(cudnn.data_type.BFLOAT16)}, {}, True, False),
    ("d512-fp32-bias", {"d_qk": 512, "d_v": 512}, {}, False, True),
    ("io-bias", {"bias_t": _Bias(cudnn.data_type.BFLOAT16)}, {}, True, False),
    ("per-batch-head", {"bias_dim": (B, H, SQ, SK), "bias_stride": (H * SQ * PITCH, SQ * PITCH, PITCH, 1)}, {}, True, False),
    ("thd", {"thd": True}, {}, False, False),
    ("paged", {"has_paged_kv": True}, {}, False, False),
    ("overrides", {"shape_overrides": True}, {}, False, False),
    ("split", {}, {"split_kv": 2}, False, True),
    ("pack-gqa", {}, {"pack_gqa": True}, False, True),
    ("strided-keys", {"bias_stride": (SQ * PITCH, SQ * PITCH, PITCH, 2)}, {}, False, True),
    ("unaligned-pitch", {"bias_stride": (SQ * 141, SQ * 141, 141, 1)}, {}, False, True),
    ("mismatched-dtype", {"bias_t": _Bias(cudnn.data_type.HALF)}, {}, False, True),
    ("native-d64", {"d_qk": 64, "d_v": 64}, {}, False, False),
]


@pytest.mark.parametrize("change,knobs,served,names_bias", [case[1:] for case in _GATE], ids=[case[0] for case in _GATE])
def test_sm90_bias_graph_capability_gate(change, knobs, served, names_bias):
    """Engine 20540 serves a dense, unsplit, unpacked bias in FP32 or the io dtype, with aligned pitches."""
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs, mismatch

    record = SdpaFwdKnobs(**{"tile_m": 64, "tile_n": 64, "cga": 1, "split_kv": 1, "pack_gqa": False, **knobs})
    reason = mismatch(_caps(THREE_WG), _facts(**change), record)
    assert (reason is None) == served, reason
    if names_bias:
        assert "bias" in reason, reason


def test_sm90_bias_keeps_off_the_narrow_bodies():
    """The D64 and dedicated D128 rows decline a bias graph, so it falls back to engine 20540."""
    from cudnn.sdpa.fwd import config_sm90
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs, mismatch

    assert mismatch(_caps("sdpa_fwd_sm90_d128_multi_q"), _facts(), SdpaFwdKnobs(tile_m=128, tile_n=128, cga=1)) is not None
    assert mismatch(_caps("sdpa_fwd_sm90_d64_one_wg"), _facts(d_qk=64, d_v=64), SdpaFwdKnobs(tile_m=64, tile_n=64, cga=1)) is not None
    assert config_sm90.pick_flavor(128, 128, config_sm90.FLAVORS, config_sm90.declared_features(bias=True)) == config_sm90.D512_FLAVOR
    assert "additive bias" in config_sm90.d128_decline(128, 128, config_sm90.declared_features(bias=True))
    assert "bias" in config_sm90.native_d64_mismatch(bias=True)


@pytest.mark.parametrize(
    "change",
    [{"thd_varlen": True, "seq_kv_lens_present": True}, {"page_size": 64, "seq_kv_lens_present": True}, {"split_kv": 2}, {"pack_gqa": True, "qh_per_kh": 2}],
)
def test_sm90_bias_template_refuses_unserved_legs(change):
    import dataclasses
    from cudnn.sdpa.fwd import config_sm90

    config_sm90.validate_params(dataclasses.replace(config_sm90.TemplateParams(), has_bias=True))
    config_sm90.validate_params(dataclasses.replace(config_sm90.TemplateParams(), has_bias=True, bias_fp32=True, head_tile=256))
    with pytest.raises(ValueError, match="FP32 bias tile"):
        config_sm90.validate_params(dataclasses.replace(config_sm90.TemplateParams(), has_bias=True, bias_fp32=True, head_tile=512))
    with pytest.raises(ValueError, match="bias"):
        config_sm90.validate_params(dataclasses.replace(config_sm90.TemplateParams(), has_bias=True, **change))
    with pytest.raises(ValueError, match="bias_fp32"):
        config_sm90.validate_params(dataclasses.replace(config_sm90.TemplateParams(), bias_fp32=True))


# Short-Q GQA over a long KV: every bias-free mode-A set of engine 20540 splits or packs, which the bias leg refuses.
_SHORT_Q_GQA = {"b": 4, "h_q": 16, "h_kv": 4, "s_q": 32, "s_kv": 1024}
_DECODE_GQA = {"b": 1, "h_q": 32, "h_kv": 4, "s_q": 1, "s_kv": 512}


@pytest.mark.parametrize(
    "change",
    [{}, {"s_q": 1}, {"h_q": 32, "h_kv": 4}, {"s_q": 4096, "s_kv": 4096, "d_qk": 256, "d_v": 256, "causal": True}, _SHORT_Q_GQA],
    ids=["prefill", "decode", "gqa8", "d256-causal", "short-q-gqa-long-kv"],
)
def test_sm90_bias_default_discovery_offers_the_d512_body(change):
    """Automatic planning proposes only unsplit, unpacked engine-20540 plans for a bias graph, and at least one,
    including where every bias-free set splits or packs."""
    from cudnn.engines.manifest import MANIFEST
    from cudnn.sdpa.fwd import heuristics

    offered = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd").offered_ids()
    plans = heuristics.recommend("A", _facts(device_sm_count=132, **change), offered)
    assert plans
    for plan in plans:
        assert plan.engine_id == offered[THREE_WG], plan
        assert (plan.knobs.split_kv or 1) == 1 and not plan.knobs.pack_gqa, plan


_MODE_A_SHAPES = [
    ("b4-h16-kv4-q32-k1k", _SHORT_Q_GQA),
    ("b1-h32-kv4-q1-k512", _DECODE_GQA),
    ("d256-q4-k4k", {**_DECODE_GQA, "s_q": 4, "s_kv": 4096, "d_qk": 256, "d_v": 256}),
    ("q16-k32k", {**_DECODE_GQA, "s_q": 16, "s_kv": 32768}),
]


@pytest.mark.parametrize(
    "change, bias_dtype",
    [
        pytest.param(change, dtype, id=f"{name}-{label}")
        for name, change in _MODE_A_SHAPES
        for label, dtype in (("fp32-bias", cudnn.data_type.FLOAT), ("io-bias", cudnn.data_type.BFLOAT16))
    ]
    + [pytest.param({**_DECODE_GQA, "s_q": 67, "s_kv": 4096, "d_qk": 512, "d_v": 512}, cudnn.data_type.BFLOAT16, id="d512-q67-k4k-io-bias")],
)
def test_sm90_bias_mode_a_keeps_an_unsplit_unpacked_plan(change, bias_dtype):
    """Mode A still proposes a plan for a short-Q GQA bias graph whose bias-free sets all split or pack, and every plan it
    proposes is unsplit and unpacked, the bias leg's domain (#2872 review: B1, H_q 32 / H_kv 4, S_q 1, S_kv 512)."""
    from cudnn.engines.manifest import MANIFEST
    from cudnn.sdpa.fwd import heuristics

    s_q, s_kv = change["s_q"], change["s_kv"]
    pitch = -(-s_kv // 8) * 8
    facts = _facts(
        device_sm_count=132,
        bias_t=_Bias(bias_dtype),
        bias_dim=(1, 1, s_q, s_kv),
        bias_stride=(s_q * pitch, s_q * pitch, pitch, 1),
        **change,
    )
    offered = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd").offered_ids()
    plans = heuristics.recommend("A", facts, offered)
    assert plans
    for plan in plans:
        assert (plan.knobs.split_kv or 1) == 1 and not plan.knobs.pack_gqa, plan


def test_sm90_bias_fields_are_appended_last():
    """has_bias and bias_fp32 follow the D128 claim counter, defaulting to False: every earlier specialization keeps its
    defaults (the append-only rule)."""
    import dataclasses
    from cudnn.sdpa.fwd import config_sm90 as c

    fields = dataclasses.fields(c.TemplateParams)
    names = [f.name for f in fields]
    start = names.index("has_bias")
    assert start > names.index("causal_claim"), names
    assert [(f.name, f.default) for f in fields[start : start + 2]] == [("has_bias", False), ("bias_fp32", False)]
