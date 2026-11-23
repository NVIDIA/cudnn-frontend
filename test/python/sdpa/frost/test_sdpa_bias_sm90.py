# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""SM90 additive bias on the D512 body (engine 20540) and the dedicated D128 template (engine 20541): declared storage,
routing, rebinding and capture."""

import cudnn
import pytest
import torch

from frost_test_utils import _SM, requires_dsl

THREE_WG = "sdpa_fwd_sm90_f16_three_wg"
D128 = "sdpa_fwd_sm90_d128_multi_q"
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
    engine=THREE_WG,
):
    """A graph pinned to ``engine`` (20540 at 64 x 64, or 20541 at 128 x 128) over strided Q/K/V/O, a key-padded bias
    view and strided Stats."""
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

    engine_id = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd").offered_ids()[engine]
    tile = 128 if engine == D128 else 64
    g.create_execution_plan(engine_id, SdpaFwdKnobs(tile_m=tile, tile_n=tile, cga=1, pack_gqa=False, split_kv=1))
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
    "engine,dtype,bias_fp32,broadcast,mask",
    [
        (THREE_WG, torch.float16, False, (1, 1), "none"),
        (THREE_WG, torch.float16, True, (2, 4), "none"),
        (THREE_WG, torch.bfloat16, False, (2, 1), "causal"),
        (THREE_WG, torch.bfloat16, True, (1, 4), "window"),
        (THREE_WG, torch.bfloat16, True, (2, 4), "padded"),
        (D128, torch.float16, False, (1, 1), "none"),
        (D128, torch.float16, False, (2, 4), "causal"),
        (D128, torch.bfloat16, False, (2, 1), "none"),
        (D128, torch.bfloat16, False, (1, 4), "causal"),
    ],
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_sm90_bias_rebind_capture(engine, dtype, bias_fp32, broadcast, mask, monkeypatch, cudnn_handle):
    import cutlass.cute as cute

    g, vp, ws, bufs, tensors, storage = _case(dtype, bias_fp32, broadcast, mask, engine=engine)
    compiled = g._compiled_plans[g._plan_index]
    assert compiled._prepared is not None and compiled._prepared.spec.has_bias
    assert compiled._compiled.kernel_module.PARAMS.has_bias
    assert compiled._compiled.kernel_module.TEMPLATE_NAME == ("prefill_d128_f16" if engine == D128 else "prefill_d512_f16")
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
@pytest.mark.parametrize(
    "dtype,bias_fp32,tile,template",
    [
        (torch.float16, True, None, "prefill_d512_f16"),
        (torch.bfloat16, False, 64, "prefill_d512_f16"),
        (torch.bfloat16, False, None, "prefill_d128_f16"),
        (torch.float16, False, 128, "prefill_d128_f16"),
    ],
    ids=["fp32-bias-d512", "io-bias-pinned-d512", "io-bias-d128", "io-bias-pinned-d128"],
)
def test_sm90_bias_standalone_and_validation(dtype, bias_fp32, tile, template, monkeypatch):
    """The standalone adapter takes the D128 template for an exact (128, 128) bias in the Q/K/V dtype unless a 64 tile is
    pinned, and the D512 body for an FP32 one; either binds, rebinds, captures and refuses a bad bias before launch."""
    g, vp, ws, bufs, tensors, storage = _case(dtype, bias_fp32, (2, 4))
    api = _standalone(bufs, **({} if tile is None else dict(tile_m=tile, tile_n=tile)))
    api.check_support()
    api.compile()
    assert api.scratch_workspace_bytes() == 0
    assert api.kernel_template == template
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
@pytest.mark.parametrize("engine,bias_fp32", [(THREE_WG, True), (D128, False)], ids=["d512-fp32-bias", "d128-io-bias"])
def test_sm90_bias_scale_composition(scale, engine, bias_fp32):
    g, vp, ws, bufs, _, storage = _case(torch.bfloat16, bias_fp32, (2, 4), scale=scale, engine=engine)
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
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "mask,stats,stats_log2", [("none", False, False), ("causal", True, True), ("none", True, False)], ids=["no-stats", "causal-log2", "dense"]
)
def test_sm90_d128_bias_dead_rows_and_stats(dtype, mask, stats, stats_log2):
    """Engine 20541 with a bias in the Q/K/V dtype: an all -inf bias row returns O = 0 and Stats = -inf across every KV
    tile, Stats stay optional and take either log base, and the partial last Q and KV tiles read no bias past S_q or
    S_kv as a logit."""
    g, vp, ws, bufs, _, storage = _case(dtype, False, (2, 4), mask, stats=stats, stats_log2=stats_log2, engine=D128)
    bufs["bias"][:, :, 0, :] = -float("inf")
    bufs["bias"][1, 2, 5, 70:] = -float("inf")
    g.execute(vp, ws)
    _check(bufs, storage, mask, stats_log2=stats_log2)


# The D128 template's records, each under the api_dsl switches that select it at check_support time.
_D128_RECORDS = {
    "v1": dict(D128_INTRA_WG_OVERLAP=False),
    "overlap": dict(D128_PERSISTENT=False),
    "persistent": dict(D128_PERSISTENT_CAUSAL_WALK=1, D128_CAUSAL_WALK_ROUTE=False),
    "persistent-walk": dict(D128_PERSISTENT_CAUSAL_WALK=2, D128_CAUSAL_WALK_ROUTE=False),
}


@requires_hopper
@pytest.mark.parametrize("record", list(_D128_RECORDS))
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
def test_sm90_d128_bias_records(record, causal, monkeypatch):
    """Every D128 record carries the bias slots across its KV tiles and, persistent, across the items a CTA walks: 2 x 16
    heads x 8 Q tiles = 256 items, so each of at most 132 CTAs crosses an item boundary, where the slots' phases carry on
    from the K ring's. The last Q and KV tiles are partial; heads share a KV head four to one."""
    from cudnn.sdpa.fwd import api_dsl

    if record == "persistent-walk" and not causal:
        pytest.skip("the paired walk orders causal records only")
    for name, value in _D128_RECORDS[record].items():
        monkeypatch.setattr(api_dsl, name, value)
    gen = torch.Generator().manual_seed(1377)
    b, h, hk, sq, sk, d = 2, 16, 4, 1000, 1100, 128
    q = (torch.randn(b, h, sq, d, generator=gen) * 0.3).to(torch.bfloat16).cuda()
    k = (torch.randn(b, hk, sk, d, generator=gen) * 0.3).to(torch.bfloat16).cuda()
    v = (torch.randn(b, hk, sk, d, generator=gen) * 0.3).to(torch.bfloat16).cuda()
    bias = (torch.randn(1, h, sq, 1104, generator=gen) * 1.5).to(torch.bfloat16).cuda()[..., :sk]
    o = torch.full_like(q, float("nan"))
    lse = torch.full((b, h, sq), float("nan"), dtype=torch.float32, device="cuda")
    api = api_dsl.SdpaFwdDslSm90(q, k, v, o, sample_lse=lse, sample_bias=bias, scale_softmax=d**-0.5, pack_gqa=False, is_causal=causal)
    api.check_support()
    api.compile()
    assert api.kernel_template == "prefill_d128_f16" and api.params.has_bias and not api.params.causal_claim
    assert api.params.intra_wg_overlap == (record != "v1")
    assert api.params.persistent == record.startswith("persistent"), api.params
    assert api.params.causal_walk == int(record == "persistent-walk"), api.params
    api.execute(q, k, v, o, lse, bias_tensor=bias)
    scores = (q.double() @ k.double().repeat_interleave(h // hk, 1).transpose(-1, -2)) * d**-0.5 + bias.double()
    if causal:
        scores.masked_fill_(torch.arange(sk, device="cuda")[None, :] > torch.arange(sq, device="cuda")[:, None], -float("inf"))
    torch.testing.assert_close(o.double(), scores.softmax(-1) @ v.double().repeat_interleave(h // hk, 1), atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(lse.double(), scores.logsumexp(-1), atol=3e-4, rtol=3e-4)


@requires_hopper
@pytest.mark.parametrize("logit", [3e38, -3e38])
@pytest.mark.parametrize("engine,bias_fp32", [(THREE_WG, True), (D128, False)], ids=["d512-fp32-bias", "d128-io-bias"])
def test_sm90_bias_large_finite_logits(logit, engine, bias_fp32):
    g, vp, ws, bufs, _, storage = _case(torch.bfloat16, bias_fp32, engine=engine)
    bufs["bias"].fill_(logit)
    g.execute(vp, ws)
    _check(bufs, storage, "none")


@requires_hopper
@pytest.mark.parametrize(
    "engine,bias_fp32,fill",
    [(THREE_WG, True, -1e9), (THREE_WG, True, -1e30), (THREE_WG, True, "finfo"), (THREE_WG, False, "finfo"), (D128, False, -1e9), (D128, False, "finfo")],
    ids=["fp32-1e9", "fp32-1e30", "fp32-min", "io-min", "d128-io-1e9", "d128-io-min"],
)
def test_sm90_bias_large_negative_finite_masks(engine, bias_fp32, fill):
    """HF-style masks: masked keys carry a large finite negative bias, not -inf, so a row's logits spread far past the
    fast EX2's input range; the masked keys still get probability 0 and the rest a normal softmax."""
    g, vp, ws, bufs, _, storage = _case(torch.bfloat16, bias_fp32, (2, 1), engine=engine)
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
    assert _caps(D128).bias and _caps(D128).bias_io_dtype_only
    assert not _caps("sdpa_fwd_sm90_d64_one_wg").bias


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


_IO = _Bias(cudnn.data_type.BFLOAT16)
# Engine 20541's gate: (id, fact changes, served, a bias-specific decline names "bias"). Every case is an exact (128, 128)
# graph; the row's own knob domain (one 128 x 128 tile, unsplit, unpacked) refuses split and packed records first.
_D128_GATE = [
    ("io-bias", {"bias_t": _IO}, True, False),
    ("io-bias-per-batch-head", {"bias_t": _IO, "bias_dim": (B, H, SQ, SK), "bias_stride": (H * SQ * PITCH, SQ * PITCH, PITCH, 1)}, True, False),
    ("io-bias-causal", {"bias_t": _IO, "causal": True}, True, False),
    ("fp32-bias", {}, False, True),
    ("io-bias-strided-keys", {"bias_t": _IO, "bias_stride": (SQ * PITCH, SQ * PITCH, PITCH, 2)}, False, True),
    ("io-bias-unaligned-pitch", {"bias_t": _IO, "bias_stride": (SQ * 141, SQ * 141, 141, 1)}, False, True),
    ("mismatched-dtype", {"bias_t": _Bias(cudnn.data_type.HALF)}, False, True),
    ("io-bias-overrides", {"bias_t": _IO, "shape_overrides": True}, False, False),
    ("io-bias-decode", {"bias_t": _IO, "s_q": 1}, False, False),
]


@pytest.mark.parametrize("change,served,names_bias", [case[1:] for case in _D128_GATE], ids=[case[0] for case in _D128_GATE])
def test_sm90_d128_bias_graph_capability_gate(change, served, names_bias):
    """Engine 20541 serves a dense bias in the Q/K/V dtype with aligned pitches; an FP32 bias keeps engine 20540, whose
    gate serves the same graph."""
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs, mismatch

    reason = mismatch(_caps(D128), _facts(**change), SdpaFwdKnobs(tile_m=128, tile_n=128, cga=1, split_kv=1, pack_gqa=False))
    assert (reason is None) == served, reason
    if names_bias:
        assert "bias" in reason, reason


def test_sm90_bias_fp32_and_d64_keep_off_the_narrow_bodies():
    """The dedicated D128 template stages its bias tile in the Q/K/V dtype: the row, the picker and the template refusal
    keep an FP32 bias on engine 20540 and take one in the Q/K/V dtype. The native D64 body declines every bias."""
    from cudnn.sdpa.fwd import config_sm90
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs, mismatch

    tile = SdpaFwdKnobs(tile_m=128, tile_n=128, cga=1)
    assert mismatch(_caps(D128), _facts(), tile) is not None
    assert mismatch(_caps(D128), _facts(bias_t=_IO), tile) is None
    assert mismatch(_caps("sdpa_fwd_sm90_d64_one_wg"), _facts(d_qk=64, d_v=64), SdpaFwdKnobs(tile_m=64, tile_n=64, cga=1)) is not None
    fp32 = config_sm90.declared_features(bias_fp32=True)
    assert config_sm90.pick_flavor(128, 128, config_sm90.FLAVORS, fp32) == config_sm90.D512_FLAVOR
    assert config_sm90.pick_flavor(128, 128, config_sm90.FLAVORS, config_sm90.declared_features()) == config_sm90.D128_FLAVOR
    assert "FP32 additive bias" in config_sm90.d128_decline(128, 128, fp32)
    assert config_sm90.d128_graph_decline(_facts(s_q=256, s_kv=256, bias_t=_IO)) is None
    assert "FP32 additive bias" in config_sm90.d128_graph_decline(_facts(s_q=256, s_kv=256))
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
    config_sm90.validate_params(dataclasses.replace(config_sm90.TemplateParams(), has_bias=True), flavor=config_sm90.D128_FLAVOR)
    with pytest.raises(ValueError, match="FP32 additive bias"):
        config_sm90.validate_params(dataclasses.replace(config_sm90.TemplateParams(), has_bias=True, bias_fp32=True), flavor=config_sm90.D128_FLAVOR)
    # The D128 template refuses these legs by its own table first (config_sm90.D128_NOT_SERVED).
    with pytest.raises(ValueError, match="^SM90 D128 does not serve "):
        config_sm90.validate_params(dataclasses.replace(config_sm90.TemplateParams(), has_bias=True, **change), flavor=config_sm90.D128_FLAVOR)
    claim = dict(has_bias=True, causal=True, intra_wg_overlap=True, persistent=True, causal_claim=config_sm90.D128_CLAIM_NATURAL)
    with pytest.raises(ValueError, match="claim counter"):
        config_sm90.validate_params(dataclasses.replace(config_sm90.TemplateParams(), **claim), flavor=config_sm90.D128_FLAVOR)
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
    """Automatic planning proposes only unsplit, unpacked engine-20540 plans for an FP32 bias graph, and at least one,
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


@pytest.mark.parametrize(
    "change,leads",
    [
        ({"s_q": 256, "s_kv": 256}, True),
        ({"s_q": 1024, "s_kv": 4096, "h_q": 32, "h_kv": 8, "causal": True}, True),
        ({"s_q": 64, "s_kv": 4096}, False),
        ({"s_q": 1}, False),
    ],
    ids=["prefill", "gqa-causal", "below-lead-floor", "decode"],
)
def test_sm90_io_bias_default_discovery_leads_with_d128(change, leads):
    """Automatic planning for an exact (128, 128) bias graph in the Q/K/V dtype leads with engine 20541's 128 x 128 plan
    from config_sm90.D128_LEAD_MIN_SQ query rows, as for a bias-free graph, and keeps engine 20540's unsplit, unpacked
    plans behind it (or alone below that floor and for decode)."""
    from cudnn.engines.manifest import MANIFEST
    from cudnn.sdpa.fwd import heuristics

    offered = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd").offered_ids()
    s_q, s_kv = change.get("s_q", SQ), change.get("s_kv", SK)
    facts = _facts(device_sm_count=132, bias_t=_IO, bias_dim=(1, 1, s_q, s_kv), bias_stride=(s_q * 4096, s_q * 4096, 4096, 1), **change)
    plans = heuristics.recommend("A", facts, offered)
    assert plans
    assert (plans[0].engine_id == offered[D128]) == leads, plans[0]
    for plan in plans:
        assert plan.engine_id in (offered[D128], offered[THREE_WG]), plan
        assert (plan.knobs.split_kv or 1) == 1 and not plan.knobs.pack_gqa, plan
        if plan.engine_id == offered[D128]:
            assert (plan.knobs.tile_m, plan.knobs.tile_n) == (128, 128), plan


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
