# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Explicit one-Q topology: public identity, origins, replay and rescaling."""

import math

import pytest


def _verify_plan(graph):
    import cudnn

    plan = graph._compiled_plans[graph._plan_index]
    module = plan._compiled.kernel_module
    assert module.__file__.endswith("sm107/prefill_d256_f16.py")
    assert module.CFG.TILE_K == module.CFG.TILE_O == 128
    assert module.CFG.TILES_Q == module.CFG.SOFTMAX_WARPGROUPS == 1
    assert module.CFG.THREADS_PER_CTA == 384 and module.CFG.CTA_MMA == 1
    assert module.PARAMS.softmax_wgs == 1 and module.PARAMS.s_slots == 2
    assert plan._prepared.spec.origin_specs is not None
    engine, public = graph.get_engine_and_knobs_at_index(graph._plan_index)
    assert public[cudnn.knob_type.WARP_SPEC_CFG] == 1
    return engine, public


@pytest.mark.parametrize(
    "dtype_name,stats_layout,policy,total_hints",
    [("bfloat16", "token", 0, False), ("float16", "head", 0, True), ("bfloat16", "head", 1, True), ("float16", "token", 1, False)],
    ids=["bf16-token-natural", "fp16-head-natural", "bf16-head-lpt", "fp16-token-lpt"],
)
def test_one_q_origins(dtype_name, stats_layout, policy, total_hints, monkeypatch):
    import cudnn
    import torch
    from test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture as run

    create, build, execute = cudnn.pygraph.create_execution_plan, cudnn.pygraph.build_plans, cudnn.pygraph.execute
    checked = []

    def select(graph, engine_id, knobs=None):
        return create(graph, engine_id, {**(knobs or {}), cudnn.knob_type.WARP_SPEC_CFG: 1, cudnn.knob_type.SCHED_POLICY: policy})

    def verify(graph, *args, **kwargs):
        result = build(graph, *args, **kwargs)
        checked.append(_verify_plan(graph))
        return result

    def execute_without_sync(graph, *args, **kwargs):
        previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            return execute(graph, *args, **kwargs)
        finally:
            torch.cuda.set_sync_debug_mode(previous)

    monkeypatch.setattr(cudnn.pygraph, "create_execution_plan", select)
    monkeypatch.setattr(cudnn.pygraph, "build_plans", verify)
    monkeypatch.setattr(cudnn.pygraph, "execute", execute_without_sync)
    run(128, dtype_name, stats_layout, total_hints, False, 1, engine_name="sdpa_fwd_prefill_sm107", cga=1)
    assert checked


def test_one_q_public_identity_and_domain(monkeypatch):
    import cudnn
    from cudnn.sdpa.fwd.config_sm107 import supports_native_d128_single_q
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    import dataclasses
    import torch
    from cudnn.frost import buffers
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100
    from test_sdpa_fwd_dsl_sm107 import _caps, _desc, _f16_facts, _fake_cc

    _fake_cc(monkeypatch, (10, 7))
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    caps = _caps("sdpa_fwd_prefill_sm107")
    facts = _f16_facts(thd=True, padded=True)
    knobs = SdpaFwdKnobs(cga=1, warp_spec_cfg=1)
    assert engines.mismatch(caps, facts, knobs) is None
    for change in ({"cga": 2}, {"tile_n": 64}, {"split_kv": 2}, {"pack_gqa": True}, {"warp_spec_cfg": 2}):
        assert engines.mismatch(caps, facts, dataclasses.replace(knobs, **change)) is not None
    assert engines.mismatch(_caps("sdpa_fwd_prefill_sm100"), dataclasses.replace(facts, device_cc=(10, 0)), knobs) is not None
    args = tuple(_desc((2, 8, 512, 128), torch.float16, name) for name in ("q", "k", "v", "o"))
    api = SdpaFwdDslSm100(*args, None, thd=True, seq_kv_lens_present=True, cga=1, softmax_wgs=1)
    assert api.check_support()
    params = api.template_params()
    assert params.softmax_wgs == 1 and params.s_slots == 2 and params.q_in_tmem == 0
    for change in ({"cga": 2}, {"tile_n": 64}, {"thd": False}, {"split_kv": 2}, {"pack_gqa": True}):
        invalid_api = SdpaFwdDslSm100(*args, None, **{**dict(thd=True, seq_kv_lens_present=True, cga=1, softmax_wgs=1), **change})
        with pytest.raises((ValueError, NotImplementedError)):
            invalid_api.check_support()

    public = {cudnn.knob_type.WARP_SPEC_CFG: 1, cudnn.knob_type.TILE_CGA_M: 1, cudnn.knob_type.TILE_N: 128}
    assert SdpaFwdKnobs.from_public(public).to_public() == public
    assert cudnn.knob_type.WARP_SPEC_CFG not in SdpaFwdKnobs(cga=1, tile_n=128).to_public()
    assert SdpaFwdKnobs.from_public({cudnn.knob_type.WARP_SPEC_CFG: 0}).warp_spec_cfg == 0
    valid = dict(d_shape=(128, 128), device_cc=(10, 7), half_precision=True, thd=True, paged=False, cga=1)
    assert supports_native_d128_single_q(**valid)
    for change in (
        {"d_shape": (64, 64)},
        {"d_shape": (256, 256)},
        {"device_cc": (10, 0)},
        {"half_precision": False},
        {"thd": False},
        {"paged": True},
        {"cga": 2},
        {"cga": None},
        {"tile_m": 64},
        {"tile_n": 64},
        {"split_kv": 2},
        {"pack_gqa": True},
        {"has_bias": True},
        {"has_kv_mask": True},
        {"has_gate": True},
        {"mixed_kv": True},
    ):
        assert not supports_native_d128_single_q(**{**valid, **change}), change
    for invalid in (True, 1.0, "1"):
        with pytest.raises(ValueError):
            SdpaFwdKnobs.from_public({cudnn.knob_type.WARP_SPEC_CFG: invalid})


@pytest.mark.parametrize(
    "dtype_name,bottom_right,window_left,window_right,policy",
    [("bfloat16", False, None, 0, 0), ("float16", True, None, 0, 1), ("bfloat16", True, None, 17, 1), ("float16", True, 192, 0, 0)],
    ids=["bf16-top-left", "fp16-bottom-right", "bf16-right-band", "fp16-window"],
)
def test_one_q_forced_rescale(dtype_name, bottom_right, window_left, window_right, policy):
    import cudnn
    import torch
    from cudnn.engines.manifest import family_for

    torch.manual_seed(671)
    dtype = getattr(torch, dtype_name)
    hq, hk, sq, sk, dim = 8, 2, 513, 641, 128
    q = torch.zeros(1, sq, hq, dim, dtype=dtype, device="cuda").transpose(1, 2)
    k = torch.zeros(1, sk, hk, dim, dtype=dtype, device="cuda").transpose(1, 2)
    v = (torch.randn(1, sk, hk, dim, dtype=dtype, device="cuda") * 0.125).transpose(1, 2)
    q[..., 0] = 1
    k[..., 0] = (torch.arange(sk, device="cuda") // 128).to(dtype) * (12 * math.log(2) * 8)
    o = torch.empty_like(q)
    lse = torch.empty(sq, hq, dtype=torch.float32, device="cuda")
    ql = torch.tensor([sq], dtype=torch.int32, device="cuda").view(1, 1, 1, 1)
    kl = torch.tensor([sk], dtype=torch.int32, device="cuda").view(1, 1, 1, 1)
    tensors = dict(q=q, k=k, v=v, o=o)
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF if dtype_name == "float16" else cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    nodes = {name: graph.tensor_like(value) for name, value in tensors.items() if name != "o"}
    offsets = {name: torch.tensor([0, value.shape[2] * value.stride(2)], dtype=torch.int64, device="cuda").view(2, 1, 1, 1) for name, value in tensors.items()}
    offsets["stats"] = torch.tensor([0, sq * hq], dtype=torch.int64, device="cuda").view(2, 1, 1, 1)
    offset_nodes = {name: graph.tensor_like(value) for name, value in offsets.items()}
    for name in ("q", "k", "v"):
        nodes[name].set_ragged_offset(offset_nodes[name])
    nql, nkl = graph.tensor_like(ql), graph.tensor_like(kl)
    extra = dict(diagonal_alignment=cudnn.diagonal_alignment.BOTTOM_RIGHT) if bottom_right else {}
    if window_left is not None:
        extra["diagonal_band_left_bound"] = window_left
    nodes["o"], stats = graph.sdpa(
        q=nodes["q"],
        k=nodes["k"],
        v=nodes["v"],
        is_inference=False,
        attn_scale=0.125,
        diagonal_band_right_bound=window_right,
        use_padding_mask=True,
        seq_len_q=nql,
        seq_len_kv=nkl,
        **extra,
    )
    nodes["o"].set_output(True).set_dim(o.shape).set_stride(o.stride()).set_ragged_offset(offset_nodes["o"])
    stats.set_output(True).set_dim((1, hq, sq, 1)).set_stride((sq * hq, 1, hq, 1)).set_data_type(cudnn.data_type.FLOAT).set_ragged_offset(offset_nodes["stats"])
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plan(
        family_for(graph).offered_ids()["sdpa_fwd_prefill_sm107"],
        {cudnn.knob_type.WARP_SPEC_CFG: 1, cudnn.knob_type.TILE_CGA_M: 1, cudnn.knob_type.SCHED_POLICY: policy},
    )
    graph.select_plan(0)
    graph.check_support()
    graph.build_plans()
    _verify_plan(graph)
    workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda")
    pack = {nodes[name]: value for name, value in tensors.items()}
    pack.update({offset_nodes[name]: value for name, value in offsets.items()})
    pack.update({stats: lse, nql: ql, nkl: kl})
    qr = q.double().cpu()[0]
    kr = k.double().cpu()[0].repeat_interleave(hq // hk, dim=0)
    vr = v.double().cpu()[0].repeat_interleave(hq // hk, dim=0)
    scores = qr @ kr.transpose(-1, -2) * 0.125
    assert ((scores[:, -1, 128] - scores[:, -1, 0]) * math.log2(math.e) > 8).all()
    rows = torch.arange(sq)[:, None] + (sk - sq if bottom_right else 0)
    cols = torch.arange(sk)[None, :]
    keep = cols <= rows + window_right
    if window_left is not None:
        keep &= cols > rows - window_left
    scores.masked_fill_(~keep, -float("inf"))
    expected = scores.softmax(-1) @ vr
    expected_lse = scores.logsumexp(-1).T

    def check():
        torch.testing.assert_close(o.double().cpu()[0], expected, atol=0.025 if dtype_name == "bfloat16" else 0.004, rtol=0.02)
        torch.testing.assert_close(lse.double().cpu(), expected_lse, atol=0.003, rtol=0.003)

    graph.execute(pack, workspace)
    torch.cuda.synchronize()
    check()
    captured = torch.cuda.CUDAGraph()
    with torch.cuda.graph(captured):
        graph.execute(pack, workspace)
    o.fill_(float("nan"))
    lse.fill_(float("nan"))
    workspace.fill_(255)
    captured.replay()
    torch.cuda.synchronize()
    check()
