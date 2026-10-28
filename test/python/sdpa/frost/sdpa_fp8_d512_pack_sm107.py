# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native D512 role-split packed-row checks, collected by the SM107 entry."""

from dataclasses import replace

import pytest
import torch

import cudnn
import test_sdpa_prepared_fp8 as dense
from sdpa_fp8_d256_pack_sm107 import _check_half_unpacked_control
from sdpa_fp8_d256_padding import _check, _clone_with_slack, _record_route

_CASES = [
    (512, 512, 2, 17, 257, False, False, None, False, 0),
    (384, 336, 4, 129, 512, True, False, None, True, 1),
    (272, 496, 8, 33, 257, True, True, None, True, 0),
    (496, 272, 16, 17, 128, True, False, 64, True, 1),
    (320, 448, 32, 129, 256, True, True, 64, True, 0),
    (512, 512, 4, 513, 1024, True, True, None, False, 1),
    (512, 512, 64, 3, 4096, False, False, None, False, 0),
    (384, 384, 128, 3, 257, True, True, None, True, 1),
]


def _run(case, dtype, output, monkeypatch, *, half=False, pack=True, adapter=False, split=1):
    d, dv, group, sq, skv, causal, bottom_right, window, sink, sched = case
    heads = 2 * group
    seen = _record_route(monkeypatch)
    kwargs = dict(softmax_precision=cudnn.data_type.HALF) if half else {}
    if causal:
        kwargs.update(right_bound=0, diagonal_alignment=cudnn.diagonal_alignment.BOTTOM_RIGHT if bottom_right else cudnn.diagonal_alignment.TOP_LEFT)
    if window is not None:
        kwargs["left_bound"] = window
    sinks = torch.linspace(-5, 8, heads, device="cuda").view(1, heads, 1, 1) if sink else None
    g, vp, _, bufs, tensors = dense._case(
        arch="sm107",
        d=d,
        dv=dv,
        b=2,
        sq=sq,
        skv=skv,
        dtype=dtype,
        output_dtype=output,
        sink=sinks,
        padded=True,
        output_padding=16,
        heads=(heads, 2),
        seq_lens=([sq, max(sq - 3, 0)], [skv, max(skv - 37, 0)]),
        sdpa_kwargs=kwargs,
    )
    selected = g.plans[g._plan_index]
    g.create_execution_plan(selected.engine_id, replace(selected.knobs, pack_gqa=pack, split_kv=split, sched_policy=sched, cga=2))
    g.select_plan(len(g.plans) - 1)
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(1, g.get_workspace_size()), dtype=torch.uint8, device="cuda")
    assert seen and set(seen) == {"sm107/prefill_d512_fp8.py"}, seen
    assert g.plans[g._plan_index].knobs.pack_gqa == pack
    for name in ("q", "k", "v"):
        value = bufs[name]
        value._base[..., value.shape[-1] :].fill_(float("nan"))
        value.copy_((value.float() * 2.5).to(dtype))
    check = dict(causal=causal, bottom_right=bottom_right, window=window, softmax_f16=half, split=split, pack=pack, q_tile_rows=256)
    g.execute(vp, ws)
    _check(bufs, **check)
    if half and split == 1:
        _check_half_unpacked_control(g, bufs, tensors, check)
    if adapter:
        plan = g._compiled_plans[g._plan_index]
        prepared, takes_variant_pack = plan._prepared, plan.takes_variant_pack
        assert prepared is not None
        plan._prepared, plan.takes_variant_pack = None, False
        try:
            g.execute(vp, ws)
            _check(bufs, **check)
        finally:
            plan._prepared, plan.takes_variant_pack = prepared, takes_variant_pack
    assert torch.all(bufs["o_storage"][..., dv:] == 12)
    with dense._cuda_graph() as capture:
        with torch.cuda.graph(capture):
            g.execute(vp, ws)
        fresh = {n: _clone_with_slack(v, 0 if v.dtype == torch.int32 else float("nan")) for n, v in bufs.items() if n != "o_storage"}
        fresh["descale_q"].fill_(1.1)
        fresh["scale_o"].fill_(0.6)
        if sink:
            fresh["sinks"].copy_(sinks.flip(1))
        g.execute({t: fresh[n] for n, t in tensors.items()}, torch.empty_like(ws))
        _check(fresh, **check)
        if half and split == 1:
            _check_half_unpacked_control(g, fresh, tensors, check)
        bufs["seq_q"].copy_(torch.tensor([min(sq, 7), 0], device="cuda", dtype=torch.int32).view(2, 1, 1, 1))
        bufs["seq_kv"].copy_(torch.tensor([min(skv, 129), 0], device="cuda", dtype=torch.int32).view(2, 1, 1, 1))
        bufs["descale_v"].fill_(0.3)
        if sink:
            bufs["sinks"].copy_(sinks.flip(1) - 4)
        for name in ("o", "lse", "amax_o"):
            bufs[name].fill_(float("nan"))
        capture.replay()
        _check(bufs, **check)
        if half and split == 1:
            _check_half_unpacked_control(g, bufs, tensors, check)
        assert torch.all(bufs["o_storage"][..., dv:] == 12)


@pytest.mark.parametrize("case", _CASES, ids=["exact", "causal", "narrow_qk", "narrow_v", "swa", "multiwave", "short_q", "large_group"])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("output", [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2])
def test_native_d512_pack_rebind_replay(case, dtype, output, monkeypatch):
    _run(case, dtype, output, monkeypatch)


@pytest.mark.parametrize("case", [_CASES[1], _CASES[7]], ids=["causal", "large_group"])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("output", [torch.bfloat16, torch.float8_e4m3fn])
def test_native_d512_pack_half_adapter(case, dtype, output, monkeypatch):
    _run(case, dtype, output, monkeypatch, half=True, adapter=True)


@pytest.mark.parametrize("case", [_CASES[1], _CASES[5]], ids=["causal", "multiwave"])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_native_d512_unpacked_lpt(case, dtype, monkeypatch):
    _run(case, dtype, torch.bfloat16, monkeypatch, pack=False)


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"qh_per_kh": 3}, "whole head group"),
        ({"qh_per_kh": 256}, "whole head group"),
        ({"thd_varlen": True, "seq_kv_lens_present": True}, "dense-only"),
        ({"sched_policy": 2}, "NATURAL/LPT"),
    ],
)
def test_native_d512_pack_declines(overrides, message):
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    from cudnn.sdpa.fwd.config_sm107 import make_cfg_d512

    params = dict(dtype_qkv=0, dtype_o=2, cta_mma=2, pack_gqa=True, qh_per_kh=4)
    params.update(overrides)
    with pytest.raises(ValueError, match=message):
        make_cfg_d512(TemplateParams(**params))
