# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""``sdpa_bwd_sm100_d192``: dense D_QK = 192 / D_V = 128 half backward on the accumulation chain."""

import pytest
import torch

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell, select_engine
from test_sdpa_bwd_dsl_sm107 import _TOL, _bshd_empty, _build_graph, _causal_keep, _check, _reference64

pytestmark = [pytest.mark.L0, requires_dsl]
_ENGINE = "sdpa_bwd_sm100_d192"
_D, _DV = 192, 128


def _spec():
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS

    spec = next((s for s in ENGINE_SPECS if s.name == _ENGINE), None)
    assert spec is not None
    return spec


def test_row_ranks_ahead_of_the_d256_row():
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS

    names = [s.name for s in ENGINE_SPECS]
    assert names.index(_ENGINE) < names.index("sdpa_bwd_sm100_d256")
    c = _spec().capabilities
    assert (
        c.d == frozenset({128, 192})
        and c.dqk_ge_dv
        and c.causal
        and not c.gqa
        and not c.swa
        and not c.deterministic
        and c.thd
        and c.thd_declared_totals
        and c.thd_ragged_offsets
    )


@pytest.mark.parametrize(
    "kw,served",
    [
        (dict(), True),
        (dict(causal=False), True),
        (dict(hq=4, hkv=2), False),
        (dict(use_causal_mask=True, diagonal_band_left_bound=128), False),
        (dict(d=128, d_v=128), False),
        (dict(d=256, d_v=256), False),
        (dict(use_causal_mask_bottom_right=True), False),
    ],
    ids=["causal", "dense", "gqa", "window", "d128", "d256", "bottom-right"],
)
def test_declaration_admission(monkeypatch, kw, served):
    from test_sdpa_bwd_d256_sm100 import _decline_reason

    args = dict(d=_D, d_v=_DV)
    args.update(kw)
    reason = _decline_reason(monkeypatch, engine=_ENGINE, **args)
    assert (reason is None) == served, reason


def test_workspace_layout_d192():
    from cudnn.sdpa.bwd.d128_contract import workspace_layout

    regions, total = workspace_layout(2, 3, 300, 192)
    assert regions[0][1] == (2, 384, 3, 192) and regions[0][2] == (384 * 3 * 192, 3 * 192, 192, 1)
    assert regions[1][0] == 4 * 2 * 384 * 3 * 192 and total == 4 * 2 * 3 * 384 * 193 + 240


def _run(dt, sq, sk, causal, b=2, h=3, seed=11, replay=False, replay_count=1):
    keep = _causal_keep(sq, sk, causal=causal)
    g, ports, grad_ports = _build_graph(b=b, hq=h, hkv=h, sq=sq, skv=sk, d=_D, d_v=_DV, dt=dt, **({"use_causal_mask": True} if causal else {}))
    select_engine(g, _ENGINE)
    g.check_support()
    g.build_plans()
    rounded = (sq + 127) // 128 * 128
    assert g.get_workspace_size() == 4 * b * h * rounded * (_D + 1) + 240
    ws = torch.empty(g.get_workspace_size(), dtype=torch.uint8, device="cuda")
    grads = (_bshd_empty(b, sq, h, _D, dt), _bshd_empty(b, sk, h, _D, dt), _bshd_empty(b, sk, h, _DV, dt))

    def inputs(s):
        gen = torch.Generator(device="cpu").manual_seed(s)

        def draw(n, dim):
            return torch.randn(b, n, h, dim, generator=gen).to(device="cuda", dtype=dt).permute(0, 2, 1, 3)

        q, k, v, do = draw(sq, _D), draw(sk, _D), draw(sk, _DV), draw(sq, _DV)
        o64, lse64, dead, *refs = _reference64(q, k, v, do, keep)
        o = _bshd_empty(b, sq, h, _DV, dt)
        o.copy_(o64.to(dt))
        stats = lse64.float().masked_fill(dead, 0).unsqueeze(-1).contiguous()
        return dict(q=q, k=k, v=v, do=do, o=o, stats=stats), refs

    def pack(values):
        return {**{ports[name]: value for name, value in values.items()}, **dict(zip(grad_ports, grads))}

    def poison():
        ws.fill_(255)
        for grad in grads:
            grad.fill_(float("nan"))

    def check(refs):
        for name, got, ref in zip(("dQ", "dK", "dV"), grads, refs):
            if not ref.bool().any():
                assert torch.isfinite(got.float()).all(), f"{name}: non-finite output"
                torch.testing.assert_close(got.float(), ref.float(), atol=_TOL[dt]["atol"], rtol=0)
                continue
            _check(name, got, ref, dt)

    values, refs = inputs(seed)
    bindings = pack(values)
    poison()
    g.execute(bindings, ws)
    torch.cuda.synchronize()
    check(refs)
    if replay:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture, stream=stream):
            g.execute(bindings, ws)
        torch.cuda.current_stream().wait_stream(stream)
        fresh, fresh_refs = inputs(seed + 1)
        for name, value in values.items():
            value.copy_(fresh[name])
        poison()
        for i in range(replay_count):
            capture.replay()
            if i % 8 == 7:
                torch.cuda.current_stream().synchronize()
        torch.cuda.synchronize()
        check(fresh_refs)


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize(
    "sq,sk,causal",
    [(512, 512, False), (512, 512, True), (257, 385, True), (640, 576, False), (1, 129, True), (129, 1, False), (65, 193, True), (2048, 2048, True)],
    ids=["dense", "causal", "causal-tail", "dense-tail", "1x129", "129x1", "65x193", "2k"],
)
def test_d192_accumulation(dt, sq, sk, causal):
    _run(dt, sq, sk, causal)


@requires_pre_rubin_blackwell
def test_d192_causal_empty_kv_tiles():
    # Empty tiles must acknowledge the loader before releasing its K buffer.
    # Repeated replay catches a store racing ahead of the initial empty wait.
    _run(torch.bfloat16, 256, 1024, True, b=4, h=32, replay=True, replay_count=8192)


@requires_pre_rubin_blackwell
def test_d192_replay():
    _run(torch.bfloat16, 640, 640, True, replay=True)
