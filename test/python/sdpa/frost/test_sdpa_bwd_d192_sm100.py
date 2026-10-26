# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Rectangular half backward: real QK/V widths, poisoned replay and rebinding."""

import pytest
import torch

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell, select_engine
from test_sdpa_bwd_dsl_sm107 import _bshd_empty, _build_graph, _causal_keep, _check, _reference64

pytestmark = [pytest.mark.L0, requires_dsl]
_ENGINE = "sdpa_bwd_sm100_d256"


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize(
    "sq,sk,hk,causal,br,left,budget",
    [
        (256, 512, 4, False, False, None, None),
        (257, 385, 2, True, True, None, None),
        (257, 513, 1, True, False, 128, None),
        (257, 385, 2, False, False, 128, None),
        (257, 385, 4, True, False, None, None),
        (512, 512, 2, True, False, None, 1 << 20),
        (512, 512, 1, False, False, None, 1 << 19),
    ],
    ids=["dense", "gqa-tail-br", "mqa-window", "window-noncausal", "mha-tail", "chunk-partials", "head-subgroups"],
)
def test_rectangular_backward_replay_and_rebind(monkeypatch, dt, sq, sk, hk, causal, br, left, budget):
    if budget is not None:
        from cudnn.sdpa.bwd import api_dsl_sm107

        monkeypatch.setattr(api_dsl_sm107, "_SM107_WS_BUDGET_BYTES", budget)
    b, hq, d, dv = 2, 4, 192, 128
    keep = _causal_keep(sq, sk, causal=causal, bottom_right=br, left=left)
    kwargs = {}
    if causal:
        kwargs["use_causal_mask_bottom_right" if br else "use_causal_mask"] = True
    if left is not None:
        kwargs["diagonal_band_left_bound"] = left
    g, ports, grad_ports = _build_graph(b=b, hq=hq, hkv=hk, sq=sq, skv=sk, d=d, d_v=dv, dt=dt, **kwargs)
    select_engine(g, _ENGINE)
    g.check_support()
    g.build_plans()
    assert g._compiled_plans[g._plan_index]._prepared is not None
    ws = torch.empty(max(1, g.get_workspace_size()), dtype=torch.uint8, device="cuda")
    grads = (_bshd_empty(b, sq, hq, d, dt), _bshd_empty(b, sk, hk, d, dt), _bshd_empty(b, sk, hk, dv, dt))

    def inputs(seed):
        gen = torch.Generator(device="cpu").manual_seed(seed)

        def draw(s, h, dim):
            return torch.randn(b, s, h, dim, generator=gen).to(device="cuda", dtype=dt).permute(0, 2, 1, 3)

        q, k, v, do = draw(sq, hq, d), draw(sk, hk, d), draw(sk, hk, dv), draw(sq, hq, dv)
        o64, lse64, dead, *refs = _reference64(q, k, v, do, keep, hq // hk)
        o = _bshd_empty(b, sq, hq, dv, dt)
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
            _check(name, got, ref, dt)

    values, refs = inputs(31)
    bindings = pack(values)
    poison()
    g.execute(bindings, ws)
    torch.cuda.synchronize()
    check(refs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture, stream=stream):
        g.execute(bindings, ws)
    torch.cuda.current_stream().wait_stream(stream)
    fresh, fresh_refs = inputs(47)
    for name, value in values.items():
        value.copy_(fresh[name])
    poison()
    capture.replay()
    torch.cuda.synchronize()
    check(fresh_refs)
    poison()
    g.execute(pack(fresh), ws)
    torch.cuda.synchronize()
    check(fresh_refs)


@pytest.mark.parametrize(
    "d,dv,thd,served",
    [(192, 128, False, True), (192, 128, True, False), (128, 128, False, False), (192, 192, False, False), (256, 128, False, False), (256, 256, False, True)],
)
def test_exact_head_pair_admission(monkeypatch, d, dv, thd, served):
    from test_sdpa_bwd_d256_sm100 import _decline_reason

    reason = _decline_reason(monkeypatch, d=d, d_v=dv, thd=thd)
    assert (reason is None) == served, reason
