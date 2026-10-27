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
    check_backward_replay_and_rebind(monkeypatch, dt, sq, sk, hk, causal, br, left, budget)


def check_backward_replay_and_rebind(monkeypatch, dt, sq, sk, hk, causal, br, left, budget, *, d=192, dv=128, deterministic=False):
    """Check native geometry, FP64 gradients and replay after changing data/pointers."""
    if budget is not None:
        from cudnn.sdpa.bwd import api_dsl_sm107

        monkeypatch.setattr(api_dsl_sm107, "_SM107_WS_BUDGET_BYTES", budget)
    from cudnn.sdpa.bwd.api_dsl_sm100_d256 import SdpaBwdDslSm100D256

    compiled_geometry = []
    original_compile = SdpaBwdDslSm100D256._compile_plan

    def record_compile(api, main, mm_dk, mm_dq):
        compiled_geometry.append((main.CFG.TILE_K, main.CFG.TILE_O))
        assert compiled_geometry[-1] == (d, dv), "the actual compiled body must use native geometry"
        for stage3 in (mm_dk, mm_dq):
            assert stage3.PARAMS.cgrp_tile_mn == ((128, 192) if d == 192 else (256, 256))
            assert stage3.cta_group == (1 if d == 192 else 2)
        return original_compile(api, main, mm_dk, mm_dq)

    monkeypatch.setattr(SdpaBwdDslSm100D256, "_compile_plan", record_compile)
    b, hq = 2, 4
    keep = _causal_keep(sq, sk, causal=causal, bottom_right=br, left=left)
    kwargs = dict(use_deterministic_algorithm=deterministic)
    if causal:
        kwargs["use_causal_mask_bottom_right" if br else "use_causal_mask"] = True
    if left is not None:
        kwargs["diagonal_band_left_bound"] = left
    g, ports, grad_ports = _build_graph(b=b, hq=hq, hkv=hk, sq=sq, skv=sk, d=d, d_v=dv, dt=dt, **kwargs)
    select_engine(g, _ENGINE)
    g.check_support()
    g.build_plans()
    assert compiled_geometry
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
    if deterministic:
        expected = [grad.clone() for grad in grads]
        for _ in range(2):
            poison()
            g.execute(bindings, ws)
            torch.cuda.synchronize()
            check(refs)
            for got, want in zip(grads, expected):
                assert torch.equal(got.view(torch.int16), want.view(torch.int16))
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
    expected = [grad.clone() for grad in grads] if deterministic else None
    poison()
    g.execute(pack(fresh), ws)
    torch.cuda.synchronize()
    check(fresh_refs)
    if deterministic:
        for got, want in zip(grads, expected):
            assert torch.equal(got.view(torch.int16), want.view(torch.int16)), "rebind differs from retained replay"
        poison()
        capture.replay()
        torch.cuda.synchronize()
        check(fresh_refs)
        for got, want in zip(grads, expected):
            assert torch.equal(got.view(torch.int16), want.view(torch.int16)), "retained replay changed raw gradient bits"


@pytest.mark.parametrize(
    "d,dv,thd,served",
    [(192, 128, False, True), (192, 128, True, True), (128, 128, False, False), (192, 192, False, False), (256, 128, False, False), (256, 256, False, True)],
)
def test_exact_head_pair_admission(monkeypatch, d, dv, thd, served):
    from test_sdpa_bwd_d256_sm100 import _decline_reason

    reason = _decline_reason(monkeypatch, d=d, d_v=dv, thd=thd)
    assert (reason is None) == served, reason


@pytest.mark.parametrize("profile,thd,dqk,dv", [(2, False, 192, 128), (1, False, 128, 128), (1, False, 256, 128)])
def test_unqualified_native_geometry_is_rejected(profile, thd, dqk, dv):
    from cudnn.sdpa.bwd.config_d256_2x2 import FAMILY_F16, TemplateParams2x2, make_cfg_d256_2x2

    params = TemplateParams2x2(datapath_2x2_profile=profile, thd_varlen=thd, head_dim_qk=dqk, head_dim_v=dv)
    with pytest.raises(ValueError, match="head geometry"):
        make_cfg_d256_2x2(params, FAMILY_F16)


@pytest.mark.parametrize("sm", [100, 103, 106, 107, 119])
@pytest.mark.parametrize("d,dv", [(192, 128), (256, 256)])
def test_stage3_native_width_stays_on_the_sm100_half_line(sm, d, dv):
    from types import SimpleNamespace

    from cudnn.sdpa.bwd.api_dsl_sm100_d256 import SdpaBwdDslSm100D256

    api = SimpleNamespace(head_dim_qk=d, head_dim_v=dv)
    tile = SdpaBwdDslSm100D256._stage3_tile_mn(api, sm)
    if sm <= 106 and d == 192:
        assert tile == (128, 192)
    else:
        assert tile == (256, 256), "D256 and other architecture lines retain their existing stage3 tile"


def test_stage3_native_width_keeps_fp8_on_its_validated_tile():
    from cudnn.frost.tile_dsl.constants import DTYPE_E4M3
    from cudnn.sdpa.bwd.config_sm100 import EPI_QUANT, MatmulTemplateParams, validate_matmul_params

    with pytest.raises(ValueError, match=r"\(256, 256\) row only"):
        validate_matmul_params(MatmulTemplateParams(dtype_qkv=DTYPE_E4M3, cgrp_tile_mn=(128, 192), epi_mode=EPI_QUANT))
