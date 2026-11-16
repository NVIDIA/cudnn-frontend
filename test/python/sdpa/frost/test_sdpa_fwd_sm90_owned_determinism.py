# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the existing Hopper byte/reference guards on the fixed local bodies."""

import pytest
import torch
from frost_test_utils import _SM, requires_dsl
from cudnn.sdpa.fwd.kernels.sm90.d128_f16_multi_q import contract as d128
from cudnn.sdpa.fwd.kernels.sm90.f16_three_wg import contract as wide
import test_sdpa_fwd_d128_sm90_determinism as dedicated
import test_sdpa_fwd_split_sm90_determinism as split

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM != 90, reason="requires native Hopper")]


def _fix_body(monkeypatch, owner):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    original_init = SdpaFwdDslSm90.__init__
    original_compile = SdpaFwdDslSm90.compile

    def init(api, *args, **kwargs):
        original_init(api, *args, **kwargs)
        api._selected_kernel = owner.Geometry()

    def compile(api):
        original_compile(api)
        assert api._k_mod.__file__.endswith(owner.KERNEL_FILE)
        assert type(api._k_mod.PARAMS).__module__ == owner.__package__ + ".config"

    monkeypatch.setattr(SdpaFwdDslSm90, "__init__", init)
    monkeypatch.setattr(SdpaFwdDslSm90, "compile", compile)


_CELLS = [
    pytest.param(heads, mask, batch, dtype, id=f"fwd-{mask}-h{heads[0]}x{heads[1]}-q{heads[2]}-kv{heads[3]}-b{batch}-{str(dtype).split('.')[-1]}")
    for cell in dedicated._CELLS
    for heads, mask, batch, _ in [cell.values]
    for dtype in (torch.float16, torch.bfloat16)
]


@pytest.mark.parametrize("heads,mask,batch,dtype", _CELLS)
def test_owned_d128_replay_and_batch_invariance(monkeypatch, heads, mask, batch, dtype, request):
    _fix_body(monkeypatch, d128)
    dedicated.test_d128_forward_replay_and_batch_invariance(heads, mask, batch, dtype, request)


@pytest.mark.parametrize("heads,mask,batch,dtype", _CELLS)
def test_owned_d128_runtime_golden(monkeypatch, heads, mask, batch, dtype):
    _fix_body(monkeypatch, d128)
    dedicated.test_d128_forward_golden(heads, mask, batch, dtype)


@pytest.mark.parametrize("splits,pack_gqa", [(3, False), (17, True)], ids=["split3-unpacked", "split17-packed"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_owned_three_wg_split_replay_batch_and_golden(monkeypatch, splits, pack_gqa, dtype):
    _fix_body(monkeypatch, wide)
    # The original checks retain FP64 reference, byte replay, reordered/changed
    # neighbours and the runtime-golden gap. No baseline is created here.
    split.test_split_forward_determinism(splits, pack_gqa, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("claim", [False, True], ids=["kv-ring", "claim-ring"])
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
def test_owned_d128_refilled_stages(monkeypatch, dtype, claim):
    from cudnn.sdpa.fwd import api_dsl
    from warm_stress import assert_warm_stress

    _fix_body(monkeypatch, d128)
    if claim:
        monkeypatch.setattr(api_dsl, "D128_CLAIM_MIN_ROUNDS", 0)
    mask = "causal_br" if claim else "nomask"
    plan, case = dedicated._warm_case(
        f"SM90 owned D128 {dtype} claim={claim}",
        (8, 2, 512, 2048),
        mask,
        16,
        dtype,
        lambda p: p.items * 16 - 2 * min(p.items, p.ctas),
    )
    plan.launch()
    qkv = (plan.q_store[0], plan.k_store[0], plan.v_store[0])
    o, stats = dedicated._reference(qkv, mask)
    torch.testing.assert_close(plan.o[0].double().cpu(), o, atol=3e-3, rtol=1e-2)
    torch.testing.assert_close(plan.lse[0].double().cpu(), stats, atol=2e-3, rtol=1e-3)
    if claim:
        assert plan.api.params.causal_claim == 1
    assert_warm_stress(case, warm=3000, poisoned=3000)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
def test_owned_three_wg_refilled_stages(monkeypatch, dtype):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90
    from test_sdpa_fwd_dsl_sm100 import _bhsd
    from warm_stress import StressCase, assert_warm_stress

    _fix_body(monkeypatch, wide)
    b, h, hk, sq, sk, d = 8, 16, 2, 129, 513, 256
    torch.manual_seed(20540)
    q = _bhsd(b, h, sq, d, dtype)
    k, v = (_bhsd(b, hk, sk, d, dtype) for _ in range(2))
    o = _bhsd(b, h, sq, d, dtype)
    stats = torch.empty(b, h, sq, device="cuda", dtype=torch.float32)
    api = SdpaFwdDslSm90(q, k, v, o, sample_lse=stats, tile_m=64, tile_n=64)
    api.compile()
    launch = lambda: api.execute(q, k, v, o, lse_tensor=stats)
    launch()
    scores = (q.double() @ k.double().repeat_interleave(h // hk, 1).transpose(-1, -2)) * d**-0.5
    ref = scores.softmax(-1) @ v.double().repeat_interleave(h // hk, 1)
    torch.testing.assert_close(o.double(), ref, atol=5e-2, rtol=3e-2)
    torch.testing.assert_close(stats.double(), scores.logsumexp(-1), atol=2e-2, rtol=2e-2)
    physical = {"o": o.transpose(1, 2), "lse": stats}
    assert all(t.is_contiguous() for t in physical.values())
    assert_warm_stress(StressCase(f"SM90 owned three-WG {dtype}", launch, physical, thread_blocks=384, refills_per_launch=8), warm=3000, poisoned=3000)


def test_owned_d128_claim_release_rendering(monkeypatch, tmp_path):
    import test_sdpa_fwd_d128_sm90 as dedicated_main

    source = dedicated_main._SASS_PROBE
    original = '"sm90/d128_f16_multi_q/kernel.py"'
    assert original in source
    monkeypatch.setattr(dedicated_main, "_SASS_PROBE", source.replace(original, f'"{d128.KERNEL_FILE}"'))
    # Retain the original PTX dependency and SASS register/serialization guards,
    # now against the owned body with its implementation-specific symbol prefix.
    dedicated_main.test_d128_register_split_no_spills_and_no_wgmma_serialization(tmp_path, "pc1-causal")
