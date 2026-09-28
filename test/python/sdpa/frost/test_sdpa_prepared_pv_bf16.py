# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Prepared hybrid MXFP8 Q/K + BF16 V keeps independent operand widths."""

import pytest
import torch

from frost_test_utils import requires_dsl

pytestmark = [requires_dsl]


def _case(d=128, dtype=torch.float8_e4m3fn, stats=True, amax=True, v=None, has_amax_o=True):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("PV-BF16 needs pre-Rubin SM100")
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100
    from test_sdpa_fwd_mxfp8_sm100 import _quantize

    gen = torch.Generator(device="cuda").manual_seed(577)
    b, hq, hk, s = 2, 4, 2, 128
    qf = torch.randn((b, hq, s, d), device="cuda", generator=gen) * 0.5
    kf = torch.randn((b, hk, s, d), device="cuda", generator=gen) * 0.5
    q, sfq, dq, _ = _quantize(qf, b, hq, s, d, dtype, columnwise=False)
    k, sfk, dk, _ = _quantize(kf, b, hk, s, d, dtype, columnwise=False)

    def bshd(t):
        return t.transpose(1, 2).contiguous().transpose(1, 2)

    bufs = dict(
        q_tensor=bshd(q),
        k_tensor=bshd(k),
        v_tensor=bshd(torch.randn((b, hk, s, 128), device="cuda", dtype=torch.bfloat16, generator=gen)) if v is None else v,
        o_tensor=torch.empty((b, s, hq, 128), device="cuda", dtype=torch.bfloat16).transpose(1, 2),
        sf_q=sfq,
        sf_k=sfk,
        lse_tensor=torch.empty((b, hq, s), device="cuda", dtype=torch.float32) if stats else None,
        amax_o=torch.empty(1, device="cuda", dtype=torch.float32) if amax else None,
    )
    api = SdpaFwdDslSm100(
        sample_q=bufs["q_tensor"],
        sample_k=bufs["k_tensor"],
        sample_v=bufs["v_tensor"],
        sample_o=bufs["o_tensor"],
        sample_lse=bufs["lse_tensor"],
        sample_amax_o=bufs["amax_o"],
        dtype_o=torch.bfloat16,
        is_causal=True,
        scale_softmax=d**-0.5,
        pv_bf16=True,
        has_amax_o=has_amax_o,
    )
    assert api.check_support()
    api.compile()
    ws = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    return api, bufs, ws, [dq, dk]


def _check(bufs, scales):
    from test_sdpa_fwd_mxfp8_sm100 import _ref

    ref = _ref(
        bufs["q_tensor"].float() * scales[0],
        bufs["k_tensor"].float() * scales[1],
        bufs["v_tensor"].float(),
        scale=bufs["q_tensor"].shape[-1] ** -0.5,
        is_causal=True,
        return_stats=True,
    )
    torch.testing.assert_close(bufs["o_tensor"].float(), ref.output, rtol=0.025, atol=0.02)
    if bufs["lse_tensor"] is not None:
        torch.testing.assert_close(bufs["lse_tensor"], ref.stats, rtol=0.002, atol=0.002)
    if bufs["amax_o"] is not None:
        torch.testing.assert_close(bufs["amax_o"], ref.output.abs().max().reshape(1), rtol=0.025, atol=0.02)


def _poison(bufs):
    for name in ("o_tensor", "lse_tensor", "amax_o"):
        if bufs[name] is not None:
            bufs[name].fill_(float("nan"))


@pytest.mark.L0
@pytest.mark.parametrize("d", [128, 192])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize("amax", [False, True])
def test_pv_bf16_prepared_rebind_and_replay(d, dtype, stats, amax, monkeypatch):
    from cuda.bindings import driver
    from test_sdpa_prepared_fp8 import _cuda_graph

    api, bufs, ws, scales = _case(d, dtype, stats, amax)
    assert api._prepared_mxfp8 and len(api._dense_spec.quant.sf_sizes) == 2
    assert api._dense_spec.quant.has_amax == amax
    assert api._dense_spec.expect["v"] == "bfloat16"
    api.execute(**bufs, workspace=ws)
    _check(bufs, scales)
    for name, buf in bufs.items():
        if buf is not None:
            bufs[name] = buf.clone()
    # The BF16 V pitches differ from the FP8 Q/K pitches and the plan's V.
    storage = torch.full((2, 128, 2, 144), 79, device="cuda", dtype=torch.bfloat16)
    rebound = storage[..., :128].transpose(1, 2)
    rebound.copy_(-bufs["v_tensor"])
    bufs["v_tensor"] = rebound
    ws = ws.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with monkeypatch.context() as guards:

        def forbidden(*args, **kwargs):
            pytest.fail("prepared PV-BF16 executed tensor construction or compilation")

        import cutlass.cute as cute

        guards.setattr(cute.runtime, "make_fake_tensor", forbidden)
        guards.setattr(cute.runtime, "make_fake_compact_tensor", forbidden)
        guards.setattr(api, "_dummy", forbidden)
        guards.setattr(api, "_can_prepare_fp8", forbidden)
        guards.setattr(api, "_can_prepare_mxfp8", forbidden)
        guards.setattr(torch.Tensor, "view", forbidden)
        guards.setattr(cute, "compile", forbidden)
        for name in ("empty", "empty_like", "zeros", "zeros_like", "ones", "full"):
            guards.setattr(torch, name, forbidden)
        torch.cuda.set_sync_debug_mode("error")
        try:
            api.execute(**bufs, workspace=ws, current_stream=driver.CUstream(stream.cuda_stream))
        finally:
            torch.cuda.set_sync_debug_mode("default")
    torch.cuda.current_stream().wait_stream(stream)
    _check(bufs, scales)
    with _cuda_graph() as captured:
        with torch.cuda.graph(captured):
            api.execute(**bufs, workspace=ws)
        for name, index in (("sf_q", 0), ("sf_k", 1)):
            bufs[name].view(torch.uint8).add_(1)
            scales[index] = scales[index] * 2
            rebound.neg_()
            _poison(bufs)
            captured.replay()
            _check(bufs, scales)
    assert torch.all(storage[..., 128:] == 79)
    with pytest.raises(ValueError, match="does not consume sf_v"):
        api.execute(**bufs, workspace=ws, sf_v=bufs["sf_k"])
    with pytest.raises(ValueError, match="bfloat16"):
        api.execute(**dict(bufs, v_tensor=bufs["v_tensor"].to(dtype)), workspace=ws)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("d", [128, 192])
def test_pv_bf16_v_batch_stride_above_int32(d):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("PV-BF16 needs pre-Rubin SM100")
    from test_sdpa_prepared_fp8 import _cuda_graph

    compact = 2 * 128 * 128
    try:
        v = torch.empty_strided((2, 2, 128, 128), (2**32 + compact, 128, 2 * 128, 1), device="cuda", dtype=torch.bfloat16)
    except torch.OutOfMemoryError:
        pytest.skip("physical BF16 V stride requires 8 GiB")
    v[0].fill_(0.25)
    v[1].fill_(0.5)
    decoy = torch.as_strided(v, (2, 128, 128), (128, 2 * 128, 1), storage_offset=compact)
    decoy.fill_(-0.5)
    api, bufs, ws, scales = _case(d=d, v=v)
    assert api._prepared_mxfp8
    api.execute(**bufs, workspace=ws)
    _check(bufs, scales)
    with _cuda_graph() as captured:
        with torch.cuda.graph(captured):
            api.execute(**bufs, workspace=ws)
        v[1].fill_(-0.5)
        decoy.fill_(0.5)
        _poison(bufs)
        captured.replay()
        _check(bufs, scales)


@pytest.mark.L0
@pytest.mark.parametrize("d", [128, 192])
@pytest.mark.parametrize("amax", [False, True])
def test_pv_bf16_staged_conversion_omits_dead_operands(d, amax, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("PV-BF16 needs pre-Rubin SM100")
    gen = torch.Generator(device="cuda").manual_seed(587)
    # Legal dense layout whose 258-byte head pitch cannot bind a TMA directly.
    storage = torch.randn((2, 128, 2, 129), device="cuda", dtype=torch.bfloat16, generator=gen)
    v = storage[..., :128].transpose(1, 2)
    api, bufs, ws, scales = _case(d=d, amax=amax, v=v)
    assert not api._prepared_mxfp8 and api._staged_spec is not None
    core = api._staged_spec.core
    original = core.fn
    seen = []

    def launch(*args, **kwargs):
        assert args[core.index["sf_v_ptr"]] is None, "hybrid pointer entry constructed unused SF_V"
        assert args[core.index["scale_o_ptr"]] is None
        assert core.quant.has_amax == amax
        if amax:
            assert args[core.index["amax_o_ptr"]] == bufs["amax_o"].data_ptr()
        seen.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(core, "fn", launch)
    api.execute(**bufs, workspace=ws)
    _check(bufs, scales)
    assert seen


@pytest.mark.L0
@pytest.mark.parametrize("d", [128, 192])
@pytest.mark.parametrize("native_layout", [False, True])
def test_pv_bf16_no_amax_flag_with_sample_descriptor(d, native_layout):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("PV-BF16 needs pre-Rubin SM100")
    v = None
    if not native_layout:
        storage = torch.full((2, 128, 2, 129), 0.5, device="cuda", dtype=torch.bfloat16)
        v = storage[..., :128].transpose(1, 2)
    api, bufs, ws, scales = _case(d=d, amax=True, v=v, has_amax_o=False)
    assert api._prepared_mxfp8 == native_layout
    assert not api.has_amax_o
    with pytest.raises(ValueError, match="has_amax_o=False"):
        api.execute(**bufs, workspace=ws)
    ignored_amax = bufs["amax_o"]
    ignored_amax.fill_(79)
    bufs["amax_o"] = None
    api.execute(**bufs, workspace=ws)
    _check(bufs, scales)
    assert ignored_amax.item() == 79
