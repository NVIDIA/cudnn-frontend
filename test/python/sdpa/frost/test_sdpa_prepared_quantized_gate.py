# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Quantized gate launches rebind BF16 gate storage without changing ungated Amax."""

import pytest
import torch

from frost_test_utils import requires_dsl

pytestmark = [requires_dsl]


def _helper(mxfp8):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("quantized gate needs SM107")
    if mxfp8:
        import test_sdpa_prepared_mxfp8 as helper
    else:
        import test_sdpa_prepared_fp8 as helper
    return helper


def _poison(bufs):
    for name in ("o", "lse", "amax_o"):
        if name in bufs:
            bufs[name].fill_(float("nan"))


@pytest.mark.L0
@pytest.mark.parametrize("mxfp8", [False, True])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize("amax", [False, True])
def test_quantized_gate_prepared_rebind_and_replay(mxfp8, output_dtype, stats, amax, monkeypatch):
    helper = _helper(mxfp8)
    gen = torch.Generator(device="cuda").manual_seed(569)
    gate = torch.randn((2, 128, 4, 256), device="cuda", dtype=torch.bfloat16, generator=gen).transpose(1, 2)
    g, vp, ws, bufs, tensors = helper._case(d=256, arch="sm107", output_dtype=output_dtype, stats=stats, amax=amax, gate=gate)
    prepared = g._compiled_plans[g._plan_index]._prepared
    assert prepared is not None and prepared.spec.gate_expect == "bfloat16"
    g.execute(vp, ws)
    helper._check(bufs, thd=False)
    for name, tensor in tensors.items():
        if name in bufs:
            bufs[name] = bufs[name].clone()
            vp[tensor] = bufs[name]
    # Rebind G to a different physical layout, not just a new address.
    storage = torch.full((2, 128, 4, 272), 77, device="cuda", dtype=torch.bfloat16)
    rebound = storage[..., :256].transpose(1, 2)
    rebound.copy_(gate * 2)
    vp[tensors["gate"]] = bufs["gate"] = rebound
    if mxfp8:
        helper._change_scales(bufs)
    else:
        bufs["descale_v"].fill_(0.5)
        bufs["scale_o"].fill_(1.7)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), monkeypatch.context() as guards:

        def forbidden(*args, **kwargs):
            pytest.fail("prepared gate execution allocated a tensor")

        for name in ("empty", "empty_like", "zeros", "zeros_like", "ones", "full"):
            guards.setattr(torch, name, forbidden)
        torch.cuda.set_sync_debug_mode("error")
        try:
            g.execute(vp, ws)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    torch.cuda.current_stream().wait_stream(stream)
    helper._check(bufs, thd=False)
    with helper._cuda_graph() as captured:
        with torch.cuda.graph(captured):
            g.execute(vp, ws)
        # The gated output moves from exactly zero to ungated O. Stats and
        # requested Amax still describe SDPA before sigmoid multiplication.
        for value in (-1e4, 1e4):
            rebound.fill_(value)
            _poison(bufs)
            captured.replay()
            helper._check(bufs, thd=False)
    assert torch.all(storage[..., 256:] == 77), "gate input padding was modified"


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("mxfp8", [False, True])
def test_quantized_gate_batch_stride_above_int32(mxfp8):
    helper = _helper(mxfp8)
    compact = 4 * 128 * 256
    try:
        gate = torch.empty_strided((2, 4, 128, 256), (2**32 + compact, 256, 4 * 256, 1), device="cuda", dtype=torch.bfloat16)
    except torch.OutOfMemoryError:
        pytest.skip("physical gate-stride probe requires an 8-GiB allocation")
    gate[0].fill_(-1e4)
    gate[1].fill_(1e4)
    # A deliberately narrowed batch stride lands on this valid, wrong island.
    # This makes the negative control numerical rather than an invalid access.
    decoy = torch.as_strided(gate, (4, 128, 256), (256, 4 * 256, 1), storage_offset=compact)
    decoy.fill_(-1e4)
    g, vp, ws, bufs, _ = helper._case(d=256, arch="sm107", gate=gate)
    assert g._compiled_plans[g._plan_index]._prepared is not None
    g.execute(vp, ws)
    helper._check(bufs, thd=False)
    with helper._cuda_graph() as captured:
        with torch.cuda.graph(captured):
            g.execute(vp, ws)
        gate[0].fill_(1e4)
        gate[1].fill_(-1e4)
        decoy.fill_(1e4)
        _poison(bufs)
        captured.replay()
        helper._check(bufs, thd=False)


@pytest.mark.L0
def test_quantized_gate_bounded_batch_override(monkeypatch):
    helper = _helper(False)
    gate = torch.ones((2, 128, 4, 256), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    g, vp, ws, bufs, tensors = helper._case(d=256, arch="sm107", override=True, gate=gate)
    prepared = g._compiled_plans[g._plan_index]._prepared
    assert prepared is not None
    owner = prepared.spec.owner
    uids, shapes, strides = [], [], []
    for name in ("q", "k", "v", "o", "lse", "gate"):
        tensor = tensors[name]
        bufs[name] = bufs[name][:1].clone()
        vp[tensor] = bufs[name]
        shape = list(tensor.get_dim())
        shape[0] = 1
        uids.append(tensor.get_uid())
        shapes.append(shape)
        strides.append(list(tensor.get_stride()))
    import cutlass.cute as cute

    monkeypatch.setattr(cute, "compile", lambda *args, **kwargs: pytest.fail("gate override compiled a new artifact"))
    g.execute(vp, ws, override_uids=uids, override_shapes=shapes, override_strides=strides)
    helper._check(bufs, thd=False, b=1)
    assert prepared.spec.owner is owner
