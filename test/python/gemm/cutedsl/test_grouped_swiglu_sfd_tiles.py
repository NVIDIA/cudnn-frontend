# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Both GLU tile widths must write scale outputs for every live output block."""

import math

import pytest
import torch
from cuda.bindings import driver as cuda

pytestmark = pytest.mark.L0


@pytest.fixture
def api():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("SM100+ is required")
    from cudnn.gemm.cutedsl.grouped.swiglu import api as module

    return module


def _pack_scales(exponent):
    rows, blocks = exponent.shape[-2:]
    raw = (exponent + 127).to(torch.uint8).reshape(-1, rows // 128, 4, 32, blocks // 4, 4)
    return raw.permute(0, 1, 4, 3, 2, 5).contiguous().view(torch.float8_e8m0fnu)


def _operands(experts, n, canonical):
    generator = torch.Generator().manual_seed(147731 + experts + n)
    counts = [256 if i % 3 else 512 for i in range(experts)]
    if experts > 1:
        counts[1] = 0
    rows, k = sum(counts), 1024
    # Generate on CPU: the data does not depend on the device's SM count.
    a = (torch.randn((rows, k), generator=generator) * 0.5).to(torch.float8_e4m3fn)
    b = (torch.randn((experts, n, k), generator=generator) * 0.125).to(torch.float8_e4m3fn)
    aexp = torch.arange(rows * (k // 32)).reshape(rows, k // 32) % 3 - 1
    bexp = torch.arange(experts * n * (k // 32)).reshape(experts, n, k // 32) % 3 - 1
    sfa, sfb = _pack_scales(aexp), _pack_scales(bexp)
    adev = (a.float().reshape(rows, k // 32, 32) * torch.pow(2.0, aexp).unsqueeze(-1)).reshape(rows, k)
    bdev = (b.float().reshape(experts, n, k // 32, 32) * torch.pow(2.0, bexp).unsqueeze(-1)).reshape(experts, n, k)
    kwargs = dict(
        a_tensor=a.cuda(),
        b_tensor=b.cuda(),
        sfa_tensor=sfa.cuda(),
        sfb_tensor=sfb.cuda(),
        padded_offsets=torch.tensor(counts, dtype=torch.int32).cumsum(0).to(torch.int32).cuda(),
        alpha_tensor=torch.linspace(0.25, 1.0, experts, dtype=torch.float32).cuda(),
        norm_const_tensor=torch.tensor([0.75], dtype=torch.float32).cuda(),
        prob_tensor=torch.linspace(0.25, 0.875, rows, dtype=torch.float32).cuda(),
    )
    if not canonical:
        kwargs["a_tensor"] = kwargs["a_tensor"].unsqueeze(-1)
        kwargs["b_tensor"] = kwargs["b_tensor"].permute(1, 2, 0)
        for name in ("sfa_tensor", "sfb_tensor"):
            kwargs[name] = kwargs[name].permute(3, 4, 1, 5, 2, 0)
        kwargs["prob_tensor"] = kwargs["prob_tensor"].reshape(rows, 1, 1)
    return kwargs, counts, adev.cuda(), bdev.cuda()


def _call(api, inputs, tile, **options):
    return api.grouped_gemm_swiglu_wrapper_sm100(
        **inputs,
        sf_vec_size=32,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        mma_tiler_mn=(256, tile),
        cluster_shape_mn=(2, 1),
        m_aligned=256,
        **options,
    )


def _logical_codes(sf, rows, cols, canonical):
    if not canonical:
        sf = sf.permute(5, 2, 4, 0, 1, 3)
    raw = sf.view(torch.uint8).reshape(1, math.ceil(rows / 128), math.ceil(cols / 128), 32, 4, 4)
    logical = raw.permute(0, 1, 4, 3, 2, 5).contiguous().reshape(math.ceil(rows / 128) * 128, math.ceil(cols / 128) * 4)
    return logical[:rows, : math.ceil(cols / 32)]


def _dequantized(result, rows, n, canonical):
    d = result["d_tensor"].reshape(rows, n // 2).float()
    dc = result["d_col_tensor"].reshape(rows, n // 2).float()
    row = _logical_codes(result["sfd_row_tensor"], rows, n // 2, canonical)
    col = _logical_codes(result["sfd_col_tensor"], n // 2, rows, canonical)
    assert not bool((row == 255).any()), "live row SFD contains unwritten output poison"
    assert not bool((col == 255).any()), "live column SFD contains unwritten output poison"
    scale_row = torch.pow(2.0, row.int() - 127)
    scale_col = torch.pow(2.0, col.int() - 127)
    plain_row = (d.reshape(rows, n // 64, 32) * scale_row.unsqueeze(-1) / 0.75).reshape(rows, n // 2)
    plain_col = (dc.reshape(rows // 32, 32, n // 2) * scale_col.T.unsqueeze(1) / 0.75).reshape(rows, n // 2)
    assert torch.isfinite(plain_row).all() and torch.isfinite(plain_col).all()
    return row, col, plain_row, plain_col


def _poison_outputs(monkeypatch):
    guards = []
    original = torch.empty

    def allocate(*args, **kwargs):
        if kwargs.get("dtype") != torch.float8_e8m0fnu:
            return original(*args, **kwargs)
        shape = args[0]
        extent = math.prod(shape)
        storage = torch.full((extent + 2048,), 0xA5, dtype=torch.uint8, device=kwargs["device"])
        output = storage[1024 : 1024 + extent].view(torch.float8_e8m0fnu).reshape(shape)
        output.view(torch.uint8).fill_(255)
        guards.append((storage, extent))
        return output

    monkeypatch.setattr(torch, "empty", allocate)
    return guards


def _assert_guards(guards):
    for storage, extent in guards:
        assert bool((storage[:1024] == 0xA5).all())
        assert bool((storage[1024 + extent :] == 0xA5).all())


@pytest.mark.parametrize("experts,n", [(1, 256), (4, 512), (31, 1024), (33, 1024)])
@pytest.mark.parametrize("canonical", [True, False])
def test_narrow_sfd_matches_wide_and_independent_activation(api, monkeypatch, experts, n, canonical):
    inputs, counts, a, b = _operands(experts, n, canonical)
    guards = _poison_outputs(monkeypatch)
    wide, narrow = _call(api, inputs, 256), _call(api, inputs, 128)
    rows = sum(counts)
    wr, wc, wd, wdc = _dequantized(wide, rows, n, canonical)
    nr, nc, nd, ndc = _dequantized(narrow, rows, n, canonical)
    assert torch.equal(nr, wr) and torch.equal(nc, wc)
    torch.testing.assert_close(nd, wd, rtol=0, atol=0)
    torch.testing.assert_close(ndc, wdc, rtol=0, atol=0)
    torch.testing.assert_close(narrow["amax_tensor"], wide["amax_tensor"], rtol=0, atol=0)
    start = 0
    for expert, count in enumerate(counts):
        if count:
            c = torch.nn.functional.linear(a[start : start + count], b[expert]) * inputs["alpha_tensor"][expert]
            pair = c.reshape(count, n // 64, 2, 32)
            ref = (torch.nn.functional.silu(pair[:, :, 0]) * pair[:, :, 1]).reshape(count, n // 2)
            ref *= inputs["prob_tensor"].reshape(rows)[start : start + count, None]
            for actual in (nd, ndc):
                difference = torch.linalg.vector_norm(actual[start : start + count] - ref) / torch.linalg.vector_norm(ref)
                assert difference < 0.004
        start += count
    _assert_guards(guards)


@pytest.mark.parametrize("side_stream", [True, False])
def test_narrow_sfd_plan_reuse_preserves_previous_outputs(api, monkeypatch, side_stream):
    inputs, counts, _, _ = _operands(4, 1024, True)
    guards = _poison_outputs(monkeypatch)
    stream = torch.cuda.Stream() if side_stream else torch.cuda.current_stream()
    stream.wait_stream(torch.cuda.current_stream())
    launch = cuda.CUstream(stream.cuda_stream)
    first = _call(api, inputs, 128, current_stream=launch)
    torch.cuda.current_stream().wait_stream(stream)
    rows = sum(counts)
    snapshot = tuple(x.clone() for x in _dequantized(first, rows, 1024, True))
    plans = len(api._cache_of_GroupedGemmSwigluSm100Objects)
    inputs["alpha_tensor"].zero_()
    stream.wait_stream(torch.cuda.current_stream())
    second = _call(api, inputs, 128, current_stream=launch)
    torch.cuda.current_stream().wait_stream(stream)
    assert len(api._cache_of_GroupedGemmSwigluSm100Objects) == plans
    _, _, row, col = _dequantized(second, rows, 1024, True)
    assert torch.count_nonzero(row) == torch.count_nonzero(col) == 0
    expected_amax = torch.tensor([0.0 if n else -float("inf") for n in counts], device="cuda").reshape(-1, 1)
    torch.testing.assert_close(second["amax_tensor"], expected_amax, rtol=0, atol=0)
    for current, saved in zip(_dequantized(first, rows, 1024, True), snapshot):
        torch.testing.assert_close(current, saved, rtol=0, atol=0)
    _assert_guards(guards)


def test_narrow_sfd_graph_replay_changes_live_scales(api, monkeypatch):
    inputs, counts, _, _ = _operands(4, 512, True)
    guards = _poison_outputs(monkeypatch)
    _call(api, inputs, 128)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = _call(api, inputs, 128)
        graph.replay()
        first = tuple(x.clone() for x in _dequantized(captured, sum(counts), 512, True))
        inputs["alpha_tensor"].mul_(0.125)
        inputs["norm_const_tensor"].fill_(1.5)
        graph.replay()
        cold = _call(api, inputs, 128)
        for name in ("d_tensor", "d_col_tensor", "sfd_row_tensor", "sfd_col_tensor", "amax_tensor"):
            assert torch.equal(captured[name].contiguous().view(torch.uint8), cold[name].contiguous().view(torch.uint8)), name
        row, col = _logical_codes(captured["sfd_row_tensor"], sum(counts), 256, True), _logical_codes(captured["sfd_col_tensor"], 256, sum(counts), True)
        assert not torch.equal(row, first[0]) and not torch.equal(col, first[1])
        assert not bool((row == 255).any()) and not bool((col == 255).any())
        _assert_guards(guards)
    finally:
        graph.reset()
