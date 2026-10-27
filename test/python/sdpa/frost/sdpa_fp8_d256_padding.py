# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared D256 FP8 envelope cases, collected by the architecture CI entries."""

from dataclasses import replace

import pytest
import torch

import cudnn
import test_sdpa_prepared_fp8 as dense

_HEADS = [(16, 144), (224, 16), (160, 160), (224, 208), (256, 208), (224, 256), (256, 256)]
_INPUTS = [torch.float8_e4m3fn, torch.float8_e5m2]
_OUTPUTS = [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2]


def _record_route(monkeypatch):
    from cudnn.sdpa.fwd import api_dsl

    seen = []
    load = api_dsl._load_kernel_template

    def record(filename, params, tag):
        seen.append(filename)
        return load(filename, params, tag)

    monkeypatch.setattr(api_dsl, "_load_kernel_template", record)
    return seen


def _clone_with_slack(value, fill):
    span = 1 + sum((n - 1) * stride for n, stride in zip(value.shape, value.stride()))
    raw = torch.full((span,), fill, dtype=value.dtype, device=value.device)
    clone = raw.as_strided(value.shape, value.stride())
    clone.copy_(value)
    return clone


def _check(bufs, *, page=0, causal=False, bottom_right=False, window=None):
    from sdpa.fp8_ref import compute_ref
    from test_sdpa_fwd_paged_sm100 import _check_fp8_o, _gather_pages_fp8

    q, k, v = (bufs[n] for n in ("q", "k", "v"))
    b, h, sq, d = q.shape
    padding = None
    if page:
        lens = bufs["seq_kv"].view(b)
        for name in ("k", "v"):
            table = bufs[name + "_table"].view(b, -1)
            gathered = _gather_pages_fp8(bufs[name], table, lens.tolist(), True, page, table.shape[1] * page)
            if name == "k":
                k = gathered
            else:
                v = gathered
        padding = (bufs["seq_q"].view(b), lens)
    # D256 casts P without D128's extra 2**4 factor, even for a narrow Q/K
    # head paired with wide V. The FP64 oracle follows that native pipeline.
    ref, stats, _ = compute_ref(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_scale=d**-0.5,
        q_descale=bufs["descale_q"].item(),
        k_descale=bufs["descale_k"].item(),
        v_descale=bufs["descale_v"].item(),
        s_scale=16.0,
        s_descale=1.0 / 16.0,
        torch_itype=q.dtype,
        torch_otype=torch.float16,
        padding=padding,
        left_bound=window,
        right_bound=0 if causal else None,
        diag_align=cudnn.diagonal_alignment.BOTTOM_RIGHT if bottom_right else cudnn.diagonal_alignment.TOP_LEFT,
        sink_token=bufs.get("sinks"),
        rescale_threshold=4.0,
        dtype=torch.float64,
        quantize_o=False,
        sink_in_max=False,
    )
    ref = ref.transpose(1, 2)
    in_key = "e4m3" if q.dtype == torch.float8_e4m3fn else "e5m2"
    _check_fp8_o(bufs["o"], ref * bufs["scale_o"], bufs["o"].dtype, in_key)
    # Match the existing prepared-dense Stats bound; paged cases retain their
    # existing absolute bound.
    torch.testing.assert_close(bufs["lse"].reshape(b, h, sq), stats.reshape(b, h, sq), atol=0.01, rtol=0 if page else 0.01)
    assert abs(bufs["amax_o"].item() - ref.abs().max().item()) <= 0.03


def _check_dense(d, dv, dtype, output, monkeypatch, *, causal=False, bottom_right=False, window=None, softmax_precision=None):
    arch = "sm107" if torch.cuda.get_device_capability() == (10, 7) else "sm100"
    seen = _record_route(monkeypatch)
    idx = _HEADS.index((d, dv)) if (d, dv) in _HEADS else 3
    split = 4 if arch == "sm100" and idx in (3, 6) else 1
    pack = arch == "sm100" and idx in (2, 3, 5, 6)
    sinks = torch.linspace(-5, 8, 4, device="cuda").view(1, 4, 1, 1) if idx in (1, 2, 3, 6) or causal else None
    sq, skv = (17 if split > 1 else 129), 256
    kwargs = (
        dict(right_bound=0, diagonal_alignment=cudnn.diagonal_alignment.BOTTOM_RIGHT if bottom_right else cudnn.diagonal_alignment.TOP_LEFT) if causal else {}
    )
    if window is not None:
        kwargs["left_bound"] = window
    if softmax_precision is not None:
        kwargs["softmax_precision"] = softmax_precision
    g, vp, ws, bufs, tensors = dense._case(
        arch=arch,
        d=d,
        dv=dv,
        b=2,
        sq=sq,
        skv=skv,
        split_kv=split,
        dtype=dtype,
        output_dtype=output,
        sink=sinks,
        padded=True,
        output_padding=16,
        sdpa_kwargs=kwargs,
    )
    selected = g.plans[g._plan_index]
    g.create_execution_plan(selected.engine_id, replace(selected.knobs, pack_gqa=pack, sched_policy=idx % (2 if arch == "sm107" else 3)))
    g.select_plan(len(g.plans) - 1)
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(1, g.get_workspace_size()), dtype=torch.uint8, device="cuda")
    assert f"{arch}/prefill_d256_fp8.py" in seen and all(n.startswith(arch + "/") for n in seen), seen
    for name in ("q", "k", "v"):
        value = bufs[name]
        value._base[..., value.shape[-1] :].fill_(float("nan"))
        value.copy_((value.float() * 2.5).to(dtype))
    check = dict(causal=causal, bottom_right=bottom_right, window=window)
    g.execute(vp, ws)
    _check(bufs, **check)
    assert torch.all(bufs["o_storage"][..., dv:] == 12)
    with dense._cuda_graph() as capture:
        with torch.cuda.graph(capture):
            g.execute(vp, ws)
        fresh = {n: _clone_with_slack(v, float("nan")) for n, v in bufs.items() if n != "o_storage"}
        fresh["descale_q"].fill_(1.1)
        fresh["scale_o"].fill_(0.6)
        if sinks is not None:
            fresh["sinks"].fill_(6)
        g.execute({t: fresh[n] for n, t in tensors.items()}, torch.empty_like(ws))
        _check(fresh, **check)
        bufs["descale_v"].fill_(0.3)
        if sinks is not None:
            bufs["sinks"].fill_(-4)
        for name in ("o", "lse", "amax_o"):
            bufs[name].fill_(float("nan"))
        capture.replay()
        _check(bufs, **check)
        assert torch.all(bufs["o_storage"][..., dv:] == 12)


@pytest.mark.parametrize("d,dv", _HEADS)
@pytest.mark.parametrize("dtype", _INPUTS)
@pytest.mark.parametrize("output", _OUTPUTS)
def test_dense_rebind_replay(d, dv, dtype, output, monkeypatch):
    _check_dense(d, dv, dtype, output, monkeypatch)


@pytest.mark.parametrize("bottom_right,window", [(False, None), (True, None), (False, 64), (True, 64)])
@pytest.mark.parametrize("dtype", _INPUTS)
def test_dense_masks(bottom_right, window, dtype, monkeypatch):
    _check_dense(224, 208, dtype, torch.bfloat16, monkeypatch, causal=True, bottom_right=bottom_right, window=window)


@pytest.mark.parametrize("d,dv", [(16, 144), (224, 208)])
@pytest.mark.parametrize("dtype", _INPUTS)
@pytest.mark.parametrize("output", [torch.bfloat16, torch.float8_e4m3fn])
def test_dense_half_softmax(d, dv, dtype, output, monkeypatch):
    """Native HALF exponentiation retains the padded binding/replay contract."""
    _check_dense(d, dv, dtype, output, monkeypatch, softmax_precision=cudnn.data_type.HALF)


@pytest.mark.parametrize(
    "d,dv,page,k_hnd,v_hnd,split",
    [
        (16, 144, 16, False, True, 1),
        (224, 16, 128, True, False, 1),
        (160, 160, 16, True, False, 4),
        (224, 208, 128, False, True, 4),
        (256, 208, 16, False, False, 1),
        (224, 256, 128, True, True, 4),
        (256, 256, 16, True, True, 4),
        (224, 208, 256, False, False, 1),
    ],
)
@pytest.mark.parametrize("dtype", _INPUTS)
@pytest.mark.parametrize("output", _OUTPUTS)
def test_paged_rebind_replay(d, dv, page, k_hnd, v_hnd, split, dtype, output, monkeypatch):
    import test_sdpa_fwd_paged_sm100 as paged

    seen = _record_route(monkeypatch)
    pages, sq = 512 // page, 17
    sink = torch.linspace(-5, 8, 4, device="cuda").view(1, 4, 1, 1)
    g, vp, ws, bufs, tensors = paged._run_graph_fp8(
        2,
        4,
        2,
        d,
        page,
        pages,
        [512, 512],
        k_hnd,
        in_key="e4m3" if dtype == torch.float8_e4m3fn else "e5m2",
        out_dt=output,
        s_q=sq,
        explicit_split=split,
        explicit_pack=split > 1,
        explicit_cga=1,
        sinks=sink,
        return_case=True,
        v_table_layout="strided",
        v_hnd=v_hnd,
        d_v=dv,
        head_padding=True,
    )
    assert "sm100/prefill_d256_fp8.py" in seen and all(n.startswith("sm100/") for n in seen), seen
    storage = bufs["o"]._base
    assert torch.all(storage[..., dv:] == 12)
    with dense._cuda_graph() as capture:
        with torch.cuda.graph(capture):
            g.execute(vp, ws)
        # Preserve every declared scale UID, including scale_s/descale_s that
        # this kernel ignores but a backend-lowerable graph still declares.
        rebound = {t: _clone_with_slack(v, 0 if v.dtype == torch.int32 else float("nan")) for t, v in vp.items()}
        fresh = {n: rebound[t] for n, t in tensors.items()}
        fresh["v_table"].copy_(fresh["k_table"].flip(2))
        fresh["descale_v"].mul_(0.5)
        fresh["scale_o"].fill_(1.9)
        g.execute(rebound, torch.empty_like(ws))
        _check(fresh, page=page)
        bufs["seq_kv"].copy_(torch.tensor([129, 0], device="cuda", dtype=torch.int32).view(2, 1, 1, 1))
        bufs["seq_q"].copy_(torch.tensor([7, 17], device="cuda", dtype=torch.int32).view(2, 1, 1, 1))
        bufs["descale_q"].mul_(0.7)
        bufs["sinks"].fill_(6)
        bufs["v_table"].copy_(bufs["k_table"].flip(2))
        for name in ("o", "lse", "amax_o"):
            bufs[name].fill_(float("nan"))
        capture.replay()
        _check(bufs, page=page)
        assert torch.all(storage[..., dv:] == 12)
