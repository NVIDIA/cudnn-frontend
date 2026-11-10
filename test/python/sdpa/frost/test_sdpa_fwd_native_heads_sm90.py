# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native Hopper D128/D256 contraction/storage, masks and prepared replay."""

import math

import pytest
import torch

import test_sdpa_fwd_dsl_sm100 as shared
import test_sdpa_fwd_split_sm90 as split_checks
import test_sdpa_fwd_split_sm90_determinism as determinism
import test_sdpa_fwd_thd_sm90_offsets as origins
from frost_test_utils import _SM, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM != 90, reason="requires native SM90")]


@pytest.fixture(autouse=True)
def native(monkeypatch):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    monkeypatch.setattr(shared, "_ARCH", "sm90")
    original = SdpaFwdDslSm90.compile

    def compile_checked(api):
        original(api)
        expected = next(width for width in (128, 256, 512) if max(api.head_dim_qk, api.head_dim_v) <= width)
        assert api.params.head_tile == expected
        assert api._k_mod.SM90FusedMultiHeadAttentionForward.HEAD_TILE == expected

    monkeypatch.setattr(SdpaFwdDslSm90, "compile", compile_checked)
    with torch.cuda.stream(torch.cuda.Stream()):
        yield


@pytest.mark.parametrize("dims", [(128, 128), (256, 256), (192, 128), (120, 72), (248, 136)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("mask", ["full", "causal", "swa", "bottom-right"])
def test_native_graph(dims, dtype, mask):
    d, dv = dims
    b, h, hk, sq, sk = 2, 4, 2, 65, 129
    rng = torch.Generator().manual_seed(190128)
    source = [torch.randn(shape, generator=rng).to(dtype) * 0.25 for shape in ((b, sq, h, d), (b, sk, hk, d), (b, sk, hk, dv))]
    q, k, v = [value.cuda().transpose(1, 2) for value in source]
    kw = {}
    if mask in ("causal", "swa"):
        kw["use_causal_mask"] = True
    elif mask == "bottom-right":
        kw["use_causal_mask_bottom_right"] = True
    if mask == "swa":
        kw["sliding_window_length"] = 32
    out, stats = shared._run_dsl_graph(q, k, v, scale=1 / math.sqrt(d), dtype=dtype, sdpa_kwargs=kw, pack_gqa=False, return_stats=True)
    qq, kk, vv = [value.transpose(1, 2).double() for value in source]
    kk, vv = (value.repeat_interleave(h // hk, 1) for value in (kk, vv))
    scores = qq @ kk.transpose(-1, -2) / math.sqrt(d)
    qi, ki = torch.arange(sq)[:, None], torch.arange(sk)[None, :]
    diag = qi + (sk - sq if mask == "bottom-right" else 0)
    if mask != "full":
        scores.masked_fill_(ki > diag, -torch.inf)
    if mask == "swa":
        scores.masked_fill_(ki < diag - 31, -torch.inf)
    lse = scores.logsumexp(-1)
    ref = torch.exp(scores - lse[..., None]) @ vv
    assert torch.isfinite(out).all() and torch.isfinite(stats).all()
    torch.testing.assert_close(out.cpu().double(), ref, atol=3e-3, rtol=1e-2)
    torch.testing.assert_close(stats.squeeze(-1).cpu().double(), lse, atol=2e-3, rtol=1e-3)


@pytest.mark.parametrize("width", [128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("layout,mask", [("dense", "padded"), ("thd", "causal_br"), ("thd", "swa_br")])
def test_layouts_and_masks(width, dtype, layout, mask):
    sink = torch.linspace(-2, 2, 8, device="cuda").reshape(1, 8, 1, 1)
    (shared._combo_dense if layout == "dense" else shared._combo_thd)(width, dtype, 8, 2, 1 / math.sqrt(width), sink, mask)


@pytest.mark.parametrize("dims", [(128, 128), (256, 256), (192, 128)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("pack", [False, True])
def test_split_replay(dims, dtype, pack, cudnn_handle):
    split_checks.test_dense_split_partial_initialization(dims, dtype, True, pack, True, cudnn_handle)


@pytest.mark.parametrize("width", [128, 256])
@pytest.mark.parametrize("group", [12, 128])
def test_partial_pack_replay(width, group, cudnn_handle):
    split_checks.test_dense_split_partial_initialization((width, width), torch.bfloat16, True, True, True, cudnn_handle, _heads=2 * group)


@pytest.mark.parametrize("width", [128, 256])
@pytest.mark.parametrize("sched,mask,scale", [(0, "none", 0.125), (1, "br_window", -0.125), (2, "br_widened", 0.0)])
def test_multiple_stages(width, sched, mask, scale):
    split_checks.test_split_multiple_kv_stages(torch.bfloat16, sched, True, mask, scale, 3, _d=width)


@pytest.mark.parametrize("width", [128, 256])
@pytest.mark.parametrize("stats", ["token_major", "head_major"])
def test_thd_physical_origins(width, stats, monkeypatch):
    monkeypatch.setattr(origins, "_D", width)
    origins.test_sm90_thd_stats_read_their_own_offsets(stats, "tokens")


@pytest.mark.parametrize("width", [128, 256])
def test_thd_origin_replay(width, monkeypatch):
    monkeypatch.setattr(origins, "_D", width)
    origins.test_sm90_thd_cuda_graph_replay_reads_new_offsets()


@pytest.mark.parametrize("width", [128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("splits,pack", [(1, False), (3, True)])
def test_batch_replay_and_golden(width, dtype, splits, pack):
    determinism.test_split_forward_determinism(splits, pack, dtype, _d=width)
