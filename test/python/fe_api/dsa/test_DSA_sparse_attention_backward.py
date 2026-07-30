# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SparseAttentionBackward.

The forward pass is a PyTorch reference (see dsa_reference.ref_sparse_attention_forward);
the production forward is FlashMLA (C++, out of scope). Gradients are generated
via autograd on the reference forward.
"""

import math

import pytest
import torch

from test_utils import torch_fork_set_rng

from fe_api.dsa.dsa_utils import (
    _require_sm90,
    dsa_init,
    with_dsa_sparse_attention_backward_params,
)
from fe_api.dsa.dsa_reference import (
    ref_sparse_attention_forward,
    check_ref_dsa_sparse_attention_backward,
)


def _allocate(cfg, has_topk_length: bool):
    total_s_q = cfg["s_q"]
    total_s_kv = cfg["s_kv"]
    h = cfg["h_q"] if cfg.get("h_q") else 64
    d = cfg["head_dim"]
    topk = cfg["topk"]
    device = "cuda"

    q = torch.randn(total_s_q, h, d, dtype=torch.bfloat16, device=device)
    kv = torch.randn(total_s_kv, d, dtype=torch.bfloat16, device=device)
    attn_sink = torch.randn(h, dtype=torch.float32, device=device)

    topk_k = min(topk, total_s_kv)
    topk_idxs = torch.stack([torch.randperm(total_s_kv, device=device)[:topk_k] for _ in range(total_s_q)]).to(torch.int32)
    if topk_k < topk:
        pad = torch.full((total_s_q, topk - topk_k), -1, dtype=torch.int32, device=device)
        topk_idxs = torch.cat([topk_idxs, pad], dim=-1)

    topk_length = None
    if has_topk_length:
        topk_length = torch.randint(1, topk_k + 1, (total_s_q,), dtype=torch.int32, device=device)

    return q, kv, attn_sink, topk_idxs, topk_length


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_dsa_sparse_attention_backward_params
def test_DSA_sparse_attention_backward_wrapper(
    dtype,
    acc_dtype,
    head_dim,
    head_dim_v,
    num_heads,
    topk,
    has_topk_length,
    request,
):
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    cfg = dsa_init(
        request=request,
        dtype=dtype,
        acc_dtype=acc_dtype,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        num_heads=num_heads,
        topk=topk,
        has_topk_length=has_topk_length,
        min_compute_capability=90,
        s_q_default=1024,
        s_kv_default=4096,
    )
    cfg["h_q"] = num_heads

    q, kv, attn_sink, topk_idxs, topk_length = _allocate(cfg, has_topk_length)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    # Run reference forward to get out + FlashMLA-style KV-only lse for backward.
    out, lse = ref_sparse_attention_forward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
    )
    dout = torch.randn_like(out)

    try:
        result = DSA.sparse_attention_backward_wrapper(
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            softmax_scale=softmax_scale,
            topk_length=topk_length,
            stream=stream,
        )
    except (ValueError, NotImplementedError, RuntimeError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    dq, dkv, d_sink = result["dq"], result["dkv"], result["d_sink"]

    if not cfg["skip_ref"]:
        # The DKV accumulation is known to have precision limitations
        # vs. the FP32 autograd reference; use generous tolerances.
        check_ref_dsa_sparse_attention_backward(
            q,
            kv,
            attn_sink,
            topk_idxs,
            out,
            dout,
            lse,
            dq,
            dkv,
            d_sink,
            softmax_scale=softmax_scale,
            topk_length=topk_length,
            atol=5e-2,
            rtol=5e-2,
        )


@pytest.mark.L0
@torch_fork_set_rng(seed=433)
@pytest.mark.parametrize(
    "head_dim,num_heads,topk_length_values",
    [
        pytest.param(512, 64, (-3, 0, 1, 63, 64, 65, 128), id="d512-mixed"),
        pytest.param(576, 32, None, id="d576-all-empty"),
    ],
)
def test_DSA_sparse_attention_backward_sm100_zero_topk_length(head_dim, num_heads, topk_length_values):
    """SM100 must treat a nonpositive top-k length as an empty attention row."""
    if not torch.cuda.is_available():
        pytest.skip("SM100 GPU required")
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        pytest.skip("zero top-k length regression test targets the SM100 kernel")

    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    s_q = len(topk_length_values) if topk_length_values is not None else 2
    s_kv, topk = 256, 128
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device) / 10
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device) / 10
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    base_topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)

    # The D512 mixed case covers defensive negative handling and both sides of
    # the 64-row tile boundary. The all-empty cases make every expected
    # gradient exactly zero; D576/H32 additionally covers the feature tail and
    # a partial 64-head CTA.
    topk_length_cases = [torch.zeros(s_q, dtype=torch.int32, device=device)]
    if topk_length_values is not None:
        topk_length_cases.insert(0, torch.tensor(topk_length_values, dtype=torch.int32, device=device))
    positions = torch.arange(topk, device=device).unsqueeze(0)

    for topk_length in topk_length_cases:
        topk_idxs = base_topk_idxs.clone()
        topk_idxs[positions >= topk_length.unsqueeze(1)] = -1
        empty_rows = topk_length <= 0
        # Invalid values in ignored slots exercise that the empty fast path
        # does not consult topk_idxs before exiting.
        topk_idxs[empty_rows] = torch.iinfo(torch.int32).max

        out, lse = ref_sparse_attention_forward(
            q,
            kv,
            attn_sink,
            topk_idxs,
            topk_length=topk_length,
            softmax_scale=softmax_scale,
        )
        assert torch.equal(out[empty_rows], torch.zeros_like(out[empty_rows]))
        assert torch.isneginf(lse[empty_rows]).all()
        dout = torch.randn_like(out)

        # In particular, the empty-row fast path must overwrite caller-owned
        # storage; torch.empty_like() could otherwise hide a missing dQ store.
        dq_buffer = torch.full_like(q, float("nan"))
        dkv_buffer = torch.full_like(kv, float("nan"))
        result = DSA.sparse_attention_backward_wrapper(
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            softmax_scale=softmax_scale,
            topk_length=topk_length,
            dq=dq_buffer,
            dkv=dkv_buffer,
        )
        # Reaching the synchronization is also the regression check for the
        # empty-CTA barrier deadlock.
        torch.cuda.synchronize()

        dq, dkv, d_sink = result["dq"], result["dkv"], result["d_sink"]
        assert dq.data_ptr() == dq_buffer.data_ptr()
        assert dkv.data_ptr() == dkv_buffer.data_ptr()
        assert torch.isfinite(dq).all()
        assert torch.isfinite(dkv).all()
        assert torch.isfinite(d_sink).all()
        assert torch.equal(dq[empty_rows], torch.zeros_like(dq[empty_rows]))

        check_ref_dsa_sparse_attention_backward(
            q,
            kv,
            attn_sink,
            topk_idxs,
            out,
            dout,
            lse,
            dq,
            dkv,
            d_sink,
            softmax_scale=softmax_scale,
            topk_length=topk_length,
            atol=5e-2,
            rtol=5e-2,
        )

        if torch.all(empty_rows):
            assert torch.equal(dkv, torch.zeros_like(dkv))
            assert torch.equal(d_sink, torch.zeros_like(d_sink))


@pytest.mark.L0
@torch_fork_set_rng(seed=385)
def test_DSA_sparse_attention_backward_qh32_uses_per_query_topk_without_padding(monkeypatch):
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm90
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm90()
    device = torch.device("cuda")
    s_q, s_kv, num_heads = 2, 128, 32
    head_dim, head_dim_v, topk = 576, 512, 64
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device)
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    # The old packed-M mapping used row 0 for both queries in one CTA. Make
    # the two rows disjoint so sharing a top-k row is unambiguously incorrect.
    topk_idxs = torch.stack(
        (
            torch.arange(0, topk, dtype=torch.int32, device=device),
            torch.arange(topk, 2 * topk, dtype=torch.int32, device=device),
        )
    )

    out, lse = ref_sparse_attention_forward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        softmax_scale=softmax_scale,
    )
    # Match the head-major backing used by the original #385 reproduction.
    out = out.transpose(0, 1).contiguous().transpose(0, 1)
    dout = torch.randn(num_heads, s_q, head_dim_v, dtype=torch.bfloat16, device=device).transpose(0, 1)
    assert out.stride() == dout.stride() == (head_dim_v, s_q * head_dim_v, 1)

    def fail_on_pad(*args, **kwargs):
        pytest.fail("qh32 must be handled by the SM90 main kernel without torch padding")

    monkeypatch.setattr(torch.nn.functional, "pad", fail_on_pad)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    result = DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        softmax_scale=softmax_scale,
        stream=stream,
    )
    torch.cuda.synchronize()

    # qhpkv controls the query-head tiling and must remain part of the main
    # compile key, independently of the runtime tensor shape.
    assert any(key[1:4] == (head_dim, head_dim_v, num_heads) for key in _interface_sm90.flash_attn_bwd_sm90.compile_cache)

    dq, dkv, d_sink = result["dq"], result["dkv"], result["d_sink"]
    assert torch.isfinite(dq).all()
    assert torch.isfinite(dkv).all()
    check_ref_dsa_sparse_attention_backward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        out,
        dout,
        lse,
        dq,
        dkv,
        d_sink,
        softmax_scale=softmax_scale,
        atol=5e-2,
        rtol=5e-2,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_DSA_sparse_attention_backward_staged_store():
    """Regression test for the staged P/dS store paths of the SM100 kernel.

    The compute->MMA smem handoff for P and dS must stay correct when
    compute_mma_P_stage / compute_mma_dS_stage are raised above 1: the
    store-side layouts and the per-iteration store slot have to follow the
    producer pipeline state. This test deepens both pipelines to 2 and
    requires dq to match the stage-1 baseline bitwise (dq is written by TMA
    with a single writer per element, hence deterministic; dkv/d_sink are
    fp32 atomic reductions, so they are only gated on NaN and relative
    parity).
    """
    try:
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import (
            api as _dsa_bwd_api,
            _interface_sm100 as _dsa_bwd_iface,
            dsa_bwd_sm100 as _dsa_bwd_kmod,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        pytest.skip("staged-store regression test targets the SM100 kernel")

    s_q = s_kv = 512
    h, d, topk = 64, 512, 256  # topk/64 = 4 tiles -> the pipelines really cycle
    device = "cuda"
    q = torch.randn(s_q, h, d, dtype=torch.bfloat16, device=device) / 10
    kv = torch.randn(s_kv, d, dtype=torch.bfloat16, device=device) / 10
    attn_sink = torch.randn(h, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)
    topk_length = torch.full((s_q,), topk, dtype=torch.int32, device=device)
    softmax_scale = 1.0 / math.sqrt(d)

    out, lse = ref_sparse_attention_forward(q, kv, attn_sink, topk_idxs, topk_length=topk_length, softmax_scale=softmax_scale)
    dout = torch.randn_like(out)

    orig_setup = _dsa_bwd_kmod.FlashAttentionDSABackwardSm100._setup_attributes

    def run(stage_overrides):
        def patched(self):
            orig_setup(self)
            for name, value in stage_overrides.items():
                setattr(self, name, value)

        _dsa_bwd_kmod.FlashAttentionDSABackwardSm100._setup_attributes = patched
        _dsa_bwd_iface.flash_attn_bwd_sm100.compile_cache.clear()
        _dsa_bwd_api._cache_of_SparseAttentionBackwardObjects.clear()
        try:
            result = DSA.sparse_attention_backward_wrapper(
                q,
                kv,
                out,
                dout,
                lse,
                attn_sink,
                topk_idxs,
                softmax_scale=softmax_scale,
                topk_length=topk_length,
            )
            torch.cuda.synchronize()
            return result["dq"], result["dkv"], result["d_sink"]
        finally:
            _dsa_bwd_kmod.FlashAttentionDSABackwardSm100._setup_attributes = orig_setup
            _dsa_bwd_iface.flash_attn_bwd_sm100.compile_cache.clear()
            _dsa_bwd_api._cache_of_SparseAttentionBackwardObjects.clear()

    dq_ref, dkv_ref, d_sink_ref = run({})
    assert not torch.isnan(dq_ref).any(), "stage-1 baseline produced NaN"

    dq, dkv, d_sink = run({"compute_mma_P_stage": 2, "compute_mma_dS_stage": 2})

    assert not torch.isnan(dq).any() and not torch.isnan(dkv).any() and not torch.isnan(d_sink).any(), "staged store paths produced NaN gradients"
    assert torch.equal(dq, dq_ref), "dq must be bitwise-identical between stage 1 and stage 2"

    def rel_l2(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()

    assert rel_l2(dkv, dkv_ref) < 1e-4, "dkv parity vs stage-1 baseline"
    assert rel_l2(d_sink, d_sink_ref) < 1e-4, "d_sink parity vs stage-1 baseline"
