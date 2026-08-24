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
    _require_sm100,
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


@pytest.mark.parametrize(
    "num_heads,head_dim,expected_backend,expected_block_tile",
    [
        (16, 576, "h16_m128", 128),
        (16, 512, "generic_m64", 64),
        (32, 576, "generic_m64", 64),
        (64, 576, "generic_m64", 64),
    ],
)
@pytest.mark.L0
def test_DSA_sparse_attention_backward_sm100_auto_dispatch(
    num_heads,
    head_dim,
    expected_backend,
    expected_block_tile,
):
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100 import _select_sm100_backend
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    assert _select_sm100_backend(num_heads, head_dim) == (expected_backend, expected_block_tile)


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
@pytest.mark.parametrize("has_topk_length", [False, True], ids=["full-topk", "lengths"])
@torch_fork_set_rng(seed=418)
def test_DSA_sparse_attention_backward_sm100_576_includes_sink_in_normalization(has_topk_length):
    """The 576/512 specialization must include sink mass in every gradient."""
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        pytest.skip("sink-normalization regression test targets the SM100 kernel")

    device = torch.device("cuda")
    s_q = 5
    s_kv = topk = 512
    num_heads, head_dim = 32, 576
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device)
    # Keep the sink probability significant even for the full-top-k row so
    # both compiled variants fail clearly if they normalize by KV-only LSE.
    attn_sink = torch.full((num_heads,), math.log(topk), dtype=torch.float32, device=device)
    topk_idxs = torch.arange(topk, dtype=torch.int32, device=device).expand(s_q, -1).contiguous()
    lengths = torch.tensor([1, 63, 64, 65, 512], dtype=torch.int32, device=device)
    topk_length = lengths if has_topk_length else None

    out, lse = ref_sparse_attention_forward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
    )
    # O.dO is positive for every head, so a zero d_sink cannot pass by
    # landing inside the absolute tolerance.
    dout = out.clone()
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
        topk_length=topk_length,
        stream=stream,
    )

    check_ref_dsa_sparse_attention_backward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        out,
        dout,
        lse,
        result["dq"],
        result["dkv"],
        result["d_sink"],
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
        pytest.param(576, 16, (0, 1, 127, 128, 129, 511, 512, 513), id="d576-h16-m128-boundaries"),
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
    is_h16 = head_dim == 576 and num_heads == 16
    s_kv, topk = (640, 513) if is_h16 else (256, 128)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device) / 10
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device) / 10
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    base_topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)

    # D512 covers defensive negative handling around the 64-row tile boundary.
    # D576/H16 covers both sides of the dedicated kernel's 128-row boundary and
    # its final feature/top-k tails. The all-empty cases make every expected
    # gradient exactly zero; D576/H32 also covers a partial 64-head CTA.
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
@torch_fork_set_rng(seed=434)
def test_DSA_sparse_attention_backward_sm100_h16_partial_max_topk():
    """H16 M128 handles a maximum top-k ending in a partial sparse tile."""
    if not torch.cuda.is_available():
        pytest.skip("SM100 GPU required")
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        pytest.skip("H16 M128 regression test targets the SM100 kernel")

    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    s_q, s_kv, num_heads = 2, 640, 16
    head_dim, topk = 576, 513
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device) / 10
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device) / 10
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)

    out, lse = ref_sparse_attention_forward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        softmax_scale=softmax_scale,
    )
    dout = torch.randn_like(out)
    result = DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        softmax_scale=softmax_scale,
    )
    torch.cuda.synchronize()

    check_ref_dsa_sparse_attention_backward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        out,
        dout,
        lse,
        result["dq"],
        result["dkv"],
        result["d_sink"],
        softmax_scale=softmax_scale,
        atol=5e-2,
        rtol=5e-2,
    )


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


@pytest.mark.L0
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@torch_fork_set_rng(seed=7)
def test_DSA_sparse_attention_backward_nondefault_stream_zero_init_ordering():
    """The SM100 interface allocates and zero-initializes dq/dkv/d_sink and the
    two workspaces with plain torch calls, which enqueue on the ambient torch
    stream, while the kernel launches on the caller-provided ``current_stream``.
    Without explicit stream scoping the two are unordered: with a busy ambient
    stream, the semantically required zero-fills land *after* the kernel and
    wipe the dkv/d_sink accumulation (or, in the other interleaving, the kernel
    accumulates on top of uninitialized memory).

    The ambient default stream is parked on ``torch.cuda._sleep`` so the
    unordered interleaving is reached reliably (the zero-fills cannot start
    until the sleep retires, while the side-stream kernel is free to run);
    the test needs the GPU to itself for that reason."""
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm100()
    device = torch.device("cuda")
    s_q, s_kv, num_heads = 256, 1024, 64
    head_dim, topk = 512, 64
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device) / 10
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device) / 10
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)

    out, lse = ref_sparse_attention_forward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        softmax_scale=softmax_scale,
    )
    dout = torch.randn_like(out)

    def run(stream):
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
        return result["dq"], result["dkv"], result["d_sink"]

    # Control on the ambient (default) stream; also primes the compile cache
    # so the raced call below is a pure execute.
    default_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    dq_ref, dkv_ref, d_sink_ref = run(default_stream)
    assert (dkv_ref != 0).any(), "control dkv must have nonzero content"
    assert (d_sink_ref != 0).any(), "control d_sink must have nonzero content"

    # Park the ambient default stream, then launch on a side stream. The
    # interface's zero-fills must be ordered with the side-stream kernel, not
    # queued behind the sleep on the default stream.
    side_stream = torch.cuda.Stream()
    torch.cuda._sleep(2_000_000_000)
    dq, dkv, d_sink = run(cuda.CUstream(side_stream.cuda_stream))

    assert (dkv != 0).any(), "dkv accumulation was wiped by a zero-init racing on another stream"
    assert (d_sink != 0).any(), "d_sink accumulation was wiped by a zero-init racing on another stream"

    def rel_l2(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()

    assert torch.equal(dq, dq_ref), "dq must not depend on the launch stream"
    assert rel_l2(dkv, dkv_ref) < 1e-4, "dkv parity vs default-stream control"
    assert rel_l2(d_sink, d_sink_ref) < 1e-4, "d_sink parity vs default-stream control"


@pytest.mark.L0
@pytest.mark.parametrize("head_dim,num_heads", [(512, 64), (576, 16)])
@torch_fork_set_rng(seed=16)
def test_DSA_sparse_attention_backward_fp16_sm100_numerics(head_dim, num_heads):
    """SM100 must compile FP16 inputs with FP16 MMA/storage semantics."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm100()
    device = torch.device("cuda")
    s_q, s_kv, topk = 4, 128, 64
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.float16, device=device)
    kv = torch.randn(s_kv, head_dim, dtype=torch.float16, device=device)
    attn_sink = torch.linspace(-2.0, 2.0, num_heads, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)
    topk_length = torch.tensor([16, 32, 48, 64], dtype=torch.int32, device=device)

    out, lse = ref_sparse_attention_forward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
    )
    dout = torch.randn_like(out)
    assert q.dtype == kv.dtype == out.dtype == dout.dtype == torch.float16
    assert lse.dtype == torch.float32

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
    dq, dkv, d_sink = result["dq"], result["dkv"], result["d_sink"]

    assert dq.dtype == dkv.dtype == torch.float16
    assert d_sink.dtype == torch.float32
    assert torch.isfinite(dq).all()
    assert torch.isfinite(dkv).all()
    assert torch.isfinite(d_sink).all()
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
@torch_fork_set_rng(seed=23)
def test_DSA_sparse_attention_backward_noncontiguous_aux_inputs():
    """The interface normalizes q/kv/out/dout/lse to contiguous but not
    attn_sink/topk_idxs/topk_length. Non-contiguous aux tensors previously
    escaped down to the CuTe DSL layer and failed there with low-level stride
    errors (a signature mismatch against the shared compile-cache entry on the
    warm path, a leading-stride assert on the cold path). They must be
    normalized like every other input; both cache paths are covered here."""
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm100
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm100()
    device = torch.device("cuda")
    s_q, s_kv, num_heads = 256, 1024, 64
    head_dim, topk = 576, 64
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device) / 10
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device) / 10
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)
    topk_length = torch.randint(1, topk + 1, (s_q,), dtype=torch.int32, device=device)

    out, lse = ref_sparse_attention_forward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
    )
    dout = torch.randn_like(out)

    def run(attn_sink_, topk_idxs_, topk_length_):
        dq, dkv, d_sink = _interface_sm100.flash_attn_bwd_sm100(
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink_,
            topk_idxs_,
            softmax_scale=softmax_scale,
            topk_length=topk_length_,
        )
        torch.cuda.synchronize()
        return dq.clone(), dkv.clone(), d_sink.clone()

    def strided_copy_1d(t):
        base = torch.zeros(2 * t.shape[0], dtype=t.dtype, device=t.device)
        base[::2] = t
        view = base[::2]
        assert not view.is_contiguous() and torch.equal(view, t)
        return view

    def strided_copy_2d(t):
        base = torch.zeros(t.shape[0], 2 * t.shape[1], dtype=t.dtype, device=t.device)
        base[:, ::2] = t
        view = base[:, ::2]
        assert not view.is_contiguous() and torch.equal(view, t)
        return view

    def rel_l2(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()

    # Cold path: nothing cached for this compile key, first call is
    # non-contiguous (previously a leading-stride error inside the DSL).
    _interface_sm100.flash_attn_bwd_sm100.compile_cache.clear()
    dq_cold, dkv_cold, d_sink_cold = run(strided_copy_1d(attn_sink), strided_copy_2d(topk_idxs), strided_copy_1d(topk_length))

    # Contiguous control (same compile key, warm cache).
    dq_ref, dkv_ref, d_sink_ref = run(attn_sink, topk_idxs, topk_length)

    # Warm path: non-contiguous call against the cached contiguous signature
    # (previously a signature stride mismatch inside the DSL).
    dq, dkv, d_sink = run(strided_copy_1d(attn_sink), strided_copy_2d(topk_idxs), strided_copy_1d(topk_length))

    for tag, (dq_t, dkv_t, d_sink_t) in {"cold": (dq_cold, dkv_cold, d_sink_cold), "warm": (dq, dkv, d_sink)}.items():
        assert torch.equal(dq_t, dq_ref), f"{tag}: dq must not depend on aux input contiguity"
        assert rel_l2(dkv_t, dkv_ref) < 1e-4, f"{tag}: dkv parity vs contiguous control"
        assert rel_l2(d_sink_t, d_sink_ref) < 1e-4, f"{tag}: d_sink parity vs contiguous control"


@pytest.mark.L0
@torch_fork_set_rng(seed=42)
def test_DSA_sparse_attention_backward_cross_shape_validation():
    """The compiled kernel takes every dimension as a dynamic value derived
    from q, so mis-shaped companion tensors do not fail: a transposed dout or
    lse runs without any error and returns silently corrupted gradients
    (measured rel-L2 vs the correct result: ~1.1 and ~45 respectively).
    The interface must validate the cross-tensor contract up front. Caller
    provided dq/dkv must additionally be contiguous: the compile cache is
    keyed without output strides, so a strided out-parameter would be written
    through the wrong layout."""
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm100
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm100()
    device = torch.device("cuda")
    s_q, s_kv, num_heads = 4, 128, 64
    head_dim, head_dim_v, topk = 576, 512, 64
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device)
    out = torch.randn(s_q, num_heads, head_dim_v, dtype=torch.bfloat16, device=device)
    dout = torch.randn_like(out)
    lse = torch.randn(s_q, num_heads, dtype=torch.float32, device=device)
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)
    topk_length = torch.full((s_q,), topk, dtype=torch.int32, device=device)

    good = dict(
        q=q,
        kv=kv,
        out=out,
        dout=dout,
        lse=lse,
        attn_sink=attn_sink,
        topk_idxs=topk_idxs,
        topk_length=topk_length,
        dq=None,
        dkv=None,
    )

    def call(args):
        _interface_sm100.flash_attn_bwd_sm100(
            args["q"],
            args["kv"],
            args["out"],
            args["dout"],
            args["lse"],
            args["attn_sink"],
            args["topk_idxs"],
            softmax_scale=softmax_scale,
            topk_length=args["topk_length"],
            dq=args["dq"],
            dkv=args["dkv"],
        )

    shape_cases = {
        "kv": kv[:, : head_dim - 64].contiguous(),
        "out": out.transpose(1, 2).contiguous(),
        "dout": dout.transpose(1, 2).contiguous(),
        "lse": lse.transpose(0, 1).contiguous(),
        "attn_sink": attn_sink[: num_heads // 2].contiguous(),
        "topk_idxs": topk_idxs[: s_q - 1].contiguous(),
        "topk_length": topk_length[: s_q - 1].contiguous(),
    }
    for name, bad_tensor in shape_cases.items():
        args = dict(good)
        args[name] = bad_tensor
        with pytest.raises(AssertionError, match=f"{name} shape mismatch"):
            call(args)

    args = dict(good)
    args["topk_length"] = topk_length.to(torch.int64)
    with pytest.raises(AssertionError, match="topk_length dtype mismatch"):
        call(args)

    # Caller-provided out-params must be contiguous (they are not copied).
    dq_strided = torch.empty(s_q, num_heads, 2 * head_dim, dtype=q.dtype, device=device)[..., ::2]
    args = dict(good)
    args["dq"] = dq_strided
    with pytest.raises(AssertionError, match="dq must be contiguous"):
        call(args)
    dkv_strided = torch.empty(s_kv, 2 * head_dim, dtype=kv.dtype, device=device)[:, ::2]
    args = dict(good)
    args["dkv"] = dkv_strided
    with pytest.raises(AssertionError, match="dkv must be contiguous"):
        call(args)


@pytest.mark.L0
@torch_fork_set_rng(seed=51)
def test_DSA_sparse_attention_backward_check_support_validates_contract():
    """check_support works on metadata-only descriptors and is the advertised
    support gate, so the cross-tensor contract must be enforced there as well,
    not only by the runtime asserts in the execution interface."""
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import SparseAttentionBackward
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm100()
    device = torch.device("cuda")
    s_q, s_kv, num_heads = 4, 128, 64
    head_dim, head_dim_v, topk = 576, 512, 64

    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device)
    out = torch.randn(s_q, num_heads, head_dim_v, dtype=torch.bfloat16, device=device)
    dout = torch.randn_like(out)
    lse = torch.randn(s_q, num_heads, dtype=torch.float32, device=device)
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)
    topk_length = torch.full((s_q,), topk, dtype=torch.int32, device=device)

    good = dict(
        sample_q=q,
        sample_kv=kv,
        sample_out=out,
        sample_dout=dout,
        sample_lse=lse,
        sample_attn_sink=attn_sink,
        sample_topk_idxs=topk_idxs,
        sample_topk_length=topk_length,
    )
    assert SparseAttentionBackward(**good).check_support()

    fp16_good = dict(good)
    for name in ("sample_q", "sample_kv", "sample_out", "sample_dout"):
        fp16_good[name] = fp16_good[name].to(torch.float16)
    assert SparseAttentionBackward(**fp16_good).check_support()

    bad_cases = {
        "sample_kv": kv[:, : head_dim - 64].contiguous(),
        "sample_out": out.transpose(1, 2).contiguous(),
        "sample_dout": dout.transpose(1, 2).contiguous(),
        "sample_lse": lse.transpose(0, 1).contiguous(),
        "sample_attn_sink": attn_sink[: num_heads // 2].contiguous(),
        "sample_topk_idxs": topk_idxs[: s_q - 1].contiguous(),
        "sample_topk_length": topk_length[: s_q - 1].contiguous(),
    }
    for name, bad_tensor in bad_cases.items():
        kwargs = dict(good)
        kwargs[name] = bad_tensor
        with pytest.raises(ValueError):
            SparseAttentionBackward(**kwargs).check_support()

    # Device placement: check_support must reject CPU inputs that the
    # SM90/SM100 runtime would otherwise reject at launch time.
    cpu_kwargs = {name: tensor.to("cpu") for name, tensor in good.items()}
    with pytest.raises(ValueError, match="Q must live on CUDA"):
        SparseAttentionBackward(**cpu_kwargs).check_support()

    # Cross-device: Q stays on CUDA, KV moved to CPU (no new CUDA allocation;
    # reuses the good tensors) -> device-consistency failure.
    cross_kwargs = dict(good)
    cross_kwargs["sample_kv"] = good["sample_kv"].to("cpu")
    with pytest.raises(ValueError, match="must share Q's device"):
        SparseAttentionBackward(**cross_kwargs).check_support()

    # head_dim must be one of {512, 576}: the kernel is tiled only for those.
    bad_head_dim = dict(good)
    bad_head_dim["sample_q"] = torch.randn(s_q, num_heads, 128, dtype=torch.bfloat16, device=device)
    bad_head_dim["sample_kv"] = torch.randn(s_kv, 128, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="head_dim must be 512 or 576"):
        SparseAttentionBackward(**bad_head_dim).check_support()


@pytest.mark.L0
@torch_fork_set_rng(seed=7)
def test_DSA_sparse_attention_backward_rejects_unsupported_head_dim_runtime():
    """The SM100 kernel is tiled only for head_dim in {512, 576}; any other
    head_dim indexes shared memory out of bounds and crashes inside the kernel.
    The interface must reject it before any compile/launch."""
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm100
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm100()
    device = torch.device("cuda")
    s_q, s_kv, num_heads = 4, 128, 64
    head_dim, head_dim_v, topk = 128, 128, 64
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device)
    out = torch.randn(s_q, num_heads, head_dim_v, dtype=torch.bfloat16, device=device)
    dout = torch.randn_like(out)
    lse = torch.randn(s_q, num_heads, dtype=torch.float32, device=device)
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)

    with pytest.raises(AssertionError, match="head_dim must be 512 or 576"):
        _interface_sm100.flash_attn_bwd_sm100(
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            softmax_scale=softmax_scale,
        )
