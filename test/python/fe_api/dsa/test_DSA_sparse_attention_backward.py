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
    _require_exact_sm100,
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
        (32, 576, "h32_m64", 64),
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
def test_DSA_sparse_attention_backward_deterministic_policy_is_independent():
    """Keep deterministic scheduling policy separate from ordinary tuning."""
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100 import FlashAttentionDSABackwardSm100
        from cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_deterministic import (
            FlashAttentionDSABackwardSm100Deterministic,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    assert issubclass(FlashAttentionDSABackwardSm100Deterministic, FlashAttentionDSABackwardSm100)
    assert FlashAttentionDSABackwardSm100Deterministic.num_dkv_shards == 128
    assert FlashAttentionDSABackwardSm100Deterministic.q_wave_ctas == FlashAttentionDSABackwardSm100Deterministic.num_dkv_shards
    assert FlashAttentionDSABackwardSm100Deterministic.dkv_fold_group_size == 8
    assert FlashAttentionDSABackwardSm100Deterministic.serialize_head_blocks
    assert FlashAttentionDSABackwardSm100.q_wave_ctas == 0
    assert not FlashAttentionDSABackwardSm100.serialize_head_blocks


def _exercise_deterministic_sm100_case(num_heads, head_dim, s_q, s_kv, repeats, check_short_workspace=False):
    """Run one deterministic case against bitwise and numerical contracts."""
    from cudnn import DSA
    from cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100 import flash_attn_bwd_sm100_workspace_size

    _require_sm100()
    device = torch.device("cuda")
    topk = min(64, s_kv)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device) / 10
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device) / 10
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)
    topk_length = torch.randint(1, topk + 1, (s_q,), dtype=torch.int32, device=device)
    if s_q > 1:
        topk_length[s_q // 2] = 0
    out, lse = ref_sparse_attention_forward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
    )
    dout = torch.randn_like(out)
    workspace_bytes = flash_attn_bwd_sm100_workspace_size(s_q, s_kv, head_dim, num_heads, deterministic=True)
    api = DSA.SparseAttentionBackward(
        sample_q=q,
        sample_kv=kv,
        sample_out=out,
        sample_dout=dout,
        sample_lse=lse,
        sample_attn_sink=attn_sink,
        sample_topk_idxs=topk_idxs,
        sample_topk_length=topk_length,
        softmax_scale=softmax_scale,
        deterministic=True,
    )
    assert api.check_support()
    assert api.scratch_workspace_bytes() == workspace_bytes
    expected_lse_odo_bytes = num_heads * math.ceil(s_q / 8) * 8 * 2 * torch.float32.itemsize
    expected_dkv_bytes = 128 * math.ceil(s_kv / 8) * 8 * math.ceil(head_dim / 8) * 8 * torch.float32.itemsize
    assert workspace_bytes == expected_lse_odo_bytes + expected_dkv_bytes
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)

    def run():
        """Execute after dirtying caller scratch to verify in-kernel reset."""
        # The compiled kernel, not execute-side Torch code, must initialize
        # caller-owned scratch on every reuse.
        workspace.fill_(0xA5)
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
            deterministic=True,
            workspace=workspace,
        )
        torch.cuda.synchronize()
        return result["dq"], result["dkv"], result["d_sink"]

    reference = run()
    if check_short_workspace:
        with pytest.raises(ValueError, match=rf"requires a {workspace_bytes}-byte workspace"):
            DSA.sparse_attention_backward_wrapper(
                q,
                kv,
                out,
                dout,
                lse,
                attn_sink,
                topk_idxs,
                softmax_scale=softmax_scale,
                topk_length=topk_length,
                deterministic=True,
                workspace=torch.empty(workspace_bytes - 1, dtype=torch.uint8, device=device),
            )

    output_names = ("dQ", "dKV", "dSink")
    for repeat in range(1, repeats + 1):
        actual = run()
        for name, actual_tensor, reference_tensor in zip(output_names, actual, reference):
            assert torch.equal(actual_tensor, reference_tensor), f"{name} differs from the first run at repetition {repeat}"

    check_ref_dsa_sparse_attention_backward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        out,
        dout,
        lse,
        *reference,
        softmax_scale=softmax_scale,
        topk_length=topk_length,
        atol=5e-2,
        rtol=5e-2,
    )


@torch_fork_set_rng(seed=419)
@pytest.mark.parametrize(
    "num_heads,head_dim,repeats",
    [
        pytest.param(16, 576, 3, marks=pytest.mark.L0, id="H16-D576-tail-smoke"),
        pytest.param(16, 576, 1000, marks=pytest.mark.L2, id="H16-D576-repeat1000"),
        pytest.param(32, 512, 1000, marks=pytest.mark.L2, id="H32-D512-repeat1000"),
        pytest.param(64, 512, 1000, marks=pytest.mark.L2, id="H64-D512-repeat1000"),
        pytest.param(64, 576, 1000, marks=pytest.mark.L2, id="H64-D576-repeat1000"),
        pytest.param(96, 512, 1000, marks=pytest.mark.L2, id="H96-D512-repeat1000"),
        pytest.param(128, 576, 1000, marks=pytest.mark.L2, id="H128-D576-repeat1000"),
    ],
)
def test_DSA_sparse_attention_backward_sm100_deterministic_bounded_waves(num_heads, head_dim, repeats):
    """Masked and multi-block heads must be bitwise reproducible."""
    try:
        _exercise_deterministic_sm100_case(num_heads, head_dim, 257, 256, repeats)
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")


@pytest.mark.L1
@torch_fork_set_rng(seed=421)
@pytest.mark.parametrize(
    "num_heads,head_dim,s_q,s_kv",
    [
        pytest.param(64, 512, 1, 1, id="H64-q1-kv1"),
        pytest.param(16, 576, 8, 65, id="H16-q8-kv65"),
        pytest.param(32, 512, 9, 127, id="H32-q9-kv127"),
        pytest.param(64, 576, 127, 128, id="H64-q127-kv128"),
        pytest.param(96, 512, 128, 129, id="H96-q128-kv129"),
        pytest.param(128, 576, 129, 130, id="H128-q129-kv130"),
    ],
)
def test_DSA_sparse_attention_backward_sm100_deterministic_boundaries(num_heads, head_dim, s_q, s_kv):
    """Exercise head tails, vector padding, wave boundaries, and exact scratch."""
    try:
        _exercise_deterministic_sm100_case(num_heads, head_dim, s_q, s_kv, repeats=1, check_short_workspace=True)
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")


@pytest.mark.L0
def test_DSA_sparse_attention_backward_sm100_h128_two_cta_dispatch_is_fail_closed():
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100 import _select_sm100_backend
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    supported = {
        "head_dim_v": 512,
        "dtype": torch.bfloat16,
        "max_topk": 512,
    }
    for device_capability in ((10, 0), (10, 3)):
        for max_topk in (128, 512, 1024, 1152, 2048):
            kwargs = {**supported, "device_capability": device_capability, "max_topk": max_topk}
            assert _select_sm100_backend(128, 512, **kwargs) == ("h128_2cta_m64", 64)

    supported = {**supported, "device_capability": (10, 0)}

    assert _select_sm100_backend(128, 512) == ("generic_m64", 64)
    assert _select_sm100_backend(64, 512, **supported) == ("generic_m64", 64)
    assert _select_sm100_backend(128, 576, **supported) == ("generic_m64", 64)
    # deterministic=True always routes to the generic M64 kernel, including
    # the two-CTA envelope: its bounded-wave dKV shards need one CTA per
    # query token, whereas the two-CTA path accumulates dKV with FP32 atomics.
    assert _select_sm100_backend(128, 512, **supported, deterministic=True) == ("generic_m64", 64)
    assert _select_sm100_backend(16, 576, deterministic=True) == ("generic_m64", 64)

    fallback_cases = [
        {**supported, "head_dim_v": 256},
        {**supported, "dtype": torch.float16},
        {**supported, "max_topk": 0},
        {**supported, "max_topk": 64},
        {**supported, "max_topk": 513},
        {**supported, "max_topk": 1151},
        {**supported, "max_topk": 1153},
        {**supported, "max_topk": 2112},
        {**supported, "device_capability": (10, 7)},
    ]
    for kwargs in fallback_cases:
        assert _select_sm100_backend(128, 512, **kwargs) == ("generic_m64", 64)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("topk", [128, 512, 1024, 1152, 2048])
@pytest.mark.parametrize("has_topk_length", [False, True], ids=["full-topk", "lengths"])
@torch_fork_set_rng(seed=20260829)
def test_DSA_sparse_attention_backward_sm100_h128_two_cta_masks_active_positive_oob_indices(has_topk_length, topk):
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm100
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_exact_sm100()

    s_q, s_kv, num_heads, head_dim = 4, topk + 64, 128, 512
    device = "cuda"
    dtype = torch.bfloat16
    # Match the production-scale inputs used by the DSA benchmark and the
    # other backward numerical tests while keeping the stricter 1e-2 oracle
    # tolerance below. This test targets active-OOB masking, not stress-scale
    # BF16 rounding.
    q = torch.randn(s_q, num_heads, head_dim, dtype=dtype, device=device) / 10
    kv = torch.randn(s_kv, head_dim, dtype=dtype, device=device) / 10
    dout = torch.randn_like(q) / 10
    attn_sink = torch.linspace(-1.5, 1.5, num_heads, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)

    # Positive out-of-range entries are inactive under the public reference
    # contract, exactly like -1 sentinels.  Keep at least one valid active
    # entry in every row so this exercises the normal pipeline rather than the
    # empty-row fast path.
    topk_idxs[0, 1] = s_kv
    topk_idxs[1, topk // 2 - 1] = torch.iinfo(torch.int32).max
    topk_idxs[2, topk - 2] = s_kv + 17
    topk_idxs[3, topk - 1] = s_kv
    topk_length = None
    if has_topk_length:
        topk_length = torch.tensor([2, topk // 2, topk - 1, topk], dtype=torch.int32, device=device)

    softmax_scale = head_dim**-0.5
    out, lse = ref_sparse_attention_forward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
    )
    dq = torch.full_like(q, float("nan"))
    dkv = torch.full_like(kv, float("nan"))
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
        dq=dq,
        dkv=dkv,
        stream=stream,
    )

    expected_cache_suffix = (topk, has_topk_length)
    assert any(
        key[0] == "h128_2cta_m64" and key[-2:] == expected_cache_suffix for key in _interface_sm100.flash_attn_bwd_sm100.compile_cache
    ), "H128/D512 call did not execute the two-CTA backend"

    assert not torch.isnan(result["dq"]).any()
    assert not torch.isnan(result["dkv"]).any()
    assert not torch.isnan(result["d_sink"]).any()
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
    )


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@torch_fork_set_rng(seed=20260830)
def test_DSA_sparse_attention_backward_sm100_h128_dsink_reduction_covers_tail():
    """Every query block, including a one-row tail, contributes to dSink."""
    _require_exact_sm100()
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    s_q, s_kv, num_heads, head_dim, topk = 257, 256, 128, 512, 128
    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device="cuda") / 10
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device="cuda") / 10
    dout = torch.randn_like(q) / 10
    attn_sink = torch.linspace(-2.0, 2.0, num_heads, dtype=torch.float32, device="cuda")
    topk_idxs = torch.stack([torch.randperm(s_kv, device="cuda")[:topk] for _ in range(s_q)]).to(torch.int32)
    pattern = torch.tensor([0, 1, 63, 64, 65, 127, 128, -3], dtype=torch.int32, device="cuda")
    topk_length = pattern.repeat((s_q + pattern.numel() - 1) // pattern.numel())[:s_q]
    topk_length[-1] = topk
    softmax_scale = head_dim**-0.5
    out, lse = ref_sparse_attention_forward(q, kv, attn_sink, topk_idxs, topk_length=topk_length, softmax_scale=softmax_scale)
    dq = torch.empty_like(q)
    dkv = torch.empty_like(kv)
    d_sink_runs = []
    final_result = None
    for _ in range(8):
        final_result = DSA.sparse_attention_backward_wrapper(
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            softmax_scale=softmax_scale,
            topk_length=topk_length,
            dq=dq,
            dkv=dkv,
        )
        d_sink_runs.append(final_result["d_sink"])
    torch.cuda.synchronize()
    assert final_result is not None
    delta = (out.double() * dout.double()).sum(dim=-1)
    denominator = torch.logaddexp(lse.double(), attn_sink.double().unsqueeze(0))
    tail_contribution = -torch.exp(attn_sink.double() - denominator[-1]) * delta[-1]
    assert tail_contribution.abs().max() > 1e-8
    d_sink_ref = (-torch.exp(attn_sink.double().unsqueeze(0) - denominator) * delta).sum(dim=0)
    for d_sink in d_sink_runs:
        torch.testing.assert_close(
            d_sink.double(),
            d_sink_ref,
            atol=2e-6,
            rtol=2e-6,
        )
    assert torch.isfinite(final_result["dq"]).all()
    assert torch.isfinite(final_result["dkv"]).all()


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@torch_fork_set_rng(seed=20260831)
def test_DSA_sparse_attention_backward_sm100_h128_two_cta_cuda_graph():
    try:
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm100
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_exact_sm100()

    s_q, s_kv, num_heads, head_dim, topk = 4, 192, 128, 512, 128
    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device="cuda") / 10
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device="cuda") / 10
    dout = torch.randn_like(q) / 10
    attn_sink = torch.linspace(-1.5, 1.5, num_heads, dtype=torch.float32, device="cuda")
    topk_idxs = torch.stack([torch.randperm(s_kv, device="cuda")[:topk] for _ in range(s_q)]).to(torch.int32)
    topk_length = torch.full((s_q,), topk, dtype=torch.int32, device="cuda")
    softmax_scale = head_dim**-0.5
    out, lse = ref_sparse_attention_forward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
    )
    dq = torch.empty_like(q)
    dkv = torch.empty_like(kv)

    def run():
        return DSA.sparse_attention_backward_wrapper(
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            softmax_scale=softmax_scale,
            topk_length=topk_length,
            dq=dq,
            dkv=dkv,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    dq.fill_(float("nan"))
    dkv.fill_(float("nan"))
    captured["d_sink"].fill_(float("nan"))
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    assert any(
        key[0] == "h128_2cta_m64" and key[-2:] == (topk, True) for key in _interface_sm100.flash_attn_bwd_sm100.compile_cache
    ), "CUDA Graph replay did not execute the two-CTA backend"
    assert torch.isfinite(captured["dq"]).all()
    assert torch.isfinite(captured["dkv"]).all()
    assert torch.isfinite(captured["d_sink"]).all()
    check_ref_dsa_sparse_attention_backward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        out,
        dout,
        lse,
        captured["dq"],
        captured["dkv"],
        captured["d_sink"],
        softmax_scale=softmax_scale,
        topk_length=topk_length,
    )


def _run_DSA_sparse_attention_backward_wrapper(
    *,
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
    _run_DSA_sparse_attention_backward_wrapper(
        dtype=dtype,
        acc_dtype=acc_dtype,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        num_heads=num_heads,
        topk=topk,
        has_topk_length=has_topk_length,
        request=request,
    )


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("has_topk_length", [False, True], ids=["full-topk", "lengths"])
@torch_fork_set_rng(seed=0)
def test_DSA_sparse_attention_backward_wrapper_h128(has_topk_length, request):
    _run_DSA_sparse_attention_backward_wrapper(
        dtype=torch.bfloat16,
        acc_dtype=torch.float32,
        head_dim=512,
        head_dim_v=512,
        num_heads=128,
        topk=512,
        has_topk_length=has_topk_length,
        request=request,
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
        pytest.param(512, 128, (-3, 0, 1, 63, 64, 65, 128), id="d512-h128-two-cta"),
        pytest.param(576, 32, (0, 1, 64, 65, 128, 0), id="d576-mixed"),
        pytest.param(576, 16, (0, 1, 127, 128, 129, 511, 512, 513), id="d576-h16-m128-boundaries"),
        pytest.param(576, 32, (0, 1, 63, 64, 65, 127, 128), id="d576-h32-m64-boundaries"),
    ],
)
def test_DSA_sparse_attention_backward_zero_topk_length(head_dim, num_heads, topk_length_values):
    """A nonpositive top-k length must be treated as an empty attention row."""
    if not torch.cuda.is_available():
        pytest.skip("SM90+ GPU required")
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 90:
        pytest.skip("sparse attention backward requires SM90+")
    if num_heads == 128:
        _require_exact_sm100()

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
    # D512 covers defensive negative handling around the 64-row tile boundary.
    # D576/H16 covers both sides of the dedicated kernel's 128-row boundary and
    # its final feature/top-k tails. D576/H32 covers the corresponding M64
    # boundary (a partial 64-row CTA on SM100), and every case also exercises
    # the all-empty fast path, where the expected gradients are exactly zero.
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
        # empty-row failures: an SM100 barrier deadlock, and on SM90 both a
        # one-sided named-barrier wait and an out-of-range top-k index read
        # from the unconditional first n_block.
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
@torch_fork_set_rng(seed=676)
@pytest.mark.parametrize(
    "invalid_value,topk_length_value,topk_width",
    [
        pytest.param(-1, None, 64, id="negative-padding"),
        pytest.param(-1, None, 128, id="multi-tile-negative-padding"),
        pytest.param(torch.iinfo(torch.int32).max, 1, 64, id="past-length"),
        pytest.param(-1, 64, 64, id="active-negative"),
        pytest.param(4097, None, 64, id="active-positive-oob"),
    ],
)
def test_DSA_sparse_attention_backward_sm100_masks_invalid_topk_rows(invalid_value, topk_length_value, topk_width):
    """Invalid sparse rows must contribute neither probability nor an OOB KV access."""
    _require_sm100()

    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    num_heads, head_dim, head_dim_v = 64, 576, 512
    q = torch.full((1, num_heads, head_dim), 8.0, dtype=torch.bfloat16, device=device)
    kv = torch.full((1, head_dim), -8.0, dtype=torch.bfloat16, device=device)
    dout = torch.randn(1, num_heads, head_dim_v, dtype=torch.bfloat16, device=device)
    attn_sink = torch.full((num_heads,), -math.inf, dtype=torch.float32, device=device)
    topk_idxs = torch.full((1, topk_width), invalid_value, dtype=torch.int32, device=device)
    topk_idxs[:, 0] = 0
    topk_length = None if topk_length_value is None else torch.full((1,), topk_length_value, dtype=torch.int32, device=device)

    out = kv[0, :head_dim_v].view(1, 1, head_dim_v).expand(1, num_heads, head_dim_v).contiguous()
    lse = ((q.float() * kv[0].float()).sum(dim=-1) * (192**-0.5)).contiguous()
    result = DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        softmax_scale=192**-0.5,
        topk_length=topk_length,
    )

    expected_dkv = torch.zeros_like(kv)
    expected_dkv[0, :head_dim_v] = dout.float().sum(dim=(0, 1)).to(torch.bfloat16)
    assert torch.isfinite(result["dq"]).all()
    assert torch.isfinite(result["dkv"]).all()
    assert torch.isfinite(result["d_sink"]).all()
    torch.testing.assert_close(result["dq"].float(), torch.zeros_like(result["dq"], dtype=torch.float32), atol=3e-5, rtol=0)
    torch.testing.assert_close(result["dkv"], expected_dkv, atol=5e-2, rtol=1e-2)
    assert torch.equal(result["d_sink"], torch.zeros_like(result["d_sink"]))


@pytest.mark.L0
@torch_fork_set_rng(seed=677)
@pytest.mark.parametrize("sink_value,empty_row", [(-math.inf, True), (math.inf, False)], ids=["empty-no-mass", "positive-infinite-sink"])
def test_DSA_sparse_attention_backward_sm100_handles_infinite_sink_limits(sink_value, empty_row):
    """Sink/LSE infinities must use their limiting probabilities without inf-inf."""
    _require_sm100()
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    num_heads, head_dim, head_dim_v = 64, 576, 512
    q = torch.randn(1, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    kv = torch.randn(1, head_dim, dtype=torch.bfloat16, device=device)
    out = torch.zeros(1, num_heads, head_dim_v, dtype=torch.bfloat16, device=device)
    dout = torch.randn_like(out)
    attn_sink = torch.full((num_heads,), sink_value, dtype=torch.float32, device=device)
    topk_idxs = torch.full((1, 64), torch.iinfo(torch.int32).max if empty_row else -1, dtype=torch.int32, device=device)
    topk_length = torch.zeros(1, dtype=torch.int32, device=device) if empty_row else None
    if empty_row:
        lse = torch.full((1, num_heads), -math.inf, dtype=torch.float32, device=device)
    else:
        topk_idxs[:, 0] = 0
        lse = ((q.float() * kv[0].float()).sum(dim=-1) * (192**-0.5)).contiguous()

    result = DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        softmax_scale=192**-0.5,
        topk_length=topk_length,
    )

    assert torch.equal(result["dq"], torch.zeros_like(result["dq"]))
    assert torch.equal(result["dkv"], torch.zeros_like(result["dkv"]))
    assert torch.equal(result["d_sink"], torch.zeros_like(result["d_sink"]))


@pytest.mark.L0
@torch_fork_set_rng(seed=678)
def test_DSA_sparse_attention_backward_sm100_accumulates_odo_in_fp32():
    """dSink uses the saved BF16 O but must multiply O*dO in FP32."""
    _require_sm100()
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    s_q, s_kv, num_heads, head_dim = 8, 64, 64, 576
    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device)
    attn_sink = torch.linspace(2.0, 5.0, num_heads, dtype=torch.float32, device=device)
    topk_idxs = torch.arange(s_kv, dtype=torch.int32, device=device).expand(s_q, -1).contiguous()
    out, lse = ref_sparse_attention_forward(q, kv, attn_sink, topk_idxs, softmax_scale=192**-0.5)
    dout = torch.randn_like(out)

    result = DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        softmax_scale=192**-0.5,
    )
    p_sink = torch.exp(attn_sink.view(1, -1) - torch.logaddexp(lse, attn_sink.view(1, -1)))
    expected_d_sink = (-p_sink * (out.float() * dout.float()).sum(dim=-1)).sum(dim=0)
    torch.testing.assert_close(result["d_sink"], expected_d_sink, atol=2e-5, rtol=2e-3)


@pytest.mark.L0
@torch_fork_set_rng(seed=436)
@pytest.mark.parametrize("compact", [True, False], ids=["compact", "non-compact"])
def test_DSA_sparse_attention_backward_sm90_padded_topk_columns_contribute_zero(compact):
    """Padded top-k columns must carry exactly zero probability."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm90()
    device = torch.device("cuda")
    head_dim, num_heads = 576, 32
    s_q, s_kv, topk = 4, 256, 128
    n_valid = 100  # not a multiple of the 64-row KV tile
    softmax_scale = 1.0 / math.sqrt(head_dim)

    # Opposing Q/K signs drive the KV-only LSE below the FP32 exp2 threshold.
    q = (2.2 + torch.randn(s_q, num_heads, head_dim, device=device) * 0.01).to(torch.bfloat16)
    kv = (-2.2 + torch.randn(s_kv, head_dim, device=device) * 0.01).to(torch.bfloat16)
    attn_sink = torch.full((num_heads,), -400.0, dtype=torch.float32, device=device)

    max_topk = topk if compact else n_valid + 18
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:max_topk] for _ in range(s_q)]).to(torch.int32)
    topk_length = None
    if compact:
        topk_length = torch.full((s_q,), n_valid, dtype=torch.int32, device=device)
    else:
        topk_idxs[:, n_valid:] = -1
        topk_idxs[:, 3] = -1
        topk_idxs[:, 70] = -1

    out, lse = ref_sparse_attention_forward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
    )
    # Ensure a zero-filled padded lane would overflow exp2(-LSE_log2)
    # without the probability mask.
    assert (-lse.max() * math.log2(math.e)).item() > 132
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
        topk_length=topk_length,
    )
    torch.cuda.synchronize()

    dq, dkv, d_sink = result["dq"], result["dkv"], result["d_sink"]
    assert torch.isfinite(dq).all()

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
@torch_fork_set_rng(seed=437)
@pytest.mark.parametrize(
    "sink_value",
    [
        pytest.param(2.4e38, id="finite-but-rescale-overflows"),
        pytest.param(float("inf"), id="pos-inf"),
    ],
)
def test_DSA_sparse_attention_backward_sm90_saturating_attn_sink(sink_value):
    """A saturating attention sink must not turn the gradients into NaN."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm90()
    device = torch.device("cuda")
    head_dim, num_heads, s_q, s_kv, topk = 576, 32, 4, 256, 128
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = (torch.randn(s_q, num_heads, head_dim, device=device) / 10).to(torch.bfloat16)
    kv = (torch.randn(s_kv, head_dim, device=device) / 10).to(torch.bfloat16)
    attn_sink = torch.full((num_heads,), sink_value, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)
    topk_length = torch.full((s_q,), topk, dtype=torch.int32, device=device)

    out, lse = ref_sparse_attention_forward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        topk_length=topk_length,
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
        topk_length=topk_length,
    )
    torch.cuda.synchronize()

    dq, dkv, d_sink = result["dq"], result["dkv"], result["d_sink"]
    assert torch.equal(d_sink, torch.zeros_like(d_sink))
    assert torch.equal(out, torch.zeros_like(out))
    assert torch.equal(dq, torch.zeros_like(dq))
    assert torch.equal(dkv, torch.zeros_like(dkv))


@pytest.mark.L0
@torch_fork_set_rng(seed=434)
@pytest.mark.parametrize(
    "num_heads,s_kv,topk",
    [
        pytest.param(16, 640, 513, id="h16-m128"),
        pytest.param(32, 4096, 2047, id="h32-m64"),
    ],
)
def test_DSA_sparse_attention_backward_sm100_partial_max_topk(num_heads, s_kv, topk):
    """Dedicated SM100 kernels handle a maximum top-k ending in a partial tile."""
    if not torch.cuda.is_available():
        pytest.skip("SM100 GPU required")
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        pytest.skip("dedicated H16/H32 regression test targets the SM100 kernel")

    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    s_q = 2
    head_dim = 576
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
@pytest.mark.parametrize("num_heads,topk", [(64, 64), (128, 128)], ids=["generic", "h128-two-cta"])
@pytest.mark.parametrize("has_topk_length", [False, True], ids=["full-topk", "lengths"])
@torch_fork_set_rng(seed=7)
def test_DSA_sparse_attention_backward_nondefault_stream_zero_init_ordering(num_heads, topk, has_topk_length):
    """The SM100 interface must establish zero state on the launch stream.

    The generic path uses torch zero-fills, while the H128 two-CTA path launches
    its compiled ``zero_init`` kernel. Both must be ordered before the backward
    kernel on the caller-provided ``current_stream``. Otherwise a busy ambient
    stream can delay initialization until after the kernel or let the kernel
    accumulate on top of uninitialized memory.

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
    if num_heads == 128:
        _require_exact_sm100()
    device = torch.device("cuda")
    s_q, s_kv = 256, 1024
    head_dim = 512
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device) / 10
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device) / 10
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)
    topk_length = torch.full((s_q,), topk, dtype=torch.int32, device=device) if has_topk_length else None

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
            topk_length=topk_length,
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
    if topk_length is not None:
        with torch.cuda.stream(side_stream):
            topk_length.fill_(0)
            topk_length.fill_(topk)
    dq, dkv, d_sink = run(cuda.CUstream(side_stream.cuda_stream))

    assert (dkv != 0).any(), "dkv accumulation was wiped by a zero-init racing on another stream"
    assert (d_sink != 0).any(), "d_sink accumulation was wiped by a zero-init racing on another stream"

    def rel_l2(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()

    assert torch.equal(dq, dq_ref), "dq must not depend on the launch stream"
    assert rel_l2(dkv, dkv_ref) < 1e-4, "dkv parity vs default-stream control"
    assert rel_l2(d_sink, d_sink_ref) < 1e-4, "d_sink parity vs default-stream control"


@pytest.mark.L0
@pytest.mark.parametrize("head_dim,num_heads", [(512, 64), (576, 16), (576, 32)])
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
    workspace = torch.empty(
        _interface_sm100.flash_attn_bwd_sm100_workspace_size(s_q, s_kv, head_dim, num_heads),
        dtype=torch.uint8,
        device=device,
    )

    def run(attn_sink_, topk_idxs_, topk_length_):
        """Execute once and clone outputs before the shared buffers are reused."""
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
            workspace=workspace,
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
        workspace=torch.empty(
            _interface_sm100.flash_attn_bwd_sm100_workspace_size(s_q, s_kv, head_dim, num_heads),
            dtype=torch.uint8,
            device=device,
        ),
    )

    def call(args):
        """Invoke the interface with one mutated tensor-contract case."""
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
            workspace=args["workspace"],
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

    # A contiguous view may still begin at a four-byte storage offset. The
    # SM100 loader moves FP32 pairs and must reject that pointer rather than
    # promising an unverified eight-byte alignment to the compiler.
    lse_storage = torch.empty(lse.numel() + 1, dtype=lse.dtype, device=device)
    lse_misaligned = lse_storage[1:].view_as(lse)
    lse_misaligned.copy_(lse)
    assert lse_misaligned.is_contiguous() and lse_misaligned.data_ptr() % 8 == 4
    args = dict(good)
    args["lse"] = lse_misaligned
    with pytest.raises(ValueError, match="lse must be 8-byte aligned"):
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
    assert SparseAttentionBackward(**good, deterministic=True).check_support()

    deterministic_h32 = dict(good)
    deterministic_h32["sample_q"] = q[:, :32].contiguous()
    deterministic_h32["sample_out"] = out[:, :32].contiguous()
    deterministic_h32["sample_dout"] = dout[:, :32].contiguous()
    deterministic_h32["sample_lse"] = lse[:, :32].contiguous()
    deterministic_h32["sample_attn_sink"] = attn_sink[:32].contiguous()
    assert SparseAttentionBackward(**deterministic_h32, deterministic=True).check_support()

    deterministic_h48 = dict(good)
    deterministic_h48["sample_q"] = q[:, :48].contiguous()
    deterministic_h48["sample_out"] = out[:, :48].contiguous()
    deterministic_h48["sample_dout"] = dout[:, :48].contiguous()
    deterministic_h48["sample_lse"] = lse[:, :48].contiguous()
    deterministic_h48["sample_attn_sink"] = attn_sink[:48].contiguous()
    with pytest.raises(ValueError, match="heads in"):
        SparseAttentionBackward(**deterministic_h48, deterministic=True).check_support()

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
