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
