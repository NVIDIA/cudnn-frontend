import pytest
import torch

from test_utils import torch_fork_set_rng

from fe_api.dsa.dsa_utils import dsa_init, with_dsa_indexer_backward_params
from fe_api.dsa.dsa_reference import (
    _indexer_predict_distribution,
    check_ref_indexer_backward,
)


def _allocate(cfg, sm_scale: float):
    b = cfg["b"]
    s_q = cfg["s_q"]
    s_k = cfg["s_kv"]
    d = cfg["head_dim"]
    h = cfg["qhead_per_kv_head"]
    topk = cfg["topk"]
    device = "cuda"

    index_q = torch.randn(b, s_q, h, d, dtype=torch.bfloat16, device=device)
    weights = torch.randn(b, s_q, h, dtype=torch.bfloat16, device=device)
    index_k = torch.randn(b, s_k, d, dtype=torch.bfloat16, device=device)

    topk_k = min(topk, s_k)
    topk_indices = torch.stack([torch.stack([torch.randperm(s_k, device=device)[:topk_k] for _ in range(s_q)]) for _ in range(b)]).to(torch.int32)
    if topk_k < topk:
        pad = torch.full((b, s_q, topk - topk_k), -1, dtype=torch.int32, device=device)
        topk_indices = torch.cat([topk_indices, pad], dim=-1)

    # ``index_score`` is the predict distribution the kernel consumes. It
    # must be consistent with ``(index_q, weights, index_k)`` under the
    # forward scoring math; otherwise the kernel's grad_signal will be
    # computed for a predict that doesn't match the reference's
    # autograd-through-the-forward computation.
    with torch.no_grad():
        index_score = _indexer_predict_distribution(
            index_q.float(),
            index_k.float(),
            weights.float(),
            topk_indices,
            sm_scale,
        ).contiguous()

    # Target distribution — keep random so the KL grad is non-trivial.
    attn_score = (
        torch.softmax(
            torch.randn(b, s_q, topk, device=device),
            dim=-1,
        )
        .float()
        .contiguous()
    )

    return index_q, weights, index_k, attn_score, index_score, topk_indices


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_dsa_indexer_backward_params
def test_DSA_indexer_backward_wrapper(
    dtype,
    acc_dtype,
    head_dim,
    qhead_per_kv_head,
    block_I,
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
        qhead_per_kv_head=qhead_per_kv_head,
        block_I=block_I,
        min_compute_capability=90,
        s_q_default=128,
        s_kv_default=512,
    )
    sm_scale = 1.0
    # Configure loss_coeff and grad_loss so the kernel's internal
    # grad_scale = (loss_coeff / (B * S_q)) * grad_loss equals 1.0 — then
    # the reference (which uses a unit grad_scale) and the kernel agree.
    b_cfg = cfg["b"]
    s_q_cfg = cfg["s_q"]
    loss_coeff = float(b_cfg * s_q_cfg)
    grad_loss = 1.0
    grad_scale_expected = (loss_coeff / (b_cfg * s_q_cfg)) * grad_loss  # = 1.0

    (
        index_q,
        weights,
        index_k,
        attn_score,
        index_score,
        topk_indices,
    ) = _allocate(cfg, sm_scale=sm_scale)
    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    # The kernel mutates attn_score + index_score in-place during its
    # score-grad stage. Keep pre-call copies so the reference can consume the
    # same inputs the kernel was given.
    attn_score_ref = attn_score.clone()
    index_score_ref = index_score.clone()
    torch_stream.wait_stream(torch.cuda.current_stream())
    try:
        result = DSA.indexer_backward_wrapper(
            index_q,
            weights,
            index_k,
            attn_score,
            index_score,
            topk_indices,
            sm_scale=sm_scale,
            loss_coeff=loss_coeff,
            grad_loss=grad_loss,
            block_I=block_I,
            stream=stream,
        )
    except (ValueError, NotImplementedError, RuntimeError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    torch_stream.synchronize()

    d_index_q = result["d_index_q"]
    d_weights = result["d_weights"]
    d_index_k = result["d_index_k"]

    assert d_index_q.shape == index_q.shape
    assert d_weights.shape == weights.shape
    assert d_index_k.shape == index_k.shape
    assert torch.isfinite(d_index_q.float()).all()
    assert torch.isfinite(d_weights.float()).all()
    assert torch.isfinite(d_index_k.float()).all()

    if not cfg["skip_ref"]:
        check_ref_indexer_backward(
            index_q,
            weights,
            index_k,
            attn_score_ref,
            index_score_ref,
            topk_indices,
            d_index_q,
            d_weights,
            d_index_k,
            sm_scale=sm_scale,
            grad_scale=grad_scale_expected,
        )
