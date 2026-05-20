import pytest
import torch

from test_utils import torch_fork_set_rng

from fe_api.dsa.dsa_utils import dsa_init, with_dsa_dense_indexer_backward_params
from fe_api.dsa.dsa_reference import (
    _bottom_right_causal_mask,
    check_ref_dense_indexer_backward,
    ref_dense_indexer_score_recompute,
)


def _allocate(cfg, sm_scale: float, ratio: int):
    b = cfg["b"]
    s_q = cfg["s_q"]
    s_k = cfg["s_kv"]
    d = cfg["head_dim"]
    h = cfg["qhead_per_kv_head"]
    device = "cuda"

    index_q = torch.randn(b, s_q, h, d, dtype=torch.bfloat16, device=device)
    weights = torch.randn(b, s_q, h, dtype=torch.bfloat16, device=device)
    index_k = torch.randn(b, s_k, d, dtype=torch.bfloat16, device=device)

    with torch.no_grad():
        index_score, index_lse = ref_dense_indexer_score_recompute(
            index_q,
            index_k.unsqueeze(2),
            weights,
            ratio=ratio,
        )
        if sm_scale != 1.0:
            index_score = index_score * sm_scale
            valid = _bottom_right_causal_mask(s_q, s_k, ratio, device)
            index_lse = torch.logsumexp(
                index_score.masked_fill(~valid.unsqueeze(0), float("-inf")),
                dim=-1,
            )

        valid = _bottom_right_causal_mask(s_q, s_k, ratio, device)
        attn_score = torch.rand(b, s_q, s_k, dtype=torch.float32, device=device)
        attn_score = attn_score.masked_fill(~valid.unsqueeze(0), 0.0).contiguous()
        attn_l1norm = attn_score.sum(dim=-1).contiguous()

    return (
        index_q,
        weights,
        index_k,
        attn_score.contiguous(),
        attn_l1norm,
        index_score.contiguous(),
        index_lse.contiguous(),
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_dsa_dense_indexer_backward_params
def test_DSA_dense_indexer_backward_wrapper(
    dtype,
    acc_dtype,
    head_dim,
    qhead_per_kv_head,
    block_I,
    ratio,
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
        ratio=ratio,
        min_compute_capability=90,
        s_q_default=128,
        s_kv_default=512,
    )
    sm_scale = 1.0
    b_cfg = cfg["b"]
    s_q_cfg = cfg["s_q"]
    loss_coeff = float(b_cfg * s_q_cfg)
    grad_loss = 1.0
    grad_scale_expected = (loss_coeff / (b_cfg * s_q_cfg)) * grad_loss

    (
        index_q,
        weights,
        index_k,
        attn_score,
        attn_l1norm,
        index_score,
        index_lse,
    ) = _allocate(cfg, sm_scale=sm_scale, ratio=ratio)
    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    attn_score_ref = attn_score.clone()
    attn_l1norm_ref = attn_l1norm.clone()
    torch_stream.wait_stream(torch.cuda.current_stream())
    try:
        result = DSA.dense_indexer_backward_wrapper(
            index_q,
            weights,
            index_k,
            attn_score,
            attn_l1norm,
            index_score,
            index_lse,
            sm_scale=sm_scale,
            loss_coeff=loss_coeff,
            grad_loss=grad_loss,
            block_I=block_I,
            ratio=ratio,
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
        check_ref_dense_indexer_backward(
            index_q,
            weights,
            index_k,
            attn_score_ref,
            attn_l1norm_ref,
            d_index_q,
            d_weights,
            d_index_k,
            sm_scale=sm_scale,
            ratio=ratio,
            grad_scale=grad_scale_expected,
        )
