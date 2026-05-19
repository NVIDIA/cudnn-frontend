import pytest
import torch

from test_utils import torch_fork_set_rng

from fe_api.dsa.dsa_utils import dsa_init, with_dsa_indexer_forward_params
from fe_api.dsa.dsa_reference import check_ref_indexer_forward


def _allocate_inputs(cfg):
    b = cfg["b"]
    s_q = cfg["s_q"]
    s_k = cfg["s_kv"]
    d = cfg["head_dim"]
    qhpkv = cfg["qhead_per_kv_head"]
    h_kv = cfg["h_kv"]
    h_q = h_kv * qhpkv

    q = torch.randn(b, s_q, h_q, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(b, s_k, h_kv, d, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(b, s_q, h_q, dtype=torch.bfloat16, device="cuda")
    return q, k, w


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_dsa_indexer_forward_params
def test_DSA_indexer_forward_wrapper(
    dtype,
    acc_dtype,
    head_dim,
    qhead_per_kv_head,
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
        ratio=ratio,
        s_q_default=256,
        s_kv_default=512,
    )
    q, k, w = _allocate_inputs(cfg)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    try:
        result = DSA.indexer_forward_wrapper(
            q,
            k,
            w,
            ratio=ratio,
            qhead_per_kv_head=qhead_per_kv_head,
            stream=stream,
        )
    except (ValueError, NotImplementedError, RuntimeError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    scores = result["scores"]
    if not cfg["skip_ref"]:
        check_ref_indexer_forward(q, k, w, scores, ratio)
