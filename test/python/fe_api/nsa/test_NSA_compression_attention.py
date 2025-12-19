import torch

import pytest
from test_utils import torch_fork_set_rng

from fe_api.nsa.nsa_utils import (
    with_nsa_compression_attention_params,
    nsa_init,
    allocate_input_tensors,
    allocate_output_tensors,
)
from fe_api.nsa.nsa_reference import check_ref_nsa_compression_attention


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_nsa_compression_attention_params
def test_nsa_compression_compile_execute(
    layout,
    dtype,
    acc_dtype,
    mma_tiler_mn,
    is_persistent,
    scale_q,
    scale_k,
    scale_v,
    inv_scale_o,
    scale_softmax,
    request,
):
    try:
        from cudnn import NSA
        from cuda.bindings import driver as cuda
    except ImportError as e:
        pytest.skip(
            "Environment not supported: cudnn optional dependencies not installed"
        )

    cfg = nsa_init(
        request=request,
        layout=layout,
        dtype=dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        is_persistent=is_persistent,
        scale_q=scale_q,
        scale_k=scale_k,
        scale_v=scale_v,
        inv_scale_o=inv_scale_o,
        scale_softmax=scale_softmax,
    )

    Q, K, V, _, _, _, cum_seqlen_q, cum_seqlen_k, max_s_q, max_s_k = (
        allocate_input_tensors(cfg)
    )
    O, LSE, _, _, _ = allocate_output_tensors(cfg)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if cfg["layout"] == "bshd":
        LSE = LSE.contiguous()
    elif cfg["layout"] == "thd":
        LSE = LSE.permute(2, 1, 0).contiguous().permute(2, 1, 0)

    comp_attn = NSA.CompressionAttention(
        sample_q=Q,
        sample_k=K,
        sample_v=V,
        sample_o=O,
        sample_lse=LSE,
        sample_cum_seqlen_q=cum_seqlen_q,
        sample_cum_seqlen_k=cum_seqlen_k,
        mma_tiler_mn=cfg["mma_tiler_mn"],
        qk_acc_dtype=cfg["acc_dtype"],
        pv_acc_dtype=cfg["acc_dtype"],
        is_persistent=cfg["is_persistent"],
        scale_q=cfg["scale_q"],
        scale_k=cfg["scale_k"],
        scale_v=cfg["scale_v"],
        inv_scale_o=cfg["inv_scale_o"],
        scale_softmax=cfg["scale_softmax"],
    )

    assert comp_attn.check_support()
    comp_attn.compile(current_stream=stream)
    comp_attn.execute(
        q_tensor=Q,
        k_tensor=K,
        v_tensor=V,
        o_tensor=O,
        lse_tensor=LSE,
        cum_seqlen_q_tensor=cum_seqlen_q,
        cum_seqlen_k_tensor=cum_seqlen_k,
        scale_softmax=cfg["scale_softmax"],
        current_stream=stream,
    )

    check_ref_nsa_compression_attention(
        Q,
        K,
        V,
        O,
        LSE,
        scale_output=1.0,
        scale_softmax=cfg["scale_softmax"],
        atol=2e-3,
        rtol=2e-3,
        test_config=cfg,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_nsa_compression_attention_params
def test_nsa_compression_wrapper(
    layout,
    dtype,
    acc_dtype,
    mma_tiler_mn,
    is_persistent,
    scale_q,
    scale_k,
    scale_v,
    inv_scale_o,
    scale_softmax,
    request,
):
    try:
        from cudnn import NSA
        from cuda.bindings import driver as cuda
    except ImportError as e:
        pytest.skip(
            "Environment not supported: cudnn optional dependencies not installed"
        )

    cfg = nsa_init(
        request=request,
        layout=layout,
        dtype=dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        is_persistent=is_persistent,
        scale_q=scale_q,
        scale_k=scale_k,
        scale_v=scale_v,
        inv_scale_o=inv_scale_o,
        scale_softmax=scale_softmax,
    )

    Q, K, V, _, _, _, cum_seqlen_q, cum_seqlen_k, max_s_q, max_s_k = (
        allocate_input_tensors(cfg)
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    O, LSE = NSA.compression_attention_wrapper(
        q_tensor=Q,
        k_tensor=K,
        v_tensor=V,
        cum_seqlen_q_tensor=cum_seqlen_q,
        cum_seqlen_k_tensor=cum_seqlen_k,
        enable_lse=True,
        mma_tiler_mn=cfg["mma_tiler_mn"],
        o_dtype=cfg["dtype"],
        qk_acc_dtype=cfg["acc_dtype"],
        pv_acc_dtype=cfg["acc_dtype"],
        is_persistent=cfg["is_persistent"],
        scale_q=cfg["scale_q"],
        scale_k=cfg["scale_k"],
        scale_v=cfg["scale_v"],
        inv_scale_o=cfg["inv_scale_o"],
        scale_softmax=cfg["scale_softmax"],
    )

    check_ref_nsa_compression_attention(
        Q,
        K,
        V,
        O,
        LSE,
        scale_output=1.0,
        scale_softmax=cfg["scale_softmax"],
        atol=2e-3,
        rtol=2e-3,
        test_config=cfg,
    )
