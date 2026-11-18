# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The NSA top-k reduction and compression attention kernels use tcgen05, so their check_support() declines cc 12.x.

A cc 12.x part reports a major above 10, so the former ``compute_capability < 100`` gate admitted it and the kernel then failed to
build (ptxas rejects the kernels' tcgen05 forms for ``sm_120a``)."""

import pytest
import torch

from native_sparse_attention.cutedsl.nsa_utils import allocate_input_tensors, allocate_output_tensors, nsa_init

pytestmark = pytest.mark.L0


def _report_cc12(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (12, 0))


def test_topk_reduction_declines_cc12(request, monkeypatch):
    try:
        from cudnn import NSA
    except ImportError as error:
        pytest.skip(f"cudnn optional dependencies not installed: {error}")
    cfg = nsa_init(
        request=request,
        layout="bshd",
        dtype=torch.float16,
        acc_dtype=torch.float32,
        selection_block_size=64,
        compress_stride=32,
        k_value=16,
        is_causal=True,
        mma_tiler_mn=(128, 128),
        s_q_default_override=4096,
        s_kv_default_override=128,
    )
    q, k, _, lse, _, _, cum_q, cum_kv, max_s_q, max_s_kv = allocate_input_tensors(cfg)
    _, _, _, topk_scores, topk_indices = allocate_output_tensors(cfg)
    plan = NSA.TopKReduction(
        sample_q=q,
        sample_k=k,
        sample_lse=lse,
        sample_topk_scores=topk_scores,
        sample_topk_indices=topk_indices,
        sample_cum_seqlen_q=cum_q,
        sample_cum_seqlen_k=cum_kv,
        max_s_q=max_s_q,
        max_s_k=max_s_kv,
        acc_dtype=cfg["acc_dtype"],
        k_value=cfg["k_value"],
        selection_block_size=cfg["selection_block_size"],
        compress_stride=cfg["compress_stride"],
        is_causal=cfg["is_causal"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        scale_softmax=None,
    )
    _report_cc12(monkeypatch)
    with pytest.raises(RuntimeError, match=r"not cc 12\.x"):
        plan.check_support()


def test_compression_attention_declines_cc12(request, monkeypatch):
    try:
        from cudnn import NSA
    except ImportError as error:
        pytest.skip(f"cudnn optional dependencies not installed: {error}")
    cfg = nsa_init(
        request=request,
        layout="bshd",
        dtype=torch.float16,
        acc_dtype=torch.float32,
        mma_tiler_mn=(128, 128),
        is_persistent=False,
        scale_q=1.0,
        scale_k=1.0,
        scale_v=1.0,
        inv_scale_o=1.0,
        scale_softmax=None,
    )
    q, k, v, _, _, _, cum_q, cum_k, _, _ = allocate_input_tensors(cfg)
    o, lse, _, _, _ = allocate_output_tensors(cfg)
    plan = NSA.CompressionAttention(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=o,
        sample_lse=lse.contiguous(),
        sample_cum_seqlen_q=cum_q,
        sample_cum_seqlen_k=cum_k,
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
    _report_cc12(monkeypatch)
    with pytest.raises(RuntimeError, match=r"not cc 12\.x"):
        plan.check_support()
