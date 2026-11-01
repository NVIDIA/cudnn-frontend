# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The DSA indexer backward and score recompute plans run their SM100 tcgen05 kernels on every non-SM90 part, so their gates
decline cc 12.x as well as pre-SM90 parts.

cc 12.x reports a major above 10, so the former ``major != 9 and major < 10`` gate admitted it: check_support() returned True and
the tcgen05 kernel then failed to build for ``sm_120a``."""

import pytest
import torch

pytestmark = pytest.mark.L0


def _report_cc(monkeypatch, cc):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: cc)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_indexer_backward_declines_cc12(monkeypatch):
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    b, s_q, s_k, h, d, topk = 1, 1, 128, 64, 128, 128
    plan = DSA.IndexerBackward(
        sample_index_q=torch.empty((b, s_q, h, d), dtype=torch.bfloat16, device="cuda"),
        sample_weights=torch.empty((b, s_q, h), dtype=torch.bfloat16, device="cuda"),
        sample_index_k=torch.empty((b, s_k, d), dtype=torch.bfloat16, device="cuda"),
        sample_d_index_q=torch.empty((b, s_q, h, d), dtype=torch.bfloat16, device="cuda"),
        sample_d_weights=torch.empty((b, s_q, h), dtype=torch.bfloat16, device="cuda"),
        sample_d_index_k=torch.empty((b, s_k, d), dtype=torch.bfloat16, device="cuda"),
        sample_attn_score=torch.empty((b, s_q, topk), dtype=torch.float32, device="cuda"),
        sample_index_score=torch.empty((b, s_q, topk), dtype=torch.float32, device="cuda"),
        sample_topk_indices=torch.empty((b, s_q, topk), dtype=torch.int32, device="cuda"),
        sm_scale=1.0,
        block_I=128,
    )
    _report_cc(monkeypatch, (12, 0))
    with pytest.raises(RuntimeError, match=r"not cc 12\.x"):
        plan.check_support()


@pytest.mark.parametrize(
    "cc,declined", [((8, 0), True), ((9, 0), False), ((10, 0), False), ((10, 3), False), ((11, 0), False), ((12, 0), True), ((12, 1), True)]
)
def test_score_recompute_arch_gate(monkeypatch, cc, declined):
    try:
        from cudnn.deepseek_sparse_attention.score_recompute import api
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    from cudnn.api_base import APIBase

    class _Plan(APIBase):
        def check_support(self):
            return True

        def compile(self):
            pass

        def execute(self, *args, **kwargs):
            pass

    _report_cc(monkeypatch, cc)
    if declined:
        with pytest.raises(RuntimeError, match="requires SM90 or SM100"):
            api._check_score_arch(_Plan())
    else:
        api._check_score_arch(_Plan())
