# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dedicated coverage for rarely-used SDPA features that are no longer exercised
by the randomized configurations in test_mhas_v2.py: ALiBi masking, the
score_max/score_sum_exp softmax outputs, and dropout. A few deterministic
configs verify that basic graph lowering and numerics keep working.
"""

import cudnn
import pytest
import torch

from sdpa.fp16 import exec_sdpa
from sdpa.random_config import ExecConfig


def make_config(
    *,
    data_type=torch.bfloat16,
    is_infer=True,
    is_alibi=False,
    is_dropout=False,
    dropout_prob=0.0,
    with_score_max=False,
    with_score_sum_exp=False,
    right_bound=None,
    implementation=cudnn.attention_implementation.AUTO,
    s_q=512,
    s_kv=512,
    with_sink_token=False,
):
    cfg = ExecConfig(
        data_type=data_type,
        rng_data_seed=1234,
        rng_geom_seed=5678,
        is_alibi=is_alibi,
        is_infer=is_infer,
        is_paged=False,
        is_bias=False,
        is_block_mask=False,
        is_padding=False,
        is_cu_seq_len=False,
        is_ragged=False,
        is_dropout=is_dropout,
        is_determin=False,
        with_score_max=with_score_max,
        with_score_sum_exp=with_score_sum_exp,
        batches=2,
        d_qk=64,
        d_v=64,
        s_q=s_q,
        s_kv=s_kv,
        h_q=4,
        h_k=4,
        h_v=4,
        diag_align=cudnn.diagonal_alignment.TOP_LEFT,
        left_bound=None,
        right_bound=right_bound,
        dropout_prob=dropout_prob,
        implementation=implementation,
        with_sink_token=with_sink_token,
    )
    cfg.fill_derived_fields()
    return cfg


@pytest.mark.L0
@pytest.mark.parametrize("is_infer", [True, False], ids=["fwd", "bwd"])
def test_sdpa_alibi(is_infer, request, cudnn_handle):
    # ALiBi requires a causal mask (right_bound=0, top-left alignment).
    cfg = make_config(is_infer=is_infer, is_alibi=True, right_bound=0)
    exec_sdpa(cfg, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize("is_infer", [True, False], ids=["fwd", "bwd"])
def test_sdpa_score_max_sum_exp(is_infer, request, cudnn_handle):
    cfg = make_config(is_infer=is_infer, with_score_max=True, with_score_sum_exp=True)
    exec_sdpa(cfg, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize("is_infer", [True, False], ids=["fwd", "bwd"])
def test_sdpa_dropout(is_infer, request, cudnn_handle):
    cfg = make_config(
        data_type=torch.float16,
        is_infer=is_infer,
        is_dropout=True,
        dropout_prob=0.1,
        right_bound=0,
    )
    exec_sdpa(cfg, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize("data_type", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("right_bound", [None, 0], ids=["no_mask", "causal"])
@pytest.mark.parametrize("s_q_s_kv", [(512, 512), (300, 1000)], ids=["s512", "s300x1000"])
def test_sdpa_dropout_bwd_unified(data_type, right_bound, s_q_s_kv, request, cudnn_handle):
    """Backward dropout on the unified SDPA backward node: the kernel regenerates the forward's Philox mask
    (the test compares the gradients against a reference built from the forward's RNG dump)."""
    cfg = make_config(
        data_type=data_type,
        is_infer=False,
        is_dropout=True,
        dropout_prob=0.1,
        right_bound=right_bound,
        s_q=s_q_s_kv[0],
        s_kv=s_q_s_kv[1],
    )
    # The unified forward does not generate Stats with dropout, so the forward runs on AUTO (composite) and only
    # the backward is pinned to the unified node.
    cfg.bwd_implementation = cudnn.attention_implementation.UNIFIED
    exec_sdpa(cfg, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize("bwd_implementation", [None, cudnn.attention_implementation.UNIFIED], ids=["auto", "unified_bwd"])
@pytest.mark.parametrize("s_q_s_kv", [(300, 1000), (92, 92)], ids=["s300x1000", "s92"])
def test_sdpa_dropout_with_sink_token(bwd_implementation, s_q_s_kv, request, cudnn_handle):
    """Dropout together with a learnable sink token (fwd + bwd). Previously untested: the sink gradient was scaled
    by the keep probability, and the SM100 forward weighted the sink by the keep probability in the row sum."""
    cfg = make_config(
        data_type=torch.bfloat16,
        is_infer=False,
        is_dropout=True,
        dropout_prob=0.1,
        right_bound=0,
        with_sink_token=True,
        s_q=s_q_s_kv[0],
        s_kv=s_q_s_kv[1],
    )
    if bwd_implementation is not None:
        cfg.bwd_implementation = bwd_implementation
    exec_sdpa(cfg, request, cudnn_handle)
