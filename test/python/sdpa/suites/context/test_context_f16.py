# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context (prefill forward) suites, f16 — the 16-bit family: each config
draws fp16 or bf16 (data_type fuzz), like fp8 draws e4m3/e5m2. Also hosts
the deterministic mixed seq-len form cases."""

import cudnn
import pytest

from sdpa.random_config import ExecConfig
from sdpa.fp16 import exec_sdpa
from sdpa.suites.common import run_suite, suite_seeds


@pytest.mark.L0
@pytest.mark.parametrize(
    "test_no", suite_seeds("context.f16.dense"), ids=lambda p: f"test{p[0]}"
)
def test_context_f16_dense(env_info, test_no, request, cudnn_handle):
    run_suite("context.f16.dense", env_info, test_no, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize(
    "test_no", suite_seeds("context.f16.thd"), ids=lambda p: f"test{p[0]}"
)
def test_context_f16_thd(env_info, test_no, request, cudnn_handle):
    run_suite("context.f16.thd", env_info, test_no, request, cudnn_handle)


@pytest.mark.L1
@pytest.mark.parametrize(
    "test_no", suite_seeds("context.f16.thd_offset_mult"), ids=lambda p: f"test{p[0]}"
)
def test_context_f16_thd_offset_mult(env_info, test_no, request, cudnn_handle):
    run_suite("context.f16.thd_offset_mult", env_info, test_no, request, cudnn_handle)


MIXED_SEQ_LEN_FORM_CASES = [
    ("q", cudnn.diagonal_alignment.TOP_LEFT, None),
    ("kv", cudnn.diagonal_alignment.TOP_LEFT, None),
    ("q", cudnn.diagonal_alignment.BOTTOM_RIGHT, 0),
    ("kv", cudnn.diagonal_alignment.BOTTOM_RIGHT, 0),
]


@pytest.mark.parametrize(
    "cu_sides,diag_align,right_bound",
    MIXED_SEQ_LEN_FORM_CASES,
    ids=["cu_q", "cu_kv", "cu_q_brcm", "cu_kv_brcm"],
)
@pytest.mark.L0
def test_context_mixed_seq_len_forms(
    env_info, cu_sides, diag_align, right_bound, request, cudnn_handle
):
    """Mixed-form sequence lengths: cumulative on one side, per-batch on the
    other. Deterministic configs with non-uniform per-batch lengths, so
    misreading one side's form cannot produce a passing result. Requires
    cuDNN 9.25+ (skips below via exec_sdpa)."""
    import torch

    cfg = ExecConfig(
        data_type=torch.bfloat16,
        rng_data_seed=1234,
        rng_geom_seed=5678,
        is_alibi=False,
        is_infer=True,
        is_paged=False,
        is_bias=False,
        is_block_mask=False,
        is_padding=True,
        is_cu_seq_len=True,
        cu_seq_len_sides=cu_sides,
        is_ragged=False,
        is_dropout=False,
        is_determin=False,
        batches=4,
        d_qk=64,
        d_v=64,
        s_q=256,
        s_kv=512,
        h_q=3,
        h_k=3,
        h_v=3,
        diag_align=diag_align,
        left_bound=None,
        right_bound=right_bound,
        seq_len_q=[128, 100, 256, 37],
        seq_len_kv=[96, 64, 512, 200],
    )
    cfg.fill_derived_fields()
    exec_sdpa(cfg, request, cudnn_handle)
