# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context (prefill forward) suites, f16 — the 16-bit family: each config
draws fp16 or bf16 (data_type fuzz), like fp8 draws e4m3/e5m2. Knob
factories and SuiteSpecs live here, next to their shims; registry.py
aggregates the per-file SUITES lists. Also hosts the deterministic mixed
seq-len form cases (pinned regression tests, not a fuzz suite)."""

import cudnn
import torch
import pytest
from sdpa.random_config import (
    RandomBatchSize,
    RandomChoice,
    RandomHeadGenerator,
    RandomHiddenDimSize,
    RandomSequenceLength,
    SlidingWindowMaskGenerator,
)
from sdpa.random_config import ExecConfig
from sdpa.fp16 import exec_sdpa
from sdpa.suites.common import (
    COMMON_FUZZ,
    MASK_FUZZ,
    SuiteSpec,
    THD_FUZZ,
    run_suite,
)
from sdpa.suites.knobs import (
    DIAG_BOTH,
    DIAG_BR_HEAVY,
    SW_FULL,
    _f16,
)


def dense_fwd():
    return dict(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1, 4]),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=1,
            s_q_max=8192,
            s_kv_min=1,
            s_kv_max=16384,
            s_q_distribution={
                "s_q=1": 0,
                "s_q=s_kv": 5,
                "s_q=random": 10,
                "s_q>s_kv": 3,
            },
        ),
        d_qk_d_v=RandomHiddenDimSize(
            d_qk_min=1,
            d_qk_max=256,
            d_v_min=1,
            d_v_max=256,
            head_dim_distribution={"d_qk=d_v": 1, "d_qk=random": 1},
            with_high_probability=[(64, 64), (128, 128), (192, 128), (256, 256)],
        ),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=_f16(),
        with_sliding_mask=SlidingWindowMaskGenerator(**SW_FULL),
        diag_align=RandomChoice(DIAG_BOTH),
        is_ragged_or_padded_or_full=RandomChoice({"padded": 1, "cu_padded": 1, "full": 1}),
        with_sink_token=RandomChoice({True: 1, False: 3}),
        is_bias=RandomChoice({True: 1, False: 5}),
    )


DENSE = SuiteSpec(
    name="context.f16.dense",
    phase="context",
    dtype="f16",
    level="L0",
    num_tests=512,
    rng_seed=888,
    knobs=dense_fwd,
    fuzzed=COMMON_FUZZ + MASK_FUZZ + ("layout padded/cu_padded/full", "sink", "bias(1:5)"),
    pinned=("infer",),
)


def thd_fwd():
    return dict(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1, 4]),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=1,
            s_q_max=8192,
            s_kv_min=1,
            s_kv_max=16384,
            s_q_distribution={
                "s_q=1": 0,
                "s_q=s_kv": 5,
                "s_q=random": 10,
                "s_q>s_kv": 3,
            },
        ),
        d_qk_d_v=RandomHiddenDimSize(
            d_qk_min=1,
            d_qk_max=256,
            d_v_min=1,
            d_v_max=256,
            head_dim_distribution={"d_qk=d_v": 1, "d_qk=random": 1},
            with_high_probability=[(64, 64), (128, 128), (192, 128), (256, 256)],
        ),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=_f16(),
        with_sliding_mask=SlidingWindowMaskGenerator(**SW_FULL),
        diag_align=RandomChoice(DIAG_BOTH),
        is_ragged_or_padded_or_full=RandomChoice({"ragged": 2, "cu_ragged": 1}),
        with_sink_token=RandomChoice({True: 1, False: 3}),
        ragged_stats_layout=RandomChoice({"token_major": 1, "head_major": 1}),
        total_token_slack=RandomChoice({"packed": 1, "slack": 1}),
        declare_total_seq_len=RandomChoice({True: 1, False: 1}),
    )


THD = SuiteSpec(
    name="context.f16.thd",
    phase="context",
    dtype="f16",
    level="L0",
    num_tests=768,
    rng_seed=890,
    knobs=thd_fwd,
    fuzzed=COMMON_FUZZ + MASK_FUZZ + THD_FUZZ + ("sink",),
    pinned=("infer", "layout THD (ragged/cu_ragged)"),
)


def dense_chunked():
    # Chunked prefill (context phase), dense layouts: query chunks up to 1k
    # tokens against a long KV history (up to 16k), per-batch lengths in
    # padded (B-entry) and cu_padded (B+1 cumulative) forms plus plain full.
    return dict(
        batches=RandomBatchSize(min=1, max=16, with_high_probability=[1, 4]),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=1,
            s_q_max=1024,
            s_kv_min=1,
            s_kv_max=16384,
            s_q_distribution={"s_q=1": 3, "s_q=s_kv": 1, "s_q=random": 10},
        ),
        d_qk_d_v=RandomHiddenDimSize(
            d_qk_min=1,
            d_qk_max=128,
            d_v_min=1,
            d_v_max=128,
            head_dim_distribution={"d_qk=d_v": 1, "d_qk=random": 1},
            with_high_probability=[(64, 64), (128, 128), (192, 128)],
        ),
        head_count=RandomHeadGenerator(min=1, max=32, head_group_options=(1, 4, 1)),
        data_type=_f16(),
        with_sliding_mask=SlidingWindowMaskGenerator(**SW_FULL),
        diag_align=RandomChoice(DIAG_BR_HEAVY),
        is_ragged_or_padded_or_full=RandomChoice({"padded": 2, "cu_padded": 1, "full": 1}),
    )


DENSE_CHUNKED = SuiteSpec(
    name="context.f16.dense_chunked",
    phase="context",
    dtype="f16",
    level="L0",
    num_tests=128,
    rng_seed=446,
    knobs=dense_chunked,
    fuzzed=COMMON_FUZZ + MASK_FUZZ + ("layout padded/cu_padded/full",),
    pinned=("infer", "s_q<=1024 chunks", "s_kv up to 16k"),
    notes="chunked prefill, dense: query chunks against a long KV history",
)


def thd_chunked():
    # Chunked prefill (context phase), packed THD: the serving shape — query
    # chunks up to 1k tokens packed varlen against a long KV history (16k),
    # seq-len form fuzzed ragged (B-entry) / cu_ragged (B+1 cumulative).
    return dict(
        batches=RandomBatchSize(min=1, max=16, with_high_probability=[1, 4]),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=1,
            s_q_max=1024,
            s_kv_min=1,
            s_kv_max=16384,
            s_q_distribution={"s_q=1": 3, "s_q=s_kv": 1, "s_q=random": 10},
        ),
        d_qk_d_v=RandomHiddenDimSize(
            d_qk_min=1,
            d_qk_max=128,
            d_v_min=1,
            d_v_max=128,
            head_dim_distribution={"d_qk=d_v": 1, "d_qk=random": 1},
            with_high_probability=[(64, 64), (128, 128), (192, 128)],
        ),
        head_count=RandomHeadGenerator(min=1, max=32, head_group_options=(1, 4, 1)),
        data_type=_f16(),
        with_sliding_mask=SlidingWindowMaskGenerator(**SW_FULL),
        diag_align=RandomChoice(DIAG_BR_HEAVY),
        is_ragged_or_padded_or_full=RandomChoice({"ragged": 2, "cu_ragged": 1}),
        ragged_stats_layout=RandomChoice({"token_major": 1, "head_major": 1}),
        total_token_slack=RandomChoice({"packed": 1, "slack": 1}),
        declare_total_seq_len=RandomChoice({True: 1, False: 1}),
    )


THD_CHUNKED = SuiteSpec(
    name="context.f16.thd_chunked",
    phase="context",
    dtype="f16",
    level="L0",
    num_tests=256,
    rng_seed=445,
    knobs=thd_chunked,
    fuzzed=COMMON_FUZZ + MASK_FUZZ + THD_FUZZ + ("layout ragged/cu_ragged",),
    pinned=("infer", "s_q<=1024 chunks", "s_kv up to 16k", "layout THD"),
    notes="chunked prefill, packed THD: the serving shape",
)

SUITES = [DENSE, THD, DENSE_CHUNKED, THD_CHUNKED]


@pytest.mark.L0
@pytest.mark.parametrize("test_no", DENSE.seeds(), ids=lambda p: f"test{p[0]}")
def test_context_f16_dense(env_info, test_no, request, cudnn_handle):
    run_suite(DENSE, env_info, test_no, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize("test_no", THD.seeds(), ids=lambda p: f"test{p[0]}")
def test_context_f16_thd(env_info, test_no, request, cudnn_handle):
    run_suite(THD, env_info, test_no, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize("test_no", DENSE_CHUNKED.seeds(), ids=lambda p: f"test{p[0]}")
def test_context_f16_dense_chunked(env_info, test_no, request, cudnn_handle):
    run_suite(DENSE_CHUNKED, env_info, test_no, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize("test_no", THD_CHUNKED.seeds(), ids=lambda p: f"test{p[0]}")
def test_context_f16_thd_chunked(env_info, test_no, request, cudnn_handle):
    run_suite(THD_CHUNKED, env_info, test_no, request, cudnn_handle)


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
def test_context_mixed_seq_len_forms(env_info, cu_sides, diag_align, right_bound, request, cudnn_handle):
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
