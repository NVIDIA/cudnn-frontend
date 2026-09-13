# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bprop (training fwd+bwd) suites, f16."""

import pytest
from sdpa.random_config import (
    RandomBatchSize,
    RandomChoice,
    RandomHeadGenerator,
    RandomHiddenDimSize,
    RandomSequenceLength,
    SlidingWindowMaskGenerator,
)
from sdpa.suites.common import (
    COMMON_FUZZ,
    MASK_FUZZ,
    SuiteSpec,
    THD_FUZZ,
    post_train,
    run_suite,
)
from sdpa.suites.knobs import (
    DIAG_BOTH,
    SW_FULL,
    _f16,
)


def dense_bwd():
    return dict(
        batches=RandomBatchSize(min=8, max=16),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=1,
            s_q_max=8192,
            s_kv_min=1,
            s_kv_max=8192,
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
            head_dim_distribution={"d_qk=d_v": 5, "d_qk=random": 1},
            with_high_probability=[(64, 64), (128, 128), (192, 128), (256, 256)],
        ),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=_f16(),
        with_sliding_mask=SlidingWindowMaskGenerator(**SW_FULL),
        diag_align=RandomChoice(DIAG_BOTH),
        is_ragged_or_padded_or_full=RandomChoice({"ragged": 0, "padded": 4, "full": 1}),
        is_deterministic=RandomChoice({True: 3, False: 1}),
        with_sink_token=RandomChoice({True: 1, False: 3}),
        is_bias=RandomChoice({True: 1, False: 7}),
    )


DENSE = SuiteSpec(
    name="bprop.f16.dense",
    phase="bprop",
    dtype="f16",
    level="L0",
    num_tests=512,
    rng_seed=844,
    knobs=dense_bwd,
    post=post_train,
    fuzzed=COMMON_FUZZ + MASK_FUZZ + ("layout padded/full", "deterministic", "sink", "bias(1:7)"),
    pinned=("train"),
)


def thd_bwd():
    return dict(
        batches=RandomBatchSize(min=8, max=16),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=1,
            s_q_max=8192,
            s_kv_min=1,
            s_kv_max=8192,
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
            head_dim_distribution={"d_qk=d_v": 5, "d_qk=random": 1},
            with_high_probability=[(64, 64), (128, 128), (192, 128), (256, 256)],
        ),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=_f16(),
        with_sliding_mask=SlidingWindowMaskGenerator(**SW_FULL),
        diag_align=RandomChoice(DIAG_BOTH),
        is_ragged_or_padded_or_full=RandomChoice({"ragged": 1}),
        is_deterministic=RandomChoice({True: 3, False: 1}),
        ragged_stats_layout=RandomChoice({"token_major": 1, "head_major": 1}),
        with_sink_token=RandomChoice({True: 1, False: 3}),
        total_token_slack=RandomChoice({"packed": 1, "slack": 1}),
        declare_total_seq_len=RandomChoice({True: 1, False: 1}),
    )


THD = SuiteSpec(
    name="bprop.f16.thd",
    phase="bprop",
    dtype="f16",
    level="L0",
    num_tests=768,
    rng_seed=845,
    knobs=thd_bwd,
    post=post_train,
    fuzzed=COMMON_FUZZ + MASK_FUZZ + THD_FUZZ + ("deterministic", "sink"),
    pinned=("train", "layout THD (ragged)"),
)

SUITES = [DENSE, THD]


@pytest.mark.L0
@pytest.mark.parametrize("test_no", DENSE.seeds(), ids=lambda p: f"test{p[0]}")
def test_bprop_f16_dense(env_info, test_no, request, cudnn_handle):
    run_suite(DENSE, env_info, test_no, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize("test_no", THD.seeds(), ids=lambda p: f"test{p[0]}")
def test_bprop_f16_thd(env_info, test_no, request, cudnn_handle):
    run_suite(THD, env_info, test_no, request, cudnn_handle)
