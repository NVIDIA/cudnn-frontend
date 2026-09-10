# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bprop (training fwd+bwd) suites, fp8 (e4m3 in; see issue #955 for
the QKV-vs-dO dtype-mix extension)."""

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
from sdpa.suites.common import (
    COMMON_FUZZ,
    MASK_FUZZ,
    SuiteSpec,
    post_train,
    run_suite,
)
from sdpa.suites.knobs import (
    DIAG_BOTH,
    SW_FULL,
)


def fp8_bwd():
    return dict(
        batches=RandomBatchSize(min=1, max=4, with_high_probability=[1, 2]),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=64,
            s_q_max=8192,
            s_kv_min=64,
            s_kv_max=8192,
            s_q_distribution={"s_q=1": 0, "s_q=s_kv": 5, "s_q=random": 5},
        ),
        d_qk_d_v=RandomHiddenDimSize(
            d_qk_min=64,
            d_qk_max=192,
            d_v_min=64,
            d_v_max=128,
            head_dim_distribution={"d_qk=d_v": 1, "d_qk=random": 0},
            with_high_probability=[(64, 64), (128, 128), (192, 128)],
        ),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float8_e4m3fn: 1}),
        output_type=RandomChoice({torch.float8_e4m3fn: 1, torch.float16: 1}),
        with_sliding_mask=SlidingWindowMaskGenerator(**SW_FULL),
        diag_align=RandomChoice(DIAG_BOTH),
        is_ragged_or_padded_or_full=RandomChoice({"full": 1}),
        is_deterministic=RandomChoice({True: 1, False: 1}),
        with_sink_token=RandomChoice({True: 1, False: 2}),
    )


DENSE = SuiteSpec(
    name="bprop.fp8.dense",
    phase="bprop",
    dtype="fp8",
    level="L0",
    num_tests=384,
    rng_seed=998,
    knobs=fp8_bwd,
    exec_kind="fp8",
    post=post_train,
    fuzzed=COMMON_FUZZ + MASK_FUZZ + ("out fp8/fp16", "deterministic", "sink"),
    pinned=("train", "e4m3 in", "layout full"),
)


def fp8_thd_bwd():
    return dict(
        batches=RandomBatchSize(min=1, max=4, with_high_probability=[1, 2]),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=64,
            s_q_max=8192,
            s_kv_min=64,
            s_kv_max=8192,
            s_q_distribution={"s_q=1": 0, "s_q=s_kv": 5, "s_q=random": 5},
        ),
        d_qk_d_v=RandomHiddenDimSize(
            d_qk_min=64,
            d_qk_max=128,
            d_v_min=64,
            d_v_max=128,
            head_dim_distribution={"d_qk=d_v": 1, "d_qk=random": 0},
            with_high_probability=[(64, 64), (128, 128)],
        ),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float8_e4m3fn: 1}),
        output_type=RandomChoice({torch.float8_e4m3fn: 1, torch.float16: 1}),
        with_sliding_mask=SlidingWindowMaskGenerator(**SW_FULL),
        diag_align=RandomChoice(DIAG_BOTH),
        is_ragged_or_padded_or_full=RandomChoice({"ragged": 1}),
        is_deterministic=RandomChoice({True: 1, False: 1}),
        with_sink_token=RandomChoice({True: 1, False: 2}),
        total_token_slack=RandomChoice({"packed": 1, "slack": 1}),
    )


THD = SuiteSpec(
    name="bprop.fp8.thd",
    phase="bprop",
    dtype="fp8",
    level="L0",
    num_tests=384,
    rng_seed=995,
    knobs=fp8_thd_bwd,
    exec_kind="fp8",
    post=post_train,
    fuzzed=COMMON_FUZZ + MASK_FUZZ + ("out fp8/fp16", "deterministic", "sink", "total_q/kv slack"),
    pinned=(
        "train",
        "e4m3 in",
        "layout THD (ragged)",
    ),
    notes="ragged FP8 backward requires cuDNN > 9.21.0",
)

SUITES = [DENSE, THD]


@pytest.mark.L0
@pytest.mark.parametrize("test_no", DENSE.seeds(), ids=lambda p: f"test{p[0]}")
def test_bprop_fp8_dense(env_info, test_no, request, cudnn_handle):
    run_suite(DENSE, env_info, test_no, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize("test_no", THD.seeds(), ids=lambda p: f"test{p[0]}")
def test_bprop_fp8_thd(env_info, test_no, request, cudnn_handle):
    run_suite(THD, env_info, test_no, request, cudnn_handle)
