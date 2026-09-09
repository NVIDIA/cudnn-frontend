# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bprop (training fwd+bwd) suites, mxfp8 (SM100+)."""

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
    combine,
    post_mxfp8,
    post_mxfp8_bwd_flags,
    post_train,
    run_suite,
)
from sdpa.suites.knobs import (
    DIAG_BOTH,
    SW_FULL,
)


def mxfp8_bwd():
    return dict(
        batches=RandomBatchSize(min=1, max=4),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=256,
            s_q_max=8192,
            s_kv_min=256,
            s_kv_max=8192,
            s_q_distribution={"s_q=1": 0, "s_q=s_kv": 1, "s_q=random": 1},
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
        data_type=RandomChoice({torch.float8_e4m3fn: 2}),
        output_type=RandomChoice({torch.float16: 2, torch.bfloat16: 1}),
        with_sliding_mask=SlidingWindowMaskGenerator(**SW_FULL),
        diag_align=RandomChoice(DIAG_BOTH),
        is_ragged_or_padded_or_full=RandomChoice({"full": 1}),
        is_deterministic=RandomChoice({True: 1}),
        with_sink_token=RandomChoice({True: 1, False: 2}),
    )


DENSE = SuiteSpec(
    name="bprop.mxfp8.dense",
    phase="bprop",
    dtype="mxfp8",
    level="L0",
    num_tests=384,
    rng_seed=1002,
    knobs=mxfp8_bwd,
    exec_kind="mxfp8",
    min_sm=(10, 0),
    post=combine(post_mxfp8, post_mxfp8_bwd_flags, post_train),
    fuzzed=COMMON_FUZZ + MASK_FUZZ + ("out fp16/bf16", "sink"),
    pinned=(
        "train",
        "e4m3 in",
        "deterministic",
        "layout full",
        "SM100+",
    ),
)

SUITES = [DENSE]


@pytest.mark.L0
@pytest.mark.parametrize("test_no", DENSE.seeds(), ids=lambda p: f"test{p[0]}")
def test_bprop_mxfp8_dense(env_info, test_no, request, cudnn_handle):
    run_suite(DENSE, env_info, test_no, request, cudnn_handle)
