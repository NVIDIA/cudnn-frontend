# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generation (decode / paged) suites, fp8 (e4m3/e5m2)."""

import torch
import pytest
from sdpa.random_config import (
    RandomBatchSize,
    RandomBlockSize,
    RandomChoice,
    RandomHeadGenerator,
    RandomHiddenDimSize,
    RandomSequenceLength,
    SlidingWindowMaskGenerator,
)
from sdpa.suites.common import (
    COMMON_FUZZ,
    SuiteSpec,
    post_paged,
    run_suite,
)
from sdpa.suites.knobs import (
    DIAG_BOTH,
    DIAG_TL,
    SW_NONE,
)


def fp8_decode():
    return dict(
        batches=RandomBatchSize(min=1, max=16, with_high_probability=[1, 4]),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=1,
            s_q_max=1,
            s_kv_min=1,
            s_kv_max=16384,
            s_q_distribution={"s_q=1": 100, "s_q=s_kv": 1, "s_q=random": 0},
        ),
        d_qk_d_v=RandomHiddenDimSize(
            d_qk_min=64,
            d_qk_max=192,
            d_v_min=64,
            d_v_max=128,
            head_dim_distribution={"d_qk=d_v": 2, "d_qk=random": 1},
            with_high_probability=[(64, 64), (128, 128), (192, 128)],
        ),
        head_count=RandomHeadGenerator(min=1, max=16, head_group_options=(1, 5, 2)),
        data_type=RandomChoice({torch.float8_e4m3fn: 2, torch.float8_e5m2: 1}),
        output_type=RandomChoice({torch.float8_e4m3fn: 1, torch.float8_e5m2: 1, torch.float16: 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(**SW_NONE),
        diag_align=RandomChoice(DIAG_BOTH),
        # padded + full: the split-KV / lean regime with per-batch lengths
        # (this suite absorbed the former generation.fp8.lean).
        is_ragged_or_padded_or_full=RandomChoice({"padded": 1, "full": 1}),
    )


DECODE = SuiteSpec(
    name="generation.fp8.decode",
    phase="generation",
    dtype="fp8",
    level="L0",
    num_tests=256,
    rng_seed=993,
    knobs=fp8_decode,
    exec_kind="fp8",
    fuzzed=COMMON_FUZZ
    + (
        "e4m3/e5m2 in",
        "out fp8/fp16",
        "diag TL/BR",
        "layout padded/full",
    ),
    pinned=("infer", "s_q=1", "s_kv up to 16k (split-KV regime)", "no mask"),
    notes="absorbed generation.fp8.lean: long-KV split-KV draws (padded + full) are routine here",
)


def fp8_paged():
    return dict(
        batches=RandomBatchSize(min=1, max=4, with_high_probability=[1, 2]),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=64,
            s_q_max=256,
            s_kv_min=64,
            s_kv_max=16384,
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
        head_count=RandomHeadGenerator(min=1, max=4, head_group_options=(1, 2, 0)),
        data_type=RandomChoice({torch.float8_e4m3fn: 2, torch.float8_e5m2: 1}),
        output_type=RandomChoice({torch.float8_e4m3fn: 1, torch.float8_e5m2: 1, torch.float16: 1}),
        with_sliding_mask=SlidingWindowMaskGenerator(**SW_NONE),
        diag_align=RandomChoice(DIAG_TL),
        is_ragged_or_padded_or_full=RandomChoice({"padded": 1}),
        block_size=RandomBlockSize(min=16, max=128, with_high_probability=[16, 32, 64]),
    )


PAGED = SuiteSpec(
    name="generation.fp8.paged",
    phase="generation",
    dtype="fp8",
    level="L0",
    num_tests=128,
    rng_seed=997,
    knobs=fp8_paged,
    exec_kind="fp8",
    post=post_paged,
    fuzzed=COMMON_FUZZ
    + (
        "e4m3/e5m2 in",
        "out fp8/fp16",
        "block size 16..128",
    ),
    pinned=("infer", "no mask", "diag TL", "layout padded", "paged KV"),
)

SUITES = [DECODE, PAGED]


@pytest.mark.L0
@pytest.mark.parametrize("test_no", DECODE.seeds(), ids=lambda p: f"test{p[0]}")
def test_generation_fp8_decode(env_info, test_no, request, cudnn_handle):
    run_suite(DECODE, env_info, test_no, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize("test_no", PAGED.seeds(), ids=lambda p: f"test{p[0]}")
def test_generation_fp8_paged(env_info, test_no, request, cudnn_handle):
    run_suite(PAGED, env_info, test_no, request, cudnn_handle)
