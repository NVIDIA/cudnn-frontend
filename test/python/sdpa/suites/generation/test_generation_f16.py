# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generation (decode / paged) suites, f16."""

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
    MASK_FUZZ,
    SuiteSpec,
    post_paged,
    run_suite,
)
from sdpa.suites.knobs import (
    DIAG_BOTH,
    SW_FULL,
    SW_NONE,
    _f16,
)


def decode():
    # s_q == 1 generation step against a long KV history (up to 16k, so the
    # split-KV / lean regime is drawn routinely — this suite absorbed the
    # former generation.*.lean suites), dense-full and padded layouts.
    return dict(
        batches=RandomBatchSize(min=1, max=32),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=1,
            s_q_max=1,
            s_kv_min=1,
            s_kv_max=16384,
            s_q_distribution={"s_q=1": 100, "s_q=s_kv": 1, "s_q=random": 0},
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
        with_sliding_mask=SlidingWindowMaskGenerator(**SW_NONE),
        diag_align=RandomChoice(DIAG_BOTH),
        is_ragged_or_padded_or_full=RandomChoice({"padded": 1, "full": 1}),
        # sink_token / dropout not supported with s_q == 1
    )


DECODE = SuiteSpec(
    name="generation.f16.decode",
    phase="generation",
    dtype="f16",
    level="L0",
    num_tests=384,
    rng_seed=111,
    knobs=decode,
    fuzzed=COMMON_FUZZ + ("diag TL/BR", "layout padded/full"),
    pinned=("infer", "s_q=1", "s_kv up to 16k (split-KV regime)", "no mask"),
    notes="absorbed generation.f16.lean: long-KV split-KV draws are routine here",
)


def paged():
    # Chunked generation (s_q <= 64) against a paged KV cache.
    return dict(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1, 4]),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=1,
            s_q_max=64,
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
            d_qk_max=128,
            d_v_min=1,
            d_v_max=128,
            head_dim_distribution={"d_qk=d_v": 1, "d_qk=random": 1},
            with_high_probability=[(64, 64), (128, 128), (192, 128)],
        ),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=_f16(),
        with_sliding_mask=SlidingWindowMaskGenerator(**SW_FULL),
        diag_align=RandomChoice(DIAG_BOTH),
        # paged + ragged (packed THD Q/O against a paged KV cache) is a valid
        # serving combo but deferred: harness support only existed for f16 —
        # tracked as a suite-wide extension (f16 + fp8) in the issue tracker.
        is_ragged_or_padded_or_full=RandomChoice({"padded": 2, "cu_padded": 1}),
        block_size=RandomBlockSize(min=1, max=1024, with_high_probability=[1, 32, 128]),
        with_sink_token=RandomChoice({True: 1, False: 3}),
    )


PAGED = SuiteSpec(
    name="generation.f16.paged",
    phase="generation",
    dtype="f16",
    level="L0",
    num_tests=384,
    rng_seed=887,
    knobs=paged,
    post=post_paged,
    fuzzed=COMMON_FUZZ + MASK_FUZZ + ("layout padded/cu_padded", "block size 1..1024", "sink"),
    pinned=("infer", "s_q<=64", "layout padded", "paged KV"),
)

SUITES = [DECODE, PAGED]


@pytest.mark.L0
@pytest.mark.parametrize("test_no", DECODE.seeds(), ids=lambda p: f"test{p[0]}")
def test_generation_f16_decode(env_info, test_no, request, cudnn_handle):
    run_suite(DECODE, env_info, test_no, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize("test_no", PAGED.seeds(), ids=lambda p: f"test{p[0]}")
def test_generation_f16_paged(env_info, test_no, request, cudnn_handle):
    run_suite(PAGED, env_info, test_no, request, cudnn_handle)
