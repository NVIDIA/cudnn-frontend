# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context (prefill forward) suites, fp8 (e4m3/e5m2)."""

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
    run_suite,
)
from sdpa.suites.knobs import (
    DIAG_BOTH,
    DIAG_BR_HEAVY,
    SW_FULL,
)


def fp8_fwd():
    return dict(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[4]),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=1,
            s_q_max=8192,
            s_kv_min=1,
            s_kv_max=16384,
            s_q_distribution={"s_q=1": 2, "s_q=s_kv": 5, "s_q=random": 2},
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
        with_sliding_mask=SlidingWindowMaskGenerator(**SW_FULL),
        diag_align=RandomChoice(DIAG_BOTH),
        is_ragged_or_padded_or_full=RandomChoice({"padded": 1, "full": 1}),
        with_sink_token=RandomChoice({True: 1, False: 2}),
    )


DENSE = SuiteSpec(
    name="context.fp8.dense",
    phase="context",
    dtype="fp8",
    level="L0",
    num_tests=512,
    rng_seed=999,
    knobs=fp8_fwd,
    exec_kind="fp8",
    fuzzed=COMMON_FUZZ
    + MASK_FUZZ
    + (
        "e4m3/e5m2 in",
        "out fp8/fp16",
        "layout padded/full",
        "sink",
    ),
    pinned=("infer",),
)


def fp8_thd_fwd():
    return dict(
        batches=RandomBatchSize(min=1, max=4, with_high_probability=[1, 2]),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=64,
            s_q_max=8192,
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
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float8_e4m3fn: 2, torch.float8_e5m2: 1}),
        output_type=RandomChoice({torch.float8_e4m3fn: 1, torch.float8_e5m2: 1, torch.float16: 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(**SW_FULL),
        diag_align=RandomChoice(DIAG_BR_HEAVY),
        is_ragged_or_padded_or_full=RandomChoice({"ragged": 2, "cu_ragged": 1}),
        with_sink_token=RandomChoice({True: 1, False: 2}),
        total_token_slack=RandomChoice({"packed": 1, "slack": 1}),
        declare_total_seq_len=RandomChoice({True: 1, False: 1}),
    )


THD = SuiteSpec(
    name="context.fp8.thd",
    phase="context",
    dtype="fp8",
    level="L0",
    num_tests=512,
    rng_seed=996,
    knobs=fp8_thd_fwd,
    exec_kind="fp8",
    fuzzed=COMMON_FUZZ
    + MASK_FUZZ
    + (
        "e4m3/e5m2 in",
        "out fp8/fp16",
        "layout ragged/cu_ragged",
        "sink",
        "total_q/kv slack",
        "declare totals on graph",
    ),
    pinned=("infer",),
    notes="diag BR-weighted 2:1 — production context-phase alignment",
)


def fp8_dense_chunked():
    # fp8 chunked prefill, dense layouts (padded/full). Dense fp8 supports
    # d_qk=192 (no ragged-hang envelope here).
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
            d_qk_min=64,
            d_qk_max=192,
            d_v_min=64,
            d_v_max=128,
            head_dim_distribution={"d_qk=d_v": 2, "d_qk=random": 1},
            with_high_probability=[(64, 64), (128, 128), (192, 128)],
        ),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float8_e4m3fn: 2, torch.float8_e5m2: 1}),
        output_type=RandomChoice({torch.float8_e4m3fn: 1, torch.float8_e5m2: 1, torch.float16: 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(**SW_FULL),
        diag_align=RandomChoice(DIAG_BR_HEAVY),
        is_ragged_or_padded_or_full=RandomChoice({"padded": 2, "full": 1}),
        with_sink_token=RandomChoice({True: 1, False: 2}),
    )


DENSE_CHUNKED = SuiteSpec(
    name="context.fp8.dense_chunked",
    phase="context",
    dtype="fp8",
    level="L0",
    num_tests=128,
    rng_seed=992,
    knobs=fp8_dense_chunked,
    exec_kind="fp8",
    fuzzed=COMMON_FUZZ
    + MASK_FUZZ
    + (
        "e4m3/e5m2 in",
        "out fp8/fp16",
        "layout padded/full",
        "sink",
    ),
    pinned=("infer", "s_q<=1024 chunks", "s_kv up to 16k"),
    notes="chunked prefill, fp8 dense (d up to 192 — no ragged-hang envelope)",
)


def fp8_thd_chunked():
    # fp8 chunked prefill, packed THD, seq-len form ragged/cu_ragged. d capped
    # at 128: fp8 ragged THD with d_qk > 128 hangs the backend kernel (see
    # model_knobs_fp8).
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
            d_qk_min=64,
            d_qk_max=128,
            d_v_min=64,
            d_v_max=128,
            head_dim_distribution={"d_qk=d_v": 1, "d_qk=random": 0},
            with_high_probability=[(64, 64), (128, 128)],
        ),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float8_e4m3fn: 2, torch.float8_e5m2: 1}),
        output_type=RandomChoice({torch.float8_e4m3fn: 1, torch.float8_e5m2: 1, torch.float16: 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(**SW_FULL),
        diag_align=RandomChoice(DIAG_BR_HEAVY),
        is_ragged_or_padded_or_full=RandomChoice({"ragged": 2, "cu_ragged": 1}),
        with_sink_token=RandomChoice({True: 1, False: 2}),
        total_token_slack=RandomChoice({"packed": 1, "slack": 1}),
        declare_total_seq_len=RandomChoice({True: 1, False: 1}),
    )


THD_CHUNKED = SuiteSpec(
    name="context.fp8.thd_chunked",
    phase="context",
    dtype="fp8",
    level="L0",
    num_tests=192,
    rng_seed=991,
    knobs=fp8_thd_chunked,
    exec_kind="fp8",
    fuzzed=COMMON_FUZZ
    + MASK_FUZZ
    + (
        "e4m3/e5m2 in",
        "out fp8/fp16",
        "layout ragged/cu_ragged",
        "sink",
        "capacity slack",
        "declared totals",
    ),
    pinned=("infer", "s_q<=1024 chunks", "s_kv up to 16k", "layout THD", "d<=128 (fp8 ragged d>128 hangs the backend)"),
    notes="chunked prefill, fp8 packed THD",
)

SUITES = [DENSE, THD, DENSE_CHUNKED, THD_CHUNKED]


@pytest.mark.L0
@pytest.mark.parametrize("test_no", DENSE.seeds(), ids=lambda p: f"test{p[0]}")
def test_context_fp8_dense(env_info, test_no, request, cudnn_handle):
    run_suite(DENSE, env_info, test_no, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize("test_no", THD.seeds(), ids=lambda p: f"test{p[0]}")
def test_context_fp8_thd(env_info, test_no, request, cudnn_handle):
    run_suite(THD, env_info, test_no, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize("test_no", DENSE_CHUNKED.seeds(), ids=lambda p: f"test{p[0]}")
def test_context_fp8_dense_chunked(env_info, test_no, request, cudnn_handle):
    run_suite(DENSE_CHUNKED, env_info, test_no, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize("test_no", THD_CHUNKED.seeds(), ids=lambda p: f"test{p[0]}")
def test_context_fp8_thd_chunked(env_info, test_no, request, cudnn_handle):
    run_suite(THD_CHUNKED, env_info, test_no, request, cudnn_handle)
