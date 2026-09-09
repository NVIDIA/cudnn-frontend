# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context (prefill forward) suites, mxfp8 (SM100+)."""

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
    Fixed,
    MASK_FUZZ,
    SuiteSpec,
    post_mxfp8,
    run_suite,
)
from sdpa.suites.knobs import (
    DIAG_BOTH,
    DIAG_BR_HEAVY,
    SW_FULL,
)


def mxfp8_fwd():
    return dict(
        batches=RandomBatchSize(min=1, max=4),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=128,
            s_q_max=8192,
            s_kv_min=128,
            s_kv_max=16384,
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
        data_type=RandomChoice({torch.float8_e4m3fn: 3, torch.float8_e5m2: 1}),
        output_type=RandomChoice({torch.float16: 2, torch.bfloat16: 1}),
        with_sliding_mask=SlidingWindowMaskGenerator(**SW_FULL),
        diag_align=RandomChoice(DIAG_BOTH),
        # Full-only: the sdpa_mxfp8 python API has no seq_len/padding
        # arguments and exec_sdpa_mxfp8 never reads cfg.seq_len_q/kv, so a
        # "padded" draw would silently run dense-full (see GitHub #646).
        # Re-add padded/ragged once the API grows seq-len support.
        is_ragged_or_padded_or_full=RandomChoice({"full": 1}),
        with_sink_token=RandomChoice({True: 1, False: 2}),
    )


DENSE = SuiteSpec(
    name="context.mxfp8.dense",
    phase="context",
    dtype="mxfp8",
    level="L0",
    num_tests=512,
    rng_seed=1001,
    knobs=mxfp8_fwd,
    exec_kind="mxfp8",
    min_sm=(10, 0),
    post=post_mxfp8,
    fuzzed=COMMON_FUZZ
    + MASK_FUZZ
    + (
        "e4m3/e5m2 in",
        "out fp16/bf16",
        "sink",
    ),
    pinned=(
        "infer",
        "SM100+",
        "layout full (mxfp8 API has no seq-len args, #646)",
    ),
)


def mxfp8_thd_fwd():
    # Forward THD/ragged mxfp8 (SM100+): packed tokens + ragged offsets +
    # packed per-sequence-TILE-padded SF (engine contract from
    # frost/test_sdpa_fwd_mxfp8_sm100.py). Causal / no-mask, TL alignment,
    # token-major stats — the validated THD mxfp8 envelope.
    return dict(
        batches=RandomBatchSize(min=1, max=4, with_high_probability=[1, 2]),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=128,
            s_q_max=2048,
            s_kv_min=128,
            s_kv_max=2048,
            s_q_distribution={"s_q=1": 0, "s_q=s_kv": 5, "s_q=random": 5},
        ),
        # The FROST mxfp8 prefill engine's THD leg is d=128/128 only
        # (thd_d_shapes; the d192x128 kernel is dense-only) — any other d
        # declines to the native backend, which cannot run THD mxfp8.
        d_qk_d_v=Fixed((128, 128)),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float8_e4m3fn: 3, torch.float8_e5m2: 1}),
        output_type=RandomChoice({torch.float16: 2, torch.bfloat16: 1}),
        with_sliding_mask=SlidingWindowMaskGenerator(**SW_FULL),
        diag_align=RandomChoice(DIAG_BR_HEAVY),
        is_ragged_or_padded_or_full=RandomChoice({"ragged": 2, "cu_ragged": 1}),
        with_sink_token=RandomChoice({True: 1, False: 2}),
        total_token_slack=RandomChoice({"packed": 1, "slack": 1}),
        declare_total_seq_len=RandomChoice({True: 1, False: 1}),
    )


THD = SuiteSpec(
    name="context.mxfp8.thd",
    phase="context",
    dtype="mxfp8",
    level="L0",
    num_tests=192,
    rng_seed=1003,
    knobs=mxfp8_thd_fwd,
    exec_kind="mxfp8",
    min_sm=(10, 0),
    post=post_mxfp8,
    fuzzed=COMMON_FUZZ
    + MASK_FUZZ
    + (
        "e4m3/e5m2 in",
        "out fp16/bf16",
        "layout ragged/cu_ragged",
        "sink",
        "total_q/kv slack",
        "declare totals on graph",
    ),
    pinned=(
        "infer",
        "stats token-major TH1",
        "d=128/128 (frost THD leg)",
        "SM100+",
    ),
    notes="diag BR-weighted 2:1 (production context alignment); "
    "fwd only (no THD mxfp8 bwd engine); needs opt-in FROST engine "
    "(CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1) — skips otherwise: the native "
    "backend check_support-accepts THD mxfp8 but cannot execute it",
)

SUITES = [DENSE, THD]


@pytest.mark.L0
@pytest.mark.parametrize("test_no", DENSE.seeds(), ids=lambda p: f"test{p[0]}")
def test_context_mxfp8_dense(env_info, test_no, request, cudnn_handle):
    run_suite(DENSE, env_info, test_no, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize("test_no", THD.seeds(), ids=lambda p: f"test{p[0]}")
def test_context_mxfp8_thd(env_info, test_no, request, cudnn_handle):
    run_suite(THD, env_info, test_no, request, cudnn_handle)
