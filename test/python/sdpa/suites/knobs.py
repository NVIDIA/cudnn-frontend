# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named knob-set factories for the sdpa/suites framework.

Each factory returns the kwargs for a RandomizationContext. Conventions:

  - 16-bit is ONE family: f16 suites draw torch.float16 or torch.bfloat16
    per config (data_type fuzz), exactly like the fp8 suites draw e4m3/e5m2.
  - the sliding-window / causal mask flavor is a fuzz axis of every knob set
    that supports it (there is no separate mask suite).
  - bias is a low-weight fuzz axis of the dense knob sets (no separate bias
    suite); unsupported combinations waive through graph check_support.
  - THD knob sets fuzz the first-class packed capacities: total_token_slack
    widens total_q/total_kv beyond the packed minimum, declare_total_seq_len
    fuzzes declaring them on the forward graph.
"""

import torch
import cudnn

from sdpa.random_config import (
    RandomBatchSize,
    RandomBlockSize,
    RandomChoice,
    RandomHeadGenerator,
    RandomHiddenDimSize,
    RandomSequenceLength,
    SlidingWindowMaskGenerator,
)
from sdpa.suites.common import Fixed

# Mask-flavor fuzz used by every suite that supports masks: causal, one-sided
# windows, band around the diagonal, and no mask.
SW_FULL = dict(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10)
SW_NONE = dict(no_mask=10)

DIAG_BOTH = {
    cudnn.diagonal_alignment.TOP_LEFT: 1,
    cudnn.diagonal_alignment.BOTTOM_RIGHT: 1,
}
DIAG_TL = {cudnn.diagonal_alignment.TOP_LEFT: 1}
# Production context-phase weighting: BOTTOM_RIGHT is the alignment real
# serving stacks use, so THD context suites draw it more often.
DIAG_BR_HEAVY = {
    cudnn.diagonal_alignment.BOTTOM_RIGHT: 2,
    cudnn.diagonal_alignment.TOP_LEFT: 1,
}


def _f16():
    # One 16-bit family: each config draws fp16 or bf16, like fp8 draws
    # e4m3/e5m2.
    return RandomChoice({torch.float16: 1, torch.bfloat16: 1})


# ---- fp16 / bf16 -----------------------------------------------------------


def dense_fwd():
    return dict(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1, 4]),
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


def thd_fwd():
    return dict(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1, 4]),
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


def decode():
    # s_q == 1 generation step against a long KV history.
    return dict(
        batches=RandomBatchSize(min=1, max=32),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=1,
            s_q_max=1,
            s_kv_min=1,
            s_kv_max=8192,
            s_q_distribution={"s_q=1": 100, "s_q=s_kv": 1, "s_q=random": 0},
        ),
        d_qk_d_v=RandomHiddenDimSize(
            d_qk_min=1,
            d_qk_max=128,
            d_v_min=1,
            d_v_max=128,
            head_dim_distribution={"d_qk=d_v": 1, "d_qk=random": 1},
            with_high_probability=[(128, 128), (192, 128)],
        ),
        head_count=RandomHeadGenerator(min=1, max=32, head_group_options=(1, 4, 1)),
        data_type=_f16(),
        with_sliding_mask=SlidingWindowMaskGenerator(**SW_NONE),
        diag_align=RandomChoice(DIAG_BOTH),
        is_ragged_or_padded_or_full=RandomChoice({"full": 1}),
        # sink_token / dropout not supported with s_q == 1
    )


def lean_attn():
    # Decode against a long KV (513..8192): the lean-attention split regime.
    return dict(
        batches=RandomBatchSize(min=1, max=32),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=1,
            s_q_max=1,
            s_kv_min=513,
            s_kv_max=8192,
            s_q_distribution={"s_q=1": 100, "s_q=s_kv": 0, "s_q=random": 0},
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
    )


def paged():
    # Chunked generation (s_q <= 64) against a paged KV cache.
    return dict(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1, 4]),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=1,
            s_q_max=64,
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
        is_ragged_or_padded_or_full=RandomChoice({"padded": 2, "cu_padded": 1}),
        block_size=RandomBlockSize(min=1, max=1024, with_high_probability=[1, 32, 128]),
        with_sink_token=RandomChoice({True: 1, False: 3}),
    )


def fp8_fwd():
    return dict(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[4]),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=1,
            s_q_max=8192,
            s_kv_min=1,
            s_kv_max=8192,
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


def fp8_decode():
    return dict(
        batches=RandomBatchSize(min=1, max=16, with_high_probability=[1, 4]),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=1,
            s_q_max=1,
            s_kv_min=1,
            s_kv_max=8192,
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
        is_ragged_or_padded_or_full=RandomChoice({"full": 1}),
    )


def fp8_thd_fwd():
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
        data_type=RandomChoice({torch.float8_e4m3fn: 2, torch.float8_e5m2: 1}),
        output_type=RandomChoice({torch.float8_e4m3fn: 1, torch.float8_e5m2: 1, torch.float16: 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(**SW_FULL),
        diag_align=RandomChoice(DIAG_BR_HEAVY),
        is_ragged_or_padded_or_full=RandomChoice({"ragged": 2, "cu_ragged": 1}),
        with_sink_token=RandomChoice({True: 1, False: 2}),
        total_token_slack=RandomChoice({"packed": 1, "slack": 1}),
        declare_total_seq_len=RandomChoice({True: 1, False: 1}),
    )


def fp8_paged():
    return dict(
        batches=RandomBatchSize(min=1, max=4, with_high_probability=[1, 2]),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=64,
            s_q_max=256,
            s_kv_min=64,
            s_kv_max=512,
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


# ---- mxfp8 -----------------------------------------------------------------


def mxfp8_fwd():
    return dict(
        batches=RandomBatchSize(min=1, max=4),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=128,
            s_q_max=8192,
            s_kv_min=128,
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


def thd_chunked():
    # Ragged (THD) chunked generation: short query chunks (s_q <= 64) against a
    # long KV history, packed varlen on both sides — the varlen continuation /
    # chunked-prefill shape. Not covered by test_mhas_v2 (its THD suites are
    # prefill-sized, its decode suites dense-only).
    return dict(
        batches=RandomBatchSize(min=1, max=16, with_high_probability=[1, 4]),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=1,
            s_q_max=64,
            s_kv_min=1,
            s_kv_max=8192,
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
        diag_align=RandomChoice(DIAG_BOTH),
        is_ragged_or_padded_or_full=RandomChoice({"ragged": 1}),
        ragged_stats_layout=RandomChoice({"token_major": 1, "head_major": 1}),
        total_token_slack=RandomChoice({"packed": 1, "slack": 1}),
        declare_total_seq_len=RandomChoice({True: 1, False: 1}),
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


# ---- model presets ----------------------------------------------------------


def model_knobs(preset, phase):
    """Knob set for a popular-model preset: heads/dims pinned to the model,
    everything else (batch, seq lens, layout, mask flavor, data) fuzzed.
    ``phase``: context (prefill fwd), generation (decode fwd), bprop (training)."""
    sink = RandomChoice({True: 1, False: 1}) if (preset.with_sink and phase != "generation") else Fixed(False)

    if phase == "generation":
        return dict(
            batches=RandomBatchSize(min=1, max=32, with_high_probability=[1, 8]),
            s_q_s_kv=RandomSequenceLength(
                s_q_min=1,
                s_q_max=1,
                s_kv_min=1,
                s_kv_max=8192,
                s_q_distribution={"s_q=1": 100, "s_q=s_kv": 1, "s_q=random": 0},
            ),
            d_qk_d_v=Fixed((preset.head_dim_qk, preset.head_dim_vo)),
            head_count=Fixed((preset.num_q_heads, preset.num_kv_heads, preset.num_kv_heads)),
            data_type=_f16(),
            with_sliding_mask=SlidingWindowMaskGenerator(**SW_NONE),
            diag_align=RandomChoice(DIAG_BOTH),
            is_ragged_or_padded_or_full=RandomChoice({"padded": 1}),
            block_size=RandomBlockSize(min=16, max=256, with_high_probability=[16, 32, 128]),
        )

    return dict(
        batches=RandomBatchSize(min=1, max=4, with_high_probability=[1, 2]),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=1,
            s_q_max=8192,
            s_kv_min=1,
            s_kv_max=8192,
            s_q_distribution={
                "s_q=1": 0,
                "s_q=s_kv": 8,
                "s_q=random": 4,
                "s_q>s_kv": 1,
            },
        ),
        d_qk_d_v=Fixed((preset.head_dim_qk, preset.head_dim_vo)),
        head_count=Fixed((preset.num_q_heads, preset.num_kv_heads, preset.num_kv_heads)),
        data_type=_f16(),
        with_sliding_mask=SlidingWindowMaskGenerator(**preset.mask_weights),
        diag_align=RandomChoice(DIAG_BOTH),
        is_ragged_or_padded_or_full=RandomChoice({"ragged": 2, "padded": 1, "full": 1}),
        with_sink_token=sink,
        ragged_stats_layout=RandomChoice({"token_major": 1, "head_major": 1}),
        total_token_slack=RandomChoice({"packed": 1, "slack": 1}),
        declare_total_seq_len=RandomChoice({True: 1, False: 1}),
    )
