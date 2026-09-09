# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared knob-set ingredients for the sdpa/suites framework.

Random-suite knob factories live next to their SuiteSpecs in the phase
test files (context/, generation/, bprop/); this module keeps only what
is shared across files: the mask/diag tables, the 16-bit family draw, and
the model knob factories (parameterized over models/catalog.py presets).
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
            # THD and padded decode both; paging is drawn in the post hook and
            # only when the layout is not ragged (paged+ragged is not a form).
            is_ragged_or_padded_or_full=RandomChoice({"ragged": 1, "padded": 1}),
            block_size=RandomBlockSize(min=16, max=256, with_high_probability=[16, 32, 128]),
            total_token_slack=RandomChoice({"packed": 1, "slack": 1}),
            declare_total_seq_len=RandomChoice({True: 1, False: 1}),
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


def model_knobs_fp8(preset, phase):
    """fp8 flavor of model_knobs for the fp8-trained presets (dsv3, kimi_k3):
    heads/dims pinned, e4m3/e5m2 input and fp8/fp16 output fuzzed, layouts and
    masks fuzzed within the fp8 harness envelope (no head-major stats, no
    padded-dense bwd)."""
    # fp8 THD (ragged) with d_qk > 128 makes the backend kernel spin forever
    # (GPU 100%, never returns — even at b=2, s=296): the ragged fp8 engines
    # were only ever exercised at d <= 128 (the generic fp8 THD suites cap d
    # there). Until the backend supports or rejects the combo, keep presets
    # with bigger head dims (dsv3/kimi_k3 d_qk=192, qwen35 d=256) dense-only.
    thd_ok = preset.head_dim_qk <= 128 and preset.head_dim_vo <= 128
    if phase == "generation":
        return dict(
            batches=RandomBatchSize(min=1, max=16, with_high_probability=[1, 8]),
            s_q_s_kv=RandomSequenceLength(
                s_q_min=1,
                s_q_max=1,
                s_kv_min=1,
                s_kv_max=8192,
                s_q_distribution={"s_q=1": 100, "s_q=s_kv": 1, "s_q=random": 0},
            ),
            d_qk_d_v=Fixed((preset.head_dim_qk, preset.head_dim_vo)),
            head_count=Fixed((preset.num_q_heads, preset.num_kv_heads, preset.num_kv_heads)),
            data_type=RandomChoice({torch.float8_e4m3fn: 2, torch.float8_e5m2: 1}),
            output_type=RandomChoice({torch.float8_e4m3fn: 1, torch.float16: 2}),
            with_sliding_mask=SlidingWindowMaskGenerator(**SW_NONE),
            diag_align=RandomChoice(DIAG_BOTH),
            # THD and padded decode both; paging only when not ragged (post hook).
            is_ragged_or_padded_or_full=RandomChoice({"ragged": 1, "padded": 1} if thd_ok else {"padded": 1}),
            block_size=RandomBlockSize(min=16, max=128, with_high_probability=[16, 32, 64]),
            total_token_slack=RandomChoice({"packed": 1, "slack": 1}),
        )

    if phase == "bprop":
        return dict(
            batches=RandomBatchSize(min=1, max=2, with_high_probability=[1]),
            s_q_s_kv=RandomSequenceLength(
                s_q_min=64,
                s_q_max=2048,
                s_kv_min=64,
                s_kv_max=2048,
                s_q_distribution={"s_q=1": 0, "s_q=s_kv": 8, "s_q=random": 4},
            ),
            d_qk_d_v=Fixed((preset.head_dim_qk, preset.head_dim_vo)),
            head_count=Fixed((preset.num_q_heads, preset.num_kv_heads, preset.num_kv_heads)),
            data_type=RandomChoice({torch.float8_e4m3fn: 1}),
            output_type=RandomChoice({torch.float8_e4m3fn: 1, torch.float16: 1}),
            with_sliding_mask=SlidingWindowMaskGenerator(**preset.mask_weights),
            diag_align=RandomChoice(DIAG_BOTH),
            is_ragged_or_padded_or_full=RandomChoice({"ragged": 2, "full": 1} if thd_ok else {"full": 1}),
            is_deterministic=RandomChoice({True: 1, False: 1}),
            total_token_slack=RandomChoice({"packed": 1, "slack": 1}),
        )

    # Seq cap 2048 and tiny batches: these presets carry 64-128 heads, so
    # the fp32 reference cost per config is ~8-16x the generic fp8 suites'
    # (the fp8 BACKWARD reference materializes the full (b,h,s,s) score
    # matrix — 17 GB at h=128 s=4096), and big draws also blow the per-test
    # crash-isolation deadline (conftest, 1500 s).
    return dict(
        batches=RandomBatchSize(min=1, max=2, with_high_probability=[1]),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=1,
            s_q_max=2048,
            s_kv_min=1,
            s_kv_max=2048,
            s_q_distribution={"s_q=1": 0, "s_q=s_kv": 8, "s_q=random": 4, "s_q>s_kv": 1},
        ),
        d_qk_d_v=Fixed((preset.head_dim_qk, preset.head_dim_vo)),
        head_count=Fixed((preset.num_q_heads, preset.num_kv_heads, preset.num_kv_heads)),
        data_type=RandomChoice({torch.float8_e4m3fn: 2, torch.float8_e5m2: 1}),
        output_type=RandomChoice({torch.float8_e4m3fn: 1, torch.float16: 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(**preset.mask_weights),
        diag_align=RandomChoice(DIAG_BR_HEAVY),
        is_ragged_or_padded_or_full=RandomChoice({"ragged": 2, "cu_ragged": 1, "full": 1} if thd_ok else {"full": 1}),
        total_token_slack=RandomChoice({"packed": 1, "slack": 1}),
        declare_total_seq_len=RandomChoice({True: 1, False: 1}),
    )


def model_knobs_mxfp8(preset, phase):
    """mxfp8 flavor of model_knobs: context and bprop only (no decode-shaped
    mxfp8 engine exists), dense full-only layouts (the mxfp8 harness has no
    seq-len plumbing, see #646). Presets whose head dims fall outside the
    mxfp8 d-envelope (e.g. qwen35 d=256) waive at graph build. Seq cap 4096
    and tiny batches for the same reference-cost reason as the fp8 flavor."""
    assert phase in ("context", "bprop"), phase
    sink = RandomChoice({True: 1, False: 1}) if preset.with_sink else Fixed(False)
    s_min = 256 if phase == "bprop" else 128
    return dict(
        batches=RandomBatchSize(min=1, max=2, with_high_probability=[1]),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=s_min,
            s_q_max=4096,
            s_kv_min=s_min,
            s_kv_max=4096,
            s_q_distribution={"s_q=1": 0, "s_q=s_kv": 8, "s_q=random": 4},
        ),
        d_qk_d_v=Fixed((preset.head_dim_qk, preset.head_dim_vo)),
        head_count=Fixed((preset.num_q_heads, preset.num_kv_heads, preset.num_kv_heads)),
        data_type=RandomChoice({torch.float8_e4m3fn: 3, torch.float8_e5m2: 1}),
        output_type=RandomChoice({torch.float16: 2, torch.bfloat16: 1}),
        with_sliding_mask=SlidingWindowMaskGenerator(**preset.mask_weights),
        diag_align=RandomChoice(DIAG_BOTH),
        is_ragged_or_padded_or_full=RandomChoice({"full": 1}),
        with_sink_token=sink,
    )
