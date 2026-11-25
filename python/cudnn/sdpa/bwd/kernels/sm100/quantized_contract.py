# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which SM100 FP8 / MXFP8 backward body a declaration runs; no runtime imports."""

D192_FP8 = "sdpa_bwd_sm100_d192_fp8"  # the dK/dV + dQ pair at (192, 128): one body, its own row
D128_FP8_ROW = "sdpa_bwd_sm100_d128_fp8"
D128_FP8_PAIR = "sdpa_bwd_sm100_d128_fp8_pair"
D128_FP8_ACCUMULATION = "sdpa_bwd_sm100_d128_fp8_accumulation"
MXFP8_ROW = "sdpa_bwd_sm100_mxfp8"
MXFP8_PAIR = "sdpa_bwd_sm100_mxfp8_pair"
D256_MXFP8 = "sdpa_bwd_sm100_d256_mxfp8"
# api_dsl_mxfp8_sm100._PAIR_ROUTE: the heads the MXFP8 row runs on the pair in block-scale mode.
MXFP8_PAIR_HEADS = ((64, 64), (128, 128), (192, 128))


def select_d128_fp8(facts):
    """prepared_sm100_d128_fp8.compile_plan: unwindowed D128 runs the pair; everything else the accumulation body."""
    return D128_FP8_PAIR if facts.window_left is None and facts.d_qk == 128 else D128_FP8_ACCUMULATION


def select_mxfp8(facts):
    """api_dsl_mxfp8_sm100._pair_route: the pair heads run the pair in block-scale mode; D256 runs its own chain."""
    return MXFP8_PAIR if (facts.d_qk, facts.d_v) in MXFP8_PAIR_HEADS else D256_MXFP8
