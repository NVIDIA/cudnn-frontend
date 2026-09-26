# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin `categorize()` to real kernel names observed in a Gated DeltaNet training step.

A kernel that no substring matches lands in "other", which silently deflates the
component it belongs to -- the share table still adds to 100% and looks healthy. This
guards the name lists against upstream renames; every name below was taken from a
torch-profiler trace of `Qwen3.8/run_model.py` on SM100 (FLA Triton + cuDNN SDPA).
"""

from pathlib import Path
import sys
import unittest

E2E_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2E_DIR))

from _perfshare import categorize  # noqa: E402

OBSERVED = {
    # FLA short convolution -- both directions.
    "causal_conv1d_fwd_kernel": "short_conv",
    "causal_conv1d_bwd_kernel": "short_conv",
    "causal_conv1d_update_kernel": "short_conv",
    # FLA chunked Gated DeltaNet. The chunk_*/wy_repr/recompute names carry no "gdn"
    # or "delta" substring, so they need their own entries.
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": "linear_attn",
    "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64": "linear_attn",
    "chunk_gated_delta_rule_fwd_kernel_kkt_solve": "linear_attn",
    "chunk_fwd_kernel_o": "linear_attn",
    "chunk_fwd_kernel_h": "linear_attn",
    "chunk_bwd_kernel_dqkwg": "linear_attn",
    "prepare_wy_repr_fwd_kernel": "linear_attn",
    "prepare_wy_repr_bwd_kernel": "linear_attn",
    "recompute_w_u_fwd_kernel": "linear_attn",
    "l2norm_fwd_kernel": "linear_attn",
    # cuDNN fused Gated DeltaNet (the cudnn.fla arm).
    "kernel_cutlass_kernel_GdnCfgio_dtype_bfloat16": "linear_attn",
    "kernel_cutlass_kernel_GdnBwdCfguse_initial_stateFalse": "linear_attn",
    "kernel_cutlass_kernel_GdnRecomputeCfg_dtype_bfloat16": "linear_attn",
    # Full attention, both arms of --full_attn_backend.
    "cudnn_generated_fort_native_sdpa_sm100_flash_fprop_f16_knob_2_128x128x256": "full_attn",
    "cudnn_generated_fort_native_sdpa_sm100_flash_bprop_f16_knob_2_128x128x128": "full_attn",
    # GEMM and the SwiGLU MLP.
    "nvjet_sm100_tst_256x128_64x5_2x4_2cta_h_bz_NTT": "gemm",
    "swiglu_fwd_kernel": "misc",
    "swiglu_fwdbwd_kernel": "misc",
    "layer_norm_bwd_kernel1": "norm",
}


class TestPerfshareCategories(unittest.TestCase):
    def test_observed_kernels_are_categorized(self):
        wrong = {k: (categorize(k), want) for k, want in OBSERVED.items() if categorize(k) != want}
        self.assertEqual(wrong, {}, f"miscategorized (got, want): {wrong}")

    def test_nothing_observed_falls_into_other(self):
        leaked = sorted(k for k in OBSERVED if categorize(k) == "other")
        self.assertEqual(leaked, [], f"these land in 'other' and deflate their component: {leaked}")

    def test_unknown_kernels_still_fall_through_to_other(self):
        self.assertEqual(categorize("some_future_kernel_nobody_has_seen"), "other")


if __name__ == "__main__":
    unittest.main()
