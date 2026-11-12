# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native half packed backward through D256 compiled for SM120.

Shares the row-predicated cp.async / mma.sync compute and prepared pointer ABI
with SM80. One Q/dO stage and direct FP32 dQ atomics fit the SM120 shared-memory
budget. This is a native packed implementation, with no TMA performance claim;
every caller port retains its own physical origins and declared row strides.
The shared internal SM80 binder name is retained for its native alias checks;
the graph's public engine identity remains sdpa_bwd_sm120 (20600).
"""

from .api_dsl import SdpaBwdDslSm80
from .sm120_thd_contract import native_tiles, validate_bounds
from .d128_contract import validate_accumulation_scale


class SdpaBwdDslSm120Thd(SdpaBwdDslSm80):
    _native_template_overrides = (("qo_stages", 1), ("dq_smem_coalesce", False))

    def _check_native_device(self, capability, device):
        self._not_implemented_error_if(capability not in ((12, 0), (12, 1)), f"SM120 THD backward requires cc 12.0 or 12.1; got {capability} on {device}")

    def _check_native_tiles(self):
        m, n = native_tiles(self.q_desc.shape[-1], self.v_desc.shape[-1], self.tile_m, self.tile_n)
        self._native_template_overrides = (
            ("qo_stages", 1),
            ("dq_smem_coalesce", False),
            ("tile_q", m),
            ("tile_kv", n),
            ("warps_per_sg", n // 16),
        )

    def _check_native_scale(self, scale):
        validate_accumulation_scale(self.scale_softmax if scale is None else scale)

    def check_support(self):
        self._not_implemented_error_if(not self.thd, "SM120 packed backward requires thd=True; dense graphs use the dense adapter")
        for enabled, label in (
            (self.deterministic, "deterministic"),
            (self.window_size_left is not None or self.window_size_right is not None, "window"),
            (self.sink_desc is not None or self.dsink_desc is not None, "sink / dSink"),
            (self.bias_desc is not None or self.dbias_desc is not None or self._has_bias, "bias / dBias"),
            (self._has_rope, "RoPE"),
        ):
            self._not_implemented_error_if(enabled, f"SM120 THD backward does not support {label}")
        b, h, sq, d = self.q_desc.shape
        _, hk, skv, _ = self.k_desc.shape
        validate_bounds(b, h, hk, sq, skv, d, self.v_desc.shape[-1], self.max_total_seq_len_q, self.max_total_seq_len_kv, self.scale_softmax)
        return super().check_support()
