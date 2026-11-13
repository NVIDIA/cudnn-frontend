# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM120 native wide backward with independent gradient-column CTAs."""

from .api_dsl import SdpaBwdDslSm80, _SM80_BWD_FLAVOR_DIMS
from .d128_contract import validate_accumulation_scale
from .sm120_wide_contract import validate_bounds


class SdpaBwdDslSm120Wide(SdpaBwdDslSm80):
    _native_flavor_dims = {**_SM80_BWD_FLAVOR_DIMS, "wide": (512, 512)}
    _native_template_overrides = (("qo_stages", 1), ("dq_smem_coalesce", False), ("tile_q", 16), ("tile_kv", 16), ("warps_per_sg", 1), ("grad_dim_splits", 2))

    def _check_native_device(self, capability, device):
        self._not_implemented_error_if(capability not in ((12, 0), (12, 1)), f"SM120 wide backward requires cc 12.0 or 12.1; got {capability} on {device}")

    def _check_native_tiles(self):
        self._value_error_if(self.tile_m not in (None, 16) or self.tile_n not in (None, 16), "SM120 wide backward requires native 16x16 tiles")

    def _check_native_scale(self, scale):
        validate_accumulation_scale(self.scale_softmax if scale is None else scale)

    def check_support(self):
        b, h, sq, dq = self.q_desc.shape
        _, hk, skv, _ = self.k_desc.shape
        validate_bounds(
            b, h, hk, sq, skv, dq, self.v_desc.shape[-1], self.max_total_seq_len_q, self.max_total_seq_len_kv, self.scale_softmax, self.thd, self.deterministic
        )
        self._not_implemented_error_if(
            self._has_bias or self.sink_desc is not None or self.dsink_desc is not None, "SM120 wide backward bias and sink operands are not yet qualified"
        )
        self._not_implemented_error_if(self._has_rope, "SM120 wide backward does not support RoPE")
        return super().check_support()
