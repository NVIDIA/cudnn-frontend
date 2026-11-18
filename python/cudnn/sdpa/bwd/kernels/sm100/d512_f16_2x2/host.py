# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed two-by-two stage 2 using the shared backward binder and algorithm chain."""

from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm100, _rubin_line

from .contract import KERNEL_FILE, NAME, STAGE2_TAG


class SdpaBwdDslSm100D512TwoByTwo(SdpaBwdDslSm100):
    _NAME = NAME
    _STAGE2_TAG = STAGE2_TAG
    # Preserve the shared native chain's validation contract.
    _BINDING_NAME = "sdpa_bwd_sm100"

    def _stage2_file(self) -> str:
        return KERNEL_FILE

    def _stage2_record(self, stage2_fields: dict):
        from cudnn.sdpa.bwd.config_sm100 import SM107_USABLE_DYN_SMEM_2X2, TemplateParams2x2

        major, minor = self._device_cc()
        wide_smem = _rubin_line(major * 10 + minor)
        return TemplateParams2x2(
            **stage2_fields,
            stages_kv=8 if wide_smem else 4,
            cast_stages=2 if wide_smem else 1,
            **({"smem_cap_bytes": SM107_USABLE_DYN_SMEM_2X2} if wide_smem else {}),
        )
