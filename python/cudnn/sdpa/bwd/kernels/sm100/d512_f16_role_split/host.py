# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed role-split stage 2 using the shared backward binder and algorithm chain."""

from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm100

from .contract import KERNEL_FILE, NAME, STAGE2_TAG


class SdpaBwdDslSm100D512RoleSplit(SdpaBwdDslSm100):
    _NAME = NAME
    _STAGE2_TAG = STAGE2_TAG
    # Preserve the shared native chain's validation contract.
    _BINDING_NAME = "sdpa_bwd_sm100"

    def _stage2_file(self) -> str:
        return KERNEL_FILE

    def _stage2_record(self, stage2_fields: dict):
        from cudnn.sdpa.bwd.config_sm100 import TemplateParams

        return TemplateParams(**stage2_fields)
