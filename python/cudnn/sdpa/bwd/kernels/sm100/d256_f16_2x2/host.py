# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed SM107 contracts for the shared two-by-two device body."""

from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107

from .config import PROFILE_SM100, PROFILE_SM107_INTERLEAVED
from .contract import KERNEL_FILE, SM107_D192, SM107_D256, SM107_TEMPLATE_TAG


class SdpaBwdDslSm107D256TwoByTwo(SdpaBwdDslSm107):
    _NAME = SM107_D256
    _BINDING_NAME = "sdpa_bwd_sm107"
    _PROFILE = PROFILE_SM107_INTERLEAVED
    _HEAD_PAIR = (256, 256)
    _THD_SUPPORTED = False

    def _check_head_dims(self):
        self._value_error_if(
            (self.head_dim_qk, self.head_dim_v) != self._HEAD_PAIR,
            f"{self._NAME}: this fixed contract requires (D_QK, D_V) = {self._HEAD_PAIR}",
        )

    def _datapath_2x2_profile(self):
        return self._PROFILE

    def _kernel_file(self):
        return KERNEL_FILE

    def _template_tag(self):
        return SM107_TEMPLATE_TAG


class SdpaBwdDslSm107D192TwoByTwo(SdpaBwdDslSm107D256TwoByTwo):
    _NAME = SM107_D192
    _PROFILE = PROFILE_SM100
    _HEAD_PAIR = (192, 128)
    _THD_SUPPORTED = True
