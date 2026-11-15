# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The native D256 body over the shared parser, workspace and prepared chain."""

from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107

from .config import PROFILE
from .contract import KERNEL_FILE, LEGACY_ROW, NAME, TEMPLATE_TAG


class SdpaBwdDslSm107D256FourByOne(SdpaBwdDslSm107):
    _NAME = NAME
    _BINDING_NAME = LEGACY_ROW

    def _check_head_dims(self):
        self._value_error_if(
            (self.head_dim_qk, self.head_dim_v) != (256, 256),
            f"{self._NAME}: the fixed native body requires D_QK = D_V = 256",
        )

    def _datapath_2x2_profile(self):
        return PROFILE

    def _kernel_file(self):
        return KERNEL_FILE

    def _template_tag(self):
        return TEMPLATE_TAG
