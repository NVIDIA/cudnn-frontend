# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed FP8 body over the shared parser, workspace and prepared chain."""

from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107Fp8 as _SharedAdapter

from .config import PROFILE
from .contract import KERNEL_FILE, NAME, TEMPLATE_TAG


class SdpaBwdDslSm107Fp8(_SharedAdapter):
    _NAME = NAME

    def _datapath_2x2_profile(self):
        return PROFILE

    def _kernel_file(self):
        return KERNEL_FILE

    def _template_tag(self):
        return TEMPLATE_TAG
