# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed recompute adapter using the shared declaration checks and native binder."""

from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128
from cudnn.sdpa.bwd.d128_contract import RECOMPUTE

from .contract import LEGACY_ROW, NAME


class SdpaBwdDslSm100D128Recompute(SdpaBwdDslSm100D128):
    _NAME = NAME
    _BINDING_NAME = LEGACY_ROW

    def check_support(self):
        if self.thd or self._uses_grouped_heads():
            raise NotImplementedError(f"{self._NAME}: recompute requires dense MHA")
        return super().check_support()

    def _choose_dense_implementation(self, cc, version):
        return RECOMPUTE
