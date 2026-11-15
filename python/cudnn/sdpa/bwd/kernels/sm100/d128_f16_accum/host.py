# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed one-CTA adapter with shared declaration checks and pointer binding."""

from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128
from cudnn.sdpa.bwd.d128_contract import ACCUMULATION, select_implementation
from cudnn.sdpa.bwd.d128_selection import accumulation_available
from .contract import LEGACY_ROW, NAME


class SdpaBwdDslSm100D128Accumulation(SdpaBwdDslSm100D128):
    _NAME = NAME
    _BINDING_NAME = LEGACY_ROW
    cta_pair = "off"

    def check_support(self):
        if self.deterministic:
            raise NotImplementedError(f"{self._NAME}: accumulation does not serve deterministic requests")
        return super().check_support()

    def _check_grouped_implementation(self):
        from cudnn.sdpa.bwd.prepared_sm100_d128_grouped import check_support

        return check_support(self)

    def _compile_grouped_implementation(self):
        from cudnn.sdpa.bwd.prepared_sm100_d128_grouped import compile_plan

        return compile_plan(self)

    def _choose_dense_implementation(self, cc, version):
        natural = select_implementation(
            cc,
            version,
            int(self.batch_size),
            int(self.h_q),
            int(self.s_q_max),
            int(self.s_k_max),
            self.scale_softmax,
            d_qk=int(self.head_dim_qk),
            d_v=int(self.head_dim_v),
        )
        if not accumulation_available(natural, self.head_dim_qk, int(self.s_k_max), self.window_size_left):
            raise NotImplementedError(f"{self._NAME}: declaration is outside the one-CTA accumulation envelope")
        return ACCUMULATION
