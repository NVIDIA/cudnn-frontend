# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed two-CTA adapter; failed cluster admission declines this candidate."""

from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128
from cudnn.sdpa.bwd.d128_grouped_2cta_contract import PAIR_BODY
from .contract import LEGACY_ROW, NAME


class SdpaBwdDslSm100D128AccumulationTwoCTA(SdpaBwdDslSm100D128):
    _NAME = NAME
    _BINDING_NAME = LEGACY_ROW
    cta_pair = "on"

    def check_support(self):
        if self.deterministic or not self._uses_grouped_heads():
            raise NotImplementedError(f"{self._NAME}: two-CTA accumulation requires nondeterministic grouped attention")
        return self._check_grouped_implementation()

    def _check_grouped_implementation(self):
        from cudnn.sdpa.bwd.prepared_sm100_d128_grouped_2cta import check_support, resolve_cta_pair_body

        resolution = resolve_cta_pair_body(self)
        if resolution.mode != "on" or resolution.body != PAIR_BODY:
            raise NotImplementedError(f"{self._NAME}: the explicit two-CTA implementation cannot change its body")
        return check_support(self)

    def _compile_grouped_implementation(self):
        from cudnn.sdpa.bwd.prepared_sm100_d128_grouped_2cta import compile_plan

        return compile_plan(self)
