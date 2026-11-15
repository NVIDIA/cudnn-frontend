# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed pair adapter using the shared declaration parser and native binder."""

from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128
from cudnn.sdpa.bwd.d128_contract import PAIR, select_implementation
from cudnn.sdpa.bwd.d128_selection import pair_available

from .contract import LEGACY_ROW, NAME


class SdpaBwdDslSm100D128Pair(SdpaBwdDslSm100D128):
    _NAME = NAME
    _BINDING_NAME = LEGACY_ROW

    def check_support(self):
        if self.thd:
            raise NotImplementedError(f"{self._NAME}: the deterministic pair requires dense attention")
        if self._uses_grouped_heads() and not self.deterministic:
            raise NotImplementedError(f"{self._NAME}: grouped pair admission requires a deterministic request")
        return super().check_support()

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
        if not pair_available(natural, cc, self.head_dim_qk, int(self.s_k_max), self.window_size_left):
            raise NotImplementedError(f"{self._NAME}: the declaration is outside the deterministic pair envelope")
        return PAIR
