# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native half backward for SM89 (Ada) on the SM80 mma.sync body.

The SM80 kernels assume the A100's 164 KiB opt-in shared memory; an SM89 part
offers 99 KiB per block. The body already carries the smaller-budget knobs the
SM120 packed adapter uses (one Q/dO stage, direct FP32 dQ atomics instead of the
staged coalesced store), so this adapter only resolves the tile from the padded
head dims through the same 99 KiB rule (``sm120_thd_contract.native_tiles``):
64x64 up to the D128 envelope, 32x32 above it. Dense and packed graphs, masks,
GQA, deterministic dQ, bias and sinks take the SM80 adapter's paths unchanged;
every compile is native to the device the plan is built on.
"""

from .api_dsl import SdpaBwdDslSm80, _sm80_bwd_pick_flavor
from .sm120_thd_contract import native_tiles


class SdpaBwdDslSm89(SdpaBwdDslSm80):
    def _check_native_device(self, capability, device):
        self._not_implemented_error_if(capability != (8, 9), f"SdpaBwdDslSm89 requires SM89 (Ada); found SM{capability[0]}{capability[1]} on {device}")

    def _check_native_tiles(self):
        super()._check_native_tiles()  # the row exposes no tile knobs
        flavor = _sm80_bwd_pick_flavor(self.q_desc.shape[-1], self.v_desc.shape[-1], self._native_flavor_dims)
        m, n = native_tiles(*self._native_flavor_dims[flavor])
        self._native_template_overrides = (
            ("qo_stages", 1),
            ("dq_smem_coalesce", False),
            ("tile_q", m),
            ("tile_kv", n),
            ("warps_per_sg", n // 16),
        )
