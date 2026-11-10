# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM90 (Hopper) FP16/BF16 SDPA backward: the SM80 prepared chain around a
warpgroup-MMA main kernel (``kernels/sm90/bprop_f16.py``).

The adapter keeps the SM80 lifecycle, plan facts, declared-stride binding,
workspace layout, GQA reduction, dSink, deterministic dQ relay and THD
origins; only the main kernel changes.  It serves the D64 and D128 head-dim
envelopes (logical widths padded up inside them).  Bias / dBias and RoPE are
SM80-only operands and are declined here.  The graph's engine identity is
``sdpa_bwd_sm90``; the shared internal binder keeps its SM80 name.
"""

from .api_dsl import SdpaBwdDslSm80

# Head-dim envelopes the warpgroup kernel is built for.
SM90_BWD_ENVELOPES = ((64, 64), (128, 128))


class SdpaBwdDslSm90(SdpaBwdDslSm80):
    """Compile and execute the SM90 SDPA backward (cc 9.0)."""

    _bwd_kernel_file = "sm90/bprop_f16.py"

    def _check_native_device(self, capability, device):
        major, minor = capability
        self._value_error_if(capability != (9, 0), f"SdpaBwdDslSm90 requires SM90 (cc 9.0); found SM{major}{minor} on {device}")

    def _check_native_tiles(self):
        self._not_implemented_error_if(
            self.tile_m not in (None, 64) or self.tile_n not in (None, 64),
            "SM90 backward runs the native 64 x 64 tile; tile_m / tile_n must be unset or 64",
        )

    def check_support(self) -> bool:
        self._not_implemented_error_if(
            self._has_bias or self.bias_desc is not None or self.dbias_desc is not None, "SM90 backward does not support bias / dBias"
        )
        self._not_implemented_error_if(self._has_rope, "SM90 backward does not support RoPE")
        d_qk, d_v = int(self.q_desc.shape[-1]), int(self.v_desc.shape[-1])
        self._not_implemented_error_if(
            max(d_qk, d_v) > 128,
            f"SM90 backward serves head dims up to 128 (envelopes {SM90_BWD_ENVELOPES}); got (D_QK={d_qk}, D_V={d_v})",
        )
        super().check_support()
        self._not_implemented_error_if(
            (self.flavor_d_qk, self.flavor_d_v) not in SM90_BWD_ENVELOPES,
            f"SM90 backward has no envelope for (D_QK={d_qk}, D_V={d_v}); flavor {self.flavor} ({self.flavor_d_qk}, {self.flavor_d_v})",
        )
        # The SM80 dense d=64 fast path is an mma.sync kernel; the warpgroup kernel serves d=64 itself.
        self._use_d64 = False
        return True
