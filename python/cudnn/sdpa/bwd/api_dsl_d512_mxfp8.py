# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""D512 MXFP8 backward using the native owner pair and canonical block scales."""

from .api_dsl_mxfp8_sm100 import SdpaBwdDslSm100Mxfp8, _ROLES
from .d128_contract import validate_dense_bounds, validate_accumulation_scale
from .d512_fp8_contract import environment_error


class SdpaBwdDslD512Mxfp8(SdpaBwdDslSm100Mxfp8):
    _HEAD_PAIRS = ((512, 512),)
    _PAIR_ROUTE = _HEAD_PAIRS

    def check_support(self):
        from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_state
        from cudnn.frost.device import compute_capability, resolve_device

        def require(value, message):
            if not value:
                raise NotImplementedError(f"sdpa_bwd_d512_mxfp8: {message}")

        require(self.q_desc.device.type == "cuda", "Q must reside on CUDA")
        cc = compute_capability(resolve_device(self.q_desc.device))
        error = environment_error(cc, cutedsl_state()[1]) or cutedsl_arch_requirement_error(cc)
        require(error is None, error)
        require(not self.thd, "THD is not supported")
        require(all(x is None for x in (self.sink_desc, self.dsink_desc, self.bias_desc, self.dbias_desc)), "sink/bias gradients are not supported")
        require(self.window_size_right is None, "right-band widening is not supported")
        require(self.p_scale_log2 == 8, "P uses the fixed MXFP8 descale 2^-8")
        b, h, hk, sq, sk = self.batch_size, self.h_q, self.h_kv, self.s_q_max, self.s_k_max
        validate_dense_bounds(b, h, sq, sk, self.window_size_left)
        validate_accumulation_scale(self.scale_softmax)
        require(max(b, h, hk) <= 65535, "batch/head grid dimensions exceed CUDA limits")
        require(4 * ((max(sq, sk) + 127) // 128) <= 2**31 - 1, "gradient tile grid exceeds CUDA limits")
        require(b * h * ((sq + 127) // 128 * 128) <= 2**31 - 1, "rounded delta index exceeds Int32")
        require(max(sq, sk) * h * 512 * 2 < 2**40, "TMA byte strides exceed supported range")
        for role in _ROLES[:13]:
            desc = getattr(self, role + "_desc")
            kv = role in ("k", "v", "dk", "dv", "k_T")
            shape = (b, h, sq, 1) if role == "stats" else (b, hk if kv else h, sk if kv else sq, 512)
            require(tuple(desc.shape) == shape, f"{role} must have shape {shape}")
            require(desc.device == self.q_desc.device, f"{role} must reside on Q's device")
        for role, desc in self.sf_descs.items():
            require(desc.dtype.itemsize == 1 and desc.device == self.q_desc.device, f"{role} must contain scale bytes on Q's device")
        super().check_support()
        self._is_supported = True
        return True
