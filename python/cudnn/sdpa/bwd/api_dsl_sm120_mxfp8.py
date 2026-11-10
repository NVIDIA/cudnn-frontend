# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native SM120 MXFP8 backward over canonical block-scale storage."""

import math
import torch
from .api_dsl_mxfp8_sm100 import SdpaBwdDslSm100Mxfp8, _ROLES
from .sm120_fp8_contract import environment_error, validate_bounds


class SdpaBwdDslSm120Mxfp8(SdpaBwdDslSm100Mxfp8):
    """Canonical MXFP8 backward with optional global FP32 gradient amax outputs.

    ``amax_requested`` is a subset of ``amax_dQ``, ``amax_dK`` and ``amax_dV``.
    Bind exactly those outputs at execution. Maxima are taken before the final
    half-precision gradient conversion and do not change gradient arithmetic.
    """

    def __init__(self, *args, amax_requested=(), **kwargs):
        names = tuple(amax_requested)
        allowed = ("amax_dQ", "amax_dK", "amax_dV")
        if any(name not in allowed for name in names):
            raise ValueError(f"SM120 MXFP8 backward expects a subset of {allowed}; got {names}")
        self.amax_requested = frozenset(names)
        super().__init__(*args, **kwargs)

    def _pair_route(self):
        return True

    def check_support(self):
        from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_state
        from cudnn.frost.device import compute_capability, resolve_device
        from .prepared import _same_geometry

        def require(value, message):
            if not value:
                raise NotImplementedError(f"sdpa_bwd_sm120_mxfp8: {message}")

        require(self.q_desc.device.type == "cuda", "Q must reside on CUDA")
        cc = compute_capability(resolve_device(self.q_desc.device))
        error = environment_error(cc, cutedsl_state()[1]) or cutedsl_arch_requirement_error(cc)
        require(error is None, error)
        require(not self.thd, "THD is not supported")
        require(all(x is None for x in (self.sink_desc, self.dsink_desc, self.bias_desc, self.dbias_desc)), "sink/bias gradients are not supported")
        require(not self.seq_q_lens_present and not self.seq_kv_lens_present, "padding masks are not supported")
        require(not self.causal_bottom_right and self.window_size_right is None, "bottom-right and right-band masks are not supported")
        require(self.tile_m is None and self.tile_n is None, "the kernel has fixed tile geometry")
        require(self.p_scale_log2 == 8, "P uses the fixed MXFP8 descale 2^-8")
        b, h, hk, sq, sk = self.batch_size, self.h_q, self.h_kv, self.s_q_max, self.s_k_max
        d, dv = self.head_dim_qk, self.head_dim_v
        try:
            validate_bounds(b, h, hk, sq, sk, d, dv, self.scale_softmax, self.window_size_left)
        except ValueError as error:
            require(False, str(error))
        require(self.dtype == torch.float8_e4m3fn, "Q must be E4M3")
        require(self.out_dtype in (torch.float16, torch.bfloat16), "O/dO_f16/gradients must be FP16 or BF16")
        half = {"o", "do_f16", "dq", "dk", "dv"}
        for role in _ROLES[:13]:
            desc = getattr(self, role + "_desc")
            require(desc.device == self.q_desc.device, f"{role} must reside on Q's device")
            if role == "stats":
                require(tuple(desc.shape) == (b, h, sq, 1) and desc.dtype == torch.float32 and desc.is_contiguous(), "Stats must be contiguous FP32 BHS1")
                continue
            kv = role in ("k", "v", "dk", "dv", "k_T")
            heads, seq = (hk, sk) if kv else (h, sq)
            width = dv if role in ("v", "o", "do", "dv", "do_T", "do_f16") else d
            shape = (b, heads, seq, width)
            want = self.out_dtype if role in half else self.dtype
            require(tuple(desc.shape) == shape and desc.dtype == want, f"{role} must have shape {shape} and dtype {want}")
            require(_same_geometry((desc.shape, desc.stride), (shape, (seq * heads * width, width, heads * width, 1))), f"{role} must use compact BSHD storage")
        for role, desc in self.sf_descs.items():
            require(desc.dtype.itemsize == 1 and desc.device == self.q_desc.device, f"{role} must contain scale bytes on Q's device")
            require(math.prod(desc.shape) == self._sf_expected_bytes(role), f"{role} must cover the canonical F8_128x4 atom layout")
        self._is_supported = True
        return True
