# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Dense D512 per-tensor FP8 backward with fixed gradient ownership."""

import math
import torch

from .api_dsl_sm100_d192_fp8 import SdpaBwdDslSm100D192Fp8, _GRAD_DTYPES
from .d128_contract import ACCUMULATION, validate_accumulation_scale, validate_dense_bounds
from .d512_fp8_contract import environment_error

D_QK = D_V = 512


class SdpaBwdDslD512Fp8(SdpaBwdDslSm100D192Fp8):
    """D512 MHA/GQA, unmasked or top-left causal, with optional left window."""

    _NAME = "sdpa_bwd_d512_fp8"

    def check_support(self):
        from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_state
        from cudnn.frost.device import compute_capability, resolve_device
        from .prepared import ROLES, _same_geometry

        def require(condition, reason):
            if not condition:
                raise NotImplementedError(f"{self._NAME}: {reason}")

        require(self.q_desc.device.type == "cuda", "Q must reside on CUDA")
        cc = compute_capability(resolve_device(self.q_desc.device))
        installed, version = cutedsl_state()
        error = environment_error(cc, version if installed else None)
        require(error is None, error)
        arch_error = cutedsl_arch_requirement_error(cc)
        require(arch_error is None, arch_error)
        require(not self.thd and not self._thd_ro_spec, "THD / ragged is not supported")
        require(not self._external_delta, "a caller-provided delta is not supported")
        require(not self.seq_q_lens_present and not self.seq_kv_lens_present, "padding masks are not supported")
        require(not self.causal_bottom_right, "bottom-right causal is not supported")
        require(self.window_size_right is None, "right-band widening is not supported")
        require(all(d is None for d in (self.sink_desc, self.dsink_desc, self.bias_desc, self.dbias_desc)), "sink / dSink and bias / dBias are not supported")
        require(self.tile_m is None and self.tile_n is None, "the kernel has fixed tile geometry")
        for role in ROLES[:9]:
            desc = getattr(self, role + "_desc")
            require(desc.ndim == 4 and all(int(n) > 0 for n in desc.shape), f"{role} requires positive rank-four dimensions")
            require(desc.device == self.q_desc.device, f"{role} must reside on {self.q_desc.device}")
        b, h, sq, d = self.q_desc.shape
        bk, hk, sk, dk = self.k_desc.shape
        require(d == D_QK and dk == D_QK and self.v_desc.shape[3] == D_V, "D_QK and D_V must equal 512")
        require(b == bk and h >= hk and h % hk == 0, "Q/KV require equal batches and an integral positive head ratio")
        try:
            validate_dense_bounds(b, h, sq, sk, self.window_size_left)
            validate_accumulation_scale(D_QK**-0.5 if self.scale_softmax is None else self.scale_softmax)
        except ValueError as error:
            require(False, str(error))
        self.dtype = self.q_desc.dtype
        require(self.dtype == torch.float8_e4m3fn, "Q must be an E4M3 payload")
        self.grad_dtype = self.dq_desc.dtype
        require(self.grad_dtype in _GRAD_DTYPES, f"dQ/dK/dV dtype {self.grad_dtype} not in {_GRAD_DTYPES}")
        require(self.dk_desc.dtype == self.grad_dtype and self.dv_desc.dtype == self.grad_dtype, "dQ/dK/dV must share one dtype")
        for role in ROLES[:9]:
            desc = getattr(self, role + "_desc")
            if role == "stats":
                require(tuple(desc.shape) == (b, h, sq, 1) and desc.dtype == torch.float32, "Stats must be FP32 (B, H, S_q, 1)")
                require(desc.is_contiguous(), "Stats must be contiguous")
                continue
            kv_side = role in ("k", "v", "dk", "dv")
            s, heads = (sk, hk) if kv_side else (sq, h)
            width = D_V if role in ("v", "o", "do", "dv") else D_QK
            expected = (b, heads, s, width)
            want = self.grad_dtype if role in ("dq", "dk", "dv") else self.dtype
            require(tuple(desc.shape) == expected and desc.dtype == want, f"{role} must have shape {expected} and dtype {want}")
            require(
                _same_geometry((desc.shape, desc.stride), (expected, (s * heads * width, width, heads * width, 1))), f"{role} must use compact BSHD storage"
            )
        self.batch_size, self.h_q, self.h_kv = b, h, hk
        self.s_q_max, self.s_k_max = sq, sk
        self.head_dim_qk, self.head_dim_v = D_QK, D_V
        self.scale_softmax = D_QK**-0.5 if self.scale_softmax is None else float(self.scale_softmax)
        require(math.isfinite(self.scale_softmax) and self.scale_softmax > 0, "attn_scale must be finite and positive")
        require(max(h, hk, b) <= 65535, "batch/head grid dimensions must fit CUDA grid.y/z")
        require(4 * ((max(sq, sk) + 127) // 128) <= 2**31 - 1, "gradient tile grid exceeds CUDA grid.x")
        require(b * h * ((sq + 127) // 128 * 128) <= 2**31 - 1, "rounded delta index exceeds the Int32 range")
        require(max(sq, sk) * h * 512 * 2 < 2**40, "BSHD byte strides exceed TMA limits")
        self._implementation = ACCUMULATION
        self._is_supported = True
        return True
