# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Prepared native SM107 dense MHA D128 backward graph adapter."""

import torch

from cudnn.sdpa.bwd.api_dsl import SdpaBwdDsl
from .d128_contract import ACCUMULATION, workspace_layout
from .d128_sm107_contract import validate_declaration, validate_scale


class SdpaBwdDslSm107D128(SdpaBwdDsl):
    """BF16/FP16 dense MHA, mask-free or top-left causal, unordered FP32 dQ."""

    _NAME = "sdpa_bwd_sm107_d128"

    def _initialize_implementation(self):
        self._prepared = None
        self._implementation = None
        self._is_supported = False

    def check_support(self):
        from cudnn import _pybind_module
        from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_state
        from cudnn.frost.device import compute_capability, resolve_device
        from .prepared import ROLES, _same_geometry

        def require(condition, reason):
            if not condition:
                raise NotImplementedError(f"{self._NAME}: {reason}")

        self._is_supported = False
        require(not self.thd, "THD / ragged is not supported")
        require(
            self.max_total_seq_len_q is None and self.max_total_seq_len_kv is None and not self.thd_stats_token_major and not self.thd_stats_head_stride,
            "THD capacities and packed Stats metadata are not supported",
        )
        require(not self.deterministic, "use_deterministic_algorithm is not supported: dQ uses unordered FP32 reduce-add")
        require(not self.seq_q_lens_present and not self.seq_kv_lens_present, "padding masks are not supported")
        require(not self.causal_bottom_right, "bottom-right causal is not supported")
        require(self.window_size_left is None, "sliding window is not supported")
        require(self.window_size_right is None, "right-band widening is not supported")
        require(all(d is None for d in (self.sink_desc, self.dsink_desc, self.bias_desc, self.dbias_desc)), "sink / dSink and bias / dBias are not supported")
        require(self.tile_m is None and self.tile_n is None, "the kernel has fixed tile geometry")
        require(self.q_desc.device.type == "cuda", "Q must reside on CUDA")
        for role in ROLES[:9]:
            desc = getattr(self, role + "_desc")
            require(desc.ndim == 4 and all(type(n) is int and n > 0 for n in desc.shape), f"{role} requires positive rank-four dimensions")
            require(desc.device == self.q_desc.device, f"{role} must reside on {self.q_desc.device}")
        b, h, sq, d = self.q_desc.shape
        bk, hk, sk, dk = self.k_desc.shape
        require(d == dk == 128, "D_QK and D_V must equal 128")
        require(b == bk and h == hk, "Q and KV require equal batch and head counts (dense MHA)")
        self.dtype = self.q_desc.dtype
        require(self.dtype in (torch.bfloat16, torch.float16), "only BF16 and FP16 are supported")
        for role in ROLES[:9]:
            desc = getattr(self, role + "_desc")
            if role == "stats":
                require(tuple(desc.shape) == (b, h, sq, 1) and desc.dtype == torch.float32, "Stats must be FP32 (B, H, S_q, 1)")
                require(tuple(desc.stride) == (h * sq, sq, 1, 1), "Stats must be contiguous (B, H, S_q, 1)")
            else:
                s = sk if role in ("k", "v", "dk", "dv") else sq
                expected = (b, h, s, d)
                require(tuple(desc.shape) == expected and desc.dtype == self.dtype, f"{role} must have shape {expected} and Q's dtype")
                require(
                    _same_geometry((desc.shape, desc.stride), (expected, (s * h * d, d, h * d, 1))),
                    f"{role} must use compact BSHD storage; got stride {desc.stride}",
                )
        installed, version = cutedsl_state()
        cc = compute_capability(resolve_device(self.q_desc.device))
        scale = d**-0.5 if self.scale_softmax is None else self.scale_softmax
        try:
            validate_declaration(cc, version if installed else None, b, h, sq, sk, scale)
        except ValueError as error:
            require(False, str(error))
        arch_error = cutedsl_arch_requirement_error(cc)
        require(arch_error is None, arch_error)
        binder = getattr(_pybind_module, "_SdpaBwdBinder", None)
        require(getattr(binder, "supports_zero_workspace", False), "requires a frontend extension with native backward binding support")
        self.batch_size, self.h_q, self.h_kv = b, h, h
        self.s_q_max, self.s_k_max = sq, sk
        self.head_dim_qk = self.head_dim_v = d
        self.scale_softmax = float(scale)
        self._implementation = ACCUMULATION
        self._is_supported = True
        return True

    def compile(self):
        from .prepared_sm107_d128 import compile_plan

        self._ensure_support_checked()
        self._prepared = compile_plan(self)
        self._compiled_kernel = self._prepared.artifact

    def scratch_workspace_bytes(self):
        self._ensure_support_checked()
        return workspace_layout(self.batch_size, self.h_q, self.s_q_max)[1]

    def execute(
        self,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        do_tensor,
        stats_tensor,
        dq_tensor,
        dk_tensor,
        dv_tensor,
        scale_softmax=None,
        workspace=None,
        current_stream=None,
        **optional,
    ):
        from .prepared_sm107 import execute_standalone

        if self._prepared is None:
            raise RuntimeError(f"{self._NAME}: compile the plan before execute")
        if any(value is not None for value in optional.values()):
            raise ValueError(f"{self._NAME}: optional operands were not compiled into this plan")
        if scale_softmax is not None:
            validate_scale(scale_softmax)
        execute_standalone(
            self,
            (q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, stats_tensor, dq_tensor, dk_tensor, dv_tensor),
            workspace,
            current_stream,
            scale_softmax,
        )
