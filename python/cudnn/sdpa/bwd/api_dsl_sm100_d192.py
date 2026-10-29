# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Graph adapter for dense and packed D_QK = 192 / D_V = 128 half backward with prepared launches."""

import math
from contextlib import nullcontext

import torch

from cudnn.sdpa.bwd.api_dsl import SdpaBwdDsl
from .d128_contract import ACCUMULATION, select_implementation, validate_dense_bounds, workspace_layout

D_QK = 192
D_V = 128


class SdpaBwdDslSm100D192(SdpaBwdDsl):
    """Dense and packed MHA with 192-wide queries / keys and 128-wide values, top-left causal or unmasked."""

    _NAME = "sdpa_bwd_sm100_d192"

    def __init__(
        self,
        sample_q,
        sample_k,
        sample_v,
        sample_o,
        sample_do,
        sample_stats,
        sample_dq,
        sample_dk,
        sample_dv,
        sample_sink=None,
        sample_dsink=None,
        sample_bias=None,
        sample_dbias=None,
        is_causal=False,
        causal_bottom_right=False,
        window_size_left=None,
        window_size_right=None,
        deterministic=False,
        scale_softmax=None,
        tile_m=None,
        tile_n=None,
        seq_kv_lens_present=False,
        seq_q_lens_present=False,
        thd=False,
        max_total_seq_len_q=None,
        max_total_seq_len_kv=None,
        thd_stats_token_major=False,
        thd_stats_head_stride=None,
        thd_ragged_offsets=None,
    ):
        self._thd_ro_spec = dict(thd_ragged_offsets or {})
        super().__init__(
            sample_q,
            sample_k,
            sample_v,
            sample_o,
            sample_do,
            sample_stats,
            sample_dq,
            sample_dk,
            sample_dv,
            sample_sink,
            sample_dsink,
            sample_bias,
            sample_dbias,
            is_causal,
            causal_bottom_right,
            window_size_left,
            window_size_right,
            deterministic,
            scale_softmax,
            tile_m,
            tile_n,
            seq_kv_lens_present,
            seq_q_lens_present,
            thd,
            max_total_seq_len_q,
            max_total_seq_len_kv,
            thd_stats_token_major,
            thd_stats_head_stride,
        )

    def _initialize_implementation(self):
        self._prepared = None
        self._implementation = None

    def check_support(self):
        if self.thd:
            from .prepared_sm100_d192_thd import check_support

            return check_support(self)

        from cudnn import _pybind_module
        from cudnn.frost.buffers import CUTEDSL_MIN_VERSION, cutedsl_arch_requirement_error, cutedsl_state, cutedsl_too_old
        from cudnn.frost.device import compute_capability, resolve_device
        from .prepared import ROLES, _same_geometry

        def require(condition, reason):
            if not condition:
                raise NotImplementedError(f"{self._NAME}: {reason}")

        installed, version = cutedsl_state()
        require(installed and not cutedsl_too_old(version), f"requires nvidia-cutlass-dsl >= {'.'.join(map(str, CUTEDSL_MIN_VERSION))}; found {version}")
        require(self.q_desc.device.type == "cuda", "Q must reside on CUDA")
        cc = compute_capability(resolve_device(self.q_desc.device))
        require(100 <= cc[0] * 10 + cc[1] <= 106, f"requires SM100-106; got {cc}")
        arch_error = cutedsl_arch_requirement_error(cc)
        require(arch_error is None, arch_error)
        binder = getattr(_pybind_module, "_SdpaBwdBinder", None)
        require(getattr(binder, "supports_zero_workspace", False), "requires a frontend extension with zero-workspace backward binding support")
        require(not self.thd and not self._thd_ro_spec, "THD / ragged is not supported")
        require(not self.seq_q_lens_present and not self.seq_kv_lens_present, "padding masks are not supported")
        require(not self.causal_bottom_right, "bottom-right causal is not supported")
        require(self.window_size_left is None and self.window_size_right is None, "sliding windows are not supported")
        require(not self.deterministic, "deterministic reductions are not supported")
        require(all(d is None for d in (self.sink_desc, self.dsink_desc, self.bias_desc, self.dbias_desc)), "sink / dSink and bias / dBias are not supported")
        require(self.tile_m is None and self.tile_n is None, "the kernel has fixed tile geometry")
        for role in ROLES[:9]:
            desc = getattr(self, role + "_desc")
            require(desc.ndim == 4 and all(int(n) > 0 for n in desc.shape), f"{role} requires positive rank-four dimensions")
            require(desc.device == self.q_desc.device, f"{role} must reside on {self.q_desc.device}")
        b, h, sq, d = self.q_desc.shape
        bk, hk, sk, dk = self.k_desc.shape
        require(d == D_QK and dk == D_QK and self.v_desc.shape[3] == D_V, "D_QK must equal 192 and D_V 128")
        require(b == bk and h == hk, "Q and KV require equal batch and head counts (dense MHA)")
        try:
            validate_dense_bounds(b, h, sq, sk)
        except ValueError as error:
            require(False, str(error))
        self.dtype = self.q_desc.dtype
        require(self.dtype in (torch.bfloat16, torch.float16), "only BF16 and FP16 are supported")
        for role in ROLES[:9]:
            desc = getattr(self, role + "_desc")
            if role == "stats":
                require(tuple(desc.shape) == (b, h, sq, 1) and desc.dtype == torch.float32, "Stats must be FP32 (B, H, S_q, 1)")
                require(desc.is_contiguous(), "Stats must be contiguous")
                continue
            s = sk if role in ("k", "v", "dk", "dv") else sq
            width = D_V if role in ("v", "o", "do", "dv") else D_QK
            expected = (b, h, s, width)
            require(tuple(desc.shape) == expected and desc.dtype == self.dtype, f"{role} must have shape {expected} and Q's dtype")
            require(_same_geometry((desc.shape, desc.stride), (expected, (s * h * width, width, h * width, 1))), f"{role} must use compact BSHD storage")
        self.batch_size, self.h_q, self.h_kv = b, h, h
        self.s_q_max, self.s_k_max = sq, sk
        self.head_dim_qk, self.head_dim_v = D_QK, D_V
        self.scale_softmax = D_QK**-0.5 if self.scale_softmax is None else float(self.scale_softmax)
        require(math.isfinite(self.scale_softmax) and self.scale_softmax != 0, "attn_scale must be finite and nonzero")
        implementation = select_implementation(cc, version, int(b), int(h), int(sq), int(sk), self.scale_softmax, d_qk=D_QK, d_v=D_V)
        require(implementation == ACCUMULATION, "the declaration is outside the accumulation envelope")
        self._implementation = ACCUMULATION
        self._is_supported = True
        return True

    def compile(self):
        self._ensure_support_checked()
        if self.thd:
            from .prepared_sm100_d192_thd import compile_plan
        else:
            from .prepared_sm100_d192 import compile_plan

        self._prepared = compile_plan(self)
        self._compiled_kernel = self._prepared.artifact

    def scratch_workspace_bytes(self):
        self._ensure_support_checked()
        if self.thd:
            return self._thd_workspace_bytes
        return workspace_layout(self.batch_size, self.h_q, self.s_q_max, D_QK)[1]

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
        from cudnn._device import ensure_current_context
        from cudnn.sdpa.fwd.prepared import facts_of_tensor
        from .prepared import ROLES, execute

        if self._prepared is None:
            raise RuntimeError(f"{self._NAME}: compile the plan before execute")
        if self.thd:
            from .prepared_sm100_d192_thd import execute_standalone

            tensors = (q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, stats_tensor, dq_tensor, dk_tensor, dv_tensor)
            execute_standalone(self, tensors, workspace, current_stream, scale_softmax, optional)
            return None
        if any(value is not None for value in optional.values()):
            raise ValueError(f"{self._NAME}: optional operands were not compiled into this plan")
        tensors = (q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, stats_tensor, dq_tensor, dk_tensor, dv_tensor)
        facts = dict(zip(ROLES[:9], map(facts_of_tensor, tensors)))
        geometry = tuple((getattr(self, role + "_desc").shape, getattr(self, role + "_desc").stride) for role in ROLES[:9])
        spec = self._prepared
        workspace_ptr = 0
        if spec.workspace_bytes:
            ws = facts_of_tensor(workspace)
            if ws is None or ws.dtype != "uint8" or not ws.contiguous or ws.span < spec.workspace_bytes or ws.device != (2, spec.device_index):
                raise ValueError(f"{self._NAME} requires {spec.workspace_bytes} bytes of contiguous uint8 workspace on CUDA device {spec.device_index}")
            workspace_ptr = ws.ptr
        stream = int(torch.cuda.current_stream(self.q_desc.device).cuda_stream if current_stream is None else current_stream)
        context = nullcontext() if torch.cuda.current_device() == spec.device_index else torch.cuda.device(spec.device_index)
        with context:
            ensure_current_context(stream, spec.device_index)
            execute(spec, facts, workspace_ptr, stream, scale=scale_softmax, geometry=geometry)
        return None
