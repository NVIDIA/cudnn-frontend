# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Graph adapter for dense d128 per-tensor FP8 backward with prepared launches."""

import math
from contextlib import nullcontext

import torch

from cudnn.sdpa.bwd.api_dsl import SdpaBwdDsl
from .d128_contract import ACCUMULATION, select_implementation, validate_dense_bounds, workspace_layout

FP8_AMAX = ("amax_dQ", "amax_dK", "amax_dV", "amax_dP")
_GRAD_DTYPES = (torch.float8_e4m3fn, torch.bfloat16, torch.float16)


class SdpaBwdDslSm100D128Fp8(SdpaBwdDsl):
    """Dense MHA ``sdpa_fp8_backward`` at d = 128 with top-left causal or unmasked attention."""

    _NAME = "sdpa_bwd_sm100_d128_fp8"

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
        amax_requested=(),
        external_delta=False,
    ):
        names = tuple(amax_requested)
        unknown = [n for n in names if n not in FP8_AMAX]
        if unknown:
            raise ValueError(f"{self._NAME}: unknown amax outputs {unknown}; expected a subset of {FP8_AMAX}")
        self.amax_requested = frozenset(names)
        self._external_delta = bool(external_delta)
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
        require(not self.thd and not self._thd_ro_spec, "THD / ragged is not supported")
        require(not self._external_delta, "a caller-provided delta is not supported")
        require(not self.seq_q_lens_present and not self.seq_kv_lens_present, "padding masks are not supported")
        require(not self.causal_bottom_right, "bottom-right causal is not supported")
        require(self.window_size_right is None, "right-band widening is not supported")
        require(
            self.window_size_left is None or (type(self.window_size_left) is int and self.window_size_left >= 0),
            "sliding window radius must be a nonnegative integer",
        )
        require(not self.deterministic, "deterministic reductions are not supported")
        require(all(d is None for d in (self.sink_desc, self.dsink_desc, self.bias_desc, self.dbias_desc)), "sink / dSink and bias / dBias are not supported")
        require(self.tile_m is None and self.tile_n is None, "the kernel has fixed tile geometry")
        for role in ROLES[:9]:
            desc = getattr(self, role + "_desc")
            require(desc.ndim == 4 and all(int(n) > 0 for n in desc.shape), f"{role} requires positive rank-four dimensions")
            require(desc.device == self.q_desc.device, f"{role} must reside on {self.q_desc.device}")
        b, h, sq, d = self.q_desc.shape
        bk, hk, sk, dk = self.k_desc.shape
        require(d == dk and d in (64, 128), "D_QK and D_V must equal 64 or 128")
        require(self.window_size_left is None or d == 64, "a sliding window is served at d64 only")
        require(b == bk, "Q and KV require equal batch counts")
        require(h >= hk and h % hk == 0, "H_q must be a positive multiple of H_kv")
        try:
            validate_dense_bounds(b, h, sq, sk, self.window_size_left)
        except ValueError as error:
            require(False, str(error))
        self.dtype = self.q_desc.dtype
        require(self.dtype == torch.float8_e4m3fn, "Q must be an E4M3 payload")
        self.grad_dtype = self.dq_desc.dtype
        require(self.grad_dtype in _GRAD_DTYPES, f"dQ/dK/dV dtype {self.grad_dtype} not in {_GRAD_DTYPES}")
        require(self.dk_desc.dtype == self.grad_dtype and self.dv_desc.dtype == self.grad_dtype, "dQ/dK/dV must share one dtype")
        require(not (h > hk and d == 64 and self.grad_dtype == torch.float16), "FP16 gradients are not served on the grouped d64 route")
        for role in ROLES[:9]:
            desc = getattr(self, role + "_desc")
            if role == "stats":
                require(tuple(desc.shape) == (b, h, sq, 1) and desc.dtype == torch.float32, "Stats must be FP32 (B, H, S_q, 1)")
                require(desc.is_contiguous(), "Stats must be contiguous")
                continue
            kv_side = role in ("k", "v", "dk", "dv")
            s, heads = (sk, hk) if kv_side else (sq, h)
            expected = (b, heads, s, d)
            want = self.grad_dtype if role in ("dq", "dk", "dv") else self.dtype
            require(tuple(desc.shape) == expected and desc.dtype == want, f"{role} must have shape {expected} and dtype {want}")
            require(_same_geometry((desc.shape, desc.stride), (expected, (s * heads * d, d, heads * d, 1))), f"{role} must use compact BSHD storage")
        self.batch_size, self.h_q, self.h_kv = b, h, hk
        self.s_q_max, self.s_k_max = sq, sk
        self.head_dim_qk = self.head_dim_v = d
        self.scale_softmax = d**-0.5 if self.scale_softmax is None else float(self.scale_softmax)
        require(math.isfinite(self.scale_softmax) and self.scale_softmax > 0, "attn_scale must be finite and positive")
        implementation = select_implementation(cc, version, int(b), int(h), int(sq), int(sk), self.scale_softmax, d_qk=int(d), d_v=int(d), input_bytes=1)
        require(implementation == ACCUMULATION, "the declaration is outside the accumulation envelope")
        self._implementation = ACCUMULATION
        self._is_supported = True
        return True

    def compile(self):
        from .prepared_sm100_d128_fp8 import compile_plan

        self._ensure_support_checked()
        self._prepared = compile_plan(self)
        self._compiled_kernel = self._prepared.artifact

    def scratch_workspace_bytes(self):
        self._ensure_support_checked()
        if self.window_size_left is None and self.head_dim_qk == 128:
            from .kernels.sm100.prepared_d192_fp8_host import workspace_layout as pair_workspace_layout

            return pair_workspace_layout(self.batch_size, self.h_q, self.s_q_max)[1]
        if self.h_q != self.h_kv:
            from .prepared_sm100_d128_fp8 import grouped_workspace_layout

            return grouped_workspace_layout(self.batch_size, self.h_q, self.s_q_max, self.s_k_max, self.head_dim_qk)[1]
        return workspace_layout(self.batch_size, self.h_q, self.s_q_max, self.head_dim_qk)[1]

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
        from .prepared import execute
        from .prepared_sm100_d128_fp8 import ROLES_FP8_D128

        if self._prepared is None:
            raise RuntimeError(f"{self._NAME}: compile the plan before execute")
        tensors = dict(zip(ROLES_FP8_D128[:9], (q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, stats_tensor, dq_tensor, dk_tensor, dv_tensor)))
        for name in ROLES_FP8_D128[9:]:
            value = optional.pop(name, None)
            if value is not None:
                tensors[name] = value
        if any(value is not None for value in optional.values()):
            raise ValueError(f"{self._NAME}: unknown optional operands {sorted(k for k, v in optional.items() if v is not None)}")
        facts = {name: facts_of_tensor(tensor) for name, tensor in tensors.items()}
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
            execute(spec, facts, workspace_ptr, stream, scale=scale_softmax)
