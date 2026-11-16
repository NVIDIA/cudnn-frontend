# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 / SM103 D128 MXFP8 backward adapter for the prep -> dK/dV -> persistent dQ chain.

The geometry module owns launch planning; the kernel module owns computation.
The prepared binder validates current operands and binds caller-owned workspace.
"""

from __future__ import annotations

import math

import torch

from .api_dsl_mxfp8_sm100 import _ATTRIBUTES, _ROLES, SdpaBwdDslSm100Mxfp8
from .kernels.sm100._geometry_d128_mxfp8 import CUTEDSL_FLOOR, MAX_ROWS, MAX_SCALE, padded_rows

ROW = "sdpa_bwd_sm100_d128_mxfp8"
_AMAX_ROLES = ("amax_dq", "amax_dk", "amax_dv")
_AMAX_ATTRIBUTES = ("amax_dQ", "amax_dK", "amax_dV")


def _bhsd_compact(desc) -> bool:
    b, h, s, d = (int(x) for x in desc.shape)
    return tuple(int(x) for x in desc.stride) == (h * s * d, s * d, d, 1)


class SdpaBwdDslSm100D128Mxfp8(SdpaBwdDslSm100Mxfp8):
    """D = 128 block-scale MXFP8 backward on SM100 / SM103 (see the module docstring)."""

    def __init__(self, *args, amax_requested=(), **kwargs) -> None:
        # names among amax_dQ / amax_dK / amax_dV the graph made real outputs (the sibling rows' spelling)
        self.amax_requested = frozenset(amax_requested)
        super().__init__(*args, **kwargs)

    def check_support(self) -> bool:
        """Validate the adapter contract independently of graph eligibility."""
        from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old
        from cudnn.frost.device import compute_capability

        installed, version = cutedsl_state()
        floor = ".".join(str(c) for c in CUTEDSL_FLOOR)
        self._runtime_error_if(not installed, f"{ROW} requires nvidia-cutlass-dsl>={floor}, which is not installed")
        self._runtime_error_if(cutedsl_too_old(version, floor=CUTEDSL_FLOOR), f"{ROW} requires nvidia-cutlass-dsl>={floor}; found {version[1]}")
        dev = int(self.q_desc.device.index or 0)
        self._value_error_if(
            compute_capability(dev) not in ((10, 0), (10, 3)), f"{ROW}: the kernels target compute capability 10.0 or 10.3; got {compute_capability(dev)}"
        )
        self._value_error_if(
            (self.head_dim_qk, self.head_dim_v) != (128, 128),
            f"{ROW}: (d_qk, d_v) must be (128, 128); got ({self.head_dim_qk}, {self.head_dim_v})",
        )
        self._value_error_if(self.h_kv < 1 or self.h_q % self.h_kv != 0, f"{ROW}: h_q ({self.h_q}) must be a multiple of h_kv ({self.h_kv})")
        self._value_error_if(self.s_q_max != self.s_k_max, f"{ROW}: S_q must equal S_kv; got {self.s_q_max} and {self.s_k_max}")
        self._value_error_if(self.s_q_max < 1, f"{ROW}: S must be positive")
        self._value_error_if(self.dtype != torch.float8_e4m3fn, f"{ROW}: the kernels consume E4M3 payloads only; got {self.dtype}")
        for desc in (self.k_desc, self.v_desc, self.do_desc, self.q_T_desc, self.k_T_desc, self.do_T_desc):
            self._value_error_if(desc.dtype != self.dtype, f"{ROW}: FP8 payload {desc.name} dtype {desc.dtype} != q {self.dtype}")
        self._value_error_if(self.out_dtype != torch.bfloat16, f"{ROW}: o / dO_f16 / dQ / dK / dV must be bf16; got {self.out_dtype}")
        for desc in (self.do_f16_desc, self.dq_desc, self.dk_desc, self.dv_desc):
            self._value_error_if(desc.dtype != torch.bfloat16, f"{ROW}: {desc.name} must be bf16; got {desc.dtype}")
        for desc in (
            self.q_desc, self.k_desc, self.v_desc, self.o_desc, self.do_desc, self.dq_desc, self.dk_desc, self.dv_desc,
            self.q_T_desc, self.k_T_desc, self.do_T_desc, self.do_f16_desc,
        ):  # fmt: skip
            self._value_error_if(
                not _bhsd_compact(desc),
                f"{ROW}: {desc.name} must be BHSD-compact (the batch folds into the heads); got shape {tuple(desc.shape)} stride {tuple(desc.stride)}",
            )
        b, hq, sq = self.batch_size, self.h_q, self.s_q_max
        self._value_error_if(tuple(int(x) for x in self.stats_desc.shape) != (b, hq, sq, 1), f"{ROW}: stats must be (B, H_q, S, 1)")
        self._value_error_if(tuple(int(x) for x in self.stats_desc.stride) != (hq * sq, sq, 1, 1), f"{ROW}: stats must be contiguous")
        self._value_error_if(self.stats_desc.dtype != torch.float32, f"{ROW}: stats must be fp32; got {self.stats_desc.dtype}")
        for name, desc in self.sf_descs.items():
            have = int(math.prod(int(x) for x in desc.shape))
            want = self._sf_expected_bytes(name)
            self._value_error_if(have != want, f"{ROW}: {name} has {have} bytes; the F8_128x4 layout for this shape needs {want}")
        self._value_error_if(padded_rows(b, hq, sq) > MAX_ROWS, f"{ROW}: B * H_q * S exceeds the kernels' int32 row range")
        self._value_error_if(self.seq_kv_lens_present or self.seq_q_lens_present, f"{ROW}: padding masks (seq lens) are not implemented")
        self._value_error_if(self.window_size_left is not None, f"{ROW}: sliding windows are not implemented")
        # bottom-right causal: with S_q == S_kv (checked above) its diagonal offset S_kv - S_q is 0, the top-left mask
        self._value_error_if(self.causal_bottom_right and not self.is_causal, f"{ROW}: a bottom-right anchor needs a causal mask")
        self._value_error_if((self.window_size_right or 0) != 0, f"{ROW}: right-band widening is not implemented")
        self._value_error_if(
            not (math.isfinite(self.scale_softmax) and 0 < self.scale_softmax <= MAX_SCALE),
            f"{ROW}: attn_scale must be finite in (0, 2^12]; got {self.scale_softmax}",
        )
        self._value_error_if(self.tile_m not in (None, 128) or self.tile_n not in (None, 128), f"{ROW}: tiles are fixed at 128 x 128")
        return True

    def _geometry(self):
        from cudnn.sdpa.bwd.kernels.sm100 import _geometry_d128_mxfp8 as G

        return G.geometry(
            int(self.s_q_max),
            int(self.batch_size),
            int(self.h_q),
            int(self.h_kv),
            float(self.scale_softmax),
            bool(self.amax_requested),
            bool(self.is_causal),
        )

    def scratch_workspace_bytes(self) -> int:
        """Per-row delta / -LSE / -delta / LSE-copy rows, the dQ unit counter, and 16 bytes for amax outputs the graph
        leaves virtual (the kernels write all three when any is requested)."""
        from cudnn.sdpa.bwd.kernels.sm100 import _geometry_d128_mxfp8 as G

        return G.workspace_bytes(self._geometry()) + 16

    def _operands(self):
        from .prepared import Operand

        operands = []
        for role in _ROLES:
            if role.startswith("sf_"):
                count = self._sf_expected_bytes(role)
                operands.append(Operand("int8", (count,), (1,), count, 16, 1, opaque_bytes=True))
            else:
                desc = getattr(self, role + "_desc")
                shape, strides = tuple(desc.shape), tuple(desc.stride)
                span = 1 + sum((int(d) - 1) * int(st) for d, st in zip(shape, strides))
                operands.append(Operand(str(desc.dtype).split(".")[-1], shape, strides, span, 16, desc.dtype.itemsize))
        if self.amax_requested:
            for name in _AMAX_ATTRIBUTES:
                operands.append(Operand("float32", (1, 1, 1, 1), (1, 1, 1, 1), 1, 4, 4) if name in self.amax_requested else None)
        return tuple(operands)

    def compile(self) -> None:
        self._ensure_support_checked()
        if getattr(self, "_compiled", None) is not None and getattr(self, "_prepared", None) is not None and self._prepared.name == ROW:
            return self._compiled
        from types import SimpleNamespace

        from cudnn.frost.compiled_cache import dsl_compile_lock, positional_entry
        from cudnn.frost.device import compute_capability
        from cudnn.sdpa.bwd.kernels.sm100 import _geometry_d128_mxfp8 as G
        from cudnn.sdpa.bwd.kernels.sm100 import bprop_d128_mxfp8 as kernels

        from .prepared import BwdLaunchSpec

        device = int(self.q_desc.device.index or 0)
        workspace = self.scratch_workspace_bytes()
        amax_scratch = workspace - 16
        amax = bool(self.amax_requested)
        roles = _ROLES + (_AMAX_ROLES if amax else ())
        attributes = _ATTRIBUTES + (_AMAX_ATTRIBUTES if amax else ())
        n = len(roles)

        g = self._geometry()
        # the plan (from the device's co-resident cluster capacities; no allocation, no synchronization) and the one
        # compile it needs, both at plan time; execute is the positional entry alone
        with torch.cuda.device(device), dsl_compile_lock():
            plan = G.plan(g, lambda cluster: kernels.cluster_capacity(device, cluster), batch_invariant=bool(self.deterministic))
            major, minor = compute_capability(device)
            entry = kernels.compile_chain(g, plan.split, plan.kv_cluster, major * 10 + minor)
        chain = positional_entry(entry)
        if chain is None:
            raise NotImplementedError(f"{ROW} requires a positional tvm-ffi entry")
        kv_grid, dq_clusters = int(plan.kv_grid), int(plan.dq_clusters)

        def fn(*frame):
            # the prepared binder's frame: one device pointer per role (None: absent), workspace, attn_scale, stream
            ptrs = [int(x or 0) for x in frame[:n]]
            ws = int(frame[n])
            if amax:
                ptrs[len(_ROLES) :] = [p or (ws + amax_scratch + 4 * i) for i, p in enumerate(ptrs[len(_ROLES) :])]
            else:
                ptrs += [0, 0, 0]
            chain(*ptrs, ws, kv_grid, dq_clusters, frame[n + 2])

        owner = SimpleNamespace(entry=entry, plan=plan, workspace_bytes=workspace)
        self._prepared = BwdLaunchSpec(
            owner,
            fn,
            self._operands(),
            workspace,
            device,
            self.scale_softmax,
            name=ROW,
            roles=roles,
            attributes=attributes,
            scale_log2=False,
            native_binding=False,
        )
        self._compiled = owner
        return owner

    def launch_plan(self):
        """The dK/dV / dQ launch geometry this plan uses (walk split, cluster sizes, grids, device capacities)."""
        return self.compile().plan


__all__ = ["SdpaBwdDslSm100D128Mxfp8", "ROW"]
