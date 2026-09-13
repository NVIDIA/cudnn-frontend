# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""SM100 (Blackwell) d=256 block-scale MXFP8 backward adapter: SF repack -> dQ -> fused dK/dV.

Two CuTe DSL kernels ported from Xinbo Zhao's ``fmha_mxfp8_large_head_dim``
each a 2-CTA warp-specialized
MMA pipeline over E4M3 payloads with 1x32 E8M0 block scales: the dQ kernel
(Q.K^T, dO.V^T, dS.K) and a fused dK/dV kernel (Q.K^T, dO.V^T, dS^T.Q, P^T.dO).
dS is quantized in-kernel with an online per-block E8M0 scale (the upstream
kernels' "fixed scale 1" mode is deliberately not exposed: its accuracy depends
on the dS magnitude); P is quantized with a fixed 2^-8 descale, cuDNN's MXFP8
convention.

Operand contract (all logical BHSD over BSHD-physical storage; exactly the
``sdpa_mxfp8_backward`` port set): FP8 ``q/k/v/dO`` plus the
transposed-quantization payloads ``q_T/k_T/dO_T`` (each contraction axis needs
its own 1x32 quantization -- a transposed payload is NOT the transpose of the
payload), half-precision ``o_f16/dO_f16``, fp32 ``stats`` (natural-log LSE),
and seven F8_128x4 scale tensors. The gradients ``dQ/dK/dV`` are half precision
(the ``o_f16`` dtype); ``amax_*`` outputs are not produced.

Scale-factor layout: the kernels read scale factors through TMA in a 2-CTA
slot layout the upstream quantizer emits, not cuDNN's canonical F8_128x4 --
see ``kernels/bprop_sf_repack_mxfp8_sm100.py`` for the layouts and why a TMA
descriptor cannot address the shifted copy. The seven graph SF tensors are
therefore repacked (eleven small launches, one per kernel operand form) into
workspace ahead of the two kernels. This is a documented, deliberate exception
to Hard Rule 2 (python/cudnn/AGENTS.md) taken to ship the kernels as validated
upstream; it costs ~1-2% of the backward (SF bytes are 1/32 of the payload)
and is the first thing to remove once the kernels' SF path reads canonical
atoms.

Lives in its own module (not ``api_dsl.py``) so the MXFP8 lowering's imports
stay out of the half-precision adapters' way; ``engines._adapter`` resolves it
by name.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from cuda.bindings import driver as cuda

from cudnn.api_base import TensorDesc
from cudnn.sdpa.bwd.api_dsl import SdpaBwdDsl
from cudnn.sdpa.fwd.api_dsl import WorkspaceCarver, _torch_stream_context, ws_align

_HEAD_DIM = 256
_SF_BLOCK = 32
# The two D256 kernels are compiled at O2 upstream: O3 extends live ranges
# enough to spill the dQ kernel in the public CuTe DSL (source repo
# Agent/CUTE_COMPILER_COMPATIBILITY.md section 2.5); dKdV follows the same
# policy in its harness.
_COMPILE_OPTIONS = "--opt-level 2"


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


class SdpaBwdDslSm100Mxfp8(SdpaBwdDsl):
    """d=256 block-scale MXFP8 backward on Blackwell (see the module docstring)."""

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
        *,
        sample_q_T,
        sample_k_T,
        sample_do_T,
        sample_do_f16,
        sample_sf_q,
        sample_sf_q_T,
        sample_sf_k,
        sample_sf_k_T,
        sample_sf_v,
        sample_sf_do,
        sample_sf_do_T,
        p_scale_log2: int = 8,
        **kwargs,
    ) -> None:
        # Stashed raw; descs are built in _initialize_implementation, which the
        # base __init__ calls once APIBase's own state exists.
        self._mxfp8_samples = dict(
            q_T=sample_q_T,
            k_T=sample_k_T,
            dO_T=sample_do_T,
            dO_f16=sample_do_f16,
            sf_q=sample_sf_q,
            sf_q_T=sample_sf_q_T,
            sf_k=sample_sf_k,
            sf_k_T=sample_sf_k_T,
            sf_v=sample_sf_v,
            sf_do=sample_sf_do,
            sf_do_T=sample_sf_do_T,
        )
        self.p_scale_log2 = int(p_scale_log2)
        super().__init__(sample_q, sample_k, sample_v, sample_o, sample_do, sample_stats, sample_dq, sample_dk, sample_dv, **kwargs)

    # --- geometry --------------------------------------------------------------
    @staticmethod
    def _bshd_physical_ok(desc: TensorDesc) -> bool:
        b, h, s, d = (int(x) for x in desc.shape)
        return tuple(int(x) for x in desc.stride) == (s * h * d, d, h * d, 1)

    def _sf_plan(self):
        """The eleven kernel-side SF buffers: (name, graph SF, rows, k_groups, planes, layout, plane_major).

        Rowwise graph tensors (rows = S, blocks along D) are plane-major
        canonical; columnwise ones (rows = D, blocks along S) carry the
        producer's 2-D swizzle with the D tile outside the head plane (see the
        repack module).
        """
        from cudnn.sdpa.bwd.kernels.bprop_sf_repack_mxfp8_sm100 import SF_LAYOUT_SFA, SF_LAYOUT_SFB

        b, hq, hk, sq, sk, d = self.batch_size, self.h_q, self.h_kv, self.s_q_max, self.s_k_max, self.head_dim_qk
        lq, lk = b * hq, b * hk
        dg = d // _SF_BLOCK
        sqg, skg = _cdiv(sq, _SF_BLOCK), _cdiv(sk, _SF_BLOCK)
        return (
            # dQ kernel operands
            ("dq_sf_q", "sf_q", sq, dg, lq, SF_LAYOUT_SFA, True),
            ("dq_sf_k", "sf_k", sk, dg, lk, SF_LAYOUT_SFB, True),
            ("dq_sf_kt", "sf_k_T", d, skg, lk, SF_LAYOUT_SFB, False),
            ("dq_sf_v", "sf_v", sk, dg, lk, SF_LAYOUT_SFB, True),
            ("dq_sf_do", "sf_do", sq, dg, lq, SF_LAYOUT_SFA, True),
            # fused dK/dV kernel operands
            ("dkdv_sf_q", "sf_q", sq, dg, lq, SF_LAYOUT_SFB, True),
            ("dkdv_sf_qt", "sf_q_T", d, sqg, lq, SF_LAYOUT_SFB, False),
            ("dkdv_sf_k", "sf_k", sk, dg, lk, SF_LAYOUT_SFA, True),
            ("dkdv_sf_v", "sf_v", sk, dg, lk, SF_LAYOUT_SFA, True),
            ("dkdv_sf_do", "sf_do", sq, dg, lq, SF_LAYOUT_SFB, True),
            ("dkdv_sf_dot", "sf_do_T", d, sqg, lq, SF_LAYOUT_SFB, False),
        )

    def _sf_expected_bytes(self, graph_sf: str) -> int:
        """Byte count of a graph SF tensor under cuDNN's F8_128x4 padding rules
        (rows to 128, block columns to 4)."""
        b, hq, hk, sq, sk, d = self.batch_size, self.h_q, self.h_kv, self.s_q_max, self.s_k_max, self.head_dim_qk
        d_sc_pad = _cdiv(_cdiv(d, _SF_BLOCK), 4) * 4
        d_pad = _cdiv(d, 128) * 128
        if graph_sf in ("sf_q", "sf_do"):
            return b * hq * (_cdiv(sq, 128) * 128) * d_sc_pad
        if graph_sf in ("sf_k", "sf_v"):
            return b * hk * (_cdiv(sk, 128) * 128) * d_sc_pad
        if graph_sf in ("sf_q_T", "sf_do_T"):
            return b * hq * (_cdiv(_cdiv(sq, _SF_BLOCK), 4) * 4) * d_pad
        if graph_sf == "sf_k_T":
            return b * hk * (_cdiv(_cdiv(sk, _SF_BLOCK), 4) * 4) * d_pad
        raise ValueError(graph_sf)

    def _initialize_implementation(self) -> None:
        s = self._mxfp8_samples
        self.q_T_desc = self._make_tensor_desc(s["q_T"], name="q_T")
        self.k_T_desc = self._make_tensor_desc(s["k_T"], name="k_T")
        self.do_T_desc = self._make_tensor_desc(s["dO_T"], name="dO_T")
        self.do_f16_desc = self._make_tensor_desc(s["dO_f16"], name="dO_f16")
        self.sf_descs = {n: self._make_tensor_desc(s[n], name=n) for n in ("sf_q", "sf_q_T", "sf_k", "sf_k_T", "sf_v", "sf_do", "sf_do_T")}

        q_shape = tuple(int(x) for x in self.q_desc.shape)  # logical BHSD
        k_shape = tuple(int(x) for x in self.k_desc.shape)
        self.batch_size, self.h_q, self.s_q_max, self.head_dim_qk = q_shape
        self.h_kv, self.s_k_max = int(k_shape[1]), int(k_shape[2])
        self.head_dim_v = int(tuple(self.v_desc.shape)[3])
        self.dtype = self.q_desc.dtype  # FP8 payload dtype
        self.out_dtype = self.o_desc.dtype  # half-precision side (o_f16/dO_f16/dQ/dK/dV)
        if self.scale_softmax is None or self.scale_softmax == 0.0:
            self.scale_softmax = 1.0 / math.sqrt(self.head_dim_qk)
        self._gqa_group = self.h_q // max(self.h_kv, 1)
        self._compiled = None

    # --- capability backstop ---------------------------------------------------
    def check_support(self) -> bool:
        """Re-check what the Capabilities row promised.

        Reaching a raise here means the row lied -- these are backstops, not
        the gate. ValueError, never assert.
        """
        d = self.head_dim_qk
        self._value_error_if(d != _HEAD_DIM or self.head_dim_v != _HEAD_DIM, f"SM100 MXFP8 bwd: d_qk and d_v must both be 256; got {d} / {self.head_dim_v}")
        self._value_error_if(self.h_kv < 1 or self.h_q % self.h_kv != 0, f"SM100 MXFP8 bwd: h_q ({self.h_q}) must be a multiple of h_kv ({self.h_kv})")
        self._value_error_if(self.dtype != torch.float8_e4m3fn, f"SM100 MXFP8 bwd: the kernels consume E4M3 payloads only; got {self.dtype}")
        for desc in (self.k_desc, self.v_desc, self.do_desc, self.q_T_desc, self.k_T_desc, self.do_T_desc):
            self._value_error_if(desc.dtype != self.dtype, f"SM100 MXFP8 bwd: FP8 payload {desc.name} dtype {desc.dtype} != q {self.dtype}")
        self._value_error_if(self.out_dtype not in (torch.float16, torch.bfloat16), f"SM100 MXFP8 bwd: o_f16 must be fp16/bf16; got {self.out_dtype}")
        for desc in (self.do_f16_desc, self.dq_desc, self.dk_desc, self.dv_desc):
            self._value_error_if(desc.dtype != self.out_dtype, f"SM100 MXFP8 bwd: {desc.name} dtype {desc.dtype} != o_f16 {self.out_dtype}")
        # The kernels address BSHD-compact storage only (their head/batch
        # strides are derived, not read). No staging copies: decline.
        for desc in (
            self.q_desc,
            self.k_desc,
            self.v_desc,
            self.o_desc,
            self.do_desc,
            self.dq_desc,
            self.dk_desc,
            self.dv_desc,
            self.q_T_desc,
            self.k_T_desc,
            self.do_T_desc,
            self.do_f16_desc,
        ):
            self._value_error_if(
                not self._bshd_physical_ok(desc),
                f"SM100 MXFP8 bwd: {desc.name} must be BSHD-physical; got shape {tuple(desc.shape)} stride {tuple(desc.stride)}",
            )
        b, hq, sq = self.batch_size, self.h_q, self.s_q_max
        self._value_error_if(
            tuple(int(x) for x in self.stats_desc.shape) != (b, hq, sq, 1),
            f"SM100 MXFP8 bwd: stats must be (B, H_q, S_q, 1); got {tuple(self.stats_desc.shape)}",
        )
        self._value_error_if(
            tuple(int(x) for x in self.stats_desc.stride) != (hq * sq, sq, 1, 1),
            f"SM100 MXFP8 bwd: stats must be contiguous; got stride {tuple(self.stats_desc.stride)}",
        )
        self._value_error_if(self.stats_desc.dtype != torch.float32, f"SM100 MXFP8 bwd: stats must be fp32; got {self.stats_desc.dtype}")
        # A reordered SF tensor is an opaque byte layout: only the byte count is
        # checked (the graph may declare any dims with the right total).
        for name, desc in self.sf_descs.items():
            have = int(math.prod(int(x) for x in desc.shape))
            want = self._sf_expected_bytes(name)
            self._value_error_if(have != want, f"SM100 MXFP8 bwd: {name} has {have} bytes; the F8_128x4 layout for this shape needs {want}")
        self._value_error_if(self.seq_kv_lens_present or self.seq_q_lens_present, "SM100 MXFP8 bwd: padding masks (seq lens) are not implemented")
        self._value_error_if(self.window_size_left is not None, "SM100 MXFP8 bwd: sliding window is not implemented")
        self._value_error_if(self.causal_bottom_right, "SM100 MXFP8 bwd: bottom-right causal is not implemented")
        self._value_error_if(self.is_causal and (self.window_size_right or 0) != 0, "SM100 MXFP8 bwd: causal right-band widening is not implemented")
        self._value_error_if(self.tile_m not in (None, 128) or self.tile_n not in (None, 128), "SM100 MXFP8 bwd: tiles are fixed at 128x128")
        return True

    # --- workspace -------------------------------------------------------------
    def _kernel_workspace_bytes(self) -> int:
        import cutlass

        from cudnn.sdpa.bwd.kernels._bprop_mxfp8_common_sm100 import get_workspace_size

        return int(get_workspace_size(self.s_q_max, self.head_dim_qk, self.h_q, self.batch_size, cutlass.Float32))

    def scratch_workspace_bytes(self) -> int:
        """Kernel scratch (rowsum(O*dO) + scaled LSE + the dQ fp32 slot the
        kernels size but do not use) plus the eleven repacked scale-factor
        buffers. A pure function of the plan geometry; all of it is carved
        from the caller's buffer at execute."""
        from cudnn.sdpa.bwd.kernels.bprop_sf_repack_mxfp8_sm100 import repack_geometry

        total = ws_align(self._kernel_workspace_bytes())
        for _name, _src, rows, kg, l, layout, _pm in self._sf_plan():
            total += ws_align(repack_geometry(rows, kg, l, layout)[3])
        return total

    # --- compilation -----------------------------------------------------------
    def _kernel_view_geoms(self):
        """(shape, stride) of the kernels' 5-D operand views over BSHD storage.

        Kernel operands are ``(S, D, H_r, H_kv, B)`` views of the compact
        ``[B, S, H_q, D]`` buffer, Q heads ordered (kv head, group member), so
        Q head ``h`` maps to KV head ``h // group`` -- cuDNN's GQA convention.
        """
        b, hq, hk, sq, sk, d = self.batch_size, self.h_q, self.h_kv, self.s_q_max, self.s_k_max, self.head_dim_qk
        hr = self._gqa_group
        q_geom = ((sq, d, hr, hk, b), (hq * d, 1, d, hr * d, sq * hq * d))
        kv_geom = ((sk, d, 1, hk, b), (hk * d, 1, d, d, sk * hk * d))
        lse_geom = ((sq, hr, hk, b), (1, sq, sq * hr, sq * hq))
        return q_geom, kv_geom, lse_geom

    def _mask_types(self):
        from cudnn.sdpa.bwd.kernels import _bprop_mxfp8_masks_sm100 as masks

        # Upstream's selection, kept verbatim: the residual mask is needed only
        # for a Q-side tail. A KV-side tail (S_kv not a multiple of 128) is
        # handled inside the window-mask path -- the kv trip count covers the
        # partial tile and its columns are masked -- so an aligned S_q with a
        # ragged S_kv stays on WINDOW_MASK. Pinned by
        # test_sdpa_bwd_mxfp8_sm100.py::test_ragged_kv_only (S_kv = 160 and 96).
        if self.is_causal or self.s_q_max % 128 == 0:
            return masks.MaskEnum.WINDOW_MASK, masks.MaskEnum.WINDOW_MASK_BWD
        return masks.MaskEnum.RESIDUAL_MASK, masks.MaskEnum.RESIDUAL_MASK_BWD

    def compile(self) -> None:
        """Plan-time JIT: the eleven SF repacks, the dQ kernel and the fused dK/dV
        kernel, all against fake operands of the plan's exact geometry."""
        self._ensure_support_checked()
        if self._compiled is not None:
            return self._compiled
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
        from cutlass.cute.typing import Float32, Int32

        from cudnn.sdpa.bwd.kernels.bprop_dkdv_d256_mxfp8_sm100 import BlackwellFmhaBackwardDKDV256
        from cudnn.sdpa.bwd.kernels.bprop_dq_d256_mxfp8_sm100 import BlackwellFmhaBackwardDQ256
        from cudnn.sdpa.bwd.kernels.bprop_sf_repack_mxfp8_sm100 import Mxfp8SfRepackSm100

        E4M3, E8M0 = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU
        out_dt = cutlass.BFloat16 if self.out_dtype == torch.bfloat16 else cutlass.Float16
        b, hk, sq, sk, d = self.batch_size, self.h_kv, self.s_q_max, self.s_k_max, self.head_dim_qk
        hr = self._gqa_group
        q_geom, kv_geom, lse_geom = self._kernel_view_geoms()
        stream = make_fake_stream()

        def fake(dt, geom):
            return make_fake_tensor(dt, geom[0], geom[1], assumed_align=16)

        def fake_flat(dt, n):
            return make_fake_tensor(dt, (n,), (1,), assumed_align=16)

        # SF repacks: one specialization per (rows, groups, planes, layout, source order).
        repacks = {}
        for name, _src, rows, kg, l, layout, pm in self._sf_plan():
            rk = Mxfp8SfRepackSm100(rows, kg, l, layout, src_plane_major=pm)
            fn = cute.compile(rk, fake_flat(cutlass.Int8, rk.src_bytes), fake_flat(cutlass.Int8, rk.dst_bytes), stream)
            repacks[name] = (rk, fn)

        # dS always gets the in-kernel per-block scale; the kernels' fixed-scale
        # specialization exists upstream but is not a mode this engine offers.
        online = True
        mt_dq, mt_dkdv = self._mask_types()
        wr = Int32(0) if self.is_causal else None
        problem_shape = (sq, sk, d, ((hr, hk), b))
        mma_tiler = (128, 128, d)
        ws_fake = fake_flat(cutlass.Uint8, self._kernel_workspace_bytes())

        fq, fkv, flse = fake(E4M3, q_geom), fake(E4M3, kv_geom), fake(Float32, lse_geom)
        fo, fdq = fake(out_dt, q_geom), fake(out_dt, q_geom)
        fdkv = fake(out_dt, kv_geom)

        def fsf(name):
            return fake_flat(E8M0, repacks[name][0].dst_bytes)

        dq_kernel = BlackwellFmhaBackwardDQ256(
            out_dt,
            Float32,
            mma_tiler,
            False,
            mt_dq,
            is_persistent=False,
            online_ds_scale=online,
            store_num_bits_per_copy=(out_dt.width if sq == 1 else None),
        )
        dq_fn = cute.compile(
            dq_kernel,
            problem_shape,
            fq,  # Q
            fkv,  # K
            fkv,  # K_MN
            fkv,  # V
            fo,  # O
            fsf("dq_sf_q"),
            fsf("dq_sf_k"),
            fsf("dq_sf_kt"),
            fsf("dq_sf_v"),
            fsf("dq_sf_do"),
            fdq,  # dQ
            fdkv,  # dK (ABI slot; this kernel never writes it)
            fdkv,  # dV
            fq,  # dO (fp8)
            fo,  # dO_16bits
            flse,
            None,
            None,
            Float32(self.scale_softmax),
            None,
            wr,
            ws_fake,
            stream,
            False,  # skip_sum_odo: this launch computes rowsum(O*dO) and the scaled LSE
            options=_COMPILE_OPTIONS,
        )
        dkdv_kernel = BlackwellFmhaBackwardDKDV256(
            out_dt,
            Float32,
            mma_tiler,
            False,
            mt_dkdv,
            is_persistent=False,
            online_ds_scale=online,
            p_scale_log2=self.p_scale_log2,
        )
        dkdv_fn = cute.compile(
            dkdv_kernel,
            problem_shape,
            fq,  # Q
            fq,  # Q_MN
            fkv,  # K
            fkv,  # V
            fo,  # O
            fsf("dkdv_sf_q"),
            fsf("dkdv_sf_qt"),
            fsf("dkdv_sf_k"),
            fsf("dkdv_sf_v"),
            fsf("dkdv_sf_do"),
            fsf("dkdv_sf_dot"),
            fdkv,  # dK
            fdkv,  # dV
            fq,  # dO (fp8)
            fq,  # dO_MN
            fo,  # dO_16bits
            flse,
            None,
            None,
            Float32(self.scale_softmax),
            None,
            wr,
            ws_fake,
            stream,
            True,  # skip_sum_odo: reuse the dQ launch's workspace prologue
            options=_COMPILE_OPTIONS,
        )
        self._compiled = (repacks, dq_fn, dkdv_fn, problem_shape, wr)
        return self._compiled

    # --- execution -------------------------------------------------------------
    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        do_tensor: torch.Tensor,
        stats_tensor: torch.Tensor,
        dq_tensor: torch.Tensor,
        dk_tensor: torch.Tensor,
        dv_tensor: torch.Tensor,
        scale_softmax: Optional[float] = None,
        workspace: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
        seq_q_lens: Optional[torch.Tensor] = None,
        seq_kv_lens: Optional[torch.Tensor] = None,
        sink_tensor: Optional[torch.Tensor] = None,
        dsink_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        dbias_tensor: Optional[torch.Tensor] = None,
        *,
        q_T_tensor: Optional[torch.Tensor] = None,
        k_T_tensor: Optional[torch.Tensor] = None,
        do_T_tensor: Optional[torch.Tensor] = None,
        do_f16_tensor: Optional[torch.Tensor] = None,
        sf_q: Optional[torch.Tensor] = None,
        sf_q_T: Optional[torch.Tensor] = None,
        sf_k: Optional[torch.Tensor] = None,
        sf_k_T: Optional[torch.Tensor] = None,
        sf_v: Optional[torch.Tensor] = None,
        sf_do: Optional[torch.Tensor] = None,
        sf_do_T: Optional[torch.Tensor] = None,
    ) -> None:
        import cutlass
        from cutlass.cute.runtime import from_dlpack
        from cutlass.cute.typing import Float32

        for _t_name, _t_val in (("sink", sink_tensor), ("dSink", dsink_tensor), ("bias", bias_tensor), ("dBias", dbias_tensor)):
            self._value_error_if(_t_val is not None, f"SM100 MXFP8 bwd: {_t_name} is not implemented")
        self._value_error_if(seq_q_lens is not None or seq_kv_lens is not None, "SM100 MXFP8 bwd: padding masks (seq lens) are not implemented")
        extras = dict(
            q_T=q_T_tensor,
            k_T=k_T_tensor,
            dO_T=do_T_tensor,
            dO_f16=do_f16_tensor,
            sf_q=sf_q,
            sf_q_T=sf_q_T,
            sf_k=sf_k,
            sf_k_T=sf_k_T,
            sf_v=sf_v,
            sf_do=sf_do,
            sf_do_T=sf_do_T,
        )
        missing = [n for n, t in extras.items() if t is None]
        self._value_error_if(bool(missing), f"SM100 MXFP8 bwd: execute needs {missing}")
        # The scale is a runtime argument of the kernels, but the plan's value is
        # the graph's; a different one here would silently change the graph.
        if scale_softmax is not None and scale_softmax != 0.0 and not math.isclose(float(scale_softmax), float(self.scale_softmax), rel_tol=1e-6):
            raise ValueError(f"SM100 MXFP8 bwd: scale_softmax {scale_softmax} differs from the plan's {self.scale_softmax}")

        repacks, dq_fn, dkdv_fn, problem_shape, wr = self.compile()
        b, hk, sq, sk, d = self.batch_size, self.h_kv, self.s_q_max, self.s_k_max, self.head_dim_qk
        hr = self._gqa_group
        E4M3, E8M0 = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU
        stream = self._get_default_stream(current_stream)

        def fp8_view(t, s, h_kv_, h_r_):
            # Logical BHSD over BSHD storage -> the kernel's (S, D, H_r, H_kv, B)
            # view. permute+contiguous is a no-op view here (check_support
            # admitted BSHD-physical only); the int8 view is the DSL's raw-byte
            # ABI for E4M3 payloads.
            st = t.permute(0, 2, 1, 3).contiguous().view(torch.int8).view(b, s, h_kv_, h_r_, d).permute(1, 4, 3, 2, 0)
            ct = from_dlpack(st, assumed_align=16)
            ct.element_type = E4M3
            return ct

        def half_view(t, s, h_kv_, h_r_):
            return from_dlpack(t.permute(0, 2, 1, 3).contiguous().view(b, s, h_kv_, h_r_, d).permute(1, 4, 3, 2, 0), assumed_align=16)

        def sf_bytes(t):
            flat = t.contiguous()
            if flat.dtype != torch.int8:
                flat = flat.view(torch.int8)
            return flat.reshape(-1)

        with _torch_stream_context(current_stream, q_tensor.device):
            carver = WorkspaceCarver(workspace, self.scratch_workspace_bytes(), "sdpa_bwd_sm100_mxfp8")
            ws_kernel = carver.take(self._kernel_workspace_bytes(), torch.uint8)
            sf_bufs = {}
            for name, src, _rows, _kg, _l, _layout, _pm in self._sf_plan():
                rk, fn = repacks[name]
                dst = carver.take(rk.dst_bytes, torch.int8)
                fn(from_dlpack(sf_bytes(extras[src]), assumed_align=16), from_dlpack(dst, assumed_align=16), stream)
                ct = from_dlpack(dst, assumed_align=16)
                ct.element_type = E8M0
                sf_bufs[name] = ct

            Q, Q_MN = fp8_view(q_tensor, sq, hk, hr), fp8_view(q_T_tensor, sq, hk, hr)
            K, K_MN = fp8_view(k_tensor, sk, hk, 1), fp8_view(k_T_tensor, sk, hk, 1)
            V = fp8_view(v_tensor, sk, hk, 1)
            DO, DO_MN = fp8_view(do_tensor, sq, hk, hr), fp8_view(do_T_tensor, sq, hk, hr)
            O, DO16 = half_view(o_tensor, sq, hk, hr), half_view(do_f16_tensor, sq, hk, hr)
            dQ = half_view(dq_tensor, sq, hk, hr)
            dK, dV = half_view(dk_tensor, sk, hk, 1), half_view(dv_tensor, sk, hk, 1)
            LSE = from_dlpack(stats_tensor.reshape(b, hk, hr, sq).permute(3, 2, 1, 0), assumed_align=16)
            WS = from_dlpack(ws_kernel, assumed_align=16)
            scale = Float32(self.scale_softmax)

            dq_fn(
                problem_shape,
                Q,
                K,
                K_MN,
                V,
                O,
                sf_bufs["dq_sf_q"],
                sf_bufs["dq_sf_k"],
                sf_bufs["dq_sf_kt"],
                sf_bufs["dq_sf_v"],
                sf_bufs["dq_sf_do"],
                dQ,
                dK,
                dV,
                DO,
                DO16,
                LSE,
                None,
                None,
                scale,
                None,
                wr,
                WS,
                stream,
                False,
            )
            dkdv_fn(
                problem_shape,
                Q,
                Q_MN,
                K,
                V,
                O,
                sf_bufs["dkdv_sf_q"],
                sf_bufs["dkdv_sf_qt"],
                sf_bufs["dkdv_sf_k"],
                sf_bufs["dkdv_sf_v"],
                sf_bufs["dkdv_sf_do"],
                sf_bufs["dkdv_sf_dot"],
                dK,
                dV,
                DO,
                DO_MN,
                DO16,
                LSE,
                None,
                None,
                scale,
                None,
                wr,
                WS,
                stream,
                True,
            )


__all__ = ["SdpaBwdDslSm100Mxfp8"]
