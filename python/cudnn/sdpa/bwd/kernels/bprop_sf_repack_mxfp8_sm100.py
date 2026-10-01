# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""MXFP8 scale-factor repack for the SM100 D256 backward kernels.

The D256 dQ / dKdV kernels (``bprop_dq_d256_mxfp8_sm100`` /
``bprop_dkdv_d256_mxfp8_sm100``) read their E8M0 scale factors through TMA in
a **2-CTA slot layout** the source repo's quantizer emits: every logical scale
plane is stored twice, once canonical and once shifted by 64 rows for the
peer CTA of the pair (the ``SFB`` form), or four times with an extra 64-row
tile shift (the ``SFA`` form). cuDNN's graph declares the canonical
``F8_128x4`` layout, which the kernels cannot address natively: the 64-row
shift is an 8-byte offset inside a 512-byte SF atom and TMA cannot start a
box mid-atom.

This module bridges the two: Q/dO and K/V are fused into two launches, each
writing both slot layouts for two canonical sources into caller-provided workspace. It is a
documented exception to Hard Rule 2 (no adapter-side layout copies), taken
consciously to ship the kernels as validated upstream; the follow-up is to
teach the kernels' SF path to read canonical atoms.

Layout facts (byte offsets, ``l`` = head plane, ``rt``/``ct`` = 128-row /
4-group atom tile, ``m0``/``m1``/``k0`` = row%32 / row//32%4 / group%4):

* canonical (cuDNN F8_128x4 == CUTLASS ``tile_atom_to_shape_SF`` order (2,1,3)),
  as produced for the ROWWISE scales (rows = S, groups along D):
  ``((l*rest_m + rt)*rest_k + ct)*512 + m0*16 + m1*4 + k0``
* the COLUMNWISE scales (rows = D, groups along S; ``descale_*_T`` and the
  forward's ``descale_v``) come from the producer's 2-D swizzle of the
  ``[D, (B*H*S)/32]`` matrix (TE ``swizzle_col_scaling_kernel``, replicated by
  ``test/python/sdpa/mxfp8_quant.swizzle_sf_columnwise``), whose atom order
  puts the 128-row D tile OUTSIDE the head plane:
  ``((rt*l_total + l)*rest_k + ct)*512 + m0*16 + m1*4 + k0``.
  For D <= 128 (one D tile) both orders coincide, which is why the
  distinction never surfaced before d=256. ``src_plane_major=False`` selects
  this source order.
* SFB slot layout, planes ``2l`` (slot 0) and ``2l+1`` (slot 1):
  slot 0 = canonical; slot 1 = rows 64..127 moved to m1 0..1, m1 2..3 = 0.
* SFA slot layout: planes ``2l``/``2l+1`` and row tiles ``2rt`` (even) /
  ``2rt+1`` (odd):
  slot 0 even = canonical; slot 0 odd = m1<2 ← m1+2, m1>=2 ← m1;
  slot 1 even = m1==1 ← m1 1, else 0; slot 1 odd = m1==1 ← m1 3, else 0.

Rows past the valid extent and groups past the valid group count are written
as the E8M0 identity (0x7F, scale 1.0), which is what the upstream harness
feeds the kernels for tail tiles.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int8, Int32

THREADS = 256
SF_LAYOUT_SFA = "sfa"
SF_LAYOUT_SFB = "sfb"
_E8M0_ONE = 0x7F


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def repack_geometry(rows: int, k_groups: int, l: int, sf_layout: str) -> tuple:
    """``(rest_m, rest_k, packed_rest_m, dst_bytes)`` for one SF tensor.

    ``rows`` is the operand's non-contraction extent (S for the rowwise SF of
    Q/K/V/dO, D for the columnwise SF of Q_T/K_T/dO_T), ``k_groups`` the number
    of 32-wide contraction blocks, ``l`` the number of (batch, head) planes.
    """
    rest_m = _ceil_div(rows, 128)
    rest_k = _ceil_div(k_groups, 4)
    packed_rest_m = rest_m * 2 if sf_layout == SF_LAYOUT_SFA else rest_m
    return rest_m, rest_k, packed_rest_m, 2 * l * packed_rest_m * rest_k * 512


class Mxfp8SfRepackPairSm100:
    """Build the SFA and SFB forms of one canonical tensor in one launch.

    One thread owns a contiguous 16-byte ``m0`` row of an F8_128x4 atom.
    Besides halving the launch count, this decodes the atom coordinates once
    and reuses each source byte for both consumers instead of independently
    traversing the two expanded destinations byte by byte.
    """

    def __init__(self, rows: int, k_groups: int, l: int, src_plane_major: bool = True):
        self.rows = int(rows)
        self.k_groups = int(k_groups)
        self.l = int(l)
        self.src_plane_major = bool(src_plane_major)
        self.rest_m, self.rest_k, self.sfa_rest_m, self.sfa_bytes = repack_geometry(rows, k_groups, l, SF_LAYOUT_SFA)
        _, _, self.sfb_rest_m, self.sfb_bytes = repack_geometry(rows, k_groups, l, SF_LAYOUT_SFB)
        self.src_bytes = self.l * self.rest_m * self.rest_k * 512
        self.chunks = self.src_bytes // 16

    @cute.jit
    def __call__(self, src: cute.Tensor, dst_sfa: cute.Tensor, dst_sfb: cute.Tensor, stream):
        self.kernel(src, dst_sfa, dst_sfb).launch(
            grid=(_ceil_div(self.chunks, THREADS), 1, 1),
            block=[THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, src: cute.Tensor, dst_sfa: cute.Tensor, dst_sfb: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        chunk = bidx * THREADS + tidx
        if chunk < self.chunks:
            m0 = chunk % 32
            atom = chunk // 32
            ct = atom % self.rest_k
            atom = atom // self.rest_k
            if cutlass.const_expr(self.src_plane_major):
                rt = atom % self.rest_m
                l_idx = atom // self.rest_m
            else:
                l_idx = atom % self.l
                rt = atom // self.l

            src_base = chunk * 16
            sfb0 = (((2 * l_idx) * self.sfb_rest_m + rt) * self.rest_k + ct) * 512 + m0 * 16
            sfb1 = ((((2 * l_idx + 1) * self.sfb_rest_m + rt) * self.rest_k + ct) * 512) + m0 * 16
            sfa0_even = (((2 * l_idx) * self.sfa_rest_m + 2 * rt) * self.rest_k + ct) * 512 + m0 * 16
            sfa0_odd = (((2 * l_idx) * self.sfa_rest_m + 2 * rt + 1) * self.rest_k + ct) * 512 + m0 * 16
            sfa1_even = ((((2 * l_idx + 1) * self.sfa_rest_m + 2 * rt) * self.rest_k + ct) * 512) + m0 * 16
            sfa1_odd = ((((2 * l_idx + 1) * self.sfa_rest_m + 2 * rt + 1) * self.rest_k + ct) * 512) + m0 * 16

            for j in cutlass.range_constexpr(16):
                m1 = j // 4
                k0 = j % 4
                src_m = rt * 128 + m1 * 32 + m0
                group = ct * 4 + k0
                value = Int8(_E8M0_ONE)
                if src_m < self.rows and group < self.k_groups:
                    value = src[src_base + j]

                dst_sfb[sfb0 + j] = value
                dst_sfa[sfa0_even + j] = value

                shifted_j = j + 8 if j < 8 else j
                shifted_m1 = shifted_j // 4
                shifted_m = rt * 128 + shifted_m1 * 32 + m0
                shifted_value = Int8(_E8M0_ONE)
                if shifted_m < self.rows and group < self.k_groups:
                    shifted_value = src[src_base + shifted_j]
                dst_sfa[sfa0_odd + j] = shifted_value

                dst_sfb[sfb1 + j] = shifted_value if j < 8 else Int8(_E8M0_ONE)
                dst_sfa[sfa1_even + j] = value if 4 <= j and j < 8 else Int8(_E8M0_ONE)
                dst_sfa[sfa1_odd + j] = shifted_value if 4 <= j and j < 8 else Int8(_E8M0_ONE)


class Mxfp8SfRepackTwoPairSm100:
    """Build SFA/SFB forms for two equal-geometry sources in one launch.

    Q and dO share one geometry, as do K and V.  Processing each pair in one
    thread block keeps the established byte mapping while removing two fixed
    launch costs from every backward execution.
    """

    def __init__(self, rows: int, k_groups: int, l: int, src_plane_major: bool = True):
        self.rows = int(rows)
        self.k_groups = int(k_groups)
        self.l = int(l)
        self.src_plane_major = bool(src_plane_major)
        if self.k_groups != 8:
            raise ValueError(f"packed two-source repack requires D=256 (8 scale groups), got {self.k_groups}")
        self.rest_m, self.rest_k, self.sfa_rest_m, self.sfa_bytes = repack_geometry(rows, k_groups, l, SF_LAYOUT_SFA)
        _, _, self.sfb_rest_m, self.sfb_bytes = repack_geometry(rows, k_groups, l, SF_LAYOUT_SFB)
        self.src_bytes = self.l * self.rest_m * self.rest_k * 512
        self.chunks = self.src_bytes // 16

    @cute.jit
    def __call__(
        self,
        src0: cute.Tensor,
        dst0_sfa: cute.Tensor,
        dst0_sfb: cute.Tensor,
        src1: cute.Tensor,
        dst1_sfa: cute.Tensor,
        dst1_sfb: cute.Tensor,
        stream,
    ):
        self.kernel(src0, dst0_sfa, dst0_sfb, src1, dst1_sfa, dst1_sfb).launch(
            grid=(_ceil_div(self.chunks, THREADS), 1, 1),
            block=[THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        src0: cute.Tensor,
        dst0_sfa: cute.Tensor,
        dst0_sfb: cute.Tensor,
        src1: cute.Tensor,
        dst1_sfa: cute.Tensor,
        dst1_sfb: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        chunk = bidx * THREADS + tidx
        if chunk < self.chunks:
            m0 = chunk % 32
            atom = chunk // 32
            ct = atom % self.rest_k
            atom = atom // self.rest_k
            if cutlass.const_expr(self.src_plane_major):
                rt = atom % self.rest_m
                l_idx = atom // self.rest_m
            else:
                l_idx = atom % self.l
                rt = atom // self.l

            # D=256 has exactly two four-group atoms, so each m1 row is one
            # packed uint32.  Preserve the tail-row identity fill while doing
            # one word operation for the four k0 bytes that always share it.
            src0_words = cute.make_tensor(
                cute.recast_ptr(src0.iterator, dtype=cutlass.Int32),
                cute.make_layout((self.src_bytes // 4,), stride=(1,)),
            )
            src1_words = cute.make_tensor(
                cute.recast_ptr(src1.iterator, dtype=cutlass.Int32),
                cute.make_layout((self.src_bytes // 4,), stride=(1,)),
            )
            dst0_sfa_words = cute.make_tensor(
                cute.recast_ptr(dst0_sfa.iterator, dtype=cutlass.Int32),
                cute.make_layout((self.sfa_bytes // 4,), stride=(1,)),
            )
            dst0_sfb_words = cute.make_tensor(
                cute.recast_ptr(dst0_sfb.iterator, dtype=cutlass.Int32),
                cute.make_layout((self.sfb_bytes // 4,), stride=(1,)),
            )
            dst1_sfa_words = cute.make_tensor(
                cute.recast_ptr(dst1_sfa.iterator, dtype=cutlass.Int32),
                cute.make_layout((self.sfa_bytes // 4,), stride=(1,)),
            )
            dst1_sfb_words = cute.make_tensor(
                cute.recast_ptr(dst1_sfb.iterator, dtype=cutlass.Int32),
                cute.make_layout((self.sfb_bytes // 4,), stride=(1,)),
            )

            src_word = chunk * 4
            sfb0 = ((((2 * l_idx) * self.sfb_rest_m + rt) * self.rest_k + ct) * 512 + m0 * 16) // 4
            sfb1 = (((((2 * l_idx + 1) * self.sfb_rest_m + rt) * self.rest_k + ct) * 512) + m0 * 16) // 4
            sfa0_even = ((((2 * l_idx) * self.sfa_rest_m + 2 * rt) * self.rest_k + ct) * 512 + m0 * 16) // 4
            sfa0_odd = ((((2 * l_idx) * self.sfa_rest_m + 2 * rt + 1) * self.rest_k + ct) * 512 + m0 * 16) // 4
            sfa1_even = (((((2 * l_idx + 1) * self.sfa_rest_m + 2 * rt) * self.rest_k + ct) * 512) + m0 * 16) // 4
            sfa1_odd = (((((2 * l_idx + 1) * self.sfa_rest_m + 2 * rt + 1) * self.rest_k + ct) * 512) + m0 * 16) // 4

            identity = Int32(0x7F7F7F7F)
            value00, value01, value02, value03 = identity, identity, identity, identity
            value10, value11, value12, value13 = identity, identity, identity, identity
            row_base = rt * 128 + m0
            if row_base < self.rows:
                value00, value10 = src0_words[src_word], src1_words[src_word]
            if row_base + 32 < self.rows:
                value01, value11 = src0_words[src_word + 1], src1_words[src_word + 1]
            if row_base + 64 < self.rows:
                value02, value12 = src0_words[src_word + 2], src1_words[src_word + 2]
            if row_base + 96 < self.rows:
                value03, value13 = src0_words[src_word + 3], src1_words[src_word + 3]

            for word in cutlass.range_constexpr(4):
                value0 = (value00, value01, value02, value03)[word]
                value1 = (value10, value11, value12, value13)[word]
                shifted0 = (value02, value03, value02, value03)[word]
                shifted1 = (value12, value13, value12, value13)[word]

                dst0_sfb_words[sfb0 + word] = value0
                dst1_sfb_words[sfb0 + word] = value1
                dst0_sfa_words[sfa0_even + word] = value0
                dst1_sfa_words[sfa0_even + word] = value1
                dst0_sfa_words[sfa0_odd + word] = shifted0
                dst1_sfa_words[sfa0_odd + word] = shifted1

                dst0_sfb_words[sfb1 + word] = shifted0 if word < 2 else identity
                dst1_sfb_words[sfb1 + word] = shifted1 if word < 2 else identity
                dst0_sfa_words[sfa1_even + word] = value0 if word == 1 else identity
                dst1_sfa_words[sfa1_even + word] = value1 if word == 1 else identity
                dst0_sfa_words[sfa1_odd + word] = shifted0 if word == 1 else identity
                dst1_sfa_words[sfa1_odd + word] = shifted1 if word == 1 else identity


__all__ = ["Mxfp8SfRepackPairSm100", "Mxfp8SfRepackTwoPairSm100", "SF_LAYOUT_SFA", "SF_LAYOUT_SFB", "THREADS", "repack_geometry"]
