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

This module bridges the two: one launch per canonical SF source writes both
slot layouts into caller-provided workspace. It is a
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
from cutlass.cute.typing import Int8

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
    """Build the required SFA/SFB forms of one canonical tensor in one launch.

    One thread owns a contiguous 16-byte ``m0`` row of an F8_128x4 atom.
    Besides halving the launch count when both forms are needed, this decodes
    the atom coordinates once and reuses each source byte for both consumers.
    ``write_sfb=False`` suppresses a dead SFB output for direct-canonical users.
    """

    def __init__(self, rows: int, k_groups: int, l: int, src_plane_major: bool = True, write_sfb: bool = True):
        self.rows = int(rows)
        self.k_groups = int(k_groups)
        self.l = int(l)
        self.src_plane_major = bool(src_plane_major)
        self.write_sfb = bool(write_sfb)
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

                if cutlass.const_expr(self.write_sfb):
                    dst_sfb[sfb0 + j] = value
                dst_sfa[sfa0_even + j] = value

                shifted_j = j + 8 if j < 8 else j
                shifted_m1 = shifted_j // 4
                shifted_m = rt * 128 + shifted_m1 * 32 + m0
                shifted_value = Int8(_E8M0_ONE)
                if shifted_m < self.rows and group < self.k_groups:
                    shifted_value = src[src_base + shifted_j]
                dst_sfa[sfa0_odd + j] = shifted_value

                if cutlass.const_expr(self.write_sfb):
                    dst_sfb[sfb1 + j] = shifted_value if j < 8 else Int8(_E8M0_ONE)
                dst_sfa[sfa1_even + j] = value if 4 <= j and j < 8 else Int8(_E8M0_ONE)
                dst_sfa[sfa1_odd + j] = shifted_value if 4 <= j and j < 8 else Int8(_E8M0_ONE)


__all__ = ["Mxfp8SfRepackPairSm100", "SF_LAYOUT_SFA", "SF_LAYOUT_SFB", "THREADS", "repack_geometry"]
