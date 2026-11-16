# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan-time geometry of ``sdpa_bwd_sm103_d128_mxfp8``: every constant the kernels specialize on, the shared-memory
struct layouts, and the tcgen05 descriptor constants, computed on the host in Python (no device readback)."""

import math
import struct
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

# The row's admission limits, shared by its declaration check (engines.py) and its support check (the adapter):
# - the nvidia-cutlass-dsl floor (Rule 7): 4.8.1, the version (cu13 flavor) every result of this row was measured with;
# - the attn_scale range: logit magnitude grows with the scale and P's precision follows it (exp2 of fp32 scores);
# - the kernels' int32 row index: B * H_q * S_ALLOC rows (S rounded up to 256 per head) must fit.
CUTEDSL_FLOOR = (4, 8, 1)
MAX_SCALE = 2.0**12
MAX_ROWS = 2**31 - 1


def padded_rows(b: int, h_q: int, s: int) -> int:
    """Rows the kernels index: every head of every batch element padded to a multiple of 256."""
    return b * h_q * (-(-s // 256) * 256)


HD = 128
LOG2E = 1.4426950408889634
B_BK = 128  # keys per dQ tile
B_TILE = B_BK * HD  # 16 KB fp8 tile
P_KV, P_KT = 4, 3  # persistent dQ ring stages: K/V (released after S/dP), Kt (released after dQ)
P_W0 = 3  # first softmax warp of the persistent dQ kernel
NSMX_B = 16  # dQ softmax warps: 4 per TMEM lane quadrant, one 32-key block each
P_NTHREADS = 32 * (P_W0 + NSMX_B)
P_READERS = 2 * (P_W0 + NSMX_B) - 1  # warps reading a unit index (all but rank 0's unit warp)

# persistent dQ TMEM columns
TB_DQ, TB_S, TB_DP, TB_Q, TB_DO, TB_DS = 0, 128, 256, 384, 416, 448
TB_SFQ, TB_SFDO, TB_SFK, TB_SFV, TB_SFKT, TB_SFDS = 480, 484, 488, 492, 496, 500


def f32(x: float) -> float:
    """``x`` rounded to float32 (and back to a Python float, exactly)."""
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def idesc_mx(m: int, n: int, trans_b: int, sfa_id: int, sfb_id: int) -> int:
    return (sfb_id << 4) | (trans_b << 16) | ((n >> 3) << 17) | (1 << 23) | ((m >> 7) << 27) | (sfa_id << 29)


def sdesc_sw128(saddr: int) -> int:
    """128B-swizzled operand tile descriptor: rows of 128 bytes, 8-row atoms 1024 bytes apart."""
    return ((saddr & 0x3FFFF) >> 4) | (1 << 16) | ((1024 >> 4) << 32) | (1 << 46) | (2 << 61)


SDESC_HI = sdesc_sw128(0) >> 32  # the constant upper word of every sw128 descriptor
SDESC_LO_FLAGS = sdesc_sw128(0) & 0xFFFFFFFF


def struct_layout(fields: Sequence[Tuple[str, int, int]]) -> Tuple[Dict[str, int], int]:
    """C struct layout of (name, element alignment, bytes) fields: offsets and the padded size."""
    off, offsets, align = 0, {}, 1
    for name, a, nbytes in fields:
        off = (off + a - 1) // a * a
        offsets[name] = off
        off += nbytes
        align = max(align, a)
    return offsets, (off + align - 1) // align * align


# ---- dK/dV kernel (KV-stationary)
A_BQ = 128  # queries per tile
A_NST = 3  # TMA ring stages (q, dO, dOt, qt per stage)
A_TILE = A_BQ * HD  # 16 KB fp8 tile
A_NSMX = 16  # softmax warps: 4 per TMEM lane quadrant, each owning one 32-query MX block
A_NTHREADS = 32 * (2 + A_NSMX)
# TMEM column map (512 columns). A-operand scale slots written by the softmax threads themselves: an M=128 MMA reads
# the scales of rows 32c..32c+31 from column (slot + c) of lane quadrant c only, so each thread stores just its own
# word and consecutive slots may overlap (slot k, quadrant q -> column TA_SFA + 2k + q).
TA_DV, TA_DK, TA_SD, TA_DS, TA_V, TA_P = 0, 128, 256, 384, 416, 448
TA_SFQ, TA_SFDO, TA_SFDOT, TA_SFQT, TA_SFA = 480, 484, 488, 492, 496
TA_SFK, TA_SFV, TA_SFP, TA_SFDS = TA_SFA, TA_SFA + 2, TA_SFA + 4, TA_SFA + 6  # dS: 4 slots

# dK/dV shared-memory buffer table. Release discipline of every buffer read with ordinary shared-memory loads (Rule 11):
# - tiles / sf / kmat: read only by the tensor cores (descriptors, tcgen05.cp: async proxy); released by tcgen05.commit.
# - lse, dlt (stage s): the softmax warps' ld.shared values feed P and dS, which they store to tensor memory and wait for
#   (tcgen05.wait::st) before arriving dp_pulled / ds_full; the stage is released by ANOTHER warp (the MMA warp's commit
#   after the dK MMA), so the helper does not fit: STORE BEFORE RELEASE.
# - sfkv: each loaded word is stored to tensor memory (tcgen05.st + wait) before the end-of-task CTA barrier behind which
#   the next task's bulk copy refills it: STORE BEFORE RELEASE.
# - tiles 64K..128K as the walk-split receive area: the kept half's ld.shared partials are added and the sums stored
#   (st.shared) before fence.proxy.async.shared::cta and the named barrier in front of the TMA stores and of the next
#   task's TMA refill: STORE BEFORE RELEASE, plus the proxy fence (the refill is an async-proxy write).
# - tmem_base: read once per launch, never refilled.
SMEM_A_FIELDS = (
    ("tiles", 1, A_NST * 4 * A_TILE),  # q, dO, dOt, qt
    ("kmat", 1, 128 * HD),
    ("sf", 1, A_NST * 4 * 512),  # sfq, sfdO, sfdOt, sfqt atoms
    ("lse", 4, A_NST * A_BQ * 4),
    ("dlt", 4, A_NST * A_BQ * 4),
    ("sfkv", 1, 2 * 512),
    # half a: q, dO, their scales, lse, delta (rank 0, needed first); half b: dOt, qt and their scales (rank 1)
    ("full_a", 8, A_NST * 8),
    ("full_b", 8, A_NST * 8),
    ("empty_a", 8, A_NST * 8),
    ("empty_b", 8, A_NST * 8),
    ("sd_full", 8, 8),
    ("s_pulled", 8, 8),
    ("dp_pulled", 8, 8),
    ("ds_full", 8, 8),
    ("ds_empty", 8, 8),
    ("kv_ready", 8, 8),
    ("sfkv_full", 8, 8),
    ("done", 8, 8),
    ("rx_ready", 8, 8),  # walk split: receiver's stage area free / partials landed
    ("rx_full", 8, 8),
    ("tmem_base", 4, 4),
)
SMEM_A_OFF, SMEM_A_SIZE = struct_layout(SMEM_A_FIELDS)
SMEM_A = max(SMEM_A_SIZE + 1024, 120 * 1024)  # dynamic smem the dK/dV kernels request

# persistent dQ shared-memory buffer table. Release discipline of every buffer read with ordinary shared-memory loads
# (Rule 11):
# - sched: every reader warp's ld.shared of the unit index releases the slot through consume_then_arrive_on_peer on
#   rank 0's sched_empty (the address is computed from the index through the kernel's release-zero argument).
# - stq / strow: the softmax warps' ld.shared words (16 Q or dO words and one stats and one delta value per lane) release
#   the staging buffer through consume_then_arrive on st_empty (lane 0 arrives; every lane's fold runs in front of it).
# - stsf, kv, kt, sfkv, sfkt: read only by the tensor cores (tcgen05.cp, descriptors: async proxy); released by commits.
# - stq as the bf16 dQ tile: written by the softmax warps, read by the TMA store (async proxy); bulk_wait_read0 before
#   the unit warp restages it.
# - tmem_base: read once per launch, never refilled.
SMEM_P_FIELDS = (
    ("kv", 1, P_KV * 2 * B_TILE),  # K, V tiles (S, dP operands): released once S/dP are done
    ("kt", 1, P_KT * B_TILE),  # Kt tiles (dQ operand): released once dQ is done
    ("stq", 1, 2 * B_TILE),  # staged Q, dO tiles (128B swizzle); also carries the bf16 dQ tile out
    ("sfkv", 1, P_KV * 2 * 512),
    ("sfkt", 1, P_KT * 512),
    ("stsf", 1, 2 * 512),  # staged sfq, sfdO atoms
    ("strow", 4, 2 * 128 * 4),  # staged stats, delta rows
    ("sched", 4, 2 * 4),
    ("kv_full", 8, P_KV * 8),
    ("kv_empty", 8, P_KV * 8),
    ("kt_full", 8, P_KT * 8),
    ("kt_empty", 8, P_KT * 8),
    ("s_full", 8, 8),
    ("s_empty", 8, 8),
    ("p_full", 8, 8),
    ("p_empty", 8, 8),
    ("done", 8, 8),
    ("dq_free", 8, 8),
    ("st_full", 8, 8),
    ("st_empty", 8, 8),
    ("q_ready", 8, 8),
    ("dq_staged", 8, 8),
    ("sched_full", 8, 2 * 8),
    ("sched_empty", 8, 2 * 8),
    ("tmem_base", 4, 4),
)
SMEM_P_OFF, SMEM_P_SIZE = struct_layout(SMEM_P_FIELDS)
SMEM_P = SMEM_P_SIZE + 1024  # dynamic smem the dQ kernel requests (1024 for the base alignment)
assert SMEM_P <= 232448, "persistent dQ shared memory"


@dataclass(frozen=True)
class Geometry:
    """One compiled problem: its compile-time parameters and everything derived from them."""

    seq: int
    b: int
    h_q: int
    h_kv: int
    scale: float  # float32 value of attn_scale
    neg_log2_scale: float  # float32 value of -log2(attn_scale)
    amax: bool
    causal: bool

    @property
    def nhq(self) -> int:
        return self.b * self.h_q

    @property
    def nhkv(self) -> int:
        return self.b * self.h_kv

    @property
    def grp(self) -> int:
        return self.h_q // self.h_kv

    @property
    def nb(self) -> int:  # 128-row blocks per head holding rows < S
        return -(-self.seq // 128)

    @property
    def nbt(self) -> int:  # 128-row blocks per head in the work decomposition
        return (self.nb + 1) // 2 * 2

    @property
    def s_alloc(self) -> int:
        return self.nbt * 128

    @property
    def tail(self) -> bool:
        return self.seq % 256 != 0

    @property
    def rows(self) -> int:  # workspace rows per array
        return self.nhq * self.s_alloc

    @property
    def c1(self) -> float:  # attn_scale * log2(e) as a float32
        return f32(f32(self.scale) * f32(LOG2E))  # the double product of two floats is exact; one rounding

    @property
    def ndelta_scale(self) -> float:  # -attn_scale / 256 (exact)
        return f32(-f32(self.scale) / 256.0)

    @property
    def c2(self) -> float:  # attn_scale / 256 (exact)
        return f32(f32(self.scale) / 256.0)

    @property
    def a_nit(self) -> int:  # query tiles walked per 128-key block
        return self.grp * self.nb

    @property
    def kv_split_maxt(self) -> int:
        return 160

    @property
    def has_split(self) -> bool:
        """the walk-split dK/dV variant is built where it can pay: its 2 * Hkv * NBT tasks could run in one wave on a
        part with up to 160 SMs, there are >= 2 query tiles to share, and not causal MHA with an odd block count"""
        return (
            self.nbt % 2 == 0
            and 2 * self.h_kv * self.nbt <= self.kv_split_maxt
            and self.grp * self.nb >= 2
            and not (self.causal and self.h_q == self.h_kv and self.nb % 2 == 1)
        )

    @property
    def kv_cl1(self) -> int:
        """cluster size of the plain dK/dV kernel (= the Q / dO multicast width): 4 if 4 | NBT, else 2 (a 4-CTA cluster's
        4 key blocks must share a KV head); the walk split always uses 4"""
        return 4 if self.nbt % 4 == 0 else 2

    @property
    def p_uph(self) -> int:  # unit pairs per query head
        return self.nbt // 2

    @property
    def p_nunits(self) -> int:
        return self.nhq * self.p_uph

    @property
    def p_nkt(self) -> int:  # key tiles per unit (tail builds: the last one is masked at S)
        return self.nb


def geometry(seq: int, b: int, h_q: int, h_kv: int, scale: float, amax: bool, causal: bool) -> Geometry:
    s32 = f32(scale)
    return Geometry(int(seq), int(b), int(h_q), int(h_kv), s32, f32(-math.log2(float(scale))), bool(amax), bool(causal))


def workspace_bytes(g: Geometry) -> int:
    """delta, -LSE term, -delta term, LSE copy (``rows`` fp32 each), then the dQ unit counter (64 words)."""
    return (4 * g.rows + 64) * 4


@dataclass(frozen=True)
class Plan:
    split: int  # dK/dV walk split: 1 or 2 CTAs per 128-key block
    kv_cluster: int
    kv_tasks: int
    kv_cap: int  # co-resident dK/dV CTAs
    kv_grid: int
    dq_cap: int  # co-resident 2-CTA dQ clusters
    dq_clusters: int


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def plan(g: Geometry, clusters_of, batch_invariant: bool = True, split: int = 0) -> Plan:
    """The launch plan from the device's co-resident cluster capacities (``clusters_of(n)``: how many n-CTA clusters of a
    one-CTA-per-SM kernel fit at once), never from an SM count.

    * walk split: on iff the split tasks fit one wave of the 4-CTA capacity. It is the only plan choice that changes
      rounding, so a deterministic request decides it from ONE batch element (batch invariant); any other request
      decides from all B of them, which keeps a large batch off the split's extra exchange.
    * dK/dV grid: the fewest whole clusters that keep the minimum number of waves.
    * persistent dQ: min(2-CTA capacity, unit pairs) clusters, dynamically scheduled."""
    t1_per_batch = g.h_kv * g.nbt
    t1_decide = t1_per_batch if batch_invariant else g.b * t1_per_batch
    if split == 0:
        split = 2 if (g.has_split and 2 * t1_decide <= 4 * clusters_of(4)) else 1
    if split == 2 and not g.has_split:
        raise ValueError("sdpa_bwd_sm103_d128_mxfp8: the walk-split dK/dV variant is not built for this problem")
    cl = 4 if split == 2 else g.kv_cl1
    tasks = g.b * t1_per_batch * split
    cap = clusters_of(cl) * cl
    waves = _cdiv(tasks, cap)
    grid = _cdiv(_cdiv(tasks, waves), cl) * cl
    dq_cap = clusters_of(2)
    return Plan(split, cl, tasks, cap, grid, dq_cap, min(dq_cap, g.p_nunits))
