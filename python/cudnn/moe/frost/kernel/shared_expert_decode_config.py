# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config of the shared-expert DECODE kernel pair (``shared_expert_decode_f16.py``): the geometry it serves and the compile-time knobs of
its two kernel bodies.  Importable without the CuTe DSL (the host entry validates a ``Cfg`` before the DSL version gate runs), so nothing
here imports ``cutlass``.

The op (one tensor-parallel rank's bf16 shared expert next to a routed MoE, decode-time token counts): ``h = x @ W_gu^T`` (``W_gu``
``[2 I, H]`` row-major, gate rows ``0 .. I-1``, up rows ``I .. 2I-1``), ``act = silu(h[:, :I]) * h[:, I:]``, ``y = act @ W_d^T`` (``W_d``
``[H, I]`` row-major), ``y *= sigmoid(x @ W_g^T)`` (``W_g`` ``[H]``), ``x`` / ``y`` ``[T, H]`` bf16 with ``T`` tokens in ``1 .. MAX_TOKENS``.
Geometry ``Geometry(HIDDEN, INTERMEDIATE)`` = ``(H, I)``; the validated envelope is H 8192 with I 256 (8-way tensor parallelism of a
2048-wide shared expert) or 128 (16-way) -- the host entry declines the rest by name, this module's validator checks the structure.

Two launches, one stream, programmatic dependent launch between them (knobs):

* launch 1 ``gate_up`` -- ``(I / 4 / N_GROUPS) x K_SPLIT`` CTAs in clusters of ``K_SPLIT`` along K: cluster ``c`` owns the row-pair GROUPS
  ``g = c N_GROUPS + n`` (``n < N_GROUPS``), group ``g`` = the four row PAIRS ``(j, I + j)``, ``j = 4g .. 4g + 3`` (one ``m16n8k16`` n-tile whose n
  columns are ordered gate_0, up_0, .., gate_3, up_3 by where the bulk copies LAND in shared memory); CTA rank ``ks`` the K slice
  ``[ks H/K_SPLIT, (ks+1) H/K_SPLIT)``; 256 threads.  The W rows stream once by one ``cp.async.bulk`` per row and phase (``W_PHASES``
  mbarriers, one issuing lane per row).  ``x``: in the DEFAULT bulk form (``X_BULK_ROWS``) the ``T`` live rows of the K slice land by one
  ``cp.async.bulk`` each on ``mb_x`` (rows past ``T`` are never loaded; every read of such a row is clamped to row 0), no ring; in the v1 RING
  form ``x`` streams through a ``cp.async`` ring of ``CHUNK_K``-column chunks (rows past ``T`` zero-filled).  Warp ``w`` runs k16 steps
  ``4w .. 4w + 3`` of every 512-column chunk of the slice; the warps' accumulators fold through shared memory in warp order, the cluster's
  through one DSMEM ``st.async`` ship per lane, n-tile and m-tile onto the leader's ``mb_kred``; the leader's warp 0 writes
  ``act[t][4g + i] = bf16(silu(gate) * up)`` for ``t < T``.  Warp ``1 + n`` of every CTA computes the ``GATE_COLS`` columns of the gate logit
  ``x @ W_g^T`` of group ``n`` (leaf ``ks GROUPS + g`` of ``GATE_PARTS``); under ``GATE_CLUSTER_FOLD`` the peers ship the leaf (4 B per
  token, ``st.async``) and the leader's warp ``1 + n`` adds the ``K_SPLIT`` leaves in rank order into slot ``g`` of ``GATE_SLOTS = GROUPS``,
  so launch 2 folds ``GROUPS / 32`` slots per lane instead of ``GATE_PARTS / 32`` leaves (the v1 path, ``GATE_CLUSTER_FOLD`` off).
* launch 2 ``down`` -- ``H / ROWS_PER_CTA`` CTAs of ``4 x ROWS_PER_CTA`` threads: warp ``w`` owns output rows ``n0 + 8w ..`` over the FULL
  K = I (one accumulator chain per output element, no cross-warp reduction); the W_d rows prefetch by ``cp.async`` BEFORE
  ``griddepcontrol.wait``, ``act`` and the gate partials load after it, warp ``w`` folds the ``GATE_SLOTS`` partials of tokens
  ``w, w + WARPS, ..`` into ``sigmoid`` in shared memory; the epilogue writes ``y[t][n] = bf16(acc x sigmoid[t])`` for ``t < T``.

Knobs (every one a ``cutlass.const_expr`` axis of the same two kernel bodies; the default is the ONE-WAVE form: launch 1 over all SMs
(two CTAs per SM at I 256, four at I 128 and T <= 16) with the whole 8 MB / 4 MB weight stream and the live x rows in flight within the first
microsecond):

* ``K_SPLIT`` (1, 2, 4, 8): CTAs per cluster over K -- a ROUNDING-ORDER axis (the slice sum ``((C_0 + C_1) + ..)`` and the gate-logit tree,
  ``GATE_PARTS`` leaves, both move with it); pinned per geometry by ``k_split_for`` (8 on the validated envelope: 1024-column K slices,
  clusters of eight) so the slice order of every output element is a function of the GEOMETRY ALONE -- never of ``T``, never of the part's
  SM count (a part too small to hold the pinned form in one wave is declined by type, the pin does not move).  The first release's default
  was 2 at I 256 / 4 at I 128 (128 CTAs of 64 / 32 KB; ``v1_cfg``): its goldens are the ``ksplit2`` / ``ksplit4`` sets, the ksplit8
  default's the ``ksplit8`` sets.
* ``TOKEN_TILES`` (1, 2): m-tiles of 16 tokens -- the FORM axis the host entry picks from ``T`` (1 for ``T <= 16``, 2 for ``17 .. 24``);
  bit-neutral by the row-position invariance of the ``m16n8k16`` datapath (a token's row of the A operand is contracted on its own; the
  premise the batch-invariance cells of the determinism suite pin).
* ``N_GROUPS`` (0 = the one-wave rule, 1, 2): row-pair groups (n8 tiles) per CTA of launch 1 -- the second FORM axis: ``0`` picks the
  two-group form when its grid still covers every SM in one wave (``SM_COUNT <= GRID1 <= SM_COUNT x co_resident_ctas1``: I 256, 256 CTAs
  of 32 KB, two per SM at every T -- measured +6..+16 % over the 512-CTA one-group form at T 4-6), else the smallest group count that
  fits one wave (I 128: 256 one-group CTAs).  Bit-neutral: the per-element chain does not depend on how many n-tiles a CTA owns.
* ``X_BULK_ROWS`` (default on): x resident by one bulk copy per live row (no ring, ``X_STAGES`` unused); off = the v1 ``cp.async`` ring.
  Bit-neutral: who loads x and how does not enter the sum.
* ``GATE_CLUSTER_FOLD`` (default on): the gate-logit leaves of a cluster are added on the leader in rank order and launch 2 folds
  ``GROUPS`` slots; off = the v1 path (launch 2 folds all ``GATE_PARTS`` leaves).  A ROUNDING-ORDER axis of the gate tree (the leaf values
  are the same; the tree over them differs): the ``ksplit2`` / ``ksplit4`` goldens are recorded with it off, the ``ksplit8`` goldens with it
  on.  Forced off at ``K_SPLIT`` 1 (one leaf per group: the two trees coincide).
* ``KRED_SPLIT`` (default off): the leader's cluster-fold barrier split in two -- ``mb_kred`` takes the accumulator ships alone (waited by
  warp 0), ``mb_kred_gate`` the gate leaves (armed and waited by the gate warps), so warp 0's act epilogue no longer waits for the slowest
  peer's gate leaf and the gate warps not for the accumulator ships.  Bit-neutral (no addition moves; the same values land in the same
  slots).  Forced off without the cluster-side gate fold (no gate ship to separate).  An A/B arm until measured.
* ``X_STAGES`` (ring form only; 0 = the budget rule: every chunk resident when the shared memory holds them, else the deepest ring that fits;
  >= 2 always): the ``cp.async`` ring depth of launch 1.  Bit-neutral.
* ``W_PHASES`` (1, 2): mbarriers over the W slice's K halves so the first k-steps start on the first half.  Bit-neutral.
* ``ROWS_PER_CTA`` (64, 32): W_d rows per CTA of launch 2 (128 / 256 CTAs; 256 / 128 threads).  Bit-neutral: a row's K order does not move.
* ``MIN_CTAS_PER_SM`` (1, 2): the shared-memory budget launch 1's RING is sized against (the per-CTA opt-in, or half the SM for a ring form
  that must co-reside twice); the validator pins the co-residency it claims.
* ``SM_COUNT``: the part's multiprocessor count the one-wave rule reads (the host entry stamps the device's; the module default 148 is the
  smaller SM100-class part, so a 148- and a 152-SM part pick the same forms).
* ``PDL`` / ``PDL_RELEASE`` / ``PDL_PAIR``: ``PDL`` = launch 1's programmatic-stream-serialization attribute (its predecessor is the norm that
  writes ``x``), ``PDL_PAIR`` = launch 2's (its predecessor is launch 1), ``PDL_RELEASE`` = ``griddepcontrol.launch_dependents`` at the entry
  of both kernels (measured a LOSS of 10-33 % in every form on the GB300: launch 2's warps spin on SMs launch 1 still needs; kept as the
  A/B arm, never a default).  Both kernel bodies take ``griddepcontrol.wait`` UNCONDITIONALLY (a no-op without the attribute), so one cubin
  serves the programmatic and the plain arm; the attributes are launch facts.  CALLER OBLIGATION under ``PDL``: the launch's immediate
  stream predecessor must not write ``w_gate_up`` / ``w_gate`` (the bulk copies of the W slice and every lane's gate-weight words issue
  before the wait; ``x`` is read only after it; the serving chain's weights are static).  Under ``PDL_PAIR`` the obligation is launch 1's
  own (``w_down`` prefetches before launch 2's wait and is static).  Every default is off: a knob measured before it defaults on (``PDL`` on
  launch 1 measured +7..+16 % at T 1 with a flush kernel as predecessor; the e2e A/B with the norm as predecessor decides its default).
* ``SMEM_OPTIN_BYTES`` / ``SMEM_PER_SM_BYTES``: the part's readings (the host entry replaces the cc 10.x module constants with the device's).
"""

from dataclasses import dataclass, replace
from typing import Tuple

# --- fixed kernel constants ---
MMA_M = 16  # tokens per m-tile (mma.sync.m16n8k16)
MMA_N = 8  # weight rows per n-tile
MMA_K = 16  # k per mma step
PAIRS_PER_GROUP = 4  # (gate, up) row pairs per row-pair group = one n8 tile of launch 1
ROWS_PER_GROUP = 2 * PAIRS_PER_GROUP  # 8 W_gu rows per n8 tile
MAX_K_SPLIT = 8  # the portable cluster size
SM100_SM_COUNT = 148  # the smaller SM100-class part (cc 10.0); cc 10.3 has 152 -- the one-wave rule reads the host entry's stamp
CHUNK_K = 512  # x columns per ring chunk of launch 1
PITCH_PAD_BYTES = 16  # every shared-memory row pitch is 128 n + 16 B: consecutive rows shift by 4 banks (ldmatrix 8-row phases cover 32)
COPY_BYTES = 16  # one cp.async granule
MAX_TOKENS = 24  # the served T ceiling (the 25 .. 32 band of the two-tile form is unmeasured and declined)
SCRATCH_ROWS = 32  # rows of the entry-owned act / gate_part scratch (two m-tiles)
THREADS_PER_ROW_TILE = 32  # launch 2: one warp per n8 tile of ROWS_PER_CTA rows
# barrier init counts (SUM(issuing lanes) == init on every path): one elected lane arms each tx mbarrier, the copies complete the bytes
W_MBAR_ARRIVERS = 1
X_MBAR_ARRIVERS = 1
KRED_MBAR_ARRIVERS = 1
# the shuffle butterfly of a warp reduction: 5 xor steps (offsets 1, 2, 4, 8, 16)
BUTTERFLY_STEPS = 5
# the cc 10.x readings (228 KB per SM, 227 KB per-CTA opt-in, the driver's 1 KB per-CTA reserve); the host entry reads the part
SM100_SMEM_PER_SM_BYTES = 228 * 1024
SM100_SMEM_OPTIN_BYTES = 227 * 1024
SMEM_CTA_RESERVE_BYTES = 1024
SMEM_RING_MARGIN_BYTES = 1024  # kept free of the ring: the allocation granularity and the mbarrier slots' alignment padding

NAME = "shared_expert_decode"


@dataclass(frozen=True)
class Geometry:
    """``(HIDDEN, INTERMEDIATE)`` of one rank's shared expert: ``W_gu`` is ``[2 INTERMEDIATE, HIDDEN]``, ``W_d`` ``[HIDDEN, INTERMEDIATE]``."""

    HIDDEN: int
    INTERMEDIATE: int

    def __post_init__(self):
        for name in ("HIDDEN", "INTERMEDIATE"):
            v = getattr(self, name)
            if isinstance(v, bool) or not hasattr(v, "__index__") or int(v) < 1:
                raise ValueError(f"{NAME}: Geometry.{name} must be a positive int, got {v!r}")
            object.__setattr__(self, name, int(v))

    def describe(self) -> str:
        return f"hidden {self.HIDDEN}, intermediate {self.INTERMEDIATE}"


DEFAULT_GEOMETRY = Geometry(8192, 256)
#: the geometries the pair was validated on (8- and 16-way tensor parallelism of a hidden-8192 model's 2048-wide shared expert); the host
#: entry declines any other by name (the structural validator below admits more, unmeasured)
SERVED_GEOMETRIES = (Geometry(8192, 256), Geometry(8192, 128))


@dataclass(frozen=True)
class Cfg:
    THREADS: int = 256
    K_SPLIT: int = MAX_K_SPLIT
    TOKEN_TILES: int = 1
    N_GROUPS: int = 0
    X_BULK_ROWS: bool = True
    GATE_CLUSTER_FOLD: bool = True
    KRED_SPLIT: bool = False
    X_STAGES: int = 0
    W_PHASES: int = 1
    ROWS_PER_CTA: int = 64
    MIN_CTAS_PER_SM: int = 1
    PDL: bool = False
    PDL_RELEASE: bool = False
    PDL_PAIR: bool = False
    SMEM_OPTIN_BYTES: int = SM100_SMEM_OPTIN_BYTES
    SMEM_PER_SM_BYTES: int = SM100_SMEM_PER_SM_BYTES
    SM_COUNT: int = SM100_SM_COUNT
    GEOM: Geometry = DEFAULT_GEOMETRY

    # ---- geometry shorthands ----
    @property
    def H(self) -> int:
        return self.GEOM.HIDDEN

    @property
    def I(self) -> int:  # noqa: E743 - the intermediate width, as the op is written
        return self.GEOM.INTERMEDIATE

    @property
    def WARPS(self) -> int:
        return self.THREADS // 32

    # ---- launch 1: gate_up ----
    @property
    def GROUPS(self) -> int:
        """Row-pair groups (n8 tiles) of the geometry: ``I / 4``."""
        return self.I // PAIRS_PER_GROUP

    @property
    def N_GROUPS_EFF(self) -> int:
        """Groups per CTA: the knob, or the one-wave rule -- the two-group form (fewer, fatter CTAs: 32 KB of W each, two per SM) when its
        grid still places a CTA on every SM and fits one wave (measured the best form at I 256: 256 CTAs on 148-152 SMs), else the smallest
        of 1, 2 whose grid fits one wave (I 128: 256 one-group CTAs; the two-group grid of 128 would leave SMs idle), else 2."""
        if self.N_GROUPS:
            return self.N_GROUPS
        for n in (2, 1):
            form = replace(self, N_GROUPS=n)
            if self.GROUPS % n == 0 and form.GRID1 >= self.SM_COUNT and form.GRID1 <= self.SM_COUNT * form.co_resident_ctas1:
                return n
        for n in (1, 2):
            form = replace(self, N_GROUPS=n)
            if self.GROUPS % n == 0 and form.GRID1 <= self.SM_COUNT * form.co_resident_ctas1:
                return n
        return 2

    @property
    def CLUSTERS(self) -> int:
        """Clusters of launch 1 = ``GROUPS / N_GROUPS``."""
        return self.GROUPS // self.N_GROUPS_EFF

    @property
    def W_ROWS(self) -> int:
        """W_gu rows resident per CTA of launch 1: 8 per group."""
        return ROWS_PER_GROUP * self.N_GROUPS_EFF

    @property
    def GRID1(self) -> int:
        return self.CLUSTERS * self.K_SPLIT

    @property
    def WAVES1(self) -> int:
        """Launch-1 waves on ``SM_COUNT`` SMs at the form's shared-memory co-residency (registers are a ptxas fact checked on the SASS)."""
        slots = self.SM_COUNT * max(self.co_resident_ctas1, 1)
        return -(-self.GRID1 // slots)

    @property
    def K_SLICE(self) -> int:
        """K columns one CTA of launch 1 owns."""
        return self.H // self.K_SPLIT

    @property
    def W_PITCH_ELEMS(self) -> int:
        """bf16 elements between two W rows in shared memory (the slice plus the 16-B pad)."""
        return self.K_SLICE + PITCH_PAD_BYTES // 2

    @property
    def W_SMEM_BYTES(self) -> int:
        return self.W_ROWS * self.W_PITCH_ELEMS * 2

    @property
    def W_SLICE_BYTES(self) -> int:
        """Bytes of the W slice one CTA streams (the sum of its mbarriers' expect_tx)."""
        return self.W_ROWS * self.K_SLICE * 2

    @property
    def W_PHASE_BYTES(self) -> int:
        """expect_tx of ONE W mbarrier: ``W_ROWS`` row pieces of the phase's K range."""
        return self.W_SLICE_BYTES // self.W_PHASES

    @property
    def W_PHASE_ROW_BYTES(self) -> int:
        """Bytes of one bulk copy: one W row's piece of one phase."""
        return self.K_SLICE * 2 // self.W_PHASES

    @property
    def W_PHASE_K(self) -> int:
        """K columns of one W phase."""
        return self.K_SLICE // self.W_PHASES

    @property
    def TPAD(self) -> int:
        return MMA_M * self.TOKEN_TILES

    @property
    def X_ROWS(self) -> int:
        """Resident x rows of the bulk form: the form's tokens (16, or the 24-token ceiling of the two-tile form; rows past ``T`` are never
        loaded, every read of one is clamped to row 0)."""
        return min(self.TPAD, MAX_TOKENS)

    @property
    def X_ROW_PITCH_ELEMS(self) -> int:
        """bf16 elements between two x rows of the bulk form (the K slice plus the 16-B pad)."""
        return self.K_SLICE + PITCH_PAD_BYTES // 2

    @property
    def X_ROW_BYTES(self) -> int:
        """One bulk copy of the bulk form: a token's K slice."""
        return self.K_SLICE * 2

    @property
    def X_CHUNKS(self) -> int:
        """Ring chunks of the CTA's K slice."""
        return self.K_SLICE // CHUNK_K

    @property
    def CHUNKS_PER_W_PHASE(self) -> int:
        return self.X_CHUNKS // self.W_PHASES

    @property
    def X_PITCH_ELEMS(self) -> int:
        return CHUNK_K + PITCH_PAD_BYTES // 2

    @property
    def X_CHUNK_BYTES(self) -> int:
        """One ring stage: TPAD rows at the padded pitch."""
        return self.TPAD * self.X_PITCH_ELEMS * 2

    @property
    def X_COPIES(self) -> int:
        """16-B ``cp.async`` copies per thread per chunk (every thread commits every chunk; a dead row is a zero-filled copy)."""
        return self.TPAD * CHUNK_K * 2 // COPY_BYTES // self.THREADS

    @property
    def X_GRANULES_PER_ROW(self) -> int:
        return CHUNK_K * 2 // COPY_BYTES

    @property
    def ACC_TILES(self) -> int:
        """Accumulator tiles per lane: n-tiles x m-tiles (4 fp32 each)."""
        return self.N_GROUPS_EFF * self.TOKEN_TILES

    @property
    def RED_SMEM_BYTES(self) -> int:
        """The intra-CTA fold: warps 1 .. WARPS-1 publish 4 fp32 per lane per (n-tile, m-tile)."""
        return (self.WARPS - 1) * self.ACC_TILES * 32 * 4 * 4

    @property
    def MAIL_SMEM_BYTES(self) -> int:
        """The cluster fold's landing buffer on the leader: ranks 1 .. K_SPLIT-1 ship 4 fp32 per lane per (n-tile, m-tile) (16 B per lane)."""
        return (self.K_SPLIT - 1) * self.ACC_TILES * 32 * 4 * 4

    @property
    def GATE_CLUSTER_FOLD_EFF(self) -> bool:
        """The cluster-side gate fold needs a cluster (at K_SPLIT 1 one leaf per group is the slot itself: the two trees coincide)."""
        return bool(self.GATE_CLUSTER_FOLD) and self.K_SPLIT > 1

    @property
    def GMAIL_SMEM_BYTES(self) -> int:
        """The gate leaves' landing buffer on the leader: ranks 1 .. K_SPLIT-1 ship one fp32 per token (TPAD rows) and group."""
        return (self.K_SPLIT - 1) * self.N_GROUPS_EFF * self.TPAD * 4 if self.GATE_CLUSTER_FOLD_EFF else 0

    @property
    def KRED_SPLIT_EFF(self) -> bool:
        """The split cluster-fold barrier needs a gate ship to separate (the cluster-side gate fold); off otherwise."""
        return bool(self.KRED_SPLIT) and self.GATE_CLUSTER_FOLD_EFF

    @property
    def KRED_BYTES(self) -> int:
        """expect_tx of the leader's ``mb_kred``: every peer's accumulator v4s and, under the cluster fold WITHOUT the split, the gate leaves."""
        return self.MAIL_SMEM_BYTES + (0 if self.KRED_SPLIT_EFF else self.GMAIL_SMEM_BYTES)

    @property
    def KRED_GATE_BYTES(self) -> int:
        """expect_tx of the leader's ``mb_kred_gate`` (the split form): every peer's gate leaves; 0 without the split."""
        return self.GMAIL_SMEM_BYTES if self.KRED_SPLIT_EFF else 0

    @property
    def KRED_GATE_GROUP_BYTES(self) -> int:
        """One gate warp's share of ``mb_kred_gate``'s expect: the ``K_SPLIT - 1`` peers' leaves of its group (``TPAD`` fp32 each)."""
        return (self.K_SPLIT - 1) * self.TPAD * 4 if self.KRED_SPLIT_EFF else 0

    @property
    def KRED_GATE_MBAR_ARRIVERS(self) -> int:
        """Init count of ``mb_kred_gate``: one arriving lane per gate warp (lane 0 of warps ``1 .. N_GROUPS``), each with its group's expect."""
        return self.N_GROUPS_EFF if self.KRED_SPLIT_EFF else 0

    @property
    def MBAR_SMEM_BYTES(self) -> int:
        """``mb_w`` (W_PHASES slots), ``mb_x`` (bulk form), ``mb_kred`` (K_SPLIT > 1) and ``mb_kred_gate`` (KRED_SPLIT) in 16-B-aligned arrays."""
        return (self.W_PHASES * 8 + 15) // 16 * 16 + (16 if self.X_BULK_ROWS else 0) + (16 if self.K_SPLIT > 1 else 0) + (16 if self.KRED_SPLIT_EFF else 0)

    @property
    def SMEM1_FIXED_BYTES(self) -> int:
        """Launch 1's shared memory without x."""
        return self.W_SMEM_BYTES + self.RED_SMEM_BYTES + self.MAIL_SMEM_BYTES + self.GMAIL_SMEM_BYTES + self.MBAR_SMEM_BYTES

    @property
    def SMEM1_BUDGET_BYTES(self) -> int:
        """What launch 1 may allocate: the per-CTA opt-in, or the SM's share of ``MIN_CTAS_PER_SM`` co-resident CTAs (each with the
        driver's reserve), whichever is smaller."""
        return min(self.SMEM_OPTIN_BYTES, self.SMEM_PER_SM_BYTES // self.MIN_CTAS_PER_SM - SMEM_CTA_RESERVE_BYTES)

    @property
    def X_STAGES_AUTO(self) -> int:
        """The budget rule: every chunk resident when they fit, else the deepest ring that does (the validator pins >= 2)."""
        free = self.SMEM1_BUDGET_BYTES - self.SMEM1_FIXED_BYTES - SMEM_RING_MARGIN_BYTES
        return min(self.X_CHUNKS, free // self.X_CHUNK_BYTES) if self.X_CHUNK_BYTES > 0 else 0

    @property
    def X_STAGES_EFF(self) -> int:
        """The ring depth (0 in the bulk form: no ring)."""
        if self.X_BULK_ROWS:
            return 0
        return self.X_STAGES if self.X_STAGES else self.X_STAGES_AUTO

    @property
    def X_SMEM_BYTES(self) -> int:
        if self.X_BULK_ROWS:
            return self.X_ROWS * self.X_ROW_PITCH_ELEMS * 2
        return self.X_STAGES_EFF * self.X_CHUNK_BYTES

    @property
    def SMEM1_BYTES(self) -> int:
        return self.SMEM1_FIXED_BYTES + self.X_SMEM_BYTES

    @property
    def X_PROLOGUE_GROUPS(self) -> int:
        """``cp.async`` groups committed before the mainloop (chunks ``0 .. X_STAGES - 2``)."""
        return self.X_STAGES_EFF - 1

    @property
    def X_WAIT_PENDING(self) -> int:
        """The ``cp.async.wait_group`` immediate at the top of every iteration (groups that may stay in flight)."""
        return self.X_STAGES_EFF - 2

    # ---- the distributed gate logit ----
    @property
    def GATE_PARTS(self) -> int:
        """Leaves of ``x @ W_g^T`` per token = ``K_SPLIT x GROUPS`` (CTA ``(c, ks)``, group ``n`` owns leaf ``ks GROUPS + c N_GROUPS + n``);
        independent of ``N_GROUPS``."""
        return self.K_SPLIT * self.GROUPS

    @property
    def GATE_COLS(self) -> int:
        """Columns of one leaf: the K slice split over the groups."""
        return self.H // self.GATE_PARTS

    @property
    def GATE_PAIRS_PER_LANE(self) -> int:
        """Column pairs each of a gate warp's 32 lanes walks (lane ``l`` owns columns ``2l + 64p``, ``p < GATE_PAIRS_PER_LANE``, predicated on
        ``2l + 64p < GATE_COLS``; a dead pair adds nothing)."""
        return (self.GATE_COLS + 63) // 64

    @property
    def GATE_SLOTS(self) -> int:
        """Partials per token in the ``gate_part`` scratch: the ``GROUPS`` cluster-folded slots, or every leaf on the v1 path."""
        return self.GROUPS if self.GATE_CLUSTER_FOLD_EFF else self.GATE_PARTS

    @property
    def GATE_SLOTS_PER_LANE(self) -> int:
        """Launch 2's fold: lane ``l`` reads slots ``l x GATE_SLOTS_PER_LANE ..`` of a token (``GATE_SLOTS / 32``)."""
        return self.GATE_SLOTS // 32

    # ---- launch 2: down ----
    @property
    def THREADS2(self) -> int:
        return self.ROWS_PER_CTA // MMA_N * THREADS_PER_ROW_TILE

    @property
    def WARPS2(self) -> int:
        return self.THREADS2 // 32

    @property
    def GRID2(self) -> int:
        return self.H // self.ROWS_PER_CTA

    @property
    def WD_PITCH_ELEMS(self) -> int:
        return self.I + PITCH_PAD_BYTES // 2

    @property
    def WD_GRANULES_PER_ROW(self) -> int:
        return self.I * 2 // COPY_BYTES

    @property
    def WD_COPIES(self) -> int:
        """16-B ``cp.async`` copies per thread of the CTA's W_d rows (group W)."""
        return self.ROWS_PER_CTA * self.I * 2 // COPY_BYTES // self.THREADS2

    @property
    def ACT_COPIES(self) -> int:
        """16-B ``cp.async`` copies per thread of the ``TPAD`` act rows (group A; rows past T zero-filled)."""
        return self.TPAD * self.I * 2 // COPY_BYTES // self.THREADS2

    @property
    def K_STEPS2(self) -> int:
        return self.I // MMA_K

    @property
    def TOKENS_PER_WARP2(self) -> int:
        """Gate-fold tokens per warp of launch 2 (token ``w + WARPS2 x j``)."""
        return self.TPAD // self.WARPS2

    @property
    def WD_SMEM_BYTES(self) -> int:
        return self.ROWS_PER_CTA * self.WD_PITCH_ELEMS * 2

    @property
    def ACT_SMEM_BYTES(self) -> int:
        return self.TPAD * self.WD_PITCH_ELEMS * 2

    @property
    def SIG_SMEM_BYTES(self) -> int:
        return SCRATCH_ROWS * 4

    @property
    def SMEM2_BYTES(self) -> int:
        return self.WD_SMEM_BYTES + self.ACT_SMEM_BYTES + self.SIG_SMEM_BYTES

    @property
    def co_resident_ctas1(self) -> int:
        """CTAs of launch 1 the part's shared memory holds per SM (each with the driver's reserve)."""
        return self.SMEM_PER_SM_BYTES // (self.SMEM1_BYTES + SMEM_CTA_RESERVE_BYTES)


def _pitch_ok(pitch_bytes: int) -> bool:
    """A row pitch of ``128 n + 16`` bytes: the eight row addresses of one ``ldmatrix`` phase cover all 32 banks."""
    return pitch_bytes % 128 == PITCH_PAD_BYTES


def validate_cfg(cfg: Cfg) -> None:
    """Raise the typed error of every unexpressible knob combination (structural invariants raise here, never a module assert)."""
    e = f"{NAME}:"
    if not isinstance(cfg.GEOM, Geometry):
        raise ValueError(f"{e} GEOM must be a Geometry, got {type(cfg.GEOM).__name__}")
    if cfg.THREADS != 256:
        raise ValueError(f"{e} THREADS must be 256 (eight warps x four k16 steps = the 32 steps of a {CHUNK_K}-column chunk), got {cfg.THREADS}")
    if cfg.K_SPLIT not in (1, 2, 4, 8):
        raise ValueError(f"{e} K_SPLIT must be 1, 2, 4 or 8 (a cluster of at most {MAX_K_SPLIT}), got {cfg.K_SPLIT}")
    if cfg.TOKEN_TILES not in (1, 2):
        raise ValueError(f"{e} TOKEN_TILES must be 1 or 2 (16 or 32 token rows; T <= {MAX_TOKENS}), got {cfg.TOKEN_TILES}")
    if cfg.N_GROUPS not in (0, 1, 2):
        raise ValueError(f"{e} N_GROUPS must be 0 (the one-wave rule), 1 or 2, got {cfg.N_GROUPS}")
    if cfg.W_PHASES not in (1, 2):
        raise ValueError(f"{e} W_PHASES must be 1 or 2, got {cfg.W_PHASES}")
    if cfg.ROWS_PER_CTA not in (32, 64):
        raise ValueError(f"{e} ROWS_PER_CTA must be 32 or 64, got {cfg.ROWS_PER_CTA}")
    if cfg.MIN_CTAS_PER_SM not in (1, 2):
        raise ValueError(f"{e} MIN_CTAS_PER_SM must be 1 or 2, got {cfg.MIN_CTAS_PER_SM}")
    if isinstance(cfg.SM_COUNT, bool) or not isinstance(cfg.SM_COUNT, int) or cfg.SM_COUNT < 1:
        raise ValueError(f"{e} SM_COUNT must be a positive int (the part's multiprocessor count), got {cfg.SM_COUNT!r}")
    if cfg.SMEM_PER_SM_BYTES <= 0 or cfg.SMEM_OPTIN_BYTES <= 0 or cfg.SMEM_OPTIN_BYTES > cfg.SMEM_PER_SM_BYTES:
        raise ValueError(
            f"{e} SMEM_PER_SM_BYTES={cfg.SMEM_PER_SM_BYTES} / SMEM_OPTIN_BYTES={cfg.SMEM_OPTIN_BYTES} must be positive with the per-CTA opt-in no larger than the SM"
        )
    H, I = cfg.H, cfg.I
    if I % PAIRS_PER_GROUP or I % MMA_K:
        raise ValueError(f"{e} INTERMEDIATE={I} must be a multiple of {PAIRS_PER_GROUP} (row pairs per group) and of {MMA_K} (launch 2's k16 steps)")
    if cfg.GROUPS % cfg.N_GROUPS_EFF:
        raise ValueError(f"{e} N_GROUPS={cfg.N_GROUPS_EFF} must divide the {cfg.GROUPS} row-pair groups of INTERMEDIATE={I}")
    if H % (cfg.K_SPLIT * CHUNK_K):
        raise ValueError(f"{e} HIDDEN={H} must be a multiple of K_SPLIT x {CHUNK_K} = {cfg.K_SPLIT * CHUNK_K} (whole {CHUNK_K}-column chunks per K slice)")
    if cfg.X_CHUNKS % cfg.W_PHASES:
        raise ValueError(f"{e} W_PHASES={cfg.W_PHASES} must divide the {cfg.X_CHUNKS} chunks of the K slice")
    if cfg.W_PHASE_ROW_BYTES % 16 or cfg.W_PHASE_K % MMA_K:
        raise ValueError(
            f"{e} a W phase piece must be a 16-byte multiple and a whole number of k16 steps; got {cfg.W_PHASE_ROW_BYTES} B / {cfg.W_PHASE_K} columns"
        )
    if cfg.W_ROWS > 32:
        raise ValueError(f"{e} W_ROWS={cfg.W_ROWS} must not exceed the 32 issuing lanes of warp 0 (one bulk copy per row and phase)")
    if H % cfg.ROWS_PER_CTA:
        raise ValueError(f"{e} HIDDEN={H} must be a multiple of ROWS_PER_CTA={cfg.ROWS_PER_CTA}")
    if cfg.TPAD * CHUNK_K * 2 % (COPY_BYTES * cfg.THREADS):
        raise ValueError(f"{e} the x chunk ({cfg.TPAD} x {CHUNK_K} bf16) does not split into 16-B copies over {cfg.THREADS} threads")
    if (cfg.ROWS_PER_CTA * I * 2) % (COPY_BYTES * cfg.THREADS2) or (cfg.TPAD * I * 2) % (COPY_BYTES * cfg.THREADS2):
        raise ValueError(
            f"{e} the W_d rows ({cfg.ROWS_PER_CTA} x {I}) or the act rows ({cfg.TPAD} x {I}) do not split into 16-B copies over {cfg.THREADS2} threads"
        )
    if cfg.TPAD % cfg.WARPS2:
        raise ValueError(f"{e} the {cfg.TPAD} gate-fold tokens do not split over the {cfg.WARPS2} warps of launch 2")
    if 1 + cfg.N_GROUPS_EFF > cfg.WARPS:
        raise ValueError(f"{e} the {cfg.N_GROUPS_EFF} gate warps (1 ..) and warp 0 exceed the {cfg.WARPS} warps")
    # the distributed gate logit: the partition is derived from the grid and must tile H exactly
    if cfg.GATE_PARTS * cfg.GATE_COLS != H:
        raise ValueError(f"{e} GATE_PARTS={cfg.GATE_PARTS} x GATE_COLS={cfg.GATE_COLS} must equal HIDDEN={H}")
    if cfg.GATE_COLS % 2:
        raise ValueError(f"{e} GATE_COLS={cfg.GATE_COLS} must be even (one bf16x2 word per lane and pair)")
    if cfg.GATE_SLOTS % 32:
        raise ValueError(f"{e} GATE_SLOTS={cfg.GATE_SLOTS} must be a multiple of 32 (launch 2's lane fold reads GATE_SLOTS / 32 partials per lane)")
    if cfg.KRED_BYTES + cfg.KRED_GATE_BYTES != cfg.MAIL_SMEM_BYTES + cfg.GMAIL_SMEM_BYTES:
        raise ValueError(
            f"{e} the cluster-fold ledger mb_kred {cfg.KRED_BYTES} + mb_kred_gate {cfg.KRED_GATE_BYTES} != mail {cfg.MAIL_SMEM_BYTES} + gmail {cfg.GMAIL_SMEM_BYTES}"
        )
    if cfg.KRED_GATE_GROUP_BYTES * cfg.KRED_GATE_MBAR_ARRIVERS != cfg.KRED_GATE_BYTES:
        raise ValueError(
            f"{e} the mb_kred_gate ledger: {cfg.KRED_GATE_MBAR_ARRIVERS} arriving lanes x {cfg.KRED_GATE_GROUP_BYTES} B != {cfg.KRED_GATE_BYTES} B"
        )
    # the pitches
    pitches = [("w_s", cfg.W_PITCH_ELEMS * 2), ("wd_s / act_s", cfg.WD_PITCH_ELEMS * 2)]
    pitches.append(("x_s", cfg.X_ROW_PITCH_ELEMS * 2) if cfg.X_BULK_ROWS else ("x_s", cfg.X_PITCH_ELEMS * 2))
    for name, pitch in pitches:
        if not _pitch_ok(pitch):
            raise ValueError(f"{e} the {name} row pitch {pitch} B is not 128 n + 16 (the ldmatrix bank cover)")
    if cfg.X_BULK_ROWS:
        if cfg.X_ROWS > 32 or cfg.X_ROW_BYTES % 16:
            raise ValueError(
                f"{e} the bulk form needs <= 32 resident x rows (one issuing lane each) and a 16-B-multiple row; got {cfg.X_ROWS} rows of {cfg.X_ROW_BYTES} B"
            )
        if cfg.SMEM1_BYTES > cfg.SMEM_OPTIN_BYTES:
            raise ValueError(
                f"{e} launch 1 shared memory {cfg.SMEM1_BYTES} B (x resident: {cfg.X_ROWS} rows of {cfg.X_ROW_BYTES} B) exceeds the per-CTA opt-in "
                f"{cfg.SMEM_OPTIN_BYTES} B -- a ring form (X_BULK_ROWS off) or a larger K_SPLIT"
            )
    else:
        # the ring ledger needs two stages (one landing while one is read) and never more than the chunks
        stages = cfg.X_STAGES_EFF
        if stages < 2 or stages > cfg.X_CHUNKS:
            raise ValueError(
                f"{e} X_STAGES={cfg.X_STAGES} (effective {stages}) must be in [2, X_CHUNKS={cfg.X_CHUNKS}]: launch 1 needs {cfg.SMEM1_FIXED_BYTES} B fixed + "
                f"{cfg.X_CHUNK_BYTES} B per stage within the {cfg.SMEM1_BUDGET_BYTES}-byte budget"
            )
        if cfg.SMEM1_BYTES > cfg.SMEM_OPTIN_BYTES:
            raise ValueError(f"{e} launch 1 shared memory {cfg.SMEM1_BYTES} B exceeds the per-CTA opt-in {cfg.SMEM_OPTIN_BYTES} B")
        if cfg.MIN_CTAS_PER_SM * (cfg.SMEM1_BYTES + SMEM_CTA_RESERVE_BYTES) > cfg.SMEM_PER_SM_BYTES:
            raise ValueError(f"{e} MIN_CTAS_PER_SM={cfg.MIN_CTAS_PER_SM} CTAs of {cfg.SMEM1_BYTES} B do not fit the SM's {cfg.SMEM_PER_SM_BYTES} B")
    if cfg.SMEM2_BYTES > cfg.SMEM_OPTIN_BYTES:
        raise ValueError(f"{e} launch 2 shared memory {cfg.SMEM2_BYTES} B exceeds the per-CTA opt-in {cfg.SMEM_OPTIN_BYTES} B")


def validate_one_wave(cfg: Cfg) -> None:
    """The default forms' pin: launch 1 fits ONE wave on the part (a knob arm may choose two waves knowingly; ``default_cfg`` / ``form_for``
    never do)."""
    if cfg.WAVES1 != 1:
        raise ValueError(
            f"{NAME}: the form (K_SPLIT {cfg.K_SPLIT}, TOKEN_TILES {cfg.TOKEN_TILES}, N_GROUPS {cfg.N_GROUPS_EFF}) needs {cfg.WAVES1} waves: {cfg.GRID1} CTAs of "
            f"{cfg.SMEM1_BYTES} B at {cfg.co_resident_ctas1} per SM on {cfg.SM_COUNT} SMs"
        )


def make_cfg(**overrides) -> Cfg:
    """A validated ``Cfg``."""
    cfg = replace(Cfg(), **overrides)
    validate_cfg(cfg)
    return cfg


def k_split_for(geometry: Geometry) -> int:
    """The pinned K_SPLIT of a geometry -- a function of the GEOMETRY ALONE (the rounding class of every output element never depends on the
    part): the largest cluster (8 on the validated envelope: 1024-column K slices) whose K slices are whole 512-column chunks and whose gate
    leaves keep an even column count (one bf16x2 word per lane and pair).  A part too small to hold the pinned form in one wave is declined
    by type (``validate_one_wave``), never served with another pin."""
    groups = geometry.INTERMEDIATE // PAIRS_PER_GROUP
    for s in (8, 4, 2, 1):
        if geometry.HIDDEN % (s * CHUNK_K) == 0 and (geometry.HIDDEN // (s * groups)) % 2 == 0:
            return s
    return 1


def default_cfg(
    geometry: Geometry = DEFAULT_GEOMETRY,
    smem_per_sm: int = SM100_SMEM_PER_SM_BYTES,
    smem_optin: int = SM100_SMEM_OPTIN_BYTES,
    sm_count: int = SM100_SM_COUNT,
) -> Cfg:
    """The default form of a geometry on a part: the module defaults with the geometry's pinned ``K_SPLIT`` and the part's readings; one wave
    (a part whose SM count cannot hold the pinned form in one wave is the typed error of :func:`validate_one_wave`: the pin never moves)."""
    if not isinstance(geometry, Geometry):
        raise ValueError(f"{NAME}: geometry must be a Geometry, got {type(geometry).__name__}")
    cfg = make_cfg(GEOM=geometry, K_SPLIT=k_split_for(geometry), **part_readings(smem_per_sm, smem_optin, sm_count))
    validate_one_wave(cfg)
    return cfg


def part_readings(smem_per_sm: int, smem_optin: int, sm_count: int = SM100_SM_COUNT) -> dict:
    """The ``Cfg`` fields of a part reading ``(shared memory per SM, per-CTA opt-in, multiprocessors)``, validated."""
    for name, value in (("smem_per_sm", smem_per_sm), ("smem_optin", smem_optin), ("sm_count", sm_count)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{NAME}: {name} must be a positive int (a device reading), got {value!r}")
    if smem_optin > smem_per_sm:
        raise ValueError(f"{NAME}: the per-CTA shared-memory opt-in ({smem_optin} B) cannot exceed the SM's ({smem_per_sm} B)")
    return dict(SMEM_PER_SM_BYTES=smem_per_sm, SMEM_OPTIN_BYTES=smem_optin, SM_COUNT=sm_count)


def token_tiles_for(tokens: int) -> int:
    """The form of a token count: one m-tile up to 16 tokens, two up to ``MAX_TOKENS``."""
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 1:
        raise ValueError(f"{NAME}: tokens must be a positive int, got {tokens!r}")
    if tokens > MAX_TOKENS:
        raise NotImplementedError(f"{NAME} serves at most {MAX_TOKENS} tokens per call (the two-m-tile form's 25 .. 32 band is unmeasured); got {tokens}")
    return 1 if tokens <= MMA_M else 2


def form_for(cfg: Cfg, tokens: int) -> Cfg:
    """``cfg`` with the token count's form (``TOKEN_TILES``; ``N_GROUPS`` follows through the one-wave rule when the knob is 0), validated;
    a form under the one-wave rule that still needs two waves is the typed error of :func:`validate_one_wave`."""
    tiles = token_tiles_for(tokens)
    form = cfg if cfg.TOKEN_TILES == tiles else replace(cfg, TOKEN_TILES=tiles)
    validate_cfg(form)
    if form.N_GROUPS == 0:
        validate_one_wave(form)
    return form


def ring_ledger(x_chunks: int, x_stages: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """A pure-Python walk of launch 1's ``cp.async`` ring (P18): returns ``(landed_before_iteration, stage_refilled_at)``.

    ``landed_before_iteration[i]`` = the number of chunks guaranteed landed when iteration ``i`` passes its ``wait_group``
    (``committed - pending`` with ``committed = (x_stages - 1) + i`` and ``pending <= x_stages - 2``), which must be ``>= i + 1``;
    ``stage_refilled_at[i]`` = the stage iteration ``i`` refills (``(i + x_stages - 1) % x_stages``, or -1 past the chunks), which must be the
    stage iteration ``i - 1`` read (``(i - 1) % x_stages``) -- so the refill sits behind iteration ``i``'s barrier, after that stage's last read."""
    if x_stages < 2 or x_stages > x_chunks:
        raise ValueError(f"{NAME}: the ring needs 2 <= X_STAGES <= X_CHUNKS, got {x_stages} / {x_chunks}")
    landed, refilled = [], []
    for i in range(x_chunks):
        committed = (x_stages - 1) + i
        landed.append(committed - (x_stages - 2))
        nxt = i + x_stages - 1
        refilled.append((nxt % x_stages) if nxt < x_chunks else -1)
    return tuple(landed), tuple(refilled)


def smem_table1(cfg: Cfg) -> dict:
    """Launch 1's SMEM buffer table in allocation order: ``{name: (bytes, row pitch bytes or None)}`` (the host tests pin it)."""
    return {
        "w_s": (cfg.W_SMEM_BYTES, cfg.W_PITCH_ELEMS * 2),
        "x_s": (cfg.X_SMEM_BYTES, (cfg.X_ROW_PITCH_ELEMS if cfg.X_BULK_ROWS else cfg.X_PITCH_ELEMS) * 2),
        "red_s": (cfg.RED_SMEM_BYTES, 16),
        "mail_s": (cfg.MAIL_SMEM_BYTES, 16),
        "gmail_s": (cfg.GMAIL_SMEM_BYTES, 4),
        "mbarriers": (cfg.MBAR_SMEM_BYTES, None),
    }


def smem_table2(cfg: Cfg) -> dict:
    """Launch 2's SMEM buffer table."""
    return {
        "wd_s": (cfg.WD_SMEM_BYTES, cfg.WD_PITCH_ELEMS * 2),
        "act_s": (cfg.ACT_SMEM_BYTES, cfg.WD_PITCH_ELEMS * 2),
        "sig_s": (cfg.SIG_SMEM_BYTES, None),
    }


def v1_cfg(geometry: Geometry = DEFAULT_GEOMETRY, **overrides) -> Cfg:
    """The v1 form of a geometry (the rounding class of the ``ksplit2`` / ``ksplit4`` goldens): K_SPLIT 2 at I 256 / 4 at I 128, the x ring,
    the launch-2 gate fold, one group per CTA -- kept compilable as the A/B arm."""
    groups = geometry.INTERMEDIATE // PAIRS_PER_GROUP
    k_split = 4 if groups * 4 <= 128 else 2 if groups * 2 <= 128 else 1
    return make_cfg(GEOM=geometry, K_SPLIT=k_split, N_GROUPS=1, X_BULK_ROWS=False, GATE_CLUSTER_FOLD=False, **overrides)


# the module forms (the cc 10.x readings; the host entry stamps the part's)
CFG = default_cfg()  # I 256: K_SPLIT 8, N_GROUPS 2: 256 CTAs x 81,328 B (2 per SM) at TOKEN_TILES 1, 256 x 113,072 B at TOKEN_TILES 2
CFG_TP16 = default_cfg(Geometry(8192, 128))  # I 128: K_SPLIT 8, 256 CTAs x 57,200 B; 256 x 81,328 B at TOKEN_TILES 2
CFG_V1 = v1_cfg()  # I 256: K_SPLIT 2, 128 CTAs x 202,912 B (the v1 goldens' class)
CFG_V1_TP16 = v1_cfg(Geometry(8192, 128))  # I 128: K_SPLIT 4, 128 CTAs x 104,608 B
