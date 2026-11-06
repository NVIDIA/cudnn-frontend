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

* launch 1 ``gate_up`` -- ``(I / 4) x K_SPLIT`` CTAs in clusters of ``K_SPLIT`` along K: cluster ``c`` owns the four row PAIRS ``(j, I + j)``,
  ``j = 4c .. 4c + 3`` (one ``m16n8k16`` n-tile whose n columns are ordered gate_0, up_0, .., gate_3, up_3 by where the eight bulk copies
  LAND in shared memory), CTA rank ``ks`` the K slice ``[ks H/K_SPLIT, (ks+1) H/K_SPLIT)``; 256 threads; the W slice streams once by eight
  ``cp.async.bulk`` row copies on ``W_PHASES`` mbarriers, ``x`` streams through a ``cp.async`` ring of ``CHUNK_K``-column chunks (rows past
  ``T`` zero-filled), warp ``w`` runs k16 steps ``4w .. 4w + 3`` of every chunk, the warps' accumulators fold through shared memory in warp
  order, the cluster's through one DSMEM ``st.async`` ship per lane and m-tile onto the leader's ``mb_kred``; the leader's warp 0 writes
  ``act[t][4c + i] = bf16(silu(gate) * up)`` for ``t < T``.  Warp 1 of every CTA also computes its ``GATE_COLS`` columns of the gate logit
  ``x @ W_g^T`` (partial ``q = ks (I/4) + c`` of ``GATE_PARTS``), so no CTA serializes a GEMV.
* launch 2 ``down`` -- ``H / ROWS_PER_CTA`` CTAs of ``4 x ROWS_PER_CTA`` threads: warp ``w`` owns output rows ``n0 + 8w ..`` over the FULL
  K = I (one accumulator chain per output element, no cross-warp reduction); the W_d rows prefetch by ``cp.async`` BEFORE
  ``griddepcontrol.wait``, ``act`` and the gate partials load after it, warp ``w`` folds the ``GATE_PARTS`` partials of tokens
  ``w, w + WARPS, ..`` into ``sigmoid`` in shared memory; the epilogue writes ``y[t][n] = bf16(acc x sigmoid[t])`` for ``t < T``.

Knobs (every one a ``cutlass.const_expr`` axis of the same two kernel bodies; the default is the one-CTA-per-SM form that keeps the whole
8 MB / 4 MB weight stream in flight within the first microsecond):

* ``K_SPLIT`` (1, 2, 4): CTAs per cluster over K -- a ROUNDING-ORDER axis (the slice sum ``((C_0 + C_1) + ..)`` and the gate-logit tree,
  ``GATE_PARTS`` leaves, both move with it); pinned per geometry by ``default_cfg`` (2 at I 256: 128 CTAs of 64 KB; 4 at I 128: 128 CTAs of
  32 KB) so the slice order of every output element is a function of the geometry alone, never of ``T``.
* ``TOKEN_TILES`` (1, 2): m-tiles of 16 tokens -- the FORM axis the host entry picks from ``T`` (1 for ``T <= 16``, 2 for ``17 .. 24``);
  bit-neutral by the row-position-invariance premise of section 4 of the design (a token's row of the A operand is contracted on its own).
* ``X_STAGES`` (0 = the budget rule: every chunk resident when the shared memory holds them, else the deepest ring that fits; >= 2 always):
  the ``cp.async`` ring depth of launch 1.  Bit-neutral.
* ``W_PHASES`` (1, 2): mbarriers over the W slice's K halves so the first k-steps start on the first half.  Bit-neutral.
* ``ROWS_PER_CTA`` (64, 32): W_d rows per CTA of launch 2 (128 / 256 CTAs; 256 / 128 threads).  Bit-neutral: a row's K order does not move.
* ``MIN_CTAS_PER_SM`` (1, 2): the shared-memory budget launch 1's ring is sized against (the per-CTA opt-in, or half the SM for a form that
  must co-reside twice, e.g. ``K_SPLIT`` 4 at I 256: 256 CTAs on 152 SMs); the validator pins the co-residency it claims.
* ``PDL`` / ``PDL_RELEASE`` / ``PDL_PAIR``: ``PDL`` = launch 1's programmatic-stream-serialization attribute (its predecessor is the norm that
  writes ``x``), ``PDL_PAIR`` = launch 2's (its predecessor is launch 1), ``PDL_RELEASE`` = ``griddepcontrol.launch_dependents`` at the entry
  of both kernels.  Both kernel bodies take ``griddepcontrol.wait`` UNCONDITIONALLY (a no-op without the attribute), so one cubin serves the
  programmatic and the plain arm; the attributes are launch facts.  CALLER OBLIGATION under ``PDL``: the launch's immediate stream
  predecessor must not write ``w_gate_up`` / ``w_gate`` (they load before the wait; the serving chain's weights are static).  Under
  ``PDL_PAIR`` the obligation is launch 1's own (``w_down`` is static).  Every default is off: a knob measured before it defaults on.
* ``SMEM_OPTIN_BYTES`` / ``SMEM_PER_SM_BYTES``: the part's readings (the host entry replaces the cc 10.x module constants with the device's).
"""

from dataclasses import dataclass, replace
from typing import Tuple

# --- fixed kernel constants ---
MMA_M = 16  # tokens per m-tile (mma.sync.m16n8k16)
MMA_N = 8  # weight rows per n-tile
MMA_K = 16  # k per mma step
PAIRS_PER_CLUSTER = 4  # (gate, up) row pairs per cluster = one n8 tile of launch 1
W_ROWS = 2 * PAIRS_PER_CLUSTER  # 8 W_gu rows resident per CTA of launch 1
CHUNK_K = 512  # x columns per ring chunk of launch 1
PITCH_PAD_BYTES = 16  # every shared-memory row pitch is 128 n + 16 B: consecutive rows shift by 4 banks (ldmatrix 8-row phases cover 32)
COPY_BYTES = 16  # one cp.async granule
MAX_TOKENS = 24  # the served T ceiling (the 25 .. 32 band of the two-tile form is unmeasured and declined)
SCRATCH_ROWS = 32  # rows of the entry-owned act / gate_part scratch (two m-tiles)
THREADS_PER_ROW_TILE = 32  # launch 2: one warp per n8 tile of ROWS_PER_CTA rows
# barrier init counts (SUM(issuing lanes) == init on every path): one elected lane arms each tx mbarrier, the copies complete the bytes
W_MBAR_ARRIVERS = 1
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
    K_SPLIT: int = 2
    TOKEN_TILES: int = 1
    X_STAGES: int = 0
    W_PHASES: int = 1
    ROWS_PER_CTA: int = 64
    MIN_CTAS_PER_SM: int = 1
    PDL: bool = False
    PDL_RELEASE: bool = False
    PDL_PAIR: bool = False
    SMEM_OPTIN_BYTES: int = SM100_SMEM_OPTIN_BYTES
    SMEM_PER_SM_BYTES: int = SM100_SMEM_PER_SM_BYTES
    GEOM: Geometry = DEFAULT_GEOMETRY

    # ---- geometry shorthands ----
    @property
    def H(self) -> int:
        return self.GEOM.HIDDEN

    @property
    def I(self) -> int:  # noqa: E743 - the design's name for the intermediate width
        return self.GEOM.INTERMEDIATE

    @property
    def WARPS(self) -> int:
        return self.THREADS // 32

    # ---- launch 1: gate_up ----
    @property
    def CLUSTERS(self) -> int:
        """Clusters of launch 1 = row-pair groups: ``I / 4``."""
        return self.I // PAIRS_PER_CLUSTER

    @property
    def GRID1(self) -> int:
        return self.CLUSTERS * self.K_SPLIT

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
        return W_ROWS * self.W_PITCH_ELEMS * 2

    @property
    def W_SLICE_BYTES(self) -> int:
        """Bytes of the W slice one CTA streams (the sum of its mbarriers' expect_tx)."""
        return W_ROWS * self.K_SLICE * 2

    @property
    def W_PHASE_BYTES(self) -> int:
        """expect_tx of ONE W mbarrier: eight row pieces of the phase's K range."""
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
    def RED_SMEM_BYTES(self) -> int:
        """The intra-CTA fold: warps 1 .. WARPS-1 publish 4 fp32 per lane per m-tile."""
        return (self.WARPS - 1) * self.TOKEN_TILES * 32 * 4 * 4

    @property
    def MAIL_SMEM_BYTES(self) -> int:
        """The cluster fold's landing buffer on the leader: ranks 1 .. K_SPLIT-1 ship 4 fp32 per lane per m-tile (16 B per lane)."""
        return (self.K_SPLIT - 1) * self.TOKEN_TILES * 32 * 4 * 4

    @property
    def KRED_BYTES(self) -> int:
        """expect_tx of the leader's ``mb_kred`` = every peer's ship: (K_SPLIT - 1) x TOKEN_TILES x 32 lanes x 16 B."""
        return self.MAIL_SMEM_BYTES

    @property
    def MBAR_SMEM_BYTES(self) -> int:
        """``mb_w`` (W_PHASES slots) and ``mb_kred`` in 16-B-aligned arrays."""
        return (self.W_PHASES * 8 + 15) // 16 * 16 + (16 if self.K_SPLIT > 1 else 0)

    @property
    def SMEM1_FIXED_BYTES(self) -> int:
        """Launch 1's shared memory without the x ring."""
        return self.W_SMEM_BYTES + self.RED_SMEM_BYTES + self.MAIL_SMEM_BYTES + self.MBAR_SMEM_BYTES

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
        return self.X_STAGES if self.X_STAGES else self.X_STAGES_AUTO

    @property
    def X_SMEM_BYTES(self) -> int:
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
        """Partials of ``x @ W_g^T`` per token = the CTAs of launch 1 (CTA ``(c, ks)`` owns partial ``ks x CLUSTERS + c``)."""
        return self.GRID1

    @property
    def GATE_COLS(self) -> int:
        """Columns of one partial: the CTA's K slice split over the clusters."""
        return self.H // self.GATE_PARTS

    @property
    def GATE_PAIRS_PER_LANE(self) -> int:
        """Column pairs each of warp 1's 32 lanes walks (lane ``l`` owns columns ``2l + 64p``, ``p < GATE_PAIRS_PER_LANE``, predicated on
        ``2l + 64p < GATE_COLS``; a dead pair adds nothing)."""
        return (self.GATE_COLS + 63) // 64

    @property
    def GATE_PARTS_PER_LANE(self) -> int:
        """Launch 2's fold: lane ``l`` reads partials ``l x GATE_PARTS_PER_LANE ..`` of a token (``GATE_PARTS / 32``)."""
        return self.GATE_PARTS // 32

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
    if cfg.K_SPLIT not in (1, 2, 4):
        raise ValueError(f"{e} K_SPLIT must be 1, 2 or 4, got {cfg.K_SPLIT}")
    if cfg.TOKEN_TILES not in (1, 2):
        raise ValueError(f"{e} TOKEN_TILES must be 1 or 2 (16 or 32 token rows; T <= {MAX_TOKENS}), got {cfg.TOKEN_TILES}")
    if cfg.W_PHASES not in (1, 2):
        raise ValueError(f"{e} W_PHASES must be 1 or 2, got {cfg.W_PHASES}")
    if cfg.ROWS_PER_CTA not in (32, 64):
        raise ValueError(f"{e} ROWS_PER_CTA must be 32 or 64, got {cfg.ROWS_PER_CTA}")
    if cfg.MIN_CTAS_PER_SM not in (1, 2):
        raise ValueError(f"{e} MIN_CTAS_PER_SM must be 1 or 2, got {cfg.MIN_CTAS_PER_SM}")
    if cfg.SMEM_PER_SM_BYTES <= 0 or cfg.SMEM_OPTIN_BYTES <= 0 or cfg.SMEM_OPTIN_BYTES > cfg.SMEM_PER_SM_BYTES:
        raise ValueError(
            f"{e} SMEM_PER_SM_BYTES={cfg.SMEM_PER_SM_BYTES} / SMEM_OPTIN_BYTES={cfg.SMEM_OPTIN_BYTES} must be positive with the per-CTA opt-in no larger than the SM"
        )
    H, I = cfg.H, cfg.I
    if I % PAIRS_PER_CLUSTER or I % MMA_K:
        raise ValueError(f"{e} INTERMEDIATE={I} must be a multiple of {PAIRS_PER_CLUSTER} (row pairs per cluster) and of {MMA_K} (launch 2's k16 steps)")
    if H % (cfg.K_SPLIT * CHUNK_K):
        raise ValueError(f"{e} HIDDEN={H} must be a multiple of K_SPLIT x {CHUNK_K} = {cfg.K_SPLIT * CHUNK_K} (whole ring chunks per K slice)")
    if cfg.X_CHUNKS % cfg.W_PHASES:
        raise ValueError(f"{e} W_PHASES={cfg.W_PHASES} must divide the {cfg.X_CHUNKS} ring chunks of the K slice")
    if cfg.W_PHASE_ROW_BYTES % 16 or cfg.W_PHASE_K % MMA_K:
        raise ValueError(
            f"{e} a W phase piece must be a 16-byte multiple and a whole number of k16 steps; got {cfg.W_PHASE_ROW_BYTES} B / {cfg.W_PHASE_K} columns"
        )
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
    # the distributed gate logit: the partition is derived from the grid and must tile H exactly
    if cfg.GATE_PARTS * cfg.GATE_COLS != H:
        raise ValueError(f"{e} GATE_PARTS={cfg.GATE_PARTS} x GATE_COLS={cfg.GATE_COLS} must equal HIDDEN={H}")
    if cfg.GATE_COLS % 2:
        raise ValueError(f"{e} GATE_COLS={cfg.GATE_COLS} must be even (one bf16x2 word per lane and pair)")
    if cfg.GATE_PARTS % 32:
        raise ValueError(f"{e} GATE_PARTS={cfg.GATE_PARTS} must be a multiple of 32 (launch 2's lane fold reads GATE_PARTS / 32 partials per lane)")
    # the pitches
    for name, pitch in (("w_s", cfg.W_PITCH_ELEMS * 2), ("x_s", cfg.X_PITCH_ELEMS * 2), ("wd_s / act_s", cfg.WD_PITCH_ELEMS * 2)):
        if not _pitch_ok(pitch):
            raise ValueError(f"{e} the {name} row pitch {pitch} B is not 128 n + 16 (the ldmatrix bank cover)")
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


def make_cfg(**overrides) -> Cfg:
    """A validated ``Cfg``."""
    cfg = replace(Cfg(), **overrides)
    validate_cfg(cfg)
    return cfg


def k_split_for(geometry: Geometry) -> int:
    """The pinned K_SPLIT of a geometry: 128 CTAs of launch 1 on the validated envelope (2 at I 256, 4 at I 128); otherwise the split
    that brings the CTA count nearest 128 from below without exceeding the 4-way maximum."""
    clusters = geometry.INTERMEDIATE // PAIRS_PER_CLUSTER
    for s in (4, 2, 1):
        if clusters * s <= 128 and geometry.HIDDEN % (s * CHUNK_K) == 0:
            return s
    return 1


def default_cfg(geometry: Geometry = DEFAULT_GEOMETRY, smem_per_sm: int = SM100_SMEM_PER_SM_BYTES, smem_optin: int = SM100_SMEM_OPTIN_BYTES) -> Cfg:
    """The default form of a geometry on a part: the module defaults with the geometry's pinned ``K_SPLIT`` and the part's readings."""
    if not isinstance(geometry, Geometry):
        raise ValueError(f"{NAME}: geometry must be a Geometry, got {type(geometry).__name__}")
    return make_cfg(GEOM=geometry, K_SPLIT=k_split_for(geometry), **part_readings(smem_per_sm, smem_optin))


def part_readings(smem_per_sm: int, smem_optin: int) -> dict:
    """The ``Cfg`` fields of a part reading ``(shared memory per SM, per-CTA opt-in)`` in bytes, validated."""
    for name, value in (("smem_per_sm", smem_per_sm), ("smem_optin", smem_optin)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{NAME}: {name} must be a positive int (a device reading in bytes), got {value!r}")
    if smem_optin > smem_per_sm:
        raise ValueError(f"{NAME}: the per-CTA shared-memory opt-in ({smem_optin} B) cannot exceed the SM's ({smem_per_sm} B)")
    return dict(SMEM_PER_SM_BYTES=smem_per_sm, SMEM_OPTIN_BYTES=smem_optin)


def token_tiles_for(tokens: int) -> int:
    """The form of a token count: one m-tile up to 16 tokens, two up to ``MAX_TOKENS``."""
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 1:
        raise ValueError(f"{NAME}: tokens must be a positive int, got {tokens!r}")
    if tokens > MAX_TOKENS:
        raise NotImplementedError(f"{NAME} serves at most {MAX_TOKENS} tokens per call (the two-m-tile form's 25 .. 32 band is unmeasured); got {tokens}")
    return 1 if tokens <= MMA_M else 2


def form_for(cfg: Cfg, tokens: int) -> Cfg:
    """``cfg`` with the token count's form (``TOKEN_TILES``), validated."""
    tiles = token_tiles_for(tokens)
    form = cfg if cfg.TOKEN_TILES == tiles else replace(cfg, TOKEN_TILES=tiles)
    validate_cfg(form)
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
        "x_s": (cfg.X_SMEM_BYTES, cfg.X_PITCH_ELEMS * 2),
        "red_s": (cfg.RED_SMEM_BYTES, 16),
        "mail_s": (cfg.MAIL_SMEM_BYTES, 16),
        "mbarriers": (cfg.MBAR_SMEM_BYTES, None),
    }


def smem_table2(cfg: Cfg) -> dict:
    """Launch 2's SMEM buffer table."""
    return {
        "wd_s": (cfg.WD_SMEM_BYTES, cfg.WD_PITCH_ELEMS * 2),
        "act_s": (cfg.ACT_SMEM_BYTES, cfg.WD_PITCH_ELEMS * 2),
        "sig_s": (cfg.SIG_SMEM_BYTES, None),
    }


# the module forms (the cc 10.x readings; the host entry stamps the part's)
CFG = default_cfg()  # I 256: K_SPLIT 2, 128 CTAs x ~203 KB
CFG_TP16 = default_cfg(Geometry(8192, 128))  # I 128: K_SPLIT 4, 128 CTAs x ~105 KB
