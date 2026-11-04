# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config of the fused Gated DeltaNet DECODE step kernel (``gdn_decode_fused_f16.py``): the fixed model geometry it serves and
the compile-time knobs of its one kernel body.  Importable without the CuTe DSL (the host entry validates a ``Cfg`` before the DSL
version gate runs), so nothing here imports ``cutlass``.

Geometry (the Qwen 3.5-architecture dense 27B GDN sub-layer at tensor-parallel rank 1): 16 K heads, 48 V heads (three per K head), head
dim 128, causal-conv width 4, hidden 5120; the QKV slab is ``[rows, 10240]`` = Q (16 x 128) | K (16 x 128) | V (48 x 128); the fp32
recurrent state is ``[slots, 48, 128, 128]`` (one 64 KB slab per V head); the conv history is ``[slots, 10240, 3]`` (three taps per
channel, any strides the bulk copy can express -- channel-contiguous taps or tap-contiguous channels).

Default (``default_cfg``; the host entry's pick when the caller passes no ``Cfg``): the form follows two plan-time facts, the request
rows and the device's SM count.  The picks were measured on a cc 10.0 part (cold code and data, CUDA-graph replays), every form bitwise
with the reference kernel -- FlashInfer's built-in ``frost`` backend for the Qwen 3.5 GDN block, whose numerics these kernels reproduce.
Every form takes ``bulk``, TPR 4 and ``SINGLE_TAIL`` (the tail traced once: faster at every rows count than the twice-traced body).
While the ``rows x 96`` CTAs of the two-CTAs-per-head form have an SM each (``rows * 96 <= SM count``: rows 1 on 148 / 152 SMs)
``PAIR_CFG`` serves -- ``TWO_CTAS_PER_HEAD``, ``CTAS_PER_SM`` 0 -- because the pair buys a second SM's bulk-copy path per head (the
fastest form at rows 1; the pair loses once it has to share an SM).  While every CTA of the ``rows x 48`` grid has an SM of its own
(``rows * 48 <= SM count``) the one-CTA-per-SM form ``CFG`` serves.  Past that the register-resident form under the 128-register cap
``CAPPED_CFG`` serves -- ``MIN_CTAS_PER_SM`` 2, ``CTAS_PER_SM`` 0 -- so two CTAs share an SM and the grid runs in ONE wave (slightly ahead
of the streamed form ``STREAMED_CFG``) -- on a part where two of its CTAs CAN share an SM (``Cfg.co_resident_ctas`` >= 2: the cc 10.0
part's 228 KB per SM holds three 75 KB footprints, the register cap two CTAs of 256 threads; spill-free there with one tail).  Two CTAs per
SM is a PER-PART premise, read from ``co_resident_ctas``: on a 100 KB part (cc 12.0: 102,400 B per SM, 101,376 B opt-in) the whole-slab
form is one CTA per SM by shared memory alone (2 x (74,848 + 1,024) = 151,744 B > 102,400 B), so the 128-register cap buys nothing there
and costs spills -- measured on sm_120a (``cuobjdump -res-usage``, 256 threads): ``CAPPED_CFG`` 128 registers + 16 B stack + 7 STL / LDL,
``STREAMED_CFG`` 128 / 0, ``CFG`` 145 / 0, ``PAIR_CFG`` 111 / 0, ``PAIR_CFG`` with ``MIN_CTAS_PER_SM`` 2 109 / 0 (the cap a no-op: the pair
is two per SM by registers without it, and 2 x (42,112 + 1,024) = 86,272 B fit the 100 KB SM).  ``PDL`` stays off in every form: it needs
the caller's stream contract (its knob below).  A caller's ``Cfg`` overrides the pick as a whole.

Knobs (every one a ``cutlass.const_expr`` axis of the same kernel body):

* ``THREADS`` 256 (two warpgroups: BA on warps 0-3, conv on warps 4-7) or 128 (one warpgroup does both, sequentially).
* ``TPR`` threads per state row.  A thread holds ``R = 128 * TPR / THREADS`` rows x ``C = 128 / TPR`` consecutive columns of the state
  in registers (``R * C`` = 64 at 256 threads).  Lane ``l`` of a warp is ``tpr = l // RG`` (its column slice) and ``rg = l % RG`` (its
  row group, ``RG = 32 / TPR`` per warp); its rows are ``BOX_ROWS * rg + R * warp + m`` (``BOX_ROWS = 128 / RG``): ``R`` consecutive,
  ``R``-aligned rows of state box ``rg``.  The row dot products reproduce the reference 16-leaf balanced tree with ``log2(TPR)``
  shuffle steps (offsets ``RG << step``); the norm reproduces its 128-leaf tree from ``RG x WARPS`` aligned-block partials.
* ``K_RESIDENT``: keep the thread's k slice in registers across both passes (else re-read it from shared memory per 8-column group).
* ``WARPGROUP_SPLIT``: run the BA dots (warpgroup 0) and the conv + RMS scales (warpgroup 1) in parallel; off = both on warpgroup 0.
* ``CTAS_PER_SM``: 0 = natural shared memory, 1 / 2 = a dummy pad that caps the co-resident CTAs per SM at one / two (the occupancy
  axis), derived from ``SMEM_PER_SM_BYTES`` -- the DEVICE's shared memory per SM, which the host entry reads from the part (the
  module constant is the cc 10.0 reading, 228 KB) -- never from a size carried across an architecture line.
* ``STATE_LOAD_PATH``: ``"ldg128"`` = 16 x ``ld.global.v4`` per thread straight into the row registers (and ``st.global.v4`` back);
  ``"bulk"`` = the 64 KB slab lands in shared memory by ``RG`` ``cp.async.bulk`` boxes on one mbarrier (box pitch padded by 16 B so the
  ``LDS.128`` of a row slice by the TPR lanes is bank-conflict-free), ``LDS.128`` into the row registers, ``STS.128`` back after the
  update and ``RG`` bulk stores to global -- full 128-B line efficiency both ways.
* ``STATE_STREAM`` (bulk only): pass 2 re-reads each row from shared memory and recomputes ``h = S * decay`` (bit-identical) instead of
  keeping the 64 updated values live across the two passes -- the <= 128-register form that lets two CTAs share an SM.
* ``MIN_CTAS_PER_SM``: the launch's ``min_blocks_per_mp`` = ptxas's register cap (1 -> 255, 2 -> 128 registers per thread at 256 threads);
  the SASS gate (no ``STL`` / ``LDL``) decides whether a form may use it, and ``Cfg.co_resident_ctas`` (the CTAs per SM the part's shared
  memory and the cap guarantee together) whether it buys anything on the part: a cap on a form whose footprint already keeps the SM at
  one CTA is pure cost (sm_120a: ``CAPPED_CFG`` spills 7).
* ``PDL`` / ``PDL_RELEASE``: ``griddepcontrol.wait`` AFTER the predecessor-independent loads (slot, parameters, BA weight rows, the state
  slab, the conv weights and history) and BEFORE ``z``, ``hidden``, the ``x`` copies and every global write (the pre-wait prefetch order
  of the reference verify kernel); ``griddepcontrol.launch_dependents`` at entry; the launch carries the programmatic-stream-
  serialization attribute when ``PDL`` is set.  CALLER OBLIGATION under ``PDL``: the kernel immediately preceding this launch in the
  stream must not write this layer's ``state``, conv state or slot indices -- the pre-wait loads read them BEFORE
  ``griddepcontrol.wait``, so a predecessor that writes them (the same layer launched twice in a row: a unit-test loop, a one-layer
  graph) races the slab / history reads against its stores -- a wrong state, no error.  The serving chain satisfies it (the
  predecessor is the QKVZ projection, which writes ``x`` / ``z`` only, both read after the wait); the entry cannot check it.
* ``PIN_ENTRY_LOADS`` (ldg128 only): scheduling pins the assembler cannot move loads below -- a ``bar.warp.sync`` after the entry-load
  block and the BA warpgroup's named barrier right after its state-load block.  Measured on cc 10.0 as a COST (the early state traffic
  delays the BA data in the L1), so it is off by default; the one-flag A/B stays.
* ``CONV_FIRST`` (bulk only): ONE lane -- lane 0 of the conv role's first warp -- initialises both mbarriers and issues the conv-operand
  copies (6 KB) BEFORE the state slab boxes (64 KB), so the small copies no longer queue behind the slab in the SM's bulk-copy path
  (measured: the conv operands landed well after their derived time and the conv role gated the CTA hand-over while the BA role
  idled).  Off: the committed order (the BA role's warp 0 issues the slab right after the slot, the conv role its copies).
* ``STORE_PIPELINE`` (bulk, 256 threads): the write-back of each warpgroup's half of every state box (the mapping gives warps 0-3 the
  first ``BOX_ROWS / 2`` rows of every box and warps 4-7 the second half: contiguous runs in both memories) is issued by that
  warpgroup's lane 0 as soon as the warpgroup's pass 2 is published to it (a named barrier its warps 1-3 arrive on and its warp 0
  syncs on) instead of after the CTA-wide norm barrier; two draining lanes.  The excess it targets: the 64 KB write-back read out of
  shared memory after the last compute.
* ``SINGLE_TAIL`` (default ON): the two warpgroup roles reconverge before the CTA hand-over and the role-independent tail (the slab
  wait, the two passes, the norm, the outputs, the history shift, the slab stores) is traced ONCE instead of once per role arm --
  one PTX site per CTA-wide barrier, a smaller image (1784 instructions against 2360 for the twice-traced body on the one-CTA form;
  the reference kernel's 1672).  Off: the twice-traced body (the A/B twin).  A no-op on the 128-thread form (one role arm already) and
  on the register state path (its row registers would have to cross the join, so that path keeps the twice-traced body).
* ``TWO_CTAS_PER_HEAD`` (bulk only): a cluster of two CTAs per V head, each owning half the state rows (``CTA_ROWS`` 64: a 32 KB slab
  per SM, so the slab's bulk-copy time in and out and the shared-memory port floors halve, as do the two passes); both CTAs recompute
  the BA (decay, beta) and the conv + RMS (q, k, v) from the same inputs (26 KB of L2-resident re-reads per head, no exchange on the
  critical path).  The ONLY cross-CTA datum is the norm: ``T128 = T64(rows 0-63) + T64(rows 64-127)`` -- each CTA builds its T64 (the
  aligned-block tree over its own 64 rows) and ships it to the peer as one ``st.async`` 4-byte store that completes 4 tx bytes on the
  peer's ``mb_norm`` (initialised with the other mbarriers, published cluster-wide by the relaxed cluster arrive at entry + wait before
  the exchange: the one cluster barrier of the kernel); fp32 addition is commutative, so both ranks compute the same bits.  Six readers
  per K head: ``TICKET_LIMIT`` 5; each CTA shifts the V-conv history of its own 64 channels.  Use with ``CTAS_PER_SM`` 0 (the natural
  footprint, 42 KB: registers decide the co-residency) and the grid is ``(96, rows)`` in clusters of 2.
* ``DEBUG``: the reference kernel's debug outputs (``convout``, ``baout``, ``raw``) for intermediate byte-compares.
"""

from dataclasses import dataclass, replace

# --- the served geometry (the model's constants; the host entry declines everything else) ---
K_HEADS = 16
V_HEADS = 48
HEAD_DIM = 128
CONV_WIDTH = 4
HIDDEN = 5120
QKV_CHANNELS = 2 * K_HEADS * HEAD_DIM + V_HEADS * HEAD_DIM  # 10240
K_CHANNEL0 = K_HEADS * HEAD_DIM  # 2048: the first K channel of the QKV slab
V_CHANNEL0 = 2 * K_HEADS * HEAD_DIM  # 4096: the first V channel
BA_THREADS = 128  # the reference BA partition: thread t sums elements chunk*1024 + t*8 + j, five chunks
BA_CHUNKS = HIDDEN // (BA_THREADS * 8)  # 5
CONV_THREADS = 128  # one Q, one K and one V channel per thread
RMS_LANES = 16  # the reference RMS layout: lane l of a 16-lane group holds the 8 raw q / k values of columns 8 l .. 8 l + 7
TICKET_STRIDE = 512  # counts[row * 512 + kh * 32]: the reference layout of the per-(row, K head) reader ticket
TICKET_LIMIT = V_HEADS // K_HEADS - 1  # 2: the third reader of a K head's conv history is the last

# bytes the conv-operand bulk copies deliver on the one mbarrier: x pieces 3 x 256, conv weights 3 x 1024, history 3 x 3 x 256
CONV_BULK_BYTES = 3 * HEAD_DIM * 2 + 3 * HEAD_DIM * CONV_WIDTH * 2 + 3 * (CONV_WIDTH - 1) * HEAD_DIM * 2  # 6144
# arrivals the conv-operand mbarrier is initialised for: ONE elected lane of the conv warpgroup's first warp arms the tx phase
# (arrive.expect_tx), the bulk copies complete the bytes -- SUM(arriving lanes) == init count on every path
CONV_MBAR_ARRIVERS = 1

# the state slab of one V head (128 x 128 fp32) and its mbarrier on the bulk path: ONE arrive.expect_tx(STATE_SLAB_BYTES) by warp 0's
# lane 0, the RG box copies complete the bytes
STATE_SLAB_BYTES = HEAD_DIM * HEAD_DIM * 4  # 65536
STATE_MBAR_ARRIVERS = 1
# one 16-B chunk between two state boxes shifts the next box by 4 banks: with a phase's 8 lanes on RG distinct boxes (TPR >= 4) or
# RG consecutive row groups, their 16-B chunks land in 8 distinct bank quads (the bank map: bank = 4 rg + (128 / TPR) tpr + 4 c4)
STATE_BOX_PAD_BYTES = 16

# TWO_CTAS_PER_HEAD: the norm partial of the peer CTA lands by ONE st.async of 4 bytes on this CTA's mb_norm; its init count is the ONE
# local arrive.expect_tx (the peer's complete_tx carries no software arrive) -- SUM(arriving lanes) == init count on every path
NORM_MBAR_ARRIVERS = 1
NORM_XFER_BYTES = 4

STATE_LOAD_PATHS = ("ldg128", "bulk")

# The cc 10.0 part's shared memory per SM (228 KB), its per-CTA opt-in maximum (227 KB) and the driver's per-CTA reserve (1 KB); the
# host entry REPLACES the first two with the device's own readings (``cudaDevAttrMaxSharedMemoryPerMultiprocessor`` / ``PerBlockOptin``),
# so a hardcoded size never decides an occupancy pad on a part it does not describe.
SM100_SMEM_PER_SM_BYTES = 228 * 1024
SM100_SMEM_OPTIN_BYTES = 227 * 1024
# The cc 12.0 part's readings (100 KB per SM, 99 KB per-CTA opt-in, the same 1 KB reserve; a 170-SM RTX 5090-class part): documentation
# and tests only, never a default -- the host entry reads the part.
SM120_SMEM_PER_SM_BYTES = 100 * 1024
SM120_SMEM_OPTIN_BYTES = 99 * 1024
SMEM_CTA_RESERVE_BYTES = 1024
SMEM_PAD_MARGIN_BYTES = 4096  # keeps the cap robust to the allocation granularity


@dataclass(frozen=True)
class Cfg:
    THREADS: int = 256
    TPR: int = 4
    K_RESIDENT: bool = True
    WARPGROUP_SPLIT: bool = True
    CTAS_PER_SM: int = 1
    STATE_LOAD_PATH: str = "bulk"
    STATE_STREAM: bool = False
    MIN_CTAS_PER_SM: int = 1
    PDL: bool = False
    PDL_RELEASE: bool = False
    PIN_ENTRY_LOADS: bool = False
    CONV_FIRST: bool = False
    SINGLE_TAIL: bool = True
    STORE_PIPELINE: bool = False
    TWO_CTAS_PER_HEAD: bool = False
    DEBUG: bool = False
    SMEM_PER_SM_BYTES: int = SM100_SMEM_PER_SM_BYTES
    SMEM_OPTIN_BYTES: int = SM100_SMEM_OPTIN_BYTES

    # --- derived (read-only; computed by make_cfg) ---
    @property
    def WARPS(self) -> int:
        return self.THREADS // 32

    @property
    def CTAS_PER_HEAD(self) -> int:
        return 2 if self.TWO_CTAS_PER_HEAD else 1

    @property
    def CTA_ROWS(self) -> int:
        """State rows one CTA owns (128, or 64 on the two-CTAs-per-head form: CTA ``r`` of the pair owns rows ``64 r ..``)."""
        return HEAD_DIM // self.CTAS_PER_HEAD

    @property
    def R(self) -> int:
        """State rows per thread."""
        return self.CTA_ROWS * self.TPR // self.THREADS

    @property
    def C(self) -> int:
        """State columns per thread (one 8-column group = one reference lane partial)."""
        return HEAD_DIM // self.TPR

    @property
    def GROUPS(self) -> int:
        return self.C // 8

    @property
    def RG(self) -> int:
        """Row groups per warp = the lanes that share a column slice index; lane ``l`` is ``tpr = l // RG``, ``rg = l % RG``."""
        return 32 // self.TPR

    @property
    def BOX_ROWS(self) -> int:
        """State rows per box (the rows the ``rg`` lanes of every warp hold together); box ``rg`` = this CTA's rows ``BOX_ROWS rg ..``."""
        return self.CTA_ROWS // self.RG

    @property
    def ROWS_PER_WARP(self) -> int:
        return self.CTA_ROWS // self.WARPS

    @property
    def V_PERM_STRIDE(self) -> int:
        """The permuted ``v_s`` index of head row ``u`` is ``(u % BOX_ROWS) * V_PERM_STRIDE + u // BOX_ROWS`` (``RG x CTAS_PER_HEAD``): the
        RG lanes of a phase read consecutive words, and a two-CTA pair reads disjoint halves of the same 128-entry array."""
        return HEAD_DIM // self.BOX_ROWS

    @property
    def STATE_SLAB_BYTES_CTA(self) -> int:
        """Bytes of the state rows this CTA loads and stores (the slab mbarrier's expect_tx): 65536, or 32768 on the two-CTA form."""
        return self.CTA_ROWS * HEAD_DIM * 4

    @property
    def TICKET_LIMIT(self) -> int:
        """The last reader ticket of a K head's conv history: ``3 x CTAS_PER_HEAD`` CTAs read it, the one that draws this value shifts it."""
        return (V_HEADS // K_HEADS) * self.CTAS_PER_HEAD - 1

    @property
    def SLICE_PITCH(self) -> int:
        """fp32 elements between two threads' q / k slices in shared memory: C + 4 spreads the TPR slices over distinct banks."""
        return self.C + 4

    @property
    def BULK_STATE(self) -> bool:
        return self.STATE_LOAD_PATH == "bulk"

    @property
    def STATE_BOX_BYTES(self) -> int:
        """Bytes of one state box = ``BOX_ROWS`` rows of 128 fp32, one contiguous run in global memory and in shared memory."""
        return self.BOX_ROWS * HEAD_DIM * 4

    @property
    def STATE_BOX_PITCH_ELEMS(self) -> int:
        """fp32 elements between two boxes in shared memory (the box plus the 16-B pad)."""
        return self.BOX_ROWS * HEAD_DIM + STATE_BOX_PAD_BYTES // 4

    @property
    def STATE_SMEM_ELEMS(self) -> int:
        """fp32 elements of the shared-memory state buffer: the padded boxes on the bulk path, one 16-B slot otherwise."""
        return self.RG * self.STATE_BOX_PITCH_ELEMS if self.BULK_STATE else 4

    @property
    def NORM_PARTS(self) -> int:
        """Norm partials published through shared memory: one aligned-block sum per (row group, warp)."""
        return self.RG * self.WARPS

    @property
    def SHIFT_LANES(self) -> int:
        """The lanes that shift a K head's Q / K conv history at exit: one per state row group (``tpr == TPR - 1``)."""
        return self.THREADS // self.TPR

    @property
    def NATURAL_SMEM_BYTES(self) -> int:
        """The kernel's own shared-memory footprint in bytes, buffer by buffer in allocation order (16-byte slots): x pieces, conv
        weights, conv history, raw q / k, the padded q / k slices, v, scalars, ticket, BA partials, norm partials, the two mbarriers,
        the state buffer."""
        return (
            3 * HEAD_DIM * 2  # x_s
            + 3 * HEAD_DIM * CONV_WIDTH * 2  # cw_s
            + 3 * (CONV_WIDTH - 1) * HEAD_DIM * 2  # hist_s
            + 2 * HEAD_DIM * 4  # qk_raw
            + 2 * self.TPR * self.SLICE_PITCH * 4  # q_s, k_s
            + HEAD_DIM * 4  # v_s
            + 16
            + 16
            + 32  # scal_s, ticket_s, ba_part
            + (self.NORM_PARTS * 4 + 15) // 16 * 16  # norm_part
            + 16
            + 16  # mb_ops, mb_state (8 bytes each in a 16-byte slot)
            + (32 if self.TWO_CTAS_PER_HEAD else 0)  # norm_peer_s (the peer's T64 in a 16-byte slot) + mb_norm
            + self.STATE_SMEM_ELEMS * 4  # state_s
        )

    @property
    def SMEM_PAD_BYTES(self) -> int:
        """The occupancy pad: sized so that ``CTAS_PER_SM + 1`` CTAs (each its footprint + pad + the driver's reserve) exceed the
        device's shared memory per SM while ``CTAS_PER_SM`` of them fit -- the SM holds at most ``CTAS_PER_SM`` CTAs by shared memory."""
        if self.CTAS_PER_SM == 0:
            return 0
        pad = self.SMEM_PER_SM_BYTES // (self.CTAS_PER_SM + 1) + SMEM_PAD_MARGIN_BYTES - SMEM_CTA_RESERVE_BYTES - self.NATURAL_SMEM_BYTES
        return max(0, (pad + 15) // 16 * 16)

    @property
    def SMEM_BYTES(self) -> int:
        return self.NATURAL_SMEM_BYTES + self.SMEM_PAD_BYTES

    @property
    def co_resident_ctas(self) -> int:
        """The CTAs of this form that share one SM on the part, as the host can GUARANTEE them from the two budgets it knows: by shared
        memory, how many launch footprints plus the driver's per-CTA reserve fit ``SMEM_PER_SM_BYTES``; by registers, ``MIN_CTAS_PER_SM``
        (the cap the launch asks ptxas for -- 2 = 128 registers, two CTAs of 256 threads in the 64 K file; an uncapped form may take up to
        255, so the host counts one) -- the smaller of the two.  The ONE predicate the default pick, its docs and its tests read: a
        ``MIN_CTAS_PER_SM`` 2 form is worth its cap only where this is 2 (the cc 10.0 readings: ``CAPPED_CFG`` 2; the cc 12.0 readings: 1,
        2 x (74,848 + 1,024) B > 102,400 B), and the half-slab pair under the cap is two per SM on both (42,112 B)."""
        by_smem = self.SMEM_PER_SM_BYTES // (self.SMEM_BYTES + SMEM_CTA_RESERVE_BYTES)
        return min(by_smem, self.MIN_CTAS_PER_SM)


def validate_cfg(cfg: Cfg) -> None:
    """Raise the typed error of every unexpressible knob combination (structural invariants raise here, never a module assert)."""
    if cfg.THREADS not in (128, 256):
        raise ValueError(f"gdn_decode_fused: THREADS must be 128 or 256, got {cfg.THREADS}")
    if cfg.TPR not in (1, 2, 4, 8, 16):
        raise ValueError(f"gdn_decode_fused: TPR must be 1, 2, 4, 8 or 16, got {cfg.TPR}")
    if cfg.CTA_ROWS * cfg.TPR % cfg.THREADS != 0 or cfg.R < 1:
        raise ValueError(f"gdn_decode_fused: THREADS={cfg.THREADS} x TPR={cfg.TPR} does not tile the {cfg.CTA_ROWS} x 128 state rows of one CTA")
    if cfg.R * cfg.C * cfg.THREADS != cfg.CTA_ROWS * HEAD_DIM:
        raise ValueError(f"gdn_decode_fused: R={cfg.R} x C={cfg.C} x THREADS={cfg.THREADS} != {cfg.CTA_ROWS} x 128")
    if cfg.C % 8 != 0:
        raise ValueError(f"gdn_decode_fused: C={cfg.C} columns per thread must be a multiple of the 8-column reference group")
    if cfg.R & (cfg.R - 1):
        raise ValueError(f"gdn_decode_fused: R={cfg.R} rows per thread must be a power of two (the norm tree pre-sum)")
    if cfg.BOX_ROWS != cfg.R * cfg.WARPS:
        raise ValueError(f"gdn_decode_fused: BOX_ROWS={cfg.BOX_ROWS} != R={cfg.R} x WARPS={cfg.WARPS} (the row-group boxes do not tile the slab)")
    if cfg.WARPGROUP_SPLIT and cfg.THREADS != 256:
        raise ValueError("gdn_decode_fused: WARPGROUP_SPLIT needs THREADS=256 (two warpgroups)")
    if cfg.CTAS_PER_SM not in (0, 1, 2):
        raise ValueError(f"gdn_decode_fused: CTAS_PER_SM must be 0, 1 or 2, got {cfg.CTAS_PER_SM}")
    if cfg.MIN_CTAS_PER_SM not in (1, 2):
        raise ValueError(f"gdn_decode_fused: MIN_CTAS_PER_SM must be 1 or 2, got {cfg.MIN_CTAS_PER_SM}")
    if cfg.CTAS_PER_SM and cfg.MIN_CTAS_PER_SM > cfg.CTAS_PER_SM:
        raise ValueError(
            f"gdn_decode_fused: MIN_CTAS_PER_SM={cfg.MIN_CTAS_PER_SM} (the register cap aims at that many CTAs per SM) contradicts CTAS_PER_SM={cfg.CTAS_PER_SM} (the pad caps the SM at fewer)"
        )
    if cfg.STATE_LOAD_PATH not in STATE_LOAD_PATHS:
        raise ValueError(f"gdn_decode_fused: STATE_LOAD_PATH must be one of {STATE_LOAD_PATHS}, got {cfg.STATE_LOAD_PATH!r}")
    if cfg.STATE_STREAM and not cfg.BULK_STATE:
        raise ValueError("gdn_decode_fused: STATE_STREAM re-reads the rows from shared memory and needs STATE_LOAD_PATH='bulk'")
    if cfg.PIN_ENTRY_LOADS and cfg.BULK_STATE:
        raise ValueError("gdn_decode_fused: PIN_ENTRY_LOADS pins the ldg128 state loads; there are none on STATE_LOAD_PATH='bulk'")
    if cfg.TWO_CTAS_PER_HEAD and not cfg.BULK_STATE:
        raise ValueError("gdn_decode_fused: TWO_CTAS_PER_HEAD splits the shared-memory slab between a cluster pair and needs STATE_LOAD_PATH='bulk'")
    if cfg.STORE_PIPELINE and not cfg.BULK_STATE:
        raise ValueError("gdn_decode_fused: STORE_PIPELINE pipelines the shared-memory box stores; there are none on STATE_LOAD_PATH='ldg128'")
    if cfg.STORE_PIPELINE and cfg.THREADS != 256:
        raise ValueError("gdn_decode_fused: STORE_PIPELINE splits every box between the two warpgroups and needs THREADS=256")
    if cfg.CONV_FIRST and not cfg.BULK_STATE:
        raise ValueError(
            "gdn_decode_fused: CONV_FIRST orders the conv-operand copies ahead of the state slab boxes; there are no slab boxes on STATE_LOAD_PATH='ldg128'"
        )
    if cfg.SMEM_PER_SM_BYTES <= 0 or cfg.SMEM_OPTIN_BYTES <= 0 or cfg.SMEM_OPTIN_BYTES > cfg.SMEM_PER_SM_BYTES:
        raise ValueError(
            f"gdn_decode_fused: SMEM_PER_SM_BYTES={cfg.SMEM_PER_SM_BYTES} / SMEM_OPTIN_BYTES={cfg.SMEM_OPTIN_BYTES} must be positive with the per-CTA opt-in no larger than the SM"
        )
    if cfg.SMEM_BYTES > cfg.SMEM_OPTIN_BYTES:
        raise ValueError(f"gdn_decode_fused: shared memory {cfg.SMEM_BYTES} B exceeds the per-CTA opt-in maximum {cfg.SMEM_OPTIN_BYTES} B")
    if cfg.CTAS_PER_SM and cfg.CTAS_PER_SM * (cfg.SMEM_BYTES + SMEM_CTA_RESERVE_BYTES) > cfg.SMEM_PER_SM_BYTES:
        raise ValueError(f"gdn_decode_fused: CTAS_PER_SM={cfg.CTAS_PER_SM} CTAs of {cfg.SMEM_BYTES} B do not fit the SM's {cfg.SMEM_PER_SM_BYTES} B")


def make_cfg(**overrides) -> Cfg:
    """A validated ``Cfg``.  The field defaults are the one-CTA-per-SM form of the measured pick (``default_cfg``): 256 threads, TPR 4,
    the shared-memory state path, the single traced tail, one CTA per SM by the pad and the 255-register budget, the pin and PDL off."""
    cfg = replace(Cfg(), **overrides)
    validate_cfg(cfg)
    return cfg


# the forms of the default (the module docstring's "Default" paragraph)
PAIR_CFG = make_cfg(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0)  # a cluster of two CTAs per head -- while the rows x 96 CTAs have an SM each
CFG = make_cfg()  # one CTA per SM: bulk, TPR 4 -- while the rows x 48 CTAs have an SM each
CAPPED_CFG = make_cfg(MIN_CTAS_PER_SM=2, CTAS_PER_SM=0)  # register-resident under the 128-register cap: two CTAs per SM, one wave -- past that
STREAMED_CFG = make_cfg(
    STATE_STREAM=True, MIN_CTAS_PER_SM=2, CTAS_PER_SM=0
)  # the streamed two-CTAs-per-SM form (a measured knob form, slightly behind CAPPED_CFG)


def default_cfg(rows: int, sm_count: int) -> Cfg:
    """The measured pick per rows: ``PAIR_CFG`` while every CTA of the ``rows x 96`` cluster grid has an SM of its own (rows 1 on a
    148- or 152-SM part), ``CFG`` (one CTA per SM) while every CTA of the ``rows x 48`` grid has one (rows 2 and 3), ``CAPPED_CFG``
    (two CTAs per SM, one wave) once the grid outgrows the device's SMs (rows 4 and up: 192 CTAs).  ``rows`` is in the compile key
    and ``sm_count`` is the device's multiprocessor count, both plan-time facts (never a runtime value)."""
    if not isinstance(rows, int) or rows < 1:
        raise ValueError(f"gdn_decode_fused: rows must be a positive int, got {rows!r}")
    if not isinstance(sm_count, int) or sm_count < 1:
        raise ValueError(f"gdn_decode_fused: the SM count must be a positive int, got {sm_count!r}")
    if rows * V_HEADS * PAIR_CFG.CTAS_PER_HEAD <= sm_count:
        return PAIR_CFG
    return CFG if rows * V_HEADS <= sm_count else CAPPED_CFG
