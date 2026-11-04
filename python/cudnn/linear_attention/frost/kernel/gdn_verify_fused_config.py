# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config of the fused Gated DeltaNet VERIFY step kernel (``gdn_verify_fused_f16.py``): the fixed model geometry it serves, the
compile-time knobs of its one kernel body and the shared-memory budget per tokens-per-request count.  Importable without the
CuTe DSL (the host entry validates a ``Cfg`` before the DSL version gate runs), so nothing here imports ``cutlass``.

Geometry (the Qwen 3.5-architecture dense 27B GDN sub-layer at tensor-parallel rank 1, the same as the fused decode step's): 16 K heads,
48 V heads (three per K head), head dim 128, causal-conv width 4, hidden 5120; the QKV slab is ``[rows, 10240]`` = Q (16 x 128) | K
(16 x 128) | V (48 x 128); the fp32 recurrent state is ``[slots, 48, 128, 128]`` (one 64 KB slab per V head); the conv history of a
verify slot holds ``T + 2`` columns of 10240 channels, stored channel-contiguous (``[slots, T + 2, 10240]`` viewed ``[slots, 10240, T + 2]``).

The verify step: a request of ``T`` tokens (``T`` = the second extent of the slot-index table, a compile-time constant of the
kernel: tokens per request of the serving plan) starts from the state of its last accepted token (slot ``indices[n, a - 1]``,
``a = num_accepted_tokens[n]``) and writes the state after every token ``t`` to slot ``indices[n, t]`` (a slot ``<= 0`` skips
that write).  One CTA per (request, V head), 256 threads = two warpgroups; the ``T`` tokens run sequentially on the
register-resident (or shared-memory-resident) state rows; every token's updated slab goes out by ``cp.async.bulk`` stores
from a ring of padded shared-memory boxes.

Default (``default_cfg``; the host entry's pick when the caller passes no ``Cfg``): while every CTA of the ``requests x 96`` cluster
grid has an SM of its own (``requests * 96 <= SM count``: one request on a 148- / 152-SM part) the two-CTAs-per-head PAIR form
``PAIR_CFG`` serves (``TWO_CTAS_PER_HEAD``: each CTA of a cluster pair owns half the state rows, so every per-SM term -- the 32 KB slab in,
the 32 KB write-back per token, the register-resident rows, the pass math, the straight-line code -- halves, which is what the cold-L2
serving regime pays for: every layer's launch finds its code and data evicted); ``ROLLED_TOKENS`` is ON in ``PAIR_CFG`` and in ``CFG``
(with the L2 swept before every launch the rolled forms are the fastest at one and at two requests on a cc 10.0 part; warm, with the
code resident, the unrolled pair is marginally faster at one request -- the cold figure decides); while every CTA of the ``requests x
48`` grid has an SM of its own (``requests * 48 <= SM count``) the one-CTA-per-SM form ``CFG`` serves (TPR 4, the 3-deep ring); past that
the two-CTAs-per-SM form ``TWO_CTA_CFG`` serves (the register-capped streamed form, so the grid runs in one wave) -- on a part where two of
its CTAs CAN share an SM (``Cfg.co_resident_ctas(tokens)`` >= 2 at the minimum token count: the cc 10.0 part's 228 KB per SM holds two
80 KB footprints, the register cap two CTAs of 256 threads).  Two CTAs per SM is a PER-PART premise, read from ``co_resident_ctas``: on a
100 KB part (cc 12.0: 102,400 B per SM, 101,376 B opt-in) every whole-slab form is one CTA per SM by shared memory alone (2 x (79,200 +
1,024) = 160,448 B > 102,400 B at T 2), so the 128-register cap buys nothing there and costs spills -- measured on sm_120a (``cuobjdump
-res-usage``, 256 threads, T 6): ``TWO_CTA_CFG`` 128 registers + 56 B stack + 34 STL / LDL; the rolled register-resident forms ``PAIR_CFG``
with NBUF 2 187 / 0, with NBUF 1 187 / 0, the one-CTA NBUF-1 form 183 / 0; the NBUF-1 pair under the cap 128 / 0 (spill-free: the one
two-per-SM verify form of that part, at T <= 3: 2 x (50,112 + 1,024) = 102,272 B).  Where the regime's primary form is not ADMISSIBLE on
the part (``admissible``: its ring holds the minimum token count in the per-CTA opt-in and, under the register cap, two CTAs share an
SM) the pick walks the regime's ladder (``PAIR_FORMS`` / ``ONE_CTA_FORMS`` / ``BEYOND_FORMS``: the primary, then the forms a smaller
shared memory holds, deepest ring first) and is token-aware -- on the cc 12.0 reading ``PAIR_NBUF2_CFG`` at one request and
``ONE_CTA_NBUF1_CFG`` at two and more for T <= 8, ``PAIR_NBUF1_CFG`` for T 9 .. 17 at every requests count, and T >= 18 is the typed
decline naming 17 (``max_tokens_served``, the bound FlashInfer's probe reads); where the primary is admissible (every cc 10.x part) the
pick and its token bound are the measured ones above, whatever the step's token count.  Every form is bitwise with the reference
kernel -- FlashInfer's built-in ``frost`` backend for the Qwen 3.5 GDN block, whose numerics these kernels reproduce: no knob moves an
fp32 addition.  ``PDL`` stays off in every form (it needs the caller's stream contract: its knob below).  A caller's ``Cfg`` overrides
the pick as a whole; a ``NORM_FORM`` of ``None`` is resolved from the requests count by the entry (``resolve_cfg``).

Knobs (every one a ``cutlass.const_expr`` axis of the same kernel body):

* ``TPR`` threads per state row.  A thread holds ``R = TPR / 2`` rows x ``C = 128 / TPR`` consecutive columns of the state (``R * C``
  = 64 fp32); lane ``l`` of a warp is ``tpr = l // RG`` (its column slice) and ``rg = l % RG`` (its row group, ``RG = 32 / TPR``);
  its rows are ``BOX_ROWS * rg + R * warp + m`` (``BOX_ROWS = 128 / RG``): ``R`` consecutive, ``R``-aligned rows of state box ``rg``.
  The row dot products reproduce the reference 16-leaf balanced tree with ``log2(TPR)`` shuffle steps (offsets ``RG << step``).
* ``K_RESIDENT``: keep the token's k slice in registers across both passes (else re-read it per 8-column group).
* ``NBUF``: the depth of the padded-box ring the per-token state write-back goes through (one 64 KB slab per buffer).  At ``NBUF >= 2``
  the ring's WAR wait is ``cp.async.bulk.wait_group.read (NBUF - 2)`` by the issuing lane ahead of the per-token barrier (one
  barrier per token); at ``NBUF == 1`` the single box is re-used every token behind a second barrier (the issuing lane's
  ``wait_group.read 0`` between the token's two passes).
* ``STATE_STREAM``: the state lives in the shared-memory boxes, not in registers: pass 1 reads each row and drops it, pass 2
  re-reads it, recomputes ``h = S * decay`` (bit-identical) and writes the updated row to the next box -- the <= 128-register form
  that lets two CTAs share an SM where the part's shared memory holds two of its footprints (``NBUF`` 1 or 2 there).
* ``MIN_CTAS_PER_SM``: the launch's ``min_blocks_per_mp`` = ptxas's register cap (1 -> 255, 2 -> 128 registers per thread); the SASS
  gate (no ``STL`` / ``LDL``) decides whether a form may use it, and ``Cfg.co_resident_ctas(tokens)`` (the CTAs per SM the part's shared
  memory and the cap guarantee together) whether it buys anything on the part: a cap on a form whose footprint already keeps the SM at
  one CTA is pure cost (sm_120a: ``TWO_CTA_CFG`` spills 34).
* ``CTAS_PER_SM``: 0 = natural shared memory, 1 / 2 = a dummy pad that caps the co-resident CTAs per SM at one / two, derived from
  ``SMEM_PER_SM_BYTES`` -- the DEVICE's shared memory per SM, which the host entry reads from the part.
* ``BA_PREFETCH_CHUNKS``: how many of the three 2048-element chunks of the BA projection's ``hidden`` rows (all ``T`` tokens) are in
  flight at once in the BA warpgroup (3 = one load latency, 20 registers per token per chunk; 1 = three latencies, a third of
  the registers -- the register-capped forms).
* ``NORM_FORM``: the association of the gated norm's sum of squares, which the reference verify kernel fixes by its CTAs-per-head
  policy: ``"ctas2"`` (one request: two 64-row halves of 16 four-row chains each, summed last) or ``"ctas1"`` (sixteen eight-row
  chains); ``None`` = that policy for the compiled requests count.
* ``STATE_LOAD_PATH``: ``"bulk"`` (the only served path: the slab lands by ``RG`` ``cp.async.bulk`` boxes on one mbarrier in
  16-B-padded shared-memory boxes, ``LDS.128`` into the row registers).
* ``STATE_STORE_PATH``: how a token's updated slab reaches its slot: ``"bulk"`` = ``STS.128`` into a ring box, ``fence.proxy.async``,
  the per-token barrier, ``RG`` ``cp.async.bulk`` box stores (the bulk-copy engine reads the box out: ~64 KB per token per SM through one
  engine); ``"stg"`` = every lane's 16 x ``st.global.v4`` of its row slices straight to the slot (the LSU path, no ring / fence / wait;
  the register-resident state only, ``NBUF`` 1 = the landing slab).
* ``BULK_ISSUE``: ``"single"`` (default) = one lane issues every bulk copy and store (warp 0 lane 0 the slab boxes and the box stores,
  warp 4 lane 0 the conv operands); ``"spread"`` = lane 0 of several warps (slab box ``b`` by warp ``b mod 4`` of the BA warpgroup, the
  input rows of token ``t`` by conv warp ``1 + t mod 3``, the store of box ``b`` by warp ``b mod 8``; each issuing lane keeps its own
  bulk-group ledger).  Under ``spread`` the BA warpgroup passes one more named barrier (``bar.sync 2, 128``) between warp 0 lane 0's
  state-mbarrier init + fence and the live arm: the issuing warps 1-3 complete bytes on that mbarrier, and a copy landing before the
  init is visible is undefined (the transaction is lost to the init and the wait never completes -- the form hung at 16 requests
  before this barrier existed).  Measured neutral against ``single`` at one, two and four requests on cc 10.0 parts: the issue cost is
  not on the critical path; the store engine's read-out of a 64 KB slab per token per SM is.
* ``STATE_STORE_PATH`` ``"stg"`` is a measured LOSS on cc 10.0 (close to twice the bulk path's time at one request): with this thread mapping an
  ``STG.128`` warp instruction touches 32 lines (16 B per 128-B line), ~4k L1 wavefronts per token per CTA against the bulk engine's
  read-out; kept as the one-flag A/B.
* ``PDL`` / ``PDL_RELEASE``: ``griddepcontrol.wait`` AFTER the predecessor-independent loads (slot chain, parameters, BA weight rows,
  the state slab, the conv weights and history) and BEFORE ``z``, ``hidden``, the ``x`` copies and every global write;
  ``griddepcontrol.launch_dependents`` at entry under ``PDL_RELEASE``.  CALLER OBLIGATION under ``PDL``: the kernel immediately
  preceding this launch in the stream must not write this layer's ``state``, conv state or slot indices -- the pre-wait loads
  read them BEFORE ``griddepcontrol.wait``, so a predecessor that writes them (the same layer launched twice in a row) races the
  slab / history copies against its stores: a wrong state, no error.  The serving chain satisfies it (the predecessor is the
  QKVZ projection, which writes ``x`` / ``z`` only); the entry cannot check it.
* ``TWO_CTAS_PER_HEAD``: a cluster of two CTAs per V head (grid ``(96, requests)`` in clusters of 2 along x), CTA ``r`` of the pair owning
  state rows ``64 r .. 64 r + 63`` (``CTA_ROWS`` 64: a 32 KB slab in, 32 KB of write-back per token, ``R = 64 TPR / 256`` rows per thread);
  both CTAs recompute the BA (decay, beta) and the conv + RMS (q, k, v) of every token from the same inputs; each shifts the V-conv history
  of its own 64 channels and the reader ticket wraps at 5 (six readers per K head).  The recurrence is row-local, so the T-token chain has
  ONE cross-CTA datum per token: the gated norm's sum of squares.  The reference association (``NORM_FORM``) never straddles the CTA
  boundary (chains of 4 or 8 rows; 64 is a multiple of both), so each CTA's norm warp sums its ``NORM_CHAINS_CTA`` local chains, ships
  the values to the peer (``NORM_XFER_FLOATS`` lanes, one ``st.async ... mbarrier::complete_tx::bytes.f32`` each through ``mapa``, completing
  ``NORM_XFER_BYTES`` on the peer's ``mb_norm`` stage ``t % 2``) and, after the peer's values land, every lane holds head chain ``l``'s value
  (its own or the received one, selected on the cluster rank) and the reference sequence of fp32 additions runs unchanged -> both CTAs
  hold the same bits, bitwise with the one-CTA form.  The ship sits at the top of the next token's iteration and the wait at its bottom,
  so the DSMEM latency hides under that token's passes; the two mbarrier stages alternate by token parity (the peer's ship of token t + 2
  follows its wait for token t + 1, which needs this CTA's ship of t + 1, which follows this CTA's wait of token t: the stage is never
  re-armed or rewritten before its reader is done).  The mbarrier inits are published cluster-wide by the init lane's
  ``fence.mbarrier_init.release.cluster`` + every thread's ``barrier.cluster.arrive.relaxed`` at entry and one ``barrier.cluster.wait``
  (acquire) per thread before the first ship (the live tail after the hand-off; the null tail at its start): the kernel's only cluster
  barrier; no GPU-scope membar, no release arrive.  Needs ``R >= 1`` (TPR >= 4 at 256 threads); use with ``CTAS_PER_SM`` 0 (the natural
  footprint, ~126 KB at T6, keeps one CTA per SM).  Served by default while ``requests x 96 <= SM count``.
* ``ROLLED_TOKENS``: the T-token loop traced ONCE (``cutlass.range(T, unroll=1)``; the token body is one ``@cute.jit`` step with the row
  registers and a two-deep ring of core values loop-carried) instead of once per token: every per-token shared address is computed from
  the loop variable, the deferred norm and outputs run under the uniform predicates ``t >= 1`` / ``t >= 2`` and the gate ``z`` of token
  ``t - 2`` is picked from the preloaded T values by a select chain.  The same arithmetic in the same order (bitwise with the unrolled
  body); what changes is the image: 3712 instructions against 5496 on the pair at T6 (the remaining straight-line code is the T-token BA /
  conv prologue), for the COLD instruction fetch of the serving regime (every layer's launch finds its code evicted).  Needs the
  register-resident rows (``STATE_STREAM`` off).  ON in ``PAIR_CFG`` (the fastest pair form with the L2 swept before every launch on a
  cc 10.0 part; marginally behind the unrolled pair when the code is warm) and in ``CFG`` (the one-CTA form at 2-3 requests, where the
  rolled loop is the larger win), OFF in ``TWO_CTA_CFG`` (the streamed form carries no row registers).
* ``DEBUG``: the intermediate dumps (``convout``, ``baout``, ``raw``) for byte-compares of the stages.
"""

from dataclasses import dataclass, replace
from typing import Optional

# --- the served geometry (the model's constants; the host entry declines everything else) ---
K_HEADS = 16
V_HEADS = 48
HEAD_DIM = 128
CONV_WIDTH = 4
HIDDEN = 5120
QKV_CHANNELS = 2 * K_HEADS * HEAD_DIM + V_HEADS * HEAD_DIM  # 10240
K_CHANNEL0 = K_HEADS * HEAD_DIM  # 2048: the first K channel of the QKV slab
V_CHANNEL0 = 2 * K_HEADS * HEAD_DIM  # 4096: the first V channel
OUT_ROW = V_HEADS * HEAD_DIM  # 6144 outputs per token row
THREADS = 256
WARPS = THREADS // 32
WARPGROUP = 128
TOKENS_MIN = 2  # one token per request is the decode step
CONV_HIST_TAPS = CONV_WIDTH - 1  # the three history columns a - 1 .. a + 1 a token window starts in
# the reference verify's BA partition: virtual thread v (0..255) sums elements c * 2048 + 8 v + j of the three 2048-element chunks
# while c * 2048 + 8 v < 5120 (chunk 2 ends at v 128); every lane of the BA warpgroup carries virtual threads v = lane and v = lane + 128
BA_VTHREADS = 256
BA_CHUNK = 2048
BA_CHUNKS = (HIDDEN + BA_CHUNK - 1) // BA_CHUNK  # 3
RMS_LANES = 16  # the reference RMS layout: lane l of a 16-lane group holds the 8 raw q / k values of columns 8 l .. 8 l + 7
RMS_COLS = HEAD_DIM // RMS_LANES
TICKET_STRIDE = 512  # counts[request * 512 + kh * 32]: the per-(request, K head) reader ticket of the conv history
TICKET_LIMIT = V_HEADS // K_HEADS - 1  # 2: the third reader of a K head's conv history is the last
NORM_FORMS = ("ctas2", "ctas1")
STATE_LOAD_PATHS = ("bulk",)
STATE_STORE_PATHS = ("bulk", "stg")
BULK_ISSUES = ("spread", "single")

# the state slab of one V head (128 x 128 fp32) and its mbarrier: ONE arrive.expect_tx(STATE_SLAB_BYTES) by warp 0's lane 0, the RG
# box copies complete the bytes
STATE_SLAB_BYTES = HEAD_DIM * HEAD_DIM * 4  # 65536
STATE_MBAR_ARRIVERS = 1
# the conv-operand mbarrier: ONE arrive.expect_tx(conv_bulk_bytes(T)) by the conv warpgroup's first lane, the bulk copies complete
CONV_MBAR_ARRIVERS = 1
# one 16-B chunk between two state boxes shifts the next box by 4 banks: with a phase's 8 lanes on RG distinct boxes (TPR >= 4) or
# RG consecutive row groups, their 16-B chunks land in 8 distinct bank quads (the bank map: bank = 4 rg + (128 / TPR) tpr + 4 c4)
STATE_BOX_PAD_BYTES = 16
# TWO_CTAS_PER_HEAD: the peer's norm chain values of token t land by NORM_XFER_FLOATS st.async stores of 4 bytes each on this CTA's
# mb_norm stage t % 2; its init count is the ONE local arrive.expect_tx by the norm warp's lane 0 (a complete_tx carries no software
# arrive) -- SUM(arriving lanes) == init count on every path; two stages (token parity), a receive slot of up to 16 chain values each
NORM_MBAR_ARRIVERS = 1
NORM_MBAR_STAGES = 2
NORM_RECV_MAX_FLOATS = 16

# The cc 10.0 / 10.3 part's shared memory per SM (228 KB), its per-CTA opt-in maximum (227 KB) and the driver's per-CTA reserve (1 KB);
# the host entry REPLACES the first two with the device's own readings, so a hardcoded size never decides an occupancy pad on a
# part it does not describe.
SM100_SMEM_PER_SM_BYTES = 228 * 1024
SM100_SMEM_OPTIN_BYTES = 227 * 1024
# The cc 12.0 part's readings (100 KB per SM, 99 KB per-CTA opt-in, the same 1 KB reserve; a 170-SM RTX 5090-class part): documentation
# and tests only, never a default -- the host entry reads the part.
SM120_SMEM_PER_SM_BYTES = 100 * 1024
SM120_SMEM_OPTIN_BYTES = 99 * 1024
SMEM_CTA_RESERVE_BYTES = 1024
SMEM_PAD_MARGIN_BYTES = 4096  # keeps the cap robust to the allocation granularity


def conv_bulk_copies(tokens: int) -> int:
    """The ``cp.async.bulk`` copies of one CTA's conv operands: 3 weight runs, 3 pieces x 3 history taps, 3 pieces x T input rows."""
    return 3 + 3 * CONV_HIST_TAPS + 3 * tokens


def conv_bulk_bytes(tokens: int) -> int:
    """The bytes those copies deliver on the conv-operand mbarrier (= its ``expect_tx``): weights 3 x 1 KB, history 9 x 256 B,
    inputs 3 T x 256 B."""
    return 3 * HEAD_DIM * CONV_WIDTH * 2 + 3 * CONV_HIST_TAPS * HEAD_DIM * 2 + 3 * tokens * HEAD_DIM * 2


def norm_chain_rows(norm_form: str) -> int:
    """Rows per gated-norm sum-of-squares chain: the reference's half-warp row count (4 at two CTAs per head, 8 at one)."""
    return 4 if norm_form == "ctas2" else 8


def norm_form_for(requests: int) -> str:
    """The reference verify kernel's CTAs-per-head policy as a norm association: two CTAs at one request, one otherwise."""
    return "ctas2" if requests == 1 else "ctas1"


def _slot16(nbytes: int) -> int:
    return (nbytes + 15) // 16 * 16


@dataclass(frozen=True)
class Cfg:
    TPR: int = 4
    K_RESIDENT: bool = True
    NBUF: int = 3
    STATE_STREAM: bool = False
    MIN_CTAS_PER_SM: int = 1
    CTAS_PER_SM: int = 1
    BA_PREFETCH_CHUNKS: int = 3
    NORM_FORM: Optional[str] = None
    STATE_LOAD_PATH: str = "bulk"
    STATE_STORE_PATH: str = "bulk"
    BULK_ISSUE: str = "single"
    PDL: bool = False
    PDL_RELEASE: bool = False
    TWO_CTAS_PER_HEAD: bool = False
    ROLLED_TOKENS: bool = False
    DEBUG: bool = False
    SMEM_PER_SM_BYTES: int = SM100_SMEM_PER_SM_BYTES
    SMEM_OPTIN_BYTES: int = SM100_SMEM_OPTIN_BYTES

    # --- derived (read-only) ---
    @property
    def THREADS(self) -> int:
        return THREADS

    @property
    def WARPS(self) -> int:
        return WARPS

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
        return self.CTA_ROWS * self.TPR // THREADS

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
    def BULK_STORE(self) -> bool:
        return self.STATE_STORE_PATH == "bulk"

    @property
    def SPREAD_ISSUE(self) -> bool:
        return self.BULK_ISSUE == "spread"

    @property
    def STATE_BOX_BYTES(self) -> int:
        """Bytes of one state box = ``BOX_ROWS`` rows of 128 fp32, one contiguous run in global memory and in shared memory."""
        return self.BOX_ROWS * HEAD_DIM * 4

    @property
    def STATE_BOX_PITCH_ELEMS(self) -> int:
        """fp32 elements between two boxes in shared memory (the box plus the 16-B pad)."""
        return self.BOX_ROWS * HEAD_DIM + STATE_BOX_PAD_BYTES // 4

    @property
    def SLAB_SMEM_ELEMS(self) -> int:
        """fp32 elements of one ring buffer: the RG padded boxes of a slab."""
        return self.RG * self.STATE_BOX_PITCH_ELEMS

    @property
    def RING_SMEM_ELEMS(self) -> int:
        return self.NBUF * self.SLAB_SMEM_ELEMS

    @property
    def NORM_CHAINS(self) -> int:
        """Sum-of-squares chains of the gated norm (one lane of the norm warp each): 32 four-row chains or 16 eight-row chains."""
        return HEAD_DIM // norm_chain_rows(self.NORM_FORM)

    @property
    def NORM_CHAIN_ROWS(self) -> int:
        return norm_chain_rows(self.NORM_FORM)

    @property
    def NORM_CHAINS_CTA(self) -> int:
        """Sum-of-squares chains whose rows this CTA owns (16 or 8 on the two-CTA form; all of them on the one-CTA form)."""
        return self.NORM_CHAINS // self.CTAS_PER_HEAD

    @property
    def NORM_XFER_FLOATS(self) -> int:
        """Chain values each CTA of a pair ships to its peer per token (one lane, one ``st.async`` of 4 bytes each); 0 on the one-CTA form."""
        return self.NORM_CHAINS_CTA if self.TWO_CTAS_PER_HEAD else 0

    @property
    def NORM_XFER_BYTES(self) -> int:
        """The peer's ``mb_norm`` expect_tx per token (= the bytes the NORM_XFER_FLOATS ``st.async`` stores complete)."""
        return 4 * self.NORM_XFER_FLOATS

    def natural_smem_bytes(self, tokens: int) -> int:
        """The kernel's own shared-memory footprint for ``tokens`` per request, buffer by buffer in allocation order (16-byte slots):
        the x rows, conv weights, conv history, raw q / k, the padded q / k slices, v, the per-token scalars, the two core-value
        slots of the norm, the two norm inverses, the BA partials, the ticket, the two mbarriers, (the pair's norm receive slots and
        its two mbarrier stages,) the state ring."""
        return (
            _slot16(3 * tokens * HEAD_DIM * 2)  # x_s
            + 3 * HEAD_DIM * CONV_WIDTH * 2  # cw_s
            + 3 * CONV_HIST_TAPS * HEAD_DIM * 2  # hist_s
            + 2 * tokens * HEAD_DIM * 4  # qk_raw
            + 2 * tokens * self.TPR * self.SLICE_PITCH * 4  # q_s, k_s
            + tokens * HEAD_DIM * 4  # v_s
            + _slot16(4 * tokens * 4)  # scal_s
            + 2 * HEAD_DIM * 4  # o_s
            + 16  # inv_s
            + _slot16(2 * WARPS * tokens * 4)  # ba_part
            + 16  # ticket_s
            + 16
            + 16  # mb_ops, mb_state (8 bytes each in a 16-byte slot)
            + ((NORM_MBAR_STAGES * NORM_RECV_MAX_FLOATS * 4 + NORM_MBAR_STAGES * 8) if self.TWO_CTAS_PER_HEAD else 0)  # norm_recv_s, mb_norm
            + self.RING_SMEM_ELEMS * 4  # ring
        )

    def smem_pad_bytes(self, tokens: int) -> int:
        """The occupancy pad: sized so that ``CTAS_PER_SM + 1`` CTAs (each its footprint + pad + the driver's reserve) exceed the
        device's shared memory per SM while ``CTAS_PER_SM`` of them fit -- the SM holds at most ``CTAS_PER_SM`` CTAs by shared memory."""
        if self.CTAS_PER_SM == 0:
            return 0
        pad = self.SMEM_PER_SM_BYTES // (self.CTAS_PER_SM + 1) + SMEM_PAD_MARGIN_BYTES - SMEM_CTA_RESERVE_BYTES - self.natural_smem_bytes(tokens)
        return max(0, _slot16(pad))

    def smem_bytes(self, tokens: int) -> int:
        return self.natural_smem_bytes(tokens) + self.smem_pad_bytes(tokens)

    def co_resident_ctas(self, tokens: int) -> int:
        """The CTAs of this form that share one SM on the part at ``tokens`` per request, as the host can GUARANTEE them from the two
        budgets it knows: by shared memory, how many launch footprints plus the driver's per-CTA reserve fit ``SMEM_PER_SM_BYTES``; by
        registers, ``MIN_CTAS_PER_SM`` (the cap the launch asks ptxas for -- 2 = 128 registers, two CTAs of 256 threads in the 64 K file;
        an uncapped form may take up to 255, so the host counts one) -- the smaller of the two.  The ONE predicate the default pick, its
        docs and its tests read: a ``MIN_CTAS_PER_SM`` 2 form is worth its cap only where this is 2 at the minimum token count (the cc
        10.0 readings: ``TWO_CTA_CFG`` 2; the cc 12.0 readings: 1, 2 x (79,200 + 1,024) B > 102,400 B), and the NBUF-1 pair under the cap
        is two per SM on the cc 12.0 readings at T <= 3 (2 x (50,112 + 1,024) = 102,272 B) and one from T 4 (53,648 B)."""
        by_smem = self.SMEM_PER_SM_BYTES // (self.smem_bytes(tokens) + SMEM_CTA_RESERVE_BYTES)
        return min(by_smem, self.MIN_CTAS_PER_SM)

    def max_tokens(self) -> int:
        """The largest tokens-per-request count whose footprint fits this part's per-CTA opt-in (and, with a pad, the SM)."""
        best = 0
        for tokens in range(TOKENS_MIN, 65):
            if self.natural_smem_bytes(tokens) > self.SMEM_OPTIN_BYTES:
                break
            if self.CTAS_PER_SM and self.CTAS_PER_SM * (self.smem_bytes(tokens) + SMEM_CTA_RESERVE_BYTES) > self.SMEM_PER_SM_BYTES:
                break
            best = tokens
        return best


def validate_cfg(cfg: Cfg) -> None:
    """Raise the typed error of every unexpressible knob combination (structural invariants raise here, never a module assert)."""
    if cfg.TPR not in (2, 4, 8, 16):
        raise ValueError(f"gdn_verify_fused: TPR must be 2, 4, 8 or 16, got {cfg.TPR}")
    if cfg.CTA_ROWS * cfg.TPR % THREADS or cfg.R < 1 or cfg.R * cfg.C * THREADS != cfg.CTA_ROWS * HEAD_DIM or cfg.C % 8 or cfg.BOX_ROWS != cfg.R * WARPS:
        raise ValueError(f"gdn_verify_fused: TPR={cfg.TPR} does not tile the {cfg.CTA_ROWS} x 128 state rows of one CTA over {THREADS} threads")
    if cfg.TWO_CTAS_PER_HEAD and any(cfg.CTA_ROWS % norm_chain_rows(f) or (HEAD_DIM // norm_chain_rows(f)) % cfg.CTAS_PER_HEAD for f in NORM_FORMS):
        raise ValueError("gdn_verify_fused: TWO_CTAS_PER_HEAD needs every norm association's chains to split by CTA without straddling")
    if cfg.TWO_CTAS_PER_HEAD and cfg.NORM_FORM is not None and cfg.NORM_CHAINS_CTA > NORM_RECV_MAX_FLOATS:
        raise ValueError(
            f"gdn_verify_fused: TWO_CTAS_PER_HEAD ships at most {NORM_RECV_MAX_FLOATS} chain values per token, {cfg.NORM_FORM} needs {cfg.NORM_CHAINS_CTA}"
        )
    if cfg.NBUF not in (1, 2, 3, 4):
        raise ValueError(f"gdn_verify_fused: NBUF must be 1 .. 4, got {cfg.NBUF}")
    if cfg.STATE_STREAM and cfg.NBUF > 2:
        raise ValueError(f"gdn_verify_fused: STATE_STREAM keeps the state in the boxes and alternates two of them at most; NBUF={cfg.NBUF}")
    if cfg.MIN_CTAS_PER_SM not in (1, 2):
        raise ValueError(f"gdn_verify_fused: MIN_CTAS_PER_SM must be 1 or 2, got {cfg.MIN_CTAS_PER_SM}")
    if cfg.CTAS_PER_SM not in (0, 1, 2):
        raise ValueError(f"gdn_verify_fused: CTAS_PER_SM must be 0, 1 or 2, got {cfg.CTAS_PER_SM}")
    if cfg.CTAS_PER_SM and cfg.MIN_CTAS_PER_SM > cfg.CTAS_PER_SM:
        raise ValueError(
            f"gdn_verify_fused: MIN_CTAS_PER_SM={cfg.MIN_CTAS_PER_SM} (the register cap aims at that many CTAs per SM) contradicts CTAS_PER_SM={cfg.CTAS_PER_SM} (the pad caps the SM at fewer)"
        )
    if cfg.BA_PREFETCH_CHUNKS not in (1, 2, 3):
        raise ValueError(f"gdn_verify_fused: BA_PREFETCH_CHUNKS must be 1, 2 or 3, got {cfg.BA_PREFETCH_CHUNKS}")
    if cfg.NORM_FORM is not None and cfg.NORM_FORM not in NORM_FORMS:
        raise ValueError(f"gdn_verify_fused: NORM_FORM must be one of {NORM_FORMS} or None (the reference policy per requests), got {cfg.NORM_FORM!r}")
    if cfg.STATE_LOAD_PATH not in STATE_LOAD_PATHS:
        raise ValueError(f"gdn_verify_fused: STATE_LOAD_PATH must be one of {STATE_LOAD_PATHS}, got {cfg.STATE_LOAD_PATH!r}")
    if cfg.STATE_STORE_PATH not in STATE_STORE_PATHS:
        raise ValueError(f"gdn_verify_fused: STATE_STORE_PATH must be one of {STATE_STORE_PATHS}, got {cfg.STATE_STORE_PATH!r}")
    if cfg.STATE_STORE_PATH == "stg" and (cfg.STATE_STREAM or cfg.NBUF != 1):
        raise ValueError(
            "gdn_verify_fused: STATE_STORE_PATH='stg' stores the register-resident rows straight to the slot: it needs STATE_STREAM off and NBUF=1 (the landing slab only)"
        )
    if cfg.ROLLED_TOKENS and cfg.STATE_STREAM:
        raise ValueError("gdn_verify_fused: ROLLED_TOKENS carries the register-resident rows across the token loop; STATE_STREAM has none")
    if cfg.BULK_ISSUE not in BULK_ISSUES:
        raise ValueError(f"gdn_verify_fused: BULK_ISSUE must be one of {BULK_ISSUES}, got {cfg.BULK_ISSUE!r}")
    if cfg.SMEM_PER_SM_BYTES <= 0 or cfg.SMEM_OPTIN_BYTES <= 0 or cfg.SMEM_OPTIN_BYTES > cfg.SMEM_PER_SM_BYTES:
        raise ValueError(
            f"gdn_verify_fused: SMEM_PER_SM_BYTES={cfg.SMEM_PER_SM_BYTES} / SMEM_OPTIN_BYTES={cfg.SMEM_OPTIN_BYTES} must be positive with the per-CTA opt-in no larger than the SM"
        )
    if cfg.max_tokens() < TOKENS_MIN:
        raise ValueError(
            f"gdn_verify_fused: the shared memory of this form does not fit {TOKENS_MIN} tokens per request on a {cfg.SMEM_OPTIN_BYTES}-byte opt-in"
        )


def validate_tokens(cfg: Cfg, tokens: int) -> None:
    """The typed decline of a tokens-per-request count this form cannot hold (the ring plus the per-token arrays must fit the
    per-CTA opt-in and, with an occupancy pad, ``CTAS_PER_SM`` CTAs must fit the SM)."""
    if not isinstance(tokens, int) or tokens < TOKENS_MIN:
        raise NotImplementedError(
            f"gdn_verify_fused serves at least {TOKENS_MIN} tokens per request (one token per request is the decode step); got {tokens!r}"
        )
    if tokens > cfg.max_tokens():
        raise NotImplementedError(
            f"gdn_verify_fused: {tokens} tokens per request need {cfg.natural_smem_bytes(tokens)} B of shared memory per CTA; this form (TPR {cfg.TPR}, NBUF {cfg.NBUF}, CTAS_PER_SM {cfg.CTAS_PER_SM}) serves at most {cfg.max_tokens()} on this part"
        )


def make_cfg(**overrides) -> Cfg:
    """A validated ``Cfg``.  The field defaults are the one-CTA-per-SM form of the measured pick (``CFG``) with the token loop UNROLLED
    (``CFG`` itself sets ``ROLLED_TOKENS``): TPR 4, the 3-deep ring, the register-resident state, one CTA per SM by the pad and the
    255-register budget, PDL off."""
    cfg = replace(Cfg(), **overrides)
    validate_cfg(cfg)
    return cfg


def resolve_cfg(cfg: Cfg, requests: int) -> Cfg:
    """``cfg`` with its ``NORM_FORM`` resolved for ``requests`` (the reference policy when None), validated."""
    if not isinstance(requests, int) or requests < 1:
        raise ValueError(f"gdn_verify_fused: requests must be a positive int, got {requests!r}")
    out = cfg if cfg.NORM_FORM is not None else replace(cfg, NORM_FORM=norm_form_for(requests))
    validate_cfg(out)
    return out


# the three forms of the default (the module docstring's "Default" paragraph)
PAIR_CFG = make_cfg(
    TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, ROLLED_TOKENS=True
)  # a cluster of two CTAs per head, half the rows each, the token body traced once -- while the requests x 96 CTAs have an SM each
CFG = make_cfg(
    ROLLED_TOKENS=True
)  # one CTA per SM: TPR 4, NBUF 3, register-resident, the token body traced once -- while the requests x 48 CTAs have an SM each
TWO_CTA_CFG = make_cfg(
    TPR=4, NBUF=1, STATE_STREAM=True, MIN_CTAS_PER_SM=2, CTAS_PER_SM=0, BA_PREFETCH_CHUNKS=1
)  # two CTAs per SM, one wave -- where the part's shared memory holds two (co_resident_ctas(TOKENS_MIN) >= 2: the cc 10.0 readings)

# The forms a 99 KB per-CTA opt-in holds (the three picks above carry a 3-deep ring of 64 KB -- or 32 KB half -- slabs and hold no token
# count there): the same kernel body under existing knobs, bitwise with the picks (no knob moves an fp32 addition; the sm_120a run of
# record replayed every saved reference set byte for byte).  Footprint law: 6,464 + 3,536 T + ring (+ 144 on the pair), ring = NBUF x
# 65,664 (one CTA) or NBUF x 32,896 (the pair).  What the ring depth buys: at NBUF 3 token t + 1's write-back waits only for the store of
# t - 2, at NBUF 2 for the store of t - 1, at NBUF 1 the box read-out is exposed between the two passes of every token.
PAIR_NBUF2_CFG = make_cfg(
    TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, ROLLED_TOKENS=True, NBUF=2
)  # the half-slab pair with a 2-deep ring: 79,472 / 86,544 / 93,616 B at T 2 / 4 / 6, T <= 8 on the cc 12.0 reading (187 registers, 0 spills on sm_120a)
PAIR_NBUF1_CFG = make_cfg(
    TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, ROLLED_TOKENS=True, NBUF=1
)  # the half-slab pair, the landing slab only: 46,576 / 53,648 / 60,720 B, T <= 17 there (187 / 0; under MIN_CTAS_PER_SM 2: 128 / 0, two per SM at T <= 3)
ONE_CTA_NBUF1_CFG = make_cfg(
    ROLLED_TOKENS=True, NBUF=1
)  # the whole slab, the landing slab only, no cluster: 79,200 / 86,272 / 93,344 B, T <= 8 there (183 / 0)

# The ladder of each SM-count regime: the primary form (the measured cc 10.0 pick) first, then the forms a smaller shared memory still
# holds, deepest ring first.  ``default_cfg`` returns the primary wherever it is admissible on the part (every cc 10.x part), so the picks
# and the token envelopes there never move; on a part whose shared memory does not hold the primary it walks the rest, token-aware.
PAIR_FORMS = (PAIR_CFG, PAIR_NBUF2_CFG, PAIR_NBUF1_CFG)
ONE_CTA_FORMS = (CFG, ONE_CTA_NBUF1_CFG, PAIR_NBUF1_CFG)
BEYOND_FORMS = (TWO_CTA_CFG, ONE_CTA_NBUF1_CFG, PAIR_NBUF1_CFG)


def part_readings(smem_per_sm: int, smem_optin: int) -> dict:
    """The ``Cfg`` fields of a part reading ``(shared memory per SM, per-CTA opt-in)`` in bytes, validated (positive, the opt-in no larger
    than the SM) -- ``replace(form, **part_readings(...))`` is a form on that part."""
    for name, value in (("smem_per_sm", smem_per_sm), ("smem_optin", smem_optin)):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"gdn_verify_fused: {name} must be a positive int (a device reading in bytes), got {value!r}")
    if smem_optin > smem_per_sm:
        raise ValueError(f"gdn_verify_fused: the per-CTA shared-memory opt-in ({smem_optin} B) cannot exceed the SM's ({smem_per_sm} B)")
    return dict(SMEM_PER_SM_BYTES=smem_per_sm, SMEM_OPTIN_BYTES=smem_optin)


def admissible(form: Cfg, smem_per_sm: int = SM100_SMEM_PER_SM_BYTES, smem_optin: int = SM100_SMEM_OPTIN_BYTES) -> bool:
    """Whether ``form`` may serve on a part: its shared memory holds the minimum token count in the per-CTA opt-in, and a form that asks
    ptxas for the two-CTA register cap (``MIN_CTAS_PER_SM`` 2) has two CTAs per SM to show for it (``co_resident_ctas(TOKENS_MIN)`` >= 2)
    -- a cap on a part whose shared memory keeps the SM at one CTA buys nothing and spills (sm_120a: ``TWO_CTA_CFG`` 34 STL / LDL)."""
    on = replace(form, **part_readings(smem_per_sm, smem_optin))
    return on.max_tokens() >= TOKENS_MIN and (on.MIN_CTAS_PER_SM == 1 or on.co_resident_ctas(TOKENS_MIN) >= 2)


def _no_form(forms, smem_per_sm: int, smem_optin: int) -> NotImplementedError:
    """The typed decline of a part none of ``forms`` fits: it names the opt-in and the smallest footprint asked of it."""
    need = min(form.natural_smem_bytes(TOKENS_MIN) for form in forms)
    return NotImplementedError(
        f"gdn_verify_fused: this part's {smem_optin}-byte shared-memory opt-in per CTA ({smem_per_sm} B per SM) holds no verify form at "
        f"{TOKENS_MIN} tokens per request (the smallest form asks {need} B)"
    )


def first_fitting(forms, smem_per_sm: int = SM100_SMEM_PER_SM_BYTES, smem_optin: int = SM100_SMEM_OPTIN_BYTES) -> Cfg:
    """The first of ``forms`` admissible on the part (the form an entry starts from: ``CFG`` on the cc 10.0 readings, ``ONE_CTA_NBUF1_CFG``
    on the cc 12.0 ones for ``ONE_CTA_FORMS``), else the typed decline of a part none of them fits."""
    for form in forms:
        if admissible(form, smem_per_sm, smem_optin):
            return form
    raise _no_form(forms, smem_per_sm, smem_optin)


def _ladder(requests: int, sm_count: int):
    """The regime's ladder by the SM count: the pair forms while every CTA of the ``requests x 96`` cluster grid has an SM of its own, the
    one-CTA forms while every CTA of the ``requests x 48`` grid has one, the beyond forms past that.  Plan-time facts, validated."""
    if not isinstance(requests, int) or requests < 1:
        raise ValueError(f"gdn_verify_fused: requests must be a positive int, got {requests!r}")
    if not isinstance(sm_count, int) or sm_count < 1:
        raise ValueError(f"gdn_verify_fused: the SM count must be a positive int, got {sm_count!r}")
    if requests * V_HEADS * PAIR_CFG.CTAS_PER_HEAD <= sm_count:
        return PAIR_FORMS
    return ONE_CTA_FORMS if requests * V_HEADS <= sm_count else BEYOND_FORMS


def default_cfg(
    requests: int,
    sm_count: int,
    smem_per_sm: int = SM100_SMEM_PER_SM_BYTES,
    smem_optin: int = SM100_SMEM_OPTIN_BYTES,
    tokens: Optional[int] = None,
) -> Cfg:
    """The pick per requests on a part (its SM count and shared-memory readings; the defaults are the cc 10.0 readings, so a two-argument
    call is the measured cc 10.0 pick).  The regime by SM count selects the ladder: ``PAIR_CFG`` (two CTAs per head) while every CTA of
    the ``requests x 96`` cluster grid has an SM of its own (one request on a 148- or 152-SM part, two on a 212-SM one), ``CFG`` (one CTA
    per SM) while every CTA of the ``requests x 48`` grid has one, ``TWO_CTA_CFG`` (two CTAs per SM, one wave) once the grid outgrows the
    device's SMs -- on a 148- or 152-SM part the switches are between 1 request (96 cluster CTAs) and 2 (192 as a pair; 96 one-CTA CTAs),
    and between 3 requests (144 CTAs) and 4 (192).  The regime's primary form is returned wherever it is ``admissible`` on the readings
    (every cc 10.x part), and ``tokens`` is then ignored: ``validate_tokens`` bounds the step by that form as before (35 / 8 / 45 on the cc
    10.0 readings).  Where the primary is not admissible (a 99 KB opt-in holds none of the 3-deep rings; a 100 KB SM holds no two
    ``TWO_CTA_CFG`` footprints) the first admissible form of the rest that holds ``tokens`` (the minimum token count when None) is
    returned, else the admissible form with the largest token bound (so ``validate_tokens`` names the part's true bound), else the typed
    decline of a part with no form.  On the cc 12.0 readings (170 / 188 SMs): one request ``PAIR_NBUF2_CFG`` for T <= 8 (96 cluster CTAs,
    one wave), two and more ``ONE_CTA_NBUF1_CFG`` for T <= 8 (one wave to 3 requests, two from 4), ``PAIR_NBUF1_CFG`` for T 9 .. 17 at every
    requests count, T >= 18 the decline naming 17; ``TWO_CTA_CFG`` never (one CTA per SM there, 34 spills).  All are plan-time facts (never
    a runtime value)."""
    ladder = _ladder(requests, sm_count)
    if tokens is not None and (not isinstance(tokens, int) or tokens < 1):
        raise ValueError(f"gdn_verify_fused: tokens must be a positive int or None, got {tokens!r}")
    want = max(TOKENS_MIN, tokens or TOKENS_MIN)
    if admissible(ladder[0], smem_per_sm, smem_optin):
        return ladder[0]
    rest = [form for form in ladder[1:] if admissible(form, smem_per_sm, smem_optin)]
    if not rest:
        raise _no_form(ladder, smem_per_sm, smem_optin)
    on = part_readings(smem_per_sm, smem_optin)
    for form in rest:
        if replace(form, **on).max_tokens() >= want:
            return form
    return max(rest, key=lambda form: replace(form, **on).max_tokens())


def max_tokens_served(requests: int, sm_count: int, smem_per_sm: int = SM100_SMEM_PER_SM_BYTES, smem_optin: int = SM100_SMEM_OPTIN_BYTES) -> int:
    """The largest tokens-per-request count ``default_cfg`` serves for ``requests`` on the part: the primary form's bound where it is
    admissible (35 / 8 / 45 on the cc 10.0 readings by regime), else the largest bound over the admissible rest of the ladder (17 on the
    cc 12.0 readings at every requests count); the typed decline of a part with no form.  The one bound the entry, its ``check_support``
    and FlashInfer's probe answer alike."""
    ladder = _ladder(requests, sm_count)
    on = part_readings(smem_per_sm, smem_optin)
    if admissible(ladder[0], smem_per_sm, smem_optin):
        return replace(ladder[0], **on).max_tokens()
    bounds = [replace(form, **on).max_tokens() for form in ladder[1:] if admissible(form, smem_per_sm, smem_optin)]
    if not bounds:
        raise _no_form(ladder, smem_per_sm, smem_optin)
    return max(bounds)
