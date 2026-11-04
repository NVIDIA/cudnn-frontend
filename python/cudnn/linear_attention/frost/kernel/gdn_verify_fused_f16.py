# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Fused Gated DeltaNet VERIFY step (SM100-class parts, Cutlass primitives) for the geometry of ``gdn_verify_fused_config.py`` (the
Qwen 3.5-architecture dense 27B GDN sub-layer at TP 1): the speculative causal conv + SiLU over the request's ``T`` tokens, the BA projection of every token,
the Q / K RMS scales, the gated delta-rule recurrence over the tokens (the state after every token written to its own slot), the
gated RMSNorm, the static FP8 output quantization and the conv-history shift, in one launch after the stock FP8 QKVZ projection of
an MTP verify step.

Per CTA (one V head ``h`` of one request ``n``; K head ``kh = h // 3``), 256 threads = two warpgroups.  Lane ``l`` of warp ``w`` owns
the column slice ``tpr = l // RG`` (``C = 128 / TPR`` columns) of the ``R = CTA_ROWS x TPR / 256`` consecutive state rows ``CTA_ROWS r +
BOX_ROWS * rg + R * w + m`` (``rg = l % RG``, ``RG = 32 / TPR``, ``BOX_ROWS = CTA_ROWS / RG``; ``CTA_ROWS`` 128 and ``r`` 0 on the one-CTA
form): the RG row groups of a warp sit in RG different state BOXES, so the eight lanes of a 128-bit shared-memory phase never share a
bank.  The row slice is REGISTER-RESIDENT across the request's tokens (``R x C`` = 64 fp32 per thread, 32 on the pair) unless
``STATE_STREAM`` keeps it in the boxes.  ``T`` (tokens per request) is a compile-time constant; the request's token ``t`` is row
``bos + t`` of the row-major operands (``bos = cu_seqlens[n]``).

``TWO_CTAS_PER_HEAD`` (the pair; the default pick while ``requests x 96 <= SM count``): the grid is ``(96, requests)`` in clusters of two
along x, CTA ``r = bx % 2`` of head ``bx // 2`` owns rows ``64 r .. 64 r + 63`` (a 32 KB slab in, 32 KB of write-back per token through
its own SM's bulk-copy path), both CTAs recompute the BA and the conv + RMS of every token, each shifts the V-conv history of its own
64 channels, the reader ticket wraps at 5.  The recurrence is row-local; the ONE cross-CTA datum per token is the gated norm's sum of
squares: the norm warp of each CTA sums its ``NORM_CHAINS_CTA`` local chains of token ``t - 1`` at the TOP of iteration ``t`` (the slot
was published by barrier ``t - 1``), lane 0 arms ``mb_norm[(t - 1) % 2]`` with ``expect_tx(NORM_XFER_BYTES)``, lanes ``< NORM_CHAINS_CTA``
ship their chain value into the peer's ``norm_recv_s`` slot by one ``st.async ... mbarrier::complete_tx::bytes.f32`` each (both addresses
``mapa``-translated), the token's passes run, and at the BOTTOM the warp waits that stage (parity ``((t - 1) // 2) & 1``), every lane
selects head chain ``lane``'s value (own or received, by the cluster rank) and the reference sequence of additions runs unchanged on
both ranks -> ``inv_{t-1}``; the epilogue does the same for token ``T - 1`` around the outputs of ``T - 2``.  Cluster bring-up: the
init lane's ``fence.mbarrier_init.release.cluster`` + every thread's ``barrier.cluster.arrive.relaxed`` at entry (both warpgroups, on
every path) + one ``barrier.cluster.wait`` (acquire) per thread before anything crosses the pair (the live tail right after the
hand-off barrier; the null tail at its start) -- the kernel's only cluster barrier; no GPU-scope membar, no release arrive.

  entry      every thread: the slot chain (``a = accepted[n]``, ``src = indices[n, a - 1]`` when ``0 < a <= T``, ``cv_slot =
             indices[n, 0]``, ``bos``, ``ntok``; a request is LIVE when ``src > 0``, ``cv_slot > 0`` and ``ntok > 0``);
             warpgroup 0 (warps 0-3): warp 0 lane 0 initialises the state mbarrier (+ fence; under ``BULK_ISSUE`` spread ``bar.sync 2,
             128`` publishes it to the issuing warps 1-3 before the live arm); the two BA weight rows of the reference's
             256-virtual-thread partition (10 x LDG.128 per lane: virtual threads ``lane`` and ``lane + 128``);
             warpgroup 1 (warps 4-7): warp 4 lane 0 initialises the conv-operand mbarrier (+ fence), ``bar.sync 1, 128`` publishes it.
  live       warp 0 lane 0 arms ``expect_tx(65536)``; the RG ``cp.async.bulk`` boxes of the source slab land in padded shared-memory
             boxes (box ``b`` issued by BA warp ``b mod 4``'s lane 0 under ``BULK_ISSUE`` spread, by warp 0 lane 0 under single);
             warp 4 lane 0 arms ``expect_tx(conv_bulk_bytes(T))`` and copies the head's conv weights (3 x 1 KB) and the history taps
             ``a - 1 .. a + 1`` of its three channel pieces (9 x 256 B); the ``T x 3`` input rows (256 B each) by conv warp
             ``5 + t mod 3``'s lane 0 (spread) after the PDL wait when any; under PDL every thread's ``griddepcontrol.wait`` sits here,
             before ``hidden`` / ``x`` / ``z`` and every global write.
  BA (WG0)   the reference partition per virtual thread: fp32 chains over the three 2048-element chunks (``+= lo * b_lo; += hi * b_hi``
             per word, every token its own chain), one 5-step butterfly per (virtual warp, token), the 8 virtual-warp partials in
             shared memory (``bar.sync 2, 128``), the totals summed in virtual-warp order by threads ``< 2 T``, rounded to bf16:
             ``beta_t = 1 / (1 + exp(-b))``, ``decay_t = exp(-exp(A_log) * softplus(a + dt_bias))`` (linear above 20) -> ``scal_s[t]``.
  conv (WG1) mbarrier wait; warp 4 lane 0 draws the reader ticket (``atom.relaxed.gpu.inc``); each thread's Q, K and V channel of
             every token: the window ``seq[t .. t + 3]`` of ``seq = [old[a - 1], old[a], old[a + 1], x_0 .. x_{T-1}]``, ``acc +=
             bf16(tap * weight)`` x 4 in tap order, the stock SiLU, bf16 -> the raw q / k of the token (smem) and its v (permuted
             index); ``bar.sync 1, 128``; half-warp ``hw`` on token ``8 r + hw``: the reference 16-lane RMS partials and butterflies ->
             ``qscale_t`` / ``kscale_t`` -> the SCALED q / k slices of the token into the padded per-slice arrays.
  bar.sync 0 {q, k, v of every token, decay / beta of every token, the ticket} published; the state mbarrier init published.
  slab       every thread waits the state mbarrier; register-resident: 16 x LDS.128 of its row slices (conflict-free by the box pad).
  token t    (unrolled) pass 1: ``h = mul.rn(S, decay_t)``; per 8-column group an 8-step FFMA chain against ``k_t`` (from +0), the
             balanced tree over the thread's groups, ``log2(TPR)`` xor shuffles (= the reference 16-leaf tree); ``inn = (v_t - dot) *
             beta_t``; pass 2: ``h = fma.rn(inn, k_t, h)``; ``o = h . q_t`` (same tree); ``core = f32(bf16(o))``; the updated slice
             16 x STS.128 into ring box ``t % NBUF`` (streamed: read from box ``t % NBUF``, written to ``(t + 1) % NBUF``); the
             ``tpr == 0`` lanes publish the token's 128 core values (``o_s``, two token slots); every thread's
             ``fence.proxy.async.shared::cta``; warp 1: the gated-norm sum of token ``t - 1`` -- the reference's sequential
             ``ss += core^2`` chains over 4-row (NORM_FORM ctas2) or 8-row (ctas1) blocks, the chains summed in order, ctas2's two
             64-row halves summed last -- ``inv_{t-1} = rsqrt(ss / 128 + eps)`` -> ``inv_s``; the ``tpr < R`` lanes: the OUTPUTS of
             token ``t - 2`` (``bf16(((core * inv) * w) * silu(z))`` -> ``out``, ``e4m3(out / scale)`` -> ``qout``); each issuing lane
             (lane 0 of warp ``b mod 8`` for box ``b`` under spread, warp 0 lane 0 under single): ``cp.async.bulk.wait_group.read
             (NBUF - 2)`` (NBUF >= 2); ``bar.sync 0``; the issuing lanes: when ``t < ntok`` and ``indices[n, t] > 0`` the bulk store of
             their box to that slot, ``commit_group`` on every path.  NBUF 1: the one box is re-used every token behind a second
             ``bar.sync 0`` (after the issuing lanes' ``wait_group.read 0``) placed between pass 1 and pass 2 (pass 2 writes the box).
             ``STATE_STORE_PATH`` stg: no ring / fence / bulk store -- every lane's 16 x ``st.global.v4`` of its updated row slices
             straight to the slot right after pass 2 (the per-token barrier stays for ``o_s`` / ``inv_s``).
  epilogue   the outputs of token ``T - 2``; warp 1: the norm of token ``T - 1``; ``bar.sync 0``; the outputs of token ``T - 1``; the
             conv-state shift: the new slot is ``[old[a], old[a + 1], x_0 .. x_{T-1}]`` -- this CTA's 128 V channels always, the K head's
             256 Q / K channels by the CTA that drew the last ticket (16-B copies from smem by threads < 48 per column); the issuing
             lanes: ``cp.async.bulk.wait_group.read 0`` before exit.
  not live   zero ``out`` / ``qout`` for the request's ``ntok`` rows of this head; no state, conv-state or ticket write; nothing in flight.

Barriers executed per thread (T tokens): ``bar.sync 0`` x (T + 2) (+ T - 1 at NBUF 1), ``bar.sync 1`` x 2 (WG1) or ``bar.sync 2`` x 1
(WG0; x 2 under spread), one or two mbarrier waits, one relaxed atomic (WG1's lane); no ``membar``, no ``cp.async`` (LDGSTS); the
one-CTA form has no cluster op, the pair one relaxed cluster arrive + one acquire wait per thread and T ``mb_norm`` waits on its norm
warp.  ``ROLLED_TOKENS`` traces the token body once (``_token_step_rolled`` under ``cutlass.range(T, unroll=1)``) with the same barrier
sites per token.  Barrier table (init count = the lanes that arrive, on every path):

  mbarrier ``mb_ops``     init ``CONV_MBAR_ARRIVERS`` = 1 by warp 4 lane 0 (+ fence, published by ``bar.sync 1`` #1); one
                          ``arrive.expect_tx(conv_bulk_bytes(T))`` by the same lane inside the live arm; the ``3 + 9 + 3 T`` bulk copies
                          complete the bytes; waited by the 128 conv-role threads (phase 0) before the conv.
  mbarrier ``mb_state``   init ``STATE_MBAR_ARRIVERS`` = 1 by warp 0 lane 0 (+ fence; published to the 256 waiters by the hand-off
                          ``bar.sync 0``, and under spread to the issuing warps 1-3 by ``bar.sync 2`` #1 right after the init, BEFORE
                          the live arm: a copy completing bytes on an mbarrier whose init is not yet visible is undefined -- the
                          transaction is lost to the init and the wait never completes); one ``arrive.expect_tx(65536)`` by the same
                          lane inside the live arm; the RG box copies (issued by up to four lanes; a copy landing before the arm only
                          drives the tx-count negative, the phase cannot complete before the arrival) complete the bytes; waited by
                          every thread (phase 0) after the hand-off.
  ``barrier.sync 1, 128``  (non-aligned) the conv warpgroup, twice: after the mbarrier init (visibility), after the raw q / k / v
                          writes of every token (before the RMS reads).
  ``barrier.sync 2, 128``  (non-aligned) the BA warpgroup, once: the eight virtual-warp partials before the totals; under spread once
                          more, first: after the state mbarrier's init + fence and before the live arm, on every path (the init ->
                          fence -> barrier -> cross-warp-use order the issuing warps 1-3 need; the mirror of ``bar.sync 1`` #1).
  ``barrier.sync 0, 256`` (non-aligned) the hand-off; then one per token (every thread's STS + ``fence.proxy.async`` of the updated
                          rows before the one lane's bulk stores read the box; ``o_s`` / ``inv_s`` published for the next iteration's
                          norm / outputs); the epilogue's (token T - 1's ``inv``); NBUF 1: one more per token t >= 1 between the passes.
  ring WAR                 every issuing lane's ``cp.async.bulk.wait_group.read (NBUF - 2)`` right before each per-token barrier (NBUF
                          >= 2): with ONE commit per token per issuing lane on every path (an empty group when the slot is skipped)
                          "at most NBUF - 2 of my groups pending" = my store that read box ``(t + 1) % NBUF`` has finished reading; the
                          barrier makes that true for every box before any thread's STS of token ``t + 1``.  NBUF 1: ``wait_group.read
                          0`` + the mid-token barrier.  (``stg``: no ring, nothing to wait.)
  bulk stores              box ``b``'s ``cp.async.bulk.global.shared::cta.bulk_group`` by lane 0 of warp ``b mod 8`` (spread; warp 0 lane
                          0 for every box under single) + that thread's ``commit_group`` after each per-token barrier (stores skipped,
                          commit kept, for a skipped slot or a token past ``ntok``); each issuing lane's ``wait_group.read 0`` is its
                          last instruction (the boxes are read out before the CTA retires; the global writes complete with the grid).
  ticket                   one ``atom.relaxed.gpu.inc`` by warp 4 lane 0 after the conv mbarrier wait (every history byte of this CTA has
                          landed, so the last reader's shift never races a peer's read); consumed at exit.
  PDL (knob)               ``griddepcontrol.wait`` by every thread in both arms, after the predecessor-independent loads and issues (the
                          slot chain, parameters, BA weights, the slab, the conv weights and history) and before ``z``, ``hidden``, the
                          ``x`` copies and every global write; ``launch_dependents`` at entry under ``PDL_RELEASE``.  CALLER OBLIGATION:
                          the kernel immediately preceding this launch in the stream must not write this layer's ``state``, conv state
                          or slot indices -- the pre-wait loads read them before the wait.  The serving chain's predecessor (the QKVZ
                          projection) writes only ``x`` / ``z``, which are read after the wait; the host entry cannot check it.  The
                          pair form keeps both sites where they are (the cluster arrive precedes the wait, the exchange follows it).
  not live                 no arrive, nothing in flight at exit (both mbarriers initialised, never armed, never waited).
  mbarrier ``mb_norm[s]``  (pair) two stages, init ``NORM_MBAR_ARRIVERS`` = 1 each by warp 0 lane 0 with ``mb_state`` (one fence), published
                          cluster-wide by the relaxed cluster arrive at entry + the cluster wait in the tail; per token ``t`` ONE
                          ``arrive.expect_tx(NORM_XFER_BYTES)`` by warp 1 lane 0 on stage ``t % 2`` (top of iteration ``t + 1``; the
                          epilogue for ``T - 1``) and the PEER's warp 1 lanes ``< NORM_CHAINS_CTA`` each complete 4 tx bytes (st.async) =
                          ``NORM_XFER_BYTES``: SUM(software arrives) 1 == init 1; waited by warp 1 (parity ``(t // 2) & 1``) at the bottom
                          of iteration ``t + 1``.  Stage reuse (tokens ``t`` and ``t + 2``): the peer ships ``t + 2`` after its wait for
                          ``t + 1``, which needs this CTA's ship of ``t + 1``, which follows this CTA's wait + read of ``t`` -- the slot is
                          read before it is rewritten and the phase has completed before the next bytes land (the waiter's own wait is
                          in the chain).  Every ship has a reachable wait (T ships, T waits per CTA) before either CTA exits.
  cluster barrier          (pair) ``barrier.cluster.arrive.relaxed`` by every thread of both CTAs at entry (after the init lane's fence),
                          ``barrier.cluster.wait`` by every thread once (live: after the hand-off; null: at the start) -- 2 x 256 arrive,
                          2 x 256 wait; it publishes the mbarrier inits before any ``st.async`` can target them.

Numerics: the reference kernel's arithmetic order and PTX forms (the reference kernel: FlashInfer's built-in ``frost`` verify backend
for the Qwen 3.5 GDN block), so the fp32 state slots, the conv state, the bf16 ``out``
and the fp8 ``qout`` are bitwise with it under every knob: the conv taps and SiLU, the 256-virtual-thread BA partition and its
virtual-warp-ordered totals, the 16-lane RMS tree, the explicit ``mul.rn`` decay and ``fma.rn`` update, the row dots from +0 with the
16-leaf tree (the thread mapping moves leaves between threads, never the association), the bf16 core; the gated-norm sum of squares
is the reference's per-half-warp sequential chain (its ``fma.rn.f32.bf16`` of a bf16 core into an fp32 accumulator equals an exact
fp32 ``mul.rn`` of the bf16 value plus one rounded add: a bf16 x bf16 product is exact in fp32), the chains summed in half-warp order
and, at two CTAs per head, the two halves last (fp32 addition commutes exactly); the division by 128 is exact.

Shared memory (bytes; no swizzle: no descriptor reads these buffers, every per-lane pitch that could hit one bank is padded): the T x 3
input pieces 768 T, conv weights 3072, conv history 2304, raw q / k 1024 T, the q / k slices 2 x T x TPR x (C + 4) x 4 (the slices of a
phase land in distinct banks), v 512 T (index-permuted so the RG lanes of a phase read consecutive words), scalars 16 T, the two
core slots 1024, the inverses 16, BA partials 64 T, ticket 16, the two mbarriers 16 + 16, (the pair: the norm receive slots 2 x 16 x 4 --
written by the peer's st.async, read by warp 1 lane l at word ``l % NORM_CHAINS_CTA``: consecutive words, conflict-free; its own array,
aliasing nothing -- and the two ``mb_norm`` stages 16,) the ring NBUF x RG boxes x (BOX_ROWS x 512 + 16) B (box pitch padded by one 16-B
chunk -> bank = 4 rg + (128 / TPR) tpr + 4 c4 mod 32, eight distinct quads per 8-lane phase at every TPR, on the one-CTA form and on the
pair's half-size boxes), plus the occupancy pad of ``Cfg.CTAS_PER_SM`` derived from the device's shared memory per SM.
"""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
from cutlass._mlir.dialects import arith

from cudnn.frost.tile_dsl.barrier import launch_dependent_grids, wait, wait_on_dependent_grids
from cudnn.frost.tile_dsl.tma import ld_global_v4, ld_shared_v4, st_global_v4, st_shared_v4

from .gdn_verify_fused_config import (
    BA_CHUNK,
    BA_CHUNKS,
    CONV_HIST_TAPS,
    CONV_MBAR_ARRIVERS,
    CONV_WIDTH,
    HEAD_DIM,
    HIDDEN,
    K_CHANNEL0,
    NORM_MBAR_ARRIVERS,
    NORM_MBAR_STAGES,
    NORM_RECV_MAX_FLOATS,
    OUT_ROW,
    QKV_CHANNELS,
    RMS_COLS,
    RMS_LANES,
    STATE_MBAR_ARRIVERS,
    THREADS,
    TICKET_STRIDE,
    V_CHANNEL0,
    V_HEADS,
    WARPGROUP,
    WARPS,
    conv_bulk_bytes,
)

FP32 = cutlass.Float32
BF16 = cutlass.BFloat16
INT32 = cutlass.Int32
INT64 = cutlass.Int64
UINT8 = cutlass.Uint8

# the reference's q scale constant 128 ** -0.5 (its fp32 literal), the RMS epsilon and the softplus threshold
Q_SCALE = 0.08838834764831845
RMS_EPS = 1e-6
SOFTPLUS_LINEAR_ABOVE = 20.0
ROW_BYTES = HEAD_DIM * 4  # one state row in the shared-memory boxes (the natural 512-B pitch inside a box)
NORM_WARP = 1  # the warp that sums the gated norm's squares (warp 0's lane 0 owns the state mbarrier and the bulk stores)
SHIFT_THREADS = 3 * HEAD_DIM // 8  # 48: one 16-B (8-channel) group of the three pieces per thread and conv-state column


# --------------------------------------------------------------------------------------------------------------------------
# PTX forms shared with the reference kernel (inline PTX through the DSL primitive; outputs number first, inputs after)
# --------------------------------------------------------------------------------------------------------------------------
def _unpack_bf16x2(word):
    """A bf16x2 word -> (lo, hi) as fp32: ``shl 16`` / ``and 0xffff0000`` (the reference helper's exact ``* 1.0`` dropped)."""
    return nvvm.inline_ptx(
        "{\n\t.reg .b32 t;\n\tshl.b32 t, $2, 16;\n\tmov.b32 $0, t;\n\tand.b32 t, $2, 0xffff0000;\n\tmov.b32 $1, t;\n\t}",
        write_only_types=[FP32, FP32],
        read_only_args=[word],
    )


def _stock_conv_silu(value):
    """The stock causal-conv SiLU: ``x / (1 + ex2(-x * log2 e))`` with ``ex2.approx`` and ``div.full``."""
    return nvvm.inline_ptx(
        "{\n\t.reg .f32 t, e, d;\n\tmul.f32 t, $1, 0fBFB8AA3B;\n\tex2.approx.f32 e, t;\n\tadd.f32 d, e, 0f3F800000;\n\tdiv.full.f32 $0, $1, d;\n\t}",
        write_only_types=[FP32],
        read_only_args=[value],
    )


def _quant_fp8_word(value, scale):
    """The stock per-tensor FP8 quantization: ``v * (1 / scale)`` with a ``div.rn`` reciprocal, ``cvt.rn.satfinite.e4m3x2``; the
    e4m3 byte is the low byte of the returned word."""
    return nvvm.inline_ptx(
        "{\n\t.reg .f32 inv, v;\n\t.reg .b16 p;\n\tdiv.rn.f32 inv, 0f3F800000, $2;\n\tmul.f32 v, $1, inv;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 p, 0f00000000, v;\n\tcvt.u32.u16 $0, p;\n\t}",
        write_only_types=[INT32],
        read_only_args=[value, scale],
    )


def _mul_rn_f32(a, b):
    """An fp32 product the compiler cannot fuse into a following add (the reference's explicit ``mul.rn`` of the decay step and the
    unfused squares of the norm)."""
    return nvvm.inline_ptx("mul.rn.f32 $0, $1, $2;", write_only_types=[FP32], read_only_args=[a, b])


def _fma_rn_f32(a, b, c):
    """The reference's explicit ``fma.rn.f32`` of the state update (``h = fma(inn, k, h)``: one rounding, never re-contracted)."""
    return nvvm.inline_ptx("fma.rn.f32 $0, $1, $2, $3;", write_only_types=[FP32], read_only_args=[a, b, c])


def _atom_inc_relaxed_gpu(addr, limit):
    """``atom.relaxed.gpu.global.inc.u32``: the reader ticket; returns the old value, wraps to 0 past ``limit``."""
    return nvvm.inline_ptx("atom.relaxed.gpu.global.inc.u32 $0, [$1], $2;", write_only_types=[INT32], read_only_args=[addr, limit])


def _smem_addr(arr):
    """32-bit shared-window address of a shared-memory ``cutlass.Array``."""
    return INT32(arr.data_ptr().toint())


def _mapa_u32(addr32, rank):
    """``mapa.shared::cluster``: this CTA's shared-window address translated to the same offset in cluster CTA ``rank``'s window."""
    return nvvm.inline_ptx("mapa.shared::cluster.u32 $0, $1, $2;", write_only_types=[INT32], read_only_args=[addr32, rank])


def _st_async_f32(peer_addr32, value, peer_mbar32):
    """One fp32 into a peer CTA's shared memory, completing 4 tx bytes on the peer's mbarrier (``st.async``; the DSMEM path, not the
    bulk-copy unit -- the write-back boxes queued there would delay it).  Both addresses are ``mapa``-translated."""
    nvvm.inline_ptx("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.f32 [$0], $1, [$2];", read_only_args=[peer_addr32, value, peer_mbar32])


def _cta_barrier():
    """The CTA-wide barrier (id 0, every thread) in its NON-aligned form: the role arms are two traced bodies (two PCs), which the
    ``.aligned`` promise does not cover -- synccheck flags the aligned form as divergent; the hardware barrier counts arrivals per id."""
    nvvm.barrier_cta_sync(0, thread_count=THREADS)


def _wg_barrier(barrier_id):
    """A warpgroup-local named barrier (128 threads): the NON-aligned ``barrier.sync id, 128``, reached from a warp-role branch."""
    nvvm.barrier_cta_sync(barrier_id, thread_count=WARPGROUP)


def _select_f32(cond, a, b):
    """``cond ? a : b`` on fp32 values as one select (no branch)."""
    return FP32(arith.select(cond.ir_value(), a.ir_value(), b.ir_value()))


def _balanced_sum(parts):
    """The balanced binary tree over ``parts`` in order (``((p0+p1)+(p2+p3))+...``); one part returns itself."""
    while len(parts) > 1:
        parts = [parts[i] + parts[i + 1] for i in range(0, len(parts), 2)]
    return parts[0]


def _gmem_ptr(tensor, elem_offset, dtype):
    """A 16-byte-aligned GLOBAL pointer to ``tensor[elem_offset]`` for the bulk copies (``cp.async.bulk`` wants an explicit gmem
    address space)."""
    return cute.make_ptr(dtype, (tensor.iterator + elem_offset).toint(), cute.AddressSpace.gmem, assumed_align=16)


def _gaddr(tensor, elem_offset):
    """The 64-bit generic address of ``tensor[elem_offset]`` (for the 128-bit vector loads / stores)."""
    return (tensor.iterator + elem_offset).toint()


# --------------------------------------------------------------------------------------------------------------------------
# straight-line helpers (plain Python: traced inline, no DSL control flow inside)
# --------------------------------------------------------------------------------------------------------------------------
def _conv_piece_channel0(piece, head, kh):
    """The first QKV channel of piece ``piece`` (0 = this K head's Q, 1 = its K, 2 = this V head's V)."""
    if piece == 0:
        return kh * HEAD_DIM
    if piece == 1:
        return K_CHANNEL0 + kh * HEAD_DIM
    return V_CHANNEL0 + head * HEAD_DIM


def _box_offset(cfg, slot, head, r, SS, b):
    """The element offset of box ``b`` of the rows CTA ``r`` of head ``head`` owns in slot ``slot``: the one-CTA form's committed
    arithmetic (``slot SS + head 16384 + b BOX_ROWS 128``), plus ``CTA_ROWS r x 128`` on the pair only (so the one-CTA cubin is untouched)."""
    off = slot * SS + head * HEAD_DIM * HEAD_DIM + b * cfg.BOX_ROWS * HEAD_DIM
    if cfg.TWO_CTAS_PER_HEAD:
        off = off + r * (cfg.CTA_ROWS * HEAD_DIM)
    return off


def _issue_state_box(cfg, state, ring, mb_state, slot, head, r, SS, b):
    """The slab's box ``b`` (this CTA's rows ``BOX_ROWS b ..``, one contiguous run in global memory) into ring buffer 0 at box pitch
    ``STATE_BOX_PITCH_ELEMS``, completing on the state mbarrier (armed once, by warp 0 lane 0, with the whole slab's bytes)."""
    nvvm.cp_async_bulk_shared_cluster_global(
        ring.subview(b * cfg.STATE_BOX_PITCH_ELEMS),
        _gmem_ptr(state, _box_offset(cfg, slot, head, r, SS, b), FP32),
        mb_state,
        cfg.STATE_BOX_BYTES,
    )


def _issue_state_store_box(cfg, state, ring, buf, slot, head, r, SS, b):
    """Box ``b`` of ring buffer ``buf`` to slot ``slot`` (``cp.async.bulk.global.shared::cta.bulk_group``); the issuing lane commits."""
    nvvm.cp_async_bulk_global_shared_cta(
        _gmem_ptr(state, _box_offset(cfg, slot, head, r, SS, b), FP32),
        ring.subview(buf * cfg.SLAB_SMEM_ELEMS + b * cfg.STATE_BOX_PITCH_ELEMS),
        cfg.STATE_BOX_BYTES,
    )


def _store_row_gmem(state, slot, head, rowm, col0, vals, SS):
    """One updated row slice straight to its slot by 16-B vector stores (the ``stg`` store path)."""
    base = (state.iterator + (slot * SS + (head * HEAD_DIM + rowm) * HEAD_DIM + col0)).toint()
    for c4 in range(len(vals) // 4):
        st_global_v4(base + 16 * c4, vals[4 * c4 : 4 * c4 + 4], FP32)


def _issue_conv_param_copies(cw, cs, cw_s, hist_s, mb_arr, head, kh, cv_slot, hist0, CSS, CST):
    """The ``cp.async.bulk`` copies of this head's conv WEIGHTS (3 x 1 KB) and the HISTORY taps ``a - 1 .. a + 1`` of its three pieces
    (9 x 256 B; ``hist0`` = column ``a - 1`` of the channel-contiguous pool): written by earlier steps, never by the predecessor
    grid, so they are issued ahead of a PDL wait."""
    for piece in range(3):
        ch0 = _conv_piece_channel0(piece, head, kh)
        nvvm.cp_async_bulk_shared_cluster_global(
            cw_s.subview(piece * HEAD_DIM * CONV_WIDTH), _gmem_ptr(cw, ch0 * CONV_WIDTH, BF16), mb_arr, HEAD_DIM * CONV_WIDTH * 2
        )
        for tap in range(CONV_HIST_TAPS):
            nvvm.cp_async_bulk_shared_cluster_global(
                hist_s.subview((piece * CONV_HIST_TAPS + tap) * HEAD_DIM),
                _gmem_ptr(cs, cv_slot * CSS + (hist0 + tap) * CST + ch0, BF16),
                mb_arr,
                HEAD_DIM * 2,
            )


def _issue_conv_x_copies(qkv, x_s, mb_arr, row, head, kh, XS, t):
    """The three 256-B copies of this head's Q / K / V pieces of token ``t``'s input row (``row`` = its 64-bit row index, clamped to
    the request's last token): the predecessor's output, issued after a PDL wait."""
    for piece in range(3):
        ch0 = _conv_piece_channel0(piece, head, kh)
        nvvm.cp_async_bulk_shared_cluster_global(x_s.subview((t * 3 + piece) * HEAD_DIM), _gmem_ptr(qkv, row * XS + ch0, BF16), mb_arr, HEAD_DIM * 2)


def _conv_value(x_s, hist_s, taps, t, piece, u):
    """The reference conv unit for channel ``u`` of ``piece`` at token ``t``: the four window entries ``seq[t + e]`` of ``seq =
    [old[a - 1], old[a], old[a + 1], x_0 .. x_{T-1}]`` times taps ``0 .. 3`` (tap 3 on the newest), each product rounded to bf16, summed
    in tap order from +0, the stock SiLU, bf16.  Returns the bf16 value as fp32."""
    acc = FP32(0)
    for e in range(CONV_WIDTH):
        s = t + e
        if s < CONV_HIST_TAPS:
            xv = hist_s[(piece * CONV_HIST_TAPS + s) * HEAD_DIM + u].to(FP32)
        else:
            xv = x_s[((s - CONV_HIST_TAPS) * 3 + piece) * HEAD_DIM + u].to(FP32)
        prod = (xv * taps[e]).to(BF16).to(FP32)
        acc = acc + prod
    return _stock_conv_silu(acc).to(BF16).to(FP32)


def _slice_values(base_addr, n):
    """``n`` fp32 values (a multiple of 4) from shared memory by 16-B vector loads."""
    vals = []
    for c4 in range(n // 4):
        vals.extend(ld_shared_v4(base_addr + 16 * c4, FP32))
    return vals


def _store_slice(base_addr, vals):
    """``vals`` (a multiple of 4 fp32) to shared memory by 16-B vector stores."""
    for c4 in range(len(vals) // 4):
        st_shared_v4(base_addr + 16 * c4, vals[4 * c4 : 4 * c4 + 4], FP32)


def _load_row_smem(sbase, m, C):
    """One row slice (``C`` fp32) from the thread's shared-memory box (``sbase`` = its first row's slice address; 512-B row pitch)."""
    return _slice_values(sbase + m * ROW_BYTES, C)


def _store_row_smem(sbase, m, vals):
    _store_slice(sbase + m * ROW_BYTES, vals)


def _k_group(k, kbase, g, cfg):
    """The 8 k values of 8-column group ``g``: from the resident slice, or re-read from shared memory (``K_RESIDENT`` off)."""
    return k[8 * g : 8 * g + 8] if cfg.K_RESIDENT else _slice_values(kbase + 32 * g, 8)


def _row_reduce(cfg, parts):
    """The reference 16-leaf tree over a row's 8-column partials: the balanced tree over the thread's groups, then ``log2(TPR)``
    xor shuffles with offsets ``RG << step`` (the partners differ in the tpr lane bits)."""
    s = _balanced_sum(parts)
    for step in range(cfg.TPR.bit_length() - 1):
        s = s + cute.arch.shuffle_sync_bfly(s, offset=cfg.RG << step)
    return s


def _pass1(cfg, S_rows, src_base, k, k_base, decay):
    """Pass 1 of a token: ``h = mul.rn(S, decay)`` per row (kept for the register-resident form, dropped when streamed) and ``dot =
    h . k`` per row: per 8-column group an 8-step FFMA chain from +0 (the group's k values loaded once and shared by the thread's
    rows), the groups combined by the reference 16-leaf tree.  Returns (the decayed rows or None, the dots)."""
    R, C, G = cfg.R, cfg.C, cfg.GROUPS
    streamed = S_rows is None
    h = []
    for m in range(R):
        S = _load_row_smem(src_base, m, C) if streamed else S_rows[m]
        h.append([_mul_rn_f32(S[j], decay) for j in range(C)])
    parts = [[] for _ in range(R)]
    for g in range(G):
        kg = _k_group(k, k_base, g, cfg)
        for m in range(R):
            p = FP32(0)
            for j in range(8):
                p = p + h[m][8 * g + j] * kg[j]
            parts[m].append(p)
    dots = [_row_reduce(cfg, parts[m]) for m in range(R)]
    return (None if streamed else h), dots


def _pass2(cfg, h_rows, src_base, dst_base, k, k_base, q_base, inn, decay, store_smem):
    """Pass 2 of a token: ``h = fma.rn(inn, k, h)`` (the streamed form re-reads the row and recomputes ``mul.rn(S, decay)`` first:
    bit-identical), ``o = h . q`` (the same chains and tree; each q / k group loaded once per token), the updated slice to ring box
    ``dst_base``.  Returns (the new rows or None, the per-row ``f32(bf16(o))`` core values)."""
    R, C, G = cfg.R, cfg.C, cfg.GROUPS
    streamed = h_rows is None
    if streamed:
        h = []
        for m in range(R):
            S = _load_row_smem(src_base, m, C)
            h.append([_mul_rn_f32(S[j], decay) for j in range(C)])
    else:
        h = h_rows
    parts = [[] for _ in range(R)]
    for g in range(G):
        kg = _k_group(k, k_base, g, cfg)
        qg = _slice_values(q_base + 32 * g, 8)
        for m in range(R):
            o = FP32(0)
            for j in range(8):
                idx = 8 * g + j
                h[m][idx] = _fma_rn_f32(inn[m], kg[j], h[m][idx])
                o = o + h[m][idx] * qg[j]
            parts[m].append(o)
    values = []
    for m in range(R):
        values.append(_row_reduce(cfg, parts[m]).to(BF16).to(FP32))
        if store_smem:
            _store_row_smem(dst_base, m, h[m])
    return (None if streamed else h), values


def _pick_row_value(values, tpr):
    """``values[tpr]`` for ``tpr < R`` by a select chain (every lane holds every row's value after the butterflies)."""
    val = values[0]
    for m in range(1, len(values)):
        val = _select_f32(tpr == m, values[m], val)
    return val


def _norm_inverse(cfg, o_s, slot, lane, EPS):
    """The gated-norm inverse of one token from its 128 core values (``o_s`` slot ``slot``, a Python int or a traced value), by the 32 lanes of one warp: lane ``c``
    sums the squares of chain ``c``'s rows sequentially (the reference half-warp's ``ss += core * core``, an fp32 chain over 4 rows
    at NORM_FORM ctas2 or 8 rows at ctas1), the chains are summed in chain order (ctas2: the two 64-row halves summed last; fp32
    addition commutes exactly, so both of the reference's ranks hold the same bits) and ``inv = rsqrt(ss / 128 + eps)``."""
    CR = cfg.NORM_CHAIN_ROWS
    chains = cfg.NORM_CHAINS
    chain = lane if chains == 32 else lane % 16
    base = slot * HEAD_DIM + chain * CR
    v0 = o_s[base]
    ss = _mul_rn_f32(v0, v0)
    for i in range(1, CR):
        v = o_s[base + i]
        ss = ss + _mul_rn_f32(v, v)
    if chains == 32:
        tot0 = cute.arch.shuffle_sync(ss, 0)
        for c in range(1, 16):
            tot0 = tot0 + cute.arch.shuffle_sync(ss, c)
        tot1 = cute.arch.shuffle_sync(ss, 16)
        for c in range(17, 32):
            tot1 = tot1 + cute.arch.shuffle_sync(ss, c)
        total = tot0 + tot1
    else:
        total = cute.arch.shuffle_sync(ss, 0)
        for c in range(1, 16):
            total = total + cute.arch.shuffle_sync(ss, c)
    return cute.math.rsqrt(total / FP32(HEAD_DIM) + FP32(EPS))


def _norm_chain_sum(cfg, o_s, slot, lane):
    """TWO_CTAS_PER_HEAD: lane ``l`` of the norm warp sums the squares of LOCAL chain ``l % NORM_CHAINS_CTA`` (``NORM_CHAIN_ROWS``
    consecutive rows of this CTA's 64, the reference half-warp's sequential ``ss += core * core``) from ``o_s`` slot ``slot`` (a Python
    int on the unrolled loop, a traced value on the rolled one: plain address arithmetic either way)."""
    CR = cfg.NORM_CHAIN_ROWS
    base = slot * HEAD_DIM + (lane % cfg.NORM_CHAINS_CTA) * CR
    v0 = o_s[base]
    ss = _mul_rn_f32(v0, v0)
    for i in range(1, CR):
        v = o_s[base + i]
        ss = ss + _mul_rn_f32(v, v)
    return ss


def _norm_inverse_pair(cfg, ss, peer, r, lane, EPS):
    """TWO_CTAS_PER_HEAD: the gated-norm inverse of one token from this CTA's chain sums (``ss``: lane ``l`` holds local chain ``l %
    NCH``) and the peer's (``peer``: lane ``l`` holds the peer's chain ``l % NCH``, read from the receive slot).  Lane ``l < NORM_CHAINS``
    holds HEAD chain ``l``: its own when ``l // NCH == r``, the peer's otherwise (one select on the cluster rank); then the reference
    association runs over the head chains in order exactly as the one-CTA form's ``_norm_inverse`` (ctas2: the two 64-row halves
    summed sequentially, then added -- fp32 addition commutes exactly, so both ranks hold the same bits; ctas1: one sequence)."""
    NCH = cfg.NORM_CHAINS_CTA
    chains = cfg.NORM_CHAINS
    val = _select_f32((lane // NCH) == r, ss, peer)
    if chains == 32:
        tot0 = cute.arch.shuffle_sync(val, 0)
        for c in range(1, 16):
            tot0 = tot0 + cute.arch.shuffle_sync(val, c)
        tot1 = cute.arch.shuffle_sync(val, 16)
        for c in range(17, 32):
            tot1 = tot1 + cute.arch.shuffle_sync(val, c)
        total = tot0 + tot1
    else:
        total = cute.arch.shuffle_sync(val, 0)
        for c in range(1, 16):
            total = total + cute.arch.shuffle_sync(val, c)
    return cute.math.rsqrt(total / FP32(HEAD_DIM) + FP32(EPS))


def _pick_dyn(values, idx):
    """``values[idx]`` for a DYNAMIC ``idx`` by a select chain over the static list."""
    val = values[0]
    for i in range(1, len(values)):
        val = _select_f32(idx == i, values[i], val)
    return val


def _emit_output(cfg, values, tpr, inv, wv, zv, scv, oi, out, qout, raw):
    """The output of one (token, row) pair: ``bf16(((core * inv) * w) * (z / (1 + exp(-z))))`` -> ``out``, its static FP8 -> ``qout``."""
    val = _pick_row_value(values, tpr)
    rounded = (((val * inv) * wv) * (zv / (FP32(1) + cute.math.exp(-zv)))).to(BF16)
    qout[oi] = _quant_fp8_word(rounded.to(FP32), scv).to(UINT8)
    out[oi] = rounded
    if cutlass.const_expr(cfg.DEBUG):
        raw[oi] = val.to(BF16)


# --------------------------------------------------------------------------------------------------------------------------
# the BA role (warpgroup 0): the state mbarrier + slab issue, the reference 256-virtual-thread BA partition for T tokens
# --------------------------------------------------------------------------------------------------------------------------
@cute.jit
def _ba_role(
    cfg: cutlass.Constexpr,
    tid,
    head,
    r,
    n,
    bos,
    ntok,
    live,
    src,
    al,
    dt,
    hidden,
    baweight,
    state,
    scal_s,
    ba_part,
    mb_state,
    mb_norm,
    ring,
    baout,
    SS: cutlass.Constexpr,
    T: cutlass.Constexpr,
):
    u = tid  # 0 .. 127: virtual threads u (chunks 0 .. 2) and u + 128 (chunks 0, 1) of the reference partition
    wl = tid // 32
    lane = tid % 32
    P = cfg.BA_PREFETCH_CHUNKS
    if wl == 0:
        if lane == 0:
            nvvm.mbarrier_init(mb_state, STATE_MBAR_ARRIVERS)
            if cutlass.const_expr(cfg.TWO_CTAS_PER_HEAD):
                # the two stages of the norm exchange (token parity): the peer's st.async completes bytes on them, so their init is
                # published cluster-wide (fence.mbarrier_init.release.cluster + the relaxed cluster arrive below + the cluster wait in
                # the tail) before any ship
                for s in cutlass.range_constexpr(NORM_MBAR_STAGES):
                    nvvm.mbarrier_init(mb_norm.subview(s), NORM_MBAR_ARRIVERS)
            nvvm.fence_mbarrier_init()  # published to the 256 waiters by the hand-off barrier, which every waiter passes first
    if cutlass.const_expr(cfg.SPREAD_ISSUE):
        # the issuing warps 1-3 complete bytes on mb_state from the live arm below: the init + fence must reach them through a barrier
        # FIRST (a copy completing on an mbarrier whose init is not yet visible is undefined -- the transaction is lost to the init and
        # the wait never completes; the form hung at 16 requests without this) -- the mirror of the conv warpgroup's bar.sync 1 after
        # its init.  Under single the init lane is the only issuing lane and the hand-off barrier publishes the init to the waiters.
        _wg_barrier(2)
    if cutlass.const_expr(cfg.TWO_CTAS_PER_HEAD):
        # P4 step 3: every thread of both CTAs arrives (relaxed; the init lane's fence.mbarrier_init orders its inits ahead) -- the
        # matching cluster wait sits in the tail, before the first st.async may land on this CTA's mb_norm
        nvvm.barrier_cluster_arrive_relaxed()
    alv = al[head].to(FP32)
    dtv = dt[head].to(FP32)
    # the two BA weight rows: parameters, never a predecessor's output -> ahead of a PDL wait
    bw0, aw0, bw1, aw1 = [], [], [], []
    for wc in cutlass.range_constexpr(BA_CHUNKS):
        kk = wc * BA_CHUNK + u * 8
        bw0.append(ld_global_v4(_gaddr(baweight, head * HIDDEN + kk), INT32))
        aw0.append(ld_global_v4(_gaddr(baweight, (head + V_HEADS) * HIDDEN + kk), INT32))
        if cutlass.const_expr(wc * BA_CHUNK + WARPGROUP * 8 < HIDDEN):
            bw1.append(ld_global_v4(_gaddr(baweight, head * HIDDEN + kk + WARPGROUP * 8), INT32))
            aw1.append(ld_global_v4(_gaddr(baweight, (head + V_HEADS) * HIDDEN + kk + WARPGROUP * 8), INT32))
    if live:
        if lane == 0:
            # warp 0 arms the slab's bytes once; box b is issued by warp b mod 4 (spread: the init reached these lanes through the
            # warpgroup barrier above) or by warp 0 (single) -- a copy that lands before the arm only drives the tx-count negative
            # until the arm, the phase cannot complete before the arrival
            if wl == 0:
                nvvm.mbarrier_arrive_expect_tx(mb_state, cfg.STATE_SLAB_BYTES_CTA)
            for b in cutlass.range_constexpr(cfg.RG):
                if cutlass.const_expr(cfg.SPREAD_ISSUE):
                    if wl == b % (WARPGROUP // 32):
                        _issue_state_box(cfg, state, ring, mb_state, src, head, r, SS, b)
                else:
                    if wl == 0:
                        _issue_state_box(cfg, state, ring, mb_state, src, head, r, SS, b)
        if cutlass.const_expr(cfg.PDL):
            # everything above was written before this step's graph; the normed hidden the BA reads is the step's own
            wait_on_dependent_grids()
        rows = []
        for t in cutlass.range_constexpr(T):
            rows.append((bos + cutlass.min(INT32(t), ntok - 1)).to(INT64) * HIDDEN)
        # the hidden pieces, one FLAT list per virtual thread indexed c * T + t: a nested subscript store (hx[c][t] = ...) would mark
        # every name of its inner index as written in this dynamic region and the DSL would hand P back as a traced value
        hx0 = [None] * (BA_CHUNKS * T)
        hx1 = [None] * (BA_CHUNKS * T)
        bd0, ad0, bd1, ad1 = [], [], [], []
        for _ in cutlass.range_constexpr(T):
            bd0.append(FP32(0))
            ad0.append(FP32(0))
            bd1.append(FP32(0))
            ad1.append(FP32(0))
        # the hidden rows: chunks 0 .. P - 1 in flight at once, the rest issued as a chunk's chains complete
        for c in cutlass.range_constexpr(P):
            for t in cutlass.range_constexpr(T):
                hx0[c * T + t] = ld_global_v4(_gaddr(hidden, rows[t] + c * BA_CHUNK + u * 8), INT32)
                if cutlass.const_expr(c < BA_CHUNKS - 1):
                    hx1[c * T + t] = ld_global_v4(_gaddr(hidden, rows[t] + c * BA_CHUNK + WARPGROUP * 8 + u * 8), INT32)
        for c in cutlass.range_constexpr(BA_CHUNKS):
            b0, a0, b1, a1 = [], [], [], []
            for j in cutlass.range_constexpr(4):
                b0.extend(_unpack_bf16x2(bw0[c][j]))
                a0.extend(_unpack_bf16x2(aw0[c][j]))
                if cutlass.const_expr(c < BA_CHUNKS - 1):
                    b1.extend(_unpack_bf16x2(bw1[c][j]))
                    a1.extend(_unpack_bf16x2(aw1[c][j]))
            for t in cutlass.range_constexpr(T):
                for j in cutlass.range_constexpr(4):
                    hlo, hhi = _unpack_bf16x2(hx0[c * T + t][j])
                    bd0[t] = bd0[t] + hlo * b0[2 * j]
                    bd0[t] = bd0[t] + hhi * b0[2 * j + 1]
                    ad0[t] = ad0[t] + hlo * a0[2 * j]
                    ad0[t] = ad0[t] + hhi * a0[2 * j + 1]
                    if cutlass.const_expr(c < BA_CHUNKS - 1):
                        glo, ghi = _unpack_bf16x2(hx1[c * T + t][j])
                        bd1[t] = bd1[t] + glo * b1[2 * j]
                        bd1[t] = bd1[t] + ghi * b1[2 * j + 1]
                        ad1[t] = ad1[t] + glo * a1[2 * j]
                        ad1[t] = ad1[t] + ghi * a1[2 * j + 1]
            if cutlass.const_expr(c + P < BA_CHUNKS):
                for t in cutlass.range_constexpr(T):
                    hx0[(c + P) * T + t] = ld_global_v4(_gaddr(hidden, rows[t] + (c + P) * BA_CHUNK + u * 8), INT32)
                    if cutlass.const_expr(c + P < BA_CHUNKS - 1):
                        hx1[(c + P) * T + t] = ld_global_v4(_gaddr(hidden, rows[t] + (c + P) * BA_CHUNK + WARPGROUP * 8 + u * 8), INT32)
        # one 5-step butterfly per (virtual warp, token): virtual warps wl (thread u) and wl + 4 (thread u + 128) pair the same lanes
        for t in cutlass.range_constexpr(T):
            for step in cutlass.range_constexpr(5):
                bd0[t] = bd0[t] + cute.arch.shuffle_sync_bfly(bd0[t], offset=1 << step)
                ad0[t] = ad0[t] + cute.arch.shuffle_sync_bfly(ad0[t], offset=1 << step)
                bd1[t] = bd1[t] + cute.arch.shuffle_sync_bfly(bd1[t], offset=1 << step)
                ad1[t] = ad1[t] + cute.arch.shuffle_sync_bfly(ad1[t], offset=1 << step)
            if lane == 0:
                ba_part[(wl * T + t) * 2] = bd0[t]
                ba_part[(wl * T + t) * 2 + 1] = ad0[t]
                ba_part[((wl + 4) * T + t) * 2] = bd1[t]
                ba_part[((wl + 4) * T + t) * 2 + 1] = ad1[t]
        _wg_barrier(2)
        if u < 2 * T:
            # thread 2 t: token t's b -> beta; thread 2 t + 1: its a -> decay (the reference's thread roles and summation order)
            total = FP32(0)
            for wi in cutlass.range_constexpr(WARPS):
                total = total + ba_part[wi * 2 * T + u]
            gv = total.to(BF16).to(FP32)
            tok = u // 2
            if u % 2 == 0:
                scal_s[4 * tok + 1] = FP32(1) / (FP32(1) + cute.math.exp(-gv))
                if cutlass.const_expr(cfg.DEBUG):
                    baout[(bos + cutlass.min(tok, ntok - 1)).to(INT64) * (2 * V_HEADS) + head] = gv.to(BF16)
            else:
                gx = gv + dtv
                soft = gx
                if gx <= FP32(SOFTPLUS_LINEAR_ABOVE):
                    soft = cute.math.log(FP32(1) + cute.math.exp(gx))
                scal_s[4 * tok] = cute.math.exp(-cute.math.exp(alv) * soft)
                if cutlass.const_expr(cfg.DEBUG):
                    baout[(bos + cutlass.min(tok, ntok - 1)).to(INT64) * (2 * V_HEADS) + head + V_HEADS] = gv.to(BF16)


# --------------------------------------------------------------------------------------------------------------------------
# the conv role (warpgroup 1): the conv-operand mbarrier + copies, the ticket, T tokens of conv + SiLU, the RMS scales, the staging
# --------------------------------------------------------------------------------------------------------------------------
@cute.jit
def _conv_role(
    cfg: cutlass.Constexpr,
    tid,
    head,
    kh,
    n,
    bos,
    ntok,
    live,
    cv_slot,
    acc,
    qkv,
    cs,
    cw,
    counts,
    x_s,
    cw_s,
    hist_s,
    qk_raw,
    q_s,
    k_s,
    v_s,
    scal_s,
    ticket_s,
    mb_arr,
    convout,
    XS: cutlass.Constexpr,
    CSS: cutlass.Constexpr,
    CST: cutlass.Constexpr,
    T: cutlass.Constexpr,
):
    C = cfg.C
    PITCH = cfg.SLICE_PITCH
    RMS_ROUNDS = (T + WARPGROUP // RMS_LANES - 1) // (WARPGROUP // RMS_LANES)
    u = tid - WARPGROUP  # 0 .. 127: the channel this thread convolves in each piece
    wl = u // 32
    lane = tid % 32
    hw = u // RMS_LANES  # the half-warp (0 .. 7): one token per RMS round
    l16 = u % RMS_LANES  # the RMS lane: columns 8 l16 .. 8 l16 + 7
    if wl == 0:
        if lane == 0:
            nvvm.mbarrier_init(mb_arr, CONV_MBAR_ARRIVERS)
            nvvm.fence_mbarrier_init()
    _wg_barrier(1)
    if cutlass.const_expr(cfg.TWO_CTAS_PER_HEAD):
        nvvm.barrier_cluster_arrive_relaxed()  # P4 step 3 for this warpgroup's 128 threads (the wait sits in the tail)
    if live:
        rows = []
        for t in cutlass.range_constexpr(T):
            rows.append((bos + cutlass.min(INT32(t), ntok - 1)).to(INT64))
        if lane == 0:
            # warp 4 arms the operand bytes once and issues the weights + history; token t's input rows by warp 5 + t mod 3 (spread)
            if wl == 0:
                nvvm.mbarrier_arrive_expect_tx(mb_arr, conv_bulk_bytes(T))
                _issue_conv_param_copies(cw, cs, cw_s, hist_s, mb_arr, head, kh, cv_slot, (acc - 1).to(INT64), CSS, CST)
            if cutlass.const_expr(not cfg.PDL):
                for t in cutlass.range_constexpr(T):
                    if cutlass.const_expr(cfg.SPREAD_ISSUE):
                        if wl == 1 + t % 3:
                            _issue_conv_x_copies(qkv, x_s, mb_arr, rows[t], head, kh, XS, t)
                    else:
                        if wl == 0:
                            _issue_conv_x_copies(qkv, x_s, mb_arr, rows[t], head, kh, XS, t)
        if cutlass.const_expr(cfg.PDL):
            # the QKVZ projection's rows are read only past this point
            wait_on_dependent_grids()
            if lane == 0:
                for t in cutlass.range_constexpr(T):
                    if cutlass.const_expr(cfg.SPREAD_ISSUE):
                        if wl == 1 + t % 3:
                            _issue_conv_x_copies(qkv, x_s, mb_arr, rows[t], head, kh, XS, t)
                    else:
                        if wl == 0:
                            _issue_conv_x_copies(qkv, x_s, mb_arr, rows[t], head, kh, XS, t)
        wait(mb_arr, 0)
        if wl == 0:
            if lane == 0:
                # every history byte of this CTA has landed: the ticket orders this CTA's reads before the last reader's shift
                ticket_s[0] = _atom_inc_relaxed_gpu(_gaddr(counts, n * TICKET_STRIDE + kh * 32), INT32(cfg.TICKET_LIMIT))
        taps = []
        for piece in cutlass.range_constexpr(3):
            tp = []
            for e in cutlass.range_constexpr(CONV_WIDTH):
                tp.append(cw_s[piece * HEAD_DIM * CONV_WIDTH + u * CONV_WIDTH + e].to(FP32))
            taps.append(tp)
        # the permuted index of head row u (the RG lanes of a phase read consecutive words; a two-CTA pair reads disjoint halves)
        vperm = (u % cfg.BOX_ROWS) * cfg.V_PERM_STRIDE + u // cfg.BOX_ROWS
        for t in cutlass.range_constexpr(T):
            qval = _conv_value(x_s, hist_s, taps[0], t, 0, u)
            kval = _conv_value(x_s, hist_s, taps[1], t, 1, u)
            vval = _conv_value(x_s, hist_s, taps[2], t, 2, u)
            qk_raw[t * 2 * HEAD_DIM + u] = qval
            qk_raw[t * 2 * HEAD_DIM + HEAD_DIM + u] = kval
            v_s[t * HEAD_DIM + vperm] = vval
            if cutlass.const_expr(cfg.DEBUG):
                if ntok > t:
                    convout[rows[t] * QKV_CHANNELS + V_CHANNEL0 + head * HEAD_DIM + u] = vval.to(BF16)
                    if head % 3 == 0:
                        convout[rows[t] * QKV_CHANNELS + kh * HEAD_DIM + u] = qval.to(BF16)
                        convout[rows[t] * QKV_CHANNELS + K_CHANNEL0 + kh * HEAD_DIM + u] = kval.to(BF16)
        _wg_barrier(1)
        # the per-token RMS scales: half-warp hw reduces token 8 r + hw as the reference's half-warps do (same lanes, same butterfly),
        # then writes that token's SCALED q / k slices (lanes past the last token compute a duplicate and store nothing)
        qk_base = _smem_addr(qk_raw)
        q_base = _smem_addr(q_s)
        k_base = _smem_addr(k_s)
        sl_off = (8 * l16 // C) * PITCH + (8 * l16 % C)  # the slice position of columns 8 l16 .. 8 l16 + 7
        for r in cutlass.range_constexpr(RMS_ROUNDS):
            pt = cutlass.min(INT32(r * (WARPGROUP // RMS_LANES)) + hw, INT32(T - 1))
            qv = _slice_values(qk_base + (pt * 2 * HEAD_DIM + RMS_COLS * l16) * 4, RMS_COLS)
            kv = _slice_values(qk_base + (pt * 2 * HEAD_DIM + HEAD_DIM + RMS_COLS * l16) * 4, RMS_COLS)
            qs = FP32(0)
            ks = FP32(0)
            for j in cutlass.range_constexpr(RMS_COLS):
                qs = qs + qv[j] * qv[j]
                ks = ks + kv[j] * kv[j]
            for step in cutlass.range_constexpr(RMS_LANES.bit_length() - 1):
                qs = qs + cute.arch.shuffle_sync_bfly(qs, offset=1 << step)
                ks = ks + cute.arch.shuffle_sync_bfly(ks, offset=1 << step)
            qscale = FP32(Q_SCALE) / cute.math.sqrt(qs + FP32(RMS_EPS))
            kscale = FP32(1) / cute.math.sqrt(ks + FP32(RMS_EPS))
            if INT32(r * (WARPGROUP // RMS_LANES)) + hw < INT32(T):
                qsc = []
                ksc = []
                for j in cutlass.range_constexpr(RMS_COLS):
                    qsc.append(qv[j] * qscale)
                    ksc.append(kv[j] * kscale)
                _store_slice(q_base + (pt * cfg.TPR * PITCH + sl_off) * 4, qsc)
                _store_slice(k_base + (pt * cfg.TPR * PITCH + sl_off) * 4, ksc)
                if l16 == 0:
                    scal_s[4 * pt + 2] = qscale
                    scal_s[4 * pt + 3] = kscale


@cute.jit
def _token_step_rolled(
    cfg: cutlass.Constexpr,
    t,
    S_rows,
    c_prev2,
    tid,
    head,
    r,
    n,
    bos,
    ntok,
    orow,
    rowbase,
    rowbase_cta,
    col0,
    vbase,
    tpr,
    warp,
    lane,
    slice_base,
    k_base0,
    q_base0,
    recv_base,
    peer,
    wv,
    scv,
    zv,
    state,
    indices,
    out,
    qout,
    raw,
    q_s,
    k_s,
    v_s,
    scal_s,
    o_s,
    inv_s,
    norm_recv_s,
    mb_norm,
    ring,
    SS: cutlass.Constexpr,
    IS: cutlass.Constexpr,
    EPS: cutlass.Constexpr,
):
    """One token of the ROLLED loop (a DYNAMIC ``t``): the body of the unrolled loop with every per-token address computed from ``t``, the
    deferred norm (``t >= 1``) and outputs (``t >= 2``: the core values of token ``t - 2`` = ``c_prev2``) under uniform predicates.
    Returns the updated row registers and this token's core values."""
    TPR = cfg.TPR
    R = cfg.R
    C = cfg.C
    RG = cfg.RG
    NBUF = cfg.NBUF
    PITCH = cfg.SLICE_PITCH
    TWO = cfg.TWO_CTAS_PER_HEAD
    NCH = cfg.NORM_CHAINS_CTA
    tm1 = t - 1
    dst_buf = t % NBUF
    dst_base = slice_base + dst_buf * (cfg.SLAB_SMEM_ELEMS * 4)
    decay = scal_s[4 * t]
    beta = scal_s[4 * t + 1]
    k_base = k_base0 + t * (TPR * PITCH * 4)
    q_base = q_base0 + t * (TPR * PITCH * 4)
    ss_prev = FP32(0)
    if cutlass.const_expr(TWO):
        if t >= 1:
            if warp == NORM_WARP:
                ss_prev = _norm_chain_sum(cfg, o_s, tm1 % 2, lane)
                if lane == 0:
                    nvvm.mbarrier_arrive_expect_tx(mb_norm.subview(tm1 % 2), cfg.NORM_XFER_BYTES)
                if lane < NCH:
                    _st_async_f32(
                        _mapa_u32(recv_base + ((tm1 % 2) * NORM_RECV_MAX_FLOATS + lane) * 4, peer),
                        ss_prev,
                        _mapa_u32(_smem_addr(mb_norm) + (tm1 % 2) * 8, peer),
                    )
    k = _slice_values(k_base, C) if cutlass.const_expr(cfg.K_RESIDENT) else None
    h_rows, dots = _pass1(cfg, S_rows, slice_base, k, k_base, decay)
    inn = []
    for m in cutlass.range_constexpr(R):
        inn.append((v_s[t * HEAD_DIM + vbase + m * cfg.V_PERM_STRIDE] - dots[m]) * beta)
    if cutlass.const_expr(cfg.BULK_STORE and NBUF == 1):
        if t >= 1:
            if lane == 0:
                nvvm.cp_async_bulk_wait_group(0, read=True)
            _cta_barrier()
    S_new, values = _pass2(cfg, h_rows, slice_base, dst_base, k, k_base, q_base, inn, decay, cfg.BULK_STORE)
    if cutlass.const_expr(not cfg.BULK_STORE):
        dst_stg = indices[n * IS + t].to(INT64)
        if ntok > t:
            if dst_stg > 0:
                for m in cutlass.range_constexpr(R):
                    _store_row_gmem(state, dst_stg, head, rowbase + m, col0, S_new[m], SS)
    if tpr == 0:
        for m in cutlass.range_constexpr(R):
            o_s[(t % 2) * HEAD_DIM + rowbase_cta + m] = values[m]
    if cutlass.const_expr(cfg.BULK_STORE):
        cute.arch.fence_view_async_shared()
    if t >= 1:
        if warp == NORM_WARP:
            if cutlass.const_expr(TWO):
                wait(mb_norm.subview(tm1 % 2), (tm1 >> 1) & 1)
                inv_prev = _norm_inverse_pair(cfg, ss_prev, norm_recv_s[(tm1 % 2) * NORM_RECV_MAX_FLOATS + lane % NCH], r, lane, EPS)
            else:
                inv_prev = _norm_inverse(cfg, o_s, tm1 % 2, lane, EPS)
            if lane == 0:
                inv_s[tm1 % 2] = inv_prev
    if t >= 2:
        if tpr < R:
            if ntok > t - 2:
                oi = (bos + (t - 2)).to(INT64) * OUT_ROW + head * HEAD_DIM + orow
                _emit_output(cfg, c_prev2, tpr, inv_s[(t - 2) % 2], wv, _pick_dyn(zv, t - 2), scv, oi, out, qout, raw)
    if cutlass.const_expr(cfg.BULK_STORE and NBUF >= 2):
        if lane == 0:
            nvvm.cp_async_bulk_wait_group(NBUF - 2, read=True)
    _cta_barrier()
    if cutlass.const_expr(cfg.BULK_STORE):
        if lane == 0:
            dst = indices[n * IS + t].to(INT64)
            if ntok > t:
                if dst > 0:
                    for b in cutlass.range_constexpr(RG):
                        if cutlass.const_expr(cfg.SPREAD_ISSUE):
                            if warp == b % WARPS:
                                _issue_state_store_box(cfg, state, ring, dst_buf, dst, head, r, SS, b)
                        else:
                            if warp == 0:
                                _issue_state_store_box(cfg, state, ring, dst_buf, dst, head, r, SS, b)
            nvvm.cp_async_bulk_commit_group()
    return S_new, values


# --------------------------------------------------------------------------------------------------------------------------
# the role-independent tail (traced once): the hand-off, the slab, the T tokens, the deferred norm / outputs, the shift, the drain
# --------------------------------------------------------------------------------------------------------------------------
@cute.jit
def _tail_live(
    cfg: cutlass.Constexpr,
    tid,
    head,
    r,
    kh,
    n,
    bos,
    ntok,
    cv_slot,
    state,
    indices,
    z,
    w,
    scale,
    out,
    qout,
    cs,
    raw,
    x_s,
    hist_s,
    q_s,
    k_s,
    v_s,
    scal_s,
    o_s,
    inv_s,
    norm_recv_s,
    ticket_s,
    mb_state,
    mb_norm,
    ring,
    ZS: cutlass.Constexpr,
    SS: cutlass.Constexpr,
    IS: cutlass.Constexpr,
    CSS: cutlass.Constexpr,
    CST: cutlass.Constexpr,
    EPS: cutlass.Constexpr,
    T: cutlass.Constexpr,
):
    TPR = cfg.TPR
    R = cfg.R
    C = cfg.C
    RG = cfg.RG
    NBUF = cfg.NBUF
    PITCH = cfg.SLICE_PITCH
    TWO = cfg.TWO_CTAS_PER_HEAD
    NCH = cfg.NORM_CHAINS_CTA
    warp = tid // 32
    lane = tid % 32
    tpr = lane // RG
    rg = lane % RG
    rowbase_cta = cfg.BOX_ROWS * rg + R * warp  # this CTA's row (0 .. CTA_ROWS - 1): the o_s index and the box row
    rowbase = (cfg.CTA_ROWS * r + rowbase_cta) if TWO else rowbase_cta  # the head row (the one-CTA form adds nothing: its arithmetic is the committed one)
    col0 = tpr * C
    orow = rowbase + cutlass.min(tpr, R - 1)  # the row this thread outputs (the tpr < R threads each take one of the row group's R rows)
    vbase = ((R * warp) * cfg.V_PERM_STRIDE + r * RG + rg) if TWO else (R * warp) * RG + rg  # v_s index of row rowbase (+ m V_PERM_STRIDE)
    slice_base = _smem_addr(ring) + rg * (cfg.STATE_BOX_PITCH_ELEMS * 4) + (R * warp) * ROW_BYTES + col0 * 4  # in ring buffer 0
    q_base0 = _smem_addr(q_s) + tpr * (PITCH * 4)
    k_base0 = _smem_addr(k_s) + tpr * (PITCH * 4)
    wv = w[orow].to(FP32)
    scv = scale[0]
    zv = []
    for t in cutlass.range_constexpr(T):
        zv.append(z[(bos + cutlass.min(INT32(t), ntok - 1)).to(INT64) * ZS + head * HEAD_DIM + orow].to(FP32))
    peer = INT32(0)
    recv_base = INT32(0)
    if cutlass.const_expr(TWO):
        # the norm exchange's addresses: the peer's receive slot of each thread's chain value and the peer's mb_norm stages (mapa)
        peer = INT32(cfg.CTAS_PER_HEAD - 1) - r
        recv_base = _smem_addr(norm_recv_s)
    _cta_barrier()  # the hand-off: q / k / v / scalars / ticket published; the state mbarrier init published
    if cutlass.const_expr(TWO):
        # P4 step 3: both CTAs' mbarrier inits are visible cluster-wide from here (the acquire wait pairs with the entry's relaxed arrives
        # and the init lane's fence.mbarrier_init); nothing crosses the pair before this point
        nvvm.barrier_cluster_wait()
    wait(mb_state, 0)
    S_rows = None
    if cutlass.const_expr(not cfg.STATE_STREAM):
        S_rows = []
        for m in cutlass.range_constexpr(R):
            S_rows.append(_load_row_smem(slice_base, m, C))
    core_hist = []
    T_UNROLLED = 0 if cutlass.const_expr(cfg.ROLLED_TOKENS) else T  # the rolled form below traces the body once; the unrolled loop then has no iterations
    for t in cutlass.range_constexpr(T_UNROLLED):
        src_buf = 0 if cutlass.const_expr(not cfg.STATE_STREAM) else t % NBUF
        dst_buf = t % NBUF if cutlass.const_expr(not cfg.STATE_STREAM) else (t + 1) % NBUF
        src_base = slice_base + src_buf * (cfg.SLAB_SMEM_ELEMS * 4)
        dst_base = slice_base + dst_buf * (cfg.SLAB_SMEM_ELEMS * 4)
        decay = scal_s[4 * t]
        beta = scal_s[4 * t + 1]
        k_base = k_base0 + t * (TPR * PITCH * 4)
        q_base = q_base0 + t * (TPR * PITCH * 4)
        ss_prev = FP32(0)
        if cutlass.const_expr(TWO and t >= 1):
            # the norm exchange of token t - 1, SHIP half: the norm warp sums its local chains from o_s[(t - 1) % 2] (published by
            # barrier t - 1), lane 0 arms this CTA's mb_norm stage (t - 1) % 2 for the peer's bytes, lanes < NCH ship their chain
            # value into the peer's receive slot (st.async completing on the peer's stage); the wait sits after this token's passes
            if warp == NORM_WARP:
                ss_prev = _norm_chain_sum(cfg, o_s, (t - 1) % 2, lane)
                if lane == 0:
                    nvvm.mbarrier_arrive_expect_tx(mb_norm.subview((t - 1) % 2), cfg.NORM_XFER_BYTES)
                if lane < NCH:
                    _st_async_f32(
                        _mapa_u32(recv_base + (((t - 1) % 2) * NORM_RECV_MAX_FLOATS + lane) * 4, peer),
                        ss_prev,
                        _mapa_u32(_smem_addr(mb_norm.subview((t - 1) % 2)), peer),
                    )
        k = _slice_values(k_base, C) if cutlass.const_expr(cfg.K_RESIDENT) else None
        h_rows, dots = _pass1(cfg, S_rows, src_base, k, k_base, decay)
        inn = []
        for m in cutlass.range_constexpr(R):
            inn.append((v_s[t * HEAD_DIM + vbase + m * cfg.V_PERM_STRIDE] - dots[m]) * beta)
        if cutlass.const_expr(cfg.BULK_STORE and NBUF == 1 and t >= 1):
            # the one box: the previous token's bulk store must have read it before pass 2 writes it (pass 1 only read)
            if lane == 0:
                nvvm.cp_async_bulk_wait_group(0, read=True)
            _cta_barrier()
        S_rows, values = _pass2(cfg, h_rows, src_base, dst_base, k, k_base, q_base, inn, decay, cfg.BULK_STORE)
        if cutlass.const_expr(not cfg.BULK_STORE):
            # the stg path: every lane's 16-B stores of its updated row slices straight to the token's slot (no ring, no fence)
            dst_stg = indices[n * IS + t].to(INT64)
            if ntok > t:
                if dst_stg > 0:
                    for m in cutlass.range_constexpr(R):
                        _store_row_gmem(state, dst_stg, head, rowbase + m, col0, S_rows[m], SS)
        if tpr == 0:
            for m in cutlass.range_constexpr(R):
                o_s[(t % 2) * HEAD_DIM + rowbase_cta + m] = values[m]
        if cutlass.const_expr(cfg.BULK_STORE):
            cute.arch.fence_view_async_shared()  # this thread's STS of the updated rows -> visible to the bulk store (async proxy)
        if cutlass.const_expr(t >= 1):
            if warp == NORM_WARP:
                if cutlass.const_expr(TWO):
                    # the norm exchange of token t - 1, COMBINE half: the peer's chain values have landed (phase ((t - 1) // 2) & 1 of
                    # stage (t - 1) % 2); every lane takes head chain `lane`'s value (own or received) and runs the reference sequence
                    wait(mb_norm.subview((t - 1) % 2), ((t - 1) // 2) & 1)
                    inv_prev = _norm_inverse_pair(cfg, ss_prev, norm_recv_s[((t - 1) % 2) * NORM_RECV_MAX_FLOATS + lane % NCH], r, lane, EPS)
                else:
                    inv_prev = _norm_inverse(cfg, o_s, (t - 1) % 2, lane, EPS)
                if lane == 0:
                    inv_s[(t - 1) % 2] = inv_prev
        if cutlass.const_expr(t >= 2):
            if tpr < R:
                if ntok > t - 2:
                    oi = (bos + (t - 2)).to(INT64) * OUT_ROW + head * HEAD_DIM + orow
                    _emit_output(cfg, core_hist[t - 2], tpr, inv_s[(t - 2) % 2], wv, zv[t - 2], scv, oi, out, qout, raw)
        if cutlass.const_expr(cfg.BULK_STORE and NBUF >= 2):
            if lane == 0:
                nvvm.cp_async_bulk_wait_group(NBUF - 2, read=True)  # this thread's box of (t + 1) % NBUF is read out before the next token's STS
        _cta_barrier()
        if cutlass.const_expr(cfg.BULK_STORE):
            if lane == 0:
                # box b is stored by warp b mod 8 (spread) or by warp 0 (single); every issuing lane commits one group per token on
                # every path (empty for a skipped slot or a token past ntok), so each lane's ring ledger stays exact
                dst = indices[n * IS + t].to(INT64)
                if ntok > t:
                    if dst > 0:
                        for b in cutlass.range_constexpr(RG):
                            if cutlass.const_expr(cfg.SPREAD_ISSUE):
                                if warp == b % WARPS:
                                    _issue_state_store_box(cfg, state, ring, dst_buf, dst, head, r, SS, b)
                            else:
                                if warp == 0:
                                    _issue_state_store_box(cfg, state, ring, dst_buf, dst, head, r, SS, b)
                nvvm.cp_async_bulk_commit_group()
        core_hist.append(values)
    if cutlass.const_expr(cfg.ROLLED_TOKENS):
        # ROLLED_TOKENS: the same token body traced ONCE (`cutlass.range(T, unroll=1)`) inside `_token_step_rolled`: the row registers
        # and the two-deep ring of core values are loop-carried, every per-token shared address is computed from the loop variable, the
        # deferred stages run under the uniform predicates t >= 1 / t >= 2 -- a third of the straight-line image (the cold fetch)
        c_prev1 = []
        c_prev2 = []
        values = []  # every name the loop assigns exists before it with the same structure (the DSL's loop-carry rule)
        for _ in cutlass.range_constexpr(R):
            c_prev1.append(FP32(0))
            c_prev2.append(FP32(0))
            values.append(FP32(0))
        for t in cutlass.range(T, unroll=1):
            S_rows, values = _token_step_rolled(
                cfg, t, S_rows, c_prev2, tid, head, r, n, bos, ntok, orow, rowbase, rowbase_cta, col0, vbase, tpr, warp, lane, slice_base, k_base0, q_base0,
                recv_base, peer, wv, scv, zv, state, indices, out, qout, raw, q_s, k_s, v_s, scal_s, o_s, inv_s, norm_recv_s, mb_norm, ring, SS, IS, EPS,
            )  # fmt: skip
            c_prev2 = c_prev1
            c_prev1 = values
        core_hist = [c_prev2, c_prev1]
    # ---- epilogue: the two deferred tokens' norm / outputs ----
    ss_last = FP32(0)
    if cutlass.const_expr(TWO):
        # token T - 1's exchange, SHIP half (its o_s slot was published by barrier T - 1); the outputs of T - 2 below give the bytes time
        if warp == NORM_WARP:
            ss_last = _norm_chain_sum(cfg, o_s, (T - 1) % 2, lane)
            if lane == 0:
                nvvm.mbarrier_arrive_expect_tx(mb_norm.subview((T - 1) % 2), cfg.NORM_XFER_BYTES)
            if lane < NCH:
                _st_async_f32(
                    _mapa_u32(recv_base + (((T - 1) % 2) * NORM_RECV_MAX_FLOATS + lane) * 4, peer),
                    ss_last,
                    _mapa_u32(_smem_addr(mb_norm.subview((T - 1) % 2)), peer),
                )
    if cutlass.const_expr(T >= 2):
        if tpr < R:
            if ntok > T - 2:
                oi = (bos + (T - 2)).to(INT64) * OUT_ROW + head * HEAD_DIM + orow
                _emit_output(cfg, core_hist[-2], tpr, inv_s[(T - 2) % 2], wv, zv[T - 2], scv, oi, out, qout, raw)
    if warp == NORM_WARP:
        if cutlass.const_expr(TWO):
            wait(mb_norm.subview((T - 1) % 2), ((T - 1) // 2) & 1)
            inv_last = _norm_inverse_pair(cfg, ss_last, norm_recv_s[((T - 1) % 2) * NORM_RECV_MAX_FLOATS + lane % NCH], r, lane, EPS)
        else:
            inv_last = _norm_inverse(cfg, o_s, (T - 1) % 2, lane, EPS)
        if lane == 0:
            inv_s[(T - 1) % 2] = inv_last
    _cta_barrier()
    if tpr < R:
        if ntok > T - 1:
            oi = (bos + (T - 1)).to(INT64) * OUT_ROW + head * HEAD_DIM + orow
            _emit_output(cfg, core_hist[-1], tpr, inv_s[(T - 1) % 2], wv, zv[T - 1], scv, oi, out, qout, raw)
    # ---- the conv-state shift: the new slot is [old[a], old[a + 1], x_0 .. x_{T-1}] (16-B groups of 8 channels per thread) ----
    if tid < SHIFT_THREADS:
        piece = tid // (HEAD_DIM // 8)
        cg = tid % (HEAD_DIM // 8)
        # this CTA's V channels (its own 64 of the head's 128 on the pair: 16-B groups 8 r .. 8 r + 7); the K head's Q / K channels belong
        # to the last of its 3 x CTAS_PER_HEAD readers
        own = piece == 2
        if cutlass.const_expr(cfg.TWO_CTAS_PER_HEAD):
            own = (piece == 2) & (cg // (HEAD_DIM // 8 // cfg.CTAS_PER_HEAD) == r)
            if ticket_s[0] == cfg.TICKET_LIMIT:
                own = own | (piece < 2)
        else:
            if ticket_s[0] == cfg.TICKET_LIMIT:
                own = piece < 3
        if own:
            ch0 = kh * HEAD_DIM + piece * K_CHANNEL0  # pieces 0 / 1: this K head's Q / K channels
            if piece == 2:
                ch0 = V_CHANNEL0 + head * HEAD_DIM
            dst_base_cs = cv_slot * CSS + ch0 + cg * 8
            x_addr = _smem_addr(x_s) + (piece * HEAD_DIM + cg * 8) * 2
            h_addr = _smem_addr(hist_s) + (piece * CONV_HIST_TAPS * HEAD_DIM + cg * 8) * 2
            for col in cutlass.range_constexpr(T + 2):
                if cutlass.const_expr(col < 2):
                    words = ld_shared_v4(h_addr + (col + 1) * HEAD_DIM * 2, INT32)
                    st_global_v4(_gaddr(cs, dst_base_cs + col * CST), words, INT32)
                else:
                    if ntok > col - 2:
                        words = ld_shared_v4(x_addr + (col - 2) * 3 * HEAD_DIM * 2, INT32)
                        st_global_v4(_gaddr(cs, dst_base_cs + col * CST), words, INT32)
    if cutlass.const_expr(cfg.BULK_STORE):
        if lane == 0:
            nvvm.cp_async_bulk_wait_group(0, read=True)  # every issuing lane: its boxes are read out of smem before this CTA retires


@cute.jit
def _tail_null(cfg: cutlass.Constexpr, tid, head, r, bos, ntok, out, qout, raw, T: cutlass.Constexpr):
    """A request without a source state (or without a conv slot / tokens): zero outputs for its rows of this head, nothing else (on the
    pair the cluster wait pairs with the entry arrive -- both CTAs of a pair are null together, nothing is shipped)."""
    R = cfg.R
    RG = cfg.RG
    warp = tid // 32
    lane = tid % 32
    tpr = lane // RG
    rg = lane % RG
    orow = cfg.BOX_ROWS * rg + R * warp + cutlass.min(tpr, R - 1)
    if cutlass.const_expr(cfg.TWO_CTAS_PER_HEAD):
        orow = orow + cfg.CTA_ROWS * r
        nvvm.barrier_cluster_wait()
    if cutlass.const_expr(cfg.PDL):
        wait_on_dependent_grids()
    if tpr < R:
        for t in cutlass.range_constexpr(T):
            if ntok > t:
                oi = (bos + t).to(INT64) * OUT_ROW + head * HEAD_DIM + orow
                qout[oi] = UINT8(0)
                out[oi] = BF16(0)
                if cutlass.const_expr(cfg.DEBUG):
                    raw[oi] = BF16(0)


# --------------------------------------------------------------------------------------------------------------------------
# the kernel
# --------------------------------------------------------------------------------------------------------------------------
@cute.kernel
def frost_gdn_verify_fused(
    cfg: cutlass.Constexpr,
    qkv: cute.Tensor,
    z: cute.Tensor,
    hidden: cute.Tensor,
    baweight: cute.Tensor,
    al: cute.Tensor,
    dt: cute.Tensor,
    state: cute.Tensor,
    indices: cute.Tensor,
    cu: cute.Tensor,
    accepted: cute.Tensor,
    w: cute.Tensor,
    scale: cute.Tensor,
    counts: cute.Tensor,
    out: cute.Tensor,
    qout: cute.Tensor,
    cs: cute.Tensor,
    cw: cute.Tensor,
    raw: Optional[cute.Tensor],
    convout: Optional[cute.Tensor],
    baout: Optional[cute.Tensor],
    XS: cutlass.Constexpr,
    ZS: cutlass.Constexpr,
    SS: cutlass.Constexpr,
    IS: cutlass.Constexpr,
    CSS: cutlass.Constexpr,
    CST: cutlass.Constexpr,
    EPS: cutlass.Constexpr,
    T: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    bx, n, _ = cute.arch.block_idx()
    # the two-CTAs-per-head form: blocks (2 h, 2 h + 1) are one cluster along x; r = the CTA's rank in it (= %cluster_ctarank) = its row half
    head = bx // cfg.CTAS_PER_HEAD if cfg.TWO_CTAS_PER_HEAD else bx
    r = bx % cfg.CTAS_PER_HEAD if cfg.TWO_CTAS_PER_HEAD else INT32(0)
    kh = head // 3
    if cutlass.const_expr(cfg.PDL_RELEASE):
        launch_dependent_grids()

    SMEM = cutlass.AddressSpace.smem
    x_s = cutlass.Array(BF16, 3 * T * HEAD_DIM, space=SMEM, alignment=16)
    cw_s = cutlass.Array(BF16, 3 * HEAD_DIM * CONV_WIDTH, space=SMEM, alignment=16)
    hist_s = cutlass.Array(BF16, 3 * CONV_HIST_TAPS * HEAD_DIM, space=SMEM, alignment=16)
    qk_raw = cutlass.Array(FP32, 2 * T * HEAD_DIM, space=SMEM, alignment=16)
    q_s = cutlass.Array(FP32, T * cfg.TPR * cfg.SLICE_PITCH, space=SMEM, alignment=16)
    k_s = cutlass.Array(FP32, T * cfg.TPR * cfg.SLICE_PITCH, space=SMEM, alignment=16)
    v_s = cutlass.Array(FP32, T * HEAD_DIM, space=SMEM, alignment=16)
    scal_s = cutlass.Array(FP32, 4 * T, space=SMEM, alignment=16)
    o_s = cutlass.Array(FP32, 2 * HEAD_DIM, space=SMEM, alignment=16)
    inv_s = cutlass.Array(FP32, 4, space=SMEM, alignment=16)
    ba_part = cutlass.Array(FP32, 2 * WARPS * T, space=SMEM, alignment=16)
    ticket_s = cutlass.Array(INT32, 4, space=SMEM, alignment=16)
    mb_arr = cutlass.Array(INT64, 1, space=SMEM, alignment=16)
    mb_state = cutlass.Array(INT64, 1, space=SMEM, alignment=16)
    # the two-CTA form: the peer's norm chain values of token t land in norm_recv_s slot t % 2 and complete bytes on mb_norm stage t % 2
    # (one-CTA form: unused aliases of existing arrays, nothing allocated)
    norm_recv_s = cutlass.Array(FP32, NORM_MBAR_STAGES * NORM_RECV_MAX_FLOATS, space=SMEM, alignment=16) if cfg.TWO_CTAS_PER_HEAD else scal_s
    mb_norm = cutlass.Array(INT64, NORM_MBAR_STAGES, space=SMEM, alignment=16) if cfg.TWO_CTAS_PER_HEAD else mb_state
    ring = cutlass.Array(FP32, cfg.RING_SMEM_ELEMS, space=SMEM, alignment=16)
    if cutlass.const_expr(cfg.smem_pad_bytes(T) > 0):
        pad_s = cutlass.Array(INT32, cfg.smem_pad_bytes(T) // 4, space=SMEM, alignment=16)
        if head == INT32(0x7FFFFFF):  # never true for a 48-wide grid: keeps the occupancy pad allocated
            pad_s[0] = tid

    # ---- the slot chain (never a predecessor's output): the source slot, the conv slot, the token range, the live predicate ----
    bos = cu[n]
    ntok = cu[n + 1] - bos
    acc = accepted[n]
    src = INT64(0)
    if acc > 0:
        if acc <= INT32(T):
            src = indices[n * IS + acc - 1].to(INT64)
    cv_slot = indices[n * IS].to(INT64)
    live = INT32(0)
    if src > 0:
        if cv_slot > 0:
            if ntok > 0:
                live = INT32(1)
    is_live = live == 1

    if tid < WARPGROUP:
        _ba_role(cfg, tid, head, r, n, bos, ntok, is_live, src, al, dt, hidden, baweight, state, scal_s, ba_part, mb_state, mb_norm, ring, baout, SS, T)
    else:
        _conv_role(
            cfg, tid, head, kh, n, bos, ntok, is_live, cv_slot, acc, qkv, cs, cw, counts,
            x_s, cw_s, hist_s, qk_raw, q_s, k_s, v_s, scal_s, ticket_s, mb_arr, convout, XS, CSS, CST, T,
        )  # fmt: skip
    if is_live:
        _tail_live(
            cfg, tid, head, r, kh, n, bos, ntok, cv_slot, state, indices, z, w, scale, out, qout, cs, raw,
            x_s, hist_s, q_s, k_s, v_s, scal_s, o_s, inv_s, norm_recv_s, ticket_s, mb_state, mb_norm, ring, ZS, SS, IS, CSS, CST, EPS, T,
        )  # fmt: skip
    else:
        _tail_null(cfg, tid, head, r, bos, ntok, out, qout, raw, T)


frost_gdn_verify_fused.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def launch(
    cfg: cutlass.Constexpr,
    qkv: cute.Tensor,
    z: cute.Tensor,
    hidden: cute.Tensor,
    baweight: cute.Tensor,
    al: cute.Tensor,
    dt: cute.Tensor,
    state: cute.Tensor,
    indices: cute.Tensor,
    cu: cute.Tensor,
    accepted: cute.Tensor,
    w: cute.Tensor,
    scale: cute.Tensor,
    counts: cute.Tensor,
    out: cute.Tensor,
    qout: cute.Tensor,
    cs: cute.Tensor,
    cw: cute.Tensor,
    raw: Optional[cute.Tensor],
    convout: Optional[cute.Tensor],
    baout: Optional[cute.Tensor],
    REQUESTS: cutlass.Constexpr,
    XS: cutlass.Constexpr,
    ZS: cutlass.Constexpr,
    SS: cutlass.Constexpr,
    IS: cutlass.Constexpr,
    CSS: cutlass.Constexpr,
    CST: cutlass.Constexpr,
    EPS: cutlass.Constexpr,
    T: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    """One CTA per (V head, request) -- a cluster of two per head under ``TWO_CTAS_PER_HEAD``; 256 threads; ``MIN_CTAS_PER_SM`` sets
    ptxas's register cap (1 -> 255, 2 -> 128) and ``CTAS_PER_SM`` the shared-memory cap.  ``raw`` / ``convout`` / ``baout`` are the DEBUG
    dumps: ``None`` unless ``cfg.DEBUG`` -- the DSL specializes on ``None`` and the slots are compiled out (every use sits under the
    DEBUG constexpr), never bound to an allocation."""
    kernel = frost_gdn_verify_fused(
        cfg, qkv, z, hidden, baweight, al, dt, state, indices, cu, accepted, w, scale, counts, out, qout, cs, cw, raw, convout, baout,
        XS, ZS, SS, IS, CSS, CST, EPS, T,
    )  # fmt: skip
    if cutlass.const_expr(cfg.TWO_CTAS_PER_HEAD):
        kernel.launch(
            grid=(V_HEADS * cfg.CTAS_PER_HEAD, REQUESTS, 1),
            block=(THREADS, 1, 1),
            cluster=(cfg.CTAS_PER_HEAD, 1, 1),
            stream=stream,
            min_blocks_per_mp=cfg.MIN_CTAS_PER_SM,
            use_pdl=cfg.PDL,
        )
    else:
        kernel.launch(
            grid=(V_HEADS, REQUESTS, 1),
            block=(THREADS, 1, 1),
            stream=stream,
            min_blocks_per_mp=cfg.MIN_CTAS_PER_SM,
            use_pdl=cfg.PDL,
        )
