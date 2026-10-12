# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gated attention block, backward -- bf16 / fp16 over a proj_slab record (the unfused assembly plus its first fused step), the
per-tensor fp8 backward over the fp8 training record (``quant=QuantSpec``) and the MXFP8 backward over the MXFP8 training record
(``quant=MxQuantSpec``), with the MXFP8 pipeline's fp4 weight modes (an MXFP4 ``W_qkvg``, an NVFP4 / MXFP4 ``W_o``) on the same record.

Read :mod:`cudnn.gated_attention_block.api` first. This module is defined
against that file's :class:`~cudnn.gated_attention_block.api.SavedForBackward`
contract and its ``qkvg_offsets`` ordering; neither is restated here.

.. note::

   **MAINTENANCE.** Same rule as the forward: this docstring, the launch and
   workspace tables below and the per-stage **Fusion status** paragraphs are
   the as-built record of this backward. Update them in the same commit as any stage
   changing, any fusion, any dtype; the measured numbers behind the tolerances and
   the launch counts live in ``test_block_backward.py``'s module docstring.

The op graph, in LAUNCH order -- one stream, nothing overlaps anything::

    dY [B, S, d_model]
     |
     +-- (B2) out_proj dgrad     dO_gated = dY @ W_o                          -> ws.do_gated [T, H_q, D]
     |   (B3) gate backward, elementwise, s = sigmoid(GATE):
     |          dO      = dO_gated * s                                        -> IN PLACE over ws.do_gated
     |          dG      = dO_gated * O * s * (1 - s)                          -> ws.dqkvg [GATE band]
     |          O_gated = O * s                        (need_dw_o only)       -> ws.o_gated
     +-- (B1) out_proj wgrad     dW_o = dY^T @ O_gated             K = B*S    (AFTER B3: it takes O_gated from B3's third output)
     |
     |   recompute  Q, K = norm + RoPE(q_pre, k_pre)  -- the forward's stage (2)+(3) kernel over the proj_slab bands
     |              V    = compact copy of the V band                         -> ws.recompute / recompute_k / recompute_v
     |   (B4) SDPA backward   (dO, Q, K, V, O, LSE) -> dQ, dK, dV  COMPACT    -> ws.dq / ws.dk / ws.dv
     |   (B5) RoPE^T + (B6) RMSNorm backward -- ONE kernel over dQ and dK (+ the bit-exact dV copy):
     |          ws.dqkvg [Q band] = dQ_pre, [K band] = dK_pre, [V band] = dV; fp32 dW_norm partials per CTA
     |        + the fixed-order reduce of the partials                        -> dW_q_norm, dW_k_norm   (qk_norm only)
     |
     +-- (B7) qkv+gate wgrad     dW_qkvg = dQKVG^T @ h              K = B*S
     +-- (B8) qkv+gate dgrad     dh      = dQKVG @ W_qkvg
              v
             dh [B, S, d_model]

Orientation follows the forward's ``nn.Linear`` TN form (``api.py``):
``QKVG = h @ W_qkvg^T`` with ``W_qkvg [N, d_model]`` and
``Y = O_gated @ W_o^T`` with ``W_o [d_model, H_q*D]``. For ``Y = X @ W^T`` the
gradients are ``dW = dY^T @ X`` and ``dX = dY @ W`` -- so both wgrads above
land directly in their weight's own ``[out, in]`` layout, and both dgrads
read the weight un-transposed (``kernels/proj_gemm.py::run_wgrad_gemm`` /
``run_dgrad_gemm``: M-major A / N-major B plans, zero-copy views, the declared
strides asserted before every bind).

**Six GEMMs and two attention kernels under one API.** That is roughly three
times the forward, and it is the reason the forward's saved-tensor contract was
pinned before any of it was written.

Two corrections to the first skeleton, both load-bearing:

* **B4 writes COMPACT dQ / dK / dV** (workspace slots), never the ``dqkvg``
  bands: the Rubin backward adapter (``sdpa/bwd/api_dsl_sm107.py``) requires
  every one of its eight io operands to be BSHD-physical -- stride exactly
  ``(S*H*D, D, H*D, 1)`` -- and its main kernel is compiled against compact
  fakes. The elementwise backward kernel (B5+B6) then writes the ``dqkvg``
  bands, reading the compact slots; the ``[T, N]`` slab B7 / B8 consume is
  still written exactly once per column.
* **B1 runs after B3**, taking ``O_gated`` from the gate-backward kernel's
  optional third output instead of a separate ``o * sigmoid(gate)`` launch (two
  reads and one launch saved). It is still the filler: it depends on nothing
  the SDPA backward produces.

**Fusion knob: ``fuse_gate_bwd`` (default False).**  The SDPA backward's first
launch is ``delta = rowsum(dO * O)`` over the very ``dO`` B3 just wrote and the
``O`` it just read; with the knob the gate-backward kernel emits ``delta`` as a
fourth output (``kernels/sigmoid_gate_bwd.py``: the bf16-ROUNDED ``dO`` it
stores, summed in the chain's own ``dot_do_o`` order, the pad rows zeroed) into a
block-owned fp32 ``[B, H_q, S_pad]`` region (under ``thd`` the dense arm at
``B = 1, S = T``: ``[1, H_q, ceil128(T)]``, which IS the packed chain's head-major
delta layout), and the adapter is built with
``external_delta=True`` and handed that tensor -- one launch and one read each of
``O`` and ``dO`` fewer, the adapter's own ``delta`` region gone from its scratch.
Bitwise the unfused block (same fp32 operations in the same order; pinned by
``test_fused_gate_bwd_is_bitwise_the_unfused_block`` and, packed, by
``test_thd_fused_gate_bwd_is_bitwise_the_unfused_packed_block``), so it is a performance
knob in the Rule-9 sense: the same function under either value.

**Scheduling knob: ``fuse_wgrad_overlap`` (default False).**  The two
weight-gradient GEMMs are consumed by nothing inside the backward: B1 (``dW_o``,
ready after B3) and B7 (``dW_qkvg``, ready after B5+B6).  In order they sit on
the launch stream between stages they do not feed, so their SM time is
serialised with the SDPA backward chain (below full occupancy at small S, with
launch gaps between its kernels) and with the ``dh`` dgrad.  With the knob each
of them is issued on a block-owned SIDE stream instead: forked from the launch
stream through an event recorded right after its producer (B3 / B5+B6), joined
back through an event the launch stream waits on at the END of ``execute`` --
the latest legal point, because the side GEMMs read only buffers no later
stage writes (``dy``, ``o_gated``, the finished ``dqkvg`` slab, ``saved.h``) and
carve their split-K scratch, were the tile ever to need one, out of their own
appended ``gemm_scratch_side`` region, never the ``gemm_scratch`` the
launch-stream GEMMs share (``_WgradSideStream``).  Every write the caller can
observe is still ordered on the launch stream before ``execute`` returns, so an
ambient or an explicit launch stream sees the whole backward exactly as before
(Rule 5 kept; ``test_a_caller_stream_orders_every_stage`` runs under the knob);
the same kernels launch (the CUPTI count is unchanged) and the gradients are
bitwise the in-order block's -- the GEMMs are deterministic
(``test_fuse_wgrad_overlap_is_bitwise_the_in_order_block``).  The side stream
is the block's OWN -- one dedicated non-blocking stream at the device's lowest
priority, created through the driver, never a torch pool stream (so never a
caller's launch stream) -- and it and the four events are created once at
``compile()`` and released with the block (Rule 1: nothing per execute).  One
compiled block may be driven from several host threads on different launch
streams (the convenience wrapper caches a block process-wide): the fork, the
side GEMM's enqueue and the join record are one locked section, and the join
takes the same lock (``_WgradSideStream``).  CUDA-graph
capture of ``execute`` records the fork and the join as graph edges -- a side
stream that joins before the capture ends is the canonical fork/join pattern --
and the replay is bitwise the eager run
(``test_cuda_graph_capture_replays_bitwise``, both knob values).  The two do
not compose on ONE block: a capture of ``execute`` must not overlap an eager
``execute`` of the same compiled block from another thread, nor a second
capture of it -- the block has one side stream, inside the capture from its
first fork to its join -- and the overlap is a typed ``RuntimeError`` at the
eager fork or join, never a merged or invalidated graph (``_WgradSideStream``,
``test_fuse_wgrad_overlap_capture_and_eager_executes_do_not_overlap``).
Declined typed when neither ``need_dw_o`` nor ``need_dw_qkvg`` is set (nothing
to overlap);
the convenience wrapper, whose needs follow ``requires_grad``, runs the
in-order block instead when the weights are frozen (a frozen-weights phase of
a training loop must not fail over a scheduling knob).

**Packed sequences: ``thd`` (default False).**  The backward of a packed
(THD / varlen) training forward: ``dy``, ``saved.h``, ``cos`` / ``sin`` and
``dh`` are token matrices ``[T, .]`` (or ``[1, T, .]``) over ``T`` packed tokens
of ``num_sequences`` sequences, each at most ``max_seq_len`` long, whose
lengths the record carries as ``saved.seq_lens`` (int32 ``[B]`` lengths, or
``[B+1]`` prefix sums under ``cu_seqlens=True``; ``saved.seq_lens_form`` names
the form) -- REQUIRED, and never read on the host.  Inside the block that is
``B = 1, S = T``: every token-wise stage (the six GEMMs over ``K = T`` /
``M = T``, the gate backward, the Q / K rebuild, the V compaction, the norm +
RoPE backward with the caller's PER-TOKEN ``cos`` / ``sin``) runs unchanged on
the same workspace carve, and only the SDPA backward differs: it is the
adapter's packed chain, declared over the envelope ``(num_sequences, H,
max_seq_len, D)`` with both packed totals at ``T``, fed the record's lengths
as both length operands, and reading ``saved.lse`` head-major at head stride
exactly ``T`` -- the same contiguous ``[1, H_q, T]`` the packed forward
writes (``_thd_lse_head_stride``, ONE definition for both directions).
Caller contract on the lengths (device data): every length in
``[0, max_seq_len]``, prefix sums non-decreasing, and ``sum(lengths) == T`` --
the SDPA leaves rows past the live total unwritten while the weight-gradient
GEMMs contract over all ``T`` rows, so slack rows would contaminate
``dW_qkvg`` / ``dW_o``.  ``num_sequences * max_seq_len >= T`` is enforced at
declaration because a smaller product silently caps the adapter's packed
capacity below ``T`` (the chain would process the first ``B * S_max`` tokens
and report nothing).  Served under ``thd``: ``fuse_gate_bwd`` (the gate
backward's dense delta arm at ``B = 1, S = T`` writes the packed head-major
``[1, H_q, ceil128(T)]`` delta the packed chain reads -- the same bytes, tail
zeroed -- so the fused packed block is bitwise the unfused one and the chain's
``dot_do_o`` launch is gone) and ``fuse_wgrad_overlap`` (THD-agnostic).
Declined under ``thd``: ``seq_lens_present`` (mutually exclusive: a dense
padding mask is a different contract).  A packed record handed to a dense block, and a dense (padded)
record handed to a THD block, are typed declines naming the form.

**The quantized backward: ``quant`` (default None).**  The per-tensor fp8
TRAINING forward (``GatedAttentionBlockFwd(quant=QuantSpec, save_for_backward=True)``)
writes the SAME bf16 record as the bf16 forward -- a bf16 ``proj_slab`` with
PRE-norm Q / K bands, the bf16 pre-gate ``o``, the exact fp32 ``lse``,
``rstd_*`` -- with ``saved.h`` the caller's e4m3 codes.  Declared with the
forward's own ``QuantSpec`` (plan-time constants: the static ``scale_q / scale_k /
scale_v / scale_o`` of the record's SDPA operands, the descales of the e4m3
``h`` / ``W_qkvg`` / ``W_o``), the backward differentiates that record AS WRITTEN
-- e4m3 ``saved.h``, e4m3 weights, bf16 ``dy`` -- and returns bf16 ``dh`` /
``dW_qkvg`` / ``dW_o`` and fp32 ``dW_*_norm``.  What changes against the bf16
chain (:meth:`GatedAttentionBlockBwd._execute_quant` is the launch order):

* **the gradients are quantized to e4m3 before the GEMMs consume them**, each
  with a per-tensor scale derived ON DEVICE from its own amax pass
  (``grad_scaling="current"``: ``scale = 2**(floor(log2(448 / amax)) -
  FP8_GRAD_SCALE_MARGIN_LOG2)``; ``"delayed"``: the caller's previous-step
  ``scale_dy / scale_do / scale_dqkvg`` instead, the amax passes still
  published) -- ``dY -> dy8 [T, d_model]``, ``dO -> do8 [T, H_q, D]`` (its amax
  folded by the gate backward over the very dO words it stores), ``dQKVG ->
  dqkvg8 [T, N]``; no host readback, no allocation (Rule 1);
* **the four GEMMs run on e4m3 operands** (``dy8`` / ``og8`` / ``dqkvg8`` /
  ``saved.h`` and the two weights) at the forced tile's 64-byte MMA K form with a
  bf16 output and the fp32 ``alpha = descale_A * descale_B`` epilogue read from
  a slot of the scalar block (``alpha_b1 = descale_dY / scale_o``, ``alpha_b2 =
  descale_dY * descale_w_o``, ``alpha_b7 = descale_dQKVG * descale_h``,
  ``alpha_b8 = descale_dQKVG * descale_w_qkvg``, each ONE fp32 multiply
  published by the quantize launch that produced the descale);
* **the gate backward's fp8 arm** writes the e4m3 ``og8 = sat_e4m3(bf16(O * s) *
  scale_o)`` (bitwise the forward's quantize of the gated O) instead of the bf16
  ``O_gated``, and its ``delta = rowsum(dO * O)`` is MANDATORY: it is the fp8 SDPA
  row's external delta (the row's own pre-pass would recompute it over the e4m3
  payloads -- two roundings against the block's delta contract), so
  ``fuse_gate_bwd`` has no second arm here and is accepted and inert;
* **the SDPA backward is the fp8 row** (``SdpaBwdDslSm107Fp8``, external delta,
  ``amax_dP`` requested) over ``q8 / k8 / v8`` recomputed from the slab bands at
  the forward's STATIC scales -- bit-exactly the forward's own operands; ``v8``
  is quantized straight from the slab's V band, so there is no V compaction --
  the e4m3 ``do8``, the record's ``lse`` and the twelve row scalars: ``descale_q /
  k / v = 1 / scale_q / k / v`` and the dead ``descale_o = 1 / scale_o`` (plan-time
  constants, read from THEIR SLOTS of the scalar block; the dead ``o`` operand is
  bound to ``og8``, else ``do8``), ``scale_s = 2**FP8_SCALE_S_LOG2`` and its
  reciprocal (slots too), ``descale_dO`` and ``descale_dP`` from the scalar block,
  ``scale_dP`` = the caller's ``execute(scale_dp=)`` (cuDNN's fp8 backward
  contract: the dP scale is the caller's; ``descale_dp = 1 / scale_dp`` is derived
  on device), ``scale_dQ = scale_dK = scale_dV = 1.0`` (bf16 gradients out of the
  row; one shared slot);
* **a 256-B fp32 scalar block** in the workspace (``QUANT_SCALAR_SLOTS``: the four
  amax targets, the published scales / descales, the four alphas,
  ``descale_dp`` and the ``QuantSpec``'s fourteen PLAN-TIME CONSTANTS,
  ``QUANT_CONST_SLOTS``), written by the FIRST launch of every execute (the
  prologue's scalar-init job: every slot zeroed -- the row's ``amax_dP`` is an
  ``atomicMax`` target and must start from zero; the three gradient amax slots
  are PUBLISHED by their quantize launches from per-CTA partials --, then
  ``descale_dp`` and the constants stored from the launch's kernel arguments).
  NOTHING is written to the device at ``compile()``: a device tensor filled there
  would be enqueued on whatever stream was ambient at compile time, while
  ``execute`` reads it on the caller's stream with nothing ordering the two --
  the first execute of a block on a busy ambient stream could consume the
  constants before their fills landed.  Written by the launch that consumes
  them, on its stream, they are ordered by construction, and a CUDA-graph replay
  rewrites them because that launch is captured.  Readable through
  :meth:`GatedAttentionBlockBwd.quant_scalars` as zero-copy views after the step
  -- the amax of every quantized gradient and the row's ``amax_dP`` for the
  caller's scale bookkeeping;
* **the small kernels share launches** (the fp8 backward is HOST-bound at short
  sequences: ~15 us kernels behind tens of microseconds of host work per launch).
  Nine quantization-plumbing launches became three without a byte changing:
  the PROLOGUE launch (``kernels/fp8_bwd_fused.py``, block-range dispatch) runs
  the scalar init (the zeroing, ``descale_dp`` and the plan-time constants out of
  its kernel arguments; the rebuild's and ``v8``'s static scales reach that launch
  as kernel arguments too -- the values of their slots -- never as slot reads, since
  its init job writes those slots), the dY amax as per-CTA PARTIALS (no slot to zero first, so it
  can share the first launch; the dY quantize reduces them and publishes
  ``amax_dy``), the Q / K rebuild with an e4m3 epilogue (``q8`` / ``k8`` cast at the
  forward's static scales straight out of the TMA norm + RoPE body's registers --
  bf16 rounding first, so byte-identical to quantizing the bf16 rebuild; no bf16
  ``recompute`` buffers written or carved) and ``v8``; the dO and dqkvg amax are
  PER-CTA PARTIALS their producers store (the gate backward's max |dO| and max |dG|
  from a persistent grid, the norm backward's max over its Q / K / V bands: one
  plain store per CTA, never an atomic -- the slot form's same-address ``atomicMax``
  serialised at the L2 and cost the gate backward 0.7 ms at S = 32K on Rubin), the
  dO quantize reduces the first and publishes ``amax_do``, and the EPILOGUE launch
  runs the fixed-order ``dW_norm`` reduce (one column per block -- the same fp32
  chain as the standalone reduce) and the dqkvg quantize, every block reducing
  ``amax_dqkvg`` from the dG and band partials (order-free: bitwise the pass it
  replaces).  Every fusion is BITWISE the unfused chain (pinned by the quantized
  backward's suite); the launch count is the table below.

Packed sequences (``thd=True``) are served over the packed per-tensor fp8 training
record as written: every stage but the SDPA row is token-wise at ``B = 1, S = T``
(the prologue's TMA rebuild indexes tokens; the gate backward's delta at ``s = T``
IS the packed head-major ``[1, H_q, ceil128(T)]`` delta the packed chain reads), and
the fp8 SDPA row runs its THD chain over the envelope ``(num_sequences, max_seq_len)``
with that delta as its external one -- no ``dot`` pre-pass over the e4m3 payloads (one
rounding of dO, the delta contract above).  The MXFP8 backward stays dense (below).

``grad_scaling`` is a DECLARATION ATTRIBUTE, not a knob: it moves the e4m3 points
the gradients are rounded at (a knob is performance-only -- the same function
under any value).  ``FP8_SCALE_S_LOG2`` / ``FP8_GRAD_SCALE_MARGIN_LOG2`` are module
constants for the same reason.  Declined (typed, at declaration, naming the
attribute; an ``MxQuantSpec`` is no decline -- it selects the MXFP8 backward,
below): e5m2 codes, an fp16 ``dy`` (the quantized backward is bf16), the record's
``h`` or the weights in the wrong dtype BOTH ways, and a geometry whose Q / K
rebuild only the LDG norm + RoPE kernel can tile (the fused prologue runs the TMA
kernel, whose ``tile_rows`` must divide ``h_q``, be a multiple of ``h_kv`` and of
its 4 warps -- nothing in 1..16 does for ``h_q = 20`` MHA or ``h_q = 6`` over
``h_kv = 2``; declined by the prologue stage at ``check_support``, naming the LDG
kernel, while the bf16 backward serves such a geometry through it).  There is NO
``B*S`` rule: the two
weight-gradient GEMMs contract over the token axis with MN-major e4m3 operands
(an M-major A, an N-major B), and the TMA 16-byte contiguous-extent rule binds
an operand's CONTIGUOUS axis only, so a ragged token count (S = 1000 at B = 1)
is served with its weight gradients -- the K tail is TMA zero-fill, as for the
bf16 twins.  Every bf16 decline is unchanged.

**The MXFP8 backward: ``quant=MxQuantSpec``.**  The MXFP8 TRAINING forward
(``GatedAttentionBlockFwd(quant=MxQuantSpec, sample_h_sf=, sample_w_qkvg_sf=,
save_for_backward=True)``) writes the SAME bf16 record as the bf16 and the fp8
forwards -- a bf16 ``proj_slab`` with PRE-norm Q / K bands, the bf16 pre-gate
``o``, the exact fp32 ``lse``, ``rstd_*`` -- with ``saved.h`` the caller's e4m3
codes (``h_sf`` is a forward input, never a record field).  Declared with the
forward's own ``MxQuantSpec`` (``descale_w_o``, ``scale_o``: the per-tensor pair of
the out projection -- the one per-tensor side of that pipeline), the backward
differentiates that record AS WRITTEN -- e4m3 ``saved.h`` and weights, a bf16
``dy`` -- and returns bf16 ``dh`` / ``dW_qkvg`` / ``dW_o`` and fp32 ``dW_*_norm``.
What differs from the per-tensor fp8 chain (:meth:`GatedAttentionBlockBwd._execute_mxfp8`
is the launch order):

* **one per-tensor gradient, dY** (``dy8 [T, d_model]`` at ``scale_dy``: the
  out-projection side is per-tensor fp8, as in the forward), its amax a standalone
  per-CTA-partials launch, its scale ``grad_scaling``'s -- ``"current"`` derived on
  device, ``"delayed"`` the caller's ``execute(scale_dy=)``; ``scale_do`` /
  ``scale_dqkvg`` are REFUSED (dO and dQKVG are block-scaled: their 32-element
  blocks carry their own E8M0 scales) and so is ``scale_dp`` (the MXFP8 SDPA row
  has no dP scalar and no amax);
* **the SDPA operands are MXFP8 block quantizations of bf16 values**, in the
  SDPA's own scale-factor layouts and with the standalone quantize kernel's exact
  arm (``abs_max_tree -> e8m0 -> x * rcp -> fp32_to_fp8_pack``): dO ROWWISE
  (``do8`` + ``sf_do``, the row's dP operand) AND COLUMNWISE (``do_T8`` + ``sf_do_T``,
  its dV operand) by ONE dual-axis launch over the gate backward's bf16 dO (one
  read, both quantizations: the slow 2-byte-store columnwise arm is gone from the
  chain); Q and K rowwise AND columnwise (``q8 / q_T8``, ``k8 / k_T8``) by the fused
  PROLOGUE's rebuild job -- the TMA norm + RoPE body on a ONE-head x 32-TOKEN tile,
  which holds a block of BOTH axes, rounding to bf16 first and quantizing out of
  registers, so no bf16 ``recompute`` / ``recompute_k`` buffer is written or carved
  (the fp8 chain's one-token x 16-head tile holds no 32-token block; the MX arm's
  tile does); V ROWWISE straight from the slab's V band (``v8`` + ``sf_v``, the
  prologue's last job: the backward's dP operand is rowwise, so the forward's
  columnwise ``v8`` cannot serve).  Every payload and blob is bitwise the standalone
  quantize of the bf16 values, and ``q8 / sf_q`` / ``k8 / sf_k`` are bitwise the
  forward's own workspace bytes;
* **the SDPA backward is the MXFP8 row** (``SdpaBwdDslSm107Mxfp8``, external
  delta, its block-scaled dS chain) over those payloads and blobs, the record's
  ``lse`` and the gate backward's delta; its dead half-precision ports (``o_f16`` =
  ``saved.o``, ``dO_f16`` = the bf16 dO) bound as the ABI requires and read by
  nothing.  Under GQA the row folds its per-Q-head dK partials in fp32 and rounds
  the sum once, like the reference, while its per-Q-head dV partials are bf16
  (the kernel stores them from its epilogue; fp32 ones do not fit its 327 KiB
  shared-memory budget), so dV carries one bf16 rounding per group member where
  a once-rounded reference carries one in total (relative RMS about 3e-3 at a
  group of 4, the geometry the tests run, measured on the per-tensor fp8 row
  before it moved to fp32 partials); the modelled oracle folds dV the same way
  and the distance to a once-rounded fold is reported per cell.  On the
  block-scale arm the row launches its dQ GEMM once per head chunk under GQA,
  like the plain renderings (its dQ record takes ``b_head_group`` = the group,
  so B and its scale factors are indexed by ``h // group``; bitwise the
  per-member launches it replaced);
* **the two projection GEMMs are block-scale GEMMs over TRANSPOSED operands**,
  the E8M0 dequant exact in the MMA (no alpha): ``dW_qkvg = dQKVG^T . h^T`` reads
  the transposed MXFP8 quantization of dQKVG (``dqkvg_t8 [N, T]``, 32-token blocks
  along T, with ``sf_dqkvg_t`` in the GEMM's canonical F8_128x4 order) and the
  CALLER's ``h_t`` -- ``h`` re-quantized along tokens, e4m3 ``[d_model, T]``
  contiguous -- with its blob ``h_t_sf``; ``dh = dQKVG . W_qkvg^T`` reads the rowwise
  quantization (``dqkvg8 [T, N]`` + ``sf_dqkvg``) and the caller's ``w_qkvg_t`` --
  ``W_qkvg`` re-quantized along N, e4m3 ``[d_model, N]`` -- with ``w_qkvg_t_sf``.  The
  four artifacts are ``execute`` keywords (``h_t`` / ``h_t_sf`` REQUIRED iff
  ``need_dw_qkvg``, ``w_qkvg_t`` / ``w_qkvg_t_sf`` iff ``need_dh``; each refused when
  not read), checked on the host before any launch: dtype, shape, the contiguous
  K-major storage (a ``.t()`` view of the un-transposed codes is refused by name),
  16-B alignment, and the blobs' padded byte count.  The scale-factor blob of a
  transposed artifact is sized by ``sf_blob_bytes(rows, k) = ceil128(rows) x
  ceil128(k) / 32``, which is the same number for ``(rows, k)`` and ``(k, rows)``: the
  byte count does not validate the blob's orientation.  A blob built over the
  un-transposed matrix (the forward's ``h_sf`` handed as ``h_t_sf``) passes every
  host check and produces a wrong weight gradient; build it over the transposed
  matrix exactly as the artifact it scales, and verify a new caller against the
  reference once.  The weight-gradient GEMM contracts over ``T = B*S`` through one
  E8M0 scale per 32-element K block, and the transposed quantize writes whole
  32-token blocks, so **``B*S % 32 == 0`` is required exactly when
  ``need_dw_qkvg``** (typed at declaration, with the fixes named); the data
  gradients contract over ``d_model`` / ``n_qkvg`` and are served at any ``T``;
* **the out projection stays per-tensor fp8**: B2 / B1 are the fp8 chain's e4m3
  GEMMs (``alpha_b2 = descale_dy * descale_w_o``, ``alpha_b1 = descale_dy / scale_o``)
  over ``dy8`` and the gate backward's e4m3 ``og8`` at ``scale_o``;
* **the scalar block is the SAME 29-slot tuple** (``QUANT_SCALAR_SLOTS``), written by
  the fused PROLOGUE's init job (the ``descale_dp``-less arm: no dP scalar) --
  eight slots live (``amax_dy / scale_dy / descale_dy / alpha_b1 / alpha_b2`` from the
  dY quantize, the constants ``scale_o / descale_o / descale_w_o`` from the launch's
  arguments), the 21 others exactly 0.0 (never 1.0: a misrouted read of a dead slot
  zeroes an output the finite checks catch); readable through
  :meth:`GatedAttentionBlockBwd.quant_scalars`;
* **the small launches are fused as on the fp8 chain -- 10 block launches with
  every gradient** (the table below; 20 before): the PROLOGUE
  (``kernels/mxfp8_bwd_fused.py``: scalar init | dY amax partials | the Q / K
  rebuild's MX epilogue writing ``q8 / q_T8 / k8 / k_T8`` with their blobs | ``v8``),
  the dY cast, B2, B3, ONE dual-axis dO launch (``do8`` + ``do_T8`` from one read),
  B1, the row, B5+B6, the EPILOGUE (the ``dW_norm`` reduce | the dual-axis dQKVG
  cast: ``dqkvg8`` + the transposed ``dqkvg_t8`` from one read), B7, B8 -- every
  payload, blob, scalar and gradient BITWISE the unfused chain's (the suite's
  bitwise layer against the standalone quantize of every source, and a dump of
  every intermediate on a 0xFF-poisoned workspace against the unfused tree).  The
  workspace carve follows the prologue's ARM, a plan-time fact
  (:attr:`GatedAttentionBlockBwd.mx_prologue_arm`): under the MX-epilogue arm the
  bf16 ``recompute`` / ``recompute_k`` regions are not carved; the alternative arm
  that keeps the bf16 TMA store (and quantizes ``q_T / k_T`` from the bf16 buffers
  by a dual-axis launch) carves them, so taking it never re-opens the carve.
  Measured on Rubin cc 10.7 (212 SMs, locked clocks, CUPTI device time, the 397B
  geometry at S = 8K): the PROLOGUE 0.099 ms against the eight launches it
  replaces at 0.184 (+85 %), the dual-axis dO launch 0.042 against 0.116 (+178 %),
  the EPILOGUE 0.088 against 0.122 (+39 %); the PROLOGUE reads its own bytes at
  3.7 TB/s against the per-tensor fp8 prologue's 6.3 on its (the two-pass
  32-token tile runs 5-6 CTAs per SM against the shipped tile's 14) -- the
  honest number behind the fusion, stated so the alternative arm is a measured
  choice, not a guess;
* **measured** on two Rubin parts (cc 10.7; a 204-SM part and a 212-SM part -- two
  datasets of one tree, torch's Philox draws following the SM count; the accept suite's
  module docstring carries every cell of both; the figures below are the 204-SM dataset's,
  the 212-SM one held every gate with its own margins): every stage inside the bound calibrated for it -- the GEMM-side
  bounds at 0.15-0.23 of theirs (the gate-composite dO, B1, B7, B8 on the block's own
  e4m3 operands, B7 / B8 through the caller's blobs), the SDPA stage bitwise the
  row's own pre-pass and at most 0.272 / 0.279 / 0.234 of the bf16 bound form against
  the fold-modelled reference (dQ / dK / dV), dK within 1.3e-4 relative RMS of the
  once-rounded reference (the fp32 partials) and dV within 1.1e-4 of the fold-modelled
  one (2.6e-3..2.8e-3 from a once-rounded fold: the bf16 partials above), the seeded
  ``dh`` / ``dW_o`` at 0.774 / 0.532 of the bf16 block's bound and ``dW_qkvg`` inside its
  ``1e-5 x rows x keys`` row budget with every outside row a near-amax ``dqkvg_t8``
  code flip (at most 20 of 8192 rows against budgets of 13.1-103); end to end, the
  fold-modelled oracle inside that row budget on every output of every cell (cos >=
  0.999991), the once-rounded one over it on five GQA outputs (the dV fold's bf16
  partials), the unquantized one on every cell (cos >= 0.9983).  The launch census
  and the workspace delta are below and in :meth:`GatedAttentionBlockBwd.get_workspace_size`.

Declined (typed, naming the attribute) on top of the fp8 arm's: an e5m2 ``dtype``,
``thd=True`` with an MxQuantSpec (dense-only: the backward's SDPA-layout MX quantize
stages run the quantizer's dense arm only -- its packed per-sequence scale-factor arm is
not wired into them yet, while the packed MXFP8 training record the forward writes and
the packed head-major delta both exist; the packed MXFP8 backward is a follow-up),
``B*S % 32 != 0`` when a projection weight
gradient is requested, ``scale_dp`` / ``scale_do`` / ``scale_dqkvg`` at ``execute``, an
artifact given without its need or a need without its artifact, a ``.t()``-view
artifact, a wrong blob byte count or dtype, an artifact in the wrong dtype for its
weight's mode (an e4m3 ``w_qkvg_t`` under an e2m1 ``W_qkvg`` and the reverse; uint8 bytes
of packed codes get the ``.view(torch.float4_e2m1fn_x2)`` hint), a LOGICAL ``[rows, K]``
fp4 artifact (twice the packed data), ``w_o_t`` / ``w_o_t_sf`` without ``o_fp4``, the OTHER
fp4 format's ``w_o_t_sf`` (its byte count differs by the block ratio).  Every bf16
decline is unchanged.  Nothing here changes the MXFP8 SDPA row's capabilities: the
block binds the row's dense plan with an external delta.

**The fp4 weight modes: ``quant=MxQuantSpec(w_qkvg_dtype=torch.float4_e2m1fn_x2)`` and / or
``MxQuantSpec(o_fp4=Fp4Format.NVFP4 | MXFP4)``.**  The MXFP8 training forward's two fp4 modes
write the SAME record as the MXFP8 forward, byte for byte (the fp4 tail only replaces the
workspace's per-tensor ``o8`` by the e2m1 ``o4`` + ``sf_o``; the SDPA writes bf16 ``o``), so
this backward is the MXFP8 backward with the two DATA-gradient GEMMs on the FROST
block-scale catalog's fp4 rows -- the forward's own renderings at the dgrad's shapes --
over the caller's TRANSPOSED e2m1 artifacts; the weight gradients stay 8-bit (``h`` is
e4m3 in every fp4 mode, B1 per-tensor e4m3, B7 the MXFP8 block-scale row) and nothing
else moves:

* **an MXFP4 ``W_qkvg``** puts B8 ``dh = dQKVG8 . W_qkvg^T`` on the MIXED row (e4m3 A x e2m1
  W, one E8M0 scale per 32 on both sides): the caller's ``w_qkvg_t`` is then the packed
  e2m1 ``[d_model, N // 2]`` (``torch.float4_e2m1fn_x2``, two codes per byte along N, low
  nibble = even n -- the SAME ``execute`` keyword, its dtype keyed on
  ``MxQuantSpec.w_qkvg_dtype``) with the UNCHANGED E8M0 / 32 blob ``w_qkvg_t_sf``
  (``sf_blob_bytes(d_model, N)``).  Nothing else changes: the MXFP8 launch census (15 at
  the test geometry, 15 RoPE-only, 14 MHA -- the table below), the same carve;
* **an fp4 ``W_o``** (``o_fp4``; ``scale_o == descale_w_o == 1.0`` by ``MxQuantSpec``'s own rule)
  puts B2 ``dO_gated = dY . W_o^T`` on a block-scale row over the caller's
  ``w_o_t`` -- ``W_o`` re-quantized along ``d_model`` in the format of ``o_fp4``, packed e2m1
  ``[H_q*D, d_model // 2]`` with its blob ``w_o_t_sf`` (``sf_blob_bytes(H_q*D, d_model, block)``:
  e4m3 scales per 16 under NVFP4, E8M0 per 32 under MXFP4) -- REQUIRED at ``execute`` under
  ``o_fp4`` whatever the ``need_*`` set (the gate backward needs ``dO_gated``), refused
  without; and it needs a BLOCK quantization of dY as B2's A operand, ONE more launch
  right after the per-tensor dY quantize published ``scale_dy``:

  - ``Fp4Format.MXFP4``: the MIXED row again -- dY block-quantized ROWWISE to e4m3
    (``dy_mx8 [T, d_model]``, 32-element blocks along ``d_model``) with its GEMM-canonical E8M0
    blob (``sf_dy_mx``, ``sf_blob_bytes(T, d_model)``: the quantizer's canonical mode, the
    dQKVG quantize's), the gradient stays 8-bit;
  - ``Fp4Format.NVFP4``: the NVFP4 x NVFP4 row (no e4m3 x e2m1 row with e4m3 scales exists in
    the catalog), so dY itself is cast to NVFP4 (``dy4 [T, d_model // 2]`` packed e2m1 + the
    canonical e4m3-per-16 blob ``sf_dy4``, ``sf_blob_bytes(T, d_model, 16)``) by the forward's
    fp4 quantize kernel -- a 4-bit GRADIENT operand, the NVFP4 training recipe's own
    choice -- **as a TWO-LEVEL cast**: neither fp4 format carries a per-tensor scale, and an
    NVFP4 block's e4m3 scale is ``max(amax / 6, 2^-9)``, so a 16-block whose amax sits below
    the e2m1 midpoint ``2^-11`` would quantize to ALL ZEROS (right for the forward's O(1)
    gated output, wrong for a raw output gradient at 1e-4 .. 1e-6).  The kernel therefore
    quantizes ``scale_dy x dY`` -- the live power-of-two per-tensor scale the dY quantize
    published one launch earlier, read from its slot in-kernel (the quantize kernel's
    appended pre-scale read; no new slot) -- which lifts the tensor's amax into ``[224,
    448]`` so a block is zeroed only when its amax sits more than ~19 octaves below the
    tensor's; B2's output is then ``scale_dy x dO_gated``, and the gate backward's dY
    descale arm multiplies it by ``descale_dy`` (its slot) before every use -- ``dO``,
    ``dG``, the delta; ``og8`` is cast from ``O`` and untouched -- exact for a power of two.
    Wherever the single-level scale byte was a normal e4m3 value the codes are identical
    (a power of two only shifts the e4m3 exponent), so the pre-scale is purely a floor
    remedy; a power-of-two scaling of dY leaves the whole chain bitwise equivariant (the
    suite's ``2^-13`` pin), which the single-level cast fails on its first assertion.

  Under either format ``alpha_b2`` is still published by the dY quantize (one fp32
  multiply) and read by no GEMM (B2 has no alpha epilogue), B1 reads ``alpha_b1 =
  descale_dy`` (``descale_o = 1``), and ``dy8`` stays (B1's A).
* The oracle's dgrad path dequantizes the transposed e2m1 artifacts THROUGH their blobs
  (a wrong caller blob is a wrong oracle, never a silent agreement), never the forward's
  row-quantized weight -- two fake-quants of one master weight along its two axes, the
  fp4 training recipe's straight-through estimator -- and under an NVFP4 ``W_o`` takes the
  SAME two-level point (``fq_nvfp4(bf16 dY, global_scale=scale_dy)``, descaled downstream);
  the accept suite is ``test_block_backward_fp4.py`` (the five configurations x the cells).

Launch table (one stream -- under ``fuse_wgrad_overlap`` rows 3 and 9 are
issued on the block's side stream, forked and joined as above, the same
launches; ``g = h_q / h_kv``, ``c`` = the adapter's head
chunks, ``q`` = the adapter's dQ GEMM launches per chunk -- 1 under its
single-launch dQ rendering (``MatmulTemplateParams.b_head_group = g``, the
shipped default), ``g`` under the per-group-member twin; all ``need_*`` True)::

    #   stage                           launches
    1   B2  run_dgrad_gemm              1
    2   B3  sigmoid_gate_bwd            1            (+ delta = rowsum(dO * O) as a 4th output under fuse_gate_bwd)
    3   B1  run_wgrad_gemm              1            (need_dw_o)
    4   Q/K recompute (_QkNormRope)     1
    5   V compaction (_VCompaction)     1
    6   B4  SdpaBwdDslSm107.execute     3 + c*(2+q)  fill_i32 + dot_do_o + c x [main + dK GEMM + q x dQ GEMM] + dkv_reduce (g > 1)
                                          + 3 at S % 128 != 0 (q / dO / lse pads), + 2 at S % 256 != 0 (k / v pads),
                                          + 2 fold copy-outs when GQA and kv-padded (+1 when MHA and kv-padded);
                                          MHA (g = 1): no dkv_reduce -> 2 + 3c;
                                          fuse_gate_bwd: no dot_do_o (external_delta) -> 2 + c*(2+q)
                                          thd: the packed chain -- 2 + c*(2 + 2*(1+q)) + dkv_reduce (g > 1): setup + dot_do_o + c x [the main kernel's own setup + main + (descriptor patch + GEMM) x (1 + q)]
                                          thd + fuse_gate_bwd: 1 + c*(2 + 2*(1+q)) + dkv_reduce (g > 1) -- no dot_do_o (the gate backward's delta at s = T IS the packed one)
                                          (MEASURED: 17 kernels at the test geometry, three sequences -- the 18 below less dot_do_o)
                                          thd + quant (the fp8 row): 2 + [zero-fill] + c*(2 + 2*(1+q)) + 1 -- the THD metadata setup + the amax
                                          resets (two launches: no kv-length fill under THD, the resets stay their own) + c x [own setup + main +
                                          (patch + dK) + q x (patch + dQ)] + the fold launch (dV, and dK under GQA: every group), no dot (the
                                          delta is the gate backward's); the quantized backward's own 10 launches precede it
                                          (MEASURED: 19 kernels for the whole fp8 backward at the test geometry, three sequences, GQA 8/2)
                                          [+ 1 zero-fill on the untrimmed / wide-tile twins only]; no pads, no fold copy-outs
                                          (MEASURED: 18 kernels for the whole backward at the test geometry, three sequences)
    7   B5+B6 qk_norm_rope_bwd          1
    8   dW_norm reduce                  1            (need_dw_norms)
    9   B7  run_wgrad_gemm              1            (need_dw_qkvg)
    10  B8  run_dgrad_gemm              1            (need_dh)
                                       ---
                                        12 + c*(2+q)  -- 15 at the test geometry (h_q=8, h_kv=2, q=1; 22 at S=1000), 15 / 18 at 397B (c=1 / 2)
                                        11 + c*(2+q)  under fuse_gate_bwd -- 14 / 21 at the test geometry, 14 / 17 at 397B

The count is CHECKED by CUPTI in the tests (``test_launch_count_is_honest``:
15 / 22 at the test geometry, 14 / 21 with ``fuse_gate_bwd``, MEASURED on Rubin
cc 10.7), never quoted from this formula: the padded / zero-fill / MHA arms
change it, and ``q`` is read off the adapter's dQ record
(``prepared_host._dq_launches``), never assumed.

Launch table of the QUANTIZED backward (``quant=QuantSpec``; one stream, the
same ``fuse_wgrad_overlap`` treatment of rows 6 and 10; ``c`` / ``q`` as above
for the fp8 row; the two FUSED launches dispatch their jobs by block range)::

    #     stage                            launches
    1     PROLOGUE (fp8_bwd_fused)         1            init: slots[:] = 0, descale_dp = 1 / scale_dp, the 14 plan-time constants (kernel arguments) |
                                                        dY amax PARTIALS (one per CTA) | Q / K rebuild -> q8 / k8 (e4m3 epilogue at the forward's
                                                        static scales, kernel arguments) | v8 (the slab's V band)
    2     quantize dY                      1            amax_dy = max(partials) published; dy8; scale_dy, descale_dy, alpha_b1, alpha_b2 published
    3     B2  run_dgrad_gemm (e4m3, K64)   1            dO_gated = dy8 @ W_o8 * alpha_b2
    4     B3  sigmoid_gate_bwd (fp8 arm)   1            dO, dG, og8 (need_dw_o), delta; per-CTA PARTIALS of max |dO| and of max |dG| (persistent grid,
                                                        one plain store each -- never an atomic)
    5     quantize dO                      1            amax_do = max(B3's dO partials) published; do8; scale_do, descale_do published
    6     B1  run_wgrad_gemm (e4m3, K64)   1            need_dw_o: dW_o = dy8^T @ og8 * alpha_b1
    7     B4  SdpaBwdDslSm107Fp8.execute   2 + c*(2+q)  setup (fill_i32 + the amax resets, ONE launch) + c x [main + dK GEMM + q x dQ GEMM] + ONE fold
                                                        launch (dV; and dK under g > 1, in the same launch over fp32 partials the fold rounds once)
                                                        (no dot_do_o: external delta; no fold copy-outs: the folds write dv / dk directly)
                                                        + 3 at S % 128 != 0 (q / dO / lse pads), + 2 at S % 256 != 0 (k / v pads) [+ 1 zero-fill on the wide-tile twins]
    8     B5+B6 qk_norm_rope_bwd           1            + per-CTA PARTIALS of max |dQ_pre / dK_pre / dV| (one plain store per CTA)
    9     EPILOGUE (fp8_bwd_fused)         1            dW_norm reduce (need_dw_norms; one column per block) | dqkvg8 with amax_dqkvg = max over B3's dG
                                                        partials and B5+B6's band partials, reduced in every cast block and published;
                                                        scale_dqkvg, descale_dqkvg, alpha_b7, alpha_b8 published
    10    B7  run_wgrad_gemm (e4m3, K64)   1            need_dw_qkvg: dW_qkvg = dqkvg8^T @ h8 * alpha_b7
    11    B8  run_dgrad_gemm (e4m3, K64)   1            need_dh: dh = dqkvg8 @ W_qkvg8 * alpha_b8
                                           ---
                                           12 + c*(2+q)  -- 15 at the test geometry (norm, GQA 8/2, c = q = 1; 15 rope_only -- the epilogue launch
                                                            stays for the quantize --, 15 MHA: the row's dK fold shares the dV fold's launch), the same
                                                            under both grad_scaling recipes; 20 at S = 992 / 1000 (+3 q pads, +2 kv pads); 17 before
                                                            the row merged its setup and fold launches, 24 before the launch fusion

CHECKED by CUPTI in the quantized backward's own suite, never quoted from this
table (``c`` and ``q`` off the adapter, the pads off ``S``).

Launch table of the MXFP8 backward (``quant=MxQuantSpec``; one stream, the same
``fuse_wgrad_overlap`` treatment of rows 6 and 10; ``g = h_q / h_kv``, ``c`` = the
adapter's head chunks, ``q = 1`` dQ launch per chunk -- the row's block-scale arm
launches dQ once per head chunk under GQA too, its dQ record taking
``b_head_group = g``; the two FUSED launches dispatch their jobs by block range)::

    #     stage                                 launches
    1     PROLOGUE (mxfp8_bwd_fused)            1            init: slots[:] = 0, the 14 plan-time constants from kernel arguments (3 live: scale_o,
                                                             descale_o, descale_w_o; 0.0 elsewhere; no descale_dp: no dP scalar) | dY amax PARTIALS
                                                             (one per CTA) | Q / K rebuild, MX epilogue: norm + RoPE from the slab's PRE-norm bands on a
                                                             one-head x 32-token tile -> q8 + sf_q, q_T8 + sf_q_T, k8 + sf_k, k_T8 + sf_k_T (rowwise
                                                             AND columnwise from the same tile; no bf16 recompute buffer) | v8 + sf_v (the slab's V band)
    2     quantize dY                           1            amax_dy = max(partials) published; dy8; scale_dy, descale_dy, alpha_b1, alpha_b2
    2b    quantize dY (block)   [o_fp4 only]    1            MXFP4 W_o: dy_mx8 + sf_dy_mx (MX rowwise, canonical); NVFP4 W_o: dy4 + sf_dy4 (the
                                                             two-level NVFP4 cast of scale_dy x dY, scale_dy read from its slot)
    3     B2  run_dgrad_gemm (e4m3, K64)        1            dO_gated = dy8 @ W_o8 * alpha_b2; under an fp4 W_o the block-scale dgrad over the
                                                             caller's e2m1 w_o_t (dy_mx8 / dy4 as A, no alpha; NVFP4: scale_dy x dO_gated)
    4     B3  sigmoid_gate_bwd (fp8 arm)        1            dO (in place), dG (GATE band), og8 (need_dw_o), delta ALWAYS; no amax partials;
                                                             NVFP4 W_o: x descale_dy before every use (the dY descale arm)
    5     quantize dO, DUAL-AXIS (MXFP8)        1            do8 + sf_do (rowwise, the SDPA's layout) AND do_T8 + sf_do_T (columnwise, D-plane-major)
                                                             from ONE read of the bf16 dO
    6     B1  run_wgrad_gemm (e4m3, K64)        1            need_dw_o: dW_o = dy8^T @ og8 * alpha_b1  (side stream under the knob, forked after 5)
    7     B4  SdpaBwdDslSm107Mxfp8.execute      1 + c*(2+q)  fill_i32 + c x [main + dK GEMM + q x dQ GEMM] + dkv_reduce (g > 1)
                                                + (g > 1)    + 8 at S % 128 != 0 (q / dO / lse / dO_T pads, the sf_q / sf_do / sf_do_T /
                                                             sf_q_T re-stagings), + 7 at S % 256 != 0 under GQA (k / v pads, sf_k / sf_v /
                                                             sf_k_T re-stagings, two fold copy-outs; + 6 MHA: one copy-out), + 4 under a
                                                             dS zero-fill (the first payload, the second, the two atom tensors)
    8     B5+B6 qk_norm_rope_bwd                1            bf16 dqkvg bands, dW partials (no amax fold)
    9     EPILOGUE (mxfp8_bwd_fused, 256 thr.)  1            dW_norm reduce (need_dw_norms; two columns per block, the standalone chain per column) |
                                                             dqkvg DUAL-AXIS cast from ONE read: dqkvg8 [T, N] + sf_dqkvg (need_dh; canonical) AND
                                                             dqkvg_t8 [N, T] + sf_dqkvg_t (need_dw_qkvg; canonical, whole 32-token blocks) -- a half
                                                             folded out when its gradient is not requested; the launch exists iff one of its three jobs does
    10    B7  run_wgrad_gemm_block_scale        1            need_dw_qkvg: dW_qkvg = dqkvg_t8 . h_t^T  (sf_dqkvg_t, the caller's h_t_sf; forked after 9)
    11    B8  run_dgrad_gemm_block_scale        1            need_dh: dh = dqkvg8 . w_qkvg_t^T          (sf_dqkvg, the caller's w_qkvg_t_sf;
                                                             an MXFP4 W_qkvg: the mixed row over the packed e2m1 w_qkvg_t, the same launch)
                                               ---
                                                10 + 1 + c*(2+q) + (g > 1)  -- the table's arithmetic: 15 at the test geometry (norm, GQA 8/2:
                                                c = 1, q = 1), 15 rope_only (the epilogue stays for the cast), 14 MHA (q = 1, no dkv_reduce),
                                                15 / 18 at the 397B geometry (g = 16, c = 1 / 2) -- the fp8 chain's count; each omitted gradient
                                                drops ITS rows (need_dw_qkvg=False: row 10 and the epilogue's transposed half; need_dw_o=False:
                                                row 6; need_dw_norms=False: the epilogue's reduce job -- the launch stays while a cast half
                                                remains); + 1 under an fp4 W_o (row 2b) -- 16 / 16 / 15 at the test geometry; an MXFP4 W_qkvg
                                                alone adds nothing.  The unfused chain ran 20 block launches: 28 / 27 / 24 and 40 / 58 on the
                                                same cells over the row's per-member dQ (q = g), 25 / 24 / 24 and 25 / 28 once the row's dQ GEMM
                                                ran once per head chunk; the fused chain over the per-member dQ 18 / 18 / 14 and 30 / 48

MEASURED by CUPTI on Rubin (cc 10.7, 204 SMs) in the MXFP8 backward's own suite (its launch
census, ``test_mxfp8_launch_count_is_honest``: ``len(kernels) == formula-from-facts == expected``,
0 memsets and 0 memcpys on every census cell): 15 at the test geometry (S = 512, B = 2, GQA 8/2,
norm), 15 RoPE-only, 14 at both MHA cells (S = 512 causal and S = 1024 dense), 30 at the two
q- and kv-padded GQA cells with weight gradients (S = 992 at B = 1, S = 1008 at B = 2: ``+ 8 + 7``),
29 at the padded dgrad-only cell (S = 1000: row 10 gone, the epilogue keeps its rowwise half, ``+ 8 + 7``),
28 at the padded MHA cell (S = 992: ``+ 8 + 6``), 22 at the kv-side-only padded cell (S = 384: ``+ 7``),
and 15 at the 397B geometry (B = 1, S = 512, causal, norm, GQA 32/2, ``c = 1``, ``g = 16``: its own
census cell).  Before the two changes the same cells measured 28 / 27 / 24 / 24 / 43 / 43 / 41 / 38 / 35
and 40 (the unfused chain over the row's per-member dQ), 25 / 24 / 24 / 24 / 40 / 40 / 38 / 38 / 32 and
25 (unfused, the single-launch dQ), 18 / 18 / 14 / 14 / 33 / 33 / 32 / 28 / 25 and 30 (fused, per-member dQ).  The
suite's expectation is COMPUTED from the block's rows by the cell's needs plus the row's terms
read off the adapter (``c``, ``q``, the pads, the zero-fill), never typed -- and never quoted
from this table.

Workspace table (``WorkspaceLayout(align=256)``, ``e`` = activation bytes,
``T = B*S`` -- the packed token total under ``thd``, the same carve at
``B = 1, S = T``; every region is a slot of the CALLER's one uint8 buffer)::

    region            shape             dtype   writer                      reader
    do_gated (= do)   [T, H_q, D]       act     B2; then B3 in place         B3; B4 (as dO)
    dqkvg             [T, N]            act     B3 (GATE), B5+B6 (Q, K, V)   B7, B8
    o_gated           [T, H_q, D]       act     B3 (og)          need_dw_o  B1
    recompute (q)     [T, H_q, D]       act     _QkNormRope                  B4          (bf16 / fp16 only: under quant the rebuild writes q8 / k8)
    recompute_k       [T, H_kv, D]      act     _QkNormRope                  B4          (idem)
    recompute_v       [T, H_kv, D]      act     _VCompaction                 B4          (idem: v8 is V's compaction)
    dq / dk / dv      compact           act     B4                           B5+B6
    dw_partials_q/k   [n_ctas_x, D]     fp32    B5+B6         need_dw_norms  the reduce   (EXACTLY n_ctas_for(recipe, T) rows)
    sdpa_bwd_ws       opaque            uint8   the adapter's own carver (delta -- unless fuse_gate_bwd --, ONE dS chunk, pads, GQA partials;
                                                thd: the packed delta (again unless fuse_gate_bwd), ONE kv-BLOCKED dS chunk qh_chunk x ceil256(T + 256 B) x ceil128(S_max),
                                                its metadata / descriptor words, the GQA partials [1, T, H_q, D] x2 -- no pads)
    gemm_scratch      opaque            uint8   the FROST GEMM (max(plan.workspace_bytes) over B1 / B2 / B7 / B8, never 0)
    delta             [B, H_q, S_pad]   fp32    B3 (4th output)  fuse_gate_bwd  B4 (external_delta; S_pad = the adapter's external_delta_shape;
                                                                 thd: the packed [1, H_q, ceil128(T)])
                                                                 -- ALWAYS under quant (the fp8 row's external delta)
    gemm_scratch_side opaque            uint8   the side-stream wgrad GEMMs (B1 / B7) under fuse_wgrad_overlap: max(plan.workspace_bytes) over them, never 0
    -- quant=QuantSpec only (appended AFTER every region above; o_gated, recompute, recompute_k and recompute_v are then NOT carved) --
    dy8               [T, d_model]      e4m3    the dY quantize              B1 (A, M-major), B2 (A, K-major)
    do8               [T, H_q, D]       e4m3    the dO quantize              B4 (dO); the adapter's dead o when og8 is absent
    og8               [T, H_q, D]       e4m3    B3's fp8 arm     need_dw_o  B1 (B, N-major); the adapter's dead o
    q8 / k8 / v8      compact           e4m3    the three static-scale quantizes (v8 straight from the slab's V band)   B4
    dqkvg8            [T, N]            e4m3    the dqkvg quantize           B7 (A, M-major), B8 (A, K-major)
    quant_scalars     QUANT_SCALARS_BYTES fp32  the prologue's init job (the zeroing, descale_dp, the plan-time   the GEMM alphas and their
                                                constants), the quantize publishes (every amax reduced from       factors, scale_o, the row's
                                                partials), the row's amax_dP                                       eleven slot scalars, quant_scalars()
                                                (slot i at + QUANT_SCALAR_STRIDE * i; QUANT_SCALAR_SLOTS names them; the tail QUANT_CONST_SLOTS holds the constants)
    amax_partials     [amax_partials_n] fp32    the prologue's dY amax job (one word per CTA, written unconditionally; the dY quantize (reduced in every
                                                SMs x 8 cap, n_partials written per execute)                                 CTA's prologue, published to amax_dy)
    amax_partials_do  [gate_partials_n] fp32    B3: per-CTA max |dO| (one word per CTA of its persistent grid,                the dO quantize (reduced, published
                                                SMs x 8 cap, n written per execute)                                          to amax_do)
    amax_partials_dg  [gate_partials_n] fp32    B3: per-CTA max |dG| (the GATE band's half of the dqkvg amax)                 the EPILOGUE's cast blocks
    amax_partials_bands [band_partials_n] fp32  B5+B6: per-CTA max over the Q / K / V bands it stored (EXACTLY               the EPILOGUE's cast blocks (reduced
                                                sum(n_ctas_for(recipe, T)) words, one per CTA of its grid)                   with the dG partials, published to
                                                                                                                              amax_dqkvg)
    -- quant=MxQuantSpec only (appended AFTER every bf16 region; o_gated / recompute_v and the dO / dG / band partials are NOT carved, and
       neither are recompute / recompute_k under the fused prologue's MX-epilogue arm (mx_prologue_arm="mx_epilogue": the rebuild job writes
       the four Q / K payloads out of registers) -- the bf16-rebuild arm carves both; every blob uint8, SDPA-layout ones
       _sf_slot_bytes(B, H, S, D) = D / 32 bytes per row, canonical ones kernels.proj_gemm.sf_blob_bytes) --
    dy8               [T, d_model]      e4m3    the dY quantize              B1 (A, M-major), B2 (A, K-major)
    do8 + sf_do       [T, H_q, D]       e4m3    the rowwise dO quantize      B4 (dO, sf_do)
    do_T8 + sf_do_T   [T, H_q, D]       e4m3    the columnwise dO quantize   B4 (dO_T, sf_do_T)
    og8               [T, H_q, D]       e4m3    B3's fp8 arm     need_dw_o  B1 (B, N-major)
    q8 + sf_q         compact           e4m3    the rowwise q quantize       B4             (bitwise the forward's own q8 / sf_q bytes)
    q_T8 + sf_q_T     compact           e4m3    the columnwise q quantize    B4
    k8 + sf_k         compact           e4m3    the rowwise k quantize       B4             (bitwise the forward's)
    k_T8 + sf_k_T     compact           e4m3    the columnwise k quantize    B4
    v8 + sf_v         compact           e4m3    the ROWWISE v quantize       B4             (the slab's V band; the forward's v8 is columnwise)
    dqkvg8 + sf_dqkvg [T, N]            e4m3    the rowwise dqkvg quantize   B8 (A, K-major; sf canonical over (T, N))
    dqkvg_t8 + sf_dqkvg_t [N, T]        e4m3    the transposed dqkvg quantize  need_dw_qkvg  B7 (A, K-major; sf canonical over (N, T))
    quant_scalars     QUANT_SCALARS_BYTES fp32  the PROLOGUE's init job (the zeroing, the 3 live constants), the dY quantize's publishes   quant_scalars()
    amax_partials     [amax_partials_n] fp32    the PROLOGUE's dY amax job (one word per CTA; SMs x 8 cap)                  the dY quantize
    -- MxQuantSpec.o_fp4 only (appended LAST: every MXFP8 offset above is unchanged) --
    dy_mx8 + sf_dy_mx [T, d_model]      e4m3    the MX-rowwise dY quantize (MXFP4 W_o; sf canonical over (T, d_model))      B2 (A, K-major)
    dy4 + sf_dy4      [T, d_model / 2]  e2m1    the two-level NVFP4 cast of scale_dy x dY (NVFP4 W_o; e4m3 scales per 16,  B2 (A, K-major)
                                                sf canonical over (T, d_model))

The adapter's dS chunk dominates at scale: ``qh_chunk x S_q_pad x S_kv_pad x 2 B``
with ``qh_chunk`` a multiple of the GQA group -- 4.25 / 8.50 / 33.0 GiB at
S = 8K / 16K / 32K for the 397B geometry at B=1 (MEASURED by constructing the
adapter). ``gemm_scratch`` is the max with the backend's OWN heuristic plan's
number, which at the test geometry auto-splits 3-way (12 MiB) although the
forced JIT that runs needs none; 0 at 397B.

One engine, no fallback (AGENTS.md Rule 9, stated as a design decision):
the block binds ``SdpaBwdDslSm107`` directly, exactly as the forward binds its
FROST SDPA class, so it declines every non-Rubin device with a typed message
and never falls back to the cuDNN backend's d=256 backward engine. The graph
route (``manifest.py`` selection) is a later option if a second arch needs it.

P0 limits (all typed, at declaration -- ``check_support``): bf16 / fp16, and
per-tensor fp8 over the fp8 training record (``quant=QuantSpec``: bf16
activations and gradients, dense or packed (``thd=True``), any ``B*S``), and MXFP8 over the MXFP8
training record (``quant=MxQuantSpec``: bf16 activations and gradients, dense
only, ``B*S % 32 == 0`` when a projection weight gradient is requested, the
caller's transposed artifacts at ``execute`` -- e2m1 ones under the fp4 weight
modes, ``w_o_t`` / ``w_o_t_sf`` under ``o_fp4``);
Rubin (SM107); ``d_head = 256``; ``seq_len >= 2`` (S_q = 1 is decode, out of
the ``sdpa_bwd_sm107`` prefill bodies' scope; under ``thd`` the bound is
``2 <= max_seq_len <= T``); ``d_model % 256 == 0`` (the
forced 256-wide GEMM tile behind the determinism contract below; ``h_q * D``
satisfies it through ``d_head = 256``); ``saved.proj_slab`` present (the
gate-copy record -- V is a slab band with no field of its own -- is a follow-up PR);
no dense PADDING (``seq_lens`` as a per-batch KV padding mask: the
``sdpa_bwd_sm107`` row declines ``seq_kv_lens_present``; a follow-up PR flips
it) -- declined from ``seq_lens_present``, from a dense sample record's
``seq_lens`` at declaration AND from every dense record handed to ``execute``
(the tensor's presence is the fact, never its values) -- while PACKED
sequences (``thd=True``, above) are served, ``fuse_gate_bwd`` under ``thd``
included; ``window_left > 0`` only, ``window_right`` unbounded (or 0) only;
``dw_norm_dtype = torch.float32`` only.

Determinism
-----------

No atomics on the bf16 / fp16 chain, so two executes are bitwise equal (pinned on
Rubin by ``test_two_runs_are_bitwise`` and ``test_a_caller_stream_orders_every_stage``,
workspace poisoned in between, on Rubin cc 10.7).  The quantized backward's amax of
every gradient (``amax_dy`` / ``amax_do`` / ``amax_dqkvg``) is a ``max`` over PER-CTA
PARTIALS its producer stored (the prologue's dY job, the gate backward's dO / dG
folds, the norm backward's band folds -- one plain store per CTA, no atomic anywhere
on the block's own kernels) reduced by the consuming quantize launch; ``max`` is
order-free, so the result is bitwise the fp32 max whatever the CTA schedule.  The
only atomic left is the fp8 SDPA row's ``amax_dP`` (an int32 ``atomicMax`` of
non-negative fp32 bit patterns, which order as int32: order-free too).  The scale
derived from an amax, the e4m3 casts and the alpha products are per element, so two
quantized executes are bitwise equal too (the quantized backward's own suite pins it
under every knob set).  The MXFP8 backward has NO atomic anywhere: its one amax (dY)
is the same max over per-CTA partials, its block quantizes are per element, the
MXFP8 SDPA row's chain runs none (its GQA fold is a fixed-order reduce) and the
block-scale GEMMs are deterministic -- two MXFP8 executes are bitwise equal under
every knob set (its own suite pins it).  The four GEMMs run the FORCED
``CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma`` tile at one split-K
slice (no reducer). That premise is ENFORCED, not assumed: ``check_support``
declines a geometry the forced tile cannot take (``d_model % 256 != 0``;
``kernels/proj_gemm.py::_forced_tile_config``), and ``compile()`` refuses,
typed, a plan that fell back to the heuristic (a different tile, route and
possibly a split-K reducer) -- so a served block never runs an unpinned GEMM.
The bf16 / fp16 Rubin backward chain has no atomic and its GQA fold is
a fixed-order reduction; the elementwise kernels are per element; ``dW_norm``
is per-CTA partials plus a fixed-order reduce launch. Bitwise PER KNOB SET and
per device (the SM-fill CTA count enters the partial order) -- never claimed
across devices. This is a contract decision, not an implementation detail.
"""

from __future__ import annotations

import dataclasses
import math
import threading
import weakref
from contextlib import contextmanager
from types import SimpleNamespace
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import torch
from cuda.bindings import driver as cuda

from cudnn._torch_stream import as_torch_stream, stream_context
from cudnn.api_base import APIBase, TensorDesc, TupleDict
from cudnn.frost.workspace import WorkspaceLayout

from .api import (
    _ELEMENTWISE_THREADS,
    _FP4_X2,
    _QK_NORM_ROPE_THREADS,
    _SF_TILE_ROWS,
    _SM107_CC,
    _THD_FORM_PREFIX,
    _WS_ALIGN,
    Fp4Format,
    GatedAttentionBlockFwd,
    GatedAttentionBlockGeometry,
    MXFP8_BLOCK_SIZE,
    MxQuantSpec,
    QuantSpec,
    SavedForBackward,
    _bhsd_desc,
    _check_norm_weights_agree,
    _check_sf_blob,
    _cols,
    _itemsize,
    _QkNormRope,
    _QuantizeFp4,
    _QuantizeMxfp8,
    _Stage,
    _thd_lse_head_stride,
    _thd_seq_lens_form,
    _VCompaction,
    _view,
    saved_slab_views,
)

_ACT_DTYPES = (torch.bfloat16, torch.float16)
_FP8_CODE_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)  # a QUANTIZED training forward's `saved.h`: the caller's fp8 codes

# --- the quantized (per-tensor fp8) backward's constants -- module constants, never knobs (a flip is numerics-changing) ---
# scale_s = 2**FP8_SCALE_S_LOG2 is the fp8 SDPA row's P scale (P <= 1, so 2**8 keeps P * scale_s <= 448 with three bits of
# headroom over the row suite's 32); a flip re-runs the quantized backward's accept matrix.
FP8_SCALE_S_LOG2: int = 8
# The "current" recipe's power-of-two headroom: scale = 2**(floor(log2(448 / amax)) - MARGIN), derived in-kernel from the
# amax the pass read (kernels/quantize.py::grad_scale_from_amax mirrors it bitwise on the host).
FP8_GRAD_SCALE_MARGIN_LOG2: int = 0
# The fp32 SCALAR BLOCK of the quantized backward's workspace: slot i is the fp32 at byte offset
# ws.quant_scalars + QUANT_SCALAR_STRIDE * i.  Zeroed by the prologue's scalar-init job at the top of every execute (the fp8
# row's amax_dP is an atomicMax target and must start from zero; the three gradient amax slots are PUBLISHED by their quantize
# launches from per-CTA partials, so the zero is only what quant_scalars() reads before they land), then written on device
# only; read back with quant_scalars().
# APPEND-ONLY: a new slot goes at the END -- every index below is an ABI the init job, the carve and quant_scalars() share.
# ONE tuple for both quantized arms.  Under an MxQuantSpec EIGHT slots are live -- amax_dy / scale_dy / descale_dy / alpha_b1 / alpha_b2
# (the dY quantize's publishes: dY is that pipeline's one per-tensor gradient) and the constants scale_o / descale_o / descale_w_o --
# and the other 21 read exactly 0.0 after the init launch (no dO / dQKVG / dP scalar: those gradients are block-scaled, the MXFP8
# row has no dP; no per-tensor static scales of the fp8 record).  0.0, never 1.0: a misrouted read of a dead slot zeroes an output.
QUANT_SCALAR_SLOTS: tuple = (
    "amax_dy",  # 0-2   published by the dY / dO / dqkvg quantize launches: max over the producers' per-CTA partials (order-free)
    "amax_do",
    "amax_dqkvg",
    "amax_dp",  # 3     the fp8 SDPA row's amax_dP (its own int32-bit-pattern atomicMax target)
    "scale_dy",  # 4-7   published by the dY / dO quantize launches (scale, descale = 1 / scale)
    "descale_dy",
    "scale_do",
    "descale_do",
    "scale_dqkvg",  # 8-9   published by the dqkvg quantize launch
    "descale_dqkvg",
    "alpha_b1",  # 10-13 the GEMM epilogue products: descale_dY * (1 / scale_o), descale_dY * descale_w_o,
    "alpha_b2",  #       descale_dQKVG * descale_h, descale_dQKVG * descale_w_qkvg
    "alpha_b7",
    "alpha_b8",
    "descale_dp",  # 14    1 / scale_dp, written by the scalar-init launch (one fp32 division on device; exact for a power of two)
    # 15-28 the QuantSpec's PLAN-TIME CONSTANTS (QUANT_CONST_SLOTS), stored by the scalar-init launch on EVERY execute from its
    #       kernel arguments (GatedAttentionBlockBwd._quant_const_values: the values as Python floats).  Slots, never device
    #       tensors filled at compile(): such a fill is enqueued on whatever stream is ambient at compile time, while execute's
    #       launches read it on the caller's stream with nothing ordering the two -- the first execute of a block on a busy
    #       ambient stream could consume the constants before the fills landed (corrupt quantizers, alphas, gradients).
    #       Written by the launch that consumes them, they are stream-ordered by construction and a CUDA-graph replay rewrites them.
    "scale_q",  # 15-18 the static quantizers' scale_q / scale_k / scale_v, and B3's og8 arm's scale_o
    "scale_k",
    "scale_v",
    "scale_o",
    "descale_q",  # 19-22 the fp8 SDPA row's descale_q / k / v and its dead descale_o (the SAME slot is alpha_b1's factor 1 / scale_o)
    "descale_k",
    "descale_v",
    "descale_o",
    "descale_w_o",  # 23-25 the alpha factors: alpha_b2 = descale_dy * descale_w_o, alpha_b7 = descale_dqkvg * descale_h,
    "descale_h",  #        alpha_b8 = descale_dqkvg * descale_w_qkvg
    "descale_w_qkvg",
    "scale_s",  # 26-27 the row's P scale 2**FP8_SCALE_S_LOG2 and its exact reciprocal
    "descale_s",
    "scale_dqkv",  # 28    the row's scale_dQ = scale_dK = scale_dV = 1.0 (bf16 gradients out of the row)
)
# The plan-time constants' slots in KERNEL-ARGUMENT order: the TAIL of QUANT_SCALAR_SLOTS from "scale_q" -- one contiguous range
# the init kernel stores with one range_constexpr (_plan_bwd_workspace pins the tail property and the names' uniqueness).
QUANT_CONST_SLOTS: tuple = QUANT_SCALAR_SLOTS[QUANT_SCALAR_SLOTS.index("scale_q") :]
QUANT_SCALARS_BYTES: int = 256  # the region (256-B aligned; len(QUANT_SCALAR_SLOTS) x QUANT_SCALAR_STRIDE bytes used)
# Bytes between slots.  4-B views are legal for EVERY consumer: the fp8 SDPA adapter declares its scalars and its amax at 4-B
# alignment, the FROST GEMM runtime asks a scalar aux for `elem_bytes` only, and the quantize / init kernels declare
# assumed_align=4 on every slot pointer -- so the init launch zeroes ONE contiguous fp32 [n_slots] view.  That view is the
# COUPLING: the stride equals the fp32 element size in THREE places -- `_scalar()` (slot i at `QUANT_SCALAR_STRIDE * i`),
# the contiguous fp32 `[n_slots]` view `_execute_quant` hands the init launch, and the init kernel's store pitch
# (`kernels/quantize.py::_init_scalars`, `base + i * 4`).  A move to a 16-B stride must change all three together, or the
# readers sit on bytes the init never zeroed (an amax slot that never grows: silently wrong gradients, no crash) --
# `_plan_bwd_workspace` pins the equality so the first mismatched edit raises at declaration instead.
QUANT_SCALAR_STRIDE: int = 4
_GRAD_SCALING = ("current", "delayed")  # GatedAttentionBlockBwd(grad_scaling=): a DECLARATION attribute (numerics-changing), never a knob
# GatedAttentionBlockBwd(grad_scale_margin_log2=): the "current" recipe's headroom under the e4m3 maximum, in octaves -- a DECLARATION
# attribute like grad_scaling (it moves the e4m3 rounding points; a compile-time constant of the quantize artifacts, never a slot or a
# knob).  [0, 8]: eight octaves already drop the e4m3 grid's top eight binades of range for nothing a gradient needs.
_GRAD_SCALE_MARGIN_LOG2_MAX: int = 8
# The MMA-instruction K width of every e4m3 backward GEMM stage (B1 / B2 / B7 / B8): the 64-byte form is the measured one for
# the dense fp8 GEMMs of this backward (+5.6 .. +16.3 % over K32 at S = 2K .. 32K on B2's shape) and is passed EXPLICITLY --
# never derived from the dtype here or in the driver, so the forward's fp8 plans stay at their pinned K32.
_FP8_GEMM_MMA_TILE_K_BYTES: int = 64


# --- the MXFP8 backward's fused-prologue ARMS -- the plan-time fact the workspace carve is keyed on (never the quant spec's type) ---
# "mx_epilogue": the Q / K rebuild's MX token-tile epilogue writes the four block-scaled payloads (q8 / q_T8 / k8 / k_T8 with their blobs)
# straight out of registers, so no bf16 `recompute` / `recompute_k` region exists; "bf16_rebuild": the rebuild keeps its bf16 TMA store
# and the block quantizes read the bf16 buffers (the unfused chain, and the alternative fused shape that quantizes q_T / k_T from them
# by a dual-axis launch) -- both regions carved.  `_plan_bwd_workspace(mx_prologue_arm=)` reads the arm; a stage builder publishes it
# (`GatedAttentionBlockBwd.mx_prologue_arm`), so switching the arm never re-opens the carve's contract.
MX_PROLOGUE_ARM_MX_EPILOGUE = "mx_epilogue"
MX_PROLOGUE_ARM_BF16_REBUILD = "bf16_rebuild"
MX_PROLOGUE_ARMS = (MX_PROLOGUE_ARM_MX_EPILOGUE, MX_PROLOGUE_ARM_BF16_REBUILD)

# ---------------------------------------------------------------------------
# 1. Recompute policy — the knob a decomposed graph cannot offer
# ---------------------------------------------------------------------------


class RecomputePolicy(Enum):
    """How much of the forward the backward re-runs instead of reading.

    The forward's ``SavedForBackward`` says which tensors EXIST; this says what
    to do about the ones that do not. Both halves of the trade are real at
    scale, which is the whole argument for owning them at the block level.

    Which ``SavedForBackward`` fields each policy READS, and what it REBUILDS
    (the schema is ``api.py``'s; it is append-only and is not widened here):

    ``SAVE_ALL``
        Reads the pre-norm Q / K from the save set. With a ``proj_slab`` record
        (the forward's default save mode, and the ONLY record served today)
        they are its column bands -- as are the GATE and V -- so nothing is
        recomputed at all: the fastest backward, the largest footprint
        (34 KiB/token at the 397B geometry).

    ``RECOMPUTE_QK_PRE``
        "Rebuild what is ``None``". With a ``proj_slab`` record
        nothing is ``None`` and the policy reads the bands exactly like
        ``SAVE_ALL``. With a gate-copy record (``proj_slab=None``, a compact
        ``gate``) it re-runs the Q, K and V column slices of the forward's
        stage-(1) GEMM from ``saved.h`` -- a partial GEMM in exchange for the
        16 KiB/token the gate-copy save keeps. **Expected default**; the
        gate-copy half is a follow-up PR and :meth:`GatedAttentionBlockBwd.check_support`
        declines such a record typed until it lands.

    ``RECOMPUTE_GATE``
        RESERVED -- not servable against the current schema. It would
        additionally drop the GATE from the save set and re-run its slice too
        (another 16 GiB saved; the same GEMM, so at that point it is a full
        stage-(1) recompute). But the training forward ALWAYS saves the GATE
        -- as the compact ``gate`` (gate-copy save mode) or as a band of
        ``proj_slab`` (proj_slab save mode, where ``gate`` may be ``None``
        because ``saved_slab_views(proj_slab, ...)[1]`` derives it) -- and the
        stage-(1) GEMM always produces the GATE columns, so today there is
        nothing to drop. It becomes real when the forward learns to skip the
        GATE columns; until then :meth:`GatedAttentionBlockBwd.check_support`
        declines it rather than recomputing a tensor that is present.

    Two operands the schema does NOT hold, under every policy:

    * **Post-norm, post-RoPE Q and K are never saved.** The SDPA backward (B4)
      consumes them, and they are rebuilt from ``q_pre`` / ``k_pre`` by
      re-running the forward's stages (2)+(3) kernel (``api._QkNormRope``) --
      one bandwidth pass, no GEMM; the rstd it recomputes over the same inputs
      IS the forward's, and the SAVED ``rstd_q`` / ``rstd_k`` feed B6.
    * **V is a column slice of stage (1) and has no field of its own.** With a
      ``proj_slab`` it is the V band, compacted by one elementwise copy for the
      adapter (whose operands must be BSHD-physical). An appended ``Optional``
      ``v`` slot on ``SavedForBackward`` is a schema change and is not made here.

    Whatever is recomputed must be recomputed BIT-IDENTICALLY to the forward, or
    gradients acquire a noise floor that looks like a kernel bug. Same tile
    config, same accumulation order, same epilogue rounding.

    Under ``geometry.qk_norm=False`` the policy still decides ``q_pre`` /
    ``k_pre`` (the RoPE-only backward reads them through the same slots), but
    there is no ``rstd`` to save or rebuild -- stage (B6) does not exist.
    """

    SAVE_ALL = 0
    RECOMPUTE_QK_PRE = 1
    RECOMPUTE_GATE = 2


# ---------------------------------------------------------------------------
# 2. Workspace
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BwdIntermediates:
    """Byte offsets into the caller's workspace for the backward's scratch.

    ``dqkvg`` is one buffer in ``qkvg_offsets`` order so B7/B8 each see a single
    contiguous operand -- the whole reason the forward's concat ordering is
    contract rather than convenience. It is written exactly once per column:
    the GATE band by B3, the Q / K / V bands by B5+B6 (from the COMPACT B4 slots
    ``dq`` / ``dk`` / ``dv`` -- the adapter's BSHD-physical contract).

    ``-1`` means the region does not exist for this declaration (a ``need_*``
    False, a knob folded out) or aliases another one (``do``).

    Append-only: the first eight fields keep their positions; the block backward appended
    the rest (the gate-copy follow-up appends its recompute slabs after them).
    """

    do_gated: int  # [T, H_q, D]   B2's output
    do: int  # -1: ALIASES do_gated (B3 writes dO in place -- `do` MAY alias `dog`, kernels/sigmoid_gate_bwd.py)
    dqkvg: int  # [T, N]        dQ_pre | dG | dK_pre | dV, written ONLY by the elementwise kernels
    o_gated: int  # [T, H_q, D]   B3's third output (`og`) for B1; -1 when need_dw_o is False (has_og=False recipe)
    sdpa_bwd_ws: int  # the adapter's scratch_workspace_bytes(), 256-B aligned offset (it carves 128-B chunks inside)
    recompute: int  # [T, H_q, D]   rebuilt post-norm / post-RoPE Q (the adapter's input); -1 under quant (the rebuild writes q8 directly)

    total_bytes: int
    base_align: int
    # APPENDED with the block backward
    recompute_k: int = -1  # [T, H_kv, D]  rebuilt post-norm / post-RoPE K
    recompute_v: int = -1  # [T, H_kv, D]  compact V (copied from the proj_slab band)
    dq: int = -1  # [T, H_q, D]   B4 output (compact)
    dk: int = -1  # [T, H_kv, D]
    dv: int = -1  # [T, H_kv, D]
    dw_partials_q: int = -1  # [n_ctas_q, D] fp32 -- TWO planes, not one: the kernel takes two contiguous tensors
    dw_partials_k: int = -1  # [n_ctas_k, D] fp32    (exactly n_ctas_for(recipe, T) rows each); -1 without need_dw_norms
    gemm_scratch: int = -1  # max(plan.workspace_bytes for B1, B2, B7, B8), never 0 (max(.., 1))
    n_ctas_q: int = 0  # the plane row counts the carve was sized for (the reduce is launched with t=T and re-derives them)
    n_ctas_k: int = 0
    sdpa_bwd_bytes: int = 0  # the adapter's slice length (the region is padded to the carve alignment)
    gemm_scratch_bytes: int = 0
    # APPENDED with fuse_gate_bwd: B3's delta = rowsum(dO * O) for the adapter's external_delta; -1 / () when the knob is off
    delta: int = -1  # [B, H_q, S_pad] fp32 (the adapter's external_delta_shape)
    delta_shape: tuple = ()
    # APPENDED with fuse_wgrad_overlap: the side-stream wgrad GEMMs' (B1 / B7) own scratch -- they may run concurrently
    # with B8's use of gemm_scratch, so they never share it; -1 / 0 when the knob is off
    gemm_scratch_side: int = -1
    gemm_scratch_side_bytes: int = 0
    # APPENDED with the quantized (per-tensor fp8) backward (`quant=QuantSpec`): the e4m3 operands the fp8 GEMMs and the fp8 SDPA
    # row read, and the fp32 scalar block.  -1 under quant=None (the bf16 / fp16 layout is byte-identical to before); under quant
    # the bf16 `o_gated` (B3's third output becomes the e4m3 `og8`) and `recompute_v` (V is quantized straight from the slab's V
    # band: `v8` IS its compaction) are -1 instead.
    dy8: int = -1  # [T, d_model]  e4m3  the dY quantize           -> B1 (A, M-major), B2 (A, K-major)
    do8: int = -1  # [T, H_q, D]   e4m3  the dO quantize           -> B4's dO (and the adapter's dead `o` when og8 is absent)
    og8: int = -1  # [T, H_q, D]   e4m3  B3's fp8 arm (sat_e4m3(bf16(O * s) * scale_o)) -> B1 (B, N-major); -1 without need_dw_o
    q8: int = -1  # [T, H_q, D]   e4m3  the recomputed Q at the forward's static scale_q  -> B4
    k8: int = -1  # [T, H_kv, D]  e4m3  idem K (scale_k)                                 -> B4
    v8: int = -1  # [T, H_kv, D]  e4m3  the slab's V band at scale_v (V's compaction)      -> B4
    dqkvg8: int = -1  # [T, N]     e4m3  the dqkvg quantize        -> B7 (A, M-major), B8 (A, K-major)
    quant_scalars: int = -1  # QUANT_SCALARS_BYTES: the fp32 scalar block, slot i at + QUANT_SCALAR_STRIDE * i (QUANT_SCALAR_SLOTS)
    # APPENDED with the launch fusion: the dY amax PARTIALS (fp32 [amax_partials_n], one per CTA of the prologue's amax job, written
    # unconditionally; the dY quantize reduces them and publishes amax_dy).  -1 / 0 under quant=None and on a pure carve
    # (amax_partials_n=0, the default: the layout pins; compile() always passes the recipe's cap).  Under quant the bf16
    # `recompute` / `recompute_k` are -1 as well: the prologue's rebuild writes the e4m3 q8 / k8 straight out of its registers.
    amax_partials: int = -1
    amax_partials_n: int = 0
    # APPENDED with the partials form of the gradient amax: the gate backward's per-CTA max |dO| and max |dG| partials (fp32
    # [gate_partials_n] each, its persistent cap; the dO quantize reduces the first, the fused epilogue the second) and the norm
    # backward's per-CTA band partials (fp32 [band_partials_n], EXACTLY its grid for this T; the epilogue reduces them with the
    # dG partials).  -1 / 0 under quant=None and on a pure carve (the defaults; compile() passes the recipes' counts).
    amax_partials_do: int = -1
    amax_partials_dg: int = -1
    gate_partials_n: int = 0
    amax_partials_bands: int = -1
    band_partials_n: int = 0
    # APPENDED with the MXFP8 backward (`quant=MxQuantSpec`): the block-scaled operands the MXFP8 SDPA row and the two block-scale
    # GEMMs read -- each e4m3 payload with its uint8 E8M0 scale-factor blob.  -1 under quant=None AND under a QuantSpec (the bf16
    # and the per-tensor fp8 layouts are byte-identical to before).  The MXFP8 carve REUSES dy8 / do8 / og8 / q8 / k8 / v8 / dqkvg8 /
    # quant_scalars / amax_partials above (the same roles: per-tensor dy8 / og8, the rowwise payloads), carves the bf16
    # `recompute` / `recompute_k` like the bf16 backward (the rebuild writes bf16; the four Q / K quantizes read them) and never
    # `o_gated`, `recompute_v` (v8 is V's compaction) or the dO / dG / band amax partials (no per-tensor dO / dQKVG scale exists).
    sf_do: int = -1  # uint8 _sf_slot_bytes(B, H_q, S, D): the rowwise dO scale factors (the SDPA's per-(b, h, tile) layout) -> B4
    do_T8: int = -1  # [T, H_q, D] e4m3  the COLUMNWISE dO (32-token blocks along S; the row's dV operand)                   -> B4
    sf_do_T: int = -1  # uint8, the same byte count, D-plane-major                                                             -> B4
    sf_q: int = -1  # uint8 _sf_slot_bytes(B, H_q, S, D): the rowwise Q scale factors                                           -> B4
    q_T8: int = -1  # [T, H_q, D] e4m3  the columnwise Q (the row's dK operand)                                                 -> B4
    sf_q_T: int = -1  # uint8, D-plane-major                                                                                    -> B4
    sf_k: int = -1  # uint8 _sf_slot_bytes(B, H_kv, S, D)                                                                       -> B4
    k_T8: int = -1  # [T, H_kv, D] e4m3 the columnwise K (the row's dQ operand)                                                 -> B4
    sf_k_T: int = -1  # uint8, D-plane-major                                                                                    -> B4
    sf_v: int = -1  # uint8 _sf_slot_bytes(B, H_kv, S, D): V's ROWWISE scale factors (the backward quantizes V rowwise)        -> B4
    sf_dqkvg: int = -1  # uint8 sf_blob_bytes(T, N): the rowwise dQKVG scale factors in the GEMM's canonical F8_128x4 order   -> B8
    dqkvg_t8: int = -1  # [N, T] e4m3  dQKVG TRANSPOSED (32-token blocks along T, K-major for the wgrad)                        -> B7
    sf_dqkvg_t: int = -1  # uint8 sf_blob_bytes(N, T), canonical                                                                -> B7
    # APPENDED with the fp4 weight modes' backward (`MxQuantSpec.o_fp4`): the BLOCK quantization of dY that the out-projection dgrad
    # reads against the caller's e2m1 `w_o_t` -- next to the per-tensor `dy8` B1 keeps.  Under an MXFP4 W_o the MX-rowwise e4m3 `dy_mx8`
    # with its GEMM-canonical E8M0 blob (the mixed e4m3 x e2m1 row); under an NVFP4 W_o the two-level NVFP4 cast `dy4` (packed e2m1, two
    # codes per byte along d_model, of `scale_dy x dY`) with its e4m3-per-16 canonical blob (the NVFP4 x NVFP4 row).  -1 without `o_fp4`:
    # every MXFP8 offset above is byte-identical to before (these are carved LAST).
    dy_mx8: int = -1  # [T, d_model] e4m3  dY block-quantized rowwise (32-blocks along d_model)                                 -> B2 (A, K-major)
    sf_dy_mx: int = -1  # uint8 sf_blob_bytes(T, d_model), canonical                                                            -> B2
    dy4: int = -1  # [T, d_model // 2] packed e2m1  the NVFP4 cast of scale_dy x dY (16-blocks along d_model)                  -> B2 (A, K-major)
    sf_dy4: int = -1  # uint8 sf_blob_bytes(T, d_model, 16), canonical e4m3 scales                                              -> B2


def _mx_sf_bytes(geom: GatedAttentionBlockGeometry, b: int, s: int, heads: int) -> int:
    """Bytes of ONE SDPA-layout scale-factor blob over ``[B, heads, S, D]`` -- the forward's ``_sf_slot_bytes`` (the MXFP8
    SDPA adapter's ``_sf_expected_bytes`` count, the same for a rowwise and a columnwise blob); derived, never a literal."""
    from .api import _sf_slot_bytes

    return _sf_slot_bytes(b, heads, s, geom.d_head)


def _mx_canonical_sf_bytes(rows: int, k: int) -> int:
    """Bytes of ONE GEMM-canonical F8_128x4 blob over a ``[rows, K]`` matrix (``kernels.proj_gemm.sf_blob_bytes``: rows padded
    to 128, 32-element K blocks padded to 4) -- the block-scale GEMMs' SFA / SFB declaration; derived, never a literal."""
    from .kernels.proj_gemm import sf_blob_bytes

    return sf_blob_bytes(rows, k)


def _plan_bwd_workspace(
    geom: GatedAttentionBlockGeometry,
    b: int,
    s: int,
    dtype: torch.dtype,
    policy: RecomputePolicy,
    *,
    need: Optional[dict] = None,
    sdpa_bwd_bytes: int = 0,
    gemm_scratch_bytes: int = 0,
    n_ctas_q: int = 0,
    n_ctas_k: int = 0,
    delta_shape: Optional[tuple] = None,
    side_gemm_scratch_bytes: Optional[int] = None,
    quant: Optional[Union[QuantSpec, MxQuantSpec]] = None,
    need_og8: Optional[bool] = None,
    amax_partials_n: int = 0,
    gate_partials_n: int = 0,
    band_partials_n: int = 0,
    mx_prologue_arm: Optional[str] = None,
) -> _BwdIntermediates:
    """Reserve every backward intermediate and report the total.

    The positional head is the declaration; the keyword facts come from
    ``compile()`` -- the adapter's scratch, the four plans' ``workspace_bytes``
    and the dW-partial plane rows ``n_ctas_for(recipe, T)`` exist only once the
    artifacts do, which is why :meth:`GatedAttentionBlockBwd.get_workspace_size`
    requires ``compile()`` first. ``need`` = ``{"dw_o", "dw_norms", "dw_qkvg"}`` flags
    (missing = True; ``dw_qkvg`` is read by the MXFP8 carve only); ``policy`` is kept for the gate-copy follow-up's recompute slabs (a
    ``proj_slab`` record carves none).  ``delta_shape`` (``fuse_gate_bwd``: the adapter's
    ``external_delta_shape``, ``(B, H_q, S_pad)``) carves the fp32 ``delta`` region B3 writes
    and B4 reads; None carves none (the adapter keeps its own).  ``side_gemm_scratch_bytes`` (``fuse_wgrad_overlap``:
    ``max(plan.workspace_bytes)`` over the wgrad GEMMs) carves the side-stream GEMMs' own scratch LAST, ``max(.., 1)`` like
    ``gemm_scratch``; None carves none (every GEMM shares ``gemm_scratch``, in order).

    ``quant`` (appended; the quantized backward's ``QuantSpec``) swaps two bf16 regions for e4m3 ones and appends the rest
    AFTER every bf16 region, so the ``quant=None`` layout is byte-identical to before: ``o_gated`` is NOT carved (B3's third
    output is the e4m3 ``og8``, under ``need_dw_o`` as before) and neither is ``recompute_v`` (``v8`` is V's compaction); then,
    in this order, ``dy8`` ``[T, d_model]``, ``do8`` ``[T, H_q, D]``, ``og8`` ``[T, H_q, D]`` (``need_dw_o``), ``q8`` / ``k8`` /
    ``v8``, ``dqkvg8`` ``[T, N]`` (every one 1 B/elem) and the ``QUANT_SCALARS_BYTES`` fp32 scalar block (slot ``i`` at
    ``quant_scalars + QUANT_SCALAR_STRIDE * i``).  ``delta_shape`` is REQUIRED under ``quant`` (the gate backward's delta is
    the fp8 SDPA row's external delta -- there is no other producer).  ``need_og8`` (appended) overrides the ``og8`` carve:
    ``None`` = ``need["dw_o"]``; ``True`` carves it on a block without the wgrad too (an e4m3 operand of the dead ``o``'s shape
    for the adapter); ``False`` with ``need_dw_o`` is a contradiction (B1 reads it) and raises.  ``amax_partials_n`` (appended,
    default 0; ``quant`` only, >= 0) carves the fp32 dY amax partials ``[amax_partials_n]`` LAST when positive -- the prologue's
    amax job writes one per CTA, the dY quantize reduces them; ``compile()`` passes the compiled recipe's cap, and the default
    carves none (``amax_partials = -1``: a pure carve, as the layout pins and the host-side stand-ins build; ``execute`` refuses
    such a layout typed before any launch).  Under ``quant`` the bf16 ``recompute`` / ``recompute_k`` are NOT carved either (the
    rebuild writes the e4m3 ``q8`` / ``k8`` directly).  ``gate_partials_n`` / ``band_partials_n`` (appended, default 0; ``quant``
    only, >= 0) carve, after the dY partials and in this order, the gate backward's fp32 ``amax_partials_do`` and
    ``amax_partials_dg`` (``[gate_partials_n]`` each: its persistent cap) and the norm backward's ``amax_partials_bands``
    (``[band_partials_n]``: EXACTLY its grid for this ``T``) -- every amax of the quantized backward is a max over per-CTA
    partials its producer stored; the defaults carve none (a pure carve, as above).

    ``quant`` may also be an ``MxQuantSpec`` (the MXFP8 backward).  Its carve keeps the bf16 ``recompute`` / ``recompute_k`` (the
    bf16 rebuild feeds the four Q / K block quantizes -- a 16-row TMA tile holds no 32-token block), drops ``o_gated`` (B3's third
    output is the e4m3 ``og8``) and ``recompute_v`` (``v8`` is V's compaction) exactly like the fp8 carve, requires ``delta_shape``
    (the gate backward's delta is the MXFP8 row's external delta), and appends AFTER every bf16 region, in this order: ``dy8``
    ``[T, d_model]``; ``do8`` + ``sf_do``; ``do_T8`` + ``sf_do_T``; ``og8`` (``need_dw_o`` / ``need_og8``); ``q8`` + ``sf_q``; ``q_T8`` +
    ``sf_q_T``; ``k8`` + ``sf_k``; ``k_T8`` + ``sf_k_T``; ``v8`` + ``sf_v``; ``dqkvg8`` + ``sf_dqkvg`` (``sf_blob_bytes(T, N)``,
    canonical); ``dqkvg_t8`` ``[N, T]`` + ``sf_dqkvg_t`` (``sf_blob_bytes(N, T)``) iff ``need["dw_qkvg"]`` -- B7's operands alone,
    whole 32-token blocks along T, so a dgrad-only block at a ragged T carves neither (and ``sf_blob_bytes`` is never asked for a
    K it refuses); the scalar block; the dY amax partials (``amax_partials_n``, the standalone amax launch's cap).  The
    SDPA-layout blobs are ``_sf_slot_bytes`` each (the forward's count: rowwise and columnwise blobs have the same byte count).
    ``gate_partials_n`` / ``band_partials_n`` must be 0 under an
    ``MxQuantSpec`` (no per-tensor dO / dQKVG scale: nothing reduces them) -- a non-zero count is a typed contradiction.
    Under an ``MxQuantSpec`` with ``o_fp4`` (the fp4 weight modes' backward) ONE more pair is appended LAST, after the dY amax partials,
    so every MXFP8 offset stays byte-identical: ``dy_mx8`` ``[T, d_model]`` e4m3 + ``sf_dy_mx`` (``sf_blob_bytes(T, d_model)``, canonical)
    under ``Fp4Format.MXFP4`` -- the MX-rowwise dY the mixed-row out-projection dgrad reads --, or ``dy4`` (``T * d_model / 2`` bytes of
    packed e2m1) + ``sf_dy4`` (``sf_blob_bytes(T, d_model, 16)``) under ``Fp4Format.NVFP4`` -- the two-level NVFP4 cast of dY the NVFP4-row
    dgrad reads.  An MXFP4 ``W_qkvg`` alone carves nothing new (B8 swaps its B operand's dtype, not its A).

    ``mx_prologue_arm`` (appended, default None; ``MxQuantSpec`` only, one of ``MX_PROLOGUE_ARMS`` or None): the fused MXFP8 prologue's
    ARM, the plan-time fact the two bf16 rebuild regions are keyed on -- ``"mx_epilogue"`` (the block's prologue: the rebuild job writes
    ``q8 / q_T8 / k8 / k_T8`` out of registers) carves NEITHER ``recompute`` NOR ``recompute_k``, every later region moving up by exactly
    their two aligned sizes; ``"bf16_rebuild"`` and the default None (the unfused chain, and the alternative fused shape that quantizes
    ``q_T / k_T`` from the bf16 buffers) carve both, byte-identical to before.  Keyed on the arm and never on the quant spec's type, so
    taking the other arm changes one argument here and nothing else.  Given without an ``MxQuantSpec`` it is a typed contradiction.

    Regions are ``_WS_ALIGN`` (256 B) aligned so every typed ``_view`` and the
    adapter's own 128-B carve are legal; ``gemm_scratch`` is ``max(.., 1)`` so
    the slice handed to the GEMM driver is never empty.
    """
    need = dict(need or {})
    want_og = bool(need.get("dw_o", True))
    want_dw = bool(need.get("dw_norms", geom.qk_norm))
    want_dw_qkvg = bool(need.get("dw_qkvg", True))  # read by the MXFP8 carve only: B7's transposed dQKVG payload and its blob
    t, e, d = int(b) * int(s), _itemsize(dtype), geom.d_head
    # The two quantized arms are keyed APART: `fp8` is the per-tensor QuantSpec carve (no bf16 rebuild buffers: the fused prologue
    # writes q8 / k8), `mx` the MxQuantSpec carve (the bf16 rebuild buffers ARE carved); `quantized` is what they share.
    fp8 = isinstance(quant, QuantSpec)
    mx = isinstance(quant, MxQuantSpec)
    quantized = quant is not None
    spec_name = type(quant).__name__ if quantized else "None"
    if quantized:
        if not (fp8 or mx):
            raise ValueError(
                f"quant must be a QuantSpec (the per-tensor fp8 backward), an MxQuantSpec (the MXFP8 backward) or None, got {type(quant).__name__}"
            )
        if delta_shape is None:
            raise ValueError(
                f"quant={spec_name}: delta_shape is required -- the quantized backward ALWAYS carves the fp32 delta region (the gate backward's "
                "rowsum(dO * O) is the quantized SDPA row's external delta; the row's own pre-pass would recompute it over the e4m3 payloads)"
            )
        if QUANT_SCALAR_STRIDE != _itemsize(torch.float32):
            raise ValueError(
                f"QUANT_SCALAR_STRIDE={QUANT_SCALAR_STRIDE} must equal the fp32 element size ({_itemsize(torch.float32)} B): the scalar-init "
                "launch zeroes the block as ONE contiguous fp32 [n_slots] view (the `slots` view of GatedAttentionBlockBwd._execute_quant; "
                "kernels/quantize.py::_init_scalars stores at base + i * 4) while _scalar() places slot i at QUANT_SCALAR_STRIDE * i -- at any "
                "other stride the readers would sit on bytes the init never zeroed (an amax slot that never grows: silently wrong gradients, "
                "no crash); move the three together"
            )
        if len(QUANT_SCALAR_SLOTS) * QUANT_SCALAR_STRIDE > QUANT_SCALARS_BYTES:
            raise ValueError(
                f"the scalar block holds {len(QUANT_SCALAR_SLOTS)} slots at a QUANT_SCALAR_STRIDE={QUANT_SCALAR_STRIDE}-byte stride, more than its "
                f"QUANT_SCALARS_BYTES={QUANT_SCALARS_BYTES} region: a slot past the region would alias the next buffer"
            )
        if (
            len(set(QUANT_SCALAR_SLOTS)) != len(QUANT_SCALAR_SLOTS)
            or QUANT_SCALAR_SLOTS[len(QUANT_SCALAR_SLOTS) - len(QUANT_CONST_SLOTS) :] != QUANT_CONST_SLOTS
        ):
            raise ValueError(
                "QUANT_SCALAR_SLOTS must name every slot once, with QUANT_CONST_SLOTS as its TAIL: the scalar-init launch stores the plan-time "
                "constants as ONE contiguous slot range from its kernel arguments (slots are append-only; a new constant goes at the END)"
            )
        if isinstance(amax_partials_n, bool) or not isinstance(amax_partials_n, int) or amax_partials_n < 0:
            raise ValueError(
                f"quant={spec_name}: amax_partials_n must be a non-negative int (the dY amax job writes one fp32 partial per CTA: compile() "
                f"passes the compiled recipe's cap; 0, the default, carves no partials region -- a pure carve), got {amax_partials_n!r}"
            )
        for pname, pn in (("gate_partials_n", gate_partials_n), ("band_partials_n", band_partials_n)):
            if isinstance(pn, bool) or not isinstance(pn, int) or pn < 0:
                raise ValueError(
                    f"quant={spec_name}: {pname} must be a non-negative int (one fp32 partial per CTA of the producer's grid: compile() passes the "
                    f"compiled recipe's count; 0, the default, carves no region -- a pure carve), got {pn!r}"
                )
        if mx and (gate_partials_n or band_partials_n):
            raise ValueError(
                f"quant=MxQuantSpec: gate_partials_n={gate_partials_n} / band_partials_n={band_partials_n} must be 0 -- the MXFP8 backward "
                "block-quantizes dO and dQKVG (one E8M0 scale per 32-element block), so no per-tensor dO / dQKVG amax exists and nothing "
                "would reduce the gate backward's or the norm backward's partials"
            )
    elif amax_partials_n or gate_partials_n or band_partials_n:
        raise ValueError(
            f"amax_partials_n={amax_partials_n} / gate_partials_n={gate_partials_n} / band_partials_n={band_partials_n} without quant: the amax "
            "partials exist on the quantized backward only"
        )
    if mx_prologue_arm is not None:
        if not mx:
            raise ValueError(
                f"mx_prologue_arm={mx_prologue_arm!r} without an MxQuantSpec: the fused MXFP8 prologue's arm keys the MXFP8 carve alone (the bf16 "
                "backward always carves the rebuild buffers, the per-tensor fp8 one never does); pass None"
            )
        if mx_prologue_arm not in MX_PROLOGUE_ARMS:
            raise ValueError(f"mx_prologue_arm must be one of {MX_PROLOGUE_ARMS} or None, got {mx_prologue_arm!r}")
    want_og8 = (quantized and want_og) if need_og8 is None else bool(need_og8)
    if want_og8 and not quantized:
        raise ValueError("need_og8=True without quant: the e4m3 og8 region exists on the quantized backward only")
    if quantized and want_og and not want_og8:
        raise ValueError("need_og8=False with need dw_o=True: the out_proj wgrad (B1) reads og8 under quant; the two cannot disagree")
    layout = WorkspaceLayout(align=_WS_ALIGN)
    do_gated = layout.add(t * geom.h_q * d * e)
    dqkvg = layout.add(t * geom.n_qkvg * e)
    o_gated = layout.add(t * geom.h_q * d * e) if (want_og and not quantized) else -1
    # The bf16 rebuild buffers: the bf16 backward's, and the MXFP8 backward's under the bf16-rebuild arm (its Q / K block quantizes read
    # them); the per-tensor fp8 backward carves none (its fused prologue writes q8 / k8 straight out of the norm + RoPE body's registers)
    # and neither does the MXFP8 backward under its MX-epilogue prologue (the same, for the four block-scaled Q / K payloads and their
    # blobs) -- keyed on the ARM, never on the quant spec's type.
    carve_rebuild = not fp8 and not (mx and mx_prologue_arm == MX_PROLOGUE_ARM_MX_EPILOGUE)
    recompute = layout.add(t * geom.h_q * d * e) if carve_rebuild else -1
    recompute_k = layout.add(t * geom.h_kv * d * e) if carve_rebuild else -1
    recompute_v = layout.add(t * geom.h_kv * d * e) if not quantized else -1
    dq = layout.add(t * geom.h_q * d * e)
    dk = layout.add(t * geom.h_kv * d * e)
    dv = layout.add(t * geom.h_kv * d * e)
    if want_dw:
        if n_ctas_q < 1 or n_ctas_k < 1:
            raise ValueError(f"need_dw_norms needs the dW partial plane rows from the compiled recipe (n_ctas_q={n_ctas_q}, n_ctas_k={n_ctas_k})")
        dw_partials_q = layout.add(int(n_ctas_q) * d * 4)
        dw_partials_k = layout.add(int(n_ctas_k) * d * 4)
    else:
        dw_partials_q = dw_partials_k = -1
        n_ctas_q = n_ctas_k = 0
    if sdpa_bwd_bytes < 1:
        raise ValueError(f"the SDPA backward adapter's scratch must be known (got {sdpa_bwd_bytes} bytes); it is a pure function of the geometry")
    sdpa_bwd_ws = layout.add(int(sdpa_bwd_bytes))
    gemm_scratch_bytes = max(int(gemm_scratch_bytes), 1)
    gemm_scratch = layout.add(gemm_scratch_bytes)
    delta = -1
    if delta_shape is not None:
        delta_shape = tuple(int(x) for x in delta_shape)
        if len(delta_shape) != 3 or delta_shape[0] != b or delta_shape[1] != geom.h_q or delta_shape[2] < s:
            raise ValueError(f"delta_shape must be the adapter's (B={b}, H_q={geom.h_q}, S_pad >= {s}), got {delta_shape}")
        delta = layout.add(int(math.prod(delta_shape)) * 4)
    gemm_scratch_side, gemm_scratch_side_bytes = -1, 0
    if side_gemm_scratch_bytes is not None:
        gemm_scratch_side_bytes = max(int(side_gemm_scratch_bytes), 1)
        gemm_scratch_side = layout.add(gemm_scratch_side_bytes)
    # The quantized backward's regions, appended AFTER every bf16 one (1 B/elem e4m3 codes, then the fp32 scalar block).
    dy8 = do8 = og8 = q8 = k8 = v8 = dqkvg8 = quant_scalars = amax_partials = -1
    amax_partials_do = amax_partials_dg = amax_partials_bands = -1
    sf_do = do_T8 = sf_do_T = sf_q = q_T8 = sf_q_T = sf_k = k_T8 = sf_k_T = sf_v = sf_dqkvg = dqkvg_t8 = sf_dqkvg_t = -1
    dy_mx8 = sf_dy_mx = dy4 = sf_dy4 = -1
    if fp8:
        e8 = _itemsize(quant.dtype)
        dy8 = layout.add(t * geom.d_model * e8)
        do8 = layout.add(t * geom.h_q * d * e8)
        og8 = layout.add(t * geom.h_q * d * e8) if want_og8 else -1
        q8 = layout.add(t * geom.h_q * d * e8)
        k8 = layout.add(t * geom.h_kv * d * e8)
        v8 = layout.add(t * geom.h_kv * d * e8)
        dqkvg8 = layout.add(t * geom.n_qkvg * e8)
        quant_scalars = layout.add(QUANT_SCALARS_BYTES)
        amax_partials = layout.add(int(amax_partials_n) * 4) if amax_partials_n else -1
        amax_partials_do = layout.add(int(gate_partials_n) * 4) if gate_partials_n else -1
        amax_partials_dg = layout.add(int(gate_partials_n) * 4) if gate_partials_n else -1
        amax_partials_bands = layout.add(int(band_partials_n) * 4) if band_partials_n else -1
    elif mx:
        # The MXFP8 backward's regions: every block-scaled payload right before its E8M0 scale-factor blob (the SDPA-layout blobs at
        # the forward's `_sf_slot_bytes`, the GEMM-canonical ones at `sf_blob_bytes`), the per-tensor dy8 / og8 as the fp8 carve.
        e8 = _itemsize(quant.dtype)
        hq, hk, n = geom.h_q, geom.h_kv, geom.n_qkvg
        dy8 = layout.add(t * geom.d_model * e8)
        do8 = layout.add(t * hq * d * e8)
        sf_do = layout.add(_mx_sf_bytes(geom, b, s, hq))
        do_T8 = layout.add(t * hq * d * e8)
        sf_do_T = layout.add(_mx_sf_bytes(geom, b, s, hq))
        og8 = layout.add(t * hq * d * e8) if want_og8 else -1
        q8 = layout.add(t * hq * d * e8)
        sf_q = layout.add(_mx_sf_bytes(geom, b, s, hq))
        q_T8 = layout.add(t * hq * d * e8)
        sf_q_T = layout.add(_mx_sf_bytes(geom, b, s, hq))
        k8 = layout.add(t * hk * d * e8)
        sf_k = layout.add(_mx_sf_bytes(geom, b, s, hk))
        k_T8 = layout.add(t * hk * d * e8)
        sf_k_T = layout.add(_mx_sf_bytes(geom, b, s, hk))
        v8 = layout.add(t * hk * d * e8)
        sf_v = layout.add(_mx_sf_bytes(geom, b, s, hk))
        dqkvg8 = layout.add(t * n * e8)
        sf_dqkvg = layout.add(_mx_canonical_sf_bytes(t, n))
        if want_dw_qkvg:
            # B7's A operand alone: the TRANSPOSED quantization, its canonical blob over (rows = N, K = T) in whole 32-token blocks along T
            # (check_support's B*S % 32 rule, bound to need_dw_qkvg) -- a dgrad-only block serves a ragged T and carves neither
            dqkvg_t8 = layout.add(n * t * e8)
            sf_dqkvg_t = layout.add(_mx_canonical_sf_bytes(n, t))
        quant_scalars = layout.add(QUANT_SCALARS_BYTES)
        amax_partials = layout.add(int(amax_partials_n) * 4) if amax_partials_n else -1
        if quant.o_fp4 is not None:
            # The fp4 W_o arms' dY BLOCK quantization (B2's A operand), appended LAST: the MX-rowwise e4m3 dY with its canonical E8M0 blob
            # (MXFP4 W_o, the mixed row) or the packed e2m1 two-level NVFP4 cast with its canonical e4m3 blob (NVFP4 W_o, the NVFP4 row);
            # the per-tensor dy8 above stays (B1's A).  The byte counts are the GEMM's (sf_blob_bytes at the format's block), never typed.
            from .kernels.proj_gemm import FP4_CODES_PER_BYTE, sf_blob_bytes

            if quant.o_fp4 is Fp4Format.MXFP4:
                dy_mx8 = layout.add(t * geom.d_model * e8)
                sf_dy_mx = layout.add(_mx_canonical_sf_bytes(t, geom.d_model))
            else:
                dy4 = layout.add(t * geom.d_model // FP4_CODES_PER_BYTE)
                sf_dy4 = layout.add(sf_blob_bytes(t, geom.d_model, quant.o_fp4.block_size))
    else:
        amax_partials_n = gate_partials_n = band_partials_n = 0
    return _BwdIntermediates(
        do_gated=do_gated,
        do=-1,
        dqkvg=dqkvg,
        o_gated=o_gated,
        sdpa_bwd_ws=sdpa_bwd_ws,
        recompute=recompute,
        total_bytes=layout.size,
        base_align=layout.base_align,
        recompute_k=recompute_k,
        recompute_v=recompute_v,
        dq=dq,
        dk=dk,
        dv=dv,
        dw_partials_q=dw_partials_q,
        dw_partials_k=dw_partials_k,
        gemm_scratch=gemm_scratch,
        n_ctas_q=int(n_ctas_q),
        n_ctas_k=int(n_ctas_k),
        sdpa_bwd_bytes=int(sdpa_bwd_bytes),
        gemm_scratch_bytes=gemm_scratch_bytes,
        delta=delta,
        delta_shape=tuple(delta_shape) if delta_shape is not None else (),
        gemm_scratch_side=gemm_scratch_side,
        gemm_scratch_side_bytes=gemm_scratch_side_bytes,
        dy8=dy8,
        do8=do8,
        og8=og8,
        q8=q8,
        k8=k8,
        v8=v8,
        dqkvg8=dqkvg8,
        quant_scalars=quant_scalars,
        amax_partials=amax_partials,
        amax_partials_n=int(amax_partials_n),
        amax_partials_do=amax_partials_do,
        amax_partials_dg=amax_partials_dg,
        gate_partials_n=int(gate_partials_n),
        amax_partials_bands=amax_partials_bands,
        band_partials_n=int(band_partials_n),
        sf_do=sf_do,
        do_T8=do_T8,
        sf_do_T=sf_do_T,
        sf_q=sf_q,
        q_T8=q_T8,
        sf_q_T=sf_q_T,
        sf_k=sf_k,
        k_T8=k_T8,
        sf_k_T=sf_k_T,
        sf_v=sf_v,
        sf_dqkvg=sf_dqkvg,
        dqkvg_t8=dqkvg_t8,
        sf_dqkvg_t=sf_dqkvg_t,
        dy_mx8=dy_mx8,
        sf_dy_mx=sf_dy_mx,
        dy4=dy4,
        sf_dy4=sf_dy4,
    )


# ---------------------------------------------------------------------------
# 3. Stages — one FROST kernel each, in pipeline order
# ---------------------------------------------------------------------------


class _GemmStage(_Stage):
    """Shared body of the four projection GEMMs (B1, B2, B7, B8) on the shipped
    FROST GEMM through ``kernels/proj_gemm.py``'s backward drivers.

    ``kind`` picks the driver and the plan's majors: ``"wgrad"`` is
    ``dW[rows, cols] = dy_like[T, rows]^T @ x[T, cols]`` (A M-major, B N-major,
    ``m=rows, k=T, n=cols``); ``"dgrad"`` is ``dX[T, N] = dy_like[T, K] @ w[K, N]``
    with ``w`` the UN-transposed row-major weight (A K-major, B N-major,
    ``m=T, k=K, n=N``). Both bind zero-copy views of contiguous ``[rows, cols]``
    storage and refuse anything else, typed, before any launch -- never hand
    them a column band of the slab.

    The tile is the block's FORCED ``..._cluster2x1_2ctamma`` config whenever
    ``n % 256 == 0`` (every backward GEMM of the block), at one split-K slice on
    the JIT route; ``plan.tile_config_name`` / ``plan.route`` / ``plan.jit``
    record what runs and the tests pin them (a fallback to the heuristic is a
    FAILURE: different config, route and possibly a split-K reducer).
    :meth:`expected_tile_config_name` spells the name a served plan must carry
    -- the forced K32 config, or its 64-byte-MMA-K twin when the stage asked
    for it -- derived from the one constant through ``tile_config.as_mma_tile_k``,
    never a second literal.

    ``mma_tile_k_bytes`` (appended, default ``None`` = the named config's own
    width, byte-identical plans): the MMA-instruction K width of an 8-bit
    (e4m3) stage.  The quantized backward passes ``64`` EXPLICITLY -- the
    measured form of its dense fp8 GEMMs -- and it is never derived from the
    dtype here or in the driver, so the forward's fp8 plans stay at their
    pinned K32.  On a bf16 / fp16 stage any value is a typed decline at
    ``check_support`` (one MMA K width exists for 2-byte operands); on an e4m3
    stage the knob is REQUIRED (32 or 64), so a stage can never fall back to a
    width nobody chose.

    ``out_dtype`` / ``alpha`` (appended after ``mma_tile_k_bytes``, defaults
    ``None`` / ``False`` = today's bf16 / fp16 stage, byte-identical plan
    request): the e4m3 stage's declaration.  An e4m3 stage is served in exactly
    ONE form -- ``alpha=True`` with ``out_dtype=torch.bfloat16``: fp32
    accumulate over the e4m3 codes, ONE fused scalar-multiply epilogue by the
    descale product ``alpha = descale_A * descale_B`` (B1 ``descale_dY *
    (1 / scale_o)``, B2 ``descale_dY * descale_w_o``, B7 ``descale_dQKVG *
    descale_h``, B8 ``descale_dQKVG * descale_w_qkvg``) read from a 1-element
    fp32 DEVICE slot the caller owns (``execute(alpha=)``, required iff the plan
    carries it -- never a silent 1.0, never a dropped value), and a bf16
    output.  Anything else is a typed decline naming the field: e4m3 without
    ``alpha`` (the codes would be multiplied unscaled), an e4m3 or half
    ``out_dtype`` on an e4m3 stage, ``alpha`` / ``out_dtype`` on a bf16 / fp16
    stage (no such epilogue).  The e4m3 renderings are the driver's validated
    MN-major table (``FP8_MN_MAJOR_VALIDATED``: the wgrad ``("m", "n")`` and the
    dgrad ``("k", "n")`` triples at the forced tile, K32 and K64), and the TMA
    16-byte contiguous-extent rule is checked here on the axes it binds -- an
    operand's CONTIGUOUS axis: the MN extents of the MN-major operands (16 e4m3
    elements) and ``K % 16 == 0`` only when an operand is K-major (a dgrad's A;
    the dgrads contract over ``d_model`` / ``n_qkvg``, both multiples of 256).
    A wgrad (M-major A, N-major B) has no K-contiguous operand, so its token
    count ``K = B*S`` is unconstrained: the ragged K tail is TMA zero-fill, as
    for the bf16 twins.

    ``block_scale`` / ``w_dtype`` / ``block_size`` / ``sf_dtype`` (appended after
    ``alpha``, defaults ``False`` / ``None`` / ``32`` / ``None`` = today's stages,
    byte-identical plan request): the block-scale GEMM stage -- the MXFP8
    backward's B7 / B8 and the fp4 weight modes' two dgrads -- served in exactly
    the rows of :attr:`BLOCK_SCALE_ROWS` (``(A dtype, W dtype, scale block)``,
    resolved and typed by ``block_scale_pairing``): the **MXFP8 x MXFP8** row
    (e4m3 codes on both operands, one E8M0 scale per 32-element K block: B7 /
    B8); the **mixed** row (e4m3 A, ``w_dtype=torch.float4_e2m1fn_x2``, E8M0 per
    32 on both sides: B8 under an MXFP4 ``W_qkvg``, B2 under an MXFP4 ``W_o`` with
    an MX-rowwise e4m3 dY); the **NVFP4 x NVFP4** row (``dtype = w_dtype`` e2m1,
    ``block_size=16``, ``sf_dtype`` ``FP8_E4M3`` or None: B2 under an NVFP4 ``W_o``
    with dY cast to NVFP4).  ``w_dtype`` None or a member of
    :attr:`BLOCK_SCALE_W_DTYPES` (e4m3, e2m1); every other catalog pair -- the
    mixed row the other way round, MXFP4 x MXFP4 -- is a typed decline naming the
    row (no stage of the block declares it: the MXFP4 ``W_o`` arm keeps the
    gradient 8-bit through the mixed row).  NO alpha: the block dequant is exact
    and happens IN the MMA (a power of two per block on each operand, or the e4m3
    scale of the NVFP4 row), so there is no descale product to apply and
    ``alpha=True`` is refused; ``out_dtype=torch.bfloat16``; the 64-byte MMA K -- an unset
    ``mma_tile_k_bytes`` resolves to it at declaration, so
    :meth:`expected_tile_config_name` and the block's forced-tile pin name the
    forced tile's K64 twin before ``compile`` runs, and 32 is refused (the
    block-scale renderings are validated at K64 only); and **``K % 32 == 0``** --
    ``build_proj_gemm``'s rule for the block-scale rows, said at declaration with
    the fix named: the weight gradient contracts over the token axis ``K = T =
    B*S``, so pad or batch the sequence to a multiple of 32, pass
    ``need_dw_qkvg=False`` (the data gradients contract over ``d_model`` /
    ``n_qkvg`` and are served at any ``T``), or run the per-tensor fp8 backward
    (``quant=QuantSpec``), whose weight gradients take no block scales.  Both
    operands are K-major TRANSPOSED artifacts (:attr:`majors` is ``("k", "k")``:
    the block-scale rows lay their F8_128x4 scale-factor blobs over K-major rows
    and take no other major, so the TMA 16-byte rule falls on ``K``, which
    ``K % 32`` covers): the wgrad binds the transposed, block-quantized
    ``dQKVG^T [N, T]`` + its blob against the caller's ``h^T [d_model, T]`` +
    ``h_t_sf``; the dgrad the rowwise ``dQKVG [T, N]`` + its blob against the
    caller's ``W_qkvg^T [d_model, N]`` + ``w_qkvg_t_sf`` -- an e2m1 side as its
    PACKED ``[.., K // 2]`` storage (two codes per byte along K), the drivers'
    storage rule.  ``execute(sf_a=,
    sf_b=)`` takes the two PADDED blobs (``proj_gemm.sf_blob_bytes``), required
    iff the plan is block-scale and refused otherwise (never a silent unit scale),
    and dispatches to ``run_wgrad_gemm_block_scale`` /
    ``run_dgrad_gemm_block_scale``, whose operand checks -- contiguous row-major
    ``[rows, K]`` storage (a ``.t()`` view of the un-transposed tensor or a slice
    of a wider slab is a typed refusal naming the operand and both strides) and
    the blobs' dtype / size / alignment under the drivers' own keywords -- run
    before any launch.  B1 / B2 stay per-tensor e4m3 under MXFP8: the
    out-projection side of the quantized backward carries one scale per tensor.
    """

    kind: str = ""
    # The weight dtypes a block-scale stage serves: e4m3 (the MXFP8 x MXFP8 row) and e2m1 (the mixed row against e4m3 codes, the
    # NVFP4 x NVFP4 row against e2m1 codes); the stage tests key their `w_dtype` expectations on it.  A torch without the fp4 storage
    # dtype serves e4m3 alone (the same `getattr` guard as `proj_gemm._is_fp4`).
    BLOCK_SCALE_W_DTYPES: tuple = tuple(dt for dt in (torch.float8_e4m3fn, getattr(torch, "float4_e2m1fn_x2", None)) if dt is not None)
    # The block-scale rows a backward GEMM stage SERVES, as `(A dtype, W dtype, scale block)` -- exactly the renderings validated on Rubin
    # through the blobs (the forward's own renderings at the backward's shapes), nothing else of the catalog: MXFP8 x MXFP8 (B7 / B8 of
    # the MXFP8 backward), the mixed e4m3 x e2m1 row at E8M0 per 32 (B8 under an MXFP4 W_qkvg; B2 under an MXFP4 W_o with an MX-rowwise
    # e4m3 dY) and NVFP4 x NVFP4 at e4m3 per 16 (B2 under an NVFP4 W_o: dY itself cast to NVFP4).  `block_scale_pairing` resolves and
    # types the (sf_dtype, block) of a pair; the row check of `check_support` then declines the other catalog pairs by name (e2m1 x e4m3;
    # MXFP4 x MXFP4 -- the MXFP4 W_o arm keeps its gradient 8-bit through the mixed row).
    BLOCK_SCALE_ROWS: tuple = tuple(
        (a, w, blk)
        for a, w, blk in (
            (torch.float8_e4m3fn, torch.float8_e4m3fn, 32),
            (torch.float8_e4m3fn, getattr(torch, "float4_e2m1fn_x2", None), 32),
            (getattr(torch, "float4_e2m1fn_x2", None), getattr(torch, "float4_e2m1fn_x2", None), 16),
        )
        if a is not None and w is not None
    )

    def __init__(
        self,
        *,
        m: int,
        k: int,
        n: int,
        dtype: torch.dtype,
        label: str,
        mma_tile_k_bytes: Optional[int] = None,
        out_dtype: Optional[torch.dtype] = None,
        alpha: bool = False,
        block_scale: bool = False,
        w_dtype: Optional[torch.dtype] = None,
        block_size: int = 32,
        sf_dtype=None,
    ) -> None:
        """Record the declaration as given (``m`` / ``k`` / ``n`` as ints); validation is ``check_support``'s, the plan ``compile``'s.
        ONE value is resolved here rather than recorded: a block-scale stage declared without ``mma_tile_k_bytes`` takes the 64-byte
        MMA K (``_FP8_GEMM_MMA_TILE_K_BYTES``, the width of every e4m3 backward GEMM stage) BEFORE ``check_support`` /
        :meth:`expected_tile_config_name` / the block's forced-tile pin read it, so all three name the forced tile's K64 twin and
        ``compile`` builds exactly that plan -- the driver's own block-scale resolution would pick the same width on Rubin, but
        silently, after the name was checked."""
        self.m, self.k, self.n = int(m), int(k), int(n)
        self.dtype = dtype
        self.label = label
        self.block_scale = bool(block_scale)
        self.w_dtype = w_dtype
        self.block_size = block_size
        self.sf_dtype = sf_dtype
        if self.block_scale and mma_tile_k_bytes is None:
            mma_tile_k_bytes = _FP8_GEMM_MMA_TILE_K_BYTES
        self.mma_tile_k_bytes = mma_tile_k_bytes
        self.out_dtype = out_dtype
        self.alpha = bool(alpha)
        self.plan = None

    @property
    def is_e4m3(self) -> bool:
        """e4m3 codes in: the per-tensor fp8 stage (the descale product in the alpha epilogue, bf16 out) or, under
        ``block_scale``, the MXFP8 stage (one E8M0 scale per 32-element K block, dequantized in the MMA, bf16 out)."""
        e4m3 = getattr(torch, "float8_e4m3fn", None)
        return e4m3 is not None and self.dtype == e4m3

    @property
    def is_e2m1(self) -> bool:
        """e2m1 codes in (``torch.float4_e2m1fn_x2``, two per byte): the NVFP4 x NVFP4 block-scale stage's A -- the NVFP4 cast of dY
        under an NVFP4 ``W_o``; never a dense or per-tensor stage's dtype."""
        e2m1 = getattr(torch, "float4_e2m1fn_x2", None)
        return e2m1 is not None and self.dtype == e2m1

    @classmethod
    def _block_scale_rows_menu(cls) -> str:
        """The served rows, spelled for a decline message."""
        from .kernels.proj_gemm import _dtype_word

        return "; ".join(f"A {_dtype_word(a)} x W {_dtype_word(w)} at one scale per {blk}" for a, w, blk in cls.BLOCK_SCALE_ROWS)

    @property
    def majors(self) -> tuple:
        """``(a_major, b_major)`` of the plan: a block-scale stage binds two K-major TRANSPOSED artifacts (``("k", "k")`` -- the
        drivers' ``_check_k_major_block_scale_plan``); a per-tensor stage the wgrad's ``("m", "n")`` or the dgrad's ``("k", "n")``."""
        if self.block_scale:
            return ("k", "k")
        return ("m", "n") if self.kind == "wgrad" else ("k", "n")

    def expected_tile_config_name(self) -> Optional[str]:
        """The catalog name of the plan a served stage compiles to, or ``None`` where the forced tile does not apply
        (``n % 256 != 0``, the heuristic's pick): the block's forced K32 config (``proj_gemm._forced_tile_config``),
        re-spelled at the requested MMA K width through ``tile_config.as_mma_tile_k`` -- the K64 twin
        ``..._128x256x64_cluster2x1_2ctamma`` when the stage asked for 64 (every block-scale stage: its unset
        ``mma_tile_k_bytes`` resolved to 64 at declaration), the config's own name otherwise."""
        from cudnn.gemm.frost.tile_config import as_mma_tile_k, by_name

        from .kernels.proj_gemm import _forced_tile_config

        want = _forced_tile_config(self.n)
        if want is None or self.mma_tile_k_bytes is None:
            return want
        return as_mma_tile_k(by_name(want), int(self.mma_tile_k_bytes)).name

    def check_support(self) -> None:
        """Typed declines before any graph.  An e4m3 stage: its one served form -- ``alpha=True``, ``out_dtype=torch.bfloat16``,
        an explicit ``mma_tile_k_bytes`` of 32 or 64 -- and the TMA 16-byte rule on ``K`` (``ValueError`` naming the field).
        A bf16 / fp16 stage: any ``alpha`` / ``out_dtype`` / ``mma_tile_k_bytes`` (``NotImplementedError`` -- the e4m3 stage's
        declaration); any other dtype is a typed decline.  Every stage: an ``M`` of an M-major A or the ``N`` of the N-major
        B off the TMA 16-byte rule (``ValueError``).  A block-scale stage: its served rows -- e4m3 or (against an e2m1 weight) e2m1
        codes on A, ``w_dtype`` None or in :attr:`BLOCK_SCALE_W_DTYPES` (any other ``dtype`` / ``w_dtype`` is a ``NotImplementedError``
        naming the field), ``alpha=False``, ``out_dtype=torch.bfloat16``, the 64-byte MMA K, ``(sf_dtype, block_size)`` typed by
        ``block_scale_pairing`` and the resolved ``(A, W, block)`` one of :attr:`BLOCK_SCALE_ROWS` (any other catalog pair is a
        ``NotImplementedError`` naming the row), and ``K`` a multiple of the scale block that pairing resolves (32, or 16 under NVFP4:
        ``build_proj_gemm``'s rule) with the fix named (``ValueError``); both operands are K-major, so the MN rule does not apply to it."""
        if self.block_scale:
            # The block-scale stage's served rows.  Each field is checked by name so a wrong declaration says which.
            from .kernels.proj_gemm import _dtype_word, block_scale_pairing, check_sf_torch_dtype

            if not (self.is_e4m3 or self.is_e2m1):
                raise NotImplementedError(
                    f"{self.name}: block_scale=True serves e4m3 codes with per-block E8M0 scale factors (the MXFP8 backward's block-quantized dQKVG "
                    f"against the caller's transposed artifacts) or, against an e2m1 weight, e2m1 codes with e4m3 scales per 16 (the NVFP4 cast of dY), "
                    f"got dtype={self.dtype}; the bf16 / fp16 stages and the per-tensor e4m3 stage take block_scale=False"
                )
            if self.w_dtype is not None and self.w_dtype not in self.BLOCK_SCALE_W_DTYPES:
                raise NotImplementedError(
                    f"{self.name}: w_dtype={self.w_dtype} on a block-scale GEMM stage -- the backward serves e4m3 (the MXFP8 x MXFP8 row) and "
                    f"torch.float4_e2m1fn_x2 (the mixed e4m3 x e2m1 row, the NVFP4 x NVFP4 row) weights: {self.BLOCK_SCALE_W_DTYPES}"
                )
            if self.alpha:
                raise ValueError(
                    f"{self.name}: alpha=True on a block-scale GEMM stage -- the E8M0 dequant is exact and happens in the MMA (one power of two per "
                    "32-element block on each operand), so there is no descale product to apply; declare alpha=False"
                )
            if self.out_dtype != torch.bfloat16:
                raise ValueError(
                    f"{self.name}: a block-scale GEMM stage writes a bf16 output -- declare out_dtype=torch.bfloat16 (got out_dtype={self.out_dtype}); "
                    "the quantized backward's gradients are bf16"
                )
            if self.mma_tile_k_bytes != _FP8_GEMM_MMA_TILE_K_BYTES:
                raise ValueError(
                    f"{self.name}: mma_tile_k_bytes must be {_FP8_GEMM_MMA_TILE_K_BYTES} on a block-scale GEMM stage (or None, which resolves to it at "
                    f"declaration): the block-scale renderings are validated at the forced tile's 64-byte-MMA-K twin only, and it is the width the "
                    f"driver's own block-scale resolution picks on Rubin; got {self.mma_tile_k_bytes!r}"
                )
            # (sf_dtype, block_size) against the driver's served pairs -- E8M0 per 32 for e4m3 x e4m3 -- a ValueError naming the pair.
            # The block it resolves is the K rule's unit: `build_proj_gemm` needs K to hold whole scale blocks (32 on every row this
            # backward declares; it re-checks that at compile, together with a packed e2m1 operand's TMA extent -- two E4M3 blocks,
            # the same 32), so the stage asks the pairing for the number instead of restating it.
            w_dtype = self.w_dtype if self.w_dtype is not None else self.dtype
            sf_dtype, k_block = block_scale_pairing(dtype=self.dtype, w_dtype=w_dtype, sf_dtype=self.sf_dtype, block_size=self.block_size, label=self.name)
            if (self.dtype, w_dtype, k_block) not in self.BLOCK_SCALE_ROWS:
                # A catalog pair the pairing serves but no stage of this backward declares (e2m1 x e4m3; MXFP4 x MXFP4): declined by name,
                # never rendered unvalidated.
                raise NotImplementedError(
                    f"{self.name}: the block-scale row A {_dtype_word(self.dtype)} x W {_dtype_word(w_dtype)} at one scale per {k_block} elements is not a "
                    f"rendering this backward declares; its GEMM stages run {self._block_scale_rows_menu()} -- the MXFP4 W_o arm keeps the gradient 8-bit "
                    "through the mixed row, and no stage multiplies e2m1 codes by an e4m3 weight"
                )
            sf_word = str(check_sf_torch_dtype(sf_dtype)).replace("torch.", "")
            if self.k % k_block:
                if self.kind == "wgrad":
                    axis, fix = (
                        "the token axis T = B*S",
                        f"pad or batch the sequence to a multiple of {k_block}, pass need_dw_qkvg=False (the data gradients dh / dW_o contract over "
                        "d_model / n_qkvg and are served at any T), or run the per-tensor fp8 backward (quant=QuantSpec), whose weight gradients "
                        "take no block scales",
                    )
                else:
                    axis, fix = (
                        "the projection's input width",
                        f"the contracted feature width (d_model / n_qkvg) is a multiple of {k_block} at every geometry the block serves -- declare one",
                    )
                raise ValueError(
                    f"{self.name}: block_scale=True contracts over K={self.k} ({axis}) through the block-scale GEMM, which takes one {sf_word} scale per "
                    f"{k_block}-element K block, so K must be a multiple of {k_block} (got K={self.k}); {fix}"
                )
            # Both operands are K-major (the TMA 16-byte rule falls on K, covered above): no MN rule for this stage.
            return
        # The block-scale declaration's fields have no meaning on a per-tensor stage (no scale-factor blobs, no block along K): each is
        # refused by name rather than dropped -- the driver would silently ignore `sf_dtype` / `block_size` on a dense plan.
        for field, val, default in (("w_dtype", self.w_dtype, None), ("block_size", self.block_size, 32), ("sf_dtype", self.sf_dtype, None)):
            if val != default:
                raise NotImplementedError(
                    f"{self.name}: {field}={val!r} is the block-scale (MXFP8) GEMM stage's declaration; a per-tensor stage has no scale-factor blobs "
                    f"-- leave it at its default ({default!r}) or declare block_scale=True"
                )
        if self.is_e4m3:
            # The e4m3 stage's ONE served form.  Each field is checked by name so a wrong declaration says which.
            if not self.alpha:
                raise ValueError(
                    f"{self.name}: an e4m3 GEMM stage carries its descale product (descale_A * descale_B) in the alpha epilogue -- declare "
                    "alpha=True (the codes would otherwise be multiplied unscaled)"
                )
            if self.out_dtype != torch.bfloat16:
                raise ValueError(
                    f"{self.name}: an e4m3 GEMM stage writes a bf16 output -- declare out_dtype=torch.bfloat16 (got out_dtype={self.out_dtype}); the "
                    "quantized backward's gradients are bf16"
                )
            if self.mma_tile_k_bytes is None:
                raise ValueError(
                    f"{self.name}: mma_tile_k_bytes is REQUIRED on an e4m3 GEMM stage (32 or 64; the quantized backward passes 64 explicitly) -- the "
                    "MMA K width is never derived from the dtype, here or in the driver"
                )
            if self.mma_tile_k_bytes not in (32, 64):
                raise ValueError(f"{self.name}: mma_tile_k_bytes must be 32 or 64 (the tcgen05 MMA K widths), got {self.mma_tile_k_bytes!r}")
            a_major, b_major = self.majors
            if self.k % 16 and (a_major == "k" or b_major == "k"):
                # The TMA 16-byte contiguous-extent rule at 1 B/elem binds an operand's CONTIGUOUS axis (build_proj_gemm says the
                # same at compile()): K only when an operand is K-major -- a dgrad's A.  A wgrad (M-major A, N-major B) has no
                # K-contiguous operand, so its token count K = B*S is free: the ragged K tail is TMA zero-fill.
                which = " and ".join(f"{op} ({mj}-major)" for op, mj in (("A", a_major), ("B", b_major)) if mj == "k")
                raise ValueError(
                    f"{self.name}: an e4m3 GEMM needs K % 16 == 0 when K is an operand's contiguous axis (the TMA 16-byte rule at 1 byte per "
                    f"element, on {which}), got K={self.k}"
                )
        elif self.dtype in _ACT_DTYPES:
            if self.alpha:
                raise NotImplementedError(
                    f"{self.name}: alpha=True is the e4m3 stage's epilogue (descale_A * descale_B); a {self.dtype} stage has no scale to apply -- leave it False"
                )
            if self.out_dtype is not None:
                raise NotImplementedError(
                    f"{self.name}: out_dtype={self.out_dtype} is the e4m3 stage's declaration; a {self.dtype} stage writes its own dtype -- leave it None"
                )
            if self.mma_tile_k_bytes is not None:
                raise NotImplementedError(
                    f"{self.name}: mma_tile_k_bytes={self.mma_tile_k_bytes} is a knob of an 8-bit (e4m3) GEMM stage; a {self.dtype} stage issues one "
                    "MMA K width -- leave it None"
                )
        else:
            raise NotImplementedError(
                f"{self.name}: the backward GEMM drivers serve bf16 / fp16 (and e4m3 with alpha=True, out_dtype=torch.bfloat16), got {self.dtype}"
            )
        # The TMA 16-byte contiguous-extent rule falls on the MN-major operands (build_proj_gemm's
        # _check_mn_major_tma_rule would say the same at compile(); the block says it at declaration).
        elems16 = 16 // _itemsize(self.dtype)
        a_major, _b_major = self.majors
        if a_major == "m" and self.m % elems16:
            raise ValueError(
                f"{self.name}: A is M-major (a transposed operand), so M={self.m} must be a multiple of {elems16} ({self.dtype}: the TMA 16-byte rule)"
            )
        if self.n % elems16:
            raise ValueError(f"{self.name}: B is N-major, so N={self.n} must be a multiple of {elems16} ({self.dtype}: the TMA 16-byte rule)")

    def compile(self) -> None:
        """Build the plan: ``build_proj_gemm`` at the majors ``kind`` implies, ``mma_tile_k_bytes`` forwarded as declared
        (``None`` = the named config's own width; a block-scale stage resolved it at declaration), the block-scale declaration
        (``block_scale`` / ``w_dtype`` / ``block_size`` / ``sf_dtype``) forwarded as given."""
        from .kernels.proj_gemm import build_proj_gemm

        a_major, b_major = self.majors
        # `out_dtype=None, alpha=False, block_scale=False, w_dtype=None, block_size=32, sf_dtype=None` ARE the driver's defaults: a
        # bf16 / fp16 stage's -- and the per-tensor e4m3 stage's -- plan request is byte-identical to before.
        self.plan = build_proj_gemm(
            m=self.m,
            k=self.k,
            n=self.n,
            dtype=self.dtype,
            label=self.label,
            a_major=a_major,
            b_major=b_major,
            mma_tile_k_bytes=self.mma_tile_k_bytes,
            out_dtype=self.out_dtype,
            alpha=self.alpha,
            block_scale=self.block_scale,
            w_dtype=self.w_dtype,
            block_size=self.block_size,
            sf_dtype=self.sf_dtype,
        )

    def workspace_bytes(self) -> int:
        if self.plan is None:
            raise RuntimeError(f"{self.name}: call compile() before workspace_bytes()")
        return int(self.plan.workspace_bytes)

    def execute(
        self,
        dy_like: torch.Tensor,
        other: torch.Tensor,
        out: torch.Tensor,
        workspace: torch.Tensor,
        *,
        stream,
        alpha: Optional[torch.Tensor] = None,
        sf_a: Optional[torch.Tensor] = None,
        sf_b: Optional[torch.Tensor] = None,
    ) -> None:
        """``alpha`` (appended): the 1-element fp32 DEVICE view of the epilogue-scale slot -- a slot of the block's scalar
        block written on the launch stream by the quantize launch before this GEMM, or a plan-time constant -- required iff
        ``plan.has_alpha`` (the per-tensor e4m3 stage; a block-scale plan has none) and refused otherwise, both directions typed
        HERE before the driver's own check (Rule 1: never a silent 1.0, never a dropped value); the drivers bind it as the
        ``[1, 1, 1]`` scalar aux (a view).  ``sf_a`` / ``sf_b`` (appended): the PADDED F8_128x4 scale-factor blobs of ``dy_like``
        and ``other`` -- required iff ``plan.block_scale`` and refused otherwise, both directions typed here (never a silent unit
        scale); the block-scale drivers check their dtype, size and alignment under their own keywords before the launch."""
        from .kernels.proj_gemm import run_dgrad_gemm, run_dgrad_gemm_block_scale, run_wgrad_gemm, run_wgrad_gemm_block_scale

        if self.plan is None:
            raise RuntimeError(f"{self.name}: call compile() before execute()")
        if bool(self.plan.has_alpha) != (alpha is not None):
            raise ValueError(
                f"{self.name}: alpha "
                + (
                    "is required: this plan carries the e4m3 descale epilogue (alpha=True) and never assumes 1.0 -- pass the slot's 1-element fp32 view"
                    if self.plan.has_alpha
                    else "was given but this plan has no alpha epilogue (built with alpha=False); refusing to drop the value silently"
                )
            )
        for name, sf, of_what in (("sf_a", sf_a, "dy_like"), ("sf_b", sf_b, "other")):
            if bool(self.plan.block_scale) != (sf is not None):
                raise ValueError(
                    f"{self.name}: {name} "
                    + (
                        f"is required: this plan carries per-block scale factors (block_scale=True) and never assumes a unit scale -- pass the padded "
                        f"F8_128x4 scale-factor blob of {of_what}"
                        if self.plan.block_scale
                        else "was given but this plan has no block-scale dequant (built with block_scale=False); refusing to drop the blob silently"
                    )
                )
        if self.plan.block_scale:
            # Both operands K-major transposed artifacts with their blobs: the wgrad's `dy_like` is dQKVG^T [rows, T] (`sf_a` along T)
            # and `other` the caller's h^T [cols, T] (`sf_b`); the dgrad's `dy_like` is dQKVG [T, K] (`sf_a` along K) and `other` the
            # caller's W^T [N, K] (`sf_b`).  The drivers' keywords name the blobs in their own messages.
            if self.kind == "wgrad":
                run_wgrad_gemm_block_scale(self.plan, dy_like, other, out, workspace, sf_dy_t=sf_a, sf_x_t=sf_b, stream=stream)
            else:
                run_dgrad_gemm_block_scale(self.plan, dy_like, other, out, workspace, sf_dy=sf_a, sf_w_t=sf_b, stream=stream)
            return
        runner = run_wgrad_gemm if self.kind == "wgrad" else run_dgrad_gemm
        runner(self.plan, dy_like, other, out, workspace, stream=stream, alpha=alpha)


class _OutProjWgrad(_GemmStage):
    """(B1) ``dW_o = dY^T @ O_gated``. Contracts over TOKENS: ``K = B*S``.

    ``W_o`` is ``[d_model, H_q*D]`` and the forward computes ``O_gated @ W_o^T``,
    so the weight gradient is ``dY^T @ O_gated`` and lands in ``W_o``'s own
    layout -- no transpose on the way out. ``O_gated`` is not saved: B3 writes
    it as its optional third output (``has_og`` = ``need_dw_o``), so this stage
    runs AFTER B3 and reads the workspace ``o_gated`` slot.

    **Fusion status: this is the backward's filler, and ``fuse_wgrad_overlap``
    schedules it as one.** It depends on nothing the SDPA backward produces, so
    under the knob it is issued on the block's side stream right after B3 (fork
    event) and joined back at the end of ``execute`` -- overlapping the Q / K
    rebuild, the SDPA backward chain, the norm backward and both projection
    GEMMs that follow it on the launch stream.  A scheduling change only: the
    same launch, the same bytes, bitwise.  PDL (the DSL kernels' ``use_pdl`` and
    the GEMM's launch attribute) is the other half of that question and is not
    taken here.

    Forced tile at one split-K slice; a pinned split (``split_k >= 2``) would
    reduce in fixed order (module docstring, "Determinism").

    Under MXFP8 this stage stays per-tensor e4m3 (``block_scale=False``, the
    alpha epilogue): the out-projection side of the quantized backward carries
    one scale per tensor.
    """

    name = "out_proj_wgrad"
    kind = "wgrad"


class _OutProjDgrad(_GemmStage):
    """(B2) ``dO_gated = dY @ W_o``, contracting over ``d_model``.

    **Fusion status: per-head independent, therefore legal as an SDPA-backward
    prologue -- and deliberately NOT taken.** ``dO`` is read with two different
    tilings by the SDPA backward (dK/dV loop q-tiles for a fixed kv-tile; dQ
    loops kv-tiles for a fixed q-tile), so it is materialized either way and
    the prologue fusion saves nothing on the dK/dV side.

    Under MXFP8 this stage stays per-tensor e4m3 (``block_scale=False``, the
    alpha epilogue), like B1.
    """

    name = "out_proj_dgrad"
    kind = "dgrad"


class _QkvGateWgrad(_GemmStage):
    """(B7) ``dW_qkvg = dQKVG^T @ h``. Contracts over TOKENS: ``K = B*S``.

    ``W_qkvg`` is ``[N, d_model]`` and the forward computes ``h @ W_qkvg^T``, so
    the weight gradient is ``dQKVG^T @ h``: ``[N, d_model]``, the exact
    ``qkvg_offsets`` row layout the forward consumes, so a caller never
    re-slices. At 397B: ``17408 x 4096``. Same tiling family as B1 and nothing
    else in the block; reads ``saved.h`` as ``[T, d_model]``.

    **Fusion status:** consumed by nothing in the backward, so under
    ``fuse_wgrad_overlap`` it is issued on the block's side stream right after
    B5+B6 wrote the ``dqkvg`` bands (fork event), overlapping the ``dW_norm``
    reduce and the B8 dgrad on the launch stream, and joined back before
    ``execute`` returns.

    Under MXFP8 (``block_scale=True``) its A is the TRANSPOSED, block-quantized
    ``dQKVG^T [N, T]`` (K-major; one E8M0 scale per 32 tokens) and its B the
    caller's ``h^T [d_model, T]`` with ``h_t_sf`` -- both K-major, so the token
    axis is the contraction and ``T % 32 == 0`` is the rule of THIS stage alone
    (the data gradients are served at any ``T``); no alpha, bf16 out.
    """

    name = "qkv_gate_wgrad"
    kind = "wgrad"


class _QkvGateDgrad(_GemmStage):
    """(B8) ``dh = dQKVG @ W_qkvg``, contracting over N.

    The block's output gradient. Nothing downstream of it here.

    Under MXFP8 (``block_scale=True``) its A is the rowwise block-quantized
    ``dQKVG [T, N]`` (scales along N) and its B the caller's ``W_qkvg^T
    [d_model, N]`` with ``w_qkvg_t_sf``, quantized once per weight update; the
    contraction ``N = n_qkvg`` is a multiple of 256 at every served geometry, so
    the ``K % 32`` rule never binds here; no alpha, bf16 out.
    """

    name = "qkv_gate_dgrad"
    kind = "dgrad"


# ---------------------------------------------------------------------------
# 3a. The quantized backward's scalar + quantize stages
# ---------------------------------------------------------------------------


class _QuantizeGrad(_Stage):
    """A GRADIENT's per-tensor e4m3 quantization over ``[T, heads, D]`` -- dY (viewed ``[T, d_model / D, D]``), dO
    (``[T, H_q, D]``), dQKVG (``[T, N / D, D]``) -- with the scale derived ON DEVICE: no host readback, no allocation.

    Two launches at most, on the one launch stream (Rule 5):

    * the amax pass (``own_amax``): ``amax_slot = max |fp32(src)|`` as one int32 ``atomicMax`` per warp of non-negative fp32
      bit patterns (they order as int32, so the fold is order-free and bitwise the fp32 max); the slot was zeroed by the
      prologue launch's scalar-init job (:class:`_QuantPrologue`) at the top of the execute.  The block uses ``own_amax=False``
      everywhere: the dY amax is the prologue's partials job, the dO amax the gate backward's per-CTA partials (``amax_src=
      "partials"`` on both casts, which reduce them and publish the slot) -- same-address atomics serialise at the L2, a
      measured 4.3x on the standalone pass, so every producer stores partials instead;
    * the quantize launch.  ``grad_scaling="current"`` (``scale_src="amax"``): every CTA derives
      ``scale = 2**(floor(log2(448 / amax)) - FP8_GRAD_SCALE_MARGIN_LOG2)`` itself (``kernels/quantize.py::
      grad_scale_from_amax`` is the host mirror), casts ``dst = sat_e4m3(src * scale)``, and lane 0 of CTA 0 publishes
      ``scale_out``, ``descale_out = 1 / scale`` and the ``n_alpha`` GEMM epilogue products ``alpha_outs[i] = descale *
      alpha_consts[i]``.  ``"delayed"`` (``scale_src="given"``): the launch reads the caller's ``scale_in`` (the previous
      step's scale) instead and publishes the same slots from it -- the amax pass STILL runs, so ``quant_scalars()``
      reports this step's amax for the caller's next-step scale.

    The published pair satisfies ``amax * scale <= 448`` for the amax the pass read; a ``"delayed"`` block fed the
    ``"current"`` run's scales replays it bitwise (one code path, two writers of the scale).  Kernels:
    ``kernels/quantize.py`` (``run_amax`` / ``run_quantize``; any CuTe-DSL device with the e4m3 ``cvt``).
    """

    def __init__(
        self,
        geometry: GatedAttentionBlockGeometry,
        *,
        batch: int,
        seq_len: int,
        dtype_in: torch.dtype,
        heads: int,
        name: str,
        grad_scaling: str,
        n_alpha: int,
        own_amax: bool = True,
        margin_log2: int = FP8_GRAD_SCALE_MARGIN_LOG2,
        amax_src: str = "slot",
        persistent: bool = False,
    ) -> None:
        """``amax_src`` (appended): ``"slot"`` -- the amax is a pre-folded slot (this stage's own pass under ``own_amax``, the
        producer's fold otherwise); ``"partials"`` -- the amax is a producer's per-CTA partials (the prologue's dY job, the gate
        backward's dO fold), reduced in every CTA of the quantize launch and PUBLISHED into the slot (``own_amax`` must be False).
        ``persistent`` (appended): the quantize launch strides a capped grid over its row groups (``kernels/quantize.py``, the
        persistent cast) -- a knob the block leaves OFF: a persistent cast loses memory-level parallelism (MEASURED -19 % at
        S = 32K on Rubin) and the per-block reduce of a producer's <= SMs x 8 partials is cheap with four loads in flight."""
        if grad_scaling not in _GRAD_SCALING:
            raise ValueError(f"{name}: grad_scaling must be one of {_GRAD_SCALING}, got {grad_scaling!r}")
        if amax_src not in ("slot", "partials"):
            raise ValueError(f"{name}: amax_src must be 'slot' or 'partials', got {amax_src!r}")
        if amax_src == "partials" and own_amax:
            raise ValueError(f"{name}: amax_src='partials' takes the prologue launch's partials; own_amax must be False (there is no pass of its own)")
        self.name = name
        self.geom = geometry
        self.batch, self.seq_len = int(batch), int(seq_len)
        self.dtype_in = dtype_in
        self.heads = int(heads)
        self.grad_scaling = grad_scaling
        self.scale_src = "amax" if grad_scaling == "current" else "given"
        self.n_alpha = int(n_alpha)
        self.own_amax = bool(own_amax)
        self.margin_log2 = int(margin_log2)
        self.amax_src = amax_src
        self.persistent = bool(persistent)
        self._amax = None
        self._quant = None

    def check_support(self) -> None:
        from .kernels.quantize import validate_shape

        if self.dtype_in != torch.bfloat16:
            raise NotImplementedError(f"{self.name}: the quantized backward's gradients are bf16 before their e4m3 cast, got {self.dtype_in}")
        validate_shape(self.geom.d_head, _ELEMENTWISE_THREADS)

    def compile(self) -> None:
        from .kernels.quantize import compile_amax, compile_quantize

        d = self.geom.d_head
        if self.own_amax:
            self._amax = compile_amax(dtype_in=self.dtype_in, h=self.heads, d=d, threads_per_cta=_ELEMENTWISE_THREADS)
        # publish=True: BOTH recipes publish scale_out / descale (and the alphas) so quant_scalars() is complete under either and
        # a "delayed" block replays a "current" run bitwise; the kernel implies it under "amax" or n_alpha > 0 and needs it
        # spelled for a "given" launch without alphas (the dO quantize).
        self._quant = compile_quantize(
            dtype_in=self.dtype_in,
            h=self.heads,
            d=d,
            threads_per_cta=_ELEMENTWISE_THREADS,
            scale_src=self.scale_src,
            n_alpha=self.n_alpha,
            margin_log2=self.margin_log2,
            publish=True,
            # "partials" reduces the prologue's per-CTA maxima and publishes the amax; a "given" launch (the dO quantize, the delayed
            # recipe) reads no amax at all, a "current" slot launch the pre-folded slot
            amax_src=self.amax_src if self.amax_src == "partials" else ("slot" if self.scale_src == "amax" else "none"),
            persistent=self.persistent,
        )

    def moved_bytes(self) -> int:
        """HBM traffic of the stage: the amax pass's read (when it is this stage's) plus the quantize's read and 1-B write."""
        rows = self.batch * self.seq_len * self.heads * self.geom.d_head
        return rows * ((_itemsize(self.dtype_in) if self.own_amax else 0) + _itemsize(self.dtype_in) + 1)

    def execute(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        *,
        stream,
        amax_slot: torch.Tensor,
        scale_in: Optional[torch.Tensor] = None,
        scale_out: torch.Tensor,
        descale_out: torch.Tensor,
        alpha_consts: tuple = (),
        alpha_outs: tuple = (),
        partials: Optional[torch.Tensor] = None,
        n_partials: Optional[int] = None,
    ) -> None:
        """``src`` ``[T, heads, D]`` bf16 (compact, or a slab view), ``dst`` the e4m3 twin; ``amax_slot`` the block's ``amax_*``
        slot (filled here under ``own_amax``, by the producer otherwise -- or, under ``amax_src="partials"``, PUBLISHED here from
        ``partials`` / ``n_partials``, a producer's per-CTA maxima, REQUIRED then and refused otherwise); ``scale_in``
        the caller's scale -- REQUIRED under ``"delayed"``, REFUSED under ``"current"`` (Rule 1, both directions); ``scale_out`` /
        ``descale_out`` / ``alpha_outs`` the slots lane 0 of CTA 0 publishes, ``alpha_consts`` the plan-time constants they multiply."""
        from .kernels.quantize import run_amax, run_quantize

        if self._quant is None or (self.own_amax and self._amax is None):
            raise RuntimeError(f"{self.name}: call compile() before execute()")
        if self.scale_src == "amax" and scale_in is not None:
            raise ValueError(
                f"{self.name}: grad_scaling='current' derives the scale from the amax pass on device; a caller scale would be silently ignored (Rule 1)"
            )
        if self.scale_src == "given" and scale_in is None:
            raise ValueError(f"{self.name}: grad_scaling='delayed' reads the caller's scale; scale_in must be bound at execute (Rule 1: no silent fallback)")
        if self.amax_src == "partials" and (partials is None or n_partials is None):
            raise ValueError(f"{self.name}: amax_src='partials' reduces a producer's partials; partials and n_partials must be bound at execute (Rule 1)")
        if self.amax_src != "partials" and (partials is not None or n_partials is not None):
            raise ValueError(f"{self.name}: this stage reads the amax slot; passing partials / n_partials would silently ignore them (Rule 1)")
        if self.own_amax:
            run_amax(self._amax, src, amax_slot, stream=stream)
        if self.amax_src == "partials":
            # the slot is PUBLISHED (amax_out), never read; the scale is derived from the reduced partials or given
            run_quantize(
                self._quant,
                src,
                dst,
                scale_in,
                stream=stream,
                scale_out=scale_out,
                descale=descale_out,
                alpha_consts=tuple(alpha_consts),
                alpha_outs=tuple(alpha_outs),
                partials=partials,
                n_partials=int(n_partials),
                amax_out=amax_slot,
            )
        elif self.scale_src == "amax":
            run_quantize(
                self._quant,
                src,
                dst,
                None,
                stream=stream,
                amax=amax_slot,
                scale_out=scale_out,
                descale=descale_out,
                alpha_consts=tuple(alpha_consts),
                alpha_outs=tuple(alpha_outs),
            )
        else:
            run_quantize(
                self._quant,
                src,
                dst,
                scale_in,
                stream=stream,
                scale_out=scale_out,
                descale=descale_out,
                alpha_consts=tuple(alpha_consts),
                alpha_outs=tuple(alpha_outs),
            )


class _QuantPrologue(_Stage):
    """The quantized backward's FIRST launch -- four independent jobs behind one block-range dispatch
    (``kernels/fp8_bwd_fused.py::frost_fp8_bwd_prologue``; module docstring, "The quantized backward"):

    * the scalar init (one thread): ``slots[0:n] = 0``, ``descale_dp = 1 / scale_dp``, then the ``QuantSpec``'s ``n_consts``
      plan-time constants into their slots (``QUANT_CONST_SLOTS``, from ``const_slot0``) out of the launch's KERNEL ARGUMENTS
      (``kernels/quantize.py::init_scalars_body``, the standalone init kernel's body) -- the fp8 row's ``amax_dp`` (an
      ``atomicMax`` target) must be zero first, every other slot is stored here or published by a later launch, and nothing
      else in THIS launch touches a slot: the rebuild's and ``v8``'s static scales are kernel arguments too (the values of their
      slots), never slot reads.  The constants ride as runtime fp32 arguments (one artifact per slot layout, never one per
      value) because the launch that consumes them is the one writer whose ordering against every consumer is given: a device
      tensor filled at ``compile()`` sits on the stream that was ambient THEN, and an ``execute`` on another stream has no
      event connecting the two -- so every later launch (the quantize launches' ``alpha_consts`` and ``scale_o``, the fp8 row's
      constant scalars) reads a slot of the block instead;
    * the dY amax as PER-CTA PARTIALS (persistent, ``n_partials_for`` CTAs): one fp32 maximum per CTA, written unconditionally --
      no zeroed slot to order before it, which is what lets the pass share the first launch; the dY quantize (the next launch)
      reduces them and publishes ``amax_dy``;
    * the Q / K rebuild with the e4m3 epilogue: the TMA norm + RoPE body over the slab's PRE-norm bands, each lane rounding to
      bf16 FIRST and then casting at the forward's static ``scale_q`` / ``scale_k`` -- byte-identical to quantizing the bf16
      rebuild (the forward's own SDPA operands), with no bf16 ``recompute`` buffers written or carved;
    * ``v8`` straight from the slab's V band at ``scale_v`` (V's compaction).

    It replaces five launches (scalar init, amax dY, the bf16 rebuild, the q8 / k8 casts) plus the v8 cast with one.  The
    rebuild needs the TMA tiling (``_QkNormRope.resolve_impl() == "tma"``: ``tile_rows`` dividing ``h_q``, a multiple of ``h_kv`` and
    of the warps); a geometry only the LDG rebuild can tile is declined typed here, before any stage compiles.
    """

    name = "fp8_bwd_prologue"

    def __init__(
        self, geometry: GatedAttentionBlockGeometry, *, batch: int, seq_len: int, dtype: torch.dtype, n_slots: int, const_slot0: int = 0, n_consts: int = 0
    ) -> None:
        self.geom = geometry
        self.batch, self.seq_len = int(batch), int(seq_len)
        self.dtype = dtype
        self.n_slots = int(n_slots)
        self.const_slot0, self.n_consts = int(const_slot0), int(n_consts)
        # the forward's own rebuild stage, for its TMA tiling facts (resolve_impl / resolve_tile_rows); never compiled here
        self._rebuild = _QkNormRope(geometry, batch=batch, seq_len=seq_len, dtype=dtype, want_rstd=False)
        self._recipe = None

    def check_support(self) -> None:
        from .kernels.quantize import MAX_INIT_CONSTS, validate_shape

        if self.dtype != torch.bfloat16:
            raise NotImplementedError(f"{self.name}: the quantized backward's record is bf16, got {self.dtype}")
        if self.n_slots < 1:
            raise ValueError(f"{self.name}: the scalar block needs at least one fp32 slot, got n_slots={self.n_slots}")
        if self.const_slot0 < 0 or self.n_consts < 0 or self.const_slot0 + self.n_consts > self.n_slots:
            raise ValueError(
                f"{self.name}: the {self.n_consts} plan-time constants at slots [{self.const_slot0}, {self.const_slot0 + self.n_consts}) must lie "
                f"inside the {self.n_slots}-slot block"
            )
        if self.n_consts > MAX_INIT_CONSTS:
            raise ValueError(f"{self.name}: {self.n_consts} plan-time constants exceed the {MAX_INIT_CONSTS} kernel arguments the init job's ABI reserves")
        validate_shape(self.geom.d_head, _ELEMENTWISE_THREADS)
        self._rebuild.check_support()
        g = self.geom
        if self._rebuild.resolve_tile_rows() == 0:  # the GEOMETRY (device-free): no tile in 1..16 fits the head counts
            raise NotImplementedError(
                f"{self.name}: the quantized backward's fused prologue rebuilds Q / K with the TMA norm + RoPE kernel, whose tile must divide h_q={g.h_q}, "
                f"be a multiple of h_kv={g.h_kv} and spread over {_QK_NORM_ROPE_THREADS // 32} warps; this geometry tiles for the LDG kernel only "
                "(the bf16 backward serves it through that kernel; the quantized one does not yet)"
            )
        if self._rebuild.resolve_impl() != "tma":  # the ARCH (cp.async.bulk.tensor is SM90+); the block's Rubin gate names it first
            raise NotImplementedError(
                f"{self.name}: the fused prologue's Q / K rebuild is the TMA norm + RoPE kernel (SM90 or newer); this device cannot run it"
            )

    def compile(self) -> None:
        from .kernels.fp8_bwd_fused import compile_fp8_bwd_prologue

        g = self.geom
        self._recipe = compile_fp8_bwd_prologue(
            dtype=self.dtype,
            h_q=g.h_q,
            h_kv=g.h_kv,
            d_model=g.d_model,
            d=g.d_head,
            rope_dim=g.rope_dim,
            eps=g.qk_norm_eps,
            apply_norm=bool(g.qk_norm),
            n_slots=self.n_slots,
            tile_rows=self._rebuild.resolve_tile_rows(),
            stages=self._rebuild.stages,
            threads_per_cta=_ELEMENTWISE_THREADS,
            const_slot0=self.const_slot0,
            n_consts=self.n_consts,
        )

    @property
    def n_partials_cap(self) -> int:
        """The most partials one launch writes (SMs x the persistent cap) -- the partials region's length; needs ``compile()``."""
        if self._recipe is None:
            raise RuntimeError(f"{self.name}: call compile() before n_partials_cap")
        return int(self._recipe.n_ctas_cap)

    def n_partials(self) -> int:
        """The partials THIS declaration's launch writes (``prologue_grid``'s amax width): what the dY quantize reduces."""
        from .kernels.fp8_bwd_fused import prologue_grid

        if self._recipe is None:
            raise RuntimeError(f"{self.name}: call compile() before n_partials()")
        return int(prologue_grid(self._recipe, self.batch * self.seq_len)[0])

    def execute(
        self, *, slots, scale_dp, descale_dp_out, dy, partials, q_pre, k_pre, w_q, w_k, cos, sin, q8, k8, scale_q, scale_k, v, v8, scale_v, stream, consts=()
    ) -> int:
        """One launch; returns the partials written (``n_partials()``).  ``dy`` is the ``[T, d_model / D, D]`` view, ``q_pre`` /
        ``k_pre`` / ``v`` the slab bands, ``cos`` / ``sin`` ``[T, rope_dim]``, the e4m3 outputs compact; ``scale_q`` / ``scale_k`` /
        ``scale_v`` Python floats -- kernel ARGUMENTS of the rebuild / v8 jobs (the values the init job stores into their slots in this
        same launch; a slot read would race it); ``consts`` the ``n_consts`` plan-time constants as Python floats in ``QUANT_CONST_SLOTS``
        order -- the init job's arguments, stored into ``slots[const_slot0:]`` by this launch on ``stream``."""
        from .kernels.fp8_bwd_fused import run_fp8_bwd_prologue

        if self._recipe is None:
            raise RuntimeError(f"{self.name}: call compile() before execute()")
        return run_fp8_bwd_prologue(
            self._recipe,
            slots=slots,
            scale_dp=scale_dp,
            descale_dp_out=descale_dp_out,
            dy=dy,
            partials=partials,
            q=q_pre,
            k=k_pre,
            w_q=w_q,
            w_k=w_k,
            cos=cos,
            sin=sin,
            q8=q8,
            k8=k8,
            scale_q=scale_q,
            scale_k=scale_k,
            v=v,
            v8=v8,
            scale_v=scale_v,
            stream=stream,
            consts=tuple(consts),
        )


class _QuantEpilogue(_Stage):
    """The quantized backward's launch after the norm backward -- two independent jobs behind one block-range dispatch
    (``kernels/fp8_bwd_fused.py::frost_fp8_bwd_epilogue``): the fixed-order ``dW_norm`` reduce (``need_dw_norms``; per column the
    SAME fp32 chain as the standalone reduce, at one column per block) and the dQKVG quantize -- ``dqkvg8 = sat_e4m3(dqkvg *
    scale)`` with ``amax_dqkvg`` reduced in every cast block from TWO partials arrays, the gate backward's per-CTA ``max |dG|``
    (the GATE band) and the norm backward's per-CTA maxima over the Q / K / V bands, the scale derived from it (or the caller's
    under ``"delayed"``), and the cast blocks striding PERSISTENTLY over the row groups so the reduce is paid once per resident
    block; the first cast block publishes ``amax_dqkvg``, ``scale_dqkvg``, ``descale_dqkvg``, ``alpha_b7``, ``alpha_b8``.  Replaces
    three launches (the dqkvg amax pass, the reduce, the quantize) with one.
    """

    name = "fp8_bwd_epilogue"

    def __init__(
        self,
        geometry: GatedAttentionBlockGeometry,
        *,
        batch: int,
        seq_len: int,
        dtype: torch.dtype,
        want_dw: bool,
        grad_scaling: str,
        n_alpha: int,
        margin_log2: int = FP8_GRAD_SCALE_MARGIN_LOG2,
    ) -> None:
        """Record the declaration: ``want_dw`` = the dW_norm reduce job exists (``need_dw_norms``), ``grad_scaling`` picks the dqkvg
        quantize's scale source (``"current"``: from the reduced amax; ``"delayed"``: the caller's slot), ``n_alpha`` the alpha products
        the quantize publishes (``alpha_b7`` / ``alpha_b8``), ``margin_log2`` (appended) the "current" recipe's power-of-two headroom
        (``GatedAttentionBlockBwd.grad_scale_margin_log2``; the module constant by default, so the artifact of every existing caller is
        unchanged -- the margin is a compile-time constant of the fused epilogue's quantize arm, keyed like its other knobs).
        Validation is ``check_support``'s, the artifact ``compile``'s."""
        if grad_scaling not in _GRAD_SCALING:
            raise ValueError(f"{self.name}: grad_scaling must be one of {_GRAD_SCALING}, got {grad_scaling!r}")
        self.geom = geometry
        self.batch, self.seq_len = int(batch), int(seq_len)
        self.dtype = dtype
        self.want_dw = bool(want_dw)
        self.grad_scaling = grad_scaling
        self.scale_src = "amax" if grad_scaling == "current" else "given"
        self.n_alpha = int(n_alpha)
        self.margin_log2 = int(margin_log2)
        self._recipe = None

    def check_support(self) -> None:
        """Typed declines: the quantized backward's gradients are bf16 before their cast; a dW_norm job needs the norm; the quantize
        kernel's row geometry (``validate_shape``)."""
        from .kernels.quantize import validate_shape

        if self.dtype != torch.bfloat16:
            raise NotImplementedError(f"{self.name}: the quantized backward's gradients are bf16 before their e4m3 cast, got {self.dtype}")
        if self.want_dw and not self.geom.qk_norm:
            raise ValueError(f"{self.name}: geometry.qk_norm=False computes no RMSNorm and therefore no dW_norm; need_dw_norms must be False")
        validate_shape(self.geom.d_head, _ELEMENTWISE_THREADS)

    def compile(self) -> None:
        """Build the fused epilogue's artifact (``kernels/fp8_bwd_fused.py``: the dW_norm reduce + the dqkvg quantize, by block range)."""
        from .kernels.fp8_bwd_fused import compile_fp8_bwd_epilogue

        self._recipe = compile_fp8_bwd_epilogue(
            dtype=self.dtype,
            n_cols=self.geom.n_qkvg,
            d=self.geom.d_head,
            want_dw=self.want_dw,
            scale_src=self.scale_src,
            n_alpha=self.n_alpha,
            margin_log2=self.margin_log2,
            threads_per_cta=_ELEMENTWISE_THREADS,
        )

    def execute(
        self,
        *,
        plane_q,
        plane_k,
        dw_q_norm,
        dw_k_norm,
        src,
        dst,
        scale_in,
        partials_gate,
        n_gate,
        partials_bands,
        n_bands,
        amax_out,
        scale_out,
        descale_out,
        alpha_consts,
        alpha_outs,
        stream,
    ) -> None:
        """``plane_*`` / ``dw_*_norm`` under ``want_dw`` (``None`` otherwise); ``src`` the ``[T, N / D, D]`` view of ``dqkvg``, ``dst`` its
        e4m3 twin; ``scale_in`` REQUIRED under ``"delayed"`` and refused under ``"current"`` (Rule 1); ``partials_gate`` / ``n_gate`` the gate
        backward's dG partials and the count it wrote, ``partials_bands`` / ``n_bands`` the norm backward's, ``amax_out`` the ``amax_dqkvg``
        slot the reduced amax is published to -- under BOTH recipes (the kernel's host wrapper checks every one)."""
        from .kernels.fp8_bwd_fused import run_fp8_bwd_epilogue

        if self._recipe is None:
            raise RuntimeError(f"{self.name}: call compile() before execute()")
        if self.scale_src == "amax" and scale_in is not None:
            raise ValueError(
                f"{self.name}: grad_scaling='current' derives the scale from the amax slot on device; a caller scale would be silently ignored (Rule 1)"
            )
        if self.scale_src == "given" and scale_in is None:
            raise ValueError(f"{self.name}: grad_scaling='delayed' reads the caller's scale; scale_in must be bound at execute (Rule 1: no silent fallback)")
        run_fp8_bwd_epilogue(
            self._recipe,
            plane_q=plane_q,
            plane_k=plane_k,
            dw_q=dw_q_norm,
            dw_k=dw_k_norm,
            src=src,
            dst=dst,
            scale=scale_in,
            partials_gate=partials_gate,
            n_gate=int(n_gate),
            partials_bands=partials_bands,
            n_bands=int(n_bands),
            amax_out=amax_out,
            scale_out=scale_out,
            descale=descale_out,
            alpha_consts=tuple(alpha_consts),
            alpha_outs=tuple(alpha_outs),
            stream=stream,
        )


class _MxQuantPrologue(_Stage):
    """The MXFP8 backward's FIRST launch -- four independent jobs behind one block-range dispatch
    (``kernels/mxfp8_bwd_fused.py::frost_mxfp8_bwd_prologue``; module docstring, "The MXFP8 backward"), the MXFP8 twin of
    :class:`_QuantPrologue`:

    * the scalar init (one thread): ``slots[0:n] = 0``, then the ``MxQuantSpec``'s ``n_consts`` plan-time constants into their slots
      (``QUANT_CONST_SLOTS``, from ``const_slot0``) out of the launch's KERNEL ARGUMENTS -- the standalone init launch's body WITHOUT the
      ``1 / scale_dp`` division (``descale_dp=False``: the MXFP8 row has no dP scalar; :class:`_InitScalars` is that job's one-launch
      form).  Nothing else in this launch touches a slot;
    * the dY amax as PER-CTA PARTIALS (persistent, ``n_partials()`` CTAs): one fp32 maximum per CTA, written unconditionally; the dY
      quantize (the next launch) reduces them and publishes ``amax_dy`` (:class:`_AmaxPartials` is the job's one-launch form);
    * the Q / K rebuild with the MX epilogue: the TMA norm + RoPE body over the slab's PRE-norm bands on a ONE-head x 32-TOKEN tile
      (``qk_norm_rope_tma.qk_norm_rope_tma_mx_body``), each lane rounding to bf16 FIRST and the tile then block-quantized twice --
      ROWWISE (``q8 / sf_q``, ``k8 / sf_k``: 32-element blocks along D) and COLUMNWISE (``q_T8 / sf_q_T``, ``k_T8 / sf_k_T``: 32-token blocks
      along S, the D-plane-major blob) -- with the standalone quantizer's exact arm (``abs_max_tree -> e8m0 -> x * rcp ->
      fp32_to_fp8_pack``), so every payload and blob is byte-identical to quantizing the bf16 rebuild and ``q8 / sf_q`` / ``k8 / sf_k`` stay
      the forward's own bytes; no bf16 ``recompute`` / ``recompute_k`` buffer is written or carved (:attr:`arm`, the carve's key);
    * ``v8 / sf_v`` straight from the slab's V band (V's compaction, ROWWISE: the row's dP operand), the standalone rowwise quantize's
      body as a job.

    It replaces eight launches (the scalar init, the dY amax partials, the bf16 rebuild, the four Q / K block quantizes, the V quantize)
    with one, and the three slow columnwise SDPA-layout quantizes (the 2-byte-store arm) leave the chain.  The MX tile needs TMA (SM90
    or newer) and the warp-per-row geometry -- ``d_head = 256``, whole 32-token tiles over the CTA's warps, the RoPE rules -- typed by
    ``validate_mx_shape``; unlike the fp8 prologue it has no ``tile_rows`` / head-count rule (the tile is one head x 32 tokens for any
    ``h_q`` / ``h_kv``).

    **The rate the arm reads at, honestly.**  Measured on Rubin cc 10.7 (212 SMs, locked clocks, CUPTI device time, the 397B geometry at
    S = 8K): this launch moves its 354 MiB in 0.099 ms (3.7 TB/s on its own bytes) against the eight standalone launches' 0.184 ms (+85 %
    on the set) -- at 60 % of the per-tensor fp8 prologue's rate on ITS bytes (6.3 TB/s): the two-pass token tile is resident 5-6 CTAs per
    SM against the shipped one-token x 16-head tile's 14.  The alternative shape -- the rowwise MX epilogue on the shipped tile plus the
    bf16 TMA store, and one dual-axis launch over the bf16 buffers for ``q_T / k_T`` (the ``recompute`` / ``recompute_k`` carve kept) -- is
    the open follow-up; :attr:`arm` is the plan-time fact the carve is keyed on, so that shape plugs in without re-opening the carve.
    """

    name = "mxfp8_bwd_prologue"
    # The recipe's shape, the carve's key (`_plan_bwd_workspace(mx_prologue_arm=)`): the MX epilogue writes the four Q / K payloads out
    # of registers -- no bf16 rebuild buffer.
    arm: str = MX_PROLOGUE_ARM_MX_EPILOGUE

    def __init__(
        self, geometry: GatedAttentionBlockGeometry, *, batch: int, seq_len: int, dtype: torch.dtype, n_slots: int, const_slot0: int = 0, n_consts: int = 0
    ) -> None:
        """``n_slots`` fp32 slots of the scalar block (zeroed by the init job), ``n_consts`` plan-time constants stored from the launch's
        arguments at slots ``[const_slot0, const_slot0 + n_consts)`` (the ``QUANT_CONST_SLOTS`` tail) -- exactly ``_InitScalars``'s facts."""
        self.geom = geometry
        self.batch, self.seq_len = int(batch), int(seq_len)
        self.dtype = dtype
        self.n_slots = int(n_slots)
        self.const_slot0, self.n_consts = int(const_slot0), int(n_consts)
        self._recipe = None

    def check_support(self) -> None:
        """Typed declines, geometry first (device-free, so they run on any host): a bf16 record; the slot arithmetic (at least one slot,
        the constants inside the block and within the init job's ABI); ``d_model % d_head == 0`` (dY is viewed ``[T, d_model / D, D]``);
        the amax job's row geometry (``quantize.validate_shape``), the MX token-tile arm's (``validate_mx_shape``: ``d_head = 256``, the
        32-token tile over the CTA's warps, the RoPE rules) and the v8 job's (``quantize_mxfp8.validate_shape(.., "row")``); then the
        arch: the TMA ring needs SM90 or newer (the block's own Rubin gate names the part first)."""
        from cudnn.frost.device import ambient_device, compute_capability

        from .kernels.mxfp8_bwd_fused import PROLOGUE_THREADS
        from .kernels.qk_norm_rope_tma import validate_mx_shape
        from .kernels.quantize import MAX_INIT_CONSTS, validate_shape
        from .kernels.quantize_mxfp8 import validate_shape as validate_mx_quant_shape

        if self.dtype != torch.bfloat16:
            raise NotImplementedError(f"{self.name}: the MXFP8 backward's record is bf16, got {self.dtype}")
        if self.n_slots < 1:
            raise ValueError(f"{self.name}: the scalar block needs at least one fp32 slot, got n_slots={self.n_slots}")
        if self.const_slot0 < 0 or self.n_consts < 0 or self.const_slot0 + self.n_consts > self.n_slots:
            raise ValueError(
                f"{self.name}: the {self.n_consts} plan-time constants at slots [{self.const_slot0}, {self.const_slot0 + self.n_consts}) must lie "
                f"inside the {self.n_slots}-slot block"
            )
        if self.n_consts > MAX_INIT_CONSTS:
            raise ValueError(f"{self.name}: {self.n_consts} plan-time constants exceed the {MAX_INIT_CONSTS} kernel arguments the init job's ABI reserves")
        g = self.geom
        if g.d_model % g.d_head:
            raise ValueError(f"{self.name}: d_model={g.d_model} must be a multiple of d_head={g.d_head} (dY is quantized through a [T, d_model / D, D] view)")
        validate_shape(g.d_head, PROLOGUE_THREADS)  # the amax job (the quantize layout's row geometry)
        validate_mx_shape(g.d_head, g.rope_dim, PROLOGUE_THREADS)  # the rebuild's MX token-tile arm
        validate_mx_quant_shape(g.d_head, PROLOGUE_THREADS, "row")  # the v8 job (the standalone rowwise quantize's layout)
        major, _minor = compute_capability(ambient_device())
        if major < 9:
            raise NotImplementedError(
                f"{self.name}: the fused prologue's Q / K rebuild stages its tiles through a TMA ring (cp.async.bulk.tensor: SM90 or newer); "
                "this device cannot run it"
            )

    def compile(self) -> None:
        """Build the fused prologue's artifact (``kernels/mxfp8_bwd_fused.py``: the four jobs by block range; the rebuild job's persistent
        cap is the MX arm's RESIDENCY, the amax job's its ``SMs x 8``)."""
        from .kernels.mxfp8_bwd_fused import PROLOGUE_THREADS, compile_mxfp8_bwd_prologue

        g = self.geom
        self._recipe = compile_mxfp8_bwd_prologue(
            dtype=self.dtype,
            h_q=g.h_q,
            h_kv=g.h_kv,
            d_model=g.d_model,
            d=g.d_head,
            rope_dim=g.rope_dim,
            eps=g.qk_norm_eps,
            apply_norm=bool(g.qk_norm),
            n_slots=self.n_slots,
            threads_per_cta=PROLOGUE_THREADS,
            const_slot0=self.const_slot0,
            n_consts=self.n_consts,
        )

    @property
    def n_partials_cap(self) -> int:
        """The most partials one launch writes (SMs x the amax job's persistent cap) -- the partials region's length; needs ``compile()``."""
        if self._recipe is None:
            raise RuntimeError(f"{self.name}: call compile() before n_partials_cap")
        return int(self._recipe.n_ctas_cap)

    def n_partials(self) -> int:
        """The partials THIS declaration's launch writes (``prologue_grid``'s amax width): what the dY quantize reduces."""
        from .kernels.mxfp8_bwd_fused import prologue_grid

        if self._recipe is None:
            raise RuntimeError(f"{self.name}: call compile() before n_partials()")
        return int(prologue_grid(self._recipe, self.batch, self.seq_len)[0])

    def execute(
        self,
        *,
        slots,
        dy,
        partials,
        q_pre,
        k_pre,
        w_q,
        w_k,
        cos,
        sin,
        q8,
        sf_q,
        q_T8,
        sf_q_T,
        k8,
        sf_k,
        k_T8,
        sf_k_T,
        v,
        v8,
        sf_v,
        stream,
        consts=(),
    ) -> int:
        """One launch; returns the partials written (``n_partials()``).  ``dy`` the ``[T, d_model / D, D]`` view, ``q_pre`` / ``k_pre`` /
        ``v`` the slab bands (strided ``[T, H, D]``), the norm weights both ``None`` on a RoPE-only geometry, ``cos`` / ``sin`` ``[T,
        rope_dim]``, the five e4m3 payloads compact ``[T, H, D]`` and their five SDPA-layout blobs flat; ``consts`` the ``n_consts``
        plan-time constants as Python floats in ``QUANT_CONST_SLOTS`` order -- the init job's arguments, stored into
        ``slots[const_slot0:]`` by this launch on ``stream``.  Host checks only, no allocation (the kernel's host wrapper)."""
        from .kernels.mxfp8_bwd_fused import run_mxfp8_bwd_prologue

        if self._recipe is None:
            raise RuntimeError(f"{self.name}: call compile() before execute()")
        return run_mxfp8_bwd_prologue(
            self._recipe,
            slots=slots,
            dy=dy,
            partials=partials,
            q=q_pre,
            k=k_pre,
            w_q=w_q,
            w_k=w_k,
            cos=cos,
            sin=sin,
            q8=q8,
            sf_q=sf_q,
            q_T8=q_T8,
            sf_q_T=sf_q_T,
            k8=k8,
            sf_k=sf_k,
            k_T8=k_T8,
            sf_k_T=sf_k_T,
            v=v,
            v8=v8,
            sf_v=sf_v,
            batch=self.batch,
            seq_len=self.seq_len,
            stream=stream,
            consts=tuple(consts),
        )


class _MxQuantEpilogue(_Stage):
    """The MXFP8 backward's launch after the norm backward -- two independent jobs behind one block-range dispatch at 256 threads
    (``kernels/mxfp8_bwd_fused.py::frost_mxfp8_bwd_epilogue``), the MXFP8 twin of :class:`_QuantEpilogue`: the fixed-order ``dW_norm``
    reduce (``want_dw`` = ``need_dw_norms``; two columns per block at ``lanes = REDUCE_LANES`` -- the standalone reduce's per-column chain
    never depends on the columns per block, so the sum is bitwise the standalone launch's) and the DUAL-AXIS dQKVG cast -- from ONE read
    of the bf16 dqkvg slab the GEMM-canonical rowwise ``dqkvg8 [T, N]`` + ``sf_dqkvg`` (``want_row`` = ``need_dh``: B8's A) and the TRANSPOSED
    ``dqkvg_t8 [N, T]`` + ``sf_dqkvg_t`` (``want_col`` = ``need_dw_qkvg``: B7's A), each half folded out of the artifact when its gradient is
    not requested, the standalone quantizer's exact arm either way.  Replaces three launches (the reduce, the two canonical quantizes)
    with one; the block builds it iff at least one of its three jobs exists (a RoPE-only block keeps it for the cast; a block wanting
    none of ``dW_norm`` / ``dh`` / ``dW_qkvg`` has no epilogue).  ``want_col`` keeps the block's ``B*S % 32 == 0`` rule (the transposed
    store writes whole 32-token blocks).
    """

    name = "mxfp8_bwd_epilogue"

    def __init__(
        self, geometry: GatedAttentionBlockGeometry, *, batch: int, seq_len: int, dtype: torch.dtype, want_dw: bool, want_row: bool, want_col: bool
    ) -> None:
        """Record the declaration: ``want_dw`` = the dW_norm reduce job, ``want_row`` / ``want_col`` = the two halves of the dual-axis cast
        (the rowwise canonical ``dqkvg8`` for the dgrad, the transposed ``dqkvg_t8`` for the wgrad).  Validation is ``check_support``'s, the
        artifact ``compile``'s."""
        self.geom = geometry
        self.batch, self.seq_len = int(batch), int(seq_len)
        self.dtype = dtype
        self.want_dw, self.want_row, self.want_col = bool(want_dw), bool(want_row), bool(want_col)
        self._recipe = None

    def check_support(self) -> None:
        """Typed declines: bf16 gradients before their cast; a dW_norm job needs the norm; at least one job; ``n_qkvg % d_head == 0`` (the
        slab is quantized through a ``[T, N / D, D]`` view); the dual-axis body's row geometry (``validate_dual_shape``); the transposed
        half's whole-32-token-block rule on ``B*S`` (the block's ``check_support`` names it first, with the fixes)."""
        from .kernels.mxfp8_bwd_fused import EPILOGUE_THREADS
        from .kernels.quantize_mxfp8 import validate_dual_shape

        if self.dtype != torch.bfloat16:
            raise NotImplementedError(f"{self.name}: the MXFP8 backward's gradients are bf16 before their e4m3 cast, got {self.dtype}")
        if self.want_dw and not self.geom.qk_norm:
            raise ValueError(f"{self.name}: geometry.qk_norm=False computes no RMSNorm and therefore no dW_norm; need_dw_norms must be False")
        if not (self.want_dw or self.want_row or self.want_col):
            raise ValueError(
                f"{self.name}: nothing to launch (no dW_norm reduce, no rowwise and no transposed dQKVG cast) -- the block builds no epilogue then"
            )
        g = self.geom
        if g.n_qkvg % g.d_head:
            raise ValueError(f"{self.name}: n_qkvg={g.n_qkvg} must be a multiple of d_head={g.d_head} (dQKVG is quantized through a [T, N / D, D] view)")
        validate_dual_shape(g.d_head, EPILOGUE_THREADS)
        if self.want_col and (self.batch * self.seq_len) % MXFP8_BLOCK_SIZE:
            raise ValueError(
                f"{self.name}: the transposed dQKVG cast writes whole {MXFP8_BLOCK_SIZE}-token blocks along T = B*S = {self.batch * self.seq_len}; "
                "B*S must be a multiple of it (or need_dw_qkvg=False)"
            )

    def compile(self) -> None:
        """Build the fused epilogue's artifact (``kernels/mxfp8_bwd_fused.py``: the reduce arm and the two cast halves by block range, the
        halves not wanted folded out)."""
        from .kernels.mxfp8_bwd_fused import EPILOGUE_THREADS, compile_mxfp8_bwd_epilogue

        self._recipe = compile_mxfp8_bwd_epilogue(
            dtype=self.dtype,
            n_cols=self.geom.n_qkvg,
            d=self.geom.d_head,
            want_dw=self.want_dw,
            want_row=self.want_row,
            want_col=self.want_col,
            threads_per_cta=EPILOGUE_THREADS,
        )

    def execute(self, *, plane_q, plane_k, dw_q_norm, dw_k_norm, src, dst, sf, dst_t, sf_t, stream) -> None:
        """``plane_*`` / ``dw_*_norm`` under ``want_dw`` (``None`` otherwise); ``src`` the ``[T, N / D, D]`` view of ``dqkvg``; ``dst`` / ``sf``
        (the compact e4m3 ``[T, N / D, D]`` twin and its canonical blob over ``(T, N)``) under ``want_row``, ``dst_t`` / ``sf_t`` (the
        contiguous e4m3 ``[N, T]`` matrix and its canonical blob over ``(N, T)``) under ``want_col`` -- each pair REQUIRED when its half is
        traced and refused otherwise (Rule 1, the kernel's host wrapper)."""
        from .kernels.mxfp8_bwd_fused import run_mxfp8_bwd_epilogue

        if self._recipe is None:
            raise RuntimeError(f"{self.name}: call compile() before execute()")
        run_mxfp8_bwd_epilogue(
            self._recipe,
            plane_q=plane_q,
            plane_k=plane_k,
            dw_q=dw_q_norm,
            dw_k=dw_k_norm,
            src=src,
            dst=dst,
            sf=sf,
            dst_t=dst_t,
            sf_t=sf_t,
            stream=stream,
        )


class _InitScalars(_Stage):
    """The scalar block's init as a STANDALONE launch -- ``slots[:] = 0``, then the
    ``MxQuantSpec``'s plan-time constants into their slots (``QUANT_CONST_SLOTS``, from ``const_slot0``) out of the launch's
    KERNEL ARGUMENTS (``kernels/quantize.py::compile_init_scalars(descale_dp=False)``: the body WITHOUT the ``1 / scale_dp``
    division -- the MXFP8 SDPA row has no dP scalar, so no ``scale_dp`` is taken and no ``descale_dp`` written).  The per-tensor
    fp8 backward runs the same body as the fused prologue's block 0 (:class:`_QuantPrologue`), and so does the MXFP8 backward
    (:class:`_MxQuantPrologue`, whose first job this is); this class is the job's ONE-LAUNCH form -- the fused module's tests pin the
    prologue's slots against it -- and no block builds it today.  Ordered on the launch stream before every consumer of a slot --
    B3's ``scale_o``, the dY quantize's alpha factors, the readers of ``quant_scalars()``; every slot the MXFP8 arm does not use (the
    dO / dQKVG / dP scalars, the per-tensor static scales) reads 0.0 after it, by construction.
    """

    name = "init_scalars"

    def __init__(self, *, n_slots: int, const_slot0: int = 0, n_consts: int = 0) -> None:
        """``n_slots`` fp32 slots to zero; ``n_consts`` plan-time constants stored from the launch's arguments at slots
        ``[const_slot0, const_slot0 + n_consts)`` (the ``QUANT_CONST_SLOTS`` tail)."""
        self.n_slots = int(n_slots)
        self.const_slot0, self.n_consts = int(const_slot0), int(n_consts)
        self._recipe = None

    def check_support(self) -> None:
        """Typed declines: at least one slot, the constants inside the block, and no more constants than the init launch's ABI
        reserves kernel arguments for (``MAX_INIT_CONSTS``)."""
        from .kernels.quantize import MAX_INIT_CONSTS

        if self.n_slots < 1:
            raise ValueError(f"{self.name}: the scalar block needs at least one fp32 slot, got n_slots={self.n_slots}")
        if self.const_slot0 < 0 or self.n_consts < 0 or self.const_slot0 + self.n_consts > self.n_slots:
            raise ValueError(
                f"{self.name}: the {self.n_consts} plan-time constants at slots [{self.const_slot0}, {self.const_slot0 + self.n_consts}) must lie "
                f"inside the {self.n_slots}-slot block"
            )
        if self.n_consts > MAX_INIT_CONSTS:
            raise ValueError(f"{self.name}: {self.n_consts} plan-time constants exceed the {MAX_INIT_CONSTS} kernel arguments the init launch's ABI reserves")

    def compile(self) -> None:
        """Build the scalar-init artifact WITHOUT the ``1 / scale_dp`` division (``descale_dp=False``: the MXFP8 row takes no dP
        scalar); the recipe is keyed on it, so the fp8 chain's arm is untouched."""
        from .kernels.quantize import compile_init_scalars

        # descale_dp=False: the arm WITHOUT the reciprocal (the MXFP8 row takes no dP scalar); the artifact is keyed on it.
        self._recipe = compile_init_scalars(self.n_slots, self.const_slot0, self.n_consts, descale_dp=False)

    def execute(self, *, slots, consts=(), stream) -> None:
        """``slots`` the contiguous fp32 ``[n_slots]`` view of the scalar block; ``consts`` the ``n_consts`` plan-time constants as
        Python floats in ``QUANT_CONST_SLOTS`` order -- the launch's arguments, stored into ``slots[const_slot0:]`` on ``stream``."""
        from .kernels.quantize import run_init_scalars

        if self._recipe is None:
            raise RuntimeError(f"{self.name}: call compile() before execute()")
        run_init_scalars(self._recipe, slots, consts=tuple(consts), stream=stream)


class _AmaxPartials(_Stage):
    """The dY amax as PER-CTA PARTIALS, a STANDALONE launch (``kernels/quantize.py::compile_amax_partials``): ``partials[cta] =
    max |dY|`` over the row groups a persistent CTA strode (one plain store per CTA, no atomic, no slot to zero first), which the
    dY quantize (``_QuantizeGrad(amax_src="partials")``) reduces and publishes as ``amax_dy``.  The per-tensor fp8 backward runs
    this job inside its fused prologue; the MXFP8 backward -- whose rebuild writes bf16 and whose other casts are block-scaled, so
    the prologue's jobs do not apply -- launches it on its own (module docstring, "The MXFP8 backward").
    """

    name = "amax_dy_partials"

    def __init__(self, geometry: GatedAttentionBlockGeometry, *, batch: int, seq_len: int, dtype_in: torch.dtype, heads: int, name: str) -> None:
        self.name = name
        self.geom = geometry
        self.batch, self.seq_len = int(batch), int(seq_len)
        self.dtype_in = dtype_in
        self.heads = int(heads)
        self._recipe = None

    def check_support(self) -> None:
        """Typed declines: a bf16 source (the quantized backward's gradients before their cast) and the amax kernel's row geometry."""
        from .kernels.quantize import validate_shape

        if self.dtype_in != torch.bfloat16:
            raise NotImplementedError(f"{self.name}: the quantized backward's gradients are bf16 before their cast, got {self.dtype_in}")
        validate_shape(self.geom.d_head, _ELEMENTWISE_THREADS)

    def compile(self) -> None:
        """Build the standalone per-CTA amax partials artifact (``kernels/quantize.py``: one plain store per CTA of a persistent grid)."""
        from .kernels.quantize import compile_amax_partials

        self._recipe = compile_amax_partials(dtype_in=self.dtype_in, h=self.heads, d=self.geom.d_head, threads_per_cta=_ELEMENTWISE_THREADS)

    @property
    def n_partials_cap(self) -> int:
        """The most partials one launch writes (SMs x the persistent cap) -- the partials region's length; needs ``compile()``."""
        if self._recipe is None:
            raise RuntimeError(f"{self.name}: call compile() before n_partials_cap")
        return int(self._recipe.n_ctas_cap)

    def n_partials(self) -> int:
        """The partials THIS declaration's launch writes (its grid): what the dY quantize reduces."""
        from .kernels.quantize import n_partials_for

        if self._recipe is None:
            raise RuntimeError(f"{self.name}: call compile() before n_partials()")
        return int(n_partials_for(self._recipe, self.batch * self.seq_len))

    def moved_bytes(self) -> int:
        """HBM traffic of the launch: one read of the bf16 source."""
        return self.batch * self.seq_len * self.heads * self.geom.d_head * _itemsize(self.dtype_in)

    def execute(self, src: torch.Tensor, partials: torch.Tensor, *, stream) -> int:
        """``src`` the ``[T, heads, D]`` bf16 view (dY viewed ``[T, d_model / D, D]``), ``partials`` the fp32 ``amax_partials`` region;
        returns the partials written (``n_partials()``), every one overwritten."""
        from .kernels.quantize import run_amax_partials

        if self._recipe is None:
            raise RuntimeError(f"{self.name}: call compile() before execute()")
        return int(run_amax_partials(self._recipe, src, partials, stream=stream))


class _SigmoidGateBwd(_Stage):
    """(B3) gate backward, elementwise over ``[T, H_q, D]``::

        s       = sigmoid(GATE)
        dO      = dO_gated * s                      (IN PLACE over dO_gated)
        dG      = dO_gated * O * s * (1 - s)        (-> the GATE band of dqkvg)
        O_gated = O * s                             (optional third output, for B1 -- has_og = need_dw_o)

    Three input reads, two (or three) writes, one pass. ``s`` in fp32 regardless
    of storage dtype; ``s * (1 - s)`` in bf16 loses bits exactly where the gate
    saturates and the gradient matters least -- but it is free to do right.

    ``dG`` is written straight into the ``dqkvg`` slab at the GATE offset and
    never touched again: B5+B6 write only the Q, K and V bands. GATE is read as
    a strided column band of ``saved.proj_slab`` -- every operand carries its own
    token stride, so no repack anywhere.

    **Hazard, mirroring the forward's stage (5):** ``O`` here is the SDPA's
    *substituted* output (``saved.o``). For a dead row it is an exact zero,
    which correctly makes ``dG`` zero; if a future fused epilogue ever hands
    this stage pre-substitution residue instead, ``dG`` becomes NaN for rows
    whose forward output was fine. P0 declines padding, so the only dead rows
    would be mask-made ones, and with ``S_q == S_kv`` a causal / windowed row
    always keeps its diagonal (``window_left == 0`` is declined first).

    **Fusion status: the SDPA backward's ``dot_do_o`` pre-pass is folded in HERE
    under ``fuse_gate_bwd``** (``want_delta``): a fourth output ``delta[b, h, q] =
    rowsum(dO * O)`` in the chain's own order over the bf16-rounded ``dO`` this
    kernel stores, so the adapter (``external_delta=True``) skips its first launch
    and its second read of ``O`` and ``dO``; bitwise the unfused chain.  The other
    direction -- folding this kernel into the SDPA backward's prologue -- stays
    untaken: ``dO`` is read under two tilings there (``_OutProjDgrad``).

    **The quantized backward's arm** (``og_fp8`` / ``want_amax_do``, appended): the third output is the e4m3 ``og8 =
    sat_e4m3(bf16(O * s) * scale_o)`` -- the bf16 ROUNDING first, so ``og8`` is bitwise the forward's own quantize of the
    gated O -- with ``scale_o`` read in-kernel, and the kernel folds ``max |bf16(dO)|`` over the very dO words it stores (live
    rows only) into PER-CTA PARTIALS (``amax_do``: one plain store per CTA of a persistent grid, no atomic -- the slot form
    serialised at the L2, 0.72 ms of the kernel's 1.1 ms at S = 32K on Rubin), which the dO quantize reduces and publishes, so
    it needs no amax pass of its own.  ``want_delta`` is always on there: the delta is the fp8 SDPA row's external delta.
    ``want_amax_dg`` (appended): the same fold of the stored ``dG`` words into a second partials array (``amax_dg``), the GATE
    band's half of ``amax_dqkvg``, which the fused epilogue reduces with the norm backward's band partials -- the dqkvg amax
    pass is gone.

    **The dY descale arm** (``want_dy_descale``, appended; the fp4 weight modes' backward under an NVFP4 ``W_o``): the
    out-projection dgrad feeding this stage then reads the two-level NVFP4 cast of ``scale_dy x dY``, so its ``dO_gated`` is
    ``scale_dy x`` the true value; the kernel reads ``descale_dy`` from its slot (the pattern of ``scale_o``) and multiplies
    ``dO_gated`` by it before EVERY use -- ``dO``, ``dG`` and the delta (``og8``'s operand is ``O``, untouched) -- exact for a power
    of two.  OFF on every other recipe (byte-identical artifacts, pinned by the recipe-cache key); ``execute(descale_dy=)`` is then
    REQUIRED and otherwise refused (Rule 1, both ways, by the kernel's host wrapper).

    Kernel: ``kernels/sigmoid_gate_bwd.py`` (plain LDG/STG, any CuTe-DSL device).
    """

    name = "sigmoid_gate_bwd"

    def __init__(
        self,
        geometry: GatedAttentionBlockGeometry,
        *,
        batch: int,
        seq_len: int,
        dtype: torch.dtype,
        want_og: bool,
        want_delta: bool = False,
        og_fp8: bool = False,
        want_amax_do: bool = False,
        want_amax_dg: bool = False,
        want_dy_descale: bool = False,
    ) -> None:
        self.geom = geometry
        self.batch, self.seq_len = int(batch), int(seq_len)
        self.dtype = dtype
        self.want_og = bool(want_og)
        self.want_delta = bool(want_delta)
        self.og_fp8 = bool(og_fp8)
        self.want_amax_do = bool(want_amax_do)
        self.want_amax_dg = bool(want_amax_dg)
        self.want_dy_descale = bool(want_dy_descale)
        self._recipe = None

    def check_support(self) -> None:
        """Typed declines: bf16 / fp16 activations, the e4m3 ``og8`` arm needs the ``og`` output it casts, the kernel's row geometry."""
        from .kernels.sigmoid_gate_bwd import DEFAULT_THREADS_PER_CTA, validate_shape

        if self.dtype not in _ACT_DTYPES:
            raise NotImplementedError(f"{self.name}: bf16 / fp16 only, got {self.dtype}")
        if self.og_fp8 and not self.want_og:
            raise ValueError(f"{self.name}: og_fp8=True needs want_og=True (the e4m3 O_gated IS the third output; there is nothing to quantize without it)")
        validate_shape(self.geom.d_head, DEFAULT_THREADS_PER_CTA)

    def compile(self) -> None:
        """Build the gate backward's artifact for exactly this stage's arms (``has_og`` / ``has_delta`` / ``og_fp8`` / the two amax folds /
        the dY descale) -- every arm is in the recipe-cache key, so the default artifacts stay byte-identical."""
        from .kernels.sigmoid_gate_bwd import compile_sigmoid_gate_bwd

        self._recipe = compile_sigmoid_gate_bwd(
            dtype=self.dtype,
            h=self.geom.h_q,
            d=self.geom.d_head,
            has_og=self.want_og,
            has_seq_lens=False,
            has_delta=self.want_delta,
            og_fp8=self.og_fp8,
            has_amax_do=self.want_amax_do,
            has_amax_dg=self.want_amax_dg,
            has_dy_descale=self.want_dy_descale,
        )

    @property
    def n_partials_cap(self) -> int:
        """The most partials one launch writes (SMs x the persistent cap under an amax fold) -- the partials regions' length;
        needs ``compile()``."""
        if self._recipe is None:
            raise RuntimeError(f"{self.name}: call compile() before n_partials_cap")
        return int(self._recipe.n_ctas_cap)

    def n_partials(self) -> int:
        """The partials THIS declaration's launch writes (its grid, ``n_partials_for``): what the dO quantize and the epilogue reduce."""
        from .kernels.sigmoid_gate_bwd import n_partials_for

        if self._recipe is None:
            raise RuntimeError(f"{self.name}: call compile() before n_partials()")
        return int(n_partials_for(self._recipe, self.batch * self.seq_len))

    def execute(self, dog, o, gate, do, dg, og, *, stream, delta=None, scale_o=None, amax_do=None, amax_dg=None, descale_dy=None) -> int:
        """``delta`` (``want_delta`` only): the fp32 ``[B, H_q, S_pad]`` region the adapter reads as its external delta.
        ``scale_o`` (``og_fp8`` only): the forward's static ``scale_o`` as a 1-element fp32 device tensor; ``amax_do``
        (``want_amax_do`` only): the fp32 ``amax_partials_do`` region -- one ``max |dO|`` per CTA, every one overwritten;
        ``amax_dg`` (``want_amax_dg`` only): the ``amax_partials_dg`` region, the GATE band's ``max |dG|`` per CTA; ``descale_dy``
        (appended; ``want_dy_descale`` only): the 1-element fp32 slot the kernel multiplies ``dO_gated`` by before every use -- each
        checked BOTH ways by the kernel's host wrapper (Rule 1).  Returns the partials written (``n_partials()``)."""
        from .kernels.sigmoid_gate_bwd import run_sigmoid_gate_bwd

        if self._recipe is None:
            raise RuntimeError(f"{self.name}: call compile() before execute()")
        return run_sigmoid_gate_bwd(
            self._recipe,
            dog,
            o,
            gate,
            do,
            dg,
            og=og,
            seq_lens=None,
            s=self.seq_len if delta is not None else None,
            stream=stream,
            delta=delta,
            scale_o=scale_o,
            amax_do=amax_do,
            amax_dg=amax_dg,
            descale_dy=descale_dy,
        )


class _SdpaBwd(_Stage):
    """(B4) ``(dO, Q, K, V, O, LSE) -> dQ, dK, dV``, GQA-reduced over H_q/H_kv.

    **Writes no new kernel**: drives the shipped Rubin d=256 backward adapter
    ``cudnn.sdpa.bwd.api_dsl_sm107.SdpaBwdDslSm107`` -- a graph-free building
    block (construct over nine ``TensorDesc``s, ``check_support()``,
    ``compile()``, ``execute(..., workspace=<uint8 slice>, current_stream=)``)
    that carves its own scratch (delta, ONE dS chunk, padded staging, the GQA
    partials) out of the slice the block reserves for it and allocates nothing.
    ONE engine class, no backend fallback (module docstring, Rule 9).

    Consumes ``Q`` and ``K`` **post-norm, post-RoPE** -- the same tensors the
    forward's SDPA saw -- and ``V``, all COMPACT ``[B, S, H, D]`` (the adapter
    requires BSHD-physical operands: the block rebuilds Q / K through the
    forward's ``_QkNormRope`` from the ``proj_slab`` bands and compacts V with
    ``_VCompaction``, then hands the ``.transpose(1, 2)`` views the binder
    demands -- exactly what the forward's ``_Sdpa`` stage does). Writes COMPACT
    ``dQ`` / ``dK`` / ``dV`` into the workspace slots B5+B6 read.

    ``LSE`` is mandatory here and binds as the forward wrote it: ``[B, H_q, S]``
    fp32 NATURAL log -- the backward applies ``log2e`` itself at its stats
    prefetch (the block's forward never sets
    ``STATS_LOG2``). That is the whole reason ``save_for_backward`` implies
    ``return_lse`` in the forward.

    Declared with ``deterministic=False`` (the row declines ``True``); the
    chain has no atomics and the block's two-run bitwise test is the pin.
    Masks: the block's ``is_causal`` / ``causal_bottom_right`` / ``window_left``
    (> 0) / ``window_right`` (0) map one-to-one onto the adapter's; what the row
    cannot serve is declined by the block first, naming its own field.

    **Fusion status:** under ``fuse_gate_bwd`` (``external_delta``) the adapter is
    built with ``external_delta=True`` and :meth:`execute` hands it B3's ``delta``
    region, so the chain's ``dot_do_o`` launch does not exist and its scratch has
    no ``delta`` region (``delta_shape`` is the adapter's contract for the carve).

    **Packed sequences (``thd``):** the adapter is built with ``thd=True`` over the
    ENVELOPE declarations ``(B = num_sequences, H, S_max = max_seq_len, D)`` a
    ragged graph would make, both packed totals at ``T`` (self-attention over one
    packing), and the Stats packing of the training forward -- head-major
    ``[1, H_q, T]`` at head stride exactly ``T`` (``_thd_lse_head_stride``, ONE
    definition for the forward that writes ``saved.lse`` and this consumer).  The
    packed ``[1, T, H, D]`` buffers show up at :meth:`execute` as today's
    ``.transpose(1, 2)`` views (the binder matches ``(1, H, T, D)`` against the
    plan's packed geometry), together with the record's ``seq_lens`` as BOTH
    length operands.  Under ``fuse_gate_bwd`` the adapter takes the external
    delta in its PACKED form -- the head-major ``[1, H_q, ceil128(T)]`` the packed
    chain reads at the packed token index -- which is exactly what B3's dense
    delta arm writes at ``B = 1, S = T`` (tail zeroed), so the fused packed block
    is bitwise the unfused one and the chain's ``dot`` launch does not exist.
    """

    name = "sdpa_bwd"

    def __init__(
        self,
        geometry: GatedAttentionBlockGeometry,
        *,
        batch: int,
        seq_len: int,
        dtype: torch.dtype,
        device,
        external_delta: bool = False,
        thd: bool = False,
        num_sequences: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        cu_seqlens: bool = False,
    ) -> None:
        self.geom = geometry
        self.batch, self.seq_len = int(batch), int(seq_len)
        self.dtype = dtype
        self.device = device
        self.external_delta = bool(external_delta)
        # THD (appended): batch = 1, seq_len = T (the packed token total); the adapter is declared over the envelope
        # (num_sequences, max_seq_len).  cu_seqlens is the record's length FORM; the adapter derives it from the tensor's
        # numel at execute, so it is kept here only as the declaration's fact.
        self.thd = bool(thd)
        self.num_sequences = None if num_sequences is None else int(num_sequences)
        self.max_seq_len = None if max_seq_len is None else int(max_seq_len)
        self.cu_seqlens = bool(cu_seqlens)
        self._impl = None

    def _build_impl(self):
        from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107

        g, b, s, d, act, dev = self.geom, self.batch, self.seq_len, self.geom.d_head, self.dtype, self.device
        if self.thd:
            # THD: the declarations carry the ENVELOPE (B = num_sequences, S_max = max_seq_len) exactly as a ragged graph
            # declares them; the packed [1, T, H, D] buffers (compact: element stride 1, head stride D, token stride H*D --
            # what the adapter admits as packed rows) show up at execute.  Stats is declared (B, H_q, S_max, 1) -- the adapter
            # pins the DIMS only under THD -- and bound head-major at head stride T: saved.lse IS the contiguous [1, H_q, T]
            # the packed forward wrote.  Both packed totals are T (self-attention over one packing).  The external delta
            # (fuse_gate_bwd) is the adapter's packed head-major [1, H_q, ceil128(T)] -- B3's dense delta at B = 1, S = T is
            # exactly that tensor -- so the knob passes through as it does dense.
            t = b * s
            b, s = self.num_sequences, self.max_seq_len
            stats = TensorDesc(dtype=torch.float32, shape=(b, g.h_q, s, 1), stride=(g.h_q * s, s, 1, 1), stride_order=(3, 2, 1, 0), device=dev, name="stats")
            kw = dict(thd=True, max_total_seq_len_q=t, max_total_seq_len_kv=t, thd_stats_token_major=False, thd_stats_head_stride=_thd_lse_head_stride(t))
            external = self.external_delta
        else:
            # The row REQUIRES rank-4 (B, H_q, S_q, 1) stats with exactly this stride; saved.lse [B, H_q, S] binds to it as is
            # (the binder checks contiguity + element count only for Stats).
            stats = TensorDesc(dtype=torch.float32, shape=(b, g.h_q, s, 1), stride=(g.h_q * s, s, 1, 1), stride_order=(3, 2, 1, 0), device=dev, name="stats")
            kw, external = {}, self.external_delta
        return SdpaBwdDslSm107(
            sample_q=_bhsd_desc(b, g.h_q, s, d, act, dev, "q"),
            sample_k=_bhsd_desc(b, g.h_kv, s, d, act, dev, "k"),
            sample_v=_bhsd_desc(b, g.h_kv, s, d, act, dev, "v"),
            sample_o=_bhsd_desc(b, g.h_q, s, d, act, dev, "o"),
            sample_do=_bhsd_desc(b, g.h_q, s, d, act, dev, "dO"),
            sample_stats=stats,
            sample_dq=_bhsd_desc(b, g.h_q, s, d, act, dev, "dQ"),
            sample_dk=_bhsd_desc(b, g.h_kv, s, d, act, dev, "dK"),
            sample_dv=_bhsd_desc(b, g.h_kv, s, d, act, dev, "dV"),
            is_causal=bool(g.is_causal),
            causal_bottom_right=bool(g.causal_bottom_right),
            window_size_left=None if g.window_left < 0 else int(g.window_left),
            window_size_right=None if g.window_right < 0 else int(g.window_right),
            deterministic=False,
            scale_softmax=float(g.scale),
            seq_kv_lens_present=False,
            external_delta=external,
            **kw,
        )

    @property
    def delta_shape(self) -> tuple:
        """The adapter's ``external_delta_shape`` -- ``(B, H_q, S_pad)`` fp32 (``(1, H_q, ceil128(T))`` under ``thd``), the region B3
        fills under ``fuse_gate_bwd``."""
        if self._impl is None:
            self._impl = self._build_impl()
        return tuple(int(x) for x in self._impl.external_delta_shape)

    def check_support(self) -> None:
        from cudnn.sdpa.bwd.api_dsl_sm107 import _SM107_D  # the row's own head size -- derived, never re-literalled here

        if self.dtype not in _ACT_DTYPES:
            raise NotImplementedError(f"{self.name}: the sdpa_bwd_sm107 row serves bf16 / fp16 only, got {self.dtype}")
        if self.geom.d_head != _SM107_D:
            raise NotImplementedError(f"{self.name}: the Rubin d{_SM107_D} backward serves d_head = {_SM107_D} exactly, got {self.geom.d_head}")
        dev = torch.device(self.device)
        cc = tuple(torch.cuda.get_device_capability(dev)) if dev.type == "cuda" else None
        if cc != _SM107_CC:
            raise NotImplementedError(
                f"gated_attention_block backward targets Rubin (SM{_SM107_CC[0]}{_SM107_CC[1]}) only for now; found "
                + (f"SM{cc[0]}{cc[1]}" if cc is not None else str(dev))
            )
        self._impl = self._build_impl()
        self._impl.check_support()

    def scratch_workspace_bytes(self) -> int:
        """A pure function of the geometry: callable right after construction (no compile)."""
        if self._impl is None:
            self._impl = self._build_impl()
        return int(self._impl.scratch_workspace_bytes())

    def compile(self) -> None:
        if self._impl is None:
            self._impl = self._build_impl()
        self._impl.compile()

    def execute(self, q, k, v, o, do, lse, dq, dk, dv, *, workspace: torch.Tensor, stream, delta=None, seq_lens=None) -> None:
        """Every io tensor COMPACT ``[B, S, H, D]``; transposed here into the ``(B, H, S, D)`` view the binder demands.
        ``delta`` (``external_delta`` only): B3's fp32 ``[B, H_q, S_pad]`` region (``[1, H_q, ceil128(T)]`` under ``thd``), the
        adapter's ``delta_tensor``.
        ``seq_lens`` (``thd`` only): the record's packed lengths (``[B]`` int32 lengths or ``[B+1]`` prefix sums), handed to
        the adapter as BOTH ``seq_q_lens`` and ``seq_kv_lens`` -- self-attention over one packing; ``lse`` is then the
        head-major ``[1, H_q, T]`` ``saved.lse`` and the eight io views are the packed ``(1, H, T, D)``."""
        if self._impl is None:
            raise RuntimeError(f"{self.name}: call compile() before execute()")
        lens = dict(seq_q_lens=seq_lens, seq_kv_lens=seq_lens) if self.thd else {}
        self._impl.execute(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            o.transpose(1, 2),
            do.transpose(1, 2),
            lse,
            dq.transpose(1, 2),
            dk.transpose(1, 2),
            dv.transpose(1, 2),
            workspace=workspace,
            current_stream=cuda.CUstream(int(stream)),
            delta_tensor=delta,
            **lens,
        )


# ---------------------------------------------------------------------------
# 3b. The quantized backward's SDPA stage
# ---------------------------------------------------------------------------


class _SdpaBwdFp8(_Stage):
    """(B4, fp8) the sibling of :class:`_SdpaBwd` over the Rubin d=256 per-tensor fp8 backward adapter
    ``cudnn.sdpa.bwd.api_dsl_sm107.SdpaBwdDslSm107Fp8``, ALWAYS built with ``external_delta=True`` and
    ``amax_requested=("amax_dP",)``.  ONE engine class, no backend fallback (module docstring, Rule 9): a geometry the
    row cannot serve surfaces the row's own typed message through :meth:`check_support`, never a cuDNN plan.

    **What it consumes.**  The recomputed e4m3 ``q8 / k8 / v8`` (quantized with the forward's STATIC ``scale_q / k /
    v``, bitwise the forward's own SDPA operands), the e4m3 ``do8`` (the gate backward's bf16 ``dO`` quantized with the
    step's ``scale_dO``), the forward's exact fp32 NATURAL-log ``lse`` (``[B, H_q, S]``, bound as the row's contiguous
    ``(B, H_q, S, 1)`` Stats -- the row applies ``log2e`` itself, as :class:`_SdpaBwd` documents), the block's fp32
    ``delta`` and the row's TWELVE fp8 scalars; it writes ``grad_dtype`` (bf16) ``dq / dk / dv`` into the compact slots
    the norm backward reads (``scale_dQ / dK / dV = 1.0``: TRUE-unit bf16 gradients, no amax requested for them) and the
    row's ``amax_dP`` -- ``max |dS|`` in fp32 right before its ``scale_dP`` cast -- into a slot of the block's scalar
    block.

    **The external delta is in TRUE units, and it is not the row's own pre-pass.**  Quoting the adapter's contract
    (``api_dsl_sm107.py``, "An externally computed delta"): *"the fp8 row reads delta in TRUE units, UNSCALED -- its own
    pre-pass is the rowsum of the e4m3 payload codes times ``descale_o * descale_dO``, nobody applies those descales to a
    caller's delta, so a delta derived from the bf16 O / dO binds AS IS and is NOT bitwise the row's own pre-pass (its
    tests assert against an oracle fed the same delta, never bitwise)."*  The gate backward's ``delta = rowsum(bf16 dO *
    bf16 O)`` is exactly such a tensor: fp32 contiguous ``[B, H_q, S_pad]`` (:attr:`delta_shape`), finite zeros on the
    pad rows, 16-B aligned -- the adapter validates it like Stats before any bind.  The kernel then forms ``dP`` from the
    e4m3 ``do8`` while ``delta`` came from the bf16 ``dO``, so the softmax identity ``sum_j P_ij dP_ij = delta_i`` holds
    only to the dO quantization error; the oracle is fed the SAME delta, so the comparison stays consistent and a residual
    of that size is the contract, not a defect.  Why not the row's own pre-pass: it would recompute delta over the e4m3
    payloads -- two roundings against the block's delta contract -- and cost a launch and a second read of O / dO.

    **The dead operands.**  The row's ``o`` and ``descale_o`` stay REQUIRED by its append-only ABI and are read by
    NOTHING under an external delta (the adapter's contract, same paragraph: *"The ``o`` / ``descale_o`` (fp8) ... operands
    stay required under the flag and are read by nothing"*).  The block therefore binds an EXISTING e4m3 operand of the
    same compact ``[B, S, H_q, D]`` shape as the dead ``o`` -- ``og8`` when the out-projection weight gradient carved it,
    else ``do8`` -- and ``1 / scale_o`` as the dead descale; no slot is carved for a tensor nobody reads.  The binder's
    only aliasing check is caller-workspace-vs-operand, never operand-vs-operand, so one buffer may stand in two roles
    (``test_sdpa_bwd_fp8_stage_binds_the_dead_o_without_a_slot`` pins that the gradients are bitwise whatever is bound
    there).

    **The scalars.**  ``execute(scalars=)`` takes ``{name: 1-element fp32 CUDA tensor}`` for ALL TWELVE names of
    ``prepared_sm107.FP8_SCALARS`` (:meth:`scalar_names`).  ELEVEN of them are slots of the block's scalar block (the row's
    ``scale_dQ / dK / dV`` all read the one ``scale_dqkv`` slot): ten names the scalar-init launch writes on EVERY execute --
    the plan-time constants out of its kernel arguments (``descale_q / k / v = 1 / scale_q / k / v``, the dead ``descale_o``,
    ``descale_s / scale_s``, the unit ``scale_dQ / dK / dV``) and ``descale_dP = 1 / scale_dP`` -- and ``descale_dO``,
    published by the dO quantize launch; the twelfth is the caller's ``scale_dP``.  Nothing comes from ``compile()``: every
    slot is written on the launch stream by a launch that precedes the row's.  Each is checked here (name set, dtype, element
    count, device, the 4-byte alignment the row declares) BEFORE the adapter's own checks, so a missing or misspelled scalar
    names itself.  Slot views at a 4-byte stride are legal operands: the row declares every scalar and the amax at 4-byte
    alignment.

    Declared with ``deterministic=False`` (the row declines ``True``), ``seq_kv_lens_present=False`` (the block declines
    padding first), the geometry's masks exactly as :class:`_SdpaBwd` maps them.

    **Packed sequences (``thd``):** exactly :class:`_SdpaBwd`'s packed declaration, over the fp8 row -- the ENVELOPE
    ``(B = num_sequences, H, S_max = max_seq_len, D)``, both packed totals at ``T``, the head-major ``[1, H_q, T]`` Stats at
    head stride ``T`` (``_thd_lse_head_stride``), the packed ``[1, T, H, D]`` e4m3 operands as :meth:`execute`'s
    ``.transpose(1, 2)`` views and the record's ``seq_lens`` as BOTH length operands.  The external delta stays mandatory and
    is the PACKED head-major ``[1, H_q, ceil128(T)]`` the fp8 THD chain reads at the packed token index -- what the gate
    backward's dense delta arm writes at ``B = 1, S = T`` (tail zeroed), in TRUE units as dense -- so the chain's own scaled
    pre-pass over the e4m3 payloads never runs under THD either (one rounding of dO, the delta contract above); every other
    stage of the quantized backward is token-wise at ``B = 1, S = T``.
    """

    name = "sdpa_bwd_fp8"

    def __init__(
        self,
        geometry: GatedAttentionBlockGeometry,
        *,
        batch: int,
        seq_len: int,
        grad_dtype: torch.dtype,
        device,
        thd: bool = False,
        num_sequences: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        cu_seqlens: bool = False,
    ) -> None:
        self.geom = geometry
        self.batch, self.seq_len = int(batch), int(seq_len)
        self.grad_dtype = grad_dtype
        self.device = device
        # THD (appended): batch = 1, seq_len = T (the packed token total); the adapter is declared over the envelope
        # (num_sequences, max_seq_len) exactly as _SdpaBwd's.  cu_seqlens is the record's length FORM (the adapter derives it
        # from the tensor's numel at execute), kept here as the declaration's fact.
        self.thd = bool(thd)
        self.num_sequences = None if num_sequences is None else int(num_sequences)
        self.max_seq_len = None if max_seq_len is None else int(max_seq_len)
        self.cu_seqlens = bool(cu_seqlens)
        self._impl = None

    @staticmethod
    def scalar_names() -> tuple:
        """The row's twelve fp8 scalars in its own order (``prepared_sm107.FP8_SCALARS``): exactly the keys ``execute(scalars=)`` takes."""
        from cudnn.sdpa.bwd.prepared_sm107 import FP8_SCALARS

        return tuple(FP8_SCALARS)

    def _build_impl(self):
        """``SdpaBwdDslSm107Fp8`` over e4m3 ``_bhsd_desc`` samples for q / k / v / o / dO, fp32 ``(B, H_q, S, 1)`` stats at
        stride ``(H_q * S, S, 1, 1)``, ``grad_dtype`` dq / dk / dv, the geometry's masks as :class:`_SdpaBwd`,
        ``deterministic=False``, ``seq_kv_lens_present=False``, ``amax_requested=("amax_dP",)``, ``external_delta=True``;
        under ``thd`` the envelope declarations, the packed Stats and both packed totals exactly as :class:`_SdpaBwd`'s THD arm."""
        from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107Fp8

        g, b, s, d, dev = self.geom, self.batch, self.seq_len, self.geom.d_head, self.device
        code = torch.float8_e4m3fn
        if self.thd:
            # THD: _SdpaBwd's packed declaration over the fp8 row -- every sample over the ENVELOPE (B = num_sequences, S_max =
            # max_seq_len), Stats declared (B, H_q, S_max, 1) and bound head-major at head stride T (saved.lse IS the contiguous
            # [1, H_q, T] the packed forward wrote), both packed totals T.  The external delta is then the adapter's packed
            # head-major [1, H_q, ceil128(T)] (external_delta_shape under thd): B3's dense delta at B = 1, S = T is exactly that tensor.
            t = b * s
            b, s = self.num_sequences, self.max_seq_len
            kw = dict(thd=True, max_total_seq_len_q=t, max_total_seq_len_kv=t, thd_stats_token_major=False, thd_stats_head_stride=_thd_lse_head_stride(t))
        else:
            kw = {}
        # The row REQUIRES rank-4 (B, H_q, S_q, 1) stats with exactly this stride; saved.lse [B, H_q, S] binds to it as is
        # (the binder checks contiguity + element count only for Stats; under THD the adapter pins the DIMS only).
        stats = TensorDesc(dtype=torch.float32, shape=(b, g.h_q, s, 1), stride=(g.h_q * s, s, 1, 1), stride_order=(3, 2, 1, 0), device=dev, name="stats")
        return SdpaBwdDslSm107Fp8(
            sample_q=_bhsd_desc(b, g.h_q, s, d, code, dev, "q"),
            sample_k=_bhsd_desc(b, g.h_kv, s, d, code, dev, "k"),
            sample_v=_bhsd_desc(b, g.h_kv, s, d, code, dev, "v"),
            sample_o=_bhsd_desc(b, g.h_q, s, d, code, dev, "o"),
            sample_do=_bhsd_desc(b, g.h_q, s, d, code, dev, "dO"),
            sample_stats=stats,
            sample_dq=_bhsd_desc(b, g.h_q, s, d, self.grad_dtype, dev, "dQ"),
            sample_dk=_bhsd_desc(b, g.h_kv, s, d, self.grad_dtype, dev, "dK"),
            sample_dv=_bhsd_desc(b, g.h_kv, s, d, self.grad_dtype, dev, "dV"),
            is_causal=bool(g.is_causal),
            causal_bottom_right=bool(g.causal_bottom_right),
            window_size_left=None if g.window_left < 0 else int(g.window_left),
            window_size_right=None if g.window_right < 0 else int(g.window_right),
            deterministic=False,
            scale_softmax=float(g.scale),
            seq_kv_lens_present=False,
            amax_requested=("amax_dP",),
            external_delta=True,
            **kw,
        )

    def _ensure_impl(self):
        if self._impl is None:
            self._impl = self._build_impl()
        return self._impl

    @property
    def delta_shape(self) -> tuple:
        """The adapter's ``external_delta_shape`` -- ``(B, H_q, S_pad)`` fp32, the region the gate backward fills under ``quant``."""
        return tuple(int(x) for x in self._ensure_impl().external_delta_shape)

    def check_support(self) -> None:
        if self.grad_dtype != torch.bfloat16:
            raise NotImplementedError(
                f"{self.name}: the quantized block backward's SDPA gradients are bf16 (the norm backward's operand dtype); grad_dtype={self.grad_dtype} "
                "is not wired here (the row's fp16 / e4m3 gradient arms would need their own accept cells)"
            )
        from cudnn.sdpa.bwd.api_dsl_sm107 import _SM107_D  # the row's own head size -- derived, never re-literalled here

        if self.geom.d_head != _SM107_D:
            raise NotImplementedError(f"{self.name}: the Rubin d{_SM107_D} fp8 backward serves d_head = {_SM107_D} exactly, got {self.geom.d_head}")
        dev = torch.device(self.device)
        cc = tuple(torch.cuda.get_device_capability(dev)) if dev.type == "cuda" else None
        if cc != _SM107_CC:
            raise NotImplementedError(
                f"gated_attention_block backward targets Rubin (SM{_SM107_CC[0]}{_SM107_CC[1]}) only for now; found "
                + (f"SM{cc[0]}{cc[1]}" if cc is not None else str(dev))
            )
        # The row's own contract check (d = 256, the e4m3 payload dtypes, the grad dtype, the masks, the dense Stats layout):
        # what it declines surfaces typed and by its own name -- the block mirrors no decline the row does not make.
        self._ensure_impl().check_support()

    def scratch_workspace_bytes(self) -> int:
        """A pure function of the geometry: callable right after construction (no compile).  Under ``external_delta`` the
        adapter's carve has NO ``delta`` region: the block's own region is the delta."""
        return int(self._ensure_impl().scratch_workspace_bytes())

    def compile(self) -> None:
        self._ensure_impl().compile()

    def _check_scalar(self, what: str, t, dev: torch.device) -> None:
        if not isinstance(t, torch.Tensor) or t.numel() != 1 or t.dtype != torch.float32 or not t.is_cuda:
            got = f"{type(t).__name__}" + (f" {tuple(t.shape)} {t.dtype} on {t.device}" if isinstance(t, torch.Tensor) else "")
            raise ValueError(
                f"{self.name}: {what} must be a 1-element fp32 CUDA tensor (a slot of the block's scalar block, or the caller's scale_dP), got {got}"
            )
        if t.device.index != dev.index:
            raise ValueError(f"{self.name}: {what} is on {t.device} but the stage launches on {dev}; every scalar of one launch lives on the launch device")
        if t.data_ptr() % 4:
            raise ValueError(f"{self.name}: {what} must be 4-byte aligned (the row declares its scalars and amax at 4 B), got {t.data_ptr():#x}")

    def execute(self, q8, k8, v8, o_dead8, do8, lse, dq, dk, dv, *, workspace: torch.Tensor, stream, delta, scalars: dict, amax_dp, seq_lens=None) -> None:
        """``q8 .. dv`` COMPACT ``[B, S, H, D]`` (transposed here into the ``(B, H, S, D)`` views the binder demands);
        ``o_dead8`` an existing e4m3 operand bound as the adapter's dead ``o`` (``og8`` when it exists, else ``do8``);
        ``lse`` the forward's fp32 ``[B, H_q, S]``; ``delta`` the block's fp32 ``[B, H_q, S_pad]`` region (the adapter
        validates its layout); ``scalars`` ``{name: 1-element fp32 tensor}`` for ALL TWELVE of the row's fp8 scalars (a
        missing or extra name, a wrong dtype / count / device / alignment is a typed ``ValueError`` here, before the
        adapter's); ``amax_dp`` the scalar block's slot view the row's ``amax_dP`` lands in (pre-zeroed by the caller: an
        ``atomicMax`` only grows).  ``seq_lens`` (``thd`` only; appended): the record's packed lengths (``[B]`` int32 lengths or
        ``[B+1]`` prefix sums), handed to the adapter as BOTH ``seq_q_lens`` and ``seq_kv_lens`` -- self-attention over one
        packing; ``lse`` is then the head-major ``[1, H_q, T]`` ``saved.lse``, ``delta`` the packed ``[1, H_q, ceil128(T)]`` and
        the eight io views the packed ``(1, H, T, D)``."""
        if self._impl is None:
            raise RuntimeError(f"{self.name}: call compile() before execute()")
        names = self.scalar_names()
        if not isinstance(scalars, dict):
            raise ValueError(f"{self.name}: scalars must be a dict {{name: 1-element fp32 CUDA tensor}} over {names}, got {type(scalars).__name__}")
        missing = [n for n in names if n not in scalars]
        extra = [n for n in scalars if n not in names]
        if missing or extra:
            raise ValueError(f"{self.name}: scalars must name exactly the row's twelve fp8 scalars {names}: missing {missing}, unexpected {extra}")
        dev = torch.device(self.device)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        for n in names:
            self._check_scalar(n, scalars[n], dev)
        self._check_scalar("amax_dp", amax_dp, dev)
        lens = dict(seq_q_lens=seq_lens, seq_kv_lens=seq_lens) if self.thd else {}
        self._impl.execute(
            q8.transpose(1, 2),
            k8.transpose(1, 2),
            v8.transpose(1, 2),
            o_dead8.transpose(1, 2),
            do8.transpose(1, 2),
            lse,
            dq.transpose(1, 2),
            dk.transpose(1, 2),
            dv.transpose(1, 2),
            workspace=workspace,
            current_stream=cuda.CUstream(int(stream)),
            delta_tensor=delta,
            amax_dP=amax_dp,
            **{n: scalars[n] for n in names},
            **lens,
        )


class _SdpaBwdMxfp8(_Stage):
    """(B4, MXFP8) the sibling of :class:`_SdpaBwdFp8` over the Rubin d=256 MXFP8 backward adapter
    ``cudnn.sdpa.bwd.api_dsl_sm107.SdpaBwdDslSm107Mxfp8``, ALWAYS built with ``external_delta=True``.  ONE engine class,
    no backend fallback (module docstring, Rule 9): a geometry the row cannot serve surfaces the row's own typed message
    through :meth:`check_support`, never a cuDNN plan.

    **What it consumes.**  The e4m3 MXFP8 payloads with their F8_128x4 E8M0 scale factors -- ``q8 / k8`` ROWWISE (32-element
    blocks along D: the forward's own SDPA operands, recomputed bitwise), ``v8`` ROWWISE (the backward's dP operand; the
    forward consumed V COLUMNWISE for its BMM2 and the backward re-quantizes it along D: the row's own convention, its
    ``descale_v`` is the rowwise V scale), ``do8`` ROWWISE (the gate backward's bf16 ``dO`` quantized along D, the dP
    operand) and the transposed-quantization payloads ``q_T8 / k_T8 / do_T8`` COLUMNWISE (32-token blocks along S, the
    dK / dQ / dV contraction axes; the same compact BSHD shape as their rowwise twins) -- the forward's exact fp32
    NATURAL-log ``lse`` (``[B, H_q, S]``, bound as the row's contiguous ``(B, H_q, S, 1)`` Stats; the row applies ``log2e``
    itself, as :class:`_SdpaBwd` documents) and the block's fp32 ``delta``; it writes ``grad_dtype`` (bf16) ``dq / dk / dv``
    into the compact slots the norm backward reads.  No scalars and no amax: the row has none (the block scales dequantize
    inside the MMAs, the gradients are TRUE-unit bf16).

    **The external delta IS bitwise the row's own pre-pass here** -- unlike the fp8 stage.  Quoting the adapter's contract
    (``api_dsl_sm107.py``, "An externally computed delta"): *"Units are the row's: the half row and the MXFP8 row read the
    raw half-precision dot (the MXFP8 row's own pre-pass is ``dot_do_o`` over its ``o_f16`` / ``dO_f16`` ports, so a producer
    forming it in that order is bitwise the chain's own)"*.  The gate backward's ``delta = rowsum(bf16 dO * bf16 O)`` is
    formed in ``dot_do_o``'s own reduction order, so a block fed it returns dQ / dK / dV ``torch.equal`` a standalone
    ``SdpaBwdDslSm107Mxfp8(external_delta=False)`` run over the same operands -- the row's own bitwise pin
    (``test_adapter_external_delta_is_bitwise_the_rows_own_pre_pass``), asserted again at the block
    (``test_sdpa_bwd_mxfp8_stage_external_delta_is_bitwise_the_rows_own_pre_pass``).  Known modelled difference, the same
    as the fp8 stage's: the kernel forms ``dP`` from the rowwise e4m3 ``do8`` while ``delta`` is the bf16 ``dO``'s row-sum,
    so the softmax identity ``sum_j P_ij dP_ij = delta_i`` holds only to the dO quantization error; the oracle is fed the
    SAME delta (``mxfp8_ref.compute_ref_backward(delta=)``), so the comparison stays consistent and a residual of that size
    is the contract, not a defect.

    **The dead operands.**  The row's ``o_f16`` and ``dO_f16`` ports stay REQUIRED by its append-only ABI and are read by
    NOTHING under an external delta (the adapter's contract, same paragraph: *"The ``o`` / ``descale_o`` (fp8) and ``o_f16``
    / ``dO_f16`` (MXFP8) operands stay required under the flag and are read by nothing (append-only ABI; the gated block's
    training record saves O anyway)"*).  The block binds EXISTING bf16 buffers of the ports' compact ``[B, S, H_q, D]``
    shape -- the record's pre-gate ``saved.o`` as ``o_f16`` and the gate backward's bf16 ``dO`` as ``dO_f16`` -- so no slot
    is carved for a tensor nobody reads (``test_sdpa_bwd_mxfp8_stage_binds_the_dead_ports_without_a_slot`` pins that the
    gradients are bitwise whatever bf16 tensors stand there).

    **The scale factors.**  ``execute(sf=)`` takes ``{name: uint8 tensor}`` for EXACTLY the seven names of
    ``api_dsl_sm107._MXFP8_SF_ROLES`` (:meth:`sf_roles`), each the F8_128x4 blob of its payload in the kernel's own layout
    (:meth:`sf_shapes`): ``sf_q / sf_do`` ``(B, H_q, ceil128(S), D/32)`` and ``sf_k / sf_v`` ``(B, H_kv, ceil128(S), D/32)``
    ROWWISE -- one 1 KiB tile per ``(b, h, 128-row tile)`` -- and ``sf_q_T / sf_do_T`` ``(B, H_q, D/32, ceil128(S))`` /
    ``sf_k_T`` ``(B, H_kv, D/32, ceil128(S))`` COLUMNWISE, D-plane-major.  The shapes are the row's declared dims (``sf_v``
    MUST be the rowwise form: the adapter asserts that shape at declaration, because a columnwise blob of the same byte count
    would be a wrong dV); at execute the adapter trusts the byte count, and each blob is checked here (name set, dtype,
    device, byte count, contiguity, the 16-B alignment the row declares) BEFORE the adapter's own checks, so a missing or
    misspelled blob names itself.  A blob in the WRONG layout has the right byte count and passes every host check: the
    bitwise pins of the block's tests against the torch quantization are the only guard on the byte order.

    **The GQA fold is the row's.**  Under GQA the MXFP8 SDPA backward folds its per-Q-head dK partials in fp32 and rounds
    the sum once, like the reference, while its per-Q-head dV partials are bf16 (the kernel stores them from its epilogue;
    fp32 ones do not fit its 327 KiB shared-memory budget), so dV carries one bf16 rounding per group member where a
    once-rounded reference carries one in total (relative RMS about 3e-3 at a group of 4, the geometry the tests run,
    measured on the per-tensor fp8 row before it moved to fp32 partials); the modelled oracle folds dV the same way and the
    distance to a once-rounded fold is reported per cell.  The row's block-scale dQ GEMM runs once per head chunk under GQA,
    like the plain renderings (its dQ record takes ``b_head_group`` = the group: B and its scale-factor descriptor are indexed
    by ``h // group``): :meth:`dq_launches_per_chunk` reports that count off the adapter's own record and :meth:`head_chunks`
    the row's head chunking, so the block's launch census reads the row, never a formula.

    Declared with ``deterministic=False`` (the row declines ``True``), ``seq_kv_lens_present=False`` (the block declines
    padding first), the geometry's masks exactly as :class:`_SdpaBwd` maps them.  Dense only: the MXFP8 block backward
    declines ``thd`` at declaration (its SDPA-layout MX quantizes run the quantizer's dense arm only; the packed MXFP8 training
    record exists, the packed MXFP8 backward is a follow-up; the per-tensor fp8 sibling :class:`_SdpaBwdFp8` serves the packed record).
    """

    name = "sdpa_bwd_mxfp8"

    def __init__(self, geometry: GatedAttentionBlockGeometry, *, batch: int, seq_len: int, grad_dtype: torch.dtype, device) -> None:
        """Record the declaration; the adapter (``SdpaBwdDslSm107Mxfp8``) is built lazily by ``_ensure_impl`` (``_build_impl``: constructor
        arithmetic from these facts, no device work) -- ``delta_shape`` / ``scratch_workspace_bytes`` read it before ``compile``."""
        self.geom = geometry
        self.batch, self.seq_len = int(batch), int(seq_len)
        self.grad_dtype = grad_dtype
        self.device = device
        self._impl = None

    @staticmethod
    def sf_roles() -> tuple:
        """The row's seven scale-factor operands in its own order (``api_dsl_sm107._MXFP8_SF_ROLES``): exactly the keys ``execute(sf=)`` takes."""
        from cudnn.sdpa.bwd.api_dsl_sm107 import _MXFP8_SF_ROLES

        return tuple(_MXFP8_SF_ROLES)

    def sf_shapes(self) -> dict:
        """The seven scale-factor blobs' declared dims, in the kernel's documented shapes (``sm107/bprop_d256_mxfp8.py``, the
        operand table): rowwise ``(B, H, ceil128(S), D / 32)`` for ``sf_q / sf_do / sf_k / sf_v``, columnwise ``(B, H, D / 32,
        ceil128(S))`` for ``sf_q_T / sf_do_T / sf_k_T`` -- the 128-row atom pad and the 32-element block from the adapter's own
        constants, never re-literalled.  The byte count of each is what the adapter checks at execute."""
        from cudnn.sdpa.bwd.api_dsl_sm107 import _MXFP8_BLOCK, _MXFP8_SF_ATOM_ROWS

        g, b, s, d = self.geom, self.batch, self.seq_len, self.geom.d_head
        rows = -(-s // _MXFP8_SF_ATOM_ROWS) * _MXFP8_SF_ATOM_ROWS
        groups = d // _MXFP8_BLOCK
        return dict(
            sf_q=(b, g.h_q, rows, groups),
            sf_q_T=(b, g.h_q, groups, rows),
            sf_k=(b, g.h_kv, rows, groups),
            sf_k_T=(b, g.h_kv, groups, rows),
            sf_v=(b, g.h_kv, rows, groups),
            sf_do=(b, g.h_q, rows, groups),
            sf_do_T=(b, g.h_q, groups, rows),
        )

    def _build_impl(self):
        """``SdpaBwdDslSm107Mxfp8`` over e4m3 ``_bhsd_desc`` samples for q / k / v / dO and the transposed-quantization
        q_T / k_T / dO_T, bf16 ``o`` / ``dO_f16`` (the dead ports, the record's O shape), fp32 ``(B, H_q, S, 1)`` stats at stride
        ``(H_q * S, S, 1, 1)``, ``grad_dtype`` dq / dk / dv, the seven uint8 scale-factor descriptors of :meth:`sf_shapes`, the
        geometry's masks as :class:`_SdpaBwd`, ``deterministic=False``, ``seq_kv_lens_present=False``, ``external_delta=True``."""
        from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107Mxfp8

        g, b, s, d, dev = self.geom, self.batch, self.seq_len, self.geom.d_head, self.device
        code, act = torch.float8_e4m3fn, self.grad_dtype
        # The row REQUIRES rank-4 (B, H_q, S_q, 1) stats with exactly this stride; saved.lse [B, H_q, S] binds to it as is
        # (the binder checks contiguity + element count only for Stats).
        stats = TensorDesc(dtype=torch.float32, shape=(b, g.h_q, s, 1), stride=(g.h_q * s, s, 1, 1), stride_order=(3, 2, 1, 0), device=dev, name="stats")

        def sf_desc(name, shape):
            """A contiguous uint8 ``TensorDesc`` of one scale-factor blob in the row's declared dims (``sf_shapes``)."""
            stride = tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))
            return TensorDesc(
                dtype=torch.uint8, shape=shape, stride=stride, stride_order=TensorDesc._compute_stride_order(shape, stride), device=dev, name=name
            )

        sf = {n: sf_desc(n, shape) for n, shape in self.sf_shapes().items()}
        return SdpaBwdDslSm107Mxfp8(
            sample_q=_bhsd_desc(b, g.h_q, s, d, code, dev, "q"),
            sample_k=_bhsd_desc(b, g.h_kv, s, d, code, dev, "k"),
            sample_v=_bhsd_desc(b, g.h_kv, s, d, code, dev, "v"),
            sample_o=_bhsd_desc(b, g.h_q, s, d, act, dev, "o"),  # the dead o_f16 port: the record's bf16 pre-gate O stands there
            sample_do=_bhsd_desc(b, g.h_q, s, d, code, dev, "dO"),
            sample_stats=stats,
            sample_dq=_bhsd_desc(b, g.h_q, s, d, act, dev, "dQ"),
            sample_dk=_bhsd_desc(b, g.h_kv, s, d, act, dev, "dK"),
            sample_dv=_bhsd_desc(b, g.h_kv, s, d, act, dev, "dV"),
            sample_q_T=_bhsd_desc(b, g.h_q, s, d, code, dev, "q_T"),
            sample_k_T=_bhsd_desc(b, g.h_kv, s, d, code, dev, "k_T"),
            sample_do_T=_bhsd_desc(b, g.h_q, s, d, code, dev, "dO_T"),
            sample_do_f16=_bhsd_desc(b, g.h_q, s, d, act, dev, "dO_f16"),  # the dead dO_f16 port: the gate backward's bf16 dO stands there
            **{f"sample_{n}": sf[n] for n in self.sf_roles()},
            is_causal=bool(g.is_causal),
            causal_bottom_right=bool(g.causal_bottom_right),
            window_size_left=None if g.window_left < 0 else int(g.window_left),
            window_size_right=None if g.window_right < 0 else int(g.window_right),
            deterministic=False,
            scale_softmax=float(g.scale),
            seq_kv_lens_present=False,
            external_delta=True,
        )

    def _ensure_impl(self):
        """The adapter, built once on first use (its plan is constructor arithmetic: no ``check_support``, no device work)."""
        if self._impl is None:
            self._impl = self._build_impl()
        return self._impl

    @property
    def delta_shape(self) -> tuple:
        """The adapter's ``external_delta_shape`` -- ``(B, H_q, S_pad)`` fp32, the region the gate backward fills under ``quant``."""
        return tuple(int(x) for x in self._ensure_impl().external_delta_shape)

    def head_chunks(self) -> int:
        """``c``: the head chunks the row's main kernel and GEMMs run per execute (``H_q / qh_chunk``, the adapter's dS-workspace
        chunking against the sm107 rows' shared budget) -- a launch-count term of the block's census, read off the adapter."""
        impl = self._ensure_impl()
        return impl.h_q // int(impl._qh_chunk)

    def dq_launches_per_chunk(self) -> int:
        """``q``: dQ GEMM launches per head chunk -- ONE on the block-scale arm as on the plain renderings (its dQ record takes
        ``b_head_group = group``: B and its scale-factor descriptor are indexed by ``h // group``), ``group`` on the per-member
        twin (``b_head_group == 1``), read off the adapter's record (``prepared_host._dq_launches``), never assumed.  The record's
        ``b_head_group`` is copied at :meth:`compile`; before it the constructor's default (1) reads as ``group`` launches."""
        from cudnn.sdpa.bwd.kernels.sm107.prepared_host import _dq_launches

        impl = self._ensure_impl()
        return _dq_launches(impl.h_q // max(impl.h_kv, 1), int(impl._dq_b_head_group))

    def check_support(self) -> None:
        """Typed declines: bf16 gradients (the row's only gradient dtype), the row's own head size, Rubin, then the ADAPTER's own
        contract check (its declines surface by their own name -- the block mirrors none the row does not make)."""
        if self.grad_dtype != torch.bfloat16:
            raise NotImplementedError(
                f"{self.name}: the quantized block backward's SDPA gradients are bf16 (the norm backward's operand dtype and the row's only gradient "
                f"dtype); grad_dtype={self.grad_dtype} is not wired here"
            )
        from cudnn.sdpa.bwd.api_dsl_sm107 import _SM107_D  # the row's own head size -- derived, never re-literalled here

        if self.geom.d_head != _SM107_D:
            raise NotImplementedError(f"{self.name}: the Rubin d{_SM107_D} MXFP8 backward serves d_head = {_SM107_D} exactly, got {self.geom.d_head}")
        dev = torch.device(self.device)
        cc = tuple(torch.cuda.get_device_capability(dev)) if dev.type == "cuda" else None
        if cc != _SM107_CC:
            raise NotImplementedError(
                f"gated_attention_block backward targets Rubin (SM{_SM107_CC[0]}{_SM107_CC[1]}) only for now; found "
                + (f"SM{cc[0]}{cc[1]}" if cc is not None else str(dev))
            )
        # The row's own contract check (d = 256, the e4m3 payload dtypes, the bf16 side, the masks, the dense Stats layout, the
        # scale-factor byte counts and the rowwise sf_v shape, the Rubin-line gate of its block-scaled dS chain): what it declines
        # surfaces typed and by its own name -- the block mirrors no decline the row does not make.
        self._ensure_impl().check_support()

    def scratch_workspace_bytes(self) -> int:
        """A pure function of the geometry: callable right after construction (no compile).  Under ``external_delta`` the
        adapter's carve has NO ``delta`` region: the block's own region is the delta.  It carries the row's block-scaled dS
        payloads and atoms for one head chunk and, under GQA, the per-Q-head partials (fp32 dK, bf16 dV)."""
        return int(self._ensure_impl().scratch_workspace_bytes())

    def compile(self) -> None:
        """Compile the MXFP8 row's chain (its host, main kernel and block-scale GEMM renderings)."""
        self._ensure_impl().compile()

    def _check_sf(self, sf, dev: torch.device) -> None:
        """``execute(sf=)``'s dict, typed HERE before the adapter's own check: exactly the seven role names, each a contiguous uint8
        CUDA tensor on the block's device of the declared byte count."""
        names = self.sf_roles()
        if not isinstance(sf, dict):
            raise ValueError(f"{self.name}: sf must be a dict {{name: uint8 CUDA tensor}} over {names}, got {type(sf).__name__}")
        missing = [n for n in names if n not in sf]
        extra = [n for n in sf if n not in names]
        if missing or extra:
            raise ValueError(f"{self.name}: sf must name exactly the row's seven scale-factor blobs {names}: missing {missing}, unexpected {extra}")
        for n in names:
            t = sf[n]
            if not isinstance(t, torch.Tensor) or t.dtype != torch.uint8 or not t.is_cuda:
                got = f"{type(t).__name__}" + (f" {tuple(t.shape)} {t.dtype} on {t.device}" if isinstance(t, torch.Tensor) else "")
                raise ValueError(f"{self.name}: sf[{n!r}] must be a uint8 CUDA tensor (the F8_128x4 E8M0 scale-factor blob of the {n[3:]} payload), got {got}")
            if t.device.index != dev.index:
                raise ValueError(
                    f"{self.name}: sf[{n!r}] is on {t.device} but the stage launches on {dev}; every blob of one launch lives on the launch device"
                )
            want = int(self._impl._sf_expected_bytes(n))
            if t.numel() != want:
                raise ValueError(
                    f"{self.name}: sf[{n!r}] holds {t.numel()} bytes; the F8_128x4 layout for this geometry needs {want} (declared dims {self.sf_shapes()[n]})"
                )
            if not t.is_contiguous():
                raise ValueError(f"{self.name}: sf[{n!r}] must be contiguous (the row binds the blob by its bytes), got strides {tuple(t.stride())}")
            if t.data_ptr() % 16:
                raise ValueError(f"{self.name}: sf[{n!r}] must be 16-byte aligned (the row declares its scale-factor blobs at 16 B), got {t.data_ptr():#x}")

    def execute(self, q8, k8, v8, o16, do8, lse, dq, dk, dv, *, workspace: torch.Tensor, stream, delta, q_T8, k_T8, do_T8, do16, sf: dict) -> None:
        """``q8 / k8 / v8 / do8 / q_T8 / k_T8 / do_T8`` the compact ``[B, S, H, D]`` e4m3 payloads and ``dq / dk / dv`` the compact
        bf16 slots (transposed here into the ``(B, H, S, D)`` views the binder demands); ``o16`` / ``do16`` EXISTING bf16
        ``[B, S, H_q, D]`` buffers bound as the adapter's dead ``o_f16`` / ``dO_f16`` (the record's pre-gate O and the gate
        backward's dO); ``lse`` the forward's fp32 ``[B, H_q, S]``; ``delta`` the block's fp32 ``[B, H_q, S_pad]`` region (the
        adapter validates its layout); ``sf`` ``{name: uint8 blob}`` for ALL SEVEN of the row's scale-factor operands (a missing
        or extra name, a wrong dtype / count / device / alignment is a typed ``ValueError`` here, before the adapter's)."""
        if self._impl is None:
            raise RuntimeError(f"{self.name}: call compile() before execute()")
        dev = torch.device(self.device)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        self._check_sf(sf, dev)
        self._impl.execute(
            q8.transpose(1, 2),
            k8.transpose(1, 2),
            v8.transpose(1, 2),
            o16.transpose(1, 2),
            do8.transpose(1, 2),
            lse,
            dq.transpose(1, 2),
            dk.transpose(1, 2),
            dv.transpose(1, 2),
            workspace=workspace,
            current_stream=cuda.CUstream(int(stream)),
            delta_tensor=delta,
            q_T_tensor=q_T8.transpose(1, 2),
            k_T_tensor=k_T8.transpose(1, 2),
            do_T_tensor=do_T8.transpose(1, 2),
            do_f16_tensor=do16.transpose(1, 2),
            **{n: sf[n] for n in self.sf_roles()},
        )


class _QkNormRopeBwd(_Stage):
    """(B5)+(B6) ONE kernel: inverse RoPE then RMSNorm backward on dQ and dK, the
    V band copied bit-exactly, fp32 ``dW_norm`` partials -- plus the fixed-order
    reduce launch. Merged for the same reason (2) and (3) merge in the forward:
    one pass over dQ, one over dK.

    (B5) RoPE is orthogonal, so the backward is the transpose. The shipped form
    is the EXACT adjoint for ANY table, ``dx = dy*cos - rotate_half(dy*sin)``
    (``y = C x + S R x`` with ``R^T = -R`` gives ``dx = C dy - R (S dy)``; the
    naive ``dy*cos - rotate_half(dy)*sin`` is exact only for duplicated-half
    tables, which ``build_rope_tables`` happens to guarantee -- pinned by the
    random-table test). Dims ``[ROPE_DIM, D)`` pass through. Same ``cos`` /
    ``sin`` tables the forward was given; the caller re-supplies them rather
    than the block saving a copy.

    (B6) per-head RMSNorm backward over D, on dQ and dK -- **not on dV**::

        dx = rstd * (g*w - x_hat * mean(g*w*x_hat))     with g = RoPE^T(dy), x_hat = x * rstd
        dW = sum over ALL tokens and ALL heads of (g * x_hat)

    ``dx`` is a per-row reduction over D (one lane group per row); ``dW_q_norm``
    / ``dW_k_norm`` are ``[D]`` accumulated over ``B*S*H`` rows -- per-CTA fp32
    partials (``[n_ctas_q, D]`` / ``[n_ctas_k, D]``, EXACTLY ``n_ctas_for(recipe,
    T)`` rows: the reduce sums every row it is handed) plus a second launch that
    reduces them in a FIXED order, never atomics (module docstring,
    "Determinism"). Needs ``x`` (pre-norm Q / K: the ``proj_slab`` bands) and the
    SAVED ``rstd``. Do **not** try to recover ``x`` by dividing the normed value
    by ``w``: undefined at a zero norm weight, hostile at a small one.

    Writes dQ_pre / dK_pre / dV into the Q / K / V bands of ``dqkvg``, reading
    the COMPACT B4 slots.

    **Under ``geometry.qk_norm=False``** the RoPE-only adjoint is traced
    (``apply_norm=False``): no norm weights, no rstd, no partials, no reduce --
    both directions typed by the kernel's own presence checks.

    Kernel: ``kernels/qk_norm_rope_bwd.py`` (LDG/STG + one butterfly shuffle +
    a 4 KiB SMEM combine; any CuTe-DSL device).
    """

    name = "qk_norm_rope_bwd"

    def __init__(self, geometry: GatedAttentionBlockGeometry, *, batch: int, seq_len: int, dtype: torch.dtype, want_dw: bool, want_amax: bool = False) -> None:
        """``want_amax`` (appended; the quantized backward): fold ``max |.|`` of the STORED Q / K / V bands into PER-CTA PARTIALS
        (one plain store per CTA of the persistent grid at CTA end); the fused epilogue reduces them with the gate backward's
        GATE-band partials into ``max |dqkvg|``."""
        self.geom = geometry
        self.batch, self.seq_len = int(batch), int(seq_len)
        self.dtype = dtype
        self.want_dw = bool(want_dw)
        self.want_amax = bool(want_amax)
        self._recipe = None

    def check_support(self) -> None:
        from .kernels.qk_norm_rope_bwd import DEFAULT_THREADS_PER_CTA, validate_shape

        if self.dtype not in _ACT_DTYPES:
            raise NotImplementedError(f"{self.name}: bf16 / fp16 only, got {self.dtype}")
        if self.want_dw and not self.geom.qk_norm:
            raise ValueError(f"{self.name}: geometry.qk_norm=False computes no RMSNorm and therefore no dW_norm; need_dw_norms must be False")
        validate_shape(self.geom.d_head, self.geom.rope_dim, DEFAULT_THREADS_PER_CTA)

    def compile(self) -> None:
        from .kernels.qk_norm_rope_bwd import compile_qk_norm_rope_bwd

        g = self.geom
        # rows_per_group=None -> the MEASURED per-arm optimum (1 with the norm, 2 RoPE-only); never pass 1 for the RoPE-only arm.
        self._recipe = compile_qk_norm_rope_bwd(
            dtype=self.dtype,
            h_q=g.h_q,
            h_kv=g.h_kv,
            d=g.d_head,
            rope_dim=g.rope_dim,
            eps=g.qk_norm_eps,
            apply_norm=bool(g.qk_norm),
            want_dw=self.want_dw,
            has_seq_lens=False,
            want_amax=self.want_amax,
        )

    def n_ctas(self) -> tuple:
        """``(n_ctas_q, n_ctas_k, n_ctas_v)`` for this declaration's ``T`` -- the dW partial plane rows."""
        from .kernels.qk_norm_rope_bwd import n_ctas_for

        if self._recipe is None:
            raise RuntimeError(f"{self.name}: call compile() before n_ctas()")
        return n_ctas_for(self._recipe, self.batch * self.seq_len)

    def n_amax_partials(self) -> int:
        """The partials a ``want_amax`` launch writes for this declaration's ``T`` -- its grid, ``sum(n_ctas())``: the
        ``amax_partials_bands`` region's length and the count the epilogue reduces."""
        from .kernels.qk_norm_rope_bwd import n_amax_partials_for

        if self._recipe is None:
            raise RuntimeError(f"{self.name}: call compile() before n_amax_partials()")
        return int(n_amax_partials_for(self._recipe, self.batch * self.seq_len))

    def execute(self, dq, dk, dv, xq, xk, rstd_q, rstd_k, w_q, w_k, cos, sin, out_q, out_k, out_v, plane_q, plane_k, *, stream, amax_out=None) -> int:
        """``amax_out`` (appended): the fp32 ``amax_partials_bands`` region of a ``want_amax`` stage (one word per CTA, every one
        overwritten), refused otherwise (the kernel's host wrapper checks both ways).  Returns the partials written (0 without
        the fold)."""
        from .kernels.qk_norm_rope_bwd import run_qk_norm_rope_bwd

        if self._recipe is None:
            raise RuntimeError(f"{self.name}: call compile() before execute()")
        return run_qk_norm_rope_bwd(
            self._recipe,
            dq,
            dk,
            dv,
            xq,
            xk,
            rstd_q,
            rstd_k,
            w_q,
            w_k,
            cos,
            sin,
            out_q,
            out_k,
            out_v,
            dw_partials_q=plane_q,
            dw_partials_k=plane_k,
            seq_lens=None,
            stream=stream,
            amax_out=amax_out,
        )

    def reduce(self, plane_q, plane_k, dw_q_norm, dw_k_norm, *, stream) -> None:
        """The second launch: the partial planes -> ``dW_q_norm`` / ``dW_k_norm`` (fp32 ``[D]``), fixed order; ``t=T`` arms the plane cross-check."""
        from .kernels.qk_norm_rope_bwd import run_dw_norm_reduce

        if self._recipe is None:
            raise RuntimeError(f"{self.name}: call compile() before reduce()")
        run_dw_norm_reduce(self._recipe, plane_q, plane_k, dw_q_norm, dw_k_norm, stream=stream, t=self.batch * self.seq_len)


# ---------------------------------------------------------------------------
# 4. Host-only checks shared by check_support and execute
# ---------------------------------------------------------------------------


def _fp4_storage_shape(shape: tuple, dtype) -> tuple:
    """The shape a tensor of ``dtype`` with the LOGICAL ``[.., K]`` ``shape`` is STORED in: ``[.., K // 2]`` for packed e2m1 codes
    (``torch.float4_e2m1fn_x2``, two codes per byte along the contiguous axis -- the forward's ``_fp4_storage_shape`` on a torch
    tensor, whose ``.shape`` already IS the storage), ``shape`` itself for every other dtype."""
    from .kernels.proj_gemm import storage_k

    return tuple(shape[:-1]) + (storage_k(int(shape[-1]), dtype),)


def _byte_range(t: torch.Tensor):
    """``[lo, hi)`` of the bytes a (possibly strided) tensor can touch; ``None`` for an empty one."""
    if t.numel() == 0:
        return None
    span = 1 + sum((int(sz) - 1) * int(st) for sz, st in zip(t.shape, t.stride()) if int(sz) > 1)
    lo = int(t.data_ptr())
    return lo, lo + span * t.element_size()


def _check_no_overlap(written, read) -> None:
    """No buffer the backward WRITES overlaps any other buffer it binds (a caller aliasing ``saved.o`` onto the
    workspace, or ``dh`` onto ``dy``, would silently corrupt an operand another stage still reads). Read-only
    inputs may alias each other freely (shared weights, ``saved.gate`` inside ``saved.proj_slab``)."""
    # ONE byte range per tensor (the pairwise loop recomputed them: 105 range computations per execute on the host-bound fp8
    # path, now one per bound tensor), then integer comparisons only.
    w_ranges = [(wn, _byte_range(wt)) for wn, wt in written]
    r_ranges = [(name, _byte_range(ten)) for name, ten in read if ten is not None]
    for i, (wn, wr) in enumerate(w_ranges):
        if wr is None:
            continue
        for name, r in w_ranges[i + 1 :] + r_ranges:
            if r is None:
                continue
            if wr[0] < r[1] and r[0] < wr[1]:
                raise ValueError(
                    f"{wn} overlaps {name} in memory ([{wr[0]:#x}, {wr[1]:#x}) vs [{r[0]:#x}, {r[1]:#x})): the backward writes {wn} while "
                    f"other stages still read {name}; give every output and the workspace its own storage"
                )


def _seq_lens_form(saved: SavedForBackward) -> Optional[str]:
    """``saved.seq_lens_form``: ``None`` for a DENSE record (``seq_lens`` is then the per-batch KV padding mask the forward ran
    with, or absent), ``"lengths"`` / ``"prefix"`` for a PACKED (THD) record (``seq_lens`` is the per-sequence ``[B]`` lengths /
    ``[B+1]`` prefix sums).  The forward writes and verifies it (``_check_saved_set``); the backward reads it as a declaration fact."""
    return saved.seq_lens_form


def _packed_record_on_dense_block(form: str) -> str:
    """The typed message for a PACKED (THD) record handed to a DENSE block -- named by the record's ``seq_lens_form`` before the
    padding decline could misname it."""
    return (
        f"SavedForBackward.seq_lens_form={form!r} marks a PACKED (THD) record -- its seq_lens are per-sequence "
        f"{'[B+1] prefix sums' if form == _THD_FORM_PREFIX else '[B] lengths'}, not a per-batch KV padding mask -- but this block was declared dense "
        f"(thd=False); declare the backward with thd=True, num_sequences and max_seq_len{' and cu_seqlens=True' if form == _THD_FORM_PREFIX else ''}"
    )


def _check_packed_lengths(saved: SavedForBackward, num_sequences: Optional[int], cu_seqlens: bool, dev) -> None:
    """The packed record's lengths under ``thd=True`` (host-only, no device read, naming the field): ``saved.seq_lens`` is REQUIRED,
    ``saved.seq_lens_form`` must say the record IS packed and in the declared FORM (``"prefix"`` iff ``cu_seqlens``), and the tensor
    is a contiguous 1-D int32 CUDA tensor on the block's device with ``B`` (lengths) or ``B+1`` (prefix sums) elements -- the
    element count is checked once ``num_sequences`` is known (the declaration's own check names a missing ``num_sequences``)."""
    form, want_form = _seq_lens_form(saved), _thd_seq_lens_form(cu_seqlens)
    if saved.seq_lens is None:
        raise ValueError(
            "thd=True: SavedForBackward.seq_lens must be the int32 [B] lengths (or [B+1] prefix sums under cu_seqlens=True) tensor the "
            "forward ran with; got None (a THD record always carries its lengths)"
        )
    if form is None:
        raise ValueError(
            "thd=True: SavedForBackward.seq_lens_form is None -- a DENSE record (its seq_lens is the per-batch KV padding mask of a padded "
            "forward, not per-sequence packed lengths); the THD backward needs the packed record a thd=True training forward wrote "
            "(seq_lens_form 'lengths' or 'prefix')"
        )
    if form != want_form:
        raise ValueError(
            f"thd=True: SavedForBackward.seq_lens_form={form!r} does not match this block's declaration (cu_seqlens={bool(cu_seqlens)} -> "
            f"{want_form!r}): the forward packed its lengths as {'[B+1] prefix sums' if form == _THD_FORM_PREFIX else '[B] lengths'}; declare the backward "
            f"with cu_seqlens={form == _THD_FORM_PREFIX}"
        )
    sl = saved.seq_lens
    if not isinstance(sl, torch.Tensor):
        raise ValueError(f"thd=True: saved.seq_lens must be an int32 tensor, got {type(sl).__name__}")
    if sl.dtype != torch.int32:
        raise ValueError(f"thd=True: saved.seq_lens must be int32 (the SDPA's packed-length operand), got {sl.dtype}")
    if sl.dim() != 1:
        raise ValueError(f"thd=True: saved.seq_lens must be 1-D ([B] lengths or [B+1] prefix sums), got shape {tuple(sl.shape)}")
    if not sl.is_contiguous():
        raise ValueError(f"thd=True: saved.seq_lens must be contiguous, got strides {tuple(sl.stride())}")
    if sl.device != dev:
        raise ValueError(f"thd=True: saved.seq_lens must live on dy's device {dev}, got {sl.device}")
    if num_sequences is not None:
        want = int(num_sequences) + (1 if cu_seqlens else 0)
        if sl.numel() != want:
            raise ValueError(
                f"thd=True: saved.seq_lens has {sl.numel()} elements; this block declares num_sequences={int(num_sequences)} with "
                f"cu_seqlens={bool(cu_seqlens)}, so it must have {want} ({'[B+1] prefix sums' if cu_seqlens else '[B] lengths'})"
            )


def _check_token_rows(name: str, ten, b: int, s: int, width: int, dtype: torch.dtype, dev, *, thd: bool) -> None:
    """A per-token caller matrix (``dy`` / ``dh`` / ``saved.h`` at ``d_model``, ``cos`` / ``sin`` at ``rope_dim``): dense ``[B, S, width]``;
    under ``thd`` the PACKED ``[T, width]`` or ``[1, T, width]`` (``B = 1, S = T`` inside the block) -- the same dtype / device /
    contiguity / 16-B rules either way (:meth:`GatedAttentionBlockFwd._check_saved_tensor`)."""
    shape = (b, s, width)
    if thd:
        nd = ten.ndim if isinstance(ten, torch.Tensor) else -1
        if nd not in (2, 3) or (nd == 3 and int(ten.shape[0]) != 1):
            got = tuple(ten.shape) if isinstance(ten, torch.Tensor) else type(ten).__name__
            raise ValueError(f"thd=True: {name} is the packed token matrix [T, {width}] (or [1, T, {width}]), got {got}")
        if nd == 2:
            shape = (b * s, width)
    GatedAttentionBlockFwd._check_saved_tensor(name, ten, shape, dtype, dev)


def _check_saved_record(
    saved: SavedForBackward,
    geom: GatedAttentionBlockGeometry,
    b: int,
    s: int,
    act: torch.dtype,
    dev,
    *,
    at: str,
    thd: bool = False,
    num_sequences: Optional[int] = None,
    cu_seqlens: bool = False,
    h_dtype: Optional[torch.dtype] = None,
) -> tuple:
    """Validate a :class:`SavedForBackward` record against the declaration (host-only, typed, names the field) and
    return ``(proj [T, N] view of proj_slab, o [T, H_q, D] view of saved.o)``.  ``at`` = ``"declaration"`` (the sample
    record at ``check_support``: a gate-copy record is a typed ``NotImplementedError`` naming the follow-up) or ``"execute"``
    (a record disagreeing with the declaration is a ``ValueError``).  ``thd`` (appended): the record is a PACKED one --
    ``saved.h`` may be rank-2 ``[T, d_model]``, ``saved.seq_lens`` is REQUIRED (``num_sequences`` entries, +1 under ``cu_seqlens``)
    and ``saved.seq_lens_form`` must say so (:func:`_check_packed_lengths`); a packed record handed to a dense block is declined
    the same way, naming the form.  Every other record tensor keeps the dense shape at ``(1, T)``.  ``h_dtype`` (appended):
    the dtype ``saved.h`` must carry -- ``None`` = ``act`` (the bf16 / fp16 backward: a record with e4m3 codes is declined,
    naming the dequantized-h contract and the ``quant=QuantSpec`` declaration), the QuantSpec's e4m3 under ``quant`` (a bf16
    ``saved.h`` is then the wrong record: the quantized backward's GEMMs read the caller's codes).  Every other activation of
    the record (slab, ``o``, ``lse``, ``rstd_*``) is ``act`` either way."""
    if not isinstance(saved, SavedForBackward):
        raise ValueError(f"saved must be a SavedForBackward record, got {type(saved).__name__}")
    # The record's two PRESENCE facts come first, before any buffer is validated, so a gate-copy or a padded record
    # gets the answer that matters (the typed decline) and not a complaint about one of its buffers.
    if saved.proj_slab is None:
        msg = (
            "SavedForBackward.proj_slab is None (the gate-copy save mode): P0 of the block backward serves proj_slab records only -- "
            "V is a band of the slab with no field of its own, so a gate-copy record needs the stage-(1) column-slice recompute "
            "(RecomputePolicy.RECOMPUTE_QK_PRE over saved.h), which a follow-up PR lands. Run the training forward with saved_gate_copy=False"
        )
        raise NotImplementedError(msg) if at == "declaration" else ValueError(msg + " (this block was declared over a proj_slab record)")
    if thd:
        _check_packed_lengths(saved, num_sequences, cu_seqlens, dev)
    elif _seq_lens_form(saved) is not None:
        # A PACKED (THD) record into a dense block: its seq_lens are per-sequence lengths, not the padding mask the dense
        # chain would read them as -- declined naming the form, before the padding decline below could misname it.
        raise ValueError(_packed_record_on_dense_block(_seq_lens_form(saved)))
    elif saved.seq_lens is not None:
        # A PADDED forward's record carries its seq_lens tensor; its PRESENCE is the fact (never its values -- no device
        # read). Declined at declaration (P0), and refused at execute for a record that contradicts the dense declaration:
        # a padded record run through the dense chain reads LSE = -inf / O = 0 on its dead rows and every gradient is NaN.
        msg = (
            "saved.seq_lens is a tensor (a PADDED forward's record): padding is not served by the block backward yet -- the "
            "sdpa_bwd_sm107 row declines seq_kv_lens_present; a follow-up PR flips it with the row's `padded` capability"
        )
        raise (
            NotImplementedError(msg)
            if at == "declaration"
            else ValueError(msg + " -- and this block was declared WITHOUT padding (seq_lens_present=False), so the record contradicts the declaration")
        )
    check = GatedAttentionBlockFwd._check_saved_tensor
    t, d = b * s, geom.d_head
    if geom.qk_norm:
        if saved.rstd_q is None or saved.rstd_k is None:
            raise ValueError("geometry.qk_norm=True: SavedForBackward.rstd_q and rstd_k are required (the forward wrote them; stage B6 consumes them)")
        check("saved.rstd_q", saved.rstd_q, (b, s, geom.h_q), torch.float32, dev)
        check("saved.rstd_k", saved.rstd_k, (b, s, geom.h_kv), torch.float32, dev)
    elif saved.rstd_q is not None or saved.rstd_k is not None:
        raise ValueError("geometry.qk_norm=False: SavedForBackward.rstd_q / rstd_k must be None (the forward wrote none; stage B6 does not exist)")
    want_h = act if h_dtype is None else h_dtype
    if isinstance(saved.h, torch.Tensor) and saved.h.dtype in _FP8_CODE_DTYPES and want_h not in _FP8_CODE_DTYPES:
        # The per-tensor FP8 / MXFP8 training forward writes the SAME bf16 record as the bf16 forward but keeps `saved.h`
        # as the caller's e4m3 codes (its own input; the device never dequantizes it). This backward is declared over the
        # activation dtype and carries no quant spec to dequantize with, so such a record is consumed with the dequantized
        # h -- a contract the generic dtype mismatch below would not name -- or differentiated natively by the quantized
        # backward (quant=QuantSpec / MxQuantSpec, the forward's spec), which reads the codes as they are.
        raise ValueError(
            f"saved.h is {saved.h.dtype}: a QUANTIZED (per-tensor FP8 / MXFP8) training forward's record, whose h is the caller's e4m3 codes. "
            f"This backward is declared over {act} and consumes such a record given the DEQUANTIZED {act} h -- "
            "dataclasses.replace(saved, h=h_dequantized), with h_dequantized = codes * descale_h (QuantSpec) or the codes scaled by their MXFP8 "
            "block scale factors (h_sf) -- and the dequantized weights; or declare the backward with quant=QuantSpec (the forward's spec) for the "
            "native per-tensor fp8 backward over the record as written (e4m3 h and weights), or with quant=MxQuantSpec for the native MXFP8 backward "
            "(e4m3 h and weights, and the caller's transposed artifacts h_t / h_t_sf and w_qkvg_t / w_qkvg_t_sf at execute)"
        )
    if want_h in _FP8_CODE_DTYPES and isinstance(saved.h, torch.Tensor) and saved.h.dtype != want_h:
        # The quantized backward's GEMMs read saved.h as an e4m3 operand (the fp8 B7's B side with descale_h folded into the epilogue;
        # the MXFP8 arm reads the caller's transposed h_t instead and keeps saved.h as the record's own codes): a bf16 h is the bf16
        # backward's record, not this one's.
        raise ValueError(
            f"saved.h is {saved.h.dtype}: a quantized backward (quant=QuantSpec / MxQuantSpec) needs the quantized forward's record: saved.h is the "
            f"caller's e4m3 codes ({want_h}), the same h the quantized training forward consumed (the per-tensor fp8 weight-gradient GEMM reads them "
            f"with descale_h folded into its epilogue). A record with a {act} h belongs to the bf16 backward (quant=None), which takes it with the "
            "dequantized weights"
        )
    _check_token_rows("saved.h", saved.h, b, s, geom.d_model, want_h, dev, thd=thd)
    check("saved.lse", saved.lse, (b, geom.h_q, s), torch.float32, dev)
    check("saved.o", saved.o, (b, s, geom.h_q, d), act, dev)
    ps = saved.proj_slab
    if not isinstance(ps, torch.Tensor):
        raise ValueError(f"saved.proj_slab must be a tensor, got {type(ps).__name__}")
    if ps.dtype != act:
        raise ValueError(f"saved.proj_slab must be the activation dtype {act} (the projection GEMM's output dtype), got {ps.dtype}")
    if ps.device != dev:
        raise ValueError(f"saved.proj_slab must live on dy's device {dev}, got {ps.device}")
    want_q, want_gate, want_k, _want_v = saved_slab_views(ps, geom, b, s)  # numel / contiguity / 16-B alignment, typed
    for nm, given, want in (("gate", saved.gate, want_gate), ("q_pre", saved.q_pre, want_q), ("k_pre", saved.k_pre, want_k)):
        if given is None:
            continue
        ok = (
            isinstance(given, torch.Tensor)
            and given.data_ptr() == want.data_ptr()
            and given.dtype == want.dtype
            and tuple(given.shape) == tuple(want.shape)
            and tuple(given.stride()) == tuple(want.stride())
        )
        if not ok:
            raise ValueError(
                f"saved.{nm} must alias saved.proj_slab's band exactly as saved_slab_views(proj_slab, geometry, batch, seq_len) spells it "
                f"([B, S, heads, D], token stride n_qkvg, storage offset at qkvg_offsets), or be None (the backward derives it from the slab)"
            )
    return ps.view(t, geom.n_qkvg), saved.o.view(t, geom.h_q, d)


def _release_side_stream(handle: int) -> None:
    """``cuStreamDestroy`` of a block's side stream -- the driver defers the destruction until the stream's work has
    drained.  Errors are swallowed: this also runs at interpreter exit, when the context may already be gone."""
    try:
        cuda.cuStreamDestroy(cuda.CUstream(handle))
    except Exception:  # noqa: BLE001 -- the teardown order at exit is not ours to control
        pass


def _capture_state(stream_handle: int) -> "tuple[bool, Optional[int]]":
    """``(capturing, capture_id)`` of ``stream_handle`` -- ``cuStreamGetCaptureInfo``; every stream joined to one
    capture reports that capture's id, and a stream whose capture has been INVALIDATED is still held by it (reported
    as ``(True, None)``: the driver gives no id for a dead capture).  A status query, legal under any capture mode from
    any thread (verified on torch 2.13 and 2.14 / CUDA 13, two device classes: it neither invalidates nor joins a
    capture another thread holds open -- the reason the guard below can run from the eager thread).  The driver
    refuses it on the legacy default stream while another stream captures (``CUDA_ERROR_STREAM_CAPTURE_IMPLICIT``);
    that stream cannot be capturing, so a refusal reads as ``(False, None)``."""
    out = cuda.cuStreamGetCaptureInfo(cuda.CUstream(stream_handle))
    if int(out[0]) != 0 or out[1] == cuda.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE:
        return False, None
    if out[1] == cuda.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE:
        return True, int(out[2])
    return True, None  # CU_STREAM_CAPTURE_STATUS_INVALIDATED


class _WgradSideStream:
    """The block-owned side stream of ``fuse_wgrad_overlap`` and its events -- created ONCE per compiled block
    (Rule 1: nothing per execute), used by :meth:`GatedAttentionBlockBwd.execute` as a fork / join pair per
    weight-gradient GEMM.

    Protocol (Rule 5 preserved exactly -- every write the caller can observe is ordered on the LAUNCH stream before
    ``execute`` returns)::

        with issue(launch, tag) as side_handle:   # ONE locked section (host-side enqueues only):
            [guard]                                  the capture guard (below)
            record ev_fork[tag] on the launch stream (after the GEMM's producer stage); side waits ev_fork[tag]
            <the caller launches the GEMM on side_handle>
            record ev_join[tag] on side              (right after the GEMM, on leaving the section)
        join(launch, tag):  [guard] + the launch stream waits ev_join[tag], under the same lock
                            (at the END of execute: the latest legal point)

    The side stream is DEDICATED: one ``cuStreamCreateWithPriority`` on the device's primary context (torch's), never a
    torch pool stream -- the pool's 32 default-priority streams go round-robin to every ``torch.cuda.Stream()`` in the
    process, so a pool stream could be the caller's launch stream (a same-stream fork / join = an in-order backward
    with no overlap and no error) or carry a stranger's work in front of the wgrad GEMM.  ``CU_STREAM_NON_BLOCKING``
    (no implicit ordering against the legacy default stream, which a default-stream launch stream would otherwise
    impose on both sides and lose the overlap to), at the LOWEST priority the device offers
    (``cuCtxGetStreamPriorityRange``'s least -- the level of torch's default-priority streams; a high-priority launch
    stream outranks it, so the GEMMs stay a filler).  Released by ``cuStreamDestroy`` when the owner is collected
    (``weakref.finalize``).  The four events are torch events (no timing), created eagerly here by one record on the
    side stream, so no CUDA object is created on the execute path -- inside a CUDA-graph capture included, where the
    record / wait pair becomes a graph edge and the side stream joins the capture (torch's ``Event`` /
    ``Stream.wait_event`` are ``cudaEventRecord`` / ``cudaStreamWaitEvent``).  The launch stream reaches here as the raw
    handle every stage takes; it is wrapped through :func:`cudnn._torch_stream.as_torch_stream` (torch's current /
    default stream object when it is one of them, an ``ExternalStream`` view otherwise -- a Python object, no CUDA
    allocation).

    Reentrant across host threads driving ONE compiled block on DIFFERENT launch streams (the convenience wrapper
    caches a block for the process; the in-order block is stateless, and this must not be less):

    * issue -- the capture guard, the fork pair (record on the launch stream, side waits), the side GEMM's launch
      call and the join record are ONE section under ``_issue_lock``.  A re-record of the shared fork event by
      another thread between record and wait would point the side stream at the OTHER launch stream's producer (this
      thread's dW GEMM running before its own operand exists); a capture forking between this thread's guard and its
      GEMM launch would absorb the eager GEMM and its join record into the graph.  Host-side enqueues only: the lock
      never waits for device work.
    * join -- the guard and the launch stream's wait under the same lock.  A record on the ONE side stream marks a
      point after everything enqueued on it so far, this execute's GEMM included, so an eager wait sees a record at
      or after the GEMM it waits for (over-waiting on another thread's GEMM at worst, never under-waiting), and no
      capture can re-record the shared event in capture mode between the guard and the wait.
    * the GEMMs of two executes serialise on the one side stream (each runs at full width anyway) and touch only their
      own execute's buffers (the caller's ``gemm_scratch_side`` carve and gradients).

    Two threads on the SAME launch stream are the trivially safe case (every record is later in that stream's order
    than the one it replaces).  Pinned by ``test_fuse_wgrad_overlap_is_reentrant_across_launch_streams`` and
    ``test_wgrad_side_stream_is_dedicated_and_released``.

    A CUDA-graph CAPTURE of ``execute`` is the one thing the shared stream cannot share.  The capturing thread's fork
    puts the side stream INTO its capture (a stream that waits an event recorded in a capturing stream joins the
    capture) until the capture ends after the join; an eager execute of the same block in that window would enqueue
    its fork wait onto a capturing stream (``cudaErrorStreamCaptureIsolation`` for it, and the capture INVALIDATED)
    or -- having forked before the capture did -- consume a capture-mode join record and have its OWN launch stream
    join the capture silently (its later work lands in the other thread's graph; both verified on torch 2.13).  So a
    capture of ``execute`` must not overlap an eager ``execute`` of the same compiled block from another thread, nor
    another capture of it; ``issue`` and ``join`` refuse the overlap TYPED (``_refuse_a_foreign_capture``: a
    ``RuntimeError`` before anything touches the capture -- the guard, the fork pair, the GEMM's enqueue and the join
    record share one lock, so the eager GEMM can never land in the graph -- and the capture itself survives; a capture
    that has been INVALIDATED still holds the side stream and is refused too, since an eager wait onto it would
    silently proceed) rather than corrupt the graph -- a detector for the overlap the contract forbids, not a licence
    for it (with the events shared, a capture that begins AND ends between one eager execute's issue section and its
    join is undetectable: the join's wait then sees the capture's re-record of the shared event).  The
    capturing thread's own second fork and its joins pass: both streams report the same capture id.  Pinned by
    ``test_wgrad_side_stream_refuses_a_foreign_capture`` (the class alone, all three overlaps),
    ``test_wgrad_side_stream_refuses_a_dead_capture``,
    ``test_fuse_wgrad_overlap_capture_and_eager_executes_do_not_overlap`` (two executes of one block) and
    ``test_fuse_wgrad_overlap_capture_cannot_absorb_an_eager_side_gemm`` (a capture that forks while an eager issue
    section is open waits for it).
    """

    TAGS = ("o", "qkvg")  # B1 (dW_o) and B7 (dW_qkvg)
    _EAGER_VS_CAPTURE = (
        "fuse_wgrad_overlap: the block's side stream is inside another thread's CUDA-graph capture; an eager execute "
        "cannot share it -- capture and eager executes of one compiled block must not overlap (one side stream per block)"
    )
    _TWO_CAPTURES = (
        "fuse_wgrad_overlap: the block's side stream is inside another CUDA-graph capture; two concurrent captures of "
        "one compiled block are unsupported (one side stream per block)"
    )
    _DEAD_CAPTURE = (
        "fuse_wgrad_overlap: the block's side stream is inside a CUDA-graph capture that has been invalidated; end that "
        "capture (its capture_end reports the error) before executing the block again (one side stream per block)"
    )

    def __init__(self, device) -> None:
        from cudnn.frost.device import device_context

        dev = torch.device(device)
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        self.device = torch.device("cuda", idx)
        with device_context(idx):  # the stream belongs to this device's primary context, whatever is current later
            err, least, _greatest = cuda.cuCtxGetStreamPriorityRange()
            if int(err) != 0:
                raise RuntimeError(f"fuse_wgrad_overlap: cuCtxGetStreamPriorityRange failed: {err}")
            err, handle = cuda.cuStreamCreateWithPriority(cuda.CUstream_flags.CU_STREAM_NON_BLOCKING, int(least))
            if int(err) != 0:
                raise RuntimeError(f"fuse_wgrad_overlap: cuStreamCreateWithPriority failed: {err}")
        self._handle = int(handle)
        self.priority = int(least)
        self._finalizer = weakref.finalize(self, _release_side_stream, self._handle)
        self.side = torch.cuda.ExternalStream(self._handle, device=self.device)
        self.ev_fork = {tag: torch.cuda.Event() for tag in self.TAGS}
        self.ev_join = {tag: torch.cuda.Event() for tag in self.TAGS}
        for ev in list(self.ev_fork.values()) + list(self.ev_join.values()):
            ev.record(self.side)  # eager creation (torch creates the CUDA event at the first record)
        self._issue_lock = threading.Lock()

    @property
    def handle(self) -> int:
        """The side stream as the raw handle the GEMM drivers take."""
        return self._handle

    def _refuse_a_foreign_capture(self, launch: torch.cuda.Stream, step: str) -> None:
        """``RuntimeError`` when the side stream is inside a CUDA-graph capture ``launch`` is not part of -- BEFORE the
        record / wait would touch that capture (class docstring).  One driver status query on the common eager path
        (the side stream is not capturing), two inside a capture; no CUDA object is created."""
        side_capturing, side_capture = _capture_state(self._handle)
        if not side_capturing:
            return
        if side_capture is None:  # a dead capture still holds the side stream: nothing may enqueue onto it, eager or captured
            raise RuntimeError(f"{self._DEAD_CAPTURE} (at the {step})")
        launch_capturing, launch_capture = _capture_state(launch.cuda_stream)
        if launch_capturing and launch_capture == side_capture:
            return  # the capturing thread's own fork / join
        raise RuntimeError(f"{self._EAGER_VS_CAPTURE if not launch_capturing else self._TWO_CAPTURES} (at the {step})")

    @contextmanager
    def issue(self, launch: torch.cuda.Stream, tag: str):
        """The side GEMM's issue as ONE locked section: the capture guard, the fork (the launch stream's work so far,
        the producer stage included, precedes the side GEMM: record on the launch stream, the side stream waits), the
        caller's GEMM launch on the yielded side-stream handle, and the join record right after it.  Under the lock a
        concurrent execute cannot re-record the fork event between record and wait, and a capture cannot fork onto
        the side stream between this thread's guard and its GEMM launch (the eager GEMM would otherwise become a node
        of that graph).  Host-side enqueues only -- the lock never waits for device work; if the launch raises, the
        lock is released and no join is recorded.  Refused typed when the side stream is inside a CUDA-graph capture
        the launch stream is not part of: a capture of ``execute`` must not overlap an eager ``execute`` (or another
        capture) of the same compiled block from another thread -- one side stream per block (class docstring)."""
        with self._issue_lock:
            self._refuse_a_foreign_capture(launch, f"fork of dW_{tag}")
            self.ev_fork[tag].record(launch)
            self.side.wait_event(self.ev_fork[tag])
            yield self._handle
            self.ev_join[tag].record(self.side)

    def join(self, launch: torch.cuda.Stream, tag: str) -> None:
        """The launch stream waits for the side GEMM -- before execute returns -- under the issue lock.  Refused typed
        when the side stream is inside a capture the launch stream is not part of: an eager wait on a capture-mode
        join record would join the launch stream to that capture (class docstring)."""
        with self._issue_lock:
            self._refuse_a_foreign_capture(launch, f"join of dW_{tag}")
            launch.wait_event(self.ev_join[tag])


# ---------------------------------------------------------------------------
# 5. The public API
# ---------------------------------------------------------------------------


class GatedAttentionBlockBwd(APIBase):
    """Gated attention block, backward. One call, one workspace, eight stages.

    Built against a forward that ran with ``save_for_backward=True`` in the
    proj_slab save mode. It does not re-derive the geometry: pass the same
    :class:`~cudnn.gated_attention_block.api.GatedAttentionBlockGeometry`
    instance, and ``check_support`` checks the saved tensors against it rather
    than trusting either alone.

    Lifecycle, exactly the forward's: ``check_support()``; ``compile()``;
    ``get_workspace_size()`` (needs the compiled artifacts -- see the method);
    ``execute(...)`` any number of times, allocation-free.
    """

    def __init__(
        self,
        sample_dy: torch.Tensor,  # [B, S, d_model]
        sample_saved: SavedForBackward,
        sample_w_qkvg: torch.Tensor,  # [N, d_model]
        sample_w_q_norm: Optional[torch.Tensor],  # [D]; None (both) iff geometry.qk_norm is False -- same positions
        sample_w_k_norm: Optional[torch.Tensor],  # [D]
        sample_cos: torch.Tensor,  # [B, S, ROPE_DIM]
        sample_sin: torch.Tensor,  # [B, S, ROPE_DIM]
        sample_w_o: torch.Tensor,  # [d_model, H_q * D]
        geometry: GatedAttentionBlockGeometry,
        *,
        recompute: RecomputePolicy = RecomputePolicy.RECOMPUTE_QK_PRE,
        # Which input gradients the caller actually wants. A partial request
        # skips whole GEMMs (dh alone needs neither wgrad), so this is a real
        # scheduling input, not a convenience -- and it must be fixed at build
        # time, because it changes which artifacts exist.
        need_dh: bool = True,
        need_dw_qkvg: bool = True,
        need_dw_o: bool = True,
        # None -> geometry.qk_norm: the norm-weight gradients exist exactly when
        # the norm does. An explicit True under qk_norm=False is a typed
        # decline (there is no dW_*_norm to compute), never a silent False.
        need_dw_norms: Optional[bool] = None,
        # APPENDED with the block backward, keyword-only, defaulted, LAST -- append-only forever.
        # The padding the forward ran with is a DECLARATION fact; P0 declines it typed (the
        # sdpa_bwd_sm107 row serves no seq_lens), from this flag OR from sample_saved.seq_lens.
        seq_lens_present: bool = False,
        # The dtype of dW_q_norm / dW_k_norm.  P0 serves fp32 ONLY (the kernel writes fp32
        # partials and the reduce fp32 [D] outputs; a cast would be an extra launch nobody measured).
        dw_norm_dtype: torch.dtype = torch.float32,
        # Fusion knob (module docstring, "Fusion knob"): the gate-backward kernel also emits the SDPA backward's
        # delta = rowsum(dO * O), and the adapter is built with external_delta=True -- one launch and one read each of
        # O and dO fewer, bitwise the unfused block.  Performance-only: the same function under either value.
        fuse_gate_bwd: bool = False,
        # Scheduling knob (module docstring, "Scheduling knob"): the two weight-gradient GEMMs (B1 dW_o, B7 dW_qkvg) --
        # consumed by nothing inside the backward -- run on a block-owned side stream forked from and joined back to
        # the launch stream through events, overlapping the SDPA backward chain and the dh dgrad.  Performance-only:
        # the same launches, the same function, bitwise; Rule 5 kept (every observable write is on the launch stream
        # before execute returns).  Declined typed when no weight-gradient GEMM exists to overlap.
        fuse_wgrad_overlap: bool = False,
        # APPENDED (THD): packed / ragged sequences -- the training forward's four knobs, same names, same meanings.  ``sample_dy``,
        # ``saved.h``, ``cos`` / ``sin`` and ``dh`` are then PACKED token matrices ([T, .] or [1, T, .]; B = 1, S = T inside the
        # block, so every per-token stage runs unchanged); ``saved.seq_lens`` is REQUIRED (the int32 [B] lengths, or [B+1]
        # prefix sums under cu_seqlens=True, the forward ran with -- the record's seq_lens_form says which) and is handed to the
        # SDPA backward for the Q and the KV side alike; ``saved.lse`` is the head-major [1, H_q, T] the packed forward wrote.
        # ``num_sequences`` (B) and ``max_seq_len`` (S_max, the longest sequence the plan admits) size the SDPA's unit grid,
        # metadata and kv-blocked workspace at build time; a caller whose B varies pads with zero-length sequences.  Caller
        # contract on the lengths (device data, never read on the host): every length in [0, max_seq_len], prefix sums
        # non-decreasing, and sum(lengths) == T -- the SDPA leaves rows past the live total unwritten while the weight-gradient
        # GEMMs contract over all T rows.  ``seq_lens_present`` (a dense padding mask) is mutually exclusive with thd;
        # ``fuse_gate_bwd`` and ``fuse_wgrad_overlap`` are both served under thd (the gate backward's delta at B = 1, S = T is the
        # packed chain's head-major [1, H_q, ceil128(T)] delta, bitwise the chain's own).
        thd: bool = False,
        num_sequences: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        cu_seqlens: bool = False,
        # APPENDED (the quantized backward): the per-tensor fp8 TRAINING forward's own `QuantSpec` (plan-time constants: the
        # static scale_q / scale_k / scale_v / scale_o of the record's operands, the descales of the e4m3 h / W_qkvg / W_o)
        # selects the native fp8 backward over the record AS WRITTEN -- e4m3 `saved.h`, e4m3 weights, the bf16 slab / O /
        # LSE / rstd -- with e4m3 GEMMs, the fp8 SDPA row and bf16 gradients out (module docstring, "The quantized backward").
        # None = the bf16 / fp16 backward (which takes a quantized record only with the DEQUANTIZED h and weights).  An
        # `MxQuantSpec` (the MXFP8 training forward's spec) selects the native MXFP8 backward over the MXFP8 record as written --
        # block-scaled dO / dQKVG and Q / K / V, the MXFP8 SDPA row, the two block-scale projection GEMMs over the caller's transposed
        # artifacts (execute(h_t=, h_t_sf=, w_qkvg_t=, w_qkvg_t_sf=)), the per-tensor fp8 out projection (module docstring, "The
        # MXFP8 backward").  Its fp4 weight modes (`w_qkvg_dtype` e2m1: an MXFP4 W_qkvg; `o_fp4`: an fp4 gated O with an e2m1 W_o of the
        # same format) select the SAME pipeline with the two data-gradient GEMMs on their fp4 block-scale rows over the caller's e2m1
        # transposed artifacts (`w_qkvg_t` in the quant spec's dtype; execute(w_o_t=, w_o_t_sf=) under `o_fp4`) and, under an NVFP4 W_o, the
        # two-level NVFP4 cast of dY (module docstring, "The fp4 weight modes").
        quant: Optional[Union[QuantSpec, MxQuantSpec]] = None,
        # The gradient-scale recipe of the quantized backward -- a DECLARATION ATTRIBUTE (it changes the e4m3 points the
        # gradients are rounded at), never a knob.  "current": every gradient's scale is derived on device from its own
        # amax pass in this step (2**(floor(log2(448 / amax)) - FP8_GRAD_SCALE_MARGIN_LOG2)); "delayed": the caller hands the
        # previous step's scale_dy / scale_do / scale_dqkvg to execute() and the amax passes still publish this step's amax
        # through quant_scalars() for the caller's next-step update.  Under an MxQuantSpec it governs the ONE per-tensor gradient
        # of that pipeline, dY (dO and dQKVG carry their 32-blocks' E8M0 scales): "delayed" takes execute(scale_dy=) alone.
        grad_scaling: str = "current",
        # APPENDED (the gradient-scale margin; keyword-only, defaulted, LAST): the "current" recipe's headroom under the e4m3 maximum,
        # in octaves -- scale = 2**(floor(log2(448 / amax)) - grad_scale_margin_log2) in EVERY gradient quantize (dY, dO and dQKVG on
        # the per-tensor fp8 chain; dY alone under an MxQuantSpec).  A DECLARATION ATTRIBUTE like grad_scaling: it moves the e4m3
        # rounding points, and it is a compile-time constant of the quantize artifacts (a different artifact per value), so it is
        # neither a knob nor a scalar slot.  An int in [0, 8]; the module constant FP8_GRAD_SCALE_MARGIN_LOG2 (0) by default, so every
        # existing caller traces the same artifacts.  Refused non-default without quant (nothing is quantized) and under
        # grad_scaling="delayed" (the caller's scales carry their own headroom there).  quant_scalars() publishes the scales that ran.
        grad_scale_margin_log2: int = FP8_GRAD_SCALE_MARGIN_LOG2,
    ):
        super().__init__()
        self._warn_experimental_api()
        # The four declaration-time contracts (typed, in this order, BEFORE anything else -- pinned by
        # test_block_end_to_end.py::test_bwd_declaration_contracts_fire_first):
        _check_norm_weights_agree(geometry.qk_norm, sample_w_q_norm, sample_w_k_norm, prefix="sample_")
        if need_dw_norms is None:
            need_dw_norms = bool(geometry.qk_norm)
        elif need_dw_norms and not geometry.qk_norm:
            raise ValueError(
                "need_dw_norms=True with geometry.qk_norm=False: a RoPE-only block has no norm weights, so there is no dW_q_norm / dW_k_norm to compute"
            )
        if not geometry.qk_norm and (sample_saved.rstd_q is not None or sample_saved.rstd_k is not None):
            raise ValueError("geometry.qk_norm=False: SavedForBackward.rstd_q / rstd_k must be None (the forward wrote none; stage B6 does not exist)")
        self.need_dw_norms = bool(need_dw_norms)
        if not isinstance(recompute, RecomputePolicy):
            raise TypeError(f"recompute must be a RecomputePolicy, got {type(recompute).__name__}")
        # Block-sparse attention (geometry.qsa) has no training record: the forward declines save_for_backward under a
        # QsaSpec, and sparse-attention training (the sparse backward and the indexer loss) is out of scope -- declined
        # here, typed, before any shape is read, so the four-band unpacks below never meet a five-band geometry.
        if geometry.qsa is not None:
            raise NotImplementedError(
                "GatedAttentionBlockBwd: the geometry declares block-sparse attention (geometry.qsa); sparse-attention training is out of scope "
                "and the forward writes no sparse training record (save_for_backward is declined under QsaSpec). The backward differentiates "
                "the dense record only."
            )
        self.thd = bool(thd)
        self.num_sequences = None if num_sequences is None else int(num_sequences)
        self.max_seq_len = None if max_seq_len is None else int(max_seq_len)
        self.cu_seqlens = bool(cu_seqlens)
        if self.thd:
            # The packed token matrix [T, d_model] or [1, T, d_model] -> B = 1, S = T inside the block.
            shp = tuple(int(x) for x in sample_dy.shape)
            if not (len(shp) == 2 or (len(shp) == 3 and shp[0] == 1)):
                raise ValueError(f"thd=True: sample_dy is the packed token matrix [T, d_model] (or [1, T, d_model]), got {shp}")
            self.batch, self.seq_len, d_model = 1, shp[-2], shp[-1]
        else:
            if sample_dy.ndim != 3:
                raise ValueError(f"sample_dy must be [B, S, d_model], got {tuple(sample_dy.shape)}")
            self.batch, self.seq_len, d_model = (int(x) for x in sample_dy.shape)
        if d_model != geometry.d_model:
            raise ValueError(f"sample_dy last dim {d_model} != geometry.d_model {geometry.d_model}")
        self.geom = geometry
        self.act_dtype = sample_dy.dtype
        self.device = sample_dy.device
        self.recompute = recompute
        self.need_dh, self.need_dw_qkvg, self.need_dw_o = bool(need_dh), bool(need_dw_qkvg), bool(need_dw_o)
        self.seq_lens_present = bool(seq_lens_present)
        self.dw_norm_dtype = dw_norm_dtype
        self.fuse_gate_bwd = bool(fuse_gate_bwd)
        self.fuse_wgrad_overlap = bool(fuse_wgrad_overlap)
        self._side: Optional[_WgradSideStream] = None  # created at compile() under fuse_wgrad_overlap
        # The quantized backward's declaration facts (typed here, cheap and device-free; every other quant check sits in
        # check_support behind the bf16 contracts, so a bf16 declaration is untouched by them).
        if quant is not None and not isinstance(quant, (QuantSpec, MxQuantSpec)):
            raise TypeError(
                f"quant must be a QuantSpec (the per-tensor fp8 training forward's spec), an MxQuantSpec (the MXFP8 training forward's spec) or None "
                f"(the bf16 / fp16 backward), got {type(quant).__name__}"
            )
        if quant is not None:
            quant.validate()  # e5m2 codes are a typed NotImplementedError, a non-positive scale a ValueError -- the forward's own contract
            # (MxQuantSpec.validate pins scale_o == descale_w_o == 1.0 under o_fp4: no per-tensor scale on either fp4 side of the out projection)
        if not isinstance(grad_scaling, str) or grad_scaling not in _GRAD_SCALING:
            raise ValueError(
                f"grad_scaling must be one of {_GRAD_SCALING} (the quantized backward's gradient-scale recipe: derived on device from this step's amax, or "
                f"the caller's previous-step scales), got {grad_scaling!r}"
            )
        if quant is None and grad_scaling != _GRAD_SCALING[0]:
            raise ValueError(
                f"grad_scaling={grad_scaling!r} is an attribute of the quantized backward (quant=QuantSpec / MxQuantSpec): a bf16 / fp16 block quantizes no "
                f"gradient and takes the default {_GRAD_SCALING[0]!r} only"
            )
        if (
            isinstance(grad_scale_margin_log2, bool)
            or not isinstance(grad_scale_margin_log2, int)
            or not (0 <= grad_scale_margin_log2 <= _GRAD_SCALE_MARGIN_LOG2_MAX)
        ):
            raise ValueError(
                f"grad_scale_margin_log2 must be an int in [0, {_GRAD_SCALE_MARGIN_LOG2_MAX}] (the 'current' recipe's power-of-two headroom under the "
                f"e4m3 maximum: scale = 2**(floor(log2(448 / amax)) - margin)), got {grad_scale_margin_log2!r}"
            )
        if quant is None and grad_scale_margin_log2 != FP8_GRAD_SCALE_MARGIN_LOG2:
            raise ValueError(
                f"grad_scale_margin_log2={grad_scale_margin_log2!r} is an attribute of the quantized backward (quant=QuantSpec / MxQuantSpec): a bf16 / "
                f"fp16 block quantizes no gradient and takes the default {FP8_GRAD_SCALE_MARGIN_LOG2!r} only"
            )
        if grad_scaling != _GRAD_SCALING[0] and grad_scale_margin_log2 != FP8_GRAD_SCALE_MARGIN_LOG2:
            raise ValueError(
                f"grad_scale_margin_log2={grad_scale_margin_log2!r} with grad_scaling={grad_scaling!r}: the margin belongs to the 'current' recipe (the "
                f"scale is derived on device from this step's amax); under 'delayed' the caller's scale_dy / scale_do / scale_dqkvg carry their own "
                f"headroom -- pass the default {FP8_GRAD_SCALE_MARGIN_LOG2!r}"
            )
        if self.thd and isinstance(quant, MxQuantSpec):
            # At construction, right after the THD shape facts and BEFORE any stage is built, so the decline names the block's
            # own attributes.  Independent of the record's content, so a placeholder record (no proj_slab yet) gets this answer
            # and not the gate-copy one.  The per-tensor fp8 backward (quant=QuantSpec) is SERVED packed -- the fp8 SDPA row's THD
            # chain reads the gate backward's packed delta (_SdpaBwdFp8, "Packed sequences") -- the MXFP8 one is not, for the one
            # reason named: its SDPA-layout MX quantize stages run the quantizer's dense arm only (the packed MXFP8 training record
            # and the packed head-major delta both exist; the packed MXFP8 backward is a follow-up).
            raise ValueError(
                "thd=True with quant=MxQuantSpec: the MXFP8 block backward is dense-only for now -- its SDPA-layout MX quantize stages run the quantizer's "
                "dense arm only (the packed per-sequence scale-factor arm the packed MXFP8 forward uses is not wired into the backward yet), while the "
                "packed MXFP8 training record and the packed head-major delta both exist; run the dense MXFP8 backward (thd=False), the packed per-tensor "
                "fp8 backward (quant=QuantSpec over the packed fp8 training record) or the packed bf16 backward over the dequantized record; the packed "
                "MXFP8 backward is a follow-up"
            )
        self.quant: Optional[Union[QuantSpec, MxQuantSpec]] = quant
        self.grad_scaling = grad_scaling
        self.grad_scale_margin_log2 = int(grad_scale_margin_log2)
        # The code dtype of saved.h and of the e4m3 weights under quant (the quant spec's `dtype`), the activation dtype otherwise -- the GEMM
        # operand dtype every MN-major rule below is spelled in.
        self.w_dtype = quant.dtype if quant is not None else self.act_dtype
        # The dtype EACH weight carries (the forward's `_expected_weight_dtypes`): under an MxQuantSpec `w_qkvg` carries `w_qkvg_dtype`
        # (packed e2m1 under the MXFP4 weight mode: STORAGE [N, d_model // 2]) and `w_o` is packed e2m1 under `o_fp4` ([d_model, H_q*D // 2]);
        # otherwise both carry `w_dtype`.  The fp4 modes' transposed artifacts follow the same two dtypes (`_check_artifacts`).
        self.o_fp4: Optional[Fp4Format] = quant.o_fp4 if isinstance(quant, MxQuantSpec) else None
        self.w_qkvg_dtype = quant.w_qkvg_dtype if isinstance(quant, MxQuantSpec) else self.w_dtype
        self.w_o_dtype = _FP4_X2 if self.o_fp4 is not None else self.w_dtype
        # the QuantSpec's plan-time constants as Python floats {QUANT_CONST_SLOTS name: value}, resolved at compile(): the scalar-init
        # launch's kernel arguments -- never device tensors (a fill at compile time has no ordering against an execute on another stream)
        self._quant_vals: Optional[dict] = None
        # The declaration's samples, re-read by check_support (shapes / dtypes / the record's presence facts; no device read).
        self._samples = dict(
            dy=sample_dy,
            saved=sample_saved,
            w_qkvg=sample_w_qkvg,
            w_q_norm=sample_w_q_norm,
            w_k_norm=sample_w_k_norm,
            cos=sample_cos,
            sin=sample_sin,
            w_o=sample_w_o,
        )
        # Stages, in launch order.  Building them costs no device work; the GEMM stages get a plan at compile().
        self._prologue = self._epilogue = None  # the quantized backwards' fused launches (_QuantPrologue / _QuantEpilogue, _MxQuantPrologue / _MxQuantEpilogue)
        # The MXFP8 backward's STANDALONE launches of the unfused chain -- the scalar init, the dY amax partials, the columnwise dO
        # quantize, the five SDPA-operand quantizes, the two canonical dQKVG quantizes: every one a job of its fused PROLOGUE / dual-axis dO
        # / EPILOGUE launch now, so every attribute stays None on every arm (kept so a reader of the unfused chain finds them named).
        self._scalar_init = self._amax_dy = None
        self._quant_do_T = self._quant_q = self._quant_q_T = self._quant_k = self._quant_k_T = self._quant_v = None
        self._quant_dqkvg = self._quant_dqkvg_T = None
        self._quant_dy_block = None  # the fp4 W_o arms' dY BLOCK quantize (MX-rowwise canonical, or the two-level NVFP4 cast)
        if self.quant is None:
            self._build_stages_bf16()
        elif isinstance(self.quant, MxQuantSpec):
            self._build_stages_mxfp8()
        else:
            self._build_stages_fp8()
        self._ws: Optional[_BwdIntermediates] = None
        # The workspace-view cache (ONE entry: the last workspace handed to execute): every typed view of the caller's buffer
        # and the shaped twins the launches take, keyed on (data_ptr, numel, device) -- `_workspace_views` / `release_workspace_views`.
        self._ws_views_key = None
        self._ws_views = None

    def _build_stages_bf16(self) -> None:
        """The bf16 / fp16 backward's stages (module docstring: the launch table) -- unchanged by the quantized arm."""
        g, act, b, s = self.geom, self.act_dtype, self.batch, self.seq_len
        t, dm, hd, n = b * s, g.d_model, g.h_q * g.d_head, g.n_qkvg
        self._out_proj_dgrad = _OutProjDgrad(m=t, k=dm, n=hd, dtype=act, label="out_proj_dgrad")
        self._gate_bwd = _SigmoidGateBwd(g, batch=b, seq_len=s, dtype=act, want_og=self.need_dw_o, want_delta=self.fuse_gate_bwd)
        self._out_proj_wgrad = _OutProjWgrad(m=dm, k=t, n=hd, dtype=act, label="out_proj_wgrad") if self.need_dw_o else None
        self._recompute_qk = _QkNormRope(g, batch=b, seq_len=s, dtype=act, want_rstd=False)
        self._compact_v = _VCompaction(g, batch=b, seq_len=s, dtype=act)
        self._sdpa = _SdpaBwd(
            g,
            batch=b,
            seq_len=s,
            dtype=act,
            device=self.device,
            external_delta=self.fuse_gate_bwd,
            thd=self.thd,
            num_sequences=self.num_sequences,
            max_seq_len=self.max_seq_len,
            cu_seqlens=self.cu_seqlens,
        )
        self._norm_bwd = _QkNormRopeBwd(g, batch=b, seq_len=s, dtype=act, want_dw=self.need_dw_norms)
        self._qkv_gate_wgrad = _QkvGateWgrad(m=n, k=t, n=dm, dtype=act, label="qkv_gate_wgrad") if self.need_dw_qkvg else None
        self._qkv_gate_dgrad = _QkvGateDgrad(m=t, k=n, n=dm, dtype=act, label="qkv_gate_dgrad") if self.need_dh else None
        self._quant_dy = self._quant_do = None
        self._stages = [
            st
            for st in (
                self._out_proj_dgrad,
                self._gate_bwd,
                self._out_proj_wgrad,
                self._recompute_qk,
                self._compact_v,
                self._sdpa,
                self._norm_bwd,
                self._qkv_gate_wgrad,
                self._qkv_gate_dgrad,
            )
            if st is not None
        ]

    def _build_stages_fp8(self) -> None:
        """The quantized (per-tensor fp8) backward's stages, in launch order (module docstring, "The quantized backward"):

        the fused PROLOGUE (scalar init + the dY amax partials + the Q / K rebuild with its e4m3 epilogue + v8) -> quantize dY
        (reduces the partials, publishes amax_dy / scale_dy / descale_dy / alpha_b1 / alpha_b2) -> (B2) e4m3 out_proj dgrad ->
        (B3) the gate backward's fp8 arm (dO, dG, e4m3 og8 under need_dw_o, delta ALWAYS, per-CTA partials of max |dO| and of
        max |dG| on a persistent grid) -> quantize dO (reduces B3's dO partials, publishes amax_do) -> (B1) e4m3
        out_proj wgrad -> (B4) the fp8 SDPA row -> (B5+B6) the norm / RoPE backward (+ per-CTA partials of the Q / K / V bands'
        max) -> the fused EPILOGUE (the dW_norm reduce + quantize dQKVG from the dG and band partials, publishing amax_dqkvg /
        alpha_b7 / alpha_b8) -> (B7) e4m3 qkv_gate wgrad -> (B8) e4m3 qkv_gate dgrad.  No atomic on any of the block's own
        kernels: every gradient amax is a max over per-CTA partials, reduced by the launch that needs it.

        Every e4m3 GEMM stage is declared with ``alpha=True`` (the fp32 epilogue scale read from a slot of the scalar block),
        a bf16 output and the EXPLICIT 64-byte MMA K; the gate backward's delta is mandatory (the row's external delta), so
        ``fuse_gate_bwd`` has no second arm here and is inert; the stage list is the same under ``thd`` -- every stage but the
        SDPA row is token-wise at ``B = 1, S = T`` (the prologue's TMA rebuild indexes tokens, the gate backward's delta at
        ``s = T`` IS the packed chain's delta) and :class:`_SdpaBwdFp8` runs the fp8 row's THD chain over the envelope.  No bf16
        rebuild, no standalone quantizers, no amax passes: the
        prologue and the epilogue own them (``_QuantPrologue`` / ``_QuantEpilogue``).
        """
        g, act, b, s, q = self.geom, self.act_dtype, self.batch, self.seq_len, self.quant
        t, dm, hd, n, d = b * s, g.d_model, g.h_q * g.d_head, g.n_qkvg, g.d_head
        e4, k64, gs = q.dtype, _FP8_GEMM_MMA_TILE_K_BYTES, self.grad_scaling
        # the prologue's init job zeroes every slot, derives descale_dp and stores the plan-time constants (the tail QUANT_CONST_SLOTS)
        # from its kernel arguments; its rebuild / v8 jobs take the static scales as kernel arguments too (the values of those slots)
        self._prologue = _QuantPrologue(
            g,
            batch=b,
            seq_len=s,
            dtype=act,
            n_slots=len(QUANT_SCALAR_SLOTS),
            const_slot0=QUANT_SCALAR_SLOTS.index(QUANT_CONST_SLOTS[0]),
            n_consts=len(QUANT_CONST_SLOTS),
        )
        # dY viewed [T, d_model / D, D]: the quantize kernels' row geometry is D wide (d_model % 256 == 0 is a declaration rule); its
        # amax comes from the prologue's partials, which this launch reduces and publishes
        self._quant_dy = _QuantizeGrad(
            g,
            batch=b,
            seq_len=s,
            dtype_in=act,
            heads=dm // d,
            name="quantize_dy",
            grad_scaling=gs,
            n_alpha=2,
            own_amax=False,
            amax_src="partials",
            margin_log2=self.grad_scale_margin_log2,
        )
        self._out_proj_dgrad = _OutProjDgrad(m=t, k=dm, n=hd, dtype=e4, label="out_proj_dgrad", mma_tile_k_bytes=k64, out_dtype=act, alpha=True)
        self._gate_bwd = _SigmoidGateBwd(
            g, batch=b, seq_len=s, dtype=act, want_og=self.need_dw_o, want_delta=True, og_fp8=self.need_dw_o, want_amax_do=True, want_amax_dg=True
        )
        # the dO amax: B3's per-CTA partials, reduced here and published
        self._quant_do = _QuantizeGrad(
            g,
            batch=b,
            seq_len=s,
            dtype_in=act,
            heads=g.h_q,
            name="quantize_do",
            grad_scaling=gs,
            n_alpha=0,
            own_amax=False,
            amax_src="partials",
            margin_log2=self.grad_scale_margin_log2,
        )
        self._out_proj_wgrad = (
            _OutProjWgrad(m=dm, k=t, n=hd, dtype=e4, label="out_proj_wgrad", mma_tile_k_bytes=k64, out_dtype=act, alpha=True) if self.need_dw_o else None
        )
        # the scalar init, the bf16 rebuild, the V compaction and the three static quantizers are the prologue's jobs; the dqkvg amax
        # pass is the gate and norm backwards' partials; the dqkvg quantize and the dW_norm reduce are the epilogue's jobs
        self._recompute_qk = None
        self._compact_v = None
        self._sdpa = _SdpaBwdFp8(
            g,
            batch=b,
            seq_len=s,
            grad_dtype=act,
            device=self.device,
            thd=self.thd,
            num_sequences=self.num_sequences,
            max_seq_len=self.max_seq_len,
            cu_seqlens=self.cu_seqlens,
        )
        self._norm_bwd = _QkNormRopeBwd(g, batch=b, seq_len=s, dtype=act, want_dw=self.need_dw_norms, want_amax=True)
        self._epilogue = _QuantEpilogue(
            g, batch=b, seq_len=s, dtype=act, want_dw=self.need_dw_norms, grad_scaling=gs, n_alpha=2, margin_log2=self.grad_scale_margin_log2
        )
        self._qkv_gate_wgrad = (
            _QkvGateWgrad(m=n, k=t, n=dm, dtype=e4, label="qkv_gate_wgrad", mma_tile_k_bytes=k64, out_dtype=act, alpha=True) if self.need_dw_qkvg else None
        )
        self._qkv_gate_dgrad = (
            _QkvGateDgrad(m=t, k=n, n=dm, dtype=e4, label="qkv_gate_dgrad", mma_tile_k_bytes=k64, out_dtype=act, alpha=True) if self.need_dh else None
        )
        self._stages = [
            st
            for st in (
                self._prologue,
                self._quant_dy,
                self._out_proj_dgrad,
                self._gate_bwd,
                self._quant_do,
                self._out_proj_wgrad,
                self._sdpa,
                self._norm_bwd,
                self._epilogue,
                self._qkv_gate_wgrad,
                self._qkv_gate_dgrad,
            )
            if st is not None
        ]

    def _build_stages_mxfp8(self) -> None:
        """The MXFP8 backward's stages, in launch order (module docstring, "The MXFP8 backward"): the fused PROLOGUE (the scalar init's
        ``descale_dp``-less arm -- the MXFP8 row has no dP scalar --, the dY amax partials, the Q / K rebuild's MX token-tile epilogue
        writing ``q8 / q_T8 / k8 / k_T8`` with their blobs, the rowwise ``v8`` straight from the slab's V band: V's compaction, since the
        forward's columnwise ``v8`` cannot serve the backward's rowwise dP operand) -> quantize dY (per-tensor e4m3, the one per-tensor
        gradient; publishes ``amax_dy / scale_dy / descale_dy / alpha_b1 / alpha_b2``) -> (B2) e4m3 out_proj dgrad -> (B3) the gate
        backward's fp8 arm (``og8`` under ``need_dw_o``, the delta ALWAYS, NO amax partials: dO is block-scaled) -> the DUAL-AXIS MXFP8
        quantize of dO (rowwise, the row's dP operand, AND columnwise, its dV operand, from one read) -> (B1) e4m3 out_proj wgrad -> (B4)
        the MXFP8 SDPA row -> (B5+B6) the norm / RoPE backward (no amax fold) -> the fused EPILOGUE (the dW_norm reduce + the dual-axis
        GEMM-canonical cast of dQKVG: rowwise ``dqkvg8`` for B8 under ``need_dh``, TRANSPOSED ``dqkvg_t8`` for B7 under ``need_dw_qkvg``;
        built iff one of its three jobs exists) -> (B7) the block-scale qkv_gate wgrad over ``dqkvg_t8`` and the caller's ``h_t`` -> (B8)
        the block-scale qkv_gate dgrad over ``dqkvg8`` and the caller's ``w_qkvg_t``.  No atomic anywhere: the one amax (dY) is a max over
        per-CTA partials.  Ten block launches with every gradient (the fp8 chain's count); every byte bitwise the unfused chain's, whose
        standalone launches (``_InitScalars``, ``_AmaxPartials``, the bf16 ``_QkNormRope`` rebuild, the single-axis ``_QuantizeMxfp8``
        quantizes, the norm backward's ``reduce``) are the jobs' one-launch forms.

        The out-projection GEMMs are the per-tensor fp8 stages of ``_build_stages_fp8`` (``alpha=True``, bf16 out, the explicit
        64-byte MMA K); the projection GEMMs are block-scale stages (``block_scale=True``: the E8M0 dequant is exact in the MMA, so
        ``alpha=False``; bf16 out; the same 64-byte MMA K, the block-scale rows' only form).  The fp4 weight modes (module docstring,
        "The fp4 weight modes") re-key two of them on the quant spec: B8 on the mixed row under an MXFP4 ``W_qkvg`` (``w_dtype`` e2m1), B2 on the
        mixed row (MXFP4 ``W_o``) or the NVFP4 x NVFP4 row (NVFP4 ``W_o``) with ONE stage inserted after the dY quantize -- the dY block
        quantize B2 reads (the MX-rowwise canonical mode, or the fp4 quantize fed ``scale_dy``'s slot) -- and the gate backward's dY
        descale arm on under NVFP4; ``o_fp4 = None`` builds the MXFP8 list, byte for byte.  The gate backward's delta is mandatory
        (the row's external delta), so ``fuse_gate_bwd`` is inert; the stage list is the DENSE one (``thd`` + an ``MxQuantSpec`` is
        declined at construction; the per-tensor fp8 arm is served packed).  The workspace carve follows the prologue's arm (:attr:`mx_prologue_arm`: no bf16 rebuild buffers under the
        MX-epilogue arm); the launch count is the module docstring's table, the MXFP8 suite's CUPTI census its check.
        """
        g, act, b, s, q = self.geom, self.act_dtype, self.batch, self.seq_len, self.quant
        t, dm, hd, n, d = b * s, g.d_model, g.h_q * g.d_head, g.n_qkvg, g.d_head
        e4, k64, gs = q.dtype, _FP8_GEMM_MMA_TILE_K_BYTES, self.grad_scaling
        # The fp4 weight modes (module docstring, "The fp4 weight modes"): an MXFP4 W_qkvg puts B8 on the mixed e4m3 x e2m1 row (its B
        # becomes the caller's packed e2m1 `w_qkvg_t`); an fp4 W_o puts B2 on a block-scale row over the caller's packed e2m1 `w_o_t` with
        # a BLOCK quantization of dY as its A -- the mixed row over an MX-rowwise e4m3 dY (MXFP4), or the NVFP4 x NVFP4 row over the
        # two-level NVFP4 cast of `scale_dy x dY` (NVFP4), undone by B3's dY descale arm.  `o_fp4 = None` is the MXFP8 list, byte for byte.
        o_fp4 = self.o_fp4
        # 1. the PROLOGUE: the scalar init (every slot zeroed, then the plan-time constants -- the tail QUANT_CONST_SLOTS -- from the
        #    launch's kernel arguments: the three live ones scale_o / descale_o / descale_w_o and 0.0 for every slot this pipeline never
        #    reads; no descale_dp), the dY amax as per-CTA partials, the Q / K rebuild's MX epilogue (q8 / q_T8 / k8 / k_T8 + blobs; no bf16
        #    recompute buffer: the carve reads `mx_prologue_arm`) and the rowwise v8 from the slab's V band -- one launch, four jobs
        self._prologue = _MxQuantPrologue(
            g,
            batch=b,
            seq_len=s,
            dtype=act,
            n_slots=len(QUANT_SCALAR_SLOTS),
            const_slot0=QUANT_SCALAR_SLOTS.index(QUANT_CONST_SLOTS[0]),
            n_consts=len(QUANT_CONST_SLOTS),
        )
        # the standalone scalar init, the dY amax partials, the bf16 rebuild and the five SDPA-operand quantizes are the prologue's jobs
        self._scalar_init = self._amax_dy = None
        self._recompute_qk = None
        self._compact_v = None
        self._quant_q = self._quant_q_T = self._quant_k = self._quant_k_T = self._quant_v = None
        # 2. the per-tensor e4m3 quantize of dY viewed [T, d_model / D, D]; its amax comes from the prologue's partials, which this launch
        #    reduces and publishes
        self._quant_dy = _QuantizeGrad(
            g,
            batch=b,
            seq_len=s,
            dtype_in=act,
            heads=dm // d,
            name="quantize_dy",
            grad_scaling=gs,
            n_alpha=2,
            own_amax=False,
            amax_src="partials",
            margin_log2=self.grad_scale_margin_log2,
        )
        # 2b. (fp4 W_o only) the BLOCK quantization of dY the out-projection dgrad reads, right after the per-tensor quantize published
        #     scale_dy on this stream: MXFP4 -> the MX-rowwise e4m3 dY viewed [T, d_model / D, D] with its GEMM-canonical E8M0 blob (the
        #     same mode as the dqkvg quantize); NVFP4 -> the shipped fp4 quantize with the appended pre-scale slot read, fed scale_dy's slot
        #     (the two-level cast: the block scale and the codes are those of scale_dy x dY; the single-level cast would zero every
        #     16-block of a raw gradient whose amax sits under the e2m1 midpoint of the e4m3 scale floor)
        if o_fp4 is Fp4Format.MXFP4:
            self._quant_dy_block = _QuantizeMxfp8(g, batch=b, seq_len=s, dtype_in=act, heads=dm // d, axis="row", name="quantize_mxfp8_dy", sf_layout="gemm")
        elif o_fp4 is Fp4Format.NVFP4:
            self._quant_dy_block = _QuantizeFp4(g, batch=b, seq_len=s, dtype_in=act, heads=dm // d, fmt=o_fp4, name="quantize_fp4_dy", scale_in=True)
        # 3. (B2) the out_proj dgrad: per-tensor e4m3 with the alpha epilogue, or -- under an fp4 W_o -- a block-scale stage over the
        #    caller's packed e2m1 W_o^T [H_q*D, d_model // 2]: the mixed row (e4m3 dy_mx8 A, E8M0 per 32 both sides) under MXFP4, the NVFP4 x
        #    NVFP4 row (packed e2m1 dy4 A, e4m3 scales per 16) under NVFP4 -- no alpha (the block dequant is exact in the MMA), bf16 out
        if o_fp4 is None:
            self._out_proj_dgrad = _OutProjDgrad(m=t, k=dm, n=hd, dtype=e4, label="out_proj_dgrad", mma_tile_k_bytes=k64, out_dtype=act, alpha=True)
        elif o_fp4 is Fp4Format.MXFP4:
            self._out_proj_dgrad = _OutProjDgrad(
                m=t, k=dm, n=hd, dtype=e4, label="out_proj_dgrad", mma_tile_k_bytes=k64, out_dtype=act, block_scale=True, w_dtype=_FP4_X2
            )
        else:
            self._out_proj_dgrad = _OutProjDgrad(
                m=t,
                k=dm,
                n=hd,
                dtype=_FP4_X2,
                label="out_proj_dgrad",
                mma_tile_k_bytes=k64,
                out_dtype=act,
                block_scale=True,
                w_dtype=_FP4_X2,
                block_size=o_fp4.block_size,
                sf_dtype=o_fp4.sf_cudnn_dtype,
            )
        # 4. (B3) the gate backward's fp8 arm WITHOUT the dO / dG amax folds (dO is block-scaled); under an NVFP4 W_o its dY descale arm
        #    multiplies B2's scaled dO_gated by descale_dy before every use (dO, dG, delta) -- exact for the power-of-two scale_dy
        self._gate_bwd = _SigmoidGateBwd(
            g,
            batch=b,
            seq_len=s,
            dtype=act,
            want_og=self.need_dw_o,
            want_delta=True,
            og_fp8=self.need_dw_o,
            want_amax_do=False,
            want_amax_dg=False,
            want_dy_descale=o_fp4 is Fp4Format.NVFP4,
        )
        # 5. dO block-quantized ROWWISE (the row's dP operand) AND COLUMNWISE (its dV operand) from ONE read of the bf16 buffer: the
        #    dual-axis arm of the quantize stage (the second half's outputs at execute: dst_T / sf_T)
        self._quant_do = _QuantizeMxfp8(g, batch=b, seq_len=s, dtype_in=act, heads=g.h_q, axis="row", name="quantize_mxfp8_do", dual=True)
        self._quant_do_T = None
        # 6. (B1) per-tensor e4m3 out_proj wgrad (dy8^T . og8 * alpha_b1)
        self._out_proj_wgrad = (
            _OutProjWgrad(m=dm, k=t, n=hd, dtype=e4, label="out_proj_wgrad", mma_tile_k_bytes=k64, out_dtype=act, alpha=True) if self.need_dw_o else None
        )
        # 7. (B4) the MXFP8 SDPA row (external delta, block-scaled dS); 8. (B5+B6) the norm / RoPE backward, no amax fold, no reduce of
        #    its own (the epilogue's job)
        self._sdpa = _SdpaBwdMxfp8(g, batch=b, seq_len=s, grad_dtype=act, device=self.device)
        self._norm_bwd = _QkNormRopeBwd(g, batch=b, seq_len=s, dtype=act, want_dw=self.need_dw_norms, want_amax=False)
        # 9. the EPILOGUE: the fixed-order dW_norm reduce (need_dw_norms) + dQKVG block-quantized from ONE read into the GEMMs' canonical
        #    F8_128x4 scale-factor order -- rowwise [T, N] (B8's A, need_dh) and TRANSPOSED [N, T] (B7's A, need_dw_qkvg: whole 32-token
        #    blocks along T, the B*S % 32 rule of check_support); no launch when none of the three jobs exists
        self._epilogue = (
            _MxQuantEpilogue(g, batch=b, seq_len=s, dtype=act, want_dw=self.need_dw_norms, want_row=self.need_dh, want_col=self.need_dw_qkvg)
            if (self.need_dw_norms or self.need_dh or self.need_dw_qkvg)
            else None
        )
        self._quant_dqkvg = self._quant_dqkvg_T = None
        # 10. + 11. the two block-scale projection GEMMs: K-major transposed artifacts on both sides, the E8M0 dequant in the MMA
        #     (no alpha), bf16 out, the 64-byte MMA K (the block-scale rows' form; `_forced_tile_name` names the K64 twin).  B8's B is the
        #     caller's w_qkvg_t in the QUANT SPEC's dtype: e4m3 (MXFP8 x MXFP8) or packed e2m1 [d_model, N // 2] (the mixed row under an MXFP4
        #     W_qkvg); B7 stays e4m3 x e4m3 (h is e4m3 in every fp4 mode: the weight gradients stay 8-bit).
        self._qkv_gate_wgrad = (
            _QkvGateWgrad(m=n, k=t, n=dm, dtype=e4, label="qkv_gate_wgrad", mma_tile_k_bytes=k64, out_dtype=act, block_scale=True)
            if self.need_dw_qkvg
            else None
        )
        self._qkv_gate_dgrad = (
            _QkvGateDgrad(
                m=t,
                k=n,
                n=dm,
                dtype=e4,
                label="qkv_gate_dgrad",
                mma_tile_k_bytes=k64,
                out_dtype=act,
                block_scale=True,
                w_dtype=_FP4_X2 if q.w_qkvg_fp4 else None,
            )
            if self.need_dh
            else None
        )
        self._stages = [
            st
            for st in (
                self._prologue,
                self._quant_dy,
                self._quant_dy_block,
                self._out_proj_dgrad,
                self._gate_bwd,
                self._quant_do,
                self._out_proj_wgrad,
                self._sdpa,
                self._norm_bwd,
                self._epilogue,
                self._qkv_gate_wgrad,
                self._qkv_gate_dgrad,
            )
            if st is not None
        ]

    # -- facts ------------------------------------------------------------------

    @property
    def mx_prologue_arm(self) -> Optional[str]:
        """The MXFP8 backward's fused-prologue ARM (``MX_PROLOGUE_ARMS``) -- the plan-time fact the workspace carve is keyed on
        (``_plan_bwd_workspace(mx_prologue_arm=)``): ``"mx_epilogue"`` on this backward (the rebuild job writes the four Q / K payloads
        out of registers: no bf16 ``recompute`` / ``recompute_k`` region), ``None`` on the bf16 / fp16 and per-tensor fp8 arms."""
        return self._prologue.arm if isinstance(self.quant, MxQuantSpec) else None

    @property
    def gemm_plans(self) -> dict:
        """``{label: ProjGemmPlan}`` of the compiled GEMM stages (the tests pin tile / route / majors)."""
        return {st.label: st.plan for st in self._stages if isinstance(st, _GemmStage)}

    def _check_inputs(self, dy, w_qkvg, w_q_norm, w_k_norm, cos, sin, w_o) -> None:
        """The non-record operands: shape / dtype / device / contiguity / 16-B alignment, typed, naming the operand.
        Every one is a TMA-loaded GEMM operand or a 16-B vector-load operand of an elementwise kernel."""
        g, b, s, act, dev = self.geom, self.batch, self.seq_len, self.act_dtype, self.device
        check = GatedAttentionBlockFwd._check_saved_tensor
        _check_token_rows("dy", dy, b, s, g.d_model, act, dev, thd=self.thd)
        for nm, w, rows, k in (("w_qkvg", w_qkvg, g.n_qkvg, g.d_model), ("w_o", w_o, g.d_model, g.h_q * g.d_head)):
            self._check_weight_codes(nm, w)
            # a packed e2m1 weight is checked against its STORAGE shape [rows, k // 2] (two codes per byte along K); every other dtype
            # against the logical [rows, k] -- the forward's own per-weight rule (`GatedAttentionBlockFwd._check_weight`)
            expect = self.w_qkvg_dtype if nm == "w_qkvg" else self.w_o_dtype
            check(nm, w, _fp4_storage_shape((rows, k), expect), expect, dev)
        _check_norm_weights_agree(g.qk_norm, w_q_norm, w_k_norm)
        if g.qk_norm:
            check("w_q_norm", w_q_norm, (g.d_head,), act, dev)
            check("w_k_norm", w_k_norm, (g.d_head,), act, dev)
        _check_token_rows("cos", cos, b, s, g.rope_dim, act, dev, thd=self.thd)
        _check_token_rows("sin", sin, b, s, g.rope_dim, act, dev, thd=self.thd)

    def _check_weight_codes(self, nm: str, w) -> None:
        """A weight's dtype against the declaration, BOTH directions, naming the attribute (the generic shape / dtype check
        below would only say "must be bf16"): e4m3 codes belong to the quantized backward (``quant=QuantSpec``), which reads
        ``W_o`` / ``W_qkvg`` as e4m3 GEMM operands with ``descale_w_o`` / ``descale_w_qkvg`` folded into the epilogue; the
        bf16 / fp16 backward takes the DEQUANTIZED weights.  Under an ``MxQuantSpec`` each weight carries ITS dtype -- ``w_qkvg``
        the quant spec's ``w_qkvg_dtype`` (packed e2m1 under the MXFP4 weight mode), ``w_o`` packed e2m1 under ``o_fp4`` -- so a packed
        e2m1 weight handed to a block whose spec names e4m3 (and the reverse) is refused by the FIELD that selects it, and uint8
        storage of packed codes gets the ``.view(torch.float4_e2m1fn_x2)`` hint (torch can view, not cast, to fp4)."""
        if not isinstance(w, torch.Tensor):
            return  # the shape / dtype check names it
        fp4_codes = _FP4_X2 is not None and w.dtype == _FP4_X2
        if self.quant is None and (w.dtype in _FP8_CODE_DTYPES or fp4_codes):
            raise ValueError(
                f"{nm} is {w.dtype} (a quantized forward's weight codes) but this backward was declared without quant: declare it with "
                f"quant=<the forward's QuantSpec / MxQuantSpec> for the native quantized backward over the record as written, or hand the "
                f"{self.act_dtype} backward the DEQUANTIZED weights (codes * descale, or the codes scaled by their block scale factors)"
            )
        expect = self.w_qkvg_dtype if nm == "w_qkvg" else self.w_o_dtype
        if self.quant is not None and w.dtype != expect:
            field = "MxQuantSpec.w_qkvg_dtype" if nm == "w_qkvg" else "MxQuantSpec.o_fp4"
            if _FP4_X2 is not None and expect == _FP4_X2:
                hint = (
                    " -- torch can VIEW but not cast to fp4: hand over the packed codes as storage.view(torch.float4_e2m1fn_x2), never uint8"
                    if w.dtype == torch.uint8
                    else ""
                )
                raise ValueError(
                    f"quant=MxQuantSpec: {nm} must be the forward's packed torch.float4_e2m1fn_x2 codes ({field} names an e2m1 {nm}: the block-scale "
                    f"dgrad reads the caller's e2m1 transposed artifact against it), got {w.dtype}{hint}"
                )
            if fp4_codes:
                raise ValueError(
                    f"quant={type(self.quant).__name__}: {nm} is torch.float4_e2m1fn_x2 but this block expects the forward's {expect} codes -- a packed "
                    f"e2m1 {nm} rides the MXFP8 pipeline's fp4 weight mode only: declare it with {field} as the forward did"
                )
            raise ValueError(
                f"quant={type(self.quant).__name__}: {nm} must be the forward's {expect} codes (the quantized backward's GEMMs read W_o / W_qkvg as "
                f"e4m3 operands -- the per-tensor arm with descale_w_o / descale_w_qkvg folded into the epilogue, the MXFP8 arm through W_o's per-tensor "
                f"descale and the caller's block-scaled w_qkvg_t), got {w.dtype}; a dequantized {self.act_dtype} weight belongs to the bf16 backward "
                "(quant=None)"
            )

    # -- support ------------------------------------------------------------

    def check_support(self) -> bool:
        """Validate the declaration, the saved set against the geometry and the
        policy, then ask every enabled stage.  Every decline is typed and, where
        the row the block binds is the reason, names it -- in this order:

        ``geometry.validate()``; the activation dtype (bf16 / fp16; under
        ``quant`` bf16 only -- the quantized backward's record and gradients
        are bf16, named with the attribute);
        ``RecomputePolicy.RECOMPUTE_GATE`` (reserved -- the GATE is always
        saved); a gate-copy record (``saved.proj_slab`` None: a follow-up PR); a
        ``need_*`` combination that leaves no work; ``fuse_wgrad_overlap`` with
        no weight-gradient GEMM to overlap; under ``thd`` the packed-sequence
        declines in this order -- ``seq_lens_present`` (mutually exclusive), the record's
        ``seq_lens`` / ``seq_lens_form`` (REQUIRED, the declared form, a
        contiguous 1-D int32 tensor of ``B`` or ``B+1`` entries on ``dy``'s
        device), ``num_sequences`` / ``max_seq_len`` present, ``T >= 1``, the
        bounds ``num_sequences >= 1``, ``2 <= max_seq_len <= T`` and
        ``num_sequences * max_seq_len >= T`` -- and, dense, the THD-only knobs
        refused (``thd`` together with an ``MxQuantSpec`` is declined at CONSTRUCTION,
        naming both attributes: the backward's SDPA-layout MX quantizes run the quantizer's
        dense arm only, the packed MXFP8 training record exists; the per-tensor
        fp8 backward is served packed); ``dw_norm_dtype`` other than fp32; a PACKED record handed to a
        dense block; padding (``seq_lens_present`` or ``sample_saved.seq_lens``
        on a dense block -- the ``sdpa_bwd_sm107`` row declines it, a follow-up
        PR flips it; no device read); the record buffers (shape / dtype /
        contiguity / 16-B alignment, the slab's bands aliasing; under ``quant``
        ``saved.h`` must be the QuantSpec's e4m3 codes, without it a record with e4m3
        codes is declined naming the dequantized-h contract and the
        ``quant=QuantSpec`` declaration); the operands (the weights carry the
        QuantSpec's codes under ``quant`` and may not without it, both named);
        ``seq_len >= 2`` (S_q = 1 is decode, which the adapter would refuse with
        an untyped ``ValueError``; under ``thd`` the bound is on ``max_seq_len``
        above); ``rope_dim > 0``;
        under an ``MxQuantSpec`` with ``need_dw_qkvg``, ``B*S % 32 == 0`` (the
        projection weight gradient contracts over the tokens through the
        block-scale GEMM's 32-element K blocks and the transposed quantize writes
        whole 32-token blocks; the fixes named: a multiple of 32, ``need_dw_qkvg=False``,
        the per-tensor fp8 backward);
        the TMA 16-byte rule on the MN-major GEMM operands (at the GEMM operand
        dtype: 16 elements under ``quant``; the token axis ``B*S`` the two
        weight-gradient GEMMs contract over is NOT bound by it -- no operand of
        an MN-major wgrad is K-contiguous, so any ``B*S`` is served); the forced GEMM
        tile's precondition (``d_model % 256 == 0`` -- the determinism
        contract's premise, never a silent heuristic fallback); the mask knobs
        the row cannot serve (``window_left == 0``, ``window_right > 0``); Rubin
        only; then each stage's own ``check_support`` (the adapter's contract
        last).

        After ``compile()`` the declaration's sample tensors are released (the
        convenience wrapper caches the block for the process lifetime; it must
        hold artifacts, not the first call's buffers), so a repeated call is a
        no-op returning the already-established ``True``.
        """
        if self._samples is None:
            return True
        g, act = self.geom, self.act_dtype
        g.validate()
        if act not in _ACT_DTYPES:
            raise NotImplementedError(f"gated_attention_block backward serves bf16 / fp16 only (the elementwise kernels and the sdpa_bwd_sm107 row), got {act}")
        if self.quant is not None and act != torch.bfloat16:
            raise ValueError(
                f"quant={type(self.quant).__name__} with a {act} sample_dy: the quantized backward's activation dtype is bf16 -- the quantized training "
                "forward writes a bf16 record (slab, O) and the backward's gradients are bf16 before their e4m3 cast; declare dy (and dh / dW_*) in "
                "torch.bfloat16"
            )
        if self.recompute is RecomputePolicy.RECOMPUTE_GATE:
            raise NotImplementedError(
                "RecomputePolicy.RECOMPUTE_GATE is reserved: the training forward always saves the GATE (as saved.gate or as a proj_slab band), "
                "so there is nothing to drop; use SAVE_ALL or RECOMPUTE_QK_PRE"
            )
        sv = self._samples["saved"]
        if not isinstance(sv, SavedForBackward):
            raise ValueError(f"sample_saved must be a SavedForBackward record, got {type(sv).__name__}")
        if sv.proj_slab is None:
            _check_saved_record(
                sv, g, self.batch, self.seq_len, act, self.device, at="declaration", thd=self.thd, num_sequences=self.num_sequences, cu_seqlens=self.cu_seqlens
            )  # raises the typed gate-copy decline
        if not (self.need_dh or self.need_dw_qkvg or self.need_dw_o or self.need_dw_norms):
            raise ValueError("no work: need_dh, need_dw_qkvg, need_dw_o and need_dw_norms are all False -- nothing to compute")
        if self.fuse_wgrad_overlap and not (self.need_dw_o or self.need_dw_qkvg):
            raise ValueError(
                "fuse_wgrad_overlap=True with need_dw_o=False and need_dw_qkvg=False: no weight-gradient GEMM exists to overlap "
                "(the knob schedules dW_o / dW_qkvg on a side stream); pass fuse_wgrad_overlap=False"
            )
        t_tokens = self.batch * self.seq_len
        if self.thd:
            # Packed sequences: the typed THD declines, in this order, before anything reads the record's buffers.  The fusion knob
            # fuse_gate_bwd is served here as dense -- the gate backward's delta at B = 1, S = T is the packed chain's head-major
            # [1, H_q, ceil128(T)] delta (the adapter's external_delta_shape under thd) -- so it has no row in this list.
            if self.seq_lens_present:
                raise ValueError(
                    "thd=True and seq_lens_present=True are mutually exclusive on GatedAttentionBlockBwd: under THD saved.seq_lens carries the "
                    "per-sequence packed lengths ([B] int32 lengths, or [B+1] int32 prefix sums with cu_seqlens=True) for the Q and the KV side "
                    "alike, and there is no per-batch KV padding mask (the SDPA backward adapter carries the lengths in its packed metadata); "
                    "pass seq_lens_present=False"
                )
            _check_packed_lengths(sv, self.num_sequences, self.cu_seqlens, self.device)
            if self.num_sequences is None or self.max_seq_len is None:
                raise ValueError(
                    "thd=True needs num_sequences (B: the length tensor has B entries, or B+1 prefix sums under cu_seqlens=True) and max_seq_len "
                    "(S_max, the longest sequence the plan admits): the SDPA's unit grid, metadata and the backward's kv-blocked workspace are "
                    "sized from them at build time"
                )
            if t_tokens == 0:
                raise ValueError(
                    "thd=True needs T >= 1 packed tokens (sample_dy has 0 rows): the SDPA adapters refuse a zero packed capacity ('the packed token "
                    "capacities must be positive') and a GEMM over M = 0 has nothing to launch -- an empty step is the caller's early-out"
                )
            if not (self.num_sequences >= 1 and 2 <= self.max_seq_len <= t_tokens and self.num_sequences * self.max_seq_len >= t_tokens):
                # The product bound is THE guard against a SILENT truncation: the SDPA backward's packed capacity is
                # min(num_sequences * max_seq_len, T) -- below T the chain processes only the first B * S_max tokens (its delta
                # sized to the cap, its dS rows from it, its descriptors clamped at it) and reports NOTHING; the packed forward's
                # units past its envelope likewise never run.  Declined here, with the symptom named, instead.
                raise ValueError(
                    f"thd=True: need num_sequences >= 1, 2 <= max_seq_len <= T and num_sequences * max_seq_len >= T (every length is <= "
                    f"max_seq_len and the lengths sum to T; S = 1 is decode, out of the prefill bodies' scope; a smaller product would cap the "
                    f"SDPA backward's packed capacity below T); got num_sequences={self.num_sequences}, max_seq_len={self.max_seq_len}, T={t_tokens}"
                )
        elif self.num_sequences is not None or self.max_seq_len is not None or self.cu_seqlens:
            raise ValueError("num_sequences / max_seq_len / cu_seqlens are THD-only (thd=True); a dense [B, S, d_model] block takes none of them")
        if self.dw_norm_dtype != torch.float32:
            raise NotImplementedError(
                f"dw_norm_dtype={self.dw_norm_dtype}: P0 writes dW_q_norm / dW_k_norm in fp32 only (the kernel's partials and its reduce are fp32); a cast "
                "would be an extra launch nobody measured -- pass torch.float32"
            )
        if not self.thd:
            form = _seq_lens_form(sv)
            if form is not None:
                raise ValueError(_packed_record_on_dense_block(form))  # a THD record into a dense block: named before the padding decline could misname it
            if self.seq_lens_present or sv.seq_lens is not None:
                raise NotImplementedError(
                    "padding (seq_lens) is not served by the block backward yet: the sdpa_bwd_sm107 row declines seq_kv_lens_present, so a padded "
                    "forward's save set (saved.seq_lens is a tensor, or seq_lens_present=True) is declined here at declaration -- a follow-up PR flips it "
                    "with the row's `padded` capability"
                )
        if isinstance(self.quant, MxQuantSpec) and self.need_dw_qkvg and t_tokens % MXFP8_BLOCK_SIZE:
            # The MXFP8 arm's ONE shape rule, and it binds B*S exactly when the projection weight gradient is requested: B7 contracts
            # over the token axis through the block-scale GEMM (one E8M0 scale per 32-element K block -- build_proj_gemm's K % 32 rule)
            # and the transposed dQKVG quantize writes whole 32-token blocks; B8 contracts over N (always a multiple of 32) and the
            # out-projection GEMMs are per-tensor MN-major, so without the wgrad any T is served.  Named here, with the fixes, before
            # the GEMM stage's own K % 32 decline could answer in the driver's words.
            raise ValueError(
                f"quant=MxQuantSpec with need_dw_qkvg=True: the weight-gradient GEMM of the projection contracts over the token axis T = B*S = {t_tokens} "
                "through the block-scale GEMM, which takes one E8M0 scale per 32-element K block, and the transposed quantization of dQKVG writes whole "
                f"32-token blocks -- so B*S must be a multiple of 32 (got {t_tokens}); pad or batch the sequence to a multiple of 32, pass "
                "need_dw_qkvg=False (the data gradients dh / dW_o contract over d_model / n_qkvg and are served at any T), or run the per-tensor fp8 "
                "backward (quant=QuantSpec), whose weight gradients take no block scales"
            )
        _check_saved_record(
            sv,
            g,
            self.batch,
            self.seq_len,
            act,
            self.device,
            at="declaration",
            thd=self.thd,
            num_sequences=self.num_sequences,
            cu_seqlens=self.cu_seqlens,
            h_dtype=self.w_dtype if self.quant is not None else None,
        )
        self._check_inputs(*(self._samples[k] for k in ("dy", "w_qkvg", "w_q_norm", "w_k_norm", "cos", "sin", "w_o")))
        if not self.thd and self.seq_len < 2:  # under thd the bound is on max_seq_len (checked above; T >= max_seq_len >= 2 follows)
            raise NotImplementedError(
                f"seq_len={self.seq_len}: the sdpa_bwd_sm107 row's prefill bodies serve S_q >= 2 (S_q = 1 is decode, out of the block backward's scope)"
            )
        if g.rope_dim == 0:
            raise NotImplementedError(
                "rope_dim=0 (no RoPE) is not exercised by the block backward yet; the norm+RoPE backward kernel is traced with rope_dim > 0"
            )
        gemm_dtype = self.w_dtype  # every GEMM operand is e4m3 under quant (dy8 / og8 / dqkvg8 / h8 / the weights), the activation dtype otherwise
        elems16 = 16 // _itemsize(gemm_dtype)
        for label, extent in (("d_model", g.d_model), ("h_q * d_head", g.h_q * g.d_head), ("n_qkvg", g.n_qkvg)):
            if extent % elems16:
                raise ValueError(
                    f"{label}={extent} must be a multiple of {elems16} ({gemm_dtype}): it is the contiguous extent of an MN-major GEMM operand (the TMA 16-byte rule)"
                )
        from .kernels.proj_gemm import _forced_tile_config

        for label, extent in (("d_model", g.d_model), ("h_q * d_head", g.h_q * g.d_head)):
            if _forced_tile_config(extent) is None:
                raise NotImplementedError(
                    f"{label}={extent} is not a multiple of 256: the four backward GEMMs run the block's FORCED 256-wide N tile "
                    "(kernels/proj_gemm.py::_forced_tile_config) at one split-K slice -- the premise of the bitwise-determinism contract -- "
                    "and the block serves no heuristic fallback (a different tile, route and possibly a split-K reducer)"
                )
        if g.window_left == 0:
            raise NotImplementedError(
                "geometry.window_left=0 is not served: the sdpa_bwd_sm107 row needs window_left > 0 (a zero-width window keeps only the diagonal); use -1 for unbounded"
            )
        if g.window_right > 0:
            raise NotImplementedError(
                f"geometry.window_right={g.window_right} is not served: the sdpa_bwd_sm107 row accepts an unbounded (or 0) right band only"
            )
        dev = torch.device(self.device)
        cc = tuple(torch.cuda.get_device_capability(dev)) if dev.type == "cuda" else None
        if cc != _SM107_CC:
            raise NotImplementedError(
                f"gated_attention_block backward targets Rubin (SM{_SM107_CC[0]}{_SM107_CC[1]}) only for now; found "
                + (f"SM{cc[0]}{cc[1]}" if cc is not None else str(dev))
            )
        for st in self._stages:
            st.check_support()
        self._is_supported = True
        return True

    # -- workspace ----------------------------------------------------------

    def get_workspace_size(self) -> int:
        """Bytes the caller must provide, INCLUDING the reused SDPA backward's
        scratch (it carves from the same buffer, at an offset this layout
        reserves for it) and the GEMMs' scratch.

        **Requires ``compile()`` first** (``RuntimeError`` otherwise): the dW
        partial plane rows come from the compiled recipe's SM-fill cap, and the
        four ``plan.workspace_bytes`` exist only after the plans do -- the same
        order the forward's own ``get_workspace_size`` imposes
        (``check_support(); compile(); get_workspace_size()``).

        Closed form (bf16, ``e = 2``, per token): ``(6*H_q + 6*H_kv)*D*e`` block
        regions -- ``dO``, the ``dqkvg`` slab ``(2*H_q + 2*H_kv)*D``, ``o_gated``, the
        recomputed Q / K / V, compact ``dQ`` / ``dK`` / ``dV`` (``(5*H_q + 6*H_kv)*D*e``
        without ``o_gated``, i.e. ``need_dw_o=False``) -- + the adapter's
        ``delta + dv_part + dk_part`` (64 KiB/token at 397B: fp32 partials under GQA, the fold rounds once) + ONE dS chunk
        ``qh_chunk x S_q_pad x S_kv_pad x e`` (4.25 / 8.50 / 33.0 GiB at S = 8K /
        16K / 32K, 397B, B=1) + ``(n_ctas_q + n_ctas_k) x D x 4`` dW partials +
        ``max(plan.workspace_bytes)`` (12 MiB at the test geometry, 0 at 397B).
        Under ``fuse_gate_bwd`` the adapter's ``delta`` moves out of its scratch
        into the block's own ``delta`` region of the same size (``B x H_q x S_pad x 4``;
        ``1 x H_q x ceil128(T) x 4`` under ``thd``, the same move).
        Under ``thd`` the block's own carve is the dense ``B = 1, S = T`` one
        (``t = T``, no new slot) and only the adapter's region differs: its
        packed ``delta [1, H_q, ceil128(T)]`` (unless ``fuse_gate_bwd``), ONE head chunk of the kv-BLOCKED
        dS ``qh_chunk x ceil256(T + 256 B) x ceil128(S_max) x e``, its metadata
        and per-sequence descriptor words, and the GQA partials ``[1, T, H_q, D]``
        x2 -- declare ``max_seq_len`` tight, it is a factor of the chunk.
        Under ``quant`` (per-tensor fp8) the bf16 ``o_gated`` and ``recompute_v``
        regions are not carved and the e4m3 ``dy8`` / ``do8`` / ``og8`` / ``q8`` /
        ``k8`` / ``v8`` / ``dqkvg8`` plus the 256-B scalar block are appended
        (``(d_model + 3 H_q D + 2 H_kv D + N) - (H_q + H_kv) D e`` bytes per token
        more: ~+12 KiB/token at 397B, the bf16 rebuild buffers not carved), the ``delta``
        region is always carved, the fp32 amax PARTIALS regions follow (the dY job's and
        the gate backward's at their SMs x 8 caps, the norm backward's at its grid: a few
        tens of KiB in all), and the SDPA scratch is the fp8 row's (its e4m3 dS chunk is
        half the bf16 one; its ``qh_chunk`` may differ -- read the adapter, never assume).
        Under ``quant=MxQuantSpec`` (MXFP8) the bf16 ``recompute`` / ``recompute_k`` go too
        (the fused prologue's MX epilogue writes the four Q / K payloads out of registers:
        :attr:`mx_prologue_arm`; the bf16-rebuild arm would keep them), ``o_gated`` /
        ``recompute_v`` go, and the appended regions are the per-tensor ``dy8`` / ``og8``, every block-scaled payload
        with its E8M0 scale-factor blob -- ``do8`` / ``do_T8``, ``q8`` / ``q_T8``, ``k8`` /
        ``k_T8``, ``v8`` (``_sf_slot_bytes`` each: ``D / 32`` bytes per row), ``dqkvg8`` ``[T, N]``
        and ``dqkvg_t8`` ``[N, T]`` (``sf_blob_bytes`` each: the GEMM-canonical padded blobs) --
        the scalar block and the dY amax partials: ``d_model + 4 H_q D + 5 H_kv D + 2 N``
        bytes of codes per token plus their scale bytes, minus the ``2 (H_q + H_kv) D e`` of the
        four bf16 regions not carved; under ``MxQuantSpec.o_fp4`` one more pair LAST -- ``dy_mx8`` +
        ``sf_dy_mx`` (``d_model`` bytes of e4m3 codes per token + ``d_model / 32`` scale bytes, MXFP4)
        or ``dy4`` + ``sf_dy4`` (``d_model / 2`` bytes of packed e2m1 codes + ``d_model / 16`` scale
        bytes, NVFP4) -- and nothing else (an MXFP4 ``W_qkvg`` alone carves nothing new).  MEASURED (``get_workspace_size()`` after ``compile()``,
        default knobs, Rubin cc 10.7): at the 397B geometry, B = 1, S = 512, 122,912,000 B
        against the bf16 block's 89,031,168 B = **+64.6 KiB/token** (+81.6 with the unfused
        chain's two bf16 rebuild regions), of which the block's own carve is +47.75 KiB/token
        (the carve's arithmetic plus its 256-B alignments) and the MXFP8 row's scratch +16.9
        KiB/token (the block-scaled dS chunk and the per-Q-head partials; +17.9 at S = 1024,
        +19.9 at S = 2048: the row's share grows with S, the carve's is flat), the GEMM
        scratch 0 on both; at the test geometry (``d_model 512, h_q 8, h_kv 2``), B = 2, S = 512:
        +4.9 KiB/token (carve +12.65, row scratch +4.22, GEMM scratch -12.00: the bf16 GEMM
        plans carve a 12 MiB split-K region, the MXFP8 K64 block-scale plans 1 B).  ``test_mxfp8_workspace_size_is_honest`` checks
        the reported size is exact and never exceeded; the transposed ``dqkvg_t8`` /
        ``sf_dqkvg_t`` pair is carved only with the projection weight gradient
        (``need_dw_qkvg``); the ``delta`` region is always carved; the SDPA scratch is the
        MXFP8 row's (its block-scaled dS: ``2 + 2/32`` bytes per element, two e4m3 payloads
        plus their E8M0 atoms -- the bf16 chain's size plus 1/16; under GQA its per-Q-head
        partials: bf16 ``dv_part``, fp32 ``dk_part``).  Honest and never exceeded.
        """
        if self._ws is None:
            raise RuntimeError(
                "GatedAttentionBlockBwd.get_workspace_size() needs compile() first: the dW partial plane rows and the four GEMM plans' "
                "workspace_bytes exist only once the artifacts do (check_support(); compile(); get_workspace_size())"
            )
        return int(self._ws.total_bytes)

    def _layout(self) -> _BwdIntermediates:
        """The compiled workspace carve (``_plan_bwd_workspace``'s result); the tests read every region offset through it."""
        if self._ws is None:
            raise RuntimeError("call compile() before _layout()")
        return self._ws

    def release_workspace_views(self) -> None:
        """Drop the cached views of the last workspace ``execute`` saw.  The block keeps ONE entry of typed views (and the
        shaped twins built from them) so a training loop that reuses its workspace constructs no view on the execute path;
        those views hold a reference to that buffer's storage, so a caller that frees its workspace while the block lives
        (the convenience wrapper allocates one per call) releases them here -- the next ``execute`` rebuilds them for whatever
        buffer it is handed (the cache keys on the pointer, size and device and never assumes a hit)."""
        self._ws_views_key = None
        self._ws_views = None

    def _workspace_views(self, workspace: torch.Tensor) -> SimpleNamespace:
        """Every typed view of the caller's workspace the stages bind, and the shaped twins the launches take -- built once per
        workspace (keyed on ``(data_ptr, numel, device)``) and reused by every later ``execute`` handed the same buffer.  Views
        only, never a copy or an allocation (Rule 1); the key is checked on EVERY call, so a different buffer (or the same
        address re-allocated with another size) rebuilds.  ``_check_workspace`` ran first.  The host-bound fp8 path built ~50
        views and scalar slices per call before this (a measured ~0.4 ms of its enqueue)."""
        key = (workspace.data_ptr(), workspace.numel(), workspace.device)
        if self._ws_views is not None and self._ws_views_key == key:
            return self._ws_views
        g, b, s, act = self.geom, self.batch, self.seq_len, self.act_dtype
        t, hd, n, d = b * s, g.h_q * g.d_head, g.n_qkvg, g.d_head
        ws = self._ws
        o_q, o_g, o_k, o_v = g.qkvg_offsets  # the four dense bands: a block-sparse geometry (a fifth band) is declined at declaration
        side = self._side
        v = SimpleNamespace()
        v.do_gated = _view(workspace, ws.do_gated, (t, g.h_q, d), act)
        v.do_gated_hd = v.do_gated.view(t, hd)
        v.do_gated_bshd = v.do_gated.view(b, s, g.h_q, d)
        v.dqkvg = _view(workspace, ws.dqkvg, (t, n), act)
        v.dqkvg_q = _cols(v.dqkvg, o_q, g.h_q, d)
        v.dqkvg_g = _cols(v.dqkvg, o_g, g.h_q, d)
        v.dqkvg_k = _cols(v.dqkvg, o_k, g.h_kv, d)
        v.dqkvg_v = _cols(v.dqkvg, o_v, g.h_kv, d)
        v.o_gated = _view(workspace, ws.o_gated, (t, g.h_q, d), act) if (self.need_dw_o and ws.o_gated >= 0) else None  # bf16 only (quant: og8)
        v.o_gated_hd = v.o_gated.view(t, hd) if v.o_gated is not None else None
        v.rq = _view(workspace, ws.recompute, (t, g.h_q, d), act) if ws.recompute >= 0 else None  # bf16 only (quant: the rebuild writes q8 / k8)
        v.rk = _view(workspace, ws.recompute_k, (t, g.h_kv, d), act) if ws.recompute_k >= 0 else None
        v.rv = _view(workspace, ws.recompute_v, (t, g.h_kv, d), act) if ws.recompute_v >= 0 else None  # bf16 only (quant: v8 is V's compaction)
        v.rq_bshd = v.rq.view(b, s, g.h_q, d) if v.rq is not None else None
        v.rk_bshd = v.rk.view(b, s, g.h_kv, d) if v.rk is not None else None
        v.rv_bshd = v.rv.view(b, s, g.h_kv, d) if v.rv is not None else None
        v.dq = _view(workspace, ws.dq, (t, g.h_q, d), act)
        v.dk = _view(workspace, ws.dk, (t, g.h_kv, d), act)
        v.dv = _view(workspace, ws.dv, (t, g.h_kv, d), act)
        v.dq_bshd, v.dk_bshd, v.dv_bshd = v.dq.view(b, s, g.h_q, d), v.dk.view(b, s, g.h_kv, d), v.dv.view(b, s, g.h_kv, d)
        v.plane_q = _view(workspace, ws.dw_partials_q, (ws.n_ctas_q, d), torch.float32) if self.need_dw_norms else None
        v.plane_k = _view(workspace, ws.dw_partials_k, (ws.n_ctas_k, d), torch.float32) if self.need_dw_norms else None
        v.sdpa_ws = workspace[ws.sdpa_bwd_ws : ws.sdpa_bwd_ws + ws.sdpa_bwd_bytes]
        v.gemm_ws = workspace[ws.gemm_scratch : ws.gemm_scratch + ws.gemm_scratch_bytes]
        v.gemm_ws_side = workspace[ws.gemm_scratch_side : ws.gemm_scratch_side + ws.gemm_scratch_side_bytes] if side is not None else v.gemm_ws
        v.delta = _view(workspace, ws.delta, ws.delta_shape, torch.float32) if ws.delta >= 0 else None  # fuse_gate_bwd, or ALWAYS under quant
        if isinstance(self.quant, MxQuantSpec):
            # The MXFP8 backward's views: the per-tensor dy8 / og8 as the fp8 arm, every block-scaled payload with its scale-factor blob
            # -- the SDPA-layout blobs as the 4-D uint8 tensors the row's adapter declares them in (the kernel's documented shapes:
            # rowwise (B, H, ceil128(S), D / 32), columnwise (B, H, D / 32, ceil128(S)); sf_v ROWWISE, the one shape the adapter
            # asserts), the GEMM-canonical blobs flat -- and the scalar block.  No dO / dG / band amax partials exist on this arm.
            e4 = self.quant.dtype
            if ws.amax_partials < 0 or ws.amax_partials_n < 1:
                raise RuntimeError(
                    "the workspace layout carves no dY amax partials region (the zero count is the pure-carve form of _plan_bwd_workspace); "
                    "compile() carves it from the compiled recipe -- call compile() before execute()"
                )
            if ws.amax_partials_do >= 0 or ws.amax_partials_bands >= 0 or ws.gate_partials_n or ws.band_partials_n:
                raise RuntimeError(
                    "the MXFP8 carve must hold no dO / dG / band amax partials (dO and dQKVG are block-scaled): the layout disagrees with the declaration"
                )
            s_pad, groups = -(-s // _SF_TILE_ROWS) * _SF_TILE_ROWS, d // MXFP8_BLOCK_SIZE
            # the bf16 rebuild buffers exist under the bf16-rebuild prologue arm only (the MX-epilogue prologue writes the four Q / K
            # payloads out of registers and the carve holds no region for them: `mx_prologue_arm`)
            v.rq = _view(workspace, ws.recompute, (t, g.h_q, d), act) if ws.recompute >= 0 else None
            v.rk = _view(workspace, ws.recompute_k, (t, g.h_kv, d), act) if ws.recompute_k >= 0 else None
            v.rq_bshd = v.rq.view(b, s, g.h_q, d) if v.rq is not None else None
            v.rk_bshd = v.rk.view(b, s, g.h_kv, d) if v.rk is not None else None
            v.dy8 = _view(workspace, ws.dy8, (t, g.d_model), e4)
            v.dy8_rows = v.dy8.view(t, g.d_model // d, d)
            v.do8 = _view(workspace, ws.do8, (t, g.h_q, d), e4)
            v.do_T8 = _view(workspace, ws.do_T8, (t, g.h_q, d), e4)
            v.do8_bshd, v.do_T8_bshd = v.do8.view(b, s, g.h_q, d), v.do_T8.view(b, s, g.h_q, d)
            v.og8 = _view(workspace, ws.og8, (t, g.h_q, d), e4) if ws.og8 >= 0 else None
            v.og8_hd = v.og8.view(t, hd) if v.og8 is not None else None
            v.q8 = _view(workspace, ws.q8, (t, g.h_q, d), e4)
            v.q_T8 = _view(workspace, ws.q_T8, (t, g.h_q, d), e4)
            v.k8 = _view(workspace, ws.k8, (t, g.h_kv, d), e4)
            v.k_T8 = _view(workspace, ws.k_T8, (t, g.h_kv, d), e4)
            v.v8 = _view(workspace, ws.v8, (t, g.h_kv, d), e4)
            v.q8_bshd, v.q_T8_bshd = v.q8.view(b, s, g.h_q, d), v.q_T8.view(b, s, g.h_q, d)
            v.k8_bshd, v.k_T8_bshd, v.v8_bshd = v.k8.view(b, s, g.h_kv, d), v.k_T8.view(b, s, g.h_kv, d), v.v8.view(b, s, g.h_kv, d)
            # the SDPA-layout scale-factor blobs: flat (the quantize launches' operand) and 4-D (the row's operand) over the same bytes
            v.sf_flat, v.sf = {}, {}
            for name, heads, columnwise in (
                ("sf_q", g.h_q, False),
                ("sf_q_T", g.h_q, True),
                ("sf_k", g.h_kv, False),
                ("sf_k_T", g.h_kv, True),
                ("sf_v", g.h_kv, False),
                ("sf_do", g.h_q, False),
                ("sf_do_T", g.h_q, True),
            ):
                n_bytes = _mx_sf_bytes(g, b, s, heads)
                flat = _view(workspace, getattr(ws, name), (n_bytes,), torch.uint8)
                v.sf_flat[name] = flat
                v.sf[name] = flat.view(b, heads, groups, s_pad) if columnwise else flat.view(b, heads, s_pad, groups)
            # the GEMM-canonical blobs (flat: the block-scale drivers bind them by byte count) and the two dQKVG payloads
            v.dqkvg8 = _view(workspace, ws.dqkvg8, (t, n), e4) if ws.dqkvg8 >= 0 else None
            v.dqkvg_rows = v.dqkvg.view(t, n // d, d)
            v.dqkvg8_rows = v.dqkvg8.view(t, n // d, d) if v.dqkvg8 is not None else None
            v.sf_dqkvg = _view(workspace, ws.sf_dqkvg, (_mx_canonical_sf_bytes(t, n),), torch.uint8) if ws.sf_dqkvg >= 0 else None
            v.dqkvg_t8 = _view(workspace, ws.dqkvg_t8, (n, t), e4) if ws.dqkvg_t8 >= 0 else None
            v.sf_dqkvg_t = _view(workspace, ws.sf_dqkvg_t, (_mx_canonical_sf_bytes(n, t),), torch.uint8) if ws.sf_dqkvg_t >= 0 else None
            v.slots = _view(workspace, ws.quant_scalars, (len(QUANT_SCALAR_SLOTS),), torch.float32)
            v.partials = _view(workspace, ws.amax_partials, (ws.amax_partials_n,), torch.float32)
            v.sc = {name: self._scalar(workspace, name) for name in QUANT_SCALAR_SLOTS}
            v.alpha_b1, v.alpha_b2 = (v.sc[name].view(1, 1, 1) for name in ("alpha_b1", "alpha_b2"))
            # the fp4 W_o arms' dY BLOCK quantization (B2's A operand) with its GEMM-canonical blob: the MX-rowwise e4m3 dy_mx8 (viewed
            # [T, d_model / D, D] for the quantize launch, [T, d_model] for the GEMM) or the packed e2m1 dy4 (uint8 bytes re-viewed as the
            # fp4 storage dtype the driver binds); None without o_fp4
            v.dy_mx8 = v.dy_mx8_rows = v.sf_dy_mx = v.dy4 = v.sf_dy4 = None
            if ws.dy_mx8 >= 0:
                v.dy_mx8 = _view(workspace, ws.dy_mx8, (t, g.d_model), e4)
                v.dy_mx8_rows = v.dy_mx8.view(t, g.d_model // d, d)
                v.sf_dy_mx = _view(workspace, ws.sf_dy_mx, (_mx_canonical_sf_bytes(t, g.d_model),), torch.uint8)
            if ws.dy4 >= 0:
                from .kernels.proj_gemm import FP4_CODES_PER_BYTE, sf_blob_bytes

                v.dy4 = _view(workspace, ws.dy4, (t, g.d_model // FP4_CODES_PER_BYTE), torch.uint8).view(_FP4_X2)
                v.sf_dy4 = _view(workspace, ws.sf_dy4, (sf_blob_bytes(t, g.d_model, self.o_fp4.block_size),), torch.uint8)
        elif self.quant is not None:
            e4 = self.quant.dtype
            if (
                ws.amax_partials < 0
                or ws.amax_partials_n < 1
                or ws.amax_partials_do < 0
                or ws.gate_partials_n < 1
                or ws.amax_partials_bands < 0
                or ws.band_partials_n < 1
            ):
                raise RuntimeError(
                    "the workspace layout carves no amax partials regions (the zero counts are the pure-carve form of _plan_bwd_workspace); "
                    "compile() carves them from the compiled recipes -- call compile() before execute()"
                )
            v.dy8 = _view(workspace, ws.dy8, (t, g.d_model), e4)
            v.dy8_rows = v.dy8.view(t, g.d_model // d, d)
            v.do8 = _view(workspace, ws.do8, (t, g.h_q, d), e4)
            v.do8_bshd = v.do8.view(b, s, g.h_q, d)
            v.og8 = _view(workspace, ws.og8, (t, g.h_q, d), e4) if ws.og8 >= 0 else None
            v.og8_hd = v.og8.view(t, hd) if v.og8 is not None else None
            v.o_dead8_bshd = (v.og8 if v.og8 is not None else v.do8).view(b, s, g.h_q, d)  # the adapter's dead `o`: og8 when it exists, else do8
            v.q8 = _view(workspace, ws.q8, (t, g.h_q, d), e4)
            v.k8 = _view(workspace, ws.k8, (t, g.h_kv, d), e4)
            v.v8 = _view(workspace, ws.v8, (t, g.h_kv, d), e4)
            v.q8_bshd, v.k8_bshd, v.v8_bshd = v.q8.view(b, s, g.h_q, d), v.k8.view(b, s, g.h_kv, d), v.v8.view(b, s, g.h_kv, d)
            v.dqkvg8 = _view(workspace, ws.dqkvg8, (t, n), e4)
            v.dqkvg_rows, v.dqkvg8_rows = v.dqkvg.view(t, n // d, d), v.dqkvg8.view(t, n // d, d)
            v.slots = _view(workspace, ws.quant_scalars, (len(QUANT_SCALAR_SLOTS),), torch.float32)
            v.partials = _view(workspace, ws.amax_partials, (ws.amax_partials_n,), torch.float32)
            v.partials_do = _view(workspace, ws.amax_partials_do, (ws.gate_partials_n,), torch.float32)
            v.partials_dg = _view(workspace, ws.amax_partials_dg, (ws.gate_partials_n,), torch.float32)
            v.partials_bands = _view(workspace, ws.amax_partials_bands, (ws.band_partials_n,), torch.float32)
            v.sc = {name: self._scalar(workspace, name) for name in QUANT_SCALAR_SLOTS}
            v.alpha_b1, v.alpha_b2, v.alpha_b7, v.alpha_b8 = (v.sc[name].view(1, 1, 1) for name in ("alpha_b1", "alpha_b2", "alpha_b7", "alpha_b8"))
            # the fp8 row's scalars minus the caller's scale_dP (added per call): eleven SLOTS of the scalar block -- the plan-time
            # constants and descale_dp the prologue's init job stores on every execute, descale_dO published by the dO quantize
            v.row_scalars = dict(
                descale_q=v.sc["descale_q"],
                descale_k=v.sc["descale_k"],
                descale_v=v.sc["descale_v"],
                descale_s=v.sc["descale_s"],
                scale_s=v.sc["scale_s"],
                descale_o=v.sc["descale_o"],
                descale_dO=v.sc["descale_do"],
                descale_dP=v.sc["descale_dp"],
                scale_dQ=v.sc["scale_dqkv"],
                scale_dK=v.sc["scale_dqkv"],
                scale_dV=v.sc["scale_dqkv"],
            )
        self._ws_views_key, self._ws_views = key, v
        return v

    def _check_workspace(self, workspace) -> None:
        """The caller's workspace against the carve: ``get_workspace_size()`` bytes of contiguous uint8 on the block's device at
        the carve's base alignment -- the same checks for ``execute`` and ``quant_scalars``."""
        if workspace is None:
            raise ValueError("workspace is required: get_workspace_size() bytes of uint8 on dy's device")
        req, dev = self.get_workspace_size(), self.device
        if not isinstance(workspace, torch.Tensor) or workspace.dtype != torch.uint8 or not workspace.is_contiguous() or workspace.device != dev:
            got = (
                f"{workspace.dtype} strides {tuple(workspace.stride())} on {workspace.device}"
                if isinstance(workspace, torch.Tensor)
                else type(workspace).__name__
            )
            raise ValueError(f"workspace must be a contiguous uint8 tensor on {dev}, got {got}")
        if workspace.numel() < req:
            raise ValueError(f"workspace is {workspace.numel()} bytes, need {req}")
        if workspace.data_ptr() % self._ws.base_align:
            raise ValueError(f"workspace base must be {self._ws.base_align}-byte aligned (the carve assumes it), got data_ptr={workspace.data_ptr():#x}")

    def _scalar(self, workspace: torch.Tensor, name: str) -> torch.Tensor:
        """The 1-element fp32 VIEW of slot ``name`` of the scalar block (``QUANT_SCALAR_SLOTS``): byte offset
        ``quant_scalars + QUANT_SCALAR_STRIDE * index`` -- derived, never a literal; zero-copy."""
        return _view(workspace, self._ws.quant_scalars + QUANT_SCALAR_STRIDE * QUANT_SCALAR_SLOTS.index(name), (1,), torch.float32)

    def quant_scalars(self, workspace: torch.Tensor) -> dict:
        """The quantized backward's fp32 scalar block as ``{name: 1-element fp32 view}`` over ``QUANT_SCALAR_SLOTS`` -- the
        amax of every quantized gradient (``amax_dy`` / ``amax_do`` / ``amax_dqkvg``) and the fp8 SDPA row's ``amax_dp`` of
        this execute, the scales / descales the quantize launches published, the four GEMM epilogue products,
        ``descale_dp``, and the ``QuantSpec``'s plan-time constants (the tail, ``QUANT_CONST_SLOTS``: the static scales, the
        row's descales, ``scale_s`` / ``descale_s``, the alpha factors) the scalar-init launch stored this execute.  Zero-copy
        views of the caller's workspace (Rule 1), valid until the next ``execute`` zeroes the block:
        the caller synchronises its stream after the step before reading them (a ``"delayed"`` recipe reads this step's
        amax here to set the next step's ``scale_dy`` / ``scale_do`` / ``scale_dqkvg``); nothing here reads the device.
        ``ValueError`` on a block declared without ``quant``; ``RuntimeError`` before ``compile()``.

        The SAME 29 names under an ``MxQuantSpec``, of which EIGHT are live -- the dY point ``amax_dy`` / ``scale_dy`` /
        ``descale_dy`` / ``alpha_b1`` / ``alpha_b2`` (published by the dY quantize: dY is the one per-tensor gradient of that
        pipeline) and the constants ``scale_o`` / ``descale_o`` (B3's e4m3 ``og8`` scale and ``alpha_b1``'s factor) /
        ``descale_w_o`` (``alpha_b2``'s factor) -- while the other 21 read exactly 0.0 after every execute: the dO / dQKVG / dP
        scalars (dO and dQKVG carry their 32-blocks' E8M0 scales, the MXFP8 row has no dP scalar) and the per-tensor static
        scales / descales of the fp8 record (``scale_q / k / v``, ``descale_q / k / v``, ``descale_h``, ``descale_w_qkvg``,
        ``scale_s`` / ``descale_s``, ``scale_dqkv``) -- 0.0 by the init launch, never 1.0, so a misrouted read of a dead slot
        zeroes an output the finite / sentinel checks catch instead of passing silently.  Under an fp4 ``W_o``
        (``MxQuantSpec.o_fp4``) the same eight are written -- ``scale_o`` / ``descale_o`` / ``descale_w_o`` read 1.0 (``MxQuantSpec.validate`` pins them)
        and ``alpha_b2 = descale_dy`` is published by the dY quantize but read by no GEMM (the block-scale out-projection dgrad has no
        alpha epilogue; ``descale_dy`` is what the gate backward's dY descale arm reads under NVFP4)."""
        if self.quant is None:
            raise ValueError(
                "quant_scalars() belongs to the quantized backward (quant=QuantSpec / MxQuantSpec): this block was declared without quant and has no "
                "scalar block"
            )
        if self._ws is None:
            raise RuntimeError("call compile() before quant_scalars() (the scalar block is a region of the compiled carve)")
        self._check_workspace(workspace)
        return {name: self._scalar(workspace, name) for name in QUANT_SCALAR_SLOTS}

    def update_quant_scales(self, spec: Union[QuantSpec, MxQuantSpec]) -> None:
        """Re-point a COMPILED quantized backward's plan-time constants at ``spec`` without recompiling: ``self.quant = spec`` and
        ``self._quant_vals`` re-resolved (:meth:`_quant_const_values`) -- the host floats the NEXT execute's prologue launch stores
        into the scalar block from its kernel arguments (``QUANT_CONST_SLOTS``), stream-ordered by construction and with no device
        write here.  So the SDPA operands ``q8 / k8 / v8`` are rebuilt at the forward's CURRENT ``scale_q / scale_k / scale_v``, B3's
        ``og8`` at its ``scale_o``, and the GEMM alphas at its ``descale_h / descale_w_qkvg / descale_w_o`` -- the training-loop
        recipe: the same ``spec`` the forward of that layer and step ran at, applied right before its backward's ``execute``
        (``GatedAttentionBlockFwd.update_quant_scales`` is the forward half).  A CUDA graph captured BEFORE the call keeps the old
        constants (they are kernel arguments of the captured launch); re-capture after a recalibration.

        Typed refusals, before anything changes: a block declared without ``quant`` (``ValueError``); a spec of the other class
        (``TypeError``: the two classes select different chains); a differing plan fact -- ``dtype``, and under MXFP8 ``block_size``
        / ``w_qkvg_dtype`` / ``o_fp4`` -- (``ValueError`` naming the field: those select the kernels, the artifacts and the carve);
        ``spec``'s own ``validate()`` exactly as the declaration applied it (a zero / inf / NaN scale is its own ``ValueError``);
        a block not yet compiled (``RuntimeError``)."""
        if self.quant is None:
            raise ValueError(
                "update_quant_scales() belongs to the quantized backward (quant=QuantSpec / MxQuantSpec): this block was declared without quant and "
                "holds no plan-time constant to update"
            )
        if type(spec) is not type(self.quant):
            raise TypeError(
                f"update_quant_scales(): spec must be a {type(self.quant).__name__}, the class this block was declared with (QuantSpec and "
                f"MxQuantSpec select different chains), got {type(spec).__name__}"
            )
        for name in ("dtype", "block_size", "w_qkvg_dtype", "o_fp4") if isinstance(self.quant, MxQuantSpec) else ("dtype",):
            if getattr(spec, name) != getattr(self.quant, name):
                raise ValueError(
                    f"update_quant_scales(): {type(spec).__name__}.{name} is a plan fact (it selects the kernels, the artifacts and the carve): "
                    f"declared {getattr(self.quant, name)!r}, got {getattr(spec, name)!r}; only the scales may change -- declare a new block for a "
                    f"new {name}"
                )
        spec.validate()  # the declaration's own call: e5m2 codes a typed NotImplementedError, a non-positive scale a ValueError
        if self._ws is None:
            raise RuntimeError("call compile() before update_quant_scales(): the plan-time constants it re-resolves are resolved there")
        self.quant = spec
        self._quant_vals = self._quant_const_values()

    # -- compile ------------------------------------------------------------

    @staticmethod
    def _forced_tile_name(st: _GemmStage) -> Optional[str]:
        """The tile config name a GEMM stage's plan must carry: the block's forced 256-wide N tile, at the stage's EXPLICIT MMA K
        width -- an e4m3 stage (``mma_tile_k_bytes=64``) names the catalog's 64-byte twin of the same geometry through
        ``tile_config.as_mma_tile_k``, never a typed literal."""
        from .kernels.proj_gemm import _forced_tile_config

        name = _forced_tile_config(st.n)
        if name is not None and st.mma_tile_k_bytes is not None:
            from cudnn.gemm.frost.tile_config import as_mma_tile_k, by_name

            name = as_mma_tile_k(by_name(name), int(st.mma_tile_k_bytes)).name
        return name

    def _quant_const_values(self) -> Optional[dict]:
        """The ``QuantSpec``'s plan-time constants as PYTHON FLOATS keyed by their ``QUANT_CONST_SLOTS`` name, in slot order --
        the scalar-init launch's kernel arguments (it stores them into the scalar block on every execute, on the launch
        stream, so every consumer reads a slot), never device tensors: a tensor filled here would be enqueued on the stream
        ambient at ``compile()`` with nothing ordering it before an ``execute`` on another stream.  The three static
        quantizers' ``scale_q / scale_k / scale_v`` and B3's ``scale_o``; the fp8 SDPA row's ``descale_q / descale_k /
        descale_v``, its dead ``descale_o`` (REQUIRED by the row's contract, read by nothing under the external delta; the
        SAME slot is ``alpha_b1``'s factor ``1 / scale_o``), ``scale_s`` / ``descale_s`` (``2**FP8_SCALE_S_LOG2`` and its exact
        reciprocal) and the shared ``scale_dqkv = 1.0`` (``scale_dQ = scale_dK = scale_dV``: bf16 gradients out of the row);
        the alpha factors ``descale_w_o`` (``alpha_b2``), ``descale_h`` (``alpha_b7``) and ``descale_w_qkvg`` (``alpha_b8``) the
        quantize launches multiply with the published descales.  ``None`` without ``quant``.

        Under an ``MxQuantSpec`` the SAME 14 names in the same order, with the three the MXFP8 pipeline reads at their values
        -- ``scale_o`` (B3's ``og8`` cast), ``descale_o = 1 / scale_o`` (``alpha_b1``'s factor) and ``descale_w_o`` (``alpha_b2``'s)
        -- and every other one 0.0: no per-tensor static scale exists for Q / K / V (block scales), no ``descale_h`` /
        ``descale_w_qkvg`` (the block-scale GEMMs dequantize in the MMA), no ``scale_s`` / ``descale_s`` / ``scale_dqkv`` (the
        MXFP8 row takes no scalar).  0.0 rather than 1.0 on purpose: a misrouted read of a dead slot then zeroes an output the
        finite / sentinel gates catch, where 1.0 would pass silently."""
        if self.quant is None:
            return None
        q = self.quant
        if isinstance(q, MxQuantSpec):
            vals = {name: 0.0 for name in QUANT_CONST_SLOTS}
            vals.update(scale_o=float(q.scale_o), descale_o=1.0 / float(q.scale_o), descale_w_o=float(q.descale_w_o))
            if tuple(vals) != QUANT_CONST_SLOTS:
                raise RuntimeError(
                    f"the plan-time constants {tuple(vals)} must name QUANT_CONST_SLOTS {QUANT_CONST_SLOTS} in slot order (the init launch's argument order)"
                )
            return vals
        scale_s = float(2.0**FP8_SCALE_S_LOG2)
        vals = dict(
            scale_q=float(q.scale_q),
            scale_k=float(q.scale_k),
            scale_v=float(q.scale_v),
            scale_o=float(q.scale_o),
            descale_q=1.0 / float(q.scale_q),
            descale_k=1.0 / float(q.scale_k),
            descale_v=1.0 / float(q.scale_v),
            descale_o=1.0 / float(q.scale_o),
            descale_w_o=float(q.descale_w_o),
            descale_h=float(q.descale_h),
            descale_w_qkvg=float(q.descale_w_qkvg),
            scale_s=scale_s,
            descale_s=1.0 / scale_s,
            scale_dqkv=1.0,
        )
        if tuple(vals) != QUANT_CONST_SLOTS:
            raise RuntimeError(
                f"the plan-time constants {tuple(vals)} must name QUANT_CONST_SLOTS {QUANT_CONST_SLOTS} in slot order (the init launch's argument order)"
            )
        return vals

    def compile(self) -> None:
        """Build the artifacts for the enabled stages only, then the workspace carve (and, under ``quant``, resolve the
        plan-time constants' VALUES -- Python floats, the scalar-init launch's arguments).  Nothing is written to the device
        here: a fill enqueued at compile time would sit on the ambient stream with nothing ordering it before an ``execute``
        on another stream."""
        self._ensure_support_checked()
        for st in self._stages:
            st.compile()
        # The determinism contract's premise, verified on the plans that will run: the forced tile compiled (plan.jit) under
        # its own name (the 64-byte MMA K twin for the e4m3 stages). The driver logs a WARNING and takes the graph heuristic
        # when the forced config is refused for a shape -- a different tile, route and possibly a split-K reducer -- which the
        # block does not serve.
        for st in self._stages:
            if isinstance(st, _GemmStage):
                want, plan = self._forced_tile_name(st), st.plan
                if plan.jit is None or plan.tile_config_name != want:
                    raise NotImplementedError(
                        f"{st.label} (m={st.m}, k={st.k}, n={st.n}, {st.dtype}): the forced tile {want} did not compile for this shape and the "
                        f"plan fell back to the heuristic (tile={plan.tile_config_name}, route={plan.route}); the block backward serves the "
                        "forced tile only -- the premise of its bitwise-determinism contract (module docstring, Determinism)"
                    )
        n_ctas_q, n_ctas_k = (0, 0)
        if self.need_dw_norms:
            n_ctas_q, n_ctas_k, _n_v = self._norm_bwd.n_ctas()
        gemm_scratch = max([st.workspace_bytes() for st in self._stages if isinstance(st, _GemmStage)] + [1])
        # fuse_wgrad_overlap: the side-stream GEMMs (B1 / B7) get their OWN scratch -- B7 runs concurrently with B8, and B1
        # with everything after B3, so they must never share `gemm_scratch` with the launch-stream GEMMs (the forced tile
        # at one split-K slice touches no scratch at all today; the carve is the contract, not an assumption).
        side_scratch = None
        if self.fuse_wgrad_overlap:
            side_scratch = max([st.workspace_bytes() for st in (self._out_proj_wgrad, self._qkv_gate_wgrad) if st is not None] + [1])
            self._side = _WgradSideStream(self.device)
        self._quant_vals = self._quant_const_values()
        fp8, mx = isinstance(self.quant, QuantSpec), isinstance(self.quant, MxQuantSpec)
        self._ws = _plan_bwd_workspace(
            self.geom,
            self.batch,
            self.seq_len,
            self.act_dtype,
            self.recompute,
            need=dict(dw_o=self.need_dw_o, dw_norms=self.need_dw_norms, dw_qkvg=self.need_dw_qkvg),
            sdpa_bwd_bytes=self._sdpa.scratch_workspace_bytes(),
            gemm_scratch_bytes=gemm_scratch,
            n_ctas_q=n_ctas_q,
            n_ctas_k=n_ctas_k,
            # the fp32 delta region: fuse_gate_bwd's on the bf16 backward, ALWAYS on the quantized ones (the row's external delta)
            delta_shape=self._sdpa.delta_shape if (self.fuse_gate_bwd or self.quant is not None) else None,
            side_gemm_scratch_bytes=side_scratch,
            quant=self.quant,
            # the dY amax partials: one fp32 per CTA of the amax job (the fp8 prologue's, or the MXFP8 prologue's), at most the compiled
            # recipe's persistent cap
            amax_partials_n=self._prologue.n_partials_cap if (fp8 or mx) else 0,
            # the gate backward's dO / dG partials (its persistent cap) and the norm backward's band partials (EXACTLY its grid): the
            # per-tensor fp8 arm only -- the MXFP8 arm block-scales dO and dQKVG and folds no amax of theirs
            gate_partials_n=self._gate_bwd.n_partials_cap if fp8 else 0,
            band_partials_n=self._norm_bwd.n_amax_partials() if fp8 else 0,
            # the MXFP8 prologue's arm: the bf16 rebuild regions are carved under the bf16-rebuild arm only (None on the other arms)
            mx_prologue_arm=self.mx_prologue_arm,
        )
        self._compiled_kernel = self._ws  # APIBase's "compiled" marker
        # The declaration's tensors are not needed past here: hold artifacts and facts, never the sample buffers (the
        # convenience wrapper caches this object for the process lifetime -- at 397B, S=32K that would pin ~1.6 GiB of
        # the first call's proj_slab / dy / h / o).
        self._samples = None

    # -- execute ------------------------------------------------------------

    def execute(
        self,
        dy: torch.Tensor,
        saved: SavedForBackward,
        w_qkvg: torch.Tensor,
        w_q_norm: Optional[torch.Tensor],  # None (both) iff geometry.qk_norm is False
        w_k_norm: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        w_o: torch.Tensor,
        dh: Optional[torch.Tensor] = None,
        dw_qkvg: Optional[torch.Tensor] = None,
        dw_o: Optional[torch.Tensor] = None,
        dw_q_norm: Optional[torch.Tensor] = None,  # must be None under qk_norm=False (need_dw_norms resolves False)
        dw_k_norm: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
        *,
        # APPENDED (the quantized backward; keyword-only, defaulted): the caller's 1-element fp32 CUDA scalars, read in-kernel.
        # scale_dp is REQUIRED under quant (cuDNN's fp8 backward contract: the dP scale is the caller's; descale_dp is derived
        # from it on device) and REFUSED without; scale_dy / scale_do / scale_dqkvg are REQUIRED under grad_scaling="delayed"
        # (the previous step's scales) and REFUSED under "current" and without quant -- Rule 1, both directions.
        scale_dp: Optional[torch.Tensor] = None,
        scale_dy: Optional[torch.Tensor] = None,
        scale_do: Optional[torch.Tensor] = None,
        scale_dqkvg: Optional[torch.Tensor] = None,
        # APPENDED (the MXFP8 backward; keyword-only, defaulted, LAST): the caller's TRANSPOSED block-scaled artifacts -- `h_t` the e4m3
        # `[d_model, T]` of `h` re-quantized along TOKENS (K-major, contiguous: the storage itself, never a .t() view of the codes) with
        # its padded F8_128x4 blob `h_t_sf` (`sf_blob_bytes(d_model, T)` bytes), REQUIRED iff need_dw_qkvg (B7's B operand); `w_qkvg_t`
        # the e4m3 `[d_model, N]` of W_qkvg re-quantized along N with `w_qkvg_t_sf` (`sf_blob_bytes(d_model, N)`), REQUIRED iff need_dh
        # (B8's B operand) -- each REFUSED when not read, and on a QuantSpec / bf16 block (Rule 1, both directions).  Their byte counts do
        # not validate the blobs' ORIENTATION (sf_blob_bytes is symmetric in (rows, K)): build them over the transposed matrices.
        h_t: Optional[torch.Tensor] = None,
        h_t_sf: Optional[torch.Tensor] = None,
        w_qkvg_t: Optional[torch.Tensor] = None,
        w_qkvg_t_sf: Optional[torch.Tensor] = None,
        # APPENDED (the fp4 weight modes' backward; keyword-only, defaulted, LAST): the caller's TRANSPOSED e2m1 W_o -- `w_o_t` the packed
        # `torch.float4_e2m1fn_x2` `[H_q*D, d_model // 2]` of W_o re-quantized along d_model in the format of `MxQuantSpec.o_fp4` (K-major,
        # contiguous storage, two codes per byte along d_model) with its padded F8_128x4 blob `w_o_t_sf` (`sf_blob_bytes(H_q*D, d_model,
        # block)` bytes: e4m3 scales per 16 under NVFP4, E8M0 per 32 under MXFP4) -- REQUIRED iff `o_fp4` (the out-projection dgrad runs on
        # every quant arm: the gate backward needs dO_gated), REFUSED otherwise (Rule 1, both directions).  Under an MXFP4 W_qkvg the
        # existing `w_qkvg_t` is the packed e2m1 `[d_model, N // 2]` (the quant spec's dtype) with the UNCHANGED E8M0 / 32 blob `w_qkvg_t_sf`.
        w_o_t: Optional[torch.Tensor] = None,
        w_o_t_sf: Optional[torch.Tensor] = None,
    ) -> None:
        """Launch the enabled stages, in this order, onto the ONE launch stream
        (module docstring: the launch table).  Everything is stream-ordered, so
        nothing overlaps anything else; B1 is issued right after B3 because
        that is when its operand exists, and it depends on nothing below it --
        that independence is what makes it the filler ``fuse_wgrad_overlap``
        schedules: under the knob B1 and B7 are issued on the block's side
        stream (``_WgradSideStream``: fork event after B3 / after B5+B6, join
        event waited by the launch stream at the end of this call, their own
        ``gemm_scratch_side``), and the caller's stream still sees every write
        before this call returns::

            (B2) out_proj_dgrad     dy, w_o                     -> ws.do_gated
            (B3) sigmoid_gate_bwd   ws.do_gated, saved.o,
                                    proj_slab[GATE]             -> ws.do_gated (in place), ws.dqkvg[GATE], ws.o_gated
                                                                   (+ ws.delta = rowsum(dO * O) under fuse_gate_bwd)
            (B1) out_proj_wgrad     dy, ws.o_gated              -> dw_o
                 qk_norm_rope       proj_slab[Q], [K]           -> ws.recompute, ws.recompute_k   (post-norm / post-RoPE)
                 compact_v          proj_slab[V]                -> ws.recompute_v
            (B4) sdpa_bwd           ws.do_gated, ws.recompute*,
                                    saved.o, saved.lse
                                    (+ ws.delta: no dot_do_o)   -> ws.dq, ws.dk, ws.dv   (COMPACT)
            (B5+B6) qk_norm_rope_bwd ws.dq/dk/dv, proj_slab[Q], [K],
                                    saved.rstd_*, w_*_norm      -> ws.dqkvg[Q], [K], [V]; ws.dw_partials_*
                 dw_norm reduce     ws.dw_partials_*            -> dw_q_norm, dw_k_norm       (qk_norm only)
            (B7) qkv_gate_wgrad     ws.dqkvg, saved.h           -> dw_qkvg
            (B8) qkv_gate_dgrad     ws.dqkvg, w_qkvg            -> dh

        A ``need_*`` that was False at build time means the corresponding output
        argument must be ``None`` here -- a provided-but-uncompiled tensor raises
        rather than being silently ignored (Rule 1, both directions). The norm
        weights follow ``geometry.qk_norm`` the same way (both ``None`` iff
        False). ``seq_lens`` is refused on a dense block (declared without
        padding); under ``thd`` it may only be ``saved.seq_lens`` itself (or
        ``None``: the record carries the packed lengths, and they are handed to
        the SDPA backward as both length operands).

        No allocation, no D2H read, no implicit conversion. The record and every
        operand are re-validated (host-only) on each call, and no written buffer
        may overlap any other buffer the backward binds.

        Under ``quant`` the launch order is :meth:`_execute_quant`'s (module
        docstring, "The quantized backward"): the scalar block first, every
        gradient quantized to e4m3 with its scale derived on device (or the
        caller's under ``"delayed"``), the four GEMMs on e4m3 operands with their
        ``alpha`` epilogue read from the scalar block, the fp8 SDPA row over the
        recomputed ``q8 / k8 / v8`` and the gate backward's delta; the gradients
        come out bf16, the scalar block is readable through :meth:`quant_scalars`.
        Under an ``MxQuantSpec`` it is :meth:`_execute_mxfp8`'s (module docstring,
        "The MXFP8 backward"), and the caller's transposed artifacts ``h_t`` /
        ``h_t_sf`` (``need_dw_qkvg``) and ``w_qkvg_t`` / ``w_qkvg_t_sf`` (``need_dh``)
        are REQUIRED -- the two block-scale projection GEMMs read them as their
        K-major B operands; ``scale_dp`` / ``scale_do`` / ``scale_dqkvg`` are refused
        there (no dP scalar; dO and dQKVG are block-scaled) and ``scale_dy`` follows
        ``grad_scaling`` alone.
        """
        if self._ws is None:
            raise RuntimeError("call compile() before execute()")
        g, b, s, act, dev = self.geom, self.batch, self.seq_len, self.act_dtype, self.device
        t, dm, hd, n, d = b * s, g.d_model, g.h_q * g.d_head, g.n_qkvg, g.d_head
        if self.thd:
            # The record carries the packed lengths; an explicit seq_lens may only restate it (identity, never a copy: no device read).
            if seq_lens is not None and seq_lens is not saved.seq_lens:
                raise ValueError("thd=True: execute(seq_lens=) must be saved.seq_lens itself (the record carries the packed lengths), or None")
        elif seq_lens is not None:
            raise ValueError(
                "seq_lens was given but this block was declared without padding (seq_lens_present=False; P0 declines it anyway -- the "
                "sdpa_bwd_sm107 row serves no seq_lens): pass None"
            )
        # Outputs, both directions (Rule 1).
        check = GatedAttentionBlockFwd._check_saved_tensor
        for name, need, ten, shape, dtype, knob in (
            ("dh", self.need_dh, dh, (b, s, dm), act, "need_dh"),
            ("dw_qkvg", self.need_dw_qkvg, dw_qkvg, (n, dm), act, "need_dw_qkvg"),
            ("dw_o", self.need_dw_o, dw_o, (dm, hd), act, "need_dw_o"),
            ("dw_q_norm", self.need_dw_norms, dw_q_norm, (d,), self.dw_norm_dtype, "need_dw_norms"),
            ("dw_k_norm", self.need_dw_norms, dw_k_norm, (d,), self.dw_norm_dtype, "need_dw_norms"),
        ):
            if need and ten is None:
                raise ValueError(f"{name} is required: this block was declared with {knob}=True")
            if not need and ten is not None:
                raise ValueError(
                    f"{name} was given but this block was declared with {knob}=False; a provided-but-uncompiled output is refused rather than silently ignored"
                )
            if need:
                if name.startswith("dw_") and name.endswith("_norm") and ten.dtype != dtype:
                    raise ValueError(f"{name} must be {dtype} (dw_norm_dtype={self.dw_norm_dtype}; P0 serves fp32 only), got {ten.dtype}")
                if name == "dh":
                    _check_token_rows("dh", ten, b, s, dm, act, dev, thd=self.thd)  # packed: [T, d_model] or [1, T, d_model]
                else:
                    check(name, ten, shape, dtype, dev)
        # The quantized backward's scalar inputs, both directions (Rule 1) -- before the workspace and the record, so a wrong
        # recipe is named before anything else; then the MXFP8 backward's transposed artifacts, both directions too.
        self._check_scalar_inputs(scale_dp=scale_dp, scale_dy=scale_dy, scale_do=scale_do, scale_dqkvg=scale_dqkvg)
        self._check_artifacts(h_t=h_t, h_t_sf=h_t_sf, w_qkvg_t=w_qkvg_t, w_qkvg_t_sf=w_qkvg_t_sf, w_o_t=w_o_t, w_o_t_sf=w_o_t_sf)
        self._check_workspace(workspace)
        proj, o_flat = _check_saved_record(
            saved,
            g,
            b,
            s,
            act,
            dev,
            at="execute",
            thd=self.thd,
            num_sequences=self.num_sequences,
            cu_seqlens=self.cu_seqlens,
            h_dtype=self.w_dtype if self.quant is not None else None,
        )
        self._check_inputs(dy, w_qkvg, w_q_norm, w_k_norm, cos, sin, w_o)
        _check_no_overlap(
            [
                (nm, ten)
                for nm, ten in (("dh", dh), ("dw_qkvg", dw_qkvg), ("dw_o", dw_o), ("dw_q_norm", dw_q_norm), ("dw_k_norm", dw_k_norm), ("workspace", workspace))
                if ten is not None
            ],
            [
                ("dy", dy),
                ("saved.h", saved.h),
                ("saved.o", saved.o),
                ("saved.lse", saved.lse),
                ("saved.rstd_q", saved.rstd_q),
                ("saved.rstd_k", saved.rstd_k),
                ("saved.proj_slab", saved.proj_slab),
                ("saved.seq_lens", saved.seq_lens),
                ("w_qkvg", w_qkvg),
                ("w_o", w_o),
                ("w_q_norm", w_q_norm),
                ("w_k_norm", w_k_norm),
                ("cos", cos),
                ("sin", sin),
                ("scale_dp", scale_dp),
                ("scale_dy", scale_dy),
                ("scale_do", scale_do),
                ("scale_dqkvg", scale_dqkvg),
                ("h_t", h_t),
                ("h_t_sf", h_t_sf),
                ("w_qkvg_t", w_qkvg_t),
                ("w_qkvg_t_sf", w_qkvg_t_sf),
                ("w_o_t", w_o_t),
                ("w_o_t_sf", w_o_t_sf),
            ],
        )
        # THE launch stream (Rule 5): the caller's, else torch's current stream on dy's device -- resolved once and handed
        # to EVERY stage (the CuTe-DSL kernels and the GEMM drivers take the raw int; the adapter a CUstream of it).
        stream = int(current_stream) if current_stream is not None else torch.cuda.current_stream(dev).cuda_stream
        # fuse_wgrad_overlap (Rule 5 kept): the launch stream as a torch stream object for the event record / wait pair;
        # the side stream is the block's own (dedicated, never a caller's).  `side is None` = the in-order path, unchanged.
        side = self._side
        launch_ts = as_torch_stream(stream, dev) if side is not None else None
        # The workspace's typed views and their shaped twins: built once per workspace, reused while the caller hands the same
        # buffer (`_workspace_views`; the key is checked on every call).
        v = self._workspace_views(workspace)
        do_gated, dqkvg, o_gated, rq, rk, rv, dq, dk, dv = v.do_gated, v.dqkvg, v.o_gated, v.rq, v.rk, v.rv, v.dq, v.dk, v.dv
        plane_q, plane_k, sdpa_ws, gemm_ws, gemm_ws_side, delta = v.plane_q, v.plane_k, v.sdpa_ws, v.gemm_ws, v.gemm_ws_side, v.delta
        o_q, o_g, o_k, o_v = g.qkvg_offsets  # the four dense bands: a block-sparse geometry (a fifth band) is declined at declaration
        q_pre_b = _cols(proj, o_q, g.h_q, d)
        gate_b = _cols(proj, o_g, g.h_q, d)
        k_pre_b = _cols(proj, o_k, g.h_kv, d)
        v_b = _cols(proj, o_v, g.h_kv, d)
        dy2 = dy.view(t, dm)
        if self.quant is not None:
            # The quantized backward's launch order lives in its own method (the two arms share every check and view above).
            c = SimpleNamespace(
                saved=saved,
                w_qkvg=w_qkvg,
                w_o=w_o,
                w_q_norm=w_q_norm,
                w_k_norm=w_k_norm,
                cos=cos,
                sin=sin,
                dh=dh,
                dw_qkvg=dw_qkvg,
                dw_o=dw_o,
                dw_q_norm=dw_q_norm,
                dw_k_norm=dw_k_norm,
                workspace=workspace,
                stream=stream,
                side=side,
                launch_ts=launch_ts,
                v=v,
                o_flat=o_flat,
                q_pre_b=q_pre_b,
                gate_b=gate_b,
                k_pre_b=k_pre_b,
                v_b=v_b,
                dy2=dy2,
            )
            if isinstance(self.quant, MxQuantSpec):
                self._execute_mxfp8(c, scale_dy=scale_dy, h_t=h_t, h_t_sf=h_t_sf, w_qkvg_t=w_qkvg_t, w_qkvg_t_sf=w_qkvg_t_sf, w_o_t=w_o_t, w_o_t_sf=w_o_t_sf)
            else:
                self._execute_quant(c, scale_dp=scale_dp, scale_dy=scale_dy, scale_do=scale_do, scale_dqkvg=scale_dqkvg)
            return

        # (B2) dO_gated = dY @ W_o
        self._out_proj_dgrad.execute(dy2, w_o, v.do_gated_hd, gemm_ws, stream=stream)
        # (B3) dO (in place), dG -> the GATE band, O_gated (need_dw_o), delta = rowsum(dO * O) (fuse_gate_bwd)
        self._gate_bwd.execute(do_gated, o_flat, gate_b, do_gated, v.dqkvg_g, o_gated, stream=stream, delta=delta)
        # (B1) dW_o = dY^T @ O_gated -- the filler, issued as soon as its operand exists; under fuse_wgrad_overlap on the
        # side stream (fork: B3's o_gated precedes it), joined at the end of this call -- it reads nothing written below
        if self.need_dw_o:
            if side is not None:
                with side.issue(launch_ts, "o") as side_stream:  # fork + the GEMM's enqueue + the join record, one locked section
                    self._out_proj_wgrad.execute(dy2, v.o_gated_hd, dw_o, gemm_ws_side, stream=side_stream)
            else:
                self._out_proj_wgrad.execute(dy2, v.o_gated_hd, dw_o, gemm_ws, stream=stream)
        # Q / K rebuilt post-norm / post-RoPE from the slab bands (the forward's stage (2)+(3) kernel; rstd recomputed by
        # the SAME kernel over the SAME inputs = the forward's), V compacted -- the adapter's operands must be BSHD-physical.
        self._recompute_qk.execute(q_pre_b, k_pre_b, w_q_norm, w_k_norm, cos, sin, q_out=rq, k_out=rk, current_stream=stream)
        self._compact_v.execute(v_b, rv, current_stream=stream)
        # (B4) the Rubin d256 backward chain -> COMPACT dQ / dK / dV (under thd: the packed [1, T, H, D] views, the record's
        # lengths for both sides, saved.lse as the head-major [1, H_q, T] Stats)
        self._sdpa.execute(
            v.rq_bshd,
            v.rk_bshd,
            v.rv_bshd,
            saved.o,
            v.do_gated_bshd,
            saved.lse,
            v.dq_bshd,
            v.dk_bshd,
            v.dv_bshd,
            workspace=sdpa_ws,
            stream=stream,
            delta=delta,
            seq_lens=saved.seq_lens if self.thd else None,
        )
        # (B5+B6) RoPE^T + RMSNorm backward into the Q / K bands, dV into the V band, fp32 dW partials
        norm = g.qk_norm
        self._norm_bwd.execute(
            dq,
            dk,
            dv,
            q_pre_b if norm else None,
            k_pre_b if norm else None,
            saved.rstd_q.view(t, g.h_q) if norm else None,
            saved.rstd_k.view(t, g.h_kv) if norm else None,
            w_q_norm,
            w_k_norm,
            cos.view(t, g.rope_dim),
            sin.view(t, g.rope_dim),
            v.dqkvg_q,
            v.dqkvg_k,
            v.dqkvg_v,
            plane_q,
            plane_k,
            stream=stream,
        )
        # (B7) dW_qkvg = dQKVG^T @ h -- under fuse_wgrad_overlap forked here, right after B5+B6 finished the dqkvg slab, so
        # it overlaps the dW_norm reduce and B8; in order it keeps its place after the reduce
        if self.need_dw_qkvg and side is not None:
            with side.issue(launch_ts, "qkvg") as side_stream:
                self._qkv_gate_wgrad.execute(dqkvg, saved.h.view(t, dm), dw_qkvg, gemm_ws_side, stream=side_stream)
        if self.need_dw_norms:
            self._norm_bwd.reduce(plane_q, plane_k, dw_q_norm, dw_k_norm, stream=stream)
        if self.need_dw_qkvg and side is None:
            self._qkv_gate_wgrad.execute(dqkvg, saved.h.view(t, dm), dw_qkvg, gemm_ws, stream=stream)
        # (B8) dh = dQKVG @ W_qkvg
        if self.need_dh:
            self._qkv_gate_dgrad.execute(dqkvg, w_qkvg, dh.view(t, dm), gemm_ws, stream=stream)
        # fuse_wgrad_overlap: JOIN -- the launch stream waits for both side GEMMs before this call returns (Rule 5: the
        # caller's stream semantics are exactly the in-order block's; the convenience wrapper's workspace, freed at
        # return, is reused only behind this point)
        if side is not None:
            if self.need_dw_o:
                side.join(launch_ts, "o")
            if self.need_dw_qkvg:
                side.join(launch_ts, "qkvg")

    def _check_scalar_inputs(self, **scalars) -> None:
        """The appended ``execute`` scalars (``scale_dp`` / ``scale_dy`` / ``scale_do`` / ``scale_dqkvg``), BOTH directions
        (Rule 1): required exactly when the declaration reads them -- ``scale_dp`` under ``quant``, the three gradient scales
        under ``quant`` with ``grad_scaling="delayed"`` -- and refused otherwise (a provided-but-unread scalar is never
        silently ignored); each a 1-element fp32 CUDA tensor on the block's device at a 4-byte-aligned address (read in-kernel,
        never on the host)."""
        q, dev = self.quant, self.device
        mx = isinstance(q, MxQuantSpec)
        delayed = q is not None and self.grad_scaling == "delayed"
        if q is None:
            why_not = "this block was declared without quant (the bf16 / fp16 backward quantizes no gradient and takes no fp8 scale)"
        else:
            why_not = "this block was declared with grad_scaling='current' (the gradient scales are derived on device from this step's amax passes; read them back through quant_scalars())"
        # The MXFP8 arm's refusals, by name: dY is its ONE per-tensor gradient (scale_dy follows grad_scaling); dO and dQKVG are
        # block-scaled and its SDPA row has no dP scalar.
        mx_why_not = dict(
            scale_dp="this block was declared with quant=MxQuantSpec: the MXFP8 SDPA backward has no dP scalar (its dS is block-scaled, one E8M0 scale per 32 elements) and takes no scale_dp",
            scale_do="this block was declared with quant=MxQuantSpec: dO is block-scaled (the 32-element blocks carry their own E8M0 scales), so there is no per-tensor scale_do",
            scale_dqkvg="this block was declared with quant=MxQuantSpec: dQKVG is block-scaled (the 32-element blocks carry their own E8M0 scales), so there is no per-tensor scale_dqkvg",
        )
        for name, ten in scalars.items():
            if mx:
                want = delayed if name == "scale_dy" else False
            else:
                want = (q is not None) if name == "scale_dp" else delayed
            if want and ten is None:
                what = (
                    "the fp8 SDPA row's dP scale (its descale is derived from it on device)"
                    if name == "scale_dp"
                    else f"the previous step's scale of {name[6:]} under grad_scaling='delayed'"
                )
                raise ValueError(f"{name} is required: this block was declared with quant={type(q).__name__} -- {what}; a 1-element fp32 CUDA tensor on {dev}")
            if not want and ten is not None:
                why = mx_why_not.get(name, why_not) if mx else why_not
                raise ValueError(f"{name} was given but {why}; a provided-but-unread scalar is refused rather than silently ignored")
            if want:
                if not isinstance(ten, torch.Tensor) or ten.dtype != torch.float32 or ten.numel() != 1 or not ten.is_cuda:
                    got = f"{ten.dtype} x {ten.numel()} on {ten.device}" if isinstance(ten, torch.Tensor) else type(ten).__name__
                    raise ValueError(f"{name} must be a 1-element fp32 CUDA tensor (read in-kernel; no host readback), got {got}")
                if ten.device != dev:
                    raise ValueError(f"{name} must live on dy's device {dev}, got {ten.device}")
                if ten.data_ptr() % 4:
                    raise ValueError(f"{name} must sit at a 4-byte-aligned address, got {ten.data_ptr():#x}")

    def _execute_quant(self, c: SimpleNamespace, *, scale_dp, scale_dy, scale_do, scale_dqkvg) -> None:
        """The quantized (per-tensor fp8) backward's launches, in this order, on the ONE launch stream -- every check and
        every shared view was made by :meth:`execute` (``c`` carries them; module docstring, "The quantized backward")::

             1  PROLOGUE (one launch, four jobs by block range -- _QuantPrologue):
                  init            slots[:] = 0; descale_dp = 1 / scale_dp; the plan-time constants (QUANT_CONST_SLOTS) from the
                                  launch's kernel arguments
                  amax dY         partials[c] = max |dY| per CTA                                       (dY viewed [T, d_model / D, D])
                  Q / K rebuild   norm + RoPE from the slab's PRE-norm bands -> q8 / k8 = e4m3(bf16(.) * scale_q / scale_k)
                                                                                                        (the scales: kernel arguments)
                  v8              e4m3(V band * scale_v)                                               (V's compaction; idem)
             2  quantize dY       amax_dy = max(partials) PUBLISHED; dy8 = e4m3(dY * scale_dy); publishes scale_dy, descale_dy,
                                  alpha_b1 = descale_dy / scale_o, alpha_b2 = descale_dy * descale_w_o
             3  (B2) out_proj dgrad dO_gated (bf16) = dy8 @ W_o8 * alpha_b2
             4  (B3) gate backward  dO (bf16, in place), dG (GATE band), og8 (need_dw_o), delta; per-CTA partials of max |dO| and max |dG|
             5  quantize dO         amax_do = max(B3's dO partials) PUBLISHED; do8 = e4m3(dO * scale_do); publishes scale_do, descale_do
             6  (B1) out_proj wgrad dW_o = dy8^T @ og8 * alpha_b1                                      (need_dw_o; the side stream
                                                                                                        under fuse_wgrad_overlap:
                                                                                                        og8 AND alpha_b1 precede the fork)
             7  (B4) fp8 SDPA bwd   q8, k8, v8, do8, lse, delta, the twelve scalars (11 slots + scale_dp) -> bf16 dq / dk / dv, amax_dp
             8  (B5+B6) norm / RoPE bf16 dqkvg bands, dW partials, per-CTA partials of max |dQ_pre / dK_pre / dV|
             9  EPILOGUE (one launch, two jobs by block range -- _QuantEpilogue):
                  dW_norm reduce  the fixed-order sum of the partial planes                             (qk_norm)
                  quantize dQKVG  amax_dqkvg = max(B3's dG partials, B5+B6's band partials) PUBLISHED; dqkvg8 = e4m3(dqkvg * scale_dqkvg);
                                  publishes scale_dqkvg, descale_dqkvg, alpha_b7 = descale * descale_h, alpha_b8 = descale * descale_w_qkvg
            10  (B7) qkv_gate wgrad dW_qkvg = dqkvg8^T @ h8 * alpha_b7                 (need_dw_qkvg; forked AFTER 9 under the knob)
            11  (B8) qkv_gate dgrad dh = dqkvg8 @ W_qkvg8 * alpha_b8                                     (need_dh)

        Under ``"delayed"`` the quantize launches read the caller's ``scale_dy / scale_do / scale_dqkvg`` instead of deriving
        them, and every amax is still folded / published.  The adapter's dead ``o`` is bound to ``og8`` when it exists, else to
        ``do8`` (an e4m3 operand of the same shape that the plan already binds; nothing is read through it under the external delta).
        Every scalar a launch below reads is a SLOT of the scalar block written by launch 1's init job on this stream -- the
        plan-time constants included -- or one of the caller's ``execute`` scalars; nothing comes from ``compile()``.  Launch 1's
        own rebuild / v8 jobs take the static scales as kernel ARGUMENTS (the values of their slots): a block of a launch must
        not read a slot another block of the same launch writes.
        """
        g, b, s = self.geom, self.batch, self.seq_len
        t, dm, d = b * s, g.d_model, g.d_head
        stream, side, launch_ts, v, vals = c.stream, c.side, c.launch_ts, c.v, self._quant_vals
        dy8, do8, og8, dqkvg8, sc = v.dy8, v.do8, v.og8, v.dqkvg8, v.sc
        partials, partials_do, partials_dg, partials_bands = v.partials, v.partials_do, v.partials_dg, v.partials_bands
        h8 = c.saved.h.view(t, dm)
        dy_rows = c.dy2.view(t, dm // d, d)
        cos_t, sin_t = c.cos.view(t, g.rope_dim), c.sin.view(t, g.rope_dim)

        # 1. the prologue: the scalar block (zero; the row's amax_dp atomics start from it, the gradient amax slots are published
        #    later from partials; descale_dp on device; the plan-time constants stored from the launch's arguments -- on THIS stream,
        #    so every consumer below (the alpha factors, scale_o, the row's constant scalars) is ordered behind their writer by
        #    construction), the dY amax partials, the Q / K rebuild straight into q8 / k8 and v8 at the forward's static scales --
        #    handed to this launch as kernel arguments (the values of their slots, which its own init job is writing)
        n_partials = self._prologue.execute(
            slots=v.slots,
            scale_dp=scale_dp,
            descale_dp_out=sc["descale_dp"],
            dy=dy_rows,
            partials=partials,
            q_pre=c.q_pre_b,
            k_pre=c.k_pre_b,
            w_q=c.w_q_norm,
            w_k=c.w_k_norm,
            cos=cos_t,
            sin=sin_t,
            q8=v.q8,
            k8=v.k8,
            scale_q=vals["scale_q"],
            scale_k=vals["scale_k"],
            v=c.v_b,
            v8=v.v8,
            scale_v=vals["scale_v"],
            consts=tuple(vals[n] for n in QUANT_CONST_SLOTS),
            stream=stream,
        )
        # 2. dY -> dy8 (the amax reduced from the partials and published; + alpha_b1 = descale_dy * (1 / scale_o), alpha_b2 = descale_dy * descale_w_o)
        self._quant_dy.execute(
            dy_rows,
            v.dy8_rows,
            stream=stream,
            amax_slot=sc["amax_dy"],
            scale_in=scale_dy,
            scale_out=sc["scale_dy"],
            descale_out=sc["descale_dy"],
            alpha_consts=(sc["descale_o"], sc["descale_w_o"]),
            alpha_outs=(sc["alpha_b1"], sc["alpha_b2"]),
            partials=partials,
            n_partials=n_partials,
        )
        # 3. (B2) dO_gated = dy8 @ W_o8 * alpha_b2
        self._out_proj_dgrad.execute(dy8, c.w_o, v.do_gated_hd, v.gemm_ws, stream=stream, alpha=v.alpha_b2)
        # 4. (B3) dO in place, dG -> the GATE band, og8 (need_dw_o), delta = rowsum(dO * O) ALWAYS, the per-CTA partials of max |dO|
        #    (the dO quantize's amax) and of max |dG| (the GATE band's half of amax_dqkvg)
        n_gate = self._gate_bwd.execute(
            v.do_gated,
            c.o_flat,
            c.gate_b,
            v.do_gated,
            v.dqkvg_g,
            og8,
            stream=stream,
            delta=v.delta,
            scale_o=sc["scale_o"] if self._gate_bwd.og_fp8 else None,  # the og8 arm's scale only (no og8 without need_dw_o)
            amax_do=partials_do,
            amax_dg=partials_dg,
        )
        # 5. dO -> do8 (amax_do reduced from B3's partials and published; no pass of its own)
        self._quant_do.execute(
            v.do_gated,
            do8,
            stream=stream,
            amax_slot=sc["amax_do"],
            scale_in=scale_do,
            scale_out=sc["scale_do"],
            descale_out=sc["descale_do"],
            partials=partials_do,
            n_partials=n_gate,
        )
        # 6. (B1) dW_o = dy8^T @ og8 * alpha_b1 -- after 5, so og8 AND alpha_b1 are written on the launch stream before the fork
        if self.need_dw_o:
            if side is not None:
                with side.issue(launch_ts, "o") as side_stream:
                    self._out_proj_wgrad.execute(dy8, v.og8_hd, c.dw_o, v.gemm_ws_side, stream=side_stream, alpha=v.alpha_b1)
            else:
                self._out_proj_wgrad.execute(dy8, v.og8_hd, c.dw_o, v.gemm_ws, stream=stream, alpha=v.alpha_b1)
        # 7. (B4) the fp8 row: the twelve scalars (plan-time constants + slots, cached with the views; the caller's scale_dp per call),
        #    the delta, amax_dp
        self._sdpa.execute(
            v.q8_bshd,
            v.k8_bshd,
            v.v8_bshd,
            v.o_dead8_bshd,
            v.do8_bshd,
            c.saved.lse,
            v.dq_bshd,
            v.dk_bshd,
            v.dv_bshd,
            workspace=v.sdpa_ws,
            stream=stream,
            delta=v.delta,
            scalars=dict(v.row_scalars, scale_dP=scale_dp),
            amax_dp=sc["amax_dp"],
            seq_lens=c.saved.seq_lens if self.thd else None,
        )
        # 8. (B5+B6) RoPE^T + RMSNorm backward into the Q / K bands, dV into the V band, fp32 dW partials -- bf16, unchanged --
        #    plus the per-CTA partials of the bands' max |.| (B3 wrote the GATE band's)
        norm = g.qk_norm
        n_bands = self._norm_bwd.execute(
            v.dq,
            v.dk,
            v.dv,
            c.q_pre_b if norm else None,
            c.k_pre_b if norm else None,
            c.saved.rstd_q.view(t, g.h_q) if norm else None,
            c.saved.rstd_k.view(t, g.h_kv) if norm else None,
            c.w_q_norm,
            c.w_k_norm,
            cos_t,
            sin_t,
            v.dqkvg_q,
            v.dqkvg_k,
            v.dqkvg_v,
            v.plane_q,
            v.plane_k,
            stream=stream,
            amax_out=partials_bands,
        )
        # 9. the epilogue: the fixed-order dW_norm reduce (qk_norm) + dQKVG -> dqkvg8 with amax_dqkvg reduced from B3's dG partials
        #    and B5+B6's band partials and published (+ alpha_b7 = descale_dqkvg * descale_h, alpha_b8 = descale_dqkvg * descale_w_qkvg)
        self._epilogue.execute(
            plane_q=v.plane_q,
            plane_k=v.plane_k,
            dw_q_norm=c.dw_q_norm,
            dw_k_norm=c.dw_k_norm,
            src=v.dqkvg_rows,
            dst=v.dqkvg8_rows,
            scale_in=scale_dqkvg,
            partials_gate=partials_dg,
            n_gate=n_gate,
            partials_bands=partials_bands,
            n_bands=n_bands,
            amax_out=sc["amax_dqkvg"],
            scale_out=sc["scale_dqkvg"],
            descale_out=sc["descale_dqkvg"],
            alpha_consts=(sc["descale_h"], sc["descale_w_qkvg"]),
            alpha_outs=(sc["alpha_b7"], sc["alpha_b8"]),
            stream=stream,
        )
        # 10. (B7) dW_qkvg = dqkvg8^T @ h8 * alpha_b7 -- forked HERE under fuse_wgrad_overlap: dqkvg8 and alpha_b7 are written
        if self.need_dw_qkvg:
            if side is not None:
                with side.issue(launch_ts, "qkvg") as side_stream:
                    self._qkv_gate_wgrad.execute(dqkvg8, h8, c.dw_qkvg, v.gemm_ws_side, stream=side_stream, alpha=v.alpha_b7)
            else:
                self._qkv_gate_wgrad.execute(dqkvg8, h8, c.dw_qkvg, v.gemm_ws, stream=stream, alpha=v.alpha_b7)
        # 11. (B8) dh = dqkvg8 @ W_qkvg8 * alpha_b8
        if self.need_dh:
            self._qkv_gate_dgrad.execute(dqkvg8, c.w_qkvg, c.dh.view(t, dm), v.gemm_ws, stream=stream, alpha=v.alpha_b8)
        # fuse_wgrad_overlap: JOIN before this call returns (Rule 5) -- and before the NEXT execute's prologue zeroes the block
        if side is not None:
            if self.need_dw_o:
                side.join(launch_ts, "o")
            if self.need_dw_qkvg:
                side.join(launch_ts, "qkvg")

    def _check_artifacts(self, *, h_t, h_t_sf, w_qkvg_t, w_qkvg_t_sf, w_o_t=None, w_o_t_sf=None) -> None:
        """The MXFP8 backward's appended ``execute`` artifacts, BOTH directions (Rule 1): ``h_t`` / ``h_t_sf`` REQUIRED iff the block
        reads them (``quant=MxQuantSpec`` with ``need_dw_qkvg``: B7's K-major B operand and its scale factors), ``w_qkvg_t`` /
        ``w_qkvg_t_sf`` iff ``need_dh`` (B8's), ``w_o_t`` / ``w_o_t_sf`` (appended) iff ``MxQuantSpec.o_fp4`` (B2's: the out-projection
        dgrad runs on every quant arm), each REFUSED otherwise -- on a QuantSpec / bf16 block, or with the need off (a
        provided-but-unread artifact is never silently ignored).  Then the layout contract of each, host-only and before any launch:
        the codes are the ``[rows, K]`` STORAGE ITSELF in the dtype the quant spec names for that weight -- e4m3 ``[d_model, T]`` / ``[d_model,
        N]`` / (``h_t``, an e4m3 ``w_qkvg_t``), or PACKED e2m1 ``torch.float4_e2m1fn_x2`` ``[d_model, N // 2]`` (``w_qkvg_t`` under an MXFP4
        ``W_qkvg``) / ``[H_q*D, d_model // 2]`` (``w_o_t``: two codes per byte along K; a LOGICAL ``[rows, K]`` fp4 tensor holds twice the
        data and is refused, uint8 bytes get the ``.view(torch.float4_e2m1fn_x2)`` hint, an e4m3 ``w_qkvg_t`` under an e2m1 spec and the
        reverse are refused by the field that selects them) -- contiguous (strides ``(K_storage, 1)``: the block-scale rows read K-major
        operands whose scale factors run along K, so a ``.t()`` VIEW of the un-transposed codes is refused by name), 16-B aligned, on the
        block's device; the blobs ``_check_sf_blob``'s (dtype, the PADDED ``sf_blob_bytes(rows, K, block)`` count at the format's block --
        E8M0 per 32 for ``h_t_sf`` / ``w_qkvg_t_sf`` whatever ``w_qkvg_t``'s dtype, the ``o_fp4`` format's e4m3 per 16 or E8M0 per 32 for
        ``w_o_t_sf``, so the OTHER format's blob is the byte-count decline --, contiguity, 16-B alignment) on the same device.  The byte
        count does NOT validate a blob's orientation (``sf_blob_bytes(rows, k) == sf_blob_bytes(k, rows)``): a blob built over the
        un-transposed matrix passes here and produces a wrong gradient -- the MXFP8 and the fp4 suites pin that failure numerically on
        every accept cell."""
        mx = isinstance(self.quant, MxQuantSpec)
        g, dev, t = self.geom, self.device, self.batch * self.seq_len
        if not mx:
            why = (
                "this block was declared without quant (the bf16 / fp16 backward reads saved.h and w_qkvg as they are)"
                if self.quant is None
                else "this block was declared with quant=QuantSpec (the per-tensor fp8 backward reads saved.h and w_qkvg as e4m3 operands with their descales in the GEMM epilogue)"
            )
            for name, ten in (("h_t", h_t), ("h_t_sf", h_t_sf), ("w_qkvg_t", w_qkvg_t), ("w_qkvg_t_sf", w_qkvg_t_sf), ("w_o_t", w_o_t), ("w_o_t_sf", w_o_t_sf)):
                if ten is not None:
                    raise ValueError(
                        f"{name} was given but {why}; the transposed block-scaled artifacts belong to the MXFP8 backward (quant=MxQuantSpec) -- a "
                        "provided-but-unread artifact is refused rather than silently ignored"
                    )
            return
        hd, e4 = g.h_q * g.d_head, self.quant.dtype
        # (name, sf_name, codes, blob, need, the knob that reads it, rows, K, the codes' dtype, the blob's (block, dtypes), what it is)
        pairs = (
            (
                "h_t",
                "h_t_sf",
                h_t,
                h_t_sf,
                self.need_dw_qkvg,
                "need_dw_qkvg=True",
                g.d_model,
                t,
                e4,
                (MXFP8_BLOCK_SIZE, None),
                "h re-quantized along TOKENS (the projection weight gradient's B operand)",
            ),
            (
                "w_qkvg_t",
                "w_qkvg_t_sf",
                w_qkvg_t,
                w_qkvg_t_sf,
                self.need_dh,
                "need_dh=True",
                g.d_model,
                g.n_qkvg,
                self.w_qkvg_dtype,
                (MXFP8_BLOCK_SIZE, None),
                "W_qkvg re-quantized along its row axis N (the dh dgrad's B operand)",
            ),
            (
                "w_o_t",
                "w_o_t_sf",
                w_o_t,
                w_o_t_sf,
                self.o_fp4 is not None,
                f"o_fp4={self.o_fp4}",
                hd,
                g.d_model,
                self.w_o_dtype,
                (self.o_fp4.block_size, (torch.uint8, self.o_fp4.sf_torch_dtype)) if self.o_fp4 is not None else (MXFP8_BLOCK_SIZE, None),
                "W_o re-quantized along d_model in the format of o_fp4 (the out-projection dgrad's B operand under an fp4 W_o)",
            ),
        )
        for name, sf_name, codes, sf, need, knob, rows, k, dtype, (block, sf_dtypes), what in pairs:
            fp4 = _FP4_X2 is not None and dtype == _FP4_X2
            k_store = _fp4_storage_shape((rows, k), dtype)[-1]
            sf_word = "F8_128x4 " + ("e4m3-per-16" if sf_dtypes is not None and self.o_fp4 is Fp4Format.NVFP4 and name == "w_o_t" else f"E8M0-per-{block}")
            if need:
                for nm, ten in ((name, codes), (sf_name, sf)):
                    if ten is None:
                        codes_word = f"packed torch.float4_e2m1fn_x2 [{rows}, {k} // 2 = {k_store}]" if fp4 else f"{dtype} [{rows}, {k}]"
                        raise ValueError(
                            f"{nm} is required: this block was declared with quant=MxQuantSpec and {knob} -- {what}; pass the {codes_word} codes as {name} "
                            f"and their padded {sf_word} blob as {sf_name} (kernels.proj_gemm.sf_blob_bytes({rows}, {k}, {block}) bytes)"
                        )
            else:
                for nm, ten in ((name, codes), (sf_name, sf)):
                    if ten is not None:
                        why = (
                            "this block was declared without MxQuantSpec.o_fp4 (the out projection stays per-tensor e4m3 and reads w_o as it is)"
                            if name == "w_o_t"
                            else f"this block was declared with {knob.replace('=True', '=False')}"
                        )
                        raise ValueError(
                            f"{nm} was given but {why}, so nothing reads it ({what}); a provided-but-unread artifact is refused rather than silently ignored"
                        )
                continue
            if not isinstance(codes, torch.Tensor):
                raise ValueError(f"{name} must be the {dtype} codes of {what}, got {type(codes).__name__}")
            if codes.dtype != dtype:
                field = "MxQuantSpec.w_qkvg_dtype" if name == "w_qkvg_t" else "MxQuantSpec.o_fp4"
                if fp4:
                    hint = (
                        " -- torch can VIEW but not cast to fp4: hand over the packed codes as storage.view(torch.float4_e2m1fn_x2), never uint8"
                        if codes.dtype == torch.uint8
                        else ""
                    )
                    raise ValueError(
                        f"{name} must be packed torch.float4_e2m1fn_x2 codes ({field} names an e2m1 weight, so its transposed artifact is e2m1 too: "
                        f"[{rows}, {k} // 2 = {k_store}], two codes per byte along K), got {codes.dtype}{hint}"
                    )
                if _FP4_X2 is not None and codes.dtype == _FP4_X2:
                    raise ValueError(
                        f"{name} is torch.float4_e2m1fn_x2 but this block's spec names {dtype} codes for it ({field}); the artifact carries the dtype "
                        "of the weight it transposes -- declare the fp4 weight mode on the quant spec as the forward did, or hand over the e4m3 artifact"
                    )
                raise ValueError(f"{name} must be the {dtype} codes of {what}, got {codes.dtype}")
            shape, stride = tuple(int(x) for x in codes.shape), tuple(int(x) for x in codes.stride())
            if shape != (rows, k_store):
                packed = (
                    f" -- the PACKED e2m1 storage: two codes per byte along K = {k}, so a LOGICAL [{rows}, {k}] fp4 tensor holds twice the data the GEMM declares"
                    if fp4
                    else ""
                )
                rows_word = f"h_q*d_head={rows}" if name == "w_o_t" else f"d_model={rows}"
                raise ValueError(f"{name} must be [{rows_word}, {k_store}] (the TRANSPOSED matrix, K = {k} contiguous{packed}), got shape {shape}")
            if stride != (k_store, 1):
                raise ValueError(
                    f"{name} has strides {stride}, but the block-scale GEMM binds the contiguous K-major storage [{rows}, {k_store}] (strides {(k_store, 1)}): "
                    "hand it the transposed matrix as the caller STORES it (re-quantized along its K axis), never a .t() view of the un-transposed "
                    "codes -- the scale factors of a view would run along the wrong axis"
                )
            if codes.device != dev:
                raise ValueError(f"{name} must live on dy's device {dev}, got {codes.device}")
            if codes.data_ptr() % 16:
                raise ValueError(f"{name} must be 16-byte aligned (TMA-fed), got data_ptr={codes.data_ptr():#x}")
            if not isinstance(sf, torch.Tensor):
                raise ValueError(f"{sf_name} must be the uint8 {sf_word} blob of {name} (sf_blob_bytes({rows}, {k}, {block}) bytes), got {type(sf).__name__}")
            if sf_dtypes is None:
                _check_sf_blob(sf, sf_name, rows, k)
            else:
                _check_sf_blob(sf, sf_name, rows, k, block=block, sf_dtypes=sf_dtypes)
            if sf.device != dev:
                raise ValueError(f"{sf_name} must live on dy's device {dev}, got {sf.device}")

    def _execute_mxfp8(self, c: SimpleNamespace, *, scale_dy, h_t, h_t_sf, w_qkvg_t, w_qkvg_t_sf, w_o_t=None, w_o_t_sf=None) -> None:
        """The MXFP8 backward's launches, in this order, on the ONE launch stream -- every check and every shared view was made by
        :meth:`execute` (``c`` carries them; module docstring, "The MXFP8 backward")::

             1  PROLOGUE (one launch, four jobs by block range -- _MxQuantPrologue):
                  init            slots[:] = 0; the plan-time constants (QUANT_CONST_SLOTS: scale_o, descale_o, descale_w_o live, 0.0
                                  elsewhere) from the launch's kernel arguments -- no scale_dp, no descale_dp (the MXFP8 row has no dP scalar)
                  amax dY         partials[c] = max |dY| per CTA                                       (dY viewed [T, d_model / D, D])
                  Q / K rebuild   norm + RoPE from the slab's PRE-norm bands on a one-head x 32-token tile -> bf16-rounded, then
                                  q8 + sf_q, q_T8 + sf_q_T, k8 + sf_k, k_T8 + sf_k_T (rowwise AND columnwise; no bf16 recompute buffer)
                  v8              v8 + sf_v straight from the slab's V band (V's compaction, rowwise: the row's dP operand)
             2  quantize dY       amax_dy = max(partials) PUBLISHED; dy8 = e4m3(dY * scale_dy); publishes scale_dy, descale_dy,
                                  alpha_b1 = descale_dy / scale_o, alpha_b2 = descale_dy * descale_w_o      (the one per-tensor gradient)
            2b  quantize dY (block) o_fp4 only -- MXFP4 W_o: dy_mx8 + sf_dy_mx (MX rowwise over [T, d_model / D, D], canonical blob);
                                  NVFP4 W_o: dy4 + sf_dy4 = the two-level NVFP4 cast of scale_dy x dY (scale_dy's slot read in-kernel)
             3  (B2) out_proj dgrad dO_gated (bf16) = dy8 @ W_o8 * alpha_b2; under an fp4 W_o the block-scale dgrad over the caller's packed
                                  e2m1 w_o_t and its blob (dy_mx8 . w_o_t^T on the mixed row; dy4 . w_o_t^T on the NVFP4 row = scale_dy x dO_gated)
             4  (B3) gate backward  dO (bf16, in place), dG (GATE band), og8 (need_dw_o), delta ALWAYS; NO amax partials (dO is block-scaled);
                                  NVFP4 W_o: dO_gated x descale_dy (its slot) before every use -- the dY descale arm
             5  quantize dO (dual) do8 + sf_do (32-element blocks along D, the row's dP operand, the SDPA's rowwise layout) AND do_T8 + sf_do_T
                                  (32-token blocks along S, the row's dV operand, D-plane-major) from ONE read of the bf16 dO
             6  (B1) out_proj wgrad dW_o = dy8^T @ og8 * alpha_b1          (need_dw_o; the side stream under fuse_wgrad_overlap, forked
                                                                            AFTER 5: og8 and alpha_b1 are written on the launch stream first)
             7  (B4) MXFP8 SDPA bwd q8 / q_T8, k8 / k_T8, v8, do8 / do_T8 with their seven scale-factor blobs, lse, delta, the dead
                                  o16 = saved.o / do16 = dO ports -> bf16 dq / dk / dv (the row's block-scaled dS chain; dQ once per
                                  head chunk on the block-scale arm too)
             8  (B5+B6) norm / RoPE bf16 dqkvg bands, dW partials                                      (no amax fold)
             9  EPILOGUE (one launch, two jobs by block range at 256 threads -- _MxQuantEpilogue; iff one of its jobs exists):
                  dW_norm reduce  the fixed-order sum of the partial planes                          (need_dw_norms)
                  dqkvg cast      dqkvg8 [T, N] + sf_dqkvg (GEMM-canonical F8_128x4 over (rows = T, K = N); need_dh) AND
                                  dqkvg_t8 [N, T] + sf_dqkvg_t (canonical over (rows = N, K = T); whole 32-token blocks; need_dw_qkvg)
                                  from ONE read of the dqkvg slab; a half not requested is folded out of the artifact
            10  (B7) qkv_gate wgrad dW_qkvg = dqkvg_t8 . h_t^T   (sf_dqkvg_t, the caller's h_t_sf)    (need_dw_qkvg; forked AFTER 9 under
                                                                                                        the knob: dqkvg_t8 and its blob are written)
            11  (B8) qkv_gate dgrad dh = dqkvg8 . w_qkvg_t^T     (sf_dqkvg, the caller's w_qkvg_t_sf)  (need_dh)

        Under ``"delayed"`` launch 2 reads the caller's ``scale_dy`` instead of deriving it and still publishes the amax.  Every scalar a
        launch reads is a SLOT of the scalar block written by launch 1's init job on this stream (the three live constants included) or
        published by launch 2; nothing comes from ``compile()``.  The E8M0 dequant of the two block-scale GEMMs is exact and happens in the
        MMA: no alpha, no descale slot.  No atomic anywhere: the one amax (dY) is a max over per-CTA partials, and the MXFP8 row's chain has
        none -- two executes are bitwise equal under every knob set, and every byte the unfused chain wrote.
        """
        g, b, s = self.geom, self.batch, self.seq_len
        t, dm, d = b * s, g.d_model, g.d_head
        stream, side, launch_ts, v, vals = c.stream, c.side, c.launch_ts, c.v, self._quant_vals
        sc, partials = v.sc, v.partials
        dy_rows = c.dy2.view(t, dm // d, d)
        cos_t, sin_t = c.cos.view(t, g.rope_dim), c.sin.view(t, g.rope_dim)
        sf, sfl = v.sf, v.sf_flat

        # 1. the PROLOGUE: the scalar block (zeroed, then the plan-time constants from the launch's arguments -- on THIS stream, so every
        #    consumer below -- scale_o, the alpha factors, the readers of quant_scalars() -- is ordered behind its writer by construction),
        #    the dY amax partials, the Q / K rebuild straight into the four block-scaled payloads and their blobs, v8 from the slab's V band
        n_partials = self._prologue.execute(
            slots=v.slots,
            dy=dy_rows,
            partials=partials,
            q_pre=c.q_pre_b,
            k_pre=c.k_pre_b,
            w_q=c.w_q_norm,
            w_k=c.w_k_norm,
            cos=cos_t,
            sin=sin_t,
            q8=v.q8,
            sf_q=sfl["sf_q"],
            q_T8=v.q_T8,
            sf_q_T=sfl["sf_q_T"],
            k8=v.k8,
            sf_k=sfl["sf_k"],
            k_T8=v.k_T8,
            sf_k_T=sfl["sf_k_T"],
            v=c.v_b,
            v8=v.v8,
            sf_v=sfl["sf_v"],
            consts=tuple(vals[n] for n in QUANT_CONST_SLOTS),
            stream=stream,
        )
        # 2. dY -> dy8 with the amax reduced from the prologue's partials and published (+ alpha_b1, alpha_b2)
        self._quant_dy.execute(
            dy_rows,
            v.dy8_rows,
            stream=stream,
            amax_slot=sc["amax_dy"],
            scale_in=scale_dy,
            scale_out=sc["scale_dy"],
            descale_out=sc["descale_dy"],
            alpha_consts=(sc["descale_o"], sc["descale_w_o"]),
            alpha_outs=(sc["alpha_b1"], sc["alpha_b2"]),
            partials=partials,
            n_partials=n_partials,
        )
        # 2b. (fp4 W_o only) the BLOCK quantization of dY the out-projection dgrad reads -- after launch 2 published scale_dy on this
        #     stream: MXFP4 -> the MX-rowwise e4m3 dy_mx8 with its canonical E8M0 blob; NVFP4 -> the two-level NVFP4 cast dy4 of scale_dy x dY
        #     (the slot read in-kernel) with its canonical e4m3 blob
        if self.o_fp4 is Fp4Format.MXFP4:
            self._quant_dy_block.execute(dy_rows, v.dy_mx8_rows, v.sf_dy_mx, batch=b, seq_len=s, current_stream=stream)
        elif self.o_fp4 is Fp4Format.NVFP4:
            self._quant_dy_block.execute(dy_rows, v.dy4, v.sf_dy4, current_stream=stream, scale_in=sc["scale_dy"])
        # 3. (B2) dO_gated = dy8 @ W_o8 * alpha_b2 (per-tensor e4m3), or -- under an fp4 W_o -- the block-scale dgrad over the caller's packed
        #    e2m1 W_o^T with its blob: dy_mx8 . w_o_t^T (the mixed row) or dy4 . w_o_t^T (the NVFP4 row: scale_dy x the true dO_gated, undone
        #    by B3's descale arm); no alpha on either block-scale row
        if self.o_fp4 is None:
            self._out_proj_dgrad.execute(v.dy8, c.w_o, v.do_gated_hd, v.gemm_ws, stream=stream, alpha=v.alpha_b2)
        elif self.o_fp4 is Fp4Format.MXFP4:
            self._out_proj_dgrad.execute(v.dy_mx8, w_o_t, v.do_gated_hd, v.gemm_ws, stream=stream, sf_a=v.sf_dy_mx, sf_b=w_o_t_sf)
        else:
            self._out_proj_dgrad.execute(v.dy4, w_o_t, v.do_gated_hd, v.gemm_ws, stream=stream, sf_a=v.sf_dy4, sf_b=w_o_t_sf)
        # 4. (B3) dO in place, dG -> the GATE band, og8 (need_dw_o), delta = rowsum(dO * O) ALWAYS; no amax fold on this arm; under an
        #    NVFP4 W_o the dY descale arm multiplies B2's scaled dO_gated by descale_dy (its slot, written by launch 2) before every use
        self._gate_bwd.execute(
            v.do_gated,
            c.o_flat,
            c.gate_b,
            v.do_gated,
            v.dqkvg_g,
            v.og8,
            stream=stream,
            delta=v.delta,
            scale_o=sc["scale_o"] if self._gate_bwd.og_fp8 else None,  # the og8 arm's scale only (no og8 without need_dw_o)
            descale_dy=sc["descale_dy"] if self._gate_bwd.want_dy_descale else None,
        )
        # 5. dO block-quantized rowwise (the dP operand) AND columnwise (the dV operand) from ONE read of the same bf16 buffer
        self._quant_do.execute(v.do_gated, v.do8, sfl["sf_do"], batch=b, seq_len=s, current_stream=stream, dst_T=v.do_T8, sf_T=sfl["sf_do_T"])
        # 6. (B1) dW_o = dy8^T @ og8 * alpha_b1 -- after 5, so og8 AND alpha_b1 are written on the launch stream before the fork
        if self.need_dw_o:
            if side is not None:
                with side.issue(launch_ts, "o") as side_stream:
                    self._out_proj_wgrad.execute(v.dy8, v.og8_hd, c.dw_o, v.gemm_ws_side, stream=side_stream, alpha=v.alpha_b1)
            else:
                self._out_proj_wgrad.execute(v.dy8, v.og8_hd, c.dw_o, v.gemm_ws, stream=stream, alpha=v.alpha_b1)
        # 7. (B4) the MXFP8 row: the payloads and their seven scale-factor blobs, the record's lse, the block's delta, the dead
        #    half-precision ports (saved.o and the bf16 dO: required by the row's append-only ABI, read by nothing under the external delta)
        self._sdpa.execute(
            v.q8_bshd,
            v.k8_bshd,
            v.v8_bshd,
            c.saved.o,
            v.do8_bshd,
            c.saved.lse,
            v.dq_bshd,
            v.dk_bshd,
            v.dv_bshd,
            workspace=v.sdpa_ws,
            stream=stream,
            delta=v.delta,
            q_T8=v.q_T8_bshd,
            k_T8=v.k_T8_bshd,
            do_T8=v.do_T8_bshd,
            do16=v.do_gated_bshd,
            sf=dict(sf),
        )
        # 8. (B5+B6) RoPE^T + RMSNorm backward into the Q / K bands, dV into the V band, fp32 dW partials -- bf16, no amax fold
        norm = g.qk_norm
        self._norm_bwd.execute(
            v.dq,
            v.dk,
            v.dv,
            c.q_pre_b if norm else None,
            c.k_pre_b if norm else None,
            c.saved.rstd_q.view(t, g.h_q) if norm else None,
            c.saved.rstd_k.view(t, g.h_kv) if norm else None,
            c.w_q_norm,
            c.w_k_norm,
            cos_t,
            sin_t,
            v.dqkvg_q,
            v.dqkvg_k,
            v.dqkvg_v,
            v.plane_q,
            v.plane_k,
            stream=stream,
        )
        # 9. the EPILOGUE: the fixed-order dW_norm reduce (need_dw_norms) + dQKVG block-quantized from ONE read into the GEMMs' canonical
        #    scale-factor order -- rowwise [T, N] for B8 (need_dh), transposed [N, T] for B7 (need_dw_qkvg); the halves not requested are
        #    folded out of the artifact and their operands stay unbound
        if self._epilogue is not None:
            self._epilogue.execute(
                plane_q=v.plane_q,
                plane_k=v.plane_k,
                dw_q_norm=c.dw_q_norm,
                dw_k_norm=c.dw_k_norm,
                src=v.dqkvg_rows,
                dst=v.dqkvg8_rows if self.need_dh else None,
                sf=v.sf_dqkvg if self.need_dh else None,
                dst_t=v.dqkvg_t8 if self.need_dw_qkvg else None,
                sf_t=v.sf_dqkvg_t if self.need_dw_qkvg else None,
                stream=stream,
            )
        # 10. (B7) dW_qkvg = dqkvg_t8 . h_t^T over the caller's transposed h -- forked HERE under fuse_wgrad_overlap: the epilogue wrote
        #     dqkvg_t8 and its blob on the launch stream first
        if self.need_dw_qkvg:
            if side is not None:
                with side.issue(launch_ts, "qkvg") as side_stream:
                    self._qkv_gate_wgrad.execute(v.dqkvg_t8, h_t, c.dw_qkvg, v.gemm_ws_side, stream=side_stream, sf_a=v.sf_dqkvg_t, sf_b=h_t_sf)
            else:
                self._qkv_gate_wgrad.execute(v.dqkvg_t8, h_t, c.dw_qkvg, v.gemm_ws, stream=stream, sf_a=v.sf_dqkvg_t, sf_b=h_t_sf)
        # 11. (B8) dh = dqkvg8 . w_qkvg_t^T over the caller's transposed W_qkvg
        if self.need_dh:
            self._qkv_gate_dgrad.execute(v.dqkvg8, w_qkvg_t, c.dh.view(t, dm), v.gemm_ws, stream=stream, sf_a=v.sf_dqkvg, sf_b=w_qkvg_t_sf)
        # fuse_wgrad_overlap: JOIN before this call returns (Rule 5) -- and before the NEXT execute's prologue zeroes the block
        if side is not None:
            if self.need_dw_o:
                side.join(launch_ts, "o")
            if self.need_dw_qkvg:
                side.join(launch_ts, "qkvg")


# ---------------------------------------------------------------------------
# 6. Convenience wrapper — allocates, then delegates
# ---------------------------------------------------------------------------

_BWD_CACHE: dict = {}


def _detach(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """A view sharing storage that the kernels can take through DLPack (a tensor with ``requires_grad`` cannot be exported)."""
    return None if x is None else x.detach()


def gated_attention_block_backward(
    dy: torch.Tensor,
    saved: SavedForBackward,
    w_qkvg: torch.Tensor,
    w_q_norm: Optional[torch.Tensor],  # None (both) iff geometry.qk_norm is False
    w_k_norm: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    w_o: torch.Tensor,
    geometry: GatedAttentionBlockGeometry,
    *,
    seq_lens: Optional[torch.Tensor] = None,
    recompute: RecomputePolicy = RecomputePolicy.RECOMPUTE_QK_PRE,
    current_stream: Optional[cuda.CUstream] = None,
    fuse_gate_bwd: bool = False,
    fuse_wgrad_overlap: bool = False,
    thd: bool = False,
    max_seq_len: Optional[int] = None,
    quant: Optional[Union[QuantSpec, MxQuantSpec]] = None,
    grad_scaling: str = "current",
    scale_dp: Optional[torch.Tensor] = None,
    scale_dy: Optional[torch.Tensor] = None,
    scale_do: Optional[torch.Tensor] = None,
    scale_dqkvg: Optional[torch.Tensor] = None,
    h_t: Optional[torch.Tensor] = None,
    h_t_sf: Optional[torch.Tensor] = None,
    w_qkvg_t: Optional[torch.Tensor] = None,
    w_qkvg_t_sf: Optional[torch.Tensor] = None,
    w_o_t: Optional[torch.Tensor] = None,
    w_o_t_sf: Optional[torch.Tensor] = None,
    grad_scale_margin_log2: int = FP8_GRAD_SCALE_MARGIN_LOG2,
) -> TupleDict:
    """Allocate gradients + workspace, cache the compiled block, and run it.

    Returns ``{"dh", "dw_qkvg", "dw_o", "dw_q_norm", "dw_k_norm"}``. Which
    entries are non-``None`` follows ``requires_grad`` on the corresponding
    forward inputs (``saved.h``, ``w_qkvg``, ``w_o``, ``w_q_norm`` / ``w_k_norm``),
    snapshotted here -- the same policy a torch autograd ``Function`` would
    apply, and the natural place to wire one later; nothing requiring a
    gradient is a ``ValueError``. ``dw_q_norm`` / ``dw_k_norm`` are ``None``
    under ``geometry.qk_norm=False``. The tensors are handed to the block
    DETACHED (views, no copy). The compiled block is cached per declaration
    (shapes, dtypes, device, geometry, needs, policy); the workspace and the
    gradients are allocated per call -- this is the convenience path, the class
    is the allocation-free one. Both are allocated, and the block is run, on the
    LAUNCH stream (``current_stream``, else torch's current stream): the caching
    allocator orders a buffer's reuse only against the stream it was allocated
    on, so a workspace allocated on the ambient stream for a side-stream launch
    would be freed into the ambient pool at return and handed to the caller's
    next allocation while the backward is still writing it.  ``fuse_gate_bwd``
    and ``fuse_wgrad_overlap`` (appended, default off) are the block's fusion /
    scheduling knobs, part of the cache key; under ``fuse_wgrad_overlap`` the
    side-stream GEMMs are joined back to the launch stream before ``execute``
    returns, so the per-call workspace freed here is reused only behind them.
    With the weights frozen (neither ``w_o`` nor ``w_qkvg`` requires a
    gradient) there is no weight-gradient GEMM to overlap, and the wrapper runs
    the in-order block instead of surfacing the class's typed decline -- the
    needs follow ``requires_grad`` here, not an explicit declaration, and a
    frozen-weights phase of a training loop must not fail over a scheduling
    knob; the EFFECTIVE knob value is what the cache key carries.
    ``thd`` / ``max_seq_len`` (appended): the packed-sequence backward over the
    record a ``thd=True`` training forward wrote -- ``num_sequences`` and the
    length FORM (``cu_seqlens``) are derived from the record itself
    (``saved.seq_lens.numel()`` and ``saved.seq_lens_form``; a record without
    them is the class's typed decline, never a wrapper crash) and join the cache
    key with ``thd`` and ``max_seq_len``; ``thd=True`` without ``max_seq_len`` is
    a ``ValueError`` naming ``max_seq_len`` alone (the wrapper has no
    ``num_sequences`` to ask for); ``seq_lens`` passes through unchanged
    (``None`` or ``saved.seq_lens`` itself) and ``fuse_gate_bwd`` passes through
    (served under ``thd`` exactly as dense: the packed block stays bitwise).
    ``quant`` / ``grad_scaling`` (appended): the quantized backward's declaration
    attributes, part of the cache key (``dataclasses.astuple(quant)``); the
    gradients are then allocated in ``dy``'s dtype (bf16 -- ``saved.h`` and the
    weights are e4m3 codes under ``quant``, so ``empty_like`` would get it
    wrong), the weight gradients at the GEOMETRY's logical shapes ``(n_qkvg,
    d_model)`` and ``(d_model, H_q * D)`` -- a packed e2m1 weight of the fp4
    modes has its storage shape ``[rows, K // 2]``, so ``empty_like`` would halve
    them; such a weight carries ``requires_grad`` like any other tensor, and its
    gradient is requested the same way; ``scale_dp`` / ``scale_dy`` / ``scale_do``
    / ``scale_dqkvg`` pass through to ``execute`` unchanged (the class checks them
    both ways).
    ``h_t`` / ``h_t_sf`` / ``w_qkvg_t`` / ``w_qkvg_t_sf`` (appended): the MXFP8
    backward's transposed block-scaled artifacts, passed through to ``execute``
    unchanged (the class requires / refuses each by the block's needs); their
    PRESENCE joins the cache key (a block declared over an artifact set is one
    declaration), their bytes never do.  ``w_o_t`` / ``w_o_t_sf`` (appended): the
    fp4 weight modes' transposed e2m1 ``W_o`` with its blob, required iff
    ``MxQuantSpec.o_fp4`` -- the same pass-through, the same key rule.
    ``grad_scale_margin_log2`` (appended): the quantized backward's gradient-scale
    margin, a declaration attribute like ``grad_scaling`` -- handed to the class
    and part of the cache key (a different margin is a different compiled block).
    """
    need_dh = bool(saved.h.requires_grad)
    need_dw_qkvg = bool(w_qkvg.requires_grad)
    need_dw_o = bool(w_o.requires_grad)
    need_dw_norms = bool(geometry.qk_norm and ((w_q_norm is not None and w_q_norm.requires_grad) or (w_k_norm is not None and w_k_norm.requires_grad)))
    if not (need_dh or need_dw_qkvg or need_dw_o or need_dw_norms):
        raise ValueError("gated_attention_block_backward: nothing requires a gradient (saved.h, w_qkvg, w_o, w_q_norm / w_k_norm all have requires_grad=False)")
    # Frozen weights + the scheduling knob: nothing to put on the side stream, so run the in-order block (the class
    # keeps its typed decline for an EXPLICIT need_* declaration).  The effective value reaches the block and the key.
    fuse_wgrad_overlap = bool(fuse_wgrad_overlap) and (need_dw_o or need_dw_qkvg)
    # THD: the record says how many sequences and in which form it packed its lengths; the class validates both.  The one
    # knob the wrapper cannot derive is max_seq_len -- asked for by name here, since the class's message would also name
    # num_sequences, which this wrapper has no parameter for.
    thd = bool(thd)
    if thd and max_seq_len is None:
        raise ValueError(
            "thd=True on gated_attention_block_backward needs max_seq_len (S_max, the longest sequence the plan admits); num_sequences and "
            "the length form are derived from the record (saved.seq_lens.numel(), saved.seq_lens_form)"
        )
    cu_seqlens = thd and _seq_lens_form(saved) == _THD_FORM_PREFIX
    num_sequences = None
    if thd and isinstance(saved.seq_lens, torch.Tensor):
        num_sequences = int(saved.seq_lens.numel()) - (1 if cu_seqlens else 0)
    seq_lens_present = ((seq_lens is not None) or (saved.seq_lens is not None)) and not thd
    saved_d = dataclasses.replace(
        saved,
        h=_detach(saved.h),
        gate=_detach(saved.gate),
        o=_detach(saved.o),
        lse=_detach(saved.lse),
        rstd_q=_detach(saved.rstd_q),
        rstd_k=_detach(saved.rstd_k),
        q_pre=_detach(saved.q_pre),
        k_pre=_detach(saved.k_pre),
        proj_slab=_detach(saved.proj_slab),
        seq_lens=saved.seq_lens,
    )
    dy_d, w_qkvg_d, w_q_d, w_k_d, cos_d, sin_d, w_o_d = (_detach(x) for x in (dy, w_qkvg, w_q_norm, w_k_norm, cos, sin, w_o))
    key = (
        tuple(dy.shape),
        dy.dtype,
        str(dy.device),
        tuple(sorted(dataclasses.asdict(geometry).items())),
        need_dh,
        need_dw_qkvg,
        need_dw_o,
        need_dw_norms,
        recompute,
        seq_lens_present,
        bool(fuse_gate_bwd),
        bool(fuse_wgrad_overlap),
        thd,
        None if max_seq_len is None else int(max_seq_len),
        num_sequences,
        cu_seqlens,
        # a wrong-typed quant misses the cache and reaches the class's typed decline (its key is its type name)
        (type(quant).__name__, dataclasses.astuple(quant)) if dataclasses.is_dataclass(quant) and not isinstance(quant, type) else (type(quant).__name__,),
        grad_scaling,
        # the margin by TYPE and value: True == 1 and 1.0 == 1 in Python, and a cached int-margin block must never serve a caller
        # whose bool / float reaches the class's ValueError only on a miss
        (type(grad_scale_margin_log2), grad_scale_margin_log2),
        # the MXFP8 (and fp4) artifacts' PRESENCE (never their bytes): which of the six the caller handed over
        tuple(x is not None for x in (h_t, h_t_sf, w_qkvg_t, w_qkvg_t_sf, w_o_t, w_o_t_sf)),
    )
    blk = _BWD_CACHE.get(key)
    if blk is None:
        blk = GatedAttentionBlockBwd(
            dy_d,
            saved_d,
            w_qkvg_d,
            w_q_d,
            w_k_d,
            cos_d,
            sin_d,
            w_o_d,
            geometry,
            recompute=recompute,
            need_dh=need_dh,
            need_dw_qkvg=need_dw_qkvg,
            need_dw_o=need_dw_o,
            need_dw_norms=need_dw_norms,
            seq_lens_present=seq_lens_present,
            fuse_gate_bwd=fuse_gate_bwd,
            fuse_wgrad_overlap=fuse_wgrad_overlap,
            thd=thd,
            num_sequences=num_sequences,
            max_seq_len=max_seq_len,
            cu_seqlens=cu_seqlens,
            quant=quant,
            grad_scaling=grad_scaling,
            grad_scale_margin_log2=grad_scale_margin_log2,
        )
        blk.check_support()
        blk.compile()
        _BWD_CACHE[key] = blk
    dev = dy.device
    # Rule 5 / recipe R2: the per-call scratch and the gradients are allocated, and the block is launched, on the
    # LAUNCH stream.  The workspace reference dies at return; allocated on torch's ambient stream while
    # ``current_stream`` names a side stream it would be freed into the ambient pool and could back the caller's next
    # allocation while the backward is still writing it (silent gradient corruption under load).  ``stream_context``
    # is a no-op for ``None`` and for a handle equal to torch's current stream.
    with stream_context(current_stream, dev):
        workspace = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device=dev)
        # the gradients in the ACTIVATION dtype (dy's): under quant saved.h / the weights are e4m3 codes, the gradients bf16.  The
        # weight gradients take the GEOMETRY's logical shapes, never the weight's own: a packed e2m1 weight (the fp4 weight modes'
        # float4_e2m1fn_x2 W_qkvg / W_o, which carries requires_grad like any other tensor) has its STORAGE shape [rows, K // 2], so
        # empty_like would allocate a half-width gradient that execute refuses.  dh keeps saved.h's shape (h is e4m3 in every quant
        # mode, never packed).
        dh = torch.empty_like(saved_d.h, dtype=dy_d.dtype) if need_dh else None
        dw_qkvg = torch.empty(geometry.n_qkvg, geometry.d_model, dtype=dy_d.dtype, device=dev) if need_dw_qkvg else None
        dw_o = torch.empty(geometry.d_model, geometry.h_q * geometry.d_head, dtype=dy_d.dtype, device=dev) if need_dw_o else None
        dw_q_norm = torch.empty(geometry.d_head, dtype=torch.float32, device=dev) if need_dw_norms else None
        dw_k_norm = torch.empty(geometry.d_head, dtype=torch.float32, device=dev) if need_dw_norms else None
        blk.execute(
            dy_d,
            saved_d,
            w_qkvg_d,
            w_q_d,
            w_k_d,
            cos_d,
            sin_d,
            w_o_d,
            dh=dh,
            dw_qkvg=dw_qkvg,
            dw_o=dw_o,
            dw_q_norm=dw_q_norm,
            dw_k_norm=dw_k_norm,
            workspace=workspace,
            seq_lens=seq_lens,
            current_stream=current_stream,
            scale_dp=scale_dp,
            scale_dy=scale_dy,
            scale_do=scale_do,
            scale_dqkvg=scale_dqkvg,
            h_t=_detach(h_t),
            h_t_sf=_detach(h_t_sf),
            w_qkvg_t=_detach(w_qkvg_t),
            w_qkvg_t_sf=_detach(w_qkvg_t_sf),
            w_o_t=_detach(w_o_t),
            w_o_t_sf=_detach(w_o_t_sf),
        )
    # the per-call workspace dies at return: drop the block's cached views of it (they would pin its storage until the next call)
    blk.release_workspace_views()
    return TupleDict(dh=dh, dw_qkvg=dw_qkvg, dw_o=dw_o, dw_q_norm=dw_q_norm, dw_k_norm=dw_k_norm)
