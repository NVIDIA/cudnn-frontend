# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The checks of a sample run against the plain-PyTorch reference.

Four checks, run after the timed loop on the ids the loop produced:

1. **replay** -- the same inputs from a fresh state (new allocations) give byte-identical ids and logits.
2. **blocks** -- teacher-forced: every provider block that is not the reference chain itself is compared, call by call, with
   the reference block on the SAME operands and the same state, so an error cannot accumulate and a wiring error is told apart
   from rounding.  The band of a block is named in its provider row (``BANDS``) and is never widened here.  Every band is
   judged element by element and asks for finite values first: a NaN is a failure of the block that produced it, never a tie.
   A run in which no block is a provider's (every judged block selected onto its reference row) has nothing to judge: the
   line is then REPORTED, ``Checks.judged`` is 0, and a sample does not print ``PASS`` for it.
3. **tokens** -- end to end: the reference stack runs the same inputs with its own state.  The largest distance between the
   logits of the two stacks must be finite and inside the logits band, which the family declares beforehand in bf16 units of
   the run's largest reference logit (``logits_band_units``); and the greedy token must equal the reference's at every step
   whose reference margin (largest minus second largest logit) exceeds ``LOGITS_FACTOR`` times that band.  Steps below it are
   counted and printed as near-ties; they are not failures and not hidden.  The band is declared, not taken from the run: a
   band that follows the run's own distance can never fail.  The check cannot be sharper than the two stacks are close: a
   chain of 8-bit and 4-bit quantisers turns a one-unit rounding difference into different codes, so the logits of two
   correct stacks differ by more than one rounding, and the block check above is what verifies each provider.  Where a
   family can declare no band for a model -- none that passes two correct stacks and fails a wrong one -- it declares
   ``None``: the line is then REPORTED, prints the distance and the count of equal tokens, and fails only on a logit that
   is not finite.  A line that cannot fail on a finite value is not printed as PASS.
4. **batch invariance** -- with more than one sequence, every block is run again on ONE sequence alone, on the same operands
   and state; the bytes of that sequence must be equal.  The sequence that runs alone rotates with the step (step ``t``:
   sequence ``t`` modulo the count), so every row of the batch is covered at no extra launch.  A gate for a block whose
   provider row lists the GPU class under ``batch_invariant``; a printed fact for the rest.

A block whose calls several provider rows serve by call site (``blocks.BySite``) is judged per call site, under ``<block>
at <call site>``: each part against its own twin, in its own row's band, with its own row's batch-invariance statement.

A fifth check, asked for by the run (``run_checks(..., kernels=True)``; a sample's ``--check-kernels``), is not against the
reference but against the plan: **kernels** -- every launch of the traced steps is held against the census entries of the
blocks (``model_samples.census``: the form on the objects, attribution, count, order, launch record, copies).  It reads what
the traced pass kept (``Session.traced``).  Where the profiler's records cannot be had it is neither PASS nor FAIL: the line
says ``NOT OBSERVED`` with the reason, ``Checks.unobserved`` carries it, and a sample then ends in ``RAN``, never in ``PASS``.

**Where the reference's weights come from.**  The reference blocks multiply bf16 copies of the weights' code values.  A
run keeps such a copy of every weight that a block of its plans reads as one; a weight that every plan multiplies on its
stored format has none (``Session.uncopied_weights``).  For those the checks make the copy themselves, for one call
at a time, and drop it after the call (``weights.OnDemand``): in the teacher-forced pass for the twin's call, and in
the reference stack's own passes for every call (``Session(on_demand=...)``).  Such a copy is made from the stored
bytes as they are at the call: bit for bit the copy a load would have kept as long as no block has written into its
weight, and then no check line depends on which of the two a run has.  A block that did write would move its own
reference with it and pass its band, so that condition is itself a check, **stored weights**: the loader takes a
checksum of every weight it keeps on its stored format alone, and the checks take it again before their first pass and
after their last.  A weight whose stored bytes are no longer those of the load fails the run, named by block and layer,
whatever the block lines say.  What the copies cost is printed under the checks (``on demand``: how many copies, the
largest, their bytes in all, the seconds), and the largest one is part of the run's memory sentence.  A run that keeps
every weight as a copy prints neither line.

**A session whose step feeds several tokens per sequence** (the verify step; ``Session(tokens=T)``) is checked by the same
five checks on its own steps: the rows of a step are sequences x T, the reference stack runs the same verify steps, the
reference of the fused verify block is T reference decode steps, each started from the state the kernel left for the
token before it -- so its line says, token by token, that the state the kernel wrote is one decode step on from the state
it wrote before, in the band of the decode step and with that band's meaning (a rounding on the other side of a midpoint
is seen at its token, once; a cell is one value head of one sequence over the tokens of its steps) -- and the one
sequence that runs alone runs with its T rows.  Where an entry itself states that one sequence alone is not byte-equal to the sequence among several
(``Block.batch_note``), the batch line carries that statement and gates nothing on the block.  ``verify_checks`` then holds
such a session against the decode steps of a second session on the same weights: the same token stream through verify
steps that keep every possible number of tokens once, the greedy tokens of both, and after every step the state the
verify run stands on against the state of the decode steps of the kept tokens.

**A session that feeds its prompt through decode steps and verifies behind it** (``Session(draft_tokens=k)``) has two
kinds of step on one state.  Its own checks replay the whole recorded run from a zero state -- the prompt's decode steps
under the block check of the decode plan, the verify steps under the block check of the verify plan -- and print
``blocks`` and ``batch invariance`` once per kind of step.  ``speculative_checks`` holds such a run against decode steps of
the same tokens: beside the tokens and the kept state, three lines that no rounding can fail -- the position (an
integer), the prompt (what every layer stands on behind it, byte for byte: both runs made the same decode steps) and the
rejected rows (a second replay with other tokens behind the kept ones leaves every kept row as it was) -- and ``bytes``,
gated where a verify step is that many decode steps of one chain (the reference stack) and REPORTED elsewhere.  A
comparison of greedy tokens alone does not see a position that is off by one or a kept row that read a rejected token
where a model's token does not follow its state; those three lines are there for that.  For a session with a drafter
the comparison also writes the drafter's passes out a second time, at one row per call from books of its own, and holds
the session's passes to them: ``drafter cache`` and ``draft proposals`` -- what a step hands its passes, not the
arithmetic of a block, which both runs share.

Bands: ``bf16_rounding`` / ``bf16_unit`` = every element finite and within one bf16 unit of the reference, the unit taken at
the larger magnitude and not below 2**-8 of the tensor's largest value (a multiply that rounds its unscaled sum to bf16 and the
scaled result again differs from the once-rounded reference by one unit on a share of its elements); ``gdn_step`` = see below;
``gate_fp8`` = see below; ``add_norm_quant`` = see below; ``swiglu_fp4`` = see below; ``attention_prepare`` = see below;
``same`` = the provider is the reference chain, nothing to compare.
``attention_bf16_weights`` = see below: the bound of a stated model, for a library kernel that documents no contract.

**What a provider block is compared with.**  The reference block of the same name -- unless the provider's kernel rounds at
other points than the reference chain: then the block names its twin (``Block.twin``), a reference block written at the
kernel's rounding points, and is compared with that.  The reference is matched to the kernel's documented arithmetic; a
band is never widened to cover a chain that rounds elsewhere.  **A band is judged per output**: it names the outputs it
applies to and what it needs of the twin; an output it does not name is compared by its dtype (bf16: within one bf16
unit; anything else: bytes), and state a call updates is compared byte for byte unless the band has a rule for it.

**The one exception to a twin at the kernel's rounding points** is a library kernel that documents no numerics contract
and whose rounding points are not established: its twin is the reference chain itself, which also keeps the operand
terms of a bound, and its band is that bound -- derived from a stated model of the kernel's arithmetic, declared with
that model before any run of the row, and never taken from a measured distance (``attention_bf16_weights`` below).

``gate_fp8``, the band of the attention gate + quantise entry, is that entry's documented numerics contract, not a share.
Its kernel forms the gated value as the quotient ``attn / (1 + exp(-gate))`` in fp32, rounds it once to bf16 and once to an
8-bit code.  The twin forms the same quotient in the reference's accumulation precision and keeps the two bf16 values
around it.  The band: every byte is the code of one of those two bf16 values, and the code of the nearest one wherever the
quotient is more than 2.5 fp32 units from their midpoint; every value finite.  Inside that margin the fp32 quotient may
fall on the other side of the midpoint, which is the kernel's own arithmetic; the line prints how many bytes did.

``add_norm_quant``, the band of the add + RMSNorm + quantise entry, is that entry's documented numerics contract too
(``docs/operations/normalizations/add_norm_quant_decode.md``).  Its kernel normalises in fp32, rounds the normed value
once to bf16 and forms its codes from that bf16 value; the twin evaluates the same value in fp64 and keeps the two bf16
values around it.  Per output of a call, whatever the form of its call site writes:

* the residual stream: byte for byte ``bf16(fp32(branch) + fp32(residual))``;
* the normed rows: every element one of the two bf16 values around the normalised value, and the nearest one wherever
  that value is more than 20 fp32 units from their midpoint (the entry's page: its fp32 value is within 20 x 2**-24 of
  the fp64 one);
* 8-bit codes: where the form also writes the normed rows, byte for byte the static quantisation of THOSE rows; where
  it writes codes alone, every byte the code of one of the two bf16 values, the nearest one's beyond the margin;
* 4-bit codes: every block scale ``e4m3(global * (largest * fp32(1/6)))`` of a block maximum the two bf16 values admit;
  and, under the block scale the kernel stored, every code inside the bracket of its two approximate reciprocals
  around the code of an admitted bf16 value (``torch_ref.nvfp4_code_bracket``: two adjacent codes at most).

Every value finite.  The line prints how many elements took the other bf16 value, and how many codes are not the ones
of the nearest value; anything outside these sets is a failure, never a share.

``swiglu_fp4``, the band of the fused 4-bit gate / up entry, is that entry's documented numerics contract as well
(``docs/operations/gemm_fusions/swiglu_fp4_decode.md``), and it is an interval, not a bracket.  The kernel accumulates the
gate and the up product in fp32 over the whole input width, rounds each once to bf16, forms ``silu(gate) * up`` with an
approximate exponential and division, rounds the product to bf16 and writes 4-bit codes under block scales it forms from
that value; nothing of the product is stored.  An fp32 sum of thousands of products is not within a few units of the
fp64 one, so the twin keeps no two bf16 values: it evaluates the chain in fp64 on the exactly dequantised operands and
propagates the entry's error bound to every element (``torch_ref.gate_up_bound``: the terms the entry's own suite
derives).  Per call:

* every block-scale byte lies in the interval of bytes that bound admits: between the bytes of ``global * (largest -
  error) / 6`` and of ``global * (largest + error) / 6``, each rounded to nearest even, with the relative slack of the
  kernel's three fp32 multiplications;
* every 4-bit code, read under the block scale the kernel stored, is within half a code step of the fp64 value plus
  that element's error bound (and what a saturating block clips).

Every value finite.  The line prints how many block scales and codes are not those of the fp64 chain rounded at the
kernel's points, and the largest share of its bound a code used; anything outside is a failure, never a share.  What
this band cannot see is the entry's to hold: one bf16 unit of the product hides behind half a 4-bit code step, which
the exact cells of the entry's own suite cover.

``attention_prepare``, the band of the attention preparation entry, is that entry's documented numerics contract
(``docs/operations/attention/attention_prepare_decode.md``).  Its kernel normalises a head row in fp32 with an approximate
division and reciprocal root, rounds the normalised value once to bf16, rotates the rotary part of it with one fp32 product
and one fused sum per element and rounds once more; the gate rows and the value rows are bit copies; and it writes the
keys and values of a step's rows into the cache's pool in the same launch.  The twin has the same rounding points with
the mean and the root evaluated in fp64, and writes the cache through its accessor at the positions of the call, which
it is handed as a host integer.  Per call:

* a query element that is not rotated: within one bf16 unit of the twin's, the unit taken at the larger magnitude;
* a rotated one: inside that unit plus ``|cos|`` and ``|sin|`` times one unit of its two normalised operands (each may be
  one unit off in the kernel; the twin keeps that part per element, ``torch_ref.prepare_rows``);
* the gate rows: the twin's bytes;
* the cache: the key rows at the positions of the call held as the query rows are, against the key rows the twin wrote;
  their value rows the twin's bytes; and every other byte of the pool the byte the twin's pool has -- a row the kernel
  wrote at another slot, or a row no position of the call names, fails here.  Which rows of the pool the positions of
  the call are is formed from host integers: the position, the tokens per sequence and the shape of the pool.

Every value a call wrote must be finite: the queries, the gate rows, and the key and the value rows at the positions of
the call.  The gate rows and the value rows are held finite beside their bytes: the twin copies them from the same
operand, so a NaN or an infinity there is the twin's byte too.  The line prints how many query and key elements are not
the twin's byte -- the approximate root moves a few by a unit, inside their band; the entry's page states no share for
them, and none is gated -- and how many of them are outside their band.

``attention_bf16_weights``, the band of an attention kernel of a library that documents no numerics contract, is not a
contract of that kernel: it is the bound of a stated model of the kernel's arithmetic, declared here before any run of a
row that names it.  Which values such a kernel rounds is not established, so no twin can be written at its rounding
points.  The twin is the reference chain itself (``torch_ref.AttentionBound``), which also keeps, per output element, the
two operand terms of the bound -- in fp64, whatever precision the run's reference accumulates in.  With ``p_j`` the
softmax weight of the attended row ``j`` and ``v_j`` that row's value::

    chain  = sum_j p_j v_j        (the reference attention before its rounding to bf16)
    spread = sum_j p_j |v_j|

The model: every softmax weight carries one rounding to nearest at eight significant bits, ``p_j (1 + d_j)`` with ``|d_j|
<= u`` and ``u = 2**-8 / (1 + 2**-8)``; the number the weighted sum is divided by carries one of its own, ``1 + e`` with
``|e| <= u``; and the output is rounded to bf16 once.  Before that last rounding the model's output is ``(sum_j p_j (1 +
d_j) v_j) / (1 + e)``, so its distance from the chain is ``|sum_j p_j d_j v_j - e chain| / (1 + e)``, at most ``u (spread +
|chain|) / (1 - u)`` -- and ``u / (1 - u)`` is exactly ``2**-8``.  The output's own rounding adds half a spacing of bf16
values at the output.  The band, element by element::

    out is finite and |out - chain| <= 1 x [ 2**-8 x (spread + |chain|) + 1 x spacing ]

``spacing`` is the distance between two neighbouring bf16 values at the magnitude of the chain (at the smallest normal
fp32 value where the chain is below it): the chain's alone, neither the output's nor the tensor's largest value's.  The
coefficient, the weight term and the count of spacings are ``ATTENTION_BAND_COEFFICIENT``, ``ATTENTION_WEIGHT_TERM`` and
``ATTENTION_SPACINGS``.  An element whose chain or spread is not finite is outside, whatever the output is.  The call
writes nothing, so the cache it reads is held byte for byte.  The line prints how many
elements are not the reference chain's byte, how many are outside their bound, and the largest share of its bound an
element used with the chain and the spread of that element; the share is printed, not gated.

What this band is and is not.  It is derived from a model and from no measured distance, and it claims nothing about
what the kernel rounds.  It is the model's bound to first order: one spacing at the chain covers the output's rounding
wherever the chain is not far below its bound, and where it is, half a spacing at the output is more than one spacing at
the chain -- there an output of the model itself can use up to ``1 + 2**-8`` of its bound.  So an element outside is a
finding, read with its share, its chain and its spread; it is never a reason for another coefficient, a further term or
a shape left out.  A block in this band is not within one bf16 unit of the chain: the band's term is the weighted
magnitude of the attended values, not the output's.  And what the band cannot see is the row's to say: one wrong,
missing or extra attended row whose softmax weight is below the band's term.

``gdn_step``, the band of the fused linear-attention step, is judged twice.  Over the whole run: more than 99.9 % of the 8-bit
output bytes equal, more than 99.9 % of the recurrent state's elements within 1e-4, none farther than two bf16 units (2**-6)
of the state's largest value (of 1 when that is larger), every value finite, the conv history byte-identical and the null
slot untouched.  And per cell -- one value head of one sequence in one layer, which is one block of the step's launch grid
and so its natural unit of failure: over the run, more than 99 % of the cell's output bytes equal and more than 99.9 % of
the cell's state within 1e-4.  The null slot is in no share.

Why the state band has a share and a cap: the fused step rounds its beta / alpha projection, its conv taps and its SiLU to
bf16 as a serving chain does, and it accumulates them in fp32 in its own order.  Against an fp64-accumulated reference a few of
those roundings fall on the other side of a midpoint; one such unit moves one head's write gate, its decay or one key column,
and with it that head's state, by up to a few 1e-3 of the state's largest value, while every other element agrees to fp32
rounding.  A state "within 1e-4 everywhere" therefore does not hold against this reference (measured by this sample: the
largest distance of a default run is a few 1e-4 to a few 1e-3), and it is not what a wiring error looks like: that moves every
element.  The cap is one unit for each of the two gates of a head.

What the band tolerates: such a move of one head at one step.  It is an event, not a state of the head: the reference starts
every call from the provider's state, so the move is seen once.  Measured over runs of this sample with two to eight layers,
the worst cell of a run had about 0.1 % of its output bytes different and about 0.01 % of its state outside 1e-4, a tenth of
what the band allows.  What it still catches: a value head of one sequence that is wrong at every step, from two of its 128
output bytes or 0.1 % of its state's elements on; a NaN anywhere; any error of the wiring or of an operand's layout (with the
beta and alpha rows exchanged, 23 % of the output bytes were equal, a tenth of the state was within 1e-4 and the largest
distance was 0.99); a write to the null slot; and any element beyond the cap.  One wrong output byte of one head at every
step stays inside the band.  The line of the step also prints the state's rms ratio -- the rms of the difference over the rms
of the reference state, at the worst call -- as a second, averaged view of the same comparison: one flipped head barely moves
it, an error of the wiring moves it to order one.  It is printed, not gated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from . import census
from .blocks import GIB, call_site, scoped
from .loop import Session
from .sampling import margin
from .weights import OnDemand

LOGITS_FACTOR = 2.0
GDN_EQUAL_BYTES = 0.999
GDN_CELL_BYTES = 0.99
GDN_STATE_BAND = 1e-4
GDN_STATE_SHARE = 0.999
GDN_CELL_STATE = 0.999
GDN_STATE_CAP = 2.0**-6
#: the gate entry's contract: beyond this many fp32 units from the midpoint of the two bf16 values around the gated value, the
#: byte is the code of the nearest one (``docs/operations/attention/gate_quant_decode.md``: measured over every operand
#: pair at five scales, doubled).  Not a tolerance of this sample: it moves only with that page
GATE_MARGIN_ULP32 = 2.5
#: the norm entry's contract: its fp32 value before the one rounding to bf16 is within 20 x 2**-24 of the fp64 value
#: (``docs/operations/normalizations/add_norm_quant_decode.md``), which is at most 20 fp32 units; beyond that distance
#: from the midpoint of the two bf16 values around it the kernel rounds to the nearest one.  It moves only with that page
NORM_MARGIN_ULP32 = 20.0
#: the band of a library attention kernel that documents no numerics contract (``attention_bf16_weights``): an element is
#: inside where ``|out - chain| <= COEFFICIENT * (WEIGHT_TERM * (spread + |chain|) + SPACINGS * spacing(|chain|))``.  The
#: bound of a stated model of the kernel's arithmetic (the module's docstring), declared before any run of a row that
#: names the band and never taken from a measured distance: an element outside is a finding, and none of the three
#: moves to make a run pass
ATTENTION_BAND_COEFFICIENT = 1.0
ATTENTION_WEIGHT_TERM = 2.0**-8
ATTENTION_SPACINGS = 1.0
BANDS = {
    "same": "the provider is the reference chain",
    "bitwise": "byte-identical",
    "bf16_rounding": "finite and within one bf16 rounding",
    "bf16_unit": "finite and within one bf16 unit",
    "gdn_step": (
        f"8-bit bytes: more than {GDN_EQUAL_BYTES:.1%} equal over the run and more than {GDN_CELL_BYTES:.0%} in every cell (layer, sequence, value head); "
        f"state: more than {GDN_STATE_SHARE:.1%} within {GDN_STATE_BAND:g} over the run and more than {GDN_CELL_STATE:.1%} in every cell, "
        "all of it finite and within two bf16 units; conv history equal; null slot untouched"
    ),
    "gate_fp8": (
        "8-bit bytes against the gate at the kernel's rounding points: every byte the code of one of the two bf16 values around the gated value "
        f"attn / (1 + exp(-gate)), and the code of the nearest one wherever that value is more than {GATE_MARGIN_ULP32:g} fp32 units from their midpoint; "
        "every value finite"
    ),
    "add_norm_quant": (
        "against the norm at the kernel's rounding points: the residual stream byte for byte; every normed element one of the two bf16 values around "
        f"the normalised value, the nearest one wherever that value is more than {NORM_MARGIN_ULP32:g} fp32 units from their midpoint; 8-bit codes those of "
        "the normed rows the kernel wrote, or of one of the two bf16 values where it wrote codes alone; 4-bit block scales those of a block maximum the "
        "two values admit, 4-bit codes inside the bracket of the kernel's two approximate reciprocals under its own block scale; every value finite"
    ),
    "swiglu_fp4": (
        "against the gate / up multiply with SiLU x up evaluated in fp64 on the dequantised operands, with the error bound of the kernel's contract: "
        "every block-scale byte inside the interval of bytes that bound admits for the largest magnitude of its block, every 4-bit code within half "
        "a code step of the fp64 value under the kernel's own block scale plus that bound; every value finite"
    ),
    "attention_prepare": (
        "against the preparation at the kernel's rounding points (the norm's mean and root in fp64): every query element that is not rotated within one "
        "bf16 unit at the larger magnitude, every rotated one inside that unit plus |cos| and |sin| times one unit of its two normalised operands; the gate "
        "rows bitwise; in the cache, the key rows at the positions of the call held as the query rows, their value rows and every other byte of the pool "
        "bitwise; every value finite"
    ),
    "attention_bf16_weights": (
        "against the reference chain before its rounding to bf16 (chain = sum p v over the attended rows under their softmax weights p, spread = sum p |v|, "
        f"both in fp64): every element finite and within {ATTENTION_BAND_COEFFICIENT:g} x [2**{math.log2(ATTENTION_WEIGHT_TERM):g} x (spread + |chain|) + "
        f"{ATTENTION_SPACINGS:g} spacing of bf16 values at |chain|] -- the bound of a stated model of the kernel's arithmetic (every weight one rounding at "
        "eight significant bits, the divisor one of its own, the output one), declared before any run and never taken from a measured distance; the cache "
        "bitwise"
    ),
}


def _tensors(obj) -> tuple:
    if obj is None:
        return ()
    if isinstance(obj, torch.Tensor):
        return (obj,)
    if isinstance(obj, (tuple, list)):
        return tuple(t for o in obj for t in _tensors(o))
    return tuple(obj.tensors())


def _rows(obj, start: int, stop: int):
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj[start:stop]
    if isinstance(obj, (tuple, list)):
        return tuple(_rows(o, start, stop) for o in obj)
    return obj.rows(start, stop)


def bytes_equal(a, b) -> bool:
    ta, tb = _tensors(a), _tensors(b)
    return len(ta) == len(tb) and all(
        x.shape == y.shape and torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)) for x, y in zip(ta, tb)
    )


def bf16_units(out: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """``|out - ref|`` in bf16 units at the larger magnitude of the two (not below 2**-8 of the reference's largest value)."""
    a, r = out.float(), ref.float()
    floor = (r.abs().amax() * 2.0**-8).clamp_min(torch.finfo(torch.float32).tiny)
    exponent = torch.frexp(torch.maximum(torch.maximum(a.abs(), r.abs()), floor)).exponent
    return (a - r).abs() / torch.pow(2.0, exponent.float() - 8)  # not torch.exp2: that one is compiled at first use on the GPU


def rms_ratio(out: torch.Tensor, ref: torch.Tensor) -> float:
    """The rms of ``out - ref`` over the rms of ``ref``, both accumulated in fp64."""
    out, ref = out.double(), ref.double()
    return float((out - ref).square().mean().sqrt() / ref.square().mean().sqrt().clamp_min(1e-12))


def _largest(so_far: float, value: float) -> float:
    """The larger of the two, and NaN as soon as one of them is: Python's ``max(x, nan)`` returns ``x``."""
    if math.isnan(so_far) or math.isnan(value):
        return float("nan")
    return max(so_far, value)


@dataclass
class _Cells:
    """One layer's (sequence, value head) cells over the run: equal output bytes and state elements inside the band."""

    equal: torch.Tensor  # [rows, value heads]
    inside: torch.Tensor  # [slots - 1, value heads]
    calls: int = 0


@dataclass
class BlockStats:
    band: str
    calls: int = 0
    ok: bool = True
    finite: bool = True
    elements: int = 0
    off_by_one: int = 0
    worst_units: float = 0.0
    equal_bytes: int = 0
    total_bytes: int = 0
    worst_call: float = 1.0
    state_error: float = 0.0
    state_rms: float = 0.0
    state_inside: int = 0
    state_elements: int = 0
    within_cap: bool = True
    conv_equal: bool = True
    null_equal: bool = True
    state_equal: bool = True  # of a band without a rule for state: what the call updated, byte for byte
    outside: int = 0  # gate_fp8, add_norm_quant: bytes (elements) that are the code of neither bf16 value around the judged value
    off_nearest: int = 0  # gate_fp8, add_norm_quant: bytes off the nearest value's code although the judged value is beyond the margin
    other_value: int = 0  # add_norm_quant: normed elements that are the other bf16 value around the normalised value (inside the margin)
    not_of_out: int = 0  # add_norm_quant: 8-bit codes that are not those of the normed rows the same call wrote
    blocks: int = 0  # swiglu_fp4: block scales judged, and those of them that are the byte of the fp64 chain
    blocks_equal: int = 0
    worst_share: float = 0.0  # swiglu_fp4: the largest distance of a code's value from the fp64 value, as a share of its bound
    keys_outside: int = 0  # attention_prepare: key elements at the positions of a call outside their band (``outside``: query elements)
    gates_equal: bool = True  # attention_prepare: the gate rows, byte for byte
    values_equal: bool = True  # attention_prepare: the value rows at the positions of a call, byte for byte (``state_equal``: the rest of the pool)
    #: attention_bf16_weights: the chain and the spread of the element that used the largest share of its bound (``worst_share``, for this band the
    #: largest distance of an element from the chain as a share of its bound)
    worst_at: Tuple[float, float] = (0.0, 0.0)
    cells: Dict[int, _Cells] = field(default_factory=dict)
    cell_bytes: Dict[int, Tuple[float, str]] = field(default_factory=dict)  # layer -> (the smallest share of a cell, which cell)
    cell_state: Dict[int, Tuple[float, str]] = field(default_factory=dict)
    batch_calls: int = 0
    batch_equal: int = 0
    notes: List[str] = field(default_factory=list)

    def worst_cell(self, of: Dict[int, Tuple[float, str]]) -> Tuple[float, str]:
        return min(of.values(), default=(1.0, "no cell"))

    def line(self) -> str:
        loud = "" if self.finite else "; NON-FINITE VALUES"
        if self.band == "gdn_step":
            share = self.equal_bytes / max(self.total_bytes, 1)
            inside = self.state_inside / max(self.state_elements, 1)
            (cell_bytes, bytes_at), (cell_state, state_at) = self.worst_cell(self.cell_bytes), self.worst_cell(self.cell_state)
            return (
                f"{share:.3%} of 8-bit bytes equal (worst call {self.worst_call:.3%}; worst cell {cell_bytes:.3%}: {bytes_at}); "
                f"{inside:.4%} of the state within {GDN_STATE_BAND:g} (worst cell {cell_state:.4%}: {state_at}), largest distance {self.state_error:.1e}, "
                f"rms ratio {self.state_rms:.1e} at the worst call (printed, not gated); conv history {'equal' if self.conv_equal else 'DIFFERS'}; "
                f"null slot {'untouched' if self.null_equal else 'WRITTEN'}{loud}"
            )
        if self.band == "gate_fp8":
            other = self.total_bytes - self.equal_bytes - self.outside
            return (
                f"{self.equal_bytes / max(self.total_bytes, 1):.4%} of 8-bit bytes are the code of the nearest bf16 value; {other:,} of {self.total_bytes:,} "
                f"the code of the other one around the gated value, {self.off_nearest:,} of them more than {GATE_MARGIN_ULP32:g} fp32 units from the "
                f"midpoint; {self.outside:,} the code of neither{loud}"
            )
        if self.band == "add_norm_quant":
            loose = self.total_bytes - self.equal_bytes
            return (
                f"residual stream {'equal' if self.state_equal else 'DIFFERS'}; {self.elements:,} normed elements, {self.other_value:,} of them the other "
                f"bf16 value around the normalised value; {self.total_bytes:,} codes and block scales, {loose:,} of them not those of the nearest value; "
                f"{self.not_of_out:,} 8-bit codes not those of the normed rows of their call; {self.off_nearest:,} off the nearest value more than "
                f"{NORM_MARGIN_ULP32:g} fp32 units from the midpoint; {self.outside:,} outside{loud}"
            )
        if self.band == "swiglu_fp4":
            return (
                f"{self.blocks:,} block scales, {self.blocks - self.blocks_equal:,} of them another byte than the fp64 chain's; {self.total_bytes:,} codes, "
                f"{self.total_bytes - self.equal_bytes:,} of them another code than the fp64 chain's; the largest distance of a code from the fp64 value "
                f"{self.worst_share:.0%} of its bound; {self.outside:,} outside their bounds{loud}"
            )
        if self.band == "attention_prepare":
            return (
                f"{self.elements:,} query and key elements, {self.other_value:,} of them not the twin's byte; "
                f"{self.outside:,} query and {self.keys_outside:,} key elements outside their band; gate rows {'equal' if self.gates_equal else 'DIFFER'}; "
                f"value rows {'equal' if self.values_equal else 'DIFFER'}; the rest of the pool {'untouched' if self.state_equal else 'WRITTEN'}{loud}"
            )
        if self.band == "attention_bf16_weights":
            chain, spread = self.worst_at
            return (
                f"{self.elements:,} elements, {self.other_value:,} of them not the reference chain's byte; {self.outside:,} outside their bound; the largest "
                f"distance {self.worst_share:.3f} of its bound, at a chain of {chain:.4g} with a spread of {spread:.4g} (printed, not gated); "
                f"the cache {'untouched' if self.state_equal else 'WRITTEN'}{loud}"
            )
        share = self.off_by_one / max(self.elements, 1)
        state = "" if self.state_equal else "; STATE DIFFERS"
        return f"largest distance {self.worst_units:.2f} bf16 units; {share:.1%} of elements differ by one unit{state}{loud}"


def _smallest(counts: torch.Tensor, total: int, layer: int) -> Tuple[float, str]:
    """The cell with the fewest counted elements of ``counts`` ``[sequences, value heads]``: its share of ``total`` and its name."""
    at = int(counts.argmin())
    sequence, head = divmod(at, counts.shape[1])
    return float(counts.flatten()[at]) / max(total, 1), f"layer {layer}, sequence {sequence}, value head {head}"


class BlockTap:
    """The hook of a checked pass: runs the provider block, then the reference block and -- with more than one sequence -- the
    provider block on one sequence alone (sequence ``step`` modulo the count), each from the same state, and restores the
    provider's results before it returns them.  The reference block reads a bf16 copy of its weight: where the run keeps
    none, ``on_demand`` (``weights.OnDemand``) makes it for that one call; the provider block is always handed the run's
    own weight."""

    def __init__(self, plan, reference, on_demand=None):
        self.plan, self.reference, self.on_demand = plan, reference, on_demand
        self.stats: Dict[str, BlockStats] = {}
        self.rows: Dict[str, object] = {}  # the provider row behind every entry of ``stats``
        self.blocks: Dict[str, object] = {}  # and the block: for a block several rows serve, the part of that call site
        self.step = 0  # set by the checked pass before every step
        self.calls = 0  # the block calls this tap was handed: none where no step of its plan ran under the check

    def _part(self, name, block, operands) -> tuple:
        """``(label, row, block)`` of what serves this call: the block and its row, or -- for a block whose calls several
        rows serve by call site -- the part of the call's site, judged and counted under ``<block> at <call site>``: each
        part has the band, the twin and the batch-invariance statement of its own row."""
        if getattr(block, "of", None) is None:
            return name, self.plan.rows[name], block
        site = call_site(operands)
        row, part = block.of(site)
        return scoped(name, site), row, part

    @staticmethod
    def _state(block, operands):
        return [operands[i] for i in block.state_args]

    @staticmethod
    def _save(holders):  # a holder is a state object, a plain tensor a call updates in place, or nothing (``None``)
        return [tuple(t.clone() for t in _tensors(h)) for h in holders]

    @staticmethod
    def _load(holders, saved):
        for h, s in zip(holders, saved):
            for t, c in zip(_tensors(h), s):
                t.copy_(c)

    def _own(self, holder, sequence: int) -> tuple:
        """Where ``sequence`` lives in each tensor of a state holder: a pool or a cache says so itself (``of``); a plain tensor
        a call updates in place holds one row per row of the step, the rows of a sequence together."""
        if hasattr(holder, "of"):
            return holder.of(sequence)
        tokens = self.plan.tokens
        return (slice(sequence * tokens, (sequence + 1) * tokens),) * len(_tensors(holder))

    def __call__(self, name, layer, block, operands):
        self.calls += 1
        label, row, block = self._part(name, block, operands)
        ref = block.twin(self.reference[name])  # the reference block, or the block's own twin
        independent = type(ref) is not type(block)
        alone = self.plan.sequences > 1
        if not independent and not alone:
            return block(*operands)
        stats = self.stats.setdefault(label, BlockStats(row.band if independent else "same"))
        self.rows[label], self.blocks[label] = row, block
        stats.calls += 1
        holders = self._state(block, operands)
        before = self._save(holders)
        out = block(*operands)
        compared = tuple(t.clone() for t in _tensors(block.compared(out)))
        after = self._save(holders)
        if independent:
            self._load(holders, before)
            if hasattr(ref, "anchor"):  # a reference block that follows a kernel through a step of several tokens: handed what the kernel left
                ref.anchor = after
            ref_out = ref(*(operands if self.on_demand is None else self.on_demand.operands(operands)))  # a copy made now is dropped here
            self._judge(stats, compared, _tensors(ref.compared(ref_out)), after, self._save(holders), layer, ref)
        if alone:
            s = self.step % self.plan.sequences
            first, last = s * self.plan.tokens, (s + 1) * self.plan.tokens  # the rows of sequence s in a step: one per token it is fed
            self._load(holders, before)
            single = [(_rows(o, first, last) if i in block.per_row else o) for i, o in enumerate(operands)]
            alone_out = block(*single)
            same = bytes_equal(tuple(t[first:last] for t in compared), block.compared(alone_out))
            for holder, full, one in zip(holders, after, self._save(holders)):
                at = self._own(holder, s)  # a slot pool keeps sequence s in slot s + 1 (and in a slot per further token), a cache in row s
                same = same and bytes_equal(tuple(t[i] for t, i in zip(full, at)), tuple(t[i] for t, i in zip(one, at)))
            stats.batch_calls += 1
            stats.batch_equal += int(same)
        self._load(holders, after)
        return out

    @staticmethod
    def _judge(stats: BlockStats, out, ref, state, ref_state, layer: int = 0, twin=None) -> None:
        """One call of a provider block against its twin: ``out`` / ``ref`` are the compared outputs of the two, ``state`` /
        ``ref_state`` what each left in the state the call updates, ``twin`` the block ``ref`` came from (a band reads from
        it what it needs beyond the outputs).  The band of the row decides per output."""
        if stats.band == "gate_fp8":  # output 0: the 8-bit codes, against the twin's bracket of the gated value
            if getattr(twin, "bracket", None) is None:  # a table that names this band for a block whose twin is another chain
                raise ValueError(
                    "the band gate_fp8 is judged against the gate at the kernel's rounding points: the block's twin keeps no bracket of the gated value"
                )
            codes, nearest = out[0], ref[0]
            other, margin = twin.bracket
            same = codes == nearest
            stats.equal_bytes += int(same.sum())
            stats.total_bytes += codes.numel()
            stats.outside += int((~same & (codes != other)).sum())
            stats.off_nearest += int((~same & (margin > GATE_MARGIN_ULP32)).sum())
            stats.finite = stats.finite and bool(((codes & 0x7F) != 0x7F).all())  # E4M3 has no infinity; these two codes are its NaN
            stats.state_equal = stats.state_equal and bytes_equal(state, ref_state)
            stats.ok = stats.finite and stats.outside == 0 and stats.off_nearest == 0 and stats.state_equal
            return
        if stats.band == "add_norm_quant":
            _judge_norm(stats, out, ref, twin)
            return
        if stats.band == "swiglu_fp4":
            _judge_swiglu(stats, out, ref, twin)
            stats.state_equal = stats.state_equal and bytes_equal(state, ref_state)
            stats.ok = stats.ok and stats.state_equal
            return
        if stats.band == "attention_prepare":
            _judge_prepare(stats, out, ref, state, ref_state, twin)
            return
        if stats.band == "attention_bf16_weights":
            _judge_attention_core(stats, out, ref, twin)
            stats.state_equal = stats.state_equal and bytes_equal(state, ref_state)  # the call writes nothing: what it reads, byte for byte
            stats.ok = stats.ok and stats.state_equal
            return
        if stats.band == "gdn_step":
            codes, ref_codes = out[0], ref[0]
            (recurrent, conv), (ref_recurrent, ref_conv) = state[0], ref_state[0]
            heads, dim = recurrent.shape[1], recurrent.shape[2]
            same = codes == ref_codes
            equal = int(same.sum())
            stats.equal_bytes += equal
            stats.total_bytes += codes.numel()
            stats.worst_call = min(stats.worst_call, equal / codes.numel())
            # slot 0 is the null slot (model_samples.state): it is in no share and must be left as it was
            live, ref_live = recurrent[1:], ref_recurrent[1:]
            distance = (live - ref_live).abs() / max(1.0, float(ref_recurrent.abs().amax()))
            inside = distance <= GDN_STATE_BAND  # False for a NaN
            stats.state_error = _largest(stats.state_error, float(distance.amax()))
            stats.state_rms = _largest(stats.state_rms, rms_ratio(recurrent, ref_recurrent))
            stats.state_inside += int(inside.sum())
            stats.state_elements += distance.numel()
            stats.conv_equal = stats.conv_equal and bytes_equal(conv, ref_conv)
            stats.null_equal = stats.null_equal and bytes_equal(recurrent[0], ref_recurrent[0])
            stats.within_cap = stats.within_cap and bool((distance <= GDN_STATE_CAP).all())  # element by element: False for a NaN
            stats.finite = (
                stats.finite
                and bool(((codes & 0x7F) != 0x7F).all())  # E4M3 has no infinity; these two codes are its NaN
                and bool(torch.isfinite(recurrent).all())
                and bool(torch.isfinite(conv).all())
            )
            equal_cells, inside_cells = same.view(-1, heads, dim).sum(-1), inside.flatten(2).sum(-1)
            # a step of several tokens per sequence (the twin says how many its last call had): a cell is still one value head
            # of one sequence -- one block of the launch grid -- over the tokens of the step; the output rows are sequence by
            # sequence, the state slots token by token
            tokens = getattr(twin, "tokens", 1)
            if tokens > 1:
                equal_cells, inside_cells = equal_cells.view(-1, tokens, heads).sum(1), inside_cells.view(tokens, -1, heads).sum(0)
            cells = stats.cells.get(layer)
            if cells is None:
                cells = stats.cells[layer] = _Cells(torch.zeros_like(equal_cells), torch.zeros_like(inside_cells))
            cells.equal += equal_cells
            cells.inside += inside_cells
            cells.calls += tokens  # token steps: a call of a step of several tokens is that many
            stats.cell_bytes[layer] = _smallest(cells.equal, cells.calls * dim, layer)
            stats.cell_state[layer] = _smallest(cells.inside, cells.calls * dim * dim, layer)
            stats.ok = (
                stats.finite
                and stats.equal_bytes > GDN_EQUAL_BYTES * stats.total_bytes
                and stats.state_inside > GDN_STATE_SHARE * stats.state_elements
                and stats.worst_cell(stats.cell_bytes)[0] > GDN_CELL_BYTES
                and stats.worst_cell(stats.cell_state)[0] > GDN_CELL_STATE
                and stats.within_cap
                and stats.conv_equal
                and stats.null_equal
            )
            return
        for o, r in zip(out, ref):
            if o.is_floating_point():
                stats.finite = stats.finite and bool(torch.isfinite(o).all())
            if stats.band == "bitwise" or o.dtype != torch.bfloat16:
                stats.ok = stats.ok and bytes_equal(o, r)
                continue
            units = bf16_units(o, r)
            stats.elements += units.numel()
            stats.off_by_one += int((units >= 0.5).sum())
            stats.worst_units = _largest(stats.worst_units, float(units.amax()))
            stats.ok = stats.ok and bool((units <= 1.0).all())  # element by element: False for a NaN
        stats.state_equal = stats.state_equal and bytes_equal(state, ref_state)  # no rule of this band for state: byte for byte
        stats.ok = stats.ok and stats.finite and stats.state_equal


def _judge_norm(stats: BlockStats, out, ref, twin) -> None:
    """One call of the add + RMSNorm + quantise entry against its twin (band ``add_norm_quant``): ``out`` / ``ref`` are the
    compared outputs of the two in the order the twin names (``twin.outputs``); the twin keeps the bf16 bracket of the
    normalised value and the quantisation of the call.  The residual stream the call updated in place is one of the
    outputs, so the state is not judged a second time."""
    from .providers import torch_ref
    from .weights import BLOCK, quantize_fp8

    if getattr(twin, "bracket", None) is None:  # a table that names this band for a block whose twin is another chain
        raise ValueError("the band add_norm_quant is judged against the norm at the kernel's rounding points: the block's twin keeps no bracket")
    names = tuple(twin.outputs)
    if len(out) != len(names) or len(ref) != len(names):
        stats.outside += 1  # the call wrote other outputs than the form of its site names: nothing can be compared
        stats.ok = False
        return
    got, quant = dict(zip(names, out)), twin.quant
    nearest, other, margin = twin.bracket
    loose = margin <= NORM_MARGIN_ULP32  # where the kernel's fp32 value may lie on the other side of the midpoint
    stats.state_equal = stats.state_equal and bytes_equal(got["residual"], dict(zip(names, ref))["residual"])
    if "normed" in got:
        y = got["normed"]
        bits, at_nearest, at_other = torch_ref._bf16_bits(y), torch_ref._bf16_bits(nearest), torch_ref._bf16_bits(other)
        same, second = bits == at_nearest, bits == at_other
        stats.finite = stats.finite and bool(torch.isfinite(y).all())
        stats.elements += y.numel()
        stats.other_value += int((~same & second).sum())
        stats.outside += int((~same & ~second).sum())
        stats.off_nearest += int((~same & second & ~loose).sum())
    if "codes" in got and "scales" not in got:  # 8-bit codes
        codes, of_nearest, of_other = got["codes"], quantize_fp8(nearest, quant.scale).codes, quantize_fp8(other, quant.scale).codes
        same, second = codes == of_nearest, codes == of_other
        stats.finite = stats.finite and bool(((codes & 0x7F) != 0x7F).all())  # E4M3 has no infinity; these two codes are its NaN
        stats.total_bytes += codes.numel()
        stats.equal_bytes += int(same.sum())
        stats.outside += int((~same & ~second).sum())
        stats.off_nearest += int((~same & second & ~loose).sum())
        if "normed" in got:  # the codes are those of the bf16 rows the same launch wrote
            stats.not_of_out += int((codes != quantize_fp8(got["normed"], quant.scale).codes).sum())
    if "scales" in got:  # 4-bit codes with their block scales, row by row
        packed, scales = got["codes"], got["scales"]
        a, b = nearest.float().abs(), other.float().abs()
        firm = torch.where(loose, torch.minimum(a, b), a).unflatten(-1, (-1, BLOCK)).amax(-1)  # the smallest and the largest block
        high = torch.where(loose, torch.maximum(a, b), a).unflatten(-1, (-1, BLOCK)).amax(-1)  # maximum the bracket admits
        low_scale, high_scale = torch_ref.nvfp4_block_scales(firm, quant.scale), torch_ref.nvfp4_block_scales(high, quant.scale)
        of_nearest = torch_ref.nvfp4_block_scales(a.unflatten(-1, (-1, BLOCK)).amax(-1), quant.scale)
        stats.finite = stats.finite and bool(((scales & 0x7F) != 0x7F).all())
        stats.total_bytes += scales.numel()
        stats.equal_bytes += int((scales == of_nearest).sum())
        stats.outside += int(((scales != low_scale) & (scales != high_scale)).sum())
        codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2)  # one nibble per element, the even element first
        low, top = torch_ref.nvfp4_code_bracket(nearest, scales, quant.scale)  # under the block scales the kernel stored
        low_other, top_other = torch_ref.nvfp4_code_bracket(other, scales, quant.scale)
        same, second = (codes == low) | (codes == top), (codes == low_other) | (codes == top_other)
        stats.total_bytes += codes.numel()
        stats.equal_bytes += int(same.sum())
        stats.outside += int((~same & ~second).sum())
        stats.off_nearest += int((~same & second & ~loose).sum())
    stats.ok = stats.finite and stats.outside == 0 and stats.off_nearest == 0 and stats.not_of_out == 0 and stats.state_equal


def _judge_prepare(stats: BlockStats, out, ref, state, ref_state, twin) -> None:
    """One call of the attention preparation entry against its twin (band ``attention_prepare``): ``out`` / ``ref`` are the
    queries and the gates of the two, ``state`` / ``ref_state`` the pool of the layer's cache as each left it; the twin
    keeps the band of every rotated element, the key rows it wrote and the position of the call.  The rows of the pool
    the call's positions name are formed here from host integers -- the position and the tokens per sequence of the
    call, the sequences of the step and the shape of the pool -- and from no table and no accessor of the cache."""
    from .providers import torch_ref

    terms = getattr(twin, "terms", None)
    if terms is None:  # a table that names this band for a block whose twin is another chain
        raise ValueError(
            "the band attention_prepare is judged against the preparation at the kernel's rounding points: the block's twin keeps no band of its rotated elements"
        )
    pools = [tensors[0] for tensors in (*state, *ref_state) if len(tensors) == 1 and tensors[0].dim() == 4]
    if len(out) != 2 or len(ref) != 2 or any(o.shape != r.shape for o, r in zip(out, ref)) or len(pools) != 2 or pools[0].shape != pools[1].shape:
        stats.outside += 1  # other outputs than the queries and the gates, or a cache that is no pool of pages: nothing can be compared
        stats.ok = False
        return

    def held(got, want, band) -> int:
        """The elements of ``got`` outside ``|got - want| <= unit(the larger magnitude) + band``; a value that is not finite is outside."""
        g, w = got.double(), want.double()
        stats.finite = stats.finite and bool(torch.isfinite(g).all())
        stats.elements += g.numel()
        stats.other_value += int((torch_ref._bf16_bits(got) != torch_ref._bf16_bits(want)).sum())
        return int((~((g - w).abs() <= torch_ref.bf16_unit(torch.maximum(g.abs(), w.abs())) + band)).sum())  # element by element: outside for a NaN

    (q, gate), (q_ref, gate_ref) = out, ref
    stats.outside += held(q, q_ref, terms.q_band)
    stats.gates_equal = stats.gates_equal and bytes_equal(gate, gate_ref)
    pool, ref_pool = pools
    tokens, page, width = terms.tokens, int(pool.shape[2]), int(pool.shape[-1]) // 2
    sequences = int(q.shape[0]) // tokens
    pages = int(pool.shape[0]) // sequences  # of one sequence: sequence i owns the pages i * pages .. (i + 1) * pages - 1
    at = [(sequence * pages + (terms.position + token) // page, (terms.position + token) % page) for sequence in range(sequences) for token in range(tokens)]
    index = torch.tensor(at, dtype=torch.long, device=pool.device)  # the row of the pool of every row of the step, in the step's order
    written = pool[index[:, 0], :, index[:, 1]]  # [rows, key / value heads, 2 * head width]
    stats.keys_outside += held(written[..., :width], terms.k, terms.k_band)
    stats.values_equal = stats.values_equal and bytes_equal(written[..., width:], terms.v)
    # the gate rows and the value rows are bit copies of the call's operand, which the twin copies too: a NaN or an infinity
    # there is the twin's byte, so the two comparisons above do not see it
    stats.finite = stats.finite and bool(torch.isfinite(gate).all()) and bool(torch.isfinite(written[..., width:]).all())
    rest = torch.ones(pool.shape[0], page, dtype=torch.bool, device=pool.device)
    rest[index[:, 0], index[:, 1]] = False
    stats.state_equal = stats.state_equal and bytes_equal(pool.transpose(1, 2)[rest], ref_pool.transpose(1, 2)[rest])
    stats.ok = stats.finite and stats.outside == 0 and stats.keys_outside == 0 and stats.gates_equal and stats.values_equal and stats.state_equal


def _judge_attention_core(stats: BlockStats, out, ref, twin) -> None:
    """One call of a library attention kernel that documents no numerics contract against its twin (band
    ``attention_bf16_weights``): ``out`` is the kernel's output and ``ref`` the reference chain's bytes, ``[rows, heads *
    D]`` bf16 each; the twin keeps the chain of every element before its rounding to bf16 and the weighted magnitude of
    the values it attends to, both in fp64.  The bound, the distance and the comparison are formed in fp64, element by
    element; the spacing is the chain's own.  The largest share of its bound an element used is kept with the chain and
    the spread of that element: printed, not gated."""
    from .providers import torch_ref

    if getattr(twin, "bound", None) is None:  # a table that names this band for a block whose twin is another chain
        raise ValueError(
            "the band attention_bf16_weights is judged against the reference chain with the operand terms of its bound: the block's twin keeps no such terms"
        )
    chain, spread = twin.bound
    if len(out) != 1 or len(ref) != 1 or out[0].dtype != torch.bfloat16 or out[0].shape != ref[0].shape or out[0].shape != chain.shape:
        stats.outside += 1  # other outputs than the one bf16 tensor of the chain's shape: nothing can be compared
        stats.ok = False
        return
    got = out[0].double()
    spacing = torch_ref.bf16_unit(chain.abs().clamp_min(torch.finfo(torch.float32).tiny))  # of two neighbouring bf16 values at |chain|
    bound = ATTENTION_BAND_COEFFICIENT * (ATTENTION_WEIGHT_TERM * (spread + chain.abs()) + ATTENTION_SPACINGS * spacing)
    distance = (got - chain).abs()
    stats.finite = stats.finite and bool(torch.isfinite(got).all())
    stats.elements += got.numel()
    stats.other_value += int((torch_ref._bf16_bits(out[0]) != torch_ref._bf16_bits(ref[0])).sum())
    inside = (distance <= bound) & torch.isfinite(bound)  # element by element: a NaN is outside, and so is an element whose chain is not finite
    stats.outside += int((~inside).sum())
    share = (distance / bound).flatten()
    at = int(torch.where(torch.isnan(share), torch.full_like(share, float("inf")), share).argmax())  # an element that is no number comes first
    worst = float(share[at])
    if math.isnan(worst) or worst > stats.worst_share:  # (a share that is no number stays, whatever follows)
        stats.worst_share, stats.worst_at = worst, (float(chain.flatten()[at]), float(spread.flatten()[at]))
    stats.ok = stats.finite and stats.outside == 0


def _judge_swiglu(stats: BlockStats, out, ref, twin) -> None:
    """One call of the fused 4-bit gate / up entry against its twin (band ``swiglu_fp4``): ``out`` are the packed codes and
    the block scales the kernel wrote, row by row; ``ref`` the same of the fp64 chain at the kernel's rounding points; the
    twin keeps the fp64 value of every element with its error bound and the quantisation of the call."""
    from .providers import torch_ref
    from .weights import BLOCK, E2M1_MAX, _e2m1_values, _e4m3_values

    if getattr(twin, "bound", None) is None:  # a table that names this band for a block whose twin is another chain
        raise ValueError("the band swiglu_fp4 is judged against the gate / up step with its error bound: the block's twin keeps no bound")
    if len(out) != 2 or len(ref) != 2 or any(o.shape != r.shape for o, r in zip(out, ref)):
        stats.outside += 1  # the call wrote other outputs than packed codes with their block scales: nothing can be compared
        stats.ok = False
        return
    (packed, scales), (ref_packed, ref_scales) = out, ref
    value, error = twin.bound
    scale = torch_ref.nvfp4_global_scale(twin.quant.scale, value.device).double()  # the one element the kernel reads
    largest, spread = (t.unflatten(-1, (-1, BLOCK)).amax(-1) for t in (value.abs(), error))
    slack = torch_ref.SWIGLU_SCALE_SLACK
    low = torch_ref.e4m3_nearest(scale * (largest - spread).clamp(min=0) / E2M1_MAX * (1.0 - slack))
    high = torch_ref.e4m3_nearest(scale * (largest + spread) / E2M1_MAX * (1.0 + slack))
    stats.finite = stats.finite and bool(((scales & 0x7F) != 0x7F).all())  # E4M3 has no infinity; these two codes are its NaN
    stats.blocks += scales.numel()
    stats.blocks_equal += int((scales == ref_scales).sum())
    stats.outside += int(((scales < low) | (scales > high)).sum())
    step = (_e4m3_values(scales, torch.float64) / scale).repeat_interleave(BLOCK, dim=-1)  # of the block scale the kernel stored
    nibbles = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2)  # one code per element, the even element first
    codes = _e2m1_values(nibbles, torch.float64)
    half = torch.where(codes.abs() < 2, 0.25, torch.where(codes.abs() < 4, 0.5, 1.0))  # half the step between two codes there
    bound = half * step + (value.abs() + error - E2M1_MAX * step).clamp(min=0) + error
    distance = (codes * step - value).abs()
    stats.total_bytes += nibbles.numel()
    stats.equal_bytes += int((nibbles == torch.stack((ref_packed & 0xF, ref_packed >> 4), dim=-1).flatten(-2)).sum())
    stats.outside += int((~(distance <= bound)).sum())  # element by element: outside for a NaN
    stats.worst_share = _largest(stats.worst_share, float((distance / bound).amax()))
    stats.ok = stats.finite and stats.outside == 0


def token_check(logits: torch.Tensor, ref_logits: torch.Tensor, ids: torch.Tensor, ref_ids: torch.Tensor, band_units: Optional[float]) -> Tuple[bool, str]:
    """The end-to-end check of a run's greedy tokens against the reference stack's: ``(held, the text of its line)``.

    ``band_units`` is the declared logits band in bf16 units of the largest reference logit of the run.  The check holds when
    every logit is finite, the largest distance between the two stacks' logits is inside the band, and the token equals the
    reference's at every step whose reference margin exceeds ``LOGITS_FACTOR`` times the band.

    ``band_units=None`` is a model whose family declares no band: the line is then a reported one.  It holds when every
    logit of both stacks is finite; the distance is printed and judged against nothing."""
    a, r = logits.float(), ref_logits.float()
    finite = bool(torch.isfinite(a).all()) and bool(torch.isfinite(r).all())
    top = float(r.abs().amax()) if finite else float("nan")
    unit = 2.0 ** (math.frexp(top)[1] - 8) if top > 0.0 else float("nan")  # False for a NaN
    distance = float((a - r).abs().amax())
    measured = f"largest logit distance {distance:.4g} = {distance / unit:.2f} bf16 units of the largest logit"
    if band_units is None:
        return finite, measured + "  [no band is declared for this model: reported, gated on finite logits]" + ("" if finite else "; NON-FINITE LOGITS")
    band = band_units * unit
    decidable = margin(ref_logits) > LOGITS_FACTOR * band  # nowhere when the band is not a number
    wrong = int((decidable & ~(ids == ref_ids)).sum())
    count, near = int(decidable.sum()), int((~decidable).sum())
    held = finite and distance <= band and wrong == 0
    text = (
        f"{count - wrong} of {count} decidable greedy tokens equal to the reference; {near} near-ties (margin not above {LOGITS_FACTOR:g} x the band); "
        f"{measured}  [band: {band_units:g} units = {band:.4g}, declared]"
    )
    return held, text + ("" if finite else "; NON-FINITE LOGITS")


@dataclass
class Checks:
    lines: List[str]
    ok: bool
    judged: int = 0  # provider blocks compared with the reference; 0: every block of the run was the reference chain itself
    kernels: str = ""  # the kernel check, when it was asked for: PASS | FAIL | NOT OBSERVED
    unobserved: str = ""  # why the kernels were not observed
    on_demand: Optional[OnDemand] = None  # what the reference's copies of the weights the run keeps no copy of cost
    positions: Tuple[int, ...] = ()  # of a run with judged positions: the steps at which its blocks were held against the reference

    def text(self) -> str:
        return "\n".join(self.lines)


def _verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


#: ``run_checks(kernels=DEFERRED)``: the kernel check of this run is made later, by its caller, behind a traced request that has
#: not run yet (``kernel_check``); ``Checks.kernels`` then holds this word and the caller prints no verdict before that check
DEFERRED = "DEFERRED"


def positions_text(positions: Sequence[int]) -> str:
    """Step positions as a run prints them: in ascending order, three or more consecutive ones as ``first-last``
    (``0-15, 63, 64, 312, 313``)."""
    runs: List[List[int]] = []
    for position in sorted(set(positions)):
        if runs and position == runs[-1][-1] + 1:
            runs[-1].append(position)
        else:
            runs.append([position])
    return ", ".join(f"{run[0]}-{run[-1]}" if len(run) > 2 else ", ".join(str(position) for position in run) for run in runs)


def changed_weights(weights) -> List[str]:
    """The weights a load keeps on their stored format alone (``alone`` of a family's weights: ``(block, layer, weight)``)
    whose stored bytes are no longer those of the load, each named by its block and layer: the loader's checksum of the
    weight against the same checksum taken now (``weights.Dequantized.intact``)."""
    return [block if layer is None else f"{block} of layer {layer}" for block, layer, weight in getattr(weights, "alone", ()) if not weight.intact()]


def stored_weights_line(count: int, before: List[str], after: List[str]) -> str:
    """The line of the checks about the ``count`` weights the run keeps on their stored format alone: whether each still
    holds the bytes of its load -- ``before`` and ``after`` name those that did not before the first checked pass and
    after the last reference pass.  The reference's copy of such a weight is made from these bytes, so a change fails the
    run and says what the block lines of the named weights are worth."""
    changed = list(dict.fromkeys((*before, *after)))
    text = f"  stored weights    {count - len(changed)} of {count} weights this run keeps on their stored format alone hold the bytes of their load"
    if not changed:
        return f"{text}, before the checks and after them: the reference's copies were made from what was loaded    PASS"
    when = [f"{name} ({'before the checks' if name in before else 'during the checks'})" for name in changed]
    return (
        f"{text}; written to since: {', '.join(when)}.  The reference's copy of such a weight was made from the changed bytes: "
        "its block was compared with a reference that moved with it    FAIL"
    )


def on_demand_line(cost: OnDemand, blocks=()) -> str:
    """The line of the checks that says what the reference's own copies cost (``weights.OnDemand``): how many were made of
    the weights of ``blocks`` -- those the run keeps on their stored format alone -- the largest one, which is what is held
    at a time, their bytes in all and the seconds they took.  It carries no verdict: it is a cost, in GiB (2^30 bytes)."""
    kept = ", ".join(blocks) or "some weights"
    return (
        f"  on demand         the reference read {cost.copies:,} bf16 copies of weights this run keeps on their stored format alone ({kept}): "
        f"each made for one call and dropped after it; the largest {cost.largest / GIB:.2f} GiB, {cost.bytes / GIB:.1f} GiB in all, "
        f"{cost.seconds:.1f} s of the checks (orientation only)"
    )


def kernel_check(session: Session) -> Tuple[str, str, List[str]]:
    """The kernel check on what the session's traced pass kept: ``(PASS | FAIL | NOT OBSERVED, why not observed, lines)``.  A
    session whose traced pass kept no launch records (it was not asked to) has observed nothing."""
    traced = session.traced
    if traced is None:
        reason = "the traced pass kept no launch records"
        return census.NOT_OBSERVED, reason, [f"  kernels           not observed: {reason}    {census.NOT_OBSERVED}"]
    status, sentences = traced.judge()
    lines = [f"  kernels           {traced.summary()}    {status}"]
    lines += [f"    kernels: {sentence}" for sentence in sentences]
    lines += [f"    {detail}" for detail in traced.details()]
    return status, traced.unobserved if status == census.NOT_OBSERVED else "", lines


def _block_lines(tap: BlockTap, precision: str, what: str = "") -> Tuple[bool, int, List[str]]:
    """The ``blocks`` line of a checked pass and one line per judged block: ``(held, provider blocks judged, lines)``.
    ``what`` names the kind of step where a run has two (``"the prompt's decode steps: "``)."""
    if what and not tap.calls:  # a run of two kinds of step, and no step of this kind ran under the check: nothing was judged
        return True, 0, [f"  blocks            {what}no step of this kind was judged    REPORTED"]
    judged = {n: s for n, s in tap.stats.items() if s.band != "same"}
    blocks_ok = all(s.ok for s in judged.values())
    inside = sum(s.ok for s in judged.values())
    if judged:
        lines = [
            f"  blocks            {what}{inside} of {len(judged)} provider blocks inside their bands against the {precision}-accumulated "
            f"reference, teacher-forced    {_verdict(blocks_ok)}"
        ]
    else:  # a line that compares nothing cannot fail: it is not printed as a pass
        lines = [f"  blocks            {what}no provider block to judge: every block of this run is the reference chain itself    REPORTED"]
    for name, s in judged.items():
        lines.append(f"    {name:<22} {s.calls:>4} calls  {s.line()}  [band: {s.band}]  {_verdict(s.ok)}")
    for band in dict.fromkeys(s.band for s in judged.values()):  # each band's text once, on a line of its own
        lines.append(f"    band {band}: {BANDS[band]}")
    return blocks_ok, len(judged), lines


def _batch_lines(tap: BlockTap, sequences: int, facts, what: str = "") -> Tuple[bool, List[str]]:
    """The ``batch invariance`` line of a checked pass of several sequences, and what an entry states beside it: ``(held,
    lines)``.  ``what`` names the kind of step where a run has two."""
    if what and not tap.calls:  # as for the blocks: no step of this kind ran under the check
        return True, [f"  batch invariance  {what}no step of this kind was judged    REPORTED"]
    cls = f"cc{facts.cc[0]}.{facts.cc[1]}"
    batch_ok, differs, gated = True, [], []
    for name, s in tap.stats.items():
        is_gate = cls in tap.rows[name].batch_invariant
        if is_gate:
            gated.append(f"{name} {'equal' if s.batch_equal == s.batch_calls else 'DIFFERS'} (gated)")
            batch_ok = batch_ok and s.batch_equal == s.batch_calls
        if s.batch_equal != s.batch_calls:
            differs.append(f"{name} ({s.batch_calls - s.batch_equal} of {s.batch_calls} calls)")
    gate_text = "; ".join(gated) if gated else f"no block is gated on {facts.class_text}"
    rest = f"{len(tap.stats) - len(differs)} of {len(tap.stats)} blocks equal" + (f"; differs: {', '.join(differs)}" if differs else "")
    verdict = _verdict(batch_ok) if gated else "REPORTED"
    rotation = f"one sequence alone vs among {sequences} (step t: sequence t mod {sequences})"
    lines = [f"  batch invariance  {what}{rotation}: {gate_text}; {rest}    {verdict}"]
    for name in tap.stats:  # what an entry itself states about one sequence alone and among several: printed beside the line
        note = getattr(tap.blocks[name], "batch_note", "")
        if note:
            lines.append(f"    {name}: {note}")
    return batch_ok, lines


def run_checks(
    session: Session,
    prompt_tokens: int,
    max_steps: Optional[int] = None,
    second_reference: bool = False,
    kernels: Optional[object] = None,  # True | False | DEFERRED; None: the session's own decision
    max_prompt_steps: Optional[int] = None,
    positions: Optional[Sequence[int]] = None,
) -> Checks:
    """Replay, blocks, tokens and batch invariance for the run ``session`` has just recorded: its recorded steps, or the first
    ``max_steps`` of them (a long run of a deep model is checked on its first steps).  Leaves the session reset.

    ``kernels`` adds the kernel check on the traced pass the session made before this call (``Session.trace`` with launch
    records): a failure fails the run; kernels that could not be observed are reported, and ``Checks.unobserved`` says why.
    Left out, it is the session's own decision (``Session.check_kernels``): a session of a stack that declares a target
    kernel set is checked without being asked, so no caller can print a verdict for such a run that never looked at its
    kernels.  For such a session ``kernels=False`` is refused.  ``kernels=DEFERRED`` is for a caller that traces after
    these checks (the performance mode: no profiler pass comes before its last timed request): no kernel line is printed
    here, ``Checks.kernels`` says ``DEFERRED``, and the caller makes the check itself behind its traced request
    (``kernel_check``) before it prints a verdict.

    ``positions`` are judged positions: steps of the recorded run, counted from 0, at which the blocks are held against
    the reference.  The run is then replayed up to the last of them -- past ``max_steps``, which stays the bound of the
    reference stack's own pass (``tokens``) -- with the block check and the batch-invariance check on at the judged
    positions alone and off at every other step; ``replay`` holds the ids and logits of every replayed step.  The first
    line of the checks then names the positions, and ``Checks.positions`` holds them: what was judged, for a caller that
    says so.  A position the run did not record is a ``ValueError``, never left out; so is a session built with
    ``draft_tokens=``.

    ``second_reference`` runs the reference stack once more with the other accumulation precision (IEEE fp32 where the
    reference accumulates in fp64) and prints, beside the token check, how far that second reference is from the first and
    on how many greedy tokens the two agree.  It is a scale, printed and not gated: on a trained checkpoint with 4-bit
    activations two stacks that differ only in rounding end up tens of bf16 units apart (see the family's logits band).

    **A session built with ``draft_tokens=``** has two kinds of step, and its verify steps began on the state its prompt
    left: the whole recorded run is replayed from a zero state -- the prompt's decode steps (all of them, from the kept
    prompt), then the recorded verify steps with their kept counts.  The prompt's decode steps run under the block check
    of the decode plan (the first ``max_prompt_steps`` of them judged, the rest replayed; default: all), the verify steps
    under the block check of the verify plan; the reference sessions are built with the session's ``draft_tokens`` and
    replay the same run; ``replay`` and ``tokens`` cover the recorded prompt steps and the verify steps; the lines
    ``blocks`` and ``batch invariance`` are printed once per kind of step.  ``prompt_tokens`` is not read for such a
    session: its prompt is the one it kept.

    **With a drafter** the replay runs the drafter's passes again -- over the prompt, and in front of every verify step,
    which is fed the recorded window -- and ``replay`` holds the drafter's proposals of every recorded step beside the
    ids and logits.  The reference sessions have no drafter: they are fed the recorded windows.  Under the kernel check
    stands what a run can say about the kernels of the passes: nothing (``DRAFTER_KERNELS``)."""
    if kernels is None:
        kernels = session.check_kernels
    elif not kernels and session.stack in getattr(session.model, "target_stacks", ()):
        raise ValueError(f"stack {session.stack} declares a target kernel set: its kernel check cannot be switched off")
    if positions is not None and session.draft_tokens is not None:
        raise ValueError("judged positions are steps of a decode run; a session built with draft_tokens= has two kinds of step")
    session._closed("run_checks")
    main = session.record
    steps = last = min(main.count, main.inputs.shape[0])  # ``steps``: of the reference stack's own pass; ``last``: the steps replayed here
    if max_steps is not None:
        steps = min(steps, max_steps)
    judged_at = None if positions is None else frozenset(positions)
    if judged_at and not 0 <= min(judged_at) <= max(judged_at) < last:  # a position that was asked for is judged, or the run says no
        raise ValueError(f"judged positions {positions_text(judged_at)}: the run recorded {last} steps, the positions 0 to {last - 1}")
    last = steps if judged_at is None else max(steps, max(judged_at, default=-1) + 1)
    inputs, logits, ids = main.inputs[:last].clone(), main.logits[:last].clone(), main.outputs[:last].clone()
    accepted = list(main.accepted[:last])  # per step the tokens behind the first that the run kept: 0 for a decode step
    proposals = None if main.proposals is None else main.proposals[:steps].clone()  # of a session with a drafter: what it proposed
    # a session built with draft_tokens=: the prompt its verify steps stand on, whole, and what it recorded of the prompt's steps
    drafted = session.draft_tokens is not None
    prompt = session.prompt_record.prompt if drafted else None
    fed = 0 if prompt is None else int(prompt.shape[1])
    judged_prompt = fed if max_prompt_steps is None else min(fed, max_prompt_steps)
    recorded = min(session.prompt_record.count, session.prompt_record.inputs.shape[0]) if drafted else 0
    prompt_logits = session.prompt_record.logits[:recorded].clone() if drafted else None
    prompt_ids = session.prompt_record.outputs[:recorded].clone() if drafted else None
    kind = {"draft_tokens": session.draft_tokens} if drafted else {"tokens": session.tokens}  # the reference runs the same kinds of step
    facts = session.plan.facts
    # the weights this run keeps on their stored format alone: the reference's copies are made from their stored bytes, so
    # those must be the bytes of the load -- now, after the run's own steps, and again after the last reference pass
    alone = len(getattr(session.weights, "alone", ()))
    changed_before = changed_weights(session.weights)
    cost = OnDemand()  # the reference's bf16 copies of these weights: one per call
    reference = Session(
        session.model,
        session.text,
        session.sequences,
        stack="reference",
        device=session.device,
        reference_precision=session.reference_precision,
        capacity=session.capacity,
        share=session,
        record_steps=max(steps, recorded, 1),
        on_demand=cost,
        **kind,
    )
    reference.load()

    def replayed(other: Session) -> Tuple[torch.Tensor, torch.Tensor]:
        """The recorded run on another session, from a fresh state of its own: ``(logits, ids)`` of what was recorded."""
        other.reset(fresh=True)
        for t in range(fed):
            other._feed(prompt[:, t].contiguous())
        for t in range(steps):
            other._advance(inputs[t], accepted[t])
        if not drafted:
            return other.record.logits[:steps], other.record.outputs[:steps]
        return (
            torch.cat([other.prompt_record.logits[:recorded].flatten(0, 1), other.record.logits[:steps].flatten(0, 1)]),
            torch.cat([other.prompt_record.outputs[:recorded].flatten(), other.record.outputs[:steps].flatten()]),
        )

    tap = BlockTap(session.plan, reference.plan, cost)
    prompt_tap = BlockTap(session.prefill_plan, reference.prefill_plan, cost) if drafted else None
    session.reset(fresh=True)
    try:
        for t in range(fed):  # the prompt's decode steps, on the decode plan: the first of them under its block check
            session.tap = prompt_tap if t < judged_prompt else None
            prompt_tap.step = t
            session._feed(prompt[:, t].contiguous())
        for t in range(last):  # every step replayed; the blocks judged at every one, or at the judged positions alone
            session.tap = tap if judged_at is None or t in judged_at else None
            tap.step = t
            session._advance(inputs[t], accepted[t])
    finally:
        session.tap = None
    replay_ok = bytes_equal(session.record.logits[:last], logits) and bytes_equal(session.record.outputs[:last], ids)
    logits, ids = logits[:steps], ids[:steps]  # the reference stack's own pass stays on the first ``steps``
    if proposals is not None:  # ... and what the drafter proposed in front of every step
        replay_ok = replay_ok and bytes_equal(session.record.proposals[:steps], proposals)
    if drafted:  # ... and the prompt's steps, which the verify steps stand on
        again = session.prompt_record
        replay_ok = replay_ok and bytes_equal(again.logits[:recorded], prompt_logits) and bytes_equal(again.outputs[:recorded], prompt_ids)
        logits, ids = torch.cat([prompt_logits.flatten(0, 1), logits.flatten(0, 1)]), torch.cat([prompt_ids.flatten(), ids.flatten()])

    band_units = session.model.logits_band_units  # declared by the family, in its configs.py; None: the token line is reported
    band_units = None if band_units is None else float(band_units)
    ref_logits, ref_ids = replayed(reference)
    tokens_ok, tokens_text = token_check(logits, ref_logits, ids, ref_ids, band_units)
    tokens_text = f"{int((ids == ref_ids).sum())} of {ids.numel()} greedy tokens equal to the reference stack's; " + tokens_text
    if second_reference:
        other = "ieee-fp32" if session.reference_precision == "fp64" else "fp64"
        second = Session(
            session.model,
            session.text,
            session.sequences,
            stack="reference",
            device=session.device,
            reference_precision=other,
            capacity=session.capacity,
            share=session,
            record_steps=max(steps, recorded, 1),
            on_demand=cost,
            **kind,
        )
        matmul = torch.backends.cuda.matmul
        kept = getattr(matmul, "fp32_precision", None)  # a session with IEEE fp32 accumulation sets it for the process
        try:
            second.load()
            second_logits, second_ids = replayed(second)
        finally:
            if kept is not None:
                matmul.fp32_precision = kept
        _, scale_text = token_check(second_logits, ref_logits, second_ids, ref_ids, band_units)
        distance = scale_text.split("largest logit distance ", 1)[1].split("  [", 1)[0]
        tokens_text += (
            f"; for scale, the reference stack with {other} accumulation is {distance} from the one with {session.reference_precision} "
            f"and agrees with it on {int((second_ids == ref_ids).sum())} of {ids.numel()} greedy tokens (printed, not gated)"
        )

    changed_after = changed_weights(session.weights)  # after the last pass that read a copy made on demand
    stored_ok = not changed_before and not changed_after
    lines = [
        f"checks ({session.sequences} sequence{'s' if session.sequences > 1 else ''}, {'the first ' if steps < main.count else ''}{steps} steps: "
        f"{min(prompt_tokens, steps)} prompt + {max(steps - prompt_tokens, 0)} decode)"
    ]
    if judged_at is not None:  # judged positions: what was replayed, where the blocks were judged, what the token line covers
        lines = [
            f"checks ({session.sequences} sequence{'s' if session.sequences > 1 else ''}, {last} steps replayed: {min(prompt_tokens, last)} prompt + "
            f"{max(last - prompt_tokens, 0)} decode; blocks judged at positions {positions_text(judged_at)}; tokens on the first {steps} steps)"
        ]
    if session.tokens > 1:  # the steps of this session are verify steps: no prompt goes through them one token at a time
        lines = [
            f"checks ({session.sequences} sequence{'s' if session.sequences > 1 else ''} x {session.tokens} tokens per sequence, "
            f"{'the first ' if steps < main.count else ''}{steps} verify steps; tokens kept behind the first: {', '.join(str(count) for count in accepted)})"
        ]
    # the kinds of step of the run, each with the tap of its plan: one, or -- behind a prompt -- the prompt's decode steps first
    taps = [("", tap)]
    if drafted:
        some = f", the first {judged_prompt} of them judged" if judged_prompt < fed else ""
        lines[0] = lines[0][:-1] + f"; behind a prompt of {fed} decode steps{some})"
        taps = [("the prompt's decode steps: ", prompt_tap), ("the verify steps: ", tap)]
    what = "ids and logits" if proposals is None else "ids, logits and the drafter's proposals"
    lines.append(f"  replay            {what} byte-identical on a second run from a fresh state    {_verdict(replay_ok)}")
    blocks_ok, judged = True, 0
    for what, each in taps:
        held, count, block_lines = _block_lines(each, session.reference_precision, what)
        blocks_ok, judged, lines = blocks_ok and held, judged + count, lines + block_lines
    lines.append(f"  tokens            {tokens_text}    {'REPORTED' if band_units is None and tokens_ok else _verdict(tokens_ok)}")
    batch_ok = True
    if session.sequences > 1:
        for what, each in taps:
            held, batch_lines = _batch_lines(each, session.sequences, facts, what)
            batch_ok, lines = batch_ok and held, lines + batch_lines
    else:
        lines.append("  batch invariance  not applicable at 1 sequence")
    if alone:  # said only where the run keeps a weight without its bf16 copy: whether the reference's copies were made from the load's bytes
        lines.append(stored_weights_line(alone, changed_before, changed_after))
    if cost.copies:  # what the checks cost beyond the run: said only where a copy was made on demand
        lines.append(on_demand_line(cost, getattr(session.weights, "uncopied", ())))
    status, unobserved = "", ""
    if kernels is DEFERRED:  # the caller's, behind its traced request
        status = DEFERRED
    elif kernels:
        status, unobserved, kernel_lines = kernel_check(session)
        lines += kernel_lines
        if drafted:  # the traced window holds the verify steps alone: what this check did not look at is said beside its verdict
            lines.append(
                f"    the prompt's decode steps: {fed} replayed in front of the traced window, their kernels not observed (the check is of the verify steps)"
            )
        if session.drafter is not None:
            lines.append(f"    {DRAFTER_KERNELS}")
    session.reset(fresh=True)
    ok = replay_ok and blocks_ok and tokens_ok and batch_ok and stored_ok and status != census.FAIL
    return Checks(lines, ok, judged, status, unobserved, cost, tuple(sorted(judged_at or ())))


@dataclass
class Stream:
    """The token stream of a run of decode steps, kept for a comparison: per step the ids fed, the logits and the greedy
    ids of every sequence."""

    inputs: torch.Tensor  # [steps, sequences] int32
    logits: torch.Tensor  # [steps, sequences, vocabulary] bf16
    outputs: torch.Tensor  # [steps, sequences] int32

    @classmethod
    def of(cls, decode: Session, max_steps: Optional[int] = None) -> "Stream":
        """What ``decode`` -- a session of one token per sequence -- has recorded of its run so far: copies, so that the
        session can go on (its own checks reset it)."""
        if decode.tokens != 1:
            raise ValueError("a token stream is the record of decode steps: of a session of one token per sequence")
        record = decode.record
        steps = min(record.count, record.inputs.shape[0], *(() if max_steps is None else (max_steps,)))
        return cls(record.inputs[:steps].clone(), record.logits[:steps].clone(), record.outputs[:steps].clone())

    @property
    def steps(self) -> int:
        return int(self.inputs.shape[0])


def accept_schedule(tokens: int, steps: int) -> List[Tuple[int, int]]:
    """The verify steps that cover a token stream of ``steps`` decode steps at ``tokens`` tokens per sequence, as ``(first
    token of the step in the stream, tokens kept behind the first)``: the kept counts go 0, 1, .., tokens - 1 and round
    again, each step starting where the kept tokens of the one before it end, for as long as a whole step fits.
    ``ValueError`` for a stream too short to keep every count once."""
    schedule, start = [], 0
    while start + tokens <= steps:
        kept = len(schedule) % tokens
        schedule.append((start, kept))
        start += kept + 1
    if len(schedule) < tokens:
        need = tokens * (tokens - 1) // 2 + tokens
        raise ValueError(
            f"a comparison that keeps every count of tokens from 0 to {tokens - 1} once needs a stream of {need} decode steps at {tokens} tokens "
            f"per sequence; this one has {steps}"
        )
    return schedule


#: the blocks in front of a model's first mixer: where each is the same row of the table in two plans and that row states that
#: a row alone is the row in a batch on the GPU class (``batch_invariant``), the two plans feed that mixer the same bytes
FRONT_BLOCKS = ("embedding", "add_norm_quantize", "projection.fp8.in")


def _first_layer_fed_alike(session: Session, decode: Session) -> bool:
    """Whether the first layer's mixer is fed the same bytes by both sessions by the table's own statements: every block in
    front of it is the same provider row in both plans, and -- unless it is the embedding lookup -- a row that gates batch
    invariance on this GPU class (its bytes do not move with the row count)."""
    cc = session.plan.facts.cc
    cls = f"cc{cc[0]}.{cc[1]}"
    for name in FRONT_BLOCKS:
        mine, theirs = session.plan.rows.get(name), decode.plan.rows.get(name)
        if mine is None or theirs is None or mine.key != theirs.key or mine.fallback != theirs.fallback:
            return False
        if name != FRONT_BLOCKS[0] and not (cls in mine.batch_invariant and cls in theirs.batch_invariant):
            return False
    return True


class _KeptState:
    """What a run of verify steps stands on against the decode steps of the kept tokens, compared after every step
    (``DecodeState.standing`` of both): the rule of the line ``kept state``.  A linear-attention layer whose three history
    taps were the same bytes at every comparison was fed the same rows by both runs, and its recurrent state is gated by
    the state rule of band ``gdn_step``; a layer whose taps differ is printed, and so are the attention layers' caches."""

    def __init__(self, layer_kinds):
        self.linear = [i for i, kind in enumerate(layer_kinds) if kind == "linear_attention"]
        self.same_rows = {i: True for i in self.linear}  # the layers whose history taps were byte-equal at every comparison so far
        # per layer and comparison: (elements inside the band, elements, largest distance, within the cap, finite, kept)
        self.compared = {i: [] for i in self.linear}
        self.cache_equal, self.cache_units = {}, 0.0
        self.equal = {}  # layer -> byte-equal at every comparison so far, state and cache alike

    def add(self, mine, theirs, kept: int) -> None:
        """One comparison: ``mine`` / ``theirs`` are ``standing()`` of the two runs, ``kept`` the count the step kept."""
        for i, (own, other) in enumerate(zip(mine, theirs)):
            equal = bytes_equal(own, other)
            self.equal[i] = self.equal.get(i, True) and equal
            if i in self.compared:
                (state, taps), (ref_state, ref_taps) = own, other
                self.same_rows[i] = self.same_rows[i] and bytes_equal(taps, ref_taps)
                distance = (state - ref_state).abs() / max(1.0, float(ref_state.abs().amax()))
                inside = distance <= GDN_STATE_BAND  # False for a NaN
                finite = bool(torch.isfinite(state).all())
                self.compared[i].append((int(inside.sum()), distance.numel(), float(distance.amax()), bool((distance <= GDN_STATE_CAP).all()), finite, kept))
            else:
                self.cache_equal[i] = self.cache_equal.get(i, True) and equal
                if not equal:  # caches of two lengths: the two runs do not stand at the same position
                    units = [float(bf16_units(a, b).amax()) if a.shape == b.shape else float("inf") for a, b in zip(own, other)]
                    self.cache_units = _largest(self.cache_units, max(units))

    def line(self, first_layer_fed_alike: bool) -> Tuple[bool, str, int, str]:
        """``(held, the text of the line, linear-attention layers gated, its verdict)``.  ``first_layer_fed_alike``: the
        table itself says that both runs feed the first layer the same bytes -- other taps there are no rounding."""
        linear, compared, same_rows = self.linear, self.compared, self.same_rows

        def worst(layers_of) -> Tuple[float, str]:
            found = [(largest, f"layer {i}, after keeping {kept}") for i in layers_of for _, _, largest, _, _, kept in compared[i]]
            return max(found, key=lambda item: (math.isnan(item[0]), item[0]), default=(0.0, "no layer"))

        gated = [i for i in linear if same_rows[i]]
        loose = [i for i in linear if not same_rows[i]]
        # the first layer, where the table itself says that both runs feed it the same bytes: other taps there are no rounding
        astray = bool(linear) and linear[0] == 0 and not same_rows[0] and first_layer_fed_alike
        inside = sum(count for i in gated for count, *_ in compared[i])
        elements = sum(total for i in gated for _, total, *_ in compared[i])
        # with no layer gated nothing is judged here: the line is then REPORTED and fails nothing
        state_ok = not gated or (all(within and finite for i in gated for _, _, _, within, finite, _ in compared[i]) and inside > GDN_STATE_SHARE * elements)
        state_ok = state_ok and not astray
        text = (
            f"after each step, the state of the kept tokens against the state of that many decode steps: {len(gated)} of {len(linear)} linear-attention "
            f"layers were fed the same rows by both runs (their history taps byte-equal at every comparison)"
        )
        if gated:
            text += f": recurrent state {inside / max(elements, 1):.4%} within {GDN_STATE_BAND:g}, largest distance {worst(gated)[0]:.1e} ({worst(gated)[1]})"
        if loose:
            text += f"; in the {len(loose)} fed other rows, largest distance {worst(loose)[0]:.1e} ({worst(loose)[1]}; printed, not gated)"
        if astray:
            text += (
                "; LAYER 0 IS FED THE SAME BYTES BY BOTH RUNS (the same row-wise kernels in front of it) AND ITS HISTORY TAPS DIFFER: "
                "the verify run does not stand where its kept tokens end"
            )
        if self.cache_equal:
            text += f"; cached keys and values byte-equal in {sum(self.cache_equal.values())} of {len(self.cache_equal)} attention layers"
            text += f", largest distance {self.cache_units:.2f} bf16 units (printed, not gated)" if not all(self.cache_equal.values()) else ""
        return state_ok, text, len(gated), (_verdict(state_ok) if gated or astray else "REPORTED")


def verify_checks(session: Session, decode: Session, stream: Optional[Stream] = None) -> Checks:
    """The verify steps of ``session`` (several tokens per sequence) against the decode steps of ``decode``: a session of
    one token per sequence, for the same sequences, on the same weights, whose run is the token stream both are fed
    (``stream``; default: what ``decode`` has recorded).

    The stream goes through ``session`` in verify steps (:func:`accept_schedule`): each is fed ``tokens`` tokens of the
    stream per sequence and keeps 0, 1, .., ``tokens - 1`` of them behind the first in turn, so every count is kept at
    least once and every step after the first starts from a state a partial keep left.  Every row of every step is a
    token of the stream given the stream in front of it, so each is compared with the decode step of the same token:

    * **tokens** -- the greedy token and the logits of every row against the decode step's, by the family's declared logits
      band (:func:`token_check`; REPORTED where the family declares none).  The two runs are not byte-equal by
      construction: a block that is called at another row count may take another kernel, and the entry of the verify
      step states its own difference (below).
    * **kept state** -- after every step, what the verify run stands on (``DecodeState.standing``) against the state of
      the decode steps of the kept tokens.  A linear-attention layer whose three history taps are the same bytes in both
      runs at every comparison was fed the same rows by both, so there the two kernels are what is compared: its
      recurrent state is gated by the state rule of band ``gdn_step`` (the share within the band, the cap, finite).
      A layer whose taps differ was fed other rows -- a block in front of it rounds otherwise at another row count --
      and its distance is printed, not gated: that every token's state is the state of that many decode steps on the
      SAME rows is the block line of the session's own checks, in every layer.  The caches of the attention layers
      are printed likewise.  With no layer gated the line is REPORTED, never PASS.  One layer cannot be excused that
      way: where the model's first layer is a linear-attention layer and every block in front of it is the same
      row-wise kernel in both plans (the same provider row, gated on batch invariance on this GPU class), both runs
      feed it the same bytes by the table's own statements -- taps that differ there mean that the verify run does
      not stand where its kept tokens end, and the line FAILS.
    * what the entry of the verify block states about one sequence alone and among several (``Block.batch_note``), where
      it states something.

    The verify session keeps the steps of the schedule as its record -- its own checks (``run_checks``) and its traced
    pass then run on them -- and ``decode`` is left reset."""
    tokens, sequences = session.tokens, session.sequences
    if tokens == 1 or decode.tokens != 1:
        raise ValueError("verify_checks compares a session of several tokens per sequence (tokens=) with a session of one token per sequence")
    if decode.sequences != sequences or decode.weights is not session.weights or decode.weights is None:
        raise ValueError("verify_checks compares two sessions of the same sequences on one load (share=), both loaded")
    stream = Stream.of(decode) if stream is None else stream
    schedule = accept_schedule(tokens, stream.steps)
    if len(schedule) > session.record_steps:
        raise ValueError(f"the verify session records {session.record_steps} steps; this comparison makes {len(schedule)} (record_steps=)")
    session.reset(fresh=True)
    decode.reset(fresh=True)
    standing, layers = _KeptState(session.model.geometry.layer_kinds), len(session.model.geometry.layer_kinds)
    verify_logits, decode_logits, verify_ids, decode_ids = [], [], [], []
    for start, kept in schedule:
        ids = stream.inputs[start : start + tokens].t().contiguous()  # [sequences, tokens]: the stream's tokens of this step
        verify_ids.append(session.verify(ids, kept))
        verify_logits.append(session.record.logits[session.record.count - 1].clone())
        steps = stream.logits[start : start + tokens]  # [tokens, sequences, vocabulary] -> the step's row order, sequence by sequence
        decode_logits.append(steps.transpose(0, 1).reshape(sequences * tokens, -1))
        decode_ids.append(stream.outputs[start : start + tokens].t())
        while decode.state.position < start + kept + 1:
            decode._advance(stream.inputs[decode.state.position])
        standing.add(session.state.standing(), decode.state.standing(), kept)
    a, r = torch.cat(verify_logits), torch.cat(decode_logits)
    ids_a, ids_r = torch.cat([ids.reshape(-1) for ids in verify_ids]), torch.cat([ids.reshape(-1) for ids in decode_ids])
    band_units = session.model.logits_band_units
    band_units = None if band_units is None else float(band_units)
    tokens_ok, tokens_text = token_check(a, r, ids_a, ids_r, band_units)
    tokens_text = tokens_text.replace("equal to the reference;", "equal;")  # of this comparison: the decode steps, named in front
    tokens_text = f"{int((ids_a == ids_r).sum())} of {ids_a.numel()} greedy tokens equal to those of the decode steps of the same tokens; " + tokens_text

    state_ok, text, gated, state_verdict = standing.line(_first_layer_fed_alike(session, decode))
    counts = ", ".join(str(kept) for _, kept in schedule)
    lines = [
        f"verify against decode ({sequences} sequence{'s' if sequences > 1 else ''} x {tokens} tokens per sequence, {len(schedule)} verify steps over "
        f"{schedule[-1][0] + tokens} tokens of a run of {stream.steps} decode steps; tokens kept behind the first: {counts}; {layers} layers)"
    ]
    lines.append(f"  tokens            {tokens_text}    {'REPORTED' if band_units is None and tokens_ok else _verdict(tokens_ok)}")
    lines.append(f"  kept state        {text}    {state_verdict}")
    for name, block in session.plan.items():
        if getattr(block, "batch_note", ""):
            lines.append(f"    {name}: {block.batch_note}")
    decode.reset(fresh=True)
    return Checks(lines, tokens_ok and state_ok, gated)


#: what a run says of the kernels of a drafter's passes: they are attributed like every launch, and held to no census
DRAFTER_KERNELS = (
    "kernel identity of the drafter's passes: not stated (the kernel census holds the verify step and the output head behind it; "
    "no section holds a draft pass)"
)


class _Shadow:
    """The drafter's passes of a run written out a second time, for the two lines ``speculative_checks`` adds for a session
    with a drafter.  It keeps a cache of its own and its own books: the kept tokens of the run in order -- the prompt, then
    the tokens of the rows every step kept -- and, for every kept position, the model's final-norm row of it.  A position
    gets its pass when the token that follows it is known: ``body`` at ONE row.  The passes of a step are then the head
    on the last of those rows and, for every later draft, ``body`` on the row and the proposal of the pass before, one
    position further, and the head.  It never calls the drafter's ``draft``: that function is what the lines hold.  Both
    sides run the same blocks, so the lines hold which rows, tokens and positions a step's passes are handed -- not the
    arithmetic of a block."""

    def __init__(self, session: Session):
        self.drafter, self.weights, self.call, self.passes = session.drafter, session.draft_weights, session._draft_call, session.draft_tokens
        self.cache = session.drafter.state(session.sequences, session.capacity, session.device)
        self.tokens: List[torch.Tensor] = []  # the kept token of every position, [sequences] each
        self.rows: List[torch.Tensor] = []  # the model's final-norm row of every kept position, [sequences, hidden] each
        self.done = 0  # the positions whose pass has run
        self.last = None  # the output row of the last of them
        self.cache_steps = self.proposal_steps = 0  # the steps whose cache / whose proposals were the session's

    def keep(self, tokens: torch.Tensor, hidden: torch.Tensor) -> None:
        """``tokens`` ``[sequences, n]`` were kept at the next ``n`` positions; ``hidden`` ``[sequences, n, hidden]`` are
        the model's final-norm rows of the steps that fed them (copied: the model's next step may write its buffer)."""
        for t in range(tokens.shape[1]):
            self.tokens.append(tokens[:, t].clone())
            self.rows.append(hidden[:, t].clone())

    def step(self, first: torch.Tensor) -> torch.Tensor:
        """The passes of the step whose window starts with ``first`` ``[sequences]``, the token that follows the last kept
        position -> the proposals ``[sequences, passes]``."""
        following = [*self.tokens[1:], first]
        while self.done < len(self.rows):  # every kept position that has not had its pass: one row, the token that follows it
            self.last = self.drafter.body(self.call, self.weights, self.cache, self.rows[self.done], following[self.done], self.done)
            self.done += 1
        row, drafts = self.last, [self.drafter.propose(self.call, self.weights, self.last)]
        for index in range(1, self.passes):
            row = self.drafter.body(self.call, self.weights, self.cache, row, drafts[-1], self.done - 1 + index, index)
            drafts.append(self.drafter.propose(self.call, self.weights, row, index))
        return torch.stack(drafts, dim=1)

    def compare(self, session: Session, recorded: torch.Tensor, proposals: torch.Tensor) -> None:
        """After the passes of a step on both sides: the session's cache up to the last kept position against this one's, and
        the proposals the session's own run recorded for the step against this one's."""
        mine, theirs = session.state.draft, self.cache
        upto = self.done  # positions 0 .. Q - 1: what the first pass of the step and the prompt's passes wrote from the model's rows
        self.cache_steps += int(bytes_equal((mine.k[:, :, :upto], mine.v[:, :, :upto]), (theirs.k[:, :, :upto], theirs.v[:, :, :upto])))
        self.proposal_steps += int(torch.equal(recorded, proposals))


def forced_drafts(following: Sequence[int], length: int, k: int, vocab: int, variant: int = 0) -> List[int]:
    """The ``k`` drafted tokens of a scheduled verify step, for a run that schedules how many tokens its steps keep.
    ``following`` are the tokens a plain run produced behind the token kept last.  The first ``length`` drafted tokens are
    ``following[:length]``, the next is another token than ``following[length]``, and the rest are filler that differs
    with ``variant`` -- so a step that is correct keeps exactly ``length`` drafted tokens by the greedy rule, and two
    variants of a step differ behind its kept tokens alone.  ``ValueError`` for a length outside ``0 .. k`` or a
    ``following`` too short to say which token the first rejected draft must not be."""
    if not 0 <= length <= k or len(following) < min(length + 1, k):
        raise ValueError(
            f"a step of {k} drafted tokens keeps 0 to {k} of them and needs the tokens behind the kept one; got length={length}, {len(following)} tokens"
        )
    drafts = [int(token) for token in following[:length]]
    if length < k:
        wrong = (int(following[length]) + 1) % vocab  # another token than the one the plain run produced there
        drafts += [wrong, *((wrong + 1 + variant + index) % vocab for index in range(k - length - 1))]
    return drafts


def speculative_checks(session: Session, decode: Session, scheduled: Optional[Sequence[int]] = None) -> Checks:
    """The run of ``session`` -- built with ``draft_tokens=``: a prompt through decode steps, then verify steps on the same
    state -- against decode steps of the same tokens on ``decode``, a session of one token per sequence for the same
    sequences on the same weights.

    What the session recorded is replayed from a zero state: its prompt, the windows it verified, the counts it kept.
    ``decode`` is fed the same kept tokens -- the prompt, then per verify step the tokens of the kept rows -- so every
    kept row is a token given the same tokens in front of it in both runs.  Rows behind the kept ones follow a token the
    kept run never had and are not compared.  The lines:

    * **tokens** -- every kept row against the decode step of the same position, by the family's declared logits band
      (:func:`token_check`); and the greedy rule of the loop itself: the first token of every window is the greedy token
      of the row the step before kept last (of the prompt's last step, for the first window), and every kept drafted
      token is the greedy token of the row in front of it -- a loop that keeps a count it did not compare, or emits a
      drafted token, fails here.  Where the family declares no band the line is REPORTED: how many kept tokens equal
      the decode step's choice, the first that differs, and how many are greedy tokens of the session's own steps.
    * **kept state** -- after every verify step, what the session stands on against the decode session's state, by the
      rule of :func:`verify_checks` (gated in the linear-attention layers both runs fed the same rows).
    * **position** -- after the prompt and after every verify step the session's position is the decode session's and
      the count of the prompt's tokens and the kept ones, and its state stands on the token the step kept; and the
      session's OWN run ended at that position on the token its last step kept (read before the first replay; held when
      every step it ran is in its record).  Integers: no rounding is involved, on any stack.
    * **prompt** -- after the prompt, what every layer stands on (the attention caches included) and the logits and ids
      of every prompt step are the decode session's bytes.  Both ran the same decode rows at the same row count, on the
      same kernels: gated on every stack.
    * **recorded** -- the ids and logits of every step the session recorded of its own run, the prompt's decode steps
      and the verify steps, are the bytes of the first replay.  The replays here are driven by this comparison (the
      family's step on the session's plans and state); this line and the own run's position hold that what they judge
      is the run the session's own methods ran.  Two runs of the same kernels on the same rows: gated on every stack.
    * **bytes** -- the logits of every kept row and what every layer stands on after every verify step are the decode
      session's bytes.  Gated on the reference stack, where a verify step is that many decode steps of the same chain;
      REPORTED on a provider stack, where the two kinds of step launch other kernels for some blocks.
    * **rejected rows** -- the verify steps are replayed a second time with other tokens behind the kept ones: the ids
      and logits of every kept row and what the session stands on after every step are byte-identical to the first
      replay.  A kept row does not read what a rejected token wrote; no rounding separates the two replays on any stack.
    * **acceptance** -- with ``scheduled`` (the count each step was scheduled to keep, as :func:`forced_drafts` drafts
      it): every step in front of the first near-tie among the compared rows kept its scheduled count, and every count
      from 0 to ``draft_tokens`` was kept at least once.

    For a session with a drafter the first replay runs the drafter's passes as the session runs them, and a shadow run
    beside it writes them out at one row per call from books of its own (:class:`_Shadow`).  Two more lines:

    * **drafter cache** -- after the passes of every step, the session's drafter cache up to the last kept position is
      the shadow's, byte for byte.  It sees the rows, tokens and positions of the first pass of a step and of the passes
      over the prompt: a pass one position off, a prompt position without its pass, a drafted token where the kept one
      belongs, the rows of an older step.  It does not see which row the step goes on from, nor the later passes.
    * **draft proposals** -- the proposals the session's own run recorded for every step are the shadow's.  It sees the
      row a step goes on from and what every later pass is handed.

    Both sides call the same blocks: neither line is a check of a block's arithmetic.

    Both sessions are left reset; ``session`` keeps its records, so its traced pass and its own checks (``run_checks``)
    run on them afterwards -- a new prompt set starts with ``Session.reset``, as always.  The comparison is made once
    per run, on the session as its run left it: a session that stands behind no verify step and holds recorded ones was
    reset since (by this comparison), and is a ``ValueError``.  The comparison keeps what the session stands on after
    every replayed step until the second replay has ended."""
    if session.draft_tokens is None or decode.tokens != 1:
        raise ValueError("speculative_checks compares a session built with draft_tokens= with a session of one token per sequence")
    if decode.sequences != session.sequences or decode.weights is not session.weights or decode.weights is None:
        raise ValueError("speculative_checks compares two sessions of the same sequences on one load (share=), both loaded")
    session._closed("speculative_checks")
    S, T, vocab = session.sequences, session.tokens, session.model.geometry.vocab
    ran, ran_prompt = session.record, session.prompt_record  # the session's own run: what it recorded of its steps
    steps = min(ran.count, ran.inputs.shape[0])
    if steps < 1:
        raise ValueError("speculative_checks compares the verify steps a session recorded behind its prompt; this session recorded none")
    # ... and where that run left the session, read before the first replay resets it: the position, the token it stands on
    own_position, own_kept = session.state.position, session.state.accepted
    if not session.state.verify_steps:
        raise ValueError(
            f"speculative_checks holds a session where its run left it; this session recorded {steps} verify step{'s' if steps > 1 else ''} and stands "
            "behind none: it was reset since (the comparison leaves a session reset with its records, so it is made once per run)"
        )
    windows = session.record.inputs[:steps].clone().view(steps, S, T)
    counts = list(session.record.accepted[:steps])
    prompt = ran_prompt.prompt
    prompt = torch.zeros((S, 0), dtype=torch.int32, device=session.device) if prompt is None else prompt
    fed, layers = int(prompt.shape[1]), len(session.model.geometry.layer_kinds)
    if scheduled is not None and len(scheduled) != steps:
        raise ValueError(f"the schedule names {len(scheduled)} steps; the session recorded {steps}")

    def held(state) -> list:
        return [tuple(t.clone() for t in layer) for layer in state.standing()]

    # ---- the first replay, with the decode session beside it (and, with a drafter, the shadow of its passes)
    session.reset(fresh=True)
    decode.reset(fresh=True)
    shadow = _Shadow(session) if session.drafter is not None else None
    prompt_steps = layers_alike = greedy_kept = greedy_all = as_ran = 0
    recorded = min(fed, ran_prompt.inputs.shape[0])  # the prompt steps whose ids and logits a run keeps
    last = None  # the greedy ids of the row the run kept last: what the next window starts with
    for t in range(fed):
        ids = prompt[:, t].contiguous()
        session._fill(ids)  # with a drafter: its pass over the position in front, as ``prefill`` runs it
        mine, theirs = session._run(session.prefill_plan, 1, ids), decode._run(decode.plan, 1, ids)
        if shadow is not None:
            shadow.keep(ids[:, None], session.state.hidden[:, None])
        prompt_steps += int(bytes_equal(mine, theirs))  # the next ids and the logits of the step
        if t < min(recorded, ran_prompt.count):  # the step as the session's own run recorded it: the ids fed, the ids chosen, the logits
            as_ran += int(torch.equal(ids, ran_prompt.inputs[t]) and bytes_equal(mine, (ran_prompt.outputs[t], ran_prompt.logits[t])))
        last = mine[0]
    for mine, theirs in zip(session.state.standing(), decode.state.standing()):
        layers_alike += int(bytes_equal(mine, theirs))
    expected = fed
    positions = [(session.state.position, decode.state.position, expected)]
    standing = _KeptState(session.model.geometry.layer_kinds)
    first, kept_logits, kept_ids, ref_logits, ref_ids, margins = [], [], [], [], [], []
    stands, ordinals = [], []  # per step the token the state stands on with the count the step kept; per kept row (its number behind the prompt, its sequence)
    for j in range(steps):
        a = counts[j]
        if shadow is not None:  # the passes of this step on both sides, then what they left and what the run recorded of them
            session._draft()
            shadow.compare(session, ran.proposals[j], shadow.step(windows[j][:, 0]))
        ids, logits = session._run(session.plan, T, windows[j].reshape(-1))
        session.state.accept(a)
        if shadow is not None:  # the rows this step kept, each with its token
            shadow.keep(windows[j][:, : a + 1], session.state.hidden.view(S, T, -1)[:, : a + 1])
        as_ran += int(bytes_equal((ids, logits), (ran.outputs[j], ran.logits[j])))
        stands.append((session.state.accepted, a))
        ordinals += [(expected - fed + t + 1, i) for i in range(S) for t in range(a + 1)]
        rows = logits.view(S, T, -1)[:, : a + 1].reshape(S * (a + 1), -1)  # the kept rows, sequence by sequence
        chosen = ids.view(S, T)[:, : a + 1].reshape(-1)
        # the greedy rule: the window starts with the greedy token kept last, and a kept drafted token is the greedy token in front of it
        greedy_kept += int((windows[j][:, 1 : a + 1] == ids.view(S, T)[:, :a]).sum()) + (S if last is None else int((windows[j][:, 0] == last).sum()))
        greedy_all += S * (a + 1)
        last = ids.view(S, T)[:, a].clone()
        decoded = [decode._run(decode.plan, 1, windows[j][:, t].contiguous()) for t in range(a + 1)]
        ref_rows = torch.stack([each[1] for each in decoded], dim=1).reshape(S * (a + 1), -1)
        ref_chosen = torch.stack([each[0] for each in decoded], dim=1).reshape(-1)
        expected += a + 1
        positions.append((session.state.position, decode.state.position, expected))
        own = held(session.state)
        standing.add(own, decode.state.standing(), a)
        first.append((chosen.clone(), rows.clone(), own))
        kept_logits.append(rows)
        kept_ids.append(chosen)
        ref_logits.append(ref_rows)
        ref_ids.append(ref_chosen)
        margins.append(margin(ref_rows))

    # ---- the second replay: other tokens behind the kept ones
    other = windows.clone()
    for j in range(steps):
        other[j][:, counts[j] + 1 :] = (other[j][:, counts[j] + 1 :] + 1) % vocab
    replaced = sum(S * (T - 1 - count) for count in counts)
    session.reset(fresh=True)
    for t in range(fed):
        session._run(session.prefill_plan, 1, prompt[:, t].contiguous())
    untouched = 0  # the steps whose kept rows and standing state did not move with the tokens behind them
    for j in range(steps):
        a = counts[j]
        ids, logits = session._run(session.plan, T, other[j].reshape(-1))
        session.state.accept(a)
        again = (ids.view(S, T)[:, : a + 1].reshape(-1), logits.view(S, T, -1)[:, : a + 1].reshape(S * (a + 1), -1), held(session.state))
        untouched += int(bytes_equal(again, first[j]))

    # ---- the lines
    band_units = session.model.logits_band_units
    band_units = None if band_units is None else float(band_units)
    plural = "s" if S > 1 else ""
    lines = [
        f"speculative against decode ({S} sequence{plural} x {T} tokens per sequence, {steps} engine steps behind a prompt of {fed} tokens; "
        f"tokens kept behind the first: {', '.join(str(count) for count in counts)}; {layers} layers)"
    ]
    near = [False] * steps
    a, r, ids_a, ids_r = torch.cat(kept_logits), torch.cat(ref_logits), torch.cat(kept_ids), torch.cat(ref_ids)
    tokens_ok, tokens_text = token_check(a, r, ids_a, ids_r, band_units)
    tokens_text = tokens_text.replace("equal to the reference;", "equal;")  # of this comparison: the decode steps, named in front
    # the kept rows of a step stand sequence by sequence: the first that differs is the one with the smallest number behind the prompt
    differs = [ordinals[index] for index in (ids_a != ids_r).nonzero().flatten().tolist()]
    token, sequence = min(differs, default=(0, 0))
    where = f"the first that differs is kept token {token}" + (f" of sequence {sequence}" if S > 1 else "") if differs else "none differs"
    tokens_text = (
        f"{int((ids_a == ids_r).sum())} of {ids_a.numel()} kept tokens equal to those of the decode steps of the same tokens ({where}); "
        f"{greedy_kept} of {greedy_all} are the greedy token of the session's own step in front of them; " + tokens_text
    )
    if band_units is not None:  # a loop that kept a token its steps did not choose is no rounding
        tokens_ok = tokens_ok and greedy_kept == greedy_all
        # ... and the steps with a near-tie among their compared rows: what the acceptance line stops at
        unit = 2.0 ** (math.frexp(float(r.float().abs().amax()))[1] - 8)
        near = [bool((each <= LOGITS_FACTOR * band_units * unit).any()) for each in margins]
    lines.append(f"  tokens            {tokens_text}    {'REPORTED' if band_units is None and tokens_ok else _verdict(tokens_ok)}")
    _, text, gated, state_verdict = standing.line(_first_layer_fed_alike(session, decode))
    lines.append(f"  kept state        {text}    {state_verdict}")
    astray = next((index for index, (mine, theirs, own) in enumerate(positions) if not mine == theirs == own), None)
    off = next((index for index, (stood, kept) in enumerate(stands) if stood != kept), None)  # the state not on the token the step kept
    whole = ran.count == steps  # every verify step the session ran is in its record: where its own run ended is then known
    own_ok = not whole or (own_position, own_kept) == (expected, counts[-1])
    position_ok = astray is None and off is None and own_ok
    if astray is not None:
        mine, theirs, own = positions[astray]
        at = "after the prompt" if astray == 0 else f"after step {astray}"
        position_text = f"{at} the session stands at position {mine}, the decode steps at {theirs}; the prompt's tokens and the kept ones are {own}"
    elif off is not None:
        position_text = f"after step {off + 1} the state stands on token {stands[off][0]} of the step; the step kept {stands[off][1]} tokens behind the first"
    elif not own_ok:
        position_text = (
            f"the session's own run ended at position {own_position} on token {own_kept} of its last step; the prompt's tokens and the kept ones "
            f"are {expected}, and its last step kept {counts[-1]} tokens behind the first"
        )
    else:
        ended = (
            "the session's own run ended there"
            if whole
            else f"where the session's own run ended is not held (it ran {ran.count} verify steps and recorded {steps})"
        )
        position_text = (
            f"after the prompt and after each of the {steps} steps the session stands at the decode steps' position, on the token the step kept: "
            f"{fed} prompt tokens + the kept ones = {expected}; {ended}"
        )
    lines.append(f"  position          {position_text}    {_verdict(position_ok)}")
    prompt_ok = prompt_steps == fed and layers_alike == layers
    if fed:
        lines.append(
            f"  prompt            after the prompt, what the session stands on is the decode session's bytes in {layers_alike} of {layers} layers "
            f"(states and caches); ids and logits of {prompt_steps} of {fed} prompt steps byte-equal    {_verdict(prompt_ok)}"
        )
    else:  # a line that compares nothing is not printed as a pass
        lines.append("  prompt            no prompt was fed: the verify steps of this run began on a zero state    REPORTED")
    lines.append(
        f"  recorded          ids and logits of {as_ran} of {recorded + steps} steps the session recorded of its own run ({recorded} of its prompt, "
        f"{steps} verify steps) are this replay's bytes    {_verdict(as_ran == recorded + steps)}"
    )
    rows_equal = sum(int(bytes_equal(mine[n], theirs[n])) for mine, theirs in zip(kept_logits, ref_logits) for n in range(mine.shape[0]))
    rows_all = sum(int(mine.shape[0]) for mine in kept_logits)
    layers_equal = sum(standing.equal.values())
    bytes_ok = rows_equal == rows_all and layers_equal == layers
    gate_bytes = session.stack == "reference"
    bytes_text = (
        f"the logits of {rows_equal} of {rows_all} kept rows are the decode steps' bytes; after every step {layers_equal} of {layers} layers "
        "stand on the decode steps' bytes"
    )
    if not gate_bytes:
        bytes_text += f" (reported: on stack {session.stack} the two kinds of step may launch other kernels for a block)"
    lines.append(f"  bytes             {bytes_text}    {_verdict(bytes_ok) if gate_bytes else 'REPORTED'}")
    rejected_ok = untouched == steps
    rejected_text = (
        f"a second replay with other tokens in the {replaced} rows behind the kept ones: ids and logits of the kept rows and the standing "
        f"state byte-identical to the first replay after {untouched} of {steps} steps"
    )
    lines.append(f"  rejected rows     {rejected_text}    {_verdict(rejected_ok) if replaced else 'REPORTED'}")
    if shadow is not None:
        lines.append(
            f"  drafter cache     after the passes of {shadow.cache_steps} of {steps} steps the drafter's cache up to the last kept position is the "
            f"bytes of the passes written out at one row per position    {_verdict(shadow.cache_steps == steps)}"
        )
        lines.append(
            f"  draft proposals   the {session.draft_tokens} proposals the run recorded are those of the passes written out at one row, "
            f"in {shadow.proposal_steps} of {steps} steps    {_verdict(shadow.proposal_steps == steps)}"
        )
    if scheduled is not None:
        stop = next((index for index, tie in enumerate(near) if tie), steps)  # a near-tie ends what a schedule can say
        as_scheduled = sum(int(counts[j] == scheduled[j]) for j in range(stop))
        never = [str(count) for count in range(T) if count not in counts]
        acceptance_ok = as_scheduled == stop and not never
        text = f"{as_scheduled} of {stop} steps kept the count they were scheduled to keep"
        if stop < steps:
            text += f" (the comparison ends at step {stop + 1}: a near-tie among its compared rows)"
        text += f"; every count from 0 to {T - 1} was kept at least once" if not never else f"; never kept: {', '.join(never)}"
        lines.append(f"  acceptance        {text}    {_verdict(acceptance_ok)}")
    for name, block in session.plan.items():
        if getattr(block, "batch_note", ""):
            lines.append(f"    {name}: {block.batch_note}")
    session.reset(fresh=True)
    session.record, session.prompt_record = ran, ran_prompt
    decode.reset(fresh=True)
    # a line that fails fails the comparison: the verdict each line prints is the one that counts, a REPORTED line gates nothing
    return Checks(lines, not any(line.endswith(f"    {_verdict(False)}") for line in lines), gated)
