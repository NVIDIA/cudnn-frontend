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

Bands: ``bf16_rounding`` / ``bf16_unit`` = every element finite and within one bf16 unit of the reference, the unit taken at
the larger magnitude and not below 2**-8 of the tensor's largest value (a multiply that rounds its unscaled sum to bf16 and the
scaled result again differs from the once-rounded reference by one unit on a share of its elements); ``gdn_step`` = see below;
``gate_fp8`` = see below; ``add_norm_quant`` = see below; ``same`` = the provider is the reference chain, nothing to compare.

**What a provider block is compared with.**  The reference block of the same name -- unless the provider's kernel rounds at
other points than the reference chain: then the block names its twin (``Block.twin``), a reference block written at the
kernel's rounding points, and is compared with that.  The reference is matched to the kernel's documented arithmetic; a
band is never widened to cover a chain that rounds elsewhere.  **A band is judged per output**: it names the outputs it
applies to and what it needs of the twin; an output it does not name is compared by its dtype (bf16: within one bf16
unit; anything else: bytes), and state a call updates is compared byte for byte unless the band has a rule for it.

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
from typing import Dict, List, Optional, Tuple

import torch

from . import census
from .blocks import GIB
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
        self.step = 0  # set by the checked pass before every step

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

    def __call__(self, name, layer, block, operands):
        row, ref = self.plan.rows[name], block.twin(self.reference[name])  # the reference block, or the block's own twin
        independent = type(ref) is not type(block)
        alone = self.plan.sequences > 1
        if not independent and not alone:
            return block(*operands)
        stats = self.stats.setdefault(name, BlockStats(row.band if independent else "same"))
        stats.calls += 1
        holders = self._state(block, operands)
        before = self._save(holders)
        out = block(*operands)
        compared = tuple(t.clone() for t in _tensors(block.compared(out)))
        after = self._save(holders)
        if independent:
            self._load(holders, before)
            ref_out = ref(*(operands if self.on_demand is None else self.on_demand.operands(operands)))  # a copy made now is dropped here
            self._judge(stats, compared, _tensors(ref.compared(ref_out)), after, self._save(holders), layer, ref)
        if alone:
            s = self.step % self.plan.sequences
            self._load(holders, before)
            single = [(_rows(o, s, s + 1) if i in block.per_row else o) for i, o in enumerate(operands)]
            alone_out = block(*single)
            same = bytes_equal(tuple(t[s : s + 1] for t in compared), block.compared(alone_out))
            for holder, full, one in zip(holders, after, self._save(holders)):
                at = s + 1 if hasattr(holder, "conv_storage") else s  # a slot pool keeps sequence s in slot s + 1, a cache in row s
                same = same and bytes_equal(tuple(t[at] for t in full), tuple(t[at] for t in one))
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
            cells = stats.cells.get(layer)
            if cells is None:
                cells = stats.cells[layer] = _Cells(torch.zeros_like(same.view(-1, heads, dim).sum(-1)), torch.zeros_like(inside.flatten(2).sum(-1)))
            cells.equal += same.view(-1, heads, dim).sum(-1)
            cells.inside += inside.flatten(2).sum(-1)
            cells.calls += 1
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

    def text(self) -> str:
        return "\n".join(self.lines)


def _verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


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


def run_checks(session: Session, prompt_tokens: int, max_steps: Optional[int] = None, second_reference: bool = False, kernels: Optional[bool] = None) -> Checks:
    """Replay, blocks, tokens and batch invariance for the run ``session`` has just recorded: its recorded steps, or the first
    ``max_steps`` of them (a long run of a deep model is checked on its first steps).  Leaves the session reset.

    ``kernels`` adds the kernel check on the traced pass the session made before this call (``Session.trace`` with launch
    records): a failure fails the run; kernels that could not be observed are reported, and ``Checks.unobserved`` says why.
    Left out, it is the session's own decision (``Session.check_kernels``): a session of a stack that declares a target
    kernel set is checked without being asked, so no caller can print a verdict for such a run that never looked at its
    kernels.  For such a session ``kernels=False`` is refused.

    ``second_reference`` runs the reference stack once more with the other accumulation precision (IEEE fp32 where the
    reference accumulates in fp64) and prints, beside the token check, how far that second reference is from the first and
    on how many greedy tokens the two agree.  It is a scale, printed and not gated: on a trained checkpoint with 4-bit
    activations two stacks that differ only in rounding end up tens of bf16 units apart (see the family's logits band)."""
    if kernels is None:
        kernels = session.check_kernels
    elif not kernels and session.stack in getattr(session.model, "target_stacks", ()):
        raise ValueError(f"stack {session.stack} declares a target kernel set: its kernel check cannot be switched off")
    main = session.record
    steps = min(main.count, main.inputs.shape[0])
    if max_steps is not None:
        steps = min(steps, max_steps)
    inputs, logits, ids = main.inputs[:steps].clone(), main.logits[:steps].clone(), main.outputs[:steps].clone()
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
        record_steps=steps,
        on_demand=cost,
    )
    reference.load()

    tap = BlockTap(session.plan, reference.plan, cost)
    session.reset(fresh=True)
    session.tap = tap
    try:
        for t in range(steps):
            tap.step = t
            session._advance(inputs[t])
    finally:
        session.tap = None
    replay_ok = bytes_equal(session.record.logits[:steps], logits) and bytes_equal(session.record.outputs[:steps], ids)

    reference.reset(fresh=True)
    for t in range(steps):
        reference._advance(inputs[t])
    band_units = session.model.logits_band_units  # declared by the family, in its configs.py; None: the token line is reported
    band_units = None if band_units is None else float(band_units)
    ref_logits, ref_ids = reference.record.logits[:steps], reference.record.outputs[:steps]
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
            record_steps=steps,
            on_demand=cost,
        )
        matmul = torch.backends.cuda.matmul
        kept = getattr(matmul, "fp32_precision", None)  # a session with IEEE fp32 accumulation sets it for the process
        try:
            second.load()
            second.reset(fresh=True)
            for t in range(steps):
                second._advance(inputs[t])
        finally:
            if kept is not None:
                matmul.fp32_precision = kept
        _, scale_text = token_check(second.record.logits[:steps], ref_logits, second.record.outputs[:steps], ref_ids, band_units)
        distance = scale_text.split("largest logit distance ", 1)[1].split("  [", 1)[0]
        tokens_text += (
            f"; for scale, the reference stack with {other} accumulation is {distance} from the one with {session.reference_precision} "
            f"and agrees with it on {int((second.record.outputs[:steps] == ref_ids).sum())} of {ids.numel()} greedy tokens (printed, not gated)"
        )

    changed_after = changed_weights(session.weights)  # after the last pass that read a copy made on demand
    stored_ok = not changed_before and not changed_after
    judged = {n: s for n, s in tap.stats.items() if s.band != "same"}
    blocks_ok = all(s.ok for s in judged.values())
    lines = [
        f"checks ({session.sequences} sequence{'s' if session.sequences > 1 else ''}, {'the first ' if steps < main.count else ''}{steps} steps: "
        f"{min(prompt_tokens, steps)} prompt + {max(steps - prompt_tokens, 0)} decode)"
    ]
    lines.append(f"  replay            ids and logits byte-identical on a second run from a fresh state    {_verdict(replay_ok)}")
    inside = sum(s.ok for s in judged.values())
    if judged:
        lines.append(
            f"  blocks            {inside} of {len(judged)} provider blocks inside their bands against the {session.reference_precision}-accumulated "
            f"reference, teacher-forced    {_verdict(blocks_ok)}"
        )
    else:  # a line that compares nothing cannot fail: it is not printed as a pass
        lines.append("  blocks            no provider block to judge: every block of this run is the reference chain itself    REPORTED")
    for name, s in judged.items():
        lines.append(f"    {name:<22} {s.calls:>4} calls  {s.line()}  [band: {s.band}]  {_verdict(s.ok)}")
    for band in dict.fromkeys(s.band for s in judged.values()):  # each band's text once, on a line of its own
        lines.append(f"    band {band}: {BANDS[band]}")
    lines.append(f"  tokens            {tokens_text}    {'REPORTED' if band_units is None and tokens_ok else _verdict(tokens_ok)}")
    batch_ok = True
    if session.sequences > 1:
        cls = f"cc{facts.cc[0]}.{facts.cc[1]}"
        differs, gated = [], []
        for name, s in tap.stats.items():
            is_gate = cls in session.plan.rows[name].batch_invariant
            if is_gate:
                gated.append(f"{name} {'equal' if s.batch_equal == s.batch_calls else 'DIFFERS'} (gated)")
                batch_ok = batch_ok and s.batch_equal == s.batch_calls
            if s.batch_equal != s.batch_calls:
                differs.append(f"{name} ({s.batch_calls - s.batch_equal} of {s.batch_calls} calls)")
        gate_text = "; ".join(gated) if gated else f"no block is gated on {facts.class_text}"
        rest = f"{len(tap.stats) - len(differs)} of {len(tap.stats)} blocks equal" + (f"; differs: {', '.join(differs)}" if differs else "")
        verdict = _verdict(batch_ok) if gated else "REPORTED"
        rotation = f"one sequence alone vs among {session.sequences} (step t: sequence t mod {session.sequences})"
        lines.append(f"  batch invariance  {rotation}: {gate_text}; {rest}    {verdict}")
    else:
        lines.append("  batch invariance  not applicable at 1 sequence")
    if alone:  # said only where the run keeps a weight without its bf16 copy: whether the reference's copies were made from the load's bytes
        lines.append(stored_weights_line(alone, changed_before, changed_after))
    if cost.copies:  # what the checks cost beyond the run: said only where a copy was made on demand
        lines.append(on_demand_line(cost, getattr(session.weights, "uncopied", ())))
    status, unobserved = "", ""
    if kernels:
        status, unobserved, kernel_lines = kernel_check(session)
        lines += kernel_lines
    session.reset(fresh=True)
    ok = replay_ok and blocks_ok and tokens_ok and batch_ok and stored_ok and status != census.FAIL
    return Checks(lines, ok, len(judged), status, unobserved, cost)
