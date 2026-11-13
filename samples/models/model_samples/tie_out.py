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

Bands: ``bf16_rounding`` / ``bf16_unit`` = every element finite and within one bf16 unit of the reference, the unit taken at
the larger magnitude and not below 2**-8 of the tensor's largest value (a multiply that rounds its unscaled sum to bf16 and the
scaled result again differs from the once-rounded reference by one unit on a share of its elements); ``gdn_step`` = see below;
``gate_fp8`` = see below; ``same`` = the provider is the reference chain, nothing to compare.

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
from .loop import Session
from .sampling import margin

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
    outside: int = 0  # gate_fp8: bytes that are the code of neither bf16 value around the gated value
    off_nearest: int = 0  # gate_fp8: bytes off the nearest value's code although the gated value is beyond the margin
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
    provider's results before it returns them."""

    def __init__(self, plan, reference):
        self.plan, self.reference = plan, reference
        self.stats: Dict[str, BlockStats] = {}
        self.step = 0  # set by the checked pass before every step

    @staticmethod
    def _state(block, operands):
        return [operands[i] for i in block.state_args]

    @staticmethod
    def _save(holders):
        return [tuple(t.clone() for t in h.tensors()) for h in holders]

    @staticmethod
    def _load(holders, saved):
        for h, s in zip(holders, saved):
            for t, c in zip(h.tensors(), s):
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
            ref_out = ref(*operands)
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

    def text(self) -> str:
        return "\n".join(self.lines)


def _verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


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


def run_checks(session: Session, prompt_tokens: int, max_steps: Optional[int] = None, second_reference: bool = False, kernels: bool = False) -> Checks:
    """Replay, blocks, tokens and batch invariance for the run ``session`` has just recorded: its recorded steps, or the first
    ``max_steps`` of them (a long run of a deep model is checked on its first steps).  Leaves the session reset.

    ``kernels`` adds the kernel check on the traced pass the session made before this call (``Session.trace`` with launch
    records): a failure fails the run; kernels that could not be observed are reported, and ``Checks.unobserved`` says why.

    ``second_reference`` runs the reference stack once more with the other accumulation precision (IEEE fp32 where the
    reference accumulates in fp64) and prints, beside the token check, how far that second reference is from the first and
    on how many greedy tokens the two agree.  It is a scale, printed and not gated: on a trained checkpoint with 4-bit
    activations two stacks that differ only in rounding end up tens of bf16 units apart (see the family's logits band)."""
    main = session.record
    steps = min(main.count, main.inputs.shape[0])
    if max_steps is not None:
        steps = min(steps, max_steps)
    inputs, logits, ids = main.inputs[:steps].clone(), main.logits[:steps].clone(), main.outputs[:steps].clone()
    facts = session.plan.facts
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
    )
    reference.load()

    tap = BlockTap(session.plan, reference.plan)
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
    status, unobserved = "", ""
    if kernels:
        status, unobserved, kernel_lines = kernel_check(session)
        lines += kernel_lines
    session.reset(fresh=True)
    return Checks(lines, replay_ok and blocks_ok and tokens_ok and batch_ok and status != census.FAIL, len(judged), status, unobserved)
