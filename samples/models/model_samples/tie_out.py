# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The checks of a sample run against the plain-PyTorch reference.

Four checks, run after the timed loop on the ids the loop produced:

1. **replay** -- the same inputs from a fresh state (new allocations) give byte-identical ids and logits.
2. **blocks** -- teacher-forced: every provider block that is not the reference chain itself is compared, call by call, with
   the reference block on the SAME operands and the same state, so an error cannot accumulate and a wiring error is told apart
   from rounding.  The band of a block is named in its provider row (``BANDS``) and is never widened here.
3. **tokens** -- end to end and margin-aware: the reference stack runs the same inputs with its own state; the greedy token must
   equal the reference's at every step whose reference margin (largest minus second largest logit) exceeds the logits band,
   ``LOGITS_FACTOR`` times the largest logit distance between the two stacks in this run.  Steps below the band are counted and
   printed as near-ties; they are not failures and not hidden.  The check cannot be sharper than the two stacks are close:
   a chain of 8-bit and 4-bit quantisers turns a one-unit rounding difference into different codes, so the logits of two
   correct stacks differ by far more than one rounding, and the block check above is what verifies each provider.
4. **batch invariance** -- with more than one sequence, every block is run again on sequence 0 alone, on the same operands
   and state; the bytes of sequence 0 must be equal.  A gate for a block whose provider row lists the GPU class under
   ``batch_invariant``; a printed fact for the rest.

Bands: ``bf16_rounding`` / ``bf16_unit`` = every element within one bf16 unit of the reference, the unit taken at the larger
magnitude and not below 2**-8 of the tensor's largest value (a multiply that rounds its unscaled sum to bf16 and the scaled
result again differs from the once-rounded reference by one unit on a share of its elements); ``gdn_step`` = more than
99.9 % of the 8-bit output bytes of the run equal, more than 99.9 % of the recurrent state's elements within 1e-4 and none
farther than two bf16 units (2**-6) of the state's largest value (of 1 when that is larger), and the conv history
byte-identical; ``same`` = the provider is the reference chain, nothing to compare.

Why the state band has two parts: the fused step rounds its beta / alpha projection, its conv taps and its SiLU to bf16 as a
serving chain does, and it accumulates them in fp32 in its own order.  Against an fp64-accumulated reference a few of those
roundings fall on the other side of a midpoint; one such unit moves one head's write gate, its decay or one key column, and
with it that head's state, by up to a few 1e-3 of the state's largest value, while every other element agrees to fp32
rounding.  A state "within 1e-4 everywhere" therefore does not hold against this reference (measured by this sample: the
largest distance of a default run is a few 1e-4 to a few 1e-3), and it is not what a wiring error looks like: that moves every
element.  The cap is one unit for each of the two gates of a head.

What the band tolerates: such moves of single heads, as long as they stay below 0.1 % of the state's elements over the run and
below the cap.  What it still catches: a head that is wrong at every step (one value head of 48 is 2 % of the elements), any
error of the wiring or of an operand's layout (with the beta and alpha rows exchanged, 28 % of the state was within 1e-4 and
the largest distance was 0.99), and any element beyond the cap.  The line of the step also prints the state's rms ratio --
the rms of the difference over the rms of the reference state, at the worst call -- as a second, averaged view of the same
comparison: one flipped head barely moves it, an error of the wiring moves it to order one.  It is printed, not gated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch

from .loop import Session
from .sampling import margin

LOGITS_FACTOR = 2.0
GDN_EQUAL_BYTES = 0.999
GDN_STATE_BAND = 1e-4
GDN_STATE_SHARE = 0.999
GDN_STATE_CAP = 2.0**-6
BANDS = {
    "same": "the provider is the reference chain",
    "bitwise": "byte-identical",
    "bf16_rounding": "within one bf16 rounding",
    "bf16_unit": "within one bf16 unit",
    "gdn_step": f"more than {GDN_EQUAL_BYTES:.1%} equal 8-bit bytes, more than {GDN_STATE_SHARE:.1%} of the state within {GDN_STATE_BAND:g} and all of it within two bf16 units, conv history equal",
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


@dataclass
class BlockStats:
    band: str
    calls: int = 0
    ok: bool = True
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
    conv_equal: bool = True
    batch_calls: int = 0
    batch_equal: int = 0
    notes: List[str] = field(default_factory=list)

    def line(self) -> str:
        if self.band == "gdn_step":
            share = self.equal_bytes / max(self.total_bytes, 1)
            conv = "equal" if self.conv_equal else "DIFFERS"
            inside = self.state_inside / max(self.state_elements, 1)
            return (
                f"{share:.3%} of 8-bit bytes equal (worst call {self.worst_call:.3%}); {inside:.4%} of the state within {GDN_STATE_BAND:g}, "
                f"largest distance {self.state_error:.1e}, rms ratio {self.state_rms:.1e} at the worst call (printed, not gated); conv history {conv}"
            )
        share = self.off_by_one / max(self.elements, 1)
        return f"largest distance {self.worst_units:.2f} bf16 units; {share:.1%} of elements differ by one unit"


class BlockTap:
    """The hook of a checked pass: runs the provider block, then the reference block and -- with more than one sequence -- the
    provider block on sequence 0 alone, each from the same state, and restores the provider's results before it returns them."""

    def __init__(self, plan, reference):
        self.plan, self.reference = plan, reference
        self.stats: Dict[str, BlockStats] = {}

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
        row, ref = self.plan.rows[name], self.reference[name]
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
            self._judge(stats, compared, _tensors(ref.compared(ref_out)), after, self._save(holders))
        if alone:
            self._load(holders, before)
            first = [(_rows(o, 0, 1) if i in block.per_row else o) for i, o in enumerate(operands)]
            alone_out = block(*first)
            same = bytes_equal(tuple(t[0:1] for t in compared), block.compared(alone_out))
            for holder, full, single in zip(holders, after, self._save(holders)):
                at = 1 if hasattr(holder, "conv_storage") else 0  # a slot pool keeps sequence 0 in slot 1, a cache in row 0
                same = same and bytes_equal(tuple(t[at] for t in full), tuple(t[at] for t in single))
            stats.batch_calls += 1
            stats.batch_equal += int(same)
        self._load(holders, after)
        return out

    @staticmethod
    def _judge(stats: BlockStats, out, ref, state, ref_state) -> None:
        if stats.band == "gdn_step":
            equal = int((out[0] == ref[0]).sum())
            stats.equal_bytes += equal
            stats.total_bytes += out[0].numel()
            stats.worst_call = min(stats.worst_call, equal / out[0].numel())
            (recurrent, conv), (ref_recurrent, ref_conv) = state[0], ref_state[0]
            distance = (recurrent - ref_recurrent).abs() / max(1.0, float(ref_recurrent.abs().amax()))
            stats.state_error = max(stats.state_error, float(distance.amax()))
            stats.state_rms = max(stats.state_rms, rms_ratio(recurrent, ref_recurrent))
            stats.state_inside += int((distance <= GDN_STATE_BAND).sum())
            stats.state_elements += distance.numel()
            stats.conv_equal = stats.conv_equal and bytes_equal(conv, ref_conv)
            stats.ok = (
                stats.equal_bytes > GDN_EQUAL_BYTES * stats.total_bytes
                and stats.state_inside > GDN_STATE_SHARE * stats.state_elements
                and stats.state_error <= GDN_STATE_CAP
                and stats.conv_equal
            )
            return
        for o, r in zip(out, ref):
            if stats.band == "bitwise" or o.dtype != torch.bfloat16:
                stats.ok = stats.ok and bytes_equal(o, r)
                continue
            units = bf16_units(o, r)
            stats.elements += units.numel()
            stats.off_by_one += int((units >= 0.5).sum())
            stats.worst_units = max(stats.worst_units, float(units.amax()))
            stats.ok = stats.ok and stats.worst_units <= 1.0


@dataclass
class Checks:
    lines: List[str]
    ok: bool

    def text(self) -> str:
        return "\n".join(self.lines)


def _verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def run_checks(session: Session, prompt_tokens: int) -> Checks:
    """Replay, blocks, tokens and batch invariance for the run ``session`` has just recorded.  Leaves the session reset."""
    main = session.record
    steps = min(main.count, main.inputs.shape[0])
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
    )
    reference.load()

    tap = BlockTap(session.plan, reference.plan)
    session.reset(fresh=True)
    session.tap = tap
    try:
        for t in range(steps):
            session._advance(inputs[t])
    finally:
        session.tap = None
    replay_ok = bytes_equal(session.record.logits[:steps], logits) and bytes_equal(session.record.outputs[:steps], ids)

    reference.reset(fresh=True)
    for t in range(steps):
        reference._advance(inputs[t])
    ref_logits = reference.record.logits[:steps]
    distance = float((logits.float() - ref_logits.float()).abs().amax())
    band = LOGITS_FACTOR * distance
    decidable = margin(ref_logits) > band
    equal = ids == reference.record.outputs[:steps]
    wrong = int((decidable & ~equal).sum())
    count, near = int(decidable.sum()), int((~decidable).sum())
    tokens_ok = wrong == 0

    judged = {n: s for n, s in tap.stats.items() if s.band != "same"}
    blocks_ok = all(s.ok for s in judged.values())
    lines = [
        f"checks ({session.sequences} sequence{'s' if session.sequences > 1 else ''}, {steps} steps: {prompt_tokens} prompt + {steps - prompt_tokens} decode)"
    ]
    lines.append(f"  replay            ids and logits byte-identical on a second run from a fresh state    {_verdict(replay_ok)}")
    inside = sum(s.ok for s in judged.values())
    lines.append(
        f"  blocks            {inside} of {len(judged)} provider blocks inside their bands against the {session.reference_precision}-accumulated "
        f"reference, teacher-forced    {_verdict(blocks_ok)}"
    )
    for name, s in judged.items():
        lines.append(f"    {name:<22} {s.calls:>4} calls  {s.line()}  [band: {BANDS[s.band]}]  {_verdict(s.ok)}")
    lines.append(
        f"  tokens            {count - wrong} of {count} decidable greedy tokens equal to the reference; {near} near-ties below the logits band "
        f"({LOGITS_FACTOR:g} x {distance:.4g}, the largest logit distance of this run)    {_verdict(tokens_ok)}"
    )
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
        lines.append(f"  batch invariance  sequence 0 alone vs among {session.sequences}: {gate_text}; {rest}    {verdict}")
    else:
        lines.append("  batch invariance  not applicable at 1 sequence")
    session.reset(fresh=True)
    return Checks(lines, replay_ok and blocks_ok and tokens_ok and batch_ok)
