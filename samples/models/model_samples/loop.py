# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``Session``: plan, load, prefill, step for a fixed set of sequences.

A session decides everything before it loads anything: ``plan_table()`` resolves the family's provider table for this GPU, the
number of sequences and the stack, and states the memory the run needs -- for this session and for the sessions that will
follow it on its weights, so that nothing can decline once the first of them has loaded.  Only a run that fits then lets its
entries build their kernels on scratch operands (their first use, where an entry asks what it needs of the installed
toolchain): the memory sentence comes before the first allocation on the GPU.  ``load()`` then reads (or draws) and
merges the weights and keeps each in the forms the blocks of these plans read: the stored form of a weight that a block
multiplies on (``Block.stored_weights``), the bf16 copy of a weight that a block reads as one -- so no copy of a weight
that every plan of the run multiplies on its stored format.  A block that would read a form the load did not keep fails
there, by name, before the first step.  ``load()`` then compiles what a provider compiles (a compiled form is launched
once) and runs one warm-up step, each as a timed stage.  The reference session of a run's checks reads bf16 copies; where
the load kept none it gets one per call (``on_demand``).  ``prefill`` feeds the prompts and ``step`` advances every
sequence by one token; the chosen ids stay on the GPU, and the one host read of a step is the caller's (``read``), in the
sample's own loop -- there is no ``generate()`` here.  ``trace`` replays the first steps of the recorded run under the profiler for the report's GPU-side
lines (``timing``); it leaves the run's state and record as they were.  Asked for launch records, the same pass also keeps
what the kernel check needs (``census.Traced``): the sites of a step in wiring order, the log of every block call of the
traced steps and the profiler's record of every launch -- and, for a stack that declares a target kernel set, what the run can
say about that set (``claim``).

The set of sequences is fixed before the first step: a sequence that reaches its end token stays in the batch, so the number of
rows, and with it the kernel path, does not change inside a run.  A session keeps no state between two prompt sets other than
what ``reset`` clears.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from . import blocks, census
from .sampling import GREEDY
from .state import DecodeState
from .timing import Report, Stages, now, trace_steps

STACKS = ("fe-cublas", "native", "fe-entries", "reference")
REFERENCE = "reference"  # the stack of the plain-PyTorch reference blocks: what the checks of a run compare it with
PREFILL_ROUTES = ("decode-step",)


class StopRule:
    """When a sequence ends: at one of the end tokens or after ``max_new_tokens``; ``force_tokens`` ignores the end token and
    runs exactly that many steps (the protocol of a timed run)."""

    def __init__(self, eos_ids: Sequence[int], max_new_tokens: int = 256, force_tokens: Optional[int] = None):
        if max_new_tokens < 1 or (force_tokens is not None and force_tokens < 1):
            raise ValueError("a stop rule needs at least one new token")
        self.eos_ids = frozenset(int(i) for i in eos_ids)
        self.max_new_tokens, self.force_tokens = max_new_tokens, force_tokens

    @property
    def limit(self) -> int:
        return self.force_tokens if self.force_tokens is not None else self.max_new_tokens

    def ends(self, token: int) -> bool:
        return self.force_tokens is None and int(token) in self.eos_ids


class Record:
    """The ids fed to, the logits of and the ids chosen by the first steps of a run, kept on the GPU for the checks."""

    def __init__(self, steps: int, sequences: int, vocab: int, device):
        self.inputs = torch.zeros((steps, sequences), dtype=torch.int32, device=device)
        self.outputs = torch.zeros((steps, sequences), dtype=torch.int32, device=device)
        self.logits = torch.zeros((steps, sequences, vocab), dtype=torch.bfloat16, device=device)
        self.count = 0

    def add(self, ids: torch.Tensor, logits: torch.Tensor, next_ids: torch.Tensor) -> None:
        if self.count < self.inputs.shape[0]:
            self.inputs[self.count].copy_(ids)
            self.outputs[self.count].copy_(next_ids)
            self.logits[self.count].copy_(logits)
        self.count += 1


class Session:
    def __init__(
        self,
        model,
        text,
        sequences: int = 1,
        sampling=GREEDY,
        stack: str = "fe-cublas",
        prefill: str = "decode-step",
        device="cuda:0",
        overrides: Optional[dict] = None,
        reference_precision: str = "fp64",
        capacity: int = 256,
        checkpoint=None,
        seed: int = 0,
        merge_rule: str = "largest_scale",
        share: Optional["Session"] = None,
        record_steps: Optional[int] = None,
        check_kernels: bool = False,
        on_demand=None,
    ):
        """``capacity`` is the number of positions (prompt + new tokens) a sequence may reach.  ``checkpoint`` defaults to the
        model's seeded weights.  ``share`` reuses another session's loaded weights (the reference session of the tie-out).
        ``record_steps`` bounds the steps whose ids and logits are kept on the GPU for the checks (default: every step).
        ``check_kernels`` says that the run will check its kernels: what that needs of a provider is then part of the plan
        (a decline is a SKIP before the load), and the traced pass keeps its launch records.  A stack that declares a target
        kernel set (the family's ``target_stacks``) always checks its kernels: its statement is about them.
        ``on_demand`` (``weights.OnDemand``) is for the reference session of a run's checks, which runs on the weights of the
        session it checks: where that load kept a weight on its stored format alone, the blocks of this session get a bf16
        copy of it made for the one call and dropped after it, and the object counts what that cost.  It is refused for
        any other session (``ValueError``): a copy made per call is a cast and two synchronisations in front of a block,
        which no step of a run that is timed, traced or checked may carry."""
        if sampling is not GREEDY:
            raise ValueError("this stack chooses tokens greedily; pass sampling=GREEDY")
        if on_demand is not None and (share is None or stack != REFERENCE):
            raise ValueError(
                f"on_demand is for the reference session of a run's checks (stack {REFERENCE}, on the weights of the session it checks: share=); "
                f"a session of stack {stack}{'' if share is None else ' on shared weights'} reads the forms its load kept and never dequantises a weight per call"
            )
        if prefill not in PREFILL_ROUTES:
            raise ValueError(f"prefill route must be one of {PREFILL_ROUTES}; got {prefill!r}")
        if sequences < 1:
            raise ValueError(f"a session needs at least one sequence; got {sequences}")
        self.model, self.text, self.sequences, self.stack, self.prefill_route = model, text, int(sequences), stack, prefill
        self.device = torch.device(device)
        self.overrides, self.reference_precision, self.capacity = dict(overrides or {}), reference_precision, int(capacity)
        self.checkpoint, self.seed, self.merge_rule, self._share = checkpoint, seed, merge_rule, share
        self.record_steps = self.capacity if record_steps is None else max(1, min(int(record_steps), self.capacity))
        self.check_kernels = bool(check_kernels) or stack in getattr(model, "target_stacks", ())
        self.on_demand = on_demand  # the maker of the bf16 copies this session's blocks read and its weights do not hold
        self.traced: Optional[census.Traced] = None  # of the last traced pass that kept its launch records
        self._call_log: Optional[list] = None  # the block calls of the traced step that is running
        self.stages = Stages(sync=lambda: torch.cuda.synchronize(self.device) if self.device.type == "cuda" else None)
        self.plan: Optional[blocks.Plan] = None
        self.weights = None
        self.state: Optional[DecodeState] = None
        self.record: Optional[Record] = None
        self.tap = None  # tie_out's hook: tap(block name, layer, block, operands) -> the block's result
        self._followers: tuple = ()  # the sessions that will run on this one's weights, as ``plan_table`` was told
        self._report = Report(model.name, reduced=bool(getattr(model, "reductions", ())))
        self._calls = 0
        self._prefill_start = 0.0

    # ------------------------------------------------------------------ plan
    def resolve(self) -> blocks.Plan:
        if self.plan is None:
            options = {
                "reference_precision": self.reference_precision,
                "device": str(self.device),
                "check_kernels": self.check_kernels,
                "merge_rule": self.merge_rule,
            }
            facts = self.device if self.device.type == "cuda" else blocks.DeviceFacts((0, 0), "cpu")
            self.plan = blocks.plan(
                self.model.provider_table,
                self.model.regimes,
                facts,
                self.sequences,
                1,
                self.stack,
                self.overrides,
                model=self.model,
                options=options,
                source=self.model.table_source,
            )
        return self.plan

    def stored_weights(self, own: bool = False) -> tuple:
        """The blocks whose weights this session's load keeps on their stored format: every block that a provider
        multiplying on the stored format serves (``Block.stored_weights``) in this session's plan or in the plan of a
        session that will run on its weights (``plan_table(later)``).  ``own``: the blocks of this session's plan alone
        -- the weights its own steps read on the stored format."""
        plans = [self.resolve(), *(session.resolve() for session in (() if own else self._followers))]
        return tuple(dict.fromkeys(name for plan in plans for name, block in plan.items() if getattr(block, "stored_weights", False)))

    def uncopied_weights(self) -> tuple:
        """The blocks whose weights this session's load keeps on their stored format ALONE, without a bf16 copy: a provider
        multiplying on the stored format serves them in this session's plan and in the plan of every session that will
        run on its weights (``plan_table(later)``), so no block of the run reads a copy.  The reference blocks of the
        checks do, and get theirs per call (``on_demand``)."""
        plans = [self.resolve(), *(session.resolve() for session in self._followers)]
        return tuple(name for name in self.stored_weights() if all(getattr(plan.get(name), "stored_weights", False) for plan in plans))

    def _kept(self) -> dict:
        """What the plans of the run ask the family's loader to keep, as its keyword arguments: nothing for a run whose every
        multiply reads a bf16 copy (a family without such a block is never asked)."""
        stored, uncopied = self.stored_weights(), self.uncopied_weights()
        return {**({"stored": stored} if stored else {}), **({"uncopied": uncopied} if uncopied else {})}

    def _check_forms(self) -> None:
        """Before the first step: every form of a weight a block of this session's plan reads is one the load of its weights
        kept.  A session that runs on another session's weights was not necessarily planned with it; a ``ValueError`` then
        names the block and the form."""
        kept, uncopied = getattr(self.weights, "stored", None), getattr(self.weights, "uncopied", ())
        if kept is None:  # a family whose loader holds one form of every weight
            return
        other = "the weights of this session were loaded for another plan (plan every session of a run before the first load: Session.plan_table)"
        for name, block in self.plan.items():
            on_stored = getattr(block, "stored_weights", False)
            if on_stored and name not in kept:
                raise ValueError(f"{block.entry} multiplies on the stored format of the weight of {name}, and the load kept none: {other}")
            if not on_stored and name in uncopied and self.on_demand is None:
                raise ValueError(f"{block.entry} reads a bf16 copy of the weight of {name}, and the load kept its stored format alone: {other}")

    def claim(self) -> Optional[census.Claim]:
        """What this session's run can say about the target kernel set of its stack -- the identity of every block call, the
        three numbers, the reasons its GPU class is not claimed (``census.Claim``) -- or ``None`` for a stack that declares no
        target.  Decided from the plan and the family's census file: before anything is loaded.  ``ValueError`` when that
        file is not a census of this model."""
        if self.stack not in getattr(self.model, "target_stacks", ()):
            return None
        return self.model.target().claim(self.resolve(), self.model)

    def plan_table(self, later: Sequence["Session"] = ()) -> str:
        """The served-by table and the memory sentence; raises ``NotImplementedError`` (a SKIP) before anything is loaded.

        ``later`` are the sessions that will run after this one in the same process on its weights (``share=self``).  Their
        plans are resolved here too, and the memory sentence then covers the whole run -- this session's need plus their
        state -- against what is free now: every decline of a run comes before its first load, none after it.

        In this order: the tables of every session are resolved, which allocates nothing; the memory sentence says whether
        the whole run fits; and only then do the entries of each plan make their first use (``Plan.first_use``: an entry
        builds its kernel on scratch operands as large as the run's, and asks there what it needs of the installed
        toolchain).  A run that does not fit is a SKIP with its estimate, and nothing was allocated for it."""
        plan = self.resolve()
        for session in later:
            session.resolve()
        self._followers = tuple(later)
        sentence = ""
        if self.device.type == "cuda":
            # what the load of the first session holds follows the plans: the stored format of the weights a block multiplies
            # on, and no bf16 copy of those no block reads as one (asked of a family's model only where a plan has such a block)
            held = self._kept() if self._share is None else {}
            need = self.model.memory_bytes(self.sequences, self.capacity, **held) - (self.model.copy_bytes() if self._share is not None else 0)
            need += sum(session.model.state_bytes(session.sequences, session.capacity) for session in later)
            counts = [self.sequences, *(session.sequences for session in later)]
            what = f" for {', '.join(str(n) for n in counts[:-1])} and {counts[-1]} sequences, one count after the other" if later else ""
            # the part of the need that is the checks': the reference's copy of a weight the load keeps no copy of, one at a time
            checks = self.model.check_bytes(held["uncopied"]) if "uncopied" in held else 0
            sentence = "\n  " + plan.memory(need, what, checks)  # before anything is built on the GPU: this is where a run that does not fit ends
        for session in (self, *later):
            session.plan.first_use()
        return plan.table() + sentence

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        plan = self.resolve()
        plan.first_use()  # made with the table (``plan_table``); a session that is loaded without one makes it here, before the weights
        if self.reference_precision == "ieee-fp32" and hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "ieee"
        if self._share is not None:
            self.weights = self._share.weights
        else:
            if self.checkpoint is None:
                from .weights import Checkpoint

                self.checkpoint = Checkpoint.synthetic(self.model, self.seed)
            # the forms the plans of the run read, asked of a family's loader only where a block multiplies on a stored format
            self.weights = self.model.load(self.checkpoint, self.device, self.stages, self.merge_rule, **self._kept())
        self._check_forms()
        for name, block in plan.items():
            if getattr(block, "compiles", False):
                with self.stages.stage(f"compile and first launch {name} ({self.sequences} row{'s' if self.sequences > 1 else ''})"):
                    block.compile(self.sequences, 1)
        self.reset(fresh=True)
        with self.stages.stage("warm-up"):
            self.model.step(self._call, self.weights, self.state, torch.zeros(self.sequences, dtype=torch.int32, device=self.device))
        self._report.calls_per_step = self._calls
        self.reset(fresh=True)

    def reset(self, fresh: bool = True) -> None:
        """Back to the start of a prompt set: zero state at position 0 and an empty record; ``fresh`` allocates both anew."""
        if fresh or self.state is None:
            self.state = DecodeState(self.model.geometry.layer_kinds, self.sequences, self.capacity, self.model.state_shapes(), self.device)
            self.record = Record(self.record_steps, self.sequences, self.model.geometry.vocab, self.device)
        else:
            self.state.position = 0
            for layer in self.state.layers:
                for t in layer.tensors():
                    t.zero_()
            self.record.count = 0
        self._calls = 0
        self._report.clock.launch.clear()
        self._report.clock.ids.clear()

    # ------------------------------------------------------------------ run
    def _call(self, name: str, layer: int, *operands):
        self._calls += 1
        block = self.plan[name]
        if self.on_demand is not None:  # a weight the load kept without a bf16 copy: this session's blocks get one for this call
            operands = self.on_demand.operands(operands)
        out = self.tap(name, layer, block, operands) if self.tap is not None else block(*operands)
        if self._call_log is not None:  # a traced step: the call site, and what the adapter says served the call it has just made
            self._call_log.append(census.Call(name, layer, self.sequences, *block.serving()))
        return out

    def _advance(self, ids: torch.Tensor) -> torch.Tensor:
        if self.state.position >= self.capacity:
            raise ValueError(f"the session was built for {self.capacity} positions per sequence; build it with a larger capacity")
        next_ids, logits = self.model.step(self._call, self.weights, self.state, ids)
        self.record.add(ids, logits, next_ids)
        return next_ids

    def prefill(self, prompts: List[torch.Tensor]) -> torch.Tensor:
        """Feed one prompt per sequence and return the first new ids ``[sequences]`` on the GPU.  Route ``decode-step``: every
        prompt token goes through the decode step, one step per token, so the prompts of one run have one length."""
        if len(prompts) != self.sequences:
            raise ValueError(f"this session has {self.sequences} sequences; got {len(prompts)} prompts")
        lengths = {int(p.numel()) for p in prompts}
        if len(lengths) != 1 or 0 in lengths:
            raise ValueError(f"route decode-step feeds the prompts in lockstep: they need one non-zero length; got {sorted(lengths)}")
        self._prefill_start = now()
        ids = torch.stack([p.to(torch.int32) for p in prompts]).to(self.device)
        self._report.text_path["to_device"] = now() - self._prefill_start
        self._report.prompt_tokens = ids.shape[1]
        self._report.prefill_route = "decode step"
        next_ids = None
        for t in range(ids.shape[1]):
            next_ids = self._advance(ids[:, t].contiguous())
        return next_ids

    def step(self, ids: torch.Tensor) -> torch.Tensor:
        """One decode step for every sequence: the ids to feed ``[sequences]`` -> the next ids ``[sequences]``, on the GPU."""
        self._report.clock.mark_launch()
        return self._advance(ids)

    def read(self, ids: torch.Tensor) -> List[int]:
        """The one host read of a step: the chosen ids, 4 bytes per sequence."""
        tokens = ids.tolist()
        if not self._report.clock.ids and not self._report.clock.launch:
            self._report.prefill_seconds = now() - self._prefill_start
        else:
            self._report.clock.mark_ids()
        return tokens

    def trace(self, steps: int = 32, records: Optional[bool] = None) -> None:
        """The traced pass of the report: replay the first ``steps`` recorded inputs of this run from a zero state under the
        profiler, each step ending in the host read of its ids, as in the loop.  The run's own state and record are put
        back afterwards.  Where kernel activity records cannot be had the report says why.

        With ``records`` (default: the session's ``check_kernels``) the pass also keeps ``self.traced`` for the kernel check:
        the sites of a step with what the entries state about their launches, read before the first traced step; the log
        of every block call of every traced step; and the launch records of the pass, cut into steps -- or why there are
        none."""
        if self.on_demand is not None:  # the reference of a run's checks: its calls carry the copies it makes, which no step of a run does
            raise ValueError(
                "a session that makes bf16 copies of its weights per call (on_demand) is the reference of a run's checks; its steps are not traced"
            )
        records = self.check_kernels if records is None else bool(records)
        sites = census.expected(self.plan, self.model) if records else []
        claim = self.claim() if records else None
        count = min(steps, self.record.count, self.record.inputs.shape[0])
        if count < 1:
            self._report.trace_missing = "needs a recorded step on a CUDA device"
            if records:
                self.traced = census.Traced(sites, [], missing=self._report.trace_missing, rows=self.sequences, claim=claim)
            return
        inputs, kept, calls = self.record.inputs[:count], (self.state, self.record), self._calls
        self.state = DecodeState(self.model.geometry.layer_kinds, self.sequences, self.capacity, self.model.state_shapes(), self.device)
        self.record = Record(0, self.sequences, self.model.geometry.vocab, self.device)
        call_log: List[list] = []
        launches: list = []

        def step(i: int) -> None:
            if records:
                self._call_log = []
                call_log.append(self._call_log)
            self._advance(inputs[i]).tolist()

        try:  # the device is trace_steps' to decline: what is swapped here is put back whatever it does
            if records:
                self._report.trace, self._report.trace_missing = trace_steps(step, count, self.device, records=launches)
            else:
                self._report.trace, self._report.trace_missing = trace_steps(step, count, self.device)
        finally:
            (self.state, self.record), self._calls, self._call_log = kept, calls, None
        if records:
            self.traced = census.Traced.of(sites, call_log, launches, count, self._report.trace_missing, self.sequences, claim)

    def new_report(self) -> None:
        """A fresh report for the next prompt set of this session; the block calls per step stay the session's."""
        calls = self._report.calls_per_step
        self._report = Report(self.model.name, reduced=bool(getattr(self.model, "reductions", ())))
        self._report.calls_per_step = calls

    def report(self) -> Report:
        if hasattr(self.model, "bytes_per_step") and self.state is not None:
            # a block of this session's plan that multiplies on the stored format reads its weight there, not as a copy
            # (asked of a family's model only where the plan has such a block)
            stored = self.stored_weights(own=True)
            read = {"stored": stored} if stored else {}
            self._report.bytes = self.model.bytes_per_step(self.sequences, self.state.position, **read)
        return self._report
