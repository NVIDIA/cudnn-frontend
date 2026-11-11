# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``Session``: plan, load, prefill, step for a fixed set of sequences.

A session decides everything before it loads anything: ``plan_table()`` resolves the family's provider table for this GPU, the
number of sequences and the stack, and states the memory the run needs -- for this session and for the sessions that will
follow it on its weights, so that nothing can decline once the first of them has loaded.  ``load()`` then reads (or draws),
merges and dequantises the weights, compiles what a provider compiles (a compiled form is launched once) and runs one
warm-up step, each as a timed stage.  ``prefill`` feeds the prompts and ``step`` advances every sequence by one token; the
chosen ids stay on the GPU, and the one host read of a step is the caller's (``read``), in the sample's own loop -- there is
no ``generate()`` here.  ``trace`` replays the first steps of the recorded run under the profiler for the report's GPU-side
lines (``timing``); it leaves the run's state and record as they were.

The set of sequences is fixed before the first step: a sequence that reaches its end token stays in the batch, so the number of
rows, and with it the kernel path, does not change inside a run.  A session keeps no state between two prompt sets other than
what ``reset`` clears.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from . import blocks
from .sampling import GREEDY
from .state import DecodeState
from .timing import Report, Stages, now, trace_steps

STACKS = ("fe-cublas", "reference")
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
    ):
        """``capacity`` is the number of positions (prompt + new tokens) a sequence may reach.  ``checkpoint`` defaults to the
        model's seeded weights.  ``share`` reuses another session's loaded weights (the reference session of the tie-out).
        ``record_steps`` bounds the steps whose ids and logits are kept on the GPU for the checks (default: every step)."""
        if sampling is not GREEDY:
            raise ValueError("this stack chooses tokens greedily; pass sampling=GREEDY")
        if prefill not in PREFILL_ROUTES:
            raise ValueError(f"prefill route must be one of {PREFILL_ROUTES}; got {prefill!r}")
        if sequences < 1:
            raise ValueError(f"a session needs at least one sequence; got {sequences}")
        self.model, self.text, self.sequences, self.stack, self.prefill_route = model, text, int(sequences), stack, prefill
        self.device = torch.device(device)
        self.overrides, self.reference_precision, self.capacity = dict(overrides or {}), reference_precision, int(capacity)
        self.checkpoint, self.seed, self.merge_rule, self._share = checkpoint, seed, merge_rule, share
        self.record_steps = self.capacity if record_steps is None else max(1, min(int(record_steps), self.capacity))
        self.stages = Stages(sync=lambda: torch.cuda.synchronize(self.device) if self.device.type == "cuda" else None)
        self.plan: Optional[blocks.Plan] = None
        self.weights = None
        self.state: Optional[DecodeState] = None
        self.record: Optional[Record] = None
        self.tap = None  # tie_out's hook: tap(block name, layer, block, operands) -> the block's result
        self._report = Report(model.name, reduced=bool(getattr(model, "reductions", ())))
        self._calls = 0
        self._prefill_start = 0.0

    # ------------------------------------------------------------------ plan
    def resolve(self) -> blocks.Plan:
        if self.plan is None:
            options = {"reference_precision": self.reference_precision, "device": str(self.device)}
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

    def plan_table(self, later: Sequence["Session"] = ()) -> str:
        """The served-by table and the memory sentence; raises ``NotImplementedError`` (a SKIP) before anything is loaded.

        ``later`` are the sessions that will run after this one in the same process on its weights (``share=self``).  Their
        plans are resolved here too, and the memory sentence then covers the whole run -- this session's need plus their
        state -- against what is free now: every decline of a run comes before its first load, none after it."""
        plan = self.resolve()
        for session in later:
            session.resolve()
        if self.device.type != "cuda":
            return plan.table()
        need = self.model.memory_bytes(self.sequences, self.capacity) - (self.model.copy_bytes() if self._share is not None else 0)
        need += sum(session.model.state_bytes(session.sequences, session.capacity) for session in later)
        counts = [self.sequences, *(session.sequences for session in later)]
        what = f" for {', '.join(str(n) for n in counts[:-1])} and {counts[-1]} sequences, one count after the other" if later else ""
        return plan.table() + "\n  " + plan.memory(need, what)

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        plan = self.resolve()
        if self.reference_precision == "ieee-fp32" and hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "ieee"
        if self._share is not None:
            self.weights = self._share.weights
        else:
            if self.checkpoint is None:
                from .weights import Checkpoint

                self.checkpoint = Checkpoint.synthetic(self.model, self.seed)
            self.weights = self.model.load(self.checkpoint, self.device, self.stages, self.merge_rule)
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
        if self.tap is not None:
            return self.tap(name, layer, self.plan[name], operands)
        return self.plan[name](*operands)

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

    def trace(self, steps: int = 32) -> None:
        """The traced pass of the report: replay the first ``steps`` recorded inputs of this run from a zero state under the
        profiler, each step ending in the host read of its ids, as in the loop.  The run's own state and record are put
        back afterwards.  Where kernel activity records cannot be had the report says why."""
        count = min(steps, self.record.count, self.record.inputs.shape[0])
        if count < 1 or self.device.type != "cuda":
            self._report.trace_missing = "needs a recorded step on a CUDA device"
            return
        inputs, kept, calls = self.record.inputs[:count], (self.state, self.record), self._calls
        self.state = DecodeState(self.model.geometry.layer_kinds, self.sequences, self.capacity, self.model.state_shapes(), self.device)
        self.record = Record(0, self.sequences, self.model.geometry.vocab, self.device)
        try:
            self._report.trace, self._report.trace_missing = trace_steps(lambda i: self._advance(inputs[i]).tolist(), count)
        finally:
            (self.state, self.record), self._calls = kept, calls

    def new_report(self) -> None:
        """A fresh report for the next prompt set of this session; the block calls per step stay the session's."""
        calls = self._report.calls_per_step
        self._report = Report(self.model.name, reduced=bool(getattr(self.model, "reductions", ())))
        self._report.calls_per_step = calls

    def report(self) -> Report:
        if hasattr(self.model, "bytes_per_step") and self.state is not None:
            self._report.bytes = self.model.bytes_per_step(self.sequences, self.state.position)
        return self._report
