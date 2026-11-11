# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host timestamps of a sample run and the text of its report.

One monotonic host clock.  Per step, ``t_launch`` is taken just before the step's first launch and ``t_ids`` when the chosen ids
are back on the host:

* **period** = ``t_launch(i + 1) - t_launch(i)``: the step as the loop pays for it;
* **in the step** = ``t_ids(i) - t_launch(i)``;
* **turnaround** = ``t_launch(i + 1) - t_ids(i)``: detokenise, print, the stop check.

and **rate** = tokens produced / wall time of the decode loop, per sequence: the period seen from the other side, never a
median step turned into a rate.

The **text path** is timed where it runs: **render** = chat messages to prompt text, **tokenize** = prompt text to ids
(also per prompt token), **ids to GPU** = the copy of the prompt's ids to the device, **text out** = ids to text per
produced token, inside the loop, so with whatever the step left in the processor's caches.  With a checkpoint's template and
tokenizer the first call of a process pays for caches the library builds; the same call repeated is printed beside it.
The **first token** line states the prefill route: while the prompt goes through the decode step, one step per token, its
time is not a time to first token of record.

The GPU side comes from one **traced pass**: the first steps of the run are replayed from a zero state under the PyTorch
profiler with kernel activity records only (no CPU-side records, so the pass adds little to a step).  A step of this loop
ends in its one host read, which the records show as one device-to-host copy; the records between two such copies are one
step.  Per step:

* **GPU busy** = the union of the intervals of every kernel and copy of the step;
* **idle inside the step** = from the start of the step's first record to the end of its last, less GPU busy: time in
  which the GPU waited for the next launch;
* **launches** = kernels of the step, and copies and memory sets beside them;
* **host-exposed** = period - GPU busy: what the loop pays beyond the kernels' own time (the period is the untraced one;
  the traced period is printed beside it).

No pair of CUDA events is used for any of these: an event is stamped when the stream reaches it, so a pair spans every gap
in which the GPU waited for the host.

**Bytes per step** is arithmetic on the model's tensor list, not a measurement: the bytes of the weights a step reads in
their stored formats plus the state it reads and writes ("model"), against the same with the weights as the running stack
holds them ("this stack moves").

These are host-side timings and counts for orientation only: no clock is locked, and nothing here is a measurement of
record.  A run on a reduced model prefixes its times with "reduced model" and prints no rate.
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


def now() -> float:
    return time.perf_counter()


class Stages:
    """Seconds per named stage of a load (draw or read, merge, dequantise, compile, warm-up), accumulated in first-use order."""

    def __init__(self, sync=None):
        self.seconds: Dict[str, float] = {}
        self._sync = sync  # called at both ends of a stage, so a stage is charged its own device work

    @contextmanager
    def stage(self, name: str):
        if self._sync is not None:
            self._sync()
        start = now()
        try:
            yield
        finally:
            if self._sync is not None:
                self._sync()
            self.seconds[name] = self.seconds.get(name, 0.0) + now() - start

    def add(self, name: str, seconds: float) -> None:
        self.seconds[name] = self.seconds.get(name, 0.0) + seconds

    def line(self) -> str:
        return "stage  " + " | ".join(f"{name} {seconds:.1f} s" for name, seconds in self.seconds.items())


class StepClock:
    """The two timestamps of every step of one decode loop."""

    def __init__(self):
        self.launch: List[float] = []
        self.ids: List[float] = []

    def mark_launch(self) -> None:
        self.launch.append(now())

    def mark_ids(self) -> None:
        self.ids.append(now())

    def summary(self) -> Optional[dict]:
        """Median and mean period, time in the step and turnaround in ms over the steps that have a successor; None below 2 steps."""
        n = min(len(self.launch) - 1, len(self.ids))
        if n < 1:
            return None
        period = [(self.launch[i + 1] - self.launch[i]) * 1e3 for i in range(n)]
        inside = [(self.ids[i] - self.launch[i]) * 1e3 for i in range(n)]
        around = [(self.launch[i + 1] - self.ids[i]) * 1e3 for i in range(n)]
        return {
            "steps": n,
            "period_median": statistics.median(period),
            "period_mean": statistics.fmean(period),
            "in_step_median": statistics.median(inside),
            "turnaround_median": statistics.median(around),
        }


@dataclass
class Trace:
    """Medians over the steps of one traced pass (milliseconds; counts per step)."""

    steps: int
    busy: float
    idle: float
    kernels: int
    copies: int
    period: float  # the host-side period under the profiler


def step_facts(records: List[Tuple[str, float, float]]) -> List[Tuple[float, float, int, int]]:
    """Per step ``(GPU busy, idle inside the step, kernels, copies)`` from kernel activity records ``(name, start, end)`` on
    one clock.  A step ends with its host read: the record of a device-to-host copy closes it.  Records after the last
    such copy belong to no step."""
    steps, current = [], []
    for name, start, end in sorted(records, key=lambda r: r[1]):
        current.append((name, start, end))
        if name.startswith("Memcpy DtoH"):
            busy, reach = 0.0, None
            for _, a, b in current:  # the union of the intervals, in start order
                if reach is None or a > reach:
                    busy, reach = busy + (b - a), b
                elif b > reach:
                    busy, reach = busy + (b - reach), b
            span = max(b for _, _, b in current) - current[0][1]
            copies = sum(n.startswith(("Memcpy", "Memset")) for n, _, _ in current)
            steps.append((busy, span - busy, len(current) - copies, copies))
            current = []
    return steps


def trace_steps(step: Callable[[int], None], count: int) -> Tuple[Optional[Trace], str]:
    """Run ``step(i)`` for ``i`` in ``range(count)`` under the PyTorch profiler with kernel activity records only and return
    ``(Trace, "")``, or ``(None, reason)`` where the records cannot be had.  ``step`` ends in the loop's one host read."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    if ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        return None, "this PyTorch build records no kernel activity"

    def failed(error) -> Tuple[None, str]:
        return None, f"the profiler failed ({(str(error).splitlines() or [type(error).__name__])[0][:120]})"

    periods = []
    profiler = profile(activities=[ProfilerActivity.CUDA])
    try:
        profiler.__enter__()
    except RuntimeError as error:  # the profiler could not start its kernel activity records
        return failed(error)
    try:  # an error of a step is the step's, not the profiler's: it is not caught here
        for i in range(count):
            start = now()
            step(i)
            periods.append((now() - start) * 1e3)
    except BaseException:
        try:
            profiler.__exit__(None, None, None)
        except RuntimeError:
            pass
        raise
    try:
        profiler.__exit__(None, None, None)
        device_type = torch.autograd.DeviceType.CUDA
        records = [(e.name, e.time_range.start / 1e3, e.time_range.end / 1e3) for e in profiler.events() if e.device_type == device_type]
    except RuntimeError as error:  # the profiler could not stop, or has no records to hand over
        return failed(error)
    facts = step_facts(records)
    if not records:
        return None, "the profiler recorded no kernel activity on this GPU"
    if len(facts) != count:
        return None, f"the records show {len(facts)} host reads for {count} steps"
    median = statistics.median
    return (
        Trace(
            count, median(f[0] for f in facts), median(f[1] for f in facts), int(median(f[2] for f in facts)), int(median(f[3] for f in facts)), median(periods)
        ),
        "",
    )


class Report:
    """What a run prints after its loop: the text path, the first-token line with its route, the step timings, the traced
    pass and the bytes of a step."""

    def __init__(self, tag: str, reduced: bool):
        self.tag, self.reduced = tag, reduced
        self.text_path: Dict[str, float] = {}
        self.prompt_tokens = 0
        self.prefill_route = "decode step"
        self.prefill_seconds = 0.0
        self.clock = StepClock()
        self.calls_per_step = 0
        self.trace: Optional[Trace] = None
        self.trace_missing = ""  # why there is no traced pass, when one was asked for
        self.bytes: Dict[str, int] = {}  # the model's bytes_per_step, when the family states them
        self.notes: List[str] = []

    def text(self) -> str:
        prefix = "reduced model: " if self.reduced else ""
        lines = []
        if self.text_path:
            tp = self.text_path
            tokenize = tp.get("tokenize_repeated", tp.get("tokenize", 0.0))
            per_token = tokenize * 1e6 / max(self.prompt_tokens, 1)
            if "render_repeated" in tp:  # a checkpoint's template and tokenizer: the first call of the process, and the call repeated
                render_text = f"render {tp.get('render', 0.0) * 1e3:.3f} ms first call, {tp['render_repeated'] * 1e3:.3f} repeated"
                tokenize_text = (
                    f"tokenize {tp.get('tokenize', 0.0) * 1e3:.3f} ms first call, {tokenize * 1e3:.3f} repeated = {per_token:.2f} us per prompt token"
                )
            else:
                render_text = f"render {tp.get('render', 0.0) * 1e3:.3f} ms"
                tokenize_text = f"tokenize {tokenize * 1e3:.3f} ms = {per_token:.2f} us per prompt token"
            lines.append(
                f"text path             {render_text} | {tokenize_text} | ids to GPU {tp.get('to_device', 0.0) * 1e3:.3f} ms | "
                f"text out {tp.get('text_out', 0.0) * 1e6:.2f} us per token"
            )
        lines.append(
            f"first token           {prefix}prefill route: {self.prefill_route}, {self.prompt_tokens} tokens x one step; "
            f"not a time to first token of record ({self.prefill_seconds * 1e3:.1f} ms)"
        )
        s = self.clock.summary()
        if s is not None:
            lines.append(
                f"steps ({len(self.clock.launch)})            {prefix}period {s['period_median']:.3f} ms median / {s['period_mean']:.3f} mean | "
                f"in the step {s['in_step_median']:.3f} ms | turnaround {s['turnaround_median']:.3f} ms | {self.calls_per_step} block calls per step"
            )
        if self.trace is not None:
            t = self.trace
            period = s["period_median"] if s is not None else t.period
            exposed = period - t.busy
            stretch = f" ({(t.period / period - 1) * 100:+.0f} %)" if s is not None and period > 0 else ""
            lines.append(
                f"traced pass ({t.steps})      {prefix}GPU busy {t.busy:.3f} ms | idle inside the step {t.idle:.3f} ms | "
                f"host-exposed {exposed:.3f} ms = period - GPU busy ({exposed / period:.0%} of the period) | "
                f"{t.kernels} kernel launches + {t.copies} copies per step | period under the profiler {t.period:.3f} ms{stretch}"
            )
        elif self.trace_missing:
            lines.append(f"traced pass           not available: {self.trace_missing}")
        if self.reduced:
            lines.append("rate                  not printed for a reduced model")
        elif s is None:
            lines.append("rate                  not printed below three decode steps")
        else:
            wall = self.clock.ids[-1] - self.clock.launch[0]
            lines.append(
                f"rate                  {len(self.clock.ids) / wall:.1f} tokens per second per sequence = {len(self.clock.ids)} tokens / {wall:.3f} s of the decode loop"
            )
        if self.bytes:
            gb = 1e9
            model = self.bytes["weights"] + self.bytes["state"]
            moved = self.bytes["copies"] + self.bytes["state"]
            lines.append(
                f"bytes per step        {prefix}model {model / gb:.2f} GB (weights {self.bytes['weights'] / gb:.2f} in their stored formats, state and cache "
                f"{self.bytes['state'] / gb:.2f}); this stack moves {moved / gb:.2f} GB ({moved / model:.2f}x): bf16 copies of the weights (arithmetic, not a measurement)"
            )
        lines.append("note                  this stack's step is bound by the Python launch path: no floor and no speed-of-light figure is printed")
        lines.extend(self.notes)
        return "\n".join(lines)
