# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host timestamps of a sample run and the text of its report.

One monotonic host clock.  Per step, ``t_launch`` is taken just before the step's first launch and ``t_ids`` when the chosen ids
are back on the host:

* **period** = ``t_launch(i + 1) - t_launch(i)``: the step as the loop pays for it;
* **in the step** = ``t_ids(i) - t_launch(i)``;
* **turnaround** = ``t_launch(i + 1) - t_ids(i)``: detokenise, print, the stop check.

These are host-side timings for orientation only: no clock is locked and nothing here measures a kernel.  A run on a reduced
model prefixes its times with "reduced model" and prints no rate.  GPU-side quantities (GPU busy, idle inside the step, launches
per step) need kernel activity records and arrive with the report of a real checkpoint.
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from typing import Dict, List, Optional


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


class Report:
    """What a run prints after its loop: the text path, the first-token line with its route, the step timings."""

    def __init__(self, tag: str, reduced: bool):
        self.tag, self.reduced = tag, reduced
        self.text_path: Dict[str, float] = {}
        self.prompt_tokens = 0
        self.prefill_route = "decode step"
        self.prefill_seconds = 0.0
        self.clock = StepClock()
        self.calls_per_step = 0
        self.notes: List[str] = []

    def text(self) -> str:
        prefix = "reduced model: " if self.reduced else ""
        lines = []
        if self.text_path:
            tp = self.text_path
            per_token = tp.get("tokenize", 0.0) * 1e6 / max(self.prompt_tokens, 1)
            lines.append(
                f"text path             render {tp.get('render', 0.0) * 1e3:.3f} ms | tokenize {tp.get('tokenize', 0.0) * 1e3:.3f} ms = "
                f"{per_token:.2f} us per prompt token | ids to GPU {tp.get('to_device', 0.0) * 1e3:.3f} ms | text out {tp.get('text_out', 0.0) * 1e6:.2f} us per token"
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
        if self.reduced or s is None:
            lines.append("rate                  not printed for a reduced model")
        else:
            wall = self.clock.ids[-1] - self.clock.launch[0]
            lines.append(
                f"rate                  {len(self.clock.ids) / wall:.1f} tokens per second per sequence = {len(self.clock.ids)} tokens / {wall:.3f} s of the decode loop"
            )
        lines.append("note                  this stack's step is bound by the Python launch path: no floor and no speed-of-light figure is printed")
        lines.extend(self.notes)
        return "\n".join(lines)
