# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The performance mode of a model sample: timed requests of the decode loop, one definition per statistic, a report on the
screen and a report file.

A **timed request** is one prompt from a zero state, its first new token, then a fixed number of forced decode steps (the
end token is ignored).  A run of the mode (``run``) is one process and makes, in this order:

1. the **checked request**, with the recorder of the checks on, and the reference checks on it (``tie_out.run_checks``):
   replay over the whole request, every provider block against the reference at the **judged positions**
   (``judged_positions``), the tokens, the stored weights.  A check that fails ends the run: nothing is timed, no file;
2. a **warm-up request** and the **timed requests**, each from the state zeroed in place -- the device synchronised behind
   the zeroing, which is outside every statistic -- with nothing but the step inside: no recorder, no call log, no
   profiler, no text.  A child process samples the GPU beside them (``observe``): this process starts it in front of the
   warm-up request, reads the GPU's process list once more behind the last timed request, and makes no other call that
   concerns it before the traced request has ended;
3. the **traced request**: the prompt untraced, then the first timed positions under the profiler -- the first profiler
   pass of the process starts here, behind the last timed request.  Its first step is traced alone and sizes the rest:
   as many steps as the record budget holds;
4. the **checks after**: every timed request and the traced request produced the checked request's token ids, step for
   step; every timed request left the state in the same bytes; the stored weights still hold the bytes of their load; and
   the kernel check on the traced steps (``tie_out.kernel_check``);
5. the report: one data structure, written as the report file and rendered as the screen (``screen``).

**Stamps**, on one monotonic host clock (``timing.now``): ``t_request`` when the request is handed to the text path;
``t_first`` when the ids of the first new token are on the host; per forced step ``t_launch`` just before its first
launch call and ``t_token`` when its ids are on the host.  Between two host reads are the step's launch calls, its
kernels, the one host read and keeping the ids in a list; inside the time to first token are render, tokenize, the copy
of the ids, the prompt's decode steps and the first host read.  Everything else of a run is outside every statistic.
``DEFINITIONS`` holds the one definition of every statistic; the file carries each beside its number.

**The GPU side** comes from the launch records of the traced request and never from a timed one.  GPU busy of a step is
the union of the intervals of its kernel and copy records (``timing.step_facts``).  What a timed step leaves beside GPU
busy of the traced step at the same position and beside its own host time is formed step by step, before anything is
aggregated; the mean of it is the run's launch overhead (``step_split``), which adds up with the means of the same
steps: step = GPU busy + launch overhead + host.  It is an estimate -- a timed step has no GPU-side record, and GPU busy
is the traced request's, taken under the profiler, and is an upper bound -- printed as a term of the ``step`` line, and
called an estimate there, only where it is larger than the mean GPU busy; the median of the per-step remainders is
printed under that line and nowhere else, because one stalled step moves their mean: where the step line says that
the rest is not resolved, no figure of the remainder is printed.  The per-kernel table (``kernel_table``)
gives every row the time its launches add to the GPU's timeline (``added_ms``): a row is a block, found by its place in
the wiring by the kernel check's own walk (``census.Row``), and the blocks on fallback rows that no exact kernel
separates are one row.  A row says what serves it (``_row_kind``): the row of a block a library serves carries the
library's entry as the plan names it and holds the launches of the library's kernel; what a block's call launches
beside a kernel of its own is the row of its place in the wiring and is named so.  The added times of a step
partition the union, so the rows sum to GPU busy.  A step launched from Python says so over its table: a kernel's
time there is not its time inside a captured sequence.

**A run whose steps are replays** (a session that records its decode step: the stack ``native`` by default).  Its checked
request, its warm-up request and its timed requests are replays of the recorded step: one launch and one host read
per token.  The checked request is replayed with the recorder on, and the checks launch every step of it again from
Python: ids and logits of all its steps byte for byte, the step's own tail behind the request, and the digest of the
state the recomputation leaves against the replayed request's.  The traced request is launched from Python, as in
every run -- the kernel check and the per-kernel table need the call log, and the table says so --, and behind it a
second traced request replays its first steps under the profiler: their launch records are held to the traced
steps launched from Python (``census.recorded_step``), and their GPU busy is the one the ``step`` line prints
(``gpu_busy_replayed``).  A replayed step has no launch path from Python between its blocks, so no launch overhead
and no stretch of tracing is formed for such a run.

**What a run may call itself** (``record_class``).  A run is a ``controlled`` run only when it observed every condition
itself -- a checkpoint, no block on a fallback row, a captured launch sequence, the three statements of its ``checks``
lines, no other process on the GPU, the clocks -- and prints ``orientation`` with every reason otherwise.
The GPU's other processes and its clock are observed through the driver's tool (``observe``) or printed as not observed
with the reason; a run that is ``orientation`` says so on the second line of the screen and inside the rate line.  A report holds no speed-up, no baseline and no figure of another GPU class.
"""

from __future__ import annotations

import ctypes
import decimal
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import textwrap
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import census, observe, tie_out
from .timing import now, step_facts

SCHEMA = "model-sample-performance/6"
REPLAYED_TRACED = 16  # the steps of the second traced request of a run whose steps are replays
WIDTH = 110  # of the screen: no line is wider
ORIENTATION, CONTROLLED = "orientation", "controlled"
PASS, FAIL, NOT_OBSERVED = census.PASS, census.FAIL, census.NOT_OBSERVED
GIB = 2.0**30
NOT_COMPUTED = "the timed requests did not compute the checked request's tokens"
CLOCK = "time.perf_counter, one process"  # of every statistic that is taken from stamps
NOT_THE_PLAN = "the kernels of the traced steps are not the plan's"
#: what launch overhead is wherever it is printed or stored, and why: the file's ``label`` and the line under the equation
ESTIMATE = "an estimate (GPU busy is the traced request's and an upper bound)"
LABEL = 58  # of the per-kernel table: the widest name of a row
#: what the per-kernel table of a step launched from Python says of itself
FROM_PYTHON = "launched from Python: not the time of a kernel inside a captured sequence"
#: who compiled an exact kernel that is no library's, in the words of a census entry (``census.Kernel.origin``): an entry
#: of cuDNN Frontend, an operator of PyTorch.  An exact kernel of any other origin is a library's, which the plan calls
#: through an entry of that library
OWN_ORIGINS = ("cuDNN FE", "PyTorch")
#: the kinds of a row of the per-kernel table whose launches are one record of one kind each: the kernel of an entry
#: of cuDNN Frontend or of an operator of PyTorch (``exact``), a library's kernel (``library``), the host read
#: (``read``: a copy to the host, not a kernel).  A time per launch is printed for them.  The edge of ``exact``
#: (``_row_kind``): a block that two library entries serve, and a block call that launches a library's kernel
#: together with one of cuDNN Frontend or with an operator of PyTorch, are ``exact`` too -- every launch of such a
#: row is still one kernel
ONE_KERNEL = ("exact", "library", "read")

#: statistic -> (unit, definition): the sentence the report file carries beside each number
DEFINITIONS: Dict[str, Tuple[str, str]] = {
    "time_per_output_token": (
        "ms",
        (
            "Per timed request: (time the last token's ids reach the host - time the first new token's ids reach the host) / (new tokens - 1). Of a run: the "
            "median over its timed requests."
        ),
    ),
    "decode_rate": (
        "tokens/s per sequence",
        "1000 / time per output token in ms. Never a median step turned into a rate. With several sequences the line adds the total.",
    ),
    "token_to_token_interval": (
        "ms",
        (
            "The time between the host reads of two consecutive tokens of a timed request, pooled over the run's timed requests; percentiles are nearest-rank. "
            "The intervals span a growing attention cache as well as jitter: the file gives the first and the last timed position."
        ),
    ),
    "time_to_first_token": (
        "ms",
        (
            "Per timed request: from the request being handed to the text path to the first new token's ids on the host -- render, tokenize, the copy of the "
            "ids, the prompt's decode steps (one per prompt token: this sample has no prefill kernel) and the host read. Of a run: the median over its timed "
            "requests. The load is outside it."
        ),
    ),
    "step": (
        "ms",
        (
            "The median token-to-token interval over the timed steps at the positions the traced request covers, all timed requests. Equal to the median of the"
            " token-to-token interval when the whole request is traced."
        ),
    ),
    "gpu_busy": (
        "ms",
        (
            "Per traced step: the length of the union of the intervals of every kernel and copy between two host reads, from the profiler's kernel activity "
            "records; the median over the traced steps. An upper bound: tracing lengthens kernels."
        ),
    ),
    "host": (
        "ms",
        (
            "From a token's ids on the host to the next step's first launch call: the loop's own work (keeping the ids); the median over the timed steps at the"
            " traced positions."
        ),
    ),
    "launch_overhead": (
        "ms",
        (
            "Per timed step at a traced position: step - GPU busy of the traced step at the same position - host, formed step by step before anything is "
            "aggregated; of a run the mean over those steps. With the means of the step, of GPU busy and of host over the same steps, which this file holds, "
            "it adds up: step = GPU busy + launch overhead + host. It stands for time inside a step in which the GPU waited for the next launch, or the host "
            "for the read, and it is an estimate, not a measurement: a timed step has no GPU-side record, and GPU busy is the traced request's, taken under "
            "the profiler, and is an upper bound. Printed as a term of the step, and called an estimate there, only where it is larger than the mean GPU busy; "
            "otherwise the line says that launch overhead and host are not resolved by this method. The file holds the remainder as computed, which can be "
            "negative, the count, the median, the extremes and the percentiles of the per-step remainders, and the launch overhead per block call."
        ),
    ),
    "gpu_busy_replayed": (
        "ms",
        (
            "Of a run whose steps are replays of a recorded step. Per step of a second traced request, whose steps are replays: the length of the union of "
            "the intervals of every kernel and copy between two host reads, from the profiler's kernel activity records; the median over those steps. An "
            "upper bound: tracing lengthens kernels."
        ),
    ),
    "tracing_stretch": (
        "%",
        (
            "The median time from a step's first launch to its ids on the host in the traced request, divided by the same in the timed requests at the same "
            "positions, minus 1."
        ),
    ),
    "tokens_per_second_wall": (
        "tokens/s per sequence",
        (
            "Per timed request: new tokens / (time the last token's ids reach the host - time the request is handed to the text path). Of a run: the median "
            "over its timed requests. It holds the time to first token and is not the decode rate; where no decode rate is printed it is null too."
        ),
    ),
    "launches_per_step": (
        "launches",
        (
            "From the plan: the launches of a step on exact rows -- the rows one named kernel serves, each launch of which the kernel check holds one by one "
            "-- by provider, and the block calls of a step on fallback rows. Observed by the kernel check on the traced steps, which also counts the kernels "
            "and copies a step launches beside them."
        ),
    ),
    "kernel_launches": (
        "launches",
        (
            "Of a row of the per-kernel table: the kernel and copy records of a traced step that the kernel check attributes to the row's blocks; the median "
            "per step."
        ),
    ),
    "kernel_time_per_launch": (
        "us",
        (
            "Of a row: the time a launch adds to the GPU's timeline -- its end minus the later of its start and the latest end of any earlier record of the "
            "step, not below zero; the median over the row's launches in all traced steps. It needs time stamps finer than a microsecond: the file records "
            "the resolution found, and where the profiler hands over whole microseconds the column is not printed. The file also holds the median recorded "
            "duration."
        ),
    ),
    "kernel_time_per_step": (
        "ms",
        (
            "Of a row: per traced step the sum of the row's added times, the median over the steps; its share is that over the median GPU busy. Within one "
            "step the added times of all records sum exactly to that step's GPU busy. The medians of the rows need not sum to the median GPU busy, nor their "
            "shares to the whole of it: the file also holds each row's mean, and the means sum to the mean GPU busy."
        ),
    ),
    "memory": (
        "GiB",
        (
            "Weights, state and cache: the family's arithmetic over the tensors held on the device, in GiB (2^30 bytes). Peak reserved: PyTorch's peak between "
            "the emptied pool behind the checks and the end of the timed requests. The file also gives the weights in GB (10^9 bytes), the unit of an ordinary "
            "run's bytes per step line, and the whole process's peak."
        ),
    ),
    "startup": (
        "s",
        (
            "The seconds of every stage of a run. Import, describe and plan are host times; every later stage -- of the load, the checks, the zeroing of the "
            "state, the requests -- has the device synchronised at both ends. The stage of the timed requests spans the requests the statistics are taken "
            "from; no other stage enters a rate or the time to first token."
        ),
    ),
}

DEFINITIONS.update(observe.DEFINITIONS)  # the GPU and the host during the timed requests: defined beside what observes them


@dataclass(frozen=True)
class Protocol:
    """The numbers of a run of the mode: forced decode steps of the warm-up request, timed decode steps of a request, timed
    requests, and the most launch records the traced request may hold."""

    warmup: int = 32
    steps: int = 256
    requests: int = 3
    trace_records: int = 160_000

    def decode_steps(self, prompt_tokens: int, replayed: bool = False) -> int:
        """The most decode steps of the stack a run makes: the checked request, its replay by the checks, the timed requests
        and a traced request of every timed step, and the warm-up request, each behind the prompt's own steps -- and,
        for a run whose steps are replays of a recorded step (``replayed``), the second traced request of
        ``REPLAYED_TRACED`` replayed steps behind its prompt.  The reference's own passes of the checks are not steps
        of the stack and are not counted."""
        again = prompt_tokens + min(REPLAYED_TRACED, self.steps) if replayed else 0
        return (3 + self.requests) * (prompt_tokens + self.steps) + prompt_tokens + self.warmup + again

    def line(self, prompt_tokens: int, replayed: bool = False) -> str:
        """What a run of the mode will do, said before the load: a rule and counts, never a time.  ``replayed``: the run
        records its decode step and replays it."""
        each = f"{self.requests} timed request{'s' if self.requests != 1 else ''} of {self.steps} decode steps each"
        again = f" and, replayed under the profiler, one of {min(REPLAYED_TRACED, self.steps)}" if replayed else ""
        return (
            f"performance mode: at most {self.decode_steps(prompt_tokens, replayed):,} decode steps of this stack in all: one checked request and its "
            f"replay by the checks, one warm-up request of {self.warmup} decode steps behind its first new token, {each}, one traced request of up to "
            f"{self.steps}{again}; a prompt of {prompt_tokens} tokens is {prompt_tokens} decode steps in front of each; the reference's own passes are "
            "not counted"
        )


def judged_positions(prompt_tokens: int, steps: int, page: Optional[int] = None, first: int = 16) -> List[Tuple[int, str]]:
    """``(position, why)`` of every position at which the checked request's blocks are held against the reference: the
    first ``first`` positions of the request; the two positions on either side of the first page boundary of the attention
    cache that has both inside the timed steps (``page`` positions a page; ``None``: the family names no page); and the
    last two timed positions.  The timed steps are the positions ``prompt_tokens`` to ``prompt_tokens + steps - 1``."""
    last = prompt_tokens + steps - 1
    why = {position: f"one of the first {first} positions of the request" for position in range(min(first, last + 1))}
    boundary = None if not page else next((b for b in range(page, last + 1, page) if b > prompt_tokens), None)
    if boundary is not None:
        for position in (boundary - 1, boundary):
            why.setdefault(position, f"on either side of the first page boundary of the attention cache inside the timed steps ({page} positions a page)")
    for position in (last - 1, last):
        if position >= prompt_tokens:
            why.setdefault(position, "one of the last two timed positions")
    return sorted(why.items())


@dataclass
class Request:
    """The stamps and the produced ids of one request.  ``ids[0]`` is the read of the first new token, ``ids[i + 1]`` the
    read of forced step ``i``: each the ids of every sequence."""

    kind: str  # "checked" | "warm-up" | "timed" | "traced"
    t_request: float = 0.0
    t_first: float = 0.0
    t_launch: List[float] = field(default_factory=list)
    t_token: List[float] = field(default_factory=list)
    ids: List[List[int]] = field(default_factory=list)
    state: str = ""  # the SHA-256 of the state the request left; "" where it was not taken

    def intervals_ms(self) -> List[float]:
        """The time between the host reads of two consecutive tokens, one per forced step."""
        reads = [self.t_first, *self.t_token]
        return [(after - before) * 1e3 for before, after in zip(reads, reads[1:])]

    def host_ms(self) -> List[float]:
        """Per forced step: from the ids of the token before on the host to the step's first launch call."""
        return [(launch - read) * 1e3 for read, launch in zip([self.t_first, *self.t_token], self.t_launch)]

    def in_step_ms(self) -> List[float]:
        """Per forced step: from its first launch call to its ids on the host."""
        return [(token - launch) * 1e3 for launch, token in zip(self.t_launch, self.t_token)]

    def time_per_output_token_ms(self) -> float:
        """(last token on the host - first new token on the host) / (new tokens - 1): the request has one new token more
        than forced steps."""
        return (self.t_token[-1] - self.t_first) * 1e3 / len(self.t_token)

    def time_to_first_token_ms(self) -> float:
        return (self.t_first - self.t_request) * 1e3

    def ids_digest(self) -> str:
        return hashlib.sha256(b"".join(int(token).to_bytes(4, "little") for read in self.ids for token in read)).hexdigest()


def nearest_rank(samples: Sequence[float], percent: float) -> float:
    """The nearest-rank percentile: the sample of rank ceil(percent x n / 100) of the n sorted samples."""
    ordered = sorted(samples)
    return ordered[max(1, math.ceil(percent * len(ordered) / 100)) - 1]


def summary(samples: Sequence[float]) -> dict:
    """What the file holds of a list of samples; ``{"n": 0}`` for none."""
    if not samples:
        return {"n": 0}
    return {
        "n": len(samples),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "minimum": min(samples),
        "maximum": max(samples),
        "p90": nearest_rank(samples, 90),
        "p99": nearest_rank(samples, 99),
    }


def step_split(
    steps: Sequence[Sequence[float]], hosts: Sequence[Sequence[float]], busy: Optional[Sequence[float]], untraced_ms: float = 0.0, traced_ms: float = 0.0
) -> dict:
    """What the ``step`` line takes a step apart into.  ``steps`` and ``hosts``: per timed request its token-to-token
    intervals and the loop's host times at the traced positions, in ms; ``busy``: GPU busy of the traced request's step
    at each of those positions (``None``: not measured).  ``remainders``: per timed step what it leaves beside GPU busy
    of the traced step at the same position and beside its own host time -- formed step by step, before anything is
    aggregated: the median step, the median GPU busy and the median host time need not be the same step's, so their
    difference can be a remainder that no step has.  ``remainder_ms``: the mean of them, as computed; with the means of
    the same steps it adds up, step = GPU busy + remainder + host, to the rounding of a double.  It is an estimate: a
    timed step has no GPU-side record, and GPU busy is taken under the profiler and is an upper bound, so the remainder
    can be negative.  ``term``: whether launch overhead is printed as a term of the step -- only where the remainder is
    larger than the mean GPU busy; that decides what is printed, never that the remainder was measured.  ``stretch``:
    the time from a step's first launch to its ids on the host in the traced request (``traced_ms``) over the same in
    the timed requests (``untraced_ms``), minus 1; ``None`` where it cannot be formed.  ``ValueError`` where a timed
    request does not hold exactly the traced positions: nothing is paired by cutting a list short."""
    if busy is None:
        return {"remainder_ms": None, "term": False, "stretch": None, "remainders": []}
    lengths = sorted({len(of) for of in (*steps, *hosts)})
    if lengths != [len(busy)] or len(steps) != len(hosts):
        raise ValueError(f"GPU busy of {len(busy)} traced steps cannot be paired with the steps of the timed requests at the traced positions ({lengths})")
    remainders = [step - at - host for of, own in zip(steps, hosts) for step, at, host in zip(of, busy, own)]
    remainder = statistics.fmean(remainders)
    stretch = traced_ms / untraced_ms - 1 if untraced_ms > 0 else None
    return {"remainder_ms": remainder, "term": remainder > statistics.fmean(busy), "stretch": stretch, "remainders": remainders}


def added_ms(step: Sequence[census.Launch]) -> List[float]:
    """Per launch record of a traced step, in start order: the time the record adds to the GPU's timeline, in ms -- its end
    minus the later of its start and the latest end of any earlier record of the step, not below zero.  The added times
    of a step partition the union of its records' intervals: a stretch in which two records run is counted once, for
    the one that started first, and the sum is the step's GPU busy."""
    added, reach = [], None
    for record in step:
        begin = record.start if reach is None or record.start > reach else reach
        added.append(max(record.end - begin, 0.0))
        reach = record.end if reach is None or record.end > reach else reach
    return added


def _recorded_us(record: census.Launch) -> float:
    """The duration of a launch record in microseconds as the trace holds it; of a record that carries none (it was not
    read from a trace) its end less its start, to the nanosecond."""
    return record.duration if record.duration is not None else round((record.end - record.start) * 1e3, 3)


def time_resolution_us(steps: Sequence[Sequence[census.Launch]]) -> Optional[float]:
    """The resolution of the profiler's time stamps as the launch records of the traced steps show it, in microseconds:
    one over ten to the most decimals of a microsecond that a recorded duration has -- 1.0 where every duration is a
    whole number of microseconds.  ``None`` where there is no record."""
    durations = {_recorded_us(record) for step in steps for record in step}
    places = [-decimal.Decimal(repr(duration)).normalize().as_tuple().exponent for duration in durations]
    return 10.0 ** -max(0, *places) if places else None


def _row_kind(row: census.Row, sites: Sequence[census.Site]) -> Tuple[str, str]:
    """``(kind, entry)`` of a row of the walk in the per-kernel table, decided from the census entries the plan's rows
    declare for the row's block calls and never from a kernel's name.  The kind is the row's relation, with two
    exceptions.  ``library``: an exact row whose block calls one entry serves and whose exact kernels are a library's
    (``OWN_ORIGINS``); ``entry`` is then that entry as the plan names it, and the row holds the launches of the
    library's kernel, which the walk finds like every exact kernel.  ``beside``: a row of a wiring gap in which every
    block call that declares a fallback family also has an exact kernel of its own -- its records are what those
    calls launch beside their kernels, and no block of the row is on a fallback entry."""
    calls = [sites[index] for index in row.sites]
    if row.relation == "exact":
        origins = {kernel.origin for site in calls for kernel in site.kernels if kernel.relation == "exact"}
        entries = {site.entry for site in calls}
        return ("library", next(iter(entries))) if len(entries) == 1 and not origins & set(OWN_ORIGINS) else ("exact", "")
    chains = [site for site in calls if any(kernel.relation == "fallback" for kernel in site.kernels)]
    if chains and all(any(kernel.relation == "exact" for kernel in site.kernels) for site in chains):
        return "beside", ""
    return row.relation, ""


def _row_label(row: census.Row, kind: str = "", entry: str = "") -> str:
    """The name of a row of the per-kernel table: its block -- behind the entry that serves it, as the plan names it,
    where that is a library's (``kind`` and ``entry`` of ``_row_kind``); for the blocks of a wiring gap their names,
    with ``(one row)`` where they are several, behind what the row is -- ``beside its kernel:`` for what block calls
    launch beside kernels of their own, ``fallback:`` for blocks on fallback rows, ``host path:`` for host blocks
    alone."""
    if row.relation == "read":
        return "host read"
    if row.relation == "exact":
        return ((f"{entry}: " if kind == "library" else "") + row.blocks[0])[:LABEL]
    prefix = "host path: " if row.relation == "host" else ("beside its kernel: " if kind == "beside" else "fallback: ")
    names = " + ".join(row.blocks) + (" (one row)" if len(row.blocks) > 1 else "")
    if len(prefix + names) > LABEL:
        names = f"{row.blocks[0]} + {len(row.blocks) - 1} more (one row)"
    return (prefix + names)[:LABEL]


def kernel_table(traced: Optional[census.Traced], busy: Sequence[float], unobserved: str = "") -> dict:
    """The file's ``per_kernel`` section: per row of the walk's table (``census.Row``) what its launch records are and what
    they add to the GPU's timeline, over the traced steps.  ``busy``: GPU busy of every traced step, in ms.  A table is
    made only of a pass whose kernel check holds: where a record has no row (the check failed), or nothing was observed
    (``unobserved``: why), ``rows`` is ``None`` and ``reason`` says why no table is printed."""
    table = {"launch_path": "python", "launch_path_note": FROM_PYTHON, "us_per_launch_printed": False, "reason": None, "rows": None}
    if traced is None or traced.status != PASS or traced.result is None or not busy:
        failed = traced is not None and traced.status == FAIL
        return {**table, "reason": NOT_THE_PLAN if failed else f"the kernels of the traced steps were not observed ({unobserved or 'no launch record'})"}
    rows, steps, sites = traced.result.table, traced.steps, traced.sites
    kernels, copied, ms = ([[] for _ in rows] for _ in range(3))  # per row: its kernels, its copies and its added ms in every traced step
    added, recorded = ([[] for _ in rows] for _ in range(2))  # per row: the added and the recorded time of every launch, in us
    copies: Dict[Tuple[str, str], bool] = {}  # (kind, name) of a record -> whether the kernel check counts it as a copy
    for step, held in zip(steps, traced.result.attributed):
        totals = [[0, 0, 0.0] for _ in rows]  # kernels, copies, added ms of this step
        for record, row, adds in zip(step, held, added_ms(step)):
            copy = copies.setdefault((record.kind, record.name), census.canonical(record) == census.COPY)
            totals[row][1 if copy else 0] += 1
            totals[row][2] += adds
            added[row].append(adds * 1e3)
            recorded[row].append(_recorded_us(record))
        for row, (launched, copy, adds) in enumerate(totals):
            kernels[row].append(launched)
            copied[row].append(copy)
            ms[row].append(adds)
    gpu_busy, median, resolution = statistics.median(busy), statistics.median, time_resolution_us(steps)
    made = []
    for number, row in enumerate(rows):
        blocks = []
        for name in row.blocks:
            calls = [index for index in row.sites if sites[index].block == name]
            blocks.append({"block": name, "calls_per_step": len(calls), "entry": sites[calls[0]].entry})
        exact, (kind, entry) = row.relation == "exact", _row_kind(row, sites)
        shapes = Counter((r.name, r.grid, r.block, r.shared_memory) for r, at in zip(steps[0], traced.result.attributed[0]) if exact and at == number)
        made.append(
            {
                "label": _row_label(row, kind, entry),
                "relation": kind,
                "blocks": blocks,
                "launches_per_step": median([k + c for k, c in zip(kernels[number], copied[number])]),
                "planned_launches_per_step": sum(k.launches for index in row.sites for k in sites[index].kernels if k.relation == "exact") if exact else None,
                "kernels_per_step": median(kernels[number]),
                "copies_per_step": median(copied[number]),
                "us_per_launch_added": median(added[number]) if added[number] else None,
                "us_per_launch_recorded": median(recorded[number]) if recorded[number] else None,
                "ms_per_step": median(ms[number]),
                "ms_per_step_mean": statistics.fmean(ms[number]),
                "share_percent": median(ms[number]) / gpu_busy * 100 if gpu_busy > 0 else None,
                "launch_shapes": [dict(zip(("symbol", "grid", "block", "shared_memory", "launches"), (*shape, n))) for shape, n in shapes.items()] or None,
            }
        )
    return {**table, "us_per_launch_printed": resolution is not None and resolution < 1.0, "rows": made}


def record_class(
    reduced: bool,
    fallback_rows: bool,
    captured: bool,
    checks: str,
    others: str = "other processes not observed",
    clocks: Sequence[str] = ("clocks not observed",),
) -> Tuple[str, List[str]]:
    """``(class, reasons)`` of a run, the reasons in the order of the conditions.  A run is ``controlled`` only when every
    condition was observed by the run itself; each argument is a condition, or the reason it is not met (``""``: met):
    ``reduced`` -- no checkpoint; ``fallback_rows`` -- a block of the plan is on a fallback row; ``captured`` -- the launch
    sequence is captured, not launched from Python; ``checks`` -- why the three statements of the ``checks`` lines do not
    all hold; ``others`` -- why the run cannot say it had the GPU alone; ``clocks`` -- every reason for which its clocks
    are not a controlled run's (one reason, or several)."""
    reasons = [
        "reduced model" if reduced else "",
        "fallback rows present" if fallback_rows else "",
        "" if captured else "launched from Python",
        checks,
        others,
        *([clocks] if isinstance(clocks, str) else clocks),
    ]
    reasons = [reason for reason in reasons if reason]
    return (ORIENTATION if reasons else CONTROLLED), reasons


def state_digest(state) -> str:
    """The SHA-256 over the position and the bytes of every tensor of a decode state.  Reads the device: behind a request,
    never inside one."""
    digest = hashlib.sha256(str(state.position).encode())
    for layer in state.layers:
        for tensor in layer.tensors():
            host = tensor.detach().cpu().contiguous()
            size = host.numel() * host.element_size()
            digest.update(ctypes.string_at(host.data_ptr(), size) if size else b"")
    return digest.hexdigest()


def planned_launches(plan, model) -> dict:
    """What a plan states of the launches of one step, call by call of the model's wiring: the launches on exact rows -- the
    rows one named kernel serves -- in all and by provider, the block calls on fallback rows, and whether a part of the
    plan is on a fallback row."""
    served = [plan.at(name, site) for (name, _), site in zip(model.wiring(), census.call_sites(model))]
    by_provider: Counter = Counter()
    for _, block in served:
        launches = sum(kernel.launches for kernel in block.kernels if kernel.relation == "exact")
        if launches:
            by_provider[block.provider.replace("cuDNN FE", "cuDNN Frontend")] += launches
    return {
        "exact_launches_per_step": sum(by_provider.values()),
        "by_provider": dict(by_provider.most_common()),
        "fallback_calls": sum(bool(row.fallback) for row, _ in served),
        "fallback_rows": any(row.fallback for _, _, _, row, _ in plan.parts()),
    }


def kernel_facts(session, status: str) -> dict:
    """What the ``kernels`` lines say: from the plan, the launches of a step on exact rows, by provider, and the block calls
    on fallback rows (``planned_launches``); and, from the traced steps, the kernels and copies a step launches in all
    (``None`` where they were not observed)."""
    facts = planned_launches(session.plan, session.model)
    result = None if session.traced is None else session.traced.result
    facts["kernels_per_step"] = facts["copies_per_step"] = None
    if result is not None and result.launches and status != NOT_OBSERVED:
        facts["kernels_per_step"], facts["copies_per_step"] = int(statistics.median(result.launches)), int(statistics.median(result.copies))
    return facts


def _wrapped(label: str, text: str, column: int = 22) -> List[str]:
    """A line of the screen: the label, padded to ``column``, and the text, wrapped under itself at ``WIDTH``."""
    parts = textwrap.wrap(text, WIDTH - column, break_on_hyphens=False) or [""]  # a word wider than a line (a file's name) is cut
    return [f"{label:<{column}}{parts[0]}", *(" " * column + part for part in parts[1:])]


def _packed(label: str, parts: Sequence[str], column: int = 22, joint: str = " | ", hang: str = "") -> List[str]:
    """A line of the screen made of parts: as many of them on a line as fit ``WIDTH``, a line break in the place of the joint
    where the next one does not (``hang`` then ends the line); a part that is wider than a line by itself is wrapped by
    its words."""
    rows = [""]
    for part in parts:
        if rows[-1] and column + len(rows[-1]) + len(joint) + len(part) > WIDTH:
            rows[-1] += hang
            rows.append(part)
        else:
            rows[-1] += (joint if rows[-1] else "") + part
    return [line for index, row in enumerate(rows) for line in _wrapped("" if index else label, row, column)]


def _checks_lines(checks: dict) -> List[str]:
    """The two ``checks`` lines: three statements, each with its verdict, none folded into another."""
    lines = _wrapped("checks", f"blocks against the reference at positions {checks['positions_text']}: {checks['blocks']}", 12)
    if checks.get("recorded_step"):  # of a run whose steps are replays: the checked request, launched again from Python
        held = f"the replayed steps are the same steps launched from Python, ids and logits, all {checks['recomputed_steps']}"
        lines += _wrapped("", f"{held}, and leave the same state: {checks['recorded_step']}", 12)
    second = f"timed requests give the checked request's tokens and state: {checks['timed_requests']}"
    third = f"kernels, traced steps: {checks['kernels']}"
    if 12 + len(second) + 3 + len(third) <= WIDTH:
        return lines + [" " * 12 + f"{second} | {third}"]
    return lines + _wrapped("", second, 12) + _wrapped("", third, 12)


def screen(data: dict, unwritten: str = "") -> List[str]:
    """The screen of a run, rendered from the data the report file holds and from nothing else: at most ``WIDTH`` characters
    a line.  Every time of a reduced model carries ``reduced model:``, and such a run prints no rate.  ``unwritten``: why
    the run could not write that file (``write_report``); the last line then says so in the place of the file's name.
    ``ValueError`` for the data of a file of another schema than ``SCHEMA``, in one sentence that names both: such a
    file has other sections and statistics, and nothing of it is rendered."""
    found = data.get("schema") if isinstance(data, dict) else None
    if found != SCHEMA:
        raise ValueError(f"a report file of schema {found!r} is not read by this version of the performance mode, which reads and writes {SCHEMA!r}")
    model, gpu, kernels, protocol, traced = data["model"], data["environment"]["gpu"], data["kernels"], data["protocol"], data["traced"]
    stats, memory, record = data["statistics"], data["memory"], data["record"]
    prefix = "reduced model: " if model["reduced"] else ""
    lines = ["cuDNN Frontend model sample: decode performance report"]
    if record["class"] == CONTROLLED:
        lines += ["record class          controlled run in one process; a number of record is the median of 3 such runs"]
    else:
        first, *more = record["reasons"]
        lines += _packed("record class", [f"{record['class']}, not a number of record: {first}", *more], joint="; ", hang=";")
    lines += ["scope       the decode loop of one process: no request queue, no scheduler, no network"]
    sequences = f"{model['sequences']} sequence{'s' if model['sequences'] != 1 else ''}"
    lines += _packed("model", [f"{model['name']}, {model['layers']} layers", sequences, model["decoding"]], 12)
    seen = f"other compute processes: {observe.others_text(data['observation'])}"
    lines += _packed("GPU", [f"{gpu['name']}, compute capability {gpu['compute_capability']}, {gpu['sm_count']} SMs", seen], 12)
    lines += _packed("", [f"clocks observed: {observe.clocks_text(data['observation'])}", observe.host_text(data["observation"])], 12)
    calls = f"{kernels['fallback_calls'] or 'no'} block call{'s' if kernels['fallback_calls'] != 1 else ''} on fallback rows"
    exact = kernels["exact_launches_per_step"]
    beside = "what a step also launches was not observed"
    if kernels["kernels_per_step"] is not None:
        others, copies = kernels["kernels_per_step"] - exact, kernels["copies_per_step"]
        beside = f"a step also launches {others:,} kernel{'s' if others != 1 else ''} and {copies:,} cop{'ies' if copies != 1 else 'y'}"
    providers = ", ".join(f"{provider} {count}" for provider, count in kernels["by_provider"].items()) or "none"
    lines += _wrapped("kernels", f"{exact:,} launch{'es' if exact != 1 else ''} per step on exact rows: {providers}", 12)
    lines += _packed("", [calls, beside], 12, "; ")
    path = "launch sequence captured" if protocol["launch_path"] == "captured" else "launched from Python, not captured"
    timed = f"timed {protocol['timed_steps']} steps x {protocol['timed_requests']}, end token ignored"
    lines += _packed("protocol", [f"warm-up {protocol['warmup_steps']} steps", timed, path], 12)
    lines += _checks_lines(data["checks"]) + [""]
    rate, tpot = stats["decode_rate"], stats["time_per_output_token"]
    if rate["value"] is None:
        lines += _wrapped("decode rate", rate["reason"])
    else:
        label = "decode rate (orientation)" if record["class"] == ORIENTATION else "decode rate"
        total = f" ({rate['total']:.1f} in all)" if model["sequences"] > 1 else ""
        requests = f"{tpot['n']} request{'s' if tpot['n'] != 1 else ''} within {tpot['spread_percent']:.1f} %"
        value = [f"{rate['value']:.1f}  tokens/s per sequence{total}", f"({tpot['value']:.3f} ms per token; {requests})"]
        lines += _packed(label, value, 28 if record["class"] == ORIENTATION else 22, "   ")
    interval = stats["token_to_token_interval"]
    lines += [" " * 22 + f"token-to-token interval: {prefix}median {interval['median']:.3f}, p90 {interval['p90']:.3f}, p99 {interval['p99']:.3f} ms"]
    prompt = protocol["prompt"]["tokens"]
    route = f"({prompt} decode steps: this sample has no prefill kernel)"
    lines += _wrapped("time to first token", f"{prefix}{stats['time_to_first_token']['value']:.1f}  ms for a {prompt}-token prompt {route}")
    lines += _step_lines(stats, traced, prefix)
    lines += ["with draft tokens     not run by this script (plain decode)"]
    peak = "peak reserved not observed" if memory["peak_reserved_gib"] is None else f"peak reserved {memory['peak_reserved_gib']:.1f} GiB"
    held = [f"weights {memory['weights_gib']:.1f} GiB", f"state and cache {memory['state_gib']:.2f} GiB", f"{peak}  (GiB = 2^30 bytes)"]
    lines += _packed("memory", held)
    overhead = stats["launch_overhead"]
    if overhead["value"] is not None:  # a term of the step: what it is, from the run's own figures
        per_call = overhead["host_time_per_block_call_us"]
        each = "" if per_call is None else f" = {per_call:.0f} us per block call"
        share = f"{prefix}launch overhead (an estimate) {overhead['value'] / stats['step']['mean'] * 100:.0f} % of the mean step{each}"
        lines += _wrapped("note", f"{share}: time in this sample's call path from Python, not kernel time")
    lines += [""] + _kernel_lines(data["per_kernel"], prefix)
    if unwritten:
        return lines + [""] + _wrapped("report file", f"not written: {unwritten}")
    lines += [""] + _packed("report file", [data["report_file"], "(every number, its definition, the protocol, the environment)"], joint=" ")
    return lines


def _step_lines(stats: dict, traced: dict, prefix: str) -> List[str]:
    """The ``step`` line in its three forms.  GPU busy not measured: the step, the positions it was taken at and why.
    Measured: the step -- the median -- and its mean taken apart, ``mean = GPU busy + launch overhead + host``, every
    term the mean over the same steps, only where launch overhead is larger than the mean GPU busy, and under it the
    line that calls launch overhead an estimate; otherwise the step, its mean beside the mean GPU busy as the upper
    bound it is, and that the rest is not resolved.  Under the line that calls launch overhead an estimate, and only
    there, a line with the median of the per-step remainders, which the file holds beside their mean: one stalled step
    moves the mean -- the launch overhead of the equation -- and not the median.  Where the rest is not resolved no
    figure of the remainder is printed, its median neither: under that sentence it would read as the resolved figure
    the sentence denies (the file holds it in either form).  Under both forms a line with the traced steps, their
    positions, their median GPU busy -- what the shares of the per-kernel table are taken of -- and by how much tracing
    lengthened a step."""
    value, busy, overhead = f"{prefix}{stats['step']['value']:.3f}  ms", stats["gpu_busy"]["value"], stats["launch_overhead"]
    again = traced.get("replayed")
    if again:  # a run whose steps are replays: the step is one launch, and GPU busy is the replayed traced steps'
        launches = "the recorded step" if again["launches"] is None else f"{again['launches']:,} launches"
        lines = _wrapped("step", f"{value}: one replay of {launches} and one host read")
        if again["gpu_busy_ms"] is None:
            return lines + _wrapped("", f"GPU busy of a replayed step not measured ({again['reason']})")
        several = f"{again['steps']} traced replayed steps, positions {again['first_position']} to {again['last_position']}"
        return lines + _wrapped("", f"{prefix}GPU busy {again['gpu_busy_ms']:.3f} ms in {several} (an upper bound: tracing lengthens kernels)")
    one, first, last = traced["steps"] == 1, traced["first_position"], traced["last_position"]
    if busy is None:
        at = f"at the traced position, {first}" if one else f"at the {traced['steps']} traced positions, {first} to {last}"
        return _wrapped("step", f"{value} {at if traced['steps'] else 'over every timed step'}; GPU busy not measured ({traced['gpu_busy_reason']})")
    mean, of_gpu = stats["step"]["mean"], stats["gpu_busy"]["mean"]  # with the remainder and the mean host time: the means of the same steps
    if overhead["value"] is not None:
        equation = f"mean {mean:.3f} = GPU busy {of_gpu:.3f} + launch overhead {overhead['value']:.3f} + host {stats['host']['mean']:.3f}"
        lines = _packed("step", [value, equation], joint="; ", hang=";") + _wrapped("", f"launch overhead: {overhead['label']}")
        middle = round(overhead["median"], 3) + 0.0  # + 0.0: a median that rounds to zero carries no sign
        lines += _wrapped("", f"median of the per-step remainders: {middle:.3f} ms (launch overhead is their mean)")
    else:  # not resolved: no figure of the remainder under that sentence, its median neither (the file holds both)
        bound = f"mean GPU busy {of_gpu:.3f} ms in a traced request (an upper bound: tracing lengthens kernels)"
        lines = _packed("step", [f"{value} (not traced), mean {mean:.3f}", bound, "launch overhead and host together: not resolved by this method"])
    stretch = stats["tracing_stretch"]["value"]
    lengthens = "the stretch of tracing was not formed" if stretch is None else f"tracing lengthens a step by {stretch:.1f} %"
    steps = f"the traced step, position {first}" if one else f"{traced['steps']} traced steps, positions {first} to {last}"
    return lines + _packed("", [f"(GPU busy from {steps}", f"median {busy:.3f} ms", f"{lengthens})"], joint="; ", hang=";")


def _kernel_lines(table: dict, prefix: str) -> List[str]:
    """The per-kernel table of the screen, from the file's ``per_kernel`` section: per row its launches of a step, the time
    a launch adds to the GPU's timeline, the row's time per step and its share of GPU busy; or the one line that says
    why there is no table.  ``us/launch`` is printed for a row whose launches are one record of one kind each
    (``ONE_KERNEL``: the kernel of an exact row, a library's kernel, the host read -- a copy to the host, not a
    kernel), not for a row of fallback or host blocks or of launches beside a kernel (chains of many kernels), nor
    at all where the profiler's stamps are whole microseconds."""
    if table["rows"] is None:
        return _wrapped("per kernel, per step", f"not printed: {table['reason']}")
    fine = table["us_per_launch_printed"]
    columns = f"{'launches':>9}" + (f"{'us/launch':>11}" if fine else "") + f"{'ms/step':>9}  share of GPU busy"
    each = "= the time a launch adds to the GPU's timeline" if fine else "not printed: the profiler's time stamps are whole microseconds here"
    lines = [f"{f'per kernel, per step ({prefix}traced request)':<{LABEL + 2}}{columns}", f"  {table['launch_path_note']}", f"  us/launch {each}"]
    for row in table["rows"]:
        count, each = row["launches_per_step"], "--"
        if row["relation"] in ONE_KERNEL and row["us_per_launch_added"] is not None:
            each = f"{row['us_per_launch_added']:.2f}"
        launches = f"{count:,.0f}" if float(count).is_integer() else f"{count:,.1f}"
        share = "--" if row["share_percent"] is None else f"{row['share_percent']:.1f} %"
        lines.append(f"  {row['label']:<{LABEL}}{launches:>9}" + (f"{each:>11}" if fine else "") + f"{row['ms_per_step']:>9.3f}{share:>9}")
    return lines


def default_report_name() -> str:
    """``report-<UTC time>.json``: the name of a report file nobody named, so that one command can be run twice."""
    return f"report-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"


def write_report(path: str, data: dict) -> str:
    """Write the report file: strict JSON (no NaN, no infinity), never over a file that exists.  Returns ``""`` or, where
    the file cannot be written after all, the reason with the file's own name."""
    name = os.path.basename(os.path.normpath(path))
    try:
        with open(path, "x", encoding="utf-8") as file:
            json.dump(data, file, indent=1, allow_nan=False)
            file.write("\n")
    except (OSError, ValueError) as error:  # a file that appeared meanwhile, a folder that went away, a full disk; a value that is not a number
        return f"--perf-report {name}: {getattr(error, 'strerror', None) or str(error) or type(error).__name__}"
    return ""


@dataclass
class Outcome:
    """What ``run`` hands back to the script: whether every check held, the provider blocks judged, why the kernels were not
    observed (``""``: they were), the ids every sequence produced in the checked request, what could not be written, and
    PyTorch's peaks of the whole process in bytes (allocated, reserved), which the mode's own reset would otherwise hide."""

    passed: bool
    judged: int = 0
    unobserved: str = ""
    produced: List[List[int]] = field(default_factory=list)
    unwritten: str = ""
    peaks: Tuple[int, int] = (0, 0)
    data: Optional[dict] = None


def _statistics(
    timed: Sequence[Request],
    covered: int,
    sequences: int,
    rate_missing: str,
    busy: Sequence[float] = (),
    traced: Optional[Request] = None,
    calls: int = 0,
    unmeasured: str = "no traced step was observed",
) -> dict:
    """The entries of the file's ``statistics`` section, each with its definition.  ``covered``:
    the traced positions, the first so many steps of every timed request (0: none was observed, and the step is taken
    over every timed step).  ``busy``: GPU busy of every traced step, in ms, one per traced position -- each is set
    against the step of every timed request at the same position (``step_split``) -- with the traced request
    (``traced``) and the block calls of a step (``calls``); empty where it was not measured, and ``unmeasured`` says
    why."""

    def entry(key: str, **values) -> dict:
        unit, definition = DEFINITIONS[key]
        return {"definition": definition, "unit": unit, **({"clock": CLOCK} if "events" in values else {}), **values}

    per_token = [request.time_per_output_token_ms() for request in timed]
    first = [request.time_to_first_token_ms() for request in timed]
    intervals = [interval for request in timed for interval in request.intervals_ms()]
    value, last = statistics.median(per_token), covered or None
    wall = [len(request.ids) / (request.t_token[-1] - request.t_request) for request in timed]
    spread = (max(per_token) - min(per_token)) / value * 100
    rate = (
        {"value": None, "reason": rate_missing, "total": None} if rate_missing else {"value": 1000 / value, "reason": None, "total": sequences * 1000 / value}
    )
    per_wall = {"value": None, "reason": rate_missing} if rate_missing else {"value": statistics.median(wall), "reason": None, "per_request": wall}
    stats = {
        "time_per_output_token": entry("time_per_output_token", events="t_token[last] - t_first", value=value, per_request=per_token, spread_percent=spread),
        "decode_rate": entry("decode_rate", **rate),
        "token_to_token_interval": entry("token_to_token_interval", events="t_token[i] - t_token[i - 1]", value=statistics.median(intervals)),
        "time_to_first_token": entry("time_to_first_token", events="t_first - t_request", value=statistics.median(first), per_request=first),
        "step": entry("step", events="t_token[i] - t_token[i - 1] at the traced positions"),
        "host": entry("host", events="t_launch[i] - t_token[i - 1] at the traced positions"),
        "tokens_per_second_wall": entry("tokens_per_second_wall", events="new tokens / (t_token[last] - t_request)", **per_wall),
    }
    samples = {"time_per_output_token": per_token, "token_to_token_interval": intervals, "time_to_first_token": first}
    samples["step"] = [interval for request in timed for interval in request.intervals_ms()[:last]]
    samples["host"] = [host for request in timed for host in request.host_ms()[:last]]
    for key, of in samples.items():
        stats[key].update(summary(of), value=stats[key].get("value", statistics.median(of)))
    # the GPU side of the step, from the traced request: GPU busy, what every timed step leaves beside it at its own position, tracing's stretch
    measured = bool(busy) and traced is not None
    untraced = statistics.median([inside for request in timed for inside in request.in_step_ms()[:last]])
    under = statistics.median(traced.in_step_ms()[:last]) if measured else 0.0
    at = [(request.intervals_ms()[:last], request.host_ms()[:last]) for request in timed]  # per timed request: its steps at the traced positions
    split = step_split([steps for steps, _ in at], [hosts for _, hosts in at], busy if measured else None, untraced, under)
    missing = f"GPU busy not measured ({unmeasured})"
    stats["gpu_busy"] = entry("gpu_busy", value=statistics.median(busy) if measured else None, reason=None if measured else missing, **summary(busy))
    stats["launch_overhead"] = entry(
        "launch_overhead",
        value=split["remainder_ms"] if split["term"] else None,
        reason=None if split["term"] else ("not resolved by this method: the remainder is not larger than the mean GPU busy" if measured else missing),
        label=ESTIMATE if measured else None,
        remainder_ms=split["remainder_ms"],
        host_time_per_block_call_us=split["remainder_ms"] * 1e3 / calls if measured and calls else None,
        **summary(split["remainders"]),
    )
    formed = split["stretch"] is not None
    reason = None if formed else ("not formed: the timed requests spend no time in a step" if measured else missing)
    events = "t_token[i] - t_launch[i] at the traced positions: the traced request over the timed requests"
    stats["tracing_stretch"] = entry("tracing_stretch", events=events, value=split["stretch"] * 100 if formed else None, reason=reason)
    sections = {"launches_per_step": "kernels", "memory": "memory", "startup": "startup"}
    sections.update(kernel_launches="per_kernel", kernel_time_per_launch="per_kernel", kernel_time_per_step="per_kernel")
    for key, section in sections.items():
        stats[key] = entry(key, value=None, reason=f"see the section {section} of this file")
    return stats


def run(
    session,
    text,
    tokenize: Callable[[], list],
    protocol: Protocol,
    say: Callable[..., None],
    *,
    name: str,
    facts,
    environment,
    sink,
    report: str,
    verdict: Callable[[bool, int, str], str],
    tag: str = "[00]",
    real: bool = True,
    source: str = "",
    text_path: Optional[dict] = None,
    observer: Optional[observe.Observer] = None,
) -> Outcome:
    """One run of the mode on a loaded session (the module docstring has the phases).  ``tokenize()`` is the text path of
    the request -- render and tokenize -- and returns one prompt per sequence; ``say`` prints a line of the run;
    ``name`` is the model as the screen names it; ``facts`` the device's; ``environment`` the versions the run stated;
    ``sink`` takes the text of sequence 0 once, behind the report; ``report`` is the report file; ``verdict(passed,
    judged, unobserved)`` is the script's last line without its tag; ``source`` says where the prompt came from;
    ``observer`` is the observation of the GPU the caller began before the load (``None``: it begins here)."""
    import torch

    model, device, stages = session.model, session.device, session.stages
    cuda = device.type == "cuda"
    watch = observe.Observer(device) if observer is None else observer

    def begin(kind: str) -> Tuple[Request, Callable[[int], None]]:
        """A request up to its first new token, and the loop body of its forced steps."""
        made = Request(kind)
        made.t_request = now()
        ids = [session.prefill(tokenize())]  # render, tokenize, the copy of the ids, the prompt's decode steps
        made.ids.append(session.read(ids[0]))  # the host read of the first new token
        made.t_first = now()

        def step(_index: int) -> None:
            made.t_launch.append(now())
            ids[0] = session.step(ids[0])
            tokens = session.read(ids[0])  # the one host read of a step
            made.t_token.append(now())
            made.ids.append(tokens)

        return made, step

    def request(kind: str, steps: int) -> Request:
        made, step = begin(kind)
        for index in range(steps):
            step(index)
        made.state = state_digest(session.state)  # behind the last stamp
        return made

    def zeroed() -> None:
        """The state zeroed in place for the next request, nothing recorded: a stage of its own, so the device is
        synchronised behind the zeroing and none of it runs inside the request's time to first token."""
        with stages.stage("zeroing the state"):
            session.reset(fresh=False, recorded=False)
            session.record_step()  # of a session that replays: where the checks left a fresh state, its step is recorded here, in front of the request

    def peaks_now() -> Tuple[int, int]:
        return (torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)) if cuda else (0, 0)

    # phase 3: the checked request, launched from Python with the recorder on, and the reference checks on it
    prompts = tokenize()  # for the counts, the prompt's hash and the text: every request renders and tokenizes for itself
    prompt_tokens = int(prompts[0].numel())
    positions = judged_positions(prompt_tokens, protocol.steps, getattr(model, "attention_page", None))
    judged = [position for position, _ in positions]
    listed = tie_out.positions_text(judged)
    if session.record.inputs.shape[0] < prompt_tokens + protocol.steps:  # the checks replay the request and judge its last positions
        raise ValueError(f"the checked request is {prompt_tokens + protocol.steps} steps and the session records {session.record.inputs.shape[0]}")
    first_check = "replay" if session.python_reason else "every replayed step launched again from Python"
    say(
        f"{tag} performance mode: the checked request ({prompt_tokens} prompt tokens through the decode step, the first new token, {protocol.steps} forced "
        f"steps), then the checks against the {session.reference_precision}-accumulated reference: {first_check}, the blocks at positions {listed}, the tokens",
        flush=True,
    )
    with stages.stage("the checked request"):
        checked = request("checked", protocol.steps)
        session.read_tail()  # of a replayed request: the step's own tail, read before anything replays again
    replayed = session.replays  # whether the steps of this run are replays of the recorded step
    produced = [[read[sequence] for read in checked.ids] for sequence in range(session.sequences)]
    with stages.stage("the checks before"):
        before = tie_out.run_checks(
            session,
            prompt_tokens=prompt_tokens,
            max_steps=16,
            second_reference=real,
            kernels=tie_out.DEFERRED,
            positions=judged,
            digest=state_digest if replayed else None,
        )
    peaks = peaks_now()
    same_state = not replayed or before.end_state == checked.state  # the recomputation from Python leaves the replayed request's state
    if not (before.ok and same_state):  # as an ordinary run: its lines, and nothing is timed; no file
        say(before.text())
        if not same_state:
            say("  recorded step     the state behind the steps launched again from Python is not the state the replayed request left    FAIL")
        return Outcome(False, before.judged, "", produced, peaks=peaks)
    listed = tie_out.positions_text(before.positions)  # from here on: the positions the checks judged, as the checks hand them back

    # phases 5 to 7: settle, the warm-up request, the timed requests -- nothing but the step inside, and no profiler pass so far
    say(
        f"{tag} performance mode: a warm-up request, {protocol.requests} timed request{'s' if protocol.requests != 1 else ''}, then a traced request and "
        "the checks behind it; the report follows",
        flush=True,
    )
    zeroed()  # the record of the checks is dropped before the pool is emptied
    gc.collect()  # ... and so are the reference sessions of the checks, with their states
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    watch.read("before the warm-up request")  # the GPU and the host at rest ...
    watch.start()  # ... and the sampler: a child process, which this process does not touch again before the traced request has ended
    requests = [checked]
    with stages.stage("the warm-up request"):
        requests.append(request("warm-up", protocol.warmup))
    for _ in range(protocol.requests):
        zeroed()
        with stages.stage("the timed requests"):
            requests.append(request("timed", protocol.steps))
    timed, timed_peaks = requests[2:], peaks_now()
    peaks = (max(peaks[0], timed_peaks[0]), max(peaks[1], timed_peaks[1]))
    watch.read("behind the last timed request")  # the other processes and the host's load again; the sampler goes on beside the traced request

    # phase 9: the traced request -- the first profiler pass of the process: one step, whose records size the rest
    zeroed()
    with stages.stage("the traced request"):
        traced, step = begin("traced")
        session.trace_request(step, 1)
        first = session.traced
        sizing = len(first.steps[0]) if first.steps else 0
        more = min(protocol.steps, protocol.trace_records // sizing) - 1 if sizing else 0
        if more > 0:
            session.trace_request(step, more)
            if session.traced.steps is not None:  # one traced request: the records and the call logs of both passes
                both = (first.call_log + session.traced.call_log, first.steps + session.traced.steps)
                session.traced = census.Traced(first.sites, *both, rows=session.sequences)
        requests.append(traced)
    again, replays_trace, replays_missing = None, None, ""
    if replayed:  # a second traced request: the first traced steps once more, as replays, under the profiler
        zeroed()
        with stages.stage("the replayed traced request"):
            again, step = begin("traced replay")
            count = min(REPLAYED_TRACED, len(session.traced.steps or []) or REPLAYED_TRACED, protocol.steps)
            replays_trace, replays_missing = session.trace_replays(step, count)
            requests.append(again)
    watch.stop()  # the sampler's loops end and their files are read: behind every request
    watched = watch.section([(made.kind, made.t_request, made.t_token[-1] if made.t_token else made.t_first) for made in requests[1:]])

    # phase 10: the checks after
    with stages.stage("the checks after"):
        held, missing = PASS, ""  # statement 2, and where it does not hold why no rate is printed: the first thing that differs
        for number, made in enumerate([*timed, traced, *([again] if again is not None else [])], 1):
            who = f"request {number}" if made.kind == "timed" else ("traced request" if made.kind == "traced" else "replayed traced request")
            differs = next((index for index, read in enumerate(made.ids) if read != checked.ids[index]), None)
            if differs is not None and held == PASS:
                held, missing = f"FAIL ({who}, step {differs})", NOT_COMPUTED if made.kind == "timed" else NOT_COMPUTED.replace("timed requests", who)
            if made.kind == "timed" and made.state != checked.state and held == PASS:
                held, missing = f"FAIL ({who}, end state)", "a timed request did not leave the checked request's state"
        changed = tie_out.changed_weights(session.weights)
        if changed and held == PASS:
            held, missing = f"FAIL (stored weights: {', '.join(changed)})", "a stored weight no longer holds the bytes of its load"
        status, unobserved, kernel_lines = tie_out.kernel_check(session)
    say("\n".join([*before.lines, *kernel_lines]))
    covered = len(session.traced.steps or []) if status != NOT_OBSERVED else 0  # the traced positions: the first so many timed steps
    # GPU busy, idle, kernels and copies of every traced step: the union of the intervals of the step's own launch records
    observed = (session.traced.steps or [])[:covered]
    facts_of = [fact for one in observed for fact in step_facts([(launch.name, launch.start, launch.end) for launch in one])]
    busy = [fact[0] for fact in facts_of] if len(facts_of) == covered else []
    unmeasured = "" if busy else (unobserved or "the launch records could not be cut into the traced steps")
    kernels = kernel_facts(session, status)
    passed = held == PASS and status != FAIL
    kinds = f"{session.traced.differences} kind{'s' if session.traced.differences != 1 else ''} of difference"
    third = {PASS: PASS, FAIL: f"FAIL ({kinds}: the kernels lines above name them)"}.get(status, f"{NOT_OBSERVED} ({unobserved})")
    failed = "a check did not hold" if not passed else ("kernels not observed" if status == NOT_OBSERVED else "")
    failed = failed or ("" if before.judged else "no provider block was judged")
    reduced = bool(getattr(model, "reductions", ()))
    rate_missing = f"not printed: {missing}" if held != PASS else (f"not printed: {NOT_THE_PLAN}" if status == FAIL else "")
    rate_missing = rate_missing or ("not printed for a reduced model" if reduced else "")
    alone, steady = observe.others_reason(watched), observe.clock_reasons(watched)
    kind, reasons = record_class(reduced, kernels["fallback_rows"], replayed, failed, alone, steady)
    weights = model.copy_bytes(session.uncopied_weights()) + model.stored_bytes(session.stored_weights())
    raw = []
    for made in requests:
        stamps = {"t_first_s": made.t_first - made.t_request, "t_launch_s": [t - made.t_request for t in made.t_launch]}
        stamps["t_token_s"] = [t - made.t_request for t in made.t_token]
        raw.append({"kind": made.kind, "t_request_s": 0.0, **stamps, "ids_sha256": made.ids_digest(), "end_state_sha256": made.state or None})
    env = environment
    frontend = {"version": env.cudnn_version, "distribution": env.distribution_version, "editable": env.editable, "source_hash": env.source_hash}
    gpu = {"name": facts.name, "compute_capability": f"{facts.cc[0]}.{facts.cc[1]}", "sm_count": facts.sm_count, "memory_gib": facts.total_bytes / GIB}
    host = {"logical_cores": os.cpu_count(), "usable_cores": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}
    host["load"] = {made.when: made.load for made in watch.readings}  # the load average over 1, 5 and 15 minutes at every reading
    teardown = {key: os.environ.get(key) for key in ("TEARDOWN_CUPTI", "DISABLE_CUPTI_LAZY_REINIT", "KINETO_LOG_LEVEL")}
    prompt_hash = hashlib.sha256(b"".join(int(i).to_bytes(4, "little") for prompt in prompts for i in prompt.tolist())).hexdigest()
    stats = _statistics(timed, covered, session.sequences, rate_missing, busy, traced, session.report().calls_per_step, unmeasured)
    # GPU busy of the replayed traced steps, where the run's steps are replays; such a run forms no launch overhead and no stretch
    again_steps = (session.traced.replayed or []) if replayed and session.traced is not None else []
    again_facts = [fact for one in again_steps for fact in step_facts([(launch.name, launch.start, launch.end) for launch in one])]
    again_busy = [fact[0] for fact in again_facts] if again_steps and len(again_facts) == len(again_steps) else []
    not_replayed = "not formed: the steps of this run are launched from Python"
    unreplayed = "" if again_busy else (replays_missing or "the launch records could not be cut into the replayed steps")
    unit, definition = DEFINITIONS["gpu_busy_replayed"]
    stats["gpu_busy_replayed"] = {
        "definition": definition,
        "unit": unit,
        "value": statistics.median(again_busy) if again_busy else None,
        "reason": None if again_busy else (f"GPU busy of a replayed step not measured ({unreplayed})" if replayed else not_replayed),
        **summary(again_busy),
    }
    if replayed:
        no_path = "not formed: a replayed step has no launch path from Python between its blocks"
        stats["launch_overhead"].update(value=None, reason=no_path, label=None, host_time_per_block_call_us=None)
        stats["tracing_stretch"].update(value=None, reason="not formed: the traced steps are launched from Python, the timed steps are replays")
    recorded = session.recorded if replayed else None
    data = {
        "schema": SCHEMA,
        "kind": "run",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "report_file": os.path.basename(os.path.normpath(report)),
        "record": {"class": kind, "reasons": reasons, "timing_role": ORIENTATION, "performance_claim_eligible": False},
        "model": {
            "name": name,
            "layers": len(model.geometry.layer_kinds),
            "sequences": session.sequences,
            "decoding": "greedy",
            "reduced": reduced,
            "reductions": list(getattr(model, "reductions", ())),
            "checkpoint": session.checkpoint.summary() if real and session.checkpoint is not None else None,
        },
        "environment": {
            "gpu": gpu,
            "python": platform.python_version(),
            "pytorch": env.torch_version,
            "pytorch_cuda": torch.version.cuda,
            "cudnn_frontend": frontend,
            "cute_dsl": env.dsl_version or None,
            "driver": watch.driver(),
            "profiler_teardown": teardown,
            "host": host,
        },
        "observation": watched,
        "kernels": kernels,
        "protocol": {
            "stack": session.stack,
            "launch_path": "captured" if replayed else "python",
            "launch_path_reason": None if replayed else session.python_reason,  # why the steps of this run are launched from Python
            "recorded_step": (
                None
                if recorded is None
                else {
                    "launches": None if recorded.launches is None else recorded.launches["launches"],
                    "held": recorded.launches,
                    "block_launches_planned": recorded.planned,
                }
            ),
            "warmup_steps": protocol.warmup,
            "timed_steps": protocol.steps,
            "timed_requests": protocol.requests,
            "new_tokens_per_request": protocol.steps + 1,
            "intervals_per_request": protocol.steps,
            "profiler_first_started": "behind the last timed request, in the traced request" if covered else None,  # None: no traced step was observed
            "stop_rule": "forced",
            "in_a_timed_request": {
                "recorder": "off (a record of 0 steps)",
                "call_log": "off",
                "profiler": "not started",
                "text": "none",
                "sampler": watch.in_a_request(),
            },
            "restart": "every request starts from the state zeroed in place at position 0, the device synchronised behind the zeroing",
            "prompt": {"tokens": prompt_tokens, "sha256": prompt_hash, "source": source},
            "first_timed_position": prompt_tokens,
            "last_timed_position": prompt_tokens + protocol.steps - 1,
        },
        "checks": {
            "judged_positions": [{"position": position, "why": why} for position, why in positions if position in before.positions],
            "positions_text": listed,
            "before": list(before.lines),
            "after": {"kernels": [line.strip() for line in kernel_lines], "stored_weights_changed": changed},
            "blocks": PASS if before.judged else "REPORTED (no provider block to judge)",
            "recorded_step": before.recorded or None,  # of a run whose steps are replays: the checked request launched again from Python
            "recomputed_steps": before.recomputed or None,
            "timed_requests": held,
            "kernels": third,
            "last_line": f"{tag} {verdict(passed, before.judged, unobserved)}",
        },
        "statistics": stats,
        "raw": {"requests": raw},
        "traced": {
            "steps": covered,
            "first_position": prompt_tokens if covered else None,
            "last_position": prompt_tokens + covered - 1 if covered else None,
            "record_budget": protocol.trace_records,
            "records_of_the_sizing_step": sizing,
            "gpu_busy_ms": stats["gpu_busy"]["value"],
            "gpu_busy_reason": unmeasured or None,
            "per_step": {key: [fact[index] for fact in facts_of] if busy else [] for index, key in enumerate(("gpu_busy_ms", "idle_ms", "kernels", "copies"))},
            "tracing_stretch_percent": stats["tracing_stretch"]["value"],
            "time_resolution_us": time_resolution_us(observed),
            "replayed": (
                {
                    "steps": len(again_steps) or (len(again.t_token) if again is not None else 0),
                    "first_position": prompt_tokens,
                    "last_position": prompt_tokens + (len(again_steps) or len(again.t_token)) - 1,
                    "launches": None if recorded is None or recorded.launches is None else recorded.launches["launches"],
                    "gpu_busy_ms": stats["gpu_busy_replayed"]["value"],
                    "reason": unreplayed or None,
                    "per_step": {
                        key: [fact[index] for fact in again_facts] if again_busy else []
                        for index, key in enumerate(("gpu_busy_ms", "idle_ms", "kernels", "copies"))
                    },
                }
                if replayed
                else None
            ),
        },
        "per_kernel": kernel_table(session.traced, busy, unmeasured),
        "memory": {
            "weights_gib": weights / GIB,
            "weights_gb": weights / 1e9,
            "state_gib": session.state.bytes() / GIB,
            "peak_reserved_gib": timed_peaks[1] / GIB if cuda else None,
            "peak_allocated_gib": timed_peaks[0] / GIB if cuda else None,
            "process_peak_reserved_gib": peaks[1] / GIB if cuda else None,
        },
        "startup": {f"{stage}_s": seconds for stage, seconds in stages.seconds.items()},
        "text_path": {f"{what}_s": seconds for what, seconds in {**(text_path or {}), **session.report().text_path}.items()},
    }
    data["statistics"].update(observe.statistic())
    # phase 11: the file, the screen, the text
    eos = frozenset(getattr(text, "eos_ids", ()))
    ended = next((index for index, token in enumerate(produced[0]) if token in eos), None)
    kept = produced[0] if ended is None else produced[0][: ended + 1]
    data["text_path"].update(tokens_written=len(kept) if real else 0, cut_at_an_end_token=ended is not None)
    unwritten = write_report(report, data)
    say("")
    say("\n".join(screen(data, unwritten)))
    stream = text.decoder(prompts[0])
    written = "".join(stream.push(token) for token in kept) + stream.close()
    if not real:
        say("\n".join(_wrapped("text out", f"sequence 0, {len(produced[0])} tokens (seeded weights: not language): {written[:48]!r}")))
    else:
        upto = f"all {len(produced[0])} new tokens (no end token)" if ended is None else f"to its first end token (token {ended + 1} of {len(produced[0])})"
        say("\n".join(_wrapped("text out", f"sequence 0, {upto}: {sink.reported(written)}")), flush=True)
        sink.write(written + "\n")
    return Outcome(passed, before.judged, unobserved, produced, unwritten, peaks, data)
