# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The performance mode of a model sample: timed requests of the decode loop, one definition per statistic, a report on the
screen and a report file.

A **timed request** is what a serving benchmark calls a request: one prompt from a zero state, its first new token, then a
fixed number of forced decode steps (the end token is ignored).  A run of the mode (``run``) is one process and makes, in
this order:

1. the **checked request**, with the recorder of the checks on, and the reference checks on it (``tie_out.run_checks``):
   replay over the whole request, every provider block against the reference at the **judged positions**
   (``judged_positions``), the tokens, the stored weights.  A check that fails ends the run: nothing is timed, no file;
2. a **warm-up request** and the **timed requests**, each from the state zeroed in place, with nothing but the step inside:
   no recorder, no call log, no profiler, no text;
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
``DEFINITIONS`` holds the one definition of every statistic; the file carries each beside its number.  The GPU side of a
step (GPU busy, and what a step spends between its launches) is not taken from the traced request in this version: the
``step`` line says so.

**What a run may call itself** (``record_class``).  A run is a ``controlled`` run only when it observed every condition
itself -- a checkpoint, every block on a claimed target kernel, a captured launch sequence, the three statements of its
``checks`` lines, no other process on the GPU, the clocks -- and prints ``orientation`` with every reason otherwise.
Nothing in this module observes the GPU's other processes or its clock yet, and no stack of this area is captured: every
run is ``orientation``, on the second line of the screen and inside the rate line.  A report holds no speed-up, no
baseline and no figure of another GPU class.
"""

from __future__ import annotations

import ctypes
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

from . import census, tie_out
from .timing import now

SCHEMA = "model-sample-performance/1"
WIDTH = 110  # of the screen: no line is wider
ORIENTATION, CONTROLLED = "orientation", "controlled"
PASS, FAIL, NOT_OBSERVED = census.PASS, census.FAIL, census.NOT_OBSERVED
GIB = 2.0**30
#: why this version states nothing about the GPU's other processes and its clock
UNOBSERVED = "this version does not observe the GPU"
NOT_COMPUTED = "the timed requests did not compute the checked request's tokens"
NOT_THE_PLAN = "the kernels of the traced steps are not the plan's"

#: statistic -> (unit, definition, serving counterpart or None): the sentences the report file carries beside each number
DEFINITIONS: Dict[str, Tuple[str, str, Optional[str]]] = {
    "time_per_output_token": (
        "ms",
        (
            "Per timed request: (time the last token's ids reach the host - time the first new token's ids reach the host) / (new tokens - 1). Of a run: the "
            "median over its timed requests."
        ),
        (
            "The same formula as a serving benchmark's time per output token, (end of stream - first text chunk) / (output tokens - 1). Differences: the clock "
            "is inside the process, the requests repeat one prompt, and a timed request of N steps delivers N + 1 new tokens where a serving request of N "
            "output tokens delivers N."
        ),
    ),
    "decode_rate": (
        "tokens/s per sequence",
        "1000 / time per output token in ms. Never a median step turned into a rate. With several sequences the line adds the total.",
        (
            "The same as a serving benchmark's tokens per second per user, 1000 / time per output token. It is not output tokens / wall time of the run, which "
            "contains every time to first token; that figure is in this file as tokens_per_second_wall."
        ),
    ),
    "token_to_token_interval": (
        "ms",
        (
            "The time between the host reads of two consecutive tokens of a timed request, pooled over the run's timed requests; percentiles are nearest-rank. "
            "The intervals span a growing attention cache as well as jitter: the file gives the first and the last timed position."
        ),
        None,
    ),
    "time_to_first_token": (
        "ms",
        (
            "Per timed request: from the request being handed to the text path to the first new token's ids on the host -- render, tokenize, the copy of the "
            "ids, the prompt's decode steps (one per prompt token: this sample has no prefill kernel) and the host read. Of a run: the median over its timed "
            "requests. The load is outside it."
        ),
        (
            "The same pair of events as a serving benchmark's time to first token without the network; not the same quantity: a serving engine prefills the "
            "prompt in one pass and adds its queue and its network path."
        ),
    ),
    "step": (
        "ms",
        (
            "The median token-to-token interval over the timed steps at the positions the traced request covers, all timed requests. Equal to the median of the"
            " token-to-token interval when the whole request is traced. This version does not take GPU busy from the traced request, so the step is not taken "
            "apart."
        ),
        "A serving engine's step is not measured directly: it is its time per output token, times the kept tokens per step with draft tokens.",
    ),
    "host": (
        "ms",
        (
            "From a token's ids on the host to the next step's first launch call: the loop's own work (keeping the ids); the median over the timed steps at the"
            " traced positions."
        ),
        None,
    ),
    "launches_per_step": (
        "launches",
        (
            "From the plan and the kernel census: the launches of a step that are on the target kernel set, by provider and as three numbers -- the target's "
            "own compiled kernels, cuDNN Frontend entries with byte-equal results, declared stand-ins. Observed by the kernel check on the traced steps, which "
            "also counts the kernels and copies a step launches beside them."
        ),
        "The verdict of a kernel census.",
    ),
    "memory": (
        "GiB",
        (
            "Weights, state and cache: the family's arithmetic over the tensors held on the device, in GiB (2^30 bytes). Peak reserved: PyTorch's peak between "
            "the emptied pool behind the checks and the end of the timed requests. The file also gives the weights in GB (10^9 bytes), the unit of an ordinary "
            "run's bytes per step line, and the whole process's peak."
        ),
        None,
    ),
    "startup": (
        "s",
        (
            "The seconds of every stage of the load, of the checks and of the requests, the device synchronised at both ends of a stage. None of it enters a "
            "rate or the time to first token."
        ),
        None,
    ),
}


@dataclass(frozen=True)
class Protocol:
    """The numbers of a run of the mode: forced decode steps of the warm-up request, timed decode steps of a request, timed
    requests, and the most launch records the traced request may hold."""

    warmup: int = 32
    steps: int = 256
    requests: int = 3
    trace_records: int = 160_000

    def decode_steps(self, prompt_tokens: int) -> int:
        """The most decode steps the requests of a run make: the checked request, the warm-up request, the timed requests
        and a traced request of every timed step, each behind the prompt's own steps."""
        return (2 + self.requests) * (prompt_tokens + self.steps) + prompt_tokens + self.warmup

    def line(self, prompt_tokens: int) -> str:
        """What a run of the mode will do, said before the load: a rule and counts, never a time."""
        each = f"{self.requests} timed request{'s' if self.requests != 1 else ''} of {self.steps} decode steps each"
        return (
            f"performance mode: at most {self.decode_steps(prompt_tokens):,} decode steps in all: one checked request, one warm-up request of "
            f"{self.warmup} decode steps behind its first new token, {each}, one traced request of up to {self.steps}; a prompt of {prompt_tokens} "
            f"tokens is {prompt_tokens} decode steps in front of each; the checks replay the checked request once"
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


def record_class(
    reduced: bool,
    fallback_rows: bool,
    identity: str,
    captured: bool,
    checks: str,
    others: str = "other processes not observed",
    clocks: str = "clocks not observed",
) -> Tuple[str, List[str]]:
    """``(class, reasons)`` of a run, the reasons in the order of the conditions.  A run is ``controlled`` only when every
    condition was observed by the run itself; each argument is a condition, or the reason it is not met (``""``: met):
    ``reduced`` -- no checkpoint; ``fallback_rows`` -- a block of the plan is on a fallback row; ``identity`` -- why the
    kernels are not a claimed target set on this GPU class; ``captured`` -- the launch sequence is captured, not launched
    from Python; ``checks`` -- why the three statements of the ``checks`` lines do not all hold; ``others`` -- why the run
    cannot say it had the GPU alone; ``clocks`` -- why it cannot state the clock."""
    reasons = [
        "reduced model" if reduced else "",
        "fallback rows present" if fallback_rows else "",
        identity,
        "" if captured else "launched from Python",
        checks,
        others,
        clocks,
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


def kernel_facts(session, status: str) -> dict:
    """What the ``kernels`` lines say: from the plan and the family's census, the launches of a step that are on the target
    kernel set, by provider and as three numbers; the block calls on fallback rows; and, from the traced steps, the
    kernels and copies a step launches beside the target launches (``None`` where they were not observed)."""
    plan, model, claim = session.plan, session.model, session.claim()
    served = [plan.at(name, site) for (name, _), site in zip(model.wiring(), census.call_sites(model))]
    facts = {"declared": claim is not None, "stated": bool(claim is not None and claim.known), "fallback_calls": sum(bool(row.fallback) for row, _ in served)}
    facts["fallback_rows"] = any(row.fallback for _, _, _, row, _ in plan.parts())
    facts["census"] = None if claim is None else claim.line(status)
    facts["not_stated"] = "" if claim is None or claim.known else claim.coverage()[0]
    if facts["stated"]:
        numbers = claim.numbers()
        on_target = [identity.relation in (census.TARGET, census.STAND_IN) for identity in claim.identities]
        providers = Counter(block.provider.replace("cuDNN FE", "cuDNN Frontend") for (_, block), held in zip(served, on_target) if held)
        stand_ins = sum(identity.relation == census.STAND_IN for identity in claim.identities)
        facts.update(
            launches=numbers["launches"],
            code=numbers["target_code"],
            port=numbers["target_port"],
            stand_ins=stand_ins,
            by_provider=dict(providers.most_common()),
        )
        facts["absent"] = len(claim.absent)
        facts["on_target"] = facts["code"] + facts["port"] + stand_ins
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
    second = f"timed requests give the checked request's tokens and state: {checks['timed_requests']}"
    third = f"kernels, traced steps: {checks['kernels']}"
    if 12 + len(second) + 3 + len(third) <= WIDTH:
        return lines + [" " * 12 + f"{second} | {third}"]
    return lines + _wrapped("", second, 12) + _wrapped("", third, 12)


def screen(data: dict) -> List[str]:
    """The screen of a run, rendered from the data the report file holds and from nothing else: at most ``WIDTH`` characters
    a line.  Every time of a reduced model carries ``reduced model:``, and such a run prints no rate."""
    model, gpu, kernels, protocol, traced = data["model"], data["environment"]["gpu"], data["kernels"], data["protocol"], data["traced"]
    stats, memory, record = data["statistics"], data["memory"], data["record"]
    prefix = "reduced model: " if model["reduced"] else ""
    lines = ["cuDNN Frontend model sample: decode performance report"]
    if record["class"] == CONTROLLED:
        lines += ["record class          controlled run in one process; a number of record is the median of 3 such runs"]
    else:
        first, *more = record["reasons"]
        lines += _packed("record class", [f"{record['class']}, not a number of record: {first}", *more], joint="; ", hang=";")
    lines += ["scope       a decode-loop rate, not a served rate: one process, no request queue, no scheduler, no network"]
    sequences = f"{model['sequences']} sequence{'s' if model['sequences'] != 1 else ''}"
    lines += _packed("model", [f"{model['name']}, {model['layers']} layers", sequences, model["decoding"]], 12)
    seen = f"other compute processes: not observed ({data['observation']['reason_short']})"
    lines += _packed("GPU", [f"{gpu['name']}, compute capability {gpu['compute_capability']}, {gpu['sm_count']} SMs", seen], 12)
    lines += [" " * 12 + f"clocks observed: no ({data['observation']['reason']})"]
    calls = f"{kernels['fallback_calls'] or 'no'} block call{'s' if kernels['fallback_calls'] != 1 else ''} on fallback rows"
    if kernels.get("absent"):  # a launch of the target that no block of the plan stands for
        calls += f" + {kernels['absent']} launch{'es' if kernels['absent'] != 1 else ''} of the target that {'have' if kernels['absent'] != 1 else 'has'} no block yet"
    also = "also " if kernels["stated"] else ""
    beside = f"what a step {also}launches was not observed"
    if kernels["kernels_per_step"] is not None:
        others, copies = kernels["kernels_per_step"] - (kernels.get("on_target") or 0), kernels["copies_per_step"]
        beside = f"a step {also}launches {others:,} kernel{'s' if others != 1 else ''} and {copies:,} cop{'ies' if copies != 1 else 'y'}"
    if kernels["stated"]:
        providers = ", ".join(f"{provider} {count}" for provider, count in kernels["by_provider"].items()) or "none"
        lines += _wrapped("kernels", f"{kernels['on_target']} of {kernels['launches']} launches per step are on the target kernel set: {providers}", 12)
        ports = f"{kernels['port']} cuDNN Frontend entr{'y' if kernels['port'] == 1 else 'ies'} with byte-equal results"
        lines += [" " * 12 + f"= {kernels['code']} the target's own compiled kernel{'s' if kernels['code'] != 1 else ''} + {ports}"]
        lines += [" " * 14 + f"+ {kernels['stand_ins']} declared stand-in{'s' if kernels['stand_ins'] != 1 else ''}"]
        lines += _packed("", [f"not yet: {calls}", beside], 12, "; ")
    else:
        lines += _wrapped("kernels", kernels["not_stated"] or "no target kernel set is declared for this stack", 12)
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
    covered = "over every timed step"
    if traced["steps"] == 1:
        covered = f"at the traced position, {traced['first_position']}"
    elif traced["steps"]:
        covered = f"at the {traced['steps']} traced positions, {traced['first_position']} to {traced['last_position']}"
    lines += _wrapped("step", f"{prefix}{stats['step']['value']:.3f}  ms {covered}; GPU busy not measured ({traced['gpu_busy_reason']})")
    lines += ["with draft tokens     not run by this script (plain decode)"]
    peak = "peak reserved not observed" if memory["peak_reserved_gib"] is None else f"peak reserved {memory['peak_reserved_gib']:.1f} GiB"
    held = [f"weights {memory['weights_gib']:.1f} GiB", f"state and cache {memory['state_gib']:.2f} GiB", f"{peak}  (GiB = 2^30 bytes)"]
    lines += _packed("memory", held)
    lines += [""] + _packed("report file", [data["report_file"], "(every number, its definition, the protocol, the environment)"], joint=" ")
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


def _statistics(timed: Sequence[Request], covered: int, sequences: int, rate_missing: str) -> dict:
    """The entries of the file's ``statistics`` section, each with its definition and its serving counterpart.  ``covered``:
    the traced positions, the first so many steps of every timed request (0: none was observed, and the step is taken
    over every timed step)."""

    def entry(key: str, **values) -> dict:
        unit, definition, counterpart = DEFINITIONS[key]
        return {"definition": definition, "serving_counterpart": counterpart, "unit": unit, "clock": "time.perf_counter, one process", **values}

    per_token = [request.time_per_output_token_ms() for request in timed]
    first = [request.time_to_first_token_ms() for request in timed]
    intervals = [interval for request in timed for interval in request.intervals_ms()]
    value, last = statistics.median(per_token), covered or None
    wall = [len(request.ids) / (request.t_token[-1] - request.t_request) for request in timed]
    spread = (max(per_token) - min(per_token)) / value * 100
    rate = (
        {"value": None, "reason": rate_missing, "total": None} if rate_missing else {"value": 1000 / value, "reason": None, "total": sequences * 1000 / value}
    )
    stats = {
        "time_per_output_token": entry("time_per_output_token", events="t_token[last] - t_first", value=value, per_request=per_token, spread_percent=spread),
        "decode_rate": entry("decode_rate", **rate),
        "token_to_token_interval": entry("token_to_token_interval", events="t_token[i] - t_token[i - 1]", value=statistics.median(intervals)),
        "time_to_first_token": entry("time_to_first_token", events="t_first - t_request", value=statistics.median(first), per_request=first),
        "step": entry("step", events="t_token[i] - t_token[i - 1] at the traced positions"),
        "host": entry("host", events="t_launch[i] - t_token[i - 1] at the traced positions"),
        "tokens_per_second_wall": {"unit": "tokens/s per sequence", "value": statistics.median(wall), "per_request": wall},
    }
    samples = {"time_per_output_token": per_token, "token_to_token_interval": intervals, "time_to_first_token": first}
    samples["step"] = [interval for request in timed for interval in request.intervals_ms()[:last]]
    samples["host"] = [host for request in timed for host in request.host_ms()[:last]]
    for key, of in samples.items():
        stats[key].update(summary(of), value=stats[key].get("value", statistics.median(of)))
    for key, section in (("launches_per_step", "kernels"), ("memory", "memory"), ("startup", "startup")):
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
) -> Outcome:
    """One run of the mode on a loaded session (the module docstring has the phases).  ``tokenize()`` is the text path of
    the request -- render and tokenize -- and returns one prompt per sequence; ``say`` prints a line of the run;
    ``name`` is the model as the screen names it; ``facts`` the device's; ``environment`` the versions the run stated;
    ``sink`` takes the text of sequence 0 once, behind the report; ``report`` is the report file; ``verdict(passed,
    judged, unobserved)`` is the script's last line without its tag; ``source`` says where the prompt came from."""
    import torch

    model, device, stages = session.model, session.device, session.stages
    cuda = device.type == "cuda"

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

    def peaks_now() -> Tuple[int, int]:
        return (torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)) if cuda else (0, 0)

    # phase 3: the checked request, launched from Python with the recorder on, and the reference checks on it
    prompts = tokenize()  # for the counts, the prompt's hash and the text: every request renders and tokenizes for itself
    prompt_tokens = int(prompts[0].numel())
    positions = judged_positions(prompt_tokens, protocol.steps, getattr(model, "attention_page", None))
    judged = [position for position, _ in positions]
    listed = tie_out.positions_text(judged)
    say(
        f"{tag} performance mode: the checked request ({prompt_tokens} prompt tokens through the decode step, the first new token, {protocol.steps} forced "
        f"steps), then the checks against the {session.reference_precision}-accumulated reference: replay, the blocks at positions {listed}, the tokens",
        flush=True,
    )
    with stages.stage("the checked request"):
        checked = request("checked", protocol.steps)
    produced = [[read[sequence] for read in checked.ids] for sequence in range(session.sequences)]
    with stages.stage("the checks before"):
        before = tie_out.run_checks(session, prompt_tokens=prompt_tokens, max_steps=16, second_reference=real, kernels=tie_out.DEFERRED, positions=judged)
    peaks = peaks_now()
    if not before.ok:  # as an ordinary run: its lines, and nothing is timed; no file
        say(before.text())
        return Outcome(False, before.judged, "", produced, peaks=peaks)

    # phases 5 to 7: settle, the warm-up request, the timed requests -- nothing but the step inside, and no profiler pass so far
    say(
        f"{tag} performance mode: a warm-up request, {protocol.requests} timed request{'s' if protocol.requests != 1 else ''}, then a traced request and "
        "the checks behind it; the report follows",
        flush=True,
    )
    session.reset(fresh=False, recorded=False)  # the record of the checks is dropped before the pool is emptied
    gc.collect()  # ... and so are the reference sessions of the checks, with their states
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    requests = [checked]
    with stages.stage("the warm-up request"):
        requests.append(request("warm-up", protocol.warmup))
    with stages.stage("the timed requests"):
        for _ in range(protocol.requests):
            session.reset(fresh=False, recorded=False)
            requests.append(request("timed", protocol.steps))
    timed, timed_peaks = requests[2:], peaks_now()
    peaks = (max(peaks[0], timed_peaks[0]), max(peaks[1], timed_peaks[1]))

    # phase 9: the traced request -- the first profiler pass of the process: one step, whose records size the rest
    with stages.stage("the traced request"):
        session.reset(fresh=False, recorded=False)
        traced, step = begin("traced")
        session.trace_request(step, 1)
        first = session.traced
        sizing = len(first.steps[0]) if first.steps else 0
        more = min(protocol.steps, protocol.trace_records // sizing) - 1 if sizing else 0
        if more > 0:
            session.trace_request(step, more)
            if session.traced.steps is not None:  # one traced request: the records and the call logs of both passes
                both = (first.call_log + session.traced.call_log, first.steps + session.traced.steps)
                session.traced = census.Traced(first.sites, *both, rows=session.sequences, claim=first.claim)
        requests.append(traced)

    # phase 10: the checks after
    with stages.stage("the checks after"):
        held = PASS
        for number, made in enumerate([*timed, traced], 1):
            who = f"request {number}" if made.kind == "timed" else "traced request"
            differs = next((index for index, read in enumerate(made.ids) if read != checked.ids[index]), None)
            if differs is not None and held == PASS:
                held = f"FAIL ({who}, step {differs})"
            if made.kind == "timed" and made.state != checked.state and held == PASS:
                held = f"FAIL ({who}, end state)"
        changed = tie_out.changed_weights(session.weights)
        if changed and held == PASS:
            held = f"FAIL (stored weights: {', '.join(changed)})"
        status, unobserved, kernel_lines = tie_out.kernel_check(session)
    say("\n".join([*before.lines, *kernel_lines]))
    covered = len(session.traced.steps or []) if status != NOT_OBSERVED else 0  # the traced positions: the first so many timed steps
    kernels = kernel_facts(session, status)
    passed = held == PASS and status != FAIL
    kinds = f"{session.traced.differences} kind{'s' if session.traced.differences != 1 else ''} of difference"
    third = {PASS: PASS, FAIL: f"FAIL ({kinds}: the kernels lines above name them)"}.get(status, f"{NOT_OBSERVED} ({unobserved})")
    identity = "kernel identity not claimed on this GPU class"  # of a stack that declares a target: unless its census line is EQUAL
    if not kernels["declared"]:
        identity = "no target kernel set declared"
    elif ": EQUAL (" in kernels["census"]:
        identity = ""
    failed = "a check did not hold" if not passed else ("kernels not observed" if status == NOT_OBSERVED else "")
    failed = failed or ("" if before.judged else "no provider block was judged")
    reduced = bool(getattr(model, "reductions", ()))
    rate_missing = f"not printed: {NOT_COMPUTED}" if held != PASS else (f"not printed: {NOT_THE_PLAN}" if status == FAIL else "")
    rate_missing = rate_missing or ("not printed for a reduced model" if reduced else "")
    kind, reasons = record_class(reduced, kernels["fallback_rows"], identity, False, failed)
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
    teardown = {key: os.environ.get(key) for key in ("TEARDOWN_CUPTI", "DISABLE_CUPTI_LAZY_REINIT", "KINETO_LOG_LEVEL")}
    prompt_hash = hashlib.sha256(b"".join(int(i).to_bytes(4, "little") for prompt in prompts for i in prompt.tolist())).hexdigest()
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
            "driver": {"value": None, "reason": f"not observed ({UNOBSERVED})"},
            "profiler_teardown": teardown,
            "host": host,
        },
        "observation": {"source": "none", "reason": UNOBSERVED, "reason_short": "this version", "other_compute_processes": None, "clocks": None},
        "kernels": kernels,
        "protocol": {
            "stack": session.stack,
            "launch_path": "python",
            "warmup_steps": protocol.warmup,
            "timed_steps": protocol.steps,
            "timed_requests": protocol.requests,
            "new_tokens_per_request": protocol.steps + 1,
            "intervals_per_request": protocol.steps,
            "profiler_first_started": "behind the last timed request, in the traced request",
            "stop_rule": "forced",
            "in_a_timed_request": {"recorder": "off (a record of 0 steps)", "call_log": "off", "profiler": "not started", "text": "none"},
            "restart": "every request starts from the state zeroed in place at position 0",
            "prompt": {"tokens": prompt_tokens, "sha256": prompt_hash, "source": source},
            "first_timed_position": prompt_tokens,
            "last_timed_position": prompt_tokens + protocol.steps - 1,
        },
        "checks": {
            "judged_positions": [{"position": position, "why": why} for position, why in positions],
            "positions_text": listed,
            "before": list(before.lines),
            "after": {"kernels": [line.strip() for line in kernel_lines], "stored_weights_changed": changed},
            "blocks": PASS if before.judged else "REPORTED (no provider block to judge)",
            "timed_requests": held,
            "kernels": third,
            "last_line": f"{tag} {verdict(passed, before.judged, unobserved)}",
        },
        "statistics": _statistics(timed, covered, session.sequences, rate_missing),
        "raw": {"requests": raw},
        "traced": {
            "steps": covered,
            "first_position": prompt_tokens,
            "last_position": prompt_tokens + covered - 1,
            "record_budget": protocol.trace_records,
            "records_of_the_sizing_step": sizing,
            "gpu_busy_ms": None,
            "gpu_busy_reason": "this version" if covered else (unobserved or "the launch records could not be cut into the traced steps"),
        },
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
    # phase 11: the file, the screen, the text
    eos = frozenset(getattr(text, "eos_ids", ()))
    ended = next((index for index, token in enumerate(produced[0]) if token in eos), None)
    kept = produced[0] if ended is None else produced[0][: ended + 1]
    data["text_path"].update(tokens_written=len(kept) if real else 0, cut_at_an_end_token=ended is not None)
    unwritten = write_report(report, data)
    say("")
    say("\n".join(screen(data)))
    stream = text.decoder(prompts[0])
    written = "".join(stream.push(token) for token in kept) + stream.close()
    if not real:
        say("\n".join(_wrapped("text out", f"sequence 0, {len(produced[0])} tokens (seeded weights: not language): {written[:48]!r}")))
    else:
        upto = f"all {len(produced[0])} new tokens (no end token)" if ended is None else f"to its first end token (token {ended + 1} of {len(produced[0])})"
        say("\n".join(_wrapped("text out", f"sequence 0, {upto}: {sink.reported(written)}")), flush=True)
        sink.write(written + "\n")
    return Outcome(passed, before.judged, unobserved, produced, unwritten, peaks, data)
