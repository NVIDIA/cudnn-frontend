# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a run of the performance mode observes of its GPU and of its host: the other compute processes on the GPU, the SM
clock, the limit reasons the driver reports, the host's load.

The driver is asked through its ``nvidia-smi`` tool, in child processes, and through nothing else: no package is added.
An ``Observer`` makes

* a **reading** where the run is at rest -- before the load, before the warm-up request, behind the last timed request
  (``Observer.read``): two bounded calls of the tool, for the device (driver version, SM clock, memory in use,
  utilisation) and for its list of compute processes, and the host's load average;
* the **sampler** (``Observer.start``, ``Observer.stop``): one child process, the keeper, which starts two loops of the
  tool -- the SM clock, the power and the clock event reasons; the list of compute processes -- each writing a line every
  ``PERIOD_MS`` into a file of its own, and then waits for its standard input to close.  The process that times holds the
  other end of that pipe and does nothing else with the child between the start and the stop: no read, no poll, no
  thread.  The loops end when the pipe closes, at the stop or when the run is gone for any reason: the keeper ends
  them, kills one that does not end, and on Linux none outlives its keeper.  The files are read behind the stop.

**The cut.**  The tool stamps every line on the host's wall clock.  A sample belongs to a request when its stamp lies
between the request being handed to the text path and its last token's ids on the host (``summary``); the bounds of a
request are brought to the wall clock by one pair of clock readings taken when the sampler starts, and a second pair
at its stop shows a wall clock that was set meanwhile -- by more than ``DRIFT_LIMIT_MS``, and the samples are not used.

**What is stated** (``others``, ``clocks``).  ``none seen`` only when no other process id was in a reading or in a sample,
this process's own id was in every reading and in a sample of every timed request, and the device's memory in use agreed
with this process's own in every reading: a list that hides processes ends in ``not observed``, and so does a list
that shows other ids and not this process's own -- no count is stated of it.  The clocks are ``yes``
with the median and the minimum over the samples of the timed requests, ``thin`` where a timed request has fewer than
``LEAST_SAMPLES``, or ``no``.  A lock is a declaration of the caller: it is held against the observed clock and never
set.  Whatever cannot be read is ``not observed`` with one of ``REASONS``, a closed list of phrases: never the tool's
path, its own words or an exception's text.
"""

from __future__ import annotations

import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from .timing import now

TOOL = "nvidia-smi"
PERIOD_MS = 50  # of the sampler's two loops
LEAST_SAMPLES = 10  # clock samples every timed request must have
FLOOR, CEILING = 0.97, 1.03  # the minimum against a declared lock, or against the median where none is declared; the maximum against a declared lock
MEMORY_SLACK_MIB = 64  # device memory in use above this process's own that still agrees with it
ASK_LIMIT_S = 10.0  # of one call of the tool, and of the wait for the keeper at the stop
LOOP_LIMIT_S = 3.0  # of the keeper's wait for a loop it has ended: a loop that is still there is killed
DRIFT_LIMIT_MS = 1.0  # the wall clock against the clock of the stamps, between the sampler's start and its stop: the resolution of the tool's stamp
FORMAT = "--format=csv,noheader,nounits"
READING_FIELDS = "driver_version,clocks.sm,memory.used,utilization.gpu"
CLOCK_FIELDS = "timestamp,clocks.sm,power.draw,clocks_event_reasons.active"
PROCESS_FIELDS = "timestamp,pid,used_memory"
#: the bits of the driver's clock event reasons, lowest first, by the tool's own field names; the first says that the GPU
#: idled when the sample was taken and is counted apart: it is no limit
EVENTS = (
    "gpu_idle",
    "applications_clocks_setting",
    "sw_power_cap",
    "hw_slowdown",
    "sync_boost",
    "sw_thermal_slowdown",
    "hw_thermal_slowdown",
    "hw_power_brake_slowdown",
    "display_clock_setting",
)
NOT_FOUND = "nvidia-smi not found"
NO_DEVICE = "no CUDA device"
NO_ID = "PyTorch names no id for this GPU"
NO_ANSWER = "nvidia-smi gave no answer"
TOOL_ERROR = "nvidia-smi ended with an error"
UNREAD = "nvidia-smi's answer could not be read"
NO_SAMPLER = "the sampler could not be started"
NOT_LISTED = "this process is not in the driver's list"
NO_MEMORY = "the driver reports no memory figure"
MEMORY_BEYOND = "device memory in use exceeds this process's own"
NO_PROCESS_SAMPLE = "no process sample inside a timed request"
NO_CLOCK_SAMPLE = "no clock sample inside a timed request"
NO_CLOCK = "the driver reports no SM clock"
CLOCK_MOVED = "the host's clock moved during the timed requests"
NO_READING = "no reading was made"
#: what the file's ``protocol`` says of the sampler inside a timed request
SAMPLER = "a child process that the timing process neither reads nor polls between its start and its stop"
#: every reason a ``not observed`` can carry
REASONS = (
    NOT_FOUND,
    NO_DEVICE,
    NO_ID,
    NO_ANSWER,
    TOOL_ERROR,
    UNREAD,
    NO_SAMPLER,
    NOT_LISTED,
    NO_MEMORY,
    MEMORY_BEYOND,
    NO_PROCESS_SAMPLE,
    NO_CLOCK_SAMPLE,
    NO_CLOCK,
    CLOCK_MOVED,
    NO_READING,
)
#: the child that keeps the two loops: it ends them when its standard input closes or holds a byte, or when it is ended,
#: and kills a loop that has not ended within its limit; on Linux a loop is bound to its keeper (the parent-death signal,
#: prctl option 1), so that none outlives a keeper that was killed
KEEPER = f"""
import signal, subprocess, sys
signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
tool, target, period, limit, *files = sys.argv[1:]
try:
    import ctypes
    prctl = ctypes.CDLL(None).prctl
except (ImportError, OSError, AttributeError):
    prctl = None
bound = (lambda: prctl(1, int(signal.SIGKILL))) if prctl else None
loops = []
try:
    for query, path in zip(("--query-gpu={CLOCK_FIELDS}", "--query-compute-apps={PROCESS_FIELDS}"), files):
        with open(path, "wb") as out:
            command = [tool, "--id=" + target, query, "{FORMAT}", "--loop-ms=" + period]
            loops.append(subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.DEVNULL, preexec_fn=bound))
    sys.stdin.buffer.read(1)
finally:
    for loop in loops:
        loop.terminate()
    for loop in loops:
        try:
            loop.wait(float(limit))
        except subprocess.TimeoutExpired:
            loop.kill()
            loop.wait()
"""

#: statistic -> (unit, definition, serving counterpart): the one definition of what this module observes, as the report
#: file carries it
DEFINITIONS: Dict[str, Tuple[str, str, Optional[str]]] = {
    "gpu_and_host": (
        "MHz, W, MiB",
        (
            "Per timed request: the SM clock samples (count, minimum, median, mean, maximum), the power median and the limit reasons the driver reports; "
            "other compute processes on the GPU before, during and after the timed requests; device memory in use and utilisation before the load; the "
            "host's load average and the cores this process may use. Read from the driver by a child process; a field that cannot be read is not observed "
            "with the reason. A sample belongs to a request when its time stamp lies between the request being handed to the text path and its last "
            "token's ids on the host. The screen's clock is the median and the minimum over the samples of all timed requests. none seen is stated only "
            "when this process itself is in the driver's list and the device's memory in use agrees with this process's own. <n> seen is the number of "
            "distinct other compute processes over every reading and every sample, the reading before the load and the time between two requests "
            "included, and is stated only when no reading and no timed request's samples showed a list without this process itself; of such a list "
            "no count is stated: not observed, with the reason."
        ),
        "The same observations as a serving benchmark's sampler makes.",
    ),
}


class Clock(NamedTuple):
    """One line of the clock loop: the stamp in seconds of the wall clock, the SM clock in MHz, the power in W and the bit
    mask of the clock event reasons; ``None`` for a field the driver does not give."""

    at: float
    mhz: Optional[float]
    watts: Optional[float]
    events: Optional[int]


class Seen(NamedTuple):
    """One line of the process loop: the stamp, a compute process's id and its memory on the device in MiB."""

    at: float
    pid: int
    mib: Optional[float]


def _number(text: str) -> Optional[float]:
    """A field of the tool that is a size -- MHz, W, MiB, percent -- or ``None``: a field the driver does not give ("[N/A]",
    "[Not Supported]"), and one that is no size (not finite, or below zero)."""
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) and value >= 0.0 else None


def _stamp(text: str) -> Optional[float]:
    """The tool's time stamp, ``YYYY/MM/DD HH:MM:SS.mmm`` in local time, in seconds of the host's wall clock."""
    whole, _, milli = text.partition(".")
    try:
        return time.mktime(time.strptime(whole, "%Y/%m/%d %H:%M:%S")) + int(milli or 0) / 1000
    except (ValueError, OverflowError):
        return None


def clock_sample(line: str) -> Optional[Clock]:
    """A line of the clock loop, or ``None`` for a line that is not one."""
    fields = [part.strip() for part in line.split(",")]
    at = _stamp(fields[0]) if len(fields) == 4 else None
    if at is None:
        return None
    try:
        events = int(fields[3], 16)
    except ValueError:
        events = None
    return Clock(at, _number(fields[1]) or None, _number(fields[2]), events)  # a clock of zero is no clock


def process_sample(line: str) -> Optional[Seen]:
    """A line of the process loop, or ``None`` for a line that is not one."""
    fields = [part.strip() for part in line.split(",")]
    at = _stamp(fields[0]) if len(fields) == 3 else None
    if at is None or not fields[1].isdigit():
        return None
    return Seen(at, int(fields[1]), _number(fields[2]))


def limits_of(events: int) -> List[str]:
    """The limit reasons of one sample's bit mask, by the driver's names; a bit this module does not know as ``bit <n>``.
    ``gpu_idle`` is not among them."""
    known = [name for bit, name in enumerate(EVENTS) if bit and events >> bit & 1]
    return known + [f"bit {bit}" for bit in range(len(EVENTS), events.bit_length()) if events >> bit & 1]


def host_load() -> Optional[List[float]]:
    """The host's load average over the last 1, 5 and 15 minutes; ``None`` on a host that has none."""
    try:
        return list(os.getloadavg())
    except (OSError, AttributeError):
        return None


def host_cores() -> int:
    """The cores this process may use."""
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def target_of(device) -> str:
    """The run's GPU as the tool takes it for ``--id=``: by its UUID, which no list of visible devices renumbers."""
    import torch

    return f"GPU-{torch.cuda.get_device_properties(device).uuid}"


def _ask(tool: str, target: str, query: str) -> Tuple[List[List[str]], str]:
    """One bounded call of the tool: ``(the rows of its answer, "")``, or no row and the reason -- one of ``REASONS``,
    never the tool's own words."""
    try:
        command, text = [tool, f"--id={target}", query, FORMAT], {"encoding": "utf-8", "errors": "replace"}
        done = subprocess.run(command, capture_output=True, timeout=ASK_LIMIT_S, stdin=subprocess.DEVNULL, **text)
    except subprocess.TimeoutExpired:
        return [], NO_ANSWER
    except OSError:
        return [], TOOL_ERROR
    if done.returncode:
        return [], TOOL_ERROR
    return [[part.strip() for part in line.split(",")] for line in done.stdout.splitlines() if line.strip()], ""


@dataclass
class Reading:
    """The GPU and the host where the run is at rest.  ``reason``: why the driver could not be read (``""``: it was)."""

    when: str
    reason: str = ""
    driver: Optional[str] = None
    clock: Optional[float] = None  # MHz
    used: Optional[float] = None  # the device's memory in use, MiB
    utilization: Optional[float] = None  # percent
    listed: bool = False  # this process is in the driver's list
    own: Optional[float] = None  # its memory on the device, MiB
    others: Dict[int, Optional[float]] = field(default_factory=dict)  # every other compute process: id -> MiB
    load: Optional[List[float]] = None
    seconds: float = 0.0  # what the reading took

    def agrees(self) -> bool:
        """The device's memory in use is this process's own, ``MEMORY_SLACK_MIB`` aside."""
        return self.used is not None and self.own is not None and self.used - self.own <= MEMORY_SLACK_MIB

    def data(self) -> dict:
        """What the file holds of a reading: of another process its memory, never its id or its name."""
        made = {"when": self.when, "reason": self.reason or None, "host_load": self.load, "seconds": self.seconds}
        if not self.reason:
            made.update(
                sm_clock_mhz=self.clock,
                utilization_percent=self.utilization,
                device_memory_used_mib=self.used,
                this_process_listed=self.listed,
                this_process_mib=self.own,
                memory_agrees=self.agrees(),
                other_compute_processes=len(self.others),
                other_memory_mib=list(self.others.values()),
            )
        return made


def reading(tool: str, target: str, pid: int, when: str) -> Reading:
    """One reading: the device and its list of compute processes through two calls of the tool, and the host's load."""
    start = now()
    made = Reading(when, load=host_load())
    device, made.reason = _ask(tool, target, f"--query-gpu={READING_FIELDS}")
    listed = []
    if not made.reason:
        listed, made.reason = _ask(tool, target, "--query-compute-apps=pid,used_memory")
    if not made.reason and (len(device) != 1 or len(device[0]) != 4 or any(len(row) != 2 or not row[0].isdigit() for row in listed)):
        made.reason = UNREAD
    if not made.reason:
        made.driver = None if device[0][0].startswith("[") else device[0][0] or None  # "[N/A]": a version the driver does not give
        made.clock, made.used, made.utilization = _number(device[0][1]) or None, _number(device[0][2]), _number(device[0][3])
        made.others = {int(row[0]): _number(row[1]) for row in listed}
        made.listed = pid in made.others
        made.own = made.others.pop(pid, None)
    made.seconds = now() - start
    return made


def _pair() -> Tuple[float, float]:
    """One moment on both clocks, ``(wall clock, the clock of the stamps)``: the wall clock read between two readings of
    the other, and of three such readings the one whose two lie closest -- a read that was put off between its clocks
    is not the one taken."""
    tries = []
    for _ in range(3):
        before, wall, after = now(), time.time(), now()
        tries.append((after - before, wall, (before + after) / 2))
    return min(tries)[1:]


class Sampler:
    """The two loops of the tool behind their keeper.  ``start`` returns ``""`` or why there is no sampler; ``stop`` ends the
    loops and reads their files into ``clocks`` and ``processes``.  Nothing else is called between the two."""

    def __init__(self, tool: str, target: str, period: int = PERIOD_MS):
        self.tool, self.target, self.period = tool, target, period
        self.clocks: List[Clock] = []
        self.processes: List[Seen] = []
        self.unread = 0  # lines of the two files that are not samples
        self.child = self.folder = None
        self.started = self.stopped = (0.0, 0.0)  # (wall clock, the clock of the stamps) at the start and at the stop

    def start(self) -> str:
        if not sys.executable:
            return NO_SAMPLER
        try:
            self.folder = tempfile.TemporaryDirectory(prefix="perf-observe-")
        except OSError:  # no folder for temporary files on this host
            return NO_SAMPLER
        files = [os.path.join(self.folder.name, name) for name in ("clocks.csv", "processes.csv")]
        command = [sys.executable, "-I", "-S", "-c", KEEPER, self.tool, self.target, str(self.period), str(LOOP_LIMIT_S), *files]
        try:
            self.child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            self.folder.cleanup()
            return NO_SAMPLER
        self.started = _pair()
        return ""

    def stop(self) -> None:
        if self.child is None:
            return
        child, self.child = self.child, None
        self.stopped = _pair()
        try:
            child.stdin.write(b"\n")
            child.stdin.close()
        except OSError:  # the keeper is gone already
            pass
        try:
            child.wait(timeout=ASK_LIMIT_S)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        for name, parse, into in (("clocks.csv", clock_sample, self.clocks), ("processes.csv", process_sample, self.processes)):
            try:
                with open(os.path.join(self.folder.name, name), encoding="utf-8", errors="replace") as file:
                    samples = [parse(line) for line in file if line.strip()]
            except OSError:
                samples = []
            into.extend(sample for sample in samples if sample is not None)
            self.unread += sum(sample is None for sample in samples)
        self.folder.cleanup()


def _share(values: Sequence[float], below: Optional[float]) -> Optional[float]:
    return None if below is None or not values else sum(value < below for value in values) / len(values)


def request_facts(kind: str, lines: Sequence[Clock], seen: Sequence[Seen], pid: int, lock: Optional[int] = None) -> dict:
    """What the file holds of the samples of one request: the SM clock (count, minimum, median, mean, maximum, the share
    of samples below a declared lock), the power median, the limit reasons, the samples in which the driver said the GPU
    idled, and of the process list whether this process was in it and how many others were."""
    mhz = [sample.mhz for sample in lines if sample.mhz is not None]
    watts = [sample.watts for sample in lines if sample.watts is not None]
    masks = [sample.events for sample in lines if sample.events is not None]
    limits = sorted({name for mask in masks for name in limits_of(mask)})
    clock = {"samples": len(mhz), "minimum_mhz": None, "median_mhz": None, "mean_mhz": None, "maximum_mhz": None}
    if mhz:
        clock.update(minimum_mhz=min(mhz), median_mhz=statistics.median(mhz), mean_mhz=statistics.fmean(mhz), maximum_mhz=max(mhz))
    clock["share_below_declared_lock"] = _share(mhz, lock)
    return {
        "kind": kind,
        "clock": clock,
        "clock_lines": len(lines),
        "power_median_w": statistics.median(watts) if watts else None,
        "limit_reasons": limits,
        "idle_samples": sum(mask & 1 for mask in masks),
        "process_samples": len(seen),
        "this_process_listed": any(sample.pid == pid for sample in seen),
        "other_compute_processes": len({sample.pid for sample in seen} - {pid}),
    }


def others(readings: Sequence[Reading], requests: Sequence[dict], strangers: int, failed: str = "") -> dict:
    """Condition C5, the other compute processes: ``seen`` with their count where another process id was in a reading or
    in a sample -- and this process itself was in every reading that was made and in the samples of every timed request
    that has any: of a list that shows other ids and not this process's own no count is stated, one of them may be this
    process under another id; else ``not observed`` with the reason where the driver could not be read (``failed``: why
    there are no samples), a timed request has no process sample, this process is missing from a reading or from the
    samples of a timed request, the driver gives no memory figure, or the device's memory in use is not this process's
    own; else ``none seen``."""
    timed = [made for made in requests if made["kind"] == "timed"]
    hidden = any(not made.reason and not made.listed for made in readings) or any(made["process_samples"] and not made["this_process_listed"] for made in timed)
    if strangers and not hidden:
        return {"verdict": "seen", "seen": strangers, "reason": None}
    reason = NOT_LISTED if strangers else next((made.reason for made in readings if made.reason), failed)
    if not reason and not (timed and all(made["process_samples"] for made in timed)):
        reason = NO_PROCESS_SAMPLE
    if not reason and not (readings and all(made.listed for made in readings) and all(made["this_process_listed"] for made in timed)):
        reason = NOT_LISTED
    if not reason and any(made.used is None or made.own is None for made in readings):
        reason = NO_MEMORY
    if not reason and not all(made.agrees() for made in readings):
        reason = MEMORY_BEYOND
    return {"verdict": "not observed" if reason else "none seen", "seen": None if reason else 0, "reason": reason or None}


def clocks(requests: Sequence[dict], pooled: Sequence[float], lock: Optional[int] = None, failed: str = "") -> dict:
    """Condition C6, the clocks, from the facts of the timed requests and their SM clock samples, pooled: ``no`` with the
    reason, ``thin`` where a timed request has fewer than ``LEAST_SAMPLES``, else ``yes``; and every reason for which the
    clocks are not a controlled run's -- too few samples; the minimum below ``FLOOR`` of the median, or of a declared
    lock; the maximum above ``CEILING`` of a declared lock; a limit reason."""
    timed = [made for made in requests if made["kind"] == "timed"]
    limits = sorted({name for made in timed for name in made["limit_reasons"]})
    made = {"verdict": "no", "reason": None, "samples": None, "median_mhz": None, "minimum_mhz": None, "maximum_mhz": None, "limit_reasons": limits}
    if failed or not pooled:
        lines = any(request["clock_lines"] for request in timed)
        return {**made, "reason": failed or (NO_CLOCK if lines else NO_CLOCK_SAMPLE), "reasons": ["clocks not observed"]}
    fewest, median, low, high = min(request["clock"]["samples"] for request in timed), statistics.median(pooled), min(pooled), max(pooled)
    reasons = ["too few clock samples"] if fewest < LEAST_SAMPLES else []
    if lock is None and low < FLOOR * median:
        reasons.append(f"SM clock fell to {low:.0f} MHz")
    if lock is not None and low < FLOOR * lock:
        reasons.append("below the declared lock")
    if lock is not None and high > CEILING * lock:
        reasons.append("above the declared lock")
    if limits:
        reasons.append("limit reported: " + ", ".join(limits))
    made.update(verdict="thin" if fewest < LEAST_SAMPLES else "yes", samples=fewest, median_mhz=median, minimum_mhz=low, maximum_mhz=high)
    return {**made, "reasons": reasons}


def summary(
    readings: Sequence[Reading],
    clock_samples: Sequence[Clock],
    process_samples: Sequence[Seen],
    requests: Sequence[Tuple[str, float, float]],
    pid: int,
    lock: Optional[int] = None,
    failed: str = "",
) -> dict:
    """The file's ``observation`` section.  ``requests``: ``(kind, start, stop)`` of every request the sampler ran beside,
    on the clock of the samples' stamps; a sample belongs to the request whose bounds hold its stamp, both ends included.
    ``failed``: why the samples cannot be used (``""``: they can)."""
    facts, pooled = [], []
    for kind, start, stop in requests:
        inside = [sample for sample in clock_samples if start <= sample.at <= stop]
        facts.append(request_facts(kind, inside, [sample for sample in process_samples if start <= sample.at <= stop], pid, lock))
        pooled += [sample.mhz for sample in inside if sample.mhz is not None and kind == "timed"]
    strangers = {other for made in readings for other in made.others} | {sample.pid for sample in process_samples}
    last = readings[-1] if readings else None
    return {
        "source": TOOL,
        "reason": None,
        "sampler_period_ms": PERIOD_MS,
        "clock_lock_declared_mhz": lock,
        "rules": {
            "least_clock_samples": LEAST_SAMPLES,
            "clock_floor_percent": round(FLOOR * 100),
            "clock_ceiling_percent": round(CEILING * 100),
            "memory_slack_mib": MEMORY_SLACK_MIB,
        },
        "readings": [made.data() for made in readings],
        "requests": facts,
        "other_compute_processes": others(readings, facts, len(strangers - {pid}), failed),
        "clocks": clocks(facts, pooled, lock, failed),
        "host": {"load": last.load[0] if last is not None and last.load else None, "cores": host_cores()},
    }


def others_text(watched: dict) -> str:
    """The value of ``other compute processes:`` on the screen, from the file's ``observation`` section."""
    seen = watched["other_compute_processes"]
    if seen["verdict"] == "seen":
        return f"{seen['seen']} seen"
    return "none seen" if seen["verdict"] == "none seen" else f"not observed ({seen['reason']})"


def clocks_text(watched: dict) -> str:
    """The value of ``clocks observed:`` on the screen.  A lock that was declared is named in every form."""
    made, lock = watched["clocks"], watched["clock_lock_declared_mhz"]
    declared = "" if lock is None else f"; lock {lock} MHz declared"
    if made["verdict"] == "no":
        return f"no ({made['reason']}{declared})"
    if made["verdict"] == "thin":
        return f"thin ({made['samples']} sample{'s' if made['samples'] != 1 else ''} in a timed request{declared})"
    limits = "limit reported: " + ", ".join(made["limit_reasons"]) if made["limit_reasons"] else "no limit reported"
    return f"yes (SM {made['median_mhz']:.0f} MHz median, {made['minimum_mhz']:.0f} min{declared}; {limits})"


def host_text(watched: dict) -> str:
    """The host's load on the screen: its 1-minute load average behind the last timed request, and the cores this
    process may use."""
    host = watched["host"]
    return "host load not observed" if host["load"] is None else f"host load {host['load']:.1f} on {host['cores']} core{'s' if host['cores'] != 1 else ''}"


def others_reason(watched: dict) -> str:
    """Why the run cannot say it had the GPU alone, for the record class (``""``: it can)."""
    return {"seen": "shared GPU", "not observed": "other processes not observed"}.get(watched["other_compute_processes"]["verdict"], "")


def clock_reasons(watched: dict) -> List[str]:
    """Every reason for which the clocks are not a controlled run's, for the record class."""
    return list(watched["clocks"]["reasons"])


def statistic() -> dict:
    """The entry of the file's ``statistics`` section: the definition, beside the section it defines."""
    unit, definition, counterpart = DEFINITIONS["gpu_and_host"]
    entry = {"definition": definition, "serving_counterpart": counterpart, "unit": unit, "value": None, "reason": "see the section observation of this file"}
    return {"gpu_and_host": entry}


class Observer:
    """The observation of one run: its readings, its sampler and the section of the file.  ``lock``: the SM clock in MHz
    that the caller declares as fixed outside the run; nothing here sets a clock.  ``reason``: why nothing can be asked of
    the driver (``""``: it can be asked)."""

    def __init__(self, device, lock: Optional[int] = None):
        self.lock, self.pid = lock, os.getpid()
        self.readings: List[Reading] = []
        self.sampler: Optional[Sampler] = None
        self.failed = ""  # why the sampler gave nothing
        self.tool = self.target = None
        self.reason = "" if getattr(device, "type", None) == "cuda" else NO_DEVICE
        if not self.reason:
            self.tool = shutil.which(TOOL)
            self.reason = "" if self.tool else NOT_FOUND
        if not self.reason:
            try:
                self.target = target_of(device)
            except (AttributeError, RuntimeError, AssertionError):
                self.reason = NO_ID

    def read(self, when: str) -> None:
        """A reading where the run is at rest; without a tool, the host's load alone."""
        self.readings.append(Reading(when, self.reason, load=host_load()) if self.reason else reading(self.tool, self.target, self.pid, when))

    def start(self) -> None:
        """The sampler's child process: called outside every request, in front of the warm-up request."""
        if not self.reason:
            self.sampler = Sampler(self.tool, self.target)
            self.failed = self.sampler.start()

    def stop(self) -> None:
        """The loops end and their files are read: called outside every request, behind the traced request."""
        if self.sampler is not None:
            self.sampler.stop()

    def in_a_request(self) -> str:
        """What the sampler is to a timed request, for the file's ``protocol``: ``SAMPLER`` where its child process was
        started, else that none ran, with the reason."""
        if self.sampler is not None and not self.failed:
            return SAMPLER
        return f"none ({self.reason or self.failed or 'not started'})"

    def driver(self) -> dict:
        """The driver's version for the file's ``environment``, or why it is not there."""
        version = next((made.driver for made in self.readings if made.driver), None)
        missing = self.reason or next((made.reason for made in self.readings if made.reason), "") or (UNREAD if self.readings else NO_READING)
        return {"value": version, "reason": None if version else f"not observed ({missing})"}

    def section(self, requests: Sequence[Tuple[str, float, float]]) -> dict:
        """The file's ``observation`` section; ``requests``: ``(kind, t_request, last t_token)`` of every request behind the
        sampler's start, on the clock of the run's stamps."""
        if self.reason:
            made = summary(self.readings, [], [], [], self.pid, self.lock, self.reason)
            return {**made, "source": "none", "reason": self.reason, "sampler_period_ms": None, "unread_lines": None, "clock_offset_drift_ms": None}
        sampler, failed = self.sampler, self.failed
        offset = sampler.started[0] - sampler.started[1] if sampler is not None else 0.0  # the stamps' clock to the wall clock
        drift = (sampler.stopped[0] - sampler.stopped[1] - offset) * 1e3 if sampler is not None else None
        if drift is not None and not failed and abs(drift) > DRIFT_LIMIT_MS:
            failed = CLOCK_MOVED
        bounds = [(kind, start + offset, stop + offset) for kind, start, stop in requests]
        clock_samples, process_samples = ([], []) if sampler is None or failed else (sampler.clocks, sampler.processes)
        made = summary(self.readings, clock_samples, process_samples, bounds, self.pid, self.lock, failed)
        period = sampler.period if sampler is not None else None  # of the loops as they were started
        return {**made, "sampler_period_ms": period, "unread_lines": sampler.unread if sampler is not None else None, "clock_offset_drift_ms": drift}
