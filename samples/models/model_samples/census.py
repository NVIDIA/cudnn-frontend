# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The kernel census of a decode step: which kernels a step launched, in which shape, and whether those are the planned launches.

The served-by table of a sample says who serves a block.  This module holds the same statement one level down, at the kernel
launches.  Three things meet here:

* **what a block declares** -- its census entries (:class:`Kernel`, ``Block.kernels``): per kernel a canonical id, how many launches
  one call makes, and where the expected launch shape comes from.  An entry is ``exact`` (checked one by one: name, order,
  count, and the launch shape where a source is named), ``fallback`` (a reference chain or a multiply over a copy: its
  launches are attributed by family, their number is not a fact of the row) or ``host`` (the sample's own host path, listed
  and not compared);
* **what the adapters wrote down** -- the call log (:class:`Call`): for every block call of a traced step the call site, the entry
  instance that served it and the fields of the row's form as THAT instance states them, read from the object that was
  called after the call (``Block.serving``), never copied from the row;
* **what the profiler recorded** -- one :class:`Launch` per kernel and per copy of the traced steps, read from the profiler's
  exported trace (:func:`read_trace`): the name, the grid, the block, the shared memory of a thread block.

The kernel check (:func:`check_forms`, :func:`compare`; a sample runs it with ``--check-kernels``):

0. **Form, on the objects.**  Every field a provider row names in its ``form`` equals the entry's own form for the run's row
   count, and every logged call was served by an entry instance that states those same values.  A row of an entry that states
   its launch names at least one field: with none the entry would be called on its own pick, and nothing would be compared.
   This is the only place a form is shown: see below.
1. Every launch of a traced step is attributed.  The exact kernels must be there in the wiring order; a launch between two of
   them must belong to a family that a fallback or host row between them declares.  An unattributed kernel fails the run and is
   named.
2. Per kernel the launches of a step equal the plan's: calls x launches per call.
3. The exact launches come in the wiring order.
4. No kernel of a fallback family sits between two exact rows that have no fallback row between them.
5. An exact launch whose entry states its launch (``Block.launch_facts``) carries one of the entry's own symbols and exactly the
   entry's grid, block and shared memory.
6. Copies are listed.  Declared are the one host read that ends a step and the copies a fallback or host row makes; a copy
   between two exact rows is not.

**A kernel name proves nothing about a form.**  A generated kernel symbol is the kernel's name followed by the text of its
compile-time arguments, cut at a fixed length: two forms of one kernel can share a symbol, and they can share a launch shape
too.  So the form is compared on the objects (item 0), never read out of a name; the name and the launch shape (item 5) show
that the launch is the entry's and has the shape the entry states.

Where the profiler cannot run, or its records carry no launch arguments, nothing was observed: the check then says so and a
sample reports ``RAN ... (kernels not observed: <reason>)`` instead of ``PASS``.  A form that does not hold fails either way.

This module imports nothing beyond the standard library.
"""

from __future__ import annotations

import difflib
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Mapping, NamedTuple, Optional, Sequence, Tuple

RELATIONS = ("exact", "fallback", "host")
#: where the expected launch shape of an exact kernel comes from: the entry's own launch facts, or nowhere (the name alone)
SHAPE_SOURCES = ("", "entry")
PASS, FAIL, NOT_OBSERVED = "PASS", "FAIL", "NOT OBSERVED"


@dataclass(frozen=True)
class Kernel:
    """One census entry of a block: a kernel the block launches, or a family of kernels a fallback row launches."""

    id: str  # a canonical kernel id ("gdn_decode_fused") or, for a fallback or host entry, the id of a family ("pytorch")
    origin: str  # who compiled it, in the words of the provider column: "cuDNN FE" | "cuBLAS" | "PyTorch"
    relation: str = "fallback"  # one of RELATIONS
    launches: Optional[int] = None  # per call of the block; None where the number is not a fact of the row (a chain of operators)
    shape: str = ""  # one of SHAPE_SOURCES

    def __post_init__(self):
        if self.relation not in RELATIONS or self.shape not in SHAPE_SOURCES:
            raise ValueError(f"census entry {self.id}: relation {self.relation!r} / shape source {self.shape!r} is not declared")
        if self.relation == "exact" and (not isinstance(self.launches, int) or self.launches < 1):
            raise ValueError(f"census entry {self.id}: an exact kernel states its launches per call")
        if self.relation != "exact" and (self.launches is not None or self.shape):
            raise ValueError(f"census entry {self.id}: only an exact kernel states launches per call and a launch shape")


class Launch(NamedTuple):
    """One kernel launch or one copy as the profiler recorded it.  Times are milliseconds on the profiler's clock; a launch
    argument the record does not carry is ``None``."""

    name: str
    kind: str  # "kernel" | "copy"
    start: float
    end: float
    grid: Optional[Tuple[int, int, int]] = None
    block: Optional[Tuple[int, int, int]] = None
    shared_memory: Optional[int] = None
    registers: Optional[int] = None  # per thread: recorded, never compared (it moves with the compiler)
    graph: Tuple[int, int] = (0, 0)  # (graph id, graph node id): zero for a launch issued from Python
    bytes: Optional[int] = None  # of a copy

    @property
    def shape(self) -> tuple:
        return (self.grid, self.block, self.shared_memory)


class Call(NamedTuple):
    """One line of the call log: the block call ``(block, layer)`` at ``rows`` rows was served by ``instance``, which states
    ``form`` for the fields the provider row names."""

    block: str
    layer: int
    rows: int
    instance: str  # the entry instance, as the adapter names it; "" for a block without one
    form: tuple  # (field, value) pairs as the serving instance states them; () where nothing was passed


class Facts(NamedTuple):
    """What an entry states about its launch at one row count: the symbols of the kernels it has compiled for it, its form as
    plain data and its launch shape ``(grid, block, shared memory)``."""

    symbols: Tuple[str, ...]
    form: Mapping[str, object]
    shape: tuple


@dataclass
class Site:
    """One block call of a step in wiring order, with what is expected of it."""

    block: str
    layer: int
    kernels: Tuple[Kernel, ...]
    entry: str = ""  # what is called, as the table prints it
    form: tuple = ()  # the provider row's form
    facts: Optional[Facts] = None  # of an entry that states its launch
    wrong: str = ""  # why such an entry stated none: a wrong table

    @property
    def name(self) -> str:
        return self.block if self.layer < 0 else f"{self.block} (layer {self.layer})"


# ---------------------------------------------------------------------------------------------------------- canonical ids
#: kernel name -> canonical id, first match.  Symbols of cuDNN Frontend entries come first; then the kernels of other libraries
#: and of PyTorch that the blocks of this area launch.  A chain that gains an operator whose kernel no pattern names adds its
#: pattern in the same change: an unnamed kernel fails the kernel check by name.
FAMILIES: Tuple[Tuple[str, "re.Pattern"], ...] = (
    ("gdn_decode_fused", re.compile(r"^cudnn_kernel_frost_gdn_decode_fused_")),
    ("cudnn_fe", re.compile(r"^cudnn_kernel_frost_")),  # a kernel of cuDNN Frontend that no entry above names: never a family member
    ("embedding_gather", re.compile(r"\bat::native::(?:\(anonymous namespace\)::)?indexSelect(?:Small|Large)Index<")),
    ("cublas", re.compile(r"cublas|^nvjet_|\bcutlass::Kernel2?<cutlass_|^(?:maxwell|volta|ampere)_\w*gem[mv]", re.IGNORECASE)),
    ("pytorch", re.compile(r"\bat::native::|\(anonymous namespace\)::(?:softmax_warp_forward|elementwise_kernel_with_index)<")),
)
#: a kernel with an id of its own that is also a member of a family.  Where a plan has an exact row for it, every launch of
#: it in a step is held against that row (a chain that launched the same kernel would count as an extra launch: no chain of
#: this area does); where it has none, the kernel counts as its family's
FAMILY_OF = {"embedding_gather": "pytorch"}
FAMILY_IDS = ("cublas", "pytorch")
COPY = "copy"
#: a copy is a record of the profiler's copy or fill categories -- or, for the copy node of a replayed launch graph that was
#: captured before the process's first profiler pass, a kernel record of this name
COPY_NODE = re.compile(r"^memcpy\d+_post$")
HOST_READ = "Memcpy DtoH"


def copy_name(name: str) -> bool:
    """Whether a record's name is a copy's, in either spelling."""
    return name.startswith(("Memcpy", "Memset")) or bool(COPY_NODE.match(name))


def canonical(record: Launch) -> str:
    """The canonical id of a record: ``"copy"`` for a copy, the id of the first pattern of :data:`FAMILIES` that names the
    kernel, ``""`` for a kernel no pattern names."""
    if record.kind == COPY or COPY_NODE.match(record.name):
        return COPY
    for kernel_id, pattern in FAMILIES:
        if pattern.search(record.name):
            return kernel_id
    return ""


# ---------------------------------------------------------------------------------------------------------- the trace
_SKIP = re.compile(r"[\s,]*")
_RECORDS = {"kernel": "kernel", "gpu_memcpy": COPY, "gpu_memset": COPY}


def trace_events(path: str, chunk: int = 1 << 20) -> Iterator[dict]:
    """The events of an exported profiler trace (``torch.profiler``'s ``export_chrome_trace``: one JSON object with a
    ``traceEvents`` array), one at a time.  The file is read in chunks and never held as a whole: it grows with the launches
    of a step (32 steps of a two-layer model: about 30 MB).  ``ValueError`` for a file that is not such a trace or ends
    inside an event."""
    decoder = json.JSONDecoder()
    with open(path, encoding="utf-8") as trace:
        buffer, position = trace.read(chunk), -1
        while position < 0:
            key = buffer.find('"traceEvents"')
            position = buffer.find("[", key) if key >= 0 else -1
            if position < 0:
                more = trace.read(chunk)
                if not more:
                    raise ValueError("the exported trace has no traceEvents array")
                buffer = (buffer[key:] if key >= 0 else buffer[-16:]) + more
        position += 1
        while True:
            position = _SKIP.match(buffer, position).end()
            if position < len(buffer) and buffer[position] == "]":
                return
            try:
                if position >= len(buffer):
                    raise ValueError("the buffer ends between two events")
                event, end = decoder.raw_decode(buffer, position)
            except ValueError:  # the buffer ends inside an event (json.JSONDecodeError is a ValueError), or between two
                more = trace.read(chunk)
                if not more:
                    raise ValueError("the exported trace ends inside its traceEvents array") from None
                buffer, position = buffer[position:] + more, 0
                continue
            position = end
            if isinstance(event, dict):
                yield event


def _triple(value) -> Optional[Tuple[int, int, int]]:
    return tuple(int(v) for v in value) if isinstance(value, (list, tuple)) and len(value) == 3 else None


def _integer(value) -> Optional[int]:
    return None if value is None else int(value)


def read_trace(path: str, chunk: int = 1 << 20) -> List[Launch]:
    """The kernel and copy records of an exported profiler trace, in start order.  Every other record of the trace (the
    runtime's calls, flow arrows, metadata) is passed over."""
    names: Dict[str, str] = {}  # one string per distinct name: a traced pass repeats a few dozen names thousands of times
    records = []
    for event in trace_events(path, chunk):
        kind = _RECORDS.get(event.get("cat"))
        if kind is None:
            continue
        arguments = event.get("args") or {}
        name = str(event.get("name", ""))
        start = float(event.get("ts", 0.0)) / 1e3
        records.append(
            Launch(
                names.setdefault(name, name),
                kind,
                start,
                start + float(event.get("dur", 0.0)) / 1e3,
                _triple(arguments.get("grid")),
                _triple(arguments.get("block")),
                _integer(arguments.get("shared memory")),
                _integer(arguments.get("registers per thread")),
                (int(arguments.get("graph id") or 0), int(arguments.get("graph node id") or 0)),
                _integer(arguments.get("bytes")),
            )
        )
    records.sort(key=lambda record: record.start)
    return records


def split_steps(records: Sequence[Launch]) -> Tuple[List[List[Launch]], List[Launch]]:
    """``(steps, rest)``: a step ends with its host read, the record of a device-to-host copy, which is the last record of
    its list; records after the last host read belong to no step."""
    steps, current = [], []
    for record in records:
        current.append(record)
        if record.name.startswith(HOST_READ):
            steps.append(current)
            current = []
    return steps, current


# ---------------------------------------------------------------------------------------------------------- the plan's side
def expected(plan, model, rows: Optional[int] = None) -> List[Site]:
    """The block calls of one step in the model's wiring order (``model.wiring()``), each with its block's census entries, the
    provider row's form and -- for an entry that states its launch -- the entry's facts at ``rows`` (default: the plan's
    sequences x tokens).  An entry that declines to state them for a row count its row claims is recorded on the site
    (``wrong``): the check then fails, because the table is wrong."""
    rows = plan.sequences * plan.tokens if rows is None else rows
    stated: Dict[str, Tuple[Optional[Facts], str]] = {}
    sites = []
    for name, layer in model.wiring():
        block = plan[name]
        if name not in stated:
            facts, wrong = None, ""
            if any(kernel.shape == "entry" for kernel in block.kernels):
                try:
                    facts = block.launch_facts(rows)
                except (ValueError, NotImplementedError) as error:
                    wrong = str(error)
                if facts is None and not wrong:
                    wrong = "the adapter states no launch facts"
            stated[name] = (facts, wrong)
        sites.append(Site(name, layer, tuple(block.kernels), block.entry, tuple(plan.rows[name].form), *stated[name]))
    return sites


def form_text(form: Sequence[Tuple[str, object]]) -> str:
    """A form as the table prints it: `` [PDL False, PDL_RELEASE False]``, or nothing for an empty form."""
    return " [" + ", ".join(f"{name} {value}" for name, value in form) + "]" if form else ""


def shape_text(shape) -> str:
    grid, block, shared = shape
    return f"grid {tuple(grid)}, block {tuple(block)}, shared memory {shared:,} B"


def _short(name: str, width: int = 96) -> str:
    return repr(name if len(name) <= width else name[: width - 3] + "...")


def check_forms(sites: Sequence[Site], call_log: Sequence[Sequence[Call]] = ()) -> List[str]:
    """Item 0, on the objects: for every block whose entry states its launch, the provider row names a form and each field
    it names equals the entry's own form; and every logged call of a block was served by an entry instance that states the
    row's form (``Call.form``: what the adapter read from the object it called).  Returns the sentences of what does not
    hold."""
    failures = []
    for site in {site.block: site for site in sites}.values():
        if site.wrong:
            failures.append(f"{site.block}: {site.entry} states no launch for this run ({site.wrong}): the provider table is wrong")
        if not site.form and any(kernel.shape == "entry" for kernel in site.kernels):
            failures.append(f"{site.block}: the row names no form; {site.entry} would be called on its own pick")
        if site.facts is None:
            continue
        for name, value in site.form:
            if name not in site.facts.form:
                failures.append(f"{site.block}: the row's form names {name}, which the form of {site.entry} does not have")
            elif site.facts.form[name] != value:
                failures.append(f"{site.block}: the row's form says {name} {value!r}, {site.entry} states {name} {site.facts.form[name]!r} for this run")
    by_site = {(site.block, site.layer): site for site in sites}
    served = {}  # (block, the form the serving instance states) -> the first call site: one sentence per kind of difference
    for calls in call_log:
        for call in calls:
            site = by_site.get((call.block, call.layer))
            if site is not None and tuple(call.form) != tuple(site.form):
                served.setdefault((call.block, tuple(call.form)), (site, call))
    for site, call in served.values():
        failures.append(
            f"{site.name}: the adapter served the call with the form{form_text(call.form) or ' []'} ({call.instance or 'no entry instance'}); "
            f"the row says{form_text(site.form) or ' []'}"
        )
    return failures


@dataclass
class Comparison:
    """What :func:`compare` found."""

    failures: List[Tuple[int, str]] = field(default_factory=list)  # (step, sentence)
    unobserved: str = ""  # why a launch shape could not be compared, when the records carry none
    steps: int = 0
    launches: List[int] = field(default_factory=list)  # kernels per step
    copies: List[int] = field(default_factory=list)  # copies per step, the host read included
    exact: int = 0  # exact launches per step, as planned
    families: Dict[str, List[int]] = field(default_factory=dict)  # family -> attributed launches per step
    picks: Dict[str, Counter] = field(default_factory=dict)  # family -> kernel name -> the most launches in one step

    @property
    def ok(self) -> bool:
        return not self.failures

    def sentences(self, limit: int = 8) -> List[str]:
        """The failures, one line per distinct sentence with the steps it holds for, at most ``limit`` of them."""
        steps: Dict[str, List[int]] = {}
        for step, sentence in self.failures:
            steps.setdefault(sentence, []).append(step)
        lines = []
        for sentence, at in list(steps.items())[:limit]:
            more = f" (and {len(at) - 1} more of the {self.steps} traced steps)" if len(at) > 1 else ""
            lines.append(f"step {at[0]}{more}: {sentence}")
        if len(steps) > limit:
            lines.append(f"... and {len(steps) - limit} more kinds of difference")
        return lines


def _span(counts: Sequence[int]) -> str:
    return "0" if not counts else (f"{min(counts):,}" if min(counts) == max(counts) else f"{min(counts):,} to {max(counts):,}")


def compare(steps: Sequence[Sequence[Launch]], sites: Sequence[Site], call_log: Sequence[Sequence[Call]] = ()) -> Comparison:
    """Items 1 to 6 of the kernel check: the launch records of every traced step (each list ends with the step's host read,
    as :func:`split_steps` cuts them) against the sites of one step in wiring order (:func:`expected`); and, where a call log
    is given, the block calls each step made against the wiring."""
    anchors = [(index, kernel) for index, site in enumerate(sites) for kernel in site.kernels if kernel.relation == "exact" for _ in range(kernel.launches)]
    planned = [kernel.id for _, kernel in anchors]
    anchor_ids = set(planned)
    gaps = []  # per gap between two exact launches (before the first, after the last): the families declared there, and its text
    for gap in range(len(anchors) + 1):
        low = anchors[gap - 1][0] if gap else -1
        high = anchors[gap][0] if gap < len(anchors) else len(sites)
        if low == high:  # two launches of one exact row: nothing may sit between them
            gaps.append((set(), f"inside the launches of {sites[low].name}"))
            continue
        declared = {kernel.id for site in sites[max(low, 0) : min(high + 1, len(sites))] for kernel in site.kernels if kernel.relation != "exact"}
        if low < 0:
            where = f"before {sites[high].name}" if high < len(sites) else "in a step whose plan has no exact kernel"
        elif high >= len(sites):
            where = f"after {sites[low].name}"
        else:
            where = f"between {sites[low].name} and {sites[high].name}"
        gaps.append((declared, where))
    wiring = [(site.block, site.layer) for site in sites]
    result = Comparison(steps=len(steps), exact=len(anchors))
    fail = result.failures.append
    for index, step in enumerate(steps):
        calls = [(call.block, call.layer) for call in call_log[index]] if index < len(call_log) else wiring
        if calls != wiring:
            first = next((n for n, (a, b) in enumerate(zip(calls, wiring)) if a != b), min(len(calls), len(wiring)))
            had = Site(*calls[first], ()).name if first < len(calls) else "nothing"
            want = sites[first].name if first < len(sites) else "nothing"
            fail((index, f"the step made {len(calls)} block calls, the wiring has {len(wiring)}; call {first + 1} is {had}, the wiring has {want} there"))
        body = step[:-1] if step and step[-1].name.startswith(HOST_READ) else list(step)
        ids = [canonical(record) for record in body]
        result.launches.append(sum(kernel_id != COPY for kernel_id in ids))
        result.copies.append(len(step) - result.launches[-1])
        at = [position for position, kernel_id in enumerate(ids) if kernel_id in anchor_ids]
        found = [ids[position] for position in at]
        if found != planned:
            for sentence in _sequence_sentences(planned, found, anchors, sites, [body[position] for position in at]):
                fail((index, sentence))
            continue
        bounds = [-1, *at, len(body)]
        seen: Counter = Counter()
        names: Dict[str, Counter] = {}
        for gap, (declared, where) in enumerate(gaps):
            for position in range(bounds[gap] + 1, bounds[gap + 1]):
                record, kernel_id = body[position], ids[position]
                if kernel_id == COPY:
                    if not declared:
                        fail(
                            (
                                index,
                                f"an undeclared copy {_short(record.name)} {where}: declared are the one host read of a step and the copies of a fallback row",
                            )
                        )
                    continue
                family = kernel_id if kernel_id in declared else FAMILY_OF.get(kernel_id, "")
                if family and family in declared:
                    seen[family] += 1
                    names.setdefault(family, Counter())[record.name] += 1
                elif not kernel_id:
                    fail((index, f"launched {_short(record.name)} {where}; no block of the plan launches it"))
                elif kernel_id in FAMILY_IDS or family:
                    none = "; nothing may sit between two exact rows" if not declared else ""
                    fail(
                        (
                            index,
                            f"launched {_short(record.name)} {where}: a kernel of the fallback family {family or kernel_id}, which no row there declares{none}",
                        )
                    )
                else:
                    fail((index, f"launched {_short(record.name)} {where}: the kernel {kernel_id}, which no block of the plan declares"))
        for family in {family for declared, _ in gaps for family in declared}:
            result.families.setdefault(family, []).append(seen[family])
        for family, counts in names.items():
            picks = result.picks.setdefault(family, Counter())
            for name, count in counts.items():
                picks[name] = max(picks[name], count)
        for position, (site_index, kernel) in zip(at, anchors):
            record, site = body[position], sites[site_index]
            if kernel.shape != "entry" or site.facts is None:
                continue
            if record.name not in site.facts.symbols:  # two symbols of one kernel differ late in their text: both are printed whole
                stated = ", ".join(_short(symbol, 240) for symbol in site.facts.symbols) or "no compiled kernel"
                fail((index, f"{site.name} launched {_short(record.name, 240)}; {site.entry} names {stated} for this run"))
            elif None in record.shape:
                result.unobserved = "the profiler's kernel records carry no launch arguments (grid, block, shared memory)"
            elif record.shape != tuple(site.facts.shape):
                fail((index, f"{site.name} launched {kernel.id} with {shape_text(record.shape)}; {site.entry} states {shape_text(site.facts.shape)}"))
    return result


def _sequence_sentences(planned, found, anchors, sites, records) -> List[str]:
    """Why the exact launches of a step are not the planned ones: a count that differs (items 1 and 2: a missing or an extra
    launch, each named), or the same launches in another order (item 3)."""
    if Counter(planned) == Counter(found):
        first = next(n for n, (a, b) in enumerate(zip(planned, found)) if a != b)
        swapped = first + 1 < len(planned) and planned[first] == found[first + 1] and planned[first + 1] == found[first]
        site = sites[anchors[first][0]]
        other = f"; it is swapped with {sites[anchors[first + 1][0]].name}" if swapped else ""
        return [
            f"the exact launches are not in the wiring order: launch {first + 1} of {len(planned)} is {found[first]} ({_short(records[first].name)}), "
            f"the wiring has {site.name} there{other}"
        ]
    sentences = []
    want, have = Counter(planned), Counter(found)
    calls = Counter(kernel.id for _, kernel in {(index, kernel.id): (index, kernel) for index, kernel in anchors}.values())
    per = {kernel.id: kernel.launches for _, kernel in anchors}

    def count(kernel_id: str) -> str:
        plan = f"{want[kernel_id]} = {calls[kernel_id]} call(s) x {per[kernel_id]} per call" if kernel_id in per else "none"
        return f"the step has {have[kernel_id]} launch(es) of {kernel_id}, the plan {plan}"

    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, planned, found, autojunk=False).get_opcodes():
        if tag in ("delete", "replace"):
            for n in range(i1, i2):
                sentences.append(f"{sites[anchors[n][0]].name} did not launch {planned[n]}: {count(planned[n])}")
        if tag in ("insert", "replace"):
            for n in range(j1, j2):
                after = f"after {sites[anchors[i1 - 1][0]].name}" if i1 else "at the start of the step"
                sentences.append(f"launched {found[n]} ({_short(records[n].name)}) {after}, where the plan has no such launch: {count(found[n])}")
    return sentences


# ---------------------------------------------------------------------------------------------------------- a traced pass
@dataclass
class Traced:
    """What a session keeps of one traced pass for the kernel check: the sites of a step, the call log of every traced step and
    the launch records, cut into steps -- or the reason there are none."""

    sites: List[Site]
    call_log: List[List[Call]]
    steps: Optional[List[List[Launch]]] = None
    missing: str = ""  # why nothing was observed
    undeclared: str = ""  # a host read inside a step: the records cannot be cut into the traced steps
    rows: int = 1
    status: str = ""  # set by judge(): PASS | FAIL | NOT OBSERVED
    unobserved: str = ""  # the reason of NOT OBSERVED
    differences: int = 0  # the lines judge() returned
    result: Optional[Comparison] = None

    @classmethod
    def of(cls, sites, call_log, records: Sequence[Launch], count: int, missing: str, rows: int) -> "Traced":
        """From the launch records of a pass of ``count`` steps (``records`` empty: ``missing`` says why)."""
        if not records:
            return cls(sites, call_log, missing=missing or "the profiler recorded no kernel activity on this GPU", rows=rows)
        steps, _ = split_steps(records)
        if len(steps) != count:  # a block read the device inside a step: an undeclared copy, and no step can be told from the next
            return cls(sites, call_log, undeclared=f"the records show {len(steps)} host reads for {count} traced steps", rows=rows)
        return cls(sites, call_log, steps, rows=rows)

    def judge(self) -> Tuple[str, List[str]]:
        """Run the kernel check on this pass: ``(PASS | FAIL | NOT OBSERVED, the lines under the check's own line)``."""
        failures = check_forms(self.sites, self.call_log)
        lines = [f"form: {sentence}" for sentence in failures]
        if self.undeclared:
            lines.append(f"copies: {self.undeclared}: a block reads the device inside a step, and only the one host read of a step is declared")
        if self.steps is not None:
            self.result = compare(self.steps, self.sites, self.call_log)
            lines += self.result.sentences()
            self.unobserved = self.result.unobserved
        else:
            self.unobserved = self.missing
        self.differences = len(lines)
        self.status = FAIL if lines else (NOT_OBSERVED if self.unobserved else PASS)
        return self.status, lines

    def summary(self) -> str:
        """The check's own line, without its verdict."""
        if self.steps is None:
            return f"not observed: {self.unobserved}" if not self.undeclared else "the launch records cannot be cut into the traced steps"
        r = self.result
        if self.status == FAIL:  # what was attributed is not stated for a pass that does not hold
            kinds = f"{self.differences} kind{'s' if self.differences != 1 else ''} of difference"
            return f"{r.steps} traced steps: {_span(r.launches)} launches per step; {kinds} between the launches and the plan"
        families = ", ".join(f"{family} {_span(counts)}" for family, counts in sorted(r.families.items()) if max(counts, default=0))
        by_family = sum(statistics.median(counts) for counts in r.families.values()) if r.families else 0
        text = (
            f"{r.steps} traced steps: {_span(r.launches)} launches per step; {r.exact} on exact rows, in the wiring order; "
            f"{int(by_family):,} attributed by family to fallback rows and the host path ({families or 'none'}); {_span(r.copies)} copies per step"
        )
        if r.unobserved:
            text += f"; launch shapes not observed: {r.unobserved}"
        return text

    def details(self) -> List[str]:
        """Per exact row what a step launched, and the kernels a library chose inside the fallback rows (printed, not gated)."""
        lines = []
        calls = Counter(site.block for site in self.sites)
        for site in {site.block: site for site in self.sites}.values():
            for kernel in site.kernels:
                if kernel.relation != "exact":
                    continue
                made, total = calls[site.block], calls[site.block] * kernel.launches
                count = f"{total} launch{'es' if total != 1 else ''} per step ({made} call{'s' if made != 1 else ''} x {kernel.launches} per call)"
                if site.facts is not None:
                    symbols = ", ".join(site.facts.symbols) or "none compiled"
                    lines.append(
                        f"{site.block}: {site.entry}{form_text(site.form)}, {count}; the entry states {shape_text(site.facts.shape)} and the kernel {symbols}"
                    )
                elif kernel.shape == "entry":
                    lines.append(f"{site.block}: {site.entry}{form_text(site.form)}, {count}; the entry states no launch for this run")
                else:
                    lines.append(f"{site.block}: {site.entry}, {count}; {kernel.id} by name (no launch shape is declared for it)")
        if self.result is not None:
            for family, picks in sorted(self.result.picks.items()):
                if family == "pytorch":
                    continue
                chosen = ", ".join(f"{name if len(name) <= 72 else name[:69] + '...'} x {count}" for name, count in picks.most_common(6))
                more = f", and {len(picks) - 6} more" if len(picks) > 6 else ""
                lines.append(f"chosen by a library inside the fallback rows (printed, not gated): {family}: {chosen}{more}")
        return lines

    def dump(self) -> dict:
        """The census of this pass as plain data: the sites of a step -- each with the row's form (``form``) and, from the call
        log, the entry instance that served it and the form that instance states (``instance``, ``served``) -- and the launch
        records of every traced step, names stored once."""
        names: Dict[str, int] = {}
        steps = []
        for step in self.steps or []:
            steps.append(
                [[names.setdefault(r.name, len(names)), canonical(r), r.kind, r.grid, r.block, r.shared_memory, r.registers, *r.graph, r.bytes] for r in step]
            )
        logged = {(call.block, call.layer): call for calls in self.call_log for call in calls}
        unlogged = Call("", -1, self.rows, "", ())  # of a pass that made no traced step
        return {
            "rows": self.rows,
            "check": self.status,
            "not_observed": self.unobserved,
            "sites": [
                {
                    "block": site.block,
                    "layer": site.layer,
                    "entry": site.entry,
                    "instance": logged.get((site.block, site.layer), unlogged).instance,
                    "form": {name: value for name, value in site.form},
                    "served": {name: value for name, value in logged.get((site.block, site.layer), unlogged).form},
                    "kernels": [{"id": k.id, "origin": k.origin, "relation": k.relation, "launches": k.launches, "shape": k.shape} for k in site.kernels],
                    "stated": (
                        None
                        if site.facts is None
                        else {"symbols": list(site.facts.symbols), "launch": [list(part) if isinstance(part, tuple) else part for part in site.facts.shape]}
                    ),
                }
                for site in self.sites
            ],
            "names": list(names),
            "fields": ["name", "kernel", "kind", "grid", "block", "shared_memory", "registers", "graph_id", "graph_node_id", "bytes"],
            "steps": steps,
        }


# ---------------------------------------------------------------------------------------------------------- the two tables
def _total(calls: str) -> int:
    return sum(int(part) for part in calls.split("/") if part.strip().isdigit())


def table(plan) -> str:
    """The kernel table, first part: planned.  Per block the census entries of its row -- which kernels a call launches and
    how the kernel check will judge them -- decided from the plan alone, before anything is loaded."""
    head = ("block", "calls", "provider", "entry [form]", "kernel", "launches", "checked")
    lines = [head]
    exact = rows = chains = 0
    for name, block in plan.items():
        row, calls = plan.rows[name], plan.calls.get(name, "")
        kernels = tuple(block.kernels)
        launches, checked = "--", "nothing declared"
        if any(kernel.relation == "exact" for kernel in kernels):
            per_call = sum(kernel.launches for kernel in kernels if kernel.relation == "exact")
            launches = str(_total(calls) * per_call) if calls else f"{per_call} per call"
            exact, rows = exact + _total(calls) * per_call, rows + 1
            checked = "name and launch shape, as the entry states them" if any(kernel.shape == "entry" for kernel in kernels) else "name"
        elif any(kernel.relation == "host" for kernel in kernels):
            checked = "host path, not compared"
        elif kernels:
            chains += _total(calls)
            checked = "by family" + (f"; fallback: {row.fallback}" if row.fallback else "")
        lines.append((name, calls, block.provider, block.entry + form_text(row.form), ", ".join(kernel.id for kernel in kernels) or "--", launches, checked))
    widths = [max(len(line[i]) for line in lines) for i in range(len(head))]
    sms = f", {plan.facts.sm_count} SMs" if plan.facts.sm_count else ""
    run = f"stack {plan.stack}, {plan.sequences} sequence{'s' if plan.sequences > 1 else ''}, {plan.facts.class_text}{sms}"
    text = [f"kernels, planned (decided before the load from {plan.source}; {run})"]
    text += ["  " + "  ".join(cell.ljust(width) for cell, width in zip(line, widths)).rstrip() for line in lines]
    text.append(
        f"  per step: {exact} launch{'es' if exact != 1 else ''} on {rows} exact row{'s' if rows != 1 else ''}, checked one by one in the wiring order; "
        f"the launches of {chains} fallback call{'s' if chains != 1 else ''} and of the host path are attributed by family"
    )
    return "\n".join(text)


def stated(plan, rows: Optional[int] = None) -> str:
    """The kernel table, second part: what the entries of the plan state about their launch once they are built -- the form an
    entry resolved for this run's row count, its launch shape and the symbol of the kernel it compiled.  Never a row: an
    entry's own pick is its answer, not a choice of the table."""
    rows = plan.sequences * plan.tokens if rows is None else rows
    text = [f"kernels, as the entries state them (after the load; {rows} row{'s' if rows != 1 else ''}; never a row of the table)"]
    for name, block in plan.items():
        if not any(kernel.shape == "entry" for kernel in block.kernels):
            continue
        try:
            facts = block.launch_facts(rows)
        except (ValueError, NotImplementedError) as error:
            text.append(f"  {name}  {block.entry}: states no launch for this run ({error})")
            continue
        symbols = ", ".join(facts.symbols) or "no kernel compiled yet"
        text.append(f"  {name}  {block.entry}{form_text(plan.rows[name].form)}: {shape_text(facts.shape)}")
        text.append(f"    {symbols}")
    if len(text) == 1:
        text.append("  no block of this plan is served by an entry that states its launch")
    return "\n".join(text)
