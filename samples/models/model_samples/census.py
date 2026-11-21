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
   A traced pass keeps one call log per traced step, and a step without its log fails the check (:meth:`Traced.judge`):
   nothing would say what served its calls.  The objects are where a form is compared: see below.
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

**The rows of a per-kernel table.**  The walk of items 1 to 4 also says, for every launch record of a traced step, which
row of a per-kernel table it belongs to (:class:`Row`, ``Comparison.attributed``): the block of an exact kernel; the
blocks a gap of the wiring has on fallback or host entries, which are one row together; or the host read that ends the
step.  A row is found by its place in the wiring, never by a kernel's name.  A record the check does not attribute has
no row, and a pass with such a record has no table.

**An entry that a step calls in several forms.**  A provider row may name the call sites of its block (``ProviderRow.sites``:
per site the further fields of the entry's form there), and the family names the site of every call of its wiring
(``model.sites()``).  Every item above then holds per call site: a site carries the form of ITS call, the facts of the entry
instance of that form, and the call log says which instance served it.  A call served by the instance of a neighbouring
form fails item 0 with its own sentence, although the kernel's name and launch shape may be the same.

**A block whose calls several rows serve.**  A provider row may serve some call sites of its block and hand the others on
to the next row (``ProviderRow.sites``: a site without fields); the plan then holds one part per call site
(``Plan.at``, ``Plan.parts``).  Everything here that reads a row or a block of a call reads the part of that call's
site: its census entries, its form, its entry's facts and its identity.  So a block can be an exact kernel at one call
site and a fallback family at another, and both tables print one line per part, ``<block> at <call site>``.

**A kernel name proves nothing about a form.**  A generated kernel symbol is the kernel's name followed by the text of its
compile-time arguments, cut at a fixed length: two forms of one kernel can share a symbol, and they can share a launch shape
too.  So the form is compared on the objects (item 0), never read out of a name; the name and the launch shape (item 5) show
that the launch is the entry's and has the shape the entry states.

Where the profiler cannot run, or its records carry no launch arguments, nothing was observed: the check then says so and a
sample reports ``RAN ... (kernels not observed: <reason>)`` instead of ``PASS``.  A form that does not hold fails either way.

**A stack that declares a target kernel set** is also held against a recorded census of that set (:class:`Target`, read from
the family's ``kernel_census.json``): per block call of a step the kernel the target launches there, its form and -- per GPU
class, SM count and row count -- its recorded launch shape.  How a block call stands to the target is data of that file,
never a property of an adapter: :meth:`Target.claim` gives every call of the plan an :class:`Identity` (``target``,
``stand-in``, ``not yet (<fallback reason>)``, ``not the target``), counts them (always as three numbers: the target's own
compiled kernels, cuDNN Frontend entries with byte-equal results, declared stand-ins) and lists every reason the class is
not claimed.  A section holds for the GPU class, the SM count, the row count and the layer widths it was recorded with: a
run that differs in one of them is told that no identity is stated, never held to another run's kernels.  The check then
has a seventh item:

7. Every launch of a block call whose identity is ``target`` has the launch shape the census recorded for it on this GPU
   class, SM count and row count.

A kernel of a library has no entry that states its launch: what is held of it is the census's.  Where the library serves
one call with several classes of kernel, the census entry names the class (``class``, a part of the kernel's name) and item
7 holds it with the launch shape.  Where the library's own tuner chooses the kernel, the census records that tuner's pick
(``pick``) and no launch shape: the launch is held as a member of the kernels the call can launch, and the run prints the
kernel it launched beside the recorded pick -- compared, never gated.  Where a kernel's launch follows a fact of the run
beyond its row count -- the maximum length a paged attention kernel is told -- the entry names the fact it was recorded
with (``run``), and item 7 holds its launch shape and its class for a run that states the same value alone: at another
one the launch is held as a member of its kernel's family, and the census line says so.

A census holds one configuration per kind of step: ``plain`` for the decode step (one token per sequence) and ``verify``
for the step that feeds several (:func:`configuration_of`); a section of the latter also names the tokens per sequence it
was recorded with, and holds for no other count.

and one more line: ``target census <configuration> / <class> (recorded <date>): EQUAL (<what was compared>)``, or
``NOT CLAIMED (<reasons>)``.  Either way the line says how many launches of a step were compared by launch shape -- as three
numbers -- and how many by membership alone: ``EQUAL`` never stands for a comparison that was not made.

A session with a drafter traces whole steps of speculative decoding: the sites of such a step are, for every draft
pass, the drafter's wiring with the pass's number in the place of the layer (:func:`passes`), then the model's.  No
section of a census holds a draft pass, so nothing is stated about the kernels of the passes: they are attributed like
every launch, and the claim -- the verify plan's -- speaks about the model's sites, which begin at ``Traced.first``.

This module imports nothing beyond the standard library.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field, replace
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
    #: the duration in microseconds as the trace holds it; ``end - start`` is the same time on a clock of milliseconds
    #: since the profiler's origin, where a double no longer holds the trace's last decimals.  ``None``: not read from a trace
    duration: Optional[float] = None

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
    site: str = ""  # the call site, for a block whose entry is called in several forms (``form`` and ``facts`` are this site's)

    @property
    def name(self) -> str:
        return self.block if self.layer < 0 else f"{self.block} (layer {self.layer})"

    @property
    def row(self) -> str:
        """The block and, where its row names several, the call site: what a sentence about the row's form names."""
        return f"{self.block} at {self.site}" if self.site else self.block


# ---------------------------------------------------------------------------------------------------------- canonical ids
#: kernel name -> canonical id, first match.  Symbols of cuDNN Frontend entries come first; then the kernels of other libraries
#: and of PyTorch that the blocks of this area launch.  A chain that gains an operator whose kernel no pattern names adds its
#: pattern in the same change: an unnamed kernel fails the kernel check by name.
FAMILIES: Tuple[Tuple[str, "re.Pattern"], ...] = (
    ("gdn_decode_fused", re.compile(r"^cudnn_kernel_frost_gdn_decode_fused_")),
    ("gdn_verify_fused", re.compile(r"^cudnn_kernel_frost_gdn_verify_fused_")),
    ("gate_quant", re.compile(r"^cudnn_kernel_frost_attention_gate_quant_")),
    ("add_norm_quant", re.compile(r"^cudnn_kernel_frost_add_rms_norm_quant_")),
    ("fp8_projection", re.compile(r"^cudnn_kernel_frost_fp8_projection_decode_")),
    ("swiglu_fp4", re.compile(r"^cudnn_kernel_frost_swiglu_fp4_decode_")),
    ("attention_prepare", re.compile(r"^cudnn_kernel_frost_attention_prepare_")),
    ("cudnn_fe", re.compile(r"^cudnn_kernel_frost_")),  # a kernel of cuDNN Frontend that no entry above names: never a family member
    # FlashInfer's multiplies on the stored formats.  The 4-bit multiply launches one of two kernel classes of one module, and
    # the class is in the name (a census entry names it: ``class``).  The 8-bit multiply is served by the kernel the library
    # chooses among its own: a cuBLASLt kernel of 8-bit operands, or the library's own CUTLASS multiply, whose name the
    # profiler leaves mangled -- of that one the name must carry what the call launches: the SM100 mainloop on tensor-core
    # multiplies and E4M3 operands (another CUTLASS multiply is no member).  Both stand in front of the family they would
    # otherwise fall to
    ("fi_mm_fp4", re.compile(r"^kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100")),
    (
        "cublaslt_fp8_gemm",
        re.compile(r"^nvjet_sm\d+_qq|^_ZN7cutlass13device_kernelINS_4gemm6kernel13GemmUniversalI\w*MainloopSm100TmaUmmaWarpSpecializedI\w*NS_12float_e4m3_tE"),
    ),
    # ``torch.index_select`` launches one of three kernels, by the number of rows it gathers and the size of a table row: the
    # small-index kernel up to 16 rows, and from 17 the large-index kernel or PyTorch's vectorized gather (seen on a bf16
    # table of 5120 columns with int32 ids)
    ("embedding_gather", re.compile(r"\bat::native::(?:\(anonymous namespace\)::)?(?:indexSelect(?:Small|Large)Index|vectorized_gather_kernel)<")),
    ("cublas", re.compile(r"cublas|^nvjet_|\bcutlass::Kernel2?<cutlass_|^(?:maxwell|volta|ampere)_\w*gem[mv]", re.IGNORECASE)),
    ("pytorch", re.compile(r"\bat::native::|\(anonymous namespace\)::(?:softmax_warp_forward|elementwise_kernel_with_index)<")),
)
#: a kernel with an id of its own that is also a member of a family.  Where a plan has an exact row for it, every launch of
#: it in a step is held against that row (a chain that launched the same kernel would count as an extra launch: no chain of
#: this area does -- the embedding is the one ``index_select`` of a step, and a chain that gathers rows indexes its tensor,
#: which is another kernel of PyTorch); where it has none, the kernel counts as its family's
FAMILY_OF = {"embedding_gather": "pytorch", "cublaslt_fp8_gemm": "cublas"}
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
                float(event.get("dur", 0.0)),
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
def call_sites(model) -> Tuple[str, ...]:
    """The call site of every call of ``model.wiring()``, as the family names it (``model.sites()``): ``""`` for a call of a
    block that is called in one form everywhere, and for every call of a model that names no sites."""
    wiring = tuple(model.wiring())
    named = getattr(model, "sites", None)
    sites = tuple(named()) if callable(named) else ("",) * len(wiring)
    if len(sites) != len(wiring):
        raise ValueError(f"the model names {len(sites)} call sites for the {len(wiring)} calls of its wiring")
    return sites


def _part(plan, block: str, site: str = "") -> tuple:
    """``(row, block)`` of what serves the call of ``block`` at ``site`` in ``plan`` (``Plan.at``): the block's own row and
    block, or -- where several rows serve its calls by call site -- those of that site."""
    at = getattr(plan, "at", None)
    return at(block, site) if callable(at) else (plan.rows[block], plan[block])


def _parts(plan) -> list:
    """``(label, block name, call site, row, block)`` of everything that serves ``plan`` (``Plan.parts``): one per block,
    and one per call site of a block whose calls several rows serve."""
    parts = getattr(plan, "parts", None)
    return parts() if callable(parts) else [(name, name, "", plan.rows[name], block) for name, block in plan.items()]


def expected(plan, model, rows: Optional[int] = None) -> List[Site]:
    """The block calls of one step in the model's wiring order (``model.wiring()``), each with its block's census entries, the
    provider row's form at that call site (``ProviderRow.site_form``) and -- for an entry that states its launch -- the entry's
    facts there at ``rows`` (default: the plan's sequences x tokens).  The row and the block are those of the call's site
    (``Plan.at``): where several rows serve the calls of a block, each call has the census entries of its own row.  An entry that declines to state them for a row count
    its row claims, or a call site the row has no form for, is recorded on the site (``wrong``): the check then fails,
    because the table is wrong."""
    rows = plan.sequences * plan.tokens if rows is None else rows
    stated: Dict[Tuple[str, str], Tuple[tuple, Optional[Facts], str]] = {}
    sites = []
    for (name, layer), at in zip(model.wiring(), call_sites(model)):
        unserved = ""
        try:
            row, block = _part(plan, name, at)  # of this call's site, for a block whose calls several rows serve
        except ValueError as error:  # a call site no row of the block names: the block as a whole, and the table is wrong
            row, block, unserved = plan.rows[name], plan[name], str(error)
        if (name, at) not in stated:
            form, facts, wrong = (), None, unserved
            try:
                form = () if wrong else tuple(row.site_form(at))
            except ValueError as error:
                wrong = str(error)
            if not wrong and any(kernel.shape == "entry" for kernel in block.kernels):
                try:
                    facts = block.launch_facts(rows, at) if at else block.launch_facts(rows)
                except (ValueError, NotImplementedError) as error:
                    wrong = str(error)
                if facts is None and not wrong:
                    wrong = "the adapter states no launch facts"
            stated[(name, at)] = (form, facts, wrong)
        sites.append(Site(name, layer, tuple(block.kernels), block.entry, *stated[(name, at)], at))
    return sites


def passes(plan, drafter, count: int) -> List[Site]:
    """The block calls of the ``count`` draft passes of an engine step, in the order a step makes them: the wiring of one
    pass (:func:`expected` on the drafter's plan) once per pass, with the pass's number, from 0, in the place of the
    layer.  The sites of an engine step are these and then the model's."""
    one = expected(plan, drafter)
    return [replace(site, layer=index) for index in range(count) for site in one]


def form_text(form: Sequence[Tuple[str, object]]) -> str:
    """A form as the table prints it: `` [PDL False, PDL_RELEASE False]``, or nothing for an empty form."""
    return " [" + ", ".join(f"{name} {value}" for name, value in form) + "]" if form else ""


def shape_text(shape) -> str:
    grid, block, shared = shape
    return f"grid {tuple(grid)}, block {tuple(block)}, shared memory {shared:,} B"


def _named(name: str, width: int = 72) -> str:
    """A kernel name as a line names it where the name itself is the statement: whole up to ``width`` characters, else its
    start with its length and a digest of the whole name -- two long names that differ late are told apart by the digest
    (a mangled CUTLASS name runs to more than a thousand characters and carries its operand types far from its start)."""
    if len(name) <= width:
        return name
    return f"{name[: width - 3]}... ({len(name):,} characters, sha256 {hashlib.sha256(name.encode()).hexdigest()[:12]})"


def _short(name: str, width: int = 96) -> str:
    return repr(name if len(name) <= width else name[: width - 3] + "...")


def check_forms(sites: Sequence[Site], call_log: Sequence[Sequence[Call]] = ()) -> List[str]:
    """Item 0, on the objects: for every block whose entry states its launch, the provider row names a form and each field
    it names equals the entry's own form; and every logged call of a block was served by an entry instance that states the
    row's form (``Call.form``: what the adapter read from the object it called).  Returns the sentences of what does not
    hold."""
    failures = []
    for site in {(site.block, site.site): site for site in sites}.values():
        if site.wrong:
            failures.append(f"{site.row}: {site.entry} states no launch for this run ({site.wrong}): the provider table is wrong")
        if not site.form and any(kernel.shape == "entry" for kernel in site.kernels):
            failures.append(f"{site.row}: the row names no form; {site.entry} would be called on its own pick")
        if site.facts is None:
            continue
        for name, value in site.form:
            if name not in site.facts.form:
                failures.append(f"{site.row}: the row's form names {name}, which the form of {site.entry} does not have")
            elif site.facts.form[name] != value:
                failures.append(f"{site.row}: the row's form says {name} {value!r}, {site.entry} states {name} {site.facts.form[name]!r} for this run")
    by_site = {}  # (block, layer) -> its first call of a step: where a step's calls are not the wiring's, a call is found by name
    for site in sites:
        by_site.setdefault((site.block, site.layer), site)
    served = {}  # (block, call site, the form the serving instance states) -> the first call: one sentence per kind of difference
    for calls in call_log:
        for index, call in enumerate(calls):
            # a block may be called twice in a layer, each call at a site of its own: a call is held against the site at its
            # own place of the wiring
            at = sites[index] if index < len(sites) and (sites[index].block, sites[index].layer) == (call.block, call.layer) else None
            site = at or by_site.get((call.block, call.layer))
            if site is not None and tuple(call.form) != tuple(site.form):
                served.setdefault((call.block, site.site, tuple(call.form)), (site, call))
    for site, call in served.values():
        where = f"{site.name}, the call at {site.site}" if site.site else site.name
        failures.append(
            f"{where}: the adapter served the call with the form{form_text(call.form) or ' []'} ({call.instance or 'no entry instance'}); "
            f"the row says{form_text(site.form) or ' []'}"
        )
    return failures


class Row(NamedTuple):
    """One row of a per-kernel table: block calls of a step, found by their place in the wiring, whose launch records are
    one row.  An exact kernel is a row of its block.  The records between two exact kernels belong to the blocks the
    wiring has there on fallback or host entries: those blocks are one row together, because nothing in the records
    tells one block's launches from the next one's.  The host read that ends a step is a row of its own."""

    blocks: Tuple[str, ...]  # the block of an exact row; the blocks of a gap in wiring order, each once; none for the host read
    relation: str  # "exact" | "fallback" (a host block may share the gap) | "host" | "read"
    sites: Tuple[int, ...]  # the indices, among the sites of a step, of the block calls the row holds


def _table(sites: Sequence[Site], anchors: Sequence[Tuple[int, Kernel]]) -> Tuple[List[Row], List[int], List[Optional[int]]]:
    """``(rows, the row of every exact launch, the row of every gap)`` for the sites of a step and its exact launches
    (``anchors``: site index and census entry, in wiring order).  A gap that declares no family has no row (``None``):
    nothing may be launched there.  The rows stand in the wiring order of their first block call -- the exact rows,
    then the rows of the gaps -- and the host read's row is the last."""
    keys: Dict[tuple, int] = {}
    found: List[list] = []

    def row(blocks: tuple, relation: str, members: Sequence[int]) -> int:
        at = keys.setdefault((blocks, relation), len(found))
        if at == len(found):
            found.append([blocks, relation, []])
        found[at][2] += [index for index in members if index not in found[at][2]]
        return at

    of_anchor = [row((sites[index].block,), "exact", [index]) for index, _ in anchors]
    of_gap: List[Optional[int]] = []
    for gap in range(len(anchors) + 1):
        low = anchors[gap - 1][0] if gap else -1
        high = anchors[gap][0] if gap < len(anchors) else len(sites)
        between = range(max(low, 0), min(high + 1, len(sites))) if low != high else ()
        members = [index for index in between if any(kernel.relation != "exact" for kernel in sites[index].kernels)]
        fallback = any(kernel.relation == "fallback" for index in members for kernel in sites[index].kernels)
        blocks = tuple(dict.fromkeys(sites[index].block for index in members))
        of_gap.append(row(blocks, "fallback" if fallback else "host", members) if members else None)
    row((), "read", [])
    return [Row(blocks, relation, tuple(members)) for blocks, relation, members in found], of_anchor, of_gap


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
    served: Dict[str, Counter] = field(default_factory=dict)  # exact row whose entry states no launch -> kernel name -> launches over the pass
    table: List[Row] = field(default_factory=list)  # the rows of a per-kernel table of the pass
    #: per step, per record of the step in its order: the index of the record's row in ``table``; -1 for a record the
    #: check did not attribute (it failed there)
    attributed: List[List[int]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def sentences(self, limit: int = 8) -> List[str]:
        """The failures, one line per distinct sentence with the steps it holds for, at most ``limit`` of them."""
        steps: Dict[str, List[int]] = {}
        for step, sentence in self.failures:
            at = steps.setdefault(sentence, [])
            if step not in at:  # a block that is called twice in a layer can fail twice in one step with one sentence
                at.append(step)
        lines = []
        for sentence, at in list(steps.items())[:limit]:
            more = f" (and {len(at) - 1} more of the {self.steps} traced steps)" if len(at) > 1 else ""
            lines.append(f"step {at[0]}{more}: {sentence}")
        if len(steps) > limit:
            lines.append(f"... and {len(steps) - limit} more kinds of difference")
        return lines


def _span(counts: Sequence[int]) -> str:
    return "0" if not counts else (f"{min(counts):,}" if min(counts) == max(counts) else f"{min(counts):,} to {max(counts):,}")


def compare(
    steps: Sequence[Sequence[Launch]],
    sites: Sequence[Site],
    call_log: Sequence[Sequence[Call]] = (),
    recorded: Optional[Mapping[int, tuple]] = None,
    census: str = "the target census",
    named: Optional[Mapping[int, str]] = None,
) -> Comparison:
    """Items 1 to 7 of the kernel check: the launch records of every traced step (each list ends with the step's host read,
    as :func:`split_steps` cuts them) against the sites of one step in wiring order (:func:`expected`); and, where a call log
    is given, the block calls each step made against the wiring.  ``recorded`` (item 7) maps the index of a site to the
    launch shape a target census recorded for it (:attr:`Claim.recorded`), ``named`` to the kernel class it recorded for a
    library kernel there (:attr:`Claim.classes`: a part of the kernel's name); ``census`` names that census in a sentence."""
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
    table, of_anchor, of_gap = _table(sites, anchors)
    result = Comparison(steps=len(steps), exact=len(anchors), table=table)
    fail = result.failures.append
    for index, step in enumerate(steps):
        calls = [(call.block, call.layer) for call in call_log[index]] if index < len(call_log) else wiring
        if calls != wiring:
            first = next((n for n, (a, b) in enumerate(zip(calls, wiring)) if a != b), min(len(calls), len(wiring)))
            had = Site(*calls[first], ()).name if first < len(calls) else "nothing"
            want = sites[first].name if first < len(sites) else "nothing"
            fail((index, f"the step made {len(calls)} block calls, the wiring has {len(wiring)}; call {first + 1} is {had}, the wiring has {want} there"))
        body = step[:-1] if step and step[-1].name.startswith(HOST_READ) else list(step)
        rows = [-1] * len(body) + [len(table) - 1] * (len(step) - len(body))  # the row of every record; the host read's is the last
        result.attributed.append(rows)
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
                    else:
                        rows[position] = of_gap[gap]
                    continue
                family = kernel_id if kernel_id in declared else FAMILY_OF.get(kernel_id, "")
                if family and family in declared:
                    rows[position] = of_gap[gap]
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
        for position, (site_index, kernel), row in zip(at, anchors, of_anchor):
            record, site = body[position], sites[site_index]
            rows[position] = row
            target = None if recorded is None else recorded.get(site_index)
            if target is not None:  # item 7: the launch shape the census recorded for the target's kernel at this place
                if None in record.shape:
                    result.unobserved = "the profiler's kernel records carry no launch arguments (grid, block, shared memory)"
                elif record.shape != tuple(target):
                    fail((index, f"{site.name} launched {kernel.id} with {shape_text(record.shape)}; {census} has {shape_text(target)}"))
            kind = None if named is None else named.get(site_index)
            if kind and kind not in record.name:  # item 7, a library kernel: the class the census recorded, as the kernel's name carries it
                fail((index, f"{site.name} launched {_short(record.name, 160)}; {census} has the kernel class {kind} of {kernel.id} there"))
            if kernel.shape != "entry" or site.facts is None:
                result.served.setdefault(site.row, Counter())[record.name] += 1  # no entry states its symbol: the name is what the row shows
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


# ---------------------------------------------------------------------------------------------------------- the target
SCHEMA = 2
#: how a block call of a plan stands to the target kernel set
TARGET, STAND_IN, NOT_YET, OTHER, HOST = "target", "stand-in", "not yet", "not the target", "host"
#: the kinds of a target kernel, in the words a run counts them by (one, several)
KINDS = {
    "code": ("the target's own compiled kernel", "the target's own compiled kernels"),
    "port": ("cuDNN Frontend entry with byte-equal results", "cuDNN Frontend entries with byte-equal results"),
}


def _counted(count: int, kind: str) -> str:
    return f"{count} {KINDS[kind][count != 1]}"


BASES = ("recorded", "planned")
#: the parts of a step in a census section besides its layers: before them, after them, after the step
PARTS = ("prologue", "epilogue", "after_step")
#: what an entry of a section may state beside its block and its kernel: ``launch`` = [grid, block, shared memory in bytes]
#: of one launch at the section's row count; ``at`` = the fields of the model's geometry, beyond the section's layer widths,
#: that this one launch shape was recorded with (the output head's: the vocabulary); ``form`` = the fields of the kernel's
#: form the target passes at THIS call, beyond the form its kernel has everywhere (a kernel that is called in several forms);
#: ``class`` = the kernel class of a library kernel at this call, as the kernel's name carries it (a library that serves one
#: call with several classes of kernel: held like the launch shape, and with it); ``pick`` = the kernel a library's tuner
#: chose at this call in the recorded run, by name (printed and compared, never gated: another run's tuner may choose another);
#: ``run`` = the facts of the run, beyond the section's row count, that this one launch shape and class were recorded with
#: (:data:`RUN_FACTS`): held for a run that states the same values, as a block's options carry them, and for no other
ENTRY_FACTS = ("launch", "at", "form", "class", "pick", "run")
#: the facts of a run an entry may name under ``run``: the positions a sequence of the run may reach at most
RUN_FACTS = ("max_model_len",)
#: an optional part of a section's row count: per block the facts that differ at that block's first call in the model's
#: first layer (the first norm of a step has no residual to add)
FIRST_LAYER = "first_layer"
#: the packages the providers of this area import, as a run names them: of the versions a census records, these are
#: compared with what is installed.  A change that brings a provider of another package adds the package here -- unless
#: the provider's adapter holds that package to a validated version itself and does not run on another (FlashInfer:
#: ``providers.flashinfer_calls.VALIDATED``); what such a package depends on is not compared
PACKAGES = {"torch": "PyTorch", "nvidia-cutlass-dsl": "CuTe DSL"}


#: the configurations of a census: the decode step, and the step of several tokens per sequence
PLAIN, VERIFY = "plain", "verify"


def configuration_of(plan) -> str:
    """The configuration of a census that a plan is held against: ``plain`` for a plan of one token per sequence (the decode
    step), ``verify`` for a plan of several (the verify step of speculative decoding)."""
    return PLAIN if getattr(plan, "tokens", 1) == 1 else VERIFY


class Identity(NamedTuple):
    """How one block call of a plan stands to the target kernel set: it launches the target's kernel in the target's form
    (``target``, of kind ``code`` or ``port``), a declared stand-in for it, a fallback row (``not yet``), another kernel or
    another form than the target's (``not the target``), or it is the sample's host path."""

    relation: str  # TARGET | STAND_IN | NOT_YET | OTHER | HOST
    kernel: str = ""  # the target's kernel at this place
    kind: str = ""  # of that kernel: "code" | "port"; "" for a stand-in and for the host path
    reason: str = ""  # of NOT_YET: the row's fallback reason; of OTHER: what differs; of TARGET / STAND_IN: why ``launch`` is not held
    launch: Optional[tuple] = None  # the launch shape the census recorded for the target's kernel at this place
    named: str = ""  # the kernel class the census recorded there for a library kernel (a part of its name); held with ``launch``
    pick: str = ""  # the kernel a library's tuner chose there in the recorded run: printed and compared, never gated
    run: str = ""  # of TARGET / STAND_IN: the facts of this run that are not the ones ``launch`` was recorded with; neither it nor ``named`` is held then

    def text(self) -> str:
        if self.relation in (NOT_YET, OTHER):
            return f"{self.relation} ({self.reason})"
        return "host path, not compared" if self.relation == HOST else self.relation


def _class_text(key: str) -> str:
    """``"cc10.0/sm148"`` as a run prints it: ``cc 10.0 / 148 SMs``."""
    cc, sms = key.split("/")
    return f"cc {cc[2:]} / {sms[2:]} SMs"


def _plain(value):
    """A value of a form or of a geometry as it is compared and printed: a JSON file holds a list where a provider table and a
    geometry hold a tuple (the sections of a rotary width), and the two are one value."""
    return tuple(_plain(item) for item in value) if isinstance(value, (list, tuple)) else value


def _width(value) -> bool:
    """Whether ``value`` is a layer width as a census section names one: a whole number, or a list of whole numbers (a width
    that has parts: the sections of a rotary width)."""

    def whole(n):
        return isinstance(n, int) and not isinstance(n, bool)

    return whole(value) or (isinstance(value, (list, tuple)) and bool(value) and all(whole(n) for n in value))


def _other_widths(recorded: Mapping[str, object], geometry) -> str:
    """The fields of a model's geometry that are not the recorded ones, as a run names them: ``hidden 4096 (recorded 5120)``;
    empty where every recorded field is the model's."""
    return ", ".join(
        f"{name} {_plain(getattr(geometry, name, 'none'))} (recorded {_plain(value)})"
        for name, value in recorded.items()
        if _plain(getattr(geometry, name, None)) != _plain(value)
    )


def _other_run(recorded: Mapping[str, int], options: Mapping) -> str:
    """The facts of a run that are not the recorded ones, as a run names them: ``max_model_len 4096 (recorded 2051)``; empty
    where the run states every recorded fact.  ``options``: what the plan handed the block of the call; a fact it does
    not carry is one the run does not share."""
    stated = {name: options.get(name) for name in recorded}
    return ", ".join(
        f"{name} {'not stated' if stated[name] is None else stated[name]} (recorded {value})"
        for name, value in recorded.items()
        if type(stated[name]) is not int or stated[name] != value
    )


def _run_facts(value) -> bool:
    """Whether ``value`` is the ``run`` of a census entry: an object of facts of a run this reader knows, a whole number each."""
    return isinstance(value, dict) and all(name in RUN_FACTS and isinstance(n, int) and not isinstance(n, bool) and n >= 1 for name, n in value.items())


def _launch_shape(value) -> bool:
    """Whether ``value`` is a launch shape as a census file writes it: ``[grid of three, block of three, shared memory in bytes]``."""

    def whole(n, least):
        return isinstance(n, int) and not isinstance(n, bool) and n >= least

    def triple(t):
        return isinstance(t, list) and len(t) == 3 and all(whole(n, 1) for n in t)

    return isinstance(value, list) and len(value) == 3 and triple(value[0]) and triple(value[1]) and whole(value[2], 0)


def installed_versions(packages: Sequence[str]) -> Dict[str, Optional[str]]:
    """The installed version of every package of ``packages`` (``None``: not installed), read from the distributions'
    metadata: no package is imported for it."""
    import importlib.metadata as metadata

    found: Dict[str, Optional[str]] = {}
    for package in packages:
        try:
            found[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            found[package] = None
    return found


@dataclass
class Claim:
    """What one run can say about the target kernel set: the identity of every block call of its plan, the three numbers,
    and every reason its GPU class is not claimed."""

    census: str  # the census file, by its own name
    configuration: str
    device: str  # "cc 10.0 / 148 SMs"
    rows: int
    recorded_on: str = ""  # the date of the section's record
    basis: str = ""  # of the section: "recorded" | "planned"; "" where the census has no section for the device
    claimed: bool = False  # what the section says
    blocks: List[str] = field(default_factory=list)  # the block of every call of the wiring
    identities: List[Identity] = field(default_factory=list)  # one per call; empty where no section holds this run's row count
    absent: List[Tuple[str, str, str]] = field(default_factory=list)  # (block, kernel, kind) of the target that the plan has no block for
    reasons: List[str] = field(default_factory=list)  # why the class is not claimed; empty: it is
    sites: List[str] = field(default_factory=list)  # the call site of every call of the wiring ("" for a block that names none)

    @property
    def known(self) -> bool:
        """Whether the census holds this run (its device and its row count): only then is an identity stated."""
        return bool(self.identities)

    @property
    def name(self) -> str:
        if not self.basis:
            return f"target census {self.configuration}"
        how = f"recorded {self.recorded_on}" if self.basis == "recorded" else self.basis
        return f"target census {self.configuration} / {self.device} ({how})"

    @property
    def recorded(self) -> Dict[int, tuple]:
        """Index of a block call in the wiring -> the launch shape the census holds for it: the calls whose identity is
        ``target`` or ``stand-in`` and whose launch the census records (item 7 of the kernel check)."""
        return {index: identity.launch for index, identity in enumerate(self.identities) if identity.relation in (TARGET, STAND_IN) and identity.launch}

    @property
    def classes(self) -> Dict[int, str]:
        """Index of a block call in the wiring -> the kernel class the census holds for a library kernel there (item 7 of the
        kernel check: the class is a part of the launched kernel's name)."""
        return {index: identity.named for index, identity in enumerate(self.identities) if identity.relation in (TARGET, STAND_IN) and identity.named}

    @property
    def picks(self) -> Dict[int, str]:
        """Index of a block call in the wiring -> the kernel a library's tuner chose there in the recorded run: what a run
        prints its own kernel beside.  Never gated."""
        return {index: identity.pick for index, identity in enumerate(self.identities) if identity.relation in (TARGET, STAND_IN) and identity.pick}

    def of(self, block: str, site: Optional[str] = None) -> str:
        """The identity column of a block: the identity of its calls (of its calls at ``site``, for a part of a block whose
        calls several rows serve); ``--`` for the host path."""
        at = self.sites if len(self.sites) == len(self.blocks) else [""] * len(self.blocks)
        mine = [identity for name, where, identity in zip(self.blocks, at, self.identities) if name == block and site in (None, where)]
        return "; ".join(dict.fromkeys("--" if identity.relation == HOST else identity.text() for identity in mine))

    def numbers(self) -> dict:
        """The three numbers of the target for this model, and of what the plan reaches of them."""
        calls = [(block, identity.relation, identity.kind) for block, identity in zip(self.blocks, self.identities) if identity.relation != HOST]
        calls += [(block, NOT_YET, kind) for block, _, kind in self.absent]
        not_yet: Counter = Counter(block for block, relation, kind in calls if kind and relation == NOT_YET)
        other: Counter = Counter(block for block, relation, kind in calls if kind and relation == OTHER)
        return {
            "launches": len(calls),
            "code": sum(kind == "code" for _, _, kind in calls),
            "port": sum(kind == "port" for _, _, kind in calls),
            "stand_ins": sum(not kind for _, _, kind in calls),
            "target_code": sum(relation == TARGET and kind == "code" for _, relation, kind in calls),
            "target_port": sum(relation == TARGET and kind == "port" for _, relation, kind in calls),
            "not_yet": dict(not_yet),
            "not_the_target": dict(other),
            "stand_ins_not_yet": list(dict.fromkeys(block for block, relation, kind in calls if not kind and relation != STAND_IN)),
        }

    def coverage(self) -> List[str]:
        """The identity lines of the kernel table: what the target is for this model, always as three numbers, and how much of
        it the plan launches."""
        if not self.known:
            why = next((reason for reason in self.reasons if reason.startswith("no recorded section")), "no recorded section")
            return [f"kernel identity: not stated for this run ({why})"]
        n = self.numbers()
        stand = f"{n['stand_ins']} declared stand-in{'s' if n['stand_ins'] != 1 else ''}"
        whole = f"{n['launches']} launches per step = {_counted(n['code'], 'code')} + {_counted(n['port'], 'port')} + {stand}"
        reached = n["target_code"] + n["target_port"]
        if reached == n["code"] + n["port"] and not n["stand_ins_not_yet"]:
            return [f"kernel identity, as planned: {whole}"]
        text = (
            f"kernel identity, as planned: {reached} of {n['launches']} launches of a step are target kernels "
            f"({_counted(n['target_code'], 'code')} + {_counted(n['target_port'], 'port')})"
        )
        for words, blocks in (("not yet", n["not_yet"]), ("not the target", n["not_the_target"])):
            if blocks:
                text += f"; {words}: " + ", ".join(f"{block} {count}" for block, count in blocks.items())
        text += f"; {stand}" + (f" (not yet: {', '.join(n['stand_ins_not_yet'])})" if n["stand_ins_not_yet"] else "")
        return [f"the whole target kernel set of a step of this model (what a complete stack launches; this stack's share is the next line): {whole}", text]

    def compared(self) -> Tuple[int, str, int]:
        """What item 7 holds of a step: ``(launches compared by launch shape, the same as three numbers, launches held by
        membership alone)``.  A target launch is compared by launch shape where the census records one for it; where it
        records none (a kernel a library's tuner selects, a stand-in) the launch is held as a member of its kernel's
        family or entry and in no shape."""
        held = [identity for identity in self.identities if identity.relation in (TARGET, STAND_IN)]
        shaped = [identity for identity in held if identity.launch]
        stand = sum(identity.relation == STAND_IN for identity in shaped)
        split = " + ".join(_counted(sum(identity.relation == TARGET and identity.kind == kind for identity in shaped), kind) for kind in KINDS)
        text = f"{len(shaped)} launch{'es' if len(shaped) != 1 else ''} of a step compared by launch shape ({split} + {stand} declared stand-in{'s' if stand != 1 else ''})"
        return len(shaped), text, len(held) - len(shaped)

    def line(self, status: str = "") -> str:
        """The census line of the kernel check, given the check's own verdict.  It says what was compared: ``EQUAL`` and the
        sentence under ``NOT CLAIMED`` both carry the number of launches of a step that were held to a recorded launch
        shape, as three numbers, and the number held by membership alone."""
        count, compared, loose = self.compared()
        if self.reasons:
            text = f"{self.name}: NOT CLAIMED ({'; '.join(self.reasons)})"
            if count and status == PASS:
                text += f"; {compared}: each in the recorded one" + (f"; {loose} more by membership alone" if loose else "")
            elif count and status == NOT_OBSERVED:
                text += "; the launch shapes of its target launches were not observed"
            return text
        if status == PASS:
            return f"{self.name}: EQUAL ({compared}, {loose} by membership alone)"
        verdicts = {FAIL: "NOT EQUAL (the kernel check did not hold)", NOT_OBSERVED: "NOT OBSERVED"}
        return f"{self.name}: {verdicts.get(status, 'not compared')}"

    def dump(self) -> dict:
        n = self.numbers() if self.known else {}
        return {
            "census": self.census,
            "configuration": self.configuration,
            "device": self.device,
            "rows": self.rows,
            "basis": self.basis,
            "recorded": self.recorded_on,
            "claimed": self.claimed and not self.reasons,
            "not_claimed": list(self.reasons),
            "numbers": n,
        }


class Target:
    """A census of a target kernel set, as a family checks it in (``kernel_census.json``): the kernels a decode step launches
    when every block runs on that set.

    * ``kernels``: per canonical kernel id its origin, its relation -- ``target`` of kind ``code`` (the target's own compiled
      kernel: an entry, a library call or a framework operator the sample calls as the target does) or ``port`` (a kernel
      whose home is cuDNN Frontend: an entry with byte-equal results), or ``stand-in`` -- and, once a stack serves it, the
      entry and the form the target calls it in.  A kernel with ``selection`` is chosen by a library's tuner: no launch
      shape is recorded for it;
    * ``layers``: which entry list of a section a layer kind of the model reads;
    * ``configurations``: per configuration and per ``ccMAJOR.MINOR/smCOUNT`` a section: ``claimed``, its ``basis``
      (``recorded`` or ``planned``), what was recorded, ``geometry`` -- the layer widths of the model it holds for, under
      the names of the family's geometry (a whole number each, or a list of them for a width that has parts) -- ``tokens`` -- the tokens per sequence of the step it holds for, where that
      is more than one -- and per row count the entries of a step as ``[block, kernel id, facts]``,
      before the layers, per layer kind, after the layers, after the step.  The facts of an entry (:data:`ENTRY_FACTS`):
      ``launch`` = ``[grid, block, shared memory]``, ``at`` = further fields of the geometry that this one launch
      shape was recorded with, and ``form`` = the fields of the kernel's form the target passes at this call beyond the
      form the kernel has everywhere (a kernel the target calls in several forms), and ``run`` = the facts of the run
      that this one launch shape was recorded with (:data:`RUN_FACTS`).  ``first_layer`` (optional): per
      block the facts that differ at its first call in the model's first layer.

    ``ValueError`` for a file that is not such a census; for a section that says ``claimed`` on a basis that is not
    ``recorded``, without the widths it holds for, or with a target kernel whose launch shape it does not record (only a
    kernel a tuner selects and a stand-in have none); and for an entry with a fact this reader does not know or a
    ``launch`` that is not a launch shape -- a misspelt key would otherwise read as "nothing recorded"."""

    def __init__(self, data: Mapping, name: str = "kernel_census.json"):
        self.name = name
        self.data = self._checked(data)

    @classmethod
    def load(cls, path: str) -> "Target":
        name = os.path.basename(os.path.normpath(path))
        try:
            with open(path, encoding="utf-8") as census:
                data = json.load(census)
        except OSError as error:
            raise ValueError(f"{name}: {error.strerror or type(error).__name__}") from None
        except ValueError as error:
            raise ValueError(f"{name}: not JSON ({error})") from None
        return cls(data, name)

    def _checked(self, data) -> Mapping:
        def wrong(what: str):
            raise ValueError(f"{self.name}: {what}")

        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            wrong(f"not a kernel census of schema {SCHEMA}")
        kernels, layers, configurations = data.get("kernels"), data.get("layers"), data.get("configurations")
        if not all(isinstance(part, dict) for part in (kernels, layers, configurations)):
            wrong("a kernel census holds kernels, layers and configurations")
        for kernel_id, meta in kernels.items():
            relation = meta.get("relation") if isinstance(meta, dict) else None
            if relation not in (TARGET, STAND_IN) or (relation == TARGET and meta.get("kind") not in KINDS) or not isinstance(meta.get("form", {}), dict):
                wrong(f"kernel {kernel_id}: a kernel is a target of kind {' or '.join(KINDS)}, or a stand-in; its form is an object")
        for configuration, sections in configurations.items():
            for key, section in sections.items():
                where = f"{configuration} / {key}"
                if not re.fullmatch(r"cc\d+\.\d+/sm\d+", key):
                    wrong(f"{where}: a section is keyed by ccMAJOR.MINOR/smCOUNT")
                if not isinstance(section.get("claimed"), bool) or section.get("basis") not in BASES or not isinstance(section.get("rows"), dict):
                    wrong(f"{where}: a section holds claimed (true or false), basis ({' or '.join(BASES)}) and rows")
                if section["claimed"] and section["basis"] != "recorded":
                    wrong(f"{where}: claimed on the basis {section['basis']}; only a recorded section can be claimed")
                geometry = section.get("geometry")
                if not isinstance(geometry, dict) or not all(_width(v) for v in geometry.values()):
                    wrong(f"{where}: a section holds geometry, the layer widths of the model it was taken on (name: whole number, or a list of them)")
                if section["claimed"] and not geometry:
                    wrong(f"{where}: claimed without the layer widths it holds for")
                tokens = section.get("tokens", 1)
                if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1 or (tokens == 1) != (configuration != VERIFY):
                    wrong(f"{where}: tokens is the tokens per sequence of the section's step: more than one in {VERIFY}, one (or absent) elsewhere")
                for rows, parts in section["rows"].items():
                    first = parts.get(FIRST_LAYER, {})
                    if not isinstance(first, dict) or not all(isinstance(facts, dict) and set(facts) <= set(ENTRY_FACTS) for facts in first.values()):
                        wrong(f"{where}, {rows} rows: {FIRST_LAYER} is not an object of block: facts ({' and '.join(ENTRY_FACTS)})")
                    for block, facts in first.items():
                        if not isinstance(facts.get("form", {}), dict) or ("launch" in facts and not _launch_shape(facts["launch"])):
                            wrong(f"{where}, {rows} rows, {FIRST_LAYER}: the form of {block} is not an object, or its launch is not a launch shape")
                        if not _run_facts(facts.get("run", {})):
                            wrong(
                                f"{where}, {rows} rows, {FIRST_LAYER}: run of {block} is not an object of facts of a run ({' and '.join(RUN_FACTS)}: whole number)"
                            )
                    for part in (*PARTS, *layers.values()):
                        if not isinstance(parts.get(part), list):
                            wrong(f"{where}, {rows} rows: no entry list {part}")
                        for entry in parts[part]:
                            if not isinstance(entry, list) or len(entry) != 3 or entry[1] not in kernels or not isinstance(entry[2], dict):
                                wrong(f"{where}, {rows} rows, {part}: {entry!r} is not [block, kernel of the kernels table, facts]")
                            block, kernel_id, facts = entry
                            unknown = sorted(set(facts) - set(ENTRY_FACTS))
                            if unknown:
                                wrong(f"{where}, {rows} rows, {part}: {block} states {', '.join(unknown)}; an entry states {' and '.join(ENTRY_FACTS)}")
                            if "launch" in facts and not _launch_shape(facts["launch"]):
                                wrong(f"{where}, {rows} rows, {part}: the launch of {block} is not [grid of three, block of three, shared memory in bytes]")
                            at = facts.get("at", {})
                            if not isinstance(at, dict) or not all(isinstance(v, int) and not isinstance(v, bool) for v in at.values()):
                                wrong(f"{where}, {rows} rows, {part}: at of {block} is not an object of geometry fields (name: whole number)")
                            if not isinstance(facts.get("form", {}), dict):
                                wrong(f"{where}, {rows} rows, {part}: the form of {block} is not an object of form fields")
                            if any(name in facts and (not isinstance(facts[name], str) or not facts[name]) for name in ("class", "pick")):
                                wrong(f"{where}, {rows} rows, {part}: class and pick of {block} are names: a kernel class, a kernel")
                            if not _run_facts(facts.get("run", {})):
                                wrong(
                                    f"{where}, {rows} rows, {part}: run of {block} is not an object of facts of a run ({' and '.join(RUN_FACTS)}: whole number)"
                                )
                            meta = kernels[kernel_id]
                            if section["claimed"] and "launch" not in facts and meta["relation"] == TARGET and "selection" not in meta:
                                wrong(
                                    f"{where}, {rows} rows, {part}: claimed without the launch shape of {block} ({kernel_id}); "
                                    "only a kernel a library's tuner selects and a stand-in have none"
                                )
        return data

    def sequence(self, parts: Mapping, layer_kinds: Sequence[str]) -> List[Tuple[str, str, Mapping]]:
        """The entries of one step of a model with these layer kinds, in step order: ``(block, kernel id, facts)``.  Where
        the section says what differs in the first layer (``first_layer``), the first call of each block it names there
        carries those facts, its form merged field by field."""
        try:
            layers = [[tuple(entry) for entry in parts[self.data["layers"][kind]]] for kind in layer_kinds]
        except KeyError as error:
            raise ValueError(f"{self.name}: the census has no entry list for the layer kind {error.args[0]!r}") from None
        for block, facts in parts.get(FIRST_LAYER, {}).items() if layers else ():
            at = next((index for index, entry in enumerate(layers[0]) if entry[0] == block), None)
            if at is None:
                raise ValueError(f"{self.name}: {FIRST_LAYER} names {block}, which the first layer of this model does not call")
            _, kernel_id, own = layers[0][at]
            merged = {**own, **facts}
            if "form" in facts:
                merged["form"] = {**own.get("form", {}), **facts["form"]}
            layers[0][at] = (block, kernel_id, merged)
        return [tuple(entry) for entry in (*parts["prologue"], *(entry for layer in layers for entry in layer), *parts["epilogue"], *parts["after_step"])]

    def kind(self, kernel_id: str) -> str:
        """``code`` or ``port`` for a target kernel, ``""`` for a stand-in."""
        return self.data["kernels"][kernel_id].get("kind", "") if self.data["kernels"][kernel_id]["relation"] == TARGET else ""

    def numbers(self, parts: Mapping, layer_kinds: Sequence[str]) -> Tuple[int, int, int, int]:
        """``(launches, the target's own compiled kernels, ports, stand-ins)`` of one step of a model with these layer kinds."""
        kinds = [self.kind(kernel_id) for _, kernel_id, _ in self.sequence(parts, layer_kinds)]
        return len(kinds), kinds.count("code"), kinds.count("port"), kinds.count("")

    def claim(self, plan, model, configuration: str = "plain", rows: Optional[int] = None, installed: Optional[Mapping[str, Optional[str]]] = None) -> Claim:
        """What a run of ``plan`` can say about this target: see :class:`Claim`.  ``installed`` replaces the installed versions
        of the packages the census records (tests).  ``ValueError`` when the census does not follow the model's wiring."""
        rows = plan.sequences * plan.tokens if rows is None else rows
        wiring = list(model.wiring())
        key = f"cc{plan.facts.cc[0]}.{plan.facts.cc[1]}/sm{plan.facts.sm_count}"
        sections = self.data["configurations"].get(configuration, {})
        section = sections.get(key)
        claim = Claim(self.name, configuration, _class_text(key), rows, blocks=[block for block, _ in wiring], sites=list(call_sites(model)))
        reasons = [f"not the target set: {block} selected by --provider" for block in getattr(plan, "overridden", ())]
        called = set(claim.blocks)
        if any(row.fallback for _, block, _, row, _ in _parts(plan) if block in called):
            reasons.append("fallback rows present")
        if section is None:
            have = ", ".join(_class_text(other) for other in sections) or "none"
            reasons.append(f"no recorded section for {claim.device}: recorded is {have}")
        else:
            claim.recorded_on, claim.basis, claim.claimed = str(section.get("recorded", {}).get("date", "")), section["basis"], section["claimed"]
            parts = section["rows"].get(str(rows))
            widths = _other_widths(section["geometry"], model.geometry)
            tokens = getattr(plan, "tokens", 1)
            if widths:  # a model of the family with other layer widths: other grids, and where an entry picks by width other kernels
                reasons.append(f"no recorded section for this model's widths on {claim.device}: {widths}")
            elif section.get("tokens", 1) != tokens:  # a step of another tokens count per sequence: another kernel form, other launch shapes
                reasons.append(f"no recorded section for {tokens} tokens per sequence on {claim.device}: recorded is {section.get('tokens', 1)}")
            elif parts is None:
                have = [f"{count} row{'s' if count != '1' else ''}" for count in section["rows"]]
                reasons.append(f"no recorded section for {rows} row{'s' if rows != 1 else ''} on {claim.device}: recorded is {' and '.join(have) or 'none'}")
            else:
                self._identify(claim, plan, wiring, self.sequence(parts, model.geometry.layer_kinds), model.geometry, claim.sites)
                found = []  # per call, so that a block that is the target's at one call site and not at another is named
                for block, identity in zip(claim.blocks, claim.identities):
                    if identity.relation == OTHER and block not in getattr(plan, "overridden", ()):
                        found.append(f"not the target set: {block} ({identity.reason})")
                    elif identity.relation in (TARGET, STAND_IN):
                        found += [f"no recorded launch shape of {block} for this model: {identity.reason}"] if identity.reason else []
                        found += [f"no recorded launch shape of {block} for this run: {identity.run}"] if identity.run else []
                reasons += list(dict.fromkeys(found))
                reasons += [f"the plan has no block for {block}" for block in dict.fromkeys(block for block, _, _ in claim.absent)]
                if not reasons and not section["claimed"]:
                    reasons.append("the census does not claim this class yet")
        recorded = {package: version for package, version in self.data.get("validated", {}).get("recorded", {}).items() if package in PACKAGES}
        installed = installed_versions(tuple(recorded)) if installed is None else installed
        other = [
            f"{PACKAGES[package]} {installed[package].split('+')[0]} (recorded with {version})"
            for package, version in recorded.items()
            if installed.get(package) is not None and installed[package].split("+")[0] != version
        ]
        if other:
            reasons.append("versions other than the recorded ones: " + ", ".join(other))
        claim.reasons = reasons
        return claim

    def _identify(self, claim: Claim, plan, wiring, sequence, geometry=None, sites: Sequence[str] = ()) -> None:
        """Walk the wiring beside the census's entries of a step and give every block call its identity (``sites``: the call
        site of every call, ``call_sites``).  A census entry whose block the plan does not have is kept as absent; any other
        disagreement of the two orders is a ``ValueError``."""
        position = 0
        for index, (block, _) in enumerate(wiring):
            try:
                declared = tuple(_part(plan, block, sites[index] if index < len(sites) else "")[1].kernels)
            except ValueError:  # a call site no row of the block names: said by the identity of the call, below
                declared = tuple(plan[block].kernels)
            if declared and all(kernel.relation == "host" for kernel in declared):
                claim.identities.append(Identity(HOST))
                continue
            while position < len(sequence) and sequence[position][0] not in plan:
                claim.absent.append((sequence[position][0], sequence[position][1], self.kind(sequence[position][1])))
                position += 1
            if position >= len(sequence) or sequence[position][0] != block:
                have = sequence[position][0] if position < len(sequence) else "nothing"
                raise ValueError(
                    f"{self.name}: the census does not follow this model's wiring: call {index + 1} of a step is {block}, the census has {have} there"
                )
            _, kernel_id, facts = sequence[position]
            position += 1
            claim.identities.append(self._identity(plan, block, kernel_id, facts, geometry, sites[index] if index < len(sites) else ""))
        for block, kernel_id, _ in sequence[position:]:
            if block in plan:
                raise ValueError(f"{self.name}: the census does not follow this model's wiring: it has {block} after the last call of a step")
            claim.absent.append((block, kernel_id, self.kind(kernel_id)))

    def _identity(self, plan, block: str, kernel_id: str, facts: Mapping, geometry=None, site: str = "") -> Identity:
        meta, kind = self.data["kernels"][kernel_id], self.kind(kernel_id)
        try:
            row, part = _part(plan, block, site)  # of this call's site, for a block whose calls several rows serve
        except ValueError as error:  # the wiring calls the block at a site no row of it names
            return Identity(OTHER, kernel_id, kind, str(error))
        exact = [kernel for kernel in part.kernels if kernel.relation == "exact"]
        if not exact:
            return Identity(NOT_YET, kernel_id, kind, row.fallback or "no kernel of its own")
        if len(exact) != 1 or exact[0].id != kernel_id:
            return Identity(OTHER, kernel_id, kind, f"the row launches {', '.join(kernel.id for kernel in exact)}, the target {kernel_id}")
        # the target's form at this call: the form its kernel has everywhere and, behind it, what the census names for this call
        # (a field of several values is a list in the file and a tuple in a provider table: one value)
        want = tuple((name, _plain(value)) for name, value in {**meta.get("form", {}), **facts.get("form", {})}.items())
        try:
            have = tuple((name, _plain(value)) for name, value in row.site_form(site))
        except ValueError as error:  # the wiring calls the block at a site its row has no form for
            return Identity(OTHER, kernel_id, kind, str(error))
        if dict(have) != dict(want):
            where = f" at {site}" if site else ""
            return Identity(OTHER, kernel_id, kind, f"the row's form{where} is{form_text(have) or ' []'}, the target's{form_text(want) or ' []'}")
        launch = facts.get("launch")
        shape = None if launch is None else (tuple(launch[0]), tuple(launch[1]), int(launch[2]))  # JSON lists: rebuilt before any comparison
        # a launch shape recorded with a field of the geometry this model does not share (the output head's: the vocabulary) is
        # not this model's: it is held by nothing, and the claim says so
        other = _other_widths(facts.get("at", {}), geometry) if shape is not None else ""
        # ... and so is one recorded at a fact of the run that this run does not state (the maximum length a kernel is told):
        # asked of what the plan handed the block of this call
        apart = _other_run(facts.get("run", {}), getattr(part, "options", None) or {})
        held = not (other or apart)
        named = str(facts.get("class", "")) if held else ""  # recorded with the launch shape: held where that one is
        relation = STAND_IN if meta["relation"] == STAND_IN else TARGET
        return Identity(relation, kernel_id, kind, other, shape if held else None, named, str(facts.get("pick", "")), apart)


# ---------------------------------------------------------------------------------------------------------- a traced pass
@dataclass
class Traced:
    """What a session keeps of one traced pass for the kernel check: the sites of a step, the call log of every traced step
    (one list of calls per step) and the launch records, cut into steps -- or the reason there are none."""

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
    claim: Optional[Claim] = None  # of a stack that declares a target kernel set: what this run can say about it
    #: the index of the first site the claim speaks about: the sites in front of it are a drafter's passes, which no census holds
    first: int = 0

    @classmethod
    def of(cls, sites, call_log, records: Sequence[Launch], count: int, missing: str, rows: int, claim: Optional[Claim] = None, first: int = 0) -> "Traced":
        """From the launch records of a pass of ``count`` steps (``records`` empty: ``missing`` says why).  ``first``: where
        the model's sites begin among ``sites`` (behind the passes of a drafter)."""
        if not records:
            return cls(sites, call_log, missing=missing or "the profiler recorded no kernel activity on this GPU", rows=rows, claim=claim, first=first)
        steps, _ = split_steps(records)
        if len(steps) != count:  # a block read the device inside a step: an undeclared copy, and no step can be told from the next
            undeclared = f"the records show {len(steps)} host reads for {count} traced steps"
            return cls(sites, call_log, undeclared=undeclared, rows=rows, claim=claim, first=first)
        return cls(sites, call_log, steps, rows=rows, claim=claim, first=first)

    def _recorded(self) -> Dict[int, tuple]:
        """Site index -> the launch shape the claim's census recorded there, counted among this pass's own sites."""
        return {} if self.claim is None else {index + self.first: shape for index, shape in self.claim.recorded.items()}

    def _classes(self) -> Dict[int, str]:
        """Site index -> the kernel class the claim's census recorded there for a library kernel, counted like :meth:`_recorded`."""
        return {} if self.claim is None else {index + self.first: named for index, named in self.claim.classes.items()}

    def _picks(self) -> Dict[int, str]:
        """Site index -> the kernel a library's tuner chose there in the recorded run, counted like :meth:`_recorded`."""
        return {} if self.claim is None else {index + self.first: pick for index, pick in self.claim.picks.items()}

    def judge(self) -> Tuple[str, List[str]]:
        """Run the kernel check on this pass: ``(PASS | FAIL | NOT OBSERVED, the lines under the check's own line)``.  Where the
        pass has launch records it has one call log per traced step, or it fails: the form check reads from the log which
        entry instance served a call, and the comparison whether a step made the calls of the wiring."""
        failures = check_forms(self.sites, self.call_log)
        lines = [f"form: {sentence}" for sentence in failures]
        if self.undeclared:
            lines.append(f"copies: {self.undeclared}: a block reads the device inside a step, and only the one host read of a step is declared")
        if self.steps is not None and len(self.call_log) != len(self.steps):
            logged, traced = len(self.call_log), len(self.steps)
            has = "no step" if not logged else f"{logged} step{'s' if logged != 1 else ''}"
            lines.append(
                f"calls: the call log holds the block calls of {has} for {traced} traced step{'s' if traced != 1 else ''}: "
                "without one log per traced step nothing shows which entry instance served a call"
            )
        if self.steps is not None:
            recorded, name = (None, "") if self.claim is None else (self._recorded(), f"the {self.claim.name}")
            self.result = compare(self.steps, self.sites, self.call_log, recorded, name, None if self.claim is None else self._classes())
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
        calls = Counter((site.block, site.site) for site in self.sites)
        # what the census records is recorded per call: a block whose call sites two rows serve has it for one of them alone
        recorded = {(self.sites[index].block, self.sites[index].site): shape for index, shape in self._recorded().items()}
        classes = {(self.sites[index].block, self.sites[index].site): named for index, named in self._classes().items()}
        for site in {(site.block, site.site): site for site in self.sites}.values():
            for kernel in site.kernels:
                if kernel.relation != "exact":
                    continue
                made, total = calls[(site.block, site.site)], calls[(site.block, site.site)] * kernel.launches
                count = f"{total} launch{'es' if total != 1 else ''} per step ({made} call{'s' if made != 1 else ''} x {kernel.launches} per call)"
                if site.facts is not None:
                    symbols = ", ".join(site.facts.symbols) or "none compiled"
                    lines.append(
                        f"{site.row}: {site.entry}{form_text(site.form)}, {count}; the entry states {shape_text(site.facts.shape)} and the kernel {symbols}"
                    )
                elif kernel.shape == "entry":
                    lines.append(f"{site.row}: {site.entry}{form_text(site.form)}, {count}; the entry states no launch for this run")
                elif (site.block, site.site) in recorded:  # no entry states its shape; the census of the target records one for this call site
                    at = (site.block, site.site)
                    what = "the launch shape" if at not in classes else f"the kernel class {classes[at]} and the launch shape"
                    lines.append(
                        f"{site.row}: {site.entry}{form_text(site.form)}, {count}; {kernel.id} by name, and in {what} the census records: "
                        f"{shape_text(recorded[at])}"
                    )
                else:
                    lines.append(f"{site.row}: {site.entry}{form_text(site.form)}, {count}; {kernel.id} by name (no launch shape is declared for it)")
        lines += self.served()
        lines += self.chosen()
        if self.result is not None:
            for family, picks in sorted(self.result.picks.items()):
                if family == "pytorch":
                    continue
                chosen = ", ".join(f"{name if len(name) <= 72 else name[:69] + '...'} x {count}" for name, count in picks.most_common(6))
                more = f", and {len(picks) - 6} more" if len(picks) > 6 else ""
                lines.append(f"chosen by a library inside the fallback rows (printed, not gated): {family}: {chosen}{more}")
        if self.claim is not None:  # a stack that declares a target: how much of it this run launched, and the census line
            lines += [*self.claim.coverage(), self.claim.line(self.status)]
        return lines

    def served(self) -> List[str]:
        """For every block whose row names several call sites of an entry: which entry instance served how many calls of the
        first traced step, in which form -- counted from the call log, which the adapters wrote after each call from the
        object they had called.  (Whether each call was served in the form of ITS site is item 0 of the check; this is the
        count.)  A block that is called at named sites and served by no entry there has no instance to count."""
        lines = []
        if not self.call_log:
            return lines
        named = {site.block for site in self.sites if site.site and site.facts is not None}
        counts: Dict[str, Counter] = {}
        for index, call in enumerate(self.call_log[0]):
            # the call's own place of the wiring, where the step made the wiring's calls: a call at a site that no entry
            # serves (another row's site of a block several rows serve) has no instance to count
            at = self.sites[index] if index < len(self.sites) and (self.sites[index].block, self.sites[index].layer) == (call.block, call.layer) else None
            if call.block in named and (at is None or at.facts is not None):
                counts.setdefault(call.block, Counter())[(call.instance, tuple(call.form))] += 1
        for block, by_instance in counts.items():
            parts = "; ".join(f"{count} x {instance or 'no entry instance'}{form_text(form)}" for (instance, form), count in by_instance.items())
            lines.append(f"{block}, as the call log of a traced step has it ({sum(by_instance.values())} calls): {parts}")
        return lines

    def chosen(self) -> List[str]:
        """For every block whose kernel a library's tuner chooses -- the census records the tuner's pick of the recorded run,
        never a launch shape -- the kernels the traced steps launched for it, and whether they are that pick.  Printed and
        compared, never gated: the choice is the library's, and another process's tuner may make another."""
        if self.claim is None or self.result is None:
            return []
        lines, rows = [], {}
        for index, pick in self._picks().items():
            rows.setdefault(self.sites[index].row, pick)
        for row, pick in rows.items():
            names = self.result.served.get(row)
            if not names:
                continue
            launched = ", ".join(f"{_named(name)} x {count}" for name, count in names.most_common(4))
            same = "the pick the census records" if set(names) == {pick} else f"not the pick the census records ({pick}): another kernel of the same entry"
            lines.append(f"chosen by the library for {row} (printed and compared, never gated): {launched} over the traced steps; {same}")
        return lines

    def dump(self) -> dict:
        """The census of this pass as plain data: the sites of a step -- each with the row's form (``form``) and, from the call
        log, the entry instance that served it and the form that instance states (``instance``, ``served``) -- and the launch
        records of every traced step, names stored once, each with its start and its end in milliseconds on the
        profiler's clock."""
        names: Dict[str, int] = {}
        steps = []
        for step in self.steps or []:
            facts = [[r.kind, r.grid, r.block, r.shared_memory, r.registers, *r.graph, r.bytes, r.start, r.end] for r in step]
            steps.append([[names.setdefault(r.name, len(names)), canonical(r), *rest] for r, rest in zip(step, facts)])
        logged = {(call.block, call.layer): call for calls in self.call_log for call in calls}
        unlogged = Call("", -1, self.rows, "", ())  # of a pass that made no traced step
        first = self.call_log[0] if self.call_log else []

        def call_at(index: int, site: Site) -> Call:
            """The call of the first traced step at this place of the wiring (a block may be called twice in a layer), else a
            call of that block and layer from any step."""
            if index < len(first) and (first[index].block, first[index].layer) == (site.block, site.layer):
                return first[index]
            return logged.get((site.block, site.layer), unlogged)

        stated = self.claim is not None and self.claim.known
        return {
            "rows": self.rows,
            "check": self.status,
            "not_observed": self.unobserved,
            "target": None if self.claim is None else self.claim.dump(),
            "sites": [
                {
                    "block": site.block,
                    "layer": site.layer,
                    "site": site.site,
                    "entry": site.entry,
                    "identity": self.claim.identities[index - self.first].text() if stated and index >= self.first else "",
                    "instance": call_at(index, site).instance,
                    "form": {name: value for name, value in site.form},
                    "served": {name: value for name, value in call_at(index, site).form},
                    "kernels": [{"id": k.id, "origin": k.origin, "relation": k.relation, "launches": k.launches, "shape": k.shape} for k in site.kernels],
                    "stated": (
                        None
                        if site.facts is None
                        else {"symbols": list(site.facts.symbols), "launch": [list(part) if isinstance(part, tuple) else part for part in site.facts.shape]}
                    ),
                }
                for index, site in enumerate(self.sites)
            ],
            "names": list(names),
            "fields": ["name", "kernel", "kind", "grid", "block", "shared_memory", "registers", "graph_id", "graph_node_id", "bytes", "start_ms", "end_ms"],
            "steps": steps,
        }


# ---------------------------------------------------------------------------------------------------------- the two tables
def _total(calls: str) -> int:
    return sum(int(part) for part in calls.split("/") if part.strip().isdigit())


def table(plan, claim: Optional[Claim] = None) -> str:
    """The kernel table, first part: planned.  Per block the census entries of its row -- which kernels a call launches and
    how the kernel check will judge them -- decided from the plan alone, before anything is loaded.  With the ``claim`` of a
    stack that declares a target kernel set, a last column says how each block stands to the target, and the lines under the
    table how much of the target the plan launches.  A block whose calls several rows serve by call site has one line per
    site, ``<block> at <call site>``, with the row, the entry and the whole form of that site."""
    stated = claim is not None and claim.known  # the identity column is printed where the census holds this run
    head = ("block", "calls", "provider", "entry [form]", "kernel", "launches", "checked") + (("identity",) if stated else ())
    lines = [head]
    exact = rows = chains = 0
    parts = _parts(plan)
    for label, name, site, row, block in parts:
        calls = plan.calls.get(label, "")
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
            checked = "by family" + (f"; fallback: {row.fallback}" if row.fallback and claim is None else "")
        # a part of a block several rows serve is one call site: its form is printed whole.  A row that serves every call
        # of its block in a form per call site prints the fields they share, and the sites under the table
        forms = f" + one of {len(row.served)} forms, by call site" if row.sites and not site else ""
        form = row.site_form(site) if site else row.form
        cells = (label, calls, block.provider, block.entry + form_text(form) + forms, ", ".join(kernel.id for kernel in kernels) or "--", launches, checked)
        lines.append(cells + ((claim.of(name, site or None),) if stated else ()))
    widths = [max(len(line[i]) for line in lines) for i in range(len(head))]
    sms = f", {plan.facts.sm_count} SMs" if plan.facts.sm_count else ""
    several = plan.tokens_text() if hasattr(plan, "tokens_text") else ""
    run = f"stack {plan.stack}, {plan.sequences} sequence{'s' if plan.sequences > 1 else ''}{several}, {plan.facts.class_text}{sms}"
    text = [f"kernels, planned (decided before the load from {plan.source}; {run})"]
    text += ["  " + "  ".join(cell.ljust(width) for cell, width in zip(line, widths)).rstrip() for line in lines]
    for _, name, site, row, _ in parts:  # a row whose entry is called in several forms: the form of every call site it serves
        text += [f"  {name} at {at}:{form_text(fields)}" for at, fields in row.sites if fields is not None and not site]
    text.append(
        f"  per step: {exact} launch{'es' if exact != 1 else ''} on {rows} exact row{'s' if rows != 1 else ''}, checked one by one in the wiring order; "
        f"the launches of {chains} fallback call{'s' if chains != 1 else ''} and of the host path are attributed by family"
    )
    if claim is not None:
        text += [f"  {line}" for line in claim.coverage()]
    return "\n".join(text)


def stated(plan, rows: Optional[int] = None) -> str:
    """The kernel table, second part: what the entries of the plan state about their launch once they are built -- the form an
    entry resolved for this run's row count, its launch shape and the symbol of the kernel it compiled.  Never a row: an
    entry's own pick is its answer, not a choice of the table."""
    rows = plan.sequences * plan.tokens if rows is None else rows
    text = [f"kernels, as the entries state them (after the load; {rows} row{'s' if rows != 1 else ''}; never a row of the table)"]
    for _, name, at, row, block in _parts(plan):
        if not any(kernel.shape == "entry" for kernel in block.kernels):
            continue
        for site in [at] if at else list(row.served) or [""]:  # an entry that is called in several forms states each one's launch
            where = f"{name} at {site}" if site else name
            try:
                facts = block.launch_facts(rows, site) if site else block.launch_facts(rows)
            except (ValueError, NotImplementedError) as error:
                text.append(f"  {where}  {block.entry}: states no launch for this run ({error})")
                continue
            symbols = ", ".join(facts.symbols) or "no kernel compiled yet"
            text.append(f"  {where}  {block.entry}{form_text(row.site_form(site))}: {shape_text(facts.shape)}")
            text.append(f"    {symbols}")
    if len(text) == 1:
        text.append("  no block of this plan is served by an entry that states its launch")
    return "\n".join(text)
