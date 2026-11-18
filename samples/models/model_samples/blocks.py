# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Who serves each block of a model: provider rows as data, the plan that resolves them before anything is loaded, and the
served-by table a sample prints before it runs.

A model family declares, per block, an ordered list of :class:`ProviderRow`.  :func:`plan` picks, for every block, the FIRST row
whose GPU classes, sequence range, tokens-per-sequence range and stack contain the run -- and, for a row that names the layer
widths its entry serves (``widths``), whose widths are the model's -- builds its :class:`Block` and asks it whether it serves
the run.  Nothing is chosen at run time and nothing falls back on failure:

* ``NotImplementedError`` from a selection or from ``Block.check_support`` means "not served here": the sample prints
  ``[NN] SKIP <sentence>`` and exits 0;
* ``ValueError`` means the caller violated a contract (an unknown stack, a provider key that is not a declared row).

The stack is explicit (``"fe-cublas"``, ``"native"``, ``"fe-entries"``, ``"reference"``): a row outside the requested stack is
never selected, whatever is installed.  ``--provider BLOCK=KEY`` selects one declared row of a block for an A/B; the plan
keeps which blocks were selected that way (``Plan.overridden``).

A row also says in which FORM its entry is called (``form``: the fields of the entry's form the row fixes, which the adapter
passes on and the kernel check compares with the entry's own answer), and a block declares the kernels it launches as census
entries (``Block.kernels``, :mod:`model_samples.census`).  An entry that a step calls in several forms has them in its row
too (``sites``: per call site of the family's wiring the further fields of the form there; ``ProviderRow.site_form``): which
form serves which call is data of the table, never a choice of the adapter.  A row may also serve some call sites of its
block only: it names every other site without fields, which hands the calls there on to the next row of the stack that
serves the run.  The plan then holds, under the block's name, one part per call site (:class:`BySite`): the row and the
block that serve the calls there.  ``Plan.at`` gives the part of a call and ``Plan.parts`` every part of a plan; a block
that one row serves is one part.

This module imports nothing beyond the standard library: a plan on stated device facts (:class:`DeviceFacts`) needs no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Protocol, Sequence, Tuple, Union

from .census import Kernel

GIB = 1024**3


@dataclass(frozen=True)
class ProviderRow:
    """One row of a provider table (data in ``<family>/configs.py``)."""

    classes: Tuple[str, ...]  # ("cc10.0", "cc10.3") | ("cc10.x", "cc12.x") | ("any",)
    sequences: Tuple[int, Optional[int]]  # the selection key is (GPU class, sequences, tokens per sequence):
    tokens: Tuple[int, Optional[int]]  # some entries have a two-dimensional limit
    key: str  # "fe.gdn_decode_fused" | "cublas.bf16_dequantized" | ...: one of the closed list in providers.KEYS
    stack: Union[str, Tuple[str, ...]]  # "fe-cublas" | "native" | "fe-entries" | "reference" | "selectable" (only through --provider), or several
    serves: str  # what the entry admits, in rows
    measured: Tuple[str, ...] = ()  # where a published record of the entry exists (GPU class, rows); the table prints "none" otherwise
    fallback: str = ""  # non-empty for a reference chain or a dequantised copy: the reason printed in the table
    band: str = "same"  # the tie-out band against the reference block (tie_out.BANDS); never widened in place
    batch_invariant: Tuple[str, ...] = ()  # GPU classes on which "a sequence alone = the sequence in a batch" is a gate
    ci: bool = True  # False for a row that no test of this repository runs: the table prints "unverified in CI"
    form: Tuple[Tuple[str, object], ...] = ()  # what the adapter passes to the entry: (field of the entry's form, value) pairs
    rows: Tuple[int, Optional[int]] = (1, None)  # a third selection range: rows = sequences x tokens per sequence
    # an entry that is called in several forms: (call site of the model's wiring, the further fields of the entry's form there).
    # ``None`` in place of the fields: the row serves no call at that site and hands it on to the next row that serves the run
    sites: Tuple[Tuple[str, Optional[Tuple[Tuple[str, object], ...]]], ...] = ()
    # the layer widths the row's entry serves, where it serves some only: (field of the model's geometry, value) pairs.  A
    # model that differs in one of them is not this row's, and the next row of the block is asked
    widths: Tuple[Tuple[str, int], ...] = ()

    @property
    def stacks(self) -> Tuple[str, ...]:
        return (self.stack,) if isinstance(self.stack, str) else tuple(self.stack)

    def serves_class(self, cc: Tuple[int, int]) -> bool:
        return any(c == "any" or c == f"cc{cc[0]}.{cc[1]}" or c == f"cc{cc[0]}.x" for c in self.classes)

    def site_form(self, site: str = "") -> Tuple[Tuple[str, object], ...]:
        """The form the row's entry is called in at ``site`` (a call site as the family's wiring names it, ``model.sites()``):
        the row's ``form`` and, for a row that names its sites, the fields of that site behind it.  ``ValueError`` for a site
        such a row does not name -- the wiring calls the block where the table says nothing -- and for a site it hands on
        to the next row (``passed``): the form of a call there is that row's."""
        if not self.sites:
            return tuple(self.form)
        named = dict(self.sites)
        if site not in named:
            raise ValueError(f"provider row {self.key}: no form for the call site {site!r}; the row names {', '.join(name for name, _ in self.sites)}")
        if named[site] is None:
            raise ValueError(f"provider row {self.key}: it hands the call site {site!r} on to the next row of its block and has no form there")
        return (*self.form, *named[site])

    @property
    def served(self) -> Tuple[str, ...]:
        """The call sites the row names and serves, in its order."""
        return tuple(site for site, fields in self.sites if fields is not None)

    @property
    def passed(self) -> Tuple[str, ...]:
        """The call sites the row names and hands on: the calls there are served by the next row of the stack that serves
        the run (``plan`` resolves it; the plan's block is then a :class:`BySite`)."""
        return tuple(site for site, fields in self.sites if fields is None)

    def serves_run(self, sequences: int, tokens: int) -> bool:
        return _inside(self.sequences, sequences) and _inside(self.tokens, tokens) and _inside(self.rows, sequences * tokens)

    def serves_model(self, model) -> bool:
        """Whether the row serves ``model``: a row that names layer widths (``widths``) serves a model whose geometry has every
        one of them -- so not a plan without a model -- and a row that names none serves any."""
        geometry = getattr(model, "geometry", None)
        return all(getattr(geometry, name, None) == value for name, value in self.widths)

    def widths_text(self, model=None) -> str:
        """The widths the row names, ``hidden 5120``; with ``model``, what that model has in the ones that differ."""
        if model is None:
            return ", ".join(f"{name} {value}" for name, value in self.widths)
        geometry = getattr(model, "geometry", None)
        return ", ".join(f"{name} {getattr(geometry, name, 'none')}" for name, value in self.widths if getattr(geometry, name, None) != value)

    def classes_text(self) -> str:
        if "any" in self.classes:
            return "any compute capability"
        return "compute capability " + " or ".join(c[2:] for c in self.classes)


def _inside(bounds: Tuple[int, Optional[int]], value: int) -> bool:
    return bounds[0] <= value and (bounds[1] is None or value <= bounds[1])


@dataclass(frozen=True)
class DeviceFacts:
    """What a plan needs to know about the GPU.  ``DeviceFacts.probe`` reads a real device; a host-only plan states them."""

    cc: Tuple[int, int]
    name: str = "stated device"
    sm_count: int = 0
    total_bytes: int = 0
    free_bytes: int = 0

    @classmethod
    def probe(cls, device) -> "DeviceFacts":
        import torch

        if not torch.cuda.is_available():
            raise NotImplementedError("needs a CUDA GPU")
        index = torch.device(device).index
        index = torch.cuda.current_device() if index is None else index
        props = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        return cls(tuple(torch.cuda.get_device_capability(index)), props.name, int(props.multi_processor_count), int(total), int(free))

    @property
    def class_text(self) -> str:
        return f"cc {self.cc[0]}.{self.cc[1]}"


class Block(Protocol):
    """One block of a model as a provider serves it.  ``__call__`` takes the block's operands (per block kind: see the family's
    ``model.py``) and launches on the current stream: no host read, no synchronisation."""

    row: ProviderRow
    entry: str  # what is called, as printed in the table
    provider: str  # "cuDNN FE" | "cuBLAS" | "PyTorch" | ...
    kernels: Tuple[Kernel, ...]  # the block's census entries: the kernels a call launches, or the families of a fallback row
    per_row: Tuple[int, ...]  # the operands that have one leading row per sequence
    state_args: Tuple[int, ...]  # the operands the call updates in place

    def check_support(self, sequences: int, tokens: int) -> None:
        """Raise the provider's typed decline when it does not serve this run; return otherwise.  Allocates nothing on the
        GPU: the memory sentence of the plan comes after it."""

    def first_use(self, sequences: int, tokens: int) -> None:
        """What the block builds on the GPU before the load so that a decline still comes with the plan (an adapter to an
        entry builds the entry's kernel on scratch operands of the run's row count).  Made after the memory sentence held
        (``Plan.first_use``), so a run that does not fit allocates nothing.  Raises as ``check_support`` does."""

    def compile(self, sequences: int, tokens: int) -> None:
        """Build everything the run needs before its first step."""

    def reserve_bytes(self) -> int:
        """Device bytes the provider takes outside PyTorch's pool, for the memory sentence of the plan."""

    def first_launch(self) -> dict:
        """What the first launch of each compiled form took, measured at load."""

    def serving(self) -> Tuple[str, tuple]:
        """``(entry instance, form)`` of what served the block's last call: the line a traced step writes into its call log
        after the call.  The form is read from the object that was called -- the fields of the row's form as that entry
        instance states them -- and never copied from the row; a block without an entry has no instance and passed nothing."""

    def launch_facts(self, rows: int, site: str = ""):
        """What the block's entry states about its launch at ``rows`` (``census.Facts``: the symbols of its compiled kernels, its
        form, its launch shape), or ``None`` for a block without such an entry.  ``site`` names the call site for a row whose
        entry is called in several forms (``ProviderRow.sites``): each form is an entry instance of its own.  ``ValueError`` /
        ``NotImplementedError`` when the entry declines to state them for a row count the row claims."""

    def __call__(self, *operands):
        """Launch the block on the current stream."""


def scoped(block: str, site: str = "") -> str:
    """How the calls of a block at one call site are named where they are told apart from its other calls -- a line of a
    table, a sentence of a check, a weight a load is asked to keep: ``<block> at <call site>``; the block's own name for
    all of its calls."""
    return f"{block} at {site}" if site else block


def call_site(operands: Sequence) -> str:
    """The call site of a block call: a family's step hands it over as the call's last operand, a name of its wiring
    (``model.sites()``); ``""`` for a call of a block that is called in one form everywhere."""
    return operands[-1] if operands and isinstance(operands[-1], str) else ""


class BySite:
    """What a plan holds for a block whose calls several provider rows serve, by call site: ``parts`` is ``(call site, row,
    block)`` for every site the block's first row names, in that row's order -- the sites it serves with its own block,
    and each site it hands on with the next row that serves the run and that row's block.

    A call goes to the block of its site (the family's step hands the site over as the last operand), and what a plan asks
    of a block as a whole -- whether it serves the run, its first use, what it compiles and reserves -- is asked of every
    part.  What differs between the parts is never read from this object: the row, the entry, the kernels of a call and
    the form of the weight it reads are those of the call's part (``of``; ``Plan.at``, ``Plan.parts``)."""

    stored_weights = False  # asked of the parts: each reads its weight in its own form
    stored_layout = None
    batch_note = ""

    def __init__(self, name: str, parts: Sequence[tuple]):
        self.name, self.parts = name, tuple(parts)
        if not self.parts:
            raise ValueError(f"block {name}: no call site to serve")
        distinct = list({id(block): block for _, _, block in self.parts}.values())
        first = distinct[0]
        if any((tuple(block.per_row), tuple(block.state_args)) != (tuple(first.per_row), tuple(first.state_args)) for block in distinct):
            raise ValueError(f"the provider table is wrong: the rows that serve the call sites of {name} do not take the same operands")
        self._blocks = distinct
        self._last = None  # the part that served the last call
        self.row = self.parts[0][1]
        self.entry = " | ".join(dict.fromkeys(block.entry for block in distinct))
        self.provider = " | ".join(dict.fromkeys(block.provider for block in distinct))
        self.kernels = tuple(dict.fromkeys(kernel for block in distinct for kernel in block.kernels))
        self.per_row, self.state_args = tuple(first.per_row), tuple(first.state_args)
        self.compiles = any(getattr(block, "compiles", False) for block in distinct)

    def of(self, site: str) -> tuple:
        """``(row, block)`` of the calls at ``site``.  ``ValueError`` for a site no part names: the wiring calls the block
        where the table says nothing."""
        for name, row, block in self.parts:
            if name == site:
                return row, block
        raise ValueError(f"block {self.name}: no provider row for the call site {site!r}; its rows name {', '.join(name for name, _, _ in self.parts)}")

    def check_support(self, sequences: int, tokens: int) -> None:
        for block in self._blocks:
            block.check_support(sequences, tokens)

    def first_use(self, sequences: int, tokens: int) -> None:
        for block in self._blocks:
            block.first_use(sequences, tokens)

    def compile(self, sequences: int, tokens: int) -> None:
        for block in self._blocks:
            if getattr(block, "compiles", False):
                block.compile(sequences, tokens)

    def reserve_bytes(self) -> int:
        return sum(block.reserve_bytes() for block in self._blocks)

    def first_launch(self) -> dict:
        return {rows: taken for block in self._blocks for rows, taken in block.first_launch().items()}

    def serving(self) -> tuple:
        """What served the last call: the answer of the part that call went to (nothing before the first call)."""
        return ("", ()) if self._last is None else self._last.serving()

    def launch_facts(self, rows: int, site: str = ""):
        return self.of(site)[1].launch_facts(rows, site)

    def compared(self, out):
        return out if self._last is None else self._last.compared(out)

    def __call__(self, *operands):
        self._last = self.of(call_site(operands))[1]
        return self._last(*operands)


@dataclass(frozen=True)
class Regime:
    """One row of a family's regime table: for these GPU classes and sequence counts, what bounds a step."""

    classes: Tuple[str, ...]
    sequences: Tuple[int, Optional[int]]
    name: str  # "launch-bound" | "byte-bound"
    note: str

    def serves(self, cc: Tuple[int, int], sequences: int) -> bool:
        return ProviderRow(self.classes, self.sequences, (1, None), "", "", "").serves_class(cc) and _inside(self.sequences, sequences)


class Plan(dict):
    """Block name -> :class:`Block`, with what was decided: the selected rows, the device facts and the run."""

    def __init__(self, blocks, rows, calls, facts, sequences, tokens, stack, regime, source, overridden=()):
        super().__init__(blocks)
        self.rows: Dict[str, ProviderRow] = rows
        self.calls: Mapping[str, str] = calls
        self.facts: DeviceFacts = facts
        self.sequences, self.tokens, self.stack = sequences, tokens, stack
        self.regime: Optional[Regime] = regime
        self.source = source
        self.overridden: Tuple[str, ...] = tuple(overridden)  # the blocks whose row ``--provider`` selected: not the stack's own choice
        self.first_used = False  # whether the blocks have made their first use on the GPU (``first_use``)

    def first_use(self) -> None:
        """The first use every block makes of its entry on the GPU (``Block.first_use``), once per plan.  It is the one step
        of a plan that allocates on the GPU, so it is a step of its own: ``plan`` resolves the table and builds nothing, the
        memory sentence (``memory``) says whether the run fits, and only a run that fits comes here -- before anything is
        loaded.  Raises as ``plan`` does: ``NotImplementedError`` (a SKIP) where the installation does not serve an entry,
        ``ValueError`` where an entry declines what its adapter built."""
        if not self.first_used:
            for block in self.values():
                block.first_use(self.sequences, self.tokens)
            self.first_used = True

    def at(self, name: str, site: str = "") -> Tuple[ProviderRow, "Block"]:
        """``(row, block)`` of what serves the call of block ``name`` at the call site ``site``: the block's row and block,
        or -- for a block whose calls several rows serve by call site (:class:`BySite`) -- the row and the block of that
        site.  ``ValueError`` for a site such a block has no part for."""
        block = self[name]
        of = getattr(block, "of", None)
        return (self.rows[name], block) if of is None else of(site)

    def parts(self) -> list:
        """Everything that serves this plan, in table order, as ``(label, block name, call site, row, block)``: one part per
        block (label = the block's name, call site ``""``), and one per call site for a block whose calls several rows
        serve (label ``<block> at <call site>``, :func:`scoped`)."""
        found = []
        for name, block in self.items():
            split = getattr(block, "parts", None)
            if split is None:
                found.append((name, name, "", self.rows[name], block))
            else:
                found += [(scoped(name, site), name, site, row, part) for site, row, part in split]
        return found

    def announcements(self) -> list:
        """What the first use of this plan will do beyond building an entry's kernel, one sentence per kind
        (``Block.first_use_note``; a library that builds kernels of its own with the installed compiler), in table order.  A
        sample prints them before it makes that first use (``first_use``): a first build prints nothing while it runs."""
        return list(dict.fromkeys(note for _, _, _, _, block in self.parts() if (note := getattr(block, "first_use_note", ""))))

    def __eq__(self, other):
        """Two plans are equal when they hold the same blocks and the same decisions.  Against a plain mapping the comparison is
        the mapping's own (the blocks alone)."""
        if not isinstance(other, Plan):
            return NotImplemented
        mine = (self.rows, self.calls, self.facts, self.sequences, self.tokens, self.stack, self.regime, self.source, self.overridden)
        theirs = (other.rows, other.calls, other.facts, other.sequences, other.tokens, other.stack, other.regime, other.source, other.overridden)
        return dict.__eq__(self, other) and mine == theirs

    __hash__ = None  # a mapping with decisions attached: unhashable, like the dict it extends

    def by_provider(self) -> str:
        """The "by provider" line: block kinds per provider; a row with a ``fallback`` reason counts as a fallback."""
        kinds: Dict[str, set] = {}
        for _, name, _, row, block in self.parts():
            group = block.provider if not row.fallback else ("cuBLAS over copies" if "cuBLAS" in block.provider else "PyTorch chains")
            kinds.setdefault(group, set()).add(name.rsplit(".in", 1)[0].rsplit(".out", 1)[0] if name.startswith("projection.") else name)
        count = {group: len(members) for group, members in kinds.items()}
        fallbacks = {g: n for g, n in count.items() if g in ("cuBLAS over copies", "PyTorch chains")}
        parts = [f"{g} {n}" for g, n in count.items() if g not in fallbacks and g != "PyTorch"]
        if fallbacks:
            parts.append(
                f"fallbacks {sum(fallbacks.values())} ("
                + ", ".join(f"{g} {fallbacks[g]}" for g in ("cuBLAS over copies", "PyTorch chains") if g in fallbacks)
                + ")"
            )
        if "PyTorch" in count:
            parts.append(f"PyTorch {count['PyTorch']}")
        unverified = sum(not row.ci for _, _, _, row, _ in self.parts())
        if unverified:
            parts.append(f"unverified in CI: {unverified}")
        return "by provider: " + " | ".join(parts)

    def table(self) -> str:
        """The served-by table: decided before the load, printed before every run."""
        head = ("block", "calls", "provider", "entry", "serves", "published record")  # of the entry: not a measurement of this run or GPU
        lines = [head]
        for label, _, _, row, block in self.parts():
            if row.fallback:
                measured = f"fallback: {row.fallback}"
            else:
                measured = "; ".join(row.measured) if row.measured else "none"
            if not row.ci:
                measured += "; unverified in CI"
            lines.append((label, self.calls.get(label, ""), block.provider, block.entry, row.serves, measured))
        widths = [max(len(line[i]) for line in lines) for i in range(len(head))]
        run = f"stack {self.stack}, {self.sequences} sequence{'s' if self.sequences > 1 else ''}{self.tokens_text()}, {self.facts.class_text}"
        text = [f"served by (decided before the load from {self.source}; {run})"]
        text += ["  " + "  ".join(cell.ljust(width) for cell, width in zip(line, widths)).rstrip() for line in lines]
        text.append("  " + self.by_provider())
        if self.regime is not None:
            text.append(f"  declared regime: {self.regime.name} ({self.regime.note})")
        return "\n".join(text)

    def tokens_text(self) -> str:
        """`` x 6 tokens per sequence`` for a plan of a step that feeds several; nothing for a plan of the decode step."""
        return f" x {self.tokens} tokens per sequence" if self.tokens > 1 else ""

    def reserve_bytes(self) -> int:
        return sum(block.reserve_bytes() for block in self.values())

    def memory(self, need_bytes: int, what: str = "", checks_bytes: int = 0) -> str:
        """The memory sentence, in GiB (2^30 bytes); ``what`` names what the need covers when that is more than this plan's
        run.  ``checks_bytes`` is the part of ``need_bytes`` that is the checks' own -- the bf16 copy their reference
        makes, one at a time, of a weight the run keeps on its stored format alone: the sentence then states it, in GiB
        to two places (a copy of one weight is a fraction of one).  Raises ``NotImplementedError`` (a SKIP that names
        what to do) when the GPU has too little free, and says so when the need exceeds what the GPU has in total:
        freeing memory cannot help there."""
        need = need_bytes + self.reserve_bytes()
        free, total = self.facts.free_bytes, self.facts.total_bytes
        if checks_bytes:
            what += f", {checks_bytes / GIB:.2f} GiB of it for the checks (the reference's bf16 copy of one weight at a time, where the run keeps none)"
        if total and need > total:
            raise NotImplementedError(
                f"stack {self.stack} needs about {need / GIB:.1f} GiB on this GPU{what}; this GPU has {total / GIB:.1f} GiB in total: "
                "the run does not fit on it (CUDA_VISIBLE_DEVICES selects another GPU)"
            )
        if need > free:
            raise NotImplementedError(
                f"stack {self.stack} needs about {need / GIB:.1f} GiB on this GPU{what}; {free / GIB:.1f} GiB are free: "
                "free memory on this GPU, or select another one with CUDA_VISIBLE_DEVICES"
            )
        return f"this stack needs about {need / GIB:.1f} GiB on this GPU{what}; {free / GIB:.1f} GiB are free"


def parse_overrides(pairs: Sequence[str]) -> Dict[str, str]:
    """``["BLOCK=KEY", ...]`` of ``--provider`` as a dict; a malformed pair is a ``ValueError``."""
    out = {}
    for pair in pairs:
        block, sep, key = pair.partition("=")
        if not sep or not block or not key:
            raise ValueError(f"--provider takes BLOCK=KEY; got {pair!r}")
        out[block] = key
    return out


def select_row(
    block: str, rows: Sequence[ProviderRow], facts: DeviceFacts, sequences: int, tokens: int, stack: str, key: Optional[str] = None, model=None
) -> ProviderRow:
    """The row that serves ``block`` in this run: the first match of the stack's rows, or of the rows with ``key``.  A row
    that names layer widths matches the ``model`` that has them."""
    if key is not None:
        candidates = [r for r in rows if r.key == key]
        if not candidates:
            raise ValueError(f"--provider {block}={key}: not a declared row of this block; declared: {', '.join(sorted({r.key for r in rows}))}")
    else:
        candidates = [r for r in rows if stack in r.stacks]
        if not candidates:
            raise NotImplementedError(f"block {block} has no provider row in stack {stack}")
    for row in candidates:
        if row.serves_class(facts.cc) and row.serves_model(model) and row.serves_run(sequences, tokens):
            return row
    first = candidates[0]
    if tokens > 1:  # a step of several tokens per sequence: a decline is named by the block's rows for such a step, where it has some
        first = next((r for r in candidates if r.tokens[0] > 1), first)
    of_class = [r for r in candidates if r.serves_class(facts.cc)]
    if not of_class:
        raise NotImplementedError(f"{entry_name(first.key)} serves {first.classes_text()}; this device is {facts.cc[0]}.{facts.cc[1]}")
    if not any(r.serves_model(model) for r in of_class):
        narrow = of_class[0]
        has = f"this model has {narrow.widths_text(model)}" if model is not None else "this plan names no model"
        raise NotImplementedError(f"{entry_name(narrow.key)} serves {narrow.widths_text()}; {has}")
    if tokens > 1:  # ... by the row of the run's tokens count, else by the last of them
        several = [r for r in of_class if r.serves_model(model) and r.tokens[0] > 1]
        first = next((r for r in several if _inside(r.tokens, tokens)), several[-1] if several else first)
    raise NotImplementedError(f"{entry_name(first.key)} serves {first.serves}; this run has {sequences} sequences x {tokens} tokens per sequence")


def entry_name(key: str) -> str:
    from .providers import ENTRY_NAMES

    return ENTRY_NAMES.get(key, key)


def plan(
    provider_table: Mapping[str, Sequence[ProviderRow]],
    regimes: Sequence[Regime],
    device: Union[DeviceFacts, str, object],
    sequences: int,
    tokens: int = 1,
    stack: str = "fe-cublas",
    overrides: Optional[Mapping[str, str]] = None,
    model=None,
    options: Optional[Mapping[str, object]] = None,
    source: str = "the family's configs.py",
    factories: Optional[Mapping[str, Callable]] = None,
) -> Plan:
    """Resolve the provider table for (GPU class, sequences, tokens per sequence, stack) and the model's layer widths, and
    build the blocks.

    ``device`` is a :class:`DeviceFacts` or a device to probe.  ``overrides`` maps a block to the key of one of its declared rows.
    ``model`` and ``options`` are handed to the block factories (geometry, norm epsilon, the reference precision).  Raises
    ``NotImplementedError`` (a SKIP) when a selected row declines and ``ValueError`` for a contract violation."""
    from .providers import KEYS, factories as registry

    facts = device if isinstance(device, DeviceFacts) else DeviceFacts.probe(device)
    known_stacks = {s for rows in provider_table.values() for r in rows for s in r.stacks} - {"selectable"}
    if stack not in known_stacks:
        raise ValueError(f"unknown stack {stack!r}; this family declares: {', '.join(sorted(known_stacks))}")
    overrides = dict(overrides or {})
    for block in overrides:
        if block not in provider_table:
            raise ValueError(f"--provider {block}=...: no such block; blocks: {', '.join(provider_table)}")
    build = dict(registry() if factories is None else factories)
    blocks, selected = {}, {}
    for block, rows in provider_table.items():
        for r in rows:
            if r.key not in KEYS:
                raise ValueError(f"block {block}: provider key {r.key!r} is not in the closed list of provider keys")
        row = select_row(block, rows, facts, sequences, tokens, stack, overrides.get(block), model)
        built = build[row.key](row, model, facts, dict(options or {}))
        built.check_support(sequences, tokens)
        if row.passed:  # the row serves some call sites of the block: each other site is the next row's that serves the run
            parts, waiting, last = {site: (row, built) for site in row.served}, list(row.passed), row
            while waiting:
                behind = list(rows)[next(index for index, r in enumerate(rows) if r is last) + 1 :]
                if not any(stack in r.stacks for r in behind):  # a site handed on to nothing: no decline of a GPU or a run, a wrong table
                    raise ValueError(
                        f"provider row {last.key}: it hands the call site {waiting[0]!r} of {block} on, and no row of stack {stack} stands behind it"
                    )
                last = select_row(block, behind, facts, sequences, tokens, stack, None, model)
                part = build[last.key](last, model, facts, dict(options or {}))
                part.check_support(sequences, tokens)
                unnamed = [site for site in waiting if last.sites and site not in dict(last.sites)]
                if unnamed:
                    raise ValueError(f"provider row {last.key}: no form for the call site {unnamed[0]!r}, which the row of {row.key} above it hands on")
                parts.update({site: (last, part) for site in waiting if site not in last.passed})
                waiting = [site for site in waiting if site in last.passed]
            built = BySite(block, [(site, parts[site][0], parts[site][1]) for site, _ in row.sites])
        blocks[block], selected[block] = built, row
    regime = next((r for r in regimes if r.serves(facts.cc, sequences)), None)
    calls = dict(model.calls_text()) if model is not None else {}
    named = getattr(model, "sites", None)
    if callable(named):  # the calls of a step at each call site: what a table prints for a part of a block several rows serve
        at = {}
        for (block, _), site in zip(model.wiring(), named()):
            at[(block, site)] = at.get((block, site), 0) + 1
        for block, built in blocks.items():
            calls.update({scoped(block, site): str(at.get((block, site), 0)) for site, _, _ in getattr(built, "parts", ())})
    return Plan(blocks, selected, calls, facts, sequences, tokens, stack, regime, source, tuple(overrides))
