# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Who serves each block of a model: provider rows as data, the plan that resolves them before anything is loaded, and the
served-by table a sample prints before it runs.

A model family declares, per block, an ordered list of :class:`ProviderRow`.  :func:`plan` picks, for every block, the FIRST row
whose GPU classes, sequence range, tokens-per-sequence range and stack contain the run, builds its :class:`Block` and asks it
whether it serves the run.  Nothing is chosen at run time and nothing falls back on failure:

* ``NotImplementedError`` from a selection or from ``Block.check_support`` means "not served here": the sample prints
  ``[NN] SKIP <sentence>`` and exits 0;
* ``ValueError`` means the caller violated a contract (an unknown stack, a provider key that is not a declared row).

The stack is explicit (``"fe-cublas"``, ``"reference"``, later ``"native"``): a row outside the requested stack is never
selected, whatever is installed.  ``--provider BLOCK=KEY`` selects one declared row of a block for an A/B.

This module imports nothing beyond the standard library: a plan on stated device facts (:class:`DeviceFacts`) needs no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Protocol, Sequence, Tuple, Union

GIB = 1024**3


@dataclass(frozen=True)
class ProviderRow:
    """One row of a provider table (data in ``<family>/configs.py``)."""

    classes: Tuple[str, ...]  # ("cc10.0", "cc10.3") | ("cc10.x", "cc12.x") | ("any",)
    sequences: Tuple[int, Optional[int]]  # the selection key is (GPU class, sequences, tokens per sequence):
    tokens: Tuple[int, Optional[int]]  # some entries have a two-dimensional limit
    key: str  # "fe.gdn_decode_fused" | "cublas.bf16_dequantized" | ...: one of the closed list in providers.KEYS
    stack: Union[str, Tuple[str, ...]]  # "fe-cublas" | "native" | "reference" | "selectable" (only through --provider), or several
    serves: str  # what the entry admits, in rows
    measured: Tuple[str, ...] = ()  # rows with a published record; elsewhere the table prints "served, not measured here"
    fallback: str = ""  # non-empty for a reference chain or a dequantised copy: the reason printed in the table
    band: str = "same"  # the tie-out band against the reference block (tie_out.BANDS); never widened in place
    batch_invariant: Tuple[str, ...] = ()  # GPU classes on which "a sequence alone = the sequence in a batch" is a gate
    ci: bool = True  # False for a row that no test of this repository runs: the table prints "unverified in CI"

    @property
    def stacks(self) -> Tuple[str, ...]:
        return (self.stack,) if isinstance(self.stack, str) else tuple(self.stack)

    def serves_class(self, cc: Tuple[int, int]) -> bool:
        return any(c == "any" or c == f"cc{cc[0]}.{cc[1]}" or c == f"cc{cc[0]}.x" for c in self.classes)

    def serves_run(self, sequences: int, tokens: int) -> bool:
        return _inside(self.sequences, sequences) and _inside(self.tokens, tokens)

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
    kernels: Tuple[str, ...]  # kernel names in launch order: the key of a traced pass's attribution (empty until one exists)
    per_row: Tuple[int, ...]  # the operands that have one leading row per sequence
    state_args: Tuple[int, ...]  # the operands the call updates in place

    def check_support(self, sequences: int, tokens: int) -> None: ...

    def compile(self, sequences: int, tokens: int) -> None: ...

    def reserve_bytes(self) -> int: ...

    def first_launch(self) -> dict: ...

    def __call__(self, *operands): ...


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

    def __init__(self, blocks, rows, calls, facts, sequences, tokens, stack, regime, source):
        super().__init__(blocks)
        self.rows: Dict[str, ProviderRow] = rows
        self.calls: Mapping[str, str] = calls
        self.facts: DeviceFacts = facts
        self.sequences, self.tokens, self.stack = sequences, tokens, stack
        self.regime: Optional[Regime] = regime
        self.source = source

    def by_provider(self) -> str:
        """The "by provider" line: block kinds per provider; a row with a ``fallback`` reason counts as a fallback."""
        kinds: Dict[str, set] = {}
        for name, block in self.items():
            row = self.rows[name]
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
        unverified = sum(not row.ci for row in self.rows.values())
        if unverified:
            parts.append(f"unverified in CI: {unverified}")
        return "by provider: " + " | ".join(parts)

    def table(self) -> str:
        """The served-by table: decided before the load, printed before every run."""
        head = ("block", "calls", "provider", "entry", "serves", "measured here")
        lines = [head]
        for name, block in self.items():
            row = self.rows[name]
            if row.fallback:
                measured = f"fallback: {row.fallback}"
            else:
                measured = "; ".join(row.measured) if row.measured else "served, not measured here"
            if not row.ci:
                measured += "; unverified in CI"
            lines.append((name, self.calls.get(name, ""), block.provider, block.entry, row.serves, measured))
        widths = [max(len(line[i]) for line in lines) for i in range(len(head))]
        text = [
            f"served by (decided before the load from {self.source}; stack {self.stack}, {self.sequences} sequence{'s' if self.sequences > 1 else ''}, {self.facts.class_text})"
        ]
        text += ["  " + "  ".join(cell.ljust(width) for cell, width in zip(line, widths)).rstrip() for line in lines]
        text.append("  " + self.by_provider())
        if self.regime is not None:
            text.append(f"  regime: {self.regime.name} ({self.regime.note})")
        return "\n".join(text)

    def reserve_bytes(self) -> int:
        return sum(block.reserve_bytes() for block in self.values())

    def memory(self, need_bytes: int, what: str = "") -> str:
        """The memory sentence.  Raises ``NotImplementedError`` (a SKIP that names what to do) when the GPU has too little free."""
        need = need_bytes + self.reserve_bytes()
        free = self.facts.free_bytes
        if need > free:
            raise NotImplementedError(
                f"stack {self.stack} needs about {need / GIB:.1f} GB on this GPU{what}; {free / GIB:.1f} GB are free: "
                "free memory on this GPU, or select another one with CUDA_VISIBLE_DEVICES"
            )
        return f"this stack needs about {need / GIB:.1f} GB on this GPU; {free / GIB:.1f} GB are free"


def parse_overrides(pairs: Sequence[str]) -> Dict[str, str]:
    """``["BLOCK=KEY", ...]`` of ``--provider`` as a dict; a malformed pair is a ``ValueError``."""
    out = {}
    for pair in pairs:
        block, sep, key = pair.partition("=")
        if not sep or not block or not key:
            raise ValueError(f"--provider takes BLOCK=KEY; got {pair!r}")
        out[block] = key
    return out


def select_row(block: str, rows: Sequence[ProviderRow], facts: DeviceFacts, sequences: int, tokens: int, stack: str, key: Optional[str] = None) -> ProviderRow:
    """The row that serves ``block`` in this run: the first match of the stack's rows, or of the rows with ``key``."""
    if key is not None:
        candidates = [r for r in rows if r.key == key]
        if not candidates:
            raise ValueError(f"--provider {block}={key}: not a declared row of this block; declared: {', '.join(sorted({r.key for r in rows}))}")
    else:
        candidates = [r for r in rows if stack in r.stacks]
        if not candidates:
            raise NotImplementedError(f"block {block} has no provider row in stack {stack}")
    for row in candidates:
        if row.serves_class(facts.cc) and row.serves_run(sequences, tokens):
            return row
    first = candidates[0]
    if not any(r.serves_class(facts.cc) for r in candidates):
        raise NotImplementedError(f"{entry_name(first.key)} serves {first.classes_text()}; this device is {facts.cc[0]}.{facts.cc[1]}")
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
    """Resolve the provider table for (GPU class, sequences, tokens per sequence, stack) and build the blocks.

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
        row = select_row(block, rows, facts, sequences, tokens, stack, overrides.get(block))
        built = build[row.key](row, model, facts, dict(options or {}))
        built.check_support(sequences, tokens)
        blocks[block], selected[block] = built, row
    regime = next((r for r in regimes if r.serves(facts.cc, sequences)), None)
    calls = model.calls_text() if model is not None else {}
    return Plan(blocks, selected, calls, facts, sequences, tokens, stack, regime, source)
