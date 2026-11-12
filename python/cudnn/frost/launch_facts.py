# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch facts of the direct-call FROST entries: what a caller needs to tell WHICH kernel an entry launches for a row count, and in
what shape, without reading a config out of a kernel name.

A CuTe DSL kernel symbol is the kernel's Python name followed by the text of its compile-time arguments, cut at a fixed length.  Two
forms of one kernel can therefore share a symbol (the field that tells them apart sits behind the cut, or is not a kernel argument at
all), and a symbol alone never proves a form.  So an entry states three read-only facts per served row count, under ONE naming on every
entry.  ``<rows>`` below is the row count as the entry's own ``cfg_for`` names it -- ``(rows)``, or the step shape ``(requests, tokens)``
on an entry whose launch depends on both -- with EVERY argument required: where ``cfg_for`` lets an argument default (one config can
stand behind every tokens count), the facts do not, because the launch differs.  A fact answers for the step it is given, never for an
assumed one:

* ``kernel_symbols_for(<rows>)`` -- the symbols of the kernels the entry has COMPILED for that row count: a tuple without repeats, in
  compile order; empty until a form of that row count is compiled; more than one element only when one row count was compiled for
  several operand layouts whose symbols differ.  Read from the compiled objects (:func:`kernel_symbols`), never rebuilt from a config.
* ``form_for(<rows>)`` -- the form as plain data (:func:`plain_form`): the fields of the config the row count compiles with
  (``cfg_for``) under the config's own field names, a nested config as a nested ``dict``, plus the facts of the entry's compile key that
  the config does not hold (the row count where the kernel is specialised on it, the tokens per request, the norm epsilon).  The operand
  layout a compile is also keyed on (row strides, parameter dtypes) is a fact of a call's operands, not of a row count, and is not part
  of it.  A fresh ``dict`` on every call: comparable field by field, serialisable as it is.
* ``launch_shape_for(<rows>)`` -- :class:`LaunchShape` ``(grid, block, shared_memory)``: what the launch of that row count asks of the
  device, from the entry's own configuration arithmetic.  A profiler's kernel record of the launch (grid, block, static plus dynamic
  shared memory per thread block) reads the same three values; every entry's device test holds that equality on compiled forms.

The three are answers about a plan: none of them compiles, launches, allocates or reads device memory, and none changes what an entry
compiles or launches.  A row count the entry does not serve is the entry's own typed decline.

Importable without torch and without the CuTe DSL (python/cudnn/AGENTS.md, the import-time rules)."""

from dataclasses import fields, is_dataclass
from typing import Any, Dict, Iterable, Mapping, NamedTuple, Tuple

_SCALARS = (bool, int, float, str, type(None))


class LaunchShape(NamedTuple):
    """What one launch asks of the device.  ``grid`` and ``block`` are the ``(x, y, z)`` extents of the launch in thread blocks and in
    threads; ``shared_memory`` is the bytes of shared memory one thread block of the launch holds -- the kernel's static allocations plus
    what the launch reserves -- which is the figure a profiler's kernel record reports.  A plain tuple of plain values: it compares equal
    to ``((gx, gy, gz), (bx, by, bz), shared_memory)`` and serialises as that."""

    grid: Tuple[int, int, int]
    block: Tuple[int, int, int]
    shared_memory: int


def _plain(value, where: str):
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name), f"{where}.{f.name}") for f in fields(value)}
    if isinstance(value, _SCALARS):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_plain(v, f"{where}[{i}]") for i, v in enumerate(value))
    raise TypeError(f"launch facts: {where} is a {type(value).__name__}; a form holds bool, int, float, str and None values, tuples and nested configs")


def plain_form(cfg, **entry_facts) -> Dict[str, Any]:
    """A config as plain data: one key per field of the dataclass ``cfg`` under the field's own name, a nested dataclass as a nested
    ``dict``, a tuple or list as a tuple, then ``entry_facts`` (the facts of the entry's compile key the config does not hold) appended
    under their names.  Derived properties of a config are not fields and are not part of it.  ``TypeError`` for a ``cfg`` that is not a
    dataclass instance or a value that is not plain data, ``ValueError`` for an entry fact that would replace a config field -- both
    programming errors of the entry, never a caller's."""
    if not is_dataclass(cfg) or isinstance(cfg, type):
        raise TypeError(f"launch facts: a form is built from a config dataclass instance, got {type(cfg).__name__}")
    form = _plain(cfg, type(cfg).__name__)
    for name, value in entry_facts.items():
        if name in form:
            raise ValueError(f"launch facts: the entry fact {name} would replace the config field of that name")
        form[name] = _plain(value, name)
    return form


def kernel_symbols(compiled: Iterable[Any], what: str) -> Tuple[str, ...]:
    """The kernel symbols of compiled CuTe DSL host functions, without repeats, in the order given (an entry passes the compiled objects
    of one row count in compile order).  A compiled host function carries ``kernel_info``, the mapping the DSL loads its kernels by:
    its keys are the symbols of the kernels that function launches, in launch order.  They are the names a profiler reports for the
    launches.  ``NotImplementedError`` naming ``what`` and the installed DSL when a compiled object has no such mapping (a DSL whose
    compiled functions do not expose their kernels): the symbols are then not available from this entry, and nothing is guessed."""
    symbols = []
    for fn in compiled:
        info = getattr(fn, "kernel_info", None)
        if not isinstance(info, Mapping) or not all(isinstance(symbol, str) for symbol in info):
            from cudnn.frost.buffers import cutedsl_state

            _, version = cutedsl_state()
            found = " ".join(version) if version else "unknown version"
            raise NotImplementedError(
                f"{what}: the installed CuTe DSL ({found}) does not expose the kernel symbols of a compiled function (its kernel_info "
                f"mapping); the symbols of compiled forms are not available on it"
            )
        symbols.extend(info)
    return tuple(dict.fromkeys(symbols))
