# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The profiler's record of a kernel launch, for the device cells that compare an entry's launch facts with a launch.

An entry states what it launches for a row count (``cudnn.frost.launch_facts``: ``kernel_symbols_for`` / ``form_for`` /
``launch_shape_for``).  The independent reading of the same launch is the profiler's kernel record: ``torch.profiler`` (CUPTI) names the
kernel and, in its exported trace, carries the launch's grid, block and shared memory (static plus dynamic, per thread block).
:func:`kernel_records` returns those records for one call; :func:`assert_facts_equal_the_launch` is the one comparison every entry's
device cell makes, so the four cells cannot drift apart.

A non-test module (no ``test_`` prefix): imported by name from the entry suites."""

import json
import os
import tempfile
from typing import Callable, List, NamedTuple, Optional, Tuple

import pytest
import torch


class KernelRecord(NamedTuple):
    """One kernel launch as the profiler recorded it; a field the trace does not carry is ``None``."""

    name: str
    grid: Optional[Tuple[int, int, int]]
    block: Optional[Tuple[int, int, int]]
    shared_memory: Optional[int]


def _triple(value) -> Optional[Tuple[int, int, int]]:
    return tuple(int(v) for v in value) if isinstance(value, (list, tuple)) and len(value) == 3 else None


def kernel_records(launch: Callable[[], object]) -> List[KernelRecord]:
    """The kernel records of ONE call of ``launch`` under ``torch.profiler``, in launch order, after one warm call (which also compiles
    on first use).  Memory copies and fills are not kernels and are not returned.  An empty list when the profiler recorded no kernel
    (a node without CUPTI): the caller reports that as a skip, never as a pass."""
    from torch.profiler import ProfilerActivity, profile

    launch()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        launch()
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "trace.json")
        prof.export_chrome_trace(path)
        with open(path) as f:
            trace = json.load(f)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    kernels = sorted((e for e in events if e.get("cat") == "kernel"), key=lambda e: e.get("ts", 0))
    records = []
    for event in kernels:
        args = event.get("args", {})
        smem = args.get("shared memory")
        records.append(KernelRecord(event["name"], _triple(args.get("grid")), _triple(args.get("block")), None if smem is None else int(smem)))
    return records


def assert_facts_equal_the_launch(entry, launch: Callable[[], object], *rows, what: str = "") -> KernelRecord:
    """One launch of ``entry`` at the row count ``rows`` (the argument list of its ``cfg_for``) is ONE kernel whose profiler record equals
    the entry's launch facts: the name is the entry's symbol of the compiled form, and grid, block and shared memory are its launch
    shape.  ``launch`` is a zero-argument call of the entry on device operands.  Skips, naming the reason, where the profiler records no
    kernel or no launch arguments (the symbol is still compared in the second case); returns the record."""
    label = f"{what or type(entry).__name__} at {rows}"
    records = kernel_records(launch)
    if not records:
        pytest.skip(f"{label}: torch.profiler recorded no kernel launch on this node (no CUPTI): the launch facts were not compared with a launch")
    assert len(records) == 1, f"{label}: one call launched {len(records)} kernels: {[r.name for r in records]}"
    record = records[0]
    symbols = entry.kernel_symbols_for(*rows)
    assert symbols == (record.name,), f"{label}: the entry names {symbols}, the profiler recorded {record.name!r}"
    if record.grid is None or record.block is None or record.shared_memory is None:
        pytest.skip(f"{label}: the profiler's kernel record carries no grid / block / shared memory here ({record}); the symbol matched")
    shape = entry.launch_shape_for(*rows)
    assert (record.grid, record.block, record.shared_memory) == shape, f"{label}: the entry states {shape}, the profiler recorded {record}"
    return record
