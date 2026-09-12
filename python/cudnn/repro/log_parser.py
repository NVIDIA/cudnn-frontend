# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Log reading and JSON context entry extraction."""

import copy
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def read_lines(source: str) -> List[str]:
    """Read lines from a file or stdin."""
    if source == "-":
        return sys.stdin.read().splitlines()
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Log file '{source}' not found")
    return path.read_text().splitlines()


EXECUTE_GRAPH_PATTERN = re.compile(r"Executing gid (\d+)")
TENSOR_DUMP_PATTERN = re.compile(r"Tensor Dump uid:\s*(-?\d+).*?Data:\s*(\[.*\])")


def _parse_context_entry(line: str) -> Tuple[str, dict] | None:
    stripped = line.strip()
    if '"context"' not in stripped:
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return stripped, payload


def _parse_tensor_dump(line: str) -> Tuple[int, List[int]] | None:
    match = TENSOR_DUMP_PATTERN.search(line)
    if match is None:
        return None
    return int(match.group(1)), [int(value) for value in json.loads(match.group(2))]


def _apply_tensor_dumps(entry: Tuple[str, dict], tensor_dumps_by_uid: Dict[int, List[int]]) -> Tuple[str, dict]:
    if not tensor_dumps_by_uid:
        return entry
    raw_line, payload = entry
    payload = copy.deepcopy(payload)
    tensors = payload.get("tensors", [])
    tensor_uids = set()
    ragged_offset_uids = set()
    for tensor in tensors:
        uid = tensor.get("uid")
        if uid is not None:
            uid = int(uid)
            tensor_uids.add(uid)
        ragged_offset_uid = tensor.get("ragged_offset_uid")
        if ragged_offset_uid is not None:
            ragged_offset_uids.add(int(ragged_offset_uid))
        if uid is not None and uid in tensor_dumps_by_uid:
            tensor["pass_by_value"] = tensor_dumps_by_uid[uid]

    for uid in sorted(ragged_offset_uids - tensor_uids):
        if uid in tensor_dumps_by_uid:
            tensors.append({"uid": uid, "pass_by_value": tensor_dumps_by_uid[uid]})
    return raw_line, payload


def iter_graph_entries(lines: Iterable[str]) -> Iterable[Tuple[str, dict]]:
    """Extract serialized graph JSON entries from log lines."""
    for line in lines:
        parsed = _parse_context_entry(line)
        if parsed is not None:
            yield parsed


def iter_context_entries(lines: Iterable[str]) -> Iterable[Tuple[str, dict]]:
    """Extract execution-linked context entries from log lines.

    Prefer execution order when `Executing gid ...` markers are present.
    Fall back to serialized graph order for older logs.
    """
    graph_entries = list(iter_graph_entries(lines))
    graph_entries_by_gid = {}
    for raw_line, payload in graph_entries:
        gid = payload.get("gid")
        if gid is not None:
            graph_entries_by_gid[int(gid)] = (raw_line, payload)

    execution_entries = []
    current_entry = None
    current_dumps = {}
    for line in lines:
        match = EXECUTE_GRAPH_PATTERN.search(line)
        if match is not None:
            if current_entry is not None:
                execution_entries.append(_apply_tensor_dumps(current_entry, current_dumps))
            current_dumps = {}
            gid = int(match.group(1))
            current_entry = graph_entries_by_gid.get(gid)
            continue

        dump = _parse_tensor_dump(line)
        if dump is not None:
            uid, values = dump
            current_dumps[uid] = values

    if current_entry is not None:
        execution_entries.append(_apply_tensor_dumps(current_entry, current_dumps))

    if execution_entries:
        yield from execution_entries
        return

    yield from graph_entries
