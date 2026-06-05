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
TENSOR_DUMP_PATTERN = re.compile(r"Tensor Dump tid:\s*(-?\d+).*?Data:\s*(\[.*\])")


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


def _apply_tensor_dumps(entry: Tuple[str, dict], tensor_dumps_by_tid: Dict[int, List[int]]) -> Tuple[str, dict]:
    if not tensor_dumps_by_tid:
        return entry
    raw_line, payload = entry
    payload = copy.deepcopy(payload)
    for tensor in payload.get("tensors", []):
        if tensor.get("tid") in tensor_dumps_by_tid:
            tensor["pass_by_value"] = tensor_dumps_by_tid[tensor["tid"]]
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
            tid, values = dump
            current_dumps[tid] = values

    if current_entry is not None:
        execution_entries.append(_apply_tensor_dumps(current_entry, current_dumps))

    if execution_entries:
        yield from execution_entries
        return

    yield from graph_entries
