"""Log reading and JSON context entry extraction."""

import json
import re
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


def read_lines(source: str) -> List[str]:
    """Read lines from a file or stdin."""
    if source == "-":
        return sys.stdin.read().splitlines()
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Log file '{source}' not found")
    return path.read_text().splitlines()


EXECUTE_GRAPH_UID_PATTERN = re.compile(r"Executing graph_uid (\d+)")


def _parse_context_entry(line: str) -> Tuple[str, dict] | None:
    stripped = line.strip()
    if '"context"' not in stripped:
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return stripped, payload


def iter_graph_entries(lines: Iterable[str]) -> Iterable[Tuple[str, dict]]:
    """Extract serialized graph JSON entries from log lines."""
    for line in lines:
        parsed = _parse_context_entry(line)
        if parsed is not None:
            yield parsed


def iter_context_entries(lines: Iterable[str]) -> Iterable[Tuple[str, dict]]:
    """Extract execution-linked context entries from log lines.

    Prefer execution order when `Executing graph_uid ...` markers are present.
    Fall back to serialized graph order for older logs.
    """
    graph_entries = list(iter_graph_entries(lines))
    graph_entries_by_uid = {}
    for raw_line, payload in graph_entries:
        graph_uid = payload.get("graph_uid")
        if graph_uid is not None:
            graph_entries_by_uid[int(graph_uid)] = (raw_line, payload)

    execution_entries = []
    for line in lines:
        match = EXECUTE_GRAPH_UID_PATTERN.search(line)
        if match is None:
            continue
        graph_uid = int(match.group(1))
        entry = graph_entries_by_uid.get(graph_uid)
        if entry is not None:
            execution_entries.append(entry)

    if execution_entries:
        yield from execution_entries
        return

    yield from graph_entries
