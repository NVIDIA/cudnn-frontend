"""Stage 0: Log reading and JSON context entry extraction."""

import json
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


def iter_context_entries(lines: Iterable[str]) -> Iterable[Tuple[str, dict]]:
    """Extract JSON context entries from log lines.

    Yields:
        Tuple of (raw_line, parsed_json_payload)
    """
    for line in lines:
        if '"context"' not in line:
            continue
        stripped = line.strip()
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        yield stripped, payload
