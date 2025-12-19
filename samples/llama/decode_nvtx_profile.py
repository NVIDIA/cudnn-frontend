import os
import bisect
import sqlite3
from typing import Optional


SQL = {
    # CUDA kernel
    "kernel": """
        SELECT
            k.start,
            k.end,
            k.correlationId,
            s.value as demangledName
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        LEFT JOIN StringIds s ON k.demangledName = s.id
    """,
    # CPU calls
    "cpu": """
        SELECT
            c.start,
            c.end,
            c.correlationId
        FROM CUPTI_ACTIVITY_KIND_RUNTIME c
        ORDER BY c.start ASC, c.end DESC
    """,
    # NVTX events
    "nvtx": """
        SELECT
            n.start,
            n.end,
            n.text
        FROM NVTX_EVENTS n
        WHERE n.text IS NOT NULL AND n.end IS NOT NULL
        ORDER BY n.start ASC, n.end DESC
    """,
}


def get_data(conn) -> tuple[list[dict], list[dict], list[dict]]:
    """Query the database for kernel, memory copy, CPU call, and NVTX event data

    Args:
        conn: SQLite connection object

    Returns:
        A tuple of (kernels, cpus, events) for each is a list of dictionaries.
    """
    cur = conn.cursor()

    # Get kernels
    cur.execute(SQL["kernel"])
    columns: list[str] = [desc[0] for desc in cur.description]
    kernels = [dict(zip(columns, row)) for row in cur.fetchall()]

    # Get CPU calls
    cur.execute(SQL["cpu"])
    columns: list[str] = [desc[0] for desc in cur.description]
    cpus = [dict(zip(columns, row)) for row in cur.fetchall()]

    # Get NVTX events
    cur.execute(SQL["nvtx"])
    columns: list[str] = [desc[0] for desc in cur.description]
    events = [dict(zip(columns, row)) for row in cur.fetchall()]

    return kernels, cpus, events


def find_event(op: dict, events: list[dict], exclude: Optional[int] = None) -> int:
    """Find the narrowest NVTX event that encompasses the operation.

    Args:
        op: The operation to find the event for. Assumes to have start and end times.
        events: List of NVTX events. Each event has a start and end time and the list is sorted.
        exclude: Index of the event to exclude from the search.

    Returns:
        The index of the event in the list that encompasses the operation, or None if no event is found.

    Notes:
        The list is sorted using (start, end) as key. Therefore, scanning
        backwards should find the narrowest event.
    """
    # use bisect to find the starting point, then scan backwards
    end_idx = bisect.bisect_right(events, op["start"], key=lambda x: x["start"])
    for i in range(min(end_idx, len(events) - 1), -1, -1):
        if i == exclude:
            continue
        event = events[i]
        if op["start"] >= event["start"] and op["end"] <= event["end"]:
            return i
    return None


def aggregate_statistics(event: dict, events: list[dict], cpus: list[dict]):
    """Aggregate the statistics into the event tree"""
    if "kernel_count" in event:
        return  # this event has already been visited
    stats = {
        "kernel_count": 0,
        "kernel_time": 0,
    }
    for cpu_idx in event.get("cpu", []):
        call = cpus[cpu_idx]
        for kernel in call.get("kernel", []):
            stats["kernel_count"] += 1
            stats["kernel_time"] += kernel["end"] - kernel["start"]
    for child_idx in event.get("children", []):
        child = events[child_idx]
        aggregate_statistics(child, events, cpus)
        stats["kernel_count"] += child["kernel_count"]
        stats["kernel_time"] += child["kernel_time"]
    event.update(stats)


def analyze_cuda_profile(sqlite_file: str, top_event: str = ":fwd") -> list[dict]:
    """
    Summarize a CUDA profile SQLite database, focusing on kernel launches as aggregated by a NVTX event.

    Args:
        sqlite_file: Path to the SQLite database file
        top_event: The topmost level NVTX event to analyze. Defaults to ":fwd".

    Returns:
        List of dictionaries of each NVTX events reporting the statistics of
        kernel launches.
    """
    if not os.path.exists(sqlite_file):
        print(f"Error: File {sqlite_file} does not exist.")
        return []

    print(f"Analyzing CUDA profile from: {sqlite_file}")

    # Get kernel, memcpy, and event information
    conn = sqlite3.connect(sqlite_file)
    conn.row_factory = sqlite3.Row  # This allows accessing columns by name
    kernels, cpus, events = get_data(conn)
    conn.close()

    print(f"Found {len(kernels)} kernels, {len(cpus)} CPU calls, and {len(events)} events")

    # Look for the specified topmost level event, clip the events to within this event,
    # and keep only the events that start with ":" prefix (those we want to analyze).
    try:
        event = [e for e in events if e["text"] == top_event][0]
        events = [
            e for e in events if e["start"] >= event["start"] and e["end"] <= event["end"] and e["text"].startswith(":")
        ]
    except IndexError:
        print(f"Error: Could not find event '{top_event}'")
        return []

    # Assign kernel launches to CPU calls based on the correlation ID
    corr_lookup = {cpu["correlationId"]: idx for idx, cpu in enumerate(cpus)}
    for kernel in kernels:
        if kernel["correlationId"] not in corr_lookup:
            continue  # ignore if can't find the correlation ID
        call = cpus[corr_lookup[kernel["correlationId"]]]
        if "kernel" not in call:
            call["kernel"] = []
        call["kernel"].append(kernel)

    # Associate CPU calls to the narrowest NVTX event
    for idx, call in enumerate(cpus):
        event_idx = find_event(call, events)
        if event_idx is None:
            continue
        event = events[event_idx]
        if "cpu" not in event:
            event["cpu"] = []
        event["cpu"].append(idx)

    # Find the immediate parent event of each event
    for idx, event in enumerate(events):
        parent_idx = find_event(event, events, exclude=idx)
        if parent_idx is None:
            continue
        event["parent"] = parent_idx
        parent = events[parent_idx]
        if "children" not in parent:
            parent["children"] = []
        parent["children"].append(idx)

    # Recursively aggregate the statistics into the event tree
    for event in events:
        aggregate_statistics(event, events, cpus)

    # summarize the statistics based on the event name
    results = {}
    for event in events:
        event_name = event["text"]
        if event_name not in results:
            results[event_name] = {"nvtx event": event_name}
            results[event_name].update({n: 0.0 for n in ["num calls", "kernel count", "kernel time total"]})
        results[event_name]["num calls"] += 1
        results[event_name]["kernel count"] += event["kernel_count"]
        results[event_name]["kernel time total"] += event["kernel_time"]
    # sort results by kernel_time
    results = sorted(results.values(), key=lambda x: x["nvtx event"])
    return results


if __name__ == "__main__":
    import argparse
    import tabulate

    parser = argparse.ArgumentParser(description="Analyze CUDA profile SQLite database")
    parser.add_argument("sqlite_file", help="Path to the SQLite database file or 'default' for the default file")
    parser.add_argument(
        "--top-event",
        help="The topmost level NVTX event to analyze. Such as ':forward_backward' or ':fwd'.",
        default=":fwd",
    )
    args = parser.parse_args()

    # Analyze with filter applied
    # use :forward_backward to show timing for both forward and backward passes
    # or use :fwd to show timing for only the forward pass
    results = analyze_cuda_profile(args.sqlite_file, top_event=args.top_event)
    print(tabulate.tabulate(results, headers="keys", tablefmt="simple", floatfmt=",.0f"))
