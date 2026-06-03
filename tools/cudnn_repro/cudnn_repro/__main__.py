#!/usr/bin/env python3
"""Generate pytest SDPA repro commands from cuDNN Frontend logs.

This tool processes cuDNN frontend log files and generates pytest commands
to reproduce specific SDPA test cases.
"""

import argparse
import os
from pathlib import Path

from . import log_parser
from . import operations
from . import repro_command
from . import utils


def main() -> None:
    """Main entry point for the repro tool."""
    parser = argparse.ArgumentParser(description="Generate pytest sdpa repro command from cuDNN Frontend log.")
    parser.add_argument("logfile", help="Path to sdpa log (use '-' to read from stdin)")
    parser.add_argument("--all", action="store_true", help="Emit commands for every context entry (default: only the last one)")
    args = parser.parse_args()
    debug_repro = os.environ.get("CUDNN_DEBUG_REPRO", "0") == "1"

    lines = log_parser.read_lines(args.logfile)
    entries = list(log_parser.iter_context_entries(lines))
    if not entries:
        raise SystemExit("No context entries found in log.")

    selected = entries if args.all else [entries[-1]]
    full_log_text = "\n".join(lines)

    for idx, (raw_line, payload) in enumerate(selected):
        operation = operations.select_operation(payload)
        annotated_payload = operation.extract_and_annotate(raw_line, payload, full_log_text)
        seed = annotated_payload.get("repro_metadata", {}).get("rng_data_seed")
        cfg = operation.build_cfg(raw_line, annotated_payload, seed)
        command = repro_command.build_pretty_command(cfg)

        print(command)

        if debug_repro:
            suffix = f"_{idx}" if args.all else ""
            utils.try_write_text(Path(f"cudnn_repro_log{suffix}.txt"), full_log_text)
            utils.try_write_text(Path(f"cudnn_repro_payload{suffix}.json"), utils.format_json_pretty(annotated_payload))
            utils.try_write_text(Path(f"cudnn_repro_command{suffix}.txt"), command)


if __name__ == "__main__":
    main()
