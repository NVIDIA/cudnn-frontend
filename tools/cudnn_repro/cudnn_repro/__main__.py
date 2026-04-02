#!/usr/bin/env python3
"""Generate pytest SDPA repro commands from cuDNN Frontend logs.

This tool processes cuDNN frontend log files and generates pytest commands
to reproduce specific SDPA test cases.
"""

import argparse
import os
from pathlib import Path

from . import routing
from . import stage0_extract_json as stage0
from . import utils


# Expose functions for backward compatibility with tests
def _iter_context_entries(lines):
    """Wrapper for stage0.iter_context_entries for test compatibility."""
    return stage0.iter_context_entries(lines)


def _build_cfg(raw_line: str, payload: dict, seed=None):
    """Wrapper for stage1 build_cfg for test compatibility."""
    stage1, _ = routing.select_stage_modules(payload)
    return stage1.build_cfg(raw_line, payload, seed)


def _build_command(cfg: dict, payload: dict | None = None) -> str:
    """Wrapper for stage2 build_command for test compatibility."""
    if payload is None:
        is_bwd = cfg.get("is_infer") is False
        stage2 = routing.select_stage2_module({"nodes": [{"tag": "SDPA_BWD" if is_bwd else "SDPA_FWD"}]})
        return stage2.build_command(cfg)
    _, stage2 = routing.select_stage_modules(payload)
    return stage2.build_command(cfg)


def main() -> None:
    """Main entry point for the repro tool."""
    parser = argparse.ArgumentParser(description="Generate pytest sdpa repro command from cuDNN Frontend log.")
    parser.add_argument("logfile", help="Path to sdpa log (use '-' to read from stdin)")
    parser.add_argument("--all", action="store_true", help="Emit commands for every context entry (default: only the last one)")
    args = parser.parse_args()
    debug_repro = os.environ.get("CUDNN_DEBUG_REPRO", "0") == "1"

    # Stage 0: Read log and extract context entries
    lines = stage0.read_lines(args.logfile)
    entries = list(stage0.iter_context_entries(lines))
    if not entries:
        raise SystemExit("No context entries found in log.")

    selected = entries if args.all else [entries[-1]]
    full_log_text = "\n".join(lines)

    # Process each selected entry
    for idx, (raw_line, payload) in enumerate(selected):
        # Detect operation type and route to appropriate stage modules
        stage1, stage2 = routing.select_stage_modules(payload)

        # Stage 1: Extract and annotate
        stage1_json = stage1.extract_and_annotate(raw_line, payload, full_log_text)

        # Stage 2: Build repro command
        stage2_txt = stage2.build_repro_command(raw_line, stage1_json)
        print(stage2_txt)

        # Debug output if requested
        if debug_repro:
            suffix = "" if len(selected) == 1 else f"_{idx}"
            utils.try_write_text(Path(f"cudnn_repro_stage0{suffix}.txt"), full_log_text)
            utils.try_write_text(Path(f"cudnn_repro_stage1{suffix}.json"), utils.format_json_pretty(stage1_json))
            utils.try_write_text(Path(f"cudnn_repro_stage2{suffix}.txt"), stage2_txt)


if __name__ == "__main__":
    main()
