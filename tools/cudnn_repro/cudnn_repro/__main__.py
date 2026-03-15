#!/usr/bin/env python3
"""Generate pytest SDPA repro commands from cuDNN Frontend logs.

This tool processes cuDNN frontend log files and generates pytest commands
to reproduce specific SDPA test cases.
"""

import argparse
import os
from pathlib import Path

from . import stage0_extract_json as stage0
from . import stage1_annotate_sdpa_fwd as stage1_fwd
from . import stage1_annotate_sdpa_bwd as stage1_bwd
from . import stage2_build_repro_sdpa_fwd as stage2_fwd
from . import stage2_build_repro_sdpa_bwd as stage2_bwd
from . import utils


def detect_operation_type(payload: dict) -> str:
    """Detect the operation type from the JSON payload.

    Returns:
        'fwd' for forward operations
        'bwd' for backward operations
    """
    for node in payload.get("nodes", []):
        tag = node.get("tag", "")
        if tag == "SDPA_FWD":
            return "fwd"
        if tag in ("SDPA_BWD", "SDPA_FP8_BWD"):
            return "bwd"
    # Default to forward if unclear
    return "fwd"


# Expose functions for backward compatibility with tests
def _iter_context_entries(lines):
    """Wrapper for stage0.iter_context_entries for test compatibility."""
    return stage0.iter_context_entries(lines)


def _build_cfg(raw_line: str, payload: dict, seed=None):
    """Wrapper for stage1 build_cfg for test compatibility (forward only)."""
    return stage1_fwd.build_cfg(raw_line, payload, seed)


def _build_command(cfg: dict) -> str:
    """Wrapper for stage2 build_command for test compatibility (forward only)."""
    return stage2_fwd.build_command(cfg)


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
        op_type = detect_operation_type(payload)

        if op_type == "fwd":
            stage1 = stage1_fwd
            stage2 = stage2_fwd
        else:  # bwd
            stage1 = stage1_bwd
            stage2 = stage2_bwd

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
