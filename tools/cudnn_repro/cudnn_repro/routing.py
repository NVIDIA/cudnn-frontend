"""Shared operation routing helpers for repro stages."""

from . import stage1_annotate_sdpa_bwd as stage1_bwd
from . import stage1_annotate_sdpa_fwd as stage1_fwd
from . import stage1_annotate_sdpa_fp8_bwd as stage1_fp8_bwd
from . import stage1_annotate_sdpa_fp8_fwd as stage1_fp8_fwd
from . import stage2_build_repro_sdpa_bwd as stage2_bwd
from . import stage2_build_repro_sdpa_fwd as stage2_fwd
from . import stage2_build_repro_sdpa_fp8_bwd as stage2_fp8_bwd
from . import stage2_build_repro_sdpa_fp8_fwd as stage2_fp8_fwd


def detect_operation_key(payload: dict) -> str:
    """Detect the operation key from the JSON payload."""
    for node in payload.get("nodes", []):
        tag = node.get("tag", "")
        if tag == "SDPA_FP8_FWD":
            return "sdpa_fp8_fwd"
        if tag == "SDPA_FP8_BWD":
            return "sdpa_fp8_bwd"
        if tag == "SDPA_BWD":
            return "sdpa_bwd"
        if tag in ("SDPA_FWD", "SDPA"):
            return "sdpa_fwd"
    return "sdpa_fwd"


def detect_operation_type(payload: dict) -> str:
    """Backward-compatible coarse operation type classification."""
    return "bwd" if detect_operation_key(payload).endswith("_bwd") else "fwd"


def select_stage1_module(payload: dict):
    """Return the stage-1 module for the payload."""
    return {
        "sdpa_fwd": stage1_fwd,
        "sdpa_bwd": stage1_bwd,
        "sdpa_fp8_fwd": stage1_fp8_fwd,
        "sdpa_fp8_bwd": stage1_fp8_bwd,
    }[detect_operation_key(payload)]


def select_stage2_module(payload: dict):
    """Return the stage-2 module for the payload."""
    return {
        "sdpa_fwd": stage2_fwd,
        "sdpa_bwd": stage2_bwd,
        "sdpa_fp8_fwd": stage2_fp8_fwd,
        "sdpa_fp8_bwd": stage2_fp8_bwd,
    }[detect_operation_key(payload)]


def select_stage_modules(payload: dict):
    """Return matching stage-1 and stage-2 modules for the payload."""
    return select_stage1_module(payload), select_stage2_module(payload)
