"""Operation detection and dispatch for repro generation."""

from . import sdpa_bwd
from . import sdpa_fp8_bwd
from . import sdpa_fp8_fwd
from . import sdpa_fwd


def detect_operation_key(payload: dict) -> str:
    """Detect the operation key from the JSON payload."""
    for node in payload.get("nodes", []):
        tag = node.get("tag", "")
        if tag in ("SDPA_FP8_FWD", "SDPA_MXFP8_FWD"):
            return "sdpa_fp8_fwd"
        if tag in ("SDPA_FP8_BWD", "SDPA_MXFP8_BWD"):
            return "sdpa_fp8_bwd"
        if tag == "SDPA_BWD":
            return "sdpa_bwd"
        if tag in ("SDPA_FWD", "SDPA"):
            return "sdpa_fwd"
    return "sdpa_fwd"


def detect_operation_type(payload: dict) -> str:
    """Return the coarse forward/backward operation type."""
    return "bwd" if detect_operation_key(payload).endswith("_bwd") else "fwd"


def select_operation(payload: dict):
    """Return the operation module for the payload."""
    return {
        "sdpa_fwd": sdpa_fwd,
        "sdpa_bwd": sdpa_bwd,
        "sdpa_fp8_fwd": sdpa_fp8_fwd,
        "sdpa_fp8_bwd": sdpa_fp8_bwd,
    }[detect_operation_key(payload)]
