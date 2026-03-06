"""Stage 1 router: Routes to appropriate SDPA annotator based on operation type."""

from . import stage1_annotate_sdpa_fwd as sdpa_fwd
from . import stage1_annotate_sdpa_bwd as sdpa_bwd


def detect_operation_type(payload: dict) -> str:
    """Detect the operation type from the JSON payload."""
    for node in payload.get("nodes", []):
        tag = node.get("tag", "")
        if tag == "SDPA_FWD":
            return "fwd"
        if tag in ("SDPA_BWD", "SDPA_FP8_BWD"):
            return "bwd"
    return "fwd"


def extract_and_annotate(raw_line: str, payload: dict) -> dict:
    """Route to appropriate annotator based on operation type."""
    op_type = detect_operation_type(payload)
    if op_type == "fwd":
        return sdpa_fwd.extract_and_annotate(raw_line, payload)
    else:
        return sdpa_bwd.extract_and_annotate(raw_line, payload)


def build_cfg(raw_line: str, payload: dict, seed=None) -> dict:
    """Route to appropriate config builder based on operation type."""
    op_type = detect_operation_type(payload)
    if op_type == "fwd":
        return sdpa_fwd.build_cfg(raw_line, payload, seed)
    else:
        return sdpa_bwd.build_cfg(raw_line, payload, seed)
