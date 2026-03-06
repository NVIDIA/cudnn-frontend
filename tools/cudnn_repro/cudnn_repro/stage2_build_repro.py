"""Stage 2 router: Routes to appropriate repro builder based on operation type."""

from . import stage2_build_repro_sdpa_fwd as sdpa_fwd
from . import stage2_build_repro_sdpa_bwd as sdpa_bwd


def detect_operation_type(payload: dict) -> str:
    """Detect the operation type from the JSON payload."""
    for node in payload.get("nodes", []):
        tag = node.get("tag", "")
        if tag == "SDPA_FWD":
            return "fwd"
        if tag in ("SDPA_BWD", "SDPA_FP8_BWD"):
            return "bwd"
    return "fwd"


def build_repro_command(raw_line: str, stage1_json: dict) -> str:
    """Route to appropriate repro builder based on operation type."""
    op_type = detect_operation_type(stage1_json)
    if op_type == "fwd":
        return sdpa_fwd.build_repro_command(raw_line, stage1_json)
    else:
        return sdpa_bwd.build_repro_command(raw_line, stage1_json)
