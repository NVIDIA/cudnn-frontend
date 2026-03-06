"""Stage 1: Extract and annotate SDPA backward config from JSON payload."""

import json
from collections import OrderedDict
from typing import Optional

from . import utils


def build_cfg(raw_line: str, payload: dict, seed: Optional[int] = None) -> dict:
    """Build test configuration from JSON payload for SDPA backward.

    TODO: Implement SDPA backward config extraction:
    - Find SDPA_BWD or SDPA_FP8_BWD node
    - Extract backward tensors: dO (grad output), dQ, dK, dV (grad inputs)
    - Extract forward outputs that backward needs: O, Stats, etc.
    - Handle backward-specific fields: is_deterministic
    - Compute same geometry as forward: batches, d_qk, d_v, s_q, s_kv, heads
    """
    raise NotImplementedError("SDPA backward support not yet implemented")


def extract_seq_and_ragged(payload: dict, seed: int) -> dict:
    """Extract sequence lengths and ragged offsets for backward.

    TODO: Same as forward, but look for SDPA_BWD node instead.
    """
    raise NotImplementedError("SDPA backward support not yet implemented")


def extract_and_annotate(raw_line: str, payload: dict) -> dict:
    """Phase 1: Extract config and annotate with repro metadata for backward."""
    seed = utils.sha1_seed(raw_line)
    phase1_json = json.loads(json.dumps(payload))
    phase1_json["repro_metadata"] = extract_seq_and_ragged(phase1_json, seed)
    return phase1_json
