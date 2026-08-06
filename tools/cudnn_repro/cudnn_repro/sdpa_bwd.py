# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract and annotate SDPA backward config from JSON payload."""

import json
from typing import Optional

from . import sdpa_fwd
from . import utils


def _find_bwd_node(payload: dict) -> dict:
    for candidate in payload.get("nodes", []):
        tag = candidate.get("tag")
        if tag == "SDPA_BWD":
            return candidate
        if tag in ("SDPA_FP8_BWD", "SDPA_MXFP8_BWD"):
            raise NotImplementedError("SDPA FP8 backward repro is not yet implemented")
    if payload.get("nodes"):
        return payload["nodes"][0]
    raise ValueError("SDPA backward node not found in log")


def _unsupported_features(payload: dict, node: dict) -> list[str]:
    inputs = node.get("inputs", {})

    unsupported = []
    if any("PAGED_ATTENTION" in key for key in inputs):
        unsupported.append("paged_attention")
    return unsupported


def _as_forward_payload(payload: dict, node: dict) -> dict:
    inputs = node.get("inputs", {})
    fwd_node = dict(node)
    fwd_node["tag"] = "SDPA_FWD"
    fwd_node["inputs"] = {
        name: inputs[name]
        for name in (
            "Q",
            "K",
            "V",
            "SEQ_LEN_Q",
            "SEQ_LEN_KV",
            "RAGGED_OFFSET_Q",
            "RAGGED_OFFSET_KV",
            "RAGGED_OFFSETS_Q",
            "RAGGED_OFFSETS_KV",
            "SINK_TOKEN",
            "BIAS",
            "BLOCK_MASK",
        )
        if name in inputs
    }
    fwd_node["outputs"] = {"O": inputs.get("O")}
    fwd_node["generate_stats"] = True

    rewritten = dict(payload)
    rewritten["nodes"] = [fwd_node]
    return rewritten


def build_cfg(raw_line: str, payload: dict, seed: Optional[int] = None) -> dict:
    """Build test configuration from JSON payload for simple SDPA backward."""
    utils.validate_payload_version(payload)
    node = _find_bwd_node(payload)
    unsupported = _unsupported_features(payload, node)
    if unsupported:
        joined = ", ".join(unsupported)
        raise NotImplementedError(f"Simple SDPA backward repro does not yet support: {joined}")

    cfg = sdpa_fwd.build_cfg(raw_line, _as_forward_payload(payload, node), seed)
    stats_entry = utils.tensor_entry(payload.get("tensors", []), node.get("inputs", {}).get("Stats"))
    cfg["is_determin"] = bool(node.get("is_deterministic_algorithm", False))
    cfg["shape_stats"] = utils.shape(stats_entry)
    cfg["stride_stats"] = utils.stride(stats_entry)
    if utils.has_rope(payload):
        cfg["with_rope"] = True
    return cfg


def extract_seq_and_ragged(payload: dict, seed: int) -> dict:
    """Extract sequence lengths and ragged offsets for simple backward."""
    node = _find_bwd_node(payload)
    return sdpa_fwd.extract_seq_and_ragged(_as_forward_payload(payload, node), seed)


def extract_and_annotate(raw_line: str, payload: dict, full_log_text: Optional[str] = None) -> dict:
    """Phase 1: Extract config and annotate with repro metadata for backward."""
    seed = utils.sha1_seed(raw_line)
    phase1_json = json.loads(json.dumps(payload))
    phase1_json["repro_metadata"] = extract_seq_and_ragged(phase1_json, seed)
    return phase1_json
