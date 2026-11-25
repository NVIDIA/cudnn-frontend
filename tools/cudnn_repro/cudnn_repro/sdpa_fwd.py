# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract and annotate SDPA forward config from JSON payload."""

import json
from collections import OrderedDict
from typing import Optional

from . import utils


def build_cfg(raw_line: str, payload: dict, seed: Optional[int] = None) -> dict:
    """Build test configuration from JSON payload."""
    utils.validate_payload_version(payload)
    node = None
    for candidate in payload.get("nodes", []):
        if candidate.get("tag") == "SDPA_FWD":
            node = candidate
            break
    if node is None and payload.get("nodes"):
        node = payload["nodes"][0]
    if node is None:
        raise ValueError("SDPA node not found in log")

    tensors = payload.get("tensors", [])
    inputs = node.get("inputs", {})
    outputs = node.get("outputs", {})

    q_entry = utils.tensor_entry(tensors, inputs.get("Q"))
    k_entry = utils.tensor_entry(tensors, inputs.get("K"))
    v_entry = utils.tensor_entry(tensors, inputs.get("V"))
    o_entry = utils.tensor_entry(tensors, outputs.get("O"))

    shape_q = utils.shape(q_entry)
    shape_k = utils.shape(k_entry)
    shape_v = utils.shape(v_entry)
    shape_o = utils.shape(o_entry)

    stride_q = utils.stride(q_entry)
    stride_k = utils.stride(k_entry)
    stride_v = utils.stride(v_entry)
    stride_o = utils.stride(o_entry)

    paged_geometry = utils.paged_kv_geometry(node, tensors, shape_k, shape_v)
    is_paged = paged_geometry is not None
    if not is_paged:
        shape_k, stride_k = utils.normalize_k_layout(shape_k, stride_k)

    seq_len_q, seq_len_kv, cu_sides = utils.forward_seq_lengths(tensors, inputs)

    batches = next((shape[0] for shape in (shape_q, shape_k, shape_v, shape_o) if shape), None)
    d_qk = shape_q[3] if shape_q and len(shape_q) > 3 else (shape_k[3] if shape_k and len(shape_k) > 3 else None)
    d_v = shape_v[3] if shape_v and len(shape_v) > 3 else (shape_o[3] if shape_o and len(shape_o) > 3 else None)
    s_q = shape_q[2] if shape_q and len(shape_q) > 2 else (shape_o[2] if shape_o and len(shape_o) > 2 else None)
    s_kv = shape_v[2] if shape_v and len(shape_v) > 2 else (shape_k[2] if shape_k and len(shape_k) > 2 else node.get("max_seq_len_kv"))
    h_q = shape_q[1] if shape_q and len(shape_q) > 1 else (shape_o[1] if shape_o and len(shape_o) > 1 else None)
    h_k = shape_k[1] if shape_k and len(shape_k) > 1 else (shape_v[1] if shape_v and len(shape_v) > 1 else None)
    h_v = shape_v[1] if shape_v and len(shape_v) > 1 else None

    block_size = None
    if is_paged:
        block_size, s_kv = paged_geometry
        # The runner builds page pools from logical BHSD tensors.
        shape_k = (batches, h_k, s_kv, d_qk)
        shape_v = (batches, h_v, s_kv, d_v)
        stride_k = stride_v = None

    diag_align_map = {"TOP_LEFT": 0, "BOTTOM_RIGHT": 1}
    diag_align = diag_align_map.get(node.get("diagonal_alignment", "TOP_LEFT"), 0)

    dropout_prob = utils.parse_hex_float(node.get("dropout_probability")) or 0.0
    is_ragged = utils.is_ragged_payload(inputs, (q_entry, k_entry, v_entry, o_entry), payload)

    cfg = OrderedDict()
    cfg["data_type"] = utils.torch_dtype(payload.get("context", {}).get("io_data_type"))
    cfg["rng_data_seed"] = seed if seed is not None else utils.sha1_seed(raw_line)
    cfg["is_alibi"] = node.get("alibi_mask")
    cfg["is_infer"] = True if cu_sides else not node.get("generate_stats", False)
    if cu_sides:
        # The cumulative ports are forward-only. Stats are an optional forward output.
        cfg["is_cu_seq_len"] = True
        cfg["cu_seq_len_sides"] = cu_sides
        cfg["fwd_stats"] = bool(node.get("generate_stats", False))
    cfg["is_paged"] = is_paged
    cfg["is_bias"] = (is_paged and "Bias" in inputs) or utils.bool_from_inputs(inputs, "BIAS")
    cfg["is_block_mask"] = utils.bool_from_inputs(inputs, "BLOCK_MASK")
    cfg["is_padding"] = is_ragged or node.get("padding_mask") or bool(seq_len_q or seq_len_kv or cu_sides)
    cfg["is_ragged"] = is_ragged
    cfg["is_dropout"] = dropout_prob > 0.0
    cfg["is_determin"] = None
    cfg["with_score_max"] = "Max" in outputs
    cfg["with_score_sum_exp"] = "Sum_exp" in outputs
    cfg["with_sink_token"] = "SINK_TOKEN" in inputs
    if utils.has_rope(payload):
        cfg["with_rope"] = True
    if "unfuse_fma" in node:
        cfg["with_unfuse_fma"] = bool(node.get("unfuse_fma", False))
    if "stats_use_log2" in node:
        cfg["with_stats_log2"] = bool(node.get("stats_use_log2", False))
    left_bound = utils.parse_optional_int(node.get("left_bound"))
    right_bound = utils.parse_optional_int(node.get("right_bound"))
    if right_bound is None and node.get("causal_mask", False):
        right_bound = 0
        diag_align = diag_align_map.get("TOP_LEFT", 0)
    if right_bound is None and node.get("causal_mask_bottom_right", False):
        right_bound = 0
        diag_align = diag_align_map.get("BOTTOM_RIGHT", 1)
    cfg["diag_align"] = diag_align
    cfg["left_bound"] = left_bound
    cfg["right_bound"] = right_bound
    cfg["batches"] = batches
    cfg["d_qk"] = d_qk
    cfg["d_v"] = d_v
    cfg["s_q"] = s_q
    cfg["s_kv"] = s_kv
    cfg["h_q"] = h_q
    cfg["h_k"] = h_k
    cfg["h_v"] = h_v
    cfg["block_size"] = block_size
    cfg["shape_q"] = shape_q
    cfg["stride_q"] = stride_q
    cfg["shape_k"] = shape_k
    cfg["stride_k"] = stride_k
    cfg["shape_v"] = shape_v
    cfg["stride_v"] = stride_v
    cfg["shape_o"] = shape_o
    cfg["stride_o"] = stride_o
    if cfg.get("is_padding") and batches:
        if not seq_len_q:
            seq_len_q = [s_q] * batches if s_q else []
        if not seq_len_kv:
            seq_len_kv = [s_kv] * batches if s_kv else []

    cfg["seq_len_q"] = seq_len_q
    cfg["seq_len_kv"] = seq_len_kv
    cfg["dropout_prob"] = dropout_prob
    cfg["implementation"] = node.get("implementation", "AUTO")

    return cfg


def extract_seq_and_ragged(payload: dict, seed: int) -> dict:
    """Extract sequence lengths and ragged offsets from payload."""
    node = None
    for candidate in payload.get("nodes", []):
        if candidate.get("tag") == "SDPA_FWD":
            node = candidate
            break
    if node is None and payload.get("nodes"):
        node = payload["nodes"][0]
    if node is None:
        return {
            "seq_len_q": [],
            "seq_len_kv": [],
            "ragged_offset_q": [],
            "ragged_offset_kv": [],
        }

    tensors = payload.get("tensors", [])
    inputs = node.get("inputs", {})

    seq_len_q, seq_len_kv, _ = utils.forward_seq_lengths(tensors, inputs)
    ragged_q_entry = utils.tensor_entry(tensors, inputs.get("RAGGED_OFFSET_Q"))
    ragged_kv_entry = utils.tensor_entry(tensors, inputs.get("RAGGED_OFFSET_KV"))

    if ragged_q_entry is None:
        ragged_q_entry = utils.tensor_entry(tensors, inputs.get("RAGGED_OFFSETS_Q"))
    if ragged_kv_entry is None:
        ragged_kv_entry = utils.tensor_entry(tensors, inputs.get("RAGGED_OFFSETS_KV"))
    if ragged_q_entry is None:
        q_entry = utils.tensor_entry(tensors, inputs.get("Q"))
        ragged_q_entry = utils.ragged_offset_entry(tensors, q_entry)
    if ragged_kv_entry is None:
        k_entry = utils.tensor_entry(tensors, inputs.get("K"))
        v_entry = utils.tensor_entry(tensors, inputs.get("V"))
        ragged_kv_entry = utils.ragged_offset_entry(tensors, k_entry, v_entry)

    return {
        "seq_len_q": seq_len_q,
        "seq_len_kv": seq_len_kv,
        "ragged_offset_q": utils.seq_len(ragged_q_entry),
        "ragged_offset_kv": utils.seq_len(ragged_kv_entry),
        "rng_data_seed": seed,
    }


def extract_and_annotate(raw_line: str, payload: dict, full_log_text: Optional[str] = None) -> dict:
    """Phase 1: Extract config and annotate with repro metadata."""
    seed = utils.sha1_seed(raw_line)
    phase1_json = json.loads(json.dumps(payload))
    phase1_json["repro_metadata"] = extract_seq_and_ragged(phase1_json, seed)
    utils.add_ragged_tensor_names(phase1_json, full_log_text)
    return phase1_json
