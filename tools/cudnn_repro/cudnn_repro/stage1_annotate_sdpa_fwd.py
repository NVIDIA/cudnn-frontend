"""Stage 1: Extract and annotate SDPA forward config from JSON payload."""

import json
from collections import OrderedDict
from typing import Optional

from . import utils


def build_cfg(raw_line: str, payload: dict, seed: Optional[int] = None) -> dict:
    """Build test configuration from JSON payload."""
    node = None
    for candidate in payload.get("nodes", []):
        if candidate.get("tag") == "SDPA_FWD":
            node = candidate
            break
    if node is None and payload.get("nodes"):
        node = payload["nodes"][0]
    if node is None:
        raise ValueError("SDPA node not found in log")

    tensors = payload.get("tensors", {})
    node_name = node.get("name")
    inputs = node.get("inputs", {})
    outputs = node.get("outputs", {})

    q_entry = utils.tensor_entry(tensors, node_name, "Q", inputs.get("Q"))
    k_entry = utils.tensor_entry(tensors, node_name, "K", inputs.get("K"))
    v_entry = utils.tensor_entry(tensors, node_name, "V", inputs.get("V"))
    o_entry = utils.tensor_entry(tensors, node_name, "O", outputs.get("O"))

    seq_q_entry = utils.tensor_entry(tensors, node_name, "SEQ_LEN_Q", inputs.get("SEQ_LEN_Q"))
    seq_kv_entry = utils.tensor_entry(tensors, node_name, "SEQ_LEN_KV", inputs.get("SEQ_LEN_KV"))

    shape_q = utils.shape(q_entry)
    shape_k = utils.shape(k_entry)
    shape_v = utils.shape(v_entry)
    shape_o = utils.shape(o_entry)

    stride_q = utils.stride(q_entry)
    stride_k = utils.stride(k_entry)
    stride_v = utils.stride(v_entry)
    stride_o = utils.stride(o_entry)

    # Convert K from KT [B,H,D,S] to BHSD [B,H,S,D] when detected by strides.
    if shape_k and stride_k and len(shape_k) == 4 and len(stride_k) == 4:
        if int(stride_k[2]) == 1 and int(stride_k[3]) != 1:
            shape_k = (shape_k[0], shape_k[1], shape_k[3], shape_k[2])
            stride_k = (stride_k[0], stride_k[1], stride_k[3], stride_k[2])

    seq_len_q = utils.seq_len(seq_q_entry)
    seq_len_kv = utils.seq_len(seq_kv_entry)

    batches = next((shape[0] for shape in (shape_q, shape_k, shape_v, shape_o) if shape), None)
    d_qk = shape_q[3] if shape_q and len(shape_q) > 3 else (shape_k[3] if shape_k and len(shape_k) > 3 else None)
    d_v = shape_v[3] if shape_v and len(shape_v) > 3 else (shape_o[3] if shape_o and len(shape_o) > 3 else None)
    s_q = shape_q[2] if shape_q and len(shape_q) > 2 else (shape_o[2] if shape_o and len(shape_o) > 2 else None)
    s_kv = shape_v[2] if shape_v and len(shape_v) > 2 else (shape_k[2] if shape_k and len(shape_k) > 2 else node.get("max_seq_len_kv"))
    h_q = shape_q[1] if shape_q and len(shape_q) > 1 else (shape_o[1] if shape_o and len(shape_o) > 1 else None)
    h_k = shape_k[1] if shape_k and len(shape_k) > 1 else (shape_v[1] if shape_v and len(shape_v) > 1 else None)
    h_v = shape_v[1] if shape_v and len(shape_v) > 1 else None

    diag_align_map = {"TOP_LEFT": 0, "BOTTOM_RIGHT": 1}
    diag_align = diag_align_map.get(node.get("diagonal_alignment", "TOP_LEFT"), 0)

    dropout_prob = utils.parse_hex_float(node.get("dropout_probability")) or 0.0
    repro_metadata = payload.get("repro_metadata", {})
    ragged_tensor_names = set(repro_metadata.get("ragged_tensor_names", []))
    is_ragged = any(
        entry is not None and (utils.parse_optional_int(entry.get("ragged_offset_uid")) is not None or entry.get("name") in ragged_tensor_names)
        for entry in (q_entry, k_entry, v_entry, o_entry)
    )

    cfg = OrderedDict()
    cfg["data_type"] = utils.torch_dtype(payload.get("context", {}).get("io_data_type"))
    cfg["rng_data_seed"] = seed if seed is not None else utils.sha1_seed(raw_line)
    cfg["is_alibi"] = node.get("alibi_mask")
    cfg["is_infer"] = not node.get("generate_stats", False)
    cfg["is_paged"] = any("PAGED_ATTENTION" in key for key in inputs)
    cfg["is_bias"] = utils.bool_from_inputs(inputs, "BIAS")
    cfg["is_block_mask"] = utils.bool_from_inputs(inputs, "BLOCK_MASK")
    cfg["is_padding"] = node.get("padding_mask") or bool(seq_len_q or seq_len_kv)
    cfg["is_ragged"] = is_ragged
    cfg["is_dropout"] = dropout_prob > 0.0
    cfg["is_determin"] = None
    cfg["with_score_max"] = "Max" in outputs
    cfg["with_score_sum_exp"] = "Sum_exp" in outputs
    cfg["with_sink_token"] = "SINK_TOKEN" in inputs
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
    cfg["block_size"] = None
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

    tensors = payload.get("tensors", {})
    node_name = node.get("name")
    inputs = node.get("inputs", {})

    seq_q_entry = utils.tensor_entry(tensors, node_name, "SEQ_LEN_Q", inputs.get("SEQ_LEN_Q"))
    seq_kv_entry = utils.tensor_entry(tensors, node_name, "SEQ_LEN_KV", inputs.get("SEQ_LEN_KV"))
    ragged_q_entry = utils.tensor_entry(tensors, node_name, "RAGGED_OFFSET_Q", inputs.get("RAGGED_OFFSET_Q"))
    ragged_kv_entry = utils.tensor_entry(tensors, node_name, "RAGGED_OFFSET_KV", inputs.get("RAGGED_OFFSET_KV"))

    if ragged_q_entry is None:
        ragged_q_entry = utils.tensor_entry(tensors, node_name, "RAGGED_OFFSETS_Q", inputs.get("RAGGED_OFFSETS_Q"))
    if ragged_kv_entry is None:
        ragged_kv_entry = utils.tensor_entry(tensors, node_name, "RAGGED_OFFSETS_KV", inputs.get("RAGGED_OFFSETS_KV"))

    return {
        "seq_len_q": utils.seq_len(seq_q_entry),
        "seq_len_kv": utils.seq_len(seq_kv_entry),
        "ragged_offset_q": utils.seq_len(ragged_q_entry),
        "ragged_offset_kv": utils.seq_len(ragged_kv_entry),
        "rng_data_seed": seed,
    }


def extract_and_annotate(raw_line: str, payload: dict, full_log_text: Optional[str] = None) -> dict:
    """Phase 1: Extract config and annotate with repro metadata."""
    seed = utils.sha1_seed(raw_line)
    phase1_json = json.loads(json.dumps(payload))
    phase1_json["repro_metadata"] = extract_seq_and_ragged(phase1_json, seed)
    phase1_json["repro_metadata"]["ragged_tensor_names"] = utils.parse_ragged_tensor_names(full_log_text)
    return phase1_json
