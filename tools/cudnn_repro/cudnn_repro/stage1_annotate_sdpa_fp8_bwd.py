"""Stage 1: Extract and annotate SDPA FP8 backward config from JSON payload."""

import json
from collections import OrderedDict
from typing import Optional

from . import utils


def _find_node(payload: dict) -> dict:
    node = utils.node_by_tag(payload, "SDPA_FP8_BWD")
    if node is None:
        raise ValueError("SDPA FP8 backward node not found in log")
    return node


def build_cfg(raw_line: str, payload: dict, seed: Optional[int] = None) -> dict:
    """Build FP8 backward test configuration from JSON payload."""
    node = _find_node(payload)
    if utils.is_mxfp8_payload(payload, node):
        raise NotImplementedError("MXFP8 repro is not yet implemented")

    tensors = payload.get("tensors", {})
    node_name = node.get("name")
    inputs = node.get("inputs", {})
    outputs = node.get("outputs", {})

    q_entry = utils.tensor_entry(tensors, node_name, "Q", inputs.get("Q"))
    k_entry = utils.tensor_entry(tensors, node_name, "K", inputs.get("K"))
    v_entry = utils.tensor_entry(tensors, node_name, "V", inputs.get("V"))
    o_entry = utils.tensor_entry(tensors, node_name, "O", inputs.get("O"))
    stats_entry = utils.tensor_entry(tensors, node_name, "Stats", inputs.get("Stats"))
    dq_entry = utils.tensor_entry(tensors, node_name, "dQ", outputs.get("dQ"))
    dk_entry = utils.tensor_entry(tensors, node_name, "dK", outputs.get("dK"))
    dv_entry = utils.tensor_entry(tensors, node_name, "dV", outputs.get("dV"))
    seq_q_entry = utils.tensor_entry(tensors, node_name, "SEQ_LEN_Q", inputs.get("SEQ_LEN_Q"))
    seq_kv_entry = utils.tensor_entry(tensors, node_name, "SEQ_LEN_KV", inputs.get("SEQ_LEN_KV"))
    page_table_k_entry = utils.tensor_entry(tensors, node_name, "Page_table_K", inputs.get("Page_table_K"))

    output_dtypes = {dtype for dtype in (utils.tensor_dtype(dq_entry), utils.tensor_dtype(dk_entry), utils.tensor_dtype(dv_entry)) if dtype is not None}
    if len(output_dtypes) > 1:
        raise ValueError(f"Inconsistent FP8 backward output dtypes: {sorted(output_dtypes)}")

    shape_q = utils.shape(q_entry)
    shape_k = utils.shape(k_entry)
    shape_v = utils.shape(v_entry)
    shape_o = utils.shape(o_entry)
    shape_stats = utils.shape(stats_entry)

    stride_q = utils.stride(q_entry)
    stride_k = utils.stride(k_entry)
    stride_v = utils.stride(v_entry)
    stride_o = utils.stride(o_entry)
    stride_stats = utils.stride(stats_entry)
    shape_k, stride_k = utils.normalize_k_layout(shape_k, stride_k)

    seq_len_q = utils.seq_len(seq_q_entry)
    seq_len_kv = utils.seq_len(seq_kv_entry)

    is_paged = any(label.startswith("Page_table_") or "PAGED_ATTENTION" in label for label in inputs)
    repro_metadata = payload.get("repro_metadata", {})
    ragged_tensor_names = set(repro_metadata.get("ragged_tensor_names", []))
    is_ragged = any(
        entry is not None and (
            utils.parse_optional_int(entry.get("ragged_offset_uid")) is not None or entry.get("name") in ragged_tensor_names
        )
        for entry in (q_entry, k_entry, v_entry, o_entry, dq_entry, dk_entry, dv_entry)
    )

    batches = shape_q[0] if shape_q else None
    h_q = shape_q[1] if shape_q else None
    s_q = shape_q[2] if shape_q else None
    d_qk = shape_q[3] if shape_q else None
    h_k = shape_k[1] if shape_k else None
    h_v = shape_v[1] if shape_v else None
    d_v = shape_o[3] if shape_o else (shape_v[3] if shape_v else None)
    s_kv = max(seq_len_kv) if is_paged and seq_len_kv else None
    if s_kv is None:
        s_kv = shape_v[2] if shape_v else (shape_k[2] if shape_k else None)

    diag_align_map = {"TOP_LEFT": 0, "BOTTOM_RIGHT": 1}
    diag_align = diag_align_map.get(node.get("diagonal_alignment", "TOP_LEFT"), 0)
    dropout_prob = utils.parse_hex_float(node.get("dropout_probability")) or 0.0
    block_size = utils.infer_block_size(page_table_k_entry, seq_len_kv, k_entry) if is_paged else None

    cfg = OrderedDict()
    cfg["data_type"] = utils.torch_dtype(payload.get("context", {}).get("io_data_type"))
    cfg["output_type"] = next(iter(output_dtypes), None)
    cfg["rng_data_seed"] = seed if seed is not None else utils.sha1_seed(raw_line)
    cfg["is_alibi"] = node.get("alibi_mask")
    cfg["is_infer"] = False
    cfg["is_paged"] = is_paged
    cfg["is_bias"] = utils.bool_from_inputs(inputs, "BIAS")
    cfg["is_block_mask"] = utils.bool_from_inputs(inputs, "BLOCK_MASK")
    cfg["is_padding"] = node.get("padding_mask") or bool(seq_len_q or seq_len_kv)
    cfg["is_ragged"] = is_ragged
    cfg["is_dropout"] = dropout_prob > 0.0
    cfg["is_determin"] = bool(node.get("is_deterministic_algorithm", False))
    cfg["is_mxfp8"] = False
    cfg["with_score_max"] = "Max" in outputs
    cfg["with_score_sum_exp"] = "Sum_exp" in outputs
    cfg["with_sink_token"] = "SINK_TOKEN" in inputs or "DSINK_TOKEN" in outputs

    left_bound = utils.parse_optional_int(node.get("left_bound"))
    right_bound = utils.parse_optional_int(node.get("right_bound"))
    if right_bound is None and node.get("causal_mask", False):
        right_bound = 0
        diag_align = diag_align_map["TOP_LEFT"]
    if right_bound is None and node.get("causal_mask_bottom_right", False):
        right_bound = 0
        diag_align = diag_align_map["BOTTOM_RIGHT"]

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
    cfg["shape_k"] = (batches, h_k, s_kv, d_qk) if None not in (batches, h_k, s_kv, d_qk) else shape_k
    cfg["stride_k"] = None if is_paged else stride_k
    cfg["shape_v"] = (batches, h_v, s_kv, d_v) if None not in (batches, h_v, s_kv, d_v) else shape_v
    cfg["stride_v"] = None if is_paged else stride_v
    cfg["shape_o"] = shape_o
    cfg["stride_o"] = stride_o
    cfg["shape_stats"] = shape_stats
    cfg["stride_stats"] = stride_stats
    if cfg["is_padding"] and batches:
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
    """Extract sequence lengths and ragged offsets from an FP8 backward payload."""
    node = _find_node(payload)
    tensors = payload.get("tensors", {})
    node_name = node.get("name")
    inputs = node.get("inputs", {})
    return {
        "seq_len_q": utils.seq_len(utils.tensor_entry(tensors, node_name, "SEQ_LEN_Q", inputs.get("SEQ_LEN_Q"))),
        "seq_len_kv": utils.seq_len(utils.tensor_entry(tensors, node_name, "SEQ_LEN_KV", inputs.get("SEQ_LEN_KV"))),
        "ragged_offset_q": utils.seq_len(utils.tensor_entry(tensors, node_name, "RAGGED_OFFSET_Q", inputs.get("RAGGED_OFFSET_Q"))),
        "ragged_offset_kv": utils.seq_len(utils.tensor_entry(tensors, node_name, "RAGGED_OFFSET_KV", inputs.get("RAGGED_OFFSET_KV"))),
        "rng_data_seed": seed,
    }


def extract_and_annotate(raw_line: str, payload: dict, full_log_text: Optional[str] = None) -> dict:
    """Phase 1: Extract config and annotate with repro metadata for FP8 backward."""
    seed = utils.sha1_seed(raw_line)
    phase1_json = json.loads(json.dumps(payload))
    phase1_json["repro_metadata"] = extract_seq_and_ragged(phase1_json, seed)
    phase1_json["repro_metadata"]["ragged_tensor_names"] = utils.parse_ragged_tensor_names(full_log_text)
    return phase1_json
