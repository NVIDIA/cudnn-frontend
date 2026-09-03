# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python-native validation for the SDPA op family (issue #704).

``pygraph.validate()`` classically lowers every backend-lowerable graph to C++
and runs the backend's ``validate()`` — which couples frontend-only (FROST)
engines to the installed backend's version: a graph a python engine fully
serves is rejected at validate() because the *backend* is too old to know an
attribute it will never execute. This module is the replacement for the SDPA
family: the graph-SEMANTIC subset of the classic C++ pre-validation (shape and
stride invariants, ill-formed attribute combinations), expressed on the python
IR with the classic error types and messages. It is wired in through the
manifest -- ``EngineFamily.validator`` on the frost_sdpa families resolves to
``validate_graph`` here (``cudnn._gemm_validate`` is the GEMM family's
counterpart); ``pygraph.validate()`` consults it only when the manifest also
offers a python engine for the graph.

Deliberately absent: every backend-version gate (``get_backend_version() < X``)
and device-arch gate (``prop_major == Y``) from the C++ surface. Those are
support-surface answers, not graph-validity answers — at planning time each
python engine's ``check_support()`` gives its own, and the backend gives its
own when it is lowered (declines recorded by ``backend_plan_entries()``, and
surfaced by ``plan()`` only if no engine proposes a plan).

Error-type parity with the pybind ``throw_if`` mapping:
``GRAPH_NOT_SUPPORTED`` -> ``cudnn.cudnnGraphNotSupportedError`` (callers catch
it to skip a config); ``ATTRIBUTE_NOT_SET`` / ``INVALID_VALUE`` ->
``std::invalid_argument`` -> ``ValueError``.

Import-light on purpose: only the IR types. Never import ``cudnn.sdpa`` (that
pulls torch/cutlass) or the compiled binding at module scope.
"""

from __future__ import annotations

from typing import Any, Optional

from .graph_types import NodeType

_FWD_TYPES = (NodeType.SDPA, NodeType.SDPA_FP8, NodeType.SDPA_MXFP8)
_BWD_TYPES = (NodeType.SDPA_BWD, NodeType.SDPA_FP8_BWD, NodeType.SDPA_MXFP8_BWD)

# Node types this module fully validates. pygraph.validate() may defer the C++
# lowering only for graphs whose every node is covered here.
COVERED_NODE_TYPES = frozenset(_FWD_TYPES + _BWD_TYPES)


def _not_supported(message: str) -> Exception:
    """Classic GRAPH_NOT_SUPPORTED parity: the error type callers catch to skip a config."""
    import cudnn

    return cudnn.cudnnGraphNotSupportedError(message)


def _dtype_name(t: Any) -> Optional[str]:
    """Enum name of a tensor's data type, or None when the tensor/dtype is unset."""
    dt = t.get_data_type() if t is not None else None
    return getattr(dt, "name", None)


def _tensor(node, port: str):
    """The tensor bound to ``port`` on either side of the node, or None."""
    return node.inputs.get(port) or node.outputs.get(port)


def _check_dim_stride(node, port: str, t: Any) -> None:
    """Classic parity: rank-4 dim/stride assigned, unit stride on the head dim."""
    if t is None:
        raise ValueError(f"{node.name}: required tensor {port} is missing")
    if len(t.get_dim() or ()) != 4:
        raise ValueError(f"The dim for {port} is invalid")
    if len(t.get_stride() or ()) != 4:
        raise ValueError(f"The stride for {port} is invalid")
    if t.get_stride()[3] != 1:
        raise _not_supported("The stride for the last dimension corresponding to the embedding size per head should be 1 for " + port)


def _diagonal(params: dict) -> tuple:
    """(alignment_name, left_bound, right_bound) with the pybind's spelling
    precedence: the use_causal_mask* flags override diagonal_alignment and pin
    right_bound to 0; diagonal_band_left_bound overrides sliding_window_length."""
    causal_tl = bool(params.get("use_causal_mask"))
    causal_br = bool(params.get("use_causal_mask_bottom_right"))
    if causal_br:
        alignment = "BOTTOM_RIGHT"
    elif causal_tl:
        alignment = "TOP_LEFT"
    else:
        alignment = getattr(params.get("diagonal_alignment"), "name", "TOP_LEFT")
    right = 0 if (causal_tl or causal_br) else params.get("diagonal_band_right_bound")
    left = params.get("diagonal_band_left_bound")
    if left is None:
        left = params.get("sliding_window_length")
    return alignment, left, right


def _dropout(node) -> tuple:
    """(probability | None, has_custom_mask) from the captured dropout tuple:
    (float prob, seed, offset) is probability form; an all-tensor tuple
    ((mask, scale) / (mask, scale, scale_inv)) is the custom-mask form."""
    if not node.params.get("_dropout_n"):
        return None, False
    prob = node.params.get("dropout_0")
    if isinstance(prob, (int, float)) and not isinstance(prob, bool):
        return float(prob), False
    return None, "dropout_0" in node.inputs


def _is_ragged(*tensors) -> bool:
    """Whether any given tensor carries a ragged (packed THD) offset."""
    return any(t is not None and t.ragged_offset is not None for t in tensors)


def _has(node, port: str) -> bool:
    """Whether an input port is bound."""
    return node.inputs.get(port) is not None


def validate_graph(graph) -> bool:
    """The frost_sdpa families' native validator (engines.manifest.EngineFamily.validator):
    validate every node when all of them are SDPA-family; return False without
    raising when the graph holds a node this module does not cover, so the caller
    falls back to the classic eager C++ lowering."""
    nodes = list(graph._nodes)
    if not nodes or any(n.node_type not in COVERED_NODE_TYPES for n in nodes):
        return False
    for node in nodes:
        validate_node(node)
    return True


def validate_node(node) -> None:
    """Validate one SDPA-family node, raising with classic error semantics."""
    if node.node_type in _FWD_TYPES:
        _validate_forward(node)
    elif node.node_type in _BWD_TYPES:
        _validate_backward(node)
    else:
        raise AssertionError(f"{node.node_type} is not an SDPA-family node")


def _validate_common(node, q, k, v, o, s_q, s_kv) -> None:
    """Checks shared verbatim by the forward and backward C++ surfaces."""
    params = node.params
    _, h_q, _, _ = q.get_dim()
    h_k = k.get_dim()[1]
    h_v = v.get_dim()[1]

    if (h_q % h_k != 0) or (h_q % h_v != 0):
        raise _not_supported("For group-query attention, number of heads for key and query must be a factor of number of heads for query")

    alignment, left, right = _diagonal(params)
    causal_br = alignment == "BOTTOM_RIGHT" and right is not None
    padding = bool(params.get("use_padding_mask"))
    score_mod = params.get("score_mod") is not None
    alibi = bool(params.get("use_alibi_mask"))

    if alibi and right != 0:
        raise _not_supported("When alibi mask is used, diagonal_band_right_bound needs to be set to 0.")

    bias = node.inputs.get("bias")
    if bias is not None and _dtype_name(bias) == "BOOLEAN":
        raise _not_supported("Bias mask data type cannot be boolean")

    # Padding requires a per-sequence length tensor on each side; each side
    # independently uses exactly one representation (per-batch or cumulative).
    has_seq_q, has_cu_q = _has(node, "seq_len_q"), _has(node, "cu_seq_len_q")
    has_seq_kv, has_cu_kv = _has(node, "seq_len_kv"), _has(node, "cu_seq_len_kv")
    if has_seq_q and has_cu_q:
        raise ValueError("seq_len_q and cu_seq_len_q cannot both be set.")
    if has_seq_kv and has_cu_kv:
        raise ValueError("seq_len_kv and cu_seq_len_kv cannot both be set.")
    has_any_q, has_any_kv = has_seq_q or has_cu_q, has_seq_kv or has_cu_kv
    if padding and not (has_any_q and has_any_kv):
        raise ValueError("Padding mask requires seq_len_q/seq_len_kv (or cu_seq_len_q/cu_seq_len_kv) to be set.")
    if (not padding and not score_mod) and (has_any_q or has_any_kv):
        raise ValueError("seq_len_q/seq_len_kv (or cu_seq_len_q/cu_seq_len_kv) needs to be set only if padding mask is enabled.")
    if has_any_q != has_any_kv:
        raise ValueError(
            "A Q-side and a KV-side sequence length tensor must be provided together " "(each side may independently use seq_len_* or cu_seq_len_*)."
        )

    is_ragged = _is_ragged(q, k, v, o)
    if (params.get("max_total_seq_len_q") is not None or params.get("max_total_seq_len_kv") is not None) and not is_ragged:
        raise _not_supported("max_total_seq_len_q/kv is only supported with packed (ragged) layout")

    prob, custom_mask = _dropout(node)
    if prob is not None and custom_mask:
        raise ValueError("Using both, custom dropout mask and internal-mask generation using dropout probability, is ill-formed.")
    if prob == 1.0:
        raise ValueError("Dropout probability cannot be 1 as corresponding scale wont be well formed.")
    is_dropout = prob is not None or custom_mask

    if causal_br and not padding and s_kv is not None and s_q > s_kv:
        raise _not_supported(
            "Bottom right causal mask does not support max_s_q > max_s_kv. Please virtually slice the Q tensor and " "pass it as max_s_q == max_s_kv"
        )
    if causal_br and (bias is not None or alibi or is_dropout or (is_ragged and not padding)):
        raise _not_supported(
            "Bottom right causal mask is only supported with is_bias=False, is_alibi=False, is_dropout=False. "
            "Further is_ragged==True is only allowed when padding_mask=True."
        )

    if left is not None:
        if not padding and s_kv is not None and s_q > s_kv:
            raise _not_supported("Sliding window attention is only supported with max_s_q <= max_s_kv.")
        if is_ragged and not padding:
            raise _not_supported("Left and right bounds with is_ragged==True is only allowed when padding_mask=True.")
    if right is not None and right < 0:
        raise ValueError("Right bound needs to be larger than or equal to zero")


def _validate_forward(node) -> None:
    """SDPA / SDPA_FP8 / SDPA_MXFP8 forward: the classic C++ pre-validation rules that
    do not depend on backend version or device arch."""
    params = node.params
    q, k, v = node.inputs.get("q"), node.inputs.get("k"), node.inputs.get("v")
    o = node.outputs.get("O")
    for port, t in (("Q", q), ("K", k), ("V", v), ("O", o)):
        _check_dim_stride(node, port, t)

    _, _, s_q, d_qk = q.get_dim()
    d_v = v.get_dim()[3]
    paged = _has(node, "paged_attention_k_table") or _has(node, "paged_attention_v_table")
    max_seq_kv = params.get("paged_attention_max_seq_len_kv")
    # With paged caches K/V carry container extents; the logical s_kv is the
    # explicit maximum when given, unknowable here otherwise.
    s_kv = k.get_dim()[2] if not paged else max_seq_kv

    _validate_common(node, q, k, v, o, s_q, s_kv)

    stats = node.outputs.get("Stats")
    if stats is not None and _dtype_name(stats) not in (None, "FLOAT"):
        raise _not_supported("The Stats output of sdpa must be an FP32 tensor.")

    is_ragged = _is_ragged(q, k, v, o)
    score_mod = params.get("score_mod") is not None
    if is_ragged and not (bool(params.get("use_padding_mask")) or score_mod):
        raise _not_supported("Ragged offsets are only supported with padding mask.")

    _alignment, left, right = _diagonal(params)
    if score_mod and (bool(params.get("use_alibi_mask")) or right is not None or bool(params.get("use_padding_mask")) or left is not None):
        raise _not_supported("Attention score mod enabled and hence other subgraphs are disabled.")

    if node.node_type == NodeType.SDPA:
        if (d_qk % 8 != 0) or (d_v % 8 != 0):
            raise _not_supported("hidden_dim should be multiple of 8")
        if s_q == 1 and _has(node, "sink_token"):
            raise _not_supported("decode only mode, i.e. s_q == 1, not supported with sink_token")

        has_any_q = _has(node, "seq_len_q") or _has(node, "cu_seq_len_q")
        has_any_kv = _has(node, "seq_len_kv") or _has(node, "cu_seq_len_kv")
        if paged and not (has_any_q and has_any_kv):
            raise _not_supported(
                "Paged caches can only be used in combination with padding mask and variable sequence lengths "
                "for both Q and KV (each side independently via seq_len_* or cu_seq_len_*)."
            )
        if not paged and max_seq_kv is not None:
            raise _not_supported("When not using paged attention, there is no need to explicitly set max kv sequence length.")
        if max_seq_kv is not None:
            bias = node.inputs.get("bias")
            if bias is not None and bias.get_dim()[3] != max_seq_kv:
                raise _not_supported("Value set through set_paged_attention_max_seq_len_kv is incompatible with the sequence length of the bias")
            rng_dump = node.outputs.get("rng_dump")
            if rng_dump is not None and rng_dump.get_dim()[3] != max_seq_kv:
                raise _not_supported("Value set through set_paged_attention_max_seq_len_kv is incompatible with the sequence length of the RNG_DUMP")

    if node.node_type == NodeType.SDPA_FP8:
        if node.inputs.get("bias") is not None:
            raise _not_supported("SDPA FP8 does not support bias")
        if (d_qk % 16 != 0) or (d_v % 16 != 0):
            raise _not_supported("hidden_dim should be multiple of 16")

    if node.node_type == NodeType.SDPA_MXFP8:
        _validate_mxfp8_descales(node, q, k, v, s_kv if s_kv is not None else k.get_dim()[2])


def _validate_mxfp8_descales(node, q, k, v, s_kv: int) -> None:
    """MXFP8 block-scale descale tensors: F8_128x4 reordering and batch/head dims matching
    their base operand."""
    b, h_q, _, d = q.get_dim()
    h_k, h_v = k.get_dim()[1], v.get_dim()[1]
    block_size = 32  # MXFP8 block size is fixed at 32
    d_scale = (d + block_size - 1) // block_size
    s_scale = (s_kv + block_size - 1) // block_size

    for port, h, small_axis, small_min in (
        ("descale_q", h_q, 3, d_scale),
        ("descale_k", h_k, 3, d_scale),
        ("descale_v", h_v, 2, s_scale),
    ):
        t = node.inputs.get(port)
        if t is None:
            raise ValueError(f"{node.name}: MXFP8 SDPA requires {port}")
        cap = port.title().replace("_q", "_Q").replace("_k", "_K").replace("_v", "_V")
        if _dtype_name(t) != "FP8_E8M0":
            raise ValueError(f"MXFP8 SDPA requires {cap} to have FP8_E8M0 data type")
        if getattr(t.get_reordering_type(), "name", None) != "F8_128x4":
            raise ValueError(f"MXFP8 SDPA requires {cap} to have F8_128x4 reordering")
        dim = t.get_dim()
        if dim[0] != b or dim[1] != h:
            base = port.split("_")[1].upper()
            raise ValueError(f"MXFP8 SDPA: {cap} batch/head dimensions must match {base}")
        if dim[small_axis] < small_min:
            what = "d_scale" if small_axis == 3 else "s_scale"
            raise ValueError(f"MXFP8 SDPA: {cap} {what} dimension too small (expected >= {small_min})")


def _validate_backward(node) -> None:
    """SDPA_BWD / FP8 / MXFP8 backward: the version- and arch-agnostic classic rules,
    including the s_q = s_kv = 1 rejection."""
    q, k, v = node.inputs.get("q"), node.inputs.get("k"), node.inputs.get("v")
    # sdpa_mxfp8_backward carries O as its f16 twin (port "o_f16")
    o = node.inputs.get("o") or node.inputs.get("o_f16")
    dO = node.inputs.get("dO")
    dQ, dK, dV = node.outputs.get("dQ"), node.outputs.get("dK"), node.outputs.get("dV")
    for port, t in (("Q", q), ("K", k), ("V", v), ("O", o), ("dO", dO), ("dQ", dQ), ("dK", dK), ("dV", dV)):
        _check_dim_stride(node, port, t)

    s_q = q.get_dim()[2]
    s_kv = v.get_dim()[2]
    if s_q == 1 and s_kv == 1:
        raise _not_supported("s_q = s_kv = 1 is not supported.")

    _validate_common(node, q, k, v, o, s_q, s_kv)

    params = node.params
    _alignment, left, right = _diagonal(params)
    score_mod = params.get("score_mod") is not None
    if score_mod and (bool(params.get("use_alibi_mask")) or bool(params.get("use_padding_mask")) or right is not None or left is not None):
        raise _not_supported("Attention score mod enabled and hence other subgraphs are disabled.")

    if left is not None and left <= 0:
        raise ValueError("Left bound (Sliding window length) should be greater than zero when set.")

    if node.outputs.get("dSink_token") is not None and node.inputs.get("sink_token") is None:
        raise ValueError("dSink_token requires sink_token to be provided")

    d_qk = q.get_dim()[3]
    d_v = v.get_dim()[3]
    if node.node_type == NodeType.SDPA_BWD:
        if (d_qk % 8 != 0) or (d_v % 8 != 0):
            raise _not_supported("hidden_dim should be multiple of 8")
