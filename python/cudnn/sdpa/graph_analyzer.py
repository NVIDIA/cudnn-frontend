# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Engine-agnostic SDPA graph analysis: ``graph.nodes`` -> :class:`SdpaGraphFacts`.

``cudnn.pygraph`` is the Python-native graph IR: it records its op DAG
directly, exposed via ``graph.nodes`` — no construction-time hooks are needed.
This module extracts *facts* (what the graph asks for) without judging
supportedness; each registered engine matches the facts against its own
capability declaration (see ``cudnn.sdpa.fwd.engines``). :func:`analyze` is the
callable the SDPA family names in ``engines/manifest.py``; PLANNING runs it once
per graph -- after the backend's layout inference has landed and the graph is
frozen -- and attaches the record, so the ranking and the engine share it rather
than each parsing the graph.

Also hosts the graph-side runtime helpers shared by every SDPA engine:
variant-pack resolution and TensorDesc construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import cudnn

_LOG = logging.getLogger(__name__)


# The facts vocabulary is cudnn.data_type, not torch.dtype. Facts are what every
# engine of a family reads, so expressing them in one framework's types would
# make the whole dispatch path require that framework; lowering converts at its
# own boundary, where a torch tensor is actually being allocated.
_TORCH_FROM_CUDNN = {
    cudnn.data_type.HALF: "float16",
    cudnn.data_type.BFLOAT16: "bfloat16",
    cudnn.data_type.FP8_E4M3: "float8_e4m3fn",
    cudnn.data_type.FP8_E5M2: "float8_e5m2",
    cudnn.data_type.FLOAT: "float32",
    cudnn.data_type.INT32: "int32",
}
_KNOWN_DTYPES = frozenset(_TORCH_FROM_CUDNN)


def _torch_cuda_device():
    """The lowering boundary's torch device — imported here, not at module level."""
    import torch

    return torch.device("cuda", torch.cuda.current_device())


def to_torch_dtype(dt):
    """cudnn.data_type -> torch.dtype, for the lowering boundary only.

    Declines rather than raising KeyError on a type this family has no mapping
    for: only Q's dtype is capability-checked, so O / Stats / a side output can
    still carry one, and tensor_desc_from_ir() runs on every bound tensor.
    """
    import torch

    if dt not in _TORCH_FROM_CUDNN:
        raise NotImplementedError(f"cudnn.sdpa: no lowering for data type {dt}")
    return getattr(torch, _TORCH_FROM_CUDNN[dt])


# BHSD logical / BSHD physical, size-1 dims wildcarded.
_REQ_STRIDE_ORDER = (3, 1, 2, 0)


def _device_props():
    """The current device as cuDNN describes it, or None without a device.

    cudnn.create_device_properties() is the backend's OWN device descriptor —
    the same object the C++ deviceless-AoT path serializes and replays. Reading
    it here instead of torch.cuda keeps the facts path framework-neutral, and
    is the step that makes a deviceless python engine possible at all: facts
    computed from a serialized descriptor need no live device.

    Cached per device id: the descriptor costs a backend call and the device
    does not change under a process.
    """
    import json

    from cudnn.frost.buffers import current_device_id

    dev = current_device_id()
    if dev is None:
        return None
    cached = _DEVICE_PROPS_CACHE.get(dev)
    if cached is None:
        try:
            cached = json.loads(cudnn.create_device_properties(dev).serialize())
        except Exception:  # noqa: BLE001 — no device / older backend: facts stay device-less
            cached = {}
        _DEVICE_PROPS_CACHE[dev] = cached
    return cached or None


_DEVICE_PROPS_CACHE: dict = {}


def _device_cc() -> Optional[tuple]:
    """Compute capability of the current CUDA device, or None without CUDA."""
    props = _device_props()
    if not props or "deviceVer" not in props:
        return None
    ver = int(props["deviceVer"])  # 1000 == SM 10.0
    return (ver // 100, (ver % 100) // 10)


def _device_sm_count() -> Optional[int]:
    """SM count of the current CUDA device, or None without CUDA."""
    props = _device_props()
    return int(props["multiProcessorCount"]) if props and "multiProcessorCount" in props else None


def _stride_order(dim: tuple, stride: tuple) -> tuple:
    return tuple(i for i, _ in sorted(enumerate(stride), key=lambda x: (x[1], dim[x[0]])))


def bshd_layout_ok(dim: tuple, stride: tuple) -> bool:
    """Strict BSHD physical stride order for a rank-4 (B, H, S, D) tensor:
    axes sorted by ascending stride must come out D, H, S, B, with size-1
    dims wildcarded. Order only — padded (oversized) strides are allowed;
    compactness and aliasing are dense_layout_ok's concern. Used by the
    facts extraction below and by engine gates that require true-BSHD views
    (e.g. THD ragged, where the kernel addresses packed rows directly)."""
    order = _stride_order(dim, stride)
    act = tuple(ax for ax in order if dim[ax] != 1)
    exp = tuple(ax for ax in _REQ_STRIDE_ORDER if dim[ax] != 1)
    return act == exp


def dense_layout_ok(dim: tuple, stride: tuple) -> bool:
    """Relaxed DENSE layout soundness for a rank-4 (B, H, S, D) tensor: the
    real requirement of the SM100 DSL lowering, which normalizes any such
    layout to the kernel's canonical BSHD-compact buffers (zero-copy when the
    tensor already is BSHD-compact, one gather/scatter copy otherwise) — so
    the B/H/S physical stride ORDER is free, and strides may be padded
    (oversized) arbitrarily. What must hold:

      * head dim (axis 3) innermost-contiguous: stride 1 (size-1 wildcarded);
      * no zero stride on a size>1 dim — a broadcast view aliases distinct
        logical elements onto one address, which is ill-defined for the O
        write-back and rejected uniformly;
      * non-overlapping: visiting dims by ascending stride, each stride must
        cover the span of every dim below it (dense == equality; padded is
        larger; SMALLER than dense would alias, so it is rejected).

    Size-1 dims are wildcards: any stride, 0 and garbage included (cuDNN's
    own harnesses corrupt them deliberately).
    """
    if dim[3] != 1 and stride[3] != 1:
        return False
    active = sorted((int(st), int(sz)) for st, sz in zip(stride, dim) if sz != 1)
    span = 1
    for st, sz in active:
        if st < span:  # st == 0 (broadcast) or overlapping / sub-dense
            return False
        span = st * sz
    return True


@dataclass(frozen=True)
class SdpaGraphFacts:
    """What a single-SDPA graph asks for. Pure description — no support judgment.

    Why extraction never judges: supportedness is per-engine knowledge (one
    kernel has paged-KV, another has sinks). The moment a shared parser
    starts rejecting, it becomes an if-ladder that must know every kernel —
    which stops scaling at the second kernel. So this record only DESCRIBES
    (``has_bias=True`` is a fact, not an error) and each engine's
    Capabilities row does the judging in ``fwd/engines.mismatch``.

    ``invalid`` is the one exception: a graph-consistency error (malformed
    regardless of which kernel would run — K/V shape mismatch, padding mask
    without seq_len_kv, ...); when set, every engine is ineligible.
    """

    invalid: Optional[str] = None

    # geometry
    b: int = 0
    h_q: int = 0
    h_kv: int = 0
    s_q: int = 0
    s_kv: int = 0
    d_qk: int = 0
    d_v: int = 0

    # dtypes / layout
    dtype: Optional[Any] = None  # Q dtype as cudnn.data_type (None if unrecognized)
    uniform_dtype: bool = True  # K/V (and O, for half) dtypes equal Q's
    bshd_layout: bool = True  # all of Q/K/V/O in BSHD-physical order
    # Relaxed dense-layout soundness: every one of Q/K/V/O has the head dim
    # innermost-contiguous (stride 1) with non-broadcast, non-overlapping
    # strides — any B/H/S order, padded strides allowed (see dense_layout_ok).
    # The actual per-tensor stride tuples stay available via q_t/k_t/v_t/o_t.
    dense_layout: bool = True
    # Backward only: (name, dim, stride) per rank-4 layout port, with the
    # K/V transposed-port rewrite undone; () on forward graphs.
    port_layouts: tuple = ()
    is_mxfp8: bool = False  # block-scale MXFP8 (FP8 Q/K/V + per-32-block E8M0 SF)
    is_fp8: bool = False  # per-tensor FP8 (FP8 Q/K/V + scalar descales)
    dtype_o: Optional[Any] = None  # O dtype as cudnn.data_type

    # masks (resolved cuDNN semantics)
    causal: bool = False  # effective causal upper bound (right band == 0)
    bottom_right: bool = False  # diagonal anchored bottom-right
    window_left: Optional[int] = None  # SWA offset W (cuDNN length - 1)
    right_band_widening: bool = False  # diagonal_band_right_bound > 0

    # requested features
    has_bias: bool = False
    has_dropout: bool = False
    has_score_mod: bool = False
    has_paged_kv: bool = False
    has_alibi: bool = False
    has_unfuse_fma: bool = False
    has_block_mask: bool = False
    has_rng_dump: bool = False
    is_backward: bool = False  # sdpa_backward() node (NodeType.SDPA_BWD)
    right_bound: Optional[int] = None  # raw resolved right band (0 == causal)
    deterministic: bool = False  # sdpa_backward(use_deterministic_algorithm=True)
    has_dbias: bool = False  # dBias output requested (backward)
    has_dsink: bool = False  # dSink_token output requested (backward)
    has_score_max: bool = False  # per-row/tile score-max side output requested
    has_score_sum_exp: bool = False  # per-row/tile sum-of-exp side output requested
    dynamic_scale: bool = False  # attn_scale passed as a tensor
    seq_q_trim: bool = False  # seq_len_q given without a padding mask

    padded: bool = False  # per-batch KV lengths present (padding mask or THD)
    thd: bool = False  # ragged (THD) Q/K/V
    # cu_seq_len_q / cu_seq_len_kv (cuDNN 9.24+): (B+1,) prefix sums, a
    # contract of their own — neither seq_len_* nor ragged_offset. A fact,
    # not a verdict: engines that don't consume the form must decline (reading
    # the graph as plain padded gave wrong output — 14.9% of O on
    # test_sdpa_mixed_seq_len_forms_L0[cu_q_brcm]).
    has_cu_seq_len: bool = False
    has_sink: bool = False
    wants_stats: bool = False

    scale: Optional[float] = None
    device_cc: Optional[tuple] = None  # current CUDA device capability
    device_sm_count: Optional[int] = None  # current CUDA device SM count

    # IR tensor refs for binding
    q_t: Any = None
    k_t: Any = None
    v_t: Any = None
    o_t: Any = None
    stats_t: Any = None
    sink_t: Any = None
    seq_kv_t: Any = None
    seq_q_t: Any = None
    # (B+1,) prefix-sum IR refs (cuDNN 9.24+ cu_seq_len form); None when the
    # graph carries the per-batch seq_len_* form on that side instead.
    cu_seq_q_t: Any = None
    cu_seq_kv_t: Any = None
    # Feature operands (bias / block-mask / score-stat outputs).
    bias_t: Any = None
    block_mask_t: Any = None
    score_max_t: Any = None
    score_sum_exp_t: Any = None
    # Backward-only refs.
    do_t: Any = None
    dq_t: Any = None
    dk_t: Any = None
    dv_t: Any = None
    dbias_t: Any = None
    dsink_t: Any = None
    # MXFP8 block-scale (descale) tensors + Amax_O output.
    sf_q_t: Any = None
    sf_k_t: Any = None
    sf_v_t: Any = None
    amax_o_t: Any = None
    # Per-tensor FP8 scalar descale tensors + Amax_S output.
    descale_q_t: Any = None
    descale_k_t: Any = None
    descale_v_t: Any = None
    scale_o_t: Any = None
    # cuDNN's Scale_S/Descale_S: they quantize P, the softmax OUTPUT. SM120
    # applies them; SM100 converts P unscaled (exact for a reciprocal pair) and
    # declines a non-reciprocal one (api_dsl._require_reciprocal_s_scales).
    descale_s_t: Any = None
    scale_s_t: Any = None
    amax_s_t: Any = None


def _single_sdpa_node(graph: "cudnn.pygraph") -> Optional[Any]:
    """The graph's sole SDPA node (forward, backward, or an FP8/MXFP8 flavor),
    or None if the graph is anything else."""
    try:
        nodes = graph.nodes
    except Exception:  # noqa: BLE001 — non-IR graph objects
        return None
    if len(nodes) != 1:
        return None
    node = nodes[0]
    if node.node_type not in (cudnn.NodeType.SDPA, cudnn.NodeType.SDPA_BWD, cudnn.NodeType.SDPA_MXFP8, cudnn.NodeType.SDPA_FP8):
        return None
    return node


def _record_from_node(node: Any) -> dict:
    """Flatten an SDPA node into one kwargs-style dict.

    ``node.params`` holds the scalar sdpa() kwargs verbatim; tensor kwargs are
    named ports in ``node.inputs`` (port name == kwarg name); O / Stats (and
    the output-style kwargs like rng_dump) live in ``node.outputs``.
    """
    rec: dict = dict(node.params)
    for port, t in node.inputs.items():
        rec.setdefault(port, t)
    # Forward: O / Stats are node outputs. Backward: o / stats are INPUT
    # ports (already folded above) — don't clobber them with the absent
    # forward output ports.
    if node.outputs.get("O") is not None:
        rec["o"] = node.outputs.get("O")
    if node.outputs.get("Stats") is not None:
        rec["stats"] = node.outputs.get("Stats")
    rec["_is_backward"] = node.node_type == cudnn.NodeType.SDPA_BWD
    if rec["_is_backward"]:
        for port in ("dQ", "dK", "dV", "dBias", "dSink_token"):
            if rec.get(port) is None:
                rec[port] = node.outputs.get(port)
    # Output-style kwargs (passed as sdpa() arguments but recorded in
    # node.outputs): fold each one in so engines see every requested output.
    # Missing one here lets an engine that never writes it pass the probe and
    # silently leave that output buffer as garbage (see score_max/score_sum_exp).
    for out_kwarg in ("rng_dump", "score_max", "score_sum_exp"):
        if rec.get(out_kwarg) is None:
            rec[out_kwarg] = node.outputs.get(out_kwarg)
    # MXFP8 / per-tensor FP8: descale_q/k/v (+ scale_o, etc. for FP8) arrive via
    # node.inputs above; Amax_S / Amax_O are outputs. The FP8 input dtype + these ports
    # distinguish these ops from plain sdpa().
    rec["_is_mxfp8"] = node.node_type == cudnn.NodeType.SDPA_MXFP8
    rec["_is_fp8"] = node.node_type == cudnn.NodeType.SDPA_FP8
    rec["amax_o"] = node.outputs.get("Amax_O")
    rec["amax_s"] = node.outputs.get("Amax_S")
    if node.params.get("_dropout_n"):
        rec["dropout"] = True  # any dropout spec (tensors or probability) is requested
    return rec


def _invalid(reason: str) -> SdpaGraphFacts:
    return SdpaGraphFacts(invalid=f"cudnn.sdpa: {reason}")


def _first_not_none(*vals):
    """First non-None value (used to fold the op family's several mask spellings)."""
    for v in vals:
        if v is not None:
            return v
    return None


def _extract_facts(rec: dict) -> SdpaGraphFacts:
    is_backward = bool(rec.get("_is_backward"))
    q, k, v, o = rec.get("q"), rec.get("k"), rec.get("v"), rec.get("o")
    if q is None or k is None or v is None or o is None:
        return _invalid("missing q/k/v/o on the sdpa node")

    rank4_ports = [("q", q), ("k", k), ("v", v), ("o", o)]
    if is_backward:
        for name in ("dO", "dQ", "dK", "dV"):
            t = rec.get(name)
            if t is None:
                return _invalid(f"missing {name} on the sdpa_backward node")
            rank4_ports.append((name, t))

    dims = {}
    strides = {}
    for name, t in rank4_ports:
        d = tuple(t.get_dim())
        if len(d) != 4:
            return _invalid(f"{name} must be rank-4 (B, H, S, D); got rank {len(d)}")
        dims[name] = d
        strides[name] = tuple(t.get_stride())
    q_dim, q_stride = dims["q"], strides["q"]
    k_dim, k_stride = dims["k"], strides["k"]
    v_dim, v_stride = dims["v"], strides["v"]
    o_dim, o_stride = dims["o"], strides["o"]

    b, h_q, s_q, d_qk = q_dim

    # ``build_operation_graph`` rewrites the BACKWARD node's K / V ports to
    # transposed (B, H, D, S) views (the K^T / V^T the lowering consumes);
    # the underlying buffer keeps the user's (B, H, S, D) shape.  Canonicalize
    # so probing works both before and after the native build.
    if is_backward:

        def _square_transposed(dim: tuple, stride: tuple) -> bool:
            """Square (S == D) rewritten views are extent-ambiguous; the stride
            order disambiguates: the transposed view keeps the buffer's unit
            stride, which lands on axis 2 instead of axis 3."""
            return dim[2] == dim[3] != 1 and stride[2] == 1 and stride[3] != 1

        if (k_dim[3] != d_qk and k_dim[2] == d_qk) or _square_transposed(k_dim, k_stride):
            k_dim = (k_dim[0], k_dim[1], k_dim[3], k_dim[2])
            k_stride = (k_stride[0], k_stride[1], k_stride[3], k_stride[2])
        _, h_kv, s_kv, _ = k_dim
        if (v_dim[2] != s_kv and v_dim[3] == s_kv) or _square_transposed(v_dim, v_stride):
            v_dim = (v_dim[0], v_dim[1], v_dim[3], v_dim[2])
            v_stride = (v_stride[0], v_stride[1], v_stride[3], v_stride[2])
        dims["k"], dims["v"] = k_dim, v_dim
        strides["k"], strides["v"] = k_stride, v_stride
    _, h_kv, s_kv, _ = k_dim
    d_v = v_dim[-1]
    if k_dim != (b, h_kv, s_kv, d_qk):
        return _invalid(f"K shape mismatch (q_dim={q_dim}, k_dim={k_dim})")
    if v_dim != (b, h_kv, s_kv, d_v):
        return _invalid(f"V shape mismatch (k_dim={k_dim}, v_dim={v_dim})")
    if o_dim != (b, h_q, s_q, d_v):
        return _invalid(f"O shape mismatch (q_dim={q_dim}, o_dim={o_dim})")
    if is_backward:
        if dims["dO"] != (b, h_q, s_q, d_v):
            return _invalid("dO shape mismatch")
        if dims["dQ"] != dims["q"] or dims["dK"] != dims["k"] or dims["dV"] != dims["v"]:
            return _invalid("dQ/dK/dV must match Q/K/V shapes")
    if any(x <= 0 for x in (b, h_q, h_kv, s_q, s_kv, d_qk, d_v)):
        return _invalid("B/H/S/D must all be > 0")
    if h_q % h_kv != 0:
        return _invalid("H_q must be divisible by H_kv (GQA / MQA)")

    is_mxfp8 = bool(rec.get("_is_mxfp8"))
    is_fp8 = bool(rec.get("_is_fp8"))
    _fp8_family = is_mxfp8 or is_fp8
    q_dtype = q.get_data_type() if q.get_data_type() in _KNOWN_DTYPES else None
    o_dtype = o.get_data_type() if o.get_data_type() in _KNOWN_DTYPES else None
    if _fp8_family:
        # FP8 in: O dtype is independent of the input; only K/V must match Q.
        uniform = all(t.get_data_type() == q_dtype for t in (k, v))
    else:
        _uniform_ports = [k, v, o] + ([rec["dO"], rec["dQ"], rec["dK"], rec["dV"]] if is_backward else [])
        uniform = all(t.get_data_type() == q_dtype for t in _uniform_ports)
    _layout_ports = [(q_dim, q_stride), (k_dim, k_stride), (v_dim, v_stride), (o_dim, o_stride)]
    if is_backward:
        _layout_ports += [(dims[name], strides[name]) for name in ("dO", "dQ", "dK", "dV")]
    bshd = all(bshd_layout_ok(d, s) for d, s in _layout_ports)
    dense_layout = all(dense_layout_ok(d, s) for d, s in _layout_ports)

    # descale_q/k/v are the block-scale SF tensors for MXFP8, or scalar per-tensor
    # descales for FP8. Both arrive on the same-named node.inputs ports.
    dsc_q, dsc_k, dsc_v = rec.get("descale_q"), rec.get("descale_k"), rec.get("descale_v")
    if _fp8_family and (dsc_q is None or dsc_k is None or dsc_v is None):
        op = "sdpa_mxfp8" if is_mxfp8 else "sdpa_fp8"
        return _invalid(f"{op} requires descale_q / descale_k / descale_v")

    # Masks: resolve cuDNN's several spellings to (causal, bottom_right, window_left).
    use_causal = bool(rec.get("use_causal_mask", False))
    use_causal_br = bool(rec.get("use_causal_mask_bottom_right", False))
    # Left window (== length; window offset is length-1). The op family uses several
    # spellings for the same knob: sdpa → sliding_window_length; sdpa_mxfp8 →
    # diagonal_band_left_bound; sdpa_fp8 → left_bound / sliding_window.
    left_bound = _first_not_none(
        rec.get("sliding_window_length"),
        rec.get("diagonal_band_left_bound"),
        rec.get("left_bound"),
        rec.get("sliding_window"),
    )
    if use_causal or use_causal_br:
        resolved_right = 0
        align_is_br = use_causal_br
    else:
        # Right band: diagonal_band_right_bound (sdpa/mxfp8) or right_bound (fp8).
        resolved_right = _first_not_none(rec.get("diagonal_band_right_bound"), rec.get("right_bound"))
        # Alignment is a property OF the diagonal band; with no band bound at all
        # there is no diagonal, so BOTTOM_RIGHT is inert — recording it as a fact
        # would make every engine reject an effectively-unmasked graph.
        has_band = resolved_right is not None or left_bound is not None
        align_is_br = has_band and rec.get("diagonal_alignment") == cudnn.diagonal_alignment.BOTTOM_RIGHT
    if resolved_right is not None and resolved_right < 0:
        return _invalid(f"diagonal-band right bound must be >= 0; got {resolved_right}")
    right_widening = resolved_right not in (0, None)
    causal = (resolved_right is not None) and not right_widening
    if left_bound is not None and left_bound < 1:
        return _invalid(f"sliding-window length must be >= 1; got {left_bound}")
    window_left = (left_bound - 1) if left_bound is not None else None

    cu_seq_q = rec.get("cu_seq_len_q")
    cu_seq_kv = rec.get("cu_seq_len_kv")
    has_cu_seq_len = cu_seq_q is not None or cu_seq_kv is not None

    # Padding / THD. Per-batch lengths arrive as seq_len_* ((B,) lengths) or
    # cu_seq_len_* ((B+1,) prefix sums, cuDNN 9.24+) per side; either form
    # satisfies the length requirement. Both-on-one-side is NOT flagged
    # invalid here (invalid means malformed-for-everyone; the backend accepts
    # it with its own precedence) — an engine that serves the cu form must
    # decline the ambiguous combination itself.
    thd = getattr(q, "ragged_offset", None) is not None
    use_padding_mask = bool(rec.get("use_padding_mask", False))
    seq_len_kv = rec.get("seq_len_kv")
    seq_len_q = rec.get("seq_len_q")
    q_lens_given = seq_len_q is not None or cu_seq_q is not None
    kv_lens_given = seq_len_kv is not None or cu_seq_kv is not None
    seq_q_trim = False
    if thd:
        if getattr(k, "ragged_offset", None) is None or getattr(v, "ragged_offset", None) is None:
            return _invalid("THD (ragged) requires ragged Q, K, and V")
        if not q_lens_given or not kv_lens_given:
            return _invalid("THD (ragged) requires seq_len_q/cu_seq_len_q and seq_len_kv/cu_seq_len_kv")
        padded = True
    else:
        if use_padding_mask and not kv_lens_given:
            return _invalid("use_padding_mask requires seq_len_kv or cu_seq_len_kv")
        seq_q_trim = q_lens_given and not use_padding_mask
        padded = use_padding_mask and kv_lens_given

    # The kernels consume per-batch lengths / prefix sums as int32 directly;
    # there is no implicit conversion anywhere on the execute path (it would
    # allocate and launch a cast kernel).
    for name, t in (("seq_len_q", seq_len_q), ("seq_len_kv", seq_len_kv), ("cu_seq_len_q", cu_seq_q), ("cu_seq_len_kv", cu_seq_kv)):
        if t is not None:
            if t.get_data_type() != cudnn.data_type.INT32:
                return _invalid(f"{name} must be int32; got {t.get_data_type()}")

    sink_token = rec.get("sink_token")
    if sink_token is not None:
        sink_dim = tuple(sink_token.get_dim())
        if sink_dim != (1, h_q, 1, 1):
            return _invalid(f"sink_token must be (1, H_q, 1, 1); got {sink_dim}")
        # The kernels consume fp32 sink logits directly; there is no implicit
        # conversion anywhere on the execute path (it would allocate and
        # launch a cast kernel).
        if sink_token.get_data_type() != cudnn.data_type.FLOAT:
            return _invalid(f"sink_token must be float32; got {sink_token.get_data_type()}")

    stats = rec.get("stats")
    if is_backward:
        wants_stats = False
        if stats is None:
            return _invalid("sdpa_backward requires the forward stats tensor")
    else:
        generate_stats = rec.get("generate_stats")
        is_inference = rec.get("is_inference")
        if generate_stats is not None:
            wants_stats = bool(generate_stats)
        elif is_inference is not None:
            wants_stats = not bool(is_inference)
        else:
            wants_stats = False
        if wants_stats and stats is None:
            return _invalid("generate_stats=True but no Stats tensor was returned")

    attn_scale = rec.get("attn_scale")
    dynamic_scale = attn_scale is not None and not isinstance(attn_scale, (int, float))
    scale = float(attn_scale) if (attn_scale is not None and not dynamic_scale) else None

    return SdpaGraphFacts(
        b=b,
        h_q=h_q,
        h_kv=h_kv,
        s_q=s_q,
        s_kv=s_kv,
        d_qk=d_qk,
        d_v=d_v,
        dtype=q_dtype,
        uniform_dtype=uniform,
        bshd_layout=bshd,
        dense_layout=dense_layout,
        port_layouts=(tuple((name, dims[name], strides[name]) for name, _ in rank4_ports) if is_backward else ()),
        is_mxfp8=is_mxfp8,
        is_fp8=is_fp8,
        dtype_o=(o_dtype if _fp8_family else q_dtype),
        causal=causal,
        bottom_right=bool(align_is_br),
        window_left=window_left,
        right_band_widening=right_widening,
        is_backward=is_backward,
        right_bound=resolved_right,
        deterministic=bool(rec.get("use_deterministic_algorithm", False)),
        has_dbias=rec.get("dBias") is not None,
        has_dsink=rec.get("dSink_token") is not None,
        has_bias=rec.get("bias") is not None,
        has_dropout=rec.get("dropout") is not None,
        has_score_mod=rec.get("fn") is not None,
        has_paged_kv=(rec.get("paged_attention_k_table") is not None or rec.get("paged_attention_v_table") is not None),
        has_alibi=bool(rec.get("use_alibi_mask")),
        has_unfuse_fma=bool(rec.get("unfuse_fma")),
        has_block_mask=rec.get("block_mask") is not None,
        has_rng_dump=rec.get("rng_dump") is not None,
        has_score_max=rec.get("score_max") is not None,
        has_score_sum_exp=rec.get("score_sum_exp") is not None,
        dynamic_scale=dynamic_scale,
        seq_q_trim=seq_q_trim,
        padded=padded,
        thd=thd,
        has_cu_seq_len=has_cu_seq_len,
        has_sink=sink_token is not None,
        wants_stats=wants_stats,
        scale=scale,
        device_cc=_device_cc(),
        device_sm_count=_device_sm_count(),
        q_t=q,
        k_t=k,
        v_t=v,
        o_t=o,
        stats_t=(stats if (wants_stats or is_backward) else None),
        bias_t=rec.get("bias"),
        block_mask_t=rec.get("block_mask"),
        score_max_t=rec.get("score_max"),
        score_sum_exp_t=rec.get("score_sum_exp"),
        do_t=rec.get("dO"),
        dq_t=rec.get("dQ"),
        dk_t=rec.get("dK"),
        dv_t=rec.get("dV"),
        dbias_t=rec.get("dBias"),
        dsink_t=rec.get("dSink_token"),
        sink_t=sink_token,
        seq_kv_t=seq_len_kv,
        seq_q_t=seq_len_q,
        cu_seq_q_t=cu_seq_q,
        cu_seq_kv_t=cu_seq_kv,
        sf_q_t=(dsc_q if is_mxfp8 else None),
        sf_k_t=(dsc_k if is_mxfp8 else None),
        sf_v_t=(dsc_v if is_mxfp8 else None),
        amax_o_t=rec.get("amax_o"),
        descale_q_t=(dsc_q if is_fp8 else None),
        descale_k_t=(dsc_k if is_fp8 else None),
        descale_v_t=(dsc_v if is_fp8 else None),
        scale_o_t=(rec.get("scale_o") if is_fp8 else None),
        descale_s_t=(rec.get("descale_s") if is_fp8 else None),
        scale_s_t=(rec.get("scale_s") if is_fp8 else None),
        # Amax_S: the op RETURNS the port unconditionally; only a real
        # (non-virtual, set_output(True)) tensor is a requested output.
        amax_s_t=(rec.get("amax_s") if (is_fp8 and rec.get("amax_s") is not None and not getattr(rec.get("amax_s"), "is_virtual", True)) else None),
    )


def analyze(graph: "cudnn.pygraph") -> Optional[SdpaGraphFacts]:
    """Facts for a single-SDPA graph, or None if the graph is anything else.

    Pure: attaching and caching is the graph's job (validate() ->
    _attach_facts), so the ranking and the engine share one record instead of
    each holding a private one. This is the callable the manifest names in the
    SDPA family's ``analyzer``.
    """
    node = _single_sdpa_node(graph)
    if node is None:
        return None
    return _extract_facts(_record_from_node(node))


# ---------------------------------------------------------------------------
# Graph-side runtime helpers shared by SDPA engines
# ---------------------------------------------------------------------------


@dataclass
class SdpaBinding:
    q: Any
    k: Any
    v: Any
    o: Any
    stats: Any = None
    sink_token: Any = None
    seq_len_kv: Any = None
    seq_len_q: Any = None
    # (B+1,) prefix-sum form (cuDNN 9.24+); at most one form per side.
    cu_seq_len_q: Any = None
    cu_seq_len_kv: Any = None
    # MXFP8 block-scale (descale) tensors + Amax_O output.
    sf_q: Any = None
    sf_k: Any = None
    sf_v: Any = None
    amax_o: Any = None
    # Per-tensor FP8 scalar descales + Amax_S output.
    descale_q: Any = None
    descale_k: Any = None
    descale_v: Any = None
    scale_o: Any = None
    descale_s: Any = None
    scale_s: Any = None
    # SM80 feature operands + backward ports.
    bias: Any = None
    block_mask: Any = None
    score_max: Any = None
    score_sum_exp: Any = None
    do: Any = None
    dq: Any = None
    dk: Any = None
    dv: Any = None
    dbias: Any = None
    dsink: Any = None

    # Built once on first use and reused. Rebuilding it per execute cost ~1.3 us
    # per bound operand: three passes over the bound list and five dict
    # constructions, not any one expensive getter.
    #
    # What makes the cache safe is the graph, not this class: a binding is
    # constructed by the engine's lowering AFTER the graph is frozen, and a
    # frozen graph can no longer re-uid or rename a tensor, so the names and
    # uids indexed here cannot move. The binding itself is still an ordinary
    # mutable dataclass -- reassigning a field after the first index() would go
    # unnoticed. Nothing does; an ordered-slot binding would remove the question.
    # init=False so a replace()d binding rebuilds rather than inheriting a
    # cache for the operands it no longer has; compare/repr excluded so the
    # cache cannot change how a binding prints or compares.
    _index: Optional[tuple] = field(default=None, init=False, repr=False, compare=False)

    def _build_index(self) -> tuple:
        bound = [
            t
            for t in (
                self.q,
                self.k,
                self.v,
                self.o,
                self.stats,
                self.sink_token,
                self.seq_len_kv,
                self.seq_len_q,
                self.cu_seq_len_q,
                self.cu_seq_len_kv,
                self.sf_q,
                self.sf_k,
                self.sf_v,
                self.amax_o,
                self.descale_q,
                self.descale_k,
                self.descale_v,
                self.scale_o,
                self.descale_s,
                self.scale_s,
                self.bias,
                self.block_mask,
                self.score_max,
                self.score_sum_exp,
                self.do,
                self.dq,
                self.dk,
                self.dv,
                self.dbias,
                self.dsink,
            )
            if t is not None
        ]
        name_counts: dict = {}
        uid_counts: dict = {}
        names, uids = [], []
        for t in bound:
            nm = _safe_name(t)
            names.append(nm)
            if nm is not None:
                name_counts[nm] = name_counts.get(nm, 0) + 1
            uid = _safe_uid(t)
            uids.append(uid)
            if uid is not None:
                uid_counts[uid] = uid_counts.get(uid, 0) + 1
        # A name or uid carried by two bound tensors identifies neither.
        by_obj = {id(t): t for t in bound}
        by_name = {nm: t for nm, t in zip(names, bound) if nm is not None and name_counts[nm] == 1}
        by_uid = {uid: t for uid, t in zip(uids, bound) if uid is not None and uid_counts[uid] == 1}
        self._index = (tuple(bound), by_obj, by_uid, by_name)
        return self._index

    def index(self) -> tuple:
        """``(bound, by_obj, by_uid, by_name)`` — the resolution tables."""
        return self._index or self._build_index()

    def bound_tensors(self) -> tuple:
        """The bound tensors, in slot order. A tuple: this is the binding's own
        record, not a working list for a caller to edit."""
        return self.index()[0]


def _safe_name(t: Any) -> Optional[str]:
    try:
        nm = t.get_name()
    except Exception:  # noqa: BLE001
        return None
    return nm or None


def _safe_uid(t: Any) -> Optional[int]:
    """The tensor's user-assigned uid, or None. ``uid_assigned`` distinguishes an
    explicit ``set_uid`` (any value, including 0) from the provisional uid that
    ``get_uid`` mints on demand — matching by a provisional uid would misbind
    variant packs keyed by the user's own numbering."""
    try:
        if not getattr(t, "uid_assigned", False):
            return None
        uid = t.get_uid()
    except Exception:  # noqa: BLE001
        return None
    return uid if isinstance(uid, int) and uid >= 0 else None


def resolve_variant_pack(variant_pack: dict, binding: SdpaBinding) -> dict:
    if not isinstance(variant_pack, dict):
        raise TypeError(
            f"cudnn.sdpa: compiled plans are called with a variant-pack dict {{cudnn_tensor | uid | name: buffer}}; got {type(variant_pack).__name__}"
        )
    _bound, by_obj, by_uid, by_name = binding.index()

    resolved: dict = {}
    for key, buf in variant_pack.items():
        if id(key) in by_obj:
            t = by_obj[id(key)]
        elif isinstance(key, int) and key in by_uid:
            t = by_uid[key]
        elif isinstance(key, str) and key in by_name:
            t = by_name[key]
        else:
            continue
        resolved[id(t)] = buf
    return resolved


def tensor_desc_from_ir(t: Any, name: str = "") -> "TensorDesc":
    from cudnn.api_base import TensorDesc

    shape = tuple(t.get_dim())
    stride = tuple(t.get_stride())
    dtype = to_torch_dtype(t.get_data_type())
    return TensorDesc(
        dtype=dtype,
        shape=shape,
        stride=stride,
        stride_order=_stride_order(shape, stride),
        device=_torch_cuda_device(),
        name=name,
    )


def adapter_mask_args(facts: "SdpaGraphFacts") -> dict:
    """Map resolved graph mask facts onto the standalone adapters'
    (is_causal, window_size, causal_bottom_right) vocabulary."""
    is_causal = facts.right_bound is not None
    win_left = facts.window_left if facts.window_left is not None else -1
    win_right = facts.right_bound if (facts.right_bound is not None and facts.right_bound > 0) else -1
    # A BOTTOM_RIGHT alignment only means something combined with a causal
    # bound and/or a left window; bare BR on a dense graph is a no-op.
    bottom_right = facts.bottom_right and (is_causal or facts.window_left is not None)
    return dict(
        is_causal=is_causal,
        window_size=(win_left, win_right),
        causal_bottom_right=bottom_right,
    )


@dataclass
class FeatureOperands:
    """The optional feature operands of one sdpa graph, resolved from a
    variant pack.  Raw buffers exactly as the caller provided them; each
    lowering applies its own normalization on top."""

    bias: Any = None
    sinks: Any = None
    seq_kv_lens: Any = None
    seq_len_q: Any = None
    block_mask: Any = None
    alibi: bool = False


def resolve_feature_operands(facts: "SdpaGraphFacts", resolved: dict) -> FeatureOperands:
    """Presence-checked resolution of the feature operands the facts demand.

    A feature the graph requests whose buffer is absent from the variant pack
    is an error here — every lowering would otherwise fail later and worse
    (a silently-dense mask, a null-deref in the kernel host code).
    """

    def _need(t_ref, label):
        buf = resolved.get(id(t_ref)) if t_ref is not None else None
        if buf is None:
            raise ValueError(f"cudnn.sdpa: {label} requested but no buffer was provided")
        return buf

    ops = FeatureOperands(alibi=facts.has_alibi)
    if facts.padded:
        # Either length form satisfies a side: per-batch seq_len_* or the
        # (B+1,) cu_seq_len_* prefix sums (cuDNN 9.24+) — the cu buffer
        # travels through the same operand slot (the adapter was constructed
        # knowing the form).
        if facts.cu_seq_kv_t is not None:
            ops.seq_kv_lens = _need(facts.cu_seq_kv_t, "padding mask (cu_seq_len_kv)")
        else:
            ops.seq_kv_lens = _need(facts.seq_kv_t, "padding mask (seq_len_kv)")
        if facts.cu_seq_q_t is not None:
            ops.seq_len_q = _need(facts.cu_seq_q_t, "per-batch query lengths (cu_seq_len_q)")
        elif facts.seq_q_t is not None:
            ops.seq_len_q = _need(facts.seq_q_t, "per-batch query lengths (seq_len_q)")
    if facts.has_bias:
        ops.bias = _need(facts.bias_t, "bias")
    if facts.has_sink:
        ops.sinks = _need(facts.sink_t, "sink_token")
    if facts.has_block_mask:
        ops.block_mask = _need(facts.block_mask_t, "block_mask")
    return ops


def adapter_feature_buffers(facts: "SdpaGraphFacts", resolved: dict) -> dict:
    """:func:`resolve_feature_operands` mapped onto the standalone adapters'
    kwarg vocabulary, with the flat-tensor normalization their kernels expect."""
    ops = resolve_feature_operands(facts, resolved)
    out: dict = {}
    # Dtypes are facts-gated (seq lens int32, sinks fp32): pure views only —
    # a .to() here would allocate and launch a cast kernel per execute.
    if ops.seq_kv_lens is not None:
        out["seq_kv_lens"] = ops.seq_kv_lens.reshape(-1)
    if ops.seq_len_q is not None:
        out["seq_len_q"] = ops.seq_len_q.reshape(-1)
    if ops.bias is not None:
        out["bias_tensor"] = ops.bias
    if ops.sinks is not None:
        out["sinks"] = ops.sinks.reshape(-1)
    if ops.block_mask is not None:
        out["block_mask"] = ops.block_mask
    if ops.alibi:
        out["alibi"] = True
    return out


def to_bshd_physical(t: "torch.Tensor") -> "torch.Tensor":
    """BSHD-physical (stride order 3,1,2,0) copy of a rank-4 BHSD-logical
    tensor; zero-copy when the buffer already is.  Delivers the dense_flex
    layout relaxation for lowerings whose adapters require this order
    (mirrors the DSL executor's canonical-buffer gather)."""
    strides = t.stride()
    # already BSHD-physical (size-1 dims wildcarded): D innermost, then H, S, B
    order = sorted(range(4), key=lambda i: (strides[i], t.shape[i]))
    act = tuple(ax for ax in order if t.shape[ax] != 1)
    exp = tuple(ax for ax in (3, 1, 2, 0) if t.shape[ax] != 1)
    if act == exp:
        return t
    return t.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)


def expand_gqa_heads(t: "torch.Tensor", h_q: int) -> "torch.Tensor":
    """Expand K/V heads to H_q for a dense forward lowering (BHSD dim 1).

    The forward kernels' native dense-GQA path is not exercised by the
    upstream validation harness (which expands like this); keep the
    validated shape until kernel-level dense GQA is qualified.
    """
    h = t.shape[1]
    if h == h_q:
        return t
    return t.repeat_interleave(h_q // h, dim=1)
