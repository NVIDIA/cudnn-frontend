# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python-native SDPA validation (issue #704).

With a FROST engine candidate, ``pygraph.validate()`` must not lower to C++ —
the eager backend validation is what coupled frontend-only engines to the
installed backend's version. Semantic invalidity must still raise from
validate(), with classic error types (``cudnnGraphNotSupportedError`` for
GRAPH_NOT_SUPPORTED parity, ``ValueError`` for ATTRIBUTE_NOT_SET/INVALID_VALUE
parity). Without a candidate, the classic eager-lowering timing is unchanged.

Device-free: candidates are stubbed via ``manifest.engines_for``; nothing here
builds plans or executes.
"""

import math

import pytest

import cudnn
from cudnn.engines import manifest
from frost_test_utils import requires_sm80

pytestmark = pytest.mark.L0


@pytest.fixture
def frost_candidate(monkeypatch):
    """Pretend the manifest offers a python engine for every graph."""
    monkeypatch.setattr(manifest, "engines_for", lambda graph: [object()])


@pytest.fixture
def no_candidates(monkeypatch):
    """Manifest offers no python engine (frost off / family unavailable)."""
    monkeypatch.setattr(manifest, "engines_for", lambda graph: [])


def _sdpa_graph(
    b=1,
    h_q=2,
    h_kv=2,
    s_q=64,
    s_kv=64,
    d=64,
    **sdpa_kwargs,
):
    """A minimal SDPA forward pygraph; returns (graph, tensors-by-name)."""
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q = g.tensor(name="q", dim=[b, h_q, s_q, d], stride=[h_q * s_q * d, s_q * d, d, 1])
    k = g.tensor(name="k", dim=[b, h_kv, s_kv, d], stride=[h_kv * s_kv * d, s_kv * d, d, 1])
    v = g.tensor(name="v", dim=[b, h_kv, s_kv, d], stride=[h_kv * s_kv * d, s_kv * d, d, 1])
    o, stats = g.sdpa(q, k, v, generate_stats=True, **sdpa_kwargs)
    o.set_output(True)
    if stats is not None:
        stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    return g, dict(q=q, k=k, v=v, o=o, stats=stats)


def test_valid_graph_does_not_lower(frost_candidate):
    """With a python candidate, validate() skips the eager C++ lowering."""
    g, _ = _sdpa_graph(use_causal_mask=True)
    g.validate()
    assert g._lowered_graph is None, "validate() must not lower to C++ when a python engine is a candidate"


@requires_sm80
def test_classic_path_still_lowers(no_candidates):
    """No python candidate: validate() keeps the classic eager C++ lowering."""
    g, _ = _sdpa_graph(use_causal_mask=True)
    g.validate()
    assert g._lowered_graph is not None, "without python candidates the classic eager lowering must be unchanged"


@requires_sm80
def test_mixed_graph_still_lowers(frost_candidate):
    """A node outside COVERED_NODE_TYPES (pointwise on O) forces the classic eager
    lowering even with a python candidate: the routing rule is every-node-covered."""
    g, ts = _sdpa_graph(use_causal_mask=True)
    g.swish(ts["o"]).set_output(True)
    g.validate()
    assert g._lowered_graph is not None, "a graph with an uncovered node must keep the classic eager lowering"


def test_rejected_graph_stays_unvalidated(frost_candidate):
    """A native-validation rejection leaves the graph unvalidated: the flag is set
    only after every check passes, so build_operation_graph() re-validates and
    raises the same error rather than planning a rejected graph."""
    g, _ = _sdpa_graph(h_q=3, h_kv=2)
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="group-query attention"):
        g.validate()
    assert g._is_validated is False
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="group-query attention"):
        g.build_operation_graph()


def test_rejected_classic_graph_stays_unvalidated(no_candidates):
    """Classic path (no python candidate): a backend rejection rolls the lowering
    back, so the retry re-runs the backend check and raises again instead of
    finding a stale lowered graph and marking the rejected graph valid."""
    g, _ = _sdpa_graph(h_q=3, h_kv=2)
    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        g.validate()
    assert g._is_validated is False and g._lowered_graph is None
    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        g.build_operation_graph()
    assert g._is_validated is False


def test_gqa_head_divisibility(frost_candidate):
    """h_q not a multiple of h_kv is rejected with the classic error type, without lowering."""
    g, _ = _sdpa_graph(h_q=3, h_kv=2)
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="group-query attention"):
        g.validate()
    assert g._lowered_graph is None


def test_bottom_right_causal_with_bias(frost_candidate):
    """Bottom-right causal + bias is a classic GRAPH_NOT_SUPPORTED combination."""
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    dims = [1, 2, 64, 64]
    strides = [2 * 64 * 64, 64 * 64, 64, 1]
    q = g.tensor(name="q", dim=dims, stride=strides)
    k = g.tensor(name="k", dim=dims, stride=strides)
    v = g.tensor(name="v", dim=dims, stride=strides)
    bias = g.tensor(name="bias", dim=[1, 1, 64, 64], stride=[64 * 64, 64 * 64, 64, 1])
    o, _ = g.sdpa(q, k, v, bias=bias, use_causal_mask_bottom_right=True, generate_stats=False)
    o.set_output(True)
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="Bottom right causal mask"):
        g.validate()


def test_padding_mask_requires_seq_lens(frost_candidate):
    """Padding mask without seq_len tensors is an ATTRIBUTE_NOT_SET-class ValueError."""
    g, _ = _sdpa_graph(use_padding_mask=True)
    with pytest.raises(ValueError, match="Padding mask requires"):
        g.validate()


def test_seq_lens_require_padding_mask(frost_candidate):
    """seq_len tensors without the padding mask are rejected (classic INVALID_VALUE parity)."""
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    dims = [2, 2, 64, 64]
    strides = [2 * 64 * 64, 64 * 64, 64, 1]
    q = g.tensor(name="q", dim=dims, stride=strides)
    k = g.tensor(name="k", dim=dims, stride=strides)
    v = g.tensor(name="v", dim=dims, stride=strides)
    seq_q = g.tensor(name="seq_q", dim=[2, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
    seq_kv = g.tensor(name="seq_kv", dim=[2, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
    o, _ = g.sdpa(q, k, v, seq_len_q=seq_q, seq_len_kv=seq_kv, generate_stats=False)
    o.set_output(True)
    with pytest.raises(ValueError, match="only if padding mask is enabled"):
        g.validate()


def test_dropout_probability_one_rejected(frost_candidate):
    """Probability-form dropout with p = 1.0 is rejected at validate()."""
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    dims = [1, 2, 64, 64]
    strides = [2 * 64 * 64, 64 * 64, 64, 1]
    q = g.tensor(name="q", dim=dims, stride=strides)
    k = g.tensor(name="k", dim=dims, stride=strides)
    v = g.tensor(name="v", dim=dims, stride=strides)
    seed = g.tensor(name="seed", dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64)
    offset = g.tensor(name="offset", dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64)
    o, _ = g.sdpa(q, k, v, dropout=(1.0, seed, offset), generate_stats=False)
    o.set_output(True)
    with pytest.raises(ValueError, match="Dropout probability cannot be 1"):
        g.validate()


def test_alibi_requires_causal(frost_candidate):
    """ALiBi without the causal mask is rejected."""
    g, _ = _sdpa_graph(use_alibi_mask=True)
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="alibi"):
        g.validate()


def test_non_fp32_stats_rejected(frost_candidate):
    """Stats must be FP32; a HALF Stats output is rejected."""
    g, ts = _sdpa_graph(use_causal_mask=True)
    ts["stats"].set_data_type(cudnn.data_type.HALF)
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="Stats output of sdpa must be an FP32"):
        g.validate()


def test_head_dim_multiple_of_8(frost_candidate):
    """Head dim not a multiple of 8 is rejected."""
    g, _ = _sdpa_graph(d=68, use_causal_mask=True)
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="multiple of 8"):
        g.validate()


def test_sliding_window_sq_gt_skv_rejected(frost_candidate):
    """Sliding window with s_q > s_kv is rejected."""
    g, _ = _sdpa_graph(s_q=128, s_kv=64, use_causal_mask=True, sliding_window_length=16)
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="Sliding window attention"):
        g.validate()


def test_bwd_sq1_skv1_rejected(frost_candidate):
    """Backward with s_q = s_kv = 1 is rejected, without lowering."""
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    dims = [1, 2, 1, 64]
    strides = [2 * 64, 64, 64, 1]
    q = g.tensor(name="q", dim=dims, stride=strides)
    k = g.tensor(name="k", dim=dims, stride=strides)
    v = g.tensor(name="v", dim=dims, stride=strides)
    o = g.tensor(name="o", dim=dims, stride=strides)
    dO = g.tensor(name="dO", dim=dims, stride=strides)
    stats = g.tensor(name="stats", dim=[1, 2, 1, 1], stride=[2, 1, 1, 1], data_type=cudnn.data_type.FLOAT)
    dQ, dK, dV = g.sdpa_backward(q, k, v, o, dO, stats)
    for t in (dQ, dK, dV):
        t.set_output(True)
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="s_q = s_kv = 1"):
        g.validate()
    assert g._lowered_graph is None


# ---------------------------------------------------------------------------
# Epilogue-gate tail (PR-A): sdpa(virtual O_v) -> sigmoid(G) -> mul(O_v, s)
#
# The three-node graph is validated python-natively (cudnn._sdpa_tail.match_gate_tail
# + the sdpa node's own rules + G/O dim-stride checks), so pygraph.validate()
# does not fall back to the eager C++ lowering -- whose pre_validate_node
# needs a rank-4 dim + stride on the sdpa node's O, which the frontend pushes
# only when USER-assigned.  An undeclared O_v is therefore a typed
# not-supported here (plan S7 Q8: a user contract, not a _pygraph change).
# ---------------------------------------------------------------------------


def _gated_sdpa_graph(b=1, h=2, s=64, d=256, *, declare_o_v=True, gate_dim=None, **sdpa_kwargs):
    """(graph, tensors-by-name) for the gate tail; ``o`` (the mul output) is the output."""
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    dims, strides = [b, h, s, d], [h * s * d, s * d, d, 1]
    q = g.tensor(name="q", dim=dims, stride=strides)
    k = g.tensor(name="k", dim=dims, stride=strides)
    v = g.tensor(name="v", dim=dims, stride=strides)
    gdim = list(gate_dim) if gate_dim is not None else dims
    gstride = []
    run = 1
    for n in reversed(gdim):
        gstride.append(run)
        run *= n
    gate = g.tensor(name="gate", dim=gdim, stride=list(reversed(gstride)))
    o_v, _ = g.sdpa(q, k, v, generate_stats=False, **sdpa_kwargs)
    if declare_o_v:
        o_v.set_dim(dims).set_stride(strides)
    sig = g.sigmoid(input=gate)
    o = g.mul(a=o_v, b=sig)
    o.set_output(True).set_dim(dims).set_stride(strides)
    return g, dict(q=q, k=k, v=v, gate=gate, o_v=o_v, s=sig, o=o)


def test_gate_tail_validates_natively(frost_candidate):
    """With a python candidate the WELL-FORMED tail validates without lowering
    to C++ (the pointwise nodes are covered through the tail matcher, not
    COVERED_NODE_TYPES), while the semantic checks still fire with the classic
    error types: a wrong-rank G is rejected, and an UNDECLARED virtual O_v is
    the typed not-supported the graph contract requires (a bare ValueError
    would otherwise escape create_execution_plans from the C++ pre-validation).
    The all-covered single-node fast path and the classic fallback for an
    uncovered node (test_mixed_graph_still_lowers) are unchanged."""
    g, _ = _gated_sdpa_graph(use_causal_mask=True)
    g.validate()
    assert g._lowered_graph is None, "the gate tail must validate python-natively when a python engine is a candidate"
    assert g._is_validated is True

    # G must be rank-4 with a unit head-dim stride, like every SDPA operand.
    g, _ = _gated_sdpa_graph(gate_dim=(1, 2, 64), use_causal_mask=True)
    with pytest.raises((ValueError, cudnn.cudnnGraphNotSupportedError)):
        g.validate()
    assert g._is_validated is False

    # A broadcast G is a legal pointwise graph but NOT the fusable tail: typed not-supported.
    g, _ = _gated_sdpa_graph(gate_dim=(1, 1, 64, 256), use_causal_mask=True)
    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        g.validate()
    assert g._is_validated is False

    # The sdpa node's own rules still apply through the tail (h_q=3 over h_kv=2 trips the GQA rule).
    gg = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q = gg.tensor(name="q", dim=[1, 3, 64, 256], stride=[3 * 64 * 256, 64 * 256, 256, 1])
    k = gg.tensor(name="k", dim=[1, 2, 64, 256], stride=[2 * 64 * 256, 64 * 256, 256, 1])
    v = gg.tensor(name="v", dim=[1, 2, 64, 256], stride=[2 * 64 * 256, 64 * 256, 256, 1])
    gate = gg.tensor(name="gate", dim=[1, 3, 64, 256], stride=[3 * 64 * 256, 64 * 256, 256, 1])
    o_v, _ = gg.sdpa(q, k, v, generate_stats=False, use_causal_mask=True)
    o_v.set_dim([1, 3, 64, 256]).set_stride([3 * 64 * 256, 64 * 256, 256, 1])
    o = gg.mul(a=o_v, b=gg.sigmoid(input=gate))
    o.set_output(True).set_dim([1, 3, 64, 256]).set_stride([3 * 64 * 256, 64 * 256, 256, 1])
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="group-query attention"):
        gg.validate()
    assert gg._lowered_graph is None


def test_gate_tail_requires_a_declared_virtual_o_v(frost_candidate):
    """An undeclared O_v (no set_dim / set_stride on the sdpa node's virtual
    output) is rejected at validate() with the classic not-supported type and
    an actionable message -- BEFORE planning, where the C++ lowering would
    otherwise raise a bare ValueError (ATTRIBUTE_NOT_SET) out of
    create_execution_plans."""
    g, _ = _gated_sdpa_graph(declare_o_v=False, use_causal_mask=True)
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="set_dim"):
        g.validate()
    assert g._is_validated is False and g._lowered_graph is None
    # The declared twin passes.
    g, _ = _gated_sdpa_graph(declare_o_v=True, use_causal_mask=True)
    g.validate()
    assert g._lowered_graph is None


def test_gate_tail_match_is_structural():
    """The matcher the analyzer and the validator share: identity on the
    tensors, mode on the pointwise nodes, virtual-ness on O_v / s -- and nothing
    else (head dims, dtype, arch are the rows' business).  Import-light: it must
    not pull cudnn.sdpa (torch / cutlass) in."""
    import sys

    from cudnn._sdpa_tail import GateTail, match_gate_tail

    assert "cudnn._sdpa_tail" in sys.modules
    g, ts = _gated_sdpa_graph(use_causal_mask=True)
    tail = match_gate_tail(g.nodes)
    assert isinstance(tail, GateTail)
    assert tail.gate is ts["gate"] and tail.o_virtual is ts["o_v"] and tail.sig_out is ts["s"] and tail.o_final is ts["o"]
    assert tail.sdpa is g.nodes[0]
    # Order-insensitive on the mul operands; a fourth node or a real O_v is not the tail.
    g2, ts2 = _gated_sdpa_graph(use_causal_mask=True)
    ts2["o_v"].set_output(True)
    assert match_gate_tail(g2.nodes) is None
    g3, ts3 = _gated_sdpa_graph(use_causal_mask=True)
    g3.relu(input=ts3["o"]).set_output(True)
    assert match_gate_tail(g3.nodes) is None
    assert match_gate_tail(_sdpa_graph(use_causal_mask=True)[0].nodes) is None, "a single sdpa node is not a tail"


def test_mxfp8_unrequested_amax_o_validates_natively(frost_candidate):
    """``sdpa_mxfp8`` RETURNS its Amax_O port unconditionally.  A caller that does
    not request it (no set_output / set_dim / set_stride -- the block's
    has_amax_o=False path) must still pass validate() on the python-native
    route: the op spec now infers the ``[1, 1, 1, 1]`` scalar the way
    ``sdpa_fp8`` does, so the IR-level "Tensor 'sdpa_mxfp8.0::Amax_O' dims not
    set" is gone and the graph is NOT lowered to C++.  The declared twin (the
    harness's ``amax=True`` form) validates the same way and keeps its dims."""
    b, h, s, d = 1, 2, 128, 128
    dims, stride = [b, h, s, d], [h * s * d, s * d, d, 1]

    def build(*, request_amax_o):
        g = cudnn.pygraph(io_data_type=cudnn.data_type.FP8_E4M3, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        q = g.tensor(name="q", dim=dims, stride=stride)
        k = g.tensor(name="k", dim=dims, stride=stride)
        v = g.tensor(name="v", dim=dims, stride=stride)

        def sf(sd):  # F8_128x4 scale factors: Q/K rowwise [b,h,s,d/32], V columnwise [b,h,s/32,d]
            return g.tensor(
                dim=sd,
                stride=[sd[1] * sd[2] * sd[3], sd[2] * sd[3], sd[3], 1],
                data_type=cudnn.data_type.FP8_E8M0,
                reordering_type=cudnn.tensor_reordering.F8_128x4,
            )

        o, stats, amax_o = g.sdpa_mxfp8(
            q,
            k,
            v,
            sf([b, h, s, d // 32]),
            sf([b, h, s, d // 32]),
            sf([b, h, s // 32, d]),
            attn_scale=1.0 / math.sqrt(d),
            generate_stats=False,
            use_causal_mask=True,
        )
        assert stats is None  # inference-mode: no Stats port
        o.set_output(True).set_dim(dims).set_stride(stride).set_data_type(cudnn.data_type.HALF)
        if request_amax_o:
            amax_o.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
        return g, amax_o

    g, amax_o = build(request_amax_o=False)
    assert amax_o.is_virtual and not amax_o.dim_assigned
    g.validate()
    assert g._is_validated and g._lowered_graph is None, "validate() must not lower to C++ when a python engine is a candidate"
    assert amax_o.is_virtual and list(amax_o.dim) == [1, 1, 1, 1] and list(amax_o.stride) == [1, 1, 1, 1]

    g, amax_o = build(request_amax_o=True)
    g.validate()
    assert g._is_validated and g._lowered_graph is None
    assert not amax_o.is_virtual and amax_o.dim_assigned and list(amax_o.dim) == [1, 1, 1, 1]
