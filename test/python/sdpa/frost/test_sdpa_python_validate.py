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

import pytest

import cudnn
from cudnn.engines import manifest

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


def test_classic_path_still_lowers(no_candidates):
    """No python candidate: validate() keeps the classic eager C++ lowering."""
    g, _ = _sdpa_graph(use_causal_mask=True)
    g.validate()
    assert g._lowered_graph is not None, "without python candidates the classic eager lowering must be unchanged"


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
