# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""pygraph.validate() for GEMM-family graphs through the manifest's per-family
validator hook (issue #704): with a python engine candidate the frost_gemm
validator runs instead of the eager C++ lowering; without one, or for a graph
holding a node it does not cover, the classic lowering is unchanged. Device-free.
"""

import pytest

import cudnn
from cudnn.engines import manifest


@pytest.fixture
def frost_candidate(monkeypatch):
    """Pretend the manifest offers a python engine for every graph."""
    monkeypatch.setattr(manifest, "engines_for", lambda graph: [object()])


@pytest.fixture
def no_candidates(monkeypatch):
    """Manifest offers no python engine (frost off / family unavailable)."""
    monkeypatch.setattr(manifest, "engines_for", lambda graph: [])


def _matmul_graph(a_dim=(2, 64, 32), b_dim=(2, 32, 48), c_dim=None):
    """A minimal batched matmul pygraph; returns (graph, C)."""
    g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)

    def rm(d):
        s, out = 1, []
        for x in reversed(d):
            out.append(s)
            s *= x
        return list(reversed(out))

    a = g.tensor(name="A", dim=list(a_dim), stride=rm(a_dim))
    b = g.tensor(name="B", dim=list(b_dim), stride=rm(b_dim))
    c = g.matmul(A=a, B=b)
    c.set_output(True)
    if c_dim is not None:
        c.set_dim(list(c_dim)).set_stride(rm(c_dim))
    return g, c


def test_family_declares_validator():
    """frost_gemm and both SDPA families declare a resolvable native validator."""
    by_name = {f.name: f for f in manifest.MANIFEST}
    for name in ("frost_gemm", "frost_sdpa_fwd", "frost_sdpa_bwd"):
        assert by_name[name].validator is not None, name
        assert callable(manifest.resolve_validator(by_name[name])), name
    for name in ("gdn", "kda", "gdn2", "gdp"):  # python_only families: never eagerly lowered anyway
        assert by_name[name].validator is None


def test_valid_matmul_does_not_lower(frost_candidate):
    """With a python candidate, a GEMM graph validates natively (no C++ lowering)."""
    g, _ = _matmul_graph()
    g.validate()
    assert g._lowered_graph is None


def test_classic_path_without_candidates(no_candidates):
    """No python candidate: the classic eager C++ lowering is unchanged."""
    g, _ = _matmul_graph()
    g.validate()
    assert g._lowered_graph is not None


def test_mixed_graph_still_lowers(frost_candidate):
    """A node the GEMM validator does not cover (pointwise epilogue) keeps the
    classic eager lowering even with a candidate: the rule is every-node-covered."""
    g, c = _matmul_graph()
    g.relu(input=c).set_output(True)
    g.validate()
    assert g._lowered_graph is not None


def test_contraction_mismatch_rejected(frost_candidate):
    """K disagreement between A and B is a GRAPH_NOT_SUPPORTED-class rejection from validate()."""
    g, _ = _matmul_graph(a_dim=(2, 64, 32), b_dim=(2, 40, 48))
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="contraction mismatch"):
        g.validate()
    assert g._lowered_graph is None and g._is_validated is False


def test_batch_broadcast_rules(frost_candidate):
    """Batch extents must be equal or 1 (broadcast)."""
    g, _ = _matmul_graph(a_dim=(1, 64, 32), b_dim=(4, 32, 48))
    g.validate()  # broadcast OK
    g, _ = _matmul_graph(a_dim=(3, 64, 32), b_dim=(4, 32, 48))
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="not broadcastable"):
        g.validate()


def test_declared_output_dims_checked(frost_candidate):
    """A user-declared C must carry the (M, N) the operands imply."""
    g, _ = _matmul_graph(c_dim=(2, 64, 40))
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="do not match"):
        g.validate()


def test_family_without_validator_lowers_classically(frost_candidate, monkeypatch):
    """A family that declares no validator (or whose module is unavailable) keeps
    the classic eager lowering even with a candidate."""
    monkeypatch.setattr(manifest, "resolve_validator", lambda family: None)
    g, _ = _matmul_graph()
    g.validate()
    assert g._lowered_graph is not None
