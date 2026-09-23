# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-tensor FP8 d=256 SDPA backward on Rubin (SM 10.7): graph admission.

Bring-up placeholder for the ``sdpa_bwd_sm107_fp8`` row.  Today it pins only
the GRAPH prerequisites on a real Rubin device: the C++ node
(``include/cudnn_frontend/node/sdpa_fp8_bwd.h``) admits ``sdpa_fp8_backward``
at ``d_qk == d_v == 256`` on cc 10.7 for the frontend engines, and -- with no
row serving it yet -- planning declines the graph with the typed
not-supported rather than a backend-API error.  The row's ACCEPT / REJECT
tests (bitwise vs the reference kernel, the fp64 oracle with fp8 flip budgets,
the degenerate-input matrix, the static SASS pins) land in this module with
the kernel.  The host-side twins of the admission cases (``sm_version`` as a
fake device) live in ``test_sdpa_graph_analyzer.py``.
"""

from __future__ import annotations

import math

import pytest

import cudnn
from frost_test_utils import requires_rubin

pytestmark = [pytest.mark.L0, requires_rubin]

_ENGINE = "sdpa_bwd_sm107_fp8"
_FP8 = cudnn.data_type.FP8_E4M3
_D = 256
# cuDNN's sdpa_fp8_backward scalar set, in the op's positional order.
_SCALARS = (
    "descale_q",
    "descale_k",
    "descale_v",
    "descale_o",
    "descale_dO",
    "descale_s",
    "descale_dP",
    "scale_s",
    "scale_dQ",
    "scale_dK",
    "scale_dV",
    "scale_dP",
)
_AMAX = ("amax_dQ", "amax_dK", "amax_dV", "amax_dP")


def _bshd(h, s, d):
    """cuDNN declares logical BHSD; the FROST rows want BSHD-physical storage."""
    return (s * h * d, d, h * d, 1)


def _build_graph(b=1, h=4, s=512, d=_D, *, causal=True):
    """``sdpa_fp8_backward`` with FP8 Q/K/V/O/dO, fp32 Stats, the twelve scalar
    descales / scales, FP8 dQ/dK/dV and all four amax outputs requested -- the
    contract the d256 row implements."""
    g = cudnn.pygraph(io_data_type=_FP8, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    dims, strides = (b, h, s, d), _bshd(h, s, d)
    ts = {name: g.tensor(dim=dims, stride=strides, data_type=_FP8, name=name) for name in ("q", "k", "v", "o", "dO")}
    ts["stats"] = g.tensor(dim=(b, h, s, 1), stride=(h * s, s, 1, 1), data_type=cudnn.data_type.FLOAT, name="stats")
    for name in _SCALARS:
        ts[name] = g.tensor(dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT, name=name)
    outs = g.sdpa_fp8_backward(name="fb", attn_scale=1.0 / math.sqrt(d), use_causal_mask=causal, **ts)
    for t in outs[:3]:
        t.set_output(True).set_dim(dims).set_stride(strides).set_data_type(_FP8)
    for t in outs[3:]:
        t.set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    return g


def test_graph_admits_d256_fp8_backward_on_rubin(monkeypatch):
    """Classic eager C++ validation (FROST off, the real device query -> cc 10.7):
    the node no longer raises the ``hidden_dim`` not-supported at d=256."""
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    g = _build_graph()
    g.validate()
    assert g._lowered_graph is not None, "classic path: the C++ graph validated the node"


def test_graph_declines_cleanly_until_a_row_serves_it():
    """FROST on (the suite's autouse opt-in): validate() is python-native and the
    backend's verdict is deferred to planning, where the backend has no plan for
    this shape (``override_heuristics_query`` returns ``{-1, {}}`` so heuristics
    runs and finds no config) and no python row claims it -- the typed
    not-supported is what surfaces, a decline callers can act on.  Retires
    itself once the row is registered: its accept tests take over."""
    from cudnn.sdpa.bwd import engines as bwd_engines

    if any(spec.name == _ENGINE for spec in bwd_engines.ENGINE_SPECS):
        pytest.skip(f"{_ENGINE} is registered; its accept tests supersede this placeholder")
    g = _build_graph()
    g.validate()
    assert g._lowered_graph is None, "with a FROST candidate the backend's verdict is deferred to planning"
    g.build_operation_graph()
    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        g.create_execution_plans([cudnn.heur_mode.A])
