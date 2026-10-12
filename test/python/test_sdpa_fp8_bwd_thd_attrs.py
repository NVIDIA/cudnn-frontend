# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``max_total_seq_len_q`` / ``max_total_seq_len_kv`` on the two quantized SDPA backward bindings -- the pybind round trip.

A ragged (THD) graph declares every tensor as the envelope ``(B, H, S_max, D)`` plus a device-side ragged offset, so the
packed token total is not otherwise expressible in the graph; ``sdpa`` / ``sdpa_backward`` / ``sdpa_fp8`` / ``sdpa_mxfp8``
take it as the two declared totals, and ``sdpa_fp8_backward`` / ``sdpa_mxfp8_backward`` now do too (the attribute lives on
``SDPA_fp8_backward_attributes``, which both bindings build).  Three tiers, all host-side and none launching a kernel:

* the PYTHON graph records the kwargs verbatim and ``graph_analyzer`` reads them as facts (no device at all);
* the C++ BINDING accepts the kwargs as trailing keywords (an append-only pin, read off the pybind signature; no device);
* the C++ NODE carries the values -- the lowering forwards them to the setters and the attribute's serialization list
  prints them (needs a CUDA device for the backend handle the lowering opens; skipped without one).

A stale extension (one built before the attribute existed) fails the signature pin with a message naming the rebuild.
"""

from __future__ import annotations

import json

import pytest

import cudnn

pytestmark = [pytest.mark.L0]

_D = 256
_B, _H = 2, 2
_LENS_Q, _LENS_KV = (300, 128), (200, 128)
_T_Q, _T_KV = sum(_LENS_Q), sum(_LENS_KV)  # the packed totals a caller declares
_E4M3 = cudnn.data_type.FP8_E4M3
_F32 = cudnn.data_type.FLOAT
_I32 = cudnn.data_type.INT32
_I64 = cudnn.data_type.INT64
_BF16 = cudnn.data_type.BFLOAT16

_FP8_SCALARS = (
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
_MXFP8_SF = ("descale_q", "descale_q_T", "descale_k", "descale_k_T", "descale_v", "descale_dO", "descale_dO_T")


def _cuda_device_available() -> bool:
    try:
        import torch
    except ImportError:  # the lowering tiers need a device; without torch we cannot even ask
        return False
    return torch.cuda.is_available()


def _cuda_cc():
    import torch

    return torch.cuda.get_device_capability() if _cuda_device_available() else None


requires_cuda_device = pytest.mark.skipif(not _cuda_device_available(), reason="the C++ lowering opens a cuDNN handle on the ambient CUDA device")


def _ceil_div(n, m):
    return -(-n // m)


def _ceil_to(n, m):
    """``n`` rounded up to a multiple of ``m``."""
    return _ceil_div(n, m) * m


def _bshd(h, s, d):
    return [s * h * d, d, h * d, 1]


def _ragged(g, name, dims, stride, dtype, *, ragged=True):
    """An envelope tensor with its own ``(B + 1, 1, 1, 1)`` int64 ragged offset (``ragged=False``: a dense twin)."""
    t = g.tensor(name=name, dim=list(dims), stride=list(stride), data_type=dtype)
    if ragged:
        ro = g.tensor(name=f"{name}_ro", dim=[_B + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=_I64)
        t.set_ragged_offset(ro)
    return t


def _lengths(g):
    tq = g.tensor(name="seq_len_q", dim=[_B, 1, 1, 1], stride=[1, 1, 1, 1], data_type=_I32)
    tk = g.tensor(name="seq_len_kv", dim=[_B, 1, 1, 1], stride=[1, 1, 1, 1], data_type=_I32)
    return dict(use_padding_mask=True, seq_len_q=tq, seq_len_kv=tk)


def _finish_outputs(g, outs, grad_dims, grad_dtype, *, ragged=True):
    """dQ / dK / dV real with dims, strides, dtype and (THD) a ragged offset; every amax port dimensioned and real."""
    for out, (nh, s) in zip(outs[:3], grad_dims):
        out.set_output(True).set_data_type(grad_dtype).set_dim([_B, nh, s, _D]).set_stride(_bshd(nh, s, _D))
        if ragged:
            ro = g.tensor(name=f"{out.get_name()}_ro", dim=[_B + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=_I64)
            out.set_ragged_offset(ro)
    for a in outs[3:]:
        a.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(_F32)


def _fp8_backward_graph(*, declare_totals=True, ragged=True, hkv=_H, **sdpa_kwargs):
    """``sdpa_fp8_backward`` over packed e4m3 Q/K/V/O/dO, head-major packed fp32 Stats and the twelve scalar (de)scales."""
    s_q, s_kv = max(_LENS_Q), max(_LENS_KV)
    g = cudnn.pygraph(io_data_type=_E4M3, intermediate_data_type=_F32, compute_data_type=_F32)
    t = dict(
        q=_ragged(g, "q", (_B, _H, s_q, _D), _bshd(_H, s_q, _D), _E4M3, ragged=ragged),
        k=_ragged(g, "k", (_B, hkv, s_kv, _D), _bshd(hkv, s_kv, _D), _E4M3, ragged=ragged),
        v=_ragged(g, "v", (_B, hkv, s_kv, _D), _bshd(hkv, s_kv, _D), _E4M3, ragged=ragged),
        o=_ragged(g, "o", (_B, _H, s_q, _D), _bshd(_H, s_q, _D), _E4M3, ragged=ragged),
        dO=_ragged(g, "dO", (_B, _H, s_q, _D), _bshd(_H, s_q, _D), _E4M3, ragged=ragged),
        stats=_ragged(g, "stats", (_B, _H, s_q, 1), [_H * _T_Q, _T_Q, 1, 1] if ragged else [_H * s_q, s_q, 1, 1], _F32, ragged=ragged),
    )
    for name in _FP8_SCALARS:
        t[name] = g.tensor(name=name, dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=_F32)
    kw = dict(name="fp8_bwd", attn_scale=_D**-0.5, **t)
    if ragged:
        kw.update(_lengths(g))
    if declare_totals:
        kw.update(max_total_seq_len_q=_T_Q, max_total_seq_len_kv=_T_KV)
    kw.update(sdpa_kwargs)
    outs = g.sdpa_fp8_backward(**kw)
    assert len(outs) == 7
    _finish_outputs(g, outs, ((_H, s_q), (hkv, s_kv), (hkv, s_kv)), _E4M3, ragged=ragged)
    return g


def _mxfp8_backward_graph(*, declare_totals=True, ragged=True, hkv=_H, **sdpa_kwargs):
    """``sdpa_mxfp8_backward`` over packed e4m3 Q/Q_T/K/K_T/V/dO/dO_T, bf16 O/dO, packed fp32 Stats and the seven block
    scale-factor tensors (E8M0, the F8_128x4 reordering)."""
    s_q, s_kv = max(_LENS_Q), max(_LENS_KV)
    g = cudnn.pygraph(io_data_type=_E4M3, intermediate_data_type=_F32, compute_data_type=_F32)
    t = {}
    for name, nh, s, dt in (
        ("q", _H, s_q, _E4M3),
        ("q_T", _H, s_q, _E4M3),
        ("k", hkv, s_kv, _E4M3),
        ("k_T", hkv, s_kv, _E4M3),
        ("v", hkv, s_kv, _E4M3),
        ("o_f16", _H, s_q, _BF16),
        ("dO_f16", _H, s_q, _BF16),
        ("dO", _H, s_q, _E4M3),
        ("dO_T", _H, s_q, _E4M3),
    ):
        t[name] = _ragged(g, name, (_B, nh, s, _D), _bshd(nh, s, _D), dt, ragged=ragged)
    t["stats"] = _ragged(g, "stats", (_B, _H, s_q, 1), [_H * _T_Q, _T_Q, 1, 1] if ragged else [_H * s_q, s_q, 1, 1], _F32, ragged=ragged)

    def _sf(name, nh, s):
        # rowwise scale factors: one E8M0 per 32 d, rows padded to 128, blocks to 4 -- the SM100 suite's `_sf_dims` row form
        dims = (_B, nh, _ceil_to(s, 128), _ceil_to(_ceil_div(_D, 32), 4))
        stride = [dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1]
        t[name] = g.tensor(name=name, dim=list(dims), stride=stride, data_type=cudnn.data_type.FP8_E8M0, reordering_type=cudnn.tensor_reordering.F8_128x4)

    for name in _MXFP8_SF:
        _sf(name, hkv if name.startswith("descale_k") or name == "descale_v" else _H, s_kv if name.startswith("descale_k") or name == "descale_v" else s_q)
    kw = dict(name="mxfp8_bwd", attn_scale=_D**-0.5, **t)
    if ragged:
        kw.update(_lengths(g))
    if declare_totals:
        kw.update(max_total_seq_len_q=_T_Q, max_total_seq_len_kv=_T_KV)
    kw.update(sdpa_kwargs)
    outs = g.sdpa_mxfp8_backward(**kw)
    assert len(outs) == 6
    _finish_outputs(g, outs, ((_H, s_q), (hkv, s_kv), (hkv, s_kv)), _BF16, ragged=ragged)
    return g


_ROWS = {"fp8": (_fp8_backward_graph, "SDPA_FP8_BWD"), "mxfp8": (_mxfp8_backward_graph, "SDPA_MXFP8_BWD")}


# --------------------------------------------------------------------------- tier 1: the python graph + the analyzer (no device)


@pytest.mark.parametrize("row", sorted(_ROWS))
def test_declared_totals_are_analyzer_facts(row):
    """The ragged quantized backward node records the two totals and ``graph_analyzer`` reports them as facts -- what the
    engine rows that claim ``thd_declared_totals`` gate on."""
    from cudnn.sdpa import graph_analyzer as ga

    build, _ = _ROWS[row]
    g = build()
    (node,) = g.nodes
    assert node.params["max_total_seq_len_q"] == _T_Q and node.params["max_total_seq_len_kv"] == _T_KV
    facts = ga.analyze(g)
    assert facts is not None and facts.invalid is None, facts
    assert facts.is_backward and facts.thd
    assert (facts.is_fp8, facts.is_mxfp8) == ((True, False) if row == "fp8" else (False, True))
    assert facts.max_total_seq_len_q == _T_Q
    assert facts.max_total_seq_len_kv == _T_KV


@pytest.mark.parametrize("row", sorted(_ROWS))
def test_omitting_the_totals_keeps_the_old_facts(row):
    """Without the kwargs nothing changes: the node carries no such params and the facts read None (infer from geometry)."""
    from cudnn.sdpa import graph_analyzer as ga

    build, _ = _ROWS[row]
    g = build(declare_totals=False)
    (node,) = g.nodes
    assert "max_total_seq_len_q" not in node.params and "max_total_seq_len_kv" not in node.params
    facts = ga.analyze(g)
    assert facts is not None and facts.invalid is None, facts
    assert facts.thd and facts.max_total_seq_len_q is None and facts.max_total_seq_len_kv is None


@pytest.mark.parametrize("row", sorted(_ROWS))
def test_totals_on_a_dense_graph_are_refused_by_the_python_validator(row, monkeypatch):
    """The totals describe a packed layout; on a dense graph the SDPA family's python-native validator refuses them (the
    same rule the C++ node enforces, see ``test_dense_graph_with_totals_is_refused_by_the_node``).  That validator runs
    only when a python engine is on offer (the FROST rows, opt-in); otherwise validate() is the backend's eager C++ check,
    whose first verdict on a dense fp8 graph is arch- and version-dependent -- not this test's subject."""
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    build, _ = _ROWS[row]
    g = build(ragged=False)
    if g._python_native_validator() is None:
        pytest.skip("no python-native SDPA validator here (no python engine offered for this graph on this device)")
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="max_total_seq_len"):
        g.validate()


# --------------------------------------------------------------------------- tier 2: the pybind signature (no device)


@pytest.mark.parametrize("method", ["sdpa_fp8_backward", "sdpa_mxfp8_backward"])
def test_bindings_take_the_totals_as_trailing_keywords(method):
    """Append-only public signature: the two totals come LAST, after ``dSink_token``, defaulting to None -- positional callers
    across C++, pybind and the python wrappers keep working.  Read off the signature pybind writes into the docstring; a
    stale extension built before the attribute existed fails here, with the fix named."""
    doc = getattr(cudnn._pybind_module.backend_graph, method).__doc__ or ""
    signature = doc.splitlines()[0]
    for name in ("max_total_seq_len_q", "max_total_seq_len_kv"):
        assert (
            f"{name}: object = None" in signature
        ), f"{method} does not take {name}: the compiled extension predates the attribute -- rebuild python/cudnn/_compiled_module*.so\n{signature}"
    tail = signature.split("dSink_token", 1)[1]
    assert tail.index("max_total_seq_len_q") < tail.index("max_total_seq_len_kv") < tail.index(")"), signature
    # the half-precision backward binding is the precedent the quantized ones mirror
    assert "max_total_seq_len_q: object = None" in (cudnn._pybind_module.backend_graph.sdpa_backward.__doc__ or "").splitlines()[0]


# --------------------------------------------------------------------------- tier 3: the C++ node (lowering; needs a device)


def _lowered_node(g, tag):
    """Lower the python graph to the C++ one and return the JSON of its single SDPA node (the attribute's serialization)."""
    raw = g._lower_to_cpp()
    nodes = [n for n in json.loads(repr(raw))["nodes"] if n.get("tag") == tag]
    assert len(nodes) == 1, [n.get("tag") for n in json.loads(repr(raw))["nodes"]]
    return nodes[0]


@requires_cuda_device
@pytest.mark.parametrize("row", sorted(_ROWS))
def test_lowering_forwards_the_totals_to_the_node(row):
    """The binding forwards the kwargs to ``set_max_total_seq_len_q/kv`` and the attribute serializes them: the values the
    python graph recorded are the ones the C++ node prints."""
    build, tag = _ROWS[row]
    node = _lowered_node(build(), tag)
    assert node["max_total_seq_len_q"] == _T_Q
    assert node["max_total_seq_len_kv"] == _T_KV


@requires_cuda_device
@pytest.mark.parametrize("row", sorted(_ROWS))
def test_lowering_without_the_totals_leaves_them_unset(row):
    build, tag = _ROWS[row]
    node = _lowered_node(build(declare_totals=False), tag)
    assert node["max_total_seq_len_q"] is None and node["max_total_seq_len_kv"] is None


@pytest.mark.skipif(_cuda_cc() is None or _cuda_cc() < (9, 0), reason="the fp8 backward node refuses pre-Hopper devices before it reads the attribute")
@pytest.mark.parametrize("row", sorted(_ROWS))
def test_dense_graph_with_totals_is_refused_by_the_node(row):
    """The C++ node mirrors the half-precision backward: the totals are only valid on a packed (ragged) layout.  Run against
    the lowered graph directly so the verdict is the node's, not the python validator's."""
    build, _ = _ROWS[row]
    raw = build(ragged=False)._lower_to_cpp()
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="max_total_seq_len"):
        raw.validate()
