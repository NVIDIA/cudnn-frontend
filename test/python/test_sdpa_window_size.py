# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""window_size=(left, right): the FlashAttention-convention alias for the SDPA
diagonal band. Each side is an OFFSET from the (aligned) diagonal, -1 or None =
unbounded; equivalent to diagonal_band_left_bound = left + 1 and
diagonal_band_right_bound = right. These tests pin the alias to the explicit
band spellings: same facts at the analyzer, bit-identical O at execute, and a
lowering-time error for every disallowed combination."""

import math

import cudnn
import pytest
import torch

pytestmark = pytest.mark.L0

B, H, S, D = 2, 4, 256, 128


def _build_graph(**kw):
    g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    dims, strides = (B, H, S, D), (S * H * D, D, H * D, 1)
    q = g.tensor(dim=dims, stride=strides, data_type=cudnn.data_type.HALF, name="q")
    k = g.tensor(dim=dims, stride=strides, data_type=cudnn.data_type.HALF, name="k")
    v = g.tensor(dim=dims, stride=strides, data_type=cudnn.data_type.HALF, name="v")
    o, _ = g.sdpa(name="sdpa", q=q, k=k, v=v, attn_scale=0.1, generate_stats=False, **kw)
    o.set_output(True).set_dim(dims).set_stride(strides)
    o.set_data_type(cudnn.data_type.HALF)
    return g


def _band_facts(**kw):
    """(causal, bottom_right, window_left, right_bound) facts the FROST analyzer resolves."""
    from cudnn.sdpa import graph_analyzer as ga

    f = ga.analyze(_build_graph(**kw))
    assert f is not None and f.invalid is None, getattr(f, "invalid", "no sdpa node")
    return (f.causal, f.bottom_right, f.window_left, f.right_bound)


_EQUIVALENT_SPELLINGS = [
    (dict(window_size=(63, 0)), dict(sliding_window_length=64, use_causal_mask=True)),
    (dict(window_size=(-1, 0)), dict(use_causal_mask=True)),
    (dict(window_size=(-1, 40)), dict(diagonal_band_right_bound=40)),
    (dict(window_size=(31, 78)), dict(diagonal_band_left_bound=32, diagonal_band_right_bound=78)),
    (dict(window_size=(31, -1)), dict(diagonal_band_left_bound=32)),
    (dict(window_size=(None, 0), diagonal_alignment=cudnn.diagonal_alignment.BOTTOM_RIGHT), dict(use_causal_mask_bottom_right=True)),
]
_EQUIVALENT_IDS = ["swa-causal", "causal", "right-widen", "band", "swa-only", "causal-br"]


@pytest.mark.parametrize("kw_window,kw_explicit", _EQUIVALENT_SPELLINGS, ids=_EQUIVALENT_IDS)
def test_window_size_facts_match_explicit_band(kw_window, kw_explicit):
    assert _band_facts(**kw_window) == _band_facts(**kw_explicit)


def test_window_size_list_form():
    assert _band_facts(window_size=[31, 78]) == _band_facts(window_size=(31, 78))


def _run_gpu(**kw):
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)
    o = torch.empty(B, S, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)
    scale = 1.0 / math.sqrt(D)
    g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tq, tk, tv = g.tensor_like(q), g.tensor_like(k), g.tensor_like(v)
    to, _ = g.sdpa(name="sdpa", q=tq, k=tk, v=tv, generate_stats=False, attn_scale=scale, **kw)
    to.set_output(True).set_dim(o.shape).set_stride(o.stride())
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    g.check_support()
    g.build_plans()
    g.execute({tq: q, tk: k, tv: v, to: o}, torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8))
    torch.cuda.synchronize()
    return o


@pytest.mark.parametrize("kw_window,kw_explicit", _EQUIVALENT_SPELLINGS, ids=_EQUIVALENT_IDS)
def test_window_size_execute_matches_explicit_band(cudnn_handle, kw_window, kw_explicit):
    assert torch.equal(_run_gpu(**kw_window), _run_gpu(**kw_explicit))


@pytest.mark.parametrize(
    "kw,fragment",
    [
        (dict(window_size=(1, 0), use_causal_mask=True), "cannot be combined"),
        (dict(window_size=(1, 0), use_causal_mask_bottom_right=True), "cannot be combined"),
        (dict(window_size=(1, 0), sliding_window_length=2), "cannot be combined"),
        (dict(window_size=(1, 0), diagonal_band_left_bound=2), "cannot be combined"),
        (dict(window_size=(1, 0), diagonal_band_right_bound=0), "cannot be combined"),
        (dict(window_size=(1, 2, 3)), "exactly two"),
        (dict(window_size=(-2, 0)), ">= 0"),
        (dict(window_size=(0, -3)), ">= 0"),
        (dict(window_size=5), "tuple or list"),
        (dict(window_size=(0.5, 0)), "int or None"),
    ],
    ids=[
        "with-causal",
        "with-causal-br",
        "with-sliding-window",
        "with-left-bound",
        "with-right-bound",
        "three-entries",
        "negative-left",
        "negative-right",
        "not-a-pair",
        "float-entry",
    ],
)
def test_window_size_rejects_bad_usage(kw, fragment):
    """The pybind layer validates at lowering (build_operation_graph)."""
    g = _build_graph(**kw)
    with pytest.raises(Exception, match=fragment):
        g.validate()
        g.build_operation_graph()
