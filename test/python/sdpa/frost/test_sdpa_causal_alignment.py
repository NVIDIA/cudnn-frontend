# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conflicting causal spellings must never silently change a non-square mask (#2299).

These are declaration checks: all six SDPA APIs, both Python and compiled
builders, and the shared facts consumed by every architecture's kernels.
"""

import json

import pytest

import cudnn

pytestmark = pytest.mark.L0
_TL = cudnn.diagonal_alignment.TOP_LEFT
_BR = cudnn.diagonal_alignment.BOTTOM_RIGHT
_CONFLICT = "use_causal_mask.*diagonal_alignment"
_CASES = [
    pytest.param(api, dtype, id=f"{api}-{dtype.name.lower()}")
    for api, dtypes in (
        ("sdpa", (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16)),
        ("sdpa_backward", (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16)),
        ("sdpa_fp8", (cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2)),
        ("sdpa_fp8_backward", (cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2)),
        ("sdpa_mxfp8", (cudnn.data_type.FP8_E4M3,)),
        ("sdpa_mxfp8_backward", (cudnn.data_type.FP8_E4M3,)),
    )
    for dtype in dtypes
]


@pytest.fixture(autouse=True)
def _synthetic_device(monkeypatch):
    from cudnn.sdpa import graph_analyzer as ga

    monkeypatch.setattr(ga, "_device_cc", lambda: (10, 0))
    monkeypatch.setattr(ga, "_device_sm_count", lambda: 148)


def _graph(api, dtype, *, compiled=False, **mask):
    graph_type = cudnn._pybind_module.backend_graph if compiled else cudnn.pygraph
    g = graph_type(io_data_type=dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, sm_version=100)
    backward, mx = api.endswith("backward"), "mxfp8" in api
    b, h, sq, sk, d = 1, 2, 16, 64, 256

    def tensor(name, seq=sq, width=d, dt=dtype, **kwargs):
        return g.tensor(name=name, dim=[b, h, seq, width], stride=[seq * h * width, width, h * width, 1], data_type=dt, **kwargs)

    def scalar(name):
        return g.tensor(name=name, dim=[1] * 4, stride=[1] * 4, data_type=cudnn.data_type.FLOAT)

    inputs = dict(q=tensor("q"), k=tensor("k", sk), v=tensor("v", sk))
    kwargs = dict(attn_scale=1.0 / 16, **mask)
    if backward:
        inputs["stats"] = g.tensor(name="stats", dim=[b, h, sq, 1], stride=[h * sq, sq, 1, 1], data_type=cudnn.data_type.FLOAT)
        inputs["dO"] = tensor("dO")
        if mx:
            inputs.update(q_T=tensor("q_T"), k_T=tensor("k_T", sk), dO_T=tensor("dO_T"))
            inputs.update(o_f16=tensor("o_f16", dt=cudnn.data_type.BFLOAT16), dO_f16=tensor("dO_f16", dt=cudnn.data_type.BFLOAT16))
        else:
            inputs["o"] = tensor("o")
    else:
        kwargs["generate_stats"] = True
    if "fp8" in api and not mx:
        scale_names = (
            (
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
            if backward
            else ("descale_q", "descale_k", "descale_v", "descale_s", "scale_s", "scale_o")
        )
        inputs.update({name: scalar(name) for name in scale_names})
    if mx:
        sf_names = ("q", "q_T", "k", "k_T", "v", "dO", "dO_T") if backward else ("q", "k", "v")
        for port in sf_names:
            seq = sk if port.startswith("k") or port == "v" else sq
            # Forward V scales the sequence axis; backward V uses rowwise blocks.
            sf_seq, sf_width = ((seq + 31) // 32, d) if port == "v" and not backward else (seq, (d + 31) // 32)
            inputs["descale_" + port] = tensor("descale_" + port, sf_seq, sf_width, cudnn.data_type.FP8_E8M0, reordering_type=cudnn.tensor_reordering.F8_128x4)
    outputs = getattr(g, api)(**inputs, **kwargs)
    for i, out in enumerate(outputs):
        if out is None:
            continue
        if (backward and i < 3) or (not backward and i == 0):
            seq = sk if backward and i in (1, 2) else sq
            out.set_dim([b, h, seq, d]).set_stride([seq * h * d, d, h * d, 1])
            out.set_data_type(cudnn.data_type.BFLOAT16 if mx else dtype).set_output(True)
        elif not backward and i == 1:
            out.set_dim([b, h, sq, 1]).set_stride([h * sq, sq, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
        else:
            out.set_dim([1] * 4).set_stride([1] * 4).set_data_type(cudnn.data_type.FLOAT)
    return g


@pytest.mark.parametrize("api,dtype", _CASES)
@pytest.mark.parametrize("python_candidate", [False, True], ids=["backend", "frost"])
def test_python_validation_rejects_conflicting_alignment(api, dtype, python_candidate, monkeypatch):
    from cudnn.engines import manifest

    monkeypatch.setattr(manifest, "engines_for", lambda graph: [object()] if python_candidate else [])
    g = _graph(api, dtype, use_causal_mask=True, diagonal_alignment=_BR)
    # Lower actual descriptors and call the compiled SDPA builder without a
    # device handle: this invalid declaration must fail before C++ validate.
    monkeypatch.setattr(g, "_backend_handle_for_lowering", lambda kwargs: None)
    for validate in (g.validate, g.build_operation_graph):
        with pytest.raises(RuntimeError, match=_CONFLICT):
            validate()
        assert not g._is_validated and g._lowered_graph is None


@pytest.mark.parametrize("api,dtype", _CASES)
def test_compiled_builder_rejects_conflicting_alignment(api, dtype):
    # Catch it before a backend plan, CUDA context or kernel can be constructed.
    with pytest.raises(RuntimeError, match=_CONFLICT):
        _graph(api, dtype, compiled=True, use_causal_mask=True, diagonal_alignment=_BR)


@pytest.mark.parametrize("cc", [(8, 0), (9, 0), (10, 0), (10, 3), (10, 7), (12, 0), (12, 1)])
@pytest.mark.parametrize("api,dtype", _CASES)
def test_every_architecture_sees_the_invalid_mask(api, dtype, cc, monkeypatch):
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd import engines as bwd
    from cudnn.sdpa.fwd import engines as fwd

    monkeypatch.setattr(ga, "_device_cc", lambda: cc)
    control = ga.analyze(_graph(api, dtype, use_causal_mask=True, diagonal_alignment=_TL))
    assert control is not None and control.invalid is None and control.device_cc == cc
    g = _graph(api, dtype, use_causal_mask=True, diagonal_alignment=_BR)
    facts = ga.analyze(g)
    assert facts is not None and facts.invalid is not None
    assert "use_causal_mask" in facts.invalid and "diagonal_alignment" in facts.invalid
    family = bwd if api.endswith("backward") else fwd
    for spec in family.ENGINE_SPECS:
        assert family.mismatch(spec.capabilities, facts) == facts.invalid, spec.name


@pytest.mark.parametrize("api,dtype", _CASES)
@pytest.mark.parametrize(
    "mask,causal,bottom_right",
    [
        ({"use_causal_mask": True}, True, False),
        ({"use_causal_mask": True, "diagonal_alignment": _TL}, True, False),
        ({"use_causal_mask_bottom_right": True}, True, True),
        ({"use_causal_mask_bottom_right": True, "diagonal_alignment": _BR}, True, True),
        ({"diagonal_band_right_bound": 0, "diagonal_alignment": _BR}, True, True),
        ({"use_causal_mask": True, "diagonal_band_left_bound": 8}, True, False),
        ({"use_causal_mask_bottom_right": True, "diagonal_band_left_bound": 8}, True, True),
        ({"use_causal_mask": False, "diagonal_alignment": _BR}, False, False),
    ],
    ids=[
        "legacy-top-left",
        "explicit-top-left",
        "legacy-bottom-right",
        "explicit-bottom-right",
        "band-bottom-right",
        "top-left-window",
        "bottom-right-window",
        "unmasked",
    ],
)
def test_legal_mask_spellings_keep_their_meaning(api, dtype, mask, causal, bottom_right, monkeypatch):
    from cudnn.engines import manifest
    from cudnn.sdpa import graph_analyzer as ga

    mask = dict(mask)
    if api in ("sdpa_fp8", "sdpa_fp8_backward", "sdpa_mxfp8_backward"):
        for side in ("left", "right"):
            if f"diagonal_band_{side}_bound" in mask:
                mask[f"{side}_bound"] = mask.pop(f"diagonal_band_{side}_bound")
    monkeypatch.setattr(manifest, "engines_for", lambda graph: [object()])
    monkeypatch.setattr(ga, "_device_cc", lambda: (10, 7))
    g = _graph(api, dtype, **mask)
    g.validate()
    facts = ga.analyze(g)
    assert facts is not None and facts.invalid is None, getattr(facts, "invalid", None)
    assert (facts.causal, facts.bottom_right) == (causal, bottom_right)
    compiled = _graph(api, dtype, compiled=True, **mask)
    (node,) = json.loads(str(compiled))["nodes"]
    assert node.get("right_bound") == (0 if causal else None)
    if causal:
        assert node["diagonal_alignment"] == ("BOTTOM_RIGHT" if bottom_right else "TOP_LEFT")
    assert node.get("left_bound") == mask.get("diagonal_band_left_bound", mask.get("left_bound"))
