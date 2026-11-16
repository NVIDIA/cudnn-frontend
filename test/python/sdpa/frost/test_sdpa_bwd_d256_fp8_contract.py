# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph and standalone declaration agreement without allocating large tensors."""

from dataclasses import replace

import cudnn
import pytest
import torch

from cudnn.api_base import TensorDesc
from d256_fp8_bwd_helpers import DTYPES, NAME
from frost_test_utils import requires_dsl
from test_sdpa_bwd_fp8_sm107 import _graph_and_ports

pytestmark = [pytest.mark.L0, requires_dsl]


def verdicts(monkeypatch, *, cc=(10, 0), version=("nvidia-cutlass-dsl", "4.8.0"), **kwargs):
    from cudnn.frost import buffers, device
    from cudnn.sdpa.bwd import engines
    from cudnn.sdpa.bwd.kernels.sm100.d256_fp8.host import SdpaBwdDslSm100D256Fp8

    monkeypatch.setattr(engines, "cutedsl_state", lambda: (True, version))
    monkeypatch.setattr(engines, "cutedsl_arch_requirement_error", lambda cc: None)
    monkeypatch.setattr(buffers, "cutedsl_state", lambda: (True, version))
    monkeypatch.setattr(buffers, "cutedsl_arch_requirement_error", lambda cc: None)
    monkeypatch.setattr(device, "compute_capability", lambda dev: cc)
    options = dict(b=1, h=2, s=17, d=256, hkv=1, skv=33, causal=False)
    options.update(kwargs)
    graph, ports = _graph_and_ports(**options)
    facts = replace(graph._facts_for(engines.ga.analyze), device_cc=cc)
    spec = next(s for s in engines.ENGINE_SPECS if s.name == NAME)
    graph_reason = engines.mismatch(spec.capabilities, facts)
    dtype_of = {v: k for k, v in DTYPES.items()}
    dtype_of[cudnn.data_type.FLOAT] = torch.float32

    def descriptor(name):
        t = ports[name]
        shape, stride = tuple(t.get_dim()), tuple(t.get_stride())
        return TensorDesc(dtype_of[t.get_data_type()], shape, stride, TensorDesc._compute_stride_order(shape, stride), torch.device("cuda", 0))

    try:
        api = SdpaBwdDslSm100D256Fp8(
            *[descriptor(n) for n in ("q", "k", "v", "o", "dO", "stats", "dQ", "dK", "dV")],
            scale_softmax=None if options.get("scale", "default") == "default" else options.get("scale"),
            is_causal=options["causal"],
            window_size_left=None if options.get("left_bound") is None else options["left_bound"] - 1,
            causal_bottom_right=options.get("bottom_right", False),
            window_size_right=options.get("right_bound"),
        )
        api.check_support()
    except (NotImplementedError, ValueError) as exc:
        return graph_reason, str(exc)
    return graph_reason, None


@pytest.mark.parametrize("cc", [(10, 0), (10, 3)], ids=["sm100", "sm103"])
@pytest.mark.parametrize("itype", [cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2], ids=["e4m3", "e5m2"])
@pytest.mark.parametrize("otype", list(DTYPES.values()), ids=["e4m3-grad", "e5m2-grad", "fp16-grad", "bf16-grad"])
def test_graph_direct_dtype_agreement(monkeypatch, cc, itype, otype):
    assert verdicts(monkeypatch, cc=cc, fp8=itype, grad_dtypes=(otype,) * 3) == (None, None)


@pytest.mark.parametrize("cc", [(8, 0), (9, 0), (10, 1), (10, 7), (12, 0)], ids=["sm80", "sm90", "sm101", "sm107", "sm120"])
def test_other_architectures_decline(monkeypatch, cc):
    assert all(verdicts(monkeypatch, cc=cc))


@pytest.mark.parametrize("version", [None, ("nvidia-cutlass-dsl", "4.7.0"), ("another-dsl", "4.8.0")])
def test_dsl_floor_is_shared(monkeypatch, version):
    assert all(verdicts(monkeypatch, version=version))


@pytest.mark.parametrize(
    "allowed,kwargs",
    [
        (True, dict(b=65535, h=1, hkv=1, s=1)),
        (False, dict(b=65536, h=1, hkv=1, s=1)),
        (True, dict(h=65535, hkv=65535, s=1, skv=32768)),
        (False, dict(h=65536, hkv=65536, s=1)),
        (False, dict(h=65535, hkv=65535, s=1, skv=32769)),
        (True, dict(b=65535, h=256, hkv=256, s=128, skv=1)),
        (False, dict(b=65535, h=257, hkv=257, s=128, skv=1)),
        (True, dict(h=1, hkv=1, s=1, skv=2**31 - 128, left_bound=1)),
        (False, dict(h=1, hkv=1, s=1, skv=2**31 - 128, left_bound=2)),
        (False, dict(h=1, hkv=1, s=1, skv=2**31 - 127)),
        (False, dict(scale=0)),
        (False, dict(scale=-0.0625)),
        (False, dict(scale=float("inf"))),
        (False, dict(scale=float("nan"))),
        (False, dict(scale=1e-50)),
        (False, dict(scale=1e39)),
        (True, dict(scale=None)),
        (False, dict(d=128)),
        (False, dict(d_v=128)),
        (False, dict(bottom_right=True)),
        (False, dict(right_bound=8)),
        (False, dict(o_dtype=cudnn.data_type.FP8_E5M2)),
        (False, dict(grad_dtypes=(cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, cudnn.data_type.HALF))),
    ],
    ids=[
        "batch-edge",
        "batch-over",
        "stride-edge",
        "head-over",
        "stride-over",
        "delta-edge",
        "delta-over",
        "window-edge",
        "window-over",
        "sequence-over",
        "zero-scale",
        "negative-scale",
        "inf-scale",
        "nan-scale",
        "fp32-underflow",
        "fp32-overflow",
        "default-scale",
        "wrong-qk-width",
        "wrong-v-width",
        "bottom-right",
        "right-band",
        "mixed-input",
        "mixed-grad",
    ],
)
def test_graph_direct_bounds_agree(monkeypatch, allowed, kwargs):
    reasons = verdicts(monkeypatch, **kwargs)
    assert tuple(r is None for r in reasons) == (allowed, allowed), reasons


def test_new_row_trails_backend_and_neighbors_unchanged(monkeypatch):
    from cudnn.engines.heuristics import BACKEND, LAST_RESORT
    from cudnn.sdpa.bwd import engines, heuristics

    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    graph, _ = _graph_and_ports(1, 2, 17, 256)
    facts = replace(graph._facts_for(engines.ga.analyze), device_cc=(10, 0))
    plans = heuristics.propose("A", facts, {NAME: 20637})
    assert plans[:2] == [BACKEND, LAST_RESORT]
    assert [p.engine_id for p in plans[2:]] == [20637]
    # This addition must not broaden any existing FP8 owner to D256/SM100.
    for spec in engines.ENGINE_SPECS:
        if spec.name != NAME and spec.capabilities.is_fp8:
            assert engines.mismatch(spec.capabilities, facts) is not None, spec.name
