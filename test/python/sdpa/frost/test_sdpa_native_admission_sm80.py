# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM80 serves accepted declarations natively, or declines before compilation."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import cudnn
from cudnn.api_base import TensorDesc
from cudnn.sdpa import graph_analyzer as ga
from cudnn.sdpa.bwd import engines as backward
from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm80
from cudnn.sdpa.fwd import engines as forward
from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm80

pytestmark = pytest.mark.L0
DATA_ROLES = ("q", "k", "v", "o", "do", "dq", "dk", "dv")


@pytest.fixture(autouse=True)
def declared_sm80(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)


def _descs(d=96, dv=80, *, bad_role=None, thd=False, singleton=False):
    descs = {}
    for role in DATA_ROLES:
        width = d if role in ("q", "k", "dq", "dk") else dv
        shape = (1 if singleton else 2, 1 if singleton else 4, 32, width)
        hs = width + 8
        ts = shape[1] * hs + 16
        stride = (7 if thd or singleton else shape[2] * ts + 8, 3 if singleton else hs, ts, 1)
        if role == bad_role:
            stride = (shape[2] * (ts + 1) + 8, hs, ts + 1, 1)
        descs[role] = TensorDesc(dtype=torch.bfloat16, shape=shape, stride=stride, stride_order=(3, 1, 2, 0), device=torch.device("cuda:0"), name=role)
    b, h, s, _ = descs["q"].shape
    descs["stats"] = TensorDesc(dtype=torch.float32, shape=(b, h, s), stride=(h * s, s, 1), stride_order=(2, 1, 0), device=torch.device("cuda:0"), name="stats")
    return descs


def _api(direction, descs, **kwargs):
    if direction == "bwd":
        return SdpaBwdDslSm80(**{"sample_" + name: desc for name, desc in descs.items()}, **kwargs)
    return SdpaFwdDslSm80(**{"sample_" + name: descs[name] for name in ("q", "k", "v", "o")}, sample_lse=descs["stats"], **kwargs)


def _facts(direction, descs, *, thd=False):
    def ref(desc):
        return SimpleNamespace(get_dim=lambda: desc.shape, get_stride=lambda: desc.stride, get_data_type=lambda: cudnn.data_type.BFLOAT16)

    q, k, v = (descs[n] for n in ("q", "k", "v"))
    return ga.SdpaGraphFacts(
        b=q.shape[0],
        h_q=q.shape[1],
        h_kv=k.shape[1],
        s_q=q.shape[2],
        s_kv=k.shape[2],
        d_qk=q.shape[3],
        d_v=v.shape[3],
        dtype=cudnn.data_type.BFLOAT16,
        device_cc=(8, 0),
        is_backward=direction == "bwd",
        scale=0.125,
        port_layouts=tuple((n, descs[n].shape, descs[n].stride) for n in DATA_ROLES) if direction == "bwd" else (),
        **{n + "_t": ref(descs[n]) for n in ("q", "k", "v", "o")},
        thd=thd,
        padded=thd,
        max_total_seq_len_q=64 if thd else None,
        max_total_seq_len_kv=64 if thd else None,
    )


def _mismatch(direction, facts):
    module = backward if direction == "bwd" else forward
    return module.mismatch(module._sm80_spec().capabilities, facts)


@pytest.mark.parametrize("direction,role", [("fwd", n) for n in DATA_ROLES[:4]] + [("bwd", n) for n in DATA_ROLES])
def test_unaligned_data_declines_before_compile(direction, role):
    descs = _descs(bad_role=role)
    api = _api(direction, descs)
    with pytest.raises(NotImplementedError, match=rf"{role}.*stride"):
        api.check_support()
    reason = _mismatch(direction, _facts(direction, descs))
    assert reason is not None and role.lower() in reason.lower() and "stride" in reason.lower()


@pytest.mark.parametrize("direction,role", [("fwd", n) for n in DATA_ROLES[:4]] + [("bwd", n) for n in DATA_ROLES])
def test_graph_declines_each_unaligned_port(direction, role):
    reason = _mismatch(direction, _facts(direction, _descs(bad_role=role)))
    assert reason is not None and role.lower() in reason.lower() and "stride" in reason.lower()


@pytest.mark.parametrize("direction", ["fwd", "bwd"])
@pytest.mark.parametrize("d,dv", [(94, 80), (96, 78)])
def test_unaligned_width_declines(direction, d, dv):
    descs = _descs(d, dv)
    with pytest.raises(NotImplementedError, match="multiple.*8"):
        _api(direction, descs).check_support()
    assert "8" in _mismatch(direction, _facts(direction, descs))


@pytest.mark.parametrize("direction", ["fwd", "bwd"])
def test_rope_declines_before_compile(direction):
    kwargs = dict(rope_max_s=32)
    if direction == "bwd":
        kwargs["has_rope"] = True
    with pytest.raises(NotImplementedError, match="RoPE"):
        _api(direction, _descs(128, 128), **kwargs).check_support()


@pytest.mark.parametrize("direction", ["fwd", "bwd"])
@pytest.mark.parametrize("d,dv", [(16, 256), (48, 32), (64, 96), (96, 80), (128, 128), (160, 112), (192, 128), (224, 208), (256, 256)])
@pytest.mark.parametrize("singleton", [False, True])
def test_aligned_native_envelopes_remain_supported(direction, d, dv, singleton):
    descs = _descs(d, dv, singleton=singleton)
    api = _api(direction, descs)
    assert api.check_support()
    assert _mismatch(direction, _facts(direction, descs)) is None
    if direction == "bwd":
        assert api._native_pointer and api._staged_layout is None


@pytest.mark.parametrize("d,dv", [(16, 256), (64, 96), (96, 80), (192, 128)])
def test_backward_thd_ignores_unused_batch_stride(d, dv):
    descs = _descs(d, dv, thd=True)
    api = _api("bwd", descs, thd=True, max_total_seq_len_q=64, max_total_seq_len_kv=64)
    assert api.check_support()
    assert api._native_pointer and api._staged_layout is None
    assert _mismatch("bwd", _facts("bwd", descs, thd=True)) is None


def _forward_thd_case(d, dv, *, singleton=False):
    descs = _descs(d, dv, thd=True, singleton=singleton)
    b, h, s, _ = descs["q"].shape
    descs["stats"] = TensorDesc(dtype=torch.float32, shape=(b, h, s), stride=(7, 64, 1), stride_order=(2, 1, 0), device=torch.device("cuda:0"), name="stats")
    facts = replace(_facts("fwd", descs, thd=True), seq_q_t=SimpleNamespace(), seq_kv_t=SimpleNamespace())
    return descs, facts


@pytest.mark.parametrize("d,dv", [(48, 32), (64, 64), (64, 96), (96, 80), (128, 128), (192, 112), (192, 128), (224, 208), (256, 256)])
@pytest.mark.parametrize("singleton", [False, True])
def test_forward_thd_native_pitches(d, dv, singleton):
    descs, facts = _forward_thd_case(d, dv, singleton=singleton)
    api = _api("fwd", descs, thd=True, max_total_seq_len_q=64, max_total_seq_len_kv=64, seq_kv_lens_present=True)
    assert api.check_support()
    assert _mismatch("fwd", facts) is None


@pytest.mark.parametrize("role", ["q", "k", "v", "o"])
@pytest.mark.parametrize("bad_stride", [(7, 128, 128, 1), (7, 120, 544, 1), (7, 136, 561, 1), (7, 136, 560, 2)])
def test_forward_thd_invalid_pitches_decline(role, bad_stride):
    descs, _ = _forward_thd_case(128, 128)
    descs[role] = TensorDesc(
        dtype=torch.bfloat16, shape=descs[role].shape, stride=bad_stride, stride_order=(3, 1, 2, 0), device=torch.device("cuda:0"), name=role
    )
    facts = replace(_facts("fwd", descs, thd=True), seq_q_t=SimpleNamespace(), seq_kv_t=SimpleNamespace())
    with pytest.raises(NotImplementedError, match=rf"(?i){role}.*stride"):
        _api("fwd", descs, thd=True, max_total_seq_len_q=64, max_total_seq_len_kv=64, seq_kv_lens_present=True).check_support()
    reason = _mismatch("fwd", facts)
    assert reason is not None and role.lower() in reason.lower() and "stride" in reason.lower()


@pytest.mark.parametrize("missing", ["bias", "dbias", "dsink"])
def test_backward_incomplete_auxiliary_declarations_decline(missing):
    descs = _descs()
    b, h, s, _ = descs["q"].shape
    aux = {
        n: TensorDesc(dtype=torch.float32, shape=shape, stride=stride, stride_order=tuple(reversed(range(len(shape)))), device=torch.device("cuda:0"), name=n)
        for n, shape, stride in [
            ("bias", (1, h, s, s), (h * s * s, s * s, s, 1)),
            ("dbias", (1, h, s, s), (h * s * s, s * s, s, 1)),
            ("sink", (h,), (1,)),
            ("dsink", (h,), (1,)),
        ]
    }
    kwargs = {"sample_" + n: desc for n, desc in aux.items() if n != missing}
    with pytest.raises(NotImplementedError, match=missing):
        _api("bwd", descs, has_bias=True, bias_is_fp32=True, **kwargs).check_support()
