# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native binding keeps the hybrid's two SF inputs and BF16 V byte width."""

import sdpa_binding_reference as binding_reference

import pytest
import torch

import cudnn
from cudnn.sdpa.fwd import prepared as prep
from test_sdpa_native_mxfp8_binding import _fixture as _mx_fixture, _execute

pytestmark = [pytest.mark.L1]


def _fixture(d=128, amax=True):
    s, facts, frames, combined, roles = _mx_fixture(d=d)
    s.quant = s.quant._replace(sf_sizes=s.quant.sf_sizes[:2], has_amax=amax)
    s.expect["v"] = "bfloat16"
    s.elem_bytes["v"] = 2
    facts["v"] = facts["v"]._replace(dtype="bfloat16")
    facts.pop("sf_v")
    if not amax:
        facts.pop("amax_o", None)
    s.native = cudnn._pybind_module._SdpaDenseBinder(s)
    return s, facts, frames, combined, roles


@pytest.mark.L0
@pytest.mark.parametrize("d", [128, 192])
@pytest.mark.parametrize("amax", [False, True])
def test_native_pv_bf16_actual_host_frame(d, amax):
    s, facts, frames, combined, roles = _fixture(d, amax)
    for offset in (0, 2**33):
        fresh = {name: f._replace(ptr=f.ptr + offset) if f is not None else None for name, f in facts.items()}
        binding_reference.execute_quantized(s, fresh, 0x50000000 + offset, 17, 17)
        expected = frames.pop()
        _execute(s, fresh, roles, False, 0x50000000 + offset)
        actual = frames.pop()
        assert actual == expected
        assert actual[s.index["sf_v_ptr"]] is None
        assert actual[s.index["sf_tiles"]][2] == 0
        assert actual[s.index["scale_o_ptr"]] is None
        assert combined == []


@pytest.mark.parametrize("role", ["sf_q", "sf_k"])
@pytest.mark.parametrize("bad", ["missing", "device", "alignment", "span", "hole", "overlap", "workspace", "amax", "address"])
def test_native_pv_bf16_current_sf_rejected_before_writes(role, bad, monkeypatch):
    writes = []
    monkeypatch.setattr(prep._buffers, "fill_word_async", lambda *a: writes.append(a))
    s, facts, frames, combined, roles = _fixture()
    _execute(s, facts, roles, False)
    frames.clear()
    f = facts[role]
    changed = dict(
        missing=None,
        device=f._replace(device=(1, 0)),
        alignment=f._replace(ptr=f.ptr + 1),
        span=f._replace(span=f.span - 1),
        hole=f._replace(strides=(*f.strides[:-1], 2)),
        overlap=f._replace(strides=(f.strides[0], 0, *f.strides[2:])),
        workspace=f._replace(ptr=0x50000000),
        amax=f._replace(ptr=facts["amax_o"].ptr),
        address=f._replace(ptr=2**63 - 16),
    )[bad]
    with pytest.raises(ValueError):
        _execute(s, dict(facts, **{role: changed}), roles, False)
    assert frames == combined == writes == []


@pytest.mark.parametrize("bad", ["sf_v", "scale_o", "descale_v", "fp8_v", "v_span", "v_address"])
def test_native_pv_bf16_rejects_dead_operands_and_wrong_width(bad):
    s, facts, frames, combined, roles = _fixture()
    _execute(s, facts, roles, False)
    frames.clear()
    if bad in ("sf_v", "scale_o", "descale_v"):
        facts[bad] = facts["sf_k"] if bad == "sf_v" else facts["amax_o"]._replace(ptr=0x71000000)
    elif bad == "fp8_v":
        facts["v"] = facts["v"]._replace(dtype="float8_e4m3fn")
    elif bad == "v_span":
        facts["v"] = facts["v"]._replace(span=facts["v"].span - 1)
    else:
        facts["v"] = facts["v"]._replace(ptr=2**63 - 16)
    with pytest.raises(ValueError):
        _execute(s, facts, roles, False)
    assert frames == combined == []


@pytest.mark.parametrize("d", [128, 192])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize("amax", [False, True])
def test_native_pv_bf16_rebind_and_replay(d, dtype, stats, amax, monkeypatch):
    import test_sdpa_prepared_pv_bf16 as existing

    original = existing._case

    def native_case(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result[0]._dense_spec.native is not None
        return result

    monkeypatch.setattr(existing, "_case", native_case)
    monkeypatch.setattr(prep, "execute_quantized", lambda *a, **k: pytest.fail("PV-BF16 entered Python binding"), raising=False)
    # Retain the established independent dequantized numerical oracle, changed
    # SF_Q/SF_K, fresh strided BF16 V, nondefault stream and allocation guards.
    existing.test_pv_bf16_prepared_rebind_and_replay(d, dtype, stats, amax, monkeypatch)


@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("d", [128, 192])
@pytest.mark.parametrize("product", [False, True])
def test_native_pv_bf16_physical_v_int64(d, product):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("PV-BF16 requires an existing SM100/SM103 specialization")
    from test_sdpa_prepared_pv_bf16 import _case, _check, _poison
    from test_sdpa_prepared_fp8 import _cuda_graph

    b, compact = (5 if product else 2), 2 * 128 * 128
    stride = (2**30 if product else 2**32) + compact
    try:
        v = torch.empty_strided((b, 2, 128, 128), (stride, 128, 2 * 128, 1), device="cuda", dtype=torch.bfloat16)
    except torch.OutOfMemoryError:
        pytest.skip("physical BF16 V address test requires over 8 GiB")
    for i in range(b):
        v[i].fill_((i + 1) / 8)
    wrapped = ((b - 1) * stride) % 2**32
    decoy = torch.as_strided(v, (2, 128, 128), (128, 2 * 128, 1), storage_offset=wrapped)
    decoy.fill_(-0.75)
    assert (b - 1) * stride > 2**32
    api, bufs, ws, scales = _case(d=d, v=v, b=b)
    assert api._dense_spec.native is not None
    api.execute(**bufs, workspace=ws)
    _check(bufs, scales)
    with _cuda_graph() as graph:
        with torch.cuda.graph(graph):
            api.execute(**bufs, workspace=ws)
        v[-1].neg_()
        decoy.fill_(0.75)
        _poison(bufs)
        graph.replay()
        _check(bufs, scales)
    assert torch.all(decoy == 0.75)
