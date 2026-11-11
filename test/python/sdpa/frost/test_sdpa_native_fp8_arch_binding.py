# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""SM107 and SM120 per-tensor FP8 retain their own partial dtype and host ABI."""

import sdpa_binding_reference as binding_reference

import pytest
import torch

import cudnn
from cudnn.sdpa.fwd import prepared as prep
from test_sdpa_native_fp8_binding import (
    _fixture,
    _native_pack,
    _INDICES,
    test_native_fp8_graph_rebinds_scales_and_replays as _graph,
    test_native_fp8_physical_wide_operand_address as _wide,
)

pytestmark = [pytest.mark.L1]
_WIDTHS = {"sm107": [(64, 64), (128, 128), (192, 128), (256, 256), (512, 512)], "sm120": [(48, 80), (128, 128), (240, 256), (384, 320), (512, 512)]}
_CASES = [(arch, d, dv, split) for arch, widths in _WIDTHS.items() for d, dv in widths for split in ([1, 4] if arch == "sm120" or d in (128, 192) else [1])]


def _device(arch):
    from frost_test_utils import _dsl_installed

    cc = ((10, 7),) if arch == "sm107" else ((12, 0), (12, 1))
    if torch.cuda.get_device_capability() not in cc or not _dsl_installed():
        pytest.skip(f"{arch} and supported CuTe DSL required")
    return cc


@pytest.mark.parametrize("arch,d,dv,split", _CASES)
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("output", ["float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"])
def test_fp8_arch_actual_host_frame_and_partial_width(arch, d, dv, split, dtype, output):
    s, facts, frames, combined = _fixture(d, dv, split, dtype, output, arch=arch)
    for delta in (0, 2**33):
        fresh = {r: f._replace(ptr=f.ptr + delta) for r, f in facts.items()}
        workspace = 0x50000000 + delta
        binding_reference.execute_quantized(s, fresh, workspace, 17, 17)
        expected, tail = frames.pop(), combined.pop() if split > 1 else ()
        actual, actual_tail, identity = s.native.bind_quantized(_native_pack(fresh), _INDICES, workspace, 17)
        assert tuple(actual) == expected and tuple(actual_tail) == tail and identity == 0
        if split > 1:
            width = 2 if arch == "sm120" else 4
            assert s.combine.lse_offset == split * s.b * s.qh * s.s_q_max * dv * width
            assert actual_tail[1] == workspace + s.combine.lse_offset
            if output.startswith("float8"):
                assert actual[s.index["scale_o_ptr"]] is None
                assert actual_tail[-2] == fresh["scale_o"].ptr
        s.native.execute(_native_pack(fresh), _INDICES, 17, workspace=workspace)
        assert frames.pop() == expected
        if split > 1:
            assert combined.pop() == tail


@pytest.mark.parametrize("output", ["float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("bad", ["partial_dtype", "partial_offset", "scalar_offset", "overflow"])
def test_fp8_half_partial_contract_rejected_before_entry(output, bad):
    s, _, frames, combined = _fixture(128, 128, 4, "float8_e4m3fn", output, arch="sm120")
    if bad == "partial_dtype":
        s.expect["o"] = "float32"
    elif bad == "partial_offset":
        s.combine = s.combine._replace(lse_offset=s.combine.lse_offset - 16)
    elif bad == "scalar_offset":
        s.quant = s.quant._replace(scratch_offset=s.quant.scratch_offset - 16)
    else:
        s.b = 2**62
    with pytest.raises(ValueError):
        cudnn._pybind_module._SdpaDenseBinder(s)
    assert frames == combined == []


@pytest.mark.parametrize("arch,d,dv,split", _CASES)
@pytest.mark.parametrize("output", [torch.bfloat16, torch.float8_e4m3fn])
def test_fp8_arch_graph_rebinding_and_changed_scale_replay(arch, d, dv, split, output, monkeypatch):
    _graph(d, dv, split, output, monkeypatch, arch=arch, cc=_device(arch))


@pytest.mark.L0
@pytest.mark.parametrize("arch", ["sm107", "sm120"])
def test_fp8_arch_default_smoke(arch, monkeypatch):
    _graph(128, 128, 4, torch.float8_e4m3fn, monkeypatch, arch=arch, cc=_device(arch))


@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("arch", ["sm107", "sm120"])
@pytest.mark.parametrize("role", ["q", "k", "v", "o"])
@pytest.mark.parametrize("product", [False, True])
def test_fp8_arch_physical_wide_operand(arch, role, product, monkeypatch):
    _wide(role, product, monkeypatch, arch=arch, cc=_device(arch))


@pytest.mark.parametrize("arch,d,dv", [("sm120", d, dv) for d, dv in _WIDTHS["sm120"]] + [("sm107", 128, 128), ("sm107", 192, 128)])
@pytest.mark.parametrize("output", [torch.bfloat16, torch.float8_e4m3fn])
def test_fp8_arch_standalone_omitted_scales_replay(arch, d, dv, output, monkeypatch):
    from test_sdpa_prepared_fp8_split import test_prepared_fp8_split_standalone_default_scales as check

    _device(arch)
    monkeypatch.setattr(prep, "execute_quantized", lambda *a, **k: pytest.fail("native FP8 entered Python binder"), raising=False)
    check(arch, d, dv, output)
