# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small-head/window backward: numerical boundaries, fresh bindings and replay."""

import cudnn
import pytest
import torch

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
from test_sdpa_bwd_d128_sm100 import _case, _check_case
from test_sdpa_bwd_dsl_sm107 import _reference64

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]


@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "d,causal,left,sq,sk",
    [
        (64, False, None, 65, 129),
        (64, True, None, 257, 193),
        (64, True, 1, 65, 129),
        (64, True, 2, 1, 2),
        (64, True, 128, 257, 385),
        (64, True, 129, 65, 129),
        (64, False, 1, 65, 129),
        (64, False, 65, 193, 257),
        (128, True, 128, 257, 385),
        (128, False, 257, 193, 257),
    ],
)
def test_small_swa_rebind_replay(dt, d, causal, left, sq, sk, cudnn_handle):
    original_stream = cudnn.get_stream(cudnn_handle)
    capture = None
    try:
        case = _case(dt=dt, d=d, causal=causal, left=left, b=2, h=3, sq=sq, sk=sk, deterministic=True, accumulation=None)
        for _ in range(2):
            fresh = {name: torch.empty_like(tensor) for name, tensor in case.tensors.items()}
            for name in ("q", "k", "v", "do"):
                fresh[name].normal_()
            o, lse, _, *expected = _reference64(*(fresh[name] for name in ("q", "k", "v", "do")), case.keep)
            fresh["o"].copy_(o)
            fresh["stats"].copy_(lse.unsqueeze(-1))
            for name in ("dq", "dk", "dv"):
                fresh[name].fill_(float("nan"))
            case.tensors, case.expected = fresh, expected
            case.pack = {case.refs[name]: tensor for name, tensor in fresh.items()}
            case.graph.execute(case.pack, None)
            _check_case(case)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        cudnn.set_stream(cudnn_handle, stream.cuda_stream)
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture, stream=stream):
            case.graph.execute(case.pack, None, handle=cudnn_handle)
        expected_bits = {name: case.tensors[name].clone() for name in ("dq", "dk", "dv")}
        for _ in range(3):
            for name in expected_bits:
                case.tensors[name].fill_(float("nan"))
            capture.replay()
            _check_case(case)
            for name, expected in expected_bits.items():
                assert torch.equal(case.tensors[name], expected)
        for name in ("q", "k", "v", "do"):
            case.tensors[name].normal_()
        o, lse, _, *case.expected = _reference64(*(case.tensors[name] for name in ("q", "k", "v", "do")), case.keep)
        case.tensors["o"].copy_(o)
        case.tensors["stats"].copy_(lse.unsqueeze(-1))
        for name in expected_bits:
            case.tensors[name].fill_(float("nan"))
        capture.replay()
        _check_case(case)
    finally:
        try:
            if capture is not None:
                capture.reset()
        finally:
            cudnn.set_stream(cudnn_handle, original_stream)


def test_native_head_host_specializations(monkeypatch):
    """The pointer host must compile each width's actual body and retain the window key."""
    from pathlib import Path

    import cutlass

    from cudnn.frost.template_loader import load_template
    from cudnn.sdpa.bwd.kernels.sm100 import prepared_d128_host as wide

    narrow = load_template(wide.__file__, 64, tag="sdpa_bwd_d64_host")
    body = load_template(str(Path(wide.__file__).with_name("bprop_d128_f16.py")), 64, tag="sdpa_bwd_d64")
    assert body.CFG.HEAD_DIM == 64 and body.CFG.TILE_N == 128
    assert body.CFG.SLAB_ELEMS >= body.CFG.AUX + body.CFG.TILE_M * body.CFG.TILE_N
    captured = []

    def compile_host(fn, *args, **kwargs):
        result = object()
        captured.append((fn, kwargs["cache_key"], result))
        return result

    monkeypatch.setattr(wide, "compile_cached", compile_host)
    monkeypatch.setattr(narrow, "compile_cached", compile_host)
    for d, window in ((128, 0), (64, 1), (64, 129)):
        geometry = tuple(((1, 65, 3, d), (65 * 3 * d, 3 * d, d, 1)) if i != 5 else ((1, 3, 65), (195, 65, 1)) for i in range(9))
        assert wide.compile_host(geometry, cutlass.BFloat16, True, 100, window) is captured[-1][2]
        assert captured[-1][0] is (narrow.host if d == 64 else wide.host)
    assert narrow.tensor_host is body.host
    assert len({key for _, key, _ in captured}) == 3
