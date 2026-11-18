# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-query SM80 graphs retain the native addressing and replay contract."""

import pytest
import torch
import cutlass.cute as cute
import cudnn

from frost_test_utils import _SM, requires_dsl
from test_sdpa_fwd_paged_sm80 import _POINTS, test_native_paged_adapter_rebind_replay as _paged
from test_sdpa_fwd_origins_sm80 import test_independent_origins_rebind_replay as _origins
from test_sdpa_prepared_sm80 import _case, _check

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM != 80, reason="requires native SM80")]


@pytest.mark.parametrize("d,dv", [(48, 32), (64, 64), (96, 80), (128, 128), (136, 112), (192, 128), (248, 200), (256, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["plain", "features", "o_only"])
def test_dense_single_q_rebind_replay(d, dv, dtype, mode, cudnn_handle, monkeypatch):
    features = mode != "plain"
    case = _case(
        d,
        dv,
        dtype=dtype,
        sq=1,
        layout="gapped",
        stats=mode != "o_only",
        features=features,
        causal=mode != "plain",
        scheduler=["plain", "features", "o_only"].index(mode),
        heads=(4, 1 if mode == "o_only" else 2),
        scale=0.0 if mode == "features" else -0.3 if mode == "o_only" else None,
    )
    artifact = case.graph
    for binding in range(2):
        if binding:
            for name, old in list(case.bufs.items()):
                span = 1 + sum((n - 1) * stride for n, stride in zip(old.shape, old.stride()))
                storage = torch.full((span + 16,), 77, dtype=old.dtype, device="cuda")
                new = storage.as_strided(old.shape, old.stride(), 8).copy_(old)
                case.storage[name], case.bufs[name] = storage, new
                case.pack[case.refs[name]] = new
        capture = torch.cuda.CUDAGraph()

        def run():
            previous = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                with monkeypatch.context() as patch:

                    def forbidden(*args, **kwargs):
                        raise AssertionError("single-query execute must not allocate, copy, compile or synchronize")

                    for obj, names in ((cute, ("compile",)), (torch, ("empty", "zeros")), (torch.Tensor, ("clone", "contiguous"))):
                        for name in names:
                            patch.setattr(obj, name, forbidden)
                    case.graph.execute(case.pack, case.workspace, handle=cudnn_handle)
            finally:
                torch.cuda.set_sync_debug_mode(previous)

        try:
            for phase in range(2):
                case.bufs["v"].mul_(-0.7)
                case.bufs["o"].fill_(77)
                if features:
                    # Keep an addressable Q row with no keys, separately from a trimmed Q.
                    case.bufs["seq_q"].copy_(torch.tensor([1, phase], dtype=torch.int32, device="cuda"))
                    case.bufs["seq_kv"].copy_(torch.tensor([0 if phase else 58, 65 if phase else 0], dtype=torch.int32, device="cuda"))
                    case.bufs["sink"].add_(0.2)
                previous_stream = cudnn.get_stream(cudnn_handle)
                try:
                    cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
                    if phase == 0:
                        run()
                    else:
                        capture.replay()
                finally:
                    cudnn.set_stream(cudnn_handle, previous_stream)
                _check(case)
                assert case.graph is artifact
                if binding:
                    for storage in case.storage.values():
                        assert torch.all(storage[:8] == 77) and torch.all(storage[-8:] == 77)
                if phase == 0:
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    previous_stream = cudnn.get_stream(cudnn_handle)
                    try:
                        cudnn.set_stream(cudnn_handle, stream.cuda_stream)
                        with torch.cuda.graph(capture, stream=stream):
                            run()
                    finally:
                        cudnn.set_stream(cudnn_handle, previous_stream)
        finally:
            capture.reset()


@pytest.mark.parametrize("point", _POINTS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_paged_single_q_rebind_replay(point, dtype, thd, graph_mode, cudnn_handle, monkeypatch):
    _paged(point, dtype, thd, graph_mode, cudnn_handle, monkeypatch, single_q=True)


@pytest.mark.parametrize("d,dv", [(48, 32), (96, 80), (192, 128), (256, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("token_major", [False, True])
def test_thd_single_q_independent_origins(d, dv, dtype, token_major, cudnn_handle, monkeypatch):
    _origins(d, dv, dtype, token_major, cudnn_handle, monkeypatch, single_q=True)
