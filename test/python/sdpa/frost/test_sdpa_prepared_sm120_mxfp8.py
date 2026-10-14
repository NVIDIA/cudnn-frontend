# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""SM120 MXFP8 pointer launches preserve opaque SF storage, rebinding, capture and wide-address contracts.

Runs the shared MXFP8 prepared suite (``test_sdpa_prepared_mxfp8``) on the SM120 row -- per-batch THD lengths,
no PackGQA, head dims up to 256 -- plus an SM120-sized wide scale-address case: the SM120 kernel addresses scale
bytes directly, so a head stride just above 2**32 bytes reaches the Int64 path.
"""

from functools import partial

import pytest
import torch

import test_sdpa_prepared_mxfp8 as shared
from frost_test_utils import requires_blackwell_geforce, requires_dsl
from sdpa.frost.test_sdpa_prepared_fp8 import _cuda_graph

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell_geforce]

_HEAD_DIMS = [(128, 128), (192, 128), (256, 256), (96, 160)]


@pytest.fixture(autouse=True)
def sm120_cases(monkeypatch):
    monkeypatch.setattr(shared, "_case", partial(shared._case, arch="sm120"))


def _no_tensor_fakes(monkeypatch):
    import cutlass.cute as cute

    monkeypatch.setattr(cute.runtime, "make_fake_tensor", lambda *a, **k: pytest.fail("prepared plan constructed a tensor fake"))
    monkeypatch.setattr(cute.runtime, "make_fake_compact_tensor", lambda *a, **k: pytest.fail("prepared plan constructed a compact tensor fake"))


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("d,dv", _HEAD_DIMS)
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_sm120_mxfp8_rebind_scales_and_buffers(thd, dtype, d, dv, output_dtype, monkeypatch):
    _no_tensor_fakes(monkeypatch)
    shared.test_prepared_mxfp8_rebind_scales_and_buffers(thd, True, True, dtype, d, dv, output_dtype)


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("output_dtype", [torch.float16, torch.float8_e5m2])
def test_sm120_mxfp8_rebind_other_output_dtypes(thd, output_dtype, monkeypatch):
    _no_tensor_fakes(monkeypatch)
    shared.test_prepared_mxfp8_rebind_scales_and_buffers(thd, True, True, torch.float8_e4m3fn, 128, 128, output_dtype)


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("d,dv", [(128, 128), (256, 256)])
def test_sm120_mxfp8_without_optional_outputs(thd, d, dv, monkeypatch):
    _no_tensor_fakes(monkeypatch)
    shared.test_prepared_mxfp8_rebind_scales_and_buffers(thd, False, False, torch.float8_e4m3fn, d, dv, torch.bfloat16)


@pytest.mark.parametrize("thd,split", [(False, 1), (True, 1), (False, 4)])
@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256)])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_sm120_mxfp8_capture_reads_current_scales(thd, split, d, dv, output_dtype):
    shared.test_prepared_mxfp8_capture_reads_current_scales(thd, split, d, dv, output_dtype)


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256)])
@pytest.mark.parametrize("carrier", [torch.uint8, torch.int32])
def test_sm120_mxfp8_sf_physical_permutation(thd, d, dv, carrier):
    shared.test_prepared_mxfp8_sf_physical_permutation(thd, d, dv, carrier)


@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256)])
def test_sm120_mxfp8_strided_dense_inputs(d, dv):
    shared.test_prepared_mxfp8_strided_dense_inputs(d, dv)


@pytest.mark.parametrize("thd,split", [(False, 1), (True, 1), (False, 4)])
def test_sm120_mxfp8_execute_has_no_allocation_or_sync(thd, split, monkeypatch):
    shared.test_prepared_mxfp8_execute_has_no_allocation_or_sync(thd, split, monkeypatch)


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256)])
def test_sm120_mxfp8_graph_and_adapter_bind_same_frame(thd, d, dv, monkeypatch):
    shared.test_prepared_mxfp8_graph_and_adapter_bind_same_frame(thd, d, dv, monkeypatch)


@pytest.mark.parametrize("d,dv", [(128, 128), (256, 256)])
@pytest.mark.parametrize("layout", ["padded", "batch_inner"])
def test_sm120_mxfp8_runtime_stats_layout_keeps_generic_branch(d, dv, layout):
    shared.test_prepared_mxfp8_runtime_stats_layout_keeps_generic_branch(d, dv, layout)


@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("d,dv", [(128, 128), (256, 256)])
def test_sm120_mxfp8_thd_output_stride_above_int32(dtype, d, dv):
    # Return the allocator's cached blocks from earlier cases: only memory in use should skip.
    torch.cuda.empty_cache()
    shared.test_prepared_mxfp8_thd_output_stride_above_int32(dtype, d, dv)


@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("names", [("sf_q",), ("sf_k", "sf_v")], ids=["q", "kv"])
@pytest.mark.parametrize("d,dv", [(128, 128), (256, 256)])
def test_sm120_mxfp8_sf_head_stride_above_int32(names, d, dv):
    """Head 1 of a THD scale tensor starts 2**32 + 2 tiles of bytes after head 0. An offset narrowed to
    32 bits lands on head 0's tile 2, which holds unit scales, so the narrowed read gives a different O.
    K and V scales share one packed tile count (the binder enforces it), so they widen together."""
    torch.cuda.empty_cache()
    g, vp, ws, bufs, ts = shared._case(thd=True, d=d, dv=dv, hq=2, hk=2)
    shapes = {name: bufs[name].shape for name in names}
    tiles = max((2**32 + 2 * shape[-1]) // shape[-1] for shape in shapes.values())
    need = sum(2 * tiles * shape[-1] for shape in shapes.values())
    if torch.cuda.mem_get_info()[0] < need + 2**30:
        pytest.skip(f"physical SF head-stride regression needs {need / 2**30 + 1:.0f} GiB free")
    for name, shape in shapes.items():
        try:
            wide = torch.empty((1, 2, tiles, shape[-1]), device="cuda", dtype=torch.uint8)
        except torch.OutOfMemoryError:
            pytest.skip("physical SF head-stride storage unavailable")
        # Touch only the live tiles and the narrow-offset alias of head 1.
        wide[:, :, : shape[2]].copy_(bufs[name])
        wide[:, 0, 2 : 2 + shape[2]].fill_(127)
        bufs[name] = wide
        vp[ts[name]] = wide
    bufs["o"].fill_(float("nan"))
    g.execute(vp, ws)
    torch.cuda.synchronize()
    shared._check(bufs, thd=True)
    with _cuda_graph() as captured:
        with torch.cuda.graph(captured):
            g.execute(vp, ws)
        bufs["o"].fill_(float("nan"))
        captured.replay()
        torch.cuda.synchronize()
        shared._check(bufs, thd=True)
