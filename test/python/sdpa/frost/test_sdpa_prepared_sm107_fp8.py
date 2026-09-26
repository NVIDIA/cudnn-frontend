# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""SM107 per-tensor FP8 binds current pointers through prepared execution."""

import pytest
import torch

from frost_test_utils import requires_dsl
import test_sdpa_prepared_fp8 as shared

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(torch.cuda.get_device_capability() != (10, 7), reason="SM107 required")]


@pytest.mark.parametrize("d,dv,split", [(128, 128, 1), (192, 128, 1), (256, 256, 1), (512, 512, 1), (128, 128, 4)])
@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_sm107_prepared_fp8_routes(d, dv, split, thd, dtype):
    if thd and split > 1:
        pytest.skip("THD split is not served")
    g, vp, ws, bufs, tensors = shared._case(d=d, dv=dv, split_kv=split, thd=thd, output_dtype=dtype, arch="sm107")
    assert g._compiled_plans[g._plan_index]._prepared is not None
    g.execute(vp, ws)
    shared._check(bufs, thd=thd)
    with shared._cuda_graph() as graph:
        with torch.cuda.graph(graph):
            g.execute(vp, ws)
        bufs["descale_v"].fill_(0.3)
        bufs["scale_o"].fill_(1.9)
        bufs["o"].fill_(float("nan"))
        bufs["amax_o"].fill_(999)
        graph.replay()
        shared._check(bufs, thd=thd)


@pytest.fixture(autouse=True)
def _sm107_case(monkeypatch):
    original = shared._case

    def case(**kwargs):
        kwargs.setdefault("arch", "sm107")
        return original(**kwargs)

    monkeypatch.setattr(shared, "_case", case)


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("stats,amax", [(True, True), (False, False)])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256), (512, 512)])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2])
def test_sm107_fp8_rebind(thd, stats, amax, dtype, d, dv, output_dtype, monkeypatch):
    shared.test_prepared_fp8_rebind_scales_and_buffers(thd, stats, amax, dtype, d, dv, output_dtype, monkeypatch)


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2])
def test_sm107_fp8_capture(thd, d, dv, dtype, monkeypatch):
    shared.test_prepared_fp8_capture_replay_reads_current_scales(thd, monkeypatch, d, dv, dtype)


@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256), (512, 512)])
def test_sm107_fp8_physical_thd_stride_int64(dtype, d, dv):
    shared.test_prepared_fp8_thd_output_row_stride_above_int32(dtype, d, dv)


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256), (512, 512)])
def test_sm107_fp8_override_and_adapter(thd, d, dv, monkeypatch):
    shared.test_prepared_fp8_bounded_batch_override(thd, monkeypatch, d, dv)
    shared.test_prepared_fp8_graph_and_adapter_bind_the_same_frame(thd, monkeypatch, d, dv)


@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256), (512, 512)])
def test_sm107_fp8_kv_and_input_strides(d, dv, monkeypatch):
    shared.test_prepared_fp8_dense_kv_override(d, dv, monkeypatch)
    shared.test_prepared_fp8_dense_input_strides(d, dv)
    shared.test_prepared_fp8_empty_thd_resets_amax_without_attention(d, dv, monkeypatch)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("splits", [2, 4])
def test_sm107_fp8_split_rebind(dtype, output_dtype, splits, monkeypatch):
    import test_sdpa_prepared_fp8_split as split

    split.test_prepared_fp8_split_rebind("sm107", 128, 128, dtype, output_dtype, splits, monkeypatch)


@pytest.mark.parametrize("stats,amax", [(True, True), (False, False)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_sm107_fp8_split_capture(stats, amax, dtype):
    import test_sdpa_prepared_fp8_split as split

    split.test_prepared_fp8_split_capture("sm107", 128, 128, stats, amax, dtype)


def test_sm107_fp8_split_overrides(monkeypatch):
    import test_sdpa_prepared_fp8_split as split

    split.test_prepared_fp8_split_runtime_kv_and_strides("sm107", 128, 128, monkeypatch)


@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_sm107_fp8_split_physical_stride_int64(dtype):
    import test_sdpa_prepared_fp8_split as split

    split.test_prepared_fp8_split_output_stride_int64("sm107", 128, 128, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_sm107_fp8_split_omitted_scales(dtype):
    import test_sdpa_prepared_fp8_split as split

    # The shared SM100 API owns both arch lines and selects SM107 at runtime.
    split.test_prepared_fp8_split_standalone_default_scales("sm100", 128, 128, dtype)


@pytest.mark.parametrize("d,dv", [(64, 64), (112, 96), (384, 320)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_sm107_fp8_dense_envelopes(d, dv, dtype):
    g, vp, ws, bufs, _ = shared._case(d=d, dv=dv, sq=16, skv=256, output_dtype=dtype, override=True)
    assert g._compiled_plans[g._plan_index]._prepared is not None
    g.execute(vp, ws)
    shared._check(bufs, thd=False, sq=16, skv=256)
