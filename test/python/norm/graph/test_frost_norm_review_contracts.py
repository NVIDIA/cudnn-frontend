# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host contracts exposed by block-scaling review; no kernel execution needed."""

from types import SimpleNamespace
import pytest
import cudnn

pytestmark = pytest.mark.L0


@pytest.mark.parametrize(
    "stride,axis,transpose,expected_y,expected_s",
    [
        ([4000, 100, 1], 2, False, [4000, 100, 1], [96, 3, 1]),
        ([32, 1, 64], None, True, [32, 1, 64], [32, 1, 64]),
        ([], 2, False, [3072, 96, 1], [96, 3, 1]),
        ([3072, 96, 1], 1, True, [32, 1, 64], [1, 1, 2]),
    ],
)
def test_block_quant_stride_contract(stride, axis, transpose, expected_y, expected_s):
    from cudnn._pygraph import _infer_block_quant_strides

    g = cudnn.pygraph()
    x = g.tensor(dim=[2, 32, 96], data_type=cudnn.data_type.FLOAT)
    x.stride = stride
    y, s = g.block_scale_quantize(input=x, block_size=32, axis=axis, transpose=transpose)
    y.set_dim([2, 32, 96])
    s.set_dim([2, 1, 96] if axis == 1 else [2, 32, 3])
    _infer_block_quant_strides(g.nodes[-1])
    assert y.get_stride() == expected_y
    assert s.get_stride() == expected_s


def test_norm_block_scale_placement_preserves_backend_preference():
    from cudnn.norm.heuristics import propose
    from cudnn.engines.heuristics import BACKEND

    offered = {"norm_fprop_sm100": 20000}
    plain = propose("A", SimpleNamespace(quantize_nodes=()), offered)
    quant = propose("A", SimpleNamespace(quantize_nodes=(object(),)), offered)
    assert [p.engine_id for p in plain] == [20000]
    assert [p.engine_id for p in quant] == [BACKEND.engine_id, 20000]


def test_norm_device_properties_follow_current_device(monkeypatch):
    import torch
    from cudnn.norm import utils

    monkeypatch.setattr(utils, "_DEVICE_PROPERTIES", {})
    monkeypatch.setattr(utils, "_SM_COUNT", None)
    monkeypatch.setattr(utils, "_SMEM_CAPACITY", None)
    monkeypatch.setattr(utils, "_SMEM_PER_SM", None)
    current = [0]
    calls = []
    from cudnn.frost import device

    monkeypatch.setattr(device, "current_device", lambda: current[0])
    monkeypatch.setattr(device, "multiprocessor_count", lambda index: (148, 108)[index])
    monkeypatch.setattr(device, "shared_memory_per_block_optin", lambda index: (232448, 101376)[index])
    monkeypatch.setattr(device, "oversized_shared_memory_per_block", lambda index: (0, 334848)[index])
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current[0])

    def properties(index):
        calls.append(index)
        return SimpleNamespace(
            shared_memory_per_multiprocessor=(233472, 335872)[index],
        )

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    for index in (0, 1, 0, 1):
        current[0] = index
        assert utils.sm_count() == (148, 108)[index]
        assert utils.smem_capacity() == (232448, 101376)[index]
        assert utils.smem_per_sm() == (233472, 335872)[index]
    assert calls == [0, 1]


def test_unknown_compiler_retains_internal_build_policy(monkeypatch):
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "cutedsl_compiler_version", lambda: None)
    assert buffers.cutedsl_compiler_requirement_error((10, 7)) is None


def test_cooperative_occupancy_cache_separates_devices(monkeypatch):
    from cudnn.norm import utils

    current = [0]
    calls = []
    monkeypatch.setattr(utils, "_COOP_OCC", {})
    monkeypatch.setattr(utils, "device_key", lambda: current[0])

    def launch(occ):
        calls.append((current[0], occ))
        if occ > (2 if current[0] == 0 else 1):
            raise RuntimeError("CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE")
        return occ

    for dev in (0, 1, 0, 1):
        current[0] = dev
        assert utils.run_coop(("kernel",), launch) == (2 if dev == 0 else 1)
    assert calls == [(0, 4), (0, 2), (1, 4), (1, 2), (1, 1), (0, 2), (1, 1)]


def test_block_scale_grid_limit_declines_at_plan_time():
    import torch
    from cudnn.norm.block_scale import mismatch
    from cudnn.norm.graph_analyzer import NormGraphFacts
    from cudnn.norm.config_sm100 import NormVariant

    g = cudnn.pygraph()
    shape = (1, 2**30, 8192)
    x = g.tensor(dim=shape, data_type=cudnn.data_type.BFLOAT16)
    gamma = g.tensor(dim=(1, 1, 8192), data_type=cudnn.data_type.FLOAT)
    y = g.tensor(dim=shape, data_type=cudnn.data_type.FLOAT).set_output(False)
    rstd = g.tensor(dim=(1, 2**30, 1), data_type=cudnn.data_type.FLOAT).set_output(True)
    q, s = g.block_scale_quantize(input=y, block_size=32, axis=2)
    q.set_output(True).set_dim(shape).set_stride(x.get_stride()).set_data_type(cudnn.data_type.FP8_E4M3)
    s.set_output(True).set_dim((1, 2**30, 256)).set_stride((2**38, 256, 1)).set_data_type(cudnn.data_type.FP8_E8M0)
    facts = NormGraphFacts(
        variant=NormVariant.RMS_NORM,
        phase="fprop",
        x_shape=shape,
        normalized_shape=(8192,),
        dtype=torch.bfloat16,
        has_mean=False,
        device_cc=(10, 0),
        x_t=x,
        scale_t=gamma,
        y_t=y,
        inv_var_t=rstd,
        quantize_nodes=(g.nodes[-1],),
    )
    assert mismatch(facts) == "block quantization grid exceeds the CUDA grid-x limit"


@pytest.mark.parametrize("io", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize(
    "kind,shape,options",
    [
        ("LAYER_NORM", (64, 128), {"normalized_shape": (128,)}),
        ("RMS_NORM", (64, 128), {"normalized_shape": (128,)}),
        ("GROUP_NORM", (2, 32, 4, 4), {"num_groups": 4}),
        ("GROUP_NORM", (2, 32, 4, 4), {"num_groups": 4, "nhwc": True}),
        ("INSTANCE_NORM", (2, 32, 4, 4), {}),
        ("INSTANCE_NORM", (2, 32, 4, 4), {"nhwc": True}),
        ("BATCH_NORM", (2, 32, 4, 4), {}),
        ("BATCH_NORM", (2, 32, 4, 4), {"nhwc": True}),
    ],
)
def test_workspace_sizers_need_shapes_not_input_tensors(monkeypatch, io, kind, shape, options):
    from cudnn.norm import NormVariant, utils
    from cudnn.norm.fprop.api import fprop_workspace_bytes
    from cudnn.norm.bprop.api import bprop_workspace_bytes

    # Geometry is computed on the host. No allocation or kernel is needed to
    # detect accidental tensor/device guards in these shape-only entry points.
    monkeypatch.setattr(utils, "_SM_COUNT", 148)
    monkeypatch.setattr(utils, "_SMEM_CAPACITY", 232448)
    variant = getattr(NormVariant, kind)
    forward = fprop_workspace_bytes(variant, shape, io, **options)
    backward = bprop_workspace_bytes(variant, shape, io, **options)
    assert isinstance(forward, int) and forward >= 0
    assert isinstance(backward, int) and backward > 0
    if kind in ("LAYER_NORM", "RMS_NORM"):
        assert forward == 0
    elif kind == "BATCH_NORM":
        assert forward > 0


@pytest.mark.parametrize("reported", [None, 0])
def test_smem_per_sm_queries_driver_when_torch_property_is_missing(monkeypatch, reported):
    from cudnn.norm import utils
    from cuda.bindings import driver

    props = SimpleNamespace() if reported is None else SimpleNamespace(shared_memory_per_multiprocessor=reported)
    monkeypatch.setattr(utils, "_SMEM_PER_SM", None)
    monkeypatch.setattr(utils, "_device_properties", lambda: props)
    monkeypatch.setattr(utils, "device_key", lambda: 1)
    calls = []

    def get_device(ordinal):
        calls.append(("device", ordinal))
        return driver.CUresult.CUDA_SUCCESS, 17

    def query(attribute, device):
        calls.append((attribute, device))
        return driver.CUresult.CUDA_SUCCESS, 233472

    monkeypatch.setattr(driver, "cuDeviceGet", get_device)
    monkeypatch.setattr(driver, "cuDeviceGetAttribute", query)
    assert utils.smem_per_sm() == 233472
    assert calls == [("device", 1), (driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, 17)]


@pytest.mark.parametrize("storage", ["object", "cpu_bytes", "cpu_float"])
def test_norm_workspace_rejects_non_cuda_storage(storage):
    import torch
    from cudnn.norm.utils import Workspace

    buf = object() if storage == "object" else torch.empty(32, dtype=torch.uint8 if storage == "cpu_bytes" else torch.float32)
    with pytest.raises(ValueError, match="CUDA uint8 torch.Tensor"):
        Workspace(buf)
    assert Workspace().take(16, torch.float32) is None


def test_norm_resources_follow_build_scope_and_optin_limit(monkeypatch):
    from cudnn.frost import device
    from cudnn.norm import utils

    monkeypatch.setattr(utils, "_SM_COUNT", None)
    monkeypatch.setattr(utils, "_SMEM_CAPACITY", None)
    monkeypatch.setattr(device, "ambient_device", lambda: 0)
    monkeypatch.setattr(device, "multiprocessor_count", lambda index: (148, 108)[index])
    monkeypatch.setattr(device, "shared_memory_per_block_optin", lambda index: (232448, 101376)[index])
    monkeypatch.setattr(device, "oversized_shared_memory_per_block", lambda index: (0, 334848)[index])
    assert utils.device_key() == 0
    assert utils.sm_count() == 148
    assert utils.smem_capacity() == 232448
    with device.build_device(1):
        assert utils.device_key() == 1
        assert utils.sm_count() == 108
        # Even an available oversized carveout does not raise the norm budget.
        assert utils.smem_capacity() == 101376
    assert utils.device_key() == 0
    assert utils.sm_count() == 148
    assert utils.smem_capacity() == 232448
