# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host contracts for independent QKV and dO formats in the backward harnesses."""

from dataclasses import fields
from types import SimpleNamespace

import cudnn
import pytest
import torch

from sdpa import fp8, mxfp8
from sdpa.random_config import ExecConfig

pytestmark = pytest.mark.L0
FP8_TYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


@pytest.mark.parametrize("qkv_type", FP8_TYPES)
@pytest.mark.parametrize("do_type", (None,) + FP8_TYPES)
def test_do_dtype_repro_round_trip(qkv_type, do_type):
    cfg = ExecConfig(data_type=qkv_type, dO_type=do_type)
    serialized = cfg.serialize()
    assert serialized["dO_type"] == (str(do_type) if do_type is not None else None)
    restored = ExecConfig.deserialize(serialized.copy())
    assert restored.data_type == qkv_type
    assert restored.dO_type == do_type
    assert restored.serialize() == serialized
    assert [f.name for f in fields(ExecConfig)].index("dO_type") > [f.name for f in fields(ExecConfig)].index("paged_distinct_v_table")


def test_old_repro_keeps_qkv_dtype_for_do():
    cfg = ExecConfig.deserialize({"data_type": "torch.float8_e4m3fn", "is_infer": False})
    assert cfg.dO_type is None
    assert cfg.data_type == torch.float8_e4m3fn


def test_backward_suites_draw_both_do_formats_without_changing_qkv():
    import random

    from sdpa.graph.suites.bprop.test_bprop_fp8 import fp8_bwd, fp8_thd_bwd
    from sdpa.graph.suites.bprop.test_bprop_mxfp8 import mxfp8_bwd
    from sdpa.graph.suites.knobs import model_knobs_fp8, model_knobs_mxfp8
    from sdpa.graph.suites.models.catalog import CATALOG

    knob_sets = [(factory(), {torch.float8_e4m3fn}) for factory in (fp8_bwd, fp8_thd_bwd, mxfp8_bwd)]
    for preset in CATALOG:
        knob_sets += [(model_knobs_fp8(preset, "bprop"), {torch.float8_e4m3fn}), (model_knobs_mxfp8(preset, "bprop"), set(FP8_TYPES))]
    for knobs, qkv_types in knob_sets:
        rng = random.Random(955)
        draws = [knobs["dO_type"](rng) for _ in range(128)]
        assert set(draws) == set(FP8_TYPES)
        assert draws.count(torch.float8_e5m2) > draws.count(torch.float8_e4m3fn)
        assert {knobs["data_type"](rng) for _ in range(128)} == qkv_types


class _Tensor:
    def __init__(self, **attrs):
        self.attrs = attrs

    def __getattr__(self, name):
        if name.startswith("set_"):

            def setter(value):
                self.attrs[name[4:]] = value
                return self

            return setter
        raise AttributeError(name)


class _Graph:
    """Record graph declarations without building a backend execution plan."""

    def __init__(self, **kwargs):
        self.tensors = {}

    def tensor(self, **kwargs):
        tensor = _Tensor(**kwargs)
        self.tensors[kwargs["uid"]] = tensor
        return tensor

    def sdpa_fp8_backward(self, **kwargs):
        return tuple(_Tensor() for _ in range(7))

    def sdpa_mxfp8_backward(self, **kwargs):
        return tuple(_Tensor() for _ in range(6))


@pytest.mark.parametrize("qkv_type", (cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2))
@pytest.mark.parametrize("do_type", (None, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2))
@pytest.mark.parametrize("family", ("fp8", "fp8_thd", "mxfp8"))
def test_backward_graph_declares_do_independently(monkeypatch, qkv_type, do_type, family):
    monkeypatch.setattr(cudnn, "pygraph", _Graph)
    kwargs = dict(b=2, h_q=4, h_k=2, h_v=2, s_qo=128, s_kv=256, d_qk=64, d_vo=64, attn_scale=0.125, deterministic=True)
    module = mxfp8 if family == "mxfp8" else fp8
    if family != "mxfp8":
        kwargs["is_ragged"] = family == "fp8_thd"
    graph = module.generate_graph_bwd(**kwargs, cudnn_itype=qkv_type, cudnn_otype=cudnn.data_type.HALF, cudnn_dO_type=do_type)
    uid = module.GraphBwdUid
    for role in ("q", "k", "v") + (("q_t", "k_t") if family == "mxfp8" else ()):
        assert graph.tensors[getattr(uid, role)].attrs["data_type"] == qkv_type
    for role in ("dO", "dO_t") if family == "mxfp8" else ("dO",):
        assert graph.tensors[getattr(uid, role)].attrs["data_type"] == (qkv_type if do_type is None else do_type)
    if family == "mxfp8":
        assert graph.tensors[uid.dO_f16].attrs["data_type"] == cudnn.data_type.BFLOAT16
        for role in ("sf_dO", "sf_dO_t"):
            assert graph.tensors[getattr(uid, role)].attrs["data_type"] == cudnn.data_type.FP8_E8M0
    elif family == "fp8_thd":
        assert "ragged_offset" in graph.tensors[uid.dO].attrs


@pytest.mark.parametrize("qkv_type", FP8_TYPES)
@pytest.mark.parametrize("do_type", (None,) + FP8_TYPES)
@pytest.mark.parametrize("family", ("fp8", "fp8_thd", "mxfp8"))
def test_harness_binds_quantized_do_in_its_declared_format(monkeypatch, qkv_type, do_type, family):
    """Run real host preparation and quantizers on CPU; stop before any kernel.

    Perf mode omits numerical comparisons and references. The intercepted backward
    execute checks storage, declarations and scale factors, not kernel results.
    """
    module = mxfp8 if family == "mxfp8" else fp8
    observed = {}

    class ReachedBackward(BaseException):
        pass

    class Graph(_Graph):
        def __getattr__(self, name):
            if name in ("validate", "build_operation_graph", "create_execution_plans", "check_support", "build_plans"):
                return lambda *args, **kwargs: None
            raise AttributeError(name)

        def get_workspace_size(self):
            return 0

        def execute(self, variant_pack, *args, **kwargs):
            if module.GraphBwdUid.dO in variant_pack:
                observed.update(graph=self, pack=variant_pack)
                raise ReachedBackward

    def cpu_factory(factory):
        def allocate(*args, **kwargs):
            if "device" in kwargs:
                kwargs["device"] = "cpu"
            return factory(*args, **kwargs)

        return allocate

    for name in ("empty", "full", "zeros", "ones", "tensor", "randn", "Generator"):
        monkeypatch.setattr(torch, name, cpu_factory(getattr(torch, name)))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 0))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=0))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(cudnn, "set_stream", lambda **kwargs: None)
    monkeypatch.setattr(cudnn, "backend_version_string", lambda: "9.27.0")
    monkeypatch.setattr(cudnn, "pygraph", Graph)
    monkeypatch.setattr(module, "generate_graph_fwd", lambda *args, **kwargs: Graph())
    monkeypatch.setattr(module, "note_frost_routing", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "time_execution", lambda *args, **kwargs: torch.tensor([0.0]))
    monkeypatch.setattr(module, "profile_execution", lambda *args, **kwargs: None)
    cfg = ExecConfig(
        data_type=qkv_type,
        dO_type=do_type,
        output_type=torch.bfloat16,
        batches=1,
        h_q=2,
        h_k=1,
        h_v=1,
        s_q=64,
        s_kv=64,
        d_qk=64,
        d_v=64,
        is_infer=False,
        is_ragged=family == "fp8_thd",
        seq_len_q=[32],
        seq_len_kv=[48],
        rng_data_seed=955,
    )
    cfg.fill_derived_fields()
    request = SimpleNamespace(config=SimpleNamespace(option=SimpleNamespace(dryrun=False), getoption=lambda name: name == "--perf"))
    with pytest.raises(ReachedBackward):
        (module.exec_sdpa_mxfp8 if family == "mxfp8" else module.exec_sdpa_fp8)(cfg, request, None)
    uid = module.GraphBwdUid
    graph, pack = observed["graph"], observed["pack"]
    expected_type = qkv_type if do_type is None else do_type
    for role in ("q", "k", "v"):
        assert pack[getattr(uid, role)].dtype == qkv_type
    for role in ("dO", "dO_t") if family == "mxfp8" else ("dO",):
        assert pack[getattr(uid, role)].dtype == expected_type
        assert graph.tensors[getattr(uid, role)].attrs["data_type"] == fp8.convert_to_cudnn_type(expected_type)
    if family != "mxfp8":
        # The generated dO has amax=2. E5M2's scale differs by 256x from
        # E4M3's, so changing only the tensor dtype cannot pass this check.
        descale = pack[uid.dO_descale].item()
        assert descale == fp8.get_fp8_descale_factor(2.0, expected_type)
