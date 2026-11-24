# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public graph coverage; dgrad and higher derivatives are unsupported."""

import builtins
from contextlib import nullcontext
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import cudnn
from cutlass import utils as cutlass_utils

from cudnn.conv.frost import wgrad_engine as engine
from cudnn.conv.frost.utils import _channels_last_3d_stride
from cudnn.conv.frost.wgrad_engine import FrostConvWgradEngine, fixed_knobs
from cudnn.engines.base import ExecutionContext
from cudnn.engines.engine_ids import FROST_CONV_WGRAD_ID_BASE
from cudnn.engines.heuristics import BACKEND
from cudnn.engines.manifest import engine_for_id, engines_for, family_for, resolve_analyzer, resolve_heuristics
from cudnn.frost import buffers

from utils import make_input, requires_dense_cutedsl
from warm_stress import StressCase, assert_release_dependence, assert_warm_stress, stress_validation

pytestmark = pytest.mark.L0
requires_wgrad_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)), reason="requires SM100 or SM103"
)


DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float8_e4m3fn, torch.float8_e5m2)
FE_DTYPES = dict(zip(DTYPES, (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2)))


def declaration(
    xshape,
    wshape,
    *,
    dtype=cudnn.data_type.BFLOAT16,
    padding=(0, 0, 0),
    stride=(1, 1, 1),
    dilation=(1, 1, 1),
    handle=None,
    output_dtype=None,
    post_padding=None,
    epilogue=None,
):
    post_padding = padding if post_padding is None else post_padding
    yshape = (
        xshape[0],
        wshape[0],
        *((xshape[i + 2] + padding[i] + post_padding[i] - (wshape[i + 2] - 1) * dilation[i] - 1) // stride[i] + 1 for i in range(3)),
    )
    g = cudnn.pygraph(handle=handle, io_data_type=dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    x = g.tensor(dim=xshape, stride=_channels_last_3d_stride(xshape), data_type=dtype)
    dy = g.tensor(dim=yshape, stride=_channels_last_3d_stride(yshape), data_type=dtype)
    dw = g.conv_wgrad(x, dy, pre_padding=padding, post_padding=post_padding, stride=stride, dilation=dilation)
    dw.set_dim(wshape).set_stride(_channels_last_3d_stride(wshape))
    if epilogue is not None:
        dw.set_data_type(cudnn.data_type.FLOAT)
        dw = epilogue.graph_fn(g, dw)
    dw.set_output(True).set_data_type(output_dtype or dtype).set_dim(wshape).set_stride(_channels_last_3d_stride(wshape))
    return g, (x, dy, dw)


def reference(x, dy, wshape, *, padding=(0, 0, 0), stride=(1, 1, 1), dilation=(1, 1, 1), output_dtype=torch.bfloat16, epilogue=None):
    # Independent explicit filter-position contraction in FP64, no conv backend.
    n, k, z, p, q = dy.shape
    result = torch.empty(wshape, device=x.device, dtype=torch.float64)
    for t in range(wshape[2]):
        for r in range(wshape[3]):
            for s in range(wshape[4]):
                coords = [
                    torch.arange(size, device=x.device) * st - pad + f * dil for size, st, pad, f, dil in zip((z, p, q), stride, padding, (t, r, s), dilation)
                ]
                valid = [(idx >= 0) & (idx < x.shape[i + 2]) for i, idx in enumerate(coords)]
                indices = [idx.clamp(0, x.shape[i + 2] - 1) for i, idx in enumerate(coords)]
                # Convert before advanced indexing: Torch has no FP8 indexing kernel.
                patch = x.double()[:, :, indices[0][:, None, None], indices[1][None, :, None], indices[2][None, None, :]]
                patch = patch * (valid[0][:, None, None] & valid[1][None, :, None] & valid[2][None, None, :])
                result[:, :, t, r, s] = torch.einsum("nkzpq,nczpq->kc", dy.double(), patch)
    result = result.float()
    if epilogue is not None:
        result = epilogue.reference_fn(result)
    return result.to(output_dtype)


def prepare(monkeypatch, xshape, wshape, *, handle=None, seed=3, dtype=torch.bfloat16, output_dtype=None, knobs=None, **geometry):
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    output_dtype = output_dtype or dtype
    g, (gx, gy, gw) = declaration(xshape, wshape, handle=handle, dtype=FE_DTYPES[dtype], output_dtype=FE_DTYPES[output_dtype], **geometry)
    x, dy = make_input(xshape, dtype, seed), make_input(tuple(gy.dim), dtype, seed + 1)
    dw = torch.empty(wshape, dtype=output_dtype, device="cuda", memory_format=torch.channels_last_3d)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plan(FROST_CONV_WGRAD_ID_BASE, knobs or fixed_knobs())
    g.select_plan(len(g.plans) - 1)
    g.check_support()
    g.build_plans()
    assert g.get_workspace_size() == 0
    workspace = torch.empty(0, dtype=torch.uint8, device="cuda")
    return g, {gx: x, gy: dy, gw: dw}, workspace, (x, dy, dw)


@requires_dense_cutedsl
@requires_wgrad_gpu
@pytest.mark.parametrize(
    "xshape,wshape",
    [
        ((1, 128, 1, 8, 8), (128, 128, 1, 1, 1)),
        ((2, 256, 3, 6, 10), (256, 256, 2, 3, 3)),
        ((1, 128, 17, 1, 64), (128, 128, 17, 1, 1)),
        ((1, 128, 1, 17, 64), (128, 128, 1, 17, 1)),
        ((1, 128, 1, 4, 32), (256, 128, 1, 1, 17)),
        ((2, 128, 4, 7, 10), (128, 128, 1, 4, 3)),
    ],
)
def test_frost_wgrad_numerics(monkeypatch, xshape, wshape):
    g, pack, ws, (x, dy, dw) = prepare(monkeypatch, xshape, wshape)
    g.execute(pack, ws)
    want = reference(x, dy, wshape)
    torch.testing.assert_close(dw, want, atol=0, rtol=0)
    # Quantized random values above give exact sums; retain a general rounding case.
    x.normal_(std=0.1)
    dy.normal_(std=0.1)
    g.execute(pack, ws)
    assert torch.isfinite(dw).all()
    torch.testing.assert_close(dw, reference(x, dy, wshape), atol=0.001, rtol=0.008)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"dtype": cudnn.data_type.INT8}, "matching"),
        ({"padding": (17, 0, 0)}, "corners"),
        ({"stride": (1, 9, 1)}, "at most 8"),
        ({"dilation": (1, 1, 0)}, "positive"),
        ({"xshape": (1, 65, 1, 8, 8), "wshape": (128, 65, 1, 1, 1)}, "aligned"),
        ({"wshape": (65, 128, 1, 1, 1)}, "aligned"),
        ({"xshape": (1, 128, 18, 1, 64), "wshape": (128, 128, 18, 1, 1)}, "corners"),
    ],
)
def test_frost_wgrad_rejections(monkeypatch, change, match):
    monkeypatch.setattr(engine, "_check_sm100", lambda: None)
    monkeypatch.setattr(engine, "_check_cutedsl", lambda *a: None)
    args = dict(xshape=(1, 128, 1, 8, 8), wshape=(128, 128, 1, 1, 1)) | change
    graph, _ = declaration(**args)
    with pytest.raises(NotImplementedError, match=match):
        engine.FrostConvWgradEngine().check_support(graph)


@pytest.mark.parametrize("axis", range(3), ids=("D", "H", "W"))
@pytest.mark.parametrize("filter_extent,value", ((1, 64), (2, 31), (2, 32), (3, 15), (3, 16), (4, 11)))
def test_frost_wgrad_filter_offset_limit(monkeypatch, axis, filter_extent, value):
    monkeypatch.setattr(engine, "_check_sm100", lambda: None)
    monkeypatch.setattr(engine, "_check_cutedsl", lambda *a: None)
    monkeypatch.setattr(engine, "build_device", lambda device: nullcontext())
    dilation = tuple(value if i == axis else 1 for i in range(3))
    filter_shape = tuple(filter_extent if i == axis else 1 for i in range(3))
    # Keep corners and output dimensions legal so only the offset gate can
    # reject this graph. Unit filters have zero offset regardless of dilation.
    post_padding = tuple((extent - 1) * dil for extent, dil in zip(filter_shape, dilation))
    graph, _ = declaration((1, 16, 3, 3, 3), (16, 16, *filter_shape), dilation=dilation, post_padding=post_padding)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not name.endswith("templates.sm100_conv_wgrad"), "filter-offset rejection must precede kernel import"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    candidate = FrostConvWgradEngine()
    if (filter_extent - 1) * value <= 31:
        candidate.check_support(graph)
    else:
        for call in (lambda: candidate.check_support(graph), lambda: candidate.build_plan(graph, SimpleNamespace(knobs={}))):
            with pytest.raises(NotImplementedError, match="dilated filter offsets .* at most 31"):
                call()


@requires_dense_cutedsl
@requires_wgrad_gpu
def test_frost_wgrad_dynamic_replay(monkeypatch):
    # Keep newer compiler primitives behind requires_dense_cutedsl during collection.
    from cudnn.conv.frost.templates import sm100_conv_wgrad as module

    module._compile_cached.cache_clear()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    handle = cudnn.create_handle()
    cudnn.set_stream(handle, stream.cuda_stream)
    retained = []
    try:
        with torch.cuda.stream(stream):
            for i, (xs, ws) in enumerate(
                [
                    ((1, 128, 1, 8, 8), (128, 128, 1, 1, 1)),
                    ((2, 256, 3, 6, 10), (256, 256, 2, 3, 3)),
                    ((1, 128, 17, 1, 64), (128, 128, 17, 1, 1)),
                ]
            ):
                g, pack, scratch, (x, dy, dw) = prepare(monkeypatch, xs, ws, handle=handle)
                index = next(j for j in range(len(g.plans)) if g.get_plan_name_at_index(j).split("[")[0] == "frost_conv_wgrad")
                eid, knobs = g.get_engine_and_knobs_at_index(index)
                assert eid == FROST_CONV_WGRAD_ID_BASE
                assert knobs == fixed_knobs()
                g.create_execution_plan(eid, knobs)
                g.select_plan(len(g.plans) - 1)
                g.build_plans()
                if i == 0:
                    artifact = module.compile()
                    monkeypatch.setattr(module.cute, "compile", lambda *a, **k: pytest.fail("shape change compiled another artifact"))
                assert module.compile() is artifact
                g.execute(pack, scratch)
                capture = torch.cuda.CUDAGraph()
                with torch.cuda.graph(capture, stream=stream):
                    g.execute(pack, scratch)
                x.mul_(0.5)
                capture.replay()
                torch.testing.assert_close(dw, reference(x, dy, ws), atol=0, rtol=0)
                retained.append((g, pack, scratch, capture))
            # Older captured host descriptors must remain independent of later shapes.
            for g, pack, scratch, capture in retained:
                x, dy, dw = pack.values()
                capture.replay()
                torch.testing.assert_close(dw, reference(x, dy, dw.shape), atol=0, rtol=0)
        stream.synchronize()
    finally:
        torch.cuda.current_stream().wait_stream(stream)
        cudnn.destroy_handle(handle)


@requires_dense_cutedsl
@requires_wgrad_gpu
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: "in-" + str(dt).removeprefix("torch."))
@pytest.mark.parametrize("group", (1, 2))
def test_frost_wgrad_warm_refill(monkeypatch, dtype, group):
    g, pack, ws, (x, dy, dw) = prepare(
        monkeypatch, (2, 384, 5, 10, 10), (384, 384, 4, 3, 3), dtype=dtype, knobs={cudnn.knob_type.CTA_GROUP: group, cudnn.knob_type.TILEK: 64}
    )
    # L=256: operand slots wrap for every dtype with the 64-byte stage below.
    out = dw.permute(0, 2, 3, 4, 1)
    want = reference(x, dy, dw.shape, output_dtype=dtype).permute(0, 2, 3, 4, 1).contiguous()
    assert_warm_stress(
        StressCase(
            f"frost_conv_wgrad_{dtype}_group{group}",
            lambda: g.execute(pack, ws),
            {"dw": out},
            min(324 if group == 1 else 216, cutlass_utils.HardwareInfo().get_max_active_clusters(group)) * group,
            256 // (64 // dtype.itemsize) - 3,
            {"dw": want},
        ),
        validation=stress_validation(cluster=group),
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("layout", "channels-last"),
        ("rank", "rank-5"),
        ("channels", "channels differ"),
        ("loss", "dY shape"),
        ("compute", "FP32"),
        ("mode", "convolution mode"),
        ("epilogue", "virtual intermediate"),
        ("extent", "positive"),
    ],
)
def test_frost_wgrad_graph_rejections(monkeypatch, mutation, match):
    monkeypatch.setattr(engine, "_check_sm100", lambda: None)
    monkeypatch.setattr(engine, "_check_cutedsl", lambda *a: None)
    graph, (x, dy, dw) = declaration((1, 128, 1, 8, 8), (128, 128, 1, 1, 1))
    if mutation == "layout":
        x.set_stride((8192, 64, 64, 8, 1))
    elif mutation == "rank":
        x.set_dim((1, 128, 8, 8))
    elif mutation == "channels":
        x.set_dim((1, 256, 1, 8, 8)).set_stride((16384, 1, 16384, 2048, 256))
    elif mutation == "loss":
        dy.set_dim((1, 128, 1, 4, 16)).set_stride((8192, 1, 8192, 2048, 128))
    elif mutation == "compute":
        graph.nodes[0].compute_data_type = cudnn.data_type.HALF
    elif mutation == "mode":
        graph.nodes[0].params["convolution_mode"] = cudnn._pybind_module.convolution_mode.CONVOLUTION
    elif mutation == "epilogue":
        graph.relu(dw).set_output(True)
    else:
        x.set_dim((0, 128, 1, 8, 8))
    with pytest.raises(NotImplementedError, match=match):
        engine.FrostConvWgradEngine().check_support(graph)


@pytest.mark.parametrize("version", ["4.8.0", "4.8.2"])
def test_frost_wgrad_dependency_gate(monkeypatch, version):
    graph, _ = declaration((1, 128, 1, 8, 8), (128, 128, 1, 1, 1))
    monkeypatch.setattr(buffers, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", version)))
    with pytest.raises(NotImplementedError, match=r">= 4.9"):
        engine.FrostConvWgradEngine().check_support(graph)


def test_frost_wgrad_knob_rejection():
    engine = FrostConvWgradEngine()
    assert engine.knobs_to_public(engine.knobs_from_public(fixed_knobs())) == fixed_knobs()
    for key in fixed_knobs():
        with pytest.raises(ValueError, match="configuration"):
            engine.knobs_from_public(fixed_knobs() | {key: 99})
    with pytest.raises(ValueError):
        engine.knobs_from_public({cudnn.knob_type.SPLIT_K_SLC: 2})
    for value in (True, 1.0):
        with pytest.raises(ValueError):
            engine.knobs_from_public({cudnn.knob_type.CTA_GROUP: value})


@pytest.mark.parametrize("role", [0, 1, 2])
@pytest.mark.parametrize("field", ["shape", "stride", "dtype", "alignment"])
def test_frost_wgrad_binding_rejection(monkeypatch, role, field):
    graph, tensors = declaration((1, 128, 1, 8, 8), (128, 128, 1, 1, 1))
    analysis = engine.analyze_wgrad(graph)
    calls = []
    monkeypatch.setattr(engine, "current_device", lambda: 0)
    plan = engine._WgradPlan(lambda *a: calls.append(a), analysis, 0)
    operands = []
    for i, tensor in enumerate(tensors):
        stride = tuple(tensor.stride)
        v = SimpleNamespace(shape=tuple(tensor.dim), stride=lambda stride=stride: stride, dtype="bfloat16", data_ptr=lambda: 16)
        if i == role:
            if field == "shape":
                v.shape = (1,)
            elif field == "stride":
                v.stride = lambda: (1,) * 5
            elif field == "dtype":
                v.dtype = "float16"
            else:
                v.data_ptr = lambda: 17
        operands.append(v)
    indices = {t.uid: i for i, t in enumerate(tensors)}
    pack = SimpleNamespace(index_of=lambda uid: indices[uid], operands=lambda i: operands)
    with pytest.raises(ValueError):
        plan.execute(graph, pack, ExecutionContext(stream=0))
    assert not calls


@requires_dense_cutedsl
@requires_wgrad_gpu
def test_frost_wgrad_release_rendering(tmp_path):
    # Compile in a fresh process so the dump options precede the DSL import.
    source = "import cudnn\n" f"assert cudnn.__file__ == {cudnn.__file__!r}\n" "from cudnn.conv.frost.templates.sm100_conv_wgrad import compile\n" "compile()\n"
    env = dict(os.environ, CUTE_DSL_KEEP="ptx,cubin", CUTE_DSL_DUMP_DIR=str(tmp_path), CUTE_DSL_CACHE_DIR=str(tmp_path / "cache"))
    result = subprocess.run([sys.executable, "-c", source], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    renders = list(Path(tmp_path).glob("*.ptx"))
    assert len(renders) == 1
    # Two accumulation slots, so one release site per slot, each folding its own fragment.
    assert_release_dependence(renders[0].read_text(), sites=2, words=32)


def test_frost_wgrad_slot_and_ranking(monkeypatch):
    graph, _ = declaration((1, 128, 1, 8, 8), (128, 128, 1, 1, 1))
    family = family_for(graph)
    assert family.name == "frost_conv_wgrad"
    assert family.engine_id == FROST_CONV_WGRAD_ID_BASE
    assert set(family.slots) == {"frost_conv_wgrad"}
    assert family.slots["frost_conv_wgrad"].slot == 0
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    assert engines_for(graph) == []
    assert engine_for_id(FROST_CONV_WGRAD_ID_BASE) is None
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    engines = engines_for(graph)
    assert [(e.name, e.engine_id) for e in engines] == [("frost_conv_wgrad", FROST_CONV_WGRAD_ID_BASE)]
    assert engine_for_id(FROST_CONV_WGRAD_ID_BASE) is engines[0]
    facts = resolve_analyzer(family)(graph)
    candidates = resolve_heuristics(family)("A", facts, family.offered_ids())
    assert candidates[0] is BACKEND
    assert candidates[1].engine_id == FROST_CONV_WGRAD_ID_BASE


@requires_dense_cutedsl
@requires_wgrad_gpu
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: "in-" + str(dt).removeprefix("torch."))
@pytest.mark.parametrize("output_dtype", DTYPES, ids=lambda dt: "out-" + str(dt).removeprefix("torch."))
@pytest.mark.parametrize("minimal", (False, True), ids=("tail", "minimum-pitch"))
def test_frost_wgrad_all_dtypes(monkeypatch, dtype, output_dtype, minimal):
    # Every axis has a partial tile; L=105 also tests the final reduction stage.
    c = max(16 // dtype.itemsize, 16 // output_dtype.itemsize) if minimal else 80
    k = 16 // dtype.itemsize if minimal else 144
    g, pack, ws, (x, dy, dw) = prepare(monkeypatch, (1, c, 3, 5, 7), (k, c, 1, 1, 1), dtype=dtype, output_dtype=output_dtype)
    g.execute(pack, ws)
    torch.testing.assert_close(dw.float(), reference(x, dy, dw.shape, output_dtype=output_dtype).float(), atol=0, rtol=0)


# All geometric knob combinations, including dtype-dependent epilogue refusals.
CONFIGS = [(m, n, red, group) for m in (64, 128) for n in range(32, 257, 32) for red in (64, 128) for group in (1, 2)]


@requires_dense_cutedsl
@requires_wgrad_gpu
@pytest.mark.parametrize("config", CONFIGS, ids=lambda cfg: "M%d-N%d-R%d-G%d" % cfg)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: "in-" + str(dt).removeprefix("torch."))
@pytest.mark.parametrize("output_dtype", DTYPES, ids=lambda dt: "out-" + str(dt).removeprefix("torch."))
def test_frost_wgrad_configuration_sweep(monkeypatch, config, dtype, output_dtype):
    m, n, red, group = config
    k = cudnn.knob_type
    knobs = {k.TILE_M: m, k.TILE_N: n, k.TILEK: red, k.CTA_GROUP: group}
    # Explicit independent legality oracle for the implemented TMEM/store paths.
    legal = not (m == 64 and ((group == 2 and n % 64) or (group == 1 and output_dtype in DTYPES[3:] and n % 64)))
    if not legal:
        with pytest.raises(ValueError, match="epilogue datapath"):
            prepare(monkeypatch, (1, 80, 1, 3, 37), (144, 80, 1, 1, 1), dtype=dtype, output_dtype=output_dtype, knobs=knobs)
        return
    g, pack, ws, (x, dy, dw) = prepare(monkeypatch, (1, 80, 1, 3, 37), (144, 80, 1, 1, 1), dtype=dtype, output_dtype=output_dtype, knobs=knobs)
    g.execute(pack, ws)
    torch.testing.assert_close(dw.float(), reference(x, dy, dw.shape, output_dtype=output_dtype).float(), atol=0, rtol=0)


@requires_dense_cutedsl
@requires_wgrad_gpu
@pytest.mark.parametrize(
    "geometry",
    [
        dict(padding=(1, 2, 0), post_padding=(0, 1, 2), stride=(2, 3, 1), dilation=(2, 1, 2)),
        dict(padding=(16, 0, 0), post_padding=(1, 16, 1), stride=(8, 1, 2), dilation=(1, 1, 1)),
        dict(padding=(0, 0, 0), post_padding=(0, 0, 0), stride=(1, 8, 3), dilation=(16, 2, 1)),
    ],
)
def test_frost_wgrad_geometry(monkeypatch, geometry):
    g, pack, ws, (x, dy, dw) = prepare(monkeypatch, (2, 80, 19, 11, 7), (144, 80, 2, 2, 2), **geometry)
    g.execute(pack, ws)
    ref_args = {key: value for key, value in geometry.items() if key != "post_padding"}
    torch.testing.assert_close(dw, reference(x, dy, dw.shape, **ref_args), atol=0, rtol=0)


from epilogue_cases import EPILOGUE_CASES


@requires_dense_cutedsl
@requires_wgrad_gpu
@pytest.mark.parametrize("epilogue", EPILOGUE_CASES, ids=lambda case: case.name)
def test_frost_wgrad_epilogues(monkeypatch, epilogue):
    g, pack, ws, (x, dy, dw) = prepare(monkeypatch, (1, 80, 1, 3, 71), (144, 80, 1, 1, 1), output_dtype=torch.float32, epilogue=epilogue)
    # Bound the output domain, including positive-only functions and exact zeros.
    if epilogue.input_profile == "positive":
        x.abs_().add_(0.25)
        dy.abs_().add_(0.25)
    else:
        x.mul_(0.25)
        dy.mul_(0.25)
    g.execute(pack, ws)
    torch.testing.assert_close(dw, reference(x, dy, dw.shape, output_dtype=torch.float32, epilogue=epilogue), atol=3e-6, rtol=3e-5)


@requires_dense_cutedsl
@requires_wgrad_gpu
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: "in-" + str(dt).removeprefix("torch."))
def test_frost_wgrad_geometry_artifact_reuse(monkeypatch, cudnn_handle, dtype):
    from cudnn.conv.frost.templates import sm100_conv_wgrad as module

    module._compile_cached.cache_clear()
    cases = [
        ((1, 80, 3, 5, 7), (144, 80, 1, 1, 1), {}),
        ((2, 48, 7, 5, 9), (80, 48, 2, 3, 2), dict(padding=(1, 2, 1), post_padding=(0, 1, 2), stride=(2, 1, 3), dilation=(2, 1, 2))),
    ]
    cases.append(((1, 16, 35, 3, 7), (32, 16, 2, 1, 1), dict(post_padding=(31, 0, 0), dilation=(31, 1, 1))))
    retained = []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    cudnn.set_stream(cudnn_handle, stream.cuda_stream)
    with torch.cuda.stream(stream):
        for i, (xs, ws, geometry) in enumerate(cases):
            g, pack, scratch, (x, dy, dw) = prepare(
                monkeypatch, xs, ws, dtype=dtype, output_dtype=torch.float32, knobs={cudnn.knob_type.CTA_GROUP: 2}, handle=cudnn_handle, **geometry
            )
            if i == 0:
                monkeypatch.setattr(module.cute, "compile", lambda *a, **kw: pytest.fail("shape/geometry change compiled a new artifact"))
            g.execute(pack, scratch)
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture, stream=stream):
                g.execute(pack, scratch)
            x.copy_(-x.float())
            capture.replay()
            ref_args = {key: value for key, value in geometry.items() if key != "post_padding"}
            torch.testing.assert_close(dw, reference(x, dy, dw.shape, output_dtype=torch.float32, **ref_args), atol=0, rtol=0)
            retained.append((capture, pack, geometry))
        for capture, pack, geometry in retained:
            x, dy, dw = pack.values()
            capture.replay()
            ref_args = {key: value for key, value in geometry.items() if key != "post_padding"}
            torch.testing.assert_close(dw, reference(x, dy, dw.shape, output_dtype=torch.float32, **ref_args), atol=0, rtol=0)
    stream.synchronize()
    cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)


@requires_dense_cutedsl
@requires_wgrad_gpu
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: "in-" + str(dt).removeprefix("torch."))
def test_frost_wgrad_rounding(monkeypatch, dtype):
    g, pack, ws, (x, dy, dw) = prepare(monkeypatch, (2, 80, 3, 7, 19), (144, 80, 2, 2, 2), dtype=dtype, output_dtype=torch.float32)
    generator = torch.Generator(device="cuda").manual_seed(17)
    x.copy_(torch.randn(x.shape, device="cuda", generator=generator) * 0.1)
    dy.copy_(torch.randn(dy.shape, device="cuda", generator=generator) * 0.1)
    g.execute(pack, ws)
    want = reference(x, dy, dw.shape, output_dtype=torch.float32)
    # TF32 multiplies rounded/truncated mantissas; other input formats are
    # exactly represented by the independent FP64 reference's multiplicands.
    torch.testing.assert_close(dw, want, atol=0.002 if dtype == torch.float32 else 2e-6, rtol=0.002 if dtype == torch.float32 else 2e-5)


@requires_dense_cutedsl
def test_frost_wgrad_operand_descriptor_swizzles():
    from cudnn.conv.frost.templates.sm100_conv_wgrad import OperandLayout
    from cutlass.experimental import primitives as prims

    for bits in (8, 16, 32):
        for channels in (16, 32, 48, 64, 96, 128, 256):
            layout = OperandLayout(channels, 128 * 8 // bits, bits)
            expected = (
                prims.Tcgen05SmemSwizzle.SWIZZLE_128B_ATOM_32B
                if bits == 32
                else {32: prims.Tcgen05SmemSwizzle.SWIZZLE_32B, 64: prims.Tcgen05SmemSwizzle.SWIZZLE_64B, 128: prims.Tcgen05SmemSwizzle.SWIZZLE_128B}[
                    layout.tma_box_bytes
                ]
            )
            # CUDA tensor-map enums are IntEnums too, but have different values.
            # Passing one directly to build() silently takes its integer value.
            assert layout.descriptor_layout == expected
            assert layout.stage_bytes >= channels * 128
            assert layout.stride_bytes == (512 if bits == 32 else 8 * layout.tma_box_bytes)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("input_dtype", "matching"),
        ("output_pitch", "aligned"),
        ("intermediate", "FP32 intermediate"),
        ("epilogue_compute", "FP32 epilogue"),
        ("extra_operand", "one input"),
        ("attribute", "Unsupported attributes"),
    ],
)
def test_frost_wgrad_extended_rejections(monkeypatch, mutation, match):
    monkeypatch.setattr(engine, "_check_sm100", lambda: None)
    monkeypatch.setattr(engine, "_check_cutedsl", lambda *a: None)
    g, (x, dy, dw) = declaration((1, 8, 1, 3, 7), (16, 8, 1, 1, 1), epilogue=EPILOGUE_CASES[0])
    if mutation == "input_dtype":
        dy.set_data_type(cudnn.data_type.HALF)
    elif mutation == "output_pitch":
        dw.set_data_type(cudnn.data_type.FP8_E4M3)
    elif mutation == "intermediate":
        g.nodes[0].outputs["DW"].set_data_type(cudnn.data_type.BFLOAT16)
    elif mutation == "epilogue_compute":
        g.nodes[1].compute_data_type = cudnn.data_type.HALF
    elif mutation == "extra_operand":
        g.nodes[1].inputs["extra"] = x
    else:
        g.nodes[1].params["unsupported_scalar"] = 1
    with pytest.raises((NotImplementedError, RuntimeError), match=match):
        FrostConvWgradEngine().check_support(g)


@requires_dense_cutedsl
@requires_wgrad_gpu
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("group", (1, 2))
@pytest.mark.parametrize(
    "dilation,filter_shape",
    (((31, 1, 1), (2, 2, 2)), ((1, 31, 1), (2, 2, 2)), ((1, 1, 31), (2, 2, 2)), ((15, 15, 15), (3, 3, 3))),
)
def test_frost_wgrad_filter_offset_boundary(monkeypatch, cudnn_handle, dtype, group, dilation, filter_shape):
    # Exercise the five-bit im2col limit independently on each spatial axis
    # and offsets near the limit with wider filters.
    post_padding = tuple((extent - 1) * dil for extent, dil in zip(filter_shape, dilation))
    geometry = dict(padding=(1, 0, 2), post_padding=post_padding, dilation=dilation)
    g, pack, ws, (x, dy, dw) = prepare(
        monkeypatch,
        (2, 80, 35, 3, 37),
        (144, 80, *filter_shape),
        dtype=dtype,
        output_dtype=torch.float32,
        knobs={cudnn.knob_type.CTA_GROUP: group},
        handle=cudnn_handle,
        **geometry,
    )
    g.execute(pack, ws)
    torch.testing.assert_close(dw, reference(x, dy, dw.shape, output_dtype=torch.float32, padding=geometry["padding"], dilation=dilation), atol=0, rtol=0)

    from determinism_utils import assert_replays

    def run():
        g.execute(pack, ws)
        return {"dw": dw}

    assert_replays(run)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    cudnn.set_stream(cudnn_handle, stream.cuda_stream)
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture, stream=stream):
        g.execute(pack, ws)
    cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
    x.copy_(-x.float())
    capture.replay()
    torch.testing.assert_close(dw, reference(x, dy, dw.shape, output_dtype=torch.float32, padding=geometry["padding"], dilation=dilation), atol=0, rtol=0)
