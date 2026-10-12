# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Per-tensor FP8 binding preserves scalar storage, host frames and split ownership."""

import sdpa_binding_reference as binding_reference

import ast
from pathlib import Path

import pytest
import torch

import cudnn
from cudnn.sdpa.fwd import prepared as prep
from test_sdpa_native_prefill_binding import _prefill_fixture

pytestmark = [pytest.mark.L1]
_ROLES = prep._NATIVE_DENSE_ROLES + prep._QUANT_ROLES
_INDICES = tuple(range(len(_ROLES)))


def _fixture(d, dv, split, dtype, output, *, missing=None, has_amax=True, arch="sm100"):
    partial_dtype = "bfloat16" if output == "bfloat16" else "float16"
    s, facts, frames, combined = _prefill_fixture(d, dv, False, split, partial_dtype, arch=arch)
    path = Path(prep.__file__).parent / "kernels" / "_fp8_host.py"
    if arch == "sm120":
        path = path.parent / "sm120" / "prepared_host.py"
    host = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == ("fp8_host" if arch == "sm120" else "host"))
    s.order = [a.arg for a in host.args.args if "Constexpr" not in ast.unparse(a.annotation)]
    s.index = {name: i for i, name in enumerate(s.order)}
    s.template = [None] * len(s.order)
    s.elem_bytes = dict(q=2, k=2, v=2, o=4 if s.fp32_partial else 2)
    for role in ("q", "k", "v"):
        s.expect[role] = dtype
        s.elem_bytes[role] = 1
        facts[role] = facts[role]._replace(dtype=dtype)
    if split == 1:
        s.expect["o"] = output
        s.elem_bytes["o"] = 1 if output.startswith("float8") else 2
    else:
        s.combine = s.combine._replace(output_dtype=output)
    facts["o"] = facts["o"]._replace(dtype=output)
    for i, (role, f) in enumerate(facts.items()):
        if f is not None:
            facts[role] = f._replace(ptr=0x10000000 + i * 0x1000000)
    scratch = 0 if split == 1 else s.combine.lse_offset + s.combine.lse.numel * 4
    s.quant = prep.QuantizedLaunchSpec(has_amax, scratch)
    for i, role in enumerate(prep._QUANT_ROLES):
        if role != missing:
            facts[role] = prep.BufferFacts(0x40000000 + 16 * i, "float32", (2, 0), 1, (1,), (1,))
    s.native = cudnn._pybind_module._SdpaDenseBinder(s)
    return s, facts, frames, combined


def _native_pack(facts):
    pack = cudnn._pybind_module.VariantPackNative(len(_ROLES))
    for i, role in enumerate(_ROLES):
        prep._set_native_fact(pack, i, facts.get(role))
    return pack


@pytest.mark.parametrize(
    "d,dv,split,dtype,output",
    [
        pytest.param(d, dv, split, dtype, output, marks=pytest.mark.L0 if (d, split, dtype, output) == (128, 1, "float8_e4m3fn", "bfloat16") else ())
        for d, dv in ((128, 128), (192, 128), (256, 256), (512, 512))
        for split in (1, 4)
        for dtype in ("float8_e4m3fn", "float8_e5m2")
        for output in ("float16", "bfloat16", "float8_e4m3fn", "float8_e5m2")
    ],
)
def test_native_fp8_matches_actual_host_and_python(d, dv, split, dtype, output):
    s, facts, frames, combined = _fixture(d, dv, split, dtype, output)
    for offset in (0, 0x100000000):
        fresh = {role: f._replace(ptr=f.ptr + offset) if f is not None else None for role, f in facts.items()}
        workspace = 0x50000000 + offset
        binding_reference.execute_quantized(s, fresh, workspace, 17, 17)
        expected, combine = frames.pop(), combined.pop() if split > 1 else ()
        actual, tail, identity = s.native.bind_quantized(_native_pack(fresh), _INDICES, workspace, 17)
        assert tuple(actual) == expected and tuple(tail) == combine and identity == 0
        s.native.execute(_native_pack(fresh), _INDICES, 17, workspace=workspace)
        assert frames.pop() == expected
        if split > 1:
            assert combined.pop() == combine


@pytest.mark.parametrize(
    "role,kind", [(r, k) for r in prep._QUANT_ROLES for k in ("dtype", "device", "null", "alignment", "shape", "span", "workspace")] + [("amax_o", "q_alias")]
)
def test_native_fp8_rejects_current_scalars_before_launch(role, kind):
    s, facts, frames, combined = _fixture(128, 128, 4, "float8_e4m3fn", "bfloat16")
    workspace = 0x50000000
    s.native.execute(_native_pack(facts), _INDICES, 17, workspace=workspace)
    frames.clear()
    combined.clear()
    f = facts[role]
    bad = dict(
        dtype=f._replace(dtype="int32"),
        device=f._replace(device=(1, 0)),
        null=f._replace(ptr=0),
        alignment=f._replace(ptr=f.ptr + 1),
        shape=f._replace(shape=(2,), span=2),
        span=f._replace(span=0),
        workspace=f._replace(ptr=workspace),
        q_alias=f._replace(ptr=facts["q"].ptr),
    )[kind]
    with pytest.raises(ValueError):
        s.native.execute(_native_pack(dict(facts, **{role: bad})), _INDICES, 17, workspace=workspace)
    assert frames == combined == []


@pytest.mark.parametrize(
    "d,dv,split,output",
    [
        pytest.param(d, dv, split, output, marks=pytest.mark.L0 if (d, split, output) in ((128, 1, torch.bfloat16), (256, 4, torch.bfloat16)) else ())
        for d, dv in ((128, 128), (192, 128), (256, 256), (512, 512))
        for split in ([1] if d == 512 else [1, 4])
        for output in (torch.bfloat16, torch.float8_e4m3fn)
    ],
)
def test_native_fp8_graph_rebinds_scales_and_replays(d, dv, split, output, monkeypatch, *, arch="sm100", cc=((10, 0), (10, 3))):
    from frost_test_utils import _dsl_installed
    from test_sdpa_prepared_fp8 import _case, _check

    if torch.cuda.get_device_capability() not in cc or not _dsl_installed():
        pytest.skip("SM100/SM103 and supported CuTe DSL required")
    g, vp, workspace, buffers, tensors = _case(arch=arch, d=d, dv=dv, output_dtype=output, split_kv=split)
    spec = g._compiled_plans[g._plan_index]._prepared.spec
    assert spec.native is not None
    monkeypatch.setattr(prep, "facts_of_roles", lambda *a: pytest.fail("native FP8 rebuilt Python operand facts"))
    monkeypatch.setattr(prep, "execute_quantized", lambda *a, **k: pytest.fail("native FP8 entered Python binding"), raising=False)
    g.execute(vp, workspace)
    _check(buffers, thd=False)
    for name in ("q", "k", "v", "o", "descale_q", "descale_k", "descale_v", "scale_o", "amax_o", "lse"):
        buffers[name] = buffers[name].clone()
        vp[tensors[name]] = buffers[name]
    workspace = torch.empty_like(workspace)
    g.execute(vp, workspace)
    _check(buffers, thd=False)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g.execute(vp, workspace)
    buffers["descale_q"].fill_(1.1)
    buffers["descale_v"].fill_(0.3)
    buffers["scale_o"].fill_(1.4)
    graph.replay()
    torch.cuda.synchronize()
    _check(buffers, thd=False)
    graph.reset()


@pytest.mark.parametrize("missing", prep._QUANT_ROLES)
@pytest.mark.parametrize("split", [1, 4])
def test_native_fp8_omitted_scalars_use_current_workspace(missing, split, monkeypatch):
    writes = []
    monkeypatch.setattr(prep._buffers, "fill_word_async", lambda *args: writes.append(args))
    s, facts, frames, combined = _fixture(128, 128, split, "float8_e4m3fn", "float8_e4m3fn", missing=missing)
    for workspace in (0x50000000, 0x60000000):
        binding_reference.execute_quantized(s, facts, workspace, 17, 17)
        expected_frame = frames.pop()
        expected_combine = combined.pop() if split > 1 else ()
        expected_writes = writes[:]
        writes.clear()
        s.native.execute(_native_pack(facts), _INDICES, 17, workspace=workspace)
        assert frames.pop() == expected_frame and writes == expected_writes
        if split > 1:
            assert combined.pop() == expected_combine
        writes.clear()
    frames.clear()
    s.quant = s.quant._replace(scratch_offset=max(16, s.quant.scratch_offset))
    s.native = cudnn._pybind_module._SdpaDenseBinder(s)
    with pytest.raises(ValueError):
        s.native.execute(_native_pack(facts), _INDICES, 17, workspace=2**63 - 16)
    assert frames == writes == []


@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("role", ["q", "k", "v", "o"])
@pytest.mark.parametrize("product", [False, True])
def test_native_fp8_physical_wide_operand_address(role, product, monkeypatch, *, arch="sm100", cc=((10, 0), (10, 3))):
    from frost_test_utils import _dsl_installed
    from test_sdpa_prepared_fp8 import _case, _check

    if torch.cuda.get_device_capability() not in cc or not _dsl_installed():
        pytest.skip("SM100/SM103 and supported CuTe DSL required")
    import ctypes

    batch = 5 if product else 2
    g, vp, workspace, buffers, tensors = _case(arch=arch, override=True, b=batch)
    spec = g._compiled_plans[g._plan_index]._prepared.spec
    assert spec.native is not None
    tensor, original = tensors[role], buffers[role]
    stride = list(original.stride())
    stride[0] += 2**30 if product else 2**32
    origin = 2**31 if product else 0
    span = 1 + sum((n - 1) * st for n, st in zip(original.shape, stride))
    torch.cuda.empty_cache()
    if (origin + span) * original.element_size() + 1024**3 > torch.cuda.mem_get_info()[0]:
        pytest.skip("insufficient GPU memory for physical FP8 wide-address probe")
    try:
        backing = torch.empty(origin + span, dtype=original.dtype, device="cuda")
    except torch.OutOfMemoryError:
        pytest.skip("another allocation consumed physical FP8 probe capacity")
    shape = list(original.shape)
    shape[0] = 1
    guards = []
    for index in range(1, batch):
        offset = index * stride[0]
        narrowed = ctypes.c_int32(offset).value
        if offset != narrowed:
            guard = backing.as_strided(shape, original.stride(), origin + narrowed)
            guard.fill_(float("nan"))
            guards.append(guard)
    wide = backing.as_strided(original.shape, stride, origin)
    wide.copy_(original) if role != "o" else wide.fill_(float("nan"))
    buffers[role], vp[tensor] = wide, wide
    monkeypatch.setattr(prep, "facts_of_roles", lambda *a: pytest.fail("native FP8 rebuilt Python operand facts"))
    g.execute(vp, workspace, override_uids=[tensor.get_uid()], override_shapes=[tensor.get_dim()], override_strides=[stride])
    _check(buffers, thd=False, b=batch)
    assert guards and all(torch.isnan(guard.float()).all().item() for guard in guards)


@pytest.mark.parametrize("hnd", [False, True])
@pytest.mark.parametrize("split", [1, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_native_fp8_paged_route_preserves_strides_and_capture(hnd, split, dtype, monkeypatch):
    from frost_test_utils import _dsl_installed
    from test_sdpa_prepared_fp8_paged import test_prepared_fp8_paged_strides_and_capture as check

    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)) or not _dsl_installed():
        pytest.skip("SM100/SM103 and supported CuTe DSL required")
    monkeypatch.setattr(prep, "facts_of_roles", lambda *a: pytest.fail("native FP8 page pool rebuilt Python operand facts"))
    check(hnd, split, dtype, monkeypatch)


@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_native_fp8_standalone_route_uses_caller_scalar_scratch(d, dv, dtype, monkeypatch):
    from frost_test_utils import _dsl_installed
    from test_sdpa_prepared_fp8_split import test_prepared_fp8_split_standalone_default_scales as check

    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)) or not _dsl_installed():
        pytest.skip("SM100/SM103 and supported CuTe DSL required")
    monkeypatch.setattr(prep, "execute_quantized", lambda *a, **k: pytest.fail("standalone FP8 entered Python binding"), raising=False)
    check("sm100", d, dv, dtype)
