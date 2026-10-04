# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native packed FP8 validates current widths/scalars before any stream writes."""

import ast
from pathlib import Path

import pytest
import torch

import cudnn
from cudnn.sdpa.fwd import prepared as prep
from test_sdpa_native_thd_binding import _fixture as _half_fixture

pytestmark = [pytest.mark.L1]
_ROLES = prep._NATIVE_THD_ROLES + prep._QUANT_ROLES
_INDICES = tuple(range(len(_ROLES)))


def _fixture(d=128, dv=128, dtype="float8_e4m3fn", output="bfloat16", arch="sm100", missing=None, layout="NH"):
    s, facts, frames = _half_fixture(layout=layout)
    old = dict(zip(s.order, s.template))
    path = Path(prep.__file__).parent / "kernels"
    path = path / "sm120/prepared_host.py" if arch == "sm120" else path / "_fp8_host.py"
    host = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == ("fp8_host" if arch == "sm120" else "host"))
    s.order = [a.arg for a in host.args.args if "Constexpr" not in ast.unparse(a.annotation)]
    s.index = {name: i for i, name in enumerate(s.order)}
    s.template = [old.get(name) for name in s.order]
    s.d_qk, s.d_v = d, dv
    s.fixed_batch = False
    s.workspace_alignment, s.scratch_bytes, s.split_workspace = 16, 8192, None
    s.quant = prep.QuantizedLaunchSpec(True, s.scratch_bytes)
    for role, heads, seq, dim in (("q", 8, 4, d), ("k", 2, 128, d), ("v", 2, 128, dv), ("o", 8, 4, dv)):
        s.expect[role] = output if role == "o" else dtype
        s.decl[role] = (heads, dim, heads * dim, dim, 1, heads * dim)
        facts[role] = facts[role]._replace(
            dtype=s.expect[role], shape=(4, heads, seq, dim), strides=(seq * heads * dim, dim, heads * dim, 1), span=4 * seq * heads * dim
        )
    for i, (role, fact) in enumerate(facts.items()):
        facts[role] = fact._replace(ptr=0x10000000 + i * 0x1000000)
    for i, role in enumerate(prep._QUANT_ROLES):
        if role != missing:
            facts[role] = prep.BufferFacts(0x40000000 + i * 16, "float32", (2, 0), 1, (1,), (1,))
    s.native = cudnn._pybind_module._SdpaThdBinder(s)
    return s, facts, frames


def _pack(facts):
    return prep._native_pack_from_facts(facts, _ROLES)


def _reference(s, facts, workspace=0x50000000):
    native, s.native = s.native, None
    try:
        return prep.execute_quantized(s, facts, workspace, 17, 17)
    finally:
        s.native = native


@pytest.mark.parametrize("arch", ["sm100", "sm107", "sm120"])
@pytest.mark.parametrize("d,dv", [pytest.param(128, 128, marks=pytest.mark.L0), (192, 128), (256, 256), (512, 512)])
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("output", ["float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"])
def test_thd_fp8_actual_host_frames_and_current_pointers(arch, d, dv, dtype, output):
    s, facts, frames = _fixture(d, dv, dtype, output, arch)
    for offset in (0, 2**33):
        fresh = {role: f._replace(ptr=f.ptr + offset) for role, f in facts.items()}
        workspace = 0x50000000 + offset
        assert _reference(s, fresh, workspace)
        expected = frames.pop()
        frame, identity, amax = s.native.bind_quantized(_pack(fresh), _INDICES, workspace, 17)
        assert tuple(frame) == expected and identity == 0 and amax == fresh["amax_o"].ptr
        assert s.native.execute(_pack(fresh), _INDICES, workspace, 17)
        assert frames.pop() == expected


@pytest.mark.parametrize("role", prep._QUANT_ROLES)
@pytest.mark.parametrize("kind", ["dtype", "device", "null", "alignment", "shape", "span", "workspace", "amax_alias"])
def test_thd_fp8_revalidates_scalars_before_writes(role, kind, monkeypatch):
    writes = []
    monkeypatch.setattr(prep._buffers, "fill_word_async", lambda *a: writes.append(a))
    monkeypatch.setattr(prep._buffers, "memset_zero_async", lambda *a: writes.append(a))
    s, facts, frames = _fixture()
    assert s.native.execute(_pack(facts), _INDICES, 0x50000000, 17)
    frames.clear()
    f = facts[role]
    changed = dict(
        dtype=f._replace(dtype="int32"),
        device=f._replace(device=(1, 0)),
        null=f._replace(ptr=0),
        alignment=f._replace(ptr=f.ptr + 1),
        shape=f._replace(shape=(2,), span=2),
        span=f._replace(span=0),
        workspace=f._replace(ptr=0x50000000),
        amax_alias=f._replace(ptr=facts["q"].ptr if role == "amax_o" else facts["amax_o"].ptr),
    )[kind]
    with pytest.raises(ValueError):
        s.native.execute(_pack(dict(facts, **{role: changed})), _INDICES, 0x50000000, 17)
    assert frames == writes == []


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("missing", prep._QUANT_ROLES)
def test_thd_fp8_missing_scalars_and_empty_amax_use_current_workspace(empty, missing, monkeypatch):
    writes = []
    monkeypatch.setattr(prep._buffers, "fill_word_async", lambda *a: writes.append(("one", *a)))
    monkeypatch.setattr(prep._buffers, "memset_zero_async", lambda *a: writes.append(("zero", *a)))
    s, facts, frames = _fixture(missing=missing)
    if empty:
        facts["q"] = facts["q"]._replace(span=0)
    for workspace in (0x50000000, 0x70000000):
        launched = _reference(s, facts, workspace)
        expected_frames, expected_writes = frames[:], writes[:]
        frames.clear()
        writes.clear()
        assert s.native.execute(_pack(facts), _INDICES, workspace, 17) == launched == (not empty)
        assert frames == expected_frames and writes == expected_writes
        frames.clear()
        writes.clear()


@pytest.mark.parametrize("role", ["q", "k", "v", "o"])
@pytest.mark.parametrize("dtype", ["bfloat16", "float8_e4m3fn"])
def test_thd_fp8_alignment_uses_each_operand_element_width(role, dtype):
    s, facts, frames = _fixture(output=dtype)
    f = facts[role]
    # Eight elements satisfy a half TMA stride, but do not satisfy FP8 alignment.
    head_stride = f.shape[-1] + 8
    token_stride = f.shape[1] * head_stride
    bad = f._replace(strides=(f.shape[2] * token_stride, head_stride, token_stride, 1), span=4 * f.shape[2] * token_stride)
    if role == "o" and dtype == "bfloat16":
        assert s.native.execute(_pack(dict(facts, **{role: bad})), _INDICES, 0x50000000, 17)
    else:
        with pytest.raises(ValueError, match="16-byte multiple"):
            s.native.execute(_pack(dict(facts, **{role: bad})), _INDICES, 0x50000000, 17)
        assert frames == []


@pytest.mark.parametrize("arch", ["sm100", "sm107", "sm120"])
@pytest.mark.parametrize("d,dv", [pytest.param(128, 128, marks=pytest.mark.L0), (192, 128), (256, 256), (512, 512)])
@pytest.mark.parametrize("output", [torch.bfloat16, torch.float8_e4m3fn])
def test_thd_fp8_native_graph_current_scales_and_replay(arch, d, dv, output, monkeypatch):
    from frost_test_utils import _dsl_installed
    from test_sdpa_prepared_fp8 import _case, _check

    cc = torch.cuda.get_device_capability()
    expected = "sm120" if cc[0] == 12 else "sm107" if cc == (10, 7) else "sm100"
    if cc not in ((10, 0), (10, 3), (10, 7), (12, 0), (12, 1)) or arch != expected or not _dsl_installed():
        pytest.skip("requires this prepared FP8 architecture")
    g, vp, ws, bufs, tensors = _case(thd=True, arch=arch, d=d, dv=dv, output_dtype=output)
    spec = g._compiled_plans[g._plan_index]._prepared.spec
    assert spec.native is not None
    monkeypatch.setattr(prep, "facts_of_roles", lambda *a: pytest.fail("native THD rebuilt Python facts"))
    monkeypatch.setattr(prep, "execute_quantized", lambda *a, **k: pytest.fail("native THD entered Python scalar binding"))
    g.execute(vp, ws)
    _check(bufs, thd=True)
    for name in ("q", "k", "v", "o", "lse", "descale_q", "descale_k", "descale_v", "scale_o", "amax_o"):
        bufs[name] = bufs[name].clone()
        vp[tensors[name]] = bufs[name]
    ws = torch.empty_like(ws)
    g.execute(vp, ws)
    _check(bufs, thd=True)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            g.execute(vp, ws)
        bufs["descale_q"].fill_(1.1)
        bufs["descale_v"].fill_(0.3)
        bufs["scale_o"].fill_(1.4)
        graph.replay()
        torch.cuda.synchronize()
        _check(bufs, thd=True)
    finally:
        graph.reset()


@pytest.mark.parametrize("kind", ["workspace", "operand_end", "amax_end", "scratch_offset"])
def test_thd_fp8_int64_address_overflow_rejects_before_writes(kind, monkeypatch):
    writes = []
    monkeypatch.setattr(prep._buffers, "fill_word_async", lambda *a: writes.append(a))
    monkeypatch.setattr(prep._buffers, "memset_zero_async", lambda *a: writes.append(a))
    s, facts, frames = _fixture(missing="descale_v")
    workspace = 0x50000000
    if kind == "workspace":
        workspace = 2**63 - 16
    elif kind == "operand_end":
        facts["q"] = facts["q"]._replace(ptr=2**63 - 16)
    elif kind == "amax_end":
        facts["amax_o"] = facts["amax_o"]._replace(ptr=2**63 - 4)
    else:
        s.quant = s.quant._replace(scratch_offset=2**63 - 4)
        with pytest.raises(ValueError, match="int64"):
            cudnn._pybind_module._SdpaThdBinder(s)
        assert frames == writes == []
        return
    with pytest.raises(ValueError, match="int64"):
        s.native.execute(_pack(facts), _INDICES, workspace, 17)
    assert frames == writes == []


@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("role", ["q", "k", "v", "o"])
@pytest.mark.parametrize("product", [False, True], ids=["wide-stride", "wide-product"])
def test_thd_fp8_physical_wide_operand_address(role, product, monkeypatch):
    import ctypes

    from frost_test_utils import _dsl_installed
    from test_sdpa_prepared_fp8 import _case, _check

    cc = torch.cuda.get_device_capability()
    if cc not in ((10, 0), (10, 3), (10, 7), (12, 0), (12, 1)) or not _dsl_installed():
        pytest.skip("requires an existing per-tensor FP8 THD architecture")
    arch = "sm120" if cc[0] == 12 else "sm107" if cc == (10, 7) else "sm100"
    seq = 5 if product else 2
    g, vp, workspace, buffers, tensors = _case(thd=True, b=1, sq=seq, skv=seq, override=True, arch=arch)
    spec = g._compiled_plans[g._plan_index]._prepared.spec
    assert spec.native is not None
    original, tensor = buffers[role], tensors[role]
    stride = list(original.stride())
    stride[0] += 2**30 if product else 2**32
    origin = 2**31 if product else 0
    span = 1 + sum((n - 1) * st for n, st in zip(original.shape, stride))
    torch.cuda.empty_cache()
    if (origin + span) * original.element_size() + 1024**3 > torch.cuda.mem_get_info()[0]:
        pytest.skip("physical FP8 probe needs guarded token slabs")
    try:
        backing = torch.empty(origin + span, device="cuda", dtype=original.dtype)
    except torch.OutOfMemoryError:
        pytest.skip("another allocation consumed physical FP8 probe capacity")
    shape = (1, *original.shape[1:])
    guards = []
    for index in range(1, seq):
        offset = index * stride[0]
        narrowed = ctypes.c_int32(offset).value
        if offset != narrowed:
            guard = backing.as_strided(shape, original.stride(), origin + narrowed)
            guard.fill_(float("nan"))
            guards.append(guard)
    wide = backing.as_strided(original.shape, stride, origin)
    wide.copy_(original) if role != "o" else wide.fill_(float("nan"))
    buffers[role], vp[tensor] = wide, wide
    declared = list(tensor.get_stride())
    declared[2] = stride[0]
    overrides = dict(override_uids=[tensor.get_uid()], override_shapes=[tensor.get_dim()], override_strides=[declared])
    monkeypatch.setattr(prep, "facts_of_roles", lambda *a: pytest.fail("native THD rebuilt Python operand facts"))

    def run():
        g.execute(vp, workspace, **overrides)

    def check():
        _check(buffers, thd=True, b=1, sq=seq, skv=seq)
        assert guards and all(torch.isnan(guard.float()).all().item() for guard in guards)

    run()
    check()
    captured = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(captured):
            run()
        buffers["descale_v"].fill_(0.3)
        buffers["o"].fill_(float("nan"))
        captured.replay()
        check()
    finally:
        captured.reset()


@pytest.mark.L0
@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("carrier", ["exchange", "fallback"])
def test_thd_fp8_rejects_measured_empty_workspace_before_writes(ordered, native, empty, carrier):
    from frost_test_utils import _dsl_installed
    from cudnn.frost.buffers import DeviceView
    from test_sdpa_prepared_fp8 import _case, _check

    cc = torch.cuda.get_device_capability()
    if cc not in ((10, 0), (10, 3), (10, 7), (12, 0), (12, 1)) or not _dsl_installed():
        pytest.skip("requires a prepared FP8 architecture")
    arch = "sm120" if cc[0] == 12 else "sm107" if cc == (10, 7) else "sm100"
    g, vp, ws, bufs, tensors = _case(thd=True, amax=False, arch=arch)
    plan = g._compiled_plans[g._plan_index]
    assert plan._prepared.spec.native is not None and g.get_workspace_size() > 1
    if not native:
        plan._prepared.spec.native = None
    if empty:
        vp[tensors["q"]] = bufs["q"][:0]

    def call(workspace):
        if ordered:
            g.execute(tuple(vp.values()), workspace, tensor_uids=tuple(t.get_uid() for t in vp))
        else:
            g.execute(vp, workspace)

    # The full backing allocation makes the old-code RED control safe: only
    # the advertised view is undersized, so a missed guard cannot corrupt memory.
    for size in (0, 1, g.get_workspace_size() - 1):
        view = (
            cudnn._pybind_module.make_operand_buffer(ws.data_ptr(), [size], 1, 8, ws.device.index)
            if carrier == "exchange"
            else DeviceView(ws.data_ptr(), (size,), "uint8", ws.device.index)
        )
        if carrier == "exchange":
            assert cudnn._pybind_module.read_buffer_extent(view) == (ws.data_ptr(), size)
        else:
            assert cudnn._pybind_module.read_buffer_extent(view) is None
        ws.fill_(0xA5)
        bufs["o"].fill_(23)
        bufs["lse"].fill_(17)
        with pytest.raises(ValueError, match=rf"needs a .*workspace, got {size} bytes"):
            call(view)
        torch.cuda.synchronize()
        assert torch.all(ws == 0xA5).item()
        assert torch.all(bufs["o"] == 23).item()
        assert torch.all(bufs["lse"] == 17).item()
        # A failed call must not poison the plan. A real allocation and the
        # supported raw-pointer form both recover without a new preparation.
        for valid in (ws, ws.data_ptr()):
            call(valid)
            torch.cuda.synchronize()
            if empty:
                assert torch.all(bufs["o"] == 23).item()
            else:
                _check(bufs, thd=True)
