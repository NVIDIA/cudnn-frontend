# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed backward host ABI parity and current-storage validation."""

import ast
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cudnn
import pytest

from frost_test_utils import requires_dsl

from cudnn.sdpa.bwd import prepared as prep
from cudnn.sdpa.fwd.prepared import BufferFacts, _set_native_fact

pytestmark = [pytest.mark.L0]
_ARCHES = ("sm80", "sm100", "sm107", "sm107_thd", "sm120")
_WORKSPACE = 0x700000000000


def _fixture(arch="sm100", dtype="bfloat16", features=True, wide=0):
    family = arch.split("_thd")[0]
    path = Path(prep.__file__).parent / "kernels" / family / "prepared_host.py"
    entry = "host_f16_thd" if arch == "sm107_thd" else "host_f16" if family == "sm107" else "host"
    host = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == entry)
    names = [arg.arg for arg in host.args.args if "Constexpr" not in ast.unparse(arg.annotation)]
    # SM80 binds these four immutable launch bounds through functools.partial.
    if family == "sm80":
        assert names[:4] == ["t_q", "t_kv", "max_sq", "max_skv"]
        names = names[4:]
    aliases = {"lse": "stats", "q_lens": "seq_q", "kv_lens": "seq_kv"}
    roles = tuple(aliases.get(name.removesuffix("_ptr"), name.removesuffix("_ptr")) for name in names[: names.index("workspace")])
    ops, declared, facts = [], [], {}
    for i, role in enumerate(roles):
        enabled = i < 9 or (features and role != "delta")
        if not enabled:
            ops.append(None)
            declared.append(None)
            continue
        dt = "int32" if role.startswith("seq_") else "float32" if role in ("stats", "sink", "dsink", "bias", "dbias") else dtype
        shape, strides = ((2,), (1,)) if i >= 9 else ((2, 3, 5, 8), (160, 8, 32, 1))
        if role == "stats":
            shape, strides = (2, 3, 5, 1), (21, 7, 1, 1)
        if wide and i < 9:
            strides = ((2**32 + strides[0]) if wide == 1 else 2**30, *strides[1:])
            shape = ((2 if wide == 1 else 6), *shape[1:])
        span = 1 + sum((n - 1) * st for n, st in zip(shape, strides))
        size = 2 if dt in ("float16", "bfloat16") else 4
        alignment = 16 if i < 9 and role != "stats" else size
        ops.append(prep.Operand(dt, shape, strides, span, alignment, size))
        declared.append((shape, strides))
        facts[role] = BufferFacts(0x100000000000 + i * 2**40, dt, (2, 0), span, shape, strides)
    frames = []

    def entry(*frame):
        assert len(frame) == len(names), (names, frame)
        frames.append(frame)

    spec = prep.BwdLaunchSpec(
        object(),
        entry,
        tuple(ops),
        4096,
        0,
        0.125,
        "sdpa_bwd_" + family,
        roles=roles,
        scale_log2="scale_log2" in names,
        length_form=any(n in names for n in ("lens_form", "length_form")),
    )
    native = cudnn._pybind_module._SdpaBwdBinder(spec, tuple(declared))
    return spec, native, facts, tuple(declared), frames


def _pack(spec, facts):
    pack = cudnn._pybind_module.VariantPackNative(len(spec.operands))
    for i, role in enumerate(spec.roles):
        _set_native_fact(pack, i, facts.get(role))
    return pack


def _native(spec, native, facts, workspace=_WORKSPACE, stream=17, overridden=()):
    return native.bind(_pack(spec, facts), tuple(range(len(spec.operands))), workspace, stream, overridden)


@pytest.mark.parametrize("arch", _ARCHES)
@pytest.mark.parametrize("scale", [None, 0.0, -0.0, 0.375])
def test_standalone_uses_the_native_host_contract(arch, scale, monkeypatch):
    spec, _, facts, geometry, frames = _fixture(arch)
    expected = prep._bind_python(spec, facts, _WORKSPACE, 29, scale=scale, geometry=geometry)
    # An explicit 0 is a zero scale; only an omitted scale takes the plan's. hex() keeps -0.0 distinct from 0.0.
    slot = len(spec.operands) + 1 + spec.scale_log2
    assert expected[slot].hex() == float(spec.scale if scale is None else scale).hex()
    spec = replace(spec, native_binding=True)
    monkeypatch.setattr(prep, "_bind_python", lambda *a, **k: pytest.fail("half template entered the Python binder"))
    prep.execute(spec, facts, _WORKSPACE, 29, scale=scale, geometry=geometry)
    assert frames == [tuple(expected)]
    assert frames[0][slot].hex() == expected[slot].hex()
    changed = {role: fact._replace(ptr=fact.ptr + 0x10000) for role, fact in facts.items()}
    rebound = prep.bind(spec, changed, _WORKSPACE, 31, geometry=geometry)
    for i, role in enumerate(spec.roles):
        if role in changed:
            assert rebound[i] == changed[role].ptr
    assert rebound[-1] == 31


@pytest.mark.parametrize("arch", ["sm80", "sm100", "sm107_thd"])
@pytest.mark.parametrize("q_prefix,kv_prefix", [(False, False), (True, False), (False, True), (True, True)])
def test_native_standalone_prefix_form_and_storage(arch, q_prefix, kv_prefix):
    spec, _, facts, _, _ = _fixture(arch)
    ops = list(spec.operands)
    for role, prefix in (("seq_q", q_prefix), ("seq_kv", kv_prefix)):
        i = spec.roles.index(role)
        ops[i] = replace(ops[i], allowed_numels=(2, 3))
        count = 3 if prefix else 2
        facts[role] = facts[role]._replace(shape=(count,), strides=(1,), span=count)
    spec = replace(spec, operands=tuple(ops), native_binding=True)
    frame = prep.bind(spec, facts, _WORKSPACE, 17)
    assert frame[-2] == int(q_prefix) | (int(kv_prefix) << 1)
    for role, prefix in (("seq_q", q_prefix), ("seq_kv", kv_prefix)):
        if prefix:
            with pytest.raises(ValueError, match="backing storage"):
                prep.bind(spec, dict(facts, **{role: facts[role]._replace(span=2)}), _WORKSPACE, 17)
        with pytest.raises(ValueError, match="contiguous"):
            prep.bind(spec, dict(facts, **{role: facts[role]._replace(strides=(2,), span=6)}), _WORKSPACE, 17)


@pytest.mark.parametrize("arch", _ARCHES)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("features", [False, True])
@pytest.mark.parametrize("wide", [0, 1, 2], ids=["ordinary", "wide_stride", "wide_product"])
def test_actual_backward_host_frames_match_python(arch, dtype, features, wide):
    spec, native, facts, declared, frames = _fixture(arch, dtype, features, wide)
    held = []
    for delta, stream in ((0, 0), (2**34, 17), (2**35, 29)):
        current = {role: f._replace(ptr=f.ptr + delta) for role, f in facts.items()}
        actual = _native(spec, native, current, _WORKSPACE + delta, stream)
        expected = prep.bind(spec, current, _WORKSPACE + delta, stream, raw_storage=True)
        assert list(actual) == expected
        native.execute(_pack(spec, current), tuple(range(len(spec.operands))), _WORKSPACE + delta, stream, ())
        assert list(frames[-1]) == expected
        held.append((actual, tuple(expected)))
    assert all(actual == expected for actual, expected in held)


@pytest.mark.parametrize("arch", _ARCHES)
@pytest.mark.parametrize("role", ["q", "stats", "dq", "dk", "dv"])
@pytest.mark.parametrize("fault", ["missing", "dtype", "device", "cpu", "alignment", "short", "overlap"])
def test_backward_rejects_current_storage_after_warmup(arch, role, fault):
    spec, native, facts, declared, frames = _fixture(arch)
    _native(spec, native, facts)
    f = facts[role]
    bad = {
        "missing": None,
        "dtype": f._replace(dtype="int32" if f.dtype != "int32" else "float32"),
        "device": f._replace(device=(2, 1)),
        "cpu": f._replace(device=(1, 0)),
        "alignment": f._replace(ptr=f.ptr + 1),
        "short": f._replace(span=f.span - 1),
        "overlap": f._replace(ptr=_WORKSPACE),
    }[fault]
    changed = dict(facts, **{role: bad})
    with pytest.raises(ValueError):
        prep.bind(spec, changed, _WORKSPACE, 0, raw_storage=True)
    with pytest.raises(ValueError):
        native.execute(_pack(spec, changed), tuple(range(len(spec.operands))), _WORKSPACE, 0, ())
    assert frames == []
    assert list(_native(spec, native, facts)) == prep.bind(spec, facts, _WORKSPACE, 17, raw_storage=True)


@pytest.mark.parametrize("arch", _ARCHES)
@pytest.mark.parametrize("role", ["q", "stats", "dv"])
def test_backward_raw_storage_and_explicit_geometry(arch, role):
    spec, native, facts, declared, _ = _fixture(arch)
    i = spec.roles.index(role)
    f = facts[role]
    changed = dict(facts, **{role: f._replace(shape=(f.span,), strides=(1,))})
    # Producer shape is not a graph override. Both binders accept raw storage.
    assert list(_native(spec, native, changed)) == prep.bind(spec, changed, _WORKSPACE, 17, raw_storage=True)
    geometry = tuple(g if j == i else None for j, g in enumerate(declared))
    with pytest.raises(ValueError, match="runtime geometry"):
        prep.bind(spec, changed, _WORKSPACE, 17, geometry=geometry, raw_storage=True)
    with pytest.raises(ValueError, match="runtime geometry"):
        _native(spec, native, changed, overridden=(i,))
    # Singleton dimensions and their strides do not change addressed geometry.
    same = f._replace(shape=(*f.shape, 1), strides=(*f.strides, 999))
    assert list(_native(spec, native, dict(facts, **{role: same}), overridden=(i,))) == prep.bind(
        spec, dict(facts, **{role: same}), _WORKSPACE, 17, geometry=geometry, raw_storage=True
    )


@pytest.mark.parametrize("arch", _ARCHES)
def test_backward_optional_and_unknown_raw_addresses(arch):
    spec, native, facts, _, frames = _fixture(arch, features=False)
    unknown = {role: f._replace(dtype="", device=(-1, -1), span=-1, shape=(), strides=()) for role, f in facts.items()}
    assert list(_native(spec, native, unknown)) == prep.bind(spec, unknown, _WORKSPACE, 17, raw_storage=True)
    absent = next((role for role, op in zip(spec.roles, spec.operands) if op is None), None)
    if absent:
        changed = dict(facts, **{absent: facts["stats"]})
        with pytest.raises(ValueError, match="was not compiled"):
            _native(spec, native, changed)
    assert not frames


@pytest.mark.parametrize("workspace", [0, 1, -16, 2**63 - 16])
def test_backward_workspace_and_pointer_ranges(workspace):
    spec, native, facts, _, frames = _fixture()
    with pytest.raises(ValueError):
        native.execute(_pack(spec, facts), tuple(range(len(spec.operands))), workspace, 17, ())
    assert not frames


def test_backward_frames_do_not_share_runtime_storage():
    spec, native, facts, _, _ = _fixture()

    def bind(i):
        current = {role: f._replace(ptr=f.ptr + i * 4096) for role, f in facts.items()}
        actual = _native(spec, native, current, _WORKSPACE + i * 4096, i)
        assert list(actual) == prep.bind(spec, current, _WORKSPACE + i * 4096, i, raw_storage=True)
        return actual

    with ThreadPoolExecutor(max_workers=4) as pool:
        frames = list(pool.map(bind, range(32)))
    assert len({frame[0] for frame in frames}) == 32
    assert [frame[-1] for frame in frames] == list(range(32))


@pytest.mark.L1
@requires_dsl
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("hkv", [4, 2])
def test_half_backward_graph_uses_native_binding(dtype, causal, hkv, monkeypatch):
    import torch

    cc = torch.cuda.get_device_capability()
    if cc in ((10, 0), (10, 3)):
        from test_sdpa_bwd_dsl_sm100 import _prepared_case, _check_prepared

        case = _prepared_case(dtype=getattr(torch, dtype), causal=causal, hkv=hkv)
    elif cc == (10, 7):
        from test_sdpa_bwd_dsl_sm107 import _prepared_case, _check_prepared

        case = _prepared_case(dt=getattr(torch, dtype), causal=causal, hkv=hkv, sq=128, skv=128)
    elif cc == (8, 0):
        from test_sdpa_bwd_prepared_sm80 import _case, _check as _check_prepared

        case = _case(dtype=getattr(torch, dtype), causal=causal, hk=hkv, features=True)
        case.tensors = case.bufs
    elif cc in ((12, 0), (12, 1)):
        from test_sdpa_bwd_dsl_sm120 import _prepared_bwd_case, _check_prepared_bwd as _check_prepared

        case = _prepared_bwd_case(getattr(torch, dtype), ("mha" if hkv == 4 else "gqa") if causal else "det2k")
    else:
        pytest.skip("requires a native half backward engine")
    launch = case.graph._compiled_plans[case.graph._plan_index]._prepared
    assert getattr(launch, "_native", None) is not None
    monkeypatch.setattr(prep, "facts_of_roles", lambda *args: pytest.fail("native backward reconstructed Python facts"))
    monkeypatch.setattr(prep, "bind", lambda *args, **kwargs: pytest.fail("native backward used the Python binder"))
    torch.cuda.set_sync_debug_mode("error")
    try:
        case.graph.execute(case.pack, case.workspace)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    _check_prepared(case)
    captured = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(captured):
            case.graph.execute(case.pack, case.workspace)
        for role in ("dq", "dk", "dv"):
            case.tensors[role].fill_(float("nan"))
        captured.replay()
        _check_prepared(case)
    finally:
        captured.reset()
