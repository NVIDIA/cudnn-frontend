# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed SM80 frames, caller-specific storage contracts and native execution."""

import ast
import math
from dataclasses import replace
from pathlib import Path

import cudnn
import pytest

from frost_test_utils import requires_dsl
from sm80_binding_reference import bind as reference_bind
from cudnn.sdpa.fwd import prepared_sm80 as prep
from cudnn.sdpa.fwd.prepared import BufferFacts, _set_native_fact

pytestmark = [pytest.mark.L0]
_INDICES = tuple(range(9))


def _fixture(dtype="bfloat16", features=True, wide=0, rope=False):
    path = Path(prep.__file__).parent / "kernels/sm80/prepared_host.py"
    host = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == ("_dense_host" if rope else "host"))
    names = [a.arg for a in host.args.args if "Constexpr" not in ast.unparse(a.annotation)]
    roles = prep.ROLES[: 10 if rope else 9]
    assert tuple(names[:-3]) == roles
    assert names[-3:] == ["scale_log2", "inv_scale", "stream"]
    ops, facts, frames = [], {}, []
    for i, role in enumerate(roles):
        if i >= 4 and not features and role != "rope":
            ops.append(None)
            continue
        dt = dtype if i < 4 else "int32" if role.startswith("seq_") else "float32"
        if i < 4:
            shape, strides = (2, 3, 5, 8), (120, 8, 24, 1)
            if wide:
                shape = (2 if wide == 1 else 6, *shape[1:])
                strides = (2**32 + 120 if wide == 1 else 2**30, *strides[1:])
        elif role == "rope":
            shape, strides = (7, 4, 2), (8, 2, 1)
        elif role == "stats":
            shape, strides = (2, 3, 5, 1), (30, 10, 2, 1)
        elif role == "bias":
            shape, strides = (1, 3, 5, 7), (105, 35, 7, 1)
        else:
            shape, strides = (3 if role == "sink" else 2,), (1,)
        span = 1 + sum((n - 1) * st for n, st in zip(shape, strides))
        op = prep.Operand(shape, strides, dt, span, 16 if i < 4 else 4, role not in ("stats",) and not wide)
        ops.append(op)
        facts[role] = BufferFacts(0x100000000000 + i * 2**40, dt, (2, 0), span, shape, strides)

    def record(*frame):
        assert len(frame) == len(names)
        frames.append(frame)

    spec = prep.LaunchSpec(object(), record, tuple(ops), 0, 0.125)
    return spec, facts, frames


def _pack(facts):
    roles = prep.ROLES[: 10 if "rope" in facts else 9]
    pack = cudnn._pybind_module.VariantPackNative(len(roles))
    for i, role in enumerate(roles):
        _set_native_fact(pack, i, facts.get(role))
    return pack


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("features", [False, True])
@pytest.mark.parametrize("wide", [0, 1, 2])
@pytest.mark.parametrize("scale", [None, 0.0, 0.25, -0.5])
@pytest.mark.parametrize("rope", [False, True])
def test_sm80_actual_host_frame_matches_python(dtype, features, wide, scale, rope):
    spec, facts, frames = _fixture(dtype, features, wide, rope)
    indices = tuple(range(len(spec.operands)))
    if scale == 0.0:
        # 0 is a zero scale, not "use the plan's"; the kernel cannot run it (#1435), and both binders refuse it alike.
        with pytest.raises(ValueError, match="#1435"):
            prep.bind(spec, facts, 0, scale=scale)
        with pytest.raises(ValueError, match="#1435"):
            spec.native.bind(_pack(facts), indices, 0, scale, (), False)
        assert not frames
        return
    resolved = spec.scale if scale is None else scale
    held = []
    for delta, stream in ((0, 0), (2**34, 17), (2**35, 29)):
        current = {role: f._replace(ptr=f.ptr + delta) for role, f in facts.items()}
        expected = reference_bind(spec, current, stream, scale=scale)
        actual = spec.native.bind(_pack(current), indices, stream, scale, (), False)
        assert list(actual) == expected
        assert expected[-3:-1] == [resolved * math.log2(math.e), 1.0 / resolved]
        assert prep.bind(spec, current, stream, scale=scale) == expected
        spec.native.execute(_pack(current), indices, stream, scale, (), False)
        assert list(frames[-1]) == expected
        held.append((actual, tuple(expected)))
    assert all(actual == expected for actual, expected in held)


@pytest.mark.parametrize("role", prep.ROLES[:9])
@pytest.mark.parametrize("bad", ["missing", "dtype", "device", "cpu", "alignment", "short"])
def test_sm80_revalidates_every_current_operand(role, bad):
    spec, facts, frames = _fixture()
    spec.native.bind(_pack(facts), _INDICES, 0, None, (), True)
    f = facts[role]
    changed = {
        "missing": None,
        "dtype": f._replace(dtype="int32" if f.dtype != "int32" else "float32"),
        "device": f._replace(device=(2, 1)),
        "cpu": f._replace(device=(1, 0)),
        "alignment": f._replace(ptr=f.ptr + 1),
        "short": f._replace(span=f.span - 1),
    }[bad]
    current = dict(facts, **{role: changed})
    with pytest.raises(ValueError):
        reference_bind(spec, current, 0, raw_storage=True)
    with pytest.raises(ValueError):
        spec.native.execute(_pack(current), _INDICES, 0, None, (), True)
    assert not frames


@pytest.mark.parametrize("role", prep.ROLES[:9])
def test_sm80_graph_storage_and_overrides_are_distinct(role):
    spec, facts, _ = _fixture()
    index = prep.ROLES.index(role)
    f = facts[role]
    current = dict(facts, **{role: f._replace(shape=(f.span + 1,), strides=(1,))})
    assert list(spec.native.bind(_pack(current), _INDICES, 17, None, (), True)) == reference_bind(spec, current, 17, raw_storage=True)
    with pytest.raises(ValueError, match="runtime geometry"):
        spec.native.bind(_pack(current), _INDICES, 17, None, (index,), True)
    singleton = dict(facts, **{role: f._replace(shape=(*f.shape, 1), strides=(*f.strides, 999))})
    assert list(spec.native.bind(_pack(singleton), _INDICES, 17, None, (index,), True)) == reference_bind(
        spec, singleton, 17, overridden={role}, raw_storage=True
    )


@pytest.mark.parametrize("role", ["seq_q", "seq_kv", "sink", "stats", "bias"])
@pytest.mark.parametrize("valid", [False, True])
def test_sm80_standalone_carriers_keep_their_contract(role, valid):
    spec, facts, _ = _fixture()
    f = facts[role]
    if role == "bias":
        # Only the first [H,SQ,SKV] plane participates, even with a larger B carrier.
        shape, strides = (2, *f.shape[1:]), f.strides
        if not valid:
            strides = (strides[0], strides[1] + 1, *strides[2:])
    else:
        count = f.numel if valid else f.numel + 1
        shape, strides = (count, 1), (1, 999)
    changed = dict(facts, **{role: f._replace(shape=shape, strides=strides, span=max(f.span, 4096))})
    if valid:
        assert list(spec.native.bind(_pack(changed), _INDICES, 17, None, (), False)) == reference_bind(spec, changed, 17)
    else:
        with pytest.raises(ValueError):
            reference_bind(spec, changed, 17)
        with pytest.raises(ValueError):
            spec.native.bind(_pack(changed), _INDICES, 17, None, (), False)


def test_sm80_unknown_storage_and_checked_int64_ranges():
    spec, facts, frames = _fixture(features=False)
    unknown = {role: f._replace(dtype="", device=(-1, -1), span=-1, shape=(), strides=()) for role, f in facts.items()}
    assert list(spec.native.bind(_pack(unknown), _INDICES, 17, None, (), True)) == reference_bind(spec, unknown, 17, raw_storage=True)
    for changed in (dict(facts, sink=facts["q"]), dict(facts, q=facts["q"]._replace(ptr=2**63 - 16))):
        with pytest.raises(ValueError):
            spec.native.execute(_pack(changed), _INDICES, 17, None, (), True)
    with pytest.raises(ValueError, match="int64"):
        replace(spec, operands=(replace(spec.operands[0], span=2**62), *spec.operands[1:]))
    assert not frames


@pytest.mark.parametrize("fault", ["missing", "dtype", "device", "short", "geometry"])
def test_sm80_rope_is_validated_before_host_launch(fault):
    spec, facts, frames = _fixture(rope=True)
    rope = facts["rope"]
    changed = {
        "missing": None,
        "dtype": rope._replace(dtype="float16"),
        "device": rope._replace(device=(2, 1)),
        "short": rope._replace(span=rope.span - 1),
        "geometry": rope._replace(shape=(rope.numel,), strides=(1,)),
    }[fault]
    with pytest.raises(ValueError):
        prep.execute(spec, dict(facts, rope=changed), 17)
    assert not frames
    prep.execute(spec, facts, 19, scale=0.25)
    assert frames == [tuple(reference_bind(spec, facts, 19, scale=0.25))]


@requires_dsl
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("dq,dv", [(64, 64), (96, 128), (192, 128), (256, 256)])
@pytest.mark.parametrize("standalone", [False, True])
def test_sm80_native_fresh_bindings_capture_and_scale(dtype, dq, dv, standalone, monkeypatch):
    import torch
    import cutlass.cute as cute
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm80
    from cudnn.sdpa.fwd import prepared as common
    from test_sdpa_prepared_sm80 import _case, _check

    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("requires SM80")
    case = _case(dq, dv, dtype=getattr(torch, dtype), features=True, layout="gapped", causal=True)
    plan = case.graph._compiled_plans[case.graph._plan_index]._prepared
    assert getattr(plan.spec, "native", None) is not None
    api = None
    if standalone:
        api = SdpaFwdDslSm80(
            *(case.bufs[name] for name in ("q", "k", "v", "o", "stats")),
            has_sink=True,
            seq_q_lens_present=True,
            seq_kv_lens_present=True,
            bias_present=True,
            bias_fp32=True,
            is_causal=True,
        )
        api.check_support()
        api.compile()
        assert api._sm80_spec.native is not None

    def forbidden(*args, **kwargs):
        pytest.fail("native execute must not use Python facts/binding or compile")

    monkeypatch.setattr(prep, "bind", forbidden)
    monkeypatch.setattr(common, "facts_of_tensor", forbidden)
    monkeypatch.setattr(cute, "compile", forbidden)

    def execute(scale=None):
        if api is None:
            case.graph.execute(case.pack, case.workspace)
        else:
            api.execute(
                *(case.bufs[name] for name in ("q", "k", "v", "o", "stats")),
                sinks=case.bufs["sink"],
                seq_q_lens=case.bufs["seq_q"],
                seq_kv_lens=case.bufs["seq_kv"],
                bias_tensor=case.bufs["bias"],
                scale_softmax=scale,
            )

    for iteration in range(2):
        if iteration:
            for role, old in tuple(case.bufs.items()):
                case.bufs[role] = torch.empty_strided(old.shape, old.stride(), dtype=old.dtype, device="cuda").copy_(old)
                case.pack[case.refs[role]] = case.bufs[role]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured, stream=stream):
            execute()
        torch.cuda.current_stream().wait_stream(stream)
        case.bufs["v"].mul_(0.7)
        case.bufs["o"].fill_(float("nan"))
        case.bufs["stats"].fill_(float("nan"))
        captured.replay()
        _check(case)
        torch.cuda.set_sync_debug_mode("error")
        try:
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            execute()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
        finally:
            torch.cuda.set_sync_debug_mode("default")
        _check(case)
        captured.reset()
    if standalone:
        # Changing scale is equivalent to scaling Q for this reference, including
        # the independently added bias/sinks; the plan's stored scale stays fixed.
        scale = api._sm80_spec.scale
        execute(scale * 2)
        saved_q = case.bufs["q"]
        case.bufs["q"] = saved_q * 2
        try:
            _check(case)
        finally:
            case.bufs["q"] = saved_q
        assert api._sm80_spec.scale == scale


@requires_dsl
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("explicit_stream", [False, True])
def test_sm80_direct_standalone_with_another_device_current(dtype, explicit_stream, monkeypatch):
    """Launch on Q's device and restore the caller, including capture on Q's stream."""
    import torch
    from cuda.bindings import driver
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm80
    from test_sdpa_prepared_sm80 import _case, _check

    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    target = next((i for i in range(torch.cuda.device_count()) if torch.cuda.get_device_capability(i) == (8, 0)), None)
    if target is None:
        pytest.skip("requires an SM80 device")
    other = next(i for i in range(torch.cuda.device_count()) if i != target)
    with torch.cuda.device(target):
        case = _case(128, 128, dtype=getattr(torch, dtype), features=True, layout="gapped", causal=True)
        api = SdpaFwdDslSm80(
            *(case.bufs[name] for name in ("q", "k", "v", "o", "stats")),
            has_sink=True,
            seq_q_lens_present=True,
            seq_kv_lens_present=True,
            bias_present=True,
            bias_fp32=True,
            is_causal=True,
        )
        api.check_support()
        api.compile()
        assert api._sm80_spec is not None and api._sm80_copy_spec is None
        stream = torch.cuda.Stream(device=target)
        stream.wait_stream(torch.cuda.current_stream())
        captured = torch.cuda.CUDAGraph()
        real_execute = prep.execute_tensors

        def checked_execute(*args, **kwargs):
            assert torch.cuda.current_device() == target, "SM80 direct launch must run in Q's context"
            assert args[2] == stream.cuda_stream, "the implicit stream must belong to Q's device"
            return real_execute(*args, **kwargs)

        monkeypatch.setattr(prep, "execute_tensors", checked_execute)

        def execute():
            with torch.cuda.device(other):
                api.execute(
                    *(case.bufs[name] for name in ("q", "k", "v", "o", "stats")),
                    sinks=case.bufs["sink"],
                    seq_q_lens=case.bufs["seq_q"],
                    seq_kv_lens=case.bufs["seq_kv"],
                    bias_tensor=case.bufs["bias"],
                    current_stream=driver.CUstream(stream.cuda_stream) if explicit_stream else None,
                )
                assert torch.cuda.current_device() == other

        with torch.cuda.stream(stream):
            execute()
        torch.cuda.current_stream().wait_stream(stream)
        _check(case)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(captured, stream=stream):
            execute()
        torch.cuda.current_stream().wait_stream(stream)
        case.bufs["v"].mul_(0.5)
        case.bufs["o"].fill_(float("nan"))
        case.bufs["stats"].fill_(float("nan"))
        captured.replay()
        _check(case)
        captured.reset()
        # Validation failure must restore the caller's device as well.
        original = case.bufs["k"]
        case.bufs["k"] = original.to(dtype=torch.float32)
        try:
            with torch.cuda.stream(stream), torch.cuda.device(other):
                with pytest.raises(ValueError, match="k must be"):
                    execute()
                assert torch.cuda.current_device() == other
        finally:
            case.bufs["k"] = original
