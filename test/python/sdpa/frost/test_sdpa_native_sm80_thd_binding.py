# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The native packed binder preserves the actual SM80 pointer-host ABI."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import cudnn
import pytest

from cudnn.sdpa.fwd import prepared_sm80_thd
from cudnn.sdpa.fwd.prepared import BufferFacts, _native_pack_from_facts

pytestmark = [pytest.mark.L0]
ROLES = ("q", "k", "v", "o", "stats", "cu_q", "cu_k", "sink")


def _fixture(dtype="bfloat16", prefix_dtype="int32", sink=True, wide=0, empty=False):
    tq, tk = (0 if empty else 13), 17
    frames = []
    spec = SimpleNamespace(
        fn=lambda *frame: frames.append(frame),
        artifact=object(),
        device_index=0,
        heads=(4, 2),
        dimensions=(64, 64),
        n_seq=2,
        dtype=dtype,
        prefix_dtypes=(prefix_dtype, prefix_dtype),
        sink_dtype="float32",
        has_sink=sink,
    )
    facts = {}
    for i, role in enumerate(ROLES):
        if role == "sink" and not sink:
            continue
        if i < 4:
            t, h = (tq, 4) if i in (0, 3) else (tk, 2)
            shape, strides = (1, t, h, 64), (t * h * 64, h * 64, 64, 1)
            if wide and i < 3:
                strides = (0, 2**32 if wide == 1 else 2**30, 64, 1)
            dt = dtype
        elif role == "stats":
            shape, strides, dt = (1, 4, tq), (4 * max(tq, 1), max(tq, 1), 1), "float32"
        else:
            # Prefix and sink strides are ABI values, not mandatory compact carriers.
            shape, strides, dt = (4 if role == "sink" else 3,), (2,), "float32" if role == "sink" else prefix_dtype
        count = math.prod(shape)
        span = 1 + sum((n - 1) * st for n, st in zip(shape, strides)) if count else 0
        facts[role] = BufferFacts(2**48 + i * 2**44 if count else 0, dt, (2, 0), span, shape, strides)
    return spec, cudnn._pybind_module._SdpaSm80ThdBinder(spec), facts, frames


def _expected(facts, max_sq=19, scale=0.125, stream=17):
    return (
        *(facts[r].ptr if r in facts else None for r in ROLES),
        facts["q"].shape[1],
        facts["k"].shape[1],
        max_sq,
        *(stride for role in ROLES[:3] for stride in facts[role].strides[1:3]),
        scale * math.log2(math.e),
        1 / scale,
        0,
        tuple(facts[r].strides[0] if r in facts else 1 for r in ("cu_q", "cu_k", "sink")),
        stream,
    )


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("prefix_dtype", ["int32", "int64"])
@pytest.mark.parametrize("sink", [False, True])
@pytest.mark.parametrize("wide", [0, 1, 2])
def test_packed_native_frames_match_host_signature_and_rebind(dtype, prefix_dtype, sink, wide):
    spec, launch, facts, frames = _fixture(dtype, prefix_dtype, sink, wide)
    path = Path(prepared_sm80_thd.__file__).parent / "kernels/sm80/prepared_host.py"
    host = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "thd_host")
    names = [a.arg for a in host.args.args if "Constexpr" not in ast.unparse(a.annotation)]
    assert names == [
        *ROLES,
        "t_q",
        "t_kv",
        "max_sq",
        "q_s",
        "q_h",
        "k_s",
        "k_h",
        "v_s",
        "v_h",
        "scale_log2",
        "inv_scale",
        "right_bound",
        "aux_strides",
        "stream",
    ]
    for delta, stream in ((0, 17), (2**35, 29)):
        current = {r: f._replace(ptr=f.ptr + delta) for r, f in facts.items()}
        launch.execute(_native_pack_from_facts(current, ROLES), 19, 0.125, 0, stream)
        assert frames[-1] == _expected(current, stream=stream)
    assert frames[0][0] != frames[1][0]


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("fault", ["missing", "dtype", "device", "alignment", "short"])
def test_packed_rejects_bad_storage_before_host(role, fault):
    _, launch, facts, frames = _fixture()
    f = facts[role]
    bad = {
        "missing": None,
        "dtype": f._replace(dtype="uint8"),
        "device": f._replace(device=(2, 1)),
        "alignment": f._replace(ptr=f.ptr + 1),
        "short": f._replace(span=f.span - 1),
    }[fault]
    with pytest.raises(ValueError):
        launch.execute(_native_pack_from_facts(dict(facts, **{role: bad}), ROLES), 19, 0.125, 0, 17)
    assert frames == []


def test_empty_packed_query_capacity_has_no_live_address():
    _, launch, facts, frames = _fixture(empty=True)
    launch.execute(_native_pack_from_facts(facts, ROLES), 19, 0.125, 0, 17)
    assert frames == [_expected(facts)]


def test_packed_spans_cannot_overflow_int64():
    _, launch, facts, frames = _fixture()
    q = facts["q"]._replace(span=-1, strides=(0, 2**62, 64, 1))
    with pytest.raises(ValueError, match="int64"):
        launch.execute(_native_pack_from_facts(dict(facts, q=q), ROLES), 19, 0.125, 0, 17)
    assert frames == []
