# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Fused split binding retains the final outputs alongside FP32 partials."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import sdpa_binding_reference as reference

import cudnn
from cudnn.sdpa.fwd import prepared as prep
from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100
from test_sdpa_native_dense_binding import _pack
from test_sdpa_native_split_binding import _split_fixture

pytestmark = [pytest.mark.L0]


def _fused_fixture(stats):
    spec, facts, frames, combined = _split_fixture(d=256, splits=8, lengths=False, lse=stats)
    path = Path(prep.__file__).parent / "kernels/sm107/decode_d256_f16.py"
    host = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_host")
    values = dict(zip(spec.order, spec.template))
    spec.order = [arg.arg for arg in host.args.args if "Constexpr" not in ast.unparse(arg.annotation)]
    spec.index = {name: i for i, name in enumerate(spec.order)}
    spec.template = [values.get(name) for name in spec.order]
    spec.fused_split = True
    spec.native = cudnn._pybind_module._SdpaDenseBinder(spec)
    return spec, facts, frames, combined


@pytest.mark.parametrize("stats", [False, True])
def test_fused_split_native_python_frames_and_single_launch(stats):
    spec, facts, frames, combined = _fused_fixture(stats)
    # Independent final pitches, including >32-bit batch products. The scalar
    # final store requires only half alignment, unlike the FP32 partial slab.
    strides = (2**33, 257, 8 * 257, 1)
    facts["o"] = facts["o"]._replace(ptr=0x4002, strides=strides, span=1 + sum((n - 1) * s for n, s in zip(facts["o"].shape, strides)))
    if stats:
        strides = (2**34, 3, 1)
        facts["lse"] = facts["lse"]._replace(strides=strides, span=1 + sum((n - 1) * s for n, s in zip(facts["lse"].shape, strides)))
    for offset, stream in [(0, 17), (0x200000, 29)]:
        current = {role: f._replace(ptr=f.ptr + offset) for role, f in facts.items()}
        workspace = 2**40 + offset
        expected, combine = reference.bind_dense_split(spec, current, workspace, stream, stream)
        for name, value in {
            "final_o_ptr": current["o"].ptr,
            "final_lse_ptr": current["lse"].ptr if stats else None,
            # The existing scalar-combine oracle normalizes unused singleton
            # axes; the fused store consumes exactly that same geometry.
            "final_o_strides": combine[6][:3],
            "final_lse_strides": combine[7],
        }.items():
            expected[spec.index[name]] = value
        actual = spec.native.bind_split(_pack(current), prep._NATIVE_DENSE_INDICES, workspace, stream)
        python_frame, python_combine = prep.bind_dense_split(spec, current, workspace, stream, stream)
        assert list(actual[0]) == python_frame == expected
        assert actual[1] == python_combine == combine
        spec.native.execute(_pack(current), prep._NATIVE_DENSE_INDICES, stream, workspace=workspace)
        assert frames[-1] == tuple(expected)
    assert len(frames) == 2 and combined == []


@pytest.mark.parametrize("role", ["q", "k", "v", "o", "lse"])
@pytest.mark.parametrize("change", ["short", "missing", "wrong_device", "wrong_dtype", "misaligned"])
def test_fused_split_rejects_changed_storage_before_launch(role, change):
    spec, facts, frames, combined = _fused_fixture(True)
    spec.native.bind_split(_pack(facts), prep._NATIVE_DENSE_INDICES, 0x10000000, 17)
    changed = dict(facts)
    f = facts[role]
    if change == "missing":
        del changed[role]
    else:
        updates = {
            "short": dict(span=0),
            "wrong_device": dict(device=(2, 1)),
            "wrong_dtype": dict(dtype="float32" if f.dtype != "float32" else "bfloat16"),
            "misaligned": dict(ptr=f.ptr + 1),
        }
        changed[role] = f._replace(**updates[change])
    with pytest.raises(ValueError):
        spec.native.execute(_pack(changed), prep._NATIVE_DENSE_INDICES, 17, workspace=0x10000000)
    with pytest.raises(ValueError):
        prep.bind_dense_split(spec, changed, 0x10000000, 17, 17)
    assert frames == combined == []


def test_fused_split_rejects_workspace_alias_before_launch():
    spec, facts, frames, combined = _fused_fixture(True)
    with pytest.raises(ValueError, match="overlaps"):
        spec.native.execute(_pack(facts), prep._NATIVE_DENSE_INDICES, 17, workspace=facts["o"].ptr)
    assert frames == combined == []


@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize("change", [None, "old_extension", "sq", "cc", "pdl", "log2", "gate", "kernel_decline"])
def test_cluster_compile_selection_preserves_other_plans(monkeypatch, stats, change):
    def compile(*, has_lse=True, lse_kind="dense", d_qk=256, d_v=256, use_pdl=False, cluster_combine=False, has_final_lse=True):
        raise AssertionError("host-only selection must not compile")

    km = SimpleNamespace(compile=compile, cluster_combine_supported=lambda dq, dv: change != "kernel_decline")
    api = SimpleNamespace(
        packed_thd_split=False,
        thd=False,
        thd_decode_leg=False,
        _k_mod=km,
        lse_desc=object() if stats else None,
        paged=False,
        use_pdl=change == "pdl",
        stats_log2=change == "log2",
        gate_desc=object() if change == "gate" else None,
        _device_cc=(10, 0) if change == "cc" else (10, 7),
        s_q_max=2 if change == "sq" else 1,
        head_dim_qk=256,
        head_dim_v=256,
        split_kv=8,
    )
    monkeypatch.setattr(cudnn._pybind_module, "_SdpaDenseBinder", SimpleNamespace(supports_fused_split=change != "old_extension"))
    result = SdpaFwdDslSm100._explicit_compile_kwargs(api)
    assert api._cluster_combine == result["cluster_combine"] == (change is None)
    assert result["has_final_lse"] == stats
    assert result["has_lse"] is True  # partial LSE is needed even for O-only
