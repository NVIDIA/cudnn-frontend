# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one knob vocabulary: ``KnobType_t`` values are a persisted contract, and
python engines describe their plans in it like the backend does.

No GPU work here; the frost-engine round trip through a real graph lives in
``sdpa/frost/test_knob_vocabulary_frost.py``."""

import pytest

import cudnn
from cudnn.engines import BaseEngine, PlanConfig, public_knobs_repr
from cudnn.engines.engine_ids import PYTHON_ENGINE_ID_BASE

pytestmark = pytest.mark.L0


# ---------------------------------------------------------------------------
# The enum itself
# ---------------------------------------------------------------------------


def test_backend_mirror_values_are_frozen():
    """Downstream autotune caches persist ``int(knob_type)``; the backend-mirror
    band must keep the numbers it shipped with."""
    kt = cudnn.knob_type
    frozen = {
        "NOT_SET": 0,
        "SWIZZLE": 1,
        "TILE_SIZE": 2,
        "EDGE": 3,
        "MULTIPLY": 4,
        "SPLIT_K_BUF": 5,
        "TILEK": 6,
        "STAGES": 7,
        "REDUCTION_MODE": 8,
        "SPLIT_K_SLC": 9,
        "IDX_MODE": 10,
        "SPECFILT": 11,
        "KERNEL_CFG": 12,
        "WORKSPACE": 13,
        "TILE_CGA_M": 14,
        "TILE_CGA_N": 15,
        "BLOCK_SIZE": 16,
        "OCCUPANCY": 17,
        "ARRAY_SIZE_PER_THREAD": 18,
        "SPLIT_COLS": 19,
        "TILE_ROWS": 20,
        "TILE_COLS": 21,
        "LOAD_SIZE": 22,
        "CTA_COUNT": 23,
        "STREAM_K": 24,
        "SPLIT_P_SLC": 25,
        "TILE_M": 26,
        "TILE_N": 27,
        "WARP_SPEC_CFG": 28,
        "SWAP_AB": 29,
        "INPUT_TMA_ENABLE": 30,
        "OUTPUT_TMA_ENABLE": 31,
        "TILE_CGA": 32,
    }
    backend_band = {name: int(member) for name, member in kt.__members__.items() if not cudnn.is_frontend_knob_type(member)}
    # Every member the backend band has today, with the value it shipped with;
    # a new backend-mirrored type must be APPENDED here (next free value).
    assert backend_band == frozen
    assert all(value < cudnn.FRONTEND_KNOB_TYPE_BASE for value in backend_band.values())


def test_frontend_only_band():
    kt = cudnn.knob_type
    assert cudnn.FRONTEND_KNOB_TYPE_BASE == 1000
    fe_only = {
        "SCHED_POLICY": 1000,
        "PACK_GQA": 1001,
        "SPLIT_KV": 1002,
        "PIPELINE_ARCH": 1003,
        "MMA_TILE_M": 1004,
        "MMA_TILE_N": 1005,
        "MMA_TILE_K": 1006,
        "CTA_GROUP": 1007,
        "WARPS_M": 1008,
        "WARPS_N": 1009,
    }
    # The frontend band is persisted too: append-only, frozen here like the backend band.
    assert {name: int(member) for name, member in kt.__members__.items() if cudnn.is_frontend_knob_type(member)} == fe_only
    for knob in (kt.TILE_M, kt.TILE_N, kt.TILE_CGA_M, kt.STREAM_K, kt.SPLIT_K_SLC):
        assert not cudnn.is_frontend_knob_type(knob)
    # the python enum is the C++ enum: round-trips through the integer
    assert kt(1002) == kt.SPLIT_KV and int(kt(1002)) == 1002


def test_frontend_only_knob_is_refused_by_a_backend_plan():
    """A frontend-only knob has no backend counterpart; handing one to a
    backend engine must fail loudly rather than silently mis-map."""
    handle = cudnn.create_handle()
    g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, handle=handle)
    a = g.tensor(name="a", dim=[1, 64, 64], stride=[64 * 64, 64, 1])
    b = g.tensor(name="b", dim=[1, 64, 64], stride=[64 * 64, 64, 1])
    c = g.matmul(name="mm", A=a, B=b)
    c.set_output(True)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    backend_id, _ = g.get_engine_and_knobs_at_index(0)
    # convert_to_backend_knob_type answers CUDNN_STATUS_NOT_SUPPORTED for the
    # frontend-only band; the binding surfaces that as a RuntimeError at replay.
    with pytest.raises(RuntimeError, match=r"convert_to_backend_knob_type.*CUDNN_STATUS_NOT_SUPPORTED"):
        g.create_execution_plan(backend_id, {cudnn.knob_type.SPLIT_KV: 2})


def test_backend_plan_knobs_are_a_dict_never_none():
    handle = cudnn.create_handle()
    g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, handle=handle)
    a = g.tensor(name="a", dim=[1, 128, 128], stride=[128 * 128, 128, 1])
    b = g.tensor(name="b", dim=[1, 128, 128], stride=[128 * 128, 128, 1])
    c = g.matmul(name="mm", A=a, B=b)
    c.set_output(True)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    for i in range(g.get_execution_plan_count()):
        engine_id, knobs = g.get_engine_and_knobs_at_index(i)
        assert isinstance(knobs, dict), (i, knobs)
        assert all(isinstance(k, cudnn.knob_type) for k in knobs), knobs
        assert all(isinstance(v, int) for v in knobs.values()), knobs
        assert engine_id < PYTHON_ENGINE_ID_BASE


# ---------------------------------------------------------------------------
# BaseEngine contract + the frost SDPA adapters (pure python)
# ---------------------------------------------------------------------------


class _Plain(BaseEngine):
    name = "plain"
    engine_id = PYTHON_ENGINE_ID_BASE + 99_999  # never registered; only the adapters are exercised


def test_base_engine_default_adapters():
    e = _Plain()
    assert e.knobs_to_public(None) == {}
    assert e.knobs_to_public({cudnn.knob_type.TILE_M: 128}) == {cudnn.knob_type.TILE_M: 128}
    assert e.knobs_from_public({}) is None
    assert e.knobs_from_public({cudnn.knob_type.TILE_M: 128}) == {cudnn.knob_type.TILE_M: 128}
    with pytest.raises(TypeError):
        e.knobs_to_public(object())  # a typed knob object needs an engine override


def test_public_knobs_repr_is_sorted_by_name():
    kt = cudnn.knob_type
    assert public_knobs_repr({kt.TILE_N: 128, kt.TILE_M: 64, kt.SPLIT_KV: 2}) == "SPLIT_KV=2, TILE_M=64, TILE_N=128"
    assert public_knobs_repr({}) == ""


def test_sdpa_fwd_knobs_round_trip():
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    kt = cudnn.knob_type
    native = SdpaFwdKnobs(sched_policy=1, tile_m=128, tile_n=128, cga=2, pack_gqa=True, split_kv=4)
    public = native.to_public()
    assert public == {
        kt.SCHED_POLICY: 1,
        kt.TILE_M: 128,
        kt.TILE_N: 128,
        kt.TILE_CGA_M: 2,
        kt.PACK_GQA: 1,
        kt.SPLIT_KV: 4,
    }
    assert SdpaFwdKnobs.from_public(public) == native
    # A persisted record is ints; anything that would round-trip to a different
    # native knob than the one recorded is refused rather than coerced.
    assert SdpaFwdKnobs.from_public({kt.PACK_GQA: True}) == SdpaFwdKnobs(pack_gqa=True)
    for bad in ({kt.PACK_GQA: 2}, {kt.PACK_GQA: "0"}, {kt.TILE_M: 128.0}, {kt.SPLIT_KV: "4"}, {kt.SCHED_POLICY: True}):
        with pytest.raises(ValueError):
            SdpaFwdKnobs.from_public(bad)
    # numerics-changing requests are not knobs: the softmax accumulator precision
    # is the sdpa(softmax_precision=) op attribute, not a SdpaFwdKnobs field
    assert "softmax_precision" not in SdpaFwdKnobs.__dataclass_fields__
    assert not hasattr(kt, "SOFTMAX_PRECISION")
    # unset fields do not appear; integer keys (a persisted record) are accepted
    assert SdpaFwdKnobs(tile_m=64).to_public() == {kt.TILE_M: 64}
    assert SdpaFwdKnobs.from_public({int(kt.TILE_M): 64, int(kt.PACK_GQA): 0}) == SdpaFwdKnobs(tile_m=64, pack_gqa=False)
    with pytest.raises(ValueError):
        SdpaFwdKnobs.from_public({kt.SPLIT_K_SLC: 2})  # not an SDPA-forward axis


def test_sdpa_bwd_knobs_round_trip():
    from cudnn.sdpa.bwd.engines import SdpaBwdKnobs

    kt = cudnn.knob_type
    native = SdpaBwdKnobs(tile_m=64, tile_n=128)
    assert native.to_public() == {kt.TILE_M: 64, kt.TILE_N: 128}
    assert SdpaBwdKnobs.from_public(native.to_public()) == native
    with pytest.raises(ValueError):
        SdpaBwdKnobs.from_public({kt.SPLIT_KV: 2})


def test_frost_sdpa_engines_speak_the_public_vocabulary(monkeypatch):
    """The engine-level adapters used by pygraph at the public boundary."""
    from cudnn.engines import manifest
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")  # the slots are opt-in
    kt = cudnn.knob_type
    fam = next(f for f in manifest.MANIFEST if f.name == "frost_sdpa_fwd")
    engine = manifest.engine_for_id(fam.offered_ids()["sdpa_fwd_prefill_sm100"])
    assert engine is not None
    public = {kt.TILE_M: 128, kt.TILE_N: 128, kt.SPLIT_KV: 2}
    native = engine.knobs_from_public(public)
    assert isinstance(native, SdpaFwdKnobs) and native.split_kv == 2
    assert engine.knobs_to_public(native) == public
    assert engine.knobs_to_public(None) == {}
    assert engine.knobs_from_public({}) is None
    # PlanConfig keeps the native form; only the boundary speaks public
    assert PlanConfig(engine.engine_id, native).knobs is native
