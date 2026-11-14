# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host contracts for the independent small-head per-tensor FP8 records."""

from dataclasses import asdict
import importlib
from pathlib import Path
import subprocess
import sys

import pytest

import cudnn
from cudnn.sdpa.fwd import config_sm100 as legacy
from cudnn.sdpa.fwd import engines, heuristics
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0


@pytest.mark.parametrize(
    "variant,record",
    [
        ("multi_q", "CfgD64"),
        ("multi_q", "CfgD128"),
        ("dual_kv", "CfgD128"),
        ("swap_ab", "CfgD128Fp8Decode"),
    ],
)
def test_configuration_defaults_are_independent(variant, record, monkeypatch):
    module = importlib.import_module(f"cudnn.sdpa.fwd.kernels.sm100.d128_fp8_{variant}.config")
    own_record = getattr(module, record)
    expected = asdict(getattr(legacy, record)())
    assert own_record.__bases__ == (object,)
    assert asdict(own_record()) == expected
    # Class-level role tuning in a legacy sibling must not change this body.
    monkeypatch.setattr(getattr(legacy, record), "STAGES_KV", 999)
    assert own_record.STAGES_KV == expected["STAGES_KV"]


@pytest.mark.parametrize("dual", [False, True])
def test_prefill_factory_cannot_switch_main_implementation(dual):
    variant = "dual_kv" if dual else "multi_q"
    module = importlib.import_module(f"cudnn.sdpa.fwd.kernels.sm100.d128_fp8_{variant}.config")
    params = legacy.TemplateParams(dtype_qkv=0, dtype_o=2, cta_mma=1, dual_kv_stream=dual)
    config, _ = module.make_cfg_d128(params)
    assert bool(config.DUAL_KV_STREAM) == dual
    with pytest.raises(ValueError, match="different FP8 main implementation"):
        module.make_cfg_d128(legacy.TemplateParams(dtype_qkv=0, dual_kv_stream=not dual))


@pytest.mark.parametrize("variant", ["multi_q", "dual_kv", "swap_ab"])
def test_factories_do_not_call_legacy_implementations(variant, monkeypatch):
    module = importlib.import_module(f"cudnn.sdpa.fwd.kernels.sm100.d128_fp8_{variant}.config")

    def forbidden(*args, **kwargs):
        raise AssertionError("owner consulted a legacy implementation factory")

    for name in ("make_cfg_d64", "make_cfg_d128", "make_cfg_d128_fp8_decode", "_validate_cfg_d128"):
        monkeypatch.setattr(legacy, name, forbidden)
    if variant == "swap_ab":
        params = legacy.TemplateParamsDecode(dtype_qkv=0, cta_mma=1, decode_q_tile=16, decode_live_columns=8)
        config, _ = module.make_cfg_d128_fp8_decode(params)
        assert config.N_Q == 16
    else:
        params = legacy.TemplateParams(dtype_qkv=0, cta_mma=1, dual_kv_stream=variant == "dual_kv")
        config, _ = module.make_cfg_d128(params)
        assert config.TILE_K == 128
        if variant == "multi_q":
            config, _ = module.make_cfg_d64(params)
            assert config.TILE_K == 64


def _facts(**overrides):
    values = dict(
        b=2,
        h_q=8,
        h_kv=2,
        s_q=1,
        s_kv=4096,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.FP8_E4M3,
        dtype_o=cudnn.data_type.BFLOAT16,
        is_fp8=True,
        uniform_dtype=True,
        device_cc=(10, 0),
    )
    values.update(overrides)
    return SdpaGraphFacts(**values)


@pytest.mark.parametrize(
    "overrides,knobs,expected",
    [
        ({}, dict(cga=1, pack_gqa=True), "dual_kv"),
        ({}, dict(cga=1, pack_gqa=True, swap_ab=1), "swap_ab"),
        ({}, dict(cga=2, pack_gqa=True), "multi_q"),
        ({}, dict(pack_gqa=True), "multi_q"),
        ({"s_q": 33}, dict(cga=1, pack_gqa=True), "multi_q"),
        ({"s_q": 32}, dict(cga=1, pack_gqa=True), "dual_kv"),
        ({"has_paged_kv": True}, dict(cga=1, pack_gqa=True), "swap_ab"),
        ({"has_paged_kv": True, "s_q": 3}, dict(cga=1, pack_gqa=True), "dual_kv"),
        ({"has_paged_kv": True, "device_cc": (10, 3)}, dict(cga=1, pack_gqa=True), "dual_kv"),
        ({"has_paged_kv": True, "b": 65536}, dict(cga=1, pack_gqa=True), "dual_kv"),
        ({"has_paged_kv": True, "thd": True}, dict(cga=1, pack_gqa=True), "multi_q"),
        ({"has_paged_kv": True, "o_block_scale": 32}, dict(cga=1, pack_gqa=True), "dual_kv"),
        ({"has_paged_kv": True, "d_qk": 64, "d_v": 64, "s_q": 8}, dict(cga=1, pack_gqa=True, swap_ab=1), "swap_ab"),
        ({"has_paged_kv": True, "d_qk": 64, "d_v": 64, "s_q": 8}, dict(cga=1, pack_gqa=True, swap_ab=0), "multi_q"),
    ],
)
def test_central_selection_preserves_body_boundaries(overrides, knobs, expected, monkeypatch):
    monkeypatch.setattr(legacy, "FP8_DUAL_KV_STREAM_ROUTED", True)
    selected = heuristics.select_sm100_small_fp8_geometry(_facts(**overrides), engines.SdpaFwdKnobs(**knobs))
    assert selected.implementation == f"sdpa_fwd_sm100_d128_fp8_{expected}"


@pytest.mark.parametrize("variant", ["multi_q", "dual_kv", "swap_ab"])
def test_fixed_host_never_reenters_selection(variant, monkeypatch):
    host = importlib.import_module(f"cudnn.sdpa.fwd.kernels.sm100.d128_fp8_{variant}.host")

    def forbidden(*args, **kwargs):
        raise AssertionError("fixed host reentered implementation selection")

    monkeypatch.setattr(heuristics, "select_sm100_small_fp8_geometry", forbidden)
    monkeypatch.setattr(heuristics, "recommend", forbidden)
    calls = []
    host.lower(None, _facts(), engines.SdpaFwdKnobs(cga=1), lower_impl=lambda *args, **kwargs: calls.append((args, kwargs)), knob_type=engines.SdpaFwdKnobs)
    assert len(calls) == 1
    args, kwargs = calls[0]
    selected = kwargs["selected_kernel"]
    assert selected.implementation.endswith(variant)
    assert engines.SdpaFwdKnobs.from_public(args[2].to_public()) == args[2]
    params = selected.configure_params(legacy.TemplateParams(dtype_qkv=0, dtype_o=2))
    # Existing global routing switches cannot change a retained implementation.
    monkeypatch.setattr(legacy, "FP8_DUAL_KV_STREAM_ROUTED", False)
    assert selected.configure_params(params) == params
    cfg_mod = importlib.import_module(f"cudnn.sdpa.fwd.kernels.sm100.d128_fp8_{variant}.config")
    factory = cfg_mod.make_cfg_d128_fp8_decode if variant == "swap_ab" else cfg_mod.make_cfg_d128
    cfg, _ = factory(params)
    if variant == "swap_ab":
        assert (args[2].tile_m, cfg.N_Q, params.decode_live_columns) == (16, 16, 8)
        assert args[2].sched_policy == 0
    else:
        assert bool(cfg.DUAL_KV_STREAM) == (variant == "dual_kv")


@pytest.mark.parametrize("variant", ["multi_q", "dual_kv", "swap_ab"])
def test_cold_host_import_does_not_import_registry(variant):
    package_root = str(Path(cudnn.__file__).resolve().parent.parent)
    code = "\n".join(
        (
            "import sys",
            "sys.meta_path[:] = [f for f in sys.meta_path if 'editable' not in str(f).lower()]",
            f"sys.path.insert(0, {package_root!r})",
            f"import cudnn.sdpa.fwd.kernels.sm100.d128_fp8_{variant}.host",
            "assert 'cudnn.sdpa.fwd.engines' not in sys.modules",
        )
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
