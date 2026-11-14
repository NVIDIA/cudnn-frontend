# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host contracts for the independent small-head per-tensor FP8 records."""

from dataclasses import asdict
import importlib

import pytest

from cudnn.sdpa.fwd import config_sm100 as legacy

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
