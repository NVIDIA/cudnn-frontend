# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host contracts for the per-tensor FP8 D256 implementation owner."""

import pytest

from cudnn.sdpa.fwd import config_sm100 as legacy
from cudnn.sdpa.fwd.config_common import TemplateParams
from cudnn.sdpa.fwd.kernels.sm100.d256_fp8_single_q import config
from cudnn.sdpa.fwd.kernels.sm100.d256_f16_lookahead import config as half_config

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("dtype", [0, 1])
def test_per_tensor_configuration_is_independent_of_half_and_mxfp8(monkeypatch, dtype):
    def forbidden(*args, **kwargs):
        raise AssertionError("per-tensor configuration consulted a sibling factory or validator")

    monkeypatch.setattr(legacy, "_make_cfg_d256", forbidden)
    monkeypatch.setattr(legacy, "_validate_cfg_d256", forbidden)
    monkeypatch.setattr(half_config, "make_cfg_d256", forbidden)
    params = TemplateParams(dtype_qkv=dtype, dtype_o=2, cta_mma=1)
    cfg, _ = legacy.make_cfg_d256(params)
    assert type(cfg) is config.CfgD256
    assert config.CfgD256 is not legacy.CfgD256
    assert config.CfgD256 is not half_config.CfgD256
    assert (cfg.CGA_M, cfg.CTA_MMA, cfg.TILE_M, cfg.STAGES_KV) == (1, 1, 128, 2)


def test_per_tensor_factory_is_not_on_the_half_or_block_scaled_path(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("sibling configuration consulted the per-tensor factory")

    monkeypatch.setattr(config, "make_cfg_d256", forbidden)
    half, _ = legacy.make_cfg_d256(TemplateParams(dtype_qkv=2, cta_mma=2))
    block, _ = legacy.make_cfg_d256_mxfp8(TemplateParams(dtype_qkv=0, dtype_o=2, cta_mma=1))
    assert type(half) is half_config.CfgD256
    assert type(block) is legacy.CfgD256


@pytest.mark.parametrize("dtype", [2, 3])
def test_per_tensor_owner_rejects_half_inputs(dtype):
    with pytest.raises(ValueError, match="requires E4M3/E5M2"):
        config.make_cfg_d256(TemplateParams(dtype_qkv=dtype, cta_mma=2))
