# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host contracts for isolated half D192 multi-Q and shared single-Q bodies."""

from dataclasses import asdict, replace
import importlib

import pytest

from cudnn.sdpa.fwd import config_sm100 as legacy

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("owner,record", [("d192_f16_multi_q", "CfgD192"), ("d128_f16_single_q", "CfgD128Decode"), ("d128_f16_single_q", "CfgD64Decode")])
def test_configuration_defaults_are_independent(owner, record, monkeypatch):
    module = importlib.import_module(f"cudnn.sdpa.fwd.kernels.sm100.{owner}.config")
    own = getattr(module, record)
    expected = asdict(getattr(legacy, record)())
    assert own.__bases__ == (object,)
    assert asdict(own()) == expected
    monkeypatch.setattr(getattr(legacy, record), "STAGES_KV", 999)
    assert own.STAGES_KV == expected["STAGES_KV"]


@pytest.mark.parametrize("dtype", [0, 1])
def test_multi_q_cannot_become_a_quantized_body(dtype):
    from cudnn.sdpa.fwd.kernels.sm100.d192_f16_multi_q.config import make_cfg_d192

    with pytest.raises(ValueError, match="FP16/BF16"):
        make_cfg_d192(legacy.TemplateParams(dtype_qkv=dtype))


@pytest.mark.parametrize("head_dim", [64, 128, 192])
def test_single_q_cannot_select_quadrant_cooperation(head_dim):
    from cudnn.sdpa.fwd.kernels.sm100.d128_f16_single_q import config

    params = legacy.TemplateParams(
        cta_mma=1,
        d_flavor=64 if head_dim == 64 else 128,
        decode_tile=head_dim == 64,
        single_q_head_dim=192 if head_dim == 192 else 128,
        thd_varlen=head_dim == 192,
        seq_kv_lens_present=head_dim == 192,
    )
    factory = config.make_cfg_d64_decode if head_dim == 64 else config.make_cfg_d128_decode
    cfg, _ = factory(params)
    assert cfg.TILES_Q == 1 and cfg.TILE_K == head_dim
    with pytest.raises(ValueError, match="quadrant"):
        factory(replace(params, decode_quad=True))


@pytest.mark.parametrize(
    "owner,function,params",
    [
        ("d192_f16_multi_q", "make_cfg_d192", legacy.TemplateParams()),
        ("d128_f16_single_q", "make_cfg_d128_decode", legacy.TemplateParams(cta_mma=1, single_q_head_dim=192, thd_varlen=True, seq_kv_lens_present=True)),
        ("d128_f16_single_q", "make_cfg_d64_decode", legacy.TemplateParams(cta_mma=1, d_flavor=64, decode_tile=True)),
    ],
)
def test_factories_do_not_call_legacy_implementations(owner, function, params, monkeypatch):
    module = importlib.import_module(f"cudnn.sdpa.fwd.kernels.sm100.{owner}.config")

    def forbidden(*args, **kwargs):
        raise AssertionError("isolated body consulted a legacy configuration factory")

    for name in ("make_cfg_d192", "make_cfg_d128_decode", "make_cfg_d64_decode", "_validate_cfg_d192", "_validate_cfg_d128_decode", "_validate_cfg_d64_decode"):
        monkeypatch.setattr(legacy, name, forbidden)
    cfg, _ = getattr(module, function)(params)
    assert cfg.DTYPE_QKV == params.dtype_qkv


@pytest.mark.parametrize("owner,cga,query_tiles", [("d192_f16_multi_q", 1, 2), ("d192_f16_multi_q", 2, 2), ("d128_f16_single_q", 1, 1)])
def test_fixed_host_owns_its_geometry(owner, cga, query_tiles):
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    module = importlib.import_module(f"cudnn.sdpa.fwd.kernels.sm100.{owner}.host")
    config = importlib.import_module(f"cudnn.sdpa.fwd.kernels.sm100.{owner}.config")
    spec, facts = object(), object()
    seen = []

    def lower_impl(actual_spec, actual_facts, knobs, *, selected_kernel):
        assert actual_spec is spec and actual_facts is facts
        params = legacy.TemplateParams(dtype_qkv=2, cta_mma=cga, thd_varlen=query_tiles == 1, seq_kv_lens_present=query_tiles == 1)
        params = selected_kernel.configure_params(params)
        factory = config.make_cfg_d192 if query_tiles == 2 else config.make_cfg_d128_decode
        cfg, _ = factory(params)
        assert cfg.TILES_Q == query_tiles and cfg.TILE_K == 192
        assert cfg.CGA_M == selected_kernel.cluster_m == knobs.cga == cga
        assert not params.decode_quad and not params.early_q
        seen.append(selected_kernel)
        return selected_kernel

    result = module.lower(spec, facts, SdpaFwdKnobs(cga=cga), lower_impl=lower_impl, knob_type=SdpaFwdKnobs)
    assert seen == [result]


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("cga", [1, 2])
def test_thd_body_boundaries_are_explicit(paged, cga):
    import cudnn
    from cudnn.sdpa.graph_analyzer import SdpaGraphFacts
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs
    from cudnn.sdpa.fwd.kernels.sm100.d192_f16_multi_q import contract as multi
    from cudnn.sdpa.fwd.kernels.sm100.d128_f16_single_q import contract as single

    facts = SdpaGraphFacts(
        b=2,
        h_q=8,
        h_kv=8,
        s_q=257,
        s_kv=1024,
        d_qk=192,
        d_v=128,
        dtype=cudnn.data_type.BFLOAT16,
        dtype_o=cudnn.data_type.BFLOAT16,
        device_cc=(10, 0),
        thd=True,
        has_paged_kv=paged,
    )
    knobs = SdpaFwdKnobs(cga=cga, pack_gqa=False)
    assert (single.mismatch(facts, knobs) is None) == (not paged and cga == 1)
    assert (multi.mismatch(facts, knobs) is None) == (paged or cga == 2)
