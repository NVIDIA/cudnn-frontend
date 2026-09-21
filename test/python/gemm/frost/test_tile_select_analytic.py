# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline (no-GPU) checks for the analytic tile/cluster selection.

There is ONE selection path for every graph type, so these cover the support constraints
that used to sit around the old N-bucket rule rather than a single branch of it.
Selection is pure geometry, so every invariant is checkable on CPU:
  * whatever it returns must be a real CATALOG entry;
  * it must never emit a geometry the kernel templates cannot run
    (2-CTA needs an even cluster M; block-scale needs 128-multiples);
  * multi-GEMM must stay inside the shared N budget, on cta_group=1;
  * N-major B must get a per-CTA N extent that is a whole number of swizzle groups;
  * ``K`` is optional, and omitting it must still resolve to a runnable config.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from cudnn.gemm.frost.tile_config import as_pipeline, by_name, select_config

MS = (1, 4, 16, 32, 64, 96, 128, 129, 256, 512, 1024, 4096)
NS = (32, 64, 128, 256, 512, 1024, 4096, 8192, 10240)
KS = (256, 1024, 2048, 4096, 8192)


@pytest.mark.L0
def test_joint_planner_can_remove_an_unprofitable_split():
    from cudnn.gemm.frost.fusion_ir import FusionChain, MatmulSpec, OutputSpec
    from cudnn.gemm.frost.planning import DeviceProperties, select_strategy

    chain = FusionChain(matmul=MatmulSpec(M=128, N=128, K=1024), output_specs=[OutputSpec(source_ref=-1, dtype="bf16")])
    baseline = replace(select_config(128, 128, 1, K=1024, sm_count=148), split_k_slices=8)
    device = DeviceProperties(arch=100, name="NVIDIA B200", sm_count=148, l2_bytes=132644864)
    config = select_strategy(chain, baseline, device=device, probe=lambda chain, config: None)
    assert config.split_k_slices == 1
    assert not config.swap_ab


def _plain_strategy_chain(M=128, N=128, K=4096, batch=1):
    from cudnn.gemm.frost.fusion_ir import FusionChain, MatmulSpec, OutputSpec

    return FusionChain(
        matmul=MatmulSpec(M=M, N=N, K=K, batch=batch, a_batch=batch, b_batch=batch),
        output_specs=[OutputSpec(source_ref=-1, dtype="bf16")],
    )


def _strategy_device(arch=100, name="NVIDIA B200", sm_count=148):
    from cudnn.gemm.frost.planning import DeviceProperties

    return DeviceProperties(arch=arch, name=name, sm_count=sm_count, l2_bytes=132644864)


@pytest.fixture(params=["b200", "rubin"])
def calibrated_device(request):
    if request.param == "rubin":
        return replace(_strategy_device(107, "NVIDIA Graphics Device", 208), l2_bytes=125829120)
    return _strategy_device()


@pytest.mark.L0
@pytest.mark.parametrize("dtype,block_size,sf_dtype", [("fp4_e2m1", 16, "fp8_e4m3"), ("fp4_e2m1", 32, "fp8_e8m0"), ("fp8_e4m3", 32, "fp8_e8m0")])
def test_block_scale_joint_profile_and_legal_geometry(calibrated_device, dtype, block_size, sf_dtype):
    from cudnn.gemm.frost import planning
    from cudnn.gemm.frost.tile_config import as_mma_tile_k

    chain = _block_scaled_chain(M=128, N=256, K=4096, quantized=False)
    chain = replace(
        chain,
        matmul=replace(chain.matmul, a_dtype=dtype, b_dtype=dtype),
        block_scale=replace(
            chain.block_scale,
            a_dtype=dtype,
            b_dtype=dtype,
            block_size_a=(1, block_size),
            block_size_b=(block_size, 1),
            sf_dtype_a=sf_dtype,
            sf_dtype_b=sf_dtype,
        ),
    )
    device = calibrated_device
    baseline = select_config(128, 256, 1, K=4096, block_scale=True, sm_count=device.sm_count)
    baseline = as_mma_tile_k(baseline, 64 if device.arch == 107 else 32)
    profile = planning.profile_for(chain, baseline, device)
    assert profile is not None
    configs = planning.candidate_configs(chain, baseline, device, profile)
    assert {cfg.swap_ab for cfg in configs} == {False, True}
    assert all(cfg.mma_tile_m % 128 == cfg.mma_tile_n % 128 == 0 for cfg in configs)
    assert all(cfg.mma_tile_k_bytes == baseline.mma_tile_k_bytes for cfg in configs)
    assert any(cfg.split_k_slices == 3 for cfg in configs)
    assert len(planning.select_strategies(chain, baseline, device=device, probe=lambda chain, config: None)) == 8


@pytest.mark.L0
def test_block_scale_cost_accounts_for_padded_scale_storage_and_staging():
    from cudnn.gemm.frost.planning import cost_features, split_candidates

    chain = _block_scaled_chain(M=160, N=192, K=1088, quantized=False)
    config = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
    nv = cost_features(chain, config, _strategy_device())
    mx = cost_features(replace(chain, block_scale=replace(chain.block_scale, block_size_a=(1, 32), block_size_b=(32, 1))), config, _strategy_device())
    assert nv.scale_input_mb == 2 * 256 * 68 / 1e6
    assert mx.scale_input_mb == 2 * 256 * 36 / 1e6
    assert nv.scale_stage_kb == 2 * mx.scale_stage_kb
    assert nv.input_mb == mx.input_mb == 1088 * (160 + 192) / 2e6
    assert list(split_candidates(chain, config)) == [1, 2, 3, 4, 5]


@pytest.mark.L0
@pytest.mark.parametrize(
    "a_dtype,b_dtype", [("fp4_e2m1", "fp4_e2m1"), ("fp8_e4m3", "fp8_e4m3"), ("fp8_e5m2", "fp8_e5m2"), ("fp8_e4m3", "fp8_e5m2"), ("fp8_e5m2", "fp8_e4m3")]
)
@pytest.mark.parametrize("out_dtype", ["bf16", "fp16"])
def test_block_scale_profiles_share_widths_and_preserve_unmeasured_paths(calibrated_device, a_dtype, b_dtype, out_dtype):
    from cudnn.gemm.frost import planning

    reference = _block_scaled_chain(M=128, N=256, K=4096, quantized=False)
    fp4 = a_dtype == "fp4_e2m1"
    bs = reference.block_scale
    if not fp4:
        bs = replace(bs, a_dtype="fp8_e4m3", b_dtype="fp8_e4m3", block_size_a=(1, 32), block_size_b=(32, 1), sf_dtype_a="fp8_e8m0", sf_dtype_b="fp8_e8m0")
        reference = replace(reference, matmul=replace(reference.matmul, a_dtype=bs.a_dtype, b_dtype=bs.b_dtype), block_scale=bs)
    chain = replace(
        reference,
        matmul=replace(reference.matmul, a_dtype=a_dtype, b_dtype=b_dtype, out_dtype=out_dtype),
        block_scale=replace(bs, a_dtype=a_dtype, b_dtype=b_dtype, sf_dtype_a="fp8_e8m0", sf_dtype_b="fp8_e8m0", block_size_a=(1, 32), block_size_b=(32, 1)),
        output_specs=[replace(reference.output_specs[0], dtype=out_dtype)],
    )
    baseline = select_config(128, 256, 1, K=4096, block_scale=True, sm_count=148)
    device = calibrated_device
    profile = planning.profile_for(reference, baseline, device)
    assert profile is not None
    assert planning.profile_for(chain, baseline, device) is profile
    if not fp4:
        assert planning.profile_for(replace(chain, block_scale=None), baseline, device) is not profile
    outside = [
        replace(chain, block_scale=replace(chain.block_scale, fake_dequant_a=True)),
        replace(chain, block_scale=replace(chain.block_scale, sfa_reorder=None)),
        replace(chain, matmul=replace(chain.matmul, accum_dtype="int32")),
        replace(
            chain,
            matmul=replace(chain.matmul, a_dtype="fp4_e2m1", b_dtype="fp8_e4m3"),
            block_scale=replace(chain.block_scale, a_dtype="fp4_e2m1", b_dtype="fp8_e4m3"),
        ),
    ]
    assert all(planning.select_strategies(view, baseline, device=device, probe=lambda chain, config: None) == [baseline] for view in outside)
    assert all(planning.profile_for(chain, baseline, _strategy_device(arch=arch)) is None for arch in (103, 110, 120))


@pytest.mark.L0
@pytest.mark.parametrize("input_dtype,output_dtype", [("bf16", "fp16"), ("fp16", "bf16"), ("fp16", "fp16")])
def test_same_bit_widths_share_strategy_parameters(calibrated_device, input_dtype, output_dtype):
    from cudnn.gemm.frost.planning import profile_for, select_strategies

    reference = _plain_strategy_chain(M=256, N=256, K=16384)
    chain = replace(
        reference,
        matmul=replace(reference.matmul, a_dtype=input_dtype, b_dtype=input_dtype, out_dtype=output_dtype),
        output_specs=[replace(reference.output_specs[0], dtype=output_dtype)],
    )
    config = select_config(256, 256, 1, K=16384, sm_count=148)
    device = calibrated_device
    assert profile_for(reference, config, device) is not None
    assert profile_for(chain, config, device) is profile_for(reference, config, device)
    assert select_strategies(chain, config, device=device, probe=lambda chain, config: None) == select_strategies(
        reference, config, device=device, probe=lambda chain, config: None
    )


@pytest.mark.L0
@pytest.mark.parametrize("a_dtype", ["fp8_e4m3", "fp8_e5m2"])
@pytest.mark.parametrize("b_dtype", ["fp8_e4m3", "fp8_e5m2"])
@pytest.mark.parametrize("output_dtype", ["bf16", "fp16"])
def test_fp8_formats_share_parameters_but_not_integer_accumulation(calibrated_device, a_dtype, b_dtype, output_dtype):
    from cudnn.gemm.frost.planning import profile_for, select_strategies

    plain = _plain_strategy_chain()
    reference = replace(plain, matmul=replace(plain.matmul, a_dtype="fp8_e4m3", b_dtype="fp8_e4m3"))
    chain = replace(
        reference,
        matmul=replace(reference.matmul, a_dtype=a_dtype, b_dtype=b_dtype, out_dtype=output_dtype),
        output_specs=[replace(reference.output_specs[0], dtype=output_dtype)],
    )
    config = select_config(128, 128, 1, K=4096, sm_count=148)
    device = calibrated_device
    profile = profile_for(reference, config, device)
    assert profile is not None
    assert profile_for(chain, config, device) is profile
    assert profile is not profile_for(plain, config, device)
    assert select_strategies(chain, config, device=device, probe=lambda chain, config: None) == select_strategies(
        reference, config, device=device, probe=lambda chain, config: None
    )
    integer = replace(chain, matmul=replace(chain.matmul, a_dtype="int8", b_dtype="int8", accum_dtype="int32"))
    assert profile_for(integer, config, device) is None


@pytest.mark.L0
@pytest.mark.parametrize(
    "arch,pipeline,name,calibrated",
    [
        (100, "sm100", "NVIDIA B200", True),
        (103, "sm100", "NVIDIA B200", False),
        (107, "sm100", "NVIDIA B200", False),
        (107, "sm100", "NVIDIA Graphics Device", True),
        (100, "sm100", "NVIDIA Graphics Device", False),
        (107, "sm103", "NVIDIA Graphics Device", False),
        (100, "sm103", "NVIDIA B200", False),
        (120, "sm120", "NVIDIA B200", False),
        (100, "sm100", "different SKU", False),
    ],
)
def test_strategy_profiles_do_not_alias_gpu_arch_and_pipeline(arch, pipeline, name, calibrated):
    from cudnn.gemm.frost.planning import profile_for

    config = (
        by_name("CONFIG_sm103_128x128x384_128x128x48_cluster1x1_1ctamma")
        if pipeline == "sm103"
        else as_pipeline(by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"), pipeline)
    )
    assert (profile_for(_plain_strategy_chain(), config, _strategy_device(arch, name)) is not None) == calibrated


@pytest.mark.L0
def test_strategy_device_facts_follow_build_device(monkeypatch):
    from cudnn.frost import device
    from cudnn.gemm.frost.planning import current_device_properties

    monkeypatch.setattr(device, "is_available", lambda: True)
    monkeypatch.setattr(device, "compute_capability", lambda ordinal: (10, 7) if ordinal == 1 else (10, 0))
    monkeypatch.setattr(device, "device_name", lambda ordinal: f"device-{ordinal}")
    monkeypatch.setattr(device, "multiprocessor_count", lambda ordinal: 208 if ordinal == 1 else 148)
    monkeypatch.setattr(device, "l2_cache_bytes", lambda ordinal: 1 << (26 + ordinal))
    with device.build_device(1):
        facts = current_device_properties()
    assert (facts.arch, facts.name, facts.sm_count, facts.l2_bytes) == (107, "device-1", 208, 1 << 27)
    monkeypatch.setattr(device, "is_available", lambda: False)
    assert current_device_properties() is None


@pytest.mark.L0
def test_joint_candidates_include_every_integer_and_both_orientations(calibrated_device):
    from cudnn.gemm.frost.planning import candidate_configs, profile_for

    chain = _plain_strategy_chain(M=256, N=256, K=16384)
    baseline = replace(select_config(256, 256, 1, K=16384, sm_count=148), split_k_slices=4)
    device = calibrated_device
    configs = candidate_configs(chain, baseline, device, profile_for(chain, baseline, device))
    assert baseline in configs
    assert len(configs) == len(set(configs)) <= 193
    for swap in (False, True):
        assert {c.split_k_slices for c in configs if c.swap_ab == swap} == set(range(1, 33))
    assert replace(baseline, split_k_slices=13) in configs


@pytest.mark.L0
@pytest.mark.parametrize("K,batch,maximum", [(64, 1, 1), (129, 1, 3), (1024, 1, 16), (16384, 1, 32), (16384, 24000, 2), (16384, 60000, 1)])
def test_split_candidates_respect_k_tiles_and_grid_z(K, batch, maximum):
    from cudnn.gemm.frost.planning import split_candidates

    chain = _plain_strategy_chain(K=K, batch=batch)
    config = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
    assert list(split_candidates(chain, config)) == list(range(1, maximum + 1))


@pytest.mark.L0
def test_dtype_width_changes_k_partition_and_input_traffic_only():
    from cudnn.gemm.frost.planning import cost_features, split_candidates

    f16 = _plain_strategy_chain(K=2048)
    f8 = replace(f16, matmul=replace(f16.matmul, a_dtype="fp8_e4m3", b_dtype="fp8_e5m2"))
    config = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma_splitK4")
    f16_cost = cost_features(f16, config, _strategy_device())
    f8_cost = cost_features(f8, config, _strategy_device())
    assert max(split_candidates(f16, config)) == 32
    assert max(split_candidates(f8, config)) == 16
    assert f16_cost.k_steps == 2 * f8_cost.k_steps
    assert f16_cost.input_mb == 2 * f8_cost.input_mb
    assert f16_cost.partial_cache_mb == f8_cost.partial_cache_mb
    assert f16_cost.reduce_steps == f8_cost.reduce_steps
    integer = replace(f8, matmul=replace(f8.matmul, a_dtype="int8", b_dtype="int8", accum_dtype="int32"))
    assert list(split_candidates(integer, config)) == [1]


@pytest.mark.L0
def test_candidate_geometry_preserves_pipeline_and_mma_k_width():
    from cudnn.gemm.frost.planning import candidate_configs, profile_for
    from cudnn.gemm.frost.tile_config import as_mma_tile_k

    chain = _plain_strategy_chain()
    baseline = as_mma_tile_k(select_config(128, 128, 1, K=4096, sm_count=148), 64)
    device = _strategy_device()
    configs = candidate_configs(chain, baseline, device, profile_for(chain, baseline, device))
    assert {cfg.pipeline for cfg in configs} == {baseline.pipeline}
    assert {cfg.mma_tile_k_bytes for cfg in configs} == {64}


@pytest.mark.L0
def test_strategy_cost_counts_cluster_padding_batch_and_integer_k():
    from cudnn.gemm.frost.planning import cost_features

    chain = _plain_strategy_chain(M=129, N=129, K=257, batch=3)
    config = by_name("CONFIG_sm100_64x32x128_64x32x32_cluster2x4_2ctamma_splitK3")
    features = cost_features(chain, config, _strategy_device(sm_count=32))
    assert features.waves == 9
    assert features.k_steps == 18
    assert features.partial_cache_mb == pytest.approx(2 * 4 * 3 * 3 * 129 * 129 / 1e6)
    assert features.partial_spill_mb == 0
    small_cache = cost_features(chain, config, replace(_strategy_device(sm_count=32), l2_bytes=0))
    assert small_cache.partial_spill_mb == features.partial_cache_mb
    assert small_cache.partial_cache_mb == 0


@pytest.mark.L0
def test_strategy_cost_uses_transformed_output_layout():
    from cudnn.gemm.frost.fusion_ir import swap_ab
    from cudnn.gemm.frost.planning import cost_features

    chain = _plain_strategy_chain(M=128, N=256)
    config = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
    original = cost_features(chain, config, _strategy_device())
    swapped = cost_features(swap_ab(chain), replace(config, swap_ab=True), _strategy_device())
    assert original.m_store == 0 and swapped.m_store > 0
    assert original.input_mb == swapped.input_mb


@pytest.mark.L0
@pytest.mark.parametrize("M,N", [(128, 4096), (4096, 128)])
def test_joint_selection_changes_tile_and_split_together(M, N):
    from cudnn.gemm.frost.planning import select_strategy

    chain = _plain_strategy_chain(M=M, N=N, K=8192)
    baseline = select_config(M, N, 1, K=8192, sm_count=148)
    selected = select_strategy(chain, baseline, device=_strategy_device(), probe=lambda chain, config: None)
    assert selected.cta_tile_m == selected.cta_tile_n == 128
    assert selected.cta_group == 1 and selected.split_k_slices == 4
    assert not selected.swap_ab


@pytest.mark.L0
def test_joint_selection_reuses_support_probe_on_original_chain():
    from cudnn.gemm.frost.planning import select_strategy

    chain = _plain_strategy_chain(M=128, N=4096, K=8192)
    baseline = select_config(128, 4096, 1, K=8192, sm_count=148)
    seen = []

    def probe(original, config):
        assert original is chain
        seen.append(config)
        if not config.swap_ab or config.split_k_slices == 4:
            raise NotImplementedError("candidate rejected by the engine")

    selected = select_strategy(chain, baseline, device=_strategy_device(), probe=probe)
    assert selected.swap_ab and selected.split_k_slices != 4
    assert any(c.split_k_slices == 4 for c in seen)


@pytest.mark.L0
def test_rubin_fp8_preserves_incumbent_for_small_predicted_improvement():
    from cudnn.gemm.frost.planning import select_strategies

    device = replace(_strategy_device(107, "NVIDIA Graphics Device", 208), l2_bytes=125829120)
    chain = _plain_strategy_chain(M=256, N=256, K=8192)
    chain = replace(chain, matmul=replace(chain.matmul, a_dtype="fp8_e4m3", b_dtype="fp8_e4m3"))
    baseline = by_name("CONFIG_sm100_64x32x128_64x32x32_cluster2x4_2ctamma_splitK6")
    configs = select_strategies(chain, baseline, device=device, probe=lambda chain, config: None)
    assert configs[0] == baseline
    assert len(configs) == 8
    assert any(cfg.split_k_slices == 1 for cfg in configs)


@pytest.mark.L0
def test_joint_selection_does_not_hide_probe_bugs():
    from cudnn.gemm.frost.planning import select_strategy

    def probe(chain, config):
        raise RuntimeError("failing support probe")

    with pytest.raises(RuntimeError, match="failing support probe"):
        select_strategy(_plain_strategy_chain(), select_config(128, 128, 1), device=_strategy_device(), probe=probe)


@pytest.mark.L0
@pytest.mark.parametrize("arch", [103, 110, 120])
def test_uncalibrated_architecture_preserves_existing_recommendation(arch):
    from cudnn.gemm.frost.planning import select_strategies, select_strategy

    baseline = replace(select_config(128, 128, 1, K=4096), split_k_slices=18)

    def probe(chain, config):
        pytest.fail("an uncalibrated profile must not start candidate search")

    assert select_strategy(_plain_strategy_chain(), baseline, device=_strategy_device(arch), probe=probe) is baseline
    assert select_strategies(_plain_strategy_chain(), baseline, device=_strategy_device(arch), probe=probe) == [baseline]


@pytest.mark.L0
def test_shortlist_covers_strategies_and_keeps_first_choice():
    from cudnn.gemm.frost.planning import profile_for, rank_candidates, select_strategies, select_strategy

    chain = _plain_strategy_chain(M=256, N=256, K=16384)
    baseline = replace(select_config(256, 256, 1, K=16384, sm_count=148), split_k_slices=4)
    device = _strategy_device()
    configs = select_strategies(chain, baseline, device=device, probe=lambda chain, config: None)
    assert len(configs) == len(set(configs)) == 8
    assert configs[0] == select_strategy(chain, baseline, device=device, probe=lambda chain, config: None)
    assert {cfg.swap_ab for cfg in configs} == {False, True}
    assert len({replace(cfg, split_k_slices=1, swap_ab=False) for cfg in configs}) >= 2
    assert any(cfg.split_k_slices & (cfg.split_k_slices - 1) for cfg in configs)
    ranked = rank_candidates(chain, configs[1:], device, profile_for(chain, baseline, device))
    assert configs[1:] == [cfg for _, cfg in ranked]


@pytest.mark.L0
def test_shortlist_returns_only_supported_configs_without_padding():
    from cudnn.gemm.frost.planning import select_strategies

    chain = _plain_strategy_chain()
    baseline = select_config(128, 128, 1, K=4096, sm_count=148)
    allowed = {replace(baseline, split_k_slices=slices) for slices in (7, 13)}
    seen = set()

    def probe(original, config):
        assert original is chain
        assert config not in seen
        seen.add(config)
        if config not in allowed:
            raise NotImplementedError("unsupported config")

    assert set(select_strategies(chain, baseline, device=_strategy_device(), probe=probe)) == allowed
    allowed.clear()
    seen.clear()
    with pytest.raises(NotImplementedError, match="no supported joint strategy"):
        select_strategies(chain, baseline, device=_strategy_device(), probe=probe)


@pytest.mark.L0
def test_strategy_profile_preserves_unmeasured_graph_paths(calibrated_device):
    from cudnn.gemm.frost.fusion_ir import FusionOp, OutputSpec
    from cudnn.gemm.frost.planning import profile_for

    chain = _plain_strategy_chain()
    baseline = select_config(128, 128, 1)
    outside = [
        replace(chain, ops=[FusionOp("relu", parent_idx=-1)]),
        replace(chain, matmul=replace(chain.matmul, a_dtype="fp32", b_dtype="fp32")),
        replace(chain, matmul=replace(chain.matmul, b_major="n")),
        replace(chain, matmul=replace(chain.matmul, out_dtype="fp32"), output_specs=[OutputSpec(source_ref=-1, dtype="fp32")]),
        replace(chain, output_specs=[OutputSpec(source_ref=-1, dtype="bf16", stride=(128 * 128, 1, 128))]),
        _plain_strategy_chain(K=256),
        _plain_strategy_chain(M=16),
        _plain_strategy_chain(N=8192),
        _plain_strategy_chain(K=32768),
        _plain_strategy_chain(batch=9),
        _block_scaled_chain(M=128),
        _block_scaled_chain(M=128, moe_groups=4),
    ]
    assert all(profile_for(view, baseline, calibrated_device) is None for view in outside)


@pytest.mark.L0
def test_shared_width_profile_does_not_override_dtype_support(monkeypatch):
    from cudnn.gemm.frost import kernel_registry, planning
    from cudnn.gemm.frost.sm100 import compiler

    monkeypatch.setattr(compiler, "_current_arch", lambda: 100)
    reference = _plain_strategy_chain()
    chain = replace(reference, matmul=replace(reference.matmul, a_dtype="fp16", b_dtype="bf16"))
    config = select_config(128, 128, 1, K=4096, sm_count=148)
    assert planning.profile_for(chain, config, _strategy_device()) is not None

    def probe(original, candidate):
        reason = kernel_registry.mma_arch_reject(original, kernel_registry.GraphType.MATMUL, candidate.pipeline)
        if reason:
            raise NotImplementedError(reason)

    with pytest.raises(NotImplementedError, match="no supported joint strategy"):
        planning.select_strategies(chain, config, device=_strategy_device(), probe=probe)


@pytest.mark.L0
@pytest.mark.parametrize("family", ["sm100", "sm120"])
def test_both_compilers_delegate_joint_selection_but_replay_and_dynamic_bypass_it(monkeypatch, family):
    import importlib

    from cudnn.gemm.frost import planning
    from cudnn.gemm.frost.knobs import GemmKnobs

    compiler = importlib.import_module(f"cudnn.gemm.frost.{family}.compiler")
    calls = []

    def select(chain, baseline, *, probe):
        assert probe == compiler.probe_chain
        calls.append(baseline)
        return replace(baseline, split_k_slices=3)

    monkeypatch.setattr(planning, "select_strategy", select)
    chain = _plain_strategy_chain()
    assert compiler.plan_config(chain).split_k_slices == 3
    assert len(calls) == 1
    assert compiler.plan_config(chain, dynamic_shapes=True).split_k_slices == 1
    config = replace(calls[0], split_k_slices=7, swap_ab=True)
    assert compiler.plan_config(None, knobs=GemmKnobs.from_config(config)) == config
    assert len(calls) == 1
    monkeypatch.setattr(planning, "select_strategies", lambda chain, baseline, *, probe: [select(chain, baseline, probe=probe), baseline])
    assert compiler.plan_configs(chain)[0].split_k_slices == 3
    assert len(calls) == 2
    dynamic = compiler.plan_configs(chain, dynamic_shapes=True)
    assert len(dynamic) == 1 and dynamic[0].split_k_slices == 1
    assert len(calls) == 2


@pytest.mark.L0
@pytest.mark.parametrize("family", ["sm100", "sm120"])
@pytest.mark.parametrize("block_scale", [False, True])
def test_moe_planning_skips_split_k_selection(monkeypatch, family, block_scale):
    import importlib

    from cudnn.gemm.frost import kernel_registry, planning
    from cudnn.gemm.frost.fusion_ir import MoeSpec

    compiler = importlib.import_module(f"cudnn.gemm.frost.{family}.compiler")
    chain = (
        _block_scaled_chain(M=128, N=128, K=4096, quantized=False, moe_groups=4)
        if block_scale
        else replace(_plain_strategy_chain(), moe=MoeSpec(num_experts=4, num_groups=4))
    )

    def unexpected_split_k(*args, **kwargs):
        pytest.fail("MoE planning must not invoke split-K selection")

    monkeypatch.setattr(kernel_registry, "preferred_strategy", lambda chain, config: as_pipeline(config, family))
    monkeypatch.setattr(compiler, "_auto_split_k", unexpected_split_k)
    monkeypatch.setattr(planning, "split_candidates", unexpected_split_k)
    config = compiler.plan_config(chain)
    assert config.split_k_slices == 1
    assert compiler.plan_configs(chain) == [config]
    assert "MoE grouped matmul" in compiler._splitk_reject_reason(chain, replace(config, split_k_slices=2))


@pytest.mark.L0
@pytest.mark.parametrize(
    "batch,M,N,K", [(1, 128, 128, 1024), (1, 128, 4096, 8192), (1, 256, 256, 16384), (3, 96, 160, 4096), (1, 192, 160, 2112), (1, 129, 96, 8192)]
)
def test_joint_planner_runs_real_support_gates(monkeypatch, batch, M, N, K):
    from cudnn.gemm.frost import planning, tile_config
    from cudnn.gemm.frost.sm100 import compiler

    monkeypatch.setattr(planning, "current_device_properties", _strategy_device)
    monkeypatch.setattr(tile_config, "_sm_count", lambda: 148)
    monkeypatch.setattr(compiler, "_sm_count", lambda: 148)
    monkeypatch.setattr(compiler, "_current_arch", lambda: 100)
    chain = _plain_strategy_chain(M=M, N=N, K=K, batch=batch)
    config = compiler.plan_config(chain)
    configs = compiler.plan_configs(chain)
    assert len(configs) == len(set(configs)) == 8
    assert configs[0] == config
    for config in configs:
        compiler.probe_chain(chain, config)
        assert by_name(config.name) == config


def test_swap_ab_is_a_named_tile_config_option():
    base = by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma")
    swapped = replace(base, swap_ab=True)
    assert swapped.name == f"{base.name}_swapAB"
    assert swapped.geometry_name.endswith("_swapAB")
    assert by_name(swapped.name) == swapped
    assert as_pipeline(swapped, "sm120").swap_ab is True


def test_swap_ab_transforms_the_fusion_ir_without_moving_storage():
    from cudnn.gemm.frost.fusion_ir import FusionChain, FusionOp, MatmulSpec, OutputSpec, TensorRef, swap_ab

    chain = FusionChain(
        matmul=MatmulSpec(M=64, N=192, K=128, batch=3, a_batch=1, b_batch=3, a_major="m", b_major="n", a_dtype="fp16", b_dtype="bf16"),
        aux_tensors=[TensorRef("bias", (1, 1, 192), (192, 192, 1), "fp32", "per_col")],
        ops=[FusionOp("gen_index", parent_idx=-1, attrs=(("axis", 2.0),))],
        output_specs=[OutputSpec(source_ref=0, dtype="bf16")],
        mainloop_a_ops=[FusionOp("relu", parent_idx=-1)],
    )

    swapped = swap_ab(chain)
    assert (swapped.matmul.M, swapped.matmul.N, swapped.matmul.a_batch, swapped.matmul.b_batch) == (192, 64, 3, 1)
    assert (swapped.matmul.a_major, swapped.matmul.b_major) == ("m", "n")
    assert (swapped.matmul.a_dtype, swapped.matmul.b_dtype) == ("bf16", "fp16")
    assert swapped.aux_tensors[0].bcast_mode == "per_row"
    assert swapped.ops[0].attrs == (("axis", 1.0),)
    assert swapped.output_specs[0].dim == (3, 192, 64)
    assert swapped.output_specs[0].stride == (64 * 192, 1, 192)
    assert swapped.out_major == "m"
    assert not swapped.mainloop_a_ops and swapped.mainloop_b_ops == chain.mainloop_a_ops


@pytest.mark.parametrize("block_scale", [False, True])
def test_analytic_selection_is_always_runnable(block_scale):
    for M in MS:
        for N in NS:
            for K in KS:
                cfg = select_config(M, N, 1, K=K, block_scale=block_scale)
                assert by_name(cfg.name) is cfg, f"{cfg.name} not in CATALOG"
                if cfg.cta_group == 2:
                    assert cfg.cga_size_m % 2 == 0, f"2-CTA needs even cluster M: {cfg.name}"
                if block_scale:
                    assert cfg.cta_tile_m % 128 == 0 and cfg.cta_tile_n % 128 == 0


def test_omitting_k_is_accepted():
    """K is optional for callers that do not have it to hand: the small-K bias goes
    neutral and everything else still resolves to a real CATALOG entry."""
    for M in MS:
        for N in NS:
            for num_gemms in (1, 2, 4):
                cfg = select_config(M, N, num_gemms)
                assert by_name(cfg.name) is cfg
                if cfg.cta_group == 2:
                    assert cfg.cga_size_m % 2 == 0


def test_multi_gemm_budget_and_constraints():
    """One selection path, but the multi-GEMM support constraints still hold: the N
    tile is capped by the shared 256-wide budget, and only the 1ctamma template
    implements multi-GEMM."""
    for ng in (2, 4, 8):
        cap = max(32, min(256, 256 // ng))
        for M in (64, 512, 4096):
            cfg = select_config(M, 8192, ng, K=4096)
            assert cfg.cta_tile_n <= cap
            assert cfg.cta_group == 1, "multi-GEMM is 1ctamma-only"


@pytest.mark.L0
@pytest.mark.parametrize("M", [384, 768, 1408])
def test_terminal_quant_strategy_can_select_one_cta(M):
    """The strategy override selects a runnable 1-CTA config without pinning a
    private tile or cluster identity."""
    config = select_config(M, 5120, 1, K=2048, block_scale=True, sm_count=148, force_cta_group=1)
    assert config.cta_group == 1
    assert by_name(config.name) is config


@pytest.mark.L0
def test_force_cta_group_rejects_invalid_or_unsupported_values():
    """Invalid overrides and a 2-CTA multi-GEMM request fail explicitly."""
    for value in (True, 0, 3):
        with pytest.raises(ValueError, match="force_cta_group"):
            select_config(384, 5120, 1, force_cta_group=value)
    with pytest.raises(NotImplementedError, match="multi-GEMM is 1ctamma-only"):
        select_config(384, 5120, 2, force_cta_group=2)


def _block_scaled_chain(*, M, N=5120, K=2048, batch=1, quantized=True, moe_groups=None):
    """Build the semantic IR consumed by the planner, without GPU storage."""
    from cudnn.gemm.frost.fusion_ir import (
        BlockQuantizeSpec,
        BlockScaleSpec,
        FusionChain,
        MatmulSpec,
        MoeSpec,
        OutputSpec,
        gemm_source,
    )

    if moe_groups is not None and batch != 1:
        raise ValueError("MoE analyzer IR always has batch=1")
    matmul = MatmulSpec(
        M=M,
        N=N,
        K=K,
        batch=batch,
        a_batch=1,
        b_batch=batch,
        a_dtype="fp4_e2m1",
        b_dtype="fp4_e2m1",
    )
    block_scale = BlockScaleSpec(
        a_dtype="fp4_e2m1",
        b_dtype="fp4_e2m1",
        block_size_a=(1, 16),
        block_size_b=(16, 1),
        sf_dtype_a="fp8_e4m3",
        sf_dtype_b="fp8_e4m3",
        sfa_reorder="F8_128x4",
        sfb_reorder="F8_128x4",
        dequant_compute_a="fp32",
        dequant_compute_b="fp32",
        dequant_out_a="fp32",
        dequant_out_b="fp32",
    )
    source = gemm_source(0)
    quants = [BlockQuantizeSpec(source_ref=source, block_size=16, scale_dtype="fp8_e4m3")] if quantized else []
    output = OutputSpec(source_ref=source, dtype="fp4_e2m1", quant_idx=0) if quantized else OutputSpec(source_ref=source, dtype="bf16")
    moe = None if moe_groups is None else MoeSpec(num_experts=moe_groups, num_groups=moe_groups)
    return FusionChain(matmul=matmul, block_scale=block_scale, moe=moe, quants=quants, output_specs=[output])


@pytest.mark.L0
@pytest.mark.parametrize("M,expected_cta_group", [(384, 1), (1408, 1), (1409, 2), (4096, 2)])
def test_planner_bounds_terminal_quant_one_cta_to_measured_total_rows(monkeypatch, M, expected_cta_group):
    """The terminal-quant override stops at the measured total-row boundary."""
    from cudnn.gemm.frost import compiler, kernel_registry

    monkeypatch.setattr(kernel_registry, "preferred_strategy", lambda _chain, config: config)
    config = compiler.plan_config(_block_scaled_chain(M=M))
    assert config.cta_group == expected_cta_group


@pytest.mark.L0
def test_planner_uses_total_m_not_unobservable_moe_row_distribution(monkeypatch):
    """A low average cannot extend the override past total M; naturally M-starved
    grouped problems may still select 1-CTA through the ordinary heuristic."""
    from cudnn.gemm.frost import compiler, kernel_registry

    monkeypatch.setattr(kernel_registry, "preferred_strategy", lambda _chain, config: config)
    assert compiler.plan_config(_block_scaled_chain(M=2816, moe_groups=2)).cta_group == 2
    assert compiler.plan_config(_block_scaled_chain(M=2816, moe_groups=23)).cta_group == 1


@pytest.mark.L0
def test_planner_counts_every_dense_batch_row(monkeypatch):
    """A small per-batch M does not bypass the total declared-row boundary."""
    from cudnn.gemm.frost import compiler, kernel_registry

    monkeypatch.setattr(kernel_registry, "preferred_strategy", lambda _chain, config: config)
    assert compiler.plan_config(_block_scaled_chain(M=384, batch=3)).cta_group == 1
    assert compiler.plan_config(_block_scaled_chain(M=384, batch=4)).cta_group == 2


@pytest.mark.L0
def test_non_quantizing_graph_keeps_default_cta_policy(monkeypatch):
    """A block-scaled graph without a materialized quantizer is unaffected."""
    from cudnn.gemm.frost import compiler, kernel_registry

    monkeypatch.setattr(kernel_registry, "preferred_strategy", lambda _chain, config: config)
    config = compiler.plan_config(_block_scaled_chain(M=384, N=2048, K=5120, quantized=False))
    assert config.cta_group == 2


def test_n_major_b_lifts_the_n_tile():
    """N-major B is TMA-loaded a swizzle group at a time, so the per-CTA N extent must
    be a whole number of groups -- under the analytic tile choice too."""
    for eb in (1, 2):
        group_elems = 128 // eb
        for M in (64, 512):
            cfg = select_config(M, 8192, 1, K=4096, b_n_major=True, b_elem_bytes=eb)
            assert cfg.cta_tile_n % (group_elems * cfg.cta_group) == 0


def test_a_new_pipeline_must_register_its_hardware_facts():
    """A family that registers a config class but forgets a per-pipeline table
    must raise, not inherit another family's value: the tables are hardware
    facts, and a wrong MMA-inst K renders a descriptor that is silently wrong."""
    import dataclasses

    from cudnn.gemm.frost import tile_config as tc

    @dataclasses.dataclass(frozen=True)
    class ConfigSmFake(tc.TileConfig):
        pass

    tc._CONFIG_CLASS_BY_PIPELINE["sm_fake"] = ConfigSmFake
    try:
        with pytest.raises(NotImplementedError, match="MMA-inst K width not known for pipeline"):
            tc.as_pipeline(tc.DEFAULT_CONFIG, "sm_fake")
    finally:
        del tc._CONFIG_CLASS_BY_PIPELINE["sm_fake"]
