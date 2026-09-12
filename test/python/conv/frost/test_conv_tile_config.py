# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tile-catalog and selected-geometry coverage for Frost convolution."""

from __future__ import annotations

import itertools

import pytest

from cudnn.conv.frost.tile_config import CATALOG, DEFAULT_CONFIG, ConvTileConfig, _get_config, by_name
from utils import make_input, requires_block_scale_cutedsl, requires_dense_cutedsl, requires_sm100


@pytest.mark.L0
def test_conv_tile_catalog_is_complete_unique_and_canonical() -> None:
    expected = {
        (cta_m, cta_n, cta_k_bytes, cta_group)
        for cta_m, cta_n, cta_k_bytes, cta_group in itertools.product(
            (64, 128),
            range(32, 257, 32),
            (32, 64, 96, 128),
            (1, 2),
        )
    }
    actual = {(config.cta_tile_m, config.cta_tile_n, config.cta_tile_k_bytes, config.cta_group) for config in CATALOG}

    assert actual == expected
    assert len({config.name for config in CATALOG}) == len(CATALOG)
    assert all(by_name(config.name) is config for config in CATALOG)


@pytest.mark.L0
def test_default_config_remains_available_for_explicit_replay() -> None:
    assert DEFAULT_CONFIG.name == "CONFIG_sm100_128x128x128_128x128x32_cluster2x1_2ctamma"
    assert DEFAULT_CONFIG.mma_tiler_mn == (256, 128)
    assert DEFAULT_CONFIG.cluster_shape_mn == (2, 1)
    assert DEFAULT_CONFIG.mma_inst_tile_k == 4
    assert DEFAULT_CONFIG.cta_tile_k(16) == 64


@pytest.mark.L0
@pytest.mark.parametrize("cta_k_bytes, fp4_elements", ((32, 64), (64, 128), (96, 192), (128, 256)))
def test_conv_tile_config_accepts_dense_and_block_scale_k_widths(cta_k_bytes, fp4_elements) -> None:
    config = ConvTileConfig(128, 128, cta_k_bytes, 1)

    assert config.cta_tile_k_bytes == cta_k_bytes
    assert config.cta_tile_k(4) == fp4_elements
    assert config in CATALOG


@pytest.mark.L0
def test_private_selector_is_not_exported_from_frost_package() -> None:
    import cudnn.conv.frost as frost

    assert "get_config" not in frost.__all__
    assert "_get_config" not in frost.__all__
    assert not hasattr(frost, "get_config")
    assert not hasattr(frost, "_get_config")


def _get_dense_config(M: int, N: int, K: int, channel_bytes: int, sm_count: int = 148) -> ConvTileConfig:
    from cudnn.conv.frost.templates.sm100_conv import _is_dense_selection_candidate

    return _get_config(
        M,
        N,
        K,
        predicate=lambda config: _is_dense_selection_candidate(config, M, N, channel_bytes),
        sm_count=sm_count,
    )


@pytest.mark.L0
@requires_dense_cutedsl
def test_dense_predicate_selects_cta_group_from_implicit_m() -> None:
    one_cta = _get_dense_config(128, 128, 1024, channel_bytes=128)
    two_cta = _get_dense_config(129, 128, 1024, channel_bytes=128)

    assert one_cta.cta_group == 1
    assert two_cta.cta_group == 2
    assert one_cta.cta_tile_m == two_cta.cta_tile_m == 128


@pytest.mark.L0
@requires_dense_cutedsl
def test_dense_predicate_scores_n_and_selects_widest_legal_k_tile() -> None:
    full_k = _get_dense_config(4096, 512, 4096, channel_bytes=128)
    narrow_k = _get_dense_config(4096, 512, 4096, channel_bytes=64)

    assert full_k.cta_tile_n == narrow_k.cta_tile_n == 128
    assert full_k.cta_tile_k_bytes == 128
    assert narrow_k.cta_tile_k_bytes == 64
    assert full_k in CATALOG and narrow_k in CATALOG


@pytest.mark.L0
@pytest.mark.parametrize(
    "args, predicate, sm_count, error",
    [
        ((0, 128, 1024), lambda config: True, 148, "must be positive"),
        ((128, 128, 1024), lambda config: False, 148, "no compatible"),
        ((128, 128, 1024), lambda config: True, 0, "sm_count"),
    ],
)
def test_private_get_config_rejects_invalid_inputs(args, predicate, sm_count, error) -> None:
    with pytest.raises(ValueError, match=error):
        _get_config(*args, predicate=predicate, sm_count=sm_count)


@pytest.mark.L0
def test_shared_tile_ranker_rejects_an_empty_filtered_candidate_set() -> None:
    from cudnn.conv.frost.tile_config import _select_best_config

    with pytest.raises(ValueError, match="no compatible convolution tile configurations"):
        _select_best_config(128, 128, 128, (), sm_count=148)


@pytest.mark.parametrize("cta_group, expected_cluster, expected_mma_m", [(1, (1, 1), 128), (2, (2, 1), 256)])
@pytest.mark.L0
def test_cta_group_derives_mma_and_minimal_cluster(cta_group, expected_cluster, expected_mma_m) -> None:
    config = ConvTileConfig(128, 128, 64, cta_group)

    assert config.cluster_shape_mn == expected_cluster
    assert config.mma_tiler_mn == (expected_mma_m, 128)
    assert config.use_2cta_instrs is (cta_group == 2)
    assert config.mma_inst_tile_k == 2
    assert config.cta_tile_k(8) == 64
    assert config.cta_tile_k(16) == 32
    assert config.cta_tile_k(32) == 16


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"cta_tile_m": 256, "cta_tile_n": 128, "cta_tile_k_bytes": 128, "cta_group": 2}, "cta_tile_m"),
        ({"cta_tile_m": 128, "cta_tile_n": 16, "cta_tile_k_bytes": 128, "cta_group": 2}, "cta_tile_n"),
        ({"cta_tile_m": 128, "cta_tile_n": 128, "cta_tile_k_bytes": 48, "cta_group": 2}, "cta_tile_k_bytes"),
        ({"cta_tile_m": 128, "cta_tile_n": 128, "cta_tile_k_bytes": 128, "cta_group": 4}, "cta_group"),
    ],
)
@pytest.mark.L0
def test_invalid_conv_tile_axes_are_rejected(kwargs, error) -> None:
    with pytest.raises(ValueError, match=error):
        ConvTileConfig(**kwargs)


@pytest.mark.L0
def test_noncanonical_config_name_is_rejected() -> None:
    with pytest.raises(KeyError, match="not canonical"):
        by_name("CONFIG_sm100_128x128x128_64x128x32_cluster2x1_2ctamma")


@pytest.mark.L0
@pytest.mark.parametrize("cta_k_bytes", (32, 96))
def test_block_scale_k_width_names_are_complete_catalog_entries(cta_k_bytes) -> None:
    name = f"CONFIG_sm100_128x128x{cta_k_bytes}_128x128x32_cluster1x1_1ctamma"

    assert by_name(name) == ConvTileConfig(128, 128, cta_k_bytes, 1)


@pytest.mark.L0
@pytest.mark.parametrize("cta_k_bytes", (32, 96))
@requires_dense_cutedsl
def test_dense_compile_rejects_block_scale_only_k_widths_before_cute_compile(monkeypatch, cta_k_bytes) -> None:
    from cudnn.conv.frost.templates import sm100_conv

    def unexpected_compile(*args, **kwargs):
        pytest.fail("cute.compile must not be called for an invalid dense tile config")

    monkeypatch.setattr(sm100_conv.cute, "compile", unexpected_compile)
    with pytest.raises(ValueError, match=rf"{cta_k_bytes}.*CTA K must be 64 or 128|CTA K must be 64 or 128.*{cta_k_bytes}"):
        sm100_conv.compile(
            ncdhw=(1, 128, 1, 1, 1),
            ktrs=(128, 1, 1, 1),
            tile_config=ConvTileConfig(128, 128, cta_k_bytes, 1),
        )


@pytest.mark.L0
@requires_dense_cutedsl
def test_dense_compile_rejects_catalog_geometry_incompatible_with_dense_epilogue(monkeypatch) -> None:
    from cudnn.conv.frost.templates import sm100_conv

    def unexpected_compile(*args, **kwargs):
        pytest.fail("cute.compile must not be called for an invalid dense tile config")

    monkeypatch.setattr(sm100_conv.cute, "compile", unexpected_compile)
    config = ConvTileConfig(64, 32, 64, 2)
    with pytest.raises(ValueError, match=rf"{config.name}.*M=64 2-CTA requires CTA N divisible by 64"):
        sm100_conv.compile(
            ncdhw=(1, 32, 1, 1, 1),
            ktrs=(128, 1, 1, 1),
            tile_config=config,
        )


def _get_block_scale_config(M: int, N: int, K: int, C: int, sm_count: int = 148) -> ConvTileConfig:
    from cudnn.conv.frost.templates.sm100_block_scale_conv import _is_block_scale_selection_candidate

    return _get_config(
        M,
        N,
        K,
        predicate=lambda config: _is_block_scale_selection_candidate(config, M, N, C),
        sm_count=sm_count,
    )


@pytest.mark.L0
@pytest.mark.parametrize(
    "implicit_m, output_n, channels, expected_n, expected_k_bytes, expected_group",
    (
        (128, 64, 64, 64, 32, 1),
        (128, 64, 128, 64, 64, 1),
        (128, 64, 192, 64, 96, 1),
        (128, 64, 256, 64, 128, 1),
        (128, 64, 512, 64, 128, 1),
        (128, 512, 256, 128, 128, 1),
        (4096, 192, 64, 64, 32, 2),
        (4096, 192, 128, 64, 64, 2),
        (4096, 192, 192, 256, 96, 2),
        (4096, 192, 256, 64, 128, 2),
        (4096, 192, 512, 64, 128, 2),
        (4096, 512, 128, 128, 64, 2),
        (4096, 512, 192, 256, 96, 2),
    ),
)
@requires_block_scale_cutedsl
def test_block_scale_config_selection(
    implicit_m,
    output_n,
    channels,
    expected_n,
    expected_k_bytes,
    expected_group,
) -> None:
    config = _get_block_scale_config(implicit_m, output_n, channels, channels)

    assert (config.cta_tile_m, config.cta_tile_n, config.cta_tile_k_bytes, config.cta_group) == (
        128,
        expected_n,
        expected_k_bytes,
        expected_group,
    )


@pytest.mark.L0
@requires_block_scale_cutedsl
def test_block_scale_2cta_96_byte_k_filters_small_n_tiles() -> None:
    config = _get_block_scale_config(129, 64, 192, 192)

    assert config == ConvTileConfig(128, 256, 96, 2)


@pytest.mark.L0
@pytest.mark.parametrize(
    "config, channels, error",
    (
        (ConvTileConfig(64, 128, 64, 1), 128, "CTA M must be 128"),
        (ConvTileConfig(128, 128, 32, 1), 128, "64-byte K tile requires C=128"),
        (ConvTileConfig(128, 128, 128, 1), 192, "96-byte K tile requires C=192"),
        (ConvTileConfig(128, 128, 64, 1), 512, "128-byte K tile requires C divisible by 256"),
        (ConvTileConfig(128, 32, 64, 1), 128, "CTA N>=64"),
        (ConvTileConfig(128, 128, 96, 2), 192, "2-CTA.*CTA N>=192"),
    ),
)
@requires_block_scale_cutedsl
def test_block_scale_explicit_config_validation(config, channels, error) -> None:
    from cudnn.conv.frost.templates.sm100_block_scale_conv import _validate_block_scale_config

    with pytest.raises(ValueError, match=rf"{config.name}.*{error}"):
        _validate_block_scale_config(config, channels)


@pytest.mark.L0
@requires_block_scale_cutedsl
def test_block_scale_compile_rejects_invalid_explicit_config_before_cute_compile(monkeypatch) -> None:
    from cudnn.conv.frost.templates import sm100_block_scale_conv

    def unexpected_compile(*args, **kwargs):
        pytest.fail("cute.compile must not be called for an invalid block-scale tile config")

    monkeypatch.setattr(sm100_block_scale_conv.cute, "compile", unexpected_compile)
    config = ConvTileConfig(128, 128, 96, 2)
    with pytest.raises(ValueError, match=rf"{config.name}.*CTA N>=192"):
        sm100_block_scale_conv.compile(
            ncdhw=(1, 192, 1, 1, 129),
            ktrs=(64, 1, 1, 1),
            tile_config=config,
        )


@pytest.mark.L0
@requires_block_scale_cutedsl
@pytest.mark.parametrize("cta_n", (32, 96, 160, 224))
@pytest.mark.parametrize("cta_group", (1, 2))
def test_block_scale_rejects_unaligned_sfb_tiles_before_compile(monkeypatch, cta_n, cta_group) -> None:
    from cudnn.conv.frost.templates import sm100_block_scale_conv

    def unexpected_compile(*args, **kwargs):
        pytest.fail("an unaligned SFB tile must be rejected before compilation")

    monkeypatch.setattr(sm100_block_scale_conv, "_compile_cached", unexpected_compile)
    config = ConvTileConfig(128, cta_n, 128, cta_group)
    with pytest.raises(ValueError, match=rf"{config.name}.*CTA N.*64"):
        sm100_block_scale_conv.compile(
            ncdhw=(1, 1024, 1, 44, 80),
            ktrs=(1024, 3, 3, 3),
            lower_padding_dhw=(2, 1, 1),
            upper_padding_dhw=(0, 1, 1),
            tile_config=config,
        )


@pytest.mark.L0
@requires_block_scale_cutedsl
def test_block_scale_sweep_keeps_only_sfb_aligned_tiles() -> None:
    from cudnn.conv.frost.templates.sm100_block_scale_conv import _block_scale_config_violation

    # This is the same predicate used by the benchmark's --configs all sweep.
    selected = {config for config in CATALOG if _block_scale_config_violation(config, 1024) is None}
    assert selected == {ConvTileConfig(128, n, 128, group) for n in (64, 128, 192, 256) for group in (1, 2)}


@pytest.mark.L0
@requires_block_scale_cutedsl
def test_block_scale_compile_resolves_config_before_cached_compiler(monkeypatch) -> None:
    from cudnn.conv.frost.templates import sm100_block_scale_conv

    monkeypatch.setattr(sm100_block_scale_conv, "_compile_cached", lambda *args, **kwargs: kwargs["tile_config"])

    automatic = sm100_block_scale_conv.compile(ncdhw=(1, 64, 1, 1, 128), ktrs=(64, 1, 1, 1))
    explicit_config = ConvTileConfig(128, 192, 32, 2)
    explicit = sm100_block_scale_conv.compile(
        ncdhw=(1, 64, 1, 1, 128),
        ktrs=(64, 1, 1, 1),
        tile_config=explicit_config,
    )

    assert automatic == ConvTileConfig(128, 64, 32, 1)
    assert explicit is explicit_config


@pytest.mark.L0
@requires_block_scale_cutedsl
def test_block_scale_compiler_cache_keys_on_resolved_tile_config(monkeypatch) -> None:
    from cudnn.conv.frost.templates import sm100_block_scale_conv

    sm100_block_scale_conv._compile_cached.cache_clear()
    monkeypatch.setattr(sm100_block_scale_conv.cute, "compile", lambda kernel, *args, **kwargs: kernel.tile_config)
    config_64 = ConvTileConfig(128, 64, 32, 1)
    config_128 = ConvTileConfig(128, 128, 32, 1)
    try:
        result_64 = sm100_block_scale_conv.compile(ncdhw=(1, 64, 1, 1, 128), ktrs=(64, 1, 1, 1), tile_config=config_64)
        result_128 = sm100_block_scale_conv.compile(ncdhw=(1, 64, 1, 1, 128), ktrs=(64, 1, 1, 1), tile_config=config_128)
        repeated_64 = sm100_block_scale_conv.compile(ncdhw=(1, 64, 1, 1, 128), ktrs=(64, 1, 1, 1), tile_config=config_64)
        cache_info = sm100_block_scale_conv._compile_cached.cache_info()
    finally:
        sm100_block_scale_conv._compile_cached.cache_clear()

    assert (result_64, result_128, repeated_64) == (config_64, config_128, config_64)
    assert cache_info.misses == 2
    assert cache_info.hits == 1


@pytest.mark.L0
@requires_block_scale_cutedsl
def test_block_scale_compiler_cache_keys_on_epilogue_attributes(monkeypatch) -> None:
    from cudnn.conv.frost.templates import sm100_block_scale_conv

    sm100_block_scale_conv._compile_cached.cache_clear()
    monkeypatch.setattr(sm100_block_scale_conv, "get_epilogue_op", lambda epilogue, attrs: attrs)
    monkeypatch.setattr(sm100_block_scale_conv.cute, "compile", lambda kernel, *args, **kwargs: args[6])
    config = ConvTileConfig(128, 64, 32, 1)
    compile_kwargs = {
        "ncdhw": (1, 64, 1, 1, 128),
        "ktrs": (64, 1, 1, 1),
        "epilogue": "swish",
        "tile_config": config,
    }
    beta_two = (("swish_beta", 2.0),)
    try:
        default = sm100_block_scale_conv.compile(**compile_kwargs)
        attributed = sm100_block_scale_conv.compile(**compile_kwargs, epilogue_attrs=beta_two)
        repeated = sm100_block_scale_conv.compile(**compile_kwargs, epilogue_attrs=beta_two)
        cache_info = sm100_block_scale_conv._compile_cached.cache_info()
    finally:
        sm100_block_scale_conv._compile_cached.cache_clear()

    assert default == ()
    assert attributed == repeated == beta_two
    assert cache_info.misses == 2
    assert cache_info.hits == 1


@pytest.mark.L0
@requires_block_scale_cutedsl
def test_block_scale_compiler_cache_keys_on_tensor_shapes_not_runtime_geometry(monkeypatch) -> None:
    from cudnn.conv.frost.templates import sm100_block_scale_conv

    sm100_block_scale_conv._compile_cached.cache_clear()
    artifacts = []

    def record_compile(*args, **kwargs):
        artifact = object()
        artifacts.append(artifact)
        return artifact

    monkeypatch.setattr(sm100_block_scale_conv.cute, "compile", record_compile)
    compile_kwargs = {
        "ncdhw": (1, 64, 1, 1, 128),
        "ktrs": (64, 1, 1, 1),
        "tile_config": ConvTileConfig(128, 64, 32, 1),
    }
    try:
        upper_padded = sm100_block_scale_conv.compile(
            **compile_kwargs,
            upper_padding_dhw=(0, 0, 1),
        )
        lower_padded = sm100_block_scale_conv.compile(
            **compile_kwargs,
            lower_padding_dhw=(0, 0, 1),
        )
        unpadded = sm100_block_scale_conv.compile(**compile_kwargs)
        cache_info = sm100_block_scale_conv._compile_cached.cache_info()
    finally:
        sm100_block_scale_conv._compile_cached.cache_clear()

    assert upper_padded is lower_padded
    assert unpadded is not upper_padded
    assert artifacts == [upper_padded, unpadded]
    assert cache_info.misses == 2
    assert cache_info.hits == 1


@pytest.mark.L0
@requires_block_scale_cutedsl
def test_block_scale_kernel_derives_geometry_from_tile_config() -> None:
    import cutlass

    from cudnn.conv.frost.templates.sm100_block_scale_conv import _Sm100BlockScaledPersistentDenseImplicitGemmKernel

    config = ConvTileConfig(128, 192, 96, 2)
    kernel = _Sm100BlockScaledPersistentDenseImplicitGemmKernel(cutlass.Float32, 16, config)

    assert kernel.tile_config == config
    assert kernel.mma_tiler_mn == (256, 192)
    assert kernel.preferred_cluster_shape_mn == kernel.fallback_cluster_shape_mn == (2, 1)
    assert kernel.use_2cta_instrs
    assert kernel.cta_tile_k == 192


_REPRESENTATIVE_CONFIGS = (
    ConvTileConfig(64, 32, 64, 1),
    ConvTileConfig(64, 256, 128, 1),
    ConvTileConfig(64, 64, 64, 2),
    ConvTileConfig(64, 128, 128, 2),
    ConvTileConfig(128, 32, 64, 1),
    ConvTileConfig(128, 256, 128, 1),
    ConvTileConfig(128, 64, 64, 2),
    ConvTileConfig(128, 128, 128, 2),
)

_DENSE_CATALOG = tuple(
    config for config in CATALOG if config.cta_tile_k_bytes in (64, 128) and not (config.cta_tile_m == 64 and config.cta_group == 2 and config.cta_tile_n % 64)
)


def _run_bf16_config(tile_config: ConvTileConfig) -> None:
    import torch
    from cuda.bindings import driver as cuda

    from cudnn.conv.frost.templates.sm100_conv import compile

    channels = tile_config.cta_tile_k(16)
    image_shape = (1, channels, 2, 8, 10)
    weight_shape = (320, channels, 1, 1, 1)
    image_gpu = make_input(image_shape, torch.bfloat16, seed=1234)
    weight_gpu = make_input(weight_shape, torch.bfloat16, seed=5678)
    reference = torch.nn.functional.conv3d(image_gpu.float(), weight_gpu.float())
    output_gpu = torch.empty(reference.shape, dtype=torch.bfloat16, device="cuda", memory_format=torch.channels_last_3d)

    compiled = compile(
        ncdhw=image_shape,
        ktrs=(weight_shape[0], *weight_shape[2:]),
        tile_config=tile_config,
    )
    compiled(
        image_gpu.permute(0, 2, 3, 4, 1),
        weight_gpu.permute(0, 2, 3, 4, 1),
        output_gpu.permute(0, 2, 3, 4, 1),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    torch.cuda.synchronize()

    epsilon = float(torch.finfo(torch.bfloat16).eps)
    torch.testing.assert_close(output_gpu.float(), reference.to(torch.bfloat16).float(), atol=2 * epsilon, rtol=2 * epsilon)


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _unpack_fp4(packed, torch):
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=packed.device)
    low = lut[(packed & 0xF).long()]
    high = lut[(packed >> 4).long()]
    return torch.stack((low, high), dim=-1).flatten(-2)


def _reorder_f8_128x4(scales, torch):
    rows, columns = scales.shape
    row_blocks = (rows + 127) // 128
    column_blocks = (columns + 3) // 4
    padded = torch.zeros(row_blocks * 128, column_blocks * 4, dtype=scales.dtype, device=scales.device)
    padded[:rows, :columns] = scales
    blocks = padded.view(row_blocks, 128, column_blocks, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _run_block_scale_bf16_config(tile_config: ConvTileConfig, channels: int, implicit_m: int) -> None:
    import cutlass
    import torch
    from cuda.bindings import driver as cuda

    from cudnn.conv.frost.templates.sm100_block_scale_conv import compile

    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("PyTorch lacks native float4_e2m1fn_x2")

    output_channels = 256
    image_shape = (1, channels, 1, 1, implicit_m)
    generator = torch.Generator(device="cuda").manual_seed(2026 + channels + implicit_m)
    image_bytes = torch.randint(0, 256, (1, 1, 1, implicit_m, channels // 2), dtype=torch.uint8, device="cuda", generator=generator)
    weight_bytes = torch.randint(0, 256, (output_channels, 1, 1, 1, channels // 2), dtype=torch.uint8, device="cuda", generator=generator)
    image = image_bytes.view(torch.float4_e2m1fn_x2).permute(0, 4, 1, 2, 3)
    weight = weight_bytes.view(torch.float4_e2m1fn_x2).permute(0, 4, 1, 2, 3)

    sf_channels = channels // 16
    sfa_logical = (torch.randint(1, 3, (implicit_m, sf_channels), dtype=torch.int32, device="cuda", generator=generator) / 32).to(torch.float8_e4m3fn)
    sfb_logical = (torch.randint(1, 3, (output_channels, sf_channels), dtype=torch.int32, device="cuda", generator=generator) / 32).to(torch.float8_e4m3fn)
    sfa_columns = (sf_channels + 3) // 4 * 4
    sfa = torch.zeros(implicit_m, sfa_columns, 1, dtype=torch.float8_e4m3fn, device="cuda")
    sfa[:, :sf_channels, 0] = sfa_logical
    sfb = _reorder_f8_128x4(sfb_logical, torch)

    image_dequant = _unpack_fp4(image_bytes, torch) * sfa_logical.view(1, 1, 1, implicit_m, sf_channels).float().repeat_interleave(16, dim=-1)
    weight_dequant = _unpack_fp4(weight_bytes, torch) * sfb_logical.view(output_channels, 1, 1, 1, sf_channels).float().repeat_interleave(16, dim=-1)
    reference = torch.nn.functional.conv3d(image_dequant.permute(0, 4, 1, 2, 3), weight_dequant.permute(0, 4, 1, 2, 3))
    output = torch.empty(reference.shape, dtype=torch.bfloat16, device="cuda", memory_format=torch.channels_last_3d)

    compiled = compile(
        ncdhw=image_shape,
        ktrs=(output_channels, 1, 1, 1),
        d_dtype=cutlass.BFloat16,
        tile_config=tile_config,
    )
    compiled(
        image.permute(0, 2, 3, 4, 1),
        weight.permute(0, 2, 3, 4, 1),
        output.permute(0, 2, 3, 4, 1),
        sfa,
        sfb,
        cutlass.Float32(1.0),
        None,
        cutlass.Float32(1.0),
        None,
        None,
        *(cutlass.Int32(value) for value in ((0,) * 6 + (1,) * 6)),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    torch.cuda.synchronize()

    epsilon = float(torch.finfo(torch.bfloat16).eps)
    torch.testing.assert_close(output.float(), reference.to(torch.bfloat16).float(), atol=2 * epsilon, rtol=2 * epsilon)


_REPRESENTATIVE_BLOCK_SCALE_CONFIGS = (
    (ConvTileConfig(128, 64, 32, 1), 64, 128),
    (ConvTileConfig(128, 128, 64, 2), 128, 256),
    (ConvTileConfig(128, 256, 96, 1), 192, 128),
    (ConvTileConfig(128, 256, 96, 2), 192, 256),
    (ConvTileConfig(128, 256, 128, 2), 256, 256),
)


@pytest.mark.L2
@requires_sm100
@requires_block_scale_cutedsl
@pytest.mark.parametrize(
    "tile_config, channels, implicit_m",
    _REPRESENTATIVE_BLOCK_SCALE_CONFIGS,
    ids=lambda value: value.name if isinstance(value, ConvTileConfig) else str(value),
)
def test_representative_block_scale_conv_tile_configs(tile_config: ConvTileConfig, channels: int, implicit_m: int) -> None:
    _run_block_scale_bf16_config(tile_config, channels, implicit_m)


@pytest.mark.L2
@requires_sm100
@requires_dense_cutedsl
@pytest.mark.parametrize("tile_config", _REPRESENTATIVE_CONFIGS, ids=lambda config: config.name)
def test_representative_conv_tile_configs(tile_config: ConvTileConfig) -> None:
    _run_bf16_config(tile_config)


@pytest.mark.L4
@requires_sm100
@requires_dense_cutedsl
@pytest.mark.parametrize("tile_config", _DENSE_CATALOG, ids=lambda config: config.name)
def test_all_bf16_conv_tile_configs_compile_and_execute(tile_config: ConvTileConfig) -> None:
    _run_bf16_config(tile_config)
