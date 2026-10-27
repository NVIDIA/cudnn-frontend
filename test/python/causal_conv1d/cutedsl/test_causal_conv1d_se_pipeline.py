# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Partial tiles, ring-buffer reuse, and partial warp groups in SE pipelines."""

import json
import os
from pathlib import Path

import pytest

from causal_conv1d.cutedsl.se_bitwise import run_case
from causal_conv1d.cutedsl.test_causal_conv1d_se import _dsl

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


def _check(case):
    """Run a boundary case, retain optional diagnostics, and require its acceptance contract."""
    result = run_case(case)
    if root := os.environ.get("CUDNN_SE_BITWISE_RESULTS"):
        path = Path(root)
        path.mkdir(parents=True, exist_ok=True)
        (path / (case["name"] + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    assert result["accepted"], result


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("silu", [False, True])
@pytest.mark.parametrize("tiles", [4, 8])
def test_forward_pipeline_partial_tiles(monkeypatch, dtype, silu, tiles):
    """Force the staged path below its performance threshold to test both tails."""
    cute, _, _ = _dsl()
    import cutlass
    from cudnn.causal_conv1d_se import kernel

    @cute.jit
    def staged(
        x,
        w,
        b,
        y,
        stream,
        width: cutlass.Constexpr,
        activation: cutlass.Constexpr,
        nwh: cutlass.Constexpr,
        rows: cutlass.Constexpr,
        vector: cutlass.Constexpr,
        packed: cutlass.Constexpr,
    ):
        """Force the selected pipeline configuration to exercise partial tiles below its tuning threshold."""
        kernel.nwh_fwd_pipeline(x, w, b, y, width, activation, tiles, 64, 128).launch(
            grid=(cute.ceil_div(x.shape[2], 128), cute.ceil_div(x.shape[1], 64 * tiles), x.shape[0]),
            block=(128, 1, 1),
            stream=stream,
        )

    monkeypatch.setattr(kernel, "launch_fwd", staged)
    _check(
        dict(
            name=f"pipeline_fwd_{dtype}_{silu}_{tiles}",
            family="nwh",
            direction="forward",
            shape=[2, 136, 1031],
            width=4,
            mixer_width=None,
            dtype=dtype,
            silu=silu,
            iterations=3,
            min_device_version=800,
        )
    )


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("silu", [False, True])
def test_backward_pipeline_partial_tiles(dtype, silu):
    """Check backward sequence and channel tails across the pipelined NWH path."""
    _dsl()
    _check(
        dict(
            name=f"pipeline_bwd_{dtype}_{silu}",
            family="nwh",
            direction="backward",
            shape=[2, 1032, 1031],
            width=3,
            mixer_width=None,
            dtype=dtype,
            silu=silu,
            iterations=3,
            min_device_version=800,
        )
    )


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_fused_backward_partial_warp_group(dtype):
    """Check fused backward with a partial group of feature warps."""
    _dsl()
    _check(
        dict(
            name=f"fused_bwd_partial_group_{dtype}",
            family="b2b",
            direction="backward",
            shape=[1, 7, 1031],
            width=3,
            mixer_width=3,
            dtype=dtype,
            silu=False,
            iterations=3,
            min_device_version=800,
        )
    )


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("width", range(2, 9))
@pytest.mark.parametrize("silu", [False, True])
def test_backward_seqpar_rounding(dtype, width, silu):
    """Cover full/masked 64x64 tiles and both backend double-buffer stages."""
    _dsl()
    _check(
        dict(
            name=f"seqpar_rounding_{dtype}_{width}_{silu}",
            family="nwh",
            direction="backward",
            shape=[2, 1032, 1031],
            width=width,
            mixer_width=None,
            dtype=dtype,
            silu=silu,
            iterations=3,
            min_device_version=800,
        )
    )


@pytest.mark.parametrize(
    "dtype,width,channels,length,silu",
    [
        ("bfloat16", 33, 65, 257, True),
        ("bfloat16", 33, 768, 256, False),
        ("bfloat16", 64, 768, 256, False),
        ("bfloat16", 64, 65, 257, True),
        ("bfloat16", 65, 65, 257, True),
        ("bfloat16", 128, 65, 257, False),
        ("bfloat16", 128, 65, 257, True),
        ("float16", 64, 65, 257, True),
        ("float32", 16, 66, 257, True),
        ("float32", 128, 768, 256, False),
        ("float32", 128, 768, 256, True),
        ("float32", 128, 66, 257, True),
    ],
)
def test_forward_backend_fallback(dtype, width, channels, length, silu):
    """Backend fallback changes tap order and half-precision SiLU evaluation."""
    _dsl()
    _check(
        dict(
            name=f"fwd_fallback_{dtype}_{width}_{channels}_{length}_{silu}",
            family="nwh",
            direction="forward",
            shape=[1, channels, length],
            width=width,
            mixer_width=None,
            dtype=dtype,
            silu=silu,
            iterations=3,
            min_device_version=800,
        )
    )


@pytest.mark.parametrize(
    "dtype,width,tiles,tile_s,tile_h",
    [
        ("bfloat16", 4, 4, 64, 64),
        ("float16", 4, 4, 64, 64),
        ("bfloat16", 4, 16, 32, 128),
        ("float16", 4, 16, 32, 128),
        ("float32", 7, 4, 64, 64),
        ("bfloat16", 3, 8, 64, 64),
        ("float16", 3, 8, 64, 64),
        ("bfloat16", 4, 4, 64, 128),
        ("float16", 4, 4, 64, 128),
        ("bfloat16", 4, 8, 32, 128),
        ("float16", 4, 8, 32, 128),
    ],
)
@pytest.mark.parametrize("silu", [False, True])
def test_forward_triple_buffer_partial_tiles(monkeypatch, dtype, width, tiles, tile_s, tile_h, silu):
    """Exercise shared halo carry, buffer wraparound, and both partial dimensions."""
    cute, _, _ = _dsl()
    import cutlass
    from cudnn.causal_conv1d_se import kernel

    @cute.jit
    def staged(
        x,
        w,
        b,
        y,
        stream,
        width: cutlass.Constexpr,
        activation: cutlass.Constexpr,
        nwh: cutlass.Constexpr,
        rows: cutlass.Constexpr,
        vector: cutlass.Constexpr,
        packed: cutlass.Constexpr,
    ):
        """Force the selected pipeline configuration to exercise partial tiles below its tuning threshold."""
        kernel.nwh_fwd_pipeline3(x, w, b, y, width, activation, tiles, tile_s, tile_h).launch(
            grid=(cute.ceil_div(x.shape[2], tile_h), cute.ceil_div(x.shape[1], tile_s * tiles), x.shape[0]), block=(128, 1, 1), stream=stream
        )

    monkeypatch.setattr(kernel, "launch_fwd", staged)
    _check(
        dict(
            name=f"triple_fwd_{dtype}_{width}_{tiles}_{tile_s}_{tile_h}_{silu}",
            family="nwh",
            direction="forward",
            shape=[2, 136, 1031],
            width=width,
            mixer_width=None,
            dtype=dtype,
            silu=silu,
            iterations=3,
            min_device_version=800,
        )
    )


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("projection_width,mixer_width", [(9, 17), (32, 256)])
@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_fused_wide_partial_tiles(dtype, projection_width, mixer_width, direction):
    """Wide register/shared filters must work with partial warps and sequence tiles."""
    _dsl()
    _check(
        dict(
            name=f"fused_wide_{dtype}_{projection_width}_{mixer_width}_{direction}",
            family="b2b",
            direction=direction,
            shape=[1, 7, 1031],
            width=projection_width,
            mixer_width=mixer_width,
            dtype=dtype,
            silu=False,
            iterations=3,
            min_device_version=800,
        )
    )


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("width", [32, 128])
def test_ncw_backward_wide_silu(dtype, width):
    """Distributed weights also feed the recomputed SiLU halo."""
    _dsl()
    _check(
        dict(
            name=f"ncw_wide_silu_{dtype}_{width}",
            family="ncw",
            direction="backward",
            shape=[1, 7, 257],
            width=width,
            mixer_width=None,
            dtype=dtype,
            silu=True,
            iterations=3,
            min_device_version=800,
        )
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("length", [256, 1031])
def test_fused_wide_compute_precision(dtype, length):
    """Pin the skip/current-tap FMA order before projection gradients."""
    _dsl()
    _check(
        dict(
            name=f"fused_wide_{dtype}_32_256_backward_{length}",
            family="b2b",
            direction="backward",
            shape=[1, 7, length],
            width=32,
            mixer_width=256,
            dtype=dtype,
            silu=False,
            iterations=3,
            min_device_version=800,
        )
    )


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_fused_split_sequence_aligned(dtype, direction):
    """Check split-CTA halo reconstruction with multiple batches and partial warps."""
    _dsl()
    _check(
        dict(
            name=f"fused_split_aligned_{dtype}_{direction}",
            family="b2b",
            direction=direction,
            shape=[2, 7, 1024],
            width=32,
            mixer_width=256,
            dtype=dtype,
            silu=False,
            iterations=3,
            min_device_version=800,
        )
    )


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("silu", [False, True])
@pytest.mark.parametrize("channels,length", [(64, 128), (65, 129)])
def test_forward_shared_filter_boundaries(dtype, silu, channels, length):
    """Cover complete/partial shared-filter CTAs and output rows at K128."""
    _dsl()
    _check(
        dict(
            name=f"shared_filter_{dtype}_{silu}_{channels}_{length}",
            family="nwh",
            direction="forward",
            shape=[2, channels, length],
            width=128,
            mixer_width=None,
            dtype=dtype,
            silu=silu,
            iterations=3,
            min_device_version=800,
        )
    )


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("mixer_width", [64, 128])
@pytest.mark.parametrize("length", [1024, 1031])
def test_fused_split_backward_intermediate_widths(dtype, mixer_width, length):
    """Future projection gradients need a second halo tile near split boundaries."""
    _dsl()
    _check(
        dict(
            name=f"fused_split_bwd_{dtype}_{mixer_width}_{length}",
            family="b2b",
            direction="backward",
            shape=[2, 7, length],
            width=32,
            mixer_width=mixer_width,
            dtype=dtype,
            silu=False,
            iterations=3,
            min_device_version=800,
        )
    )


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("silu", [False, True])
@pytest.mark.parametrize("length", [255, 256])
def test_ncw_short_reduction_boundaries(dtype, silu, length):
    """Check one-warp K32 reduction CTAs with aligned and masked input vectors."""
    _dsl()
    _check(
        dict(
            name=f"ncw_short_reduction_{dtype}_{silu}_{length}",
            family="ncw",
            direction="backward",
            shape=[2, 7, length],
            width=32,
            mixer_width=None,
            dtype=dtype,
            silu=silu,
            iterations=3,
            min_device_version=800,
        )
    )
