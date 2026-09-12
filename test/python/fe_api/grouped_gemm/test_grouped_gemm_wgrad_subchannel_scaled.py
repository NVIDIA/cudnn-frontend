# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the Subchannel-Scaled Grouped GEMM Weight Gradient Kernel (SM100+)

The 2Dx2D wgrad grouped GEMM with a second-level A scale (SFA2 only, no SFB2): NVFP4 A/B
with per-(1, 16) e4m3 first-level scales plus one f32 SFA2 per (hidden row x sgk tokens),
ragged per-expert token counts along K, BF16 dW. The kernel output is compared BYTE-EXACT
(``torch.equal``) against the pure-torch reference -- do not loosen this to a
tolerance-based comparison; find the real gap instead.
"""

import pytest
import torch

from test_utils import torch_fork_set_rng
from fe_api.test_fe_api_utils import reencode_sf_tensor_as_ue5m3
from fe_api.grouped_gemm.test_grouped_gemm_wgrad_utils import _skip_unless_e5m3_supported
from fe_api.grouped_gemm.test_grouped_gemm_wgrad_subchannel_scaled_utils import (
    assert_bytes_equal,
    make_wgrad_subchannel_problem,
    wgrad_baseline_reference,
    wgrad_subchannel_reference,
)


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        pytest.skip("SM100 is required")


def _wrapper():
    from cudnn import grouped_gemm_wgrad_subchannel_scaled_wrapper_sm100

    return grouped_gemm_wgrad_subchannel_scaled_wrapper_sm100


def _run(problem, output_mode="dense", **overrides):
    kwargs = dict(
        a_tensor=problem["a_tensor"],
        b_tensor=problem["b_tensor"],
        sfa_tensor=problem["sfa_tensor"],
        sfb_tensor=problem["sfb_tensor"],
        sfa2_tensor=problem["sfa2_tensor"],
        offsets_tensor=problem["offsets_tensor"],
        sgk=problem["sgk"],
        output_mode=output_mode,
        global_scale_a=problem["global_scale_a"],
        global_scale_b=problem["global_scale_b"],
    )
    kwargs.update(overrides)
    return _wrapper()(**kwargs)["wgrad_tensor"]


def _apply_sf_override(problem, sf_fp8_dtype_override):
    """Prepare ``problem`` for ``sf_fp8_dtype_override`` and return the wrapper kwargs for it.

    For ``"e5m3"`` the e4m3 first-level scale bytes are rewritten in place as UE5M3. Every
    non-negative finite e4m3 value is exactly representable in UE5M3 (same 3-bit mantissa,
    wider exponent range), so the dequantized reference operands are unchanged while the
    bytes handed to the kernel differ -- a kernel still decoding them as e4m3 would fail
    the byte-exact comparison.
    """
    if sf_fp8_dtype_override is None:
        return {}
    assert sf_fp8_dtype_override == "e5m3"
    _skip_unless_e5m3_supported()
    reencode_sf_tensor_as_ue5m3(problem["sfa_tensor"])
    reencode_sf_tensor_as_ue5m3(problem["sfb_tensor"])
    return {"sf_fp8_dtype_override": "e5m3"}


SF_OVERRIDES = pytest.mark.parametrize("sf_fp8_dtype_override", [None, "e5m3"], ids=["sf_e4m3", "sf_e5m3"])

CONFIGS = pytest.mark.parametrize(
    "mma_tiler_mn, cluster_shape_mn",
    [((128, 128), (1, 1)), ((256, 128), (2, 1))],
    ids=["t128x128-c1x1", "t256x128-c2x1"],
)

# Each set has an empty expert and a single-block expert; token counts are sgk-aligned.
SHAPES = pytest.mark.parametrize(
    "sgk, hidden, intermediate, token_counts",
    [
        pytest.param(256, 512, 512, (256, 512, 0, 768, 256), id="sg256-ragged-l5"),
        pytest.param(512, 512, 512, (512, 1536, 0, 512), id="sg512-ragged-l4"),
    ],
)

OUTPUT_MODES = pytest.mark.parametrize("output_mode", ["dense", "discrete"])


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@CONFIGS
@SHAPES
@OUTPUT_MODES
@SF_OVERRIDES
def test_wgrad_subchannel_matches_reference(mma_tiler_mn, cluster_shape_mn, sgk, hidden, intermediate, token_counts, output_mode, sf_fp8_dtype_override):
    problem = make_wgrad_subchannel_problem(hidden, intermediate, token_counts, sgk)
    sf_kwargs = _apply_sf_override(problem, sf_fp8_dtype_override)
    out = _run(problem, output_mode=output_mode, mma_tiler_mn=mma_tiler_mn, cluster_shape_mn=cluster_shape_mn, **sf_kwargs)
    assert_bytes_equal(out, wgrad_subchannel_reference(problem))


@pytest.mark.L1
@torch_fork_set_rng(seed=0)
@CONFIGS
def test_wgrad_subchannel_large(mma_tiler_mn, cluster_shape_mn):
    problem = make_wgrad_subchannel_problem(1024, 1024, (2048, 512, 1024, 512), 512)
    out = _run(problem, mma_tiler_mn=mma_tiler_mn, cluster_shape_mn=cluster_shape_mn)
    assert_bytes_equal(out, wgrad_subchannel_reference(problem))


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@OUTPUT_MODES
def test_wgrad_subchannel_no_global_scale(output_mode):
    # global scales absent is a distinct compile branch (const_expr alpha off).
    problem = make_wgrad_subchannel_problem(512, 512, (512, 1024, 512), 512, global_scales=False)
    out = _run(problem, output_mode=output_mode)
    assert_bytes_equal(out, wgrad_subchannel_reference(problem))


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_wgrad_subchannel_ones_sfa2_matches_baseline():
    # sfa2 == 1 makes the kernel mathematically identical to the single-level GEMM --
    # isolates the accumulator-splitting plumbing from the scale math (the per-block
    # partial sums recombine exactly in f32).
    problem = make_wgrad_subchannel_problem(512, 512, (512, 1024, 0, 512), 512)
    ones = torch.ones_like(problem["sfa2_tensor"].T).T
    out = _run(problem, sfa2_tensor=ones)
    assert_bytes_equal(out, wgrad_baseline_reference(problem))


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@OUTPUT_MODES
@SF_OVERRIDES
def test_wgrad_subchannel_accumulate(output_mode, sf_fp8_dtype_override):
    # accumulate_on_output TMA-reduce-adds the bf16 tile into the caller's buffer; a
    # zero-token expert must reduce-add zeros (init preserved).
    problem = make_wgrad_subchannel_problem(512, 512, (512, 0, 512, 1024), 512)
    sf_kwargs = _apply_sf_override(problem, sf_fp8_dtype_override)
    l = len(problem["token_counts"])
    init = torch.randn((l, 512, 512), device="cuda").to(torch.bfloat16)
    out_buf = init.clone()
    out = _run(problem, output_mode=output_mode, wgrad_tensor=out_buf, accumulate_on_output=True, **sf_kwargs)
    assert out is out_buf
    assert_bytes_equal(out, wgrad_subchannel_reference(problem, out_init=init, accumulate=True))


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_wgrad_subchannel_class_api_dense():
    from cudnn import GroupedGemmWgradSubchannelScaledSm100

    problem = make_wgrad_subchannel_problem(512, 512, (768, 256), 256)
    out = torch.empty((2, 512, 512), dtype=torch.bfloat16, device="cuda")
    op = GroupedGemmWgradSubchannelScaledSm100(
        sample_a=problem["a_tensor"],
        sample_b=problem["b_tensor"],
        sample_sfa=problem["sfa_tensor"],
        sample_sfb=problem["sfb_tensor"],
        sample_sfa2=problem["sfa2_tensor"],
        sample_offsets=problem["offsets_tensor"],
        sgk=256,
        sample_wgrad=out,
        sample_global_scale_a=problem["global_scale_a"],
        sample_global_scale_b=problem["global_scale_b"],
    )
    assert op.check_support()
    op.compile()
    op.execute(
        a_tensor=problem["a_tensor"],
        b_tensor=problem["b_tensor"],
        sfa_tensor=problem["sfa_tensor"],
        sfb_tensor=problem["sfb_tensor"],
        sfa2_tensor=problem["sfa2_tensor"],
        offsets_tensor=problem["offsets_tensor"],
        wgrad_tensor=out,
        global_scale_a=problem["global_scale_a"],
        global_scale_b=problem["global_scale_b"],
    )
    assert_bytes_equal(out, wgrad_subchannel_reference(problem))


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@OUTPUT_MODES
def test_wgrad_subchannel_dynamic_tokens_reuse_compile(output_mode):
    # The token axis is dynamic: two token distributions with the same expert count share
    # one compiled kernel, and both stay byte-exact.
    from cudnn.gemm.cutedsl.grouped.wgrad_subchannel_scaled import api

    p1 = make_wgrad_subchannel_problem(512, 512, (512, 1024, 512), 512, seed=1)
    p2 = make_wgrad_subchannel_problem(512, 512, (1536, 0, 512), 512, seed=2)
    before = len(api._cache_of_GroupedGemmWgradSubchannelScaledSm100Objects)
    out1 = _run(p1, output_mode=output_mode)
    assert_bytes_equal(out1, wgrad_subchannel_reference(p1))
    after_first = len(api._cache_of_GroupedGemmWgradSubchannelScaledSm100Objects)
    out2 = _run(p2, output_mode=output_mode)
    assert_bytes_equal(out2, wgrad_subchannel_reference(p2))
    assert len(api._cache_of_GroupedGemmWgradSubchannelScaledSm100Objects) == after_first
    assert after_first >= before


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_wgrad_subchannel_rejects():
    problem = make_wgrad_subchannel_problem(512, 512, (512, 512), 512)
    # Token counts aligned to 256 but not to sgk=512 (tokens_sum itself is sgk-aligned).
    bad = make_wgrad_subchannel_problem(512, 512, (256, 768), 256)
    with pytest.raises(ValueError, match="sgk-aligned"):
        _run(bad, sgk=512, sfa2_tensor=problem["sfa2_tensor"])
    # sfa2 must be hidden-contiguous (the producer's view); a row-major copy is rejected.
    with pytest.raises(ValueError, match="hidden-contiguous"):
        _run(problem, sfa2_tensor=problem["sfa2_tensor"].contiguous())
    # Wrong sfa2 column count.
    with pytest.raises(ValueError):
        _run(problem, sfa2_tensor=problem["sfa2_tensor"][:, :1])
    # sgk must be a multiple of the 256-token MMA K tile.
    with pytest.raises(ValueError, match="MMA K tile"):
        _run(problem, sgk=128)
    # Only the harness-validated tilers / clusters.
    with pytest.raises(ValueError, match="mma_tiler_mn must be one of"):
        _run(problem, mma_tiler_mn=(256, 256))
    with pytest.raises(ValueError, match="cluster_shape_mn must be one of"):
        _run(problem, mma_tiler_mn=(128, 128), cluster_shape_mn=(3, 1))
    # Unknown override formats.
    with pytest.raises(ValueError, match="sf_fp8_dtype_override must be"):
        _run(problem, sf_fp8_dtype_override="e4m3")


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_wgrad_subchannel_e5m3_rejected_off_rubin(monkeypatch):
    # e5m3 scale factors decode only in the sm107 MMA op; pretend to be Blackwell.
    import cudnn.api_base as api_base
    from cudnn import GroupedGemmWgradSubchannelScaledSm100

    monkeypatch.setattr(api_base, "get_device_type", lambda: "blackwell")
    problem = make_wgrad_subchannel_problem(512, 512, (512, 512), 512)
    op = GroupedGemmWgradSubchannelScaledSm100(
        sample_a=problem["a_tensor"],
        sample_b=problem["b_tensor"],
        sample_sfa=problem["sfa_tensor"],
        sample_sfb=problem["sfb_tensor"],
        sample_sfa2=problem["sfa2_tensor"],
        sample_offsets=problem["offsets_tensor"],
        sgk=512,
        sample_wgrad=torch.empty((2, 512, 512), dtype=torch.bfloat16, device="cuda"),
        sf_fp8_dtype_override="e5m3",
    )
    with pytest.raises(ValueError, match="requires Rubin"):
        op.check_support()


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_wgrad_subchannel_e5m3_is_not_cached_as_e4m3():
    """sf_fp8_dtype_override must take part in the compile cache key: identical scale
    bytes decode differently under E4M3 and UE5M3."""
    _skip_unless_e5m3_supported()
    problem = make_wgrad_subchannel_problem(512, 512, (512, 512), 512)
    w_e4m3 = _run(problem).float().clone()
    w_e5m3 = _run(problem, sf_fp8_dtype_override="e5m3").float().clone()
    torch.cuda.synchronize()
    assert not torch.equal(w_e4m3, w_e5m3), "e5m3 and e4m3 produced identical output from identical scale-factor bytes"
