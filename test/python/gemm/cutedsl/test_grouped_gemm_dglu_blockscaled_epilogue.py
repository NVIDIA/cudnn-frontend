# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block-scaled dGLU (SM100) epilogue across tiles, cluster shapes and schedulers.

The SF TMEM columns are the SFA / SFB TMEM layouts' footprints, so where the overlapping accumulator meets the SF
region, and when it is released, depends on the tile and the format.  The column SFD takes one warp reduction per
column and computes each column's scale in its own lane.  Neither changes a value: every tile, cluster shape and
scheduler must give the same output bytes.

- MXFP8: the E4M3 row and column D, both SFD planes and the deterministic dprob slots, on the stock dynamic scheduler
  (2-CTA 256 x 256 and 1-CTA 128 x 256), static clusters and the streaming input signal scheduler (every M tile
  ready).  Ragged experts (one empty) vary the per-expert column SFD strides; NaN pre-activations exercise the
  NaN-propagating maxima.
- MXF4 (32-element E8M0 scale vectors) and NVF4 (16-element E4M3 or E8M0 scale vectors): BF16 D, each also checked
  against the float reference.
"""

import pytest
import torch

ROWS = [256, 1280, 512, 0, 768, 2048, 256, 1024]
K, I = 1024, 1024
#: (MMA tile, cluster shape, dynamic scheduler, extra options); the first is the reference.
CONFIGS = {
    "2cta_dynamic": ((256, 256), (2, 1), True, {}),
    "2cta_static_21": ((256, 256), (2, 1), False, {}),
    "2cta_static_22": ((256, 256), (2, 2), False, {}),
    "1cta_dynamic": ((128, 256), (1, 1), True, {}),
    "1cta_static_12": ((128, 256), (1, 2), False, {}),
    "signal": ((256, 256), (2, 1), True, dict(streaming_signal_capacity=sum(ROWS) // 256)),
}
REFERENCE = "2cta_dynamic"


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("the SM100 block-scaled dGLU kernel")
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

    if cutedsl_too_old(cutedsl_state()[1]):
        pytest.skip("CuTe DSL older than the library floor")


# ---- MXFP8: E4M3 D with both SFD planes ----


def make_inputs(nan):
    g = torch.Generator(device="cuda").manual_seed(3)
    m, e = sum(ROWS), len(ROWS)
    fp8 = lambda *s: (torch.randn(*s, generator=g, device="cuda") * 2).to(torch.float8_e4m3fn)
    scales = lambda n: torch.randint(124, 131, (n,), generator=g, device="cuda", dtype=torch.uint8)
    c = (torch.randn(m, 2 * I, generator=g, device="cuda") * 2).to(torch.bfloat16)
    if nan:
        c.view(-1)[::99991] = float("nan")
    return dict(
        a=fp8(m, K),
        sfa=scales(m * K // 32),
        b=fp8(e, I, K),
        sfb=scales(e * I * K // 32),
        c=c,
        offsets=torch.tensor(ROWS, device="cuda").cumsum(0).to(torch.int32),
        alpha=torch.rand(e, generator=g, device="cuda") + 0.5,
        beta=torch.rand(e, generator=g, device="cuda") + 0.5,
        prob=torch.rand(m, 1, 1, generator=g, device="cuda"),
    )


def run_mxfp8(inputs, config):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import OperandMajorMode
    from cutlass.cute.runtime import from_dlpack
    from cutlass.utils import HardwareInfo
    from cudnn.gemm.cutedsl.grouped.dglu.moe_blockscaled_grouped_gemm_dglu_dbias import BlockScaledMoEGroupedGemmDgluDbiasKernel
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode

    m, e = sum(ROWS), len(ROWS)
    mma_tiler_mn, cluster_shape_mn, dynamic, options = CONFIGS[config]
    kernel = BlockScaledMoEGroupedGemmDgluDbiasKernel(
        sf_vec_size=32,
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=mma_tiler_mn[0] == 256,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        vectorized_f32=False,
        discrete_col_sfd=True,
        expert_cnt=e,
        weight_mode=MoEWeightMode.DENSE,
        use_dynamic_sched=dynamic,
        act_func="dswiglu",
        deterministic=True,
        **options,
    )
    fill = lambda n: torch.full((n,), 0xA5, device="cuda", dtype=torch.uint8)
    out = dict(d=fill(m * 2 * I), d_col=fill(m * 2 * I), sfd_row=fill(m * 2 * I // 32), sfd_col=fill(m * 2 * I // 32))
    dprob = torch.zeros(m, I // 256, device="cuda")
    workspace = torch.zeros(max(256, kernel.get_workspace_bytes()), device="cuda", dtype=torch.uint8)
    t = lambda x: from_dlpack(x, assumed_align=16)
    e8 = lambda x: t(x.view(torch.float8_e8m0fnu))
    codes = lambda x: t(x.view(m, 2 * I).view(torch.float8_e4m3fn))
    args = [
        t(inputs["a"]),
        t(inputs["b"].permute(1, 2, 0)),
        e8(inputs["sfb"]),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int64(0),
        OperandMajorMode.K,
        from_dlpack(workspace, assumed_align=128).iterator,
        t(inputs["c"]),
        codes(out["d"]),
        codes(out["d_col"]),
        e8(inputs["sfa"]),
        e8(out["sfd_row"]),
        e8(out["sfd_col"]),
        None,
        t(torch.ones(1, device="cuda")),
        t(inputs["offsets"]),
        t(inputs["alpha"]),
        t(inputs["beta"]),
        t(inputs["prob"]),
        t(dprob.unsqueeze(-1)),
        None,
    ]
    if kernel.use_streaming_signal_sched:
        from cudnn.gemm.cutedsl.grouped.moe_kernel_helpers import reset_streaming_signal_workspace

        reset_streaming_signal_workspace(workspace, kernel, all_complete=True)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    cluster_size = cluster_shape_mn[0] * cluster_shape_mn[1]
    compiled = cute.compile(kernel, *args, HardwareInfo().get_max_active_clusters(cluster_size), stream)
    compiled(*[x for i, x in enumerate(args) if i != 6], stream)
    torch.cuda.synchronize()
    out["dprob"] = dprob
    return out


MXFP8_REFERENCE = {}


@pytest.mark.L0
@pytest.mark.parametrize("nan", [False, True], ids=["finite", "nan_inputs"])
@pytest.mark.parametrize("config", [config for config in CONFIGS if config != REFERENCE])
def test_mxfp8_dglu_epilogue_matches_across_configs(config, nan):
    inputs = make_inputs(nan)
    if nan not in MXFP8_REFERENCE:
        MXFP8_REFERENCE[nan] = run_mxfp8(inputs, REFERENCE)
    want = MXFP8_REFERENCE[nan]
    got = run_mxfp8(inputs, config)
    for name in want:
        assert torch.equal(got[name].view(torch.uint8), want[name].view(torch.uint8)), f"{config}: {name} differs from {REFERENCE}"
    assert not (want["sfd_col"] == 0xA5).all(), "the column SFD was written"


# ---- MXF4 and NVF4: BF16 D ----

#: (scale vector size, scale dtype)
FP4_FORMATS = {
    "mxf4": (32, torch.float8_e8m0fnu),
    "nvf4": (16, torch.float8_e4m3fn),
    "nvf4_e8m0": (16, torch.float8_e8m0fnu),
}
FP4_CONFIGS = [config for config in CONFIGS if config not in (REFERENCE, "signal")]
FP4_L0 = {"mxf4", "nvf4"}
FP4_REFERENCE = {}


def run_fp4(fmt, config, request):
    """The dense dGLU through GroupedGemmDgluSm100 on the same seeded inputs (also checked against the float
    reference); returns the output tensors by name."""
    from gemm.cutedsl.test_grouped_gemm_dglu import _test_grouped_gemm_dglu_dense_compile_execute

    sf_vec_size, sf_dtype = FP4_FORMATS[fmt]
    mma_tiler_mn, cluster_shape_mn, dynamic, _ = CONFIGS[config]
    torch.manual_seed(5)
    _, outputs, _ = _test_grouped_gemm_dglu_dense_compile_execute(
        ab_dtype=torch.float4_e2m1fn_x2,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        b_major="k",
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        vector_f32=False,
        discrete_col_sfd=False,
        use_dynamic_sched=dynamic,
        request=request,
    )
    return {name: value.clone() for name, value in outputs.items() if isinstance(value, torch.Tensor)}


@pytest.mark.parametrize(
    "fmt, config",
    [
        pytest.param(fmt, config, marks=pytest.mark.L0 if fmt in FP4_L0 else pytest.mark.L1, id=f"{fmt}-{config}")
        for fmt in FP4_FORMATS
        for config in FP4_CONFIGS
    ],
)
def test_fp4_dglu_epilogue_matches_across_configs(fmt, config, request):
    if fmt not in FP4_REFERENCE:
        FP4_REFERENCE[fmt] = run_fp4(fmt, REFERENCE, request)
    want = FP4_REFERENCE[fmt]
    got = run_fp4(fmt, config, request)
    assert set(got) == set(want)
    for name in want:
        if name.startswith("dprob"):
            # The dprob partials are summed with atomics in tile arrival order: equal within FP32 reassociation.
            atol = 8 * torch.finfo(torch.float32).eps * max(want[name].abs().max().item(), 1.0)
            torch.testing.assert_close(got[name], want[name], rtol=1e-5, atol=atol)
        else:
            same = torch.equal(got[name].reshape(-1).view(torch.uint8), want[name].reshape(-1).view(torch.uint8))
            assert same, f"{fmt} {config}: {name} differs from {REFERENCE}"
