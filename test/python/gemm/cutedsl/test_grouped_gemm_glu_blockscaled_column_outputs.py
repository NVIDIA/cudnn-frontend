# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The MXFP8 GLU kernel (SM100) without its column outputs.

d_col and sfd_col None: the row-quantized D and row SFD only, byte-equal to the kernel that also writes the column
pair; the column pair is not written, and C is staged once when that buys another A/B stage.  The outputs are E4M3 D
with its SFD (the EP layer's FC1 output), which the streaming signal suite (BF16 D) does not cover.
"""

import pytest
import torch

from gemm.cutedsl import _streaming_signal_blockscaled as bs
from gemm.cutedsl._streaming_signal import signal_ready
from gemm.cutedsl._streaming_signal_blockscaled import require_sm10x  # noqa: F401  (autouse fixture)

#: E4M3 NaN codes and E8M0 NaN scales: an output the kernel should write but does not stays poisoned.
NAN_CODE, NAN_SCALE = 0x7F, 0xFF


@pytest.fixture
def require_sm100():
    if bs.is_rubin():
        pytest.skip("the SM100 kernels")


def glu_buffers(problem, kernel):
    """C, D, D_col, SFD_row, SFD_col, norm_const and the workspace: NaN C, poisoned E4M3 codes and E8M0 scales."""
    rows, n = problem["rows"], problem["n"]
    width = n // 2

    def codes():
        return torch.full((rows, width), NAN_CODE, device="cuda", dtype=torch.uint8)

    def scales():
        return torch.full((rows * width // bs.SF_VEC_SIZE,), NAN_SCALE, device="cuda", dtype=torch.uint8)

    return dict(
        c=torch.full((rows, n), float("nan"), device="cuda", dtype=torch.bfloat16),
        d=codes(),
        d_col=codes(),
        sfd_row=scales(),
        sfd_col=scales(),
        norm_const=torch.ones(1, device="cuda", dtype=torch.float32),
        workspace=torch.zeros(max(kernel.get_workspace_bytes(), 1), device="cuda", dtype=torch.uint8),
    )


def glu_args(kernel, problem, buf, col, c=True):
    import cutlass
    from cutlass.cute.nvgpu import OperandMajorMode
    from cutlass.cute.runtime import from_dlpack

    def tensor(x):
        return from_dlpack(x, assumed_align=16)

    def e4m3(x):
        return tensor(x.view(torch.float8_e4m3fn))

    def e8m0(x):
        return tensor(x.view(torch.float8_e8m0fnu))

    n, k = problem["n"], problem["k"]
    head = [tensor(problem["a"]), tensor(problem["b"]), e8m0(problem["sfb"])]
    head += [cutlass.Int32(n), cutlass.Int32(k), cutlass.Int64(k), OperandMajorMode.K, from_dlpack(buf["workspace"], assumed_align=128).iterator]
    offsets, alpha, prob = from_dlpack(problem["offsets"], assumed_align=4), tensor(problem["alpha"]), tensor(problem["prob"])
    sfa = e8m0(problem["a_exp"] if kernel.test_compact else problem["sfa"])
    d_col = e4m3(buf["d_col"]) if col else None
    sfd_col = e8m0(buf["sfd_col"]) if col else None
    # (c, d, d_col, sfa, sfd_row, sfd_col, amax, norm_const, offsets, alpha, prob, bias)
    return head + [
        tensor(buf["c"]) if c else None,
        e4m3(buf["d"]),
        d_col,
        sfa,
        e8m0(buf["sfd_row"]),
        sfd_col,
        None,
        tensor(buf["norm_const"]),
        offsets,
        alpha,
        prob,
        None,
    ]


COMPILED = {}


def run_glu(problem, col=True, c=True, **options):
    """One launch with E4M3 D and SFD (and the column pair unless ``col`` is False, C unless ``c`` is False); returns
    the outputs by name."""
    import cutlass.cute as cute
    from cutlass.utils import HardwareInfo

    kernel = bs.make_kernel("glu", len(problem["layout"]) - 1, generate_sfd=True, **options)
    buf = glu_buffers(problem, kernel)
    args = glu_args(kernel, problem, buf, col, c)
    clusters = HardwareInfo().get_max_active_clusters(kernel.test_cluster_size)
    key = (kernel.test_compile_key, col, c, problem["rows"], problem["n"])
    if key not in COMPILED:
        COMPILED[key] = cute.compile(kernel, *args, clusters, bs.cu_stream())
    if kernel.test_signal:
        signal_ready(kernel)(buf["workspace"])
    COMPILED[key](*[x for i, x in enumerate(args) if i != bs.b_major_arg(kernel)], bs.cu_stream())
    torch.cuda.synchronize()
    rows = bs.valid_rows(problem)
    names = ["d", "sfd_row"] + (["c"] if c else []) + (["d_col", "sfd_col"] if col else [])
    return {name: buf[name][:rows] if name in ("c", "d", "d_col") else buf[name] for name in names}


def assert_equal_bytes(got, want, label):
    for name in got:
        a, b = got[name], want[name]
        if a.dtype == torch.bfloat16:
            a, b = a.view(torch.uint8), b.view(torch.uint8)
        assert torch.equal(a, b), f"{label}: {name} differs"


#: (scheduler variant, routing): the stock dynamic scheduler, and the signal scheduler with compact SFA.
VARIANTS = ("dynamic", "compact_signal")
CASES = [(variant, routing) for variant in VARIANTS for routing in sorted(bs.ROUTINGS)]
L0 = {("dynamic", "uneven"), ("compact_signal", "many_experts_64")}


@pytest.mark.parametrize("variant, routing", bs.levels(CASES, L0, lambda variant, routing: f"{variant}-{routing}"))
def test_glu_without_column_outputs_matches_row_outputs(variant, routing, require_sm100):
    problem = bs.make_problem("glu", bs.ROUTINGS[routing], seed=3)
    options = bs.variant_options("glu", problem, variant)
    want = run_glu(problem, **options)
    assert not torch.isnan(want["c"].float()).any()
    got = run_glu(problem, col=False, **options)
    assert_equal_bytes(got, want, f"no column outputs {variant} {routing}")
