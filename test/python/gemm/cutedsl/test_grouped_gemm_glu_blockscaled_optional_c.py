# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The MXFP8 GLU kernel (SM100) without its pre-activation output C.

c None: no C store and no C shared memory stages (the smem goes to A/B stages).  D and its SFD, with and without the
column pair, are byte-equal to the kernel that also writes C, on the stock dynamic scheduler and on the input signal
scheduler (TMA SFA atoms and compact SFA); also through GroupedGemmGluSm100 with ``sample_c=None`` over repeated
launches.
"""

import pytest
import torch

from gemm.cutedsl import _streaming_signal_blockscaled as bs
from gemm.cutedsl._streaming_signal_blockscaled import require_sm10x  # noqa: F401  (autouse fixture)
from gemm.cutedsl.test_grouped_gemm_swiglu_utils import allocate_grouped_gemm_input_tensors, allocate_grouped_gemm_output_tensors
from gemm.cutedsl.test_grouped_gemm_glu_blockscaled_column_outputs import NAN_SCALE, assert_equal_bytes, require_sm100, run_glu  # noqa: F401

#: (scheduler variant, routing, column pair written)
VARIANTS = ("dynamic", "signal", "compact_signal")
CASES = [(variant, routing, col) for variant in VARIANTS for routing in sorted(bs.ROUTINGS) for col in (True, False)]
L0 = {
    ("dynamic", "uneven", True),
    ("dynamic", "uneven", False),
    ("signal", "many_experts_64", False),
    ("compact_signal", "many_experts_64", True),
    ("compact_signal", "small_experts_64", False),
}


@pytest.mark.parametrize(
    "variant, routing, col",
    bs.levels(CASES, L0, lambda variant, routing, col: f"{variant}-{routing}-{'col' if col else 'nocol'}"),
)
def test_glu_without_c_matches_with_c(variant, routing, col, require_sm100):
    problem = bs.make_problem("glu", bs.ROUTINGS[routing], seed=5)
    options = bs.variant_options("glu", problem, variant)
    want = run_glu(problem, col=col, **options)
    assert not torch.isnan(want["c"].float()).any()
    got = run_glu(problem, col=col, c=False, **options)
    assert "c" not in got
    want.pop("c")
    assert_equal_bytes(got, want, f"no C {variant} {routing} col={col}")
    assert not (got["sfd_row"] == NAN_SCALE).all(), "the row SFD was written"


#: Launches without C after the reference launch with C (the TMA-store buffer reuse check, as the Rubin twin's).
NO_C_LAUNCHES = 16


@pytest.mark.L0
def test_glu_api_without_c_matches_with_c(require_sm100):
    """GroupedGemmGluSm100 with ``sample_c=None``: every launch writes the same D, D_col and both SFD planes as the
    launch that also writes C.

    Without C, nothing but the D store's own producer_acquire keeps a later subtile from overwriting sD while an earlier
    bulk store still reads it; with C the C store's acquire would hide a missing one.  The shape is the Rubin twin's
    (32 experts, K = 7168, N = 4096 gate + up, 2-CTA 256 x 256 tiles, dynamic scheduler), launched NO_C_LAUNCHES times.
    """
    from cudnn import GroupedGemmGluSm100
    from cuda.bindings import driver as cuda

    n, k, experts = 4096, 7168, 32
    ab_dtype, sf_dtype, sf_vec_size = torch.float8_e4m3fn, torch.float8_e8m0fnu, 32
    torch.manual_seed(0)
    inputs = allocate_grouped_gemm_input_tensors(
        n=n,
        k=k,
        l=experts,
        group_m_list=[2048] * experts,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        sf_vec_size=sf_vec_size,
        m_aligned=256,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    checked = ("d_tensor", "d_col_tensor", "sfd_row_tensor", "sfd_col_tensor")
    golden = None
    for with_c in (True, False):
        outputs = allocate_grouped_gemm_output_tensors(
            tensor_m=inputs["tensor_m"],
            n=n,
            l=experts,
            ab_dtype=ab_dtype,
            c_dtype=torch.bfloat16,
            d_dtype=torch.float8_e4m3fn,
            cd_major="n",
            sf_dtype=sf_dtype,
            sf_vec_size=sf_vec_size,
        )
        c_tensor = outputs["c_tensor"] if with_c else None
        api = GroupedGemmGluSm100(
            sample_a=inputs["a_tensor"],
            sample_c=c_tensor,
            sample_d=outputs["d_tensor"],
            sample_sfa=inputs["sfa_tensor"],
            sample_padded_offsets=inputs["padded_offsets_tensor"],
            sample_alpha=inputs["alpha_tensor"],
            sample_d_col=outputs["d_col_tensor"],
            sample_b=inputs["b_tensor"],
            sample_sfb=inputs["sfb_tensor"],
            sample_sfd_row=outputs["sfd_row_tensor"],
            sample_sfd_col=outputs["sfd_col_tensor"],
            sample_norm_const=inputs.get("norm_const_tensor"),
            sample_prob=inputs.get("prob_tensor"),
            acc_dtype=torch.float32,
            mma_tiler_mn=(256, 256),
            cluster_shape_mn=(2, 1),
            sf_vec_size=sf_vec_size,
            vector_f32=True,
            m_aligned=256,
            discrete_col_sfd=False,
            act_func="swiglu",
            use_dynamic_sched=True,
        )
        assert api.check_support()
        api.compile()
        for launch in range(1 if with_c else NO_C_LAUNCHES):
            for name in checked:
                outputs[name].view(torch.uint8).fill_(0xFF)
            api.execute(
                a_tensor=inputs["a_tensor"],
                c_tensor=c_tensor,
                d_tensor=outputs["d_tensor"],
                sfa_tensor=inputs["sfa_tensor"],
                padded_offsets=inputs["padded_offsets_tensor"],
                alpha_tensor=inputs["alpha_tensor"],
                b_tensor=inputs["b_tensor"],
                sfb_tensor=inputs["sfb_tensor"],
                d_col_tensor=outputs["d_col_tensor"],
                sfd_row_tensor=outputs["sfd_row_tensor"],
                sfd_col_tensor=outputs["sfd_col_tensor"],
                norm_const_tensor=inputs.get("norm_const_tensor"),
                prob_tensor=inputs.get("prob_tensor"),
                current_stream=stream,
            )
            torch.cuda.synchronize()
            if golden is None:
                assert not torch.isnan(outputs["c_tensor"].float()).any()
                golden = {name: outputs[name].view(torch.uint8).clone() for name in checked}
                continue
            for name in checked:
                differing = (outputs[name].view(torch.uint8) != golden[name]).sum().item()
                assert differing == 0, f"launch {launch} without C: {differing} bytes of {name} differ from the launch with C"
