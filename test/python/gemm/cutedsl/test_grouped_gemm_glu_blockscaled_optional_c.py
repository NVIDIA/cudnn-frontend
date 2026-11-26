# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The MXFP8 GLU kernel (SM100) without its pre-activation output C.

c None: no C store and no C shared memory stages (the smem goes to A/B stages).  D and its row and column SFD (E4M3 D)
or the BF16 D are byte-equal to the kernel that also writes C, on the dynamic and static schedulers with 2-CTA and
1-CTA tiles; also through GroupedGemmGluSm100 with ``sample_c=None`` over repeated launches.

The problem and kernel helpers here are shared with test_grouped_gemm_dglu_blockscaled_h.py.
"""

import pytest
import torch

from gemm.cutedsl.test_grouped_gemm_swiglu_utils import allocate_grouped_gemm_input_tensors, allocate_grouped_gemm_output_tensors

ROWS_ALIGNMENT = 256
SF_VEC_SIZE = 32
#: Scale-factor atom: 128 rows x 4 scales, 512 bytes (tcgen05 block-scaled MMA layout).
SF_ATOM_COLS = 4
HIDDEN = 512
#: Rows per expert: empty and one-row experts, more than 32 experts, many small experts, one expert of many M tiles.
ROUTINGS = {
    "uneven": [700, 0, 256, 1],
    "many_experts_64": [(0, 1, 255, 256, 257, 700, 1300)[i % 7] for i in range(64)],
    "small_experts_64": [(0, 1, 256, 300, 512, 600, 768)[i % 7] for i in range(64)],
    "big_expert": [256, 5000, 0, 513, 1],
}
#: E4M3 NaN codes and E8M0 NaN scales: an output the kernel should write but does not stays poisoned.
NAN_CODE, NAN_SCALE = 0x7F, 0xFF


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("the SM100 block-scaled kernels")
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

    if cutedsl_too_old(cutedsl_state()[1]):
        pytest.skip("CuTe DSL older than the library floor")


def levels(cases, l0, ids):
    """``cases`` as pytest params: L0 for those in ``l0``, L1 for the rest."""
    return [pytest.param(*case, marks=pytest.mark.L0 if case in l0 else pytest.mark.L1, id=ids(*case)) for case in cases]


def padded(tokens):
    return -(-tokens // ROWS_ALIGNMENT) * ROWS_ALIGNMENT


def swizzle_sf(scales):
    """Row-major scales [rows, cols] (rows % 128 == 0, cols % 4 == 0) -> the flat SF-atom bytes the kernel reads:
    128-row blocks outermost, then 4-column atoms, each atom 32 x (4 x 4) bytes."""
    rows, cols = scales.shape
    return scales.view(rows // 128, 4, 32, cols // 4, 4).permute(0, 3, 2, 1, 4).contiguous().view(-1)


def mxfp8_codes(rows, k, generator):
    """E4M3 codes [rows, k] and their row-major E8M0 exponents (2^-3 .. 2^3; padding columns 2^0)."""
    codes = (torch.randn(rows, k, generator=generator, device="cuda") * 2).to(torch.float8_e4m3fn)
    cols = -(-k // (SF_VEC_SIZE * SF_ATOM_COLS)) * SF_ATOM_COLS
    exponents = torch.full((rows, cols), 127, dtype=torch.uint8, device="cuda")
    exponents[:, : k // SF_VEC_SIZE] = torch.randint(124, 131, (rows, k // SF_VEC_SIZE), generator=generator, device="cuda", dtype=torch.uint8)
    return codes, exponents


def make_problem(tokens_per_expert, n, k=HIDDEN, seed=0, dglu=False):
    """Expert-major rows padded to ROWS_ALIGNMENT (padding rows hold data too): A [rows, k], B [experts, n, k] with
    their SF atoms, inclusive row ends ``offsets``, alpha, prob [rows, 1, 1]; with ``dglu`` the saved C [rows, 2n] and
    beta [experts]."""
    experts = len(tokens_per_expert)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    rows = max(sum(padded(tokens) for tokens in tokens_per_expert), ROWS_ALIGNMENT)
    a, a_exp = mxfp8_codes(rows, k, generator)
    b, b_exp = mxfp8_codes(experts * n, k, generator)
    b_exp = b_exp.view(experts, n, -1)
    offsets = torch.tensor([padded(tokens) for tokens in tokens_per_expert], device="cuda", dtype=torch.int64).cumsum(0).to(torch.int32)
    problem = dict(
        a=a,
        sfa=swizzle_sf(a_exp),
        b=b.view(experts, n, k),
        sfb=torch.cat([swizzle_sf(b_exp[e]) for e in range(experts)]),
        offsets=offsets,
        alpha=torch.rand(experts, generator=generator, device="cuda") + 0.5,
        prob=torch.rand(rows, 1, 1, generator=generator, device="cuda"),
        rows=rows,
        n=n,
        k=k,
    )
    if dglu:
        generator = torch.Generator(device="cuda").manual_seed(seed + 1000)
        problem["c"] = (torch.randn(rows, 2 * n, generator=generator, device="cuda") * 2).to(torch.bfloat16)
        problem["beta"] = torch.rand(experts, generator=generator, device="cuda") + 0.5
    return problem


def valid_rows(problem):
    return int(problem["offsets"][-1].item())


def make_kernel(kind, experts, one_cta=False, **options):
    """The SM100 MXFP8 GLU ("glu") or dSwiGLU ("dglu") kernel, dense weights, cluster (2, 1) with the 2-CTA 256 x 256
    tile or (1, 1) with the 1-CTA 128 x 256 tile; ``options`` add or override construction arguments."""
    import cutlass
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode

    args = dict(
        sf_vec_size=SF_VEC_SIZE,
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=not one_cta,
        mma_tiler_mn=(128, 256) if one_cta else (256, 256),
        cluster_shape_mn=(1, 1) if one_cta else (2, 1),
        vectorized_f32=True,
        discrete_col_sfd=False,
        expert_cnt=experts,
        weight_mode=MoEWeightMode.DENSE,
    )
    if kind == "glu":
        from cudnn.gemm.cutedsl.grouped.glu.moe_blockscaled_grouped_gemm_glu_bias import BlockScaledMoEGroupedGemmGluBiasKernel as cls

        args.update(generate_sfd=True, act_func="swiglu")
    else:
        from cudnn.gemm.cutedsl.grouped.dglu.moe_blockscaled_grouped_gemm_dglu_dbias import BlockScaledMoEGroupedGemmDgluDbiasKernel as cls

        args.update(act_func="dswiglu", deterministic=True)
    args.update(options)
    kernel = cls(**args)
    kernel.test_cluster_size = 1 if one_cta else 2
    kernel.test_compile_key = (kind, tuple(sorted((name, repr(value)) for name, value in args.items())))
    return kernel


def cu_stream():
    import cuda.bindings.driver as cuda

    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


#: Index of the compile-time ``b_major_mode`` among the kernels' arguments (not passed to a compiled kernel).
B_MAJOR_ARG = 6
COMPILED = {}


def launch(kernel, args, key, **kwargs):
    """Compile ``kernel`` once per ``key`` in this process, launch it and wait."""
    import cutlass.cute as cute
    from cutlass.utils import HardwareInfo

    if key not in COMPILED:
        clusters = HardwareInfo().get_max_active_clusters(kernel.test_cluster_size)
        COMPILED[key] = cute.compile(kernel, *args, clusters, cu_stream(), **kwargs)
    COMPILED[key](*[x for i, x in enumerate(args) if i != B_MAJOR_ARG], cu_stream(), **kwargs)
    torch.cuda.synchronize()


def run_glu(problem, c=True, fp8_d=True, **options):
    """One GLU launch with E4M3 D and both SFD planes (BF16 D without SFD unless ``fp8_d``), and C unless ``c`` is
    False; returns the outputs by name, D and C over the experts' rows."""
    import cutlass
    from cutlass.cute.nvgpu import OperandMajorMode
    from cutlass.cute.runtime import from_dlpack

    one_cta = options.pop("one_cta", False)
    kernel = make_kernel("glu", len(problem["offsets"]), one_cta=one_cta, generate_sfd=fp8_d, **options)
    rows, n, width = problem["rows"], problem["n"], problem["n"] // 2

    def tensor(x):
        return from_dlpack(x, assumed_align=16)

    def e4m3(x):
        return tensor(x.view(torch.float8_e4m3fn))

    def e8m0(x):
        return tensor(x.view(torch.float8_e8m0fnu))

    workspace = torch.zeros(max(kernel.get_workspace_bytes(), 1), device="cuda", dtype=torch.uint8)
    out = dict(c=torch.full((rows, n), float("nan"), device="cuda", dtype=torch.bfloat16))
    if fp8_d:
        out.update({name: torch.full((rows, width), NAN_CODE, device="cuda", dtype=torch.uint8) for name in ("d", "d_col")})
        out.update({name: torch.full((rows * width // SF_VEC_SIZE,), NAN_SCALE, device="cuda", dtype=torch.uint8) for name in ("sfd_row", "sfd_col")})
        d, d_col, sfd_row, sfd_col = e4m3(out["d"]), e4m3(out["d_col"]), e8m0(out["sfd_row"]), e8m0(out["sfd_col"])
        norm_const = tensor(torch.ones(1, device="cuda", dtype=torch.float32))
    else:
        out["d"] = torch.full((rows, width), float("nan"), device="cuda", dtype=torch.bfloat16)
        d = d_col = tensor(out["d"])
        sfd_row = sfd_col = norm_const = None
    args = [
        tensor(problem["a"]),
        tensor(problem["b"]),
        e8m0(problem["sfb"]),
        cutlass.Int32(n),
        cutlass.Int32(problem["k"]),
        cutlass.Int64(problem["k"]),
        OperandMajorMode.K,
        from_dlpack(workspace, assumed_align=128).iterator,
        # (c, d, d_col, sfa, sfd_row, sfd_col, amax, norm_const, offsets, alpha, prob, bias)
        tensor(out["c"]) if c else None,
        d,
        d_col,
        e8m0(problem["sfa"]),
        sfd_row,
        sfd_col,
        None,
        norm_const,
        from_dlpack(problem["offsets"], assumed_align=4),
        tensor(problem["alpha"]),
        tensor(problem["prob"]),
        None,
    ]
    launch(kernel, args, (kernel.test_compile_key, c, rows, n))
    valid = valid_rows(problem)
    if not c:
        out.pop("c")
    return {name: value[:valid] if value.dim() == 2 else value for name, value in out.items()}


def assert_equal_bytes(got, want, label):
    for name in want:
        a, b = got[name].view(torch.uint8), want[name].view(torch.uint8)
        assert torch.equal(a, b), f"{label}: {(a != b).sum().item()} bytes of {name} differ"


#: Scheduler variants: the dynamic scheduler with 2-CTA and 1-CTA tiles, the static scheduler.
VARIANTS = {
    "dynamic": dict(use_dynamic_sched=True),
    "dynamic_1cta": dict(use_dynamic_sched=True, one_cta=True),
    "static": dict(use_dynamic_sched=False),
}
CASES = [(variant, routing) for variant in VARIANTS for routing in sorted(ROUTINGS)]
L0 = {
    ("dynamic", "uneven"),
    ("dynamic", "small_experts_64"),
    ("dynamic_1cta", "big_expert"),
    ("static", "many_experts_64"),
}


@pytest.mark.parametrize("variant, routing", levels(CASES, L0, lambda variant, routing: f"{variant}-{routing}"))
def test_glu_without_c_matches_with_c(variant, routing):
    problem = make_problem(ROUTINGS[routing], n=1024, seed=5)
    want = run_glu(problem, **VARIANTS[variant])
    assert not torch.isnan(want.pop("c").float()).any()
    got = run_glu(problem, c=False, **VARIANTS[variant])
    assert "c" not in got
    assert_equal_bytes(got, want, f"no C {variant} {routing}")
    assert not (got["sfd_row"] == NAN_SCALE).all(), "the row SFD was written"


@pytest.mark.L0
@pytest.mark.parametrize("routing", ["uneven", "many_experts_64"])
def test_glu_bf16_d_without_c_matches_with_c(routing):
    problem = make_problem(ROUTINGS[routing], n=1024, seed=6)
    want = run_glu(problem, fp8_d=False, use_dynamic_sched=True)
    assert not torch.isnan(want.pop("c").float()).any()
    got = run_glu(problem, c=False, fp8_d=False, use_dynamic_sched=True)
    assert not torch.isnan(got["d"].float()).any(), "every D row was written"
    assert_equal_bytes(got, want, f"no C, BF16 D {routing}")


#: Launches without C after the reference launch with C (the TMA-store buffer reuse check, as the Rubin twin's).
NO_C_LAUNCHES = 16


@pytest.mark.L0
def test_glu_api_without_c_matches_with_c():
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
        for launch_idx in range(1 if with_c else NO_C_LAUNCHES):
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
                assert differing == 0, f"launch {launch_idx} without C: {differing} bytes of {name} differ from the launch with C"
