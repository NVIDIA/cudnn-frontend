"""
Utilities and parameterization for GEMM SwiGLU tests.
Contains test configuration fixtures, tensor creation, and reference implementations.
"""

import torch
import pytest


GEMM_SWIGLU_PARAM_MARKS = [
    pytest.mark.parametrize("a_major", ["k", "m"]),
    pytest.mark.parametrize("b_major", ["k", "n"]),
    pytest.mark.parametrize("c_major", ["m", "n"]),
    pytest.mark.parametrize(
        "ab_dtype",
        [
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ],
    ),
    pytest.mark.parametrize("c_dtype", [torch.float16, torch.bfloat16, torch.float32]),
    pytest.mark.parametrize(
        "acc_dtype", [torch.float32]
    ),  # Note: float16 accumulator is supported but disabled in testing
    pytest.mark.parametrize("glu_dtype", [torch.float16, torch.bfloat16]),
    pytest.mark.parametrize("use_2cta_instrs", [True, False]),
    pytest.mark.parametrize(
        "mma_tiler_mn", [(128, 128), (128, 64), (256, 256), (256, 128)]
    ),
    pytest.mark.parametrize("cluster_shape_mn", [(1, 1), (2, 2), (4, 4)]),
]


def with_gemm_swiglu_params(func):
    for mark in reversed(GEMM_SWIGLU_PARAM_MARKS):
        func = mark(func)
    return func


def gemm_swiglu_init(
    request,
    a_major,
    b_major,
    c_major,
    ab_dtype,
    c_dtype,
    acc_dtype,
    glu_dtype,
    use_2cta_instrs,
    mma_tiler_mn,
    cluster_shape_mn,
):
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip(
            f"Environment not supported: requires compute capability >= 10, found {major}"
        )

    mnkl_str = request.config.getoption("--gemm-swiglu-mnkl", default=None)
    mma_tiler_str = request.config.getoption("--gemm-swiglu-mma-tiler", default=None)
    cluster_shape_str = request.config.getoption(
        "--gemm-swiglu-cluster-shape", default=None
    )
    alpha_opt = request.config.getoption("--gemm-swiglu-alpha", default=None)
    skip_ref = request.config.getoption("--gemm-swiglu-skip-ref", default=False)

    if mnkl_str is not None:
        m, n, k, l = [int(x.strip()) for x in mnkl_str.split(",")]
    else:
        m, n, k, l = 256, 256, 512, 2

    if mma_tiler_str is not None:
        mma_tiler_mn = tuple(int(x.strip()) for x in mma_tiler_str.split(","))
    if cluster_shape_str is not None:
        cluster_shape_mn = tuple(int(x.strip()) for x in cluster_shape_str.split(","))

    alpha = float(alpha_opt) if alpha_opt is not None else 1.0

    return {
        "m": m,
        "n": n,
        "k": k,
        "l": l,
        "mma_tiler_mn": mma_tiler_mn,
        "cluster_shape_mn": cluster_shape_mn,
        "alpha": alpha,
        "skip_ref": skip_ref,
        "a_major": a_major,
        "b_major": b_major,
        "c_major": c_major,
        "ab_dtype": ab_dtype,
        "c_dtype": c_dtype,
        "acc_dtype": acc_dtype,
        "glu_dtype": glu_dtype,
        "use_2cta_instrs": use_2cta_instrs,
    }


# Create and permute tensor A/B/C
def create_and_permute_tensor(
    l, mode0, mode1, is_mode0_major, dtype, is_dynamic_layout=True
):
    # is_mode0_major: (l, mode1, mode0) -> (mode0, mode1, l)
    # else: (l, mode0, mode1) -> (mode0, mode1, l)
    shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
    permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
    is_unsigned = dtype in {torch.uint8}
    min_val = 0 if is_unsigned else -2
    max_val = 4 if is_unsigned else 2

    # Generate random values according to dtype support
    if dtype in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}:
        # randint upper bound is exclusive
        torch_tensor = torch.randint(
            int(min_val), int(max_val), shape, dtype=dtype, device="cuda"
        ).permute(permute_order)
    elif dtype in {torch.float8_e4m3fn, torch.float8_e5m2}:
        torch_tensor = (
            torch.empty(shape, dtype=torch.float32, device="cuda")
            .uniform_(float(min_val), float(max_val))
            .to(dtype)
            .permute(permute_order)
        )
    else:
        try:
            torch_tensor = (
                torch.empty(shape, dtype=dtype, device="cuda")
                .uniform_(float(min_val), float(max_val))
                .permute(permute_order)
            )
        except NotImplementedError:
            # Fallback: generate in float32 and cast
            torch_tensor = (
                torch.empty(shape, dtype=torch.float32, device="cuda")
                .uniform_(float(min_val), float(max_val))
                .to(dtype)
                .permute(permute_order)
            )

    return torch_tensor


def run_gemm_swiglu_ref(a_ref, b_ref, alpha):
    c_ref, glu_ref = None, None
    if a_ref.dtype in {torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2}:
        c_ref = alpha * torch.einsum("mkl,nkl->mnl", (a_ref).cpu(), (b_ref).cpu())
    else:
        c_ref = (alpha * torch.einsum("mkl,nkl->mnl", (a_ref), (b_ref))).cpu()

    group = 32
    n = b_ref.shape[0]
    assert n % group == 0, "N must be divisible by 32 for GLU block grouping"
    num_blocks = n // group
    assert (
        num_blocks % 2 == 0
    ), "Number of 32-col blocks must be even (pairs of input/gate)"

    cols = torch.arange(n, device=c_ref.device, dtype=torch.long)
    block_cols = cols.view(num_blocks, group)
    input_idx = block_cols[0::2].reshape(-1)
    gate_idx = block_cols[1::2].reshape(-1)
    glu_ref = c_ref.index_select(1, input_idx) * (
        c_ref.index_select(1, gate_idx) * torch.sigmoid(c_ref.index_select(1, gate_idx))
    )
    glu_ref = glu_ref.to(torch.float32)

    return c_ref, glu_ref


def check_ref_gemm_swiglu(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    glu: torch.Tensor,
    alpha: float = 1.0,
    skip_ref: bool = False,
):
    if not skip_ref:
        a_ref = a.clone().to(torch.float32)
        b_ref = b.clone().to(torch.float32)
        c_ref, glu_ref = run_gemm_swiglu_ref(a_ref, b_ref, alpha)

        is_c_fp8 = c.dtype in {torch.float8_e4m3fn, torch.float8_e5m2}
        if is_c_fp8:
            torch.testing.assert_close(
                c.cpu().to(torch.float32), c_ref.to(torch.float32), atol=0.1, rtol=0.1
            )
        else:
            torch.testing.assert_close(
                c.cpu(), c_ref.to(c.dtype), atol=0.01, rtol=9e-03
            )

        is_glu_fp8 = glu.dtype in {torch.float8_e4m3fn, torch.float8_e5m2}
        if is_glu_fp8:
            torch.testing.assert_close(
                glu.cpu().to(torch.float32),
                glu_ref.to(torch.float32),
                atol=0.1,
                rtol=0.1,
            )
        else:
            torch.testing.assert_close(
                glu.cpu(), glu_ref.to(glu.dtype), atol=0.01, rtol=9e-03
            )
    else:
        print("Skipping reference check")


def allocate_input_tensors(m, n, k, l, ab_dtype, a_major, b_major):
    a_tensor = create_and_permute_tensor(
        l, m, k, a_major == "m", ab_dtype, is_dynamic_layout=True
    )
    b_tensor = create_and_permute_tensor(
        l, n, k, b_major == "n", ab_dtype, is_dynamic_layout=True
    )

    return a_tensor, b_tensor


def allocate_output_tensors(m, n, l, c_dtype, glu_dtype, c_major):
    c_tensor = create_and_permute_tensor(
        l, m, n, c_major == "m", c_dtype, is_dynamic_layout=True
    )
    glu_tensor = create_and_permute_tensor(
        l, m, n // 2, c_major == "m", glu_dtype, is_dynamic_layout=True
    )

    return c_tensor, glu_tensor
