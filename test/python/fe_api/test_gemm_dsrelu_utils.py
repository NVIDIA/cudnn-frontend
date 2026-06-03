import pytest
import torch

from test_fe_api_utils import (
    compute_reference_amax,
    create_and_permute_tensor,
    create_scale_factor_tensor,
)

GEMM_DSRELU_PARAM_MARKS_FP4 = [
    pytest.mark.parametrize("a_major", ["k"]),
    pytest.mark.parametrize("b_major", ["k"]),
    pytest.mark.parametrize("c_major", ["n"]),
    pytest.mark.parametrize("ab_dtype", [torch.float4_e2m1fn_x2]),
    pytest.mark.parametrize("c_dtype", [torch.bfloat16]),
    pytest.mark.parametrize("d_dtype", [torch.bfloat16]),
    pytest.mark.parametrize("acc_dtype", [torch.float32]),
    pytest.mark.parametrize("mma_tiler_mn", [(256, 256), (128, 256)]),
    pytest.mark.parametrize("cluster_shape_mn", [(2, 1), (1, 1)]),
    pytest.mark.parametrize("sf_vec_size", [16]),
    pytest.mark.parametrize("sf_dtype", [torch.float8_e8m0fnu]),
    pytest.mark.parametrize("vector_f32", [True, False]),
]


def with_gemm_dsrelu_params_fp4(func):
    for mark in reversed(GEMM_DSRELU_PARAM_MARKS_FP4):
        func = mark(func)
    return func


def gemm_dsrelu_init(
    request,
    a_major,
    b_major,
    c_major,
    ab_dtype,
    c_dtype,
    d_dtype,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    vector_f32,
):
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        pytest.skip(f"Environment not supported: requires compute capability >= 10, found {major}")

    mnkl_str = request.config.getoption("--gemm-dsrelu-mnkl", default=None)
    if mnkl_str is not None:
        m, n, k, l = [int(x.strip()) for x in mnkl_str.split(",")]
    else:
        m, n, k, l = 256, 256, 512, 2

    return {
        "m": m,
        "n": n,
        "k": k,
        "l": l,
        "a_major": a_major,
        "b_major": b_major,
        "c_major": c_major,
        "ab_dtype": ab_dtype,
        "c_dtype": c_dtype,
        "d_dtype": d_dtype,
        "acc_dtype": acc_dtype,
        "mma_tiler_mn": mma_tiler_mn,
        "cluster_shape_mn": cluster_shape_mn,
        "sf_vec_size": sf_vec_size,
        "sf_dtype": sf_dtype,
        "vector_f32": vector_f32,
        "alpha": 1.0,
        "skip_ref": request.config.getoption("--skip-ref", default=False),
    }


def allocate_gemm_dsrelu_tensors(cfg):
    a_ref, a_tensor = create_and_permute_tensor(cfg["l"], cfg["m"], cfg["k"], cfg["a_major"] == "m", cfg["ab_dtype"])
    b_ref, b_tensor = create_and_permute_tensor(cfg["l"], cfg["n"], cfg["k"], cfg["b_major"] == "n", cfg["ab_dtype"])
    c_ref, c_tensor = create_and_permute_tensor(cfg["l"], cfg["m"], cfg["n"], cfg["c_major"] == "m", cfg["c_dtype"])
    sfa_ref, sfa_tensor = create_scale_factor_tensor(cfg["l"], cfg["m"], cfg["k"], cfg["sf_vec_size"], cfg["sf_dtype"])
    sfb_ref, sfb_tensor = create_scale_factor_tensor(cfg["l"], cfg["n"], cfg["k"], cfg["sf_vec_size"], cfg["sf_dtype"])
    prob_ref, prob_tensor = create_and_permute_tensor(cfg["l"], cfg["m"], 1, cfg["a_major"] == "m", torch.float32)

    return {
        "a_ref": a_ref,
        "a_tensor": a_tensor,
        "b_ref": b_ref,
        "b_tensor": b_tensor,
        "c_ref": c_ref,
        "c_tensor": c_tensor,
        "sfa_ref": sfa_ref,
        "sfa_tensor": sfa_tensor,
        "sfb_ref": sfb_ref,
        "sfb_tensor": sfb_tensor,
        "prob_ref": prob_ref,
        "prob_tensor": prob_tensor,
    }


def allocate_gemm_dsrelu_outputs(cfg):
    _, d_tensor = create_and_permute_tensor(cfg["l"], cfg["m"], cfg["n"], cfg["c_major"] == "m", cfg["d_dtype"])
    dprob_tensor = torch.zeros((cfg["m"], 1, cfg["l"]), dtype=torch.float32, device="cuda")

    sfd_tensor = None
    if cfg["d_dtype"] in {torch.float8_e4m3fn, torch.float8_e5m2}:
        _, sfd_tensor = create_scale_factor_tensor(cfg["l"], cfg["m"], cfg["n"], cfg["sf_vec_size"], cfg["sf_dtype"])

    amax_tensor = None
    if cfg["ab_dtype"] in {torch.float4_e2m1fn_x2, torch.uint8} and cfg["d_dtype"] in {torch.bfloat16, torch.float16, torch.float32}:
        amax_tensor = torch.full((1,), float("-inf"), dtype=torch.float32, device="cuda")

    norm_const_tensor = None
    if sfd_tensor is not None:
        norm_const_tensor = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    return {
        "d_tensor": d_tensor,
        "dprob_tensor": dprob_tensor,
        "sfd_tensor": sfd_tensor,
        "amax_tensor": amax_tensor,
        "norm_const_tensor": norm_const_tensor,
    }


def gemm_dsrelu_reference(inputs, cfg):
    res_a = torch.einsum("mkl,mkl->mkl", inputs["a_ref"], inputs["sfa_ref"])
    res_b = torch.einsum("nkl,nkl->nkl", inputs["b_ref"], inputs["sfb_ref"])
    x_ref = cfg["alpha"] * torch.einsum("mkl,nkl->mnl", res_a, res_b)
    d_ref = inputs["c_ref"].float() * inputs["prob_ref"].expand(-1, cfg["n"], -1).float() * 2 * torch.relu(x_ref)
    dprob_ref = torch.sum(torch.relu(x_ref) ** 2 * inputs["c_ref"].float(), dim=1, keepdim=True)
    return d_ref, dprob_ref


def check_ref_gemm_dsrelu(inputs, outputs, cfg, check_d=True):
    if cfg["skip_ref"]:
        return

    d_ref, dprob_ref = gemm_dsrelu_reference(inputs, cfg)
    torch.testing.assert_close(outputs["dprob_tensor"].float(), dprob_ref.float(), atol=0.12, rtol=0.02)

    if check_d:
        torch.testing.assert_close(outputs["d_tensor"].float(), d_ref.float(), atol=0.12, rtol=0.02)
        if outputs["amax_tensor"] is not None:
            amax_ref = torch.tensor(
                [compute_reference_amax(d_ref)],
                dtype=torch.float32,
                device=outputs["amax_tensor"].device,
            )
            torch.testing.assert_close(outputs["amax_tensor"], amax_ref, atol=0.12, rtol=0.02)
