"""Tests for the FE-OSS RMSNorm + RHT + amax API."""

import math

import pytest
import torch

from test_utils import torch_fork_set_rng

SUPPORTED_N_NUM_THREADS = [
    (2048, 128),
    (4096, 256),
    (7168, 128),
    (8192, 512),
    (16384, 1024),
    (32768, 512),
]


def _hadamard_matrix(n: int, *, device: torch.device) -> torch.Tensor:
    matrix = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while matrix.shape[0] < n:
        top = torch.cat((matrix, matrix), dim=1)
        bottom = torch.cat((matrix, -matrix), dim=1)
        matrix = torch.cat((top, bottom), dim=0)
    return matrix


def _rmsnorm_rht_amax_ref(x: torch.Tensor, w: torch.Tensor, eps: float, rows_per_cta: int):
    m, n = x.shape
    x_f32 = x.float()
    rms = torch.sqrt((x_f32 * x_f32).mean(dim=-1, keepdim=True) + eps)
    y = x_f32 / rms * w.float().unsqueeze(0)

    had_block = 16
    hadamard = _hadamard_matrix(had_block, device=x.device) / math.sqrt(had_block)
    y = y.view(m, n // had_block, had_block)
    y = torch.matmul(y, hadamard).view(m, n)

    num_ctas = m // rows_per_cta
    amax = y.abs().view(num_ctas, rows_per_cta, n).amax(dim=(1, 2))
    return y.to(torch.bfloat16), amax.to(torch.float32)


def _make_inputs(*, m: int, n: int):
    x = torch.randn((m, n), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((n,), dtype=torch.bfloat16, device="cuda")
    return x.contiguous(), w.contiguous()


def _assert_ref_close(x, w, o, amax, *, eps: float, rows_per_cta: int, skip_ref: bool = False):
    if skip_ref:
        return
    o_ref, amax_ref = _rmsnorm_rht_amax_ref(x, w, eps, rows_per_cta)
    torch.testing.assert_close(o.float().cpu(), o_ref.float().cpu(), atol=4e-2, rtol=1e-2)
    torch.testing.assert_close(amax.cpu(), amax_ref.cpu(), atol=2e-3, rtol=1e-3)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("n,num_threads", SUPPORTED_N_NUM_THREADS)
def test_rmsnorm_rht_amax_compile_execute(n, num_threads, request):
    try:
        from cudnn import RmsNormRhtAmaxSm100
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    skip_ref = request.config.getoption("--skip-ref", default=False)
    eps = 1e-5
    m = 256
    rows_per_cta = 2
    x, w = _make_inputs(m=m, n=n)
    o = torch.empty_like(x)
    amax = torch.full((m // rows_per_cta,), float("-inf"), dtype=torch.float32, device="cuda")

    api = RmsNormRhtAmaxSm100(
        sample_x=x,
        sample_w=w,
        sample_o=o,
        sample_amax=amax,
        eps=eps,
        num_threads=num_threads,
        rows_per_cta=rows_per_cta,
    )

    try:
        assert api.check_support(), "Unsupported testcase"
    except (ValueError, RuntimeError) as exc:
        pytest.skip(f"Unsupported testcase: {exc}")

    api.compile()
    api.execute(x_tensor=x, w_tensor=w, o_tensor=o, amax_tensor=amax)
    _assert_ref_close(x, w, o, amax, eps=eps, rows_per_cta=rows_per_cta, skip_ref=skip_ref)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("n,num_threads", SUPPORTED_N_NUM_THREADS)
@pytest.mark.parametrize("rows_per_cta", [2, 4, 8])
def test_rmsnorm_rht_amax_wrapper(n, num_threads, rows_per_cta, request):
    try:
        from cudnn import rmsnorm_rht_amax_wrapper_sm100
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    skip_ref = request.config.getoption("--skip-ref", default=False)
    eps = 1e-5
    m = 256
    x, w = _make_inputs(m=m, n=n)

    try:
        outputs = rmsnorm_rht_amax_wrapper_sm100(
            x_tensor=x,
            w_tensor=w,
            eps=eps,
            num_threads=num_threads,
            rows_per_cta=rows_per_cta,
        )
    except (ValueError, RuntimeError) as exc:
        pytest.skip(f"Unsupported testcase: {exc}")

    assert outputs["o_tensor"].shape == (m, n)
    assert outputs["amax_tensor"].shape == (m // rows_per_cta,)
    _assert_ref_close(x, w, outputs["o_tensor"], outputs["amax_tensor"], eps=eps, rows_per_cta=rows_per_cta, skip_ref=skip_ref)
