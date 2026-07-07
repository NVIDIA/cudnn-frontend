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
def test_rmsnorm_rht_amax_class_call(n, num_threads, request):
    try:
        from cudnn import RmsNormRhtAmaxSm100
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    skip_ref = request.config.getoption("--skip-ref", default=False)
    eps = 1e-5
    m = 256
    rows_per_cta = 2
    x, w = _make_inputs(m=m, n=n)

    api = RmsNormRhtAmaxSm100(
        sample_x=x,
        sample_w=w,
        eps=eps,
        num_threads=num_threads,
        rows_per_cta=rows_per_cta,
    )
    assert api.eps == eps
    assert api.requested_num_threads == num_threads
    assert api.requested_rows_per_cta == rows_per_cta

    try:
        assert api.check_support(), "Unsupported testcase"
    except (ValueError, RuntimeError) as exc:
        pytest.skip(f"Unsupported testcase: {exc}")

    outputs = api(x, w)
    o, amax = outputs
    assert outputs["o_tensor"] is o
    assert outputs["amax_tensor"] is amax
    _assert_ref_close(x, w, o, amax, eps=eps, rows_per_cta=rows_per_cta, skip_ref=skip_ref)


@pytest.mark.L0
def test_rmsnorm_rht_amax_class_rejects_runtime_signature_mismatch(monkeypatch):
    try:
        from cudnn import RmsNormRhtAmaxSm100
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    sample_x = torch.empty((4, 256), dtype=torch.bfloat16)
    sample_w = torch.empty((256,), dtype=torch.bfloat16)
    api = RmsNormRhtAmaxSm100(sample_x, sample_w, num_threads=32, rows_per_cta=2)
    api._compiled_kernel = lambda **_kwargs: None
    monkeypatch.setattr(api._kernel, "infer_output", lambda: pytest.fail("output inference must follow runtime signature validation"))

    with pytest.raises(ValueError, match="sample_x tensor shape mismatch"):
        api.execute(torch.empty((2, 256), dtype=torch.bfloat16), sample_w)
    with pytest.raises(ValueError, match="sample_w dtype mismatch"):
        api.execute(sample_x, torch.empty((256,), dtype=torch.float32))


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


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_rmsnorm_rht_amax_wrapper_reuses_owned_kernel(monkeypatch):
    try:
        from cudnn.rmsnorm_rht_amax import api as rmsnorm_api
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    construction_count = 0
    original_init = rmsnorm_api.RMSNormRHTAmaxKernel.__init__

    def counted_init(self, *args, **kwargs):
        nonlocal construction_count
        construction_count += 1
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(rmsnorm_api.RMSNormRHTAmaxKernel, "__init__", counted_init)
    rmsnorm_api._cache_of_RmsNormRhtAmaxSm100Objects.clear()

    m, n = 256, 2048
    rows_per_cta = 2
    x, w = _make_inputs(m=m, n=n)
    try:
        first = rmsnorm_api.rmsnorm_rht_amax_wrapper_sm100(x, w, num_threads=128, rows_per_cta=rows_per_cta)
        second = rmsnorm_api.rmsnorm_rht_amax_wrapper_sm100(x, w, num_threads=128, rows_per_cta=rows_per_cta)
    except (ValueError, RuntimeError) as exc:
        pytest.skip(f"Unsupported testcase: {exc}")
    finally:
        rmsnorm_api._cache_of_RmsNormRhtAmaxSm100Objects.clear()

    assert construction_count == 1
    assert first["o_tensor"].shape == second["o_tensor"].shape == (m, n)
    assert first["amax_tensor"].shape == second["amax_tensor"].shape == (m // rows_per_cta,)


@pytest.mark.L0
def test_rmsnorm_rht_amax_wrapper_delegates_output_allocation_to_class(monkeypatch):
    try:
        from cudnn.rmsnorm_rht_amax import api as rmsnorm_api
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    constructed = {}
    output = torch.empty((3, 5), dtype=torch.bfloat16)
    amax = torch.full((7,), -2.0, dtype=torch.float32)

    class FakeApi:
        def __init__(self, **kwargs):
            constructed.update(kwargs)

        def check_support(self):
            return True

        def compile(self):
            pass

        def execute(self, **kwargs):
            constructed["execute"] = kwargs
            return rmsnorm_api.TupleDict(o_tensor=output, amax_tensor=amax)

    monkeypatch.setattr(rmsnorm_api, "RmsNormRhtAmaxSm100", FakeApi)
    rmsnorm_api._cache_of_RmsNormRhtAmaxSm100Objects.clear()

    x = torch.empty((4, 16), dtype=torch.bfloat16)
    w = torch.empty((16,), dtype=torch.bfloat16)
    try:
        result = rmsnorm_api.rmsnorm_rht_amax_wrapper_sm100(x, w, rows_per_cta=4)
    finally:
        rmsnorm_api._cache_of_RmsNormRhtAmaxSm100Objects.clear()

    assert set(constructed) == {"sample_x", "sample_w", "eps", "num_threads", "rows_per_cta", "execute"}
    assert result["o_tensor"] is output
    assert result["amax_tensor"] is amax
    assert torch.all(result["amax_tensor"] == -2.0)
    assert constructed["execute"] == {"x_tensor": x, "w_tensor": w, "current_stream": None}
