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
    assert api.eps == eps
    assert api.requested_num_threads == num_threads
    assert api.requested_rows_per_cta == rows_per_cta

    try:
        assert api.check_support(), "Unsupported testcase"
    except (ValueError, RuntimeError) as exc:
        pytest.skip(f"Unsupported testcase: {exc}")

    api.compile()
    result = api.execute(x_tensor=x, w_tensor=w, o_tensor=o, amax_tensor=amax)
    assert result is None
    _assert_ref_close(x, w, o, amax, eps=eps, rows_per_cta=rows_per_cta, skip_ref=skip_ref)


@pytest.mark.L0
def test_rmsnorm_rht_amax_class_rejects_invalid_output_exemplars(monkeypatch):
    try:
        from cudnn import RmsNormRhtAmaxSm100
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    sample_x = torch.empty((4, 256), dtype=torch.bfloat16)
    sample_w = torch.empty((256,), dtype=torch.bfloat16)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (10, 0))

    cases = (
        (torch.empty((2, 256), dtype=torch.bfloat16), torch.empty((2,), dtype=torch.float32), "O must have shape"),
        (torch.empty((4, 256), dtype=torch.float32), torch.empty((2,), dtype=torch.float32), "O must have dtype"),
        (torch.empty_strided((4, 256), (1, 4), dtype=torch.bfloat16), torch.empty((2,), dtype=torch.float32), "O must be row-major contiguous"),
        (torch.empty((4, 256), dtype=torch.bfloat16), torch.empty((1,), dtype=torch.float32), "Amax must have shape"),
        (torch.empty((4, 256), dtype=torch.bfloat16), torch.empty((2,), dtype=torch.bfloat16), "Amax must have dtype"),
    )
    for sample_o, sample_amax, message in cases:
        with pytest.raises(ValueError, match=message):
            RmsNormRhtAmaxSm100(
                sample_x,
                sample_w,
                sample_o,
                sample_amax,
                num_threads=32,
                rows_per_cta=2,
            ).check_support()

    strided_amax = torch.empty_strided((2,), (2,), dtype=torch.float32)
    assert RmsNormRhtAmaxSm100(
        sample_x,
        sample_w,
        torch.empty_like(sample_x),
        strided_amax,
        num_threads=32,
        rows_per_cta=2,
    ).check_support()


@pytest.mark.L0
def test_rmsnorm_rht_amax_execute_checks_dynamic_amax_shape():
    try:
        from cudnn import RmsNormRhtAmaxSm100
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    x = torch.empty((4, 256), dtype=torch.bfloat16)
    w = torch.empty((256,), dtype=torch.bfloat16)
    o = torch.empty_like(x)
    amax = torch.empty((2,), dtype=torch.float32)
    api = RmsNormRhtAmaxSm100(x, w, o, amax, num_threads=32, rows_per_cta=2)
    api._op.check_support()
    api._compiled_kernel = lambda **_kwargs: None

    with pytest.raises(ValueError, match="Amax tensor shape mismatch"):
        api.execute(x, w, o, torch.empty((1,), dtype=torch.float32))

    dynamic_x = torch.empty((8, 256), dtype=torch.bfloat16)
    dynamic_o = torch.empty_like(dynamic_x)
    dynamic_amax = torch.empty((4,), dtype=torch.float32)
    assert api.execute(dynamic_x, w, dynamic_o, dynamic_amax, current_stream=object()) is None


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
def test_rmsnorm_rht_amax_wrapper_reuses_compiled_kernel(monkeypatch):
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

    first_m, second_m, n = 256, 512, 2048
    rows_per_cta = 2
    first_x, w = _make_inputs(m=first_m, n=n)
    second_x, _ = _make_inputs(m=second_m, n=n)
    try:
        try:
            first = rmsnorm_api.rmsnorm_rht_amax_wrapper_sm100(first_x, w, num_threads=128, rows_per_cta=rows_per_cta)
        except (ValueError, RuntimeError) as exc:
            pytest.skip(f"Unsupported testcase: {exc}")

        second = rmsnorm_api.rmsnorm_rht_amax_wrapper_sm100(second_x, w, num_threads=128, rows_per_cta=rows_per_cta)
        _assert_ref_close(second_x, w, second["o_tensor"], second["amax_tensor"], eps=1e-5, rows_per_cta=rows_per_cta)

        assert construction_count == 1
        assert first["o_tensor"].shape == (first_m, n)
        assert second["o_tensor"].shape == (second_m, n)
        assert first["amax_tensor"].shape == (first_m // rows_per_cta,)
        assert second["amax_tensor"].shape == (second_m // rows_per_cta,)
    finally:
        rmsnorm_api._cache_of_RmsNormRhtAmaxSm100Objects.clear()


@pytest.mark.L0
def test_rmsnorm_rht_amax_wrapper_passes_allocated_outputs_to_class(monkeypatch):
    try:
        from cudnn.rmsnorm_rht_amax import api as rmsnorm_api
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    constructed = {}

    class FakeApi:
        def __init__(self, **kwargs):
            constructed.update(kwargs)

        def check_support(self):
            return True

        def compile(self):
            pass

        def execute(self, **kwargs):
            constructed["execute"] = kwargs

    monkeypatch.setattr(rmsnorm_api, "RmsNormRhtAmaxSm100", FakeApi)
    rmsnorm_api._cache_of_RmsNormRhtAmaxSm100Objects.clear()

    x = torch.empty((4, 256), dtype=torch.bfloat16)
    w = torch.empty((256,), dtype=torch.bfloat16)
    try:
        result = rmsnorm_api.rmsnorm_rht_amax_wrapper_sm100(x, w, num_threads=32, rows_per_cta=4)
    finally:
        rmsnorm_api._cache_of_RmsNormRhtAmaxSm100Objects.clear()

    assert set(constructed) == {"sample_x", "sample_w", "sample_o", "sample_amax", "eps", "num_threads", "rows_per_cta", "execute"}
    assert constructed["sample_o"] is result["o_tensor"]
    assert constructed["sample_amax"] is result["amax_tensor"]
    assert torch.isneginf(result["amax_tensor"]).all()
    assert constructed["execute"] == {
        "x_tensor": x,
        "w_tensor": w,
        "o_tensor": result["o_tensor"],
        "amax_tensor": result["amax_tensor"],
        "current_stream": None,
    }
