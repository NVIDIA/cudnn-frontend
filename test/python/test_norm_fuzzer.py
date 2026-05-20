"""
Norm Fuzzer - Randomized stress testing for cuDNN norm operations.

This fuzzer tests rmsnorm and layernorm with randomized:
- Problem sizes (token/batch dimension N, embedding dimension C)
- Data types (fp16, bf16, fp32)
- Epsilon values
- Optional RMSNorm bias
- Forward and backward execution

Run with:
    pytest -vv -s -rA test_norm_fuzzer.py

Environment overrides:
    CUDNN_FE_NORM_FUZZ_L0_TESTS   Number of model-like shape tests (default: 1024)
    CUDNN_FE_NORM_FUZZ_L1_TESTS   Number of irregular/edge-case shape tests (default: 4096)
    CUDNN_FE_NORM_FUZZ_L0_SEED    Seed for aligned configs
    CUDNN_FE_NORM_FUZZ_L1_SEED    Seed for irregular configs
"""

import ast
import cudnn
import math
import os
import pytest
import random
import signal
import sys
import torch
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from typing import Optional, Tuple

from sdpa.helpers import create_sparse_int_tensor, print_tensor_stats, compare_tensors

# fmt: off


def _sigint_handler(sig, frame):
    """Flush pending CUDA ops on Ctrl-C to avoid leaving the GPU in a dirty state."""
    print("\n\nInterrupted by user (Ctrl-C), exiting...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sys.exit(1)


signal.signal(signal.SIGINT, _sigint_handler)

if __name__ == "__main__":
    print("This is pytest script. Run with: pytest -vv -s -rA test_norm_fuzzer.py")
    sys.exit(0)


class NormOp(IntEnum):
    RMSNORM = 0
    LAYERNORM = 1


SUPPORTED_DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_gpu_arch() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"SM_{major * 10 + minor}"


def get_sm_count() -> int:
    props = torch.cuda.get_device_properties(0)
    return props.multi_processor_count


def get_gpu_name() -> str:
    return torch.cuda.get_device_name()


def get_available_gpu_memory_mb() -> float:
    torch.cuda.synchronize()
    free, _total = torch.cuda.mem_get_info()
    return free / (1024 * 1024)


def norm_op_name(norm_op: NormOp) -> str:
    return "rmsnorm" if norm_op == NormOp.RMSNORM else "layernorm"


def dtype_short(dtype: torch.dtype) -> str:
    return {
        torch.float16: "f16",
        torch.bfloat16: "bf16",
        torch.float32: "f32",
    }.get(dtype, "unk")


def epsilon_label(value: float) -> str:
    """Format epsilon for test IDs: 1e-05 -> '1e-5', 3.14e-05 -> '3.1e-5'."""
    s = f"{value:.1e}"
    mantissa, exp = s.split("e")
    exp_val = int(exp)
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{exp_val}"


def fill_with_garbage(tensor: torch.Tensor, nan_probability: float = 0.1) -> None:
    """Fill tensor with random non-zero values; ~10% of elements are NaN so cuDNN
    trips loudly if it reads the buffer before writing it.
    """
    if tensor.dtype in (torch.float16, torch.bfloat16):
        lo, hi = -1e4, 1e4
    else:
        lo, hi = -1e6, 1e6

    tensor.uniform_(lo, hi)

    if nan_probability > 0 and tensor.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        nan_mask = torch.rand(tensor.shape, device=tensor.device) < nan_probability
        tensor[nan_mask] = float("nan")


@dataclass
class NormConfig:
    norm_op: NormOp
    N: int
    C: int
    x_dtype: torch.dtype
    epsilon: float
    has_bias: bool
    rng_seed: int

    x_shape: Optional[Tuple[int, ...]] = None
    scale_shape: Optional[Tuple[int, ...]] = None
    bias_shape: Optional[Tuple[int, ...]] = None
    stat_shape: Optional[Tuple[int, ...]] = None
    x_strides: Optional[Tuple[int, ...]] = None
    scale_strides: Optional[Tuple[int, ...]] = None
    bias_strides: Optional[Tuple[int, ...]] = None
    stat_strides: Optional[Tuple[int, ...]] = None
    x_elems: int = 0
    scale_elems: int = 0
    bias_elems: int = 0
    stat_elems: int = 0

    def to_repro_dict(self) -> dict:
        return {
            "norm_op": int(self.norm_op),
            "N": self.N,
            "C": self.C,
            "x_dtype": str(self.x_dtype),
            "epsilon": self.epsilon,
            "has_bias": self.has_bias,
            "rng_seed": self.rng_seed,
        }


class ConfigGenerator:
    """Generate randomized, bounded norm configurations.

    L0 draws (N, C) from a curated list of common model-like sizes where C
    values are typical transformer hidden dimensions and N values are round
    token counts.  Epsilon is mostly 1e-5 or 1e-3.

    L1 independently samples N and C from pools of values that deliberately
    sit near but off common alignment boundaries (e.g. 2^k +/- 1, odd sizes).
    Epsilon spans a wider range including 1e-6 and 1e-4.
    """

    _L0_PROBLEMS = [
        (512, 768),
        (512, 1024),
        (512, 1280),
        (512, 1600),
        (1024, 2048),
        (1024, 4096),
        (512, 6144),
        (512, 8192),
        (512, 12288),
        (2048, 128),
        (4096, 128),
        (8192, 128),
        (16384, 128),
    ]
    _L1_N = [
        255, 257, 383, 511, 769, 1023, 1535, 1537,
        3071, 3073, 6143, 6145, 12287, 12289, 24575,
    ]
    _L1_C = [
        63, 65, 96, 127, 129, 160, 192, 320, 384, 640,
        768, 1023, 1025, 1280, 1536, 2048, 2560, 3072,
        4096, 5120, 6144,
    ]
    _MAX_ELEMS = 16 * 1024 * 1024  # per-test N*C cap; see estimate_memory_mb for the per-test footprint at this size
    # _L1_VALID is built via explicit for-loops, not a comprehension: Python's
    # class-scope rules don't let comprehensions see sibling class attributes
    # beyond the outermost iterable (i.e. _L1_C and _MAX_ELEMS aren't visible).
    _L1_VALID = []
    for _n in _L1_N:
        for _c in _L1_C:
            if _n * _c <= _MAX_ELEMS:
                _L1_VALID.append((_n, _c))
    del _n, _c

    def __init__(self, seed: int, allow_unaligned: bool = False):
        self.rng = random.Random(seed)
        self.allow_unaligned = allow_unaligned

    def random_problem(self) -> Tuple[int, int]:
        if not self.allow_unaligned:
            return self.rng.choice(self._L0_PROBLEMS)
        return self.rng.choice(self._L1_VALID)

    def random_op(self) -> NormOp:
        return self.rng.choice(list(NormOp))

    def random_dtype(self) -> torch.dtype:
        return self.rng.choice(SUPPORTED_DTYPES)

    def random_epsilon(self) -> float:
        if self.allow_unaligned:
            return self.rng.choice([1e-6, 1e-5, 1e-4, 1e-3])
        return self.rng.choice([1e-5, 1e-5, 1e-3])

    def random_has_bias(self, norm_op: NormOp) -> bool:
        if norm_op == NormOp.LAYERNORM:
            return True
        return self.rng.choice([False, False, False, True])

    def generate(self,
                 force_norm_op: Optional[NormOp] = None,
                 force_has_bias: Optional[bool] = None) -> NormConfig:
        norm_op = force_norm_op if force_norm_op is not None else self.random_op()
        n, c = self.random_problem()
        has_bias = force_has_bias if force_has_bias is not None else self.random_has_bias(norm_op)

        return NormConfig(
            norm_op=norm_op,
            N=n,
            C=c,
            x_dtype=self.random_dtype(),
            epsilon=self.random_epsilon(),
            has_bias=has_bias,
            rng_seed=self.rng.randint(0, 2**31 - 1),
        )


def _make_torch_generator(seed: int) -> torch.Generator:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    return gen


def _make_target_like(x: torch.Tensor, seed: int) -> torch.Tensor:
    gen = _make_torch_generator(seed)
    return create_sparse_int_tensor(tuple(x.size()), x.dtype, gen, memory_format=torch.channels_last)


def _populate_tensor_metadata(config: NormConfig, x_gpu, scale_gpu, bias_gpu):
    """Fill in shape/stride/elem metadata on config after tensors are created."""
    stat_shape = (config.N, 1, 1, 1)
    config.x_shape = tuple(x_gpu.size())
    config.scale_shape = tuple(scale_gpu.size())
    config.bias_shape = tuple(bias_gpu.size()) if bias_gpu is not None else None
    config.stat_shape = stat_shape
    config.x_strides = tuple(x_gpu.stride())
    config.scale_strides = tuple(scale_gpu.stride())
    config.bias_strides = tuple(bias_gpu.stride()) if bias_gpu is not None else None
    config.stat_strides = (1, 1, 1, 1)
    config.x_elems = x_gpu.numel()
    config.scale_elems = scale_gpu.numel()
    config.bias_elems = bias_gpu.numel() if bias_gpu is not None else 0
    config.stat_elems = math.prod(stat_shape)


def create_tensors(config: NormConfig):
    torch_rng = _make_torch_generator(config.rng_seed)
    x_shape = (config.N, config.C, 1, 1)
    scale_shape = (1, config.C, 1, 1)

    x_gpu = create_sparse_int_tensor(x_shape, config.x_dtype, torch_rng, memory_format=torch.channels_last)
    x_gpu.requires_grad_(True)

    scale_gpu = create_sparse_int_tensor(scale_shape, config.x_dtype, torch_rng, memory_format=torch.channels_last)
    scale_gpu.requires_grad_(True)

    if config.has_bias:
        bias_gpu = create_sparse_int_tensor(scale_shape, config.x_dtype, torch_rng, memory_format=torch.channels_last)
        bias_gpu.requires_grad_(True)
    else:
        bias_gpu = None

    epsilon_cpu = torch.full((1, 1, 1, 1), config.epsilon, device="cpu", dtype=torch.float32)
    target_gpu = _make_target_like(x_gpu.detach(), config.rng_seed + 1)

    _populate_tensor_metadata(config, x_gpu, scale_gpu, bias_gpu)

    return x_gpu, scale_gpu, bias_gpu, epsilon_cpu, target_gpu


def compute_reference(config: NormConfig,
                      x_gpu: torch.Tensor,
                      scale_gpu: torch.Tensor,
                      bias_gpu: Optional[torch.Tensor],
                      target_gpu: torch.Tensor):
    if config.norm_op == NormOp.RMSNORM:
        norm_x = torch.mean(x_gpu * x_gpu, dim=(1, 2, 3), keepdim=True)
        inv_var = torch.rsqrt(norm_x.float() + config.epsilon)
        y_expected = scale_gpu * (x_gpu * inv_var.to(x_gpu.dtype))
        if bias_gpu is not None:
            y_expected = y_expected + bias_gpu
        mean_expected = None
    else:
        y_expected = torch.nn.functional.layer_norm(
            x_gpu,
            [config.C, 1, 1],
            weight=scale_gpu.squeeze(0),
            bias=bias_gpu.squeeze(0),
            eps=config.epsilon,
        )
        mean_expected = x_gpu.to(torch.float32).mean(dim=(1, 2, 3), keepdim=True)
        inv_var = torch.rsqrt(torch.var(x_gpu.to(torch.float32), dim=(1, 2, 3), keepdim=True, correction=0) + config.epsilon)

    y_expected.retain_grad()
    x_gpu.retain_grad()
    scale_gpu.retain_grad()
    if bias_gpu is not None:
        bias_gpu.retain_grad()

    loss = torch.nn.functional.mse_loss(y_expected, target_gpu)
    loss.backward()

    return {
        "y": y_expected.detach(),
        "mean": mean_expected.detach() if mean_expected is not None else None,
        "inv_var": inv_var.detach(),
        "dy": y_expected.grad.detach(),
        "dx": x_gpu.grad.detach(),
        "dscale": scale_gpu.grad.detach(),
        "dbias": bias_gpu.grad.detach() if bias_gpu is not None else None,
    }


def _build_pass_by_value_tensor(graph, name: str, value_tensor: torch.Tensor):
    return graph.tensor(
        name=name,
        dim=value_tensor.size(),
        stride=value_tensor.stride(),
        is_pass_by_value=True,
        data_type=value_tensor.dtype,
    )


def run_cudnn_forward(config: NormConfig,
                      x_gpu: torch.Tensor,
                      scale_gpu: torch.Tensor,
                      bias_gpu: Optional[torch.Tensor],
                      epsilon_cpu: torch.Tensor,
                      cudnn_handle):
    try:
        stream = torch.cuda.current_stream().cuda_stream
        cudnn.set_stream(handle=cudnn_handle, stream=stream)

        graph = cudnn.pygraph(
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=cudnn_handle,
        )

        X = graph.tensor(name="X", dim=x_gpu.size(), stride=x_gpu.stride(), data_type=x_gpu.dtype)
        scale = graph.tensor(name="scale", dim=scale_gpu.size(), stride=scale_gpu.stride(), data_type=scale_gpu.dtype)
        bias = None
        if bias_gpu is not None:
            bias = graph.tensor(name="bias", dim=bias_gpu.size(), stride=bias_gpu.stride(), data_type=bias_gpu.dtype)
        epsilon = _build_pass_by_value_tensor(graph, "epsilon", epsilon_cpu)

        if config.norm_op == NormOp.RMSNORM:
            Y, inv_var = graph.rmsnorm(
                name="RMS",
                norm_forward_phase=cudnn.norm_forward_phase.TRAINING,
                input=X,
                scale=scale,
                bias=bias,
                epsilon=epsilon,
            )
            mean = None
        else:
            Y, mean, inv_var = graph.layernorm(
                name="LN",
                norm_forward_phase=cudnn.norm_forward_phase.TRAINING,
                input=X,
                scale=scale,
                bias=bias,
                epsilon=epsilon,
            )

        Y.set_output(True).set_data_type(x_gpu.dtype)
        inv_var.set_output(True).set_data_type(torch.float32)
        if mean is not None:
            mean.set_output(True).set_data_type(torch.float32)

        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans()

        y_actual = torch.empty_like(x_gpu)
        fill_with_garbage(y_actual)
        inv_var_actual = torch.empty(config.stat_shape, device="cuda", dtype=torch.float32)
        fill_with_garbage(inv_var_actual, nan_probability=0.0)
        mean_actual = None
        if mean is not None:
            mean_actual = torch.empty(config.stat_shape, device="cuda", dtype=torch.float32)
            fill_with_garbage(mean_actual, nan_probability=0.0)

        workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
        if workspace.numel() > 0:
            workspace.random_(0, 256)
            nan_mask = torch.rand(workspace.numel(), device="cuda") < 0.1
            workspace[nan_mask] = 0xFF

        variant_pack = {
            X: x_gpu.detach(),
            scale: scale_gpu.detach(),
            epsilon: epsilon_cpu,
            Y: y_actual,
            inv_var: inv_var_actual,
        }
        if bias is not None:
            variant_pack[bias] = bias_gpu.detach()
        if mean is not None:
            variant_pack[mean] = mean_actual

        graph.execute(variant_pack, workspace, handle=cudnn_handle)
        torch.cuda.synchronize()

        return True, {
            "y": y_actual,
            "mean": mean_actual,
            "inv_var": inv_var_actual,
        }
    except cudnn.cudnnGraphNotSupportedError as e:
        return False, f"Graph not supported: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def run_cudnn_backward(config: NormConfig,
                       x_gpu: torch.Tensor,
                       scale_gpu: torch.Tensor,
                       dy_gpu: torch.Tensor,
                       mean_gpu: Optional[torch.Tensor],
                       inv_var_gpu: torch.Tensor,
                       cudnn_handle):
    try:
        stream = torch.cuda.current_stream().cuda_stream
        cudnn.set_stream(handle=cudnn_handle, stream=stream)

        graph = cudnn.pygraph(
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=cudnn_handle,
        )

        DY = graph.tensor(name="DY", dim=dy_gpu.size(), stride=dy_gpu.stride(), data_type=dy_gpu.dtype)
        X = graph.tensor(name="X", dim=x_gpu.size(), stride=x_gpu.stride(), data_type=x_gpu.dtype)
        scale = graph.tensor(name="scale", dim=scale_gpu.size(), stride=scale_gpu.stride(), data_type=scale_gpu.dtype)
        inv_var = graph.tensor(name="inv_var", dim=inv_var_gpu.size(), stride=inv_var_gpu.stride(), data_type=inv_var_gpu.dtype)

        if config.norm_op == NormOp.RMSNORM:
            DX, Dscale, Dbias = graph.rmsnorm_backward(
                name="DRMS",
                grad=DY,
                input=X,
                scale=scale,
                inv_variance=inv_var,
                has_dbias=config.has_bias,
            )
            mean = None
        else:
            mean = graph.tensor(name="mean", dim=mean_gpu.size(), stride=mean_gpu.stride(), data_type=mean_gpu.dtype)
            DX, Dscale, Dbias = graph.layernorm_backward(
                name="DLN",
                grad=DY,
                input=X,
                scale=scale,
                mean=mean,
                inv_variance=inv_var,
            )

        DX.set_output(True).set_data_type(x_gpu.dtype)
        Dscale.set_output(True).set_data_type(scale_gpu.dtype)
        if Dbias is not None:
            Dbias.set_output(True).set_data_type(scale_gpu.dtype)

        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans()

        dx_actual = torch.empty_like(x_gpu)
        dscale_actual = torch.empty_like(scale_gpu)
        fill_with_garbage(dx_actual)
        fill_with_garbage(dscale_actual)
        dbias_actual = None
        if Dbias is not None:
            dbias_actual = torch.empty_like(scale_gpu)
            fill_with_garbage(dbias_actual)

        workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
        if workspace.numel() > 0:
            workspace.random_(0, 256)
            nan_mask = torch.rand(workspace.numel(), device="cuda") < 0.1
            workspace[nan_mask] = 0xFF

        variant_pack = {
            X: x_gpu.detach(),
            scale: scale_gpu.detach(),
            DY: dy_gpu.detach(),
            inv_var: inv_var_gpu.detach(),
            DX: dx_actual,
            Dscale: dscale_actual,
        }
        if mean is not None:
            variant_pack[mean] = mean_gpu.detach()
        if Dbias is not None:
            variant_pack[Dbias] = dbias_actual

        graph.execute(variant_pack, workspace, handle=cudnn_handle)
        torch.cuda.synchronize()

        return True, {
            "dx": dx_actual,
            "dscale": dscale_actual,
            "dbias": dbias_actual,
        }
    except cudnn.cudnnGraphNotSupportedError as e:
        return False, f"Graph not supported: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def _tolerances_for(name: str, config) -> Tuple[float, float]:
    """Return (rtol, atol) for a single tensor comparison.

    Default matches the cuDNN backend's flat per-dtype tolerances (bf16:
    rtol=atol=8e-3; fp16/fp32: rtol=atol=1e-3). Override for rmsnorm fp16/bf16
    Y only, where 1-ULP rounding noise in the 1/sqrt(mean(x**2)) reduction
    exceeds the flat bound at format magnitude boundaries (fp16 Y: 1e-2,
    bf16 Y: 8e-2).
    """
    if (config.norm_op == NormOp.RMSNORM
            and config.x_dtype in (torch.float16, torch.bfloat16)
            and name == "Y"):
        if config.x_dtype == torch.float16:
            return 1e-2, 1e-2
        return 8e-2, 8e-2
    if config.x_dtype == torch.bfloat16:
        return 8e-3, 8e-3
    return 1e-3, 1e-3


def compare_result(name: str, actual: torch.Tensor, expected: torch.Tensor,
                   rtol: float, atol: float, num_diffs: int) -> Tuple[bool, str]:
    passed, _num_mismatches, msg = compare_tensors(actual, expected, rtol=rtol, atol=atol, num_diffs=num_diffs)
    return passed, f"{name}: {msg}"


def estimate_memory_mb(config: NormConfig) -> float:
    elem_bytes = 2 if config.x_dtype in (torch.float16, torch.bfloat16) else 4
    x_elems = config.N * config.C
    stat_elems = config.N

    total = 0
    total += x_elems * elem_bytes * 8
    total += x_elems * 4 * 4
    total += stat_elems * 4 * 4
    total += config.C * elem_bytes * (3 if config.has_bias else 2)

    return total / (1024 * 1024)


def format_test_header(config: NormConfig, test_num: int, total_tests: int, test_name: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu_info = f"{get_gpu_arch()} ({get_sm_count()} SM-s, {get_gpu_name()})"
    checks = ["Y", "INV_VAR", "DX", "DSCALE"]
    if config.norm_op == NormOp.LAYERNORM:
        checks.insert(2, "MEAN")
    if config.has_bias:
        checks.append("DBIAS")

    lines = [
        "",
        "=" * 90,
        f"#### Test #{test_num} of {total_tests} at {timestamp}",
        "",
        f"test_name        = {test_name}",
        f"platform_info    = {gpu_info}, cudnn_ver={cudnn.backend_version()}",
        f"rng_data_seed    = {config.rng_seed}",
        f"norm_op          = {norm_op_name(config.norm_op)}",
        f"problem_dims     = [N={config.N}, C={config.C}, H=1, W=1]",
        f"norm_shape       = [C={config.C}, H=1, W=1]",
        f"x_dtype          = {config.x_dtype}",
        f"epsilon          = {config.epsilon}",
        f"has_bias         = {config.has_bias}",
        f"checks           = {', '.join(checks)}",
        f"x(N,C,1,1)       = dim={config.x_shape}, strides={config.x_strides}, elems={config.x_elems}, type={config.x_dtype}",
        f"scale(1,C,1,1)   = dim={config.scale_shape}, strides={config.scale_strides}, elems={config.scale_elems}, type={config.x_dtype}",
        f"stat(N,1,1,1)    = dim={config.stat_shape}, strides={config.stat_strides}, elems={config.stat_elems}, type=torch.float32",
    ]
    if config.has_bias:
        lines.append(
            f"bias(1,C,1,1)    = dim={config.bias_shape}, strides={config.bias_strides}, elems={config.bias_elems}, type={config.x_dtype}"
        )

    lines.extend([
        f"est_memory       = {estimate_memory_mb(config):.1f} MB",
        f"repro_cmd        = pytest -vv -s -rA {__file__}::test_repro --repro \"{config.to_repro_dict()}\"",
        " ",
    ])
    return "\n".join(lines)


@pytest.fixture
def num_diffs(request):
    return request.config.getoption("--diffs")


def tlist_with_configs(*, num_tests: int, rng_seed: int, allow_unaligned: bool = False):
    rng = random.Random(rng_seed)
    params = []
    for i in range(num_tests):
        config_seed = rng.randint(65536, 2**31 - 1)
        generator = ConfigGenerator(config_seed, allow_unaligned=allow_unaligned)
        config = generator.generate()
        params.append((i + 1, num_tests, config_seed, config))
    return params


def make_test_id(param, prefix: str = "t"):
    test_num, _total_tests, _config_seed, config = param
    op = "rms" if config.norm_op == NormOp.RMSNORM else "ln"
    bias = "b1" if config.has_bias else "b0"
    return f"{prefix}{test_num}_{op}_N{config.N}_C{config.C}_{dtype_short(config.x_dtype)}_{bias}_e{epsilon_label(config.epsilon)}"


DEFAULT_NUM_TESTS_L0 = _env_int("CUDNN_FE_NORM_FUZZ_L0_TESTS", 1024)
DEFAULT_NUM_TESTS_L1 = _env_int("CUDNN_FE_NORM_FUZZ_L1_TESTS", 4096)
DEFAULT_SEED_L0 = _env_int("CUDNN_FE_NORM_FUZZ_L0_SEED", 31415)
DEFAULT_SEED_L1 = _env_int("CUDNN_FE_NORM_FUZZ_L1_SEED", 27182)
MAX_RETRIES = 8  # outer retry budget when a generated config can't run (memory, not-supported, OOM)

TEST_PARAMS_L0 = tlist_with_configs(num_tests=DEFAULT_NUM_TESTS_L0, rng_seed=DEFAULT_SEED_L0, allow_unaligned=False)
TEST_PARAMS_L1 = tlist_with_configs(num_tests=DEFAULT_NUM_TESTS_L1, rng_seed=DEFAULT_SEED_L1, allow_unaligned=True)


def _regenerate_config(current_seed: int, attempt: int, allow_unaligned: bool,
                       norm_op: NormOp, has_bias: bool) -> Tuple[NormConfig, int]:
    new_seed = current_seed + 1000 * (attempt + 1)  # space retries apart in seed space; same-seed +1 produces near-identical configs
    generator = ConfigGenerator(new_seed, allow_unaligned=allow_unaligned)
    return generator.generate(force_norm_op=norm_op, force_has_bias=has_bias), new_seed


def _compare_all_outputs(fwd: dict, bwd: dict, ref: dict, config: NormConfig,
                         num_diffs: int) -> list:
    """Run per-tensor comparisons between cuDNN outputs and the PyTorch reference.

    Returns the list of failure messages (empty if all checks passed).
    """
    checks = [
        ("Y",       fwd["y"],       ref["y"]),
        ("INV_VAR", fwd["inv_var"], ref["inv_var"]),
    ]
    if config.norm_op == NormOp.LAYERNORM:
        checks.append(("MEAN", fwd["mean"], ref["mean"]))
    checks += [
        ("DX",     bwd["dx"],     ref["dx"]),
        ("DSCALE", bwd["dscale"], ref["dscale"]),
    ]
    if config.has_bias:
        checks.append(("DBIAS", bwd["dbias"], ref["dbias"]))

    failures = []
    for name, actual, expected in checks:
        rtol, atol = _tolerances_for(name, config)
        passed, msg = compare_result(name, actual, expected, rtol, atol, num_diffs)
        print(f"%%%% {msg}")
        if not passed:
            failures.append(msg)
    return failures


def _run_one_attempt(config: NormConfig, cudnn_handle, num_diffs: int,
                     test_name: str, test_num: int, total_tests: int) -> Tuple[str, list]:
    """Run one attempt of the norm fuzzer end-to-end.

    Returns one of:
      ("pass", [])              all comparisons passed
      ("fail", [messages])      one or more comparisons failed
      ("not_supported", [msg])  cuDNN forward or backward reported not-supported

    Caller is responsible for memory pre-check, OOM catch, retry policy, and
    tensor cleanup (via torch.cuda.empty_cache() after each attempt).
    """
    x_gpu, scale_gpu, bias_gpu, epsilon_cpu, target_gpu = create_tensors(config)
    print(format_test_header(config, test_num, total_tests, test_name))
    sys.stdout.flush()

    ref = compute_reference(config, x_gpu, scale_gpu, bias_gpu, target_gpu)

    success, fwd = run_cudnn_forward(config, x_gpu, scale_gpu, bias_gpu, epsilon_cpu, cudnn_handle)
    if not success:
        print(f"%%%% Forward execution failed: {fwd}")
        return ("not_supported", [f"forward: {fwd}"])

    success, bwd = run_cudnn_backward(config, x_gpu, scale_gpu, ref["dy"], fwd["mean"], fwd["inv_var"], cudnn_handle)
    if not success:
        print(f"%%%% Backward execution failed: {bwd}")
        return ("not_supported", [f"backward: {bwd}"])

    # Hash + zero/NaN/Inf summary of Y; pytest includes this captured
    # stdout in failure output for triage. Only Y because it's the
    # cleanest signal that the kernel produced meaningful values.
    print_tensor_stats(fwd["y"], tag="Y_gpu")

    failures = _compare_all_outputs(fwd, bwd, ref, config, num_diffs)
    if failures:
        print("@@@@ Overall result: FAILED, numerical mismatch!")
        return ("fail", failures)

    print("@@@@ Overall result: PASSED, everything looks good!")
    return ("pass", [])


def run_norm_test_with_retry(config: NormConfig, config_seed: int, test_num: int, total_tests: int,
                             test_name_prefix: str, cudnn_handle, num_diffs: int,
                             allow_unaligned: bool = False, allow_retries: bool = True) -> bool:
    current_config = config
    current_seed = config_seed
    memory_threshold = 0.50  # OOM safety net; pre-skip if estimated footprint > half of free memory

    for attempt in range(MAX_RETRIES + 1):
        try:
            available_mb = get_available_gpu_memory_mb()
            estimated_mb = estimate_memory_mb(current_config)
            if estimated_mb > available_mb * memory_threshold:
                print(f"%%%% Memory check: need ~{estimated_mb:.1f}MB, available {available_mb:.1f}MB (threshold {memory_threshold*100:.0f}%)")
                if allow_retries and attempt < MAX_RETRIES:
                    current_config, current_seed = _regenerate_config(
                        current_seed, attempt, allow_unaligned, current_config.norm_op, current_config.has_bias
                    )
                    continue
                pytest.skip(f"Insufficient GPU memory: need ~{estimated_mb:.1f}MB, available {available_mb:.1f}MB")

            test_name = f"{test_name_prefix}[{make_test_id((test_num, total_tests, current_seed, current_config))}]"
            if attempt > 0:
                test_name += f" (retry {attempt})"

            outcome, info = _run_one_attempt(
                current_config, cudnn_handle, num_diffs, test_name, test_num, total_tests
            )

            if outcome == "pass":
                return True
            if outcome == "fail":
                pytest.fail("\n".join(info))
            # outcome == "not_supported"
            if allow_retries and attempt < MAX_RETRIES:
                current_config, current_seed = _regenerate_config(
                    current_seed, attempt, allow_unaligned, current_config.norm_op, current_config.has_bias
                )
                continue
            if allow_retries:
                pytest.skip(f"cuDNN not supported after {MAX_RETRIES} retries: {info}")
            pytest.fail(f"cuDNN failure for repro config: {info}")

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("%%%% CUDA out of memory!")
            if allow_retries and attempt < MAX_RETRIES:
                current_config, current_seed = _regenerate_config(
                    current_seed, attempt, allow_unaligned, current_config.norm_op, current_config.has_bias
                )
                continue
            if allow_retries:
                pytest.skip(f"CUDA out of memory after {MAX_RETRIES} retries")
            pytest.fail("CUDA out of memory while running repro config")
        finally:
            torch.cuda.empty_cache()

    pytest.skip(f"Failed to find supported configuration after {MAX_RETRIES} retries")
    return False


@pytest.mark.L0
@pytest.mark.parametrize("test_num,total_tests,config_seed,config", TEST_PARAMS_L0,
                         ids=[make_test_id(p) for p in TEST_PARAMS_L0])
def test_norm_random_L0(test_num: int, total_tests: int, config_seed: int, config: NormConfig,
                        cudnn_handle, num_diffs):
    if cudnn_handle is None:
        pytest.skip("cuDNN handle not available")

    run_norm_test_with_retry(
        config=config,
        config_seed=config_seed,
        test_num=test_num,
        total_tests=total_tests,
        test_name_prefix="test_norm_random_L0",
        cudnn_handle=cudnn_handle,
        num_diffs=num_diffs,
        allow_unaligned=False,
    )


@pytest.mark.L1
@pytest.mark.parametrize("test_num,total_tests,config_seed,config", TEST_PARAMS_L1,
                         ids=[make_test_id(p, prefix="u") for p in TEST_PARAMS_L1])
def test_norm_random_L1(test_num: int, total_tests: int, config_seed: int, config: NormConfig,
                        cudnn_handle, num_diffs):
    if cudnn_handle is None:
        pytest.skip("cuDNN handle not available")

    run_norm_test_with_retry(
        config=config,
        config_seed=config_seed,
        test_num=test_num,
        total_tests=total_tests,
        test_name_prefix="test_norm_random_L1",
        cudnn_handle=cudnn_handle,
        num_diffs=num_diffs,
        allow_unaligned=True,
    )


@pytest.mark.L0
def test_repro(cudnn_handle, num_diffs, request):
    if cudnn_handle is None:
        pytest.skip("cuDNN handle not available")

    repro_str = request.config.getoption("--repro")
    if repro_str is None:
        pytest.skip("No --repro argument provided")

    repro = ast.literal_eval(repro_str)
    dtype_map = {
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.float32": torch.float32,
    }

    config = NormConfig(
        norm_op=NormOp(repro["norm_op"]),
        N=repro["N"],
        C=repro["C"],
        x_dtype=dtype_map[repro["x_dtype"]],
        epsilon=float(repro["epsilon"]),
        has_bias=bool(repro["has_bias"]),
        rng_seed=repro["rng_seed"],
    )

    run_norm_test_with_retry(
        config=config,
        config_seed=config.rng_seed,
        test_num=1,
        total_tests=1,
        test_name_prefix="test_repro",
        cudnn_handle=cudnn_handle,
        num_diffs=num_diffs,
        allow_unaligned=False,
        allow_retries=False,
    )
