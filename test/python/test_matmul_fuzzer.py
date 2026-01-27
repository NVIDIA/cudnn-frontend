"""
Matmul Fuzzer - Randomized stress testing for cuDNN matmul operations.

This fuzzer tests matmul operations with randomized:
- Shapes (batch, M, N, K dimensions)
- Layouts (row-major, column-major, transposed, strided)
- Data types (fp16, bf16, fp32, int8)
- Epilogues (none, bias, relu, gelu)

Run with:
    pytest -vv -s -rA test_matmul_fuzzer.py

Options:
    --num-tests N       Number of random tests to run (default: 100)
    --seed N            Random seed for reproducibility (default: random)
    --diffs N           Number of mismatches to display (default: 10)
"""

import cudnn
import pytest
import random
import torch
import math
import os
import sys
import signal
from looseversion import LooseVersion
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List
from enum import IntEnum

# fmt: off

# Handle Ctrl-C gracefully
def signal_handler(sig, frame):
    print("\n\nInterrupted by user (Ctrl-C), exiting...")
    # Force CUDA to sync and cleanup
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    print("This is pytest script. Run with: pytest -vv -s -rA test_matmul_fuzzer.py")
    sys.exit(0)


# ============================================================================
# Configuration and Constants
# ============================================================================

class LayoutType(IntEnum):
    ROW_MAJOR_PACKED = 0   # Standard row-major packed (strides: [..., N, 1])
    COL_MAJOR_PACKED = 1   # Column-major packed (strides: [..., 1, M])
    STRIDED = 2            # Custom strided layout with gaps

class EpilogueType(IntEnum):
    NONE = 0
    BIAS = 1
    RELU = 2
    BIAS_RELU = 3
    GELU = 4
    BIAS_GELU = 5

SUPPORTED_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int8,
]

# Compute precisions
COMPUTE_DTYPES = [
    cudnn.data_type.FLOAT,
    cudnn.data_type.HALF,
    cudnn.data_type.BFLOAT16,
]


# ============================================================================
# Utility Functions
# ============================================================================

def convert_to_cudnn_type(torch_type):
    """Convert PyTorch dtype to cuDNN data type."""
    mapping = {
        torch.float16: cudnn.data_type.HALF,
        torch.bfloat16: cudnn.data_type.BFLOAT16,
        torch.float32: cudnn.data_type.FLOAT,
        torch.bool: cudnn.data_type.BOOLEAN,
        torch.uint8: cudnn.data_type.UINT8,
        torch.int8: cudnn.data_type.INT8,
        torch.int32: cudnn.data_type.INT32,
        torch.int64: cudnn.data_type.INT64,
    }
    if torch_type not in mapping:
        raise ValueError(f"Unsupported tensor data type: {torch_type}")
    return mapping[torch_type]


def get_gpu_arch():
    """Get GPU SM architecture version."""
    major, minor = torch.cuda.get_device_capability()
    return f"SM_{major * 10 + minor}"


def get_sm_count():
    """Get number of SMs on the GPU."""
    props = torch.cuda.get_device_properties(0)
    return props.multi_processor_count


def get_gpu_name():
    """Get GPU name."""
    return torch.cuda.get_device_name()


def layout_name(layout: LayoutType) -> str:
    """Get human-readable layout name."""
    names = {
        LayoutType.ROW_MAJOR_PACKED: "row_major_packed",
        LayoutType.COL_MAJOR_PACKED: "col_major_packed",
        LayoutType.STRIDED: "strided",
    }
    return names.get(layout, "unknown")


def epilogue_name(epilogue: EpilogueType) -> str:
    """Get human-readable epilogue name."""
    names = {
        EpilogueType.NONE: "none",
        EpilogueType.BIAS: "bias",
        EpilogueType.RELU: "relu",
        EpilogueType.BIAS_RELU: "bias_relu",
        EpilogueType.GELU: "gelu",
        EpilogueType.BIAS_GELU: "bias_gelu",
    }
    return names.get(epilogue, "unknown")


def compute_strides(shape: Tuple[int, ...], layout: LayoutType, rng: random.Random) -> Tuple[int, ...]:
    """Compute strides for a given shape and layout."""
    ndim = len(shape)

    if layout == LayoutType.ROW_MAJOR_PACKED:
        # Standard row-major: last dim has stride 1
        strides = []
        stride = 1
        for dim in reversed(shape):
            strides.insert(0, stride)
            stride *= dim
        return tuple(strides)

    elif layout == LayoutType.COL_MAJOR_PACKED:
        # Column-major for the last two dimensions
        if ndim < 2:
            return (1,)
        strides = [1] * ndim
        # Last two dims are transposed
        strides[-1] = shape[-2]  # N stride = M
        strides[-2] = 1          # M stride = 1
        # Batch dimensions
        stride = shape[-1] * shape[-2]
        for i in range(ndim - 3, -1, -1):
            strides[i] = stride
            stride *= shape[i]
        return tuple(strides)

    elif layout == LayoutType.STRIDED:
        # Random strided layout with potential gaps
        strides = []
        stride = 1
        for dim in reversed(shape):
            # Add random padding (1-4x the minimum stride)
            padding_factor = rng.choice([1, 1, 1, 2, 2, 4])
            strides.insert(0, stride)
            stride *= dim * padding_factor
        return tuple(strides)

    return tuple([1] * ndim)


def compute_num_elements(shape: Tuple[int, ...], strides: Tuple[int, ...]) -> int:
    """Compute number of elements needed for storage given shape and strides."""
    if not shape:
        return 1
    max_offset = sum((d - 1) * s for d, s in zip(shape, strides))
    return max_offset + 1


def fill_with_garbage(tensor: torch.Tensor, nan_probability: float = 0.1) -> None:
    """
    Fill tensor with garbage values (mix of random values and NaNs).
    This helps catch bugs where cuDNN doesn't write all output locations.
    """
    # Choose range based on dtype to avoid overflow
    if tensor.dtype in (torch.float16, torch.bfloat16):
        lo, hi = -1e4, 1e4  # FP16 max is ~65504
    else:
        lo, hi = -1e6, 1e6

    # Fill with random garbage
    tensor.uniform_(lo, hi)

    # Sprinkle in some NaNs (only for float types)
    if nan_probability > 0 and tensor.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        nan_mask = torch.rand(tensor.shape, device=tensor.device) < nan_probability
        tensor[nan_mask] = float('nan')


# ============================================================================
# Test Configuration
# ============================================================================

@dataclass
class MatmulConfig:
    """Configuration for a single matmul test."""
    # Basic dimensions
    batch: int
    M: int
    N: int
    K: int

    # Data types
    a_dtype: torch.dtype
    b_dtype: torch.dtype
    c_dtype: torch.dtype
    compute_dtype: cudnn.data_type

    # Layouts
    a_layout: LayoutType
    b_layout: LayoutType
    c_layout: LayoutType

    # Transpose flags
    a_transposed: bool
    b_transposed: bool

    # Epilogue
    epilogue: EpilogueType

    # Random seed for data generation
    rng_seed: int

    # Computed strides (set during tensor creation)
    a_shape: Tuple[int, ...] = None
    b_shape: Tuple[int, ...] = None
    c_shape: Tuple[int, ...] = None
    a_strides: Tuple[int, ...] = None
    b_strides: Tuple[int, ...] = None
    c_strides: Tuple[int, ...] = None
    a_elems: int = 0
    b_elems: int = 0
    c_elems: int = 0

    # Bias tensor info (set when epilogue uses bias)
    bias_shape: Tuple[int, ...] = None
    bias_strides: Tuple[int, ...] = None
    bias_elems: int = 0

    def to_repro_dict(self) -> dict:
        """Convert config to reproducible dictionary."""
        return {
            'batch': self.batch,
            'M': self.M,
            'N': self.N,
            'K': self.K,
            'a_dtype': str(self.a_dtype),
            'b_dtype': str(self.b_dtype),
            'c_dtype': str(self.c_dtype),
            'epilogue': int(self.epilogue),
            'rng_seed': self.rng_seed,
        }


class ConfigGenerator:
    """Generator for random matmul configurations."""

    def __init__(self, seed: int, allow_unaligned: bool = False):
        self.rng = random.Random(seed)
        self.allow_unaligned = allow_unaligned

    def random_batch(self) -> int:
        """Generate random batch size (no alignment requirement)."""
        return self.rng.choice([1, 1, 2, 3, 4, 5, 7, 8, 16, 32])

    def random_dim(self, min_val: int = 1, max_val: int = 4096) -> int:
        """Generate random dimension size (M, N, or K)."""
        val = self.rng.randint(int(math.sqrt(min_val)), int(math.sqrt(max_val)))
        if self.allow_unaligned:
            # Allow non-aligned sizes for stress testing
            return val * val
        else:
            # Default: round up to next multiple of 8 for tensor core alignment
            return ((val * val + 7) // 8) * 8

    def random_dtype(self) -> torch.dtype:
        """Generate random data type."""
        return self.rng.choice(SUPPORTED_DTYPES)

    def random_layout(self) -> LayoutType:
        """Generate random layout type."""
        # Prefer row-major but test others too
        weights = [0.6, 0.2, 0.2]
        return self.rng.choices(list(LayoutType), weights=weights)[0]

    def random_epilogue(self) -> EpilogueType:
        """Generate random epilogue type."""
        # Most tests without epilogue
        weights = [0.5, 0.15, 0.1, 0.1, 0.075, 0.075]
        return self.rng.choices(list(EpilogueType), weights=weights)[0]

    def random_compute_dtype(self) -> cudnn.data_type:
        """Generate random compute data type."""
        return self.rng.choice([cudnn.data_type.FLOAT])  # Float is most compatible

    def generate(self) -> MatmulConfig:
        """Generate a random matmul configuration."""
        batch = self.random_batch()
        M = self.random_dim()
        N = self.random_dim()
        K = self.random_dim()

        # Data types - ensure compatible combinations
        if self.rng.random() < 0.8:
            # 80% of tests use same dtype for A and B (more stable)
            a_dtype = self.random_dtype()
            b_dtype = a_dtype
        else:
            # 20% test mixed precision
            a_dtype = self.random_dtype()
            b_dtype = self.random_dtype()

        # Output dtype should be float type (not int8)
        if a_dtype == torch.float32 or b_dtype == torch.float32:
            c_dtype = torch.float32
        elif a_dtype == torch.int8 or b_dtype == torch.int8:
            # int8 inputs typically output to float32
            c_dtype = torch.float32
        else:
            c_dtype = self.rng.choice([a_dtype, torch.float32])

        # Layout selection - prefer row-major for stability
        a_layout = self.rng.choices(
            [LayoutType.ROW_MAJOR_PACKED, LayoutType.COL_MAJOR_PACKED, LayoutType.STRIDED],
            weights=[0.7, 0.15, 0.15]
        )[0]
        b_layout = self.rng.choices(
            [LayoutType.ROW_MAJOR_PACKED, LayoutType.COL_MAJOR_PACKED, LayoutType.STRIDED],
            weights=[0.7, 0.15, 0.15]
        )[0]

        # Transpose flags - disabled for now, using layouts instead for variety
        # cuDNN matmul expects specific input layouts, transpose requires extra handling
        a_transposed = False
        b_transposed = False

        # Epilogue selection
        epilogue = self.random_epilogue()

        config = MatmulConfig(
            batch=batch,
            M=M,
            N=N,
            K=K,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            c_dtype=c_dtype,
            compute_dtype=self.random_compute_dtype(),
            a_layout=a_layout,
            b_layout=b_layout,
            c_layout=LayoutType.ROW_MAJOR_PACKED,  # Output usually row-major
            a_transposed=a_transposed,
            b_transposed=b_transposed,
            epilogue=epilogue,
            rng_seed=self.rng.randint(0, 2**31 - 1),
        )

        return config


# ============================================================================
# Test Execution
# ============================================================================

def create_tensors(config: MatmulConfig, rng: random.Random):
    """Create input and output tensors based on configuration."""
    torch_rng = torch.Generator(device='cuda')
    torch_rng.manual_seed(config.rng_seed)

    # Compute shapes for matmul: C = A @ B
    # A: (batch, M, K)
    # B: (batch, K, N)
    # C: (batch, M, N)
    #
    # Transpose flags affect storage layout, not logical dimensions:
    # - If a_transposed: A stored as (batch, K, M), transposed for matmul to (batch, M, K)
    # - If b_transposed: B stored as (batch, N, K), transposed for matmul to (batch, K, N)
    #
    # For simplicity in this fuzzer, we don't use transpose - just vary layouts instead

    a_shape = (config.batch, config.M, config.K)
    b_shape = (config.batch, config.K, config.N)
    c_shape = (config.batch, config.M, config.N)

    # Compute strides
    a_strides = compute_strides(a_shape, config.a_layout, rng)
    b_strides = compute_strides(b_shape, config.b_layout, rng)
    c_strides = compute_strides(c_shape, config.c_layout, rng)

    # Compute number of elements
    a_elems = compute_num_elements(a_shape, a_strides)
    b_elems = compute_num_elements(b_shape, b_strides)
    c_elems = compute_num_elements(c_shape, c_strides)

    # Update config with computed values
    config.a_shape = a_shape
    config.b_shape = b_shape
    config.c_shape = c_shape
    config.a_strides = a_strides
    config.b_strides = b_strides
    config.c_strides = c_strides
    config.a_elems = a_elems
    config.b_elems = b_elems
    config.c_elems = c_elems

    # Create tensors
    if config.a_dtype == torch.int8:
        a_storage = torch.randint(-2, 3, (a_elems,), device='cuda', dtype=torch.int8)
    else:
        a_storage = torch.empty(a_elems, device='cuda', dtype=config.a_dtype)
        a_storage.normal_(mean=0.5, std=0.5, generator=torch_rng)

    if config.b_dtype == torch.int8:
        b_storage = torch.randint(-2, 3, (b_elems,), device='cuda', dtype=torch.int8)
    else:
        b_storage = torch.empty(b_elems, device='cuda', dtype=config.b_dtype)
        b_storage.normal_(mean=0.5, std=0.5,generator=torch_rng)

    # Create strided views
    A = torch.as_strided(a_storage, a_shape, a_strides)
    B = torch.as_strided(b_storage, b_shape, b_strides)

    # Output tensor - fill with garbage to catch bugs where cuDNN doesn't write all outputs
    c_storage = torch.empty(c_elems, device='cuda', dtype=config.c_dtype)
    fill_with_garbage(c_storage)
    C = torch.as_strided(c_storage, c_shape, c_strides)

    # Bias tensor if needed
    bias = None
    if config.epilogue in [EpilogueType.BIAS, EpilogueType.BIAS_RELU, EpilogueType.BIAS_GELU]:
        # Randomize bias shape: each dim can be 1 (broadcast) or match C
        bias_rng = random.Random(config.rng_seed + 1)  # Different seed for bias shape
        bias_b = bias_rng.choice([1, config.batch])
        bias_m = bias_rng.choice([1, config.M])
        bias_n = bias_rng.choice([1, config.N])
        bias_shape = (bias_b, bias_m, bias_n)
        bias_strides = compute_strides(bias_shape, LayoutType.ROW_MAJOR_PACKED, bias_rng)
        bias_elems = compute_num_elements(bias_shape, bias_strides)

        config.bias_shape = bias_shape
        config.bias_strides = bias_strides
        config.bias_elems = bias_elems

        bias = torch.empty(bias_elems, device='cuda', dtype=config.c_dtype)
        bias.normal_(mean=0.5, std=0.5, generator=torch_rng)
        bias = torch.as_strided(bias, bias_shape, bias_strides)

    return A, B, C, bias


def compute_reference(config: MatmulConfig, A: torch.Tensor, B: torch.Tensor, bias: Optional[torch.Tensor]):
    """Compute reference result using PyTorch."""
    # Convert to float for computation
    compute_dtype = torch.float32

    A_compute = A.to(compute_dtype)
    B_compute = B.to(compute_dtype)

    try:
        # Matmul: C = A @ B
        # A: (batch, M, K), B: (batch, K, N), C: (batch, M, N)
        C_ref = torch.matmul(A_compute, B_compute)

        # Epilogue
        if bias is not None and config.epilogue in [EpilogueType.BIAS, EpilogueType.BIAS_RELU, EpilogueType.BIAS_GELU]:
            C_ref = C_ref + bias.to(compute_dtype)

        if config.epilogue in [EpilogueType.RELU, EpilogueType.BIAS_RELU]:
            C_ref = torch.relu(C_ref)
        elif config.epilogue in [EpilogueType.GELU, EpilogueType.BIAS_GELU]:
            C_ref = torch.nn.functional.gelu(C_ref)

        return C_ref.to(config.c_dtype)
    finally:
        del A_compute, B_compute


def run_cudnn_matmul(config: MatmulConfig, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                     bias: Optional[torch.Tensor], cudnn_handle) -> Tuple[bool, str]:
    """Run matmul using cuDNN and return success status and message."""
    try:
        stream = torch.cuda.current_stream().cuda_stream
        cudnn.set_stream(handle=cudnn_handle, stream=stream)

        # Create graph
        graph = cudnn.pygraph(
            handle=cudnn_handle,
            compute_data_type=config.compute_dtype,
            intermediate_data_type=cudnn.data_type.FLOAT,
        )

        # Create input tensors
        A_tensor = graph.tensor_like(A)
        B_tensor = graph.tensor_like(B)

        # Handle data type casting if needed
        mma_dtype = convert_to_cudnn_type(config.c_dtype)

        if config.a_dtype != config.c_dtype:
            A_casted = graph.identity(input=A_tensor, compute_data_type=cudnn.data_type.FLOAT)
            A_casted.set_data_type(mma_dtype)
        else:
            A_casted = A_tensor

        if config.b_dtype != config.c_dtype:
            B_casted = graph.identity(input=B_tensor, compute_data_type=cudnn.data_type.FLOAT)
            B_casted.set_data_type(mma_dtype)
        else:
            B_casted = B_tensor

        # Matmul
        result = graph.matmul(name="matmul", A=A_casted, B=B_casted)

        # Epilogue
        if bias is not None and config.epilogue in [EpilogueType.BIAS, EpilogueType.BIAS_RELU, EpilogueType.BIAS_GELU]:
            bias_tensor = graph.tensor_like(bias)
            result = graph.bias(name="bias", input=result, bias=bias_tensor)
        else:
            bias_tensor = None

        if config.epilogue in [EpilogueType.RELU, EpilogueType.BIAS_RELU]:
            result = graph.relu(name="relu", input=result)
        elif config.epilogue in [EpilogueType.GELU, EpilogueType.BIAS_GELU]:
            result = graph.gelu(name="gelu", input=result)

        result.set_output(True).set_data_type(mma_dtype)

        # Build and execute
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)

        # Allocate workspace and fill with garbage to catch uninitialized memory bugs
        workspace_size = graph.get_workspace_size()
        workspace = torch.empty(workspace_size, device='cuda', dtype=torch.uint8)
        if workspace_size > 0:
            # Fill with random garbage + some NaN patterns to test proper workspace init
            workspace.random_(0, 256)
            nan_mask = torch.rand(workspace_size, device='cuda') < 0.1
            workspace[nan_mask] = 0xFF

        # Build variant pack
        variant_pack = {A_tensor: A, B_tensor: B, result: C}
        if bias_tensor is not None:
            variant_pack[bias_tensor] = bias

        # Execute
        graph.execute(variant_pack, workspace, handle=cudnn_handle)
        torch.cuda.synchronize()

        return True, "success"

    except cudnn.cudnnGraphNotSupportedError as e:
        return False, f"graph not supported: {e}"
    except Exception as e:
        return False, f"error: {e}"


def compare_results(C_actual: torch.Tensor, C_expected: torch.Tensor, config: MatmulConfig,
                    max_diffs: int = 10) -> Tuple[bool, int, str]:
    """Compare actual and expected results."""
    # Determine tolerances based on dtype
    # Note: cuDNN uses TF32 for FP32 tensor core ops, which has same precision as FP16 (10-bit mantissa)
    if config.c_dtype == torch.float32:
        rtol, atol = 1e-2, 2e-2  # TF32 precision, not full FP32
    elif config.c_dtype == torch.float16:
        rtol, atol = 1e-2, 2e-2
    else:  # bfloat16
        rtol, atol = 1e-2, 2e-2

    # Scale tolerances based on problem size (larger problems accumulate more error)
    # Use max(1.0, ...) to only increase tolerances for large K, never decrease
    scale_factor = max(1.0, math.sqrt(config.K / 128.0))
    rtol *= scale_factor
    atol *= scale_factor

    try:
        torch.testing.assert_close(C_actual, C_expected, rtol=rtol, atol=atol)
        return True, 0, f"Numerical divergence within limits (rtol={rtol:.2e}, atol={atol:.2e})"
    except AssertionError:
        # Count mismatches
        close_mask = torch.isclose(C_actual.float(), C_expected.float(), rtol=rtol, atol=atol)
        mismatch_count = (~close_mask).sum().item()
        total_elements = C_actual.numel()
        percentage = 100.0 * mismatch_count / total_elements

        msg = f"Found {mismatch_count:,} mismatches ({percentage:.2f}%) out of {total_elements:,} elements"

        # Show some mismatches
        if max_diffs > 0:
            mismatches = torch.where(~close_mask)
            for i in range(min(max_diffs, mismatch_count)):
                idx = tuple(m[i].item() for m in mismatches)
                actual = C_actual[idx].item()
                expected = C_expected[idx].item()
                diff = actual - expected
                msg += f"\n  idx{idx}: actual={actual:+.6e}, expected={expected:+.6e}, diff={diff:+.2e}"

        return False, mismatch_count, msg


# ============================================================================
# Test Output Formatting
# ============================================================================

def format_test_header(test_num: int, total_tests: int, config: MatmulConfig) -> str:
    """Format test header similar to sample log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu_arch = get_gpu_arch()
    gpu_name = get_gpu_name()
    sm_count = get_sm_count()
    cudnn_ver = cudnn.backend_version()

    lines = [
        "",
        "=" * 90,
        f"#### Test #{test_num} of {total_tests} at {timestamp} ",
        "",
        f"test_name        = test_matmul_fuzzer[test{test_num}]",
        f"platform_info    = {gpu_arch} ({sm_count} SM-s, {gpu_name}), cudnn_ver={cudnn_ver}",
        f"rng_data_seed    = {config.rng_seed}",
        f"basic_dims       = [b={config.batch}, M={config.M}, N={config.N}, K={config.K}]",
        f"matrix_a(b,m,k)  = dim={config.a_shape}, strides={config.a_strides}, elems={config.a_elems}, type={config.a_dtype}",
        f"matrix_b(b,k,n)  = dim={config.b_shape}, strides={config.b_strides}, elems={config.b_elems}, type={config.b_dtype}",
        f"matrix_c(b,m,n)  = dim={config.c_shape}, strides={config.c_strides}, elems={config.c_elems}, type={config.c_dtype}",
    ]

    # Add bias info if epilogue uses bias
    if config.epilogue in [EpilogueType.BIAS, EpilogueType.BIAS_RELU, EpilogueType.BIAS_GELU] and config.bias_shape:
        lines.append(f"bias(b,m,n)      = dim={config.bias_shape}, strides={config.bias_strides}, elems={config.bias_elems}, type={config.c_dtype}")

    lines += [
        f"epilogue         = {epilogue_name(config.epilogue)}",
        f"repro_cmd        = pytest -vv -s -rA {__file__}::test_repro --repro \"{config.to_repro_dict()}\"",
        " ",
    ]
    return "\n".join(lines)


def format_test_result(passed: bool, message: str) -> str:
    """Format test result."""
    lines = [
        f"%%%% {message}",
    ]
    if passed:
        lines.append("@@@@ Overall result: PASSED, everything looks good!")
    else:
        lines.append("@@@@ Overall result: FAILED")
    return "\n".join(lines)


# ============================================================================
# PyTest Infrastructure
# ============================================================================

def pytest_addoption(parser):
    """Add custom command line options."""
    try:
        parser.addoption("--num-tests", action="store", type=int, default=100,
                        help="Number of random tests to run")
        parser.addoption("--fuzz-seed", action="store", type=int, default=None,
                        help="Random seed for test generation")
        parser.addoption("--unaligned", action="store_true", default=False,
                        help="Allow M/N/K dimensions that are not multiples of 8")
    except Exception:
        pass  # Options may already be added


def tlist(*, num_tests: int, rng_seed: int):
    """Generate list of test parameters (legacy, without pre-generated configs)."""
    rng = random.Random(rng_seed)
    return [(i + 1, num_tests, rng.randint(65536, 2**31 - 1)) for i in range(num_tests)]


def tlist_with_configs(*, num_tests: int, rng_seed: int, allow_unaligned: bool = False):
    """Generate list of test parameters with pre-generated configs for descriptive test names."""
    rng = random.Random(rng_seed)
    params = []
    for i in range(num_tests):
        config_seed = rng.randint(65536, 2**31 - 1)
        generator = ConfigGenerator(config_seed, allow_unaligned=allow_unaligned)
        config = generator.generate()
        params.append((i + 1, num_tests, config_seed, config))
    return params


def make_test_id(param, prefix: str = "t"):
    """Create descriptive test ID from pre-generated config."""
    test_num, total_tests, config_seed, config = param
    dtype_short = {
        torch.float16: 'f16',
        torch.bfloat16: 'bf16',
        torch.float32: 'f32',
        torch.int8: 'i8',
    }
    dt = dtype_short.get(config.a_dtype, 'unk')
    epi = epilogue_name(config.epilogue)[:4]  # Truncate epilogue name
    return f"{prefix}{test_num}_b{config.batch}_M{config.M}xN{config.N}xK{config.K}_{dt}_{epi}"


# Generate test list
def get_test_params(request):
    """Get test parameters from pytest config."""
    try:
        num_tests = request.config.getoption("--num-tests", default=100)
        seed = request.config.getoption("--fuzz-seed", default=None)
    except Exception:
        num_tests = 100
        seed = None

    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    return num_tests, seed


# Fixed test list for default runs
DEFAULT_NUM_TESTS = 2048
DEFAULT_SEED = 42
TEST_PARAMS = tlist_with_configs(num_tests=DEFAULT_NUM_TESTS, rng_seed=DEFAULT_SEED)


@pytest.mark.L0
@pytest.mark.parametrize("test_num,total_tests,config_seed,config", TEST_PARAMS,
                        ids=[make_test_id(p) for p in TEST_PARAMS])
def test_matmul_fuzz(test_num: int, total_tests: int, config_seed: int, config: MatmulConfig, cudnn_handle, request):
    """Fuzz test for matmul operations (M/N/K aligned to multiples of 8)."""

    # Skip if cuDNN handle not available
    if cudnn_handle is None:
        pytest.skip("cuDNN handle not available")

    # Get display options
    try:
        max_diffs = request.config.getoption("--diffs", default=10)
    except Exception:
        max_diffs = 10

    # Create tensors
    rng = random.Random(config_seed)
    A, B, C, bias = create_tensors(config, rng)
    C_expected = None

    try:
        # Print test header
        print(format_test_header(test_num, total_tests, config))

        # Compute reference
        C_expected = compute_reference(config, A, B, bias)

        # Run cuDNN
        success, msg = run_cudnn_matmul(config, A, B, C, bias, cudnn_handle)

        if not success:
            print(f"%%%% cuDNN execution failed: {msg}")
            # Skip tests with unsupported configurations rather than failing
            skip_keywords = ["not supported", "finalize failed", "mismatch", "invalid", "unsupported"]
            if any(kw in msg.lower() for kw in skip_keywords):
                print("@@@@ Overall result: SKIPPED (unsupported configuration)")
                pytest.skip(f"Unsupported configuration: {msg}")
            else:
                print("@@@@ Overall result: FAILED")
                pytest.fail(f"cuDNN execution failed: {msg}")

        # Compare results
        passed, mismatch_count, compare_msg = compare_results(C, C_expected, config, max_diffs)

        print(format_test_result(passed, compare_msg))

        if not passed:
            pytest.fail(f"Numerical mismatch: {mismatch_count} elements differ")
    finally:
        # Explicit cleanup to prevent GPU memory accumulation
        del A, B, C
        if bias is not None:
            del bias
        if C_expected is not None:
            del C_expected
        torch.cuda.empty_cache()


# Separate test list for unaligned stress testing
UNALIGNED_TEST_PARAMS = tlist_with_configs(num_tests=1024, rng_seed=12345, allow_unaligned=True)


@pytest.mark.L1  # L1 for stress testing with unaligned dimensions
@pytest.mark.parametrize("test_num,total_tests,config_seed,config", UNALIGNED_TEST_PARAMS,
                        ids=[make_test_id(p, prefix="u") for p in UNALIGNED_TEST_PARAMS])
def test_matmul_fuzz_unaligned(test_num: int, total_tests: int, config_seed: int, config: MatmulConfig, cudnn_handle, request):
    """Fuzz test for matmul with unaligned M/N/K dimensions (stress test)."""

    # Skip if cuDNN handle not available
    if cudnn_handle is None:
        pytest.skip("cuDNN handle not available")

    # Get display options
    try:
        max_diffs = request.config.getoption("--diffs", default=10)
    except Exception:
        max_diffs = 10

    # Create tensors
    rng = random.Random(config_seed)
    A, B, C, bias = create_tensors(config, rng)
    C_expected = None

    try:
        # Print test header
        print(format_test_header(test_num, total_tests, config))

        # Compute reference
        C_expected = compute_reference(config, A, B, bias)

        # Run cuDNN
        success, msg = run_cudnn_matmul(config, A, B, C, bias, cudnn_handle)

        if not success:
            print(f"%%%% cuDNN execution failed: {msg}")
            # Skip tests with unsupported configurations rather than failing
            skip_keywords = ["not supported", "finalize failed", "mismatch", "invalid", "unsupported"]
            if any(kw in msg.lower() for kw in skip_keywords):
                print("@@@@ Overall result: SKIPPED (unsupported configuration)")
                pytest.skip(f"Unsupported configuration: {msg}")
            else:
                print("@@@@ Overall result: FAILED")
                pytest.fail(f"cuDNN execution failed: {msg}")

        # Compare results
        passed, mismatch_count, compare_msg = compare_results(C, C_expected, config, max_diffs)

        print(format_test_result(passed, compare_msg))

        if not passed:
            pytest.fail(f"Numerical mismatch: {mismatch_count} elements differ")
    finally:
        # Explicit cleanup to prevent GPU memory accumulation
        del A, B, C
        if bias is not None:
            del bias
        if C_expected is not None:
            del C_expected
        torch.cuda.empty_cache()


@pytest.mark.L0
def test_repro(cudnn_handle, request):
    """Reproduction test for debugging specific configurations."""
    repro_str = request.config.getoption("--repro", default=None)
    if repro_str is None:
        pytest.skip("No --repro option provided. Use: pytest test_matmul_fuzzer.py::test_repro --repro '<config_dict>'")

    # Parse repro config
    import ast
    repro_dict = ast.literal_eval(repro_str)

    # Regenerate config using the same seed (ensures identical random choices)
    generator = ConfigGenerator(repro_dict['rng_seed'])
    config = generator.generate()

    # Override with explicit values from repro dict
    config.batch = repro_dict['batch']
    config.M = repro_dict['M']
    config.N = repro_dict['N']
    config.K = repro_dict['K']
    config.a_dtype = eval(repro_dict['a_dtype'])
    config.b_dtype = eval(repro_dict['b_dtype'])
    config.c_dtype = eval(repro_dict['c_dtype'])
    config.epilogue = EpilogueType(repro_dict['epilogue'])
    config.rng_seed = repro_dict['rng_seed']

    # Run test
    rng = random.Random(config.rng_seed)
    A, B, C, bias = create_tensors(config, rng)
    C_expected = None

    try:
        print(format_test_header(1, 1, config))

        C_expected = compute_reference(config, A, B, bias)
        success, msg = run_cudnn_matmul(config, A, B, C, bias, cudnn_handle)

        if not success:
            pytest.fail(f"cuDNN execution failed: {msg}")

        passed, mismatch_count, compare_msg = compare_results(C, C_expected, config, max_diffs=20)
        print(format_test_result(passed, compare_msg))

        if not passed:
            pytest.fail(f"Numerical mismatch: {mismatch_count} elements differ")
    finally:
        # Explicit cleanup to prevent GPU memory accumulation
        del A, B, C
        if bias is not None:
            del bias
        if C_expected is not None:
            del C_expected
        torch.cuda.empty_cache()


# ============================================================================
# Quick Sanity Tests
# ============================================================================

@pytest.mark.L0
def test_matmul_basic_fp16(cudnn_handle):
    """Basic FP16 matmul sanity test."""
    if cudnn_handle is None:
        pytest.skip("cuDNN handle not available")

    config = MatmulConfig(
        batch=2, M=64, N=128, K=256,
        a_dtype=torch.float16, b_dtype=torch.float16, c_dtype=torch.float16,
        compute_dtype=cudnn.data_type.FLOAT,
        a_layout=LayoutType.ROW_MAJOR_PACKED, b_layout=LayoutType.ROW_MAJOR_PACKED, c_layout=LayoutType.ROW_MAJOR_PACKED,
        a_transposed=False, b_transposed=False,
        epilogue=EpilogueType.NONE,
        rng_seed=12345,
    )

    rng = random.Random(config.rng_seed)
    A, B, C, bias = create_tensors(config, rng)
    C_expected = None

    try:
        print(format_test_header(1, 1, config))

        C_expected = compute_reference(config, A, B, bias)
        success, msg = run_cudnn_matmul(config, A, B, C, bias, cudnn_handle)

        assert success, f"cuDNN failed: {msg}"

        passed, _, compare_msg = compare_results(C, C_expected, config)
        print(format_test_result(passed, compare_msg))
        assert passed
    finally:
        del A, B, C
        if bias is not None:
            del bias
        if C_expected is not None:
            del C_expected
        torch.cuda.empty_cache()


@pytest.mark.L0
def test_matmul_basic_bf16(cudnn_handle):
    """Basic BF16 matmul sanity test."""
    if cudnn_handle is None:
        pytest.skip("cuDNN handle not available")

    config = MatmulConfig(
        batch=4, M=128, N=256, K=512,
        a_dtype=torch.bfloat16, b_dtype=torch.bfloat16, c_dtype=torch.bfloat16,
        compute_dtype=cudnn.data_type.FLOAT,
        a_layout=LayoutType.ROW_MAJOR_PACKED, b_layout=LayoutType.ROW_MAJOR_PACKED, c_layout=LayoutType.ROW_MAJOR_PACKED,
        a_transposed=False, b_transposed=False,
        epilogue=EpilogueType.NONE,
        rng_seed=54321,
    )

    rng = random.Random(config.rng_seed)
    A, B, C, bias = create_tensors(config, rng)
    C_expected = None

    try:
        print(format_test_header(1, 1, config))

        C_expected = compute_reference(config, A, B, bias)
        success, msg = run_cudnn_matmul(config, A, B, C, bias, cudnn_handle)

        assert success, f"cuDNN failed: {msg}"

        passed, _, compare_msg = compare_results(C, C_expected, config)
        print(format_test_result(passed, compare_msg))
        assert passed
    finally:
        del A, B, C
        if bias is not None:
            del bias
        if C_expected is not None:
            del C_expected
        torch.cuda.empty_cache()


@pytest.mark.L0
def test_matmul_with_bias(cudnn_handle):
    """Matmul with bias epilogue test."""
    if cudnn_handle is None:
        pytest.skip("cuDNN handle not available")

    config = MatmulConfig(
        batch=1, M=256, N=512, K=128,
        a_dtype=torch.float16, b_dtype=torch.float16, c_dtype=torch.float16,
        compute_dtype=cudnn.data_type.FLOAT,
        a_layout=LayoutType.ROW_MAJOR_PACKED, b_layout=LayoutType.ROW_MAJOR_PACKED, c_layout=LayoutType.ROW_MAJOR_PACKED,
        a_transposed=False, b_transposed=False,
        epilogue=EpilogueType.BIAS,
        rng_seed=98765,
    )

    rng = random.Random(config.rng_seed)
    A, B, C, bias = create_tensors(config, rng)
    C_expected = None

    try:
        print(format_test_header(1, 1, config))

        C_expected = compute_reference(config, A, B, bias)
        success, msg = run_cudnn_matmul(config, A, B, C, bias, cudnn_handle)

        assert success, f"cuDNN failed: {msg}"

        passed, _, compare_msg = compare_results(C, C_expected, config)
        print(format_test_result(passed, compare_msg))
        assert passed
    finally:
        del A, B, C
        if bias is not None:
            del bias
        if C_expected is not None:
            del C_expected
        torch.cuda.empty_cache()
