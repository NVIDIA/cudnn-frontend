"""
Convolution Fuzzer - Randomized stress testing for cuDNN convolution operations.

This fuzzer tests convolution operations with randomized:
- Shapes (batch, channels, spatial dimensions)
- Spatial dimensions (2D or 3D)
- Data types (fp16, bf16, fp32, int8)
- Convolution parameters (padding, stride, dilation)
- Operation types (fprop, dgrad, wgrad)
- Epilogues (none, bias, relu, bias_relu)
- Grouped convolutions (groups > 1, including depthwise) for all operation types

Layout: NHWC/NDHWC (channels last) for memory layout
Logical dimension order: N, C, spatial_dims...

Run with:
    pytest -vv -s -rA test_conv_fuzzer.py

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
import sys
import signal
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import IntEnum
from sdpa.helpers import create_sparse_int_tensor, print_tensor_stats, compare_tensors

# fmt: off

# Handle Ctrl-C gracefully
def signal_handler(sig, frame):
    print("\n\nInterrupted by user (Ctrl-C), exiting...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    print("This is pytest script. Run with: pytest -vv -s -rA test_conv_fuzzer.py")
    sys.exit(0)


# ============================================================================
# Configuration and Constants
# ============================================================================

class ConvType(IntEnum):
    FPROP = 0   # Forward convolution
    DGRAD = 1   # Input gradient (backward data)
    WGRAD = 2   # Weight gradient (backward filter)

class EpilogueType(IntEnum):
    NONE = 0
    BIAS = 1
    RELU = 2
    BIAS_RELU = 3

SUPPORTED_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
]

# int8 convolutions have stricter requirements, test separately
SUPPORTED_DTYPES_WITH_INT8 = SUPPORTED_DTYPES + [torch.int8]


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


def get_available_gpu_memory_mb() -> float:
    """Get available GPU memory in MB."""
    torch.cuda.synchronize()
    free, _total = torch.cuda.mem_get_info()
    return free / (1024 * 1024)


def conv_type_name(conv_type: ConvType) -> str:
    """Get human-readable conv type name."""
    names = {
        ConvType.FPROP: "fprop",
        ConvType.DGRAD: "dgrad",
        ConvType.WGRAD: "wgrad",
    }
    return names.get(conv_type, "unknown")


def epilogue_name(epilogue: EpilogueType) -> str:
    """Get human-readable epilogue name."""
    names = {
        EpilogueType.NONE: "none",
        EpilogueType.BIAS: "bias",
        EpilogueType.RELU: "relu",
        EpilogueType.BIAS_RELU: "bias_relu",
    }
    return names.get(epilogue, "unknown")


def compute_channels_last_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Compute channels-last strides for NHWC (2D) or NDHWC (3D) layout.

    Logical dim order: (N, C, spatial_dims...)
    Memory order: N, spatial_dims..., C

    For 2D (NCHW logical -> NHWC memory):
        shape = (N, C, H, W)
        memory_order = (N, H, W, C) -> strides computed from last to first
        strides[N] = H*W*C, strides[C] = 1, strides[H] = W*C, strides[W] = C

    For 3D (NCDHW logical -> NDHWC memory):
        shape = (N, C, D, H, W)
        memory_order = (N, D, H, W, C) -> strides computed from last to first
    """
    ndim = len(shape)
    if ndim < 3:
        raise ValueError(f"Shape must have at least 3 dimensions, got {ndim}")

    # shape = (N, C, spatial_dims...)
    N = shape[0]
    C = shape[1]
    spatial = shape[2:]  # (H, W) or (D, H, W)

    # Memory layout: (N, spatial_dims..., C)
    # Compute strides from innermost to outermost
    strides = [0] * ndim

    # C is innermost in memory (stride = 1)
    strides[1] = 1

    # Spatial dims next (reversed order in memory)
    stride = C
    for i in range(ndim - 1, 1, -1):  # W, H, [D] order
        strides[i] = stride
        stride *= shape[i]

    # N is outermost
    strides[0] = stride

    return tuple(strides)


def compute_num_elements(shape: Tuple[int, ...], strides: Tuple[int, ...]) -> int:
    """Compute number of elements needed for storage given shape and strides."""
    if not shape:
        return 1
    max_offset = sum((d - 1) * s for d, s in zip(shape, strides))
    return max_offset + 1


def compute_output_spatial(input_spatial: int, filter_spatial: int,
                           padding: int, stride: int, dilation: int) -> int:
    """Compute output spatial dimension for convolution."""
    effective_filter = (filter_spatial - 1) * dilation + 1
    return (input_spatial + 2 * padding - effective_filter) // stride + 1


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
class ConvConfig:
    """Configuration for a single convolution test."""
    # Spatial dimensions (2 for 2D, 3 for 3D)
    spatial_dims: int

    # Basic dimensions
    batch: int           # N
    in_channels: int     # C_in
    out_channels: int    # C_out (K)

    # Grouped convolution (groups=1 is standard conv)
    # For grouped conv: in_channels and out_channels must be divisible by groups
    # Weight shape becomes (out_channels, in_channels // groups, filter...)
    groups: int

    # Spatial sizes: (H, W) for 2D or (D, H, W) for 3D
    input_spatial: Tuple[int, ...]   # Input spatial dimensions
    filter_spatial: Tuple[int, ...]  # Filter spatial dimensions

    # Convolution parameters (per spatial dimension)
    padding: Tuple[int, ...]
    stride: Tuple[int, ...]
    dilation: Tuple[int, ...]

    # Operation type
    conv_type: ConvType

    # Data types
    x_dtype: torch.dtype     # Input dtype
    w_dtype: torch.dtype     # Weight dtype
    y_dtype: torch.dtype     # Output dtype

    # Epilogue (only for fprop)
    epilogue: EpilogueType

    # Random seed for data generation
    rng_seed: int

    # Computed shapes and strides (set during tensor creation)
    # Logical order: (N, C, spatial...)
    x_shape: Tuple[int, ...] = None
    w_shape: Tuple[int, ...] = None
    y_shape: Tuple[int, ...] = None
    x_strides: Tuple[int, ...] = None
    w_strides: Tuple[int, ...] = None
    y_strides: Tuple[int, ...] = None
    x_elems: int = 0
    w_elems: int = 0
    y_elems: int = 0

    # Bias tensor info (for epilogue)
    bias_shape: Tuple[int, ...] = None
    bias_strides: Tuple[int, ...] = None
    bias_elems: int = 0

    @property
    def output_spatial(self) -> Tuple[int, ...]:
        """Compute output spatial dimensions."""
        return tuple(
            compute_output_spatial(inp, flt, pad, strd, dil)
            for inp, flt, pad, strd, dil in zip(
                self.input_spatial, self.filter_spatial,
                self.padding, self.stride, self.dilation
            )
        )

    def to_repro_dict(self) -> dict:
        """Convert config to reproducible dictionary."""
        return {
            'spatial_dims': self.spatial_dims,
            'batch': self.batch,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'groups': self.groups,
            'input_spatial': self.input_spatial,
            'filter_spatial': self.filter_spatial,
            'padding': self.padding,
            'stride': self.stride,
            'dilation': self.dilation,
            'conv_type': int(self.conv_type),
            'x_dtype': str(self.x_dtype),
            'w_dtype': str(self.w_dtype),
            'y_dtype': str(self.y_dtype),
            'epilogue': int(self.epilogue),
            'rng_seed': self.rng_seed,
        }


class ConfigGenerator:
    """Generator for random convolution configurations."""

    # Group options for grouped convolutions
    # L0: powers of 2 (aligned, fast)
    # L1: comprehensive range for edge case coverage (invalid values filtered out)
    _L0_GROUPS = [2, 4, 8, 16]
    _L1_GROUPS = list(range(2, 18)) + [24, 32]  # 2-17 plus common larger values

    def __init__(self, seed: int, allow_unaligned: bool = False):
        self.rng = random.Random(seed)
        self.sm_version = torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1]
        self.allow_unaligned = allow_unaligned

    def random_spatial_dims(self) -> int:
        """Generate random spatial dimension count (2 or 3)."""
        return self.rng.choice([2, 2, 2, 3])  # Prefer 2D

    def random_batch(self) -> int:
        """Generate random batch size."""
        return self.rng.choice([1, 1, 2, 2, 4, 4, 8])

    def random_channels(self, min_val: int = 1, max_val: int = 256) -> int:
        """Generate random channel count (reduced max for memory)."""
        val = self.rng.randint(int(math.sqrt(min_val)), int(math.sqrt(max_val)))
        val = val * val
        if self.allow_unaligned:
            return max(1, val)
        else:
            # Round up to multiple of 8 for tensor core alignment
            return max(8, ((val + 7) // 8) * 8)

    def random_spatial_size(self, min_val: int = 1, max_val: int = 128) -> int:
        """Generate random spatial dimension size (reduced max for memory)."""
        val = self.rng.randint(int(math.sqrt(min_val)), int(math.sqrt(max_val)))
        val = val * val
        if self.allow_unaligned:
            return max(1, val)
        else:
            return max(8, ((val + 7) // 8) * 8)

    def random_filter_size(self) -> int:
        """Generate random filter spatial size."""
        return self.rng.choice([1, 1, 3, 3, 3, 5, 7])

    def random_padding(self, filter_size: int) -> int:
        """Generate random padding."""
        # Padding typically 0 to (filter_size - 1) // 2
        max_pad = (filter_size - 1) // 2
        return self.rng.randint(0, max(0, max_pad))

    def random_stride(self) -> int:
        """Generate random stride."""
        return self.rng.choice([1, 1, 1, 2, 2, 3])

    def random_dilation(self) -> int:
        """Generate random dilation."""
        return self.rng.choice([1, 1, 1, 1, 2])

    def random_groups(self, in_channels: int, out_channels: int) -> int:
        """
        Generate random group count for grouped convolutions.

        For grouped convolutions:
        - groups must divide both in_channels and out_channels
        - Weight shape becomes (out_channels, in_channels // groups, filter...)

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels

        Returns:
            A valid group count that divides both in_channels and out_channels
        """
        # Find common divisors of in_channels and out_channels
        common_gcd = math.gcd(in_channels, out_channels)

        # Filter group options to those that divide the GCD
        group_options = self._L1_GROUPS if self.allow_unaligned else self._L0_GROUPS
        valid_groups = [g for g in group_options if common_gcd % g == 0]

        if not valid_groups:
            return 1  # Fallback to standard convolution

        return self.rng.choice(valid_groups)

    def random_dtype(self) -> torch.dtype:
        """Generate random data type."""
        return self.rng.choice(SUPPORTED_DTYPES)

    def random_conv_type(self) -> ConvType:
        """Generate random convolution type."""
        # Weight towards fprop but include dgrad/wgrad
        weights = [0.5, 0.25, 0.25]  # fprop, dgrad, wgrad
        return self.rng.choices(list(ConvType), weights=weights)[0]

    def random_epilogue(self) -> EpilogueType:
        """Generate random epilogue type."""
        weights = [0.6, 0.15, 0.15, 0.1]
        return self.rng.choices(list(EpilogueType), weights=weights)[0]

    def generate(self, force_grouped: bool = False, force_depthwise: bool = False,
                 force_conv_type: Optional[ConvType] = None,
                 depthwise_channels_list: Optional[list] = None) -> ConvConfig:
        """
        Generate a random convolution configuration.

        Args:
            force_grouped: If True, always generate a grouped convolution (groups > 1)
            force_depthwise: If True, generate a depthwise convolution (groups = in_channels = out_channels)
            force_conv_type: If provided, use this conv type instead of random selection
            depthwise_channels_list: If provided, use this list for depthwise channel selection
        """
        spatial_dims = self.random_spatial_dims()
        batch = self.random_batch()

        # For depthwise, pick channels upfront; otherwise generate randomly
        if force_depthwise:
            # Depthwise: groups = in_channels = out_channels
            channels_options = depthwise_channels_list if depthwise_channels_list else [8, 16, 32, 64, 128]
            depthwise_channels = self.rng.choice(channels_options)
            in_channels = depthwise_channels
            out_channels = depthwise_channels
        else:
            # For grouped convolutions, we need channels divisible by potential group values
            # Generate channels that are more likely to work with common group sizes
            in_channels = self.random_channels()
            out_channels = self.random_channels()

        # Generate spatial dimensions
        input_spatial = tuple(self.random_spatial_size() for _ in range(spatial_dims))
        filter_spatial = tuple(self.random_filter_size() for _ in range(spatial_dims))

        # Ensure output spatial dims are positive
        padding = []
        stride = []
        dilation = []
        for i in range(spatial_dims):
            flt = filter_spatial[i]
            dil = self.random_dilation()
            strd = self.random_stride()
            pad = self.random_padding(flt)

            # Check output size is positive
            effective_filter = (flt - 1) * dil + 1
            out_size = (input_spatial[i] + 2 * pad - effective_filter) // strd + 1

            # If output would be non-positive, adjust padding
            while out_size < 1:
                pad += 1
                out_size = (input_spatial[i] + 2 * pad - effective_filter) // strd + 1

            padding.append(pad)
            stride.append(strd)
            dilation.append(dil)

        padding = tuple(padding)
        stride = tuple(stride)
        dilation = tuple(dilation)

        # Convolution type
        if force_conv_type is not None:
            conv_type = force_conv_type
        else:
            conv_type = self.random_conv_type()

        # Determine groups
        # 25% of tests should be grouped (groups > 1)
        groups = 1
        if force_depthwise:
            # Depthwise: groups = in_channels = out_channels (already set above)
            groups = in_channels
        elif force_grouped or self.rng.random() < 0.25:
            # 25% chance of grouped convolution (or forced)
            groups = self.random_groups(in_channels, out_channels)
            # If we couldn't find a valid group > 1, try adjusting channels
            if groups == 1 and (force_grouped or self.rng.random() < 0.5):
                # Round up channels to be divisible by a common group size
                group_options = self._L1_GROUPS if self.allow_unaligned else self._L0_GROUPS
                target_group = self.rng.choice(group_options)
                in_channels = ((in_channels + target_group - 1) // target_group) * target_group
                out_channels = ((out_channels + target_group - 1) // target_group) * target_group
                groups = target_group

        # Data types - ensure compatible combinations
        # Keep same dtype for all tensors for stability (like test_conv_bias.py)
        x_dtype = self.random_dtype()
        w_dtype = x_dtype  # Same dtype for input and weights
        y_dtype = x_dtype  # Same dtype for output (mixed precision needs special handling)

        # Epilogue only for fprop
        if conv_type == ConvType.FPROP:
            epilogue = self.random_epilogue()
        else:
            epilogue = EpilogueType.NONE

        config = ConvConfig(
            spatial_dims=spatial_dims,
            batch=batch,
            in_channels=in_channels,
            out_channels=out_channels,
            groups=groups,
            input_spatial=input_spatial,
            filter_spatial=filter_spatial,
            padding=padding,
            stride=stride,
            dilation=dilation,
            conv_type=conv_type,
            x_dtype=x_dtype,
            w_dtype=w_dtype,
            y_dtype=y_dtype,
            epilogue=epilogue,
            rng_seed=self.rng.randint(0, 2**31 - 1),
        )

        return config


# ============================================================================
# Test Execution
# ============================================================================

def create_tensors(config: ConvConfig, rng: random.Random):
    """
    Create tensors based on configuration.

    Tensor naming convention (shapes are always the same regardless of conv_type):
      X: (N, C_in, spatial...)       - input image shape
      W: (C_out, C_in // groups, filter...)    - weight/filter shape for grouped conv
      Y: (N, C_out, output_spatial...) - output shape

    For grouped convolutions:
      - groups must divide both in_channels and out_channels
      - Weight shape is (out_channels, in_channels // groups, filter...)
      - cuDNN infers group count from the tensor dimensions

    Meaning varies by conv_type:
      FPROP: X=input, W=weights, Y=output       (compute Y from X,W)
      DGRAD: X=dX(output), W=weights, Y=dY(input)  (compute dX from dY,W)
      WGRAD: X=input, W=dW(output), Y=dY(input)    (compute dW from X,dY)
    """
    torch_rng = torch.Generator(device='cuda')
    torch_rng.manual_seed(config.rng_seed)

    # Compute shapes
    # For grouped convolutions, weight shape is (out_channels, in_channels // groups, filter...)
    x_shape = (config.batch, config.in_channels) + config.input_spatial
    w_shape = (config.out_channels, config.in_channels // config.groups) + config.filter_spatial
    y_shape = (config.batch, config.out_channels) + config.output_spatial

    # Use PyTorch's native channels_last memory format for proper cuDNN compatibility
    if config.spatial_dims == 2:
        memory_format = torch.channels_last
    else:  # 3D
        memory_format = torch.channels_last_3d

    # Create tensors - which ones are input (random) vs output (garbage) depends on conv_type
    # Output tensors are filled with garbage (random + NaNs) to catch bugs where cuDNN
    # doesn't write all output locations
    # Input tensors use sparse small integers for better low-precision testing
    if config.conv_type == ConvType.FPROP:
        # FPROP: X,W are inputs, Y is output
        X = create_sparse_int_tensor(x_shape, config.x_dtype, torch_rng, memory_format=memory_format)
        W = create_sparse_int_tensor(w_shape, config.w_dtype, torch_rng, memory_format=memory_format)
        Y = torch.empty(y_shape, device='cuda', dtype=config.y_dtype).to(memory_format=memory_format)
        fill_with_garbage(Y)  # Output - fill with garbage

    elif config.conv_type == ConvType.DGRAD:
        # DGRAD: Y(dY),W are inputs, X(dX) is output
        Y = create_sparse_int_tensor(y_shape, config.y_dtype, torch_rng, memory_format=memory_format)  # dY
        W = create_sparse_int_tensor(w_shape, config.w_dtype, torch_rng, memory_format=memory_format)  # weights
        X = torch.empty(x_shape, device='cuda', dtype=config.x_dtype).to(memory_format=memory_format)
        fill_with_garbage(X)  # dX output - fill with garbage

    else:  # WGRAD
        # WGRAD: X,Y(dY) are inputs, W(dW) is output
        X = create_sparse_int_tensor(x_shape, config.x_dtype, torch_rng, memory_format=memory_format)  # input
        Y = create_sparse_int_tensor(y_shape, config.y_dtype, torch_rng, memory_format=memory_format)  # dY
        W = torch.empty(w_shape, device='cuda', dtype=config.w_dtype).to(memory_format=memory_format)
        fill_with_garbage(W)  # dW output - fill with garbage

    # Update config with actual shapes and strides
    config.x_shape = tuple(X.size())
    config.w_shape = tuple(W.size())
    config.y_shape = tuple(Y.size())
    config.x_strides = tuple(X.stride())
    config.w_strides = tuple(W.stride())
    config.y_strides = tuple(Y.stride())
    config.x_elems = X.numel()
    config.w_elems = W.numel()
    config.y_elems = Y.numel()

    # Bias tensor if needed (only for FPROP, shape: 1, K, 1, 1, ... for broadcasting)
    bias = None
    if config.conv_type == ConvType.FPROP and config.epilogue in [EpilogueType.BIAS, EpilogueType.BIAS_RELU]:
        bias_shape = (1, config.out_channels) + (1,) * config.spatial_dims
        bias = create_sparse_int_tensor(bias_shape, config.y_dtype, torch_rng)

        config.bias_shape = tuple(bias.size())
        config.bias_strides = tuple(bias.stride())
        config.bias_elems = bias.numel()

    return X, W, Y, bias


def compute_reference(config: ConvConfig, X: torch.Tensor, W: torch.Tensor,
                      Y: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Compute reference result using PyTorch.

    Convention:
      FPROP: compute Y = conv(X, W) + bias + relu
      DGRAD: compute dX from dY(=Y) and W
      WGRAD: compute dW from X and dY(=Y)

    Returns the tensor that should match the cuDNN output:
      FPROP -> Y_ref (to compare with Y)
      DGRAD -> dX_ref (to compare with X)
      WGRAD -> dW_ref (to compare with W)
    """
    compute_dtype = torch.float32

    # Disable cuDNN for reference to avoid comparing cuDNN with itself
    with torch.backends.cudnn.flags(enabled=False):
        if config.conv_type == ConvType.FPROP:
            # FPROP: Y = conv(X, W)
            X_f = X.to(compute_dtype).contiguous()
            W_f = W.to(compute_dtype).contiguous()

            try:
                if config.spatial_dims == 2:
                    ref = torch.nn.functional.conv2d(
                        X_f, W_f,
                        padding=config.padding,
                        stride=config.stride,
                        dilation=config.dilation,
                        groups=config.groups
                    )
                else:
                    ref = torch.nn.functional.conv3d(
                        X_f, W_f,
                        padding=config.padding,
                        stride=config.stride,
                        dilation=config.dilation,
                        groups=config.groups
                    )

                # Apply epilogue
                if bias is not None and config.epilogue in [EpilogueType.BIAS, EpilogueType.BIAS_RELU]:
                    ref = ref + bias.to(compute_dtype)
                if config.epilogue in [EpilogueType.RELU, EpilogueType.BIAS_RELU]:
                    ref = torch.relu(ref)

                return ref.to(config.y_dtype)
            finally:
                del X_f, W_f

        elif config.conv_type == ConvType.DGRAD:
            # DGRAD: dX = conv_dgrad(dY, W)
            # Y contains dY (gradient from upstream), W contains weights
            # We compute dX (gradient w.r.t. input)
            dY_f = Y.to(compute_dtype).contiguous()
            W_f = W.to(compute_dtype).contiguous()

            # Use autograd to compute the reference
            # Create a dummy input and run forward, then backward to get dX
            dummy_X = torch.zeros(config.x_shape, device='cuda', dtype=compute_dtype, requires_grad=True)
            dummy_Y = None  # set before try so finally can safely del if OOM occurs during conv

            try:
                if config.spatial_dims == 2:
                    dummy_Y = torch.nn.functional.conv2d(
                        dummy_X, W_f,
                        padding=config.padding,
                        stride=config.stride,
                        dilation=config.dilation,
                        groups=config.groups
                    )
                else:
                    dummy_Y = torch.nn.functional.conv3d(
                        dummy_X, W_f,
                        padding=config.padding,
                        stride=config.stride,
                        dilation=config.dilation,
                        groups=config.groups
                    )

                # Backward pass to get dX
                dummy_Y.backward(dY_f)
                dX_ref = dummy_X.grad.clone()

                return dX_ref.to(config.x_dtype)
            finally:
                del dY_f, W_f, dummy_X
                if dummy_Y is not None:
                    del dummy_Y

        else:  # WGRAD
            # WGRAD: dW = conv_wgrad(X, dY)
            # X contains input, Y contains dY (gradient from upstream)
            # We compute dW (gradient w.r.t. weights)
            X_f = X.to(compute_dtype).contiguous()
            dY_f = Y.to(compute_dtype).contiguous()

            # Use autograd to compute the reference
            # Create a dummy weight and run forward, then backward to get dW
            dummy_W = torch.zeros(config.w_shape, device='cuda', dtype=compute_dtype, requires_grad=True)
            dummy_Y = None  # set before try so finally can safely del if OOM occurs during conv

            try:
                if config.spatial_dims == 2:
                    dummy_Y = torch.nn.functional.conv2d(
                        X_f, dummy_W,
                        padding=config.padding,
                        stride=config.stride,
                        dilation=config.dilation,
                        groups=config.groups
                    )
                else:
                    dummy_Y = torch.nn.functional.conv3d(
                        X_f, dummy_W,
                        padding=config.padding,
                        stride=config.stride,
                        dilation=config.dilation,
                        groups=config.groups
                    )

                # Backward pass to get dW
                dummy_Y.backward(dY_f)
                dW_ref = dummy_W.grad.clone()

                return dW_ref.to(config.w_dtype)
            finally:
                del X_f, dY_f, dummy_W
                if dummy_Y is not None:
                    del dummy_Y


def run_cudnn_conv(config: ConvConfig, X: torch.Tensor, W: torch.Tensor, Y: torch.Tensor,
                   bias: Optional[torch.Tensor], cudnn_handle) -> Tuple[bool, str]:
    """
    Run convolution using cuDNN and return success status and message.

    Convention:
      FPROP: inputs=X,W, output=Y     (compute Y)
      DGRAD: inputs=Y(dY),W, output=X (compute dX into X)
      WGRAD: inputs=X,Y(dY), output=W (compute dW into W)
    """
    try:
        stream = torch.cuda.current_stream().cuda_stream
        cudnn.set_stream(handle=cudnn_handle, stream=stream)

        # Determine compute and IO data types
        if config.x_dtype == torch.float32:
            io_dtype = cudnn.data_type.FLOAT
        elif config.x_dtype == torch.bfloat16:
            io_dtype = cudnn.data_type.BFLOAT16
        else:
            io_dtype = cudnn.data_type.HALF

        # Create graph
        graph = cudnn.pygraph(
            handle=cudnn_handle,
            io_data_type=io_dtype,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )

        # Build convolution operation based on conv_type
        if config.conv_type == ConvType.FPROP:
            # FPROP: Y = conv(X, W)
            X_tensor = graph.tensor(
                name="X", dim=list(X.size()), stride=list(X.stride()),
                data_type=convert_to_cudnn_type(config.x_dtype)
            )
            W_tensor = graph.tensor(
                name="W", dim=list(W.size()), stride=list(W.stride()),
                data_type=convert_to_cudnn_type(config.w_dtype)
            )

            conv_output = graph.conv_fprop(
                image=X_tensor,
                weight=W_tensor,
                padding=list(config.padding),
                stride=list(config.stride),
                dilation=list(config.dilation),
            )

            # Apply epilogue
            if config.epilogue in [EpilogueType.BIAS, EpilogueType.BIAS_RELU]:
                B_tensor = graph.tensor(
                    name="B", dim=list(bias.size()), stride=list(bias.stride()),
                    data_type=convert_to_cudnn_type(config.y_dtype)
                )
                conv_output = graph.bias(name="bias", input=conv_output, bias=B_tensor)

            if config.epilogue in [EpilogueType.RELU, EpilogueType.BIAS_RELU]:
                conv_output = graph.relu(name="relu", input=conv_output)

            conv_output.set_output(True)

            # Execution dict: X,W are inputs, Y is output
            exec_dict = {X_tensor: X, W_tensor: W, conv_output: Y}
            if config.epilogue in [EpilogueType.BIAS, EpilogueType.BIAS_RELU]:
                exec_dict[B_tensor] = bias

        elif config.conv_type == ConvType.DGRAD:
            # DGRAD: dX = conv_dgrad(dY, W)
            # Y contains dY (input), W contains weights, X is where we store dX (output)
            dY_tensor = graph.tensor(
                name="dY", dim=list(Y.size()), stride=list(Y.stride()),
                data_type=convert_to_cudnn_type(config.y_dtype)
            )
            W_tensor = graph.tensor(
                name="W", dim=list(W.size()), stride=list(W.stride()),
                data_type=convert_to_cudnn_type(config.w_dtype)
            )

            conv_output = graph.conv_dgrad(
                loss=dY_tensor,
                filter=W_tensor,
                padding=list(config.padding),
                stride=list(config.stride),
                dilation=list(config.dilation),
            )
            # Must set output dimensions explicitly for dgrad (cuDNN can't infer them)
            conv_output.set_output(True).set_dim(list(X.size())).set_stride(list(X.stride()))

            # Execution dict: Y(dY),W are inputs, X(dX) is output
            exec_dict = {dY_tensor: Y, W_tensor: W, conv_output: X}

        else:  # WGRAD
            # WGRAD: dW = conv_wgrad(X, dY)
            # X contains input, Y contains dY, W is where we store dW (output)
            X_tensor = graph.tensor(
                name="X", dim=list(X.size()), stride=list(X.stride()),
                data_type=convert_to_cudnn_type(config.x_dtype)
            )
            dY_tensor = graph.tensor(
                name="dY", dim=list(Y.size()), stride=list(Y.stride()),
                data_type=convert_to_cudnn_type(config.y_dtype)
            )

            conv_output = graph.conv_wgrad(
                image=X_tensor,
                loss=dY_tensor,
                padding=list(config.padding),
                stride=list(config.stride),
                dilation=list(config.dilation),
            )
            conv_output.set_output(True).set_dim(list(W.size())).set_stride(list(W.stride()))

            # Execution dict: X,Y(dY) are inputs, W(dW) is output
            exec_dict = {X_tensor: X, dY_tensor: Y, conv_output: W}

        # Validate and build
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans()

        # Allocate workspace and fill with garbage to catch uninitialized memory bugs
        workspace_size = graph.get_workspace_size()
        workspace = torch.empty(workspace_size, device='cuda', dtype=torch.uint8)
        if workspace_size > 0:
            # Fill with random garbage + some NaN patterns to test proper workspace init
            workspace.random_(0, 256)
            # Sprinkle in NaN bit patterns (0x7FC00000 for float32 NaN)
            nan_mask = torch.rand(workspace_size, device='cuda') < 0.1
            workspace[nan_mask] = 0xFF

        graph.execute(exec_dict, workspace, handle=cudnn_handle)
        torch.cuda.synchronize()

        return True, "Success"

    except cudnn.cudnnGraphNotSupportedError as e:
        return False, f"Graph not supported: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def compare_results(actual: torch.Tensor, ref: torch.Tensor, _dtype: torch.dtype,
                    num_diffs: int = 10) -> Tuple[bool, str]:
    """Compare cuDNN result with reference."""
    # Base tolerances - TF32/FP16/BF16 all have similar effective precision
    # cuDNN uses TF32 for FP32 tensor core ops
    # _dtype kept for potential future per-dtype tolerance tuning
    rtol, atol = 1e-2, 1e-2

    passed, _, msg = compare_tensors(actual, ref, rtol=rtol, atol=atol, num_diffs=num_diffs)
    return passed, msg


def _compute_elem_counts(config: ConvConfig) -> Tuple[int, int, int]:
    """Compute element counts for X, W, Y from config dimensions.

    This works before create_tensors() has been called (which populates
    config.x_elems etc.), so it can be used for pre-flight memory checks.
    """
    x_elems = config.batch * config.in_channels * math.prod(config.input_spatial)
    w_elems = config.out_channels * (config.in_channels // config.groups) * math.prod(config.filter_spatial)
    y_elems = config.batch * config.out_channels * math.prod(config.output_spatial)
    return x_elems, w_elems, y_elems


def estimate_memory_mb(config: ConvConfig) -> float:
    """Estimate GPU memory usage in MB for tensors (X, W, Y, Y_ref, bias)."""
    dtype_bytes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }
    elem_size = dtype_bytes.get(config.x_dtype, 4)

    # Use _compute_elem_counts so estimate works before create_tensors() has run
    # (config.x_elems etc. default to 0 until create_tensors populates them)
    x_elems, w_elems, y_elems = _compute_elem_counts(config)

    # X, W, Y tensors + Y_ref (float32 for comparison)
    x_bytes = x_elems * elem_size
    w_bytes = w_elems * elem_size
    y_bytes = y_elems * elem_size
    y_ref_bytes = y_elems * 4  # float32

    total = x_bytes + w_bytes + y_bytes + y_ref_bytes

    # Bias (only for FPROP with bias epilogue; out_channels elements)
    if config.conv_type == ConvType.FPROP and config.epilogue in [EpilogueType.BIAS, EpilogueType.BIAS_RELU]:
        total += config.out_channels * elem_size

    return total / (1024 * 1024)

def format_test_header(config: ConvConfig, test_num: int, total_tests: int, test_name: str) -> str:
    """Format test header similar to matmul fuzzer."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu_info = f"{get_gpu_arch()} ({get_sm_count()} SM-s, {get_gpu_name()})"

    spatial_str = "2D" if config.spatial_dims == 2 else "3D"
    mem_mb = estimate_memory_mb(config)

    # Describe groups
    if config.groups == 1:
        groups_str = "1 (standard conv)"
    elif config.groups == config.in_channels == config.out_channels:
        groups_str = f"{config.groups} (depthwise)"
    else:
        groups_str = f"{config.groups} (grouped)"

    lines = [
        "",  # Newline to separate from pytest's test name line
        "=" * 90,
        f"#### Test #{test_num} of {total_tests} at {timestamp} ",
        "",
        f"test_name        = {test_name}",
        f"platform_info    = {gpu_info}, cudnn_ver={cudnn.backend_version()}",
        f"rng_data_seed    = {config.rng_seed}",
        f"conv_type        = {conv_type_name(config.conv_type)} ({spatial_str})",
        f"basic_dims       = [N={config.batch}, C_in={config.in_channels}, C_out={config.out_channels}]",
        f"groups           = {groups_str}",
        f"input_spatial    = {config.input_spatial}",
        f"filter_spatial   = {config.filter_spatial}",
        f"output_spatial   = {config.output_spatial}",
        f"padding          = {config.padding}",
        f"stride           = {config.stride}",
        f"dilation         = {config.dilation}",
        f"x(N,C,spatial)   = dim={config.x_shape}, strides={config.x_strides}, elems={config.x_elems}, type={config.x_dtype}",
        f"w(K,C/g,spatial) = dim={config.w_shape}, strides={config.w_strides}, elems={config.w_elems}, type={config.w_dtype}",
        f"y(N,K,spatial)   = dim={config.y_shape}, strides={config.y_strides}, elems={config.y_elems}, type={config.y_dtype}",
    ]

    if config.bias_shape:
        lines.append(f"bias(1,K,1...)   = dim={config.bias_shape}, strides={config.bias_strides}, elems={config.bias_elems}, type={config.y_dtype}")

    lines.extend([
        f"epilogue         = {epilogue_name(config.epilogue)}",
        f"est_memory       = {mem_mb:.1f} MB",
        f"repro_cmd        = pytest -vv -s -rA {__file__}::test_repro --repro \"{config.to_repro_dict()}\"",
        " ",
    ])

    return "\n".join(lines)


# ============================================================================
# Pytest Fixtures and Configuration
# ============================================================================
# Note: pytest_addoption is defined in conftest.py
# Options used: --seed, --num-tests, --diffs, --repro

@pytest.fixture
def num_diffs(request):
    return request.config.getoption("--diffs")


# ============================================================================
# Test Parameter Generation
# ============================================================================

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
    }
    dt = dtype_short.get(config.x_dtype, 'unk')
    spatial = '2d' if config.spatial_dims == 2 else '3d'
    conv = conv_type_name(config.conv_type)[:2]  # fp, dg, wg
    epi = epilogue_name(config.epilogue)[:4]
    # Include groups in ID if > 1
    groups_str = f"_g{config.groups}" if config.groups > 1 else ""
    # Example: t1_N2_C64x128_32x32_f16_2d_fp_none or t1_N2_C64x128_g4_32x32_f16_2d_fp_none
    spatial_str = 'x'.join(str(s) for s in config.input_spatial)
    return f"{prefix}{test_num}_N{config.batch}_C{config.in_channels}x{config.out_channels}{groups_str}_{spatial_str}_{dt}_{spatial}_{conv}_{epi}"


# Pre-generated test parameter lists
DEFAULT_NUM_TESTS = 1024
DEFAULT_SEED_L0 = 42
DEFAULT_SEED_L1 = 12345

TEST_PARAMS_L0 = tlist_with_configs(num_tests=DEFAULT_NUM_TESTS, rng_seed=DEFAULT_SEED_L0, allow_unaligned=False)
TEST_PARAMS_L1 = tlist_with_configs(num_tests=DEFAULT_NUM_TESTS, rng_seed=DEFAULT_SEED_L1, allow_unaligned=True)

SKIP_TEST_NUMS_L0 = {}

# Maximum retries for unsupported grouped convolution configs
MAX_RETRIES = 10


# ============================================================================
# Test Functions
# ============================================================================

def _regenerate_config(current_seed: int, attempt: int, allow_unaligned: bool,
                       force_depthwise: bool, force_grouped: bool, original_conv_type: ConvType,
                       depthwise_channels_list: Optional[list]) -> Tuple[ConvConfig, int]:
    """Helper to regenerate a config with a new seed.

    Args:
        force_depthwise: If True, regenerate a depthwise config
        force_grouped: If True (and not force_depthwise), regenerate a grouped config
        If both are False, regenerate a standard config (no grouping constraint)
    """
    new_seed = current_seed + 1000 * (attempt + 1)
    generator = ConfigGenerator(new_seed, allow_unaligned=allow_unaligned)
    if force_depthwise:
        new_config = generator.generate(
            force_depthwise=True,
            force_conv_type=original_conv_type,
            depthwise_channels_list=depthwise_channels_list
        )
    elif force_grouped:
        new_config = generator.generate(force_grouped=True, force_conv_type=original_conv_type)
    else:
        # Standard config - no grouping constraint
        new_config = generator.generate(force_conv_type=original_conv_type)
    return new_config, new_seed


def run_conv_test_with_retry(config: ConvConfig, config_seed: int, test_num: int, total_tests: int,
                              test_name_prefix: str, cudnn_handle, num_diffs, allow_unaligned: bool = False,
                              force_depthwise: bool = False, depthwise_channels_list: Optional[list] = None) -> bool:
    """
    Run a convolution test with retry logic for unsupported configs and OOM.

    If a grouped/depthwise convolution config (groups > 1) fails check_support() or OOM,
    regenerate a new config and retry up to MAX_RETRIES times.

    Args:
        force_depthwise: If True, retries will generate depthwise configs instead of grouped
        depthwise_channels_list: Channel options for depthwise retries

    Returns True if test passed, raises pytest.fail if numerical mismatch.
    """
    current_config = config
    current_seed = config_seed
    # Memory threshold: skip if estimated memory exceeds this fraction of available
    MEMORY_THRESHOLD = 0.7

    for attempt in range(MAX_RETRIES + 1):
        X, W, Y, bias = None, None, None, None
        ref = None

        try:
            # Pre-flight memory check
            available_mb = get_available_gpu_memory_mb()
            estimated_mb = estimate_memory_mb(current_config)

            if estimated_mb > available_mb * MEMORY_THRESHOLD:
                print(f"%%%% Memory check: need ~{estimated_mb:.1f}MB, available {available_mb:.1f}MB (threshold {MEMORY_THRESHOLD*100:.0f}%)")
                if attempt < MAX_RETRIES:
                    print(f"%%%% Regenerating smaller config (attempt {attempt + 1}/{MAX_RETRIES})...")
                    # Preserve grouping type: depthwise > grouped > standard
                    is_grouped = config.groups > 1 and not force_depthwise
                    current_config, current_seed = _regenerate_config(
                        current_seed, attempt, allow_unaligned, force_depthwise, is_grouped,
                        config.conv_type, depthwise_channels_list
                    )
                    continue
                else:
                    pytest.skip(f"Insufficient GPU memory: need ~{estimated_mb:.1f}MB, available {available_mb:.1f}MB")
                    return False

            # Create tensors for current config
            rng = random.Random(current_seed)
            X, W, Y, bias = create_tensors(current_config, rng)

            # Print test header
            if attempt == 0:
                test_name = f"{test_name_prefix}[{make_test_id((test_num, total_tests, current_seed, current_config))}]"
            else:
                test_name = f"{test_name_prefix}[{make_test_id((test_num, total_tests, current_seed, current_config))}] (retry {attempt})"
            print(format_test_header(current_config, test_num, total_tests, test_name))

            # Run cuDNN
            success, msg = run_cudnn_conv(current_config, X, W, Y, bias, cudnn_handle)

            if not success:
                print(f"%%%% cuDNN execution failed: {msg}")

                # For grouped/depthwise convolutions, retry with a new config
                if current_config.groups > 1 and attempt < MAX_RETRIES:
                    config_type = "Depthwise" if force_depthwise else "Grouped"
                    print(f"%%%% {config_type} convolution not supported, regenerating config (attempt {attempt + 1}/{MAX_RETRIES})...")
                    # force_grouped=True since we know groups > 1 (unless force_depthwise)
                    current_config, current_seed = _regenerate_config(
                        current_seed, attempt, allow_unaligned, force_depthwise, not force_depthwise,
                        config.conv_type, depthwise_channels_list
                    )
                    continue
                elif current_config.groups > 1:
                    # Grouped/depthwise config exhausted all retries - fail to ensure they are tested
                    config_type = "Depthwise" if force_depthwise else "Grouped"
                    pytest.fail(f"{config_type} convolution (groups={current_config.groups}) failed after {MAX_RETRIES} retries: {msg}")
                    return False
                else:
                    # Non-grouped config not supported
                    pytest.skip(f"cuDNN not supported: {msg}")
                    return False

            # Compute reference and compare
            ref = compute_reference(current_config, X, W, Y, bias)

            # Determine which tensor to compare based on conv_type
            if current_config.conv_type == ConvType.FPROP:
                actual, dtype, name = Y, current_config.y_dtype, "Y"
            elif current_config.conv_type == ConvType.DGRAD:
                actual, dtype, name = X, current_config.x_dtype, "dX"
            else:  # WGRAD
                actual, dtype, name = W, current_config.w_dtype, "dW"

            passed, compare_msg = compare_results(actual, ref, dtype, num_diffs)

            if passed:
                print(f"%%%% Numerical divergence of '{name}' within limits ({compare_msg})")
                print("@@@@ Overall result: PASSED, everything looks good!")
                return True
            else:
                print(f"%%%% {compare_msg}")
                print("@@@@ Overall result: FAILED, numerical mismatch!")
                pytest.fail(f"Numerical mismatch: {compare_msg}")
                return False

        except torch.cuda.OutOfMemoryError:
            # OOM during tensor creation or execution - try to recover
            torch.cuda.empty_cache()
            print(f"%%%% CUDA out of memory!")
            if attempt < MAX_RETRIES:
                print(f"%%%% Regenerating smaller config (attempt {attempt + 1}/{MAX_RETRIES})...")
                # Preserve grouping type: depthwise > grouped > standard
                is_grouped = config.groups > 1 and not force_depthwise
                current_config, current_seed = _regenerate_config(
                    current_seed, attempt, allow_unaligned, force_depthwise, is_grouped,
                    config.conv_type, depthwise_channels_list
                )
                continue
            else:
                pytest.skip(f"CUDA out of memory after {MAX_RETRIES} retries")
                return False

        finally:
            # Explicit cleanup to prevent GPU memory accumulation
            if X is not None:
                del X
            if W is not None:
                del W
            if Y is not None:
                del Y
            if bias is not None:
                del bias
            if ref is not None:
                del ref
            torch.cuda.empty_cache()

    # Exhausted all retries - this should not be reached due to the check inside the loop,
    # but handle it just in case
    if current_config.groups > 1:
        pytest.fail(f"Grouped convolution testing failed after {MAX_RETRIES} retries")
    else:
        pytest.skip(f"Failed to find supported config after {MAX_RETRIES} retries")
    return False


@pytest.mark.L0
@pytest.mark.parametrize("test_num,total_tests,config_seed,config", TEST_PARAMS_L0,
                        ids=[make_test_id(p) for p in TEST_PARAMS_L0])
def test_conv_random_L0(test_num: int, total_tests: int, config_seed: int, config: ConvConfig, cudnn_handle, num_diffs, request):
    """Random convolution tests (fprop/dgrad/wgrad) with aligned dimensions (L0)."""
    # Skip known failing tests
    if test_num in SKIP_TEST_NUMS_L0:
        pytest.skip(f"Known failing test (dgrad f32 precision issue)")

    run_conv_test_with_retry(
        config=config,
        config_seed=config_seed,
        test_num=test_num,
        total_tests=total_tests,
        test_name_prefix="test_conv_random_L0",
        cudnn_handle=cudnn_handle,
        num_diffs=num_diffs,
        allow_unaligned=False
    )


@pytest.mark.L1
@pytest.mark.parametrize("test_num,total_tests,config_seed,config", TEST_PARAMS_L1,
                        ids=[make_test_id(p, prefix="u") for p in TEST_PARAMS_L1])
def test_conv_random_L1(test_num: int, total_tests: int, config_seed: int, config: ConvConfig, cudnn_handle, num_diffs, request):
    """Random convolution tests (fprop/dgrad/wgrad) with unaligned dimensions (L1)."""
    run_conv_test_with_retry(
        config=config,
        config_seed=config_seed,
        test_num=test_num,
        total_tests=total_tests,
        test_name_prefix="test_conv_random_L1",
        cudnn_handle=cudnn_handle,
        num_diffs=num_diffs,
        allow_unaligned=True
    )


@pytest.mark.L0
def test_repro(cudnn_handle, num_diffs, request):
    """Reproduce a specific test case from repro dict."""
    repro_str = request.config.getoption("--repro")
    if repro_str is None:
        pytest.skip("No --repro argument provided")
        return

    import ast
    repro = ast.literal_eval(repro_str)

    # Reconstruct config from repro dict
    dtype_map = {
        'torch.float16': torch.float16,
        'torch.bfloat16': torch.bfloat16,
        'torch.float32': torch.float32,
        'torch.int8': torch.int8,
    }

    config = ConvConfig(
        spatial_dims=repro['spatial_dims'],
        batch=repro['batch'],
        in_channels=repro['in_channels'],
        out_channels=repro['out_channels'],
        groups=repro.get('groups', 1),  # Default to 1 for backward compatibility
        input_spatial=tuple(repro['input_spatial']),
        filter_spatial=tuple(repro['filter_spatial']),
        padding=tuple(repro['padding']),
        stride=tuple(repro['stride']),
        dilation=tuple(repro['dilation']),
        conv_type=ConvType(repro['conv_type']),
        x_dtype=dtype_map[repro['x_dtype']],
        w_dtype=dtype_map[repro['w_dtype']],
        y_dtype=dtype_map[repro['y_dtype']],
        epilogue=EpilogueType(repro['epilogue']),
        rng_seed=repro['rng_seed'],
    )

    # Create tensors
    rng = random.Random(config.rng_seed)
    X, W, Y, bias = create_tensors(config, rng)
    ref = None

    try:
        # Print test header and flush to ensure repro info is saved before any potential crash
        print(format_test_header(config, 1, 1, "test_repro"))
        sys.stdout.flush()

        # Run cuDNN
        success, msg = run_cudnn_conv(config, X, W, Y, bias, cudnn_handle)

        if not success:
            print(f"%%%% cuDNN execution failed: {msg}")
            pytest.fail(f"cuDNN failed: {msg}")
            return

        # Compute reference and compare
        ref = compute_reference(config, X, W, Y, bias)

        # Determine which tensor to compare based on conv_type
        if config.conv_type == ConvType.FPROP:
            actual, dtype, name = Y, config.y_dtype, "Y"
        elif config.conv_type == ConvType.DGRAD:
            actual, dtype, name = X, config.x_dtype, "dX"
        else:  # WGRAD
            actual, dtype, name = W, config.w_dtype, "dW"

        passed, compare_msg = compare_results(actual, ref, dtype, num_diffs)

        if passed:
            print(f"%%%% Numerical divergence of '{name}' within limits ({compare_msg})")
            print("@@@@ Overall result: PASSED, everything looks good!")
        else:
            print(f"%%%% {compare_msg}")
            print("@@@@ Overall result: FAILED, numerical mismatch!")
            pytest.fail(f"Numerical mismatch: {compare_msg}")

        # Print hash and stats for determinism verification
        print_tensor_stats(actual, tag=f"{name}_gpu")
    finally:
        # Explicit cleanup to prevent GPU memory accumulation
        del X, W, Y
        if bias is not None:
            del bias
        if ref is not None:
            del ref
        torch.cuda.empty_cache()


# ============================================================================
# Depthwise Convolution Tests
# ============================================================================

def tlist_with_depthwise_configs(*, num_tests: int, rng_seed: int, allow_unaligned: bool = False,
                                  depthwise_channels: Optional[list] = None, even_conv_distribution: bool = False):
    """
    Generate list of test parameters for depthwise convolution tests.

    Args:
        num_tests: Number of tests to generate
        rng_seed: Random seed for reproducibility
        allow_unaligned: If True, use unaligned dimensions (L1 style)
        depthwise_channels: List of channel values to choose from for depthwise
        even_conv_distribution: If True, distribute conv types evenly (33% each)
    """
    rng = random.Random(rng_seed)
    params = []

    # For even distribution, cycle through conv types
    conv_types = [ConvType.FPROP, ConvType.DGRAD, ConvType.WGRAD]

    for i in range(num_tests):
        config_seed = rng.randint(65536, 2**31 - 1)
        generator = ConfigGenerator(config_seed, allow_unaligned=allow_unaligned)

        # Determine conv type
        if even_conv_distribution:
            conv_type = conv_types[i % 3]
        else:
            conv_type = None  # Let generator choose randomly

        config = generator.generate(
            force_depthwise=True,
            force_conv_type=conv_type,
            depthwise_channels_list=depthwise_channels
        )
        params.append((i + 1, num_tests, config_seed, config))
    return params


# Depthwise channel options
def _make_depthwise_channels(include_edge_cases: bool) -> list:
    """
    Generate depthwise channel options.

    Categories:
    - Powers of 2: Standard aligned values
    - 2^k ± 1: Off-by-one edge cases (memory alignment boundaries)
    - Small primes: Minimal channel counts
    """
    channels = [8, 16, 32, 64, 128]  # Powers of 2

    if include_edge_cases:
        channels.extend([2, 3, 5, 7])  # Small primes
        for k in [4, 5, 6, 7]:  # 16, 32, 64, 128
            channels.extend([2**k - 1, 2**k + 1])  # 2^k ± 1
        channels.extend([256])

    return sorted(set(channels))

# Generated at module load time - deterministic
DEPTHWISE_CHANNELS_L0 = _make_depthwise_channels(include_edge_cases=False)
DEPTHWISE_CHANNELS_L1 = _make_depthwise_channels(include_edge_cases=True)

# Pre-generated depthwise test parameters
DEFAULT_SEED_DEPTHWISE_L0 = 54321
DEFAULT_SEED_DEPTHWISE_L1 = 98765

TEST_PARAMS_DEPTHWISE_L0 = tlist_with_depthwise_configs(
    num_tests=16,
    rng_seed=DEFAULT_SEED_DEPTHWISE_L0,
    allow_unaligned=False,
    depthwise_channels=DEPTHWISE_CHANNELS_L0,
    even_conv_distribution=True
)

TEST_PARAMS_DEPTHWISE_L1 = tlist_with_depthwise_configs(
    num_tests=32,
    rng_seed=DEFAULT_SEED_DEPTHWISE_L1,
    allow_unaligned=True,
    depthwise_channels=DEPTHWISE_CHANNELS_L1,
    even_conv_distribution=True
)

@pytest.mark.L0
@pytest.mark.parametrize("test_num,total_tests,config_seed,config", TEST_PARAMS_DEPTHWISE_L0,
                        ids=[make_test_id(p, prefix="dw") for p in TEST_PARAMS_DEPTHWISE_L0])
def test_conv_depthwise_L0(test_num: int, total_tests: int, config_seed: int, config: ConvConfig, cudnn_handle, num_diffs, request):
    """Depthwise convolution tests (groups = in_channels = out_channels) (L0)."""
    # Verify this is a depthwise configuration
    assert config.groups == config.in_channels == config.out_channels, \
        f"Expected depthwise config (groups=in_channels=out_channels), got groups={config.groups}, in={config.in_channels}, out={config.out_channels}"

    run_conv_test_with_retry(
        config=config,
        config_seed=config_seed,
        test_num=test_num,
        total_tests=total_tests,
        test_name_prefix="test_conv_depthwise_L0",
        cudnn_handle=cudnn_handle,
        num_diffs=num_diffs,
        allow_unaligned=False,
        force_depthwise=True,
        depthwise_channels_list=DEPTHWISE_CHANNELS_L0
    )


@pytest.mark.L1
@pytest.mark.parametrize("test_num,total_tests,config_seed,config", TEST_PARAMS_DEPTHWISE_L1,
                        ids=[make_test_id(p, prefix="dw") for p in TEST_PARAMS_DEPTHWISE_L1])
def test_conv_depthwise_L1(test_num: int, total_tests: int, config_seed: int, config: ConvConfig, cudnn_handle, num_diffs, request):
    """Depthwise convolution tests with unaligned dimensions and varied channels (L1)."""
    # Verify this is a depthwise configuration
    assert config.groups == config.in_channels == config.out_channels, \
        f"Expected depthwise config (groups=in_channels=out_channels), got groups={config.groups}, in={config.in_channels}, out={config.out_channels}"

    run_conv_test_with_retry(
        config=config,
        config_seed=config_seed,
        test_num=test_num,
        total_tests=total_tests,
        test_name_prefix="test_conv_depthwise_L1",
        cudnn_handle=cudnn_handle,
        num_diffs=num_diffs,
        allow_unaligned=True,
        force_depthwise=True,
        depthwise_channels_list=DEPTHWISE_CHANNELS_L1
    )
