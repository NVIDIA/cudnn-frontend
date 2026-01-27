"""
Convolution Fuzzer - Randomized stress testing for cuDNN convolution operations.

This fuzzer tests convolution operations with randomized:
- Shapes (batch, channels, spatial dimensions)
- Spatial dimensions (2D or 3D)
- Data types (fp16, bf16, fp32, int8)
- Convolution parameters (padding, stride, dilation)
- Operation types (fprop, dgrad, wgrad)
- Epilogues (none, bias, relu, bias_relu)

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

    def generate(self) -> ConvConfig:
        """Generate a random convolution configuration."""
        spatial_dims = self.random_spatial_dims()

        batch = self.random_batch()
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
        conv_type = self.random_conv_type()

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
      W: (C_out, C_in, filter...)    - weight/filter shape
      Y: (N, C_out, output_spatial...) - output shape

    Meaning varies by conv_type:
      FPROP: X=input, W=weights, Y=output       (compute Y from X,W)
      DGRAD: X=dX(output), W=weights, Y=dY(input)  (compute dX from dY,W)
      WGRAD: X=input, W=dW(output), Y=dY(input)    (compute dW from X,dY)
    """
    torch_rng = torch.Generator(device='cuda')
    torch_rng.manual_seed(config.rng_seed)

    # Compute shapes (same for all conv types)
    x_shape = (config.batch, config.in_channels) + config.input_spatial
    w_shape = (config.out_channels, config.in_channels) + config.filter_spatial
    y_shape = (config.batch, config.out_channels) + config.output_spatial

    # Use PyTorch's native channels_last memory format for proper cuDNN compatibility
    if config.spatial_dims == 2:
        memory_format = torch.channels_last
    else:  # 3D
        memory_format = torch.channels_last_3d

    # Create tensors - which ones are input (random) vs output (garbage) depends on conv_type
    # Output tensors are filled with garbage (random + NaNs) to catch bugs where cuDNN
    # doesn't write all output locations
    if config.conv_type == ConvType.FPROP:
        # FPROP: X,W are inputs, Y is output
        X = torch.empty(x_shape, device='cuda', dtype=config.x_dtype).to(memory_format=memory_format)
        X.normal_(mean=0.5, std=0.1, generator=torch_rng)
        W = torch.empty(w_shape, device='cuda', dtype=config.w_dtype).to(memory_format=memory_format)
        W.normal_(mean=0.5, std=0.1, generator=torch_rng)
        Y = torch.empty(y_shape, device='cuda', dtype=config.y_dtype).to(memory_format=memory_format)
        fill_with_garbage(Y)  # Output - fill with garbage

    elif config.conv_type == ConvType.DGRAD:
        # DGRAD: Y(dY),W are inputs, X(dX) is output
        Y = torch.empty(y_shape, device='cuda', dtype=config.y_dtype).to(memory_format=memory_format)
        Y.normal_(mean=0.5, std=0.1, generator=torch_rng)  # dY - gradient from upstream
        W = torch.empty(w_shape, device='cuda', dtype=config.w_dtype).to(memory_format=memory_format)
        W.normal_(mean=0.5, std=0.1, generator=torch_rng)  # weights
        X = torch.empty(x_shape, device='cuda', dtype=config.x_dtype).to(memory_format=memory_format)
        fill_with_garbage(X)  # dX output - fill with garbage

    else:  # WGRAD
        # WGRAD: X,Y(dY) are inputs, W(dW) is output
        X = torch.empty(x_shape, device='cuda', dtype=config.x_dtype).to(memory_format=memory_format)
        X.normal_(mean=0.5, std=0.1, generator=torch_rng)  # input image
        Y = torch.empty(y_shape, device='cuda', dtype=config.y_dtype).to(memory_format=memory_format)
        Y.normal_(mean=0.5, std=0.1, generator=torch_rng)  # dY - gradient from upstream
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
        bias = torch.empty(bias_shape, device='cuda', dtype=config.y_dtype).contiguous()
        bias.normal_(mean=0.0, std=0.1, generator=torch_rng)

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
                    dilation=config.dilation
                )
            else:
                ref = torch.nn.functional.conv3d(
                    X_f, W_f,
                    padding=config.padding,
                    stride=config.stride,
                    dilation=config.dilation
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

        try:
            if config.spatial_dims == 2:
                dummy_Y = torch.nn.functional.conv2d(
                    dummy_X, W_f,
                    padding=config.padding,
                    stride=config.stride,
                    dilation=config.dilation
                )
            else:
                dummy_Y = torch.nn.functional.conv3d(
                    dummy_X, W_f,
                    padding=config.padding,
                    stride=config.stride,
                    dilation=config.dilation
                )

            # Backward pass to get dX
            dummy_Y.backward(dY_f)
            dX_ref = dummy_X.grad.clone()

            return dX_ref.to(config.x_dtype)
        finally:
            del dY_f, W_f, dummy_X, dummy_Y

    else:  # WGRAD
        # WGRAD: dW = conv_wgrad(X, dY)
        # X contains input, Y contains dY (gradient from upstream)
        # We compute dW (gradient w.r.t. weights)
        X_f = X.to(compute_dtype).contiguous()
        dY_f = Y.to(compute_dtype).contiguous()

        # Use autograd to compute the reference
        # Create a dummy weight and run forward, then backward to get dW
        dummy_W = torch.zeros(config.w_shape, device='cuda', dtype=compute_dtype, requires_grad=True)

        try:
            if config.spatial_dims == 2:
                dummy_Y = torch.nn.functional.conv2d(
                    X_f, dummy_W,
                    padding=config.padding,
                    stride=config.stride,
                    dilation=config.dilation
                )
            else:
                dummy_Y = torch.nn.functional.conv3d(
                    X_f, dummy_W,
                    padding=config.padding,
                    stride=config.stride,
                    dilation=config.dilation
                )

            # Backward pass to get dW
            dummy_Y.backward(dY_f)
            dW_ref = dummy_W.grad.clone()

            return dW_ref.to(config.w_dtype)
        finally:
            del X_f, dY_f, dummy_W, dummy_Y


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
    # (TF32 and FP16 have 10-bit mantissa, BF16 has 7-bit but we use same tolerance)
    # cuDNN uses TF32 for FP32 tensor core ops
    # _dtype kept for potential future per-dtype tolerance tuning
    rtol, atol = 1e-2, 1e-2

    if ref.shape != actual.shape:
        return False, f"Shape mismatch: actual={actual.shape}, ref={ref.shape}"

    # Compare
    actual_f = actual.to(torch.float32).contiguous()
    ref_f = ref.to(torch.float32).contiguous()

    diff = torch.abs(actual_f - ref_f)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Relative difference
    denom = torch.maximum(torch.abs(ref_f), torch.tensor(1e-6, device='cuda'))
    rel_diff = diff / denom
    max_rel_diff = rel_diff.max().item()

    # Find mismatches - element fails if it exceeds BOTH tolerances
    # mismatch_mask = (diff > atol) & (rel_diff > rtol)
    mismatch_mask =  (diff > torch.abs(atol + rtol * ref_f))
    mismatch_indices = torch.nonzero(mismatch_mask)
    num_mismatches = mismatch_indices.shape[0]

    # Pass if no elements fail both tolerance checks
    passed = num_mismatches == 0

    if passed:
        return True, f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, max_rel_diff={max_rel_diff:.2e}"
    else:
        msg = f"MISMATCH: {num_mismatches} elements differ (max_diff={max_diff:.2e}, max_rel_diff={max_rel_diff:.2e})\n"
        for i in range(min(num_diffs, num_mismatches)):
            idx = tuple(mismatch_indices[i].tolist())
            act_val = actual_f[idx].item()
            ref_val = ref_f[idx].item()
            d = diff[idx].item()
            msg += f"  [{idx}]: actual={act_val:.6f}, expected={ref_val:.6f}, diff={d:.2e} tol={atol + rtol * ref_f[idx].item():.2e}\n"

        return False, msg


def estimate_memory_mb(config: ConvConfig) -> float:
    """Estimate GPU memory usage in MB for tensors (X, W, Y, Y_ref, bias)."""
    dtype_bytes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }
    elem_size = dtype_bytes.get(config.x_dtype, 4)

    # X, W, Y tensors + Y_ref (float32 for comparison)
    x_bytes = config.x_elems * elem_size
    w_bytes = config.w_elems * elem_size
    y_bytes = config.y_elems * elem_size
    y_ref_bytes = config.y_elems * 4  # float32

    total = x_bytes + w_bytes + y_bytes + y_ref_bytes
    if config.bias_elems:
        total += config.bias_elems * elem_size

    return total / (1024 * 1024)


def format_test_header(config: ConvConfig, test_num: int, total_tests: int, test_name: str) -> str:
    """Format test header similar to matmul fuzzer."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu_info = f"{get_gpu_arch()} ({get_sm_count()} SM-s, {get_gpu_name()})"

    spatial_str = "2D" if config.spatial_dims == 2 else "3D"
    mem_mb = estimate_memory_mb(config)

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
        f"input_spatial    = {config.input_spatial}",
        f"filter_spatial   = {config.filter_spatial}",
        f"output_spatial   = {config.output_spatial}",
        f"padding          = {config.padding}",
        f"stride           = {config.stride}",
        f"dilation         = {config.dilation}",
        f"x(N,C,spatial)   = dim={config.x_shape}, strides={config.x_strides}, elems={config.x_elems}, type={config.x_dtype}",
        f"w(K,C,spatial)   = dim={config.w_shape}, strides={config.w_strides}, elems={config.w_elems}, type={config.w_dtype}",
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
    # Example: t1_N2_C64x128_32x32_f16_2d_fp_none
    spatial_str = 'x'.join(str(s) for s in config.input_spatial)
    return f"{prefix}{test_num}_N{config.batch}_C{config.in_channels}x{config.out_channels}_{spatial_str}_{dt}_{spatial}_{conv}_{epi}"


# Pre-generated test parameter lists
DEFAULT_NUM_TESTS = 1024
DEFAULT_SEED_L0 = 42
DEFAULT_SEED_L1 = 12345

TEST_PARAMS_L0 = tlist_with_configs(num_tests=DEFAULT_NUM_TESTS, rng_seed=DEFAULT_SEED_L0, allow_unaligned=False)
TEST_PARAMS_L1 = tlist_with_configs(num_tests=DEFAULT_NUM_TESTS, rng_seed=DEFAULT_SEED_L1, allow_unaligned=True)

SKIP_TEST_NUMS_L0 = {}

# ============================================================================
# Test Functions
# ============================================================================

@pytest.mark.L0
@pytest.mark.parametrize("test_num,total_tests,config_seed,config", TEST_PARAMS_L0,
                        ids=[make_test_id(p) for p in TEST_PARAMS_L0])
def test_conv_random_L0_0(test_num: int, total_tests: int, config_seed: int, config: ConvConfig, cudnn_handle, num_diffs, request):
    """Random convolution tests (fprop/dgrad/wgrad) with aligned dimensions (L0)."""
    # Skip known failing tests
    if test_num in SKIP_TEST_NUMS_L0:
        pytest.skip(f"Known failing test (dgrad f32 precision issue)")

    # Create tensors
    rng = random.Random(config_seed)
    X, W, Y, bias = create_tensors(config, rng)
    ref = None

    try:
        # Print test header
        test_name = f"test_conv_random_L0_0[{make_test_id((test_num, total_tests, config_seed, config))}]"
        print(format_test_header(config, test_num, total_tests, test_name))

        # Run cuDNN
        success, msg = run_cudnn_conv(config, X, W, Y, bias, cudnn_handle)

        if not success:
            print(f"%%%% cuDNN execution failed: {msg}")
            pytest.skip(f"cuDNN not supported: {msg}")
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
    finally:
        # Explicit cleanup to prevent GPU memory accumulation
        del X, W, Y
        if bias is not None:
            del bias
        if ref is not None:
            del ref
        torch.cuda.empty_cache()


@pytest.mark.L1
@pytest.mark.parametrize("test_num,total_tests,config_seed,config", TEST_PARAMS_L1,
                        ids=[make_test_id(p, prefix="u") for p in TEST_PARAMS_L1])
def test_conv_random_L0_1(test_num: int, total_tests: int, config_seed: int, config: ConvConfig, cudnn_handle, num_diffs, request):
    """Random convolution tests (fprop/dgrad/wgrad) with unaligned dimensions (L1)."""
    # Create tensors
    rng = random.Random(config_seed)
    X, W, Y, bias = create_tensors(config, rng)
    ref = None

    try:
        # Print test header
        test_name = f"test_conv_random_L0_1[{make_test_id((test_num, total_tests, config_seed, config), prefix='u')}]"
        print(format_test_header(config, test_num, total_tests, test_name))

        # Run cuDNN
        success, msg = run_cudnn_conv(config, X, W, Y, bias, cudnn_handle)

        if not success:
            print(f"%%%% cuDNN execution failed: {msg}")
            pytest.skip(f"cuDNN not supported: {msg}")
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
    finally:
        # Explicit cleanup to prevent GPU memory accumulation
        del X, W, Y
        if bias is not None:
            del bias
        if ref is not None:
            del ref
        torch.cuda.empty_cache()


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
        # Print test header
        print(format_test_header(config, 1, 1, "test_repro"))

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
    finally:
        # Explicit cleanup to prevent GPU memory accumulation
        del X, W, Y
        if bias is not None:
            del bias
        if ref is not None:
            del ref
        torch.cuda.empty_cache()
