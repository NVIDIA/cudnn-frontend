from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

_TORCH_DTYPE_TO_CUDNN = {
    torch.float32: 0,  # CUDNN_DATA_FLOAT
    torch.float64: 1,  # CUDNN_DATA_DOUBLE
    torch.float16: 2,  # CUDNN_DATA_HALF
    torch.bfloat16: 9,  # CUDNN_DATA_BFLOAT16
}

_REQUIRED_CUDNN_VERSION = 92600
_LONG_FFT_MIN_LENGTH = 4096
_LONG_FFT_MAX_LENGTH = 1 << 24
_BUFFER_ALIGNMENT = 256


def _dtype_to_int(dtype: torch.dtype) -> int:
    if dtype not in _TORCH_DTYPE_TO_CUDNN:
        supported = ", ".join(str(value) for value in _TORCH_DTYPE_TO_CUDNN)
        raise ValueError(f"Unsupported dtype {dtype}. Supported: {supported}.")
    return _TORCH_DTYPE_TO_CUDNN[dtype]


def _require_fft_causal_conv1d_api() -> None:
    import cudnn

    required_symbols = (
        "fft_causal_conv1d_forward",
        "fft_causal_conv1d_backward",
        "long_fft_causal_conv1d_get_buffer_sizes",
        "long_fft_causal_conv1d_forward",
        "long_fft_causal_conv1d_backward",
    )
    missing = [name for name in required_symbols if not hasattr(cudnn, name)]
    if missing or cudnn.backend_version() < _REQUIRED_CUDNN_VERSION:
        raise RuntimeError("FFT causal conv1d requires cuDNN 9.26.0 or newer and Frontend bindings built against cuDNN 9.26.0 or newer.")


def _validate_tensors(x: Tensor, weight: Tensor) -> Tuple[int, int, int, int]:
    if x.dim() != 3 or weight.dim() != 2:
        raise ValueError(f"Expected x(3D) and weight(2D); got x.shape={x.shape}, weight.shape={weight.shape}.")
    if not (x.is_cuda and weight.is_cuda):
        raise ValueError(f"x and weight must be CUDA tensors; got x.device={x.device}, weight.device={weight.device}.")
    if x.device != weight.device:
        raise ValueError(f"x and weight must be on the same device; got x.device={x.device}, weight.device={weight.device}.")
    if x.dtype != weight.dtype:
        raise TypeError(f"x and weight must have the same dtype; got x.dtype={x.dtype}, weight.dtype={weight.dtype}.")
    _dtype_to_int(x.dtype)

    batch, dim, seq_len = x.shape
    weight_dim, kernel_size = weight.shape
    if batch <= 0 or dim <= 0 or seq_len <= 0 or kernel_size <= 0:
        raise ValueError(f"All tensor dimensions must be positive; got x.shape={x.shape}, weight.shape={weight.shape}.")
    if weight_dim != dim:
        raise ValueError(f"Channel mismatch: x has dim={dim}, but weight.shape[0]={weight_dim}.")
    return batch, dim, seq_len, kernel_size


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _compatible_filter_length(filter_length: int) -> int:
    return max(128, _next_power_of_two(filter_length))


def _cuda_arch(device: torch.device) -> int:
    properties = torch.cuda.get_device_properties(device)
    return properties.major * 100 + properties.minor * 10


def _medium_fft_is_available(filter_length: int, dtype: torch.dtype, device: torch.device) -> bool:
    # Match cuhyena's public Python selector: FP64 ends at K=4096; other
    # dtypes end at K=8192 before SM90 and K=16384 on SM90 or newer.
    if dtype == torch.float64:
        return filter_length <= 4096
    return filter_length <= (16384 if _cuda_arch(device) >= 900 else 8192)


def _validate_medium_shape(seq_len: int, kernel_size: int) -> None:
    if not _is_power_of_two(kernel_size) or not 128 <= kernel_size <= 16384:
        raise ValueError(f"Medium FFT kernel_size must be a power of two in [128, 16384]; got {kernel_size}.")
    if seq_len < kernel_size or seq_len % kernel_size != 0:
        raise ValueError(f"Medium FFT requires seq_len >= kernel_size and seq_len % kernel_size == 0; got {seq_len} and {kernel_size}.")


def _validate_long_shape(seq_len: int, kernel_size: int, dtype: torch.dtype, device: torch.device) -> None:
    if seq_len != kernel_size or not _is_power_of_two(kernel_size):
        raise ValueError(f"Long FFT requires power-of-two seq_len == kernel_size; got {seq_len} and {kernel_size}.")
    if not _LONG_FFT_MIN_LENGTH <= kernel_size <= _LONG_FFT_MAX_LENGTH:
        raise ValueError(f"Long FFT kernel_size must be in [{_LONG_FFT_MIN_LENGTH}, {_LONG_FFT_MAX_LENGTH}]; got {kernel_size}.")
    if dtype == torch.float64 and kernel_size == _LONG_FFT_MAX_LENGTH:
        raise ValueError("Long FFT FP64 supports kernel_size through 8388608.")
    if kernel_size == _LONG_FFT_MAX_LENGTH and _cuda_arch(device) < 900:
        raise ValueError("Long FFT kernel_size 16777216 requires compute capability 9.0 or newer.")


def _align_buffer_size(size: int) -> int:
    return ((size + _BUFFER_ALIGNMENT - 1) // _BUFFER_ALIGNMENT) * _BUFFER_ALIGNMENT


def _long_buffer_size_bytes(batch: int, dim: int, kernel_size: int, dtype: torch.dtype) -> int:
    # This is the public backend buffer-size formula used only by the fake
    # implementation. Runtime allocations always use the backend size query.
    fft_size = 2 * kernel_size
    bits = fft_size.bit_length() - 1
    m = 1 << ((bits + 1) // 2)
    n = 1 << (bits - ((bits + 1) // 2))
    intermediate_elements = 2 * (m // 2 + 1) * n
    scratch_element_size = 8 if dtype == torch.float64 else 4
    signal_bytes = _align_buffer_size(batch * dim * intermediate_elements * scratch_element_size)
    filter_bytes = _align_buffer_size(dim * intermediate_elements * scratch_element_size)
    return signal_bytes + filter_bytes


@torch.library.custom_op("cudnn::fft_causal_conv1d_fwd_primitive", mutates_args=(), device_types="cuda")
def _medium_fwd_primitive(x: Tensor, weight: Tensor) -> Tensor:
    _require_fft_causal_conv1d_api()
    batch, dim, seq_len, kernel_size = _validate_tensors(x, weight)
    _validate_medium_shape(seq_len, kernel_size)

    x = x.contiguous()
    weight = weight.contiguous()
    y = torch.empty_like(x)

    import cudnn

    cudnn.fft_causal_conv1d_forward(
        torch.cuda.current_stream(x.device).cuda_stream,
        x.data_ptr(),
        weight.data_ptr(),
        y.data_ptr(),
        batch,
        dim,
        seq_len,
        kernel_size,
        _dtype_to_int(x.dtype),
    )
    return y


@torch.library.register_fake("cudnn::fft_causal_conv1d_fwd_primitive")
def _medium_fwd_fake(x: Tensor, weight: Tensor) -> Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("cudnn::fft_causal_conv1d_bwd_primitive", mutates_args=(), device_types="cuda")
def _medium_bwd_primitive(grad_out: Tensor, x: Tensor, weight: Tensor) -> List[Tensor]:
    _require_fft_causal_conv1d_api()
    batch, dim, seq_len, kernel_size = _validate_tensors(x, weight)
    _validate_medium_shape(seq_len, kernel_size)
    if grad_out.shape != x.shape or grad_out.device != x.device or grad_out.dtype != x.dtype:
        raise ValueError("grad_out must match x in shape, device, and dtype.")

    x = x.contiguous()
    weight = weight.contiguous()
    grad_out = grad_out.contiguous()
    # The cuDNN medium FFT backward kernel fully overwrites both output
    # buffers; dweight accumulation happens in kernel-local state.
    grad_x = torch.empty_like(x)
    grad_weight = torch.empty_like(weight)

    import cudnn

    cudnn.fft_causal_conv1d_backward(
        torch.cuda.current_stream(x.device).cuda_stream,
        x.data_ptr(),
        weight.data_ptr(),
        grad_out.data_ptr(),
        grad_x.data_ptr(),
        grad_weight.data_ptr(),
        batch,
        dim,
        seq_len,
        kernel_size,
        _dtype_to_int(x.dtype),
    )
    return [grad_x, grad_weight]


@torch.library.register_fake("cudnn::fft_causal_conv1d_bwd_primitive")
def _medium_bwd_fake(grad_out: Tensor, x: Tensor, weight: Tensor) -> List[Tensor]:
    return [torch.empty_like(x), torch.empty_like(weight)]


def _medium_setup_context(ctx, inputs, output) -> None:
    del output
    x, weight = inputs
    ctx.save_for_backward(x, weight)


@torch.compiler.allow_in_graph
def _medium_autograd_bwd(ctx, grad_out: Tensor):
    x, weight = ctx.saved_tensors
    grad_x, grad_weight = torch.ops.cudnn.fft_causal_conv1d_bwd_primitive(grad_out, x, weight)
    return grad_x, grad_weight


torch.library.register_autograd(
    "cudnn::fft_causal_conv1d_fwd_primitive",
    _medium_autograd_bwd,
    setup_context=_medium_setup_context,
)


@torch.library.custom_op("cudnn::long_fft_causal_conv1d_fwd_primitive", mutates_args=(), device_types="cuda")
def _long_fwd_primitive(x: Tensor, weight: Tensor) -> List[Tensor]:
    _require_fft_causal_conv1d_api()
    batch, dim, seq_len, kernel_size = _validate_tensors(x, weight)
    _validate_long_shape(seq_len, kernel_size, x.dtype, x.device)

    x = x.contiguous()
    weight = weight.contiguous()
    y = torch.empty_like(x)

    import cudnn

    workspace_size, reserve_size = cudnn.long_fft_causal_conv1d_get_buffer_sizes(batch, dim, seq_len, kernel_size, _dtype_to_int(x.dtype))
    workspace = torch.empty(workspace_size, device=x.device, dtype=torch.uint8)
    reserve_space = torch.empty(reserve_size, device=x.device, dtype=torch.uint8)
    cudnn.long_fft_causal_conv1d_forward(
        torch.cuda.current_stream(x.device).cuda_stream,
        x.data_ptr(),
        weight.data_ptr(),
        y.data_ptr(),
        batch,
        dim,
        seq_len,
        kernel_size,
        _dtype_to_int(x.dtype),
        workspace.data_ptr(),
        workspace_size,
        reserve_space.data_ptr(),
        reserve_size,
    )
    # Reserve space contains transformed x/weight state and must survive until
    # autograd invokes the matching long backward primitive.
    return [y, reserve_space]


@torch.library.register_fake("cudnn::long_fft_causal_conv1d_fwd_primitive")
def _long_fwd_fake(x: Tensor, weight: Tensor) -> List[Tensor]:
    batch, dim, _ = x.shape
    kernel_size = weight.shape[1]
    reserve_size = _long_buffer_size_bytes(batch, dim, kernel_size, x.dtype)
    return [torch.empty_like(x), torch.empty(reserve_size, device=x.device, dtype=torch.uint8)]


@torch.library.custom_op("cudnn::long_fft_causal_conv1d_bwd_primitive", mutates_args=(), device_types="cuda")
def _long_bwd_primitive(grad_out: Tensor, x: Tensor, weight: Tensor, reserve_space: Tensor) -> List[Tensor]:
    _require_fft_causal_conv1d_api()
    batch, dim, seq_len, kernel_size = _validate_tensors(x, weight)
    _validate_long_shape(seq_len, kernel_size, x.dtype, x.device)
    if grad_out.shape != x.shape or grad_out.device != x.device or grad_out.dtype != x.dtype:
        raise ValueError("grad_out must match x in shape, device, and dtype.")
    if reserve_space.device != x.device or reserve_space.dtype != torch.uint8:
        raise ValueError("reserve_space must be a CUDA uint8 tensor on the same device as x.")

    grad_out = grad_out.contiguous()
    reserve_space = reserve_space.contiguous()
    # The cuDNN long FFT backward stages fully overwrite both output buffers.
    grad_x = torch.empty_like(x)
    grad_weight = torch.empty_like(weight)

    import cudnn

    workspace_size, reserve_size = cudnn.long_fft_causal_conv1d_get_buffer_sizes(batch, dim, seq_len, kernel_size, _dtype_to_int(x.dtype))
    if reserve_space.numel() < reserve_size:
        raise ValueError(f"reserve_space has {reserve_space.numel()} bytes, but the backend requires {reserve_size}.")
    workspace = torch.empty(workspace_size, device=x.device, dtype=torch.uint8)
    cudnn.long_fft_causal_conv1d_backward(
        torch.cuda.current_stream(x.device).cuda_stream,
        grad_out.data_ptr(),
        grad_x.data_ptr(),
        grad_weight.data_ptr(),
        batch,
        dim,
        seq_len,
        kernel_size,
        _dtype_to_int(x.dtype),
        workspace.data_ptr(),
        workspace_size,
        reserve_space.data_ptr(),
        reserve_size,
    )
    return [grad_x, grad_weight]


@torch.library.register_fake("cudnn::long_fft_causal_conv1d_bwd_primitive")
def _long_bwd_fake(grad_out: Tensor, x: Tensor, weight: Tensor, reserve_space: Tensor) -> List[Tensor]:
    return [torch.empty_like(x), torch.empty_like(weight)]


def _long_setup_context(ctx, inputs, output) -> None:
    x, weight = inputs
    _, reserve_space = output
    ctx.save_for_backward(x, weight, reserve_space)


@torch.compiler.allow_in_graph
def _long_autograd_bwd(ctx, grad_out):
    x, weight, reserve_space = ctx.saved_tensors
    grad_y = grad_out[0]
    grad_x, grad_weight = torch.ops.cudnn.long_fft_causal_conv1d_bwd_primitive(grad_y, x, weight, reserve_space)
    return grad_x, grad_weight


torch.library.register_autograd(
    "cudnn::long_fft_causal_conv1d_fwd_primitive",
    _long_autograd_bwd,
    setup_context=_long_setup_context,
)


def fft_causal_conv1d(x: Tensor, weight: Tensor) -> Tensor:
    r"""Compute depthwise causal 1D convolution with FFT kernels.

    The public signature, path selection, and right-padding behavior match
    cuhyena's ``fft_causal_conv1d(x, weight)`` wrapper. Medium FFT kernels are
    used when the filter fits their dtype/device limit; otherwise the full
    sequence long FFT path is used. Padding introduced for either raw backend
    API is removed from the returned tensor and from autograd gradients.

    The FFT weight convention is FIR order::

        y[t] = sum(weight[j] * x[t - j], j=0..kernel_size-1)

    This is reversed relative to :func:`causal_conv1d`, whose first stored
    weight multiplies the oldest sample in the causal window.

    Requires cuDNN 9.26.0 or newer. Supports FP16, BF16, FP32, and FP64 CUDA
    tensors and integrates with ``torch.autograd`` and ``torch.compile``.

    Args:
        x (torch.Tensor): Input tensor shaped ``(batch, dim, seq_len)``.
        weight (torch.Tensor): Per-channel filters shaped
            ``(dim, kernel_size)``.

    Returns:
        torch.Tensor: Output shaped ``(batch, dim, seq_len)``.
    """
    _, _, seq_len, weight_length = _validate_tensors(x, weight)

    if _medium_fft_is_available(weight_length, x.dtype, x.device):
        filter_length = _compatible_filter_length(weight_length)
        if weight_length != filter_length:
            weight = F.pad(weight, (0, filter_length - weight_length))

        padded_seq_len = max(filter_length, ((seq_len + filter_length - 1) // filter_length) * filter_length)
        if seq_len != padded_seq_len:
            x = F.pad(x, (0, padded_seq_len - seq_len))

        y = torch.ops.cudnn.fft_causal_conv1d_fwd_primitive(x, weight)
        return y[..., :seq_len]

    filter_length = max(_compatible_filter_length(weight_length), _compatible_filter_length(seq_len))
    _validate_long_shape(filter_length, filter_length, x.dtype, x.device)
    if weight_length != filter_length:
        weight = F.pad(weight, (0, filter_length - weight_length))
    if seq_len != filter_length:
        x = F.pad(x, (0, filter_length - seq_len))

    y, _ = torch.ops.cudnn.long_fft_causal_conv1d_fwd_primitive(x, weight)
    return y[..., :seq_len]
