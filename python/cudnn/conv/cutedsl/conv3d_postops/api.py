# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct APIs for the SM100 Conv3D post-operation kernels."""

import math
import threading

import cutlass
import torch
from cuda.bindings import driver as cuda
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

from cudnn._torch_stream import as_torch_stream, stream_context
from cudnn.api_base import APIBase, TensorDesc, TupleDict
from cudnn.conv.frost._cutedsl import requirement_error as cutedsl_requirement_error

TensorLike = torch.Tensor | TensorDesc
_CUTEDSL_MIN_VERSION = (4, 9)
_SUPPORTED_CHANNEL_PAIRS = {
    (160, 160),
    (160, 320),
    (320, 320),
    (320, 640),
    (640, 640),
}
_PLAN_CACHE: dict[tuple, APIBase] = {}
_PLAN_CACHE_LOCK = threading.Lock()
_PLAN_CACHE_CAPACITY = 128


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Compute dense row-major element strides for a tensor shape."""
    stride = []
    running = 1
    for extent in reversed(shape):
        stride.append(running)
        running *= extent
    return tuple(reversed(stride))


def _is_contiguous_stride(shape: tuple[int, ...], stride: tuple[int, ...]) -> bool:
    """Return whether ``stride`` is contiguous, ignoring singleton dimensions."""
    expected = _contiguous_stride(shape)
    return all(extent == 1 or actual == wanted for extent, actual, wanted in zip(shape, stride, expected))


def _require_alignment(tensor: torch.Tensor, name: str) -> None:
    """Reject tensor pointers that are not aligned to 16 bytes."""
    remainder = tensor.data_ptr() % 16
    if remainder:
        raise ValueError(f"{name} data pointer must be 16-byte aligned, got remainder {remainder}")


def _byte_span(tensor: torch.Tensor) -> tuple[int, int]:
    """Return the half-open byte range occupied by a contiguous tensor."""
    begin = tensor.data_ptr()
    return begin, begin + tensor.numel() * tensor.element_size()


def _record_streams(tensors: tuple[torch.Tensor | None, ...], stream: torch.cuda.Stream) -> None:
    """Keep tensor storage alive until its use on the consumer stream completes."""
    for tensor in tensors:
        if tensor is not None:
            tensor.record_stream(stream)


class _Conv3dPostOpsSm100(APIBase):
    """Shared validation, compilation, and launch mechanics."""

    def __init__(
        self,
        sample_input: TensorLike,
        sample_packed_weight: TensorLike,
        sample_bias: TensorLike | None,
        sample_padded_output: TensorLike,
        *,
        sample_gamma: TensorLike | None = None,
        sample_cache_output: TensorLike | None = None,
        history_frames: int = 0,
        sample_residual: TensorLike | None = None,
        sample_residual_bias: TensorLike | None = None,
        sample_residual_output: TensorLike | None = None,
        mode: str,
    ) -> None:
        """Record sample metadata and fusion options without compiling or allocating outputs."""
        super().__init__()
        self._warn_experimental_api()

        samples = {
            "input": sample_input,
            "packed_weight": sample_packed_weight,
            "bias": sample_bias,
            "padded_output": sample_padded_output,
            "gamma": sample_gamma,
            "cache_output": sample_cache_output,
            "residual": sample_residual,
            "residual_bias": sample_residual_bias,
            "residual_output": sample_residual_output,
        }
        self.descs = {name: self._make_tensor_desc(sample, name=f"sample_{name}") for name, sample in samples.items()}
        self._sample_alignment_remainders = {name: sample.data_ptr() % 16 for name, sample in samples.items() if callable(getattr(sample, "data_ptr", None))}
        self.mode = mode
        self.config = None
        self.previous_frames = history_frames

    @staticmethod
    def _require_rank(desc: TensorDesc, rank: int, name: str) -> None:
        """Reject descriptors with an unexpected number of dimensions."""
        if desc.ndim != rank:
            raise ValueError(f"{name} must be {rank}D, got shape {desc.shape}")

    @staticmethod
    def _validate_runtime_tensor(tensor: torch.Tensor, desc: TensorDesc, name: str) -> None:
        """Require runtime tensors to match the compiled shape, strides, dtype, and device."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        if tuple(tensor.shape) != desc.shape:
            raise ValueError(f"{name} shape mismatch: expected {desc.shape}, got {tuple(tensor.shape)}")
        if tuple(tensor.stride()) != desc.stride:
            raise ValueError(f"{name} stride mismatch: expected {desc.stride}, got {tuple(tensor.stride())}")
        if tensor.dtype != desc.dtype:
            raise TypeError(f"{name} dtype mismatch: expected {desc.dtype}, got {tensor.dtype}")
        if tensor.device != desc.device:
            raise ValueError(f"{name} device mismatch: expected {desc.device}, got {tensor.device}")

    def _check_desc(self, name: str, shape: tuple[int, ...]) -> None:
        """Validate a required contiguous BF16 tensor against its expected shape."""
        desc = self.descs[name]
        if desc is None:
            raise ValueError(f"{name} is required")
        self._require_rank(desc, len(shape), name)
        self._check_tensor_shape(desc, shape, name)
        if not _is_contiguous_stride(shape, desc.stride):
            raise ValueError(f"{name} must be contiguous, got stride {desc.stride}")
        self._check_dtype(desc, dtype=torch.bfloat16, name=name)

    def check_support(self) -> bool:
        """Validate the DSL version, GPU, tensor metadata, and selected fusion contract."""
        error = cutedsl_requirement_error(self.__class__.__name__, _CUTEDSL_MIN_VERSION)
        self._not_implemented_error_if(error is not None, error or "")

        input_desc = self.descs["input"]
        weight_desc = self.descs["packed_weight"]
        if input_desc is None or weight_desc is None:
            raise ValueError("input and packed_weight are required")
        self._require_rank(input_desc, 5, "input")
        self._require_rank(weight_desc, 5, "packed_weight")

        n, input_t, input_h, input_w, input_channels = input_desc.shape
        self._value_error_if(
            min(n, input_t, input_h, input_w, input_channels) <= 0,
            "input dimensions must be positive",
        )
        self._value_error_if(
            min(input_t, input_h, input_w) < 3,
            f"input spatial dimensions must be at least 3, got {input_desc.shape}",
        )
        output_channels = weight_desc.shape[0]
        channel_pair = (input_channels, output_channels)
        self._value_error_if(
            channel_pair not in _SUPPORTED_CHANNEL_PAIRS,
            f"unsupported channel pair {channel_pair}; supported pairs: {sorted(_SUPPORTED_CHANNEL_PAIRS)}",
        )
        if self.mode in ("norm_silu", "norm_silu_pad"):
            self._value_error_if(
                output_channels not in (160, 320),
                "RMSNorm + SiLU requires 160 or 320 output channels",
            )

        output_shape = (n, input_t - 2, input_h - 2, input_w - 2, output_channels)
        packed_channels = (input_channels + 63) // 64 * 64
        packed_weight_shape = (output_channels, 3, 3, 3, packed_channels)
        self._check_desc("input", input_desc.shape)
        self._check_desc("packed_weight", packed_weight_shape)
        if self.mode == "raw":
            self._value_error_if(
                any(self.descs[name] is not None for name in ("bias", "gamma", "cache_output", "residual", "residual_bias", "residual_output")),
                "raw Conv3D does not use post-operation tensors",
            )
            self._check_desc("padded_output", output_shape)
        else:
            self._check_desc("bias", (output_channels,))

        if self.mode in ("norm_silu", "norm_silu_pad"):
            self._check_desc("gamma", (output_channels,))
            if self.mode == "norm_silu_pad":
                self._value_error_if(self.previous_frames not in (0, 1, 2), "history_frames must be 0, 1, or 2")
                cache_frames = min(2, output_shape[1] + self.previous_frames)
                self._check_desc(
                    "padded_output",
                    (
                        n,
                        output_shape[1] + 2,
                        output_shape[2] + 2,
                        output_shape[3] + 2,
                        output_channels,
                    ),
                )
                self._check_desc(
                    "cache_output",
                    (
                        n,
                        cache_frames,
                        output_shape[2],
                        output_shape[3],
                        output_channels,
                    ),
                )
            else:
                self._value_error_if(
                    self.descs["cache_output"] is not None,
                    "cache_output is only used by prepared output",
                )
                self._check_desc("padded_output", output_shape)
            has_residual = self.descs["residual"] is not None
            self._value_error_if(
                has_residual != (self.descs["residual_output"] is not None),
                "residual and residual_output must be provided together",
            )
            if has_residual:
                self._check_desc("residual", output_shape)
                self._check_desc("residual_output", output_shape)
            if self.descs["residual_bias"] is not None:
                self._value_error_if(not has_residual, "residual_bias requires residual")
                self._check_desc("residual_bias", (output_channels,))
        elif self.mode == "bias_residual_pad":
            self._value_error_if(
                any(self.descs[name] is not None for name in ("gamma", "cache_output", "residual_output")),
                "gamma, cache_output, previous, and residual_output are not used by bias + residual + pad",
            )
            self._check_desc("residual", output_shape)
            self._check_desc(
                "padded_output",
                (
                    n,
                    output_shape[1],
                    output_shape[2] + 1,
                    output_shape[3] + 1,
                    output_channels,
                ),
            )
            if self.descs["residual_bias"] is not None:
                self._check_desc("residual_bias", (output_channels,))
        elif self.mode != "raw":
            raise ValueError(f"unknown fusion mode {self.mode!r}")

        device = input_desc.device
        for name, desc in self.descs.items():
            if desc is None:
                continue
            self._value_error_if(
                desc.device.type != "cuda",
                f"{name} must be a CUDA tensor, got {desc.device}",
            )
            self._value_error_if(desc.device != device, f"{name} must be on {device}, got {desc.device}")
        for name, remainder in self._sample_alignment_remainders.items():
            self._value_error_if(
                remainder != 0,
                f"{name} data pointer must be 16-byte aligned, got remainder {remainder}",
            )

        self._runtime_error_if(not torch.cuda.is_available(), "CUDA is not available")
        capability = torch.cuda.get_device_capability(device)
        self._not_implemented_error_if(
            capability not in ((10, 0), (10, 3)),
            f"{self.__class__.__name__} requires SM100 or SM103, found SM{capability[0]}{capability[1]}",
        )

        self._is_supported = True
        return True

    def compile(self) -> None:
        """Compile the shape-specialized convolution plan after checking support."""
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return

        from .kernel import Conv3dConfig, Conv3dPostOpsLaunch

        with torch.cuda.device(self.descs["input"].device):
            max_active_clusters = int(cutlass.utils.HardwareInfo().get_max_active_clusters(2))
            n, t, h, w, ci = self.descs["input"].shape
            self.config = Conv3dConfig(
                n=n,
                t=t,
                h=h,
                w=w,
                ci=ci,
                co=self.descs["packed_weight"].shape[0],
                max_active_clusters=max_active_clusters,
            )
            launch = Conv3dPostOpsLaunch(
                self.config,
                fuse_norm=self.mode in ("norm_silu", "norm_silu_pad"),
                prepare_output=self.mode == "norm_silu_pad",
                previous_frames=self.previous_frames if self.mode == "norm_silu_pad" else None,
                has_residual=self.descs["residual"] is not None,
                has_residual_bias=self.descs["residual_bias"] is not None,
                spatial_output=self.mode == "bias_residual_pad",
            )
            fake = {name: self._make_fake_cute_tensor_from_desc(desc, assumed_align=16) for name, desc in self.descs.items()}
            fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
            self._compiled_kernel = cute.compile(
                launch,
                fake["input"],
                fake["packed_weight"],
                fake["padded_output"],
                fake_stream,
                fake["bias"],
                fake["gamma"],
                fake["padded_output"],
                fake["cache_output"],
                fake["residual"],
                fake["residual_bias"],
                fake["residual_output"],
                options="--enable-tvm-ffi --generate-line-info",
            )

    def _execute(
        self,
        tensors: dict[str, torch.Tensor | None],
        current_stream: cuda.CUstream | None,
    ) -> None:
        """Validate bound tensors and launch the compiled kernel on the requested stream."""
        self._runtime_error_if(self._compiled_kernel is None, "plan not compiled; call compile() first")
        for name, desc in self.descs.items():
            tensor = tensors[name]
            if (tensor is None) != (desc is None):
                raise ValueError(f"{name} presence must match the compiled signature")
            if tensor is not None:
                self._validate_runtime_tensor(tensor, desc, name)
                _require_alignment(tensor, name)

        live_tensors = tuple(tensor for tensor in tensors.values() if tensor is not None)
        if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in live_tensors):
            raise RuntimeError("Conv3D plans are inference-only; call under torch.no_grad()")

        output_names = ["padded_output", "cache_output", "residual_output"]
        for output_name in output_names:
            output = tensors[output_name]
            if output is None:
                continue
            output_span = _byte_span(output)
            for other_name, other in tensors.items():
                if other is None or other_name == output_name:
                    continue
                if output_span[0] < _byte_span(other)[1] and _byte_span(other)[0] < output_span[1]:
                    raise ValueError(f"{output_name} must not overlap {other_name}")

        device = tensors["input"].device
        if current_stream is None:
            consumer_stream = torch.cuda.current_stream(device)
            launch_stream = cuda.CUstream(consumer_stream.cuda_stream)
        else:
            consumer_stream = as_torch_stream(current_stream, device)
            launch_stream = current_stream

        self._compiled_kernel(
            tensors["input"],
            tensors["packed_weight"],
            tensors["padded_output"],
            launch_stream,
            tensors["bias"],
            tensors["gamma"],
            tensors["padded_output"],
            tensors["cache_output"],
            tensors["residual"],
            tensors["residual_bias"],
            tensors["residual_output"],
        )
        _record_streams(tuple(tensors.values()), consumer_stream)


class CausalConv3dWithCacheSm100(APIBase):
    """Pack input/history, add causal/spatial padding, and run Conv3D without bias.

    Currently supports only 12->160 channels. Input is BF16 NCTHW; output
    and history are contiguous BF16 NTHWC. Executes packing and convolution
    as two kernels.
    """

    def __init__(
        self,
        sample_input: TensorLike,
        sample_packed_weight: TensorLike,
        sample_padded_input: TensorLike,
        sample_cache_output: TensorLike,
        sample_output: TensorLike,
        sample_previous: TensorLike | None = None,
    ) -> None:
        """Record causal input, history, scratch, and output metadata for later validation."""
        super().__init__()
        self._warn_experimental_api()
        samples = {
            "input": sample_input,
            "packed_weight": sample_packed_weight,
            "padded_input": sample_padded_input,
            "cache_output": sample_cache_output,
            "output": sample_output,
            "previous": sample_previous,
        }
        self.descs = {name: self._make_tensor_desc(sample, name=f"sample_{name}") for name, sample in samples.items()}
        self._sample_alignment_remainders = {
            name: sample.data_ptr() % (8 if name == "previous" else 16)
            for name, sample in samples.items()
            if isinstance(sample, torch.Tensor) and name != "input"
        }
        self.previous_frames = 0
        self.input_span = 0
        self.max_active_clusters = 0
        self._compiled_pack = None

    def _check_dense(self, name: str, shape: tuple[int, ...]) -> None:
        """Validate a required dense BF16 buffer against its expected shape."""
        desc = self.descs[name]
        if desc is None:
            raise ValueError(f"{name} is required")
        self._check_tensor_shape(desc, shape, name)
        if not _is_contiguous_stride(shape, desc.stride):
            raise ValueError(f"{name} must be contiguous, got stride {desc.stride}")
        self._check_dtype(desc, dtype=torch.bfloat16, name=name)

    def check_support(self) -> bool:
        """Validate the DSL/GPU and the C12-to-C160 causal packing and cache contract."""
        error = cutedsl_requirement_error(self.__class__.__name__, _CUTEDSL_MIN_VERSION)
        self._not_implemented_error_if(error is not None, error or "")
        input_desc = self.descs["input"]
        if input_desc is None:
            raise ValueError("input is required")
        self._value_error_if(input_desc.ndim != 5, f"input must be 5D NCTHW, got {input_desc.shape}")
        n, channels, frames, height, width = input_desc.shape
        self._value_error_if(channels != 12, f"input must have 12 channels, got {channels}")
        self._value_error_if(min(n, frames, height, width) <= 0, "input dimensions must be positive")
        self._check_dtype(input_desc, dtype=torch.bfloat16, name="input")
        self._value_error_if(min(input_desc.stride) < 0, "input strides must be nonnegative")

        previous_desc = self.descs["previous"]
        if previous_desc is not None:
            self._value_error_if(previous_desc.ndim != 5, "previous must be 5D NTHWC")
            self._value_error_if(previous_desc.shape[1] not in (1, 2), "previous must contain one or two frames")
            self.previous_frames = previous_desc.shape[1]
            self._check_dense("previous", (n, self.previous_frames, height, width, 12))
        cache_frames = min(2, frames + self.previous_frames)
        self._check_dense("packed_weight", (160, 448))
        self._check_dense("padded_input", (n, frames + 2, height + 2, width + 2, 16))
        self._value_error_if(
            n * (frames + 2) * (height + 2) * (width + 2) * 16 >= 2**31,
            "packed input must contain fewer than 2**31 elements",
        )
        self._check_dense("cache_output", (n, cache_frames, height, width, 12))
        self._check_dense("output", (n, frames, height, width, 160))

        for name, remainder in self._sample_alignment_remainders.items():
            alignment = 8 if name == "previous" else 16
            self._value_error_if(remainder != 0, f"{name} must be {alignment}-byte aligned")

        device = input_desc.device
        for name, desc in self.descs.items():
            if desc is None:
                continue
            self._value_error_if(desc.device.type != "cuda", f"{name} must be a CUDA tensor")
            self._value_error_if(desc.device != device, f"{name} must be on {device}, got {desc.device}")
        self._runtime_error_if(not torch.cuda.is_available(), "CUDA is not available")
        capability = torch.cuda.get_device_capability(device)
        self._not_implemented_error_if(
            capability not in ((10, 0), (10, 3)),
            f"{self.__class__.__name__} requires SM100 or SM103, found SM{capability[0]}{capability[1]}",
        )
        self.input_span = 1 + sum((extent - 1) * stride for extent, stride in zip(input_desc.shape, input_desc.stride))
        self.max_active_clusters = torch.cuda.get_device_properties(device).multi_processor_count
        self._is_supported = True
        return True

    def compile(self) -> None:
        """Compile the input-packing and convolution kernels for this causal plan."""
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        from .causal import (
            CausalConv3dConfig,
            CausalConv3dLaunch,
            CausalConv3dPackLaunch,
        )

        input_desc = self.descs["input"]
        padded_desc = self.descs["padded_input"]
        if input_desc is None or padded_desc is None:
            raise RuntimeError("input descriptors are unavailable")
        fake = {name: self._make_fake_cute_tensor_from_desc(desc, assumed_align=16) for name, desc in self.descs.items()}
        fake_input = make_fake_compact_tensor(cutlass.BFloat16, (self.input_span,), assumed_align=2)
        fake_padded = make_fake_compact_tensor(
            cutlass.BFloat16,
            (math.prod(self.descs["padded_input"].shape),),
            assumed_align=16,
        )
        fake_cache = make_fake_compact_tensor(
            cutlass.BFloat16,
            (math.prod(self.descs["cache_output"].shape),),
            assumed_align=16,
        )
        fake_previous = (
            make_fake_compact_tensor(
                cutlass.BFloat16,
                (math.prod(self.descs["previous"].shape),),
                assumed_align=8,
            )
            if self.descs["previous"] is not None
            else None
        )
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
        self._compiled_pack = cute.compile(
            CausalConv3dPackLaunch(input_desc.shape, input_desc.stride, self.previous_frames),
            fake_input,
            fake_padded,
            fake_cache,
            fake_stream,
            fake_previous,
            options="--enable-tvm-ffi --generate-line-info",
        )
        n, frames, height, width, _ = padded_desc.shape
        self._compiled_kernel = cute.compile(
            CausalConv3dLaunch(CausalConv3dConfig(n, frames, height, width, self.max_active_clusters)),
            fake["padded_input"],
            fake["packed_weight"],
            fake["output"],
            fake_stream,
            options="--enable-tvm-ffi --generate-line-info",
        )

    def execute(
        self,
        input: torch.Tensor,
        packed_weight: torch.Tensor,
        padded_input: torch.Tensor,
        cache_output: torch.Tensor,
        output: torch.Tensor,
        previous: torch.Tensor | None = None,
        current_stream: cuda.CUstream | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write caller-provided scratch, convolution output, and raw-input cache buffers."""
        self._runtime_error_if(self._compiled_kernel is None or self._compiled_pack is None, "plan not compiled; call compile() first")
        tensors = {
            "input": input,
            "packed_weight": packed_weight,
            "padded_input": padded_input,
            "cache_output": cache_output,
            "output": output,
            "previous": previous,
        }
        for name, desc in self.descs.items():
            tensor = tensors[name]
            if (tensor is None) != (desc is None):
                raise ValueError(f"{name} presence must match the compiled signature")
            if tensor is not None:
                _Conv3dPostOpsSm100._validate_runtime_tensor(tensor, desc, name)
                if name != "input":
                    alignment = 8 if name == "previous" else 16
                    if tensor.data_ptr() % alignment:
                        raise ValueError(f"{name} must be {alignment}-byte aligned")
        if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors.values() if tensor is not None):
            raise RuntimeError("input Conv3D is inference-only; call under torch.no_grad()")
        for output_name in ("padded_input", "cache_output", "output"):
            begin, end = _byte_span(tensors[output_name])
            for name, tensor in tensors.items():
                if tensor is None or name == output_name:
                    continue
                other_begin, other_end = _byte_span(tensor)
                if name == "input":
                    other_end = other_begin + self.input_span * tensor.element_size()
                if begin < other_end and other_begin < end:
                    raise ValueError(f"{output_name} must not overlap {name}")

        device = input.device
        if current_stream is None:
            consumer_stream = torch.cuda.current_stream(device)
            launch_stream = cuda.CUstream(consumer_stream.cuda_stream)
        else:
            consumer_stream = as_torch_stream(current_stream, device)
            launch_stream = current_stream
        flat_input = input.as_strided((self.input_span,), (1,))
        self._compiled_pack(
            flat_input,
            padded_input.view(-1),
            cache_output.view(-1),
            launch_stream,
            None if previous is None else previous.view(-1),
        )
        self._compiled_kernel(padded_input, packed_weight, output, launch_stream)
        _record_streams(tuple(tensors.values()), consumer_stream)
        return output, cache_output


class Conv3dRawSm100(_Conv3dPostOpsSm100):
    """Run valid Conv3D without post-operations.

    Supported channel pairs: 160->160, 160->320, 320->320, 320->640, 640->640.
    """

    def __init__(
        self,
        sample_input: TensorLike,
        sample_packed_weight: TensorLike,
        sample_output: TensorLike,
    ) -> None:
        """Record sample metadata for a valid Conv3D plan without post-operations."""
        super().__init__(
            sample_input,
            sample_packed_weight,
            None,
            sample_output,
            mode="raw",
        )

    def execute(
        self,
        input: torch.Tensor,
        packed_weight: torch.Tensor,
        output: torch.Tensor,
        current_stream: cuda.CUstream | None = None,
    ) -> torch.Tensor:
        """Write raw convolution results into the caller-provided output buffer."""
        tensors = {
            "input": input,
            "packed_weight": packed_weight,
            "bias": None,
            "padded_output": output,
            "gamma": None,
            "cache_output": None,
            "residual": None,
            "residual_bias": None,
            "residual_output": None,
        }
        self._execute(tensors, current_stream)
        return output


class Conv3dRmsNormSiluSm100(_Conv3dPostOpsSm100):
    """Fuse valid Conv3D, bias, optional residual, RMSNorm, and SiLU.

    Output is contiguous NTHWC. Supported channel pairs: 160->160, 160->320,
    320->320. Normalization uses an L2-norm clamp of 1e-12 followed by
    sqrt(channels) scaling, not additive-epsilon RMSNorm.
    """

    def __init__(
        self,
        sample_input: TensorLike,
        sample_packed_weight: TensorLike,
        sample_bias: TensorLike,
        sample_gamma: TensorLike,
        sample_output: TensorLike,
        sample_residual: TensorLike | None = None,
        sample_residual_bias: TensorLike | None = None,
        sample_residual_output: TensorLike | None = None,
    ) -> None:
        """Record convolution, normalization, and optional residual-output metadata."""
        super().__init__(
            sample_input,
            sample_packed_weight,
            sample_bias,
            sample_output,
            sample_gamma=sample_gamma,
            sample_residual=sample_residual,
            sample_residual_bias=sample_residual_bias,
            sample_residual_output=sample_residual_output,
            mode="norm_silu",
        )

    def execute(
        self,
        input: torch.Tensor,
        packed_weight: torch.Tensor,
        bias: torch.Tensor,
        gamma: torch.Tensor,
        output: torch.Tensor,
        residual: torch.Tensor | None = None,
        residual_bias: torch.Tensor | None = None,
        residual_output: torch.Tensor | None = None,
        current_stream: cuda.CUstream | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Write activated output and any requested pre-normalization residual sum."""
        tensors = {
            "input": input,
            "packed_weight": packed_weight,
            "bias": bias,
            "padded_output": output,
            "gamma": gamma,
            "cache_output": None,
            "residual": residual,
            "residual_bias": residual_bias,
            "residual_output": residual_output,
        }
        self._execute(tensors, current_stream)
        return output, residual_output


class Conv3dRmsNormSiluPadSm100(_Conv3dPostOpsSm100):
    """Fuse Conv3D, bias, RMSNorm, SiLU, current-frame stores, and zero padding.

    History interiors in padded/cache outputs are preserved, not copied.
    The caller supplies those regions separately; ``history_frames`` is 0-2.
    Supported channel pairs and normalization are the same as
    ``Conv3dRmsNormSiluSm100``.
    """

    def __init__(
        self,
        sample_input: TensorLike,
        sample_packed_weight: TensorLike,
        sample_bias: TensorLike,
        sample_gamma: TensorLike,
        sample_padded_output: TensorLike,
        sample_cache_output: TensorLike,
        history_frames: int = 0,
        sample_residual: TensorLike | None = None,
        sample_residual_bias: TensorLike | None = None,
        sample_residual_output: TensorLike | None = None,
    ) -> None:
        """Record padded-output and cache metadata, including caller-owned history length."""
        super().__init__(
            sample_input,
            sample_packed_weight,
            sample_bias,
            sample_padded_output,
            sample_gamma=sample_gamma,
            sample_cache_output=sample_cache_output,
            history_frames=history_frames,
            sample_residual=sample_residual,
            sample_residual_bias=sample_residual_bias,
            sample_residual_output=sample_residual_output,
            mode="norm_silu_pad",
        )

    def execute(
        self,
        input: torch.Tensor,
        packed_weight: torch.Tensor,
        bias: torch.Tensor,
        gamma: torch.Tensor,
        padded_output: torch.Tensor,
        cache_output: torch.Tensor,
        residual: torch.Tensor | None = None,
        residual_bias: torch.Tensor | None = None,
        residual_output: torch.Tensor | None = None,
        current_stream: cuda.CUstream | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Write current activations, cache frames, and zeros while preserving existing history."""
        tensors = {
            "input": input,
            "packed_weight": packed_weight,
            "bias": bias,
            "gamma": gamma,
            "padded_output": padded_output,
            "cache_output": cache_output,
            "residual": residual,
            "residual_bias": residual_bias,
            "residual_output": residual_output,
        }
        self._execute(tensors, current_stream)
        return padded_output, cache_output, residual_output


class Conv3dBiasResidualPadSm100(_Conv3dPostOpsSm100):
    """Fuse valid Conv3D, bias, residual addition, and bottom/right padding.

    Supported channel pairs: 160->160, 160->320, 320->320, 320->640, 640->640.
    """

    def __init__(
        self,
        sample_input: TensorLike,
        sample_packed_weight: TensorLike,
        sample_bias: TensorLike,
        sample_residual: TensorLike,
        sample_padded_output: TensorLike,
        sample_residual_bias: TensorLike | None = None,
    ) -> None:
        """Record convolution, residual, and bottom/right-padded output metadata."""
        super().__init__(
            sample_input,
            sample_packed_weight,
            sample_bias,
            sample_padded_output,
            sample_residual=sample_residual,
            sample_residual_bias=sample_residual_bias,
            mode="bias_residual_pad",
        )

    def execute(
        self,
        input: torch.Tensor,
        packed_weight: torch.Tensor,
        bias: torch.Tensor,
        residual: torch.Tensor,
        padded_output: torch.Tensor,
        residual_bias: torch.Tensor | None = None,
        current_stream: cuda.CUstream | None = None,
    ) -> torch.Tensor:
        """Write convolution plus bias/residual into the spatially padded output buffer."""
        tensors = {
            "input": input,
            "packed_weight": packed_weight,
            "bias": bias,
            "padded_output": padded_output,
            "gamma": None,
            "cache_output": None,
            "residual": residual,
            "residual_bias": residual_bias,
            "residual_output": None,
        }
        self._execute(tensors, current_stream)
        return padded_output


class RmsNormSiluPadSm100(APIBase):
    """Fuse optional bias/residual, RMSNorm, SiLU, padding, and history/cache copies.

    No convolution is performed. Supports 160, 320, or 640 channels in NTHWC
    layout. Normalization follows ``Conv3dRmsNormSiluSm100``. Unlike that
    convolution's padding variant, this operation copies existing history.
    """

    def __init__(
        self,
        sample_input: TensorLike,
        sample_gamma: TensorLike,
        sample_padded_output: TensorLike,
        sample_cache_output: TensorLike,
        sample_input_bias: TensorLike | None = None,
        sample_previous: TensorLike | None = None,
        sample_residual: TensorLike | None = None,
        sample_residual_bias: TensorLike | None = None,
        sample_residual_output: TensorLike | None = None,
    ) -> None:
        """Record normalization, history, and optional bias/residual buffer metadata."""
        super().__init__()
        self._warn_experimental_api()
        samples = {
            "input": sample_input,
            "gamma": sample_gamma,
            "padded_output": sample_padded_output,
            "cache_output": sample_cache_output,
            "input_bias": sample_input_bias,
            "previous": sample_previous,
            "residual": sample_residual,
            "residual_bias": sample_residual_bias,
            "residual_output": sample_residual_output,
        }
        self.descs = {name: self._make_tensor_desc(sample, name=f"sample_{name}") for name, sample in samples.items()}
        self._sample_alignment_remainders = {name: sample.data_ptr() % 16 for name, sample in samples.items() if callable(getattr(sample, "data_ptr", None))}
        self.shape = None
        self.previous_frames = 0

    def _check_desc(self, name: str, shape: tuple[int, ...]) -> None:
        """Validate a required contiguous BF16 buffer against its expected shape."""
        desc = self.descs[name]
        if desc is None:
            raise ValueError(f"{name} is required")
        if desc.ndim != len(shape):
            raise ValueError(f"{name} must be {len(shape)}D, got shape {desc.shape}")
        self._check_tensor_shape(desc, shape, name)
        if not _is_contiguous_stride(shape, desc.stride):
            raise ValueError(f"{name} must be contiguous, got stride {desc.stride}")
        self._check_dtype(desc, dtype=torch.bfloat16, name=name)

    def check_support(self) -> bool:
        """Validate the DSL/GPU and the standalone normalization, padding, and cache contract."""
        error = cutedsl_requirement_error(self.__class__.__name__, _CUTEDSL_MIN_VERSION)
        self._not_implemented_error_if(error is not None, error or "")

        input_desc = self.descs["input"]
        if input_desc is None:
            raise ValueError("input is required")
        if input_desc.ndim != 5:
            raise ValueError(f"input must be 5D NTHWC, got shape {input_desc.shape}")
        n, frames, height, width, channels = input_desc.shape
        self._value_error_if(min(n, frames, height, width) <= 0, "input dimensions must be positive")
        self._value_error_if(channels not in (160, 320, 640), f"unsupported channel count {channels}")
        self._check_desc("input", input_desc.shape)
        self._check_desc("gamma", (channels,))
        self._check_desc("padded_output", (n, frames + 2, height + 2, width + 2, channels))

        previous_desc = self.descs["previous"]
        if previous_desc is not None:
            self._value_error_if(previous_desc.ndim != 5, "previous must be 5D NTHWC")
            self._value_error_if(previous_desc.shape[1] not in (1, 2), "previous must contain one or two frames")
            self.previous_frames = previous_desc.shape[1]
            self._check_desc("previous", (n, self.previous_frames, height, width, channels))
        cache_frames = min(2, frames + self.previous_frames)
        self._check_desc("cache_output", (n, cache_frames, height, width, channels))

        if self.descs["input_bias"] is not None:
            self._check_desc("input_bias", (channels,))
        has_residual = self.descs["residual"] is not None
        self._value_error_if(
            has_residual and self.descs["residual_output"] is None,
            "residual requires residual_output",
        )
        if has_residual:
            self._check_desc("residual", input_desc.shape)
        if self.descs["residual_output"] is not None:
            self._check_desc("residual_output", input_desc.shape)
        if self.descs["residual_bias"] is not None:
            self._value_error_if(not has_residual, "residual_bias requires residual")
            self._check_desc("residual_bias", (channels,))

        device = input_desc.device
        for name, desc in self.descs.items():
            if desc is None:
                continue
            self._value_error_if(desc.device.type != "cuda", f"{name} must be a CUDA tensor, got {desc.device}")
            self._value_error_if(desc.device != device, f"{name} must be on {device}, got {desc.device}")
        for name, remainder in self._sample_alignment_remainders.items():
            self._value_error_if(remainder != 0, f"{name} data pointer must be 16-byte aligned, got remainder {remainder}")

        self._runtime_error_if(not torch.cuda.is_available(), "CUDA is not available")
        capability = torch.cuda.get_device_capability(device)
        self._not_implemented_error_if(
            capability not in ((10, 0), (10, 3)),
            f"{self.__class__.__name__} requires SM100 or SM103, found SM{capability[0]}{capability[1]}",
        )
        self.shape = input_desc.shape
        self._is_supported = True
        return True

    def compile(self) -> None:
        """Compile standalone normalization and preparation for the recorded tensor signature."""
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        from .rmsnorm_silu_pad import RmsNormSiluPadLaunch

        launch = RmsNormSiluPadLaunch(
            self.shape,
            self.previous_frames,
            self.descs["input_bias"] is not None,
            self.descs["residual"] is not None,
            self.descs["residual_bias"] is not None,
        )
        fake = {name: self._make_fake_cute_tensor_from_desc(desc, assumed_align=16) for name, desc in self.descs.items()}
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
        self._compiled_kernel = cute.compile(
            launch,
            fake["input"],
            fake["gamma"],
            fake["padded_output"],
            fake["cache_output"],
            fake_stream,
            fake["input_bias"],
            fake["residual"],
            fake["residual_bias"],
            fake["residual_output"],
            fake["previous"],
            options="--enable-tvm-ffi --generate-line-info",
        )

    def execute(
        self,
        input: torch.Tensor,
        gamma: torch.Tensor,
        padded_output: torch.Tensor,
        cache_output: torch.Tensor,
        input_bias: torch.Tensor | None = None,
        previous: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        residual_bias: torch.Tensor | None = None,
        residual_output: torch.Tensor | None = None,
        current_stream: cuda.CUstream | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Write normalized/padded output, copied history/cache, and any requested preactivation."""
        self._runtime_error_if(self._compiled_kernel is None, "plan not compiled; call compile() first")
        tensors = {
            "input": input,
            "gamma": gamma,
            "padded_output": padded_output,
            "cache_output": cache_output,
            "input_bias": input_bias,
            "previous": previous,
            "residual": residual,
            "residual_bias": residual_bias,
            "residual_output": residual_output,
        }
        for name, desc in self.descs.items():
            tensor = tensors[name]
            if (tensor is None) != (desc is None):
                raise ValueError(f"{name} presence must match the compiled signature")
            if tensor is not None:
                _Conv3dPostOpsSm100._validate_runtime_tensor(tensor, desc, name)
                _require_alignment(tensor, name)

        live_tensors = tuple(tensor for tensor in tensors.values() if tensor is not None)
        if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in live_tensors):
            raise RuntimeError("RMSNorm SiLU preparation is inference-only; call under torch.no_grad()")
        for output_name in ("padded_output", "cache_output", "residual_output"):
            output = tensors[output_name]
            if output is None:
                continue
            output_span = _byte_span(output)
            for other_name, other in tensors.items():
                if other is None or other_name == output_name:
                    continue
                if output_span[0] < _byte_span(other)[1] and _byte_span(other)[0] < output_span[1]:
                    raise ValueError(f"{output_name} must not overlap {other_name}")

        device = input.device
        if current_stream is None:
            consumer_stream = torch.cuda.current_stream(device)
            launch_stream = cuda.CUstream(consumer_stream.cuda_stream)
        else:
            consumer_stream = as_torch_stream(current_stream, device)
            launch_stream = current_stream
        self._compiled_kernel(
            input,
            gamma,
            padded_output,
            cache_output,
            launch_stream,
            input_bias,
            residual,
            residual_bias,
            residual_output,
            previous,
        )
        _record_streams(tuple(tensors.values()), consumer_stream)
        return padded_output, cache_output, residual_output


def _tensor_key(tensor: torch.Tensor | None) -> tuple | None:
    """Build a plan-cache key from tensor metadata, excluding its data pointer."""
    if tensor is None:
        return None
    return (
        tensor.device.type,
        tensor.device.index,
        tensor.dtype,
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


def _cached_plan(key: tuple, factory) -> APIBase:
    """Reuse a compiled plan or build one under the lock with bounded FIFO eviction."""
    plan = _PLAN_CACHE.get(key)
    if plan is not None:
        return plan
    with _PLAN_CACHE_LOCK:
        plan = _PLAN_CACHE.get(key)
        if plan is None:
            plan = factory()
            plan.check_support()
            plan.compile()
            if len(_PLAN_CACHE) >= _PLAN_CACHE_CAPACITY:
                _PLAN_CACHE.pop(next(iter(_PLAN_CACHE)))
            _PLAN_CACHE[key] = plan
    return plan


def conv3d_raw_wrapper_sm100(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    current_stream: cuda.CUstream | None = None,
) -> torch.Tensor:
    """Allocate output and run the tuned valid Conv3D core."""
    if input.ndim != 5 or packed_weight.ndim != 5:
        raise ValueError("input and packed_weight must be 5D")
    n, input_t, input_h, input_w, _ = input.shape
    if min(input_t, input_h, input_w) < 3:
        raise ValueError("input temporal and spatial dimensions must be at least 3")
    output_shape = (n, input_t - 2, input_h - 2, input_w - 2, packed_weight.shape[0])
    with stream_context(current_stream, input.device):
        output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    key = ("raw", *(_tensor_key(tensor) for tensor in (input, packed_weight, output)))
    plan = _cached_plan(key, lambda: Conv3dRawSm100(input, packed_weight, output))
    plan.execute(input, packed_weight, output, current_stream)
    return output


def causal_conv3d_with_cache_wrapper_sm100(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    previous: torch.Tensor | None = None,
    current_stream: cuda.CUstream | None = None,
) -> TupleDict:
    """Allocate outputs and run ``CausalConv3dWithCacheSm100`` (currently 12->160 only)."""
    if input.ndim != 5:
        raise ValueError("input must be 5D NCTHW")
    n, channels, frames, height, width = input.shape
    if channels != 12:
        raise ValueError(f"input must have 12 channels, got {channels}")
    if previous is not None and previous.ndim != 5:
        raise ValueError("previous must be 5D NTHWC")
    previous_frames = 0 if previous is None else previous.shape[1]
    cache_frames = min(2, frames + previous_frames)
    with stream_context(current_stream, input.device):
        padded_input = torch.empty(
            (n, frames + 2, height + 2, width + 2, 16),
            dtype=input.dtype,
            device=input.device,
        )
        cache_output = torch.empty(
            (n, cache_frames, height, width, 12),
            dtype=input.dtype,
            device=input.device,
        )
        output = torch.empty(
            (n, frames, height, width, 160),
            dtype=input.dtype,
            device=input.device,
        )
    key = (
        "input_c12",
        *(_tensor_key(tensor) for tensor in (input, packed_weight, padded_input, cache_output, output, previous)),
    )
    plan = _cached_plan(
        key,
        lambda: CausalConv3dWithCacheSm100(
            input,
            packed_weight,
            padded_input,
            cache_output,
            output,
            previous,
        ),
    )
    plan.execute(
        input,
        packed_weight,
        padded_input,
        cache_output,
        output,
        previous,
        current_stream,
    )
    return TupleDict(output=output, cache_output=cache_output)


def conv3d_rmsnorm_silu_wrapper_sm100(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    bias: torch.Tensor,
    gamma: torch.Tensor,
    residual: torch.Tensor | None = None,
    residual_bias: torch.Tensor | None = None,
    current_stream: cuda.CUstream | None = None,
) -> TupleDict:
    """Allocate outputs and run ``Conv3dRmsNormSiluSm100``; see its channel limits."""
    if input.ndim != 5 or packed_weight.ndim != 5:
        raise ValueError("input and packed_weight must be 5D")
    n, input_t, input_h, input_w, _ = input.shape
    if min(input_t, input_h, input_w) < 3:
        raise ValueError("input temporal and spatial dimensions must be at least 3")
    output_shape = (
        n,
        input_t - 2,
        input_h - 2,
        input_w - 2,
        packed_weight.shape[0],
    )
    with stream_context(current_stream, input.device):
        output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        residual_output = torch.empty(output_shape, dtype=input.dtype, device=input.device) if residual is not None else None

    key = (
        "norm_silu",
        *(
            _tensor_key(tensor)
            for tensor in (
                input,
                packed_weight,
                bias,
                gamma,
                output,
                residual,
                residual_bias,
                residual_output,
            )
        ),
    )
    plan = _cached_plan(
        key,
        lambda: Conv3dRmsNormSiluSm100(
            input,
            packed_weight,
            bias,
            gamma,
            output,
            residual,
            residual_bias,
            residual_output,
        ),
    )
    plan.execute(
        input,
        packed_weight,
        bias,
        gamma,
        output,
        residual,
        residual_bias,
        residual_output,
        current_stream,
    )
    return TupleDict(output=output, residual_output=residual_output)


def conv3d_rmsnorm_silu_pad_wrapper_sm100(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    bias: torch.Tensor,
    gamma: torch.Tensor,
    history_frames: int = 0,
    residual: torch.Tensor | None = None,
    residual_bias: torch.Tensor | None = None,
    current_stream: cuda.CUstream | None = None,
) -> TupleDict:
    """Allocate padded/cache outputs; the caller must fill their history interiors.

    Only current frames, missing-history zeros, and spatial borders are written.
    See ``Conv3dRmsNormSiluPadSm100`` for the preallocated-output API.
    """
    if input.ndim != 5 or packed_weight.ndim != 5:
        raise ValueError("input and packed_weight must be 5D")
    n, input_t, input_h, input_w, _ = input.shape
    if min(input_t, input_h, input_w) < 3:
        raise ValueError("input temporal and spatial dimensions must be at least 3")
    if history_frames not in (0, 1, 2):
        raise ValueError("history_frames must be 0, 1, or 2")
    output_channels = packed_weight.shape[0]
    output_shape = (n, input_t - 2, input_h - 2, input_w - 2, output_channels)
    cache_shape = (
        n,
        min(2, output_shape[1] + history_frames),
        output_shape[2],
        output_shape[3],
        output_channels,
    )
    padded_shape = (
        n,
        output_shape[1] + 2,
        output_shape[2] + 2,
        output_shape[3] + 2,
        output_channels,
    )
    with stream_context(current_stream, input.device):
        padded_output = torch.empty(padded_shape, dtype=input.dtype, device=input.device)
        cache_output = torch.empty(cache_shape, dtype=input.dtype, device=input.device)
        residual_output = torch.empty(output_shape, dtype=input.dtype, device=input.device) if residual is not None else None

    key = (
        "norm_silu_pad",
        history_frames,
        *(
            _tensor_key(tensor)
            for tensor in (
                input,
                packed_weight,
                bias,
                gamma,
                padded_output,
                cache_output,
                residual,
                residual_bias,
                residual_output,
            )
        ),
    )
    plan = _cached_plan(
        key,
        lambda: Conv3dRmsNormSiluPadSm100(
            input,
            packed_weight,
            bias,
            gamma,
            padded_output,
            cache_output,
            history_frames,
            residual,
            residual_bias,
            residual_output,
        ),
    )
    plan.execute(
        input,
        packed_weight,
        bias,
        gamma,
        padded_output,
        cache_output,
        residual,
        residual_bias,
        residual_output,
        current_stream,
    )
    return TupleDict(
        padded_output=padded_output,
        cache_output=cache_output,
        residual_output=residual_output,
    )


def conv3d_bias_residual_pad_wrapper_sm100(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    residual_bias: torch.Tensor | None = None,
    current_stream: cuda.CUstream | None = None,
) -> TupleDict:
    """Allocate output and run fused Conv3D + bias + residual + spatial pad."""
    if input.ndim != 5 or packed_weight.ndim != 5:
        raise ValueError("input and packed_weight must be 5D")
    n, input_t, input_h, input_w, _ = input.shape
    if min(input_t, input_h, input_w) < 3:
        raise ValueError("input temporal and spatial dimensions must be at least 3")
    output_channels = packed_weight.shape[0]
    padded_shape = (n, input_t - 2, input_h - 1, input_w - 1, output_channels)
    with stream_context(current_stream, input.device):
        padded_output = torch.empty(padded_shape, dtype=input.dtype, device=input.device)

    key = (
        "bias_residual_pad",
        *(
            _tensor_key(tensor)
            for tensor in (
                input,
                packed_weight,
                bias,
                residual,
                residual_bias,
                padded_output,
            )
        ),
    )
    plan = _cached_plan(
        key,
        lambda: Conv3dBiasResidualPadSm100(
            input,
            packed_weight,
            bias,
            residual,
            padded_output,
            residual_bias,
        ),
    )
    plan.execute(
        input,
        packed_weight,
        bias,
        residual,
        padded_output,
        residual_bias,
        current_stream,
    )
    return TupleDict(padded_output=padded_output)


def rmsnorm_silu_pad_wrapper_sm100(
    input: torch.Tensor,
    gamma: torch.Tensor,
    input_bias: torch.Tensor | None = None,
    previous: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    residual_bias: torch.Tensor | None = None,
    current_stream: cuda.CUstream | None = None,
    *,
    save_input: bool = False,
) -> TupleDict:
    """Allocate outputs and run ``RmsNormSiluPadSm100`` (C160/C320/C640; no convolution)."""
    if input.ndim != 5:
        raise ValueError("input must be 5D NTHWC")
    if previous is not None and previous.ndim != 5:
        raise ValueError("previous must be 5D NTHWC")
    n, frames, height, width, channels = input.shape
    previous_frames = 0 if previous is None else previous.shape[1]
    padded_shape = (n, frames + 2, height + 2, width + 2, channels)
    cache_shape = (n, min(2, frames + previous_frames), height, width, channels)
    with stream_context(current_stream, input.device):
        padded_output = torch.empty(padded_shape, dtype=input.dtype, device=input.device)
        cache_output = torch.empty(cache_shape, dtype=input.dtype, device=input.device)
        residual_output = torch.empty_like(input) if residual is not None or save_input else None

    key = (
        "rmsnorm_silu_pad",
        *(
            _tensor_key(tensor)
            for tensor in (
                input,
                gamma,
                padded_output,
                cache_output,
                input_bias,
                previous,
                residual,
                residual_bias,
                residual_output,
            )
        ),
    )
    plan = _cached_plan(
        key,
        lambda: RmsNormSiluPadSm100(
            input,
            gamma,
            padded_output,
            cache_output,
            input_bias,
            previous,
            residual,
            residual_bias,
            residual_output,
        ),
    )
    plan.execute(
        input,
        gamma,
        padded_output,
        cache_output,
        input_bias,
        previous,
        residual,
        residual_bias,
        residual_output,
        current_stream,
    )
    return TupleDict(
        padded_output=padded_output,
        cache_output=cache_output,
        residual_output=residual_output,
    )


__all__ = [
    "CausalConv3dWithCacheSm100",
    "Conv3dBiasResidualPadSm100",
    "Conv3dRawSm100",
    "Conv3dRmsNormSiluPadSm100",
    "Conv3dRmsNormSiluSm100",
    "RmsNormSiluPadSm100",
    "causal_conv3d_with_cache_wrapper_sm100",
    "conv3d_bias_residual_pad_wrapper_sm100",
    "conv3d_raw_wrapper_sm100",
    "conv3d_rmsnorm_silu_pad_wrapper_sm100",
    "conv3d_rmsnorm_silu_wrapper_sm100",
    "rmsnorm_silu_pad_wrapper_sm100",
]
