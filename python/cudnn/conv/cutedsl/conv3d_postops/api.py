# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct APIs for the SM100 Conv3D post-operation kernels.

Plans validate sample metadata and storage during preparation. Execution binds
compatible tensors without revalidation; callers preserve the prepared signature,
pointer alignment, non-overlap, and inference-only contract on every call.
"""

import math
import threading

import cutlass
import torch
from cuda.bindings import driver as cuda

from cudnn._torch_stream import as_torch_stream, stream_context
from cudnn.api_base import APIBase, TensorDesc, TupleDict, tensor_byte_span
from cudnn.conv.frost._cutedsl import requirement_error as cutedsl_requirement_error

TensorLike = torch.Tensor | TensorDesc
_CUTEDSL_MIN_VERSION = (4, 9)
from ._config import SUPPORTED_CHANNEL_PAIRS

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


def _check_sample_storage(spans: dict[str, tuple[int, int]], output_names: tuple[str, ...], requires_grad: bool) -> None:
    """Validate captured storage properties without retaining sample tensors."""
    if torch.is_grad_enabled() and requires_grad:
        raise RuntimeError("Conv3D plans are inference-only; call under torch.no_grad()")
    for output_name in output_names:
        if output_name not in spans:
            continue
        begin, end = spans[output_name]
        for name, (other_begin, other_end) in spans.items():
            if name != output_name and begin < other_end and other_begin < end:
                raise ValueError(f"{output_name} must not overlap {name}")


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
        allow_tf32: bool = False,
        sample_workspace: TensorLike | None = None,
    ) -> None:
        """Record sample metadata and fusion options without compiling or allocating outputs."""
        super().__init__()
        self._warn_experimental_api()

        samples = {
            "input": sample_input,
            "workspace": sample_workspace,
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
        self._sample_spans = {name: tensor_byte_span(sample) for name, sample in samples.items() if isinstance(sample, torch.Tensor)}
        self._sample_requires_grad = any(sample.requires_grad for sample in samples.values() if isinstance(sample, torch.Tensor))
        self.allow_tf32 = allow_tf32
        self._compiled_round = None
        self.mode = mode
        self.config = None
        self.previous_frames = history_frames

    @staticmethod
    def _require_rank(desc: TensorDesc, rank: int, name: str) -> None:
        """Reject descriptors with an unexpected number of dimensions."""
        if desc.ndim != rank:
            raise ValueError(f"{name} must be {rank}D, got shape {desc.shape}")

    def _check_desc(self, name: str, shape: tuple[int, ...]) -> None:
        """Validate a required contiguous tensor against its expected shape."""
        desc = self.descs[name]
        if desc is None:
            raise ValueError(f"{name} is required")
        self._require_rank(desc, len(shape), name)
        self._check_tensor_shape(desc, shape, name)
        if not _is_contiguous_stride(shape, desc.stride):
            raise ValueError(f"{name} must be contiguous, got stride {desc.stride}")
        self._check_dtype(desc, dtype=self.descs["input"].dtype, name=name)

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

        self._value_error_if(input_desc.dtype not in (torch.bfloat16, torch.float32), "input must be BF16 or FP32")
        self._value_error_if(
            input_desc.dtype == torch.float32 and not self.allow_tf32,
            "FP32 convolution requires allow_tf32=True; strict FP32 convolution belongs to the caller",
        )
        if input_desc.dtype == torch.float32:
            self._check_desc("workspace", input_desc.shape)
        else:
            self._value_error_if(self.descs["workspace"] is not None, "BF16 convolution does not use workspace")
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
            channel_pair not in SUPPORTED_CHANNEL_PAIRS,
            f"unsupported channel pair {channel_pair}; supported pairs: {sorted(SUPPORTED_CHANNEL_PAIRS)}",
        )
        if self.mode in ("norm_silu", "norm_silu_pad"):
            self._value_error_if(
                output_channels not in (96, 160, 192, 256, 320),
                "RMSNorm + SiLU requires 96, 160, 192, 256, or 320 output channels",
            )

        output_shape = (n, input_t - 2, input_h - 2, input_w - 2, output_channels)
        tile_k = 32 if input_desc.dtype == torch.float32 else 64
        packed_channels = (input_channels + tile_k - 1) // tile_k * tile_k
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

        _check_sample_storage(self._sample_spans, ("padded_output", "cache_output", "residual_output", "workspace"), self._sample_requires_grad)
        self._is_supported = True
        return True

    def compile(self) -> None:
        """Bind a reusable convolution executable after checking support."""
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return

        from ._compile import compile_conv, compile_round
        from ._config import ConvConfig, ConvSpecialization, needs_wide_index

        device = self.descs["input"].device
        with torch.cuda.device(device):
            ctas = 2 * int(cutlass.utils.HardwareInfo().get_max_active_clusters(2))
            self.geometry = self.descs["input"].shape[:4]
            precision = "bf16" if self.descs["input"].dtype == torch.bfloat16 else "tf32"
            self.config = ConvConfig(*self.geometry, self.descs["input"].shape[-1], self.descs["packed_weight"].shape[0], ctas, precision)
            spec = ConvSpecialization(self.config.ci, self.config.co, ctas, precision)
            self._compiled_kernel = compile_conv(
                device,
                spec,
                self.mode,
                self.descs["residual"] is not None,
                self.descs["residual_bias"] is not None,
                needs_wide_index(self.config),
            )

            if precision == "tf32":
                self._compiled_round = compile_round(device)

    def _launch(self, tensors, current_stream, record_tensors=None):
        """Bind and launch; wrappers record only externally allocated inputs."""
        device = tensors["input"].device
        if current_stream is None:
            consumer_stream = torch.cuda.current_stream(device)
            launch_stream = cuda.CUstream(consumer_stream.cuda_stream)
        else:
            consumer_stream = as_torch_stream(current_stream, device)
            launch_stream = current_stream

        flat = lambda name: None if tensors[name] is None else tensors[name].view(-1)
        with torch.cuda.device(device):
            conv_input = flat("input")
            if self._compiled_round is not None:
                self._compiled_round(conv_input, flat("workspace"), tensors["input"].numel(), launch_stream)
                conv_input = flat("workspace")
            self._compiled_kernel(
                conv_input,
                flat("packed_weight"),
                flat("padded_output"),
                launch_stream,
                flat("bias"),
                flat("gamma"),
                flat("padded_output"),
                flat("cache_output"),
                flat("residual"),
                flat("residual_bias"),
                flat("residual_output"),
                geometry=self.geometry,
                history=self.previous_frames,
            )
        _record_streams(tuple(tensors.values()) if record_tensors is None else record_tensors, consumer_stream)


class CausalConv3dWithCacheSm100(APIBase):
    """Pack input/history, add causal/spatial padding, and run Conv3D without bias.

    Supports 12->160 channels. Input is strided NCTHW; output and history
    are contiguous NTHWC. BF16 and explicit FP32/TF32 execution use two
    kernels: packing (including TF32 rounding), then convolution. FP32 cache
    retains unrounded input values.
    """

    def __init__(
        self,
        sample_input: TensorLike,
        sample_packed_weight: TensorLike,
        sample_padded_input: TensorLike,
        sample_cache_output: TensorLike,
        sample_output: TensorLike,
        sample_previous: TensorLike | None = None,
        allow_tf32: bool = False,
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
        self._sample_spans = {name: tensor_byte_span(sample) for name, sample in samples.items() if isinstance(sample, torch.Tensor)}
        self._sample_requires_grad = any(sample.requires_grad for sample in samples.values() if isinstance(sample, torch.Tensor))
        self.allow_tf32 = allow_tf32
        self.previous_frames = 0
        self.input_span = 0
        self.max_active_clusters = 0
        self._compiled_pack = None

    def _check_dense(self, name: str, shape: tuple[int, ...]) -> None:
        """Validate a required dense buffer against its expected shape."""
        desc = self.descs[name]
        if desc is None:
            raise ValueError(f"{name} is required")
        self._check_tensor_shape(desc, shape, name)
        if not _is_contiguous_stride(shape, desc.stride):
            raise ValueError(f"{name} must be contiguous, got stride {desc.stride}")
        self._check_dtype(desc, dtype=self.descs["input"].dtype, name=name)

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
        self._value_error_if(input_desc.dtype not in (torch.bfloat16, torch.float32), "input must be BF16 or FP32")
        self._value_error_if(
            input_desc.dtype == torch.float32 and not self.allow_tf32,
            "FP32 convolution requires allow_tf32=True; strict FP32 convolution belongs to the caller",
        )
        self._value_error_if(min(input_desc.stride) < 0, "input strides must be nonnegative")

        previous_desc = self.descs["previous"]
        if previous_desc is not None:
            self._value_error_if(previous_desc.ndim != 5, "previous must be 5D NTHWC")
            self._value_error_if(previous_desc.shape[1] not in (1, 2), "previous must contain one or two frames")
            self.previous_frames = previous_desc.shape[1]
            self._check_dense("previous", (n, self.previous_frames, height, width, 12))
        cache_frames = min(2, frames + self.previous_frames)
        self._check_dense("packed_weight", (160, 448) if input_desc.dtype == torch.bfloat16 else (160, 3, 3, 3, 32))
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
        _check_sample_storage(self._sample_spans, ("padded_input", "cache_output", "output"), self._sample_requires_grad)
        self._is_supported = True
        return True

    def compile(self) -> None:
        """Compile the input-packing and convolution kernels for this causal plan."""
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        from ._compile import compile_causal_conv, compile_causal_pack, compile_conv
        from ._config import ConvSpecialization

        device = self.descs["input"].device
        dtype = self.descs["input"].dtype
        self._compiled_pack = compile_causal_pack(device, dtype, self.descs["previous"] is not None)
        if dtype == torch.bfloat16:
            self._compiled_kernel = compile_causal_conv(device, self.max_active_clusters)
        else:
            with torch.cuda.device(device):
                ctas = 2 * int(cutlass.utils.HardwareInfo().get_max_active_clusters(2))
            self._compiled_kernel = compile_conv(device, ConvSpecialization(16, 160, ctas, "tf32"), "raw", False, False, False)

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
        tensors = {
            "input": input,
            "packed_weight": packed_weight,
            "padded_input": padded_input,
            "cache_output": cache_output,
            "output": output,
            "previous": previous,
        }
        return self._launch(tensors, current_stream)

    def _launch(self, tensors, current_stream, record_tensors=None):
        """Bind and launch; wrappers record only externally allocated inputs."""
        input = tensors["input"]
        packed_weight = tensors["packed_weight"]
        padded_input = tensors["padded_input"]
        cache_output = tensors["cache_output"]
        output = tensors["output"]
        previous = tensors["previous"]
        device = input.device
        if current_stream is None:
            consumer_stream = torch.cuda.current_stream(device)
            launch_stream = cuda.CUstream(consumer_stream.cuda_stream)
        else:
            consumer_stream = as_torch_stream(current_stream, device)
            launch_stream = current_stream
        flat_input = input.as_strided((self.input_span,), (1,))
        with torch.cuda.device(device):
            self._compiled_pack(
                flat_input,
                padded_input.view(-1),
                cache_output.view(-1),
                launch_stream,
                None if previous is None else previous.view(-1),
                geometry=(input.shape[0], *input.shape[2:]),
                input_strides=tuple(input.stride()),
                history=self.previous_frames,
            )
            if input.dtype == torch.bfloat16:
                self._compiled_kernel(padded_input.view(-1), packed_weight.view(-1), output.view(-1), tuple(padded_input.shape[:4]), launch_stream)
            else:
                self._compiled_kernel(
                    padded_input.view(-1),
                    packed_weight.view(-1),
                    output.view(-1),
                    launch_stream,
                    None,
                    None,
                    output.view(-1),
                    None,
                    None,
                    None,
                    None,
                    geometry=tuple(padded_input.shape[:4]),
                    history=0,
                )
        _record_streams(tuple(tensors.values()) if record_tensors is None else record_tensors, consumer_stream)
        return output, cache_output


class Conv3dRawSm100(_Conv3dPostOpsSm100):
    """Run valid Conv3D without post-operations.

    Supports the channel pairs listed in the Conv3D post-operations API reference.
    FP32 requires allow_tf32=True and caller-provided input-rounding workspace.
    """

    def __init__(
        self,
        sample_input: TensorLike,
        sample_packed_weight: TensorLike,
        sample_output: TensorLike,
        allow_tf32: bool = False,
        sample_workspace: TensorLike | None = None,
    ) -> None:
        """Record sample metadata for a valid Conv3D plan without post-operations."""
        super().__init__(
            sample_input,
            sample_packed_weight,
            None,
            sample_output,
            mode="raw",
            allow_tf32=allow_tf32,
            sample_workspace=sample_workspace,
        )

    def execute(
        self,
        input: torch.Tensor,
        packed_weight: torch.Tensor,
        output: torch.Tensor,
        current_stream: cuda.CUstream | None = None,
        workspace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Write raw convolution results into the caller-provided output buffer."""
        tensors = {
            "input": input,
            "workspace": workspace,
            "packed_weight": packed_weight,
            "bias": None,
            "padded_output": output,
            "gamma": None,
            "cache_output": None,
            "residual": None,
            "residual_bias": None,
            "residual_output": None,
        }
        self._launch(tensors, current_stream)
        return output


class Conv3dRmsNormSiluSm100(_Conv3dPostOpsSm100):
    """Fuse valid Conv3D, bias, optional residual, RMSNorm, and SiLU.

    Output is contiguous NTHWC. Supported pairs have Co=96/160/192/256/320.
    FP32 requires allow_tf32=True and caller-provided input-rounding workspace. Normalization uses an L2-norm
    clamp of 1e-12 followed by sqrt(channels) scaling, not additive-epsilon RMSNorm.
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
        allow_tf32: bool = False,
        sample_workspace: TensorLike | None = None,
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
            allow_tf32=allow_tf32,
            sample_workspace=sample_workspace,
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
        workspace: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Write activated output and any requested pre-normalization residual sum."""
        tensors = {
            "input": input,
            "workspace": workspace,
            "packed_weight": packed_weight,
            "bias": bias,
            "padded_output": output,
            "gamma": gamma,
            "cache_output": None,
            "residual": residual,
            "residual_bias": residual_bias,
            "residual_output": residual_output,
        }
        self._launch(tensors, current_stream)
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
        allow_tf32: bool = False,
        sample_workspace: TensorLike | None = None,
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
            allow_tf32=allow_tf32,
            sample_workspace=sample_workspace,
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
        workspace: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Write current activations, cache frames, and zeros while preserving existing history."""
        tensors = {
            "input": input,
            "workspace": workspace,
            "packed_weight": packed_weight,
            "bias": bias,
            "gamma": gamma,
            "padded_output": padded_output,
            "cache_output": cache_output,
            "residual": residual,
            "residual_bias": residual_bias,
            "residual_output": residual_output,
        }
        self._launch(tensors, current_stream)
        return padded_output, cache_output, residual_output


class Conv3dBiasResidualPadSm100(_Conv3dPostOpsSm100):
    """Fuse valid Conv3D, bias, residual addition, and bottom/right padding.

    Supports the channel pairs listed in the Conv3D post-operations API reference.
    FP32 requires allow_tf32=True and caller-provided input-rounding workspace.
    """

    def __init__(
        self,
        sample_input: TensorLike,
        sample_packed_weight: TensorLike,
        sample_bias: TensorLike,
        sample_residual: TensorLike,
        sample_padded_output: TensorLike,
        sample_residual_bias: TensorLike | None = None,
        allow_tf32: bool = False,
        sample_workspace: TensorLike | None = None,
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
            allow_tf32=allow_tf32,
            sample_workspace=sample_workspace,
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
        workspace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Write convolution plus bias/residual into the spatially padded output buffer."""
        tensors = {
            "input": input,
            "workspace": workspace,
            "packed_weight": packed_weight,
            "bias": bias,
            "padded_output": padded_output,
            "gamma": None,
            "cache_output": None,
            "residual": residual,
            "residual_bias": residual_bias,
            "residual_output": None,
        }
        self._launch(tensors, current_stream)
        return padded_output


class RmsNormSiluPadSm100(APIBase):
    """Fuse optional bias/residual, RMSNorm, SiLU, padding, and history/cache copies.

    No convolution is performed. Supports C96/160/192/256/320/384/512/640/1024
    in contiguous NTHWC, with BF16 or FP32 storage. Normalization follows ``Conv3dRmsNormSiluSm100``. Unlike that
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
        self._sample_spans = {name: tensor_byte_span(sample) for name, sample in samples.items() if isinstance(sample, torch.Tensor)}
        self._sample_requires_grad = any(sample.requires_grad for sample in samples.values() if isinstance(sample, torch.Tensor))
        self.shape = None
        self.previous_frames = 0

    def _check_desc(self, name: str, shape: tuple[int, ...]) -> None:
        """Validate a required contiguous buffer against its expected shape."""
        desc = self.descs[name]
        if desc is None:
            raise ValueError(f"{name} is required")
        if desc.ndim != len(shape):
            raise ValueError(f"{name} must be {len(shape)}D, got shape {desc.shape}")
        self._check_tensor_shape(desc, shape, name)
        if not _is_contiguous_stride(shape, desc.stride):
            raise ValueError(f"{name} must be contiguous, got stride {desc.stride}")
        self._check_dtype(desc, dtype=self.descs["input"].dtype, name=name)

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
        self._value_error_if(channels not in (96, 160, 192, 256, 320, 384, 512, 640, 1024), f"unsupported channel count {channels}")
        self._value_error_if(input_desc.dtype not in (torch.bfloat16, torch.float32), "input must be BF16 or FP32")
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
        _check_sample_storage(self._sample_spans, ("padded_output", "cache_output", "residual_output"), self._sample_requires_grad)
        self._is_supported = True
        return True

    def compile(self) -> None:
        """Compile standalone normalization and preparation for the recorded tensor signature."""
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        from ._compile import compile_norm

        self._compiled_kernel = compile_norm(
            self.descs["input"].device,
            self.descs["input"].dtype,
            self.shape[-1],
            math.prod(self.descs["padded_output"].shape[:-1]) * (self.shape[-1] // 8) + 512 >= 2**31,
            *(self.descs[name] is not None for name in ("input_bias", "residual", "residual_bias", "residual_output", "previous")),
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
        return self._launch(tensors, current_stream)

    def _launch(self, tensors, current_stream, record_tensors=None):
        """Bind and launch; wrappers record only externally allocated inputs."""
        input = tensors["input"]
        gamma = tensors["gamma"]
        padded_output = tensors["padded_output"]
        cache_output = tensors["cache_output"]
        input_bias = tensors["input_bias"]
        previous = tensors["previous"]
        residual = tensors["residual"]
        residual_bias = tensors["residual_bias"]
        residual_output = tensors["residual_output"]
        device = input.device
        if current_stream is None:
            consumer_stream = torch.cuda.current_stream(device)
            launch_stream = cuda.CUstream(consumer_stream.cuda_stream)
        else:
            consumer_stream = as_torch_stream(current_stream, device)
            launch_stream = current_stream
        flat = lambda tensor: None if tensor is None else tensor.view(-1)
        with torch.cuda.device(device):
            self._compiled_kernel(
                flat(input),
                flat(gamma),
                flat(padded_output),
                flat(cache_output),
                launch_stream,
                flat(input_bias),
                flat(residual),
                flat(residual_bias),
                flat(residual_output),
                flat(previous),
                geometry=self.shape[:4],
                history=self.previous_frames,
            )
        _record_streams(tuple(tensors.values()) if record_tensors is None else record_tensors, consumer_stream)
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


def _run_allocating_wrapper(
    key: tuple,
    inputs: dict[str, torch.Tensor | None],
    outputs: dict[str, torch.Tensor | None],
    factory,
    current_stream: cuda.CUstream | None,
) -> None:
    """Bind cached input metadata and wrapper-owned outputs to the shared launcher.

    Outputs/scratch are fresh allocations on the launch stream: their metadata is
    derived from the key, they cannot alias live inputs, and their allocation
    stream already orders reuse. Only externally allocated inputs need recording.
    Cache misses validate the samples; hits rely on the caller's storage contract.
    """
    plan = _cached_plan((*key, *(_tensor_key(tensor) for tensor in inputs.values())), factory)
    plan._launch({**inputs, **outputs}, current_stream, tuple(inputs.values()))


def conv3d_raw_wrapper_sm100(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    current_stream: cuda.CUstream | None = None,
    allow_tf32: bool = False,
) -> torch.Tensor:
    """Allocate output and run the tuned valid Conv3D core."""
    n, input_t, input_h, input_w, _ = input.shape
    output_shape = (n, input_t - 2, input_h - 2, input_w - 2, packed_weight.shape[0])
    with stream_context(current_stream, input.device):
        workspace = torch.empty_like(input) if input.dtype == torch.float32 else None
        output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    _run_allocating_wrapper(
        ("raw", allow_tf32),
        {"input": input, "packed_weight": packed_weight, "bias": None, "gamma": None, "residual": None, "residual_bias": None},
        {"workspace": workspace, "padded_output": output, "cache_output": None, "residual_output": None},
        lambda: Conv3dRawSm100(input, packed_weight, output, allow_tf32=allow_tf32, sample_workspace=workspace),
        current_stream,
    )
    return output


def causal_conv3d_with_cache_wrapper_sm100(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    previous: torch.Tensor | None = None,
    current_stream: cuda.CUstream | None = None,
    allow_tf32: bool = False,
) -> TupleDict:
    """Allocate outputs and run ``CausalConv3dWithCacheSm100`` (currently 12->160 only)."""
    n, _, frames, height, width = input.shape
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
    _run_allocating_wrapper(
        ("input_c12", allow_tf32),
        {"input": input, "packed_weight": packed_weight, "previous": previous},
        {"padded_input": padded_input, "cache_output": cache_output, "output": output},
        lambda: CausalConv3dWithCacheSm100(input, packed_weight, padded_input, cache_output, output, previous, allow_tf32=allow_tf32),
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
    allow_tf32: bool = False,
) -> TupleDict:
    """Allocate outputs and run ``Conv3dRmsNormSiluSm100``; see its channel limits."""
    n, input_t, input_h, input_w, _ = input.shape
    output_shape = (
        n,
        input_t - 2,
        input_h - 2,
        input_w - 2,
        packed_weight.shape[0],
    )
    with stream_context(current_stream, input.device):
        workspace = torch.empty_like(input) if input.dtype == torch.float32 else None
        output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        residual_output = torch.empty(output_shape, dtype=input.dtype, device=input.device) if residual is not None else None

    _run_allocating_wrapper(
        ("norm_silu", allow_tf32),
        {"input": input, "packed_weight": packed_weight, "bias": bias, "gamma": gamma, "residual": residual, "residual_bias": residual_bias},
        {"workspace": workspace, "padded_output": output, "cache_output": None, "residual_output": residual_output},
        lambda: Conv3dRmsNormSiluSm100(
            input, packed_weight, bias, gamma, output, residual, residual_bias, residual_output, allow_tf32=allow_tf32, sample_workspace=workspace
        ),
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
    allow_tf32: bool = False,
) -> TupleDict:
    """Allocate padded/cache outputs; the caller must fill their history interiors.

    Only current frames, missing-history zeros, and spatial borders are written.
    See ``Conv3dRmsNormSiluPadSm100`` for the preallocated-output API.
    """
    n, input_t, input_h, input_w, _ = input.shape
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
        workspace = torch.empty_like(input) if input.dtype == torch.float32 else None
        padded_output = torch.empty(padded_shape, dtype=input.dtype, device=input.device)
        cache_output = torch.empty(cache_shape, dtype=input.dtype, device=input.device)
        residual_output = torch.empty(output_shape, dtype=input.dtype, device=input.device) if residual is not None else None

    _run_allocating_wrapper(
        ("norm_silu_pad", allow_tf32, history_frames),
        {"input": input, "packed_weight": packed_weight, "bias": bias, "gamma": gamma, "residual": residual, "residual_bias": residual_bias},
        {"workspace": workspace, "padded_output": padded_output, "cache_output": cache_output, "residual_output": residual_output},
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
            allow_tf32=allow_tf32,
            sample_workspace=workspace,
        ),
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
    allow_tf32: bool = False,
) -> TupleDict:
    """Allocate output and run fused Conv3D + bias + residual + spatial pad."""
    n, input_t, input_h, input_w, _ = input.shape
    output_channels = packed_weight.shape[0]
    padded_shape = (n, input_t - 2, input_h - 1, input_w - 1, output_channels)
    with stream_context(current_stream, input.device):
        workspace = torch.empty_like(input) if input.dtype == torch.float32 else None
        padded_output = torch.empty(padded_shape, dtype=input.dtype, device=input.device)

    _run_allocating_wrapper(
        ("bias_residual_pad", allow_tf32),
        {"input": input, "packed_weight": packed_weight, "bias": bias, "gamma": None, "residual": residual, "residual_bias": residual_bias},
        {"workspace": workspace, "padded_output": padded_output, "cache_output": None, "residual_output": None},
        lambda: Conv3dBiasResidualPadSm100(
            input, packed_weight, bias, residual, padded_output, residual_bias, allow_tf32=allow_tf32, sample_workspace=workspace
        ),
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
    """Allocate outputs and run ``RmsNormSiluPadSm100`` (BF16/FP32; no convolution)."""
    n, frames, height, width, channels = input.shape
    previous_frames = 0 if previous is None else previous.shape[1]
    padded_shape = (n, frames + 2, height + 2, width + 2, channels)
    cache_shape = (n, min(2, frames + previous_frames), height, width, channels)
    with stream_context(current_stream, input.device):
        padded_output = torch.empty(padded_shape, dtype=input.dtype, device=input.device)
        cache_output = torch.empty(cache_shape, dtype=input.dtype, device=input.device)
        residual_output = torch.empty_like(input) if residual is not None or save_input else None

    _run_allocating_wrapper(
        ("rmsnorm_silu_pad", save_input),
        {"input": input, "gamma": gamma, "input_bias": input_bias, "previous": previous, "residual": residual, "residual_bias": residual_bias},
        {"padded_output": padded_output, "cache_output": cache_output, "residual_output": residual_output},
        lambda: RmsNormSiluPadSm100(input, gamma, padded_output, cache_output, input_bias, previous, residual, residual_bias, residual_output),
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
