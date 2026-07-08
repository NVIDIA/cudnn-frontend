# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Torch implementation of the FE-OSS API lifecycle."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional
import cuda.bindings.driver as cuda
import cutlass
import torch

import cutlass.cute as cute
from .api_base import ApiBase, TensorDesc, TupleDict, ceil_div, is_power_of_2
from .datatypes import _convert_to_cutlass_data_type


@dataclass(frozen=True, kw_only=True)
class TorchTensorDesc(TensorDesc):
    """Torch tensor metadata, including observed storage layout and device."""

    dtype: torch.dtype
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    stride_order: Tuple[int, ...]
    device: torch.device
    interpret_uint8_as_fp4x2: bool = False
    name: str = ""

    def __post_init__(self):
        raw_dtype_name = str(self.dtype).rsplit(".", 1)[-1].lower()
        is_uint8 = self.dtype == getattr(torch, "uint8", None) or raw_dtype_name == "uint8"
        is_native_fp4x2 = self.dtype == getattr(torch, "float4_e2m1fn_x2", None) or raw_dtype_name == "float4_e2m1fn_x2"
        if (self.interpret_uint8_as_fp4x2 and is_uint8) or is_native_fp4x2:
            object.__setattr__(self, "packing", "fp4x2")
        super().__post_init__()
        device = self.device
        if not isinstance(device, torch.device):
            try:
                device = torch.device(device)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise TypeError(f"Invalid device for TorchTensorDesc: {self.device!r}") from exc
        object.__setattr__(self, "device", device)

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        *,
        name: str = "",
        interpret_uint8_as_fp4x2: bool = False,
    ) -> "TorchTensorDesc":
        """Capture logical shape and observed layout from a Torch tensor."""

        shape = tuple(tensor.shape)
        stride = tuple(tensor.stride())
        is_packed_fp4 = tensor.dtype == getattr(torch, "float4_e2m1fn_x2", None) or (interpret_uint8_as_fp4x2 and tensor.dtype == torch.uint8)
        if is_packed_fp4:
            innermost_dim = next((dim for dim, value in enumerate(stride) if value == 1), None)
            if innermost_dim is None:
                raise RuntimeError(f"tensor {name} has shape: {shape} stride {stride} – innermost contiguous (stride == 1) dimension not found. ")
            shape = tuple(size * 2 if dim == innermost_dim else size for dim, size in enumerate(shape))
            stride = tuple(value if dim == innermost_dim else value * 2 for dim, value in enumerate(stride))

        stride_order = tuple(dim for dim, _ in sorted(enumerate(stride), key=lambda item: (item[1], shape[item[0]])))
        return cls(
            dtype=tensor.dtype,
            shape=shape,
            stride=stride,
            stride_order=stride_order,
            device=tensor.device,
            interpret_uint8_as_fp4x2=interpret_uint8_as_fp4x2,
            name=name,
        )

    @staticmethod
    def _normalize_dim(dim: int, ndim: int, *, allow_new_dim: bool = False) -> int:
        min_dim = -ndim - (1 if allow_new_dim else 0)
        max_dim = ndim if allow_new_dim else ndim - 1
        if dim < min_dim or dim > max_dim:
            raise IndexError(f"Dimension out of range (expected to be in range of [{min_dim}, {max_dim}], but got {dim})")
        if dim < 0:
            dim += ndim + (1 if allow_new_dim else 0)
        return dim

    @staticmethod
    def _compute_stride_order(shape: Tuple[int, ...], stride: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(i for i, _ in sorted(enumerate(stride), key=lambda x: (x[1], shape[x[0]])))

    @staticmethod
    def _numel(shape: Tuple[int, ...]) -> int:
        numel = 1
        for size in shape:
            numel *= size
        return numel

    @staticmethod
    def _compute_contiguous_stride(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if not shape:
            return ()
        strides = [0] * len(shape)
        running = 1
        for i in range(len(shape) - 1, -1, -1):
            strides[i] = running
            running *= max(shape[i], 1)
        return tuple(strides)

    @staticmethod
    def _is_contiguous_with_order(shape: Tuple[int, ...], stride: Tuple[int, ...], order: Tuple[int, ...]) -> bool:
        expected_stride = 1
        for dim in order:
            size = shape[dim]
            if size == 1:
                continue
            if stride[dim] != expected_stride:
                return False
            expected_stride *= size
        return True

    @staticmethod
    def _compute_view_stride(
        old_shape: Tuple[int, ...],
        old_stride: Tuple[int, ...],
        new_shape: Tuple[int, ...],
    ) -> Optional[Tuple[int, ...]]:
        old_numel = TorchTensorDesc._numel(old_shape)
        if old_numel == 0:
            return TorchTensorDesc._compute_contiguous_stride(new_shape)

        new_stride = [0] * len(new_shape)
        view_dim = len(new_shape) - 1
        tensor_numel = 1
        view_numel = 1

        for tensor_dim in range(len(old_shape) - 1, -1, -1):
            tensor_numel *= old_shape[tensor_dim]
            is_contiguous_chunk_end = tensor_dim == 0 or (
                old_shape[tensor_dim - 1] != 1 and old_stride[tensor_dim - 1] != tensor_numel * old_stride[tensor_dim]
            )
            if is_contiguous_chunk_end:
                while view_dim >= 0 and (view_numel < tensor_numel or new_shape[view_dim] == 1):
                    new_stride[view_dim] = view_numel * old_stride[tensor_dim]
                    view_numel *= new_shape[view_dim]
                    view_dim -= 1

                if view_numel != tensor_numel:
                    return None

                if tensor_dim > 0:
                    tensor_numel = 1
                    view_numel = 1

        if view_dim != -1:
            return None
        return tuple(new_stride)

    def _with_layout(self, shape: Tuple[int, ...], stride: Tuple[int, ...]) -> "TorchTensorDesc":
        return TorchTensorDesc(
            dtype=self.dtype,
            shape=shape,
            stride=stride,
            stride_order=self._compute_stride_order(shape, stride),
            device=self.device,
            interpret_uint8_as_fp4x2=self.interpret_uint8_as_fp4x2,
            packing=self.packing,
            name=self.name,
        )

    def __len__(self) -> int:
        if self.ndim == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    def size(self, dim: Optional[int] = None) -> torch.Size | int:
        if dim is None:
            return torch.Size(self.shape)
        dim = self._normalize_dim(int(dim), self.ndim)
        return self.shape[dim]

    def permute(self, *dims: int | Tuple[int, ...] | List[int]) -> "TensorDesc":
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        dims = tuple(int(d) for d in dims)
        if len(dims) != self.ndim:
            raise RuntimeError(f"permute(): expected {self.ndim} dims, got {len(dims)}")
        dims = tuple(self._normalize_dim(d, self.ndim) for d in dims)
        if len(set(dims)) != self.ndim:
            raise RuntimeError(f"permute(): dims must be unique, got {dims}")

        new_shape = tuple(self.shape[d] for d in dims)
        new_stride = tuple(self.stride[d] for d in dims)
        return self._with_layout(new_shape, new_stride)

    def transpose(self, dim0: int, dim1: int) -> "TensorDesc":
        dim0 = self._normalize_dim(dim0, self.ndim)
        dim1 = self._normalize_dim(dim1, self.ndim)
        if dim0 == dim1:
            return self
        dims = list(range(self.ndim))
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        return self.permute(dims)

    def squeeze(self, dim: Optional[int | Tuple[int, ...] | List[int]] = None) -> "TensorDesc":
        if dim is None:
            keep_dims = [i for i, size in enumerate(self.shape) if size != 1]
        elif isinstance(dim, (tuple, list)):
            squeeze_dims = tuple(self._normalize_dim(int(d), self.ndim) for d in dim)
            if len(set(squeeze_dims)) != len(squeeze_dims):
                raise RuntimeError(f"squeeze(): dims must be unique, got {squeeze_dims}")
            squeeze_dims = {d for d in squeeze_dims if self.shape[d] == 1}
            keep_dims = [i for i in range(self.ndim) if i not in squeeze_dims]
        else:
            squeeze_dim = self._normalize_dim(int(dim), self.ndim)
            if self.shape[squeeze_dim] != 1:
                return self
            keep_dims = [i for i in range(self.ndim) if i != squeeze_dim]

        new_shape = tuple(self.shape[i] for i in keep_dims)
        new_stride = tuple(self.stride[i] for i in keep_dims)
        if new_shape == self.shape and new_stride == self.stride:
            return self
        return self._with_layout(new_shape, new_stride)

    def unsqueeze(self, dim: int) -> "TensorDesc":
        dim = self._normalize_dim(dim, self.ndim, allow_new_dim=True)

        if dim >= self.ndim:
            inserted_stride = 1
        else:
            inserted_stride = self.stride[dim] * self.shape[dim]

        new_shape = self.shape[:dim] + (1,) + self.shape[dim:]
        new_stride = self.stride[:dim] + (inserted_stride,) + self.stride[dim:]
        return self._with_layout(new_shape, new_stride)

    def is_contiguous(self, memory_format: torch.memory_format = torch.contiguous_format) -> bool:
        if memory_format in {torch.contiguous_format, torch.preserve_format}:
            if self._numel(self.shape) == 0:
                return True
            return self._is_contiguous_with_order(self.shape, self.stride, tuple(range(self.ndim - 1, -1, -1)))
        if memory_format == torch.channels_last:
            if self.ndim != 4:
                return False
            return self._is_contiguous_with_order(self.shape, self.stride, (1, 3, 2, 0))
        if memory_format == torch.channels_last_3d:
            if self.ndim != 5:
                return False
            return self._is_contiguous_with_order(self.shape, self.stride, (1, 4, 3, 2, 0))

        raise ValueError(f"Unsupported memory format: {memory_format}")

    def contiguous(self) -> "TensorDesc":
        if self.is_contiguous():
            return self
        contiguous_stride = self._compute_contiguous_stride(self.shape)
        return self._with_layout(self.shape, contiguous_stride)

    def view(self, *shape: int | Tuple[int, ...] | List[int]) -> "TensorDesc":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        new_shape = tuple(int(s) for s in shape)

        old_numel = self._numel(self.shape)
        infer_dim = None
        known_numel = 1
        for i, size in enumerate(new_shape):
            if size == -1:
                if infer_dim is not None:
                    raise RuntimeError("only one dimension can be inferred")
                infer_dim = i
                continue
            if size < 0:
                raise RuntimeError(f"invalid shape dimension {size}")
            known_numel *= size

        if infer_dim is not None:
            if known_numel == 0 or old_numel % known_numel != 0:
                raise RuntimeError(f"shape '{new_shape}' is invalid for input of size {old_numel}")
            inferred_size = old_numel // known_numel
            new_shape = new_shape[:infer_dim] + (inferred_size,) + new_shape[infer_dim + 1 :]
            known_numel *= inferred_size

        if known_numel != old_numel:
            raise RuntimeError(f"shape '{new_shape}' is invalid for input of size {old_numel}")

        new_stride = self._compute_view_stride(self.shape, self.stride, new_shape)
        if new_stride is None:
            raise RuntimeError(
                "view size is not compatible with input tensor's size and stride " "(at least one dimension spans across two contiguous subspaces)"
            )

        return self._with_layout(new_shape, new_stride)

    def as_strided(
        self,
        size: Tuple[int, ...] | List[int],
        stride: Tuple[int, ...] | List[int],
        storage_offset: int = 0,
    ) -> "TensorDesc":
        if storage_offset != 0:
            raise RuntimeError("TensorDesc.as_strided(): non-zero storage_offset is unsupported")
        if not isinstance(size, (tuple, list)) or not isinstance(stride, (tuple, list)):
            raise TypeError("TensorDesc.as_strided(): size and stride must be tuple/list")

        size = tuple(int(s) for s in size)
        stride = tuple(int(s) for s in stride)
        if len(size) != len(stride):
            raise RuntimeError(f"TensorDesc.as_strided(): mismatch in length of size ({len(size)}) and stride ({len(stride)})")
        if any(s < 0 for s in size):
            raise RuntimeError(f"TensorDesc.as_strided(): invalid size, got {size}")
        if any(s < 0 for s in stride):
            raise RuntimeError(f"TensorDesc.as_strided(): invalid stride, got {stride}")

        return self._with_layout(size, stride)


class ApiBaseTorch(ApiBase, ABC):
    """Abstract base class for cuDNN API wrappers.

    This class defines the common interface that all API wrapper implementations
    should follow, including configuration validation, compilation, and execution.

    Provides common functionality:
    - Logging via self._logger
    - Support validation tracking via self._is_supported
    - Compiled kernel caching via self._compiled_kernel
    - Stream management helpers

    Subclasses should implement the abstract methods to provide
    API-specific validation logic and execution behavior.

    Example:
        >>> class MyKernelAPI(ApiBaseTorch):
        ...     def __init__(self, sample_input, sample_output, config):
        ...         super().__init__()
        ...         self.sample_input = sample_input
        ...         self.sample_output = sample_output
        ...         self.config = config
        ...         self._kernel = MyKernel
        ...
        ...     def check_support(self) -> bool:
        ...         # Validate inputs and configuration
        ...         assert self.sample_input.dtype == torch.float32
        ...         self._is_supported = True
        ...         return True
        ...
        ...     def compile(self):
        ...         self._ensure_support_checked()
        ...         # Create and compile kernel
        ...         kernel = self._kernel(self.config)
        ...         self._compiled_kernel = cute.compile(kernel, ...)
        ...
        ...     def execute(self, input_tensor, output_tensor, current_stream=None):
        ...         current_stream = self._get_default_stream(current_stream)
        ...         self._compiled_kernel(input_tensor, output_tensor, current_stream)
    """

    def __init__(self):
        """Initialize the API base.

        Sets up:
        - self._is_supported: Flag indicating if configuration is validated
        - self._kernel: Kernel instance
        - self._compiled_kernel: Cache for compiled kernel
        - self._logger: Logger instance for this class
        """
        super().__init__()
        self._kernel = None
        self._compiled_kernel = None
        self._interpret_uint8_as_fp4x2 = False

    @abstractmethod
    def check_support(self) -> bool:
        """Check if the current configuration is supported by the kernel.

        This method should validate:
        - Input/output tensor shapes and strides
        - Data types compatibility
        - Hardware capabilities (compute capability, memory, etc.)
        - Configuration parameters (tile sizes, cluster shapes, etc.)

        Implementations should set self._is_supported = True if valid.

        :return: True if the configuration is supported
        :rtype: bool
        :raises AssertionError: If a configuration requirement is not met

        Example:
            >>> def check_support(self) -> bool:
            ...     self._logger.debug("Checking support")
            ...     assert self.input.dtype in {torch.float16, torch.float32}
            ...     assert self.input.shape[0] % 16 == 0, "Shape must be 16-aligned"
            ...     self._is_supported = True
            ...     return True
        """
        pass

    @abstractmethod
    def compile(self) -> None:
        """Compile the kernel with the current configuration.

        This method should:
        1. Ensure support has been checked (use self._ensure_support_checked())
        2. Create the underlying kernel implementation and fake cute tensors from the sample tensor descriptors
        3. Compile the kernel using cute.compile()
        4. Cache the compiled kernel in self._compiled_kernel

        :raises AssertionError: If the configuration is not supported

        Example:
            >>> def compile(self):
            ...     self._ensure_support_checked()
            ...
            ...     kernel = self._kernel(self.config)
            ...     sample_input_cute = self._make_fake_cute_tensor_from_desc(self.sample_input)
            ...     sample_output_cute = self._make_fake_cute_tensor_from_desc(self.sample_output)
            ...
            ...     self._compiled_kernel = cute.compile(
            ...         kernel,
            ...         sample_input_cute,
            ...         sample_output_cute
            ...     )
        """
        pass

    @abstractmethod
    def execute(
        self,
        *args,
        current_stream: Optional[cuda.CUstream] = None,
        **kwargs,
    ) -> Any:
        """Execute the kernel with the provided inputs.

        This method should execute using the cached compiled kernel.

        :param args: Positional arguments (typically input/output tensors)
        :param current_stream: CUDA stream for execution (optional)
        :type current_stream: cuda.CUstream or None
        :param kwargs: Additional keyword arguments for execution
        :return: Execution result (if any)
        :raises AssertionError: If compiled kernel is not available

        Example:
            >>> def execute(self, input_tensor, output_tensor, current_stream=None):
            ...     current_stream = self._get_default_stream(current_stream)
            ...     assert self._compiled_kernel is not None, "Kernel not compiled"
            ...     self._logger.debug("Executing with compiled kernel")
            ...     self._compiled_kernel(input_tensor, output_tensor, current_stream)
        """
        pass

    def __call__(self, *args, **kwargs) -> Any:
        """Convenience method to execute the kernel.

        This is a shorthand for compiling (if needed) and then executing.

        :param args: Positional arguments passed to execute()
        :param kwargs: Keyword arguments passed to execute()
        :return: Result from execute()

        Example:
            >>> api = MyKernelAPI(...)
            >>> api.check_support()
            >>> api.compile()
            >>> api(input_tensor, output_tensor)
        """
        if self._compiled_kernel is None:
            self.compile()
        return self.execute(*args, **kwargs)

    def _get_default_stream(self, stream: Optional[cuda.CUstream]) -> cuda.CUstream:
        """Get default CUDA stream if none provided.

        This is a convenience helper to handle optional stream parameters.
        If a stream is provided, it is returned as-is. If None, the default
        CUDA stream is returned.

        :param stream: CUDA stream or None
        :type stream: cuda.CUstream or None
        :return: CUDA stream (either provided or default)
        :rtype: cuda.CUstream

        Example:
            >>> def execute(self, input_tensor, output_tensor, current_stream=None):
            ...     current_stream = self._get_default_stream(current_stream)
            ...     # Now current_stream is guaranteed to be a valid stream
        """
        if stream is None:
            self._logger.debug(f"{self.__class__.__name__}: No CUDA stream provided, using default stream")
            return cutlass.cuda.default_stream()
        return stream

    def _pad_tensor_to_ndim(
        self,
        tensor: Optional[torch.Tensor | TensorDesc],
        ndim: int,
        name: str,
    ) -> Optional[torch.Tensor | TensorDesc]:
        """Pad a tensor/descriptor by unsqueezing at dim -1 until it reaches ndim rank.

        - If tensor is None, returns None.
        - Unsqueezes at dim -1 until tensor/descriptor rank == ndim.
        - Logs final reshape for traceability.

        :param tensor: The tensor/descriptor to pad (or None)
        :param ndim: Target rank (pad trailing dims until reached)
        :param name: Logical tensor name for logging
        :return: The padded tensor/descriptor (or None)
        """
        if tensor is None:
            return None

        if tensor.ndim < ndim:
            self._logger.info(f"Padding {name} to {ndim}D from {tensor.shape}")
            for _ in range(ndim - tensor.ndim):
                tensor = tensor.unsqueeze(-1)
        return tensor

    def _unpad_tensor_to_ndim(
        self,
        tensor: Optional[torch.Tensor | TensorDesc],
        ndim: int,
        name: str,
    ) -> Optional[torch.Tensor | TensorDesc]:
        """Unpad a tensor/descriptor by squeezing at dim -1 until it reaches ndim rank.

        - If tensor is None, returns None.
        - Squeezes at dim -1 until tensor/descriptor rank == ndim.
        - Logs final reshape for traceability.

        :param tensor: The tensor/descriptor to unpad (or None)
        :param ndim: Target rank (squeeze trailing dims until reached)
        :param name: Logical tensor name for logging
        :return: The unpadded tensor/descriptor (or None)
        """
        if tensor is None:
            return None

        if tensor.ndim > ndim:
            self._logger.info(f"Unpadding {name} from {tensor.shape} to {ndim}D")
            for _ in range(tensor.ndim - ndim):
                if tensor.shape and tensor.shape[-1] == 1:
                    tensor = tensor.squeeze(-1)
                else:
                    break

            if tensor.ndim != ndim:
                self._logger.critical(f"Unpadding {name} resulted in shape {tensor.shape}, expected {ndim}D")
        return tensor

    def _is_fp4x2(self, tensor_or_dtype: torch.Tensor | torch.dtype | TensorDesc) -> bool:
        """Check if tensor or dtype is an FP4x2 packed datatype.

        :param tensor_or_dtype: The torch tensor or dtype to check
        :type tensor_or_dtype: torch.Tensor | torch.dtype
        :return: True if tensor/dtype is an FP4x2 packed type
        :rtype: bool
        """
        if tensor_or_dtype is None:
            return False
        if isinstance(tensor_or_dtype, TensorDesc):
            return tensor_or_dtype.is_fp4
        elif isinstance(tensor_or_dtype, torch.Tensor):
            dtype = tensor_or_dtype.dtype
            interpret_uint8_as_fp4x2 = self._interpret_uint8_as_fp4x2
        else:
            dtype = tensor_or_dtype
            interpret_uint8_as_fp4x2 = self._interpret_uint8_as_fp4x2
        return (dtype == getattr(torch, "float4_e2m1fn_x2", None)) or (interpret_uint8_as_fp4x2 and dtype == torch.uint8)

    def _is_fp8(self, tensor_or_dtype: torch.Tensor | torch.dtype | TensorDesc) -> bool:
        """Check if tensor or dtype is an FP8 datatype.

        :param tensor_or_dtype: The torch tensor or dtype to check
        :type tensor_or_dtype: torch.Tensor | torch.dtype
        :return: True if tensor/dtype is an FP8 type
        :rtype: bool
        """
        if tensor_or_dtype is None:
            return False
        if isinstance(tensor_or_dtype, TensorDesc):
            return tensor_or_dtype.is_fp8
        dtype = tensor_or_dtype.dtype if isinstance(tensor_or_dtype, (torch.Tensor, TensorDesc)) else tensor_or_dtype
        return dtype in {torch.float8_e5m2, torch.float8_e4m3fn}

    def _is_f16(self, tensor_or_dtype: torch.Tensor | torch.dtype | TensorDesc) -> bool:
        """Check if tensor or dtype is an fp16 or bf16 datatype.

        :param tensor_or_dtype: The torch tensor or dtype to check
        :type tensor_or_dtype: torch.Tensor | torch.dtype
        :return: True if tensor/dtype is an fp16 or bf16 type
        :rtype: bool
        """
        if tensor_or_dtype is None:
            return False
        if isinstance(tensor_or_dtype, TensorDesc):
            return tensor_or_dtype.is_f16
        dtype = tensor_or_dtype.dtype if isinstance(tensor_or_dtype, (torch.Tensor, TensorDesc)) else tensor_or_dtype
        return dtype in {torch.float16, torch.bfloat16}

    def _get_innermost_stride_dim(self, tensor: torch.Tensor, name: str = "") -> int:
        """Return index of innermost contiguous dimension (stride == 1).

        :raises RuntimeError: If no dimension with stride 1 is found.
        """
        idx = next((i for i, s in enumerate(tensor.stride()) if s == 1), None)
        if idx is None:
            self._logger.critical(
                f"tensor {name} has shape: {tensor.shape} stride {tensor.stride()} – innermost contiguous (stride == 1) dimension not found. "
            )
            raise RuntimeError(f"tensor {name} has shape: {tensor.shape} stride {tensor.stride()} – innermost contiguous (stride == 1) dimension not found. ")
        return idx

    def _tensor_shape(
        self,
        tensor: Optional[torch.Tensor | TensorDesc],
        name: str = "",
    ) -> Optional[Tuple[int, ...]]:
        """Get the logical shape of a tensor, handling FP4x2 packed datatypes.

        For FP4x2 datatypes, two values are packed per byte. The innermost
        contiguous dimension (with stride 1) contains packed values, so the
        logical shape for that dimension is 2x the physical shape.

        :param tensor: The tensor to get shape from (or None)
        :type tensor: torch.Tensor or None
        :param name: Logical tensor name for logging
        :type name: str
        :return: The logical shape tuple (or None if tensor is None)
        :rtype: Tuple[int, ...] or None
        """
        if tensor is None:
            return None
        if isinstance(tensor, TensorDesc):
            return tensor.shape

        if self._is_fp4x2(tensor):
            innermost_dim_index = self._get_innermost_stride_dim(tensor, name=name)
            shape = tuple(dim * 2 if i == innermost_dim_index else dim for i, dim in enumerate(tensor.shape))
            self._logger.debug(f"FP4x2 tensor {name}: physical shape {tensor.shape} -> logical shape {shape}")
            return shape
        else:
            return tensor.shape

    def _tensor_stride(
        self,
        tensor: Optional[torch.Tensor | TensorDesc],
        name: str = "",
    ) -> Optional[Tuple[int, ...]]:
        """Get the logical stride of a tensor, handling FP4x2 packed datatypes.

        For FP4x2 datatypes, two values are packed per byte. The strides must
        be adjusted to reflect logical element spacing. All strides are
        multiplied by 2 since each physical element contains 2 logical elements.

        :param tensor: The tensor to get stride from (or None)
        :type tensor: torch.Tensor or None
        :param name: Logical tensor name for logging
        :type name: str
        :return: The logical stride tuple (or None if tensor is None)
        :rtype: Tuple[int, ...] or None
        """
        if tensor is None:
            return None
        if isinstance(tensor, TensorDesc):
            return tensor.stride

        if self._is_fp4x2(tensor):
            innermost_dim_index = self._get_innermost_stride_dim(tensor, name=name)
            strides = tuple(s * 2 if i != innermost_dim_index else s for i, s in enumerate(tensor.stride()))
            self._logger.debug(f"FP4x2 tensor {name}: physical stride {tensor.stride()} -> logical stride {strides}")
            return strides
        else:
            return tensor.stride()

    def _check_tensor_shape(
        self,
        tensor_or_shape: torch.Tensor | TensorDesc | Tuple[int, ...],
        shape: Tuple[int, ...] | List[Tuple[int, ...]],
        name: str = "",
    ) -> Optional[Tuple[int, ...]]:
        """Check if the shape of a tensor matches the expected shape(s).

        :param tensor_or_shape: The tensor to get shape from or the shape to check
        :type tensor_or_shape: torch.Tensor | TensorDesc | Tuple[int, ...]
        :param shape: expected shape or list of expected shapes
        :type shape: Tuple[int, ...] | List[Tuple[int, ...]]
        :param name: Logical tensor name for logging
        :type name: str
        :raises ValueError: If the shape of the tensor does not match the expected shape(s)
        :return: The logical shape of the tensor
        :rtype: Optional[Tuple[int, ...]]
        """
        if tensor_or_shape is None:
            return None
        tensor_shape = self._tensor_shape(tensor_or_shape, name=name) if isinstance(tensor_or_shape, (torch.Tensor, TensorDesc)) else tensor_or_shape
        if not isinstance(shape, (tuple, list)):
            raise ValueError(f"Expected shape to be a tuple or list, got {type(shape)}")
        return super().check_tensor_shape(tensor_shape, shape, name)

    def _check_tensor_stride(
        self,
        tensor_or_stride: torch.Tensor | TensorDesc | Tuple[int, ...],
        stride: Optional[Tuple[int, ...] | List[Tuple[int, ...]]] = None,
        stride_order: Optional[Tuple[int, ...] | List[Tuple[int, ...]]] = None,
        name: str = "",
        extra_error_msg: str = "",
    ) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        """Check if the stride of a tensor matches the expected stride(s) or stride order(s).

        :param tensor_or_stride: The tensor to get stride from or the stride to check
        :type tensor_or_stride: torch.Tensor | TensorDesc | Tuple[int, ...]
        :param stride: The expected stride(s)
        :type stride: Tuple[int, ...] | List[Tuple[int, ...]]
        :param stride_order: The expected stride order(s)
        :type stride_order: Tuple[int, ...] | List[Tuple[int, ...]]
        :param name: Logical tensor name for logging
        :type name: str
        :param extra_error_msg: Extra error message to add to the error
        :type extra_error_msg: str
        :raises ValueError: If the stride of the tensor does not match the expected stride order
        :return: The stride and stride order of the tensor
        :rtype: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]
        """
        value = self._tensor_stride(tensor_or_stride, name=name) if isinstance(tensor_or_stride, torch.Tensor) else tensor_or_stride
        return super().check_tensor_stride(
            value,
            stride=stride,
            stride_order=stride_order,
            name=name,
            extra_error_msg=extra_error_msg,
        )

    def _check_dtype(
        self,
        tensor_or_dtype: torch.Tensor | TensorDesc | torch.dtype,
        dtype: torch.dtype | List[torch.dtype],
        name: str = "",
        extra_error_msg: str = "",
    ) -> Optional[torch.dtype]:
        """Check if the dtype of a tensor or dtype matches the expected dtype(s).

        :param tensor_or_dtype: The tensor to get dtype from or the dtype to check
        :type tensor_or_dtype: torch.Tensor | TensorDesc | torch.dtype
        :param dtype: The expected dtype(s)
        :type dtype: torch.dtype | List[torch.dtype]
        :param name: Logical tensor name for logging
        :type name: str
        :raises ValueError: If the dtype of the tensor does not match the expected dtype(s)
        :return: The dtype of the tensor
        :rtype: Optional[torch.dtype]
        """
        if tensor_or_dtype is None:
            return None
        tensor_dtype = tensor_or_dtype.dtype if isinstance(tensor_or_dtype, (torch.Tensor, TensorDesc)) else tensor_or_dtype
        if not isinstance(dtype, (torch.dtype, list)):
            raise ValueError(f"Expected dtype to be a torch.dtype or list, got {type(dtype)}")
        super().check_dtype(tensor_dtype, dtype, name, extra_error_msg)
        return tensor_dtype

    def _make_fake_cute_tensor_like(
        self,
        tensor: torch.Tensor,
        assumed_align: int = 16,
        name: str = "",
    ) -> cute.Pointer:
        """Make a fake tensor like the provided tensor.
        :param tensor: The tensor to make a fake tensor like
        :type tensor: torch.Tensor
        :param assumed_align: The assumed alignment of the tensor
        :type assumed_align: int
        :param name: Logical tensor name for logging
        :type name: str
        :return: A fake tensor like the provided tensor
        :rtype: cute.Pointer
        """
        return self._make_fake_cute_tensor_from_desc(
            self._make_tensor_desc(tensor, name=name),
            assumed_align=assumed_align,
        )

    def _make_tensor_desc(
        self,
        tensor: Optional[torch.Tensor],
        name: str = "",
        interpret_uint8_as_fp4x2: Optional[bool] = None,
    ) -> Optional[TensorDesc]:
        """Capture logical tensor metadata that is sufficient for validation/compile."""
        if tensor is None:
            return None
        if interpret_uint8_as_fp4x2 is None:
            interpret_uint8_as_fp4x2 = self._interpret_uint8_as_fp4x2
        return TorchTensorDesc.from_tensor(
            tensor,
            interpret_uint8_as_fp4x2=interpret_uint8_as_fp4x2,
            name=name,
        )

    def make_tensor_desc(
        self,
        tensor: Optional[torch.Tensor],
        name: str = "",
        interpret_uint8_as_fp4x2: Optional[bool] = None,
    ) -> Optional[TorchTensorDesc]:
        """Return the kernel-visible metadata for a Torch tensor."""

        return self._make_tensor_desc(
            tensor,
            name=name,
            interpret_uint8_as_fp4x2=interpret_uint8_as_fp4x2,
        )

    def _make_fake_cute_tensor_from_desc(
        self,
        tensor_desc: Optional[TensorDesc],
        assumed_align: int = 16,
    ) -> Optional[cute.Pointer]:
        """Build a fake cute tensor from a descriptor."""
        if tensor_desc is None:
            return None
        return self._make_fake_cute_tensor(
            dtype=tensor_desc.dtype,
            shape=tensor_desc.shape,
            stride=tensor_desc.stride,
            assumed_align=assumed_align,
            interpret_uint8_as_fp4x2=tensor_desc.packing == "fp4x2",
        )

    def _make_fake_cute_tensor(
        self,
        dtype: torch.dtype,
        shape: Tuple[int, ...],
        stride: Tuple[int, ...],
        assumed_align: int = 16,
        interpret_uint8_as_fp4x2: Optional[bool] = None,
    ) -> cute.Pointer:
        """Make a fake tensor.

        :param dtype: The dtype of the tensor
        :type dtype: torch.dtype
        :param shape: The shape of the tensor
        :type shape: Tuple[int, ...]
        :param stride: The stride of the tensor
        :type stride: Tuple[int, ...]
        :param assumed_align: The assumed alignment of the tensor
        :type assumed_align: int
        :return: A fake tensor
        :rtype: cute.Pointer
        """
        if interpret_uint8_as_fp4x2 is None:
            interpret_uint8_as_fp4x2 = self._interpret_uint8_as_fp4x2
        return cute.runtime.make_fake_tensor(
            dtype=_convert_to_cutlass_data_type(dtype, interpret_uint8_as_fp4x2=interpret_uint8_as_fp4x2),
            shape=shape,
            stride=stride,
            assumed_align=assumed_align,
        )

    def _make_fake_cute_compact_tensor(
        self,
        dtype: torch.dtype,
        shape: Tuple[int, ...],
        stride_order: Tuple[int, ...],
        assumed_align: int = 16,
        dynamic_mode: Optional[int] = None,
        divisibility: int = 16,
        interpret_uint8_as_fp4x2: Optional[bool] = None,
    ) -> cute.Pointer:
        """Make a fake compact tensor.
        :param dtype: The dtype of the tensor
        :type dtype: torch.dtype
        :param shape: The shape of the tensor
        :type shape: Tuple[int, ...]
        :param stride_order: The stride order of the tensor
        :type stride_order: Tuple[int, ...]
        :param assumed_align: The assumed alignment of the tensor
        :type assumed_align: int
        :return: A fake compact tensor
        :rtype: cute.Pointer
        """
        if dynamic_mode is not None:
            dynamic_dim = cute.sym_int(divisibility=divisibility)
            shape = shape[:dynamic_mode] + (dynamic_dim,) + shape[dynamic_mode + 1 :]

        if interpret_uint8_as_fp4x2 is None:
            interpret_uint8_as_fp4x2 = self._interpret_uint8_as_fp4x2
        return cute.runtime.make_fake_compact_tensor(
            dtype=_convert_to_cutlass_data_type(dtype, interpret_uint8_as_fp4x2=interpret_uint8_as_fp4x2),
            shape=shape,
            stride_order=stride_order,
            assumed_align=assumed_align,
        )


# Source-compatible spelling retained for downstream imports and subclasses.
# Internal Torch wrappers use ``ApiBaseTorch`` so their framework is explicit.
APIBase = ApiBaseTorch


__all__ = [
    "APIBase",
    "ApiBaseTorch",
    "TorchTensorDesc",
    "TupleDict",
    "ceil_div",
    "is_power_of_2",
]
