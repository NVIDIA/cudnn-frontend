# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Base classes for cuDNN API wrappers.

This module provides abstract base classes that define common interfaces
for cuDNN API wrapper classes, including validation, compilation, and execution patterns.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional
import logging
import cuda.bindings.driver as cuda
import cutlass
import torch

import cutlass.cute as cute
from cudnn.datatypes import _convert_to_cutlass_data_type


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


class APIBase(ABC):
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
        >>> class MyKernelAPI(APIBase):
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
        ...     def compile(self, current_stream=None):
        ...         current_stream = self._get_default_stream(current_stream)
        ...         self._ensure_support_checked()
        ...         # Create and compile kernel
        ...         kernel = self._kernel(self.config)
        ...         self._compiled_kernel = cute.compile(kernel, ...)
        ...
        ...     def execute(self, input_tensor, output_tensor,
        ...                current_stream=None, skip_compile=False):
        ...         current_stream = self._get_default_stream(current_stream)
        ...         if not skip_compile:
        ...             self._compiled_kernel(input_tensor, output_tensor, current_stream)
        ...         else:
        ...             # Direct execution without cached compilation
        ...             kernel = self._kernel(self.config)
        ...             kernel(input_tensor, output_tensor, current_stream)
    """

    def __init__(self):
        """Initialize the API base.

        Sets up:
        - self._is_supported: Flag indicating if configuration is validated
        - self._kernel: Kernel instance
        - self._compiled_kernel: Cache for compiled kernel
        - self._logger: Logger instance for this class
        """
        self._is_supported = False
        self._kernel = None
        self._compiled_kernel = None
        self._interpret_uint8_as_fp4x2 = False
        self._logger = logging.getLogger(self.__class__.__name__)

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
    def compile(self, current_stream: Optional[cuda.CUstream] = None) -> None:
        """Compile the kernel with the current configuration.

        This method should:
        1. Ensure support has been checked (use self._ensure_support_checked())
        2. Get default stream if needed (use self._get_default_stream())
        3. Create the underlying kernel implementation
        4. Compile the kernel using cute.compile()
        5. Cache the compiled kernel in self._compiled_kernel

        :param current_stream: CUDA stream for compilation (optional)
        :type current_stream: cuda.CUstream or None
        :raises AssertionError: If the configuration is not supported

        Example:
            >>> def compile(self, current_stream=None):
            ...     current_stream = self._get_default_stream(current_stream)
            ...     self._ensure_support_checked()
            ...
            ...     kernel = self._kernel(self.config)
            ...     self._compiled_kernel = cute.compile(
            ...         kernel,
            ...         self.sample_input,
            ...         self.sample_output,
            ...         current_stream
            ...     )
        """
        pass

    @abstractmethod
    def execute(
        self,
        *args,
        current_stream: Optional[cuda.CUstream] = None,
        skip_compile: bool = False,
        **kwargs,
    ) -> Any:
        """Execute the kernel with the provided inputs.

        This method should handle two execution modes:
        1. With compiled kernel (skip_compile=False): Use self._compiled_kernel
        2. Without compiled kernel (skip_compile=True): Create and execute kernel directly (JIT)

        :param args: Positional arguments (typically input/output tensors)
        :param current_stream: CUDA stream for execution (optional)
        :type current_stream: cuda.CUstream or None
        :param skip_compile: If False, use cached compiled kernel;
                            If True, create and execute kernel directly
        :type skip_compile: bool
        :param kwargs: Additional keyword arguments for execution
        :return: Execution result (if any)
        :raises AssertionError: If compiled kernel is not available when skip_compile=False

        Example:
            >>> def execute(self, input_tensor, output_tensor,
            ...            current_stream=None, skip_compile=False):
            ...     current_stream = self._get_default_stream(current_stream)
            ...
            ...     if not skip_compile:
            ...         assert self._compiled_kernel is not None, "Kernel not compiled"
            ...         self._logger.debug("Executing with compiled kernel")
            ...         self._compiled_kernel(input_tensor, output_tensor, current_stream)
            ...     else:
            ...         self._logger.debug("Executing without compiled kernel (JIT)")
            ...         kernel = self._kernel(self.config)
            ...         kernel(input_tensor, output_tensor, current_stream)
        """
        pass

    def __call__(self, *args, **kwargs) -> Any:
        """Convenience method to execute the kernel.

        This is a shorthand for calling execute() with skip_compile=True,
        which bypasses the cached compiled kernel and executes directly.
        This is useful for one-off executions or when you want to ensure
        fresh compilation.

        :param args: Positional arguments passed to execute()
        :param kwargs: Keyword arguments passed to execute()
        :return: Result from execute()

        Example:
            >>> api = MyKernelAPI(...)
            >>> api.check_support()
            >>> # Direct execution without pre-compilation
            >>> api(input_tensor, output_tensor)  # Equivalent to execute(..., skip_compile=True)
        """
        return self.execute(*args, skip_compile=True, **kwargs)

    def _ensure_support_checked(self) -> None:
        """Helper to ensure check_support() was called before compilation.

        If check_support() has not been called yet (self._is_supported is False),
        this method will automatically call it. This prevents compilation
        with invalid configurations.

        :raises AssertionError: If check_support() returns False or raises

        Example:
            >>> def compile(self, current_stream=None):
            ...     self._ensure_support_checked()  # Automatic validation
            ...     # ... rest of compilation
        """
        if not self._is_supported:
            self._logger.info(
                f"{self.__class__.__name__}: check_support not previously called, calling now"
            )
            assert self.check_support(), "Unsupported configuration"

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
            >>> def compile(self, current_stream=None):
            ...     current_stream = self._get_default_stream(current_stream)
            ...     # Now current_stream is guaranteed to be a valid stream
        """
        if stream is None:
            self._logger.debug(
                f"{self.__class__.__name__}: No CUDA stream provided, using default stream"
            )
            return cutlass.cuda.default_stream()
        return stream

    def _pad_tensor_to_ndim(
        self,
        tensor: Optional[torch.Tensor],
        ndim: int,
        name: str,
    ) -> Optional[torch.Tensor]:
        """Pad a tensor by unsqueezing at dim -1 until it reaches ndim rank.

        - If tensor is None, returns None.
        - Unsqueezes at dim -1 until tensor.ndim == ndim.
        - Logs final reshape for traceability.

        :param tensor: The tensor to pad (or None)
        :param ndim: Target rank (pad trailing dims until reached)
        :param name: Logical tensor name for logging
        :return: The padded tensor (or None)
        """
        if (tensor is not None) and (tensor.ndim < ndim):
            self._logger.info(f"Padding {name} to {ndim}D from {tensor.shape}")
            for _ in range(ndim - tensor.ndim):
                tensor = tensor.unsqueeze(-1)
        return tensor

    def _unpad_tensor_to_ndim(
        self,
        tensor: Optional[torch.Tensor],
        ndim: int,
        name: str,
    ) -> Optional[torch.Tensor]:
        """Unpad a tensor by squeezing at dim -1 until it reaches ndim rank.

        - If tensor is None, returns None.
        - Squeezes at dim -1 until tensor.ndim == ndim.
        - Logs final reshape for traceability.

        :param tensor: The tensor to unpad (or None)
        :param ndim: Target rank (squeeze trailing dims until reached)
        :param name: Logical tensor name for logging
        :return: The unpadded tensor (or None)
        """
        if (tensor is not None) and (tensor.ndim > ndim):
            self._logger.info(f"Unpadding {name} from {tensor.shape} to {ndim}D")
            for _ in range(tensor.ndim - ndim):
                tensor = tensor.squeeze(-1)
            if tensor.ndim != ndim:
                self._logger.critical(
                    f"Unpadding {name} resulted in shape {tensor.shape}, expected {ndim}D"
                )
        return tensor

    def _is_fp4x2(self, tensor_or_dtype: torch.Tensor | torch.dtype) -> bool:
        """Check if tensor or dtype is an FP4x2 packed datatype.

        :param tensor_or_dtype: The torch tensor or dtype to check
        :type tensor_or_dtype: torch.Tensor | torch.dtype
        :return: True if tensor/dtype is an FP4x2 packed type
        :rtype: bool
        """
        if tensor_or_dtype is None:
            return False
        dtype = (
            tensor_or_dtype.dtype
            if isinstance(tensor_or_dtype, torch.Tensor)
            else tensor_or_dtype
        )
        return (dtype == torch.float4_e2m1fn_x2) or (
            self._interpret_uint8_as_fp4x2 and dtype == torch.uint8
        )

    def _is_fp8(self, tensor_or_dtype: torch.Tensor | torch.dtype) -> bool:
        """Check if tensor or dtype is an FP8 datatype.

        :param tensor_or_dtype: The torch tensor or dtype to check
        :type tensor_or_dtype: torch.Tensor | torch.dtype
        :return: True if tensor/dtype is an FP8 type
        :rtype: bool
        """
        if tensor_or_dtype is None:
            return False
        dtype = (
            tensor_or_dtype.dtype
            if isinstance(tensor_or_dtype, torch.Tensor)
            else tensor_or_dtype
        )
        return dtype in {torch.float8_e5m2, torch.float8_e4m3fn}

    def _get_innermost_stride_dim(self, tensor: torch.Tensor, name: str = "") -> int:
        """Return index of innermost contiguous dimension (stride == 1).

        :raises RuntimeError: If no dimension with stride 1 is found.
        """
        idx = next((i for i, s in enumerate(tensor.stride()) if s == 1), None)
        if idx is None:
            self._logger.critical(
                f"tensor {name} has shape: {tensor.shape} stride {tensor.stride()} – innermost contiguous (stride == 1) dimension not found. "
            )
            raise RuntimeError(
                f"tensor {name} has shape: {tensor.shape} stride {tensor.stride()} – innermost contiguous (stride == 1) dimension not found. "
            )
        return idx

    def _tensor_shape(
        self,
        tensor: Optional[torch.Tensor],
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

        if self._is_fp4x2(tensor):
            innermost_dim_index = self._get_innermost_stride_dim(tensor, name=name)
            shape = tuple(
                dim * 2 if i == innermost_dim_index else dim
                for i, dim in enumerate(tensor.shape)
            )
            self._logger.debug(
                f"FP4x2 tensor {name}: physical shape {tensor.shape} -> logical shape {shape}"
            )
            return shape
        else:
            return tensor.shape

    def _tensor_stride(
        self,
        tensor: Optional[torch.Tensor],
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

        if self._is_fp4x2(tensor):
            innermost_dim_index = self._get_innermost_stride_dim(tensor, name=name)
            strides = tuple(
                s * 2 if i != innermost_dim_index else s
                for i, s in enumerate(tensor.stride())
            )
            self._logger.debug(
                f"FP4x2 tensor {name}: physical stride {tensor.stride()} -> logical stride {strides}"
            )
            return strides
        else:
            return tensor.stride()

    def _check_tensor_shape(
        self,
        tensor_or_shape: torch.Tensor | Tuple[int, ...],
        shape: Tuple[int, ...] | List[Tuple[int, ...]],
        name: str = "",
    ) -> Optional[Tuple[int, ...]]:
        """Check if the shape of a tensor matches the expected shape(s).

        :param tensor_or_shape: The tensor to get shape from or the shape to check
        :type tensor_or_shape: torch.Tensor | Tuple[int, ...]
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
        tensor_shape = (
            self._tensor_shape(tensor_or_shape, name=name)
            if isinstance(tensor_or_shape, torch.Tensor)
            else tensor_or_shape
        )
        if isinstance(shape, tuple):
            if tensor_shape != shape:
                raise ValueError(
                    f"{name} tensor shape mismatch: expected {shape}, got {tensor_shape}"
                )
        elif isinstance(shape, list):
            if tensor_shape not in shape:
                raise ValueError(
                    f"{name} tensor shape mismatch: expected one of {shape}, got {tensor_shape}"
                )
        else:
            raise ValueError(f"Expected shape to be a tuple or list, got {type(shape)}")
        return tensor_shape

    def _check_tensor_stride(
        self,
        tensor_or_stride: torch.Tensor | Tuple[int, ...],
        stride: Optional[Tuple[int, ...] | List[Tuple[int, ...]]] = None,
        stride_order: Optional[Tuple[int, ...] | List[Tuple[int, ...]]] = None,
        name: str = "",
    ) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        """Check if the stride of a tensor matches the expected stride(s) or stride order(s).

        :param tensor_or_stride: The tensor to get stride from or the stride to check
        :type tensor_or_stride: torch.Tensor | Tuple[int, ...]
        :param stride: The expected stride(s)
        :type stride: Tuple[int, ...] | List[Tuple[int, ...]]
        :param stride_order: The expected stride order(s)
        :type stride_order: Tuple[int, ...] | List[Tuple[int, ...]]
        :param name: Logical tensor name for logging
        :type name: str
        :raises ValueError: If the stride of the tensor does not match the expected stride order
        :return: The stride and stride order of the tensor
        :rtype: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]
        """
        if tensor_or_stride is None:
            return None
        tensor_stride = (
            self._tensor_stride(tensor_or_stride, name=name)
            if isinstance(tensor_or_stride, torch.Tensor)
            else tensor_or_stride
        )
        tensor_stride_order = tuple(
            i for i, s in sorted(enumerate(tensor_stride), key=lambda x: x[1])
        )

        if stride is not None:
            if isinstance(stride, tuple):
                if tensor_stride != stride:
                    raise ValueError(
                        f"{name} tensor stride mismatch: expected {stride}, got {tensor_stride}"
                    )
            elif isinstance(stride, list):
                if tensor_stride not in stride:
                    raise ValueError(
                        f"{name} tensor stride mismatch: expected one of {stride}, got {tensor_stride}"
                    )
            else:
                raise ValueError(
                    f"Expected stride to be a tuple or list, got {type(stride)}"
                )
        if stride_order is not None:
            if isinstance(stride_order, tuple):
                if tensor_stride_order != stride_order:
                    raise ValueError(
                        f"{name} tensor stride order mismatch: expected {stride_order}, got {tensor_stride_order}"
                    )
            elif isinstance(stride_order, list):
                if tensor_stride_order not in stride_order:
                    raise ValueError(
                        f"{name} tensor stride order mismatch: expected one of {stride_order}, got {tensor_stride_order}"
                    )
            else:
                raise ValueError(
                    f"Expected stride order to be a tuple or list, got {type(stride_order)}"
                )
        return tensor_stride, tensor_stride_order

    def _check_dtype(
        self,
        tensor_or_dtype: torch.Tensor | torch.dtype,
        dtype: torch.dtype | List[torch.dtype],
        name: str = "",
        extra_error_msg: str = "",
    ) -> Optional[torch.dtype]:
        """Check if the dtype of a tensor or dtype matches the expected dtype(s).

        :param tensor_or_dtype: The tensor to get dtype from or the dtype to check
        :type tensor_or_dtype: torch.Tensor | torch.dtype
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
        tensor_dtype = (
            tensor_or_dtype.dtype
            if isinstance(tensor_or_dtype, torch.Tensor)
            else tensor_or_dtype
        )
        if isinstance(dtype, torch.dtype):
            if tensor_dtype != dtype:
                error_msg = (
                    f"{name} dtype mismatch: expected {dtype}, got {tensor_dtype}"
                )
                if extra_error_msg:
                    error_msg += f": {extra_error_msg}"
                raise ValueError(error_msg)
        elif isinstance(dtype, list):
            if tensor_dtype not in dtype:
                error_msg = f"{name} dtype mismatch: expected one of {dtype}, got {tensor_dtype}"
                if extra_error_msg:
                    error_msg += f": {extra_error_msg}"
                raise ValueError(error_msg)
        else:
            raise ValueError(
                f"Expected dtype to be a torch.dtype or list, got {type(dtype)}"
            )
        return tensor_dtype

    def _value_error_if(self, condition: bool, error_msg: str) -> None:
        """Raise a ValueError if the condition is true.

        :param condition: The condition to check
        :type condition: bool
        :param error_msg: The error message to raise
        :type error_msg: str
        :raises ValueError: If the condition is true
        """
        if condition:
            raise ValueError(error_msg)

    def _not_implemented_error_if(self, condition: bool, error_msg: str) -> None:
        """Raise a NotImplementedError if the condition is true.

        :param condition: The condition to check
        :type condition: bool
        :param error_msg: The error message to raise
        :type error_msg: str
        :raises NotImplementedError: If the condition is true
        """
        if condition:
            raise NotImplementedError(error_msg)

    def _runtime_error_if(self, condition: bool, error_msg: str) -> None:
        """Raise a RuntimeError if the condition is true.

        :param condition: The condition to check
        :type condition: bool
        :param error_msg: The error message to raise
        :type error_msg: str
        :raises RuntimeError: If the condition is true
        """
        if condition:
            raise RuntimeError(error_msg)

    def _make_cute_pointer(
        self, tensor: torch.Tensor, assumed_align: int = 16
    ) -> cute.Pointer:
        """Make a cute.Pointer for a tensor.

        :param tensor: The tensor to make a cute.Pointer for
        :type tensor: torch.Tensor
        :param assumed_align: The assumed alignment of the tensor
        :type assumed_align: int
        :return: A cute.Pointer for the tensor
        :rtype: cute.Pointer
        """
        if tensor is None:
            return None
        return cute.runtime.make_ptr(
            _convert_to_cutlass_data_type(
                tensor.dtype, interpret_uint8_as_fp4x2=self._interpret_uint8_as_fp4x2
            ),
            tensor.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=assumed_align,
        )

    def _make_cute_tensor_descriptor(
        self, tensor: torch.Tensor, assumed_align: int = 16, name: str = ""
    ) -> Tuple[cute.Pointer, Tuple[int, ...], Tuple[int, ...]]:
        """Make a cute.Pointer, shape, and order for a tensor.

        :param tensor: The tensor to make a cute.Pointer, shape, and order for
        :type tensor: torch.Tensor
        :param assumed_align: The assumed alignment of the tensor
        :type assumed_align: int
        :param name: Logical tensor name for logging
        :type name: str
        :return: A cute.Pointer, shape, and stride order for the tensor
        :rtype: Tuple[cute.Pointer, Tuple[int, ...], Tuple[int, ...]]
        """
        if tensor is None:
            return None, None, None
        tensor_ptr = self._make_cute_pointer(tensor, assumed_align=assumed_align)
        tensor_shape = self._tensor_shape(tensor, name=name)
        tensor_stride = self._tensor_stride(tensor, name=name)
        tensor_stride_order = tuple(
            i for i, s in sorted(enumerate(tensor_stride), key=lambda x: x[1])
        )
        return tensor_ptr, tensor_shape, tensor_stride_order
