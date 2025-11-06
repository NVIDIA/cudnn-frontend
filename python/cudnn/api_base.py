# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Base classes for cuDNN API wrappers.

This module provides abstract base classes that define common interfaces
for cuDNN API wrapper classes, including validation, compilation, and execution patterns.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import logging
import cuda.bindings.driver as cuda
import cutlass


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
