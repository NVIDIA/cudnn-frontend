# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Base class, tensor metadata, and CuTe binding for optional JAX adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from .. import data_type
from .._op_kernel import OpKernel
from .._tensor_desc import TensorDesc
from .layout import compact_stride, normalize_mode, to_canonical_axes, to_cutlass_layout, to_public_axes


class JaxTensorDesc(TensorDesc[Any]):
    """Framework-neutral tensor metadata backed by a JAX dtype."""

    @property
    def cudnn_dtype(self) -> data_type:
        from .datatypes import jax_to_cudnn_dtype

        return jax_to_cudnn_dtype(self.dtype)


def _require_array_metadata(value: Any, name: str) -> tuple[Any, ...]:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must have shape and dtype metadata")
    return tuple(value.shape)


class JaxApiBase(ABC):
    """Common tensor metadata, validation, and kernel binding for JAX adapters."""

    kernel: OpKernel

    @staticmethod
    def _to_tensor_desc(
        value: Any,
        name: str,
        *,
        mode: tuple[int, ...] | None = None,
    ) -> JaxTensorDesc:
        """Describe a public JAX array in canonical kernel-axis order.

        ``mode[kernel_axis]`` selects the corresponding public array axis.
        Public JAX arrays are modeled as compact row-major buffers.
        """

        public_shape = _require_array_metadata(value, name)
        rank = len(public_shape)
        mode = normalize_mode(rank, mode)
        public_stride_order = tuple(reversed(range(rank)))
        public_stride = compact_stride(public_shape, public_stride_order)
        canonical_axis_by_public_axis = to_public_axes(tuple(range(rank)), mode)

        return JaxTensorDesc(
            dtype=value.dtype,
            shape=to_canonical_axes(public_shape, mode),
            stride=to_canonical_axes(public_stride, mode),
            stride_order=tuple(canonical_axis_by_public_axis[axis] for axis in public_stride_order),
            name=name,
        )

    @staticmethod
    def _check_tensor_signature(
        value: Any,
        expected: TensorDesc[Any],
        *,
        mode: tuple[int, ...] | None = None,
    ) -> None:
        """Validate a public JAX value against a canonical descriptor."""

        name = expected.name or "value"
        public_shape = _require_array_metadata(value, name)
        mode = normalize_mode(expected.ndim, mode)
        if len(public_shape) != expected.ndim:
            raise ValueError(f"{name} tensor shape mismatch: expected {expected.shape}, got public shape {public_shape}")
        actual_shape = to_canonical_axes(public_shape, mode)
        if actual_shape != expected.shape:
            raise ValueError(f"{name} tensor shape mismatch: expected {expected.shape}, got {actual_shape}")

        from .datatypes import jax_to_cudnn_dtype

        actual_dtype = jax_to_cudnn_dtype(value.dtype)
        if actual_dtype != expected.cudnn_dtype:
            raise ValueError(f"{name} tensor dtype mismatch: expected {expected.cudnn_dtype}, got {actual_dtype}")

    @staticmethod
    def _to_tensor_spec(
        desc: TensorDesc[Any],
        *,
        mode: tuple[int, ...] | None = None,
        divisibility: tuple[int | None, ...] | None = None,
    ) -> Any:
        """Build a CUTLASS TensorSpec indexed by public JAX array axes.

        ``desc`` and ``divisibility`` use canonical kernel axes. TensorSpec
        layout and divisibility use public axes, while TensorSpec ``mode``
        records the canonical-to-public binding.
        """

        mode = normalize_mode(desc.ndim, mode)
        public_layout = to_cutlass_layout(
            desc.shape,
            desc.stride,
            desc.stride_order,
            mode=mode,
            name=desc.name or "tensor",
        )
        if divisibility is None:
            public_divisibility = None
        else:
            divisibility = tuple(divisibility)
            if len(divisibility) != desc.ndim:
                raise ValueError(f"divisibility rank mismatch: expected {desc.ndim}, got {len(divisibility)}")
            public_divisibility = to_public_axes(divisibility, mode)

        from cutlass.jax import TensorSpec

        return TensorSpec(
            layout=public_layout,
            mode=mode,
            divisibility=public_divisibility,
        )

    @staticmethod
    def _materialize_tensor_desc(
        desc: TensorDesc[Any],
        *,
        mode: tuple[int, ...] | None = None,
    ) -> Any:
        """Declare a canonical descriptor as a public JAX output buffer."""

        if not isinstance(desc, TensorDesc):
            raise TypeError(f"desc must be a TensorDesc, got {type(desc).__name__}")
        mode = normalize_mode(desc.ndim, mode)

        from jax import ShapeDtypeStruct

        from .datatypes import cudnn_to_jax_dtype

        return ShapeDtypeStruct(
            to_public_axes(desc.shape, mode),
            cudnn_to_jax_dtype(desc.cudnn_dtype),
        )

    def _call_kernel(
        self,
        inputs: tuple[Any, ...],
        *,
        input_spec: tuple[Any | None, ...] | None = None,
        output_spec: tuple[Any | None, ...] | None = None,
        workspace_spec: tuple[Any | None, ...] | None = None,
        allow_cuda_graph: bool = True,
        compile_options: Any = None,
        use_static_tensors: bool = True,
    ) -> tuple[Any, ...]:
        """Bind the owned kernel to JAX and return its public outputs.

        CUTLASS JAX supplies the stream first; the launcher adapts that call to
        the kernel's ``inputs, outputs, workspaces, stream`` order. Workspaces
        are declared as custom-call results so XLA owns their lifetime, then
        omitted from this method's return value. Descriptors with a non-``None``
        ``init_value`` are materialized as JAX inputs and aliased to their
        corresponding custom-call results. Inputs are a flat tuple of arrays,
        and all explicit specs are native CUTLASS ``TensorSpec`` values.
        """

        import cutlass.jax as cutlass_jax
        import jax.numpy as jnp

        inputs = tuple(inputs)
        for index, value in enumerate(inputs):
            _require_array_metadata(value, f"input #{index}")
        outputs = tuple(self.kernel.infer_output())
        workspaces = tuple(self.kernel.infer_workspace())
        if not outputs:
            raise ValueError("A JAX operation must infer at least one output")

        for label, descs in (("output", outputs), ("workspace", workspaces)):
            for desc in descs:
                if not isinstance(desc, TensorDesc):
                    raise TypeError(f"{label} inference must return TensorDesc values, got {type(desc).__name__}")

        if input_spec is None:
            input_specs = (None,) * len(inputs)
        else:
            input_specs = tuple(input_spec)
            if len(input_specs) != len(inputs):
                raise ValueError(f"Expected {len(inputs)} input specs, got {len(input_specs)}")
        if any(spec is not None and not isinstance(spec, cutlass_jax.TensorSpec) for spec in input_specs):
            raise TypeError("input_spec must contain only TensorSpec or None values")

        def resolve_specs(
            descs: tuple[TensorDesc[Any], ...],
            supplied: tuple[Any | None, ...] | None,
            label: str,
        ) -> tuple[Any, ...]:
            if supplied is None:
                supplied = (None,) * len(descs)
            else:
                supplied = tuple(supplied)
                if len(supplied) != len(descs):
                    raise ValueError(f"Expected {len(descs)} {label} specs, got {len(supplied)}")
                if any(spec is not None and not isinstance(spec, cutlass_jax.TensorSpec) for spec in supplied):
                    raise TypeError(f"{label}_spec must contain only TensorSpec or None values")
            return tuple(self._to_tensor_spec(desc) if spec is None else spec for desc, spec in zip(descs, supplied))

        output_specs = resolve_specs(outputs, output_spec, "output")
        workspace_specs = resolve_specs(workspaces, workspace_spec, "workspace")
        buffers = outputs + workspaces
        buffer_specs = output_specs + workspace_specs
        buffer_metadata = tuple(self._materialize_tensor_desc(desc, mode=getattr(spec, "mode", None)) for desc, spec in zip(buffers, buffer_specs))

        initialized = tuple(index for index, desc in enumerate(buffers) if desc.init_value is not None)
        uninitialized = tuple(index for index, desc in enumerate(buffers) if desc.init_value is None)
        seed_inputs = tuple(
            jnp.full(
                buffer_metadata[index].shape,
                buffers[index].init_value,
                dtype=buffer_metadata[index].dtype,
            )
            for index in initialized
        )
        aliases = {len(inputs) + seed_index: result_index for seed_index, result_index in enumerate(initialized)}
        call_input_specs = input_specs + tuple(buffer_specs[index] for index in initialized)

        input_count = len(inputs)
        seed_count = len(seed_inputs)

        def launcher(stream, *args):
            kernel_inputs = args[:input_count]
            seeded_buffers = args[input_count : input_count + seed_count]
            allocated_buffers = args[input_count + seed_count :]
            if len(allocated_buffers) != len(uninitialized):
                raise RuntimeError(f"Kernel received {len(allocated_buffers)} allocated buffers; expected {len(uninitialized)}")

            ordered_buffers: list[Any] = [None] * len(buffers)
            for index, value in zip(initialized, seeded_buffers):
                ordered_buffers[index] = value
            for index, value in zip(uninitialized, allocated_buffers):
                ordered_buffers[index] = value
            self.kernel(*kernel_inputs, *ordered_buffers, stream)

        call = cutlass_jax.cutlass_call(
            launcher,
            output_shape_dtype=buffer_metadata,
            input_spec=call_input_specs,
            output_spec=buffer_specs,
            input_output_aliases=aliases,
            allow_cuda_graph=allow_cuda_graph,
            compile_options=compile_options,
            use_static_tensors=use_static_tensors,
        )
        results = call(*inputs, *seed_inputs)
        if not isinstance(results, (tuple, list)):
            results = (results,)
        return tuple(results[: len(outputs)])

    @abstractmethod
    def check_support(self) -> bool:
        """Validate the operation signature and static configuration."""

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Trace or execute the operation with JAX array arguments."""

    def get_jax_callable(self) -> Callable[..., Any]:
        """Return this callable without imposing a JIT policy."""

        return self


__all__ = ["JaxApiBase", "JaxTensorDesc"]
