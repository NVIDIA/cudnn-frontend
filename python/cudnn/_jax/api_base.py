# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Base class, tensor metadata, and CuTe binding for optional JAX adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from operator import index
from typing import Any

from .. import data_type
from .._tensor_desc import TensorDesc
from .layout import compact_stride, normalize_mode, to_canonical_axes, to_cutlass_layout, to_public_axes


class JaxTensorDesc(TensorDesc[Any]):
    """Framework-neutral tensor metadata backed by a JAX dtype."""

    @property
    def cudnn_dtype(self) -> data_type:
        from .datatypes import jax_to_cudnn_dtype

        return jax_to_cudnn_dtype(self.dtype)

    def compact_like(
        self,
        *,
        cudnn_dtype: data_type,
        shape: tuple[int, ...],
        stride_order: tuple[int, ...] | None = None,
        name: str = "",
        init_value: bool | int | float | None = None,
    ) -> "JaxTensorDesc":
        """Create a compact JAX descriptor from canonical output metadata."""

        canonical = super().compact_like(
            cudnn_dtype=cudnn_dtype,
            shape=shape,
            stride_order=stride_order,
            name=name,
            init_value=init_value,
        )

        from .datatypes import cudnn_to_jax_dtype

        return JaxTensorDesc(
            dtype=cudnn_to_jax_dtype(cudnn_dtype),
            shape=canonical.shape,
            stride=canonical.stride,
            stride_order=canonical.stride_order,
            name=canonical.name,
            init_value=canonical.init_value,
        )


def _require_array_metadata(value: Any, name: str) -> tuple[Any, ...]:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must have shape and dtype metadata")
    return tuple(value.shape)


def _normalize_stride_order(rank: int, stride_order: tuple[int, ...] | None) -> tuple[int, ...]:
    if stride_order is None:
        return tuple(reversed(range(rank)))

    normalized = []
    for dimension in stride_order:
        if isinstance(dimension, bool):
            raise TypeError(f"public_stride_order entries must be integers, got {dimension!r}")
        try:
            normalized.append(index(dimension))
        except TypeError as error:
            raise TypeError(f"public_stride_order entries must be integers, got {dimension!r}") from error

    normalized_order = tuple(normalized)
    if len(normalized_order) != rank:
        raise ValueError(f"public_stride_order rank mismatch: expected {rank}, got {len(normalized_order)}")
    if tuple(sorted(normalized_order)) != tuple(range(rank)):
        raise ValueError(f"public_stride_order must be a permutation of [0, {rank - 1}], got {normalized_order}")
    return normalized_order


class JaxApiBase(ABC):
    """Common tensor metadata, validation, and CuTe binding for JAX adapters."""

    @staticmethod
    def _device_compute_capability(device: Any, operation_name: str) -> int:
        """Normalize JAX's device capability metadata to ``major * 10 + minor``."""

        reported = getattr(device, "compute_capability", None)
        try:
            if isinstance(reported, (tuple, list)):
                if len(reported) < 2:
                    raise ValueError
                major, minor = int(reported[0]), int(reported[1])
            else:
                text = str(reported)
                if "." in text:
                    major_text, minor_text = text.split(".", 1)
                    major, minor = int(major_text), int(minor_text)
                else:
                    capability = int(text)
                    major, minor = divmod(capability, 10)
            if major < 0 or minor < 0 or minor > 9:
                raise ValueError
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"{operation_name}: JAX reported an invalid compute capability {reported!r} for {device}") from error
        return major * 10 + minor

    @staticmethod
    def _local_gpu_capabilities(operation_name: str) -> tuple[tuple[Any, int], ...]:
        import jax

        try:
            devices = tuple(jax.local_devices(backend="gpu"))
        except RuntimeError as error:
            raise RuntimeError(f"{operation_name}: JAX could not discover a local GPU") from error
        return tuple((device, JaxApiBase._device_compute_capability(device, operation_name)) for device in devices)

    @staticmethod
    def _compute_capability_family(
        compute_capability: int,
        supported_compute_capabilities: tuple[int, ...],
    ) -> int | None:
        compatible = tuple(capability for capability in supported_compute_capabilities if capability <= compute_capability)
        return max(compatible, default=None)

    @staticmethod
    def _resolve_compute_capability(
        target_compute_capability: int | None,
        supported_compute_capabilities: tuple[int, ...],
        operation_name: str,
    ) -> int:
        """Resolve an exact JAX compilation target for a multi-arch adapter.

        ``supported_compute_capabilities`` lists the exact CuTe compilation
        targets implemented by the adapter. The returned value remains exact
        (for example ``103``) so the compiler selects the matching target.
        An explicit target must match every local GPU exactly.
        """

        supported = tuple(sorted(set(supported_compute_capabilities)))
        if not supported or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in supported):
            raise ValueError(f"supported_compute_capabilities must contain positive integers, got {supported_compute_capabilities!r}")

        if target_compute_capability is not None:
            if isinstance(target_compute_capability, bool) or not isinstance(target_compute_capability, int):
                raise TypeError(f"target_compute_capability must be an int or None, got {type(target_compute_capability).__name__}")
            if target_compute_capability <= 0:
                raise ValueError(f"target_compute_capability must be positive, got {target_compute_capability}")
            if target_compute_capability not in supported:
                supported_text = ", ".join(f"SM{value}" for value in supported)
                raise ValueError(f"{operation_name} has no kernel for SM{target_compute_capability}; supported targets are {supported_text}")
            local = JaxApiBase._local_gpu_capabilities(operation_name)
            if not local:
                raise RuntimeError(f"{operation_name} targets SM{target_compute_capability}, but no local JAX GPU is available")
            mismatched = tuple((device, capability) for device, capability in local if capability != target_compute_capability)
            if mismatched:
                found = ", ".join(f"{device} (SM{capability})" for device, capability in mismatched)
                raise RuntimeError(f"{operation_name} targets SM{target_compute_capability}, but found {found}")
            return target_compute_capability

        local = JaxApiBase._local_gpu_capabilities(operation_name)
        if not local:
            raise RuntimeError(f"{operation_name}: no local JAX GPU is available")

        capabilities = tuple(sorted({capability for _, capability in local}))
        if len(capabilities) != 1:
            found = ", ".join(f"SM{capability}" for capability in capabilities)
            raise RuntimeError(f"{operation_name}: local JAX GPUs have heterogeneous targets ({found})")

        resolved = capabilities[0]
        if resolved not in supported:
            supported_text = ", ".join(f"SM{value}" for value in supported)
            raise RuntimeError(f"{operation_name} has no kernel for SM{resolved}; supported targets are {supported_text}")
        return resolved

    @staticmethod
    def _check_device_compatibility(
        *,
        minimum_compute_capability: int,
        operation_name: str,
    ) -> None:
        """Require every local JAX GPU to satisfy an operation's minimum SM."""

        def compatibility_error(reason: str) -> RuntimeError:
            return RuntimeError(f"{operation_name} requires SM{minimum_compute_capability}+, {reason}")

        try:
            devices = JaxApiBase._local_gpu_capabilities(operation_name)
        except RuntimeError as error:
            reason = str(error)
            prefix = f"{operation_name}: "
            if reason.startswith(prefix):
                reason = reason[len(prefix) :]
            raise compatibility_error(f"but {reason}") from error

        if not devices:
            raise compatibility_error("but no local JAX GPU is available")

        incompatible = []
        for device, compute_capability in devices:
            if compute_capability < minimum_compute_capability:
                incompatible.append((device, compute_capability))

        if incompatible:
            found = ", ".join(f"{device} (SM{compute_capability})" for device, compute_capability in incompatible)
            raise compatibility_error(f"found {found}")

    @staticmethod
    def _to_tensor_desc(
        value: Any,
        name: str,
        *,
        mode: tuple[int, ...] | None = None,
        public_stride_order: tuple[int, ...] | None = None,
        init_value: bool | int | float | None = None,
    ) -> JaxTensorDesc:
        """Describe a public JAX array in canonical kernel-axis order.

        ``mode[kernel_axis]`` selects the corresponding public array axis.
        ``public_stride_order`` lists the public array dimensions from fastest
        to slowest. It defaults to compact row-major storage.
        """

        public_shape = _require_array_metadata(value, name)
        rank = len(public_shape)
        mode = normalize_mode(rank, mode)
        public_stride_order = _normalize_stride_order(rank, public_stride_order)
        public_stride = compact_stride(public_shape, public_stride_order)
        canonical_axis_by_public_axis = to_public_axes(tuple(range(rank)), mode)

        return JaxTensorDesc(
            dtype=value.dtype,
            shape=to_canonical_axes(public_shape, mode),
            stride=to_canonical_axes(public_stride, mode),
            stride_order=tuple(canonical_axis_by_public_axis[axis] for axis in public_stride_order),
            name=name,
            init_value=init_value,
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
        output_descs: tuple[TensorDesc[Any], ...],
        workspace_descs: tuple[TensorDesc[Any], ...] = (),
        input_spec: tuple[Any | None, ...] | None = None,
        output_spec: tuple[Any | None, ...] | None = None,
        workspace_spec: tuple[Any | None, ...] | None = None,
        allow_cuda_graph: bool = True,
        compile_options: Any = None,
        use_static_tensors: bool = True,
    ) -> tuple[Any, ...]:
        """Bind this adapter's launch hook to JAX and return public outputs.

        CUTLASS JAX supplies the stream first; the launcher adapts that call to
        the adapter's ``inputs, outputs, workspaces, stream`` launch hook. Workspaces
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
        outputs = tuple(output_descs)
        workspaces = tuple(workspace_descs)
        if not outputs:
            raise ValueError("A JAX operation must provide at least one output descriptor")

        for label, descs in (("output", outputs), ("workspace", workspaces)):
            for desc in descs:
                if not isinstance(desc, TensorDesc):
                    raise TypeError(f"{label}_descs must contain TensorDesc values, got {type(desc).__name__}")

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
            output_buffers = tuple(ordered_buffers[: len(outputs)])
            workspace_buffers = tuple(ordered_buffers[len(outputs) :])
            self._launch(tuple(kernel_inputs), output_buffers, workspace_buffers, stream)

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
    def _launch(
        self,
        inputs: tuple[Any, ...],
        outputs: tuple[Any, ...],
        workspaces: tuple[Any, ...],
        stream: Any,
    ) -> None:
        """Launch the concrete CuTe kernel for one traced custom call."""

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
