# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Base class, tensor metadata, and CuTe binding for optional JAX adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from operator import index
from typing import Any

from .. import data_type
from ..common.tensor_desc import TensorDesc
from .layout import compact_stride, normalize_mode, to_canonical_axes, to_cutlass_layout, to_public_axes


@dataclass(frozen=True)
class JaxTensorDesc(TensorDesc[Any]):
    """JAX tensor metadata and CUTLASS lowering constraints."""

    mode: tuple[int, ...] = field(default_factory=tuple)
    divisibility: tuple[int | None, ...] | None = None
    ptr_assumed_align: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self,
            "mode",
            normalize_mode(self.ndim, None if not self.mode else self.mode),
        )
        object.__setattr__(
            self,
            "divisibility",
            _normalize_divisibility(self.ndim, self.divisibility),
        )
        object.__setattr__(
            self,
            "ptr_assumed_align",
            _normalize_ptr_assumed_align(self.ptr_assumed_align),
        )

    @property
    def array_shape(self) -> tuple[int, ...]:
        """Return the shape in the axis order used by the JAX array."""

        return to_public_axes(self.shape, self.mode)

    @classmethod
    def from_array(
        cls,
        value: Any,
        *,
        name: str = "",
        mode: tuple[int, ...] | None = None,
        public_stride_order: tuple[int, ...] | None = None,
        init_value: bool | int | float | None = None,
        divisibility: tuple[int | None, ...] | None = None,
        ptr_assumed_align: int | None = None,
    ) -> "JaxTensorDesc":
        """Describe a JAX array-like value using its shape and dtype metadata."""

        shape = _require_array_metadata(value, name or "value")
        return cls.from_shape(
            shape,
            value.dtype,
            name=name,
            mode=mode,
            public_stride_order=public_stride_order,
            init_value=init_value,
            divisibility=divisibility,
            ptr_assumed_align=ptr_assumed_align,
        )

    @classmethod
    def from_shape(
        cls,
        shape: tuple[int, ...],
        dtype: Any,
        *,
        name: str = "",
        mode: tuple[int, ...] | None = None,
        public_stride_order: tuple[int, ...] | None = None,
        init_value: bool | int | float | None = None,
        divisibility: tuple[int | None, ...] | None = None,
        ptr_assumed_align: int | None = None,
    ) -> "JaxTensorDesc":
        """Describe a compact JAX tensor from its user-facing shape and dtype."""

        shape = tuple(shape)
        rank = len(shape)
        mode = normalize_mode(rank, mode)
        public_stride_order = _normalize_stride_order(rank, public_stride_order)
        public_stride = compact_stride(shape, public_stride_order)
        canonical_axis_by_public_axis = to_public_axes(tuple(range(rank)), mode)

        return cls(
            dtype=dtype,
            shape=to_canonical_axes(shape, mode),
            stride=to_canonical_axes(public_stride, mode),
            stride_order=tuple(
                canonical_axis_by_public_axis[axis]
                for axis in public_stride_order
            ),
            name=name,
            init_value=init_value,
            mode=mode,
            divisibility=divisibility,
            ptr_assumed_align=ptr_assumed_align,
        )

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
        mode: tuple[int, ...] | None = None,
        divisibility: tuple[int | None, ...] | None = None,
        ptr_assumed_align: int | None = None,
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
            mode=normalize_mode(canonical.ndim, mode),
            divisibility=divisibility,
            ptr_assumed_align=ptr_assumed_align,
        )

    def with_divisibility(
        self,
        divisibility: tuple[int | None, ...] | None,
    ) -> "JaxTensorDesc":
        """Return this descriptor with canonical-axis divisibility constraints."""

        return replace(self, divisibility=divisibility)


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


def _normalize_divisibility(
    rank: int,
    divisibility: tuple[int | None, ...] | None,
) -> tuple[int | None, ...] | None:
    if divisibility is None:
        return None
    divisibility = tuple(divisibility)
    if len(divisibility) != rank:
        raise ValueError(
            f"divisibility rank mismatch: expected {rank}, got {len(divisibility)}"
        )

    normalized = []
    for value in divisibility:
        if value is None:
            normalized.append(None)
            continue
        if isinstance(value, bool):
            raise TypeError(
                f"divisibility entries must be positive integers or None, got {value!r}"
            )
        try:
            value = index(value)
        except TypeError as error:
            raise TypeError(
                f"divisibility entries must be positive integers or None, got {value!r}"
            ) from error
        if value <= 0:
            raise ValueError(
                f"divisibility entries must be positive integers or None, got {value}"
            )
        normalized.append(value)
    return tuple(normalized)


def _normalize_ptr_assumed_align(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(
            f"ptr_assumed_align must be a positive integer or None, got {value!r}"
        )
    try:
        value = index(value)
    except TypeError as error:
        raise TypeError(
            f"ptr_assumed_align must be a positive integer or None, got {value!r}"
        ) from error
    if value <= 0:
        raise ValueError(
            f"ptr_assumed_align must be a positive integer or None, got {value}"
        )
    return value


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

    def _get_max_active_clusters(
        self,
        cluster_size: int,
        *,
        overlap_margin: int = 0,
    ) -> int:
        """Return cached occupancy for this adapter and cluster size.

        The hardware query performs its own CuTe compilation, so adapters must
        call this before entering ``cutlass.jax.cutlass_call`` lowering.
        """

        cache = getattr(self, "_max_active_clusters_cache", None)
        if cache is None:
            cache = {}
            self._max_active_clusters_cache = cache

        if cluster_size not in cache:
            import cutlass

            cache[cluster_size] = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_size)

        max_active_clusters = cache[cluster_size] - overlap_margin
        if max_active_clusters <= 0:
            raise ValueError("max_active_clusters must be positive after applying CUDNNFE_CLUSTER_OVERLAP_MARGIN")
        return max_active_clusters

    def _get_device_multiprocessor_count(self) -> int:
        """Return a cached local SM count before entering custom-call lowering."""

        count = getattr(self, "_device_multiprocessor_count", None)
        if count is None:
            import cutlass

            count = int(cutlass.utils.HardwareInfo().get_device_multiprocessor_count())
            if count <= 0:
                raise RuntimeError(f"CUDA reported an invalid multiprocessor count {count}")
            self._device_multiprocessor_count = count
        return count

    @staticmethod
    def _to_tensor_desc(
        value: Any,
        name: str,
        *,
        mode: tuple[int, ...] | None = None,
        public_stride_order: tuple[int, ...] | None = None,
        init_value: bool | int | float | None = None,
        divisibility: tuple[int | None, ...] | None = None,
        ptr_assumed_align: int | None = None,
    ) -> JaxTensorDesc:
        """Describe a public JAX array in canonical kernel-axis order.

        ``mode[kernel_axis]`` selects the corresponding public array axis.
        ``public_stride_order`` lists the public array dimensions from fastest
        to slowest. It defaults to compact row-major storage.
        """

        return JaxTensorDesc.from_array(
            value,
            name=name,
            mode=mode,
            public_stride_order=public_stride_order,
            init_value=init_value,
            divisibility=divisibility,
            ptr_assumed_align=ptr_assumed_align,
        )

    @staticmethod
    def _check_tensor_signature(
        value: Any,
        expected: JaxTensorDesc,
    ) -> None:
        """Validate a public JAX value against a canonical descriptor."""

        if not isinstance(expected, JaxTensorDesc):
            raise TypeError(
                f"expected must be a JaxTensorDesc, got {type(expected).__name__}"
            )
        name = expected.name or "value"
        public_shape = _require_array_metadata(value, name)
        if len(public_shape) != expected.ndim:
            raise ValueError(f"{name} tensor shape mismatch: expected {expected.shape}, got public shape {public_shape}")
        actual_shape = to_canonical_axes(public_shape, expected.mode)
        if actual_shape != expected.shape:
            raise ValueError(f"{name} tensor shape mismatch: expected {expected.shape}, got {actual_shape}")

        from .datatypes import jax_to_cudnn_dtype

        actual_dtype = jax_to_cudnn_dtype(value.dtype)
        if actual_dtype != expected.cudnn_dtype:
            raise ValueError(f"{name} tensor dtype mismatch: expected {expected.cudnn_dtype}, got {actual_dtype}")

    @staticmethod
    def _to_tensor_spec(
        desc: JaxTensorDesc,
    ) -> Any:
        """Build a CUTLASS TensorSpec indexed by public JAX array axes.

        ``desc.divisibility`` uses canonical kernel axes. TensorSpec
        layout and divisibility use public axes, while TensorSpec ``mode``
        records the canonical-to-public binding.
        """

        if not isinstance(desc, JaxTensorDesc):
            raise TypeError(
                f"desc must be a JaxTensorDesc, got {type(desc).__name__}"
            )
        public_layout = to_cutlass_layout(
            desc.shape,
            desc.stride,
            desc.stride_order,
            mode=desc.mode,
            name=desc.name or "tensor",
        )
        default_mode = tuple(range(desc.ndim))
        default_layout = tuple(reversed(range(desc.ndim)))
        if (
            desc.mode == default_mode
            and public_layout == default_layout
            and desc.divisibility is None
            and desc.ptr_assumed_align is None
        ):
            return None

        if desc.divisibility is None:
            public_divisibility = None
        else:
            public_divisibility = to_public_axes(
                desc.divisibility,
                desc.mode,
            )

        from cutlass.jax import TensorSpec

        options: dict[str, Any] = dict(
            layout=public_layout,
            mode=desc.mode,
        )
        if public_divisibility is not None:
            options["divisibility"] = public_divisibility
        if desc.ptr_assumed_align is not None:
            options["ptr_assumed_align"] = desc.ptr_assumed_align
        return TensorSpec(**options)

    @staticmethod
    def _to_shape_dtype_struct(
        desc: JaxTensorDesc,
    ) -> Any:
        """Convert a canonical descriptor to abstract JAX output metadata."""

        if not isinstance(desc, JaxTensorDesc):
            raise TypeError(f"desc must be a JaxTensorDesc, got {type(desc).__name__}")

        from jax import ShapeDtypeStruct

        from .datatypes import cudnn_to_jax_dtype

        return ShapeDtypeStruct(
            desc.array_shape,
            cudnn_to_jax_dtype(desc.cudnn_dtype),
        )

    def _call_kernel(
        self,
        inputs: tuple[Any, ...],
        *,
        launch: Callable[..., None],
        input_descs: tuple[JaxTensorDesc, ...],
        output_descs: tuple[JaxTensorDesc, ...],
        workspace_descs: tuple[JaxTensorDesc, ...] = (),
        allow_cuda_graph: bool = True,
        compile_options: Any = None,
        use_static_tensors: bool = True,
    ) -> tuple[Any, ...]:
        """Bind an explicit kernel launcher to JAX and return public outputs.

        CUTLASS JAX supplies the stream first; the launcher receives
        ``stream, *inputs, *outputs, *workspaces`` in that order. Workspaces
        are declared as custom-call results so XLA owns their lifetime, then
        omitted from this method's return value. Descriptors with a non-``None``
        ``init_value`` are materialized as JAX inputs and aliased to their
        corresponding custom-call results. Outputs without an initial value
        are allocated as custom-call results by XLA.
        Descriptors validate input signatures and provide all CUTLASS
        ``TensorSpec`` metadata.
        """

        import cutlass.cute as cute
        import cutlass.jax as cutlass_jax
        import jax.numpy as jnp

        if not callable(launch):
            raise TypeError(f"launch must be callable, got {type(launch).__name__}")

        inputs = tuple(inputs)
        for input_index, value in enumerate(inputs):
            _require_array_metadata(value, f"input #{input_index}")
        outputs = tuple(output_descs)
        workspaces = tuple(workspace_descs)
        if not outputs:
            raise ValueError("A JAX operation must provide at least one output descriptor")

        input_descs = tuple(input_descs)
        if len(input_descs) != len(inputs):
            raise ValueError(
                f"Expected {len(inputs)} input descriptors, got {len(input_descs)}"
            )

        for label, descs in (
            ("input", input_descs),
            ("output", outputs),
            ("workspace", workspaces),
        ):
            for desc in descs:
                if not isinstance(desc, JaxTensorDesc):
                    raise TypeError(
                        f"{label}_descs must contain JaxTensorDesc values, "
                        f"got {type(desc).__name__}"
                    )

        for value, desc in zip(inputs, input_descs):
            self._check_tensor_signature(value, desc)

        input_specs = tuple(self._to_tensor_spec(desc) for desc in input_descs)
        output_specs = tuple(self._to_tensor_spec(desc) for desc in outputs)
        workspace_specs = tuple(self._to_tensor_spec(desc) for desc in workspaces)
        buffers = outputs + workspaces
        buffer_specs = output_specs + workspace_specs
        buffer_metadata = tuple(
            self._to_shape_dtype_struct(desc) for desc in buffers
        )

        initialized_buffer_by_index: dict[int, Any] = {}
        for buffer_index, desc in enumerate(buffers):
            if desc.init_value is not None:
                initialized_buffer_by_index[buffer_index] = jnp.full(
                    buffer_metadata[buffer_index].shape,
                    desc.init_value,
                    dtype=buffer_metadata[buffer_index].dtype,
                )

        initialized = tuple(sorted(initialized_buffer_by_index))
        uninitialized = tuple(
            buffer_index
            for buffer_index in range(len(buffers))
            if buffer_index not in initialized_buffer_by_index
        )
        initialized_inputs = tuple(
            initialized_buffer_by_index[buffer_index]
            for buffer_index in initialized
        )
        aliases = {
            len(inputs) + initialized_input_index: result_index
            for initialized_input_index, result_index in enumerate(initialized)
        }
        call_input_specs = input_specs + tuple(
            buffer_specs[buffer_index] for buffer_index in initialized
        )

        input_count = len(inputs)
        initialized_count = len(initialized_inputs)

        @cute.jit(preprocess=False)
        def launcher(stream, *args):
            kernel_inputs = args[:input_count]
            initialized_buffers = args[
                input_count : input_count + initialized_count
            ]
            allocated_buffers = args[input_count + initialized_count :]
            if len(allocated_buffers) != len(uninitialized):
                raise RuntimeError(f"Kernel received {len(allocated_buffers)} allocated buffers; expected {len(uninitialized)}")

            ordered_buffers: list[Any] = [None] * len(buffers)
            for buffer_index, value in zip(initialized, initialized_buffers):
                ordered_buffers[buffer_index] = value
            for buffer_index, value in zip(uninitialized, allocated_buffers):
                ordered_buffers[buffer_index] = value
            output_buffers = tuple(ordered_buffers[: len(outputs)])
            workspace_buffers = tuple(ordered_buffers[len(outputs) :])
            launch(stream, *kernel_inputs, *output_buffers, *workspace_buffers)

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
        results = call(*inputs, *initialized_inputs)
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
