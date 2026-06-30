# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-light catalog for Torch-first FE-OSS operation support."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Mapping, Optional, Sequence, Tuple, Union


class FrontendTarget(str, Enum):
    """Framework targets recorded by the FE-OSS support catalog."""

    TORCH = "torch"
    JAX = "jax"

    @classmethod
    def normalize(cls, value: Union["FrontendTarget", str]) -> "FrontendTarget":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("target must be a Python-static 'torch' or 'jax' value, " f"got {type(value).__name__}")
        try:
            return cls(value.lower())
        except ValueError as exc:
            choices = ", ".join(target.value for target in cls)
            raise ValueError(f"Unknown frontend target {value!r}; expected {choices}") from exc


@dataclass(frozen=True)
class TargetBinding:
    """Lazy target symbol plus its mapping to the semantic operation contract.

    ``parameter_map`` and ``output_map`` use semantic names as keys and the
    target API's concrete names as values. ``target_only_parameters`` records
    compatibility controls such as a Torch stream that intentionally have no
    JAX equivalent.
    """

    module: str
    symbol: str
    parameter_map: Mapping[str, str]
    output_map: Mapping[str, str]
    target_only_parameters: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.module or not self.symbol:
            raise ValueError("TargetBinding module and symbol must not be empty")

        parameter_map = dict(self.parameter_map)
        output_map = dict(self.output_map)
        target_only_parameters = tuple(self.target_only_parameters)
        for mapping_name, mapping in (
            ("parameter_map", parameter_map),
            ("output_map", output_map),
        ):
            if not mapping or not all(isinstance(key, str) and key and isinstance(value, str) and value for key, value in mapping.items()):
                raise ValueError(f"TargetBinding {mapping_name} must contain non-empty names")
            if len(set(mapping.values())) != len(mapping):
                raise ValueError(f"TargetBinding {mapping_name} target names must be unique")
        if len(set(target_only_parameters)) != len(target_only_parameters) or any(not isinstance(name, str) or not name for name in target_only_parameters):
            raise ValueError("TargetBinding target_only_parameters must be unique non-empty names")
        if set(parameter_map.values()) & set(target_only_parameters):
            raise ValueError("Mapped and target-only parameter names must not overlap")

        object.__setattr__(self, "parameter_map", MappingProxyType(parameter_map))
        object.__setattr__(self, "output_map", MappingProxyType(output_map))
        object.__setattr__(self, "target_only_parameters", target_only_parameters)

    @property
    def qualified_name(self) -> str:
        return f"{self.module}:{self.symbol}"

    def resolve(self) -> Callable[..., Any]:
        module = importlib.import_module(self.module)
        try:
            implementation = getattr(module, self.symbol)
        except AttributeError as exc:
            raise ImportError(f"Target binding {self.qualified_name} does not exist") from exc
        if not callable(implementation):
            raise TypeError(f"Target binding {self.qualified_name} is not callable")
        return implementation


@dataclass(frozen=True)
class TargetGap:
    """Explicit, reviewable declaration of a missing optional target."""

    reason: str
    tracking_issue: str

    def __post_init__(self) -> None:
        if not self.reason or not self.tracking_issue:
            raise ValueError("A target gap requires a reason and tracking issue")


TargetStatus = Union[TargetBinding, TargetGap]


@dataclass(frozen=True)
class FrontendOperationSpec:
    """One Torch-canonical semantic operation and its optional targets."""

    name: str
    contract_signature: inspect.Signature
    targets: Mapping[FrontendTarget, TargetStatus]
    api_anchors: Tuple[str, ...]
    kernel_anchors: Tuple[str, ...]
    output_names: Tuple[str, ...]
    parity_case: Optional[str]
    required_targets: ClassVar[Tuple[FrontendTarget, ...]] = tuple(FrontendTarget)
    canonical_target: ClassVar[FrontendTarget] = FrontendTarget.TORCH

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Frontend operation name must not be empty")
        if not self.api_anchors:
            raise ValueError(f"{self.name} must declare at least one API anchor")
        if not self.kernel_anchors:
            raise ValueError(f"{self.name} must own at least one CuTe kernel")
        for anchor_kind, anchors in (
            ("API", self.api_anchors),
            ("kernel", self.kernel_anchors),
        ):
            if len(set(anchors)) != len(anchors):
                raise ValueError(f"{self.name} has duplicate {anchor_kind} ownership anchors")
            if any(":" not in anchor for anchor in anchors):
                raise ValueError(f"{self.name} {anchor_kind} anchors must use " "'module:qualified_name' syntax")
        if not self.output_names or len(set(self.output_names)) != len(self.output_names):
            raise ValueError(f"{self.name} output names must be non-empty and unique")
        if self.parity_case is not None and (not isinstance(self.parity_case, str) or not self.parity_case):
            raise ValueError(f"{self.name} parity_case must be a non-empty string")

        unsupported_kinds = {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }
        if any(parameter.kind in unsupported_kinds for parameter in self.contract_signature.parameters.values()):
            raise ValueError(f"{self.name} semantic contract must use explicit " "positional-or-keyword and keyword-only parameters")

        normalized_targets = {FrontendTarget.normalize(target): status for target, status in self.targets.items()}
        if len(normalized_targets) != len(self.targets):
            raise ValueError(f"{self.name} declares the same target more than once")
        missing = set(self.required_targets) - set(normalized_targets)
        extra = set(normalized_targets) - set(self.required_targets)
        if missing or extra:
            raise ValueError(
                f"{self.name} target declaration mismatch; missing=" f"{sorted(x.value for x in missing)}, extra=" f"{sorted(x.value for x in extra)}"
            )
        if not isinstance(normalized_targets[self.canonical_target], TargetBinding):
            raise ValueError(f"{self.name} must have a concrete canonical Torch binding")
        if not all(isinstance(status, (TargetBinding, TargetGap)) for status in normalized_targets.values()):
            raise TypeError(f"{self.name} target entries must be TargetBinding or TargetGap")

        for target, status in normalized_targets.items():
            if isinstance(status, TargetBinding) and status.symbol != self.name:
                raise ValueError(f"{self.name} {target.value} binding symbol must match the " f"semantic operation name; got {status.symbol!r}")

        contract_names = set(self.contract_signature.parameters)
        output_names = set(self.output_names)
        for target, status in normalized_targets.items():
            if not isinstance(status, TargetBinding):
                continue
            if set(status.parameter_map) != contract_names:
                raise ValueError(
                    f"{self.name} {target.value} parameter map must cover the "
                    f"semantic contract exactly; expected={sorted(contract_names)}, "
                    f"got={sorted(status.parameter_map)}"
                )
            if set(status.output_map) != output_names:
                raise ValueError(
                    f"{self.name} {target.value} output map must cover semantic "
                    f"outputs exactly; expected={sorted(output_names)}, "
                    f"got={sorted(status.output_map)}"
                )

        if isinstance(normalized_targets[FrontendTarget.JAX], TargetBinding):
            if not self.parity_case:
                raise ValueError(f"{self.name} has a JAX binding but no registered parity case")
        elif self.parity_case is not None:
            raise ValueError(f"{self.name} cannot declare a parity case without a JAX binding")

        object.__setattr__(self, "targets", MappingProxyType(normalized_targets))
        object.__setattr__(self, "api_anchors", tuple(self.api_anchors))
        object.__setattr__(self, "kernel_anchors", tuple(self.kernel_anchors))
        object.__setattr__(self, "output_names", tuple(self.output_names))

    def status(self, target: Union[FrontendTarget, str]) -> TargetStatus:
        return self.targets[FrontendTarget.normalize(target)]

    def resolve(self, target: Union[FrontendTarget, str]) -> Callable[..., Any]:
        normalized_target = FrontendTarget.normalize(target)
        status = self.targets[normalized_target]
        if isinstance(status, TargetGap):
            raise NotImplementedError(f"{self.name} has no {normalized_target.value} implementation: " f"{status.reason} ({status.tracking_issue})")

        implementation = status.resolve()
        self._validate_binding_signature(normalized_target, status, implementation)
        return implementation

    def _validate_binding_signature(
        self,
        target: FrontendTarget,
        binding: TargetBinding,
        implementation: Callable[..., Any],
    ) -> None:
        actual_signature = inspect.signature(implementation)
        actual_parameters = actual_signature.parameters
        expected_names = set(binding.parameter_map.values()) | set(binding.target_only_parameters)
        if set(actual_parameters) != expected_names:
            raise TypeError(f"{self.name} {target.value} binding parameters drifted; " f"expected {sorted(expected_names)}, got {sorted(actual_parameters)}")

        for semantic_name, target_name in binding.parameter_map.items():
            semantic_parameter = self.contract_signature.parameters[semantic_name]
            target_parameter = actual_parameters[target_name]
            if semantic_parameter.kind != target_parameter.kind:
                raise TypeError(
                    f"{self.name} {target.value} parameter kind for semantic "
                    f"parameter {semantic_name!r} drifted; expected "
                    f"{semantic_parameter.kind.description}, got "
                    f"{target_parameter.kind.description}"
                )
            if semantic_parameter.default != target_parameter.default:
                raise TypeError(
                    f"{self.name} {target.value} default for semantic parameter "
                    f"{semantic_name!r} drifted; expected "
                    f"{semantic_parameter.default!r}, got {target_parameter.default!r}"
                )


class FrontendOperationRegistry:
    """Registry used by API discovery and target-support checks."""

    def __init__(self) -> None:
        self._operations: dict[str, FrontendOperationSpec] = {}

    def register(self, operation: FrontendOperationSpec) -> None:
        if operation.name in self._operations:
            raise ValueError(f"Duplicate frontend operation {operation.name!r}")
        for existing in self._operations.values():
            shared_api_anchors = set(operation.api_anchors) & set(existing.api_anchors)
            shared_kernel_anchors = set(operation.kernel_anchors) & set(existing.kernel_anchors)
            if shared_api_anchors or shared_kernel_anchors:
                raise ValueError(
                    f"{operation.name} and {existing.name} have overlapping "
                    f"ownership: api={sorted(shared_api_anchors)}, "
                    f"kernels={sorted(shared_kernel_anchors)}"
                )
        self._operations[operation.name] = operation

    def get(self, name: str) -> FrontendOperationSpec:
        try:
            return self._operations[name]
        except KeyError as exc:
            raise KeyError(f"Unknown frontend operation {name!r}") from exc

    def operations(self) -> Tuple[FrontendOperationSpec, ...]:
        return tuple(self._operations[name] for name in sorted(self._operations))

    def audit(
        self,
        *,
        require_jax_complete: bool = False,
        resolve_bindings: bool = False,
    ) -> Tuple[str, ...]:
        """Return unresolved bindings and, optionally, declared JAX gaps."""

        issues = []
        for operation in self.operations():
            for target in operation.required_targets:
                status = operation.status(target)
                if isinstance(status, TargetGap):
                    if require_jax_complete:
                        issues.append(f"{operation.name}:{target.value}: {status.reason} " f"({status.tracking_issue})")
                elif resolve_bindings:
                    try:
                        operation.resolve(target)
                    except Exception as exc:
                        issues.append(f"{operation.name}:{target.value}: {exc}")
        return tuple(issues)


FRONTEND_OPERATION_REGISTRY = FrontendOperationRegistry()


def frontend_operation(
    *,
    name: str,
    targets: Mapping[Union[FrontendTarget, str], TargetStatus],
    api_anchors: Sequence[str],
    kernel_anchors: Sequence[str],
    output_names: Sequence[str],
    parity_case: Optional[str],
    registry: Optional[FrontendOperationRegistry] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a semantic contract without replacing its Python function."""

    selected_registry = FRONTEND_OPERATION_REGISTRY if registry is None else registry

    def decorate(contract: Callable[..., Any]) -> Callable[..., Any]:
        operation = FrontendOperationSpec(
            name=name,
            contract_signature=inspect.signature(contract),
            targets=targets,
            api_anchors=tuple(api_anchors),
            kernel_anchors=tuple(kernel_anchors),
            output_names=tuple(output_names),
            parity_case=parity_case,
        )
        selected_registry.register(operation)
        return contract

    return decorate


__all__ = [
    "FRONTEND_OPERATION_REGISTRY",
    "FrontendOperationRegistry",
    "FrontendOperationSpec",
    "FrontendTarget",
    "TargetBinding",
    "TargetGap",
    "frontend_operation",
]
