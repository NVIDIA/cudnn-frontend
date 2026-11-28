# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute capabilities served by the vendored MegaMoE MXFP8 kernels."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MegaMoeArchitecture:
    """One device architecture and its vendored forward/backward kernels."""

    name: str
    compute_capability: tuple[int, int]
    cute_dsl_arch: str
    supports_inference: bool
    # The forward kernel takes ``schedule_mode`` (``phase_interleave``) and the
    # mixed-CGA fallback/preferred cluster split.
    supports_forward_schedule_options: bool
    # The forward kernel stages FC2 bulk stores only for a quantized (MXFP8) combine.
    forward_bulk_store_requires_quantized_combine: bool
    # A vendored MXFP8 WGrad kernel serves ``MoeEp.training_wgrad``.
    supports_training_wgrad: bool
    forward_kernel: tuple[str, str]
    backward_kernel: tuple[str, str]

    @property
    def cute_dsl_arch_spellings(self) -> tuple[str, str]:
        """Accepted ``CUTE_DSL_ARCH`` values, with and without the ``a`` suffix."""

        return (self.cute_dsl_arch.removesuffix("a"), self.cute_dsl_arch)

    def forward_kernel_class(self) -> Any:
        return _import_kernel_class(*self.forward_kernel)

    def backward_kernel_class(self) -> Any:
        return _import_kernel_class(*self.backward_kernel)

    def validate_forward_tuning(self, name: str, tuning: Any, combine_format: str) -> None:
        """Decline forward tuning this architecture's forward kernel cannot run.

        ``combine_format`` is the expert-combine format (a ``MoeFormat`` or its string value).
        """

        if not self.supports_forward_schedule_options:
            if tuning.scheduling == "phase_interleave":
                raise NotImplementedError(f"MoeEp MegaMoE {name}.scheduling='phase_interleave' is not supported on {self.name}")
            if tuning.mixed_cga:
                raise NotImplementedError(f"MoeEp MegaMoE {name}.mixed_cga=True is not supported on {self.name}")
        if self.forward_bulk_store_requires_quantized_combine and tuning.fc2_bulk_store != "off" and combine_format == "bf16":
            raise NotImplementedError(f"MoeEp MegaMoE {name}.fc2_bulk_store requires combine_format='mxfp8' on {self.name}, got 'bf16'")


def _import_kernel_class(module: str, name: str) -> Any:
    """Import a vendored kernel class only when a kernel is prepared."""

    return getattr(
        importlib.import_module(f"..cutedsl_src.kernel_src.{module}", __package__),
        name,
    )


BLACKWELL_SM100 = MegaMoeArchitecture(
    name="Blackwell SM100",
    compute_capability=(10, 0),
    cute_dsl_arch="sm_100a",
    supports_inference=False,
    supports_forward_schedule_options=False,
    forward_bulk_store_requires_quantized_combine=True,
    supports_training_wgrad=False,
    forward_kernel=(
        "blackwell.training.mega.fwd_glu.glu_mxfp8_mega_moe_kernel",
        "Sm100MegaMoEMxfp8GluKernel",
    ),
    backward_kernel=(
        "blackwell.training.mega.bwd_dglu.dglu_mxfp8_mega_moe_kernel",
        "Sm100MegaMoEMxfp8DgluKernel",
    ),
)
RUBIN_SM107 = MegaMoeArchitecture(
    name="Rubin SM107",
    compute_capability=(10, 7),
    cute_dsl_arch="sm_107a",
    supports_inference=True,
    supports_forward_schedule_options=True,
    forward_bulk_store_requires_quantized_combine=False,
    supports_training_wgrad=True,
    forward_kernel=(
        "rubin.training.mega.fwd_glu.glu_mxfp8_mega_moe_kernel",
        "Sm107MegaMoEMxfp8GluKernel",
    ),
    backward_kernel=(
        "rubin.training.mega.bwd_dglu.dglu_mxfp8_mega_moe_kernel",
        "Sm107MegaMoEMxfp8DgluKernel",
    ),
)
_ARCHITECTURES = {architecture.compute_capability: architecture for architecture in (BLACKWELL_SM100, RUBIN_SM107)}


def _describe(architecture: MegaMoeArchitecture) -> str:
    major, minor = architecture.compute_capability
    return f"{architecture.name} (compute capability {major}.{minor})"


def architecture_for(compute_capability: tuple[int, int]) -> MegaMoeArchitecture:
    """Return the kernel architecture for a device, or decline it."""

    major, minor = compute_capability
    architecture = _ARCHITECTURES.get((major, minor))
    if architecture is None:
        supported = " or ".join(_describe(candidate) for candidate in _ARCHITECTURES.values())
        raise NotImplementedError(f"MoeEp MegaMoE backend requires {supported}; " f"found compute capability {major}.{minor}")
    return architecture


def inference_architecture_for(compute_capability: tuple[int, int]) -> MegaMoeArchitecture:
    """Return the kernel architecture for an inference device, or decline it."""

    architecture = architecture_for(compute_capability)
    if not architecture.supports_inference:
        raise NotImplementedError(f"MoeEp MegaMoE inference is not supported on {architecture.name}; " "that architecture supports training only")
    return architecture


__all__ = [
    "BLACKWELL_SM100",
    "RUBIN_SM107",
    "MegaMoeArchitecture",
    "architecture_for",
    "inference_architecture_for",
]
