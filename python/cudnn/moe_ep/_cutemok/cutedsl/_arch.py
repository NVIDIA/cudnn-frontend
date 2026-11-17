# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""The Blackwell part the CuTe DSL kernels compile for: sm_100a (B200 / GB200) or sm_103a (B300 / GB300), and
the options of every ``cute.compile`` of the package.

The target is resolved once, from the current CUDA device, when the first kernel module imports this one (on the
first call; ``import cudnn.moe_ep._cutemok`` does not).  With no visible or no supported device the default is sm_103a and the
per-call device check rejects a launch on the wrong part.  ``COMPILE_OPTIONS`` passes the target and TVM FFI to
each compile (``--gpu-arch``, ``--enable-tvm-ffi``), so CuteMoK sets no environment variable and does not care
whether CUTLASS was imported before it: ``CUTE_DSL_ARCH`` and ``CUTE_DSL_ENABLE_TVM_FFI`` need not be set.

A ``CUTE_DSL_ARCH`` other than the device's target is declined here, before any compile.  CuTe DSL reads it when
CUTLASS is first imported and runs an arch-specific ("a") binary only on the arch it names, so the kernels compiled
for the device could not be launched in that process.
"""

import os

import torch

ARCH_ENV = "CUTE_DSL_ARCH"
#: Compute capability -> the arch the tcgen05 kernels compile to on that part.
SUPPORTED_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
CAPABILITY_BY_ARCH = {arch: capability for capability, arch in SUPPORTED_CAPABILITIES.items()}
#: The target with no supported device (an import that never launches).
DEFAULT_ARCH = "sm_103a"


def resolve(requested: str | None, capability: tuple[int, int] | None) -> str:
    """The target for the device's compute capability (None without a device).  ``requested`` is the
    ``CUTE_DSL_ARCH`` value, or None: on a supported device it must be the device's target, otherwise this raises;
    without a supported device it is not used."""
    target = SUPPORTED_CAPABILITIES.get(capability)
    if target is None:
        return DEFAULT_ARCH
    if requested is not None and requested != target:
        raise RuntimeError(
            f"{ARCH_ENV}={requested} does not match the device (compute capability {capability}, {target}).  CuteMoK "
            f"compiles for the device and passes the target to each compile, but CuTe DSL runs an {target} binary only "
            f"when {ARCH_ENV} is unset or {target}; unset it or set it to {target}."
        )
    return target


def _device_capability() -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability(torch.cuda.current_device())


TARGET_ARCH = resolve(os.environ.get(ARCH_ENV), _device_capability())
#: The capability ``TARGET_ARCH`` is compiled for; every launch checks its device against it.
TARGET_CAPABILITY = CAPABILITY_BY_ARCH[TARGET_ARCH]
#: The options of every ``cute.compile`` of the package: the target and TVM FFI per call, not through the
#: environment.  Executor cache keys include this string, so executors for different targets are never shared.
COMPILE_OPTIONS = f"--opt-level 2 --gpu-arch {TARGET_ARCH} --enable-tvm-ffi"


def check_device(device, what: str) -> None:
    """Raise unless ``device`` is the part the kernels were compiled for."""
    capability = torch.cuda.get_device_capability(device)
    if capability != TARGET_CAPABILITY:
        raise NotImplementedError(
            f"{what} was compiled for {TARGET_ARCH} (compute capability {TARGET_CAPABILITY}, the device current when "
            f"CuteMoK's kernels were first imported); the device is compute capability {capability}.  "
            f"Supported: {SUPPORTED_CAPABILITIES}; one process runs the kernels on one part."
        )
