# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared gates and helpers for the FROST GEMM test suite."""

from __future__ import annotations

import pytest
import torch
import cudnn

from cudnn.conv.frost._cutedsl import BLOCK_SCALE_CUTEDSL_MIN_VERSION, DENSE_CUTEDSL_MIN_VERSION, requirement_error as cutedsl_requirement_error


def _active_sm() -> int | None:
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


_SM = _active_sm()

# Every e2e test in this suite JITs sm100-family templates, valid only on
# 100 <= SM < 120 (see kernel_registry.PIPELINE_ARCH_RANGES) — gate on arch, not just
# GPU presence, so wrong-arch machines skip instead of failing in the JIT.
requires_sm100 = pytest.mark.skipif(
    _SM is None or not (100 <= _SM < 120),
    reason="needs a Blackwell-family GPU (100 <= SM < 120), have " + ("none" if _SM is None else f"sm_{_SM}"),
)


def _requires_cutedsl(what: str, minimum_version: tuple[int, int]):
    error = cutedsl_requirement_error(what, minimum_version)
    return pytest.mark.skipif(error is not None, reason=error or "CuTeDSL version is supported")


requires_block_scale_cutedsl = _requires_cutedsl("Frost block-scale convolution tests", BLOCK_SCALE_CUTEDSL_MIN_VERSION)
requires_dense_cutedsl = _requires_cutedsl("Frost dense convolution tests", DENSE_CUTEDSL_MIN_VERSION)


def make_input(shape: tuple[int, ...], dtype: torch.dtype, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    # Quarter-integer values are exactly representable by every tested dtype
    # and keep the unscaled FP8 output inside E4M3's finite range.
    values = torch.randint(-2, 3, shape, dtype=torch.int32, device="cuda", generator=generator).to(torch.float32) * 0.25
    # Allocate the destination with the requested memory format explicitly.
    # Calling contiguous(memory_format=...) can retain collapsed strides when
    # one or more spatial extents are one, while Frost requires the canonical
    # compact channels-last stride even for degenerate dimensions.
    output = torch.empty(shape, dtype=dtype, device="cuda", memory_format=torch.channels_last_3d)
    return output.copy_(values)


def select_and_build_frost_plan(graph: cudnn.pygraph, plan_name: str) -> None:
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    plan_names = [graph.get_plan_name_at_index(i) for i in range(len(graph.plans))]
    if plan_name not in plan_names:
        pytest.fail(f"{plan_name} did not accept the graph; available plans: {plan_names}")
    graph.select_plan(plan_names.index(plan_name))
    graph.check_support()
    graph.build_plans()
