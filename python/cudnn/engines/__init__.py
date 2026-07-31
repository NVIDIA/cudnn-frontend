# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution backends for pygraph.

Pluggable execution backends in one flat engine-id space with the cuDNN backend.
The Router builds a ranked plan list at ``create_execution_plans()`` time; graph
construction stays backend-agnostic.

Backends:
- ReferenceMatmulEngine: pure-PyTorch correctness oracle (CPU/GPU, no JIT deps).
  Lives in ``cudnn.gemm.reference`` with the rest of the GEMM family and is
  re-exported here so ``cudnn.engines.ReferenceMatmulEngine`` keeps working.

Real DSL engines (cuTile / CuTe-DSL GEMM fusion) plug in as separate PRs — an
engine is one file implementing BaseEngine; nothing here changes.
"""

from .base import BaseEngine, CompiledPlan, ExecutionContext, PlanConfig
from .engine_ids import PYTHON_ENGINE_ID_BASE, BACKEND_HEURISTIC_ENGINE_ID, is_python_engine
from .router import Router, default_router
from cudnn.gemm.reference.reference_matmul_engine import ReferenceMatmulEngine

__all__ = [
    "BaseEngine",
    "Router",
    "PlanConfig",
    "CompiledPlan",
    "ExecutionContext",
    "default_router",
    "ReferenceMatmulEngine",
    "PYTHON_ENGINE_ID_BASE",
    "BACKEND_HEURISTIC_ENGINE_ID",
    "is_python_engine",
]
