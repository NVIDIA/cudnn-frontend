# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution engines for pygraph.

Pluggable engines in one flat engine-id space with the cuDNN backend. At
``create_execution_plans()`` the Router ranks the engines that claim the graph
against the backend's own recommendation into ONE list (``graph.plans``);
``build_plans()`` walks it. Graph construction stays engine-agnostic.

Which engines exist is the library's own business: ``manifest.py`` is the static
table this version ships, matched against a graph by a cheap node-type key
before any engine module is imported. ``graph.register_backend()`` remains as
the out-of-tree escape hatch.
"""

from .base import BaseEngine, CompiledPlan, ExecutionContext, PlanConfig
from .engine_ids import (
    BACKEND_ENGINE_ID_BASE,
    BACKEND_HEURISTIC_ENGINE_ID,
    CPP_OSS_ENGINE_ID_BASE,
    OUT_OF_TREE_ID_BASE,
    PYTHON_ENGINE_ID_BASE,
    is_backend_engine,
    is_python_engine,
)
from .heuristics import heuristics_sort
from .manifest import MANIFEST, EngineFamily
from .router import Router, decline_types, default_router


def __getattr__(name: str):
    # Engine classes stay reachable by name (tests, advanced callers) but are
    # imported on demand: an engine module drags in torch / the CuTe DSL and
    # `import cudnn` must not pay that. The manifest is the source of truth for
    # what exists; this only resolves a class name to its row.
    for row in MANIFEST:
        if row.factory == name:
            import importlib

            cls = getattr(importlib.import_module(row.module), name)
            globals()[name] = cls
            return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseEngine",
    "CompiledPlan",
    "ExecutionContext",
    "PlanConfig",
    "Router",
    "default_router",
    "decline_types",
    "heuristics_sort",
    "MANIFEST",
    "EngineFamily",
    "BACKEND_ENGINE_ID_BASE",
    "CPP_OSS_ENGINE_ID_BASE",
    "PYTHON_ENGINE_ID_BASE",
    "OUT_OF_TREE_ID_BASE",
    "BACKEND_HEURISTIC_ENGINE_ID",
    "is_python_engine",
    "is_backend_engine",
]
