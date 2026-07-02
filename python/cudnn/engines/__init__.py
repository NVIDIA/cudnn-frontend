"""Execution backends for NativeGraph.

Pluggable execution backends in one flat engine-id space with the cuDNN backend.
The Router builds a ranked plan list at ``create_execution_plans()`` time; graph
construction stays backend-agnostic.

Backends:
- ReferenceMatmulEngine: pure-PyTorch correctness oracle (CPU/GPU, no JIT deps)
- MatmulCuTileEngine: NVIDIA cuTile matmul (Blackwell SM100+); optional deps
"""

from .base import BaseEngine, CompiledPlan, ExecutionContext, PlanConfig
from .engine_ids import PYTHON_ENGINE_ID_BASE, CUDNN_HEURISTIC_ENGINE_ID, is_python_engine
from .router import Router, default_router
from .reference_matmul_engine import ReferenceMatmulEngine

__all__ = [
    "BaseEngine",
    "Router",
    "PlanConfig",
    "CompiledPlan",
    "ExecutionContext",
    "default_router",
    "ReferenceMatmulEngine",
    "PYTHON_ENGINE_ID_BASE",
    "CUDNN_HEURISTIC_ENGINE_ID",
    "is_python_engine",
]

# cuTile backend has optional native deps (cuda-tile / cuda-python); expose it
# only when importable so a plain install still gets the reference backend.
try:
    from .matmul_cutile_engine import MatmulCuTileEngine  # noqa: F401

    __all__.append("MatmulCuTileEngine")
except Exception:  # noqa: BLE001
    MatmulCuTileEngine = None  # type: ignore
