"""Execution backends for NativeGraph.

Pluggable execution backends for Python-native graphs. The Router selects one
at ``create_execution_plans()`` time; graph construction stays backend-agnostic.

Backends:
- ReferenceMatmulEngine: pure-PyTorch correctness oracle (CPU/GPU, no JIT deps)
- MatmulCuTileEngine: NVIDIA cuTile matmul (Blackwell SM100+); optional deps

See ``docs/python_native_graph_router.md`` for the architecture.
"""

from .base import BaseEngine
from .router import Router, default_router
from .reference_matmul_engine import ReferenceMatmulEngine

__all__ = ["BaseEngine", "Router", "default_router", "ReferenceMatmulEngine"]

# cuTile backend has optional native deps (cuda-tile / cuda-python); expose it
# only when importable so a plain install still gets the reference backend.
try:
    from .matmul_cutile_engine import MatmulCuTileEngine  # noqa: F401

    __all__.append("MatmulCuTileEngine")
except Exception:  # noqa: BLE001
    MatmulCuTileEngine = None  # type: ignore
