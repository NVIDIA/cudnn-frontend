"""Optional cuDNN-graph engine registration for the FROST norm backend.

This wires the norm kernels into the same FROST engine framework that
``cudnn.gemm.frost`` uses: a graph is classified, probed for eligibility, and
(if selected via ``graph.select_engines(["frost_norm_eng0"])``) built into a
callable plan. It depends on a *built* cuDNN frontend (``cudnn._compiled_module``)
and on ``cudnn.frost``; importing it is therefore kept separate from the kernel
library in :mod:`cudnn.norm.frost` so the kernels stay usable standalone.

Status: scaffold. The kernel side (fprop/bprop for all five variants) is
complete and tested via :func:`cudnn.norm.frost.norm_forward` /
:func:`cudnn.norm.frost.norm_backward`; what remains here is the graph analysis
that reads ``graph.nodes`` for ``NORM_FWD`` / ``NORM_BWD`` operations, maps the
cuDNN ``NormMode_t`` to a :class:`~cudnn.norm.frost.config.NormVariant`, resolves
the variant pack (x, scale, bias, mean, inv_variance, dy, ...), and dispatches to
the kernels. Mirror ``cudnn/gemm/frost/graph_analyzer.py`` (probe/build) and
``cudnn/gemm/frost/__init__.py`` (registration) when completing it.
"""

from __future__ import annotations

from .config import NormVariant

# Selected via graph.select_engines(["frost_norm_eng0"]).
ENGINE_NAME = "frost_norm_eng0"

# cuDNN NormMode_t (by name) -> internal NormVariant. Resolved lazily so this
# module imports without the compiled cuDNN extension present.
_NORM_MODE_NAMES = {
    "LAYER_NORM": NormVariant.LAYER_NORM,
    "RMS_NORM": NormVariant.RMS_NORM,
    "GROUP_NORM": NormVariant.GROUP_NORM,
    "BATCH_NORM": NormVariant.BATCH_NORM,
    "INSTANCE_NORM": NormVariant.INSTANCE_NORM,
}


def norm_mode_to_variant(mode) -> NormVariant:
    """Map a ``cudnn.norm_mode`` enum value to a :class:`NormVariant`."""
    name = getattr(mode, "name", str(mode)).upper()
    try:
        return _NORM_MODE_NAMES[name]
    except KeyError:
        raise NotImplementedError(f"norm mode {name!r} not supported by frost_norm") from None


def probe_norm_plan(graph) -> bool:  # pragma: no cover - scaffold
    """Cheap eligibility check; never raises. TODO: inspect ``graph.nodes``."""
    return False


def build_norm_plan(graph):  # pragma: no cover - scaffold
    """Analyze ``graph`` and build a callable norm plan. TODO."""
    raise NotImplementedError(
        "frost_norm graph engine not yet wired; use cudnn.norm.frost.norm_forward / "
        "norm_backward directly. See module docstring."
    )


def register() -> None:  # pragma: no cover - scaffold
    """Register the norm engine with the shared FROST framework.

    Call once (e.g. from an opset import) after the cuDNN frontend is available::

        from cudnn.frost import register_engine
        register_engine(ENGINE_NAME, probe_norm_plan, build_norm_plan)
    """
    from cudnn.frost import register_engine

    register_engine(ENGINE_NAME, probe_norm_plan, build_norm_plan)


__all__ = ["ENGINE_NAME", "norm_mode_to_variant", "probe_norm_plan", "build_norm_plan", "register"]
