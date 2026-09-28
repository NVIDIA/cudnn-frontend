# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""FROST norm-forward engine: a single engine (``norm_fprop_sm100``) that serves
every norm variant.

The norm variants are not distinct kernel *geometries* (the way SDPA's d256 vs
d512 are) — they are the same reduce-and-affine op that ``norm_fprop`` dispatches
to the right per-flavor kernel at lowering time. So one engine advertises the
*set* of variants it can serve via :class:`Capabilities` and its ``lower``
dispatches by ``facts.variant`` (mirroring how ``cudnn.gemm.frost`` uses one
engine for the whole GEMM family, rather than SDPA's engine-per-geometry). The
backward counterpart is ``cudnn.norm.bprop.engines`` (``norm_bprop_sm100``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from cudnn.norm import graph_analyzer as ga
from cudnn.norm.config_sm100 import NormVariant

_LOG = logging.getLogger(__name__)

_ALL_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32})
# Every variant norm_fprop can lower. (cuDNN has no native GroupNorm node, so a
# native graph never yields GROUP_NORM facts; it stays reachable via the direct
# API. Advertising it here is harmless and keeps the set = what the lowering serves.)
_ALL_VARIANTS = frozenset(NormVariant)


@dataclass(frozen=True)
class NormFwdKnobs:
    """Per-graph tuning request for the norm-forward engine.

    No tuning knobs are exposed yet (block size and cp.async/bulk staging are
    resolved automatically from the shape by ``config_sm100.make_cfg``). The type
    exists for API symmetry and so a wrong-vocabulary request is rejected in the
    probe.
    """


@dataclass(frozen=True)
class Capabilities:
    """What the norm-forward ENGINE can serve. Compared field-by-field against
    :class:`NormGraphFacts` in :func:`mismatch`."""

    variants: frozenset = _ALL_VARIANTS
    arch: tuple = (10, 0)
    phase: str = "fprop"
    dtypes: frozenset = _ALL_DTYPES
    running_stats: bool = True  # BatchNorm running mean/var update


def mismatch(capabilities: Capabilities, facts: "ga.NormGraphFacts", requested: Optional[NormFwdKnobs] = None) -> Optional[str]:
    """First reason this engine cannot serve these facts, or None if it can."""
    if facts.invalid:
        return facts.invalid
    if requested is not None and not isinstance(requested, NormFwdKnobs):
        return f"knob request is a {type(requested).__name__}, not NormFwdKnobs — wrong operation's vocabulary"
    if facts.phase != capabilities.phase:
        return f"engine serves {capabilities.phase}; graph is {facts.phase}"
    if facts.variant not in capabilities.variants:
        return f"engine does not serve norm variant {facts.variant.value}"
    if facts.device_cc != capabilities.arch:
        return f"requires SM{capabilities.arch[0]}{capabilities.arch[1]}; current device is {facts.device_cc}"
    if facts.dtype not in capabilities.dtypes:
        return f"dtype {facts.dtype} not in {sorted(str(d) for d in capabilities.dtypes)}"
    if facts.wants_running_stats and not capabilities.running_stats:
        return "graph updates running stats, which this engine does not support"
    return None


@dataclass(frozen=True)
class EngineSpec:
    name: str
    capabilities: Capabilities
    lower: "Callable[[EngineSpec, ga.NormGraphFacts, Optional[NormFwdKnobs]], Any]" = None


ENGINE_NAME = "norm_fprop_sm100"
ENGINE_SPECS = (EngineSpec(name=ENGINE_NAME, capabilities=Capabilities()),)


def analyze_for(graph):
    """This family's facts for ``graph``, or None if it is not a single norm node."""
    return ga.analyze(graph)


def probe(spec: EngineSpec, graph, requested=None) -> bool:
    facts = ga.analyze(graph)
    if facts is None:
        return False
    reason = mismatch(spec.capabilities, facts, requested)
    if reason is not None:
        _LOG.debug("cudnn.norm: %s ineligible: %s", spec.name, reason)
        return False
    return True


def build(spec: EngineSpec, graph, requested=None):
    facts = ga.analyze(graph)
    if facts is None:
        raise ValueError("cudnn.norm: graph is not a single norm-forward node")
    reason = mismatch(spec.capabilities, facts, requested)
    if reason is not None:
        raise ValueError(f"cudnn.norm: {spec.name}: {reason}")
    lower = spec.lower or lower_norm_fprop
    return lower(spec, facts, requested)


def lower_norm_fprop(spec: EngineSpec, facts: "ga.NormGraphFacts", requested: Optional[NormFwdKnobs] = None):
    """Default lowering: resolve buffers at execute time and call
    :func:`cudnn.norm.fprop.api.norm_fprop`, which dispatches to the per-flavor
    sm_100 kernel (JIT-compiled + cached on first call) by ``facts.variant``.
    Outputs are copied into the graph's Y / mean / inv_variance buffers."""
    from cudnn.norm.fprop.api import norm_fprop

    binding = ga.NormBinding(
        {
            "x": facts.x_t,
            "scale": facts.scale_t,
            "bias": facts.bias_t,
            "y": facts.y_t,
            "mean": facts.mean_t,
            "inv_var": facts.inv_var_t,
            "run_mean": facts.run_mean_t,
            "run_var": facts.run_var_t,
        }
    )

    def _execute(variant_pack):
        resolved = ga.resolve_variant_pack(variant_pack, binding)
        x = resolved[id(facts.x_t)]
        gamma = resolved.get(id(facts.scale_t)) if facts.scale_t is not None else None
        beta = resolved.get(id(facts.bias_t)) if facts.bias_t is not None else None
        rm = resolved.get(id(facts.run_mean_t)) if facts.run_mean_t is not None else None
        rv = resolved.get(id(facts.run_var_t)) if facts.run_var_t is not None else None
        y, mean, rstd = norm_fprop(
            facts.variant,
            x,
            gamma,
            beta,
            eps=facts.eps,
            normalized_shape=facts.normalized_shape,
            num_groups=facts.num_groups,
            momentum=facts.momentum,
            training=facts.training,
            running_mean=rm,
            running_var=rv,
        )
        _copy_into(resolved.get(id(facts.y_t)), y)
        if facts.mean_t is not None:
            _copy_into(resolved.get(id(facts.mean_t)), mean)
        if facts.inv_var_t is not None:
            _copy_into(resolved.get(id(facts.inv_var_t)), rstd)
        return None

    return _execute


def _copy_into(buf, src):
    if buf is not None:
        buf.copy_(src.reshape(buf.shape))


def engine_name() -> str:
    """The registered engine name (test/user convenience)."""
    return ENGINE_NAME


__all__ = ["Capabilities", "EngineSpec", "ENGINE_SPECS", "ENGINE_NAME", "analyze_for", "build", "probe", "NormFwdKnobs", "engine_name", "mismatch"]
