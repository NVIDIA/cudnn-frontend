# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""FROST norm-backward engine: a single engine (``norm_bprop_sm100``) that serves
every norm variant.

Backward counterpart of ``cudnn.norm.fprop.engines``. One engine advertises the
set of variants it can serve and dispatches by ``facts.variant`` in ``lower``;
the shared analyzer (``cudnn.norm.graph_analyzer``) supplies the facts with
``phase == "bprop"``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional

import torch

from cudnn.frost import register_engine
from cudnn.frost.dispatch import requested_knobs
from cudnn.norm import graph_analyzer as ga
from cudnn.norm.config_sm100 import NormVariant

_LOG = logging.getLogger(__name__)

_ALL_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32})
_ALL_VARIANTS = frozenset(NormVariant)


@dataclass(frozen=True)
class NormBwdKnobs:
    """Per-graph tuning request for the norm-backward engine (none exposed yet;
    present for API symmetry / wrong-vocabulary rejection)."""


@dataclass(frozen=True)
class Capabilities:
    variants: frozenset = _ALL_VARIANTS
    arch: tuple = (10, 0)
    phase: str = "bprop"
    dtypes: frozenset = _ALL_DTYPES


def mismatch(capabilities: Capabilities, facts: "ga.NormGraphFacts",
             requested: Optional[NormBwdKnobs] = None) -> Optional[str]:
    """First reason this engine cannot serve these facts, or None if it can."""
    if facts.invalid:
        return facts.invalid
    if requested is not None and not isinstance(requested, NormBwdKnobs):
        return f"knob request is a {type(requested).__name__}, not NormBwdKnobs — wrong operation's vocabulary"
    if facts.phase != capabilities.phase:
        return f"engine serves {capabilities.phase}; graph is {facts.phase}"
    if facts.variant not in capabilities.variants:
        return f"engine does not serve norm variant {facts.variant.value}"
    if facts.device_cc != capabilities.arch:
        return f"requires SM{capabilities.arch[0]}{capabilities.arch[1]}; current device is {facts.device_cc}"
    if facts.dtype not in capabilities.dtypes:
        return f"dtype {facts.dtype} not in {sorted(str(d) for d in capabilities.dtypes)}"
    if not facts.uniform_dtype:
        return "DY dtype must match X"
    return None


@dataclass(frozen=True)
class EngineSpec:
    name: str
    capabilities: Capabilities
    lower: "Callable[[EngineSpec, ga.NormGraphFacts, Optional[NormBwdKnobs]], Any]" = None


ENGINE_NAME = "norm_bprop_sm100"
ENGINE_SPECS = (EngineSpec(name=ENGINE_NAME, capabilities=Capabilities()),)


def probe(spec: EngineSpec, graph) -> bool:
    facts = ga.analyze(graph)
    if facts is None:
        return False
    reason = mismatch(spec.capabilities, facts, requested_knobs(graph))
    if reason is not None:
        _LOG.debug("cudnn.norm: %s ineligible: %s", spec.name, reason)
        return False
    return True


def build(spec: EngineSpec, graph):
    facts = ga.analyze(graph)
    if facts is None:
        raise ValueError("cudnn.norm: graph is not a single norm-backward node")
    requested = requested_knobs(graph)
    reason = mismatch(spec.capabilities, facts, requested)
    if reason is not None:
        raise ValueError(f"cudnn.norm: {spec.name}: {reason}")
    lower = spec.lower or lower_norm_bprop
    return lower(spec, facts, requested)


def lower_norm_bprop(spec: EngineSpec, facts: "ga.NormGraphFacts",
                     requested: Optional[NormBwdKnobs] = None):
    """Default lowering: resolve buffers at execute time and call
    :func:`cudnn.norm.bprop.api.norm_bprop` (dispatches by ``facts.variant``);
    outputs are copied into the graph's DX / Dscale / Dbias buffers."""
    from cudnn.norm.bprop.api import norm_bprop

    binding = ga.NormBinding({
        "dy": facts.dy_t, "x": facts.x_t, "scale": facts.scale_t,
        "mean": facts.mean_t, "inv_var": facts.inv_var_t,
        "dx": facts.dx_t, "dscale": facts.dscale_t, "dbias": facts.dbias_t,
    })

    def _execute(variant_pack):
        resolved = ga.resolve_variant_pack(variant_pack, binding)
        dy = resolved[id(facts.dy_t)]
        x = resolved[id(facts.x_t)]
        gamma = resolved.get(id(facts.scale_t)) if facts.scale_t is not None else None
        mean = resolved.get(id(facts.mean_t)) if facts.mean_t is not None else None
        rstd = resolved.get(id(facts.inv_var_t)) if facts.inv_var_t is not None else None
        dx, dgamma, dbeta = norm_bprop(
            facts.variant, dy, x, gamma, mean, rstd,
            normalized_shape=facts.normalized_shape, num_groups=facts.num_groups,
            has_beta=facts.has_beta,
        )
        _copy_into(resolved.get(id(facts.dx_t)), dx)
        if facts.dscale_t is not None:
            _copy_into(resolved.get(id(facts.dscale_t)), dgamma)
        if facts.dbias_t is not None and dbeta is not None:
            _copy_into(resolved.get(id(facts.dbias_t)), dbeta)
        return None

    return _execute


def _copy_into(buf, src):
    if buf is not None:
        buf.copy_(src.reshape(buf.shape))


def engine_name() -> str:
    return ENGINE_NAME


for _s in ENGINE_SPECS:
    register_engine(_s.name, partial(probe, _s), partial(build, _s))

__all__ = ["Capabilities", "EngineSpec", "ENGINE_SPECS", "ENGINE_NAME", "NormBwdKnobs", "engine_name", "mismatch"]
