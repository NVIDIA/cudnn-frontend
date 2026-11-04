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
from typing import Any, Callable, Optional

import torch

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
    # Device ccs served, matched exactly: B200 / GB200 and B300 / GB300 run the same kernels.
    archs: frozenset = frozenset({(10, 0), (10, 3)})
    phase: str = "bprop"
    dtypes: frozenset = _ALL_DTYPES


def mismatch(capabilities: Capabilities, facts: "ga.NormGraphFacts", requested: Optional[NormBwdKnobs] = None) -> Optional[str]:
    """First reason this engine cannot serve these facts, or None if it can."""
    if facts.invalid:
        return facts.invalid
    if facts.extended_affine:
        from cudnn.norm.affine import mismatch as affine_mismatch

        reason = affine_mismatch(facts)
        if reason is not None:
            return reason
    if requested is not None and not isinstance(requested, NormBwdKnobs):
        return f"knob request is a {type(requested).__name__}, not NormBwdKnobs — wrong operation's vocabulary"
    if facts.phase != capabilities.phase:
        return f"engine serves {capabilities.phase}; graph is {facts.phase}"
    if facts.variant not in capabilities.variants:
        return f"engine does not serve norm variant {facts.variant.value}"
    if facts.device_cc not in capabilities.archs:
        return f"requires {' or '.join(f'SM{a}{b}' for a, b in sorted(capabilities.archs))}; current device is {facts.device_cc}"
    if facts.dtype not in capabilities.dtypes:
        return f"dtype {facts.dtype} not in {sorted(str(d) for d in capabilities.dtypes)}"
    if not facts.uniform_dtype and not facts.extended_affine:
        return "DY dtype must match X"
    return None


@dataclass(frozen=True)
class EngineSpec:
    name: str
    capabilities: Capabilities
    lower: "Callable[[EngineSpec, ga.NormGraphFacts, Optional[NormBwdKnobs]], Any]" = None


ENGINE_NAME = "norm_bprop_sm100"
# The SM90 row runs the same kernels (cp.async.bulk, clusters and mapa are all Hopper).
ENGINE_SPECS = (
    EngineSpec(name=ENGINE_NAME, capabilities=Capabilities()),
    EngineSpec(name="norm_bprop_sm90", capabilities=Capabilities(archs=frozenset({(9, 0)}))),
)


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
        raise ValueError("cudnn.norm: graph is not a single norm-backward node")
    reason = mismatch(spec.capabilities, facts, requested)
    if reason is not None:
        raise ValueError(f"cudnn.norm: {spec.name}: {reason}")
    lower = spec.lower or lower_norm_bprop
    return lower(spec, facts, requested)


def lower_norm_bprop(spec: EngineSpec, facts: "ga.NormGraphFacts", requested: Optional[NormBwdKnobs] = None):
    """Default lowering: resolve buffers at execute time and call
    :func:`cudnn.norm.bprop.api.norm_bprop` (dispatches by ``facts.variant``);
    outputs are copied into the graph's DX / Dscale / Dbias buffers."""
    if facts.extended_affine:
        from cudnn.norm.affine import lower

        return lower(facts)

    from cudnn.norm.bprop.api import bprop_workspace_bytes, norm_bprop

    binding = ga.NormBinding(
        {
            "dy": facts.dy_t,
            "x": facts.x_t,
            "scale": facts.scale_t,
            "mean": facts.mean_t,
            "inv_var": facts.inv_var_t,
            "dx": facts.dx_t,
            "dscale": facts.dscale_t,
            "dbias": facts.dbias_t,
        }
    )

    def _execute(variant_pack, ws=None):
        resolved = ga.resolve_variant_pack(variant_pack, binding)
        dy = resolved[id(facts.dy_t)]
        x = resolved[id(facts.x_t)]
        gamma = resolved.get(id(facts.scale_t)) if facts.scale_t is not None else None
        mean = resolved.get(id(facts.mean_t)) if facts.mean_t is not None else None
        rstd = resolved.get(id(facts.inv_var_t)) if facts.inv_var_t is not None else None
        # Hand the graph's own buffers down so the kernels write them in place --
        # no allocation in execute (Rule 1 / Rule 8), and no full-tensor copy of
        # DX. norm_bprop ignores any buffer it cannot write directly, so
        # _copy_into still covers those.
        dx_buf = resolved.get(id(facts.dx_t))
        dg_buf = resolved.get(id(facts.dscale_t)) if facts.dscale_t is not None else None
        db_buf = resolved.get(id(facts.dbias_t)) if facts.dbias_t is not None else None
        dx, dgamma, dbeta = norm_bprop(
            facts.variant,
            dy,
            x,
            gamma,
            mean,
            rstd,
            normalized_shape=facts.normalized_shape,
            num_groups=facts.num_groups,
            has_beta=facts.has_beta,
            out=(dx_buf, dg_buf, db_buf),
            ws=ws,
        )
        _copy_into(dx_buf, dx)
        _copy_into(dg_buf, dgamma)
        if dbeta is not None:
            _copy_into(db_buf, dbeta)
        return None

    # Sized once, at build: get_workspace_size() is asked before any execute.
    _execute.workspace_bytes = bprop_workspace_bytes(
        facts.variant,
        facts.x_shape,
        _io_of(facts),
        normalized_shape=facts.normalized_shape,
        num_groups=facts.num_groups,
        has_beta=facts.has_beta,
        nhwc=_is_channels_last(facts),
    )
    return _execute


def _io_of(facts):
    """The kernels' dtype tag for this graph's X."""
    from cudnn.norm.dtypes import torch_dtype_to_str

    return torch_dtype_to_str(facts.dtype)


def _is_channels_last(facts):
    """True when X is declared channels-last; read off the graph's strides,
    because sizing happens at build time with no buffer bound."""
    shape = tuple(int(v) for v in facts.x_shape)
    if len(shape) != 4 or facts.x_t is None:
        return False
    try:
        n, c, h, w = (int(v) for v in facts.x_t.get_stride())
    except Exception:  # noqa: BLE001
        return False
    return c == 1 and w == shape[1] and h == shape[1] * shape[3]


def _copy_into(buf, src):
    """Land ``src`` in ``buf``, unless the kernel already wrote ``buf`` itself.

    When the buffer went down through ``out=``, what comes back is a view of it
    and the copy would be ``buf.copy_(buf)``. Comparing storage distinguishes
    that from the fallback, where norm_bprop used a temporary instead.
    """
    if buf is None or src is None or buf.data_ptr() == src.data_ptr():
        return
    buf.copy_(src.reshape(buf.shape))


def engine_name() -> str:
    return ENGINE_NAME


__all__ = ["Capabilities", "EngineSpec", "ENGINE_SPECS", "ENGINE_NAME", "analyze_for", "build", "probe", "NormBwdKnobs", "engine_name", "mismatch"]
