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
    # Device ccs served, matched exactly: B200 / GB200 and B300 / GB300 run the same kernels.
    archs: frozenset = frozenset({(10, 0), (10, 3)})
    phase: str = "fprop"
    dtypes: frozenset = _ALL_DTYPES
    running_stats: bool = True  # BatchNorm running mean/var update


def mismatch(capabilities: Capabilities, facts: "ga.NormGraphFacts", requested: Optional[NormFwdKnobs] = None) -> Optional[str]:
    """First reason this engine cannot serve these facts, or None if it can."""
    if facts.invalid:
        return facts.invalid
    if facts.extended_affine:
        from cudnn.norm.affine import mismatch as affine_mismatch

        reason = affine_mismatch(facts)
        if reason is not None:
            return reason
    if requested is not None and not isinstance(requested, NormFwdKnobs):
        return f"knob request is a {type(requested).__name__}, not NormFwdKnobs — wrong operation's vocabulary"
    if facts.phase != capabilities.phase:
        return f"engine serves {capabilities.phase}; graph is {facts.phase}"
    if facts.variant not in capabilities.variants:
        return f"engine does not serve norm variant {facts.variant.value}"
    if facts.device_cc not in capabilities.archs:
        return f"requires {' or '.join(f'SM{a}{b}' for a, b in sorted(capabilities.archs))}; current device is {facts.device_cc}"
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
# The SM90 row runs the same kernels: they need cp.async.bulk, clusters and mapa
# (all Hopper), and BatchNorm NHWC skips its TMEM cache tier where there is no TMEM.
ENGINE_SPECS = (
    EngineSpec(name=ENGINE_NAME, capabilities=Capabilities()),
    EngineSpec(name="norm_fprop_sm90", capabilities=Capabilities(archs=frozenset({(9, 0)}))),
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
    if facts.extended_affine:
        from cudnn.norm.affine import lower

        return lower(facts)

    from cudnn.norm.fprop.api import fprop_workspace_bytes, norm_fprop

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
            "prev_run_mean": facts.prev_run_mean_t,
            "prev_run_var": facts.prev_run_var_t,
            "epsilon": facts.eps_t,
            "momentum": facts.momentum_t,
        }
    )

    def _execute(variant_pack, ws=None):
        resolved = ga.resolve_variant_pack(variant_pack, binding)
        x = resolved[id(facts.x_t)]
        gamma = resolved.get(id(facts.scale_t)) if facts.scale_t is not None else None
        beta = resolved.get(id(facts.bias_t)) if facts.bias_t is not None else None
        rm = _running_pair(resolved, facts.run_mean_t, facts.prev_run_mean_t)
        rv = _running_pair(resolved, facts.run_var_t, facts.prev_run_var_t)
        eps = _host_scalar(resolved.get(id(facts.eps_t)) if facts.eps_t is not None else None, facts.eps)
        momentum = _host_scalar(resolved.get(id(facts.momentum_t)) if facts.momentum_t is not None else None, facts.momentum)
        # Hand the graph's own buffers down so the kernels write them in place:
        # that is what keeps execute() free of allocations (Rule 1 / Rule 8), and
        # it drops a full-tensor copy of Y that cost the graph path ~1.6x the
        # direct one. norm_fprop ignores any buffer it cannot write directly, so
        # _copy_into below still covers those.
        y_buf = resolved.get(id(facts.y_t))
        mean_buf = resolved.get(id(facts.mean_t)) if facts.mean_t is not None else None
        rstd_buf = resolved.get(id(facts.inv_var_t)) if facts.inv_var_t is not None else None
        y, mean, rstd = norm_fprop(
            facts.variant,
            x,
            gamma,
            beta,
            eps=eps,
            normalized_shape=facts.normalized_shape,
            num_groups=facts.num_groups,
            momentum=momentum,
            training=facts.training,
            running_mean=rm,
            running_var=rv,
            out=(y_buf, mean_buf, rstd_buf),
            ws=ws,
        )
        _copy_into(y_buf, y)
        _copy_into(mean_buf, mean)
        _copy_into(rstd_buf, rstd)
        return None

    # Sized once, at build: get_workspace_size() is asked before any execute.
    _execute.workspace_bytes = fprop_workspace_bytes(
        facts.variant,
        facts.x_shape,
        _io_of(facts),
        normalized_shape=facts.normalized_shape,
        num_groups=facts.num_groups,
        nhwc=_is_channels_last(facts),
    )
    return _execute


def _io_of(facts):
    """The kernels' dtype tag for this graph's X."""
    from cudnn.norm.dtypes import torch_dtype_to_str

    return torch_dtype_to_str(facts.dtype)


def _is_channels_last(facts):
    """True when X is declared channels-last, which is what picks the NHWC kernels.

    Read off the graph's declared strides rather than a tensor, because sizing
    happens at build time when no buffer is bound yet.
    """
    shape = tuple(int(v) for v in facts.x_shape)
    if len(shape) != 4 or facts.x_t is None:
        return False
    try:
        n, c, h, w = (int(v) for v in facts.x_t.get_stride())
    except Exception:  # noqa: BLE001 - a graph that will not describe itself is not NHWC
        return False
    return c == 1 and w == shape[1] and h == shape[1] * shape[3]


def _running_pair(resolved, next_t, prev_t):
    """The buffer the running statistic should be updated THROUGH.

    cuDNN exposes the previous state and the next state as separate ports, and
    they are separate buffers. Treating the OUTPUT as the input -- which is what
    binding only ``next_running_mean`` amounts to -- makes the update read
    whatever the caller left in the output: initialising it to 0 vs 100 moved the
    next mean by exactly ``(1 - momentum) * 100``, with the previous-state input
    ignored. Seeding ``next`` from ``prev`` keeps the in-place kernel update
    correct and leaves the caller's previous-state tensor untouched.
    """
    nxt = resolved.get(id(next_t)) if next_t is not None else None
    prev = resolved.get(id(prev_t)) if prev_t is not None else None
    if nxt is None:
        return prev  # graph declared a single in-place port
    if prev is not None and nxt.data_ptr() != prev.data_ptr():
        nxt.copy_(prev.reshape(nxt.shape))
    return nxt


def _host_scalar(buf, fallback):
    """A graph scalar (epsilon, momentum) read from its bound buffer.

    cuDNN takes these as host scalars, so this is a host read of host memory and
    not the device read Rule 3 bans. A device-resident one would need that read,
    so it is refused at check time instead (see the engine's decline) rather than
    silently normalised with the wrong value.
    """
    if buf is None:
        return fallback
    if hasattr(buf, "device") and buf.device.type != "cpu":
        return fallback
    try:
        return float(buf.reshape(-1)[0].item())
    except Exception:
        return fallback


def _copy_into(buf, src):
    """Land ``src`` in ``buf``, unless the kernel already wrote ``buf`` itself.

    When the output buffer was passed down through ``out=``, what comes back is
    a view of it and the copy would be ``buf.copy_(buf)`` -- same pointer, pure
    waste. Comparing the storage is what tells the two cases apart, since
    norm_fprop silently falls back to a temporary for any buffer it cannot use.
    """
    if buf is None or src is None or buf.data_ptr() == src.data_ptr():
        return
    buf.copy_(src.reshape(buf.shape))


def engine_name() -> str:
    """The registered engine name (test/user convenience)."""
    return ENGINE_NAME


__all__ = ["Capabilities", "EngineSpec", "ENGINE_SPECS", "ENGINE_NAME", "analyze_for", "build", "probe", "NormFwdKnobs", "engine_name", "mismatch"]
