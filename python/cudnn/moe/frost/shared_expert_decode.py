# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host entry of the shared-expert DECODE kernel pair (``kernel/shared_expert_decode_f16.py``): validation with typed declines, the
plan-time compile and the two launches.  No DSL import at module level -- the CuTe DSL is imported inside :meth:`SharedExpertDecode.compile`
after the version / architecture gate (python/cudnn/AGENTS.md Rule 7), so a host without it can still construct the entry and run its
declines.  The binding checks (:func:`check_binding`: dtype, device, shape, stride and alignment of every operand) run before EVERY compile
and EVERY launch, compile-cache hits included -- the compile key pins the plan-time facts only (the geometry, the token-tile FORM, the x / y
row strides, the config), and a compiled DSL entry accepts a tensor of another dtype, footprint or device silently, so the host is a
launch's only guard.

The op: ``y = sigmoid(x W_g^T) * ( (silu(x W_gate^T) * (x W_up^T)) W_d^T )`` for ``x`` [T, H] bf16 -> ``y`` [T, H] bf16, with ``w_gate_up``
[2 I, H] bf16 row-major (gate rows first), ``w_down`` [H, I] bf16 row-major, ``w_gate`` [H] (or [1, H]) bf16; fp32 accumulation in a fixed
order per output element (the kernel module's docstring), ``act`` rounded ONCE to bf16 between the launches, ``y`` once at the end.  The add
into the routed output stays with the caller.  Served (declined otherwise, by name): ``T`` in 1 .. 24 (a RUNTIME kernel argument; the form
-- one or two 16-token m-tiles -- is the compile key's), the validated geometries H 8192 with I 256 or 128, bf16 everywhere, contiguous
weights, x / y rows contiguous with a 16-byte-multiple row stride, every operand on the entry's device, a compute capability 10.x part
(12.x holds no form in its 99 KB opt-in and is declined until one is measured), a CuTe DSL >= 4.8.0.

Launch knobs ``pdl`` / ``pdl_pair`` / ``pdl_release`` set the config's ``PDL`` / ``PDL_PAIR`` / ``PDL_RELEASE``: ``pdl`` = launch 1's
programmatic-stream-serialization attribute (CALLER OBLIGATION, the config docstring: the launch's immediate stream predecessor must not
write ``w_gate_up`` / ``w_gate``, which load before the kernel's ``griddepcontrol.wait``), ``pdl_pair`` = launch 2's (its predecessor is
launch 1; ``w_down`` loads before the wait), ``pdl_release`` = ``griddepcontrol.launch_dependents`` at both kernels' entry.  Every default
is off (a knob measured before it defaults on).  Workspace: the inter-launch scratch ``act`` [32, I] bf16 and ``gate_part`` [32, GATE_PARTS]
fp32 is entry-owned, allocated once per config at the first compile (outside any capture), never on the execute path (Rule 1); the launch
reads no device memory to the host (Rule 3).  Two entries, or one entry on two streams concurrently, must not share a scratch: build one
entry per concurrent stream.  Eager-path host cost: the form of a token count is memoized per m-tile count (``cfg_for``: one validated
``Cfg`` per form, never a re-validation per call), the compile key is computed once per call, and the DSL views of the entry-owned scratch
are built once per form (static pointers); the per-layer operands (``x``, the three weights, ``out``) are wrapped per call -- they change
with every layer -- and the binding checks run on every call (Rule 1 keeps the validation, not the recomputation).
"""

from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

from cudnn.frost.buffers import cutedsl_requirement_error, cutedsl_state

from .kernel.shared_expert_decode_config import (
    DEFAULT_GEOMETRY,
    MAX_TOKENS,
    SCRATCH_ROWS,
    SERVED_GEOMETRIES,
    SM100_SMEM_OPTIN_BYTES,
    SM100_SMEM_PER_SM_BYTES,
    Cfg,
    Geometry,
    default_cfg,
    form_for,
    part_readings,
    token_tiles_for,
    validate_cfg,
)

#: the served compute capabilities as closed ranges of ``10 * major + minor``: SM100-class parts only (validated on cc 10.3; cc 10.0 / 10.7
#: unverified).  cc 12.x (99 KB per-CTA opt-in) holds no launch-1 form and stays declined until one is measured.
SUPPORTED_CC_RANGES = ((100, 109),)
SUPPORTED_CC_TEXT = "10.x (SM100-class)"
DSL_FLOOR = (4, 8, 0)
DSL_FLOOR_TEXT = ".".join(str(x) for x in DSL_FLOOR)
NAME = "the shared-expert decode pair"


def cc_supported(cc) -> bool:
    """Whether a ``(major, minor)`` compute capability is one this kernel serves; ``None`` (no CUDA device) is not."""
    if cc is None:
        return False
    code = 10 * int(cc[0]) + int(cc[1])
    return any(lo <= code <= hi for lo, hi in SUPPORTED_CC_RANGES)


def _dsl_version_error() -> Optional[str]:
    """The Rule-7 gate: the pair's own floor (4.8.0) through the one version parser of :mod:`cudnn.frost.buffers`; None when it serves."""
    installed, _ = cutedsl_state()
    if not installed:
        return f"{NAME} requires nvidia-cutlass-dsl >= {DSL_FLOOR_TEXT}; the CuTe DSL is not installed"
    return cutedsl_requirement_error(NAME, floor=DSL_FLOOR)


def device_shared_memory(device) -> Tuple[int, int]:
    """(shared memory per SM, per-CTA opt-in maximum) in bytes of a CUDA device -- the part's own reading, never a constant carried across
    an architecture line (the readers of :mod:`cudnn.frost.device`)."""
    from cudnn.frost.device import resolve_device as _resolve_ordinal
    from cudnn.frost.device import shared_memory_per_block_optin

    import torch

    ordinal = _resolve_ordinal(torch.device(device).index)
    per_sm = int(torch.cuda.get_device_properties(ordinal).shared_memory_per_multiprocessor)
    optin = int(shared_memory_per_block_optin(ordinal))
    if per_sm <= 0 or optin <= 0:
        raise RuntimeError(f"{NAME}: the driver reported no shared-memory readings for device {ordinal} (per SM {per_sm}, opt-in {optin})")
    return per_sm, optin


def device_multiprocessor_count(device) -> int:
    import torch

    return int(torch.cuda.get_device_properties(device).multi_processor_count)


def resolve_device(device):
    """``device`` as a concrete ``torch.device``: a CUDA device given without an index is the CURRENT one, so it compares with an operand's
    device (always indexed for a CUDA tensor) by type AND index; any other device is kept as given."""
    import torch

    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def resolve_geometry(geometry, cfg: Optional[Cfg] = None) -> Geometry:
    """The geometry a call is checked against: ``geometry`` when given (a ``Geometry``, else ``TypeError``), else the ``cfg``'s own, else the
    default; a ``cfg`` compiled for another geometry than the one given is a ``ValueError`` (one source)."""
    if geometry is not None and not isinstance(geometry, Geometry):
        raise TypeError(f"{NAME}: geometry must be a shared_expert_decode_config.Geometry, got {type(geometry).__name__}")
    if cfg is not None and geometry is not None and cfg.GEOM != geometry:
        raise ValueError(f"{NAME}: the cfg is compiled for {cfg.GEOM.describe()} but the geometry names {geometry.describe()}; give one source")
    if geometry is not None:
        return geometry
    return cfg.GEOM if cfg is not None else DEFAULT_GEOMETRY


def _check_dtype(name, t, want) -> None:
    if t.dtype != want:
        raise TypeError(f"{NAME}: {name} must be {want} (kernel-native; callers convert), got {t.dtype}")


def _check_aligned(name, t) -> None:
    if t.data_ptr() % 16:
        raise ValueError(f"{NAME}: {name} must start at a 16-byte aligned address for its 128-bit copies, got data_ptr % 16 = {t.data_ptr() % 16}")


def _check_device(tensors, device, anchor) -> None:
    want = resolve_device(device) if device is not None else tensors[anchor].device
    whose = "the entry's" if device is not None else f"{anchor}'s"
    for name, t in tensors.items():
        if t.device != want:
            raise ValueError(f"{NAME}: every operand must live on one device, {whose} ({want}); {name} is on {t.device}")


def check_binding(x, w_gate_up, w_down, w_gate, out=None, device=None, geometry: Optional[Geometry] = None) -> int:
    """The binding checks, run before every compile AND every launch (compile-cache hits included): ``TypeError`` for a dtype (or a
    non-tensor), ``ValueError`` for an inconsistent shape, stride, alignment or device, ``NotImplementedError`` for a token count or a
    geometry this kernel does not serve.  Returns ``T = x.shape[0]``.  ``device`` (an entry's: a concrete CUDA ordinal) requires every operand
    on exactly that device; without it the operands must share ``x``'s.  Host-only metadata arithmetic, no device work."""
    import torch

    g = resolve_geometry(geometry)
    H, I = g.HIDDEN, g.INTERMEDIATE
    tensors = dict(x=x, w_gate_up=w_gate_up, w_down=w_down, w_gate=w_gate)
    if out is not None:
        tensors["out"] = out
    for name, t in tensors.items():
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{NAME}: {name} must be a torch.Tensor, got {type(t).__name__}")
    _check_device(tensors, device, "x")
    for name, t in tensors.items():
        _check_dtype(name, t, torch.bfloat16)
    if g not in SERVED_GEOMETRIES:
        raise NotImplementedError(f"{NAME} serves {' or '.join(s.describe() for s in SERVED_GEOMETRIES)} (validated); got {g.describe()}")
    if x.dim() != 2 or x.shape[1] != H:
        raise ValueError(f"{NAME}: x must be [tokens, {H}] at {g.describe()}; got {tuple(x.shape)}")
    T = int(x.shape[0])
    if T < 1:
        raise ValueError(f"{NAME}: x must hold at least one token; got {tuple(x.shape)}")
    if T > MAX_TOKENS:
        raise NotImplementedError(f"{NAME} serves at most {MAX_TOKENS} tokens per call (decode-time counts; the 25 .. 32 band is unmeasured); got {T}")
    if x.stride(1) != 1 or x.stride(0) % 8:
        raise ValueError(f"{NAME}: x rows must be contiguous with a 16-byte-multiple row stride; got strides {tuple(x.stride())}")
    if tuple(w_gate_up.shape) != (2 * I, H) or not w_gate_up.is_contiguous():
        raise ValueError(
            f"{NAME}: w_gate_up must be a contiguous [{2 * I}, {H}] (gate rows then up rows); got {tuple(w_gate_up.shape)} strides {tuple(w_gate_up.stride())}"
        )
    if tuple(w_down.shape) != (H, I) or not w_down.is_contiguous():
        raise ValueError(f"{NAME}: w_down must be a contiguous [{H}, {I}]; got {tuple(w_down.shape)} strides {tuple(w_down.stride())}")
    if w_gate.numel() != H or not w_gate.is_contiguous() or w_gate.dim() not in (1, 2) or (w_gate.dim() == 2 and w_gate.shape[0] != 1):
        raise ValueError(f"{NAME}: w_gate must be a contiguous [{H}] or [1, {H}]; got {tuple(w_gate.shape)} strides {tuple(w_gate.stride())}")
    if out is not None:
        if out.dim() != 2 or out.shape[0] < T or out.shape[1] != H:
            raise ValueError(f"{NAME}: out must be [>= {T}, {H}]; got {tuple(out.shape)}")
        if out.stride(1) != 1 or out.stride(0) % 8:
            raise ValueError(f"{NAME}: out rows must be contiguous with a 16-byte-multiple row stride; got strides {tuple(out.stride())}")
    for name, t in tensors.items():
        _check_aligned(name, t)
    return T


def check_support(x, w_gate_up, w_down, w_gate, out=None, device_cc=None, device=None, geometry: Optional[Geometry] = None) -> int:
    """Typed declines, before any DSL import: the binding checks of :func:`check_binding` plus the architecture gate
    (``NotImplementedError`` for a part this kernel does not serve).  ``device_cc`` overrides the compute-capability probe (tests)."""
    import torch

    T = check_binding(x, w_gate_up, w_down, w_gate, out, device=device, geometry=geometry)
    dev = x.device
    if device_cc is None and dev.type != "cuda":
        raise NotImplementedError(f"{NAME} serves CUDA tensors only; the operands are on {dev}")
    cc = tuple(torch.cuda.get_device_capability(dev)) if device_cc is None else tuple(device_cc)
    if not cc_supported(cc):
        raise NotImplementedError(f"{NAME} serves compute capability {SUPPORTED_CC_TEXT}; this device is {cc[0]}.{cc[1]}")
    return T


def _flat(t):
    """A 1-D DSL view covering a (possibly strided) tensor's whole footprint (16-byte aligned; the kernel addresses by element)."""
    from cutlass.cute.runtime import from_dlpack

    n = 1 + sum((d - 1) * s for d, s in zip(t.shape, t.stride()))
    return from_dlpack(t.detach().as_strided((n,), (1,)), assumed_align=16)


class SharedExpertDecode:
    """The pair compiled per (config, token-tile form, x / y row strides) and shared by every layer: all per-layer tensors are launch
    arguments.  ``geometry`` is the rank's ``Geometry(HIDDEN, INTERMEDIATE)`` (the default geometry, hidden 8192 / intermediate 256, when
    none is given; a ``cfg``'s own otherwise); ``cfg`` overrides the config as a whole (its ``TOKEN_TILES`` is replaced by the call's form);
    without it the entry takes the geometry's default form on this part's shared-memory readings.  ``pdl`` / ``pdl_pair`` / ``pdl_release``
    are the launch knobs (None = the form's own fields; the defaults: off).  The device is resolved to a concrete CUDA ordinal at
    construction, and every operand of every compile / launch must live on exactly it.  An entry built for a non-CUDA device (host tests)
    keeps the cc 10.x readings."""

    def __init__(
        self,
        device,
        geometry: Optional[Geometry] = None,
        cfg: Optional[Cfg] = None,
        pdl: Optional[bool] = None,
        pdl_pair: Optional[bool] = None,
        pdl_release: Optional[bool] = None,
    ):
        self.device = resolve_device(device)
        self.cfg_given = cfg is not None
        self.geometry: Geometry = resolve_geometry(geometry, cfg)
        self.pdl = None if pdl is None else bool(pdl)
        self.pdl_pair = None if pdl_pair is None else bool(pdl_pair)
        self.pdl_release = None if pdl_release is None else bool(pdl_release)
        self.readings: Tuple[int, int] = (SM100_SMEM_PER_SM_BYTES, SM100_SMEM_OPTIN_BYTES)
        self.sm_count: Optional[int] = None
        self.part: Dict[str, int] = {}
        if self.device.type == "cuda":
            self.readings = device_shared_memory(self.device)
            self.part = part_readings(*self.readings)
            self.sm_count = device_multiprocessor_count(self.device)
        base = cfg if cfg is not None else default_cfg(self.geometry, *self.readings)
        self.cfg: Cfg = self._on_this_part(base)
        self.compiled: Dict[tuple, Any] = {}
        self.scratch: Dict[Cfg, Dict[str, Any]] = {}
        self.forms: Dict[int, Cfg] = {}  # token-tile count -> the validated form of self.cfg
        self.out_buffer = None

    def _on_this_part(self, cfg: Cfg) -> Cfg:
        """``cfg`` with this part's readings, the entry's geometry and the launch knobs, validated."""
        cfg = replace(cfg, GEOM=self.geometry, **self.part, **self.launch_fields(cfg))
        validate_cfg(cfg)
        return cfg

    def launch_fields(self, cfg: Cfg) -> Dict[str, bool]:
        """The ``PDL`` / ``PDL_PAIR`` / ``PDL_RELEASE`` fields the entry's knobs set on ``cfg``; a knob left None keeps the form's own field.
        The fields are part of the config, so of the compile key: a programmatic pair and a plain pair are two compiled entries (the
        cubins are the same: the attributes are launch facts)."""
        fields = {}
        if self.pdl is not None:
            fields["PDL"] = self.pdl
        if self.pdl_pair is not None:
            fields["PDL_PAIR"] = self.pdl_pair
        if self.pdl_release is not None:
            fields["PDL_RELEASE"] = self.pdl_release
        return fields

    def cfg_for(self, tokens: int) -> Cfg:
        """The config a plan for ``tokens`` compiles with: the entry's config in the token count's form (one or two m-tiles), validated
        once per form and memoized (``token_tiles_for`` still raises the typed decline of a token count past the ceiling on every call)."""
        tiles = token_tiles_for(tokens)
        form = self.forms.get(tiles)
        if form is None:
            form = self.forms[tiles] = form_for(self.cfg, tokens)
        return form

    def key(self, tokens: int, x, out) -> tuple:
        """The compile key: plan-time facts only (the form of the token count, the strides, the config -- its geometry included)."""
        return (self.cfg_for(tokens), int(x.stride(0)), int(out.stride(0)))

    def scratch_for(self, cfg: Cfg) -> Dict[str, Any]:
        """The inter-launch scratch of a config (``act`` [32, I] bf16, ``gate_part`` [32, GATE_PARTS] fp32), allocated once per config at
        its first compile (outside any capture); rows past the call's ``T`` are never written.  ``views`` holds their DSL views (built at
        the first ``_args``; the tensors never move, so the views are static)."""
        if cfg not in self.scratch:
            import torch

            self.scratch[cfg] = dict(
                act=torch.zeros(SCRATCH_ROWS, cfg.I, dtype=torch.bfloat16, device=self.device),
                gate_part=torch.zeros(SCRATCH_ROWS, cfg.GATE_PARTS, dtype=torch.float32, device=self.device),
            )
        return self.scratch[cfg]

    def _default_out(self):
        """The entry-owned output buffer [32, H] (sliced to the call's T) when the caller hands none."""
        if self.out_buffer is None:
            import torch

            self.out_buffer = torch.empty(SCRATCH_ROWS, self.geometry.HIDDEN, dtype=torch.bfloat16, device=self.device)
        return self.out_buffer

    def _args(self, cfg, x, w_gate_up, w_down, w_gate, out):
        s = self.scratch_for(cfg)
        views = s.get("views")
        if views is None:
            views = s["views"] = (_flat(s["act"]), _flat(s["gate_part"]))
        return [_flat(x), _flat(w_gate_up), _flat(w_down), _flat(w_gate), views[0], views[1], _flat(out)]

    def compile(self, x, w_gate_up, w_down, w_gate, out=None) -> None:
        """Validate (typed declines, on every call), gate the DSL, trace-compile the pair's host entry for this (form, strides) once."""
        T = check_support(x, w_gate_up, w_down, w_gate, out, device=self.device, geometry=self.geometry)
        out = out if out is not None else self._default_out()
        key = self.key(T, x, out)
        if key in self.compiled:
            return
        import torch

        msg = _dsl_version_error()
        if msg is not None:
            raise NotImplementedError(msg)
        import cutlass.cute as cute
        from cutlass.cute.runtime import make_fake_stream

        from .kernel import shared_expert_decode_f16 as mod

        cfg = key[0]
        args = self._args(cfg, x, w_gate_up, w_down, w_gate, out)
        with torch.cuda.device(self.device):
            self.compiled[key] = cute.compile(mod.launch, cfg, *args, T, int(x.stride(0)), int(out.stride(0)), make_fake_stream())

    def __call__(self, x, w_gate_up, w_down, w_gate, out=None):
        """Launch the pair on torch's current stream; returns ``out`` [T, H] bf16 (the caller's, or the entry's own buffer sliced to T).
        Compiles on first use for a new (form, strides).  Every launch re-runs the binding checks before the compiled pair is looked up."""
        import cuda.bindings.driver as cuda
        import torch

        T = check_binding(x, w_gate_up, w_down, w_gate, out, device=self.device, geometry=self.geometry)
        given = out is not None
        out = out if given else self._default_out()
        key = self.key(T, x, out)
        if key not in self.compiled:
            self.compile(x, w_gate_up, w_down, w_gate, out if given else None)
        args = self._args(key[0], x, w_gate_up, w_down, w_gate, out)
        self.compiled[key](*args, T, cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream))
        return out if given else out[:T]
