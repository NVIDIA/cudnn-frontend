# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host entry of the fused Gated DeltaNet DECODE step (``kernel/gdn_decode_fused_f16.py``): validation with typed declines, the
plan-time compile and the launch.  No DSL import at module level -- the CuTe DSL is imported inside :meth:`GdnDecodeFused.compile`
after the version / architecture gate (python/cudnn/AGENTS.md Rule 7), so a host without it can still construct the entry and run
its declines.  The binding checks (:func:`check_binding`: dtype, device, shape, stride and alignment of every operand against
``rows``) run before EVERY compile and EVERY launch, compile-cache hits included -- the compile key pins the plan-time facts only, and
a compiled DSL entry accepts a tensor of another dtype, footprint or device silently, so the host is a launch's only guard.  The entry's
device is resolved to a concrete CUDA ordinal at construction (:func:`resolve_device`) and every operand must live on exactly it (type
AND index: a CPU tensor or one on another CUDA device is a ``ValueError`` naming the operand and both devices).

Served geometry (declined otherwise, by name): the geometry constants of ``kernel/gdn_decode_fused_config.py`` -- the QKV slab
``x`` [rows, 10240] bf16 (a row-strided view is fine), the conv state ``cs`` [slots, 10240, 3] bf16 with channel-contiguous taps or
tap-contiguous channels, the conv weight ``cw`` [10240, 4] bf16, ``A_log`` [48] fp32, ``dt_bias`` [48] bf16, the fp32 state
``[slots, 48, 128, 128]`` (slot-strided), the int32 slot indices, ``z`` [rows, 48, 128] bf16 (row-strided), the norm weight [128]
bf16, the static FP8 scale (fp32, one element), ``hidden`` [rows, 5120] bf16 and the BA weight [96, 5120] bf16; every operand 16-byte
aligned; an SM100-class (compute capability 10.x) or SM120-class (12.x) part and a CuTe DSL >= 4.8.0.  The config is the caller's ``cfg`` as a whole or, without
one, the measured pick per rows on THIS part (``default_cfg`` on the device's SM count and shared-memory readings: a cluster of two CTAs
per head while the ``rows x 96`` CTAs have an SM each, one CTA per SM while the ``rows x 48`` CTAs have one, and past that the
register-resident two-CTAs-per-SM form where two of its CTAs share an SM on the part -- its third plan-time fact, ``co_resident_ctas``
-- else the one-CTA form in two waves); the device's SM count, its shared memory per SM and its per-CTA opt-in maximum (the occupancy
knob's pad and the pick's budget) are read at construction, and the entry starts from the one-CTA form the part holds (a part that holds
none is a typed ``NotImplementedError`` there; every shared-memory figure in this file's docstrings is a cc 10.0 reading, another part is
served on its own readings).  The launch knobs ``pdl`` /
``pdl_release`` set the config's ``PDL`` / ``PDL_RELEASE`` on top of whatever form serves (the pick or a given ``cfg``): under ``pdl=True``
every launch carries the programmatic-stream-serialization attribute, the kernel waits for its predecessor (``griddepcontrol.wait``)
only right before it reads the QKVZ projection's rows and the normed hidden (the slot, the parameters, the BA weight rows, the state
slab and the conv operands load ahead of the wait) and, with ``pdl_release`` (following ``pdl`` unless given), releases its successor at
entry (``griddepcontrol.launch_dependents``) so the out projection's prologue overlaps this kernel; the caller owes the config
docstring's obligation -- the launch's immediate predecessor must not write this layer's state, conv state or slot indices (the serving
chain's predecessor, the QKVZ projection, writes ``x`` / ``z`` only).  The picks keep PDL off.  The call contract mirrors the reference
kernel's (FlashInfer's built-in ``frost`` decode backend for the Qwen 3.5 GDN block) so an A/B is a one-line swap: ``entry(rows, x, cs,
cw, al, dt, state, idx, z, w, scale,
hidden, baw) -> qout`` with ``outputs(rows)`` holding ``qout`` (fp8 [rows, 48, 128]), ``out`` (bf16), ``counts`` (the reader tickets,
back to zero after every launch) and the debug buffers.
"""

from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error, cutedsl_state

from .kernel.gdn_decode_fused_config import (
    CFG,
    CONV_WIDTH,
    HEAD_DIM,
    HIDDEN,
    ONE_CTA_FORMS,
    QKV_CHANNELS,
    SM100_SMEM_OPTIN_BYTES,
    SM100_SMEM_PER_SM_BYTES,
    TICKET_STRIDE,
    V_HEADS,
    Cfg,
    default_cfg,
    first_fitting,
    validate_cfg,
)

#: the served compute capabilities as closed ranges of ``10 * major + minor``: SM100-class parts (cc 10.x; validated on cc 10.0, 10.3 and
#: 10.7) and SM120-class parts (cc 12.x; validated on cc 12.0 -- 100 KB of shared memory per SM, 99 KB per-CTA opt-in, 170 SMs -- the
#: same kernel body on the budget-derived picks; cc 12.1 shares the family admission on those picks).  cc 11.x has never run these
#: kernels and stays declined by name; the DSL's target gate (``cutedsl_arch_requirement_error``) decides per part on top.
SUPPORTED_CC_RANGES = ((100, 109), (120, 129))
SUPPORTED_CC_TEXT = "10.x (SM100-class) or 12.x (SM120-class)"
DSL_FLOOR = (4, 8, 0)
DSL_FLOOR_TEXT = ".".join(str(x) for x in DSL_FLOOR)
NAME = "the fused GDN decode step"
#: the launch is grid ``(48, rows, 1)`` (``kernel/gdn_decode_fused_f16.py``), one grid row per request: CUDA caps grid Y at 65535
MAX_ROWS = 65535


def cc_supported(cc) -> bool:
    """Whether a ``(major, minor)`` compute capability is one this kernel serves: inside one of ``SUPPORTED_CC_RANGES`` (SM100-class 10.x,
    SM120-class 12.x); ``None`` (no CUDA device) is not."""
    if cc is None:
        return False
    code = 10 * int(cc[0]) + int(cc[1])
    return any(lo <= code <= hi for lo, hi in SUPPORTED_CC_RANGES)


def _dsl_version_error() -> Optional[str]:
    """The Rule-7 gate of this kernel: its own floor (``DSL_FLOOR`` = 4.8.0, the public wheel every consumer pins, above the library's
    ``CUTEDSL_MIN_VERSION``) through the one version parser of :mod:`cudnn.frost.buffers`; None when the installed DSL serves."""
    installed, _ = cutedsl_state()
    if not installed:
        return f"{NAME} requires nvidia-cutlass-dsl >= {DSL_FLOOR_TEXT}; the CuTe DSL is not installed"
    return cutedsl_requirement_error(NAME, floor=DSL_FLOOR)


def device_shared_memory(device) -> Tuple[int, int]:
    """(shared memory per SM, per-CTA opt-in maximum) in bytes of a CUDA device, from the runtime's device properties
    (``cudaDevAttrMaxSharedMemoryPerMultiprocessor`` / ``cudaDevAttrMaxSharedMemoryPerBlockOptin``) -- the part's own reading, never a
    constant carried across an architecture line."""
    import torch

    props = torch.cuda.get_device_properties(device)
    per_sm = getattr(props, "shared_memory_per_multiprocessor", None)
    optin = getattr(props, "shared_memory_per_block_optin", None)
    if per_sm is None or optin is None:
        import cuda.bindings.driver as cuda

        attr = cuda.CUdevice_attribute
        ordinal = torch.device(device).index if torch.device(device).index is not None else torch.cuda.current_device()
        err, handle = cuda.cuDeviceGet(ordinal)
        if int(err) != 0:
            raise RuntimeError(f"{NAME}: cuDeviceGet({ordinal}) failed: {err}")
        if per_sm is None:
            err, per_sm = cuda.cuDeviceGetAttribute(attr.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, handle)
            if int(err) != 0:
                raise RuntimeError(f"{NAME}: cuDeviceGetAttribute(MAX_SHARED_MEMORY_PER_MULTIPROCESSOR) failed: {err}")
        if optin is None:
            err, optin = cuda.cuDeviceGetAttribute(attr.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, handle)
            if int(err) != 0:
                raise RuntimeError(f"{NAME}: cuDeviceGetAttribute(MAX_SHARED_MEMORY_PER_BLOCK_OPTIN) failed: {err}")
    return int(per_sm), int(optin)


def device_multiprocessor_count(device) -> int:
    """The SM count of a CUDA device (``cudaDevAttrMultiProcessorCount``), the second plan-time fact of the default pick
    (``default_cfg``: one CTA per SM while the ``rows x 48`` CTAs have an SM each; the shared-memory readings are the third)."""
    import torch

    return int(torch.cuda.get_device_properties(device).multi_processor_count)


def _check_dtype(name, t, want) -> None:
    import torch

    wants = want if isinstance(want, tuple) else (want,)
    if t.dtype not in wants:
        raise TypeError(f"{NAME}: {name} must be {' or '.join(str(x) for x in wants)} (kernel-native; callers convert), got {t.dtype}")


def _check_aligned(name, t) -> None:
    if t.data_ptr() % 16:
        raise ValueError(f"{NAME}: {name} must start at a 16-byte aligned address for its 128-bit loads, got data_ptr % 16 = {t.data_ptr() % 16}")


def resolve_device(device):
    """``device`` as a concrete ``torch.device``: a CUDA device given without an index is the CURRENT one (``torch.cuda.current_device()``),
    so it compares with an operand's device (always indexed for a CUDA tensor) by type AND index; any other device is kept as given."""
    import torch

    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def _check_device(tensors, device, anchor) -> None:
    """Every operand on ONE device: ``device`` (an entry's, resolved) when given, else the ``anchor`` operand's.  A ``torch.device``
    compares by type AND index, so a CPU tensor or one on another CUDA device is a ``ValueError`` naming the operand and both devices
    -- a compiled DSL entry would take either pointer silently on the entry's stream."""
    want = resolve_device(device) if device is not None else tensors[anchor].device
    whose = "the entry's" if device is not None else f"{anchor}'s"
    for name, t in tensors.items():
        if t.device != want:
            raise ValueError(f"{NAME}: every operand must live on one device, {whose} ({want}); {name} is on {t.device}")


def check_binding(rows, x, cs, cw, al, dt, state, idx, z, w, scale, hidden, baw, device=None) -> None:
    """The binding checks, run before every compile AND every launch (compile-cache hits included): ``TypeError`` for a dtype (or a
    non-tensor), ``ValueError`` for an inconsistent shape, stride, alignment or device, ``NotImplementedError`` for a geometry / layout /
    rows count this kernel does not serve.  ``device`` (an entry's: a concrete CUDA ordinal) requires every operand on exactly that device,
    type AND index; without it the operands must share ``x``'s.  The compile key pins the plan-time facts only (rows, the strides the
    kernel is specialized on, the config), and a compiled DSL entry accepts a tensor of another dtype, footprint or device silently, so
    these checks are a launch's only guard; host-only metadata arithmetic, no device work.  :func:`check_support` adds the architecture
    gate."""
    import torch

    if not isinstance(rows, int) or rows < 1:
        raise ValueError(f"{NAME}: rows must be a positive int, got {rows!r}")
    if rows > MAX_ROWS:
        raise NotImplementedError(f"{NAME} launches one grid row per request, at most {MAX_ROWS} (CUDA's grid Y limit); got rows={rows}")
    tensors = dict(x=x, cs=cs, cw=cw, al=al, dt=dt, state=state, idx=idx, z=z, w=w, scale=scale, hidden=hidden, baw=baw)
    for name, t in tensors.items():
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{NAME}: {name} must be a torch.Tensor, got {type(t).__name__}")
    _check_device(tensors, device, "x")
    _check_dtype("x", x, torch.bfloat16)
    _check_dtype("cs", cs, torch.bfloat16)
    _check_dtype("cw", cw, torch.bfloat16)
    _check_dtype("al (A_log)", al, torch.float32)
    _check_dtype("dt (dt_bias)", dt, torch.bfloat16)
    _check_dtype("state", state, torch.float32)
    _check_dtype("idx", idx, torch.int32)
    _check_dtype("z", z, torch.bfloat16)
    _check_dtype("w", w, torch.bfloat16)
    _check_dtype("scale", scale, torch.float32)
    _check_dtype("hidden", hidden, torch.bfloat16)
    _check_dtype("baw", baw, torch.bfloat16)
    # geometry
    if x.dim() != 2 or x.shape[0] < rows or x.shape[1] != QKV_CHANNELS:
        raise NotImplementedError(f"{NAME} serves x as [rows, {QKV_CHANNELS}] (16 K heads, 48 V heads x 128); got {tuple(x.shape)}")
    if x.stride(1) != 1 or x.stride(0) % 8:
        raise NotImplementedError(f"{NAME} needs x rows contiguous with a 16-byte-multiple row stride; got strides {tuple(x.stride())}")
    if cs.dim() != 3 or tuple(cs.shape[1:]) != (QKV_CHANNELS, CONV_WIDTH - 1):
        raise NotImplementedError(f"{NAME} serves the conv state as [slots, {QKV_CHANNELS}, {CONV_WIDTH - 1}] (conv width {CONV_WIDTH}); got {tuple(cs.shape)}")
    css, csd, cst = (int(s) for s in cs.stride())
    taps_apart = csd == 1 and cst % 8 == 0 and cst >= QKV_CHANNELS
    taps_contiguous = csd == CONV_WIDTH - 1 and cst == 1
    if css % 8 or not (taps_apart or taps_contiguous):
        raise NotImplementedError(
            f"{NAME} needs a conv-state layout its bulk copies can express (channel-contiguous taps or tap-contiguous channels, 16-byte-multiple slot stride); got strides {cs.stride()}"
        )
    if tuple(cw.shape) != (QKV_CHANNELS, CONV_WIDTH) or not cw.is_contiguous():
        raise NotImplementedError(f"{NAME} serves a contiguous conv weight [{QKV_CHANNELS}, {CONV_WIDTH}]; got {tuple(cw.shape)} strides {tuple(cw.stride())}")
    if al.numel() < V_HEADS or dt.numel() < V_HEADS or not al.is_contiguous() or not dt.is_contiguous():
        raise NotImplementedError(f"{NAME} serves contiguous A_log / dt_bias of {V_HEADS} V heads; got {tuple(al.shape)} / {tuple(dt.shape)}")
    if state.dim() != 4 or tuple(state.shape[1:]) != (V_HEADS, HEAD_DIM, HEAD_DIM):
        raise NotImplementedError(f"{NAME} serves the fp32 state as [slots, {V_HEADS}, {HEAD_DIM}, {HEAD_DIM}]; got {tuple(state.shape)}")
    if tuple(state.stride()[1:]) != (HEAD_DIM * HEAD_DIM, HEAD_DIM, 1) or state.stride(0) % 4:
        raise NotImplementedError(
            f"{NAME} needs head slabs of {HEAD_DIM} x {HEAD_DIM} contiguous fp32 and a 16-byte-multiple slot stride; got strides {tuple(state.stride())}"
        )
    if not idx.is_contiguous() or idx.numel() < rows:
        raise ValueError(f"{NAME}: idx must be a contiguous int32 vector of at least rows={rows} slots; got {tuple(idx.shape)} strides {tuple(idx.stride())}")
    if z.dim() != 3 or z.shape[0] < rows or tuple(z.shape[1:]) != (V_HEADS, HEAD_DIM):
        raise NotImplementedError(f"{NAME} serves z as [rows, {V_HEADS}, {HEAD_DIM}]; got {tuple(z.shape)}")
    if tuple(z.stride()[1:]) != (HEAD_DIM, 1) or z.stride(0) % 8:
        raise NotImplementedError(f"{NAME} needs z heads contiguous with a 16-byte-multiple row stride; got strides {tuple(z.stride())}")
    if w.numel() != HEAD_DIM or not w.is_contiguous():
        raise NotImplementedError(f"{NAME} serves a contiguous norm weight of {HEAD_DIM}; got {tuple(w.shape)}")
    if scale.numel() < 1:
        raise ValueError(f"{NAME}: scale must hold the static FP8 scale (one fp32 element); got {tuple(scale.shape)}")
    if hidden.dim() != 2 or hidden.shape[0] < rows or hidden.shape[1] != HIDDEN or not hidden.is_contiguous():
        raise NotImplementedError(f"{NAME} serves a contiguous hidden of [rows, {HIDDEN}]; got {tuple(hidden.shape)} strides {tuple(hidden.stride())}")
    if tuple(baw.shape) != (2 * V_HEADS, HIDDEN) or not baw.is_contiguous():
        raise NotImplementedError(f"{NAME} serves a contiguous BA weight of [{2 * V_HEADS}, {HIDDEN}]; got {tuple(baw.shape)}")
    for name, t in tensors.items():
        if t.numel() > 1:  # every multi-element operand is declared 16-byte aligned to the kernel (_flat); a one-element scale is not
            _check_aligned(name, t)


def check_support(rows, x, cs, cw, al, dt, state, idx, z, w, scale, hidden, baw, device_cc=None, device=None) -> None:
    """Typed declines, before any DSL import: the binding checks of :func:`check_binding` (``device`` as there) plus the architecture
    gate (``NotImplementedError`` for a part this kernel does not serve).  ``device_cc`` overrides the compute-capability probe (tests)."""
    import torch

    check_binding(rows, x, cs, cw, al, dt, state, idx, z, w, scale, hidden, baw, device=device)
    dev = x.device
    if device_cc is None and dev.type != "cuda":
        raise NotImplementedError(f"{NAME} serves CUDA tensors only; the operands are on {dev}")
    cc = tuple(torch.cuda.get_device_capability(dev)) if device_cc is None else tuple(device_cc)
    if not cc_supported(cc):
        raise NotImplementedError(f"{NAME} serves compute capability {SUPPORTED_CC_TEXT}; this device is {cc[0]}.{cc[1]}")


def _flat(t):
    """A 1-D DSL view covering a (possibly strided) tensor's whole footprint (16-byte aligned; the kernel addresses by element)."""
    from cutlass.cute.runtime import from_dlpack

    n = 1 + sum((d - 1) * s for d, s in zip(t.shape, t.stride()))
    return from_dlpack(t.detach().as_strided((n,), (1,)), assumed_align=16 if n > 1 else 4)


class GdnDecodeFused:
    """Compiled per (rows, strides, cfg) and shared by every GDN layer: all per-layer tensors are launch arguments.  ``cfg`` overrides
    the config as a whole; without it the entry picks the measured form per rows on this part (``default_cfg`` on the device's SM count
    and shared-memory readings: ``PAIR_CFG`` = two CTAs per head while the ``rows x 96`` CTAs have an SM each, ``CFG`` = one CTA per SM
    while the ``rows x 48`` CTAs have one, ``CAPPED_CFG`` = two CTAs per SM past that where the part's shared memory holds two of its
    CTAs, else ``CFG`` in two waves).  The entry starts from the one-CTA form the part holds (a part holding none is a typed
    ``NotImplementedError`` at construction).  An entry built for a non-CUDA device (host tests) has no SM count to pick by and keeps
    ``CFG`` on the cc 10.0 readings.  The device is resolved to a concrete CUDA ordinal at construction (:func:`resolve_device`), and every
    operand of every compile / launch must live on exactly it.  ``pdl`` / ``pdl_release`` (None = the form's own fields) are the launch
    knobs on top of the form that serves."""

    def __init__(self, device, eps: float, cfg: Optional[Cfg] = None, debug: bool = False, pdl: Optional[bool] = None, pdl_release: Optional[bool] = None):
        import torch

        self.device = resolve_device(device)  # a CUDA device without an index is the current ordinal: the one every operand must match
        self.eps = float(eps)
        self.cfg_given = cfg is not None
        self.debug_flag = bool(debug)
        # the launch knobs on top of whatever form serves: None keeps the form's own PDL / PDL_RELEASE fields (the picks: off)
        self.pdl: Optional[bool] = None if pdl is None else bool(pdl)
        self.pdl_release: Optional[bool] = None if pdl_release is None else bool(pdl_release)
        self.sm_count: Optional[int] = None
        # THIS part's readings, (shared memory per SM, per-CTA opt-in): the pick's budget and every form's pad; the cc 10.0 readings
        # until a CUDA device replaces them -- never a constant carried across an architecture line
        self.readings: Tuple[int, int] = (SM100_SMEM_PER_SM_BYTES, SM100_SMEM_OPTIN_BYTES)
        self.part: Dict[str, int] = {}  # the readings as Cfg fields, stamped into every form on a CUDA device (empty on a host entry)
        if self.device.type == "cuda":
            self.readings = device_shared_memory(self.device)
            self.part = dict(SMEM_PER_SM_BYTES=self.readings[0], SMEM_OPTIN_BYTES=self.readings[1])
            self.sm_count = device_multiprocessor_count(self.device)
        # the override, or the one-CTA form the pick starts from -- the one this part holds (CFG on every served part; a part that holds
        # none is the typed decline here, before any form is stamped)
        start = cfg if cfg is not None else (first_fitting(ONE_CTA_FORMS, *self.readings) if self.part else CFG)
        self.cfg = self._on_this_part(start)
        self.forms: Dict[Cfg, Cfg] = {}  # the picked forms on this part, by their module form
        self.compiled: Dict[tuple, Any] = {}
        self.buffers: Dict[int, Dict[str, Any]] = {}

    def _on_this_part(self, cfg: Cfg) -> Cfg:
        """``cfg`` with this part's shared-memory readings, the debug flag and the launch knobs, validated."""
        cfg = replace(cfg, DEBUG=self.debug_flag, **self.part, **self.launch_fields(cfg))
        validate_cfg(cfg)
        return cfg

    def launch_fields(self, cfg: Cfg) -> Dict[str, bool]:
        """The ``PDL`` / ``PDL_RELEASE`` fields the entry's ``pdl`` / ``pdl_release`` knobs set on ``cfg``: ``pdl`` sets ``PDL`` and, unless
        ``pdl_release`` says otherwise, ``PDL_RELEASE`` with it (the reference verify kernel's placement: the release at entry); a knob
        left None keeps the form's own field.  The fields are part of the config, so of the compile key: a PDL plan and a plain plan of
        one form are two compiled kernels."""
        if self.pdl is None and self.pdl_release is None:
            return {}
        pdl = cfg.PDL if self.pdl is None else self.pdl
        release = (pdl if self.pdl is not None else cfg.PDL_RELEASE) if self.pdl_release is None else self.pdl_release
        return dict(PDL=pdl, PDL_RELEASE=release)

    def cfg_for(self, rows: int) -> Cfg:
        """The config a plan for ``rows`` compiles with: the caller's ``cfg`` when one was given, else the measured pick per rows
        (``default_cfg`` on this device's SM count and shared-memory readings -- the capped two-CTAs-per-SM form only where two of its
        CTAs share an SM on this part; ``CFG`` when there is no CUDA device to pick by)."""
        if self.cfg_given or self.sm_count is None:
            return self.cfg
        form = default_cfg(rows, self.sm_count, *self.readings)
        if form not in self.forms:
            self.forms[form] = self._on_this_part(form)
        return self.forms[form]

    @property
    def debug(self) -> bool:
        return self.cfg.DEBUG

    def outputs(self, rows: int):
        """The per-rows output buffers (the reference kernel's set): ``qout`` fp8 [rows, 48, 128], ``out`` bf16 [rows, 48, 128],
        ``counts`` int32 [rows * 512] (the reader tickets, zero between launches) and, under ``DEBUG`` only, the debug buffers ``raw``
        [rows, 48, 128] bf16, ``convout`` [rows, 10240] bf16, ``baout`` [rows, 96] bf16 -- without ``DEBUG`` the kernel's debug slots are
        compiled out (``None``), never bound to an allocation.  Entry-owned scratch, allocated once per rows count at the first compile /
        call for it (outside any capture; a FlashInfer runner has no workspace contract to carve it from)."""
        if rows not in self.buffers:
            import torch

            dev = self.device
            buffers = dict(
                qout=torch.empty(rows, V_HEADS, HEAD_DIM, dtype=torch.float8_e4m3fn, device=dev),
                out=torch.empty(rows, V_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev),
                counts=torch.zeros(rows * TICKET_STRIDE, dtype=torch.int32, device=dev),
            )
            if self.debug:
                buffers.update(
                    raw=torch.zeros(rows, V_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev),
                    convout=torch.zeros(rows, QKV_CHANNELS, dtype=torch.bfloat16, device=dev),
                    baout=torch.zeros(rows, 2 * V_HEADS, dtype=torch.bfloat16, device=dev),
                )
            self.buffers[rows] = buffers
        return self.buffers[rows]

    def _debug_args(self, b) -> list:
        """The kernel's three DEBUG slots: the dumps' views under ``DEBUG``, else ``None`` x 3 -- the DSL specializes on ``None`` and
        compiles the slots out, so a dead slot is never bound to an allocation."""
        return [_flat(b["raw"]), _flat(b["convout"]), _flat(b["baout"])] if self.debug else [None, None, None]

    def key(self, rows, x, cs, state, z):
        """The compile key: plan-time facts only (rows, strides, the config of this rows count) -- never a runtime value (Rule 4)."""
        return (rows, self.cfg_for(rows), int(x.stride(0)), int(z.stride(0)), tuple(int(s) for s in cs.stride()), int(state.stride(0)), self.eps)

    def _args(self, rows, x, cs, cw, al, dt, state, idx, z, w, scale, hidden, baw):
        b = self.outputs(rows)
        import torch

        return [
            _flat(x),
            _flat(cs),
            _flat(cw),
            _flat(al),
            _flat(dt),
            _flat(state),
            _flat(idx),
            _flat(z),
            _flat(w),
            _flat(scale),
            _flat(b["counts"]),
            _flat(b["out"]),
            _flat(b["qout"].view(torch.uint8)),
            _flat(hidden),
            _flat(baw),
            *self._debug_args(b),
        ]

    def compile(self, rows, x, cs, cw, al, dt, state, idx, z, w, scale, hidden, baw) -> None:
        """Validate (typed declines, on every call: every operand on this entry's device, a compile-cache hit included), gate the DSL,
        trace-compile the host entry for this (rows, strides) once."""
        check_support(rows, x, cs, cw, al, dt, state, idx, z, w, scale, hidden, baw, device=self.device)
        key = self.key(rows, x, cs, state, z)
        if key in self.compiled:
            return
        import torch

        msg = _dsl_version_error()
        if msg is not None:
            raise NotImplementedError(msg)
        msg = cutedsl_arch_requirement_error(tuple(torch.cuda.get_device_capability(self.device)))
        if msg is not None:
            raise NotImplementedError(msg)
        import cutlass.cute as cute
        from cutlass.cute.runtime import make_fake_stream

        from .kernel import gdn_decode_fused_f16 as mod

        args = self._args(rows, x, cs, cw, al, dt, state, idx, z, w, scale, hidden, baw)
        with torch.cuda.device(self.device):
            self.compiled[key] = cute.compile(
                mod.launch,
                self.cfg_for(rows),
                *args,
                rows,
                int(x.stride(0)),
                int(z.stride(0)),
                *(int(s) for s in cs.stride()),
                int(state.stride(0)),
                self.eps,
                make_fake_stream(),
            )

    def __call__(self, rows, x, cs, cw, al, dt, state, idx, z, w, scale, hidden, baw):
        """Launch on torch's current stream; returns the fp8 ``qout`` [rows, 48, 128] for the out projection (``outputs(rows)``
        holds ``out`` and ``counts``).  Compiles on first use for a new (rows, strides).  Every launch re-runs the binding checks
        (:func:`check_binding`, every operand on this entry's device) before the compiled kernel is looked up: the key pins no dtype,
        shape, alignment or device, and the DSL checks none."""
        import cuda.bindings.driver as cuda
        import torch

        check_binding(rows, x, cs, cw, al, dt, state, idx, z, w, scale, hidden, baw, device=self.device)
        key = self.key(rows, x, cs, state, z)
        if key not in self.compiled:
            self.compile(rows, x, cs, cw, al, dt, state, idx, z, w, scale, hidden, baw)
        args = self._args(rows, x, cs, cw, al, dt, state, idx, z, w, scale, hidden, baw)
        self.compiled[key](*args, cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream))
        return self.outputs(rows)["qout"]
