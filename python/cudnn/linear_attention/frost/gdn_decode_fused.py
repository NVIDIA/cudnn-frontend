# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host entry of the fused Gated DeltaNet DECODE step (``kernel/gdn_decode_fused_f16.py``): validation with typed declines, the
plan-time compile and the launch.  No DSL import at module level -- the CuTe DSL is imported inside :meth:`GdnDecodeFused.compile`
after the version / architecture gate (python/cudnn/AGENTS.md Rule 7), so a host without it can still construct the entry and run
its declines.

Served geometry (declined otherwise, by name): the checkpoint constants of ``kernel/gdn_decode_fused_config.py`` -- the QKV slab
``x`` [rows, 10240] bf16 (a row-strided view is fine), the conv state ``cs`` [slots, 10240, 3] bf16 with channel-contiguous taps or
tap-contiguous channels, the conv weight ``cw`` [10240, 4] bf16, ``A_log`` [48] fp32, ``dt_bias`` [48] bf16, the fp32 state
``[slots, 48, 128, 128]`` (slot-strided), the int32 slot indices, ``z`` [rows, 48, 128] bf16 (row-strided), the norm weight [128]
bf16, the static FP8 scale (fp32, one element), ``hidden`` [rows, 5120] bf16 and the BA weight [96, 5120] bf16; every operand 16-byte
aligned; an SM100-class part (compute capability 10.x) and a CuTe DSL >= 4.8.0.  The config is the caller's ``cfg`` as a whole or, without
one, the measured pick per rows (``default_cfg``: a cluster of two CTAs per head while the ``rows x 96`` CTAs have an SM each on THIS
device, one CTA per SM while the ``rows x 48`` CTAs have one, the register-resident two-CTAs-per-SM form past that); the device's SM
count, its shared memory per SM and its per-CTA opt-in maximum (the occupancy knob's pad) are read at construction (every shared-memory
figure in this file's docstrings is a cc 10.0 reading; another part is served on its own readings).  The launch knobs ``pdl`` /
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
    QKV_CHANNELS,
    TICKET_STRIDE,
    V_HEADS,
    Cfg,
    default_cfg,
    validate_cfg,
)

SUPPORTED_CC_MAJOR = 10  # SM100-class parts (compute capability 10.x); the DSL's target gate (cutedsl_arch_requirement_error) decides per part
SUPPORTED_CC_TEXT = "10.x"
DSL_FLOOR = (4, 8, 0)
NAME = "the fused GDN decode step"


def cc_supported(cc) -> bool:
    """Whether a ``(major, minor)`` compute capability is one this kernel serves: every SM100-class part (10.x)."""
    return cc is not None and int(cc[0]) == SUPPORTED_CC_MAJOR


def _dsl_version_error() -> Optional[str]:
    """The Rule-7 gate of this kernel: the library floor (4.7.0) plus this kernel's own floor (4.8.0: the public wheel every
    consumer pins); None when the installed DSL serves."""
    msg = cutedsl_requirement_error(NAME)
    if msg is not None:
        return msg
    installed, version = cutedsl_state()
    if not installed:
        return f"{NAME} requires nvidia-cutlass-dsl >= 4.8.0; the CuTe DSL is not installed"
    if version is not None and version[0] == "nvidia-cutlass-dsl":
        parts = []
        for component in version[1].split("+", 1)[0].split(".")[:3]:
            digits = ""
            for ch in component:
                if not ch.isdigit():
                    break
                digits += ch
            if not digits:
                break
            parts.append(int(digits))
        parts += [0] * (3 - len(parts))
        if tuple(parts) < DSL_FLOOR:
            return f"{NAME} requires nvidia-cutlass-dsl >= 4.8.0; found {version[1]}. Upgrade with: pip install -U 'nvidia-cutlass-dsl>=4.8.0'"
    return None


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
    (``default_cfg``: one CTA per SM while the ``rows x 48`` CTAs have an SM each)."""
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


def check_support(rows, x, cs, cw, al, dt, state, idx, z, w, scale, hidden, baw, device_cc=None) -> None:
    """Typed declines, before any DSL import: ``NotImplementedError`` for a geometry / layout / architecture this kernel does not
    serve, ``TypeError`` for a dtype, ``ValueError`` for an inconsistent shape, stride or alignment.  ``device_cc`` overrides the
    compute-capability probe (tests)."""
    import torch

    if not isinstance(rows, int) or rows < 1:
        raise ValueError(f"{NAME}: rows must be a positive int, got {rows!r}")
    tensors = dict(x=x, cs=cs, cw=cw, al=al, dt=dt, state=state, idx=idx, z=z, w=w, scale=scale, hidden=hidden, baw=baw)
    dev = x.device
    for name, t in tensors.items():
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{NAME}: {name} must be a torch.Tensor, got {type(t).__name__}")
        if t.device != dev:
            raise ValueError(f"{NAME}: every operand must live on one device; {name} is on {t.device}, x on {dev}")
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
    if device_cc is None and dev.type != "cuda":
        raise NotImplementedError(f"{NAME} serves CUDA tensors only; the operands are on {dev}")
    cc = tuple(torch.cuda.get_device_capability(dev)) if device_cc is None else tuple(device_cc)
    if not cc_supported(cc):
        raise NotImplementedError(f"{NAME} serves compute capability {SUPPORTED_CC_TEXT} (SM100-class parts); this device is {cc[0]}.{cc[1]}")


def _flat(t):
    """A 1-D DSL view covering a (possibly strided) tensor's whole footprint (16-byte aligned; the kernel addresses by element)."""
    from cutlass.cute.runtime import from_dlpack

    n = 1 + sum((d - 1) * s for d, s in zip(t.shape, t.stride()))
    return from_dlpack(t.detach().as_strided((n,), (1,)), assumed_align=16 if n > 1 else 4)


class GdnDecodeFused:
    """Compiled per (rows, strides, cfg) and shared by every GDN layer: all per-layer tensors are launch arguments.  ``cfg`` overrides
    the config as a whole; without it the entry picks the measured form per rows (``default_cfg``: ``PAIR_CFG`` = two CTAs per head
    while the ``rows x 96`` CTAs have an SM each on this device, ``CFG`` = one CTA per SM while the ``rows x 48`` CTAs have one,
    ``CAPPED_CFG`` = two CTAs per SM past that).  An entry built for a non-CUDA device (host tests) has no SM count to pick by and
    keeps ``CFG``.  ``pdl`` / ``pdl_release`` (None = the form's own fields) are the launch knobs on top of the form that serves."""

    def __init__(self, device, eps: float, cfg: Optional[Cfg] = None, debug: bool = False, pdl: Optional[bool] = None, pdl_release: Optional[bool] = None):
        import torch

        self.device = torch.device(device)
        self.eps = float(eps)
        self.cfg_given = cfg is not None
        self.debug_flag = bool(debug)
        # the launch knobs on top of whatever form serves: None keeps the form's own PDL / PDL_RELEASE fields (the picks: off)
        self.pdl: Optional[bool] = None if pdl is None else bool(pdl)
        self.pdl_release: Optional[bool] = None if pdl_release is None else bool(pdl_release)
        self.sm_count: Optional[int] = None
        self.part: Dict[str, int] = {}  # THIS part's readings every form takes: shared memory per SM and per-CTA opt-in (the pad)
        if self.device.type == "cuda":
            # the occupancy pad is derived from THIS part's shared memory per SM and per-CTA opt-in, never from a carried constant
            per_sm, optin = device_shared_memory(self.device)
            self.part = dict(SMEM_PER_SM_BYTES=per_sm, SMEM_OPTIN_BYTES=optin)
            self.sm_count = device_multiprocessor_count(self.device)
        self.cfg = self._on_this_part(cfg if cfg is not None else CFG)  # the override, or the one-CTA form the pick starts from
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
        (``default_cfg`` with this device's SM count; ``CFG`` when there is no CUDA device to pick by)."""
        if self.cfg_given or self.sm_count is None:
            return self.cfg
        form = default_cfg(rows, self.sm_count)
        if form not in self.forms:
            self.forms[form] = self._on_this_part(form)
        return self.forms[form]

    @property
    def debug(self) -> bool:
        return self.cfg.DEBUG

    def outputs(self, rows: int):
        """The per-rows output buffers (the reference kernel's set): ``qout`` fp8 [rows, 48, 128], ``out`` bf16 [rows, 48, 128],
        ``counts`` int32 [rows * 512] (the reader tickets, zero between launches) and the debug buffers ``raw`` [rows, 48, 128] bf16,
        ``convout`` [rows, 10240] bf16, ``baout`` [rows, 96] bf16 (written only under ``DEBUG``)."""
        if rows not in self.buffers:
            import torch

            dev = self.device
            self.buffers[rows] = dict(
                qout=torch.empty(rows, V_HEADS, HEAD_DIM, dtype=torch.float8_e4m3fn, device=dev),
                out=torch.empty(rows, V_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev),
                counts=torch.zeros(rows * TICKET_STRIDE, dtype=torch.int32, device=dev),
                raw=torch.zeros(rows, V_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev),
                convout=torch.zeros(rows, QKV_CHANNELS, dtype=torch.bfloat16, device=dev),
                baout=torch.zeros(rows, 2 * V_HEADS, dtype=torch.bfloat16, device=dev),
            )
        return self.buffers[rows]

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
            _flat(b["raw"]),
            _flat(b["convout"]),
            _flat(b["baout"]),
        ]

    def compile(self, rows, x, cs, cw, al, dt, state, idx, z, w, scale, hidden, baw) -> None:
        """Validate (typed declines), gate the DSL, trace-compile the host entry for this (rows, strides) once."""
        key = self.key(rows, x, cs, state, z)
        if key in self.compiled:
            return
        check_support(rows, x, cs, cw, al, dt, state, idx, z, w, scale, hidden, baw)
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
        holds ``out`` and ``counts``).  Compiles on first use for a new (rows, strides)."""
        import cuda.bindings.driver as cuda
        import torch

        key = self.key(rows, x, cs, state, z)
        if key not in self.compiled:
            self.compile(rows, x, cs, cw, al, dt, state, idx, z, w, scale, hidden, baw)
        args = self._args(rows, x, cs, cw, al, dt, state, idx, z, w, scale, hidden, baw)
        self.compiled[key](*args, cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream))
        return self.outputs(rows)["qout"]
