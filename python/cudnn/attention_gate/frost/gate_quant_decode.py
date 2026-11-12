# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host entry of the attention gate + quantize DECODE kernel (``kernel/gate_quant_decode_f16.py``): validation with typed declines, the
plan-time compile and the one launch.  No DSL import at module level -- the CuTe DSL is imported inside
:meth:`AttentionGateQuantDecode.compile` after the version / architecture gates (python/cudnn/AGENTS.md Rule 7), so a host without it can
still construct the entry and run its declines.  The binding checks (:func:`check_binding`: type, device, dtype, shape, contiguity,
alignment and overlap of every operand) run before EVERY compile and EVERY launch, compile-cache hits included -- the compile key pins the
config only, and a compiled DSL entry accepts a tensor of another dtype, footprint or device silently, so the host is a launch's only guard.

The op, per element of an ``[rows, width]`` slab::

    out[r, c] = e4m3( bf16( attn[r, c] / (1 + exp(-gate[r, c])) ) * (1 / scale) )

fp32 arithmetic on the bf16 operands; the gated value is rounded ONCE to bf16 (nearest-even), multiplied by the correctly rounded fp32
reciprocal of ``scale``, and rounded ONCE to E4M3 (nearest-even, saturating at +-448).  ``attn`` is the attention output, ``gate`` the
PRE-sigmoid gate, ``scale`` the static per-tensor dequantization scale of the consumer (a following projection with an E4M3 input), ``out``
that consumer's input.  The gated value is the DIVISION ``attn / (1 + exp(-gate))``: a chain that rounds the product
``attn * sigmoid(gate)`` instead writes a code one apart on rare elements, so a reference must spell the division.

Served (declined otherwise, by name): ``rows`` 1 .. 160 (a RUNTIME kernel argument: one compiled kernel serves every row count), ``width``
a positive multiple of 8 up to 32768 (fixed at construction), ``attn`` / ``gate`` bf16 and contiguous, ``scale`` one fp32 element (0-d or
``[1]``; a positive normal value by contract -- its value is never read on the host, and a zero, subnormal, negative or non-finite scale
yields saturated, sign-flipped or NaN bytes), ``out`` E4M3 and contiguous, every operand 16-byte aligned (``scale``: 4) and on the entry's
device, ``out`` not overlapping an input, a compute capability 10.x part, a CuTe DSL >= 4.8.0 that holds the part's target and compiles it
(cc 10.7: the CUDA 13 toolchain flavour; cc 10.0 / 10.3: either flavour).  A NaN or infinite operand element yields the E4M3 NaN byte or a
saturated code for that element only; nothing is checked on the host (Rule 3).

Launch knob ``pdl_release`` sets the config's ``PDL_RELEASE`` = ``griddepcontrol.launch_dependents`` at the kernel's entry (default off; a
knob measured before it defaults on).  CALLER OBLIGATION when on: every grid that can start before this kernel completes -- a grid launched
with the programmatic-stream-serialization attribute, and transitively its own dependents -- issues its wait before it reads ``out`` and
before it writes ``attn``, ``gate`` or ``scale``.  The launch itself is ordinary (no attribute, no wait in the kernel): it starts when its
stream predecessor has completed, on ``torch.cuda.current_stream`` of the entry's device.  That device need not be the caller's current
one: the compile and the launch run under the entry's device and the caller's current device is restored behind them (Rule 5: the device
is implicit state exactly like the stream).

Output ownership.  With ``out`` the caller owns the storage and gets its own tensor back.  Without it the entry returns a view of ONE
entry-owned ``[160, width]`` buffer, allocated at the first compile or call that passes no ``out`` (never on a warm path; allocate it
outside a capture by compiling without ``out`` first): its bytes are valid until the entry's NEXT call from ANY host thread.  Stream order
does not serialize host calls, so two host threads that share one entry -- on one stream or two -- pass their own ``out`` (or build one
entry each).  Workspace: none.  The launch reads no device memory to the host (Rule 3); the warm path allocates nothing (Rule 1).
"""

from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

from cudnn._torch_stream import device_context  # imports no framework at module level
from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error, cutedsl_state
from cudnn.frost.launch_facts import LaunchShape, kernel_symbols, plain_form

from .kernel.gate_quant_decode_config import MAX_ROWS, Cfg, blocks_for, default_cfg, validate_cfg, validate_rows, validate_width

#: the served compute capabilities as closed ranges of ``10 * major + minor``: SM100-class parts only.
SUPPORTED_CC_RANGES = ((100, 109),)
SUPPORTED_CC_TEXT = "10.x (SM100-class)"
DSL_FLOOR = (4, 8, 0)
DSL_FLOOR_TEXT = ".".join(str(x) for x in DSL_FLOOR)
#: cc 10.7 only: the oldest DSL compiler observed to assemble ``sm_107a`` (the CUDA 13 flavour of DSL 4.8.0); CUDA 12.9 rejects the target.
SM107_MIN_COMPILER = (13, 4)
NAME = "the attention gate + quantize decode step"
ALIGN_BYTES = 16  # the 128-bit loads of attn / gate; out's DSL view declares the same alignment
SCALE_ALIGN_BYTES = 4


def cc_supported(cc) -> bool:
    """Whether a ``(major, minor)`` compute capability is one this kernel serves; ``None`` (no CUDA device) is not."""
    if cc is None:
        return False
    code = 10 * int(cc[0]) + int(cc[1])
    return any(lo <= code <= hi for lo, hi in SUPPORTED_CC_RANGES)


def _dsl_version_error() -> Optional[str]:
    """The Rule-7 version gate: the step's own floor (4.8.0) through the one version parser of :mod:`cudnn.frost.buffers`, without
    importing the DSL; None when it serves."""
    installed, _ = cutedsl_state()
    if not installed:
        return f"{NAME} requires nvidia-cutlass-dsl >= {DSL_FLOOR_TEXT}; the CuTe DSL is not installed"
    return cutedsl_requirement_error(NAME, floor=DSL_FLOOR)


def _dsl_target_error(device_cc) -> Optional[str]:
    """The Rule-7 target gate: a version floor does not guarantee the part's target (cc 10.7 needs a DSL build whose target list holds
    ``sm_107a``).  The first gate that imports ``cutlass.base_dsl`` -- call it after :func:`_dsl_version_error` returned None."""
    msg = cutedsl_arch_requirement_error(tuple(device_cc))
    return None if msg is None else f"{NAME}: {msg}"


def dsl_compiler_cuda_version() -> Tuple[int, int]:
    """``(major, minor)`` of the CUDA compiler the installed CuTe DSL bound (its CUDA 12 or CUDA 13 toolchain flavour: the DSL loads the
    newest installed flavour the driver supports), from ``cutlass.base_dsl.version_info.CUDA_VERSION`` -- the ONE seam the compiler gate
    reads (tests patch it).  Never ``torch.version.cuda`` or ``CUDA_HOME``: neither names the compiler the DSL wheel selected."""
    from cutlass.base_dsl.version_info import CUDA_VERSION

    return (int(CUDA_VERSION.major), int(CUDA_VERSION.minor))


def _dsl_compiler_error(device_cc) -> Optional[str]:
    """The Rule-7 compiler gate.  On cc 10.0 / 10.3 the kernel's PTX is the same text under both toolchain flavours of DSL 4.8.0 (CUDA 12.9
    and CUDA 13.4; the suite's rendering cell pins it), so both are served.  cc 10.7 is not: the CUDA 12.9 flavour rejects the ``sm_107a``
    target at compile time although the DSL's target list names it, so a compiler below ``SM107_MIN_COMPILER`` (the one observed to
    assemble it) is declined here, before JIT, naming what was found."""
    if tuple(device_cc) != (10, 7):
        return None
    found = dsl_compiler_cuda_version()
    if found >= SM107_MIN_COMPILER:
        return None
    need = ".".join(str(x) for x in SM107_MIN_COMPILER)
    return (
        f"{NAME} on compute capability 10.7 requires a CuTe DSL whose compiler is CUDA {need} or later (the sm_107a target); the installed "
        f"DSL compiles with CUDA {found[0]}.{found[1]}, which rejects that target. Install the CUDA 13 toolchain flavour "
        f"(pip install 'nvidia-cutlass-dsl[cu13]>={DSL_FLOOR_TEXT}') under a CUDA 13 driver"
    )


def _dsl_error(device_cc) -> Optional[str]:
    """The DSL gates in order -- installed, the version floor, the part's target, the compiler that assembles it; None when the installed
    DSL serves the part.  Only the last two import the DSL (``cutlass.base_dsl``), after the version gate."""
    msg = _dsl_version_error()
    if msg is None:
        msg = _dsl_target_error(device_cc)
    if msg is None:
        msg = _dsl_compiler_error(device_cc)
    return msg


def resolve_device(device):
    """``device`` as a concrete ``torch.device``: a CUDA device given without an index is the CURRENT one, so it compares with an operand's
    device (always indexed for a CUDA tensor) by type AND index; any other device is kept as given."""
    import torch

    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def _byte_range(t, elements: int) -> Tuple[int, int]:
    start = t.data_ptr()
    return start, start + elements * t.element_size()


def check_binding(attn, gate, scale, out=None, device=None, width: Optional[int] = None) -> int:
    """The binding checks, run before every compile AND every launch (compile-cache hits included): ``TypeError`` for a non-tensor or a
    dtype, ``ValueError`` for an inconsistent device, shape, contiguity, alignment or an ``out`` that overlaps an input,
    ``NotImplementedError`` for a row count or a width this kernel does not serve.  Returns ``rows = attn.shape[0]``.  ``device`` (an
    entry's: a concrete CUDA ordinal) requires every operand on exactly that device; without it the operands must share ``attn``'s.
    ``width`` (an entry's) requires ``attn.shape[1] == width``; without it the width is ``attn``'s own.  Host-only metadata arithmetic, no
    device work: ``scale``'s VALUE is never read."""
    import torch

    tensors = dict(attn=attn, gate=gate, scale=scale)
    if out is not None:
        tensors["out"] = out
    for name, t in tensors.items():
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{NAME}: {name} must be a torch.Tensor, got {type(t).__name__}")
    want = resolve_device(device) if device is not None else attn.device
    whose = "the entry's" if device is not None else "attn's"
    for name, t in tensors.items():
        if t.device != want:
            raise ValueError(f"{NAME}: every operand must live on one device, {whose} ({want}); {name} is on {t.device}")
    for name, t, dtype in (("attn", attn, torch.bfloat16), ("gate", gate, torch.bfloat16), ("scale", scale, torch.float32), ("out", out, torch.float8_e4m3fn)):
        if t is not None and t.dtype != dtype:
            raise TypeError(f"{NAME}: {name} must be {dtype} (kernel-native; callers convert), got {t.dtype}")
    if attn.dim() != 2:
        raise ValueError(f"{NAME}: attn must be [rows, width] with at least one row; got {tuple(attn.shape)}")
    if width is None:
        width = validate_width(int(attn.shape[1]))
    rows = int(attn.shape[0])
    if attn.shape[1] != width or rows < 1:
        raise ValueError(f"{NAME}: attn must be [rows, {width}] with at least one row; got {tuple(attn.shape)}")
    if gate.shape != attn.shape:
        raise ValueError(f"{NAME}: gate must have attn's shape {tuple(attn.shape)}; got {tuple(gate.shape)}")
    validate_rows(rows)
    for name, t in (("attn", attn), ("gate", gate)):
        if not t.is_contiguous():
            raise ValueError(f"{NAME}: {name} must be contiguous (the kernel addresses the slab by flat element); got strides {tuple(t.stride())}")
    if scale.numel() != 1:
        raise ValueError(f"{NAME}: scale must be one fp32 element (the consumer's static per-tensor scale); got {tuple(scale.shape)}")
    if out is not None:
        if out.dim() != 2 or out.shape[0] < rows or out.shape[1] != width:
            raise ValueError(f"{NAME}: out must be [>= {rows}, {width}]; got {tuple(out.shape)}")
        if not out.is_contiguous():
            raise ValueError(f"{NAME}: out must be contiguous (the kernel addresses the slab by flat element); got strides {tuple(out.stride())}")
    for name, t in tensors.items():
        align = SCALE_ALIGN_BYTES if name == "scale" else ALIGN_BYTES
        if t.data_ptr() % align:
            what = "its 32-bit load" if name == "scale" else ("the kernel's view of it" if name == "out" else "its 128-bit loads")
            raise ValueError(f"{NAME}: {name} must start at a {align}-byte aligned address for {what}, got data_ptr % {align} = {t.data_ptr() % align}")
    if out is not None:
        elements = rows * width
        o0, o1 = _byte_range(out, elements)
        for name, t, n in (("attn", attn, elements), ("gate", gate, elements), ("scale", scale, 1)):
            t0, t1 = _byte_range(t, n)
            if o0 < t1 and t0 < o1:
                raise ValueError(f"{NAME}: out must not overlap {name} (a thread reads its inputs and writes its eight output bytes at different offsets)")
    return rows


def check_support(attn, gate, scale, out=None, device_cc=None, device=None, width: Optional[int] = None) -> int:
    """Typed declines without a DSL import: the binding checks of :func:`check_binding` plus the architecture gate (``NotImplementedError``
    for operands off a CUDA device or a part this kernel does not serve).  ``device_cc`` overrides the compute-capability probe (tests).
    The DSL gates (installed, version floor, the part's target) run in :meth:`AttentionGateQuantDecode.compile`."""
    import torch

    rows = check_binding(attn, gate, scale, out, device=device, width=width)
    dev = attn.device
    if device_cc is None and dev.type != "cuda":
        raise NotImplementedError(f"{NAME} serves CUDA tensors only; the operands are on {dev}")
    cc = tuple(torch.cuda.get_device_capability(dev)) if device_cc is None else tuple(device_cc)
    if not cc_supported(cc):
        raise NotImplementedError(f"{NAME} serves compute capability {SUPPORTED_CC_TEXT}; this device is {cc[0]}.{cc[1]}")
    return rows


def _flat(t):
    """A 1-D DSL view covering a contiguous tensor's whole footprint.  A slab (``attn``, ``gate``, ``out``: 16-byte aligned) declares a
    DYNAMIC extent with the unit stride (Rule 4: the row count is a runtime fact, so the kernel is compiled for a runtime extent and a
    launch at another row count re-binds it -- a view left static would compile the first call's extents into the kernel's symbol and
    rely on the DSL never checking them again); the one-element ``scale`` (4-byte aligned) stays static.  The kernel addresses by flat
    element and takes the element count as its own runtime argument: it never reads a view's extent."""
    from cutlass.cute.runtime import from_dlpack

    n = t.numel()
    view = from_dlpack(t.detach().as_strided((n,), (1,)), assumed_align=ALIGN_BYTES if n > 1 else SCALE_ALIGN_BYTES)
    return view.mark_layout_dynamic() if n > 1 else view


class AttentionGateQuantDecode:
    """The step compiled once per config and shared by every layer and row count: all tensors are launch arguments, the element count is a
    runtime argument.  ``width`` is the row width (``heads x head_dim`` of the attention output), fixed at construction; ``cfg`` overrides
    the config as a whole, without it the entry takes the pick for each row count on this part (``default_cfg``: one config today);
    ``pdl_release`` is the launch knob (None = the config's own field; the default: off).  The device is resolved to a concrete CUDA
    ordinal at construction, and every operand of every compile / launch must live on exactly it.  An entry built for a non-CUDA device
    (host tests) runs every decline; its compile raises the typed decline of the device.  What a row count launches is readable without
    a launch: :meth:`kernel_symbols_for`, :meth:`form_for` and :meth:`launch_shape_for` (the launch facts of
    :mod:`cudnn.frost.launch_facts`)."""

    def __init__(self, device, width: int, cfg: Optional[Cfg] = None, *, pdl_release: Optional[bool] = None):
        self.device = resolve_device(device)
        self.width: int = validate_width(width)
        self.cfg_given = cfg is not None
        self.pdl_release = None if pdl_release is None else bool(pdl_release)
        self.sm_count: Optional[int] = None
        self.cc: Optional[Tuple[int, int]] = None
        if self.device.type == "cuda":
            import torch

            self.sm_count = int(torch.cuda.get_device_properties(self.device).multi_processor_count)
            self.cc = tuple(torch.cuda.get_device_capability(self.device))
        base = validate_cfg(cfg) if cfg is not None else None
        # the config of EVERY served row count, resolved once (index rows - 1): cfg_for is a tuple read, and compile() builds every
        # distinct config, so a warm call at any row count is a compile-cache hit whatever the pick table returns
        self.cfg_by_rows: Tuple[Cfg, ...] = tuple(
            self._with_launch_fields(base if base is not None else default_cfg(rows, self.width, self.sm_count, self.cc)) for rows in range(1, MAX_ROWS + 1)
        )
        self.cfgs: Tuple[Cfg, ...] = tuple(dict.fromkeys(self.cfg_by_rows))
        self.compiled: Dict[tuple, Any] = {}
        self.out_buffer = None
        self._out_view = None

    def _with_launch_fields(self, cfg: Cfg) -> Cfg:
        """``cfg`` with the entry's launch knob (a knob left None keeps the config's own field), validated.  The field is part of the
        config, so of the compile key: the released and the plain form are two compiled entries."""
        return validate_cfg(cfg if self.pdl_release is None else replace(cfg, PDL_RELEASE=self.pdl_release))

    def cfg_for(self, rows: int) -> Cfg:
        """The config a call of ``rows`` rows runs: the caller's, else the pick for ``(rows, width)`` on this part (resolved at
        construction; a row count outside 1 .. 160 raises its typed decline)."""
        return self.cfg_by_rows[validate_rows(rows) - 1]

    def key(self, rows: int) -> tuple:
        """The compile key: plan-time facts only -- the config.  ``rows`` selects it through the pick; neither the row count nor the width
        enters the compiled kernel: the slabs bind with a dynamic extent (:func:`_flat`) and the element count is a runtime argument, so
        the kernel compiled at the first call is the one the suite's rendering cell pins, whatever that call's shape was."""
        return (self.cfg_for(rows),)

    # ---- launch facts (cudnn.frost.launch_facts): three read-only answers per served row count ----
    def _fact_cfg(self, rows) -> Cfg:
        """The config of the row count a launch fact is asked for: ``TypeError`` for a ``rows`` that is not an ``int`` (a float or a
        string would otherwise reach the per-row table as an index), then the typed declines of :meth:`cfg_for`."""
        if isinstance(rows, bool) or not isinstance(rows, int):
            raise TypeError(f"{NAME}: rows must be an int, got {type(rows).__name__}")
        return self.cfg_for(rows)

    def kernel_symbols_for(self, rows: int) -> Tuple[str, ...]:
        """The symbol of the kernel this entry has compiled for the config of ``rows``: a one-element tuple, empty until that config is
        compiled.  One compiled kernel serves every row count that shares the config (neither the row count nor the width enters it)."""
        fn = self.compiled.get((self._fact_cfg(rows),))
        return kernel_symbols(() if fn is None else (fn,), NAME)

    def form_for(self, rows: int) -> Dict[str, Any]:
        """The form of ``rows`` as plain data: the fields of the config :meth:`cfg_for` returns (the pick or the caller's ``cfg``, the
        launch knob applied).  The compile key is the config alone, so nothing else is part of it: the row count and the width are
        runtime facts of a launch (:meth:`launch_shape_for`)."""
        return plain_form(self._fact_cfg(rows))

    def launch_shape_for(self, rows: int) -> LaunchShape:
        """``(grid, block, shared memory)`` of the launch of ``rows``: a flat grid of ``ceil(rows x width / 8 / THREADS)`` thread blocks
        (eight elements per thread: the launcher's own arithmetic, ``blocks_for``), ``THREADS`` threads, no shared memory."""
        cfg = self._fact_cfg(rows)
        return LaunchShape((blocks_for(rows * self.width, cfg.THREADS), 1, 1), (cfg.THREADS, 1, 1), 0)

    def _default_out(self):
        """The entry-owned output buffer ``[160, width]`` E4M3 (see the module docstring for its lifetime)."""
        if self.out_buffer is None:
            import torch

            self.out_buffer = torch.empty(MAX_ROWS, self.width, dtype=torch.float8_e4m3fn, device=self.device)
            self._out_view = None
        return self.out_buffer

    def output(self, rows: int):
        """The entry-owned ``[rows, width]`` E4M3 view a call without ``out`` returns (allocates the buffer on first use)."""
        return self._default_out()[: validate_rows(rows)]

    def _args(self, attn, gate, scale, out, given: bool):
        """The four DSL views of a compile or a launch (one form for both: :func:`_flat`).  ``out`` binds through its ``uint8`` view (a
        true view: one byte per E4M3 element); the entry-owned buffer never moves, so its view is built once."""
        import torch

        if given:
            out_view = _flat(out.view(torch.uint8))
        else:
            if self._out_view is None:
                self._out_view = _flat(out.view(torch.uint8))
            out_view = self._out_view
        return [_flat(attn), _flat(gate), _flat(scale), out_view]

    def compile(self, attn, gate, scale, out=None) -> None:
        """Validate (typed declines, on every call), gate the DSL, trace-compile the kernel's host entry once per config of the entry."""
        rows = check_support(attn, gate, scale, out, device=self.device, width=self.width)
        given = out is not None
        out = out if given else self._default_out()
        missing = [cfg for cfg in self.cfgs if (cfg,) not in self.compiled]
        if not missing:
            return
        import torch

        msg = _dsl_error(torch.cuda.get_device_capability(self.device))
        if msg is not None:
            raise NotImplementedError(msg)
        import cutlass.cute as cute
        from cutlass.cute.runtime import make_fake_stream

        from .kernel import gate_quant_decode_f16 as mod

        with device_context(self.device):  # a compiled launchable belongs to its device's context
            args = self._args(attn, gate, scale, out, given)
            for cfg in missing:
                self.compiled[(cfg,)] = cute.compile(mod.launch, cfg, *args, rows * self.width, make_fake_stream())

    def __call__(self, attn, gate, scale, out=None):
        """Launch on torch's current stream of the entry's device; returns ``out`` -- the caller's tensor as given, or the entry-owned
        buffer sliced to ``[rows, width]`` (E4M3).  Compiles on first use.  Every launch re-runs the binding checks before the compiled
        kernel is looked up.  The launch is scoped to the entry's device (a no-op when that device is the caller's current one) and the
        caller's current device is restored behind it: the compiled kernel belongs to that device's context (Rule 4), and a launch that
        left another device current would move the caller's next default-device work (Rule 5)."""
        import cuda.bindings.driver as cuda
        import torch

        rows = check_binding(attn, gate, scale, out, device=self.device, width=self.width)
        given = out is not None
        out = out if given else self._default_out()
        key = self.key(rows)
        if key not in self.compiled:
            self.compile(attn, gate, scale, out if given else None)
        with device_context(self.device):
            args = self._args(attn, gate, scale, out, given)
            self.compiled[key](*args, rows * self.width, cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream))
        return out if given else out[:rows]
