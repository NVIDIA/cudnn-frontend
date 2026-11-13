# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host entry of the add + RMSNorm + quantize DECODE kernel (``kernel/add_norm_quant_decode_f16.py``): validation with typed declines,
the plan-time compile and the one launch.  No DSL and no torch import at module level -- the CuTe DSL is imported inside
:meth:`AddNormQuantDecode.compile` after the version / target / compiler gates (python/cudnn/AGENTS.md Rule 7), so a host without it
can still construct the entry and run its declines.  The binding checks (:func:`check_binding`: presence, type, device, dtype, shape,
contiguity, alignment and overlap of every operand) run before EVERY launch, compile-cache hits included -- the compile key is the
config alone, and a compiled DSL entry accepts a tensor of another dtype, footprint or device silently, so the host is a launch's
only guard.

The op, per row of an ``[rows, hidden]`` slab (fp32 arithmetic on bf16 operands)::

    v        = x + residual                  (add=True)             residual <- bf16(v), in place
    v        = x                             (add=False)            residual <- x when store_residual=True
    y        = bf16( v * rsqrt(mean(v^2) + eps) * (1 + weight) )    one rounding to bf16 (nearest-even)
    out      = y                                                    (bf16=True)
    codes    = e4m3( y * (1 / scale) )                              (quant="e4m3_static": scale = the consumer's static per-tensor scale)
    codes, scales = nvfp4( y ; global scale )                       (quant="nvfp4": 16-element blocks, E4M3 block scales, 128 x 4 layout)

The row's sum of squares is accumulated in a fixed order that depends on ``hidden`` alone; the codes are those of quantizing ``y``
itself.  A decode-step entry: the residual add, the normalization and the quantization the next projection consumes, in one launch;
for a standalone normalization use the Normalizations graph API.

FORMS.  Three residual modes (``add=True``; ``add=False, store_residual=True``; neither) times five output sets (``bf16`` alone,
``bf16`` with either quant form, either quant form alone): fifteen forms, each one compiled kernel that serves every row count.
``add``, ``bf16`` and ``quant`` are required keywords -- the forms are peers, none is a default.  ``quant`` is ONE vocabulary
(``None`` and ``kernel.add_norm_quant_decode_config.QUANT_FORMS``); the call's return is the same 3-tuple ``(out, codes, scales)`` for
every form, with ``None`` in a slot the form does not produce.

Served (declined otherwise, by name): ``hidden`` 5120; ``rows`` 1 .. 1024 (a RUNTIME kernel argument); ``x`` / ``residual`` /
``weight`` / ``out`` bf16; ``scale`` one fp32 element (0-d or ``[1]``; by contract a finite positive value in ``2**-100 .. 2**100`` --
its value is never read on the host, and outside that range the reciprocal chains overflow or flush); ``codes`` E4M3
``[>= rows, hidden]`` (``"e4m3_static"``) or uint8 ``[>= rows, hidden / 2]`` (``"nvfp4"``: two codes per byte, the even element in the
low nibble); ``scales`` E4M3 ``[>= scale_rows(rows), hidden / 16]`` in the 128 x 4 scale layout over (rows, blocks); every operand
contiguous, 16-byte aligned (``scale``: 4) and on the entry's device; no written operand overlapping another operand; a compute
capability 10.x part; a CuTe DSL >= 4.8.0 that holds the part's target and compiles it (cc 10.7: the CUDA 13 toolchain flavour).
A NaN or infinite element makes its row's ``out`` non-finite and leaves that row's codes unspecified; every other row is unaffected.
A row whose sum of squares overflows fp32 normalizes to zeros.  Nothing is checked on the host (Rule 3).

Launch knobs, default off, neither changes a byte.  ``pdl=True``: the launch carries the programmatic-stream-serialization attribute
and the kernel waits for its stream predecessor AFTER it has loaded ``weight`` and ``scale`` and BEFORE it reads ``x`` / ``residual``
or writes anything.  CALLER OBLIGATION: ``weight`` and ``scale`` are not written by any launch that may still be running when this
one starts (they are constants of the layer).  ``pdl_release`` (``None`` follows ``pdl``; ``True`` needs ``pdl=True``): the kernel
releases its dependents directly after its OWN wait -- on this entry the release never precedes the wait.  CALLER OBLIGATION: every
launch that can start before this one completes waits for it before reading its outputs or writing any operand of this call.
Whether the release helps depends on the row count (a released consumer occupies SMs): a caller that wants it for some row counts
only builds two entries.  The launch runs on ``torch.cuda.current_stream`` of the entry's device, under that device, and the
caller's current device is restored behind it (Rule 5).

Output ownership is PER OUTPUT.  An output the caller passes is the caller's and is returned as given (never re-sliced: rows past
``rows`` are untouched).  An output the caller does not pass is the entry's: one set per row count holding only the form's outputs,
``scales`` zero-filled, allocated by ``compile(rows)`` / ``outputs(rows)`` or at the first call of that row count -- call
``compile(rows)`` for every row count before a stream capture.  Entry-owned outputs are valid until the entry's next call at that row
count from ANY host thread (stream order does not serialize host calls): two host threads that share one entry pass their own
outputs or build one entry each; a caller that sweeps many row counts passes its own outputs (the per-row-count sets are never
freed).  THE KERNEL WRITES THE LIVE ROWS' SCALE BYTES ONLY: a caller-owned ``scales`` buffer serves one row count, or its owner
clears it when the row count shrinks (after a call with more rows, a call with fewer leaves the earlier bytes in the rows between
``rows`` and ``scale_rows(rows)``).  Workspace: none.  The warm path allocates nothing and reads no device memory to the host.
"""

from typing import Any, Dict, Optional, Tuple

from cudnn._torch_stream import device_context  # imports no framework at module level
from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error, cutedsl_state
from cudnn.frost.launch_facts import LaunchShape, kernel_symbols, plain_form

from .kernel.add_norm_quant_decode_config import (
    MAX_ROWS,
    NAME,
    QUANT_FORMS,
    SERVED_HIDDEN,
    SF_BLOCK,
    Cfg,
    scale_rows,
    validate_cfg,
    validate_eps,
    validate_hidden,
    validate_rows,
)

#: the served compute capabilities as closed ranges of ``10 * major + minor``: SM100-class parts only.
SUPPORTED_CC_RANGES = ((100, 109),)
SUPPORTED_CC_TEXT = "10.x (SM100-class)"
DSL_FLOOR = (4, 8, 0)
DSL_FLOOR_TEXT = ".".join(str(x) for x in DSL_FLOOR)
#: cc 10.7 only: the oldest DSL compiler observed to assemble ``sm_107a`` (the CUDA 13 flavour of DSL 4.8.0); CUDA 12.9 rejects the target.
SM107_MIN_COMPILER = (13, 4)
ALIGN_BYTES = 16  # the 128-bit loads / stores of the bf16 operands; every multi-element view declares it
SCALE_ALIGN_BYTES = 4
#: the three output slots of a call, in the order of the returned tuple
OUTPUT_SLOTS = ("out", "codes", "scales")


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
    """The Rule-7 compiler gate.  On cc 10.0 / 10.3 the kernel's PTX is the same text under both toolchain flavours of DSL 4.8.0, so
    both are served.  cc 10.7 is not: the CUDA 12.9 flavour rejects the ``sm_107a`` target at compile time although the DSL's target
    list names it, so a compiler below ``SM107_MIN_COMPILER`` (the one observed to assemble it) is declined here, before JIT, naming
    what was found."""
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


def _knob(name: str, value, optional: bool = False):
    """A construction flag as given when it is a ``bool`` (or ``None`` for an optional launch knob); ``TypeError`` otherwise."""
    if (optional and value is None) or isinstance(value, bool):
        return value
    raise TypeError(f"{NAME}: {name} must be a bool{' or None' if optional else ''}, got {type(value).__name__}")


def make_cfg(hidden, eps, *, add, bf16, quant, store_residual=False, pdl=None, pdl_release=None) -> Cfg:
    """The validated config of the entry's constructor arguments (the construction declines, without torch): ``TypeError`` for a
    ``hidden`` that is not an ``int`` or a flag that is not a ``bool``; ``ValueError`` for a width outside the kernel's structure, an
    ``eps`` that is not finite and positive, ``add`` with ``store_residual``, a ``quant`` outside the vocabulary, a form without an
    output, or ``pdl_release=True`` without ``pdl=True``; ``NotImplementedError`` for a structurally valid width the entry does not
    serve.  ``pdl`` ``None`` is off; ``pdl_release`` ``None`` follows ``pdl``."""
    hidden = validate_hidden(hidden)
    if hidden not in SERVED_HIDDEN:
        raise NotImplementedError(f"{NAME} serves hidden {', '.join(str(h) for h in SERVED_HIDDEN)} (validated); got hidden {hidden}")
    eps = validate_eps(eps)
    add, bf16, store_residual = _knob("add", add), _knob("bf16", bf16), _knob("store_residual", store_residual)
    pdl, pdl_release = _knob("pdl", pdl, optional=True), _knob("pdl_release", pdl_release, optional=True)
    wait = bool(pdl)
    return validate_cfg(
        Cfg(
            HIDDEN=hidden,
            EPS=eps,
            ADD=add,
            STORE_RESIDUAL=store_residual,
            BF16_OUT=bf16,
            QUANT=quant,
            PDL=wait,
            PDL_RELEASE=wait if pdl_release is None else pdl_release,
        )
    )


def produced_slots(cfg: Cfg) -> Tuple[str, ...]:
    """The output slots the form writes, in the order of the returned tuple: ``out`` under ``BF16_OUT``, ``codes`` under a quant
    form, ``scales`` under ``"nvfp4"``."""
    writes = dict(out=cfg.BF16_OUT, codes=cfg.QUANT is not None, scales=cfg.QUANT == "nvfp4")
    return tuple(name for name in OUTPUT_SLOTS if writes[name])


def _byte_range(t, elements: int) -> Tuple[int, int]:
    start = t.data_ptr()
    return start, start + elements * t.element_size()


def check_binding(cfg: Cfg, x, residual, weight, scale, out=None, codes=None, scales=None, device=None) -> int:
    """The binding checks of a call, run before every launch (compile-cache hits included).  ``ValueError`` for an operand the form
    requires and did not get, or got and does not use (both directions), an inconsistent device, shape or alignment, or a written
    operand that overlaps another operand; ``TypeError`` for a non-tensor or a dtype; ``NotImplementedError`` for a row count above the
    served ceiling or a non-contiguous operand.  Returns ``rows = x.shape[0]``.  ``device`` (an entry's: a concrete CUDA ordinal)
    requires every operand on exactly that device; without it the operands must share ``x``'s.  Host-only metadata arithmetic, no
    device work: ``scale``'s VALUE is never read."""
    import torch

    hidden = cfg.HIDDEN
    uses_residual = cfg.ADD or cfg.STORE_RESIDUAL
    if uses_residual and residual is None:
        raise ValueError(f"{NAME}: residual is required by this form (add / store_residual)")
    if not uses_residual and residual is not None:
        raise ValueError(f"{NAME}: residual was given but the entry was built without add or store_residual; pass None")
    if cfg.QUANT is not None and scale is None:
        raise ValueError(f"{NAME}: scale is required by quant={cfg.QUANT!r}")
    if cfg.QUANT is None and scale is not None:
        raise ValueError(f"{NAME}: scale was given but the entry writes no codes; pass None")
    produced = produced_slots(cfg)
    for name, t in (("out", out), ("codes", codes), ("scales", scales)):
        if t is not None and name not in produced:
            raise ValueError(f"{NAME}: {name} was given but the form does not write it")
    tensors = {
        name: t
        for name, t in (("x", x), ("residual", residual), ("weight", weight), ("scale", scale), ("out", out), ("codes", codes), ("scales", scales))
        if t is not None
    }
    for name in ("x", "weight"):
        if name not in tensors:
            raise TypeError(f"{NAME}: {name} must be a torch.Tensor, got NoneType")
    for name, t in tensors.items():
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{NAME}: {name} must be a torch.Tensor, got {type(t).__name__}")
    want = resolve_device(device) if device is not None else x.device
    whose = "the entry's" if device is not None else "x's"
    for name, t in tensors.items():
        if t.device != want:
            raise ValueError(f"{NAME}: every operand must live on one device, {whose} ({want}); {name} is on {t.device}")
    codes_dtype = torch.uint8 if cfg.QUANT == "nvfp4" else torch.float8_e4m3fn
    dtypes = dict(
        x=torch.bfloat16, residual=torch.bfloat16, weight=torch.bfloat16, scale=torch.float32, out=torch.bfloat16, codes=codes_dtype, scales=torch.float8_e4m3fn
    )
    for name, t in tensors.items():
        if t.dtype != dtypes[name]:
            raise TypeError(f"{NAME}: {name} must be {dtypes[name]} (kernel-native; callers convert), got {t.dtype}")
    if x.dim() != 2 or x.shape[1] != hidden or x.shape[0] < 1:
        raise ValueError(f"{NAME}: x must be [rows, {hidden}] with at least one row; got {tuple(x.shape)}")
    rows = int(x.shape[0])
    if rows > MAX_ROWS:
        raise NotImplementedError(f"{NAME} serves at most {MAX_ROWS} rows per call; got {rows}")
    if tuple(weight.shape) not in ((hidden,), (1, hidden)):
        raise ValueError(f"{NAME}: weight must be [{hidden}] or [1, {hidden}]; got {tuple(weight.shape)}")
    if scale is not None and scale.numel() != 1:
        raise ValueError(f"{NAME}: scale must be one fp32 element (0-d or [1]); got {tuple(scale.shape)}")
    codes_width = hidden // 2 if cfg.QUANT == "nvfp4" else hidden
    slabs = dict(residual=(rows, hidden), out=(rows, hidden), codes=(rows, codes_width), scales=(scale_rows(rows), hidden // SF_BLOCK))
    for name, (need_rows, width) in slabs.items():
        t = tensors.get(name)
        if t is not None and (t.dim() != 2 or t.shape[0] < need_rows or t.shape[1] != width):
            raise ValueError(f"{NAME}: {name} must be [>= {need_rows}, {width}]; got {tuple(t.shape)}")
    for name, t in tensors.items():
        if not t.is_contiguous():
            raise NotImplementedError(
                f"{NAME} needs {name} contiguous (the kernel addresses rows by flat element; no copy is made); got strides {tuple(t.stride())}"
            )
    for name, t in tensors.items():
        align = SCALE_ALIGN_BYTES if name == "scale" else ALIGN_BYTES
        if t.data_ptr() % align:
            what = "its 32-bit load" if name == "scale" else "its 128-bit loads / stores"
            raise ValueError(f"{NAME}: {name} must start at a {align}-byte aligned address for {what}, got data_ptr % {align} = {t.data_ptr() % align}")
    # the bytes a launch touches of each operand: the live rows of a slab (the whole padded scale tiles), the weight row, the one scale
    touched = dict(x=rows * hidden, residual=rows * hidden, weight=hidden, scale=1, out=rows * hidden, codes=rows * codes_width)
    touched["scales"] = scale_rows(rows) * (hidden // SF_BLOCK)
    ranges = {name: _byte_range(t, touched[name]) for name, t in tensors.items()}
    for name in ("residual", "out", "codes", "scales"):
        if name not in ranges:
            continue
        a0, a1 = ranges[name]
        for other, (b0, b1) in ranges.items():
            if other != name and a0 < b1 and b0 < a1:
                raise ValueError(f"{NAME}: {name} must not overlap {other}")
    return rows


def check_support(cfg: Cfg, x, residual, weight, scale, out=None, codes=None, scales=None, device_cc=None, device=None) -> int:
    """Typed declines without a DSL import: the binding checks of :func:`check_binding` plus the architecture gate
    (``NotImplementedError`` for operands off a CUDA device or a part this kernel does not serve).  ``device_cc`` overrides the
    compute-capability probe (tests).  The DSL gates (installed, version floor, the part's target, its compiler) run in
    :meth:`AddNormQuantDecode.compile`."""
    import torch

    rows = check_binding(validate_cfg(cfg), x, residual, weight, scale, out, codes, scales, device=device)
    dev = x.device
    if device_cc is None and dev.type != "cuda":
        raise NotImplementedError(f"{NAME} serves CUDA tensors only; the operands are on {dev}")
    cc = tuple(torch.cuda.get_device_capability(dev)) if device_cc is None else tuple(device_cc)
    if not cc_supported(cc):
        raise NotImplementedError(f"{NAME} serves compute capability {SUPPORTED_CC_TEXT}; this device is {cc[0]}.{cc[1]}")
    return rows


def _flat(t):
    """A 1-D DSL view covering a contiguous tensor's whole footprint.  A multi-element operand (16-byte aligned) declares a DYNAMIC
    extent with the unit stride (Rule 4: the row count is a runtime fact, so the kernel is compiled for a runtime extent and a launch
    at another row count re-binds it); the one-element ``scale`` (4-byte aligned) stays static.  The kernel addresses by flat element
    and takes the row count as its own runtime argument: it never reads a view's extent."""
    from cutlass.cute.runtime import from_dlpack

    n = t.numel()
    view = from_dlpack(t.detach().as_strided((n,), (1,)), assumed_align=ALIGN_BYTES if n > 1 else SCALE_ALIGN_BYTES)
    return view.mark_layout_dynamic() if n > 1 else view


class AddNormQuantDecode:
    """The step in ONE form, compiled once and shared by every layer and row count: all tensors are launch arguments, the row count
    is a runtime argument.  ``hidden`` is the row width; ``eps`` the norm's epsilon (a compile-time constant); ``add`` /
    ``store_residual`` the residual mode; ``bf16`` / ``quant`` the output set; ``pdl`` / ``pdl_release`` the launch knobs (the module
    docstring).  The device is resolved to a concrete CUDA ordinal at construction, and every operand of every launch must live on
    exactly it.  An entry built for a non-CUDA device (host tests) runs every decline; its compile raises the typed decline of the
    device.  What a row count launches is readable without a launch: :meth:`kernel_symbols_for`, :meth:`form_for` and
    :meth:`launch_shape_for` (the launch facts of :mod:`cudnn.frost.launch_facts`)."""

    def __init__(
        self,
        device,
        hidden: int,
        eps: float,
        *,
        add: bool,
        bf16: bool,
        quant: Optional[str],
        store_residual: bool = False,
        pdl: Optional[bool] = None,
        pdl_release: Optional[bool] = None,
    ):
        self.cfg: Cfg = make_cfg(hidden, eps, add=add, bf16=bf16, quant=quant, store_residual=store_residual, pdl=pdl, pdl_release=pdl_release)
        self.device = resolve_device(device)
        self.compiled: Dict[tuple, Any] = {}
        self.buffers: Dict[int, Dict[str, Any]] = {}
        self._buffer_views: Dict[int, Dict[str, Any]] = {}

    # ---- the form and the compile key ----
    def cfg_for(self, rows: int) -> Cfg:
        """The config a call of ``rows`` rows runs: the entry's ONE config at every served row count (a row count outside
        1 .. 1024 raises its typed decline)."""
        validate_rows(rows)
        return self.cfg

    def key(self) -> tuple:
        """The compile key: plan-time facts only -- the config.  The row count never enters the compiled kernel: the operands bind
        with a dynamic extent (:func:`_flat`) and the row count is a runtime argument."""
        return (self.cfg,)

    # ---- launch facts (cudnn.frost.launch_facts): three read-only answers per served row count ----
    def _fact_cfg(self, rows) -> Cfg:
        """The config of the row count a launch fact is asked for: ``ValueError`` for a ``rows`` that is not an ``int`` (the decline
        every entry's launch facts give a malformed row count), then the typed declines of :meth:`cfg_for`."""
        if isinstance(rows, bool) or not isinstance(rows, int):
            raise ValueError(f"{NAME}: rows must be an int, got {rows!r}")
        return self.cfg_for(rows)

    def kernel_symbols_for(self, rows: int) -> Tuple[str, ...]:
        """The symbol of the kernel this entry has compiled: a one-element tuple, empty until the entry is compiled.  One compiled
        kernel serves every row count.  The symbol carries the form (``str`` of the config: the width, the residual mode, the
        outputs, the launch arm, the fp32 bits of ``eps``) directly behind the kernel's name, so two forms never share a symbol."""
        self._fact_cfg(rows)
        fn = self.compiled.get(self.key())
        return kernel_symbols(() if fn is None else (fn,), NAME)

    def form_for(self, rows: int) -> Dict[str, Any]:
        """The form of ``rows`` as plain data: the fields of the entry's config.  The compile key is the config alone, so nothing
        else is part of it: the row count is a runtime fact of a launch (:meth:`launch_shape_for`)."""
        return plain_form(self._fact_cfg(rows))

    def launch_shape_for(self, rows: int) -> LaunchShape:
        """``(grid, block, shared memory)`` of the launch of ``rows``: one thread block per row, ``hidden / 8`` threads, and the
        kernel's one shared-memory buffer (an fp32 slot per warp)."""
        cfg = self._fact_cfg(rows)
        return LaunchShape((int(rows), 1, 1), (cfg.THREADS, 1, 1), cfg.SMEM_BYTES)

    # ---- entry-owned outputs ----
    def outputs(self, rows: int) -> Dict[str, Any]:
        """The entry-owned output buffers of ``rows``: the form's subset of ``out`` (bf16 ``[rows, hidden]``), ``codes`` (E4M3
        ``[rows, hidden]`` or uint8 ``[rows, hidden / 2]``) and ``scales`` (E4M3 ``[scale_rows(rows), hidden / 16]``, zero-filled).
        Allocated once per row count; see the module docstring for their lifetime."""
        rows = validate_rows(rows)
        if rows not in self.buffers:
            import torch

            cfg, dev, bufs = self.cfg, self.device, {}
            if cfg.BF16_OUT:
                bufs["out"] = torch.empty(rows, cfg.HIDDEN, dtype=torch.bfloat16, device=dev)
            if cfg.QUANT == "e4m3_static":
                bufs["codes"] = torch.empty(rows, cfg.HIDDEN, dtype=torch.float8_e4m3fn, device=dev)
            if cfg.QUANT == "nvfp4":
                bufs["codes"] = torch.empty(rows, cfg.HIDDEN // 2, dtype=torch.uint8, device=dev)
                bufs["scales"] = torch.zeros(scale_rows(rows), cfg.HIDDEN // SF_BLOCK, dtype=torch.float8_e4m3fn, device=dev)
            self.buffers[rows] = bufs
        return self.buffers[rows]

    def _output_view(self, name: str, t):
        """The DSL view of an output: bf16 as it is, the code and scale bytes through their ``uint8`` view (one byte per element)."""
        import torch

        return _flat(t if t.dtype == torch.bfloat16 else t.view(torch.uint8))

    def _owned_views(self, rows: int) -> Dict[str, Any]:
        """The DSL views of the entry-owned outputs of ``rows``: the buffers never move, so each view is built once."""
        if rows not in self._buffer_views:
            self._buffer_views[rows] = {name: self._output_view(name, t) for name, t in self.outputs(rows).items()}
        return self._buffer_views[rows]

    def _args(self, x, residual, weight, scale, views: Dict[str, Any]):
        """The seven kernel operands of a compile or a launch: ``None`` exactly where the form has no such operand (the kernel drops
        a dead operand from its specialized signature; nothing is bound to a placeholder)."""
        return [
            _flat(x),
            None if residual is None else _flat(residual),
            _flat(weight),
            None if scale is None else _flat(scale),
            views.get("out"),
            views.get("codes"),
            views.get("scales"),
        ]

    # ---- compile ----
    def compile(self, rows: Optional[int] = None) -> None:
        """Plan time: the typed declines of the device, the part and the DSL, then ONE trace-compile of the entry's form (a no-op
        once compiled).  With ``rows`` it also allocates the entry-owned outputs of that row count -- call it for every row count
        before a stream capture.  Needs no operand."""
        if rows is not None:
            validate_rows(rows)
        if self.key() not in self.compiled:
            import torch

            if self.device.type != "cuda":
                raise NotImplementedError(f"{NAME} serves CUDA devices only; the entry's device is {self.device}")
            cc = tuple(torch.cuda.get_device_capability(self.device))
            if not cc_supported(cc):
                raise NotImplementedError(f"{NAME} serves compute capability {SUPPORTED_CC_TEXT}; this device is {cc[0]}.{cc[1]}")
            msg = _dsl_error(cc)
            if msg is not None:
                raise NotImplementedError(msg)
            import cutlass
            import cutlass.cute as cute
            from cutlass.cute.runtime import make_fake_stream

            from .kernel import add_norm_quant_decode_f16 as mod

            cfg = self.cfg
            with device_context(self.device):  # a compiled launchable belongs to its device's context
                # plan-time operands of one row: the views are layout-dynamic, so no extent of them enters the kernel
                x = torch.zeros(1, cfg.HIDDEN, dtype=torch.bfloat16, device=self.device)
                residual = torch.zeros(1, cfg.HIDDEN, dtype=torch.bfloat16, device=self.device) if (cfg.ADD or cfg.STORE_RESIDUAL) else None
                weight = torch.zeros(cfg.HIDDEN, dtype=torch.bfloat16, device=self.device)
                scale = torch.ones(1, dtype=torch.float32, device=self.device) if cfg.QUANT is not None else None
                bind_rows = 1 if rows is None else rows
                args = self._args(x, residual, weight, scale, self._owned_views(bind_rows))
                self.compiled[self.key()] = cute.compile(mod.launch, cfg, *args, cutlass.Int32(bind_rows), make_fake_stream())
        if rows is not None:
            self.outputs(rows)

    # ---- launch ----
    def __call__(self, x, residual, weight, scale, out=None, codes=None, scales=None):
        """Launch on torch's current stream of the entry's device; returns ``(out, codes, scales)`` -- per slot the caller's tensor
        as given, else the entry-owned buffer of this row count, else ``None`` for a slot the form does not produce.  Compiles on
        first use.  Every launch re-runs the binding checks before the compiled kernel is looked up.  The launch is scoped to the
        entry's device and the caller's current device is restored behind it."""
        import torch

        cfg = self.cfg
        rows = check_binding(cfg, x, residual, weight, scale, out, codes, scales, device=self.device)
        given = dict(out=out, codes=codes, scales=scales)
        key = self.key()
        if key not in self.compiled:
            self.compile()
        import cuda.bindings.driver as cuda  # behind compile(): its gates decline a host without the DSL by type first
        import cutlass

        with device_context(self.device):
            results, views = {}, {}
            owned = owned_views = None
            for name in produced_slots(cfg):
                if given[name] is not None:
                    results[name] = given[name]
                    views[name] = self._output_view(name, given[name])
                else:
                    if owned is None:
                        owned, owned_views = self.outputs(rows), self._owned_views(rows)
                    results[name] = owned[name]
                    views[name] = owned_views[name]
            args = self._args(x, residual, weight, scale, views)
            self.compiled[key](*args, cutlass.Int32(rows), cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream))
        return results.get("out"), results.get("codes"), results.get("scales")


__all__ = ["AddNormQuantDecode", "check_binding", "check_support", "cc_supported", "scale_rows", "make_cfg", "QUANT_FORMS"]
