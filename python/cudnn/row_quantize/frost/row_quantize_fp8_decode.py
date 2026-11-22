# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host entry of the FP8 row quantization DECODE kernel (``kernel/row_quantize_fp8_decode_f16.py``): validation with typed
declines, the plan-time compile and the one launch.  No DSL and no torch import at module level -- the CuTe DSL is imported inside
:meth:`RowQuantizeFp8Decode.compile` after the version / target / compiler gates (python/cudnn/AGENTS.md Rule 7), so a host
without it can still construct the entry and run its declines.  The binding checks (:func:`check_binding`: type, device, dtype,
shape, strides, alignment and overlap of every operand) run before EVERY launch, compile-cache hits included -- the compile key
is the config alone and the kernel takes its operands as addresses, so the host is a launch's only guard.  The overlap check
covers the outputs the call does NOT pass too: the entry's own buffers of the row count are operands of the launch like the
passed ones.

The op, per row of an ``[rows, width]`` bf16 slab (fp32 arithmetic on the bf16 values)::

    amax    = max |x_j|                                  over the row
    scale   = max( amax / 448, fp32(1 / (448 * 512)) )   one correctly rounded fp32 division; the floor is the bits 0x36924925
    inv     = 1 / scale                                  one correctly rounded fp32 division
    code_j  = e4m3( clamp(x_j * inv, -448, 448) )        one fp32 multiplication, then nearest-even to E4M3, saturating

A decode-step entry: the dynamic FP8 quantization of the rows an FP8 multiply reads next -- each row under its own scale -- in
one launch.  The consumer reads ``code * scale``.  Every step is exactly specified, so a row's bytes are a function of its values
alone, on every served part: a zero row has the floor scale and the codes of its signed zeros; the elements that hold the row
maximum have the codes of +-448 once the maximum is above ``2**-9`` (the floor's knee).

Served (declined otherwise, by name): ``width`` 17408; ``rows`` 1 .. 1024 (a RUNTIME kernel argument); ``x`` bf16
``[rows, width]``, contiguous along its last axis, its row stride a multiple of 8 elements (whole 16-byte vectors: the rows of a
wider slab bind as they are); ``codes`` E4M3 ``[>= rows, width]`` and ``scales`` fp32 ``[>= rows, 1]``, both contiguous; ``x``
16-byte aligned (its 128-bit loads), ``codes`` 8-byte (its 64-bit stores), ``scales`` 4-byte; every operand on the entry's device;
no written operand overlapping another operand; a compute capability 10.x part; a CuTe DSL >= 4.8.0 that holds the part's target
and compiles it (cc 10.7: the CUDA 13 toolchain flavour).  A row that holds a NaN or an infinity leaves THAT ROW's codes and
scale unspecified; every other row is unaffected.  Nothing is checked on the host (Rule 3).

Launch knobs, default off, neither changes a byte.  ``pdl=True``: the launch carries the programmatic-stream-serialization
attribute and the kernel waits for its stream predecessor BEFORE it reads or writes anything.  ``pdl_release`` (``None`` follows
``pdl``; ``True`` needs ``pdl=True``): the kernel releases its dependents as its FIRST statement, in front of its own wait.
CALLER OBLIGATION: every launch that can start before this one completes waits for it before reading ``codes`` / ``scales`` and
before writing any operand of this call.  The launch runs on ``torch.cuda.current_stream`` of the entry's device, under that
device, and the caller's current device is restored behind it (Rule 5).

Output ownership is PER OUTPUT.  An output the caller passes is the caller's and is returned as given (rows past the call's are
untouched).  An output the caller does not pass is the entry's: one pair per row count, allocated by ``compile(rows)`` /
``outputs(rows)`` or at the first call of that row count -- call ``compile(rows)`` for every row count before a stream capture.
The pairs are never freed (a pair of ``r`` rows holds ``r x (width + 4)`` bytes): a caller that sweeps many row counts passes its
own outputs.  THE ALLOCATION OF AN ENTRY-OWNED PAIR IS STREAM WORK of torch's current stream of the entry's device at that
moment: the pair is not initialized by the entry, but under deterministic algorithms torch fills it, and its blocks can be
recycled ones that stream still owes a write.  EVERY launch that writes bytes of the pair is ordered behind that work on WHATEVER
stream it runs, HOWEVER the tensor reached the call -- the entry's own output of the row count, or a tensor taken with
``outputs`` and passed back, whole or as a view, at this row count or another, by this entry or by another entry that orders its
launches the same way (``cudnn.frost.init_order``; the test is one of addresses).  The launch stream waits for the event recorded
behind the allocation -- a stream-side wait: no host wait, never a synchronization of a stream or of the device -- so
``compile(rows)`` may run on any stream.  A captured launch cannot carry that dependency: a call that finds the work still
pending under a capture waits for that ONE event on the host before its captured launch (set-up, never part of a replay; a pair
allocated before the capture needs no wait once its stream's earlier work is done).  NOT ordered: a pair allocated INSIDE a stream
capture (nothing is recorded: allocate before the capture).  WHAT STAYS THE CALLER'S: ``x``, which a launch only reads, and the
caller's own reads and writes of a returned tensor outside a call of an entry.  Entry-owned outputs are valid until the entry's
next call at that row count from ANY host thread (stream order does not serialize host calls: two host threads that share one
entry pass their own outputs or build one entry each).  A returned entry-owned output cannot come back as ANOTHER operand of a
call that writes it by default: such a call is declined (``ValueError``) before the launch.  Workspace: none.  The warm path
allocates nothing and reads no device memory to the host; the one host wait of the entry is the one above.
"""

from typing import Any, Dict, Optional, Tuple

from cudnn._torch_stream import device_context  # imports no framework at module level
from cudnn.frost import init_order  # imports no framework at module level
from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error, cutedsl_state
from cudnn.frost.launch_facts import LaunchShape, kernel_symbols, plain_form

from .kernel.row_quantize_fp8_decode_config import ELEMENT_BYTES, MAX_ROWS, NAME, SERVED_WIDTH, VEC, Cfg, validate_cfg, validate_rows, validate_width

#: the served compute capabilities as closed ranges of ``10 * major + minor``: SM100-class parts only.
SUPPORTED_CC_RANGES = ((100, 109),)
SUPPORTED_CC_TEXT = "10.x (SM100-class)"
DSL_FLOOR = (4, 8, 0)
DSL_FLOOR_TEXT = ".".join(str(x) for x in DSL_FLOOR)
#: cc 10.7 only: the oldest DSL compiler observed to assemble ``sm_107a`` (the CUDA 13 flavour of DSL 4.8.0); CUDA 12.9 rejects the target.
SM107_MIN_COMPILER = (13, 4)
#: the alignment of each operand's start address, in bytes: the widest access the kernel makes of it
ALIGN_BYTES = dict(x=VEC * ELEMENT_BYTES, codes=VEC, scales=4)
#: the two output slots of a call, in the order of the returned tuple
OUTPUT_SLOTS = ("codes", "scales")


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
    """The Rule-7 target gate: a version floor does not guarantee the part's target (cc 10.7 needs a DSL build whose target list
    holds ``sm_107a``).  The first gate that imports ``cutlass.base_dsl`` -- call it after :func:`_dsl_version_error` returned None."""
    msg = cutedsl_arch_requirement_error(tuple(device_cc))
    return None if msg is None else f"{NAME}: {msg}"


def dsl_compiler_cuda_version() -> Tuple[int, int]:
    """``(major, minor)`` of the CUDA compiler the installed CuTe DSL bound (its CUDA 12 or CUDA 13 toolchain flavour), from
    ``cutlass.base_dsl.version_info.CUDA_VERSION`` -- the ONE seam the compiler gate reads (tests patch it).  Never
    ``torch.version.cuda`` or ``CUDA_HOME``: neither names the compiler the DSL wheel selected."""
    from cutlass.base_dsl.version_info import CUDA_VERSION

    return (int(CUDA_VERSION.major), int(CUDA_VERSION.minor))


def _dsl_compiler_error(device_cc) -> Optional[str]:
    """The Rule-7 compiler gate: on cc 10.7 the CUDA 12.9 flavour rejects the ``sm_107a`` target at compile time although the DSL's
    target list names it, so a compiler below ``SM107_MIN_COMPILER`` is declined here, before JIT, naming what was found."""
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
    """The DSL gates in order -- installed, the version floor, the part's target, the compiler that assembles it; None when the
    installed DSL serves the part.  Only the last two import the DSL (``cutlass.base_dsl``), after the version gate."""
    msg = _dsl_version_error()
    if msg is None:
        msg = _dsl_target_error(device_cc)
    if msg is None:
        msg = _dsl_compiler_error(device_cc)
    return msg


def resolve_device(device):
    """``device`` as a concrete ``torch.device``: a CUDA device given without an index is the CURRENT one, so it compares with an
    operand's device (always indexed for a CUDA tensor) by type AND index; any other device is kept as given."""
    import torch

    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def _knob(name: str, value):
    """A launch knob as given when it is a ``bool`` or ``None``; ``TypeError`` otherwise."""
    if value is None or isinstance(value, bool):
        return value
    raise TypeError(f"{NAME}: {name} must be a bool or None, got {type(value).__name__}")


def make_cfg(width, *, pdl=None, pdl_release=None) -> Cfg:
    """The validated config of the entry's constructor arguments (the construction declines, without torch): ``TypeError`` for a
    ``width`` that is not an ``int`` or a launch knob that is not a ``bool`` / ``None``; ``ValueError`` for a width outside the
    kernel's structure or ``pdl_release=True`` without ``pdl=True``; ``NotImplementedError`` for a structurally valid width the
    entry does not serve.  ``pdl`` ``None`` is off; ``pdl_release`` ``None`` follows ``pdl``."""
    width = validate_width(width)
    if width not in SERVED_WIDTH:
        raise NotImplementedError(f"{NAME} serves width {', '.join(str(w) for w in SERVED_WIDTH)} (validated); got width {width}")
    pdl, pdl_release = _knob("pdl", pdl), _knob("pdl_release", pdl_release)
    wait = bool(pdl)
    return validate_cfg(Cfg(WIDTH=width, PDL=wait, PDL_RELEASE=wait if pdl_release is None else pdl_release))


def _touched(cfg: Cfg, rows: int, tensors: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
    """The half-open byte range a launch of ``rows`` rows touches of each operand in ``tensors`` (by operand name): the bounds of
    the ``x`` view (its row stride included), the live rows of an output."""
    elements = dict(x=None, codes=rows * cfg.WIDTH, scales=rows)
    return {name: init_order.span(t, elements[name]) for name, t in tensors.items()}


def _check_overlap(cfg: Cfg, rows: int, tensors: Dict[str, Any], written, what: str = "") -> None:
    ranges = _touched(cfg, rows, tensors)
    for name in written:
        if name not in ranges:
            continue
        a0, a1 = ranges[name]
        for other, (b0, b1) in ranges.items():
            if other != name and a0 < b1 and b0 < a1:
                raise ValueError(f"{NAME}: {what}{name} must not overlap {other}")


def check_binding(cfg: Cfg, x, codes=None, scales=None, device=None) -> int:
    """The binding checks of a call, run before every launch (compile-cache hits included).  ``TypeError`` for a non-tensor or a
    dtype; ``ValueError`` for an inconsistent device, shape, row stride or alignment, zero rows, or a written operand that
    overlaps another operand; ``NotImplementedError`` for a row count above the served ceiling or an operand that is not
    contiguous where the kernel addresses flat.  Returns ``rows = x.shape[0]``.  ``device`` (an entry's: a concrete CUDA ordinal)
    requires every operand on exactly that device; without it the operands must share ``x``'s.  Host-only metadata arithmetic, no
    device work.  The checks see the operands they are given: an output that is not passed is resolved by the entry, whose call
    checks its own buffers against these operands (:meth:`RowQuantizeFp8Decode.__call__`)."""
    import torch

    width = cfg.WIDTH
    tensors = {name: t for name, t in (("x", x), ("codes", codes), ("scales", scales)) if not (t is None and name in OUTPUT_SLOTS)}
    for name, t in tensors.items():
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{NAME}: {name} must be a torch.Tensor, got {type(t).__name__}")
    want = resolve_device(device) if device is not None else x.device
    whose = "the entry's" if device is not None else "x's"
    for name, t in tensors.items():
        if t.device != want:
            raise ValueError(f"{NAME}: every operand must live on one device, {whose} ({want}); {name} is on {t.device}")
    dtypes = dict(x=torch.bfloat16, codes=torch.float8_e4m3fn, scales=torch.float32)
    for name, t in tensors.items():
        if t.dtype != dtypes[name]:
            raise TypeError(f"{NAME}: {name} must be {dtypes[name]} (kernel-native; callers convert), got {t.dtype}")
    if x.dim() != 2 or x.shape[1] != width or x.shape[0] < 1:
        raise ValueError(f"{NAME}: x must be [rows, {width}] with at least one row; got {tuple(x.shape)}")
    rows = int(x.shape[0])
    if rows > MAX_ROWS:
        raise NotImplementedError(f"{NAME} serves at most {MAX_ROWS} rows per call; got {rows}")
    for name, columns in (("codes", width), ("scales", 1)):
        t = tensors.get(name)
        if t is not None and (t.dim() != 2 or t.shape[0] < rows or t.shape[1] != columns):
            raise ValueError(f"{NAME}: {name} must be [>= {rows}, {columns}]; got {tuple(t.shape)}")
    if int(x.stride(1)) != 1:
        raise NotImplementedError(f"{NAME} needs x contiguous along its last axis (a row is addressed flat; no copy is made); got strides {tuple(x.stride())}")
    if rows > 1 and int(x.stride(0)) % VEC:
        raise ValueError(
            f"{NAME}: x stride[0] = {int(x.stride(0))} elements must be a multiple of {VEC} (a row is whole {VEC * ELEMENT_BYTES}-byte vectors: "
            f"every row starts at a {VEC * ELEMENT_BYTES}-byte aligned address)"
        )
    for name in OUTPUT_SLOTS:
        t = tensors.get(name)
        if t is not None and not t.is_contiguous():
            raise NotImplementedError(
                f"{NAME} needs {name} contiguous (the kernel addresses rows by flat element; no copy is made); got strides {tuple(t.stride())}"
            )
    why = dict(x="for its 128-bit loads", codes="for its 64-bit stores", scales="for its 32-bit stores")
    for name, t in tensors.items():
        align = ALIGN_BYTES[name]
        if t.data_ptr() % align:
            raise ValueError(f"{NAME}: {name} must start at a {align}-byte aligned address {why[name]}, got data_ptr % {align} = {t.data_ptr() % align}")
    _check_overlap(cfg, rows, tensors, OUTPUT_SLOTS)
    return rows


def check_support(cfg: Cfg, x, codes=None, scales=None, device_cc=None, device=None) -> int:
    """Typed declines without a DSL import: the binding checks of :func:`check_binding` plus the architecture gate
    (``NotImplementedError`` for operands off a CUDA device or a part this kernel does not serve).  ``device_cc`` overrides the
    compute-capability probe (tests).  The DSL gates (installed, version floor, the part's target, its compiler) run in
    :meth:`RowQuantizeFp8Decode.compile`."""
    import torch

    rows = check_binding(validate_cfg(cfg), x, codes, scales, device=device)
    dev = x.device
    if device_cc is None and dev.type != "cuda":
        raise NotImplementedError(f"{NAME} serves CUDA tensors only; the operands are on {dev}")
    cc = tuple(torch.cuda.get_device_capability(dev)) if device_cc is None else tuple(device_cc)
    if not cc_supported(cc):
        raise NotImplementedError(f"{NAME} serves compute capability {SUPPORTED_CC_TEXT}; this device is {cc[0]}.{cc[1]}")
    return rows


def launch_arguments(cfg: Cfg, rows: int, x, codes, scales) -> Tuple[int, ...]:
    """The kernel's runtime arguments of a call as plain ints, in launch order: the three addresses with ``x``'s row stride in
    bytes behind ``x`` (``kernel.row_quantize_fp8_decode_f16.OPERANDS``), then the row count.  Call it on checked operands
    (:func:`check_binding`): it reads metadata only."""
    return (x.data_ptr(), int(x.stride(0)) * ELEMENT_BYTES, codes.data_ptr(), scales.data_ptr(), int(rows))


class RowQuantizeFp8Decode:
    """The step in ONE form, compiled once and shared by every row count: all operands are launch arguments, the row count is a
    runtime argument.  ``width`` is the row width; ``pdl`` / ``pdl_release`` the launch knobs (the module docstring).  The device
    is resolved to a concrete CUDA ordinal at construction, and every operand of every launch must live on exactly it.  An entry
    built for a non-CUDA device (host tests) runs every decline; its compile raises the typed decline of the device.  What a row
    count launches is readable without a launch: :meth:`kernel_symbols_for`, :meth:`form_for` and :meth:`launch_shape_for` (the
    launch facts of :mod:`cudnn.frost.launch_facts`)."""

    def __init__(self, device, width: int, *, pdl: Optional[bool] = None, pdl_release: Optional[bool] = None):
        self.cfg: Cfg = make_cfg(width, pdl=pdl, pdl_release=pdl_release)
        self.device = resolve_device(device)
        self.compiled: Dict[tuple, Any] = {}
        self.buffers: Dict[int, Dict[str, Any]] = {}
        #: the device's records of entry-owned allocations whose initialization may be pending (shared by every entry of the device)
        self._pending = init_order.pending_of(self.device)

    # ---- the form and the compile key ----
    def cfg_for(self, rows: int) -> Cfg:
        """The config a call of ``rows`` rows runs: the entry's ONE config at every served row count (a row count outside
        1 .. 1024 raises its typed decline)."""
        validate_rows(rows)
        return self.cfg

    def key(self) -> tuple:
        """The compile key: plan-time facts only -- the config.  No operand enters the compiled kernel: it takes addresses and
        ``x``'s row stride as runtime arguments and the row count as its grid."""
        return (self.cfg,)

    # ---- launch facts (cudnn.frost.launch_facts): three read-only answers per served row count ----
    def _fact_cfg(self, rows) -> Cfg:
        """The config of the row count a launch fact is asked for: ``ValueError`` for a ``rows`` that is not an ``int`` (the
        decline every entry's launch facts give a malformed row count), then the typed declines of :meth:`cfg_for`."""
        if isinstance(rows, bool) or not isinstance(rows, int):
            raise ValueError(f"{NAME}: rows must be an int, got {rows!r}")
        return self.cfg_for(rows)

    def kernel_symbols_for(self, rows: int) -> Tuple[str, ...]:
        """The symbol of the kernel this entry has compiled: a one-element tuple, empty until the entry is compiled.  One
        compiled kernel serves every row count.  The symbol carries the form (``str`` of the config: the width and the launch
        arm) directly behind the kernel's name, so two forms never share a symbol."""
        self._fact_cfg(rows)
        fn = self.compiled.get(self.key())
        return kernel_symbols(() if fn is None else (fn,), NAME)

    def form_for(self, rows: int) -> Dict[str, Any]:
        """The form of ``rows`` as plain data: the fields of the entry's config.  The compile key is the config alone, so
        nothing else is part of it: the row count is a runtime fact of a launch (:meth:`launch_shape_for`)."""
        return plain_form(self._fact_cfg(rows))

    def launch_shape_for(self, rows: int) -> LaunchShape:
        """``(grid, block, shared memory)`` of the launch of ``rows``: one thread block per row, the config's thread count, and
        the kernel's one shared-memory buffer (an fp32 slot per warp)."""
        cfg = self._fact_cfg(rows)
        return LaunchShape((int(rows), 1, 1), (cfg.THREADS, 1, 1), cfg.SMEM_BYTES)

    # ---- entry-owned outputs ----
    def outputs(self, rows: int) -> Dict[str, Any]:
        """The entry-owned output buffers of ``rows``: ``codes`` (E4M3 ``[rows, width]``) and ``scales`` (fp32 ``[rows, 1]``).
        Allocated once per row count; see the module docstring for their lifetime and for what is ordered behind their
        allocation.  Returns a copy of the entry's table (the same tensors), without waiting."""
        return dict(self._owned(rows))

    def _owned(self, rows: int) -> Dict[str, Any]:
        """The entry's own table of the pair of ``rows`` rows, allocated on first use.  The allocation's stream work is recorded
        (:mod:`cudnn.frost.init_order`) BEFORE the pair is in the table."""
        rows = validate_rows(rows)
        if rows not in self.buffers:
            import torch

            bufs = dict(
                codes=torch.empty(rows, self.cfg.WIDTH, dtype=torch.float8_e4m3fn, device=self.device),
                scales=torch.empty(rows, 1, dtype=torch.float32, device=self.device),
            )
            init_order.record(self._pending, bufs.values())
            self.buffers[rows] = bufs
        return self.buffers[rows]

    def _launch_writes(self, rows: int, written: Dict[str, Any]) -> Tuple[Tuple[int, int], ...]:
        """The byte ranges a launch of ``rows`` rows WRITES of ``written`` (the resolved outputs by slot): their live rows --
        what the launch is ordered by (:func:`cudnn.frost.init_order.order_behind`)."""
        return tuple(_touched(self.cfg, rows, written).values())

    def _check_entry_owned(self, rows: int, operands: Dict[str, Any]) -> None:
        """The overlap check of :func:`check_binding` for the outputs a call does not pass: the launch writes the entry's own
        buffers of ``rows`` in their place, and a tensor an earlier call returned can come back as an operand.  ``ValueError``
        when such a buffer overlaps another operand of the call.  Before the row count's buffers exist there is nothing to
        check -- what the call then allocates is fresh memory, disjoint from every live tensor.  Host-only pointer arithmetic;
        allocates nothing."""
        owned = self.buffers.get(rows)
        if owned is None:
            return
        entry_owned = {name: t for name, t in owned.items() if operands[name] is None}
        if not entry_owned:
            return
        passed = {name: t for name, t in operands.items() if t is not None}
        try:
            _check_overlap(self.cfg, rows, {**passed, **entry_owned}, tuple(entry_owned), what="the entry-owned ")
        except ValueError as err:
            raise ValueError(
                f"{err} (the output was not passed, so the launch writes the entry's own buffer of {rows} row(s), the tensor earlier "
                f"calls of that row count returned; pass a caller-owned output)"
            ) from None

    # ---- compile ----
    def compile(self, rows: Optional[int] = None) -> None:
        """Plan time: the typed declines of the device, the part and the DSL, then ONE trace-compile of the entry's form (a
        no-op once compiled).  With ``rows`` it also allocates the entry-owned outputs of that row count -- call it for every
        row count before a stream capture, on any stream: a later launch that writes them is ordered behind the allocation's
        stream work.  Needs no operand and, without ``rows``, allocates nothing."""
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

            from .kernel import row_quantize_fp8_decode_f16 as mod

            with device_context(self.device):  # a compiled launchable belongs to its device's context
                # plan-time arguments: addresses and the stride are runtime values, the zeros pin their TYPES only
                args = [cutlass.Int64(0)] * len(mod.OPERANDS) + [cutlass.Int32(1)]
                self.compiled[self.key()] = cute.compile(mod.launch, self.cfg, *args, make_fake_stream())
        if rows is not None:
            self._owned(rows)

    # ---- launch ----
    def __call__(self, x, codes=None, scales=None):
        """Launch on torch's current stream of the entry's device; returns ``(codes, scales)`` -- per slot the caller's tensor
        as given, else the entry-owned buffer of this row count.  Compiles on first use.  Every launch re-runs the binding
        checks before the compiled kernel is looked up, the overlap check on the RESOLVED operands (``ValueError``: an
        entry-owned output the launch would write must not overlap another operand; a returned output fed back in needs a
        caller-owned output).  The launch is ordered behind the pending stream work of every entry-owned allocation of the
        device it writes (the module docstring), scoped to the entry's device, and the caller's current device is restored
        behind it."""
        import torch

        cfg = self.cfg
        rows = check_binding(cfg, x, codes, scales, device=self.device)
        self._check_entry_owned(rows, dict(x=x, codes=codes, scales=scales))
        key = self.key()
        if key not in self.compiled:
            self.compile()
        import cuda.bindings.driver as cuda  # behind compile(): its gates decline a host without the DSL by type first
        import cutlass

        with device_context(self.device):
            if codes is None or scales is None:
                owned = self._owned(rows)
                codes = owned["codes"] if codes is None else codes
                scales = owned["scales"] if scales is None else scales
            values = launch_arguments(cfg, rows, x, codes, scales)
            stream = torch.cuda.current_stream(self.device)
            if self._pending:  # empty in the steady state: a warm call does nothing here
                init_order.order_behind(self._pending, stream, self._launch_writes(rows, dict(codes=codes, scales=scales)))
            self.compiled[key](*(cutlass.Int64(value) for value in values[:-1]), cutlass.Int32(values[-1]), cuda.CUstream(stream.cuda_stream))
        return codes, scales


__all__ = ["RowQuantizeFp8Decode", "check_binding", "check_support", "cc_supported", "launch_arguments", "make_cfg"]
