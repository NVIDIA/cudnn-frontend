# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host entry of the FP4 row quantization DECODE kernel (``kernel/row_quantize_fp4_decode_f16.py``): validation with typed declines,
the plan-time compile and the one launch.  No DSL and no torch import at module level -- the CuTe DSL is imported inside
:meth:`RowQuantizeFp4Decode.compile` after the version / target / compiler gates (python/cudnn/AGENTS.md Rule 7), so a host without it
can still construct the entry and run its declines.  The binding checks (:func:`check_binding`: type, device, dtype, shape,
contiguity, alignment and overlap of every operand) run before EVERY launch, compile-cache hits included -- the compile key is the
config alone, and a compiled DSL entry accepts a tensor of another dtype, footprint or device silently, so the host is a launch's
only guard.  The overlap check covers the outputs the call does NOT pass too: the entry's own buffers of the row count are operands
of the launch like the passed ones.

The op, per block of sixteen consecutive elements of a row of an ``[rows, width]`` bf16 slab (fp32 arithmetic on bf16 values)::

    amax        = max |x_j|                                 over the block
    scale byte  = e4m3( scale * (amax * fp32(1/6)) )        nearest-even, saturating at 448; ``value`` is the byte's fp32 value
    encode      = 0  when value == 0,  else  rcp( value * rcp(scale) )      two approximate reciprocals
    code_j      = e2m1( x_j * encode )                      nearest-even on {0, 0.5, 1, 1.5, 2, 3, 4, 6}, saturating at +-6, sign kept

A decode-step entry: the quantization of the rows a block-scaled FP4 (NVFP4) multiply reads next -- its codes and its block scales in
that multiply's layout -- in one launch.  ``scale`` is the GLOBAL scale the multiply's operand is quantized under: the fp32
reciprocal of the multiply's static input scale.  The bytes are those of the block-scale convention with a global scale that the
repository implements as ``cudnn.frost.tile_dsl.pointwise.nvfp4_block_scale_global``: the NVFP4 form of ``cudnn.AddNormQuantDecode``
writes the same codes and the same block scales for the same bf16 values.  For the conversion of whole 128-row bands with a caller
workspace and another rounding rule use ``cudnn.ops.nvfp4_block_scale_quantize``.

Served (declined otherwise, by name): ``width`` 5120; ``rows`` 1 .. 1024 (a RUNTIME kernel argument); ``x`` bf16 ``[rows, width]``;
``scale`` one fp32 element (0-d or ``[1]``; by contract a finite positive value in ``2**-100 .. 2**100`` -- its value is never read
on the host, and outside that range the reciprocal chain overflows or flushes); ``codes`` uint8 ``[>= rows, width / 2]`` (two codes
per byte, the even element in the low nibble); ``scales`` E4M3 ``[>= scale_rows(rows), width / 16]`` in the 128 x 4 scale layout
over (rows, blocks); every operand contiguous, 16-byte aligned (``scale``: 4) and on the entry's device; no written operand
overlapping another operand; a compute capability 10.x part; a CuTe DSL >= 4.8.0 that holds the part's target and compiles it
(cc 10.7: the CUDA 13 toolchain flavour).  A block that holds a NaN or an infinity leaves THAT BLOCK's codes and scale unspecified;
every other block of the row and every other row is unaffected.  A zero block has scale byte 0 and the codes of its signed zeros; a
block whose scale rounds to the zero code encodes as signed zeros; a block whose scale saturates encodes its largest elements at
+-6.  Nothing is checked on the host (Rule 3).

Launch knobs, default off, neither changes a byte.  ``pdl=True``: the launch carries the programmatic-stream-serialization attribute
and the kernel waits for its stream predecessor AFTER it has loaded ``scale`` and BEFORE it reads ``x`` or writes anything.  CALLER
OBLIGATION: ``scale`` is not written by any launch that may still be running when this one starts (it is a constant of the layer).
``pdl_release`` (``None`` follows ``pdl``; ``True`` needs ``pdl=True``): the kernel releases its dependents directly after its OWN
wait -- on this entry the release never precedes the wait.  CALLER OBLIGATION: every launch that can start before this one completes
waits for it before reading its outputs or writing any operand of this call.  The launch runs on ``torch.cuda.current_stream`` of
the entry's device, under that device, and the caller's current device is restored behind it (Rule 5).

Output ownership is PER OUTPUT.  An output the caller passes is the caller's and is returned as given (never re-sliced: rows past
``rows`` are untouched).  An output the caller does not pass is the entry's: one pair per row count, ``scales`` zero-filled,
allocated by ``compile(rows)`` / ``outputs(rows)`` or at the first call of that row count -- call ``compile(rows)`` for every row
count before a stream capture.  THE INITIALIZATION OF A PAIR IS STREAM WORK (the allocation and the zero-fill, on torch's current
stream of the entry's device at that moment), and a call orders itself behind it on WHATEVER stream it launches: the same stream by
stream order, another stream by a stream-side wait for the event recorded behind the initialization -- no host wait, never a
synchronization of a stream or of the device -- so ``compile(rows)`` may run on any stream.  A captured call cannot carry that
dependency: one that finds the initialization still pending waits for that ONE event on the host before its captured launch.  Once
the initialization is seen complete a call does nothing for it.  ``outputs(rows)`` returns the pair without waiting: see its
contract for a caller that touches the tensors itself.  Entry-owned outputs are valid until the entry's next call at that row
count from ANY host thread (stream order does not serialize host calls): two host threads that share one entry pass their own
outputs or build one entry each; a caller that sweeps many row counts passes its own outputs (the per-row-count pairs are never
freed).  A returned entry-owned output cannot come back as an operand of a call that would write it: a call at that row count
whose ``x``, ``scale`` or passed output overlaps an entry-owned output it does not pass is declined (``ValueError``) before the
launch.  THE KERNEL WRITES THE LIVE ROWS' SCALE BYTES ONLY: a caller-owned ``scales`` buffer serves one row count, or its owner
clears it when the row count shrinks (after a call with more rows, a call with fewer leaves the earlier bytes in the rows between
``rows`` and ``scale_rows(rows)``).  Workspace: none.  The warm path allocates nothing and reads no device memory to the host.
"""

from typing import Any, Dict, Optional, Tuple

from cudnn._torch_stream import device_context  # imports no framework at module level
from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error, cutedsl_state
from cudnn.frost.launch_facts import LaunchShape, kernel_symbols, plain_form

from .kernel.row_quantize_fp4_decode_config import BLOCK, MAX_ROWS, NAME, SERVED_WIDTH, Cfg, scale_rows, validate_cfg, validate_rows, validate_width

#: the served compute capabilities as closed ranges of ``10 * major + minor``: SM100-class parts only.
SUPPORTED_CC_RANGES = ((100, 109),)
SUPPORTED_CC_TEXT = "10.x (SM100-class)"
DSL_FLOOR = (4, 8, 0)
DSL_FLOOR_TEXT = ".".join(str(x) for x in DSL_FLOOR)
#: cc 10.7 only: the oldest DSL compiler observed to assemble ``sm_107a`` (the CUDA 13 flavour of DSL 4.8.0); CUDA 12.9 rejects the target.
SM107_MIN_COMPILER = (13, 4)
ALIGN_BYTES = 16  # the 128-bit loads of the bf16 row; every multi-element view declares it
SCALE_ALIGN_BYTES = 4
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
    """The Rule-7 compiler gate.  On cc 10.0 / 10.3 both toolchain flavours of the DSL assemble the kernel, so both are served.
    cc 10.7 is not: the CUDA 12.9 flavour rejects the ``sm_107a`` target at compile time although the DSL's target list names it, so a
    compiler below ``SM107_MIN_COMPILER`` (the one observed to assemble it) is declined here, before JIT, naming what was found."""
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


def _knob(name: str, value):
    """A launch knob as given when it is a ``bool`` or ``None``; ``TypeError`` otherwise."""
    if value is None or isinstance(value, bool):
        return value
    raise TypeError(f"{NAME}: {name} must be a bool or None, got {type(value).__name__}")


def make_cfg(width, *, pdl=None, pdl_release=None) -> Cfg:
    """The validated config of the entry's constructor arguments (the construction declines, without torch): ``TypeError`` for a
    ``width`` that is not an ``int`` or a launch knob that is not a ``bool`` / ``None``; ``ValueError`` for a width outside the
    kernel's structure or ``pdl_release=True`` without ``pdl=True``; ``NotImplementedError`` for a structurally valid width the entry
    does not serve.  ``pdl`` ``None`` is off; ``pdl_release`` ``None`` follows ``pdl``."""
    width = validate_width(width)
    if width not in SERVED_WIDTH:
        raise NotImplementedError(f"{NAME} serves width {', '.join(str(w) for w in SERVED_WIDTH)} (validated); got width {width}")
    pdl, pdl_release = _knob("pdl", pdl), _knob("pdl_release", pdl_release)
    wait = bool(pdl)
    return validate_cfg(Cfg(WIDTH=width, PDL=wait, PDL_RELEASE=wait if pdl_release is None else pdl_release))


def _byte_range(t, elements: int) -> Tuple[int, int]:
    start = t.data_ptr()
    return start, start + elements * t.element_size()


def _touched_ranges(cfg: Cfg, rows: int, tensors: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
    """The byte range a launch of ``rows`` rows touches of each operand in ``tensors`` (by operand name): the live rows of a slab
    (the whole padded scale bands), the one scale."""
    width = cfg.WIDTH
    touched = dict(x=rows * width, scale=1, codes=rows * (width // 2), scales=scale_rows(rows) * (width // BLOCK))
    return {name: _byte_range(t, touched[name]) for name, t in tensors.items()}


def check_binding(cfg: Cfg, x, scale, codes=None, scales=None, device=None) -> int:
    """The binding checks of a call, run before every launch (compile-cache hits included).  ``TypeError`` for a non-tensor or a
    dtype; ``ValueError`` for an inconsistent device, shape or alignment, zero rows, or a written operand that overlaps another
    operand; ``NotImplementedError`` for a row count above the served ceiling or a non-contiguous operand.  Returns
    ``rows = x.shape[0]``.  ``device`` (an entry's: a concrete CUDA ordinal) requires every operand on exactly that device; without
    it the operands must share ``x``'s.  Host-only metadata arithmetic, no device work: ``scale``'s VALUE is never read.  The checks
    see the operands they are given: an output that is not passed is resolved by the entry, whose call checks its own buffers
    against these operands (:meth:`RowQuantizeFp4Decode.__call__`)."""
    import torch

    width = cfg.WIDTH
    tensors = {name: t for name, t in (("x", x), ("scale", scale), ("codes", codes), ("scales", scales)) if t is not None}
    for name in ("x", "scale"):
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
    dtypes = dict(x=torch.bfloat16, scale=torch.float32, codes=torch.uint8, scales=torch.float8_e4m3fn)
    for name, t in tensors.items():
        if t.dtype != dtypes[name]:
            raise TypeError(f"{NAME}: {name} must be {dtypes[name]} (kernel-native; callers convert), got {t.dtype}")
    if x.dim() != 2 or x.shape[1] != width or x.shape[0] < 1:
        raise ValueError(f"{NAME}: x must be [rows, {width}] with at least one row; got {tuple(x.shape)}")
    rows = int(x.shape[0])
    if rows > MAX_ROWS:
        raise NotImplementedError(f"{NAME} serves at most {MAX_ROWS} rows per call; got {rows}")
    if scale.numel() != 1:
        raise ValueError(f"{NAME}: scale must be one fp32 element (0-d or [1]); got {tuple(scale.shape)}")
    slabs = dict(codes=(rows, width // 2), scales=(scale_rows(rows), width // BLOCK))
    for name, (need_rows, columns) in slabs.items():
        t = tensors.get(name)
        if t is not None and (t.dim() != 2 or t.shape[0] < need_rows or t.shape[1] != columns):
            raise ValueError(f"{NAME}: {name} must be [>= {need_rows}, {columns}]; got {tuple(t.shape)}")
    for name, t in tensors.items():
        if not t.is_contiguous():
            raise NotImplementedError(
                f"{NAME} needs {name} contiguous (the kernel addresses rows by flat element; no copy is made); got strides {tuple(t.stride())}"
            )
    for name, t in tensors.items():
        align = SCALE_ALIGN_BYTES if name == "scale" else ALIGN_BYTES
        if t.data_ptr() % align:
            if name == "scale":
                why = "for its 32-bit load"
            elif name == "codes":
                why = "(the alignment every multi-element operand is bound with; the kernel stores a block's codes as one 64-bit word)"
            elif name == "scales":
                why = "(the alignment every multi-element operand is bound with; the kernel stores scales as bytes)"
            else:
                why = "for its 128-bit loads"
            raise ValueError(f"{NAME}: {name} must start at a {align}-byte aligned address {why}, got data_ptr % {align} = {t.data_ptr() % align}")
    ranges = _touched_ranges(cfg, rows, tensors)
    for name in OUTPUT_SLOTS:
        if name not in ranges:
            continue
        a0, a1 = ranges[name]
        for other, (b0, b1) in ranges.items():
            if other != name and a0 < b1 and b0 < a1:
                raise ValueError(f"{NAME}: {name} must not overlap {other}")
    return rows


def check_support(cfg: Cfg, x, scale, codes=None, scales=None, device_cc=None, device=None) -> int:
    """Typed declines without a DSL import: the binding checks of :func:`check_binding` plus the architecture gate
    (``NotImplementedError`` for operands off a CUDA device or a part this kernel does not serve).  ``device_cc`` overrides the
    compute-capability probe (tests).  The DSL gates (installed, version floor, the part's target, its compiler) run in
    :meth:`RowQuantizeFp4Decode.compile`."""
    import torch

    rows = check_binding(validate_cfg(cfg), x, scale, codes, scales, device=device)
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


class RowQuantizeFp4Decode:
    """The step in ONE form, compiled once and shared by every row count: all tensors are launch arguments, the row count is a
    runtime argument.  ``width`` is the row width; ``pdl`` / ``pdl_release`` the launch knobs (the module docstring).  The device is
    resolved to a concrete CUDA ordinal at construction, and every operand of every launch must live on exactly it.  An entry built
    for a non-CUDA device (host tests) runs every decline; its compile raises the typed decline of the device.  What a row count
    launches is readable without a launch: :meth:`kernel_symbols_for`, :meth:`form_for` and :meth:`launch_shape_for` (the launch
    facts of :mod:`cudnn.frost.launch_facts`)."""

    def __init__(self, device, width: int, *, pdl: Optional[bool] = None, pdl_release: Optional[bool] = None):
        self.cfg: Cfg = make_cfg(width, pdl=pdl, pdl_release=pdl_release)
        self.device = resolve_device(device)
        self.compiled: Dict[tuple, Any] = {}
        self.buffers: Dict[int, Dict[str, Any]] = {}
        self._buffer_views: Dict[int, Dict[str, Any]] = {}
        #: per row count, while the pair's initialization may still be pending: ``(event recorded behind it, the stream it is on)``
        self._init: Dict[int, Tuple[Any, Any]] = {}

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
        kernel serves every row count.  The symbol carries the form (``str`` of the config: the width and the launch arm) directly
        behind the kernel's name, so two forms never share a symbol."""
        self._fact_cfg(rows)
        fn = self.compiled.get(self.key())
        return kernel_symbols(() if fn is None else (fn,), NAME)

    def form_for(self, rows: int) -> Dict[str, Any]:
        """The form of ``rows`` as plain data: the fields of the entry's config.  The compile key is the config alone, so nothing
        else is part of it: the row count is a runtime fact of a launch (:meth:`launch_shape_for`)."""
        return plain_form(self._fact_cfg(rows))

    def launch_shape_for(self, rows: int) -> LaunchShape:
        """``(grid, block, shared memory)`` of the launch of ``rows``: one thread block per row, ``width / 16`` threads, no shared
        memory."""
        cfg = self._fact_cfg(rows)
        return LaunchShape((int(rows), 1, 1), (cfg.THREADS, 1, 1), cfg.SMEM_BYTES)

    # ---- entry-owned outputs ----
    def outputs(self, rows: int) -> Dict[str, Any]:
        """The entry-owned output buffers of ``rows``: ``codes`` (uint8 ``[rows, width / 2]``) and ``scales`` (E4M3
        ``[scale_rows(rows), width / 16]``, zero-filled).  Allocated once per row count; see the module docstring for their
        lifetime.  The pair is returned WITHOUT waiting for its initialization: the allocation and the zero-fill are stream work
        of torch's current stream of the entry's device at the first ``outputs(rows)`` / ``compile(rows)`` / call of the row count.
        A call of the entry orders itself behind that work on whatever stream it launches (:meth:`_order_behind_init`); a caller
        that reads or writes the returned tensors ITSELF before the entry has been called at that row count does so on that
        stream, or orders its own stream behind it first (the rule for any tensor created on another stream)."""
        rows = validate_rows(rows)
        if rows not in self.buffers:
            import torch

            cfg, dev = self.cfg, self.device
            with device_context(dev):
                self.buffers[rows] = dict(
                    codes=torch.empty(rows, cfg.WIDTH // 2, dtype=torch.uint8, device=dev),
                    scales=torch.zeros(scale_rows(rows), cfg.WIDTH // BLOCK, dtype=torch.float8_e4m3fn, device=dev),
                )
                # Inside a stream capture the fill is a node of the captured graph and an event recorded there cannot be queried.
                if dev.type == "cuda" and not torch.cuda.is_current_stream_capturing():
                    stream = torch.cuda.current_stream(dev)
                    done = torch.cuda.Event()
                    done.record(stream)  # behind all the allocation enqueued and all this stream still owes a recycled block
                    self._init[rows] = (done, stream)
        return self.buffers[rows]

    def _order_behind_init(self, rows: int, stream, driver) -> None:
        """Order a launch on ``stream`` that writes the entry-owned outputs of ``rows`` behind their initialization (Rule 5):
        the launch may be on another stream than the one the pair was allocated on, where the zero-fill of ``scales`` -- or a
        write the allocating stream still owes a recycled block -- would otherwise land on top of the kernel's bytes.  Complete:
        the record is dropped, and a warm call does nothing here.  Pending, the same stream: stream order holds.  Pending,
        another stream: that stream waits for the event (``cudaStreamWaitEvent``; the host does not wait).  Pending, ``stream``
        capturing: a dependency on an event recorded outside the capture cannot be captured, so the HOST waits for this one
        event (``cudaEventSynchronize``) before the captured launch -- never a stream or device synchronization -- and the
        captured graph holds the kernel alone.  ``torch.cuda.graph`` synchronizes the device when it is entered, so only a
        capture begun without it can meet a pending initialization.  The event is read with this thread's capture check
        relaxed and restored: while ANY thread captures in the default mode, querying an event is refused and invalidates that
        capture."""
        import torch

        done, init_stream = self._init[rows]
        capturing = torch.cuda.is_current_stream_capturing()
        err, previous = driver.cuThreadExchangeStreamCaptureMode(driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_RELAXED)
        if int(err) != 0:
            raise RuntimeError(f"{NAME}: cuThreadExchangeStreamCaptureMode failed: {err}")
        try:
            if capturing:
                done.synchronize()
            if capturing or done.query():
                self._init.pop(rows, None)
            elif stream != init_stream:
                stream.wait_event(done)
        finally:
            driver.cuThreadExchangeStreamCaptureMode(previous)

    def _output_view(self, t):
        """The DSL view of an output: the code and scale bytes through their ``uint8`` view (one byte per element)."""
        import torch

        return _flat(t.view(torch.uint8))

    def _owned_views(self, rows: int) -> Dict[str, Any]:
        """The DSL views of the entry-owned outputs of ``rows``: the buffers never move, so each view is built once."""
        if rows not in self._buffer_views:
            self._buffer_views[rows] = {name: self._output_view(t) for name, t in self.outputs(rows).items()}
        return self._buffer_views[rows]

    def _check_entry_owned(self, rows: int, operands: Dict[str, Any]) -> None:
        """The overlap check of :func:`check_binding` for the outputs a call does not pass: the launch writes the entry's own
        buffers of ``rows`` in their place, and a tensor an earlier call returned can come back as an operand.  ``ValueError``
        when such a buffer overlaps another operand of the call (``operands``: every operand by name, ``None`` where not passed).
        Before the row count's buffers exist there is nothing to check -- what the call then allocates is fresh memory, disjoint
        from every live tensor.  Host-only pointer arithmetic; allocates nothing."""
        owned = self.buffers.get(rows)
        if owned is None:
            return
        entry_owned = {name: t for name, t in owned.items() if operands[name] is None}
        if not entry_owned:
            return
        passed = {name: t for name, t in operands.items() if t is not None}
        ranges = _touched_ranges(self.cfg, rows, {**passed, **entry_owned})
        for name in entry_owned:
            a0, a1 = ranges[name]
            for other, (b0, b1) in ranges.items():
                if other != name and a0 < b1 and b0 < a1:
                    raise ValueError(
                        f"{NAME}: the entry-owned {name} must not overlap {other} ({name} was not passed, so the launch writes the "
                        f"entry's own {name} of {rows} row(s), the tensor earlier calls of that row count returned; pass a caller-owned {name})"
                    )

    # ---- compile ----
    def compile(self, rows: Optional[int] = None) -> None:
        """Plan time: the typed declines of the device, the part and the DSL, then ONE trace-compile of the entry's form (a no-op
        once compiled).  With ``rows`` it also allocates the entry-owned outputs of that row count -- call it for every row count
        before a stream capture, on any stream: a later call orders itself behind the allocation's stream work
        (:meth:`_order_behind_init`).  The compile's own operands are temporaries nothing launches on; their blocks return to
        the allocator of the stream that filled them.  Needs no operand: the compile binds the entry-owned outputs of
        ``rows``, and of ONE row when no row count is given -- so the first ``compile()`` allocates that one-row pair (2,560
        code bytes and one 128-row scale band) even for a caller that passes its own outputs on every call."""
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

            from .kernel import row_quantize_fp4_decode_f16 as mod

            cfg = self.cfg
            with device_context(self.device):  # a compiled launchable belongs to its device's context
                # plan-time operands of one row: the views are layout-dynamic, so no extent of them enters the kernel
                x = torch.zeros(1, cfg.WIDTH, dtype=torch.bfloat16, device=self.device)
                scale = torch.ones(1, dtype=torch.float32, device=self.device)
                views = self._owned_views(1 if rows is None else rows)
                args = [_flat(x), _flat(scale), views["codes"], views["scales"]]
                self.compiled[self.key()] = cute.compile(mod.launch, cfg, *args, cutlass.Int32(1 if rows is None else rows), make_fake_stream())
        if rows is not None:
            self.outputs(rows)

    # ---- launch ----
    def __call__(self, x, scale, codes=None, scales=None):
        """Launch on torch's current stream of the entry's device; returns ``(codes, scales)`` -- per slot the caller's tensor as
        given, else the entry-owned buffer of this row count.  Compiles on first use.  Every launch re-runs the binding checks
        before the compiled kernel is looked up, the overlap check on the RESOLVED operands: an entry-owned output the launch would
        write must not overlap ``x``, ``scale`` or a passed output (``ValueError``; a returned output fed back in needs a
        caller-owned output).  The launch is scoped to the entry's device and the caller's current device is restored behind it."""
        import torch

        cfg = self.cfg
        rows = check_binding(cfg, x, scale, codes, scales, device=self.device)
        given = dict(codes=codes, scales=scales)
        self._check_entry_owned(rows, dict(x=x, scale=scale, **given))
        key = self.key()
        if key not in self.compiled:
            self.compile()
        import cuda.bindings.driver as cuda  # behind compile(): its gates decline a host without the DSL by type first
        import cutlass

        with device_context(self.device):
            results, views = {}, {}
            owned = owned_views = None
            for name in OUTPUT_SLOTS:
                if given[name] is not None:
                    results[name] = given[name]
                    views[name] = self._output_view(given[name])
                else:
                    if owned is None:
                        owned, owned_views = self.outputs(rows), self._owned_views(rows)
                    results[name] = owned[name]
                    views[name] = owned_views[name]
            stream = torch.cuda.current_stream(self.device)
            if owned is not None and rows in self._init:
                self._order_behind_init(rows, stream, cuda)
            self.compiled[key](_flat(x), _flat(scale), views["codes"], views["scales"], cutlass.Int32(rows), cuda.CUstream(stream.cuda_stream))
        return results["codes"], results["scales"]


__all__ = ["RowQuantizeFp4Decode", "check_binding", "check_support", "cc_supported", "scale_rows", "make_cfg"]
