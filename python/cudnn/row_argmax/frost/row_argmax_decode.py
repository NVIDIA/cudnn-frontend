# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host entry of the row argmax DECODE kernel (``kernel/row_argmax_decode_f16.py``): validation with typed declines, the plan-time
compile and the one launch.  No DSL and no torch import at module level -- the kernel module is imported inside
:meth:`RowArgmaxDecode.compile` after the version / target / compiler gates (python/cudnn/AGENTS.md Rule 7), so a host without
the DSL can still construct the entry and run its declines.  The binding checks (:func:`check_binding`: type, device, dtype,
shape, strides, alignment and overlap of every operand) run before EVERY launch, compile-cache hits included -- the compile key
holds no operand fact but the table kind and the kernel takes its operands as addresses, so the host is a launch's only guard.
The overlap check covers the output the call does NOT pass too: the entry's own vector of the row count is an operand of the
launch like a passed one.

The op, per row of an ``[rows, width]`` bf16 slab::

    column  = the LOWEST column whose value is greater than -inf and not below any other value of the row;
              0 when the row holds no such value (every element is -inf or a NaN)
    out[r]  = column                 without an id table   (int32)
            = ids[column]            with one              (a bit copy, in the table's dtype)

A decode-step entry: the greedy choice over the logits of a restricted vocabulary, whose column order is not the id order -- the
table maps a column to its id.  The op is exact: compares and bit copies, no arithmetic, so the result is one value on every part
and in any evaluation order.  A NaN is never chosen and never hides another element; ``+0`` and ``-0`` are equal (the lower
column wins).  On rows without a NaN the column is ``torch.argmax``'s.

Served (declined otherwise, by name): ``width`` 65536; ``rows`` 1 .. 1024 (a RUNTIME kernel argument); ``x`` bf16 ``[rows,
width]``, a row contiguous, the row stride free in whole 16-byte vectors (a view of a wider slab binds as it is; a one-row
call uses no row stride, so its ``x`` binds whatever stride the view carries), 16-byte aligned; ``ids`` absent, or int32 /
int64 ``[width]``, contiguous, aligned to its element; ``out`` ``[>= rows]`` of the result's dtype (int32 without a table,
else the table's), contiguous, aligned to its element; every operand on the entry's device; the results a call writes (the
first ``rows`` elements of ``out``) sharing no byte with ``x`` or ``ids``; a compute capability 10.x part; a CuTe DSL >= 4.8.0
that holds the part's target and compiles it (cc 10.7: the CUDA 13 toolchain flavour).  Nothing is checked on the host about a
VALUE (Rule 3).

THE TABLE KIND IS A FACT OF THE CALL: no table, an int32 table or an int64 table.  It is a load and a store width of the kernel,
so each kind is one compiled kernel under one compile key ``(config, kind)``.  :meth:`RowArgmaxDecode.compile` needs no operand
and compiles all three, so that nothing compiles inside a step whatever table a later call binds; a call on an entry that was
not compiled compiles its own kind on first use.

There is no launch knob: the launch is ordinary -- it carries no programmatic-stream-serialization attribute and the kernel holds
neither a wait nor a release -- on ``torch.cuda.current_stream`` of the entry's device, under that device, and the caller's
current device is restored behind it (Rule 5).

Output ownership.  With ``out`` the caller owns the storage and gets its own tensor back (elements past ``rows`` are untouched).
Without it the entry returns ITS vector of the row count and the result's dtype: per row count one int32 and one int64 vector of
``rows`` elements, allocated together by ``compile(rows)`` / ``outputs(rows)`` or at the first call of that row count without
``out`` -- call ``compile(rows)`` for every row count before a stream capture.  THE ALLOCATION OF AN ENTRY-OWNED PAIR IS STREAM
WORK of torch's current stream of the entry's device at that moment: the pair is not initialized by the entry, but under
deterministic algorithms torch fills it, and its blocks can be recycled ones that stream still owes a write.  EVERY launch that
writes bytes of the pair is ordered behind that work on WHATEVER stream it runs, HOWEVER the tensor reached the call -- by
default, or taken with ``outputs`` and passed back as ``out``, whole or as a view, at this row count or another, by this entry or
by another entry that orders its launches the same way (``cudnn.frost.init_order``; the test is one of addresses: the bytes a
launch writes against the pair's allocations).  The launch stream waits for the event recorded behind the allocation -- a
stream-side wait: no host wait, never a synchronization of a stream or of the device -- so ``compile(rows)`` may run on any
stream.  A captured launch cannot carry that dependency: a call that finds the work still pending under a capture waits for that
ONE event on the host before its captured launch (set-up, never part of a replay; a pair allocated before the capture needs no
wait once its stream's earlier work is done: ``torch.cuda.graph`` synchronizes when it is entered, a raw capture does not).  NOT
ordered: a pair allocated INSIDE a stream capture (nothing is recorded: allocate before the capture).  WHAT STAYS THE CALLER'S:
``x`` and ``ids``, which a launch only reads, and the caller's own reads and writes of a returned tensor outside a call of an
entry.  Entry-owned outputs are valid until the entry's next call at that row count from ANY host thread: stream order does not
serialize host calls, so two host threads that share one entry pass their own ``out`` or build one entry each.  The per-row-count
pairs are never freed.  Workspace: none.  The warm path allocates nothing and reads no device memory to the host; the one host
wait of the entry is the one above.
"""

from typing import Any, Dict, Optional, Tuple

from cudnn._torch_stream import device_context  # imports no framework at module level
from cudnn.frost import init_order  # imports no framework at module level
from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error, cutedsl_state
from cudnn.frost.launch_facts import LaunchShape, kernel_symbols, plain_form

from .kernel.row_argmax_decode_config import (
    ELEMENT_BYTES,
    INT_BYTES,
    MAX_ROWS,
    NAME,
    RESULT_DTYPE,
    SERVED_WIDTH,
    TABLE_KINDS,
    VEC_BYTES,
    Cfg,
    validate_cfg,
    validate_rows,
    validate_table_kind,
    validate_width,
)

#: the served compute capabilities as closed ranges of ``10 * major + minor``: SM100-class parts only.
SUPPORTED_CC_RANGES = ((100, 109),)
SUPPORTED_CC_TEXT = "10.x (SM100-class)"
DSL_FLOOR = (4, 8, 0)
DSL_FLOOR_TEXT = ".".join(str(x) for x in DSL_FLOOR)
#: cc 10.7 only: the oldest DSL compiler observed to assemble ``sm_107a`` (the CUDA 13 flavour of DSL 4.8.0); CUDA 12.9 rejects the target.
SM107_MIN_COMPILER = (13, 4)
ALIGN_BYTES = VEC_BYTES  # the 128-bit loads of the bf16 row
#: the result dtypes of the entry, by name: the keys of an entry-owned pair
RESULT_DTYPES = tuple(dict.fromkeys(RESULT_DTYPE.values()))


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


def make_cfg(width) -> Cfg:
    """The validated config of the entry's constructor argument (the construction declines, without torch): ``TypeError`` for a
    ``width`` that is not an ``int``; ``ValueError`` for a width outside the kernel's structure; ``NotImplementedError`` for a
    structurally valid width the entry does not serve."""
    width = validate_width(width)
    if width not in SERVED_WIDTH:
        raise NotImplementedError(f"{NAME} serves width {', '.join(str(w) for w in SERVED_WIDTH)} (validated); got width {width}")
    return validate_cfg(Cfg(WIDTH=width))


def table_kind(ids) -> str:
    """The table kind of a call that binds ``ids``: ``"none"`` without a table, else the name of the table's dtype.  ``TypeError``
    for a table that is not a tensor or whose dtype is neither int32 nor int64."""
    if ids is None:
        return "none"
    import torch

    if not isinstance(ids, torch.Tensor):
        raise TypeError(f"{NAME}: ids must be a torch.Tensor or None, got {type(ids).__name__}")
    names = {torch.int32: "int32", torch.int64: "int64"}
    if ids.dtype not in names:
        raise TypeError(f"{NAME}: ids must be torch.int32 or torch.int64 (the table's element is copied as it is), got {ids.dtype}")
    return names[ids.dtype]


def _touched(rows: int, tensors: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
    """The byte range ``[start, end)`` a launch of ``rows`` rows can touch of each operand in ``tensors`` (by operand name): the
    bounds of the slab view, the whole table, the first ``rows`` results."""
    return {name: init_order.span(t, rows if name == "out" else None) for name, t in tensors.items()}


def _check_overlap(rows: int, tensors: Dict[str, Any], what: str = "") -> None:
    ranges = _touched(rows, tensors)
    a0, a1 = ranges["out"]
    for other, (b0, b1) in ranges.items():
        if other != "out" and a0 < b1 and b0 < a1:
            raise ValueError(f"{NAME}: {what}out must not overlap {other}")


def check_binding(cfg: Cfg, x, ids=None, out=None, device=None) -> int:
    """The binding checks of a call, run before every launch (compile-cache hits included).  ``TypeError`` for a non-tensor or a
    dtype (``x`` bf16; ``ids`` int32 or int64; ``out`` the result's dtype: int32 without a table, else the table's);
    ``ValueError`` for an inconsistent device, shape, row stride (of a call of two or more rows: a one-row launch never uses
    the stride, so it is not held to whole vectors there) or alignment, an empty call, or an ``out`` whose first ``rows``
    elements overlap ``x`` or ``ids``; ``NotImplementedError`` for a row count above the served ceiling or an operand that is
    not contiguous where the kernel addresses flat.  Returns ``rows = x.shape[0]``.  ``device`` (an entry's: a concrete CUDA
    ordinal) requires every operand on exactly that device; without it the operands must share ``x``'s.  Host-only metadata
    arithmetic, no device work: no operand's VALUE is read.  The checks see the operands they are given: an output that is
    not passed is resolved by the entry, whose call checks its own vector against these operands."""
    import torch

    width = cfg.WIDTH
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{NAME}: x must be a torch.Tensor, got {type(x).__name__}")
    kind = table_kind(ids)
    if out is not None and not isinstance(out, torch.Tensor):
        raise TypeError(f"{NAME}: out must be a torch.Tensor or None, got {type(out).__name__}")
    tensors = {name: t for name, t in (("x", x), ("ids", ids), ("out", out)) if t is not None}
    want = resolve_device(device) if device is not None else x.device
    whose = "the entry's" if device is not None else "x's"
    for name, t in tensors.items():
        if t.device != want:
            raise ValueError(f"{NAME}: every operand must live on one device, {whose} ({want}); {name} is on {t.device}")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"{NAME}: x must be {torch.bfloat16} (kernel-native; callers convert), got {x.dtype}")
    result = getattr(torch, RESULT_DTYPE[kind])
    if out is not None and out.dtype != result:
        source = "the dtype of ids" if ids is not None else "the columns of a call without ids"
        raise TypeError(f"{NAME}: out must be {result} ({source}), got {out.dtype}")
    if x.dim() != 2 or x.shape[1] != width or x.shape[0] < 1:
        raise ValueError(f"{NAME}: x must be [rows, {width}] with at least one row; got {tuple(x.shape)}")
    rows = int(x.shape[0])
    if rows > MAX_ROWS:
        raise NotImplementedError(f"{NAME} serves at most {MAX_ROWS} rows per call; got {rows}")
    if ids is not None and tuple(ids.shape) != (width,):
        raise ValueError(f"{NAME}: ids must be [{width}] (one id per column of x); got {tuple(ids.shape)}")
    if out is not None and (out.dim() != 1 or out.shape[0] < rows):
        raise ValueError(f"{NAME}: out must be [>= {rows}] (one result per row of x); got {tuple(out.shape)}")
    row_stride, element_stride = (int(s) for s in x.stride())
    if element_stride != 1:
        raise NotImplementedError(f"{NAME} needs x contiguous along a row (a row is addressed flat; no copy is made); got strides {tuple(x.stride())}")
    if rows > 1 and (row_stride * ELEMENT_BYTES) % ALIGN_BYTES:  # one row: the kernel multiplies the stride by row 0
        raise ValueError(
            f"{NAME}: x stride[0] = {row_stride} elements ({row_stride * ELEMENT_BYTES} bytes) must be a multiple of {ALIGN_BYTES} bytes "
            f"for a call of {rows} rows (every row starts on a 16-byte vector)"
        )
    for name in ("ids", "out"):
        if name in tensors and int(tensors[name].stride(0)) != 1:
            raise NotImplementedError(
                f"{NAME} needs {name} contiguous (the kernel addresses it by flat element; no copy is made); got strides {tuple(tensors[name].stride())}"
            )
    if x.data_ptr() % ALIGN_BYTES:
        raise ValueError(
            f"{NAME}: x must start at a {ALIGN_BYTES}-byte aligned address for its 128-bit loads, got data_ptr % {ALIGN_BYTES} = {x.data_ptr() % ALIGN_BYTES}"
        )
    for name in ("ids", "out"):
        if name in tensors and tensors[name].data_ptr() % tensors[name].element_size():
            size = tensors[name].element_size()
            raise ValueError(f"{NAME}: {name} must be aligned to its {size}-byte element, got data_ptr % {size} = {tensors[name].data_ptr() % size}")
    if out is not None:
        _check_overlap(rows, tensors)
    return rows


def check_support(cfg: Cfg, x, ids=None, out=None, device_cc=None, device=None) -> int:
    """Typed declines without a DSL import: the binding checks of :func:`check_binding` plus the architecture gate
    (``NotImplementedError`` for operands off a CUDA device or a part this kernel does not serve).  ``device_cc`` overrides the
    compute-capability probe (tests).  The DSL gates (installed, version floor, the part's target, its compiler) run in
    :meth:`RowArgmaxDecode.compile`."""
    import torch

    rows = check_binding(validate_cfg(cfg), x, ids, out, device=device)
    dev = x.device
    if device_cc is None and dev.type != "cuda":
        raise NotImplementedError(f"{NAME} serves CUDA tensors only; the operands are on {dev}")
    cc = tuple(torch.cuda.get_device_capability(dev)) if device_cc is None else tuple(device_cc)
    if not cc_supported(cc):
        raise NotImplementedError(f"{NAME} serves compute capability {SUPPORTED_CC_TEXT}; this device is {cc[0]}.{cc[1]}")
    return rows


def launch_arguments(rows: int, x, ids, out) -> Tuple[int, int, int, int, int]:
    """The kernel's runtime arguments of a call whose binding was checked, as plain ``int`` in launch order: the address of ``x``,
    its row stride in bytes, the address of ``ids`` (0 without a table: that kernel reads none), the address of ``out`` and the
    row count.  Host-only metadata: nothing is read from the device and nothing is allocated."""
    return (x.data_ptr(), int(x.stride(0)) * ELEMENT_BYTES, 0 if ids is None else ids.data_ptr(), out.data_ptr(), int(rows))


class RowArgmaxDecode:
    """The step in ONE form, shared by every row count: all tensors are launch arguments, the row count is a runtime argument.
    ``width`` is the row width.  The device is resolved to a concrete CUDA ordinal at construction, and every operand of every
    launch must live on exactly it.  An entry built for a non-CUDA device (host tests) runs every decline; its compile raises
    the typed decline of the device.  What a row count launches is readable without a launch: :meth:`kernel_symbols_for`,
    :meth:`form_for` and :meth:`launch_shape_for` (the launch facts of :mod:`cudnn.frost.launch_facts`)."""

    def __init__(self, device, width: int):
        self.cfg: Cfg = make_cfg(width)
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

    def key(self, table: str = "none") -> tuple:
        """The compile key of a call whose table kind is ``table`` (:func:`table_kind`): plan-time facts only -- the config and
        the kind.  No operand enters the compiled kernel: it takes addresses and a byte stride as runtime arguments and the row
        count as its grid."""
        return (self.cfg, validate_table_kind(table))

    # ---- launch facts (cudnn.frost.launch_facts): three read-only answers per served row count ----
    def _fact_cfg(self, rows) -> Cfg:
        """The config of the row count a launch fact is asked for: ``ValueError`` for a ``rows`` that is not an ``int`` (the decline
        every entry's launch facts give a malformed row count), then the typed declines of :meth:`cfg_for`."""
        if isinstance(rows, bool) or not isinstance(rows, int):
            raise ValueError(f"{NAME}: rows must be an int, got {rows!r}")
        return self.cfg_for(rows)

    def kernel_symbols_for(self, rows: int) -> Tuple[str, ...]:
        """The symbols of the kernels this entry has compiled, in compile order: one per compiled table kind, none until a kind
        is compiled.  Every compiled kernel serves every row count.  A symbol carries the form (the width, then the table kind)
        directly behind the kernel's name, so two forms never share a symbol."""
        self._fact_cfg(rows)
        return kernel_symbols(self.compiled.values(), NAME)

    def form_for(self, rows: int) -> Dict[str, Any]:
        """The form of ``rows`` as plain data: the fields of the entry's config.  The table kind is a fact of a call's operands,
        not of a row count, and the row count is a runtime fact of a launch (:meth:`launch_shape_for`)."""
        return plain_form(self._fact_cfg(rows))

    def launch_shape_for(self, rows: int) -> LaunchShape:
        """``(grid, block, shared memory)`` of the launch of ``rows`` rows, whatever its table kind: one thread block per row,
        1024 threads, and the kernel's one shared-memory buffer (a maximum and an index per warp)."""
        cfg = self._fact_cfg(rows)
        return LaunchShape((int(rows), 1, 1), (cfg.THREADS, 1, 1), cfg.SMEM_BYTES)

    # ---- entry-owned outputs ----
    def outputs(self, rows: int) -> Dict[str, Any]:
        """The entry-owned result vectors of ``rows`` rows by dtype name: ``"int32"`` and ``"int64"``, ``[rows]`` each; a call
        without ``out`` returns the one of its result's dtype.  Allocated once per row count; see the module docstring for
        their lifetime and for what is ordered behind their allocation.  Returns a copy of the entry's table (the same
        tensors), without waiting."""
        return dict(self._owned(rows))

    def _owned(self, rows: int) -> Dict[str, Any]:
        """The entry's own table of the pair of ``rows`` rows, allocated on first use.  The allocation's stream work is recorded
        (:mod:`cudnn.frost.init_order`) BEFORE the pair is in the table."""
        rows = validate_rows(rows)
        if rows not in self.buffers:
            import torch

            bufs = {name: torch.empty(rows, dtype=getattr(torch, name), device=self.device) for name in RESULT_DTYPES}
            init_order.record(self._pending, bufs.values())
            self.buffers[rows] = bufs
        return self.buffers[rows]

    def _launch_writes(self, rows: int, out) -> Tuple[Tuple[int, int], ...]:
        """The byte range a launch of ``rows`` rows WRITES: the first ``rows`` results of ``out`` as resolved for the call --
        what the launch is ordered by (:func:`cudnn.frost.init_order.order_behind`)."""
        return (init_order.span(out, rows),)

    def _check_entry_owned(self, rows: int, table: str, x, ids) -> None:
        """The overlap check of :func:`check_binding` for a call that passes no ``out``: the launch writes the entry's own
        vector of ``rows`` in its place, so that vector is held to the rule of a passed one.  ``ValueError`` when that vector
        overlaps ``x`` or ``ids`` (a vector the entry allocated is a separate allocation of at most 1024 integers and cannot be
        bound as ``x`` or ``ids``: the check is the same rule on whatever the entry's table holds).  Before the row count's pair
        exists there is nothing to check -- what the call then allocates is fresh memory, disjoint from every live tensor.
        Host-only pointer arithmetic; allocates nothing."""
        owned = self.buffers.get(rows)
        if owned is None:
            return
        tensors = {name: t for name, t in (("x", x), ("ids", ids)) if t is not None}
        try:
            _check_overlap(rows, dict(tensors, out=owned[RESULT_DTYPE[table]]), what="the entry-owned ")
        except ValueError as err:
            raise ValueError(
                f"{err} (out was not passed, so the launch writes the entry's own vector of {rows} row(s), the tensor earlier calls "
                f"of that row count returned; pass a caller-owned out)"
            ) from None

    # ---- compile ----
    def _build(self, tables: Tuple[str, ...]) -> None:
        """The typed declines of the device, the part and the DSL, then one trace-compile per table kind of ``tables`` that is
        not compiled yet."""
        missing = [table for table in tables if self.key(table) not in self.compiled]
        if not missing:
            return
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

        from .kernel import row_argmax_decode_f16 as mod

        with device_context(self.device):  # a compiled launchable belongs to its device's context
            # plan-time arguments: addresses and the stride are runtime values, the zeros pin their TYPES only
            args = [cutlass.Int64(0)] * 4 + [cutlass.Int32(1)]
            for table in missing:
                self.compiled[self.key(table)] = cute.compile(mod.launch, self.cfg, table, *args, make_fake_stream())

    def compile(self, rows: Optional[int] = None) -> None:
        """Plan time: the typed declines of the device, the part and the DSL, then one trace-compile per table kind (three; a
        no-op once compiled), so that no later call compiles whatever table it binds.  With ``rows`` it also allocates the
        entry-owned outputs of that row count -- call it for every row count before a stream capture, on any stream: a later
        launch that writes them is ordered behind the allocation's stream work.  Needs no operand."""
        if rows is not None:
            validate_rows(rows)
        self._build(TABLE_KINDS)
        if rows is not None:
            self._owned(rows)

    # ---- launch ----
    def __call__(self, x, ids=None, out=None):
        """Launch on torch's current stream of the entry's device; returns ``out`` -- the caller's tensor as given, else the
        entry-owned ``[rows]`` vector of the result's dtype.  Compiles the call's table kind on first use.  Every launch
        re-runs the binding checks before the compiled kernel is looked up, the overlap check on the RESOLVED operands
        (``ValueError``: the entry-owned vector the launch would write must not overlap ``x`` or ``ids``).  The launch is
        ordered behind the pending stream work of every entry-owned allocation of the device it writes (the module docstring),
        scoped to the entry's device, and the caller's current device is restored behind it."""
        import torch

        rows = check_binding(self.cfg, x, ids, out, device=self.device)
        table = table_kind(ids)
        if out is None:
            self._check_entry_owned(rows, table, x, ids)
        key = self.key(table)
        if key not in self.compiled:
            self._build((table,))
        import cuda.bindings.driver as cuda  # behind the compile: its gates decline a host without the DSL by type first
        import cutlass

        with device_context(self.device):
            if out is None:
                out = self._owned(rows)[RESULT_DTYPE[table]]
            values = launch_arguments(rows, x, ids, out)
            stream = torch.cuda.current_stream(self.device)
            if self._pending:  # empty in the steady state: a warm call does nothing here
                init_order.order_behind(self._pending, stream, self._launch_writes(rows, out))
            self.compiled[key](*(cutlass.Int64(value) for value in values[:-1]), cutlass.Int32(values[-1]), cuda.CUstream(stream.cuda_stream))
        return out


__all__ = [
    "RowArgmaxDecode",
    "check_binding",
    "check_support",
    "cc_supported",
    "launch_arguments",
    "make_cfg",
    "table_kind",
    "INT_BYTES",
    "SERVED_WIDTH",
    "TABLE_KINDS",
]
