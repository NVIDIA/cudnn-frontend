# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host entry of the attention preparation DECODE kernel (``kernel/prepare_decode_f16.py``): validation with typed declines, the
plan-time compile and the one launch.  No DSL and no torch import at module level -- the kernel module is imported inside
:meth:`AttentionPrepareDecode.compile` after the version / target / compiler gates (python/cudnn/AGENTS.md Rule 7), so a host
without the DSL can still construct the entry and run its construction declines.  The binding checks (:func:`check_binding`:
presence, type, device, dtype, shape, strides, alignment and overlap of every operand) run before EVERY launch, compile-cache hits
included -- the compile key is the config alone and the kernel takes its operands as addresses, so the host is a launch's only
guard.  The pools and the slot vector are checked by the paged-pool contract of ``cudnn.gated_attention_block``
(``cache_write_contract.check_pool`` / ``check_slot_mapping``, unchanged); that package imports the CuTe DSL, so on a host
without the DSL a call that reaches those two checks gets the typed decline that names the missing DSL (every check in front of
them has run).  The overlap check covers the outputs the call does NOT pass too: the entry's own buffers of the token count are
operands of the launch like the passed ones.

The op, per token ``t`` of a decode step (fp32 arithmetic on bf16 operands; ``D = HEAD_DIM``, ``HALF = ROTARY_DIM / 2``)::

    n(x, w)        = bf16( x * rsqrt(mean(x^2) + eps) * (1 + w) )                 the RMSNorm of one head row, rounded once to bf16
    rot(n, p)[r]        = bf16( n[r] * cos[p_r, r] - n[r + HALF] * sin[p_r, r] )  r < HALF; p_r = the token's position for index r
    rot(n, p)[r + HALF] = bf16( n[r + HALF] * cos[p_r, r] + n[r] * sin[p_r, r] )  one product rounded to fp32, the other fused
    rot(n, p)[e]        = n[e]                                                    e >= ROTARY_DIM

    q_out[t, h]              = rot(n(q[t, h], q_weight), positions of t)          h < HEADS
    gate_out[t, h]           = gate[t, h]                                         a bit copy
    k_pool[page, h, row]     = rot(n(k[t, h], k_weight), positions of t)          h < KV_HEADS; page, row = divmod(slot_mapping[t], page_size)
    v_pool[page, h, row]     = v[t, h]                                            a bit copy

One launch prepares what a paged attention consumes for the step's tokens: the normalized, rotated queries and their gates as
compact ``[tokens, HEADS, D]`` buffers, and the tokens' keys and values written through into the paged pools.  A token whose slot is
negative, or at or past ``pages x page_size``, or that lies past the slot vector's length, writes no pool row; its query and gate
rows are written like every other token's.

CONSTRUCTION FACTS ARE THE COMPILE KEY.  The geometry, ``eps``, the position form with its section, ``page_size`` and the two integer
dtypes are compile-time constants of the kernel, so ``compile()`` needs no operand and nothing compiles inside a step.  The page
size is a constant because the kernel divides the slot by it (a shift); an integer dtype is a load width.  A pool whose page size is
not the entry's and an integer tensor of the other dtype are declined by name; a caller that needs two keys builds two entries.
The slot vector's LENGTH is a run-time argument.

Served (declined otherwise, by name): geometry ``(HEADS, KV_HEADS, HEAD_DIM, ROTARY_DIM)`` = ``(24, 4, 256, 64)``; ``page_size`` 64
or 16; tokens 1 .. 1024 (a RUNTIME kernel argument); ``q`` / ``gate`` ``[tokens, HEADS, D]`` and ``k`` / ``v`` ``[tokens, KV_HEADS,
D]`` bf16 views with ``D`` contiguous and the other two strides whole 16-byte vectors (four views of one projection slab bind as
they are); ``q_weight`` / ``k_weight`` bf16 ``[D]``; ``cos_sin`` bf16 ``[P, ROTARY_DIM]`` contiguous (per position ``HALF``
cosines, then ``HALF`` sines); ``positions`` of the entry's ``position_dtype``: ``[tokens]`` (``"1d"``) or ``[3, tokens]`` with a
free row stride (``"3d"``); ``k_pool`` / ``v_pool`` bf16 ``[pages, KV_HEADS, page_size, D]`` with ``D`` contiguous and the other
strides whole 16-byte vectors (the two halves of one packed cache bind as they are); ``slot_mapping`` of the entry's ``slot_dtype``,
contiguous, at most ``tokens`` entries; ``q_out`` / ``gate_out`` bf16 ``[>= tokens, HEADS, D]`` contiguous; every bf16 operand 16-byte
aligned, the integer operands aligned to their element; every operand on the entry's device; no written operand (``q_out``,
``gate_out``, the pools) sharing a byte with another operand, and no two rows of a pool sharing a byte (a pool view with an
expanded axis or overlapping rows is declined); a compute capability 10.x part; a CuTe DSL >= 4.8.0 that holds the
part's target and compiles it (cc 10.7: the CUDA 13 toolchain flavour).  Caller contracts that the host cannot check (device data,
never read: Rule 3): two tokens of one call do not share a non-negative slot; every position is a row of the table.  A NaN or
infinite element makes its own head row non-finite; every other row is unaffected.

Launch knobs, default off, neither changes a byte.  ``pdl=True``: the launch carries the programmatic-stream-serialization attribute
and the kernel waits for its stream predecessor AFTER it has loaded its weights and BEFORE it reads any other operand or writes
anything.  CALLER OBLIGATION: ``q_weight`` / ``k_weight`` are not written by any launch that may still be running when this one
starts (they are constants of the layer).  ``pdl_release`` (``None`` follows ``pdl``; ``True`` needs ``pdl=True``): the kernel
releases its dependents as its FIRST statement, in front of its own wait.  CALLER OBLIGATION: every launch that can start before this
one completes waits for it before reading ``q_out``, ``gate_out`` or the pools and before writing any operand of this call.  The
launch runs on ``torch.cuda.current_stream`` of the entry's device, under that device, and the caller's current device is restored
behind it (Rule 5).

Output ownership is PER OUTPUT.  ``q_out`` / ``gate_out`` the caller passes are the caller's and are returned as given (rows past
the call's tokens are untouched).  An output the caller does not pass is the entry's: one pair per token count, allocated by
``compile(rows)`` / ``outputs(rows)`` or at the first call of that token count -- call ``compile(rows)`` for every token count before
a stream capture.  Entry-owned outputs are valid until the entry's next call at that token count from ANY host thread.  The pools
are always the caller's.  Workspace: none.  The warm path allocates nothing and reads no device memory to the host.
"""

from math import gcd
from typing import Any, Dict, Optional, Tuple

from cudnn._torch_stream import device_context  # imports no framework at module level
from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error, cutedsl_state
from cudnn.frost.launch_facts import LaunchShape, kernel_symbols, plain_form

from .kernel.prepare_decode_config import (
    ELEMENT_BYTES,
    INT_BYTES,
    MAX_ROWS,
    NAME,
    SERVED_GEOMETRIES,
    SERVED_PAGE_SIZES,
    AttentionPrepareGeometry,
    Cfg,
    validate_cfg,
    validate_eps,
    validate_geometry,
    validate_int_dtype,
    validate_page_size,
    validate_positions,
    validate_rows,
)

#: the served compute capabilities as closed ranges of ``10 * major + minor``: SM100-class parts only.
SUPPORTED_CC_RANGES = ((100, 109),)
SUPPORTED_CC_TEXT = "10.x (SM100-class)"
DSL_FLOOR = (4, 8, 0)
DSL_FLOOR_TEXT = ".".join(str(x) for x in DSL_FLOOR)
#: cc 10.7 only: the oldest DSL compiler observed to assemble ``sm_107a`` (the CUDA 13 flavour of DSL 4.8.0); CUDA 12.9 rejects the target.
SM107_MIN_COMPILER = (13, 4)
ALIGN_BYTES = 16  # every bf16 operand: a whole 16-byte vector at its base and per outer stride
#: the two output slots of a call, in the order of the returned tuple
OUTPUT_SLOTS = ("q_out", "gate_out")
#: the operands of a call, in call order
OPERAND_NAMES = ("q", "gate", "k", "v", "q_weight", "k_weight", "cos_sin", "positions", "k_pool", "v_pool", "slot_mapping", "q_out", "gate_out")
#: the operands a launch writes
WRITTEN = ("q_out", "gate_out", "k_pool", "v_pool")


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
    """The Rule-7 compiler gate.  On cc 10.0 / 10.3 the kernel's PTX body is the same text under both toolchain flavours of DSL 4.8.0,
    so both are served.  cc 10.7 is not: the CUDA 12.9 flavour rejects the ``sm_107a`` target at compile time although the DSL's target
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


def _knob(name: str, value):
    """An optional launch knob as given when it is a ``bool`` or ``None``; ``TypeError`` otherwise."""
    if value is None or isinstance(value, bool):
        return value
    raise TypeError(f"{NAME}: {name} must be a bool or None, got {type(value).__name__}")


def make_cfg(geometry, eps, *, positions, page_size, position_dtype, slot_dtype, mrope_section=None, pdl=None, pdl_release=None) -> Cfg:
    """The validated config of the entry's constructor arguments (the construction declines, without torch): ``TypeError`` for a
    ``geometry`` that is not an :class:`AttentionPrepareGeometry` of ``int`` fields or a launch knob that is not a ``bool``;
    ``ValueError`` for a non-positive geometry field, an ``eps`` that is not finite and positive, a ``positions`` outside the
    vocabulary, a missing, misplaced or inconsistent ``mrope_section``, a ``page_size`` that is not a positive power-of-two ``int``, an
    integer dtype name outside ``("int32", "int64")``, or ``pdl_release=True`` without ``pdl=True``; ``NotImplementedError`` for a
    geometry or a page size the entry does not serve.  ``pdl`` ``None`` is off; ``pdl_release`` ``None`` follows ``pdl``."""
    geometry = validate_geometry(geometry)
    eps = validate_eps(eps)
    positions, section = validate_positions(positions, mrope_section, geometry.ROTARY_DIM // 2)
    page_size = validate_page_size(page_size)
    position_dtype = validate_int_dtype("position_dtype", position_dtype)
    slot_dtype = validate_int_dtype("slot_dtype", slot_dtype)
    pdl, pdl_release = _knob("pdl", pdl), _knob("pdl_release", pdl_release)
    wait = bool(pdl)
    return validate_cfg(
        Cfg(
            GEOMETRY=geometry,
            EPS=eps,
            POSITIONS=positions,
            SECTION=section,
            PAGE_SIZE=page_size,
            POSITION_DTYPE=position_dtype,
            SLOT_DTYPE=slot_dtype,
            PDL=wait,
            PDL_RELEASE=wait if pdl_release is None else pdl_release,
        )
    )


def _paged_contract():
    """``(check_pool, check_slot_mapping)`` of the paged-pool contract, imported where a call needs them (the module imports torch;
    its package imports the CuTe DSL).  On a host without the DSL the import fails: that is this step's typed decline of a missing
    DSL, raised here with the version gate's text."""
    try:
        from cudnn.gated_attention_block.cache_write_contract import check_pool, check_slot_mapping
    except ImportError as err:
        msg = _dsl_version_error() or f"{NAME}: the paged-pool contract (cudnn.gated_attention_block.cache_write_contract) could not be imported: {err}"
        raise NotImplementedError(msg) from err
    return check_pool, check_slot_mapping


# ---------------------------------------------------------------------------------------------------------------------------
# the footprint of an operand and the overlap rule
# ---------------------------------------------------------------------------------------------------------------------------
def _footprint(t, shape=None) -> Tuple[int, int, int, int]:
    """``(start, end, run, period)`` in bytes of the storage a tensor view addresses, with ``shape`` in place of its own (the live
    tokens of an output): ``[start, end)`` bounds every byte; ``run`` is the length of one contiguous run (the trailing axes that
    are laid out compactly); ``period`` is the greatest common divisor of the byte strides of the axes in front of that run that
    have more than one entry (0: the whole view is ONE run).  Every byte of the view lies in ``start + period * Z + [0, run)``."""
    shape = tuple(int(s) for s in (t.shape if shape is None else shape))
    strides = tuple(int(s) for s in t.stride())
    size = t.element_size()
    start = t.data_ptr()
    if 0 in shape:
        return start, start, 0, 0
    run, axis = 1, len(shape)
    while axis > 0 and (shape[axis - 1] == 1 or strides[axis - 1] == run):
        run *= shape[axis - 1]
        axis -= 1
    period = 0
    for extent, stride in zip(shape[:axis], strides[:axis]):
        if extent > 1:
            period = gcd(period, stride * size)
    end = start + (sum((extent - 1) * stride for extent, stride in zip(shape, strides)) + 1) * size
    return start, end, run * size, period


def _share_a_byte(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    """Whether two footprints (:func:`_footprint`) CAN share a byte.  False only when that is proven: the bounding ranges are
    disjoint, or -- for strided views interleaved in one allocation, the two halves of a packed cache -- every byte of one lies in a
    residue window modulo the common period that no byte of the other reaches.  A layout the two tests cannot separate is treated as
    overlapping (the call is declined, never launched on a guess)."""
    a0, a1, a_run, a_period = a
    b0, b1, b_run, b_period = b
    if a0 >= a1 or b0 >= b1 or a1 <= b0 or b1 <= a0:
        return False
    period = gcd(a_period, b_period)
    if period == 0 or a_run >= period or b_run >= period:
        return True
    lo, hi = sorted((a0 % period, b0 % period))
    first, second = (a_run, b_run) if a0 % period <= b0 % period else (b_run, a_run)
    return not (lo + first <= hi and hi + second <= lo + period)


def _own_rows_overlap(t) -> Optional[Tuple[int, int, int]]:
    """``(axis, stride, span)`` in elements of the first axis of a pool view whose step is shorter than what the axes inside it
    span -- two rows of the view then CAN share a byte -- or ``None`` when its rows are proven disjoint.  The axes with more than
    one entry are walked from the smallest stride outwards, starting from the contiguous last axis: each step must clear the span
    of everything inside it.  An expanded axis (stride 0) and rows that overlap by a part fail it.  A launch writes a pool row from
    one thread block per head, so two rows that share a byte are a write race the slot contract cannot exclude.  A layout the walk
    cannot separate is treated as overlapping (the call is declined, never launched on a guess)."""
    shape = tuple(int(s) for s in t.shape)
    strides = tuple(int(s) for s in t.stride())
    span = shape[-1]  # the paged contract has checked the last axis: contiguous
    for stride, extent, axis in sorted((strides[a], shape[a], a) for a in range(len(shape) - 1) if shape[a] > 1):
        if stride < span:
            return axis, stride, span
        span += (extent - 1) * stride
    return None


def _written_shape(cfg: Cfg, rows: int, name: str, t) -> Optional[Tuple[int, ...]]:
    """The shape of the part of an operand a launch of ``rows`` tokens can touch: the live tokens of an output, the whole tensor
    otherwise (which pool rows are written is device data)."""
    return (rows, cfg.HEADS, cfg.HEAD_DIM) if name in OUTPUT_SLOTS else None


def _check_overlap(cfg: Cfg, rows: int, tensors: Dict[str, Any], written, what: str = "") -> None:
    prints = {name: _footprint(t, _written_shape(cfg, rows, name, t)) for name, t in tensors.items()}
    for name in written:
        if name not in prints:
            continue
        for other, footprint in prints.items():
            if other != name and _share_a_byte(prints[name], footprint):
                raise ValueError(f"{NAME}: {what}{name} must not overlap {other}")


def check_binding(cfg: Cfg, q, gate, k, v, q_weight, k_weight, cos_sin, positions, k_pool, v_pool, slot_mapping, q_out=None, gate_out=None, device=None) -> int:
    """The binding checks of a call, run before every launch (compile-cache hits included).  ``TypeError`` for a non-tensor, a dtype
    of a bf16 operand, or a ``positions`` / ``slot_mapping`` tensor whose integer dtype is not the entry's; ``ValueError`` for an
    inconsistent device, shape, stride, alignment, an empty call, a slot vector longer than the call, a written operand that
    overlaps another operand, a pool whose own rows share a byte (an expanded axis, rows that overlap by a part: two thread blocks
    of one launch would write one address), and everything the paged contract declines about the pools and the slot vector
    (``cudnn.gated_attention_block.cache_write_contract``: its ``ValueError`` also for a non-tensor and a dtype);
    ``NotImplementedError`` for a token count above the served ceiling, a ``cos_sin`` that is not bf16, an operand that is not
    contiguous where the kernel addresses flat, and a host without the CuTe DSL (the paged contract's package imports it).  Returns
    the token count ``q.shape[0]``.  ``device`` (an entry's: a concrete CUDA
    ordinal) requires every operand on exactly that device; without it the operands must share ``q``'s.  Host-only metadata
    arithmetic, no device work: no operand's VALUE is read.  The checks see the operands they are given: an output that is not
    passed is resolved by the entry, whose call checks its own buffers against these operands."""
    import torch

    heads, kv_heads, d, rotary = cfg.HEADS, cfg.KV_HEADS, cfg.HEAD_DIM, cfg.ROTARY_DIM
    bf16 = torch.bfloat16
    given = dict(q=q, gate=gate, k=k, v=v, q_weight=q_weight, k_weight=k_weight, cos_sin=cos_sin, positions=positions, q_out=q_out, gate_out=gate_out)
    tensors = {name: t for name, t in given.items() if not (t is None and name in OUTPUT_SLOTS)}
    for name, t in tensors.items():
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{NAME}: {name} must be a torch.Tensor, got {type(t).__name__}")
    want = resolve_device(device) if device is not None else q.device
    whose = "the entry's" if device is not None else "q's"
    for name, t in tensors.items():
        if t.device != want:
            raise ValueError(f"{NAME}: every operand must live on one device, {whose} ({want}); {name} is on {t.device}")
    for name in ("q", "gate", "k", "v", "q_weight", "k_weight", "q_out", "gate_out"):
        if name in tensors and tensors[name].dtype != bf16:
            raise TypeError(f"{NAME}: {name} must be {bf16} (kernel-native; callers convert), got {tensors[name].dtype}")
    if cos_sin.dtype != bf16:
        raise NotImplementedError(
            f"{NAME} serves a {bf16} cos_sin table (the products of a rotation are then exact in fp32; a table of another dtype is "
            f"another byte class); got {cos_sin.dtype}"
        )
    int_dtypes = {"int32": torch.int32, "int64": torch.int64}
    if positions.dtype != int_dtypes[cfg.POSITION_DTYPE]:
        raise TypeError(
            f"{NAME}: positions must be {int_dtypes[cfg.POSITION_DTYPE]} (the entry's position_dtype: a compile-time load width), got {positions.dtype}"
        )

    # shapes
    if q.dim() != 3 or tuple(q.shape[1:]) != (heads, d):
        raise ValueError(f"{NAME}: q must be [tokens, {heads}, {d}]; got {tuple(q.shape)}")
    rows = int(q.shape[0])
    if rows < 1:
        raise ValueError(f"{NAME}: a call holds at least one token; got q {tuple(q.shape)}")
    if rows > MAX_ROWS:
        raise NotImplementedError(f"{NAME} serves at most {MAX_ROWS} rows (tokens) per call; got {rows}")
    for name, h in (("gate", heads), ("k", kv_heads), ("v", kv_heads)):
        if tuple(tensors[name].shape) != (rows, h, d):
            raise ValueError(f"{NAME}: {name} must be [{rows}, {h}, {d}] (q holds {rows} tokens); got {tuple(tensors[name].shape)}")
    for name in ("q_weight", "k_weight"):
        if tuple(tensors[name].shape) != (d,):
            raise ValueError(f"{NAME}: {name} must be [{d}]; got {tuple(tensors[name].shape)}")
    if cos_sin.dim() != 2 or int(cos_sin.shape[1]) != rotary or int(cos_sin.shape[0]) < 1:
        raise ValueError(
            f"{NAME}: cos_sin must be [positions, {rotary}] ({rotary // 2} cosines, then {rotary // 2} sines per position); got {tuple(cos_sin.shape)}"
        )
    want_positions = (rows,) if cfg.POSITIONS == "1d" else (3, rows)
    if tuple(positions.shape) != want_positions:
        raise ValueError(f"{NAME}: positions must be {list(want_positions)} under positions={cfg.POSITIONS!r}; got {tuple(positions.shape)}")
    for name in OUTPUT_SLOTS:
        t = tensors.get(name)
        if t is not None and (t.dim() != 3 or int(t.shape[0]) < rows or tuple(t.shape[1:]) != (heads, d)):
            raise ValueError(f"{NAME}: {name} must be [>= {rows}, {heads}, {d}]; got {tuple(t.shape)}")

    # layouts: the kernel addresses a head row flat
    for name in ("q", "gate", "k", "v"):
        strides = tuple(int(s) for s in tensors[name].stride())
        if strides[2] != 1:
            raise NotImplementedError(
                f"{NAME} needs {name} contiguous along its last axis (a head row is addressed flat; no copy is made); got strides {strides}"
            )
        for axis, stride in enumerate(strides[:2]):
            if (stride * ELEMENT_BYTES) % ALIGN_BYTES:
                raise ValueError(
                    f"{NAME}: {name} stride[{axis}] = {stride} elements ({stride * ELEMENT_BYTES} bytes) must be a multiple of {ALIGN_BYTES} bytes "
                    f"(a head row is whole 16-byte vectors)"
                )
    for name in ("q_weight", "k_weight", "cos_sin", "q_out", "gate_out"):
        if name in tensors and not tensors[name].is_contiguous():
            raise NotImplementedError(
                f"{NAME} needs {name} contiguous (the kernel addresses it by flat element; no copy is made); got strides {tuple(tensors[name].stride())}"
            )
    if int(positions.stride(-1)) != 1:
        raise NotImplementedError(
            f"{NAME} needs positions contiguous along its token axis (the row stride of a [3, tokens] tensor is free); got strides {tuple(positions.stride())}"
        )
    for name in ("q", "gate", "k", "v", "q_weight", "k_weight", "cos_sin", "q_out", "gate_out"):
        if name in tensors and tensors[name].data_ptr() % ALIGN_BYTES:
            raise ValueError(
                f"{NAME}: {name} must start at a {ALIGN_BYTES}-byte aligned address, got data_ptr % {ALIGN_BYTES} = {tensors[name].data_ptr() % ALIGN_BYTES}"
            )
    if positions.data_ptr() % positions.element_size():
        raise ValueError(f"{NAME}: positions must be aligned to its {positions.element_size()}-byte element, got data_ptr {positions.data_ptr():#x}")

    # the pools and the slot vector: the paged contract's own checks, with the ENTRY's page size
    check_pool, check_slot_mapping = _paged_contract()
    check_pool(k_pool, "k_pool", h=kv_heads, page_size=cfg.PAGE_SIZE, d=d, dtype=bf16, device=want)
    check_pool(v_pool, "v_pool", h=kv_heads, page_size=cfg.PAGE_SIZE, d=d, dtype=bf16, device=want)
    for name, pool in (("k_pool", k_pool), ("v_pool", v_pool)):  # a written operand: no two of its own rows may share a byte
        clash = _own_rows_overlap(pool)
        if clash is not None:
            raise ValueError(
                f"{NAME}: {name} must not overlap itself: stride[{clash[0]}] = {clash[1]} elements is shorter than the {clash[2]} elements "
                f"the axes inside it span, so two of its rows share bytes (two thread blocks of one launch would write one address)"
            )
    live = int(slot_mapping.numel()) if isinstance(slot_mapping, torch.Tensor) and slot_mapping.dim() == 1 else None
    if live is not None and live > rows:
        raise ValueError(f"{NAME}: slot_mapping holds at most one slot per token ({rows}); got {live} entries")
    check_slot_mapping(slot_mapping, "slot_mapping", t=rows, device=want, t_live=live)
    if slot_mapping.dtype != int_dtypes[cfg.SLOT_DTYPE]:
        raise TypeError(
            f"{NAME}: slot_mapping must be {int_dtypes[cfg.SLOT_DTYPE]} (the entry's slot_dtype: a compile-time load width), got {slot_mapping.dtype}"
        )
    if live and slot_mapping.data_ptr() % slot_mapping.element_size():
        raise ValueError(f"{NAME}: slot_mapping must be aligned to its {slot_mapping.element_size()}-byte element, got data_ptr {slot_mapping.data_ptr():#x}")

    _check_overlap(cfg, rows, dict(tensors, k_pool=k_pool, v_pool=v_pool, slot_mapping=slot_mapping), WRITTEN)
    return rows


def check_support(
    cfg: Cfg,
    q,
    gate,
    k,
    v,
    q_weight,
    k_weight,
    cos_sin,
    positions,
    k_pool,
    v_pool,
    slot_mapping,
    q_out=None,
    gate_out=None,
    device_cc=None,
    device=None,
) -> int:
    """Typed declines without a DSL import: the binding checks of :func:`check_binding` plus the architecture gate
    (``NotImplementedError`` for operands off a CUDA device or a part this kernel does not serve).  ``device_cc`` overrides the
    compute-capability probe (tests).  The DSL gates (installed, version floor, the part's target, its compiler) run in
    :meth:`AttentionPrepareDecode.compile`."""
    import torch

    rows = check_binding(validate_cfg(cfg), q, gate, k, v, q_weight, k_weight, cos_sin, positions, k_pool, v_pool, slot_mapping, q_out, gate_out, device=device)
    dev = q.device
    if device_cc is None and dev.type != "cuda":
        raise NotImplementedError(f"{NAME} serves CUDA tensors only; the operands are on {dev}")
    cc = tuple(torch.cuda.get_device_capability(dev)) if device_cc is None else tuple(device_cc)
    if not cc_supported(cc):
        raise NotImplementedError(f"{NAME} serves compute capability {SUPPORTED_CC_TEXT}; this device is {cc[0]}.{cc[1]}")
    return rows


def launch_arguments(
    cfg: Cfg, rows: int, q, gate, k, v, q_weight, k_weight, cos_sin, positions, k_pool, v_pool, slot_mapping, q_out, gate_out
) -> Tuple[int, ...]:
    """The kernel's runtime arguments of a call whose binding was checked, as plain ``int`` in launch order: the thirteen operand
    ADDRESSES, the fifteen byte strides (per token and per head of ``q`` / ``gate`` / ``k`` / ``v``; between the position rows; per
    page, head and page row of the two pools), the slot vector's length, the pools' capacity in slots (the smaller pool's) and the
    token count.  Host-only metadata: nothing is read from the device and nothing is allocated."""
    size = ELEMENT_BYTES
    addresses = [t.data_ptr() for t in (q, gate, k, v, q_weight, k_weight, cos_sin, positions, k_pool, v_pool, slot_mapping, q_out, gate_out)]
    strides = []
    for t in (q, gate, k, v):
        strides += [int(t.stride(0)) * size, int(t.stride(1)) * size]
    strides.append(int(positions.stride(0)) * INT_BYTES[cfg.POSITION_DTYPE] if cfg.POSITIONS == "3d" else 0)
    for pool in (k_pool, v_pool):
        strides += [int(pool.stride(0)) * size, int(pool.stride(1)) * size, int(pool.stride(2)) * size]
    pool_slots = min(int(k_pool.shape[0]), int(v_pool.shape[0])) * cfg.PAGE_SIZE
    return (*addresses, *strides, int(slot_mapping.numel()), pool_slots, int(rows))


class AttentionPrepareDecode:
    """The step in ONE form, compiled once and shared by every layer and token count: all tensors are launch arguments, the token
    count is a runtime argument.  ``geometry`` is an :class:`AttentionPrepareGeometry`; ``eps`` the norm's epsilon (a compile-time
    constant); ``positions`` the position form (``"1d"`` | ``"3d"``, required: the forms are peers; ``"3d"`` needs
    ``mrope_section = (t, h, w)`` with ``t + h + w == ROTARY_DIM / 2``); ``page_size`` the rows of one page of both pools;
    ``position_dtype`` / ``slot_dtype`` the integer dtypes of ``positions`` / ``slot_mapping`` by name (``"int32"`` | ``"int64"``);
    ``pdl`` / ``pdl_release`` the launch knobs (the module docstring).  The device is resolved to a concrete CUDA ordinal at
    construction, and every operand of every launch must live on exactly it.  An entry built for a non-CUDA device (host tests) runs
    every decline; its compile raises the typed decline of the device.  What a token count launches is readable without a launch:
    :meth:`kernel_symbols_for`, :meth:`form_for` and :meth:`launch_shape_for` (the launch facts of
    :mod:`cudnn.frost.launch_facts`)."""

    def __init__(
        self,
        device,
        geometry: AttentionPrepareGeometry,
        eps: float,
        *,
        positions: str,
        page_size: int,
        position_dtype: str,
        slot_dtype: str,
        mrope_section: Optional[Tuple[int, int, int]] = None,
        pdl: Optional[bool] = None,
        pdl_release: Optional[bool] = None,
    ):
        self.cfg: Cfg = make_cfg(
            geometry,
            eps,
            positions=positions,
            page_size=page_size,
            position_dtype=position_dtype,
            slot_dtype=slot_dtype,
            mrope_section=mrope_section,
            pdl=pdl,
            pdl_release=pdl_release,
        )
        self.device = resolve_device(device)
        self.compiled: Dict[tuple, Any] = {}
        self.buffers: Dict[int, Dict[str, Any]] = {}

    # ---- the form and the compile key ----
    def cfg_for(self, rows: int) -> Cfg:
        """The config a call of ``rows`` tokens runs: the entry's ONE config at every served token count (a count outside
        1 .. 1024 raises its typed decline)."""
        validate_rows(rows)
        return self.cfg

    def key(self) -> tuple:
        """The compile key: plan-time facts only -- the config.  No operand enters the compiled kernel: it takes addresses and byte
        strides as runtime arguments and the token count as its grid."""
        return (self.cfg,)

    # ---- launch facts (cudnn.frost.launch_facts): three read-only answers per served token count ----
    def _fact_cfg(self, rows) -> Cfg:
        """The config of the token count a launch fact is asked for: ``ValueError`` for a ``rows`` that is not an ``int`` (the decline
        every entry's launch facts give a malformed row count), then the typed declines of :meth:`cfg_for`."""
        if isinstance(rows, bool) or not isinstance(rows, int):
            raise ValueError(f"{NAME}: rows must be an int, got {rows!r}")
        return self.cfg_for(rows)

    def kernel_symbols_for(self, rows: int) -> Tuple[str, ...]:
        """The symbol of the kernel this entry has compiled: a one-element tuple, empty until the entry is compiled.  One compiled
        kernel serves every token count.  The symbol carries the form (``str`` of the config: the geometry, the position form with
        its section, the page size, the two integer dtypes, the launch arm, the fp32 bits of ``eps``) directly behind the kernel's
        name, so two forms never share a symbol; the ``eps`` text is the fp32 value the kernel adds."""
        self._fact_cfg(rows)
        fn = self.compiled.get(self.key())
        return kernel_symbols(() if fn is None else (fn,), NAME)

    def form_for(self, rows: int) -> Dict[str, Any]:
        """The form of ``rows`` as plain data: the fields of the entry's config, the geometry as a nested ``dict``.  The compile key
        is the config alone, so nothing else is part of it: the token count is a runtime fact of a launch
        (:meth:`launch_shape_for`).  ``EPS`` is the caller's ``float`` as given."""
        return plain_form(self._fact_cfg(rows))

    def launch_shape_for(self, rows: int) -> LaunchShape:
        """``(grid, block, shared memory)`` of the launch of ``rows`` tokens: one thread block per (token, head) over the query and
        the key / value heads, ``HEAD_DIM / 2`` threads, and the kernel's one shared-memory buffer (an fp32 slot per warp)."""
        cfg = self._fact_cfg(rows)
        return LaunchShape((int(rows), cfg.PROGRAMS, 1), (cfg.THREADS, 1, 1), cfg.SMEM_BYTES)

    # ---- entry-owned outputs ----
    def outputs(self, rows: int) -> Dict[str, Any]:
        """The entry-owned output buffers of ``rows`` tokens: ``q_out`` and ``gate_out``, bf16 ``[rows, HEADS, HEAD_DIM]``.
        Allocated once per token count; see the module docstring for their lifetime."""
        rows = validate_rows(rows)
        if rows not in self.buffers:
            import torch

            shape = (rows, self.cfg.HEADS, self.cfg.HEAD_DIM)
            self.buffers[rows] = {name: torch.empty(shape, dtype=torch.bfloat16, device=self.device) for name in OUTPUT_SLOTS}
        return self.buffers[rows]

    def _check_entry_owned(self, rows: int, operands: Dict[str, Any]) -> None:
        """The overlap check of :func:`check_binding` for the outputs a call does not pass: the launch writes the entry's own
        buffers of ``rows`` in their place, and a tensor an earlier call returned can come back as an operand.  ``ValueError``
        when such a buffer overlaps another operand of the call.  Before the token count's buffers exist there is nothing to
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
                f"{err} (the output was not passed, so the launch writes the entry's own buffer of {rows} token(s), the tensor earlier "
                f"calls of that token count returned; pass a caller-owned output)"
            ) from None

    # ---- compile ----
    def compile(self, rows: Optional[int] = None) -> None:
        """Plan time: the typed declines of the device, the part and the DSL, then ONE trace-compile of the entry's form (a no-op
        once compiled).  With ``rows`` it also allocates the entry-owned outputs of that token count -- call it for every token
        count before a stream capture.  Needs no operand."""
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

            from .kernel import prepare_decode_f16 as mod

            with device_context(self.device):  # a compiled launchable belongs to its device's context
                # plan-time arguments: addresses and strides are runtime values, the zeros pin their TYPES only
                args = [cutlass.Int64(0)] * (len(mod.OPERANDS) + len(mod.STRIDES)) + [cutlass.Int32(0), cutlass.Int64(0), cutlass.Int32(1)]
                self.compiled[self.key()] = cute.compile(mod.launch, self.cfg, *args, make_fake_stream())
        if rows is not None:
            self.outputs(rows)

    # ---- launch ----
    def __call__(self, q, gate, k, v, q_weight, k_weight, cos_sin, positions, k_pool, v_pool, slot_mapping, q_out=None, gate_out=None):
        """Launch on torch's current stream of the entry's device; returns ``(q_out, gate_out)`` -- per slot the caller's tensor as
        given, else the entry-owned buffer of this token count.  Compiles on first use.  Every launch re-runs the binding checks
        before the compiled kernel is looked up, the overlap check on the RESOLVED operands (``ValueError``: an entry-owned output
        the launch would write must not overlap another operand; a returned output fed back in needs a caller-owned output).  The
        launch is scoped to the entry's device and the caller's current device is restored behind it."""
        import torch

        cfg = self.cfg
        rows = check_binding(cfg, q, gate, k, v, q_weight, k_weight, cos_sin, positions, k_pool, v_pool, slot_mapping, q_out, gate_out, device=self.device)
        operands = dict(
            q=q,
            gate=gate,
            k=k,
            v=v,
            q_weight=q_weight,
            k_weight=k_weight,
            cos_sin=cos_sin,
            positions=positions,
            k_pool=k_pool,
            v_pool=v_pool,
            slot_mapping=slot_mapping,
            q_out=q_out,
            gate_out=gate_out,
        )
        self._check_entry_owned(rows, operands)
        key = self.key()
        if key not in self.compiled:
            self.compile()
        import cuda.bindings.driver as cuda  # behind compile(): its gates decline a host without the DSL by type first
        import cutlass

        with device_context(self.device):
            if q_out is None or gate_out is None:
                owned = self.outputs(rows)
                q_out = owned["q_out"] if q_out is None else q_out
                gate_out = owned["gate_out"] if gate_out is None else gate_out
            values = launch_arguments(cfg, rows, q, gate, k, v, q_weight, k_weight, cos_sin, positions, k_pool, v_pool, slot_mapping, q_out, gate_out)
            wide = [cutlass.Int64(value) for value in values[:-3]]
            self.compiled[key](
                *wide,
                cutlass.Int32(values[-3]),
                cutlass.Int64(values[-2]),
                cutlass.Int32(values[-1]),
                cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream),
            )
        return q_out, gate_out


__all__ = [
    "AttentionPrepareDecode",
    "AttentionPrepareGeometry",
    "check_binding",
    "check_support",
    "cc_supported",
    "launch_arguments",
    "make_cfg",
    "SERVED_GEOMETRIES",
    "SERVED_PAGE_SIZES",
]
