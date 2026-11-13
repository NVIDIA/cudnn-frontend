# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host entry of the gate / up product of an FP4 MLP for decode rows (``kernel/swiglu_fp4_decode_nvfp4.py``): validation with typed
declines, the plan-time compile, the launch and the load-time row interleave.  No DSL import at module level -- the CuTe DSL is
imported inside :meth:`SwigluFp4Decode.compile` after the version / architecture / compiler / name gate (python/cudnn/AGENTS.md
Rule 7), so a host without it can still construct the entry and run its declines.  The binding checks (:func:`check_binding`: type,
dtype, device, shape, byte count, contiguity and alignment of every operand) run before EVERY launch, compile-cache hits included: a
compiled DSL entry accepts a tensor of another dtype, footprint or device silently, so the host is a launch's only guard.

The op, for ``T`` = 1 .. 64 token rows (``H`` = hidden, ``I`` = intermediate)::

    gate[t, c] = bf16(alpha * sum_k activation[t, k] * weight[row_gate(c), k])       up[t, c] likewise with row_up(c)
    v[t, c]    = bf16(silu(gate[t, c]) * up[t, c])
    per block of 16 channels of a row: byte = E4M3(gs * (amax * fp32(1/6))), encode = 0 for a zero byte, else rcp(value * rcp(gs))
    out[t, c] = E2M1(v[t, c] * encode)  (two codes per byte, low nibble = even channel);   out_scale[t, block] = byte

``weight`` ``[2 I, H / 2]`` and ``activation`` ``[T, H / 2]`` are packed FP4 (E2M1) codes, their scales one E4M3 byte per (row, 16
columns) in the 128 x 4 scale-atom order; ``alpha`` and ``out_global_scale`` (``gs``) hold ONE fp32 element each and are read on the
device, never on the host; ``out`` ``[T, I / 2]`` and ``out_scale`` are caller-owned and required.  The weight rows are INTERLEAVED
in tiles of 64 gate + 64 up channels: :func:`interleave_gate_up` builds that layout once per weight, at load time.  fp32 accumulation
in one fixed order per element; the rounding chain is in the kernel module's docstring.

NUMERICS.  There is no numerics knob: no split, no reduction across rows.  A row's bytes are the same alone and among up to 64 rows,
at every token tile and under every ``Cfg`` field.  The token tile is therefore a performance pick by the row count (the narrowest of
8 / 16 / 32 / 64 that holds the rows, not below the constructor config's ``TOKEN_TILE``).

Served (declined otherwise, by name): the validated geometries, rows 1 .. 64, contiguous operands with aligned storage on the entry's
device, a compute capability 10.x part, a CuTe DSL >= 4.8.0 compiling with CUDA 13.4 or later that has the part's target and the
names the kernel is written on.  ``out`` / ``out_scale`` overlapping an input is undefined (not checked).

Launch knob: ``pdl`` (default off) puts the programmatic-stream-serialization attribute on every launch.  AS A DEPENDENT the kernel
then waits for its predecessor before it reads ``activation``, ``activation_scale`` and ``out_global_scale``; ``alpha`` and (with the
config's two pre-wait stages) the first weight tiles are read ahead of that wait.  CALLER OBLIGATION: no kernel that can still be
running when the launch starts -- the immediate stream predecessor and every earlier launch of an unbroken chain of early releases --
writes ``weight``, ``weight_scale`` or ``alpha``.  AS A PREDECESSOR a thread block signals its dependents when its loader warp has
finished, BEFORE the block's last output tile is written: a kernel launched behind this entry with the attribute may start while
``out`` and ``out_scale`` are still being written and must issue its own wait before it reads them; a kernel launched without the
attribute starts after this one has completed.  There is no release knob: the release point is the kernel's own.

Rules 1-5 on the execute path: views only (``view(torch.uint8)``, ``reshape(-1)`` of a contiguous tensor), no conversion, no copy, no
device allocation, no device read; the launch goes onto torch's current stream of the entry's device.  The entry owns no device memory.
"""

from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error, cutedsl_state
from cudnn.frost.launch_facts import LaunchShape, kernel_symbols, plain_form

from .kernel.swiglu_fp4_decode_config import (
    DEFAULT_GEOMETRY,
    DSL_NAMES,
    FLAT_ALIGN_BYTES,
    NAME,
    PACKED_ALIGN_BYTES,
    QUALIFIED_MIN_COMPILER,
    READING_FIELDS,
    SCALAR_ALIGN_BYTES,
    SF_ATOM_ROWS,
    SF_VEC,
    SM100_SM_COUNT,
    SM100_SMEM_OPTIN_BYTES,
    SM100_SMEM_PER_SM_BYTES,
    THREADS_PER_CTA,
    TOKEN_TILES,
    WEIGHT_TILE_ROWS,
    Cfg,
    Geometry,
    check_geometry,
    check_rows_fit,
    form_of,
    grid,
    part_readings,
    smem_bytes,
    token_tile_for,
    validate_cfg,
)

#: the served compute capabilities as closed ranges of ``10 * major + minor``: SM100-class parts (tcgen05 and tensor memory)
SUPPORTED_CC_RANGES = ((100, 109),)
SUPPORTED_CC_TEXT = "10.x (SM100-class)"
DSL_FLOOR = (4, 8, 0)
DSL_FLOOR_TEXT = ".".join(str(x) for x in DSL_FLOOR)


def cc_supported(cc) -> bool:
    """Whether a ``(major, minor)`` compute capability is one this kernel serves; ``None`` (no CUDA device) is not."""
    if cc is None:
        return False
    code = 10 * int(cc[0]) + int(cc[1])
    return any(lo <= code <= hi for lo, hi in SUPPORTED_CC_RANGES)


def dsl_names_error() -> Optional[str]:
    """The name probe: every name of ``DSL_NAMES`` must import.  Imports the DSL (callers run it after the version gate, right before
    the kernel module's own import).  None when every name is there, else a message naming the first missing piece."""
    import importlib

    _, version = cutedsl_state()
    found = " ".join(version) if version else "unknown version"
    for module, names in DSL_NAMES:
        try:
            mod = importlib.import_module(module)
        except Exception as e:  # noqa: BLE001 - any failure to reach the module is the same answer
            return f"{NAME} is written on {module}, which does not import on the installed CuTe DSL ({found}) ({type(e).__name__}: {e})"
        missing = [n for n in names if not hasattr(mod, n)]
        if missing:
            return f"{NAME} is written on {module}; the installed CuTe DSL ({found}) does not export {', '.join(missing)}"
    return None


def dsl_compiler_version() -> Optional[Tuple[int, int]]:
    """``(major, minor)`` of the CUDA compiler the installed CuTe DSL is bound to in this process, or None when the DSL does not report
    one.  Imports the DSL's version module (callers run it after the version gate)."""
    try:
        from cutlass.base_dsl.version_info import CUDA_VERSION

        return (int(CUDA_VERSION.major), int(CUDA_VERSION.minor))
    except Exception:  # noqa: BLE001 - a DSL that does not report its compiler is not declined on a value we failed to read
        return None


def dsl_compiler_error() -> Optional[str]:
    """The compiler half of the gate.  The kernel's output bytes depend on how the compiler lowers three approximate instructions and
    one constant; they were qualified with the DSL compiling with CUDA 13.4.  An older compiler is declined by what was qualified,
    never by a property of the compiler found.  None when the compiler qualifies or the DSL does not report one."""
    compiler = dsl_compiler_version()
    if compiler is None or compiler >= QUALIFIED_MIN_COMPILER:
        return None
    return (
        f"{NAME} is qualified with the CuTe DSL compiling with CUDA {QUALIFIED_MIN_COMPILER[0]}.{QUALIFIED_MIN_COMPILER[1]} or later; "
        f"the DSL in this process compiles with CUDA {compiler[0]}.{compiler[1]} -- install 'nvidia-cutlass-dsl[cu13]' and run under a "
        f"CUDA 13 driver (the DSL binds its cu13 compiler only there)"
    )


def dsl_error(device_cc=None) -> Optional[str]:
    """The Rule-7 gate, in the order a caller can act on: the DSL is installed; its version clears the kernel's floor (4.8.0, through the
    one version parser of :mod:`cudnn.frost.buffers`); it has the target of the part; the compiler it is bound to is one the kernel is
    qualified with; it provides the names the kernel is written on.  None when the kernel can be imported and compiled."""
    installed, _ = cutedsl_state()
    if not installed:
        return f"{NAME} requires nvidia-cutlass-dsl >= {DSL_FLOOR_TEXT}; the CuTe DSL is not installed"
    msg = cutedsl_requirement_error(NAME, floor=DSL_FLOOR)
    if msg is not None:
        return msg
    msg = cutedsl_arch_requirement_error(None if device_cc is None else tuple(device_cc))
    if msg is not None:
        return f"{NAME}: {msg}"
    msg = dsl_compiler_error()
    if msg is not None:
        return msg
    return dsl_names_error()


def device_readings(device) -> Tuple[int, int, int]:
    """(shared memory per SM, per-CTA opt-in maximum, multiprocessors) of a CUDA device -- the part's own readings, never constants
    carried across an architecture line (the readers of :mod:`cudnn.frost.device`)."""
    from cudnn.frost.device import resolve_device as _resolve_ordinal
    from cudnn.frost.device import shared_memory_per_block_optin

    import torch

    ordinal = _resolve_ordinal(torch.device(device).index)
    props = torch.cuda.get_device_properties(ordinal)
    per_sm = int(props.shared_memory_per_multiprocessor)
    optin = int(shared_memory_per_block_optin(ordinal))
    sm_count = int(props.multi_processor_count)
    if per_sm <= 0 or optin <= 0 or sm_count <= 0:
        raise RuntimeError(f"{NAME}: the driver reported no usable readings for device {ordinal} (per SM {per_sm} B, opt-in {optin} B, {sm_count} SMs)")
    return per_sm, optin, sm_count


def resolve_device(device):
    """``device`` as a concrete ``torch.device``: a CUDA device given without an index is the CURRENT one, so it compares with an operand's
    device (always indexed for a CUDA tensor) by type AND index; any other device is kept as given."""
    import torch

    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def _check_device(tensors, device, anchor) -> None:
    want = resolve_device(device) if device is not None else tensors[anchor].device
    whose = "the entry's" if device is not None else f"{anchor}'s"
    for name, t in tensors.items():
        if t.device != want:
            raise ValueError(f"{NAME}: every operand must live on one device, {whose} ({want}); {name} is on {t.device}")


def check_binding(
    weight, weight_scale, activation, activation_scale, alpha, out_global_scale, out, out_scale, device=None, geometry: Optional[Geometry] = None
) -> int:
    """The binding checks, run before every launch (compile-cache hits included): ``TypeError`` for a non-tensor or a dtype,
    ``ValueError`` for an inconsistent shape, byte count, alignment or device, ``NotImplementedError`` for a row count, a geometry or a
    memory layout this kernel does not serve (no copy is ever made to serve one).  Returns ``rows = activation.shape[0]``.  ``device``
    (an entry's: a concrete CUDA ordinal) requires every operand on exactly that device; without it the operands must share
    ``weight``'s.  ``geometry`` defaults to the weight's own shape.  Host-only metadata arithmetic, no device work."""
    import torch

    tensors = dict(
        weight=weight,
        weight_scale=weight_scale,
        activation=activation,
        activation_scale=activation_scale,
        alpha=alpha,
        out_global_scale=out_global_scale,
        out=out,
        out_scale=out_scale,
    )
    for name, t in tensors.items():
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{NAME}: {name} must be a torch.Tensor, got {type(t).__name__}")
    _check_device(tensors, device, "weight")
    for name in ("weight", "activation", "out"):
        if tensors[name].dtype != torch.uint8:
            raise TypeError(f"{NAME}: {name} must be torch.uint8 (two FP4 codes per byte; kernel-native, callers convert), got {tensors[name].dtype}")
    for name in ("weight_scale", "activation_scale", "out_scale"):
        if tensors[name].dtype not in (torch.float8_e4m3fn, torch.uint8):
            raise TypeError(f"{NAME}: {name} must be torch.float8_e4m3fn or torch.uint8 (one E4M3 byte per block), got {tensors[name].dtype}")
    for name in ("alpha", "out_global_scale"):
        if tensors[name].dtype != torch.float32:
            raise TypeError(f"{NAME}: {name} must be torch.float32, got {tensors[name].dtype}")
    if geometry is None:
        if weight.dim() != 2:
            raise ValueError(f"{NAME}: weight must be [2 x intermediate, hidden / 2]; got {tuple(weight.shape)}")
        geometry = Geometry(int(weight.shape[1]) * 2, max(1, int(weight.shape[0]) // 2))
    geometry = check_geometry(geometry)  # the served-geometry decline, by name
    h, i = geometry.HIDDEN, geometry.INTERMEDIATE
    if tuple(weight.shape) != (2 * i, h // 2):
        raise ValueError(f"{NAME}: weight must be [{2 * i}, {h // 2}] at {geometry.describe()}; got {tuple(weight.shape)}")
    if activation.dim() != 2 or activation.shape[1] != h // 2:
        raise ValueError(f"{NAME}: activation must be [rows, {h // 2}] at {geometry.describe()}; got {tuple(activation.shape)}")
    rows = int(activation.shape[0])
    token_tile_for(rows)  # ValueError below one row, NotImplementedError past the ceiling
    if tuple(out.shape) != (rows, i // 2):
        raise ValueError(f"{NAME}: out must be exactly [{rows}, {i // 2}] (the rows of the activation); got {tuple(out.shape)}")
    for name in ("alpha", "out_global_scale"):
        if tensors[name].numel() != 1:
            raise ValueError(f"{NAME}: {name} must hold one fp32 element; got {tuple(tensors[name].shape)}")
    want = 2 * i * h // SF_VEC
    if weight_scale.numel() != want:
        raise ValueError(f"{NAME}: weight_scale must hold exactly {want} bytes (one per weight row and {SF_VEC} columns); got {weight_scale.numel()}")
    for name, width in (("activation_scale", h), ("out_scale", i)):
        want = SF_ATOM_ROWS * width // SF_VEC
        if tensors[name].numel() < want:
            raise ValueError(
                f"{NAME}: {name} must hold at least {want} bytes (rows padded to {SF_ATOM_ROWS}: one scale-atom band of {width // SF_VEC} bytes per row); "
                f"got {tensors[name].numel()}"
            )
    for name, t in tensors.items():
        if not t.is_contiguous():
            raise NotImplementedError(f"{NAME} serves contiguous operands; {name} has shape {tuple(t.shape)} and strides {tuple(t.stride())}")
    aligns = dict(weight=PACKED_ALIGN_BYTES, activation=PACKED_ALIGN_BYTES, alpha=SCALAR_ALIGN_BYTES, out_global_scale=SCALAR_ALIGN_BYTES)
    for name, t in tensors.items():
        align = aligns.get(name, FLAT_ALIGN_BYTES)
        if t.data_ptr() % align:
            raise ValueError(f"{NAME}: {name} must start at a {align}-byte aligned address, got data_ptr % {align} = {t.data_ptr() % align}")
    return rows


def check_support(
    weight, weight_scale, activation, activation_scale, alpha, out_global_scale, out, out_scale, device_cc=None, device=None, geometry=None
) -> int:
    """Typed declines, before any DSL import: the binding checks of :func:`check_binding` plus the architecture gate
    (``NotImplementedError`` for a part this kernel does not serve).  ``device_cc`` overrides the compute-capability probe (tests)."""
    import torch

    rows = check_binding(weight, weight_scale, activation, activation_scale, alpha, out_global_scale, out, out_scale, device=device, geometry=geometry)
    dev = weight.device
    if device_cc is None and dev.type != "cuda":
        raise NotImplementedError(f"{NAME} serves CUDA tensors only; the operands are on {dev}")
    cc = tuple(torch.cuda.get_device_capability(dev)) if device_cc is None else tuple(device_cc)
    if not cc_supported(cc):
        raise NotImplementedError(f"{NAME} serves compute capability {SUPPORTED_CC_TEXT}; this device is {cc[0]}.{cc[1]}")
    return rows


def interleave_gate_up(weight, weight_scale):
    """The weight in the kernel's row order, built ONCE per weight at load time (never on the launch path).

    ``weight`` ``[2 I, H / 2]`` ``uint8`` packed FP4 codes with the gate rows ``0 .. I - 1`` first and the up rows ``I .. 2 I - 1``
    behind them; ``weight_scale`` its ``2 I x H / 16`` E4M3 scale bytes in the 128 x 4 scale-atom order (``float8_e4m3fn`` or
    ``uint8``, any shape).  Returns NEW tensors ``(weight_interleaved, weight_scale_interleaved)``: rows ``128 j .. 128 j + 63`` are
    gate channels ``64 j .. 64 j + 63``, rows ``128 j + 64 .. 128 j + 127`` the up channels of the same 64; the scale bytes are permuted
    the same way and returned flat, as ``uint8``, in the atom order of the new rows.  A byte permutation: nothing is dequantised or
    requantised.  The inputs are not modified; the two results are the caller's memory (one interleaved copy per weight).

    ``TypeError`` for a non-tensor or a dtype, ``ValueError`` for a shape the layout cannot hold (``2 I`` not a multiple of 128,
    ``H`` not a multiple of 64, another scale byte count, two devices)."""
    import torch

    for name, t in (("weight", weight), ("weight_scale", weight_scale)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{NAME}: interleave_gate_up: {name} must be a torch.Tensor, got {type(t).__name__}")
    if weight.dtype != torch.uint8:
        raise TypeError(f"{NAME}: interleave_gate_up: weight must be torch.uint8 (two FP4 codes per byte), got {weight.dtype}")
    if weight_scale.dtype not in (torch.float8_e4m3fn, torch.uint8):
        raise TypeError(f"{NAME}: interleave_gate_up: weight_scale must be torch.float8_e4m3fn or torch.uint8, got {weight_scale.dtype}")
    if weight.dim() != 2:
        raise ValueError(f"{NAME}: interleave_gate_up: weight must be [2 x intermediate, hidden / 2]; got {tuple(weight.shape)}")
    n, half_k = int(weight.shape[0]), int(weight.shape[1])
    k = half_k * 2
    if n < WEIGHT_TILE_ROWS or n % WEIGHT_TILE_ROWS or k < 64 or k % 64:
        raise ValueError(
            f"{NAME}: interleave_gate_up: the weight rows must be a multiple of {WEIGHT_TILE_ROWS} (64 gate + 64 up channels per tile) and the "
            f"columns a multiple of 64; got {n} rows and {k} columns"
        )
    if weight_scale.numel() != n * k // SF_VEC:
        raise ValueError(
            f"{NAME}: interleave_gate_up: weight_scale must hold exactly {n * k // SF_VEC} bytes (one per row and {SF_VEC} columns); got {weight_scale.numel()}"
        )
    if weight_scale.device != weight.device:
        raise ValueError(f"{NAME}: interleave_gate_up: weight is on {weight.device}, weight_scale on {weight_scale.device}")

    def interleave(rows):  # [gate rows, up rows] -> tiles of [64 gate, 64 up]
        return rows.reshape(2, n // WEIGHT_TILE_ROWS, WEIGHT_TILE_ROWS // 2, -1).permute(1, 0, 2, 3).contiguous().reshape_as(rows)

    raw = weight_scale.contiguous().view(torch.uint8).reshape(-1)
    # atom order -> one row of k / 16 bytes per weight row -> interleave the rows -> back to atoms
    linear = raw.reshape(n // 128, k // 64, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(n, k // SF_VEC)
    scales = interleave(linear).reshape(n // 128, 4, 32, k // 64, 4).permute(0, 3, 2, 1, 4).contiguous().reshape(-1)
    return interleave(weight), scales


class SwigluFp4Decode:
    """One gate / up geometry, compiled per token tile and shared by every layer of that geometry: the tensors are launch arguments,
    nothing is captured at construction and the entry owns no device memory.

    ``geometry``: the weight's ``Geometry(HIDDEN, INTERMEDIATE)`` (None = the config's, else the default).  ``cfg``: a whole ``Cfg``
    of performance knobs; its ``GEOM`` must equal ``geometry`` when both are given, and its ``TOKEN_TILE`` is a FLOOR -- a row count
    runs the narrowest token tile that holds it, not below the floor.  ``pdl``: the launch knob of the module docstring (None = the
    config's field, off by default).

    The device is resolved to a concrete CUDA ordinal at construction and every operand of every launch must live on exactly it; the
    part's readings replace the config's.  An entry built for a non-CUDA device (host tests) keeps the config's readings; it
    validates, and declines to compile.  Construction validates every form the entry can run, so a row count the entry admits never
    declines on the execute path: a part that cannot hold a form is a ``NotImplementedError`` (a part fact), a knob no SM100-class
    part can run a ``ValueError`` -- either raised here.

    What a row count launches is readable without a launch: :meth:`kernel_symbols_for`, :meth:`form_for` and :meth:`launch_shape_for`
    (the launch facts of :mod:`cudnn.frost.launch_facts`)."""

    def __init__(self, device, geometry: Optional[Geometry] = None, cfg: Optional[Cfg] = None, pdl: Optional[bool] = None):
        self.device = resolve_device(device)
        if cfg is not None and not isinstance(cfg, Cfg):
            raise TypeError(f"{NAME}: cfg must be a swiglu_fp4_decode_config.Cfg, got {type(cfg).__name__}")
        if geometry is None:
            geometry = cfg.GEOM if cfg is not None else DEFAULT_GEOMETRY
        self.geometry: Geometry = check_geometry(geometry)
        if cfg is None:
            cfg = Cfg(GEOM=self.geometry)
        elif cfg.GEOM != self.geometry:
            raise ValueError(f"{NAME}: cfg.GEOM is {cfg.GEOM!r} and geometry is {self.geometry!r}; give one source")
        if pdl is not None:
            cfg = replace(cfg, PDL=bool(pdl))
        self.device_cc: Optional[Tuple[int, int]] = None
        if self.device.type == "cuda":
            import torch

            self.device_cc = tuple(torch.cuda.get_device_capability(self.device))
            if not cc_supported(self.device_cc):
                raise NotImplementedError(f"{NAME} serves compute capability {SUPPORTED_CC_TEXT}; this device is {self.device_cc[0]}.{self.device_cc[1]}")
            cfg = replace(cfg, **part_readings(*device_readings(self.device)))
        self.pdl: bool = cfg.PDL
        self.cfg: Cfg = cfg
        validate_cfg(replace(cfg, **self._reference_readings()))  # a knob no SM100-class part can run: ValueError, by name
        self.floor: int = cfg.TOKEN_TILE
        self.forms: Dict[int, Cfg] = {tile: self._resolve(tile) for tile in TOKEN_TILES if tile >= self.floor}
        self.compiled: Dict[Cfg, Any] = {}
        self._kernel_module = None

    @staticmethod
    def _reference_readings() -> Dict[str, int]:
        return part_readings(SM100_SMEM_PER_SM_BYTES, SM100_SMEM_OPTIN_BYTES, SM100_SM_COUNT)

    def _resolve(self, tile: int) -> Cfg:
        """The validated form of one token tile on this entry's readings."""
        form = form_of(self.cfg, tile)
        try:
            validate_cfg(form)
        except ValueError as e:
            try:
                validate_cfg(replace(form, **self._reference_readings()))
            except ValueError:
                raise ValueError(f"{NAME}: the config cannot be served at token tile {tile} on any SM100-class part: {e}") from e
            raise NotImplementedError(f"{NAME} does not serve this part at token tile {tile}: {e}") from e
        return form

    def cfg_for(self, rows: int) -> Cfg:
        """The form a row count runs: the entry's config at the narrowest token tile that holds the rows, not below the config's own
        ``TOKEN_TILE``.  ``ValueError`` for a malformed row count, ``NotImplementedError`` past 64 rows (raised on every call)."""
        form = self.forms[token_tile_for(rows, self.floor)]
        check_rows_fit(form, rows)
        return form

    def key(self, rows: int) -> tuple:
        """The compile key: plan-time facts only -- the form of the row count.  The row count itself is not part of it."""
        return (self.cfg_for(rows),)

    # ---- launch facts (cudnn.frost.launch_facts): three read-only answers per served row count ----
    def kernel_symbols_for(self, rows: int) -> Tuple[str, ...]:
        """The symbol of the kernel this entry has compiled for ``rows``: a one-element tuple, empty until the form of that row count
        is compiled.  The symbol carries the form's compact text (geometry, token tile, pre-wait stages, L2 policy, grid cap, launch
        attribute) right behind the kernel's name, so two forms of this entry never share a symbol."""
        fn = self.compiled.get(self.cfg_for(rows))
        return kernel_symbols(() if fn is None else (fn,), NAME)

    def form_for(self, rows: int) -> Dict[str, Any]:
        """The form of ``rows`` as plain data: every field of the config :meth:`cfg_for` returns -- the token tile, the pre-wait
        stages, the L2 policy, the grid cap, the launch attribute, this part's readings and the geometry as a nested ``dict``.  The
        config is the whole compile key, so nothing else is part of it."""
        return plain_form(self.cfg_for(rows))

    def launch_shape_for(self, rows: int) -> LaunchShape:
        """``(grid, block, shared memory)`` of the launch of ``rows``: persistent thread blocks along z, one per 128 weight rows up
        to the grid cap; 192 threads; the shared memory of the form's token tile (the operand rings, the staging ring and the
        barriers: what the launch reserves)."""
        cfg = self.cfg_for(rows)
        return LaunchShape(grid(cfg), (THREADS_PER_CTA, 1, 1), smem_bytes(cfg))

    def compile(self, rows: Optional[int] = None) -> None:
        """Plan-time compile: the form of one row count, or EVERY form of the entry when ``rows`` is None (so that no launch of a
        served row count compiles on the execute path).  Typed declines first (the row count, the device, the compute capability),
        then the DSL gate, then one trace-compile per form that is not compiled yet, under the entry's device.  Needs no operand."""
        forms = list(self.forms.values()) if rows is None else [self.cfg_for(rows)]
        if self.device.type != "cuda":
            raise NotImplementedError(f"{NAME} serves CUDA devices only; this entry was built for {self.device}")
        if all(form in self.compiled for form in forms):
            return
        import torch

        msg = dsl_error(self.device_cc)
        if msg is not None:
            raise NotImplementedError(msg)
        from .kernel import swiglu_fp4_decode_nvfp4 as mod

        with torch.cuda.device(self.device):
            for form in forms:
                if form not in self.compiled:
                    self.compiled[form] = mod.compile_kernel(form)
        self._kernel_module = mod

    def __call__(self, weight, weight_scale, activation, activation_scale, alpha, out_global_scale, out, out_scale):
        """Launch on torch's current stream of the entry's device; returns ``(out, out_scale)`` (the caller's tensors: every byte of
        the ``rows`` rows of ``out`` and the live rows' bytes of ``out_scale`` at their atom positions are written, nothing else).
        Every launch re-runs the binding checks before the compiled kernel is looked up.  Compiles on first use of a form that
        :meth:`compile` has not seen."""
        import cuda.bindings.driver as cuda
        import torch

        rows = check_binding(
            weight, weight_scale, activation, activation_scale, alpha, out_global_scale, out, out_scale, device=self.device, geometry=self.geometry
        )
        form = self.cfg_for(rows)
        if form not in self.compiled:
            self.compile(rows)
        # A compiled launchable belongs to its device's context: launch under the one it was compiled in.  The DLPack views of the
        # operands are built under the same guard -- their export negotiates with the current device, which need not be the entry's.
        with torch.cuda.device(self.device):
            args = self._kernel_module.launch_args(
                weight,
                activation,
                weight_scale.view(torch.uint8),
                activation_scale.view(torch.uint8),
                alpha,
                out,
                out_scale.view(torch.uint8),
                out_global_scale,
            )
            self.compiled[form](*args, cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream))
        return out, out_scale
