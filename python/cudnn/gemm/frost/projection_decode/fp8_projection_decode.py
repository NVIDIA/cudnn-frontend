# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host entry of the FP8 split-K projection for decode rows (``kernel/fp8_projection_decode_e4m3.py``): validation with typed declines,
the plan-time compile and the launch.  No DSL import at module level -- the CuTe DSL is imported inside
:meth:`Fp8ProjectionDecode.compile` after the version / architecture / layer gate (python/cudnn/AGENTS.md Rule 7), so a host without it
can still construct the entry and run its declines.  The binding checks (:func:`check_binding`: type, dtype, device, shape, contiguity
and alignment of every operand) run before EVERY launch, compile-cache hits included: a compiled DSL entry accepts a tensor of another
dtype, footprint or device silently, so the host is a launch's only guard.

The op: ``out[t, n] = bf16(scale * sum_k activation[t, k] * weight[n, k])`` -- ``weight`` [OUT_FEATURES, IN_FEATURES] and ``activation``
[rows, IN_FEATURES] hold E4M3 values (``torch.float8_e4m3fn``), ``scale`` is ONE fp32 element (the caller's product of the activation
scale and the weight scale; read on the device, never on the host), ``out`` [rows, OUT_FEATURES] is bf16, caller-owned and required.
fp32 accumulation in a fixed order per output element (the kernel module's docstring): the K axis is cut into ``split_k`` slices, each
slice is one tensor-core chain in ascending K, the slice sums are added in slice order, the sum is scaled and cast once.

NUMERICS AND PICKS.  The output bytes are a function of ``(geometry, split_k)`` on one runtime.  ``split_k`` is therefore fixed at
construction -- given, or :func:`split_k_for` of the geometry alone -- and never depends on the row count: a caller that wants another
split builds another entry and owns the statement that the two are different byte classes.  Everything else is a performance ``Form``
(weight tile, ring stages, tensor-memory columns, token tile, the weight stream ahead of the dependent-launch wait): the measured
:func:`pick` per row count on THIS part's readings, or the caller's ``form`` as a whole; no form changes a byte, and a row's bytes are
the same alone and among up to 32 rows.

Served (declined otherwise, by name): the validated geometries and splits (``SERVED_CLASSES``), rows 1 .. 32, contiguous operands with
32-byte aligned storage, every operand on the entry's device, a compute capability 10.x part, a CuTe DSL >= 4.8.0 that provides the
``cutlass.cute.experimental`` layer the kernel is written on.  ``out`` must not overlap ``weight``, ``activation`` or ``scale`` (not
checked).

THE ROW COUNT IS A COMPILE KEY (a stated deviation from python/cudnn/AGENTS.md Rule 4, bounded and precedented).  The kernel is
specialised on ``rows`` (its output bounds and token tile are trace-time constants), so there is one rendering per row count: at most
32 per entry, each a fraction of a second to compile; ``GdnDecodeFused`` keys on its row count the same way.  The plan-time site is
:meth:`Fp8ProjectionDecode.compile` -- ``compile(rows)`` for one count, ``compile()`` for all 32 -- after which every launch is a
cache hit; a launch at a row count that was not compiled compiles it on first use, which a serving path should never rely on.

Launch knobs: ``pdl`` (default off) puts the programmatic-stream-serialization attribute on every launch; the kernel then waits for its
predecessor right before it reads ``activation``, while the barrier inits, the tensor-memory allocation and the first weight K tiles
run ahead of that wait.  CALLER OBLIGATION under ``pdl``: no kernel that can still be in flight when the launch starts writes
``weight`` or ``scale``.  ``pdl_release`` (default: follows ``pdl``) releases the successor after the kernel's last weight issue: a
consumer launched behind it WITH the attribute may start while this kernel runs and must issue its own wait before it reads ``out``,
and if it allocates tensor memory above that wait it must fit beside this kernel's columns (at most 96 per SM at the default forms);
a consumer launched without the attribute is unaffected.

Rules 1-5 on the execute path: views only (``view(torch.uint8)``, ``reshape(1)``), no conversion, no copy, no device allocation, no
device read; the launch goes onto torch's current stream of the entry's device.
"""

from typing import Any, Dict, Optional, Tuple

from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error, cutedsl_state

from .kernel.fp8_projection_decode_config import (
    MAX_ROWS,
    NAME,
    OPERAND_ALIGN_BYTES,
    SM100_SM_COUNT,
    SM100_SMEM_OPTIN_BYTES,
    SM100_SMEM_PER_SM_BYTES,
    Cfg,
    Form,
    Geometry,
    Numerics,
    check_geometry,
    check_numerics,
    part_readings,
    pick,
    resolve_form,
    token_tile_for,
    validate_cfg,
)

#: the served compute capabilities as closed ranges of ``10 * major + minor``: SM100-class parts
SUPPORTED_CC_RANGES = ((100, 109),)
SUPPORTED_CC_TEXT = "10.x (SM100-class)"
DSL_FLOOR = (4, 8, 0)
DSL_FLOOR_TEXT = ".".join(str(x) for x in DSL_FLOOR)
#: the DSL names the kernel module binds at import or trace time, per module: probed before the kernel module is imported, so a DSL
#: that clears the version floor but lacks the layer (a later release that folded it away, a differently numbered build) declines by
#: name instead of raising from inside the DSL
DSL_LAYER = (
    (
        "cutlass.cute.experimental",
        ("jit", "kernel", "compile", "allocate", "tma_load", "dot", "partition_and_copy", "make_tmem_layout_acc", "make_t2r_rmem_layout", "get_cta_v_map_ab")
        + ("OperationTypeEnum",),
    ),
    ("cutlass.utils.blackwell_helpers", ("make_trivial_tiled_mma", "make_smem_layout_a", "make_smem_layout_b", "get_tmem_load_op")),
    (
        "cutlass.cute.arch",
        ("alloc_tmem", "dealloc_tmem", "relinquish_tmem_alloc_permit", "retrieve_tmem_ptr", "fence_view_async_tmem_load", "griddepcontrol_wait")
        + ("griddepcontrol_launch_dependents", "cluster_arrive_relaxed", "cluster_wait", "block_idx_in_cluster"),
    ),
    ("cutlass.cute.nvgpu.tcgen05", ("commit", "make_tmem_copy", "CtaGroup", "Field")),
    ("cutlass.cutlass_dsl", ("dsl_user_op", "T")),
)


def cc_supported(cc) -> bool:
    """Whether a ``(major, minor)`` compute capability is one this kernel serves; ``None`` (no CUDA device) is not."""
    if cc is None:
        return False
    code = 10 * int(cc[0]) + int(cc[1])
    return any(lo <= code <= hi for lo, hi in SUPPORTED_CC_RANGES)


def dsl_layer_error() -> Optional[str]:
    """The layer probe: every DSL name of :data:`DSL_LAYER` must import.  Imports the DSL (callers run it after the version gate, right
    before the kernel module's own import).  None when the layer is complete, else a message naming the first missing piece."""
    import importlib

    _, version = cutedsl_state()
    found = " ".join(version) if version else "unknown version"
    for module, names in DSL_LAYER:
        try:
            mod = importlib.import_module(module)
        except Exception as e:  # noqa: BLE001 - any failure to reach the layer is the same answer
            return f"{NAME} is written on {module}, which the installed CuTe DSL ({found}) does not provide ({type(e).__name__}: {e})"
        missing = [n for n in names if not hasattr(mod, n)]
        if missing:
            return f"{NAME} is written on {module}; the installed CuTe DSL ({found}) does not export {', '.join(missing)}"
    return None


def dsl_error(device_cc=None) -> Optional[str]:
    """The Rule-7 gate, in the order a caller can act on: the DSL is installed; its version clears the kernel's floor (4.8.0, through the
    one version parser of :mod:`cudnn.frost.buffers`); it has the target of the part; it provides the layer the kernel is written on.
    None when the kernel can be imported and compiled."""
    installed, _ = cutedsl_state()
    if not installed:
        return f"{NAME} requires nvidia-cutlass-dsl >= {DSL_FLOOR_TEXT}; the CuTe DSL is not installed"
    msg = cutedsl_requirement_error(NAME, floor=DSL_FLOOR)
    if msg is not None:
        return msg
    msg = cutedsl_arch_requirement_error(None if device_cc is None else tuple(device_cc))
    if msg is not None:
        return f"{NAME}: {msg}"
    return dsl_layer_error()


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


def check_binding(weight, activation, scale, out, device=None, geometry: Optional[Geometry] = None) -> int:
    """The binding checks, run before every launch (compile-cache hits included): ``TypeError`` for a non-tensor or a dtype,
    ``ValueError`` for an inconsistent shape, alignment or device, ``NotImplementedError`` for a row count, a geometry or a memory
    layout this kernel does not serve (no copy is ever made to serve one).  Returns ``rows = activation.shape[0]``.  ``device`` (an
    entry's: a concrete CUDA ordinal) requires every operand on exactly that device; without it the operands must share ``weight``'s.
    ``geometry`` defaults to the weight's own shape.  Host-only metadata arithmetic, no device work."""
    import torch

    tensors = dict(weight=weight, activation=activation, scale=scale, out=out)
    for name, t in tensors.items():
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{NAME}: {name} must be a torch.Tensor, got {type(t).__name__}")
    _check_device(tensors, device, "weight")
    for name, want in (("weight", torch.float8_e4m3fn), ("activation", torch.float8_e4m3fn), ("scale", torch.float32), ("out", torch.bfloat16)):
        if tensors[name].dtype != want:
            raise TypeError(f"{NAME}: {name} must be {want} (kernel-native; callers convert), got {tensors[name].dtype}")
    if geometry is None:
        if weight.dim() != 2:
            raise ValueError(f"{NAME}: weight must be [out features, in features]; got {tuple(weight.shape)}")
        geometry = Geometry(int(weight.shape[0]), int(weight.shape[1]))
    geometry = check_geometry(geometry)  # the served-geometry decline, by name
    n, k = geometry.OUT_FEATURES, geometry.IN_FEATURES
    if tuple(weight.shape) != (n, k):
        raise ValueError(f"{NAME}: weight must be [{n}, {k}] at {geometry.describe()}; got {tuple(weight.shape)}")
    if activation.dim() != 2 or activation.shape[1] != k:
        raise ValueError(f"{NAME}: activation must be [rows, {k}] at {geometry.describe()}; got {tuple(activation.shape)}")
    rows = int(activation.shape[0])
    token_tile_for(rows)  # ValueError below one row, NotImplementedError past the ceiling
    if tuple(out.shape) != (rows, n):
        raise ValueError(f"{NAME}: out must be exactly [{rows}, {n}] (the rows of the activation); got {tuple(out.shape)}")
    if scale.numel() != 1:
        raise ValueError(f"{NAME}: scale must hold one fp32 element (the per-tensor scale product); got {tuple(scale.shape)}")
    for name in ("weight", "activation", "out"):
        if not tensors[name].is_contiguous():
            raise NotImplementedError(
                f"{NAME} serves contiguous operands; {name} has shape {tuple(tensors[name].shape)} and strides {tuple(tensors[name].stride())}"
            )
    for name, align in (("weight", OPERAND_ALIGN_BYTES), ("activation", OPERAND_ALIGN_BYTES), ("out", OPERAND_ALIGN_BYTES), ("scale", 4)):
        if tensors[name].data_ptr() % align:
            raise ValueError(f"{NAME}: {name} must start at a {align}-byte aligned address, got data_ptr % {align} = {tensors[name].data_ptr() % align}")
    return rows


def check_support(weight, activation, scale, out, device_cc=None, device=None, geometry: Optional[Geometry] = None) -> int:
    """Typed declines, before any DSL import: the binding checks of :func:`check_binding` plus the architecture gate
    (``NotImplementedError`` for a part this kernel does not serve).  ``device_cc`` overrides the compute-capability probe (tests)."""
    import torch

    rows = check_binding(weight, activation, scale, out, device=device, geometry=geometry)
    dev = weight.device
    if device_cc is None and dev.type != "cuda":
        raise NotImplementedError(f"{NAME} serves CUDA tensors only; the operands are on {dev}")
    cc = tuple(torch.cuda.get_device_capability(dev)) if device_cc is None else tuple(device_cc)
    if not cc_supported(cc):
        raise NotImplementedError(f"{NAME} serves compute capability {SUPPORTED_CC_TEXT}; this device is {cc[0]}.{cc[1]}")
    return rows


class Fp8ProjectionDecode:
    """One projection geometry at one split, compiled per row count and shared by every layer of that geometry: the tensors are launch
    arguments, nothing is captured at construction.

    ``geometry``: the weight's ``Geometry(OUT_FEATURES, IN_FEATURES)``.  ``split_k``: the number of K slices, a NUMERICS attribute (None =
    :func:`split_k_for` of the geometry); it is fixed for the entry's life.  ``form``: a ``Form`` that overrides the measured pick as a
    whole for every row count (it has no numerics field); without it the entry takes :func:`pick` per row count on this part's
    readings.  ``pdl`` / ``pdl_release``: the launch knobs of the module docstring (None = off / follows ``pdl``).

    The device is resolved to a concrete CUDA ordinal at construction and every operand of every launch must live on exactly it.  An
    entry built for a non-CUDA device (host tests) keeps the SM100-class reference readings; it validates and picks, and declines to
    compile.  Construction resolves and validates the form of EVERY served row count, so a row count the entry admits never declines on
    the execute path: a part that cannot hold a form is a ``NotImplementedError`` (a part fact), a given ``form`` no SM100-class part
    can run a ``ValueError`` (a knob combination) -- either raised here."""

    def __init__(
        self,
        device,
        geometry: Geometry,
        split_k: Optional[int] = None,
        form: Optional[Form] = None,
        pdl: Optional[bool] = None,
        pdl_release: Optional[bool] = None,
    ):
        self.device = resolve_device(device)
        self.numerics: Numerics = check_numerics(geometry, split_k)
        self.geometry: Geometry = self.numerics.GEOM
        self.split_k: int = self.numerics.SPLIT_K
        if form is not None and not isinstance(form, Form):
            raise TypeError(f"{NAME}: form must be a fp8_projection_decode_config.Form, got {type(form).__name__}")
        self.form: Optional[Form] = form
        self.pdl = bool(pdl) if pdl is not None else False
        self.pdl_release = bool(pdl_release) if pdl_release is not None else self.pdl
        self.device_cc: Optional[Tuple[int, int]] = None
        readings = (SM100_SMEM_PER_SM_BYTES, SM100_SMEM_OPTIN_BYTES, SM100_SM_COUNT)
        if self.device.type == "cuda":
            import torch

            self.device_cc = tuple(torch.cuda.get_device_capability(self.device))
            if not cc_supported(self.device_cc):
                raise NotImplementedError(f"{NAME} serves compute capability {SUPPORTED_CC_TEXT}; this device is {self.device_cc[0]}.{self.device_cc[1]}")
            readings = device_readings(self.device)
        self.readings: Dict[str, int] = part_readings(*readings)
        self.cfgs: Dict[int, Cfg] = {rows: self._resolve(rows) for rows in range(1, MAX_ROWS + 1)}
        self.compiled: Dict[Cfg, Any] = {}
        self._kernel_module = None

    def _resolve(self, rows: int) -> Cfg:
        """The validated rendering of one row count: the given form, or the pick on this part's readings."""
        r = self.readings
        if self.form is None:
            form = pick(self.numerics, rows, r["SM_COUNT"], r["SMEM_PER_SM_BYTES"], r["SMEM_OPTIN_BYTES"])
        else:
            form = self.form
        cfg = resolve_form(self.numerics, rows, form, r, pdl=self.pdl, pdl_release=self.pdl_release)
        try:
            validate_cfg(cfg)
        except ValueError as e:
            reference = part_readings(SM100_SMEM_PER_SM_BYTES, SM100_SMEM_OPTIN_BYTES, SM100_SM_COUNT)
            try:
                validate_cfg(resolve_form(self.numerics, rows, form, reference, pdl=self.pdl, pdl_release=self.pdl_release))
            except ValueError:
                raise ValueError(f"{NAME}: the form cannot be served at {rows} rows on any SM100-class part: {e}") from e
            raise NotImplementedError(f"{NAME} does not serve this part at {rows} rows: {e}") from e
        return cfg

    def cfg_for(self, rows: int) -> Cfg:
        """The rendering a row count runs: its numerics, its performance form (the pick or the given form) and the launch knobs; resolved
        and validated at construction (the typed declines of a row count outside 1 .. 32 are raised on every call)."""
        token_tile_for(rows)
        return self.cfgs[rows]

    def key(self, rows: int) -> tuple:
        """The compile key: plan-time facts only -- the rendering of the row count (geometry, split, rows, form, launch knobs)."""
        return (self.cfg_for(rows),)

    def compile(self, rows: Optional[int] = None) -> None:
        """Plan-time compile: one row count, or every served one when ``rows`` is None.  Typed declines first (the row count, the device,
        the compute capability), then the DSL gate, then one trace-compile per rendering that is not compiled yet, under the entry's
        device.  Needs no operand."""
        if rows is None:
            for r in range(1, MAX_ROWS + 1):
                self.compile(r)
            return
        cfg = self.cfg_for(rows)
        if self.device.type != "cuda":
            raise NotImplementedError(f"{NAME} serves CUDA devices only; this entry was built for {self.device}")
        if cfg in self.compiled:
            return
        import torch

        msg = dsl_error(self.device_cc)
        if msg is not None:
            raise NotImplementedError(msg)
        from .kernel import fp8_projection_decode_e4m3 as mod

        with torch.cuda.device(self.device):
            self.compiled[cfg] = mod.compile_kernel(cfg)
        self._kernel_module = mod

    def __call__(self, weight, activation, scale, out):
        """Launch on torch's current stream of the entry's device; returns ``out`` (the caller's tensor, every row written).  Every launch
        re-runs the binding checks before the compiled kernel is looked up.  Compiles on first use of a row count that
        :meth:`compile` has not seen."""
        import cuda.bindings.driver as cuda
        import torch

        rows = check_binding(weight, activation, scale, out, device=self.device, geometry=self.geometry)
        cfg = self.cfgs[rows]
        if cfg not in self.compiled:
            self.compile(rows)
        args = self._kernel_module.launch_args(weight.view(torch.uint8), activation.view(torch.uint8), out, scale.reshape(1))
        with torch.cuda.device(self.device):  # a compiled launchable belongs to its device's context: launch under the one it was compiled in
            self.compiled[cfg](*args, cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream))
        return out
