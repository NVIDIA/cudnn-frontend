# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host entry of the fused Gated DeltaNet VERIFY step (``kernel/gdn_verify_fused_f16.py``): validation with typed declines, the
plan-time compile and the launch.  No DSL import at module level -- the CuTe DSL is imported inside :meth:`GdnVerifyFused.compile`
after the version / architecture gate (python/cudnn/AGENTS.md Rule 7), so a host without it can still construct the entry and run
its declines.  The binding checks (:func:`check_binding`: dtype, device, shape, stride and alignment of every operand against the step
shape) run before EVERY compile and EVERY launch, compile-cache hits included -- the compile key pins the plan-time facts only, and a
compiled DSL entry accepts a tensor of another dtype, footprint or device silently, so the host is a launch's only guard.  The entry's
device is resolved to a concrete CUDA ordinal at construction (:func:`resolve_device`) and every operand must live on exactly it (type
AND index: a CPU tensor or one on another CUDA device is a ``ValueError`` naming the operand and both devices).

The call contract is the FlashInfer Qwen 3.5 GDN block's verify form (its argument names and layouts), so an A/B against the
reference kernel (FlashInfer's built-in ``frost`` verify backend) is a one-line swap: ``entry(qkv, z, hidden, ba_weight, A_log, dt_bias, ssm_state, state_indices,
cu_seqlens, num_accepted_tokens, norm_weight, out_scale, conv_state, conv_weight, out=None, out_bf16=None) -> out`` (the fp8
output; ``out_bf16`` receives the bf16 ``y`` before quantization when given, else the entry's scratch).  Served geometry (declined
otherwise, by name): the entry's ``geometry`` -- a ``Geometry(K_HEADS, V_HEADS, HIDDEN)`` of ``kernel/gdn_geometry.py`` (the rank's head
counts and hidden width; the default geometry, 16 K heads / 48 V heads / hidden 5120, when none is given, so existing callers are
unchanged; a caller's ``cfg`` carries its own in ``cfg.GEOM``) -- with head dim 128 and conv width 4 pinned: ``qkv`` [rows, QKV_CHANNELS =
2 K 128 + V 128] bf16 (a row-strided view is fine; 10240 at the default), ``z`` [rows, V, 128] bf16 (row-strided), ``hidden`` [rows,
HIDDEN] bf16 contiguous, ``ba_weight`` [2 V, HIDDEN] bf16 contiguous, ``A_log`` / ``dt_bias`` [V] fp32 or bf16, the fp32 state ``[slots, V,
128, 128]`` (slot-strided), ``state_indices`` [requests, T] int32 (any row stride; ``T`` = tokens per request, a compile-time constant),
``cu_seqlens`` [requests + 1] and ``num_accepted_tokens`` [requests] int32 contiguous, ``norm_weight`` [128] fp32 or bf16, ``out_scale``
one fp32, ``conv_state`` [slots, QKV_CHANNELS, T + 2] bf16 CHANNEL-CONTIGUOUS (stored ``[slots, T + 2, QKV_CHANNELS]``: ``stride(1) ==
1``, the slot and column strides multiples of 8 elements), ``conv_weight`` [QKV_CHANNELS, 4] bf16 contiguous; every operand 16-byte
aligned; an SM100-class (compute capability 10.x) or SM120-class (12.x) part and a CuTe DSL >= 4.8.0.  Each geometry is its own compiled
form (``Cfg.GEOM`` is in the compile key).  Semantics (the per-token state layout): request
``n`` with ``a = num_accepted_tokens[n]`` in ``[1, T]`` starts from slot ``state_indices[n, a - 1]`` and writes the state after token
``t`` to slot ``state_indices[n, t]`` (a slot ``<= 0`` skips that write); its conv slot ``state_indices[n, 0]`` holds columns ``a - 1 ..
a + 1`` of the history and becomes ``[old[a], old[a + 1], x_0 .. x_{T-1}]``; a request whose source slot is ``<= 0`` (or whose conv slot
is ``<= 0``, or with no tokens) is padding: zero outputs, nothing else touched.  Requests may carry FEWER than ``T`` tokens (a ragged
step: ``ntok = cu_seqlens[n + 1] - cu_seqlens[n] < T``, as a serving engine's warm-up pass and a request without drafts produce): the kernel reads a
request's rows ``cu_seqlens[n] .. cu_seqlens[n + 1] - 1`` only (clamped), writes outputs and per-token slots for ``t < ntok`` only and leaves
the conv-history columns past ``ntok`` as they were; such a step's operands hold fewer than ``requests x T`` rows, so the caller states the
step's row count with ``rows`` (``>= cu_seqlens[requests]``, its own guarantee -- the host cannot read ``cu_seqlens`` without a sync) and
``check_support`` bounds the operands and outputs by it instead of ``requests x T`` (the FlashInfer contract, every request exactly ``T``
tokens, needs no ``rows``).  The config is the caller's ``cfg`` as a whole or,
without one, the budget-derived pick per step shape (``default_cfg`` on THIS device's SM count and shared-memory readings: the cluster
pair, two CTAs per V head, while the ``requests x 96`` CTAs have an SM each, one CTA per SM while the ``requests x 48`` CTAs have one, the
two-CTAs-per-SM form past that -- each where the part's shared memory holds that form; where it does not, the regime's ladder of
shallower-ring forms, token-aware, so a 99 KB opt-in serves T <= 8 with the 2-deep-ring pair or the one-CTA landing-slab form and T 9 ..
17 with the landing-slab pair, ``max_tokens_for`` naming the bound); the device's SM count, its shared memory per SM and its per-CTA opt-in
maximum are read at construction, and the entry starts from the first one-CTA form the part holds (a part that holds none is a typed
``NotImplementedError`` there).  The launch knobs ``pdl`` / ``pdl_release`` set the config's ``PDL`` / ``PDL_RELEASE`` on top of whatever form serves (the
pick or a given ``cfg``): under ``pdl=True`` every launch carries the programmatic-stream-serialization attribute, the kernel waits for
its predecessor (``griddepcontrol.wait``) only right before it reads the QKVZ projection's rows and the normed hidden (the slot chain,
the parameters, the BA weight rows, the state slab, the conv weights and history load ahead of the wait) and, with ``pdl_release``
(following ``pdl`` unless given), releases its successor at entry (``griddepcontrol.launch_dependents``) so the out projection's weight
stream overlaps this kernel -- the reference verify kernel's placement.  Under ``cfg.PDL`` the caller owes the obligation in the
config's docstring (the launch's immediate predecessor must not write this layer's state, conv state or slot indices; the serving chain's
predecessor, the QKVZ projection, writes ``qkv`` / ``z`` only).  The picks keep PDL off.
"""

from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error, cutedsl_state

from .kernel.gdn_geometry import DEFAULT_GEOMETRY, Geometry
from .kernel.gdn_verify_fused_config import (
    CFG,
    CONV_WIDTH,
    HEAD_DIM,
    ONE_CTA_FORMS,
    SM100_SMEM_OPTIN_BYTES,
    SM100_SMEM_PER_SM_BYTES,
    TICKET_STRIDE,
    TOKENS_MIN,
    Cfg,
    default_cfg,
    first_fitting,
    max_tokens_served,
    resolve_cfg,
    validate_cfg,
    validate_tokens,
)

#: the served compute capabilities as closed ranges of ``10 * major + minor``: SM100-class parts (cc 10.x; validated on cc 10.0) and
#: SM120-class parts (cc 12.x; validated on cc 12.0 -- 100 KB of shared memory per SM, 99 KB per-CTA opt-in, 170 SMs -- the same kernel
#: body on the budget-derived picks; cc 12.1 shares the family admission on those picks and has not run).  cc 11.x has never run these
#: kernels and stays declined by name.  cc 12.x needs no DSL target gate of its own: the 4.8.0 floor (``DSL_FLOOR``) carries the sm_120a
#: and sm_121a targets, and ``cutedsl_arch_requirement_error`` only gates the one SM100-class target that needs a newer DSL.
SUPPORTED_CC_RANGES = ((100, 109), (120, 129))
SUPPORTED_CC_TEXT = "10.x (SM100-class) or 12.x (SM120-class)"
DSL_FLOOR = (4, 8, 0)
DSL_FLOOR_TEXT = ".".join(str(x) for x in DSL_FLOOR)
NAME = "the fused GDN verify step"
#: the launch is grid ``(V_HEADS, requests, 1)`` (``kernel/gdn_verify_fused_f16.py``), one grid row per request: CUDA caps grid Y at 65535
MAX_REQUESTS = 65535


def cc_supported(cc) -> bool:
    """Whether a ``(major, minor)`` compute capability is one this kernel serves: inside one of ``SUPPORTED_CC_RANGES`` (SM100-class 10.x,
    SM120-class 12.x); ``None`` (no CUDA device) is not."""
    if cc is None:
        return False
    code = 10 * int(cc[0]) + int(cc[1])
    return any(lo <= code <= hi for lo, hi in SUPPORTED_CC_RANGES)


OPERANDS = (
    "qkv",
    "z",
    "hidden",
    "ba_weight",
    "A_log",
    "dt_bias",
    "ssm_state",
    "state_indices",
    "cu_seqlens",
    "num_accepted_tokens",
    "norm_weight",
    "out_scale",
    "conv_state",
    "conv_weight",
)


def _dsl_version_error() -> Optional[str]:
    """The Rule-7 gate of this kernel: its own floor (``DSL_FLOOR`` = 4.8.0, the public wheel every consumer pins, above the library's
    ``CUTEDSL_MIN_VERSION``) through the one version parser of :mod:`cudnn.frost.buffers`; None when the installed DSL serves."""
    installed, _ = cutedsl_state()
    if not installed:
        return f"{NAME} requires nvidia-cutlass-dsl >= {DSL_FLOOR_TEXT}; the CuTe DSL is not installed"
    return cutedsl_requirement_error(NAME, floor=DSL_FLOOR)


def device_shared_memory(device) -> Tuple[int, int]:
    """(shared memory per SM, per-CTA opt-in maximum) in bytes of a CUDA device, from the runtime's device properties -- the part's
    own reading, never a constant carried across an architecture line."""
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
    """The SM count of a CUDA device, the plan-time fact of the default pick (the cluster pair while the ``requests x 96`` CTAs have
    an SM each, one CTA per SM while the ``requests x 48`` CTAs have one, two CTAs per SM past that)."""
    import torch

    return int(torch.cuda.get_device_properties(device).multi_processor_count)


def _check_dtype(name, t, want) -> None:
    import torch

    wants = want if isinstance(want, tuple) else (want,)
    if t.dtype not in wants:
        raise TypeError(f"{NAME}: {name} must be {' or '.join(str(x) for x in wants)} (kernel-native; callers convert), got {t.dtype}")


def _check_aligned(name, t) -> None:
    if t.data_ptr() % 16:
        raise ValueError(f"{NAME}: {name} must start at a 16-byte aligned address for its 128-bit copies, got data_ptr % 16 = {t.data_ptr() % 16}")


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


def resolve_geometry(geometry, cfg: Optional[Cfg] = None) -> Geometry:
    """The geometry a call is checked against: ``geometry`` when given (a ``Geometry``, else ``TypeError``), else the ``cfg``'s own
    (``cfg.GEOM``), else the default geometry; a ``cfg`` compiled for another geometry than the one given is a ``ValueError`` (one source)."""
    if geometry is not None and not isinstance(geometry, Geometry):
        raise TypeError(f"{NAME}: geometry must be a gdn_geometry.Geometry, got {type(geometry).__name__}")
    if cfg is not None and geometry is not None and cfg.GEOM != geometry:
        raise ValueError(f"{NAME}: the cfg is compiled for {cfg.GEOM.describe()} but the geometry names {geometry.describe()}; give one source")
    if geometry is not None:
        return geometry
    return cfg.GEOM if cfg is not None else DEFAULT_GEOMETRY


def step_shape(state_indices) -> Tuple[int, int]:
    """``(requests, tokens per request)`` of a verify step: the two extents of the slot-index table."""
    if state_indices.dim() != 2:
        raise NotImplementedError(f"{NAME} serves state_indices as [requests, tokens per request]; got {tuple(state_indices.shape)}")
    return int(state_indices.shape[0]), int(state_indices.shape[1])


def check_binding(
    qkv,
    z,
    hidden,
    ba_weight,
    A_log,
    dt_bias,
    ssm_state,
    state_indices,
    cu_seqlens,
    num_accepted_tokens,
    norm_weight,
    out_scale,
    conv_state,
    conv_weight,
    out=None,
    out_bf16=None,
    cfg: Optional[Cfg] = None,
    rows: Optional[int] = None,
    device=None,
    geometry: Optional[Geometry] = None,
) -> None:
    """The binding checks, run before every compile AND every launch (compile-cache hits included): ``TypeError`` for a dtype (or a
    non-tensor), ``ValueError`` for an inconsistent shape, stride, alignment or device, ``NotImplementedError`` for a geometry / layout /
    step shape this kernel does not serve.  ``device`` (an entry's: a concrete CUDA ordinal) requires every operand on exactly that device,
    type AND index; without it the operands must share ``qkv``'s.  ``geometry`` (an entry's) is the rank's head counts and hidden width the
    operand shapes are checked against: the given ``cfg``'s own (``cfg.GEOM``) without one, the default geometry (16 K heads, 48 V heads,
    hidden 5120) without either.  The compile key pins the plan-time facts only (the step shape, the strides the kernel is specialized on,
    the parameter dtypes, the config -- its geometry included), and a compiled DSL entry accepts a tensor of another dtype, footprint or
    device silently, so these checks are a launch's only guard; host-only metadata arithmetic, no device work.  ``cfg`` (resolved) bounds
    the tokens per request by its shared-memory budget -- an entry passes the form it picked for the step (:meth:`GdnVerifyFused.cfg_for`,
    token-aware on this part); without one the module's one-CTA form ``CFG`` bounds it (8 tokens on the cc 10.0 readings), the bound of a
    host caller that names no device and no form; ``rows`` (the step's token rows, ``>= cu_seqlens[requests]`` by the caller's contract)
    bounds the row operands and the outputs for a ragged or padded step instead of ``requests x T``.  :func:`check_support` adds the
    architecture gate."""
    import torch

    g = resolve_geometry(geometry, cfg)
    QKV, NV, HID = g.QKV_CHANNELS, g.V_HEADS, g.HIDDEN
    tensors = dict(
        zip(
            OPERANDS,
            (
                qkv,
                z,
                hidden,
                ba_weight,
                A_log,
                dt_bias,
                ssm_state,
                state_indices,
                cu_seqlens,
                num_accepted_tokens,
                norm_weight,
                out_scale,
                conv_state,
                conv_weight,
            ),
        )
    )
    if out is not None:
        tensors["out"] = out
    if out_bf16 is not None:
        tensors["out_bf16"] = out_bf16
    for name, t in tensors.items():
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{NAME}: {name} must be a torch.Tensor, got {type(t).__name__}")
    _check_device(tensors, device, "qkv")
    _check_dtype("qkv", qkv, torch.bfloat16)
    _check_dtype("z", z, torch.bfloat16)
    _check_dtype("hidden", hidden, torch.bfloat16)
    _check_dtype("ba_weight", ba_weight, torch.bfloat16)
    _check_dtype("A_log", A_log, (torch.float32, torch.bfloat16))
    _check_dtype("dt_bias", dt_bias, (torch.float32, torch.bfloat16))
    _check_dtype("ssm_state", ssm_state, torch.float32)
    _check_dtype("state_indices", state_indices, torch.int32)
    _check_dtype("cu_seqlens", cu_seqlens, torch.int32)
    _check_dtype("num_accepted_tokens", num_accepted_tokens, torch.int32)
    _check_dtype("norm_weight", norm_weight, (torch.float32, torch.bfloat16))
    _check_dtype("out_scale", out_scale, torch.float32)
    _check_dtype("conv_state", conv_state, torch.bfloat16)
    _check_dtype("conv_weight", conv_weight, torch.bfloat16)
    if out is not None:
        _check_dtype("out", out, torch.float8_e4m3fn)
    if out_bf16 is not None:
        _check_dtype("out_bf16", out_bf16, torch.bfloat16)
    # the step shape
    requests, tokens = step_shape(state_indices)
    if requests < 1:
        raise ValueError(f"{NAME}: state_indices must hold at least one request; got {tuple(state_indices.shape)}")
    if requests > MAX_REQUESTS:
        raise NotImplementedError(f"{NAME} launches one grid row per request, at most {MAX_REQUESTS} (CUDA's grid Y limit); got requests={requests}")
    if state_indices.stride(1) != 1:
        raise NotImplementedError(f"{NAME} needs the tokens of a request contiguous in state_indices; got strides {tuple(state_indices.stride())}")
    if cfg is None and tokens > CFG.max_tokens():  # a check without a form: the module default form's bound, never a statement about a part
        raise NotImplementedError(
            f"{NAME}: without a form the step is bounded by the module default form (TPR {CFG.TPR}, NBUF {CFG.NBUF}) on the cc 10.0 readings, "
            f"which holds at most {CFG.max_tokens()} tokens per request; got {tokens} -- an entry bounds a step by its own pick for the part "
            f"(GdnVerifyFused.cfg_for / max_tokens_for: 35 / 8 / 45 by regime on a cc 10.0 part, 17 on a 99 KB per-CTA opt-in)"
        )
    validate_tokens(cfg if cfg is not None else CFG, tokens)
    if rows is None:
        rows = requests * tokens  # the FlashInfer contract: every request carries exactly T tokens
        bound = f"requests x T = {rows}"
    else:
        if not isinstance(rows, int) or rows < 1:
            raise ValueError(f"{NAME}: rows must be a positive int (the step's token rows, at least cu_seqlens[requests]); got {rows!r}")
        bound = f"the step's rows = {rows}"
    # geometry (the entry's ``g``: its head counts and hidden width; head dim 128 and conv width 4 pinned)
    if qkv.dim() != 2 or qkv.shape[0] < rows or qkv.shape[1] != QKV:
        raise NotImplementedError(
            f"{NAME} serves qkv as [rows >= {bound}, {QKV}] ({g.K_HEADS} K heads, {NV} V heads x {HEAD_DIM}: {g.describe()}); got {tuple(qkv.shape)}"
        )
    if qkv.stride(1) != 1 or qkv.stride(0) % 8:
        raise NotImplementedError(f"{NAME} needs qkv rows contiguous with a 16-byte-multiple row stride; got strides {tuple(qkv.stride())}")
    if z.dim() != 3 or z.shape[0] < rows or tuple(z.shape[1:]) != (NV, HEAD_DIM):
        raise NotImplementedError(f"{NAME} serves z as [rows >= {rows}, {NV}, {HEAD_DIM}]; got {tuple(z.shape)}")
    if tuple(z.stride()[1:]) != (HEAD_DIM, 1) or z.stride(0) % 8:
        raise NotImplementedError(f"{NAME} needs z heads contiguous with a 16-byte-multiple row stride; got strides {tuple(z.stride())}")
    if hidden.dim() != 2 or hidden.shape[0] < rows or hidden.shape[1] != HID or not hidden.is_contiguous():
        raise NotImplementedError(f"{NAME} serves a contiguous hidden of [rows >= {rows}, {HID}]; got {tuple(hidden.shape)} strides {tuple(hidden.stride())}")
    if tuple(ba_weight.shape) != (2 * NV, HID) or not ba_weight.is_contiguous():
        raise NotImplementedError(f"{NAME} serves a contiguous ba_weight of [{2 * NV}, {HID}]; got {tuple(ba_weight.shape)}")
    if A_log.numel() < NV or dt_bias.numel() < NV or not A_log.is_contiguous() or not dt_bias.is_contiguous():
        raise NotImplementedError(f"{NAME} serves contiguous A_log / dt_bias of {NV} V heads; got {tuple(A_log.shape)} / {tuple(dt_bias.shape)}")
    if ssm_state.dim() != 4 or tuple(ssm_state.shape[1:]) != (NV, HEAD_DIM, HEAD_DIM):
        raise NotImplementedError(f"{NAME} serves the fp32 ssm_state as [slots, {NV}, {HEAD_DIM}, {HEAD_DIM}]; got {tuple(ssm_state.shape)}")
    if tuple(ssm_state.stride()[1:]) != (HEAD_DIM * HEAD_DIM, HEAD_DIM, 1) or ssm_state.stride(0) % 4:
        raise NotImplementedError(
            f"{NAME} needs head slabs of {HEAD_DIM} x {HEAD_DIM} contiguous fp32 and a 16-byte-multiple slot stride; got strides {tuple(ssm_state.stride())}"
        )
    if not cu_seqlens.is_contiguous() or cu_seqlens.numel() < requests + 1:
        raise ValueError(f"{NAME}: cu_seqlens must be a contiguous int32 vector of at least requests + 1 = {requests + 1}; got {tuple(cu_seqlens.shape)}")
    if not num_accepted_tokens.is_contiguous() or num_accepted_tokens.numel() < requests:
        raise ValueError(
            f"{NAME}: num_accepted_tokens must be a contiguous int32 vector of at least requests = {requests}; got {tuple(num_accepted_tokens.shape)}"
        )
    if norm_weight.numel() != HEAD_DIM or not norm_weight.is_contiguous():
        raise NotImplementedError(f"{NAME} serves a contiguous norm_weight of {HEAD_DIM}; got {tuple(norm_weight.shape)}")
    if out_scale.numel() < 1:
        raise ValueError(f"{NAME}: out_scale must hold the static FP8 scale (one fp32 element); got {tuple(out_scale.shape)}")
    if conv_state.dim() != 3 or conv_state.shape[1] != QKV:
        raise NotImplementedError(f"{NAME} serves conv_state as [slots, {QKV}, T + {CONV_WIDTH - 2}]; got {tuple(conv_state.shape)}")
    if conv_state.shape[2] != tokens + CONV_WIDTH - 2:
        raise NotImplementedError(
            f"{NAME} serves a conv_state of T + {CONV_WIDTH - 2} = {tokens + CONV_WIDTH - 2} history columns for T = {tokens} tokens per request; got {conv_state.shape[2]}"
        )
    css, csd, cst = (int(s) for s in conv_state.stride())
    if csd != 1 or cst % 8 or cst < QKV or css % 8:
        raise NotImplementedError(
            f"{NAME} needs conv_state channel-contiguous (stored [slots, T + {CONV_WIDTH - 2}, {QKV}]: stride(1) == 1, the slot and column strides multiples of 8 elements); got strides {tuple(conv_state.stride())}"
        )
    if tuple(conv_weight.shape) != (QKV, CONV_WIDTH) or not conv_weight.is_contiguous():
        raise NotImplementedError(
            f"{NAME} serves a contiguous conv_weight [{QKV}, {CONV_WIDTH}]; got {tuple(conv_weight.shape)} strides {tuple(conv_weight.stride())}"
        )
    for name, t in (("out", out), ("out_bf16", out_bf16)):
        if t is not None and (t.dim() != 3 or t.shape[0] < rows or tuple(t.shape[1:]) != (NV, HEAD_DIM) or not t.is_contiguous()):
            raise ValueError(
                f"{NAME}: {name} must be a contiguous [rows >= {bound}, {NV}, {HEAD_DIM}] tensor; got {tuple(t.shape)} strides {tuple(t.stride())}"
            )
    for name, t in tensors.items():
        if t.numel() > 1:  # every multi-element operand is declared 16-byte aligned to the kernel (_flat); a one-element scale is not
            _check_aligned(name, t)


def check_support(
    qkv,
    z,
    hidden,
    ba_weight,
    A_log,
    dt_bias,
    ssm_state,
    state_indices,
    cu_seqlens,
    num_accepted_tokens,
    norm_weight,
    out_scale,
    conv_state,
    conv_weight,
    out=None,
    out_bf16=None,
    device_cc=None,
    cfg: Optional[Cfg] = None,
    rows: Optional[int] = None,
    device=None,
    geometry: Optional[Geometry] = None,
) -> None:
    """Typed declines, before any DSL import: the binding checks of :func:`check_binding` (``cfg`` / ``rows`` / ``device`` / ``geometry``
    as there) plus the architecture gate (``NotImplementedError`` for a part this kernel does not serve).  ``device_cc`` overrides the
    compute-capability probe (tests)."""
    import torch

    check_binding(
        qkv,
        z,
        hidden,
        ba_weight,
        A_log,
        dt_bias,
        ssm_state,
        state_indices,
        cu_seqlens,
        num_accepted_tokens,
        norm_weight,
        out_scale,
        conv_state,
        conv_weight,
        out,
        out_bf16,
        cfg=cfg,
        rows=rows,
        device=device,
        geometry=geometry,
    )
    dev = qkv.device
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


class GdnVerifyFused:
    """Compiled per (requests, tokens per request, strides, parameter dtypes, config -- its geometry included) and shared by every GDN
    layer: all per-layer tensors are launch arguments.  ``cfg`` overrides the config as a whole; without it the entry picks the form per
    step shape (``default_cfg`` on this device's SM count and shared-memory readings for the entry's geometry: the cluster pair, two CTAs
    per V head, while the ``requests x 2 V`` CTAs have an SM each (``requests x 96`` at the default geometry), one CTA per SM while the
    ``requests x V`` CTAs have one (``requests x 48``), two CTAs per SM past that -- each where the part holds it, else the regime's ladder
    of shallower-ring forms, token-aware; :meth:`max_tokens_for` is the served bound).  ``geometry`` is the rank's head counts and hidden
    width (a ``kernel.gdn_geometry.Geometry``): the default geometry without one, a given ``cfg``'s own (``cfg.GEOM``) otherwise -- a
    ``cfg`` and a ``geometry`` that disagree are a ``ValueError``; every form the entry picks is stamped with it, every binding check reads
    it.  The entry starts from the first one-CTA form this part holds (``CFG`` on the cc 10.0 readings, the landing-slab form on a 99 KB
    opt-in; a part holding none is a typed ``NotImplementedError`` at construction).  An entry built for a non-CUDA device (host tests)
    has no SM count to pick by and keeps the one-CTA form ``CFG`` on the cc 10.0 readings.  The device is resolved to a concrete CUDA
    ordinal at construction (:func:`resolve_device`), and every operand of every compile / launch must live on exactly it.  ``pdl`` /
    ``pdl_release`` (None = the form's own fields) are the launch knobs on top of the form that serves."""

    def __init__(
        self,
        device,
        eps: float,
        cfg: Optional[Cfg] = None,
        debug: bool = False,
        pdl: Optional[bool] = None,
        pdl_release: Optional[bool] = None,
        geometry: Optional[Geometry] = None,
    ):
        import torch

        self.device = resolve_device(device)  # a CUDA device without an index is the current ordinal: the one every operand must match
        self.eps = float(eps)
        self.cfg_given = cfg is not None
        self.geometry: Geometry = resolve_geometry(geometry, cfg)  # the rank's head counts and hidden width: stamped into every form
        self.debug_flag = bool(debug)
        # the launch knobs on top of whatever form serves: None keeps the form's own PDL / PDL_RELEASE fields (the picks: off)
        self.pdl: Optional[bool] = None if pdl is None else bool(pdl)
        self.pdl_release: Optional[bool] = None if pdl_release is None else bool(pdl_release)
        self.sm_count: Optional[int] = None
        # THIS part's readings, (shared memory per SM, per-CTA opt-in): the pick's budget and every form's pad; the cc 10.0 readings
        # until a CUDA device replaces them
        self.readings: Tuple[int, int] = (SM100_SMEM_PER_SM_BYTES, SM100_SMEM_OPTIN_BYTES)
        self.part: Dict[str, int] = {}  # the readings as Cfg fields, stamped into every form on a CUDA device (empty on a host entry)
        if self.device.type == "cuda":
            self.readings = device_shared_memory(self.device)
            self.part = dict(SMEM_PER_SM_BYTES=self.readings[0], SMEM_OPTIN_BYTES=self.readings[1])
            self.sm_count = device_multiprocessor_count(self.device)
        # the override, or the first one-CTA form this part holds (CFG on the cc 10.0 readings and on a host without a device; the
        # landing-slab form on a 99 KB opt-in) -- a part that holds none is the typed decline here, before any form is stamped
        start = cfg if cfg is not None else (first_fitting(ONE_CTA_FORMS, *self.readings) if self.part else CFG)
        self.cfg = self._on_this_part(start)
        self.forms: Dict[Tuple[Cfg, int], Cfg] = {}  # the resolved forms on this part, by (module form, requests)
        self.compiled: Dict[tuple, Any] = {}
        self.buffers: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def _on_this_part(self, cfg: Cfg) -> Cfg:
        """``cfg`` with this part's shared-memory readings, the entry's geometry, the debug flag and the launch knobs, validated."""
        cfg = replace(cfg, DEBUG=self.debug_flag, GEOM=self.geometry, **self.part, **self.launch_fields(cfg))
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

    def cfg_for(self, requests: int, tokens: Optional[int] = None) -> Cfg:
        """The config a plan for ``requests`` (and ``tokens`` per request; None = the minimum) compiles with, its ``NORM_FORM`` resolved:
        the caller's ``cfg`` when one was given, else the pick (``default_cfg`` on this device's SM count and shared-memory readings; the
        one-CTA form without a CUDA device).  ``tokens`` moves the pick only on a part whose shared memory does not hold the regime's
        primary form (a 99 KB opt-in: T 9 .. 17 fall through to the landing-slab pair); past the ladder's bound the deepest form is
        returned, so ``validate_tokens`` on it names the part's true bound; a count below the minimum (a malformed slot table) picks at
        the minimum, so :func:`check_binding` keeps its typed decline of it.  One resolved form per (module form, requests)."""
        if tokens is not None and not isinstance(tokens, int):
            raise ValueError(f"{NAME}: tokens must be an int or None, got {tokens!r}")
        want = None if tokens is None else max(tokens, TOKENS_MIN)
        form = self.cfg if (self.cfg_given or self.sm_count is None) else default_cfg(requests, self.sm_count, *self.readings, want, self.geometry)
        key = (form, requests)
        if key not in self.forms:
            self.forms[key] = resolve_cfg(self._on_this_part(form), requests)
        return self.forms[key]

    def max_tokens_for(self, requests: int) -> int:
        """The largest tokens-per-request count a plan for ``requests`` can hold on this part: the served bound of the pick over the
        regime's whole ladder (``max_tokens_served``: 35 / 8 / 45 by regime on the cc 10.0 readings, 17 at every requests count on a 99 KB
        opt-in), or a given ``cfg``'s own bound -- the one bound ``check_support`` and FlashInfer's probe answer alike."""
        if self.cfg_given or self.sm_count is None:
            return self.cfg.max_tokens()
        return max_tokens_served(requests, self.sm_count, *self.readings, self.geometry)

    @property
    def debug(self) -> bool:
        return self.cfg.DEBUG

    def outputs(self, requests: int, tokens: int):
        """The entry-owned buffers of a step shape at the entry's geometry (V V heads, QKV channels): ``qout`` fp8 [rows, V, 128] and
        ``out`` bf16 (the defaults when the caller passes no ``out`` / ``out_bf16``), ``counts`` int32 [requests * 512] (the conv-history
        reader tickets, zero between launches) and, under ``DEBUG`` only, the debug buffers ``raw`` [rows, V, 128] bf16, ``convout`` [rows,
        QKV] bf16, ``baout`` [rows, 2 V] bf16 -- without ``DEBUG`` the kernel's debug slots are compiled out (``None``), never bound to an
        allocation.  Entry-owned scratch, allocated once per step shape at the first compile / call for it (outside any capture; a
        FlashInfer runner has no workspace contract to carve it from)."""
        shape = (requests, tokens)
        if shape not in self.buffers:
            import torch

            dev = self.device
            g = self.geometry
            rows = requests * tokens
            buffers = dict(
                qout=torch.empty(rows, g.V_HEADS, HEAD_DIM, dtype=torch.float8_e4m3fn, device=dev),
                out=torch.empty(rows, g.V_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev),
                counts=torch.zeros(requests * TICKET_STRIDE, dtype=torch.int32, device=dev),
            )
            if self.debug:
                buffers.update(
                    raw=torch.zeros(rows, g.V_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev),
                    convout=torch.zeros(rows, g.QKV_CHANNELS, dtype=torch.bfloat16, device=dev),
                    baout=torch.zeros(rows, 2 * g.V_HEADS, dtype=torch.bfloat16, device=dev),
                )
            self.buffers[shape] = buffers
        return self.buffers[shape]

    def _debug_args(self, b) -> list:
        """The kernel's three DEBUG slots: the dumps' views under ``DEBUG``, else ``None`` x 3 -- the DSL specializes on ``None`` and
        compiles the slots out, so a dead slot is never bound to an allocation."""
        return [_flat(b["raw"]), _flat(b["convout"]), _flat(b["baout"])] if self.debug else [None, None, None]

    def key(self, qkv, z, A_log, dt_bias, ssm_state, state_indices, norm_weight, conv_state):
        """The compile key: plan-time facts only (the step shape, strides, parameter dtypes, the config picked for this step shape)."""
        requests, tokens = step_shape(state_indices)
        return (
            requests,
            tokens,
            self.cfg_for(requests, tokens),
            int(qkv.stride(0)),
            int(z.stride(0)),
            int(ssm_state.stride(0)),
            int(state_indices.stride(0)),
            int(conv_state.stride(0)),
            int(conv_state.stride(2)),
            A_log.dtype,
            dt_bias.dtype,
            norm_weight.dtype,
            self.eps,
        )

    def _args(
        self,
        qkv,
        z,
        hidden,
        ba_weight,
        A_log,
        dt_bias,
        ssm_state,
        state_indices,
        cu_seqlens,
        num_accepted_tokens,
        norm_weight,
        out_scale,
        conv_state,
        conv_weight,
        out,
        out_bf16,
    ):
        import torch

        requests, tokens = step_shape(state_indices)
        b = self.outputs(requests, tokens)
        qout = b["qout"] if out is None else out
        outb = b["out"] if out_bf16 is None else out_bf16
        return [
            _flat(qkv),
            _flat(z),
            _flat(hidden),
            _flat(ba_weight),
            _flat(A_log),
            _flat(dt_bias),
            _flat(ssm_state),
            _flat(state_indices),
            _flat(cu_seqlens),
            _flat(num_accepted_tokens),
            _flat(norm_weight),
            _flat(out_scale),
            _flat(b["counts"]),
            _flat(outb),
            _flat(qout.view(torch.uint8)),
            _flat(conv_state),
            _flat(conv_weight),
            *self._debug_args(b),
        ]

    def compile(
        self,
        qkv,
        z,
        hidden,
        ba_weight,
        A_log,
        dt_bias,
        ssm_state,
        state_indices,
        cu_seqlens,
        num_accepted_tokens,
        norm_weight,
        out_scale,
        conv_state,
        conv_weight,
        out=None,
        out_bf16=None,
        rows: Optional[int] = None,
    ) -> None:
        """Validate (typed declines, on every call: every operand on this entry's device, a compile-cache hit included), gate the DSL,
        trace-compile the host entry for this (step shape, strides, dtypes) once.  ``rows``: the step's token rows for a ragged / padded
        step (see :func:`check_support`); not a compile key (the kernel reads ``cu_seqlens``)."""
        requests, tokens = step_shape(state_indices)
        check_support(
            qkv,
            z,
            hidden,
            ba_weight,
            A_log,
            dt_bias,
            ssm_state,
            state_indices,
            cu_seqlens,
            num_accepted_tokens,
            norm_weight,
            out_scale,
            conv_state,
            conv_weight,
            out,
            out_bf16,
            cfg=self.cfg_for(requests, tokens),
            rows=rows,
            device=self.device,
            geometry=self.geometry,
        )
        key = self.key(qkv, z, A_log, dt_bias, ssm_state, state_indices, norm_weight, conv_state)
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

        from .kernel import gdn_verify_fused_f16 as mod

        args = self._args(
            qkv,
            z,
            hidden,
            ba_weight,
            A_log,
            dt_bias,
            ssm_state,
            state_indices,
            cu_seqlens,
            num_accepted_tokens,
            norm_weight,
            out_scale,
            conv_state,
            conv_weight,
            out,
            out_bf16,
        )
        with torch.cuda.device(self.device):
            self.compiled[key] = cute.compile(
                mod.launch,
                self.cfg_for(requests, tokens),
                *args,
                requests,
                int(qkv.stride(0)),
                int(z.stride(0)),
                int(ssm_state.stride(0)),
                int(state_indices.stride(0)),
                int(conv_state.stride(0)),
                int(conv_state.stride(2)),
                self.eps,
                tokens,
                make_fake_stream(),
            )

    def __call__(
        self,
        qkv,
        z,
        hidden,
        ba_weight,
        A_log,
        dt_bias,
        ssm_state,
        state_indices,
        cu_seqlens,
        num_accepted_tokens,
        norm_weight,
        out_scale,
        conv_state,
        conv_weight,
        out=None,
        out_bf16=None,
        rows: Optional[int] = None,
    ):
        """Launch on torch's current stream; returns the fp8 output (``out`` when given, else the entry's buffer for this step
        shape; ``outputs(requests, tokens)`` holds the entry's buffers).  Compiles on first use for a new key (``rows`` as in
        :meth:`compile`).  Every launch re-runs the binding checks (:func:`check_binding`, against the form picked for this step shape,
        every operand on this entry's device) before the compiled kernel is looked up: the key pins no dtype, shape, alignment or
        device, and the DSL checks none."""
        import cuda.bindings.driver as cuda
        import torch

        cfg = None
        if isinstance(state_indices, torch.Tensor) and state_indices.dim() == 2 and state_indices.shape[0] >= 1:
            # the form picked for this step shape bounds the tokens per request; a malformed table is check_binding's decline
            cfg = self.cfg_for(int(state_indices.shape[0]), int(state_indices.shape[1]))
        check_binding(
            qkv,
            z,
            hidden,
            ba_weight,
            A_log,
            dt_bias,
            ssm_state,
            state_indices,
            cu_seqlens,
            num_accepted_tokens,
            norm_weight,
            out_scale,
            conv_state,
            conv_weight,
            out,
            out_bf16,
            cfg=cfg,
            rows=rows,
            device=self.device,
            geometry=self.geometry,
        )
        key = self.key(qkv, z, A_log, dt_bias, ssm_state, state_indices, norm_weight, conv_state)
        if key not in self.compiled:
            self.compile(
                qkv,
                z,
                hidden,
                ba_weight,
                A_log,
                dt_bias,
                ssm_state,
                state_indices,
                cu_seqlens,
                num_accepted_tokens,
                norm_weight,
                out_scale,
                conv_state,
                conv_weight,
                out,
                out_bf16,
                rows=rows,
            )
        args = self._args(
            qkv,
            z,
            hidden,
            ba_weight,
            A_log,
            dt_bias,
            ssm_state,
            state_indices,
            cu_seqlens,
            num_accepted_tokens,
            norm_weight,
            out_scale,
            conv_state,
            conv_weight,
            out,
            out_bf16,
        )
        self.compiled[key](*args, cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream))
        return out if out is not None else self.outputs(*step_shape(state_indices))["qout"]
