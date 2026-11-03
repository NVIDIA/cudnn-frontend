# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host entry of the fused Gated DeltaNet VERIFY step (``kernel/gdn_verify_fused_f16.py``): validation with typed declines, the
plan-time compile and the launch.  No DSL import at module level -- the CuTe DSL is imported inside :meth:`GdnVerifyFused.compile`
after the version / architecture gate (python/cudnn/AGENTS.md Rule 7), so a host without it can still construct the entry and run
its declines.  The binding checks (:func:`check_binding`: dtype, device, shape, stride and alignment of every operand against the step
shape) run at compile time and again before EVERY launch -- the compile key pins the plan-time facts only, and a compiled DSL entry
accepts a tensor of another dtype or footprint silently, so the host is a launch's only guard.

The call contract is the FlashInfer Qwen 3.5 GDN block's verify form (its argument names and layouts), so an A/B against the
reference kernel (FlashInfer's built-in ``frost`` verify backend) is a one-line swap: ``entry(qkv, z, hidden, ba_weight, A_log, dt_bias, ssm_state, state_indices,
cu_seqlens, num_accepted_tokens, norm_weight, out_scale, conv_state, conv_weight, out=None, out_bf16=None) -> out`` (the fp8
output; ``out_bf16`` receives the bf16 ``y`` before quantization when given, else the entry's scratch).  Served geometry (declined
otherwise, by name): ``qkv`` [rows, 10240] bf16 (a row-strided view is fine), ``z`` [rows, 48, 128] bf16 (row-strided), ``hidden``
[rows, 5120] bf16 contiguous, ``ba_weight`` [96, 5120] bf16 contiguous, ``A_log`` / ``dt_bias`` [48] fp32 or bf16, the fp32 state
``[slots, 48, 128, 128]`` (slot-strided), ``state_indices`` [requests, T] int32 (any row stride; ``T`` = tokens per request, a
compile-time constant), ``cu_seqlens`` [requests + 1] and ``num_accepted_tokens`` [requests] int32 contiguous, ``norm_weight`` [128]
fp32 or bf16, ``out_scale`` one fp32, ``conv_state`` [slots, 10240, T + 2] bf16 CHANNEL-CONTIGUOUS (stored ``[slots, T + 2, 10240]``:
``stride(1) == 1``, the slot and column strides multiples of 8 elements), ``conv_weight`` [10240, 4] bf16 contiguous; every operand
16-byte aligned; an SM100-class part (compute capability 10.x) and a CuTe DSL >= 4.8.0.  Semantics (the per-token state layout): request
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
without one, the pick per requests (``default_cfg``: the cluster pair, two CTAs per V head, while the ``requests x 96`` CTAs have an SM
each on THIS device, one CTA per SM while the ``requests x 48`` CTAs have one, the two-CTAs-per-SM form past that); the device's SM
count, its shared memory per SM and its per-CTA opt-in maximum are read at
construction.  The launch knobs ``pdl`` / ``pdl_release`` set the config's ``PDL`` / ``PDL_RELEASE`` on top of whatever form serves (the
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

from .kernel.gdn_verify_fused_config import (
    CFG,
    CONV_WIDTH,
    HEAD_DIM,
    HIDDEN,
    QKV_CHANNELS,
    TICKET_STRIDE,
    V_HEADS,
    Cfg,
    default_cfg,
    resolve_cfg,
    validate_cfg,
    validate_tokens,
)

SUPPORTED_CC_MAJOR = 10  # SM100-class parts (compute capability 10.x); the DSL's target gate (cutedsl_arch_requirement_error) decides per part
SUPPORTED_CC_TEXT = "10.x"
DSL_FLOOR = (4, 8, 0)
NAME = "the fused GDN verify step"
#: the launch is grid ``(48, requests, 1)`` (``kernel/gdn_verify_fused_f16.py``), one grid row per request: CUDA caps grid Y at 65535
MAX_REQUESTS = 65535


def cc_supported(cc) -> bool:
    """Whether a ``(major, minor)`` compute capability is one this kernel serves: every SM100-class part (10.x)."""
    return cc is not None and int(cc[0]) == SUPPORTED_CC_MAJOR


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
    """The Rule-7 gate of this kernel: the library floor plus this kernel's own floor (4.8.0: the public wheel every consumer
    pins); None when the installed DSL serves."""
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
) -> None:
    """The binding checks, run at compile time AND before every launch: ``TypeError`` for a dtype (or a non-tensor), ``ValueError`` for
    an inconsistent shape, stride, alignment or device, ``NotImplementedError`` for a geometry / layout / step shape this kernel does not
    serve.  The compile key pins the plan-time facts only (the step shape, the strides the kernel is specialized on, the parameter
    dtypes, the config), and a compiled DSL entry accepts a tensor of another dtype or footprint silently, so these checks are a launch's
    only guard; host-only metadata arithmetic, no device work.  ``cfg`` (resolved) bounds the tokens per request by its shared-memory
    budget; ``rows`` (the step's token rows, ``>= cu_seqlens[requests]`` by the caller's contract) bounds the row operands and the outputs
    for a ragged or padded step instead of ``requests x T``.  :func:`check_support` adds the architecture gate."""
    import torch

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
    dev = qkv.device if isinstance(qkv, torch.Tensor) else None
    for name, t in tensors.items():
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{NAME}: {name} must be a torch.Tensor, got {type(t).__name__}")
        if t.device != dev:
            raise ValueError(f"{NAME}: every operand must live on one device; {name} is on {t.device}, qkv on {dev}")
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
    validate_tokens(cfg if cfg is not None else CFG, tokens)
    if rows is None:
        rows = requests * tokens  # the FlashInfer contract: every request carries exactly T tokens
        bound = f"requests x T = {rows}"
    else:
        if not isinstance(rows, int) or rows < 1:
            raise ValueError(f"{NAME}: rows must be a positive int (the step's token rows, at least cu_seqlens[requests]); got {rows!r}")
        bound = f"the step's rows = {rows}"
    # geometry
    if qkv.dim() != 2 or qkv.shape[0] < rows or qkv.shape[1] != QKV_CHANNELS:
        raise NotImplementedError(f"{NAME} serves qkv as [rows >= {bound}, {QKV_CHANNELS}] (16 K heads, 48 V heads x 128); got {tuple(qkv.shape)}")
    if qkv.stride(1) != 1 or qkv.stride(0) % 8:
        raise NotImplementedError(f"{NAME} needs qkv rows contiguous with a 16-byte-multiple row stride; got strides {tuple(qkv.stride())}")
    if z.dim() != 3 or z.shape[0] < rows or tuple(z.shape[1:]) != (V_HEADS, HEAD_DIM):
        raise NotImplementedError(f"{NAME} serves z as [rows >= {rows}, {V_HEADS}, {HEAD_DIM}]; got {tuple(z.shape)}")
    if tuple(z.stride()[1:]) != (HEAD_DIM, 1) or z.stride(0) % 8:
        raise NotImplementedError(f"{NAME} needs z heads contiguous with a 16-byte-multiple row stride; got strides {tuple(z.stride())}")
    if hidden.dim() != 2 or hidden.shape[0] < rows or hidden.shape[1] != HIDDEN or not hidden.is_contiguous():
        raise NotImplementedError(
            f"{NAME} serves a contiguous hidden of [rows >= {rows}, {HIDDEN}]; got {tuple(hidden.shape)} strides {tuple(hidden.stride())}"
        )
    if tuple(ba_weight.shape) != (2 * V_HEADS, HIDDEN) or not ba_weight.is_contiguous():
        raise NotImplementedError(f"{NAME} serves a contiguous ba_weight of [{2 * V_HEADS}, {HIDDEN}]; got {tuple(ba_weight.shape)}")
    if A_log.numel() < V_HEADS or dt_bias.numel() < V_HEADS or not A_log.is_contiguous() or not dt_bias.is_contiguous():
        raise NotImplementedError(f"{NAME} serves contiguous A_log / dt_bias of {V_HEADS} V heads; got {tuple(A_log.shape)} / {tuple(dt_bias.shape)}")
    if ssm_state.dim() != 4 or tuple(ssm_state.shape[1:]) != (V_HEADS, HEAD_DIM, HEAD_DIM):
        raise NotImplementedError(f"{NAME} serves the fp32 ssm_state as [slots, {V_HEADS}, {HEAD_DIM}, {HEAD_DIM}]; got {tuple(ssm_state.shape)}")
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
    if conv_state.dim() != 3 or conv_state.shape[1] != QKV_CHANNELS:
        raise NotImplementedError(f"{NAME} serves conv_state as [slots, {QKV_CHANNELS}, T + {CONV_WIDTH - 2}]; got {tuple(conv_state.shape)}")
    if conv_state.shape[2] != tokens + CONV_WIDTH - 2:
        raise NotImplementedError(
            f"{NAME} serves a conv_state of T + {CONV_WIDTH - 2} = {tokens + CONV_WIDTH - 2} history columns for T = {tokens} tokens per request; got {conv_state.shape[2]}"
        )
    css, csd, cst = (int(s) for s in conv_state.stride())
    if csd != 1 or cst % 8 or cst < QKV_CHANNELS or css % 8:
        raise NotImplementedError(
            f"{NAME} needs conv_state channel-contiguous (stored [slots, T + {CONV_WIDTH - 2}, {QKV_CHANNELS}]: stride(1) == 1, the slot and column strides multiples of 8 elements); got strides {tuple(conv_state.stride())}"
        )
    if tuple(conv_weight.shape) != (QKV_CHANNELS, CONV_WIDTH) or not conv_weight.is_contiguous():
        raise NotImplementedError(
            f"{NAME} serves a contiguous conv_weight [{QKV_CHANNELS}, {CONV_WIDTH}]; got {tuple(conv_weight.shape)} strides {tuple(conv_weight.stride())}"
        )
    for name, t in (("out", out), ("out_bf16", out_bf16)):
        if t is not None and (t.dim() != 3 or t.shape[0] < rows or tuple(t.shape[1:]) != (V_HEADS, HEAD_DIM) or not t.is_contiguous()):
            raise ValueError(
                f"{NAME}: {name} must be a contiguous [rows >= {bound}, {V_HEADS}, {HEAD_DIM}] tensor; got {tuple(t.shape)} strides {tuple(t.stride())}"
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
) -> None:
    """Typed declines, before any DSL import: the binding checks of :func:`check_binding` (``cfg`` / ``rows`` as there) plus the
    architecture gate (``NotImplementedError`` for a part this kernel does not serve).  ``device_cc`` overrides the compute-capability
    probe (tests)."""
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
    )
    dev = qkv.device
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


class GdnVerifyFused:
    """Compiled per (requests, tokens per request, strides, parameter dtypes, config) and shared by every GDN layer: all per-layer
    tensors are launch arguments.  ``cfg`` overrides the config as a whole; without it the entry picks the form per requests
    (``default_cfg``: the cluster pair, two CTAs per V head, while the ``requests x 96`` CTAs have an SM each on this device, one CTA
    per SM while the ``requests x 48`` CTAs have one, two CTAs per SM past that).  An entry built for a non-CUDA device (host tests)
    has no SM count to pick by and keeps the one-CTA form.  ``pdl`` /
    ``pdl_release`` (None = the form's own fields) are the launch knobs on top of the form that serves."""

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
            per_sm, optin = device_shared_memory(self.device)
            self.part = dict(SMEM_PER_SM_BYTES=per_sm, SMEM_OPTIN_BYTES=optin)
            self.sm_count = device_multiprocessor_count(self.device)
        self.cfg = self._on_this_part(cfg if cfg is not None else CFG)  # the override, or the one-CTA form the pick starts from
        self.forms: Dict[Tuple[Cfg, int], Cfg] = {}  # the resolved forms on this part, by (module form, requests)
        self.compiled: Dict[tuple, Any] = {}
        self.buffers: Dict[Tuple[int, int], Dict[str, Any]] = {}

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

    def cfg_for(self, requests: int) -> Cfg:
        """The config a plan for ``requests`` compiles with (its ``NORM_FORM`` resolved): the caller's ``cfg`` when one was given,
        else the pick per requests (``default_cfg`` with this device's SM count; the one-CTA form without a CUDA device)."""
        form = self.cfg if (self.cfg_given or self.sm_count is None) else default_cfg(requests, self.sm_count)
        key = (form, requests)
        if key not in self.forms:
            self.forms[key] = resolve_cfg(self._on_this_part(form), requests)
        return self.forms[key]

    @property
    def debug(self) -> bool:
        return self.cfg.DEBUG

    def outputs(self, requests: int, tokens: int):
        """The entry-owned buffers of a step shape: ``qout`` fp8 [rows, 48, 128] and ``out`` bf16 (the defaults when the caller passes
        no ``out`` / ``out_bf16``), ``counts`` int32 [requests * 512] (the conv-history reader tickets, zero between launches) and the
        debug buffers ``raw`` [rows, 48, 128] bf16, ``convout`` [rows, 10240] bf16, ``baout`` [rows, 96] bf16 (written under ``DEBUG``)."""
        shape = (requests, tokens)
        if shape not in self.buffers:
            import torch

            dev = self.device
            rows = requests * tokens
            self.buffers[shape] = dict(
                qout=torch.empty(rows, V_HEADS, HEAD_DIM, dtype=torch.float8_e4m3fn, device=dev),
                out=torch.empty(rows, V_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev),
                counts=torch.zeros(requests * TICKET_STRIDE, dtype=torch.int32, device=dev),
                raw=torch.zeros(rows, V_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev),
                convout=torch.zeros(rows, QKV_CHANNELS, dtype=torch.bfloat16, device=dev),
                baout=torch.zeros(rows, 2 * V_HEADS, dtype=torch.bfloat16, device=dev),
            )
        return self.buffers[shape]

    def key(self, qkv, z, A_log, dt_bias, ssm_state, state_indices, norm_weight, conv_state):
        """The compile key: plan-time facts only (the step shape, strides, parameter dtypes, the config of this requests count)."""
        requests, tokens = step_shape(state_indices)
        return (
            requests,
            tokens,
            self.cfg_for(requests),
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
            _flat(b["raw"]),
            _flat(b["convout"]),
            _flat(b["baout"]),
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
        """Validate (typed declines), gate the DSL, trace-compile the host entry for this (step shape, strides, dtypes) once.  ``rows``:
        the step's token rows for a ragged / padded step (see :func:`check_support`); not a compile key (the kernel reads ``cu_seqlens``)."""
        key = self.key(qkv, z, A_log, dt_bias, ssm_state, state_indices, norm_weight, conv_state)
        if key in self.compiled:
            return
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
            cfg=self.cfg_for(requests),
            rows=rows,
        )
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
                self.cfg_for(requests),
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
        :meth:`compile`).  Every launch re-runs the binding checks (:func:`check_binding`, against the form picked for this requests
        count) before the compiled kernel sees the pointers: the key pins no dtype, shape or alignment, and the DSL checks none."""
        import cuda.bindings.driver as cuda
        import torch

        cfg = None
        if isinstance(state_indices, torch.Tensor) and state_indices.dim() == 2 and state_indices.shape[0] >= 1:
            cfg = self.cfg_for(int(state_indices.shape[0]))  # the picked form bounds the tokens per request; a malformed table is check_binding's decline
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
