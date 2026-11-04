# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cuDNN Frontend backend ``"frost_fe"`` of FlashInfer's experimental Qwen 3.5 GDN block (``flashinfer.experimental.qwen_3_5_gdn_block``):
the fused GDN DECODE step (:mod:`cudnn.linear_attention.frost.gdn_decode_fused`) and the fused GDN VERIFY step
(:mod:`cudnn.linear_attention.frost.gdn_verify_fused`) behind FlashInfer's backend registry, so the block's plans, plan cache, CUDA-graph
rules, reference tests and benchmark -- written for its built-in ``"frost"`` backend -- run these kernels unchanged.

Importing this module registers the backend (:func:`register_frost_fe`; FlashInfer's benchmark loads a backend module with ``--register``)::

    python benchmarks/bench_qwen_3_5_gdn_block.py --register cudnn.linear_attention.frost.flashinfer_gdn_backend --backends frost frost_fe

FlashInfer floor: the block's backend registry (``api.register_backend`` / ``get_backend``, the typed ``Qwen35GdnDecline`` kinds,
``support.device_cc``, ``Qwen35GdnPlanSpec``) -- FlashInfer main at or after PR #8131 (2026-10-31); no tagged FlashInfer release carried it
when this module was written (:data:`FLASHINFER_FLOOR`; importing this module without it raises ``ImportError`` naming the requirement),
and FlashInfer marks the block experimental: it may change or be removed without deprecation.

Served (:func:`support`, the registry's ``support_fn``: cheap, imports no kernel): both forms at the block's dense-27B geometry at
tensor-parallel rank 1 (d_model 5120, 48 V heads, 16 K heads, head dims 128, conv kernel 4) on SM100-class (compute capability 10.x) and
SM120-class (12.x) parts with a CuTe DSL >= 4.8.0.  The
``decode`` form takes up to 65535 rows (one grid row per request).  The ``verify`` form takes up to 65535 requests (one grid row each) of
``tokens_per_request`` >= 2 tokens, bounded above by the shared-memory budget of the forms the entry picks for that requests count on
THIS device (:func:`verify_max_tokens` = the entry's ``max_tokens_for``, the bound of the pick's whole ladder: the cluster pair, two CTAs
per V head, while the ``requests x 96`` CTAs have an SM each, the one-CTA-per-SM form while the ``requests x 48`` CTAs have one, the
two-CTAs-per-SM form past that -- on a cc 10.0 part 35, 8 and 45 tokens per request; on a 99 KB per-CTA opt-in (cc 12.0) the
shallower-ring forms of the same ladder, 17 tokens per request at every requests count, the pick moving with the step's token count).
Declined with FlashInfer's typed kinds: another form (``form``), another geometry (``geometry``, naming the fields), another device
(``device``: the compute capability, a failed device reading, or a shared memory that holds no decode or verify form), a missing or too-old DSL
(``dsl``), a step shape outside the envelope (``shape``, naming the deepest form and its bound).  A plan's parameter dtypes are not part
of FlashInfer's support question: the decode kernel reads ``A_log`` fp32
and ``dt_bias`` / ``norm_weight`` bf16 (FlashInfer's default ``param_dtypes``; callers convert), so a decode plan built for other dtypes
is declined by :func:`build` -- a ``NotImplementedError`` from ``prepare_qwen_3_5_gdn_block`` naming the dtypes, before any device work;
the verify kernel reads each of the three in fp32 or bf16 (every choice FlashInfer offers), compiled per dtype set.

The decode runner (:class:`FrostFeDecodeRunner`) adapts FlashInfer's ``Qwen35GdnInputs`` to the entry's call contract ``entry(rows, x,
cs, cw, al, dt, state, idx, z, w, scale, hidden, baw)``: ``qkv`` is ``x`` (a row-strided view is served), the conv pool's ``[num_slots, C,
K - 1]`` view is ``cs`` (both of its layouts, channel-contiguous taps and tap-contiguous channels, are addressed natively), ``ssm_state``
is ``state``, ``state_indices`` (int32 ``[rows]``; a slot <= 0 is a padding row -- zero outputs, no state touched -- in both conventions)
is ``idx``, ``out_scale`` the static FP8 scale the kernel reads on the device.  The caller's FP8 ``out`` and, when given, bf16
``out_bf16`` are bound into the entry's output slots before each launch (the plan's own bf16 scratch otherwise).

The verify runner (:class:`FrostFeVerifyRunner`) passes FlashInfer's verify operands to the entry by name -- its call contract IS the
block's verify form: ``state_indices`` ``[requests, T]`` with the per-token state layout (request ``n`` with ``a = num_accepted_tokens[n]``
starts from slot ``state_indices[n, a - 1]`` and writes the state after token ``t`` to slot ``state_indices[n, t]``, a slot <= 0 skipping
that write; a request whose source or conv slot is <= 0 is padding), ``cu_seqlens`` / ``num_accepted_tokens`` int32, the conv pool
``[num_slots, C, T + 2]`` CHANNEL-CONTIGUOUS (stored ``[num_slots, T + 2, C]``): the dim-first pool FlashInfer admits is a typed decline
at ``prepare`` (``NotImplementedError`` naming the layout), as in the built-in backend.  ``T`` is a compile-time constant of the kernel
(the plan's ``tokens_per_request``).

Both runners: ``prepare`` compiles for the binding's strides once (plan-time facts only; cheap once compiled; the entry's typed declines
propagate); ``run`` launches on torch's current stream and refuses a binding ``prepare`` never saw (``RuntimeError``), so a CUDA graph
captures only what an eager run compiled.  Compiled kernels are shared by every plan of one form, device and norm epsilon.  Nothing here
is imported at ``import cudnn`` time; FlashInfer is imported when the backend is registered (at this module's import) and torch inside
the functions that need it.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple

from . import gdn_decode_fused as _decode
from . import gdn_verify_fused as _verify
from .kernel.gdn_decode_fused_config import CONV_WIDTH, HEAD_DIM, HIDDEN, K_HEADS, V_HEADS
from .kernel.gdn_verify_fused_config import TOKENS_MIN as VERIFY_MIN_TOKENS

__all__ = [
    "BACKEND",
    "DECODE",
    "DECODE_MAX_ROWS",
    "FLASHINFER_FLOOR",
    "FORMS",
    "GEOMETRY",
    "VERIFY",
    "VERIFY_MAX_REQUESTS",
    "VERIFY_MIN_TOKENS",
    "FrostFeDecodeRunner",
    "FrostFeVerifyRunner",
    "build",
    "geometry_off",
    "register_frost_fe",
    "support",
    "verify_form",
    "verify_max_tokens",
]

BACKEND = "frost_fe"
DECODE = "decode"
VERIFY = "verify"
FORMS = (DECODE, VERIFY)
#: the FlashInfer this backend targets: the experimental Qwen 3.5 GDN block WITH its backend registry (``api.register_backend`` /
#: ``get_backend``, the typed ``Qwen35GdnDecline`` kinds, ``support.device_cc``, ``Qwen35GdnPlanSpec``), which FlashInfer's main branch
#: carries from PR #8131 (2026-10-31) on; no tagged FlashInfer release carried it when this module was written, and FlashInfer marks the
#: block experimental (it may change or be removed without deprecation)
FLASHINFER_FLOOR = "FlashInfer main at or after PR #8131 (2026-10-31; no tagged release carries the block's backend registry yet)"
#: the decode launch is grid ``(48, rows, 1)``: CUDA caps grid Y at 65535 -- the entry's own bound (``gdn_decode_fused.MAX_ROWS``; its
#: ``check_support`` declines past it), so the probe and the direct entry answer alike
DECODE_MAX_ROWS = _decode.MAX_ROWS
#: the verify launch is grid ``(48, requests, 1)``: the same cap, one grid row per request (``gdn_verify_fused.MAX_REQUESTS``)
VERIFY_MAX_REQUESTS = _verify.MAX_REQUESTS
#: the one geometry both kernels are written for, in FlashInfer's ``Qwen35GdnGeometry`` field names (the configs' geometry constants)
GEOMETRY: Dict[str, int] = dict(d_model=HIDDEN, num_v_heads=V_HEADS, num_k_heads=K_HEADS, head_k_dim=HEAD_DIM, head_v_dim=HEAD_DIM, conv_kernel=CONV_WIDTH)
_PARAM_DTYPE_NAMES = ("torch.float32", "torch.bfloat16", "torch.bfloat16")  # (A_log, dt_bias, norm_weight), the decode kernel's native dtypes
_ENTRIES = {DECODE: _decode, VERIFY: _verify}


def _what(form: str = DECODE) -> str:
    return f"qwen_3_5_gdn_block ({BACKEND}, {form})"


def geometry_off(geometry) -> Dict[str, Any]:
    """The fields of a FlashInfer ``Qwen35GdnGeometry`` that differ from the kernels' geometry (empty = served)."""
    return {name: getattr(geometry, name) for name, want in GEOMETRY.items() if getattr(geometry, name) != want}


def verify_form(device, requests: Optional[int] = None, norm_eps: float = 1e-6, tokens_per_request: Optional[int] = None):
    """The verify config a plan for ``requests`` (and ``tokens_per_request``; None = the minimum) compiles with on ``device``: the entry's
    own pick (``GdnVerifyFused.cfg_for``: the cluster pair -- two CTAs per V head -- while the ``requests x 96`` CTAs have an SM each, the
    one-CTA-per-SM form while the ``requests x 48`` CTAs have one, the two-CTAs-per-SM form past that, each where THIS device's shared
    memory holds it, else the regime's shallower-ring forms by the step's token count; the pick at one request when no requests count is
    given) with the device's shared-memory readings.  Reads device properties only; imports no kernel.  A device whose shared memory
    holds no verify form is the entry's ``NotImplementedError`` (``support`` maps it to kind ``device``)."""
    return _verify.GdnVerifyFused(device, norm_eps).cfg_for(
        1 if requests is None else int(requests), None if tokens_per_request is None else int(tokens_per_request)
    )


def verify_max_tokens(device, requests: Optional[int] = None) -> int:
    """The largest ``tokens_per_request`` the verify pick for ``requests`` holds on ``device``: the bound of the pick's whole ladder
    (``GdnVerifyFused.max_tokens_for``), the one ``support`` and the entry's own ``check_support`` answer alike."""
    return _verify.GdnVerifyFused(device, 1e-6).max_tokens_for(1 if requests is None else int(requests))


def support(device, geometry, form: str, requests: Optional[int], tokens_per_request: Optional[int]):
    """FlashInfer's ``support_fn``: ``None`` when served, else a typed ``Qwen35GdnDecline``.  Imports no kernel and no DSL (the DSL gate
    reads package metadata; the decode form's fit and the verify token bound read the device's shared-memory properties); ``requests`` /
    ``tokens_per_request`` are ``None`` when the caller asks without a step shape."""
    from flashinfer.experimental.qwen_3_5_gdn_block.support import Qwen35GdnDecline, device_cc

    if form not in FORMS:
        return Qwen35GdnDecline("form", f"the cuDNN Frontend backend {BACKEND!r} serves the forms {FORMS}; got {form!r}")
    entry = _ENTRIES[form]
    off = geometry_off(geometry)
    if off:
        return Qwen35GdnDecline(
            "geometry",
            f"the cuDNN Frontend GDN {form} kernel is written for d_model {HIDDEN}, {V_HEADS} V heads, {K_HEADS} K heads, head dims {HEAD_DIM} "
            f"and conv kernel {CONV_WIDTH} (the block's dense-27B geometry at tensor-parallel rank 1); got {off}",
        )
    cc = device_cc(device)
    if not entry.cc_supported(cc):
        found = "no CUDA device" if cc is None else f"SM{cc[0]}{cc[1]}"
        return Qwen35GdnDecline("device", f"the cuDNN Frontend GDN {form} kernel serves compute capability {entry.SUPPORTED_CC_TEXT}; found {found}")
    msg = entry._dsl_version_error()
    if msg is not None:
        return Qwen35GdnDecline("dsl", msg)
    if form == DECODE:
        if requests is not None and requests > DECODE_MAX_ROWS:
            return Qwen35GdnDecline(
                "shape",
                f"the cuDNN Frontend GDN decode kernel launches one grid row per request, at most {DECODE_MAX_ROWS} (CUDA's grid Y limit); got "
                f"requests={requests}",
            )
        try:
            # the one-CTA decode form must fit this device's per-CTA opt-in (74,848 B; every admitted part holds it): the entry's own
            # construction-time decline, answered here so the probe and build() agree on a part that holds no form
            _decode.first_fitting(_decode.ONE_CTA_FORMS, *_decode.device_shared_memory(device))
        except RuntimeError as exc:  # the device-attribute readers' failure; anything else is a bug and propagates
            return Qwen35GdnDecline("device", f"the cuDNN Frontend GDN decode kernel could not read this device's shared memory: {exc}")
        except NotImplementedError as exc:  # the entry's typed decline of a part whose shared memory holds no decode form
            return Qwen35GdnDecline("device", f"the cuDNN Frontend GDN decode kernel has no form for this device's shared memory: {exc}")
        return None
    if requests is not None and requests > VERIFY_MAX_REQUESTS:
        return Qwen35GdnDecline(
            "shape",
            f"the cuDNN Frontend GDN verify kernel launches one grid row per request, at most {VERIFY_MAX_REQUESTS} (CUDA's grid Y limit); got "
            f"requests={requests}",
        )
    if tokens_per_request is not None:
        if tokens_per_request < VERIFY_MIN_TOKENS:
            return Qwen35GdnDecline(
                "shape",
                f"the verify form carries at least one draft token per request (tokens_per_request >= {VERIFY_MIN_TOKENS}); one token per "
                "request is the decode form",
            )
        try:
            # the pick for this step shape: past the ladder's bound the deepest form, so its max_tokens() IS the ladder-wide bound
            cfg = verify_form(device, requests, float(geometry.norm_eps), tokens_per_request)
        except RuntimeError as exc:  # the device-attribute readers' failure; anything else is a bug and propagates (FlashInfer reports it by type)
            return Qwen35GdnDecline("device", f"the cuDNN Frontend GDN verify kernel could not read this device's shared memory or SM count: {exc}")
        except NotImplementedError as exc:  # the entry's typed decline of a part whose shared memory holds no verify form
            return Qwen35GdnDecline("device", f"the cuDNN Frontend GDN verify kernel has no form for this device's shared memory: {exc}")
        limit = cfg.max_tokens()
        if tokens_per_request > limit:
            form_name = "two CTAs per head" if cfg.TWO_CTAS_PER_HEAD else ("one CTA per SM" if cfg.CTAS_PER_SM == 1 else "two CTAs per SM")
            return Qwen35GdnDecline(
                "shape",
                f"the cuDNN Frontend GDN verify kernel's {form_name} form (the deepest pick at requests={requests}, NBUF {cfg.NBUF}) holds at "
                f"most {limit} tokens per request in this device's shared memory ({cfg.SMEM_OPTIN_BYTES} B per CTA); got "
                f"tokens_per_request={tokens_per_request}",
            )
    return None


def _param_dtypes_error(param_dtypes) -> Optional[str]:
    """Why a decode plan's ``(A_log, dt_bias, norm_weight)`` dtypes are not the kernel-native ones (``None`` = served)."""
    got = tuple(str(d) for d in param_dtypes)
    if got == _PARAM_DTYPE_NAMES:
        return None
    return (
        f"the cuDNN Frontend GDN decode kernel reads A_log {_PARAM_DTYPE_NAMES[0]}, dt_bias {_PARAM_DTYPE_NAMES[1]} and norm_weight "
        f"{_PARAM_DTYPE_NAMES[2]} (kernel-native; callers convert): FlashInfer's default param_dtypes; got {got}"
    )


# --------------------------------------------------------------------------- the runners

_COMPILED: Dict[Tuple[str, str, float], Dict[tuple, Any]] = {}
_COMPILED_LOCK = threading.Lock()


def _shared_compiled(spec) -> Dict[tuple, Any]:
    """The compiled-kernel table every plan of one form, device and norm epsilon shares (each entry's compile key carries the step
    shape, the strides, the config -- with this device's shared-memory readings and SM-count pick -- the parameter dtypes where they
    vary, and the epsilon)."""
    key = (str(spec.form), str(spec.device), float(spec.geometry.norm_eps))
    with _COMPILED_LOCK:
        return _COMPILED.setdefault(key, {})


def _not_compiled(form: str) -> RuntimeError:
    return RuntimeError(f"{_what(form)}: no kernel is compiled for this binding (its strides); run the same call once eagerly before CUDA-graph capture")


class FrostFeDecodeRunner:
    """FlashInfer's ``Qwen35GdnRunner`` for one decode plan over :class:`~cudnn.linear_attention.frost.gdn_decode_fused.GdnDecodeFused`.

    The entry's output slots (``outputs(rows)``: ``qout`` fp8, ``out`` bf16) are rebound to the caller's ``out`` / ``out_bf16`` before each
    launch, so the kernel writes FlashInfer's tensors directly; the reader tickets (``counts``) return to zero after every launch, which is
    what makes a captured launch replayable."""

    def __init__(self, spec):
        self.spec = spec
        self.kernel = _decode.GdnDecodeFused(spec.device, spec.geometry.norm_eps)
        self.kernel.compiled = _shared_compiled(spec)
        self.buffers = self.kernel.outputs(spec.rows)
        self.scratch_bf16 = self.buffers["out"]  # the bf16 normed value's home when the caller asks for none

    def _args(self, x) -> tuple:
        """``Qwen35GdnInputs`` in the entry's argument order: ``(rows, x, cs, cw, al, dt, state, idx, z, w, scale, hidden, baw)``."""
        return (
            self.spec.rows,
            x.qkv,
            x.conv_state,
            x.conv_weight,
            x.A_log,
            x.dt_bias,
            x.ssm_state,
            x.state_indices,
            x.z,
            x.norm_weight,
            x.out_scale,
            x.hidden,
            x.ba_weight,
        )

    def prepare(self, inputs) -> None:
        """Compile for this binding's strides (once per compile key; a hit returns at once).  The entry's typed declines propagate:
        ``NotImplementedError`` for a layout the kernel does not address natively, ``TypeError`` / ``ValueError`` for a dtype / shape /
        alignment FlashInfer's own operand checks let through."""
        self.kernel.compile(*self._args(inputs))

    def run(self, inputs) -> None:
        """Launch on torch's current stream into ``inputs.out`` (and ``inputs.out_bf16`` when set); ``conv_state`` / ``ssm_state`` advance in
        place.  A binding ``prepare`` never compiled is refused (``RuntimeError``), under CUDA-graph capture or not."""
        args = self._args(inputs)
        if self.kernel.key(args[0], args[1], args[2], args[6], args[8]) not in self.kernel.compiled:
            raise _not_compiled(DECODE)
        self.buffers["qout"] = inputs.out
        self.buffers["out"] = self.scratch_bf16 if inputs.out_bf16 is None else inputs.out_bf16
        self.kernel(*args)


class FrostFeVerifyRunner:
    """FlashInfer's ``Qwen35GdnRunner`` for one verify plan over :class:`~cudnn.linear_attention.frost.gdn_verify_fused.GdnVerifyFused`.

    The entry takes the block's verify operands by name, the caller's ``out`` / ``out_bf16`` among them (its own bf16 scratch when the
    caller asks for none); its reader tickets (``counts``) return to zero after every launch, so a captured launch replays.  The entry's
    buffers for this step shape are allocated at construction (outside CUDA-graph capture)."""

    def __init__(self, spec):
        self.spec = spec
        self.kernel = _verify.GdnVerifyFused(spec.device, spec.geometry.norm_eps)
        self.kernel.compiled = _shared_compiled(spec)
        self.buffers = self.kernel.outputs(spec.requests, spec.tokens_per_request)

    @staticmethod
    def _kwargs(x) -> Dict[str, Any]:
        """``Qwen35GdnInputs`` as the entry's keyword arguments (``cu_seqlens`` / ``num_accepted_tokens`` are set on the verify form)."""
        return dict(
            qkv=x.qkv,
            z=x.z,
            hidden=x.hidden,
            ba_weight=x.ba_weight,
            A_log=x.A_log,
            dt_bias=x.dt_bias,
            ssm_state=x.ssm_state,
            state_indices=x.state_indices,
            cu_seqlens=x.cu_seqlens,
            num_accepted_tokens=x.num_accepted_tokens,
            norm_weight=x.norm_weight,
            out_scale=x.out_scale,
            conv_state=x.conv_state,
            conv_weight=x.conv_weight,
            out=x.out,
            out_bf16=x.out_bf16,
        )

    def prepare(self, inputs) -> None:
        """Compile for this binding's strides (once per compile key).  The entry's typed declines propagate: ``NotImplementedError`` for a
        conv pool that is not channel-contiguous (the dim-first layout) or a tokens count the form cannot hold, ``TypeError`` /
        ``ValueError`` for a dtype / shape / alignment FlashInfer's own operand checks let through."""
        self.kernel.compile(**self._kwargs(inputs))

    def run(self, inputs) -> None:
        """Launch on torch's current stream into ``inputs.out`` (and ``inputs.out_bf16`` when set); ``conv_state`` / ``ssm_state`` advance in
        place in the per-token layout.  A binding ``prepare`` never compiled is refused (``RuntimeError``), under capture or not."""
        kw = self._kwargs(inputs)
        if (
            self.kernel.key(kw["qkv"], kw["z"], kw["A_log"], kw["dt_bias"], kw["ssm_state"], kw["state_indices"], kw["norm_weight"], kw["conv_state"])
            not in self.kernel.compiled
        ):
            raise _not_compiled(VERIFY)
        self.kernel(**kw)


def build(spec):
    """FlashInfer's ``build_fn``: the runner of one admitted plan (outside CUDA-graph capture; allocates the entry's buffers).
    ``NotImplementedError`` names a form, parameter dtypes (decode) or a device the kernels do not serve -- the dtypes first, before
    any device work, since they are not part of the support question."""
    if spec.form not in FORMS:
        raise NotImplementedError(f"{_what(spec.form)}: the cuDNN Frontend backend serves the forms {FORMS}")
    if spec.form == DECODE:
        msg = _param_dtypes_error(spec.param_dtypes)
        if msg is not None:
            raise NotImplementedError(f"{_what()}: {msg}")
    if spec.device.type != "cuda":
        raise NotImplementedError(f"{_what(spec.form)}: the cuDNN Frontend GDN {spec.form} kernel runs on CUDA devices; the plan's device is {spec.device}")
    import torch

    with torch.cuda.device(spec.device):
        return FrostFeDecodeRunner(spec) if spec.form == DECODE else FrostFeVerifyRunner(spec)


def register_frost_fe(*, replace: bool = False) -> str:
    """Register ``"frost_fe"`` with FlashInfer's registry (``api.register_backend``) and return the name.  Idempotent for this module's
    own backend (importing the module twice is a no-op); ``replace=True`` re-registers and drops the backend's cached plans; another
    backend holding the name is FlashInfer's ``ValueError`` unless ``replace`` is set."""
    from flashinfer.experimental.qwen_3_5_gdn_block import api

    if not (callable(getattr(api, "register_backend", None)) and callable(getattr(api, "get_backend", None))):
        raise ImportError(
            f"{__name__} needs FlashInfer's Qwen 3.5 GDN block backend registry (flashinfer.experimental.qwen_3_5_gdn_block.api.register_backend / "
            f"get_backend): {FLASHINFER_FLOOR}; the installed module has none"
        )
    current = api.get_backend(BACKEND)
    if current is not None and not replace and getattr(current.build_fn, "__module__", None) == __name__:
        return BACKEND
    api.register_backend(BACKEND, support, build, replace=replace)
    return BACKEND


try:
    register_frost_fe()
except ImportError as exc:
    raise ImportError(
        f"{__name__} needs FlashInfer with the experimental Qwen 3.5 GDN block and its backend registry (flashinfer.experimental.qwen_3_5_gdn_block; "
        f"{FLASHINFER_FLOOR}): {exc}"
    ) from exc
