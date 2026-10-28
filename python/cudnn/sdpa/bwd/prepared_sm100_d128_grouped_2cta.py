# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grouped THD d128 declarations on the cta_group::2 datapath: the grouped THD checks plus the pair envelope and nine maps.

Also the grouped route's DISPATCH behind the typed adapter field ``SdpaBwdDslSm100D128.cta_pair`` (the per-S heuristic of
``d128_grouped_2cta_contract``): :func:`resolve_cta_pair_body` reads the declaration and the device, :func:`check_support_selected` /
:func:`compile_plan_selected` run the chosen body's own support check and compile (the one-CTA grouped THD body or this pair body), and
:func:`cta_pair_resolution` hands the recorded decision to a driver (``token`` = ``pair=<mode>:<body>``).
"""

from dataclasses import replace
from functools import partial
from types import SimpleNamespace

from cudnn.frost.compiled_cache import positional_entry
from .config_sm100_d128_accum_2cta import resolve_diag_2cta
from .config_sm100_d128_accum_2cta_stail import (
    DBG_CLK,
    dbg_clk_records,
    dbg_clk_region_offset,
    resolve_debug_clk,
    resolve_mask_form,
    resolve_warp_roles,
    validate_fields_for_roles,
)
from .d128_grouped_2cta_contract import CTA_PAIR_BODY, CTA_PAIR_CLUSTER, ONE_CTA_BODY, PAIR_BODY, CtaPairDecline, resolve_cta_pair, select_cta_pair
from .d128_grouped_2cta_contract import THD as IMPLEMENTATION
from .d128_grouped_2cta_contract import WORKSPACE_SLACK, thd_workspace_layout, thd_workspace_offsets, validate_thd_bounds
from .d128_grouped_contract import dsl_version_decline
from .d128_thd_contract import KV_PORTS
from .prepared import ROLES, BwdLaunchSpec, Operand
from .prepared_sm100_d128_grouped import check_support as _check_support_grouped
from .prepared_sm100_d128_grouped import dkv_accum
from .prepared_sm100_d128_thd import THD_ATTRIBUTES, THD_ROLES, _GuardedEntry


def cta_pair(api):
    """The adapter record's per-S heuristic mode (``SdpaBwdDslSm100D128.cta_pair``: a ``CTA_PAIR_MODES`` name or ``None`` = "auto"), typed: a
    ``ValueError`` names the field and the domain, prefixed with the engine name."""
    try:
        return resolve_cta_pair(getattr(api, "cta_pair", None))
    except ValueError as error:
        raise ValueError(f"{api._NAME}: {error}") from None


def cta_pair_resolution(api):
    """The :class:`~.d128_grouped_2cta_contract.CtaPairResolution` recorded by :func:`resolve_cta_pair_body` on this instance (``None`` before
    ``check_support`` ran): the driver's ``pair=<mode>:<body>`` token, the facts read and the reason."""
    return getattr(api, "_cta_pair", None)


def _device_cc(api):
    """The compute capability of Q's device, or ``None`` when it is not a CUDA device (the pair predicates then fall back / decline by name)."""
    device = getattr(api.q_desc, "device", None)
    if getattr(device, "type", None) != "cuda":
        return None
    from cudnn.frost.device import compute_capability, resolve_device

    return tuple(compute_capability(resolve_device(device)))


_MAX_ACTIVE_PAIRS = {}  # device index -> the occupancy query's answer; plan-time only, one probe compile per device per process
_PROBE_FAILURES = {}  # device index -> the text of a query that failed (then the answer is 0 and the pair is not admitted)
# What a failing query raises: the DSL absent (ImportError) or without the query (AttributeError), the driver's error (RuntimeError), a bad
# device (ValueError).  Every one is the cluster predicate's typed decline, never an exception out of check_support.
_PROBE_ERRORS = (ImportError, AttributeError, RuntimeError, ValueError)


def _occupancy_query(index):
    """The DSL's cluster occupancy query on device ``index`` (``cuOccupancyMaxActiveClusters`` on a probe kernel, ``cutlass.utils.HardwareInfo``):
    the primary context is bound first through ``ensure_current_context`` (the execute path's helper).  Imports the DSL, so it runs behind the
    chain's version gate only (:func:`resolve_cta_pair_body`)."""
    import cutlass.utils

    from cudnn._device import ensure_current_context

    ensure_current_context(None, index)
    return int(cutlass.utils.HardwareInfo(index).get_max_active_clusters(CTA_PAIR_CLUSTER))


def max_active_pairs(device):
    """How many clusters of ``CTA_PAIR_CLUSTER`` CTAs the device admits at once (:func:`_occupancy_query`), 0 when the query itself fails (an
    unsupported driver or part, a DSL without the query): the failure is recorded (:func:`probe_failure`) and named in the cluster decline,
    never raised.  Cached per device index for the process: the query compiles a probe kernel, so it runs once -- and the selector consults it
    only after the cheaper predicates passed.  No device memory is owned and nothing blocks."""
    from cudnn.frost.device import resolve_device

    index = resolve_device(device)
    if index not in _MAX_ACTIVE_PAIRS:
        try:
            count = _occupancy_query(index)
        except _PROBE_ERRORS as error:
            _PROBE_FAILURES[index] = f"{type(error).__name__}: {error}"
            count = 0
        _MAX_ACTIVE_PAIRS[index] = count
    return _MAX_ACTIVE_PAIRS[index]


def probe_failure(device):
    """The text of a failed occupancy query on ``device`` (``None`` when it succeeded or never ran); the cluster decline / fallback names it."""
    from cudnn.frost.device import resolve_device

    return _PROBE_FAILURES.get(resolve_device(device))


def _probe_note(api):
    failure = probe_failure(api.q_desc.device)
    return "" if failure is None else f" (the occupancy query failed: {failure})"


def resolve_cta_pair_body(api):
    """Resolve the adapter's ``cta_pair`` for its declaration: the mode (typed), the declared S_q / S_kv (the batch's longest sequence on each
    side -- the ``(B, H, S, D)`` declaration a THD graph carries), Q's device cc (``None`` off CUDA) and, last and cached, the cluster
    admission.  The resolution is recorded on the instance (:func:`cta_pair_resolution`).  ``on`` with a failing predicate is the engine's
    typed decline, by name; a value outside the field's domain is a ``ValueError`` by name.  The chain's DSL floor (``DSL_MIN_VERSION``, both
    bodies) is gated FIRST, before any device read: the occupancy probe imports the DSL, so an absent or older wheel declines by name here
    instead of raising from inside the probe."""
    from cudnn.frost.buffers import cutedsl_state

    mode = cta_pair(api)
    decline = dsl_version_decline(*cutedsl_state())
    if decline is not None:
        raise NotImplementedError(f"{api._NAME}: grouped d128 {decline}")
    s_q_max, s_kv_max = int(api.q_desc.shape[2]), int(api.k_desc.shape[2])
    try:
        resolution = select_cta_pair(
            mode,
            thd=bool(api.thd),
            s_q_max=s_q_max,
            s_kv_max=s_kv_max,
            cc=_device_cc(api),
            max_active_pairs=lambda: max_active_pairs(api.q_desc.device),
        )
    except CtaPairDecline as error:
        note = _probe_note(api) if error.predicate == "cluster" else ""  # the query ran (cc passed on a CUDA device): its own failure, named
        raise NotImplementedError(f"{api._NAME}: {error}{note}") from None
    except ValueError as error:
        raise ValueError(f"{api._NAME}: {error}") from None
    if resolution.cluster_admitted is False:
        resolution = replace(resolution, reason=resolution.reason + _probe_note(api))
    api._cta_pair = resolution
    return resolution


def check_support_selected(api):
    """The grouped dispatch behind ``cta_pair``: resolve, then run the chosen body's OWN support predicates.  Under ``auto`` a pair predicate
    that declines sends the request to the one-CTA body with the decline recorded in the reason; under ``on`` that decline is the answer."""
    resolution = resolve_cta_pair_body(api)
    if resolution.body == PAIR_BODY:
        try:
            supported = check_support(api)
        except NotImplementedError as error:
            if resolution.mode != "auto":
                raise
            api._cta_pair = replace(resolution, body=ONE_CTA_BODY, reason=f"cta_pair=auto: the one-CTA body (the pair body declined: {error})")
        else:
            api._cta_pair = replace(resolution, pair_map=CTA_PAIR_BODY)
            return supported
    return _check_support_grouped(api)


def compile_plan_selected(api):
    """Compile the body :func:`check_support_selected` chose for this instance (the adapter runs ``check_support`` first)."""
    resolution = cta_pair_resolution(api)
    if resolution is None:
        raise RuntimeError(f"{api._NAME}: cta_pair is resolved by check_support(); compile_plan_selected needs it first")
    if resolution.body == PAIR_BODY:
        return compile_plan(api)
    from .prepared_sm100_d128_grouped import compile_plan as compile_plan_one_cta

    return compile_plan_one_cta(api)


def mask_form(api):
    """The adapter record's softmax MASK FORM on the pair body (``SdpaBwdDslSm100D128.mask_form``: a name of the s_tail body's ``CFG.MASK_FORMS``
    -- "fsel" (default) | "predicated" | "diag_mask_off" (DIAGNOSTIC: output wrong by design) -- or ``None`` = "fsel"; the non-default forms exist
    on the s_tail map only), typed: a ``ValueError`` names the field, the domain and the map."""
    try:
        return resolve_mask_form(getattr(api, "mask_form", None))
    except ValueError as error:
        raise ValueError(f"{api._NAME}: {error}") from None


def warp_roles(api):
    """The adapter record's ROLE SHAPE of the pair body (``SdpaBwdDslSm100D128.warp_roles``: a name of the s_tail body's ``CFG.WARP_ROLES`` -- "16w"
    (default: today's rendering) | "16w_fa4" (the alternate role shape: the softmax ships the dS exchange from its registers; warps 8-11 = the reduce group) -- or
    ``None`` = "16w"; "16w_fa4" exists on the s_tail map only), typed: a ``ValueError`` names the field, the domain and the map."""
    try:
        return resolve_warp_roles(getattr(api, "warp_roles", None))
    except ValueError as error:
        raise ValueError(f"{api._NAME}: {error}") from None


def diag_2cta(api):
    """The adapter record's DIAGNOSTIC probe (``SdpaBwdDslSm100D128.diag_2cta``: a ``CFG.DIAG_2CTA_NAMES`` name or ``None`` = no probe,
    the shipped body), typed: a ``ValueError`` names the field and the domain.  A set value makes the OUTPUT WRONG BY DESIGN."""
    try:
        return resolve_diag_2cta(getattr(api, "diag_2cta", None))
    except ValueError as error:
        raise ValueError(f"{api._NAME}: {error}") from None


def debug_clk(api):
    """The adapter record's STAMPS lever (``SdpaBwdDslSm100D128.debug_clk``: a bool, default False = the shipped rendering; True = the s_tail
    body brackets every mbarrier wait of one representative warp per role with %clock64 and dumps a per-CTA record into the workspace region
    this module carves; s_tail only), typed: a ``ValueError`` names the field, the type and the map.  Output CORRECT under both values."""
    try:
        return resolve_debug_clk(getattr(api, "debug_clk", None))
    except ValueError as error:
        raise ValueError(f"{api._NAME}: {error}") from None


def debug_clk_records(api, workspace):
    """The STAMPS dump of the LAST ``execute`` as an Int32 view ``[records, DBG_CLK.WORDS]`` of the caller's workspace tensor (the bytes the s_tail
    kernel wrote; no copy, no sync: the caller synchronizes); ``None`` when the lever is off.  Record r = the MAIN kernel's linear CTA
    (x + grid_x * (y + grid_y * z)); the word names are ``DBG_CLK.WORD`` (config = truth).  The payload origin is the prepared host's 256-B
    rounding of the workspace address (the same ``dbg_clk_region_offset`` arithmetic)."""
    region = getattr(api, "_dbg_clk_region", None)
    if region is None:
        return None
    import torch

    offset, n_rec = region
    base = dbg_clk_region_offset(workspace.data_ptr()) - workspace.data_ptr() + offset
    return workspace[base : base + n_rec * DBG_CLK.BYTES].view(torch.int32).view(n_rec, DBG_CLK.WORDS)


def check_support(api):
    """Grouped THD only: the shipping grouped checks, then the pair envelope and the nine-map workspace replace the 1-CTA ones."""

    def require(condition, reason):
        if not condition:
            raise NotImplementedError(f"{api._NAME}: 2-CTA datapath {reason}")

    require(api.thd, "requires THD (the dense twin of the pair body is not built)")
    require(api._uses_grouped_heads(), "requires grouped heads (Hq > Hkv); the MHA path has no pair body")
    _check_support_grouped(api)
    b, h, hk, sq, sk = api.batch_size, api.h_q, api.h_kv, api.s_q_max, api.s_k_max
    try:
        validate_thd_bounds(b, h, hk, sq, sk, api._thd_t_q, api._thd_t_kv, api._thd_t_q_phys, api._thd_t_kv_phys)
    except ValueError as error:
        require(False, str(error))
    api._grouped_workspace_regions, api._grouped_workspace_bytes = thd_workspace_layout(b, h, api._thd_t_q, api._thd_t_kv)
    # The body's whole typed field bundle, resolved HERE: a value outside a domain is a ValueError BY NAME at support time (never first at
    # compile, after auto committed to the pair); then the role shape's cross-field rule (config_sm100_d128_accum_2cta_stail.
    # validate_fields_for_roles): a combination that is not built is a typed DECLINE -- under cta_pair=auto the dispatch then serves the
    # one-CTA body with this text in its reason, under on it is the answer.
    mask_form(api)
    roles, probe, stamps, accum = warp_roles(api), diag_2cta(api), debug_clk(api), dkv_accum(api)
    try:
        validate_fields_for_roles(roles, probe, debug_clk=stamps, dkv_accum=accum)
    except ValueError as error:
        require(False, str(error))
    api._dbg_clk_region = None
    if debug_clk(api):
        # STAMPS (s_tail only; typed above): the Int32 record dump of every CTA of the MAIN grid, carved PAST the chain's regions at the first
        # 256-B boundary (config dbg_clk_region_offset / dbg_clk_records: the prepared host derives the same offset and count at launch);
        # the kernel writes every record at exit, so no zeroing; the caller's workspace grows by the region only under the lever.
        offset = dbg_clk_region_offset(thd_workspace_offsets(b, h, api._thd_t_q, api._thd_t_kv)[-1])
        n_rec = dbg_clk_records(b, h, sk)
        api._dbg_clk_region = (offset, n_rec)
        api._grouped_workspace_bytes = max(api._grouped_workspace_bytes, offset + n_rec * DBG_CLK.BYTES + WORKSPACE_SLACK)
    api._implementation = IMPLEMENTATION
    api._is_supported = True
    return True


def compile_plan(api):
    import cutlass
    import torch

    from cudnn.frost.device import compute_capability, resolve_device
    from .kernels.sm100.prepared_d128_grouped_2cta_host import compile_host

    b, h, hk = api.batch_size, api.h_q, api.h_kv
    operands = []
    for role in ROLES[:9]:
        desc = getattr(api, role + "_desc")
        if role == "stats":
            shape, strides = (
                ((api._thd_t_q_phys, h), (h, 1))
                if api.thd_stats_token_major
                else ((1, h, api._thd_head_pitch), (h * api._thd_head_pitch, api._thd_head_pitch, 1))
            )
        else:
            tokens, heads = (api._thd_t_kv_phys, hk) if role in KV_PORTS else (api._thd_t_q_phys, h)
            shape, strides = (1, tokens, heads, 128), (tokens * heads * 128, heads * 128, 128, 1)
        span = 1 + sum((int(n) - 1) * int(st) for n, st in zip(shape, strides))
        if span * desc.dtype.itemsize > 2**63 - 1:
            raise NotImplementedError(f"{api._NAME}: grouped {role} exceeds the signed pointer range")
        operands.append(Operand(str(desc.dtype).split(".")[-1], shape, strides, span, 4 if role == "stats" else 16, desc.dtype.itemsize))
    dtype = cutlass.BFloat16 if api.dtype == torch.bfloat16 else cutlass.Float16
    device = resolve_device(api.q_desc.device)
    cc = compute_capability(device)
    sm = cc[0] * 10 + cc[1]
    regions, required = api._grouped_workspace_regions, api._grouped_workspace_bytes
    if required != api.scratch_workspace_bytes():
        raise RuntimeError("grouped 2-CTA d128 compiled workspace disagrees with its advertised size")
    operands.extend(Operand("int32", (b,), (1,), b, 4, 4, (b, b + 1)) for _ in range(2))
    for name in api._thd_origin_dtypes:
        width = 8 if name == "int64" else 4
        operands.append(None if name is None else Operand(name, (b + 1,), (1,), b + 1, width, width))
    with torch.cuda.device(device):
        entry = compile_host(
            (b, h, hk, bool(api.thd_stats_token_major)),
            api._thd_origins,
            api._thd_origin_dtypes,
            dtype,
            api.is_causal,
            sm,
            diag_2cta=diag_2cta(api),
            debug_clk=debug_clk(api),
            mask_form=mask_form(api),
            dkv_accum=dkv_accum(api),
            warp_roles=warp_roles(api),
        )
    fn = positional_entry(entry)
    if fn is None:
        raise NotImplementedError("grouped 2-CTA d128 requires a positional tvm-ffi entry")
    prefix = (api._thd_t_q, api._thd_t_kv, api._thd_t_q_phys, api._thd_t_kv_phys, api.s_q_max, api.s_k_max, api._thd_head_pitch)
    fn = _GuardedEntry(partial(fn, *prefix), operands, required)
    owner = SimpleNamespace(
        entry=entry,
        workspace_bytes=required,
        workspace_regions=regions,
        implementation=IMPLEMENTATION,
        partial_dtype=str(api.dtype).split(".")[-1],
        dkv_accum=dkv_accum(api),
    )
    return BwdLaunchSpec(
        owner,
        fn,
        tuple(operands),
        required,
        device,
        api.scale_softmax,
        api._NAME,
        length_form=True,
        roles=THD_ROLES,
        attributes=THD_ATTRIBUTES,
        scale_log2=False,
        native_binding=True,
    )
