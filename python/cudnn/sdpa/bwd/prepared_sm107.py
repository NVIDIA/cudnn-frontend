# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan-time geometry, workspace regions and launch spec for the SM107 d=256 backward pointer hosts.

The three rows (``sdpa_bwd_sm107``, ``sdpa_bwd_sm107_fp8``, ``sdpa_bwd_sm107_mxfp8``) join the prepared-launch contract of ``bwd/prepared.py``:
``compile_plan`` turns an adapter's fixed plan facts into a ``BwdLaunchSpec`` whose ``fn`` is the positional tvm-ffi
entry of ONE compiled artifact (``kernels/sm107/prepared_host.py``) that runs the whole chain from device pointers.
Per call, ``bind()`` validates every operand against its ``Operand`` and hands the artifact a flat pointer frame; the
graph plan binds the normalized variant pack (``PreparedBwdLaunch``), the standalone adapter binds torch tensors
(``execute_standalone``).  Nothing here touches torch on the execute path.
"""

import math
from types import SimpleNamespace

from cudnn.frost.compiled_cache import positional_entry
from cudnn.sdpa.fwd.api_dsl import ws_align
from .prepared import ATTRIBUTES, BwdLaunchSpec, Operand, ROLES

# The half row binds the nine tensor operands plus two APPENDED slots, in the order they landed, each an operand exactly when
# its plan fact says so and None-specialized otherwise (the fixed-ABI rule the fp8 row's amax operands follow: ``bind()``
# requires an operand the plan asked for and refuses one it did not, slot by slot, the two independent of each other):
#   slot 9   ``seq_kv`` / ``SdpaBinding.seq_len_kv`` -- the caller's ``[B]`` int32 per-batch kv lengths under
#            ``seq_kv_lens_present`` (the standalone surface; a graph attribute, read strictly).
#   slot 10  ``delta`` -- STANDALONE-ONLY: the caller's ``[B, H_q, S_q_pad]`` fp32 ``rowsum(dO * O)`` under
#            ``SdpaBwdDslSm107(external_delta=True)``.  No graph binding carries it -- ``SdpaBinding`` has no such field, so the
#            half row's spec declares it ``standalone_only_roles`` and ``PreparedBwdLaunch`` frames it as absent -- and the
#            graph plan keeps the chain's own ``dot`` launch.
# The fp8 row appends the twelve scalar descales / scales of ``sdpa_fp8_backward`` and the four requested-only amax outputs;
# their role names ARE the ``SdpaBinding`` field names.  The fp8 / MXFP8 bodies take ONE uniform real kv length and compute
# their own delta, so their rows bind neither appended slot.
EXTERNAL_DELTA_ROLE = "delta"
ROLES_F16 = ROLES[:9] + ("seq_kv", EXTERNAL_DELTA_ROLE)
ATTRIBUTES_F16 = ATTRIBUTES[:9] + ("seq_len_kv", EXTERNAL_DELTA_ROLE)
# The half row's THD plan binds BOTH per-sequence length tensors (``(B,)`` lengths on the graph, ``(B,)`` or ``(B+1,)`` prefixes
# standalone -- ``bind()`` derives the form from numel) after the nine packed tensors; ``length_form=True`` on its spec.  It has
# no delta slot: the THD row declines ``external_delta`` (its delta is the chain's own ``dot_do_o`` over the packed O / dO,
# ``SdpaBwdDslSm107.check_support``), so no standalone-only role either.
ROLES_F16_THD = ROLES[:9] + ("seq_q", "seq_kv")
ATTRIBUTES_F16_THD = ATTRIBUTES[:9] + ("seq_len_q", "seq_len_kv")
FP8_SCALARS = (
    "descale_q",
    "descale_k",
    "descale_v",
    "descale_s",
    "scale_s",
    "descale_o",
    "descale_dO",
    "descale_dP",
    "scale_dQ",
    "scale_dK",
    "scale_dV",
    "scale_dP",
)
FP8_AMAX = ("amax_dQ", "amax_dK", "amax_dV", "amax_dP")
ROLES_FP8 = ROLES[:9] + FP8_SCALARS + FP8_AMAX
ATTRIBUTES_FP8 = ATTRIBUTES[:9] + FP8_SCALARS + FP8_AMAX
# The MXFP8 row appends the ``sdpa_mxfp8_backward`` ports the half node lacks: the transposed-quantization payloads, the
# half-precision dO and the seven F8_128x4 scale tensors (the SM100 MXFP8 adapter's role / attribute spelling; the attributes
# are the ``SdpaBinding`` field names).  ``o`` carries the ``o_f16`` port, ``do`` the ROWWISE e4m3 dO.
MXFP8_PAYLOADS = ("q_T", "k_T", "do_T", "do_f16")
MXFP8_SF = ("sf_q", "sf_q_T", "sf_k", "sf_k_T", "sf_v", "sf_do", "sf_do_T")
ROLES_MXFP8 = ROLES[:9] + MXFP8_PAYLOADS + MXFP8_SF
ATTRIBUTES_MXFP8 = ATTRIBUTES[:9] + ("q_T", "k_T", "dO_T", "dO_f16", "sf_q", "sf_q_T", "sf_k", "sf_k_T", "sf_v", "sf_dO", "sf_dO_T")

# Workspace region slots the hosts index (``prepared_host.R_*``), by ``_scratch_plan`` name.
_REGION_SLOTS_F16 = ("delta", "ds_ws", "seq_kv", "desc_words", "q_pad", "do_pad", "lse_pad", "k_pad", "v_pad", "dv_part", "dk_part", "dk_fold", "dv_fold")
_REGION_SLOTS_FP8 = (
    "delta",
    "ds_ws",
    "seq_kv",
    "desc_words",
    "q_pad",
    "do_pad",
    "lse_pad",
    "k_pad",
    "v_pad",
    "dv_part",
    "dk_part",
    "dq_ws",
    "q_bf16",
    "k_bf16",
    "amax_scratch",
)
# The MXFP8 row's slots (``prepared_host.R_DOT_PAD .. R_MX_DV_FOLD``): the shared nine, the dO_T staging copy, the five zero-filled SF
# pad slabs, the two dequantized bf16 stage-3 operands and the half row's partial / fold set.
_REGION_SLOTS_MXFP8 = (
    "delta",
    "ds_ws",
    "seq_kv",
    "desc_words",
    "q_pad",
    "do_pad",
    "lse_pad",
    "k_pad",
    "v_pad",
    "do_T_pad",
    "sf_q_pad",
    "sf_do_pad",
    "sf_doT_pad",
    "sf_k_pad",
    "sf_v_pad",
    "q_T_bf16",
    "k_T_bf16",
    "dv_part",
    "dk_part",
    "dk_fold",
    "dv_fold",
    # appended: the block-scaled dS chain (P-b) -- the second e4m3 payload, the two E8M0 atom tensors, the columnwise q_T / k_T
    # SF pads the block-scale GEMMs read (None under P-c, and where the shape is not ragged)
    "ds_dq",
    "sf_ds_dk",
    "sf_ds_dq",
    "sf_qT_pad",
    "sf_kT_pad",
)


def _dtype_name(torch_dtype) -> str:
    return str(torch_dtype).split(".")[-1]


def _payload_operand(desc, role):
    """``(geometry, Operand)`` of one tensor role: the kernels' compact ``[B, S, H, D]`` view of a declared logical-BHSD /
    BSHD-physical descriptor (a permute), or the contiguous ``[B, H, S_q]`` Stats."""
    shape, strides = tuple(int(x) for x in desc.shape), tuple(int(x) for x in desc.stride)
    if role == "stats":
        geom, alignment = (shape[:3], strides[:3]), 4
    else:
        geom, alignment = (tuple(shape[i] for i in (0, 2, 1, 3)), tuple(strides[i] for i in (0, 2, 1, 3))), 16
    shape, strides = geom
    span = 0 if not math.prod(shape) else 1 + sum((n - 1) * st for n, st in zip(shape, strides))
    return geom, Operand(_dtype_name(desc.dtype), shape, strides, span, alignment, desc.dtype.itemsize)


def _tensor_operands(api, roles=ROLES[:9]):
    """``(geometry, operands)`` of the tensor roles ``roles`` (default: the nine shared ones) -- see :func:`_payload_operand`."""
    geometry, operands = [], []
    for role in roles:
        geom, op = _payload_operand(getattr(api, role + "_desc"), role)
        geometry.append(geom)
        operands.append(op)
    return tuple(geometry), operands


def _regions(api, slots):
    """The workspace carve, in ``_scratch_plan`` order, as ``(offset, shape, strides)`` per host slot (None = not carved);
    the running 128-B-aligned offset must reproduce ``scratch_workspace_bytes()`` exactly."""
    offset, by_name = 0, {}
    for name, shape, dtype in api._scratch_shapes():
        shape = tuple(int(x) for x in shape)
        strides = tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))
        by_name[name] = (offset, shape, strides)
        offset += ws_align(math.prod(shape) * dtype.itemsize)
    if offset != api.scratch_workspace_bytes():
        raise RuntimeError(f"{api._NAME}: prepared workspace layout ({offset} B) differs from its advertised requirement ({api.scratch_workspace_bytes()} B)")
    unknown = set(by_name) - set(slots)
    if unknown:
        raise RuntimeError(f"{api._NAME}: scratch plan names without a host slot: {sorted(unknown)}")
    return tuple(by_name.get(name) for name in slots), offset


def _config(api):
    return (
        api.batch_size,
        api.h_q,
        api.h_kv,
        api.head_dim_qk,
        api.s_q_max,
        api.s_k_max,
        api._sq_pad,
        api._skv_pad,
        api._b_chunk,
        api._qh_chunk,
        bool(api._zero_ws),
        api.dtype.itemsize,
        api._bpe_ds,  # the dS workspace's bytes per element (fp8 row: 1 = e4m3, 2 = the bf16 twin)
        # the dQ rendering's B head group (`MatmulTemplateParams.b_head_group`, copied off the record in `compile()`): the GQA
        # group = one dQ launch per chunk, 1 = one per group member (`prepared_host._stage3`)
        int(api._dq_b_head_group),
    )


def _sm(api) -> int:
    from cudnn.frost.device import compute_capability, resolve_device

    major, minor = compute_capability(resolve_device(api.q_desc.device))
    return major * 10 + minor


def _dsl_dtype(torch_dtype):
    import cutlass

    return {"bfloat16": cutlass.BFloat16, "float16": cutlass.Float16, "float8_e4m3fn": cutlass.Float8E4M3FN}[_dtype_name(torch_dtype)]


def _delta_geometry(api):
    """The delta's ``(shape, strides)``: ``[B, H_q, S_q_pad]`` fp32 contiguous -- the ``dot_do_o`` layout, whether the chain
    carves it (``api._scratch_shapes()``'s ``delta`` entry) or the caller provides it (``external_delta_shape``)."""
    shape = tuple(int(x) for x in api.external_delta_shape)
    return shape, (shape[1] * shape[2], shape[2], 1)


def compile_plan(api, main, mm_dk, mm_dq):
    """The half row's spec.  ``main`` / ``mm_dk`` / ``mm_dq`` are the loaded templates (their ``_host`` functions are baked into the
    artifact; their ``FROST_SOURCE_DIGEST`` keys it together with the plan's geometry and carve).  Two appended operands follow
    the nine tensors, each bound exactly when its plan fact says so and independent of the other: slot 9, the caller's ``[B]``
    int32 per-batch kv lengths under ``seq_kv_lens_present``; slot 10, under ``external_delta``, the caller's delta (fp32, 16-B
    aligned, ``B * H_q * S_q_pad`` elements -- its exact layout is checked by ``SdpaBwdDslSm107.execute`` before the bind), with
    which the carve has no ``delta`` region and the artifact launches no ``dot``."""
    from .kernels.sm107.prepared_host import compile_host_f16

    geometry, operands = _tensor_operands(api)
    # Slot 9, the caller's per-batch kv lengths: a contiguous [B] int32 operand exactly when the plan was built with
    # seq_kv_lens_present (bind() then requires it, and refuses one on a plan built without -- the fixed-ABI rule the fp8 row's
    # amax operands follow).
    seq_kv_present = bool(api.seq_kv_lens_present)
    seq_kv_shape, seq_kv_strides = (api.batch_size,), (1,)
    operands.append(Operand("int32", seq_kv_shape, seq_kv_strides, api.batch_size, 4, 4) if seq_kv_present else None)
    # Slot 10, the caller's delta: the same rule under external_delta, independent of slot 9.
    delta_shape, delta_strides = _delta_geometry(api)
    external = bool(api.external_delta)
    operands.append(Operand("float32", delta_shape, delta_strides, math.prod(delta_shape), 16, 4) if external else None)
    # geometry[i] is operand i's static layout for EVERY slot, the two appended ones included (the sm80 host's convention, so the
    # next appended slot inherits no off-by-one): host_f16 views the lengths with geometry[9] and the delta with geometry[10] (the
    # dot_do_o layout the carved region has too).  Both entries ride whether or not their operand is bound, and both key the artifact.
    geometry += ((seq_kv_shape, seq_kv_strides), (delta_shape, delta_strides))
    regions, offset = _regions(api, _REGION_SLOTS_F16)
    config = _config(api)
    sm = _sm(api)
    dtype = _dsl_dtype(api.dtype)
    key = repr(
        (tuple(mod.FROST_SOURCE_DIGEST for mod in (main, mm_dk, mm_dq)), config, geometry, regions, _dtype_name(api.dtype), sm, seq_kv_present, external)
    )
    entry = compile_host_f16(
        main._host, mm_dk._host, mm_dq._host, config, geometry, regions, dtype, sm, key, seq_kv_present=seq_kv_present, external_delta=external
    )
    return _spec(api, entry, operands, offset, "sdpa_bwd_sm107", ROLES_F16, ATTRIBUTES_F16, scale_log2=False, standalone_only_roles=(EXTERNAL_DELTA_ROLE,))


def _thd_geometry(api):
    """``(geometry, operands)`` of the nine tensor roles on the half row's THD plan: PACKED ``[1, T, H, D]`` views at the plan's
    token capacities with the port's own head / token / element strides (a padded token stride is served as declared), the
    Stats in the forward's packing (``compact((T_q, H))`` token-major or ``compact((1, H, head_stride))`` head-major -- the compiled
    artifact binds the head stride as the third EXTENT)."""
    from .kernels.sm120.prepared_host import compact

    h, tq, tk = api.h_q, api._t_q_cap, api._t_kv_cap
    geometry, operands = [], []
    for role in ROLES[:9]:
        desc = getattr(api, role + "_desc")
        shape, strides = tuple(int(x) for x in desc.shape), tuple(int(x) for x in desc.stride)
        if role == "stats":
            geom = compact((tq, h)) if api._thd_lse_token_major else compact((1, h, api._thd_lse_head_stride or tq))
            alignment = 4
        else:
            tokens = tk if role in ("k", "v", "dk", "dv") else tq
            shape = (1, shape[1], tokens, shape[3])
            strides = (max(tokens, 1) * strides[2], *strides[1:])
            geom = tuple(shape[i] for i in (0, 2, 1, 3)), tuple(strides[i] for i in (0, 2, 1, 3))
            alignment = 16
        geometry.append(geom)
        shape, strides = geom
        span = 0 if not math.prod(shape) else 1 + sum((n - 1) * st for n, st in zip(shape, strides))
        operands.append(Operand(_dtype_name(desc.dtype), shape, strides, span, alignment, desc.dtype.itemsize))
    return tuple(geometry), operands


def compile_plan_thd(api, main, mm_dk, mm_dq):
    """The half row's THD spec (``api.thd``): packed geometry, the two length operands (``allowed_numels`` B / B+1), the
    workspace carve (``_REGION_SLOTS_F16`` with the pad / fold slots None: no staging under THD), and the SIBLING artifact
    ``prepared_host.host_f16_thd`` under its own cache key (``thd`` + the THD config: the dense artifact's key and ABI are
    untouched)."""
    from .kernels.sm107.prepared_host import compile_host_f16_thd

    b = api.batch_size
    geometry, operands = _thd_geometry(api)
    operands += [Operand("int32", (b,), (1,), b, 4, 4, (b, b + 1)) for _ in range(2)]
    regions, offset = _regions(api, _REGION_SLOTS_F16)
    # (B = sequences, H_q, H_kv, D, T_q cap, T_kv cap, S_q_pad (the q envelope padded), R_kv_cap (the blocked rows), head chunk,
    #  zero-fill, io itemsize, persistent grid clusters, S_q envelope, S_kv envelope, the dQ rendering's B head group) -- the
    #  host's `config`, all plan facts.
    config = (
        b,
        api.h_q,
        api.h_kv,
        api.head_dim_qk,
        api._t_q_cap,
        api._t_kv_cap,
        api._sq_pad,
        api._ws_rows_cap,
        api._qh_chunk,
        bool(api._zero_ws),
        api.dtype.itemsize,
        int(api._thd_units),
        api.s_q_max,
        api.s_k_max,
        # the dQ rendering's B head group (copied off the record in `compile()`, as the dense config does): the GQA group = one
        # dQ launch per head chunk, 1 = one per group member (`prepared_host._stage3_thd`)
        int(api._dq_b_head_group),
    )
    sm = _sm(api)
    dtype = _dsl_dtype(api.dtype)
    key = repr((tuple(mod.FROST_SOURCE_DIGEST for mod in (main, mm_dk, mm_dq)), "thd", config, geometry, regions, _dtype_name(api.dtype), sm))
    entry = compile_host_f16_thd(main._host, mm_dk._host, mm_dq._host, config, geometry, regions, dtype, sm, key)
    return _spec(api, entry, operands, offset, "sdpa_bwd_sm107", ROLES_F16_THD, ATTRIBUTES_F16_THD, scale_log2=False, length_form=True)


def compile_plan_fp8(api, main, mm_dk, mm_dq):
    """The fp8 row's spec: the nine tensors, twelve fp32 scalar operands (1 element, 4-byte aligned) and an amax operand per
    requested output (None-specialized otherwise, so ``bind()`` refuses an unrequested amax buffer and requires a requested one)."""
    from .kernels.sm107.prepared_host import compile_host_fp8

    geometry, operands = _tensor_operands(api)
    operands += [Operand("float32", (1,), (1,), 1, 4, 4) for _ in FP8_SCALARS]
    requested = tuple(name in api.amax_requested for name in FP8_AMAX)
    operands += [Operand("float32", (1,), (1,), 1, 4, 4) if flag else None for flag in requested]
    regions, offset = _regions(api, _REGION_SLOTS_FP8)
    config = _config(api)
    sm = _sm(api)
    grad_dtype = _dsl_dtype(api.grad_dtype)
    key = repr((tuple(mod.FROST_SOURCE_DIGEST for mod in (main, mm_dk, mm_dq)), config, geometry, regions, _dtype_name(api.grad_dtype), requested, sm))
    entry = compile_host_fp8(main._host, mm_dk._host, mm_dq._host, config, geometry, regions, grad_dtype, requested, sm, key)
    return _spec(api, entry, operands, offset, "sdpa_bwd_sm107_fp8", ROLES_FP8, ATTRIBUTES_FP8, scale_log2=True)


def compile_plan_mxfp8(api, main, mm_dk, mm_dq):
    """The MXFP8 row's spec: the nine shared tensors, the four extra payloads (BSHD geometry) and the seven scale-factor tensors
    as OPAQUE byte blobs (``Operand.opaque_bytes``: the graph may declare any dims with the right F8_128x4 byte total -- the
    C++ node rewrites two of their strides before lowering, so nothing but the byte count is trusted; the SM100 adapter's rule)."""
    from .kernels.sm107 import prepared_host as _host

    geometry, operands = _tensor_operands(api, ROLES[:9] + MXFP8_PAYLOADS)
    geometry = list(geometry)
    for name in MXFP8_SF:
        count = api._sf_expected_bytes(name)
        geometry.append(((count,), (1,)))
        operands.append(Operand("int8", (count,), (1,), count, 16, 1, opaque_bytes=True))
    regions, offset = _regions(api, _REGION_SLOTS_MXFP8)
    config = _config(api)
    sm = _sm(api)
    stage_sf_pads = bool(_host.MXFP8_STAGE_SF_PADS)  # read at plan build; part of the key (the poisoned-SF-pad RED twin flips it)
    ds_sf_policy = int(api._ds_policy)  # the adapter's dS policy (P-c bf16 dS | P-b block-scaled e4m3 dS): selects the chain the artifact runs
    if api._ds_block_scaled and int(api._dq_b_head_group) != 1:
        # The block-scale arm indexes its B scale-factor descriptor per A / C head and launches dQ once per GQA group member
        # (prepared_host._stage3_block_scale); a dQ record grouped by the GQA head (the plain renderings' single launch) would pair
        # Q heads with the wrong K head -- refuse here, in plain Python, before anything is compiled.
        raise ValueError(
            f"sdpa_bwd_sm107_mxfp8: the block-scaled dS chain's dQ record must keep b_head_group == 1 (one launch per GQA group member); "
            f"got {api._dq_b_head_group}"
        )
    key = repr(
        (
            tuple(mod.FROST_SOURCE_DIGEST for mod in (main, mm_dk, mm_dq)),
            config,
            tuple(geometry),
            regions,
            _dtype_name(api.grad_dtype),
            sm,
            stage_sf_pads,
            ds_sf_policy,
        )
    )
    entry = _host.compile_host_mxfp8(main._host, mm_dk._host, mm_dq._host, config, tuple(geometry), regions, sm, key, stage_sf_pads, ds_sf_policy)
    return _spec(api, entry, operands, offset, "sdpa_bwd_sm107_mxfp8", ROLES_MXFP8, ATTRIBUTES_MXFP8, scale_log2=True)


def _spec(api, entry, operands, offset, name, roles, attributes, *, scale_log2, standalone_only_roles=(), length_form=False):
    owner = SimpleNamespace(entry=entry, workspace_bytes=offset)
    fn = positional_entry(entry)
    if fn is None:
        raise NotImplementedError(f"{name} requires a positional tvm-ffi entry")
    return BwdLaunchSpec(
        owner,
        fn,
        tuple(operands),
        offset,
        int(api.q_desc.device.index or 0),
        api.scale_softmax,
        name,
        length_form=length_form,
        roles=roles,
        attributes=attributes,
        scale_log2=scale_log2,
        standalone_only_roles=tuple(standalone_only_roles),
    )


def execute_standalone(api, tensors, workspace, current_stream, scale):
    """The adapter's ``execute``: torch tensors in role order -> facts -> ``bind`` -> the artifact.  The nine tensor roles are
    held to the plan's exact geometry (Stats: contiguous with the plan's element count); scalars and amax bind by facts alone."""
    import torch
    from cudnn.sdpa.fwd.prepared import facts_of_tensor
    from .prepared import execute

    spec = api._prepared
    ws = facts_of_tensor(workspace)
    if ws is None or ws.dtype != "uint8" or not ws.contiguous or ws.span < spec.workspace_bytes or ws.device != (2, spec.device_index):
        raise ValueError(f"{spec.name} requires {spec.workspace_bytes} bytes of contiguous uint8 workspace on CUDA device {spec.device_index}")
    if current_stream is None:
        current_stream = torch.cuda.current_stream(api.q_desc.device).cuda_stream
    facts = dict(zip(spec.roles, map(facts_of_tensor, tensors)))
    geometry = []
    for i, op in enumerate(spec.operands):
        geom = None
        if op is not None and not op.opaque_bytes:
            if i == 5:
                stats = facts["stats"]
                if stats is not None and (not stats.contiguous or stats.numel != math.prod(op.shape)):
                    raise ValueError(f"{spec.name}: Stats must be contiguous with {math.prod(op.shape)} elements")
            elif len(op.shape) == 4:
                # The declared logical-BHSD geometry of the kernels' [B, S, H, D] operand view (the nine shared roles and the
                # MXFP8 row's four extra payloads); the fp8 row's [1] scalars and the opaque SF blobs bind by facts alone.
                geom = tuple(op.shape[j] for j in (0, 2, 1, 3)), tuple(op.strides[j] for j in (0, 2, 1, 3))
        geometry.append(geom)
    execute(spec, facts, ws.ptr, int(current_stream), scale=scale, geometry=geometry)
