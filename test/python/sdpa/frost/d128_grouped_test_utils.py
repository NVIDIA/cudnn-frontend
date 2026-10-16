# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent grouped references and guarded dense/THD d128 test storage.

Only the CPU oracle expands KV heads. GPU carriers always have Hkv heads.
The head-independent MHA poison, canary and numerical gates are reused intact.
"""

from types import SimpleNamespace

import torch

from d128_thd_test_utils import CANARY, D, GRADS, Q_ROLES, ROLES, SCALE, check_case, copy_case_storage, make_workspace, poison as _poison


def reference_sequences(lens_q, lens_kv, hq, hkv, dtype, causal, seed=149, inputs=None):
    """CPU FP64 autograd sums shared KV gradients before any storage cast."""
    assert hq > hkv > 0 and hq % hkv == 0
    gen = torch.Generator(device="cpu").manual_seed(seed)
    head_map = torch.arange(hq) // (hq // hkv)
    result = []
    for i, (nq, nk) in enumerate(zip(lens_q, lens_kv, strict=True)):
        if inputs is None:
            draw = lambda n, h: (torch.randn(n, h, D, generator=gen) * 0.5).to(dtype)
            q, k, v, do = draw(nq, hq), draw(nk, hkv), draw(nk, hkv), draw(nq, hq)
        else:
            q, k, v, do = (inputs[i][r].detach().cpu().to(dtype) for r in ("q", "k", "v", "do"))
        assert q.shape == do.shape == (nq, hq, D) and k.shape == v.shape == (nk, hkv, D)
        if nq == 0 or nk == 0:
            o = torch.zeros_like(q)
            stats = torch.full((nq, hq), float("-inf"), dtype=torch.float32)
            dq, dk, dv = (torch.zeros_like(x, dtype=torch.float64) for x in (q, k, v))
        else:
            q64, k64, v64 = (x.transpose(0, 1).double().requires_grad_() for x in (q, k, v))
            # index_select's backward sums gradients into ORIGINAL Hkv tensors.
            scores = (q64 @ k64.index_select(0, head_map).transpose(-1, -2)) * SCALE
            if causal:
                scores = scores.masked_fill(torch.arange(nk)[None, :] > torch.arange(nq)[:, None], float("-inf"))
            o64 = torch.softmax(scores, dim=-1) @ v64.index_select(0, head_map)
            dq, dk, dv = (x.transpose(0, 1).detach() for x in torch.autograd.grad(o64, (q64, k64, v64), do.transpose(0, 1).double()))
            o = o64.transpose(0, 1).detach().to(dtype)
            stats = torch.logsumexp(scores.detach(), dim=-1).transpose(0, 1).float()
        result.append(dict(q=q, k=k, v=v, o=o, do=do, stats=stats, dq=dq, dk=dk, dv=dv))
    return result


def cancellation_sequences(dtype, causal):
    """Exact P=1 separates native partial rounding from FP32 fold arithmetic.

    Even channels have group partials (large+0.5, -large, 1): rounding
    individual partials produces 1 rather than the full-FP64 result 1.5.
    Odd channels have (large, 1, -large): FP32 folding produces 1, while
    sequential native-dtype addition loses the middle contribution. No
    approximate softmax probability can confound this one-key fixture.
    """
    hq, hkv, nq, nk = 3, 1, 2, 1
    large = 256.0 if dtype == torch.bfloat16 else 2048.0
    inputs = {
        "q": torch.zeros(nq, hq, D, dtype=dtype),
        "k": torch.zeros(nk, hkv, D, dtype=dtype),
        "v": torch.zeros(nk, hkv, D, dtype=dtype),
        "do": torch.zeros(nq, hq, D, dtype=dtype),
    }
    inputs["do"][0, 0] = large
    inputs["do"][1, 0, ::2] = 0.5
    inputs["do"][0, 1, ::2] = -large
    inputs["do"][0, 2, ::2] = 1.0
    inputs["do"][0, 1, 1::2] = 1.0
    inputs["do"][0, 2, 1::2] = -large
    result = reference_sequences((nq,), (nk,), hq, hkv, dtype, causal, inputs=[inputs])
    partials = inputs["do"].double().sum(0).to(dtype)
    folded = torch.zeros((D,), dtype=torch.float32)
    native_fold = torch.zeros((D,), dtype=dtype)
    for partial in partials:
        folded += partial.float()
        native_fold += partial
    assert torch.all(folded == 1)
    assert torch.all(result[0]["dv"][0, 0, ::2] == 1.5)
    assert torch.all(native_fold[1::2] == 0)
    # The documented partial-storage semantics supersede a full-FP64 gradient
    # target ONLY for this deliberately ill-conditioned cancellation fixture.
    result[0]["dv"] = folded.to(dtype).double().view(nk, hkv, D)
    return result


def make_case(
    lens_q,
    lens_kv,
    *,
    hq=4,
    hkv=2,
    thd=True,
    sequences=None,
    dtype=torch.bfloat16,
    causal=False,
    stats_layout="TH1",
    offset_dtype=torch.int64,
    units="tokens",
    gaps=True,
    seed=149,
    layout_shift=0,
    envelope=None,
    capacities=None,
    origin_ports=None,
    device="cuda",
):
    """Each port has independent origins, including a nonzero first origin.

    The side's physical capacity includes every port's gaps, while the graph
    envelope only describes sequence lengths. Guarded carriers expose their
    full physical span; merely enlarging a backing allocation is insufficient.
    """
    lens_q, lens_kv = tuple(lens_q), tuple(lens_kv)
    assert len(lens_q) == len(lens_kv) and len(lens_q) > 0
    assert stats_layout in ("TH1", "HN") and units in ("elements", "tokens")
    origin_ports = set(ROLES if origin_ports is None else origin_ports) if thd else set()
    assert origin_ports <= set(ROLES)
    envelope = envelope or (max(max(lens_q), 1), max(max(lens_kv), 1))
    assert max(lens_q) <= envelope[0] and max(lens_kv) <= envelope[1]
    if not thd:
        assert len(set(lens_q)) == len(set(lens_kv)) == 1 and min(*lens_q, *lens_kv) > 0
        assert envelope == (lens_q[0], lens_kv[0])
    seqs = sequences or reference_sequences(lens_q, lens_kv, hq, hkv, dtype, causal, seed)
    origins, ends = {}, {}
    for index, role in enumerate(ROLES):
        lens = lens_q if role in Q_ROLES else lens_kv
        gapped = gaps and role in origin_ports
        cursor = 1 + index + layout_shift if gapped else 0
        starts = []
        for batch, length in enumerate(lens):
            starts.append(cursor)
            cursor += length + ((index + 2 * batch) % 5 + 1 if gapped else 0)
        origins[role], ends[role] = tuple(starts), cursor
    cap_q = max(ends[r] for r in Q_ROLES) + 8
    cap_kv = max(ends[r] for r in ROLES if r not in Q_ROLES) + 8
    # Without an explicit origin on a side, its carrier is the compact prefix
    # capacity min(B*Smax, declared total), including the positive empty case.
    if not origin_ports.intersection(Q_ROLES):
        cap_q = min(cap_q, len(lens_q) * envelope[0])
    if not origin_ports.difference(Q_ROLES):
        cap_kv = min(cap_kv, len(lens_kv) * envelope[1])
    if capacities is not None:
        assert capacities[0] >= cap_q and capacities[1] >= cap_kv
        cap_q, cap_kv = capacities
    assert origin_ports.intersection(Q_ROLES) or cap_q <= len(lens_q) * envelope[0]
    assert origin_ports.difference(Q_ROLES) or cap_kv <= len(lens_kv) * envelope[1]
    if not thd:
        cap_q, cap_kv = sum(lens_q), sum(lens_kv)
    # HN deliberately has unused columns at the end of every head slab.
    head_pitch = ((cap_q + 63) // 64) * 64 + 16
    tensors, raw, offsets, declarations, dead_rows = {}, {}, {}, {}, {}
    for role in ROLES:
        heads = hq if role in Q_ROLES else hkv
        cap = cap_q if role in Q_ROLES else cap_kv
        lens = lens_q if role in Q_ROLES else lens_kv
        is_stats = role == "stats"
        dt = torch.float32 if is_stats else dtype
        shape = ((cap_q, heads) if stats_layout == "TH1" else (1, heads, head_pitch)) if is_stats else (1, cap, heads, D)
        if is_stats and not thd:
            shape = (len(lens_q), hq, lens_q[0], 1)
        numel = 1
        for n in shape:
            numel *= n
        # The nonzero carrier storage offset remains 16-byte aligned.
        host_raw = torch.full((numel + 32,), CANARY, dtype=dt)
        host = host_raw[16:-16].view(shape)
        host.fill_(CANARY if role in GRADS else float("nan"))
        live = torch.zeros(cap, dtype=torch.bool)
        for i, (start, length) in enumerate(zip(origins[role], lens, strict=True)):
            live[start : start + length] = True
            if is_stats:
                if not thd:
                    host[i, :, :, 0].copy_(seqs[i][role].T)
                elif stats_layout == "TH1":
                    host[start : start + length].copy_(seqs[i][role])
                else:
                    host[0, :, start : start + length].copy_(seqs[i][role].T)
            elif role not in GRADS:
                host[0, start : start + length].copy_(seqs[i][role])
            else:
                host[0, start : start + length].fill_(float("nan"))
        raw[role] = host_raw.to(device=device)
        tensors[role] = raw[role][16:-16].view(shape)
        dead_rows[role] = ~live
        token_stride = (heads if stats_layout == "TH1" else 1) if is_stats else heads * D
        values = (*origins[role], cap)
        multiplier = token_stride if units == "tokens" else 1
        if role in origin_ports:
            offsets[role] = torch.tensor(values, dtype=offset_dtype, device=device) * (1 if units == "tokens" else token_stride)
            declarations[role] = (offset_dtype, multiplier)
    return SimpleNamespace(
        lens_q=lens_q,
        lens_kv=lens_kv,
        batch=len(lens_q),
        heads=hq,
        hq=hq,
        hkv=hkv,
        thd=thd,
        dtype=dtype,
        causal=causal,
        stats_layout=stats_layout,
        offset_dtype=offset_dtype,
        units=units,
        envelope=envelope,
        capacities=(cap_q, cap_kv),
        head_pitch=head_pitch,
        origins=origins,
        sequences=seqs,
        tensors=tensors,
        raw=raw,
        offsets=offsets,
        offset_declarations=declarations,
        dead_rows=dead_rows,
        lengths=(
            {"seq_q": torch.tensor(lens_q, dtype=torch.int32, device=device), "seq_kv": torch.tensor(lens_kv, dtype=torch.int32, device=device)} if thd else {}
        ),
    )


def declaration(case, role):
    h, b = (case.hq if role in Q_ROLES else case.hkv), case.batch
    s = case.envelope[0 if role in Q_ROLES else 1]
    if role == "stats":
        if not case.thd:
            return (b, h, s, 1), (h * s, s, 1, 1)
        stride = (case.capacities[0] * h, 1, h, 1) if case.stats_layout == "TH1" else (h * case.head_pitch, case.head_pitch, 1, 1)
        return (b, h, s, 1), stride
    return (b, h, s, D), (s * h * D, D, h * D, 1)


def make_adapter(case, *, compile_plan=True):
    from cudnn.api_base import TensorDesc
    from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128

    samples = {}
    for role in ROLES:
        shape, stride = declaration(case, role)
        samples["sample_" + role] = TensorDesc(
            dtype=torch.float32 if role == "stats" else case.dtype,
            shape=shape,
            stride=stride,
            stride_order=tuple(sorted(range(4), key=lambda i: (stride[i], shape[i]))),
            device=case.tensors["q"].device,
            name=role,
        )
    api = SdpaBwdDslSm100D128(
        **samples,
        is_causal=case.causal,
        scale_softmax=SCALE,
        thd=case.thd,
        max_total_seq_len_q=case.capacities[0] if case.thd else None,
        max_total_seq_len_kv=case.capacities[1] if case.thd else None,
        thd_stats_token_major=case.stats_layout == "TH1",
        thd_stats_head_stride=case.head_pitch if case.stats_layout == "HN" else None,
        thd_ragged_offsets=case.offset_declarations,
    )
    if not compile_plan:
        return api
    assert api.check_support()
    api.compile()
    assert api._prepared.native is not None
    assert api._prepared.artifact.implementation == implementation(case)
    return api


def implementation(case):
    return "accumulation_grouped_thd" if case.thd else "accumulation_grouped_dense"


def bound_tensors(case):
    if case.thd:
        return case.tensors
    # Make the public BHSD views during setup, never in the measured execute.
    return {
        r: t if r == "stats" else t.view(case.batch, case.envelope[0 if r in Q_ROLES else 1], case.hq if r in Q_ROLES else case.hkv, D).permute(0, 2, 1, 3)
        for r, t in case.tensors.items()
    }


def adapter_arguments(case):
    args = {r + "_tensor": t for r, t in bound_tensors(case).items()}
    if case.thd:
        args.update(seq_q_lens=case.lengths["seq_q"], seq_kv_lens=case.lengths["seq_kv"], thd_ragged_offsets=case.offsets)
    return args


def make_graph(case):
    import cudnn
    from frost_test_utils import select_engine

    io = cudnn.data_type.BFLOAT16 if case.dtype == torch.bfloat16 else cudnn.data_type.HALF
    graph = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    refs = {}

    def attach_offsets(role, tensor):
        if role not in case.offset_declarations:
            return
        dtype, mult = case.offset_declarations[role]
        ro = graph.tensor(
            name=role + "_ro",
            dim=[case.batch + 1, 1, 1, 1],
            stride=[1, 1, 1, 1],
            data_type=cudnn.data_type.INT32 if dtype == torch.int32 else cudnn.data_type.INT64,
        )
        tensor.set_ragged_offset(ro)
        if mult != 1:
            tensor.set_ragged_offset_multiplier(mult)
        refs[role + "_ro"] = ro

    for role in ROLES[:6]:
        shape, stride = declaration(case, role)
        refs[role] = graph.tensor(name=role, dim=shape, stride=stride, data_type=cudnn.data_type.FLOAT if role == "stats" else io)
        attach_offsets(role, refs[role])
    for role in case.lengths:
        refs[role] = graph.tensor(name=role, dim=[case.batch, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
    outs = graph.sdpa_backward(
        name="d128_grouped_bwd",
        q=refs["q"],
        k=refs["k"],
        v=refs["v"],
        o=refs["o"],
        dO=refs["do"],
        stats=refs["stats"],
        attn_scale=SCALE,
        use_causal_mask=case.causal,
        use_padding_mask=case.thd,
        seq_len_q=refs.get("seq_q"),
        seq_len_kv=refs.get("seq_kv"),
        max_total_seq_len_q=case.capacities[0] if case.thd else None,
        max_total_seq_len_kv=case.capacities[1] if case.thd else None,
    )
    for role, tensor in zip(GRADS, outs, strict=True):
        shape, stride = declaration(case, role)
        tensor.set_output(True).set_data_type(io).set_dim(shape).set_stride(stride)
        refs[role] = tensor
        attach_offsets(role, tensor)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    select_engine(graph, "sdpa_bwd_sm100_d128")
    engine, _ = graph.get_engine_and_knobs_at_index(graph._plan_index)
    assert engine == 20609, engine
    graph.check_support()
    graph.build_plans()
    spec = graph._compiled_plans[graph._plan_index]._prepared.spec
    assert spec.native is not None
    assert spec.artifact.implementation == implementation(case)
    return graph, refs


def variant_pack(case, refs):
    return (
        {refs[r]: t for r, t in bound_tensors(case).items()}
        | {refs[r + "_ro"]: t for r, t in case.offsets.items()}
        | {refs[r]: t for r, t in case.lengths.items()}
    )


def poison(case, workspace):
    """Poison all declared scratch, including native dK/dV partial regions."""
    _poison(case, workspace)
    # All-one bits are NaN in FP32, BF16 and FP16. Missing a partial write must
    # not be hidden by a small finite byte-pattern that falls below atol.
    workspace.tensor.fill_(0xFF)
