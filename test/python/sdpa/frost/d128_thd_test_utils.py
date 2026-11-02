# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Storage and independent CPU-reference fixtures for native half THD backward tests."""

from types import SimpleNamespace

import torch

ROLES = ("q", "k", "v", "o", "do", "stats", "dq", "dk", "dv")
Q_ROLES = ("q", "o", "do", "stats", "dq")
GRADS = ("dq", "dk", "dv")
CANARY = -7.0
D = 128
SCALE = D**-0.5


def reference_sequences(lens_q, lens_kv, heads, dtype, causal, seed, d_qk=D, d_v=D):
    """FP64 autograd on CPU, independently for each sequence.

    Inputs are rounded to the caller's storage dtype before the oracle sees
    them. Empty attention has O=0/LSE=-inf and exactly zero gradients; it
    never enters softmax. Nothing here knows the kernel's tile or scheduler.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    result = []
    for nq, nk in zip(lens_q, lens_kv, strict=True):
        draw = lambda n, width: (torch.randn(n, heads, width, generator=gen) * 0.5).to(dtype)
        q, k, v, do = draw(nq, d_qk), draw(nk, d_qk), draw(nk, d_v), draw(nq, d_v)
        if nq == 0 or nk == 0:
            o = torch.zeros(nq, heads, d_v, dtype=dtype)
            stats = torch.full((nq, heads), float("-inf"), dtype=torch.float32)
            dq, dk, dv = (torch.zeros_like(x, dtype=torch.float64) for x in (q, k, v))
        else:
            q64, k64, v64 = (x.transpose(0, 1).double().requires_grad_() for x in (q, k, v))
            scores = (q64 @ k64.transpose(-1, -2)) * d_qk**-0.5
            if causal:
                keep = torch.arange(nk)[None, :] <= torch.arange(nq)[:, None]
                scores = scores.masked_fill(~keep, float("-inf"))
            o64 = torch.softmax(scores, dim=-1) @ v64
            dq, dk, dv = (x.transpose(0, 1).detach() for x in torch.autograd.grad(o64, (q64, k64, v64), do.transpose(0, 1).double()))
            o = o64.transpose(0, 1).detach().to(dtype)
            stats = torch.logsumexp(scores.detach(), dim=-1).transpose(0, 1).float()
        result.append(dict(q=q, k=k, v=v, o=o, do=do, stats=stats, dq=dq, dk=dk, dv=dv))
    return result


def make_case(
    lens_q,
    lens_kv,
    *,
    heads=2,
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
    d_qk=D,
    d_v=D,
):
    """Each port has independent origins, including a nonzero first origin.

    The side's physical capacity includes every port's gaps, while the graph
    envelope only describes sequence lengths. Guarded carriers expose their
    full physical span; merely enlarging a backing allocation is insufficient.
    """
    lens_q, lens_kv = tuple(lens_q), tuple(lens_kv)
    assert len(lens_q) == len(lens_kv) and len(lens_q) > 0
    assert stats_layout in ("TH1", "HN") and units in ("elements", "tokens")
    origin_ports = set(ROLES if origin_ports is None else origin_ports)
    assert origin_ports <= set(ROLES)
    envelope = envelope or (max(max(lens_q), 1), max(max(lens_kv), 1))
    assert max(lens_q) <= envelope[0] and max(lens_kv) <= envelope[1]
    seqs = reference_sequences(lens_q, lens_kv, heads, dtype, causal, seed, d_qk, d_v)
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
    # HN deliberately has unused columns at the end of every head slab.
    head_pitch = ((cap_q + 63) // 64) * 64 + 16
    tensors, raw, offsets, declarations, dead_rows = {}, {}, {}, {}, {}
    for role in ROLES:
        cap = cap_q if role in Q_ROLES else cap_kv
        lens = lens_q if role in Q_ROLES else lens_kv
        is_stats = role == "stats"
        width = d_v if role in ("v", "o", "do", "dv") else d_qk
        dt = torch.float32 if is_stats else dtype
        shape = ((cap_q, heads) if stats_layout == "TH1" else (1, heads, head_pitch)) if is_stats else (1, cap, heads, width)
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
                if stats_layout == "TH1":
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
        token_stride = (heads if stats_layout == "TH1" else 1) if is_stats else heads * width
        values = (*origins[role], cap)
        multiplier = token_stride if units == "tokens" else 1
        if role in origin_ports:
            offsets[role] = torch.tensor(values, dtype=offset_dtype, device=device) * (1 if units == "tokens" else token_stride)
            declarations[role] = (offset_dtype, multiplier)
    return SimpleNamespace(
        d_qk=d_qk,
        d_v=d_v,
        scale=d_qk**-0.5,
        lens_q=lens_q,
        lens_kv=lens_kv,
        batch=len(lens_q),
        heads=heads,
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
        lengths={"seq_q": torch.tensor(lens_q, dtype=torch.int32, device=device), "seq_kv": torch.tensor(lens_kv, dtype=torch.int32, device=device)},
    )


def declaration(case, role):
    h, b = case.heads, case.batch
    s = case.envelope[0 if role in Q_ROLES else 1]
    if role == "stats":
        stride = (case.capacities[0] * h, 1, h, 1) if case.stats_layout == "TH1" else (h * case.head_pitch, case.head_pitch, 1, 1)
        return (b, h, s, 1), stride
    width = case.d_v if role in ("v", "o", "do", "dv") else case.d_qk
    return (b, h, s, width), (s * h * width, width, h * width, 1)


def make_adapter(case, api_type=None):
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
    api = (api_type or SdpaBwdDslSm100D128)(
        **samples,
        is_causal=case.causal,
        scale_softmax=case.scale,
        thd=True,
        max_total_seq_len_q=case.capacities[0],
        max_total_seq_len_kv=case.capacities[1],
        thd_stats_token_major=case.stats_layout == "TH1",
        thd_stats_head_stride=case.head_pitch if case.stats_layout == "HN" else None,
        thd_ragged_offsets=case.offset_declarations,
    )
    assert api.check_support()
    api.compile()
    assert api._prepared.native is not None
    return api


def adapter_arguments(case):
    # The standalone THD contract observes physical [1,T,H,D] carriers.
    args = {r + "_tensor": t for r, t in case.tensors.items()}
    args.update(seq_q_lens=case.lengths["seq_q"], seq_kv_lens=case.lengths["seq_kv"], thd_ragged_offsets=case.offsets)
    return args


def make_graph(case, engine_name="sdpa_bwd_sm100_d128", engine_id=20609):
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
    for role in ("seq_q", "seq_kv"):
        refs[role] = graph.tensor(name=role, dim=[case.batch, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
    outs = graph.sdpa_backward(
        name="native_thd_bwd",
        q=refs["q"],
        k=refs["k"],
        v=refs["v"],
        o=refs["o"],
        dO=refs["do"],
        stats=refs["stats"],
        attn_scale=case.scale,
        use_causal_mask=case.causal,
        use_padding_mask=True,
        seq_len_q=refs["seq_q"],
        seq_len_kv=refs["seq_kv"],
        max_total_seq_len_q=case.capacities[0],
        max_total_seq_len_kv=case.capacities[1],
    )
    for role, tensor in zip(GRADS, outs, strict=True):
        shape, stride = declaration(case, role)
        tensor.set_output(True).set_data_type(io).set_dim(shape).set_stride(stride)
        refs[role] = tensor
        attach_offsets(role, tensor)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    select_engine(graph, engine_name)
    engine, _ = graph.get_engine_and_knobs_at_index(graph._plan_index)
    assert engine == engine_id, engine
    graph.check_support()
    graph.build_plans()
    assert graph._compiled_plans[graph._plan_index]._prepared.spec.native is not None
    return graph, refs


def variant_pack(case, refs):
    return {refs[r]: t for r, t in case.tensors.items()} | {refs[r + "_ro"]: t for r, t in case.offsets.items()} | {refs[r]: t for r, t in case.lengths.items()}


def make_workspace(required):
    assert required > 0
    # Exercise a valid 16-byte address that needs internal 256-byte rounding.
    raw = torch.full((required + 544,), 0xA5, dtype=torch.uint8, device="cuda")
    offset = 256 + ((16 - raw.data_ptr()) % 256)
    return SimpleNamespace(raw=raw, tensor=raw.narrow(0, offset, required), start=offset, size=required)


def poison(case, workspace):
    workspace.raw.fill_(0xA5)
    for role in GRADS:
        case.tensors[role].fill_(CANARY)
        lens = case.lens_q if role == "dq" else case.lens_kv
        for start, length in zip(case.origins[role], lens, strict=True):
            case.tensors[role][0, start : start + length].fill_(float("nan"))


def check_case(case, workspace):
    from test_sdpa_bwd_dsl_sm107 import _TOL, _TOL_COS

    packed_got, packed_ref = {r: [] for r in GRADS}, {r: [] for r in GRADS}
    for role in GRADS:
        got = case.tensors[role][0].cpu()
        lens = case.lens_q if role == "dq" else case.lens_kv
        assert torch.all(got[case.dead_rows[role]] == CANARY), f"{role}: gap or capacity slack was overwritten"
        for i, (start, length) in enumerate(zip(case.origins[role], lens, strict=True)):
            actual, ref = got[start : start + length].float(), case.sequences[i][role].float()
            assert torch.isfinite(actual).all(), f"{role}/seq{i}: nonfinite gradient"
            if case.lens_q[i] == 0 or case.lens_kv[i] == 0:
                assert torch.all(actual == 0), f"{role}/seq{i}: empty attention must write exact zeros"
            torch.testing.assert_close(actual, ref, **_TOL[case.dtype], msg=lambda m: f"{role}/seq{i}: {m}")
            # Tiny exact derivatives (notably one-key dQ/dK) have ill-conditioned
            # cosine; they still obey the unchanged absolute/magnitude gate.
            if ref.numel() and torch.linalg.vector_norm(ref) > 1e-6:
                cosine = torch.nn.functional.cosine_similarity(actual.flatten(), ref.flatten(), dim=0)
                assert cosine > _TOL_COS, f"{role}/seq{i}: cosine={cosine.item()}"
            packed_got[role].append(actual.flatten())
            packed_ref[role].append(ref.flatten())
        a, b = torch.cat(packed_got[role]), torch.cat(packed_ref[role])
        if b.numel() and torch.linalg.vector_norm(b) > 1e-6:
            assert torch.nn.functional.cosine_similarity(a, b, dim=0) > _TOL_COS, role
    for role, raw in case.raw.items():
        assert torch.all(raw[:16].cpu() == CANARY) and torch.all(raw[-16:].cpu() == CANARY), f"{role}: storage guard overwritten"
    assert torch.all(workspace.raw[: workspace.start].cpu() == 0xA5), "workspace prefix overwritten"
    assert torch.all(workspace.raw[workspace.start + workspace.size :].cpu() == 0xA5), "workspace suffix overwritten"


def copy_case_storage(destination, source):
    """Change data AND device metadata without changing captured addresses."""
    assert (destination.envelope, destination.capacities, destination.head_pitch) == (source.envelope, source.capacities, source.head_pitch)
    for role in ROLES:
        destination.tensors[role].copy_(source.tensors[role])
    assert destination.offsets.keys() == source.offsets.keys()
    for role in destination.offsets:
        destination.offsets[role].copy_(source.offsets[role])
    for role in ("seq_q", "seq_kv"):
        destination.lengths[role].copy_(source.lengths[role])
    for field in ("lens_q", "lens_kv", "origins", "sequences", "dead_rows"):
        setattr(destination, field, getattr(source, field))
