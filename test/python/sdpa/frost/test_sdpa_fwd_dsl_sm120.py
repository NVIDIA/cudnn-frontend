# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""End-to-end tests for the FROST SM120 DSL SDPA-forward engine against a torch reference."""

from __future__ import annotations

import math

import pytest
import torch

from test_utils import torch_fork_set_rng
from frost_test_utils import make_dense_stats, requires_blackwell_geforce, requires_dsl, _dsl_installed


def _is_sm120() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return (major, minor) in {(12, 0), (12, 1)}


pytestmark = pytest.mark.skipif(
    not _is_sm120(),
    reason="SM120 DSL SDPA engine requires an SM120 or SM121 device.",
)


def _require_dsl() -> None:
    try:
        import cudnn  # noqa: F401
        import cudnn.sdpa  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"SM120 DSL engine not available: {exc}")
    if not _dsl_installed():
        pytest.skip("cutlass/dsl not installed")


from frost_test_utils import select_engine as _select_engine  # noqa: F401

# Exact in fp16/bf16/fp32: pre-fills O/Stats storages in the THD harness so
# no-op paths (t_q == 0) can assert the buffers came back untouched.
_THD_SENTINEL = 2048.0


def _bhsd(
    batch: int,
    heads: int,
    sequence: int,
    head_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return logical BHSD backed by compact BSHD physical storage."""

    return torch.randn(
        batch,
        sequence,
        heads,
        head_dim,
        dtype=dtype,
        device="cuda",
    ).transpose(1, 2)


def _padded_bhsd(
    batch: int,
    heads: int,
    sequence: int,
    head_dim: int,
    dtype: torch.dtype,
    *,
    head_padding: int,
    sequence_padding: int,
    batch_padding: int,
    storage_offset: int = 0,
) -> torch.Tensor:
    """Return logical BHSD with independent padding between storage axes."""

    head_stride = head_dim + head_padding
    sequence_stride = heads * head_stride + sequence_padding
    batch_stride = sequence * sequence_stride + batch_padding
    shape = (batch, heads, sequence, head_dim)
    stride = (batch_stride, head_stride, sequence_stride, 1)
    storage_size = 1 + sum((size - 1) * axis_stride for size, axis_stride in zip(shape, stride))
    storage = torch.empty(storage_size + storage_offset, dtype=dtype, device="cuda")
    tensor = storage.as_strided(shape, stride, storage_offset=storage_offset)
    tensor.normal_()
    return tensor


def _ref_sdpa_full(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    is_causal: bool = False,
    causal_bottom_right: bool = False,
    window_size_left: int | None = None,
    window_size_right: int | None = None,
    seq_q_lens: torch.Tensor | None = None,
    seq_kv_lens: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """fp32 reference matching the SM120 DSL kernel's mask semantics.
    q/k/v are BHSD; GQA (h_q > h_kv) is handled by expanding K/V. ``sinks``
    is one logit per Q head, joining the softmax as a virtual column with no
    V row. ``window_size_right`` widens the diagonal to the right by R
    columns (inclusive; keep ``j <= lim + R``) — cuDNN's
    diagonal_band_right_bound, exclusive with ``is_causal``.
    """

    b, h_q, s_q, _ = q.shape
    _, h_kv, s_kv, _ = v.shape
    dev = q.device
    g = h_q // h_kv
    k_ref = k.repeat_interleave(g, dim=1).float()
    v_ref = v.repeat_interleave(g, dim=1).float()
    scores = torch.matmul(q.float(), k_ref.transpose(-1, -2)) * scale

    q_lens = seq_q_lens.flatten().to(torch.int64) if seq_q_lens is not None else torch.full((b,), s_q, dtype=torch.int64, device=dev)
    kv_lens = seq_kv_lens.flatten().to(torch.int64) if seq_kv_lens is not None else torch.full((b,), s_kv, dtype=torch.int64, device=dev)
    i = torch.arange(s_q, device=dev).view(1, 1, s_q, 1)
    j = torch.arange(s_kv, device=dev).view(1, 1, 1, s_kv)
    lim = i + (kv_lens - q_lens).view(b, 1, 1, 1) if causal_bottom_right else i
    masked = (i >= q_lens.view(b, 1, 1, 1)) | (j >= kv_lens.view(b, 1, 1, 1))
    if is_causal:
        masked = masked | (j > lim)
    if window_size_right is not None:
        masked = masked | (j > lim + window_size_right)
    if window_size_left is not None:
        masked = masked | (j < lim - window_size_left)
    scores = scores.masked_fill(masked, float("-inf"))

    if sinks is not None:
        sink_col = sinks.flatten().float().view(1, h_q, 1, 1).expand(b, h_q, s_q, 1)
        scores = torch.cat([scores, sink_col], dim=-1)
    probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)[..., :s_kv]  # sink mass has no V row
    o = torch.matmul(probs, v_ref)
    if not return_stats:
        return o
    lse = torch.logsumexp(scores, dim=-1)  # fully-masked rows -> -inf (sink-less)
    # Rows at/past seq_len_q[b] trim to -inf even with a sink.
    lse = lse.masked_fill((i >= q_lens.view(b, 1, 1, 1)).squeeze(-1), float("-inf"))
    return o, lse


def _run_case(
    *,
    batch: int = 1,
    h_q: int = 4,
    h_kv: int = 4,
    s_q: int = 128,
    s_kv: int = 128,
    head_dim: int = 128,
    head_dim_v: int | None = None,
    dtype: torch.dtype = torch.float16,
    q_tile: int | None = None,
    kv_tile: int | None = None,
    is_causal: bool = False,
    causal_bottom_right: bool = False,
    window_size_left: int | None = None,
    window_size_right: int | None = None,
    seq_q_lens: torch.Tensor | None = None,
    seq_kv_lens: torch.Tensor | None = None,
    scale: float | None = None,
    with_sink: bool = False,
    check_stats: bool = False,
    pack_gqa: bool | None = None,
    stats_layout: str = "contiguous",
) -> None:
    q = _bhsd(batch, h_q, s_q, head_dim, dtype)
    k = _bhsd(batch, h_kv, s_kv, head_dim, dtype)
    v = _bhsd(batch, h_kv, s_kv, head_dim if head_dim_v is None else head_dim_v, dtype)
    scale = 1.0 / math.sqrt(head_dim) if scale is None else scale
    sinks = torch.randn(1, h_q, 1, 1, dtype=torch.float32, device="cuda") if with_sink else None
    mask_kwargs = dict(
        is_causal=is_causal,
        causal_bottom_right=causal_bottom_right,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        seq_q_lens=seq_q_lens,
        seq_kv_lens=seq_kv_lens,
        sinks=sinks,
    )
    result = _run_dsl_graph(
        q,
        k,
        v,
        q_tile=q_tile,
        kv_tile=kv_tile,
        pack_gqa=pack_gqa,
        scale=scale,
        return_stats=check_stats,
        stats_layout=stats_layout,
        **mask_kwargs,
    )
    if check_stats:
        output, stats = result
        expected, expected_lse = _ref_sdpa_full(q, k, v, scale=scale, return_stats=True, **mask_kwargs)
        torch.testing.assert_close(stats.squeeze(-1), expected_lse, atol=2e-2, rtol=2e-2)
    else:
        output = result
        expected = _ref_sdpa_full(q, k, v, scale=scale, **mask_kwargs)
    torch.testing.assert_close(output.float(), expected, atol=0.1, rtol=5e-2)


def _apply_mask_kwargs(sdpa_kwargs, cudnn, *, is_causal, causal_bottom_right, window_size_left, window_size_right):
    """Translate the reference mask vocabulary into graph sdpa kwargs.

    A right bound makes the mask a diagonal BAND (with the requested
    alignment; a left bound rides along, cuDNN length L = offset W + 1);
    otherwise the causal / sliding-window flags apply.
    """
    if window_size_right is not None:
        sdpa_kwargs["diagonal_band_right_bound"] = window_size_right
        sdpa_kwargs["diagonal_alignment"] = cudnn.diagonal_alignment.BOTTOM_RIGHT if causal_bottom_right else cudnn.diagonal_alignment.TOP_LEFT
        if window_size_left is not None:
            sdpa_kwargs["diagonal_band_left_bound"] = window_size_left + 1
    else:
        if causal_bottom_right:
            sdpa_kwargs["use_causal_mask_bottom_right"] = True
        elif is_causal:
            sdpa_kwargs["use_causal_mask"] = True
        if window_size_left is not None:
            sdpa_kwargs["sliding_window_length"] = window_size_left + 1


def _run_dsl_graph(
    q_gpu: torch.Tensor,
    k_gpu: torch.Tensor,
    v_gpu: torch.Tensor,
    *,
    scale: float,
    o_gpu: torch.Tensor | None = None,
    is_causal: bool = False,
    causal_bottom_right: bool = False,
    window_size_left: int | None = None,
    window_size_right: int | None = None,
    seq_q_lens: torch.Tensor | None = None,
    seq_kv_lens: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    q_tile: int | None = None,
    kv_tile: int | None = None,
    pack_gqa: bool | None = None,
    return_stats: bool = False,
    stats_layout: str = "contiguous",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Build, select, and execute the SM120 FROST graph engine.

    Returns O, or ``(O, stats)`` with a ``(B, H, Sq, 1)`` fp32 stats tensor
    when ``return_stats`` is set. ``sinks`` is a ``(1, H, 1, 1)`` fp32
    per-Q-head sink-logit tensor.
    """

    _require_dsl()
    import cudnn

    from cudnn.sdpa.fwd.engines import engine_name

    dtype = q_gpu.dtype
    io_dtype = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    if o_gpu is None:
        batch, heads, sequence, _ = q_gpu.shape
        head_dim_v = v_gpu.shape[3]
        o_gpu = torch.empty(batch, sequence, heads, head_dim_v, dtype=dtype, device="cuda").transpose(1, 2)

    graph = cudnn.pygraph(
        io_data_type=io_dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q = graph.tensor_like(q_gpu, name="q")
    k = graph.tensor_like(k_gpu, name="k")
    v = graph.tensor_like(v_gpu, name="v")
    sdpa_kwargs = {
        "name": "sdpa",
        "q": q,
        "k": k,
        "v": v,
        "generate_stats": return_stats,
        "attn_scale": scale,
    }
    variant_pack = {q: q_gpu, k: k_gpu, v: v_gpu}

    _apply_mask_kwargs(
        sdpa_kwargs,
        cudnn,
        is_causal=is_causal,
        causal_bottom_right=causal_bottom_right,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )
    if seq_q_lens is not None or seq_kv_lens is not None:
        assert seq_q_lens is not None and seq_kv_lens is not None
        seq_q = graph.tensor_like(seq_q_lens, name="seq_q")
        seq_kv = graph.tensor_like(seq_kv_lens, name="seq_kv")
        sdpa_kwargs.update(
            use_padding_mask=True,
            seq_len_q=seq_q,
            seq_len_kv=seq_kv,
        )
        variant_pack.update({seq_q: seq_q_lens, seq_kv: seq_kv_lens})
    if sinks is not None:
        sink_t = graph.tensor_like(sinks, name="sink")
        sdpa_kwargs["sink_token"] = sink_t
        variant_pack[sink_t] = sinks

    o, stats = graph.sdpa(**sdpa_kwargs)
    o.set_output(True).set_dim(o_gpu.shape).set_stride(o_gpu.stride())
    batch, heads, sequence, _ = q_gpu.shape
    stats_gpu = None
    if return_stats:
        assert stats is not None
        stats_gpu = make_dense_stats(batch, heads, sequence, stats_layout)
        stats.set_output(True).set_dim(stats_gpu.shape).set_stride(stats_gpu.stride())
        stats.set_data_type(cudnn.data_type.FLOAT)
        variant_pack[stats] = stats_gpu
    tiles = None
    if q_tile is not None or kv_tile is not None:
        tiles = (q_tile, kv_tile)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    _select_engine(graph, engine_name(arch="sm120"), tiles=tiles, pack_gqa=pack_gqa)
    graph.check_support()
    graph.build_plans()
    # Honest workspace: the SM120 kernel None-specializes the LSE store, so a
    # stats-less graph needs no dummy-LSE chunk — dense workspace is always 0.
    expected_workspace = 0
    assert graph.get_workspace_size() == expected_workspace

    variant_pack[o] = o_gpu
    graph.execute(
        variant_pack,
        torch.empty(max(expected_workspace, 1), dtype=torch.uint8, device="cuda"),
    )
    torch.cuda.synchronize()
    if return_stats:
        return o_gpu, stats_gpu
    return o_gpu


def _pack_thd(seqs: list[torch.Tensor], s_max: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack per-sequence ``(1, H, L_i, D)`` tensors into THD storage.

    Returns ``(dense_view, storage, ragged_offset)``: the ``(B, H, S_max, D)``
    packed-stride view over dense-sized storage whose first ``T*H*D`` elements
    hold the packed tokens, the raw storage, and the ``(B+1, 1, 1, 1)`` int64
    element-unit offsets (``cu_tokens * H * D``).
    """

    b, h, d = len(seqs), seqs[0].shape[1], seqs[0].shape[3]
    cu = [0]
    for s in seqs:
        cu.append(cu[-1] + s.shape[2])
    storage = torch.zeros(b * s_max * h * d, dtype=seqs[0].dtype, device="cuda")
    packed = storage[: max(cu[-1], 1) * h * d].view(max(cu[-1], 1), h, d)
    for i, s in enumerate(seqs):
        packed[cu[i] : cu[i + 1]].copy_(s[0].permute(1, 0, 2))
    view = storage.as_strided((b, h, s_max, d), (s_max * h * d, d, h * d, 1))
    ro = (torch.tensor(cu, dtype=torch.int64, device="cuda") * h * d).view(b + 1, 1, 1, 1)
    return view, storage, ro


def _run_thd_case(
    *,
    seq_q_lens: list[int],
    seq_kv_lens: list[int],
    h_q: int = 8,
    h_kv: int = 8,
    head_dim: int = 64,
    head_dim_v: int | None = None,
    dtype: torch.dtype = torch.float16,
    is_causal: bool = False,
    causal_bottom_right: bool = False,
    window_size_left: int | None = None,
    window_size_right: int | None = None,
    with_sink: bool = False,
    check_stats: bool = False,
    stats_layout: str = "token_major",
    cu_lens: bool = False,
) -> None:
    """Run a THD (ragged) graph on the SM120 engine vs per-sequence references.

    ``stats_layout`` selects the ragged Stats declaration: ``token_major``
    (``[t, h]``, sequence stride ``h_q``) or ``head_major`` (``[h, t]``,
    sequence stride 1 with a padded token-capacity head stride — FlashAttention's
    ``softmax_lse`` layout, mirroring PR #462's harness convention).
    """

    _require_dsl()
    import cudnn

    from cudnn.sdpa.fwd.engines import engine_name

    batch = len(seq_q_lens)
    s_q_max = max(max(seq_q_lens), 1)
    s_kv_max = max(max(seq_kv_lens), 1)
    d_v = head_dim if head_dim_v is None else head_dim_v
    scale = 1.0 / math.sqrt(head_dim)
    q_seqs = [_bhsd(1, h_q, max(n, 1), head_dim, dtype)[:, :, :n] for n in seq_q_lens]
    k_seqs = [_bhsd(1, h_kv, max(n, 1), head_dim, dtype)[:, :, :n] for n in seq_kv_lens]
    v_seqs = [_bhsd(1, h_kv, max(n, 1), d_v, dtype)[:, :, :n] for n in seq_kv_lens]
    q_view, _, q_ro = _pack_thd([s.contiguous() for s in q_seqs], s_q_max)
    k_view, _, k_ro = _pack_thd([s.contiguous() for s in k_seqs], s_kv_max)
    v_view, _, v_ro = _pack_thd([s.contiguous() for s in v_seqs], s_kv_max)
    o_view, o_storage, o_ro = _pack_thd([torch.zeros(1, h_q, max(n, 1), d_v, dtype=dtype, device="cuda")[:, :, :n] for n in seq_q_lens], s_q_max)
    # SENTINEL fill: the kernel writes every valid packed O token (compared
    # against the reference below); everything else — the whole buffer when
    # t_q == 0 — must come back untouched.
    o_storage.fill_(_THD_SENTINEL)
    sq_t = torch.tensor(seq_q_lens, dtype=torch.int32, device="cuda").view(batch, 1, 1, 1)
    skv_t = torch.tensor(seq_kv_lens, dtype=torch.int32, device="cuda").view(batch, 1, 1, 1)

    # cu_lens: bind the (B+1,) prefix-sum form (cu_seq_len_q/kv, cuDNN 9.24+)
    # instead of per-batch lengths.
    def _prefix(lens):
        cu = [0]
        for n in lens:
            cu.append(cu[-1] + n)
        return torch.tensor(cu, dtype=torch.int32, device="cuda").view(batch + 1, 1, 1, 1)

    cuq_t, cukv_t = (_prefix(seq_q_lens), _prefix(seq_kv_lens)) if cu_lens else (None, None)
    sinks = torch.randn(1, h_q, 1, 1, dtype=torch.float32, device="cuda") if with_sink else None

    io_dtype = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    graph = cudnn.pygraph(io_data_type=io_dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tq = graph.tensor_like(q_view, name="q")
    tk = graph.tensor_like(k_view, name="k")
    tv = graph.tensor_like(v_view, name="v")
    rq = graph.tensor_like(q_ro, name="q_ro")
    rk = graph.tensor_like(k_ro, name="k_ro")
    rv = graph.tensor_like(v_ro, name="v_ro")
    ro = graph.tensor_like(o_ro, name="o_ro")
    tq.set_ragged_offset(rq)
    tk.set_ragged_offset(rk)
    tv.set_ragged_offset(rv)
    sq = graph.tensor_like(cuq_t, name="cu_seq_q") if cu_lens else graph.tensor_like(sq_t, name="seq_q")
    skv = graph.tensor_like(cukv_t, name="cu_seq_kv") if cu_lens else graph.tensor_like(skv_t, name="seq_kv")
    sdpa_kwargs = dict(
        name="sdpa",
        q=tq,
        k=tk,
        v=tv,
        generate_stats=check_stats,
        attn_scale=scale,
        use_padding_mask=True,
    )
    if cu_lens:
        sdpa_kwargs.update(cu_seq_len_q=sq, cu_seq_len_kv=skv)
    else:
        sdpa_kwargs.update(seq_len_q=sq, seq_len_kv=skv)
    _apply_mask_kwargs(
        sdpa_kwargs,
        cudnn,
        is_causal=is_causal,
        causal_bottom_right=causal_bottom_right,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )
    variant_pack = {
        tq: q_view,
        tk: k_view,
        tv: v_view,
        rq: q_ro,
        rk: k_ro,
        rv: v_ro,
        ro: o_ro,
        sq: (cuq_t if cu_lens else sq_t),
        skv: (cukv_t if cu_lens else skv_t),
    }
    if sinks is not None:
        st = graph.tensor_like(sinks, name="sink")
        sdpa_kwargs["sink_token"] = st
        variant_pack[st] = sinks
    o, stats = graph.sdpa(**sdpa_kwargs)
    o.set_output(True).set_dim(list(o_view.shape)).set_stride(list(o_view.stride()))
    o.set_ragged_offset(ro)
    variant_pack[o] = o_view
    stats_storage = None
    t_cap = max(64, -(-sum(seq_q_lens) // 64) * 64)
    if check_stats:
        assert stats is not None
        stats.set_output(True)
        stats.set_data_type(cudnn.data_type.FLOAT)
        if stats_layout == "head_major":
            # [h, t]: tokens contiguous within a head, heads strided by the
            # padded token capacity; offsets = cu_q * stride_s = cu_q.
            stats_storage = torch.full((h_q * t_cap,), _THD_SENTINEL, dtype=torch.float32, device="cuda")
            stats.set_dim((batch, h_q, s_q_max, 1)).set_stride((h_q * t_cap, t_cap, 1, 1))
            stats_ro_t = (q_ro.flatten() // (head_dim * h_q)).view(batch + 1, 1, 1, 1).contiguous()
        else:
            # [t, h]: heads contiguous within a token; offsets = cu_q * h_q.
            stats_storage = torch.full((batch * s_q_max * h_q,), _THD_SENTINEL, dtype=torch.float32, device="cuda")
            stats.set_dim((batch, h_q, s_q_max, 1)).set_stride((s_q_max * h_q, 1, h_q, 1))
            stats_ro_t = (q_ro.flatten() // head_dim).view(batch + 1, 1, 1, 1).contiguous()
        stats_ro = graph.tensor_like(stats_ro_t, name="stats_ro")
        stats.set_ragged_offset(stats_ro)
        variant_pack[stats_ro] = stats_ro_t
        variant_pack[stats] = stats_storage

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    _select_engine(graph, engine_name(arch="sm120"))
    graph.check_support()
    graph.build_plans()
    workspace = torch.empty(max(1, graph.get_workspace_size()), dtype=torch.uint8, device="cuda")
    graph.execute(variant_pack, workspace)
    torch.cuda.synchronize()

    cu = [0]
    for n in seq_q_lens:
        cu.append(cu[-1] + n)
    if cu[-1] == 0:
        # No query token exists anywhere (t_q == 0): execute must be a
        # complete no-op — the sentinel-filled O and ragged Stats storages
        # come back untouched.
        assert (o_storage == _THD_SENTINEL).all(), "t_q == 0 wrote to O"
        if check_stats:
            assert (stats_storage == _THD_SENTINEL).all(), "t_q == 0 wrote to the ragged Stats"
        return
    packed_o = o_storage[: cu[-1] * h_q * d_v].view(max(cu[-1], 1), h_q, d_v)
    if check_stats and stats_layout == "head_major":
        packed_stats = stats_storage.view(h_q, t_cap)  # (H, head_stride); tokens at [:, cu[i]:cu[i+1]]
    elif check_stats:
        packed_stats = stats_storage[: cu[-1] * h_q].view(max(cu[-1], 1), h_q)  # (T, H)
    else:
        packed_stats = None
    for i, (nq, _nkv) in enumerate(zip(seq_q_lens, seq_kv_lens)):
        if nq == 0:
            continue
        got = packed_o[cu[i] : cu[i + 1]].permute(1, 0, 2).unsqueeze(0).float()
        ref = _ref_sdpa_full(
            q_seqs[i],
            k_seqs[i],
            v_seqs[i],
            scale=scale,
            is_causal=is_causal,
            causal_bottom_right=causal_bottom_right,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            sinks=sinks,
            return_stats=check_stats,
        )
        expected, expected_lse = ref if check_stats else (ref, None)
        torch.testing.assert_close(got, expected, atol=0.1, rtol=5e-2)
        if check_stats:
            if stats_layout == "head_major":
                got_lse = packed_stats[:, cu[i] : cu[i + 1]].unsqueeze(0)  # (H, T_i) -> (1, H, T_i)
            else:
                got_lse = packed_stats[cu[i] : cu[i + 1]].t().unsqueeze(0)  # (T_i, H) -> (1, H, T_i)
            torch.testing.assert_close(got_lse, expected_lse, atol=2e-2, rtol=2e-2)


@pytest.mark.L0
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("is_causal", [False, True], ids=["dense", "causal"])
@torch_fork_set_rng(seed=0)
def test_sdpa_fwd_dsl_sm120_graph_api(dtype: torch.dtype, is_causal: bool):
    """Execute the SM100-overlap workload through the selected SM120 engine."""

    batch, heads, sequence, head_dim = 2, 8, 256, 256
    scale = 1.0 / math.sqrt(head_dim)
    q = _bhsd(batch, heads, sequence, head_dim, dtype)
    k = _bhsd(batch, heads, sequence, head_dim, dtype)
    v = _bhsd(batch, heads, sequence, head_dim, dtype)

    output = _run_dsl_graph(q, k, v, scale=scale, is_causal=is_causal, kv_tile=64)
    expected = _ref_sdpa_full(q, k, v, scale=scale, is_causal=is_causal)
    torch.testing.assert_close(output.float(), expected, atol=0.1, rtol=5e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=1)
def test_dsl_sm120_graph_api_padded_gqa():
    """Execute a padded GQA graph through the same Unified SDPA path as SM100."""

    batch, h_q, h_kv, s_q, s_kv, head_dim = 2, 8, 1, 96, 144, 64
    scale = 1.0 / math.sqrt(head_dim)
    q = _padded_bhsd(
        batch,
        h_q,
        s_q,
        head_dim,
        torch.float16,
        head_padding=8,
        sequence_padding=16,
        batch_padding=24,
    )
    k = _padded_bhsd(
        batch,
        h_kv,
        s_kv,
        head_dim,
        torch.float16,
        head_padding=16,
        sequence_padding=24,
        batch_padding=32,
    )
    v = _padded_bhsd(
        batch,
        h_kv,
        s_kv,
        head_dim,
        torch.float16,
        head_padding=24,
        sequence_padding=32,
        batch_padding=40,
    )
    output = _padded_bhsd(
        batch,
        h_q,
        s_q,
        head_dim,
        torch.float16,
        head_padding=32,
        sequence_padding=40,
        batch_padding=48,
    )
    seq_q_lens = torch.tensor([93, 51], dtype=torch.int32, device="cuda").view(batch, 1, 1, 1)
    seq_kv_lens = torch.tensor([137, 79], dtype=torch.int32, device="cuda").view(batch, 1, 1, 1)

    result = _run_dsl_graph(
        q,
        k,
        v,
        scale=scale,
        o_gpu=output,
        is_causal=True,
        causal_bottom_right=True,
        window_size_left=47,
        seq_q_lens=seq_q_lens,
        seq_kv_lens=seq_kv_lens,
    )
    expected = _ref_sdpa_full(
        q,
        k,
        v,
        scale=scale,
        is_causal=True,
        causal_bottom_right=True,
        window_size_left=47,
        seq_q_lens=seq_q_lens,
        seq_kv_lens=seq_kv_lens,
    )
    torch.testing.assert_close(result.float(), expected, atol=0.1, rtol=5e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=10)
def test_dsl_sm120_long_sequence_multi_wave():
    """Launch far beyond one wave: ceil(4096/128) * B * H = 512 CTAs against
    ~170 SMs, causal so per-CTA work varies across the grid."""

    _run_case(batch=2, h_q=8, h_kv=8, s_q=4096, s_kv=4096, head_dim=128, is_causal=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=11)
def test_dsl_sm120_fully_masked_rows():
    """Rows with no visible keys must write O := 0, never NaN.

    A zero-length KV batch is the one graph-reachable fully-masked shape:
    native graph validation rejects the mask-geometry routes (bottom-right
    causal or SWA with S_q > S_kv) before any engine runs. The kernel side
    still covers the interesting edge — zero KV iterations for the whole
    batch, with the epilogue required to write zeros.
    """

    seq_q_lens = torch.tensor([128, 128], dtype=torch.int32, device="cuda")
    seq_kv_lens = torch.tensor([0, 96], dtype=torch.int32, device="cuda")
    _run_case(batch=2, s_q=128, s_kv=128, head_dim=64, seq_q_lens=seq_q_lens, seq_kv_lens=seq_kv_lens)


@pytest.mark.L0
@pytest.mark.parametrize(
    "mask_kwargs",
    [
        {},
        {"is_causal": True},
        {"is_causal": True, "causal_bottom_right": True},
        {"is_causal": True, "window_size_left": 31},
    ],
    ids=["dense", "causal", "causal_br", "causal_swa"],
)
@torch_fork_set_rng(seed=13)
def test_dsl_sm120_stats(mask_kwargs):
    """generate_stats=True: the Stats output matches the natural-log LSE."""

    _run_case(batch=2, h_q=4, h_kv=2, s_q=256, s_kv=256, head_dim=128, check_stats=True, **mask_kwargs)


@pytest.mark.L0
@torch_fork_set_rng(seed=59)
def test_dsl_sm120_strided_stats():
    """Dense LSE is written directly through a permuted, gapped layout."""

    _require_dsl()
    batch, h_q, h_kv, sequence, head_dim = 2, 4, 2, 128, 128
    scale = 1.0 / math.sqrt(head_dim)
    q = _bhsd(batch, h_q, sequence, head_dim, torch.float16)
    k = _bhsd(batch, h_kv, sequence, head_dim, torch.float16)
    v = _bhsd(batch, h_kv, sequence, head_dim, torch.float16)
    _, contiguous_stats = _run_dsl_graph(q, k, v, scale=scale, is_causal=True, return_stats=True)
    output, strided_stats = _run_dsl_graph(q, k, v, scale=scale, is_causal=True, return_stats=True, stats_layout="strided")
    expected, expected_stats = _ref_sdpa_full(q, k, v, scale=scale, is_causal=True, return_stats=True)
    torch.testing.assert_close(strided_stats, contiguous_stats, atol=0, rtol=0)
    torch.testing.assert_close(strided_stats.squeeze(-1), expected_stats, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(output.float(), expected, atol=0.1, rtol=5e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=14)
def test_dsl_sm120_stats_padded_trim():
    """Rows past seq_len_q[b] and rows with no visible key produce LSE = -inf.

    Batch 0 pairs a short Q length with a zero-length KV segment, so its
    valid rows are -inf via the dead-row path and its tail rows via the
    per-batch trim; batch 1 keeps finite LSE up to its Q length.
    """

    seq_q_lens = torch.tensor([96, 128], dtype=torch.int32, device="cuda")
    seq_kv_lens = torch.tensor([0, 64], dtype=torch.int32, device="cuda")
    _run_case(
        batch=2,
        s_q=128,
        s_kv=128,
        head_dim=64,
        seq_q_lens=seq_q_lens,
        seq_kv_lens=seq_kv_lens,
        check_stats=True,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=18)
def test_dsl_sm120_sink():
    """Causal + attention sink: per-Q-head logit in the softmax denominator."""

    _run_case(batch=2, h_q=8, h_kv=2, s_q=256, s_kv=256, head_dim=64, is_causal=True, with_sink=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=19)
def test_dsl_sm120_sink_stats():
    """The sink enters the LSE: Stats must match the sink-extended logsumexp."""

    _run_case(batch=2, h_q=4, h_kv=4, s_q=256, s_kv=256, head_dim=128, is_causal=True, with_sink=True, check_stats=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=20)
def test_dsl_sm120_sink_dead_rows():
    """With a sink, rows with no visible key keep a finite LSE (the sink
    alone) and O := 0; rows past seq_len_q[b] still trim to -inf."""

    seq_q_lens = torch.tensor([96, 128], dtype=torch.int32, device="cuda")
    seq_kv_lens = torch.tensor([0, 64], dtype=torch.int32, device="cuda")
    _run_case(
        batch=2,
        s_q=128,
        s_kv=128,
        head_dim=64,
        seq_q_lens=seq_q_lens,
        seq_kv_lens=seq_kv_lens,
        with_sink=True,
        check_stats=True,
    )


@pytest.mark.L1
@torch_fork_set_rng(seed=21)
def test_dsl_sm120_sink_variants():
    """Sink across bf16 + MQA, 64-wide tiles, and the direct wrapper."""

    _run_case(h_q=8, h_kv=1, s_q=256, s_kv=256, head_dim=128, dtype=torch.bfloat16, is_causal=True, with_sink=True)
    _run_case(q_tile=64, kv_tile=64, s_q=128, s_kv=128, head_dim=64, with_sink=True, check_stats=True)

    _require_dsl()
    from cudnn.sdpa.fwd import sdpa_fwd_wrapper_dsl_sm120

    q = _bhsd(2, 4, 128, 128, torch.float16)
    k = _bhsd(2, 4, 128, 128, torch.float16)
    v = _bhsd(2, 4, 128, 128, torch.float16)
    sinks = torch.randn(1, 4, 1, 1, dtype=torch.float32, device="cuda")
    scale = 1.0 / math.sqrt(128)
    o_tensor, lse_tensor = sdpa_fwd_wrapper_dsl_sm120(q, k, v, is_causal=True, sinks=sinks)
    expected_o, expected_lse = _ref_sdpa_full(q, k, v, scale=scale, is_causal=True, sinks=sinks, return_stats=True)
    torch.testing.assert_close(o_tensor.float(), expected_o, atol=0.1, rtol=5e-2)
    torch.testing.assert_close(lse_tensor, expected_lse, atol=2e-2, rtol=2e-2)


@pytest.mark.L1
@torch_fork_set_rng(seed=15)
def test_dsl_sm120_stats_variants():
    """LSE across bf16 + GQA, 64-wide tiles, large D, and multi-wave grids."""

    _run_case(h_q=8, h_kv=1, s_q=256, s_kv=256, head_dim=128, dtype=torch.bfloat16, is_causal=True, check_stats=True)
    _run_case(q_tile=64, kv_tile=64, s_q=128, s_kv=128, head_dim=64, check_stats=True)
    _run_case(head_dim=256, s_q=128, s_kv=128, check_stats=True)  # auto kv_tile=64
    _run_case(batch=2, h_q=8, s_q=4096, s_kv=4096, head_dim=128, is_causal=True, check_stats=True)


@pytest.mark.L1
@torch_fork_set_rng(seed=16)
def test_dsl_sm120_wrapper_lse():
    """The direct wrapper returns the (B, H, Sq) natural-log LSE."""

    _require_dsl()
    from cudnn.sdpa.fwd import sdpa_fwd_wrapper_dsl_sm120

    q = _bhsd(2, 4, 128, 128, torch.float16)
    k = _bhsd(2, 4, 128, 128, torch.float16)
    v = _bhsd(2, 4, 128, 128, torch.float16)
    scale = 1.0 / math.sqrt(128)
    o_tensor, lse_tensor = sdpa_fwd_wrapper_dsl_sm120(q, k, v, is_causal=True)
    expected_o, expected_lse = _ref_sdpa_full(q, k, v, scale=scale, is_causal=True, return_stats=True)
    torch.testing.assert_close(o_tensor.float(), expected_o, atol=0.1, rtol=5e-2)
    torch.testing.assert_close(lse_tensor, expected_lse, atol=2e-2, rtol=2e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=17)
def test_dsl_sm120_execute_contract_mismatches():
    """execute() rejects lse/sinks that contradict the compiled specialization.

    The kernel specializes on sink and LSE presence at compile time, so a
    mismatch at execute is a hard error: substituting a zeros sink would
    silently change the softmax denominator, and a provided-but-uncompiled
    LSE would be silently left unwritten.
    """

    _require_dsl()
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120

    q = _bhsd(1, 4, 128, 128, torch.float16)
    k = _bhsd(1, 4, 128, 128, torch.float16)
    v = _bhsd(1, 4, 128, 128, torch.float16)
    o = torch.empty_like(q)
    lse = torch.empty(1, 4, 128, dtype=torch.float32, device="cuda")
    sinks = torch.randn(1, 4, 1, 1, dtype=torch.float32, device="cuda")

    # Compiled WITH sink + LSE: both must be provided at execute.
    api = SdpaFwdDslSm120(sample_q=q, sample_k=k, sample_v=v, sample_o=o, sample_lse=lse, has_sink=True)
    assert api.check_support()
    api.compile()
    with pytest.raises(ValueError, match="sinks is required"):
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, lse_tensor=lse)
    with pytest.raises(ValueError, match="lse_tensor is required"):
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, sinks=sinks)
    # Sinks are consumed as fp32 directly — no implicit cast (which would
    # allocate and launch a kernel on the execute hot path).
    with pytest.raises(ValueError, match="sinks must be float32"):
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, lse_tensor=lse, sinks=sinks.to(torch.bfloat16))

    # Compiled WITHOUT sink or LSE: providing either is rejected, and the
    # matching call runs with no LSE buffer anywhere (store compiled out).
    api = SdpaFwdDslSm120(sample_q=q, sample_k=k, sample_v=v, sample_o=o)
    assert api.check_support()
    api.compile()
    with pytest.raises(ValueError, match="without sink support"):
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, sinks=sinks)
    with pytest.raises(ValueError, match="without an LSE output"):
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, lse_tensor=lse)
    # Same contract for per-batch lengths: a specialization compiled without
    # them must not silently ignore a provided tensor (nor, the other way,
    # substitute a zeros dummy that would mask every row).
    seq_kv = torch.full((1,), 128, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="without per-batch KV lengths"):
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, seq_kv_lens=seq_kv)
    with pytest.raises(ValueError, match="without per-batch Q lengths"):
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, seq_q_lens=seq_kv)
    api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o)
    torch.cuda.synchronize()
    expected = _ref_sdpa_full(q, k, v, scale=1.0 / math.sqrt(128))
    torch.testing.assert_close(o.float(), expected, atol=0.1, rtol=5e-2)

    # THD LSE must be declared packed: token-major [t, h] or head-major
    # [h, t]. A dense-contiguous declaration (stride (S*H, S, 1)) is valid
    # head-major (head_stride S); a padded sequence stride matches NEITHER layout
    # and is rejected up front instead of being silently mis-addressed.
    lse_padded = torch.empty(4 * 128 * 2, dtype=torch.float32, device="cuda").as_strided((1, 4, 128), (4 * 128 * 2, 128 * 2, 2))
    with pytest.raises(ValueError, match="token-major"):
        SdpaFwdDslSm120(sample_q=q, sample_k=k, sample_v=v, sample_o=o, sample_lse=lse_padded, thd=True).check_support()
    api = SdpaFwdDslSm120(sample_q=q, sample_k=k, sample_v=v, sample_o=o, sample_lse=lse, thd=True)
    assert api.check_support() and api.thd_stats_head_major and api.thd_stats_head_stride == 128

    # THD execute keeps the same presence contract as dense: the raise fires
    # before any packing or launch, so plain dense buffers suffice here.
    lse_thd = torch.empty(1 * 4 * 128, dtype=torch.float32, device="cuda").as_strided((1, 4, 128), (128 * 4, 1, 4))
    api = SdpaFwdDslSm120(sample_q=q, sample_k=k, sample_v=v, sample_o=o, sample_lse=lse_thd, thd=True)
    assert api.check_support()
    api.compile()
    with pytest.raises(ValueError, match="lse_tensor is required"):
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, seq_q_lens=seq_kv, seq_kv_lens=seq_kv)
    api = SdpaFwdDslSm120(sample_q=q, sample_k=k, sample_v=v, sample_o=o, thd=True)
    assert api.check_support()
    api.compile()
    with pytest.raises(ValueError, match="without an LSE output"):
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, seq_q_lens=seq_kv, seq_kv_lens=seq_kv, lse_tensor=lse_thd)

    # Right-band contract (band model): window_size_right is the causal
    # diagonal's right bound (0 = plain causal) and therefore requires
    # is_causal; negative bounds are rejected.
    with pytest.raises(ValueError, match="requires is_causal"):
        SdpaFwdDslSm120(sample_q=q, sample_k=k, sample_v=v, sample_o=o, window_size_right=8).check_support()
    with pytest.raises(ValueError, match="window_size_right must be >= 0"):
        SdpaFwdDslSm120(sample_q=q, sample_k=k, sample_v=v, sample_o=o, is_causal=True, window_size_right=-1).check_support()


@pytest.mark.L0
@torch_fork_set_rng(seed=22)
def test_dsl_sm120_thd():
    """THD self-attention: packed ragged batch vs per-sequence references."""

    _run_thd_case(seq_q_lens=[200, 150], seq_kv_lens=[200, 150], is_causal=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=23)
def test_dsl_sm120_thd_cross():
    """THD cross-attention: unequal packed Q and KV token totals."""

    _run_thd_case(seq_q_lens=[200, 150], seq_kv_lens=[180, 120], is_causal=True, head_dim=128)


@pytest.mark.L0
@torch_fork_set_rng(seed=24)
def test_dsl_sm120_thd_bottom_right():
    """THD + bottom-right causal: per-sequence diagonals"""

    _run_thd_case(seq_q_lens=[100, 60], seq_kv_lens=[180, 120], is_causal=True, causal_bottom_right=True)


@pytest.mark.L0
@pytest.mark.parametrize("stats_layout", ["token_major", "head_major"])
@torch_fork_set_rng(seed=30)
def test_dsl_sm120_thd_stats(stats_layout: str):
    """THD + generate_stats: the ragged Stats output is written in the
    caller's declared layout — token-major [t, h] or head-major [h, t]."""

    _run_thd_case(seq_q_lens=[200, 150], seq_kv_lens=[200, 150], is_causal=True, check_stats=True, stats_layout=stats_layout)


@pytest.mark.L1
@torch_fork_set_rng(seed=32)
def test_dsl_sm120_thd_swa_stats():
    """THD + causal left sliding window + ragged Stats: the window trims the
    per-sequence LSE denominator, and a distinct compiled specialization
    (window_size_left is a template parameter) carries the token-major store."""

    _run_thd_case(seq_q_lens=[150, 90], seq_kv_lens=[150, 90], is_causal=True, window_size_left=32, check_stats=True)


@pytest.mark.L1
@pytest.mark.parametrize("stats_layout", ["token_major", "head_major"])
@torch_fork_set_rng(seed=25)
def test_dsl_sm120_thd_gqa_sink(stats_layout: str):
    """THD + GQA + attention sink through the packed epilogue fold, with the
    sink entering the ragged Stats (both declared layouts)."""

    _run_thd_case(seq_q_lens=[130, 70], seq_kv_lens=[130, 70], h_q=8, h_kv=2, is_causal=True, with_sink=True, check_stats=True, stats_layout=stats_layout)


@pytest.mark.L1
@torch_fork_set_rng(seed=26)
def test_dsl_sm120_thd_zero_length_sequence():
    """A zero-length sequence contributes no tokens and must not perturb its
    packed neighbors (O and ragged Stats). The last sequence has Q tokens but
    ZERO keys inside a live launch: its rows must come back O := 0 with
    LSE := -inf through the kernel's row_sum <= 0 guard, not stale memory."""

    _run_thd_case(seq_q_lens=[128, 0, 64], seq_kv_lens=[100, 0, 0], is_causal=True, check_stats=True)


@pytest.mark.L1
@pytest.mark.parametrize("stats_layout", ["token_major", "head_major"])
@pytest.mark.parametrize("with_sink", [False, True], ids=["no_sink", "sink"])
@torch_fork_set_rng(seed=31)
def test_dsl_sm120_thd_all_kv_zero_stats(with_sink: bool, stats_layout: str):
    """Every KV length zero: the launch goes through the KERNEL's dead-row
    path (O := 0, LSE := -inf, or the sink value alone — the sink column
    keeps the softmax denominator alive) with the packed KV extent clamped
    to one never-dereferenced token (a zero-token K/V view cannot back a
    CuTe layout) — no adapter-side fills, in either declared layout."""

    _run_thd_case(seq_q_lens=[64, 32], seq_kv_lens=[0, 0], with_sink=with_sink, check_stats=True, stats_layout=stats_layout)


@pytest.mark.L1
@pytest.mark.parametrize("stats_layout", ["token_major", "head_major"])
@torch_fork_set_rng(seed=34)
def test_dsl_sm120_thd_all_q_zero_stats(stats_layout: str):
    """Every Q length zero (t_q == 0): no query token exists anywhere, so the
    packed O/Stats have zero rows and execute must be a complete NO-OP — the
    sentinel-filled buffers come back untouched, with live KV and with the
    fully-degenerate all-zero KV as well."""

    _run_thd_case(seq_q_lens=[0, 0], seq_kv_lens=[50, 30], check_stats=True, stats_layout=stats_layout)
    _run_thd_case(seq_q_lens=[0, 0], seq_kv_lens=[0, 0], check_stats=True, stats_layout=stats_layout)


@pytest.mark.L0
@pytest.mark.parametrize("stats_layout", ["token_major", "head_major"])
@torch_fork_set_rng(seed=35)
def test_dsl_sm120_thd_cu_seq_len_stats(stats_layout: str):
    """THD with the cu_seq_len_q/kv length form ((B+1,) prefix sums, cuDNN
    9.24+ — the form TE/PyT/vLLM natively hold): the lowering derives the
    per-batch lengths host-side from the same inherent tolist round-trip, so
    results are identical to the seq_len form, ragged Stats included."""

    _run_thd_case(seq_q_lens=[200, 150], seq_kv_lens=[200, 150], is_causal=True, check_stats=True, stats_layout=stats_layout, cu_lens=True)


@pytest.mark.L1
@torch_fork_set_rng(seed=36)
def test_dsl_sm120_thd_cu_seq_len_zero_lens():
    """cu_seq_len form with degenerate lengths: a zero-length sequence
    (repeated prefix value), an all-zero KV side (kernel dead-row path), and
    the all-zero Q no-op keep the same semantics as the seq_len form."""

    _run_thd_case(seq_q_lens=[128, 0, 64], seq_kv_lens=[100, 0, 0], is_causal=True, check_stats=True, cu_lens=True)
    _run_thd_case(seq_q_lens=[64, 32], seq_kv_lens=[0, 0], check_stats=True, cu_lens=True)
    _run_thd_case(seq_q_lens=[0, 0], seq_kv_lens=[50, 30], check_stats=True, cu_lens=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=37)
def test_dsl_sm120_thd_compile_key_plan_time_only():
    """Issue #552: the THD compile key carries NO packed totals.

    ``compile()`` builds the one artifact at plan time (the token extents
    compile dynamic, ``max_sq`` is a runtime launch argument), and executes
    with DIFFERENT packed totals re-bind it — zero ``cute.compile`` calls on
    the execute path. Keying the compile on the totals degenerated into a
    fresh multi-second compile per step under continuous batching (the
    totals change every step), and correctness is checked per total to prove
    one artifact serves them all.
    """
    _require_dsl()
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120

    b, h, s, d = 2, 4, 256, 128
    dtype = torch.float16
    scale = 1.0 / math.sqrt(d)
    q, k, v = (_bhsd(b, h, s, d, dtype) for _ in range(3))
    o = torch.zeros_like(q)
    api = SdpaFwdDslSm120(sample_q=q, sample_k=k, sample_v=v, sample_o=o, thd=True)
    assert api.check_support()
    api.compile()
    # Plan-time compile: no deferred sentinel, the artifact already exists.
    assert api._compiled_kernel != "thd-deferred"
    info_plan = api._k_mod.compile.cache_info()

    def _run_and_check(seq_lens):
        lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, seq_q_lens=lens, seq_kv_lens=lens)
        torch.cuda.synchronize()
        base_q = q.transpose(1, 2).reshape(b * s, h, d)
        base_k = k.transpose(1, 2).reshape(b * s, h, d)
        base_v = v.transpose(1, 2).reshape(b * s, h, d)
        base_o = o.transpose(1, 2).reshape(b * s, h, d)
        off = 0
        for length in seq_lens:
            qs = base_q[off : off + length].float()
            ks = base_k[off : off + length].float()
            vs = base_v[off : off + length].float()
            scores = torch.einsum("lhd,mhd->hlm", qs, ks) * scale
            ref = torch.einsum("hlm,mhd->lhd", torch.softmax(scores, dim=-1), vs)
            torch.testing.assert_close(base_o[off : off + length].float(), ref, atol=5e-2, rtol=3e-2)
            off += length

    _run_and_check([200, 150])
    _run_and_check([64, 33])
    info_exec = api._k_mod.compile.cache_info()
    assert info_exec.misses == info_plan.misses, "a THD execute minted a new kernel compile (runtime data leaked into the compile key)"
    assert info_exec.hits >= info_plan.hits + 2


@pytest.mark.L0
@torch_fork_set_rng(seed=45)
def test_dsl_sm120_thd_lens_never_reach_host():
    """Issue #552 (D2H removal): the length tensors are consumed ONLY on
    device — the setup kernel builds the metadata, the ragged views bind
    buffer capacities, and the grid is the plan-time declared-S_q envelope
    (tiles past a sequence's real length drain without loads or stores).
    The old host round-trip helper (_thd_host_lens) is GONE from the
    adapter entirely, while full numerics run in both length forms."""
    _require_dsl()
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDsl, SdpaFwdDslSm120

    assert not hasattr(SdpaFwdDsl, "_thd_host_lens") and not hasattr(SdpaFwdDslSm120, "_thd_host_lens")
    _run_thd_case(seq_q_lens=[200, 150], seq_kv_lens=[180, 120], is_causal=True, check_stats=True, stats_layout="token_major")
    _run_thd_case(seq_q_lens=[200, 150], seq_kv_lens=[180, 120], is_causal=True, check_stats=True, stats_layout="head_major", cu_lens=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=46)
def test_dsl_sm120_thd_execute_never_syncs():
    """Issue #552 endgame (SM120): the THD execute performs NO synchronizing
    CUDA call — no length D2H, no pageable H2D, no device/stream sync —
    pinned by torch's sync debug mode ("error"), which raises on any;
    results are bitwise identical to an unguarded execute."""
    _require_dsl()
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120

    b, h, s, d = 2, 4, 256, 128
    dtype = torch.float16
    q, k, v = (_bhsd(b, h, s, d, dtype) for _ in range(3))
    o = torch.zeros_like(q)
    api = SdpaFwdDslSm120(sample_q=q, sample_k=k, sample_v=v, sample_o=o, thd=True)
    assert api.check_support()
    api.compile()
    lens = torch.tensor([200, 150], dtype=torch.int32, device="cuda")

    # Warm-up outside the guarded region: allocator pools and lazy launcher
    # state populate here, so the guarded execute reuses cached blocks.
    api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, seq_q_lens=lens, seq_kv_lens=lens)
    torch.cuda.synchronize()
    o_ref = o.clone()
    o.zero_()
    prev_sync_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode(2)
    try:
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, seq_q_lens=lens, seq_kv_lens=lens)
    finally:
        torch.cuda.set_sync_debug_mode(prev_sync_mode)
    torch.cuda.synchronize()
    assert torch.equal(o, o_ref)


@pytest.mark.L0
@torch_fork_set_rng(seed=47)
def test_dsl_sm120_thd_execute_cuda_graph_capture():
    """Issue #552 endgame (SM120): THD execute is CUDA-GRAPH CAPTURABLE — no
    D2H, no pageable H2D, plan-time envelope grid. Capture once, then
    replay with DIFFERENT lengths written into the same device tensors: the
    replay must honor them (per-sequence lengths are read on device by the
    setup and main kernels), proving no host value was baked into the
    graph."""
    _require_dsl()
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120

    b, h, s, d = 2, 4, 256, 128
    dtype = torch.float16
    scale = 1.0 / math.sqrt(d)
    q, k, v = (_bhsd(b, h, s, d, dtype) for _ in range(3))
    o = torch.zeros_like(q)
    api = SdpaFwdDslSm120(sample_q=q, sample_k=k, sample_v=v, sample_o=o, thd=True)
    assert api.check_support()
    api.compile()
    lens = torch.tensor([200, 150], dtype=torch.int32, device="cuda")

    def _check(seq_lens):
        base_q = q.transpose(1, 2).reshape(b * s, h, d)
        base_k = k.transpose(1, 2).reshape(b * s, h, d)
        base_v = v.transpose(1, 2).reshape(b * s, h, d)
        base_o = o.transpose(1, 2).reshape(b * s, h, d)
        off = 0
        for length in seq_lens:
            qs = base_q[off : off + length].float()
            ks = base_k[off : off + length].float()
            vs = base_v[off : off + length].float()
            scores = torch.einsum("lhd,mhd->hlm", qs, ks) * scale
            ref = torch.einsum("hlm,mhd->lhd", torch.softmax(scores, dim=-1), vs)
            torch.testing.assert_close(base_o[off : off + length].float(), ref, atol=5e-2, rtol=3e-2)
            off += length

    api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, seq_q_lens=lens, seq_kv_lens=lens)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, seq_q_lens=lens, seq_kv_lens=lens)
    # Clobber O before each replay: the warm-up (and nothing else) has already
    # produced the [200, 150] answer, so without this the first assertion
    # would be satisfied by stale warm-up output even if replay did nothing.
    o.zero_()
    graph.replay()
    torch.cuda.synchronize()
    _check([200, 150])
    # New lengths into the SAME device tensor — replay must honor them.
    lens.copy_(torch.tensor([64, 33], dtype=torch.int32, device="cuda"))
    o.zero_()
    graph.replay()
    torch.cuda.synchronize()
    _check([64, 33])


@pytest.mark.L1
@torch_fork_set_rng(seed=48)
def test_dsl_sm120_thd_cu_nonzero_base_normalized():
    """The device-side metadata build NORMALIZES cu prefix sums (subtracts
    element 0): the packed buffers are addressed from token 0, so a cu
    tensor sliced from a larger prefix (cu[0] != 0) means the same lengths
    and must produce bitwise-identical results."""
    _require_dsl()
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120

    b, h, s, d = 2, 4, 256, 128
    dtype = torch.float16
    q, k, v = (_bhsd(b, h, s, d, dtype) for _ in range(3))
    o = torch.zeros_like(q)
    api = SdpaFwdDslSm120(sample_q=q, sample_k=k, sample_v=v, sample_o=o, thd=True, cu_seq_q_lens=True, cu_seq_kv_lens=True)
    assert api.check_support()
    api.compile()

    def _run(base_q, base_kv):
        # Distinct Q/KV prefix tensors with DIFFERENT lengths and bases: a
        # normalization that subtracts one side's base from the other (or
        # shares one tensor for both) cannot pass this by accident.
        cu_q = torch.tensor([base_q, base_q + 200, base_q + 350], dtype=torch.int32, device="cuda")
        cu_kv = torch.tensor([base_kv, base_kv + 180, base_kv + 310], dtype=torch.int32, device="cuda")
        o.zero_()
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, seq_q_lens=cu_q, seq_kv_lens=cu_kv)
        torch.cuda.synchronize()
        return o.clone()

    assert torch.equal(_run(0, 0), _run(1000, 7000))


@pytest.mark.L0
@torch_fork_set_rng(seed=9)
def test_dsl_sm120_dense_flex_bhsd_contiguous():
    """BHSD-contiguous Q/K/V/O (dense_flex): served via compact-BSHD normalization."""

    batch, h_q, h_kv, s, head_dim = 2, 8, 2, 256, 64
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(batch, h_q, s, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, h_kv, s, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch, h_kv, s, head_dim, device="cuda", dtype=torch.float16)
    output = torch.empty_like(q)

    result = _run_dsl_graph(q, k, v, scale=scale, o_gpu=output, is_causal=True)
    expected = _ref_sdpa_full(q, k, v, scale=scale, is_causal=True)
    torch.testing.assert_close(result.float(), expected, atol=0.1, rtol=5e-2)


@pytest.mark.L0
@pytest.mark.parametrize(
    ("head_dim", "kv_tile"),
    [
        (16, None),
        (112, None),
        (160, None),
        (192, None),
        (208, None),
        (224, None),
        (240, None),
        (256, None),
        (256, 64),
    ],
)
@torch_fork_set_rng(seed=2)
def test_dsl_sm120_representative_head_dimensions(head_dim: int, kv_tile: int | None):
    _run_case(head_dim=head_dim, kv_tile=kv_tile)


@pytest.mark.L0
@pytest.mark.parametrize(
    ("head_dim", "head_dim_v"),
    [(192, 128), (128, 192), (96, 32), (128, 48)],
    ids=["mla192x128", "v_wider", "small_mixed", "cross_swizzle"],
)
@torch_fork_set_rng(seed=23)
def test_dsl_sm120_mixed_head_dims(head_dim: int, head_dim_v: int):
    """D_QK != D_V: independent QK^T contraction and P@V output widths.

    (192, 128) is the MLA-style shape; the K and V SMEM tiles and TMA swizzle
    configurations differ, so a reversed pair and a small mixed pair guard the
    per-tensor address math in both directions, and (128, 48) pins K and V on
    DIFFERENT swizzle modes (s128b vs s32b)."""
    _run_case(batch=2, h_q=4, h_kv=4, s_q=256, s_kv=384, head_dim=head_dim, head_dim_v=head_dim_v)


@pytest.mark.L0
@torch_fork_set_rng(seed=24)
def test_dsl_sm120_mixed_head_dims_causal_gqa_stats():
    """MLA shape under bottom-right causal + GQA, with the LSE checked (the
    stats path depends only on D_QK; this pins that it survives the split)."""
    _run_case(
        batch=2,
        h_q=8,
        h_kv=2,
        s_q=192,
        s_kv=320,
        head_dim=192,
        head_dim_v=128,
        is_causal=True,
        causal_bottom_right=True,
        check_stats=True,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=25)
def test_dsl_sm120_thd_mixed_head_dims():
    """THD packed views carry per-tensor head dims (Q/K at 192, V/O at 128),
    with the ragged Stats checked under the bottom-right diagonal."""
    _run_thd_case(
        seq_q_lens=[33, 128, 7],
        seq_kv_lens=[65, 128, 190],
        h_q=4,
        h_kv=2,
        head_dim=192,
        head_dim_v=128,
        is_causal=True,
        causal_bottom_right=True,
        check_stats=True,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=26)
def test_dsl_sm120_mixed_head_dims_masks():
    """MLA shape under the remaining mask family: top-left causal, and causal
    with a left sliding window (the mask math lives on the QK^T side, so this
    pins that it stays anchored to D_QK after the split)."""
    _run_case(batch=2, h_q=4, h_kv=4, s_q=256, s_kv=256, head_dim=192, head_dim_v=128, is_causal=True)
    _run_case(batch=2, h_q=4, h_kv=4, s_q=256, s_kv=256, head_dim=192, head_dim_v=128, is_causal=True, window_size_left=96)


@pytest.mark.L0
@torch_fork_set_rng(seed=27)
def test_dsl_sm120_mixed_head_dims_padded_stats():
    """MLA shape + per-batch Q/KV lengths with the LSE checked: rows past
    seq_len_q[b] zero-fill O through the epilogue's D_V-guarded store and trim
    the LSE to -inf, so the changed store guard and the stats trim are
    exercised together."""
    seq_q_lens = torch.tensor([230, 120], dtype=torch.int32, device="cuda")
    seq_kv_lens = torch.tensor([180, 240], dtype=torch.int32, device="cuda")
    _run_case(
        batch=2,
        h_q=4,
        h_kv=4,
        s_q=256,
        s_kv=256,
        head_dim=192,
        head_dim_v=128,
        seq_q_lens=seq_q_lens,
        seq_kv_lens=seq_kv_lens,
        check_stats=True,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=28)
def test_dsl_sm120_mixed_head_dims_sink_bf16():
    """MLA shape + attention sink with the LSE checked (the sink logit rides
    the softmax denominator, which depends only on D_QK), on bf16 to cover the
    second dtype."""
    _run_case(
        batch=2,
        h_q=8,
        h_kv=2,
        s_q=256,
        s_kv=256,
        head_dim=192,
        head_dim_v=128,
        dtype=torch.bfloat16,
        is_causal=True,
        with_sink=True,
        check_stats=True,
    )


@pytest.mark.L0
@pytest.mark.parametrize("stats_layout", ["token_major", "head_major"])
@torch_fork_set_rng(seed=29)
def test_dsl_sm120_thd_mixed_head_dims_sink(stats_layout: str):
    """THD + mixed head dims + sink + ragged Stats: the packed per-tensor D
    views and the sink denominator together, without a mask (the THD
    mixed-dim causal case is covered above). The Stats ragged offsets derive
    from Q's (d_qk-based) offsets, so both declared LSE layouts are pinned
    under d_qk != d_v."""
    _run_thd_case(
        seq_q_lens=[130, 70, 9],
        seq_kv_lens=[130, 70, 190],
        h_q=8,
        h_kv=2,
        head_dim=192,
        head_dim_v=128,
        with_sink=True,
        check_stats=True,
        stats_layout=stats_layout,
    )


@pytest.mark.L0
@pytest.mark.parametrize(("h_q", "h_kv"), [(4, 4), (8, 2), (8, 1)], ids=["mha", "gqa", "mqa"])
@torch_fork_set_rng(seed=3)
def test_dsl_sm120_grouped_query_attention(h_q: int, h_kv: int):
    _run_case(batch=2, h_q=h_q, h_kv=h_kv, s_q=256, s_kv=256, head_dim=64)


# --- PackGQA: q_tile/G tokens x G query heads per tile -------
@pytest.mark.L0
@pytest.mark.parametrize(
    "h_q,h_kv",
    [(8, 4), (8, 2), (8, 1), (16, 1)],
    ids=["g2", "g4", "g8_mqa", "g16_mqa"],
)
@torch_fork_set_rng(seed=0)
def test_dsl_sm120_pack_gqa_ratios(h_q: int, h_kv: int):
    """Packed plans across GQA ratios (incl. MQA)."""
    _run_case(batch=2, h_q=h_q, h_kv=h_kv, s_q=40, s_kv=256, head_dim=128, is_causal=True, pack_gqa=True, check_stats=True)


@pytest.mark.L0
@pytest.mark.parametrize("s_q", [4, 16, 25], ids=["subspan", "exact_span", "tail"])
@torch_fork_set_rng(seed=1)
def test_dsl_sm120_pack_gqa_tiles(s_q: int):
    """Packed tile-geometry edges at G=8, q_tile=128 (token span 16/tile)."""
    _run_case(batch=1, h_q=64, h_kv=8, s_q=s_q, s_kv=256, head_dim=128, is_causal=True, q_tile=128, kv_tile=128, pack_gqa=True, check_stats=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=2)
def test_dsl_sm120_pack_gqa_tile64():
    """Packed at q_tile=64 (G must divide the smaller tile: 8/2 -> G=4)."""
    _run_case(batch=2, h_q=8, h_kv=2, s_q=24, s_kv=192, head_dim=64, is_causal=True, q_tile=64, kv_tile=64, pack_gqa=True, check_stats=True)


@pytest.mark.L0
@pytest.mark.parametrize(
    "mask",
    ["none", "causal", "causal_br", "swa", "band_right", "padded_qtrim", "sink_swa"],
)
@torch_fork_set_rng(seed=3)
def test_dsl_sm120_pack_gqa_features(mask: str):
    """Packed plans x the dense mask/sink/trim envelope, stats checked."""
    kw: dict = dict(batch=2, h_q=8, h_kv=2, s_q=40, s_kv=256, head_dim=128, pack_gqa=True, check_stats=True)
    if mask == "causal":
        kw.update(is_causal=True)
    elif mask == "causal_br":
        kw.update(is_causal=True, causal_bottom_right=True, window_size_right=0)
    elif mask == "swa":
        kw.update(is_causal=True, window_size_left=16)
    elif mask == "band_right":
        kw.update(window_size_right=8)
    elif mask == "padded_qtrim":
        kw.update(
            s_q=128,
            seq_q_lens=torch.tensor([37, 90], dtype=torch.int32, device="cuda"),
            seq_kv_lens=torch.tensor([180, 240], dtype=torch.int32, device="cuda"),
        )
    elif mask == "sink_swa":
        kw.update(is_causal=True, window_size_left=16, with_sink=True)
    _run_case(**kw)


@pytest.mark.L1
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("s_q, h_kv", [(1, 8), (127, 8), (127, 1)], ids=["s1-g8", "s127-g8", "s127-mqa64"])
@torch_fork_set_rng(seed=4)
def test_dsl_sm120_pack_gqa_deep(dtype: torch.dtype, s_q: int, h_kv: int):
    """Packed multi-tile / odd-length row spaces, bf16."""
    _run_case(batch=1, h_q=64, h_kv=h_kv, s_q=s_q, s_kv=2048, head_dim=128, dtype=dtype, is_causal=True, pack_gqa=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=4)
def test_dsl_sm120_causal_swa():
    """Causal sliding-window attention: keep q-W <= kv <= q."""

    _run_case(
        h_q=8,
        h_kv=2,
        s_q=384,
        s_kv=384,
        head_dim=64,
        is_causal=True,
        window_size_left=200,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=5)
def test_dsl_sm120_causal_bottom_right():
    """Bottom-right causal attention with S_q != S_kv."""

    _run_case(
        h_q=8,
        h_kv=2,
        s_q=128,
        s_kv=256,
        head_dim=64,
        is_causal=True,
        causal_bottom_right=True,
    )


@pytest.mark.L0
@pytest.mark.parametrize(
    ("head_dim", "head_dim_v"),
    [(72, 72), (104, 72), (8, 8), (200, 136)],
    ids=["d72", "mixed104x72", "d8_min", "d200x136"],
)
@torch_fork_set_rng(seed=34)
def test_dsl_sm120_head_dim_envelope(head_dim: int, head_dim_v: int):
    """Head dims that are multiples of 8 but not 16: the kernel compiles at
    tiles rounded up to 16 and the per-chunk TMA copies zero-fill columns
    past the actual extents — S, softmax, and P@V are bit-identical to the
    unpadded problem, and O stores clip at the actual D_V."""
    _run_case(batch=2, h_q=4, h_kv=4, s_q=192, s_kv=256, head_dim=head_dim, head_dim_v=head_dim_v, is_causal=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=35)
def test_dsl_sm120_head_dim_envelope_features():
    """The envelope composed with the feature family: padded + stats (LSE
    trim with pad columns), sink, and a THD ragged batch — plus a d=248 case
    that forces the auto kv_tile=64 pick (per-chunk XOR phase at the smaller
    tile)."""
    seq_q_lens = torch.tensor([150, 96], dtype=torch.int32, device="cuda")
    seq_kv_lens = torch.tensor([200, 128], dtype=torch.int32, device="cuda")
    _run_case(
        batch=2,
        h_q=4,
        h_kv=2,
        s_q=192,
        s_kv=256,
        head_dim=104,
        head_dim_v=72,
        seq_q_lens=seq_q_lens,
        seq_kv_lens=seq_kv_lens,
        check_stats=True,
    )
    _run_case(batch=2, h_q=4, h_kv=4, s_q=128, s_kv=128, head_dim=88, dtype=torch.bfloat16, with_sink=True, check_stats=True)
    # Envelope pad columns and a ragged S_kv tail in the SAME rightmost
    # tile: both zero-fill mechanisms at once, with the LSE checked.
    _run_case(batch=2, h_q=4, h_kv=4, s_q=192, s_kv=300, head_dim=104, head_dim_v=72, check_stats=True)
    _run_case(head_dim=248, head_dim_v=248, s_q=128, s_kv=128)  # auto kv_tile=64
    _run_thd_case(
        seq_q_lens=[130, 70],
        seq_kv_lens=[130, 70],
        h_q=4,
        h_kv=2,
        head_dim=104,
        head_dim_v=72,
        is_causal=True,
        check_stats=True,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=36)
def test_dsl_sm120_right_band():
    """diagonal_band_right_bound > 0: the causal machinery with the diagonal
    widened right by a compile-time constant R (keep j <= diag + R,
    inclusive). Cases: TOP_LEFT band; BOTTOM_RIGHT band (a band graph is
    NON-causal in cuDNN's vocabulary — no causal flag involved); a full band
    (left + right bounds); R + stats; R across a ragged S_kv tail."""

    _run_case(batch=2, h_q=4, h_kv=4, s_q=256, s_kv=256, window_size_right=32)
    _run_case(batch=2, h_q=4, h_kv=4, s_q=192, s_kv=320, causal_bottom_right=True, window_size_right=48)
    _run_case(batch=2, h_q=4, h_kv=4, s_q=256, s_kv=256, window_size_left=64, window_size_right=32)
    _run_case(batch=2, h_q=8, h_kv=2, s_q=256, s_kv=256, window_size_right=100, check_stats=True)
    _run_case(batch=2, h_q=4, h_kv=4, s_q=256, s_kv=300, window_size_right=32)
    # BR band + SWA + per-batch lengths: the runtime per-batch diagonal
    # offset, the R-shifted right edge, and the UNSHIFTED left anchor in
    # one launch.
    seq_q_lens = torch.tensor([230, 120], dtype=torch.int32, device="cuda")
    seq_kv_lens = torch.tensor([180, 240], dtype=torch.int32, device="cuda")
    _run_case(
        batch=2,
        h_q=4,
        h_kv=4,
        s_q=256,
        s_kv=256,
        causal_bottom_right=True,
        window_size_right=48,
        window_size_left=96,
        seq_q_lens=seq_q_lens,
        seq_kv_lens=seq_kv_lens,
    )
    # kv_tile=64 (auto-picked at d=248) with R a multiple of the tile: the
    # 3-step masked frontier at the smaller tile.
    _run_case(head_dim=248, head_dim_v=248, s_q=128, s_kv=128, window_size_right=64)
    # Degenerate R >= S_kv: the widened bound clamps to full visibility.
    _run_case(batch=2, h_q=4, h_kv=4, s_q=128, s_kv=128, window_size_right=200)


@pytest.mark.L1
@torch_fork_set_rng(seed=37)
def test_dsl_sm120_thd_right_band():
    """THD + TOP_LEFT right band: per-sequence diagonals each widened by R,
    with the ragged Stats checked."""

    _run_thd_case(seq_q_lens=[130, 70], seq_kv_lens=[130, 70], window_size_right=24, check_stats=True)
    # BOTTOM_RIGHT band under THD: each sequence's own diagonal, widened.
    _run_thd_case(seq_q_lens=[100, 60], seq_kv_lens=[180, 120], causal_bottom_right=True, window_size_right=24, check_stats=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=33)
def test_dsl_sm120_ragged_skv_tail():
    """S_kv not a multiple of the KV tile, served natively (skv_tile=0): the
    kernel's first masked step covers the partial rightmost tile in every
    configuration — no padding mask, no synthesized lengths.

    Cases: dense unmasked; top-left causal with S_q > S_kv (the corner causal_covers_tail
    excludes); causal + sliding window across a ragged tail; ragged tail with
    the LSE checked."""

    _run_case(batch=2, h_q=4, h_kv=4, s_q=256, s_kv=300, head_dim=128)
    _run_case(batch=2, h_q=4, h_kv=4, s_q=384, s_kv=200, head_dim=128, is_causal=True)
    _run_case(batch=2, h_q=4, h_kv=4, s_q=256, s_kv=300, head_dim=128, is_causal=True, window_size_left=96)
    _run_case(batch=2, h_q=8, h_kv=2, s_q=192, s_kv=333, head_dim=64, check_stats=True)
    _run_case(batch=2, h_q=4, h_kv=4, s_q=128, s_kv=40, head_dim=128)  # num_kv_tiles == 1, tail-only tile


@pytest.mark.L0
@torch_fork_set_rng(seed=6)
def test_dsl_sm120_padded():
    """Per-batch Q and KV lengths, both genuinely shorter than the padded
    shapes so the Q-row trim (O := 0) and the KV column mask are exercised."""

    seq_q_lens = torch.tensor([230, 120], dtype=torch.int32, device="cuda")
    seq_kv_lens = torch.tensor([180, 240], dtype=torch.int32, device="cuda")
    _run_case(
        batch=2,
        h_q=8,
        h_kv=8,
        s_q=256,
        s_kv=256,
        head_dim=64,
        seq_q_lens=seq_q_lens,
        seq_kv_lens=seq_kv_lens,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=7)
def test_dsl_sm120_bfloat16_and_tile_variants():
    _run_case(
        head_dim=128,
        dtype=torch.bfloat16,
        q_tile=64,
        kv_tile=64,
        is_causal=True,
        scale=0.5,
    )
