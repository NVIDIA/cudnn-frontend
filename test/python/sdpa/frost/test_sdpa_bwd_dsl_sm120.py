# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""End-to-end tests for the FROST SM120 DSL SDPA-backward engine against a torch reference."""

from __future__ import annotations

import math

import pytest
import torch

from test_utils import torch_fork_set_rng
from frost_test_utils import requires_blackwell_geforce, requires_dsl, _dsl_installed

ENGINE = "sdpa_bwd_sm120"


def _is_sm120() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return (major, minor) in {(12, 0), (12, 1)}


pytestmark = pytest.mark.skipif(
    not _is_sm120(),
    reason="SM120 DSL SDPA backward engine requires an SM120 or SM121 device.",
)


@pytest.fixture(autouse=True)
def _enable_frost(monkeypatch):
    """FROST engines resolve only under the env opt-in (read live per call)."""

    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")


def _require_dsl() -> None:
    try:
        import cudnn  # noqa: F401
        import cudnn.sdpa  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"SM120 DSL engine not available: {exc}")
    if not _dsl_installed():
        pytest.skip("cutlass/dsl not installed")


from frost_test_utils import select_engine as _select_engine  # noqa: F401


def _bhsd(batch: int, heads: int, sequence: int, head_dim: int, dtype: torch.dtype, empty: bool = False, layout: str = "bshd") -> torch.Tensor:
    """Logical BHSD over compact BSHD storage, BHSD-contiguous for
    layout="bhsd", or BSHD storage with an 8-element sub-token gap after
    every row for layout="gapped" (padded strides, 16-byte multiples)."""

    factory = torch.empty if empty else torch.randn
    if layout == "bhsd":
        return factory(batch, heads, sequence, head_dim, dtype=dtype, device="cuda")
    if layout == "gapped":
        return factory(batch, sequence, heads, head_dim + 8, dtype=dtype, device="cuda").transpose(1, 2)[..., :head_dim]
    return factory(batch, sequence, heads, head_dim, dtype=dtype, device="cuda").transpose(1, 2)


def _strided_stats(stats: torch.Tensor) -> torch.Tensor:
    """Rebuild (B, H, S, 1) stats on permuted, gapped storage (S-major) —
    the layout family the randomized upstream configs generate."""

    b, h, s, _ = stats.shape
    base = torch.empty(s + 7, h + 2, b, dtype=stats.dtype, device=stats.device)
    view = base.permute(2, 1, 0)[:, :h, :s].unsqueeze(-1)
    view.copy_(stats)
    assert not view.is_contiguous()
    return view


def _ref_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    *,
    scale: float,
    is_causal: bool = False,
    causal_bottom_right: bool = False,
    window_size_left: int | None = None,
    window_size_right: int | None = None,
    padding: tuple[list[int], list[int]] | None = None,
    sink_token: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
):
    """Reference via the canonical refs (sdpa/fp16_ref.py)."""

    import cudnn
    from sdpa.fp16_ref import compute_ref, compute_ref_backward

    diag_align = cudnn.diagonal_alignment.BOTTOM_RIGHT if causal_bottom_right else cudnn.diagonal_alignment.TOP_LEFT
    right_bound = window_size_right if window_size_right is not None else (0 if is_causal else None)
    # The refs take the cuDNN window LENGTH; window_size_left is the offset W = L - 1.
    left_bound = None if window_size_left is None else window_size_left + 1
    o_ref, stats_ref, _, _ = compute_ref(
        q,
        k,
        v,
        attn_scale=scale,
        bias=bias,
        diag_align=diag_align,
        right_bound=right_bound,
        left_bound=left_bound,
        padding=padding,
        sink_token=sink_token,
        torch_type=q.dtype,
    )
    dq, dk, dv, dbias, dsink = compute_ref_backward(
        q,
        k,
        v,
        o_ref,
        do,
        attn_scale=scale,
        bias=bias,
        diag_align=diag_align,
        right_bound=right_bound,
        left_bound=left_bound,
        padding=padding,
        sink_token=sink_token,
        torch_type=q.dtype,
    )
    from types import SimpleNamespace

    aux = SimpleNamespace(dsink=dsink, dbias=dbias)
    return o_ref.to(q.dtype), stats_ref.contiguous(), dq.to(q.dtype), dk.to(q.dtype), dv.to(q.dtype), aux


def _expect_det_2k(
    *,
    head_dim: int,
    head_dim_v: int,
    deterministic: bool,
    window_size_left: int | None,
    padded: bool,
    ws_bytes: int,
    q_tile: int | None = None,
) -> bool:
    """Mirror of SdpaBwdDslSm120._pick_det_2k (two-kernel deterministic route)."""
    from cudnn.sdpa.bwd.api_dsl import _SM120_DET_2K_HEAD_DIM_PAIRS
    from cudnn.sdpa.bwd.config_sm120 import padded_head_dims

    if not deterministic:
        return False
    pads = padded_head_dims(head_dim, head_dim_v)
    if pads not in _SM120_DET_2K_HEAD_DIM_PAIRS or pads[0] != head_dim:
        return False
    if window_size_left is not None or padded:
        return False
    if q_tile and (128 if pads[0] <= 128 else 64) % q_tile:
        return False
    # 2K whenever the full dS buffer can physically fit
    return ws_bytes <= torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory


def _expected_workspace_bytes(
    batch: int,
    h_q: int,
    s_q: int,
    head_dim: int,
    h_kv: int | None = None,
    s_kv: int | None = None,
    io_itemsize: int = 2,
    head_dim_v: int | None = None,
    deterministic: bool = False,
    window_size_left: int | None = None,
    padded: bool = False,
    dbias_batch: int = 0,
    q_tile: int | None = None,
) -> int:
    from cudnn.sdpa.bwd.config_sm120 import padded_head_dims
    from cudnn.sdpa.fwd.api_dsl import ws_align

    head_dim_v = head_dim if head_dim_v is None else head_dim_v
    # Per-side native kernel head-dim sizes — same helper the adapter uses.
    d_pad, dv_pad = padded_head_dims(head_dim, head_dim_v)
    sq_r = -(-s_q // 128) * 128
    h_kv = h_q if h_kv is None else h_kv
    s_kv_eff = s_kv if s_kv is not None else s_q
    skv_r = -(-s_kv_eff // 128) * 128
    delta_ws = ws_align(batch * h_q * sq_r * 4)
    dbias_ws = ws_align(dbias_batch * h_q * s_q * s_kv_eff * 4) if dbias_batch else 0
    # dk_ws/dv_ws GQA partials buffers in the io dtype (none carved for MHA, where the main kernel writes dk/dv directly)
    dkv_ws = 0
    if h_kv != h_q:
        dkv_ws = ws_align(batch * s_kv_eff * h_q * d_pad * io_itemsize) + ws_align(batch * s_kv_eff * h_q * dv_pad * io_itemsize)
    ds_ws_bytes = ws_align(batch * h_q * s_q * skv_r * io_itemsize)  # det_2kernel dS workspace
    if _expect_det_2k(
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        deterministic=deterministic,
        window_size_left=window_size_left,
        padded=padded,
        ws_bytes=delta_ws + ds_ws_bytes + dbias_ws + dkv_ws,  # the full two-kernel scratch
        q_tile=q_tile,
    ):
        dq_scratch = ds_ws_bytes
    else:
        dq_sem = batch * h_q * (-(-s_q // 32))  # int32 relay counters (min q-tile 32)
        dq_scratch = ws_align(batch * sq_r * h_q * d_pad * 4) + ws_align(dq_sem * 4)
    return delta_ws + dq_scratch + dbias_ws + dkv_ws


def _run_bwd_graph(
    q_gpu: torch.Tensor,
    k_gpu: torch.Tensor,
    v_gpu: torch.Tensor,
    o_gpu: torch.Tensor,
    do_gpu: torch.Tensor,
    stats_gpu: torch.Tensor,
    *,
    scale: float,
    is_causal: bool = False,
    causal_bottom_right: bool = False,
    window_size_left: int | None = None,
    window_size_right: int | None = None,
    deterministic: bool = False,
    select: bool = True,
    q_tile: int | None = None,
    kv_tile: int | None = None,
    grad_layout: "str | tuple[str, str, str]" = "bshd",
    grads: "tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None" = None,
    seq_q_lens: torch.Tensor | None = None,
    seq_kv_lens: torch.Tensor | None = None,
    sink_gpu: torch.Tensor | None = None,
    bias_gpu: torch.Tensor | None = None,
    dbias_gpu: torch.Tensor | None = None,  # filled in place when given (requires bias_gpu)
    build_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, "torch.Tensor | None", str]:
    """Build and execute the SM120 FROST backward graph; returns (dq, dk, dv, dsink, plan_name)."""

    _require_dsl()
    import cudnn

    dtype = q_gpu.dtype
    io_dtype = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    batch, h_q, _, head_dim = q_gpu.shape
    _, h_kv, _, _ = k_gpu.shape
    head_dim_v = v_gpu.shape[3]
    gl = (grad_layout,) * 3 if isinstance(grad_layout, str) else grad_layout
    if grads is not None:
        dq_gpu, dk_gpu, dv_gpu = grads
    else:
        dq_gpu = _bhsd(batch, h_q, q_gpu.shape[2], head_dim, dtype, empty=True, layout=gl[0])
        dk_gpu = _bhsd(batch, h_kv, k_gpu.shape[2], head_dim, dtype, empty=True, layout=gl[1])
        dv_gpu = _bhsd(batch, h_kv, v_gpu.shape[2], head_dim_v, dtype, empty=True, layout=gl[2])

    graph = cudnn.pygraph(
        io_data_type=io_dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q = graph.tensor_like(q_gpu, name="q")
    k = graph.tensor_like(k_gpu, name="k")
    v = graph.tensor_like(v_gpu, name="v")
    o = graph.tensor_like(o_gpu, name="o")
    do = graph.tensor_like(do_gpu, name="dO")
    stats = graph.tensor_like(stats_gpu, name="stats")

    bwd_kwargs = {
        "name": "sdpa_backward",
        "q": q,
        "k": k,
        "v": v,
        "o": o,
        "dO": do,
        "stats": stats,
        "attn_scale": scale,
    }
    if window_size_right is not None:
        bwd_kwargs["diagonal_band_right_bound"] = window_size_right
        bwd_kwargs["diagonal_alignment"] = cudnn.diagonal_alignment.BOTTOM_RIGHT if causal_bottom_right else cudnn.diagonal_alignment.TOP_LEFT
        if window_size_left is not None:
            bwd_kwargs["diagonal_band_left_bound"] = window_size_left + 1
    else:
        if causal_bottom_right:
            bwd_kwargs["use_causal_mask_bottom_right"] = True
        elif is_causal:
            bwd_kwargs["use_causal_mask"] = True
        if window_size_left is not None:
            bwd_kwargs["sliding_window_length"] = window_size_left + 1
    if deterministic:
        bwd_kwargs["use_deterministic_algorithm"] = True
    sink_t = dsink_t = dsink_gpu = None
    if sink_gpu is not None:
        sink_t = graph.tensor_like(sink_gpu, name="sink")
        dsink_gpu = torch.empty_like(sink_gpu)
        dsink_t = graph.tensor_like(dsink_gpu, name="dSink")
        bwd_kwargs.update(sink_token=sink_t, dSink_token=dsink_t)
    bias_t = dbias_t = None
    if bias_gpu is not None:
        bias_t = graph.tensor_like(bias_gpu, name="bias")
        bwd_kwargs.update(bias=bias_t)
        if dbias_gpu is not None:
            dbias_t = graph.tensor_like(dbias_gpu, name="dBias")
            bwd_kwargs.update(dBias=dbias_t)
    seq_q_t = seq_kv_t = None
    if seq_q_lens is not None or seq_kv_lens is not None:
        assert seq_q_lens is not None and seq_kv_lens is not None
        seq_q_t = graph.tensor_like(seq_q_lens, name="seq_q")
        seq_kv_t = graph.tensor_like(seq_kv_lens, name="seq_kv")
        bwd_kwargs.update(use_padding_mask=True, seq_len_q=seq_q_t, seq_len_kv=seq_kv_t)

    dq, dk, dv = graph.sdpa_backward(**bwd_kwargs)
    dq.set_output(True).set_dim(dq_gpu.shape).set_stride(dq_gpu.stride())
    dk.set_output(True).set_dim(dk_gpu.shape).set_stride(dk_gpu.stride())
    dv.set_output(True).set_dim(dv_gpu.shape).set_stride(dv_gpu.stride())

    graph.validate()
    graph.build_operation_graph()
    if q_tile is not None or kv_tile is not None:
        # A knob request rides on a plan entry (PlanConfig.knobs): append
        # exactly one (engine_id, knobs) plan — the deterministic-replay path.
        from cudnn.engines.engine_ids import FROST_SDPA_BWD_ID_BASE
        from cudnn.sdpa.bwd.engines import SdpaBwdKnobs

        graph.create_execution_plan(FROST_SDPA_BWD_ID_BASE + 0, SdpaBwdKnobs(tile_m=q_tile, tile_n=kv_tile))
        graph.select_plan(0)
    else:
        graph.create_execution_plans([cudnn.heur_mode.A])
        if select:
            _select_engine(graph, ENGINE)
    graph.check_support()
    graph.build_plans()
    # What actually runs, not what merely ranked first: build_plans settles the
    # plan index on the entry that built.
    engine = graph.selected_engine
    plan_name = engine.name if engine is not None else "backend"
    if select or q_tile is not None or kv_tile is not None:
        assert plan_name == ENGINE, f"pinned {ENGINE} but {plan_name} would run"
    if build_only:
        return dq_gpu, dk_gpu, dv_gpu, dsink_gpu, plan_name

    workspace_size = graph.get_workspace_size()
    if plan_name == ENGINE:
        assert workspace_size == _expected_workspace_bytes(
            batch,
            h_q,
            q_gpu.shape[2],
            head_dim,
            h_kv=h_kv,
            s_kv=k_gpu.shape[2],
            io_itemsize=q_gpu.element_size(),
            head_dim_v=head_dim_v,
            deterministic=deterministic,
            window_size_left=window_size_left,
            padded=seq_kv_lens is not None,
            dbias_batch=dbias_gpu.shape[0] if dbias_gpu is not None and dbias_gpu.dtype != torch.float32 else 0,
            q_tile=q_tile,
        )
    workspace = torch.empty(max(workspace_size, 1), dtype=torch.uint8, device="cuda")

    variant_pack = {
        q: q_gpu,
        k: k_gpu,
        v: v_gpu,
        o: o_gpu,
        do: do_gpu,
        stats: stats_gpu,
        dq: dq_gpu,
        dk: dk_gpu,
        dv: dv_gpu,
    }
    if seq_q_t is not None:
        variant_pack.update({seq_q_t: seq_q_lens, seq_kv_t: seq_kv_lens})
    if sink_t is not None:
        variant_pack.update({sink_t: sink_gpu, dsink_t: dsink_gpu})
    if bias_t is not None:
        variant_pack.update({bias_t: bias_gpu})
        if dbias_t is not None:
            variant_pack.update({dbias_t: dbias_gpu})
    graph.execute(variant_pack, workspace)
    torch.cuda.synchronize()
    return dq_gpu, dk_gpu, dv_gpu, dsink_gpu, plan_name


def _tolerances(dtype: torch.dtype) -> dict:
    return {"atol": 2e-2 if dtype == torch.float16 else 5e-2, "rtol": 5e-2}


def _run_case(
    *,
    batch: int = 2,
    h_q: int = 4,
    h_kv: int | None = None,
    s_q: int = 512,
    s_kv: int = 512,
    head_dim: int = 64,
    head_dim_v: int | None = None,
    dtype: torch.dtype = torch.float16,
    is_causal: bool = False,
    causal_bottom_right: bool = False,
    window_size_left: int | None = None,
    window_size_right: int | None = None,
    deterministic: bool = False,
    select: bool = True,
    q_tile: int | None = None,
    kv_tile: int | None = None,
    layout: str = "bshd",
    grad_layout: str = "bshd",
    padding: tuple[list[int], list[int]] | None = None,
    sink: bool = False,
    bias: bool = False,
    bias_dtype: torch.dtype | None = None,  # None = the io dtype
    dbias: bool = True,  # bias only: also request the dBias output
    stats_layout: str = "contiguous",
) -> str:
    h_kv = h_q if h_kv is None else h_kv
    head_dim_v = head_dim if head_dim_v is None else head_dim_v
    scale = 1.0 / math.sqrt(head_dim)
    q = _bhsd(batch, h_q, s_q, head_dim, dtype, layout=layout)
    k = _bhsd(batch, h_kv, s_kv, head_dim, dtype, layout=layout)
    v = _bhsd(batch, h_kv, s_kv, head_dim_v, dtype, layout=layout)
    do = _bhsd(batch, h_q, s_q, head_dim_v, dtype, layout=layout)
    sink_gpu = torch.randn(1, h_q, 1, 1, dtype=torch.float32, device="cuda") if sink else None
    bias_gpu = dbias_gpu = None
    if bias:
        bias_gpu = torch.randn(1, h_q, s_q, s_kv, dtype=bias_dtype or dtype, device="cuda")
        if dbias:
            dbias_gpu = torch.empty_like(bias_gpu)
    o, stats, dq_ref, dk_ref, dv_ref, aux = _ref_bwd(
        q,
        k,
        v,
        do,
        scale=scale,
        is_causal=is_causal,
        causal_bottom_right=causal_bottom_right,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        padding=padding,
        sink_token=sink_gpu,
        bias=bias_gpu.float() if bias_gpu is not None else None,
    )
    o = _bhsd(batch, h_q, s_q, head_dim_v, dtype, empty=True, layout=layout).copy_(o)
    if stats_layout == "strided":
        stats = _strided_stats(stats)
    seq_q_lens = seq_kv_lens = None
    if padding is not None:
        seq_q_lens = torch.tensor(padding[0], dtype=torch.int32, device="cuda").view(batch, 1, 1, 1)
        seq_kv_lens = torch.tensor(padding[1], dtype=torch.int32, device="cuda").view(batch, 1, 1, 1)
    dq, dk, dv, dsink, plan_name = _run_bwd_graph(
        q,
        k,
        v,
        o,
        do,
        stats,
        scale=scale,
        is_causal=is_causal,
        causal_bottom_right=causal_bottom_right,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        deterministic=deterministic,
        select=select,
        q_tile=q_tile,
        kv_tile=kv_tile,
        grad_layout=grad_layout,
        seq_q_lens=seq_q_lens,
        seq_kv_lens=seq_kv_lens,
        sink_gpu=sink_gpu,
        bias_gpu=bias_gpu,
        dbias_gpu=dbias_gpu,
    )
    tol = _tolerances(dtype)
    torch.testing.assert_close(dq.float(), dq_ref.float(), **tol)
    torch.testing.assert_close(dk.float(), dk_ref.float(), **tol)
    torch.testing.assert_close(dv.float(), dv_ref.float(), **tol)
    if sink:
        torch.testing.assert_close(dsink.float(), aux.dsink.float(), **tol)
    if dbias_gpu is not None:
        torch.testing.assert_close(dbias_gpu.float(), aux.dbias.float(), **tol)
    if padding is not None:
        for b, (len_q, len_kv) in enumerate(zip(*padding)):
            if len_q < s_q:
                assert dq[b, :, len_q:, :].abs().max().item() == 0.0, f"batch {b}: dQ padded rows must be exactly zero"
            if len_kv < s_kv:
                assert dk[b, :, len_kv:, :].abs().max().item() == 0.0, f"batch {b}: dK padded rows must be exactly zero"
                assert dv[b, :, len_kv:, :].abs().max().item() == 0.0, f"batch {b}: dV padded rows must be exactly zero"
    return plan_name


@pytest.mark.L0
@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.parametrize("is_causal", [False, True], ids=["dense", "causal"])
@torch_fork_set_rng(seed=0)
def test_sdpa_bwd_dsl_sm120_graph_api(head_dim: int, is_causal: bool):
    """FP16 numeric parity per head dim, dense and top-left causal (S_q == S_kv)."""

    _run_case(head_dim=head_dim, is_causal=is_causal)


@pytest.mark.L0
@pytest.mark.parametrize("is_causal", [False, True], ids=["dense", "causal"])
@torch_fork_set_rng(seed=1)
def test_sdpa_bwd_dsl_sm120_bf16(is_causal: bool):
    """BF16 numeric parity at d=64."""

    _run_case(dtype=torch.bfloat16, head_dim=64, is_causal=is_causal)


@pytest.mark.L0
@torch_fork_set_rng(seed=2)
def test_sdpa_bwd_dsl_sm120_cross_seqlen_causal_br():
    """Bottom-right causal with S_q < S_kv (the decode-style tail)."""

    _run_case(s_q=384, s_kv=1024, head_dim=64, is_causal=True, causal_bottom_right=True)


@pytest.mark.L0
@pytest.mark.parametrize(("s_q", "s_kv"), [(1024, 384), (193, 64)], ids=["sq_gt_skv", "sq_gt_skv_tails"])
@torch_fork_set_rng(seed=6)
def test_sdpa_bwd_dsl_sm120_causal_br_sq_gt_skv(s_q: int, s_kv: int):
    """Bottom-right causal with S_q > S_kv"""

    _require_dsl()
    import cudnn

    # Bottom right causal mask does not support max_s_q > max_s_kv in graph api.
    try:
        _run_case(s_q=s_q, s_kv=s_kv, head_dim=64, is_causal=True, causal_bottom_right=True)
        return
    except cudnn.cudnnGraphNotSupportedError as exc:
        assert "max_s_q > max_s_kv" in str(exc), f"unexpected graph rejection: {exc}"

    from cudnn.sdpa.bwd.api_dsl import sdpa_bwd_wrapper_dsl_sm120

    batch, heads, head_dim, dtype = 2, 4, 64, torch.float16
    scale = 1.0 / math.sqrt(head_dim)
    q = _bhsd(batch, heads, s_q, head_dim, dtype)
    k = _bhsd(batch, heads, s_kv, head_dim, dtype)
    v = _bhsd(batch, heads, s_kv, head_dim, dtype)
    do = _bhsd(batch, heads, s_q, head_dim, dtype)
    o, stats, dq_ref, dk_ref, dv_ref, _ = _ref_bwd(q, k, v, do, scale=scale, is_causal=True, causal_bottom_right=True)
    o = _bhsd(batch, heads, s_q, head_dim, dtype, empty=True).copy_(o)
    out = sdpa_bwd_wrapper_dsl_sm120(q, k, v, o, do, stats, is_causal=True, causal_bottom_right=True, scale_softmax=scale)
    tol = _tolerances(dtype)
    torch.testing.assert_close(out["dq_tensor"].float(), dq_ref.float(), **tol)
    torch.testing.assert_close(out["dk_tensor"].float(), dk_ref.float(), **tol)
    torch.testing.assert_close(out["dv_tensor"].float(), dv_ref.float(), **tol)
    assert out["dq_tensor"][:, :, : s_q - s_kv, :].abs().max().item() == 0.0, "fully-masked rows must have exactly zero dQ"


@pytest.mark.L0
@pytest.mark.parametrize(
    ("s_q", "s_kv"),
    [(384, 1024), (1024, 384), (64, 512)],
    ids=["sq_lt_skv", "sq_gt_skv", "empty_kv_tiles"],
)
@torch_fork_set_rng(seed=3)
def test_sdpa_bwd_dsl_sm120_cross_seqlen_causal_top_left(s_q: int, s_kv: int):
    """Top-left causal with S_q != S_kv."""

    _run_case(s_q=s_q, s_kv=s_kv, head_dim=64, is_causal=True)


@pytest.mark.L0
@pytest.mark.parametrize("mask", ["dense", "causal_br", "causal_tl"])
@torch_fork_set_rng(seed=4)
def test_sdpa_bwd_dsl_sm120_sequence_tails(mask: str):
    """Non-tile-multiple sequence tails exercise the partial-Q/KV predicates."""

    _run_case(
        s_q=193,
        s_kv=257,
        head_dim=128,
        is_causal=mask != "dense",
        causal_bottom_right=mask == "causal_br",
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=8)
def test_sdpa_bwd_dsl_sm120_sliding_window_causal():
    """Top-left causal + sliding window (the training SWA shape)."""

    _run_case(s_q=1024, s_kv=1024, head_dim=64, is_causal=True, window_size_left=127)


@pytest.mark.L0
@torch_fork_set_rng(seed=9)
def test_sdpa_bwd_dsl_sm120_sliding_window_no_causal():
    """A left window without a causal bit (band open to the right)."""

    _run_case(head_dim=64, window_size_left=63)


@pytest.mark.L0
@torch_fork_set_rng(seed=10)
def test_sdpa_bwd_dsl_sm120_sliding_window_causal_br():
    """Bottom-right causal + sliding window across unequal sequence lengths."""

    _run_case(s_q=384, s_kv=1024, head_dim=64, is_causal=True, causal_bottom_right=True, window_size_left=127)


@pytest.mark.L0
@torch_fork_set_rng(seed=11)
def test_sdpa_bwd_dsl_sm120_sliding_window_tails():
    """Sub-tile sliding window with non-tile-multiple sequence tails."""

    _run_case(s_q=193, s_kv=257, head_dim=128, is_causal=True, window_size_left=16)


@pytest.mark.L0
@torch_fork_set_rng(seed=46)
def test_sdpa_bwd_dsl_sm120_right_band():
    """diagonal_band_right_bound > 0: the causal diagonal widened right by a
    compile-time R (keep kv <= q + diag + R)."""

    _run_case(s_q=256, s_kv=256, head_dim=64, window_size_right=32)  # top-left band
    _run_case(s_q=192, s_kv=320, head_dim=64, causal_bottom_right=True, window_size_right=48)  # bottom-right anchor
    _run_case(s_q=256, s_kv=256, head_dim=64, window_size_left=64, window_size_right=32)  # full band
    _run_case(s_q=193, s_kv=257, head_dim=128, window_size_right=24)  # ragged tails
    _run_case(s_q=128, s_kv=128, head_dim=64, window_size_right=300)  # R >= S_kv clamps to dense
    _run_case(s_q=256, s_kv=256, head_dim=64, window_size_right=32, deterministic=True)  # relay turns unaffected by R
    _run_case(
        s_q=256,
        s_kv=256,
        head_dim=64,
        causal_bottom_right=True,
        window_size_right=48,
        window_size_left=96,
        padding=([230, 120], [180, 240]),
    )  # per-batch diagonal + R


@pytest.mark.L0
@torch_fork_set_rng(seed=49)
def test_sdpa_bwd_dsl_sm120_sink():
    """Sink attention"""

    _run_case(s_q=256, s_kv=256, head_dim=64, sink=True)  # dense
    _run_case(s_q=256, s_kv=256, head_dim=64, is_causal=True, sink=True)  # causal
    _run_case(h_q=8, h_kv=2, s_q=256, s_kv=256, head_dim=64, sink=True)  # GQA
    _run_case(s_q=256, s_kv=256, head_dim=64, sink=True, deterministic=True)  # fixed-order reduce
    _run_case(s_q=193, s_kv=257, head_dim=128, is_causal=True, sink=True)  # ragged tails
    _run_case(s_q=256, s_kv=256, head_dim=64, sink=True, padding=([230, 120], [180, 240]))  # padded rows skip (LSE = -inf guard)


@pytest.mark.L0
@pytest.mark.parametrize("mask", ["dense", "causal"])
@torch_fork_set_rng(seed=60)
def test_sdpa_bwd_dsl_sm120_bias(mask: str):
    """Additive bias input + dBias output ((1, H, S_q, S_kv), batch-broadcast).
    Bias indexing is head-dim independent, so one head dim per mask suffices."""

    _run_case(head_dim=64 if mask == "dense" else 128, is_causal=mask == "causal", bias=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=61)
def test_sdpa_bwd_dsl_sm120_bias_features():
    """Bias composed with the other features."""

    _run_case(head_dim=64, bias=True, dbias=False)  # no dBias output requested
    _run_case(dtype=torch.bfloat16, head_dim=64, s_q=333, s_kv=467, bias=True)  # bf16 + partial tiles (bounds-checked red.add)
    _run_case(head_dim=64, bias=True, bias_dtype=torch.float32)  # fp32 bias/dBias: in-place accumulation, no cvt kernel or accumulator ws
    _run_case(h_q=8, h_kv=2, head_dim=64, bias=True)  # GQA: dBias is per q head — no group reduction
    _run_case(batch=2, head_dim=64, s_q=512, s_kv=512, bias=True, padding=([301, 512], [512, 187]))  # padded cells contribute zero


@pytest.mark.L0
@torch_fork_set_rng(seed=67)
def test_sdpa_bwd_dsl_sm120_bias_broadcast_det_rejected():
    """B > 1 + batch-broadcast bias + dBias is rejected in deterministic mode
    (dBias reduces over B through unordered atomics)."""

    from cudnn.sdpa.bwd.api_dsl import sdpa_bwd_wrapper_dsl_sm120

    batch, heads, s, head_dim, dtype = 2, 2, 128, 64, torch.float16
    q = _bhsd(batch, heads, s, head_dim, dtype)
    o = _bhsd(batch, heads, s, head_dim, dtype)
    stats = torch.zeros(batch, heads, s, 1, dtype=torch.float32, device="cuda")
    bias = torch.randn(1, heads, s, s, dtype=dtype, device="cuda")
    with pytest.raises(ValueError, match="per-batch"):
        sdpa_bwd_wrapper_dsl_sm120(q, q, q, o, o, stats, deterministic=True, bias_tensor=bias)


@pytest.mark.L0
@pytest.mark.parametrize("head_dim", [64, 128])
@torch_fork_set_rng(seed=68)
def test_sdpa_bwd_dsl_sm120_bias_masked_rows(head_dim: int):
    """All--inf bias rows (additive masking) yield zero gradients, not NaN
    (the LSE = -inf flip): deterministic d64 pins the relay, d128 the
    two-kernel route."""

    # B = 1: deterministic + dBias requires a per-batch bias when B > 1.
    batch, h_q, s, dtype = 1, 4, 256, torch.float16
    scale = 1.0 / math.sqrt(head_dim)
    q = _bhsd(batch, h_q, s, head_dim, dtype)
    k = _bhsd(batch, h_q, s, head_dim, dtype)
    v = _bhsd(batch, h_q, s, head_dim, dtype)
    do = _bhsd(batch, h_q, s, head_dim, dtype)
    bias = torch.randn(1, h_q, s, s, dtype=dtype, device="cuda")
    bias[:, :, :3, :] = float("-inf")  # fully masked rows
    bias[:, :, 7, 1::2] = float("-inf")  # partially masked row
    o, stats, dq_ref, dk_ref, dv_ref, aux = _ref_bwd(q, k, v, do, scale=scale, bias=bias.float())
    o = _bhsd(batch, h_q, s, head_dim, dtype, empty=True).copy_(o)
    dbias = torch.empty_like(bias)
    dq, dk, dv, _, _ = _run_bwd_graph(q, k, v, o, do, stats, scale=scale, deterministic=True, bias_gpu=bias, dbias_gpu=dbias)
    tol = _tolerances(dtype)
    for name, got, want in (("dq", dq, dq_ref), ("dk", dk, dk_ref), ("dv", dv, dv_ref), ("dbias", dbias, aux.dbias)):
        assert not torch.isnan(got).any(), f"{name} has NaN"
        torch.testing.assert_close(got.float(), want.float(), **tol)


@pytest.mark.L0
@torch_fork_set_rng(seed=69)
def test_sdpa_bwd_dsl_sm120_det_2k_memory_gate(monkeypatch):
    """The two-kernel route is picked only when its FULL scratch (delta + dS
    + dBias accumulator + GQA partials) fits in device memory."""

    import cudnn.sdpa.bwd.api_dsl as api_mod
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm120

    batch, h_q, h_kv, s, d, dtype = 2, 8, 2, 256, 128, torch.float16  # GQA: the partials count too

    def mk():
        t = lambda h, dd: _bhsd(batch, h, s, dd, dtype, empty=True)
        stats = torch.empty(batch, h_q, s, 1, dtype=torch.float32, device="cuda")
        return SdpaBwdDslSm120(
            sample_q=t(h_q, d),
            sample_k=t(h_kv, d),
            sample_v=t(h_kv, d),
            sample_o=t(h_q, d),
            sample_do=t(h_q, d),
            sample_stats=stats,
            sample_dq=t(h_q, d),
            sample_dk=t(h_kv, d),
            sample_dv=t(h_kv, d),
            deterministic=True,
        )

    def picks_det_2k(api):
        api.scratch_workspace_bytes()  # runs the lazy support check that sets det_2k
        return api.det_2k

    api = mk()
    full = api.scratch_workspace_bytes()
    assert api.det_2k, "shape must be two-kernel eligible on the real device"

    real = torch.cuda.get_device_properties(torch.cuda.current_device())

    class _Props:
        def __init__(self, total):
            self.total_memory = total
            # get_device_capability reads these through get_device_properties
            self.major, self.minor = real.major, real.minor

    monkeypatch.setattr(api_mod.torch.cuda, "get_device_properties", lambda dev: _Props(full))
    assert picks_det_2k(mk())  # exactly fits
    monkeypatch.setattr(api_mod.torch.cuda, "get_device_properties", lambda dev: _Props(full - 1))
    assert not picks_det_2k(mk())  # one byte short of the FULL scratch -> relay


@pytest.mark.L0
@torch_fork_set_rng(seed=66)
def test_sdpa_bwd_dsl_sm120_bias_wrapper():
    """Wrapper bias path: io-dtype bias in, fp32 dBias out (accumulated in
    place — no accumulator workspace, no convert kernel)."""

    from cudnn.sdpa.bwd.api_dsl import sdpa_bwd_wrapper_dsl_sm120

    batch, heads, s, head_dim, dtype = 2, 4, 256, 64, torch.float16
    scale = 1.0 / math.sqrt(head_dim)
    q = _bhsd(batch, heads, s, head_dim, dtype)
    k = _bhsd(batch, heads, s, head_dim, dtype)
    v = _bhsd(batch, heads, s, head_dim, dtype)
    do = _bhsd(batch, heads, s, head_dim, dtype)
    bias = torch.randn(1, heads, s, s, dtype=dtype, device="cuda")
    o, stats, dq_ref, dk_ref, dv_ref, aux = _ref_bwd(q, k, v, do, scale=scale, is_causal=True, bias=bias.float())
    o = _bhsd(batch, heads, s, head_dim, dtype, empty=True).copy_(o)
    out = sdpa_bwd_wrapper_dsl_sm120(q, k, v, o, do, stats, is_causal=True, scale_softmax=scale, bias_tensor=bias)
    assert out["dbias_tensor"].dtype == torch.float32
    tol = _tolerances(dtype)
    torch.testing.assert_close(out["dq_tensor"].float(), dq_ref.float(), **tol)
    torch.testing.assert_close(out["dk_tensor"].float(), dk_ref.float(), **tol)
    torch.testing.assert_close(out["dv_tensor"].float(), dv_ref.float(), **tol)
    torch.testing.assert_close(out["dbias_tensor"], aux.dbias.float(), **tol)


@pytest.mark.L0
@pytest.mark.parametrize("mask", ["dense", "causal_tl", "causal_br"])
@torch_fork_set_rng(seed=20)
def test_sdpa_bwd_dsl_sm120_padding_mask(mask: str):
    """Padding mask (per-batch seq lens): full-length, tile-boundary, and
    sub-tile batches; bottom-right diagonals anchor at the actual lengths."""

    _run_case(
        batch=3,
        s_q=512,
        s_kv=512,
        head_dim=64,
        is_causal=mask != "dense",
        causal_bottom_right=mask == "causal_br",
        padding=([512, 300, 17], [512, 128, 65]),
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=21)
def test_sdpa_bwd_dsl_sm120_padding_mask_tails():
    """Padding mask on top of non-tile-multiple global sequence tails."""

    _run_case(
        batch=2,
        s_q=193,
        s_kv=257,
        head_dim=128,
        is_causal=True,
        padding=([193, 100], [200, 33]),
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=22)
def test_sdpa_bwd_dsl_sm120_padding_sliding_window():
    """Padding mask + bottom-right sliding window (the window follows the
    per-batch diagonal anchor)."""

    _run_case(
        batch=3,
        s_q=512,
        s_kv=512,
        head_dim=64,
        is_causal=True,
        causal_bottom_right=True,
        window_size_left=127,
        padding=([512, 300, 65], [512, 260, 64]),
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=23)
def test_sdpa_bwd_dsl_sm120_padding_zero_lengths():
    """Zero-length batches: seq_len_kv[b] == 0 (no visible key) and
    seq_len_q[b] == 0 (no query) drain to all-zero gradients."""

    _run_case(
        batch=3,
        s_q=512,
        s_kv=512,
        head_dim=64,
        padding=([512, 0, 33], [0, 512, 48]),
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=27)
def test_sdpa_bwd_dsl_sm120_padding_gqa():
    """Padding mask composes with GQA: the group-reduce sums per-q-head
    partials whose padded rows are zero, so dK/dV padding stays exactly zero."""

    _run_case(
        batch=3,
        h_q=4,
        h_kv=2,
        s_q=512,
        s_kv=512,
        head_dim=64,
        is_causal=True,
        causal_bottom_right=True,
        padding=([512, 300, 17], [512, 128, 65]),
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=29)
def test_sdpa_bwd_dsl_sm120_padding_rejects_cpu_seq_lens():
    """A CPU length tensor must be rejected up front — the kernel would
    otherwise receive a host pointer (illegal access or garbage lengths)."""

    _require_dsl()
    from cudnn.sdpa.bwd.api_dsl import sdpa_bwd_wrapper_dsl_sm120

    batch, heads, s, head_dim, dtype = 2, 4, 256, 64, torch.float16
    q = _bhsd(batch, heads, s, head_dim, dtype)
    k = _bhsd(batch, heads, s, head_dim, dtype)
    v = _bhsd(batch, heads, s, head_dim, dtype)
    do = _bhsd(batch, heads, s, head_dim, dtype)
    o = torch.zeros_like(q)  # never consumed: execute rejects before launching
    stats = torch.zeros(batch, heads, s, 1, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="seq_kv_lens must be on"):
        sdpa_bwd_wrapper_dsl_sm120(q, k, v, o, do, stats, seq_kv_lens=torch.tensor([s, s], dtype=torch.int32))


@pytest.mark.L0
@torch_fork_set_rng(seed=25)
def test_sdpa_bwd_dsl_sm120_padding_kv_only_wrapper():
    """KV-only padding through the direct wrapper: seq_q_lens omitted means
    every batch runs the full S_q."""

    _require_dsl()
    from cudnn.sdpa.bwd.api_dsl import sdpa_bwd_wrapper_dsl_sm120

    batch, heads, s_q, s_kv, head_dim, dtype = 2, 4, 512, 512, 64, torch.float16
    scale = 1.0 / math.sqrt(head_dim)
    kv_lens = [317, 64]
    q = _bhsd(batch, heads, s_q, head_dim, dtype)
    k = _bhsd(batch, heads, s_kv, head_dim, dtype)
    v = _bhsd(batch, heads, s_kv, head_dim, dtype)
    do = _bhsd(batch, heads, s_q, head_dim, dtype)
    o, stats, dq_ref, dk_ref, dv_ref, _ = _ref_bwd(q, k, v, do, scale=scale, padding=([s_q] * batch, kv_lens))
    o = _bhsd(batch, heads, s_q, head_dim, dtype, empty=True).copy_(o)
    out = sdpa_bwd_wrapper_dsl_sm120(q, k, v, o, do, stats, scale_softmax=scale, seq_kv_lens=torch.tensor(kv_lens, dtype=torch.int32, device="cuda"))
    tol = _tolerances(dtype)
    torch.testing.assert_close(out["dq_tensor"].float(), dq_ref.float(), **tol)
    torch.testing.assert_close(out["dk_tensor"].float(), dk_ref.float(), **tol)
    torch.testing.assert_close(out["dv_tensor"].float(), dv_ref.float(), **tol)
    for b, len_kv in enumerate(kv_lens):
        assert out["dk_tensor"][b, :, len_kv:, :].abs().max().item() == 0.0, f"batch {b}: dK padded rows must be exactly zero"
        assert out["dv_tensor"][b, :, len_kv:, :].abs().max().item() == 0.0, f"batch {b}: dV padded rows must be exactly zero"


@pytest.mark.L0
@pytest.mark.parametrize("head_dim", [192, 256])
@pytest.mark.parametrize("mask", ["dense", "causal_tl", "causal_br"])
@torch_fork_set_rng(seed=12)
def test_sdpa_bwd_dsl_sm120_large_d_wrapper(mask: str, head_dim: int):
    """D>128: the graph API's hidden-dim surface stops at 128, so the graph
    build must be rejected and the direct wrapper serves it (same fallback
    pattern as the sq_gt_skv bottom-right case above)."""

    _require_dsl()
    import cudnn

    is_causal = mask != "dense"
    causal_bottom_right = mask == "causal_br"
    try:
        _run_case(head_dim=head_dim, is_causal=is_causal, causal_bottom_right=causal_bottom_right)
        return
    except cudnn.cudnnGraphNotSupportedError as exc:
        assert "hidden_dim" in str(exc), f"unexpected graph rejection: {exc}"

    from cudnn.sdpa.bwd.api_dsl import sdpa_bwd_wrapper_dsl_sm120

    batch, heads, s_q, s_kv, dtype = 2, 4, 512, 512, torch.float16
    scale = 1.0 / math.sqrt(head_dim)
    q = _bhsd(batch, heads, s_q, head_dim, dtype)
    k = _bhsd(batch, heads, s_kv, head_dim, dtype)
    v = _bhsd(batch, heads, s_kv, head_dim, dtype)
    do = _bhsd(batch, heads, s_q, head_dim, dtype)
    o, stats, dq_ref, dk_ref, dv_ref, _ = _ref_bwd(q, k, v, do, scale=scale, is_causal=is_causal, causal_bottom_right=causal_bottom_right)
    o = _bhsd(batch, heads, s_q, head_dim, dtype, empty=True).copy_(o)
    out = sdpa_bwd_wrapper_dsl_sm120(q, k, v, o, do, stats, is_causal=is_causal, causal_bottom_right=causal_bottom_right, scale_softmax=scale)
    tol = _tolerances(dtype)
    torch.testing.assert_close(out["dq_tensor"].float(), dq_ref.float(), **tol)
    torch.testing.assert_close(out["dk_tensor"].float(), dk_ref.float(), **tol)
    torch.testing.assert_close(out["dv_tensor"].float(), dv_ref.float(), **tol)


@pytest.mark.L0
@pytest.mark.parametrize("head_dim", [8, 40, 120])
@pytest.mark.parametrize("is_causal", [False, True], ids=["dense", "causal"])
@torch_fork_set_rng(seed=13)
def test_sdpa_bwd_dsl_sm120_padded_head_dim(head_dim: int, is_causal: bool):
    """Non-native head dims compute on the next supported size via the graph path."""

    _run_case(head_dim=head_dim, is_causal=is_causal)


@pytest.mark.L0
@torch_fork_set_rng(seed=13)
def test_sdpa_bwd_dsl_sm120_head_dim_envelope_gqa_and_layouts():
    """The zero-fill envelope composed with the other native paths."""

    _run_case(head_dim=72, h_q=4, h_kv=2)
    _run_case(head_dim=96, is_causal=True, layout="gapped", grad_layout="gapped")


@pytest.mark.L0
@pytest.mark.parametrize("head_dim", [136, 200])
@torch_fork_set_rng(seed=14)
def test_sdpa_bwd_dsl_sm120_padded_head_dim_wrapper(head_dim: int):
    """Non-bin head dims above the graph API's 128 cap: graph build is
    rejected, the direct wrapper serves them zero-padded."""

    _require_dsl()
    import cudnn

    try:
        _run_case(head_dim=head_dim, is_causal=True)
        return
    except cudnn.cudnnGraphNotSupportedError as exc:
        assert "hidden_dim" in str(exc), f"unexpected graph rejection: {exc}"

    from cudnn.sdpa.bwd.api_dsl import sdpa_bwd_wrapper_dsl_sm120

    batch, heads, s_q, s_kv, dtype = 2, 4, 512, 512, torch.float16
    scale = 1.0 / math.sqrt(head_dim)
    q = _bhsd(batch, heads, s_q, head_dim, dtype)
    k = _bhsd(batch, heads, s_kv, head_dim, dtype)
    v = _bhsd(batch, heads, s_kv, head_dim, dtype)
    do = _bhsd(batch, heads, s_q, head_dim, dtype)
    o, stats, dq_ref, dk_ref, dv_ref, _ = _ref_bwd(q, k, v, do, scale=scale, is_causal=True)
    o = _bhsd(batch, heads, s_q, head_dim, dtype, empty=True).copy_(o)
    out = sdpa_bwd_wrapper_dsl_sm120(q, k, v, o, do, stats, is_causal=True, scale_softmax=scale)
    tol = _tolerances(dtype)
    torch.testing.assert_close(out["dq_tensor"].float(), dq_ref.float(), **tol)
    torch.testing.assert_close(out["dk_tensor"].float(), dk_ref.float(), **tol)
    torch.testing.assert_close(out["dv_tensor"].float(), dv_ref.float(), **tol)


@pytest.mark.L0
@pytest.mark.parametrize(
    ("h_q", "h_kv", "is_causal"),
    [(8, 2, True), (8, 1, False)],
    ids=["gqa_8_2_causal", "mqa_8_1_dense"],
)
@torch_fork_set_rng(seed=17)
def test_sdpa_bwd_dsl_sm120_gqa(h_q: int, h_kv: int, is_causal: bool):
    """GQA / MQA head groups: the grid keeps one CTA per query head; each
    head's dK/dV partial stages through dk_ws/dv_ws and the reduce kernel
    sums the group per KV head."""

    _run_case(h_q=h_q, h_kv=h_kv, s_q=1024, s_kv=1024, head_dim=64, is_causal=is_causal)


@pytest.mark.L0
@torch_fork_set_rng(seed=18)
def test_sdpa_bwd_dsl_sm120_gqa_swa_bf16():
    """GQA composed with causal + sliding window, bf16, d=128."""

    _run_case(h_q=8, h_kv=2, s_q=1024, s_kv=1024, head_dim=128, dtype=torch.bfloat16, is_causal=True, window_size_left=255)


@pytest.mark.L0
@torch_fork_set_rng(seed=19)
def test_sdpa_bwd_dsl_sm120_gqa_causal_br_tails():
    """GQA with bottom-right causal, unequal seq lens, and a partial Q tile."""

    _run_case(h_q=4, h_kv=2, s_q=193, s_kv=257, head_dim=128, is_causal=True, causal_bottom_right=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=20)
def test_sdpa_bwd_dsl_sm120_gqa_padded_head_dim():
    """GQA through the head-dim envelope (D=96 zero-pads to 128)."""

    _run_case(h_q=8, h_kv=2, s_q=512, s_kv=512, head_dim=96, is_causal=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=21)
def test_sdpa_bwd_dsl_sm120_gqa_deterministic_numeric():
    """deterministic + GQA composes: dQ uses the relay, while dK/dV come
    from the fixed-order group reduce (deterministic in both modes); the
    graph routes to the engine with parity."""

    _run_case(h_q=8, h_kv=2, s_q=512, s_kv=512, head_dim=64, is_causal=True, deterministic=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=5)
def test_sdpa_bwd_dsl_sm120_auto_routing():
    """Without an explicit select, the eligible graph auto-routes to the engine."""

    plan_name = _run_case(head_dim=64, is_causal=True, select=False)
    assert plan_name == ENGINE


@pytest.mark.L0
@torch_fork_set_rng(seed=8)
def test_sdpa_bwd_dsl_sm120_native_strided_io():
    """Non-compact io layouts are addressed natively per port (TMA inputs,
    dot O/dO reads, cvt dQ stores, MHA epilogue, GQA reduce)."""

    _run_case(head_dim=64, is_causal=True, layout="bhsd", grad_layout="bhsd")  # permuted
    _run_case(head_dim=128, is_causal=True, layout="gapped", grad_layout="gapped")  # MHA gapped
    _run_case(h_q=4, h_kv=2, head_dim=64, layout="gapped", grad_layout="gapped")  # GQA: strided reduce outputs
    # rect at 128/64: the backend's validate() caps non-packed graphs at hidden_dim 128
    _run_case(head_dim=128, head_dim_v=64, s_q=256, s_kv=256, layout="gapped")


@pytest.mark.L0
@torch_fork_set_rng(seed=8)
def test_sdpa_bwd_dsl_sm120_native_strided_io_mixed_ports():
    """Per-port stride independence: split the O/dO (dot), dK/dV (epilogue,
    reduce) pairs — one port compact, the other gapped."""

    _require_dsl()
    b, h, s, d = 2, 4, 256, 128
    scale = 1.0 / math.sqrt(d)
    q = _bhsd(b, h, s, d, torch.float16)
    k = _bhsd(b, h, s, d, torch.float16, layout="gapped")
    v = _bhsd(b, h, s, d, torch.float16)
    do = _bhsd(b, h, s, d, torch.float16, layout="gapped")
    o_ref, stats, dq_ref, dk_ref, dv_ref, _ = _ref_bwd(q, k, v, do, scale=scale, is_causal=True)
    o = _bhsd(b, h, s, d, torch.float16, empty=True).copy_(o_ref)
    dq, dk, dv, _, _ = _run_bwd_graph(q, k, v, o, do, stats, scale=scale, is_causal=True, grad_layout=("gapped", "bshd", "gapped"))
    tol = _tolerances(torch.float16)
    torch.testing.assert_close(dq.float(), dq_ref.float(), **tol)
    torch.testing.assert_close(dk.float(), dk_ref.float(), **tol)
    torch.testing.assert_close(dv.float(), dv_ref.float(), **tol)

    # GQA: dK compact + dV gapped splits the reduce output pair.
    h_q, h_kv, d = 4, 2, 64
    scale = 1.0 / math.sqrt(d)
    q = _bhsd(b, h_q, s, d, torch.float16)
    k = _bhsd(b, h_kv, s, d, torch.float16)
    v = _bhsd(b, h_kv, s, d, torch.float16, layout="gapped")
    do = _bhsd(b, h_q, s, d, torch.float16)
    o_ref, stats, dq_ref, dk_ref, dv_ref, _ = _ref_bwd(q, k, v, do, scale=scale)
    o = _bhsd(b, h_q, s, d, torch.float16, empty=True).copy_(o_ref)
    dq, dk, dv, _, _ = _run_bwd_graph(q, k, v, o, do, stats, scale=scale, grad_layout=("bshd", "bshd", "gapped"))
    torch.testing.assert_close(dq.float(), dq_ref.float(), **tol)
    torch.testing.assert_close(dk.float(), dk_ref.float(), **tol)
    torch.testing.assert_close(dv.float(), dv_ref.float(), **tol)


@pytest.mark.L0
@torch_fork_set_rng(seed=8)
def test_sdpa_bwd_dsl_sm120_envelope_pad_compact_strides():
    """D=120 over rows of exactly 128, the padded compute width: the gap
    columns must never be read (NaN poison) nor written (sentinel)."""

    _require_dsl()
    b, h, s, d = 2, 4, 128, 120
    scale = 1.0 / math.sqrt(d)
    dtype = torch.float16
    q = _bhsd(b, h, s, d, dtype)
    k = _bhsd(b, h, s, d, dtype)
    v = _bhsd(b, h, s, d, dtype)
    do_base = torch.full((b, s, h, d + 8), float("nan"), dtype=dtype, device="cuda")
    do = do_base.transpose(1, 2)[..., :d]
    do.copy_(torch.randn(b, h, s, d, dtype=dtype, device="cuda"))
    o_ref, stats, dq_ref, dk_ref, dv_ref, _ = _ref_bwd(q, k, v, do, scale=scale, is_causal=True)
    o_base = torch.full_like(do_base, float("nan"))
    o = o_base.transpose(1, 2)[..., :d]
    o.copy_(o_ref)
    grad_bases = [torch.full((b, s, h, d + 8), 7.0, dtype=dtype, device="cuda") for _ in range(3)]
    grads = tuple(base.transpose(1, 2)[..., :d] for base in grad_bases)
    dq, dk, dv, _, _ = _run_bwd_graph(q, k, v, o, do, stats, scale=scale, is_causal=True, grads=grads)
    tol = _tolerances(dtype)
    torch.testing.assert_close(dq.float(), dq_ref.float(), **tol)
    torch.testing.assert_close(dk.float(), dk_ref.float(), **tol)
    torch.testing.assert_close(dv.float(), dv_ref.float(), **tol)
    for base in grad_bases:
        assert (base[..., d:] == 7.0).all(), "a writer touched the declared gap columns"


@pytest.mark.L0
@torch_fork_set_rng(seed=8)
def test_sdpa_bwd_dsl_sm120_strided_io_rejects_unaligned():
    """Non-16-byte-multiple strides decline at build and fall through to the
    backend (Rule 2: never a copy)."""

    _require_dsl()
    batch, h, s, d = 2, 4, 128, 64
    # head stride 68 elems: not a multiple of the 8-element fp16 quantum
    q = torch.randn(batch, s, h, d + 4, dtype=torch.float16, device="cuda").transpose(1, 2)[..., :d]
    k = _bhsd(batch, h, s, d, torch.float16)
    v = _bhsd(batch, h, s, d, torch.float16)
    do = _bhsd(batch, h, s, d, torch.float16)
    scale = 1.0 / math.sqrt(d)
    o, stats, _, _, _, _ = _ref_bwd(q, k, v, do, scale=scale)
    o = _bhsd(batch, h, s, d, torch.float16, empty=True).copy_(o)
    plan_name = _run_bwd_graph(q, k, v, o, do, stats, scale=scale, select=False, build_only=True)[-1]
    assert plan_name != ENGINE


@pytest.mark.L0
@torch_fork_set_rng(seed=8)
def test_sdpa_bwd_dsl_sm120_strided_io_rejects_unaligned_base():
    """A base address that is not 16-byte aligned declines at execute (the
    only time addresses exist) — compact strides do not imply an aligned base."""

    _require_dsl()
    batch, h, s, d = 2, 4, 128, 64
    flat = torch.randn(batch * s * h * d + 4, dtype=torch.float16, device="cuda")
    q = flat[4:].view(batch, s, h, d).transpose(1, 2)  # compact BSHD, base % 16 == 8
    assert q.data_ptr() % 16 == 8 and tuple(q.stride()) == (s * h * d, d, h * d, 1)
    k = _bhsd(batch, h, s, d, torch.float16)
    v = _bhsd(batch, h, s, d, torch.float16)
    do = _bhsd(batch, h, s, d, torch.float16)
    scale = 1.0 / math.sqrt(d)
    o, stats, _, _, _, _ = _ref_bwd(q, k, v, do, scale=scale)
    o = _bhsd(batch, h, s, d, torch.float16, empty=True).copy_(o)
    with pytest.raises((ValueError, RuntimeError), match="16-byte aligned"):
        _run_bwd_graph(q, k, v, o, do, stats, scale=scale)


@pytest.mark.L0
@pytest.mark.parametrize("mask", ["dense", "causal_tl", "causal_br", "swa"])
@torch_fork_set_rng(seed=12)
def test_sdpa_bwd_dsl_sm120_deterministic_numeric(mask: str):
    """use_deterministic_algorithm=True routes to the engine and keeps parity
    (default tolerances) for every mask family the kernel serves."""

    _run_case(
        s_q=384 if mask == "causal_br" else 512,
        s_kv=1024 if mask == "causal_br" else 512,
        head_dim=64,
        is_causal=mask != "dense",
        causal_bottom_right=mask == "causal_br",
        window_size_left=127 if mask == "swa" else None,
        deterministic=True,
    )


def _run_bitwise_case(n_runs: int = 3, padding: tuple[list[int], list[int]] | None = None, **case_kwargs) -> None:
    """Same inputs, ``n_runs`` independent graph runs: outputs must be bitwise equal."""

    batch, heads, dtype = 2, 4, torch.float16
    s_q = case_kwargs.pop("s_q", 1024)
    s_kv = case_kwargs.pop("s_kv", 1024)
    head_dim = case_kwargs.pop("head_dim", 64)
    head_dim_v = case_kwargs.pop("head_dim_v", head_dim)
    h_kv = case_kwargs.pop("h_kv", heads)
    scale = 1.0 / math.sqrt(head_dim)
    q = _bhsd(batch, heads, s_q, head_dim, dtype)
    k = _bhsd(batch, h_kv, s_kv, head_dim, dtype)
    v = _bhsd(batch, h_kv, s_kv, head_dim_v, dtype)
    do = _bhsd(batch, heads, s_q, head_dim_v, dtype)
    o, stats, _, _, _, _ = _ref_bwd(
        q,
        k,
        v,
        do,
        scale=scale,
        is_causal=case_kwargs.get("is_causal", False),
        causal_bottom_right=case_kwargs.get("causal_bottom_right", False),
        window_size_left=case_kwargs.get("window_size_left"),
        padding=padding,
    )
    o = _bhsd(batch, heads, s_q, head_dim_v, dtype, empty=True).copy_(o)
    if padding is not None:
        case_kwargs["seq_q_lens"] = torch.tensor(padding[0], dtype=torch.int32, device="cuda").view(batch, 1, 1, 1)
        case_kwargs["seq_kv_lens"] = torch.tensor(padding[1], dtype=torch.int32, device="cuda").view(batch, 1, 1, 1)
    runs = [_run_bwd_graph(q, k, v, o, do, stats, scale=scale, deterministic=True, **case_kwargs) for _ in range(n_runs)]
    dq0, dk0, dv0, _, _ = runs[0]
    for run_i, (dq, dk, dv, _, _) in enumerate(runs[1:], start=1):
        assert torch.equal(dq, dq0), f"run {run_i}: dQ is not bitwise reproducible"
        assert torch.equal(dk, dk0), f"run {run_i}: dK is not bitwise reproducible"
        assert torch.equal(dv, dv0), f"run {run_i}: dV is not bitwise reproducible"


@pytest.mark.L0
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("mask", ["dense", "causal", "swa"])
@torch_fork_set_rng(seed=13)
def test_sdpa_bwd_dsl_sm120_deterministic_bitwise(head_dim: int, mask: str):
    """Repeated deterministic runs are bitwise identical (dQ relay ordering)."""

    _run_bitwise_case(
        head_dim=head_dim,
        is_causal=mask != "dense",
        window_size_left=127 if mask == "swa" else None,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=14)
def test_sdpa_bwd_dsl_sm120_deterministic_bitwise_tails_knobs():
    """Bitwise reproducibility with partial tails and a non-default tile knob."""

    _run_bitwise_case(s_q=1000, s_kv=999, head_dim=64, is_causal=True, q_tile=128, kv_tile=64)


@pytest.mark.L0
@pytest.mark.parametrize(("mask", "h_kv"), [("dense", 2), ("causal", 1)], ids=["dense_gqa2", "causal_mqa"])
@torch_fork_set_rng(seed=22)
def test_sdpa_bwd_dsl_sm120_gqa_deterministic_bitwise(mask: str, h_kv: int):
    """Repeated deterministic GQA/MQA runs are bitwise identical: the relay
    fixes dQ's fp32 add order and the reduce kernel sums the group's dK/dV
    partials in fixed q-head order."""

    _run_bitwise_case(
        h_kv=h_kv,
        head_dim=64,
        is_causal=mask != "dense",
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=26)
def test_sdpa_bwd_dsl_sm120_deterministic_bitwise_padding():
    """Bitwise reproducibility with a padding mask (per-batch relay trims)."""

    _run_bitwise_case(s_q=1024, s_kv=1024, head_dim=64, is_causal=True, padding=([1024, 300], [1000, 128]))


def _run_wrapper_det_case(head_dim: int, *, s_q: int, s_kv: int, is_causal: bool, window_size_left: int | None, n_runs: int = 1, head_dim_v: int | None = None):
    """Deterministic run(s) through the direct wrapper (D>128 has no graph
    surface); returns (outputs per run, references)."""

    from cudnn.sdpa.bwd.api_dsl import sdpa_bwd_wrapper_dsl_sm120

    batch, heads, dtype = 2, 4, torch.float16
    head_dim_v = head_dim if head_dim_v is None else head_dim_v
    scale = 1.0 / math.sqrt(head_dim)
    q = _bhsd(batch, heads, s_q, head_dim, dtype)
    k = _bhsd(batch, heads, s_kv, head_dim, dtype)
    v = _bhsd(batch, heads, s_kv, head_dim_v, dtype)
    do = _bhsd(batch, heads, s_q, head_dim_v, dtype)
    o, stats, dq_ref, dk_ref, dv_ref, _ = _ref_bwd(q, k, v, do, scale=scale, is_causal=is_causal, window_size_left=window_size_left)
    o = _bhsd(batch, heads, s_q, head_dim_v, dtype, empty=True).copy_(o)
    runs = [
        sdpa_bwd_wrapper_dsl_sm120(q, k, v, o, do, stats, is_causal=is_causal, window_size_left=window_size_left, deterministic=True, scale_softmax=scale)
        for _ in range(n_runs)
    ]
    return runs, (dq_ref, dk_ref, dv_ref)


@pytest.mark.L0
@torch_fork_set_rng(seed=15)
def test_sdpa_bwd_dsl_sm120_deterministic_large_d_numeric():
    """Deterministic relay on the single-Q-buffer branch (D=256 -> q32,
    Q_STAGES == 1): causal + sliding window + non-tile-multiple tails, vs ref."""

    _require_dsl()
    runs, (dq_ref, dk_ref, dv_ref) = _run_wrapper_det_case(256, s_q=1000, s_kv=1000, is_causal=True, window_size_left=127)
    tol = _tolerances(torch.float16)
    out = runs[0]
    torch.testing.assert_close(out["dq_tensor"].float(), dq_ref.float(), **tol)
    torch.testing.assert_close(out["dk_tensor"].float(), dk_ref.float(), **tol)
    torch.testing.assert_close(out["dv_tensor"].float(), dv_ref.float(), **tol)


@pytest.mark.L0
@pytest.mark.parametrize(
    ("head_dim", "mask"),
    [(192, "causal"), (256, "causal"), (256, "swa")],
)
@torch_fork_set_rng(seed=16)
def test_sdpa_bwd_dsl_sm120_deterministic_large_d_bitwise(head_dim: int, mask: str):
    """Repeated deterministic runs are bitwise identical on the q32 large-D path."""

    _require_dsl()
    runs, _ = _run_wrapper_det_case(
        head_dim,
        s_q=1024,
        s_kv=1024,
        is_causal=mask != "dense",
        window_size_left=127 if mask == "swa" else None,
        n_runs=3,
    )
    first = runs[0]
    for run_i, out in enumerate(runs[1:], start=1):
        for grad in ("dq_tensor", "dk_tensor", "dv_tensor"):
            assert torch.equal(out[grad], first[grad]), f"run {run_i}: {grad} is not bitwise reproducible (D={head_dim}, {mask})"


@pytest.mark.L0
@pytest.mark.parametrize(
    ("head_dim", "q_tile", "kv_tile"),
    [
        (64, 128, 64),  # sweep-tuned non-default entry (CONFIG hit)
        (64, 64, 64),  # not in CONFIG (largest_warp_partition fallback)
    ],
)
@torch_fork_set_rng(seed=7)
def test_sdpa_bwd_dsl_sm120_tile_knobs(head_dim: int, q_tile: int, kv_tile: int):
    """Explicit macro-tile knobs override the per-head-dim CONFIG default.

    One case per warp-layout source: the sweep-tuned CONFIG entry and the
    largest_warp_partition fallback. (SMEM-infeasible combinations — e.g.
    any d128 non-default — correctly fail the strict-select build in the
    kernel constructor instead.)
    """

    _run_case(head_dim=head_dim, is_causal=True, q_tile=q_tile, kv_tile=kv_tile)


# ---------------------------------------------------------------------------
# Rectangular head dims (D_QK > D_V): MLA training shapes.
# ---------------------------------------------------------------------------


def _run_mla_wrapper_case(
    *,
    head_dim: int,
    head_dim_v: int,
    h_q: int = 4,
    h_kv: int | None = None,
    s_q: int = 512,
    s_kv: int = 512,
    dtype: torch.dtype = torch.float16,
    is_causal: bool = False,
    causal_bottom_right: bool = False,
) -> None:
    """Rectangular-dims case through the direct wrapper, vs the torch ref."""

    from cudnn.sdpa.bwd.api_dsl import sdpa_bwd_wrapper_dsl_sm120

    batch = 2
    h_kv = h_q if h_kv is None else h_kv
    scale = 1.0 / math.sqrt(head_dim)
    q = _bhsd(batch, h_q, s_q, head_dim, dtype)
    k = _bhsd(batch, h_kv, s_kv, head_dim, dtype)
    v = _bhsd(batch, h_kv, s_kv, head_dim_v, dtype)
    do = _bhsd(batch, h_q, s_q, head_dim_v, dtype)
    o, stats, dq_ref, dk_ref, dv_ref, _ = _ref_bwd(q, k, v, do, scale=scale, is_causal=is_causal, causal_bottom_right=causal_bottom_right)
    o = _bhsd(batch, h_q, s_q, head_dim_v, dtype, empty=True).copy_(o)
    out = sdpa_bwd_wrapper_dsl_sm120(q, k, v, o, do, stats, is_causal=is_causal, causal_bottom_right=causal_bottom_right, scale_softmax=scale)
    tol = _tolerances(dtype)
    torch.testing.assert_close(out["dq_tensor"].float(), dq_ref.float(), **tol)
    torch.testing.assert_close(out["dk_tensor"].float(), dk_ref.float(), **tol)
    torch.testing.assert_close(out["dv_tensor"].float(), dv_ref.float(), **tol)


@pytest.mark.L0
@pytest.mark.parametrize("mask", ["dense", "causal_tl", "causal_br"])
@torch_fork_set_rng(seed=50)
def test_sdpa_bwd_dsl_sm120_mla_192_128(mask: str):
    """DeepSeek-V3 / Kimi-K2.6 MLA training shape: D_QK=192 (128 nope + 64
    rope), D_V=128. D_QK > 128 has no graph surface, so the graph build must
    be rejected and the direct wrapper serves it (same fallback pattern as
    the large-D tests)."""

    _require_dsl()
    import cudnn

    is_causal = mask != "dense"
    causal_bottom_right = mask == "causal_br"
    try:
        _run_case(head_dim=192, head_dim_v=128, is_causal=is_causal, causal_bottom_right=causal_bottom_right)
        return
    except cudnn.cudnnGraphNotSupportedError as exc:
        assert "hidden_dim" in str(exc), f"unexpected graph rejection: {exc}"

    _run_mla_wrapper_case(head_dim=192, head_dim_v=128, is_causal=is_causal, causal_bottom_right=causal_bottom_right)


@pytest.mark.L0
@torch_fork_set_rng(seed=51)
def test_sdpa_bwd_dsl_sm120_mla_192_128_gqa_bf16():
    """MLA dims + GQA + bf16 through the direct wrapper: the split-index
    group-reduce sums the D_QK-wide dK partials and D_V-wide dV partials."""

    _require_dsl()
    _run_mla_wrapper_case(head_dim=192, head_dim_v=128, h_q=8, h_kv=2, dtype=torch.bfloat16, is_causal=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=52)
def test_sdpa_bwd_dsl_sm120_mla_192_128_tails():
    """MLA dims with non-tile-multiple sequence tails (partial Q/KV tiles)."""

    _require_dsl()
    _run_mla_wrapper_case(head_dim=192, head_dim_v=128, s_q=193, s_kv=257, is_causal=True)


@pytest.mark.L0
@pytest.mark.parametrize(
    ("head_dim", "head_dim_v"),
    [(128, 64), (96, 64), (120, 72), (96, 8)],
    ids=["native_128_64", "padded_96_64", "square_bins_120_72", "enveloped_96_8"],
)
@pytest.mark.parametrize("is_causal", [False, True], ids=["dense", "causal"])
@torch_fork_set_rng(seed=53)
def test_sdpa_bwd_dsl_sm120_rect_head_dims_graph(head_dim: int, head_dim_v: int, is_causal: bool):
    """Rectangular D_QK > D_V through the graph path: native 128/64, plus
    enveloped variants. 96/64 computes on the rectangular 128/64 kernel
    sizes; 96/8 lands on 128/32 (the page drops to 32); 120/72 pads both
    sides into the SQUARE 128/128 kernel — user-rectangular but
    kernel-square, covering that envelope combination too."""

    _run_case(head_dim=head_dim, head_dim_v=head_dim_v, is_causal=is_causal)


@pytest.mark.L0
@torch_fork_set_rng(seed=54)
def test_sdpa_bwd_dsl_sm120_rect_head_dims_gqa_graph():
    """GQA + rectangular dims via the graph path: the group-reduce kernel's
    split index space (dK vectors then dV vectors) covers both partials."""

    _run_case(h_q=8, h_kv=2, head_dim=128, head_dim_v=64, is_causal=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=55)
def test_sdpa_bwd_dsl_sm120_rect_head_dims_padding_graph():
    """Rectangular dims compose with the padding mask (per-batch seq lens)."""

    _run_case(head_dim=128, head_dim_v=64, is_causal=True, padding=([512, 300], [512, 128]))


@pytest.mark.L0
@torch_fork_set_rng(seed=56)
def test_sdpa_bwd_dsl_sm120_mla_deterministic_bitwise():
    """Repeated deterministic MLA (192/128) runs are bitwise identical."""

    _require_dsl()
    runs, (dq_ref, dk_ref, dv_ref) = _run_wrapper_det_case(
        192,
        head_dim_v=128,
        s_q=1024,
        s_kv=1024,
        is_causal=True,
        window_size_left=None,
        n_runs=3,
    )
    tol = _tolerances(torch.float16)
    out = runs[0]
    torch.testing.assert_close(out["dq_tensor"].float(), dq_ref.float(), **tol)
    torch.testing.assert_close(out["dk_tensor"].float(), dk_ref.float(), **tol)
    torch.testing.assert_close(out["dv_tensor"].float(), dv_ref.float(), **tol)
    for run_i, out in enumerate(runs[1:], start=1):
        for grad in ("dq_tensor", "dk_tensor", "dv_tensor"):
            assert torch.equal(out[grad], runs[0][grad]), f"run {run_i}: {grad} is not bitwise reproducible (MLA 192/128)"


@pytest.mark.L0
@torch_fork_set_rng(seed=57)
def test_sdpa_bwd_dsl_sm120_rect_rejects_dv_gt_dqk():
    """D_V > D_QK is out of the dqk_ge_dv envelope: the adapter rejects it."""

    _require_dsl()
    from cudnn.sdpa.bwd.api_dsl import sdpa_bwd_wrapper_dsl_sm120

    batch, heads, s, dtype = 2, 4, 256, torch.float16
    q = _bhsd(batch, heads, s, 64, dtype)
    k = _bhsd(batch, heads, s, 64, dtype)
    v = _bhsd(batch, heads, s, 128, dtype)
    do = _bhsd(batch, heads, s, 128, dtype)
    o = torch.zeros_like(do)  # never consumed: check_support rejects first
    stats = torch.zeros(batch, heads, s, 1, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="D_QK >= D_V"):
        sdpa_bwd_wrapper_dsl_sm120(q, k, v, o, do, stats)


@pytest.mark.L0
@torch_fork_set_rng(seed=58)
def test_sdpa_bwd_dsl_sm120_strided_stats():
    """Non-contiguous stats are addressed natively via baked strides;
    contiguous stats keep the original variant."""

    _require_dsl()
    import cudnn

    if cudnn.backend_version() < 92600:
        # The FE gates non-packed Stats off older backends at validate().
        pytest.skip("strided Stats requires cuDNN >= 9.26")
    _run_case(head_dim=64, is_causal=True, stats_layout="strided")
    _run_case(h_q=8, h_kv=2, head_dim=128, stats_layout="strided")  # GQA
    _run_case(head_dim=128, head_dim_v=64, is_causal=True, stats_layout="strided")  # rectangular
    _run_case(head_dim=64, stats_layout="strided", padding=([512, 300], [512, 128]))  # -inf padded rows
    _run_case(head_dim=64, sink=True, stats_layout="strided", padding=([512, 300], [512, 128]))  # dSink's own LSE reads


# ---------------------------------------------------------------------------
# Deterministic two-kernel split (dS workspace + dQ GEMM). Auto-routed for
# the 128/128 and 192/128 head-dim pairs; the relay tests below pin the other
# flavor by patching the route pick.
# ---------------------------------------------------------------------------


def _force_relay(monkeypatch):
    """Pin the relay route on two-kernel-eligible shapes (adapter + the
    workspace mirror above)."""
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm120

    monkeypatch.setattr(SdpaBwdDslSm120, "_pick_det_2k", lambda self: False)
    monkeypatch.setitem(globals(), "_expect_det_2k", lambda **kw: False)


@pytest.mark.L0
@pytest.mark.parametrize("mask", ["dense", "causal", "causal_br"])
@torch_fork_set_rng(seed=30)
def test_sdpa_bwd_dsl_sm120_det_2kernel_numeric(mask: str):
    """Two-kernel deterministic route: numeric parity per mask."""

    _run_case(
        s_q=384 if mask == "causal_br" else 512,
        s_kv=1024 if mask == "causal_br" else 512,
        head_dim=128,
        is_causal=mask != "dense",
        causal_bottom_right=mask == "causal_br",
        deterministic=True,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=31)
def test_sdpa_bwd_dsl_sm120_det_2kernel_tails():
    """Non-tile-multiple S_q/S_kv: padded ws kv columns carry masked zeros,
    rows past S_q are TMA zero-fill."""

    _run_case(s_q=1000, s_kv=999, head_dim=128, is_causal=True, deterministic=True)
    _run_case(s_q=193, s_kv=161, head_dim=128, is_causal=True, deterministic=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=32)
def test_sdpa_bwd_dsl_sm120_det_2kernel_right_band():
    """Causal right-band widening (the dQ GEMM's kv bound follows the
    widened diagonal)."""

    _run_case(s_q=256, s_kv=256, head_dim=128, is_causal=True, window_size_right=32, deterministic=True)


@pytest.mark.L0
@pytest.mark.parametrize(("mask", "h_kv"), [("dense", 2), ("causal", 1)], ids=["dense_gqa2", "causal_mqa"])
@torch_fork_set_rng(seed=33)
def test_sdpa_bwd_dsl_sm120_det_2kernel_gqa(mask: str, h_kv: int):
    """GQA/MQA: dQ GEMM reads the shared KV head; dK/dV keep the
    fixed-order group reduce."""

    _run_case(h_kv=h_kv, head_dim=128, is_causal=mask != "dense", deterministic=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=35)
def test_sdpa_bwd_dsl_sm120_det_2kernel_sink():
    """Sink token: dS is sink-agnostic; dSink keeps its fixed-order reduce."""

    _run_case(s_q=256, s_kv=256, head_dim=128, sink=True, deterministic=True)


@pytest.mark.L0
@pytest.mark.parametrize("mask", ["dense", "causal"])
@torch_fork_set_rng(seed=36)
def test_sdpa_bwd_dsl_sm120_det_relay(mask: str, monkeypatch):
    """Relay route pinned on a two-kernel-eligible d128 shape: numeric +
    bitwise (d64 always routes the relay and is covered by the
    deterministic tests above)."""

    _force_relay(monkeypatch)
    _run_case(head_dim=128, is_causal=mask != "dense", deterministic=True)
    _run_bitwise_case(head_dim=128, is_causal=mask != "dense")


@pytest.mark.L0
@torch_fork_set_rng(seed=37)
def test_sdpa_bwd_dsl_sm120_det_relay_features():
    """Sliding-window and padding masks stay deterministic on the relay
    (the split serves dense/causal only)."""

    _run_case(head_dim=128, is_causal=True, window_size_left=127, deterministic=True)
    _run_case(head_dim=64, is_causal=True, padding=([500, 300], [400, 128]), deterministic=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=38)
def test_sdpa_bwd_dsl_sm120_det_2kernel_mla():
    """MLA 192/128 (the dQ GEMM only touches the QK side): numeric via the
    direct wrapper (D>128 has no graph surface), with tails in the second
    case."""

    _require_dsl()
    tol = _tolerances(torch.float16)
    for s_q, s_kv in ((512, 512), (257, 193)):
        runs, (dq_ref, dk_ref, dv_ref) = _run_wrapper_det_case(192, head_dim_v=128, s_q=s_q, s_kv=s_kv, is_causal=True, window_size_left=None)
        out = runs[0]
        torch.testing.assert_close(out["dq_tensor"].float(), dq_ref.float(), **tol)
        torch.testing.assert_close(out["dk_tensor"].float(), dk_ref.float(), **tol)
        torch.testing.assert_close(out["dv_tensor"].float(), dv_ref.float(), **tol)


@pytest.mark.L0
@torch_fork_set_rng(seed=39)
def test_sdpa_bwd_dsl_sm120_det_2kernel_mla_bitwise():
    """Repeated MLA (192/128) runs are bitwise identical."""

    _require_dsl()
    runs, _ = _run_wrapper_det_case(192, head_dim_v=128, s_q=1024, s_kv=1024, is_causal=True, window_size_left=None, n_runs=3)
    for run_i, out in enumerate(runs[1:], start=1):
        for grad in ("dq_tensor", "dk_tensor", "dv_tensor"):
            assert torch.equal(out[grad], runs[0][grad]), f"run {run_i}: {grad} is not bitwise reproducible (det_2kernel MLA)"


@pytest.mark.L0
@torch_fork_set_rng(seed=40)
def test_sdpa_bwd_dsl_sm120_det_2kernel_d256():
    """256/256 (auto pair): numeric with tails + bitwise via the wrapper."""

    _require_dsl()
    tol = _tolerances(torch.float16)
    for s_q, s_kv in ((512, 512), (257, 193)):
        runs, (dq_ref, dk_ref, dv_ref) = _run_wrapper_det_case(256, head_dim_v=256, s_q=s_q, s_kv=s_kv, is_causal=True, window_size_left=None, n_runs=3)
        out = runs[0]
        torch.testing.assert_close(out["dq_tensor"].float(), dq_ref.float(), **tol)
        torch.testing.assert_close(out["dk_tensor"].float(), dk_ref.float(), **tol)
        torch.testing.assert_close(out["dv_tensor"].float(), dv_ref.float(), **tol)
        for run_i, o2 in enumerate(runs[1:], start=1):
            for grad in ("dq_tensor", "dk_tensor", "dv_tensor"):
                assert torch.equal(o2[grad], runs[0][grad]), f"run {run_i}: {grad} is not bitwise reproducible (det_2kernel d256)"


def _prepared_bwd_case(dtype=torch.bfloat16, route="gqa", wide=None, wide_span=False):
    """Capture the public graph call, retaining its actual buffers and pinned route."""
    from types import SimpleNamespace
    from unittest.mock import patch
    import cudnn

    torch.manual_seed(903)
    b, h, s = (5 if wide_span else 2), 4, 128
    d = 64 if route == "relay" else 128
    hk = h if route == "mha" else 2
    tensors = {name: _bhsd(b, heads, s, d, dtype) for name, heads in (("q", h), ("k", hk), ("v", hk), ("do", h))}
    o, stats, dq, dk, dv, _ = _ref_bwd(**{name: tensors[name] for name in ("q", "k", "v", "do")}, scale=d**-0.5, is_causal=True)
    tensors.update(o=_bhsd(b, h, s, d, dtype, empty=True).copy_(o), stats=stats)
    expected = dict(zip(("dq", "dk", "dv"), (dq, dk, dv)))
    tensors.update({name: torch.empty_like(tensors[src]).fill_(float("nan")) for name, src in (("dq", "q"), ("dk", "k"), ("dv", "v"))})
    if wide is not None:
        src = tensors[wide]
        strides = ((2**30 if wide_span else 2**32) + src.stride(0), *src.stride()[1:])
        elements = 1 + sum((n - 1) * st for n, st in zip(src.shape, strides))
        guard = 2**31 if wide_span else 0
        free, _ = torch.cuda.mem_get_info()
        if (elements + guard) * src.element_size() + 512 * 2**20 > free:
            pytest.skip("physical Int64 probe needs room for one wide buffer")
        backing = torch.empty(elements + guard, device="cuda", dtype=src.dtype)
        # Signed Int32 product wrap can point before the view's origin. Keep
        # those addresses in an allocated guard and seed the addressed rows.
        if wide_span:
            import ctypes

            for batch in range(b):
                decoy = guard + ctypes.c_int32(batch * strides[0]).value
                backing.as_strided((1, *src.shape[1:]), src.stride(), decoy).fill_(float("nan"))
        else:
            backing.as_strided(src.shape, src.stride()).fill_(float("nan"))
        tensors[wide] = backing.as_strided(src.shape, strides, guard).copy_(src)
    calls = []
    original = cudnn.pygraph.execute

    def record(graph, *args, **kwargs):
        calls.append((graph, args, kwargs))
        return original(graph, *args, **kwargs)

    from contextlib import ExitStack
    import sys

    with ExitStack() as stack:
        stack.enter_context(patch.object(cudnn.pygraph, "execute", record))
        if route == "det2k" and wide in ("k", "dq"):
            # Declared noncompact K/dQ selects relay at D128 too.
            stack.enter_context(patch.object(sys.modules[__name__], "_expect_det_2k", return_value=False))
        result = _run_bwd_graph(
            *(tensors[name] for name in ("q", "k", "v", "o", "do", "stats")),
            scale=d**-0.5,
            is_causal=True,
            deterministic=route in ("relay", "det2k"),
            grads=tuple(tensors[name] for name in ("dq", "dk", "dv")),
        )
    assert result[-1] == ENGINE
    graph, (pack, workspace), kwargs = calls[-1]
    assert not kwargs
    refs = {name: next(ref for ref, buf in pack.items() if buf is tensor) for name, tensor in tensors.items()}
    return SimpleNamespace(graph=graph, pack=pack, workspace=workspace, tensors=tensors, refs=refs, expected=expected, scale=d**-0.5, dtype=dtype)


def _check_prepared_bwd(case, tensors=None, expected=None):
    tensors = case.tensors if tensors is None else tensors
    expected = case.expected if expected is None else expected
    for name in ("dq", "dk", "dv"):
        torch.testing.assert_close(tensors[name], expected[name], **_tolerances(case.dtype))


@requires_dsl
class TestPreparedSm120Bwd:
    @pytest.mark.L0
    @pytest.mark.parametrize("dbias_dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("hk", [4, 2])
    def test_auxiliary_rebind_and_explicit_stream_capture(self, dbias_dtype, hk):
        """Bias reset, conversion and sink reduction belong to the handle stream."""
        from unittest.mock import patch
        import cudnn

        torch.manual_seed(1903)
        b, h, s, d = 2, 4, 128, 64
        inputs = {name: _bhsd(b, heads, s, d, torch.bfloat16) for name, heads in (("q", h), ("k", hk), ("v", hk), ("do", h))}
        bias = torch.randn(1, h, s, s, dtype=torch.float32, device="cuda")
        sink = torch.randn(1, h, 1, 1, dtype=torch.float32, device="cuda")
        padding = ([128, 73], [99, 117])
        lengths = [torch.tensor(values, dtype=torch.int32, device="cuda") for values in padding]
        o, stats, *_ = _ref_bwd(**inputs, scale=d**-0.5, is_causal=True, bias=bias, sink_token=sink, padding=padding)
        o = _bhsd(b, h, s, d, torch.bfloat16, empty=True).copy_(o)
        dbias = torch.empty_like(bias, dtype=dbias_dtype)
        calls = []
        original = cudnn.pygraph.execute

        def record(graph, *args, **kwargs):
            calls.append((graph, args))
            return original(graph, *args, **kwargs)

        with patch.object(cudnn.pygraph, "execute", record):
            outputs = _run_bwd_graph(
                *(inputs[name] for name in ("q", "k", "v")),
                o,
                inputs["do"],
                stats,
                scale=d**-0.5,
                is_causal=True,
                deterministic=False,
                bias_gpu=bias,
                dbias_gpu=dbias,
                sink_gpu=sink,
                seq_q_lens=lengths[0],
                seq_kv_lens=lengths[1],
            )
        graph, (pack, workspace) = calls[-1]
        rebound = {ref: buf.clone() for ref, buf in pack.items()}
        by_pointer = {buf.data_ptr(): rebound[ref] for ref, buf in pack.items()}
        current = lambda buf: by_pointer[buf.data_ptr()]
        current(bias).mul_(0.5)
        current(sink).add_(0.25)
        padding = ([101, 64], [127, 83])
        for buf, values in zip(lengths, padding):
            current(buf).copy_(torch.tensor(values, dtype=torch.int32, device="cuda"))
        ref_o, ref_stats, dq, dk, dv, aux = _ref_bwd(
            *(current(inputs[name]) for name in ("q", "k", "v", "do")),
            scale=d**-0.5,
            is_causal=True,
            bias=current(bias),
            sink_token=current(sink),
            padding=padding,
        )
        current(o).copy_(ref_o)
        current(stats).copy_(ref_stats)
        out = [current(tensor) for tensor in (*outputs[:4], dbias)]
        expected = [dq, dk, dv, aux.dsink, aux.dbias]
        workspace = torch.empty_like(workspace).fill_(0xBD)
        stream, other = torch.cuda.Stream(), torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        other.wait_stream(torch.cuda.current_stream())
        handle = cudnn.create_handle()
        cudnn.set_stream(handle, stream.cuda_stream)
        capture = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.stream(other):
                graph.execute(rebound, workspace, handle=handle)
            torch.cuda.current_stream().wait_stream(stream)
            for actual, ref in zip(out, expected):
                torch.testing.assert_close(actual.float(), ref.float(), **_tolerances(torch.bfloat16))
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.graph(capture, stream=stream):
                with torch.cuda.stream(other):
                    graph.execute(rebound, workspace, handle=handle)
            for tensor in out:
                tensor.fill_(float("nan"))
            workspace.fill_(0xBD)
            capture.replay()
            for actual, ref in zip(out, expected):
                torch.testing.assert_close(actual.float(), ref.float(), **_tolerances(torch.bfloat16))
        finally:
            capture.reset()
            cudnn.destroy_handle(handle)

    @pytest.mark.L0
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("route", ["mha", "gqa", "relay", "det2k"])
    def test_rebind_and_replay(self, dtype, route):
        case = _prepared_bwd_case(dtype, route)
        _check_prepared_bwd(case)
        rebound = {name: tensor.clone() for name, tensor in case.tensors.items()}
        rebound["q"].mul_(0.75)
        rebound["do"].mul_(1.25)
        o, stats, dq, dk, dv, _ = _ref_bwd(*(rebound[name] for name in ("q", "k", "v", "do")), scale=case.scale, is_causal=True)
        rebound["o"].copy_(o)
        rebound["stats"].copy_(stats)
        expected = dict(zip(("dq", "dk", "dv"), (dq, dk, dv)))
        pack = {case.refs[name]: tensor for name, tensor in rebound.items()}
        workspace = torch.empty_like(case.workspace).fill_(0xBD)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            case.graph.execute(pack, workspace)
        torch.cuda.current_stream().wait_stream(stream)
        _check_prepared_bwd(case, rebound, expected)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                case.graph.execute(pack, workspace)
            for name in ("dq", "dk", "dv"):
                rebound[name].fill_(float("nan"))
            workspace.fill_(0xBD)
            graph.replay()
            _check_prepared_bwd(case, rebound, expected)
        finally:
            graph.reset()

    @pytest.mark.L0
    @pytest.mark.parametrize("route", ["mha", "gqa", "relay", "det2k"])
    def test_execute_has_no_tensor_views_or_jit(self, route, monkeypatch):
        import cutlass.cute as cute
        from cudnn.sdpa.fwd.api_dsl import WorkspaceCarver

        case = _prepared_bwd_case(route=route)
        case.graph.execute(case.pack, case.workspace)

        def forbidden(*args, **kwargs):
            raise AssertionError("prepared backward rebuilt tensor operands, allocated or compiled")

        with monkeypatch.context() as patcher:
            for name in ("view", "reshape", "as_strided", "transpose"):
                patcher.setattr(torch.Tensor, name, forbidden)
            for name in ("empty", "empty_like", "zeros", "zeros_like"):
                patcher.setattr(torch, name, forbidden)
            patcher.setattr(WorkspaceCarver, "__init__", forbidden)
            patcher.setattr(cute, "compile", forbidden)
            torch.cuda.set_sync_debug_mode("error")
            try:
                case.graph.execute(case.pack, case.workspace)
            finally:
                torch.cuda.set_sync_debug_mode("default")
        _check_prepared_bwd(case)

    @pytest.mark.L1
    @pytest.mark.gpu_exclusive
    @pytest.mark.parametrize("role", ["q", "k", "v", "o", "do", "stats", "dq", "dk", "dv"])
    @pytest.mark.parametrize("route", ["mha", "gqa", "relay", "det2k"])
    @pytest.mark.parametrize("wide_span", [False, True], ids=["wide_stride", "wide_product"])
    def test_physical_batch_stride_above_int32(self, role, route, wide_span):
        # The two-kernel route requires compact K/dQ; those declarations select
        # relay, which must still honor the same pointer/stride contract.
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        case = _prepared_bwd_case(route=route, wide=role, wide_span=wide_span)
        assert (case.tensors[role].shape[0] - 1) * case.tensors[role].stride(0) > 2**32
        _check_prepared_bwd(case)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                case.graph.execute(case.pack, case.workspace)
            for name in ("dq", "dk", "dv"):
                case.tensors[name].fill_(float("nan"))
            case.workspace.fill_(0xBD)
            graph.replay()
            _check_prepared_bwd(case)
        finally:
            graph.reset()

    @pytest.mark.L0
    @pytest.mark.parametrize("route", ["mha", "gqa", "relay", "det2k"])
    def test_standalone_and_graph_bind_same_operands(self, route):
        from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm120

        case = _prepared_bwd_case(route=route)
        api = SdpaBwdDslSm120(
            **{"sample_" + name: value for name, value in case.tensors.items()},
            is_causal=True,
            deterministic=route in ("relay", "det2k"),
            scale_softmax=case.scale,
        )
        api.check_support()
        api.compile()
        workspace = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda").fill_(0xBD)
        for name in ("dq", "dk", "dv"):
            case.tensors[name].fill_(float("nan"))
        api.execute(**{name + "_tensor": value for name, value in case.tensors.items()}, workspace=workspace)
        _check_prepared_bwd(case)

    @pytest.mark.L0
    @pytest.mark.parametrize("role", ["q", "k", "v", "o", "do", "dq", "dk", "dv"])
    def test_standalone_rejects_changed_layout_before_launch(self, role):
        from dataclasses import replace
        from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm120

        case = _prepared_bwd_case()
        api = SdpaBwdDslSm120(**{"sample_" + name: value for name, value in case.tensors.items()}, is_causal=True, scale_softmax=case.scale)
        api.check_support()
        api.compile()
        launches = []
        api._prepared = replace(api._prepared, fn=lambda *args: launches.append(args))
        args = {name + "_tensor": value for name, value in case.tensors.items()}
        args[role + "_tensor"] = case.tensors[role].contiguous()
        assert args[role + "_tensor"].stride() != case.tensors[role].stride()
        with pytest.raises(ValueError, match="runtime geometry"):
            api.execute(**args, workspace=case.workspace)
        assert not launches

    @pytest.mark.L0
    @pytest.mark.parametrize("role", ["q", "k", "v", "o", "do", "stats", "dq", "dk", "dv"])
    @pytest.mark.parametrize("ordered", [False, True])
    def test_graph_accepts_strided_raw_storage(self, role, ordered):
        case = _prepared_bwd_case()
        tensor = case.tensors[role]
        backing = torch.empty(tensor.numel() + 2, device="cuda", dtype=tensor.dtype)
        declared = backing.as_strided(tensor.shape, tensor.stride()).copy_(tensor)
        raw = backing[::2]
        assert not raw.is_contiguous()
        case.pack[case.refs[role]] = raw
        case.tensors[role] = declared
        for name in ("dq", "dk", "dv"):
            case.tensors[name].fill_(float("nan"))
        if ordered:
            items = list(reversed(list(case.pack.items())))
            case.graph.execute([buffer for _, buffer in items], case.workspace, tensor_uids=[ref.get_uid() for ref, _ in items])
        else:
            case.graph.execute(case.pack, case.workspace)
        _check_prepared_bwd(case)

    @pytest.mark.L0
    @pytest.mark.parametrize("role", ["q", "stats", "dq"])
    @pytest.mark.parametrize("ordered", [False, True])
    def test_graph_fixed_plan_override_validation(self, role, ordered, monkeypatch):
        from dataclasses import replace

        case = _prepared_bwd_case()
        ref = case.refs[role]
        kwargs = dict(override_uids=[ref.get_uid()], override_shapes=[list(ref.get_dim())], override_strides=[list(ref.get_stride())])
        pack = case.pack
        if ordered:
            items = list(reversed(list(pack.items())))
            kwargs["tensor_uids"] = [tensor.get_uid() for tensor, _ in items]
            pack = [buffer for _, buffer in items]
        case.graph.execute(pack, case.workspace, **kwargs)
        _check_prepared_bwd(case)
        plan = case.graph._compiled_plans[case.graph._plan_index]
        launches = []
        monkeypatch.setattr(plan._prepared, "spec", replace(plan._prepared.spec, fn=lambda *args: launches.append(args)))
        kwargs["override_shapes"][0][2] //= 2
        with pytest.raises(ValueError, match="runtime geometry"):
            case.graph.execute(pack, case.workspace, **kwargs)
        assert not launches

    @pytest.mark.L0
    def test_standalone_accepts_flat_contiguous_stats(self):
        from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm120

        case = _prepared_bwd_case()
        api = SdpaBwdDslSm120(**{"sample_" + name: value for name, value in case.tensors.items()}, is_causal=True, scale_softmax=case.scale)
        api.check_support()
        api.compile()
        args = {name + "_tensor": value for name, value in case.tensors.items()}
        args["stats_tensor"] = case.tensors["stats"].view(-1)
        for name in ("dq", "dk", "dv"):
            case.tensors[name].fill_(float("nan"))
        api.execute(**args, workspace=case.workspace)
        _check_prepared_bwd(case)

    @pytest.mark.L0
    @pytest.mark.parametrize("failure", ["stats_numel", "stats_contiguity"])
    def test_standalone_retains_stats_validation(self, failure):
        from dataclasses import replace
        from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm120

        case = _prepared_bwd_case()
        api = SdpaBwdDslSm120(**{"sample_" + name: value for name, value in case.tensors.items()}, is_causal=True, scale_softmax=case.scale)
        api.check_support()
        api.compile()
        launches = []
        api._prepared = replace(api._prepared, fn=lambda *args: launches.append(args))
        stats = case.tensors["stats"]
        bad = torch.empty(stats.numel() + 1, device="cuda", dtype=stats.dtype)
        if failure == "stats_contiguity":
            bad = torch.empty((*stats.shape[:-1], 2), device="cuda", dtype=stats.dtype)[..., :1]
        args = {name + "_tensor": value for name, value in case.tensors.items()}
        args["stats_tensor"] = bad
        with pytest.raises(ValueError, match="stats_tensor"):
            api.execute(**args, workspace=case.workspace)
        assert not launches

    @pytest.mark.L0
    @pytest.mark.parametrize("failure", ["missing", "dtype", "device", "span", "alignment", "extra", "workspace_alias", "geometry", "strided_lengths"])
    def test_validate_before_any_stage(self, failure, monkeypatch):
        from dataclasses import replace
        from cudnn.sdpa.bwd import prepared
        from cudnn.sdpa.fwd.prepared import facts_of_tensor

        observed = []
        execute = prepared.execute

        def observe(spec, *args, **kwargs):
            observed.append(spec)
            return execute(spec, *args, **kwargs)

        with monkeypatch.context() as patcher:
            patcher.setattr(prepared, "execute", observe)
            case = _prepared_bwd_case()
        assert observed, "the graph did not take prepared backward"
        launches = []
        spec = replace(observed[-1], fn=lambda *args: launches.append(args))
        facts = {name: facts_of_tensor(value) for name, value in case.tensors.items()}
        workspace = case.workspace.data_ptr()
        geometry = None
        if failure == "missing":
            facts.pop("q")
        elif failure == "dtype":
            facts["q"] = facts["q"]._replace(dtype="float32")
        elif failure == "device":
            facts["q"] = facts["q"]._replace(device=(1, 0))
        elif failure == "span":
            facts["q"] = facts["q"]._replace(span=1)
        elif failure == "alignment":
            facts["q"] = facts["q"]._replace(ptr=facts["q"].ptr + 2)
        elif failure == "extra":
            facts["sink"] = facts["stats"]
        elif failure == "workspace_alias":
            workspace = facts["q"].ptr
        elif failure == "geometry":
            geometry = tuple((op.shape, op.strides) if op is not None else None for op in spec.operands)
            facts["q"] = facts["q"]._replace(shape=(1, *facts["q"].shape[1:]))
        elif failure == "strided_lengths":
            operands = list(spec.operands)
            operands[9] = prepared.Operand("int32", (2,), (1,), 2, 4, 4)
            spec = replace(spec, operands=tuple(operands))
            lengths = torch.tensor([128, 0, 64, 0], device="cuda", dtype=torch.int32)[::2]
            facts["seq_q"] = facts_of_tensor(lengths)
        with pytest.raises(ValueError):
            execute(spec, facts, workspace, torch.cuda.current_stream().cuda_stream, geometry=geometry)
        assert not launches


@requires_dsl
@pytest.mark.L0
@pytest.mark.parametrize("route", ["mha", "gqa", "relay", "det2k", "aux_fp32", "aux_io"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_prepared_backward_artifact_reloads_in_fresh_process(route, dtype, tmp_path):
    from prepared_bwd_cache_utils import check_backward_artifact_reload

    check_backward_artifact_reload("sm120", route, dtype, tmp_path)
