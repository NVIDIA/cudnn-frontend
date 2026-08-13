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
    """Logical BHSD over compact BSHD storage, or BHSD-contiguous for layout="bhsd"."""

    factory = torch.empty if empty else torch.randn
    if layout == "bhsd":
        return factory(batch, heads, sequence, head_dim, dtype=dtype, device="cuda")
    return factory(batch, sequence, heads, head_dim, dtype=dtype, device="cuda").transpose(1, 2)


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
        diag_align=diag_align,
        right_bound=right_bound,
        left_bound=left_bound,
        padding=padding,
        sink_token=sink_token,
        torch_type=q.dtype,
    )
    dq, dk, dv, _, dsink = compute_ref_backward(
        q,
        k,
        v,
        o_ref,
        do,
        attn_scale=scale,
        diag_align=diag_align,
        right_bound=right_bound,
        left_bound=left_bound,
        padding=padding,
        sink_token=sink_token,
        torch_type=q.dtype,
    )
    return o_ref.to(q.dtype), stats_ref.contiguous(), dq.to(q.dtype), dk.to(q.dtype), dv.to(q.dtype), dsink


def _expected_workspace_bytes(
    batch: int,
    h_q: int,
    s_q: int,
    head_dim: int,
    staged: tuple[torch.Tensor, ...] = (),
    h_kv: int | None = None,
    s_kv: int | None = None,
    io_itemsize: int = 2,
) -> int:
    from cudnn.sdpa.bwd.config_sm120 import padded_head_dim
    from cudnn.sdpa.fwd.api_dsl import ws_align

    d_pad = padded_head_dim(head_dim)
    sq_r = -(-s_q // 128) * 128
    h_kv = h_q if h_kv is None else h_kv
    s_kv_eff = s_kv if s_kv is not None else s_q
    dq_sem = batch * h_q * (-(-s_q // 32))  # int32 relay counters (min q-tile 32)
    # dk_ws/dv_ws GQA partials buffers in the io dtype (none carved for MHA, where the main kernel writes dk/dv directly)
    dkv_ws_half = 0 if h_kv == h_q else batch * s_kv_eff * h_q * d_pad * io_itemsize
    base = ws_align(batch * h_q * sq_r * 4) + ws_align(batch * sq_r * h_q * d_pad * 4) + ws_align(dq_sem * 4) + 2 * ws_align(dkv_ws_half)
    # Padded-width staging copy per non-BSHD-compact operand (all when D pads).
    staging = sum(ws_align(t.numel() // head_dim * d_pad * t.element_size()) for t in staged if d_pad != head_dim or not t.transpose(1, 2).is_contiguous())
    return base + staging


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
    grad_layout: str = "bshd",
    seq_q_lens: torch.Tensor | None = None,
    seq_kv_lens: torch.Tensor | None = None,
    sink_gpu: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, "torch.Tensor | None", str]:
    """Build and execute the SM120 FROST backward graph; returns (dq, dk, dv, dsink, plan_name)."""

    _require_dsl()
    import cudnn

    dtype = q_gpu.dtype
    io_dtype = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    batch, h_q, _, head_dim = q_gpu.shape
    _, h_kv, _, _ = k_gpu.shape
    dq_gpu = _bhsd(batch, h_q, q_gpu.shape[2], head_dim, dtype, empty=True, layout=grad_layout)
    dk_gpu = _bhsd(batch, h_kv, k_gpu.shape[2], head_dim, dtype, empty=True, layout=grad_layout)
    dv_gpu = _bhsd(batch, h_kv, v_gpu.shape[2], head_dim, dtype, empty=True, layout=grad_layout)

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

    workspace_size = graph.get_workspace_size()
    if plan_name == ENGINE:
        staged = (q_gpu, k_gpu, v_gpu, o_gpu, do_gpu, dq_gpu, dk_gpu, dv_gpu)
        assert workspace_size == _expected_workspace_bytes(
            batch, h_q, q_gpu.shape[2], head_dim, staged, h_kv=h_kv, s_kv=k_gpu.shape[2], io_itemsize=q_gpu.element_size()
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
) -> str:
    h_kv = h_q if h_kv is None else h_kv
    scale = 1.0 / math.sqrt(head_dim)
    q = _bhsd(batch, h_q, s_q, head_dim, dtype, layout=layout)
    k = _bhsd(batch, h_kv, s_kv, head_dim, dtype, layout=layout)
    v = _bhsd(batch, h_kv, s_kv, head_dim, dtype, layout=layout)
    do = _bhsd(batch, h_q, s_q, head_dim, dtype, layout=layout)
    sink_gpu = torch.randn(1, h_q, 1, 1, dtype=torch.float32, device="cuda") if sink else None
    o, stats, dq_ref, dk_ref, dv_ref, dsink_ref = _ref_bwd(
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
    )
    o = _bhsd(batch, h_q, s_q, head_dim, dtype, empty=True, layout=layout).copy_(o)
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
    )
    tol = _tolerances(dtype)
    torch.testing.assert_close(dq.float(), dq_ref.float(), **tol)
    torch.testing.assert_close(dk.float(), dk_ref.float(), **tol)
    torch.testing.assert_close(dv.float(), dv_ref.float(), **tol)
    if sink:
        torch.testing.assert_close(dsink.float(), dsink_ref.float(), **tol)
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
@pytest.mark.parametrize("head_dim", [40, 72, 120])
@pytest.mark.parametrize("is_causal", [False, True], ids=["dense", "causal"])
@torch_fork_set_rng(seed=13)
def test_sdpa_bwd_dsl_sm120_padded_head_dim(head_dim: int, is_causal: bool):
    """Non-bin head dims run zero-padded in the next bin via the graph path."""

    _run_case(head_dim=head_dim, is_causal=is_causal)


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
def test_sdpa_bwd_dsl_sm120_dense_flex_bhsd_contiguous():
    """BHSD-contiguous operands (dense_flex): served via workspace-carved compact
    staging copies — a gather for the inputs, a scatter-back for dQ/dK/dV."""

    _run_case(head_dim=64, is_causal=True, layout="bhsd", grad_layout="bhsd")


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
    h_kv = case_kwargs.pop("h_kv", heads)
    scale = 1.0 / math.sqrt(head_dim)
    q = _bhsd(batch, heads, s_q, head_dim, dtype)
    k = _bhsd(batch, h_kv, s_kv, head_dim, dtype)
    v = _bhsd(batch, h_kv, s_kv, head_dim, dtype)
    do = _bhsd(batch, heads, s_q, head_dim, dtype)
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
    o = _bhsd(batch, heads, s_q, head_dim, dtype, empty=True).copy_(o)
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


def _run_wrapper_det_case(head_dim: int, *, s_q: int, s_kv: int, is_causal: bool, window_size_left: int | None, n_runs: int = 1):
    """Deterministic run(s) through the direct wrapper (D>128 has no graph
    surface); returns (outputs per run, references)."""

    from cudnn.sdpa.bwd.api_dsl import sdpa_bwd_wrapper_dsl_sm120

    batch, heads, dtype = 2, 4, torch.float16
    scale = 1.0 / math.sqrt(head_dim)
    q = _bhsd(batch, heads, s_q, head_dim, dtype)
    k = _bhsd(batch, heads, s_kv, head_dim, dtype)
    v = _bhsd(batch, heads, s_kv, head_dim, dtype)
    do = _bhsd(batch, heads, s_q, head_dim, dtype)
    o, stats, dq_ref, dk_ref, dv_ref, _ = _ref_bwd(q, k, v, do, scale=scale, is_causal=is_causal, window_size_left=window_size_left)
    o = _bhsd(batch, heads, s_q, head_dim, dtype, empty=True).copy_(o)
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
