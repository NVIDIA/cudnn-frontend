# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""End-to-end tests for the FROST SM120 DSL SDPA-backward engine against a torch reference."""

from __future__ import annotations

import math

import pytest
import torch

from test_utils import torch_fork_set_rng

ENGINE = "sdpa_bwd_sm120"


def _is_sm120() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return (major, minor) in {(12, 0), (12, 1)}


def _dsl_deps_available() -> bool:
    try:
        import cutlass  # noqa: F401
    except ImportError:
        return False
    return True


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
    if not _dsl_deps_available():
        pytest.skip("cutlass/dsl not installed")


def _select_engine(graph, name):
    """Pin the ranked entry named ``name`` (graph.plans holds the backend's
    plans and the python engines' in one list). A pin is strict: check_support /
    build_plans raise if that engine declines the graph."""
    names = [graph.get_plan_name_at_index(i) for i in range(len(graph.plans))]
    assert name in names, f"engine {name!r} did not claim this graph; plans={names}"
    graph.select_plan(names.index(name))
    return graph


def _bhsd(batch: int, heads: int, sequence: int, head_dim: int, dtype: torch.dtype, empty: bool = False) -> torch.Tensor:
    """Return logical BHSD backed by compact BSHD physical storage."""

    factory = torch.empty if empty else torch.randn
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
):
    """Reference via the canonical refs (sdpa/fp16_ref.py)."""

    import cudnn
    from sdpa.fp16_ref import compute_ref, compute_ref_backward

    diag_align = cudnn.diagonal_alignment.BOTTOM_RIGHT if causal_bottom_right else cudnn.diagonal_alignment.TOP_LEFT
    right_bound = 0 if is_causal else None
    # The refs take the cuDNN window LENGTH; window_size_left is the offset W = L - 1.
    left_bound = None if window_size_left is None else window_size_left + 1
    o_ref, stats_ref, _, _ = compute_ref(q, k, v, attn_scale=scale, diag_align=diag_align, right_bound=right_bound, left_bound=left_bound, torch_type=q.dtype)
    dq, dk, dv, _, _ = compute_ref_backward(
        q, k, v, o_ref, do, attn_scale=scale, diag_align=diag_align, right_bound=right_bound, left_bound=left_bound, torch_type=q.dtype
    )
    return o_ref.to(q.dtype), stats_ref.contiguous(), dq.to(q.dtype), dk.to(q.dtype), dv.to(q.dtype)


def _expected_workspace_bytes(batch: int, heads: int, s_q: int, head_dim: int) -> int:
    from cudnn.sdpa.fwd.api_dsl import ws_align

    sq_r = -(-s_q // 128) * 128
    return ws_align(batch * heads * sq_r * 4) + ws_align(batch * sq_r * heads * head_dim * 4)


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
    select: bool = True,
    q_tile: int | None = None,
    kv_tile: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """Build and execute the SM120 FROST backward graph; returns (dq, dk, dv, plan_name)."""

    _require_dsl()
    import cudnn

    dtype = q_gpu.dtype
    io_dtype = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    batch, h_q, _, head_dim = q_gpu.shape
    _, h_kv, _, _ = k_gpu.shape
    dq_gpu = _bhsd(batch, h_q, q_gpu.shape[2], head_dim, dtype, empty=True)
    dk_gpu = _bhsd(batch, h_kv, k_gpu.shape[2], head_dim, dtype, empty=True)
    dv_gpu = _bhsd(batch, h_kv, v_gpu.shape[2], head_dim, dtype, empty=True)

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
    if causal_bottom_right:
        bwd_kwargs["use_causal_mask_bottom_right"] = True
    elif is_causal:
        bwd_kwargs["use_causal_mask"] = True
    if window_size_left is not None:
        bwd_kwargs["sliding_window_length"] = window_size_left + 1

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
        assert workspace_size == _expected_workspace_bytes(batch, h_q, q_gpu.shape[2], head_dim)
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
    graph.execute(variant_pack, workspace)
    torch.cuda.synchronize()
    return dq_gpu, dk_gpu, dv_gpu, plan_name


def _tolerances(dtype: torch.dtype) -> dict:
    return {"atol": 2e-2 if dtype == torch.float16 else 5e-2, "rtol": 5e-2}


def _run_case(
    *,
    batch: int = 2,
    heads: int = 4,
    s_q: int = 512,
    s_kv: int = 512,
    head_dim: int = 64,
    dtype: torch.dtype = torch.float16,
    is_causal: bool = False,
    causal_bottom_right: bool = False,
    window_size_left: int | None = None,
    select: bool = True,
    q_tile: int | None = None,
    kv_tile: int | None = None,
) -> str:
    scale = 1.0 / math.sqrt(head_dim)
    q = _bhsd(batch, heads, s_q, head_dim, dtype)
    k = _bhsd(batch, heads, s_kv, head_dim, dtype)
    v = _bhsd(batch, heads, s_kv, head_dim, dtype)
    do = _bhsd(batch, heads, s_q, head_dim, dtype)
    o, stats, dq_ref, dk_ref, dv_ref = _ref_bwd(
        q, k, v, do, scale=scale, is_causal=is_causal, causal_bottom_right=causal_bottom_right, window_size_left=window_size_left
    )
    o = _bhsd(batch, heads, s_q, head_dim, dtype, empty=True).copy_(o)
    dq, dk, dv, plan_name = _run_bwd_graph(
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
        select=select,
        q_tile=q_tile,
        kv_tile=kv_tile,
    )
    tol = _tolerances(dtype)
    torch.testing.assert_close(dq.float(), dq_ref.float(), **tol)
    torch.testing.assert_close(dk.float(), dk_ref.float(), **tol)
    torch.testing.assert_close(dv.float(), dv_ref.float(), **tol)
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
    o, stats, dq_ref, dk_ref, dv_ref = _ref_bwd(q, k, v, do, scale=scale, is_causal=True, causal_bottom_right=True)
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
@torch_fork_set_rng(seed=5)
def test_sdpa_bwd_dsl_sm120_auto_routing():
    """Without an explicit select, the eligible graph auto-routes to the engine."""

    plan_name = _run_case(head_dim=64, is_causal=True, select=False)
    assert plan_name == ENGINE


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
