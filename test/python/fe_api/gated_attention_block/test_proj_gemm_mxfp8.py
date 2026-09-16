# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage (1) on the FROST BLOCK-SCALE (MXFP8) GEMM: ``build_proj_gemm(block_scale=True)``.

This writes no kernel either: the driver in ``kernels/proj_gemm.py`` builds cuDNN's
two-dequant graph (e4m3 codes + per-32-block E8M0 scale factors in ``F8_128x4`` order
through ``block_scale_dequantize`` into the matmul, exactly as
``test/python/gemm/frost/test_block_scale_matmul.py`` builds it), pins the ``frost_gemm``
plan or takes the JIT-only route when the backend declines the graph, and binds the
PADDED scale-factor blobs at execute.  What is under test is that wiring:

* the output equals the FAKE-QUANT reference -- the dequantized e4m3 operands' exact
  fp32 product rounded once to bf16 -- at the small shape, the tail shape and the block's
  own ``qkv_gate_proj`` shape (``M=4096, K=4096, N=17408``);
* with DISTINCTIVE per-block scale factors (E8M0 bytes 120..134, i.e. 2^-7..2^7) so a
  scale read from the wrong block, or not read at all, shows as a power-of-two error and
  not as noise -- the Step-0 landmine of PR-B section 2.1 (SF rings past 256 KiB);
* non-zero + sentinel (the >256 KiB descriptor hazard reads as EXACTLY ZERO, never as an
  error), two launches bit-identical, the K=32 and K=64 MMA forms (D15), and the resolved
  ``plan.tile_config_name`` / ``plan.mma_tile_k_bytes`` / ``plan.route`` recorded;
* the auto-tile JIT path (``N % 256 != 0``: no forced 256-wide config, the engine's own pick).

Two device gates: ``requires_frost_gemm`` (any sm100-family part) for the driver, and
``requires_rubin`` for every ``mma_tile_k_bytes=64`` build -- the 64-byte MMA-instruction K
is SM 10.7 silicon (``kernel_registry.MMA_INST_K64_ARCH_RANGES``), a typed
``NotImplementedError`` elsewhere, so those cases SKIP on Blackwell rather than error.

The scale-factor layout the driver binds is pinned on the CPU against the byte formula
of PR-B section 2.3 (F8_128x4: 128 rows x 4 blocks = 512-B atoms, row-major; byte
``(r % 32) * 16 + (r // 32) * 4 + c`` inside an atom), with
``test/python/sdpa/mxfp8_quant.py`` as the oracle -- the same helpers the SDPA MXFP8
suites use.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

pytestmark = pytest.mark.L0

from cudnn.gated_attention_block.kernels.proj_gemm import ProjGemmPlan, build_proj_gemm, run_proj_gemm, sf_blob_bytes, sf_padded_dims  # noqa: E402

_FP8 = getattr(torch, "float8_e4m3fn", None)
_E8M0 = getattr(torch, "float8_e8m0fnu", None)
_SENTINEL = 1.5e30
_BLOCK = 32
_BLOCK_SHAPE = (4096, 4096, 17408)  # the 397B qkv_gate_proj: M tokens x K=d_model x N=n_qkvg
_FORCED_K32 = "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma"


def _frost_gemm_unavailable() -> str | None:
    if not torch.cuda.is_available():
        return "no CUDA device"
    from cudnn.gemm.frost.kernel_registry import PIPELINE_ARCH_RANGES

    cc = torch.cuda.get_device_capability()
    arch = cc[0] * 10 + cc[1]
    spans = PIPELINE_ARCH_RANGES.get("sm100", ())
    if not any(lo <= arch < hi for lo, hi in spans):
        return f"the sm100 GEMM template family does not run on sm_{arch}"
    return None


requires_frost_gemm = pytest.mark.skipif(_frost_gemm_unavailable() is not None, reason=_frost_gemm_unavailable() or "")
requires_fp8 = pytest.mark.skipif(_FP8 is None or _E8M0 is None, reason="this torch has no float8_e4m3fn / float8_e8m0fnu")
_SM107 = (10, 7)
requires_rubin = pytest.mark.skipif(
    not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability()) != _SM107,
    reason="mma_tile_k_bytes=64 (the 64-byte MMA-instruction K) is SM 10.7 silicon -- kernel_registry.MMA_INST_K64_ARCH_RANGES",
)


def mxfp8_quant():
    """``test/python/sdpa/mxfp8_quant.py`` -- importable as ``sdpa.mxfp8_quant`` when pytest runs
    from ``test/python`` (how the SDPA suites spell it); a standalone driver gets the path added."""
    try:
        from sdpa import mxfp8_quant as mq
    except ImportError:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
        from sdpa import mxfp8_quant as mq
    return mq


# ---------------------------------------------------------------------------
# Scale-factor blob helpers (shared with frost_dev/probe_block_scale_gemm_sm107.py)
# ---------------------------------------------------------------------------


def blocked_sf(e: torch.Tensor) -> torch.Tensor:
    """The PADDED F8_128x4 blob of a logical ``[rows, K/32]`` uint8 scale matrix, flat.

    Rows are padded to whole 128-row atoms and blocks to whole 4-block words with
    ``0x00`` (E8M0 2^-127 -- the pad rows' data is TMA zero-filled anyway), then the
    F8_128x4 reorder of ``sdpa.mxfp8_quant.swizzle_sf_rowwise`` (``to_blocked`` in the
    FROST GEMM suite is byte-identical).  ``numel == sf_blob_bytes(rows, K)``.
    """
    mq = mxfp8_quant()
    assert e.dtype == torch.uint8
    rows, cols = e.shape
    rows_pad, cols_pad = -(-rows // 128) * 128, -(-cols // 4) * 4
    pad = torch.zeros(rows_pad, cols_pad, dtype=torch.uint8, device=e.device)
    pad[:rows, :cols] = e
    return mq.swizzle_sf_rowwise(pad).flatten()


def f8_128x4_byte(r: int, c: int, cols_pad: int) -> int:
    """Byte offset of scale ``(r, c)`` in an F8_128x4 blob of ``cols_pad`` blocks per row
    (PR-B section 2.3): atoms (128 rows x 4 blocks = 512 B) row-major, inside an atom
    ``(r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4``."""
    return ((r // 128) * (cols_pad // 4) + (c // 4)) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (c % 4)


def distinctive_case(m: int, k: int, n: int, *, seed: int = 0, lo: int = 120, hi: int = 134, device="cuda") -> dict:
    """e4m3 codes + DISTINCTIVE per-block E8M0 scale factors, their padded blobs and the
    fake-quant fp32 reference ``(a8 * 2^(e_a-127)) @ (w8 * 2^(e_w-127))^T``.

    The codes are ``randn * 0.5`` cast to e4m3 (the FROST GEMM suite's own generator); the
    exponents are uniform in ``[lo, hi]`` (2^-7 .. 2^7 by default) per 32-block, so every
    block carries its own power of two and a scale fetched from the wrong block -- or a
    version-0 SF descriptor wrapped past 256 KiB -- moves that block's contribution by a
    power of two rather than by noise.  The reference is exact fp32 (TF32 off): every
    e4m3 x e4m3 x 2^n product is exactly representable, so the only differences left are
    fp32 reassociation and the single bf16 rounding of the output.
    """
    mq = mxfp8_quant()
    torch.manual_seed(seed)
    sf_k = k // _BLOCK
    a8 = (torch.randn(m, k, device=device) * 0.5).to(_FP8)
    w8 = (torch.randn(n, k, device=device) * 0.5).to(_FP8)
    e_a = torch.randint(lo, hi + 1, (m, sf_k), dtype=torch.uint8, device=device)
    e_w = torch.randint(lo, hi + 1, (n, sf_k), dtype=torch.uint8, device=device)
    a_s = a8.float() * mq.e8m0_to_float(e_a).repeat_interleave(_BLOCK, 1)
    w_s = w8.float() * mq.e8m0_to_float(e_w).repeat_interleave(_BLOCK, 1)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False  # an fp32 reference, not a TF32 one (frost-gotchas)
    try:
        ref32 = a_s @ w_s.t()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return dict(m=m, k=k, n=n, a8=a8, w8=w8, e_a=e_a, e_w=e_w, sf_a=blocked_sf(e_a), sf_w=blocked_sf(e_w), ref32=ref32)


def launch(plan: ProjGemmPlan, case: dict, *, sf_a=None, sf_w=None) -> torch.Tensor:
    """Sentinel-fill a bf16 output, run the plan on the case (or on the given blobs), synchronize."""
    out = torch.full((case["m"], case["n"]), _SENTINEL, device="cuda", dtype=torch.bfloat16)
    ws = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda")
    run_proj_gemm(plan, case["a8"], case["w8"], out, ws, sf_a=case["sf_a"] if sf_a is None else sf_a, sf_w=case["sf_w"] if sf_w is None else sf_w)
    torch.cuda.synchronize()
    return out


def check_bf16_of_fp32(out: torch.Tensor, ref32: torch.Tensor, label: str) -> dict:
    """The kernel accumulates the SAME exactly-representable products in fp32 and rounds once
    to bf16, so it must match the fp32 reference to bf16 rounding -- one ulp of the output's
    max magnitude (2^-8 relative) plus fp32 reassociation noise.  Also: no sentinel survivor,
    all finite, and NOT silently zero (the >256 KiB descriptor hazard).  Returns the stats."""
    o = out.float()
    surv = int((o == _SENTINEL).sum().item())
    nonzero = (o != 0).float().mean().item()
    err = (o - ref32).abs()
    scale = ref32.abs().max().item()
    cos = torch.nn.functional.cosine_similarity(o.flatten(), ref32.flatten(), dim=0).item()
    stats = dict(sentinel_survivors=surv, nonzero_frac=nonzero, max_abs_err=err.max().item(), ref_max=scale, cos=cos, finite=bool(torch.isfinite(o).all()))
    assert surv == 0, f"{label}: {surv} sentinel survivors (never-written cells) -- {stats}"
    assert stats["finite"], f"{label}: non-finite output -- {stats}"
    assert nonzero > 0.99, f"{label}: only {100 * nonzero:.1f}% of the output is non-zero -- suspect a wrapped >256 KiB SMEM descriptor -- {stats}"
    assert cos > 0.9999, f"{label}: cos {cos} -- {stats}"
    assert err.max().item() <= 2.0 * scale * 2.0**-8 + 1e-3 * scale, f"{label}: max|err| {err.max().item():.3e} vs ref max {scale:.3e} -- {stats}"
    return stats


def describe(plan: ProjGemmPlan) -> str:
    return f"tile_config_name={plan.tile_config_name!r} mma_tile_k_bytes={plan.mma_tile_k_bytes} route={plan.route!r}"


def bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """Spacing of bf16 at magnitude |x| (8 significand bits: ``2^(floor(log2|x|) - 7)``), built from the
    fp32 exponent bits -- NOT ``torch.exp2`` / ``torch.pow``, which torch JIT-compiles through NVRTC and
    which fails on cc 10.7 ("invalid value for --gpu-architecture")."""
    a = x.float().abs().clamp_min(2.0**-126)
    _, e = torch.frexp(a)  # a = m * 2^e, m in [0.5, 1)  ->  floor(log2 a) = e - 1
    k = e.to(torch.int32) - 1 - 7
    return ((k + 127) << 23).to(torch.int32).view(torch.float32)  # exactly 2^k


# fp32-accumulation slack between two orderings of the same K=4096 dot product, as a fraction of the
# tensor's max magnitude: ~128 K-steps x ulp32(partial sums ~ ref_max) ~ 2^-17 x ref_max; 2^-14 is a 8x margin.
_ACCUM_SLACK = 2.0**-14


def compare_two_roundings(a: torch.Tensor, b: torch.Tensor, ref32: torch.Tensor) -> dict:
    """Two bf16 outputs of the SAME fp32-accumulated GEMM in two accumulation orders (K=32 vs K=64
    MMA forms, graph route vs JIT route): how far apart may they legitimately be?

    Per cell: one bf16 ulp of the REFERENCE magnitude (a reassociation nudge of ~0.3 on a sum of
    144896.145 flips a midpoint on the 1024-spaced bf16 grid -- measured 2026-09-15 on the dev node,
    K32 144384 / K64 145408 / ref64 144896.145) PLUS an fp32-accumulation slack of ``_ACCUM_SLACK``
    x max|ref| for the CANCELLATION cells (ref ~ +-0.01 from sum|products| ~ 1.4e5, where the two
    orderings land on different sub-unit numbers and any RELATIVE unit blows up).  Returns the stats;
    ``within`` is the assertion.
    """
    d = (a.float() - b.float()).abs()
    tol = bf16_ulp(ref32) + _ACCUM_SLACK * ref32.abs().max()
    return dict(
        bit_equal_frac=(a == b).float().mean().item(),
        n_off=int((d > 0).sum().item()),
        max_abs=d.max().item(),
        n_beyond_one_ulp=int((d > bf16_ulp(ref32)).sum().item()),
        within=bool((d <= tol).all().item()),
    )


# ---------------------------------------------------------------------------
# No GPU: the driver's typed declines, the byte formula, the binding checks
# ---------------------------------------------------------------------------


def test_block_scale_needs_e4m3_codes():
    with pytest.raises(ValueError, match="e4m3"):
        build_proj_gemm(m=256, k=512, n=512, dtype=torch.bfloat16, label="mx", block_scale=True, pin_frost=False)


@requires_fp8
def test_block_scale_refuses_alpha_and_ragged_k():
    """The E8M0 dequant IS the descale (exact, in the MMA), so alpha has nothing to multiply;
    a K that is not a whole number of 32-blocks has no scale factor for its tail."""
    with pytest.raises(ValueError, match="alpha must be False"):
        build_proj_gemm(m=256, k=512, n=512, dtype=_FP8, label="mx", block_scale=True, alpha=True, pin_frost=False)
    with pytest.raises(ValueError, match="K % 32"):
        build_proj_gemm(m=256, k=528, n=512, dtype=_FP8, label="mx", block_scale=True, pin_frost=False)


@requires_fp8
def test_mma_tile_k_bytes_vocabulary():
    """D15: the MMA K width is an explicit block-scale A/B knob -- 32 or 64, nothing else, and
    not on the dense path (which keeps the engine's preferred width)."""
    with pytest.raises(ValueError, match="mma_tile_k_bytes must be None, 32 or 64"):
        build_proj_gemm(m=256, k=512, n=512, dtype=_FP8, label="mx", block_scale=True, mma_tile_k_bytes=48, pin_frost=False)
    with pytest.raises(ValueError, match="block-scale knob"):
        build_proj_gemm(m=256, k=512, n=512, dtype=_FP8, label="dense", mma_tile_k_bytes=64, pin_frost=False)


@pytest.mark.parametrize(
    "rows, k, expect",
    [(256, 512, 256 * 16), (1000, 4096, 1024 * 128), (4096, 4096, 4096 * 128), (17408, 4096, 17408 * 128), (100, 64, 128 * 4)],
)
def test_sf_blob_bytes_is_the_padded_f8_128x4_size(rows, k, expect):
    """``ceil(rows/128)*128 * ceil(K/32/4)*4`` -- whole 512-B atoms; what the JIT binding requires."""
    assert sf_blob_bytes(rows, k) == expect
    rows_pad, sf_k_pad = sf_padded_dims(rows, k)
    assert rows_pad * sf_k_pad == expect and rows_pad % 128 == 0 and sf_k_pad % 4 == 0
    with pytest.raises(ValueError, match="multiple of the 32"):
        sf_blob_bytes(rows, 40)


@pytest.mark.parametrize("rows, k, expect", [(1000, 4096, (1024, 128)), (256, 544, (256, 20)), (392, 512, (512, 16)), (4096, 4096, (4096, 128))])
def test_sf_padded_dims_are_what_the_backend_requires_for_f8_128x4(rows, k, expect):
    """The graph declares SFA / SFB at ``(1, ceil(rows/128)*128, ceil(K/32/4)*4)``.  Measured on the
    dev node (2026-09-15): the backend's block_scale_dequantize finalize REJECTS the unpadded
    ``[1, M, K/32]`` with ``CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH`` ("must be (1, 1024, 128) but
    found (1, 1000, 128)"; K=544: "(1, 256, 20)" vs 17; N=392: "(1, 512, 16)" vs 392) and admits
    the padded one -- so the tail shape is SERVED (route graph+jit), not declined.  The
    same pair sizes the blob and the bound view."""
    assert sf_padded_dims(rows, k) == expect
    with pytest.raises(ValueError, match="rows must be > 0"):
        sf_padded_dims(0, k)


@requires_fp8
def test_blocked_sf_matches_the_byte_formula():
    """The blob the driver binds (``swizzle_sf_rowwise`` of the padded logical matrix) puts scale
    ``(r, c)`` at the PR-B section 2.3 byte: atoms row-major, ``(r%32)*16 + ((r%128)//32)*4 + c%4``
    inside.  2000 random probes over four shapes, CPU."""
    g = torch.Generator().manual_seed(1)
    for rows, k in ((256, 512), (1000, 4096), (130, 1024), (384, 64)):
        sf_k = k // _BLOCK
        e = torch.randint(0, 255, (rows, sf_k), dtype=torch.uint8, generator=g)
        blob = blocked_sf(e.clone()).cpu()
        assert blob.numel() == sf_blob_bytes(rows, k)
        cols_pad = -(-sf_k // 4) * 4
        rr = torch.randint(0, rows, (500,), generator=g)
        cc = torch.randint(0, sf_k, (500,), generator=g)
        for r, c in zip(rr.tolist(), cc.tolist()):
            assert int(blob[f8_128x4_byte(r, c, cols_pad)]) == int(e[r, c]), (rows, k, r, c)
        # pad rows / pad blocks are 0x00
        if rows % 128:
            r_pad = rows  # the first padded row
            assert all(int(blob[f8_128x4_byte(r_pad, c, cols_pad)]) == 0 for c in range(sf_k))


@requires_fp8
def test_runner_refuses_missing_or_misshaped_scale_factors():
    """A block-scale plan without its blobs, a blob of the wrong byte count / dtype, or a dense
    plan handed blobs: every one a typed ValueError on the host, before any binding."""
    m, k, n = 256, 512, 512
    bs = ProjGemmPlan(graph=None, a=None, b=None, c=None, m=m, k=k, n=n, label="mx", dtype=_FP8, out_dtype=torch.bfloat16, block_scale=True)
    dense = ProjGemmPlan(graph=None, a=None, b=None, c=None, m=m, k=k, n=n, label="dense", dtype=_FP8, out_dtype=torch.bfloat16)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a8, w8 = torch.zeros(m, k, dtype=_FP8, device=dev), torch.zeros(n, k, dtype=_FP8, device=dev)
    out, ws = torch.zeros(m, n, dtype=torch.bfloat16, device=dev), torch.zeros(1, dtype=torch.uint8, device=dev)
    good_a = torch.zeros(sf_blob_bytes(m, k), dtype=torch.uint8, device=dev)
    good_w = torch.zeros(sf_blob_bytes(n, k), dtype=torch.uint8, device=dev)
    with pytest.raises(ValueError, match="sf_a="):
        run_proj_gemm(bs, a8, w8, out, ws)
    with pytest.raises(ValueError, match="sf_w="):
        run_proj_gemm(bs, a8, w8, out, ws, sf_a=good_a)
    with pytest.raises(ValueError, match="F8_128x4 blob"):
        run_proj_gemm(bs, a8, w8, out, ws, sf_a=good_a[:-512], sf_w=good_w)  # one atom short
    with pytest.raises(ValueError, match="F8_128x4 blob"):
        run_proj_gemm(bs, a8, w8, out, ws, sf_a=good_a, sf_w=torch.zeros(n * (k // 32), dtype=torch.uint8, device=dev)[: sf_blob_bytes(n, k) - 4])
    with pytest.raises(ValueError, match="uint8 or float8_e8m0fnu"):
        run_proj_gemm(bs, a8, w8, out, ws, sf_a=good_a.to(torch.int8), sf_w=good_w)
    with pytest.raises(ValueError, match="no block-scale"):
        run_proj_gemm(dense, a8, w8, out, ws, sf_a=good_a, sf_w=good_w)
    if dev == "cpu":
        with pytest.raises(ValueError, match="CUDA tensor"):
            run_proj_gemm(bs, a8, w8, out, ws, sf_a=good_a, sf_w=good_w)


# ---------------------------------------------------------------------------
# GPU: the block-scale driver vs the fake-quant reference
# ---------------------------------------------------------------------------


@requires_frost_gemm
@requires_fp8
@pytest.mark.parametrize("m, k, n", [(256, 512, 512), (256, 512, 384), (1000, 4096, 17408), _BLOCK_SHAPE], ids=["small", "auto_tile", "tail", "block"])
def test_block_scale_matches_fake_quant_with_distinctive_scales(m, k, n):
    """e4m3 x e4m3 with per-32-block 2^n dequant IN the MMA, fp32 accumulate, one bf16 rounding,
    against the exact fp32 product of the dequantized operands.  M=1000 is a tail tile (rows
    past M; SFA declared AND bound at 1024 rows -- the backend admits it only so, and did NOT
    "decline" it as an earlier revision of this driver reported); the block shape is stage (1)
    at 4096 tokens;
    N=384 has no forced 256-wide tile, so it is the AUTO-TILE JIT path (``_auto_tile_config``:
    the engine's own ``select_config`` + ``preferred_strategy`` pick, JIT'd so the SF binding is
    the same one path) -- the resolved name must be a real catalog config, not the forced one.
    Distinctive scale factors make a mis-addressed SF a 2^n error, the sentinel a never-written
    cell, the non-zero check the wrapped-descriptor hazard; two launches bit-identical."""
    case = distinctive_case(m, k, n)
    plan = build_proj_gemm(m=m, k=k, n=n, dtype=_FP8, label="qkv_gate_proj_mx", block_scale=True)
    assert plan.block_scale and plan.jit is not None and plan.route in ("graph+jit", "jit-only"), describe(plan)
    assert plan.sfa is not None and plan.sfb is not None and plan.mma_tile_k_bytes in (32, 64)
    if n % 256:
        assert plan.tile_config_name != _FORCED_K32 and str(plan.tile_config_name).startswith("CONFIG_"), describe(plan)
    else:
        assert plan.tile_config_name == _FORCED_K32, describe(plan)
    # The graph declares the SF tensors at the PADDED F8_128x4 extents (what the backend requires
    # and what the bound blob view is); at M=1000 that is 1024 rows.
    assert tuple(plan.sfa.get_dim()) == (1, *sf_padded_dims(m, k)), plan.sfa.get_dim()
    assert tuple(plan.sfb.get_dim()) == (1, sf_padded_dims(n, k)[1], sf_padded_dims(n, k)[0]), plan.sfb.get_dim()
    out = launch(plan, case)
    stats = check_bf16_of_fp32(out, case["ref32"], f"mxfp8 {m}x{k}x{n}")
    print(f"\n[{m}x{k}x{n}] {describe(plan)} | {stats}")
    out2 = launch(plan, case)
    assert torch.equal(out2, out), "two launches differ (a first-launch race)"


@requires_frost_gemm
@requires_fp8
def test_scale_factors_are_read_and_read_from_the_right_block():
    """Same codes, two scale-factor blobs: the outputs must differ where the scales differ, and
    a blob with every scale = 1.0 (byte 127) must reproduce the plain fp32 product of the codes
    -- so the SF path is neither ignored nor constant."""
    m, k, n = 256, 512, 512
    case = distinctive_case(m, k, n)
    plan = build_proj_gemm(m=m, k=k, n=n, dtype=_FP8, label="mx_sf", block_scale=True)
    out_distinct = launch(plan, case)
    ones_a = blocked_sf(torch.full_like(case["e_a"], 127))
    ones_w = blocked_sf(torch.full_like(case["e_w"], 127))
    out_unit = launch(plan, case, sf_a=ones_a, sf_w=ones_w)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        ref_unit = case["a8"].float() @ case["w8"].float().t()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    check_bf16_of_fp32(out_unit, ref_unit, "unit scales")
    check_bf16_of_fp32(out_distinct, case["ref32"], "distinctive scales")
    rel = ((out_distinct.float() - out_unit.float()).abs() / out_unit.float().abs().clamp_min(1e-3)).median().item()
    assert rel > 0.5, f"the distinctive scales barely moved the output (median rel change {rel:.3f}) -- are the blobs read?"


@requires_frost_gemm
@requires_rubin
@requires_fp8
@pytest.mark.parametrize("m, k, n", [(256, 512, 512), _BLOCK_SHAPE], ids=["small", "block"])
def test_k32_and_k64_mma_forms(m, k, n):
    """D15: the forced 128x256 tile at both MMA-instruction K widths.  Both must match the
    fake-quant reference.  The two are NOT bit-identical on Rubin -- measured 2026-09-15 on
    the dev node (block shape, distinctive scales): bit-equal fraction 0.999655, 24572 differing
    cells of 71.3 M, max |k32-k64| = 1024 -- because the K=64 instruction reassociates the fp32
    accumulation differently from two K=32 steps, and that ~0.3 nudge (i) flips a bf16 MIDPOINT
    on the large cells (ref64 144896.145 = 144384 + 512.145 on the 1024-spaced grid: K32 rounds
    down, K64 up, both within 512.2 of the truth) and (ii) on CANCELLATION cells (ref ~ +-0.01 from
    sum|products| ~ 1.4e5) lands the two forms on different sub-unit numbers.  So the pin is
    STRUCTURAL -- one bf16 ulp of the REFERENCE magnitude plus the fp32-accumulation slack of
    ``compare_two_roundings`` per cell, and at least 99.9 % of cells bit-equal (a swapped block or
    a mis-read scale is a 2^n error, never that) -- and the fraction is printed for the record.
    PR-B's plan expected bit-identity; the silicon says otherwise, and this test records what it
    does rather than what was assumed."""
    case = distinctive_case(m, k, n)
    outs = {}
    for kb in (32, 64):
        plan = build_proj_gemm(m=m, k=k, n=n, dtype=_FP8, label=f"mx_k{kb}", block_scale=True, mma_tile_k_bytes=kb)
        assert plan.mma_tile_k_bytes == kb, describe(plan)
        assert plan.tile_config_name.startswith(f"CONFIG_sm100_128x256x128_128x256x{kb}_"), describe(plan)
        outs[kb] = launch(plan, case)
        stats = check_bf16_of_fp32(outs[kb], case["ref32"], f"K={kb}")
        print(f"\n[{m}x{k}x{n}] K={kb}: {describe(plan)} | {stats}")
    c = compare_two_roundings(outs[32], outs[64], case["ref32"])
    print(f"[{m}x{k}x{n}] K32 vs K64: {c}")
    assert c["within"], f"K=32 and K=64 differ by more than one bf16 rounding of the reference (+ fp32 accumulation slack): {c}"
    assert c["bit_equal_frac"] >= 0.999, f"K=32 and K=64 disagree on more than 0.1 % of cells: {c}"


@requires_frost_gemm
@requires_fp8
def test_route_and_resolved_tile_are_recorded_for_the_block_shape():
    """The perf table states what ran: the resolved tile config, its MMA K width and the route.
    Default K width = the named 256-wide config's own (K=32) -- arch-neutral within the family."""
    m, k, n = _BLOCK_SHAPE
    plan = build_proj_gemm(m=m, k=k, n=n, dtype=_FP8, label="qkv_gate_proj", block_scale=True)
    print(f"\n[block shape default] {describe(plan)}")
    assert plan.tile_config_name == _FORCED_K32 and plan.mma_tile_k_bytes == 32
    assert plan.route in ("graph+jit", "jit-only")
    assert plan.workspace_bytes >= 1


@requires_frost_gemm
@requires_rubin
@requires_fp8
def test_k64_retargets_the_forced_tile_for_the_block_shape():
    """``mma_tile_k_bytes=64`` re-targets the forced 256-wide config to its K=64 twin
    (``tile_config.as_mma_tile_k``) and the plan records that width.  Rubin only: the 64-byte
    MMA-instruction K is a typed ``NotImplementedError`` on Blackwell."""
    m, k, n = _BLOCK_SHAPE
    plan64 = build_proj_gemm(m=m, k=k, n=n, dtype=_FP8, label="qkv_gate_proj", block_scale=True, mma_tile_k_bytes=64)
    print(f"\n[block shape K=64]    {describe(plan64)}")
    assert plan64.tile_config_name == _FORCED_K32.replace("x32_", "x64_") and plan64.mma_tile_k_bytes == 64
    assert plan64.route in ("graph+jit", "jit-only")


@requires_frost_gemm
@requires_fp8
def test_graph_route_agrees_with_jit_route_when_both_admit():
    """When the backend admits the two-dequant graph AND ranks ``frost_gemm`` (route
    ``graph+jit``), the graph engine's own plan runs the same kernel family at its auto
    strategy; both are held to the fake-quant bound and the bit-equal fraction is reported
    (their tile configs may differ, so bit-identity is reported, not required)."""
    m, k, n = 256, 512, 512
    case = distinctive_case(m, k, n)
    plan = build_proj_gemm(m=m, k=k, n=n, dtype=_FP8, label="mx_route", block_scale=True)
    if plan.route != "graph+jit":
        pytest.skip(f"the backend did not admit the block-scale graph here ({describe(plan)}); the JIT-only route served it")
    out_jit = launch(plan, case)
    out_graph = torch.full((m, n), _SENTINEL, device="cuda", dtype=torch.bfloat16)
    ws = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda")
    from cudnn.gated_attention_block.kernels.proj_gemm import _rank3, _sf_view

    vp = {
        plan.a: _rank3(case["a8"], "a"),
        plan.b: _rank3(case["w8"], "w"),
        plan.c: _rank3(out_graph, "out"),
        plan.sfa: _sf_view(plan, case["sf_a"], "sf_a", m),
        plan.sfb: _sf_view(plan, case["sf_w"], "sf_w", n),
    }
    plan.graph.execute(vp, ws, None)
    torch.cuda.synchronize()
    check_bf16_of_fp32(out_graph, case["ref32"], "graph route")
    check_bf16_of_fp32(out_jit, case["ref32"], "jit route")
    c = compare_two_roundings(out_graph, out_jit, case["ref32"])
    print(f"\n[graph vs jit] {describe(plan)} | {c}")
    # The graph engine's auto strategy takes the K=64 form on Rubin (kernel_registry.preferred_mma_tile_k_bytes)
    # while the forced tile is the K=32 form -- the same reassociation flips as test_k32_and_k64_mma_forms.
    assert c["within"] and c["bit_equal_frac"] >= 0.999, c


@requires_frost_gemm
@requires_fp8
def test_native_reference_through_the_identical_graph():
    """``pin_frost=False``: the same two-dequant graph on whatever the backend ranks first (a
    native cuDNN reference).  A backend that does not serve it declines with the typed
    ``cudnnGraphNotSupportedError`` -> skip; anything else is a bug."""
    import cudnn

    m, k, n = 256, 512, 512
    case = distinctive_case(m, k, n)
    try:
        plan = build_proj_gemm(m=m, k=k, n=n, dtype=_FP8, label="mx_native", block_scale=True, pin_frost=False)
    except cudnn.cudnnGraphNotSupportedError as exc:
        pytest.skip(f"the backend does not serve the e4m3 + E8M0 two-dequant graph here: {exc}")
    assert plan.jit is None and plan.route == "graph", describe(plan)
    out = launch(plan, case)
    stats = check_bf16_of_fp32(out, case["ref32"], "native")
    print(f"\n[native] {describe(plan)} | {stats}")
