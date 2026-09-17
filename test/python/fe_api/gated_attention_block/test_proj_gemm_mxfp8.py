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

The second half of the module (``requires_rubin``) is the MXFP8 FORK TWIN
``kernels/proj_gemm_norm_rope_mxfp8.py`` -- the block-scale rendering with [norm +] RoPE and
the rowwise (Q/K) / columnwise (V) block quantization fused into its epilogue, writing the
compact e4m3 ``q8 / k8 / v8``, the bf16 ``gate16`` and the SDPA's three F8_128x4 E8M0
scale-factor blobs.  Oracle: ``quantize_to_mxfp8`` on the fp32 chain; the bar is SF bit-exact
(every differing byte, if any, a +-1 power-of-two boundary flip from fp32 reassociation),
codes > 99 % bit-equal and within one e4m3 ulp on agreeing blocks, GATE to bf16 rounding;
plus the tail-tile ``0x00`` SF rule, E8M0-NaN sentinels fully overwritten, two launches
bit-identical, the fused-vs-unfused (GEMM + ``qk_norm_rope`` + ``quantize_mxfp8``) cross-check
and the typed declines (``S % 128 != 0 and B > 1``, the runner/plan routing).

The scale-factor layout the driver binds is pinned on the CPU against the byte formula
of PR-B section 2.3 (F8_128x4: 128 rows x 4 blocks = 512-B atoms, row-major; byte
``(r % 32) * 16 + (r // 32) * 4 + c`` inside an atom), with
``test/python/sdpa/mxfp8_quant.py`` as the oracle -- the same helpers the SDPA MXFP8
suites use.
"""

from __future__ import annotations

import dataclasses
import os
import sys

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

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


# ---------------------------------------------------------------------------
# The MXFP8 FORK TWIN (``kernels/proj_gemm_norm_rope_mxfp8.py``): the block-scale GEMM
# with [norm +] RoPE AND the rowwise (Q/K) / columnwise (V) block quantization fused
# into its epilogue -- e4m3 codes + the SDPA's F8_128x4 E8M0 scale factors.  Rubin.
# Oracle = ``quantize_to_mxfp8`` (the SDPA suites' quantizer) on the fp32 chain
# (exact dequantized-operand product -> [norm] -> RoPE), quantized ONCE.
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cudnn.gated_attention_block import GatedAttentionBlockGeometry  # noqa: E402
from cudnn.gated_attention_block.kernels.proj_gemm import (  # noqa: E402
    FusedProjGemmPlan,
    NormRopeFusionParams,
    build_fused_proj_gemm,
    run_fused_proj_gemm,
    run_fused_proj_gemm_fp8,
    run_fused_proj_gemm_mxfp8,
    validate_norm_rope_params,
)
from cudnn.gated_attention_block.kernels.quantize_mxfp8 import n_sf_tiles, sf_bytes  # noqa: E402
from gated_block_reference import apply_partial_rope, build_rope_tables, qk_norm_rope_reference  # noqa: E402

_E4M3_NAN_BYTE = 0x7F  # e4m3fn NaN; `cvt.rn.satfinite` never produces it, so a surviving byte is a never-written cell
_E8M0_NAN_BYTE = 0xFF  # E8M0 NaN; a finite block amax never yields it, so a surviving byte is a never-written scale
# Small geometry: the fusion's constraints are on d_head (== the 256-wide tile) and rope_dim
# (whole subtile pairs), not on head counts, so h_q=4 keeps the N=3072 slab cheap while every
# Q | GATE | K | V arm is still exercised (K/V column remaps by 2048 / 2560).
_FUSED_GEOM = GatedAttentionBlockGeometry(d_model=1024, h_q=4, h_kv=2, d_head=256, rope_dim=64)


def _fused_mx_params(geom=_FUSED_GEOM, **kw) -> NormRopeFusionParams:
    return NormRopeFusionParams(d_head=geom.d_head, rope_dim=geom.rope_dim, h_q=geom.h_q, h_kv=geom.h_kv, eps=geom.qk_norm_eps, quant_mxfp8=True, **kw)


def _e4m3_ulp(x: torch.Tensor) -> torch.Tensor:
    """Spacing of e4m3 at magnitude |x|: 2^(e-3) for normals (|x| >= 2^-6), 2^-9 for subnormals -- from the fp32
    exponent bits (NOT torch.exp2 / log2: torch JIT-compiles them through NVRTC, which fails on cc 10.7)."""
    a = x.float().abs()
    _, e = torch.frexp(a.clamp_min(2.0**-6))
    k = e.to(torch.int32) - 1 - 3
    ulp_normal = ((k + 127) << 23).to(torch.int32).view(torch.float32)
    return torch.where(a >= 2.0**-6, ulp_normal, torch.full_like(a, 2.0**-9))


def unswizzle_128x4(sw: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Inverse of ``mxfp8_quant._swizzle_128x4``: an F8_128x4 blob back to the logical ``[rows, cols]`` scale matrix."""
    v = sw.reshape(rows // 128, cols // 4, 32, 4, 4)  # (rt, ct, rr, rg, cc)
    return v.permute(0, 3, 2, 1, 4).contiguous().reshape(rows, cols)  # (rt, rg, rr, ct, cc): r = rt*128 + rg*32 + rr


def mx_fused_inputs(batch: int, seq_len: int, k: int, p: NormRopeFusionParams, seed: int = 0) -> dict:
    """MXFP8-quantized h / W (the block's caller-supplied codes + padded F8_128x4 SF blobs), bf16 norm
    weights + RoPE tables, and the EXACT fp32 projection of the dequantized operands (TF32 off)."""
    mq = mxfp8_quant()
    torch.manual_seed(seed)
    m, n = batch * seq_len, p.n_qkvg
    a32 = torch.randn(m, k, device="cuda") * 0.5
    w32 = torch.randn(n, k, device="cuda") * 0.02
    a8, e_a = mq.quantize_blocks(a32.reshape(m, k // _BLOCK, _BLOCK), _FP8)
    w8, e_w = mq.quantize_blocks(w32.reshape(n, k // _BLOCK, _BLOCK), _FP8)
    a8, w8 = a8.reshape(m, k).contiguous(), w8.reshape(n, k).contiguous()
    a_s = a8.float() * mq.e8m0_to_float(e_a).repeat_interleave(_BLOCK, 1)
    w_s = w8.float() * mq.e8m0_to_float(e_w).repeat_interleave(_BLOCK, 1)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        proj32 = a_s @ w_s.t()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    dt = torch.bfloat16
    wq = (1.0 + 0.1 * torch.randn(p.d_head, device="cuda")).to(dt) if p.qk_norm else None
    wk = (1.0 + 0.1 * torch.randn(p.d_head, device="cuda")).to(dt) if p.qk_norm else None
    cos, sin = build_rope_tables(seq_len, p.rope_dim, batch=batch, device="cuda", dtype=dt)
    cos2, sin2 = cos.reshape(m, p.rope_dim).contiguous(), sin.reshape(m, p.rope_dim).contiguous()
    return dict(a8=a8, w8=w8, sf_a=blocked_sf(e_a), sf_w=blocked_sf(e_w), proj32=proj32, wq=wq, wk=wk, cos=cos2, sin=sin2, batch=batch, seq_len=seq_len)


def mx_fused_oracle(inp: dict, p: NormRopeFusionParams) -> dict:
    """The fp32 chain of the design contract, block-quantized ONCE by the SDPA suites' quantizer.

    Q / K: ``[rsqrt(mean(x^2) + eps) * w]`` then RoPE on the fp32 projection, then ``quantize_to_mxfp8``
    ROWWISE (the ``_d`` triple: blocks along D); V: the fp32 projection ``quantize_to_mxfp8`` COLUMNWISE
    (the ``_s`` triple: blocks along S, D-plane-major SF); GATE: the fp32 projection (rounded to bf16 by
    the test).  Codes come back in the fork's COMPACT ``[m, h*d]`` layout, the SF blobs flat -- byte
    for byte what ``kernels/quantize_mxfp8.py`` writes and ``sm107/prefill_d256_mxfp8.py`` reads.
    """
    mq = mxfp8_quant()
    b, s = inp["batch"], inp["seq_len"]
    m, d, r = b * s, p.d_head, p.rope_dim
    o_q, o_g, o_k, o_v = p.offsets
    y = inp["proj32"]
    cos3, sin3 = inp["cos"].reshape(1, m, r), inp["sin"].reshape(1, m, r)
    q4, k4 = y[:, o_q : o_q + p.h_q * d].view(1, m, p.h_q, d), y[:, o_k : o_k + p.h_kv * d].view(1, m, p.h_kv, d)
    if p.qk_norm:
        qn, _ = qk_norm_rope_reference(q4, inp["wq"], cos3, sin3, r, p.eps)
        kn, _ = qk_norm_rope_reference(k4, inp["wk"], cos3, sin3, r, p.eps)
    else:
        qn, kn = apply_partial_rope(q4, cos3, sin3, r), apply_partial_rope(k4, cos3, sin3, r)
    v32 = y[:, o_v : o_v + p.h_kv * d]

    def _bhsd(x, h):  # [m, h*d] (== [b, s, h, d]) -> [b, h, s, d] for the quantizer
        return x.reshape(b, s, h, d).permute(0, 2, 1, 3).contiguous()

    def _row(x, h):  # rowwise (Q/K): codes back to [m, h*d], SF flat
        fp8_d, _, sf_d, _, _, _ = mq.quantize_to_mxfp8(_bhsd(x, h), b, h, s, d, fp8_dtype=_FP8, with_ref=False)
        return fp8_d.permute(0, 2, 1, 3).reshape(m, h * d).contiguous(), sf_d.reshape(-1).contiguous()

    def _col(x, h):  # columnwise (V)
        _, _, _, fp8_s, _, sf_s = mq.quantize_to_mxfp8(_bhsd(x, h), b, h, s, d, fp8_dtype=_FP8, with_ref=False)
        return fp8_s.permute(0, 2, 1, 3).reshape(m, h * d).contiguous(), sf_s.reshape(-1).contiguous()

    q8, sf_q = _row(qn.reshape(m, p.h_q * d), p.h_q)
    k8, sf_k = _row(kn.reshape(m, p.h_kv * d), p.h_kv)
    v8, sf_v = _col(v32, p.h_kv)
    assert sf_q.numel() == sf_bytes(b, p.h_q, s, d) and sf_k.numel() == sf_v.numel() == sf_bytes(b, p.h_kv, s, d)
    return dict(
        q8=q8,
        k8=k8,
        v8=v8,
        sf_q=sf_q,
        sf_k=sf_k,
        sf_v=sf_v,
        gate32=y[:, o_g : o_g + p.h_q * d],
        qn32=qn.reshape(m, p.h_q * d),
        kn32=kn.reshape(m, p.h_kv * d),
        v32=v32,
    )


def mx_fused_sentinels(batch: int, seq_len: int, p: NormRopeFusionParams) -> dict:
    """Sentinel-filled outputs: e4m3 NaN byte for the codes, ``1.5e30`` for the bf16 gate, E8M0 NaN
    (0xFF) for every scale-factor byte -- so a survivor after the launch is a never-written cell,
    which for the SF blobs is the SDPA reading NaN scales."""
    m = batch * seq_len

    def _e4m3(width):
        return torch.full((m, width), _E4M3_NAN_BYTE, dtype=torch.uint8, device="cuda").view(_FP8)

    def _sf(heads):
        return torch.full((sf_bytes(batch, heads, seq_len, p.d_head),), _E8M0_NAN_BYTE, dtype=torch.uint8, device="cuda")

    return dict(
        q8=_e4m3(p.n_q),
        k8=_e4m3(p.n_kv),
        v8=_e4m3(p.n_kv),
        gate16=torch.full((m, p.n_gate), _SENTINEL, dtype=torch.bfloat16, device="cuda"),
        sf_q=_sf(p.h_q),
        sf_k=_sf(p.h_kv),
        sf_v=_sf(p.h_kv),
    )


def mx_fused_survivors(o: dict) -> dict:
    return dict(
        q8=int((o["q8"].view(torch.uint8) == _E4M3_NAN_BYTE).sum().item()),
        k8=int((o["k8"].view(torch.uint8) == _E4M3_NAN_BYTE).sum().item()),
        v8=int((o["v8"].view(torch.uint8) == _E4M3_NAN_BYTE).sum().item()),
        gate16=int((o["gate16"].float() == _SENTINEL).sum().item()),
        sf_q=int((o["sf_q"] == _E8M0_NAN_BYTE).sum().item()),
        sf_k=int((o["sf_k"] == _E8M0_NAN_BYTE).sum().item()),
        sf_v=int((o["sf_v"] == _E8M0_NAN_BYTE).sum().item()),
    )


def mx_launch(plan, inp: dict, p: NormRopeFusionParams, outs: dict) -> None:
    stream = torch.cuda.current_stream().cuda_stream
    run_fused_proj_gemm_mxfp8(
        plan,
        inp["a8"],
        inp["sf_a"],
        inp["w8"],
        inp["sf_w"],
        outs["q8"],
        outs["k8"],
        outs["v8"],
        outs["gate16"],
        outs["sf_q"],
        outs["sf_k"],
        outs["sf_v"],
        inp["wq"],
        inp["wk"],
        inp["cos"],
        inp["sin"],
        batch=inp["batch"],
        seq_len=inp["seq_len"],
        stream=stream,
    )
    torch.cuda.synchronize()


def compare_mx(got8, got_sf, ref8, ref_sf, *, axis: str, batch: int, heads: int, seq_len: int, d: int, label: str) -> dict:
    """An MXFP8 tensor (codes + F8_128x4 SF) against the oracle's, with the ONE legitimate discrepancy
    accounted for structurally rather than by a widened tolerance.

    The kernel accumulates the same exactly-representable products in fp32 in a different order
    (and its ``rsqrt`` is the MUFU approximation), so a block amax that lands within ~1e-6 of a power
    of two can round to the NEIGHBOURING E8M0 exponent: the whole block's codes then sit on a grid
    2x coarser or finer.  So: (1) the SF bytes must be bit-equal on >= 99.99 % of blocks and EVERY
    differing byte must differ by exactly 1 (a boundary flip; a wrong byte order or an unwritten byte
    is never that); (2) on the blocks whose SF agrees the codes are held to the FP8 fork's bar
    (> 99 % bit-equal, the rest within ONE e4m3 ulp); (3) on the flipped blocks the DEQUANTIZED
    values agree to within one code step of the coarser grid (2^-3 of the block amax).  Returns the
    stats so the test can print the measured fractions.
    """
    mq = mxfp8_quant()
    n_tiles = n_sf_tiles(seq_len)
    s_pad = n_tiles * 128
    m = batch * seq_len
    # SF bytes
    sf_eq = got_sf == ref_sf
    delta = (got_sf.to(torch.int32) - ref_sf.to(torch.int32)).abs()
    st = dict(sf_bit_equal_frac=sf_eq.float().mean().item(), n_sf_off=int((~sf_eq).sum().item()), sf_max_delta=int(delta.max().item()))
    assert st["sf_bit_equal_frac"] >= 0.9999, f"{label}: SF bytes differ on {100 * (1 - st['sf_bit_equal_frac']):.3f}% of blocks -- {st}"
    assert st["sf_max_delta"] <= 1, f"{label}: an SF byte differs by more than one exponent step (a wrong byte order / never-written byte) -- {st}"
    # logical scale matrices, per element
    if axis == "row":  # [b*h*s_pad, d/32]
        e_got = unswizzle_128x4(got_sf, batch * heads * s_pad, d // _BLOCK)
        e_ref = unswizzle_128x4(ref_sf, batch * heads * s_pad, d // _BLOCK)
        # rows [b*h*s_pad] -> [b, h, s_pad] -> the live rows [b, h, s] -> [m, h*d] element scales
        sc = (
            lambda e: mq.e8m0_to_float(e)
            .reshape(batch, heads, s_pad, d // _BLOCK)[:, :, :seq_len]
            .repeat_interleave(_BLOCK, 3)
            .permute(0, 2, 1, 3)
            .reshape(m, heads * d)
        )  # noqa: E731
    else:  # columnwise: the transposed [d, b*h*s_pad/32] scale matrix
        e_got = unswizzle_128x4(got_sf, d, batch * heads * s_pad // _BLOCK)
        e_ref = unswizzle_128x4(ref_sf, d, batch * heads * s_pad // _BLOCK)
        sc = (
            lambda e: mq.e8m0_to_float(e)
            .reshape(d, batch, heads, s_pad // _BLOCK)
            .permute(1, 2, 3, 0)
            .repeat_interleave(_BLOCK, 2)[:, :, :seq_len]
            .permute(0, 2, 1, 3)
            .reshape(m, heads * d)
        )  # noqa: E731
    sg, sr = sc(e_got), sc(e_ref)
    same_scale = sg == sr
    g, r_ = got8.float(), ref8.float()
    st["frac_cells_same_scale"] = same_scale.float().mean().item()
    # (2) codes on the agreeing blocks
    dcode = (g - r_).abs()
    ulps = dcode / _e4m3_ulp(torch.maximum(g.abs(), r_.abs()))
    _code_eq = (got8.view(torch.uint8) == ref8.view(torch.uint8))[same_scale]
    st["code_bit_equal_frac"] = _code_eq.float().mean().item()
    st["n_code_off"] = int((~_code_eq).sum().item())
    st["code_max_ulps"] = ulps[same_scale].max().item() if same_scale.any() else 0.0
    assert st["code_bit_equal_frac"] > 0.99, f"{label}: only {100 * st['code_bit_equal_frac']:.2f}% of codes bit-equal to the oracle on agreeing blocks -- {st}"
    assert st["code_max_ulps"] <= 1.0, f"{label}: codes differ by {st['code_max_ulps']:.2f} e4m3 ulps on agreeing blocks -- {st}"
    # (3) dequantized values everywhere; on flipped blocks within one coarse code step
    deq_g, deq_r = g * sg, r_ * sr
    st["deq_cos"] = torch.nn.functional.cosine_similarity(deq_g.flatten(), deq_r.flatten(), dim=0).item()
    if (~same_scale).any():
        # one e4m3 code step at the top of the COARSER grid: |code| in [256, 448] steps by 32 = 448 * 2^-3 codes -> x the coarser scale
        coarse_step = torch.maximum(sg, sr) * (448.0 * 2.0**-3)
        st["deq_max_err_flipped_over_step"] = ((deq_g - deq_r).abs() / coarse_step)[~same_scale].max().item()
        assert st["deq_max_err_flipped_over_step"] <= 1.0, f"{label}: a flipped block's dequantized values disagree beyond one coarse code step -- {st}"
    assert st["deq_cos"] > 0.9999, f"{label}: dequantized cos {st['deq_cos']} -- {st}"
    return st


def tail_sf_bytes(axis: str, *, batch: int, heads: int, seq_len: int, d: int) -> torch.Tensor:
    """Indices of the SF bytes of the PAD rows (s >= seq_len in the last 128-row tile): rowwise every block
    of a pad row; columnwise every 32-token block that lies ENTIRELY in the padding."""
    from cudnn.gated_attention_block.kernels.quantize_mxfp8 import sf_byte_columnwise, sf_byte_rowwise

    n_tiles = n_sf_tiles(seq_len)
    s_pad = n_tiles * 128
    offs = []
    for b in range(batch):
        for h in range(heads):
            if axis == "row":
                for s_ in range(seq_len, s_pad):
                    for c in range(0, d, _BLOCK):
                        offs.append(sf_byte_rowwise(b, h, s_, c, n_heads=heads, n_tiles=n_tiles, d=d))
            else:
                s_blk0 = -(-seq_len // _BLOCK) * _BLOCK  # first 32-token block that is all padding
                for s_ in range(s_blk0, s_pad, _BLOCK):
                    for dd in range(d):
                        offs.append(sf_byte_columnwise(b, h, s_, dd, n_heads=heads, n_tiles=n_tiles, batch=batch))
    return torch.tensor(sorted(set(offs)), dtype=torch.int64)


# ---- CPU: params, runner routing, the typed declines ----


def test_mxfp8_fusion_params_are_inference_only_and_keep_every_other_key():
    """``quant_mxfp8`` is the LAST field with a False default: every bf16 / FP8 params key is spelled
    identically.  Under MXFP8 there is no rstd, no ``off`` arm, and not both quant flags."""
    assert NormRopeFusionParams() == NormRopeFusionParams(quant_mxfp8=False)
    assert [f.name for f in dataclasses.fields(NormRopeFusionParams)][-3:] == ["quant_fp8", "qk_norm", "quant_mxfp8"]
    p = _fused_mx_params()
    validate_norm_rope_params(p)
    validate_norm_rope_params(_fused_mx_params(qk_norm=False))
    for src in ("ldg", "ldg_early", "const", "const_w", "const_cs"):
        validate_norm_rope_params(_fused_mx_params(norm_source=src))
    with pytest.raises(ValueError, match="want_rstd"):
        validate_norm_rope_params(_fused_mx_params(want_rstd=True))
    with pytest.raises(ValueError, match="off"):
        validate_norm_rope_params(_fused_mx_params(norm_source="off"))
    with pytest.raises(ValueError, match="at most one"):
        validate_norm_rope_params(_fused_mx_params(quant_fp8=True))
    with pytest.raises(ValueError, match="const_w"):
        validate_norm_rope_params(_fused_mx_params(qk_norm=False, norm_source="const_w"))


def test_fused_runners_refuse_the_other_forks_before_touching_a_tensor():
    """Three plans, three ABIs; every runner refuses the other two on the plan flags alone (Rule 1)."""
    pm = _fused_mx_params()
    fake_mx = FusedProjGemmPlan(params=pm, module=None, launch=None, mxfp8=True)
    fake_fp8 = FusedProjGemmPlan(params=dataclasses.replace(pm, quant_mxfp8=False, quant_fp8=True), module=None, launch=None, fp8=True)
    fake_bf16 = FusedProjGemmPlan(params=dataclasses.replace(pm, quant_mxfp8=False), module=None, launch=None)
    with pytest.raises(ValueError, match="run_fused_proj_gemm_mxfp8"):
        run_fused_proj_gemm(fake_mx, None, None, None, None, None, None, None, stream=0)
    with pytest.raises(ValueError, match="run_fused_proj_gemm_mxfp8"):
        run_fused_proj_gemm_fp8(fake_mx, *([None] * 11), stream=0)
    for other in (fake_fp8, fake_bf16):
        with pytest.raises(ValueError, match="not the MXFP8 fork"):
            run_fused_proj_gemm_mxfp8(other, *([None] * 15), batch=1, seq_len=128, stream=0)


def test_fused_mxfp8_runner_declines_tiles_that_straddle_sequences():
    """``S % 128 != 0 and B > 1``: a 128-row GEMM tile would hold rows of two sequences and the kernel's
    once-per-tile ``(b, s_tile)`` decode would be wrong -- a typed ``NotImplementedError`` BEFORE any tensor
    is read; ``B == 1`` at any S and ``B > 1`` at ``S % 128 == 0`` pass this gate.  Non-positive shapes are a
    ``ValueError``."""
    fake_mx = FusedProjGemmPlan(params=_fused_mx_params(), module=None, launch=None, mxfp8=True)
    with pytest.raises(NotImplementedError, match="multiple of 128"):
        run_fused_proj_gemm_mxfp8(fake_mx, *([None] * 15), batch=2, seq_len=1000, stream=0)
    with pytest.raises(ValueError, match="positive"):
        run_fused_proj_gemm_mxfp8(fake_mx, *([None] * 15), batch=0, seq_len=128, stream=0)
    # past the gate the next check reads the operands -- AttributeError on None proves the decline was not this one
    for b, s in ((1, 1000), (2, 256)):
        with pytest.raises(AttributeError):
            run_fused_proj_gemm_mxfp8(fake_mx, *([None] * 15), batch=b, seq_len=s, stream=0)


def test_unswizzle_inverts_the_oracle_swizzle():
    mq = mxfp8_quant()
    g = torch.Generator().manual_seed(4)
    for rows, cols in ((256, 8), (1024, 8), (256, 16)):
        e = torch.randint(0, 255, (rows, cols), dtype=torch.uint8, generator=g)
        assert torch.equal(unswizzle_128x4(mq.swizzle_sf_rowwise(e).reshape(-1), rows, cols), e)


# ---- GPU (Rubin): the twin vs the MXFP8 oracle ----


@requires_rubin
@requires_fp8
@pytest.mark.parametrize(
    "batch, seq_len, qk_norm",
    [(1, 1000, True), (1, 1000, False), (1, 900, True), (1, 256, True), (1, 256, False), (2, 256, True), (2, 384, True)],
    ids=["m1000_norm", "m1000_rope_only", "m900_norm", "m256_norm", "m256_rope_only", "b2_s256_norm", "b2_s384_norm"],
)
def test_fused_mxfp8_stage_matches_the_mxfp8_oracle(batch, seq_len, qk_norm):
    """Q/K [normed +] rotated on the fp32 block-scale accumulator and BLOCK-quantized once
    (rowwise, e4m3 codes + F8_128x4 E8M0 SF); V block-quantized COLUMNWISE; GATE bf16(acc) --
    each against the contract's fp32 chain through ``quantize_to_mxfp8``, each landing in its
    OWN compact tensor / SF blob (byte for byte what the unfused ``quantize_mxfp8`` writes).

    * ``m=1000`` / ``m=900``: a tail tile (rows past M, TMA-clipped on all four data outputs) --
      the pad rows' SF bytes must be ``0x00`` (rowwise: every pad row; columnwise at m=900: the
      three 32-token blocks that lie entirely in the padding), never the NaN residue.
    * ``b2_s256``: two sequences -> the per-tile ``(b, s_tile)`` decode, the ``(b*H + h)`` tile index
      and V's D-plane stride ``v_sf_groups = B*KH*n_tiles`` are exercised at b = 1.
    * ``b2_s384``: the two CTAs of ONE 256-row cluster tile lie in DIFFERENT sequences (rows 256..383 =
      (b0, s_tile 2), rows 384..511 = (b1, s_tile 0)) -- each CTA's decode is its own.
    * Sentinels: e4m3 NaN byte on the codes, 1.5e30 on the gate, E8M0 NaN (0xFF) on EVERY SF byte --
      all seven outputs fully overwritten.  Two launches bit-identical on all seven.
    """
    g = _FUSED_GEOM
    p = _fused_mx_params(g, qk_norm=qk_norm)
    plan = build_fused_proj_gemm(p)
    assert plan.mxfp8 and not plan.fp8
    inp = mx_fused_inputs(batch, seq_len, g.d_model, p)
    ref = mx_fused_oracle(inp, p)
    outs = mx_sentinels = mx_fused_sentinels(batch, seq_len, p)
    mx_launch(plan, inp, p, outs)
    surv = mx_fused_survivors(outs)
    assert all(v == 0 for v in surv.values()), f"sentinel survivors (never-written cells / NaN scale bytes): {surv}"

    stats = {}
    for name, axis, heads in (("Q", "row", g.h_q), ("K", "row", g.h_kv), ("V", "col", g.h_kv)):
        key8, keysf = {"Q": ("q8", "sf_q"), "K": ("k8", "sf_k"), "V": ("v8", "sf_v")}[name]
        stats[name] = compare_mx(outs[key8], outs[keysf], ref[key8], ref[keysf], axis=axis, batch=batch, heads=heads, seq_len=seq_len, d=g.d_head, label=name)
    check_bf16_of_fp32(outs["gate16"], ref["gate32"], "GATE")
    print(
        f"\n[b={batch} s={seq_len} qk_norm={qk_norm}] "
        + " | ".join(
            f"{k}: sf_eq {v['sf_bit_equal_frac']:.6f} ({v['n_sf_off']} off) code_eq {v['code_bit_equal_frac']:.7f} ({v['n_code_off']} off, max {v['code_max_ulps']:.1f} ulp) deq_cos {v['deq_cos']:.7f}"
            for k, v in stats.items()
        )
    )

    if seq_len % 128:
        # the tail rule: pad rows -> SF 0x00 (rowwise), fully-padded 32-token blocks -> 0x00 (columnwise)
        for name, axis, heads in (("sf_q", "row", g.h_q), ("sf_k", "row", g.h_kv), ("sf_v", "col", g.h_kv)):
            offs = tail_sf_bytes(axis, batch=batch, heads=heads, seq_len=seq_len, d=g.d_head)
            if offs.numel():
                pad = outs[name].cpu()[offs]
                assert int(pad.max().item()) == 0, f"{name}: {int((pad != 0).sum())} pad-row SF bytes are not 0x00 (max {int(pad.max())})"
        assert tail_sf_bytes("row", batch=batch, heads=g.h_q, seq_len=seq_len, d=g.d_head).numel() > 0
        if seq_len == 900:
            assert tail_sf_bytes("col", batch=batch, heads=g.h_kv, seq_len=seq_len, d=g.d_head).numel() == 3 * g.d_head * g.h_kv  # blocks 928 / 960 / 992

    # two-launch trick on all SEVEN outputs: a first-launch race shows as a non-zero delta
    outs2 = mx_fused_sentinels(batch, seq_len, p)
    mx_launch(plan, inp, p, outs2)
    for k_ in ("q8", "k8", "v8"):
        assert torch.equal(outs2[k_].view(torch.uint8), outs[k_].view(torch.uint8)), k_
    for k_ in ("gate16", "sf_q", "sf_k", "sf_v"):
        assert torch.equal(outs2[k_], outs[k_]), k_


@requires_rubin
@requires_fp8
@pytest.mark.parametrize(
    "batch, seq_len",
    [(1, 128), (1, 384), (3, 128), (2, 384)],
    ids=["m128_tail_cta", "m384_tail_cta", "b3_s128_tail_cta", "b2_s384_no_tail_cta"],
)
def test_fused_mxfp8_fully_tail_cta_stores_no_scale_factor_bytes(batch, seq_len):
    """The cluster tile is 256 rows and each CTA drains its own 128, so whenever ``M % 256`` is in
    ``(0, 128]`` the second CTA of the last cluster tile lies ENTIRELY past M.  Its data stores are
    TMA-clipped, but its ``(b, s_tile)`` decode (``coord_m_tile // seq_len``) reads ``b == B`` -- an
    ungated SF store lands PAST the blob (in the block: on the neighbouring ``sf_k`` slot) and, for V,
    plane 0 lands ON the valid tiles' plane 1, racing the correct bytes.  ``B=3, S=128`` is an
    ADMITTED shape (``S % 128 == 0`` passes the straddle decline).

    Every SF blob gets a 0xAA GUARD region appended and the runner is handed the exact-size prefix:
    no guard byte may change, and every blob must match the oracle under ``compare_mx`` (a raced
    plane-1 byte is 0x00 against a finite scale -- never a +-1 boundary flip).  ``b2_s384`` has no
    fully-tail CTA (M = 768) and is the control.
    """
    g = _FUSED_GEOM
    p = _fused_mx_params(g)
    plan = build_fused_proj_gemm(p)
    m = batch * seq_len
    fully_tail_cta = (m % 256) != 0 and (m % 256) <= 128
    assert fully_tail_cta == ((batch, seq_len) != (2, 384)), "the case list's fully-tail-CTA classification"
    inp = mx_fused_inputs(batch, seq_len, g.d_model, p)
    ref = mx_fused_oracle(inp, p)
    outs = mx_fused_sentinels(batch, seq_len, p)
    guard = 64 * 1024
    guarded = {}
    for name, heads in (("sf_q", g.h_q), ("sf_k", g.h_kv), ("sf_v", g.h_kv)):
        need = sf_bytes(batch, heads, seq_len, g.d_head)
        big = torch.full((need + guard,), 0xAA, dtype=torch.uint8, device="cuda")
        big[:need] = _E8M0_NAN_BYTE
        outs[name] = big[:need]  # the exact-size prefix: contiguous, 16-B aligned, numel == sf_bytes -- what the runner accepts
        guarded[name] = big
    mx_launch(plan, inp, p, outs)
    surv = mx_fused_survivors(outs)
    assert all(v == 0 for v in surv.values()), f"sentinel survivors (never-written cells / NaN scale bytes): {surv}"
    touched = {}
    for name in ("sf_q", "sf_k", "sf_v"):
        tail = guarded[name][outs[name].numel() :]
        hit = tail != 0xAA
        touched[name] = int(hit.sum().item())
        assert touched[name] == 0, f"{name}: {touched[name]} guard bytes PAST the blob were written (values {torch.unique(tail[hit]).tolist()})"
    stats = {}
    for name, axis, heads in (("Q", "row", g.h_q), ("K", "row", g.h_kv), ("V", "col", g.h_kv)):
        key8, keysf = {"Q": ("q8", "sf_q"), "K": ("k8", "sf_k"), "V": ("v8", "sf_v")}[name]
        stats[name] = compare_mx(outs[key8], outs[keysf], ref[key8], ref[keysf], axis=axis, batch=batch, heads=heads, seq_len=seq_len, d=g.d_head, label=name)
    print(
        f"\n[b={batch} s={seq_len} m={m} fully_tail_cta={fully_tail_cta}] guard bytes touched {touched} | "
        + " | ".join(f"{k}: sf_eq {v['sf_bit_equal_frac']:.6f} ({v['n_sf_off']} off) code_eq {v['code_bit_equal_frac']:.7f}" for k, v in stats.items())
    )


@requires_rubin
@requires_fp8
def test_fused_mxfp8_matches_the_unfused_block_scale_gemm_plus_quantizer():
    """The fused twin against the two kernels it replaces: ``build_proj_gemm(block_scale=True)`` on the same
    codes / SF, [norm +] RoPE on the bf16 slab, then ``quantize_mxfp8`` (S3's kernel) -- the unfused block
    path.  NOT bit-identical by construction (the unfused chain rounds the projection to bf16 before the
    norm / RoPE / quantization; the fused one stays fp32 -- the whole point), so the pin is the dequantized
    cosine and the SF byte ORDER: the two SF blobs must agree on >= 99 % of bytes (the bf16 rounding flips
    the E8M0 exponent of a block near a power-of-two boundary far more often than fp32 reassociation does)."""
    from cudnn.gated_attention_block.kernels.qk_norm_rope import compile_qk_norm_rope, run_qk_norm_rope
    from cudnn.gated_attention_block.kernels.quantize_mxfp8 import compile_quantize_mxfp8, run_quantize_mxfp8

    g = _FUSED_GEOM
    p = _fused_mx_params(g)
    batch, seq_len = 1, 512
    m, k, n = batch * seq_len, g.d_model, p.n_qkvg
    inp = mx_fused_inputs(batch, seq_len, k, p, seed=3)
    plan = build_fused_proj_gemm(p)
    outs = mx_fused_sentinels(batch, seq_len, p)
    mx_launch(plan, inp, p, outs)
    # unfused: block-scale GEMM -> bf16 slab
    gplan = build_proj_gemm(m=m, k=k, n=n, dtype=_FP8, label="qkv_gate_proj_mx", block_scale=True)
    slab = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    ws = torch.empty(gplan.workspace_bytes, dtype=torch.uint8, device="cuda")
    run_proj_gemm(gplan, inp["a8"], inp["w8"], slab, ws, sf_a=inp["sf_a"], sf_w=inp["sf_w"])
    torch.cuda.synchronize()
    o_q, o_g, o_k, o_v = p.offsets
    stream = torch.cuda.current_stream().cuda_stream
    d = g.d_head
    q_src = slab[:, o_q : o_q + g.h_q * d].view(m, g.h_q, d)
    k_src = slab[:, o_k : o_k + g.h_kv * d].view(m, g.h_kv, d)
    v_src = slab[:, o_v : o_v + g.h_kv * d].view(m, g.h_kv, d)
    nr = compile_qk_norm_rope(dtype=torch.bfloat16, h_q=g.h_q, h_kv=g.h_kv, d=d, rope_dim=g.rope_dim, eps=g.qk_norm_eps, want_rstd=False)
    run_qk_norm_rope(nr, q_src, k_src, q_src, k_src, inp["wq"], inp["wk"], inp["cos"], inp["sin"], stream=stream)
    q8u = torch.empty(m, g.h_q, d, dtype=_FP8, device="cuda")
    k8u = torch.empty(m, g.h_kv, d, dtype=_FP8, device="cuda")
    v8u = torch.empty(m, g.h_kv, d, dtype=_FP8, device="cuda")
    sfqu = torch.empty(sf_bytes(batch, g.h_q, seq_len, d), dtype=torch.uint8, device="cuda")
    sfku = torch.empty(sf_bytes(batch, g.h_kv, seq_len, d), dtype=torch.uint8, device="cuda")
    sfvu = torch.empty(sf_bytes(batch, g.h_kv, seq_len, d), dtype=torch.uint8, device="cuda")
    rq = compile_quantize_mxfp8(dtype_in=torch.bfloat16, h=g.h_q, d=d, axis="row")
    rk = compile_quantize_mxfp8(dtype_in=torch.bfloat16, h=g.h_kv, d=d, axis="row")
    rv = compile_quantize_mxfp8(dtype_in=torch.bfloat16, h=g.h_kv, d=d, axis="col")
    run_quantize_mxfp8(rq, q_src, q8u, sfqu, batch=batch, seq_len=seq_len, stream=stream)
    run_quantize_mxfp8(rk, k_src, k8u, sfku, batch=batch, seq_len=seq_len, stream=stream)
    run_quantize_mxfp8(rv, v_src, v8u, sfvu, batch=batch, seq_len=seq_len, stream=stream)
    torch.cuda.synchronize()
    mq = mxfp8_quant()
    for name, got8, got_sf, ref8, ref_sf, axis, heads in (
        ("Q", outs["q8"], outs["sf_q"], q8u.view(m, -1), sfqu, "row", g.h_q),
        ("K", outs["k8"], outs["sf_k"], k8u.view(m, -1), sfku, "row", g.h_kv),
        ("V", outs["v8"], outs["sf_v"], v8u.view(m, -1), sfvu, "col", g.h_kv),
    ):
        sf_eq = (got_sf == ref_sf).float().mean().item()
        assert sf_eq > 0.99, f"{name}: fused vs unfused SF blobs agree on only {100 * sf_eq:.2f}% of bytes -- a byte-ORDER mismatch, not rounding"
        # dequantize both through their own scales and compare the values
        n_tiles = n_sf_tiles(seq_len)
        s_pad = n_tiles * 128
        if axis == "row":
            sc = (
                lambda e: mq.e8m0_to_float(unswizzle_128x4(e, batch * heads * s_pad, d // _BLOCK))
                .reshape(batch, heads, s_pad, d // _BLOCK)[:, :, :seq_len]
                .repeat_interleave(_BLOCK, 3)
                .permute(0, 2, 1, 3)
                .reshape(m, heads * d)
            )  # noqa: E731
        else:
            sc = (
                lambda e: mq.e8m0_to_float(unswizzle_128x4(e, d, batch * heads * s_pad // _BLOCK))
                .reshape(d, batch, heads, s_pad // _BLOCK)
                .permute(1, 2, 3, 0)
                .repeat_interleave(_BLOCK, 2)[:, :, :seq_len]
                .permute(0, 2, 1, 3)
                .reshape(m, heads * d)
            )  # noqa: E731
        deq_f, deq_u = got8.float() * sc(got_sf), ref8.float() * sc(ref_sf)
        cos_fu = torch.nn.functional.cosine_similarity(deq_f.flatten(), deq_u.flatten(), dim=0).item()
        print(f"\n[fused vs unfused] {name}: sf bytes equal {100 * sf_eq:.3f}%  dequantized cos {cos_fu:.6f}")
        assert cos_fu > 0.999, f"{name}: fused vs unfused dequantized cos {cos_fu}"
    check_bf16_of_fp32(outs["gate16"], slab[:, o_g : o_g + g.h_q * d].float(), "GATE vs the unfused bf16 slab")


@requires_rubin
@requires_fp8
def test_fused_mxfp8_runner_rejects_wrong_bindings():
    g = _FUSED_GEOM
    p = _fused_mx_params(g)
    plan = build_fused_proj_gemm(p)
    batch, seq_len = 1, 256
    m = batch * seq_len
    inp = mx_fused_inputs(batch, seq_len, g.d_model, p)
    ok = mx_fused_sentinels(batch, seq_len, p)

    def launch(inp_over=None, out_over=None, **kw):
        i2 = {**inp, **(inp_over or {})}
        o2 = {**ok, **(out_over or {})}
        i2["batch"], i2["seq_len"] = kw.get("batch", batch), kw.get("seq_len", seq_len)
        mx_launch(plan, i2, p, o2)

    with pytest.raises(NotImplementedError, match="multiple of 128"):
        launch(batch=2, seq_len=125)  # would straddle sequences (m = 250 != 256 is checked AFTER the typed decline)
    with pytest.raises(ValueError, match="batch\\*seq_len"):
        launch(batch=2, seq_len=256)  # m = 256 != 512
    with pytest.raises(ValueError, match="out_q8"):
        launch(out_over=dict(q8=torch.empty(m, p.n_kv, dtype=_FP8, device="cuda")))
    with pytest.raises(ValueError, match="out_v8"):
        launch(out_over=dict(v8=torch.empty(m, 2 * p.n_kv, dtype=_FP8, device="cuda")[:, p.n_kv :]))  # non-contiguous
    with pytest.raises(ValueError, match="out_gate16"):
        launch(out_over=dict(gate16=torch.empty(m, p.n_gate, dtype=torch.float16, device="cuda")))
    with pytest.raises(ValueError, match="out_sf_q"):
        launch(out_over=dict(sf_q=ok["sf_q"][:-1024]))  # one tile short
    with pytest.raises(ValueError, match="out_sf_v"):
        launch(out_over=dict(sf_v=ok["sf_v"].to(torch.int8)))
    with pytest.raises(ValueError, match="out_sf_k"):
        launch(out_over=dict(sf_k=torch.empty(2 * ok["sf_k"].numel(), dtype=torch.uint8, device="cuda")[1::2]))  # not contiguous
    with pytest.raises(ValueError, match="sf_a"):
        launch(inp_over=dict(sf_a=inp["sf_a"][:-512]))  # the GEMM's own SF blob one atom short
    with pytest.raises(ValueError, match="sf_w"):
        launch(inp_over=dict(sf_w=inp["sf_w"].to(torch.int8)))
    with pytest.raises(ValueError, match="bf16 table"):
        launch(inp_over=dict(cos=inp["cos"].to(_FP8)))
    with pytest.raises(ValueError, match="e4m3 codes"):
        launch(inp_over=dict(a8=inp["a8"].to(torch.bfloat16)))
    with pytest.raises(ValueError, match="qk_norm=True"):
        launch(inp_over=dict(wq=None, wk=None))
