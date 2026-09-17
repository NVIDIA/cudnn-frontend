# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The block-scale projection GEMM with fp4 (E2M1) operands: ``build_proj_gemm(w_dtype=, block_size=, sf_dtype=)``.

Three operand PAIRS, each one catalog row of the FROST block-scale GEMM
(``gemm/frost/kernel_registry.py`` ``_BLOCK_SCALE_CASES``), and nothing else:

* **mixed** -- e4m3 ``A`` x e2m1 ``W`` under E8M0 scales per 32 (stage (1) with an MXFP4
  ``W_qkvg`` against the MXFP8 ``h``; the E8M0 / 32 scale blobs are today's);
* **NVFP4 x NVFP4** -- e2m1 x e2m1 under E4M3 scales per 16;
* **MXFP4 x MXFP4** -- e2m1 x e2m1 under E8M0 scales per 32

(the two both-fp4 rows are stage (6) when the gated O is quantized to fp4 and ``W_o`` is
stored in the same format).  What is under test is the driver's WIRING of those rows:

* the pairing table -- every claimed pair builds (with ``sf_dtype`` resolving from the pair
  when unspecified), every other ``(sf_dtype, block_size)`` is a typed ``ValueError`` -- because
  the DECLARED scale dtype selects the MMA's scale format (``fusion_ir.sf_scale_format``) and a
  wrong one miscomputes rather than fails;
* the graph declaration: LOGICAL dims for an fp4 side (``[1, M, K]`` codes, not bytes),
  ``data_type=FP4_E2M1`` on that side only, the scale tensors at the PADDED F8_128x4 extents of
  the pair's block (``sf_padded_dims(rows, k, block)``), ``block_scale_dequantize`` at that block;
* the scale-factor helpers at block 16 (``sf_padded_dims`` / ``sf_blob_bytes`` / the e4m3 re-view);
* the runner's NEW operand guards -- dtype must match the plan's per side, an fp4 side binds its
  PACKED ``[.., K/2]`` storage (a LOGICAL ``[.., K]`` fp4 tensor is refused), a uint8 blob is
  declined with the ``.view(torch.float4_e2m1fn_x2)`` hint;
* what ``APIBase._make_tensor_desc`` REPORTS for a ``float4_e2m1fn_x2`` tensor (the logical
  ``K``, derived once here so the block's shape checks can be written against it);
* on Rubin: every pair against the exact fp32 product of the dequantized operands at the block's
  own shapes -- ``(M, K=4096, N=17408)`` mixed, ``(M, K=8192, N=4096)`` both-fp4 -- with
  DISTINCTIVE power-of-two scales (a scale read from the wrong block is a 2^n error, never
  noise), a sentinel, two launches bit-identical, both MMA K forms (D15).

Every e2m1 x {e4m3 | e2m1} x 2^n product is exactly representable in fp32 (1+3+1+3 significand
bits at most), so the fp32 reference is exact and the bar is one bf16 rounding of the output
plus fp32 reassociation -- ``sdpa-invariants.md`` style: derived, not tuned.  The plain-torch
``float4_e2m1fn_x2`` dtype has NO casts in torch 2.13 (only ``.view``), so codes are random bytes
decoded through the 16-entry E2M1 table and the oracle never casts.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

from cudnn.gated_attention_block.kernels.proj_gemm import (  # noqa: E402
    SF_BLOCK_SIZES,
    ProjGemmPlan,
    _check_operand,
    _sf_view,
    block_scale_pairing,
    build_proj_gemm,
    run_proj_gemm,
    sf_blob_bytes,
    sf_padded_dims,
)

_FP8 = getattr(torch, "float8_e4m3fn", None)
_E8M0 = getattr(torch, "float8_e8m0fnu", None)
_FP4 = getattr(torch, "float4_e2m1fn_x2", None)
_SENTINEL = 1.5e30
_FORCED_K32 = "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma"
# The 397B geometry: stage (1) is M x K=d_model x N=n_qkvg; stage (6) is M x K=h_q*d_head x N=d_model.
_D_MODEL, _N_QKVG, _HQ_D = 4096, 17408, 32 * 256
# FP4 E2M1 value table, indexed by the 4-bit code; LOW nibble = even k (gemm_test_utils.unpack_fp4).
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

# The three pairs as (A dtype, W dtype, sf_dtype name, block).  `sf_dtype` is spelled by name and
# resolved lazily so `cudnn` stays an in-test import.
_PAIRS = {
    "mixed": ("e4m3", "e2m1", "FP8_E8M0", 32),
    "nvfp4": ("e2m1", "e2m1", "FP8_E4M3", 16),
    "mxfp4": ("e2m1", "e2m1", "FP8_E8M0", 32),
}


def _torch_dt(word: str):
    return {"e4m3": _FP8, "e2m1": _FP4, "bf16": torch.bfloat16}[word]


def _cudnn_sf(name: str):
    import cudnn

    return getattr(cudnn.data_type, name)


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
requires_fp4 = pytest.mark.skipif(_FP8 is None or _E8M0 is None or _FP4 is None, reason="this torch has no float8_e4m3fn / float8_e8m0fnu / float4_e2m1fn_x2")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device (the graph goes to the backend)")
_SM107 = (10, 7)
requires_rubin = pytest.mark.skipif(
    not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability()) != _SM107,
    reason="the fp4 block-scale rows are validated on SM 10.7 (the block's Rubin target; mma_tile_k_bytes=64 is SM 10.7 silicon)",
)


def mxfp8_quant():
    """``test/python/sdpa/mxfp8_quant.py`` -- importable as ``sdpa.mxfp8_quant`` from ``test/python``."""
    try:
        from sdpa import mxfp8_quant as mq
    except ImportError:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
        from sdpa import mxfp8_quant as mq
    return mq


# ---------------------------------------------------------------------------
# Oracle helpers: E2M1 decode, the padded F8_128x4 blob at any block, distinctive scales
# ---------------------------------------------------------------------------


def unpack_e2m1(u8: torch.Tensor) -> torch.Tensor:
    """Byte-packed e2m1 pairs -> fp32 values, low nibble first (even k), as the kernel reads them."""
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=u8.device)
    lo = lut[(u8 & 0xF).long()]
    hi = lut[(u8 >> 4).long()]
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def blocked_sf(e: torch.Tensor) -> torch.Tensor:
    """The PADDED F8_128x4 blob of a logical ``[rows, K/block]`` uint8 scale matrix, flat -- block-agnostic
    (the reorder acts on the ``[rows, cols]`` matrix; the block only sets ``cols``).  ``numel == sf_blob_bytes(rows, K, block)``."""
    mq = mxfp8_quant()
    assert e.dtype == torch.uint8
    rows, cols = e.shape
    rows_pad, cols_pad = -(-rows // 128) * 128, -(-cols // 4) * 4
    pad = torch.zeros(rows_pad, cols_pad, dtype=torch.uint8, device=e.device)
    pad[:rows, :cols] = e
    return mq.swizzle_sf_rowwise(pad).flatten()


def f8_128x4_byte(r: int, c: int, cols_pad: int) -> int:
    """Byte offset of scale ``(r, c)`` in an F8_128x4 blob of ``cols_pad`` blocks per row: atoms (128 rows x 4
    blocks = 512 B) row-major, inside an atom ``(r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4``."""
    return ((r // 128) * (cols_pad // 4) + (c // 4)) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (c % 4)


def pow2_f32(e: torch.Tensor) -> torch.Tensor:
    """Exactly ``2^e`` from the fp32 exponent bits -- NOT ``torch.pow`` / ``exp2``, which JIT through NVRTC and
    fail on cc 10.7 (frost-gotchas)."""
    return ((e.to(torch.int32) + 127) << 23).to(torch.int32).view(torch.float32)


def fp4_case(m: int, k: int, n: int, fmt: str, *, seed: int = 0, device="cuda") -> dict:
    """Operands, DISTINCTIVE per-block scales, their padded blobs and the exact fp32 reference for one pair.

    Codes: e2m1 sides are uniform random bytes (every code incl. -0.0), the e4m3 side is ``randn * 0.5``.
    Scales: E8M0 exponents uniform in ``[120, 134]`` (2^-7 .. 2^7); E4M3 scales ``2^n`` with
    ``n in [-2, 3]`` -- exactly representable, so every dequantized product is exact in fp32 and a
    scale fetched from the wrong block moves its contribution by a power of two, not by noise.
    The reference is fp32 with TF32 off.
    """
    mq = mxfp8_quant()
    a_word, w_word, sf_name, block = _PAIRS[fmt]
    torch.manual_seed(seed)
    sf_k = k // block

    def codes(rows: int, word: str):
        if word == "e2m1":
            u8 = torch.randint(0, 256, (rows, k // 2), dtype=torch.uint8, device=device)
            return u8.view(_FP4), unpack_e2m1(u8).view(rows, k)
        t8 = (torch.randn(rows, k, device=device) * 0.5).to(_FP8)
        return t8, t8.float()

    def scales(rows: int):
        if sf_name == "FP8_E8M0":
            e = torch.randint(120, 135, (rows, sf_k), dtype=torch.uint8, device=device)
            return e, mq.e8m0_to_float(e)
        n_exp = torch.randint(-2, 4, (rows, sf_k), device=device)
        sf = pow2_f32(n_exp).to(_FP8)
        return sf.view(torch.uint8), sf.float()

    a_rt, a_deq = codes(m, a_word)
    w_rt, w_deq = codes(n, w_word)
    e_a, s_a = scales(m)
    e_w, s_w = scales(n)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False  # an fp32 reference, not a TF32 one (frost-gotchas)
    try:
        ref32 = (a_deq * s_a.repeat_interleave(block, 1)) @ (w_deq * s_w.repeat_interleave(block, 1)).t()
        ref_unit = a_deq @ w_deq.t()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return dict(
        m=m, k=k, n=n, fmt=fmt, block=block, a=a_rt, w=w_rt, e_a=e_a, e_w=e_w, sf_a=blocked_sf(e_a), sf_w=blocked_sf(e_w), ref32=ref32, ref_unit=ref_unit
    )


def build_pair(m: int, k: int, n: int, fmt: str, **kw) -> ProjGemmPlan:
    a_word, w_word, sf_name, block = _PAIRS[fmt]
    return build_proj_gemm(
        m=m,
        k=k,
        n=n,
        dtype=_torch_dt(a_word),
        label=f"proj_{fmt}",
        block_scale=True,
        w_dtype=_torch_dt(w_word),
        block_size=block,
        sf_dtype=_cudnn_sf(sf_name),
        **kw,
    )


def launch(plan: ProjGemmPlan, case: dict, *, sf_a=None, sf_w=None) -> torch.Tensor:
    out = torch.full((case["m"], case["n"]), _SENTINEL, device="cuda", dtype=torch.bfloat16)
    ws = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda")
    run_proj_gemm(plan, case["a"], case["w"], out, ws, sf_a=case["sf_a"] if sf_a is None else sf_a, sf_w=case["sf_w"] if sf_w is None else sf_w)
    torch.cuda.synchronize()
    return out


def check_bf16_of_fp32(out: torch.Tensor, ref32: torch.Tensor, label: str) -> dict:
    """Exact products, fp32 accumulate, one bf16 rounding: ``rtol 2^-6`` (bf16 has 8 significand bits;
    four ulps of headroom for fp32 reassociation on cancelling cells), ``atol 2^-8 x max|ref|``, and
    ``cos > 0.9999``; plus no sentinel survivor, all finite, and NOT silently zero (the >256 KiB
    descriptor hazard reads as exactly zero)."""
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
    torch.testing.assert_close(o, ref32, rtol=2.0**-6, atol=2.0**-8 * scale, msg=lambda m: f"{label}: {m} -- {stats}")
    return stats


def describe(plan: ProjGemmPlan) -> str:
    return f"tile_config_name={plan.tile_config_name!r} mma_tile_k_bytes={plan.mma_tile_k_bytes} route={plan.route!r} w_dtype={plan.w_dtype} block={plan.block_size} sf={plan.sf_dtype}"


# ---------------------------------------------------------------------------
# No GPU: the pairing table, the helpers at block 16, the runner's operand guards, the descriptor pin
# ---------------------------------------------------------------------------


@requires_fp4
@pytest.mark.parametrize("fmt", sorted(_PAIRS))
def test_pairing_table_accepts_each_claimed_pair_and_resolves_its_default_scale_dtype(fmt):
    """Each claimed pair resolves to its one ``(sf_dtype, block)`` -- given explicitly or left as ``None``."""
    a_word, w_word, sf_name, block = _PAIRS[fmt]
    want = (_cudnn_sf(sf_name), block)
    assert block_scale_pairing(dtype=_torch_dt(a_word), w_dtype=_torch_dt(w_word), sf_dtype=_cudnn_sf(sf_name), block_size=block, label=fmt) == want
    assert block_scale_pairing(dtype=_torch_dt(a_word), w_dtype=_torch_dt(w_word), sf_dtype=None, block_size=block, label=fmt) == want
    if fmt == "mixed":  # the mixed row is served in both operand orders (kernel_registry rows fp4 x fp8 and fp8 x fp4)
        assert block_scale_pairing(dtype=_FP4, w_dtype=_FP8, sf_dtype=None, block_size=32, label=fmt) == want
    # MXFP8 x MXFP8 (today's stage (1)) is untouched by the widening.
    assert block_scale_pairing(dtype=_FP8, w_dtype=_FP8, sf_dtype=None, block_size=32, label="mx") == (_cudnn_sf("FP8_E8M0"), 32)


@requires_fp4
@pytest.mark.parametrize(
    "a_word, w_word, sf_name, block, match",
    [
        ("e4m3", "e2m1", "FP8_E4M3", 32, "no block-scale GEMM row"),  # mixed with e4m3 scales
        ("e4m3", "e2m1", "FP8_E8M0", 16, "no block-scale GEMM row"),  # mixed at block 16
        ("e2m1", "e2m1", "FP8_E8M0", 16, "no block-scale GEMM row"),  # both-fp4 (E8M0, 16)
        ("e2m1", "e2m1", "FP8_E4M3", 32, "no block-scale GEMM row"),  # both-fp4 (E4M3, 32): SM 10.7-only, unclaimed
        ("e4m3", "e4m3", "FP8_E4M3", 32, "no block-scale GEMM row"),  # MXFP8 with e4m3 scales
        ("bf16", "e2m1", "FP8_E8M0", 32, "e4m3 or e2m1 codes"),  # a non-quantized side
        ("e2m1", "e2m1", "FP8_E4M3", 48, "block_size must be one of"),  # an unknown block
        ("e2m1", "e2m1", "FP8_E4M3", 16.0, "block_size must be one of"),  # a float block: `16.0 in (16, 32)` is True, dims would go float
        ("e2m1", "e2m1", "FP8_E4M3", True, "block_size must be one of"),  # a bool block (an int subclass, == 1)
    ],
    ids=[
        "mixed_e4m3_scales",
        "mixed_block16",
        "fp4_e8m0_block16",
        "fp4_e4m3_block32",
        "mxfp8_e4m3_scales",
        "bf16_side",
        "block48",
        "block16_float",
        "block_bool",
    ],
)
def test_pairing_table_rejects_every_unlisted_pair(a_word, w_word, sf_name, block, match):
    """The scale dtype is BAKED into the MMA (idesc scale_format): an unlisted ``(sf_dtype, block)`` is a typed
    error at the table, in ``build_proj_gemm`` (ValueError) and in ``_Projection.check_support`` (NotImplementedError)."""
    from cudnn.gated_attention_block.api import _Projection

    kw = dict(dtype=_torch_dt(a_word), w_dtype=_torch_dt(w_word), sf_dtype=_cudnn_sf(sf_name), block_size=block)
    with pytest.raises(ValueError, match=match):
        block_scale_pairing(label="t", **kw)
    with pytest.raises(ValueError, match=match):
        build_proj_gemm(m=256, k=512, n=512, label="t", block_scale=True, pin_frost=False, **kw)
    if not isinstance(block, float):
        # `_Projection` coerces `int(block_size)` up front (the MxQuantSpec convention), so an integral float
        # passes ITS gate as the int it names; the table and build_proj_gemm above are where the type is pinned.
        with pytest.raises(NotImplementedError, match=match):
            _Projection(m=256, k=512, n=512, name="t", out_dtype=torch.bfloat16, block_scale=True, **kw).check_support()


@requires_fp4
def test_pairing_table_validates_the_scale_storage_dtype_at_construction():
    """A scale dtype the table accepts must also have a torch STORAGE dtype (what ``_sf_view`` re-views the blob as),
    and that check runs in ``block_scale_pairing`` -- at plan construction -- not at the first launch: a plan whose
    ``sf_dtype`` cannot be viewed never exists.  ``check_sf_torch_dtype`` names both the accepted dtypes and the bad one."""
    from cudnn.gated_attention_block.kernels import proj_gemm as pg

    assert pg.check_sf_torch_dtype(None) is torch.float8_e8m0fnu
    assert pg.check_sf_torch_dtype(_cudnn_sf("FP8_E8M0")) is torch.float8_e8m0fnu
    assert pg.check_sf_torch_dtype(_cudnn_sf("FP8_E4M3")) is torch.float8_e4m3fn
    with pytest.raises(ValueError, match="FP8_E8M0 or FP8_E4M3"):
        pg.check_sf_torch_dtype(_cudnn_sf("FP8_E5M2"))
    with pytest.raises(ValueError, match="FP8_E8M0 or FP8_E4M3"):
        pg.check_sf_torch_dtype("e4m3")
    # ... and the table reaches it: a pair the table serves, with the storage dtype missing on this torch, declines
    # by the SAME ValueError before any plan is built.
    import unittest.mock as mock

    with mock.patch.object(pg, "_E8M0", None):
        with pytest.raises(ValueError, match="no storage dtype"):
            block_scale_pairing(dtype=_FP8, w_dtype=_FP8, sf_dtype=None, block_size=32, label="t")
        with pytest.raises(ValueError, match="no storage dtype"):
            build_proj_gemm(m=256, k=512, n=512, dtype=_FP8, label="t", block_scale=True, pin_frost=False)


@requires_fp4
def test_fp4_needs_block_scale_and_the_dense_gemm_takes_one_dtype():
    """No dense fp4 MMA exists: an e2m1 side without ``block_scale`` is a ValueError, and so is a per-weight dtype
    on the dense path (mixed dtypes are block-scale rows only).  Mirrored by ``_Projection.check_support``."""
    from cudnn.gated_attention_block.api import _Projection

    with pytest.raises(ValueError, match="block-scale GEMM only"):
        build_proj_gemm(m=256, k=512, n=512, dtype=_FP4, w_dtype=_FP4, label="t", pin_frost=False)
    with pytest.raises(ValueError, match="block-scale GEMM only"):
        build_proj_gemm(m=256, k=512, n=512, dtype=_FP8, w_dtype=_FP4, label="t", pin_frost=False)
    with pytest.raises(ValueError, match="one operand dtype"):
        build_proj_gemm(m=256, k=512, n=512, dtype=torch.bfloat16, w_dtype=_FP8, label="t", pin_frost=False)
    with pytest.raises(NotImplementedError, match="under block_scale"):
        _Projection(m=256, k=512, n=512, dtype=_FP4, w_dtype=_FP4, name="t", out_dtype=torch.bfloat16).check_support()
    with pytest.raises(NotImplementedError, match="block-scale row"):
        _Projection(m=256, k=512, n=512, dtype=torch.bfloat16, w_dtype=_FP8, name="t").check_support()


@requires_fp4
@pytest.mark.parametrize("fmt", sorted(_PAIRS))
def test_alpha_stays_forbidden_and_k_stays_a_multiple_of_32_for_every_pair(fmt):
    """The per-block scales ARE the descale (no alpha for any pair -- a global fp4 scale is an appendable
    ``o_global_scale`` later, not alpha), and ``K % 32`` holds at block 16 too: it is the fp4 TMA rule
    (16 bytes = 32 codes) as much as the scale block."""
    from cudnn.gated_attention_block.api import _Projection

    a_word, w_word, sf_name, block = _PAIRS[fmt]
    kw = dict(dtype=_torch_dt(a_word), w_dtype=_torch_dt(w_word), sf_dtype=_cudnn_sf(sf_name), block_size=block)
    with pytest.raises(ValueError, match="alpha must be False"):
        build_proj_gemm(m=256, k=512, n=512, label="t", block_scale=True, alpha=True, pin_frost=False, **kw)
    with pytest.raises(ValueError, match="alpha must be False"):
        _Projection(m=256, k=512, n=512, name="t", out_dtype=torch.bfloat16, block_scale=True, alpha=True, **kw).check_support()
    with pytest.raises(ValueError, match="K % 32"):
        build_proj_gemm(m=256, k=528, n=512, label="t", block_scale=True, pin_frost=False, **kw)
    with pytest.raises(NotImplementedError, match="K % 32"):
        _Projection(m=256, k=528, n=512, name="t", out_dtype=torch.bfloat16, block_scale=True, **kw).check_support()
    # The accepted spelling of the same pair passes check_support and compiles to the same plan build_pair makes.
    _Projection(m=256, k=512, n=512, name="t", out_dtype=torch.bfloat16, block_scale=True, **kw).check_support()


@requires_fp4
def test_sm103_tile_configs_are_refused_on_a_block_scale_plan():
    """The sm103 block-scale template orders its rings A, B, SFA, SFB; on Rubin's 327 KiB budget its SF roots
    cross the 256 KiB tcgen05-descriptor line and the kernel returns NaN (pre-existing, mma-tma-matrix.md
    section 6).  A block-scale plan takes ``CONFIG_sm100_*`` only -- a host-side ValueError before any
    backend or JIT sees the graph, for every pair."""
    from cudnn.gemm.frost.tile_config import CATALOG

    sm103 = next(c.name for c in CATALOG if c.name.startswith("CONFIG_sm103_"))
    for fmt in sorted(_PAIRS):
        with pytest.raises(ValueError, match="CONFIG_sm100_"):
            build_pair(256, 512, 512, fmt, tile_config=sm103, pin_frost=False)
    with pytest.raises(ValueError, match="CONFIG_sm100_"):
        build_proj_gemm(m=256, k=512, n=512, dtype=_FP8, label="mx", block_scale=True, tile_config=sm103, pin_frost=False)


@pytest.mark.parametrize(
    "rows, k, block, expect",
    [
        (256, 512, 16, (256, 32)),  # 32 blocks, already a multiple of 4
        (1000, 8192, 16, (1024, 512)),  # the out_proj K at a tail M
        (4096, 8192, 32, (4096, 256)),
        (100, 64, 16, (128, 4)),  # 4 blocks
        (100, 48, 16, (128, 4)),  # 3 blocks -> padded to one 4-block word
        (17408, 4096, 32, (17408, 128)),  # the mixed row's W: unchanged from MXFP8
    ],
)
def test_sf_helpers_take_the_block(rows, k, block, expect):
    """``sf_padded_dims(rows, k, block)`` = whole 128-row x 4-block atoms of ``k/block`` scales; ``sf_blob_bytes`` is
    their product; block 16 doubles the K-blocks of block 32 and the default is block 32 (byte-identical MXFP8)."""
    assert sf_padded_dims(rows, k, block) == expect
    assert sf_blob_bytes(rows, k, block) == expect[0] * expect[1]
    if k % 32 == 0:
        assert sf_padded_dims(rows, k, 32) == sf_padded_dims(rows, k), "the default block is 32"
    if k % 128 == 0:
        assert sf_blob_bytes(rows, k, 16) == 2 * sf_blob_bytes(rows, k, 32)
    with pytest.raises(ValueError, match="block must be one of"):
        sf_padded_dims(rows, k, 48)
    with pytest.raises(ValueError, match="multiple of the 16"):
        sf_blob_bytes(rows, 40, 16)
    assert SF_BLOCK_SIZES == (16, 32)


@requires_fp4
def test_blocked_sf_at_block_16_matches_the_byte_formula():
    """The NVFP4 blob (``K/16`` e4m3 scales per row) uses the SAME F8_128x4 byte formula -- the block only
    sets the column count.  Random probes on the CPU; pad rows are 0x00."""
    g = torch.Generator().manual_seed(2)
    for rows, k in ((256, 512), (1000, 8192), (130, 1024)):
        sf_k = k // 16
        e = torch.randint(0, 255, (rows, sf_k), dtype=torch.uint8, generator=g)
        blob = blocked_sf(e.clone()).cpu()
        assert blob.numel() == sf_blob_bytes(rows, k, 16)
        cols_pad = -(-sf_k // 4) * 4
        rr = torch.randint(0, rows, (300,), generator=g)
        cc = torch.randint(0, sf_k, (300,), generator=g)
        for r, c in zip(rr.tolist(), cc.tolist()):
            assert int(blob[f8_128x4_byte(r, c, cols_pad)]) == int(e[r, c]), (rows, k, r, c)
        if rows % 128:
            assert all(int(blob[f8_128x4_byte(rows, c, cols_pad)]) == 0 for c in range(sf_k))


@requires_fp4
def test_runner_checks_operand_dtypes_and_the_packed_fp4_extent():
    """The graph carries the dtypes and the kernel binds a pointer, so before this widening a wrong ``w`` was
    REINTERPRETED byte-for-byte.  Now: dtype per side must match the plan; an fp4 side binds ``[.., K/2]``
    storage (a LOGICAL ``[.., K]`` fp4 tensor is refused -- it holds twice the data); uint8 storage is declined
    with the ``.view(torch.float4_e2m1fn_x2)`` hint.  Host-side, before any route, on any device."""
    m, k, n = 256, 512, 512
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nv = ProjGemmPlan(
        graph=None, a=None, b=None, c=None, m=m, k=k, n=n, label="nv", dtype=_FP4, w_dtype=_FP4, block_scale=True, block_size=16, sf_dtype=_cudnn_sf("FP8_E4M3")
    )
    mixed = ProjGemmPlan(graph=None, a=None, b=None, c=None, m=m, k=k, n=n, label="mixed", dtype=_FP8, w_dtype=_FP4, block_scale=True)
    a4 = torch.zeros(m, k // 2, dtype=torch.uint8, device=dev).view(_FP4)
    w4 = torch.zeros(n, k // 2, dtype=torch.uint8, device=dev).view(_FP4)
    a8 = torch.zeros(m, k, dtype=_FP8, device=dev)
    # The good operands pass the guard itself (the plan has no graph, so nothing further is asked of them).
    for plan, a, w in ((nv, a4, w4), (mixed, a8, w4)):
        _check_operand(plan, a, "a", plan.dtype)
        _check_operand(plan, w, "w", plan.w_dtype)
    out, ws = torch.zeros(m, n, dtype=torch.bfloat16, device=dev), torch.zeros(1, dtype=torch.uint8, device=dev)
    with pytest.raises(ValueError, match="built for torch.float4_e2m1fn_x2"):
        run_proj_gemm(nv, a4, torch.zeros(n, k, dtype=torch.bfloat16, device=dev), out, ws)  # a bf16 W on an fp4 plan
    with pytest.raises(ValueError, match=r"\.view\(torch\.float4_e2m1fn_x2\)"):
        run_proj_gemm(nv, a4, torch.zeros(n, k // 2, dtype=torch.uint8, device=dev), out, ws)  # uint8 storage, not viewed
    with pytest.raises(ValueError, match="K/2 = 256"):
        run_proj_gemm(nv, a4, torch.zeros(n, k, dtype=torch.uint8, device=dev).view(_FP4), out, ws)  # the LOGICAL [N, K] shape
    with pytest.raises(ValueError, match="K/2 = 256"):
        run_proj_gemm(nv, torch.zeros(m, k, dtype=torch.uint8, device=dev).view(_FP4), w4, out, ws)
    with pytest.raises(ValueError, match="built for torch.float8_e4m3fn"):
        run_proj_gemm(mixed, a4, w4, out, ws)  # the mixed row's A is e4m3
    with pytest.raises(ValueError, match="built for torch.float4_e2m1fn_x2"):
        run_proj_gemm(mixed, a8, torch.zeros(n, k, dtype=_FP8, device=dev), out, ws)  # ... and its W is e2m1
    # A hand-built plan with no dtype (the stream-routing tests' probe) checks nothing.
    _check_operand(ProjGemmPlan(graph=None, a=None, b=None, c=None, m=1, k=1, n=1, label="probe"), None, "a", None)


@requires_fp4
def test_sf_view_follows_the_plans_block_and_scale_dtype():
    """``_sf_view`` sizes the blob by the plan's ``block_size`` and re-views a uint8 blob as the plan's scale dtype:
    ``float8_e4m3fn`` for NVFP4 (a block-32-sized blob is one atom column short), ``float8_e8m0fnu`` otherwise.
    A blob of another one-byte dtype is refused by NAME, so an e8m0 blob cannot slip into an e4m3 plan."""
    m, k, n = 256, 512, 512
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nv = ProjGemmPlan(
        graph=None, a=None, b=None, c=None, m=m, k=k, n=n, label="nv", dtype=_FP4, w_dtype=_FP4, block_scale=True, block_size=16, sf_dtype=_cudnn_sf("FP8_E4M3")
    )
    mx = ProjGemmPlan(
        graph=None,
        a=None,
        b=None,
        c=None,
        m=m,
        k=k,
        n=n,
        label="mx4",
        dtype=_FP4,
        w_dtype=_FP4,
        block_scale=True,
        block_size=32,
        sf_dtype=_cudnn_sf("FP8_E8M0"),
    )
    blob16 = torch.zeros(sf_blob_bytes(m, k, 16), dtype=torch.uint8, device=dev)
    blob32 = torch.zeros(sf_blob_bytes(m, k, 32), dtype=torch.uint8, device=dev)
    assert blob16.numel() == 2 * blob32.numel()
    if dev == "cpu":
        with pytest.raises(ValueError, match="CUDA tensor"):
            _sf_view(nv, blob16, "sf_a", m)
        return
    v16 = _sf_view(nv, blob16, "sf_a", m)
    assert v16.dtype == _FP8 and tuple(v16.shape) == (1, *sf_padded_dims(m, k, 16)) and v16.data_ptr() == blob16.data_ptr()
    assert _sf_view(nv, blob16.view(_FP8), "sf_a", m).dtype == _FP8  # an e4m3-typed blob is accepted as is
    v32 = _sf_view(mx, blob32, "sf_a", m)
    assert v32.dtype == _E8M0 and tuple(v32.shape) == (1, *sf_padded_dims(m, k, 32))
    with pytest.raises(ValueError, match="F8_128x4 blob"):
        _sf_view(nv, blob32, "sf_a", m)  # block-32 bytes on a block-16 plan
    with pytest.raises(ValueError, match="uint8 or float8_e4m3fn"):
        _sf_view(nv, blob16.view(_E8M0), "sf_a", m)  # an e8m0-typed blob on the NVFP4 plan
    with pytest.raises(ValueError, match="uint8 or float8_e8m0fnu"):
        _sf_view(mx, blob32.view(_FP8), "sf_a", m)


@requires_fp4
def test_make_tensor_desc_reports_the_logical_k_of_an_fp4x2_tensor():
    """THE tier-1 pin the block's shape checks are written against: for a ``float4_e2m1fn_x2`` tensor stored
    ``[N, K/2]`` (one byte = two codes along the contiguous axis), ``APIBase._make_tensor_desc`` reports the
    LOGICAL shape ``(N, K)`` -- the innermost (stride-1) extent doubled -- with the innermost stride kept at 1
    and every other stride doubled, and keeps the fp4x2 dtype.  Storage stays ``element_size() == 1``,
    ``numel() == N * K/2``.  So a block check against the STORAGE shape must compare the raw tensor, and a
    check against the descriptor must expect ``K``, never both against one number."""
    from cudnn.api_base import APIBase

    class _Probe(APIBase):
        def check_support(self):
            pass

        def compile(self):
            pass

        def execute(self):
            pass

    n, k = 6, 32
    t = torch.zeros(n, k // 2, dtype=torch.uint8).view(_FP4)
    assert tuple(t.shape) == (n, k // 2) and t.element_size() == 1 and t.numel() == n * k // 2
    d = _Probe()._make_tensor_desc(t, name="w")
    assert d.dtype == _FP4
    assert tuple(d.shape) == (n, k), d.shape  # logical K, derived from storage: shape[-1] * 2
    assert tuple(d.stride) == (k, 1), d.stride  # innermost kept, the rest doubled: (K/2 * 2, 1)
    # The same tensor with a unit leading dim (the graph's rank-3 spelling) reports (1, N, K) / (N*K, K, 1).
    d3 = _Probe()._make_tensor_desc(t.unsqueeze(0), name="w3")
    assert tuple(d3.shape) == (1, n, k) and tuple(d3.stride) == (n * k, k, 1), (d3.shape, d3.stride)
    # A uint8 blob of the same bytes is NOT fp4 to the descriptor -- it reports its storage shape unchanged.
    du = _Probe()._make_tensor_desc(t.view(torch.uint8), name="u8")
    assert tuple(du.shape) == (n, k // 2) and du.dtype == torch.uint8


@requires_cuda
@requires_fp4
def _fake_compiled(config):
    """A JIT recorder that still lets the FROST GEMM engine build its plan: on a device where the ``frost_gemm``
    engine is ranked for the block-scale graph (Rubin), ``build_plans`` wraps the compiled artifact in
    ``_FrostGemmPlan``, which reads ``binding.bound_tensors()`` and ``getattr(compiled, "lowered")`` at
    construction (``gemm/frost/engine.py:33-41``); a bare ``binding=None`` recorder therefore fails there while
    passing on a box that takes the JIT-only route."""
    return SimpleNamespace(binding=SimpleNamespace(bound_tensors=lambda: [], outputs=[]), config=config, lowered=None, workspace_bytes=0)


@pytest.mark.parametrize("fmt", sorted(_PAIRS))
def test_graph_declares_logical_dims_fp4_dtypes_and_block_sized_scales(fmt, monkeypatch):
    """The declared graph, pinned WITHOUT running a kernel (the JIT is replaced by a recorder; a backend that
    declines the graph takes the typed JIT-only route, one that admits it builds its own plan -- either way the
    declaration is what is under test): A / B keep LOGICAL dims (``[1, M, K]``, ``[1, K, N]`` in codes); an e2m1
    side declares ``FP4_E2M1``, the e4m3 side ``FP8_E4M3`` (as the MXFP8 graph always did); SFA / SFB carry the
    pair's scale dtype at ``sf_padded_dims(rows, K, block)``; the plan records ``w_dtype`` / ``block_size`` /
    ``sf_dtype``.  The MXFP8 graph is byte-identical: it declares no explicit dtype on A / B."""
    import cudnn
    import cudnn.gemm.frost.compiler as compiler

    seen = {}

    def fake_jit(g, config):
        seen["config"] = config
        return _fake_compiled(config)

    monkeypatch.setattr(compiler, "jit_from_cudnn_graph", fake_jit)
    m, k, n = 256, 512, 512
    a_word, w_word, sf_name, block = _PAIRS[fmt]
    plan = build_pair(m, k, n, fmt)
    assert seen["config"].name == _FORCED_K32, describe(plan)
    assert (plan.dtype, plan.w_dtype, plan.block_size, plan.sf_dtype) == (_torch_dt(a_word), _torch_dt(w_word), block, _cudnn_sf(sf_name)), describe(plan)
    assert plan.block_scale and plan.route in ("graph+jit", "jit-only"), describe(plan)
    assert plan.a.get_dim() == [1, m, k] and plan.a.get_stride() == [m * k, k, 1], (plan.a.get_dim(), plan.a.get_stride())
    assert plan.b.get_dim() == [1, k, n] and plan.b.get_stride() == [k * n, 1, k], (plan.b.get_dim(), plan.b.get_stride())
    want = {"e2m1": cudnn.data_type.FP4_E2M1, "e4m3": cudnn.data_type.FP8_E4M3}
    assert plan.a.get_data_type() == want[a_word] and plan.b.get_data_type() == want[w_word]
    m_pad, sf_k = sf_padded_dims(m, k, block)
    n_pad, _ = sf_padded_dims(n, k, block)
    assert plan.sfa.get_dim() == [1, m_pad, sf_k] and plan.sfb.get_dim() == [1, sf_k, n_pad], (plan.sfa.get_dim(), plan.sfb.get_dim())
    assert plan.sfa.get_data_type() == _cudnn_sf(sf_name) == plan.sfb.get_data_type()
    assert plan.sfa.get_reordering_type() == cudnn.tensor_reordering.F8_128x4
    assert plan.c.get_data_type() == cudnn.data_type.BFLOAT16, "fp4 / fp8 inputs default to a bf16 output"


@requires_cuda
@requires_fp4
def test_mxfp8_graph_declaration_is_unchanged_by_the_widening(monkeypatch):
    """Today's stage (1): e4m3 x e4m3, E8M0 / 32 -- the graph it declares must be the one it always declared
    (SF dims at block 32, A / B at the graph's io dtype, plan defaults ``w_dtype == dtype``, ``block_size == 32``)."""
    import cudnn
    import cudnn.gemm.frost.compiler as compiler

    monkeypatch.setattr(compiler, "jit_from_cudnn_graph", lambda g, config: _fake_compiled(config))
    m, k, n = 256, 512, 512
    plan = build_proj_gemm(m=m, k=k, n=n, dtype=_FP8, label="mx", block_scale=True)
    assert (plan.w_dtype, plan.block_size, plan.sf_dtype) == (_FP8, 32, cudnn.data_type.FP8_E8M0)
    assert plan.a.get_data_type() == plan.b.get_data_type() == cudnn.data_type.FP8_E4M3
    assert plan.sfa.get_dim() == [1, *sf_padded_dims(m, k)] and plan.sfb.get_dim() == [1, sf_padded_dims(n, k)[1], sf_padded_dims(n, k)[0]]


# ---------------------------------------------------------------------------
# Rubin: every pair against the exact fp32 product of the dequantized operands
# ---------------------------------------------------------------------------


def _pair_shapes(fmt: str):
    """The block's own shapes: stage (1) for the mixed row, stage (6) for the both-fp4 rows; a tail M and 4096 tokens."""
    k, n = (_D_MODEL, _N_QKVG) if fmt == "mixed" else (_HQ_D, _D_MODEL)
    return [(1000, k, n), (4096, k, n)]


@requires_frost_gemm
@requires_rubin
@requires_fp4
@pytest.mark.parametrize("mma_tile_k_bytes", [None, 64], ids=["k_default", "k64"])
@pytest.mark.parametrize("fmt, m, k, n", [(f, *s) for f in sorted(_PAIRS) for s in _pair_shapes(f)], ids=lambda v: str(v))
def test_fp4_pairs_match_fake_quant_at_the_blocks_shapes(fmt, m, k, n, mma_tile_k_bytes):
    """Distinctive power-of-two scales, dequant IN the MMA, fp32 accumulate, one bf16 rounding -- against the exact
    fp32 product.  M=1000 is a tail tile (SF declared AND bound at 1024 rows); 4096 is the block at 4096 tokens.
    Both MMA K forms (D15) on the forced 256-wide tile (N is a 256-multiple at both stages); route / resolved
    config / block / scale dtype printed for the perf table; two launches bit-identical."""
    case = fp4_case(m, k, n, fmt)
    plan = build_pair(m, k, n, fmt, mma_tile_k_bytes=mma_tile_k_bytes)
    assert plan.jit is not None and plan.route in ("graph+jit", "jit-only"), describe(plan)
    want_cfg = _FORCED_K32 if mma_tile_k_bytes is None else _FORCED_K32.replace("x32_", "x64_")
    assert plan.tile_config_name == want_cfg and plan.mma_tile_k_bytes == (32 if mma_tile_k_bytes is None else 64), describe(plan)
    out = launch(plan, case)
    stats = check_bf16_of_fp32(out, case["ref32"], f"{fmt} {m}x{k}x{n}")
    print(f"\n[{fmt} {m}x{k}x{n}] {describe(plan)} | {stats}")
    out2 = launch(plan, case)
    assert torch.equal(out2, out), "two launches differ (a first-launch race)"


@requires_frost_gemm
@requires_rubin
@requires_fp4
@pytest.mark.parametrize("fmt", sorted(_PAIRS))
def test_scale_factors_are_read_from_the_right_block_for_every_pair(fmt):
    """Same codes, two blobs: unit scales (E8M0 byte 127 / e4m3 1.0) reproduce the plain product of the codes,
    distinctive scales move the output by 2^n -- so the SF path is neither ignored nor constant, and an e4m3
    blob is read AS e4m3 (a byte 0x38 = 1.0 read as E8M0 would be 2^-71 and zero the output)."""
    m, k, n = 256, 512, 512
    case = fp4_case(m, k, n, fmt)
    plan = build_pair(m, k, n, fmt)
    out_distinct = launch(plan, case)
    unit = 127 if _PAIRS[fmt][2] == "FP8_E8M0" else int(torch.ones((), dtype=_FP8).view(torch.uint8))
    out_unit = launch(plan, case, sf_a=blocked_sf(torch.full_like(case["e_a"], unit)), sf_w=blocked_sf(torch.full_like(case["e_w"], unit)))
    check_bf16_of_fp32(out_unit, case["ref_unit"], f"{fmt} unit scales")
    check_bf16_of_fp32(out_distinct, case["ref32"], f"{fmt} distinctive scales")
    rel = ((out_distinct.float() - out_unit.float()).abs() / out_unit.float().abs().clamp_min(1e-3)).median().item()
    assert rel > 0.5, f"the distinctive scales barely moved the output (median rel change {rel:.3f}) -- are the blobs read?"
