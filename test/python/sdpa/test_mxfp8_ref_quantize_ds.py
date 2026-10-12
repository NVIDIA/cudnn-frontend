# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sdpa.mxfp8_ref.compute_ref_backward(..., quantize_ds=)`` -- the dS treatment of the MXFP8 backward oracle, appended
(append-only, default = the shipped behaviour) for the SM107 MXFP8 d=256 backward's three dS policies:

* ``True`` (default): cuDNN's convention -- dS rounded to E4M3 per 1x32 block along BOTH orientations (``quantize_to_mxfp8``:
  along kv into the dQ product, along q into the dK product), what ``test_sdpa_bwd_mxfp8_sm100.py`` and the backend
  MXFP8 suite pass against, and the structural twin of an exact-1x32-both-ways kernel dS policy;
* ``False``: dS held in fp32 into the dQ / dK products -- the twin of a chain whose dQ / dK consume dS in a wider dtype
  (a bf16-dS bring-up chain), exactly the ``fp8_ref.compute_ref_backward(quantize_ds=False)`` recipe the sm107
  per-tensor fp8 suite uses for ITS bf16-dS twin (``test_sdpa_bwd_fp8_sm107.py::ref_bwd``): an e4m3-dS oracle against a
  wider-dS chain misreports the chain by the oracle's own rounding noise (MEASURED there at 0.56 % of dQ);
* ``(32, 32)``: ONE E8M0 per 32 x 32 (q x kv) tile of dS, the same dequantized tile feeding both products -- a
  one-scale-per-tile kernel dS policy (a tile scale = max over 32 kv x 32 q, coarser than cuDNN's 1x32).

Every case below is a one-KV-block problem (``s_kv = 128 = fp16_ref.BLOCK``) on the CPU, so an EXPLICIT recomputation of the
oracle's per-block math with each dS treatment can be held BITWISE against the oracle -- the default against a second
spelling of today's 1x32 quantization (not against itself), ``False`` against the fp32-dS recipe, ``(32, 32)`` against a
tile loop written out per tile.  No kernel runs; the SM100 MXFP8 GPU suite that consumes the default is not exercised here.
"""

import math

import pytest
import torch

from sdpa.fp16_ref import _ScoreMask, _kv_reduce, _pv, _qk, _score_blocks
from sdpa.mxfp8_quant import e8m0_ceil, e8m0_to_float, exp2_rcp, quantize_blocks, quantize_to_mxfp8
from sdpa.mxfp8_ref import _dequant, compute_ref, compute_ref_backward

pytestmark = pytest.mark.L0

_E4M3 = torch.float8_e4m3fn
_E4M3_MAX = 448.0
_LOG2E = math.log2(math.e)


def _quant(x, b, h, s, d):
    row, row_ref, _sw, col, col_ref, _csw = quantize_to_mxfp8(x, b, h, s, d, block_size=32, fp8_dtype=_E4M3)
    return dict(row=row, row_ref=row_ref, col=col, col_ref=col_ref)


class _Problem:
    """A small MXFP8 backward problem: GQA (h_q=2 over h_k=1), s_q=64, s_kv=128 (ONE oracle KV block), d=128, on the CPU."""

    def __init__(self, seed=0, b=1, h_q=2, h_k=1, s_q=64, s_kv=128, d=128):
        assert s_kv == 128, "one KV block: the explicit recomputation below is written for a single block"
        g = torch.Generator().manual_seed(seed)
        self.b, self.h_q, self.h_k, self.s_q, self.s_kv, self.d = b, h_q, h_k, s_q, s_kv, d
        q = torch.randn(b, h_q, s_q, d, generator=g)
        k = torch.randn(b, h_k, s_kv, d, generator=g)
        v = torch.randn(b, h_k, s_kv, d, generator=g)
        dO = torch.randn(b, h_q, s_q, d, generator=g)
        self.Q, self.K, self.V, self.DO = _quant(q, b, h_q, s_q, d), _quant(k, b, h_k, s_kv, d), _quant(v, b, h_k, s_kv, d), _quant(dO, b, h_q, s_q, d)
        self.scale = 1.0 / math.sqrt(d)
        o_ref, self.stats = compute_ref(
            self.Q["row"], self.K["row"], self.V["col"], self.Q["row_ref"], self.K["row_ref"], self.V["col_ref"], self.scale, output_type=torch.bfloat16
        )
        self.o_f16 = o_ref.to(torch.bfloat16)
        self.dO_f16 = dO.to(torch.bfloat16)

    def args(self):
        Q, K, V, DO = self.Q, self.K, self.V, self.DO
        return (
            Q["row"],
            Q["col"],
            K["row"],
            K["col"],
            V["row"],
            self.o_f16,
            self.dO_f16,
            DO["row"],
            DO["col"],
            self.scale,
            Q["row_ref"],
            Q["col_ref"],
            K["row_ref"],
            K["col_ref"],
            V["row_ref"],
            DO["row_ref"],
            DO["col_ref"],
        )

    def oracle(self, otype=torch.float32, **kw):
        dQ, dK, dV, _dsink = compute_ref_backward(*self.args(), torch_otype=otype, stats=self.stats, **kw)
        return dQ, dK, dV

    # -- the explicit one-block recomputation ---------------------------------------------------------------------------
    def _ds_and_p(self):
        """Lines 106-171 of the oracle for the single KV block: (p, p_fp8, dS) plus the dequantized operands."""
        b, h_q, h_k, s_q, d = self.b, self.h_q, self.h_k, self.s_q, self.d
        Q, K, V, DO = self.Q, self.K, self.V, self.DO
        q_dq, k_dq = _dequant(Q["row"], Q["row_ref"]), _dequant(K["row"], K["row_ref"])
        dO_dq, v_dq = _dequant(DO["row"], DO["row_ref"]), _dequant(V["row"], V["row_ref"])
        dO_t_dq, k_t_dq, q_t_dq = _dequant(DO["col"], DO["col_ref"]), _dequant(K["col"], K["col_ref"]), _dequant(Q["col"], Q["col_ref"])
        mask = _ScoreMask(
            b,
            h_q,
            s_q,
            self.s_kv,
            bias=None,
            block_mask=None,
            is_alibi=False,
            padding=None,
            diag_align=None,
            left_bound=None,
            right_bound=None,
            device=q_dq.device,
        )
        blocks = list(_score_blocks(q_dq, k_dq, None, mask))
        assert len(blocks) == 1
        start, end, s_raw = blocks[0]
        lse = self.stats.float().reshape(b, h_q, s_q, 1)
        D = (self.o_f16.float() * self.dO_f16.float()).reshape(b, h_q, s_q, d).sum(dim=-1, keepdim=True)
        p = torch.pow(2.0, s_raw * (self.scale * _LOG2E) - lse * _LOG2E).nan_to_num().float()
        p_fp8 = (p * 256.0).to(_E4M3).float() * (1.0 / 256.0)
        dP = _qk(dO_dq, v_dq[:, :, start:end, :], h_k)
        dS = p * (dP - D) * self.scale
        return p_fp8, dS, dO_t_dq, k_t_dq, q_t_dq, end - start

    def explicit(self, mode: str, otype=torch.float32):
        b, h_q, h_k, s_q = self.b, self.h_q, self.h_k, self.s_q
        p_fp8, dS, dO_t_dq, k_t_dq, q_t_dq, n = self._ds_and_p()
        if mode == "1x32":
            dS_fp8, sf_ref, _, dS_fp8_t, sf_t_ref, _ = quantize_to_mxfp8(dS, b, h_q, s_q, n, block_size=32, fp8_dtype=_E4M3)
            dS_q, dS_qt = _dequant(dS_fp8, sf_ref), _dequant(dS_fp8_t, sf_t_ref)
        elif mode == "fp32":
            dS_q = dS_qt = dS
        elif mode == "tile32":
            out = torch.empty_like(dS)
            inv = torch.tensor(1.0 / _E4M3_MAX, dtype=torch.float32)
            for bi in range(b):
                for hi in range(h_q):
                    for tq in range(0, s_q, 32):
                        for tk in range(0, n, 32):
                            tile = dS[bi, hi, tq : tq + 32, tk : tk + 32]
                            e = e8m0_ceil(tile.abs().amax().reshape(1) * inv)  # one E8M0 byte per 32x32 tile
                            codes = (tile * exp2_rcp(e)).clamp(-_E4M3_MAX, _E4M3_MAX).to(_E4M3)
                            out[bi, hi, tq : tq + 32, tk : tk + 32] = codes.float() * e8m0_to_float(e)
            dS_q = dS_qt = out
        else:
            raise ValueError(mode)
        dV = _kv_reduce(p_fp8, dO_t_dq, h_k)
        dQ = torch.zeros((b, h_q, s_q, self.d), dtype=torch.float32) + _pv(dS_q, k_t_dq, h_k)
        dK = _kv_reduce(dS_qt, q_t_dq, h_k)
        return dQ.to(otype).float(), dK.to(otype).float(), dV.to(otype).float()


@pytest.fixture(scope="module")
def prob():
    return _Problem()


def _assert_bitwise(got, want, tag):
    for g, w, name in zip(got, want, ("dQ", "dK", "dV")):
        assert torch.equal(g, w), f"{tag}: {name} differs -- max |diff| {(g - w).abs().max().item()} over {int((g != w).sum().item())} elements"


def test_default_is_todays_1x32_quantization_bitwise(prob):
    """The default (no ``quantize_ds`` passed) == ``quantize_ds=True`` == an independent spelling of the 1x32 both-ways
    dS quantization -- bitwise, in fp32 output and in the default bf16 output."""
    _assert_bitwise(prob.oracle(), prob.oracle(quantize_ds=True), "default vs quantize_ds=True")
    _assert_bitwise(prob.oracle(), prob.explicit("1x32"), "default vs the explicit 1x32 recomputation")
    _assert_bitwise(prob.oracle(otype=torch.bfloat16), prob.explicit("1x32", otype=torch.bfloat16), "default (bf16 out) vs explicit 1x32")


def test_false_holds_ds_in_fp32_like_the_fp8_suites_recipe(prob):
    """``quantize_ds=False`` == the fp32-dS recipe (``fp8_ref.compute_ref_backward(quantize_ds=False)``: dS enters the dQ / dK
    products unrounded) bitwise; it DIFFERS from the default in dQ and dK (the dS rounding is visible) and leaves dV alone."""
    got = prob.oracle(quantize_ds=False)
    _assert_bitwise(got, prob.explicit("fp32"), "quantize_ds=False vs the fp32-dS recipe")
    dq_t, dk_t, dv_t = prob.oracle(quantize_ds=True)
    assert not torch.equal(got[0], dq_t) and not torch.equal(got[1], dk_t), "the dS rounding must be visible in dQ / dK"
    assert torch.equal(got[2], dv_t), "dV does not depend on the dS treatment"


def test_tile_32x32_is_one_e8m0_per_tile_along_both_orientations(prob):
    """``quantize_ds=(32, 32)`` == a per-tile loop (amax over the 32 x 32 tile -> ``e8m0_ceil(amax * fp32(1/448))`` ->
    ``exp2_rcp`` -> clamp -> e4m3 -> dequant) feeding the SAME dequantized dS into dQ and dK -- bitwise; it differs from
    the 1x32 default (coarser) and from fp32."""
    got = prob.oracle(quantize_ds=(32, 32))
    _assert_bitwise(got, prob.explicit("tile32"), "quantize_ds=(32, 32) vs the explicit tile loop")
    dq_t, dk_t, dv_t = prob.oracle(quantize_ds=True)
    dq_f, dk_f, _ = prob.oracle(quantize_ds=False)
    assert not torch.equal(got[0], dq_t) and not torch.equal(got[1], dk_t), "a 32x32 tile scale differs from the 1x32 blocks"
    assert not torch.equal(got[0], dq_f) and not torch.equal(got[1], dk_f), "a 32x32 tile scale differs from fp32 dS"
    assert torch.equal(got[2], dv_t)
    # the list spelling of the tile is accepted too (JSON / parametrize round trips)
    _assert_bitwise(prob.oracle(quantize_ds=[32, 32]), got, "quantize_ds=[32, 32] vs (32, 32)")


@pytest.mark.parametrize("bad", [(16, 32), (32,), "1x32", 32, (32, 32, 32), None])
def test_other_values_are_refused(prob, bad):
    with pytest.raises(ValueError, match=r"quantize_ds must be True .* False .* or \(32, 32\)|only tile-quantized dS arm"):
        prob.oracle(quantize_ds=bad)


def test_tile_arm_pads_a_sequence_that_is_not_a_multiple_of_32():
    """s_q = 40 is not a multiple of 32: the oracle zero-pads the tile grid, quantizes, slices back.  Zeros never raise a
    tile's amax, so the partial (8 x 32) tiles quantize exactly as the explicit loop over the UNPADDED dS does -- bitwise."""
    p = _Problem(seed=3, s_q=40)
    got = p.oracle(quantize_ds=(32, 32))
    _assert_bitwise(got, p.explicit("tile32"), "quantize_ds=(32, 32) at s_q = 40 vs the explicit partial-tile loop")
    assert got[0].shape == (1, 2, 40, 128) and got[1].shape == (1, 1, 128, 128)
