# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-test for ``sdpa.fp8.assert_close_fp8_grad``'s fp8-flip attribution.

One flipped fp8 intermediate (a P or dS value the kernel and the reference rounded to neighbouring codes)
moves ONE output d-row by ``(c_alt - c_ref) * descale * gain * operand_row``.  The sm107 212-SM CI lane
produced exactly that on test_mhas_v2 fp8_bwd test310 (dK row +-0.5 from a negative-score q row of
amplitude 8), above the fixed ``4 * atol`` row cap that was calibrated on amplitude-4 rows.

The cap is lifted only on EVIDENCE from the reference's own intermediates (``fp8_ref.compute_ref`` /
``compute_ref_backward`` with ``return_intermediates=``): the position must be VALID (unmasked) and inside
the row's own reduction (same batch, a q head of the same GQA group), the reference's scaled fp32 value must
sit at the rounding midpoint of two ADJACENT codes of the intermediate's fp8 dtype, and that one flip must
reproduce the row -- within the ordinary tolerance, per element within the OUTPUT's own rounding (both sides
are dequantized output codes, each rounded once), and as a whole (the least-squares number of flips fitted to
the row is 1 within 1/2).  A power-of-two multiple of an operand row proves nothing by itself -- the negative
controls below are the cases that fit such a scalar-multiple test and are not flips: a masked key under a
causal mask, a row from another batch, a row from a head outside the GQA group, two identical rows each one
flip away, a nominal e4m3 step of 8192 (the largest adjacent e4m3 spacing is 32), an intermediate away from
any midpoint, a step of two / half / one-and-a-half spacings, a 1.2x step, and two-and-a-half or three flips
on a row of large gradients -- where ``rtol * |expected|`` is itself two flips wide, so the ordinary
tolerance alone would take three flips for one.

Every problem here is built by the reference itself on fp8-quantized sparse-int inputs (b=2, h_q=4,
h_kv=2, s=16, d=32; e4m3 and e5m2; the suite's attn_scale 1/8 and negative-score q rows of amplitude 8),
so ``valid``, ``p_scaled`` / ``ds_scaled`` and ``gain`` are real.  V takes 16 x {0, +-1, +-2} -- exact in
both fp8 dtypes like all the suite's data -- because one flip's effect, |dS| x |q| / 8 for dK and
P x |v| / 8 for O, scales with the dP = dO.V range and with |v|; at |v| <= 2 no O flip ever clears the cap
(as in the suite).  A midpoint is reached by NUDGING a reference input -- the backward's Stats row (the
kernel's LSE is a reference input, and dS is proportional to exp(S - lse)) or the forward's ``s_scale`` --
by less than half a code spacing, onto the midpoint of the value's OWN two adjacent codes, so the rest of
the problem barely moves.  Candidate positions are ranked by how VISIBLE the flip is (elements the shift
pushes outside the ordinary tolerance: it must beat the 1-element budget and the row cap) and must be the
only position of the row's reduction that could claim the flip.  No fp8 kernel is needed.
"""

import math

import cudnn
import pytest
import torch

from sdpa.fp8 import _MIDPOINT_WINDOW, _fp8_codes_around, assert_close_fp8_grad
from sdpa.fp8_ref import compute_ref, compute_ref_backward, gqa_kv_head
from sdpa.helpers import create_sparse_int_tensor, get_fp8_descale_factor, get_fp8_scale_factor, inject_negative_score_rows

pytestmark = pytest.mark.L0

B, HQ, HKV, S, D = 2, 4, 2, 16, 32
G = HQ // HKV
ATTN = 0.125
RTOL = 0.2
DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
IDS = ["e4m3", "e5m2"]
SEEDS = range(32)  # the searches break at the first construction that works; e4m3 half-flip controls need seed 10
# assert_close_fp8_grad's element budget is max(1, numel * 1e-5) = 1 element at this size: a visible flip must touch more.
MIN_VISIBLE = 2


def _atol(itype):
    # exec_sdpa_fp8: "E5M2 is less precise than E4M3, so its P quantization needs one wider step."
    return 0.125 if itype == torch.float8_e5m2 else 0.08


def _cap(itype):
    return 4 * _atol(itype)


def _deq(t, amax, itype):
    """What the call sites compare: the fp8 output dequantized with its amax-derived descale."""
    return t.float() * get_fp8_descale_factor(amax, itype)


def _requant(x, amax, itype):
    """The kernel's output path for a deviated row: fp32 -> the output fp8 code (RNE, the amax-derived scale) -> the same
    dequantized grid ``_deq`` puts the reference on.  A flip half an output code wide then shows as 0 or as a whole code."""
    return (x * get_fp8_scale_factor(amax, itype)).to(itype).float() * get_fp8_descale_factor(amax, itype)


def _bit_neighbours(x, itype):
    """(c_ref, c_alt, u) for ``x`` from the CODE BIT PATTERN -- an oracle independent of ``_fp8_codes_around``.

    round(x) is torch's own conversion; the adjacent code on x's side is code +- 1 in the sign-magnitude encoding
    (valid while it stays finite and keeps the sign bit); u = 0 where x is exactly a code or nothing is adjacent."""
    c = x.to(itype)
    mag = c.float().abs()
    code = c.view(torch.uint8).int()
    nb = torch.where(x.abs() > mag, code + 1, code - 1)
    alt = nb.clamp(0, 255).to(torch.uint8).view(itype).float().abs()
    possible = (x.abs() != mag) & torch.isfinite(alt) & torch.isfinite(mag) & (nb >= 0) & (nb <= 255) & ((nb & 0x80) == (code & 0x80))
    sign = torch.where(x < 0, -1.0, 1.0)
    c_ref = sign * mag
    c_alt = torch.where(possible, sign * alt, c_ref)
    return c_ref, c_alt, torch.where(possible, c_alt - c_ref, torch.zeros_like(c_ref))


def _scalar_codes(x, itype):
    c_ref, c_alt, u = (t.item() for t in _bit_neighbours(torch.as_tensor([float(x)], device="cuda"), itype))
    return c_ref, c_alt, u


class _Problem:
    """A dense BSHD fp8 SDPA problem prepared the way exec_sdpa_fp8 prepares one: sparse-int data, power-of-two
    per-tensor scales, s_scale for amax 1, the reference's own O / stats feeding the backward."""

    def __init__(self, itype, q, k, v, dO, *, causal=False, rescale_threshold=4.0):
        self.itype, self.q, self.k, self.v, self.dO = itype, q, k, v, dO
        self.causal, self.rescale_threshold = causal, rescale_threshold
        dev = q.device

        def quant(t):
            return (t * get_fp8_scale_factor(t.abs().max().item(), itype)).to(itype)

        def descale(t):
            return torch.tensor([get_fp8_descale_factor(t.abs().max().item(), itype)], device=dev)

        self.q8, self.k8, self.v8, self.dO8 = quant(q), quant(k), quant(v), quant(dO)
        self.q_descale, self.k_descale, self.v_descale, self.dO_descale = descale(q), descale(k), descale(v), descale(dO)
        for t, t8, ds in ((q, self.q8, self.q_descale), (k, self.k8, self.k_descale), (v, self.v8, self.v_descale), (dO, self.dO8, self.dO_descale)):
            assert torch.equal(t8.float() * ds, t), "the operands must be EXACT in fp8: the checker predicts a flip's effect from the dequantized operand"
        self.s_scale = torch.tensor([get_fp8_scale_factor(1.0, itype)], device=dev)
        self.s_descale = 1.0 / self.s_scale

    def _mask(self):
        return dict(right_bound=0, diag_align=cudnn.diagonal_alignment.TOP_LEFT) if self.causal else {}

    def forward(self, return_intermediates=False, quantize_o=True):
        return compute_ref(
            self.q8,
            self.k8,
            self.v8,
            attn_scale=ATTN,
            q_descale=self.q_descale,
            k_descale=self.k_descale,
            v_descale=self.v_descale,
            s_scale=self.s_scale,
            s_descale=self.s_descale,
            torch_itype=self.itype,
            torch_otype=self.itype,
            rescale_threshold=self.rescale_threshold,
            quantize_o=quantize_o,
            return_intermediates=return_intermediates,
            **self._mask(),
        )

    def backward(self, stats, return_intermediates=False):
        o8, _, o_amax = self.forward()[:3]
        return compute_ref_backward(
            self.q8,
            self.k8,
            self.v8,
            o8,
            self.dO8,
            attn_scale=ATTN,
            q_descale=self.q_descale,
            k_descale=self.k_descale,
            v_descale=self.v_descale,
            s_scale=self.s_scale,
            s_descale=self.s_descale,
            torch_itype=self.itype,
            o_descale=torch.tensor([get_fp8_descale_factor(o_amax, self.itype)], device=self.q.device),
            dO_descale=self.dO_descale,
            torch_otype=self.itype,
            stats=stats,
            return_intermediates=return_intermediates,
            **self._mask(),
        )

    def bwd_intermediates(self, stats):
        """The ``intermediates=`` callable exec_sdpa_fp8 passes: the same backward, re-run on a selection."""
        return lambda selection: self.backward(stats, return_intermediates=selection)[8]

    def fwd_intermediates(self):
        return lambda selection: self.forward(return_intermediates=selection)[3]


def _problem(itype, seed=0, *, neg_rows=False, causal=False, v_scale=16.0, edit=None):
    rng = torch.Generator(device="cuda").manual_seed(seed)
    q = create_sparse_int_tensor((B, S, HQ, D), torch.float, rng)
    k = create_sparse_int_tensor((B, S, HKV, D), torch.float, rng)
    v = create_sparse_int_tensor((B, S, HKV, D), torch.float, rng) * v_scale
    dO = create_sparse_int_tensor((B, S, HQ, D), torch.float, rng)
    if neg_rows:
        # target -64 / k_shift 2 / d 32 / attn 1/8 gives m = 8: the amplitude of test310's negative-score q rows (d192, -200).
        inject_negative_score_rows(q, k, rng, attn_scale=ATTN, head_axis=2, target=-64.0)
        assert q.abs().max().item() == 8.0
        # The injected rows share ONE vector; make them pairwise distinct (negate element n of row n: still +-8, still
        # deeply negative) so an attribution -- and the diagnostic's "closest operand row" -- has a unique answer.
        # Twins are legitimately ambiguous (either is "one flip"); control 4 builds its own pair on purpose.
        for n, (b, i, hq) in enumerate(_neg_rows(q)):
            q[b, i, hq, n] = -q[b, i, hq, n]
        assert len({tuple(q[b, i, hq].tolist()) for b, i, hq in _neg_rows(q)}) == len(_neg_rows(q))
    if edit is not None:
        edit(q, k, v, dO)
    return _Problem(itype, q, k, v, dO, causal=causal)


def _neg_rows(q):
    """(b, i, hq) of every negative-score q row (they all share one dense +-8 vector)."""
    return [tuple(r) for r in (q.abs().amax(-1) == 8.0).nonzero().tolist()]


def _nudge_ratio_ok(x0, target):
    """A nudge onto a value's own midpoint moves it by less than half a spacing -- a factor within [0.5, 2].  The one
    exception is a value that rounds to ZERO (midpoint 2^-10 / 2^-17, factor unbounded); such positions are skipped."""
    return (x0 != 0) & (target / x0 >= 0.5) & (target / x0 <= 2.0)


def _nudge_stats(prob, stats, targets):
    """Move the Stats (LSE) of row (b, hq, i) so the reference's scaled dS at (b, hq, i, j) lands on ``target``: dS is
    proportional to P = exp(S - lse) at fixed dP and D, so lse' = lse - ln(target / x0).  The kernel's Stats is a
    reference input, so the nudged reference is still THE reference of a legitimate problem."""
    inter = prob.backward(stats, return_intermediates=True)[8]
    stats = stats.clone()
    for (b, hq, i, j), target in targets:
        assert bool(inter["valid"][b, hq, i, j]), "nudge only valid positions"
        x0 = inter["ds_scaled"][b, hq, i, j].item()
        assert x0 != 0 and 0.5 <= target / x0 <= 2.0, (x0, target)
        stats[b, hq, i, 0] -= math.log(target / x0)
    return stats


def _bwd_on_own_midpoints(prob, positions):
    """Nudge each (b, hq, i, j) onto the midpoint of ITS OWN two adjacent dS codes and return (stats, backward outputs
    with intermediates, per position (x, c_ref, c_alt, u) read back from the NUDGED reference)."""
    stats0 = prob.forward()[1]
    inter0 = prob.backward(stats0, return_intermediates=True)[8]
    targets = []
    for pos in positions:
        c_ref, c_alt, u = _scalar_codes(inter0["ds_scaled"][pos].item(), prob.itype)
        assert u != 0, "the position must have an adjacent code"
        targets.append((pos, (c_ref + c_alt) / 2))
    stats = _nudge_stats(prob, stats0, targets)
    out = prob.backward(stats, return_intermediates=True)
    codes = []
    for pos, mid in targets:
        x = out[8]["ds_scaled"][pos].item()
        c_ref, c_alt, u = _scalar_codes(x, prob.itype)
        assert u != 0 and abs(x - mid) <= 1e-4 * abs(u), (x, mid, u)
        codes.append((x, c_ref, c_alt, u))
    return stats, out, codes


def _n_visible(dev, expected_row, atol):
    """How many elements of a d-row the deviation ``dev`` pushes outside the ordinary tolerance (last dim)."""
    return (dev.abs() > atol + RTOL * expected_row.abs()).sum(-1)


def _is_bad_row(actual, expected, row, atol):
    """The injected row must be outside the ordinary tolerance on more elements than the 1-element budget AND above
    the fixed row cap, or the case would be accepted by the plain budget and never reach the evidence path."""
    dev = (actual - expected)[row]
    return int(_n_visible(dev, expected[row], atol)) >= MIN_VISIBLE and dev.abs().max().item() > 4 * atol


def _decisive(dev, expected_row, atol):
    """A control's residual must show through the ordinary tolerance on at least one element, or the control would
    pass for the wrong reason."""
    return bool((dev.abs() > atol + RTOL * expected_row.abs()).any())


def _only_claimant(prob, inter, pos, u):
    """True when no OTHER position of dK row (b, j, hk)'s reduction could claim the flip: another (q head of the group,
    i') with an identical q row (the negative-score rows share one vector), a valid intermediate at a midpoint, and
    the same code step.  Such twins are legitimately indistinguishable (either one is 'one flip'), so a test that
    asserts the attribution or the rejection must not build one."""
    b, hq, i, j = pos
    hk = gqa_kv_head(hq, HQ, HKV)
    x = inter["ds_scaled"][b, hk * G : (hk + 1) * G, :, j]  # [G, S]
    c_ref, c_alt, uu = _fp8_codes_around(x, prob.itype)
    near = (uu != 0) & ((x - (c_ref + c_alt) / 2).abs() <= uu.abs() * _MIDPOINT_WINDOW)
    same_row = (prob.q[b, :, hk * G : (hk + 1) * G, :] == prob.q[b, i, hq][None, None, :]).all(-1).permute(1, 0)  # [G, S]
    claim = inter["valid"][b, hk * G : (hk + 1) * G, :, j] & near & (uu == u) & same_row
    claim[hq - hk * G, i] = False
    return not bool(claim.any())


def _ds_midpoint_case(itype, *, on_neg_row=None, flip_size=None, accept=None, rank_factor=1.0, seeds=SEEDS):
    """Search seeds and positions for a valid dS position -- on / off a negative-score row (None: any) -- nudged onto its
    own midpoint, whose single flip moves the dK row by ``flip_size`` per element (when given; both factors are powers
    of two so the comparison is exact), is the row's only claimant, and passes ``accept(prob, stats, out, pos,
    codes, dP_descale)``.  Candidates are ranked by how many dK elements the flip -- or, for a control injecting
    ``rank_factor`` x the flip, its residual ``(rank_factor - 1)`` x the flip -- pushes outside the ordinary tolerance.
    Returns (prob, stats, out, pos, codes, dP_descale)."""
    for seed in seeds:
        prob = _problem(itype, seed, neg_rows=True)
        out0 = prob.backward(prob.forward()[1], return_intermediates=True)
        dP_descale = get_fp8_descale_factor(out0[4], itype)
        inter0 = out0[8]
        x0 = inter0["ds_scaled"]
        c_ref0, c_alt0, u0 = _fp8_codes_around(x0, itype)
        neg = torch.zeros(B, HQ, S, dtype=torch.bool, device="cuda")
        for b, i, hq in _neg_rows(prob.q):
            neg[b, hq, i] = True
        rows_ok = torch.ones_like(neg) if on_neg_row is None else (neg if on_neg_row else ~neg)
        usable = inter0["valid"] & (u0 != 0) & _nudge_ratio_ok(x0, (c_ref0 + c_alt0) / 2) & rows_ok[..., None]
        # the one-flip deviation of dK row (b, j, hk(hq)) for every (b, hq, i, j), against that row's tolerance
        dev = (u0 * dP_descale)[..., None] * prob.q.permute(0, 2, 1, 3)[:, :, :, None, :]  # [b, hq, i, j, d]
        dK0 = _deq(out0[1], out0[6], itype)  # [b, j, hk, d]
        dK_of = dK0.permute(0, 2, 1, 3).repeat_interleave(G, 1)[:, :, None, :, :]  # [b, hq, 1, j, d] = dK[b, j, hk(hq)]
        n_vis = _n_visible(dev, dK_of, _atol(itype))
        usable &= (n_vis >= MIN_VISIBLE) & (dev.abs().amax(-1) > _cap(itype))
        if flip_size is not None:
            usable &= dev.abs().amax(-1) == flip_size
        if rank_factor != 1.0:
            n_vis = _n_visible((rank_factor - 1.0) * dev, dK_of, _atol(itype))
            usable &= n_vis >= 1
        score = n_vis.masked_fill(~usable, -1)
        for idx in score.flatten().argsort(descending=True)[:24].tolist():
            if score.flatten()[idx].item() < 0:
                break
            pos = tuple(int(t) for t in torch.unravel_index(torch.tensor(idx), score.shape))
            stats, out, [codes] = _bwd_on_own_midpoints(prob, [pos])
            if _only_claimant(prob, out[8], pos, codes[3]) and (accept is None or accept(prob, stats, out, pos, codes, dP_descale)):
                return prob, stats, out, pos, codes, dP_descale
    raise AssertionError("no seed / position satisfies the construction")


def _dk_flip(out, prob, pos, u, dP_descale, factor=1.0):
    """(expected, actual, row) for dK with ``factor`` x one dS flip of size ``u`` injected at (b, hq, i, j)."""
    b, hq, i, j = pos
    hk = gqa_kv_head(hq, HQ, HKV)
    expected = _deq(out[1], out[6], prob.itype)
    actual = expected.clone()
    actual[b, j, hk] += factor * u * dP_descale * prob.q[b, i, hq]
    return expected, actual, (b, j, hk)


def _accept_bad(atol, factor=1.0, decisive=True):
    """``factor`` x one flip must be a bad row; with ``decisive`` its residual against ONE flip must show through the
    ordinary tolerance (so that tolerance is what rejects it), without it the residual must hide inside that tolerance
    everywhere (so only the flip count can)."""

    def accept(prob, stats, out, pos, codes, dP_descale):
        expected, actual, row = _dk_flip(out, prob, pos, codes[3], dP_descale, factor)
        residual = (factor - 1.0) * codes[3] * dP_descale * prob.q[pos[0], pos[2], pos[1]]
        return _is_bad_row(actual, expected, row, atol) and (factor == 1.0 or _decisive(residual, expected[row], atol) == decisive)

    return accept


def _large_gradient_case(itype, factor, requantized, seeds=range(64)):
    """A midpoint dS flip on a negative-score q row whose dK row carries gradients so large that the ordinary tolerance
    ``atol + 0.2 |expected|`` swallows ``factor - 1`` flips on EVERY element -- so one flip is inside it everywhere and
    the residual of ``factor`` flips against one hides in it too -- while ``factor`` flips (exact, or requantized to the
    output grid like the kernel's row would be) are a bad row.  The verifier's seed-902 row: the ordinary tolerance on the
    residual against one flip admitted 2.8 and 3 flips there.  ``_ds_midpoint_case`` cannot find these (it ranks by
    the visibility of ONE flip); this searches for the invisibility."""
    atol = _atol(itype)
    for seed in seeds:
        prob = _problem(itype, seed, neg_rows=True)
        out0 = prob.backward(prob.forward()[1], return_intermediates=True)
        dP_descale = get_fp8_descale_factor(out0[4], itype)
        inter0 = out0[8]
        x0 = inter0["ds_scaled"]
        c_ref0, c_alt0, u0 = _fp8_codes_around(x0, itype)
        neg = torch.zeros(B, HQ, S, dtype=torch.bool, device="cuda")
        for b, i, hq in _neg_rows(prob.q):
            neg[b, hq, i] = True
        usable = inter0["valid"] & (u0 != 0) & _nudge_ratio_ok(x0, (c_ref0 + c_alt0) / 2) & neg[..., None]
        dev = (u0 * dP_descale)[..., None] * prob.q.permute(0, 2, 1, 3)[:, :, :, None, :]  # one flip on dK row (b, j, hk(hq)), [b, hq, i, j, d]
        dK_of = _deq(out0[1], out0[6], itype).permute(0, 2, 1, 3).repeat_interleave(G, 1)[:, :, None, :, :]
        usable &= _n_visible((factor - 1.0) * dev, dK_of, atol) == 0  # the residual against one flip hides everywhere
        usable &= (_n_visible(factor * dev, dK_of, atol) >= MIN_VISIBLE) & ((factor * dev).abs().amax(-1) > _cap(itype))
        for pos in usable.nonzero().tolist()[:24]:
            pos = tuple(pos)
            stats, out, [codes] = _bwd_on_own_midpoints(prob, [pos])
            if not _only_claimant(prob, out[8], pos, codes[3]):
                continue
            # the nudge moved the row a little: re-check both conditions on the reference that will be the evidence.  The
            # hiding is a property of the EXACT residual (requantized, a 3-flip deviation on a grid whose spacing is two
            # flips rounds to 2 or 4 flips, and the 4s are visible -- then the ordinary criterion is right to fire).
            expected, one, row = _dk_flip(out, prob, pos, codes[3], dP_descale)
            _, actual, _ = _dk_flip(out, prob, pos, codes[3], dP_descale, factor)
            if int(_n_visible(actual[row] - one[row], expected[row], atol)) != 0:
                continue
            if requantized:
                actual[row] = _requant(actual[row], out[6], itype)
            if _is_bad_row(actual, expected, row, atol):
                return prob, stats, out, pos, codes, dP_descale
    raise AssertionError("no seed / position gives a row whose ordinary tolerance hides the residual against one flip")


# ---------------------------------------------------------------------------------------------------------------------
# the building blocks
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("itype", DTYPES, ids=IDS)
def test_fp8_codes_around_matches_the_bit_level_neighbours(itype):
    x = torch.cat(
        [
            torch.linspace(-600, 600, 100001, device="cuda"),
            torch.linspace(-70000, 70000, 70001, device="cuda"),
            torch.tensor([0.0, 1e-6, -1e-6, 2**-10, -(2**-10), 2**-9, 25.0, 25.00008, 447.0, 448.0, 449.0, 57344.0, 60000.0], device="cuda"),
        ]
    )
    c_ref, c_alt, u = _fp8_codes_around(x, itype)
    e_ref, e_alt, e_u = _bit_neighbours(x, itype)
    assert torch.equal(c_ref, e_ref) and torch.equal(c_alt, e_alt) and torch.equal(u, e_u)
    assert torch.equal(c_ref, x.to(itype).float()), "c_ref is the reference's own conversion"
    if itype == torch.float8_e4m3fn:
        assert _scalar_codes(25.00008, itype) == (26.0, 24.0, -2.0), "the docstring's headline flip: 25.00008 -> 26, partner 24"
        assert _scalar_codes(449.0, itype)[2] == 0, "past 448 both computations saturate to the same code: no flip"
        assert _scalar_codes(447.0, itype) == (448.0, 416.0, -32.0), "the widest adjacent e4m3 spacing is 32 (416 -> 448)"
    else:
        assert _scalar_codes(60000.0, itype)[2] == 0, "e5m2 overflows to inf above 57344: no adjacent finite code"
        assert _scalar_codes(57000.0, itype) == (57344.0, 49152.0, -8192.0), "e5m2's top binade is 8192 apart"


@pytest.mark.parametrize("itype", DTYPES, ids=IDS)
def test_the_intermediates_reconstruct_the_reference_outputs(itype):
    """``p_scaled`` / ``ds_scaled`` / ``valid`` / ``gain`` are the values the reference quantized, in the reference's GQA
    mapping: the outputs rebuilt from them match the reference (O to fp32 round-off; dQ/dK/dV bit-exactly after
    requantization)."""
    prob = _problem(itype, neg_rows=True, causal=True)
    o8, stats, o_amax, inter = prob.forward(return_intermediates=True)
    o_unquant = prob.forward(quantize_o=False)[0]
    assert inter["mode"] == "full" and inter["h_q"] == HQ and inter["h_kv"] == HKV
    assert torch.equal(inter["valid"], torch.tril(torch.ones(S, S, dtype=torch.bool, device="cuda")).expand(B, HQ, S, S))
    assert inter["p_scaled"][~inter["valid"]].abs().max().item() == 0.0, "masked P = 0 by construction"
    codes = inter["p_scaled"].to(itype).float()
    v_of_hq = prob.v.permute(0, 2, 1, 3).repeat_interleave(G, 1)  # [b, hq, j, d] = V[b, j, gqa_kv_head(hq)]
    # The gain-scaled codes are no longer exactly representable in TF32. Keep
    # this tight reconstruction check independent of the CI matmul precision.
    o_rec = torch.einsum("bhij,bhjd->bhid", (codes * inter["gain"] * prob.s_descale).double(), v_of_hq.double()).permute(0, 2, 1, 3)
    assert (o_rec - o_unquant).abs().max().item() < 1e-5 * max(1.0, o_unquant.abs().max().item())

    dQ, dK, dV, _, dP_amax, dQ_amax, dK_amax, dV_amax, bi = prob.backward(stats, return_intermediates=True)
    assert bi["gain"] is None and torch.equal(bi["valid"], inter["valid"])
    ds = bi["ds_scaled"].to(itype).float()
    p = bi["p_scaled"].to(itype).float()
    dP_descale = get_fp8_descale_factor(dP_amax, itype)
    q_bhid = prob.q.permute(0, 2, 1, 3)
    dO_bhid = prob.dO.permute(0, 2, 1, 3)
    k_of_hq = prob.k.permute(0, 2, 1, 3).repeat_interleave(G, 1)
    dK_rec = torch.einsum("bhij,bhid->bhjd", ds, q_bhid).view(B, HKV, G, S, D).sum(2).permute(0, 2, 1, 3) * dP_descale
    dV_rec = torch.einsum("bhij,bhid->bhjd", p, dO_bhid).view(B, HKV, G, S, D).sum(2).permute(0, 2, 1, 3) * prob.s_descale
    dQ_rec = torch.einsum("bhij,bhjd->bhid", ds, k_of_hq).permute(0, 2, 1, 3) * dP_descale
    for rec, out, amax in ((dK_rec, dK, dK_amax), (dV_rec, dV, dV_amax), (dQ_rec, dQ, dQ_amax)):
        assert torch.equal((rec * get_fp8_scale_factor(amax, itype)).to(itype).view(torch.uint8), out.view(torch.uint8))
    assert [gqa_kv_head(h, HQ, HKV) for h in range(HQ)] == [0, 0, 1, 1], "the checker's mapping is the reference's"


@pytest.mark.parametrize("itype", DTYPES, ids=IDS)
def test_gathered_intermediates_equal_slices_of_the_full_ones(itype):
    """The q_rows / kv_cols gathers the checker asks for are the corresponding slices of the full matrices -- so the
    full matrices are never needed (b=8 h=8 s=8192 would be 17 GB each)."""
    prob = _problem(itype, neg_rows=True, causal=True)
    o8, stats, o_amax, full = prob.forward(return_intermediates=True)
    q_rows = torch.tensor([[0, 1, 3], [1, 2, 15], [1, 0, 0], [0, 3, 7]], device="cuda")
    g = prob.forward(return_intermediates={"q_rows": q_rows})[3]
    assert g["mode"] == "q_rows" and g["p_scaled"].shape == (4, S)
    for key in ("p_scaled", "valid", "gain"):
        assert torch.equal(g[key], full[key][q_rows[:, 0], q_rows[:, 1], q_rows[:, 2], :]), key
    kv_cols = torch.tensor([[0, 1, 3], [1, 0, 15], [1, 1, 0]], device="cuda")
    g = prob.forward(return_intermediates={"kv_cols": kv_cols})[3]
    assert g["mode"] == "kv_cols" and g["p_scaled"].shape == (3, G, S)
    for key in ("p_scaled", "valid", "gain"):
        for n, (b, hk, j) in enumerate(kv_cols.tolist()):
            assert torch.equal(g[key][n], full[key][b, hk * G : (hk + 1) * G, :, j]), (key, n)
    bfull = prob.backward(stats, return_intermediates=True)[8]
    bg = prob.backward(stats, return_intermediates={"kv_cols": kv_cols})[8]
    for key in ("p_scaled", "ds_scaled", "valid"):
        for n, (b, hk, j) in enumerate(kv_cols.tolist()):
            assert torch.equal(bg[key][n], bfull[key][b, hk * G : (hk + 1) * G, :, j]), (key, n)
    assert len(prob.forward()) == 3 and len(prob.backward(stats)) == 8, "the default return is untouched"
    with pytest.raises(ValueError, match="return_intermediates"):
        prob.forward(return_intermediates={"rows": q_rows})


# ---------------------------------------------------------------------------------------------------------------------
# positive: one genuine flip, at a midpoint, is accepted and attributed to its position
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("requantized", [False, True], ids=["exact", "requantized"])
@pytest.mark.parametrize("itype", DTYPES, ids=IDS)
def test_one_flipped_ds_code_at_its_own_midpoint_is_accepted_and_attributed(itype, requantized):
    """``requantized``: the deviated row goes through the output's fp8 rounding like the kernel's would, so elements whose
    output spacing is twice the flip show 0 or two flips -- inside the output-rounding bound, unbiased in the flip count."""
    prob, stats, out, pos, (x, c_ref, c_alt, u), dP_descale = _ds_midpoint_case(itype, accept=_accept_bad(_atol(itype)))
    expected, actual, row = _dk_flip(out, prob, pos, u, dP_descale)
    if requantized:
        actual[row] = _requant(actual[row], out[6], itype)
        assert _is_bad_row(actual, expected, row, _atol(itype))
    fits = assert_close_fp8_grad(
        actual,
        expected,
        _atol(itype),
        RTOL,
        tag="dK",
        keys=S,
        operand=prob.q,
        flip_unit=dP_descale,
        intermediates=prob.bwd_intermediates(stats),
        fp8_dtype=itype,
    )
    assert fits is not None and [f["position"] for f in fits] == [pos]
    assert (fits[0]["row"], fits[0]["c_ref"], fits[0]["c_alt"], fits[0]["u"], fits[0]["x"]) == (row, c_ref, c_alt, u, x)
    assert fits[0]["off_mid_spacings"] <= 1e-4 and fits[0]["operand_row"] == (pos[0], pos[2], pos[1]) and fits[0]["worst"] <= 1.0
    assert fits[0]["rounding"] <= 1.0 and abs(fits[0]["flips"] - 1.0) < 0.5 and fits[0]["n_tied"] == 0
    if not requantized:
        assert fits[0]["rounding"] == 0.0 and fits[0]["flips"] == 1.0, "an exact one-flip deviation leaves no residual at all"


def test_test310_shape_dk_row_of_half_from_a_q_row_of_amplitude_8():
    """The CI case (e4m3): dK row j uniformly +-0.5 -- one dS flip x a negative-score q row of amplitude 8, above the 0.32
    cap -- and the same flip seen through K moves dQ row i by <= 0.25, inside the cap."""
    itype = torch.float8_e4m3fn
    prob, stats, out, pos, (x, c_ref, c_alt, u), dP_descale = _ds_midpoint_case(itype, on_neg_row=True, flip_size=0.5, accept=_accept_bad(_atol(itype)))
    b, hq, i, j = pos
    hk = gqa_kv_head(hq, HQ, HKV)
    dK_exp, dK_act, row = _dk_flip(out, prob, pos, u, dP_descale)
    assert torch.equal((dK_act - dK_exp)[row].abs(), torch.full((D,), 0.5, device="cuda")), "dK row moves by exactly +-0.5"
    dQ_exp = _deq(out[0], out[5], itype)
    dQ_act = dQ_exp.clone()
    dQ_act[b, i, hq] += u * dP_descale * prob.k[b, j, hk]
    assert 0 < (dQ_act - dQ_exp).abs().max().item() <= 0.25
    bwd = prob.bwd_intermediates(stats)
    fits = assert_close_fp8_grad(dK_act, dK_exp, _atol(itype), RTOL, tag="dK", keys=S, operand=prob.q, flip_unit=dP_descale, intermediates=bwd, fp8_dtype=itype)
    assert fits is not None and [f["position"] for f in fits] == [pos] and abs(fits[0]["step"]) * 8 == 0.5
    # dQ is inside the cap: the plain row budget accepts it, no evidence is consulted
    assert (
        assert_close_fp8_grad(dQ_act, dQ_exp, _atol(itype), RTOL, tag="dQ", keys=S, operand=prob.k, flip_unit=dP_descale, intermediates=bwd, fp8_dtype=itype)
        is None
    )


@pytest.mark.parametrize("itype", DTYPES, ids=IDS)
def test_one_flipped_p_code_is_accepted_through_the_forward_normalization(itype):
    """O = sum_j quant(P) * gain * s_descale * V[j]: the flip's effect on O carries 2^rescale_threshold * exp(m_block -
    m_final) / l, which ``gain`` supplies.  (The key with the row maximum has P_unnormalized = 1 exactly, x = s_scale *
    2^-thr, a code: a flip can only sit on another key, which is why |v| = 32 is needed to clear the cap.)"""
    for seed in SEEDS:
        prob = _problem(itype, seed, neg_rows=True)
        o8, stats, o_amax, inter = prob.forward(return_intermediates=True)
        x_all = inter["p_scaled"]
        c_ref_all, c_alt_all, u_all = _fp8_codes_around(x_all, itype)
        usable = inter["valid"] & (u_all != 0) & _nudge_ratio_ok(x_all, (c_ref_all + c_alt_all) / 2)
        # the one-flip deviation of O row (b, i, hq) for every (b, hq, i, j), against that row's tolerance
        step = u_all * prob.s_descale * inter["gain"]  # [b, hq, i, j]
        dev = step[..., None] * prob.v.permute(0, 2, 1, 3).repeat_interleave(G, 1)[:, :, None, :, :]  # x V[b, j, hk(hq)]
        o_exp = _deq(o8, o_amax, itype).permute(0, 2, 1, 3)[:, :, :, None, :]  # [b, hq, i, 1, d]
        n_vis = _n_visible(dev, o_exp, _atol(itype))
        score = n_vis.masked_fill(~(usable & (n_vis >= MIN_VISIBLE) & (dev.abs().amax(-1) > _cap(itype))), -1)
        found = False
        for idx in score.flatten().argsort(descending=True)[:6].tolist():
            if score.flatten()[idx].item() < 0:
                break
            pos = tuple(int(t) for t in torch.unravel_index(torch.tensor(idx), score.shape))
            b, hq, i, j = pos
            c_ref, c_alt, _ = _scalar_codes(x_all[pos].item(), itype)
            trial = _problem(itype, seed, neg_rows=True)
            # x = p * s_scale * 2^-thr: land the position on the midpoint of its own two codes (a nudge of < half a spacing)
            trial.s_scale = trial.s_scale * ((c_ref + c_alt) / 2 / x_all[pos].item())
            trial.s_descale = 1.0 / trial.s_scale
            o8t, _, o_amax_t, inter_t = trial.forward(return_intermediates=True)
            x = inter_t["p_scaled"][pos].item()
            c_ref, c_alt, u = _scalar_codes(x, itype)
            assert u != 0 and abs(x - (c_ref + c_alt) / 2) <= 1e-4 * abs(u)
            hk = gqa_kv_head(hq, HQ, HKV)
            expected = _deq(o8t, o_amax_t, itype)
            actual = expected.clone()
            actual[b, i, hq] += u * trial.s_descale.item() * inter_t["gain"][pos].item() * trial.v[b, j, hk]
            if _is_bad_row(actual, expected, (b, i, hq), _atol(itype)):
                found = True
                break
        if found:
            break
    else:
        raise AssertionError("no seed gives a forward flip above the cap")
    fits = assert_close_fp8_grad(
        actual,
        expected,
        _atol(itype),
        RTOL,
        tag="O",
        keys=S,
        operand=trial.v,
        flip_unit=trial.s_descale.item(),
        intermediates=trial.fwd_intermediates(),
        fp8_dtype=itype,
    )
    assert fits is not None and [f["position"] for f in fits] == [pos]
    assert fits[0]["gain"] == inter_t["gain"][pos].item() and fits[0]["operand_row"] == (b, j, hk) and (fits[0]["c_ref"], fits[0]["c_alt"]) == (c_ref, c_alt)


# ---------------------------------------------------------------------------------------------------------------------
# negative: the five review controls -- each fits a scalar-multiple test and each must be rejected
# ---------------------------------------------------------------------------------------------------------------------


def _rejected(capsys, **kw):
    with pytest.raises(AssertionError):
        assert_close_fp8_grad(**kw)
    out = capsys.readouterr().out
    assert "is NOT one flipped fp8 intermediate" in out, out
    return out


@pytest.mark.parametrize("itype", DTYPES, ids=IDS)
def test_control_1_a_masked_key_under_a_causal_mask_is_rejected(itype, capsys):
    """Zero Q/K logits, V[0] = 0, causal: O row 0 is exactly 0.  Adding 0.5 * V[1] is a power-of-two multiple of an
    operand row (2^5 x s_descale) -- but key 1 is MASKED for row 0 (P[0, 1] = 0 by construction), so nothing there
    rounded, and no valid position explains the row."""

    def edit(q, k, v, dO):
        q.zero_()
        k.zero_()
        v[:, 0] = 0.0
        v[:, 1] = torch.where(torch.arange(D, device=v.device) % 2 == 0, 2.0, -2.0)

    prob = _problem(itype, causal=True, edit=edit)
    o8, stats, o_amax, inter = prob.forward(return_intermediates=True)
    assert not inter["valid"][:, :, 0, 1].any() and inter["p_scaled"][:, :, 0, 1].abs().max().item() == 0.0
    expected = _deq(o8, o_amax, itype)
    assert expected[:, 0].abs().max().item() == 0.0, "the first output row is 0"
    actual = expected.clone()
    b, hq = 1, 3
    actual[b, 0, hq] += 0.5 * prob.v[b, 1, gqa_kv_head(hq, HQ, HKV)]
    assert _is_bad_row(actual, expected, (b, 0, hq), _atol(itype))
    out = _rejected(
        capsys,
        actual=actual,
        expected=expected,
        atol=_atol(itype),
        rtol=RTOL,
        tag="O",
        keys=S,
        operand=prob.v,
        flip_unit=prob.s_descale.item(),
        intermediates=prob.fwd_intermediates(),
        fp8_dtype=itype,
    )
    assert f"(b, hq, i, j)={(b, hq, 0, 1)} is masked" in out


@pytest.mark.parametrize("itype", DTYPES, ids=IDS)
def test_control_2_a_row_from_another_batch_is_rejected(itype, capsys):
    """Batch 1 has Q = 0, so its dK is exactly 0.  A dK row of batch 1 equal to (2^k x dP_descale) x a batch-0 q row is
    a scalar multiple of an operand row -- of a row batch 1 never reduces over."""

    def edit(q, k, v, dO):
        q[1] = 0.0
        q[0, 5, 0] = torch.where(torch.arange(D, device=q.device) % 3 == 0, -8.0, 8.0)

    prob = _problem(itype, edit=edit)
    stats = prob.forward()[1]
    dQ, dK, dV, _, dP_amax, dQ_amax, dK_amax, dV_amax = prob.backward(stats)
    dP_descale = get_fp8_descale_factor(dP_amax, itype)
    expected = _deq(dK, dK_amax, itype)
    assert expected[1].abs().max().item() == 0.0
    actual = expected.clone()
    actual[1, 9, 1] += (1.0 / 8 / dP_descale) * dP_descale * prob.q[0, 5, 0]  # 2^k x flip_unit x an operand row: 1.0 per element
    assert (actual - expected).abs().max().item() == 1.0 > _cap(itype)
    out = _rejected(
        capsys,
        actual=actual,
        expected=expected,
        atol=_atol(itype),
        rtol=RTOL,
        tag="dK",
        keys=S,
        operand=prob.q,
        flip_unit=dP_descale,
        intermediates=prob.bwd_intermediates(stats),
        fp8_dtype=itype,
    )
    assert "row (0, 5, 0)" in out and "that row is in batch 0" in out and "reduces over batch 1 only" in out


@pytest.mark.parametrize("itype", DTYPES, ids=IDS)
def test_control_3_a_q_head_outside_the_gqa_group_is_rejected(itype, capsys):
    """kv head 1 serves q heads 2 and 3 (``gqa_kv_head``).  With those zeroed, dK[:, :, 1] is exactly 0; a dK row there
    equal to a multiple of a q-head-0 row is a scalar multiple of an operand row -- from a head outside the group."""

    def edit(q, k, v, dO):
        q[:, :, 2:4] = 0.0
        q[1, 5, 0] = torch.where(torch.arange(D, device=q.device) % 3 == 0, -8.0, 8.0)

    prob = _problem(itype, edit=edit)
    stats = prob.forward()[1]
    dQ, dK, dV, _, dP_amax, dQ_amax, dK_amax, dV_amax = prob.backward(stats)
    dP_descale = get_fp8_descale_factor(dP_amax, itype)
    expected = _deq(dK, dK_amax, itype)
    assert expected[:, :, 1].abs().max().item() == 0.0
    actual = expected.clone()
    actual[1, 9, 1] += (1.0 / 8 / dP_descale) * dP_descale * prob.q[1, 5, 0]
    assert (actual - expected).abs().max().item() == 1.0 > _cap(itype)
    out = _rejected(
        capsys,
        actual=actual,
        expected=expected,
        atol=_atol(itype),
        rtol=RTOL,
        tag="dK",
        keys=S,
        operand=prob.q,
        flip_unit=dP_descale,
        intermediates=prob.bwd_intermediates(stats),
        fp8_dtype=itype,
    )
    assert "row (1, 5, 0)" in out and "that row is q head 0, outside the GQA group of kv head 1 (q heads 2..3)" in out


@pytest.mark.parametrize("itype", DTYPES, ids=IDS)
def test_control_4_two_identical_rows_each_one_flip_away_are_rejected(itype, capsys):
    """Two q rows with the SAME +-8 vector (and the same dO row, so identical dS rows), same batch and GQA group, both
    sitting on a dS midpoint with the same code spacing u at the same key j, each flipped: the dK row moves by 2u x row
    -- a power-of-two multiple of one operand row, which the shape-only test accepted.  The codes at either position
    are exactly u apart, so one flip predicts half the deviation and the residual is a whole flip."""

    pair = []

    def edit(q, k, v, dO):
        b, i1, hq1 = _neg_rows(q)[0]
        hq2, i2 = hq1 + 1, (i1 + 5) % S  # the other q head of the same GQA group (group = {2g, 2g+1}), another query
        q[b, i2, hq2] = q[b, i1, hq1]
        dO[b, i2, hq2] = dO[b, i1, hq1]
        pair[:] = [(b, i1, hq1), (b, i2, hq2)]

    for seed in SEEDS:
        prob = _problem(itype, seed, neg_rows=True, edit=edit)
        (b, i1, hq1), (_, i2, hq2) = pair
        assert torch.equal(prob.q[b, i1, hq1], prob.q[b, i2, hq2]) and gqa_kv_head(hq1, HQ, HKV) == gqa_kv_head(hq2, HQ, HKV)
        hk = gqa_kv_head(hq1, HQ, HKV)
        stats0 = prob.forward()[1]
        out0 = prob.backward(stats0, return_intermediates=True)
        dP_descale = get_fp8_descale_factor(out0[4], itype)
        x1, x2 = out0[8]["ds_scaled"][b, hq1, i1], out0[8]["ds_scaled"][b, hq2, i2]
        assert torch.equal(x1, x2), "identical q and dO rows give identical dS rows"
        c1, a1, u1 = _fp8_codes_around(x1, itype)
        usable = (u1 != 0) & out0[8]["valid"][b, hq1, i1] & _nudge_ratio_ok(x1, (c1 + a1) / 2)
        # rank the shared key by the visibility of the DOUBLE flip on dK row (b, j, hk), requiring the single to show too
        dK0 = _deq(out0[1], out0[6], itype)[b, :, hk]  # [j, d]
        single = (u1 * dP_descale)[:, None] * prob.q[b, i1, hq1][None, :]
        n_vis = _n_visible(2 * single, dK0, _atol(itype))
        usable &= (n_vis >= MIN_VISIBLE) & ((2 * single).abs().amax(-1) > _cap(itype)) & (single.abs() > _atol(itype) + RTOL * dK0.abs()).any(-1)
        if not bool(usable.any()):
            continue
        j = int(n_vis.masked_fill(~usable, -1).argmax())
        stats, out, codes = _bwd_on_own_midpoints(prob, [(b, hq1, i1, j), (b, hq2, i2, j)])
        u1, u2 = codes[0][3], codes[1][3]
        assert u1 == u2, "identical values nudge to the same midpoint and round the same way"
        expected = _deq(out[1], out[6], itype)
        actual = expected.clone()
        actual[b, j, hk] += (u1 * prob.q[b, i1, hq1] + u2 * prob.q[b, i2, hq2]) * dP_descale
        if _is_bad_row(actual, expected, (b, j, hk), _atol(itype)) and _decisive(u1 * dP_descale * prob.q[b, i1, hq1], expected[b, j, hk], _atol(itype)):
            break
    else:
        raise AssertionError("no seed gives two identical rows on a common midpoint")
    assert (actual - expected)[b, j, hk].abs().max().item() == 2 * abs(u1) * dP_descale * 8
    out_txt = _rejected(
        capsys,
        actual=actual,
        expected=expected,
        atol=_atol(itype),
        rtol=RTOL,
        tag="dK",
        keys=S,
        operand=prob.q,
        flip_unit=dP_descale,
        intermediates=prob.bwd_intermediates(stats),
        fp8_dtype=itype,
    )
    assert f"a step of {u1:+g} x flip_unit x gain, which predicts the row" in out_txt and "not the fitted" in out_txt


def test_control_5_a_nominal_e4m3_step_of_8192_is_rejected(capsys):
    """8192 x dP_descale x q_row is 2^13 x flip_unit -- a power of two the shape-only test allowed (e5m2 reaches 2^13),
    but no two adjacent e4m3 codes are more than 32 apart, so no position's codes can predict it."""
    itype = torch.float8_e4m3fn
    prob = _problem(itype, neg_rows=True)
    stats = prob.forward()[1]
    dQ, dK, dV, _, dP_amax, dQ_amax, dK_amax, dV_amax = prob.backward(stats)
    dP_descale = get_fp8_descale_factor(dP_amax, itype)
    b, i, hq = _neg_rows(prob.q)[0]
    hk = gqa_kv_head(hq, HQ, HKV)
    expected = _deq(dK, dK_amax, itype)
    actual = expected.clone()
    actual[b, 4, hk] += 8192.0 * dP_descale * prob.q[b, i, hq]
    out = _rejected(
        capsys,
        actual=actual,
        expected=expected,
        atol=_atol(itype),
        rtol=RTOL,
        tag="dK",
        keys=S,
        operand=prob.q,
        flip_unit=dP_descale,
        intermediates=prob.bwd_intermediates(stats),
        fp8_dtype=itype,
    )
    assert "(= 2^13.00 x flip_unit" in out


# ---------------------------------------------------------------------------------------------------------------------
# negative: the midpoint and the step are checked at the position, not fitted
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("itype", DTYPES, ids=IDS)
def test_an_intermediate_away_from_the_midpoint_is_rejected(itype, capsys):
    """The right position, the right operand row, the right step -- but the reference's value sits a quarter of a code
    spacing from the midpoint: both computations round it the same way, so it cannot have flipped."""
    prob, stats_mid, out, pos, (x, c_ref, c_alt, u), dP_descale = _ds_midpoint_case(itype, on_neg_row=True, accept=_accept_bad(_atol(itype)))
    # from the midpoint, move a quarter spacing towards c_ref: the same two codes, no longer a tie
    stats = _nudge_stats(prob, stats_mid, [(pos, c_ref + 0.25 * (c_alt - c_ref))])
    out = prob.backward(stats, return_intermediates=True)
    x = out[8]["ds_scaled"][pos].item()
    assert _scalar_codes(x, itype)[:2] == (c_ref, c_alt) and 0.2 < abs(x - (c_ref + c_alt) / 2) / abs(u) < 0.3
    expected, actual, row = _dk_flip(out, prob, pos, u, dP_descale)
    assert _is_bad_row(actual, expected, row, _atol(itype))
    out_txt = _rejected(
        capsys,
        actual=actual,
        expected=expected,
        atol=_atol(itype),
        rtol=RTOL,
        tag="dK",
        keys=S,
        operand=prob.q,
        flip_unit=dP_descale,
        intermediates=prob.bwd_intermediates(stats),
        fp8_dtype=itype,
    )
    assert f"(b, hq, i, j)={pos}" in out_txt and "code spacings from the midpoint" in out_txt and f"a flip needs <= {_MIDPOINT_WINDOW:g}" in out_txt


@pytest.mark.parametrize(
    "itype, factor",
    [(torch.float8_e4m3fn, 2.0), (torch.float8_e4m3fn, 0.5), (torch.float8_e5m2, 2.0), (torch.float8_e5m2, 0.5), (torch.float8_e5m2, 1.2)],
    ids=["e4m3-2u", "e4m3-0.5u", "e5m2-2u", "e5m2-0.5u", "e5m2-1.2u"],
)
def test_a_step_that_is_not_the_adjacent_code_spacing_is_rejected(itype, factor, capsys):
    """Right position, on its midpoint, right operand row -- but the deviation is ``factor`` x the spacing the codes
    there actually have, and the residual against ONE flip shows through the ordinary tolerance.  2u and u/2 are powers
    of two (the shape-only test accepted them); 1.2u sat inside its 1.25x slack.  1.2u is e5m2-only: a fifth of an
    e4m3 flip hides inside ``atol + rtol * |dK|`` AND inside the output's own rounding on every element of this data,
    and the flip count 1.2 is inside the 1 +- 1/2 band -- a deviation between half and one-and-a-half flips of the
    same intermediate is not observable through fp8 output codes (``assert_close_fp8_grad``'s docstring says so);
    e5m2's 25 % code spacing makes the same fifth visible."""
    prob, stats, out, pos, (x, c_ref, c_alt, u), dP_descale = _ds_midpoint_case(
        itype, on_neg_row=True, accept=_accept_bad(_atol(itype), factor), rank_factor=factor
    )
    expected, actual, row = _dk_flip(out, prob, pos, u, dP_descale, factor)
    out_txt = _rejected(
        capsys,
        actual=actual,
        expected=expected,
        atol=_atol(itype),
        rtol=RTOL,
        tag="dK",
        keys=S,
        operand=prob.q,
        flip_unit=dP_descale,
        intermediates=prob.bwd_intermediates(stats),
        fp8_dtype=itype,
    )
    assert f"(b, hq, i, j)={pos}" in out_txt and f"a step of {u:+g} x flip_unit x gain, which predicts the row" in out_txt and "not the fitted" in out_txt


@pytest.mark.parametrize("requantized", [False, True], ids=["exact", "requantized"])
@pytest.mark.parametrize("itype", DTYPES, ids=IDS)
def test_one_and_a_half_flips_hidden_inside_the_tolerance_are_rejected_by_the_flip_count(itype, requantized, capsys):
    """Right position, on its midpoint, right operand row; the deviation is 1.5 x the flip and its residual against one
    flip -- half a flip per element -- hides inside the ordinary tolerance AND inside the output's rounding on every
    element.  Only the least-squares flip count sees it: 1.5, on the edge of the 1 +- 1/2 band, out.  (Requantized, the
    rounding noise moves the count by ~1/sqrt(d) either way; the band is what keeps a genuine flip's 1 +- noise in.)"""
    prob, stats, out, pos, (x, c_ref, c_alt, u), dP_descale = _ds_midpoint_case(itype, on_neg_row=True, accept=_accept_bad(_atol(itype), 1.5, decisive=False))
    expected, actual, row = _dk_flip(out, prob, pos, u, dP_descale, 1.5)
    if requantized:
        actual[row] = _requant(actual[row], out[6], itype)
        assert _is_bad_row(actual, expected, row, _atol(itype))
    out_txt = _rejected(
        capsys,
        actual=actual,
        expected=expected,
        atol=_atol(itype),
        rtol=RTOL,
        tag="dK",
        keys=S,
        operand=prob.q,
        flip_unit=dP_descale,
        intermediates=prob.bwd_intermediates(stats),
        fp8_dtype=itype,
    )
    assert f"(b, hq, i, j)={pos}" in out_txt and f"a step of {u:+g} x flip_unit x gain, which" in out_txt and "not the fitted" in out_txt
    assert ("flips of itself (one flip fits within 1 +- 0.5" in out_txt) or ("x the output's own rounding" in out_txt), out_txt
    if not requantized:
        assert "fits the row as 1.50 flips of itself" in out_txt


@pytest.mark.parametrize("requantized", [False, True], ids=["exact", "requantized"])
@pytest.mark.parametrize("factor", [2.5, 3.0], ids=["2.5u", "3u"])
@pytest.mark.parametrize("itype", DTYPES, ids=IDS)
def test_several_flips_on_a_row_of_large_gradients_are_rejected(itype, factor, requantized, capsys):
    """The verifier's bypass (seed 902, e4m3): a dK row with |expected| 5..13 has an ordinary tolerance of 1.1..2.7 per
    element, two to five flips wide, so 2.8 and 3 flips were accepted as ONE with a residual inside that tolerance.  On
    such a row one flip is inside the tolerance everywhere (the plain path returns None without consulting any
    evidence) and so is the residual of ``factor`` flips against one -- the ordinary criterion cannot reject them; the
    output-rounding bound (a residual of two flips is more than one output code where the spacing is one flip) or the
    flip count (2.5 / 3) must, exact or requantized to the output grid."""
    prob, stats, out, pos, (x, c_ref, c_alt, u), dP_descale = _large_gradient_case(itype, factor, requantized)
    expected, one, row = _dk_flip(out, prob, pos, u, dP_descale)
    never = lambda selection: pytest.fail("one flip on this row is inside the ordinary tolerance; the evidence must not be consulted")
    assert (
        assert_close_fp8_grad(one, expected, _atol(itype), RTOL, tag="dK", keys=S, operand=prob.q, flip_unit=dP_descale, intermediates=never, fp8_dtype=itype)
        is None
    )
    _, actual, _ = _dk_flip(out, prob, pos, u, dP_descale, factor)
    if requantized:
        actual[row] = _requant(actual[row], out[6], itype)
    assert _is_bad_row(actual, expected, row, _atol(itype))
    out_txt = _rejected(
        capsys,
        actual=actual,
        expected=expected,
        atol=_atol(itype),
        rtol=RTOL,
        tag="dK",
        keys=S,
        operand=prob.q,
        flip_unit=dP_descale,
        intermediates=prob.bwd_intermediates(stats),
        fp8_dtype=itype,
    )
    assert f"(b, hq, i, j)={pos}" in out_txt and f"a step of {u:+g} x flip_unit x gain, which" in out_txt and "not the fitted" in out_txt
    if not requantized:
        # the exact residual hides in the ordinary tolerance by construction: the other two criteria did the rejecting
        assert ("x the output's own rounding" in out_txt) or ("flips of itself" in out_txt), out_txt


# ---------------------------------------------------------------------------------------------------------------------
# the old behaviour survives, and the fallback is gated
# ---------------------------------------------------------------------------------------------------------------------


def test_the_row_budget_and_the_cap_still_govern_without_intermediates():
    itype = torch.float8_e4m3fn
    prob = _problem(itype, neg_rows=True)
    stats = prob.forward()[1]
    dQ, dK, dV, _, dP_amax, dQ_amax, dK_amax, dV_amax = prob.backward(stats)
    dP_descale = get_fp8_descale_factor(dP_amax, itype)
    b, i, hq = _neg_rows(prob.q)[0]
    row = (b, 6, gqa_kv_head(hq, HQ, HKV))
    expected = _deq(dK, dK_amax, itype)
    unit = prob.q[b, i, hq] / 8.0  # a +-1 vector
    # one bad row inside the cap: the row budget (1e-5 of rows x keys, at least 1 row) accepts it, evidence or not
    actual = expected.clone()
    actual[row] += 0.3 * unit
    assert (actual - expected).abs().max().item() < _cap(itype)
    assert assert_close_fp8_grad(actual, expected, _atol(itype), RTOL, tag="dK", keys=S) is None
    # the same row above the cap: rejected without intermediates -- with or without operand / flip_unit
    actual = expected.clone()
    actual[row] += 0.5 * unit
    assert _is_bad_row(actual, expected, row, _atol(itype))
    with pytest.raises(AssertionError):
        assert_close_fp8_grad(actual, expected, _atol(itype), RTOL, tag="dK", keys=S)
    with pytest.raises(AssertionError):
        assert_close_fp8_grad(actual, expected, _atol(itype), RTOL, tag="dK", keys=S, operand=prob.q, flip_unit=dP_descale)
    # too many bad rows: rejected even with evidence on offer (the row budget is not what the evidence replaces)
    with pytest.raises(AssertionError):
        assert_close_fp8_grad(
            expected + 0.5,
            expected,
            _atol(itype),
            RTOL,
            tag="dK",
            keys=S,
            operand=prob.q,
            flip_unit=dP_descale,
            intermediates=prob.bwd_intermediates(stats),
            fp8_dtype=itype,
        )
    # intermediates without the rest is a call-site error, not a silent cap
    with pytest.raises(TypeError, match="fp8_dtype"):
        assert_close_fp8_grad(actual, expected, _atol(itype), RTOL, tag="dK", keys=S, operand=prob.q, flip_unit=dP_descale, intermediates=lambda sel: None)
    with pytest.raises(ValueError, match="kind"):
        assert_close_fp8_grad(
            actual, expected, _atol(itype), RTOL, tag="grad", keys=S, operand=prob.q, flip_unit=dP_descale, intermediates=lambda sel: None, fp8_dtype=itype
        )


def test_a_garbage_row_is_rejected_with_intermediates(capsys):
    itype = torch.float8_e4m3fn
    prob = _problem(itype, neg_rows=True)
    stats = prob.forward()[1]
    dQ, dK, dV, _, dP_amax, dQ_amax, dK_amax, dV_amax = prob.backward(stats)
    expected = _deq(dK, dK_amax, itype)
    actual = expected.clone()
    g = torch.Generator(device="cuda").manual_seed(1)
    actual[1, 7, 0] += torch.randn(D, generator=g, device="cuda")
    _rejected(
        capsys,
        actual=actual,
        expected=expected,
        atol=_atol(itype),
        rtol=RTOL,
        tag="dK",
        keys=S,
        operand=prob.q,
        flip_unit=get_fp8_descale_factor(dP_amax, itype),
        intermediates=prob.bwd_intermediates(stats),
        fp8_dtype=itype,
    )


def test_a_packed_layout_keeps_the_plain_cap(capsys):
    """Ragged outputs are [T, h, d]: a row index does not name (b, s), so the evidence path is unavailable and the cap
    applies as before (exec_sdpa_fp8 passes intermediates=None there; a caller who does not gets the same answer)."""
    itype = torch.float8_e4m3fn
    prob = _problem(itype, neg_rows=True)
    stats = prob.forward()[1]
    dQ, dK, dV, _, dP_amax, dQ_amax, dK_amax, dV_amax = prob.backward(stats)
    dP_descale = get_fp8_descale_factor(dP_amax, itype)
    b, i, hq = _neg_rows(prob.q)[0]
    expected = _deq(dK, dK_amax, itype)
    actual = expected.clone()
    actual[b, 3, gqa_kv_head(hq, HQ, HKV)] += (1.0 / 16 / dP_descale) * dP_descale * prob.q[b, i, hq]
    packed = lambda t: t.reshape(B * S, HKV, D)
    with pytest.raises(AssertionError):
        assert_close_fp8_grad(
            packed(actual),
            packed(expected),
            _atol(itype),
            RTOL,
            tag="dK",
            keys=S,
            operand=prob.q,
            flip_unit=dP_descale,
            intermediates=prob.bwd_intermediates(stats),
            fp8_dtype=itype,
        )
    out = capsys.readouterr().out
    assert "needs a dense [b, s, h, d] output" in out and "the row-budget cap applies" in out
