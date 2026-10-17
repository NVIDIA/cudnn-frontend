# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""K1 of the triangle multiplicative update -- the LayerNorm-prologue gated dual projection on the
``cudnn.gemm.frost.ln_prologue`` substrate -- against an fp64 reference, plus the GPU-free contract tests.

Tolerance of the oracle (derived, stated): the kernel's operand is the LayerNormed row rounded ONCE to bf16 and the reference
models exactly that (fp64 LayerNorm -> bf16 -> fp64 GEMM), so the remaining differences are (i) the fp32 accumulation over
K <= 256 (<= K * 2^-24 relative -- negligible), (ii) the sigmoid through ``tanh.approx`` (<= 2^-11 absolute on the tanh, i.e.
<= 2^-12 on the sigmoid, times |value| ~ O(1)), (iii) the one bf16 rounding of the output (half-ulp = 2^-9 relative), and (iv) the
fp32-vs-fp64 LayerNorm statistics flipping the bf16 rounding of a few operands by one ulp (2^-8 * |x_j| * |W_nj| per flip).
Together: ``|diff| <= 4e-3 + |ref| / 128`` (two bf16 half-ulps relative plus an absolute floor for (ii) and (iv)); the test also
reports the measured maximum.  Masked and padded rows must be EXACTLY +0.0 (a select in the kernel, sdpa-invariants.md S 2).

The kernel is rendered for Rubin (cc 10.7, the sm107 body) and Blackwell SM100 / SM103 (cc 10.0 / 10.3, the sm100 body); the accept tests build the live device's body and skip elsewhere with the reason stated.  Every launch is a single process
under the harness's ``timeout`` (exit 124 = hang).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Triangle multiplicative update tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

from cudnn.gemm.frost import ln_prologue as sub  # noqa: E402
from cudnn.triangle_multiplicative_update.kernels import trimul_proj as k1  # noqa: E402
from sdpa.frost import frost_test_utils as sass_utils  # noqa: E402  -- the SASS probe (trace-compile + nvdisasm opcode counts) and the spill-pin bound

_SM107 = (10, 7)
#: the compute capabilities the substrate has a rendered body for: Rubin (sm107) and Blackwell SM100 / SM103 (sm100)
_SERVED = tuple(sorted(sub.ARCHS_BY_CC))


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


def _arch() -> str:
    """The substrate body the live device runs (``"sm107"`` off a served part: the host-only tests then exercise the Rubin body's text)."""
    return sub.body_arch_for_cc(_cc()) if _cc() in sub.ARCHS_BY_CC else "sm107"


requires_served_arch = pytest.mark.skipif(
    _cc() not in _SERVED, reason=f"the LN-prologue dual GEMM is rendered for compute capabilities {_SERVED} only; found {_cc()}"
)


# ---------------------------------------------------------------------------
# GPU-free: the row-block geometry, the weight interleave, the typed declines, the model counts
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n_res,bn,k_block",
    [
        # the rule's three default_k_block branches at their edges (the N 8 / 9 floors, the padded tail, a 256-row multiple, a long N), and the
        # knob's domain ends 8 / 32 / 64 / 128 / 256 -- the cover invariant at every geometry where the arithmetic turns
        (8, 1, None),
        (9, 3, None),
        (217, 217, None),
        (384, 2, None),
        (1000, 3, None),
        (384, 2, 128),  # the k_block knob: 3 full k blocks of 2 bi (the rule's 256 pads 25 % of every pair tile at N = 384)
        (384, 1, 64),
        (217, 3, 32),  # 7 blocks of 32 (224 rows per bi: 3 % padding instead of the rule's 15 %) -- the projection stage alone
        (128, 5, 256),  # a k_block above N: one bi per tile, half of every tile padding (legal)
        (1000, 3, 8),
    ],
)
def test_row_block_geometry_covers_every_valid_row_exactly_once(n_res, bn, k_block):
    g = sub.row_block_geometry(n_res, bn, k_block)
    assert g.kb == (sub.default_k_block(n_res) if k_block is None else k_block)
    assert g.r_i * g.kb == 256 and g.kb & (g.kb - 1) == 0 and g.kb >= 8 and g.kb == 1 << g.lkb
    assert g.kbc * g.rc == 128 and g.kbc == min(g.kb, 128)
    assert g.n_pad % 8 == 0 and n_res <= g.n_pad < n_res + 8
    assert g.m_sched == g.pair_tiles * 256 and g.rows_valid == bn * n_res
    covered = set()
    for t in range(g.pair_tiles):
        ib, kbi = divmod(t, g.n_kb)
        for r in range(256):
            bi = ib * g.r_i + (r >> g.lkb)
            k = kbi * g.kb + (r & (g.kb - 1))
            if bi < bn and k < n_res:
                assert (bi, k) not in covered
                covered.add((bi, k))
    assert len(covered) == bn * n_res


def test_row_block_geometry_k_block_is_a_typed_vocabulary():
    """The knob's domain: an int power of two in [8, 256]; the rule pads 25 % at N = 384 and k_block = 128 pads none (the lever's arithmetic)."""
    for bad in (0, 4, 96, 512, 7, -128, 128.0, "128", True):
        with pytest.raises(ValueError, match="k_block"):
            sub.row_block_geometry(384, 2, bad)
    assert sub.validate_k_block(128) == 128 and sub.default_k_block(384) == 256 and sub.default_k_block(128) == 128 and sub.default_k_block(33) == 64
    rule, knob = sub.row_block_geometry(384, 4), sub.row_block_geometry(384, 4, 128)
    assert rule.rows_valid == knob.rows_valid == 4 * 384
    assert rule.rows_computed == 4 * 2 * 256 == 4 * rule.rows_valid // 3 and knob.rows_computed == knob.rows_valid  # 25 % padding -> none
    assert (rule.pair_tiles, knob.pair_tiles) == (8, 6) and (rule.kbc, rule.rc) == (knob.kbc, knob.rc) == (128, 1)  # the SAME per-CTA TMA box
    assert sub.row_block_geometry(384, 1, 64).kbc == 64 and sub.row_block_geometry(384, 1, 64).rc == 2


def test_interleave_value_gate_puts_value_and_gate_blocks_in_one_256_tile():
    c, kk = 256, 64
    v = torch.arange(c * kk, dtype=torch.float32).reshape(c, kk)
    g = -v
    w = sub.interleave_value_gate(v, g)
    assert w.shape == (2 * c, kk)
    for j in range(c // 128):
        assert torch.equal(w[256 * j : 256 * j + 128], v[128 * j : 128 * j + 128])
        assert torch.equal(w[256 * j + 128 : 256 * j + 256], g[128 * j : 128 * j + 128])
    b = sub.interleave_value_gate_bias(torch.arange(c, dtype=torch.float32), -torch.arange(c, dtype=torch.float32))
    assert torch.equal(b[:128], torch.arange(128, dtype=torch.float32)) and torch.equal(b[128:256], -torch.arange(128, dtype=torch.float32))


def test_stack_projection_weights_follows_the_cueq_row_convention():
    c, cz = 128, 64
    p = torch.randn(2 * c, cz)
    g = torch.randn(2 * c, cz)
    w = k1.stack_projection_weights(p, g)
    assert w.shape == (2, 2 * c, cz) and w.is_contiguous()
    assert torch.equal(w[0], sub.interleave_value_gate(p[:c], g[:c]))  # pass a
    assert torch.equal(w[1], sub.interleave_value_gate(p[c:], g[c:]))  # pass b
    b = k1.stack_projection_biases(torch.arange(2 * c, dtype=torch.float32), torch.zeros(2 * c))
    assert b.shape == (2, 2 * c) and torch.equal(b[1, :128], torch.arange(c, 2 * c, dtype=torch.float32))


@pytest.mark.parametrize(
    "kw",
    [
        dict(k=100),
        dict(k=0),
        dict(c_out=64),
        dict(c_out=0),
        dict(passes=0),
        dict(eps=0.0),
        dict(pass_order="zigzag"),
        dict(pass_order="interleaved", passes=3),
        # the LayerNorm-once work item (T27): its vocabulary, and the forms it cannot combine with (stated in the substrate's validator)
        dict(ln_once="tile"),
        dict(ln_once="block", pass_order="interleaved"),
        dict(ln_once="block", prefetch_a_l2=True),
        dict(ln_once="block", diag="ln_first_tile_only"),
        dict(ln_once="passes", passes=3),
    ],
)
def test_params_validation_declines_typed(kw):
    with pytest.raises(ValueError):
        sub.validate_ln_dual_gemm_params(sub.LnDualGemmParams(**kw))
    with pytest.raises(TypeError):
        sub.validate_ln_dual_gemm_params(object())


def test_ln_once_item_and_region_slots_are_the_substrate_rule_and_agree_with_the_body():
    """The LayerNorm-once lever's derived geometry (one source, `ln_prologue`): the tiles per work item = passes x N tiles ("block") or passes
    ("passes"), a one-tile item a typed decline; the resident A region is double-buffered where two slots plus the shipped B ring fit in the shipped
    A/B ring's bytes (K1 at both widths), single-buffered where they do not (K3 at c 256: 2 operands x 4 stages x 2 slots would leave one B stage);
    and the ring depth / stage bytes the predicate reads are the BODY's constants (a drift here would size the region against a ring the body no
    longer has).  Host-only (the body module is executed, not compiled)."""
    from cudnn.frost.template_loader import load_template
    from cudnn.triangle_multiplicative_update.kernels import trimul_tail as k3

    p128 = k1.proj_params(128, 128, has_bias=True, has_mask=True, ln_once="block")
    p256 = k1.proj_params(256, 256, has_bias=True, has_mask=True, ln_once="block")
    p256p = k1.proj_params(256, 256, has_bias=True, has_mask=True, ln_once="passes")
    t256 = k3.tail_params(256, 256, has_bias=True, ln_once="block")
    for p in (p128, p256, p256p, t256):
        sub.validate_ln_dual_gemm_params(p)
    assert (sub.ln_once_tiles_per_item(p128), sub.ln_once_tiles_per_item(p256), sub.ln_once_tiles_per_item(p256p), sub.ln_once_tiles_per_item(t256)) == (
        2,
        4,
        2,
        2,
    )
    assert (sub.ln_once_a_slots(p128), sub.ln_once_a_slots(p256), sub.ln_once_a_slots(p256p), sub.ln_once_a_slots(t256)) == (2, 2, 2, 1)
    off = k1.proj_params(256, 256, has_bias=True, has_mask=True)
    assert sub.ln_once_tiles_per_item(off) == 1 and sub.ln_once_a_slots(off) == 0 and off.ln_once == "" and sub.LnDualGemmParams().ln_once == ""
    with pytest.raises(ValueError, match="ONE GEMM tile"):
        sub.validate_ln_dual_gemm_params(k3.tail_params(128, 128, has_bias=True, ln_once="block"))  # one N tile x one pass
    with pytest.raises(ValueError, match="ONE GEMM tile"):
        sub.validate_ln_dual_gemm_params(k3.tail_params(256, 256, has_bias=True, ln_once="passes"))  # one pass
    # the sm100 body's 6-deep ring holds ONE resident slot at K1 K = 256 (2 x 64 KiB + the 96 KiB B ring > 192 KiB) where the sm107 body's holds two
    assert (
        sub.ln_once_a_slots(p128, "sm100"),
        sub.ln_once_a_slots(p256, "sm100"),
        sub.ln_once_a_slots(p256p, "sm100"),
        sub.ln_once_a_slots(t256, "sm100"),
    ) == (
        2,
        1,
        1,
        1,
    )
    assert sub.RING_STAGES == sub.RING_STAGES_BY_ARCH["sm107"]  # the historical name is the Rubin column
    # each body's ring depth and per-stage operand bytes, per arm, are what the SMEM predicate reads: the same bytes re-cut per arm (9 x 32 = 6 x 48 KiB
    # on the 327 KiB part, 6 x 32 = 4 x 48 KiB on the 227 KiB one), the arch constants the renderer moves, BODY_ARCH the key the body reads them through
    ring_bytes = {"sm107": 288 * 1024, "sm100": 192 * 1024}
    for arch in ("sm107", "sm100"):
        for p, pairing in ((k1.proj_params(128, 128, has_bias=True, has_mask=True), "columns"), (k3.tail_params(256, 256, has_bias=True), "accumulators")):
            mod = load_template(sub.body_path(arch), p, tag=f"t27_ring_pin_{arch}")
            assert mod.BODY_ARCH == arch
            assert mod.ab_stages == sub.ring_stages(arch, pairing) == sub.RING_STAGES_BY_ARCH[arch][pairing], (arch, pairing, mod.ab_stages)
            assert (
                mod.cta_tile_mnk[0] * mod.cta_tile_mnk[2] * 2 == sub.A_STAGE_BYTES
                and mod.cta_tile_mnk[1] * mod.cta_tile_mnk[2] * 2 == sub.B_STAGE_BYTES[pairing]
            )
            assert (mod.num_a_operands, mod.num_b_operands) == (p.num_a_operands, p.num_b_operands)
            assert mod.ab_stages * (mod.num_a_operands * sub.A_STAGE_BYTES + mod.num_b_operands * sub.B_STAGE_BYTES[pairing]) == ring_bytes[arch]
            assert mod._B_STAGES == mod.ab_stages and 64 * mod.ab_stages == sub.max_k(arch, pairing)
            assert (mod.num_tmem_alloc_cols, mod.tmem_alloc_exclusive) == ((576, True) if arch == "sm107" else (512, False))
            assert mod.acc_stages * 256 <= mod.num_tmem_alloc_cols  # the accumulator ring fills the 512-column part exactly
            assert f"--gpu-arch {'sm_107a' if arch == 'sm107' else 'sm_100a'}" in mod.frost_compile_options
            assert mod._SMEM_ESTIMATE_BYTES <= sub.SMEM_BUDGET_BYTES_BY_ARCH[arch]


def test_body_path_names_one_rendered_body_per_arch():
    import os

    assert os.path.isfile(sub.body_path()) and os.path.basename(sub.body_path("sm107")) == "sm107_dual_gemm.py"
    assert os.path.isfile(sub.body_path("sm100")) and os.path.basename(sub.body_path("sm100")) == "sm100_dual_gemm.py"
    with pytest.raises(ValueError, match="rendered body"):
        sub.body_path("sm120")  # a part without a body: a typed decline, never a silent fallback to another arch's body
    assert (sub.body_arch_for_cc((10, 0)), sub.body_arch_for_cc((10, 3)), sub.body_arch_for_cc((10, 7))) == ("sm100", "sm100", "sm107")
    assert sub.ARCH_GATE_CC == {"sm100": (10, 0), "sm107": (10, 7)}
    for cc in ((9, 0), (8, 0), (12, 0), (11, 0)):
        with pytest.raises(NotImplementedError, match="10.0, 10.3, 10.7"):
            sub.body_arch_for_cc(cc)
    with pytest.raises(ValueError, match="rendered body"):
        sub.validate_ln_dual_gemm_params(k1.proj_params(128, 128, has_bias=True, has_mask=True), "sm120")


def test_the_k_cap_follows_the_arch_body():
    """``K <= 64 x ring``: 576 / 384 on the sm107 body, 384 / 256 on the sm100 one -- the same record is valid for one body and a typed decline for
    the other (c_z = 256 fills the sm100 accumulators arm's 4-stage ring exactly; 384 is out there)."""
    from cudnn.triangle_multiplicative_update.kernels import trimul_tail as k3

    assert (sub.max_k("sm107", "columns"), sub.max_k("sm107", "accumulators"), sub.max_k("sm100", "columns"), sub.max_k("sm100", "accumulators")) == (
        576,
        384,
        384,
        256,
    )
    sub.validate_ln_dual_gemm_params(k3.tail_params(384, 384, has_bias=True), "sm107")
    sub.validate_ln_dual_gemm_params(k3.tail_params(256, 256, has_bias=True), "sm100")
    sub.validate_ln_dual_gemm_params(k1.proj_params(384, 128, has_bias=True, has_mask=True), "sm100")
    with pytest.raises(ValueError, match="4-deep.*K must be <= 256"):
        sub.validate_ln_dual_gemm_params(k3.tail_params(384, 384, has_bias=True), "sm100")
    with pytest.raises(ValueError, match="6-deep.*K must be <= 384"):
        sub.validate_ln_dual_gemm_params(k1.proj_params(448, 128, has_bias=True, has_mask=True), "sm100")
    with pytest.raises(ValueError, match="9-deep.*K must be <= 576"):
        sub.validate_ln_dual_gemm_params(k1.proj_params(640, 128, has_bias=True, has_mask=True), "sm107")
    with pytest.raises(ValueError, match="6-deep.*K must be <= 384"):
        sub.validate_ln_dual_gemm_params(k3.tail_params(512, 512, has_bias=True), "sm107")  # K 512 = 8 stages: past the sm107 accumulators ring too


def test_ln_once_b_ring_depth_is_self_gating_and_buys_the_second_slot_on_sm100_only():
    """The LayerNorm-once item's B-ring depth (`ln_once_b_stages`, B-TriMul-1, 2026-10-14): honoured ONLY where the shallower ring buys the second
    resident A slot the shipped ring refuses -- the sm100 body's K1 at K = 256 (2 x 64 KiB + 4 x 16 KiB = the shipped 192 KiB) -- and the shipped ring
    everywhere else (the Rubin body already holds two slots; K1 c 128 holds two under its own ring; the two-operand K3 region never fits twice).  The
    default 0 keeps every production rendering's constants; the validator declines the field without the item, past the arch's ring, or malformed."""
    from cudnn.gemm.frost import ln_prologue as sub
    from cudnn.triangle_multiplicative_update.kernels import trimul_proj as k1
    from cudnn.triangle_multiplicative_update.kernels import trimul_tail as k3

    def p256(b):
        return k1.proj_params(
            256, 256, has_bias=True, has_mask=True, pass_order="major", ln_pass_form="smem_rolled_pf", epi_math="packed", ln_once="block", ln_once_b_stages=b
        )

    assert sub.LnDualGemmParams().ln_once_b_stages == 0 and p256(None).ln_once_b_stages == 0
    # sm100 K1 K = 256: the shipped 6-deep ring -> ONE slot; 4 (and 3, 2) -> the second slot; 5 / 6 buy nothing -> the shipped ring stays
    assert [(sub.ln_once_b_stages_effective(p256(b), "sm100"), sub.ln_once_a_slots(p256(b), "sm100")) for b in (0, 2, 3, 4, 5, 6)] == [
        (6, 1),
        (2, 2),
        (3, 2),
        (4, 2),
        (6, 1),
        (6, 1),
    ]
    # the Rubin body holds two slots under its own 9-deep ring: the field is a no-op there (no B depth traded for nothing)
    assert [(sub.ln_once_b_stages_effective(p256(b), "sm107"), sub.ln_once_a_slots(p256(b), "sm107")) for b in (0, 4)] == [(9, 2), (9, 2)]
    # K1 c 128 already holds two slots under the sm100 ring; the output stage's two-operand region never fits twice: both keep their ring
    p128 = k1.proj_params(
        128, 128, has_bias=True, has_mask=True, pass_order="major", ln_pass_form="registers", epi_math="packed", ln_once="block", ln_once_b_stages=4
    )
    assert (sub.ln_once_b_stages_effective(p128, "sm100"), sub.ln_once_a_slots(p128, "sm100")) == (6, 2)
    t256 = k3.tail_params(
        256, 256, has_bias=True, a0_prologue="mn_major", ln_pass_form="smem_rolled_pf", epi_math="packed", ln_once="block", ln_once_b_stages=2
    )
    assert (sub.ln_once_b_stages_effective(t256, "sm100"), sub.ln_once_a_slots(t256, "sm100")) == (4, 1)
    off = k1.proj_params(256, 256, has_bias=True, has_mask=True)
    assert sub.ln_once_b_stages_effective(off, "sm100") == 6 and sub.ln_once_b_stages_effective(off, "sm107") == 9
    for arch in ("sm100", "sm107"):
        sub.validate_ln_dual_gemm_params(p256(4), arch)
    with pytest.raises(ValueError, match="LayerNorm-once item only"):
        sub.validate_ln_dual_gemm_params(k1.proj_params(256, 256, has_bias=True, has_mask=True, ln_once_b_stages=4), "sm100")
    with pytest.raises(ValueError, match="deeper than the sm100 body"):
        sub.validate_ln_dual_gemm_params(p256(7), "sm100")
    sub.validate_ln_dual_gemm_params(p256(7), "sm107")  # 7 < the Rubin ring: valid there (and a no-op: two slots already)
    with pytest.raises(ValueError, match="non-negative integer"):
        sub.validate_ln_dual_gemm_params(p256(-1), "sm100")
    with pytest.raises(ValueError, match="non-negative integer"):
        sub.validate_ln_dual_gemm_params(p256(True), "sm100")


def test_sm100_declines_the_k256_register_form_with_a_packed_epilogue_typed():
    """The sm100 body's one unallocatable rendering class (ptxas C7600 at the flat 168-register entry count; the sm107 body renders every member): the
    K >= 256 register-form LayerNorm with a packed epilogue -- every accumulators-arm member (stats on or off, both prologues), the columns arm's
    bias-path members WITH the statistics write.  The validator declines exactly the measured set and nothing beside it (the plain packed columns
    rendering, the no-statistics bias-path ones, every re-read form, every K = 128 record, every sm107 record validate)."""
    from cudnn.triangle_multiplicative_update.kernels import trimul_tail as k3

    declined, served = [], []
    for stats in (True, False):
        for epm in ("packed", "packed_hoist", "packed_biasfold", "scalar"):
            for prologue in ("mn_major", "transpose"):
                declined.append(k3.tail_params(256, 256, has_bias=True, want_stats=stats, a0_prologue=prologue, ln_pass_form="registers", epi_math=epm))
            declined.append(k1.proj_params(256, 256, has_bias=True, has_mask=True, want_stats=stats, ln_pass_form="registers", epi_math=epm))
    served = [p for p in declined if p.epi_math == "scalar" or (not p.dual and (not p.want_stats or p.epi_math == "packed"))]
    declined = [p for p in declined if p not in served]
    assert len(declined) == 14 and len(served) == 10
    for p in declined:
        assert sub.sm100_register_form_unallocatable(p, "sm100") and not sub.sm100_register_form_unallocatable(p, "sm107")
        with pytest.raises(ValueError, match="C7600"):
            sub.validate_ln_dual_gemm_params(p, "sm100")
        sub.validate_ln_dual_gemm_params(p, "sm107")  # the sm107 body renders it (spilling)
    for p in served:
        assert not sub.sm100_register_form_unallocatable(p, "sm100")
        sub.validate_ln_dual_gemm_params(p, "sm100")
    for form in ("smem_rolled", "smem_rolled_pf"):
        sub.validate_ln_dual_gemm_params(k3.tail_params(256, 256, has_bias=True, want_stats=True, ln_pass_form=form, epi_math="packed_biasfold"), "sm100")
        sub.validate_ln_dual_gemm_params(
            k1.proj_params(256, 256, has_bias=True, has_mask=True, want_stats=True, ln_pass_form=form, epi_math="packed_biasfold"), "sm100"
        )
    for epm in ("packed", "packed_hoist", "packed_biasfold"):
        sub.validate_ln_dual_gemm_params(k3.tail_params(128, 128, has_bias=True, want_stats=True, ln_pass_form="registers", epi_math=epm), "sm100")
        sub.validate_ln_dual_gemm_params(
            k1.proj_params(128, 128, has_bias=True, has_mask=True, want_stats=True, ln_pass_form="registers", epi_math=epm), "sm100"
        )


def _sm100_c7600_cell(params) -> bool:
    """Whether a kernel-test rendering is in the sm100 body's typed-declined class on the LIVE device (the bitwise cells expect the decline there)."""
    return sub.sm100_register_form_unallocatable(params, _arch())


_ARCH_DELTA_CONSTANTS = (
    "ab_stages",
    "b_collector_ok",
    "swizzle_l2_budget_bytes",
    "num_tmem_alloc_cols",
    "tmem_alloc_exclusive",
    "frost_compile_options",
    "BODY_ARCH",
)


def test_the_two_bodies_differ_only_in_the_arch_delta():
    """The sm100 body is the sm107 body with the arch constants the renderer moves (and the docstring / comment lines that state them): every
    CODE line outside those constants is byte-identical, so a kernel fix that lands in one file and not the other fails here."""
    import difflib

    a = open(sub.body_path("sm107")).read().splitlines()
    b = open(sub.body_path("sm100")).read().splitlines()

    def docstring_end(lines):
        return next(i for i, ln in enumerate(lines) if ln.startswith("from __future__"))

    spans = (docstring_end(a), docstring_end(b))
    assert spans[0] > 50 and spans[1] > 50
    changed = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag == "equal":
            continue
        changed += [(0, i, a[i]) for i in range(i1, i2)] + [(1, j, b[j]) for j in range(j1, j2)]
    assert changed, "the two bodies are identical -- the sm100 body lost its arch constants"
    offenders = []
    for side, idx, ln in changed:
        in_docstring = idx < spans[side]
        is_comment = ln.lstrip().startswith("#")
        is_constant = any(ln.startswith(f"{name} = ") or ln.startswith(f"{name}: ") for name in _ARCH_DELTA_CONSTANTS)
        if not (in_docstring or is_comment or is_constant):
            offenders.append((side, idx + 1, ln))
    assert not offenders, f"code lines outside the arch delta differ between the two bodies: {offenders[:10]}"
    moved = {ln.split(" = ")[0].split(":")[0] for side, idx, ln in changed if any(ln.startswith(n) for n in _ARCH_DELTA_CONSTANTS)}
    assert moved == set(_ARCH_DELTA_CONSTANTS), moved


def test_planes_shape_and_strides_are_the_k2_contract():
    assert k1.planes_shape(128, 2, 217) == (2, 128, 434, 224)
    assert k1.plane_strides(2, 217) == (434 * 224, 224, 1)
    assert k1.planes_shape(256, 1, 512) == (2, 256, 512, 512)


class _Boom(Exception):
    pass


def _fake_plan(params):
    def _launch(*a, **k):
        raise _Boom()

    return SimpleNamespace(params=params, launch=_launch)


def test_runner_declines_before_touching_the_artifact():
    p = sub.LnDualGemmParams(k=128, c_out=128, passes=2)
    plan = _fake_plan(p)
    n_res, bn = 16, 4
    rows = bn * n_res
    bf16 = torch.bfloat16
    ok = dict(
        z=torch.zeros(rows, 128, dtype=bf16),
        w=torch.zeros(2, 256, 128, dtype=bf16),
        out=torch.zeros(2, 128, bn, 16, dtype=bf16),
        ln_weight=torch.zeros(128, dtype=bf16),
        ln_bias=torch.zeros(128, dtype=bf16),
        bias=torch.zeros(2, 256, dtype=bf16),
        mask=torch.zeros(rows, dtype=bf16),
        stats=torch.zeros(rows, 2),
    )

    def run(**over):
        kw = dict(ok, **over)
        return sub.run_ln_dual_gemm(
            plan, kw["z"], kw["w"], kw["out"], kw["ln_weight"], kw["ln_bias"], n_res=n_res, bn=bn, bias=kw["bias"], mask=kw["mask"], stats=kw["stats"], stream=0
        )

    with pytest.raises(_Boom):  # every check passes -> the artifact is reached
        run()
    with pytest.raises(ValueError, match="z must be"):
        run(z=ok["z"].float())
    with pytest.raises(ValueError, match="w must be"):
        run(w=torch.zeros(2, 128, 128, dtype=bf16))
    with pytest.raises(ValueError, match="out must be"):
        run(out=torch.zeros(2, 128, bn, 15, dtype=bf16))
    with pytest.raises(ValueError, match="bias must be bound"):
        run(bias=None)
    with pytest.raises(ValueError, match="mask must be"):
        run(mask=torch.zeros(rows + 1, dtype=bf16))
    with pytest.raises(ValueError, match="stats must be"):
        run(stats=torch.zeros(rows, 2, dtype=bf16))
    with pytest.raises(ValueError, match="ln_weight must be"):
        run(ln_weight=torch.zeros(64, dtype=bf16))
    # alignment: a contiguous but OFFSET view has the right shape and dtype and would fault (or read the wrong half-word) in the kernel
    with pytest.raises(ValueError, match="ln_weight must be 16-byte aligned"):
        run(ln_weight=torch.zeros(128 + 4, dtype=bf16)[4:])  # 8-byte offset
    with pytest.raises(ValueError, match="bias must be 16-byte aligned"):
        run(bias=torch.zeros(2 * 256 + 4, dtype=bf16)[4:].view(2, 256))
    with pytest.raises(ValueError, match="mask must be 4-byte aligned"):
        run(mask=torch.zeros(rows + 1, dtype=bf16)[1:])  # 2-byte offset: the kernel reads aligned 32-bit words
    with pytest.raises(ValueError, match="stats must be 8-byte aligned"):
        run(stats=torch.zeros(rows * 2 + 1)[1:].view(rows, 2))  # 4-byte offset: the kernel stores 8-byte (mean, rstd) pairs
    with pytest.raises(_Boom):  # aligned offset views pass (16 / 4 / 8 bytes)
        run(ln_weight=torch.zeros(128 + 8, dtype=bf16)[8:], mask=torch.zeros(rows + 2, dtype=bf16)[2:], stats=torch.zeros(rows * 2 + 2)[2:].view(rows, 2))


# ---------------------------------------------------------------------------
# Rubin: the fp64 oracle, the skip pin, the degenerate rows
# ---------------------------------------------------------------------------
def _inputs(n_res, batch, c_z, c, mask_kind, with_bias, seed=0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    bf16 = torch.bfloat16
    # per-channel offsets so the LayerNorm mean is not ~0 (exercises the two-pass statistics), unit-ish variance
    z = (torch.randn(batch, n_res, n_res, c_z, generator=g) * 0.7 + torch.randn(c_z, generator=g)[None, None, None, :] * 0.5).to(bf16)
    ln_w = (1.0 + 0.1 * torch.randn(c_z, generator=g)).to(bf16)
    ln_b = (0.1 * torch.randn(c_z, generator=g)).to(bf16)
    p_in = (torch.randn(2 * c, c_z, generator=g) / math.sqrt(c_z)).to(bf16)  # rows [a; b]
    g_in = (torch.randn(2 * c, c_z, generator=g) / math.sqrt(c_z)).to(bf16)
    if with_bias:
        p_b = (0.2 * torch.randn(2 * c, generator=g)).to(bf16)
        g_b = (0.2 * torch.randn(2 * c, generator=g)).to(bf16)
    else:
        p_b = g_b = None
    if mask_kind == "all":
        mask = torch.ones(batch, n_res, n_res)
    elif mask_kind == "padded":  # a right-padded protein per batch: valid tokens L_b, m[i, k] = (i < L) & (k < L)
        mask = torch.zeros(batch, n_res, n_res)
        for b in range(batch):
            L = (3 * n_res) // 4 if b == 0 else max(1, n_res // 3)
            mask[b, :L, :L] = 1.0
    elif mask_kind == "random":
        mask = (torch.rand(batch, n_res, n_res, generator=g) > 0.3).float()
    elif mask_kind == "fractional":  # 70 % of the rows carry a FRACTIONAL weight in [0.25, 1), the rest 0 -- distinguishes the rounding order
        mask = (torch.rand(batch, n_res, n_res, generator=g) > 0.3).float() * (0.25 + 0.75 * torch.rand(batch, n_res, n_res, generator=g))
    else:
        raise ValueError(mask_kind)
    mask = mask.to(bf16)
    return {k: (v.to(device) if v is not None else None) for k, v in dict(z=z, ln_w=ln_w, ln_b=ln_b, p_in=p_in, g_in=g_in, p_b=p_b, g_b=g_b, mask=mask).items()}


def _reference(inp, c, eps):
    """fp64: LayerNorm -> bf16 operand -> GEMM -> sigmoid(gate + b_g) * (value + b_p) * mask, channel-major [2, c, B*N, N]; stats."""
    z = inp["z"].double()
    batch, n_res = z.shape[0], z.shape[1]
    c_z = z.shape[-1]
    rows = z.reshape(-1, c_z)
    mean = rows.mean(dim=1)
    var = rows.var(dim=1, unbiased=False)
    rstd = 1.0 / torch.sqrt(var + eps)
    x = (rows - mean[:, None]) * rstd[:, None] * inp["ln_w"].double()[None, :] + inp["ln_b"].double()[None, :]
    x = x.to(torch.bfloat16).double()  # the kernel's MMA operand rounding
    m = inp["mask"].double().reshape(-1)
    outs = []
    for pss in range(2):
        wp = inp["p_in"].double()[pss * c : (pss + 1) * c]
        wg = inp["g_in"].double()[pss * c : (pss + 1) * c]
        val = x @ wp.T
        gate = x @ wg.T
        if inp["p_b"] is not None:
            val = val + inp["p_b"].double()[pss * c : (pss + 1) * c][None, :]
            gate = gate + inp["g_b"].double()[pss * c : (pss + 1) * c][None, :]
        o = torch.sigmoid(gate) * val * m[:, None]
        o = torch.where(m[:, None] == 0, torch.zeros_like(o), o)
        outs.append(o.T.reshape(c, batch * n_res, n_res))
    return torch.stack(outs), mean, rstd


def _build_k1(params):
    """The K1 plan on the live device's body (:func:`_arch`): a cell's cubin is the arch's, never the Rubin body's on a Blackwell part."""
    return k1.build_trimul_proj(params, _arch())


def _launch(plan, inp, c, planes, stats, stream, *, w_stack=None, b_stack=None, k_block=None):
    """``w_stack`` / ``b_stack`` = the interleaved weight / bias stacks (``torch.cat`` copies, GPU kernels of their own): a timed callable
    builds them ONCE and passes them in; per call they ride inside every CUDA-graph replay as extra kernels."""
    if w_stack is None:
        w_stack = k1.stack_projection_weights(inp["p_in"], inp["g_in"])
    if b_stack is None:
        b_stack = k1.stack_projection_biases(inp["p_b"], inp["g_b"]) if inp["p_b"] is not None else None
    return k1.run_trimul_proj(
        plan, inp["z"], w_stack, planes, inp["ln_w"], inp["ln_b"], bias_stacked=b_stack, mask=inp["mask"], stats=stats, stream=stream, k_block=k_block
    )


_SENTINEL = 1.5e30


@requires_served_arch
@pytest.mark.parametrize(
    "n_res,batch,c_z,c,mask_kind,with_bias",
    [
        (217, 2, 128, 128, "random", False),
        (217, 1, 128, 128, "random", True),  # an ODD mask length (47 089 rows): the last row's mask is read without touching the byte past the tensor
        (256, 1, 128, 128, "padded", True),
        (512, 1, 128, 128, "all", True),  # the register form + the bias fold, biases on (the N >= 512 pick)
        (128, 1, 256, 256, "padded", True),  # two K tiles per stage group (K = 256) and two N tiles (c = 256)
        # (N 64 / 128 at c 128 run the body the 256 / 512 cells run; the fractional mask has its own pin below)
    ],
)
def test_k1_matches_the_fp64_oracle(n_res, batch, c_z, c, mask_kind, with_bias):
    eps = 1e-5
    params = k1.proj_params(c_z, c, has_bias=with_bias, has_mask=True, want_stats=True, eps=eps)
    plan = _build_k1(params)
    inp = _inputs(n_res, batch, c_z, c, mask_kind, with_bias)
    planes = torch.full(k1.planes_shape(c, batch, n_res), _SENTINEL, device="cuda", dtype=torch.bfloat16)
    stats = torch.full((batch * n_res * n_res, 2), -1.0, device="cuda", dtype=torch.float32)
    stream = torch.cuda.current_stream().cuda_stream
    g = _launch(plan, inp, c, planes, stats, stream)
    torch.cuda.synchronize()
    npd = g.n_pad
    valid = planes[:, :, :, :n_res].float()
    assert not (valid == _SENTINEL).any(), "a never-written cell survived in the valid region"
    if npd > n_res:
        assert (planes[:, :, :, n_res:npd].view(torch.int16) == 0).all(), "the k padding must be written as exactly +0.0"
    ref, mean64, rstd64 = _reference(inp, c, eps)
    m = inp["mask"].float().reshape(-1)
    masked_rows = (m == 0).nonzero().flatten()
    if masked_rows.numel():
        got_masked = planes[:, :, :, :n_res].reshape(2, c, -1)[:, :, masked_rows]
        assert (got_masked.view(torch.int16) == 0).all(), "a masked row must be EXACTLY +0.0"
    diff = (valid.double() - ref).abs()
    tol = 4e-3 + ref.abs() / 128.0
    bad = (diff > tol).sum().item()
    assert bad == 0, f"{bad} cells beyond the derived tolerance; max |diff| {diff.max().item():.4g} at |ref| max {ref.abs().max().item():.4g}"
    rel_rms = (diff.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    assert rel_rms < 2e-3, f"relative RMS {rel_rms:.3g}"
    # the statistics: valid rows, fp32 two-pass vs fp64
    torch.testing.assert_close(stats[:, 0].double(), mean64, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(stats[:, 1].double(), rstd64, rtol=1e-4, atol=1e-5)
    # two-launch trick: a first-launch race would show here
    planes2 = torch.empty_like(planes)
    stats2 = torch.empty_like(stats)
    _launch(plan, inp, c, planes2, stats2, stream)
    torch.cuda.synchronize()
    assert torch.equal(planes2, planes) and torch.equal(stats2, stats)


# The sm_107a renderings the triangle multiplicative update SHIPS on Rubin (the heuristics' picks at the API's parameter sets: biases, mask, no
# statistics; the default tile order / prologue form / ring depth) and their spill counts, MEASURED on the branch's toolchain (nvdisasm of the
# trace-time cubin): the register split the body requests is dropped by ptxas (no `.maxnreg` directive -> C7508), so every warp runs flat at
# the entry count (REG 168).  A pin is a BOUND (frost_test_utils.assert_no_new_spills): more spills = a regression; fewer = update the pin with
# the measurement.  USETMAXREG 0 documents the dropped split -- when THAT assertion fails the DSL started emitting `.maxnreg`: re-measure the
# roles against entry x warps and update these pins.  History the dropped option rows recorded: the register-form LayerNorm row held between
# its statistics passes spills from K = 256 up (K1 240 / 241, K3 499 / 504: the unshipped class the table never picks); the ROLLED re-read
# forms (`smem_rolled`, `smem_rolled_pf`, `smem_rolled_tree`) hold no row and spill 0 / 0 at both widths (the unrolled `smem` form only halves
# the K 256 spills: 118 / 118 and 235 / 235); the bias-path packed forms (`packed_hoist`, `packed_biasfold`) add no live word (0 / 0).
# Keyed (arm, K, ln_pass_form, epi_math): the Rubin table's picks -- K1 c 128 `registers / packed_biasfold` (512 / 768 and 256 B >= 4, the block
# item), K1 c 256 `smem_rolled_pf / packed`, K3 c 256 `smem_rolled_pf / packed`, and the K3 c 128 `registers / packed` transpose rendering.
_SASS_SPILL_PINS = {
    ("k1", 128, "registers", "packed_biasfold"): dict(STL=0, LDL=0),
    ("k1", 256, "smem_rolled_pf", "packed"): dict(STL=0, LDL=0),
    ("k3", 128, "registers", "packed"): dict(STL=1, LDL=1),  # the K3 c 128 transpose rendering: 1 / 1 at STACK 8, measured like the rows above
    ("k3", 256, "smem_rolled_pf", "packed"): dict(STL=0, LDL=0),
}
_SASS_COUNTS = {**sass_utils.SASS_OPCODE_COUNTS, "USETMAXREG": ("USETMAXREG",), "FADD2": (" FADD2",), "FFMA2": (" FFMA2",), "CGAERRBAR": ("CGAERRBAR",)}
_SASS_PROBE_BODY = """
from cudnn.gemm.frost import ln_prologue
from cudnn.triangle_multiplicative_update.kernels import trimul_proj as k1
from cudnn.triangle_multiplicative_update.kernels import trimul_tail as k3
arm, kk = params_kw["arm"], int(params_kw["k"])
lnf, epm = params_kw.get("ln_pass_form", "registers"), params_kw.get("epi_math", "scalar")
has_bias, hint = bool(params_kw.get("has_bias", True)), int(params_kw.get("min_blocks_per_mp", 0))
order = params_kw.get("pass_order", "interleaved")  # the K1 tile order of the rendering (the heuristics ship "major" from N = 512 at c = 128)
ln_once = params_kw.get("ln_once", "")  # the LayerNorm-once work item ("" = the per-tile item every production rendering ships)
body_arch = params_kw.get("body_arch", "sm107")  # the substrate body: the sm_107a pins read the Rubin body, the sm_100a pins the Blackwell one
# the API's parameter sets (api.py compile(): biases on, the mask bound, no statistics, the heuristics' default knobs) at the pinned LN / epilogue forms
if arm == "k1":
    p = k1.proj_params(kk, kk, has_bias=has_bias, has_mask=True, want_stats=False, pass_order=order, clc_stages=2, ln_pass_form=lnf, epi_math=epm, min_blocks_per_mp=hint, ln_once=ln_once)
else:
    p = k3.tail_params(kk, kk, has_bias=has_bias, want_stats=False, a0_prologue=(order if order in ("mn_major", "transpose") else "mn_major"), ln_pass_form=lnf, epi_math=epm, min_blocks_per_mp=hint, ln_once=ln_once)
plan = ln_prologue.build_ln_dual_gemm(p, body_arch)
print("EXPECT_K", kk)
print("EXPECT_AB_STAGES", plan.module.ab_stages)
print("EXPECT_TMEM_COLS", plan.module.num_tmem_alloc_cols)
"""

# The LayerNorm-ONCE renderings (`ln_once` = "block", the pick; "passes" is an option; T27): the per-item A region + the per-tile B ring, every production
# form's twin at the shipped knobs (the pass-major order the lever requires).  Keyed (arm, K, ln_pass_form, has_bias, ln_once); the values are BOUNDS measured
# on the branch's toolchain (nvdisasm of the trace-time cubin) like the tables above: the bias-off c_out 256 renderings sit at ptxas's flat 80-register
# allocation (the two-CTAs-per-SM target) and spill a few words -- K1 2 / 2 (STACK 8; its production twin 0 / 0), K3 5 / 6 (STACK 24; the twin 0 / 0 at REG
# 106) -- the register-split hint does NOT lift it worth the spills it adds (the table below); the bias-on twins and the c 128 register-form renderings spill
# nothing (the K1 c 128 register-form twin's 1 / 1 at STACK 8 is GONE under the lever); the c 128 ROLLED bias-off rendering reads 1 / 1 at REG 80 / STACK 8
# under the lever (its production twin 0 / 0 at REG 80), its bias-on twin 0 / 0 at REG 112.  USETMAXREG 0 (the split dropped, as the twins), CGAERRBAR 0 (no
# non-relaxed cluster arrive: the new a_empty commit is a tcgen05.commit, the release arrives the helpers').
_SASS_LN_ONCE_PINS = {
    ("k1", 128, "registers", False, "block"): dict(STL=0, LDL=0),
    ("k1", 128, "registers", True, "block"): dict(STL=0, LDL=0),
    ("k1", 128, "smem_rolled", False, "block"): dict(STL=1, LDL=1),
    ("k1", 128, "smem_rolled", True, "block"): dict(STL=0, LDL=0),
    ("k1", 256, "smem_rolled_pf", False, "block"): dict(STL=2, LDL=2),
    ("k1", 256, "smem_rolled_pf", True, "block"): dict(STL=0, LDL=0),
    ("k3", 256, "smem_rolled_pf", False, "block"): dict(STL=5, LDL=6),
    ("k3", 256, "smem_rolled_pf", True, "block"): dict(STL=0, LDL=0),
}


@pytest.mark.parametrize("arm,k,ln_pass_form,has_bias,ln_once", sorted(_SASS_LN_ONCE_PINS))
def test_sm107_ln_once_renderings_spill_pins(tmp_path, arm, k, ln_pass_form, has_bias, ln_once):
    """Trace-compile the LayerNorm-once renderings for sm_107a (no GPU): STL / LDL within the measured bounds, USETMAXREG 0 (the dropped split, as the
    production twins), CGAERRBAR 0 (every cluster-scope arrive of the arm is a commit or a helper's release / relaxed arrive)."""
    probe = sass_utils.run_sass_probe(
        tmp_path,
        probe_src=sass_utils.sass_probe_source(_SASS_PROBE_BODY, _SASS_COUNTS),
        arch="sm_107a",
        params=dict(arm=arm, k=k, ln_pass_form=ln_pass_form, epi_math="packed", has_bias=has_bias, pass_order="major", ln_once=ln_once),
        tag=f"ln_dual_gemm_{arm}_k{k}_{ln_pass_form}_bias{int(has_bias)}_ln_once_{ln_once}",
    )
    st = probe.stats
    print(
        f"{arm} K={k} {ln_pass_form} bias {has_bias} ln_once={ln_once}: STL {st['STL']} LDL {st['LDL']} USETMAXREG {st['USETMAXREG']} CGAERRBAR {st['CGAERRBAR']} lines {st['LINES']} cubin {probe.cubin_md5}"
    )
    sass_utils.assert_no_new_spills(
        st, _SASS_LN_ONCE_PINS[(arm, k, ln_pass_form, has_bias, ln_once)], tag=f"{arm} K={k} {ln_pass_form} bias {has_bias} ln_once={ln_once}: "
    )
    assert st["USETMAXREG"] == 0, f"{arm} K={k} ln_once={ln_once}: a REAL register split (USETMAXREG {st['USETMAXREG']}) -- re-measure the roles and the pins"
    assert st["CGAERRBAR"] == 0, f"{arm} K={k} ln_once={ln_once}: a non-relaxed cluster arrive (CGAERRBAR {st['CGAERRBAR']}) -- a GPU-scope drain per arrive"


# The PRODUCTION bias-ON fold rendering of the LayerNorm-once block item (the Rubin 512 / 768 B 1 and 256 B >= 4 cells with the biases): unhinted,
# since `heuristics.register_split_hint` keys on `ln_once` and never composes the `.minnctapersm` hint with the item -- 0 / 0 at the flat
# allocation, the cubin those cells run (measured a tie with the packed twin on 512 B 1 / 256 B 4 and +0.7..+0.9 % on 768).  History: under a
# caller's `min_blocks_per_mp=1` the honoured split (USETMAXREG 6) puts the item's +4 live words of the 24-register producer roles on the stack
# (the c 128 register form 4 / 13, the rolled form 9 / 15-16, the c 256 pf forms 9 / 16 and K3 25 / 26 at STACK 144) and the whole call runs
# 0.5-1.3 % FASTER unhinted on every bias-off c 128 cell -- which is why the rule keeps the per-tile item; a caller's hint under the item is a
# public option whose correctness is bitwise-pinned (test_register_split_hint_is_bitwise_the_default_rendering).  Keyed (arm, K, ln_pass_form,
# epi_math, has_bias, min_blocks_per_mp, order, ln_once); the spill counts are BOUNDS, USETMAXREG is exact.
_SASS_LN_ONCE_FOLD_PINS = {
    ("k1", 128, "registers", "packed_biasfold", True, 0, "major", "block"): dict(STL=0, LDL=0, USETMAXREG=0),
}


@pytest.mark.parametrize("arm,k,ln_pass_form,epi_math,has_bias,hint,order,ln_once", sorted(_SASS_LN_ONCE_FOLD_PINS))
def test_sm107_ln_once_block_item_fold_rendering_spill_pin(tmp_path, arm, k, ln_pass_form, epi_math, has_bias, hint, order, ln_once):
    """Trace-compile the production bias-on fold rendering of the LayerNorm-once block item for sm_107a (no GPU): the spills within the measured
    bound, USETMAXREG EXACTLY 0 (the dropped split: the rule never hints the item), CGAERRBAR 0."""
    probe = sass_utils.run_sass_probe(
        tmp_path,
        probe_src=sass_utils.sass_probe_source(_SASS_PROBE_BODY, _SASS_COUNTS),
        arch="sm_107a",
        params=dict(arm=arm, k=k, ln_pass_form=ln_pass_form, epi_math=epi_math, has_bias=has_bias, min_blocks_per_mp=hint, pass_order=order, ln_once=ln_once),
        tag=f"ln_dual_gemm_{arm}_k{k}_{ln_pass_form}_{epi_math}_bias{int(has_bias)}_hint{hint}_{order}_ln_once_{ln_once}",
    )
    st = probe.stats
    pins = _SASS_LN_ONCE_FOLD_PINS[(arm, k, ln_pass_form, epi_math, has_bias, hint, order, ln_once)]
    print(
        f"{arm} K={k} {ln_pass_form} {epi_math} bias {has_bias} hint {hint} ln_once={ln_once}: STL {st['STL']} LDL {st['LDL']} USETMAXREG {st['USETMAXREG']} "
        f"CGAERRBAR {st['CGAERRBAR']} lines {st['LINES']} cubin {probe.cubin_md5}"
    )
    sass_utils.assert_no_new_spills(st, pins, tag=f"{arm} K={k} {ln_pass_form} {epi_math} bias {has_bias} hint {hint} ln_once={ln_once}: ")
    assert (
        st["USETMAXREG"] == pins["USETMAXREG"]
    ), f"{arm} K={k} ln_once={ln_once} hint {hint}: USETMAXREG {st['USETMAXREG']} != {pins['USETMAXREG']} -- re-measure the roles and the pins"
    assert st["CGAERRBAR"] == 0, f"{arm} K={k} ln_once={ln_once} hint {hint}: a non-relaxed cluster arrive (CGAERRBAR {st['CGAERRBAR']})"


# The `.minnctapersm` hint (`min_blocks_per_mp`; `heuristics.register_split_hint`) on the renderings the hint exists for -- the bias-OFF forms, where
# ptxas's two-CTAs-per-SM target (65536 / (2 x 384) = 80 registers) is reached with spills; the bias-ON twin of each as the identity reference.
# Keyed (arm, K, ln_pass_form, epi_math, has_bias, min_blocks_per_mp); the values are BOUNDS measured on the branch's toolchain like the table above:
#   hint 0 (the rendering as shipped): the bias-off K3 K = 256 rolled rendering at REG 80 / STACK 24 spills 5 / 7 on the EPILOGUE's hot path (three
#     swizzled staging-store addresses + two row-decode constants reloaded per tile ahead of the TMEM loads: +7-8 % on the kernel vs the bias-on
#     twin, 69.8 k local requests in ncu); the bias-off K1 K = 128 rolled rendering 1 / 1 at STACK 8; USETMAXREG 0 (the split dropped, C7508).
#   hint 1 (.minnctapersm 1 -- the SAME cubin as a .maxnreg 168 build): ptxas honours the six setmaxnreg requests, USETMAXREG 6 (2 TRY_ALLOC 232 +
#     4 DEALLOC 24), REG = the 168 entry count, STACK 0 and NO spill on the K3 renderings (bias off and on: the identity, no new spill); the K1
#     INTERLEAVED rolled renderings carry 4 STL / 13 LDL of role-prologue values under the hint (stored at kernel entry / at the INCREASE warps'
#     entry, reloaded once per role prologue) -- the per-rendering sign the rule keys on the tile order for.
# The rule's own rows (heuristics._REGISTER_SPLIT_HINTS: the `major` order the heuristics ship from N = 512 at c = 128), hinted: the register
# form's 1 / 1 at STACK 8 and the rolled form's flat REG 80 both go to 0 / 0 at the honoured split, the bias-on fold rendering is allocated its
# demand either way; and the Rubin c 128 FLOOR rendering (`registers / packed`, interleaved, the per-tile item, hint 0: every N < 256 at B 1).
# History the dropped option rows recorded: the K1 K = 128 INTERLEAVED rolled rendering under the hint carries 4 / 13 role-prologue spills (the
# interleaved pass decode + the scheduler's tile count stored at entry and reloaded once per role prologue) -- the per-rendering sign the rule
# keys on the tile order for; the bias-off K3 K = 256 rolled rendering at REG 80 / STACK 24 spills 5 / 7 on the epilogue's hot path unhinted and
# 0 / 0 hinted.
_SASS_SPLIT_HINT_PINS = {
    ("k1", 128, "registers", "packed", False, 1, "major"): dict(STL=0, LDL=0, USETMAXREG=6),
    ("k1", 128, "smem_rolled", "packed", False, 1, "major"): dict(STL=0, LDL=0, USETMAXREG=6),
    ("k1", 128, "registers", "packed_biasfold", True, 1, "major"): dict(STL=0, LDL=0, USETMAXREG=6),
    ("k1", 128, "registers", "packed", False, 0, "interleaved"): dict(STL=1, LDL=1, USETMAXREG=0),  # the floor rendering: 1 / 1 at STACK 8 unhinted
}


@pytest.mark.parametrize("arm,k,ln_pass_form,epi_math,has_bias,hint,order", sorted(_SASS_SPLIT_HINT_PINS))
def test_sm107_register_split_hint_renderings_spill_pins(tmp_path, arm, k, ln_pass_form, epi_math, has_bias, hint, order):
    """Trace-compile the rule's hinted renderings and the c 128 floor rendering for sm_107a (no GPU): the spill counts are bounds like the table
    above; USETMAXREG is EXACT -- 0 without the hint (the dropped split), 6 with it (the honoured split)."""
    probe = sass_utils.run_sass_probe(
        tmp_path,
        probe_src=sass_utils.sass_probe_source(_SASS_PROBE_BODY, _SASS_COUNTS),
        arch="sm_107a",
        params=dict(arm=arm, k=k, ln_pass_form=ln_pass_form, epi_math=epi_math, has_bias=has_bias, min_blocks_per_mp=hint, pass_order=order),
        tag=f"ln_dual_gemm_{arm}_k{k}_{ln_pass_form}_{epi_math}_bias{int(has_bias)}_hint{hint}_{order}",
    )
    st = probe.stats
    pins = _SASS_SPLIT_HINT_PINS[(arm, k, ln_pass_form, epi_math, has_bias, hint, order)]
    print(
        f"{arm} K={k} {ln_pass_form} {epi_math} bias {has_bias} hint {hint}: STL {st['STL']} LDL {st['LDL']} USETMAXREG {st['USETMAXREG']} lines {st['LINES']} cubin {probe.cubin_md5}"
    )
    sass_utils.assert_no_new_spills(st, pins, tag=f"{arm} K={k} {ln_pass_form} {epi_math} bias {has_bias} hint {hint}: ")
    assert st["USETMAXREG"] == pins["USETMAXREG"], (
        f"{arm} K={k} bias {has_bias} hint {hint}: USETMAXREG {st['USETMAXREG']} != {pins['USETMAXREG']} -- the register split is "
        f"{'honoured' if st['USETMAXREG'] else 'dropped'} where the hint says {'honoured' if pins['USETMAXREG'] else 'dropped'}; re-measure the roles and the pins"
    )


@pytest.mark.parametrize(
    "arm,k,ln_pass_form,order",
    [
        ("k1", 128, "registers", "major"),
        ("k1", 256, "smem_rolled_pf", "major"),
        ("k3", 128, "registers", "interleaved"),
        ("k3", 256, "smem_rolled_pf", "interleaved"),
    ],
)
def test_sm107_packed_bias_forms_render_one_cubin_without_biases(tmp_path, arm, k, ln_pass_form, order):
    """The two bias-path forms of the packed epilogue (`packed_hoist`, `packed_biasfold`) differ from `packed` ONLY under `has_bias`: trace-compiled
    for sm_107a WITHOUT biases (no GPU) the three forms are byte-identical cubins -- the bias-off production renderings cannot move when a bias-on
    cell picks a new form, and the register-split rule may key them as one rendering (`heuristics.register_split_hint`)."""
    md5 = {}
    for epm in sub.EPI_MATHS_PACKED:
        probe = sass_utils.run_sass_probe(
            tmp_path,
            probe_src=sass_utils.sass_probe_source(_SASS_PROBE_BODY, _SASS_COUNTS),
            arch="sm_107a",
            params=dict(arm=arm, k=k, ln_pass_form=ln_pass_form, epi_math=epm, has_bias=False, min_blocks_per_mp=0, pass_order=order),
            tag=f"ln_dual_gemm_{arm}_k{k}_{ln_pass_form}_{epm}_bias0_{order}",
        )
        md5[epm] = probe.cubin_md5
        print(f"{arm} K={k} {ln_pass_form} {epm} bias off: cubin {probe.cubin_md5} FADD2 {probe.stats.get('FADD2')}")
    assert len(set(md5.values())) == 1 and all(md5.values()), f"{arm} K={k} {ln_pass_form}: the packed forms differ WITHOUT biases: {md5}"


@pytest.mark.parametrize("arm,k,ln_pass_form,epi_math", sorted(_SASS_SPILL_PINS))
def test_sm107_rendering_spill_pins(tmp_path, arm, k, ln_pass_form, epi_math):
    """Trace-compile the rendering for sm_107a (no GPU needed), read STL / LDL / USETMAXREG off the SASS: the served widths' spill counts are
    pinned per arm and LayerNorm / epilogue form, the dropped register split is documented by USETMAXREG == 0."""
    probe = sass_utils.run_sass_probe(
        tmp_path,
        probe_src=sass_utils.sass_probe_source(_SASS_PROBE_BODY, _SASS_COUNTS),
        arch="sm_107a",
        params=dict(arm=arm, k=k, ln_pass_form=ln_pass_form, epi_math=epi_math),
        tag=f"ln_dual_gemm_{arm}_k{k}_{ln_pass_form}_{epi_math}",
    )
    st = probe.stats
    print(f"{arm} K={k} {ln_pass_form} {epi_math}: STL {st['STL']} LDL {st['LDL']} USETMAXREG {st['USETMAXREG']} lines {st['LINES']} cubin {probe.cubin_md5}")
    sass_utils.assert_no_new_spills(st, _SASS_SPILL_PINS[(arm, k, ln_pass_form, epi_math)], tag=f"{arm} K={k} {ln_pass_form} {epi_math}: ")
    assert (
        st["USETMAXREG"] == 0
    ), f"{arm} K={k}: the rendering now carries a REAL register split (USETMAXREG {st['USETMAXREG']}) -- re-measure the roles and the spill pins"


# The sm_100a renderings the triangle multiplicative update ships on Blackwell (cc 10.0 / 10.3): the SAME parameter sets as the heuristics'
# picks (the register / rolled / pipelined LayerNorm forms, the packed epilogue, the LayerNorm-once block item where the pick carries it, bias on
# and off) rendered by the sm100 body -- a 6-deep ring (4 in the output stage), 512 non-exclusive TMEM columns, `--gpu-arch sm_100a`.  The same
# register file as Rubin, so the flat entry allocation under the dropped split (USETMAXREG 0) is the same and the spill counts are near the
# sm_107a ones; the values are BOUNDS measured on the branch's toolchain (nvdisasm of the trace-time cubin), CGAERRBAR 0 (every cluster-scope
# arrive a commit or a relaxed / release helper).  Keyed (arm, K, ln_pass_form, epi_math, has_bias, order, ln_once).
_SASS_SM100_PINS = {
    ("k1", 128, "registers", "packed", True, "interleaved", ""): dict(STL=0, LDL=0),
    ("k1", 128, "registers", "packed", False, "interleaved", ""): dict(STL=0, LDL=0),
    ("k1", 128, "registers", "packed", False, "major", "block"): dict(STL=0, LDL=0),
    ("k1", 128, "smem_rolled", "packed", True, "major", "block"): dict(STL=0, LDL=0),
    ("k1", 128, "smem_rolled", "packed", False, "major", "block"): dict(STL=3, LDL=3),
    # the SM100 column's own picks (measured on a 152-SM GB200 perf node, 2026-10-14): the register form + the plain packed epilogue from N = 512
    # (B 1) / N = 384 (B >= 4), the pipelined rolled form at the 384 bucket
    ("k1", 128, "registers", "packed", True, "major", "block"): dict(STL=0, LDL=0),
    ("k1", 128, "smem_rolled_pf", "packed", True, "major", "block"): dict(STL=0, LDL=0),
    ("k1", 128, "smem_rolled_pf", "packed", False, "major", "block"): dict(STL=0, LDL=0),
    ("k1", 256, "smem_rolled_pf", "packed", True, "major", "block"): dict(STL=0, LDL=0),
    ("k1", 256, "smem_rolled_pf", "packed", False, "major", "block"): dict(STL=1, LDL=1),
    ("k3", 128, "registers", "packed", True, "transpose", ""): dict(STL=0, LDL=0),
    ("k3", 128, "registers", "packed", False, "transpose", ""): dict(STL=0, LDL=0),
    ("k3", 256, "smem_rolled_pf", "packed", True, "mn_major", "block"): dict(STL=0, LDL=0),
    # the bias-off output-stage c 256 rendering sits at ptxas's flat 80-register allocation (the two-CTAs-per-SM target) and spills a few epilogue
    # words there, as its sm_107a twin does (5 / 6 under the block item): 6 / 7 measured at the 2-deep scheduler ring this probe renders
    ("k3", 256, "smem_rolled_pf", "packed", False, "mn_major", "block"): dict(STL=6, LDL=7),
}


# The SM100 column's register-split row (`heuristics._REGISTER_SPLIT_HINTS_SM100`, measured on a 152-SM GB200 perf node, 2026-10-14): the c 128
# register-form K1 picks with the plain packed epilogue run under `.minnctapersm 1` with the biases -- the split honoured (USETMAXREG 6), no spill
# either way on sm_100a (K3 is unhinted on SM100: its hint-0 rendering is in `_SASS_SM100_PINS`).  Keyed (arm, K, ln_pass_form, epi_math,
# has_bias, min_blocks_per_mp, order, ln_once); STL / LDL are BOUNDS, USETMAXREG is exact.
_SASS_SM100_HINT_PINS = {
    ("k1", 128, "registers", "packed", True, 1, "major", "block"): dict(STL=0, LDL=0, USETMAXREG=6),
}


@pytest.mark.parametrize("arm,k,ln_pass_form,epi_math,has_bias,hint,order,ln_once", sorted(_SASS_SM100_HINT_PINS))
def test_sm100_renderings_under_the_hint_spill_pins(tmp_path, arm, k, ln_pass_form, epi_math, has_bias, hint, order, ln_once):
    """Trace-compile the SM100 column's hinted production rendering (and the output stage's twins) for sm_100a (no GPU): STL / LDL within the pins,
    USETMAXREG EXACTLY the pinned value (6 = the honoured split the SM100 row buys; 0 = the unhinted twin), CGAERRBAR 0."""
    probe = sass_utils.run_sass_probe(
        tmp_path,
        probe_src=sass_utils.sass_probe_source(_SASS_PROBE_BODY, _SASS_COUNTS),
        arch="sm_100a",
        params=dict(
            arm=arm,
            k=k,
            ln_pass_form=ln_pass_form,
            epi_math=epi_math,
            has_bias=has_bias,
            min_blocks_per_mp=hint,
            pass_order=order,
            ln_once=ln_once,
            body_arch="sm100",
        ),
        tag=f"ln_dual_gemm_sm100_hint_{arm}_k{k}_{ln_pass_form}_{epi_math}_bias{int(has_bias)}_h{hint}_{order}_{ln_once or 'tile'}",
    )
    st = probe.stats
    pins = _SASS_SM100_HINT_PINS[(arm, k, ln_pass_form, epi_math, has_bias, hint, order, ln_once)]
    print(
        f"sm100 {arm} K={k} {ln_pass_form} {epi_math} bias {has_bias} hint {hint} {order} ln_once={ln_once}: STL {st['STL']} LDL {st['LDL']} USETMAXREG {st['USETMAXREG']} "
        f"CGAERRBAR {st['CGAERRBAR']} lines {st['LINES']} cubin {probe.cubin_md5} module {probe.expect}"
    )
    assert probe.expect["EXPECT_TMEM_COLS"] == 512
    sass_utils.assert_no_new_spills(st, pins, tag=f"sm100 {arm} K={k} {ln_pass_form} {epi_math} bias {has_bias} hint {hint}: ")
    assert (
        st["USETMAXREG"] == pins["USETMAXREG"]
    ), f"sm100 {arm} K={k} hint {hint}: USETMAXREG {st['USETMAXREG']} != {pins['USETMAXREG']} -- re-measure the roles and the pins"
    assert st["CGAERRBAR"] == 0


@pytest.mark.parametrize("arm,k,ln_pass_form,epi_math,has_bias,order,ln_once", sorted(_SASS_SM100_PINS))
def test_sm100_rendering_spill_pins(tmp_path, arm, k, ln_pass_form, epi_math, has_bias, order, ln_once):
    """Trace-compile the sm100 body's production renderings for sm_100a (no GPU): the ring depth and TMEM columns the body resolved (6 / 4
    stages, 512 columns), STL / LDL within the measured bounds, USETMAXREG 0 (the dropped split, as on Rubin), CGAERRBAR 0."""
    probe = sass_utils.run_sass_probe(
        tmp_path,
        probe_src=sass_utils.sass_probe_source(_SASS_PROBE_BODY, _SASS_COUNTS),
        arch="sm_100a",
        params=dict(arm=arm, k=k, ln_pass_form=ln_pass_form, epi_math=epi_math, has_bias=has_bias, pass_order=order, ln_once=ln_once, body_arch="sm100"),
        tag=f"ln_dual_gemm_sm100_{arm}_k{k}_{ln_pass_form}_{epi_math}_bias{int(has_bias)}_{order}_{ln_once or 'tile'}",
    )
    st = probe.stats
    print(
        f"sm100 {arm} K={k} {ln_pass_form} {epi_math} bias {has_bias} {order} ln_once={ln_once}: STL {st['STL']} LDL {st['LDL']} USETMAXREG {st['USETMAXREG']} "
        f"CGAERRBAR {st['CGAERRBAR']} lines {st['LINES']} cubin {probe.cubin_md5} module {probe.expect}"
    )
    assert probe.expect["EXPECT_AB_STAGES"] == sub.ring_stages("sm100", "accumulators" if arm == "k3" else "columns") == (4 if arm == "k3" else 6)
    assert probe.expect["EXPECT_TMEM_COLS"] == 512
    sass_utils.assert_no_new_spills(
        st, _SASS_SM100_PINS[(arm, k, ln_pass_form, epi_math, has_bias, order, ln_once)], tag=f"sm100 {arm} K={k} {ln_pass_form} {epi_math} bias {has_bias}: "
    )
    assert st["USETMAXREG"] == 0, f"sm100 {arm} K={k}: a REAL register split (USETMAXREG {st['USETMAXREG']}) -- re-measure the roles and the pins"
    assert st["CGAERRBAR"] == 0, f"sm100 {arm} K={k}: a non-relaxed cluster arrive (CGAERRBAR {st['CGAERRBAR']}) -- a GPU-scope drain per arrive"


@pytest.mark.parametrize(
    "kw,axis",
    [
        (dict(ln_pass_form="rows"), "ln_pass_form"),
        (dict(epi_math="vector"), "epi_math"),
    ],
)
def test_ln_pass_form_and_epi_math_decline_typed(kw, axis):
    """The two substrate forms are a closed vocabulary (performance only); a value outside it is a typed ValueError naming the axis."""
    with pytest.raises(ValueError, match=axis):
        sub.validate_ln_dual_gemm_params(k1.proj_params(128, 128, has_bias=True, has_mask=True, **kw))


@requires_served_arch
@pytest.mark.parametrize(
    "n_res,batch,c_z,c,mask_kind,ln_once,clc_stages",
    [
        (217, 2, 128, 128, "random", "block", 2),
        (128, 1, 256, 256, "padded", "block", 1),
        (384, 1, 256, 256, "random", "passes", 1),
    ],
)
def test_ln_once_item_is_bitwise_the_per_tile_item(n_res, batch, c_z, c, mask_kind, ln_once, clc_stages):
    """The LayerNorm-once work item (the row block's A operand resident in SMEM across its passes x N tiles; T27) computes the SAME planes and
    statistics as the per-tile item of the same knob set, BITWISE: the same normalised bf16 words feed every tile of the item.  One cell per public
    ``ln_once`` value on its geometry: 217 = a padded tail (block), 128 at c 256 (block), 384 = two 128-row k blocks per 256 rows with the "passes"
    item at c 256 = two N-tile items per block (the N 33 partial row block is the host geometry test's).  Two ON launches bitwise equal (the
    two-launch trick); the ON output within the fp64 oracle's derived bound."""
    inp = _inputs(n_res, batch, c_z, c, mask_kind, True, seed=11)
    kn = th_recommend(n_res, c_z, batch)
    common = dict(has_bias=True, has_mask=True, want_stats=True, pass_order="major", clc_stages=clc_stages, ln_pass_form=kn.ln_pass_form, epi_math="packed")
    plan_off = _build_k1(k1.proj_params(c_z, c, **common))
    plan_on = _build_k1(k1.proj_params(c_z, c, ln_once=ln_once, **common))
    stream = torch.cuda.current_stream().cuda_stream
    outs = []
    for plan in (plan_off, plan_on, plan_on):
        planes = torch.full(k1.planes_shape(c, batch, n_res), _SENTINEL, device="cuda", dtype=torch.bfloat16)
        stats = torch.full((batch * n_res * n_res, 2), -1.0, device="cuda")
        _launch(plan, inp, c, planes, stats, stream, k_block=kn.k_block)
        torch.cuda.synchronize()
        outs.append((planes, stats))
    assert torch.equal(outs[1][0], outs[0][0]) and torch.equal(outs[1][1], outs[0][1]), "the LayerNorm-once item must be bitwise the per-tile item"
    assert torch.equal(outs[2][0], outs[1][0]) and torch.equal(outs[2][1], outs[1][1]), "two launches of the LayerNorm-once item differ (a first-launch race)"
    valid = outs[1][0][:, :, :, :n_res].float()
    assert not (valid == _SENTINEL).any(), "a never-written cell survived"
    ref, _, _ = _reference(inp, c, 1e-5)
    diff = (valid.double() - ref).abs()
    assert int((diff > 4e-3 + ref.abs() / 128.0).sum()) == 0, f"max |diff| {diff.max().item():.4g}"


def th_recommend(n_res, c_z, batch):
    from cudnn.triangle_multiplicative_update import heuristics as th

    return th.recommend(n_res, c_z, batch, "outgoing")


@requires_served_arch
@pytest.mark.parametrize("n_res,batch,mask_kind,b_stages", [(128, 1, "padded", 4)])
def test_ln_once_b_ring_depth_is_bitwise_the_shipped_ring(n_res, batch, mask_kind, b_stages):
    """The LayerNorm-once item's B-ring depth (`ln_once_b_stages`, B-TriMul-1): the K1 c 256 rendering with the shallower B ring (the second resident
    A slot on the sm100 body; a no-op on the sm107 body, whose ring already holds two slots) computes the SAME planes and statistics as the shipped
    ring's rendering, BITWISE (the same normalised words, the same k-step order; only the ring index space changes).  Two launches bitwise equal; the
    output within the fp64 oracle's derived bound; a sentinel fill leaves no survivor."""
    c_z = c = 256
    inp = _inputs(n_res, batch, c_z, c, mask_kind, True, seed=13)
    kn = th_recommend(n_res, c_z, batch)
    common = dict(
        has_bias=True, has_mask=True, want_stats=True, pass_order="major", clc_stages=1, ln_pass_form=kn.ln_pass_form, epi_math="packed", ln_once="block"
    )
    plan_ring = _build_k1(k1.proj_params(c_z, c, **common))
    plan_lever = _build_k1(k1.proj_params(c_z, c, ln_once_b_stages=b_stages, **common))
    stream = torch.cuda.current_stream().cuda_stream
    outs = []
    for plan in (plan_ring, plan_lever, plan_lever):
        planes = torch.full(k1.planes_shape(c, batch, n_res), _SENTINEL, device="cuda", dtype=torch.bfloat16)
        stats = torch.full((batch * n_res * n_res, 2), -1.0, device="cuda")
        _launch(plan, inp, c, planes, stats, stream, k_block=kn.k_block)
        torch.cuda.synchronize()
        outs.append((planes, stats))
    assert torch.equal(outs[1][0], outs[0][0]) and torch.equal(outs[1][1], outs[0][1]), "the shallower B ring must be bitwise the shipped ring"
    assert torch.equal(outs[2][0], outs[1][0]) and torch.equal(outs[2][1], outs[1][1]), "two launches under the shallower B ring differ (a first-launch race)"
    valid = outs[1][0][:, :, :, :n_res].float()
    assert not (valid == _SENTINEL).any(), "a never-written cell survived"
    ref, _, _ = _reference(inp, c, 1e-5)
    diff = (valid.double() - ref).abs()
    assert int((diff > 4e-3 + ref.abs() / 128.0).sum()) == 0, f"max |diff| {diff.max().item():.4g}"


# the four shipped (form, epilogue) pairs on their geometry + the two public values no pick uses (`smem`, `scalar`, `packed_hoist`) once
_FORM_CELLS = [
    ("registers", "packed", 217, 2, 128, 128, "random"),
    ("smem_rolled", "packed", 217, 2, 128, 128, "random"),
    ("registers", "packed_biasfold", 217, 2, 128, 128, "random"),
    ("smem_rolled_pf", "packed", 128, 1, 256, 256, "padded"),
    ("smem", "scalar", 128, 1, 256, 256, "padded"),
    ("registers", "packed_hoist", 217, 2, 128, 128, "random"),
]


@requires_served_arch
@pytest.mark.parametrize(
    "ln_pass_form,epi_math,n_res,batch,c_z,c,mask_kind", _FORM_CELLS, ids=[f"{f}-{e}-{n}-{b}-{cz}-{c}-{m}" for f, e, n, b, cz, c, m in _FORM_CELLS]
)
def test_ln_pass_form_and_epi_math_are_bitwise_equal_to_the_default_forms(n_res, batch, c_z, c, mask_kind, ln_pass_form, epi_math):
    """The re-read LayerNorm forms sum the same bf16 words in the same order (a re-load of the unchanged stage replaces a register), and the packed
    epilogue runs the scalar form's fp32 operations in its association on fp32x2 pairs: planes AND statistics BITWISE equal to the default forms --
    every public ``ln_pass_form`` / ``epi_math`` value once (the tree form has its own oracle-bound test)."""
    probe = k1.proj_params(c_z, c, has_bias=True, has_mask=True, want_stats=True, ln_pass_form=ln_pass_form, epi_math=epi_math)
    if _sm100_c7600_cell(probe):
        # the sm100 body's typed decline (ptxas C7600 at K = 256 for the register form with a bias-path packed epilogue and the statistics write)
        with pytest.raises(ValueError, match="C7600"):
            _build_k1(probe)
        return
    inp = _inputs(n_res, batch, c_z, c, mask_kind, True, seed=13)
    outs = []
    for lnf, epm in (("registers", "scalar"), (ln_pass_form, epi_math)):
        plan = _build_k1(k1.proj_params(c_z, c, has_bias=True, has_mask=True, want_stats=True, ln_pass_form=lnf, epi_math=epm))
        planes = torch.full(k1.planes_shape(c, batch, n_res), _SENTINEL, device="cuda", dtype=torch.bfloat16)
        stats = torch.full((batch * n_res * n_res, 2), -1.0, device="cuda")
        _launch(plan, inp, c, planes, stats, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert not (planes[:, :, :, :n_res] == _SENTINEL).any(), f"{lnf}/{epm}: a never-written cell in the valid region"
        outs.append((planes, stats))
    assert torch.equal(outs[0][0], outs[1][0]), f"{ln_pass_form}/{epi_math} changed the planes"
    assert torch.equal(outs[0][1], outs[1][1]), f"{ln_pass_form}/{epi_math} changed the statistics"


@requires_served_arch
@pytest.mark.parametrize("n_res,batch,c_z,c,mask_kind", [(217, 2, 128, 128, "random")])
def test_tree_ln_form_meets_the_oracle_bound_and_its_distance_to_the_rolled_form_is_measured(n_res, batch, c_z, c, mask_kind):
    """`ln_pass_form="smem_rolled_tree"` sums each statistics pass in FOUR partials (a different fp32 ORDER, :data:`LN_PASS_FORMS_REORDERED`).  A
    summation-order change moves a few LayerNorm outputs across a bf16 rounding midpoint, and those flips propagate through the GEMM exactly like
    the dataset-dependent flips the shipped form shows on other seeds, so BOTH forms are held to the API's documented three-part oracle on the same
    inputs: the bulk bound `4e-3 + |ref| / 128` on all but a tail of at most 2e-5 of the cells, the cap `2e-2 + |ref| / 64` on EVERY cell, rel RMS
    < 2e-3 (nothing widened: the module's bounds), the statistics within the oracle test's tolerance, and the distance to the rolled form REPORTED
    (cells differing, the statistics' max |diff|) -- a measured, named form, never a bitwise pin and never a silent default."""
    eps = 1e-5
    inp = _inputs(n_res, batch, c_z, c, mask_kind, True, seed=13)
    ref, mean64, rstd64 = _reference(inp, c, eps)
    outs = {}
    for lnf in ("smem_rolled", "smem_rolled_tree"):
        plan = _build_k1(k1.proj_params(c_z, c, has_bias=True, has_mask=True, want_stats=True, eps=eps, ln_pass_form=lnf, epi_math="packed"))
        planes = torch.full(k1.planes_shape(c, batch, n_res), _SENTINEL, device="cuda", dtype=torch.bfloat16)
        stats = torch.full((batch * n_res * n_res, 2), -1.0, device="cuda")
        _launch(plan, inp, c, planes, stats, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        valid = planes[:, :, :, :n_res].float()
        assert not (valid == _SENTINEL).any(), f"{lnf}: a never-written cell in the valid region"
        diff = (valid.double() - ref).abs()
        bulk = 4e-3 + ref.abs() / 128.0  # the module's derived bound (test_k1_matches_the_fp64_oracle), unchanged
        cap = 2e-2 + ref.abs() / 64.0
        bad_bulk, bad_cap = int((diff > bulk).sum().item()), int((diff > cap).sum().item())
        rel_rms = (diff.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        print(
            f"K1 {lnf} {n_res}:{batch}:{c_z}: {bad_bulk} of {diff.numel()} cells beyond the bulk bound, {bad_cap} beyond the cap; max |diff| {diff.max().item():.4g}; rel RMS {rel_rms:.3g}"
        )
        assert bad_cap == 0 and bad_bulk <= 2e-5 * diff.numel() and rel_rms < 2e-3, f"K1 {lnf}: {bad_bulk} bulk / {bad_cap} cap / rel RMS {rel_rms:.3g}"
        torch.testing.assert_close(stats[:, 0].double(), mean64, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(stats[:, 1].double(), rstd64, rtol=1e-4, atol=1e-5)
        outs[lnf] = (planes, stats)
    n_diff = int((outs["smem_rolled"][0].view(torch.int16) != outs["smem_rolled_tree"][0].view(torch.int16)).sum().item())
    d_stats = (outs["smem_rolled"][1] - outs["smem_rolled_tree"][1]).abs()
    print(
        f"tree vs rolled {n_res}:{batch}:{c_z}: {n_diff} plane cells differ (of {outs['smem_rolled'][0].numel()}); "
        f"max |d mean| {d_stats[:, 0].max().item():.3g}, max |d rstd| {d_stats[:, 1].max().item():.3g}"
    )
    assert sub.LN_PASS_FORMS_REORDERED == ("smem_rolled_tree",)  # the one form outside the bitwise pins, by name


@requires_served_arch
def test_skip_vote_is_bitwise_equal_to_the_unskipped_arm():
    """A padded protein masks whole 256-row tiles (every pair row with i >= L): the skipped-MMA arm and the control arm that
    computes everything must agree BITWISE on the planes and the statistics."""
    n_res, batch, c_z, c = 256, 2, 128, 128
    inp = _inputs(n_res, batch, c_z, c, "padded", True, seed=3)
    outs = []
    for skip in (True, False):
        plan = _build_k1(k1.proj_params(c_z, c, has_bias=True, has_mask=True, want_stats=True, skip_masked_tiles=skip))
        planes = torch.full(k1.planes_shape(c, batch, n_res), _SENTINEL, device="cuda", dtype=torch.bfloat16)
        stats = torch.full((batch * n_res * n_res, 2), -1.0, device="cuda")
        _launch(plan, inp, c, planes, stats, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        outs.append((planes, stats))
    assert torch.equal(outs[0][0], outs[1][0]), "the warp-vote skip changed the output"
    assert torch.equal(outs[0][1], outs[1][1]), "the warp-vote skip changed the statistics"
    m = inp["mask"].reshape(-1)
    assert (m == 0).sum() > 0 and (outs[0][0][:, :, :, :n_res].reshape(2, c, -1)[:, :, m == 0].view(torch.int16) == 0).all()


@requires_served_arch
def test_mask_multiplies_in_fp32_before_the_bf16_rounding():
    """The pin that separates the kernel's mask semantics (``bf16(sigmoid(g) * v * m)``, one rounding) from cuEquivariance's
    after-rounding form (``bf16(bf16(sigmoid(g) * v) * m)``, two roundings): with FRACTIONAL mask weights the two fp64 models
    differ by one bf16 ulp on a sizeable fraction of the cells, and the kernel must side with the before-rounding model there.
    (``{0, 1}`` and power-of-two masks cannot tell the two apart -- which is why every other cell passes under both.)"""
    n_res, batch, c_z, c = 128, 1, 128, 128
    eps = 1e-5
    inp = _inputs(n_res, batch, c_z, c, "fractional", True, seed=11)
    plan = _build_k1(k1.proj_params(c_z, c, has_bias=True, has_mask=True, want_stats=False, eps=eps))
    planes = torch.full(k1.planes_shape(c, batch, n_res), _SENTINEL, device="cuda", dtype=torch.bfloat16)
    _launch(plan, inp, c, planes, None, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    ref, _, _ = _reference(inp, c, eps)  # fp64 sigmoid(g) * v * m, unrounded
    m = inp["mask"].double().reshape(-1)
    live = m != 0
    ref_rows = ref.reshape(2, c, -1)[:, :, live]  # [2, c, live rows]: fp64 sigmoid(g) * v * m
    sv = ref_rows / m[live][None, None, :]  # fp64 sigmoid(g) * v on the live rows
    pre = ref_rows.to(torch.bfloat16).view(torch.int16)  # ONE rounding after the mask
    post = (sv.to(torch.bfloat16).double() * m[live][None, None, :]).to(torch.bfloat16).view(torch.int16)  # round, mask, round
    got = planes[:, :, :, :n_res].reshape(2, c, -1)[:, :, live].view(torch.int16)
    distinguishing = pre != post
    frac_dist = distinguishing.double().mean().item()
    match_pre = (got == pre)[distinguishing].double().mean().item()
    match_post = (got == post)[distinguishing].double().mean().item()
    msg = f"distinguishing cells {100 * frac_dist:.1f} %: kernel == before-rounding model on {100 * match_pre:.1f} %, == after-rounding model on {100 * match_post:.1f} %"
    print(msg)  # the measured fractions (pytest -rA shows them for a passing test)
    assert frac_dist > 0.05, msg  # the fractional mask must actually separate the two models
    # measured on the dev node 2026-10-07 (N = 128, seed 11): 25.8 % distinguishing cells, before-rounding 99.9 %, after-rounding 0.1 %
    assert match_pre > 0.9 and match_post < 0.1, msg


@requires_served_arch
@pytest.mark.parametrize("n_res,batch,c_z,c,mask_kind", [(217, 2, 128, 128, "random")])  # interleaved ships at c 128 below N 256 only
def test_interleaved_pass_order_is_bitwise_equal_to_pass_major(n_res, batch, c_z, c, mask_kind):
    """``pass_order="interleaved"`` (the pass folded into the scheduler's M axis so both passes of a row block are adjacent -- the
    shared-z tile order) computes the same tiles with the same arithmetic: planes and statistics BITWISE equal to ``"major"``."""
    inp = _inputs(n_res, batch, c_z, c, mask_kind, True, seed=7)
    outs = []
    for order in ("major", "interleaved"):
        plan = _build_k1(k1.proj_params(c_z, c, has_bias=True, has_mask=True, want_stats=True, pass_order=order))
        planes = torch.full(k1.planes_shape(c, batch, n_res), _SENTINEL, device="cuda", dtype=torch.bfloat16)
        stats = torch.full((batch * n_res * n_res, 2), -1.0, device="cuda")
        _launch(plan, inp, c, planes, stats, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert not (planes[:, :, :, :n_res] == _SENTINEL).any(), f"{order}: a never-written cell in the valid region"
        outs.append((planes, stats))
    assert torch.equal(outs[0][0], outs[1][0]), "the interleaved tile order changed the planes"
    assert torch.equal(outs[0][1], outs[1][1]), "the interleaved tile order changed the statistics"


@requires_served_arch
def test_no_mask_artifact_matches_the_all_valid_mask():
    """``has_mask=False`` (no mask tensor at the ABI) equals the all-valid mask run, cell for cell."""
    n_res, batch, c_z, c = 128, 1, 128, 128
    inp = _inputs(n_res, batch, c_z, c, "all", True, seed=5)
    stream = torch.cuda.current_stream().cuda_stream
    plan_m = _build_k1(k1.proj_params(c_z, c, has_bias=True, has_mask=True, want_stats=False))
    plan_n = _build_k1(k1.proj_params(c_z, c, has_bias=True, has_mask=False, want_stats=False))
    out_m = torch.full(k1.planes_shape(c, batch, n_res), _SENTINEL, device="cuda", dtype=torch.bfloat16)
    out_n = torch.empty_like(out_m)
    w_stack = k1.stack_projection_weights(inp["p_in"], inp["g_in"])
    b_stack = k1.stack_projection_biases(inp["p_b"], inp["g_b"])
    k1.run_trimul_proj(plan_m, inp["z"], w_stack, out_m, inp["ln_w"], inp["ln_b"], bias_stacked=b_stack, mask=inp["mask"], stats=None, stream=stream)
    k1.run_trimul_proj(plan_n, inp["z"], w_stack, out_n, inp["ln_w"], inp["ln_b"], bias_stacked=b_stack, mask=None, stats=None, stream=stream)
    torch.cuda.synchronize()
    assert torch.equal(out_m, out_n)


@requires_served_arch
def test_runner_declines_a_plan_for_another_k_and_a_misshaped_mask():
    plan = _build_k1(k1.proj_params(128, 128, has_bias=False, has_mask=True, want_stats=False))
    z = torch.zeros(1, 16, 16, 256, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(2, 256, 256, device="cuda", dtype=torch.bfloat16)
    planes = torch.zeros(k1.planes_shape(128, 1, 16), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="compiled for K=128"):
        k1.run_trimul_proj(
            plan,
            z,
            w,
            planes,
            torch.zeros(256, device="cuda", dtype=torch.bfloat16),
            torch.zeros(256, device="cuda", dtype=torch.bfloat16),
            mask=torch.ones(1, 16, 16, device="cuda", dtype=torch.bfloat16),
            stream=0,
        )
    z = torch.zeros(1, 16, 16, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="mask must be"):
        k1.run_trimul_proj(
            plan,
            z,
            w,
            planes,
            torch.zeros(128, device="cuda", dtype=torch.bfloat16),
            torch.zeros(128, device="cuda", dtype=torch.bfloat16),
            mask=torch.ones(1, 16, 15, device="cuda", dtype=torch.bfloat16),
            stream=0,
        )


@requires_served_arch
@pytest.mark.parametrize("n_res,batch,k_block", [(217, 1, 32)])  # the K1-only domain end; the host geometry test pins the arithmetic, K3 keeps 64
def test_k1_row_tile_knob_is_bitwise_the_rule(n_res, batch, k_block):
    """The row tile (k's per 256-row pair tile) is a RUNTIME operand of the compiled kernel: the same plan launched with ``k_block`` computes the
    SAME bits as the rule (planes AND statistics) -- a row's LayerNorm, MMA K order and epilogue do not depend on the tile it sits in.  At N = 384
    the rule pads 25 % of every pair tile (kb = 256 over 384 k); kb = 128 / 64 pad none (fewer scheduler tiles, the same per-CTA TMA box at 128)."""
    c_z = c = 128
    params = k1.proj_params(c_z, c, has_bias=True, has_mask=True, want_stats=True, eps=1e-5)
    plan = _build_k1(params)
    inp = _inputs(n_res, batch, c_z, c, "random", True, seed=31)
    stream = torch.cuda.current_stream().cuda_stream
    outs = []
    for kb in (None, k_block):
        planes = torch.full(k1.planes_shape(c, batch, n_res), _SENTINEL, device="cuda", dtype=torch.bfloat16)
        stats = torch.full((batch * n_res * n_res, 2), -1.0, device="cuda", dtype=torch.float32)
        g = _launch(plan, inp, c, planes, stats, stream, k_block=kb)
        torch.cuda.synchronize()
        assert g.kb == (sub.default_k_block(n_res) if kb is None else kb) and g.rows_valid == batch * n_res * n_res
        assert not (planes[:, :, :, :n_res].float() == _SENTINEL).any() and (stats[:, 1] > 0).all()
        outs.append((planes, stats, g))
    (p0, s0, g0), (p1, s1, g1) = outs
    assert torch.equal(p0, p1) and torch.equal(s0, s1), f"the row tile changed the output bits (kb {g0.kb} -> {g1.kb})"
    if n_res == 384:
        assert g0.rows_computed == 4 * g0.rows_valid // 3 and g1.rows_computed == g1.rows_valid  # the 25 % the lever removes
