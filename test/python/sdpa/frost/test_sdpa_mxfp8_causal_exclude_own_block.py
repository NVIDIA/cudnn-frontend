# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""``sdpa_mxfp8(causal_exclude_own_block=32)``: causal attention over completed V scale blocks only.

The op attribute tightens the causal bound to keys ``j < 32 * floor(i / 32)``, so every MXFP8 V
scale block the kernel reads covers fully visible tokens (NVIDIA/TransformerEngine#3248: an
ordinary columnwise V scale spans 32 tokens, so a token later in the query's own block can change
how an earlier, visible value is rounded).  Rows 0..31 keep no key (O = 0, Stats = -inf).  The
caller adds the query's own block and merges the two partials through their Stats.

Host-only (fake SM100 device): the attribute is a graph fact, never a knob; malformed requests are
invalid; the SM100 MXFP8 row serves D128 only (split-KV and PackGQA legs included) and every other
row declines it by name.
GPU (pre-Rubin SM100 line): numerics against the fp32 reference and bitwise invariance of every row
to the keys of its own and later blocks, on the default plan and on pinned PackGQA and split-KV plans.  Replay, batch invariance and the golden live in
``test_sdpa_mxfp8_own_block_determinism.py``.
"""

import math
from types import SimpleNamespace

import pytest
import torch

import cudnn
from cudnn.sdpa import graph_analyzer as ga
from cudnn.sdpa.fwd import engines

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
from mxfp8_own_block_plan import _E8M0, _F8_128X4, _FP8, _PIN, OWN, _inputs, _Plan
from mxfp8_test_reference import _ref
from test_utils import torch_fork_set_rng

# ---------------------------------------------------------------------------
# Host-only routing contract
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_sm100(monkeypatch):
    """Fake an SM100 device so the device-family gate passes without a real GPU (as test_sdpa_graph_analyzer.py)."""
    monkeypatch.setattr(ga, "_device_cc", lambda: (10, 0))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (10, 0))


def _mxfp8_graph(d=128, *, b=2, h=8, h_kv=None, s=256, s_kv=None, **sdpa_kwargs):
    """A dense BSHD sdpa_mxfp8 graph with half O and Stats; ``sdpa_kwargs`` go to the op verbatim."""
    h_kv = h if h_kv is None else h_kv
    s_kv = s if s_kv is None else s_kv
    dims, bshd = (b, h, s, d), (s * h * d, d, h * d, 1)
    kv_dims, kv_bshd = (b, h_kv, s_kv, d), (s_kv * h_kv * d, d, h_kv * d, 1)
    g = cudnn.pygraph(io_data_type=cudnn.data_type.FP8_E4M3, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q = g.tensor(dim=dims, stride=bshd, data_type=cudnn.data_type.FP8_E4M3, name="q")
    k, v = (g.tensor(dim=kv_dims, stride=kv_bshd, data_type=cudnn.data_type.FP8_E4M3, name=n) for n in "kv")

    def sf(sd):  # F8_128x4 scale factors: Q/K rowwise [B,H,S,d/32], V columnwise [B,H,S/32,d]
        return g.tensor(dim=list(sd), stride=[sd[1] * sd[2] * sd[3], sd[2] * sd[3], sd[3], 1], data_type=_E8M0, reordering_type=_F8_128X4)

    o, stats, _ = g.sdpa_mxfp8(
        q=q,
        k=k,
        v=v,
        descale_q=sf((b, h, s, d // 32)),
        descale_k=sf((b, h_kv, s_kv, d // 32)),
        descale_v=sf((b, h_kv, -(-s_kv // 32), d)),
        attn_scale=1.0 / math.sqrt(d),
        generate_stats=True,
        **sdpa_kwargs,
    )
    o.set_output(True).set_dim(dims).set_stride(bshd).set_data_type(cudnn.data_type.HALF)
    stats.set_output(True).set_dim((b, h, s, 1)).set_stride((h * s, s, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    return g


def _eligible(graph, knobs=None):
    return {s.name for s in engines.ENGINE_SPECS if engines.analyze_for(s, graph, knobs)[1] is None}


def _reason(graph, name, knobs=None):
    spec = next(s for s in engines.ENGINE_SPECS if s.name == name)
    return engines.analyze_for(spec, graph, knobs)[1]


_MXFP8_ROW = engines.engine_name(mxfp8=True)


@pytest.mark.L0
def test_own_block_is_an_op_attribute_and_a_fact(fake_sm100):
    """Numerics-changing (the own block leaves the softmax), so a graph FACT the rows gate on -- not a knob."""
    assert "causal_exclude_own_block" not in engines.SdpaFwdKnobs.__dataclass_fields__
    g = _mxfp8_graph(use_causal_mask=True, causal_exclude_own_block=OWN)
    facts = ga.analyze(g)
    assert facts.invalid is None and facts.causal_exclude_own_block == OWN
    # The SM100 MXFP8 row serves it, and it is the only row that does.
    assert _eligible(g) == {_MXFP8_ROW}
    # The backend has no field for it: a SET value keeps the node off the backend; unset leaves it lowerable.
    assert g._unlowerable_node() is not None
    assert _mxfp8_graph(use_causal_mask=True)._unlowerable_node() is None
    assert ga.analyze(_mxfp8_graph(use_causal_mask=True)).causal_exclude_own_block == 0


@pytest.mark.L0
@pytest.mark.parametrize("value", [-32, 48, True, 32.0, "32"])
def test_malformed_block_size_is_invalid(fake_sm100, value):
    g = _mxfp8_graph(use_causal_mask=True, causal_exclude_own_block=value)
    assert "causal_exclude_own_block must be a positive power-of-two" in (ga.analyze(g).invalid or "")
    assert not _eligible(g)


@pytest.mark.L0
def test_zero_is_plain_causal(monkeypatch, fake_sm100):
    """``causal_exclude_own_block=0`` is the off value (the kernel's default): the same facts as an
    unset attribute, backend-lowerable, and no unlocking of the opt-in slot."""
    from cudnn.engines import manifest

    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    g = _mxfp8_graph(use_causal_mask=True, causal_exclude_own_block=0)
    facts = ga.analyze(g)
    assert facts.invalid is None and facts.causal_exclude_own_block == 0
    assert g._unlowerable_node() is None
    assert _PIN not in manifest.family_for(g).offered_ids(g)


@pytest.mark.L0
def test_ragged_kv_tail_declines(fake_sm100):
    """Top-left causal with S_q > S_kv and S_kv not a KV-tile multiple needs the scalar KV-tail mask,
    which the bound is not wired with: the row declines by name instead of failing at build."""
    g = _mxfp8_graph(use_causal_mask=True, s=256, s_kv=200, causal_exclude_own_block=OWN)
    assert "scalar KV-tail mask" in (_reason(g, _MXFP8_ROW) or "")
    covered = _mxfp8_graph(use_causal_mask=True, s=200, s_kv=200, causal_exclude_own_block=OWN)
    assert _reason(covered, _MXFP8_ROW) is None


@pytest.mark.L0
@pytest.mark.parametrize(
    "mask",
    [
        {},  # no causal mask: the bound is defined on the causal diagonal
        dict(use_causal_mask=True, diagonal_band_left_bound=65),  # sliding window
        dict(diagonal_band_right_bound=8),  # widened right band (it implies the causal diagonal)
    ],
    ids=["no_causal", "swa", "right_band"],
)
def test_requires_plain_causal(fake_sm100, mask):
    g = _mxfp8_graph(causal_exclude_own_block=OWN, **mask)
    assert "causal_exclude_own_block requires a causal mask" in (ga.analyze(g).invalid or "")
    assert not _eligible(g)


@pytest.mark.L0
def test_forward_only():
    """The backward recomputes P from the forward's O / Stats; the attribute is refused there."""
    node = SimpleNamespace(params={"causal_exclude_own_block": OWN})
    facts = ga._own_block_fact(node, ga.SdpaGraphFacts(is_backward=True, is_mxfp8=True, causal=True))
    assert "forward-only" in (facts.invalid or "")


@pytest.mark.L0
def test_other_sdpa_ops_refuse_it(fake_sm100):
    """Only sdpa_mxfp8 declares the attribute; a half sdpa() graph carrying it is invalid, never served as causal."""
    dims, bshd = (2, 8, 256, 128), (256 * 8 * 128, 128, 8 * 128, 1)
    g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q, k, v = (g.tensor(dim=dims, stride=bshd, data_type=cudnn.data_type.HALF, name=n) for n in "qkv")
    o, _ = g.sdpa(q=q, k=k, v=v, attn_scale=0.1, is_inference=True, use_causal_mask=True, causal_exclude_own_block=OWN)
    o.set_output(True).set_dim(dims).set_stride(bshd).set_data_type(cudnn.data_type.HALF)
    assert "is an sdpa_mxfp8 op attribute" in (ga.analyze(g).invalid or "")
    assert not _eligible(g)


@pytest.mark.L0
def test_wider_block_size_declines_by_name(fake_sm100):
    """64 is a well-formed request the kernel's split would honor, but only the MXFP8 V block (32) is claimed."""
    g = _mxfp8_graph(use_causal_mask=True, causal_exclude_own_block=64)
    assert ga.analyze(g).invalid is None
    assert "outside the wired block sizes" in (_reason(g, _MXFP8_ROW) or "")


@pytest.mark.L0
def test_other_head_dims_decline_by_name(fake_sm100):
    g = _mxfp8_graph(256, use_causal_mask=True, causal_exclude_own_block=OWN)
    assert "wired only in the [(128, 128)] kernel flavors" in (_reason(g, _MXFP8_ROW) or "")
    assert not _eligible(g)


@pytest.mark.L0
@pytest.mark.parametrize("knobs", [engines.SdpaFwdKnobs(split_kv=2), engines.SdpaFwdKnobs(pack_gqa=True)], ids=["split_kv", "pack_gqa"])
def test_split_and_packed_legs_serve_it(fake_sm100, knobs):
    """Knobs are performance-only: the split-KV and PackGQA legs carry the bound (numerics on the GPU below)."""
    g = _mxfp8_graph(h=8, h_kv=2, use_causal_mask=True, causal_exclude_own_block=OWN)
    assert _reason(g, _MXFP8_ROW, knobs) is None


@pytest.mark.L0
def test_rubin_declines(monkeypatch, fake_sm100):
    """cc 10.7 has its own MXFP8 row without the bound: declined, never served as plain causal."""
    monkeypatch.setattr(ga, "_device_cc", lambda: (10, 7))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (10, 7))
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    assert not _eligible(_mxfp8_graph(use_causal_mask=True, causal_exclude_own_block=OWN))


@pytest.mark.L0
def test_band_form_of_causal_is_accepted(fake_sm100):
    """``diagonal_band_right_bound=0`` is the same causal mask (TransformerEngine builds it this way)."""
    g = _mxfp8_graph(diagonal_band_right_bound=0, causal_exclude_own_block=OWN)
    assert _eligible(g) == {_PIN}


@pytest.mark.L0
def test_offered_without_the_opt_in_flag(monkeypatch, fake_sm100):
    """The SM100 MXFP8 slot is opt-in, but it is the attribute's only server (the backend cannot lower the
    graph), so a graph that SETS causal_exclude_own_block is offered it with the flag unset -- as a
    candidate, as a pin, and to the facts/validator path.  Graphs without the attribute keep the gate."""
    from cudnn.engines import manifest

    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    own = _mxfp8_graph(use_causal_mask=True, causal_exclude_own_block=OWN)
    plain = _mxfp8_graph(use_causal_mask=True)
    family = manifest.family_for(own)
    slot = family.engine_id + family.slots[_PIN].slot
    assert family.offered_ids(own)[_PIN] == slot
    assert _PIN not in family.offered_ids(plain) and _PIN not in family.offered_ids()
    assert _PIN in {e.name for e in own._candidate_engines()}
    assert _PIN not in {e.name for e in plain._candidate_engines()}
    assert manifest.engine_for_id(slot, own).name == _PIN
    assert manifest.engine_for_id(slot, plain) is None and manifest.engine_for_id(slot) is None


# ---------------------------------------------------------------------------
# GPU: numerics and causality
# ---------------------------------------------------------------------------


@pytest.mark.L0
@requires_pre_rubin_blackwell
@requires_dsl
@pytest.mark.parametrize("in_key", list(_FP8))
@pytest.mark.parametrize("h_q,h_kv", [(8, 8), (8, 2)], ids=["mha", "gqa"])
@pytest.mark.parametrize("s", [128, 256, 1024])
@pytest.mark.parametrize("bottom_right", [False, True], ids=["top_left", "square_bottom_right"])
@torch_fork_set_rng(seed=0)
def test_numerics_against_reference(in_key, h_q, h_kv, s, bottom_right):
    plan = _Plan(2, h_q, h_kv, s, in_key, bottom_right=bottom_right)
    inputs, (qd, kd, vd) = plan.quantize(*_inputs(2, h_q, h_kv, s))
    o, lse = plan.execute(inputs)
    o_ref, lse_ref = _ref(qd, kd, vd, scale=1.0 / math.sqrt(128), is_causal=True, return_stats=True, exclude_own_block=OWN)
    assert not torch.isnan(o).any() and not torch.isnan(lse).any()
    # Rows 0..31 keep no key: the FROST empty-row contract, exactly.
    assert torch.equal(o[:, :, :OWN].float(), torch.zeros_like(o[:, :, :OWN].float()))
    assert torch.isneginf(lse[:, :, :OWN]).all()
    tol = 7e-2 if in_key == "e5m2" else 5e-2
    diff = (o.float() - o_ref).abs().max().item()
    assert diff <= tol, f"max|O-ref|={diff:.4f} exceeds {tol}"
    live = slice(OWN, None)
    lse_diff = (lse[:, :, live] - lse_ref[:, :, live]).abs().max().item()
    assert lse_diff <= 1e-2, f"max|LSE-ref|={lse_diff:.4f}"


@pytest.mark.L0
@requires_pre_rubin_blackwell
@requires_dsl
@pytest.mark.parametrize("in_key", list(_FP8))
@pytest.mark.parametrize("block", [0, 3, 4, 7], ids=lambda blk: f"block{blk}")
@torch_fork_set_rng(seed=0)
def test_rows_never_read_their_own_block(in_key, block):
    """Amplify the keys and values of one 32-token block by 2^10 and requantize: only that block's V scale
    blocks (and its tokens' K payloads) change.  Every row up to the end of that block must stay BITWISE
    unchanged -- including the block's own rows, whose plain-causal output would read the changed scale --
    and later rows must change (the perturbation reached the kernel)."""
    b, h, s = 2, 8, 512
    plan = _Plan(b, h, h, s, in_key)
    qf, kf, vf = _inputs(b, h, h, s)
    base, _ = plan.quantize(qf, kf, vf)
    rows = slice(block * OWN, (block + 1) * OWN)
    kf2, vf2 = kf.clone(), vf.clone()
    kf2[:, :, rows] *= 1024.0
    vf2[:, :, rows] *= 1024.0
    perturbed, _ = plan.quantize(qf, kf2, vf2)
    o_a, lse_a = plan.execute(base)
    o_b, lse_b = plan.execute(perturbed)
    end = (block + 1) * OWN
    assert torch.equal(o_a[:, :, :end].view(torch.int16), o_b[:, :, :end].view(torch.int16)), "a row read a key of its own block"
    assert torch.equal(lse_a[:, :, :end].view(torch.int32), lse_b[:, :, :end].view(torch.int32))
    assert not torch.equal(o_a[:, :, end:], o_b[:, :, end:]), "the perturbation did not reach the later rows"


@pytest.mark.L0
@requires_pre_rubin_blackwell
@requires_dsl
@pytest.mark.parametrize(
    "pack_gqa,split,b,h_q,h_kv,s",
    [
        (True, False, 2, 8, 4, 384),
        (True, False, 2, 8, 2, 1024),
        (True, False, 1, 16, 2, 2048),
        (True, False, 1, 32, 4, 1024),
        # The heuristics offer split-KV only for a launch grid small against the SM count (B200: 4 splits at
        # one head, 2 at two). Early rows have no key in the later splits, which the combine must drop.
        (False, True, 1, 1, 1, 1024),
        (False, True, 1, 1, 1, 2048),
        (False, True, 1, 2, 2, 2048),
        (True, True, 1, 2, 1, 4096),
    ],
    ids=["packed_g2", "packed_g4", "packed_g8", "packed_g8_h32", "split_s1k", "split_s2k", "split_h2", "packed_split_g2"],
)
@torch_fork_set_rng(seed=0)
def test_packed_and_split_legs(pack_gqa, split, b, h_q, h_kv, s):
    """The PackGQA and split-KV legs, pinned: numerics against the reference, empty rows 0..31, and bitwise
    invariance of every row to the keys of its own block (the leak the bound removes)."""
    plan = _Plan(b, h_q, h_kv, s, "e4m3", pack_gqa=pack_gqa, split=split)
    assert plan.knobs.pack_gqa == pack_gqa and (plan.knobs.split_kv or 1) > 1 if split else plan.knobs.pack_gqa
    qf, kf, vf = _inputs(b, h_q, h_kv, s)
    inputs, (qd, kd, vd) = plan.quantize(qf, kf, vf)
    o, lse = plan.execute(inputs)
    o_ref, lse_ref = _ref(qd, kd, vd, scale=1.0 / math.sqrt(128), is_causal=True, return_stats=True, exclude_own_block=OWN)
    assert not torch.isnan(o).any() and not torch.isnan(lse).any()
    assert torch.equal(o[:, :, :OWN].float(), torch.zeros_like(o[:, :, :OWN].float()))
    assert torch.isneginf(lse[:, :, :OWN]).all()
    diff = (o.float() - o_ref).abs().max().item()
    assert diff <= 5e-2, f"max|O-ref|={diff:.4f}"
    lse_diff = (lse[:, :, OWN:] - lse_ref[:, :, OWN:]).abs().max().item()
    assert lse_diff <= 1e-2, f"max|LSE-ref|={lse_diff:.4f}"
    # The same plan run on the same inputs, every block perturbed in turn: rows up to its end stay bitwise equal.
    for block in range(0, s // OWN, max(1, s // OWN // 6)):
        rows = slice(block * OWN, (block + 1) * OWN)
        kf2, vf2 = kf.clone(), vf.clone()
        kf2[:, :, rows] *= 1024.0
        vf2[:, :, rows] *= 1024.0
        perturbed, _ = plan.quantize(qf, kf2, vf2)
        o_b, lse_b = plan.execute(perturbed)
        end = (block + 1) * OWN
        assert torch.equal(o[:, :, :end].view(torch.int16), o_b[:, :, :end].view(torch.int16)), f"block {block}: a row read its own block"
        assert torch.equal(lse[:, :, :end].view(torch.int32), lse_b[:, :, :end].view(torch.int32)), f"block {block}"


@pytest.mark.L0
@requires_pre_rubin_blackwell
@requires_dsl
@torch_fork_set_rng(seed=0)
def test_plans_without_the_opt_in_flag(monkeypatch):
    """End to end with CUDNN_FRONTEND_ENABLE_FROST_ENGINES unset: the heuristics still return the SM100
    MXFP8 plan for the graph (nothing else can serve it) and it computes the same bytes as under the flag."""
    b, h, s = 2, 8, 256
    qf, kf, vf = _inputs(b, h, h, s)
    flagged = _Plan(b, h, h, s, "e4m3")
    inputs, _ = flagged.quantize(qf, kf, vf)
    o_a, lse_a = flagged.execute(inputs)
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    unflagged = _Plan(b, h, h, s, "e4m3")
    o_b, lse_b = unflagged.execute(inputs)
    assert torch.equal(o_a.view(torch.int16), o_b.view(torch.int16))
    assert torch.equal(lse_a.view(torch.int32), lse_b.view(torch.int32))
