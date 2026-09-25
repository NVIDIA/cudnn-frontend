# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage (4) — the block's SDPA — against the FP32 oracle, on Rubin.

The block targets SM107 only for now, so this file carries BOTH halves the
engine contract asks for: the accept tests (they need a Rubin device) and the
REJECT test, which runs anywhere and is the one that fails if the arch gate is
ever quietly widened.
"""

import os
import sys

import pytest
import torch

from cudnn.gated_attention_block import GatedAttentionBlockGeometry
from cudnn.gated_attention_block.api import _Sdpa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_reference import RefGeometry, gated_attention_block_reference, make_inputs  # noqa: E402

pytestmark = pytest.mark.L0

_SM107 = (10, 7)


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


requires_rubin = pytest.mark.skipif(_cc() != _SM107, reason=f"gated_attention_block targets SM107 only; found {_cc()}")


# Qwen3.5's full-attention geometry: d_qk == d_v == 256 picks the (256, 256)
# flavor, which on Rubin is sdpa/fwd/kernels/sm107/prefill_d256_f16.py -- and,
# under fuse_gate, the SAME kernel's epilogue_gate specialization.
def _geoms(**over):
    common = dict(d_model=512, h_q=8, h_kv=2, d_head=256, rope_dim=64)
    common.update(over)
    return GatedAttentionBlockGeometry(**common), RefGeometry(**common)


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


# ---------------------------------------------------------------------------
# The reject half — runs on any device, including none
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_cc() == _SM107, reason="this asserts the decline on NON-Rubin")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device to report a cc")
def test_declines_every_arch_but_rubin():
    geom, _ = _geoms()
    stage = _Sdpa(geom, batch=1, seq_len=256, dtype=torch.bfloat16, device=torch.device("cuda"), want_lse=False)
    with pytest.raises(NotImplementedError, match="Rubin"):
        stage.check_support()


def test_the_bshd_handover_is_zero_copy():
    """Stage (1) writes BSHD-compact; the SDPA takes BHSD. The block hands over
    ``.transpose(1, 2)``, and the adapter's ``_to_bshd`` must recognise that as
    already-canonical and return the SAME storage.

    If this ever fails, every Q handover became a gather copy -- 16 GiB at 1M
    tokens, silent, and the reason § 1 rejects the fused-slab layout."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDsl

    q_bshd = torch.empty(2, 128, 8, 256)  # [B, S, H, D] compact
    handover = q_bshd.transpose(1, 2)  # what execute() passes
    back = SdpaFwdDsl._to_bshd(handover)
    assert back.data_ptr() == q_bshd.data_ptr()
    assert back.shape == q_bshd.shape
    assert back.is_contiguous()


# ---------------------------------------------------------------------------
# The accept half — Rubin only
# ---------------------------------------------------------------------------


@requires_rubin
@pytest.mark.parametrize("seq_len", [256, 512, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_matches_the_fp32_oracle_causal(seq_len, dtype):
    geom, ref_geom = _geoms()
    inp = make_inputs(ref_geom, batch=2, seq_len=seq_len, dtype=dtype)
    ref = gated_attention_block_reference(**inp, geom=ref_geom)

    o = torch.empty_like(ref.o)
    lse = torch.empty(2, geom.h_q, seq_len, device="cuda", dtype=torch.float32)
    stage = _Sdpa(geom, batch=2, seq_len=seq_len, dtype=dtype, device=torch.device("cuda"), want_lse=True)
    stage.check_support()
    stage.compile()
    ws = torch.empty(max(stage.scratch_workspace_bytes(), 1), dtype=torch.uint8, device="cuda")
    stage.execute(ref.q.contiguous(), ref.k.contiguous(), ref.v.contiguous(), o, lse=lse, workspace=ws)

    assert torch.isfinite(o.float()).all()
    assert _cos(o, ref.o) > 0.999, f"O cos {_cos(o, ref.o)}"
    torch.testing.assert_close(lse, ref.lse, rtol=0, atol=2e-2)


@requires_rubin
def test_matches_the_fp32_oracle_non_causal():
    """A dense PASS proves nothing about the masked arm and vice versa: both are
    ``const_expr``-folded, so each needs its own config."""
    geom, ref_geom = _geoms(is_causal=False)
    inp = make_inputs(ref_geom, batch=1, seq_len=512, dtype=torch.bfloat16)
    ref = gated_attention_block_reference(**inp, geom=ref_geom)

    o = torch.empty_like(ref.o)
    stage = _Sdpa(geom, batch=1, seq_len=512, dtype=torch.bfloat16, device=torch.device("cuda"), want_lse=False)
    stage.check_support()
    stage.compile()
    ws = torch.empty(max(stage.scratch_workspace_bytes(), 1), dtype=torch.uint8, device="cuda")
    stage.execute(ref.q.contiguous(), ref.k.contiguous(), ref.v.contiguous(), o, workspace=ws)
    assert _cos(o, ref.o) > 0.999, f"O cos {_cos(o, ref.o)}"


@requires_rubin
def test_padded_kv_gives_exact_zero_and_minus_inf_on_dead_rows():
    """The degenerate case the whole epilogue-substitution rule exists for: a
    batch entry with NO valid KV column must give ``O = 0`` exactly and
    ``LSE = -inf``, not a floored denominator's -69.08 and not residue."""
    geom, ref_geom = _geoms()
    s = 512
    inp = make_inputs(ref_geom, batch=2, seq_len=s, dtype=torch.bfloat16)
    seq_lens = torch.tensor([s, 0], device="cuda", dtype=torch.int32)
    ref = gated_attention_block_reference(**inp, geom=ref_geom, seq_lens=seq_lens)

    o = torch.empty_like(ref.o)
    lse = torch.empty(2, geom.h_q, s, device="cuda", dtype=torch.float32)
    stage = _Sdpa(geom, batch=2, seq_len=s, dtype=torch.bfloat16, device=torch.device("cuda"), want_lse=True, seq_lens_present=True)
    stage.check_support()
    stage.compile()
    ws = torch.empty(max(stage.scratch_workspace_bytes(), 1), dtype=torch.uint8, device="cuda")
    stage.execute(ref.q.contiguous(), ref.k.contiguous(), ref.v.contiguous(), o, lse=lse, seq_lens=seq_lens, workspace=ws)

    assert (o[1] == 0).all(), f"dead batch O is not exactly zero: max|O| = {o[1].abs().max().item()}"
    assert torch.isinf(lse[1]).all() and (lse[1] < 0).all(), f"dead batch LSE is not -inf: {lse[1].flatten()[:4]}"
    assert _cos(o[0], ref.o[0]) > 0.999


# ---------------------------------------------------------------------------
# fuse_gate: the same adapter, the kernel's epilogue_gate specialization
# ---------------------------------------------------------------------------


def test_gate_declines_a_head_dim_the_rows_do_not_claim():
    """Runs on ANY device: the geometry pin is read off the Rubin engine row
    (``epilogue_gate_d_shapes``) and precedes the cc gate, so a d64 gated stage
    declines with the head dim and the knob in the message everywhere."""
    geom, _ = _geoms(d_head=64, rope_dim=32)
    stage = _Sdpa(geom, batch=1, seq_len=256, dtype=torch.bfloat16, device=torch.device("cuda"), want_lse=False, fuse_gate=True)
    with pytest.raises(NotImplementedError, match="256") as ei:
        stage.check_support()
    assert "fuse_gate" in str(ei.value)


@pytest.mark.parametrize(
    "dtype, gate_dtype",
    [
        pytest.param(torch.float8_e4m3fn, None, id="fp8-gate_dtype-omitted"),  # would declare an e4m3 GATE; the FP8 row reads bf16
        pytest.param(torch.bfloat16, torch.float16, id="bf16-q-fp16-gate"),  # the f16 row reads GATE in Q's dtype
    ],
)
def test_gate_declines_a_gate_dtype_the_rows_do_not_claim(dtype, gate_dtype):
    """Runs on ANY device: the gate dtype is read off the Rubin row
    (``epilogue_gate_dtypes``; None = Q's dtype) and precedes the cc gate, so a
    standalone stage that omits / mismatches ``gate_dtype`` declines naming the
    knob -- not as the cc decline off Rubin, not as the adapter's ValueError on it."""
    geom, _ = _geoms()
    stage = _Sdpa(geom, batch=1, seq_len=256, dtype=dtype, device=torch.device("cuda"), want_lse=False, fuse_gate=True, gate_dtype=gate_dtype)
    with pytest.raises(NotImplementedError, match="gate_dtype") as ei:
        stage.check_support()
    assert "Rubin" not in str(ei.value)  # the geometry / dtype pin fired, not the cc gate
    assert "bfloat16" in str(ei.value)  # both rows serve the block's activation dtype for these cases
    # The block's own configuration is served: activation-dtype gate on both rows.
    ok = _Sdpa(geom, batch=1, seq_len=256, dtype=dtype, device=torch.device("cuda"), want_lse=False, fuse_gate=True, gate_dtype=torch.bfloat16)
    ok._check_gate_geometry()


def _run_gated_stage(geom, ref, *, batch, seq_len, gate, gate_token_stride, seq_lens=None, want_lse=True):
    o = torch.full_like(ref.o, 1.5e30)  # sentinel: a cell never written keeps it
    lse = torch.full((batch, geom.h_q, seq_len), float("nan"), device="cuda", dtype=torch.float32) if want_lse else None
    stage = _Sdpa(
        geom,
        batch=batch,
        seq_len=seq_len,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
        want_lse=want_lse,
        seq_lens_present=seq_lens is not None,
        fuse_gate=True,
        gate_token_stride=gate_token_stride,
        gate_dtype=torch.bfloat16,
    )
    stage.check_support()
    stage.compile()
    ws = torch.empty(max(stage.scratch_workspace_bytes(), 1), dtype=torch.uint8, device="cuda")
    stage.execute(ref.q.contiguous(), ref.k.contiguous(), ref.v.contiguous(), o, lse=lse, seq_lens=seq_lens, workspace=ws, gate=gate)
    torch.cuda.synchronize()
    assert not (o == 1.5e30).any(), f"{(o == 1.5e30).sum().item()} O cells were never written"
    return stage, o, lse


@requires_rubin
@pytest.mark.parametrize("seq_len", [512, 1000])  # 1000: causal KV tail + rows past S
def test_gate_stage_matches_the_fp32_oracle(seq_len):
    """``O = softmax(QK^T) V * sigmoid(GATE)`` in ONE launch, against the oracle's
    ``o_gated``; the LSE is untouched by the gate.  Run twice -- GATE as the slab
    column slice the block hands over (token stride N) and as a compact copy --
    and the two must be BIT-IDENTICAL: same kernel, same math, only the gate's
    TMA descriptor differs.  A gate-off run of the same stage must give the
    oracle's PRE-gate ``o`` and the identical LSE."""
    geom, ref_geom = _geoms()
    inp = make_inputs(ref_geom, batch=2, seq_len=seq_len, dtype=torch.bfloat16)
    ref = gated_attention_block_reference(**inp, geom=ref_geom)
    assert ref.gate.stride() == (seq_len * geom.n_qkvg, geom.n_qkvg, geom.d_head, 1), "the oracle's GATE is a slab column slice"

    stage_s, o_s, lse_s = _run_gated_stage(geom, ref, batch=2, seq_len=seq_len, gate=ref.gate, gate_token_stride=geom.n_qkvg)
    assert stage_s._impl.template_params().epilogue_gate is True
    assert torch.isfinite(o_s.float()).all()
    assert _cos(o_s, ref.o_gated) > 0.999, f"gated O cos {_cos(o_s, ref.o_gated)}"
    torch.testing.assert_close(lse_s, ref.lse, rtol=0, atol=2e-2)

    _, o_c, lse_c = _run_gated_stage(geom, ref, batch=2, seq_len=seq_len, gate=ref.gate.contiguous(), gate_token_stride=0)
    assert torch.equal(o_s, o_c) and torch.equal(lse_s, lse_c), "slab-strided vs compact GATE must be bitwise"

    # Gate OFF on the same inputs: pre-gate O, identical LSE (the epilogue is the only difference).
    o_u = torch.empty_like(ref.o)
    lse_u = torch.empty_like(lse_s)
    plain = _Sdpa(geom, batch=2, seq_len=seq_len, dtype=torch.bfloat16, device=torch.device("cuda"), want_lse=True)
    plain.check_support()
    plain.compile()
    ws = torch.empty(max(plain.scratch_workspace_bytes(), 1), dtype=torch.uint8, device="cuda")
    plain.execute(ref.q.contiguous(), ref.k.contiguous(), ref.v.contiguous(), o_u, lse=lse_u, workspace=ws)
    torch.cuda.synchronize()
    assert _cos(o_u, ref.o) > 0.999
    assert torch.equal(lse_u, lse_s), "the gate must not move the LSE by a single bit"


@requires_rubin
def test_gate_stage_dead_padded_entry_is_exactly_zero():
    """``seq_lens = [s, 0]``: the dead batch entry must come out EXACTLY zero and
    ``LSE = -inf`` with the gate ON -- the select happens before the gate fma, per
    element, so neither accumulator residue nor a huge gate can leak.  The dead
    entry's GATE is filled with +-1e4 so a gate-before-select ordering would
    show up as saturation, not as a rounding."""
    geom, ref_geom = _geoms()
    s = 512
    inp = make_inputs(ref_geom, batch=2, seq_len=s, dtype=torch.bfloat16)
    seq_lens = torch.tensor([s, 0], device="cuda", dtype=torch.int32)
    ref = gated_attention_block_reference(**inp, geom=ref_geom, seq_lens=seq_lens)
    gate = ref.gate.contiguous().clone()
    signs = torch.where(torch.rand_like(gate[1], dtype=torch.float32) < 0.5, -1.0, 1.0)
    gate[1] = (signs * 1e4).to(gate.dtype)

    _, o, lse = _run_gated_stage(geom, ref, batch=2, seq_len=s, gate=gate, gate_token_stride=0, seq_lens=seq_lens)
    assert (o[1] == 0).all(), f"dead batch gated O is not exactly zero: max|O| = {o[1].abs().max().item()}"
    assert torch.isinf(lse[1]).all() and (lse[1] < 0).all(), f"dead batch LSE is not -inf: {lse[1].flatten()[:4]}"
    assert torch.isfinite(o[0].float()).all()
    assert _cos(o[0], ref.o_gated[0]) > 0.999
