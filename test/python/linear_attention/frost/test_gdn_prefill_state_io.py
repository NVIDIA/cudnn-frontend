# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The state I/O build switches of the chunked GDN prefill kernel keep the op's bytes.

``STATE_IO_COALESCED`` and ``TILE_HANDOVER_EARLY`` change how compute group 1 moves the fp32 state between global and
tensor memory and when it reads the next work item's seed; neither may change a word of the output or of a state."""

import pytest
import torch

from cudnn.linear_attention.frost import gdn_engine
from linear_attention.test_la import (
    FWD_TOL,
    STATE_TOL,
    Backend,
    assert_bitwise,
    assert_rms_close,
    clear_caches,
    frost_la_unavailable,
    make_case,
    random_state,
    reference,
    run_fwd,
    state_pool,
)

gdn_prefill_f16 = pytest.importorskip("cudnn.linear_attention.frost.kernel.gdn_prefill_f16")
gdn_warmup_forward_f16 = pytest.importorskip("cudnn.linear_attention.frost.kernel.gdn_warmup_forward_f16")

SWITCHES = {"coalesced": (True, False), "coalesced_early": (True, True)}
SPARE_FILL = 3.14159


@pytest.fixture
def frost(monkeypatch):
    """``(backend, built)``: the FROST backend and the prefill configs the test's calls build.  The host cache starts empty,
    so every static configuration of the test is built inside it."""
    reason = frost_la_unavailable()
    if reason:
        pytest.skip(reason)
    backend = Backend("frost")
    built = []
    real_build_cfg = gdn_prefill_f16.build_cfg

    def build_cfg(*args, **kwargs):
        cfg = real_build_cfg(*args, **kwargs)
        built.append(cfg)
        return cfg

    monkeypatch.setattr(gdn_prefill_f16, "build_cfg", build_cfg)
    monkeypatch.setattr(gdn_warmup_forward_f16, "warmup_forward_cache", {})
    clear_caches()
    yield backend, built
    clear_caches()


def set_switches(monkeypatch, coalesced, early):
    monkeypatch.setattr(gdn_prefill_f16, "STATE_IO_COALESCED", coalesced)
    monkeypatch.setattr(gdn_prefill_f16, "TILE_HANDOVER_EARLY", early)
    clear_caches()


def cap_sm_count(monkeypatch, sm_cap):
    """Plan for ``sm_cap`` thread blocks, so each block runs several work items one behind another."""
    if sm_cap is not None:
        monkeypatch.setattr(gdn_engine, "multiprocessor_count", lambda device: sm_cap)


def pooled_forward(backend, case, packed, **pool_kw):
    """The op on a fresh pool holding ``packed`` at the table's rows: ``(o, pool, indices, spare)``."""
    pool, indices, spare = state_pool(case, packed, fill=SPARE_FILL, **pool_kw)
    o, final_state = run_fwd(backend, case, initial_state=pool, state_indices=indices, output_final_state=True)
    assert final_state is pool
    torch.cuda.synchronize()
    return o, pool, indices, spare


def assert_path_taken(built, coalesced, early):
    assert any(cfg.state_io_coalesced for cfg in built) == coalesced
    assert any(cfg.tile_handover_early for cfg in built) == early


@pytest.mark.L0
@pytest.mark.parametrize("switches", list(SWITCHES))
def test_state_io_switches_keep_pool_bytes(monkeypatch, frost, switches):
    """Ragged sequences on more (sequence, head) tiles than the device has SMs, so thread blocks hand over between work
    items: output and pool are bitwise the same with the switches on and off, and agree with the fp64 reference."""
    backend, built = frost
    coalesced, early = SWITCHES[switches]
    case = make_case("gdn", torch.bfloat16, seq_lens=[257, 64, 320, 129, 200, 71], H=64, lo=0.9)
    packed = random_state(case)
    set_switches(monkeypatch, False, False)
    o_off, pool_off, indices, spare = pooled_forward(backend, case, packed)
    assert_path_taken(built, False, False)
    set_switches(monkeypatch, coalesced, early)
    o_on, pool_on, _, _ = pooled_forward(backend, case, packed)
    assert_path_taken(built, coalesced, early)
    assert_bitwise(f"o[{switches}]", o_on, o_off)
    assert_bitwise(f"pool[{switches}]", pool_on, pool_off)
    assert (pool_on[spare] == SPARE_FILL).all()
    o_ref, fs_ref = reference(case, initial_state=packed)
    assert_rms_close("o", o_on, o_ref, FWD_TOL[case.dtype])
    assert_rms_close("final_state", pool_on[indices.long()], fs_ref, STATE_TOL[case.dtype])


@pytest.mark.L0
def test_shared_pool_row_seeds_the_next_sequence(monkeypatch, frost):
    """One thread block, one head and consecutive sequences that name the same pool row: every sequence is seeded with
    the state its predecessor wrote, so the batch is bitwise the one concatenated sequence -- also with the early
    hand-over, whose read-ahead of the next seed must not be taken when the seed is the row being written."""
    backend, built = frost
    cap_sm_count(monkeypatch, 1)
    num_seqs, tokens = 4, 256
    whole = make_case("gdn", torch.bfloat16, seq_lens=[num_seqs * tokens], H=1, lo=0.9)
    batch = whole.clone(cu=torch.arange(0, num_seqs + 1, dtype=torch.int32, device="cuda") * tokens, N=num_seqs)
    seed = random_state(whole)
    set_switches(monkeypatch, False, False)
    o_whole, fs_whole = run_fwd(backend, whole, initial_state=seed.clone(), output_final_state=True, batch_invariant=True)
    torch.cuda.synchronize()
    row = 1
    indices = torch.full((num_seqs,), row, dtype=torch.int32, device="cuda")

    def shared_row_forward():
        pool = torch.full((3,) + tuple(seed.shape[1:]), SPARE_FILL, device="cuda", dtype=seed.dtype)
        pool[row] = seed[0]
        o, _ = run_fwd(backend, batch, initial_state=pool, state_indices=indices, output_final_state=True, batch_invariant=True)
        torch.cuda.synchronize()
        return o, pool

    o_off, pool_off = shared_row_forward()
    assert_bitwise("o[shared row]", o_off, o_whole)
    assert_bitwise("final_state[shared row]", pool_off[row : row + 1], fs_whole)
    set_switches(monkeypatch, True, True)
    o_on, pool_on = shared_row_forward()
    assert_path_taken(built, True, True)
    assert_bitwise("o[shared row, early hand-over]", o_on, o_whole)
    assert_bitwise("pool[shared row, early hand-over]", pool_on, pool_off)


FORMS = {
    "pool": dict(call="pool"),
    "pool_row_pad_int64": dict(call="pool", pool_kw=dict(padded=True, index_dtype=torch.int64)),
    "pool_k64": dict(call="pool", K=64),
    "pool_one_long_of_nine": dict(call="pool", seq_lens=[17000, 64, 130, 1, 257, 0, 64, 64, 192]),
    "seed_and_final": dict(call="packed", op_kw=dict(output_final_state=True)),
    "seed_and_final_invariant": dict(call="packed", op_kw=dict(output_final_state=True, batch_invariant=True)),
    "seed_overwritten": dict(call="packed", op_kw=dict(output_final_state=True, overwrite_initial_state=True)),
    "seed_only": dict(call="packed", op_kw=dict(output_final_state=False)),
    "final_only": dict(call="unseeded", op_kw=dict(output_final_state=True)),
}


@pytest.mark.L1
@pytest.mark.parametrize("sm_cap", [None, 1, 3], ids=["device_sms", "one_block", "three_blocks"])
@pytest.mark.parametrize("form", list(FORMS))
def test_state_io_switches_keep_bytes_across_call_forms(monkeypatch, frost, form, sm_cap):
    """Every call form that hands a state in or out, on ragged lengths with an empty sequence and a length that is not
    a chunk multiple, planned for the device's SMs and for one and three thread blocks: output and states are bitwise
    the same with the switches off, with the coalesced state I/O, and with the early hand-over on top.  One sequence
    above 16,384 tokens among eight short ones builds a plan whose first kernel may hand the whole batch to the chain
    group; the prefill then launches without a work item."""
    backend, built = frost
    cap_sm_count(monkeypatch, sm_cap)
    spec = FORMS[form]
    case = make_case("gdn", torch.bfloat16, seq_lens=spec.get("seq_lens", [130, 0, 257, 64, 321, 1, 192]), H=4, K=spec.get("K", 128), lo=0.9)
    packed = random_state(case)

    def forward():
        if spec["call"] == "pool":
            o, pool, _, _ = pooled_forward(backend, case, packed, **spec.get("pool_kw", {}))
            return o, pool
        state = packed.clone() if spec["call"] == "packed" else None
        o, final_state = run_fwd(backend, case, initial_state=state, **spec["op_kw"])
        torch.cuda.synchronize()
        return o, (final_state if final_state is not None else state)

    set_switches(monkeypatch, False, False)
    o_off, state_off = forward()
    for coalesced, early in SWITCHES.values():
        set_switches(monkeypatch, coalesced, early)
        o_on, state_on = forward()
        assert_bitwise(f"o[{form}, coalesced={coalesced}, early={early}]", o_on, o_off)
        assert_bitwise(f"state[{form}, coalesced={coalesced}, early={early}]", state_on, state_off)
    assert any(cfg.state_io_coalesced for cfg in built)
