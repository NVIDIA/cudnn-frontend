# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Head-wave storage boundaries: rectangular states and caller-owned contexts."""

import pytest
import torch

import cudnn
from cudnn.linear_attention.frost import kda_engine
from determinism_utils import assert_bitwise, snapshot
from linear_attention.determinism import pack, sequences
from linear_attention.frost.test_kda_piece_capacity import _prepared
from linear_attention.test_la import frost_la_unavailable

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


@pytest.mark.parametrize("key_dim,value_dim", [(64, 128), (128, 64)])
def test_rectangular_state_waves(monkeypatch, key_dim, value_dim):
    reason = frost_la_unavailable()
    if reason:
        pytest.skip(reason)
    monkeypatch.setattr(kda_engine, "_KDA_STATE_WAVE_BYTES", 0)
    inputs = pack(sequences("kda", [0, 17, 8193, 33], heads=4, key_dim=key_dim, value_dim=value_dim))
    inputs["cu_seqlens"] = inputs.pop("cu")
    inputs["dO"] = inputs.pop("do")
    inputs["d_final_state"] = inputs.pop("dfs")
    inputs["g"].fill_(-0.001)
    handle = cudnn.create_handle()
    cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream().cuda_stream)
    try:
        monkeypatch.setattr(kda_engine, "_checkpoint_head_divisor", lambda *args: 1)
        baseline, outputs, baseline_storage, full_size = _prepared(inputs, "bwd", False, handle)
        baseline()
        expected = snapshot(outputs)
        monkeypatch.setattr(kda_engine, "_checkpoint_head_divisor", lambda *args: 4)
        execute, result, storage, size = _prepared(inputs, "bwd", False, handle)
        assert size < full_size
        for poison in (0, 0xAB, 0xFF):
            storage[:size].fill_(poison)
            execute()
            assert all(torch.isfinite(value).all() for value in result.values())
            assert_bitwise(snapshot(result), expected, "rectangular state wave")
            assert (storage[size:] == 0xA5).all()
        assert (baseline_storage[full_size:] == 0xA5).all()
    finally:
        torch.cuda.synchronize()
        cudnn.destroy_handle(handle)


@pytest.mark.parametrize("lengths", [[8193], [8193, 17]], ids=["matching-context", "different-context-geometry"])
def test_head_waves_preserve_external_context(monkeypatch, lengths):
    reason = frost_la_unavailable()
    if reason:
        pytest.skip(reason)
    from cudnn.linear_attention.ops import kda

    inputs = pack(sequences("kda", lengths, heads=4))
    inputs["g"].fill_(-0.001)
    q, k, v, g, beta = (inputs[name] for name in ("q", "k", "v", "g", "beta"))
    scale = q.shape[-1] ** -0.5
    _, _, context = kda.kda_summary_bwd(
        inputs["do"],
        q,
        k,
        g,
        beta,
        inputs["cu"][:2],
        scale,
        output_transition=True,
        output_context=True,
        batch_invariant=True,
        plan_name="kda_summary_frost",
    )
    assert context.numel() > 0
    context_before = context.clone()
    monkeypatch.setattr(kda_engine, "_KDA_STATE_WAVE_BYTES", 0)

    def run(divisor):
        monkeypatch.setattr(kda_engine, "_checkpoint_head_divisor", lambda *args: divisor)
        kda.bprop_cache.clear()
        result = kda.kda_bwd(
            inputs["do"],
            q,
            k,
            v,
            g,
            beta,
            inputs["cu"],
            scale,
            initial_state=inputs["initial_state"],
            d_final_state=inputs["dfs"],
            context=context,
            batch_invariant=True,
            plan_name="kda_frost",
        )
        assert all(torch.isfinite(value).all() for value in result)
        return snapshot({str(i): value for i, value in enumerate(result)})

    expected = run(1)
    for divisor in (2, 4):
        assert_bitwise(run(divisor), expected, "context-backed head wave")
        assert torch.equal(context, context_before), "caller context was overwritten"


@pytest.mark.parametrize("state_waves", [False, True], ids=["checkpoint-only", "all-states"])
def test_non_invariant_head_waves(monkeypatch, state_waves):
    """The ordinary chain geometry also preserves all gradients across waves."""
    reason = frost_la_unavailable()
    if reason:
        pytest.skip(reason)
    from cudnn.linear_attention.ops import kda

    # Keep the nominal sequence/head grid small enough to select the ordinary
    # chain on SM103; its oversubscribed non-invariant path is intentionally uncut.
    inputs = pack(sequences("kda", [0, 1, 513, 131073], heads=4, key_dim=64, value_dim=64))
    inputs["g"].fill_(-0.001)
    monkeypatch.setattr(kda_engine, "_KDA_STATE_WAVE_BYTES", 0 if state_waves else 1024**3)
    selected = []

    def run(divisor):
        def choose(*args):
            selected.append(divisor)
            return divisor

        monkeypatch.setattr(kda_engine, "_checkpoint_head_divisor", choose)
        kda.bprop_cache.clear()
        result = kda.kda_bwd(
            inputs["do"],
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["beta"],
            inputs["cu"],
            64**-0.5,
            initial_state=inputs["initial_state"],
            d_final_state=inputs["dfs"],
            batch_invariant=False,
            plan_name="kda_frost",
        )
        assert all(torch.isfinite(value).all() for value in result)
        return snapshot({str(i): value for i, value in enumerate(result)})

    expected = run(1)
    for divisor in (2, 4):
        assert_bitwise(run(divisor), expected, "non-invariant chain wave")
    assert selected == [1, 2, 4], "the test must exercise the chain-wave selector"


@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse-transpose"])
def test_default_state_chain_entry(reverse):
    """Other families retain the default full-head standalone compile/replay ABI."""
    reason = frost_la_unavailable()
    if reason:
        pytest.skip(reason)
    from cudnn.linear_attention.frost.common.piece_chain import build_state_chain, run_state_chain

    generator = torch.Generator().manual_seed(20261114)
    batch, pieces, heads, dim = 2, 3, 4, 64
    h_cpu = torch.randn(batch * pieces, heads, dim, dim, generator=generator) * 0.01
    m_cpu = torch.randn(batch * pieces, heads, dim, dim, generator=generator) * 0.001 + torch.eye(dim) * 0.9
    seed_cpu = torch.randn(batch, heads, dim, dim, generator=generator) * 0.01
    expected_x = torch.empty_like(h_cpu, dtype=torch.float64)
    expected_tail = torch.empty_like(seed_cpu, dtype=torch.float64)
    expected_m = torch.empty_like(seed_cpu, dtype=torch.float64)
    for b in range(batch):
        state = seed_cpu[b].double()
        product = torch.eye(dim, dtype=torch.float64).expand(heads, dim, dim)
        for j in (range(pieces - 1, -1, -1) if reverse else range(pieces)):
            slot = b * pieces + j
            transition = m_cpu[slot].double()
            if reverse:
                transition = transition.transpose(-1, -2)
            expected_x[slot] = state
            state = state @ transition + h_cpu[slot].double()
            product = product @ transition
        expected_tail[b] = state
        expected_m[b] = product
    h, m, seed = (value.cuda() for value in (h_cpu, m_cpu, seed_cpu))
    x, tail, summary = torch.empty_like(h), torch.empty_like(seed), torch.empty_like(seed)
    compiled = build_state_chain(
        heads_out=heads,
        dim_v=dim,
        dim_k=dim,
        pieces=pieces,
        reverse=reverse,
        transpose=reverse,
        has_seed=True,
        has_tail=True,
        emit_summary=True,
        device=torch.cuda.current_device(),
        opt_level=3,
    )
    for _ in range(2):
        x.fill_(float("nan"))
        tail.fill_(float("nan"))
        summary.fill_(float("nan"))
        run_state_chain(compiled, batch, h, m, x, seed, tail, summary, torch.cuda.current_stream().cuda_stream)
        for actual, expected in ((x, expected_x), (tail, expected_tail), (summary, expected_m)):
            torch.testing.assert_close(actual.cpu().double(), expected, atol=3e-6, rtol=3e-5)
