# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public full-sequence to decode state-handoff correctness."""

import pytest
import torch
import torch.nn.functional as F
from fe_api.causal_conv1d_bulk.reference import causal_conv1d_bulk_reference

pytestmark = [
    pytest.mark.L1,
    pytest.mark.gpu_exclusive,
    pytest.mark.xdist_group(name="gpu_exclusive"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def _require_native_route() -> None:
    try:
        from cudnn._causal_conv1d_arch import is_functional_arch
        from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old
    except (ImportError, OSError) as error:
        pytest.skip(f"CuTe DSL dependencies unavailable: {error}")
    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        pytest.skip("causal_conv1d state requires nvidia-cutlass-dsl>=4.7.0")
    capability = torch.cuda.get_device_capability()
    if not is_functional_arch(capability):
        pytest.skip(f"unsupported compute capability {capability}")


def _reference_update(
    x: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = x.shape[0]
    slots = torch.arange(rows, device=x.device, dtype=torch.long) if state_indices is None else state_indices.long()
    valid = slots >= 0
    output = torch.zeros_like(x)
    expected_state = state.clone()
    if valid.any():
        valid_slots = slots[valid]
        selected = state.index_select(0, valid_slots)
        history = torch.cat((selected, x[valid].unsqueeze(-1)), dim=-1)
        window = history[..., -weight.shape[-1] :]
        accumulator = (window.float() * weight.float().unsqueeze(0)).sum(dim=-1)
        if bias is not None:
            accumulator = accumulator + bias.float()
        output[valid] = F.silu(accumulator).to(torch.bfloat16)
        expected_state.index_copy_(0, valid_slots, history[..., -state.shape[-1] :])
    return output, expected_state


def _assert_state_bits_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(
        actual.contiguous().view(torch.int16),
        expected.contiguous().view(torch.int16),
        rtol=0,
        atol=0,
    )


def _state_identity(state: torch.Tensor) -> tuple[int, tuple[int, ...], int]:
    return state.data_ptr(), tuple(state.stride()), state.storage_offset()


@torch.no_grad()
def test_dense_default_and_caller_owned_states_feed_decode_without_repacking() -> None:
    _require_native_route()
    from cudnn.ops import causal_conv1d, causal_conv1d_update

    torch.manual_seed(20260901)
    batch, tokens, channels = 2, 5, 257
    x_btd = torch.randn(batch, tokens, channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(channels, device="cuda", dtype=torch.bfloat16)
    next_x = torch.randn(batch, channels, device="cuda", dtype=torch.bfloat16)

    _, state = causal_conv1d(
        x_btd.transpose(1, 2),
        weight,
        bias,
        "silu",
        return_final_states=True,
    )
    assert state.shape == (batch, channels, 3)
    _assert_state_bits_equal(state, x_btd[:, -3:, :].transpose(1, 2))

    expected_output, expected_state = _reference_update(next_x, state, weight, bias=bias)
    identity = _state_identity(state)
    actual_output = causal_conv1d_update(next_x, state, weight, bias, activation="silu")
    assert _state_identity(state) == identity
    torch.testing.assert_close(actual_output, expected_output, atol=3e-2, rtol=3e-2)
    _assert_state_bits_equal(state, expected_state)

    # A compact caller-owned output for the same shape must also feed the
    # public update directly.
    compact_state = torch.empty(batch, channels, 3, device="cuda", dtype=torch.bfloat16)
    _, returned_state = causal_conv1d(
        x_btd.transpose(1, 2),
        weight,
        bias,
        "silu",
        return_final_states=True,
        final_states_out=compact_state,
    )
    assert returned_state is compact_state
    expected_output, expected_state = _reference_update(next_x, compact_state, weight, bias=bias)
    identity = _state_identity(compact_state)
    actual_output = causal_conv1d_update(next_x, compact_state, weight, bias, activation="silu")
    assert _state_identity(compact_state) == identity
    torch.testing.assert_close(actual_output, expected_output, atol=3e-2, rtol=3e-2)
    _assert_state_bits_equal(compact_state, expected_state)


@torch.no_grad()
def test_packed_default_state_feeds_indexed_decode_without_repacking() -> None:
    _require_native_route()
    from cudnn.ops import causal_conv1d, causal_conv1d_update

    torch.manual_seed(20260902)
    tokens, channels = 8, 257
    x_btd = torch.randn(1, tokens, channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 2, 5, 8], device="cuda", dtype=torch.int32)

    _, state = causal_conv1d(
        x_btd.transpose(1, 2),
        weight,
        activation="silu",
        cu_seqlens=cu_seqlens,
        return_final_states=True,
    )
    assert state.shape == (3, channels, 3)
    _, expected_full_state = causal_conv1d_bulk_reference(x_btd, weight, cu_seqlens=cu_seqlens)
    _assert_state_bits_equal(state, expected_full_state[..., 1:])

    next_x = torch.randn(3, channels, device="cuda", dtype=torch.bfloat16)
    state_indices = torch.tensor([2, -1, 0], device="cuda", dtype=torch.int32)
    expected_output, expected_state = _reference_update(next_x, state, weight, state_indices=state_indices)
    untouched_slot = state[1].clone()
    identity = _state_identity(state)
    actual_output = causal_conv1d_update(
        next_x,
        state,
        weight,
        activation="silu",
        conv_state_indices=state_indices,
    )

    assert _state_identity(state) == identity
    torch.testing.assert_close(actual_output, expected_output, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(actual_output[1], torch.zeros_like(actual_output[1]), rtol=0, atol=0)
    _assert_state_bits_equal(state[1], untouched_slot)
    _assert_state_bits_equal(state, expected_state)
